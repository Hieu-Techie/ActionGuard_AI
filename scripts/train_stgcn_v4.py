"""
train_stgcn_v4.py — Train Dual-Specialist ST-GCN v4-C (Lower-Body) & ST-GCN v4-B (Wrist-Head).

Uses Temporally Cleaned Dataset V4 (`X_train_v4.npy` / `y_train_v4.npy`) and evaluates
strictly on pure held-out `Val` (`830w / 143c`), then reports final metrics on
`ALL_TEST` (`906w / 147c`), including Dual-Specialist Late Fusion (`v4-C + v4-B`) and
stream-level `RED Recall / RED FPR` via `SequenceBuffer + AlertStateMachine`.
"""

from __future__ import annotations

import copy
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.action_classifier import ActionPrediction, ActionSTGCN, load_action_model
from pipeline.alert_state import AlertLevel, AlertStateMachine
from pipeline.sequence_buffer import SequenceBuffer
from utils.config import CLASS_NAMES, NUM_CLASSES, WINDOW_SIZE

WINDOWED_DIR = PROJECT_ROOT / "data" / "skeleton_dataset" / "windowed"
MODELS_DIR = PROJECT_ROOT / "models"

COCO_LR_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
)
EXTREMITY_JOINTS: tuple[int, ...] = (7, 8, 9, 10, 13, 14, 15, 16)
DUAL_SPEC_WEIGHT_C = np.array([0.75, 0.45, 0.45, 0.80, 0.70, 0.75, 0.75, 0.75], dtype=np.float64)


def _seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _augment_batch_ctv(x_b: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Fast batch-level in-the-wild augmentation on (B, 3, T, 17)."""
    out = x_b.clone()
    B, _, T, _ = out.shape
    vis = out[:, 2:3, :, :] > 0.0  # (B, 1, T, 17)

    # 1. Left-right anatomical flip (p=0.5 per sample)
    flip_mask = torch.from_numpy((rng.random(B) < 0.50).astype(bool))
    if bool(flip_mask.any()):
        sub = out[flip_mask]
        sub_vis = vis[flip_mask, 0]
        sub[:, 0] = torch.where(sub_vis, -sub[:, 0], torch.zeros_like(sub[:, 0]))
        for j_l, j_r in COCO_LR_PAIRS:
            tmp = sub[:, :, :, j_l].clone()
            sub[:, :, :, j_l] = sub[:, :, :, j_r]
            sub[:, :, :, j_r] = tmp
        out[flip_mask] = sub
        vis = out[:, 2:3, :, :] > 0.0

    # 2. Anisotropic X/Y scale (aspect variation)
    sx = torch.from_numpy(rng.uniform(0.86, 1.16, size=(B, 1, 1)).astype(np.float32))
    sy = torch.from_numpy(rng.uniform(0.86, 1.16, size=(B, 1, 1)).astype(np.float32))
    vis_2d = vis[:, 0]
    out[:, 0] = torch.where(vis_2d, out[:, 0] * sx, out[:, 0])
    out[:, 1] = torch.where(vis_2d, out[:, 1] * sy, out[:, 1])

    # 3. Extremity occlusion dropout (p=0.20 per sample)
    drop_idx = np.where(rng.random(B) < 0.20)[0]
    for b in drop_idx:
        n_j = int(rng.integers(1, 3))
        js = rng.choice(EXTREMITY_JOINTS, size=n_j, replace=False)
        t0 = int(rng.integers(0, max(1, T - 5)))
        t1 = min(T, t0 + int(rng.integers(3, 8)))
        for j in js:
            out[int(b), :, t0:t1, int(j)] = 0.0

    # 4. Mild Gaussian jitter on visible joints
    noise = torch.randn(B, 2, T, 17, dtype=torch.float32) * 0.012
    out[:, :2] = torch.where(vis, out[:, :2] + noise, out[:, :2])
    return out


def _predict_probs_numpy(model: nn.Module, X_win: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Run eval inference on (N, 16, 17, 3) numpy array -> (N, 8) float64 probabilities."""
    model.eval()
    X_ctv = torch.from_numpy(X_win.transpose(0, 3, 1, 2).astype(np.float32))
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_ctv), batch_size):
            logits = model(X_ctv[i : i + batch_size])
            outs.append(torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int = NUM_CLASSES) -> list[float]:
    f1s: list[float] = []
    for c in range(n_cls):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return f1s


def evaluate_probs(
    probs: np.ndarray,
    y_true: np.ndarray,
    clip_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute window-level, video-level, and safety metrics from probability matrix."""
    y_pred = probs.argmax(axis=1)
    win_acc = float((y_pred == y_true).mean())
    win_f1s = _per_class_f1(y_true, y_pred)
    win_macro_f1 = float(np.mean(win_f1s))

    # Fall class = index 3
    tp_f = int(((y_true == 3) & (y_pred == 3)).sum())
    fp_f = int(((y_true != 3) & (y_pred == 3)).sum())
    fn_f = int(((y_true == 3) & (y_pred != 3)).sum())
    fall_rec = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
    fall_prec = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
    fall_f1 = (2.0 * fall_prec * fall_rec / (fall_prec + fall_rec)) if (fall_prec + fall_rec) > 0 else 0.0
    safety_score = 0.50 * fall_rec + 0.25 * fall_f1 + 0.25 * win_macro_f1

    # Video-level mean-prob aggregation
    clip_groups: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(clip_ids):
        clip_groups[str(cid)].append(i)
    v_true: list[int] = []
    v_pred: list[int] = []
    for cid in sorted(clip_groups):
        idxs = clip_groups[cid]
        v_true.append(int(y_true[idxs[0]]))
        v_pred.append(int(probs[idxs].mean(axis=0).argmax()))
    vt_arr = np.asarray(v_true, dtype=np.int64)
    vp_arr = np.asarray(v_pred, dtype=np.int64)
    vid_acc = float((vp_arr == vt_arr).mean())
    vid_f1s = _per_class_f1(vt_arr, vp_arr)
    vid_macro_f1 = float(np.mean(vid_f1s))

    comp_score = 0.40 * safety_score + 0.35 * vid_macro_f1 + 0.25 * win_macro_f1
    return {
        "win_acc": win_acc,
        "win_macro_f1": win_macro_f1,
        "win_f1s": win_f1s,
        "fall_rec": fall_rec,
        "fall_prec": fall_prec,
        "fall_f1": fall_f1,
        "safety_score": safety_score,
        "vid_acc": vid_acc,
        "vid_macro_f1": vid_macro_f1,
        "vid_f1s": vid_f1s,
        "comp_score": comp_score,
    }


def evaluate_stream_alerts(
    seq_npz_path: Path,
    model_c: nn.Module,
    model_b: nn.Module | None = None,
) -> tuple[int, int, int, int]:
    """Evaluate 10-FPS per-frame stream RED Recall & RED FPR on `<split>_sequences.npz`."""
    data = np.load(str(seq_npz_path))
    model_c.eval()
    if model_b is not None:
        model_b.eval()

    fall_tp = fall_n = nonfall_fp = nonfall_n = 0
    with torch.no_grad():
        for key in sorted(data.files):
            cls_name = key.split("/", 1)[0]
            seq = data[key].astype(np.float32)
            # Collect all 16-frame rolling windows matching SequenceBuffer
            buf = SequenceBuffer(WINDOW_SIZE, max_missing_gap=3)
            sm = AlertStateMachine()
            ready_tensors: list[torch.Tensor] = []
            for t in range(len(seq)):
                fr = seq[t]
                if float(fr[:, 2].max()) <= 0.0:
                    buf.append(None)
                else:
                    buf.append(fr)
                if buf.is_ready() and buf.has_valid_subject():
                    ready_tensors.append(buf.as_stgcn_tensor())

            hit_red = False
            if ready_tensors:
                batch_t = torch.cat(ready_tensors, dim=0)
                probs = torch.softmax(model_c(batch_t), dim=-1).cpu().numpy().astype(np.float64)
                if model_b is not None:
                    probs_b = torch.softmax(model_b(batch_t), dim=-1).cpu().numpy().astype(np.float64)
                    probs = probs * DUAL_SPEC_WEIGHT_C[None, :] + probs_b * (1.0 - DUAL_SPEC_WEIGHT_C[None, :])
                    probs = probs / probs.sum(axis=-1, keepdims=True)
                for p in probs:
                    idx = int(np.argmax(p))
                    lvl = sm.update(
                        ActionPrediction(
                            label=CLASS_NAMES[idx],
                            label_idx=idx,
                            confidence=float(p[idx]),
                            model_type="stgcn",
                        )
                    )
                    if lvl == AlertLevel.RED:
                        hit_red = True

            if cls_name == "Fall":
                fall_n += 1
                if hit_red:
                    fall_tp += 1
            else:
                nonfall_n += 1
                if hit_red:
                    nonfall_fp += 1
    return fall_tp, fall_n, nonfall_fp, nonfall_n


def train_variant(
    tag: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cids_val: np.ndarray,
    class_weights: torch.Tensor,
    *,
    epochs: int,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train one ST-GCN specialist on CPU with live batch aug and select best Validation state."""
    rng = np.random.default_rng(seed)
    X_tr_ctv = torch.from_numpy(X_train.transpose(0, 3, 1, 2).astype(np.float32))
    y_tr_t = torch.from_numpy(y_train.astype(np.int64))
    loader = DataLoader(TensorDataset(X_tr_ctv, y_tr_t), batch_size=batch_size, shuffle=True, drop_last=False)

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict[str, Any] = {}
    best_epoch = 0

    print(f"\n--- Training {tag} ({epochs} epochs, {len(X_train)} train windows) ---", flush=True)
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum = 0.0
        n_samples = 0
        for xb, yb in loader:
            xb_aug = _augment_batch_ctv(xb, rng)
            optimizer.zero_grad()
            logits = model(xb_aug)
            loss = F.cross_entropy(logits, yb, weight=class_weights, label_smoothing=0.06)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(yb)
            n_samples += len(yb)

        scheduler.step()
        val_probs = _predict_probs_numpy(model, X_val)
        vm = evaluate_probs(val_probs, y_val, cids_val)
        dt = time.perf_counter() - t0

        is_best = vm["comp_score"] > best_score
        if is_best:
            best_score = vm["comp_score"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = vm
            best_epoch = ep

        star = " ★ BEST" if is_best else ""
        print(
            f"  ep {ep:2d}/{epochs} ({dt:4.1f}s) | loss={loss_sum/max(n_samples,1):.4f} | "
            f"ValComp={vm['comp_score']:.4f} | Safety={vm['safety_score']:.4f} | "
            f"FallRec={vm['fall_rec']*100:5.1f}% | WinMF1={vm['win_macro_f1']:.4f} | "
            f"VidAcc={vm['vid_acc']*100:5.1f}% VidMF1={vm['vid_macro_f1']:.4f}{star}",
            flush=True,
        )

    best_metrics["best_epoch"] = best_epoch
    return best_state, best_metrics


def _print_comparison_row(name: str, m: dict[str, Any], stream_tup: tuple[int, int, int, int]) -> None:
    ftp, fn, rfp, rn = stream_tup
    frec_s = f"{ftp:2d}/{fn:2d} ({100*ftp/max(fn,1):5.1f}%)"
    rfpr_s = f"{rfp:2d}/{rn:3d} ({100*rfp/max(rn,1):4.2f}%)"
    print(
        f"  {name:24s} | VidAcc={m['vid_acc']*100:5.2f}% VidMF1={m['vid_macro_f1']:.4f} | "
        f"WinAcc={m['win_acc']*100:5.2f}% WinMF1={m['win_macro_f1']:.4f} | "
        f"Safety={m['safety_score']:.4f} | RED_Rec={frec_s} RED_FPR={rfpr_s}",
        flush=True,
    )


def main() -> None:
    _seed_everything(42)
    X_train = np.load(str(WINDOWED_DIR / "X_train_v4.npy"))
    y_train = np.load(str(WINDOWED_DIR / "y_train_v4.npy"))
    X_val = np.load(str(WINDOWED_DIR / "X_val.npy"))
    y_val = np.load(str(WINDOWED_DIR / "y_val.npy"))
    cids_val = np.load(str(WINDOWED_DIR / "clip_ids_val.npy"), allow_pickle=True)
    X_test = np.load(str(WINDOWED_DIR / "X_test.npy"))
    y_test = np.load(str(WINDOWED_DIR / "y_test.npy"))
    cids_test = np.load(str(WINDOWED_DIR / "clip_ids_test.npy"), allow_pickle=True)

    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    raw_w = (counts.sum() / (float(NUM_CLASSES) * np.maximum(counts, 1.0))) ** 0.5
    raw_w[4] *= 1.15  # Gentle emphasis on Lying after temporal cleanup
    raw_w = raw_w / raw_w.mean()
    class_weights = torch.tensor(raw_w, dtype=torch.float32)
    print("V4 class counts :", {CLASS_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)})
    print("V4 class weights:", {CLASS_NAMES[i]: round(float(raw_w[i]), 3) for i in range(NUM_CLASSES)}, flush=True)

    # Reference baseline v3.1-C
    m_v31 = load_action_model(str(MODELS_DIR / "action_stgcn_v3_1.pth"), "stgcn", device=torch.device("cpu"))
    val_v31 = evaluate_probs(_predict_probs_numpy(m_v31, X_val), y_val, cids_val)
    test_v31 = evaluate_probs(_predict_probs_numpy(m_v31, X_test), y_test, cids_test)
    val_s_v31 = evaluate_stream_alerts(WINDOWED_DIR / "val_sequences.npz", m_v31)
    test_s_v31 = evaluate_stream_alerts(WINDOWED_DIR / "test_sequences.npz", m_v31)

    # 1. Train ST-GCN v4-C (+ Lower-Body Kinematics, warm-start from v3.1-C)
    m_v4c = load_action_model(str(MODELS_DIR / "action_stgcn_v3_1.pth"), "stgcn", device=torch.device("cpu"))
    opt_c = torch.optim.AdamW(m_v4c.parameters(), lr=4.5e-4, weight_decay=2.5e-4)
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=18, eta_min=1e-5)
    best_c_state, val_v4c = train_variant(
        "ST-GCN v4-C (Lower-Body)", m_v4c, opt_c, sched_c,
        X_train, y_train, X_val, y_val, cids_val, class_weights,
        epochs=18, batch_size=64, seed=42,
    )
    m_v4c.load_state_dict(best_c_state)
    torch.save(
        {"state_dict": best_c_state, "variant": "ST-GCN v4-C (+ Lower-Body Kinematics)", "val_metrics": val_v4c},
        str(MODELS_DIR / "action_stgcn_v4_c.pth"),
    )

    # 2. Train ST-GCN v4-B (+ Wrist-to-Head Features, transfer backbone weights from v3.0-clean)
    m_v4b = ActionSTGCN(use_wrist_head_feats=True)
    v30_raw = torch.load(str(MODELS_DIR / "action_stgcn_v3_0_clean.pth"), map_location="cpu", weights_only=True)
    v30_sd = v30_raw["state_dict"] if isinstance(v30_raw, dict) and "state_dict" in v30_raw else v30_raw
    v4b_sd = m_v4b.state_dict()
    loaded_shared = 0
    for k, v in v30_sd.items():
        if k in v4b_sd and v.shape == v4b_sd[k].shape:
            v4b_sd[k] = v.clone()
            loaded_shared += 1
    m_v4b.load_state_dict(v4b_sd)
    print(f"\nInitialized v4-B with {loaded_shared}/{len(v4b_sd)} compatible tensors from v3.0-clean.")

    new_params = []
    shared_params = []
    for name, p in m_v4b.named_parameters():
        if name.startswith(("b1.", "data_bn.", "wrist_mlp.", "head.")):
            new_params.append(p)
        else:
            shared_params.append(p)
    opt_b = torch.optim.AdamW(
        [{"params": new_params, "lr": 9.0e-4}, {"params": shared_params, "lr": 3.5e-4}],
        weight_decay=2.5e-4,
    )
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=22, eta_min=1e-5)
    best_b_state, val_v4b = train_variant(
        "ST-GCN v4-B (Wrist-Head)", m_v4b, opt_b, sched_b,
        X_train, y_train, X_val, y_val, cids_val, class_weights,
        epochs=22, batch_size=64, seed=142,
    )
    m_v4b.load_state_dict(best_b_state)
    torch.save(
        {"state_dict": best_b_state, "variant": "ST-GCN v4-B (+ Wrist-to-Head Features)", "val_metrics": val_v4b},
        str(MODELS_DIR / "action_stgcn_v4_b.pth"),
    )

    # Promote primary action_stgcn.pth to best validated v4-C
    if val_v4c["comp_score"] >= val_v31["comp_score"] or val_v4c["vid_macro_f1"] > val_v31["vid_macro_f1"]:
        torch.save(
            {"state_dict": best_c_state, "variant": "ST-GCN v4-C (+ Lower-Body Kinematics)", "val_metrics": val_v4c},
            str(MODELS_DIR / "action_stgcn.pth"),
        )
        print("\nPromoted ST-GCN v4-C -> models/action_stgcn.pth")
    else:
        print("\nWARNING: v4-C did not beat v3.1-C on Val comp_score; keeping action_stgcn.pth unchanged for review.")

    # Compute Dual-Specialist fusion probabilities on Val and Test
    p_val_c = _predict_probs_numpy(m_v4c, X_val)
    p_val_b = _predict_probs_numpy(m_v4b, X_val)
    p_val_dual = p_val_c * DUAL_SPEC_WEIGHT_C[None, :] + p_val_b * (1.0 - DUAL_SPEC_WEIGHT_C[None, :])
    p_val_dual = p_val_dual / p_val_dual.sum(axis=-1, keepdims=True)
    val_dual = evaluate_probs(p_val_dual, y_val, cids_val)

    p_test_c = _predict_probs_numpy(m_v4c, X_test)
    p_test_b = _predict_probs_numpy(m_v4b, X_test)
    p_test_dual = p_test_c * DUAL_SPEC_WEIGHT_C[None, :] + p_test_b * (1.0 - DUAL_SPEC_WEIGHT_C[None, :])
    p_test_dual = p_test_dual / p_test_dual.sum(axis=-1, keepdims=True)
    test_v4c = evaluate_probs(p_test_c, y_test, cids_test)
    test_v4b = evaluate_probs(p_test_b, y_test, cids_test)
    test_dual = evaluate_probs(p_test_dual, y_test, cids_test)

    val_s_c = evaluate_stream_alerts(WINDOWED_DIR / "val_sequences.npz", m_v4c)
    val_s_b = evaluate_stream_alerts(WINDOWED_DIR / "val_sequences.npz", m_v4b)
    val_s_dual = evaluate_stream_alerts(WINDOWED_DIR / "val_sequences.npz", m_v4c, m_v4b)

    test_s_c = evaluate_stream_alerts(WINDOWED_DIR / "test_sequences.npz", m_v4c)
    test_s_b = evaluate_stream_alerts(WINDOWED_DIR / "test_sequences.npz", m_v4b)
    test_s_dual = evaluate_stream_alerts(WINDOWED_DIR / "test_sequences.npz", m_v4c, m_v4b)

    print("\n" + "=" * 112)
    print("VALIDATION SUMMARY (830 windows / 143 clips — pure, untouched)")
    print("=" * 112)
    _print_comparison_row("ST-GCN v3.1-C (base)", val_v31, val_s_v31)
    _print_comparison_row("ST-GCN v4-C (Lower)", val_v4c, val_s_c)
    _print_comparison_row("ST-GCN v4-B (Wrist)", val_v4b, val_s_b)
    _print_comparison_row("Dual-Specialist v4", val_dual, val_s_dual)

    print("\n" + "=" * 112)
    print("ALL_TEST SUMMARY (906 windows / 147 clips — pure, untouched)")
    print("=" * 112)
    _print_comparison_row("ST-GCN v3.1-C (base)", test_v31, test_s_v31)
    _print_comparison_row("ST-GCN v4-C (Lower)", test_v4c, test_s_c)
    _print_comparison_row("ST-GCN v4-B (Wrist)", test_v4b, test_s_b)
    _print_comparison_row("Dual-Specialist v4", test_dual, test_s_dual)

    print("\nPer-Class Video-Level F1 on ALL_TEST (147 clips):")
    for i, cname in enumerate(CLASS_NAMES):
        print(
            f"  {cname:10s}: v3.1-C={test_v31['vid_f1s'][i]:.4f} | "
            f"v4-C={test_v4c['vid_f1s'][i]:.4f} | "
            f"v4-B={test_v4b['vid_f1s'][i]:.4f} | "
            f"Dual-v4={test_dual['vid_f1s'][i]:.4f}"
        )


if __name__ == "__main__":
    main()
