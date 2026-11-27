"""
build_dataset_v4.py — Build Temporally Cleaned + In-the-Wild Augmented Dataset V4.

Uses the verified `data/skeleton_dataset/fall_timestamps.csv` annotations while
preserving the exact leak-free clip split from `split_manifest.json` so `Val`,
`Test`, and `Sealed` remain 100% untouched for fair, apples-to-apples comparison.

Train cleanups & augmentations applied:
  1. `Fall` clips: keep only windows with meaningful overlap with `[fall_start, fall_end]`;
     harvest clean post-fall motionless horizontal windows into `Lying`.
  2. `Lying` clips: drop early upright/seated windows so sitting is never taught as `Lying`.
  3. Minority transition classes (`Sit_Down`, `Stand_Up`, `Crouching`) use `stride = 2`.
  4. In-the-wild skeleton augmentation (`augment_window_v4`): anatomical left-right flip,
     anisotropic X/Y viewpoint aspect scaling, extremity joint occlusion dropout, rotation,
     jitter, and speed warp.
  5. Exports `data/skeleton_dataset_v4_kaggle.zip`.
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import CLASS_NAMES, NUM_COORDS, NUM_JOINTS, WINDOW_SIZE
from utils.skeleton_utils import augment_window, sliding_windows

SKEL_ROOT = PROJECT_ROOT / "data" / "skeleton_dataset"
PER_FRAME_DIR = SKEL_ROOT / "per_frame"
WINDOWED_DIR = SKEL_ROOT / "windowed"
MANIFEST_PATH = SKEL_ROOT / "split_manifest.json"
FALL_TS_CSV = SKEL_ROOT / "fall_timestamps.csv"
KAGGLE_V4_ZIP = PROJECT_ROOT / "data" / "skeleton_dataset_v4_kaggle.zip"

LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
COCO_LR_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
)
EXTREMITY_JOINTS: tuple[int, ...] = (7, 8, 9, 10, 13, 14, 15, 16)
_CORE_JOINTS: list[int] = [0, 5, 6, 11, 12, 13, 14, 15, 16]


def _window_posture_stats(win: np.ndarray) -> dict[str, float]:
    """Compute core motion and late-window uprightness descriptors for a (16, 17, 3) window."""
    vis = (win[:, :, 2] > 0).astype(np.float32)
    vp = vis[1:] * vis[:-1]
    diffs = np.linalg.norm(win[1:, _CORE_JOINTS, :2] - win[:-1, _CORE_JOINTS, :2], axis=-1) * vp[:, _CORE_JOINTS]
    core_e = float(diffs.sum() / (vp[:, _CORE_JOINTS].sum() + 1e-6))

    sh_x = 0.5 * (win[:, 5, 0] + win[:, 6, 0])
    sh_y = 0.5 * (win[:, 5, 1] + win[:, 6, 1])
    sh_y_mean = float(np.mean(sh_y))
    sin_theta = float(np.mean(np.abs(sh_x) / (np.sqrt(sh_x ** 2 + sh_y ** 2) + 1e-6)))

    late = win[-6:]
    l_sh_x = 0.5 * (late[:, 5, 0] + late[:, 6, 0])
    l_sh_y = 0.5 * (late[:, 5, 1] + late[:, 6, 1])
    late_sh_y = float(np.mean(l_sh_y))
    late_sin = float(np.mean(np.abs(l_sh_x) / (np.sqrt(l_sh_x ** 2 + l_sh_y ** 2) + 1e-6)))
    v_late = late[:, :, 2] >= 0.30
    if bool(np.any(v_late)):
        ys = late[:, :, 1][v_late]
        xs = late[:, :, 0][v_late]
        late_yx = float((np.max(ys) - np.min(ys)) / (np.max(xs) - np.min(xs) + 1e-4))
    else:
        late_yx = 1.0

    return {
        "core_e": core_e,
        "sh_y_mean": sh_y_mean,
        "sin_theta": sin_theta,
        "late_sh_y": late_sh_y,
        "late_sin": late_sin,
        "late_yx": late_yx,
    }


def augment_window_v4(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """In-the-wild skeleton augmentation: L/R flip, X/Y aspect scale, limb dropout, rot/jitter/warp."""
    out = window.copy()
    vis = out[:, :, 2] > 0.0

    # 1. Anatomical Left-Right mirror flip (50% chance)
    if rng.random() < 0.50:
        out[:, :, 0] = np.where(vis, -out[:, :, 0], 0.0)
        for j_l, j_r in COCO_LR_PAIRS:
            tmp = out[:, j_l, :].copy()
            out[:, j_l, :] = out[:, j_r, :]
            out[:, j_r, :] = tmp
        vis = out[:, :, 2] > 0.0

    # 2. Anisotropic X/Y viewpoint aspect ratio scaling
    sx = float(rng.uniform(0.82, 1.22))
    sy = float(rng.uniform(0.85, 1.18))
    out[:, :, 0] = np.where(vis, out[:, :, 0] * sx, 0.0)
    out[:, :, 1] = np.where(vis, out[:, :, 1] * sy, 0.0)

    # 3. Contiguous extremity joint dropout (25% chance)
    if rng.random() < 0.25:
        n_drop = int(rng.integers(1, 3))
        chosen = rng.choice(EXTREMITY_JOINTS, size=n_drop, replace=False)
        t0 = int(rng.integers(0, max(1, WINDOW_SIZE - 5)))
        t1 = min(WINDOW_SIZE, t0 + int(rng.integers(4, 10)))
        for j in chosen:
            out[t0:t1, int(j), :] = 0.0

    # 4. Standard spatial rotation, jitter, and temporal speed warp
    return augment_window(
        out,
        max_angle_deg=12.0,
        jitter_sigma=0.016,
        speed_range=(0.84, 1.18),
        rng=rng,
    )


def build_v4_dataset() -> None:
    rng = np.random.default_rng(42)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ts_df = pd.read_csv(FALL_TS_CSV).set_index("clip_id")

    # Backup original V3b X_train/y_train once before writing V4
    for fname, bak_name in (
        ("X_train.npy", "X_train_v3b.npy"),
        ("y_train.npy", "y_train_v3b.npy"),
        ("clip_ids_train.npy", "clip_ids_train_v3b.npy"),
    ):
        src = WINDOWED_DIR / fname
        bak = WINDOWED_DIR / bak_name
        if src.is_file() and not bak.is_file():
            shutil.copy2(src, bak)

    base_X: dict[str, list[np.ndarray]] = {c: [] for c in CLASS_NAMES}
    base_cids: dict[str, list[str]] = {c: [] for c in CLASS_NAMES}
    stats_log = {
        "fall_dropped_prefall": 0,
        "fall_dropped_postfall": 0,
        "fall_harvested_lying": 0,
        "lying_dropped_upright": 0,
    }

    for cls in CLASS_NAMES:
        train_files = manifest["splits_by_class"][cls]["train_clips"]
        for fname in train_files:
            npy_path = PER_FRAME_DIR / cls / fname
            if not npy_path.is_file():
                continue
            seq = np.load(str(npy_path)).astype(np.float32)
            stem = npy_path.stem
            cid = f"{cls}/{stem}"

            if cls == "Fall":
                if cid in ts_df.index:
                    row = ts_df.loc[cid]
                    f_start = int(row["fall_start_tick"])
                    f_end = int(row["fall_end_tick"])
                else:
                    f_start = 0
                    f_end = len(seq) - 1
                active_len = max(1, f_end - f_start + 1)
                req_ov = min(5, max(2, (active_len + 1) // 2))

                wins = sliding_windows(seq, window_size=WINDOW_SIZE, stride=2)
                kept_fall = 0
                harvested_lying = 0
                for w_i, win in enumerate(wins):
                    s = w_i * 2
                    e = s + WINDOW_SIZE - 1
                    ov = max(0, min(e, f_end) - max(s, f_start) + 1)
                    st = _window_posture_stats(win)
                    is_static_upright = (
                        st["late_sh_y"] > 0.72 and st["late_sin"] < 0.32 and st["core_e"] < 0.025
                    )
                    if ov >= req_ov and not is_static_upright:
                        base_X["Fall"].append(win)
                        base_cids["Fall"].append(cid)
                        kept_fall += 1
                    elif s > f_end - 1 and e < len(seq):
                        stats_log["fall_dropped_postfall"] += 1
                        if (
                            harvested_lying < 2
                            and st["core_e"] < 0.050
                            and (st["sh_y_mean"] < 0.55 or st["sin_theta"] > 0.45)
                        ):
                            base_X["Lying"].append(win)
                            base_cids["Lying"].append(f"Lying_from_Fall/{stem}")
                            harvested_lying += 1
                            stats_log["fall_harvested_lying"] += 1
                    else:
                        stats_log["fall_dropped_prefall"] += 1

                if kept_fall == 0 and len(wins) > 0:
                    # Safety fallback: keep the window centred closest to f_end
                    best_i = min(len(wins) - 1, max(0, (f_start + f_end) // 4))
                    base_X["Fall"].append(wins[best_i])
                    base_cids["Fall"].append(cid)

            elif cls == "Lying":
                wins = sliding_windows(seq, window_size=WINDOW_SIZE, stride=3)
                kept = 0
                for win in wins:
                    st = _window_posture_stats(win)
                    if st["late_sh_y"] > 0.68 and st["late_sin"] < 0.38 and st["late_yx"] > 1.20:
                        stats_log["lying_dropped_upright"] += 1
                        continue
                    base_X["Lying"].append(win)
                    base_cids["Lying"].append(cid)
                    kept += 1
                if kept == 0 and len(wins) > 0:
                    base_X["Lying"].append(wins[-1])
                    base_cids["Lying"].append(cid)

            else:
                stride = 2 if cls in ("Sit_Down", "Stand_Up", "Crouching") else 4
                wins = sliding_windows(seq, window_size=WINDOW_SIZE, stride=stride)
                for win in wins:
                    base_X[cls].append(win)
                    base_cids[cls].append(cid)

    # Assemble balanced Train V4 with in-the-wild augmentation
    TARGET_PER_CLASS = 520
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    cid_list: list[str] = []

    print("V4 Train Class Composition (Temporal Clean + Wild Aug):")
    for cls_idx, cls in enumerate(CLASS_NAMES):
        arr = np.stack(base_X[cls], axis=0).astype(np.float32)
        cids = base_cids[cls]
        n_base = len(arr)

        # Ensure every class gets at least 150 diverse wild-augmented views, up to TARGET_PER_CLASS
        n_aug = max(150, TARGET_PER_CLASS - n_base) if n_base < TARGET_PER_CLASS else max(120, n_base // 3)
        aug_arr = np.empty((n_aug, WINDOW_SIZE, NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        aug_cids: list[str] = []
        src_indices = rng.integers(0, n_base, size=n_aug)
        for k, s_i in enumerate(src_indices):
            aug_arr[k] = augment_window_v4(arr[int(s_i)], rng)
            aug_cids.append(cids[int(s_i)])

        cls_X = np.concatenate([arr, aug_arr], axis=0)
        cls_y = np.full((len(cls_X),), cls_idx, dtype=np.int64)
        X_list.append(cls_X)
        y_list.append(cls_y)
        cid_list.extend(cids)
        cid_list.extend(aug_cids)
        print(f"  {cls:10s}: base={n_base:4d} + aug={n_aug:4d} -> total={len(cls_X):4d}")

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    cids_train = np.array(cid_list, dtype="<U64")

    perm = rng.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    cids_train = cids_train[perm]

    np.save(str(WINDOWED_DIR / "X_train_v4.npy"), X_train)
    np.save(str(WINDOWED_DIR / "y_train_v4.npy"), y_train)
    np.save(str(WINDOWED_DIR / "clip_ids_train_v4.npy"), cids_train)
    np.save(str(WINDOWED_DIR / "X_train.npy"), X_train)
    np.save(str(WINDOWED_DIR / "y_train.npy"), y_train)
    np.save(str(WINDOWED_DIR / "clip_ids_train.npy"), cids_train)

    print("\nCleanup summary:", stats_log)
    print(f"Saved Train V4: X_train={X_train.shape}, y_train={y_train.shape}")

    # Package Kaggle V4 zip (Val/Test pure + Train V4 + manifests; zero Sealed)
    zip_items = [
        ("windowed/X_train.npy", WINDOWED_DIR / "X_train_v4.npy"),
        ("windowed/y_train.npy", WINDOWED_DIR / "y_train_v4.npy"),
        ("windowed/clip_ids_train.npy", WINDOWED_DIR / "clip_ids_train_v4.npy"),
        ("windowed/X_val.npy", WINDOWED_DIR / "X_val.npy"),
        ("windowed/y_val.npy", WINDOWED_DIR / "y_val.npy"),
        ("windowed/clip_ids_val.npy", WINDOWED_DIR / "clip_ids_val.npy"),
        ("windowed/X_test.npy", WINDOWED_DIR / "X_test.npy"),
        ("windowed/y_test.npy", WINDOWED_DIR / "y_test.npy"),
        ("windowed/clip_ids_test.npy", WINDOWED_DIR / "clip_ids_test.npy"),
        ("windowed/val_sequences.npz", WINDOWED_DIR / "val_sequences.npz"),
        ("windowed/test_sequences.npz", WINDOWED_DIR / "test_sequences.npz"),
        ("fall_timestamps.csv", FALL_TS_CSV),
        ("schema.json", SKEL_ROOT / "schema.json"),
        ("split_manifest.json", MANIFEST_PATH),
    ]
    with zipfile.ZipFile(str(KAGGLE_V4_ZIP), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, fpath in zip_items:
            if fpath.is_file():
                zf.write(str(fpath), arcname=arcname)
    mb = KAGGLE_V4_ZIP.stat().st_size / (1024.0 * 1024.0)
    print(f"Packaged Kaggle V4 archive: {KAGGLE_V4_ZIP} ({mb:.2f} MB)")


if __name__ == "__main__":
    build_v4_dataset()
