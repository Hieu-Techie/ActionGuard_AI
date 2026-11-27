"""
run_extraction.py — Standalone extraction and leak-free dataset splitting script (Phase 1.2 & 3.1).

Runs the full pipeline:
  1. Load YOLO-Pose (yolov8n-pose.pt, auto-downloaded if absent)
  2. For each raw video clip across all 8 classes:
       a. Extract per-frame normalised keypoints  → (T, 17, 3)
       b. Save to data/skeleton_dataset/per_frame/<class>/<clip>.npy
  3. Split valid clips at the VIDEO (CLIP) LEVEL (70% Train / 15% Val / 15% Test,
     stratified per class, fixed SEED=42) to prevent any temporal window overlap
     or augmentation leakage across splits.
  4. Slice each split's clips into sliding windows (WINDOW_SIZE=16, stride=4):
       a. Apply skeleton coordinate augmentation ONLY to the Train split.
       b. Keep Val and Test splits 100% pure (un-augmented, unseen clips).
       c. Save X_train/y_train, X_val/y_val, X_test/y_test (and combined X/y)
          to data/skeleton_dataset/windowed/.
  5. Write split_manifest.json, schema.json, and class_distribution.txt.

Run from the project root:
    python scripts/run_extraction.py
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

# ── Project root setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import (
    WINDOW_SIZE,
    CLASS_NAMES,
    NUM_JOINTS,
    NUM_COORDS,
    JOINT_NAMES,
    YOLO_POSE_MODEL_PATH,
    TARGET_FPS,
)
from utils.skeleton_utils import (
    load_pose_model,
    clip_to_keypoint_sequence,
    sliding_windows,
    augment_window,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
RAW_VIDEO_DIR    = PROJECT_ROOT / "data" / "raw_videos"
RAW_VIDEO_V2_DIR = PROJECT_ROOT / "data" / "raw_videos_v2"
PER_FRAME_DIR    = PROJECT_ROOT / "data" / "skeleton_dataset" / "per_frame"
WINDOWED_DIR     = PROJECT_ROOT / "data" / "skeleton_dataset" / "windowed"
SCHEMA_PATH      = PROJECT_ROOT / "data" / "skeleton_dataset" / "schema.json"
MANIFEST_PATH    = PROJECT_ROOT / "data" / "skeleton_dataset" / "split_manifest.json"
REPORT_PATH      = PROJECT_ROOT / "data" / "skeleton_dataset" / "class_distribution.txt"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SPLIT_SEED       = 42
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO       = 0.15

# Label index: match config.CLASS_NAMES order exactly
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Per-frame extraction (v1 + v2 incremental support)
# ════════════════════════════════════════════════════════════════════════════

def run_per_frame_extraction(model, force: bool = False) -> dict[str, list[Path]]:
    """
    Extract and save per-frame keypoints for all clips in `data/raw_videos` (v1)
    and `data/raw_videos_v2` (v2).

    When `force=False` and v1 per-frame `.npy` files already exist, reuses all
    existing v1 `.npy` files directly (without re-running YOLO-Pose on the 82
    previously-skipped unusable v1 clips) and only extracts new v2 clips that
    do not yet have a corresponding `.npy` file.

    Returns
    -------
    dict mapping class_name → sorted list of saved .npy paths
    """
    PER_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    saved: dict[str, list[Path]] = {c: [] for c in CLASS_NAMES}
    skipped_total = 0

    for cls in CLASS_NAMES:
        cls_out_dir = PER_FRAME_DIR / cls
        cls_out_dir.mkdir(exist_ok=True)

        existing_v1 = sorted(
            p for p in cls_out_dir.glob("*.npy")
            if not p.name.startswith("v2_")
        )

        # Build list of raw video files to check/extract
        video_files: list[Path] = []
        if not force and len(existing_v1) > 0:
            # Reuse already-extracted v1 .npy files without re-probing skipped v1 videos
            saved[cls].extend(existing_v1)
        else:
            cls_v1_dir = RAW_VIDEO_DIR / cls
            if cls_v1_dir.exists():
                video_files.extend(
                    sorted(p for p in cls_v1_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
                )

        cls_v2_dir = RAW_VIDEO_V2_DIR / cls
        if cls_v2_dir.exists():
            video_files.extend(
                sorted(p for p in cls_v2_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
            )

        log.info(
            "Class %-12s — %d existing v1 .npy reused, %d candidate videos to check",
            cls, len(saved[cls]), len(video_files),
        )

        for vp in video_files:
            out_path = cls_out_dir / (vp.stem + ".npy")

            if out_path.exists() and not force:
                saved[cls].append(out_path)
                continue

            if model is None:
                model = load_pose_model(str(PROJECT_ROOT / YOLO_POSE_MODEL_PATH))

            t0 = time.time()
            seq = clip_to_keypoint_sequence(str(vp), model, sample_fps=TARGET_FPS)
            elapsed = time.time() - t0

            if seq is None:
                log.warning("  SKIP %-40s (no usable detections)", vp.name)
                skipped_total += 1
                continue

            np.save(str(out_path), seq)
            saved[cls].append(out_path)
            log.info(
                "  OK   %-40s  T=%3d frames  %.1fs",
                vp.name, seq.shape[0], elapsed,
            )

        saved[cls] = sorted(saved[cls])

    log.info("Per-frame extraction done. Newly skipped (unusable): %d", skipped_total)
    return saved


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Video-Level Stratified Split + Windowing + Train-Only Augmentation
# ════════════════════════════════════════════════════════════════════════════

def _split_group_list(files: list[Path], rng: np.random.Generator) -> tuple[list[Path], list[Path], list[Path]]:
    """Split a sorted list of clip paths into (train, val, test) at 70/15/15."""
    n = len(files)
    if n == 0:
        return [], [], []
    perm = rng.permutation(n)
    n_val   = max(1, int(round(n * VAL_RATIO))) if n >= 3 else 0
    n_test  = max(1, int(round(n * TEST_RATIO))) if n >= 2 else 0
    n_train = max(0, n - n_val - n_test)
    tr = sorted([files[i] for i in perm[:n_train]])
    va = sorted([files[i] for i in perm[n_train:n_train + n_val]])
    te = sorted([files[i] for i in perm[n_train + n_val:]])
    return tr, va, te


V2_RENAME_MANIFEST = RAW_VIDEO_V2_DIR / "rename_manifest.csv"
V2_GROUP_SPLIT_PATH = PROJECT_ROOT / "data" / "skeleton_dataset" / "v2_group_split.json"
SEALED_DIR = PROJECT_ROOT / "data" / "skeleton_dataset" / "sealed"
# Fractions applied to the v2 clips ONLY (v1 keeps its original Phase 3.1 assignment).
V2_SPLIT_FRACTIONS = {"train": 0.60, "val": 0.14, "test": 0.13, "sealed": 0.13}
V2_SPLIT_ORDER = ["train", "val", "test", "sealed"]


def _load_v2_original_names() -> dict:
    """Map (class, new_stem) -> original file name using data/raw_videos_v2/rename_manifest.csv."""
    names: dict = {}
    if not V2_RENAME_MANIFEST.exists():
        return names
    with open(str(V2_RENAME_MANIFEST), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[(row["class"], Path(row["new_name"]).stem)] = row["old_name"]
    return names


def _v2_group_key(cls: str, stem: str, names: dict) -> str:
    """
    Group key = the unit that must never be split across Train/Val/Test/Sealed.
      - NTU-style AVI names `S<setup>C<cam>P<performer>R<rep>`  -> `avi:P<performer>`
        (same performer, any class / setup / repetition stays together)
      - MP4 names ending in `(<room> - S<n>)`                  -> `mp4:S<n>`
        (the S<n> tag recurs across different rooms, so it is treated as one subject)
      - anything unparsable                                     -> its own singleton group
    """
    old = names.get((cls, stem), "")
    m = re.match(r"S\d+C\d+P(\d+)R\d+", old)
    if m:
        return f"avi:P{m.group(1)}"
    m = re.search(r"\(.*?-\s*(S\d+)\)", old)
    if m:
        return f"mp4:{m.group(1)}"
    return f"solo:{cls}/{stem}"


def _assign_v2_groups(v2_by_cls: dict, seed: int) -> dict:
    """
    Assign whole v2 groups to train/val/test/sealed with seeded simulated annealing so that
    per-class clip counts follow V2_SPLIT_FRACTIONS as closely as the (coarse) groups allow,
    while every held-out split keeps >= 3 clips of every class.

    Returns dict: Path -> split name. Also writes data/skeleton_dataset/v2_group_split.json.
    """
    names = _load_v2_original_names()
    groups: dict = {}
    for ci, cls in enumerate(CLASS_NAMES):
        for p in v2_by_cls.get(cls, []):
            key = _v2_group_key(cls, p.stem, names)
            g = groups.setdefault(key, {"paths": [], "counts": np.zeros(len(CLASS_NAMES))})
            g["paths"].append(p)
            g["counts"][ci] += 1

    keys = sorted(groups)
    C = np.stack([groups[k]["counts"] for k in keys])            # (G, 8)
    total = C.sum(axis=0)                                         # (8,)
    frac = np.array([V2_SPLIT_FRACTIONS[s] for s in V2_SPLIT_ORDER])[:, None]
    target = frac * total[None, :]                                # (4, 8)

    def cost(assign: np.ndarray) -> float:
        agg = np.zeros_like(target)
        for gi, s in enumerate(assign):
            agg[s] += C[gi]
        dev = (agg - target) / np.sqrt(target + 1.0)
        short = np.maximum(0.0, 3.0 - agg[1:])                    # val/test/sealed need >= 3 per class
        return float((dev ** 2).sum() + 50.0 * short.sum())

    rng = np.random.default_rng(seed)
    cur = rng.integers(0, len(V2_SPLIT_ORDER), size=len(keys))
    cur_cost = cost(cur)
    best, best_cost = cur.copy(), cur_cost
    n_iter = 40000
    for it in range(n_iter):
        temp = 5.0 * (0.01 / 5.0) ** (it / n_iter)
        cand = cur.copy()
        cand[rng.integers(0, len(keys))] = rng.integers(0, len(V2_SPLIT_ORDER))
        c_cost = cost(cand)
        if c_cost < cur_cost or rng.random() < np.exp((cur_cost - c_cost) / temp):
            cur, cur_cost = cand, c_cost
            if cur_cost < best_cost:
                best, best_cost = cur.copy(), cur_cost

    path_to_split: dict = {}
    report = {"seed": seed, "fractions": V2_SPLIT_FRACTIONS, "final_cost": best_cost, "groups": {}}
    for gi, key in enumerate(keys):
        split = V2_SPLIT_ORDER[int(best[gi])]
        for p in groups[key]["paths"]:
            path_to_split[p] = split
        report["groups"][key] = {
            "split": split,
            "clips": int(C[gi].sum()),
            "class_counts": {c: int(C[gi][i]) for i, c in enumerate(CLASS_NAMES) if C[gi][i] > 0},
        }
    with open(str(V2_GROUP_SPLIT_PATH), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("v2 group split saved to %s (cost=%.3f, %d groups)", V2_GROUP_SPLIT_PATH, best_cost, len(keys))
    return path_to_split


def split_clips_stratified(
    per_frame_paths: dict[str, list[Path]],
    seed: int = SPLIT_SEED,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]], dict[str, list[Path]], dict[str, list[Path]]]:
    """
    Video-level split into (train, val, test, sealed):
      - v1 clips: unchanged Phase 3.1 assignment (70/15/15, per class, `default_rng(seed)`);
        none of them go to `sealed`. Their subject/session is unknown, so they stay as before.
      - v2 clips: whole performer / subject groups are assigned together
        (see `_assign_v2_groups`), so no performer or recording subject appears in two splits.
      - `sealed` is a held-out final test made only of v2 groups; it must not be used for
        error analysis, model selection or threshold tuning until the final evaluation.
    """
    rng_v1 = np.random.default_rng(seed)
    train_clips: dict[str, list[Path]] = {}
    val_clips:   dict[str, list[Path]] = {}
    test_clips:  dict[str, list[Path]] = {}
    sealed_clips: dict[str, list[Path]] = {}

    v2_by_cls = {
        cls: sorted(p for p in per_frame_paths.get(cls, []) if p.name.startswith("v2_"))
        for cls in CLASS_NAMES
    }
    v2_assign = _assign_v2_groups(v2_by_cls, seed)

    for cls in CLASS_NAMES:
        files = sorted(per_frame_paths.get(cls, []))
        v1_files = [p for p in files if not p.name.startswith("v2_")]
        v2_files = v2_by_cls[cls]

        tr1, va1, te1 = _split_group_list(v1_files, rng_v1)

        train_clips[cls]  = sorted(tr1 + [p for p in v2_files if v2_assign[p] == "train"])
        val_clips[cls]    = sorted(va1 + [p for p in v2_files if v2_assign[p] == "val"])
        test_clips[cls]   = sorted(te1 + [p for p in v2_files if v2_assign[p] == "test"])
        sealed_clips[cls] = sorted(p for p in v2_files if v2_assign[p] == "sealed")

        sets = [
            {p.name for p in train_clips[cls]}, {p.name for p in val_clips[cls]},
            {p.name for p in test_clips[cls]},  {p.name for p in sealed_clips[cls]},
        ]
        for i in range(4):
            for j in range(i + 1, 4):
                assert not (sets[i] & sets[j]), f"Clip overlap between splits {i} and {j} in {cls}"

    return train_clips, val_clips, test_clips, sealed_clips


def _extract_windows_from_clips(
    clip_dict: dict[str, list[Path]],
    stride: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice clips into windows per class and track source clip ID for every window.

    Returns
    -------
    per_cls      : dict[class_name -> windows array (N_c, W, J, C)]
    per_cls_cids : dict[class_name -> clip_id string array (N_c,)]
    X            : (N_total, W, J, C)
    y            : (N_total,) integer class labels
    clip_ids     : (N_total,) string identifiers "<class>/<clip_stem>"
    """
    per_cls: dict[str, np.ndarray] = {}
    per_cls_cids: dict[str, np.ndarray] = {}
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_cids: list[np.ndarray] = []

    for cls in CLASS_NAMES:
        label_idx = LABEL_TO_IDX[cls]
        cls_wins: list[np.ndarray] = []
        cls_cids: list[np.ndarray] = []
        for npy_path in clip_dict.get(cls, []):
            seq = np.load(str(npy_path))
            wins = sliding_windows(seq, window_size=WINDOW_SIZE, stride=stride)
            cid_str = f"{cls}/{npy_path.stem}"
            cls_wins.append(wins)
            cls_cids.append(np.full(len(wins), cid_str, dtype="<U48"))

        if cls_wins:
            arr = np.concatenate(cls_wins, axis=0)
            cids = np.concatenate(cls_cids, axis=0)
            per_cls[cls] = arr
            per_cls_cids[cls] = cids
            all_X.append(arr)
            all_y.append(np.full(len(arr), label_idx, dtype=np.int64))
            all_cids.append(cids)
        else:
            per_cls[cls] = np.empty((0, WINDOW_SIZE, NUM_JOINTS, NUM_COORDS), dtype=np.float32)
            per_cls_cids[cls] = np.empty((0,), dtype="<U48")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    clip_ids = np.concatenate(all_cids, axis=0)
    return per_cls, per_cls_cids, X, y, clip_ids


def run_windowing_and_split(
    per_frame_paths: dict[str, list[Path]],
    stride: int,
    augment_times: int,
    rng_seed: int = SPLIT_SEED,
) -> dict:
    """
    Perform video-level split, extract windows, and augment ONLY the training split.
    Caps minority-class augmentation at 4x with conservative +-10 deg rotation
    to preserve vertical gravity-axis posture cues and avoid synthetic over-correction.
    """
    WINDOWED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    # 1. Video-level split
    train_clips, val_clips, test_clips, sealed_clips = split_clips_stratified(per_frame_paths, seed=rng_seed)

    # 2. Extract base windows for each split independently
    train_per_cls, train_per_cids, X_train_base, y_train_base, cids_train_base = _extract_windows_from_clips(train_clips, stride)
    val_per_cls,   _,              X_val,        y_val,        cids_val        = _extract_windows_from_clips(val_clips, stride)
    test_per_cls,  _,              X_test,       y_test,       cids_test       = _extract_windows_from_clips(test_clips, stride)
    sealed_per_cls, _,             X_sealed,     y_sealed,     cids_sealed     = _extract_windows_from_clips(sealed_clips, stride)

    train_base_counts = {cls: len(train_per_cls[cls]) for cls in CLASS_NAMES}
    val_counts        = {cls: len(val_per_cls[cls])   for cls in CLASS_NAMES}
    test_counts       = {cls: len(test_per_cls[cls])  for cls in CLASS_NAMES}

    # 3. Augment ONLY the training set (conservative cap <= 4x, rotation <= 10 deg)
    train_X_list = [X_train_base]
    train_y_list = [y_train_base]
    train_cid_list = [cids_train_base]
    train_aug_counts = {cls: 0 for cls in CLASS_NAMES}

    if augment_times > 0:
        # Target minimum ~400 windows in Train so only the 3 short-clip minority classes
        # (Sit_Down=52, Stand_Up=55, Crouching=111) receive synthetic copies, while the
        # 5 well-represented classes (>=405 base windows) stay at 1x base count.
        target_min_train = 400

        for cls in CLASS_NAMES:
            base_arr  = train_per_cls[cls]
            base_cids = train_per_cids[cls]
            n_base = len(base_arr)
            if n_base == 0:
                continue

            label_idx = LABEL_TO_IDX[cls]
            # Only augment minority classes (below target_min_train).
            # Cap at +4 augmented copies (1 base + 4 aug = 5x total windows)
            # to prevent synthetic oversaturation.
            needed_copies = max(0, int(np.ceil((target_min_train - n_base) / n_base)))
            n_copies = min(4, needed_copies)
            if n_copies == 0:
                log.info(
                    "Train %-12s — base: %4d | aug: +   0 (1x total) -> total: %4d",
                    cls, n_base, n_base,
                )
                continue

            aug_wins: list[np.ndarray] = []
            aug_cids: list[str] = []
            for _ in range(n_copies):
                indices = rng.integers(0, n_base, size=n_base)
                for idx in indices:
                    aug_wins.append(
                        augment_window(
                            base_arr[idx],
                            max_angle_deg=10.0,   # conservative +-10 deg preserves vertical posture cues
                            jitter_sigma=0.015,
                            speed_range=(0.85, 1.15),
                            rng=rng,
                        )
                    )
                    aug_cids.append(str(base_cids[idx]))

            if aug_wins:
                aug_X = np.stack(aug_wins, axis=0)
                aug_y = np.full(len(aug_X), label_idx, dtype=np.int64)
                train_X_list.append(aug_X)
                train_y_list.append(aug_y)
                train_cid_list.append(np.array(aug_cids, dtype="<U48"))
                train_aug_counts[cls] = len(aug_X)
                log.info(
                    "Train %-12s — base: %4d | aug: +%4d (×%d) -> total: %4d",
                    cls, n_base, len(aug_X), n_copies, n_base + len(aug_X),
                )

    X_train    = np.concatenate(train_X_list, axis=0)
    y_train    = np.concatenate(train_y_list, axis=0)
    cids_train = np.concatenate(train_cid_list, axis=0)

    # Shuffle Train windows deterministically
    perm_tr = rng.permutation(len(X_train))
    X_train, y_train, cids_train = X_train[perm_tr], y_train[perm_tr], cids_train[perm_tr]

    return {
        "X_train": X_train, "y_train": y_train, "cids_train": cids_train,
        "X_val":   X_val,   "y_val":   y_val,   "cids_val":   cids_val,
        "X_test":  X_test,  "y_test":  y_test,  "cids_test":  cids_test,
        "train_clips": train_clips,
        "val_clips":   val_clips,
        "test_clips":  test_clips,
        "sealed_clips": sealed_clips,
        "X_sealed": X_sealed, "y_sealed": y_sealed, "cids_sealed": cids_sealed,
        "train_base_counts": train_base_counts,
        "train_aug_counts":  train_aug_counts,
        "val_counts":        val_counts,
        "test_counts":       test_counts,
    }


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Save windowed dataset & manifest
# ════════════════════════════════════════════════════════════════════════════

def save_windowed_dataset(split_data: dict) -> None:
    """
    Save leak-free split arrays and combined arrays to data/skeleton_dataset/windowed/.
    """
    WINDOWED_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, cids_train = split_data["X_train"], split_data["y_train"], split_data["cids_train"]
    X_val,   y_val,   cids_val   = split_data["X_val"],   split_data["y_val"],   split_data["cids_val"]
    X_test,  y_test,  cids_test  = split_data["X_test"],  split_data["y_test"],  split_data["cids_test"]

    np.save(str(WINDOWED_DIR / "X_train.npy"), X_train)
    np.save(str(WINDOWED_DIR / "y_train.npy"), y_train)
    np.save(str(WINDOWED_DIR / "clip_ids_train.npy"), cids_train)
    np.save(str(WINDOWED_DIR / "X_val.npy"),   X_val)
    np.save(str(WINDOWED_DIR / "y_val.npy"),   y_val)
    np.save(str(WINDOWED_DIR / "clip_ids_val.npy"), cids_val)
    np.save(str(WINDOWED_DIR / "X_test.npy"),  X_test)
    np.save(str(WINDOWED_DIR / "y_test.npy"),  y_test)
    np.save(str(WINDOWED_DIR / "clip_ids_test.npy"), cids_test)

    # Save full per-frame sequences (10 FPS, stride=1) for Val and Test clips
    # so notebooks can run exact frame-by-frame SequenceBuffer + AlertStateMachine
    # realtime cadence (stride=1, 0.1s/tick) Event Evaluation.
    for split_tag, clip_dict in [("val", split_data["val_clips"]), ("test", split_data["test_clips"])]:
        seq_bundle: dict[str, np.ndarray] = {}
        for cls in CLASS_NAMES:
            for npy_path in clip_dict.get(cls, []):
                key = f"{cls}/{npy_path.stem}"
                seq_bundle[key] = np.load(str(npy_path)).astype(np.float32)
        np.savez_compressed(str(WINDOWED_DIR / f"{split_tag}_sequences.npz"), **seq_bundle)
        log.info("Saved %d full 10-FPS sequences to %s_sequences.npz", len(seq_bundle), split_tag)

    # Sealed final-test set (v2 performer/subject groups only). Kept OUTSIDE windowed/ so it is not
    # shipped to Kaggle and cannot be used for model selection by accident.
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(SEALED_DIR / "X_sealed.npy"), split_data["X_sealed"])
    np.save(str(SEALED_DIR / "y_sealed.npy"), split_data["y_sealed"])
    np.save(str(SEALED_DIR / "clip_ids_sealed.npy"), split_data["cids_sealed"])
    sealed_bundle: dict[str, np.ndarray] = {}
    for cls in CLASS_NAMES:
        for npy_path in split_data["sealed_clips"].get(cls, []):
            sealed_bundle[f"{cls}/{npy_path.stem}"] = np.load(str(npy_path)).astype(np.float32)
    np.savez_compressed(str(SEALED_DIR / "sealed_sequences.npz"), **sealed_bundle)
    (SEALED_DIR / "README.txt").write_text(
        "SEALED FINAL TEST - do not use for error analysis, model selection, or threshold tuning.\n"
        "Evaluate ONLY the final chosen model, once.\n",
        encoding="utf-8",
    )
    log.info("Saved sealed final-test set: %d windows / %d clips -> %s",
             len(split_data["X_sealed"]), len(sealed_bundle), SEALED_DIR)

    # Combined X.npy / y.npy for backward compatibility
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    np.save(str(WINDOWED_DIR / "X.npy"), X_all)
    np.save(str(WINDOWED_DIR / "y.npy"), y_all)

    log.info(
        "Saved leak-free splits: Train=%s, Val=%s, Test=%s (Total=%s)",
        X_train.shape, X_val.shape, X_test.shape, X_all.shape,
    )

    # Save split_manifest.json proving zero clip overlap
    manifest = {
        "split_strategy": "Stratified video-level (clip-level) split BEFORE windowing and augmentation",
        "seed": SPLIT_SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "zero_clip_overlap_verified": True,
        "val_and_test_unaugmented": True,
        "clip_counts": {
            "train": sum(len(v) for v in split_data["train_clips"].values()),
            "val":   sum(len(v) for v in split_data["val_clips"].values()),
            "test":  sum(len(v) for v in split_data["test_clips"].values()),
        },
        "window_counts": {
            "train_base": int(sum(split_data["train_base_counts"].values())),
            "train_augmented_added": int(sum(split_data["train_aug_counts"].values())),
            "train_total": int(len(X_train)),
            "val_pure":    int(len(X_val)),
            "test_pure":   int(len(X_test)),
            "total":       int(len(X_all)),
        },
        "splits_by_class": {
            cls: {
                "train_clips": [p.name for p in split_data["train_clips"][cls]],
                "val_clips":   [p.name for p in split_data["val_clips"][cls]],
                "test_clips":  [p.name for p in split_data["test_clips"][cls]],
            }
            for cls in CLASS_NAMES
        },
    }
    manifest["clip_counts"]["sealed"] = sum(len(v) for v in split_data["sealed_clips"].values())
    manifest["window_counts"]["sealed_pure"] = int(len(split_data["X_sealed"]))
    manifest["sealed_clips_by_class"] = {
        cls: [p.name for p in split_data["sealed_clips"][cls]] for cls in CLASS_NAMES
    }
    manifest["v2_grouping"] = (
        "v2 clips are split by whole performer (avi:P<id>) / subject (mp4:S<n>) groups; "
        "v1 clips keep their original Phase 3.1 assignment. See v2_group_split.json."
    )
    with open(str(MANIFEST_PATH), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Split manifest written to: %s", MANIFEST_PATH)


# ════════════════════════════════════════════════════════════════════════════
# Step 4 — Schema + distribution report
# ════════════════════════════════════════════════════════════════════════════

def write_schema(stride: int, split_data: dict) -> None:
    schema = {
        "description": (
            "Skeleton keypoint dataset for 8-class action recognition. "
            "Split at the VIDEO (CLIP) level (70/15/15) before windowing to guarantee "
            "zero temporal window overlap or augmentation leakage between Train, Val, and Test."
        ),
        "extraction_model": "yolov8n-pose.pt (COCO 17-keypoint, YOLO-Pose)",
        "sample_fps": TARGET_FPS,
        "normalisation": {
            "origin": "midpoint of left_hip (11) and right_hip (12)",
            "scale": "torso length = ||hip_mid - shoulder_mid||",
            "y_axis": "upward positive (raw pixel Y negated)",
            "fallback_scale": "1/3 of bounding-box diagonal when shoulders invisible",
        },
        "window_size": WINDOW_SIZE,
        "stride": stride,
        "num_joints": NUM_JOINTS,
        "num_coords": NUM_COORDS,
        "coordinate_convention": ["x_norm", "y_norm", "confidence"],
        "joint_names": JOINT_NAMES,
        "class_names": CLASS_NAMES,
        "label_to_index": LABEL_TO_IDX,
        "split_info": {
            "strategy": "video_level_stratified",
            "seed": SPLIT_SEED,
            "train_clips": sum(len(v) for v in split_data["train_clips"].values()),
            "val_clips":   sum(len(v) for v in split_data["val_clips"].values()),
            "test_clips":  sum(len(v) for v in split_data["test_clips"].values()),
            "train_windows": int(len(split_data["X_train"])),
            "val_windows":   int(len(split_data["X_val"])),
            "test_windows":  int(len(split_data["X_test"])),
            "val_test_augmented": False,
        },
        "files": {
            "per_frame/<class>/<clip>.npy": f"(T, {NUM_JOINTS}, {NUM_COORDS}) per-frame normalised keypoints",
            "windowed/X_train.npy": f"(N_train, {WINDOW_SIZE}, {NUM_JOINTS}, {NUM_COORDS}) train windows (base + augmented)",
            "windowed/y_train.npy": "(N_train,) integer class labels for train",
            "windowed/X_val.npy": f"(N_val, {WINDOW_SIZE}, {NUM_JOINTS}, {NUM_COORDS}) pure validation windows (no augmentation)",
            "windowed/y_val.npy": "(N_val,) integer class labels for validation",
            "windowed/X_test.npy": f"(N_test, {WINDOW_SIZE}, {NUM_JOINTS}, {NUM_COORDS}) pure test windows (no augmentation)",
            "windowed/y_test.npy": "(N_test,) integer class labels for test",
            "split_manifest.json": "Exact clip-to-split assignment proving zero data leakage",
        },
    }
    with open(str(SCHEMA_PATH), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    log.info("Schema written to: %s", SCHEMA_PATH)


def write_distribution_report(split_data: dict) -> None:
    y_train = split_data["y_train"]
    y_val   = split_data["y_val"]
    y_test  = split_data["y_test"]

    tot_tr_c = sum(len(v) for v in split_data["train_clips"].values())
    tot_va_c = sum(len(v) for v in split_data["val_clips"].values())
    tot_te_c = sum(len(v) for v in split_data["test_clips"].values())
    tot_all_c = tot_tr_c + tot_va_c + tot_te_c

    lines = [
        "Class Distribution Report (Video-Level Leak-Free Split)",
        "=" * 78,
        f"Clips  : Train={tot_tr_c} | "
        f"Val={tot_va_c} | "
        f"Test={tot_te_c} | Total={tot_all_c}",
        f"Windows: Train={len(y_train)} (base={sum(split_data['train_base_counts'].values())} + "
        f"aug={sum(split_data['train_aug_counts'].values())}) | "
        f"Val={len(y_val)} (pure) | Test={len(y_test)} (pure)",
        "",
        f"{'Class':<12} | {'Clips(T/V/Te)':>13} | {'Train(Base)':>11} | {'Train(Aug)':>10} | {'Val(Pure)':>9} | {'Test(Pure)':>10}",
        "-" * 78,
    ]
    for i, cls in enumerate(CLASS_NAMES):
        c_tr = len(split_data["train_clips"][cls])
        c_va = len(split_data["val_clips"][cls])
        c_te = len(split_data["test_clips"][cls])
        w_tr_base = split_data["train_base_counts"][cls]
        w_tr_tot  = int((y_train == i).sum())
        w_va      = int((y_val == i).sum())
        w_te      = int((y_test == i).sum())
        clip_str  = f"{c_tr:2d}/{c_va:2d}/{c_te:2d}"
        lines.append(
            f"{cls:<12} | {clip_str:>13} | {w_tr_base:>11d} | {w_tr_tot:>10d} | {w_va:>9d} | {w_te:>10d}"
        )

    lines.append("-" * 78)
    tot_clip_str = f"{tot_tr_c}/{tot_va_c}/{tot_te_c}"
    lines.append(
        f"{'TOTAL':<12} | {tot_clip_str:>13} | "
        f"{sum(split_data['train_base_counts'].values()):>11d} | "
        f"{len(y_train):>10d} | {len(y_val):>9d} | {len(y_test):>10d}"
    )

    lines.append("")
    lines.append("Sealed final test (v2 groups only, NOT in windowed/, do not peek):")
    y_sealed = split_data["y_sealed"]
    for i, cls in enumerate(CLASS_NAMES):
        lines.append(f"  {cls:<12} clips={len(split_data['sealed_clips'][cls]):3d}  windows={int((y_sealed == i).sum()):4d}")
    lines.append(f"  {'TOTAL':<12} clips={sum(len(v) for v in split_data['sealed_clips'].values()):3d}  windows={len(y_sealed):4d}")

    base_vals = [v for v in split_data["train_base_counts"].values() if v > 0]
    post_vals = [int((y_train == i).sum()) for i in range( len(CLASS_NAMES) )]
    base_ratio = max(base_vals) / min(base_vals)
    post_ratio = max(post_vals) / min(post_vals)
    lines.append("")
    lines.append(f"Train Imbalance Ratio (Base)             : {max(base_vals)}/{min(base_vals)} = {base_ratio:.2f}x")
    lines.append(f"Train Imbalance Ratio (Post-Augmentation): {max(post_vals)}/{min(post_vals)} = {post_ratio:.2f}x")

    report = "\n".join(lines)
    print(report)
    with open(str(REPORT_PATH), "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Distribution report saved to: %s", REPORT_PATH)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1.2 & 3.1 — extraction & leak-free split")
    p.add_argument(
        "--model",
        default=str(PROJECT_ROOT / YOLO_POSE_MODEL_PATH),
        help="YOLO-Pose checkpoint path",
    )
    p.add_argument("--stride",         type=int,  default=4,    help="Sliding window stride (default: 4)")
    p.add_argument("--augment_times",  type=int,  default=1,    help="Min augmented copies per train window (default: 1)")
    p.add_argument("--no_augment",     action="store_true",      help="Skip augmentation")
    p.add_argument("--force",          action="store_true",      help="Re-extract even if .npy already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    log.info("=" * 60)
    log.info("Phase 1.2 & 3.1 — Skeleton Extraction & Video-Level Split")
    log.info("  Window    : %d frames, stride=%d", WINDOW_SIZE, args.stride)
    log.info("  Augment   : %s (Train split only)", not args.no_augment)
    log.info("=" * 60)

    # Load model lazily only if needed (all 559 clips are already extracted unless --force)
    model = None
    if args.force:
        model_path = args.model if Path(args.model).exists() else "yolov8n-pose.pt"
        model = load_pose_model(model_path)

    # Step 1: Per-frame extraction (reuses existing per_frame/*.npy and incrementally extracts new v2 clips)
    per_frame_paths = run_per_frame_extraction(model, force=args.force)

    # Step 2: Video-level split + windowing + train-only augmentation
    aug_times = 0 if args.no_augment else args.augment_times
    split_data = run_windowing_and_split(
        per_frame_paths, stride=args.stride, augment_times=aug_times
    )

    # Step 3: Save arrays & split manifest
    save_windowed_dataset(split_data)

    # Step 4: Schema + report
    write_schema(stride=args.stride, split_data=split_data)
    write_distribution_report(split_data)

    log.info("Completed in %.1fs.", time.time() - t_start)


if __name__ == "__main__":
    main()
