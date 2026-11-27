"""
annotate_fall_timestamps.py — Automatic + resume-safe temporal annotator for `Fall` clips.

Splits each raw `Fall` video (resampled to TARGET_FPS = 10 Hz) into 3 explicit zones:
  • [0, fall_start_tick - 1]          -> `Idle`  (standing / sitting before collapse)
  • [fall_start_tick, fall_end_tick]  -> `Fall`  (active collapse & impact window)
  • [fall_end_tick + 1, T - 1]        -> `Lying` (motionless horizontal post-fall phase)

Outputs `data/skeleton_dataset/fall_timestamps.csv` for visual inspection or editing
via `streamlit run scripts/fall_annotator_app.py`, while preserving any manually
verified rows (`status == "verified"`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import TARGET_FPS
from utils.skeleton_utils import compute_resample_indices

FALL_SOURCES: list[tuple[str, Path]] = [
    ("v1", PROJECT_ROOT / "data" / "raw_videos" / "Fall"),
    ("v2", PROJECT_ROOT / "data" / "raw_videos_v2" / "Fall"),
]
PER_FRAME_FALL_DIR = PROJECT_ROOT / "data" / "skeleton_dataset" / "per_frame" / "Fall"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "skeleton_dataset" / "fall_timestamps.csv"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_CORE_JOINTS: list[int] = [0, 5, 6, 11, 12, 13, 14, 15, 16]


def _interp_1d(vals: np.ndarray, valid_mask: np.ndarray, fallback: float) -> np.ndarray:
    """Linearly interpolate invalid samples cleanly for smooth kinematic curves."""
    n = len(vals)
    out = vals.astype(np.float64).copy()
    if n == 0 or not bool(np.any(valid_mask)):
        return np.full((n,), float(fallback), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    good_idx = idx[valid_mask]
    good_vals = out[valid_mask]
    return np.interp(idx, good_idx, good_vals)


def estimate_fall_interval_from_keypoints(
    arr: np.ndarray,
    fps: float = float(TARGET_FPS),
) -> dict[str, Any]:
    """
    Estimate `[fall_start_tick, fall_end_tick]` from a normalised `(T, 17, 3)` sequence.

    Returns start/end tick indices together with per-tick torso height (`sh_y`),
    tilt (`sin_theta`), and collapse score for UI visualization.
    """
    t_len = int(len(arr))
    if t_len <= 0:
        return {
            "fall_start_tick": 0,
            "fall_end_tick": 0,
            "sh_y": [],
            "sin_theta": [],
            "collapse_score": [],
            "notes": "empty_sequence",
        }

    # Extract torso center (left=5, right=6), tilt, aspect ratio & core speed
    raw_sh_x = np.zeros((t_len,), dtype=np.float64)
    raw_sh_y = np.zeros((t_len,), dtype=np.float64)
    sh_valid = np.zeros((t_len,), dtype=bool)
    yx_ratio = np.ones((t_len,), dtype=np.float64)
    core_speed = np.zeros((t_len,), dtype=np.float64)

    for t in range(t_len):
        c5 = float(arr[t, 5, 2])
        c6 = float(arr[t, 6, 2])
        if c5 > 0.0 and c6 > 0.0:
            raw_sh_x[t] = 0.5 * (float(arr[t, 5, 0]) + float(arr[t, 6, 0]))
            raw_sh_y[t] = 0.5 * (float(arr[t, 5, 1]) + float(arr[t, 6, 1]))
            sh_valid[t] = True
        elif c5 > 0.0:
            raw_sh_x[t] = float(arr[t, 5, 0])
            raw_sh_y[t] = float(arr[t, 5, 1])
            sh_valid[t] = True
        elif c6 > 0.0:
            raw_sh_x[t] = float(arr[t, 6, 0])
            raw_sh_y[t] = float(arr[t, 6, 1])
            sh_valid[t] = True

        vis = arr[t, :, 2] >= 0.30
        if bool(np.any(vis)):
            ys = arr[t, :, 1][vis]
            xs = arr[t, :, 0][vis]
            yx_ratio[t] = float((np.max(ys) - np.min(ys)) / (np.max(xs) - np.min(xs) + 1e-4))

        if t > 0:
            pair_vis = (arr[t, _CORE_JOINTS, 2] > 0.0) & (arr[t - 1, _CORE_JOINTS, 2] > 0.0)
            if bool(np.any(pair_vis)):
                d = np.linalg.norm(
                    arr[t, _CORE_JOINTS, :2][pair_vis] - arr[t - 1, _CORE_JOINTS, :2][pair_vis],
                    axis=-1,
                )
                core_speed[t] = float(np.mean(d))

    sh_x = _interp_1d(raw_sh_x, sh_valid, 0.0)
    sh_y = _interp_1d(raw_sh_y, sh_valid, 0.5)
    sin_theta = np.abs(sh_x) / (np.sqrt(sh_x ** 2 + sh_y ** 2) + 1e-6)

    if t_len < 12:
        return {
            "fall_start_tick": 0,
            "fall_end_tick": t_len - 1,
            "sh_y": sh_y.tolist(),
            "sin_theta": sin_theta.tolist(),
            "collapse_score": core_speed.tolist(),
            "notes": "short_clip",
        }

    # Finite-difference vertical collapse + tilt gain + core motion spike
    step = 2
    drop_vel = np.zeros((t_len,), dtype=np.float64)
    tilt_vel = np.zeros((t_len,), dtype=np.float64)
    for t in range(t_len):
        t0 = max(0, t - step)
        t1 = min(t_len - 1, t + step)
        drop_vel[t] = max(0.0, float(sh_y[t0] - sh_y[t1]))
        tilt_vel[t] = max(0.0, float(sin_theta[t1] - sin_theta[t0]))

    collapse_score = 1.2 * drop_vel + 1.0 * tilt_vel + 2.0 * core_speed
    t_peak = int(np.argmax(collapse_score))

    if float(collapse_score[t_peak]) < 0.08:
        end_tick = min(t_len - 1, max(6, int(round(0.6 * t_len))))
        return {
            "fall_start_tick": 0,
            "fall_end_tick": end_tick,
            "sh_y": sh_y.tolist(),
            "sin_theta": sin_theta.tolist(),
            "collapse_score": collapse_score.tolist(),
            "notes": "low_motion_or_already_fallen",
        }

    # Walk backward from t_peak to find the last calm upright/seated tick before drop
    fall_start = 0
    for t in range(t_peak, -1, -1):
        is_upright = (sh_y[t] > 0.68) and (sin_theta[t] < 0.38)
        prev_t = max(0, t - 1)
        prev_calm = abs(float(sh_y[t] - sh_y[prev_t])) < 0.04 and abs(float(sin_theta[t] - sin_theta[prev_t])) < 0.05
        if is_upright and prev_calm:
            fall_start = min(t_peak, t + 1)
            break
    else:
        # Fallback: first local quiet point before peak
        quiet = np.where(collapse_score[: t_peak + 1] < 0.25 * float(collapse_score[t_peak]))[0]
        if len(quiet) > 0:
            fall_start = min(t_peak, int(quiet[-1]) + 1)
        else:
            fall_start = max(0, t_peak - 5)

    # Walk forward from t_peak to find first settled post-impact horizontal frame
    fall_end = min(t_len - 1, fall_start + 15)
    for t in range(t_peak, t_len):
        is_horiz = (sh_y[t] < 0.50) or (sin_theta[t] > 0.48) or (yx_ratio[t] < 1.10)
        next_t = min(t_len - 1, t + 1)
        settled_after = (core_speed[next_t] < 0.045) and (abs(float(sh_y[next_t] - sh_y[t])) < 0.05)
        if is_horiz and settled_after and (t - fall_start) >= 5:
            fall_end = t
            break

    # Enforce practical duration bounds: >= 6 ticks (0.6s) and <= 28 ticks (2.8s)
    min_end = min(t_len - 1, fall_start + 5)
    max_end = min(t_len - 1, fall_start + 27)
    fall_end = max(min_end, min(fall_end, max_end))
    if (fall_end - fall_start) > 27:
        fall_start = max(0, fall_end - 27)

    return {
        "fall_start_tick": int(fall_start),
        "fall_end_tick": int(fall_end),
        "sh_y": sh_y.tolist(),
        "sin_theta": sin_theta.tolist(),
        "collapse_score": collapse_score.tolist(),
        "notes": "auto_kinematic",
    }


def _probe_video_ticks(video_path: Path, target_fps: float = float(TARGET_FPS)) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0.0
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if native_fps <= 0.0 or native_fps > 240.0:
        native_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    indices, _ = compute_resample_indices(total_frames, native_fps, target_fps)
    dur_s = float(total_frames) / native_fps if total_frames > 0 else 0.0
    return int(len(indices)), dur_s


def generate_fall_timestamps_csv(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    force_overwrite: bool = False,
) -> pd.DataFrame:
    """Scan all Fall clips in `raw_videos/` + `raw_videos_v2/` and write timestamp CSV."""
    existing_verified: dict[str, dict[str, Any]] = {}
    if output_csv.exists() and not force_overwrite:
        prev_df = pd.read_csv(output_csv)
        for _, row in prev_df.iterrows():
            if str(row.get("status", "")).lower() == "verified":
                existing_verified[str(row["clip_id"])] = row.to_dict()

    rows: list[dict[str, Any]] = []
    for src_tag, src_dir in FALL_SOURCES:
        if not src_dir.is_dir():
            continue
        videos = sorted(
            (p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
            key=lambda p: p.name,
        )
        for vp in videos:
            clip_id = f"Fall/{vp.stem}"
            rel_path = str(vp.relative_to(PROJECT_ROOT)).replace("\\", "/")
            npy_path = PER_FRAME_FALL_DIR / f"{vp.stem}.npy"
            has_npy = npy_path.is_file()

            if clip_id in existing_verified:
                rows.append(existing_verified[clip_id])
                continue

            if has_npy:
                arr = np.load(npy_path)
                t_len = int(len(arr))
                _, vid_dur = _probe_video_ticks(vp)
                dur_s = vid_dur if vid_dur > 0 else float(t_len) / float(TARGET_FPS)
                est = estimate_fall_interval_from_keypoints(arr, fps=float(TARGET_FPS))
                start_t = int(est["fall_start_tick"])
                end_t = int(est["fall_end_tick"])
                notes = str(est["notes"])
            else:
                t_len, dur_s = _probe_video_ticks(vp)
                start_t = 0
                end_t = max(0, min(t_len - 1, max(5, int(round(0.7 * t_len)))))
                notes = "no_skeleton_npy"

            start_t = max(0, min(start_t, max(0, t_len - 1)))
            end_t = max(start_t, min(end_t, max(0, t_len - 1)))
            pre_ticks = start_t
            act_ticks = (end_t - start_t + 1) if t_len > 0 else 0
            post_ticks = max(0, t_len - 1 - end_t)

            rows.append(
                {
                    "clip_id": clip_id,
                    "dataset_src": src_tag,
                    "video_rel_path": rel_path,
                    "has_skeleton_npy": bool(has_npy),
                    "total_ticks": int(t_len),
                    "duration_sec": round(float(dur_s), 2),
                    "fall_start_tick": int(start_t),
                    "fall_end_tick": int(end_t),
                    "fall_start_sec": round(float(start_t) / float(TARGET_FPS), 2),
                    "fall_end_sec": round(float(end_t) / float(TARGET_FPS), 2),
                    "pre_fall_ticks": int(pre_ticks),
                    "active_fall_ticks": int(act_ticks),
                    "post_fall_lying_ticks": int(post_ticks),
                    "pre_fall_label": "Idle",
                    "active_label": "Fall",
                    "post_fall_label": "Lying",
                    "status": "auto",
                    "notes": notes,
                }
            )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate Fall start/end timestamps CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--force", action="store_true", help="Overwrite even manually verified rows")
    args = parser.parse_args()

    out_path = Path(args.output)
    df = generate_fall_timestamps_csv(output_csv=out_path, force_overwrite=bool(args.force))
    if len(df) == 0:
        print("No Fall clips found in data/raw_videos/Fall or data/raw_videos_v2/Fall.")
        return

    print(f"Saved {len(df)} Fall timestamp records -> {out_path}")
    print(f"  Skeleton .npy available  : {int(df['has_skeleton_npy'].sum())}/{len(df)} clips")
    print(f"  Verified vs Auto         : {int((df['status'] == 'verified').sum())} verified / {int((df['status'] == 'auto').sum())} auto")
    print(f"  Pre-fall Idle ticks      : {int(df['pre_fall_ticks'].sum())} ticks (across {int((df['pre_fall_ticks'] > 0).sum())} clips)")
    print(f"  Active Fall ticks        : {int(df['active_fall_ticks'].sum())} ticks (mean {df['active_fall_ticks'].mean():.1f} ticks = {df['active_fall_ticks'].mean()/10.0:.2f}s)")
    print(f"  Post-fall Lying ticks    : {int(df['post_fall_lying_ticks'].sum())} ticks (across {int((df['post_fall_lying_ticks'] > 0).sum())} clips)")


if __name__ == "__main__":
    main()
