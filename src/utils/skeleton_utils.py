"""
skeleton_utils.py — Shared keypoint extraction and normalisation functions.

This module is the SINGLE SOURCE OF TRUTH for:
  • Running YOLO-Pose inference on a BGR frame
  • Normalising raw keypoints to a hip-centred, torso-scaled coordinate system
  • Selecting the primary person in a multi-person frame
  • Slicing a per-frame keypoint sequence into fixed-length windows
  • Augmenting skeleton sequences (rotation, jitter, time-warp)

Imported by:
  • notebooks/01_data_preparation.ipynb   (training-data extraction)
  • src/pipeline/detector.py              (real-time inference)

Any change to normalisation logic MUST be made here only to prevent
train/inference skew.

Joint schema (COCO 17-keypoint, YOLO-Pose convention):
  Index → joint name  (see config.JOINT_NAMES for the full list)
  Coordinate convention after normalisation: (x_norm, y_norm, confidence)
    • Origin  : midpoint of left_hip (11) and right_hip (12)
    • Scale   : torso length = ||hip_mid − shoulder_mid||  (1 unit = 1 torso)
    • Y axis  : upward positive (raw pixel Y is downward, so Y is negated)
    • confidence : passed through unchanged from YOLO output
"""

from __future__ import annotations

import sys
import math
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from utils.config import (
    CONF_THRESHOLD,
    NUM_JOINTS,
    NUM_COORDS,
    JOINT_NAMES,
    YOLO_POSE_MODEL_PATH,
    WINDOW_SIZE,
)

log = logging.getLogger(__name__)

# ── Joint index constants ────────────────────────────────────────────────────
_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}
_LEFT_HIP       = _IDX["left_hip"]        # 11
_RIGHT_HIP      = _IDX["right_hip"]       # 12
_LEFT_SHOULDER  = _IDX["left_shoulder"]   # 5
_RIGHT_SHOULDER = _IDX["right_shoulder"]  # 6

# ── Lazy YOLO import (avoids import-time cost when module is used as a lib) ──
_YOLO_CLS = None


def _get_yolo_class():
    global _YOLO_CLS
    if _YOLO_CLS is None:
        import os
        import tempfile
        if "YOLO_CONFIG_DIR" not in os.environ:
            default_cfg = Path(os.environ.get("APPDATA", "")) / "Ultralytics" / "settings.json"
            try:
                default_cfg.exists()
            except PermissionError:
                os.environ["YOLO_CONFIG_DIR"] = tempfile.gettempdir()
        from ultralytics import YOLO as _YOLO
        _YOLO_CLS = _YOLO
    return _YOLO_CLS


# ════════════════════════════════════════════════════════════════════════════
# 1. Model loading
# ════════════════════════════════════════════════════════════════════════════

def load_pose_model(model_path: str = YOLO_POSE_MODEL_PATH):
    """
    Load (or auto-download) a YOLO-Pose checkpoint.

    Parameters
    ----------
    model_path : path to a `.pt` file; if the file does not exist,
                 Ultralytics will attempt to auto-download it.

    Returns
    -------
    ultralytics.YOLO instance ready for inference.
    """
    YOLO = _get_yolo_class()
    model = YOLO(model_path)
    log.info("Loaded pose model from: %s", model_path)
    return model


# ════════════════════════════════════════════════════════════════════════════
# 2. Single-frame extraction
# ════════════════════════════════════════════════════════════════════════════

def extract_raw_keypoints(
    frame_bgr: np.ndarray,
    model,
    conf_threshold: float = CONF_THRESHOLD,
) -> Optional[np.ndarray]:
    """
    Run YOLO-Pose on one BGR frame; return raw pixel keypoints for the
    primary person.

    Person selection: detection with the highest bounding-box confidence.

    Returns
    -------
    np.ndarray of shape (NUM_JOINTS, 3) → [x_px, y_px, kp_confidence],
    or None when no person passes the confidence threshold.
    """
    results = model(frame_bgr, verbose=False)

    if not results or results[0].keypoints is None:
        return None

    kp_data = results[0].keypoints
    boxes   = results[0].boxes

    if kp_data.xy is None or len(kp_data.xy) == 0:
        return None

    # Pick the person with the highest detection confidence
    if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
        confs    = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        if confs[best_idx] < conf_threshold:
            return None
    else:
        best_idx = 0

    # kp_data.data: (N_persons, NUM_JOINTS, 3)  columns → [x_px, y_px, conf]
    kp = kp_data.data[best_idx].cpu().numpy().astype(np.float32)   # (17, 3)

    # Guard: unexpected shape (e.g. model returned 0 joints)
    if kp.shape[0] == 0 or kp.ndim != 2 or kp.shape[1] < 3:
        return None

    # Zero-out joints whose keypoint confidence is below threshold
    kp[kp[:, 2] < conf_threshold, :] = 0.0

    return kp


def normalise_keypoints(kp_raw: np.ndarray) -> Optional[np.ndarray]:
    """
    Hip-centred, torso-length-scaled normalisation.

    Parameters
    ----------
    kp_raw : (NUM_JOINTS, 3) → [x_px, y_px, conf]

    Returns
    -------
    (NUM_JOINTS, 3) → [x_norm, y_norm, conf]  or None if hips undetected
    and shoulder-based virtual hip fallback is not applicable.

    Fallback behaviour:
      • Any joint with pixel coordinates exact (0.0, 0.0) has its confidence
        zeroed on a local copy so undetected Ultralytics joints never act as
        anchors.
      • Half-body / close-up shots: when both hips are undetected (conf == 0.0)
        but both shoulders are visible in an approximately horizontal posture
        (`sh_w > 10.0` px and `|x_L - x_R| > 0.6 * sh_w`), virtual hip anchors
        are placed at `hip_y = sh_mid_y + 1.08 * sh_w` so upper-body actions
        such as Drink / Eat are preserved instead of being dropped.
      • When shoulders are invisible, torso length falls back to 1/3 of the
        bounding-box diagonal of visible joints.
    """
    kp_work = np.asarray(kp_raw, dtype=np.float32).copy()
    zero_xy = (kp_work[:, 0] == 0.0) & (kp_work[:, 1] == 0.0)
    kp_work[zero_xy, 2] = 0.0

    left_hip  = kp_work[_LEFT_HIP]
    right_hip = kp_work[_RIGHT_HIP]

    left_sh  = kp_work[_LEFT_SHOULDER]
    right_sh = kp_work[_RIGHT_SHOULDER]

    # Shoulder-based virtual hip fallback for half-body / medium-shot videos
    if left_hip[2] == 0.0 and right_hip[2] == 0.0:
        if left_sh[2] > 0.0 and right_sh[2] > 0.0:
            sh_mid = 0.5 * (left_sh[:2] + right_sh[:2])
            sh_w = float(np.linalg.norm(left_sh[:2] - right_sh[:2]))
            if sh_w > 10.0 and abs(float(left_sh[0] - right_sh[0])) > 0.6 * sh_w:
                hip_y = float(sh_mid[1]) + 1.08 * sh_w
                kp_work[_LEFT_HIP]  = [float(left_sh[0]),  hip_y, 0.25]
                kp_work[_RIGHT_HIP] = [float(right_sh[0]), hip_y, 0.25]
                left_hip  = kp_work[_LEFT_HIP]
                right_hip = kp_work[_RIGHT_HIP]
            else:
                return None
        else:
            return None

    visible_hips = [h for h in (left_hip, right_hip) if h[2] > 0.0]
    hip_mid = np.mean([h[:2] for h in visible_hips], axis=0)  # (2,)

    visible_sh = [s for s in (left_sh, right_sh) if s[2] > 0.0]

    if visible_sh:
        sh_mid    = np.mean([s[:2] for s in visible_sh], axis=0)
        torso_len = float(np.linalg.norm(sh_mid - hip_mid))
    else:
        torso_len = 0.0

    if torso_len < 1e-6:
        # Fallback: use 1/3 of bounding-box diagonal
        visible_xy = kp_work[kp_work[:, 2] > 0.0, :2]
        if len(visible_xy) == 0:
            return None
        bbox_diag = float(np.linalg.norm(visible_xy.max(0) - visible_xy.min(0)))
        torso_len = bbox_diag / 3.0 if bbox_diag > 1e-6 else 1.0

    kp_norm = kp_work.copy()
    kp_norm[:, 0] =  (kp_work[:, 0] - hip_mid[0]) / torso_len
    kp_norm[:, 1] = -(kp_work[:, 1] - hip_mid[1]) / torso_len  # flip Y axis

    return kp_norm.astype(np.float32)


def extract_and_normalise(
    frame_bgr: np.ndarray,
    model,
    conf_threshold: float = CONF_THRESHOLD,
) -> Optional[np.ndarray]:
    """
    One-step wrapper: detect → normalise.

    This is the function called by both the notebook and
    src/pipeline/detector.py during real-time inference.

    Returns (NUM_JOINTS, 3) normalised keypoints or None.
    """
    raw = extract_raw_keypoints(frame_bgr, model, conf_threshold)
    if raw is None:
        return None
    return normalise_keypoints(raw)


# ════════════════════════════════════════════════════════════════════════════
# 3. Clip-level extraction
# ════════════════════════════════════════════════════════════════════════════

def compute_resample_indices(
    total_frames: int,
    native_fps: float,
    sample_fps: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute exact source frame indices and target timestamps (in seconds)
    to resample a video of `total_frames` at `native_fps` to `sample_fps`.

    For example, at native_fps=25.0 and sample_fps=10.0, source_indices
    are [0, 2, 5, 8, 10, 12, 15, ...] (alternating steps of 2 and 3 frames),
    achieving an exact average rate of 10.0 FPS instead of 12.5 FPS.
    """
    if total_frames <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    if native_fps <= 0:
        native_fps = 30.0
    duration = total_frames / native_fps
    target_times = np.arange(0.0, duration, 1.0 / sample_fps, dtype=np.float64)
    source_indices = np.floor(target_times * native_fps + 0.5).astype(np.int64)
    source_indices = np.clip(source_indices, 0, total_frames - 1)
    return source_indices, target_times


def clip_to_keypoint_sequence(
    video_path: str,
    model,
    conf_threshold: float = CONF_THRESHOLD,
    max_missing_ratio: float = 0.5,
    sample_fps: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Extract normalised per-frame keypoints for a video clip resampled by timestamp
    to `sample_fps` (default 10.0 FPS -> 0.10s per frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("Cannot open video: %s", video_path)
        return None

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 30.0

    native_frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        native_frames.append(frame)
    cap.release()

    if not native_frames:
        log.warning("Empty video: %s", video_path)
        return None

    source_indices, _ = compute_resample_indices(
        len(native_frames), native_fps, sample_fps=sample_fps
    )

    frames_kp: list[np.ndarray] = []
    missing = 0
    _zero = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    cache: dict[int, Optional[np.ndarray]] = {}

    for idx_val in source_indices:
        f_idx = int(idx_val)
        if f_idx not in cache:
            cache[f_idx] = extract_and_normalise(native_frames[f_idx], model, conf_threshold)
        kp = cache[f_idx]
        if kp is None:
            frames_kp.append(_zero.copy())
            missing += 1
        else:
            frames_kp.append(kp.copy())

    total = len(frames_kp)
    if total == 0:
        return None

    if missing / total > max_missing_ratio:
        log.warning(
            "Too many missing detections (%.0f%%) in: %s",
            100 * missing / total,
            video_path,
        )
        return None

    return np.stack(frames_kp, axis=0)   # (T_sampled, 17, 3)


# ════════════════════════════════════════════════════════════════════════════
# 4. Windowing
# ════════════════════════════════════════════════════════════════════════════

def sliding_windows(
    sequence: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = 4,
) -> np.ndarray:
    """
    Slice a keypoint sequence into overlapping fixed-length windows.

    Short sequences (T < window_size) are right-padded by repeating the
    last frame so that at least one window is always produced.

    Parameters
    ----------
    sequence    : (T, NUM_JOINTS, NUM_COORDS)
    window_size : number of frames per window  (default = config.WINDOW_SIZE)
    stride      : step between successive window starts (default = 4)
                  Use stride=4 for training data (high overlap, more windows).
                  Use stride=window_size for non-overlapping windows.

    Returns
    -------
    np.ndarray of shape (N_windows, window_size, NUM_JOINTS, NUM_COORDS)
    """
    T = sequence.shape[0]
    if T < window_size:
        pad_len = window_size - T
        pad = np.tile(sequence[-1:], (pad_len, 1, 1))
        sequence = np.concatenate([sequence, pad], axis=0)
        T = window_size

    windows = [
        sequence[s : s + window_size]
        for s in range(0, T - window_size + 1, stride)
    ]
    return np.stack(windows, axis=0)   # (N, W, J, C)


# ════════════════════════════════════════════════════════════════════════════
# 5. Augmentation (skeleton-coordinate domain)
# ════════════════════════════════════════════════════════════════════════════

def augment_rotation(
    window: np.ndarray,
    max_angle_deg: float = 15.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Rotate all (x, y) coordinates by a random angle in the XY plane.
    Confidence values are unchanged.

    Parameters
    ----------
    window        : (W, NUM_JOINTS, 3)
    max_angle_deg : maximum absolute rotation angle in degrees
    rng           : NumPy random generator (created if None)

    Returns
    -------
    Rotated copy of `window`, same shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    angle = rng.uniform(-max_angle_deg, max_angle_deg)
    rad   = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    out = window.copy()
    x = window[:, :, 0].copy()
    y = window[:, :, 1].copy()
    out[:, :, 0] = cos_a * x - sin_a * y
    out[:, :, 1] = sin_a * x + cos_a * y
    return out


def augment_jitter(
    window: np.ndarray,
    sigma: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add Gaussian noise to (x, y) coordinates.
    Only joints with non-zero confidence are noised (zero joints stay zero).

    Parameters
    ----------
    window : (W, NUM_JOINTS, 3)
    sigma  : standard deviation of noise in torso-length units
    rng    : NumPy random generator

    Returns
    -------
    Jittered copy of `window`.
    """
    if rng is None:
        rng = np.random.default_rng()

    out  = window.copy()
    mask = window[:, :, 2] > 0.0   # (W, J) — True for visible joints
    noise = rng.normal(0.0, sigma, size=(window.shape[0], window.shape[1], 2))
    out[:, :, 0] += noise[:, :, 0] * mask
    out[:, :, 1] += noise[:, :, 1] * mask
    return out


def augment_time_warp(
    window: np.ndarray,
    speed_range: tuple[float, float] = (0.8, 1.2),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Resample the temporal axis at a random speed factor, then resize back
    to the original window length using linear interpolation.

    Parameters
    ----------
    window      : (W, NUM_JOINTS, 3)
    speed_range : (min_speed, max_speed) multiplier; >1 speeds up (fewer
                  source frames per output), <1 slows down.
    rng         : NumPy random generator

    Returns
    -------
    Time-warped copy of `window`, same shape (W, NUM_JOINTS, 3).
    """
    if rng is None:
        rng = np.random.default_rng()

    W = window.shape[0]
    speed  = rng.uniform(*speed_range)
    # Number of source frames to sample before stretching back to W
    n_src  = max(2, int(round(W * speed)))
    src_t  = np.linspace(0, W - 1, n_src)
    dst_t  = np.linspace(0, W - 1, W)

    out = np.empty_like(window)
    for j in range(window.shape[1]):
        for c in range(window.shape[2]):
            out[:, j, c] = np.interp(dst_t, src_t, np.interp(src_t, np.arange(W), window[:, j, c]))
    return out.astype(np.float32)


def augment_window(
    window: np.ndarray,
    rotate: bool = True,
    jitter: bool = True,
    time_warp: bool = True,
    max_angle_deg: float = 15.0,
    jitter_sigma: float  = 0.02,
    speed_range: tuple[float, float] = (0.8, 1.2),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply all enabled augmentations in sequence: rotation → jitter → time-warp.

    Parameters
    ----------
    window : (W, NUM_JOINTS, 3)

    Returns
    -------
    Augmented copy, same shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = window.copy()
    if rotate:
        out = augment_rotation(out, max_angle_deg=max_angle_deg, rng=rng)
    if jitter:
        out = augment_jitter(out, sigma=jitter_sigma, rng=rng)
    if time_warp:
        out = augment_time_warp(out, speed_range=speed_range, rng=rng)
    return out
