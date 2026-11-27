"""
detector.py — Stage 1: Pretrained YOLO-Pose integration for real-time inference.

Responsibilities:
  • Load the YOLO-Pose checkpoint (once per process, cached by the caller).
  • Run per-frame person detection and keypoint extraction.
  • Return a typed PoseResult dataclass consumed by the rest of the pipeline.

All extraction/normalisation logic delegates to skeleton_utils so there is
exactly ONE implementation shared between this module and the training notebook.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Project path ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from utils.config import (
    CONF_THRESHOLD,
    TRACK_CONF_THRESHOLD,
    NUM_JOINTS,
    NUM_COORDS,
    YOLO_POSE_MODEL_PATH,
)
from utils.skeleton_utils import (
    load_pose_model,          # re-exported for convenience
    extract_raw_keypoints,
    normalise_keypoints,
)

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Output type
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PoseResult:
    """
    Result of running detect_and_extract_pose() on one frame.

    Attributes
    ----------
    keypoints_norm : np.ndarray | None
        Normalised keypoints, shape (NUM_JOINTS, 3) → [x_norm, y_norm, conf].
        None when no person is detected above the confidence threshold.
        This array is what gets pushed into SequenceBuffer.
    keypoints_raw : np.ndarray | None
        Raw pixel keypoints, shape (NUM_JOINTS, 3) → [x_px, y_px, conf].
        Used to draw skeleton overlays on the original frame.
    bbox_xyxy : np.ndarray | None
        Bounding box of the selected person in pixel coords: [x1, y1, x2, y2].
    detection_conf : float
        Detection confidence of the selected person (0.0 if none detected).
    person_detected : bool
        True when a person above conf_threshold was found in the frame.
    """
    keypoints_norm:   Optional[np.ndarray] = field(default=None)
    keypoints_raw:    Optional[np.ndarray] = field(default=None)
    bbox_xyxy:        Optional[np.ndarray] = field(default=None)
    detection_conf:   float                = 0.0
    person_detected:  bool                 = False
    hip_observed:     bool                 = False
    hip_virtual:      bool                 = False
    lower_body_valid: bool                 = False
    pose_quality:     str                  = "none"
    tracked_low_conf: bool                 = False


# ════════════════════════════════════════════════════════════════════════════
# Core API
# ════════════════════════════════════════════════════════════════════════════

def _bbox_iou(box_a: Optional[np.ndarray], box_b: Optional[np.ndarray]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    if box_a is None or box_b is None:
        return 0.0
    a = np.asarray(box_a, dtype=np.float32).reshape(-1)
    b = np.asarray(box_b, dtype=np.float32).reshape(-1)
    if a.size < 4 or b.size < 4:
        return 0.0
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-6 else 0.0


def _bbox_center_close(
    box_a: Optional[np.ndarray],
    box_b: Optional[np.ndarray],
    frame_shape: tuple[int, ...],
) -> bool:
    """Check if two boxes remain within local motion range during rapid falls."""
    if box_a is None or box_b is None:
        return False
    a = np.asarray(box_a, dtype=np.float32).reshape(-1)
    b = np.asarray(box_b, dtype=np.float32).reshape(-1)
    if a.size < 4 or b.size < 4:
        return False
    acx, acy = 0.5 * (float(a[0]) + float(a[2])), 0.5 * (float(a[1]) + float(a[3]))
    bcx, bcy = 0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3]))
    dist = float(np.hypot(acx - bcx, acy - bcy))
    b_diag = float(np.hypot(float(b[2]) - float(b[0]), float(b[3]) - float(b[1])))
    frame_max = float(max(frame_shape[1], frame_shape[0])) if len(frame_shape) >= 2 else 640.0
    max_dist = max(0.15 * frame_max, 0.60 * b_diag)
    return dist <= max_dist


@dataclass
class SubjectPoseTracker:
    """Stateful Stage-1 hysteresis tracker for occluded/low-confidence subject continuity."""

    track_conf: float = TRACK_CONF_THRESHOLD
    max_lost_ticks: int = 5
    min_iou: float = 0.10
    last_bbox: Optional[np.ndarray] = None
    lost_ticks: int = 0
    active: bool = False

    def reset(self) -> None:
        """Clear tracked subject state on stream change or scene cut."""
        self.last_bbox = None
        self.lost_ticks = 0
        self.active = False

    def is_continuation_active(self) -> bool:
        """Return True when a prior subject track is recent enough for hysteresis continuation."""
        return (
            self.active
            and self.last_bbox is not None
            and self.lost_ticks <= self.max_lost_ticks
        )


def detect_and_extract_pose(
    frame_bgr: np.ndarray,
    model,
    conf_threshold: float = CONF_THRESHOLD,
    imgsz: int | None = None,
    tracker: Optional[SubjectPoseTracker] = None,
    track_conf_threshold: Optional[float] = None,
) -> PoseResult:
    """
    Run YOLO-Pose on one BGR frame and return a PoseResult.
    `imgsz` optionally overrides the YOLO inference size (default 640; lower = faster on CPU).

    Person selection: highest bounding-box confidence score.
    If confidence < conf_threshold → returns PoseResult with person_detected=False.

    This function's output format and normalisation EXACTLY match what
    01_data_preparation.ipynb produces:
      • Same COCO 17-keypoint order (via JOINT_NAMES in config.py)
      • Same hip-centred, torso-scaled normalisation (via skeleton_utils.normalise_keypoints)
      • Same confidence zeroing for low-confidence joints

    Parameters
    ----------
    frame_bgr     : BGR frame from cv2.VideoCapture or webcam
    model         : YOLO-Pose model loaded via load_pose_model()
    conf_threshold: minimum person detection confidence

    Returns
    -------
    PoseResult dataclass
    """
    result = PoseResult()

    def _on_miss() -> PoseResult:
        if tracker is not None:
            tracker.lost_ticks += 1
            if tracker.lost_ticks > tracker.max_lost_ticks:
                tracker.reset()
        return result

    # ── Run YOLO ─────────────────────────────────────────────────────────────
    if imgsz:
        yolo_results = model(frame_bgr, imgsz=int(imgsz), verbose=False)
    else:
        yolo_results = model(frame_bgr, verbose=False)

    if not yolo_results or yolo_results[0].keypoints is None:
        return _on_miss()

    r       = yolo_results[0]
    kp_data = r.keypoints
    boxes   = r.boxes

    if kp_data.xy is None or len(kp_data.xy) == 0:
        return _on_miss()

    # ── Person selection: Primary (>= conf_threshold) or Stage-1 Hysteresis ──
    if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
        confs     = boxes.conf.cpu().numpy()
        best_idx  = int(np.argmax(confs))
        best_conf = float(confs[best_idx])
    else:
        confs     = None
        best_idx  = 0
        best_conf = 1.0

    used_hysteresis = False
    eff_kp_conf = float(conf_threshold)

    if best_conf < conf_threshold:
        hysteresis_enabled = (tracker is not None) or (track_conf_threshold is not None)
        if not hysteresis_enabled:
            return _on_miss()

        raw_track = (
            float(track_conf_threshold)
            if track_conf_threshold is not None
            else (float(tracker.track_conf) if tracker is not None else float(TRACK_CONF_THRESHOLD))
        )
        eff_track_conf = min(raw_track, float(conf_threshold))
        if best_conf < eff_track_conf:
            return _on_miss()

        cand_order = np.argsort(-confs).tolist() if confs is not None else [0]
        selected_idx: Optional[int] = None
        selected_conf = 0.0

        # Pre-filter valid candidates >= eff_track_conf
        cands = []
        for idx in cand_order:
            cand_conf = float(confs[idx]) if confs is not None else 1.0
            if cand_conf < eff_track_conf:
                break
            kp_cand = kp_data.data[idx].cpu().numpy().astype(np.float32)
            if kp_cand.ndim != 2 or kp_cand.shape[0] == 0 or kp_cand.shape[1] < 3:
                continue
            cand_bbox = None
            if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > idx:
                cand_bbox = boxes.xyxy[idx].cpu().numpy().astype(np.float32)
            cands.append((int(idx), cand_conf, kp_cand, cand_bbox))

        # Path A: active track continuation (prefer IoU overlap over unanchored local motion)
        if tracker is not None and tracker.is_continuation_active():
            best_a_score: Optional[tuple[int, float, float]] = None
            for idx, cand_conf, kp_cand, cand_bbox in cands:
                if cand_bbox is None:
                    continue
                kp_confs = kp_cand[:, 2]
                n_kp_track = int(np.sum(kp_confs >= eff_track_conf))
                n_torso_track = (
                    int(np.sum(kp_cand[[5, 6, 11, 12], 2] >= eff_track_conf))
                    if kp_cand.shape[0] >= 13
                    else 0
                )
                if n_kp_track < 4 or n_torso_track < 1:
                    continue
                iou = _bbox_iou(cand_bbox, tracker.last_bbox)
                close = _bbox_center_close(cand_bbox, tracker.last_bbox, frame_bgr.shape)
                if iou >= tracker.min_iou or close:
                    score = (1 if iou >= tracker.min_iou else 0, iou, cand_conf)
                    if best_a_score is None or score > best_a_score:
                        best_a_score = score
                        selected_idx = idx
                        selected_conf = cand_conf
                        used_hysteresis = True

        # Path B: unanchored horizontal / mid-fall recovery with strong skeleton evidence
        if selected_idx is None:
            for idx, cand_conf, kp_cand, cand_bbox in cands:
                if cand_conf < max(eff_track_conf, 0.35) or cand_bbox is None:
                    continue
                bw = max(1.0, float(cand_bbox[2] - cand_bbox[0]))
                bh = max(1.0, float(cand_bbox[3] - cand_bbox[1]))
                if (bh / bw) > 1.25:
                    continue
                kp_confs = kp_cand[:, 2]
                n_kp_40 = int(np.sum(kp_confs >= 0.40))
                n_torso_35 = (
                    int(np.sum(kp_cand[[5, 6, 11, 12], 2] >= 0.35))
                    if kp_cand.shape[0] >= 13
                    else 0
                )
                if n_kp_40 >= 6 and n_torso_35 >= 2:
                    selected_idx = idx
                    selected_conf = cand_conf
                    used_hysteresis = True
                    break

        if selected_idx is None:
            return _on_miss()

        best_idx = selected_idx
        best_conf = selected_conf
        # Keep real keypoints at >= 0.30 so virtual hip sentinel (0.25) remains strictly distinct
        eff_kp_conf = max(0.30, min(float(conf_threshold), float(eff_track_conf)))

    # ── Extract raw keypoints ────────────────────────────────────────────────
    kp_raw = kp_data.data[best_idx].cpu().numpy().astype(np.float32)   # (17, 3)

    if kp_raw.ndim != 2 or kp_raw.shape[0] == 0 or kp_raw.shape[1] < 3:
        return _on_miss()

    # Zero-out low-confidence joints (same logic as skeleton_utils)
    kp_raw[kp_raw[:, 2] < eff_kp_conf, :] = 0.0

    # ── Bounding box ─────────────────────────────────────────────────────────
    bbox = None
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > best_idx:
        bbox = boxes.xyxy[best_idx].cpu().numpy().astype(np.float32)   # [x1,y1,x2,y2]

    # ── Normalise ─────────────────────────────────────────────────────────────
    kp_norm = normalise_keypoints(kp_raw)

    hip_obs = bool(kp_raw[11, 2] >= eff_kp_conf or kp_raw[12, 2] >= eff_kp_conf)
    hip_virt = bool(not hip_obs and kp_norm is not None)
    low_valid = bool(hip_obs and (kp_raw[13, 2] >= eff_kp_conf or kp_raw[14, 2] >= eff_kp_conf))
    ank_obs = bool(kp_raw[15, 2] >= eff_kp_conf or kp_raw[16, 2] >= eff_kp_conf)
    if low_valid and ank_obs:
        quality = "full"
    elif hip_obs:
        quality = "partial"
    elif hip_virt:
        quality = "upper_only"
    else:
        quality = "low"

    if tracker is not None:
        if bbox is not None:
            tracker.last_bbox = bbox.copy()
            tracker.lost_ticks = 0
            tracker.active = True
        else:
            tracker.lost_ticks += 1
            if tracker.lost_ticks > tracker.max_lost_ticks:
                tracker.reset()

    result.keypoints_raw    = kp_raw
    result.keypoints_norm   = kp_norm        # None if hips undetected & no shoulder fallback
    result.bbox_xyxy        = bbox
    result.detection_conf   = best_conf
    result.person_detected  = True
    result.hip_observed     = hip_obs
    result.hip_virtual      = hip_virt
    result.lower_body_valid = low_valid
    result.pose_quality     = quality
    result.tracked_low_conf = used_hysteresis

    return result


# ════════════════════════════════════════════════════════════════════════════
# Visualisation helper
# ════════════════════════════════════════════════════════════════════════════

# COCO skeleton connectivity for drawing
_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),                      # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),              # arms
    (5, 11), (6, 12), (11, 12),                            # torso
    (11, 13), (13, 15), (12, 14), (14, 16),                # legs
]


def draw_pose(
    frame_bgr: np.ndarray,
    result: PoseResult,
    point_color: tuple[int, int, int] = (0, 255, 0),
    line_color: tuple[int, int, int]  = (0, 180, 255),
    bbox_color: tuple[int, int, int]  = (255, 180, 0),
    radius: int = 4,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw skeleton keypoints and bounding box on a copy of `frame_bgr`.

    Uses raw pixel keypoints (not normalised) for rendering.

    Returns
    -------
    Annotated BGR frame (copy of input).
    """
    out = frame_bgr.copy()

    if not result.person_detected or result.keypoints_raw is None:
        return out

    kp = result.keypoints_raw   # (17, 3) → [x_px, y_px, conf]

    # Draw skeleton edges
    for a, b in _SKELETON_EDGES:
        if kp[a, 2] > 0.0 and kp[b, 2] > 0.0:
            pt_a = (int(kp[a, 0]), int(kp[a, 1]))
            pt_b = (int(kp[b, 0]), int(kp[b, 1]))
            cv2.line(out, pt_a, pt_b, line_color, thickness, cv2.LINE_AA)

    # Draw keypoint circles
    for i in range(len(kp)):
        if kp[i, 2] > 0.0:
            pt = (int(kp[i, 0]), int(kp[i, 1]))
            cv2.circle(out, pt, radius, point_color, -1, cv2.LINE_AA)

    # Draw bounding box
    if result.bbox_xyxy is not None:
        x1, y1, x2, y2 = result.bbox_xyxy.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), bbox_color, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"conf:{result.detection_conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, bbox_color, 1, cv2.LINE_AA,
        )

    return out
