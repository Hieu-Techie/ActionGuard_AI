"""
object_context.py — Stage-2B Conditional Hand-Object Context Detector.

Runs a lightweight COCO object detector (`models/yolo_obj_pretrained.pt`) on the
subject's upper-body / hand-head ROI **only** when Stage-2 action probabilities
or wrist-to-nose geometry suggest an `Eat` / `Drink` / hand-near-face candidate.

Safety contract:
  • Never invoked when `Fall` probability >= 0.40, and never alters `Fall` / RED paths.
  • Returns graceful defaults (`ObjectContextResult()`) if weights are missing or
    no ROI / anchors are available.
"""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np

from utils.config import YOLO_OBJ_MODEL_PATH
from utils.skeleton_utils import _get_yolo_class

log = logging.getLogger(__name__)

# Standard COCO-80 class IDs for Stage-2B context
DRINK_CLASS_IDS: dict[int, str] = {
    39: "bottle",
    40: "wine glass",
    41: "cup",
}

EAT_CLASS_IDS: dict[int, str] = {
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
}

DISTRACTOR_CLASS_IDS: dict[int, str] = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    63: "laptop",
    73: "book",
}

RELEVANT_OBJ_IDS: list[int] = sorted(
    set(DRINK_CLASS_IDS) | set(EAT_CLASS_IDS) | set(DISTRACTOR_CLASS_IDS)
)

_UPPER_ROI_JOINTS = (0, 1, 2, 5, 6, 7, 8, 9, 10)
_SHOULDER_DISTRACTOR_IDS = (24, 26)


@dataclass
class ObjectContextResult:
    """Per-frame Stage-2B detection summary around the hand-head ROI."""

    drink_score: float = 0.0
    eat_score: float = 0.0
    distractor_score: float = 0.0
    detected_items: list[str] = field(default_factory=list)
    roi_valid: bool = False


def load_object_model(model_path: str = YOLO_OBJ_MODEL_PATH) -> Optional[Any]:
    """
    Load a COCO YOLO object detector checkpoint from `model_path`.

    Returns None gracefully with a warning if the checkpoint file is missing
    so callers never attempt unintended network downloads at runtime.
    """
    if not Path(model_path).exists():
        log.warning("Stage-2B object detector checkpoint not found: %s (skipping Stage-2B)", model_path)
        return None
    YOLO = _get_yolo_class()
    model = YOLO(model_path)
    log.info("Loaded Stage-2B object context model from: %s", model_path)
    return model


def _is_valid_joint(kp_raw: np.ndarray, idx: int) -> bool:
    """Return True when joint `idx` has non-zero confidence and non-(0,0) coords."""
    if kp_raw is None or idx >= len(kp_raw):
        return False
    x, y, c = float(kp_raw[idx, 0]), float(kp_raw[idx, 1]), float(kp_raw[idx, 2])
    return c > 0.0 and not (x == 0.0 and y == 0.0)


def _point_to_box_dist(px: float, py: float, box: tuple[float, float, float, float]) -> float:
    """Euclidean distance from point `(px, py)` to axis-aligned `box = (x1, y1, x2, y2)`."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _compute_ref_scale(kp_raw: np.ndarray, bbox_xyxy: Optional[np.ndarray]) -> float:
    """Estimate shoulder-width pixel scale (clamped to >= 50.0 px)."""
    if _is_valid_joint(kp_raw, 5) and _is_valid_joint(kp_raw, 6):
        sh_w = float(np.linalg.norm(kp_raw[5, :2] - kp_raw[6, :2]))
        if sh_w > 20.0:
            return max(sh_w, 50.0)
    if bbox_xyxy is not None and len(bbox_xyxy) >= 4:
        bx1, _, bx2, _ = [float(v) for v in bbox_xyxy[:4]]
        return max(0.35 * max(0.0, bx2 - bx1), 50.0)
    return 80.0


def _compute_upper_roi(
    frame_shape: tuple[int, ...],
    kp_raw: np.ndarray,
    bbox_xyxy: Optional[np.ndarray],
    ref_scale: float,
) -> tuple[int, int, int, int, bool]:
    """Compute padded `[rx1, ry1, rx2, ry2]` ROI around head, shoulders, and hands."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    pts = [
        (float(kp_raw[j, 0]), float(kp_raw[j, 1]))
        for j in _UPPER_ROI_JOINTS
        if _is_valid_joint(kp_raw, j)
    ]
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    elif bbox_xyxy is not None and len(bbox_xyxy) >= 4:
        bx1, by1, bx2, by2 = [float(v) for v in bbox_xyxy[:4]]
        x1, x2 = bx1, bx2
        y1 = by1
        y2 = by1 + 0.75 * max(0.0, by2 - by1)
    else:
        return 0, 0, 0, 0, False

    pad = int(max(60.0, 0.75 * ref_scale))
    rx1 = max(0, int(math.floor(x1)) - pad)
    ry1 = max(0, int(math.floor(y1)) - pad)
    rx2 = min(w, int(math.ceil(x2)) + pad)
    ry2 = min(h, int(math.ceil(y2)) + pad)
    roi_valid = bool((rx2 - rx1) >= 140 and (ry2 - ry1) >= 140 and ref_scale >= 110.0)
    return rx1, ry1, rx2, ry2, roi_valid


def detect_hand_objects(
    frame_bgr: np.ndarray,
    pose_result: Any,
    obj_model: Any,
    conf: float = 0.15,
    imgsz: int = 320,
) -> ObjectContextResult:
    """Run COCO object detection on the cropped upper-body ROI and score nearby items."""
    if (
        frame_bgr is None
        or obj_model is None
        or pose_result is None
        or not getattr(pose_result, "person_detected", False)
        or getattr(pose_result, "keypoints_raw", None) is None
    ):
        return ObjectContextResult()

    kp_raw: np.ndarray = pose_result.keypoints_raw
    bbox_xyxy = getattr(pose_result, "bbox_xyxy", None)
    ref_scale = _compute_ref_scale(kp_raw, bbox_xyxy)
    rx1, ry1, rx2, ry2, roi_valid = _compute_upper_roi(
        frame_bgr.shape, kp_raw, bbox_xyxy, ref_scale
    )
    if rx2 <= rx1 or ry2 <= ry1:
        return ObjectContextResult()

    roi_bgr = frame_bgr[ry1:ry2, rx1:rx2]
    if roi_bgr.size == 0:
        return ObjectContextResult()

    results = obj_model(
        roi_bgr,
        imgsz=int(imgsz),
        conf=float(conf),
        classes=RELEVANT_OBJ_IDS,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return ObjectContextResult(roi_valid=roi_valid)

    # Feeding anchors: visible wrists (9, 10) and nose (0)
    wrist_pts: list[tuple[float, float]] = [
        (float(kp_raw[j, 0]), float(kp_raw[j, 1]))
        for j in (9, 10)
        if _is_valid_joint(kp_raw, j)
    ]
    nose_pt: Optional[tuple[float, float]] = (
        (float(kp_raw[0, 0]), float(kp_raw[0, 1])) if _is_valid_joint(kp_raw, 0) else None
    )
    feed_pts: list[tuple[float, float]] = wrist_pts + ([nose_pt] if nose_pt is not None else [])

    # Active/raised anchors for distractors (never penalize passive items down in lap)
    active_pts: list[tuple[float, float]] = []
    if nose_pt is not None:
        active_pts.append(nose_pt)
    sh_ys = [float(kp_raw[j, 1]) for j in (5, 6) if _is_valid_joint(kp_raw, j)]
    sh_mid_y = (sum(sh_ys) / len(sh_ys)) if sh_ys else None
    for j in (9, 10):
        if not _is_valid_joint(kp_raw, j):
            continue
        wx, wy = float(kp_raw[j, 0]), float(kp_raw[j, 1])
        near_nose = nose_pt is not None and math.hypot(wx - nose_pt[0], wy - nose_pt[1]) <= 1.05 * ref_scale
        above_sh = sh_mid_y is not None and wy <= sh_mid_y + 0.35 * ref_scale
        if near_nose or above_sh:
            active_pts.append((wx, wy))

    shoulder_pts: list[tuple[float, float]] = [
        (float(kp_raw[j, 0]), float(kp_raw[j, 1]))
        for j in (5, 6)
        if _is_valid_joint(kp_raw, j)
    ]

    boxes = results[0].boxes
    xyxy_arr = boxes.xyxy.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()

    drink_score = 0.0
    eat_score = 0.0
    distractor_score = 0.0
    detected_items: list[tuple[float, str]] = []

    for b_idx in range(len(xyxy_arr)):
        cls_id = int(cls_arr[b_idx])
        det_conf = float(conf_arr[b_idx])
        bx1, by1, bx2, by2 = [float(v) for v in xyxy_arr[b_idx][:4]]
        full_box = (bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1)

        if cls_id in DRINK_CLASS_IDS:
            # Drink vessels must be near mouth/nose (ignores tabletop cups & chest bowls)
            if nose_pt is not None:
                dn = _point_to_box_dist(nose_pt[0], nose_pt[1], full_box)
                if dn > 0.85 * ref_scale:
                    continue
                dw = min((_point_to_box_dist(px, py, full_box) for px, py in wrist_pts), default=dn)
                d_ref = max(dn, 0.5 * dw)
            else:
                if not wrist_pts:
                    continue
                d_ref = min(_point_to_box_dist(px, py, full_box) for px, py in wrist_pts)
                if d_ref > 0.90 * ref_scale:
                    continue
            w = (
                1.0
                if d_ref <= 0.45 * ref_scale
                else max(0.35, 1.0 - (d_ref - 0.45 * ref_scale) / (0.65 * ref_scale))
            )
            weighted = det_conf * w
            name = DRINK_CLASS_IDS[cls_id]
            drink_score = max(drink_score, weighted)
            detected_items.append((weighted, f"{name}:{det_conf:.2f}"))
        elif cls_id in EAT_CLASS_IDS:
            if not feed_pts:
                continue
            d_min = min(_point_to_box_dist(px, py, full_box) for px, py in feed_pts)
            if d_min > 1.25 * ref_scale:
                continue
            w = (
                1.0
                if d_min <= 0.50 * ref_scale
                else max(0.35, 1.0 - (d_min - 0.50 * ref_scale) / (0.90 * ref_scale))
            )
            weighted = det_conf * w
            name = EAT_CLASS_IDS[cls_id]
            eat_score = max(eat_score, weighted)
            detected_items.append((weighted, f"{name}:{det_conf:.2f}"))
        elif cls_id in DISTRACTOR_CLASS_IDS:
            anchors = active_pts + shoulder_pts if cls_id in _SHOULDER_DISTRACTOR_IDS else active_pts
            if not anchors:
                continue
            d_min = min(_point_to_box_dist(px, py, full_box) for px, py in anchors)
            if d_min > 0.95 * ref_scale:
                continue
            name = DISTRACTOR_CLASS_IDS[cls_id]
            distractor_score = max(distractor_score, det_conf)
            detected_items.append((det_conf, f"{name}:{det_conf:.2f}"))

    detected_items.sort(key=lambda item: item[0], reverse=True)
    return ObjectContextResult(
        drink_score=drink_score,
        eat_score=eat_score,
        distractor_score=distractor_score,
        detected_items=[tag for _, tag in detected_items],
        roi_valid=roi_valid,
    )


class ObjectObservationSummary(TypedDict):
    """Aggregated Stage-2B observation summary across recent ticks."""

    max_drink: float
    max_eat: float
    max_distractor: float
    n_checks: int
    roi_valid_count: int
    summary_tag: str


class HandObjectTracker:
    """Short sliding-window tracker aggregating Stage-2B detections across ticks."""

    def __init__(self, maxlen: int = 6) -> None:
        self._history: collections.deque[ObjectContextResult] = collections.deque(maxlen=maxlen)
        self._last_obs: Optional[ObjectContextResult] = None
        self._parity: int = 0
        self._idle_streak: int = 0

    def reset(self) -> None:
        self._history.clear()
        self._last_obs = None
        self._parity = 0
        self._idle_streak = 0

    def mark_inactive(self) -> None:
        self._idle_streak += 1
        if self._idle_streak > 2:
            self.reset()

    def observe(
        self,
        frame_bgr: np.ndarray,
        pose_result: Any,
        obj_model: Any,
        stride: int = 3,
    ) -> ObjectObservationSummary:
        self._idle_streak = 0
        if self._last_obs is None or (self._parity % max(1, int(stride)) == 0):
            self._last_obs = detect_hand_objects(frame_bgr, pose_result, obj_model)
        self._parity += 1
        self._history.append(self._last_obs)

        max_drink = max(r.drink_score for r in self._history)
        max_eat = max(r.eat_score for r in self._history)
        max_distractor = max(r.distractor_score for r in self._history)
        n_checks = len(self._history)
        roi_valid_count = sum(1 for r in self._history if r.roi_valid)

        best_tag = ""
        best_val = 0.0
        for r in reversed(self._history):
            top_item = r.detected_items[0] if r.detected_items else ""
            top_sc = max(r.drink_score, r.eat_score, r.distractor_score)
            if top_item and top_sc >= best_val:
                best_val = top_sc
                best_tag = top_item

        if not best_tag and roi_valid_count >= 3:
            best_tag = "no_vessel"

        return {
            "max_drink": max_drink,
            "max_eat": max_eat,
            "max_distractor": max_distractor,
            "n_checks": n_checks,
            "roi_valid_count": roi_valid_count,
            "summary_tag": best_tag,
        }
