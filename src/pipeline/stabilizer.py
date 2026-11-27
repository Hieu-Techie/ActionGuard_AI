"""
stabilizer.py — Inference stabilizer for in-the-wild videos (Milestone 5A+).

Wraps the Stage-2 probability output with:
  1. Pose-quality assessment & physical motion-energy / geometry gating (enabled
     when `idle_gate > 0.0`):
       - Separates "Idle" (valid upright pose with no active motion) from
         "Low_Pose_Quality" (missing shoulders and missing observed hips).
       - Compensates lower-body occlusion (missing knees/ankles) using hybrid
         ST-GCN + LSTM probability blending and logs fallback metadata on
         ActionPrediction (`fallback_used`, `fallback_model`, `pose_quality`).
       - Guards upright postures (`sh_y_mean > 0.70`, `sin_theta < 0.35`) against
         false Lying predictions.
       - Restricts knee-transition checks to frames with observed hips & knees
         (`lower_body_valid`), never virtual hips.
       - Applies Fast-Attack Fall Passthrough so EMA never dilutes sudden Fall spikes.
  2. EMA smoothing over consecutive windows (reduces single-window label flicker).
  3. "Idle" gating: when the smoothed top-1 probability is below `idle_gate`,
     or physical motion/geometry indicates stationary sitting/standing,
     the output label is "Idle" (GREEN in the alert machine).
  4. Ensemble support via `pipeline.action_classifier.predict_probs`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from pipeline.action_classifier import ActionPrediction, predict_probs
from pipeline.object_context import HandObjectTracker
from pipeline.sequence_buffer import SequenceBuffer
from utils.config import CLASS_NAMES

log = logging.getLogger(__name__)

IDLE_LABEL = "Idle"
LOW_POSE_QUALITY_LABEL = "Low_Pose_Quality"

# Canonical COCO-17 index constants matching utils.config.CLASS_NAMES / JOINT_NAMES
_IDX_WALK = 0
_IDX_EAT = 1
_IDX_DRINK = 2
_IDX_FALL = 3
_IDX_LYING = 4
_IDX_SIT_DOWN = 5
_IDX_STAND_UP = 6
_IDX_CROUCHING = 7

_CORE_JOINTS: list[int] = [0, 5, 6, 11, 12, 13, 14, 15, 16]
_WRIST_JOINTS: list[int] = [9, 10]
_LEG_JOINTS: list[int] = [13, 14, 15, 16]


def _extract_window_kinematics(arr: np.ndarray) -> dict[str, float]:
    """Compute pose-quality and physical motion/geometry descriptors for a (T, 17, 3) window."""
    vis = (arr[:, :, 2] > 0).astype(np.float32)
    sh_vis = float((arr[:, 5:7, 2] > 0).mean())
    hip_obs_ratio = float(((arr[:, 11, 2] >= 0.30) | (arr[:, 12, 2] >= 0.30)).mean())
    hip_virt_ratio = float(((arr[:, 11, 2] > 0.0) & (arr[:, 11, 2] < 0.30)).mean())
    kn_vis = float((arr[:, 13:15, 2] >= 0.30).mean())
    ank_vis = float((arr[:, 15:17, 2] >= 0.30).mean())
    lower_body_valid = 1.0 if (hip_obs_ratio >= 0.50 and kn_vis >= 0.50) else 0.0

    vp = vis[1:] * vis[:-1]
    diffs = np.linalg.norm(arr[1:, :, :2] - arr[:-1, :, :2], axis=-1) * vp
    all_e = float(diffs.sum() / (vp.sum() + 1e-6))
    core_e = float(diffs[:, _CORE_JOINTS].sum() / (vp[:, _CORE_JOINTS].sum() + 1e-6))
    wr_e = float(diffs[:, _WRIST_JOINTS].sum() / (vp[:, _WRIST_JOINTS].sum() + 1e-6))
    leg_e = float(diffs[:, _LEG_JOINTS].sum() / (vp[:, _LEG_JOINTS].sum() + 1e-6))

    sh_x = 0.5 * (arr[:, 5, 0] + arr[:, 6, 0])
    sh_y = 0.5 * (arr[:, 5, 1] + arr[:, 6, 1])
    sh_y_mean = float(np.mean(sh_y))
    sin_theta = float(np.mean(np.abs(sh_x) / (np.sqrt(sh_x ** 2 + sh_y ** 2) + 1e-6)))

    d_l: list[float] = []
    d_r: list[float] = []
    wy_l: list[float] = []
    wy_r: list[float] = []
    wel_l: list[float] = []
    wel_r: list[float] = []
    for t in range(len(arr)):
        if arr[t, 9, 2] > 0:
            wy_l.append(float(arr[t, 9, 1]))
            if arr[t, 0, 2] > 0:
                d_l.append(float(np.linalg.norm(arr[t, 9, :2] - arr[t, 0, :2])))
            if arr[t, 7, 2] > 0:
                wel_l.append(float(arr[t, 9, 1] - arr[t, 7, 1]))
        if arr[t, 10, 2] > 0:
            wy_r.append(float(arr[t, 10, 1]))
            if arr[t, 0, 2] > 0:
                d_r.append(float(np.linalg.norm(arr[t, 10, :2] - arr[t, 0, :2])))
            if arr[t, 8, 2] > 0:
                wel_r.append(float(arr[t, 10, 1] - arr[t, 8, 1]))

    dl_m = float(np.mean(d_l)) if d_l else 9.9
    dr_m = float(np.mean(d_r)) if d_r else 9.9
    d_wn = min(dl_m, dr_m)
    max_wy = max(float(np.mean(wy_l)) if wy_l else -9.9, float(np.mean(wy_r)) if wy_r else -9.9)
    max_wel = max(float(np.mean(wel_l)) if wel_l else -9.9, float(np.mean(wel_r)) if wel_r else -9.9)

    kn_y_t: list[float] = []
    for t in range(len(arr)):
        ys = [float(arr[t, j, 1]) for j in (13, 14) if arr[t, j, 2] >= 0.30]
        if ys:
            kn_y_t.append(sum(ys) / len(ys))
    kn_y_mean = float(np.mean(kn_y_t)) if kn_y_t else -0.70
    kn_y_span = (max(kn_y_t) - min(kn_y_t)) if len(kn_y_t) >= 4 else 0.0
    kn_y_delta = (kn_y_t[-1] - kn_y_t[0]) if len(kn_y_t) >= 4 else 0.0

    # Late-window posture check (last ~0.6s) to catch clips starting seated before lying
    late = arr[-min(6, len(arr)):]
    late_sh_x = 0.5 * (late[:, 5, 0] + late[:, 6, 0])
    late_sh_y = 0.5 * (late[:, 5, 1] + late[:, 6, 1])
    late_sh_y_mean = float(np.mean(late_sh_y))
    late_sin_theta = float(
        np.mean(np.abs(late_sh_x) / (np.sqrt(late_sh_x ** 2 + late_sh_y ** 2) + 1e-6))
    )
    vis_late = late[:, :, 2] >= 0.30
    if bool(np.any(vis_late)):
        ys_l = late[:, :, 1][vis_late]
        xs_l = late[:, :, 0][vis_late]
        late_yx_ratio = float((np.max(ys_l) - np.min(ys_l)) / (np.max(xs_l) - np.min(xs_l) + 1e-4))
    else:
        late_yx_ratio = 1.0

    return {
        "sh_vis": sh_vis,
        "hip_obs_ratio": hip_obs_ratio,
        "hip_virt_ratio": hip_virt_ratio,
        "lower_body_valid": lower_body_valid,
        "kn_vis": kn_vis,
        "ank_vis": ank_vis,
        "all_e": all_e,
        "core_e": core_e,
        "wr_e": wr_e,
        "leg_e": leg_e,
        "sh_y_mean": sh_y_mean,
        "sin_theta": sin_theta,
        "dl_m": dl_m,
        "dr_m": dr_m,
        "d_wn": d_wn,
        "max_wy": max_wy,
        "max_wel": max_wel,
        "kn_y_mean": kn_y_mean,
        "kn_y_span": kn_y_span,
        "kn_y_delta": kn_y_delta,
        "late_sh_y_mean": late_sh_y_mean,
        "late_sin_theta": late_sin_theta,
        "late_yx_ratio": late_yx_ratio,
    }


def _assess_window_quality(kin: dict[str, float]) -> str:
    """Classify window pose quality into 'full', 'partial', 'upper_only', or 'low'."""
    if kin["sh_vis"] < 0.50 and kin["hip_obs_ratio"] < 0.50:
        return "low"
    if kin["lower_body_valid"] > 0.5 and kin["ank_vis"] >= 0.35:
        return "full"
    if kin["hip_obs_ratio"] < 0.50:
        return "upper_only"
    return "partial"


def _eat_drink_flags(kin: dict[str, float]) -> tuple[bool, bool]:
    """Return `(is_high_drink_hold, is_eat_drink_invalid)` for the window."""
    d_wn = kin["d_wn"]
    dl_m = kin["dl_m"]
    dr_m = kin["dr_m"]
    is_high_drink_hold = (
        d_wn <= 0.68
        and kin["max_wy"] >= kin["sh_y_mean"] - 0.15
        and kin["max_wel"] >= 0.20
    )
    is_seated_backrest_lean = (
        kin["lower_body_valid"] > 0.5
        and kin["kn_y_mean"] > -0.45
        and (
            (
                d_wn > 0.66
                and max(dl_m, dr_m) > 1.05
                and abs(dl_m - dr_m) > 0.30
                and kin["wr_e"] < 0.065
            )
            or (
                kin["max_wy"] < kin["sh_y_mean"] - 0.22
                and kin["max_wel"] < 0.18
                and kin["wr_e"] < 0.026
                and abs(dl_m - dr_m) > 0.22
            )
        )
    )
    is_static_far_hands = (8.9 > d_wn > 0.82 and kin["wr_e"] < 0.065) or (
        not is_high_drink_hold and kin["wr_e"] < 0.016 and kin["all_e"] < 0.018
    )
    return is_high_drink_hold, (is_static_far_hands or is_seated_backrest_lean)


def _apply_physical_priors(
    probs: np.ndarray,
    kin: dict[str, float],
    buffer: SequenceBuffer,
    models: dict[str, Any],
    active_model: str,
) -> tuple[np.ndarray, bool, str]:
    """
    Adjust raw softmax probabilities using physical pose quality and kinematics.

    Returns `(adjusted_probs, fallback_used, fallback_model)`.
    """
    out = probs.copy()
    if out.shape[0] < 8:
        return out, False, ""

    fallback_used = False
    fallback_model = ""
    raw_fall_p = float(out[_IDX_FALL])
    top_raw = int(np.argmax(out))

    # 1. Lower-body occlusion compensation (hybrid ST-GCN + LSTM fallback when
    #    ankles and knees are missing so v3.1-C does not hallucinate Sit_Down / Stand_Up).
    if kin["ank_vis"] < 0.35 and kin["kn_vis"] < 0.50:
        out[_IDX_SIT_DOWN] *= 0.05
        out[_IDX_STAND_UP] *= 0.05
        out[_IDX_CROUCHING] *= 0.10
        if (
            top_raw in (_IDX_SIT_DOWN, _IDX_STAND_UP, _IDX_CROUCHING)
            and "lstm" in models
            and active_model == "stgcn"
        ):
            pl = predict_probs(buffer, models, "lstm")
            if pl is not None and pl.shape == out.shape == (len(CLASS_NAMES),):
                out = 0.40 * out + 0.60 * pl
                out[_IDX_SIT_DOWN] *= 0.10
                out[_IDX_STAND_UP] *= 0.10
                out[_IDX_CROUCHING] *= 0.10
                fallback_used = True
                fallback_model = "lstm"
                log.debug(
                    "Hybrid fallback activated: primary=stgcn, fallback=lstm, raw_top=%s",
                    CLASS_NAMES[top_raw],
                )

    # 2. Upright / seated posture guard against false Lying when lower body is
    #    occluded OR when the subject at the end of the window is still clearly
    #    sitting/standing upright (even with ankles visible); preserves
    #    overhead/reclining Lying clips and only damps weak static Fall spikes.
    if kin["sh_y_mean"] > 0.70 and kin["sin_theta"] < 0.35:
        still_upright_end = (
            kin["late_sh_y_mean"] > 0.68
            and kin["late_sin_theta"] < 0.38
            and kin["late_yx_ratio"] > 1.20
        )
        if kin["ank_vis"] < 0.35 or still_upright_end:
            out[_IDX_LYING] *= 0.05
        if raw_fall_p < 0.70 and kin["core_e"] < 0.045:
            out[_IDX_FALL] *= 0.10

    # 3. Seated / standing transition guard (observed hips + knees only, never virtual hips)
    if kin["lower_body_valid"] > 0.5:
        if kin["kn_y_span"] < 0.12 and kin["core_e"] < 0.028:
            out[_IDX_SIT_DOWN] *= 0.10
            out[_IDX_STAND_UP] *= 0.10
        elif kin["kn_y_delta"] > 0.08:
            out[_IDX_STAND_UP] *= 0.15
        elif kin["kn_y_delta"] < -0.08:
            out[_IDX_SIT_DOWN] *= 0.15

    # 4. Walking upright with backpack strap / one arm lowered (require d_wn > 0.62
    #    so close-to-mouth standing Drink clips are never damped).
    if (
        kin["lower_body_valid"] > 0.5
        and kin["sh_y_mean"] > 0.70
        and kin["sin_theta"] < 0.35
        and kin["leg_e"] >= 0.024
        and kin["d_wn"] > 0.62
        and max(kin["dl_m"], kin["dr_m"]) > 1.30
    ):
        out[_IDX_EAT] *= 0.15
        out[_IDX_DRINK] *= 0.15

    # 5. Hand-to-mouth proximity & wrist-elevation gate for Eat / Drink
    #    (Do not renormalize afterward so damped mass falls naturally into Idle.)
    _, eat_drink_invalid = _eat_drink_flags(kin)
    if eat_drink_invalid:
        out[_IDX_EAT] *= 0.20
        out[_IDX_DRINK] *= 0.20

    return out, fallback_used, fallback_model


class PredictionStabilizer:
    def __init__(self) -> None:
        self._ema: Optional[np.ndarray] = None
        self._obj_tracker = HandObjectTracker()

    def reset(self) -> None:
        self._ema = None
        self._obj_tracker.reset()

    def _apply_object_context(
        self,
        probs: np.ndarray,
        kin: dict[str, float],
        *,
        frame_bgr: Optional[np.ndarray],
        pose_result: Optional[Any],
        object_model: Optional[Any],
        enable_object_context: bool,
    ) -> tuple[np.ndarray, str]:
        """Run conditional Stage-2B hand-object detection and fuse with action probs."""
        if (
            not enable_object_context
            or object_model is None
            or frame_bgr is None
            or pose_result is None
            or float(probs[_IDX_FALL]) >= 0.40
        ):
            self._obj_tracker.mark_inactive()
            return probs, ""

        _, eat_drink_invalid = _eat_drink_flags(kin)
        if eat_drink_invalid or (
            max(float(probs[_IDX_EAT]), float(probs[_IDX_DRINK])) < 0.20
            and kin["d_wn"] >= 0.80
        ):
            self._obj_tracker.mark_inactive()
            return probs, ""

        ctx = self._obj_tracker.observe(
            frame_bgr, pose_result, object_model, stride=3
        )
        d_sc = float(ctx["max_drink"])
        e_sc = float(ctx["max_eat"])
        x_sc = float(ctx["max_distractor"])
        object_context = str(ctx["summary_tag"])
        out = probs.copy()

        # Rule 2B-1: Drink vessel confirmed (cup / bottle / wine glass)
        if d_sc >= 0.20 and d_sc >= e_sc + 0.06:
            out[_IDX_DRINK] = min(
                1.0, float(out[_IDX_DRINK]) + 0.75 * float(out[_IDX_EAT])
            )
            out[_IDX_EAT] *= 0.20
        # Rule 2B-2: Eat utensil / bowl / food confirmed
        elif e_sc >= 0.20 and e_sc >= d_sc + 0.06:
            out[_IDX_EAT] = min(
                1.0, float(out[_IDX_EAT]) + 0.75 * float(out[_IDX_DRINK])
            )
            out[_IDX_DRINK] *= 0.20
        # Rule 2B-3: No feeding vessel / utensil detected
        elif d_sc < 0.16 and e_sc < 0.16:
            if x_sc >= 0.22:
                out[_IDX_DRINK] *= 0.18
                out[_IDX_EAT] *= 0.22
            elif int(ctx["roi_valid_count"]) >= 3:
                out[_IDX_DRINK] *= 0.25
                is_bilateral_hands = (
                    kin["dl_m"] < 0.88
                    and kin["dr_m"] < 0.88
                    and abs(kin["dl_m"] - kin["dr_m"]) <= 0.18
                )
                if not is_bilateral_hands and (
                    kin["wr_e"] < 0.032
                    or kin["leg_e"] >= 0.020
                    or float(out[_IDX_LYING]) >= 0.15
                ):
                    out[_IDX_EAT] *= 0.25

        return out, object_context

    def step(
        self,
        buffer: SequenceBuffer,
        models: dict[str, Any],
        active_model: str,
        ema_strength: float = 0.5,
        idle_gate: float = 0.45,
        *,
        frame_bgr: Optional[np.ndarray] = None,
        pose_result: Optional[Any] = None,
        object_model: Optional[Any] = None,
        enable_object_context: bool = True,
    ) -> ActionPrediction:
        """Classify the current window and return a stabilized ActionPrediction."""
        if not buffer.is_ready():
            self._ema = None
            self._obj_tracker.reset()
            return ActionPrediction(model_type=active_model, pose_quality="none")
        if not buffer.has_valid_subject():
            self._ema = None
            self._obj_tracker.reset()
            return ActionPrediction(
                label="No_Person",
                label_idx=-1,
                confidence=0.0,
                model_type=active_model,
                pose_quality="none",
            )

        probs = predict_probs(buffer, models, active_model)
        if probs is None:
            self._obj_tracker.reset()
            return ActionPrediction(model_type=active_model, pose_quality="none")

        kin: Optional[dict[str, float]] = None
        pose_quality = "full"
        fallback_used = False
        fallback_model = ""
        object_context = ""

        if idle_gate > 0.0:
            arr = np.stack(list(buffer._buffer), axis=0)
            kin = _extract_window_kinematics(arr)
            pose_quality = _assess_window_quality(kin)
            if pose_quality == "low":
                self._ema = None
                self._obj_tracker.reset()
                return ActionPrediction(
                    label=LOW_POSE_QUALITY_LABEL,
                    label_idx=-1,
                    confidence=0.0,
                    model_type=active_model,
                    pose_quality="low",
                )
            probs, fallback_used, fallback_model = _apply_physical_priors(
                probs, kin, buffer, models, active_model
            )
            probs, object_context = self._apply_object_context(
                probs,
                kin,
                frame_bgr=frame_bgr,
                pose_result=pose_result,
                object_model=object_model,
                enable_object_context=enable_object_context,
            )
        else:
            self._obj_tracker.mark_inactive()

        a = float(min(max(ema_strength, 0.0), 0.95))
        if self._ema is None or a == 0.0 or self._ema.shape != probs.shape:
            ema: np.ndarray = probs.copy()
        else:
            ema = a * self._ema + (1.0 - a) * probs
            # Fast-Attack Fall Passthrough: never let EMA dilute a critical Fall spike
            if idle_gate > 0.0 and len(ema) > _IDX_FALL and probs[_IDX_FALL] >= 0.70:
                ema[_IDX_FALL] = max(float(ema[_IDX_FALL]), float(probs[_IDX_FALL]))
        self._ema = ema

        idx = int(np.argmax(ema))
        conf = float(ema[idx])
        if conf < idle_gate:
            return ActionPrediction(
                label=IDLE_LABEL,
                label_idx=-1,
                confidence=conf,
                model_type=active_model,
                pose_quality=pose_quality,
                fallback_used=fallback_used,
                fallback_model=fallback_model,
                object_context=object_context,
            )

        label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "Unknown"
        if kin is not None and idle_gate > 0.0:
            _, eat_drink_invalid = _eat_drink_flags(kin)
            if label in ("Walk", "Sit_Down", "Stand_Up") and kin["all_e"] < 0.014:
                return ActionPrediction(
                    label=IDLE_LABEL,
                    label_idx=-1,
                    confidence=conf,
                    model_type=active_model,
                    pose_quality=pose_quality,
                    fallback_used=fallback_used,
                    fallback_model=fallback_model,
                    object_context=object_context,
                )
            if label in ("Eat", "Drink") and eat_drink_invalid:
                return ActionPrediction(
                    label=IDLE_LABEL,
                    label_idx=-1,
                    confidence=conf,
                    model_type=active_model,
                    pose_quality=pose_quality,
                    fallback_used=fallback_used,
                    fallback_model=fallback_model,
                    object_context=object_context,
                )

        return ActionPrediction(
            label=label,
            label_idx=idx,
            confidence=conf,
            model_type=active_model,
            pose_quality=pose_quality,
            fallback_used=fallback_used,
            fallback_model=fallback_model,
            object_context=object_context,
        )
