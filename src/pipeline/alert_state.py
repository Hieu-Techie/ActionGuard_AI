"""
alert_state.py — AlertStateMachine for multi-frame alert level management.

Implements spec.md Section 7 with three independent consecutive-frame counters
(`_warning_count`, `_critical_count`, `_recovery_count`) so streaks toward
YELLOW, RED, and GREEN can never bleed into one another:
  • Maps action predictions (8 classes) to GREEN / YELLOW / RED via
    LABEL_TO_ALERT_COLOR_MAPPING from config.py.
  • Enforces confidence thresholds (CONF_THRESHOLD, WARNING_THRESHOLD,
    CRITICAL_THRESHOLD, INSTANT_RED_THRESHOLD) and strictly consecutive-frame
    hysteresis counters (N_WARN_FRAMES, N_CRITICAL_FRAMES, N_RECOVERY_FRAMES).
  • Resets all consecutive counters to 0 whenever an uncertain, No_Person, or
    sub-threshold frame interrupts a streak.
  • Latches RED until explicitly cleared via .acknowledge() AND followed by
    N_RECOVERY_FRAMES strictly consecutive normal (GREEN) predictions.
  • Automatically recovers YELLOW → GREEN after N_RECOVERY_FRAMES strictly
    consecutive normal (GREEN) predictions.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
import sys
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from utils.config import (
    LABEL_TO_ALERT_COLOR_MAPPING,
    CONF_THRESHOLD,
    WARNING_THRESHOLD,
    CRITICAL_THRESHOLD,
    INSTANT_RED_THRESHOLD,
    N_WARN_FRAMES,
    N_CRITICAL_FRAMES,
    N_RECOVERY_FRAMES,
)
from pipeline.action_classifier import ActionPrediction

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Alert level enum (spec.md Section 7)
# ════════════════════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    GREEN  = "Normal"
    YELLOW = "Warning"
    RED    = "Critical - Fall"


_COLOR_STR_TO_LEVEL = {
    "GREEN":  AlertLevel.GREEN,
    "YELLOW": AlertLevel.YELLOW,
    "RED":    AlertLevel.RED,
}


def _label_to_alert_level(label: str) -> AlertLevel:
    """Map a CLASS_NAMES label to its AlertLevel via config mapping."""
    color_str = LABEL_TO_ALERT_COLOR_MAPPING.get(label, "GREEN")
    return _COLOR_STR_TO_LEVEL.get(color_str, AlertLevel.GREEN)


# ════════════════════════════════════════════════════════════════════════════
# State machine
# ════════════════════════════════════════════════════════════════════════════

class AlertStateMachine:
    """
    Multi-frame hysteresis alert state machine (spec.md Section 7).

    Maintains three strictly separated counters:
      • `_warning_count`  : consecutive valid YELLOW frames (`GREEN → YELLOW`)
      • `_critical_count` : consecutive valid RED frames (`GREEN/YELLOW → RED`)
      • `_recovery_count` : consecutive valid GREEN frames (`YELLOW/ack-RED → GREEN`)
    """

    def __init__(
        self,
        n_warn:                int   = N_WARN_FRAMES,
        n_critical:            int   = N_CRITICAL_FRAMES,
        n_recovery:            int   = N_RECOVERY_FRAMES,
        conf_threshold:        float = CONF_THRESHOLD,
        warning_threshold:     float = WARNING_THRESHOLD,
        critical_threshold:    float = CRITICAL_THRESHOLD,
        instant_red_threshold: float = INSTANT_RED_THRESHOLD,
    ) -> None:
        self._n_warn     = n_warn
        self._n_critical = n_critical
        self._n_recovery = n_recovery

        self._conf_threshold        = conf_threshold
        self._warning_threshold     = warning_threshold
        self._critical_threshold    = critical_threshold
        self._instant_red_threshold = instant_red_threshold

        # Public state
        self.level: AlertLevel = AlertLevel.GREEN

        # Independent consecutive-frame counters
        self._warning_count:  int = 0
        self._critical_count: int = 0
        self._recovery_count: int = 0

        # RED latch: False when RED is active and unacknowledged
        self._red_acknowledged: bool = True

    @property
    def is_red_latched(self) -> bool:
        """True when in RED state and waiting for operator acknowledgement."""
        return self.level == AlertLevel.RED and not self._red_acknowledged

    @property
    def warning_count(self) -> int:
        return self._warning_count

    @property
    def critical_count(self) -> int:
        return self._critical_count

    @property
    def recovery_count(self) -> int:
        return self._recovery_count

    @property
    def pending_count(self) -> int:
        """Active non-zero streak counter (for inspection/tests)."""
        return max(self._warning_count, self._critical_count, self._recovery_count)

    def _reset_all_counters(self) -> None:
        self._warning_count  = 0
        self._critical_count = 0
        self._recovery_count = 0

    # ── Core update ──────────────────────────────────────────────────────────

    def update(self, prediction: ActionPrediction) -> AlertLevel:
        """
        Process one new ActionPrediction and return the updated AlertLevel.
        """
        # 1. Uncertain, missing, or low-quality pose ("No_Person" / "Unknown" /
        #    "Low_Pose_Quality" / low confidence):
        #    Breaks all ongoing consecutive streaks and keeps current level.
        if (
            prediction.label in ("Unknown", "No_Person", "Low_Pose_Quality", "")
            or prediction.confidence < self._conf_threshold
        ):
            self._reset_all_counters()
            return self.level

        # Unreliable / upper-only Idle must never recover YELLOW or acknowledged RED
        if (
            prediction.label == "Idle"
            and getattr(prediction, "pose_quality", "full") in ("upper_only", "low", "none")
            and self.level in (AlertLevel.RED, AlertLevel.YELLOW)
        ):
            self._reset_all_counters()
            return self.level

        raw_level = _label_to_alert_level(prediction.label)
        conf = prediction.confidence

        # ── Case A: RED-mapped prediction (Fall) ─────────────────────────────
        if raw_level == AlertLevel.RED:
            # A RED frame immediately breaks any YELLOW or GREEN recovery streak
            self._warning_count  = 0
            self._recovery_count = 0

            # Instant RED override (single-frame high-confidence Fall)
            if conf >= self._instant_red_threshold:
                if self.level != AlertLevel.RED:
                    old_level = self.level
                    self.level = AlertLevel.RED
                    log.info(
                        "Instant RED transition: %s → RED (label=%s, conf=%.2f >= %.2f)",
                        old_level.name, prediction.label, conf, self._instant_red_threshold,
                    )
                self._red_acknowledged = False
                self._critical_count = 0
                return self.level

            # Below CRITICAL_THRESHOLD: breaks critical streak
            if conf < self._critical_threshold:
                self._critical_count = 0
                return self.level

            # Valid critical frame (conf in [critical_threshold, instant_red_threshold))
            self._critical_count += 1
            if self._critical_count >= self._n_critical:
                if self.level != AlertLevel.RED:
                    old_level = self.level
                    self.level = AlertLevel.RED
                    log.info(
                        "Alert transition: %s → RED (label=%s, conf=%.2f, model=%s)",
                        old_level.name, prediction.label, conf, prediction.model_type,
                    )
                self._red_acknowledged = False
                self._critical_count = 0
            return self.level

        # ── Case B: YELLOW-mapped prediction (Crouching / Lying) ─────────────
        if raw_level == AlertLevel.YELLOW:
            # A YELLOW frame immediately breaks any RED escalation or GREEN recovery streak
            self._critical_count = 0
            self._recovery_count = 0

            if conf < self._warning_threshold:
                self._warning_count = 0
                return self.level

            # Never downgrade RED (latched or acknowledged) to YELLOW
            if self.level == AlertLevel.RED:
                self._warning_count = 0
                return self.level

            if self.level == AlertLevel.GREEN:
                self._warning_count += 1
                if self._warning_count >= self._n_warn:
                    self.level = AlertLevel.YELLOW
                    self._warning_count = 0
                    log.info(
                        "Alert transition: GREEN → YELLOW (label=%s, conf=%.2f, model=%s)",
                        prediction.label, conf, prediction.model_type,
                    )
            return self.level

        # ── Case C: GREEN-mapped prediction (Walk / Eat / Drink / Sit / Stand)
        self._warning_count  = 0
        self._critical_count = 0

        # If RED and not yet acknowledged by operator, stay latched at RED
        if self.level == AlertLevel.RED and not self._red_acknowledged:
            self._recovery_count = 0
            return self.level

        # If YELLOW or acknowledged RED, count consecutive GREEN frames toward recovery
        if self.level in (AlertLevel.YELLOW, AlertLevel.RED):
            self._recovery_count += 1
            if self._recovery_count >= self._n_recovery:
                old_level = self.level
                self.level = AlertLevel.GREEN
                self._recovery_count = 0
                log.info(
                    "Alert recovery: %s → GREEN (label=%s, conf=%.2f, model=%s)",
                    old_level.name, prediction.label, conf, prediction.model_type,
                )

        return self.level

    # ── Acknowledge ──────────────────────────────────────────────────────────

    def acknowledge(self) -> None:
        """
        Explicitly unlock the RED latch.

        After calling this, the state machine will transition RED → GREEN
        once `n_recovery` strictly consecutive GREEN-mapped predictions arrive.
        """
        if self.level == AlertLevel.RED:
            self._red_acknowledged = True
            self._reset_all_counters()
            log.info("RED alert acknowledged by operator; awaiting recovery frames.")

    # ── Reset & Runtime Threshold Updates ────────────────────────────────────

    def reset(self) -> None:
        """Reset state to GREEN (e.g., when video source changes)."""
        self.level = AlertLevel.GREEN
        self._reset_all_counters()
        self._red_acknowledged = True

    def set_thresholds(
        self,
        *,
        conf_threshold:        Optional[float] = None,
        warning_threshold:     Optional[float] = None,
        critical_threshold:    Optional[float] = None,
        instant_red_threshold: Optional[float] = None,
        n_warn:                Optional[int]   = None,
        n_critical:            Optional[int]   = None,
        n_recovery:            Optional[int]   = None,
    ) -> None:
        """
        Update runtime thresholds on this state-machine instance without
        resetting current alert level, latch state, or frame counters, and
        without mutating global constants in `utils.config`.
        """
        if conf_threshold is not None:
            self._conf_threshold = float(conf_threshold)
        if warning_threshold is not None:
            self._warning_threshold = float(warning_threshold)
        if critical_threshold is not None:
            self._critical_threshold = float(critical_threshold)
        if instant_red_threshold is not None:
            self._instant_red_threshold = float(instant_red_threshold)
        if n_warn is not None:
            self._n_warn = max(1, int(n_warn))
        if n_critical is not None:
            self._n_critical = max(1, int(n_critical))
        if n_recovery is not None:
            self._n_recovery = max(1, int(n_recovery))
