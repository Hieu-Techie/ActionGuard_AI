"""
tests/test_pipeline_units.py — Official unit, equivalence & integration test suite.

Covers:
  Unit & Equivalence Tests (1–9):
    1. LABEL_TO_ALERT_COLOR_MAPPING covers all 8 CLASS_NAMES.
    2. Independent consecutive-frame counters in AlertStateMachine
       (_warning_count, _critical_count, _recovery_count).
    3. Interleaved No_Person / Unknown resets warning_count to 0.
    4. Interleaved Unknown resets critical_count to 0.
    5. YELLOW -> RED transition resets recovery_count and requires full 3 Fall frames.
    6. RED latch requires manual acknowledge() + full 5 consecutive GREEN frames.
    7. Instant RED override (confidence >= 0.90) triggers RED in 1 tick.
    8. SequenceBuffer flushes stale frames after >3 missing frames.
    9. Train/Inference Missing-Frame Equivalence (max diff = 0.0).

  Event Evaluation Integration Tests (10–14):
   10. Video Fall stream with 3 consecutive Fall predictions (>0.75) -> triggers RED
       (delta from 1st critical tick to 3rd critical tick = 2 ticks = 0.2s @ 10 FPS).
   11. Video Fall stream with 1 Fall prediction (>0.90) -> triggers RED immediately.
   12. Video Non-Fall stream with no Fall predictions -> never triggers RED.
   13. Two videos evaluated sequentially -> RED state from video 1 never leaks into video 2.
   14. Short video ending before buffer reaches 16 frames (T < 16) -> no crash.

Run from project root:
    python tests/test_pipeline_units.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypedDict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import (
    LABEL_TO_ALERT_COLOR_MAPPING,
    CLASS_NAMES,
    NUM_JOINTS,
    NUM_COORDS,
    WINDOW_SIZE,
)
from utils.skeleton_utils import sliding_windows, compute_resample_indices
from pipeline.alert_state import AlertStateMachine, AlertLevel, _label_to_alert_level
from pipeline.action_classifier import ActionPrediction
from pipeline.sequence_buffer import SequenceBuffer


class StreamEvalResult(TypedDict):
    any_fall_predicted: bool
    red_triggered: bool
    first_crit_tick: Optional[int]
    red_trigger_tick: Optional[int]
    trigger_delta_sec: Optional[float]
    final_level: AlertLevel


def simulate_video_event_stream(
    frames_kp: Sequence[Optional[np.ndarray]],
    predictor_fn: Callable[[SequenceBuffer, int], ActionPrediction],
    fps: float = 10.0,
    pad_short_clip: bool = False,
) -> StreamEvalResult:
    """
    Run a fresh SequenceBuffer + fresh AlertStateMachine over one video's
    keypoint sequence at `fps` cadence.
    """
    stream: list[Optional[np.ndarray]] = list(frames_kp)
    if pad_short_clip and 0 < len(stream) < WINDOW_SIZE:
        stream.extend([stream[-1]] * (WINDOW_SIZE - len(stream)))

    # Brand-new instances per video — zero state carryover between videos
    buf = SequenceBuffer(window_size=WINDOW_SIZE, max_missing_gap=3)
    sm  = AlertStateMachine()

    any_fall_predicted = False
    red_triggered = False
    first_crit_tick: Optional[int] = None
    red_trigger_tick: Optional[int] = None

    for tick_idx, kp in enumerate(stream):
        buf.append(kp)
        if not buf.is_ready():
            continue

        if not buf.has_valid_subject():
            pred = ActionPrediction(label="No_Person", label_idx=-1, confidence=0.0)
        else:
            pred = predictor_fn(buf, tick_idx)

        if pred.label == "Fall" and pred.confidence >= 0.50:
            any_fall_predicted = True

        prev_crit = sm.critical_count
        level = sm.update(pred)

        if prev_crit == 0 and sm.critical_count == 1 and first_crit_tick is None:
            first_crit_tick = tick_idx

        if level == AlertLevel.RED and not red_triggered:
            red_triggered = True
            red_trigger_tick = tick_idx
            if first_crit_tick is None:
                first_crit_tick = tick_idx

        if sm.critical_count == 0 and not red_triggered:
            first_crit_tick = None

    delta_sec = (
        (red_trigger_tick - first_crit_tick) / fps
        if (red_triggered and red_trigger_tick is not None and first_crit_tick is not None)
        else None
    )

    return {
        "any_fall_predicted": any_fall_predicted,
        "red_triggered": red_triggered,
        "first_crit_tick": first_crit_tick,
        "red_trigger_tick": red_trigger_tick,
        "trigger_delta_sec": delta_sec,
        "final_level": sm.level,
    }


def run_all_tests() -> None:
    print("=" * 74)
    print("Official Unit, Equivalence & Integration Test Suite")
    print("=" * 74)

    # 1. All 8 class names present in mapping
    assert set(LABEL_TO_ALERT_COLOR_MAPPING.keys()) == set(CLASS_NAMES)
    expected = {
        "Walk":      AlertLevel.GREEN,
        "Eat":       AlertLevel.GREEN,
        "Drink":     AlertLevel.GREEN,
        "Sit_Down":  AlertLevel.GREEN,
        "Stand_Up":  AlertLevel.GREEN,
        "Crouching": AlertLevel.YELLOW,
        "Lying":     AlertLevel.YELLOW,
        "Fall":      AlertLevel.RED,
    }
    for label, exp_level in expected.items():
        assert _label_to_alert_level(label) == exp_level
    print("[PASS]  1. All 8 class labels map to exact expected AlertLevel")

    sm = AlertStateMachine(
        n_warn=3, n_critical=3, n_recovery=5,
        conf_threshold=0.50, warning_threshold=0.50,
        critical_threshold=0.75, instant_red_threshold=0.90,
    )
    crouch_pred    = ActionPrediction(label="Crouching", confidence=0.75, model_type="stgcn")
    crit_fall      = ActionPrediction(label="Fall",      confidence=0.80, model_type="stgcn")
    instant_fall   = ActionPrediction(label="Fall",      confidence=0.95, model_type="stgcn")
    walk_pred      = ActionPrediction(label="Walk",      confidence=0.95, model_type="stgcn")
    no_person_pred = ActionPrediction(label="No_Person", confidence=0.00, model_type="stgcn")
    unknown_pred   = ActionPrediction(label="Unknown",   confidence=0.00, model_type="stgcn")

    # 2. Cross-level counter independence: Crouching -> Crouching -> Fall -> Fall -> Crouching
    sm.reset()
    sm.update(crouch_pred)
    sm.update(crouch_pred)
    assert sm.warning_count == 2 and sm.critical_count == 0 and sm.level == AlertLevel.GREEN
    sm.update(crit_fall)
    assert sm.warning_count == 0 and sm.critical_count == 1 and sm.level == AlertLevel.GREEN
    sm.update(crit_fall)
    assert sm.warning_count == 0 and sm.critical_count == 2 and sm.level == AlertLevel.GREEN
    sm.update(crouch_pred)
    assert sm.warning_count == 1 and sm.critical_count == 0 and sm.level == AlertLevel.GREEN
    print("[PASS]  2. Independent counters: Crouching(2) -> Fall(2) -> Crouching(1) never bleeds")

    # 3. Two Crouching frames then No_Person
    sm.reset()
    sm.update(crouch_pred)
    sm.update(crouch_pred)
    assert sm.warning_count == 2
    sm.update(no_person_pred)
    assert sm.warning_count == 0 and sm.level == AlertLevel.GREEN
    sm.update(crouch_pred)
    assert sm.warning_count == 1 and sm.level == AlertLevel.GREEN
    print("[PASS]  3. Two Crouching frames then No_Person resets warning_count to 0")

    # 4. Two Fall frames then Unknown
    sm.reset()
    sm.update(crit_fall)
    sm.update(crit_fall)
    assert sm.critical_count == 2
    sm.update(unknown_pred)
    assert sm.critical_count == 0 and sm.level == AlertLevel.GREEN
    sm.update(crit_fall)
    assert sm.critical_count == 1 and sm.level == AlertLevel.GREEN
    print("[PASS]  4. Two Fall frames then Unknown resets critical_count to 0")

    # 5. Switch from YELLOW to RED when recovery_count is non-zero
    sm.reset()
    for _ in range(3):
        sm.update(crouch_pred)
    assert sm.level == AlertLevel.YELLOW
    sm.update(walk_pred)
    sm.update(walk_pred)
    assert sm.recovery_count == 2 and sm.level == AlertLevel.YELLOW
    sm.update(crit_fall)
    assert sm.recovery_count == 0 and sm.critical_count == 1 and sm.level == AlertLevel.YELLOW
    sm.update(crit_fall)
    assert sm.critical_count == 2 and sm.level == AlertLevel.YELLOW
    sm.update(crit_fall)
    assert sm.level == AlertLevel.RED and sm.is_red_latched and sm.critical_count == 0
    print("[PASS]  5. YELLOW -> RED transition resets recovery_count and requires full 3 Fall frames")

    # 6. RED acknowledged, recovery interrupted at 4/5 frames, then requires full 5 GREEN frames
    sm.acknowledge()
    assert sm.level == AlertLevel.RED and not sm.is_red_latched
    for i in range(4):
        sm.update(walk_pred)
        assert sm.level == AlertLevel.RED and sm.recovery_count == i + 1
    sm.update(crouch_pred)
    assert sm.level == AlertLevel.RED and sm.recovery_count == 0
    for i in range(4):
        sm.update(walk_pred)
        assert sm.level == AlertLevel.RED and sm.recovery_count == i + 1
    sm.update(walk_pred)
    assert sm.level == AlertLevel.GREEN and sm.recovery_count == 0
    print("[PASS]  6. Acknowledged RED with interrupted recovery requires full 5 GREEN frames")

    # 7. Instant RED override (confidence >= 0.90) triggers RED in 1 frame
    sm.reset()
    sm.update(instant_fall)
    assert sm.level == AlertLevel.RED and sm.is_red_latched
    print("[PASS]  7. Single-frame Fall with conf >= INSTANT_RED_THRESHOLD (0.90) triggers RED immediately")

    # 8. SequenceBuffer flush after >3 missing frames
    buf = SequenceBuffer(window_size=16, max_missing_gap=3)
    rng = np.random.default_rng(123)
    for _ in range(16):
        buf.append(rng.normal(size=(NUM_JOINTS, NUM_COORDS)).astype(np.float32))
    assert buf.is_ready()
    for _ in range(3):
        buf.append(None)
    assert buf.is_ready() and buf.valid_frame_count == 13
    buf.append(None)
    assert len(buf) == 0 and not buf.is_ready()
    print("[PASS]  8. SequenceBuffer flushes stale frames after >3 consecutive missing frames")

    # 9. Exact Train/Inference Missing-Frame Equivalence Test
    raw_stream: list[Optional[np.ndarray]] = [
        rng.normal(size=(NUM_JOINTS, NUM_COORDS)).astype(np.float32) if t not in (4, 5, 11) else None
        for t in range(WINDOW_SIZE)
    ]
    zero_frame = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    train_frames = [kp if kp is not None else zero_frame.copy() for kp in raw_stream]
    train_seq = np.stack(train_frames, axis=0)
    train_win = sliding_windows(train_seq, window_size=WINDOW_SIZE, stride=4)[0]

    inf_buf = SequenceBuffer(window_size=WINDOW_SIZE, max_missing_gap=3)
    for kp in raw_stream:
        inf_buf.append(kp)

    assert np.array_equal(
        inf_buf.as_lstm_tensor().numpy(),
        train_win.reshape(1, WINDOW_SIZE, NUM_JOINTS * NUM_COORDS),
    )
    assert np.array_equal(
        inf_buf.as_stgcn_tensor().numpy(),
        train_win.transpose(2, 0, 1)[np.newaxis, ...],
    )
    print("[PASS]  9. Train/Inference Missing-Frame Equivalence: SequenceBuffer == training extraction")

    # ── Event Evaluation Integration Tests (10–14) ──────────────────────────
    valid_kp = np.ones((NUM_JOINTS, NUM_COORDS), dtype=np.float32)

    # 10. Video Fall: 3 consecutive Fall predictions > 0.75 -> triggers RED (delta = 0.2s @ 10 FPS)
    frames_25: list[Optional[np.ndarray]] = [valid_kp.copy() for _ in range(25)]

    def pred_3_consec_fall(_b: SequenceBuffer, tick: int) -> ActionPrediction:
        if tick in (18, 19, 20):
            return ActionPrediction(label="Fall", confidence=0.82, model_type="stgcn")
        return ActionPrediction(label="Walk", confidence=0.90, model_type="stgcn")

    res10 = simulate_video_event_stream(frames_25, pred_3_consec_fall, fps=10.0)
    assert res10["red_triggered"] is True and res10["any_fall_predicted"] is True
    assert res10["first_crit_tick"] == 18 and res10["red_trigger_tick"] == 20
    delta10 = res10["trigger_delta_sec"]
    assert delta10 is not None and abs(delta10 - 0.20) < 1e-6
    print("[PASS] 10. Integration: 3 consecutive Fall predictions (>0.75) trigger RED (delta = 0.20s @ 10 FPS)")

    # 11. Video Fall: 1 Fall prediction > 0.90 -> triggers RED immediately (delta = 0.0s)
    def pred_1_instant_fall(_b: SequenceBuffer, tick: int) -> ActionPrediction:
        if tick == 17:
            return ActionPrediction(label="Fall", confidence=0.94, model_type="stgcn")
        return ActionPrediction(label="Walk", confidence=0.90, model_type="stgcn")

    res11 = simulate_video_event_stream(frames_25, pred_1_instant_fall, fps=10.0)
    assert res11["red_triggered"] is True and res11["red_trigger_tick"] == 17
    delta11 = res11["trigger_delta_sec"]
    assert delta11 is not None and abs(delta11 - 0.00) < 1e-6
    print("[PASS] 11. Integration: 1 Fall prediction (>0.90) triggers RED immediately (delta = 0.00s)")

    # 12. Video Non-Fall: no Fall predictions -> never triggers RED
    def pred_non_fall(_b: SequenceBuffer, tick: int) -> ActionPrediction:
        if tick >= 18:
            return ActionPrediction(label="Crouching", confidence=0.80, model_type="stgcn")
        return ActionPrediction(label="Walk", confidence=0.90, model_type="stgcn")

    res12 = simulate_video_event_stream(frames_25, pred_non_fall, fps=10.0)
    assert res12["red_triggered"] is False and res12["any_fall_predicted"] is False
    assert res12["final_level"] == AlertLevel.YELLOW
    print("[PASS] 12. Integration: Non-Fall video (Walk + Crouching) reaches at most YELLOW, never RED")

    # 13. Two videos run sequentially -> RED state from Video 1 does NOT leak into Video 2
    res_vid1 = simulate_video_event_stream(frames_25, pred_1_instant_fall, fps=10.0)
    res_vid2 = simulate_video_event_stream(frames_25, pred_non_fall, fps=10.0)
    assert res_vid1["red_triggered"] is True and res_vid1["final_level"] == AlertLevel.RED
    assert res_vid2["red_triggered"] is False and res_vid2["final_level"] != AlertLevel.RED
    print("[PASS] 13. Integration: Sequential videos use isolated buffer/state (zero cross-video state leakage)")

    # 14. Short video ending before buffer reaches 16 frames (T=10 < 16) -> no crash
    short_frames: list[Optional[np.ndarray]] = [valid_kp.copy() for _ in range(10)]
    res_short_unpadded = simulate_video_event_stream(
        short_frames, pred_1_instant_fall, fps=10.0, pad_short_clip=False
    )
    assert res_short_unpadded["red_triggered"] is False and res_short_unpadded["final_level"] == AlertLevel.GREEN
    print("[PASS] 14. Integration: Short video (<16 frames) terminates cleanly without crashing")

    # 15. Boundary Test: confidence == 0.75 (CRITICAL_THRESHOLD) vs 0.7499
    sm.reset()
    sub_crit_fall   = ActionPrediction(label="Fall", confidence=0.7499, model_type="stgcn")
    exact_crit_fall = ActionPrediction(label="Fall", confidence=0.7500, model_type="stgcn")
    sm.update(sub_crit_fall)
    assert sm.critical_count == 0 and sm.level == AlertLevel.GREEN
    sm.update(exact_crit_fall)
    assert sm.critical_count == 1 and sm.level == AlertLevel.GREEN
    sm.update(exact_crit_fall)
    assert sm.critical_count == 2 and sm.level == AlertLevel.GREEN
    sm.update(exact_crit_fall)
    assert sm.critical_count == 0 and sm.level == AlertLevel.RED and sm.is_red_latched
    print("[PASS] 15. Boundary: confidence == 0.75 (>= CRITICAL_THRESHOLD) increments & triggers RED on 3rd tick")

    # 16. Boundary Test: confidence == 0.90 (INSTANT_RED_THRESHOLD) vs 0.8999
    sm.reset()
    just_below_instant = ActionPrediction(label="Fall", confidence=0.8999, model_type="stgcn")
    exact_instant_fall = ActionPrediction(label="Fall", confidence=0.9000, model_type="stgcn")
    sm.update(just_below_instant)
    assert sm.level == AlertLevel.GREEN and sm.critical_count == 1
    sm.reset()
    sm.update(exact_instant_fall)
    assert sm.level == AlertLevel.RED and sm.is_red_latched and sm.critical_count == 0
    print("[PASS] 16. Boundary: confidence == 0.90 (>= INSTANT_RED_THRESHOLD) triggers Instant RED in 1 tick")

    # 17. Timestamp-Based Resampling Verification (25, 30, 24, 15 FPS -> exact 10.0 FPS)
    for nat_fps in (25.0, 30.0, 24.0, 15.0):
        tot_f = int(nat_fps * 10)  # 10.0 seconds video
        idx_arr, t_arr = compute_resample_indices(tot_f, nat_fps, sample_fps=10.0)
        assert len(idx_arr) == 100, f"Expected 100 frames for 10s @ 10 FPS from {nat_fps} FPS, got {len(idx_arr)}"
        assert abs(t_arr[1] - t_arr[0] - 0.10) < 1e-9
    idx_25, _ = compute_resample_indices(250, 25.0, sample_fps=10.0)
    diffs_25 = np.diff(idx_25[:6]).tolist()
    assert diffs_25 == [3, 2, 3, 2, 3], f"Unexpected 25->10 FPS index steps: {diffs_25}"
    print("[PASS] 17. Timestamp Resampling: 25/30/24/15 FPS videos all resample to exact 10.0 FPS (0.10s spacing)")

    # 18. Milestone 5A: AlertStateMachine.set_thresholds preserves active level/counters & avoids config mutation
    import tempfile
    from utils.config import CRITICAL_THRESHOLD as GLOBAL_CRIT_THRESH
    from utils.logger import log_event, read_recent_events, export_events_csv_bytes

    sm.reset()
    sm.update(ActionPrediction(label="Crouching", confidence=0.65, model_type="stgcn"))
    assert sm.warning_count == 1 and sm.level == AlertLevel.GREEN
    sm.set_thresholds(critical_threshold=0.60, n_warn=2)
    assert sm.warning_count == 1 and sm.level == AlertLevel.GREEN
    assert GLOBAL_CRIT_THRESH == 0.75, "Global config constant must not be mutated!"
    sm.update(ActionPrediction(label="Crouching", confidence=0.65, model_type="stgcn"))
    assert sm.level == AlertLevel.YELLOW, "Updated n_warn=2 should trigger YELLOW on 2nd warning frame"
    print("[PASS] 18. Milestone 5A: AlertStateMachine.set_thresholds updates runtime thresholds without resetting state or mutating config")

    # 19. Milestone 5A: Event logger CSV persistence & readback
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_csv = Path(tmp_dir) / "test_events.csv"
        log_event(
            source="demo:Fall/video (122)_fall.mp4",
            model="stgcn",
            prev_level="GREEN",
            new_level="RED",
            action_label="Fall",
            confidence=0.9412,
            frame_idx=42,
            video_time_s=1.40,
            note="RED latched (Critical Fall)",
            csv_path=tmp_csv,
        )
        rows = read_recent_events(limit=5, csv_path=tmp_csv)
        assert len(rows) == 1 and rows[0]["new_level"] == "RED" and rows[0]["model"] == "STGCN"
        csv_bytes = export_events_csv_bytes(csv_path=tmp_csv)
        assert b"RED latched (Critical Fall)" in csv_bytes
    print("[PASS] 19. Milestone 5A: Event logger writes, reads newest-first, and exports CSV bytes cleanly")

    # 20. Ensemble probabilities == mean of ST-GCN and LSTM probabilities
    from pipeline.action_classifier import load_action_model, predict_probs
    from pipeline.stabilizer import PredictionStabilizer
    _root = Path(__file__).resolve().parents[1]
    _models = {
        "stgcn": load_action_model(str(_root / "models" / "action_stgcn.pth"), model_type="stgcn"),
        "lstm":  load_action_model(str(_root / "models" / "action_lstm.pth"),  model_type="lstm"),
    }
    _buf = SequenceBuffer()
    for _ in range(16):
        _buf.append(valid_kp.copy())
    p_s = predict_probs(_buf, _models, "stgcn")
    p_l = predict_probs(_buf, _models, "lstm")
    p_e = predict_probs(_buf, _models, "ensemble")
    assert p_s is not None and p_l is not None and p_e is not None
    assert np.allclose(p_e, (p_s + p_l) / 2.0, atol=1e-9) and abs(float(p_e.sum()) - 1.0) < 1e-6
    assert predict_probs(SequenceBuffer(), _models, "stgcn") is None
    print("[PASS] 20. Ensemble: soft-vote probabilities equal the mean of ST-GCN and LSTM; unready buffer -> None")

    # 21. Stabilizer: Idle gate + EMA + reset behaviour
    stab = PredictionStabilizer()
    pred_idle = stab.step(_buf, _models, "stgcn", ema_strength=0.0, idle_gate=1.01)
    assert pred_idle.label == "Idle" and pred_idle.label_idx == -1
    assert LABEL_TO_ALERT_COLOR_MAPPING.get("Idle", "GREEN") == "GREEN"
    stab.reset()
    pred_raw = stab.step(_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.0)
    assert pred_raw.label == CLASS_NAMES[int(np.argmax(p_s))] and abs(pred_raw.confidence - float(p_s.max())) < 1e-6
    pred_ens = stab.step(_buf, _models, "ensemble", ema_strength=0.0, idle_gate=0.0)
    assert pred_ens.model_type == "ensemble"
    assert stab.step(SequenceBuffer(), _models, "stgcn").label == "Unknown"
    print("[PASS] 21. Stabilizer: Idle gate -> 'Idle' (GREEN), EMA=0 equals raw model output, ensemble supported")

    # 22. ST-GCN v3.1-A/B/C/D Architecture & Auto-Detection Verification
    import torch
    from pipeline.action_classifier import ActionSTGCN, _instantiate_stgcn_from_state_dict
    test_x = torch.randn(4, 3, 16, 17)
    test_x[:, 2, :, :] = torch.rand(4, 16, 17)  # valid confidence
    test_x[0, :, :4, :] = 0.0                   # simulate leading all-zero missing frames
    ablation_cases: list[tuple[str, dict[str, Any]]] = [
        ("v3.1-A (Gravity)", dict(use_gravity_feats=True)),
        ("v3.1-B (Wrist-Head)", dict(use_wrist_head_feats=True)),
        ("v3.1-C (Lower-Body)", dict(use_lower_body_feats=True)),
        ("v3.1-D4 (Wrist-Head + Lower-Body B+C)", dict(use_wrist_head_feats=True, use_lower_body_feats=True)),
        ("v3.1-D3 (Full Fusion A+B+C)", dict(use_gravity_feats=True, use_wrist_head_feats=True, use_lower_body_feats=True)),
    ]
    for v_tag, kw in ablation_cases:
        m31 = ActionSTGCN(**kw)
        m31.eval()
        with torch.no_grad():
            out31 = m31(test_x)
        assert out31.shape == (4, 8) and not torch.isnan(out31).any(), f"NaN or bad shape in {v_tag}"
        st31 = m31.state_dict()
        reloaded = _instantiate_stgcn_from_state_dict(st31)
        reloaded.load_state_dict(st31)
        reloaded.eval()
        with torch.no_grad():
            assert torch.allclose(out31, reloaded(test_x), atol=1e-6)
    print("[PASS] 22. ST-GCN v3.1-A/B/C/D: All feature branches NaN-safe on missing frames & auto-detected from state_dict")

    # 23. Virtual Hip Fallback & Physical Pose-Quality Stabilizer Verification
    from utils.skeleton_utils import normalise_keypoints
    raw_half_body = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    raw_half_body[0]  = [640.0, 180.0, 0.99]  # nose
    raw_half_body[5]  = [740.0, 300.0, 0.98]  # left_shoulder
    raw_half_body[6]  = [540.0, 300.0, 0.98]  # right_shoulder (sh_w = 200 px)
    raw_half_body[9]  = [750.0, 460.0, 0.90]  # left_wrist
    raw_half_body[10] = [530.0, 450.0, 0.90]  # right_wrist
    # Hips (11, 12), knees (13, 14), ankles (15, 16) all remain (0, 0, 0)
    norm_half = normalise_keypoints(raw_half_body)
    assert norm_half is not None and norm_half.shape == (NUM_JOINTS, NUM_COORDS)
    assert abs(float(norm_half[5, 1]) - 1.0) < 0.05 and float(norm_half[0, 1]) > 1.2
    # Virtual hips carry sentinel confidence 0.25 (< 0.50), distinct from observed hips
    assert abs(float(norm_half[11, 2]) - 0.25) < 1e-6 and abs(float(norm_half[12, 2]) - 0.25) < 1e-6

    # Feed static occluded-lower-body window: verify metadata, fallback logging, and Idle gate
    static_buf = SequenceBuffer(window_size=WINDOW_SIZE)
    for _ in range(WINDOW_SIZE):
        static_buf.append(norm_half.copy())
    p_static_raw = predict_probs(static_buf, _models, "stgcn")
    assert p_static_raw is not None
    stab23 = PredictionStabilizer()
    pred23_raw = stab23.step(static_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.0)
    assert pred23_raw.label == CLASS_NAMES[int(np.argmax(p_static_raw))]
    assert pred23_raw.pose_quality == "full" and pred23_raw.fallback_used is False
    stab23.reset()
    pred23_gated = stab23.step(static_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.45)
    assert pred23_gated.label == "Idle", f"Expected stationary occluded half-body to gate to Idle, got {pred23_gated.label}"
    assert pred23_gated.pose_quality == "upper_only"
    if int(np.argmax(p_static_raw)) in (5, 6, 7):
        assert pred23_gated.fallback_used is True and pred23_gated.fallback_model == "lstm"
    print("[PASS] 23. Virtual Hip sentinel conf (0.25), upper_only metadata, LSTM fallback tracking, and Idle gate verified")

    # 24. AlertStateMachine pose-quality guards & Fast-Attack Fall Passthrough
    sm24 = AlertStateMachine()
    for _ in range(3):
        sm24.update(ActionPrediction(label="Lying", confidence=0.80, model_type="stgcn", pose_quality="full"))
    assert sm24.level == AlertLevel.YELLOW
    # Low_Pose_Quality and upper_only Idle must NEVER advance recovery toward GREEN
    for _ in range(6):
        sm24.update(ActionPrediction(label="Low_Pose_Quality", confidence=0.0, model_type="stgcn", pose_quality="low"))
        sm24.update(ActionPrediction(label="Idle", confidence=0.90, model_type="stgcn", pose_quality="upper_only"))
    assert sm24.level == AlertLevel.YELLOW and sm24.recovery_count == 0
    # Valid full-body Walk recovers YELLOW -> GREEN after 5 consecutive frames
    for _ in range(5):
        sm24.update(ActionPrediction(label="Walk", confidence=0.90, model_type="stgcn", pose_quality="full"))
    assert sm24.level == AlertLevel.GREEN

    # Fast-Attack Fall Passthrough prevents EMA from diluting a sudden Fall spike >= 0.90
    import pipeline.stabilizer as _stab_mod
    _orig_predict_probs = _stab_mod.predict_probs
    _call_idx = [0]

    def _mock_predict_probs(_buf, _m, _active="stgcn", _device=None) -> np.ndarray:
        _call_idx[0] += 1
        vec = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        if _call_idx[0] <= 2:
            vec[0] = 0.99  # Walk
        else:
            vec[3] = 0.94  # Sudden Fall spike
        return vec

    try:
        _stab_mod.predict_probs = _mock_predict_probs
        stab24 = PredictionStabilizer()
        walk_buf = SequenceBuffer(window_size=WINDOW_SIZE)
        for t in range(WINDOW_SIZE):
            wk = valid_kp.copy()
            wk[:, 0] = 0.0
            wk[:, 1] = 0.0
            wk[0] = [0.0, 1.35, 1.0]
            wk[5] = [0.35, 1.0, 1.0]
            wk[6] = [-0.35, 1.0, 1.0]
            wk[7] = [0.38 + 0.08 * float(np.sin(t * 0.8)), 0.55, 1.0]
            wk[8] = [-0.38 - 0.08 * float(np.sin(t * 0.8)), 0.55, 1.0]
            wk[9] = [0.40 + 0.14 * float(np.sin(t * 0.8)), 0.10, 1.0]
            wk[10] = [-0.40 - 0.14 * float(np.sin(t * 0.8)), 0.10, 1.0]
            wk[11] = [0.20, 0.0, 1.0]
            wk[12] = [-0.20, 0.0, 1.0]
            wk[13] = [0.20 + 0.14 * float(np.sin(t * 0.8)), -0.65, 1.0]
            wk[14] = [-0.20 - 0.14 * float(np.sin(t * 0.8)), -0.65, 1.0]
            wk[15] = [0.20 + 0.22 * float(np.sin(t * 0.8)), -1.30, 1.0]
            wk[16] = [-0.20 - 0.22 * float(np.sin(t * 0.8)), -1.30, 1.0]
            walk_buf.append(wk)
        p_w1 = stab24.step(walk_buf, _models, "stgcn", ema_strength=0.5, idle_gate=0.45)
        p_w2 = stab24.step(walk_buf, _models, "stgcn", ema_strength=0.5, idle_gate=0.45)
        p_f3 = stab24.step(walk_buf, _models, "stgcn", ema_strength=0.5, idle_gate=0.45)
        assert p_w1.label == "Walk" and p_w2.label == "Walk"
        assert p_f3.label == "Fall" and p_f3.confidence >= 0.90, f"EMA diluted Fall spike: {p_f3}"
    finally:
        _stab_mod.predict_probs = _orig_predict_probs
    print("[PASS] 24. Low_Pose_Quality/upper_only Idle preserve YELLOW/RED latch & Fast-Attack Fall Passthrough preserves >=0.90 spikes")

    # 25. Stage-2B Conditional Hand-Object Fusion & Fall Safety Invariance
    import pipeline.object_context as _obj_mod
    from pipeline.detector import PoseResult
    from pipeline.object_context import ObjectContextResult, detect_hand_objects, load_object_model

    _orig_detect_obj = _obj_mod.detect_hand_objects
    _obj_mode = ["cup"]
    _obj_calls = [0]
    _prob_vec = [np.zeros(len(CLASS_NAMES), dtype=np.float64)]

    def _mock_predict_2b(_buf, _m, _active="stgcn", _device=None) -> np.ndarray:
        return _prob_vec[0].copy()

    def _mock_detect_obj(_frame, _pose, _obj_model, **_kwargs):
        _obj_calls[0] += 1
        if _obj_mode[0] == "cup":
            return ObjectContextResult(drink_score=0.78, detected_items=["cup:0.78"], roi_valid=True)
        if _obj_mode[0] == "bowl":
            return ObjectContextResult(eat_score=0.65, detected_items=["bowl:0.65"], roi_valid=True)
        if _obj_mode[0] == "backpack":
            return ObjectContextResult(distractor_score=0.55, detected_items=["backpack:0.55"], roi_valid=True)
        return ObjectContextResult(roi_valid=True)

    hand_up_buf = SequenceBuffer(window_size=WINDOW_SIZE)
    for t in range(WINDOW_SIZE):
        fr_kp = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        fr_kp[0]  = [0.0, 1.35, 1.0]   # nose
        fr_kp[5]  = [0.35, 1.00, 1.0]  # left shoulder
        fr_kp[6]  = [-0.35, 1.00, 1.0] # right shoulder
        fr_kp[7]  = [0.40, 0.55, 1.0]  # left elbow
        fr_kp[8]  = [-0.30, 0.65, 1.0] # right elbow
        fr_kp[9]  = [0.45, 0.25, 1.0]  # left wrist resting down
        fr_kp[10] = [-0.08, 1.05 + 0.02 * float(np.sin(t)), 1.0]  # right wrist raised near nose/mouth
        fr_kp[11] = [0.20, 0.0, 1.0]; fr_kp[12] = [-0.20, 0.0, 1.0]
        fr_kp[13] = [0.20, -0.65, 1.0]; fr_kp[14] = [-0.20, -0.65, 1.0]
        fr_kp[15] = [0.20, -1.30, 1.0]; fr_kp[16] = [-0.20, -1.30, 1.0]
        hand_up_buf.append(fr_kp)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_raw = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    dummy_raw[0] = [320, 100, 0.99]; dummy_raw[5] = [385, 160, 0.99]; dummy_raw[6] = [255, 160, 0.99]
    dummy_raw[7] = [395, 240, 0.95]; dummy_raw[8] = [245, 220, 0.95]
    dummy_raw[9] = [390, 300, 0.90]; dummy_raw[10] = [305, 130, 0.95]
    dummy_pose = PoseResult(
        keypoints_norm=hand_up_buf._buffer[-1],
        keypoints_raw=dummy_raw,
        bbox_xyxy=np.array([220.0, 60.0, 420.0, 440.0], dtype=np.float32),
        detection_conf=0.95,
        person_detected=True,
        pose_quality="full",
    )

    try:
        _stab_mod.predict_probs = _mock_predict_2b
        _obj_mod.detect_hand_objects = _mock_detect_obj
        stab25 = PredictionStabilizer()

        # 25a. Cup context resolves Eat -> Drink
        _obj_mode[0] = "cup"
        _prob_vec[0] = np.array([0.0, 0.70, 0.25, 0.0, 0.0, 0.03, 0.02, 0.0], dtype=np.float64)
        p25a = stab25.step(
            hand_up_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.45,
            frame_bgr=dummy_frame, pose_result=dummy_pose, object_model=object(),
        )
        assert p25a.label == "Drink" and p25a.object_context == "cup:0.78", f"Expected Drink+cup, got {p25a}"

        # 25b. Bowl context resolves Drink -> Eat
        stab25.reset()
        _obj_mode[0] = "bowl"
        _prob_vec[0] = np.array([0.0, 0.25, 0.70, 0.0, 0.0, 0.03, 0.02, 0.0], dtype=np.float64)
        p25b = stab25.step(
            hand_up_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.45,
            frame_bgr=dummy_frame, pose_result=dummy_pose, object_model=object(),
        )
        assert p25b.label == "Eat" and p25b.object_context == "bowl:0.65", f"Expected Eat+bowl, got {p25b}"

        # 25c. Backpack distractor suppresses false Drink -> Idle
        stab25.reset()
        _obj_mode[0] = "backpack"
        _prob_vec[0] = np.array([0.20, 0.05, 0.65, 0.0, 0.0, 0.05, 0.05, 0.0], dtype=np.float64)
        p25c = stab25.step(
            hand_up_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.45,
            frame_bgr=dummy_frame, pose_result=dummy_pose, object_model=object(),
        )
        assert p25c.label == "Idle" and p25c.object_context == "backpack:0.55", f"Expected Idle+backpack, got {p25c}"

        # 25d. Repeated no_vessel checks with low wr_e suppress false Drink -> Idle
        stab25.reset()
        _obj_mode[0] = "none"
        _prob_vec[0] = np.array([0.15, 0.05, 0.65, 0.0, 0.0, 0.05, 0.10, 0.0], dtype=np.float64)
        for _ in range(2):
            stab25.step(
                hand_up_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.45,
                frame_bgr=dummy_frame, pose_result=dummy_pose, object_model=object(),
            )
        p25d = stab25.step(
            hand_up_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.45,
            frame_bgr=dummy_frame, pose_result=dummy_pose, object_model=object(),
        )
        assert p25d.label == "Idle" and p25d.object_context == "no_vessel", f"Expected Idle+no_vessel, got {p25d}"

        # 25e. Fall Safety Invariance: Fall >= 0.40 skips object detector completely
        stab25.reset()
        _obj_calls[0] = 0
        _obj_mode[0] = "cup"
        _prob_vec[0] = np.array([0.02, 0.0, 0.0, 0.94, 0.02, 0.02, 0.0, 0.0], dtype=np.float64)
        p25e = stab25.step(
            hand_up_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.45,
            frame_bgr=dummy_frame, pose_result=dummy_pose, object_model=object(),
        )
        assert _obj_calls[0] == 0 and p25e.label == "Fall" and p25e.confidence >= 0.90 and p25e.object_context == ""
    finally:
        _stab_mod.predict_probs = _orig_predict_probs
        _obj_mod.detect_hand_objects = _orig_detect_obj
    print("[PASS] 25. Stage-2B Conditional Hand-Object Fusion & Fall Safety Invariance verified")

    # 26. Stage-2B Object Model Loader & ROI Crop Smoke Test
    assert load_object_model(str(_root / "models" / "does_not_exist_obj_123.pt")) is None
    obj_m26 = load_object_model(str(_root / "models" / "yolo_obj_pretrained.pt"))
    assert obj_m26 is not None, "Expected models/yolo_obj_pretrained.pt to load"
    res26 = detect_hand_objects(dummy_frame, dummy_pose, obj_m26)
    assert isinstance(res26, ObjectContextResult) and res26.roi_valid is True
    print("[PASS] 26. Stage-2B Object Model Loader, missing-weight guard, and ROI crop inference passed")

    # 27. Stage-1 SubjectPoseTracker 0.30 Hysteresis & Spurious-Box Guard
    from pipeline.detector import SubjectPoseTracker, detect_and_extract_pose
    from pipeline.stabilizer import _extract_window_kinematics

    class _FakeArr:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = np.asarray(arr, dtype=np.float32)

        def cpu(self) -> "_FakeArr":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

        def __len__(self) -> int:
            return len(self._arr)

        def __getitem__(self, idx: int) -> "_FakeArr":
            return _FakeArr(self._arr[idx])

    class _FakeKP:
        def __init__(self, kps_list: list[np.ndarray]) -> None:
            arr = np.asarray(kps_list, dtype=np.float32) if len(kps_list) else np.zeros((0, 17, 3), dtype=np.float32)
            self.xy = arr[:, :, :2] if len(arr) else np.zeros((0, 17, 2), dtype=np.float32)
            self.data = _FakeArr(arr)

    class _FakeBoxes:
        def __init__(self, boxes_list: list[list[int]], confs_list: list[float]) -> None:
            self.xyxy = _FakeArr(np.asarray(boxes_list, dtype=np.float32)) if len(boxes_list) else _FakeArr(np.zeros((0, 4), dtype=np.float32))
            self.conf = _FakeArr(np.asarray(confs_list, dtype=np.float32)) if len(confs_list) else _FakeArr(np.zeros((0,), dtype=np.float32))

    class _FakeResult:
        def __init__(self, keypoints: _FakeKP, boxes: _FakeBoxes) -> None:
            self.keypoints = keypoints
            self.boxes = boxes

    class _FakeYOLO:
        def __init__(self) -> None:
            self.next_cands: list[tuple[list[int], float, np.ndarray]] = []

        def __call__(self, frame_bgr: np.ndarray, **kwargs: Any) -> list[_FakeResult]:
            if not self.next_cands:
                return [_FakeResult(_FakeKP([]), _FakeBoxes([], []))]
            return [
                _FakeResult(
                    _FakeKP([c[2] for c in self.next_cands]),
                    _FakeBoxes([c[0] for c in self.next_cands], [c[1] for c in self.next_cands]),
                )
            ]

    def _make_kp(joint_confs: dict[int, float]) -> np.ndarray:
        kp = np.zeros((17, 3), dtype=np.float32)
        coords = {
            0: (260, 130), 5: (240, 170), 6: (280, 170),
            7: (230, 220), 8: (290, 220), 9: (225, 260), 10: (295, 260),
            11: (245, 280), 12: (275, 280), 13: (245, 340), 14: (275, 340),
            15: (245, 390), 16: (275, 390),
        }
        for j, (x, y) in coords.items():
            c = joint_confs.get(j, 0.0)
            if c > 0.0:
                kp[j] = [float(x), float(y), float(c)]
        return kp

    ym27 = _FakeYOLO()
    tracker27 = SubjectPoseTracker(track_conf=0.30, max_lost_ticks=5, min_iou=0.10)
    f480 = np.zeros((480, 640, 3), dtype=np.uint8)

    # 27a. Primary (0.80) activates tracker, then Path A (0.38) continues & preserves 0.33 joints
    kp_primary = _make_kp({0: 0.85, 5: 0.85, 6: 0.85, 11: 0.80, 12: 0.80, 13: 0.75, 14: 0.75})
    ym27.next_cands = [([200, 100, 320, 400], 0.80, kp_primary)]
    r27a1 = detect_and_extract_pose(f480, ym27, conf_threshold=0.50, tracker=tracker27)
    assert r27a1.person_detected and not r27a1.tracked_low_conf and tracker27.active

    kp_hyst = _make_kp({0: 0.42, 5: 0.40, 6: 0.40, 11: 0.33, 12: 0.33, 13: 0.33, 14: 0.22})
    ym27.next_cands = [([205, 120, 340, 405], 0.38, kp_hyst)]
    r27a2 = detect_and_extract_pose(f480, ym27, conf_threshold=0.50, tracker=tracker27)
    assert r27a2.person_detected and r27a2.tracked_low_conf
    assert abs(r27a2.detection_conf - 0.38) < 1e-4
    assert r27a2.hip_observed and not r27a2.hip_virtual
    assert r27a2.keypoints_raw is not None and r27a2.keypoints_norm is not None
    assert r27a2.keypoints_raw[11, 2] >= 0.30 and r27a2.keypoints_raw[13, 2] >= 0.30
    assert r27a2.keypoints_raw[14, 2] == 0.0  # < 0.30 zeroed out

    # Boundary check: 0.33 tracked hips vs 0.25 virtual hips in _extract_window_kinematics
    k_tracked = _extract_window_kinematics(np.repeat(r27a2.keypoints_norm[None, ...], 16, axis=0))
    assert k_tracked["hip_obs_ratio"] == 1.0 and k_tracked["hip_virt_ratio"] == 0.0

    # 27b. Distant low-conf spurious box rejected while tracker active; upright unanchored 0.40 rejected after TTL expiry
    ym27.next_cands = [([10, 10, 70, 200], 0.35, _make_kp({5: 0.45, 6: 0.45, 11: 0.45, 12: 0.45}))]
    r27b_far = detect_and_extract_pose(f480, ym27, conf_threshold=0.50, tracker=tracker27)
    assert not r27b_far.person_detected and tracker27.lost_ticks == 1

    # Remaining 5 misses expire TTL (max_lost_ticks=5 -> reset on tick #6)
    ym27.next_cands = []
    for _ in range(5):
        detect_and_extract_pose(f480, ym27, conf_threshold=0.50, tracker=tracker27)
    assert not tracker27.active and tracker27.last_bbox is None

    # Upright unanchored 0.40 rejected by Path B
    ym27.next_cands = [([200, 100, 300, 380], 0.40, _make_kp({0: 0.6, 5: 0.6, 6: 0.6, 7: 0.5, 8: 0.5, 11: 0.5, 12: 0.5}))]
    r27b_up = detect_and_extract_pose(f480, ym27, conf_threshold=0.50, tracker=tracker27)
    assert not r27b_up.person_detected

    # 27c. Path B unanchored horizontal fall recovery (bw=260, bh=120, conf=0.40, strong pose)
    kp_fall_horiz = _make_kp({0: 0.55, 5: 0.50, 6: 0.50, 7: 0.45, 8: 0.45, 11: 0.45, 12: 0.45, 13: 0.42})
    ym27.next_cands = [([140, 300, 400, 420], 0.40, kp_fall_horiz)]
    r27c = detect_and_extract_pose(f480, ym27, conf_threshold=0.50, tracker=tracker27)
    assert r27c.person_detected and r27c.tracked_low_conf and tracker27.active

    # 27d. Two-pass priority: Path A matching tracked bbox (0.36) beats unrelated Path B box (0.46)
    tracker27.last_bbox = np.array([200, 100, 320, 400], dtype=np.float32)
    tracker27.lost_ticks = 0
    tracker27.active = True
    ym27.next_cands = [
        ([10, 320, 190, 430], 0.46, kp_fall_horiz),
        ([205, 110, 330, 405], 0.36, _make_kp({0: 0.40, 5: 0.40, 6: 0.40, 11: 0.36, 13: 0.36})),
    ]
    r27d = detect_and_extract_pose(f480, ym27, conf_threshold=0.50, tracker=tracker27)
    assert r27d.person_detected and r27d.tracked_low_conf and abs(r27d.detection_conf - 0.36) < 1e-4

    # 27e. Hysteresis + Virtual Hip composition (0.35 tracked upper-body -> virtual hips remain 0.25)
    ym27.next_cands = [([205, 110, 330, 300], 0.35, _make_kp({0: 0.60, 5: 0.65, 6: 0.65, 7: 0.50, 8: 0.50, 11: 0.10, 12: 0.10}))]
    r27e = detect_and_extract_pose(f480, ym27, conf_threshold=0.50, tracker=tracker27)
    assert r27e.person_detected and r27e.tracked_low_conf and r27e.hip_virtual and r27e.pose_quality == "upper_only"
    assert r27e.keypoints_norm is not None
    k_virt27 = _extract_window_kinematics(np.repeat(r27e.keypoints_norm[None, ...], 16, axis=0))
    assert k_virt27["hip_virt_ratio"] == 1.0 and k_virt27["hip_obs_ratio"] == 0.0

    # Without tracker/track_conf_threshold, stateless primary gate stays unchanged at 0.50
    ym27.next_cands = [([140, 300, 400, 420], 0.40, kp_fall_horiz)]
    r27f = detect_and_extract_pose(f480, ym27, conf_threshold=0.50)
    assert not r27f.person_detected
    print("[PASS] 27. Stage-1 SubjectPoseTracker 0.30 Hysteresis, Path A/B Priority & Virtual Hip Separation passed")

    # 28. Live Model Hot-Swap & Comparison Snapshot Isolation (self-contained, no assets/ required)
    from app import (
        compute_comparison_snapshot,
        create_initial_session_stats,
        update_session_stats,
    )
    from ui.overlay import annotate_frame

    seq28 = SequenceBuffer(window_size=WINDOW_SIZE, max_missing_gap=3)
    stab28 = PredictionStabilizer()
    sm28 = AlertStateMachine()
    cmp_stabs28 = {"stgcn": PredictionStabilizer(), "lstm": PredictionStabilizer()}
    cmp_sms28 = {"stgcn": AlertStateMachine(), "lstm": AlertStateMachine()}

    stand28 = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    stand28[0] = [0.0, 1.35, 1.0]
    stand28[5] = [0.30, 1.00, 1.0]; stand28[6] = [-0.30, 1.00, 1.0]
    stand28[7] = [0.35, 0.55, 1.0]; stand28[8] = [-0.35, 0.55, 1.0]
    stand28[9] = [0.40, 0.15, 1.0]; stand28[10] = [-0.40, 0.15, 1.0]
    stand28[11] = [0.20, 0.00, 1.0]; stand28[12] = [-0.20, 0.00, 1.0]
    stand28[13] = [0.20, -0.65, 1.0]; stand28[14] = [-0.20, -0.65, 1.0]
    stand28[15] = [0.20, -1.30, 1.0]; stand28[16] = [-0.20, -1.30, 1.0]

    floor28 = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    floor28[0] = [1.20, 0.15, 1.0]
    floor28[5] = [0.95, 0.10, 1.0]; floor28[6] = [0.85, -0.05, 1.0]
    floor28[7] = [0.70, 0.25, 1.0]; floor28[8] = [0.65, -0.15, 1.0]
    floor28[9] = [0.55, 0.35, 1.0]; floor28[10] = [0.50, -0.25, 1.0]
    floor28[11] = [0.15, 0.05, 1.0]; floor28[12] = [-0.15, -0.05, 1.0]
    floor28[13] = [-0.55, 0.02, 1.0]; floor28[14] = [-0.65, -0.08, 1.0]
    floor28[15] = [-1.05, 0.00, 1.0]; floor28[16] = [-1.15, -0.10, 1.0]

    last_snap28: Optional[dict[str, Any]] = None
    last_pred28: Optional[ActionPrediction] = None
    for t_i in range(28):
        if t_i < 10:
            frm_kp28 = stand28.copy()
        else:
            alpha28 = min(1.0, float(t_i - 9) / 10.0)
            frm_kp28 = (1.0 - alpha28) * stand28 + alpha28 * floor28
            frm_kp28[:, 2] = 1.0

        active28 = "lstm" if 16 <= t_i < 22 else ("ensemble" if 22 <= t_i < 25 else "stgcn")
        if t_i in (16, 22, 25):
            stab28.reset()

        seq28.append(frm_kp28)
        last_pred28 = stab28.step(seq28, _models, active28, ema_strength=0.5, idle_gate=0.45)
        sm28.update(last_pred28)
        last_snap28 = compute_comparison_snapshot(
            seq28, _models, cmp_stabs28, cmp_sms28, ema_strength=0.5, idle_gate=0.45,
        )

    assert last_pred28 is not None and isinstance(last_pred28, ActionPrediction)
    assert last_snap28 is not None and "stgcn" in last_snap28 and "lstm" in last_snap28
    assert isinstance(last_snap28["stgcn"]["prediction"], ActionPrediction)
    assert isinstance(last_snap28["lstm"]["level"], AlertLevel)
    print("[PASS] 28. Live Model Hot-Swap & Comparison Snapshot Isolation passed")

    # 29. Session Summary Stats & Overlay Formatting Polish
    stats29 = create_initial_session_stats()
    update_session_stats(stats29, AlertLevel.GREEN, AlertLevel.GREEN, "stgcn", dt_s=0.1)
    update_session_stats(stats29, AlertLevel.GREEN, AlertLevel.YELLOW, "stgcn", dt_s=0.1)
    update_session_stats(stats29, AlertLevel.YELLOW, AlertLevel.RED, "lstm", dt_s=0.1)
    update_session_stats(stats29, AlertLevel.RED, AlertLevel.RED, "lstm", dt_s=0.1)
    assert stats29["total_ticks"] == 4
    assert stats29["yellow_alerts"] == 1 and stats29["red_alerts"] == 1
    assert abs(stats29["time_in_level_s"]["GREEN"] - 0.1) < 1e-6
    assert abs(stats29["time_in_level_s"]["YELLOW"] - 0.1) < 1e-6
    assert abs(stats29["time_in_level_s"]["RED"] - 0.2) < 1e-6
    assert stats29["ticks_by_model"]["stgcn"] == 2 and stats29["ticks_by_model"]["lstm"] == 2
    assert stats29["alerts_by_model"]["stgcn"] == 1 and stats29["alerts_by_model"]["lstm"] == 1

    vis29 = annotate_frame(np.zeros((240, 320, 3), dtype=np.uint8), "Low_Pose_Quality", 0.0, "GREEN")
    assert vis29.shape == (240, 320, 3) and vis29.dtype == np.uint8
    print("[PASS] 29. Session Summary Stats & Overlay Polish passed")

    # 30. Seated-vs-Lying Late-Window Gate & Fall Timestamp Auto-Estimator
    seated_kp = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    seated_kp[0] = [0.0, 1.30, 1.0]
    seated_kp[5] = [0.25, 0.92, 1.0]; seated_kp[6] = [-0.25, 0.92, 1.0]
    seated_kp[11] = [0.20, 0.0, 1.0]; seated_kp[12] = [-0.20, 0.0, 1.0]
    seated_kp[13] = [0.20, -0.35, 1.0]; seated_kp[14] = [-0.20, -0.35, 1.0]
    seated_kp[15] = [0.20, -0.85, 1.0]; seated_kp[16] = [-0.20, -0.85, 1.0]

    horiz_kp = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    horiz_kp[0] = [1.20, 0.20, 1.0]
    horiz_kp[5] = [0.95, 0.15, 1.0]; horiz_kp[6] = [0.85, -0.05, 1.0]
    horiz_kp[11] = [0.15, 0.05, 1.0]; horiz_kp[12] = [-0.15, -0.05, 1.0]
    horiz_kp[13] = [-0.55, 0.02, 1.0]; horiz_kp[14] = [-0.65, -0.08, 1.0]
    horiz_kp[15] = [-1.05, 0.00, 1.0]; horiz_kp[16] = [-1.15, -0.10, 1.0]

    seat_buf = SequenceBuffer(window_size=WINDOW_SIZE)
    horiz_buf = SequenceBuffer(window_size=WINDOW_SIZE)
    for _ in range(WINDOW_SIZE):
        seat_buf.append(seated_kp.copy())
        horiz_buf.append(horiz_kp.copy())

    lying_probs = np.array([0.05, 0.05, 0.05, 0.02, 0.75, 0.05, 0.02, 0.01], dtype=np.float64)

    def _mock_lying_probs(_buf, _m, _active="stgcn", _device=None) -> np.ndarray:
        return lying_probs.copy()

    try:
        _stab_mod.predict_probs = _mock_lying_probs
        stab30 = PredictionStabilizer()
        p30_seat = stab30.step(seat_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.45)
        assert p30_seat.label == "Idle", f"Expected full-body seated pose to damp Lying -> Idle, got {p30_seat}"
        stab30.reset()
        p30_horiz = stab30.step(horiz_buf, _models, "stgcn", ema_strength=0.0, idle_gate=0.45)
        assert p30_horiz.label == "Lying", f"Expected horizontal pose to remain Lying, got {p30_horiz}"
    finally:
        _stab_mod.predict_probs = _orig_predict_probs

    scripts_dir = str(_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from annotate_fall_timestamps import estimate_fall_interval_from_keypoints

    seq30 = np.zeros((30, NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    for t in range(30):
        if t < 10:
            seq30[t] = seated_kp
        elif t >= 20:
            seq30[t] = horiz_kp
        else:
            alpha = float(t - 9) / 11.0
            seq30[t] = (1.0 - alpha) * seated_kp + alpha * horiz_kp
            seq30[t, :, 2] = 1.0
    est30 = estimate_fall_interval_from_keypoints(seq30, fps=10.0)
    assert 8 <= int(est30["fall_start_tick"]) <= 12, f"Bad fall_start_tick: {est30['fall_start_tick']}"
    assert 18 <= int(est30["fall_end_tick"]) <= 22, f"Bad fall_end_tick: {est30['fall_end_tick']}"

    # 30b. Verify Dual-Specialist ST-GCN fusion when "stgcn_wrist" is supplied
    from pipeline.action_classifier import ENSEMBLE_KEY, STGCN_WRIST_KEY, load_action_model as _load_act30
    from utils.config import STGCN_WRIST_MODEL_PATH

    wrist_path30 = _root / STGCN_WRIST_MODEL_PATH
    assert wrist_path30.is_file(), f"Missing Dual-Specialist checkpoint: {wrist_path30}"
    wrist_m30 = _load_act30(str(wrist_path30), model_type="stgcn")
    dual_models30 = {"stgcn": _models["stgcn"], STGCN_WRIST_KEY: wrist_m30, "lstm": _models["lstm"]}
    p_dual30 = predict_probs(horiz_buf, dual_models30, active_model="stgcn")
    assert p_dual30 is not None and p_dual30.shape == (len(CLASS_NAMES),)
    assert abs(float(p_dual30.sum()) - 1.0) < 1e-5
    p_ens30 = predict_probs(horiz_buf, dual_models30, active_model=ENSEMBLE_KEY)
    assert p_ens30 is not None and p_ens30.shape == (len(CLASS_NAMES),)
    assert abs(float(p_ens30.sum()) - 1.0) < 1e-5
    print("[PASS] 30. Seated-vs-Lying Late-Window Gate, Fall Timestamp Auto-Estimator & Dual-Specialist Fusion passed")

    print("\n" + "=" * 74)
    print("ALL 30 UNIT, EQUIVALENCE, BOUNDARY & INTEGRATION TESTS PASSED ✅")


if __name__ == "__main__":
    run_all_tests()
