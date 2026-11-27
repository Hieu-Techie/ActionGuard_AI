"""
app.py — ActionGuard AI: Milestone 5A Video-Upload Web MVP
==========================================================
Two-Stage Human Action Recognition & Fall Detection System:
  - Stage 1: YOLOv8n-pose (COCO 17-keypoint extraction & torso-length normalization)
  - Stage 2: Hot-swappable ST-GCN v2.4 (Proposed) and LSTM Baseline on 16-frame
             10-FPS sliding windows
  - Stage 3: 3-Level Alert State Machine (GREEN / YELLOW / RED with RED latch)
             + CSV transition logging (`logs/events.csv`)

Architecture Notes (Milestone 5A):
  1. Model Caching: `@st.cache_resource` loads YOLO-Pose, ST-GCN v2.4, and LSTM
     exactly once across reruns.
  2. Safe Uploaded Video Temp-File Management: Uploaded files from `st.file_uploader`
     are written to disk only once per unique `(name, size)` identity and unlinked
     on source switch or cleanup.
  3. 10-FPS Timestamp Resampling: Uses `compute_resample_indices(total_frames, native_fps, 10.0)`
     and sequential `cv2.VideoCapture.read()` stepping to match training cadence.
  4. Single-Tick Rerun Loop: Executes one sampled 10-FPS tick per Streamlit rerun
     followed by `st.rerun()`, keeping Pause, Reset, Acknowledge RED Alert,
     ST-GCN <-> LSTM hot-swap, and session threshold sliders responsive.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# Ensure `src/` is on sys.path whether invoked via `streamlit run src/app.py` or from root
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
SCENE_CUT_GRAY_DIFF = 35.0   # mean abs diff of 32x24 gray thumbnails (0-255) treated as a hard cut
PERSON_JUMP_FRAC = 0.35      # bbox-centre jump (fraction of frame width) between ticks treated as a cut
MAX_DISPLAY_WIDTH = 640      # px; frames are downscaled before being sent to the browser
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.action_classifier import (
    ActionPrediction,
    load_action_model,
)
from pipeline.alert_state import AlertLevel, AlertStateMachine
from pipeline.detector import PoseResult, SubjectPoseTracker, detect_and_extract_pose, draw_pose
from pipeline.object_context import load_object_model
from pipeline.sequence_buffer import SequenceBuffer
from pipeline.stabilizer import PredictionStabilizer
from ui.overlay import annotate_frame
from ui.sidebar import SidebarConfig, render_sidebar
from utils.config import (
    ACTIVE_ACTION_MODEL,
    CONF_THRESHOLD,
    TRACK_CONF_THRESHOLD,
    CRITICAL_THRESHOLD,
    INSTANT_RED_THRESHOLD,
    LABEL_TO_ALERT_COLOR_MAPPING,
    LSTM_MODEL_PATH,
    N_CRITICAL_FRAMES,
    N_RECOVERY_FRAMES,
    N_WARN_FRAMES,
    STGCN_MODEL_PATH,
    STGCN_WRIST_MODEL_PATH,
    TARGET_FPS,
    WARNING_THRESHOLD,
    WINDOW_SIZE,
    YOLO_OBJ_MODEL_PATH,
    YOLO_POSE_MODEL_PATH,
)
from utils.logger import ensure_events_csv, log_event, read_recent_events
from utils.skeleton_utils import compute_resample_indices, load_pose_model


# ════════════════════════════════════════════════════════════════════════════
# 1. Cached Model Loader (Loaded Once Per Process)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Stage-1 YOLO-Pose & Stage-2 (ST-GCN v4 Dual-Specialist + LSTM) models...")
def load_all_models() -> Tuple[Any, Dict[str, Any]]:
    """
    Load and cache all Stage-1 and Stage-2 models once:
      - Stage 1: `models/yolo_pose_pretrained.pt`
      - Stage 2: `models/action_stgcn.pth` (ST-GCN v4-C Lower-Body),
                 optional `models/action_stgcn_v4_b.pth` (ST-GCN v4-B Wrist-to-Head),
                 & `models/action_lstm.pth` (LSTM)
    """
    yolo_path = PROJECT_ROOT / YOLO_POSE_MODEL_PATH
    stgcn_path = PROJECT_ROOT / STGCN_MODEL_PATH
    wrist_path = PROJECT_ROOT / STGCN_WRIST_MODEL_PATH
    lstm_path = PROJECT_ROOT / LSTM_MODEL_PATH

    if not yolo_path.exists():
        raise FileNotFoundError(f"Missing YOLO-Pose checkpoint: {yolo_path}")
    if not stgcn_path.exists():
        raise FileNotFoundError(f"Missing ST-GCN checkpoint: {stgcn_path}")
    if not lstm_path.exists():
        raise FileNotFoundError(f"Missing LSTM checkpoint: {lstm_path}")

    pose_model = load_pose_model(str(yolo_path))
    action_models: Dict[str, Any] = {
        "stgcn": load_action_model(str(stgcn_path), model_type="stgcn"),
        "lstm": load_action_model(str(lstm_path), model_type="lstm"),
    }
    if wrist_path.exists():
        action_models["stgcn_wrist"] = load_action_model(str(wrist_path), model_type="stgcn")
    return pose_model, action_models


@st.cache_resource(show_spinner=False)
def load_cached_object_model() -> Optional[Any]:
    """Load and cache optional Stage-2B COCO object detector."""
    return load_object_model(str(PROJECT_ROOT / YOLO_OBJ_MODEL_PATH))


# ════════════════════════════════════════════════════════════════════════════
# 2. Session State & Video Capture Lifecycle Management
# ════════════════════════════════════════════════════════════════════════════

def _cleanup_video_capture(remove_temp_file: bool = True) -> None:
    """Release active OpenCV VideoCapture and optionally unlink temp uploaded file."""
    cap = st.session_state.get("video_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state["video_cap"] = None
    st.session_state["cap_native_pos"] = -1
    st.session_state["last_raw_bgr"] = None

    if remove_temp_file:
        tmp_path = st.session_state.get("temp_upload_path")
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        st.session_state["temp_upload_path"] = None
        st.session_state["upload_file_sig"] = None


def create_initial_session_stats() -> Dict[str, Any]:
    """Create cumulative per-session alert and runtime statistics."""
    return {
        "total_ticks": 0,
        "time_in_level_s": {"GREEN": 0.0, "YELLOW": 0.0, "RED": 0.0},
        "red_alerts": 0,
        "yellow_alerts": 0,
        "ticks_by_model": {"stgcn": 0, "lstm": 0, "ensemble": 0},
        "alerts_by_model": {"stgcn": 0, "lstm": 0, "ensemble": 0},
    }


def update_session_stats(
    stats: Dict[str, Any],
    prev_level: AlertLevel,
    new_level: AlertLevel,
    active_model: str,
    dt_s: float = 1.0 / float(TARGET_FPS),
) -> None:
    """Update cumulative session summary stats after one pipeline tick."""
    stats["total_ticks"] = int(stats["total_ticks"]) + 1
    time_map: Dict[str, float] = stats["time_in_level_s"]
    time_map[new_level.name] = float(time_map.get(new_level.name, 0.0)) + float(dt_s)
    ticks_map: Dict[str, int] = stats["ticks_by_model"]
    ticks_map[active_model] = int(ticks_map.get(active_model, 0)) + 1

    if new_level != prev_level:
        alerts_map: Dict[str, int] = stats["alerts_by_model"]
        if new_level == AlertLevel.RED:
            stats["red_alerts"] = int(stats["red_alerts"]) + 1
            alerts_map[active_model] = int(alerts_map.get(active_model, 0)) + 1
        elif new_level == AlertLevel.YELLOW and prev_level == AlertLevel.GREEN:
            stats["yellow_alerts"] = int(stats["yellow_alerts"]) + 1
            alerts_map[active_model] = int(alerts_map.get(active_model, 0)) + 1


def compute_comparison_snapshot(
    seq_buf: SequenceBuffer,
    action_models: Dict[str, Any],
    cmp_stabilizers: Dict[str, PredictionStabilizer],
    cmp_alert_sms: Dict[str, AlertStateMachine],
    *,
    ema_strength: float = 0.5,
    idle_gate: float = 0.45,
    frame_bgr: Optional[np.ndarray] = None,
    pose_result: Optional[PoseResult] = None,
    object_model: Optional[Any] = None,
    enable_object_context: bool = True,
) -> Dict[str, Any]:
    """Evaluate ST-GCN and LSTM side-by-side on the same buffered skeleton window."""
    snapshot: Dict[str, Any] = {}
    for m_key in ("stgcn", "lstm"):
        t0 = time.perf_counter()
        pred_m = cmp_stabilizers[m_key].step(
            seq_buf,
            action_models,
            m_key,
            ema_strength=ema_strength,
            idle_gate=idle_gate,
            frame_bgr=frame_bgr,
            pose_result=pose_result,
            object_model=object_model,
            enable_object_context=enable_object_context,
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        sm_m = cmp_alert_sms[m_key]
        lvl_m = sm_m.update(pred_m)
        snapshot[m_key] = {
            "prediction": pred_m,
            "level": lvl_m,
            "is_red_latched": bool(sm_m.is_red_latched),
            "latency_ms": float(lat_ms),
        }
    snapshot["agree"] = bool(
        snapshot["stgcn"]["prediction"].label == snapshot["lstm"]["prediction"].label
    )
    return snapshot


def init_session_state() -> None:
    """Initialize all persistent pipeline & UI objects in `st.session_state`."""
    ensure_events_csv()

    if "seq_buffer" not in st.session_state:
        st.session_state["seq_buffer"] = SequenceBuffer(window_size=WINDOW_SIZE, max_missing_gap=3)
    if "alert_sm" not in st.session_state:
        st.session_state["alert_sm"] = AlertStateMachine(
            n_warn=N_WARN_FRAMES,
            n_critical=N_CRITICAL_FRAMES,
            n_recovery=N_RECOVERY_FRAMES,
            conf_threshold=CONF_THRESHOLD,
            warning_threshold=WARNING_THRESHOLD,
            critical_threshold=CRITICAL_THRESHOLD,
            instant_red_threshold=INSTANT_RED_THRESHOLD,
        )

    defaults: Dict[str, Any] = {
        "is_running": False,
        "stream_finished": False,
        "active_source_id": None,
        "active_video_path": None,
        "temp_upload_path": None,
        "upload_file_sig": None,
        "video_cap": None,
        "cap_native_pos": -1,
        "last_raw_bgr": None,
        "native_fps": 30.0,
        "total_native_frames": 0,
        "resample_indices": np.empty((0,), dtype=np.int64),
        "resample_times": np.empty((0,), dtype=np.float64),
        "tick_idx": 0,
        "last_display_rgb": None,
        "last_display_jpeg": None,
        "last_pose_result": None,
        "last_prediction": ActionPrediction(
            label="Buffering...",
            label_idx=-1,
            confidence=0.0,
            model_type=ACTIVE_ACTION_MODEL,
        ),
        "last_stage1_ms": 0.0,
        "last_stage2_ms": 0.0,
        "last_total_ms": 0.0,
        "session_transitions": 0,
        "last_active_model": ACTIVE_ACTION_MODEL,
        "stabilizer": PredictionStabilizer(),
        "pose_tracker": SubjectPoseTracker(track_conf=TRACK_CONF_THRESHOLD),
        "session_stats": create_initial_session_stats(),
        "cmp_stabilizers": {
            "stgcn": PredictionStabilizer(),
            "lstm": PredictionStabilizer(),
        },
        "cmp_alert_sms": {
            "stgcn": AlertStateMachine(
                n_warn=N_WARN_FRAMES,
                n_critical=N_CRITICAL_FRAMES,
                n_recovery=N_RECOVERY_FRAMES,
                conf_threshold=CONF_THRESHOLD,
                warning_threshold=WARNING_THRESHOLD,
                critical_threshold=CRITICAL_THRESHOLD,
                instant_red_threshold=INSTANT_RED_THRESHOLD,
            ),
            "lstm": AlertStateMachine(
                n_warn=N_WARN_FRAMES,
                n_critical=N_CRITICAL_FRAMES,
                n_recovery=N_RECOVERY_FRAMES,
                conf_threshold=CONF_THRESHOLD,
                warning_threshold=WARNING_THRESHOLD,
                critical_threshold=CRITICAL_THRESHOLD,
                instant_red_threshold=INSTANT_RED_THRESHOLD,
            ),
        },
        "last_cmp_snapshot": None,
        "prev_small_gray": None,
        "prev_bbox_center": None,
        "scene_cuts": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _resolve_video_source(cfg: SidebarConfig) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the user's selected video source to `(source_id, filesystem_path)`.
    For uploaded files, writes a temporary file ONCE per `(name, size)` signature.
    """
    if cfg.source_mode == "upload":
        uf = cfg.uploaded_file
        if uf is None:
            return None, None
        file_sig = (uf.name, uf.size)
        if (
            st.session_state.get("upload_file_sig") != file_sig
            or not st.session_state.get("temp_upload_path")
            or not os.path.exists(st.session_state["temp_upload_path"])
        ):
            # Remove previous temp file before creating a new one
            _cleanup_video_capture(remove_temp_file=True)
            suffix = Path(uf.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="actionguard_") as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name
            st.session_state["temp_upload_path"] = tmp_path
            st.session_state["upload_file_sig"] = file_sig
        return f"upload:{uf.name}:{uf.size}", st.session_state["temp_upload_path"]

    # Demo clip mode
    if cfg.demo_video_path is not None and cfg.demo_video_path.exists():
        return f"demo:{cfg.demo_video_path}", str(cfg.demo_video_path)
    return None, None


def _open_and_reset_stream(
    source_id: str,
    video_path: str,
    pose_model: Any,
    cfg: SidebarConfig,
    keep_temp_file: bool = True,
) -> bool:
    """
    Open `video_path` with `cv2.VideoCapture`, compute 10-FPS timestamp resample
    indices, reset `SequenceBuffer` and `AlertStateMachine`, and decode frame 0
    for immediate UI preview.
    """
    _cleanup_video_capture(remove_temp_file=not keep_temp_file)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video file: `{video_path}`")
        return False

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0.0 or native_fps > 240.0:
        native_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total_frames > 0:
        indices, times = compute_resample_indices(
            total_frames=total_frames,
            native_fps=native_fps,
            sample_fps=float(TARGET_FPS),
        )
    else:
        indices = np.empty((0,), dtype=np.int64)
        times = np.empty((0,), dtype=np.float64)

    st.session_state["video_cap"] = cap
    st.session_state["cap_native_pos"] = -1
    st.session_state["last_raw_bgr"] = None
    st.session_state["active_source_id"] = source_id
    st.session_state["active_video_path"] = video_path
    st.session_state["native_fps"] = native_fps
    st.session_state["total_native_frames"] = total_frames
    st.session_state["resample_indices"] = indices
    st.session_state["resample_times"] = times
    st.session_state["tick_idx"] = 0
    st.session_state["stream_finished"] = False
    st.session_state["is_running"] = False

    # Reset stateful pipeline objects
    st.session_state["seq_buffer"].reset()
    st.session_state["alert_sm"].reset()
    st.session_state["stabilizer"].reset()
    for cmp_stab in st.session_state["cmp_stabilizers"].values():
        cmp_stab.reset()
    for cmp_sm in st.session_state["cmp_alert_sms"].values():
        cmp_sm.reset()
    st.session_state["last_cmp_snapshot"] = None
    pose_tracker: SubjectPoseTracker = st.session_state["pose_tracker"]
    pose_tracker.reset()
    pose_tracker.track_conf = float(cfg.track_conf_threshold)
    st.session_state["prev_small_gray"] = None
    st.session_state["prev_bbox_center"] = None
    st.session_state["scene_cuts"] = 0
    st.session_state["last_prediction"] = ActionPrediction(
        label="Buffering...",
        label_idx=-1,
        confidence=0.0,
        model_type=cfg.active_model,
    )
    st.session_state["last_stage1_ms"] = 0.0
    st.session_state["last_stage2_ms"] = 0.0
    st.session_state["last_total_ms"] = 0.0

    # Decode first frame immediately for preview before user clicks Run
    ret, frame0 = cap.read()
    if ret and frame0 is not None:
        st.session_state["cap_native_pos"] = 0
        st.session_state["last_raw_bgr"] = frame0
        pose_res = detect_and_extract_pose(
            frame0,
            pose_model,
            conf_threshold=cfg.conf_threshold,
            imgsz=cfg.yolo_imgsz,
            tracker=pose_tracker,
            track_conf_threshold=cfg.track_conf_threshold,
        )
        st.session_state["last_pose_result"] = pose_res
        vis_bgr = draw_pose(frame0, pose_res) if cfg.show_skeleton else frame0.copy()
        h_v, w_v = vis_bgr.shape[:2]
        if w_v > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / float(w_v)
            vis_bgr = cv2.resize(vis_bgr, (MAX_DISPLAY_WIDTH, int(h_v * scale)), interpolation=cv2.INTER_LINEAR)
        ok_jpg, jpg_buf = cv2.imencode(".jpg", vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        st.session_state["last_display_jpeg"] = jpg_buf.tobytes() if ok_jpg else None
        st.session_state["last_display_rgb"] = None if ok_jpg else cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    else:
        st.session_state["last_display_jpeg"] = None
        st.session_state["last_display_rgb"] = None
        st.session_state["last_pose_result"] = None

    return True


def _read_resampled_frame(tick_idx: int) -> Tuple[Optional[np.ndarray], int, float]:
    """
    Step sequentially through `video_cap` to reach the native frame corresponding
    to `tick_idx` at `TARGET_FPS = 10` Hz.

    Returns `(frame_bgr, target_native_idx, video_time_s)` or `(None, -1, 0.0)` on EOF.
    """
    cap = st.session_state.get("video_cap")
    if cap is None or not cap.isOpened():
        return None, -1, 0.0

    indices: np.ndarray = st.session_state["resample_indices"]
    times: np.ndarray = st.session_state["resample_times"]
    native_fps: float = st.session_state["native_fps"]

    if len(indices) > 0:
        if tick_idx >= len(indices):
            return None, -1, 0.0
        target_native_idx = int(indices[tick_idx])
        video_time_s = float(times[tick_idx])
    else:
        # Fallback for container formats that do not report CAP_PROP_FRAME_COUNT
        video_time_s = float(tick_idx) / float(TARGET_FPS)
        target_native_idx = int(np.floor(video_time_s * native_fps + 0.5))

    cap_pos: int = st.session_state["cap_native_pos"]
    frame_bgr: Optional[np.ndarray] = st.session_state["last_raw_bgr"]

    while cap_pos < target_native_idx:
        ret, next_frame = cap.read()
        if not ret or next_frame is None:
            return None, -1, video_time_s
        cap_pos += 1
        frame_bgr = next_frame

    st.session_state["cap_native_pos"] = cap_pos
    st.session_state["last_raw_bgr"] = frame_bgr
    return frame_bgr, target_native_idx, video_time_s


# ════════════════════════════════════════════════════════════════════════════
# 3. Single-Tick Pipeline Execution
# ════════════════════════════════════════════════════════════════════════════

def execute_single_pipeline_tick(
    pose_model: Any,
    action_models: Dict[str, Any],
    cfg: SidebarConfig,
    obj_model: Optional[Any] = None,
) -> bool:
    """
    Execute one 10-FPS sampled tick of the two-stage pipeline:
      Stage 1 (YOLO-Pose + Torso Normalization)
        -> SequenceBuffer (16 frames)
        -> Stage 2 (ST-GCN v2.4 or LSTM)
        -> AlertStateMachine (GREEN / YELLOW / RED + CSV transition log)

    Returns `True` if a tick was processed, or `False` if the video reached EOF.
    """
    tick_idx = int(st.session_state["tick_idx"])
    t_tick_start = time.perf_counter()

    frame_bgr, native_idx, video_time_s = _read_resampled_frame(tick_idx)
    if frame_bgr is None:
        st.session_state["is_running"] = False
        st.session_state["stream_finished"] = True
        return False

    # ── Stage 1: YOLO-Pose Detection & Keypoint Normalization ───────────────
    pose_tracker: SubjectPoseTracker = st.session_state["pose_tracker"]
    pose_tracker.track_conf = float(cfg.track_conf_threshold)
    t_s1 = time.perf_counter()
    pose_res: PoseResult = detect_and_extract_pose(
        frame_bgr,
        pose_model,
        conf_threshold=cfg.conf_threshold,
        imgsz=cfg.yolo_imgsz,
        tracker=pose_tracker,
        track_conf_threshold=cfg.track_conf_threshold,
    )
    stage1_ms = (time.perf_counter() - t_s1) * 1000.0

    seq_buf: SequenceBuffer = st.session_state["seq_buffer"]
    stabilizer: PredictionStabilizer = st.session_state["stabilizer"]

    # ── Scene-cut / person-jump guard (avoids fake motion spikes in ST-GCN) ──
    small_gray = cv2.cvtColor(cv2.resize(frame_bgr, (32, 24)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    cut_detected = False
    if cfg.scene_cut_guard:
        prev_gray = st.session_state.get("prev_small_gray")
        if prev_gray is not None and float(np.mean(np.abs(small_gray - prev_gray))) > SCENE_CUT_GRAY_DIFF:
            cut_detected = True
        if pose_res.person_detected and pose_res.bbox_xyxy is not None:
            bx = np.asarray(pose_res.bbox_xyxy, dtype=np.float64).reshape(-1)
            center = ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0)
            prev_c = st.session_state.get("prev_bbox_center")
            if prev_c is not None:
                jump = float(np.hypot(center[0] - prev_c[0], center[1] - prev_c[1]))
                if jump / float(max(frame_bgr.shape[1], 1)) > PERSON_JUMP_FRAC:
                    cut_detected = True
            st.session_state["prev_bbox_center"] = center
    st.session_state["prev_small_gray"] = small_gray
    if cut_detected:
        seq_buf.reset()
        stabilizer.reset()
        for cmp_stab in st.session_state["cmp_stabilizers"].values():
            cmp_stab.reset()
        pose_tracker.reset()
        st.session_state["scene_cuts"] += 1

    # ── SequenceBuffer Update (16 frames @ 10 FPS) ──────────────────────────
    seq_buf.append(pose_res.keypoints_norm)

    # ── Stage 2: Action Classification (ST-GCN / LSTM / Ensemble + EMA + Idle gate + Stage-2B)
    t_s2 = time.perf_counter()
    prediction: ActionPrediction = stabilizer.step(
        seq_buf,
        action_models,
        cfg.active_model,
        ema_strength=cfg.ema_strength,
        idle_gate=cfg.idle_gate,
        frame_bgr=frame_bgr,
        pose_result=pose_res,
        object_model=obj_model,
        enable_object_context=cfg.enable_object_context,
    )
    stage2_ms = (time.perf_counter() - t_s2) * 1000.0

    # ── Stage 3: Alert State Machine & Event Logging ────────────────────────
    alert_sm: AlertStateMachine = st.session_state["alert_sm"]
    prev_level: AlertLevel = alert_sm.level
    new_level: AlertLevel = alert_sm.update(prediction)

    if new_level != prev_level:
        note = "RED latched (Critical Fall)" if new_level == AlertLevel.RED else "State transition"
        log_event(
            source=cfg.source_label,
            model=cfg.active_model,
            prev_level=prev_level.name,
            new_level=new_level.name,
            action_label=prediction.label,
            confidence=prediction.confidence,
            frame_idx=native_idx,
            video_time_s=video_time_s,
            note=note,
        )
        st.session_state["session_transitions"] += 1

    update_session_stats(
        st.session_state["session_stats"],
        prev_level,
        new_level,
        cfg.active_model,
    )
    if cfg.compare_models:
        st.session_state["last_cmp_snapshot"] = compute_comparison_snapshot(
            seq_buf,
            action_models,
            st.session_state["cmp_stabilizers"],
            st.session_state["cmp_alert_sms"],
            ema_strength=cfg.ema_strength,
            idle_gate=cfg.idle_gate,
            frame_bgr=frame_bgr,
            pose_result=pose_res,
            object_model=obj_model,
            enable_object_context=cfg.enable_object_context,
        )

    # ── Visualization Overlay (pre-encoded JPEG bytes for fast st.image) ────
    vis_bgr = draw_pose(frame_bgr, pose_res) if cfg.show_skeleton else frame_bgr.copy()
    h_vis, w_vis = vis_bgr.shape[:2]
    if w_vis > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / float(w_vis)
        vis_bgr = cv2.resize(vis_bgr, (MAX_DISPLAY_WIDTH, int(h_vis * scale)), interpolation=cv2.INTER_LINEAR)
    vis_bgr = annotate_frame(vis_bgr, prediction.label, prediction.confidence, new_level.name)
    ok_jpg, jpg_buf = cv2.imencode(".jpg", vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    display_jpeg = jpg_buf.tobytes() if ok_jpg else None
    display_rgb = None if ok_jpg else cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    total_ms = (time.perf_counter() - t_tick_start) * 1000.0

    st.session_state["last_pose_result"] = pose_res
    st.session_state["last_prediction"] = prediction
    st.session_state["last_display_jpeg"] = display_jpeg
    st.session_state["last_display_rgb"] = display_rgb
    st.session_state["last_stage1_ms"] = stage1_ms
    st.session_state["last_stage2_ms"] = stage2_ms
    st.session_state["last_total_ms"] = total_ms
    st.session_state["tick_idx"] = tick_idx + 1

    # Optional 10-FPS real-time pacing (100 ms per sampled tick)
    if cfg.pace_realtime:
        elapsed_s = time.perf_counter() - t_tick_start
        target_period_s = 1.0 / float(TARGET_FPS)
        if elapsed_s < target_period_s:
            time.sleep(target_period_s - elapsed_s)

    return True


# ════════════════════════════════════════════════════════════════════════════
# 4. UI Rendering Helpers (Badges, Status Cards, Comparison & Summary)
# ════════════════════════════════════════════════════════════════════════════

def _format_action_label(raw_label: str) -> str:
    return {
        "Unknown": "Analyzing…",
        "No_Person": "No person",
        "Low_Pose_Quality": "Low pose quality",
    }.get(raw_label, raw_label.replace("_", " "))


def _render_status(alert_sm: AlertStateMachine, pred: ActionPrediction) -> None:
    """Polished status card: badged alert banner + action + confidence bar + metadata."""
    if alert_sm.level == AlertLevel.RED:
        bg, accent, icon, title = "#b91c1c", "#fca5a5", "🚨", "FALL ALERT"
        pill = "LATCHED" if alert_sm.is_red_latched else "RECOVERING"
        hint = "Click “Acknowledge” in the sidebar" if alert_sm.is_red_latched else "Acknowledged — awaiting GREEN frames"
    elif alert_sm.level == AlertLevel.YELLOW:
        bg, accent, icon, title = "#b45309", "#fcd34d", "⚠️", "WARNING"
        pill = "POSTURE RISK"
        hint = "Risky posture detected (Crouching / Lying)"
    else:
        bg, accent, icon, title = "#15803d", "#86efac", "✅", "NORMAL"
        pill = "MONITORING"
        hint = "No fall risk detected"

    label = _format_action_label(pred.label)
    conf_pct = max(0, min(100, int(round(pred.confidence * 100)))) if pred.confidence > 0 else 0
    conf_text = f"{conf_pct}%" if pred.confidence > 0 else "—"

    st.markdown(
        f"""
        <div style="background:{bg};border-left:5px solid {accent};color:#fff;border-radius:10px;padding:12px 16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:1.25rem;font-weight:800;">{icon} {title}</span>
            <span style="background:rgba(255,255,255,.20);border-radius:999px;padding:2px 9px;font-size:0.7rem;font-weight:700;letter-spacing:.04em;">{pill}</span>
          </div>
          <div style="font-size:0.84rem;opacity:.92;margin-top:3px;">{hint}</div>
        </div>
        <div style="margin-top:12px;font-size:0.78rem;opacity:.7;text-transform:uppercase;letter-spacing:.04em;">Detected Action</div>
        <div style="font-size:1.75rem;font-weight:750;line-height:1.15;">{label}</div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
          <div style="flex:1;height:6px;border-radius:999px;background:rgba(128,128,128,.22);overflow:hidden;">
            <div style="width:{conf_pct}%;height:100%;background:{bg};"></div>
          </div>
          <span style="font-size:0.85rem;font-weight:600;opacity:.85;">{conf_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_comparison(snapshot: Optional[Dict[str, Any]]) -> None:
    """Render side-by-side ST-GCN v3.1-C vs. LSTM Baseline live predictions."""
    if snapshot is None:
        st.caption("Live comparison: waiting for 16-frame buffer or next tick.")
        return

    level_colors = {"GREEN": "#15803d", "YELLOW": "#b45309", "RED": "#b91c1c"}
    c1, c2 = st.columns(2)
    for col, key, title in (
        (c1, "stgcn", "ST-GCN v3.1-C"),
        (c2, "lstm", "LSTM Baseline"),
    ):
        entry = snapshot[key]
        pred_m: ActionPrediction = entry["prediction"]
        lvl_name = entry["level"].name
        color = level_colors.get(lvl_name, "#15803d")
        lbl_str = _format_action_label(pred_m.label)
        conf_str = f"{pred_m.confidence * 100:.0f}%" if pred_m.confidence > 0 else "—"
        lat_str = f"{entry['latency_ms']:.1f} ms"
        col.markdown(
            f"""
            <div style="border:1px solid rgba(128,128,128,.28);border-top:4px solid {color};border-radius:8px;padding:8px 10px;">
              <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.78rem;opacity:.8;">
                <strong>{title}</strong>
                <span style="color:{color};font-weight:700;">{lvl_name}</span>
              </div>
              <div style="font-size:1.1rem;font-weight:700;margin-top:2px;">{lbl_str}</div>
              <div style="font-size:0.78rem;opacity:.75;">conf {conf_str} · {lat_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    agree_text = "✅ Agree" if snapshot.get("agree") else "⚡ Diverged"
    st.caption(f"Side-by-side status on shared 16-frame window: **{agree_text}**")


def _render_session_summary(stats: Dict[str, Any], total_ms: float = 0.0) -> None:
    """Render cumulative per-session alert counts, dwell times, and live throughput."""
    t_map: Dict[str, float] = stats["time_in_level_s"]
    fps = (1000.0 / total_ms) if total_ms > 0 else 0.0
    speed_line = f"{fps:.1f} FPS ({total_ms:.0f} ms/tick)" if total_ms > 0 else "Ready"

    st.markdown(
        f"""
        <div style="border:1px solid rgba(128,128,128,.25);border-radius:8px;padding:10px 12px;margin-top:8px;">
          <div style="font-size:0.78rem;font-weight:700;opacity:.8;margin-bottom:4px;">SESSION SUMMARY</div>
          <div style="font-size:0.86rem;">
            Alerts: <strong>{int(stats['red_alerts'])}</strong> RED ·
            <strong>{int(stats['yellow_alerts'])}</strong> WARN
            <span style="opacity:.7;">({int(stats['total_ticks'])} ticks)</span>
          </div>
          <div style="font-size:0.80rem;opacity:.8;margin-top:2px;">
            Dwell — GREEN {t_map['GREEN']:.1f}s · YELLOW {t_map['YELLOW']:.1f}s · RED {t_map['RED']:.1f}s
          </div>
          <div style="font-size:0.76rem;opacity:.72;margin-top:2px;">Throughput: {speed_line}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="ActionGuard AI — Fall Detection MVP",
        page_icon="🛡️",
        layout="wide",
    )

    init_session_state()
    pose_model, action_models = load_all_models()
    obj_model = load_cached_object_model()
    alert_sm: AlertStateMachine = st.session_state["alert_sm"]
    seq_buf: SequenceBuffer = st.session_state["seq_buffer"]

    # Render sidebar controls & capture user configuration
    cfg = render_sidebar(
        is_running=bool(st.session_state["is_running"]),
        is_red_latched=bool(alert_sm.is_red_latched),
    )

    # Sync session-isolated threshold sliders into AlertStateMachine instances
    thresh_kw: Dict[str, Any] = dict(
        conf_threshold=cfg.conf_threshold,
        warning_threshold=cfg.warning_threshold,
        critical_threshold=cfg.critical_threshold,
        instant_red_threshold=cfg.instant_red_threshold,
        n_warn=cfg.n_warn_frames,
        n_critical=cfg.n_critical_frames,
        n_recovery=cfg.n_recovery_frames,
    )
    alert_sm.set_thresholds(**thresh_kw)
    for cmp_sm in st.session_state["cmp_alert_sms"].values():
        cmp_sm.set_thresholds(**thresh_kw)

    if cfg.clear_log_clicked:
        st.session_state["session_stats"] = create_initial_session_stats()

    # Log live model hot-swap if user toggled ST-GCN <-> LSTM <-> Ensemble while active
    if st.session_state["last_active_model"] != cfg.active_model:
        st.session_state["last_active_model"] = cfg.active_model
        st.session_state["stabilizer"].reset()
        # Re-evaluate current buffer immediately so UI reflects the new model right away
        if seq_buf.is_ready():
            st.session_state["last_prediction"] = st.session_state["stabilizer"].step(
                seq_buf,
                action_models,
                cfg.active_model,
                ema_strength=cfg.ema_strength,
                idle_gate=cfg.idle_gate,
                frame_bgr=st.session_state.get("last_raw_bgr"),
                pose_result=st.session_state.get("last_pose_result"),
                object_model=obj_model,
                enable_object_context=cfg.enable_object_context,
            )

    if cfg.compare_models and seq_buf.is_ready() and st.session_state.get("last_cmp_snapshot") is None:
        st.session_state["last_cmp_snapshot"] = compute_comparison_snapshot(
            seq_buf,
            action_models,
            st.session_state["cmp_stabilizers"],
            st.session_state["cmp_alert_sms"],
            ema_strength=cfg.ema_strength,
            idle_gate=cfg.idle_gate,
            frame_bgr=st.session_state.get("last_raw_bgr"),
            pose_result=st.session_state.get("last_pose_result"),
            object_model=obj_model,
            enable_object_context=cfg.enable_object_context,
        )

    # Resolve video source & auto-reset stream when source changes
    source_id, video_path = _resolve_video_source(cfg)
    if source_id != st.session_state["active_source_id"]:
        if source_id is not None and video_path is not None:
            _open_and_reset_stream(
                source_id=source_id,
                video_path=video_path,
                pose_model=pose_model,
                cfg=cfg,
                keep_temp_file=(cfg.source_mode == "upload"),
            )
        else:
            _cleanup_video_capture(remove_temp_file=True)
            st.session_state["active_source_id"] = None
            st.session_state["active_video_path"] = None
            st.session_state["last_display_jpeg"] = None
            st.session_state["last_display_rgb"] = None
            st.session_state["last_cmp_snapshot"] = None
            st.session_state["is_running"] = False

    # Handle playback & alert buttons
    if cfg.restart_clicked and source_id is not None and video_path is not None:
        _open_and_reset_stream(
            source_id=source_id,
            video_path=video_path,
            pose_model=pose_model,
            cfg=cfg,
            keep_temp_file=(cfg.source_mode == "upload"),
        )

    if cfg.acknowledge_clicked and alert_sm.is_red_latched:
        prev_lvl = alert_sm.level
        alert_sm.acknowledge()
        for cmp_sm in st.session_state["cmp_alert_sms"].values():
            if cmp_sm.is_red_latched:
                cmp_sm.acknowledge()
        pred_now: ActionPrediction = st.session_state["last_prediction"]
        log_event(
            source=cfg.source_label,
            model=cfg.active_model,
            prev_level=prev_lvl.name,
            new_level="RED_UNLATCHED",
            action_label=pred_now.label,
            confidence=pred_now.confidence,
            frame_idx=max(0, int(st.session_state["cap_native_pos"])),
            video_time_s=float(st.session_state["tick_idx"]) / float(TARGET_FPS),
            note="Operator acknowledged RED alert (unlatched; awaiting recovery to GREEN)",
        )
        st.session_state["session_transitions"] += 1

    if cfg.start_clicked and source_id is not None:
        if st.session_state["stream_finished"] and video_path is not None:
            _open_and_reset_stream(
                source_id=source_id,
                video_path=video_path,
                pose_model=pose_model,
                cfg=cfg,
                keep_temp_file=(cfg.source_mode == "upload"),
            )
        st.session_state["is_running"] = True

    if cfg.pause_clicked:
        st.session_state["is_running"] = False

    # Button clicks change state that the (already drawn) sidebar must reflect -> one rerun
    if cfg.start_clicked or cfg.pause_clicked or cfg.restart_clicked or cfg.acknowledge_clicked:
        st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # Main view: big video with burned-in label on the left, compact status on the right
    # ════════════════════════════════════════════════════════════════════════
    col_video, col_info = st.columns([2.3, 1], gap="large")
    with col_video:
        video_ph = st.empty()
        progress_ph = st.empty()
    with col_info:
        status_ph = st.empty()
        summary_ph = st.empty()
        st.markdown("**Recent alerts**")
        events_ph = st.empty()

    def refresh(refresh_events: bool, refresh_secondary: bool = True) -> None:
        pred: ActionPrediction = st.session_state["last_prediction"]

        with status_ph.container():
            _render_status(alert_sm, pred)

        display_img = st.session_state.get("last_display_jpeg") or st.session_state.get("last_display_rgb")
        with video_ph.container():
            if display_img is not None:
                st.image(display_img, use_container_width=True)
            else:
                st.info("Upload a video or select a dataset clip in the sidebar, then press ▶ Run.")

        if refresh_secondary:
            with summary_ph.container():
                _render_session_summary(
                    st.session_state["session_stats"],
                    float(st.session_state["last_total_ms"]),
                )

            total_ticks = len(st.session_state["resample_indices"])
            with progress_ph.container():
                if total_ticks > 0:
                    curr = min(int(st.session_state["tick_idx"]), total_ticks)
                    st.progress(curr / float(total_ticks))
                if st.session_state["stream_finished"]:
                    st.success("Finished. Press ▶ Run or ↺ Reset to play again.")

        if refresh_events:
            with events_ph.container():
                rows = read_recent_events(limit=5)
                if rows:
                    st.dataframe(
                        [
                            {
                                "time": r["timestamp"][11:19],
                                "alert": f"{r['prev_level']} → {r['new_level']}",
                                "action": r["action_label"],
                                "conf": f"{float(r['confidence']) * 100:.0f}%",
                            }
                            for r in rows
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No alerts yet.")

    refresh(True, True)

    # In-place streaming loop: widget interaction interrupts the script and reruns it,
    # so Pause / Reset / Acknowledge / model switch / sliders stay responsive.
    ran_loop = False
    while st.session_state["is_running"] and source_id is not None:
        ran_loop = True
        before = st.session_state["session_transitions"]
        ok = execute_single_pipeline_tick(
            pose_model=pose_model,
            action_models=action_models,
            cfg=cfg,
            obj_model=obj_model,
        )
        cur_tick = int(st.session_state["tick_idx"])
        changed = (st.session_state["session_transitions"] != before) or not ok
        refresh(
            refresh_events=changed,
            refresh_secondary=changed or (cur_tick % 3 == 0),
        )
        if not ok:
            break

    if ran_loop:
        st.rerun()  # one final rerun so sidebar buttons reflect the stopped state


if __name__ == "__main__":
    main()
