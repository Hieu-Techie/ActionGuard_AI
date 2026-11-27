"""
sidebar.py — Streamlit sidebar for the Video-Upload MVP.

Main controls stay visible (source, model, Run/Pause/Reset, Acknowledge);
everything else is tucked into two collapsed expanders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import streamlit as st

from utils.config import (
    ACTIVE_ACTION_MODEL,
    CLASS_NAMES,
    CONF_THRESHOLD,
    TRACK_CONF_THRESHOLD,
    CRITICAL_THRESHOLD,
    INSTANT_RED_THRESHOLD,
    N_CRITICAL_FRAMES,
    N_RECOVERY_FRAMES,
    N_WARN_FRAMES,
    WARNING_THRESHOLD,
)
from utils.logger import clear_events_log, export_events_csv_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw_videos"
DEFAULT_YOLO_IMGSZ: int = 640

MODEL_NAMES: Dict[str, str] = {
    "stgcn": "ST-GCN v4 Dual-Specialist (default)",
    "lstm": "LSTM baseline",
    "ensemble": "Ensemble (ST-GCN + LSTM)",
}

# Only numbers that were measured on the held-out Test / Sealed split.
MODEL_NOTES: Dict[str, str] = {
    "stgcn": "All-Test Video acc 81.6% (Macro-F1 0.806, Win acc 74.1%) · RED FPR 0.8% (1/125)",
    "lstm": "V1 Test acc 66.6% (RED 10/13) · All-Test win acc 46.0% · CPU 1.4 ms",
    "ensemble": "V1 Test acc 78.3% · All-Test win acc 59.5% · Lowest RED FPR 2.4% (3/125)",
}


@dataclass
class SidebarConfig:
    """Snapshot of all user selections from the sidebar."""
    source_mode: Literal["upload", "demo"]
    uploaded_file: Optional[Any]
    demo_video_path: Optional[Path]
    source_label: str
    active_model: str
    show_skeleton: bool
    pace_realtime: bool
    conf_threshold: float
    warning_threshold: float
    critical_threshold: float
    instant_red_threshold: float
    n_warn_frames: int
    n_critical_frames: int
    n_recovery_frames: int
    start_clicked: bool
    pause_clicked: bool
    restart_clicked: bool
    acknowledge_clicked: bool
    clear_log_clicked: bool
    yolo_imgsz: int = DEFAULT_YOLO_IMGSZ
    ema_strength: float = 0.5
    idle_gate: float = 0.45
    scene_cut_guard: bool = True
    enable_object_context: bool = True
    track_conf_threshold: float = float(TRACK_CONF_THRESHOLD)
    compare_models: bool = False


@st.cache_data(show_spinner=False)
def _discover_demo_clips() -> Dict[str, List[Path]]:
    """Scan `data/raw_videos/<Class>/` for demo video files."""
    clips: Dict[str, List[Path]] = {}
    if not RAW_VIDEOS_DIR.exists():
        return clips
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    priority = ["Fall", "Crouching", "Lying", "Walk", "Sit_Down", "Stand_Up", "Eat", "Drink"]
    ordered = [c for c in priority if c in CLASS_NAMES] + [c for c in CLASS_NAMES if c not in priority]
    for cls in ordered:
        d = RAW_VIDEOS_DIR / cls
        if not d.is_dir():
            continue
        files = sorted((p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts),
                       key=lambda p: p.name)
        if files:
            clips[cls] = files
    return clips


_THRESHOLD_DEFAULTS = {
    "ui_yolo_imgsz": int(DEFAULT_YOLO_IMGSZ),
    "ui_conf_threshold": float(CONF_THRESHOLD),
    "ui_warning_threshold": float(WARNING_THRESHOLD),
    "ui_critical_threshold": float(CRITICAL_THRESHOLD),
    "ui_instant_red_threshold": float(INSTANT_RED_THRESHOLD),
}


def _restore_threshold_defaults() -> None:
    for k, v in _THRESHOLD_DEFAULTS.items():
        st.session_state[k] = v


def render_sidebar(*, is_running: bool, is_red_latched: bool) -> SidebarConfig:
    """Render the sidebar and return the user's selections."""
    for k, v in _THRESHOLD_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.markdown("### 🛡️ ActionGuard AI")

        # ── Video source ─────────────────────────────────────────────────────
        clips = _discover_demo_clips()
        if clips:
            choice = st.radio(
                "Video source",
                ["Upload a video", "Dataset clip"],
                key="ui_source_mode_radio",
                horizontal=True,
            )
        else:
            choice = "Upload a video"

        uploaded_file = None
        demo_video_path: Optional[Path] = None
        source_label = "No video selected"
        source_mode: Literal["upload", "demo"] = "upload"

        if choice == "Upload a video":
            uploaded_file = st.file_uploader(
                "Video file", type=["mp4", "avi", "mov", "mkv"],
                key="ui_video_uploader", label_visibility="collapsed",
            )
            if uploaded_file is not None:
                source_label = f"upload:{uploaded_file.name}"
        else:
            source_mode = "demo"
            cls = st.selectbox("Category", list(clips.keys()), key="ui_demo_class_select")
            files = clips[cls]
            preferred = {"video (122)_fall.mp4", "003.mp4", "video (1)_fall.avi"}
            default_idx = next((i for i, p in enumerate(files) if p.name in preferred), 0)
            name = st.selectbox("Clip", [p.name for p in files], index=default_idx,
                                key=f"ui_demo_clip_select_{cls}")
            demo_video_path = next((p for p in files if p.name == name), None)
            if demo_video_path is not None:
                source_label = f"demo:{cls}/{demo_video_path.name}"

        # ── Model ────────────────────────────────────────────────────────────
        keys = list(MODEL_NAMES.keys())
        active_model: str = st.radio(
            "Model", keys, index=0 if ACTIVE_ACTION_MODEL == "stgcn" else 1,
            format_func=lambda k: MODEL_NAMES[k], key="ui_active_model_radio",
        )
        st.caption(MODEL_NOTES[active_model])
        compare_models = False

        # ── Controls ─────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        start_clicked = c1.button("▶ Run", use_container_width=True, type="primary", disabled=is_running)
        pause_clicked = c2.button("⏸ Pause", use_container_width=True, disabled=not is_running)
        restart_clicked = c3.button("↺ Reset", use_container_width=True)

        acknowledge_clicked = False
        if is_red_latched:
            acknowledge_clicked = st.button("🚨 Acknowledge fall alert", use_container_width=True, type="primary")

        show_skeleton = st.toggle("Show skeleton", value=True, key="ui_show_skeleton")

        # ── Advanced (collapsed) ─────────────────────────────────────────────
        with st.expander("Advanced settings"):
            yolo_imgsz = st.select_slider(
                "Pose input size", options=[384, 480, 640], value=DEFAULT_YOLO_IMGSZ, key="ui_yolo_imgsz",
                help="640 matches training resolution; 480/384 are faster on large-subject HD clips.",
            )
            ema_strength = st.slider(
                "Prediction smoothing", 0.0, 0.9, 0.5, 0.1, key="ui_ema_strength",
                help="0 = raw output. Higher = steadier labels, slower reaction.",
            )
            idle_gate = st.slider(
                "Min confidence to show an action", 0.20, 0.80, 0.45, 0.05, key="ui_idle_gate",
                help="Below this, the label becomes 'Idle'.",
            )
            enable_object_context = st.toggle(
                "Hand-object context (Eat / Drink)",
                value=True,
                key="ui_enable_obj_ctx",
                help="Checks hand/head ROI for cup, bottle, or bowl when Eat/Drink is active.",
            )

            st.markdown("**Alert thresholds**")
            conf_threshold = st.slider("Pose confidence", 0.10, 0.95, step=0.05, key="ui_conf_threshold")
            warning_threshold = st.slider("Warning (yellow)", 0.20, 0.95, step=0.05, key="ui_warning_threshold")
            critical_threshold = st.slider("Fall (red)", 0.30, 0.99, step=0.05, key="ui_critical_threshold")
            instant_red_threshold = st.slider("Instant red", 0.50, 0.99, step=0.01, key="ui_instant_red_threshold")
            st.button("Restore defaults", on_click=_restore_threshold_defaults, use_container_width=True)

        scene_cut_guard = True
        pace_realtime = False
        track_conf_threshold = float(TRACK_CONF_THRESHOLD)
        n_warn_frames = int(N_WARN_FRAMES)
        n_critical_frames = int(N_CRITICAL_FRAMES)
        n_recovery_frames = int(N_RECOVERY_FRAMES)

        # ── Event log (collapsed) ────────────────────────────────────────────
        clear_log_clicked = False
        with st.expander("Event log"):
            st.download_button("Download CSV", data=export_events_csv_bytes(),
                               file_name="actionguard_events.csv", mime="text/csv",
                               use_container_width=True)
            clear_log_clicked = st.button("Clear log", use_container_width=True)
            if clear_log_clicked:
                clear_events_log()

    return SidebarConfig(
        source_mode=source_mode,
        uploaded_file=uploaded_file,
        demo_video_path=demo_video_path,
        source_label=source_label,
        active_model=active_model,
        show_skeleton=bool(show_skeleton),
        pace_realtime=bool(pace_realtime),
        conf_threshold=float(conf_threshold),
        warning_threshold=float(warning_threshold),
        critical_threshold=float(critical_threshold),
        instant_red_threshold=float(instant_red_threshold),
        n_warn_frames=int(n_warn_frames),
        n_critical_frames=int(n_critical_frames),
        n_recovery_frames=int(n_recovery_frames),
        start_clicked=bool(start_clicked),
        pause_clicked=bool(pause_clicked),
        restart_clicked=bool(restart_clicked),
        acknowledge_clicked=bool(acknowledge_clicked),
        clear_log_clicked=bool(clear_log_clicked),
        yolo_imgsz=int(yolo_imgsz),
        ema_strength=float(ema_strength),
        idle_gate=float(idle_gate),
        scene_cut_guard=bool(scene_cut_guard),
        enable_object_context=bool(enable_object_context),
        track_conf_threshold=float(track_conf_threshold),
        compare_models=bool(compare_models),
    )
