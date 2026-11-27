"""
cli_pipeline_test.py — End-to-end pipeline test & stage-by-stage latency profiler (no Streamlit).

Runs the full pipeline over a test video:
  video → detector → sequence_buffer → action_classifier → alert_state

Prints state transitions and a per-stage latency breakdown to console.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import (
    WINDOW_SIZE, TARGET_FPS, YOLO_POSE_MODEL_PATH,
    NUM_JOINTS, NUM_COORDS,
)
from utils.skeleton_utils import load_pose_model, compute_resample_indices
from pipeline.detector import detect_and_extract_pose
from pipeline.sequence_buffer import SequenceBuffer
from pipeline.action_classifier import (
    ActionLSTM, ActionSTGCN, ModelType, classify_sequence, load_action_model,
)
from pipeline.alert_state import AlertStateMachine, AlertLevel


def run_pipeline(
    video_path: str,
    model_type: ModelType = "stgcn",
    max_frames: int = 300,
) -> None:
    device = torch.device("cpu")
    print(f"\n{'=' * 72}")
    print("End-to-End CLI Pipeline Test & Latency Profiler")
    print(f"  Video      : {video_path}")
    print(f"  Model type : {model_type}")
    print(f"{'=' * 72}\n")

    # ── Load models ──────────────────────────────────────────────────────────
    print("[1] Loading YOLO-Pose model...")
    pose_model = load_pose_model(str(PROJECT_ROOT / YOLO_POSE_MODEL_PATH))

    print(f"[2] Loading action model ({model_type})...")
    action_model = load_action_model(
        str(PROJECT_ROOT / "models" / f"action_{model_type}.pth"),
        model_type, device,
    )
    models: dict = {model_type: action_model}
    if model_type == "stgcn":
        wrist_path = PROJECT_ROOT / "models" / "action_stgcn_v4_b.pth"
        if wrist_path.is_file():
            models["stgcn_wrist"] = load_action_model(str(wrist_path), "stgcn", device)
    label_map = None

    # ── Setup buffer and state machine ───────────────────────────────────────
    buffer = SequenceBuffer(WINDOW_SIZE)
    sm     = AlertStateMachine()

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames
    sampled_indices_arr, _ = compute_resample_indices(limit_frames, native_fps, sample_fps=TARGET_FPS)
    sampled_set = set(int(i) for i in sampled_indices_arr)

    print(f"[3] Video: native_fps={native_fps:.1f}, target_fps={TARGET_FPS}, "
          f"total_frames={total_frames}, sampled_ticks={len(sampled_set)}\n")
    print(f"{'Tick':>5} | {'Person':>6} | {'DetConf':>7} | {'Buffer':>7} | {'PredLabel':>10} | {'PredConf':>8} | Alert")
    print("-" * 72)

    frame_idx  = 0
    processed  = 0
    prev_alert = AlertLevel.GREEN

    # Per-stage latency trackers (ms)
    t_stage1_list: list[float] = []
    t_buffer_list: list[float] = []
    t_stage2_list: list[float] = []
    t_alert_list:  list[float] = []

    while frame_idx < limit_frames:
        ret, frame = cap.read()
        if not ret:
            break

        cur_idx = frame_idx
        frame_idx += 1
        if cur_idx not in sampled_set:
            continue

        processed += 1

        # Stage 1: Pose detection + keypoint normalisation
        t0 = time.perf_counter()
        result = detect_and_extract_pose(frame, pose_model)
        t1 = time.perf_counter()

        # Buffer update
        buffer.append(result.keypoints_norm)
        t2 = time.perf_counter()

        # Stage 2: Action classification (when buffer ready)
        label_str = "Buffering"
        pred_conf = 0.0
        alert_now = sm.level

        if buffer.is_ready():
            pred = classify_sequence(
                buffer, models, model_type, device=device, label_map=label_map
            )
            t3 = time.perf_counter()
            alert_now = sm.update(pred)
            t4 = time.perf_counter()

            label_str = pred.label
            pred_conf = pred.confidence
            t_stage2_list.append((t3 - t2) * 1000.0)
            t_alert_list.append((t4 - t3) * 1000.0)
        else:
            t3 = t2

        # Skip tick 1 warm-up when recording steady-state Stage 1 latency
        if processed > 1:
            t_stage1_list.append((t1 - t0) * 1000.0)
            t_buffer_list.append((t2 - t1) * 1000.0)

        person_str = "YES" if (result.person_detected and result.keypoints_norm is not None) else "NO"
        buf_str    = f"{len(buffer)}/{WINDOW_SIZE}"
        alert_str  = alert_now.name

        transition = ""
        if alert_now != prev_alert:
            transition = f"  <-- TRANSITION: {prev_alert.name} -> {alert_now.name}"
            prev_alert = alert_now

        print(f"{processed:>5} | {person_str:>6} | {result.detection_conf:>7.2f} | "
              f"{buf_str:>7} | {label_str:>10} | {pred_conf:>8.2f} | {alert_str}{transition}")

    cap.release()

    # ── Stage-by-stage latency report ────────────────────────────────────────
    m_s1  = float(np.mean(t_stage1_list)) if t_stage1_list else 0.0
    m_buf = float(np.mean(t_buffer_list)) if t_buffer_list else 0.0
    m_s2  = float(np.mean(t_stage2_list)) if t_stage2_list else 0.0
    m_al  = float(np.mean(t_alert_list))  if t_alert_list  else 0.0
    total_ms = m_s1 + m_buf + m_s2 + m_al
    steady_fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    print(f"\n{'=' * 72}")
    print("Stage-by-Stage Latency Breakdown (Steady-State CPU Mean per Tick):")
    print(f"  1. Stage 1 (YOLO-Pose + Normalise) : {m_s1:6.2f} ms")
    print(f"  2. SequenceBuffer Append           : {m_buf:6.3f} ms")
    print(f"  3. Stage 2 ({model_type.upper():5s} Inference)       : {m_s2:6.2f} ms")
    print(f"  4. AlertStateMachine Update        : {m_al:6.3f} ms")
    print(f"  ----------------------------------------------------")
    print(f"  Total Pipeline Latency / Tick      : {total_ms:6.2f} ms  (~{steady_fps:.1f} FPS)")
    print(f"  Final Alert Level                  : {sm.level.name} (red_latched={sm.is_red_latched})")
    print(f"{'=' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI end-to-end pipeline test")
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw_videos" / "Fall" / "video (1)_fall.avi"),
    )
    parser.add_argument("--model", type=str, default="stgcn", choices=["lstm", "stgcn"])
    parser.add_argument("--maxframes", type=int, default=150)
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        model_type=cast(ModelType, args.model),
        max_frames=args.maxframes,
    )


if __name__ == "__main__":
    main()
