# ActionGuard AI — Two-Stage Human Action Recognition & Fall Detection

ActionGuard AI is a single-process **Streamlit** web application and **PyTorch + YOLOv8-Pose** pipeline for human action recognition and fall detection across 8 classes (`Walk`, `Eat`, `Drink`, `Fall`, `Lying`, `Sit_Down`, `Stand_Up`, `Crouching`), mapping recognized actions to a latched 3-level safety alert state machine (`GREEN` / `YELLOW` / `RED`).

> 📘 **Tài liệu tổng hợp & hướng dẫn làm báo cáo:** Xem chi tiết toàn bộ kiến trúc, phương pháp và bảng số liệu tại [`BAO_CAO_TONG_HOP_DU_AN.md`](BAO_CAO_TONG_HOP_DU_AN.md).

---

## 1. Architecture Overview

```text
Video Source (Upload Video / Local Dataset Clip)
  │  timestamp-resampled to 10 FPS
  ▼
Stage 1 — Pose & Tracking (`src/pipeline/detector.py`, `src/utils/skeleton_utils.py`)
  • YOLOv8n-pose (COCO 17 joints) + SubjectPoseTracker hysteresis (`track_conf=0.30`)
  • Invalid `(0,0)` joint cleanup + Virtual Hip Fallback (`conf=0.25` for upper-body views)
  ▼
SequenceBuffer (`src/pipeline/sequence_buffer.py`)
  • 16-frame rolling window (`1.6s` @ 10 FPS), zero-pads gaps <= 3 ticks, flushes > 3 ticks
  ▼
Stage 2 — Action Classification & Stabilization
  • Selectable backbone: `ST-GCN v4 Dual-Specialist` (default: `v4-C` Lower-Body + `v4-B` Wrist-to-Head), `LSTM Baseline`, or `Ensemble`
  • Physical Stabilizer (`src/pipeline/stabilizer.py`): pose quality check, LSTM occlusion
    fallback, late-window Seated-vs-Lying guard, Fast-Attack Fall Passthrough (>= 0.70), EMA smoothing, and Idle gating
  • Conditional Stage-2B Hand-Object Context (`src/pipeline/object_context.py`):
    upper-ROI YOLOv8n (`yolo_obj_pretrained.pt`, cached every 3 ticks) disambiguates `Eat` vs. `Drink` when `Fall < 0.40`
  ▼
Stage 3 — Alert State Machine & Audit Log (`src/pipeline/alert_state.py`, `src/utils/logger.py`)
  • GREEN  : Walk, Eat, Drink, Sit_Down, Stand_Up, Idle
  • YELLOW : Crouching, Lying (3 consecutive frames @ >= 0.50)
  • RED    : Fall (3 consecutive frames @ >= 0.75, or 1 frame >= 0.90 Instant RED)
  • Latch  : RED stays latched until manual operator Acknowledge + 5 GREEN recovery frames
  • Output : Annotated video stream, Session Summary Stats, and `logs/events.csv`
```

---

## 2. Setup & Installation

### Prerequisites
- Python 3.10+ (tested in `.venv` on Windows with `requirements.txt`)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Required Model Files (`models/`)
Ensure the following files are present in `models/`:

| File | Role | Notes |
|---|---|---|
| `models/yolo_pose_pretrained.pt` | Stage-1 pose detector | Pretrained `yolov8n-pose.pt` (COCO 17 keypoints) |
| `models/yolo_obj_pretrained.pt` | Stage-2B hand-object context | Pretrained `yolov8n.pt` (COCO 80 objects) |
| `models/action_stgcn.pth` | Primary Stage-2 classifier | `ST-GCN v4-C (+ Lower-Body Kinematics)` (`280,999` params, default) |
| `models/action_stgcn_v4_b.pth` | Paired Stage-2 specialist | `ST-GCN v4-B (+ Wrist-to-Head Features)` (`280,999` params, fused automatically) |
| `models/action_lstm.pth` | Baseline Stage-2 classifier | `ActionLSTM` (`582,664` params) |
| `models/stgcn_adjacency.npy` | Fixed skeleton adjacency | Normalized 17x17 COCO skeleton graph |

---

## 3. Running the Application, CLI Profiler & Tests

### Web Interface (Streamlit)
```bash
streamlit run src/app.py
```
- Upload any `.mp4`, `.avi`, `.mov`, or `.mkv` video (or pick a clip from `data/raw_videos/` when available locally) and click **▶ Run**.

### Standalone CLI Pipeline Test (no Streamlit)
```bash
python scripts/cli_pipeline_test.py --video path/to/video.mp4 --model stgcn
python scripts/cli_pipeline_test.py --video path/to/video.mp4 --model lstm
```

### Unit, Equivalence & Integration Tests
```bash
python tests/test_pipeline_units.py
```

---

## 4. Model Comparison Summary

Evaluated on **Dataset V3b** (`V1_LEGACY_TEST` 83 clips / 506 windows, `V2_GROUP_HELDOUT_TEST` 64 clips / 400 windows, `ALL_TEST` 147 clips / 906 windows) and the one-shot **`SEALED_GROUP_TEST`** (89 clips / 645 windows across 3 unseen groups):

| Scope / Metric | `LSTM Baseline` | `ST-GCN v3.1-C` (V3b) | **`ST-GCN v4 Dual-Specialist` (Default)** | `Ensemble (ST-GCN + LSTM)` |
|---|---:|---:|---:|---:|
| Parameters | `582,664` | `280,999` | **`280,999` × 2 (`v4-C` + `v4-B`)** | Combined |
| `ALL_TEST` Video Acc (`147` clips) | — | `72.79%` (MF1 `0.725`) | **`81.63%` (MF1 `0.806`)** | — |
| `ALL_TEST` Window Acc (`906`w) | `46.03%` | `61.81%` | **`74.06%`** | `59.49%` (vs `v3.1`) |
| `ALL_TEST` 10-FPS RED FPR (`FP/125`) | `14/125` (`11.20%`) | `4/125` (`3.20%`) | **`1/125` (`0.80%`)** | `3/125` (`2.40%`) |
| `SEALED_GROUP_TEST` Video Acc (`89` clips) | `32.58%` | `70.79%` (MF1 `0.599`) | **`77.53%` (MF1 `0.750`)** | — |
| `SEALED_GROUP_TEST` Window Acc (`645`w) | `28.99%` | `57.21%` | **`70.39%`** | — |
| `SEALED_GROUP_TEST` Rule-1 Fall Recall (`13` Fall) | `8/13` (`61.54%`) | `11/13` (`84.62%`) | **`11/13` (`84.62%`)** | — |
| `SEALED_GROUP_TEST` 10-FPS RED FPR (`76` Non-Fall) | `16/76` (`21.05%`) | `9/76` (`11.84%`) | **`0/76` (`0.00%`)** | — |

See [models/STGCN_MODEL_CARD.md](models/STGCN_MODEL_CARD.md), [models/LSTM_MODEL_CARD.md](models/LSTM_MODEL_CARD.md), and [models/ACTION_MODEL_CARD_COMPARISON.md](models/ACTION_MODEL_CARD_COMPARISON.md) for full confusion matrices and ablation logs.

---

## 5. Known Limitations & Future Work

- **Single-person primary tracking:** Stage 1 selects and tracks the highest-confidence primary subject per frame rather than maintaining multi-person IDs simultaneously.
- **CPU throughput ceiling:** On CPU-only hosts (`torch.cuda.is_available() == False`), Stage-1 YOLOv8n-pose (`imgsz=640`) runs at ~120–200 ms/tick (~5–8 FPS); setting `Pose input size` to `480` in **Advanced settings** improves throughput with a slight keypoint trade-off.
- **Data-boundary ambiguities:** Static `Lying` clips can resemble post-fall lying segments in `Fall` training clips within a `1.6s` (`16`-frame) window, and very short `Crouching` clips (`< 16` ticks @ 10 FPS) may not form a full window.
- **Deferred Milestone 5B (Live Camera):** Live Laptop Webcam (`device index`) and Phone IP Camera (`MJPEG URL` + 90-degree portrait rotation) remain intentionally deferred after the Milestone 5A/Phase 6 Video MVP.
