# YOLO-Pose Checkpoint Note

## Model Details
- **File:** `models/yolo_pose_pretrained.pt` (6.52 MB)
- **Model:** YOLOv8n-pose (nano variant — smallest/fastest)
- **Framework:** Ultralytics v8.3.0
- **Task:** Pose estimation — COCO 17-keypoint schema
- **Source:** Downloaded from Ultralytics GitHub releases
- **Downloaded:** 2026-11-15
- **No training performed** — pretrained weights used as-is

## Input / Output
- **Input:** BGR frames at any resolution (internally resized to 640 px longest side)
- **Output per person:** `keypoints [x_px, y_px, conf]` for 17 joints + `bbox [x1, y1, x2, y2]`
- **Person selection:** Highest bounding-box confidence score (multi-person frames → one person selected)

## Keypoint Schema (COCO 17)
```
 0: nose          1: left_eye       2: right_eye
 3: left_ear      4: right_ear      5: left_shoulder
 6: right_shoulder 7: left_elbow    8: right_elbow
 9: left_wrist   10: right_wrist   11: left_hip
12: right_hip    13: left_knee     14: right_knee
15: left_ankle   16: right_ankle
```

## Normalisation (applied by `skeleton_utils.normalise_keypoints`)
- **Origin:** Hip midpoint `(joints[11] + joints[12]) / 2`
- **Scale:** Torso length `‖hip_mid → shoulder_mid‖`; fallback = `1/3 × bbox diagonal` when shoulders invisible
- **Y-axis:** Upward-positive (standard Cartesian — y_norm = −y_raw)
- **Low-confidence joints** (conf < `CONF_THRESHOLD`): zeroed out before normalisation

## Validation Results (Phase 2)
Tested on first clip from each of 8 classes:

| Class       | Detected | Conf  | kp_norm shape | Diff vs skeleton_utils |
|-------------|----------|-------|---------------|------------------------|
| Crouching   | ✅ True  | 0.905 | (17, 3)       | 0.000000               |
| Drink       | ✅ True  | 0.887 | (17, 3)       | 0.000000               |
| Eat         | ✅ True  | 0.884 | (17, 3)       | 0.000000               |
| Fall        | ✅ True  | 0.896 | (17, 3)       | 0.000000               |
| Lying       | ✅ True  | 0.828 | (17, 3)       | 0.000000               |
| Sit_Down    | ✅ True  | 0.771 | (17, 3)       | 0.000000               |
| Stand_Up    | ✅ True  | 0.815 | (17, 3)       | 0.000000               |
| Walk        | ✅ True  | 0.510 | None*         | N/A*                   |

\* Walk 001.mp4 is 160×120 resolution — person detected but hips invisible at this resolution,
  so `normalise_keypoints` returns None (expected fallback behaviour; not an error).
  Higher-resolution Walk clips normalise correctly.

**Conclusion:** Output format and normalisation are byte-identical between `detector.py` and
`skeleton_utils.py` on all clips where hips are visible. No train/inference skew.

## Decision: Keep yolov8n-pose & CPU Latency Benchmark
Keypoints are stable and anatomically plausible across all tested classes with visible hips.
No need to upgrade to `yolov8s-pose` or switch to MediaPipe at this stage.

### Measured CPU Latency Profile (`scripts/cli_pipeline_test.py`)
- **Runtime:** Local CPU-only PyTorch (`.pt` weights, `torch.cuda.is_available() == False`)
- **Input / Inference Size:** `Fall/video (1)_fall.avi` (320×240 native, resized internally to `imgsz=640`)
- **Protocol:** 53 sampled ticks @ 10 FPS; Tick 1 excluded as PyTorch warm-up; mean over Ticks 2–53:
  - **Stage 1 (`YOLO-Pose + normalise_keypoints`):** `112.08 ms` / tick (~`8.9 FPS`, 93.3% of total time)
  - **SequenceBuffer Append:** `0.009 ms` / tick
  - **Stage 2 (`ActionSTGCN` Inference):** `8.02 ms` / tick (6.7% of total time)
  - **AlertStateMachine Update:** `0.012 ms` / tick
  - **Full End-to-End Pipeline:** `120.12 ms` / tick (~`8.3 FPS`)
- **Phase 6 Tuning Note:** If strict `>= 10 FPS` (`<= 100 ms/tick`) on CPU is required during live demo, benchmark `imgsz=480` / `imgsz=384` in Phase 6 while verifying keypoint stability.
