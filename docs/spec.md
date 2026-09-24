# System Specification
## Two-Stage Human Action Recognition System (Fall Detection)

---

## 1. Purpose

Define the end-to-end technical architecture of a **single-process Streamlit application** that performs real-time (or near-real-time) fall detection from a video stream, using a two-stage pipeline — pretrained pose detection feeding a skeleton-based action classifier — without freezing the Streamlit UI thread or crashing the browser tab.

---

## 2. High-Level Architecture

```
                ┌───────────────────────────────────────────────────────────────┐
                │                     Streamlit App (app.py)                     │
                │                                                                 │
   ┌────────┐   │   ┌────────────────┐    ┌────────────────┐    ┌─────────────┐  │
   │ Video  │──▶│──▶│  Stage 1:      │───▶│  Sequence       │───▶│  Stage 2:   │  │
   │ Source │   │   │  YOLO-Pose     │    │  Buffer         │    │  Action     │  │
   │(Webcam/│   │   │  (PRETRAINED)  │    │  (Sliding Win.  │    │  Classifier │  │
   │ File/  │   │   │  person bbox + │    │   of keypoint   │    │  LSTM  or   │  │
   │ Phone  │   │   │  (X,Y,Z) joints│    │   vectors)      │    │  ST-GCN     │  │
   │ IP Cam)│   │   └────────────────┘    └────────────────┘    │ (selectable)│  │
   └────────┘   │           │                                    └──────┬──────┘  │
                │           ▼                                           ▼         │
                │   ┌────────────────┐                      ┌──────────────────┐ │
                │   │ Skeleton       │                       │ Alert State      │ │
                │   │ Overlay        │                       │ Machine          │ │
                │   │ (cv2.draw)     │                       │ (G/Y/R)          │ │
                │   └───────┬────────┘                       └──────┬───────────┘ │
                │           │                                       │             │
                │           ▼                                       ▼             │
                │   ┌────────────────────────────────────────────────────────┐   │
                │   │            st.empty() Frame Placeholder (UI Render)     │   │
                │   └────────────────────────────────────────────────────────┘   │
                │                            │                                   │
                │                            ▼                                   │
                │                ┌──────────────────────┐                        │
                │                │ CSV/JSON Event Logger │                        │
                │                └──────────────────────┘                        │
                └───────────────────────────────────────────────────────────────┘
```

**Key architectural decision:** everything above runs inside **one Python process**. There is no separate inference server. The "backend" is simply the pipeline modules described in `constitution.md`, invoked synchronously inside Streamlit's script-rerun loop.

**Key data-flow decision:** raw RGB pixels never reach Stage 2. Stage 1 reduces every frame to a compact numeric skeleton (joint coordinates); everything downstream — the buffer, both action models, and the alert logic — operates purely on those numbers. This is what makes the LSTM vs. ST-GCN comparison fair: both models are trained and evaluated on the exact same numeric dataset.

### 2.1 Supported Video Source Types

The system supports three interchangeable video source types, all consumed by the same `cv2.VideoCapture()` call — the pipeline downstream (Stage 1 onward) is **source-agnostic** and requires no changes regardless of which is selected:

| Source Type | `cv2.VideoCapture()` Argument | Typical Use Case |
|---|---|---|
| **Uploaded video file** | Local file path (from `st.file_uploader`, saved to a temp path) | Deterministic demos, regression testing |
| **Laptop webcam** | Integer device index (`0`, or `1`+ for external USB cameras) | Local testing on the machine running Streamlit |
| **Phone camera (IP Webcam app)** | HTTP MJPEG stream URL, e.g. `"http://192.168.1.15:8080/video"` | **In-person live demos** (e.g., classroom/thesis defense), where a dedicated, repositionable camera is preferred over the laptop's built-in webcam |

**Phone IP Camera — operational requirements (not a pipeline change, an environment constraint):**
- The phone (running an IP Webcam-style app) and the laptop running Streamlit **must be on the same local network** (same WiFi, or a phone-hosted hotspot used as a dedicated link if the venue's WiFi has client isolation enabled).
- The stream URL is entered by the user at runtime via a sidebar text input (Section 8), not hardcoded, since the phone's local IP can change between sessions/networks.
- Frames arrive pre-rotated according to how the phone is held; if the phone is held in portrait orientation, a `cv2.rotate()` correction is applied before the frame enters Stage 1 (configurable, since some IP Webcam apps auto-correct orientation server-side).
- Network jitter/latency from the WiFi hop is expected and acceptable — per Section 3.2, `TARGET_FPS` already assumes a modest frame rate (5–10 FPS), well within what a local WiFi MJPEG stream can sustain.

---

## 3. The Core Challenge: Video in a Script-Rerun Framework

Streamlit's execution model reruns the entire script top-to-bottom on every interaction. A naive `while True: cap.read()` loop **will freeze the app** because it blocks the single Streamlit script thread — no widgets respond, and the browser tab will appear to hang or the WebSocket connection will time out.

### 3.1 The Solution Pattern: Bounded Loop + `st.empty()` + Manual Rerun Control

Instead of one infinite blocking loop, we use a **bounded, yield-friendly loop** that:
1. Runs inside the same script execution (not a background thread — see Section 3.4 for why).
2. Processes a fixed, small number of frames per script pass (or runs until a "Stop" flag flips in `session_state`).
3. Writes each rendered frame into a **single reused `st.empty()` placeholder** rather than calling `st.image()` repeatedly (which would stack elements and balloon the DOM).
4. Yields control back to Streamlit periodically so the UI thread can process the Stop button and other widget events.

```python
# app.py (simplified skeleton)
frame_placeholder = st.empty()
status_placeholder = st.empty()

run = st.session_state.get("run_inference", False)

if run:
    cap = st.session_state.video_capture
    ret, frame = cap.read()
    if ret:
        annotated_frame, alert_level = run_pipeline(frame, models, st.session_state.seq_buffer)
        frame_placeholder.image(annotated_frame, channels="BGR")
        status_placeholder.markdown(render_alert_badge(alert_level))
        log_event_if_needed(alert_level)

    # Re-trigger the next frame WITHOUT a blocking while-loop
    st.rerun()
```

**Why `st.rerun()` instead of a `while` loop:** each call processes exactly one frame, then hands control back to Streamlit's event loop. Streamlit immediately reruns the script, which reads the next frame. This keeps the Stop button, sidebar sliders, and browser responsive because the script never blocks indefinitely inside a single pass. It is effectively a **cooperative, single-frame-per-tick loop** disguised as a rerun cycle.

### 3.2 Frame Rate Governance

Uncontrolled `st.rerun()` cycling will hammer the CPU/GPU. The spec mandates:
- A `time.sleep(1 / TARGET_FPS)` (or a `time.time()` delta check) inside the loop, capped via `src/utils/config.py::TARGET_FPS` (default: 10 FPS — sufficient for fall detection, not full 30 FPS video).
- **Frame skipping over frame queuing.** If processing falls behind the source FPS, the system drops frames (reads and discards) rather than buffering them — avoiding unbounded memory growth and lag accumulation.
- YOLO-Pose inference resolution is downscaled (e.g., 640px) independent of the display resolution to reduce Stage 1 latency.

### 3.3 Session State Contract (Preventing Re-initialization Crashes)

Because the entire script reruns on every frame, any object that is expensive to create (models, video capture handle, sequence buffer) **must** live in `st.session_state`, never be re-instantiated inline in the script body. See `tasks.md` for the full checklist; the contract is:

| Object | Where it lives | Initialized when |
|---|---|---|
| YOLO-Pose model (pretrained) | `st.cache_resource` (global, process-level) | Once per process |
| Action model — LSTM | `st.cache_resource` (global, process-level) | Once per process |
| Action model — ST-GCN | `st.cache_resource` (global, process-level) | Once per process |
| `cv2.VideoCapture` handle | `st.session_state.video_capture` | Once per session, on "Start" click, using whichever source (file path / device index / phone IP camera URL) was selected per Section 2.1 |
| Sliding window buffer of keypoint vectors (deque) | `st.session_state.seq_buffer` | Once per session, on "Start" click |
| Alert state machine instance | `st.session_state.alert_state` | Once per session |
| Active model choice (`"lstm"` or `"stgcn"`) | `st.session_state.active_model` | Once per session, changeable via sidebar |
| Run/Stop flag | `st.session_state.run_inference` | Boolean toggle |

This distinction matters: **models are shared across all users/sessions** of the app (via `@st.cache_resource`), while **video capture, buffers, and the active-model choice are per-user-session** (via `st.session_state`) since two users could be watching two different sources or comparing different models simultaneously.

**Note on caching both action models:** Both LSTM and ST-GCN weights are loaded once at startup (both cached via `@st.cache_resource`), even though only one is active per session. This trades a small amount of memory for the ability to switch models instantly via a sidebar toggle without a reload — useful for live side-by-side comparison during a demo.

### 3.4 Why Not Background Threads (For This Simplified Architecture)

A tempting alternative is running frame capture/inference in a background `threading.Thread` and pushing results into a `queue.Queue` that the main script polls. This is **explicitly out of scope** for this simplified build:
- Streamlit's `session_state` is not thread-safe by default; cross-thread writes require `st.session_state` proxies or `add_script_run_ctx`, adding real complexity.
- Debugging race conditions is disproportionate to the goals of a rapid prototype.
- The `st.rerun()` single-frame-per-tick pattern (Section 3.1) is sufficient for the target use case (fall detection does not require 30 FPS; 5–10 FPS is acceptable given falls unfold over ~0.5–1s).

If real-time performance later proves insufficient, threading is the documented **first scalability escape hatch** — but it is not part of this spec's baseline architecture.

---

## 4. Stage 1: Spatial Detection & Pose Extraction (Pretrained YOLO-Pose)

- **Input:** Raw BGR frame (`np.ndarray`) from OpenCV.
- **Model:** A pretrained, off-the-shelf YOLO-Pose checkpoint (e.g., Ultralytics `yolov8n-pose.pt` / `yolov8s-pose.pt`, trained on COCO keypoints), loaded via Ultralytics, wrapped in `src/pipeline/detector.py`. **This model is not fine-tuned or retrained in this project** — see `constitution.md` Section 6.
- **Output:** For each detected person — a bounding box `(x1, y1, x2, y2, confidence)` **and** a keypoint set: `(x, y, [z or confidence])` per joint, following the model's native joint schema (e.g., 17 COCO keypoints). If a project decision favors MediaPipe instead (e.g., for a richer 33-point/3D skeleton), the same contract applies: `detect_and_extract_pose()` returns bbox + keypoints regardless of which underlying library produced them — this choice is isolated entirely inside `src/pipeline/detector.py` and `src/utils/skeleton_utils.py`.
- **Person selection:** When multiple people are detected, the highest-confidence (or largest bounding-box-area, configurable) detection is selected as the tracked subject. Only that person's keypoints feed the sequence buffer.
- **Normalization:** Raw pixel-space keypoints are converted to a normalized, scale-invariant representation before entering the buffer (e.g., centered on the hip midpoint, scaled by torso length) — this normalization function lives in `src/utils/skeleton_utils.py` and is the **exact same function** used when building the training dataset in `01_data_preparation.ipynb` (see `constitution.md` Section 6).
- **Failure mode handling:** If zero persons are detected in a frame, the sequence buffer receives a "no-person" placeholder keypoint vector (zeros, or the last-known-good vector, configurable) rather than crashing the pipeline; the alert state machine treats prolonged "no-person" as informational, not alerting.

---

## 5. Sequence Buffer (Sliding Window of Skeletons)

- Implemented as a fixed-length `collections.deque(maxlen=WINDOW_SIZE)` living in `st.session_state.seq_buffer`.
- Each **normalized keypoint vector** from Stage 1 (not a cropped image) is appended every pipeline tick. A single vector has shape `(NUM_JOINTS * NUM_COORDS,)` where `NUM_COORDS` is 2 (X, Y) or 3 (X, Y, Z), per `src/utils/config.py`.
- Once the buffer reaches `WINDOW_SIZE` (e.g., 16–30 frames), it is converted into the tensor shape each model expects (see Section 6) and passed to Stage 2.
- Before `WINDOW_SIZE` is reached, Stage 2 is **skipped**; the UI displays a "Buffering…" status rather than a premature classification.
- The buffer is a **rolling window**, not a reset-per-classification window — every new frame shifts the window forward by one, enabling near-continuous classification rather than batch chunks.

---

## 6. Stage 2: Temporal/Action Classification (LSTM vs. ST-GCN)

This project trains and compares **two interchangeable action classifiers** on the identical skeleton dataset. Both consume the same `SequenceBuffer` output; only the tensor reshaping and internal architecture differ, and both differences are isolated inside `src/pipeline/action_classifier.py` per `constitution.md` Section 4, Rule 8.

> **Note on training provenance:** Both action models are trained on **Kaggle Notebooks** (see `plan.md` Phase 3 and `tasks.md`), then downloaded and placed into `models/`. Stage 1's pretrained YOLO-Pose weights require no training at all. This section describes each model's *runtime* contract only.

### 6.1 Option A — LSTM (Sequence Model)
- **Input tensor:** `(1, WINDOW_SIZE, NUM_JOINTS * NUM_COORDS)` — the buffer flattened per-frame into a single feature vector, stacked over time. This treats the skeleton as an unstructured feature vector per timestep and lets the LSTM learn temporal dependencies only.
- **Architecture:** Stacked LSTM (or GRU) layers followed by a classification head (dense + softmax) over `{Normal, Warning/Unsteady, Fall}`.
- **Characteristic trade-off:** Simple to implement and fast to train; does not explicitly model the *spatial* relationships between joints (e.g., that a knee is connected to a hip) — it must learn any such structure implicitly from the flattened vector.

### 6.2 Option B — ST-GCN (Spatial-Temporal Graph Convolutional Network)
- **Input tensor:** `(1, NUM_COORDS, WINDOW_SIZE, NUM_JOINTS)` plus a fixed **adjacency matrix** describing the human skeleton graph (which joints are physically connected — e.g., wrist–elbow, elbow–shoulder). The adjacency matrix is a static constant defined once in `src/utils/config.py` (or `skeleton_utils.py`) matching the chosen joint schema, and must be identical between training and inference.
- **Architecture:** Stacked spatial-temporal graph convolution blocks (alternating graph convolutions over the joint dimension and standard temporal convolutions over the frame dimension), followed by global pooling and a classification head over the same `{Normal, Warning/Unsteady, Fall}` classes.
- **Characteristic trade-off:** Explicitly encodes skeletal structure, generally expected to outperform a plain LSTM on skeleton-based action recognition benchmarks; more implementation complexity and a stricter dependency on a correct, consistent joint schema/adjacency matrix.

### 6.3 Runtime Model Selection
- `src/utils/config.py` defines `ACTIVE_ACTION_MODEL: Literal["lstm", "stgcn"]`, with both models loaded into memory at startup (Section 3.3).
- The Streamlit sidebar (Section 8) exposes a toggle to switch `st.session_state.active_model` between `"lstm"` and `"stgcn"` without restarting the app — useful for a live demo comparing both architectures on the same input stream.
- `classify_sequence(buffer, models, active_model) -> ActionPrediction` in `src/pipeline/action_classifier.py` branches internally on `active_model` to apply the correct tensor reshape before calling the corresponding model — callers never handle this distinction themselves.
- **Output (both models):** Class probabilities across `{Normal, Warning/Unsteady, Fall}` (or a binary `{Normal, Fall}`, per `plan.md`'s scope decision), plus a confidence score, returned as the same `ActionPrediction` dataclass regardless of which model produced it.

---

## 7. Alert State Machine

Implemented as an explicit class in `src/pipeline/alert_state.py`:

```python
class AlertLevel(Enum):
    GREEN = "Normal"
    YELLOW = "Warning"
    RED = "Critical - Fall"
```

**Transition rules (defaults, tunable in `config.py`):**
- `GREEN → YELLOW`: action model confidence for "Fall"/"Unsteady" class exceeds `WARNING_THRESHOLD` for `N_WARN_FRAMES` consecutive ticks.
- `YELLOW → RED`: confidence exceeds `CRITICAL_THRESHOLD` for `N_CRITICAL_FRAMES` consecutive ticks, **or** an instantaneous high-confidence Fall prediction exceeds `INSTANT_RED_THRESHOLD`.
- `RED → GREEN`: requires `N_RECOVERY_FRAMES` consecutive Normal predictions **and** a manual "Acknowledge" button click in the UI (a Red alert must never silently self-clear — this is a deliberate safety-oriented UX decision).
- `YELLOW → GREEN`: automatic after `N_RECOVERY_FRAMES` consecutive Normal predictions.

The state machine is a pure object holding its own history/counters; it is instantiated once per session in `st.session_state.alert_state` and mutated by calling `.update(prediction)` each tick — never re-instantiated mid-session (which would reset all counters, per Section 3.3 rules). Switching the active model mid-session (Section 6.3) does **not** reset the alert state machine — both models feed predictions into the same running state machine, so switching models is a like-for-like comparison of *predictions*, not a reset of the alert history.

---

## 8. Logging (No Database)

- Every state **transition** (not every frame) is appended as one row to `logs/events.csv` via `src/utils/logger.py`, using Python's `csv` module in append mode (`newline=""`, explicit flush) — chosen over a DB for zero-setup simplicity per project scope.
- Schema: `timestamp, session_id, from_state, to_state, confidence, active_model, video_source`. The `active_model` column records which classifier (`lstm` or `stgcn`) produced the transition, enabling post-hoc comparison of the two models' alerting behavior from the same log file.
- Optionally mirrored to a `.jsonl` file (one JSON object per line) if downstream tooling prefers JSON — both are flat-file, dependency-free formats consistent with the simplified stack.
- The Streamlit sidebar includes a lightweight "Recent Events" table read from this CSV via `pandas.read_csv`, refreshed on each rerun (bounded to the last N rows to avoid slow reads as the file grows).

---

## 9. Browser-Crash Prevention Checklist

This is the spec's central risk, so it is restated explicitly:

1. **Never** use `st.image()` in a loop without a persistent `st.empty()` placeholder — repeated calls stack DOM elements and degrade the browser tab over time.
2. **Never** run an unbounded `while True` frame loop inside a single script execution — it blocks Streamlit's event loop and the Stop button becomes unresponsive.
3. **Always** cap FPS (Section 3.2) — uncapped `st.rerun()` cycling can pin CPU/GPU at 100%, causing the browser tab (and the local machine) to become unresponsive. This matters more, not less, in this architecture since two models are held in memory simultaneously (Section 3.3).
4. **Always** release the `cv2.VideoCapture` handle (`cap.release()`) on "Stop" or session end — leaked handles exhaust camera/file resources across reruns.
5. **Always** downscale frames before rendering with `st.image()` — full-resolution webcam frames sent to the browser on every tick materially increase latency and memory footprint.
