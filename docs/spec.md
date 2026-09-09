# System Specification
## Two-Stage Human Action Recognition System (Fall Detection)

---

## 1. Purpose

Define the end-to-end technical architecture of a **single-process Streamlit application** that performs real-time (or near-real-time) fall detection from a video stream, using a two-stage deep learning pipeline, without freezing the Streamlit UI thread or crashing the browser tab.

---

## 2. High-Level Architecture

```
                ┌─────────────────────────────────────────────────────────┐
                │                     Streamlit App (app.py)                │
                │                                                           │
   ┌────────┐   │   ┌───────────────┐    ┌────────────────┐    ┌────────┐  │
   │ Video  │──▶│──▶│  Stage 1:     │───▶│  Sequence       │───▶│Stage 2:│  │
   │ Source │   │   │  YOLO Person  │    │  Buffer         │    │ LSTM   │  │
   │(Webcam/│   │   │  Detector     │    │  (Sliding Win.) │    │ Action │  │
   │ File/  │   │   └───────────────┘    └────────────────┘    │ Classi-│  │
   │ Phone  │   │           │                                   │ fier   │  │
   │ IP Cam)│   │           │                                   └────┬───┘  │
   └────────┘   │           ▼                                        ▼      │
                │   ┌───────────────┐                     ┌──────────────┐  │
                │   │ Bounding Box  │                      │ Alert State  │  │
                │   │ Overlay       │                      │ Machine      │  │
                │   │ (cv2.draw)    │                      │ (G/Y/R)      │  │
                │   └───────┬───────┘                      └──────┬───────┘  │
                │           │                                     │          │
                │           ▼                                     ▼          │
                │   ┌───────────────────────────────────────────────────┐   │
                │   │        st.empty() Frame Placeholder (UI Render)     │   │
                │   └───────────────────────────────────────────────────┘   │
                │                          │                                │
                │                          ▼                                │
                │              ┌──────────────────────┐                    │
                │              │ CSV/JSON Event Logger │                    │
                │              └──────────────────────┘                    │
                └─────────────────────────────────────────────────────────┘
```

### 2.1 Supported Video Source Types

The system supports three interchangeable video source types, all consumed by the same `cv2.VideoCapture()` call — the pipeline downstream (Stage 1 onward) is **source-agnostic** and requires no changes regardless of which is selected:

| Source Type | `cv2.VideoCapture()` Argument | Typical Use Case |
|---|---|---|
| **Uploaded video file** | Local file path (from `st.file_uploader`, saved to a temp path) | Deterministic demos, regression testing |
| **Laptop webcam** | Integer device index (`0`, or `1`+ for external USB cameras) | Local testing on the machine running Streamlit |
| **Phone camera (IP Webcam app)** | HTTP MJPEG stream URL, e.g. `"http://192.168.1.15:8080/video"` | **In-person live demos** (e.g., classroom/thesis defense), where a dedicated, repositionable camera is preferred over the laptop's built-in webcam |

**Phone IP Camera — operational requirements (not a pipeline change, an environment constraint):**
- The phone (running an IP Webcam-style app) and the laptop running Streamlit **must be on the same local network** (same WiFi, or a phone-hosted hotspot used as a dedicated link if the venue's WiFi has client isolation enabled).
- The stream URL is entered by the user at runtime via a sidebar text input (Section 8) — it is never hardcoded, since the phone's local IP can change between sessions/networks.
- Frames arrive pre-rotated according to how the phone is held; if the phone is held in portrait orientation, a `cv2.rotate()` correction is applied before the frame enters Stage 1 (configurable, since some IP Webcam apps auto-correct orientation server-side).
- Network jitter/latency from the WiFi hop is expected and acceptable — per Section 3.2, `TARGET_FPS` already assumes a modest frame rate (5–10 FPS), well within what a local WiFi MJPEG stream can sustain.

**Key architectural decision:** everything above runs inside **one Python process**. There is no separate inference server. The "backend" is simply the pipeline modules described in `constitution.md`, invoked synchronously inside Streamlit's script-rerun loop.

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
- YOLO inference resolution is downscaled (e.g., 640px) independent of the display resolution to reduce Stage 1 latency.

### 3.3 Session State Contract (Preventing Re-initialization Crashes)

Because the entire script reruns on every frame, any object that is expensive to create (models, video capture handle, sequence buffer) **must** live in `st.session_state`, never be re-instantiated inline in the script body. See `tasks.md` for the full checklist; the contract is:

| Object | Where it lives | Initialized when |
|---|---|---|
| YOLO model | `st.cache_resource` (global, process-level) | Once per process |
| Action LSTM model | `st.cache_resource` (global, process-level) | Once per process |
| `cv2.VideoCapture` handle | `st.session_state.video_capture` | Once per session, on "Start" click, using whichever source (file path / device index / phone IP camera URL) was selected per Section 2.1 |
| Sliding window buffer (deque) | `st.session_state.seq_buffer` | Once per session, on "Start" click |
| Alert state machine instance | `st.session_state.alert_state` | Once per session |
| Run/Stop flag | `st.session_state.run_inference` | Boolean toggle |

This distinction matters: **models are shared across all users/sessions** of the app (via `@st.cache_resource`), while **video capture and buffers are per-user-session** (via `st.session_state`) since two users could be watching two different video sources.

### 3.4 Why Not Background Threads (For This Simplified Architecture)

A tempting alternative is running frame capture/inference in a background `threading.Thread` and pushing results into a `queue.Queue` that the main script polls. This is **explicitly out of scope** for this simplified build:
- Streamlit's `session_state` is not thread-safe by default; cross-thread writes require `st.session_state` proxies or `add_script_run_ctx`, adding real complexity.
- Debugging race conditions is disproportionate to the goals of a rapid prototype.
- The `st.rerun()` single-frame-per-tick pattern (Section 3.1) is sufficient for the target use case (fall detection does not require 30 FPS; 5–10 FPS is acceptable given falls unfold over ~0.5–1s).

If real-time performance later proves insufficient, threading is the documented **first scalability escape hatch** — but it is not part of this spec's baseline architecture.

---

## 4. Stage 1: Spatial Detection (YOLO)

- **Input:** Raw BGR frame (`np.ndarray`) from OpenCV.
- **Model:** Custom-trained YOLOv8/11 (single class: `person`), loaded via Ultralytics, wrapped in `src/pipeline/detector.py`.
- **Output:** List of bounding boxes `[(x1, y1, x2, y2, confidence)]`.
- **Cropping:** The highest-confidence (or largest-area, configurable) bounding box is cropped from the frame and resized to the action model's expected input size (e.g., 224x224). This crop — not the full frame — feeds Stage 2.
- **Failure mode handling:** If zero persons are detected in a frame, the sequence buffer receives a "no-person" placeholder frame (zeros or last-known-good crop, configurable) rather than crashing the pipeline; the alert state machine treats prolonged "no-person" as informational, not alerting.

---

## 5. Sequence Buffer (Sliding Window)

- Implemented as a fixed-length `collections.deque(maxlen=WINDOW_SIZE)` living in `st.session_state.seq_buffer`.
- Each cropped, preprocessed frame from Stage 1 is appended every pipeline tick.
- Once the buffer reaches `WINDOW_SIZE` (e.g., 16–30 frames), it is converted to a tensor `(1, WINDOW_SIZE, C, H, W)` or a feature-sequence tensor (if using pose keypoints instead of raw pixels — see Section 6) and passed to Stage 2.
- Before `WINDOW_SIZE` is reached, Stage 2 is **skipped**; the UI displays a "Buffering…" status rather than a premature classification.
- The buffer is a **rolling window**, not a reset-per-classification window — every new frame shifts the window forward by one, enabling near-continuous classification rather than batch chunks.

---

## 6. Stage 2: Temporal/Action Classification (LSTM)

> **Note on training provenance:** Both the YOLO weights (Section 4) and the LSTM weights described below are trained on **Kaggle Notebooks** (see `plan.md` Phase 2/3 and `tasks.md`), then downloaded and placed into `models/`. This section describes the model's *runtime* contract only — training environment details live in `plan.md`/`tasks.md`, not here, so this spec stays environment-agnostic.

- **Input:** The sequence tensor from Section 5. Two supported input modes (choose one per `plan.md`'s data stage):
  - **Raw crop mode:** CNN feature extractor (e.g., a small pretrained CNN or MobileNet backbone) + LSTM head, processing raw cropped RGB frames.
  - **Pose-keypoint mode (recommended for simplicity):** A lightweight pose estimator reduces each crop to a keypoint vector; the LSTM operates on the much smaller keypoint sequence. This dramatically reduces compute and is more robust to lighting/background variance.
- **Output:** Class probabilities across `{Normal, Warning/Unsteady, Fall}` or a binary `{Normal, Fall}` depending on `plan.md` scope decision, plus a confidence score.
- **Wrapped in:** `src/pipeline/action_classifier.py`, exposing a pure function `classify_sequence(tensor) -> ActionPrediction`.

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

The state machine is a pure object holding its own history/counters; it is instantiated once per session in `st.session_state.alert_state` and mutated by calling `.update(prediction)` each tick — never re-instantiated mid-session (which would reset all counters, per Section 3.3 rules).

---

## 8. Logging (No Database)

- Every state **transition** (not every frame) is appended as one row to `logs/events.csv` via `src/utils/logger.py`, using Python's `csv` module in append mode (`newline=""`, explicit flush) — chosen over a DB for zero-setup simplicity per project scope.
- Schema: `timestamp, session_id, from_state, to_state, confidence, video_source`.
- Optionally mirrored to a `.jsonl` file (one JSON object per line) if downstream tooling prefers JSON — both are flat-file, dependency-free formats consistent with the simplified stack.
- The Streamlit sidebar includes a lightweight "Recent Events" table read from this CSV via `pandas.read_csv`, refreshed on each rerun (bounded to the last N rows to avoid slow reads as the file grows).

---

## 9. Browser-Crash Prevention Checklist

This is the spec's central risk, so it is restated explicitly:

1. **Never** use `st.image()` in a loop without a persistent `st.empty()` placeholder — repeated calls stack DOM elements and degrade the browser tab over time.
2. **Never** run an unbounded `while True` frame loop inside a single script execution — it blocks Streamlit's event loop and the Stop button becomes unresponsive.
3. **Always** cap FPS (Section 3.2) — uncapped `st.rerun()` cycling can pin CPU/GPU at 100%, causing the browser tab (and the local machine) to become unresponsive.
4. **Always** release the `cv2.VideoCapture` handle (`cap.release()`) on "Stop" or session end — leaked handles exhaust camera/file resources across reruns.
5. **Always** downscale frames before rendering with `st.image()` — full-resolution webcam frames sent to the browser on every tick materially increase latency and memory footprint.
