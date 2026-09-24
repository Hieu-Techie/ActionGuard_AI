# Tasks Checklist
## Two-Stage Human Action Recognition System (Fall Detection)

Granular, checkable tasks organized by `plan.md` phase. Streamlit-specific state/caching tasks are called out explicitly since they are the most common source of bugs in this architecture.

---

## Phase 0 — Environment & Repo Setup

- [x] Create repo folder structure exactly as specified in `constitution.md` Section 2.
- [x] Write `requirements.txt` with pinned versions: `ultralytics`, `torch`, `torchvision`, `opencv-python`, `streamlit`, `pandas`, `numpy`.
- [ ] Add `mediapipe` to `requirements.txt` if MediaPipe is chosen as an alternative/supplement to YOLO-Pose for keypoint extraction (per `plan.md` Phase 2 decision).
- [x] Create `src/utils/config.py` with placeholder constants (to be filled in later phases): `WINDOW_SIZE`, `TARGET_FPS`, `CONF_THRESHOLD`, `WARNING_THRESHOLD`, `CRITICAL_THRESHOLD`, `INSTANT_RED_THRESHOLD`, `N_WARN_FRAMES`, `N_CRITICAL_FRAMES`, `N_RECOVERY_FRAMES`, model file paths.
- [ ] Add placeholders in `config.py` for the new schema constants: `NUM_JOINTS`, `NUM_COORDS`, joint order/names, `ACTIVE_ACTION_MODEL` (`"lstm"` | `"stgcn"`), and (for ST-GCN) the adjacency-matrix constant/path.
- [x] Verify `torch.cuda.is_available()` and record result; decide local vs. cloud training.
- [x] Create a minimal `app.py` with a single `st.title("Fall Detection Prototype")` and confirm `streamlit run src/app.py` launches in browser.
- [x] Add `.gitignore` covering `data/`, `logs/*.csv` (except a committed sample), `__pycache__/`, `.ipynb_checkpoints/`.

---

## Phase 1 — Data Acquisition & Skeleton Extraction

### 1.1 Action Video Collection
- [ ] Source/record clips per class: `Normal`, `Warning/Unsteady`, `Fall`.
- [ ] Store raw clips under `data/raw_videos/`, organized by class folder.
- [ ] Deliberately over-collect Fall/Unsteady clips relative to natural frequency, to reduce reliance on synthetic augmentation later.

### 1.2 Pretrained-Model Pose/Keypoint Extraction
- [ ] Decide final extraction model: pretrained YOLO-Pose vs. MediaPipe (finalize alongside Phase 2 validation); document the decision and rationale.
- [ ] Implement the extraction + normalization function **once**, in `src/utils/skeleton_utils.py` — importable by both this notebook and `src/pipeline/detector.py` later (avoid duplication now to prevent train/inference skew).
- [ ] Define and document the joint schema: which joints, in what order, and coordinate convention (`X, Y, Z` or `X, Y, confidence`).
- [ ] Run extraction over every raw clip; save per-frame keypoints.
- [ ] Apply normalization (e.g., hip-centered, torso-length-scaled) at extraction time using the shared function.
- [ ] Segment each clip's keypoint sequence into fixed-length windows (length = `WINDOW_SIZE`, finalized in Phase 3); assign one label per window.
- [ ] Save the dataset to `data/skeleton_dataset/` as CSV or NumPy arrays with a documented schema file.
- [ ] Compute and log class distribution; flag imbalance.
- [ ] Apply augmentation/oversampling strategy on the **skeleton coordinates** (e.g., small rotation/jitter/time-warp) for the minority Fall class; document parameters used.
- [ ] Perform this work in `notebooks/01_data_preparation.ipynb`.

---

## Phase 2 — Stage 1 Setup: Pretrained Pose Detector Integration

*(No training in this phase — integration and validation only.)*

- [ ] Select a pretrained YOLO-Pose checkpoint (e.g., `yolov8n-pose.pt` or `yolov8s-pose.pt`); confirm Ultralytics can auto-download it, or manually cache it at `models/yolo_pose_pretrained.pt`.
- [ ] Implement `src/pipeline/detector.py::detect_and_extract_pose(frame, model) -> PoseResult` returning bbox + keypoints for the selected person.
- [ ] Implement person-selection logic (highest-confidence or largest-bbox) for the multi-person case.
- [ ] Run the detector on sample frames/videos representative of demo conditions (lighting, distance, single vs. multiple people); visually confirm keypoints are stable and anatomically plausible.
- [ ] If keypoints are unstable/inaccurate under demo-realistic conditions, evaluate a larger checkpoint (`s`/`m`) or MediaPipe as an alternative before proceeding.
- [ ] Record the exact pretrained checkpoint name/version used, in a short note under `models/` (no full `MODEL_CARD.md` needed since there's no training, per `constitution.md` Section 6).
- [ ] Confirm `detect_and_extract_pose()`'s output format and normalization exactly match what `01_data_preparation.ipynb` produces (same joint order, same coordinate convention) — diff explicitly.

---

## Phase 3 — Action Model Training (LSTM vs. ST-GCN Comparison)

**Environment:** Kaggle Notebooks (GPU accelerator), both models trained on the identical dataset.

### 3.1 Shared Setup
- [ ] Upload the Phase 1 skeleton dataset as a single Kaggle Dataset; record the dataset name/version.
- [ ] Finalize `WINDOW_SIZE` and record it in `src/utils/config.py` (single source of truth).
- [ ] Finalize the joint schema and, for ST-GCN, define the fixed skeleton adjacency matrix; record both in `config.py`/`skeleton_utils.py`.
- [ ] Define a single fixed train/val/test split (or fixed random seed) to be reused identically by both training notebooks, so the comparison is valid.

### 3.2 Model A — LSTM (`notebooks/02_action_model_training_lstm.ipynb`)
- [ ] Create Kaggle Notebook, attach the shared dataset, enable GPU accelerator.
- [ ] Add a first cell installing/pinning dependencies.
- [ ] Copy the shared preprocessing/normalization function in as a self-contained, clearly-commented cell (must stay in sync with `src/pipeline/`/`skeleton_utils.py`).
- [ ] Build the dataset/dataloader producing flattened per-frame keypoint vectors over time, shape `(WINDOW_SIZE, NUM_JOINTS * NUM_COORDS)`.
- [ ] Define the LSTM (or GRU) architecture + classification head.
- [ ] Fix random seeds (`torch.manual_seed`, `numpy.random.seed`).
- [ ] Add periodic checkpoint-saving to `/kaggle/working/`.
- [ ] Train; track loss and per-class accuracy.
- [ ] Generate a confusion matrix on the held-out test set; specifically report **Fall-class recall/False Negative rate**.
- [ ] Measure and record per-sequence inference latency.
- [ ] Download final weights and the executed notebook. Place weights at `models/action_lstm.pth`; copy notebook into `notebooks/02_action_model_training_lstm.ipynb`, marking Kaggle-only cells per `constitution.md` Section 5.1.
- [ ] Write `models/LSTM_MODEL_CARD.md` (training date, Kaggle Dataset version, window size, metrics, input shape).

### 3.3 Model B — ST-GCN (`notebooks/03_action_model_training_stgcn.ipynb`)
- [ ] Create Kaggle Notebook, attach the **same** shared dataset, enable GPU accelerator.
- [ ] Add a first cell installing/pinning dependencies (including any graph-convolution-specific packages).
- [ ] Copy the shared preprocessing/normalization function in as a self-contained, clearly-commented cell (same source as 3.2 — must not diverge).
- [ ] Build the dataset/dataloader producing tensors shaped `(NUM_COORDS, WINDOW_SIZE, NUM_JOINTS)` plus the fixed adjacency matrix.
- [ ] Verify the adjacency matrix indices exactly match the joint schema/order used in the dataset — a silent mismatch produces a nonsensical graph without an obvious error.
- [ ] Define the ST-GCN architecture (spatial-temporal graph conv blocks + pooling + classification head).
- [ ] Fix random seeds; use the **same** train/val/test split as the LSTM notebook (Section 3.1).
- [ ] Add periodic checkpoint-saving to `/kaggle/working/`.
- [ ] Train; track loss and per-class accuracy.
- [ ] Generate a confusion matrix on the held-out test set; specifically report **Fall-class recall/False Negative rate**.
- [ ] Measure and record per-sequence inference latency.
- [ ] Download final weights and the executed notebook. Place weights at `models/action_stgcn.pth`; copy notebook into `notebooks/03_action_model_training_stgcn.ipynb`, marking Kaggle-only cells per `constitution.md` Section 5.1.
- [ ] Write `models/STGCN_MODEL_CARD.md` (training date, Kaggle Dataset version, window size, metrics, input shape).

### 3.4 Comparison
- [ ] Write `models/ACTION_MODEL_CARD_COMPARISON.md` comparing both models: accuracy, per-class F1, Fall-class recall, inference latency, qualitative notes.
- [ ] Decide the default `ACTIVE_ACTION_MODEL` for `config.py` based on this comparison (both remain selectable at runtime regardless).

---

## Phase 4 — Pipeline Assembly (Pre-Streamlit)

- [ ] Implement `src/utils/skeleton_utils.py`: shared extraction + normalization functions (imported by both notebooks and `src/pipeline/`).
- [ ] Implement `src/pipeline/detector.py`:
  - [ ] `load_pose_model(path) -> model`
  - [ ] `detect_and_extract_pose(frame, model) -> PoseResult` (bbox + keypoints for the selected person)
- [ ] Implement `src/pipeline/sequence_buffer.py`:
  - [ ] `SequenceBuffer` class wrapping `collections.deque(maxlen=WINDOW_SIZE)` of normalized keypoint vectors
  - [ ] `.append(keypoint_vector)`, `.is_ready() -> bool`, `.as_lstm_tensor() -> torch.Tensor`, `.as_stgcn_tensor() -> torch.Tensor`
- [ ] Implement `src/pipeline/action_classifier.py`:
  - [ ] `load_action_model(path, model_type) -> model`
  - [ ] `classify_sequence(buffer, models, active_model) -> ActionPrediction` (dataclass with `label`, `confidence`) — branches internally on `active_model` to call the correct buffer tensor method and model
- [ ] Implement `src/pipeline/alert_state.py`:
  - [ ] `AlertLevel` enum (`GREEN`, `YELLOW`, `RED`)
  - [ ] `AlertStateMachine` class with `.update(prediction) -> AlertLevel` and internal consecutive-frame counters
  - [ ] Explicit `.acknowledge()` method required to clear `RED` (no silent auto-clear, per `spec.md` Section 7)
- [ ] Write a standalone CLI/script test: run full pipeline over a test video file, print state transitions with timestamps to console — **no Streamlit import anywhere in this test path**.
- [ ] Run the CLI test once with `ACTIVE_ACTION_MODEL = "lstm"` and once with `"stgcn"`; confirm both reproduce the expected Green→Yellow→Red sequence on a known "fall" test clip.
- [ ] Confirm preprocessing (joint schema, normalization constants) is identical between `src/pipeline/`/`skeleton_utils.py` and both training notebooks — diff the constants explicitly.

---

## Phase 5 — Streamlit Integration

### 5.1 Model Caching (`@st.cache_resource`)
- [ ] Write a `load_models()` function decorated with `@st.cache_resource` that loads the pose detector **and both** action models (LSTM and ST-GCN) and returns them as a dataclass/dict.
  ```python
  @st.cache_resource
  def load_models():
      pose_model = load_pose_model(config.YOLO_POSE_MODEL_PATH)
      lstm_model = load_action_model(config.LSTM_MODEL_PATH, "lstm")
      stgcn_model = load_action_model(config.STGCN_MODEL_PATH, "stgcn")
      return pose_model, lstm_model, stgcn_model
  ```
- [ ] Confirm via logging/print that `load_models()` executes **only once** per process across multiple reruns (add a temporary log line inside, verify it fires once, then remove).
- [ ] Do **not** decorate anything that takes `st.session_state` or mutable per-user objects as an argument with `@st.cache_resource` — cache only the models themselves.
- [ ] Add a spinner (`st.spinner("Loading models...")`) around the first call for UX during cold start — note this cold start now loads three models instead of two, so measure and mention the load time in the UI if it's non-trivial.

### 5.2 Session State Initialization
- [ ] At the top of `app.py`, initialize all session keys defensively using `if "key" not in st.session_state:` guards — never unconditional assignment (which would reset state every rerun):
  - [ ] `st.session_state.run_inference = False`
  - [ ] `st.session_state.video_capture = None`
  - [ ] `st.session_state.seq_buffer = SequenceBuffer(config.WINDOW_SIZE)`
  - [ ] `st.session_state.alert_state = AlertStateMachine()`
  - [ ] `st.session_state.active_model = config.ACTIVE_ACTION_MODEL` (default from config, overridable via sidebar)
  - [ ] `st.session_state.last_frame_time = 0.0`
  - [ ] `st.session_state.event_log_path = "logs/events.csv"`
- [ ] Implement "Start" button logic: only opens `cv2.VideoCapture` and resets buffer/state **if not already running** (guard against double-open on accidental double-click/rerun).
- [ ] Implement "Stop" button logic: sets `run_inference = False`, calls `cap.release()`, and explicitly sets `video_capture = None` in session state.
- [ ] Confirm switching `st.session_state.active_model` mid-session does **not** reset `seq_buffer` or `alert_state` — only which model is called on the next classification tick changes.
- [ ] Add a session-end/cleanup safeguard: wrap capture access in a check that re-opens gracefully if the handle was lost (e.g., app hot-reloaded mid-session).

### 5.3 Frame Loop Implementation
- [ ] Create persistent placeholders **once**, outside the loop body: `frame_placeholder = st.empty()`, `status_placeholder = st.empty()`.
- [ ] Implement FPS governance: compute `elapsed = time.time() - st.session_state.last_frame_time`; skip processing (but still rerun) if `elapsed < 1 / config.TARGET_FPS`.
- [ ] On each valid tick: read frame → `detect_and_extract_pose` → normalize keypoints → `seq_buffer.append` → if `seq_buffer.is_ready()`, `classify_sequence(..., active_model=st.session_state.active_model)` → `alert_state.update`.
- [ ] Render annotated frame with skeleton overlay via `frame_placeholder.image(...)` (never a bare `st.image()` call inside the loop).
- [ ] Render alert badge via `status_placeholder` using color-coded markdown/HTML based on `alert_state.level`; optionally show which model produced the current prediction.
- [ ] Call `st.rerun()` at the end of the tick **only if** `st.session_state.run_inference` is still `True`.
- [ ] Verify the Stop button is clickable and takes effect within one tick while inference is running (manual test, not just code review).

### 5.4 Logging Integration
- [ ] Implement `src/utils/logger.py::log_event(from_state, to_state, confidence, active_model, source)` appending a row to `events.csv` (create file with header if it doesn't exist).
- [ ] Call `log_event` only on **state transitions**, not every frame — compare `alert_state`'s previous vs. new level to detect a transition.
- [ ] Add a sidebar "Recent Events" table reading the last N rows via `pandas.read_csv(...).tail(N)`, including the `active_model` column.
- [ ] Add a "Download events.csv" button (`st.download_button`) in the sidebar.

### 5.5 UI Controls (`src/ui/sidebar.py`)
- [ ] Video source selector with three options: Webcam (device index), Upload File (`st.file_uploader`), Phone IP Camera (URL).
- [ ] For "Webcam": add a small numeric input for device index (default `0`; external USB cameras may be `1`+).
- [ ] For "Phone IP Camera": add a text input for the stream URL (e.g., `http://<phone-ip>:8080/video`); do not hardcode any default IP since it changes per network/session.
- [ ] For "Phone IP Camera": add an optional "Rotate 90°" toggle to correct portrait-orientation streams before frames enter Stage 1.
- [ ] Add an active-model toggle/radio button: "LSTM" vs. "ST-GCN", bound to `st.session_state.active_model`.
- [ ] Start / Stop buttons wired to session-state flags per Task 5.2.
- [ ] Threshold sliders (Warning/Critical confidence) for live demo tuning — read into `config` overrides, not hardcoded.
- [ ] "Acknowledge Alert" button, enabled only when `alert_state.level == RED`, calling `alert_state.acknowledge()`.

### 5.6 Edge Case Testing
- [ ] Test: no person visible in frame for extended period — confirm no crash, buffer handles gracefully per `spec.md` Section 4.
- [ ] Test: multiple people in frame — confirm person-selection logic (largest/highest-confidence) behaves predictably.
- [ ] Test: video file reaches end of stream — confirm graceful stop, not an exception on `cap.read()` returning `False`.
- [ ] Test: rapid Start/Stop/Start clicking — confirm no duplicate `VideoCapture` handles leak and no duplicate model loads occur.
- [ ] Test: switching video source mid-session — confirm old capture is released before new one opens.
- [ ] Test: switching active model (LSTM ↔ ST-GCN) mid-session, multiple times in a row — confirm no crash, no memory leak, and the alert state machine's history is preserved across the switch.
- [ ] Test: phone IP camera — connect via the exact WiFi network planned for the demo venue **in advance**; confirm the laptop can reach the stream URL (client isolation on some networks blocks device-to-device traffic even on the same WiFi).
- [ ] Test: phone IP camera — disconnect/reconnect the stream mid-session (e.g., WiFi drop) and confirm the app surfaces a clear error/status rather than freezing or crashing silently.
- [ ] Test: phone IP camera — verify orientation (portrait vs. landscape) renders correctly with the "Rotate 90°" toggle before relying on it for a live demo.
- [ ] Confirm browser tab CPU/memory stays stable over a 5+ minute continuous run (no unbounded DOM growth, no memory leak from placeholders), with all three models (pose + LSTM + ST-GCN) loaded simultaneously.

---

## Phase 6 — Hardening & Demo Polish

- [ ] Re-tune `WARNING_THRESHOLD` / `CRITICAL_THRESHOLD` / frame-count constants against Phase 3's confusion matrix for the chosen default model; sanity-check they're also reasonable for the non-default model.
- [ ] Add a bundled "demo fall clip" as a selectable video source for reliable live demonstrations.
- [ ] (Optional) Implement a "live comparison" view showing both LSTM and ST-GCN predictions side by side for the same buffered window.
- [ ] If demoing live with a phone-as-camera setup: verify the WiFi network at the demo venue in advance, keep the phone charging during the demo, and rehearse the exact Start → live action → alert sequence at least once beforehand, including a model switch.
- [ ] Polish alert badge styling (color, size, iconography) for clear at-a-glance status.
- [ ] Add session summary stats to the UI (e.g., total alerts this session, time in each state, breakdown by active model).
- [ ] Write top-level `README.md`: setup instructions, `streamlit run` command, model file placement (all three: pose + LSTM + ST-GCN), LSTM-vs-ST-GCN comparison summary, and known limitations (e.g., single-person tracking only, FPS ceiling).
- [ ] Final end-to-end test: fresh environment, fresh clone, all three model files placed manually, full demo run — including a live model switch — without any developer intervention.
