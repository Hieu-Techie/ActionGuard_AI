# Tasks Checklist
## Two-Stage Human Action Recognition System (Fall Detection)

Granular, checkable tasks organized by `plan.md` phase. Streamlit-specific state/caching tasks are called out explicitly since they are the most common source of bugs in this architecture.

---

## Phase 0 — Environment & Repo Setup

- [x] Create repo folder structure exactly as specified in `constitution.md` Section 2.
- [x] Write `requirements.txt` with pinned versions: `ultralytics`, `torch`, `torchvision`, `opencv-python`, `streamlit`, `pandas`, `numpy`.
- [x] Create `src/utils/config.py` with placeholder constants (to be filled in later phases): `WINDOW_SIZE`, `TARGET_FPS`, `CONF_THRESHOLD`, `WARNING_THRESHOLD`, `CRITICAL_THRESHOLD`, `INSTANT_RED_THRESHOLD`, `N_WARN_FRAMES`, `N_CRITICAL_FRAMES`, `N_RECOVERY_FRAMES`, model file paths.
- [x] Verify `torch.cuda.is_available()` and record result; decide local vs. cloud training.
- [x] Create a minimal `app.py` with a single `st.title("Fall Detection Prototype")` and confirm `streamlit run src/app.py` launches in browser.
- [x] Add `.gitignore` covering `data/`, `logs/*.csv` (except a committed sample), `__pycache__/`, `.ipynb_checkpoints/`.

---

## Phase 1 — Data Acquisition & Preparation

### 1.1 Person Detection Data
- [ ] Collect/source raw images or video frames containing people.
- [ ] Annotate bounding boxes with a single `person` class.
- [ ] Export in YOLO format (`images/`, `labels/`, `data.yaml`).
- [ ] Split into train/val/test; document split ratios in the notebook.
- [ ] Visualize a sample batch with boxes drawn to sanity-check annotation quality.

### 1.2 Action Sequence Data
- [ ] Source/record clips per class: `Normal`, `Warning/Unsteady`, `Fall`.
- [ ] Decide input representation: raw crops vs. pose keypoints (document decision and rationale in the notebook).
- [ ] Implement the crop/keypoint-extraction function **once**, in a location importable by both the notebook and `src/pipeline/` later (avoid duplication now to prevent train/inference skew).
- [ ] Segment clips into fixed-length windows; assign one label per window.
- [ ] Compute and log class distribution; flag imbalance.
- [ ] Apply augmentation/oversampling strategy for the minority (Fall) class; document parameters used.

---

## Phase 2 — YOLO Training

**Environment:** Kaggle Notebooks (GPU accelerator).

- [ ] Upload Phase 1.1 dataset as a Kaggle Dataset (via Kaggle UI or Kaggle API); record the dataset name/version.
- [ ] Create Kaggle Notebook, attach the dataset, enable GPU accelerator in notebook settings.
- [ ] Add a first cell installing/pinning dependencies (`!pip install -q ultralytics==<version>`); do not rely on preinstalled versions unchecked.
- [ ] Load pretrained checkpoint (`yolov8n.pt` or `yolov8s.pt`).
- [ ] Configure training run (epochs, image size, batch size); read data from `/kaggle/input/...`, write outputs to `/kaggle/working/`.
- [ ] Add periodic checkpoint-saving to `/kaggle/working/` (guard against session timeouts/disconnects).
- [ ] Train; monitor mAP@0.5, precision, recall per epoch.
- [ ] Run inference on held-out test images; visually inspect box tightness.
- [ ] Run inference on a short held-out test **video** (not just images); check for box jitter/flicker across frames.
- [ ] Download final weights from `/kaggle/working/` to local machine; place at `models/yolo_person_detector.pt`.
- [ ] Download the executed `.ipynb` from Kaggle; copy into `notebooks/02_yolo_training.ipynb`, marking Kaggle-only cells per `constitution.md` Section 5.1.
- [ ] Write `models/YOLO_MODEL_CARD.md` (training date, Kaggle Dataset version, final mAP, input size).

---

## Phase 3 — Action Model Training

**Environment:** Kaggle Notebooks (GPU accelerator).

- [ ] Upload Phase 1.2 sequence dataset as a Kaggle Dataset; record the dataset name/version.
- [ ] Create (or reuse) Kaggle Notebook, attach the dataset, enable GPU accelerator in notebook settings.
- [ ] Add a first cell installing/pinning dependencies (e.g., `torch`, pose-estimation library if using keypoint mode).
- [ ] Finalize `WINDOW_SIZE` and record it in `src/utils/config.py` (single source of truth).
- [ ] Copy the shared preprocessing function from Task 1.2 into the notebook as a self-contained cell, clearly commented that it must stay in sync with `src/pipeline/`.
- [ ] Build sequence dataset/dataloader using that preprocessing function, reading from `/kaggle/input/...`.
- [ ] Define LSTM (or GRU) architecture; if raw-frame mode, attach CNN feature extractor (consider freezing backbone initially).
- [ ] Fix random seeds (`torch.manual_seed`, `numpy.random.seed`) for reproducibility.
- [ ] Add periodic checkpoint-saving to `/kaggle/working/` (guard against session timeouts/disconnects).
- [ ] Train; track loss and per-class accuracy.
- [ ] Generate a confusion matrix on the held-out test set; specifically report **Fall-class recall/False Negative rate**.
- [ ] Measure and record per-sequence inference latency (informs `TARGET_FPS`).
- [ ] Download final weights from `/kaggle/working/` to local machine; place at `models/action_lstm.pth`.
- [ ] Download the executed `.ipynb` from Kaggle; copy into `notebooks/03_action_model_training.ipynb`, marking Kaggle-only cells per `constitution.md` Section 5.1.
- [ ] Write `models/ACTION_MODEL_CARD.md` (training date, Kaggle Dataset version, window size, metrics, input shape).

---

## Phase 4 — Pipeline Assembly (Pre-Streamlit)

- [ ] Implement `src/pipeline/detector.py`:
  - [ ] `load_yolo_model(path) -> model`
  - [ ] `detect_persons(frame, model) -> list[BoundingBox]`
  - [ ] `crop_best_box(frame, boxes) -> np.ndarray | None`
- [ ] Implement `src/pipeline/sequence_buffer.py`:
  - [ ] `SequenceBuffer` class wrapping `collections.deque(maxlen=WINDOW_SIZE)`
  - [ ] `.append(preprocessed_frame)`, `.is_ready() -> bool`, `.as_tensor() -> torch.Tensor`
- [ ] Implement `src/pipeline/action_classifier.py`:
  - [ ] `load_action_model(path) -> model`
  - [ ] `classify_sequence(tensor, model) -> ActionPrediction` (dataclass with `label`, `confidence`)
- [ ] Implement `src/pipeline/alert_state.py`:
  - [ ] `AlertLevel` enum (`GREEN`, `YELLOW`, `RED`)
  - [ ] `AlertStateMachine` class with `.update(prediction) -> AlertLevel` and internal consecutive-frame counters
  - [ ] Explicit `.acknowledge()` method required to clear `RED` (no silent auto-clear, per `spec.md` Section 7)
- [ ] Write a standalone CLI/script test: run full pipeline over a test video file, print state transitions with timestamps to console — **no Streamlit import anywhere in this test path**.
- [ ] Confirm the CLI test reproduces the expected Green→Yellow→Red sequence on a known "fall" test clip.
- [ ] Confirm preprocessing (resize/normalize constants) is identical between `src/pipeline/` and the training notebooks — diff the constants explicitly.

---

## Phase 5 — Streamlit Integration

### 5.1 Model Caching (`@st.cache_resource`)
- [ ] Write a `load_models()` function decorated with `@st.cache_resource` that loads **both** the YOLO model and the action LSTM and returns them as a tuple/dataclass.
  ```python
  @st.cache_resource
  def load_models():
      yolo = load_yolo_model(config.YOLO_MODEL_PATH)
      action = load_action_model(config.ACTION_MODEL_PATH)
      return yolo, action
  ```
- [ ] Confirm via logging/print that `load_models()` executes **only once** per process across multiple reruns (add a temporary log line inside, verify it fires once, then remove).
- [ ] Do **not** decorate anything that takes `st.session_state` or mutable per-user objects as an argument with `@st.cache_resource` — cache only the models themselves.
- [ ] Add a spinner (`st.spinner("Loading models...")`) around the first call for UX during cold start.

### 5.2 Session State Initialization
- [ ] At the top of `app.py`, initialize all session keys defensively using `if "key" not in st.session_state:` guards — never unconditional assignment (which would reset state every rerun):
  - [ ] `st.session_state.run_inference = False`
  - [ ] `st.session_state.video_capture = None`
  - [ ] `st.session_state.seq_buffer = SequenceBuffer(config.WINDOW_SIZE)`
  - [ ] `st.session_state.alert_state = AlertStateMachine()`
  - [ ] `st.session_state.last_frame_time = 0.0`
  - [ ] `st.session_state.event_log_path = "logs/events.csv"`
- [ ] Implement "Start" button logic: only opens `cv2.VideoCapture` and resets buffer/state **if not already running** (guard against double-open on accidental double-click/rerun).
- [ ] Implement "Stop" button logic: sets `run_inference = False`, calls `cap.release()`, and explicitly sets `video_capture = None` in session state.
- [ ] Add a session-end/cleanup safeguard: wrap capture access in a check that re-opens gracefully if the handle was lost (e.g., app hot-reloaded mid-session).

### 5.3 Frame Loop Implementation
- [ ] Create persistent placeholders **once**, outside the loop body: `frame_placeholder = st.empty()`, `status_placeholder = st.empty()`.
- [ ] Implement FPS governance: compute `elapsed = time.time() - st.session_state.last_frame_time`; skip processing (but still rerun) if `elapsed < 1 / config.TARGET_FPS`.
- [ ] On each valid tick: read frame → `detect_persons` → `crop_best_box` → `seq_buffer.append` → if `seq_buffer.is_ready()`, `classify_sequence` → `alert_state.update`.
- [ ] Render annotated frame via `frame_placeholder.image(...)` (never a bare `st.image()` call inside the loop).
- [ ] Render alert badge via `status_placeholder` using color-coded markdown/HTML based on `alert_state.level`.
- [ ] Call `st.rerun()` at the end of the tick **only if** `st.session_state.run_inference` is still `True`.
- [ ] Verify the Stop button is clickable and takes effect within one tick while inference is running (manual test, not just code review).

### 5.4 Logging Integration
- [ ] Implement `src/utils/logger.py::log_event(from_state, to_state, confidence, source)` appending a row to `events.csv` (create file with header if it doesn't exist).
- [ ] Call `log_event` only on **state transitions**, not every frame — compare `alert_state`'s previous vs. new level to detect a transition.
- [ ] Add a sidebar "Recent Events" table reading the last N rows via `pandas.read_csv(...).tail(N)`.
- [ ] Add a "Download events.csv" button (`st.download_button`) in the sidebar.

### 5.5 UI Controls (`src/ui/sidebar.py`)
- [ ] Video source selector with three options: Webcam (device index), Upload File (`st.file_uploader`), Phone IP Camera (URL).
- [ ] For "Webcam": add a small numeric input for device index (default `0`; external USB cameras may be `1`+).
- [ ] For "Phone IP Camera": add a text input for the stream URL (e.g., `http://<phone-ip>:8080/video`); do not hardcode any default IP since it changes per network/session.
- [ ] For "Phone IP Camera": add an optional "Rotate 90°" toggle to correct portrait-orientation streams before frames enter Stage 1.
- [ ] Start / Stop buttons wired to session-state flags per Task 5.2.
- [ ] Threshold sliders (Warning/Critical confidence) for live demo tuning — read into `config` overrides, not hardcoded.
- [ ] "Acknowledge Alert" button, enabled only when `alert_state.level == RED`, calling `alert_state.acknowledge()`.

### 5.6 Edge Case Testing
- [ ] Test: no person visible in frame for extended period — confirm no crash, buffer handles gracefully per `spec.md` Section 4.
- [ ] Test: multiple people in frame — confirm box-selection logic (largest/highest-confidence) behaves predictably.
- [ ] Test: video file reaches end of stream — confirm graceful stop, not an exception on `cap.read()` returning `False`.
- [ ] Test: rapid Start/Stop/Start clicking — confirm no duplicate `VideoCapture` handles leak and no duplicate model loads occur.
- [ ] Test: switching video source mid-session — confirm old capture is released before new one opens.
- [ ] Test: phone IP camera — connect via the exact WiFi network planned for the demo venue **in advance**; confirm the laptop can reach the stream URL (client isolation on some networks blocks device-to-device traffic even on the same WiFi).
- [ ] Test: phone IP camera — disconnect/reconnect the stream mid-session (e.g., WiFi drop) and confirm the app surfaces a clear error/status rather than freezing or crashing silently.
- [ ] Test: phone IP camera — verify orientation (portrait vs. landscape) renders correctly with the "Rotate 90°" toggle before relying on it for a live demo.
- [ ] Confirm browser tab CPU/memory stays stable over a 5+ minute continuous run (no unbounded DOM growth, no memory leak from placeholders).

---

## Phase 6 — Hardening & Demo Polish

- [ ] Re-tune `WARNING_THRESHOLD` / `CRITICAL_THRESHOLD` / frame-count constants against Phase 3's confusion matrix, prioritizing Fall-class recall.
- [ ] Add a bundled "demo fall clip" as a selectable video source for reliable live demonstrations.
- [ ] Polish alert badge styling (color, size, iconography) for clear at-a-glance status.
- [ ] Add session summary stats to the UI (e.g., total alerts this session, time in each state).
- [ ] Write top-level `README.md`: setup instructions, `streamlit run` command, model file placement, known limitations (e.g., single-person tracking only, FPS ceiling).
- [ ] Final end-to-end test: fresh environment, fresh clone, models placed manually, full demo run without any developer intervention.
