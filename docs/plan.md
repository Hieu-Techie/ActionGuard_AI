# Project Plan
## Two-Stage Human Action Recognition System (Fall Detection)

Chronological roadmap. Each phase has an explicit **exit criterion** — do not proceed to the next phase until it is met. This is a prototyping plan, so phases are intentionally sequential and single-track rather than parallelized across a team.

---

## Phase 0 — Environment & Repo Setup
**Goal:** A working, reproducible environment before any model work begins.

- Initialize repository with the structure defined in `constitution.md`.
- Create `requirements.txt` pinning: `ultralytics`, `torch`/`torchvision`, `opencv-python`, `streamlit`, `pandas`, `numpy`.
- Verify GPU availability (`torch.cuda.is_available()`) if training locally; otherwise confirm a cloud/Colab training path.
- Smoke-test: launch a bare `streamlit run src/app.py` "Hello World" to confirm the toolchain works end-to-end before any real logic is written.

**Exit criterion:** Empty Streamlit app runs locally; environment is reproducible via `requirements.txt`.

---

## Phase 1 — Data Acquisition & Preparation
**Goal:** Two labeled datasets ready — one for person detection, one for action sequences.

### 1.1 Person Detection Data (for YOLO)
- Source images/frames containing people (public fall-detection datasets, e.g., UR Fall Dataset, Le2i, or a custom-recorded set).
- Annotate bounding boxes (single class: `person`) using a tool such as CVAT, LabelImg, or Roboflow.
- Split into `train/val/test` in YOLO-expected directory format; generate the `data.yaml` config.
- Perform this work in `notebooks/01_data_preparation.ipynb`.

### 1.2 Action Sequence Data (for LSTM)
- Source or record short video clips labeled by action class (`Normal / Walking`, `Unsteady`, `Fall`).
- Decide input representation now (per `spec.md` Section 6): **raw cropped-frame sequences** vs. **pose-keypoint sequences**. Recommendation: start with pose keypoints (via a pretrained pose estimator, e.g., MediaPipe or a lightweight COCO-keypoint model) — smaller, faster to train, more robust for a prototype.
- Segment videos into fixed-length windows matching `WINDOW_SIZE` (defined in Phase 3), with appropriate class labels per window.
- Address class imbalance (falls are rare events) via oversampling, synthetic augmentation (rotation/speed variation), or weighted loss — decided and documented here, not deferred.

**Exit criterion:** Two clean, versioned datasets exist on disk with a documented directory layout and class distribution summary.

---

## Phase 2 — Stage 1: YOLO Training (Spatial Detection)
**Goal:** A `.pt` weight file that reliably detects `person` bounding boxes.

**Training environment: Kaggle Notebooks** (GPU accelerator enabled). Local `notebooks/02_yolo_training.ipynb` holds the reproducible copy of what ran on Kaggle; the actual training execution happens on Kaggle's infrastructure, not the local machine.

- Package the Phase 1.1 dataset as a **Kaggle Dataset** (upload via Kaggle UI or Kaggle API) rather than transferring files manually per session.
- Create a Kaggle Notebook, attach the dataset, and enable the GPU accelerator in notebook settings before running anything.
- First code cell installs/pins dependencies explicitly (`!pip install -q ultralytics`) — Kaggle's preinstalled versions are not assumed.
- Fine-tune a pretrained YOLOv8/11 checkpoint (`yolov8n.pt` or `yolov8s.pt` for prototype-speed) on the dataset, reading from `/kaggle/input/...` and writing checkpoints/outputs to `/kaggle/working/`.
- Save intermediate checkpoints periodically to `/kaggle/working/` to guard against Kaggle's session time limits and unexpected disconnects.
- Track mAP@0.5, precision, recall across epochs; log final metrics in a markdown cell per `constitution.md` Section 5.6.
- Run inference sanity checks on a handful of held-out images/video frames — visually confirm boxes are tight and stable (not just numerically good mAP).
- **Download** the final weights and the executed notebook from Kaggle. Place the weights at `models/yolo_person_detector.pt`, write an accompanying `MODEL_CARD.md`, and copy the notebook into `notebooks/02_yolo_training.ipynb`, annotating any Kaggle-only cells per `constitution.md` Section 5.1.

**Exit criterion:** YOLO model detects persons in held-out test frames with acceptable mAP and visually stable, non-jittery boxes across consecutive frames of a test video, and the trained weights + notebook are downloaded and present locally in the repo structure.

---

## Phase 3 — Stage 2: Action Model Training (Temporal Classification)
**Goal:** A `.pth` weight file that classifies a sequence of frames/keypoints into `Normal / Warning / Fall`.

**Training environment: Kaggle Notebooks** (GPU accelerator enabled), same rationale as Phase 2. Local `notebooks/03_action_model_training.ipynb` holds the reproducible copy of what ran on Kaggle.

- Package the Phase 1.2 sequence dataset as a Kaggle Dataset and attach it to a new (or the same) Kaggle Notebook, with GPU accelerator enabled.
- First code cell installs/pins dependencies explicitly (matching what training requires — e.g., `torch`, and a pose-estimation library if using keypoint mode).
- Finalize `WINDOW_SIZE` (sliding window length, e.g., 16–30 frames) — this value becomes a hard contract with `src/utils/config.py` and must not silently diverge between training and inference code.
- Generate training sequences from Phase 1.2 data using the **same crop/keypoint-extraction logic** that will run at inference time (extract this preprocessing into a shared, importable function early — do not duplicate it between notebook and `src/`; if the function needs to run inside the Kaggle notebook, copy it in as a self-contained cell with a clear comment that it must stay in sync with `src/pipeline/`).
- Build and train the LSTM (or GRU) sequence classifier; if using raw-frame mode, prepend a lightweight CNN feature extractor (consider freezing pretrained weights initially to reduce training time).
- Save intermediate checkpoints periodically to `/kaggle/working/` to guard against session time limits.
- Evaluate with a confusion matrix — pay particular attention to **False Negatives on the Fall class** (a missed fall is the costliest error type in this domain) over raw accuracy.
- **Download** the final weights and the executed notebook from Kaggle. Place the weights at `models/action_lstm.pth`, write an accompanying `MODEL_CARD.md`, and copy the notebook into `notebooks/03_action_model_training.ipynb`, annotating any Kaggle-only cells per `constitution.md` Section 5.1.

**Exit criterion:** Action model achieves an acceptable recall on the Fall class on held-out sequences, inference latency per sequence is measured and recorded (this number directly informs the `TARGET_FPS` decision in Phase 4), and the trained weights + notebook are downloaded and present locally in the repo structure.

---

## Phase 4 — Pipeline Assembly (Pre-Streamlit)
**Goal:** Stage 1 and Stage 2 chained together as pure Python, fully decoupled from any UI, and testable from the command line.

- Implement `src/pipeline/detector.py`, `src/pipeline/sequence_buffer.py`, `src/pipeline/action_classifier.py`, `src/pipeline/alert_state.py` per the module contracts in `constitution.md` and `spec.md`.
- Implement `src/utils/config.py` centralizing: `WINDOW_SIZE`, `TARGET_FPS`, confidence thresholds, alert-transition frame counts, model paths.
- Write a standalone script (or notebook cell) that runs the full pipeline over a saved test video file, frame by frame, printing alert-state transitions to the console — **with zero Streamlit involved**. This isolates pipeline bugs from UI/rendering bugs.

**Exit criterion:** Running the pipeline against a known "person falls" test clip produces the expected Green → Yellow → Red transition sequence from the command line, with correct timing.

---

## Phase 5 — Streamlit Integration
**Goal:** The validated pipeline wired into the live Streamlit app per `spec.md`'s frame-loop and session-state architecture.

- Implement `src/utils/logger.py` (CSV/JSON event logging).
- Implement `src/ui/components.py` (frame display, alert badge, recent-events table) and `src/ui/sidebar.py` (video source selector, Start/Stop controls, threshold sliders for demo purposes).
- Implement `app.py` as the thin orchestrator: initialize `st.session_state`, wire `@st.cache_resource` model loading, implement the bounded frame-tick + `st.rerun()` loop from `spec.md` Section 3.1.
- Test with **three video source modes**: (a) uploaded video file (deterministic, good for demos/regression testing), (b) laptop webcam (if available/needed), (c) phone camera via an IP Webcam-style app (recommended for in-person live demos — see `spec.md` Section 2.1).
- Deliberately test edge cases: no person in frame, multiple people in frame, camera disconnect mid-session, rapid Start/Stop clicking (session-state race conditions), and — specific to the phone camera source — WiFi client isolation blocking the connection, mid-session WiFi drops, and portrait/landscape orientation handling.

**Exit criterion:** The full app runs end-to-end in a browser: a user starts the feed, the system buffers, classifies, transitions through alert states visibly in the UI, logs the transition to `events.csv`, and the app remains responsive (Stop button works instantly) throughout.

---

## Phase 6 — Hardening & Demo Polish
**Goal:** A stable, presentable prototype.

- Tune alert thresholds against Phase 3's confusion matrix findings (favor recall on Fall over precision, per domain risk).
- Add a "simulate fall" mode using a pre-recorded clip for reliable live demos (real falls are hard to reproduce on demand).
- If demoing live with a phone-as-camera setup: verify the WiFi network at the demo venue in advance (client isolation can silently block the connection), keep the phone charging during the demo, and rehearse the exact Start → live action → alert sequence at least once beforehand.
- Basic UI polish: clear color-coded alert badge, session summary stats, a way to download `events.csv` from the sidebar.
- Write a top-level `README.md` covering setup, `streamlit run` instructions, and known limitations.

**Exit criterion:** A fresh clone of the repo, with `requirements.txt` installed and model weights placed in `models/`, can run `streamlit run src/app.py` and complete a full demo without developer intervention.

---

## Summary Timeline (Sequential Dependencies)

```
Phase 0 (Setup)
   └─▶ Phase 1 (Data)
          └─▶ Phase 2 (YOLO Training) ──┐
          └─▶ Phase 3 (Action Training) ┤
                                          ▼
                                   Phase 4 (Pipeline Assembly)
                                          ▼
                                   Phase 5 (Streamlit Integration)
                                          ▼
                                   Phase 6 (Hardening & Demo Polish)
```

Note: Phase 2 and Phase 3 both depend on Phase 1 but are independent of each other and may be reordered or interleaved; both must complete before Phase 4. Both also share Kaggle's GPU quota (free tier: roughly 30 GPU-hours/week, ~9–12h per session before forced disconnect) — budget training runs accordingly and rely on the checkpointing discipline described in each phase.
