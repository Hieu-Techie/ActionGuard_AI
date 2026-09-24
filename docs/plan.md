# Project Plan
## Two-Stage Human Action Recognition System (Fall Detection)

Chronological roadmap. Each phase has an explicit **exit criterion** — do not proceed to the next phase until it is met. This is a prototyping plan, so phases are intentionally sequential and single-track rather than parallelized across a team.

---

## Phase 0 — Environment & Repo Setup
**Goal:** A working, reproducible environment before any model work begins.

- Initialize repository with the structure defined in `constitution.md`.
- Create `requirements.txt` pinning: `ultralytics` (for pretrained YOLO-Pose inference), `torch`/`torchvision`, `opencv-python`, `streamlit`, `pandas`, `numpy`, and a pose-extraction library if using MediaPipe as an alternative/supplement to YOLO-Pose (`mediapipe`).
- Verify GPU availability (`torch.cuda.is_available()`) locally — mainly relevant for smooth live inference during demos, since training itself happens on Kaggle (Phase 3).
- Smoke-test: launch a bare `streamlit run src/app.py` "Hello World" to confirm the toolchain works end-to-end before any real logic is written.

**Exit criterion:** Empty Streamlit app runs locally; environment is reproducible via `requirements.txt`.

---

## Phase 1 — Data Acquisition & Skeleton Extraction
**Goal:** One clean, labeled **skeleton dataset** (CSV/NumPy) ready to train both action models. There is no separate person-detection dataset in this project — Stage 1 uses a pretrained model (Phase 2).

### 1.1 Action Video Collection
- Source or record short video clips labeled by action class (`Normal / Walking`, `Unsteady`, `Fall`) — from public fall-detection datasets (e.g., UR Fall Dataset, Le2i) and/or custom-recorded clips.
- Store raw clips under `data/raw_videos/`, organized by class folder.
- Address class imbalance at the source where possible (falls are rare events) — collect deliberately more Fall/Unsteady clips than the natural ratio would suggest, since augmentation happens later on the extracted skeletons, not the raw video.

### 1.2 Pretrained-Model Pose/Keypoint Extraction
- Run the pretrained YOLO-Pose model (or MediaPipe, per the final choice recorded in `spec.md` Section 4) over every raw clip, frame by frame, to extract per-frame joint coordinates `(X, Y, Z)` or `(X, Y, confidence)`.
- Implement this extraction **once**, in `src/utils/skeleton_utils.py`, and call it identically from this notebook and from `src/pipeline/detector.py` at inference time — this is the single most important reproducibility guarantee in the project (see `constitution.md` Section 6).
- Apply the chosen normalization (e.g., hip-centered, torso-length-scaled) at extraction time, not at training time, so the saved dataset is already in the exact numeric form both models will train on.
- Segment each clip's extracted keypoint sequence into fixed-length windows matching `WINDOW_SIZE` (finalized in Phase 3), with one class label per window.
- Save the resulting dataset under `data/skeleton_dataset/` as CSV or NumPy arrays, with a documented schema (joint order, coordinate convention, window shape).
- Compute and record class distribution; apply oversampling, synthetic augmentation (e.g., small rotation/jitter/time-warping applied to the *skeleton* coordinates), or weighted loss for the minority Fall class — decided and documented here, not deferred.
- Perform this work in `notebooks/01_data_preparation.ipynb`.

**Exit criterion:** A single versioned skeleton dataset exists on disk (`data/skeleton_dataset/`) with a documented schema and class distribution summary — this exact dataset will be uploaded to Kaggle unchanged for both action-model training runs (Phase 3).

---

## Phase 2 — Stage 1 Setup: Pretrained Pose Detector Integration
**Goal:** A working, validated Stage 1 component using an off-the-shelf model — **no training occurs in this phase.**

- Select and download a pretrained YOLO-Pose checkpoint (e.g., Ultralytics `yolov8n-pose.pt` for speed, or `yolov8s-pose.pt`/`yolov8m-pose.pt` for accuracy) — Ultralytics auto-downloads on first use, optionally cache the file in `models/yolo_pose_pretrained.pt` for offline reproducibility.
- Implement `src/pipeline/detector.py::detect_and_extract_pose(frame, model) -> PoseResult`, returning both the bounding box and joint keypoints for the selected person, per `spec.md` Section 4.
- Validate qualitatively on a handful of sample frames/videos representative of the eventual demo conditions (lighting, distance from camera, single vs. multiple people) — confirm keypoints are stable and anatomically plausible (no wild jitter or joint swapping between frames).
- If validation reveals the pretrained model struggles under demo-realistic conditions (e.g., poor lighting, unusual camera angle), this is the point to reconsider model size (`n` → `s`/`m`) or switch to MediaPipe — before any downstream dataset work depends on the choice.

**Exit criterion:** `detect_and_extract_pose()` reliably returns stable, anatomically plausible keypoints on representative test video, and the exact pretrained checkpoint used is recorded (name/version) for reproducibility.

---

## Phase 3 — Stage 2: Action Model Training (LSTM vs. ST-GCN Comparison)
**Goal:** Two trained action classifiers — an LSTM and an ST-GCN — trained on the **identical** skeleton dataset from Phase 1, so their performance can be fairly compared.

**Training environment: Kaggle Notebooks** (GPU accelerator enabled) for both models. Local `notebooks/02_action_model_training_lstm.ipynb` and `notebooks/03_action_model_training_stgcn.ipynb` hold the reproducible copies of what ran on Kaggle.

### 3.1 Shared Setup (Done Once)
- Package the Phase 1 skeleton dataset as a single **Kaggle Dataset** (upload via Kaggle UI or Kaggle API); this exact dataset version is attached to **both** training notebooks — never re-extracted or re-versioned independently per model, or the comparison becomes invalid.
- Finalize `WINDOW_SIZE`, the joint schema, and (for ST-GCN) the fixed skeleton adjacency matrix; record all three in `src/utils/config.py` as the single source of truth shared by both notebooks.

### 3.2 Model A — LSTM (`notebooks/02_action_model_training_lstm.ipynb`)
- Create a Kaggle Notebook, attach the shared skeleton dataset, enable GPU accelerator.
- First code cell installs/pins dependencies explicitly.
- Build the LSTM architecture per `spec.md` Section 6.1 (flattened per-frame keypoint vectors over time).
- Fix random seeds; save intermediate checkpoints periodically to `/kaggle/working/` to guard against session time limits.
- Train; track loss and per-class accuracy.
- Generate a confusion matrix on the held-out test set — specifically report **Fall-class recall/False Negative rate**.
- Measure and record per-sequence inference latency.
- Download final weights and the executed notebook. Place weights at `models/action_lstm.pth`, write `models/LSTM_MODEL_CARD.md`, copy the notebook into `notebooks/02_action_model_training_lstm.ipynb`, annotating Kaggle-only cells per `constitution.md` Section 5.1.

### 3.3 Model B — ST-GCN (`notebooks/03_action_model_training_stgcn.ipynb`)
- Create a Kaggle Notebook, attach the **same** shared skeleton dataset, enable GPU accelerator.
- First code cell installs/pins dependencies explicitly (including any graph-convolution-specific packages).
- Build the ST-GCN architecture per `spec.md` Section 6.2, using the fixed adjacency matrix finalized in Section 3.1 above — verify the adjacency matrix matches the joint schema exactly (a mismatch silently produces a nonsensical graph).
- Fix random seeds; save intermediate checkpoints periodically to `/kaggle/working/`.
- Train; track loss and per-class accuracy using **the same train/val/test split** as the LSTM run (use a fixed split file or fixed seed shared between both notebooks — do not re-shuffle independently).
- Generate a confusion matrix on the held-out test set — specifically report **Fall-class recall/False Negative rate**.
- Measure and record per-sequence inference latency.
- Download final weights and the executed notebook. Place weights at `models/action_stgcn.pth`, write `models/STGCN_MODEL_CARD.md`, copy the notebook into `notebooks/03_action_model_training_stgcn.ipynb`, annotating Kaggle-only cells per `constitution.md` Section 5.1.

### 3.4 Comparison
- Write `models/ACTION_MODEL_CARD_COMPARISON.md` summarizing both models side by side: accuracy, per-class F1, Fall-class recall, inference latency, and any qualitative observations (e.g., one model reacting faster to onset-of-fall frames).
- This comparison directly informs which model becomes the **default** `ACTIVE_ACTION_MODEL` in `config.py` for the live demo (Phase 5/6), while both remain available via the runtime toggle (`spec.md` Section 6.3).

**Exit criterion:** Both models are trained on the identical dataset/split, both achieve acceptable recall on the Fall class, both sets of weights + notebooks are downloaded and present locally, and a written comparison exists.

---

## Phase 4 — Pipeline Assembly (Pre-Streamlit)
**Goal:** Stage 1 (pretrained pose detector) and Stage 2 (both action models) chained together as pure Python, fully decoupled from any UI, and testable from the command line.

- Implement `src/pipeline/detector.py`, `src/pipeline/sequence_buffer.py`, `src/pipeline/action_classifier.py`, `src/pipeline/alert_state.py`, and `src/utils/skeleton_utils.py` per the module contracts in `constitution.md` and `spec.md`.
- Implement `src/utils/config.py` centralizing: `WINDOW_SIZE`, `TARGET_FPS`, confidence thresholds, alert-transition frame counts, joint schema, adjacency matrix (ST-GCN), model paths, and `ACTIVE_ACTION_MODEL`.
- Write a standalone script (or notebook cell) that runs the full pipeline over a saved test video file, frame by frame, printing alert-state transitions to the console — **with zero Streamlit involved**. Run it once with `ACTIVE_ACTION_MODEL = "lstm"` and once with `"stgcn"` to confirm both code paths work end-to-end. This isolates pipeline bugs from UI/rendering bugs.

**Exit criterion:** Running the pipeline against a known "person falls" test clip produces the expected Green → Yellow → Red transition sequence from the command line, with correct timing, **for both action models**.

---

## Phase 5 — Streamlit Integration
**Goal:** The validated pipeline wired into the live Streamlit app per `spec.md`'s frame-loop and session-state architecture.

- Implement `src/utils/logger.py` (CSV/JSON event logging, including the `active_model` column).
- Implement `src/ui/components.py` (frame display, skeleton overlay, alert badge, recent-events table) and `src/ui/sidebar.py` (video source selector, active-model toggle, Start/Stop controls, threshold sliders for demo purposes).
- Implement `app.py` as the thin orchestrator: initialize `st.session_state`, wire `@st.cache_resource` loading for the pose detector **and both** action models, implement the bounded frame-tick + `st.rerun()` loop from `spec.md` Section 3.1.
- Test with **three video source modes**: (a) uploaded video file (deterministic, good for demos/regression testing), (b) laptop webcam (if available/needed), (c) phone camera via an IP Webcam-style app (recommended for in-person live demos — see `spec.md` Section 2.1).
- Test the active-model toggle mid-session: confirm switching between LSTM and ST-GCN takes effect immediately without reloading the app or losing the alert state history.
- Deliberately test edge cases: no person in frame, multiple people in frame, camera disconnect mid-session, rapid Start/Stop clicking (session-state race conditions), and — specific to the phone camera source — WiFi client isolation blocking the connection, mid-session WiFi drops, and portrait/landscape orientation handling.

**Exit criterion:** The full app runs end-to-end in a browser: a user starts the feed, the system buffers, classifies (with either model), transitions through alert states visibly in the UI, logs the transition to `events.csv` with the correct `active_model`, and the app remains responsive (Stop button works instantly) throughout.

---

## Phase 6 — Hardening & Demo Polish
**Goal:** A stable, presentable prototype.

- Tune alert thresholds against Phase 3's confusion matrix findings for **whichever model is the chosen default** (favor recall on Fall over precision, per domain risk); confirm thresholds are still reasonable for the non-default model too, or expose per-model threshold overrides if the two models' confidence distributions differ meaningfully.
- Add a "simulate fall" mode using a pre-recorded clip for reliable live demos (real falls are hard to reproduce on demand).
- Consider a "live comparison" demo mode: run the same buffered window through both models and display both predictions side by side, so the LSTM-vs-ST-GCN comparison can be shown live, not just as offline metrics.
- If demoing live with a phone-as-camera setup: verify the WiFi network at the demo venue in advance (client isolation can silently block the connection), keep the phone charging during the demo, and rehearse the exact Start → live action → alert sequence at least once beforehand.
- Basic UI polish: clear color-coded alert badge, session summary stats, a way to download `events.csv` from the sidebar.
- Write a top-level `README.md` covering setup, `streamlit run` instructions, model comparison summary, and known limitations.

**Exit criterion:** A fresh clone of the repo, with `requirements.txt` installed and both sets of model weights placed in `models/`, can run `streamlit run src/app.py` and complete a full demo — including switching between LSTM and ST-GCN — without developer intervention.

---

## Summary Timeline (Sequential Dependencies)

```
Phase 0 (Setup)
   └─▶ Phase 1 (Data: Video Collection + Skeleton Extraction)
          └─▶ Phase 2 (Pretrained Pose Detector Integration — no training)
          └─▶ Phase 3 (Action Model Training: LSTM + ST-GCN, same dataset)
                                          │
                                          ▼
                                   Phase 4 (Pipeline Assembly)
                                          ▼
                                   Phase 5 (Streamlit Integration)
                                          ▼
                                   Phase 6 (Hardening & Demo Polish)
```

Note: Phase 2 (pretrained detector validation) and Phase 3 (action model training) both depend on Phase 1's skeleton dataset but are otherwise independent and may be reordered or interleaved; both must complete before Phase 4. The two tracks inside Phase 3 (LSTM and ST-GCN) share Kaggle's GPU quota (free tier: roughly 30 GPU-hours/week, ~9–12h per session before forced disconnect) — budget both training runs accordingly and rely on the checkpointing discipline described in Phase 3.
