# Project Constitution
## Two-Stage Human Action Recognition System (Fall Detection)

This document defines the non-negotiable engineering standards for this project. It exists to keep a **single-developer, rapid-prototyping codebase** clean enough to debug at 2 AM and simple enough to demo to a stakeholder the next morning. All contributors (human or AI-assisted) must adhere to these rules.

---

## 1. Guiding Philosophy

- **Simplicity over scalability.** We are not building a production video-surveillance platform. We are building a working, honest, demonstrable prototype. Do not introduce Docker, Kafka, databases, or microservices "for the future." YAGNI (You Aren't Gonna Need It) is law.
- **Streamlit is the only application framework.** No FastAPI backend, no separate React frontend, no Flask. The UI, the inference loop, and the alert state machine all live inside the Streamlit process. Complexity is managed through **file/module separation**, not through service separation.
- **Notebooks are for training. Scripts are for running.** `.ipynb` files never appear in the inference/runtime path. `.py` files never contain exploratory, throwaway training code.
- **Don't train what you can borrow.** Stage 1 (person detection + pose) uses an **off-the-shelf, pretrained** model (COCO-pretrained YOLO-Pose). No custom object-detection dataset, annotation, or training pipeline exists in this project. Engineering effort is concentrated entirely on Stage 2 (the action classifier), which is the actual research/comparison contribution of this project.
- **Skeleton data, not raw pixels, is the ground truth for Stage 2.** The action models never see raw RGB frames. They are trained and run exclusively on extracted joint-coordinate sequences (X, Y, Z per joint, per frame). This keeps Stage 2 lightweight, backbone-free, and directly comparable across architectures (LSTM vs. ST-GCN) since both consume the exact same numeric dataset.

---

## 2. Repository Structure (Mandatory)

```
ActionGuard_AI/
├── notebooks/
│   ├── 01_data_preparation.ipynb        # Video collection + skeleton extraction -> CSV/NumPy dataset
│   ├── 02_action_model_training_lstm.ipynb
│   └── 03_action_model_training_stgcn.ipynb
├── models/
│   ├── yolo_pose_pretrained.pt          # Pretrained COCO YOLO-Pose weights (downloaded, not trained)
│   ├── action_lstm.pth
│   └── action_stgcn.pth
├── src/
│   ├── app.py                  # Streamlit entrypoint ONLY (UI wiring)
│   ├── pipeline/
│   │   ├── detector.py         # Pretrained YOLO-Pose wrapper: person bbox + joint keypoints (Stage 1)
│   │   ├── sequence_buffer.py  # Sliding window buffer of keypoint vectors (not raw frames)
│   │   ├── action_classifier.py# Action model wrapper (Stage 2) — supports LSTM and ST-GCN interchangeably
│   │   └── alert_state.py      # Green/Yellow/Red state machine
│   ├── ui/
│   │   ├── components.py       # Reusable Streamlit render functions
│   │   └── sidebar.py          # Config/controls panel, incl. active-model selector
│   └── utils/
│       ├── config.py           # Central constants (thresholds, paths, window size, joint schema)
│       ├── logger.py           # CSV/JSON event logger
│       ├── video_io.py         # OpenCV capture helpers
│       └── skeleton_utils.py   # Shared keypoint-extraction/normalization logic (used by notebooks AND src/pipeline)
├── logs/
│   └── events.csv              # Append-only alert log (gitignored, sample committed)
├── data/
│   ├── raw_videos/             # Raw labeled action clips (gitignored)
│   └── skeleton_dataset/       # Extracted keypoint sequences as CSV/NumPy (gitignored, schema documented)
├── requirements.txt
└── docs/
    ├── constitution.md
    ├── spec.md
    ├── plan.md
    └── tasks.md
```

**Rule:** `app.py` must remain a thin orchestration layer. If `app.py` exceeds ~200 lines, logic has leaked into it and must be extracted into `src/pipeline/` or `src/ui/`.

**Rule:** `src/utils/skeleton_utils.py` is the **single source of truth** for keypoint extraction and normalization. Both the data-preparation notebook and the live inference pipeline (`src/pipeline/detector.py`) must import from this module — never duplicate the extraction logic in two places (see Section 6).

---

## 3. Streamlit UI/Logic Separation (Critical Rule)

Even though everything runs in one Streamlit process, **UI code and inference/business logic must never be interleaved in the same function.**

### 3.1 The Separation Contract

| Layer | Lives In | May Import Streamlit? | May Contain `st.*` calls? |
|---|---|---|---|
| **Logic** (detection, pose extraction, classification, state machine, logging) | `src/pipeline/`, `src/utils/` | ❌ No | ❌ Never |
| **UI/Rendering** (layout, widgets, display) | `src/ui/`, `app.py` | ✅ Yes | ✅ Yes |

- Functions in `src/pipeline/` must accept plain Python/NumPy/PyTorch inputs and return plain Python objects (dicts, dataclasses, numpy arrays). **Never pass a Streamlit widget object into a pipeline function.**
- If pipeline code needs configuration (including *which* action model is active — LSTM or ST-GCN), it is injected as a function argument or read from `src/utils/config.py` — **not** pulled from `st.session_state` directly inside pipeline modules.
- `app.py` and `src/ui/` are responsible for reading `st.session_state`, calling pipeline functions, and rendering the results.

### 3.2 Anti-Pattern (Forbidden)

```python
# ❌ FORBIDDEN — logic and UI mixed inside app.py
def process_frame(frame):
    boxes, keypoints = pose_model(frame)
    st.write(f"Detected {len(boxes)} people")   # UI call inside logic function
    if is_falling(keypoints):
        st.error("FALL DETECTED")               # UI call inside logic function
```

### 3.3 Correct Pattern

```python
# ✅ src/pipeline/detector.py — pure logic, no Streamlit
def detect_and_extract_pose(frame: np.ndarray, model) -> PoseResult:
    return model(frame)

# ✅ app.py — orchestration + UI only
pose_result = detect_and_extract_pose(frame, yolo_pose_model)
render_skeleton_overlay(frame, pose_result)   # UI function from src/ui/components.py
if alert_state.level == "RED":
    st.error("FALL DETECTED")
```

---

## 4. Clean Code Rules

1. **Type hints are mandatory** on all function signatures in `src/`. Notebooks are exempt.
2. **No magic numbers.** Thresholds (confidence, window size, fall-angle, alert cooldown) and the joint-count/joint-order schema live only in `src/utils/config.py`.
3. **Every pipeline function is pure where possible** — same input, same output, no hidden state mutation, no I/O side effects buried inside.
4. **One state machine, one owner.** The Green/Yellow/Red logic lives exclusively in `src/pipeline/alert_state.py`. No component outside that file may set alert levels.
5. **Docstrings required** for every public function in `src/` — one-line summary minimum, Args/Returns for anything non-trivial.
6. **No bare `except:`.** Catch specific exceptions (`cv2.error`, `RuntimeError`, `torch.cuda.OutOfMemoryError`, etc.) and log them.
7. **Logging over printing.** Use `src/utils/logger.py` for event logs; use Python's `logging` module for debug/console output. `print()` is not permitted outside notebooks.
8. **One model interface, two implementations.** `src/pipeline/action_classifier.py` exposes a single `classify_sequence(keypoint_tensor, model, model_type)` function. LSTM- and ST-GCN-specific tensor reshaping happens *inside* this function, branching on `model_type` — callers (including `app.py`) never need to know the internal tensor shape differences between the two architectures.

---

## 5. Jupyter Notebook Hygiene

Notebooks (`notebooks/`) are for **training and experimentation only**. They are never imported by `src/`.

1. **Linear execution.** A notebook must run top-to-bottom via "Restart & Run All" without error before being committed. No out-of-order cell dependencies.
2. **Clear the outputs of large artifacts.** Strip large image/video cell outputs before commit; keep loss curves and confusion matrices as they aid review.
3. **One notebook, one responsibility**, matching the numbered structure in Section 2 (`01_data_preparation`, `02_action_model_training_lstm`, `03_action_model_training_stgcn`). Do not create ad-hoc notebooks in the repo root.
4. **Markdown cells are mandatory** at the top of every major section explaining *why*, not just *what*.
5. **No hardcoded absolute paths.** Use relative paths from the repo root or a `PROJECT_ROOT` constant.
6. **Export, don't copy-paste.** Final trained weights are saved to `models/`. Hyperparameters used for the winning run are recorded in a markdown cell at the end of the notebook, not left buried in a mid-notebook cell.
7. **Random seeds are fixed** (`torch.manual_seed`, `numpy.random.seed`) for reproducibility of reported metrics.
8. **Shared preprocessing, not copy-pasted preprocessing.** `01_data_preparation.ipynb` must call the exact same `src/utils/skeleton_utils.py` functions that `src/pipeline/detector.py` uses at inference time (import the module directly, or copy it in as a clearly-marked, version-pinned cell per Section 5.1, Rule 4). This is the single most important reproducibility guarantee in this project — a mismatch here silently breaks both action models.

### 5.1 Kaggle-Specific Rules (Training Environment)

Training for this project (the two action-model notebooks, `02_action_model_training_lstm.ipynb` and `03_action_model_training_stgcn.ipynb`) runs on **Kaggle Notebooks**, not on the local machine. Kaggle is a separate, ephemeral, read-only-input environment, so the following additional rules apply whenever a notebook is executed there:

1. **Never hardcode Kaggle paths into `src/`.** `/kaggle/input/...` and `/kaggle/working/...` paths are valid *only* inside the training notebook cells. `src/utils/config.py` must only ever contain local repo paths (`models/action_lstm.pth`, etc.). The mapping from Kaggle output → local `models/` happens manually (download) after training, never via a shared path constant.
2. **Explicit dependency installation cell.** Kaggle does not read this repo's `requirements.txt` automatically. The first code cell of any Kaggle-run notebook must pin and install what's needed (e.g., `torch`, and any ST-GCN-specific graph-convolution dependency). Do not rely on Kaggle's preinstalled package versions without checking them first.
3. **Checkpoint defensively.** Kaggle sessions have hard time limits and can disconnect without warning. Training loops must save intermediate checkpoints to `/kaggle/working/` on a schedule (e.g., every N epochs), not only at the very end.
4. **Reproduce locally before committing.** After downloading a Kaggle-trained notebook (`.ipynb`), it is copied back into `notebooks/` in this repo. Before committing, verify it still satisfies Rule 1 ("Linear execution") — Kaggle-specific `!pip install` and `/kaggle/...` path cells should be clearly marked with a markdown note (e.g., "⚠️ Kaggle-only cell — adjust paths for local re-run") so the notebook remains legible outside Kaggle.
5. **Dataset versioning.** The skeleton dataset (CSV/NumPy, produced by `01_data_preparation.ipynb`) uploaded to Kaggle as a Kaggle Dataset must match the exact version described in Phase 1's data summary. Note the Kaggle Dataset name/version in each training notebook's opening markdown cell so results are traceable back to a specific data snapshot.
6. **Same dataset, both notebooks.** `02_action_model_training_lstm.ipynb` and `03_action_model_training_stgcn.ipynb` must attach the **identical** Kaggle Dataset version. Any comparison between LSTM and ST-GCN is only valid if both were trained/evaluated on the same data split — this is checked explicitly before comparing metrics.

---

## 6. Model & Data Handling

- **Stage 1 (YOLO-Pose) is pretrained and not fine-tuned in this project.** `models/yolo_pose_pretrained.pt` is either downloaded automatically by Ultralytics on first run or manually cached in `models/` for offline/reproducible demo use. No `MODEL_CARD.md` training log is needed for it — instead, record the exact pretrained checkpoint name/version (e.g., `yolov8n-pose.pt`) and source in a short note in `models/`.
- **Stage 2 (action classifiers) are the only trained artifacts in this project.** Both `models/action_lstm.pth` and `models/action_stgcn.pth` are committed (via Git LFS or release assets, not raw git if large), each paired with its own `MODEL_CARD.md` recording: training date, skeleton dataset version, window size, key metrics (accuracy/F1, and specifically Fall-class recall), and expected input tensor shape.
- **Raw datasets are never committed** — this covers both `data/raw_videos/` (source clips) and, generally, `data/skeleton_dataset/` (extracted keypoints), unless a small sample is deliberately kept for smoke-testing.
- **Preprocessing/extraction performed at inference time must be identical to preprocessing performed during training.** Concretely: the same joint schema (which joints, in which order), the same normalization (e.g., relative to hip-center, scale-normalized by torso length), and the same coordinate convention (X, Y, Z or X, Y + confidence) must be used in `src/utils/skeleton_utils.py` and in `01_data_preparation.ipynb`. Any change to this schema is a **breaking change** that invalidates both trained models and must be called out explicitly in both `MODEL_CARD.md` files.
- **Model comparison is a first-class artifact.** Since this project trains two architectures on the same data specifically to compare them, the final comparison (metrics table, confusion matrices side by side, inference latency) is written up as part of `models/ACTION_MODEL_CARD_COMPARISON.md`, not left implicit across two separate notebooks.

---

## 7. Definition of Done

A change is "done" only when:
- [ ] Logic and UI remain separated per Section 3.
- [ ] No magic numbers introduced outside `config.py`.
- [ ] Type hints and docstrings present on new/modified functions.
- [ ] If a notebook was touched, it runs clean end-to-end.
- [ ] If skeleton extraction logic was touched, `src/utils/skeleton_utils.py` remains the single shared implementation used by both notebooks and `src/pipeline/`.
- [ ] Manual smoke test: app launches via `streamlit run src/app.py`, video feed loads, pose skeleton renders correctly, and a full Green→Yellow→Red cycle can be observed or simulated without a crash — for **both** the LSTM and ST-GCN model options.
