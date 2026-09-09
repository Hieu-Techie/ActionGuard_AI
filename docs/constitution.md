# Project Constitution
## Two-Stage Human Action Recognition System (Fall Detection)

This document defines the non-negotiable engineering standards for this project. It exists to keep a **single-developer, rapid-prototyping codebase** clean enough to debug at 2 AM and simple enough to demo to a stakeholder the next morning. All contributors (human or AI-assisted) must adhere to these rules.

---

## 1. Guiding Philosophy

- **Simplicity over scalability.** We are not building a production video-surveillance platform. We are building a working, honest, demonstrable prototype. Do not introduce Docker, Kafka, databases, or microservices "for the future." YAGNI (You Aren't Gonna Need It) is law.
- **Streamlit is the only application framework.** No FastAPI backend, no separate React frontend, no Flask. The UI, the inference loop, and the alert state machine all live inside the Streamlit process. Complexity is managed through **file/module separation**, not through service separation.
- **Notebooks are for training. Scripts are for running.** `.ipynb` files never appear in the inference/runtime path. `.py` files never contain exploratory, throwaway training code.

---

## 2. Repository Structure (Mandatory)

```
ActionGuard_AI/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_yolo_training.ipynb
│   └── 03_action_model_training.ipynb
├── models/
│   ├── yolo_person_detector.pt
│   └── action_lstm.pth
├── src/
│   ├── app.py                  # Streamlit entrypoint ONLY (UI wiring)
│   ├── pipeline/
│   │   ├── detector.py         # YOLO wrapper (Stage 1)
│   │   ├── sequence_buffer.py  # Sliding window / frame buffer logic
│   │   ├── action_classifier.py# LSTM wrapper (Stage 2)
│   │   └── alert_state.py      # Green/Yellow/Red state machine
│   ├── ui/
│   │   ├── components.py       # Reusable Streamlit render functions
│   │   └── sidebar.py          # Config/controls panel
│   └── utils/
│       ├── config.py           # Central constants (thresholds, paths, window size)
│       ├── logger.py           # CSV/JSON event logger
│       └── video_io.py         # OpenCV capture helpers
├── logs/
│   └── events.csv              # Append-only alert log (gitignored, sample committed)
├── data/                       # Raw & annotated datasets (gitignored)
├── requirements.txt
└── docs/
    ├──constitution.md
    ├── spec.md
    ├── plan.md
    └── tasks.md
```

**Rule:** `app.py` must remain a thin orchestration layer. If `app.py` exceeds ~200 lines, logic has leaked into it and must be extracted into `src/pipeline/` or `src/ui/`.

---

## 3. Streamlit UI/Logic Separation (Critical Rule)

Even though everything runs in one Streamlit process, **UI code and inference/business logic must never be interleaved in the same function.**

### 3.1 The Separation Contract

| Layer | Lives In | May Import Streamlit? | May Contain `st.*` calls? |
|---|---|---|---|
| **Logic** (detection, classification, state machine, logging) | `src/pipeline/`, `src/utils/` | ❌ No | ❌ Never |
| **UI/Rendering** (layout, widgets, display) | `src/ui/`, `app.py` | ✅ Yes | ✅ Yes |

- Functions in `src/pipeline/` must accept plain Python/NumPy/PyTorch inputs and return plain Python objects (dicts, dataclasses, numpy arrays). **Never pass a Streamlit widget object into a pipeline function.**
- If pipeline code needs configuration, it is injected as a function argument or read from `src/utils/config.py` — **not** pulled from `st.session_state` directly inside pipeline modules.
- `app.py` and `src/ui/` are responsible for reading `st.session_state`, calling pipeline functions, and rendering the results.

### 3.2 Anti-Pattern (Forbidden)

```python
# ❌ FORBIDDEN — logic and UI mixed inside app.py
def process_frame(frame):
    boxes = yolo_model(frame)
    st.write(f"Detected {len(boxes)} people")   # UI call inside logic function
    if is_falling(boxes):
        st.error("FALL DETECTED")               # UI call inside logic function
```

### 3.3 Correct Pattern

```python
# ✅ src/pipeline/detector.py — pure logic, no Streamlit
def detect_persons(frame: np.ndarray, model) -> list[BoundingBox]:
    return model(frame)

# ✅ app.py — orchestration + UI only
boxes = detect_persons(frame, yolo_model)
render_detection_overlay(frame, boxes)   # UI function from src/ui/components.py
if alert_state.level == "RED":
    st.error("FALL DETECTED")
```

---

## 4. Clean Code Rules

1. **Type hints are mandatory** on all function signatures in `src/`. Notebooks are exempt.
2. **No magic numbers.** Thresholds (confidence, window size, fall-angle, alert cooldown) live only in `src/utils/config.py`.
3. **Every pipeline function is pure where possible** — same input, same output, no hidden state mutation, no I/O side effects buried inside.
4. **One state machine, one owner.** The Green/Yellow/Red logic lives exclusively in `src/pipeline/alert_state.py`. No component outside that file may set alert levels.
5. **Docstrings required** for every public function in `src/` — one-line summary minimum, Args/Returns for anything non-trivial.
6. **No bare `except:`.** Catch specific exceptions (`cv2.error`, `RuntimeError`, `torch.cuda.OutOfMemoryError`, etc.) and log them.
7. **Logging over printing.** Use `src/utils/logger.py` for event logs; use Python's `logging` module for debug/console output. `print()` is not permitted outside notebooks.

---

## 5. Jupyter Notebook Hygiene

Notebooks (`notebooks/`) are for **training and experimentation only**. They are never imported by `src/`.

1. **Linear execution.** A notebook must run top-to-bottom via "Restart & Run All" without error before being committed. No out-of-order cell dependencies.
2. **Clear the outputs of large artifacts.** Strip large image/video cell outputs before commit; keep loss curves and confusion matrices as they aid review.
3. **One notebook, one responsibility**, matching the numbered structure in Section 2 (`01_data_preparation`, `02_yolo_training`, `03_action_model_training`). Do not create ad-hoc notebooks in the repo root.
4. **Markdown cells are mandatory** at the top of every major section explaining *why*, not just *what*.
5. **No hardcoded absolute paths.** Use relative paths from the repo root or a `PROJECT_ROOT` constant.
6. **Export, don't copy-paste.** Final trained weights are saved to `models/`. Hyperparameters used for the winning run are recorded in a markdown cell at the end of the notebook, not left buried in a mid-notebook cell.
7. **Random seeds are fixed** (`torch.manual_seed`, `numpy.random.seed`) for reproducibility of reported metrics.

### 5.1 Kaggle-Specific Rules (Training Environment)

Training for this project (Phase 2 and Phase 3) runs on **Kaggle Notebooks**, not on the local machine. Kaggle is a separate, ephemeral, read-only-input environment, so the following additional rules apply whenever a notebook is executed there:

1. **Never hardcode Kaggle paths into `src/`.** `/kaggle/input/...` and `/kaggle/working/...` paths are valid *only* inside the training notebook cells. `src/utils/config.py` must only ever contain local repo paths (`models/yolo_person_detector.pt`, etc.). The mapping from Kaggle output → local `models/` happens manually (download) after training, never via a shared path constant.
2. **Explicit dependency installation cell.** Kaggle does not read this repo's `requirements.txt` automatically. The first code cell of any Kaggle-run notebook must pin and install what's needed, e.g. `!pip install -q ultralytics==<version>`. Do not rely on Kaggle's preinstalled package versions without checking them first (`!pip show ultralytics`).
3. **Checkpoint defensively.** Kaggle sessions have hard time limits and can disconnect without warning. Training loops must save intermediate checkpoints to `/kaggle/working/` on a schedule (e.g., every N epochs), not only at the very end.
4. **Reproduce locally before committing.** After downloading a Kaggle-trained notebook (`.ipynb`), it is copied back into `notebooks/` in this repo. Before committing, verify it still satisfies Rule 1 ("Linear execution") — Kaggle-specific `!pip install` and `/kaggle/...` path cells should be clearly marked with a markdown note (e.g., "⚠️ Kaggle-only cell — adjust paths for local re-run") so the notebook remains legible outside Kaggle.
5. **Dataset versioning.** Any dataset uploaded to Kaggle as a Kaggle Dataset must match the exact version described in Phase 1's data summary. Note the Kaggle Dataset name/version in the notebook's opening markdown cell so results are traceable back to a specific data snapshot.

---

## 6. Model & Data Handling

- Trained weights (`.pt`, `.pth`) are the only model artifacts committed to `models/` (via Git LFS or release assets, not raw git if large). Raw datasets are never committed.
- Every model file must be paired with a short `MODEL_CARD.md` note (in `models/`) recording: training date, dataset version, key metrics (mAP for YOLO; accuracy/F1 for the action classifier), and input shape expectations.
- Preprocessing performed at inference time (`src/pipeline/`) must be **identical** to preprocessing performed during training (notebooks). Any resize/normalization constant must be defined once in `src/utils/config.py` and imported by both.

---

## 7. Definition of Done

A change is "done" only when:
- [ ] Logic and UI remain separated per Section 3.
- [ ] No magic numbers introduced outside `config.py`.
- [ ] Type hints and docstrings present on new/modified functions.
- [ ] If a notebook was touched, it runs clean end-to-end.
- [ ] Manual smoke test: app launches via `streamlit run src/app.py`, webcam/video feed loads, and a full Green→Yellow→Red cycle can be observed or simulated without a crash.
