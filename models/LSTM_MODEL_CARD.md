# Model Card — ActionLSTM (Baseline Model)

## Overview
- **Model File:** `models/action_lstm.pth`
- **Architecture:** 2-layer Bidirectional LSTM (`input_size=51`, `hidden_size=128`, `dropout=0.3`, `fc: 256 -> 8`)
- **Trainable Parameters:** `582,664`
- **Training Date:** `2026-11-16`
- **Training Environment:** Kaggle Notebook (GPU accelerator), `notebooks/02_action_model_training_lstm.ipynb`
- **Dataset Split Strategy:** Leak-free video-level stratified split (`SEED=42`), exact 10.0 FPS timestamp resampling:
  - Train: `3,175` windows (`2,414` base + `761` minority-only augmentation from `393` clips; post-aug imbalance `1.80×`)
  - Validation: `524` pure un-augmented windows (`83` clips)
  - Test: `506` pure un-augmented windows (`83` clips: `13` Fall, `70` Non-Fall)
- **Window Size:** `WINDOW_SIZE = 16` frames (`1.6s` @ `TARGET_FPS = 10`)
- **Input Tensor Shape:** `(batch, 16, 51)` — flattened `(x_norm, y_norm, conf)` across 17 COCO keypoints
- **Class Label Order (`0..7`):** `['Walk', 'Eat', 'Drink', 'Fall', 'Lying', 'Sit_Down', 'Stand_Up', 'Crouching']`

---

## Evaluation Results (Held-Out Test Set: 506 Windows / 83 Unseen Videos)

### 1. Summary Metrics
| Scope | Metric | Result |
|---|---|---|
| **Validation (524 windows)** | 8-Class Accuracy / Fall Class Recall | `69.08%` / `97.59%` |
| **Test Window-Level (506 windows)** | 8-Class Accuracy / Macro-F1 | **`66.60%`** / **`0.6491`** |
| **Test Window-Level (`Fall`)** | Fall Class Recall / False Negative Rate | **`79.31%`** / **`20.69%`** |
| **Test Window-Level (`Fall`)** | Fall Class Precision / F1 | `75.82%` / **`0.7753`** |
| **Test 3-Level Alert (Window)** | Alert Accuracy / Alert Macro-F1 | **`88.54%`** / **`0.8307`** |
| **Test Video-Level Mean-Prob (83 clips)** | 8-Class Video Accuracy / Macro-F1 | **`75.90%`** / **`0.7284`** |
| **CPU Inference Latency** | Mean ± Std per sequence (100 runs, CPU) | **`1.402 ± 0.468 ms`** |

### 2. Per-Class Metrics (Test Window-Level, n=506)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `Walk` | `0.7647` | `0.8814` | `0.8189` | 118 |
| `Eat` | `0.5875` | `0.5341` | `0.5595` | 88 |
| `Drink` | `0.3390` | `0.2469` | `0.2857` | 81 |
| **`Fall`** | **`0.7582`** | **`0.7931`** | **`0.7753`** | **87** |
| `Lying` | `0.7500` | `0.6742` | `0.7101` | 89 |
| `Sit_Down` | `0.5882` | `0.9091` | `0.7143` | 11 |
| `Stand_Up` | `0.5000` | `0.5455` | `0.5217` | 11 |
| `Crouching` | `0.6774` | `1.0000` | `0.8077` | 21 |

### 3. 8×8 Confusion Matrix (rows=true, cols=pred)
Order: `[Walk, Eat, Drink, Fall, Lying, Sit_Down, Stand_Up, Crouching]`
```text
[[104   0   1   3   1   5   3   1]
 [  3  47  38   0   0   0   0   0]
 [ 28  29  20   0   4   0   0   0]
 [  0   0   0  69  12   0   0   6]
 [  1   4   0  19  60   0   3   2]
 [  0   0   0   0   1  10   0   0]
 [  0   0   0   0   2   2   6   1]
 [  0   0   0   0   0   0   0  21]]
```

**Postural Fall Confusion:**
- `Fall` misclassified as `Lying`: **12 / 87** (`13.79%`) — remains in alert (`YELLOW`) rather than silent `GREEN`
- `Fall` misclassified as `Crouching`: **6 / 87** (`6.90%`) — remains in alert (`YELLOW`)
- `Fall` misclassified as `Sit_Down` or any `GREEN` class: **0 / 87 (`0.00%`)** (zero silent misses on `Fall` windows!)
- `Lying` misclassified as `Fall`: **19 / 89** (`21.35%`)

### 4. 3-Level Alert Mapping (GREEN / YELLOW / RED)
| Alert Level | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `GREEN` (`Walk, Eat, Drink, Sit_Down, Stand_Up`) | `0.9737` | `0.9579` | `0.9657` | 309 |
| `YELLOW` (`Crouching, Lying`) | `0.7477` | `0.7545` | `0.7511` | 110 |
| `RED` (`Fall`) | `0.7582` | `0.7931` | `0.7753` | 87 |

```text
3x3 Alert CM (rows=true [G, Y, R], cols=pred [G, Y, R]):
[[296  10   3]
 [  8  83  19]
 [  0  18  69]]
```

### 5. Dual-Cadence Video-Level Stream Replay (83 Unseen Test Videos: 13 Fall, 70 Non-Fall)
| Cadence Mode | Rule 1: Fall Clip Recall | Rule 1: Video FPR | Rule 2: RED Trigger Recall | Rule 2: RED FPR | Mean Delay (1st→RED) |
|---|---|---|---|---|---|
| **Target-Cadence (10.0 FPS)** | `11/13` (**`84.62%`**) | `5/70` (`7.14%`) | `10/13` (**`76.92%`**) | `3/70` (**`4.29%`**) | `0.050 s` |
| **CPU-Realistic (8.33 FPS)** | `10/13` (`76.92%`) | `6/70` (`8.57%`) | `9/13` (`69.23%`) | `2/70` (**`2.86%`**) | `0.000 s` |

On the 87 held-out Test `Fall` windows: mean $P(\text{Fall}) = 0.766$; `62/87` windows exceed `CRITICAL_THRESHOLD (0.75)` and `58/87` exceed `INSTANT_RED_THRESHOLD (0.90)`.
