# Model Card — ActionSTGCN (`ST-GCN v4 Dual-Specialist` Proposed Default, `v3.1-C` & Earlier Baselines)

## Overview
- **Active Production Checkpoints (`ST-GCN v4 Dual-Specialist`):**
  - `models/action_stgcn.pth` (byte-identical to `models/action_stgcn_v4_c.pth` — **`ST-GCN v4-C (+ Lower-Body Kinematics)`**, `280,999` params, trained on temporally cleaned + in-the-wild augmented Dataset **V4**)
  - `models/action_stgcn_v4_b.pth` (**`ST-GCN v4-B (+ Wrist-to-Head Features)`**, `280,999` params, paired automatically at runtime via `STGCN_WRIST_MODEL_PATH` for per-class Dual-Specialist fusion + max `Fall` safety gate)
- **Archived Checkpoints:**
  - `models/action_stgcn_v3_1.pth` — **`ST-GCN v3.1-C (+ Lower-Body Kinematics) [Weighted]`** (`280,999` params, trained on group-heldout Dataset **V3b**)
  - `models/action_stgcn_v3_0_clean.pth` — **`ST-GCN v3.0-clean [Unweighted]`** (`279,251` params, `v2.4` architecture trained on Dataset **V3b**)
  - `models/action_stgcn_v2_4.pth` — **`ST-GCN v2.4 [Unweighted]`** (`279,251` params, trained on original Dataset **V1**)
  - `models/action_stgcn_v1.pth` — **`ST-GCN v1`** (`409,902` params, minimal 4-block unpartitioned baseline on **V1**)
- **Winning Architecture (`ST-GCN v3.1-C`, `280,999` trainable parameters):**
  - **Multi-Cue Input Stem (`C_in = 11`):**
    1. Base 9-channel kinematics (`v2.4`/`v3.0-clean`): Joint `(x_norm, y_norm, conf)` (`3` ch) + Anatomical Bone vectors & length `(dx_bone, dy_bone, len)` (`3` ch) + Temporal Velocity & speed `(vx, vy, speed)` (`3` ch).
    2. **Lower-Body Node Channels (`+2` ch in `v3.1-C`):** Per-joint vertical height relative to ankle midpoint (`y_v - y_ankle_mid`) and vertical velocity relative to ankle midpoint (`vy_v - vy_ankle_mid`).
  - **Spatial-Temporal Graph Backbone:** 3-subset partitioned spatial adjacency `(3, 17, 17)` + learnable adaptive adjacency `A_adap` + parallel multi-scale temporal convolutions (`k=3` & `k=5`) across 4 residual blocks (`64 -> 64 -> 128 -> 128`).
  - **Hybrid Temporal Pyramid + Lower-Body Geometric Readout (`416 -> 8`):**
    - GCN Temporal Attention & Directional Pyramid Pooling (`384` dims: `[att_pool, 2nd_half - 1st_half, spatial_temporal_max]`).
    - **Lower-Body Geometric Temporal Summary Branch (`lower_mlp: 24 -> 32` dims):** Summarizes 6 per-frame lower-body biomechanical signals (`cos(knee_angle_L)`, `cos(knee_angle_R)`, `hip_to_knee_drop`, `knee_to_ankle_drop`, `hip_to_ankle_height`, `foot_stance_width`) across the 16-frame window via `[mean, std, late_half - early_half, mean_abs_diff]`.

---

## 1. Controlled Validation Ablation Study on Dataset V3b (`830` Pure Windows / `143` Clips)

All 7 architectural variants (`v3.0-clean` baseline through `v3.1-D3`) were trained under identical conditions (`SEED=42`, `EPOCHS=60`, `BATCH_SIZE=64`, `LR=1.5e-3`) and evaluated strictly on the held-out **Validation set (`830` windows / `143` clips)**:

| Variant | Parameters | Loss Type | Val Safety Score | Val Fall Recall | Val Fall Prec | Val Macro-F1 | Val Acc | Best Epoch |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `ST-GCN v3.0-clean` (Baseline `v2.4` Arch) | `279,251` | Unweighted | `0.8034` | `93.9%` | `64.8%` | `0.5692` | `57.1%` | 12 |
| `ST-GCN v3.0-clean` (Baseline `v2.4` Arch) | `279,251` | Weighted | `0.7760` | `91.2%` | `49.5%` | `0.6373` | `63.1%` | 39 |
| `ST-GCN v3.1-A` (+ Gravity-Aware Features) | `280,999` | Unweighted | `0.7015` | `77.2%` | `55.3%` | `0.6175` | `60.5%` | 19 |
| `ST-GCN v3.1-A` (+ Gravity-Aware Features) | `280,999` | Weighted | `0.7438` | `86.0%` | `50.0%` | `0.6235` | `62.5%` | 19 |
| `ST-GCN v3.1-B` (+ Wrist-to-Head Features) | `280,999` | Unweighted | `0.7875` | `86.8%` | `71.2%` | `0.6304` | `62.8%` | 41 |
| `ST-GCN v3.1-B` (+ Wrist-to-Head Features) | `280,999` | Weighted | `0.7657` | `85.1%` | `63.8%` | `0.6316` | `65.1%` | 42 |
| `ST-GCN v3.1-C` (+ Lower-Body Kinematics) | `280,999` | Unweighted | `0.7607` | `92.1%` | `46.5%` | `0.5831` | `58.7%` | 8 |
| **`ST-GCN v3.1-C` (+ Lower-Body Kinematics) ★** | **`280,999`** | **Weighted** | **`0.8252`** | **`91.2%`** | **`73.2%`** | **`0.6636`** | **`65.7%`** | **19** |
| `ST-GCN v3.1-D1` (+ Gravity + Wrist-to-Head) | `282,747` | Unweighted | `0.7120` | `74.6%` | `73.9%` | `0.6144` | `62.2%` | 42 |
| `ST-GCN v3.1-D1` (+ Gravity + Wrist-to-Head) | `282,747` | Weighted | `0.7002` | `79.8%` | `45.3%` | `0.6264` | `63.0%` | 31 |
| `ST-GCN v3.1-D2` (+ Gravity + Lower-Body) | `282,747` | Unweighted | `0.7321` | `85.1%` | `48.5%` | `0.6087` | `61.0%` | 31 |
| `ST-GCN v3.1-D2` (+ Gravity + Lower-Body) | `282,747` | Weighted | `0.7110` | `79.8%` | `52.9%` | `0.6110` | `62.8%` | 21 |
| `ST-GCN v3.1-D3` (+ Full A+B+C Fusion) | `284,495` | Unweighted | `0.7082` | `74.6%` | `71.4%` | `0.6121` | `60.4%` | 59 |
| `ST-GCN v3.1-D3` (+ Full A+B+C Fusion) | `284,495` | Weighted | `0.6810` | `73.7%` | `52.2%` | `0.6393` | `64.0%` | 59 |

---

## 2. Held-Out Test Evaluation (`ST-GCN v3.1-C` vs. `ST-GCN v3.0-clean` vs. `ST-GCN v2.4`)

### 2.1 Progression Across Test Subsets
| Test Subset | Scope | Metric | `ST-GCN v2.4` (V1 train) | `ST-GCN v3.0-clean` (V3b train) | **`ST-GCN v3.1-C` (V3b train)** | Delta (`v3.1-C` vs `v3.0-clean`) |
|---|---|---|---:|---:|---:|---:|
| **`ALL_TEST (V1 + V2)`** | `147` independent clips | **Video-Level 8-Class Accuracy** | — | `63.27%` | **`72.79%`** | **`+9.52 pp` 🚀** |
| | `147` independent clips | **Video-Level Macro-F1** | — | `0.6315` | **`0.7254`** | **`+0.0939` 🚀** |
| | `147` independent clips | **Video-Level `Fall` F1** | — | `0.7391` | **`0.7805`** | **`+0.0414`** |
| | `906` windows | **Window 8-Class Accuracy** | `53.31%` | `60.49%` | **`61.81%`** | **`+1.32 pp`** |
| | `906` windows | **Window Macro-F1 / Weighted-F1** | `0.4891` / — | `0.6032` / `0.5994` | **`0.6268` / `0.6256`** | **`+0.0236` / `+0.0262`** |
| | `906` windows | **Window `Fall` Precision / F1** | — / `0.6232` | `73.17%` / `0.7469` | **`80.95%` / `0.7623`** | **`+7.78 pp` / `+0.0154`** |
| | `22` Fall / `125` Non-Fall | **10 FPS Fall Clip Recall (Rule 1)** | — | `18/22` (`81.82%`) | **`19/22` (`86.36%`)** | **`+4.54 pp` (+1 Fall clip)** |
| | `22` Fall / `125` Non-Fall | **10 FPS RED Trigger Recall (Rule 2)** | — | **`18/22` (`81.82%`)** | `16/22` (`72.73%`) | `-9.09 pp` (higher Prec) |
| | `22` Fall / `125` Non-Fall | **10 FPS RED FPR** | — | `5/125` (`4.00%`) | **`4/125` (`3.20%`)** | **`-0.80 pp`** |
| **`V1_LEGACY_TEST`** | `506` windows / `83` clips | **Window 8-Class Accuracy** | `72.13%` | **`76.09%`** | **`76.09%`** | Tied (`+3.96 pp` vs `v2.4`) |
| | `506` windows / `83` clips | **Window Weighted-F1** | `0.7145` | `0.7436` | **`0.7671`** | **`+0.0235`** |
| | `506` windows / `83` clips | **Window `Fall` F1** | `0.7158` | `0.8163` | **`0.8202`** | **`+0.0039` (`+0.1044` vs `v2.4`)** |
| | `13` Fall / `70` Non-Fall | **10 FPS Fall Clip Recall (Rule 1)** | `11/13` (`84.62%`) | **`13/13` (`100.00%`)** | **`13/13` (`100.00%`)** | **`100.00%`** |
| | `13` Fall / `70` Non-Fall | **10 FPS RED Trigger Recall (Rule 2)** | `11/13` (`84.62%`) | **`13/13` (`100.00%`)** | `11/13` (`84.62%`) | `4/70` (`5.71%`) RED FPR |
| **`V2_GROUP_HELDOUT_TEST`** | `400` windows / `64` clips | **Window 8-Class Accuracy** | `29.50%` | `40.75%` | **`43.75%`** | **`+3.00 pp` (`+14.25 pp` vs `v2.4`)** |
| | `400` windows / `64` clips | **Window Macro-F1** | `0.2959` | `0.4624` | **`0.5068`** | **`+0.0444` (`+0.2109` vs `v2.4`)** |
| | `400` windows / `64` clips | **Window `Fall` F1** | `0.4186` | `0.4444` | **`0.5333`** | **`+0.0889` (`Prec = 85.71%`)** |
| | `9` Fall / `55` Non-Fall | **10 FPS Fall Clip Recall (Rule 1)** | `2/9` (`22.22%`) | `5/9` (`55.56%`) | **`6/9` (`66.67%`)** | **`+11.11 pp`** |
| | `9` Fall / `55` Non-Fall | **10 FPS RED Trigger Recall (Rule 2)** | `2/9` (`22.22%`) | **`5/9` (`55.56%`)** | **`5/9` (`55.56%`)** | Tied |
| | `9` Fall / `55` Non-Fall | **10 FPS RED FPR** | — | `1/55` (`1.82%`) | **`0/55` (`0.00%`)** | **`0.00%` false alarms!** |

### 2.2 Per-Class Video-Level (`147` Clips) & Window-Level (`906` Windows) Comparison (`v3.0-clean` vs. `v3.1-C`)
| Class | Video F1 (`v3.0-clean`) | **Video F1 (`v3.1-C`)** | Video Delta | Window F1 (`v3.0-clean`) | **Window F1 (`v3.1-C`)** | `V1_LEGACY` F1 (`v3.1-C`) | `V2_GROUP_HD` F1 (`v3.1-C`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`Walk`** | `0.9231` | **`0.9231`** | `0.0000` | `0.9031` | `0.8920` | **`0.9417`** | **`0.7934`** |
| **`Eat`** | `0.4375` | **`0.6207`** | **`+0.1832`** | `0.6198` | **`0.6691`** | `0.6811` | **`0.6444`** |
| **`Drink`** | `0.3158` | **`0.4878`** | **`+0.1720`** | `0.3099` | **`0.3666`** | **`0.6165`** | `0.1798` |
| **`Fall`** | `0.7391` | **`0.7805`** | **`+0.0414`** | `0.7469` | **`0.7623`** | **`0.8202`** | **`0.5333`** |
| **`Lying`** | `0.4500` | **`0.5455`** | **`+0.0955`** | `0.3701` | **`0.4080`** | **`0.7625`** | `0.0000` |
| **`Sit_Down`** | `0.6667` | **`0.8000`** | **`+0.1333`** | `0.5743` | `0.5070` | `0.4000` | `0.5490` |
| **`Stand_Up`** | `0.6452` | **`0.7368`** | **`+0.0916`** | `0.4800` | **`0.5984`** | `0.4118` | **`0.6667`** |
| **`Crouching`** | `0.8750` | **`0.9091`** | **`+0.0341`** | `0.8219` | `0.8108` | **`0.9048`** | `0.6875` |
| **Overall / Macro** | **Acc `63.27%` / F1 `0.6315`** | **Acc `72.79%` / F1 `0.7254`** | **`+9.52 pp` / `+0.0939`** | **Acc `60.49%` / F1 `0.6032`** | **Acc `61.81%` / F1 `0.6268`** | **Acc `76.09%` / F1 `0.6923`** | **Acc `43.75%` / F1 `0.5068`** |

### 2.3 8×8 Window-Level Confusion Matrix (`ST-GCN v3.1-C`, `ALL_TEST` `n=906`, rows=true, cols=pred)
Order: `[Walk, Eat, Drink, Fall, Lying, Sit_Down, Stand_Up, Crouching]`
```text
[[161   5   1   0   6  11   1   0]
 [  2  92  28   0   1   5   6   0]
 [  0  38  57   2   3   1   4   0]
 [  1   0   4  85   6  19   3   0]
 [  3   0 111  16  61   5   2   2]
 [  3   1   0   1  12  36   2   0]
 [  6   5   5   0  10   3  38   0]
 [  0   0   0   1   0   7   4  30]]
```


---

## 3. One-Shot Final Evaluation on `SEALED_GROUP_TEST` (`v3.1-E`, `89` Clips / `645` Windows)

### 3.1 Frozen Pre-Evaluation Audit Manifest
- **Git HEAD Commit:** `26e6d6f45c0beee044d96c45690063576d620f02`
- **Primary Frozen Checkpoint:** `models/action_stgcn.pth` == `models/action_stgcn_v3_1.pth` (`ST-GCN v3.1-C [Weighted]`, `280,999` params)
  - `SHA-256 [models/action_stgcn.pth]`: `4c821d0920c342c3fa832ebff53a75556098eef12d95870648b2e1f84cb59afe`
  - `SHA-256 [models/action_stgcn_v3_0_clean.pth]`: `54fa5b2c463e2290e075ef0942c7a74beefffeaa3c2470f7304651f3d97f1aff`
  - `SHA-256 [models/action_lstm.pth]`: `c92065708f13d4aa275bf24a6445e257ddbb14e107f3d45017057e37d6b2e47e`
  - `SHA-256 [src/utils/config.py]`: `308ba91d7bd008b212d9d2774e37e5db45ab2ae0d058ff48c99c5583efae9929`
  - `SHA-256 [src/utils/skeleton_utils.py]`: `b386aa4ce728feabca12acf0918efd23d09be618f0c09f23b8b8336cdfab56ce`
  - `SHA-256 [src/pipeline/action_classifier.py]`: `53e58bb8d71f2fd271eaa636d4d5072192e0d52fa63dfe7d2e41065a5a6742e3`
  - `SHA-256 [src/pipeline/alert_state.py]`: `abe843d9728487771a22fde8ac96b604fbfb12c40522bfef77a096d5d346b106`
  - `SHA-256 [data/skeleton_dataset/sealed/X_sealed.npy]`: `38da6c0a72a62f26cae76443b82d99fccffe8fc8cbaffb056a6d95020b59ada0`
- **Frozen Runtime Thresholds:** `CONF_THRESHOLD = 0.50`, `WARNING_THRESHOLD = 0.50`, `CRITICAL_THRESHOLD = 0.75`, `INSTANT_RED_THRESHOLD = 0.90`, `N_WARN_FRAMES = 3`, `N_CRITICAL_FRAMES = 3`, `N_RECOVERY_FRAMES = 5`
- **Group Disjointness Verification:** `SEALED_GROUP_TEST` (`89` clips / `645` windows across 3 held-out groups: `avi:P002` [`30` clips], `avi:P045` [`6` clips], `mp4:S0` [`53` clips]) verified 100% disjoint from Train, Validation, and Test at both the clip level and the performer/recording-group level.

### 3.2 Overall Model Progression on `SEALED_GROUP_TEST` (`89` Clips: `13` Fall, `76` Non-Fall)
| Metric | `LSTM Baseline` (`action_lstm.pth`) | `ST-GCN v3.0-clean` (`action_stgcn_v3_0_clean.pth`) | **`ST-GCN v3.1-C [Primary Frozen]`** (`action_stgcn.pth`) | Video Bootstrap 95% CI (`v3.1-C`, 2,000 resamples) |
|---|---:|---:|---:|---|
| **Video-Level 8-Class Accuracy (`n=89`)** | `32.58%` | `67.42%` | **`70.79%` (`+3.37 pp` vs `v3.0`)** | `[60.67%, 80.90%]` |
| **Video-Level Macro-F1 / Weighted-F1** | `0.3548` / `0.3617` | **`0.6623` / `0.7071`** | `0.5994` / `0.6744` | `[0.5153, 0.6565]` (Macro-F1) |
| **Window-Level 8-Class Accuracy (`n=645`)** | `28.99%` | `56.74%` | **`57.21%` (`+0.47 pp` vs `v3.0`)** | — |
| **Window-Level Macro-F1 / Weighted-F1** | `0.2833` / `0.3265` | `0.5428` / `0.5780` | `0.5194` / `0.5736` | — |
| **10.0 FPS Fall Clip Recall (`Rule 1`, `TP/13`)** | `8/13` (`61.54%`) | `9/13` (`69.23%`) | **`11/13` (`84.62%`, `+15.39 pp`)** | `[61.54%, 100.00%]` |
| **10.0 FPS RED Trigger Recall (`Rule 2`, `TP/13`)** | `5/13` (`38.46%`) | `8/13` (`61.54%`) | **`9/13` (`69.23%`, `+7.69 pp`)** | `[46.15%, 92.31%]` (`9` Instant, `0` 3-Tick, delay `0.067s`) |
| **10.0 FPS RED FPR (`Rule 2`, `FP/76`)** | `16/76` (`21.05%`) | **`7/76` (`9.21%`)** | `9/76` (`11.84%`) | `[5.26%, 19.74%]` |
| **8.33 FPS Fall Clip Recall (`Rule 1`, `TP/13`)** | `7/13` (`53.85%`) | **`10/13` (`76.92%`)** | `9/13` (`69.23%`) | — |
| **8.33 FPS RED Trigger Recall (`Rule 2`, `TP/13`)** | `6/13` (`46.15%`) | `8/13` (`61.54%`) | **`9/13` (`69.23%`, `+7.69 pp`)** | `9` Instant, `0` 3-Tick, delay `0.033s` |
| **8.33 FPS RED FPR (`Rule 2`, `FP/76`)** | `17/76` (`22.37%`) | **`7/76` (`9.21%`)** | `8/76` (`10.53%`) | — |

### 3.3 Per-Class Window-Level (`n=645`) & Video-Level (`n=89`) Metrics (`ST-GCN v3.1-C`)
| Class | Sealed Clips | Sealed Windows | Window Prec | Window Rec | Window F1 | Video Prec | Video Rec | **Video F1** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Walk`** | 11 | 126 | `0.7667` | `0.7302` | `0.7480` | `0.8333` | `0.9091` | **`0.8696`** |
| **`Eat`** | 6 | 42 | `0.8333` | `0.9524` | **`0.8889`** | `0.8571` | `1.0000` | **`0.9231`** |
| **`Drink`** | 6 | 18 | `0.9375` | `0.8333` | **`0.8824`** | `1.0000` | `0.8333` | **`0.9091`** |
| **`Fall`** | 13 | 73 | `0.3158` | `0.4932` | `0.3850` | `0.5000` | `0.6154` | **`0.5517`** |
| **`Lying`** | 5 | 53 | `0.0357` | `0.0377` | `0.0367` | `0.0000` | `0.0000` | `0.0000` |
| **`Sit_Down`** | 19 | 125 | `0.5843` | `0.8320` | `0.6865` | `0.6552` | `1.0000` | **`0.7917`** |
| **`Stand_Up`** | 23 | 202 | `0.7921` | `0.3960` | `0.5281` | `0.8824` | `0.6522` | **`0.7500`** |
| **`Crouching`** | 6 | 6 | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| **Total / Macro** | **89** | **645** | — | — | **Acc `57.21%` / F1 `0.5194`** | — | — | **Acc `70.79%` / F1 `0.5994`** |

### 3.4 8×8 Confusion Matrices on `SEALED_GROUP_TEST` (`ST-GCN v3.1-C`, rows=true, cols=pred)
Order: `[Walk, Eat, Drink, Fall, Lying, Sit_Down, Stand_Up, Crouching]`
```text
Window-Level Confusion Matrix (n=645):
[[ 92   2   0   1  15  15   1   0]
 [  0  40   1   1   0   0   0   0]
 [  0   3  15   0   0   0   0   0]
 [  7   0   0  36   8   8  11   3]
 [  0   0   0  51   2   0   0   0]
 [  0   2   0   2   6 104   9   2]
 [ 21   1   0  23  25  45  80   7]
 [  0   0   0   0   0   6   0   0]]

Video-Level Confusion Matrix (n=89):
[[10  0  0  0  0  1  0  0]
 [ 0  6  0  0  0  0  0  0]
 [ 0  1  5  0  0  0  0  0]
 [ 1  0  0  8  0  1  2  1]
 [ 0  0  0  5  0  0  0  0]
 [ 0  0  0  0  0 19  0  0]
 [ 1  0  0  3  2  2 15  0]
 [ 0  0  0  0  0  6  0  0]]
```

### 3.5 Per-Group Breakdown & Scope Limitations on `SEALED_GROUP_TEST`
| Held-Out Group | Clips | Windows | Present Classes | Window Acc (Macro-F1 sup) | **Video Acc (Macro-F1 sup)** | Window `Fall` F1 | **10 FPS RED Recall** | **10 FPS RED FPR** |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **`avi:P002`** | 30 | 108 | `Eat(6), Drink(6), Fall(6), Sit_Down(6), Stand_Up(6)` | `91.67%` (`0.9280`) | **`93.33%` (`0.9483`)** | **`0.8485`** (`18`w) | **`6/6` (`100.0%`)** | **`0/24` (`0.0%`)** |
| **`avi:P045`** | 6 | 6 | `Crouching(6)` (1 window/clip) | `0.00%` (`0.0000`) | `0.00%` (`0.0000`) | N/A (`0` Fall) | N/A (`0/0`) | **`0/6` (`0.0%`)** |
| **`mp4:S0`** | 53 | 531 | `Walk(11), Fall(7), Lying(5), Sit_Down(13), Stand_Up(17)` | `50.85%` (`0.4431`) | **`66.04%` (`0.5425`)** | `0.2857` (`55`w) | `3/7` (`42.9%`) | `9/46` (`19.6%`) |

**Methodological Scope Note:**
- Because `SEALED_GROUP_TEST` contains only 3 held-out groups (`avi:P002`, `avi:P045`, `mp4:S0`), the 645 windows are clustered within 89 clips across those 3 groups rather than 645 independent observations.
- On unseen performer **`avi:P002`** (`30` clips across 5 action classes), `ST-GCN v3.1-C` generalizes with **`93.33%` Video-Level Accuracy (`28/30` clips)**, **`6/6 (100.0%)` 10-FPS RED Fall Recall**, and **`0/24 (0.0%)` RED False Positive Rate**.
- Conversely, all `6` `Crouching` clips come from a single performer (`avi:P045`, 1 window each, all predicted as `Sit_Down` with `0/6` false RED alarms), and all `5` `Lying` clips come from `mp4:S0` (`51/53` windows predicted as `Fall`, accounting for 5 of the 9 false RED triggers on `mp4:S0` where only 2 `v2` `Lying` clips were available in `Train`). These two small-support classes (`6` and `5` clips) explain the gap between unweighted 8-class Video Macro-F1 (`0.5994`) and overall Video Accuracy (`70.79%`).

---

## 4. Post-Freeze Inference Robustness Layer (Phase 5.7+ / Phase 6)

Without retraining or altering the frozen `ST-GCN v3.1-C` weights (`models/action_stgcn.pth`), the runtime inference pipeline adds four modular robustness stages verified on both held-out datasets and real-world videos:

1. **Invalid `(0,0)` Cleanup & Virtual Hip Fallback (`conf=0.25`):** Undetected joints are zeroed locally before hip-centering normalization; upper-body-only shots synthesize an anatomical virtual hip anchor (`conf=0.25`) rather than dropping frames.
2. **Physical Stabilizer & Hybrid LSTM Fallback (`src/pipeline/stabilizer.py`):** Separates `Low_Pose_Quality`, activates `40% ST-GCN + 60% LSTM` blending under lower-body occlusion (`ank_vis < 0.35` & `kn_vis < 0.50`), preserves sudden `Fall >= 0.70` spikes across EMA (`Fast-Attack Fall Passthrough`), and gates stationary poses to `Idle`.
3. **Stage-2B Conditional Hand-Object Context (`src/pipeline/object_context.py`):** Runs `models/yolo_obj_pretrained.pt` strictly on the upper-body hand/head ROI when `Fall < 0.40` and `Eat`/`Drink` is a candidate (`>= 0.20` or wrist-to-nose `< 0.80`), eliminating false `Drink` detections on non-drinking real-world clips (`eat.mp4`, `sit_down.mp4`, `walk_2.mp4`) with zero `Fall` interference.
4. **Stage-1 `0.30` Hysteresis Tracking (`src/pipeline/detector.py::SubjectPoseTracker`):** Continues occluded/horizontal subjects down to `TRACK_CONF_THRESHOLD = 0.30` (IoU-prioritized Path A + unanchored compact horizontal Path B `>= 0.35`, keypoints kept `>= 0.30` strictly separated from the `0.25` virtual-hip sentinel). End-to-end raw-video verification across all `89` `SEALED_GROUP_TEST` clips increased `Fall RED Recall` from **`9/13 (69.2%)` → `11/13 (84.6%)`** with **`0` added `RED FPR` (`9/76`, `11.84%`)**, and recovered **`26/63`** playable clips (up from `11/63`) and **`12/40`** `Fall RED` triggers (up from `5/40`) on previously excluded hard/occluded raw videos.

---

## 5. ST-GCN v4 Dual-Specialist Upgrade (Dataset V4 Temporal Cleanup + In-the-Wild Augmentation)

### 5.1 Root Cause & Dataset V4 Curation (`scripts/build_dataset_v4.py`)
Human review of all `189/189` `Fall` clips via `scripts/fall_annotator_app.py` (`data/skeleton_dataset/fall_timestamps.csv`: `647` pre-fall upright `Idle` ticks, `2,823` active collapse ticks [mean `1.49s`], and `3,562` post-fall floor `Lying` ticks) revealed that ~606 training `Fall` windows actually occurred *before* the drop or *after* landing on the floor. Dataset **V4** (`5,197` train windows; packaged in `data/skeleton_dataset_v4_kaggle.zip`) applies:
- **Temporal Fall Windowing:** Drops `112` pre-fall and `494` post-fall windows from `Fall` training labels and harvests `86` clean post-fall motionless floor windows into `Lying`.
- **Static Lying Pruning & Curation:** Drops `115` upright transition windows mislabeled as `Lying`.
- **In-the-Wild Geometric & Occlusion Augmentation:** Expands underrepresented classes (`Fall`, `Lying`, `Sit_Down`, `Stand_Up`, `Crouching`, `Eat`, `Drink`) with lower-body occlusion simulation, torso-compression/camera-tilt simulation, mirror flips, and spatial jitter.

### 5.2 Dual-Specialist Architecture & Runtime Fusion (`src/pipeline/action_classifier.py`)
`ST-GCN v4 Dual-Specialist` pairs two complementary `280,999`-param specialists trained on Dataset V4 (`scripts/train_stgcn_v4.py`):
- **`ST-GCN v4-C` (`models/action_stgcn.pth` / `action_stgcn_v4_c.pth`):** Lower-Body Kinematics branch (strong on `Walk`, `Eat`, `Fall`, `Stand_Up`).
- **`ST-GCN v4-B` (`models/action_stgcn_v4_b.pth`):** Wrist-to-Head Feature branch (strong on `Drink`, `Lying`, `Sit_Down`, `Crouching`).
- **Per-Class Weighted Fusion + Max Fall Safety Gate:** When `stgcn_wrist` is loaded, `predict_probs()` blends with `_DUAL_SPEC_W_C = [0.50, 0.70, 0.30, 0.60, 0.25, 0.45, 0.60, 0.40]` (`[Walk, Eat, Drink, Fall, Lying, Sit_Down, Stand_Up, Crouching]`), enforces `p[Fall] = max(p_fused[Fall], p_C[Fall], p_B[Fall])`, and renormalizes.

### 5.3 Benchmark Comparison (`ALL_TEST` `147c / 906w` & `SEALED_GROUP_TEST` `89c / 645w`)
| Split / Metric | `v3.1-C` Baseline | `v4-C` (Lower-Body) | `v4-B` (Wrist-Head) | **`v4 Dual-Specialist` (Default)** | Delta (`v4 Dual` vs `v3.1-C`) |
|---|---:|---:|---:|---:|---:|
| `ALL_TEST` **Video Accuracy (`147c`)** | `72.79%` | `75.51%` | `77.55%` | **`81.63%`** | **`+8.84 pp` 🚀** |
| `ALL_TEST` **Video Macro-F1** | `0.7254` | `0.7518` | `0.7655` | **`0.8062`** | **`+0.0808` 🚀** |
| `ALL_TEST` **Window Accuracy (`906w`)** | `61.81%` | `64.24%` | `72.85%` | **`74.06%`** | **`+12.25 pp` 🚀** |
| `ALL_TEST` **10 FPS RED Recall / FPR** | `16/22` / `4/125 (3.20%)` | `17/22` / `2/125 (1.60%)` | `16/22` / `2/125 (1.60%)` | **`16/22` / `1/125 (0.80%)`** | **4× lower RED FPR** |
| `SEALED` **Video Accuracy (`89c`)** | `70.79%` | `76.40%` | `77.53%` | **`77.53%`** | **`+6.74 pp` 🚀** |
| `SEALED` **Video Macro-F1** | `0.5994` | `0.7138` | `0.7613` | **`0.7495`** | **`+0.1501` 🚀** |
| `SEALED` **Window Accuracy (`645w`)** | `57.21%` | `65.12%` | `70.70%` | **`70.39%`** | **`+13.18 pp` 🚀** |
| `SEALED` **Rule-1 Fall Recall (`13c`)** | `11/13` (`84.6%`) | `11/13` (`84.6%`) | `11/13` (`84.6%`) | **`11/13` (`84.6%`)** | Maintained |
| `SEALED` **RED FPR (`76c`)** | `9/76` (`11.84%`) | `2/76` (`2.63%`) | `0/76` (`0.00%`) | **`0/76` (`0.00%`)** | **Zero false RED alarms!** |
| `SEALED` **Video `Lying` F1 / `Fall` F1** | `0.0000` / `0.5517` | `0.2222` / `0.6207` | `0.6154` / `0.6957` | **`0.6667` / `0.6429`** | **`+0.6667` / `+0.0912`** |

### 5.4 External Unseen Web-Video Check (`7` Real-World `.mp4` Clips)
- `fall (1).mp4` (`114` ticks): `RED` triggered cleanly during the active collapse (`29` `Fall` ticks) and transitions cleanly to `Lying` (`46` ticks) once motionless on the floor.
- `lying.mp4` (`177` ticks): `YELLOW` (`162` `Lying` ticks, `0` false `RED`).
- `eat.mp4` (`254` ticks): `GREEN` (`72` `Eat` ticks vs `64` in `v3.1-C`, `0` `Sit_Down`, `0` `Drink`).
- `sit_down.mp4` (`309` ticks): `GREEN` (`27` `Sit_Down` ticks, `0` `Fall`, `0` `Lying`, `0` `Drink`, fixing the `1` false `Fall` tick seen in `v3.1-C`).
- `walk.mp4` (`175` ticks) & `drink.mp4` (`177` ticks): `GREEN` (`122` `Walk` ticks; `37` `Drink` ticks).
- **Runtime Seated-vs-Lying Guard (`src/pipeline/stabilizer.py`):** Complements V4 with a late-window torso-uprightness check (`late_sh_y_mean > 0.68`, `late_sin_theta < 0.38`, `late_yx_ratio > 1.20`) so seated actions (`Eat` / `Drink` / `Sit_Down`) do not emit spurious `Lying` ticks.
