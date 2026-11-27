# Action Model Comparison Report: `LSTM Baseline` vs. `ST-GCN v2.4` vs. `ST-GCN v3.0-clean` vs. `ST-GCN v3.1-C`

All models are evaluated across the three strictly separated test scopes of **Dataset V3b**:
1. **`V1_LEGACY_TEST` (Legacy Holdout Test):** `506` pure windows from the `83` original Phase 3.1 Test clips (`13` Fall, `70` Non-Fall).
2. **`V2_GROUP_HELDOUT_TEST` (Group-Heldout HD Test):** `400` pure windows from `64` HD clips (`9` Fall, `55` Non-Fall) belonging to performer/recording groups (`avi:P008`, `mp4:S1`) completely disjoint from Train and Val.
3. **`ALL_TEST` (`V1 + V2` Combined):** `906` pure windows from `147` independent clips (`22` Fall, `125` Non-Fall).

---

## 1. Four-Way Head-to-Head Summary Across Test Subsets

| Test Subset | Metric | **`LSTM Baseline`** (`action_lstm.pth`, V1) | **`ST-GCN v2.4`** (`action_stgcn_v2_4.pth`, V1) | **`ST-GCN v3.0-clean`** (`action_stgcn_v3_0_clean.pth`, V3b) | **`ST-GCN v3.1-C`** (`action_stgcn.pth`, V3b) | Winner |
|---|---|---:|---:|---:|---:|---|
| **Architecture** | Trainable Parameters | `582,664` | `279,251` | **`279,251`** | `280,999` | **ST-GCN (`2.1×` smaller)** |
| | Inference Latency | **`1.402 ms`** (CPU) | `6.303 ms` (CPU) | `3.618 ms` (GPU) | `4.750 ms` (GPU) | **All well `<15 ms`** |
| **`ALL_TEST`** (`906`w / `147`c) | **Video-Level 8-Class Accuracy** | — | — | `63.27%` | **`72.79%`** | **ST-GCN v3.1-C (`+9.52 pp`)** |
| | **Video-Level Macro-F1** | — | — | `0.6315` | **`0.7254`** | **ST-GCN v3.1-C (`+0.0939`)** |
| | **Video-Level `Fall` F1** | — | — | `0.7391` | **`0.7805`** | **ST-GCN v3.1-C (`+0.0414`)** |
| | **Window 8-Class Accuracy** | `46.03%` | `53.31%` | `60.49%` | **`61.81%`** | **ST-GCN v3.1-C (`+1.32 pp`)** |
| | **Window Macro-F1** | `0.4358` | `0.4891` | `0.6032` | **`0.6268`** | **ST-GCN v3.1-C (`+0.0236`)** |
| | **Window `Fall` Precision / F1** | — / `0.5424` | — / `0.6232` | `73.17%` / `0.7469` | **`80.95%` / `0.7623`** | **ST-GCN v3.1-C (`+0.0154`)** |
| | **10 FPS Fall Clip Recall (Rule 1)** | — | — | `18/22` (`81.82%`) | **`19/22` (`86.36%`)** | **ST-GCN v3.1-C (`19/22`)** |
| | **10 FPS RED Trigger Recall (Rule 2)** | — | — | **`18/22` (`81.82%`)** | `16/22` (`72.73%`) | **ST-GCN v3.0-clean** |
| | **10 FPS RED FPR (`FP/125`)** | — | — | `5/125` (`4.00%`) | **`4/125` (`3.20%`)** | **ST-GCN v3.1-C (`3.20%`)** |
| **`V1_LEGACY_TEST`** (`506`w / `83`c) | **Window 8-Class Accuracy** | `66.60%` | `72.13%` | **`76.09%`** | **`76.09%`** | **Tie (`v3.0-clean` & `v3.1-C`)** |
| | **Window Weighted-F1** | `0.6592` | `0.7145` | `0.7436` | **`0.7671`** | **ST-GCN v3.1-C (`+0.0235`)** |
| | **Window Macro-F1** | `0.6491` | `0.7015` | **`0.7182`** | `0.6923` | **ST-GCN v3.0-clean** |
| | **Window `Fall` F1** | `0.7753` | `0.7158` | `0.8163` | **`0.8202`** | **ST-GCN v3.1-C (`0.8202`)** |
| | **10 FPS Fall Clip Recall (Rule 1)** | `11/13` (`84.62%`) | `11/13` (`84.62%`) | **`13/13` (`100.00%`)** | **`13/13` (`100.00%`)** | **Tie (`100.00%`)** |
| | **10 FPS RED Trigger Recall (Rule 2)** | `10/13` (`76.92%`) | `11/13` (`84.62%`) | **`13/13` (`100.00%`)** | `11/13` (`84.62%`) | **ST-GCN v3.0-clean (`100%`)** |
| | **10 FPS RED FPR (`FP/70`)** | **`3/70` (`4.29%`)** | `7/70` (`10.00%`) | `4/70` (`5.71%`) | `4/70` (`5.71%`) | **LSTM (`3/70`) / v3 (`4/70`)** |
| **`V2_GROUP_HELDOUT_TEST`** (`400`w / `64`c) | **Window 8-Class Accuracy** | `20.00%` | `29.50%` | `40.75%` | **`43.75%`** | **ST-GCN v3.1-C (`+3.00 pp`)** |
| | **Window Macro-F1** | `0.2262` | `0.2959` | `0.4624` | **`0.5068`** | **ST-GCN v3.1-C (`+0.0444`)** |
| | **Window `Fall` F1** | `0.1880` | `0.4186` | `0.4444` | **`0.5333`** | **ST-GCN v3.1-C (`+0.0889`)** |
| | **10 FPS Fall Clip Recall (Rule 1)** | — | `2/9` (`22.22%`) | `5/9` (`55.56%`) | **`6/9` (`66.67%`)** | **ST-GCN v3.1-C (`66.67%`)** |
| | **10 FPS RED Trigger Recall (Rule 2)** | — | `2/9` (`22.22%`) | **`5/9` (`55.56%`)** | **`5/9` (`55.56%`)** | **Tie (`55.56%`)** |
| | **10 FPS RED FPR (`FP/55`)** | — | — | `1/55` (`1.82%`) | **`0/55` (`0.00%`)** | **ST-GCN v3.1-C (`0.00%`)** |


---

## 2. One-Shot Final Evaluation on `SEALED_GROUP_TEST` (`v3.1-E`: `89` Clips / `645` Windows)

Evaluated **once** after freezing **`ST-GCN v3.1-C [Weighted]`** (`models/action_stgcn.pth`, SHA-256 `4c821d09...`) and thresholds (`CRITICAL_THRESHOLD = 0.75`, `INSTANT_RED_THRESHOLD = 0.90`, `N_CRITICAL_FRAMES = 3`, `N_RECOVERY_FRAMES = 5`):

| Scope / Metric | **`LSTM Baseline`** (`action_lstm.pth`) | **`ST-GCN v3.0-clean`** (`action_stgcn_v3_0_clean.pth`) | **`ST-GCN v3.1-C [Primary Frozen]`** (`action_stgcn.pth`) | Winner |
|---|---:|---:|---:|---|
| **Video-Level 8-Class Accuracy (`89` clips)** | `32.58%` | `67.42%` | **`70.79%`** (`95% CI [60.7%, 80.9%]`) | **ST-GCN v3.1-C (`+3.37 pp`)** |
| **Window-Level 8-Class Accuracy (`645` windows)** | `28.99%` | `56.74%` | **`57.21%`** | **ST-GCN v3.1-C (`+0.47 pp`)** |
| **10.0 FPS Fall Clip Recall (`Rule 1`, `TP/13`)** | `8/13` (`61.54%`) | `9/13` (`69.23%`) | **`11/13` (`84.62%`)** (`95% CI [61.5%, 100%]`) | **ST-GCN v3.1-C (`+15.39 pp`)** |
| **10.0 FPS RED Trigger Recall (`Rule 2`, `TP/13`)** | `5/13` (`38.46%`) | `8/13` (`61.54%`) | **`9/13` (`69.23%`)** (`95% CI [46.2%, 92.3%]`) | **ST-GCN v3.1-C (`+7.69 pp`)** |
| **10.0 FPS RED FPR (`Rule 2`, `FP/76`)** | `16/76` (`21.05%`) | **`7/76` (`9.21%`)** | `9/76` (`11.84%`) (`0/30` on `avi:P002+P045`) | **ST-GCN v3.0-clean (`7/76`)** |
| **8.33 FPS RED Trigger Recall (`Rule 2`, `TP/13`)** | `6/13` (`46.15%`) | `8/13` (`61.54%`) | **`9/13` (`69.23%`)** | **ST-GCN v3.1-C (`+7.69 pp`)** |
| **8.33 FPS RED FPR (`Rule 2`, `FP/76`)** | `17/76` (`22.37%`) | **`7/76` (`9.21%`)** | `8/76` (`10.53%`) | **ST-GCN v3.0-clean (`7/76`)** |
| **Group `avi:P002` (`30` clips, 5 classes)** | — | — | **Video Acc `93.33%` (`28/30`), RED Recall `6/6` (`100%`), RED FPR `0/24` (`0%`)** | **ST-GCN v3.1-C** |
| **Group `mp4:S0` (`53` clips, 5 classes)** | — | — | **Video Acc `66.04%` (`35/53`), RED Recall `3/7` (`42.9%`), RED FPR `9/46` (`19.6%`)** | **ST-GCN v3.1-C** |
| **Group `avi:P045` (`6` clips, `Crouching`)** | — | — | **Video Acc `0.00%` (`6/6 -> Sit_Down`), RED FPR `0/6` (`0%`)** | **Zero false RED alarms** |
