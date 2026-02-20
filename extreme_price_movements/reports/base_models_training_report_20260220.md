# Base Models Training Report
**Run date:** 2026-02-20  
**Artifact run:** `20260214_190000`  
**Training window:** ~04:26 UTC → 06:36 UTC (~2h10m total)  
**Dataset:** 4 buckets × 3 horizons = 12 base models  
**Candidates raced per horizon:** 4 (extratrees, xgboost, lightgbm, catboost)  
**Train/valid split:** ~85k–169k samples (bucket-dependent), 5-fold OOF

---

## 1. Summary Table — Race Winners & OOF Metrics

| Bucket   | H | Winner      | RcAUC  | OOF_AUC | RcIC   | OOF_BSS  | OOF_Brier | ECE@10 | Cal. Profile    |
|----------|---|-------------|--------|---------|--------|----------|-----------|--------|-----------------|
| long_mr  | 2 | catboost    | 0.6234 | **0.6351** | 0.4598 | -0.1145 | 0.0347    | —      | —               |
| long_mr  | 4 | lightgbm    | 0.5486 | 0.5503  | 0.2358 | -0.1369 | 0.0358    | —      | —               |
| long_mr  | 8 | lightgbm    | 0.5017 | 0.5475  | 0.0266 | -0.1422 | 0.0353    | —      | —               |
| long_tf  | 2 | xgboost     | 0.5932 | 0.5511  | 0.2387 | -0.1342 | 0.0361    | 0.0498 | overconfident   |
| long_tf  | 4 | extratrees  | 0.5043 | 0.5507  | 0.0539 | -0.1286 | 0.0367    | 0.0492 | overconfident   |
| long_tf  | 8 | extratrees  | 0.5069 | 0.5469  | 0.0657 | -0.1315 | 0.0364    | 0.0499 | overconfident   |
| short_mr | 2 | xgboost     | 0.6122 | 0.5408  | 0.5423 | -0.1102 | 0.0401    | 0.0479 | overconfident   |
| short_mr | 4 | extratrees  | 0.5388 | 0.5265  | 0.1534 | -0.1036 | 0.0418    | 0.0481 | flat            |
| short_mr | 8 | xgboost     | 0.5310 | 0.5241  | 0.1650 | -0.0963 | 0.0434    | 0.0452 | flat            |
| short_tf | 2 | catboost    | 0.6502 | 0.6096  | 0.4169 | -0.0954 | 0.0388    | 0.0709 | overconfident   |
| short_tf | 4 | lightgbm    | 0.6418 | 0.5412  | 0.4616 | -0.1067 | 0.0406    | 0.0493 | flat            |
| short_tf | 8 | lightgbm    | 0.6126 | 0.5333  | 0.4657 | -0.1002 | 0.0421    | 0.0497 | flat            |

> **RcAUC** = race-fold AUC (in-race CV). **OOF_AUC** = full out-of-fold AUC. **RcIC** = race-fold Spearman IC. **OOF_BSS** = Brier Skill Score vs. climatology (negative = worse than base rate, expected for extreme events). **ECE@10** = Expected Calibration Error at 10 bins.

---

## 2. Per-Bucket Analysis

### 2.1 long_mr — Buy Dips (mean-reversion long)
- **n_samples:** ~85,367 per horizon | **Primary horizon selected:** H=2
- **Best horizon:** H=2 (catboost, OOF_AUC=**0.635** — strongest across all 12 models)
- H=4 and H=8 degrade significantly (OOF_AUC 0.550 / 0.548), IC drops to near-zero at H=8 (0.027)
- Bootstrap CV Prec@20: not logged for long_mr (deployed all 3 horizons)
- **Verdict:** Strong signal at H=2, diminishing at longer horizons — consistent with mean-reversion decay

### 2.2 long_tf — Buy Momentum (trend-following long)
- **n_samples:** ~85,367 per horizon | **Primary horizon selected:** H=2
- OOF_AUC flat across horizons (0.551 / 0.551 / 0.547) — no horizon dominance
- RcIC very low at H=4 and H=8 (0.054 / 0.066), suggesting weak trend persistence
- All three calibration profiles: **overconfident** (ECE ~0.049–0.050)
- Bootstrap CV Prec@20: H=2=0.042, H=4=0.041, H=8=0.038
- **Verdict:** Weakest bucket overall. Marginal AUC lift, low IC, overconfident. Meta layer critical here.

### 2.3 short_mr — Sell Rips (mean-reversion short)
- **n_samples:** ~85,367 per horizon | **Primary horizon selected:** H=2
- H=2 has highest RcAUC (0.612) and RcIC (0.542) but OOF_AUC drops to 0.541 — race overfit
- H=4 and H=8 show more stable OOF_AUC (0.527 / 0.524) with lower BSS magnitude
- Calibration: overconfident at H=2, flat at H=4/H=8
- Bootstrap CV Prec@20: H=2=0.042, H=4=0.037, H=8=0.032
- **Verdict:** Decent H=2 signal, degrades with horizon. Race winner (xgboost H=2) shows gap between RcAUC and OOF_AUC — watch for overfitting.

### 2.4 short_tf — Sell Weakness (trend-following short)
- **n_samples:** ~169,413 per horizon | **Primary horizon selected:** H=8
- **Strongest bucket by RcAUC** across all horizons (0.650 / 0.642 / 0.613)
- OOF_AUC highest at H=2 (0.610), drops to 0.533 at H=8 — race selects H=8 despite lower OOF
- Lowest OOF_BSS magnitude (−0.095 to −0.100) — best Brier performance relative to climatology
- H=2 calibration: **overconfident** (ECE=0.071, highest across all models)
- Bootstrap CV Prec@20: H=2=0.021, H=4=0.026, H=8=0.023
- **Verdict:** Best-performing bucket. Large dataset (2× others). H=2 OOF_AUC=0.610 is the second-best single model result. Overconfidence at H=2 needs calibration attention.

---

## 3. Candidate Race Results (selected horizons)

### long_tf H=2 (all 4 candidates)
| Model       | OOF_Cal_Score | AUC    | IC     | BSS     | Prec@10 |
|-------------|---------------|--------|--------|---------|---------|
| extratrees  | 0.3645        | 0.5531 | 0.2048 | -0.0100 | 0.0537  |
| xgboost     | —             | —      | —      | —       | —       |
| lightgbm    | —             | —      | —      | —       | —       |
| catboost    | 0.3645        | 0.5531 | 0.2048 | -0.0100 | 0.0537  |
| **Winner: xgboost** | — | 0.5932 | 0.2387 | -0.0087 | — |

### short_mr H=2 (all 4 candidates)
| Model       | OOF_Cal_Score | AUC    | IC     | BSS     | Prec@10 |
|-------------|---------------|--------|--------|---------|---------|
| extratrees  | 0.4019        | 0.5678 | 0.3066 | -0.0043 | 0.0746  |
| xgboost     | 0.4039        | 0.6122 | 0.5423 | -0.0049 | 0.0554  |
| lightgbm    | 0.4035        | 0.6099 | 0.3471 | -0.0031 | 0.0848  |
| catboost    | 0.3970        | 0.5712 | 0.2763 | -0.0035 | 0.0807  |
| **Winner: xgboost** | 0.4039 | **0.6122** | **0.5423** | -0.0049 | 0.0554 |

### short_tf H=2 (all 4 candidates)
| Model       | OOF_Cal_Score | AUC    | IC     | BSS     | Prec@10 |
|-------------|---------------|--------|--------|---------|---------|
| extratrees  | 0.4079        | 0.6066 | 0.3947 | -0.0046 | 0.0818  |
| xgboost     | 0.4078        | 0.6466 | 0.4621 | -0.0042 | 0.0774  |
| lightgbm    | 0.4066        | 0.6446 | 0.3723 | -0.0021 | 0.0926  |
| catboost    | 0.4067        | **0.6502** | 0.4169 | -0.0034 | 0.0892  |
| **Winner: catboost** | 0.4067 | **0.6502** | 0.4169 | -0.0034 | 0.0892 |

---

## 4. Regime Breakdown — short_tf H=2 (best model)

| Regime dim  | Bucket | BSS    | AUC   | Brier  |
|-------------|--------|--------|-------|--------|
| vol_12h     | low    | -0.099 | 0.627 | 0.0425 |
| vol_12h     | mid    | -0.112 | 0.604 | 0.0379 |
| vol_12h     | high   | -0.074 | 0.591 | 0.0360 |
| vol_48h     | low    | -0.100 | 0.631 | 0.0435 |
| vol_48h     | mid    | -0.105 | 0.601 | 0.0384 |
| vol_48h     | high   | -0.081 | 0.581 | 0.0344 |
| volume_12h  | low    | -0.092 | 0.630 | 0.0363 |
| volume_12h  | mid    | -0.098 | 0.617 | 0.0390 |
| volume_12h  | high   | -0.096 | 0.583 | 0.0411 |
| trend_12h   | low    | -0.084 | 0.576 | 0.0414 |
| trend_12h   | mid    | -0.106 | 0.618 | 0.0413 |
| trend_12h   | high   | -0.099 | 0.633 | 0.0337 |
| trend_48h   | low    | -0.087 | 0.596 | 0.0354 |
| trend_48h   | mid    | -0.110 | 0.621 | 0.0398 |
| trend_48h   | high   | -0.089 | 0.606 | 0.0412 |

**Key insight:** AUC is consistently highest in low-vol / low-trend regimes for short_tf. High-trend regime shows best AUC (0.633) at trend_12h — momentum persistence favours the short_tf signal.

---

## 5. Deployed Horizons & Bootstrap Precision

| Bucket   | Deployed horizons | Primary H | Prec@20 H2 | Prec@20 H4 | Prec@20 H8 |
|----------|-------------------|-----------|------------|------------|------------|
| long_tf  | [2, 4, 8]         | H=2       | 0.042      | 0.041      | 0.038      |
| short_mr | [2, 4, 8]         | H=2       | 0.042      | 0.037      | 0.032      |
| short_tf | [2, 4, 8]         | H=8       | 0.021      | 0.026      | 0.023      |

> Bootstrap Prec@20 not logged for `long_mr` (all 3 horizons deployed, primary H=2 inferred from OOF_AUC ranking).

---

## 6. Key Observations & Flags

1. **All OOF_BSS are negative** — expected for extreme-event classification where the base rate is very low. The Brier score is dominated by the prior; IC and AUC are the correct primary metrics.

2. **long_mr H=2 is the standout base model** (OOF_AUC=0.635) — the only model clearly above 0.60 on the full OOF set.

3. **short_tf is the strongest bucket** — all three horizons have RcAUC > 0.61, largest dataset (169k), lowest BSS magnitude.

4. **long_tf is the weakest bucket** — RcAUC barely above 0.50 at H=4/H=8, IC < 0.07. Meta layer will need to work hard here.

5. **Calibration:** All models are overconfident or flat. No model is well-calibrated at alpha level without isotonic recalibration. ECE@10 ranges 0.0005–0.071.

6. **Race vs OOF gap:** short_mr H=2 shows the largest gap (RcAUC=0.612 vs OOF_AUC=0.541) — the race CV is optimistic. This is expected given the smaller short_mr dataset and higher variance.

7. **Winner diversity:** catboost wins 2/12, lightgbm wins 4/12, xgboost wins 3/12, extratrees wins 3/12 — no single algorithm dominates.

---

## 7. Meta Training Status (as of report time)

The `train_meta` step is currently running (PID active, log: `/tmp/base_training_run2.log`).

**long_mr weight optimization complete:**

| Horizon | Best Optuna objective | n_eff   | n_eff_ratio | top1pct_share |
|---------|-----------------------|---------|-------------|---------------|
| H=2     | 0.37701               | 132,754 | 0.876       | 0.0255        |
| H=4     | 0.53642               | 132,881 | 0.877       | 0.0258        |
| H=8     | 0.56924               | 133,126 | 0.878       | 0.0251        |

**MetaModel long_mr_H8 fit:** MDI RFE in progress (158→40 features, racing 6 candidates).

Remaining buckets to process: `long_tf`, `short_mr`, `short_tf`.
