# Comprehensive Training Report — 2026-02-10

## Run Configuration
- **Signal timestamp**: 2026-02-04 22:00 UTC
- **Universe**: 485 symbols (Top 500 by vol, variance-filtered)
- **Data**: 27,088 hourly bars, 1,261 days (2022-08-23 → 2026-02-04), 61% non-NaN
- **Features per bucket**: 48 (MDI-selected from ~360–600 candidates)
- **Meta features**: 42 raw + 5 pred_logit interactions + 6 regime bucket interactions → 10 selected

---

## 1. Alpha Models (Classification) — Selected Horizons

The pipeline trains 3 horizons per bucket and selects the best. Below are all 12 alpha models with the **selected horizon bolded**.

### long_mr (Mean Reversion, Long)

| H | Winner | Rw-AUC | OOF AUC | OOF BSS | OOF Brier | ECE@10 | OOF-Ret Corr | RcIC |
|---|--------|--------|---------|---------|-----------|--------|-------------|------|
| 2 | xgboost | 0.5238 | 0.5594 | 0.0138 | 0.2379 | 0.033 | +0.073 | 0.0001 |
| **4** | **extratrees** | **0.5494** | **0.5740** | **0.0201** | **0.2328** | **0.009** | **+0.076** | **0.073** |
| 8 | extratrees | 0.5495 | 0.5640 | 0.0180 | 0.2411 | 0.007 | +0.097 | 0.064 |

### long_tf (Trend Follow, Long)

| H | Winner | Rw-AUC | OOF AUC | OOF BSS | OOF Brier | ECE@10 | OOF-Ret Corr | RcIC |
|---|--------|--------|---------|---------|-----------|--------|-------------|------|
| **2** | **catboost** | **0.6176** | **0.6161** | **0.0446** | **0.2194** | **0.006** | **+0.116** | **0.159** |
| 4 | extratrees | 0.5567 | 0.5623 | 0.0180 | 0.2216 | 0.009 | +0.111 | 0.187 |
| 8 | catboost | 0.5578 | 0.5707 | 0.0181 | 0.2299 | 0.004 | +0.130 | 0.180 |

### short_mr (Mean Reversion, Short)

| H | Winner | Rw-AUC | OOF AUC | OOF BSS | OOF Brier | ECE@10 | OOF-Ret Corr | RcIC |
|---|--------|--------|---------|---------|-----------|--------|-------------|------|
| 2 | catboost | 0.5685 | 0.5920 | 0.0290 | 0.2214 | 0.005 | +0.109 | 0.136 |
| 4 | extratrees | 0.5514 | 0.5746 | 0.0197 | 0.2228 | 0.008 | +0.091 | 0.177 |
| **8** | **extratrees** | **0.5512** | **0.5672** | **0.0193** | **0.2301** | **0.017** | **+0.129** | **0.231** |

### short_tf (Trend Follow, Short)

| H | Winner | Rw-AUC | OOF AUC | OOF BSS | OOF Brier | ECE@10 | OOF-Ret Corr | RcIC |
|---|--------|--------|---------|---------|-----------|--------|-------------|------|
| 2 | xgboost | 0.5377 | 0.5386 | 0.0087 | 0.2395 | 0.022 | +0.009 | 0.102 |
| 4 | extratrees | 0.5544 | 0.5804 | 0.0270 | 0.2316 | 0.028 | +0.076 | 0.175 |
| **8** | **lightgbm** | **0.6077** | **0.6257** | **0.0542** | **0.2331** | **0.037** | **+0.175** | **0.219** |

---

## 2. Alpha Models — Selected Horizon Summary vs Targets

| Metric | Target | long_mr (H=4) | long_tf (H=2) | short_mr (H=8) | short_tf (H=8) |
|--------|--------|---------------|---------------|-----------------|-----------------|
| **Rw-AUC** | ≥ 0.55 | 0.549 ⚠️ | **0.618** ✅ | 0.551 ✅ | **0.608** ✅ |
| **OOF BSS** | ≥ 0 (min), ≥ 0.02 (good) | 0.020 ✅ | **0.045** ✅ | 0.019 ⚠️ | **0.054** ✅ |
| **ECE@10** | ≤ 0.10 | **0.009** ✅ | **0.006** ✅ | **0.017** ✅ | **0.037** ✅ |
| **OOF-Ret Corr** | > 0, stable | +0.076 ✅ | +0.116 ✅ | +0.129 ✅ | +0.175 ✅ |

### Verdict — Alpha Models
- **long_tf** and **short_tf**: Strong. Both exceed all targets comfortably. AUC > 0.60, BSS > 0.04.
- **long_mr**: Borderline. Rw-AUC = 0.549 (just under 0.55 threshold). BSS = 0.020 (meets "good" threshold).
- **short_mr**: Borderline. Rw-AUC = 0.551 (barely passes). BSS = 0.019 (just under "good").
- **All ECE@10 ≤ 0.037**: Excellent calibration across all buckets.
- **All OOF-Ret Corr positive**: No sign flips. Strongest for short_tf (+0.175).

---

## 3. Meta Models — Full Metrics

| Metric | Target | long_mr | long_tf | short_mr | short_tf |
|--------|--------|---------|---------|----------|----------|
| **Winner** | — | Ridge α=5.0 | Ridge α=10.0 | Ridge α=0.01 | ExtraTrees |
| **IC (Spearman)** | ≥ 0.05 (good), ≥ 0.08 (strong) | **0.215** ✅✅ | **0.152** ✅✅ | 0.028 ❌ | **0.092** ✅✅ |
| **R²** | > 0 (must), 1–5% (sweet spot) | -345 ❌ | -182 ❌ | -145 ❌ | -169 ❌ |
| **GtP@10** | ≥ 1.1 (min), ≥ 1.3 (good) | **1.337** ✅ | **1.383** ✅ | 0.994 ❌ | **1.615** ✅✅ |
| **Sharpe@10** | > 0 | **+0.089** ✅ | **+0.089** ✅ | -0.002 ❌ | **+0.152** ✅ |
| **WinRate@10** | (secondary) | 34.0% | 31.8% | 28.4% | 39.7% |
| **Base WinRate** | — | 33.8% | 30.6% | 28.4% | 38.0% |
| **AvgRet Top10** | > 0 bps | **+20.1 bps** ✅ | **+28.0 bps** ✅ | -0.7 bps ❌ | **+42.4 bps** ✅ |
| **AvgRet Bot10** | — | +13.7 bps | +23.5 bps | +7.1 bps | +32.9 bps |
| **Top10-Bot10 Spread** | > 0 | **+6.3 bps** ✅ | **+4.5 bps** ✅ | -7.7 bps ❌ | **+9.6 bps** ✅ |
| **CVaR@10** | — | -120.2 bps | -173.0 bps | -226.8 bps | -154.7 bps |
| **Trades/Day@10** | — | 2.36 | 2.31 | 2.15 | 2.24 |
| **n_timestamps** | — | 814 | 1,017 | 893 | 775 |

### Verdict — Meta Models
- **long_mr**: **Strong**. IC = 0.215 (best), GtP = 1.34, positive spread +6.3 bps.
- **long_tf**: **Solid**. IC = 0.152, GtP = 1.38, spread +4.5 bps. Biggest AvgRet.
- **short_mr**: **Failed**. IC = 0.028 (below 0.05 minimum), negative spread, GtP < 1.0. Meta adds no value — should be disabled or rebuilt.
- **short_tf**: **Best overall**. IC = 0.092, GtP = 1.62 (highest), spread +9.6 bps, Sharpe = 0.152.

### Note on R²
R² is deeply negative for all meta models. This is **expected and not a problem**: the meta predicts rank percentiles [0, 1] while raw returns are in [-0.05, +0.05]. R² = 1 - SS_res/SS_tot compares prediction scale to return scale — they're incommensurable. **IC and GtP are the correct metrics for ranking models.** R² would only be meaningful if the meta directly predicted returns in the same units.

---

## 4. Combined Scorecard

| Bucket | Alpha Rw-AUC | Alpha BSS | Alpha ECE | Meta IC | Meta GtP | Meta Spread | Overall |
|--------|-------------|-----------|-----------|---------|----------|-------------|---------|
| **long_mr** | 0.549 ⚠️ | 0.020 ✅ | 0.009 ✅ | 0.215 ✅✅ | 1.34 ✅ | +6.3 bps ✅ | **B+** |
| **long_tf** | 0.618 ✅ | 0.045 ✅ | 0.006 ✅ | 0.152 ✅✅ | 1.38 ✅ | +4.5 bps ✅ | **A** |
| **short_mr** | 0.551 ✅ | 0.019 ⚠️ | 0.017 ✅ | 0.028 ❌ | 0.99 ❌ | -7.7 bps ❌ | **D** |
| **short_tf** | 0.608 ✅ | 0.054 ✅ | 0.037 ✅ | 0.092 ✅✅ | 1.62 ✅✅ | +9.6 bps ✅ | **A** |

---

## 5. Recommendations

### Immediate (High Priority)

**R1. Disable or bypass short_mr meta model.**
The meta IC = 0.028 with negative spread means it's actively hurting selection. Options:
- (a) Fall back to raw alpha score for short_mr (bypass meta entirely)
- (b) Retrain with different features/labeling (see R5)

**R2. Fix long_mr alpha AUC.**
Rw-AUC = 0.549 is just under the 0.55 threshold. Possible fixes:
- Increase training data: long_mr H=4 only has 1,753 samples. Consider relaxing candidate selection criteria or using multi-horizon pooling.
- Try different candidate selection: `bottom_ret` may not be the best proxy for mean-reversion long opportunities.
- Add more features targeting MR-specific signals (e.g., RSI divergence, funding rate, order flow imbalance).

### Medium Priority

**R3. Add downside risk features for long_tf meta.**
long_tf has the biggest spread (+4.5 bps) but 30% win rate — high convexity. Adding features like:
- Gap/vol shock indicators
- Liquidity/spread proxies
- Realized skewness (6h/24h)
- Tail risk indicators (CVaR of recent returns)
...would help the meta learn when convex payoffs are likely to fail.

**R4. Explore quantile regression for meta objective.**
Current Ridge/ET minimize MSE on rank_pct. A quantile loss (τ=0.8 or 0.9) would directly optimize for the upper tail — aligning with the actual trading objective of selecting the best 10-30% of trades.

**R5. Rebuild short_mr meta with different approach.**
The short_mr alpha model is decent (IC=0.231, AUC=0.567) but the meta fails. Possible causes:
- Only 1,121 meta training samples (smallest bucket) — insufficient for learning
- MR short trades may have fundamentally different return dynamics that rank_pct doesn't capture well
- Try: (a) larger sample pool by relaxing n_res filter, (b) different meta features (e.g., funding rate, open interest, liquidation levels), (c) skip meta entirely and use alpha score directly

**R6. Improve short_mr alpha BSS.**
BSS = 0.019 is just under the "good" threshold of 0.02. The model is well-calibrated (ECE=0.017) but could benefit from:
- Better label definition for short MR (current `top_ret` candidate selection may be noisy)
- Temporal weighting to emphasize recent regime

### Lower Priority

**R7. Cross-validate meta across time periods.**
Current meta OOF uses PurgedKFold (3 splits). Consider walk-forward validation to check temporal stability of IC and GtP.

**R8. Add turnover/capacity constraints.**
Current AvgTrades/Day@10 ≈ 2.2-2.4. If this exceeds execution capacity, the meta should incorporate a turnover penalty or position persistence feature.

**R9. Experiment with meta ensemble.**
For buckets where Ridge and ET are close (e.g., long_tf: Ridge=-0.264 vs ET=-0.204), a simple average of both could be more robust than winner-take-all.

---

## 6. Selected Meta Features (for reference)

| Bucket | Features |
|--------|----------|
| **long_mr** | atr_pct, qv, rv_ratio_6_24, ret6h, evr_6, rv_24h, rv_6h, vol_z_30_calm + 2 more |
| **long_tf** | evr_6, pred_x_trend_pct, atr_pct, delta_stall_6, ret1h, pred_x_vol_z, pred_logit, trend_pct + 2 more |
| **short_mr** | atr_pct, vol_z, meta_abs_net_x_breakout, donch_dist_12, grind_score, accept, ft_2, evr_6 + 2 more |
| **short_tf** | atr_pct, vol_z_30_calm, clv_mean_4, excess_6h, ft_2, spike_score, delta_stall_6, vol_z + 2 more |

Notable: `atr_pct` appears in all 4 metas — volatility scaling is universally important for gating.
