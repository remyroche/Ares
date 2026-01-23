# Layer 2 Execution Review (2026-01-23)

## 1. Executive Summary
The latest Layer 2 execution (timestamp: 20260123_004612) demonstrates a highly selective but potentially biased performance profile. The pipeline achieved a 100% pass rate across all Geometry Gates, which warrants investigation into gate strictness. The model race was dominated by the `CAUSAL_SURPRISE` family, which exhibited exceptionally high Information Coefficient (IC) scores, suggesting strong signal capture or potential leakage.

## 2. Gate Diagnostics
*   **Report:** `outcomes/layer2_gate_diagnostics_20260123_004612.md`
*   **Pass Rate:** 100.00% across all tested families.
*   **Failures:** 0 failures recorded.
*   **Observation:** A 100% pass rate is statistically unusual for a robust filtering stage. It implies that either the candidates are universally high-quality (which aligns with the high metrics observed) or the gate thresholds (AUC/Stability) are set too loosely for the current market regime.

## 3. Model Performance Analysis
*   **Source:** `outcomes/layer2_raw_metric_log_20260123_004612.json`
*   **Total Models Evaluated:** 774

### Top Performing Families (by IC)
The `CAUSAL_SURPRISE` family occupies the entire top echelon of performance, with IC values approaching 0.48. This is significantly higher than typical financial signal performance (usually < 0.10).

| Rank | Family | IC | Lift | Stability |
|:---|:---|:---|:---|:---|
| 1 | CAUSAL_SURPRISE | 0.4804 | 0.4804 | 23.19 |
| 2 | CAUSAL_SURPRISE | 0.4804 | 0.4804 | 23.19 |
| 3 | CAUSAL_SURPRISE | 0.4804 | 0.4804 | 23.19 |
| 4 | CAUSAL_SURPRISE | 0.4705 | 0.4705 | 13.53 |
| 5 | CAUSAL_SURPRISE | 0.4705 | 0.4705 | 13.53 |

**Note:** The identical top scores suggest duplicate model configurations or identical signal convergence.

### Average Performance by Family (Top 10)
Meta-learning ensembles are performing well, validating the composite signal approach.

1.  **META_SUM_4H_COMPOSITE_SURPRISE...**: 0.1717
2.  **CAUSAL_SURPRISE**: 0.1715
3.  **LIQUIDITY_SPECIALIST**: 0.1505
4.  **META_REINFORCED_COMPOSITE_DERIVED_FLOW...**: 0.1448
5.  **META_REINFORCED_COMPOSITE_FLOW_PRESSURE...**: 0.1380

### Family Distribution
The pipeline is heavily weighted towards `CAUSAL_SURPRISE`, comprising 35% of the total evaluated models.

1.  **CAUSAL_SURPRISE**: 270 models
2.  **MOMENTUM_DECAY_SPECIALIST**: 75 models
3.  **VOLUME_SPECIALIST**: 45 models
4.  **VOLATILITY_SPECIALIST**: 30 models
5.  **LIQUIDITY_SPECIALIST**: 30 models

## 4. Root Cause Analysis (Added 2026-01-23)

### 4.1. Leakage in `CAUSAL_SURPRISE`
The suspiciously high ICs (0.48) have been traced to look-ahead bias in the `CausalSurpriseDetector`.
*   **Mechanism:** The `_get_global_events` method in `LabelBasedLayer2` passes the full `market_data` dataframe (including future columns) to `generate_causal_events`.
*   **Impact:** Inside `CausalSurpriseDetector`, this dataframe is used to calculate `fwd_returns` (forward returns), which are then used to optimize specialist weights (`regime_specific_reliability`) *during feature generation*.
*   **Result:** The model is effectively "told" which specialists will perform best in the future, artificially inflating performance metrics.

### 4.2. Loose Geometry Gates
The 100% pass rate is attributed to the default configuration of the gate.
*   **Mechanism:** `layer2_gate_min_score` defaults to `0.0`.
*   **Impact:** While a percentile threshold exists, the fallback to 0.0 means that if the percentile logic is bypassed (e.g., due to low candidate count triggering "min candidates" logic), or if the distribution is weak, almost any model can pass.
*   **Result:** Weak candidates are not being filtered out effectively.

## 5. Recommendations & Actions
1.  **Fix Leakage:** Modify `LabelBasedLayer2` to stop passing `market_data` to the surprise detector, forcing it to use static or backward-looking weights.
2.  **Strengthen Gates:** Increase `layer2_gate_min_score` to a non-zero value (e.g., 0.02) to enforce a baseline quality standard.
3.  **Diversity Check:** While `CAUSAL_SURPRISE` is dominant, ensure that the downstream ensemble (Layer 3) maintains access to orthogonal signals (e.g., `MOMENTUM`, `LIQUIDITY`) to prevent mode collapse.
