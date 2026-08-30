# Layer 2 & 3 Outcomes Analysis (2026-01-23)

## Executive Summary
A review of the latest outcome reports (`20260123`) reveals a robust signal generation pipeline with a 100% gate pass rate. However, a critical divergence exists between "Survival" status and "Causal Quality," where many surviving models are flagged as having "WEAK" or "FAILED" causal quality. Despite this, top-performing families like `INVENTORY_SPECIALIST` and `VOLATILITY_SPECIALIST` are demonstrating strong predictive power.

## 1. Gate Health & Survival
*   **100% Pass Rate:** The `layer2_gate_diagnostics_20260123_004612.md` report indicates that all geometry families passed the initial gating criteria.
*   **Implication:** This suggests either highly effective signal generation logic or potentially lenient gating thresholds (`min_p`, `pos_rate`) that may need tightening to filter noise earlier.

## 2. Model Performance Highlights
Based on `layer2_model_race_20260122_235434.csv`:
*   **Top Performers:**
    *   **INVENTORY_SPECIALIST:** Achieved a Layer 2 Score of **0.404** and an IC of **0.229**, indicating strong predictive capability.
    *   **VOLATILITY_SPECIALIST:** Scored **0.384** with high directional consistency (59%).
*   **High Potential / High Variance:**
    *   **CAUSAL_SURPRISE:** This family shows a bimodal performance distribution. While some candidates have low scores, others achieved exceptional ICs (up to **0.46** in raw logs), suggesting that specific parameter combinations (e.g., `return_surprise_z_100`) are extremely valuable.

## 3. The Causal Quality Gap
A significant finding is the disconnect between the two validation stages:
*   **Observation:** Many models have `survival_status: PASSED` but `causal_quality_status: FAILED` or `WEAK`.
*   **Example:** `FLOW_PRESSURE_CONTINUOUS_FLOW_ACCEL_20_POS` passed survival but failed causal quality.

### Investigation of Discrepancy
The discrepancy arises from the fundamentally different objectives and strictness levels of the two filtering layers:

| Feature | **Survival Gate** (Layer 2 Step 1) | **Causal Quality Check** (Layer 2 Step 3) |
| :--- | :--- | :--- |
| **Code Location** | `orthogonal_label_generation.py` | `causal_quality_assessment.py` |
| **Primary Objective** | **Technical Validity**: Ensure dataset is sufficient for training. | **Economic Validity**: Ensure signal provides unique, stable alpha. |
| **Correlation Check** | **Weak**: F-Stat p-value < 0.30. Allows very noisy signals. | **Strong**: IC > 0.05 & OOS R² > 0.0. Requires real predictive power. |
| **Uniqueness Check** | None (Jaccard warning only). | **Strict**: CI Score > 0.015. Requires unique info beyond the Backbone. |
| **Robustness** | None. | **Strict**: Probabilistic Sharpe Ratio (PSR) > 0.4. |
| **Sample Quality** | Minimal Rate (>0.1/day) & Balance (>0.05). | Effective N (autocorrelation adjusted). |
| **Outcome** | `survival_status` | `causal_quality_status` |

**Conclusion:** The system is functioning as designed. The **Survival Gate** is a low-pass filter that admits any technically valid dataset (preventing crashes), while the **Causal Quality Check** is a high-pass filter that demands high-quality, unique alpha. The high failure rate in the second stage indicates that while we are generating many *valid* datasets, many are either **redundant** (fails CI Score) or **non-predictive** (fails IC/PSR).

**Recommendation:** To save compute resources, we could tighten the Survival Gate by introducing a lightweight orthogonality check or raising the F-Stat threshold, closer to the Causal Quality standards.

## 4. Feature Intelligence
*   **Meta-Features:** The system is successfully evaluating higher-order features. Families like `META_REINFORCED_COMPOSITE` and `META_EWMA_1D` are appearing in the logs with competitive metrics.
*   **Sparse Covariance:** Feature importance logs confirm that models are heavily relying on `sparse_covariance_*` features (e.g., `volatility_trend_slope`, `momentum_decay`), validating the investment in these complex feature sets.

## 5. Conclusion
The pipeline is healthy and generating a high volume of potential signals. The immediate focus should be on reconciling the "Survival" vs. "Causal Quality" criteria to ensure that resources are focused on the highest-quality candidates (like the high-IC `CAUSAL_SURPRISE` variants) rather than processing a large volume of "weak" signals.
