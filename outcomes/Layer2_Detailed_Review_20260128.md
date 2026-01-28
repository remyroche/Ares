# Layer 2 Detailed Review (2026-01-28)

## 1. Executive Summary
The Layer 2 pipeline shows a dichotomy between excellent geometry gate health and significant robustness challenges in the model race. While the underlying feature geometries are stable (99.84% pass rate), the predictive models exhibit a pattern of high raw performance scores coupled with "WEAK" or "FAILED" survival statuses, indicating issues with path stability and regime generalization.

## 2. Gate Diagnostics
*   **Health**: Excellent.
*   **Pass Rate**: 99.84% (1846 total rows, only 3 failures).
*   **Failures**:
    *   `MULTI_HORIZON_SLOPE_SLOPE_DIVERGENCE_H12_H96_NEG`: 2 failures (F-STAT).
    *   `CROSS_ASSET_SURPRISE_CA__ECT_HALF_LIFE`: 1 failure (F-STAT).
*   **Implication**: The feature generation layer is robust and producing valid geometries. The downstream issues are likely in the learning or evaluation phase, not the data preparation phase.

## 3. Model Race Analysis

### Top Performers (by Score)
Despite high scores, these models are flagged as `WEAK`, suggesting they fit the data well but fail stability criteria (likely `Path_Stability` or `Interventional_Contrast` variance).

*   **`DISPERSION_SPECIALIST`**: Score **0.428**. Status: `WEAK`.
    *   Strongest raw performer. High overlap ratio (0.80) and reasonable sparsity (10.0).
*   **`VOLATILITY_INNOVATION_SPECIALIST`**: Score **0.417**. Status: `WEAK`.
    *   High raw IC (0.14) but likely penalized for path instability or regime dependence.
*   **`VOLATILITY_SPECIALIST`**: Score **0.408**. Status: `WEAK`.
    *   Solid baseline, but again, fails robustness checks.
*   **`LIQUIDITY_SPECIALIST`**: Score **0.406**. Status: `WEAK`.

### Critical Failures
*   **`SURPRISE_Z_CONTINUOUS_*` Family**: Score **0.0**. Status: `FAILED`.
    *   This entire family is failing completely. Given their `ic` in raw metrics is non-zero (often ~0.2-0.4), the 0.0 score suggests a hard disqualification, possibly due to `DSR` (Downside Risk) or `SPA_p` (Superior Predictive Ability p-value) failures, or strict stability thresholds.
*   **`CAUSAL_SURPRISE`**: Score **0.298**. Status: `FAILED`.
    *   Metrics show decent IC (0.16) but very high `stability` variance (23.2) and `path_stability_var` (17.6) in the raw logs. This variance likely triggered the failure.

## 4. Stability & Robustness
The recurring theme is high performance but low trust (WEAK/FAILED).
*   **Path Stability**: The `path_score` in raw metrics is consistently negative (e.g., -1.11), and `path_stability_var` is high (often >17, sometimes >60). This indicates that model performance is highly sensitive to the specific training fold or time path, a hallmark of overfitting to specific regimes.
*   **Regime Dispersion**: Models like `CAUSAL_SURPRISE` show high max regime lift (up to 5.9) vs min regime lift, suggesting they work exceptionally well in some market states and poorly in others.

## 5. Recommendations
1.  **Investigate `SURPRISE_Z` Disqualification**: Determine the exact hard constraint causing the 0.0 score. It is likely `Path_Stability` or `DSR`.
2.  **Regularize for Stability**: The "WEAK" specialists need stronger regularization (e.g., higher dropout, deeper trees with more pruning, or stricter feature selection) to improve Path Stability.
3.  **Regime Hardening**: Since regime dispersion is high, consider training on regime-balanced subsets or increasing the penalty for regime variance in the objective function.
