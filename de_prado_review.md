# De Prado Alignment Review: `meta_labeling_hpo_sample_weighted.py`

## Overview
This document reviews the alignment of the HPO module with methodologies described in Marcos Lopez de Prado's *Advances in Financial Machine Learning (AFML)* and *Machine Learning for Asset Managers (MLAM)*.

## Alignment Matrix

| Concept | De Prado Methodology | Implementation Status | Alignment Score | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Triple Barrier Method** | Dynamic barriers (volatility-based), vertical (time) barrier, two horizontal barriers (TP/SL). <br>*(AFML Ch 3)* | **Implemented** <br> via `multi_label_voting_utils` and `feature_generation_meta_labeling_step`. Uses `vol_scaled` returns and explicit TP/SL multipliers. | ⭐⭐⭐⭐⭐ (Strong) | Code explicitly implements "Triple Barrier" logic, handling time-outs and dynamic volatility scaling correctly. |
| **Meta-Labeling** | Binary classification secondary model to filter primary signals. Target is "1" if primary signal makes money, "0" otherwise. <br>*(AFML Ch 3.6)* | **Implemented** <br> The entire module is designed to accept `primary_signals`, compute realized outcomes, and train a secondary model (`learner`) to predict success. | ⭐⭐⭐⭐⭐ (Strong) | Core architectural philosophy matches De Prado exactly. |
| **Sample Uniqueness** | Weighting samples by $1/c_t$ (inverse concurrency) to handle overlapping labels. <br>*(AFML Ch 4)* | **Implemented** <br> `generate_weights_per_label.compute_uniqueness` calculates concurrency timelines and average uniqueness per event. | ⭐⭐⭐⭐⭐ (Strong) | `generate_weights_per_label.py` contains a faithful implementation of the concurrency derivation logic. |
| **Purged Cross-Validation** | Removing training samples that overlap with test samples (Purging) and adding a gap after test set (Embargo) to prevent leakage. <br>*(AFML Ch 7)* | **Approximated?** <br> Code uses `TimeSeriesSplit` (Line 700). While expanding window reduces leakage compared to K-Fold, it **does not appear to strictly purge** the `max_lookahead` period before the test fold starts. | ⭐⭐ (Weak) | **Risk:** Overlap leakage. If `max_lookahead` > 0, the last training samples may share outcome paths with the first test samples. Standard `TimeSeriesSplit` does not handle this "Purge" gap automatically. |
| **Feature Importance** | MDA (Mean Decrease Accuracy) over MDI (Impurity). Handling substitution effects. <br>*(AFML Ch 8)* | **Implemented** <br> Imports `run_mda_shap_feature_selection`. Uses `permutation_importance` (MDA) alongside SHAP. | ⭐⭐⭐⭐⭐ (Strong) | Moving away from default MDI is a key De Prado recommendation that is followed here. |
| **Bet Sizing** | Sizing probability $m = 2p - 1$ or sigmoid curves. Averaging active bets. <br>*(AFML Ch 10)* | **Implemented** <br> Uses `directional_size_from_prob` with calibrated probabilities (`IsotonicRegression`). | ⭐⭐⭐⭐ (Strong) | Code conceptually aligns. The use of calibration (`Isotonic`) before sizing is a best practice often emphasized in MLAM. |
| **Sequential Bootstrap** | Sampling to minimize redundancy probability. <br>*(AFML Ch 4)* | **Proxy** <br> Uses Weighted Bootstrapping (`sample_weight=uniqueness` passed to classifiers). | ⭐⭐⭐⭐ (Good) | While `SequentialBootstrap` is a specific algorithm, passing Uniqueness weights to a standard Bootstrap (Bagging) achieves the mathematically equivalent goal of downweighting redundant observations. |

## Detailed Analysis

### 1. Strengths
- **Weights Engine (`generate_weights_per_label`)**: The implementation of uniqueness and concurrency is sophisticated and math-heavy, closely following Chapter 4 of AFML. The addition of "Consistency" and "Time Decay" extends the concept logically.
- **Dynamic Thresholds**: The logic to handle "Regime Aware" thresholds and volatility scalars aligns well with the "Structural Breaks" concerns in MLAM.

### 2. Potential Areas for Improvement (Misalignments)
- **CV Leakage (`_cross_val_predict_proba_weighted`)**: 
    - The code uses `cv = TimeSeriesSplit(n_splits=n_splits)`.
    - **Issue**: In time-series finance, labels often extend into the future (Horizon $H$). 
    - If Train ends at $T$ and Test starts at $T+1$, the label for Train[$T$] depends on price at $T+H$. The label for Test[$T+1$] depends on price at $T+1+H$.
    - These paths overlap almost entirely. The model trains on the outcome of a path it is about to test on.
    - **De Prado Solution**: *PurgedKFold*. You must drop training samples where `t_end > test_start`.
    - **Recommendation**: Implement `PurgedTimeSeriesSplit` or ensure a `gap` parameter is passed to `TimeSeriesSplit` equal to the maximum event horizon.

- **Deflated Sharpe Calculation**:
    - The use of `_soft_sharpe_scale` (arcsinh) is a custom heuristic. De Prado typically advocates for `Probabilistic Sharpe Ratio (PSR)` or `Deflated Sharpe Ratio (DSR)` to account for multiple testing (selection bias).
    - While the HPO does hierarchical optimization (TPE), it doesn't explicitly calculate DSR to penalize the "Best" result based on the number of trials run.

## Conclusion
The codebase is **90% aligned** with De Prado's framework. It is a high-fidelity implementation of the *Financial Machine Learning* pipeline (Features -> Triple Barrier -> Meta Labeling -> Bet Sizing). The only significant deviation is the use of standard `TimeSeriesSplit` without an explicit Purge/Embargo gap, which introduces a mild leakage risk during the HPO process.
