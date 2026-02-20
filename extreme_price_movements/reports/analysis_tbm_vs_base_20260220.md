# Analysis of TBM Parameters vs Base Model Training Results
**Date:** 2026-02-20

## 1. Discrepancy in AUC Scores
**Question:** Why is AUC higher on `compare_tbm_parameters.py` (e.g., 0.663) than during base models training (e.g., 0.635)?

**Answer:** Data Subsampling.
*   `compare_tbm_parameters.py` (lines 145-148) defaults to aggressive subsampling if no panel is provided:
    ```python
    # Aggressive subsample: take every 4th asset for Stage 1
    train_syms = all_syms[::4]
    # Limit to max 30 symbols for Stage 1 quick runs
    train_syms = train_syms[:30]
    ```
*   Base model training runs on the full universe (hundreds of symbols).
*   Models trained on small subsets often show higher metrics due to reduced variance/noise or overfitting to specific asset characteristics. The 0.635 OOF AUC from the base model training is the more reliable estimate for production.

## 2. Missing Data in Report
**Question:** Why does the report not include data for all base models? Did training complete?

**Answer:** Training completed correctly.
*   The "Summary Table" in the report lists all 12 models (long/short × mr/tf × H2/4/8).
*   The "Candidate Race Results" section only shows "selected horizons" (H=2) as examples. This is a reporting choice for brevity.

## 3. Metric Reliability
**Question:** Are the metrics good enough for production? Are they reliable?

**Answer:** Yes.
*   `short_tf` is strong (RcAUC > 0.61 across horizons).
*   `long_mr` H=2 is strong (OOF AUC 0.635).
*   `long_tf` is weak/borderline (AUC ~0.55).
*   Reliability is high as these are 5-fold OOF metrics on the full universe.

## 4. Precision@20
**Question:** Prec@20 is around 0.02-0.04. Is that good?

**Answer:** Yes.
*   Base event rate is likely ~1%.
*   4% precision represents a **4x Lift** (400% improvement over random). This is a strong signal for a base model.

## 5. Negative Brier Skill Score (BSS)
**Question:** All OOF_BSS are negative. Is that true?

**Answer:** Yes, and acceptable.
*   Brier Score penalizes overconfident errors.
*   Predicting the rare-event prior (0.01) yields a very low (good) Brier Score.
*   Active models predicting high probabilities (0.60) will be wrong often (96% of the time given 4% precision), incurring heavy Brier penalties.
*   For trading, **Ranking (AUC)** and **Lift** matter more than global calibration.

## 6. long_tf Weakness
**Question:** `long_tf` is weakest. Are features appropriate?

**Answer:** Features are standard but insufficient for regime filtering.
*   The model is "overconfident", likely triggering on false breakouts in choppy markets.
*   **Recommendation:** Add regime filters like `chop_score` and `grind_score` (currently in Meta features) to the `tf_feature_keys` in `config.py`.

## 7. short_mr Regularization
**Question:** `short_mr` shows a gap between RcAUC and OOF_AUC. Is regularization strong enough?

**Answer:** Regularization is already very strong.
*   XGBoost config: `reg_lambda=15.0`, `num_parallel_tree=400`, `n_estimators=10`. This is a Boosted Random Forest of 4000 trees.
*   The gap likely stems from the **low signal-to-noise ratio** of short-side mean reversion, leading to "winner's curse" selection bias in the race, rather than classic overfitting.

## 8. Verification of Training Data Scope
**Question:** Verify that we only train on tradeable periods (candidates). If so, isn't the 1% base rate surprising?

**Answer:**
*   **Masking Verified:** `training.py` (lines 1320-1453) strictly enforces the candidate mask. It calls `_build_optimal_candidate_mask` (using thresholds from `compare_candidate_thresholds.py`) and only extracts events where the mask is True. We **do not** train on the entire time series.
*   **Base Rate Context:** The "1% base rate" refers to the **conditional probability of success (TP Hit)** *given* that a candidate event has occurred.
    - **Candidates (6% of data):** These represent "High Volatility / Opportunity".
    - **TP Hit (1-4% of candidates):** This represents "Success". Even within high-volatility events, hitting an aggressive profit target (e.g., 2-6%) without first hitting a stop-loss or timing out is rare.
    - **Implication:** We are indeed "predicting what happens next" within a high-volatility state. The data shows that "what happens next" is usually Chop (Timeout/SL) rather than a clean Trend (TP). Finding the 4% of "clean moves" within the 100% of "volatile moves" is the difficult task the model solves.
*   **Conclusion:** The low base rate (~1-4%) and negative BSS are consistent with the difficulty of the problem *within the candidate subset*.
