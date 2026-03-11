# Model Validation Protocol

This document defines the validation rules for machine learning models in this repository.

The purpose of this protocol is to ensure that reported results are:

- statistically valid
- out-of-sample robust
- economically meaningful
- reproducible

Models that do not satisfy this protocol must not be treated as research evidence.

---

# 1. Time-Series Validation Only

Validation must respect temporal ordering.

Allowed:

- holdout split by time
- walk-forward validation
- rolling validation
- expanding-window validation
- purged cross-validation

Not allowed:

- random train/test splits
- shuffled K-fold cross-validation
- any split that mixes future observations into training

---

# 2. Immutable Train / Validation / Test Separation

Each experiment must define:

- training period
- validation period
- final test period

Rules:

- training data is used for fitting model parameters
- validation data is used for model selection and hyperparameter tuning
- test data is used once for final evaluation

The test set must never be used for iterative tuning.

---

# 3. Walk-Forward Evaluation

Models should be evaluated across multiple sequential folds.

Example:

- train: 2012-2016, validate/test: 2017
- train: 2012-2017, validate/test: 2018
- train: 2012-2018, validate/test: 2019

Walk-forward results are preferred over a single split because they test regime robustness.

---

# 4. Purged Cross-Validation

When labels have forward horizons or overlapping event windows, standard temporal folds are insufficient.

Validation must use purging when:

- targets depend on future returns over a horizon
- events overlap in time
- labels span multiple bars

Purging rule:

- remove training observations whose label interval overlaps the validation interval

This prevents leakage from overlapping outcomes.

---

# 5. Embargo

An embargo period must be applied after each validation fold when appropriate.

Purpose:

- prevent near-boundary contamination
- reduce leakage from serial dependence
- avoid training on observations too close to the validation window

Embargo length should be based on one of:

- maximum label horizon
- a fixed fraction of the sample
- a domain-specific holding-period assumption

Embargo assumptions must be documented.

---

# 6. Label Horizon Awareness

Validation logic must explicitly account for label construction.

If:

target_t = return from t to t+H

then split logic must treat the effective observation interval as:

[t, t+H]

Validation code must not assume labels are point-in-time when they are horizon-based.

---

# 7. Hyperparameter Search Discipline

Hyperparameter tuning must occur only on training/validation data.

Required workflow:

1. fit candidate models on training data
2. compare them on validation data
3. select one configuration
4. evaluate once on the final test set

Forbidden workflow:

- tune hyperparameters based on repeated test-set evaluation
- promote a model because it “looks best” on test

---

# 8. Model Comparison Fairness

Competing models must be evaluated under identical conditions.

Keep fixed across comparisons:

- dataset version
- universe
- feature set, unless the comparison is explicitly about features
- label definition
- split logic
- transaction cost assumptions
- evaluation metrics

Do not compare models across mismatched setups.

---

# 9. Baseline Models Are Mandatory

Every model evaluation must include baseline comparators.

Examples:

- random classifier
- constant predictor
- simple linear model
- naive momentum / mean-reversion rule
- previous production or reference model

A complex model is only useful if it beats credible baselines.

---

# 10. Multiple Testing Control

Testing many hypotheses increases false discovery risk.

When many models, features, or parameter sets are tried, apply additional skepticism.

Preferred controls include:

- out-of-sample replication
- Deflated Sharpe Ratio
- Probabilistic Sharpe Ratio
- White’s Reality Check
- reporting total search breadth

Do not treat the best backtest from a large search as reliable by default.

---

# 11. Metric Requirements

Validation must report both statistical and economic metrics.

Statistical metrics may include:

- AUC
- log loss
- precision / recall
- calibration metrics

Economic metrics should include:

- annualized return
- Sharpe ratio
- maximum drawdown
- turnover
- transaction cost sensitivity

Statistical performance alone is insufficient.

---

# 12. Fold Stability

A model should not be accepted based only on strong aggregate performance.

Validation must also examine stability across folds.

Examples:

- mean metric across folds
- standard deviation across folds
- fraction of positive folds
- worst-fold performance

Models with unstable fold behavior should be treated with caution.

---

# 13. Significance of Improvement

When claiming one model improves on another, the improvement must be meaningful.

Check:

- consistency across folds
- consistency across market regimes
- robustness after costs
- robustness to small parameter changes

Small average improvements with high variance are not strong evidence.

---

# 14. Threshold Selection

If a model outputs scores or probabilities, threshold selection must be validated.

Thresholds must be chosen using:

- training data only
- validation data only
- a predefined rule

Thresholds must not be selected using the final test set.

---

# 15. Calibration

Where relevant, predicted probabilities or scores should be checked for calibration.

Poorly calibrated outputs can create misleading position sizing or trade selection behavior.

Calibration evaluation may include:

- reliability curves
- Brier score
- bucketed outcome analysis

---

# 16. Regime Robustness

Validation should test whether performance is concentrated in one regime only.

Check performance across:

- volatility regimes
- bull / bear markets
- liquidity regimes
- different asset subsets

A model that only works in one narrow environment is unlikely to generalize.

---

# 17. Reproducibility

Every validation run must be reproducible.

Record at minimum:

- experiment_id
- dataset version
- feature pipeline version
- model configuration
- split definition
- label definition
- random seeds
- code commit hash

Re-running the same validation should reproduce the same outputs.

---

# 18. Final Test Usage

The final test set is for final confirmation only.

Rules:

- do not repeatedly inspect and adjust after seeing test performance
- do not use the test set for threshold tuning
- do not use the test set for feature selection

If the test set influences development, it is no longer a valid test set.

---

# 19. Failure Conditions

A validation result should be rejected if any of the following occur:

- leakage is detected
- train/validation/test boundaries are violated
- purging is required but absent
- embargo is required but absent
- evaluation ignores transaction costs
- results are not reproducible
- test-set reuse influenced the model choice

---

# 20. Acceptance Standard

A model may be considered credible only if it demonstrates:

- valid time-aware validation
- no detectable leakage
- stable fold-level behavior
- out-of-sample economic value
- reproducible results
- improvement over reasonable baselines

Passing one backtest is not enough.
