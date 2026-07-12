# Feature Pipeline Rules

This document defines the rules governing **feature generation and feature pipelines** in this repository.

Feature engineering is one of the most common sources of:

- data leakage
- timestamp misalignment
- hidden lookahead bias
- non-reproducible signals

All feature pipelines must follow these rules.

---

# 1. Causality Requirement

Features must only use information available **at or before timestamp t**.

Formally:

feature_t = f(data ≤ t)

Targets must use **future information only**:

target_t = g(data > t)

Example:

Correct:

return_1h[t] = (price[t+1h] - price[t]) / price[t]

Incorrect:

feature_t uses price[t+1]

---

# 2. Timestamp Semantics

Every dataset must clearly define timestamp meaning.

Recommended convention:

timestamp = **bar close time**

Implication:

- OHLCV bar at time t becomes available **after the bar closes**
- indicators computed from that bar may be used by a decision stamped `t` only
  when `t` explicitly means post-close observability
- execution must use the next explicitly executable event after that decision

If the dataset uses a different convention it must be documented.

---

# 3. Rolling Window Rules

Rolling features must only include **past observations**.

Correct:

rolling_mean(price, window=20) using data `[t-19 … t]`

Incorrect:

rolling_mean(price) computed on the entire dataset.

---

# 4. Normalization Rules

Global normalization is prohibited.

Incorrect:

zscore = (x - mean(full_dataset)) / std(full_dataset)

Correct approaches:

- rolling normalization
- expanding window normalization
- normalization computed using **training data only**

Feature normalization must also preserve portability across exchanges and assets.

Required:

- Feature values must be expressed in exchange-portable and asset-portable units whenever possible.
- Features must be portable across exchanges by design, not tied to one exchange's instrument setup, contract metadata, liquidity regime, precision rules, or quote-asset conventions.
- Prefer returns, percentages, basis points, ratios, ATR-normalized distances, volatility-normalized distances, ranks, z-scores, and liquidity-normalized measures over raw prices, raw quote notionals, raw volumes, or exchange-native tick sizes.
- Any feature that depends on quote currency, contract multiplier, lot size, tick size, leverage model, or exchange-specific market metadata must be explicitly normalized into a common unit before it can be used by a model.
- Features used by both spot and perp pipelines must have identical semantics across exchanges. If semantics differ, split them into explicitly named market-mode-specific features.
- Identity features are the only exception to portability. They must be few, explicitly named, and reviewed before model use.

Not allowed:

- Asset-specific raw price levels as model inputs.
- Exchange-specific raw market metadata as model inputs.
- Hidden symbol, venue, contract, or quote-currency dependencies embedded in otherwise generic feature names.
- Per-exchange normalization logic that changes a feature's meaning without changing the feature name.
- Training a deployed model on features whose semantics only hold for one exchange implementation.

---

# 5. Cross-Sectional Features

When computing cross-sectional features across assets:

Only assets **available at timestamp t** may be used.

Examples:

valid cross-sectional operations:

- cross-sectional rank
- cross-sectional z-score
- cross-sectional percentile

Invalid operations:

- using future asset universe
- computing statistics using assets not yet listed

---

# 6. Forward-Looking Targets

Targets must be constructed **after features are finalized**.

Example:

features_t → predict → forward_return[t+1 : t+h]

Do not construct features that depend on forward returns.

---

# 7. Horizon And Lookback Consistency

If a model predicts horizon H:

target_t = return(t → t+H)

Feature lookbacks may be longer than the prediction horizon. A 3-7 hour target
may validly use daily, weekly, or longer historical context. The requirement is
that every lookback ends at or before the feature timestamp and that target data
starts only after the executable decision point.

Document both lookback and target horizon because they serve different roles.

---

# 8. Deterministic Pipelines

Feature pipelines must be **deterministic**.

Running the pipeline twice with identical inputs must produce identical outputs.

Sources of nondeterminism must be controlled:

- random seeds
- unordered operations
- multi-threaded nondeterministic reductions

---

# 9. Memory Constraints

Feature pipelines must respect repository memory limits.

Guidelines:

- prefer `float32`
- avoid unnecessary array copies
- reuse buffers when possible
- avoid large intermediate pandas objects
- use NumPy or Numba for heavy computation

---

# 10. Per-Asset Isolation And Cross-Asset Features

When computing rolling statistics:

Per-asset rolling calculations must never accidentally mix symbols.

Example:

Incorrect:

rolling volatility computed across mixed assets.

Correct:

compute rolling statistics **per asset group**.

Explicit cross-asset features are allowed and encouraged when they are causal.
Breadth, dispersion, cross-asset correlation, OI breadth, market residuals, and
peer features must use only assets and observations available at timestamp `t`.
Their feature names and metadata must identify the cross-sectional semantics.

---

# 11. Feature Stability

Features should be evaluated for stability across time.

Unstable features include:

- regime-specific artifacts
- signals dependent on microstructure anomalies
- signals that vanish under small parameter changes

Robust features should show **consistent behavior across folds**.

---

# 12. Feature Metadata

Every feature must be documented.

Minimum metadata:

feature_name
description
data sources
lookback window
units
expected range

Features without documentation should not be used in models.

Deployed models may only train on features guaranteed by their inference contract.

Required:

- Before training a deployable model, validate every selected training feature against that model's inference feature contract.
- A training feature is eligible only if inference can compute it with the same name, same semantics, same timestamp convention, same normalization, and same transform contract.
- Base-model, meta-model, LGBM leaf/value, performance-derived, cross-asset, and market-regime features all require the same contract validation.
- Features that are available in offline training but not guaranteed live must be excluded before feature selection, model fitting, and optimiser threshold discovery.
- If a feature is expensive, exchange-specific, delayed, sparse, or only available for some venues/assets, the contract must state the availability rule and strict inference must reject symbols where it is unavailable.

Not allowed:

- Training a deployed model on offline-only features.
- Training on features that inference would synthesize, approximate, stale-fill, or silently zero-fill.
- Training on features missing from `FeatureTransformContract.transformed_feature_cols` or approved passthrough outputs.
- Letting feature selection retain columns that strict inference cannot reproduce exactly.

---

# 13. Feature Validation Checks

Feature pipelines should include validation checks:

- timestamp monotonicity
- NaN ratio
- infinite values
- extreme outliers
- alignment with targets

Pipelines must fail early when invalid features are detected.

Feature computation failures must be fail-closed.

Required:

- If a feature computation fails, that feature must not be replaced by a synthetic fallback value.
- If a required model feature cannot be computed exactly, the affected symbol or inference cycle must be rejected according to the configured strict parity scope.
- Missing, non-finite, stale, or malformed required features must be surfaced as explicit validation errors with the feature name, symbol, timestamp, and reason.
- Backfills, forward fills, zero fills, medians, or cached substitutes are allowed only when they are part of the frozen training-time `FeatureTransformContract` and are applied identically in training, backtest, and inference.
- Inference must not invent live-only substitutes for model inputs.

Not allowed:

- Silent zero-fill for failed feature computation.
- Approximate fallback formulas for required model features.
- Reusing stale cached values when the current timestamp is required.
- Continuing to generate predictions after a required feature failed strict validation.

---

# 14. Reproducibility

Feature generation must depend only on:

- dataset version
- pipeline parameters
- code version

No hidden external state may influence feature values.

---

# 15. Current AE/GMM Feature Contract

- The default state-input policy is `a0bis`: use ATR-normalized momentum/trend
  variants for AE/GMM state discovery where available.
- Fit the scaler, denoising AE, and GMM on authorized training data only.
- Sample across beginning, middle, and end subperiods; record actual row counts.
- Target approximately 15k AE rows and up to 100k GMM rows when available.
- Freeze the fitted state across later growing OOS windows so cluster/posterior
  meanings do not change fold by fold.
- Downstream models may receive cluster ID, posterior vector, entropy,
  Mahalanobis/distance measures, reconstruction error, speed, and acceleration.
- Outcome-based cluster descriptions are diagnostics or train-derived priors;
  they are not live inputs unless predicted from causal pre-entry features.

---

# Summary

A valid feature pipeline must guarantee:

- strict causality
- no lookahead bias
- correct timestamp alignment
- explicit per-asset and causal cross-asset semantics
- deterministic execution
- reproducible outputs
