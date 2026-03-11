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
- indicators computed from that bar may only be used **from t+1 onward**

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

# 7. Horizon Consistency

If a model predicts horizon H:

target_t = return(t → t+H)

Feature windows must not exceed the horizon in a way that leaks future information.

Example:

If H = 10 bars:

rolling features must only use `[t-k … t]`.

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

# 10. Asset Isolation

When computing rolling statistics:

Never allow **data from one asset to influence another asset**.

Example:

Incorrect:

rolling volatility computed across mixed assets.

Correct:

compute rolling statistics **per asset group**.

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

---

# 13. Feature Validation Checks

Feature pipelines should include validation checks:

- timestamp monotonicity
- NaN ratio
- infinite values
- extreme outliers
- alignment with targets

Pipelines must fail early when invalid features are detected.

---

# 14. Reproducibility

Feature generation must depend only on:

- dataset version
- pipeline parameters
- code version

No hidden external state may influence feature values.

---

# Summary

A valid feature pipeline must guarantee:

- strict causality
- no lookahead bias
- correct timestamp alignment
- cross-asset isolation
- deterministic execution
- reproducible outputs
