# Leakage Prevention Rules

Data leakage is one of the most common sources of false alpha.

All feature pipelines must follow these rules.

---

# 1. Temporal Leakage

Features must not use future data.

Invalid:

rolling_mean(price, window) computed using future observations

Valid:

rolling_mean(price[:t], window)

---

# 2. Normalization Leakage

Global normalization is prohibited.

Invalid:

zscore = (x - mean(full_dataset)) / std(full_dataset)

Valid:

rolling_zscore using past data only.

---

# 3. Cross-Sectional Leakage

When using cross-sectional features:

only assets available at time t may be used.

Future asset additions must not influence past features.

---

# 4. Label Leakage

Targets must not influence features.

Examples of leakage:

- features derived from forward returns
- labels used during feature normalization

---

# 5. Alignment Errors

Ensure correct alignment between:

features  
targets  
timestamps

Common bug:

feature_t aligned with target_t instead of target_{t+1}

---

# 6. Validation

Pipelines must include checks for:

timestamp monotonicity  
feature/target alignment  
missing timestamps
