# Dataset Contract

This document defines the **data semantics and guarantees** for datasets used in this repository.

All datasets must satisfy these rules.

---

# 1. Timestamp Semantics

All timestamps must represent **the time at which the data becomes observable**.

Example:

For OHLCV bars:

timestamp = bar close time

This ensures that features derived from a bar are available **only after the bar completes**.

---

# 2. Feature Alignment

Features must satisfy:

feature_t uses data ≤ t

Targets must satisfy:

target_t uses data > t

Example:

feature_t = indicators computed using prices ≤ t  
target_t = return from t → t + horizon

---

# 3. Target Definition

Targets must be clearly defined.

Example:

target = forward_return(price, horizon)

Horizon must be explicitly documented.

---

# 4. Universe Definition

Each dataset must specify the asset universe.

Examples:

top N by liquidity  
fixed symbol list  
dynamic universe with filtering rules

Universe construction must be deterministic.

---

# 5. Missing Data

Missing data policy must be explicit.

Options:

- forward fill
- zero fill
- drop observations
- mask invalid rows

Implicit handling is not allowed.

---

# 6. Numeric Precision

Datasets must use memory-efficient types when possible.

Preferred types:

float32  
int32  
int8

float64 should only be used when required.
