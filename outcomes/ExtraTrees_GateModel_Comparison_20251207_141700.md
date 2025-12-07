# ExtraTrees Gate Model Comparison

_Date: 2025-12-07 14:17:00_

This note compares two ExtraTrees-based gate configurations used as the "Dumb Manager" on top of Analyst LGBM predictions.

The gate consumes:
- Regime / OHLCV features
- Trade-history features
- Analyst LGBM out-of-fold predictions (as an explicit feature)

and outputs a probability-like gate score that is thresholded to accept or block trades.

---

## 1. Configurations

### 1.1 Baseline Gate (Config 1)

```python
from sklearn.ensemble import ExtraTreesClassifier

gate_model = ExtraTreesClassifier(
    n_estimators=500,       # More trees = smoother probability surface
    max_depth=3,            # Rigid constraint: Only learn 3 levels of logic
    min_samples_leaf=0.05,  # Rule must apply to at least 5% of trades (~50 of 1000)
    max_features="sqrt",   # Force diversity among trees
    bootstrap=True,         # Use bootstrapped samples for each tree
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
```

### 1.2 Safer / Slightly Smarter Gate (Config 2)

Same backbone as Config 1, but with tweaked capacity and safety:

```python
from sklearn.ensemble import ExtraTreesClassifier

gate_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=4,            # Allowed to be slightly smarter (one extra level)
    min_samples_leaf=0.06,  # INCREASED SAFETY: ~6% of trades per leaf (~60 of 1000)
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
```

All other training details (features, label definition, calibration logic) are held constant between the two configs.

---

## 2. Hyperparameter Differences

| Hyperparameter     | Config 1 (Baseline) | Config 2 (Safer/Smarter) | Effect |
|--------------------|---------------------|---------------------------|--------|
| `n_estimators`     | 500                 | 500                       | Same ensemble size; similar variance reduction from bagging. |
| `max_depth`        | 3                   | 4                         | Config 2 can express one extra level of interaction; more flexible partitions of regime × analyst-pred space. |
| `min_samples_leaf` | 0.05                | 0.06                      | Config 2 enforces larger leaves (≥6% vs ≥5% of samples), making rules apply to more trades and reducing overfitting to small lucky clusters. |
| `max_features`     | `"sqrt"`           | `"sqrt"`                 | Same: strong randomization of feature subspaces per split. |
| `bootstrap`        | `True`              | `True`                    | Same: classic bagging effect. |
| `class_weight`     | `"balanced"`       | `"balanced"`             | Same: still compensates for class imbalance in profitable vs unprofitable trades. |
| `n_jobs`           | `-1`                | `-1`                      | Same parallelism. |
| `random_state`     | `42`                | `42`                      | Same RNG seed for reproducible comparisons. |

---

## 3. Expected Behavioral Differences

### 3.1 Capacity vs Safety

- **Config 1 (depth 3, leaf 5%)**
  - Lower capacity: fewer interaction layers between regime, trade history, and Analyst LGBM score.
  - Slightly smaller minimum leaf: can create more granular rules, including narrower regions of feature space.
  - More susceptible to capturing small pockets of luck (but still constrained compared to an unconstrained ExtraTrees).

- **Config 2 (depth 4, leaf 6%)**
  - One extra level of depth: can model slightly more complex gating logic (e.g. additional split on regime or on the Analyst prediction).
  - Larger minimum leaf: every terminal rule must hold over more trades, which:
    - Increases statistical reliability of each rule.
    - Reduces the chance that the gate overfits to small, noisy subsets.
  - Net effect: often **similar or slightly higher complexity** than Config 1, but with **stronger safety on sample size per rule**.

### 3.2 Bias–Variance Intuition

- **Config 1**
  - Somewhat higher variance (smaller leaves) but slightly higher bias (shallower depth) relative to Config 2.
  - In practice, can still overreact to rare but lucky patterns if they occupy ~5% of the data.

- **Config 2**
  - Depth 4 slightly lowers bias (more expressive trees).
  - Leaf size 0.06 raises bias but reduces variance by forcing rules to generalise across more samples.
  - Given the modest change (5% → 6%), this is a gentle nudge towards **robust rules that still allow moderate complexity**.

### 3.3 Impact on Gating Behavior

Because both configs see the **Analyst LGBM prediction** as an explicit feature, plus regime and trade-history context:

- **Config 1** is more willing to carve out slightly narrower regions of “accept” and “block” around the Analyst score and regime features.
  - May admit more aggressive, niche regimes where Analyst looks good historically.
  - Risk: if those regimes are based on limited data, real-world performance can degrade.

- **Config 2** tends to:
  - Ignore ultra-niche regimes that don’t have enough supporting samples.
  - Focus on patterns that hold across multiple dozens of trades.
  - Provide a **more conservative, stable gate**, especially in sparsely populated corners of the feature space.

---

## 4. Practical Recommendations

- If you care most about **robustness and avoiding small clusters of luck**, Config 2 is preferable:
  - Slightly deeper trees to remain expressive.
  - Slightly larger leaves to enforce a minimum sample size per rule.

- If you want maximum **granularity** and are comfortable with more variance (e.g. in simulation / exploration), Config 1 remains useful as a baseline.

For production gating, a reasonable path is:
1. Train and evaluate both on identical data and thresholds.
2. Compare:
   - Coverage (fraction of trades accepted).
   - Lift in average net return vs ungated baseline.
   - Stability across time splits.
3. Prefer the config that maintains lift with lower volatility in performance across regimes and time.
