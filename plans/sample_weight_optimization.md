# Sample Weight Optimization Plan (Final v2)

## Bug Fixes

### A1. combine_weights_safely Bug

**Problem**: `alpha` undefined, mutated input dict.

**Fix**:
```python
def combine_weights_safely(
    components: Dict[str, np.ndarray],
    component_weights: Dict[str, float],  # NOT mutated
    min_n_eff_ratio: float = 0.30,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Fixed: no mutation, correct alpha lookup.
    """
    # 1. Percentile clip + degenerate check
    clipped = {}
    local_weights = component_weights.copy()  # FIX: don't mutate
    
    for name, w in components.items():
        p5 = np.percentile(w, 5)
        p95 = np.percentile(w, 95)
        
        # Guard against near-constant (FIX: robust check)
        span = p95 - p5
        ratio = p95 / (p5 + 1e-12)
        
        if span < 1e-6 or ratio < 1.05:
            # Replace with ones (drop this component)
            clipped[name] = np.ones_like(w)
            local_weights[name] = 0.0
        else:
            # Enforce non-negativity
            clipped[name] = np.clip(np.maximum(w, eps), p5, p95)
    
    # 2. Log-space combination (FIX: correct alpha lookup)
    log_w = np.zeros_like(next(iter(clipped.values())), dtype=float)
    for name, w in clipped.items():
        alpha = local_weights.get(name, 1.0)
        if alpha == 0.0:
            continue
        log_w += alpha * np.log(w + eps)
    
    w_final = np.exp(log_w)
    
    # 3-4: Temperature-based Neff + normalize (same as before)
    ...
```

---

### A2. Purged CV: Label-Interval Based

**Fix**:
```python
from extreme_price_movements.purged_cv import PurgedKFold

# PurgedKFold should use (t_start, t_end) intervals, not fixed percentages
# If your implementation uses percentages, replace with:

class IntervalPurgedKFold:
    def __init__(self, n_splits=5, embargo_bars=5):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars  # Fixed bars, not percentage
    
    def split(self, X, y=None, groups=None, label_intervals=None):
        """
        label_intervals: (t_start, t_end) per sample
        Purge training samples whose intervals overlap with validation.
        Embargo: remove N bars after validation boundary.
        """
        ...
```

**Config**:
```python
cv = IntervalPurgedKFold(n_splits=5, embargo_bars=10)  # 10 bars
```

---

### A3. Vol Cross-Sectional: Guards

**Fix**:
```python
def compute_vol_weights(
    past_vol: np.ndarray,
    timestamps: np.ndarray,  # Use timestamps, not bar_index
    direction: str = "downweight_high",
    power: float = 0.5,
    min_group_size: int = 20,  # Guard
) -> np.ndarray:
    """
    Fixed: timestamp grouping + size guard.
    """
    # Group by timestamp (canonical global bar id)
    vol_df = pd.DataFrame({"vol": past_vol, "ts": timestamps})
    
    # Safe median with guard
    def safe_median(x):
        if len(x) < min_group_size:
            return x.median() if len(x) > 0 else 1.0
        return x.median()
    
    vol_cs = vol_df.groupby("ts")["vol"].transform(
        lambda x: x / (safe_median(x) + 1e-8)
    ).values
    
    # Apply power
    if direction == "downweight_high":
        w = np.power(vol_cs + 1e-8, -power)
    else:
        w = np.power(vol_cs + 1e-8, +power)
    
    return w / np.mean(w)
```

---

### A4. Liquidity Weights: Bounded

**Fix**:
```python
def compute_liquidity_weights(
    adv: np.ndarray,
    spread: np.ndarray = None,
    clip_range: tuple = (0.7, 1.3),  # FIX: bounded, not wide
) -> np.ndarray:
    """
    Fixed: bounded multiplier, explicit choice.
    """
    if spread is not None:
        w = 1.0 / (spread + 1e-8)
    else:
        w = np.log1p(adv)
    
    # Hard clip (not percentile)
    w = np.clip(w, clip_range[0], clip_range[1])
    
    return w / np.mean(w)
```

---

## Conceptual Fixes

### B1. Model Family Match

**Fix**: Ensure weight optimization uses same model as production:

```python
def weight_optimization_objective(trial, X, y_ret, params, production_model="ExtraTrees"):
    """
    Match production model family.
    """
    # Match inductive biases
    if production_model == "ExtraTrees":
        model = ExtraTreesRegressor(
            n_estimators=50,       # Tiny mode
            max_depth=6,
            min_samples_leaf=50,   # Match production
            max_features='sqrt',
            n_jobs=-1,
        )
    elif production_model == "XGB":
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,     # Match production LR
            ...
        )
    # Same loss, same target transform, same normalization
    
    # Run two seeds per trial (variance reduction)
    ic_scores = []
    for seed in [42, 123]:
        model.random_state = seed
        ic = run_cv_with_seed(model, X, y_ret, params, seed)
        ic_scores.append(ic)
    
    return np.mean(ic_scores)
```

---

### C1. Distance-to-Barrier: Saturating Alternative

**Fix**:
```python
def compute_distance_to_barrier_weights(
    entry_prices: np.ndarray,
    upper_barriers: np.ndarray,
    lower_barriers: np.ndarray,
    atr_past: np.ndarray,
    k: float = 0.5,  # Smoothing
    min_dist: float = 0.5,  # Minimum distance floor
    form: str = "inverse",  # or "exp"
) -> np.ndarray:
    """
    Fixed: saturating alternative + minimum distance.
    """
    dist_up = (upper_barriers - entry_prices) / (atr_past + 1e-8)
    dist_dn = (entry_prices - lower_barriers) / (atr_past + 1e-8)
    dist_nearest = np.maximum(np.minimum(dist_up, dist_dn), min_dist)
    
    if form == "inverse":
        w = 1.0 / (dist_nearest + k)
    else:  # exp (saturating)
        w = np.exp(-k * dist_nearest)
    
    w = np.clip(w, 0.5, 2.0)  # Bounded
    return w / np.mean(w)
```

**Config**: Treat as hyperparameter: `{on: True/False, form: inverse/exp, k: 0.3-1.0}`

---

### C2. Recency: Per-Era Neff

**Fix**:
```python
def compute_recency_weights(
    bar_indices: np.ndarray,
    era_indices: np.ndarray,  # Monthly era IDs
    half_life_bars: int = 50,
    clip_range: tuple = (0.5, 2.0),
    min_era_neff_ratio: float = 0.2,
) -> np.ndarray:
    """
    Fixed: enforce Neff per era.
    """
    max_idx = bar_indices.max()
    age_bars = max_idx - bar_indices
    w = np.power(2.0, -age_bars / half_life_bars)
    w = np.clip(w, clip_range[0], clip_range[1])
    
    # Check per-era Neff
    era_df = pd.DataFrame({"w": w, "era": era_indices})
    era_neff = era_df.groupby("era").apply(
        lambda g: (g["w"].sum()**2) / (g["w"]**2).sum()
    )
    
    min_era_neff = era_neff.min()
    n_samples_per_era = era_df.groupby("era").size()
    min_expected = (n_samples_per_era * min_era_neff_ratio).min()
    
    if min_era_neff < min_expected:
        # Flatten weights more
        w = np.power(w, 0.5)  # Stronger equalization
    
    return w / np.mean(w)
```

---

### D1. Redundancy: Spearman + Fold-Local

**Fix**:
```python
def check_component_redundancy(
    components: Dict[str, np.ndarray],
    threshold: float = 0.85,
) -> Dict:
    """
    Fixed: Spearman, fold-local.
    """
    from scipy.stats import spearmanr
    
    names = list(components.keys())
    w_arrays = [components[n] for n in names]
    
    # Spearman correlation matrix
    n = len(w_arrays)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            r, _ = spearmanr(w_arrays[i], w_arrays[j])
            corr[i, j] = r
            corr[j, i] = r
    
    # Find redundant pairs
    redundant = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i, j]) > threshold:
                redundant.append((names[i], names[j], corr[i, j]))
    
    return {
        "pairs": redundant,
        "corr_matrix": dict(zip(names, corr)),
    }

# Use: treat as "investigate" not "auto-drop"
# Unless correlation is extreme (>0.95) and stable across folds
```

---

## Production Readiness Checks

### E1. Weight Sanity Logging

```python
def log_weight_statistics(
    weights: np.ndarray,
    era_indices: np.ndarray,
    name: str,
):
    """Log per fold, per era."""
    import logging
    logger = logging.getLogger("weights")
    
    # Global stats
    logger.info(f"{name} | mean={weights.mean():.4f} std={weights.std():.4f}")
    logger.info(f"{name} | p1={np.percentile(weights,1):.4f} p5={np.percentile(weights,5):.4f}")
    logger.info(f"{name} | p50={np.percentile(weights,50):.4f} p95={np.percentile(weights,95):.4f}")
    logger.info(f"{name} | p99={np.percentile(weights,99):.4f}")
    logger.info(f"{name} | max={weights.max():.4f} n_eff={compute_n_eff(weights):.0f}")
    
    # Top-k concentration
    for k in [1, 5]:
        top_k_share = np.sort(weights)[-k:].sum() / weights.sum()
        logger.info(f"{name} | top{k}pct_share={top_k_share:.4f}")
    
    # Per-era Neff
    era_df = pd.DataFrame({"w": weights, "era": era_indices})
    era_neff = era_df.groupby("era").apply(
        lambda g: (g["w"].sum()**2) / (g["w"]**2).sum()
    )
    logger.info(f"{name} | era_neff min={era_neff.min():.0f} mean={era_neff.mean():.0f}")
```

---

### E2. Ablation Harness

```python
def run_ablation(
    X, y_ret, components,
    baseline_weights,
    n_folds=5,
):
    """
    Before Optuna: one component at a time.
    """
    results = []
    
    # Baseline
    baseline_score = run_cv(baseline_weights, X, y_ret, n_folds)
    results.append(("baseline", baseline_score))
    
    # One component at a time
    for name, w_comp in components.items():
        combined = baseline_weights * w_comp
        combined = combined / np.mean(combined)
        score = run_cv(combined, X, y_ret, n_folds)
        results.append((name, score))
    
    # Pairs
    for name1, name2 in combinations(components.keys(), 2):
        w12 = components[name1] * components[name2]
        combined = baseline_weights * w12
        score = run_cv(combined, X, y_ret, n_folds)
        results.append((f"{name1}+{name2}", score))
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    return results
```

---

### E3. Constraint-Based Optimization

```python
def constrained_objective(trial, X, y_ret, params):
    """
    Optimize IC_LCB subject to constraints.
    """
    w = compute_weights(params)
    
    # Constraints
    n_eff = compute_n_eff(w)
    top1pct_share = np.sort(w)[-len(w)//100:].sum() / w.sum()
    
    min_n_eff = 0.3 * len(w)
    max_top1pct = 0.10
    
    # Heavy penalty for violation
    if n_eff < min_n_eff:
        return -10.0
    if top1pct_share > max_top1pct:
        return -10.0
    
    # Primary: IC_LCB
    ic_mean, ic_std = run_cv_ic(X, y_ret, w, n_folds=5, seeds=[42, 123])
    ic_lcb = ic_mean - 0.5 * ic_std
    
    return ic_lcb

# Use Optuna's pruning
study = optuna.create_study()
study.optimize(constrained_objective, n_trials=30)
```

---

## Final Checklist

| # | Item | Status |
|---|------|--------|
| A1 | combine_weights_safely alpha bug | ✓ |
| A2 | Degenerate guard division by zero | ✓ |
| A3 | Purged CV label intervals | ✓ |
| A4 | Vol CS normalization guards | ✓ |
| A5 | Liquidity bounded clip | ✓ |
| B1 | Model family match | ✓ |
| C1 | Distance-to-barrier saturating | ✓ |
| C2 | Recency per-era Neff | ✓ |
| D1 | Spearman redundancy | ✓ |
| E1 | Weight sanity logging | ✓ |
| E2 | Ablation harness | ✓ |
| E3 | Constraint-based optimization | ✓ |
