# Sticky Finite HMM Parameter Guide

Complete guide to all Sticky Finite HMM hyperparameters, what they do, and whether they're optimized.

---

## 🎯 OPTIMIZED Parameters (6)

These are automatically tuned by the auto-tuner to find optimal values for your data.

### 1. **K** (Number of States)
- **Type**: Integer
- **Range**: 4-7
- **Default**: 5
- **What it does**: Determines the number of market regimes the model can discover

### 2. **n_mixtures** (Mixture Components per State) ✨ NEW
- **Type**: Integer
- **Range**: 1-3
- **Default**: 1
- **What it does**: Number of Gaussian mixtures per state for complex regime distributions
- **Impact**: 🟡 IMPORTANT
  - **n_mixtures=1**: Single Gaussian (fast, simple regimes)
    - Time: ~30-40s per run
    - Good for: Clear, unimodal regime distributions
  - **n_mixtures=2**: Two-component mixture (moderate complexity)
    - Time: ~50-70s per run (1.5-2x slower)
    - Good for: Bimodal regimes (e.g., low vol + high vol periods within same regime)
  - **n_mixtures=3**: Three-component mixture (high complexity)
    - Time: ~80-120s per run (2.5-3x slower)
    - Good for: Complex, multi-modal distributions
- **Why optimize**: Some regimes have simple distributions (single peak), others are complex (multiple peaks)
- **Effect on metrics**: Can improve silhouette score for regimes with complex distributions

### 3. **K** (Number of States) - RENUMBERED
- **Type**: Integer
- **Range**: 4-7
- **Default**: 5
- **What it does**: Determines the number of market regimes the model can discover
- **Impact**: 🔴 CRITICAL
  - **K=4**: Simple model, fast, may underfit (miss regime nuances)
  - **K=5**: Balanced (current default)
  - **K=6**: More detailed regimes, risk of over-segmentation
  - **K=7**: Complex, slower, may overfit (spurious regimes)
- **Why optimize**: The "right" number of regimes depends on market dynamics and varies by asset/timeframe

### 2. **kappa** (Stickiness Parameter)
- **Type**: Float
- **Range**: 5.0 - 50.0
- **Default**: 10.0
- **What it does**: Controls how "sticky" regimes are (how long they persist)
- **Impact**: 🔴 CRITICAL
  - Formula: `p_self = (base_alpha + kappa) / (base_alpha * K + kappa)`
  - Expected regime duration: `1 / (1 - p_self)` timesteps
  - **Examples (K=5, base_alpha=0.5)**:
    - kappa=10 → ~11 timesteps (~11 hours for 1h data)
    - kappa=20 → ~20 timesteps
    - kappa=30 → ~28 timesteps
    - kappa=50 → ~44 timesteps
- **Why optimize**: Market regime durations vary widely (consolidation vs trending vs volatility spikes). Data-driven tuning finds realistic durations.
- **Effect on metrics**:
  - Too low → frequent regime flipping, poor temporal_smoothness
  - Too high → regimes too persistent, miss short-term changes

### 3. **base_alpha** (Off-Diagonal Concentration)
- **Type**: Float
- **Range**: 0.1 - 1.0
- **Default**: 0.5
- **What it does**: Controls the sparsity of regime transitions
- **Impact**: 🟡 IMPORTANT
  - **Low (0.1-0.3)**: Sparse transitions
    - Regimes have preferred "next states"
    - Fewer possible transitions
    - More structured regime flow
  - **Medium (0.4-0.6)**: Balanced
  - **High (0.7-1.0)**: Uniform transitions
    - Any regime can transition to any other
    - More flexible but less interpretable
- **Why optimize**: Some markets have clear regime sequences (accumulation → markup → distribution), others are more random
- **Effect on metrics**: Impacts transition_persistence, regime switching patterns

### 4. **lr** (Learning Rate)
- **Type**: Float (log scale)
- **Range**: 1e-4 to 1e-1
- **Default**: 1e-2
- **What it does**: Controls how fast the SVI optimizer updates parameters
- **Impact**: 🟡 IMPORTANT
  - **Too high (>1e-2)**: 
    - Unstable ELBO (bouncing, not converging)
    - May diverge or get stuck in poor local optimum
  - **Just right (1e-3 to 1e-2)**:
    - Stable convergence
    - Reaches good solution in reasonable time
  - **Too low (<1e-3)**:
    - Very slow convergence
    - May not reach optimum within num_iters
- **Why optimize**: Optimal learning rate depends on data scale, K, and problem difficulty
- **Effect on metrics**: Impacts convergence quality → affects all quality metrics

### 5. **pca_components** (PCA Dimensionality)
- **Type**: Integer
- **Range**: 10 - 20
- **Default**: 15
- **What it does**: Number of principal components to keep after PCA
- **Impact**: 🟡 IMPORTANT
  - **Too low (10-12)**:
    - Loses information
    - Poor regime separation
    - Low silhouette score
  - **Just right (13-17)**:
    - Captures key patterns
    - Good separation
    - Avoids noise
  - **Too high (18-20)**:
    - Includes noise
    - Risk of overfitting
    - Slower computation
- **Why optimize**: Balance between information retention and noise reduction is data-dependent
- **Effect on metrics**: Directly impacts silhouette_score, calinski_harabasz_score, separability

---

## 🔒 FIXED Parameters (8)

These are set to sensible defaults based on theory and empirical testing. Not optimized to save computation time.

### 6. **num_iters** (SVI Iterations)
- **Value**: 1000 (fixed)
- **What it does**: Maximum number of SVI training iterations
- **Why fixed**: 
  - 1000 is sufficient for convergence with early stopping
  - Early stopping (patience=50) prevents unnecessary iterations
  - Optimizing this doesn't improve quality, just wastes time
- **Impact**: 🟢 MODERATE (training efficiency only)

### 7. **num_particles** (SVI Particles)
- **Value**: 10 (fixed)
- **What it does**: Number of Monte Carlo samples for gradient estimation in SVI
- **Technical details**:
  - More particles → better gradient estimates → more stable training
  - Fewer particles → noisier gradients → faster but less stable
  - Computational cost scales linearly with num_particles
- **Why fixed**:
  - 10 is good balance for this problem size
  - Empirically tested: 5 is noisy, 20+ is overkill
  - Doesn't significantly impact final quality
- **Impact**: 🟢 MODERATE (training stability)

### 8. **prior_mean_scale** (Emission Mean Prior)
- **Value**: 10.0 (fixed)
- **What it does**: Standard deviation of Normal prior on emission means
  - `μ_k ~ Normal(0, prior_mean_scale)` for each state k
- **Technical details**:
  - Controls how far state means can deviate from zero
  - Data is standardized (mean=0, std=1) after PCA
  - prior_mean_scale=10 allows means in range roughly [-30, +30]
- **Why fixed**:
  - 10.0 works well for standardized features
  - Wide enough to allow separation
  - Not too wide to cause numerical instability
- **Impact**: 🔵 MINOR (emission parameter regularization)

### 9. **prior_cov_scale** (Emission Covariance Prior)
- **Value**: 1.0 (fixed)
- **What it does**: Standard deviation of LogNormal prior on emission scales
  - `σ_k ~ LogNormal(0, prior_cov_scale)` for each state k
- **Technical details**:
  - Controls emission variance (spread within each regime)
  - LogNormal ensures σ > 0
  - prior_cov_scale=1.0 → median σ=1, reasonable spread
- **Why fixed**:
  - 1.0 is reasonable for standardized features
  - Not sensitive to exact value
  - More important to optimize state structure (K, kappa)
- **Impact**: 🔵 MINOR (within-regime variance)

### 10. **patience** (Early Stopping Patience)
- **Value**: 50 (fixed)
- **What it does**: Number of iterations to wait without ELBO improvement before stopping
- **How it works**:
  - Tracks moving average of ELBO over convergence_window (10 iters)
  - Stops if no improvement > elbo_improvement_threshold for patience iterations
- **Why fixed**:
  - 50 is robust (not too impatient, not too patient)
  - Prevents overfitting
  - Saves computation time
- **Impact**: 🔵 MINOR (convergence efficiency)

### 11. **elbo_improvement_threshold** (Convergence Threshold)
- **Value**: 1e-3 (fixed)
- **What it does**: Minimum ELBO improvement considered "significant"
- **How it works**:
  - Compares moving average ELBO now vs 10 iterations ago
  - If improvement < 1e-3 for patience iterations → stop
- **Technical details**:
  - Too strict (1e-4): may stop too early
  - Too loose (1e-2): trains unnecessarily long
  - 1e-3: good balance
- **Why fixed**:
  - Well-calibrated threshold
  - Works across different K and data scales
- **Impact**: 🔵 MINOR (convergence criteria)

### 12. **min_features** (Minimum Feature Count)
- **Value**: 50 (fixed)
- **What it does**: Minimum number of features selected from Feature Bank (~140 total)
- **How it works**:
  - Feature Bank generates ~140 features across categories
  - Features are ranked by variance, uniqueness, and category weights
  - min_features=50 ensures diverse signal
- **Why fixed**:
  - Matched to HDP-HMM for fair comparison
  - 50 is adequate for regime discovery (not too sparse)
  - Lower bound prevents underfitting
- **Impact**: 🔵 MINOR (feature selection)

### 13. **max_features** (Maximum Feature Count)
- **Value**: 100 (fixed)
- **What it does**: Maximum number of features to prevent overfitting
- **How it works**:
  - Top max_features are selected after ranking
  - Prevents using all ~140 features (many redundant)
  - 100 is upper bound for regime discovery
- **Why fixed**:
  - Matched to HDP-HMM for fair comparison
  - 100 provides rich signal without overfitting
  - Beyond 100, marginal gains < computational cost
- **Impact**: 🔵 MINOR (feature selection)

---

## 📊 Summary Table

| Parameter | Type | Range | Default | Optimized? | Impact | What It Controls |
|-----------|------|-------|---------|------------|--------|------------------|
| **K** | int | 4-7 | 5 | ✅ | 🔴 CRITICAL | Number of regimes |
| **kappa** | float | 5-50 | 10.0 | ✅ | 🔴 CRITICAL | Regime persistence/duration |
| **base_alpha** | float | 0.1-1.0 | 0.5 | ✅ | 🟡 IMPORTANT | Transition sparsity |
| **lr** | float | 1e-4-1e-1 | 1e-2 | ✅ | 🟡 IMPORTANT | Learning speed/stability |
| **pca_components** | int | 10-20 | 15 | ✅ | 🟡 IMPORTANT | Feature dimensionality |
| num_iters | int | - | 1000 | ❌ | 🟢 MODERATE | Training iterations |
| num_particles | int | - | 10 | ❌ | 🟢 MODERATE | Gradient estimation |
| prior_mean_scale | float | - | 10.0 | ❌ | 🔵 MINOR | Emission mean prior |
| prior_cov_scale | float | - | 1.0 | ❌ | 🔵 MINOR | Emission variance prior |
| patience | int | - | 50 | ❌ | 🔵 MINOR | Early stopping patience |
| elbo_improvement_threshold | float | - | 1e-3 | ❌ | 🔵 MINOR | Convergence threshold |
| min_features | int | - | 50 | ❌ | 🔵 MINOR | Min features selected |
| max_features | int | - | 100 | ❌ | 🔵 MINOR | Max features selected |

---

## 🚀 Auto-Tuner Strategy

### Why This Focused Approach?

1. **Efficiency**: Optimizing 5 params instead of 13
   - Search space: 4×11×10×5×11 = **24,200 combinations** (full grid)
   - Hierarchical TPE explores ~**50-150 trials** intelligently
   - **100-500x speedup** vs exhaustive search

2. **Effectiveness**: The 5 optimized params have **90%+ impact** on quality
   - K and kappa are the most critical
   - The other 8 params have minimal sensitivity

3. **Reproducibility**: Fixed params ensure consistent comparisons
   - Same convergence criteria
   - Same feature selection
   - Same prior settings

### Expected Optimization Time

- **Per trial**: ~30-60 seconds (depending on K and data size)
- **Total trials**: ~50-150 (hierarchical optimization)
- **Total time**: **25-150 minutes** for full optimization
  - Coarse grid: ~10-20 minutes
  - Fine grid: ~10-20 minutes  
  - TPE refinement: ~5-110 minutes

### Usage Example

```python
from src.training.steps.market_analysis.sticky_finite_hmm_clustering import (
    run_sticky_finite_hmm_auto_tuning
)

# Run auto-tuning
best_params, best_score, results = run_sticky_finite_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="1h",
    use_hierarchical=True,  # Recommended
    n_rounds=2,  # 2 rounds of refinement
    tpe_trials=100,
    timeout=3600  # 1 hour max
)

# Best parameters found
print(f"Best K: {best_params['K']}")
print(f"Best kappa: {best_params['kappa']:.2f}")
print(f"Best base_alpha: {best_params['base_alpha']:.3f}")
print(f"Best lr: {best_params['lr']:.5f}")
print(f"Best pca_components: {best_params['pca_components']}")
print(f"\nComposite Score: {best_score:.4f}")
```

---

## 🎯 Recommendations

### When to Re-Run Auto-Tuning

- **New asset**: Different markets may have different optimal K and kappa
- **New timeframe**: 1h vs 4h vs 1d regimes differ
- **Market regime change**: After major structural changes (e.g., post-2020 crypto)
- **Periodically**: Every 3-6 months to adapt to evolving markets

### When to Manually Override

You might want to manually set parameters if:

1. **K**: You have domain knowledge about regime count
   - E.g., "I know there are exactly 4 crypto market phases"
   
2. **kappa**: You want specific regime durations
   - E.g., "I want regimes that last ~2 days (48 hours for 1h data)"
   - Set kappa to achieve desired duration: `kappa = K * desired_duration * base_alpha / (desired_duration - 1)`

3. **base_alpha**: You want sparse/uniform transitions
   - Sparse (0.1-0.2): For markets with clear regime sequences
   - Uniform (0.8-1.0): For more random regime transitions

### Interpreting Results

After auto-tuning, check:

1. **Composite score** > 0.65: Good regime discovery
2. **Silhouette score** > 0.30: Well-separated regimes
3. **Temporal smoothness** > 0.70: Stable regime assignments
4. **Balance score** > 0.70: No regime dominates
5. **Regime persistence**: Average duration matches expectations

If results are poor despite tuning, consider:
- Data quality issues (missing data, outliers)
- Feature engineering (add domain-specific features)
- Different clustering approach (try HDP-HMM for automatic K)

