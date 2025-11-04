# Sticky Finite HMM Implementation Notes

## Overview

This module implements an **Enhanced Sticky Finite HMM** with fixed K states using Variational Bayes (Pyro + PyTorch) as an alternative to the nonparametric HDP-HMM.

### ✨ Recent Enhancements (Nov 2025)

1. **Gaussian Mixture Emissions** (n_mixtures=1-3)
   - Single Gaussian (fast, ~30-40s)
   - 2-component mixtures (moderate, ~50-70s, captures bimodal regimes)
   - 3-component mixtures (slow, ~80-120s, captures complex distributions)

2. **Multi-Timeframe (MTF) Regime Features**
   - Incorporates 4h and 1d regime context
   - 5 features per timeframe (regime, volatility, trend, strength, alignment)
   - ~10 additional features for regime context

3. **Multi-Objective Optimization**
   - Pareto front construction
   - Optimizes: composite_score, silhouette, temporal_smoothness, balance, economic_sharpe
   - Returns non-dominated solutions

4. **Economic Quality Metrics**
   - Sharpe ratio per regime
   - Returns and volatility per regime
   - Max drawdown per regime
   - Win rate and profit factor
   - Integrated into quality assessment

5. **Auto-Tuner Enhancements**
   - Now optimizes 6 parameters (added n_mixtures)
   - Multi-objective mode available
   - Pareto front analysis

## Key Design Decisions

### 1. Feature Generation - Enhanced with MTF Context ✅

**Goal**: Ensure Sticky Finite HMM uses comprehensive features with multi-timeframe context.

**Implementation**:
- Uses same `FeatureBankConfig` as HDP-HMM
- Feature weights: Volatility: 0.30, Trend: 0.25, Momentum: 0.20, Volume: 0.12, Microstructure: 0.08, Clustering: 0.05
- `min_features` (50) and `max_features` (100) range
- **NEW**: Multi-timeframe regime features (4h, 1d context)
  - Volatility regime classification (low/medium/high) using quantiles (0.33, 0.67)
  - Trend regime classification (down/sideways/up)
  - Regime alignment indicators
  - ~10 additional MTF features
- Same regime categorization and integration if available
- Same data loading from artifacts

**Files Updated**:
- `enhanced_sticky_finite_hmm_clustering_integration.py` - Lines 157-180, 314-413 (MTF method added)

### 2. PCA Components - Enhanced for Fixed K Model ✅

**Goal**: Use 15-20 PCA components (vs 10 for HDP-HMM) for better regime separation with fixed K.

**Rationale**: 
- HDP-HMM can discover optimal K nonparametrically, so 10 components suffice
- Sticky Finite HMM has fixed K=5, so more components (15-20) help differentiate regimes better

**Implementation**:
- **Default**: 15 components (was 10)
- **Range**: Can use up to 20 components
- **Fallback**: Uses `pca_variance_threshold: float = 0.95` if specified

**Files Updated**:
- `sticky_finite_hmm_clusterer.py` - Line 105: `pca_components: int = 15`
- `sticky_finite_hmm_regime_discovery_step.py` - Line 320: `pca_components = params.get('pca_components', 15)`
- `standalone_runner.py` - Line 87: `pca_components: int = 15`
- `enhanced_sticky_finite_hmm_clustering_integration.py` - Line 95: `pca_components: int = 15`
- `create_sticky_finite_hmm_clusterer()` - Line 972: default to 15

### 3. Bug Fixes ✅

#### 3.1. Error Messages Fixed
**Issue**: Error messages incorrectly said "HDP-HMM" instead of "Sticky Finite HMM"

**Fixed in** `sticky_finite_hmm_clusterer.py`:
- Line 402: "Sticky Finite HMM requires substantial data..."
- Line 409: "Sticky Finite HMM requires multiple features..."

#### 3.2. Validation Enhancements
**Added**:
- Degenerate case detection (all values identical)
- Low variance feature warnings
- More informative error messages with actionable suggestions

#### 3.3. Artifact Naming Consistency
**Fixed in** `sticky_finite_hmm_regime_discovery_step.py`:
- Added both `hdp_hmm_*` and `sticky_finite_hmm_*` artifact names for compatibility
- Ensures downstream components can find regime labels under expected names

## Data Flow Consistency

### Input Data Sources (Same as HDP-HMM)
1. `klines_downloading_processing` / `klines_data`
2. `data_collection` / `market_data`
3. `data_reading` / `ohlcv_data`

### Feature Generation Pipeline (Same as HDP-HMM)
1. Load market data → 2. Generate 50-100 features → 3. Apply PCA (15 components) → 4. Run Sticky Finite HMM

### Output Artifacts (Compatible with HDP-HMM)

**Parquet Artifacts** (via artifact_manager):
- `hdp_hmm_regime_labels` (primary) + `sticky_finite_hmm_regime_labels` (secondary)
- `hdp_hmm_regime_probabilities` + `sticky_finite_hmm_regime_probabilities`
- `hdp_hmm_transition_matrix` + `sticky_finite_hmm_transition_matrix`
- `hdp_hmm_quality_metrics`
- `hdp_hmm_cluster_statistics` + `sticky_finite_hmm_cluster_statistics`
- `hdp_hmm_emission_params`
- `hdp_hmm_features_used`

**CSV Reports** (in `outcomes/sticky_finite_hmm_clustering/{symbol}/{exchange}/{timeframe}/`):

**ONE comprehensive CSV** with all metrics:
1. `sticky_finite_hmm_all_results_{timestamp}.csv` - **Complete metrics in one file** (matches HDP-HMM + enriched)
   - **Core metrics**: composite_score, K, base_alpha, kappa, num_iters, lr, n_clusters
   - **Quality metrics**: silhouette_score, davies_bouldin_score, calinski_harabasz_score, balance_score
   - **CV metrics**: within_regime_cv, within_regime_cv_std, between_regime_cv, between_regime_cv_std, cv_ratio, economic_cv_ratio
   - **Temporal metrics**: temporal_smoothness, temporal_smoothness_raw, flip_flop_ratio, regime_persistence_bars, transition_persistence
   - **Duration distribution**: duration_mean, duration_median, duration_std, duration_min, duration_max
   - **Balance metrics**: min_cluster_size_pct, max_cluster_size_pct, cluster_size_std
   - **Per-category CV ratios**: cv_order_flow, cv_microstructure, cv_momentum, cv_volatility, cv_volume, cv_trend, cv_temporal
   - **Per-regime metrics** (19 columns × 5 regimes = 95 columns):
     - Basic: size, size_pct, duration_mean, duration_std, silhouette_mean, silhouette_std
     - Economic (13): mean_return, volatility, sharpe, skewness, max_drawdown, pct_above_target, pct_below_neg_target, pct_target_hits, risk_adj_target_hits, win_rate, return_per_vol, profit_factor
   - **Runtime/convergence**: runtime, memory_usage_mb, converged, convergence_iteration, final_elbo
   - **Status**: success, error

**Time series and matrices** (separate for structure):
2. `elbo_history_{timestamp}.csv` - ELBO convergence (iteration, elbo, elbo_ma, elbo_improvement)
3. `transition_matrix_{timestamp}.csv` - 5x5 transition matrix with labeled rows/columns

**Markdown Report**:
- `cluster_quality_report_{symbol}_StickyFiniteHMM_{timestamp}.md` - Comprehensive quality assessment report

## Comparison: HDP-HMM vs Sticky Finite HMM

| Aspect | HDP-HMM | Sticky Finite HMM |
|--------|---------|-------------------|
| **Number of States** | Nonparametric (inferred from data) | Fixed K=5 |
| **Inference** | Gibbs sampling | Variational Bayes (SVI) |
| **Library** | pyhsmm/ssm | Pyro + PyTorch |
| **Features** | 50-100 from feature bank | ✅ **Same** 50-100 features |
| **PCA Components** | 10 | ✅ **15** (can use up to 20) |
| **Stickiness** | kappa parameter in model | kappa parameter in Dirichlet prior |
| **Speed** | Slower (Gibbs sampling) | ✅ **Faster** (VB/SVI) |
| **Convergence** | Monitor state count stability | Monitor ELBO improvement |
| **Initialization** | Random or KMeans | ✅ **KMeans** (warm start) |
| **Data Sources** | Same artifact sources | ✅ **Same** artifact sources |
| **Output Format** | regime_labels, probabilities, etc. | ✅ **Same** format + ELBO history |
| **CSV Reports** | None (only tuning) | ✅ **3 comprehensive CSVs** |
| **Markdown Report** | None | ✅ **Full quality report** |

### CSV Report Details

Sticky Finite HMM generates **3 comprehensive CSV files**:

1. **sticky_finite_hmm_all_results.csv**: **ONE LARGE FILE with ALL metrics** (based on HDP-HMM tuning format + enriched)
   - **Core**: composite_score, K, base_alpha, kappa, num_iters, lr, n_clusters
   - **Quality**: silhouette, DBI, CH, balance
   - **CV**: within/between/ratio (overall + std dev), economic_cv_ratio  
   - **Temporal**: smoothness (raw + penalized), flip-flop ratio, regime persistence, transition persistence
   - **Duration**: mean, median, std, min, max
   - **Balance**: min/max cluster size %, cluster size std
   - **Per-category CV**: order_flow, microstructure, momentum, volatility, volume, trend, temporal
   - **Per-regime columns**: regime_0_size, regime_0_size_pct, regime_0_duration_mean, regime_0_silhouette_mean, regime_0_mean_return, regime_0_sharpe_ratio, regime_1_..., etc. (all 5 regimes)
   - **Runtime**: runtime, memory, converged, convergence_iteration, final_elbo
   
2. **elbo_history.csv**: Time series (iteration, elbo, elbo_ma, elbo_improvement)

3. **transition_matrix.csv**: 5x5 matrix with labeled rows/columns

## Usage Examples

### Basic Usage
```python
from src.training.steps.market_analysis.sticky_finite_hmm_clustering import run_sticky_finite_hmm_step

config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'sticky_finite_hmm_params': {
        'K': 5,
        'base_alpha': 0.5,
        'kappa': 10.0,  # Moderate stickiness
        'num_iters': 800,
        'lr': 1e-2,
        'pca_components': 15  # Can use up to 20
    }
}

results = await run_sticky_finite_hmm_step(config)
```

### Expected Regime Duration
```python
# Calculate expected dwell time
K = 5
base_alpha = 0.5
kappa = 10.0

p_self = (base_alpha + kappa) / (base_alpha * K + kappa)
expected_duration = 1.0 / (1.0 - p_self)

# For kappa=10, base_alpha=0.5, K=5:
# p_self ≈ 0.82, expected_duration ≈ 5.5 timesteps
```

### Tuning Kappa
- **Short regimes** (1-3 timesteps): kappa ∈ [0, 2]
- **Moderate regimes** (5-50 timesteps): kappa ∈ [5, 20]
- **Long regimes** (>50 timesteps): kappa > 20

## Testing

Run the test suite:
```bash
poetry run python src/training/steps/market_analysis/sticky_finite_hmm_clustering/test_sticky_finite_hmm.py
```

**Tests included**:
1. Basic instantiation
2. Clustering and ELBO convergence
3. Transition matrix properties (stochastic, stickiness)
4. Quality metrics computation
5. Prediction on new data

## Dependencies

```toml
pyro-ppl = "^1.8.0"
torch = "^2.0.0"
```

Installed via:
```bash
poetry lock
poetry install
```

## Quality Assurance

- ✅ No linter errors
- ✅ Uses same feature generation as HDP-HMM
- ✅ PCA supports 15-20 components
- ✅ Error messages are correct
- ✅ Artifact naming is consistent
- ✅ Data sources are identical to HDP-HMM
- ✅ Comprehensive validation and warnings
- ✅ All docstrings updated

## Known Limitations

1. **Fixed K**: Must choose K=5 in advance (not data-driven)
2. **VI Approximation**: Variational inference is an approximation (vs exact Gibbs)
3. **ELBO Noise**: ELBO can be noisy; use moving average for monitoring
4. **Test Import Error**: Pre-existing `DatasetCharacteristics` error in HDBSCAN module (not related to Sticky Finite HMM)

## Future Enhancements

- [ ] Hyperparameter optimization (grid search over base_alpha, kappa)
- [ ] Model selection for K (cross-validation, marginal likelihood)
- [ ] Full covariance emissions (currently diagonal)
- [ ] Regime interpretation helpers
- [ ] Batching strategy for very long sequences (>10k timesteps)

## References

- Original specification: User-provided Pyro + PyTorch implementation guide
- HDP-HMM comparison: `hdp_hmm_clustering/` module
- Feature generation: `feature_generation/integration/` module
- Quality assessment: `clusters/cluster_quality_assessor.py`

