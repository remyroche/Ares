# Training Speed Optimization Suggestions

This document provides suggestions for improving training speed across all ML pipeline steps without implementing the changes (as requested). Each section covers a specific step with actionable recommendations.

---

## Overview

For all steps, the following general optimizations are already available via `src/utils/ml_common/training_efficiency.py`:

1. **WarmStartManager**: Persists best HPO parameters and loads them as starting points for subsequent runs
2. **DynamicSubsampler**: Automatically reduces dataset size for HPO based on total data volume
3. **EfficientTrainingConfig**: Provides optimized configurations based on dataset characteristics

### Quick Integration

```python
from src.utils.ml_common.training_efficiency import (
    WarmStartManager, DynamicSubsampler, get_efficient_training_config,
    apply_warm_start_and_subsampling
)

# One-liner integration
X_sample, y_sample, warm_params, warm_mgr = apply_warm_start_and_subsampling(
    step_name="ml_breakout_bounce_regime_step",
    model_id="ETHUSDT_binance_15m_breakout",
    X=features, y=targets, config=config, for_hpo=True
)
```

---

## Step-Specific Suggestions

### 1. hmm_ml_alpha_step

**Current Bottlenecks:**
- HMM fitting on large datasets
- Multiple LightGBM models
- Quality assessment calculations

**Suggestions:**

| Suggestion | Impact | Effort | Details |
|------------|--------|--------|---------|
| Use diagonal covariance for HMM | 🔴 High | ✅ Easy | Set `covariance_type='diag'` instead of 'full'. Reduces parameters from O(d²) to O(d). |
| Reduce HMM iterations | 🟡 Medium | ✅ Easy | Set `n_iter=50`, `tol=1e-3`. Early convergence is sufficient for regime detection. |
| Cache regime features | 🟡 Medium | ⚠️ Moderate | Store computed regime features in versioned artifacts, reuse for subsequent windows. |
| Use incremental HMM | 🔴 High | 🔧 Complex | Implement partial_fit for HMM updates instead of full refitting. |

---

### 2. hmm_macro_regime

**Current Bottlenecks:**
- 1h HMM with many features
- Quality report generation
- Macro trend calculations

**Suggestions:**

| Suggestion | Impact | Effort | Details |
|------------|--------|--------|---------|
| Pre-compute trend features | 🟡 Medium | ✅ Easy | Cache EWMA, slope calculations. Only update recent bars. |
| Reduce HMM components | 🟡 Medium | ✅ Easy | For macro trends, 3-4 regimes often sufficient vs. 5+. |
| Batch quality metrics | 🟢 Low | ✅ Easy | Compute transition matrix and duration stats in single pass. |
| Parallel regime evaluation | 🟡 Medium | ⚠️ Moderate | Evaluate different regime counts in parallel during HPO. |

---

### 3. ml_reversion_regime_step / ml_mean_reversion_step

**Current Bottlenecks:**
- XGBoost training with many features
- Cross-validation for HPO
- Feature engineering

**Suggestions:**

| Suggestion | Impact | Effort | Details |
|------------|--------|--------|---------|
| Use histogram tree method | 🔴 High | ✅ Easy | `tree_method='hist'` (CPU) or `'gpu_hist'` (GPU). 10-50x faster than exact. |
| Subsample for HPO | 🔴 High | ✅ Easy | Use 10-20% of data for HPO. DynamicSubsampler handles this automatically. |
| Early stopping | 🔴 High | ✅ Easy | Set `early_stopping_rounds=25-35`. Prevents over-training. |
| Feature selection pre-HPO | 🟡 Medium | ⚠️ Moderate | Reduce features to top 100 using mutual information before HPO. |
| Warm start HPO | 🔴 High | ✅ Easy | Already implemented in StandardizedXGBTrainer. Ensure enabled. |

---

### 4. ml_smc_regime_step

**Current Bottlenecks:**
- SMC feature computation (order flow, liquidity)
- XGBoost model training
- Multiple regime classifiers

**Suggestions:**

| Suggestion | Impact | Effort | Details |
|------------|--------|--------|---------|
| Vectorize SMC calculations | 🔴 High | ⚠️ Moderate | Replace loops with numpy/pandas vectorized operations. |
| Cache order flow features | 🟡 Medium | ✅ Easy | Order book features change slowly. Cache with 5-min TTL. |
| Single multi-class model | 🟡 Medium | ⚠️ Moderate | Replace multiple binary classifiers with one multi-class XGBoost. |
| Reduce HPO search space | 🟡 Medium | ✅ Easy | Narrow parameter ranges based on warm start params. |

---

### 5. ml_breakout_bounce_regime_step

**Current Bottlenecks:**
- 2-stage model (classification + regression)
- Support/resistance calculations
- ATR-based filtering

**Suggestions:**

| Suggestion | Impact | Effort | Details |
|------------|--------|--------|---------|
| Single-stage scalar model | 🔴 High | ⚠️ Moderate | Train one model predicting 0-1 scalar directly. Halves training time. |
| Vectorize ATR filtering | 🟡 Medium | ✅ Easy | Pre-compute ATR masks for all bars, avoid per-bar filtering. |
| Cache S/R levels | 🟡 Medium | ✅ Easy | Support/resistance levels change slowly. Cache with 1h TTL. |
| Use StandardizedXGBTrainer | 🔴 High | ✅ Easy | Already implemented. Provides warm start, OOF, and scheduling. |

---

### 6. ml_liquidity_regime_step

**Current Bottlenecks:**
- Order book feature computation
- Spread/volume calculations
- Regime classification

**Suggestions:**

| Suggestion | Impact | Effort | Details |
|------------|--------|--------|---------|
| Downsample order book data | 🔴 High | ✅ Easy | Use 1-second snapshots instead of tick-by-tick. 10-100x less data. |
| Rolling window aggregation | 🟡 Medium | ⚠️ Moderate | Use pandas rolling windows instead of recomputing full history. |
| Binary liquidity classification | 🟡 Medium | ✅ Easy | Simplify from 3+ regimes to binary (liquid/illiquid) for speed. |
| Sparse features | 🟢 Low | ✅ Easy | Order book features are often sparse. Enable sparse matrix support. |

---

### 7. ml_risk_regime_step

**Current Bottlenecks:**
- Risk metric calculations (VaR, volatility)
- Rolling window computations
- Multi-factor analysis

**Suggestions:**

| Suggestion | Impact | Effort | Details |
|------------|--------|--------|---------|
| Exponential smoothing instead of rolling | 🟡 Medium | ✅ Easy | EWM is O(1) per update vs O(n) for rolling windows. |
| Pre-compute volatility regimes | 🟡 Medium | ⚠️ Moderate | Cache volatility regime labels from HMM, don't recompute. |
| Reduce risk factors | 🟢 Low | ✅ Easy | Use top 10 risk factors by importance instead of 20+. |
| GPU-accelerated risk calculations | 🔴 High | 🔧 Complex | Use cuDF/RAPIDS for risk metric computation. 10-50x speedup. |

---

### 8. ml_path_regime_step

**Current Bottlenecks:**
- Path-dependent feature engineering
- Sequential pattern detection
- XGBoost with temporal features

**Suggestions:**

| Suggestion | Impact | Effort | Details |
|------------|--------|--------|---------|
| Fixed-length path encoding | 🟡 Medium | ⚠️ Moderate | Instead of variable-length paths, use fixed k-bar patterns. |
| Convolutional feature extraction | 🔴 High | 🔧 Complex | Replace sequential features with 1D CNN on price series. Much faster. |
| Reduce path lookback | 🟡 Medium | ✅ Easy | Use 48-96 bars instead of 200+ for path patterns. |
| Batch path computation | 🟡 Medium | ⚠️ Moderate | Compute all path features in single vectorized pass. |

---

## General Recommendations

### Hardware Optimizations

1. **GPU Acceleration**
   - XGBoost: `tree_method='gpu_hist'`
   - LightGBM: `device='gpu'`
   - HMM: Use pomegranate with GPU support
   - Expected speedup: 5-50x

2. **Memory Optimization**
   - Use `float32` instead of `float64` (50% memory reduction)
   - Enable sparse matrices for high-sparsity data
   - Use memory-mapped files for large datasets

3. **Parallelism**
   - Set `n_jobs=-1` for sklearn models
   - Use `nthread=-1` for XGBoost
   - For HPO: `optuna.create_study().optimize(..., n_jobs=4)`

### Algorithmic Optimizations

1. **HPO Efficiency**
   - Warm start from previous best params (30-50% trial reduction)
   - Use Successive Halving/Hyperband for early stopping of bad trials
   - Limit search space based on domain knowledge

2. **Training Efficiency**
   - Early stopping with patience=25-35
   - Stratified subsampling for HPO (10-30% of data)
   - Binary dataset format for LightGBM

3. **Caching Strategy**
   - Cache HMM parameters between runs
   - Cache computed features with appropriate TTL
   - Cache OOF predictions for downstream use

### Implementation Priority

| Priority | Optimization | Expected Speedup | Affected Steps |
|----------|--------------|------------------|----------------|
| 1 | Warm start for HPO | 30-50% | All |
| 2 | Dynamic subsampling | 50-80% | All XGB/LGBM steps |
| 3 | Histogram tree method | 10-50x | All XGB steps |
| 4 | Binary dataset format | 80% I/O reduction | All LGBM steps |
| 5 | Diagonal covariance HMM | 5-10x | HMM steps |
| 6 | Feature selection pre-HPO | 50-70% | High-dimensional steps |

---

## Usage Example

```python
# In any ML step's training method:
from src.utils.ml_common.training_efficiency import (
    WarmStartManager,
    DynamicSubsampler,
    get_efficient_training_config
)

def _train_model(self, X, y, config):
    # 1. Get efficient config based on data size
    efficient_config = get_efficient_training_config(
        n_samples=len(X),
        n_features=len(X.columns),
        task_type=config.get('task_type', 'classification')
    )
    
    # 2. Setup warm start
    warm_manager = WarmStartManager(
        model_id=f"{config['symbol']}_{config['timeframe']}_{self.step_name}",
        model_type=self.step_name
    )
    warm_params = warm_manager.load_params() or {}
    
    # 3. Subsample for HPO
    subsampler = DynamicSubsampler()
    X_hpo, y_hpo = subsampler.sample(X, y, stratify=True)
    
    # 4. Run HPO with warm start
    best_params = self._run_hpo(
        X_hpo, y_hpo,
        initial_params=warm_params,
        n_trials=efficient_config.hpo_n_trials,
        timeout=efficient_config.hpo_timeout
    )
    
    # 5. Save best params for next run
    warm_manager.save_params(best_params, metrics=hpo_metrics)
    
    # 6. Train final model on full data
    model = self._train_final_model(X, y, best_params)
    
    return model
```

---

## Monitoring Training Speed

Use the `TrainingSpeedSuggestions` class to get step-specific suggestions:

```python
from src.utils.ml_common.training_efficiency import TrainingSpeedSuggestions

suggestions = TrainingSpeedSuggestions.get_suggestions(
    step_name='ml_breakout_bounce_regime_step',
    n_samples=50000,
    n_features=200,
    model_types=['xgboost']
)

# Print formatted markdown
print(TrainingSpeedSuggestions.format_suggestions(suggestions))
```

---

## Conclusion

The key to faster training is:
1. **Reduce data for HPO** (subsampling + early stopping)
2. **Reuse knowledge** (warm start parameters)
3. **Efficient algorithms** (histogram trees, diagonal covariance)
4. **Hardware utilization** (GPU, parallelism)

All utilities are available in `src/utils/ml_common/training_efficiency.py`. Integration requires minimal code changes to existing steps.
