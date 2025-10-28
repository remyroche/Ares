# Regime Clustering Alternatives Guide

## Overview

This guide documents two alternative regime clustering approaches implemented as alternatives to HDBSCAN clustering:

1. **HDP-HMM (Hierarchical Dirichlet Process Hidden Markov Model)** - Nonparametric Bayesian approach
2. **MS-DR (Markov-Switching Dynamic Regression)** - Switching state-space models

## Why These Alternatives?

The current HDBSCAN clustering approach has limitations:
- Fixed number of clusters requires manual tuning
- Doesn't naturally model temporal dependencies
- No explicit transition modeling between regimes
- Noise points can be problematic for regime persistence

These alternatives address these issues:
- **HDP-HMM**: Automatically infers number of regimes, handles temporal dependencies
- **MS-DR**: Explicitly models regime transitions and dynamics

---

## 1. HDP-HMM Clustering

### Overview

Hierarchical Dirichlet Process Hidden Markov Model with sticky parameter for regime persistence.

### Key Features

- **Nonparametric**: Automatically infers number of regimes from data (no need to specify K)
- **Temporal Dependencies**: Natural handling of time-series structure
- **Regime Persistence**: "Sticky" parameter encourages regimes to persist
- **Bayesian Framework**: Provides uncertainty quantification
- **Flexible**: Can discover 4-8+ regimes if they exist in data

### Parameters

```python
HDPHMMConfig(
    alpha=3.0,      # Concentration for regime diversity (higher = more regimes)
    kappa=50.0,     # Stickiness parameter (higher = longer regime durations)
    gamma=3.0,      # Hyperparameter for base distribution
    n_iterations=100,  # Gibbs sampling iterations
    max_states=20,  # Maximum number of states to consider
    enable_pca=True,
    pca_components=10
)
```

**Parameter Tuning Guide:**
- `alpha` (1-10): Controls regime diversity
  - Low (1-3): Fewer, more stable regimes
  - Medium (3-5): Balanced
  - High (5-10): More granular regimes
  
- `kappa` (10-100): Controls regime persistence
  - Low (10-30): Frequent switches
  - Medium (30-70): Balanced
  - High (70-100): Persistent regimes

### Libraries Required

```bash
# Primary (recommended):
pip install Cython numpy scipy matplotlib
pip install git+https://github.com/mattjj/pyhsmm.git

# Alternative:
pip install ssm-jax
```

### Usage Example

```python
from src.training.steps.market_analysis.hdp_hmm_clustering import (
    HDPHMMClusterer, HDPHMMConfig
)

# Create configuration
config = HDPHMMConfig(
    alpha=3.0,
    kappa=50.0,
    gamma=3.0,
    n_iterations=100,
    max_states=20
)

# Create clusterer
clusterer = HDPHMMClusterer(config)

# Fit and predict
result = clusterer.fit_predict(market_data_features)

print(f"Discovered regimes: {result.n_clusters}")
print(f"Silhouette score: {result.silhouette_score:.3f}")
print(f"Transition persistence: {result.transition_persistence:.3f}")
```

### Integration with Feature Bank

```python
from src.feature_generation.integration.enhanced_hdp_hmm_clustering_integration import (
    EnhancedHDPHMMClusteringIntegration
)

# Create integration
integration = EnhancedHDPHMMClusteringIntegration(
    min_features=50,
    max_features=100,
    alpha=3.0,
    kappa=50.0,
    n_iterations=100
)

# Cluster with comprehensive features
result = integration.cluster_with_hdp_hmm(market_data)
```

### Advantages

✅ Automatic regime number selection  
✅ Natural temporal dependency modeling  
✅ Regime persistence built-in  
✅ Bayesian uncertainty quantification  
✅ No noise points  
✅ Flexible - adapts to data complexity  

### Disadvantages

⚠️ Computationally expensive (Gibbs sampling)  
⚠️ Requires more samples for convergence  
⚠️ Library installation can be complex (pyhsmm)  
⚠️ Less interpretable than simpler methods  

### When to Use

- When you don't know the number of regimes
- When temporal dependencies are important
- When you want regime persistence
- When you have sufficient data (500+ samples)
- When computational cost is acceptable

---

## 2. MS-DR (Markov-Switching Dynamic Regression)

### Overview

Markov-Switching models that explicitly model regime-dependent dynamics and transitions.

### Key Features

- **Explicit Transitions**: Models transition probabilities explicitly
- **Regime-Dependent Dynamics**: Different AR/regression parameters per regime
- **Heteroskedasticity**: Handles variance switching across regimes
- **Model Selection**: Automatic regime selection using AIC/BIC
- **Economic Interpretability**: Clear regime characteristics

### Parameters

```python
MSDRConfig(
    n_regimes=5,                # Number of regimes (if not auto-selecting)
    switching_variance=True,     # Allow variance to switch
    switching_trend=True,        # Allow trend to switch
    model_type='autoregression', # 'autoregression' or 'regression'
    order=1,                     # AR order
    auto_select_regimes=True,    # Auto-select using IC
    min_regimes=2,
    max_regimes=10,
    ic_criterion='bic'           # 'aic', 'bic', or 'hqic'
)
```

**Parameter Tuning Guide:**
- `n_regimes`: Number of regimes (2-10 typical)
- `order`: Autoregression order (1-3 typical)
- `ic_criterion`: Information criterion for model selection
  - AIC: More flexible, may overfit
  - BIC: More conservative, penalizes complexity
  - HQIC: Between AIC and BIC

### Libraries Required

```bash
pip install statsmodels>=0.13.0
```

### Usage Example

```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    MSDRClusterer, MSDRConfig
)

# Create configuration
config = MSDRConfig(
    n_regimes=5,
    model_type='autoregression',
    order=1,
    switching_variance=True,
    auto_select_regimes=True,
    min_regimes=2,
    max_regimes=10
)

# Create clusterer
clusterer = MSDRClusterer(config)

# Fit and predict
result = clusterer.fit_predict(market_data_features)

print(f"Discovered regimes: {result.n_clusters}")
print(f"AIC: {result.aic:.2f}")
print(f"BIC: {result.bic:.2f}")
print(f"Transition persistence: {result.transition_persistence:.3f}")
```

### Integration with Feature Bank

```python
from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
    EnhancedMSDRClusteringIntegration
)

# Create integration
integration = EnhancedMSDRClusteringIntegration(
    min_features=50,
    max_features=100,
    n_regimes=5,
    auto_select_regimes=True
)

# Cluster with comprehensive features
result = integration.cluster_with_ms_dr(market_data)
```

### Advantages

✅ Explicit transition modeling  
✅ Economic interpretability  
✅ Model selection via IC  
✅ Handles heteroskedasticity  
✅ Fast fitting (EM algorithm)  
✅ Well-established library (statsmodels)  
✅ Easy installation  

### Disadvantages

⚠️ Requires specification of K (if not auto-selecting)  
⚠️ Can be sensitive to initialization  
⚠️ Assumes Markovian dynamics  
⚠️ May struggle with many features (uses PCA)  

### When to Use

- When you want explicit transition probabilities
- When economic interpretability is important
- When computational efficiency matters
- When you have moderate data (200+ samples)
- When you want model selection via IC

---

## Comparison Table

| Feature | HDBSCAN | HDP-HMM | MS-DR |
|---------|---------|---------|-------|
| **Automatic K Selection** | ❌ | ✅ | ✅ (via IC) |
| **Temporal Dependencies** | ❌ | ✅ | ✅ |
| **Transition Modeling** | ❌ | ✅ | ✅ |
| **Regime Persistence** | ⚠️ | ✅ | ✅ |
| **Noise Handling** | ✅ | N/A | N/A |
| **Computational Cost** | Medium | High | Low-Medium |
| **Interpretability** | Medium | Low | High |
| **Installation Ease** | Easy | Hard | Easy |
| **Data Requirements** | 300+ | 500+ | 200+ |

---

## Choosing the Right Method

### Use **HDBSCAN** when:
- You need density-based clustering
- You want noise point detection
- You have spatial/geometric features
- Temporal dependencies are not critical

### Use **HDP-HMM** when:
- You don't know the number of regimes
- Temporal dependencies are critical
- You want Bayesian uncertainty quantification
- You have sufficient computational resources
- You have 500+ data points

### Use **MS-DR** when:
- You want explicit transition probabilities
- Economic interpretability is important
- You need fast clustering
- Model selection via IC is desired
- You have 200+ data points

---

## Testing

### Test HDP-HMM

```bash
python minimal_test_hdp_hmm.py
```

### Test MS-DR

```bash
python minimal_test_ms_dr.py
```

---

## File Structure

```
src/training/steps/market_analysis/
├── hdp_hmm_clustering/
│   ├── __init__.py
│   └── hdp_hmm_clusterer.py          # HDP-HMM implementation
├── ms_dr_clustering/
│   ├── __init__.py
│   └── ms_dr_clusterer.py            # MS-DR implementation
└── hdbscan_clustering/
    └── ...                            # Original HDBSCAN implementation

src/feature_generation/integration/
├── enhanced_hdp_hmm_clustering_integration.py
├── enhanced_ms_dr_clustering_integration.py
└── enhanced_hdbscan_clustering_integration.py

# Test scripts
minimal_test_hdp_hmm.py
minimal_test_ms_dr.py
```

---

## Implementation Details

### HDP-HMM Implementation

**Core Algorithm:**
1. Sticky HDP-HMM with Gibbs sampling
2. Nonparametric - infers K from data
3. Sticky parameter for regime persistence
4. Gaussian observation model

**Key Classes:**
- `HDPHMMClusterer`: Main clustering class
- `HDPHMMConfig`: Configuration dataclass
- `HDPHMMResult`: Result container

### MS-DR Implementation

**Core Algorithm:**
1. Markov-Switching Autoregression/Regression
2. EM algorithm for parameter estimation
3. Model selection via AIC/BIC
4. Regime-dependent variance (optional)

**Key Classes:**
- `MSDRClusterer`: Main clustering class
- `MSDRConfig`: Configuration dataclass
- `MSDRResult`: Result container

---

## Future Enhancements

### Planned Improvements

1. **Online/Streaming Variants**
   - Real-time regime detection
   - Incremental updates

2. **Hybrid Approaches**
   - Combine HDP-HMM with MS-DR
   - Ensemble methods

3. **Advanced Features**
   - Hierarchical regime structure
   - Multi-scale regime analysis
   - Regime-specific feature selection

4. **Performance Optimizations**
   - Parallelized Gibbs sampling
   - GPU acceleration
   - Caching and memoization

---

## References

### HDP-HMM
- Fox, E. B., et al. (2011). "Sticky HDP-HMM: Bayesian nonparametric hidden Markov models with persistent states"
- pyhsmm library: https://github.com/mattjj/pyhsmm

### MS-DR
- Hamilton, J. D. (1989). "A new approach to the economic analysis of nonstationary time series"
- Kim, C. J., & Nelson, C. R. (1999). "State-space models with regime switching"
- statsmodels library: https://www.statsmodels.org/

---

## Troubleshooting

### HDP-HMM Issues

**Problem**: `pyhsmm` installation fails  
**Solution**: Install dependencies first:
```bash
pip install Cython numpy scipy matplotlib
pip install git+https://github.com/mattjj/pyhsmm.git
```

**Problem**: Gibbs sampling too slow  
**Solution**: Reduce `n_iterations` or use `ssm` library instead

**Problem**: Too many/few regimes discovered  
**Solution**: Adjust `alpha` parameter (higher = more regimes)

### MS-DR Issues

**Problem**: Model doesn't converge  
**Solution**: 
- Try different optimization method (`powell`, `bfgs`, `nm`)
- Reduce number of regimes
- Increase `max_iter`

**Problem**: Auto-selection picks wrong K  
**Solution**:
- Try different IC criterion (BIC vs AIC)
- Manually specify `n_regimes`

---

## Contributing

To add new clustering methods:

1. Create new module in `src/training/steps/market_analysis/`
2. Implement clusterer class with `fit_predict` method
3. Create config and result dataclasses
4. Add integration in `src/feature_generation/integration/`
5. Create minimal test script
6. Update this guide

---

## License

Same as parent project.

---

**Last Updated**: 2025-10-28  
**Version**: 1.0.0
