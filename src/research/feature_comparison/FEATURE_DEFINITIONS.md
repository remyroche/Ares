# Standardized Feature Definitions

This document provides comprehensive definitions for all standardized features used in the feature comparison framework.

## Naming Conventions

### Core Definitions
- `ret_t(h) = log(P_t / P_{t-h})` - Log return over h periods
- `vwap_t = Σ(P_i * V_i) / Σ(V_i)` over window W - Volume-weighted average price
- `vol_t(W) = std(ret_t)` over window W - Realized volatility proxy

### Suffixes
- `_wW` → Rolling window W (e.g., `ret_ma_w20` = rolling mean over 20 periods)
- `_ewmA` → EWMA with span A (e.g., `ret_ewm_w20` = EWMA with span 20)
- `_normvolW` → Divided by vol_t(W) (e.g., `ret_ma_w5_normvol20`)
- `_zcs` → Cross-sectional z-score at time t
- `_leadH` / `_lagH` → H-step lead/lag (e.g., `ret_lag1`, `ret_lead2`)

## Feature Categories

### 1. Core Returns Features
| Feature | Definition | Usage |
|---------|------------|-------|
| `ret_t1` | Log return: log(P_t / P_{t-1}) | Primary return feature |
| `ret_abs_t1` | Absolute return: \|ret_t1\| | Volatility proxy |
| `ret_sq_t1` | Squared return: ret_t1² | Variance proxy, more common in models |

### 2. Rolling Features
| Feature | Definition | Windows |
|---------|------------|---------|
| `ret_ma_wW` | Rolling mean of returns over window W | W ∈ {5, 10, 20, 50} |
| `ret_std_wW` | Rolling std of returns over window W | W ∈ {5, 10, 20, 50} |
| `ret_ewm_wW` | EWMA of returns with span W | W ∈ {5, 10, 20, 50} |
| `ret_skew_wW` | Rolling skewness over window W | W ∈ {5, 10, 20, 50} |
| `ret_kurt_wW` | Rolling kurtosis over window W | W ∈ {5, 10, 20, 50} |

### 3. Lagged and Lead Features
| Feature | Definition | Periods |
|---------|------------|---------|
| `ret_lagH` | H-step lagged return | H ∈ {1, 2, 3, 5, 10} |
| `ret_leadH` | H-step lead return | H ∈ {1, 2, 3, 5} |
| `ret_mom_kK` | Momentum: cumulative return over K periods | K ∈ {1, 2, 3, 5} |
| `ret_rsi_kK` | RSI-style momentum over K periods | K ∈ {1, 2, 3, 5} |
| `ret_acc_k1` | Acceleration: Δ momentum | First difference |

### 4. VWAP Features
| Feature | Definition | Windows |
|---------|------------|---------|
| `vwap_wW` | VWAP over window W | W ∈ {10, 20, 50} |
| `vwap_ret_wW` | VWAP return over window W | W ∈ {10, 20, 50} |
| `vwap_basis_wW` | VWAP basis: (price - vwap) | W ∈ {10, 20, 50} |
| `rel_vwap_dev_wW` | Relative VWAP deviation: (price - vwap)/vwap | W ∈ {10, 20, 50} |
| `ret_vwap_corr_wW` | Correlation between returns and VWAP basis | W ∈ {10, 20} |

### 5. Volatility Features
| Feature | Definition | Windows |
|---------|------------|---------|
| `vol_wW` | Realized volatility over window W | W ∈ {10, 20, 50} |
| `ret_ma_wW_normvolW2` | Return MA normalized by volatility over W2 | W1 ∈ {5, 10, 20}, W2 ∈ {10, 20} |
| `vol_wW1_std_wW2` | Volatility of volatility: std(vol_wW1) over W2 | W1 ∈ {10, 20}, W2 ∈ {5, 10} |

### 6. Regime Features
| Feature | Definition | Usage |
|---------|------------|-------|
| `regime_highvol` | High volatility regime indicator | Binary flag |
| `ret_highvol_interact` | Return × high volatility interaction | Interaction term |

### 7. Beta Features
| Feature | Definition | Windows |
|---------|------------|---------|
| `beta_market_wW` | Rolling beta to market over window W | W ∈ {10, 20} |
| `ret_normbeta_wW` | Beta-normalized returns over window W | W ∈ {10, 20} |

### 8. Volume Features
| Feature | Definition | Windows |
|---------|------------|---------|
| `vol_ret_t1` | Volume return: log(V_t / V_{t-1}) | Single period |
| `vol_ma_wW` | Volume moving average over window W | W ∈ {5, 10, 20} |
| `vol_adv_wW` | Volume/ADV ratio over window W | W ∈ {5, 10, 20} |
| `vw_ret_wW` | Volume-weighted return over window W | W ∈ {5, 10, 20} |
| `vol_price_trend` | Volume return × price return | Interaction |

### 9. Drawdown Features
| Feature | Definition | Windows |
|---------|------------|---------|
| `dd_current` | Current drawdown from peak | Rolling |
| `dd_max_wW` | Maximum drawdown over window W | W ∈ {10, 20, 50} |

### 10. Entropy Features
| Feature | Definition | Windows |
|---------|------------|---------|
| `ret_perm_entropy_wW` | Permutation entropy of returns over window W | W ∈ {10, 20} |

### 11. Autocorrelation Features
| Feature | Definition | Windows |
|---------|------------|---------|
| `ret_ac1_wW` | 1-lag autocorrelation over window W | W ∈ {10, 20} |
| `ret_pac1_wW` | 1-lag partial autocorrelation over window W | W ∈ {10, 20} |

### 12. Interaction Features
| Feature | Definition | Usage |
|---------|------------|-------|
| `ret_vol_interact` | Return × volatility interaction | Non-linear relationship |
| `vwap_vol_interact` | VWAP deviation × volatility interaction | Regime-dependent |
| `ret_highvol_interact` | Return × high volatility interaction | Regime-specific |

## Feature Consolidation Rules

### Redundancy Removal
1. **Returns**: Keep `ret_sq_t1`, remove `ret_abs_t1` (squared more common in models)
2. **Momentum**: Keep explicit `ret_mom_kK`, remove `ret_ma_wW` if same calculation
3. **Acceleration**: Keep `ret_acc_k1`, remove alternative formulations
4. **VWAP**: Keep standardized versions, remove non-standardized

### Multicollinearity Screening
- **Correlation Threshold**: Remove features with |ρ| > 0.95
- **VIF Threshold**: Remove features with VIF > 10
- **Selection**: Keep feature with higher variance from correlated pair

### Outlier Handling
- **Winsorization**: Clip features at 0.5% and 99.5% percentiles
- **Robust Scaling**: Use median and IQR for scaling
- **Validation**: Check for infinite values and high missing data

## Usage Examples

### Basic Feature Generation
```python
from feature_comparison.standardized_features import StandardizedFeatureGenerator

# Create standardized features
generator = StandardizedFeatureGenerator(data, enable_matrix_ops=True)
versions = generator.generate_standardized_features()

# Get feature definitions
definitions = generator.get_feature_definitions()
```

### Feature Consolidation
```python
from feature_comparison.feature_consolidation import FeatureConsolidator

# Consolidate features
consolidator = FeatureConsolidator()
consolidated_df = consolidator.consolidate_features(df, 'version_name')
cleaned_df = consolidator.remove_multicollinearity(consolidated_df, 'version_name')
winsorized_df = consolidator.winsorize_features(cleaned_df)
```

### Enhanced Analysis
```python
from feature_comparison.enhanced_comparison_runner import EnhancedFeatureComparisonRunner

# Run enhanced analysis
runner = EnhancedFeatureComparisonRunner(
    data=data,
    enable_consolidation=True,
    enable_validation=True
)
results = runner.run_enhanced_analysis()
```

## Quality Metrics

### Feature Validation
- **Data Quality**: Missing data %, duplicate rows, memory usage
- **Feature Quality**: Zero variance, infinite values, constant features
- **Warnings**: High missing data, zero variance features
- **Errors**: Infinite values, data type issues

### Stability Metrics
- **Rank Correlation**: Spearman ρ between methods
- **Bootstrap Stability**: Coefficient of variation across samples
- **Temporal Stability**: Consistency over time windows
- **Multicollinearity**: Correlation and VIF screening

This standardized approach ensures consistent, unambiguous feature definitions across all versions and enables reliable comparison of different feature engineering approaches.