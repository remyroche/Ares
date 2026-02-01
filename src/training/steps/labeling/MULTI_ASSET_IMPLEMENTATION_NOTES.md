# Multi-Asset Global Model Implementation

## Overview
This implementation adds critical de Prado-compliant market residualization and per-asset learning capabilities to the global multi-asset training pipeline.

## Key Components Implemented

### 1. Market-Residualized Labels (CRITICAL)
**File**: `global_meta_labeling_hpo_sample_weighted.py`
**Method**: `_market_residualize_returns()`

**What it does**:
- Computes equal-weighted market return per timestamp across all assets
- Calculates rolling beta (60-period causal window) per asset vs market
- Residualizes returns: `residual_return = asset_return - beta * market_return`
- Stores `beta_to_market`, `market_return`, `residual_return`, `relative_volatility` as features

**Why it's critical**:
- Without this, labels use raw returns = market beta + asset alpha
- Model learns "buy when market goes up" instead of asset-specific predictive patterns
- Violates de Prado's core principle of residualized predictors
- Prevents market-neutral strategies from working

**Configuration**:
```python
single_asset_config['label_return_column'] = 'residual_return'
single_asset_config['use_market_residual_labels'] = True
```

**Note**: The labeling system (`kalman_multi_triple_barrier_labels`) needs to be updated to accept and use `label_return_column` parameter. Currently it hardcodes `'close'` column for returns computation.

### 2. Asset Interaction Features
**File**: `global_meta_labeling_hpo_sample_weighted.py`
**Method**: `_add_asset_interaction_features()`

**What it does**:
- Creates asset-specific versions of key features
- Example: `ETH_volatility_normalized`, `BTC_raw_returns`, etc.
- Enables model to learn different feature importance per asset without deep trees

**Features interacted**:
- `volatility_normalized`
- `raw_returns`
- `vol_regime_asset`

**Why it matters**:
- One-hot encodings alone are weak for tree models (require deep trees + many samples)
- Interaction features create explicit asset-specific predictors
- Model can learn "ETH momentum is more predictive than BTC momentum" directly

### 3. Market Context Features
**File**: `feature_engineering_utils.py`
**Function**: `add_market_context_features()`

**Features added**:
- `market_momentum`: Equal-weighted market momentum across assets
- `relative_momentum`: Asset momentum / market momentum
- `asset_dispersion`: Cross-sectional std of asset returns (regime proxy)
- `market_breadth`: Fraction of assets with positive momentum

**Why it matters**:
- Enables model to learn cross-asset patterns
- Captures market-wide regime shifts
- Provides context for asset-specific predictions

### 4. Per-Asset Volatility Normalization (Enhanced)
**File**: `global_meta_labeling_hpo_sample_weighted.py`
**Method**: `_normalize_volatility_per_asset()`

**Enhancements**:
- Stores `raw_returns` for later market residualization
- Uses expanding window statistics (causal, no look-ahead)
- Computes `vol_regime_asset` using z-scores (not global quantiles)

**De Prado compliance**:
- Expanding window mean/std (strictly causal)
- No look-ahead bias in regime classification

## Pipeline Flow

```
Per-Asset Processing (Parallel):
├─ Load data
├─ Add asset features (one-hots, asset_id)
├─ Normalize volatility (causal expanding window)
├─ Add asset interaction features
└─ Store raw_returns

Global Processing (Sequential):
├─ Concatenate all assets
├─ Sort by timestamp
├─ Market residualization (CRITICAL):
│  ├─ Compute market returns per timestamp
│  ├─ Compute per-asset betas (rolling 60-period)
│  ├─ Compute residual_return = asset_return - beta * market_return
│  └─ Add relative_volatility
├─ Add market context features:
│  ├─ market_momentum
│  ├─ relative_momentum
│  ├─ asset_dispersion
│  └─ market_breadth
└─ Pass to labeling with label_return_column='residual_return'
```

## Configuration Parameters

### Market Residualization
- `beta_window`: 60 (causal rolling window for beta estimation)
- `min_periods`: 20 (minimum samples before computing beta)
- `beta_clip`: (-3.0, 3.0) (clip extreme betas)
- `default_beta`: 1.0 (fallback for insufficient data)

### Asset Interaction
- Interaction features: `['volatility_normalized', 'raw_returns', 'vol_regime_asset']`
- Naming: `{asset}_{feature}` (e.g., `ETH_volatility_normalized`)

### Market Context
- Market return: Equal-weighted across assets per timestamp
- Relative metrics: Asset metric / market metric
- Dispersion: Cross-sectional std of returns
- Breadth: Fraction of assets with positive momentum

## Usage

### Training Global Model
```python
python3 src/launcher/ares_launcher.py \
    global_meta_labeling_hpo_sample_weighted \
    --execution-mode small_multi_asset \
    --assets ETH BTC SOL \
    --exchange binance \
    --timeframe 15m
```

### Configuration
```python
config = {
    'assets': ['ETH', 'BTC', 'SOL'],
    'multi_asset_mode': True,
    'label_return_column': 'residual_return',  # Use residual, not raw
    'use_market_residual_labels': True,
}
```

## De Prado Alignment

| Principle | Implementation | Status |
|-----------|----------------|--------|
| **Residualized predictors** | Labels use `residual_return` | ✅ Implemented (new) |
| **Market-neutral alpha** | Beta-adjusted returns | ✅ Implemented (new) |
| **Causal beta estimation** | Rolling 60-period window | ✅ Implemented (new) |
| **Asset-specific learning** | Interaction features | ✅ Implemented (new) |
| **Cross-asset context** | Market features | ✅ Implemented (new) |
| **Sample uniqueness** | `compute_uniqueness()` in `generate_weights_per_label.py` | ✅ Already Implemented |
| **Per-asset uniqueness** | `compute_uniqueness_per_asset()` | ✅ Implemented (new) |
| **Fractional differentiation** | `_apply_fracdiff()` in `feature_engineering_utils.py` | ✅ Already Implemented |

## Known Limitations & TODOs

### 1. Labeling System Integration (CRITICAL)
**Issue**: `kalman_multi_triple_barrier_labels()` hardcodes `'close'` column for returns
**Fix needed**: Add `return_column` parameter to accept `'residual_return'`
**Location**: `src/training/steps/labeling/multi_label_voting_utils.py:1255`

**Current workaround**: 
- Temporarily replace `market_data['close']` with `market_data['residual_return']` before labeling
- Or add `market_data['close'] = market_data['residual_return']` (hacky but works)

### 2. Sample Uniqueness Weighting ✅ ALREADY IMPLEMENTED
**Status**: Already implemented in `generate_weights_per_label.py`
**Functions**:
- `compute_uniqueness()` - Computes sample weights based on label overlap/concurrency
- `compute_uniqueness_weights()` - Full implementation with timeline-based concurrency counting
- Used in: `weighted_meta_labeling_step.py`, `feature_generation_meta_labeling_step.py`, `label_based_layer_1.py`

**How it works**:
1. Counts concurrent events at each timestamp
2. Calculates uniqueness = 1 / concurrency
3. Computes average uniqueness for each event over its duration
4. Integrated into `generate_weights_per_label()` via `uniqueness_scores` parameter

**Multi-asset consideration**: 
- Current implementation computes uniqueness across ALL events (global)
- For multi-asset, may want to compute uniqueness **per asset** to avoid cross-asset concurrency penalties
- Recommendation: Add `asset_col` parameter to `compute_uniqueness()` for per-asset grouping

### 3. Fractional Differentiation ✅ ALREADY IMPLEMENTED
**Status**: Already implemented in `feature_engineering_utils.py`
**Function**: `_apply_fracdiff()` (lines 85-105)
**Usage**: Called in `apply_layer2_price_processing()` (lines 265-283)

**Implementation details**:
- Applies fractional differentiation with `d=0.4` (default, configurable)
- Uses fixed-width window with threshold-based weight truncation
- Applied to log prices per asset (when `asset_id` detected, uses groupby)
- Generates `fracdiff_log_price` feature

**Multi-asset support**: ✅ Already per-asset via groupby in `apply_layer2_price_processing()`
- Detects `asset_id`, `ticker`, or MultiIndex
- Applies `_apply_single_asset()` per asset group
- Each asset gets independent fractional differentiation

### 4. Per-Asset Uniqueness Weighting ✅ IMPLEMENTED
**Status**: Implemented in `generate_weights_per_label.py` (lines 109-204)
**Function**: `compute_uniqueness_per_asset()`

**Why**: In multi-asset training, an ETH event concurrent with a BTC event should NOT be downweighted (they're independent). Only ETH events concurrent with other ETH events should be downweighted.

**Implementation**:
- Groups events by `asset_col` (default: `'asset_id'`)
- Computes uniqueness weights independently per asset
- Combines weights across all assets
- Ensures cross-asset events don't penalize each other

**Integration**:
- Exported in `__init__.py` for use across labeling steps
- Integrated into `compute_kalman_multi_triple_barrier_sample_weights()` via `use_per_asset_uniqueness=True` parameter
- Can be used in any labeling pipeline that supports multi-asset training

**Usage**:
```python
from src.training.steps.labeling import compute_uniqueness_per_asset

# t1 is a Series with event end times, indexed by start times
# Must have asset_col accessible (as column, MultiIndex level, or attribute)
weights = compute_uniqueness_per_asset(
    t1=t1_series,
    asset_col='asset_id',
    events_index=event_timestamps,
    market_index=market_data  # Optional, for per-asset market alignment
)
```

### 5. Asset Embeddings (FUTURE ENHANCEMENT)
**Missing**: Trainable asset embeddings for better generalization
**Benefit**: 
- Replaces sparse one-hots with dense embeddings
- Enables transfer learning to new assets
- Captures asset similarity automatically
- Reduces feature dimensionality (N one-hots → K embeddings, K << N)

### 6. Per-Asset Diagnostics (FUTURE ENHANCEMENT)
**Missing**: Asset-specific reporting
**Needed**:
- Per-asset feature importance
- Per-asset calibration curves
- Per-asset regime coverage
- Cross-asset correlation analysis

## Performance Considerations

### Memory Optimization
- Downcast float64 to float32 where possible
- Use categorical dtypes for `asset_id`, `vol_regime_asset`
- Consider sparse matrices for one-hot encodings

### Computational Efficiency
- Per-asset processing is parallelizable (not yet implemented)
- Market residualization is O(N*M) where N=timestamps, M=assets
- Consider chunked processing for large asset universes

### Scalability
- Current implementation: Sequential per-asset processing
- Recommended: Use `joblib.Parallel` or `ray` for parallel processing
- Bottleneck: Market context features (requires groupby across all assets)

## Existing De Prado Implementations (Already Multi-Asset Compatible)

### Sample Uniqueness Weighting
**File**: `src/training/steps/labeling/generate_weights_per_label.py`
**Key Functions**:
- `compute_uniqueness_weights()` (lines 109-220): Full concurrency-based uniqueness calculation
- `generate_weights_per_label()` (lines 423-665): Combines magnitude, uniqueness, and time components

**Algorithm** (per de Prado AFML):
1. Build timeline of event start/end times
2. Compute cumulative concurrency at each timestamp
3. Calculate inverse concurrency (uniqueness = 1 / concurrency)
4. Average uniqueness over each event's duration
5. Normalize weights to preserve total sample count

**Integration**: 
- Used in `weighted_meta_labeling_step.py` (line 904)
- Used in `feature_generation_meta_labeling_step.py`
- Used in `label_based_layer_1.py` (line 594)

**Multi-asset note**: Currently computes global concurrency. For true multi-asset independence, should compute per-asset (see recommendation above).

### Fractional Differentiation
**File**: `src/training/steps/labeling/feature_engineering_utils.py`
**Key Functions**:
- `_apply_fracdiff()` (lines 85-105): Fixed-width window fractional differentiation
- `apply_layer2_price_processing()` (lines 170-436): Per-asset feature engineering pipeline

**Algorithm**:
1. Compute fractional differentiation weights: `w_k = -w_{k-1} * (d - k + 1) / k`
2. Truncate weights below threshold (default 1e-5)
3. Apply convolution to log prices using Numba optimization
4. Returns stationary series preserving memory

**Multi-asset support**: ✅ Already implemented
- Detects `asset_id` column (line 202)
- Groups by asset (lines 428-433)
- Applies `_apply_single_asset()` per group
- Each asset gets independent `fracdiff_log_price` feature

**Default parameters**:
- `fracdiff_d = 0.4` (configurable, typical range 0.3-0.6)
- `threshold = 1e-5` (weight truncation)

### Sample Weighting in Labeling
**File**: `src/training/steps/labeling/multi_label_voting_utils.py`
**Function**: `compute_kalman_multi_triple_barrier_sample_weights()` (lines 1190-1300)

**What it does**:
- Averages absolute returns across multiple triple-barrier configurations
- Applies economic floor (prevents zero-weighting small moves)
- Optionally weights by confidence scores
- **NEW**: Optionally applies per-asset uniqueness weighting (multi-asset mode)
- Normalizes to mean=1

**Per-asset uniqueness integration**:
```python
sample_weights = compute_kalman_multi_triple_barrier_sample_weights(
    tb_results=tb_results,
    kalman_volatility=kalman_volatility,
    use_per_asset_uniqueness=True,  # Enable per-asset uniqueness
    asset_col='asset_id',
    market_data=market_data
)
```

**Weight composition**:
- **Magnitude component**: Average absolute returns across configurations
- **Economic floor**: Minimum weight = 0.25 * mean_volatility
- **Uniqueness component** (optional): Per-asset concurrency-based downweighting
- **Final weight**: `magnitude * uniqueness * economic_floor`, normalized to mean=1

### Per-Asset Uniqueness Implementation (NEW)
**File**: `src/training/steps/labeling/generate_weights_per_label.py`
**Function**: `compute_uniqueness_per_asset()` (lines 109-204)

**Algorithm**:
1. Extract asset identifier from t1 series (column, MultiIndex, or attribute)
2. Group events by asset
3. For each asset:
   - Compute uniqueness weights using existing `compute_uniqueness_weights()`
   - Use per-asset market index if provided
4. Combine weights across all assets
5. Align with original event index

**Key features**:
- **Asset independence**: ETH events only compete with ETH events, not BTC
- **Flexible input**: Supports DataFrame, Series with MultiIndex, or Series with asset attribute
- **Fallback**: Gracefully falls back to global uniqueness if asset_col not found
- **Logging**: Reports per-asset statistics (mean, min, max weights)

**Multi-asset benefit**:
- Without per-asset uniqueness: 3 concurrent ETH events + 2 concurrent BTC events = 5 total concurrency → weight = 1/5 = 0.2
- With per-asset uniqueness: 3 concurrent ETH events → ETH weight = 1/3 = 0.33, 2 concurrent BTC events → BTC weight = 1/2 = 0.5
- **Result**: Assets are weighted independently, preventing cross-asset concurrency penalties

## Multi-Asset Reporting ✅ IMPLEMENTED

**File**: `src/training/steps/labeling/multi_asset_reporting.py`
**Class**: `MultiAssetReporter`

### Overview

Comprehensive reporting system that generates **global vs per-asset comparison metrics** for:
- ML model performance
- Feature importance
- Tree/leaf statistics
- Label quality
- Cross-asset correlations

### Reports Generated

#### 1. Model Performance Comparison
**File**: `model_performance_comparison_{timestamp}.csv/.md`

**Metrics compared**:
- AUC (global, per-asset, cross-asset statistics)
- Sharpe ratio
- Information coefficient (IC)
- Win rate
- Average return per trade
- Total PnL
- Number of trades

**Cross-asset statistics**:
- Mean across assets
- Standard deviation across assets
- Min/Max across assets
- Range (max - min)

**Example output**:
```markdown
| Metric | Global | ETH | BTC | SOL | Mean | Std | Min | Max | Range |
|--------|--------|-----|-----|-----|------|-----|-----|-----|-------|
| auc    | 0.6234 | 0.6512 | 0.6123 | 0.6089 | 0.6241 | 0.0223 | 0.6089 | 0.6512 | 0.0423 |
| sharpe | 1.23   | 1.45   | 1.12   | 1.08   | 1.22   | 0.19   | 1.08   | 1.45   | 0.37   |
```

**Insights provided**:
- Best/worst performing assets
- High cross-asset variance metrics (std > 0.1)
- Performance divergence analysis

#### 2. Feature Importance Comparison
**File**: `feature_importance_comparison_{timestamp}.csv/.md`

**Shows**:
- Top 20 features globally
- Per-asset feature importance
- Coefficient of variation (CV) across assets
- Asset-specific features (high CV > 0.5)

**Example output**:
```markdown
| Feature | Global | ETH | BTC | SOL | Mean | Std | CV |
|---------|--------|-----|-----|-----|------|-----|----|
| residual_return | 0.1234 | 0.1456 | 0.1123 | 0.1089 | 0.1223 | 0.0189 | 0.1545 |
| ETH_volatility_normalized | 0.0892 | 0.1234 | 0.0023 | 0.0019 | 0.0425 | 0.0656 | 1.5435 |
```

**Insights**:
- Shared features (low CV): Important across all assets
- Asset-specific features (high CV): Only important for specific assets
- Feature importance divergence

#### 3. Tree/Leaf Statistics Comparison
**File**: `tree_statistics_comparison_{timestamp}.csv/.md`

**Metrics**:
- Average tree depth
- Number of leaves
- Average leaf purity
- Number of split features
- Maximum tree depth

**Cross-asset analysis**:
- Mean/std/min/max across assets
- Identifies assets requiring deeper trees (more complex patterns)

#### 4. Label Quality Comparison
**File**: `label_quality_comparison_{timestamp}.csv/.md`

**Metrics**:
- Label balance (positive rate)
- Label entropy
- Event density (events per day)
- Total samples per asset

**Example output**:
```markdown
| Asset | Positive Rate | Label Count | Entropy | Events/Day | Total Samples |
|-------|---------------|-------------|---------|------------|---------------|
| ETH   | 0.4523        | 1234        | 0.9912  | 12.34      | 10000         |
| BTC   | 0.4789        | 1456        | 0.9956  | 14.56      | 12000         |
| SOL   | 0.4312        | 1089        | 0.9876  | 10.89      | 9000          |
| GLOBAL| 0.4541        | 3779        | 0.9915  | 12.60      | 31000         |
```

#### 5. Cross-Asset Correlation Analysis
**File**: `cross_asset_correlation_{timestamp}.md`

**Analyzes**:
- Raw return correlations
- Residual return correlations (after market residualization)
- Volatility correlations

**Example output**:
```markdown
## Raw Return Correlations
| Asset | ETH | BTC | SOL |
|-------|-----|-----|-----|
| ETH   | 1.00| 0.78| 0.65|
| BTC   | 0.78| 1.00| 0.72|
| SOL   | 0.65| 0.72| 1.00|

Average pairwise correlation: 0.7167

## Residual Return Correlations (Market-Adjusted)
| Asset | ETH | BTC | SOL |
|-------|-----|-----|-----|
| ETH   | 1.00| 0.23| 0.18|
| BTC   | 0.23| 1.00| 0.25|
| SOL   | 0.18| 0.25| 1.00|

Average pairwise correlation (residual): 0.2200
*Lower residual correlations indicate better market residualization.*
```

**Insight**: Lower residual correlations confirm successful market residualization.

#### 6. Master Summary Report
**File**: `multi_asset_summary_{timestamp}.md`

**Combines**:
- Configuration summary
- Dataset statistics
- Model performance summary
- Key insights
- De Prado compliance checklist

### Usage

```python
from src.training.steps.labeling import MultiAssetReporter

# Initialize reporter
reporter = MultiAssetReporter(outcomes_dir='outcomes')

# Generate all reports
report_paths = reporter.generate_multi_asset_report(
    combined_df=combined_df,
    model_results={
        'model_performance': {
            'global': {'auc': 0.62, 'sharpe': 1.23, ...},
            'ETH': {'auc': 0.65, 'sharpe': 1.45, ...},
            'BTC': {'auc': 0.61, 'sharpe': 1.12, ...},
            'SOL': {'auc': 0.61, 'sharpe': 1.08, ...}
        },
        'feature_importance': {
            'global': {'residual_return': 0.123, ...},
            'ETH': {'residual_return': 0.146, ...},
            ...
        },
        'tree_stats': {
            'global': {'avg_depth': 5.2, 'num_leaves': 32, ...},
            'ETH': {'avg_depth': 5.8, 'num_leaves': 38, ...},
            ...
        }
    },
    assets=['ETH', 'BTC', 'SOL'],
    asset_col='asset_id',
    config={'multi_asset_mode': True, 'label_return_column': 'residual_return'}
)

# Access individual report paths
print(f"Model performance: {report_paths['model_performance']}")
print(f"Feature importance: {report_paths['feature_importance']}")
print(f"Master summary: {report_paths['master_summary']}")
```

### Helper Function

For adding multi-asset metrics to existing reports:

```python
from src.training.steps.labeling import add_multi_asset_metrics_to_existing_report

# Enhance existing report data
enhanced_report = add_multi_asset_metrics_to_existing_report(
    report_data=existing_report,
    assets=['ETH', 'BTC', 'SOL'],
    asset_col='asset_id'
)

# Now includes:
# - {metric}_mean_across_assets
# - {metric}_std_across_assets
# - {metric}_min_across_assets
# - {metric}_max_across_assets
# - {metric}_range
```

### Integration with Existing Reports

The `MultiAssetReporter` can be integrated into existing reporting pipelines:

**In `global_meta_labeling_hpo_sample_weighted.py`**:
```python
# After training completes
if config.get('multi_asset_mode', False):
    from src.training.steps.labeling import MultiAssetReporter
    
    reporter = MultiAssetReporter(outcomes_dir=self.outcomes_dir)
    report_paths = reporter.generate_multi_asset_report(
        combined_df=combined_df,
        model_results=results,
        assets=assets,
        asset_col='asset_id',
        config=config
    )
    
    tprint_success(f"✅ Multi-asset reports generated: {len(report_paths)} files")
```

### Report Output Structure

```
outcomes/
├── model_performance_comparison_20250101_120000.csv
├── model_performance_comparison_20250101_120000.md
├── feature_importance_comparison_20250101_120000.csv
├── feature_importance_comparison_20250101_120000.md
├── tree_statistics_comparison_20250101_120000.csv
├── tree_statistics_comparison_20250101_120000.md
├── label_quality_comparison_20250101_120000.csv
├── label_quality_comparison_20250101_120000.md
├── cross_asset_correlation_20250101_120000.md
└── multi_asset_summary_20250101_120000.md
```

## Validation Checklist

Before deploying:
- [ ] Verify `residual_return` is used for labeling (not `close`)
- [ ] Check beta values are reasonable (-3 to 3, mean ~1.0)
- [ ] Confirm market_return is equal-weighted (not cap-weighted)
- [ ] Validate per-asset sample counts are sufficient (>1000 samples)
- [ ] Test with 2-3 assets first before scaling to 10+
- [ ] Monitor memory usage (expect 2-3x increase vs single-asset)
- [ ] Verify synchronized CV splits (all assets in same fold)
- [ ] Check for NaN propagation in residual_return computation
- [ ] **Generate multi-asset comparison reports** to verify per-asset learning
- [ ] **Review cross-asset correlation** to confirm market residualization
- [ ] **Check feature importance divergence** to validate asset-specific features

## References

- de Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 5: Fractional Differentiation
  - Chapter 7: Cross-Validation in Finance
  - Chapter 8: Feature Importance
  - Chapter 20: Multiprocessing and Vectorization

## Contact

For questions or issues with multi-asset implementation:
1. Check this documentation first
2. Review `global_meta_labeling_hpo_sample_weighted.py` implementation
3. Verify labeling system accepts `label_return_column` parameter
