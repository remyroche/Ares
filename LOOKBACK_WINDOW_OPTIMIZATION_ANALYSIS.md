# Lookback/Window Optimization Analysis: HMM Cluster Integration

## Executive Summary

Yes, the lookback/window optimization in feature engineering **does operate on a per-HMM cluster basis**, similar to the triple barrier method. The system implements both global optimization and regime-specific optimization, providing adaptive feature engineering that tailors lookback periods to different market regimes.

## Dual-Layer Optimization Architecture

### 1. Global Optimization Layer

**Location**: `diverse_lookback_optimizer.py` - `find_diverse_lookback_periods()` method

```python
async def find_diverse_lookback_periods(
    self,
    data: pd.DataFrame,
    target: pd.Series,
    regimes: Optional[pd.Series] = None,
    symbol: str = "UNKNOWN",
    exchange: str = "UNKNOWN",
    timeframe: str = "1m"
) -> dict[str, Any]:
    """Find diverse lookback periods for each feature."""

    results = {
        "diverse_lookback_periods": {},
        "regime_specific_periods": {}
    }

    # 1. Find diverse lookback periods for each feature (global)
    diverse_periods = await self._find_diverse_periods_for_all_features(data, target)
    results["diverse_lookback_periods"] = diverse_periods

    # 2. Regime-specific diverse periods (if regimes available)
    if regimes is not None and len(regimes.unique()) > 1:
        regime_periods = await self._find_regime_specific_diverse_periods(
            data, target, regimes, diverse_periods
        )
        results["regime_specific_periods"] = regime_periods
```

### 2. Regime-Specific Optimization Layer

**Location**: `diverse_lookback_optimizer.py` - `_find_regime_specific_diverse_periods()` method

```python
async def _find_regime_specific_diverse_periods(
    self,
    data: pd.DataFrame,
    target: pd.Series,
    regimes: pd.Series,
    global_periods: dict[str, Any]
) -> dict[str, Any]:
    """Find regime-specific diverse periods."""

    regime_periods = {}

    for regime in regimes.unique():
        regime_mask = regimes == regime
        regime_data = data[regime_mask]
        regime_target = target[regime_mask]

        if len(regime_data) >= 100:  # Minimum sample requirement
            self.logger.info(f"🔄 Finding diverse periods for regime {regime}...")

            regime_specific = await self._find_diverse_periods_for_all_features(
                regime_data, regime_target
            )

            regime_periods[f"regime_{regime}"] = regime_specific

    return regime_periods
```

## Per-Regime Optimization Process

### 1. Regime Data Segmentation

```python
# Extract regime-specific data
for regime in regimes.unique():
    regime_mask = regimes == regime
    regime_data = data[regime_mask]
    regime_target = target[regime_mask]
```

**Key Features**:
- **Regime Isolation**: Each regime is processed independently
- **Minimum Sample Requirement**: Requires ≥100 samples per regime
- **Regime-Specific Targets**: Uses regime-specific target variables

### 2. Regime-Specific Feature Calculation

```python
def _calculate_feature_with_period(
    self,
    data: pd.DataFrame,
    feature_name: str,
    period: int
) -> Optional[pd.Series]:
    """Calculate feature with specific lookback period."""

    if feature_name == "RSI":
        return self._calculate_rsi(data['close'], period)
    elif feature_name == "MACD_fast":
        return self._calculate_ema(data['close'], period)
    # ... other features
```

**Regime-Specific Features**:
- **RSI**: Different periods for different regimes
- **MACD**: Fast/slow periods optimized per regime
- **Bollinger Bands**: Periods tailored to regime volatility
- **Moving Averages**: Short/long periods per regime
- **ATR**: Volatility periods per regime

### 3. Regime-Specific Period Selection

```python
def _select_diverse_subset(self, meaningful_periods: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Select diverse subset using greedy algorithm."""

    target_count = min(
        self.diverse_config["target_periods_per_feature"],
        len(meaningful_periods)
    )

    # Start with the period with highest information score
    selected = [meaningful_periods[0]]
    remaining = meaningful_periods[1:]

    # Greedy selection: add periods that maximize diversity
    while len(selected) < target_count and remaining:
        best_candidate = None
        best_diversity_score = -1

        for candidate in remaining:
            # Calculate diversity score for this candidate
            candidate_set = selected + [candidate]
            diversity_score = self._calculate_set_diversity_score(candidate_set)

            if diversity_score > best_diversity_score:
                best_diversity_score = diversity_score
                best_candidate = candidate
```

## Comparison with Triple Barrier Method

### 1. Triple Barrier Regime Optimization

**Location**: `regime_aware_triple_barrier_labeling.py`

```python
@dataclass
class RegimeTripleBarrierConfig:
    """Configuration for regime-specific triple barrier parameters."""

    # Regime-specific parameters
    regime_profit_take_multipliers: Dict[str, float] = None
    regime_stop_loss_multipliers: Dict[str, float] = None
    regime_time_barrier_minutes: Dict[str, int] = None
    regime_max_lookahead: Dict[str, int] = None

    # TPSL parameters
    regime_tp_multipliers: Dict[str, float] = None
    regime_sl_multipliers: Dict[str, float] = None
    regime_position_sizes: Dict[str, float] = None
```

### 2. Lookback Period Regime Optimization

**Location**: `diverse_lookback_optimizer.py`

```python
# Regime-specific diverse periods
regime_periods = {
    "regime_0": {
        "RSI": {"selected_periods": [14, 21, 30]},
        "MACD_fast": {"selected_periods": [12, 18]},
        "Bollinger_Bands": {"selected_periods": [20, 30]}
    },
    "regime_1": {
        "RSI": {"selected_periods": [10, 16, 25]},
        "MACD_fast": {"selected_periods": [8, 14]},
        "Bollinger_Bands": {"selected_periods": [15, 25]}
    }
}
```

## Parallel Architecture: Both Methods Use Per-Regime Optimization

### 1. Triple Barrier Method
- **Regime-Specific Barriers**: Different profit take/stop loss per regime
- **Regime-Specific Timing**: Different time barriers per regime
- **Regime-Specific Position Sizing**: Different position sizes per regime

### 2. Lookback Period Optimization
- **Regime-Specific Periods**: Different lookback periods per regime
- **Regime-Specific Features**: Different feature combinations per regime
- **Regime-Specific Diversity**: Different diversity thresholds per regime

## Matrix-Based Optimization

### 1. Matrix Diverse Lookback Optimizer

**Location**: `matrix_diverse_lookback_optimizer.py`

```python
class MatrixDiverseLookbackOptimizer:
    """Matrix-based optimizer that finds diverse yet meaningful lookback periods."""

    async def find_diverse_lookback_periods_matrix(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        regimes: Optional[pd.Series] = None,
        symbol: str = "UNKNOWN",
        exchange: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> dict[str, Any]:
        """Find diverse lookback periods using matrix/vector optimization."""
```

**Matrix Optimization Features**:
- **Vectorized Operations**: Efficient matrix-based calculations
- **Parallel Processing**: Batch processing for multiple regimes
- **Matrix Correlation Analysis**: Efficient correlation computation
- **Vector-Based Diversity Scoring**: Fast diversity evaluation

## Configuration and Parameters

### 1. Global Configuration

```python
self.diverse_config = {
    "target_periods_per_feature": 3,
    "min_periods_per_feature": 2,
    "max_periods_per_feature": 3,
    "diversity_threshold": 0.3,
    "meaningful_threshold": 0.1,
    "correlation_threshold": 0.7,
    "information_diversity_weight": 0.4,
    "signal_strength_weight": 0.4,
    "correlation_penalty_weight": 0.2
}
```

### 2. Regime-Specific Configuration

```python
# Each regime can have different parameters
regime_config = {
    "regime_0": {
        "diversity_threshold": 0.25,  # More diverse periods
        "meaningful_threshold": 0.15,  # Higher signal strength
        "correlation_threshold": 0.6   # Lower correlation allowed
    },
    "regime_1": {
        "diversity_threshold": 0.35,  # Less diverse periods
        "meaningful_threshold": 0.08,  # Lower signal strength
        "correlation_threshold": 0.8   # Higher correlation allowed
    }
}
```

## Feature Engineering Integration

### 1. Step 6 Feature Engineering

**Location**: `step06_feature_engineering.py`

```python
async def _create_comprehensive_features(
    unified_data: pd.DataFrame,
    labeled_data: pd.DataFrame,
    regime_data: pd.DataFrame,
    feature_engineer: VectorizedAdvancedFeatureEngineering,
    symbol: str,
    exchange: str,
    timeframe: str
) -> Dict[str, Any]:
    """Create comprehensive features using vectorized feature engineering."""

    # Add regime-aware features if regime data is available
    if regime_data is not None:
        features_df = _add_regime_aware_features(features_df, merged_data)

    # Add HMM feature enhancement if regime data is available
    if regime_data is not None:
        features_df = _enhance_hmm_features(features_df, regime_data)
```

### 2. Regime-Aware Feature Engineering

```python
def _add_regime_aware_features(features_df: pd.DataFrame, merged_data: pd.DataFrame) -> pd.DataFrame:
    """Add regime-aware features using optimized lookback periods."""

    # Load regime-specific lookback periods
    regime_periods = diverse_optimizer.get_diverse_lookback_periods(symbol, exchange, timeframe)

    for regime in merged_data['composite_cluster_id'].unique():
        regime_mask = merged_data['composite_cluster_id'] == regime
        regime_data = merged_data[regime_mask]

        # Apply regime-specific lookback periods
        regime_features = _apply_regime_specific_periods(regime_data, regime_periods[f"regime_{regime}"])
        features_df.loc[regime_mask] = regime_features
```

## Results and Output Structure

### 1. Global Results

```json
{
    "diverse_lookback_periods": {
        "RSI": {
            "selected_periods": [14, 21, 30],
            "period_scores": [...],
            "diversity_metrics": {...}
        },
        "MACD_fast": {
            "selected_periods": [12, 18],
            "period_scores": [...],
            "diversity_metrics": {...}
        }
    }
}
```

### 2. Regime-Specific Results

```json
{
    "regime_specific_periods": {
        "regime_0": {
            "RSI": {
                "selected_periods": [10, 16, 25],
                "period_scores": [...],
                "diversity_metrics": {...}
            }
        },
        "regime_1": {
            "RSI": {
                "selected_periods": [12, 20, 35],
                "period_scores": [...],
                "diversity_metrics": {...}
            }
        }
    }
}
```

## Performance and Efficiency

### 1. Parallel Processing

```python
# Process multiple regimes in parallel
for regime in regimes.unique():
    regime_mask = regimes == regime
    regime_data = data[regime_mask]
    regime_target = target[regime_mask]

    # Parallel optimization for each regime
    regime_specific = await self._find_diverse_periods_for_all_features(
        regime_data, regime_target
    )
```

### 2. Memory Optimization

```python
@memory_efficient
@comprehensive_data_validation
async def _find_diverse_periods_for_all_features(
    self,
    data: pd.DataFrame,
    target: pd.Series
) -> dict[str, Any]:
    """Find diverse lookback periods for all features with memory optimization."""
```

## Best Practices Implemented

### 1. Regime-Specific Validation

- **Minimum Sample Requirements**: ≥100 samples per regime
- **Regime-Specific Quality Metrics**: Different thresholds per regime
- **Regime-Specific Diversity Analysis**: Tailored diversity scoring

### 2. Fallback Mechanisms

```python
# Fallback to global periods if regime-specific fails
if len(regime_data) < 100:
    self.logger.warning(f"⚠️ Insufficient data for regime {regime}, using global periods")
    regime_specific = global_periods
```

### 3. Comprehensive Logging

```python
self.logger.info(f"🔄 Finding diverse periods for regime {regime}...")
self.logger.info(f"📊 Regime {regime} optimization completed:")
self.logger.info(f"   - Selected periods: {selected_periods}")
self.logger.info(f"   - Diversity score: {diversity_score:.3f}")
self.logger.info(f"   - Information score: {information_score:.3f}")
```

## Conclusion

The lookback/window optimization in feature engineering **definitely operates on a per-HMM cluster basis**, just like the triple barrier method. The system implements:

1. **Dual-Layer Optimization**: Global + regime-specific optimization
2. **Regime-Specific Periods**: Different lookback periods for each regime
3. **Regime-Specific Features**: Different feature combinations per regime
4. **Regime-Specific Quality Metrics**: Different thresholds per regime
5. **Parallel Processing**: Efficient processing across multiple regimes

This architecture ensures that feature engineering is adaptive to different market regimes, providing more nuanced and effective feature sets that are tailored to specific market conditions, just like the regime-aware triple barrier labeling system.