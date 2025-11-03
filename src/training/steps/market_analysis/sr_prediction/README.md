# SR Performance Prediction

Multi-output LightGBM model for predicting Support/Resistance level performance metrics using SHAP for interpretability.

## Overview

This module provides a **complementary** approach to the existing SR quality scoring system (`src/tactician/sr_levels/ml_quality/`). While the quality model predicts a single composite score, this module predicts **specific tradeable performance metrics** for SR levels.

### Key Differences

| Aspect | Quality Model (`ml_quality/`) | Performance Model (`sr_prediction/`) |
|--------|-------------------------------|--------------------------------------|
| **Purpose** | General quality assessment | Tradeable performance prediction |
| **Outputs** | Single `quality_score` (0-1) | 3 metrics: bounce, hold, profit |
| **Use Case** | Filter/rank all SR levels | Predict tested level behavior |
| **Training Data** | All detected levels | Only tested levels |
| **Location** | Tactician (runtime) | Training pipeline (offline) |

## Architecture

### Prediction Targets

The model predicts three key metrics for SR levels **that get tested**:

1. **bounce_strength** (0-1): How strongly price bounces when level is tested
   - 0.0 = No bounce, level breaks immediately
   - 0.5 = Moderate bounce (~1% retracement)
   - 1.0 = Strong bounce (≥2% retracement)

2. **hold_strength** (0-1): Whether level holds or breaks when tested
   - 0.0 = Breaks immediately
   - 0.5 = Holds for ~10 bars before breaking
   - 1.0 = Holds completely (≥20 bars or never breaks)

3. **trade_profit** (-1 to 1): Simulated trade profitability when traded
   - -1.0 = Maximum loss (-2% or worse)
   - 0.0 = Breakeven
   - 1.0 = Maximum profit (≥2%)

### Features (40+)

Leverages all features from `SRQualityDataCollector`:

**Basic SR Features:**
- strength, prominence, width, volume_confirmation
- consistency, touch_count, age_bars, failure_count

**Bounce Metrics:**
- avg_bounce_ratio, max_bounce_ratio, volume_weighted_bounce
- strong_bounce_count, strong_bounce_ratio

**Time Features:**
- time_decay (30/100 bar windows), recency_score
- age_category, time_adjusted_strength

**Market Context:**
- market_volatility, market_trend, market_momentum
- market_volume_avg

**Regime-Adjusted:**
- vol_adjusted_strength, trend_alignment
- regime_appropriate_strength

**Interaction Features:**
- strength × volume, prominence × width
- touch × consistency, cluster × multi_tf

**Position Features:**
- price_position, distance_to_current_pct
- is_support

**Confluence:**
- method_count, method_confluence
- method_diversity, agreement_score

### Model Architecture

- **Framework**: LightGBM (gradient boosting decision trees)
- **Approach**: Multi-output regression (3 separate models)
- **SHAP**: TreeExplainer for feature importance and explanations
- **Anti-Overfitting**:
  - Strong L1/L2 regularization (λ=1.0)
  - Shallow trees (max_depth=4, num_leaves=15)
  - High min_data_in_leaf (50 samples)
  - Feature/bagging subsampling (70%)
  - Time series cross-validation (5 folds)

## Usage

### Basic Training

Train a model on Bitcoin 1-hour data:

```bash
python -m src.training.steps.market_analysis.sr_prediction.sr_prediction_runner \
  --symbol BTCUSDT \
  --exchange binance \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --timeframe 1h \
  --output-dir outputs/sr_prediction/btc_1h
```

### Multi-Symbol Training

Train on multiple symbols for better generalization:

```bash
python -m src.training.steps.market_analysis.sr_prediction.sr_prediction_runner \
  --symbol BTCUSDT,ETHUSDT,SOLUSDT \
  --multi-symbol \
  --exchange binance \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --output-dir outputs/sr_prediction/multi_asset
```

### Save/Load Training Data

Collect data once and reuse for multiple training runs:

```bash
# Collect and save data
python -m src.training.steps.market_analysis.sr_prediction.sr_prediction_runner \
  --symbol BTCUSDT \
  --exchange binance \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --save-data data/sr_training_btc.parquet \
  --output-dir outputs/sr_prediction/temp

# Load and train
python -m src.training.steps.market_analysis.sr_prediction.sr_prediction_runner \
  --load-data data/sr_training_btc.parquet \
  --output-dir outputs/sr_prediction/btc_1h
```

### Advanced Options

```bash
python -m src.training.steps.market_analysis.sr_prediction.sr_prediction_runner \
  --symbol BTCUSDT \
  --exchange binance \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --timeframe 4h \
  --forward-days 15 \
  --sample-freq-days 3 \
  --n-folds 5 \
  --num-boost-round 1500 \
  --early-stopping-rounds 75 \
  --use-weights \
  --weight-method tiered \
  --generate-shap \
  --output-dir outputs/sr_prediction/btc_4h_weighted
```

### Hyperparameter Optimization (HPO)

Train with automatic hyperparameter tuning:

```bash
python -m src.training.steps.market_analysis.sr_prediction.sr_prediction_runner \
  --symbol BTCUSDT \
  --exchange binance \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --use-hpo \
  --hpo-trials 100 \
  --hpo-method bayesian \
  --output-dir outputs/sr_prediction/btc_hpo
```

**HPO Methods:**
- `bayesian`: Bayesian optimization with Gaussian processes (recommended)
- `staged`: Multi-stage optimization (coarse → fine)
- `multi_objective`: Multi-objective Pareto optimization

**HPO Parameters Optimized:**
- `num_leaves`: Tree complexity (15-63)
- `max_depth`: Maximum tree depth (3-10)
- `learning_rate`: Learning rate (0.01-0.1)
- `feature_fraction`: Feature sampling ratio (0.5-0.9)
- `bagging_fraction`: Data bagging ratio (0.5-0.9)
- `bagging_freq`: Bagging frequency (1-10)
- `min_data_in_leaf`: Minimum samples per leaf (20-100)
- `lambda_l1`: L1 regularization (0.0-2.0)
- `lambda_l2`: L2 regularization (0.0-2.0)

## Command Line Arguments

### Data Collection

- `--symbol`: Trading symbol or comma-separated list (default: BTCUSDT)
- `--exchange`: Exchange name (default: binance)
- `--start-date`: Start date YYYY-MM-DD (default: 2023-01-01)
- `--end-date`: End date YYYY-MM-DD (default: 2024-01-01)
- `--timeframe`: Timeframe (1h, 4h, 1d, etc.) (default: 1h)
- `--forward-days`: Days to look forward for labeling (default: 10)
- `--sample-freq-days`: Sample frequency in days (default: 7)
- `--multi-symbol`: Enable multi-symbol mode

### Data I/O

- `--load-data PATH`: Load pre-collected data instead of collecting
- `--save-data PATH`: Save collected data to file (.csv, .parquet, .pkl)

### Training

- `--n-folds`: Number of CV folds (default: 5)
- `--num-boost-round`: Max boosting rounds (default: 1000)
- `--early-stopping-rounds`: Early stopping patience (default: 50)
- `--filter-untested`: Filter out untested levels (default: True)
- `--no-filter-untested`: Include untested levels

### Validation

- `--no-validation`: Skip validation split, use all data for training
- `--val-ratio`: Validation set ratio (default: 0.2)

### Hyperparameter Optimization

- `--use-hpo`: Enable hyperparameter optimization
- `--hpo-trials`: Number of HPO trials per target (default: 50)
- `--hpo-method`: HPO method (bayesian, staged, multi_objective) (default: bayesian)

### Sample Weighting

- `--use-weights`: Apply confidence-based sample weighting
- `--weight-method`: Weighting method (quality_based, tiered, exponential)

### SHAP Analysis

- `--generate-shap`: Generate SHAP analysis and plots (default: True)
- `--no-shap`: Skip SHAP analysis

### Output

- `--output-dir`: Output directory (default: outputs/sr_prediction)

## Programmatic Usage

### Training

```python
from src.training.steps.market_analysis.sr_prediction import (
    SRPerformancePredictor,
    SRTrainingDataBuilder
)
import asyncio

# Collect training data
builder = SRTrainingDataBuilder()
data = asyncio.run(builder.collect_data(
    symbol='BTCUSDT',
    exchange='binance',
    start_date='2023-01-01',
    end_date='2024-01-01',
    timeframe='1h'
))

# Filter untested levels
data = builder.filter_untested_levels(data)

# Train model
predictor = SRPerformancePredictor()
metrics = predictor.train(data, n_folds=5)

# Save model
predictor.save('models/sr_prediction')
```

### Prediction

```python
from src.training.steps.market_analysis.sr_prediction import SRPerformancePredictor
import pandas as pd

# Load trained model
predictor = SRPerformancePredictor()
predictor.load('models/sr_prediction')

# Prepare features (example)
features = pd.DataFrame([{
    'feature_strength': 0.85,
    'feature_prominence': 0.72,
    'feature_touch_count': 5,
    'feature_volume_weighted_bounce': 0.68,
    # ... all other features
}])

# Get predictions
predictions = predictor.predict(features)
# {'bounce_strength': [0.73], 'hold_strength': [0.81], 'trade_profit': [0.42]}

# Get single prediction with explanation
explanation = predictor.explain_prediction(features, target='bounce_strength', sample_idx=0)
# {
#   'prediction': 0.73,
#   'base_value': 0.45,
#   'shap_values': {'feature_strength': 0.12, 'feature_prominence': 0.08, ...},
#   'feature_values': {'feature_strength': 0.85, ...}
# }
```

### SHAP Analysis

```python
# Get feature importance
importance = predictor.get_feature_importance(
    target='bounce_strength',
    method='gain',
    top_n=20
)

# Generate SHAP summary plot
predictor.plot_shap_summary(
    training_data=train_data,
    target='bounce_strength',
    save_path='shap_summary_bounce.png'
)
```

## Output Structure

After training, the output directory contains:

```
outputs/sr_prediction/
├── models/
│   ├── model_bounce_strength.txt      # LightGBM model
│   ├── model_hold_strength.txt        # LightGBM model
│   ├── model_trade_profit.txt         # LightGBM model
│   └── metadata.json                  # Feature names, config, metrics
├── shap_analysis/
│   ├── shap_summary_bounce_strength.png
│   ├── shap_summary_hold_strength.png
│   ├── shap_summary_trade_profit.png
│   ├── feature_importance_bounce_strength.csv
│   ├── feature_importance_hold_strength.csv
│   └── feature_importance_trade_profit.csv
├── training_metrics.txt               # CV scores for all targets
├── validation_results.txt             # Hold-out validation results
└── data_quality_stats.txt            # Dataset statistics
```

## Data Collection Details

### Walkforward Labeling

The data collector uses a walkforward approach:

1. Load full historical OHLCV data via `RealDataLoader`
2. Sample dates at regular intervals (e.g., weekly)
3. For each sample date:
   - Use historical data to detect SR levels
   - Use future data (forward_days) to measure performance
   - Extract features from historical data
   - Label with forward performance metrics

### Performance Measurement

For each detected SR level, the collector:

1. **Checks if tested**: Did price reach the level (±0.5% tolerance)?
2. **Measures bounce**: How far did price bounce from the level?
3. **Measures hold**: Did the level hold or break?
4. **Simulates trade**: What profit would a bounce trade make?

### Quality Filtering

By default, the model filters out:
- **Untested levels** (hit_rate = 0): No information about performance
- **Weak levels** (strength < 0.3): Too noisy for reliable prediction

## Best Practices

### Training Data

- **Minimum samples**: ≥1,000 tested levels for reliable training
- **Date range**: At least 6-12 months of data
- **Multiple regimes**: Include bull, bear, and sideways markets
- **Multi-symbol**: Train on 3-5 correlated assets for generalization

### Sample Frequency

- **1h/4h timeframes**: Sample every 3-7 days
- **1d timeframe**: Sample every 7-14 days
- **Tradeoff**: More frequent = more samples, less independent

### Forward Window

- **Timeframe dependent**:
  - 1h → 5-10 days forward
  - 4h → 10-20 days forward
  - 1d → 20-40 days forward
- **Goal**: Capture bounce/break behavior without too much noise

### Model Deployment

1. **Training**: Run offline on historical data
2. **Validation**: Check performance on recent holdout data
3. **Integration**: Load model in trading system
4. **Updates**: Retrain monthly/quarterly with new data

## Interpretability with SHAP

### Feature Importance

SHAP provides better feature importance than traditional methods:
- **Global importance**: Mean absolute SHAP values across all predictions
- **Directional**: Shows positive vs. negative impact
- **Interaction-aware**: Captures feature interactions

### Prediction Explanations

For any prediction, SHAP shows:
- **Base value**: Average prediction across all samples
- **Feature contributions**: How each feature pushed prediction up/down
- **Final prediction**: Base value + sum of contributions

Example interpretation:
```
Prediction: bounce_strength = 0.73

Base value: 0.45 (average)
Contributions:
  + feature_strength (0.85) → +0.12
  + feature_volume_weighted_bounce (0.68) → +0.08
  + feature_touch_count (5) → +0.05
  - feature_age_bars (150) → -0.03
  ...
= 0.73
```

## Troubleshooting

### "Insufficient training data"

- Increase date range or decrease sample_freq_days
- Use multi-symbol training
- Check data availability in data_cache/

### Poor validation performance

- Reduce model complexity (lower num_leaves, max_depth)
- Increase regularization (lambda_l1, lambda_l2)
- Use sample weighting (--use-weights)
- Check for data leakage

### SHAP computation slow

- Subsample training data (done automatically >1000 samples)
- Skip SHAP generation (--no-shap)
- Use feature importance from LightGBM instead

## Technical Details

### Dependencies

- `lightgbm`: Gradient boosting framework
- `shap`: Model interpretability
- `scikit-learn`: Cross-validation and metrics
- `pandas`, `numpy`: Data manipulation
- `matplotlib`: Plotting

### Performance

- **Training time**: ~5-10 minutes for 5k samples (M1 Mac)
- **SHAP computation**: ~30 seconds for 1k samples per target
- **Prediction**: <1ms per level

### Memory Usage

- **Training**: ~500MB-2GB depending on sample count
- **SHAP**: ~1-2GB for large datasets (hence subsampling)
- **Deployed model**: ~10-50MB depending on tree count

## Future Enhancements

- [ ] Hyperparameter optimization (HPO) integration
- [ ] Multi-timeframe ensemble predictions
- [ ] Online learning / incremental updates
- [ ] Regime-specific models
- [ ] Calibrated probability outputs
- [ ] Integration with backtesting framework

## Related Components

- **Data Collection**: `src.tactician.sr_levels.ml_quality.SRQualityDataCollector`
- **SR Detection**: `src.tactician.sr_levels.enhanced_sr_detection.EnhancedSRDetector`
- **Feature Extraction**: `src.tactician.sr_levels.sr_modules.sr_feature_extractor`
- **Quality Model**: `src.tactician.sr_levels.ml_quality.SRQualityModel`

## License

Part of the Ares trading system.

