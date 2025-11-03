# 100% Data-Driven SR Level ML System

A complete support/resistance level prediction system with **zero heuristics**. All detection methods, features, targets, and hyperparameters are learned from data using LGBM and SHAP.

## Philosophy: Zero Heuristics

This system eliminates ALL hand-crafted rules:

- ❌ No predefined SR detection methods (HVN, Bollinger, Pivots, etc.)
- ❌ No hand-picked feature categories
- ❌ No composite score weights
- ❌ No arbitrary thresholds
- ❌ No YAML configuration files

Instead:

- ✅ Pure mathematical local extrema generation
- ✅ Exhaustive raw feature transformations (300-500 features)
- ✅ Multi-target generation (50-100 possible targets)
- ✅ AutoML target selection by validation performance
- ✅ LGBM+SHAP-only feature selection
- ✅ HPO for all hyperparameters
- ✅ Complete SHAP interpretability

## Architecture

### Pipeline

```
1. Data Collection
   └─> Walk-forward sampling from historical OHLCV
   
2. Candidate Generation
   └─> ALL local extrema (scipy.signal, no filtering)
   
3. Feature Generation  
   └─> 300-500 raw features across all windows/scales
   
4. Target Generation
   └─> 50-100 outcome targets across all horizons
   
5. Feature Selection
   └─> LGBM+SHAP importance ranking
   
6. Target Selection
   └─> AutoML: train on all targets, select best by R²
   
7. Hyperparameter Optimization
   └─> 200 trials of Optuna/HPO
   
8. Final Training
   └─> LGBM with SHAP analysis
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Candidate Generator | `candidate_level_generator.py` | Generate ALL local extrema |
| Feature Generator | `raw_feature_generator.py` | 300-500 raw transformations |
| Target Generator | `outcome_target_generator.py` | 50-100 possible targets |
| Data Collector | `sr_data_collector.py` | Walk-forward data collection |
| Feature Selector | `lgbm_shap_feature_selector.py` | SHAP-based selection |
| Target Selector | `multi_target_automl.py` | AutoML target discovery |
| HPO Trainer | `hpo_trainer.py` | Hyperparameter optimization |
| Orchestrator | `fully_data_driven_trainer.py` | Full pipeline |

## Usage

### Quick Start

```python
from fully_data_driven_trainer import FullyDataDrivenSRSystem

# Initialize
system = FullyDataDrivenSRSystem()

# Train from scratch
results = system.train_from_scratch(
    symbol='ETHUSDT',
    exchange='binance',
    timeframe='1h',
    start_date='2023-01-01',
    end_date='2024-01-01',
    n_features=50,
    sample_every_n_bars=10
)

# Access results
model = results['model']
explainer = results['explainer']
selected_features = results['selected_features']
best_target = results['best_target']
```

### Command Line

```bash
# Train on ETHUSDT 1h data
python demo_train.py --symbol ETHUSDT --exchange binance --timeframe 1h

# Train on BTCUSDT with custom parameters
python demo_train.py \
    --symbol BTCUSDT \
    --exchange binance \
    --timeframe 4h \
    --start-date 2022-01-01 \
    --end-date 2024-01-01 \
    --n-features 40 \
    --sample-every 20
```

## Features Generated

### Distance Features (36 per level)
- Distance from current close (6 windows)
- Distance from mean/median/std (6 windows each)
- Min/max distances (6 windows each)

### Crossing Features (12 per level)
- Crossing counts (6 windows)
- Crossing rates (6 windows)

### Time-at-Level Features (36 per level)
- Time at level for 3 tolerances × 6 windows
- Time-at-level rates (3 × 6)

### Volume Features (85 per level)
- Volume near level (3 distances × 5 windows)
- Volume near pct (3 × 5)
- Volume statistics: mean/std/median/min/max (5 × 5)
- Volume skew/kurt (2 × 5)

### Price Statistics (66 per level)
- Return moments: mean/std/skew/kurt (4 × 6)
- Range statistics: mean/std/median (3 × 6)
- Close statistics: mean/std/skew/kurt (4 × 6)

### Volatility Features (30 per level)
- ATR: mean/std/median/max (4 × 5)
- Volatility ratio/normalized (2 × 5)

### Interaction Features (16 per level)
- Distance/volume/crossing/volatility ratios (4 pairs × 4 types)

**Total: ~300-400 features per level**

## Targets Generated

### Price Reactions (25 per level)
- Max up/down/net/abs move (5 windows × 5 metrics)

### Touch Behavior (20 per level)
- Touch count/binary/rate/bars_to_touch (5 windows × 4)

### Reversals (15 per level)
- Reversal magnitude/direction/strength (5 windows × 3)

### Breakouts (45 per level)
- Binary/direction/time for 3 thresholds (5 windows × 3 thresholds × 3)

### Volatility/Volume Changes (30 per level)
- Vol change/spike, volume surge/spike (5 windows × 6)

**Total: ~100-135 possible targets**

## Output

### Models
```
models/sr_ml/
├── sr_ml_ETHUSDT_binance_1h_20241102_120000_model.txt
├── sr_ml_ETHUSDT_binance_1h_20241102_120000_metadata.json
├── sr_ml_ETHUSDT_binance_1h_20241102_120000_features.json
└── sr_ml_ETHUSDT_binance_1h_20241102_120000_target_analysis.json
```

### Visualizations
```
outputs/sr_ml/
├── shap/
│   ├── sr_ml_*_summary.png           # Global feature importance
│   ├── sr_ml_*_bar.png               # Mean |SHAP| values
│   ├── sr_ml_*_dependence_*.png      # Feature interaction plots
│   └── sr_ml_*_force_*.png           # Individual prediction explanations
└── performance/
    ├── sr_ml_*_scatter.png           # Prediction vs actual
    ├── sr_ml_*_residuals.png         # Residual analysis
    ├── sr_ml_*_distributions.png     # Distribution comparison
    └── sr_ml_*_metrics.json          # All metrics
```

## Example Results

The system discovers optimal settings purely from data:

- **Best Target**: Automatically selected from 100+ candidates (e.g., `max_up_20` or `break_binary_10_2pct`)
- **Best Features**: Top 50 selected by SHAP (e.g., `crosses_20`, `vol_mean_50`, `atr_10`)
- **Best Hyperparameters**: Optimized via 200 HPO trials
- **Interpretability**: Full SHAP analysis showing what drives predictions

## Integration Points

### Artifact Manager
Training data is saved using the artifact manager:
```python
from src.training.steps.pre_training.utils.artifact_manager import artifact_context

with artifact_context(symbol=symbol, exchange=exchange, information="sr_ml_training") as am:
    am.create_joint_parquet_file(...)
```

### Historical Data
Loads from processed OHLCV data:
```
historical_data/binance/ethusdt/processed/ethusdt_1h/
```

### HPO Utils
Uses existing optimization infrastructure:
```python
from src.utils.ml_common.optimization.hpo_utils import optimize_hyperparameters
```

## Performance Expectations

- **Data Collection**: ~5-10 min for 1 year of 1h data
- **Feature Selection**: ~5-10 min (5-fold CV on 300+ features)
- **Target Selection**: ~10-30 min (100+ targets × 5-fold CV)
- **HPO**: ~30-60 min (200 trials)
- **Total**: ~1-2 hours for complete training

## Key Insights

The 100% data-driven approach reveals:

1. **Which price levels matter**: Model learns from local extrema which ones actually predict future behavior
2. **Which features matter**: SHAP shows dominant features (often crossing counts and volume statistics)
3. **Which targets are learnable**: AutoML discovers most predictive outcomes (often 10-20 bar horizons)
4. **Optimal complexity**: HPO finds right balance between overfitting and underfitting

## Future Enhancements

Potential extensions (all data-driven):

- [ ] Multi-timeframe fusion (learn optimal timeframe weights)
- [ ] Ensemble models (learn when to trust each model)
- [ ] Online learning (update model with new data)
- [ ] Custom loss functions (learn from trading P&L)
- [ ] Meta-features (learn feature engineering from data)

## Citation

```
100% Data-Driven SR Level ML System
Zero Heuristics - Pure Machine Learning
Uses: LGBM, SHAP, Optuna, scipy.signal
```

