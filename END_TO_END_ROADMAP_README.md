# End-to-End Roadmap System

## Overview

This document describes the complete end-to-end roadmap system that replaces the PID-driven generation feature. The system implements a fully-specified, production-ready feature engineering pipeline with strict budgets, validation, monitoring, and deployment procedures.

## System Architecture

### Core Components

1. **System Contracts** (`config/end_to_end_roadmap_config.yaml`)
   - Feature budgets: pre-selection ≤120, post-selection 30-60
   - Latency budgets: total ≤50ms, feature compute ≤25ms, model ≤5ms
   - Lookback ceiling: 120 minutes
   - Retrain schedule: 02:00 ET daily

2. **Data Contracts** (`src/feature_engineering/data_contracts.py`)
   - Input bars with exchange calendar alignment
   - Feature store with registry paths
   - Artifacts registry for reproducibility

3. **Feature Registry** (`src/feature_engineering/feature_registry.py`)
   - 30+ parent features across 6 families
   - Exact formulas and metadata
   - Feature gates for validation

4. **Transform System** (`src/feature_engineering/transforms.py`)
   - EW-Z (stateful online) for continuous features
   - TOD Rank for seasonal features
   - Signed-log for heavy tails
   - Winsorization for outlier handling

5. **Lookback Selection** (`src/feature_engineering/lookback_selection.py`)
   - Tiny menus (3-4 options per family)
   - Hysteresis-based selection
   - Walk-forward validation

6. **Interaction Engine** (`src/feature_engineering/interactions.py`)
   - 15 locked interactions with theory-first approach
   - Regime flags for conditional interactions
   - Availability guards for missing data

7. **Patch/GRU Model** (`src/models/patch_gru.py`)
   - Minimal stacker with confidence estimation
   - 2-4h sequence lookback
   - p99 inference <5ms

8. **Assembly DAG** (`src/feature_engineering/assembly_dag.py`)
   - Complete pipeline orchestration
   - Calendar sessionization
   - Feature selection and assembly

9. **Validation System** (`src/validation/walkforward_validation.py`)
   - Walk-forward with nested CV
   - Embargo logic
   - Ablation ladder testing
   - SPA/reality check

10. **Monitoring & Retrain** (`src/monitoring/retrain_monitoring.py`)
    - Calibration monitoring
    - PSI drift detection
    - Correlation drift monitoring
    - Retrain decision tree

11. **CI/CD Gates** (`src/ci/validators.py`)
    - Budget validation
    - Latency harness
    - Golden replay
    - Unit tests

12. **Rollout Plan** (`src/deployment/rollout_plan.py`)
    - Shadow mode (1-2 sessions)
    - Canary deployment (10-20% risk)
    - Full deployment with fallback

## Feature Families

### Price/Returns (10 features)
- `p/r1`, `p/r3`, `p/r5`, `p/r10`: Log returns
- `p/mom5`, `p/mom10`, `p/mom20`: Momentum
- `p/price_ema10_pct`, `p/price_ema20_pct`: EMA percentages
- `p/bollz20`: Bollinger z-score

### Volatility (6 features)
- `p/sigma_ew`: EW standard deviation
- `p/gk_w`: Garman-Klass estimator
- `p/rv_bipower_12`: Bipower variation
- `p/rv_short_3`: Short-term realized volatility
- `p/sigma_slope_6`: Volatility slope
- `p/range_pct`: Range percentage

### Mean Reversion (4 features)
- `p/rsi7`, `p/rsi14`: RSI indicators
- `p/stochk14`: Stochastic %K
- `p/autocorr_r1_w`: Return autocorrelation

### Liquidity/Micro (6 features, book-optional)
- `p/volume_z18`: Volume z-score
- `p/tradecount_z18`: Trade count z-score
- `p/spread_z18`: Spread z-score
- `p/dollarvol_z18`: Dollar volume z-score
- `p/ofi_proxy`: Order flow imbalance
- `p/microprice_dev`: Microprice deviation

### Anchors & TOD (4 features)
- `p/vwap_session_dist`: Session VWAP distance
- `p/vwap_roll12_dist`: Rolling VWAP distance
- `p/open30`: First 30 minutes flag
- `p/last30`: Last 30 minutes flag

### Context (2 features, optional)
- `p/beta30`: Rolling beta to index
- `p/mkt_dispersion`: Market dispersion

## Transform Types

### EW-Z (Default for continuous features)
- Stateful online exponential weighted z-score
- Halflife options: {9, 12, 18}
- Maintains state for live parity

### TOD Rank (For seasonal features)
- Time-of-day percentile ranking
- 48 buckets @ 30-min granularity
- EW histogram per bucket

### Signed-Log (For heavy tails)
- `slog(x) = sign(x) * log(1 + |x|)`
- Used for spread, OFI, microprice features

### Winsorization (Post-transform)
- Clip to train quantiles [0.1%, 99.9%]
- Frozen bounds per retrain

## Interaction Engine

### 15 Locked Interactions

#### Tension (4 interactions)
- `i/tension/mom5_x_negmom20`: Short vs long momentum
- `i/tension/rsi14_x_highvol`: RSI in high volatility
- `i/tension/bollz_x_widespread`: Bollinger in wide spread
- `i/tension/vwapdist_x_open30`: VWAP distance at open

#### Microstructure (4 interactions)
- `i/micro/ofi_x_spread`: OFI vs spread
- `i/micro/tradecount_x_spread`: Trade count vs spread
- `i/micro/microprice_x_ofi`: Microprice vs OFI
- `i/micro/dollarvol_x_widespread`: Dollar volume in wide spread

#### Volatility (4 interactions)
- `i/vol/r1_x_rvshort`: 1-bar return vs short volatility
- `i/vol/r3_x_rvshort`: 3-bar return vs short volatility
- `i/vol/vwapdist_x_rvshort`: VWAP distance vs short volatility
- `i/vol/autocorr_x_rvshort`: Autocorrelation vs short volatility

#### Model (3 interactions)
- `i/model/yhat1_x_rvshort`: Model prediction vs short volatility
- `i/model/yhat1_x_vwapdist`: Model prediction vs VWAP distance
- `i/model/yhatconf_x_widespread`: Model confidence in wide spread

## Usage

### Basic Usage

```python
from src.end_to_end_roadmap import run_end_to_end_pipeline

# Load market data
bars = pd.read_csv('market_data.csv')

# Run complete pipeline
result = run_end_to_end_pipeline(
    bars=bars,
    targets=targets,  # Optional
    enable_validation=True,
    enable_monitoring=True,
    enable_deployment=False
)

if result.success:
    print(f"Generated {len(result.features.columns)} features")
    print(f"Selected {len(result.selected_features)} features")
else:
    print(f"Pipeline failed: {result.error_message}")
```

### Component Integration

```python
from src.training.steps.pre_training.end_to_end_roadmap_generation import EndToEndRoadmapComponent

# Create component
component = EndToEndRoadmapComponent(config)

# Execute
result = await component.execute(market_data, pipeline_state)
```

### Configuration

```python
from src.end_to_end_roadmap import SystemConfig

config = SystemConfig(
    feature_budget_pre=120,
    feature_budget_post=(30, 60),
    interactions_cap=15,
    latency_budget_ms=50,
    lookback_ceiling_minutes=120
)
```

## Validation

### Walk-Forward Validation
- K chronological folds with embargo
- Nested CV for hyperparameter selection
- Simplicity prior with hysteresis

### Ablation Ladder
1. Parents only
2. + Transforms
3. + Patch features
4. + 8 interactions
5. + 15 interactions

### SPA Test
- Stepwise Superior Predictive Ability
- 1000 permutations
- Data-snooping protection

## Monitoring

### Key Metrics
- **Calibration Loss**: MSE/Brier by session bucket
- **PSI**: Population Stability Index for drift
- **Correlation Drift**: Frobenius norm of correlation matrices
- **Latency**: p95/p99 per component

### Retrain Triggers
- Calibration loss > 2σ for 3 hours
- PSI > 0.3 on key features
- p99 latency > 50ms
- Scheduled daily at 02:00 ET

## Deployment

### Shadow Mode (1-2 sessions)
- Full logging, no trades
- Validate system behavior
- Check latency and accuracy

### Canary (10-20% risk, 1 session)
- Limited risk exposure
- Monitor performance metrics
- Automatic rollback on issues

### Full Deployment
- Enable retrain triggers
- Automatic fallback model
- Complete monitoring suite

## CI/CD

### Build-Time Validations
- Feature count budgets
- Lookback ceiling compliance
- Transform type validation
- Latency budget checks

### Unit Tests
- Session VWAP reset validation
- DST/half-day flag computation
- EW-Z state continuity
- Missing book data handling

### Golden Replay
- Bit-for-bit reproduction
- Hash-based validation
- Reference data storage

## Performance

### Latency Budgets
- **Total**: ≤50ms p99
- **Feature Compute**: ≤25ms p95
- **Model Inference**: ≤5ms p99
- **I/O & Orchestration**: ≤20ms p95

### Feature Budgets
- **Pre-selection**: ≤120 features
- **Post-selection**: 30-60 features (target 45)
- **Interactions**: ≤15 total
- **Transforms per parent**: ≤1

### Lookback Ceiling
- **Maximum**: 120 minutes
- **Enforced**: CI/CD validation
- **Monitoring**: Real-time checks

## Error Handling

### Graceful Degradation
- Fallback model (depth-2 LGBM)
- Auto-drop missing book features
- Size by confidence clipping

### Monitoring Alerts
- Critical: System halt required
- Warning: Monitor closely
- Info: Normal operation

## Extensions

### Cross-Sectional Features
- Per-timestamp cross-sectional ranks
- Market dispersion interactions
- Universe-wide feature selection

### Event-Time Volatility
- Trade-count normalized RV
- Tick-based microprice updates
- High-frequency regime detection

## Common Pitfalls Avoided

1. **Overfitting**: Menus ≤4 options, nested CV, hysteresis
2. **Leakage**: Session VWAP reset, causal computation
3. **Latency Creep**: p95/p99 timers, CI harness, fallback
4. **Transform Drift**: Online state, frozen bounds
5. **Interaction Explosion**: Locked 15, theory-first
6. **Selection Bias**: No SHAP mining, theory-first only

## Migration from PID System

The new system completely replaces the PID-driven generation feature with:

1. **Structured Approach**: Clear contracts and budgets
2. **Production Ready**: Monitoring, validation, deployment
3. **Maintainable**: Modular design, comprehensive testing
4. **Scalable**: Efficient algorithms, hardware optimization
5. **Robust**: Error handling, graceful degradation

## Support

For questions or issues:
1. Check the validation logs
2. Review monitoring metrics
3. Consult the CI/CD test results
4. Examine the rollout status

The system is designed to be self-documenting with comprehensive logging and monitoring throughout the pipeline.