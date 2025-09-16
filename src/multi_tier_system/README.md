# Multi-Tier Trading System

A sophisticated three-tier trading system that combines HMM regime detection, Analyst decision making, and Tactician timing prediction for optimal trading execution.

## System Architecture

### Tier 1: HMM Regime Detection (1h base, runs every 15 minutes)
- **Purpose**: Identifies market regimes based on momentum, volatility, and volume
- **Features**: 100 technical indicators and cross-timeframe features
- **Output**: Probabilities for 15-25 market regimes
- **Models**: CatBoost, Elastic Net (base) + XGBoost (meta-learner)

### Tier 2: Analyst Decision Making (5m base, runs every 2 minutes)
- **Purpose**: Decides IF we should trade based on comprehensive market analysis
- **Features**: 300+ features including cross-timeframe analysis and HMM outputs
- **Training**: Per-regime training for regime-specific decision making
- **Output**: Green light/Red light for trading opportunities
- **Models**: TCN, CatBoost, LightGBM (base) + Elastic Net (meta-learner)

### Tier 3: Tactician Timing Prediction (1m base, runs every 30 seconds)
- **Purpose**: Decides WHEN to trade when Analyst gives green light
- **Features**: 50+ high-frequency timing features
- **Training**: All regimes but only on green light periods
- **Output**: Entry timing, position sizing, and risk assessment
- **Models**: XGBoost, Random Forest, CatBoost, Elastic Net (base) + LightGBM (meta-learner)

## Key Features

### Comprehensive Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, etc.
- **Volume Analysis**: OBV, VPT, MFI, Volume ratios, Volume-price correlation
- **Cross-Timeframe Features**: Multi-timeframe momentum, volatility, and volume analysis
- **Market Microstructure**: Bid-ask spread, price impact, intraday volatility
- **Pattern Recognition**: Candlestick patterns, support/resistance levels
- **Regime Integration**: HMM regime probabilities and characteristics

### Advanced Model Architecture
- **Base Models**: Multiple specialized models for each tier
- **Meta-Learning**: Ensemble learning to combine base model predictions
- **Regime-Specific Training**: Models trained on specific market conditions
- **Feature Selection**: Automatic selection of most relevant features
- **Cross-Validation**: Robust validation across different market regimes

### Intelligent Scheduling
- **HMM**: Runs every 15 minutes on 1-hour data
- **Analyst**: Runs every 2 minutes on 5-minute data
- **Tactician**: Runs every 30 seconds on 1-minute data
- **Coordination**: Systems work together with proper data flow

### Risk Management
- **Position Sizing**: Dynamic position sizing based on risk assessment
- **Leverage Control**: Adaptive leverage based on market conditions
- **Drawdown Protection**: Risk scoring and drawdown monitoring
- **Confidence Thresholds**: Multi-tier confidence requirements

## Usage

### Basic Usage

```python
from src.multi_tier_system import create_multi_tier_trading_orchestrator
import pandas as pd

# Load market data
data_1h = pd.read_csv('data_1h.csv', index_col=0, parse_dates=True)
data_5m = pd.read_csv('data_5m.csv', index_col=0, parse_dates=True)
data_1m = pd.read_csv('data_1m.csv', index_col=0, parse_dates=True)

# Create orchestrator
orchestrator = create_multi_tier_trading_orchestrator()

# Load data
orchestrator.load_data(data_1h, data_5m, data_1m)

# Train systems
orchestrator.train_systems()

# Start trading
orchestrator.start_system()

# Run single cycle
decision = orchestrator.run_single_cycle()
if decision and decision.should_trade:
    print(f"Trade signal: {decision.decision_reasoning}")
```

### Configuration

```python
config = {
    'hmm': {
        'n_regimes': 20,
        'n_features': 100,
        'run_interval_minutes': 15
    },
    'analyst': {
        'n_features': 300,
        'run_interval_minutes': 2,
        'target_threshold': 0.5
    },
    'tactician': {
        'run_interval_seconds': 30,
        'max_position_size': 0.1,
        'max_leverage': 3.0
    }
}

orchestrator = create_multi_tier_trading_orchestrator(config)
```

### Individual System Usage

```python
# HMM System
from src.hmm_system import create_hmm_regime_detector

hmm_system = create_hmm_regime_detector()
hmm_system.train_models(data_1h)
regime_probs = hmm_system.predict_regime_probabilities(data_1h)

# Analyst System
from src.analyst_system import create_analyst_regime_predictor

analyst_system = create_analyst_regime_predictor()
analyst_system.train_regime_models(data_5m, regime_labels)
analyst_prediction = analyst_system.predict_trading_opportunity(data_5m, regime_id)

# Tactician System
from src.tactician_system import create_tactician_timing_predictor

tactician_system = create_tactician_timing_predictor()
tactician_system.train_models(data_1m, green_lights)
tactician_prediction = tactician_system.predict_entry_timing(data_1m)
```

## Data Requirements

### Required Columns
All dataframes must contain:
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

### Timeframe Requirements
- **1h data**: For HMM regime detection
- **5m data**: For Analyst decision making
- **1m data**: For Tactician timing prediction

### Data Quality
- No missing values in OHLCV data
- Proper datetime indexing
- Sufficient historical data for training (recommended: 1000+ bars per timeframe)

## Model Training

### HMM Training
1. Extracts 100 features from 1h data
2. Tests different numbers of regimes (15-25)
3. Selects optimal number based on BIC criteria
4. Trains Gaussian Mixture Model for regime detection

### Analyst Training
1. Extracts 300+ features from 5m data
2. Integrates HMM regime outputs
3. Trains separate models for each regime
4. Uses meta-learner to combine base model predictions

### Tactician Training
1. Extracts 50+ timing features from 1m data
2. Integrates HMM and Analyst outputs
3. Trains only on periods where Analyst gave green light
4. Focuses on optimal entry timing for 0.5% price changes

## Performance Monitoring

### System Metrics
- HMM runs, Analyst runs, Tactician runs
- Green lights generated, Trade signals emitted
- System uptime, Average processing time
- Error count and error rate

### Decision Tracking
- Complete decision history with timestamps
- Confidence scores for each tier
- Market conditions and reasoning
- Performance metrics over time

### Export Capabilities
- Model persistence (save/load)
- Metrics export to JSON
- Decision history export
- Performance reports

## Configuration Options

### HMM Configuration
- `n_regimes`: Number of regimes to detect (15-25)
- `n_features`: Number of features to use (100)
- `run_interval_minutes`: How often to run (15)
- `lookback_periods`: Historical data to use (24 hours)

### Analyst Configuration
- `n_features`: Number of features to use (300+)
- `target_threshold`: Price change threshold (0.5%)
- `run_interval_minutes`: How often to run (2)
- `cross_timeframe_periods`: Additional timeframes to analyze

### Tactician Configuration
- `run_interval_seconds`: How often to run (30)
- `max_position_size`: Maximum position size (0.1)
- `max_leverage`: Maximum leverage (3.0)
- `min_confidence_threshold`: Minimum confidence to trade (0.6)

## Error Handling

### Robust Error Management
- Graceful degradation when systems fail
- Automatic retry mechanisms
- Error logging and monitoring
- System status tracking

### Data Validation
- Input data validation
- Feature extraction error handling
- Model prediction validation
- Output consistency checks

## Dependencies

### Core Dependencies
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning models
- xgboost, lightgbm, catboost: Gradient boosting
- joblib: Model persistence

### Optional Dependencies
- tensorflow/keras: For TCN implementation
- ta-lib: Additional technical indicators
- plotly: Visualization (if needed)

## Examples

See `example_usage.py` for comprehensive examples including:
- Individual system usage
- Full pipeline training
- Live trading simulation
- Configuration examples
- Error handling demonstrations

## Performance Considerations

### Computational Efficiency
- Vectorized operations where possible
- Efficient feature extraction
- Model caching and persistence
- Optimized data structures

### Memory Management
- Streaming data processing
- Efficient data storage
- Garbage collection optimization
- Memory usage monitoring

### Scalability
- Modular architecture
- Configurable parameters
- Horizontal scaling support
- Performance profiling tools

## Future Enhancements

### Planned Features
- Real-time data integration
- Advanced ensemble methods
- Dynamic regime adaptation
- Enhanced risk management
- Performance optimization
- Additional technical indicators
- Machine learning model updates
- Advanced visualization tools

### Research Areas
- Alternative regime detection methods
- Deep learning integration
- Reinforcement learning for timing
- Multi-asset correlation analysis
- Advanced feature engineering
- Model interpretability
- Backtesting framework
- Live trading integration