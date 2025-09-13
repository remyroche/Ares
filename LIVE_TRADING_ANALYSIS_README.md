# Live Trading Analysis with ML Models

This guide explains how to fetch market data every 30 seconds and analyze it with your trained ML models for live trading decisions.

## 🎯 Overview

The **LiveDataCollector** system provides a production-ready solution for real-time market data collection and ML-powered analysis. It fetches market data every 30 seconds, processes it with your existing feature engineering pipeline, and generates ML predictions for trading decisions.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Exchange API  │───▶│ LiveDataCollector│───▶│   ML Analysis   │
│   (Binance)     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Trading Signals │
                       │   & Orders      │
                       └─────────────────┘
```

## 🚀 Key Features

### ✅ **30-Second Data Collection**
- Automated fetching every 30 seconds
- Precise timing with drift correction
- Error recovery and reconnection

### ✅ **ML Model Integration**
- Load your trained sklearn models
- Real-time feature engineering
- Prediction probability scoring

### ✅ **Real-Time Feature Engineering**
- Uses your existing `FeatureEngineeringOrchestrator`
- Rolling calculations (volatility, volume MA, returns)
- Compatible with all your trained features

### ✅ **Production-Ready Components**
- Memory-optimized processing
- Comprehensive error handling
- Async/await architecture
- Callback-based event system

### ✅ **Trading Integration**
- Position management
- Risk controls (stop loss, take profit)
- Performance tracking
- Signal-based entry/exit

## 📋 Prerequisites

1. **Trained ML Model**: Your model should be saved as a pickle file (`.pkl`)
2. **Exchange API Keys**: Configure Binance API credentials
3. **Feature Engineering**: Compatible with your existing feature pipeline

## 🛠️ Setup

### 1. Configure API Keys

Update your `config/config.yaml`:

```yaml
exchanges:
  binance:
    api_key: "your_api_key_here"
    api_secret: "your_api_secret_here"
```

### 2. Prepare Your ML Model

Ensure your trained model is saved and accessible:

```python
import joblib

# Load your trained model
model = joblib.load('models/your_model.pkl')

# Verify it has predict and predict_proba methods
print(hasattr(model, 'predict'))      # Should be True
print(hasattr(model, 'predict_proba')) # Should be True for confidence scores
```

### 3. Model Feature Compatibility

Your ML model should expect features that match your training data. The system will:

- Extract basic features: `close`, `volume`, `returns`, `volatility`, `volume_ma`
- Apply your existing feature engineering pipeline
- Ensure feature order matches training

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from src.live_trading.live_data_collector import start_live_collection

async def main():
    # Start live collection with ML model
    collector = await start_live_collection(
        symbol="ETH",
        exchange="binance",
        interval=CollectionInterval.STANDARD,  # 30 seconds
        ml_model_path="models/your_model.pkl"
    )

    # Let it run for a while
    await asyncio.sleep(300)  # 5 minutes

    # Stop collection
    await collector.stop_collection()

asyncio.run(main())
```

### Advanced Usage with Callbacks

```python
from src.live_trading.live_data_collector import LiveDataCollector, LiveDataConfig, CollectionInterval

async def on_new_data(data_point):
    """Process new data with ML predictions."""
    raw_data = data_point.raw_data
    ml_predictions = data_point.ml_predictions

    print(f"Price: ${raw_data['close']:.2f}")

    if ml_predictions:
        prediction = ml_predictions['prediction']
        confidence = ml_predictions['confidence']
        print(f"ML Signal: {prediction} (confidence: {confidence:.1f})")

async def main():
    # Create custom configuration
    config = LiveDataConfig(
        symbol="ETH",
        exchange="binance",
        interval=CollectionInterval.STANDARD,  # 30 seconds
        enable_ml_predictions=True,
        ml_model_path="models/your_model.pkl",
        feature_engineering=True,
        quality_level=DataQuality.HIGH
    )

    collector = LiveDataCollector(config)
    collector.add_data_callback(on_new_data)

    # Start collection
    await collector.start_collection()

    # Run for specified time
    await asyncio.sleep(3600)  # 1 hour

    await collector.stop_collection()

asyncio.run(main())
```

## 📊 Data Flow

1. **Fetch**: Get latest 1-minute kline from exchange
2. **Process**: Calculate returns, volatility, volume metrics
3. **Engineer**: Apply your full feature engineering pipeline
4. **Predict**: Run ML model for trading signals
5. **Callback**: Trigger your analysis/trading functions
6. **Buffer**: Maintain rolling window of recent data

## 🤖 ML Integration Details

### Model Loading
```python
# The system automatically loads sklearn-compatible models
self.ml_model = joblib.load(model_path)

# Supports: RandomForest, XGBoost, SVM, etc.
# Requires: predict() and predict_proba() methods
```

### Feature Preparation
```python
# Basic features extracted automatically
features = [
    data['close'],
    data['volume'],
    data['returns'],      # Price change %
    data['volatility'],   # Rolling volatility
    data['volume_ma'],    # Volume moving average
]

# Plus all features from your engineering pipeline
```

### Prediction Output
```python
ml_predictions = {
    'prediction': 1,        # 1 = bullish, 0 = bearish
    'probabilities': [0.3, 0.7],  # [bearish_prob, bullish_prob]
    'confidence': 0.7,      # Max probability
    'model_type': 'RandomForestClassifier'
}
```

## 📈 Trading Example

See `example_live_trading_analysis.py` for a complete trading bot example that:

- Fetches data every 30 seconds
- Applies ML predictions
- Manages positions with stop-loss/take-profit
- Tracks performance metrics
- Provides real-time feedback

### Key Trading Logic

```python
# Entry conditions
if prediction == 1 and confidence > 0.7:  # Bullish signal
    enter_long_position()

# Exit conditions
if current_price <= stop_loss or current_price >= take_profit:
    exit_position()

if prediction == 0 and confidence > 0.7:  # Bearish reversal
    exit_position()
```

## ⚙️ Configuration Options

### Collection Modes
```python
class CollectionMode(Enum):
    LIVE = "live"          # Real exchange data
    SIMULATED = "simulated" # Historical replay for testing
    HYBRID = "hybrid"      # Mix of live + cached data
```

### Quality Levels
```python
class DataQuality(Enum):
    HIGH = "high"      # Full processing + validation
    MEDIUM = "medium"  # Basic processing only
    LOW = "low"        # Minimal processing for speed
```

### Advanced Config
```python
config = LiveDataConfig(
    symbol="ETH",
    exchange="binance",
    interval_seconds=30,
    collection_mode=CollectionMode.LIVE,
    quality_level=DataQuality.HIGH,
    buffer_size=1000,           # Keep last 1000 data points
    enable_ml_predictions=True,
    ml_model_path="models/model.pkl",
    feature_engineering=True,
    error_recovery=True
)
```

## 🔍 Monitoring & Debugging

### Collection Statistics
```python
stats = collector.get_stats()
print(f"""
Collection Stats:
- Running: {stats['is_running']}
- Data points collected: {stats['collection_count']}
- Errors: {stats['error_count']}
- Buffer size: {stats['buffer_size']}
- ML enabled: {stats['ml_predictions_enabled']}
- Avg processing time: {stats['avg_processing_time_ms']:.2f}ms
""")
```

### Recent Data Access
```python
# Get last 100 data points
recent_data = collector.get_recent_data(100)

# Get as DataFrame for analysis
df = collector.get_processed_data_df(100)
print(df.head())
```

### Logging
The system integrates with your existing logging framework:
- Collection events every 10 data points
- ML prediction results
- Error recovery attempts
- Performance metrics

## 🛡️ Error Handling

### Automatic Recovery
- Network timeouts → Retry with exponential backoff
- API rate limits → Wait and retry
- Invalid data → Skip and continue
- ML prediction failures → Log and continue without predictions

### Manual Error Handling
```python
def handle_error(error: Exception):
    print(f"Collection error: {error}")
    # Implement your error handling logic

collector.add_error_callback(handle_error)
```

## 📊 Performance Considerations

### Memory Usage
- Rolling buffer limits memory usage
- Configurable buffer sizes (default: 1000 points)
- Memory-optimized feature engineering

### Timing
- Precise 30-second intervals
- Sub-100ms processing time typical
- Async architecture prevents blocking

### Rate Limits
- Respects exchange API limits
- Automatic rate limit handling
- Configurable retry strategies

## 🔧 Integration with Existing System

### Feature Engineering
The system automatically integrates with your existing:
- `FeatureEngineeringOrchestrator`
- Custom technical indicators
- Data validation pipelines

### Exchange Integration
Works with your current exchange setup:
- Binance API integration
- Existing authentication
- Standard market data formats

### ML Pipeline
Compatible with your training pipeline:
- Same feature engineering
- Same model formats
- Same prediction interfaces

## 🧪 Testing

### Simulated Mode
```python
config = LiveDataConfig(
    collection_mode=CollectionMode.SIMULATED,
    # ... other config
)
```

### Backtesting Integration
Use the same analysis logic for backtesting:
```python
# Run on historical data
historical_data = load_historical_data()
analyzer = LiveTradingAnalyzer()
for data_point in historical_data:
    await analyzer.on_new_data(data_point)
```

## 🚨 Important Notes

1. **Paper Trading First**: Always test with paper trading before live execution
2. **Model Validation**: Ensure your ML model performs well on recent data
3. **Risk Management**: Implement proper position sizing and risk controls
4. **Monitoring**: Set up alerts for collection failures or unusual ML predictions
5. **Data Quality**: Monitor for data gaps or anomalies in live feeds

## 📞 Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify ML model compatibility
3. Test with simulated mode first
4. Ensure exchange API credentials are correct
5. Monitor system resource usage

---

**🎯 This live trading analysis system bridges your ML training pipeline with real-time market data, enabling data-driven trading decisions every 30 seconds.**
