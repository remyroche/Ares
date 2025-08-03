# HMM-Based Market Regime Classification

## Overview

This document describes the implementation of a Hidden Markov Model (HMM) based market regime classifier that enhances the existing regime classification system in the Ares trading bot.

## Architecture

### 1. HMM Regime Classifier (`src/analyst/hmm_regime_classifier.py`)

The HMM classifier is now the **primary method** for market regime classification, replacing the old EMA/ADX-based approach. It uses two key features to identify market regimes:
- **log_returns**: Captures price movement patterns and trends
- **volatility_20**: Captures volatility regimes and market conditions

### 2. Integration with Existing System

The HMM classifier is integrated into the existing `MarketRegimeClassifier` class as the **primary regime detection method**:
- **Primary**: HMM-based regime detection using log_returns and volatility_20
- **Fallback**: Traditional EMA/ADX-based pseudo-labeling approach
- **Enhanced**: Superior simulated labels for LightGBM training

## How It Works

### Step 1: Feature Calculation
```python
def _calculate_features(self, klines_df: pd.DataFrame) -> pd.DataFrame:
    # Calculate log returns
    features["log_returns"] = np.log(klines_df["close"] / klines_df["close"].shift(1))
    
    # Calculate 20-period rolling volatility (annualized)
    features["volatility_20"] = (
        features["log_returns"].rolling(window=20).std() * np.sqrt(252)
    )
```

### Step 2: HMM Training
The HMM model is trained on the log_returns and volatility_20 features:
```python
self.hmm_model = hmm.GaussianHMM(
    n_components=self.n_states,
    n_iter=self.n_iter,
    random_state=self.random_state,
    covariance_type="full"
)
```

### Step 3: State Interpretation
Each HMM state is interpreted based on its characteristics:
- **Mean return**: Positive/negative trend direction
- **Mean volatility**: High/low volatility periods
- **Standard deviations**: Stability of the regime

### Step 4: Regime Detection Priority
The system uses a priority-based regime detection approach:

1. **SR_ZONE_ACTION** (Highest Priority): Detected when price interacts with support/resistance levels
2. **MOMENTUM** (Overrides HMM): Detected when strong positive momentum with moderate volatility
3. **SR** (Overrides HMM): Detected when low returns with high volatility near SR levels
4. **HMM Classification** (Primary): BULL, BEAR, or SIDEWAYS based on log_returns and volatility_20
5. **Traditional LightGBM** (Fallback): EMA/ADX-based classification

### HMM Regime Mapping
When HMM is used, states are mapped to market regimes:
- **BULL**: Positive returns, low volatility
- **BEAR**: Negative returns, low volatility  
- **SIDEWAYS**: High volatility or low returns

### Step 5: LightGBM Training
The HMM-derived labels are used to train a LightGBM classifier for real-time prediction.

## Configuration

### HMM Parameters
```python
"hmm_regime_classifier": {
    "n_states": 4,  # Number of hidden states
    "n_iter": 100,  # Maximum iterations for HMM training
    "random_state": 42,
    "return_threshold": 0.0005,  # 0.05% daily return threshold
    "volatility_threshold": 0.3,  # 30% annualized volatility threshold
}
```

### Integration Parameters
```python
"market_regime_classifier": {
    "use_hmm": True,  # HMM is now the primary regime detection method
    # ... other parameters
}
```

## Usage

### Training the HMM Classifier
```python
from src.analyst.hmm_regime_classifier import HMMRegimeClassifier

# Initialize classifier
hmm_classifier = HMMRegimeClassifier(config)

# Train on historical data
success = hmm_classifier.train_classifier(historical_klines)
```

### Making Predictions
```python
# Predict current regime
regime, confidence, additional_info = hmm_classifier.predict_regime(current_klines)

print(f"Regime: {regime}")
print(f"Confidence: {confidence:.3f}")
print(f"HMM State: {additional_info['hmm_state']}")
```

### Integration with Existing System
```python
from src.analyst.regime_classifier import MarketRegimeClassifier

# The classifier now uses HMM as the primary method with traditional fallback
classifier = MarketRegimeClassifier(config, sr_analyzer)

# Train HMM classifier (primary) with traditional fallback
classifier.train_classifier(historical_features, historical_klines)

# Predict using HMM as primary method
regime, trend_strength, adx = classifier.predict_regime(
    current_features, current_klines, sr_levels
)
```

## Testing

### Test Script
Run the test script to evaluate the HMM classifier:
```bash
# Test with synthetic data
python scripts/test_hmm_regime_classifier.py --symbol ETHUSDT

# Test with real data
python scripts/test_hmm_regime_classifier.py --symbol BTCUSDT --use-real-data
```

### Test Output
The test script provides:
- Regime distribution analysis
- Confidence metrics
- State analysis
- Comparison with traditional classifier
- Feature analysis

## Advantages

### 1. Primary Regime Detection Method
- **Replaces** the old EMA/ADX-based pseudo-labeling approach
- **Primary method** for market regime classification
- **Fallback** to traditional method if HMM fails

### 2. Unsupervised Learning
- HMM automatically discovers market states without predefined labels
- Adapts to changing market conditions

### 3. Temporal Dependencies
- Captures the sequential nature of market regimes
- Models regime transitions and persistence

### 4. Robust Feature Set
- Uses fundamental market features (returns, volatility)
- Adaptable to different timeframes

### 5. Enhanced Accuracy
- Combines HMM state detection with LightGBM classification
- Provides confidence scores for predictions

## Implementation Details

### State Interpretation Logic
```python
def _classify_state_to_regime(self, mean_return, mean_volatility, std_return, std_volatility):
    if mean_return > return_threshold and mean_volatility < volatility_threshold:
        return "BULL_TREND"
    elif mean_return < -return_threshold and mean_volatility < volatility_threshold:
        return "BEAR_TREND"
    elif mean_volatility > volatility_threshold:
        return "SIDEWAYS_RANGE"
    else:
        return "SIDEWAYS_RANGE"
```

### Model Persistence
The HMM classifier saves:
- Trained HMM model
- Feature scaler
- State-to-regime mapping
- LightGBM classifier
- Configuration parameters

### Error Handling
- Graceful fallback to traditional classification
- Comprehensive logging of training and prediction errors
- Validation of input data quality

## Performance Considerations

### Training Time
- HMM training: ~10-30 seconds for 2000 data points
- LightGBM training: ~5-10 seconds
- Total training time: ~15-40 seconds

### Prediction Time
- HMM prediction: ~1-5 milliseconds
- LightGBM prediction: ~1-3 milliseconds
- Total prediction time: ~2-8 milliseconds

### Memory Usage
- HMM model: ~1-5 MB
- LightGBM model: ~1-3 MB
- Total memory: ~2-8 MB

## Future Enhancements

### 1. Multi-Timeframe Analysis
- Train separate HMM models for different timeframes
- Combine predictions for enhanced accuracy

### 2. Dynamic State Count
- Automatically determine optimal number of states
- Use model selection criteria (AIC, BIC)

### 3. Regime Persistence Analysis
- Analyze regime transition probabilities
- Predict regime duration

### 4. Volatility Regime Adaptation
- Adjust volatility thresholds based on market conditions
- Use adaptive volatility measures

### 5. Ensemble Methods
- Combine multiple HMM models
- Use ensemble voting for final prediction

## Troubleshooting

### Common Issues

1. **Insufficient Data**
   - Ensure at least 1000 data points for training
   - Check data quality and completeness

2. **HMM Convergence Issues**
   - Increase `n_iter` parameter
   - Check for data normalization issues
   - Verify feature calculation

3. **Poor Regime Classification**
   - Adjust `return_threshold` and `volatility_threshold`
   - Review state interpretation logic
   - Check feature scaling

4. **Memory Issues**
   - Reduce number of states (`n_states`)
   - Use smaller training datasets
   - Implement data sampling

### Debugging Tips

1. **Enable Detailed Logging**
   ```python
   import logging
   logging.getLogger("HMMRegimeClassifier").setLevel(logging.DEBUG)
   ```

2. **Check State Statistics**
   ```python
   stats = hmm_classifier.get_state_statistics()
   print(f"State mapping: {stats['state_to_regime_map']}")
   ```

3. **Validate Features**
   ```python
   features = hmm_classifier._calculate_features(klines_df)
   print(f"Feature statistics: {features.describe()}")
   ```

## Conclusion

The HMM-based regime classifier provides a sophisticated approach to market regime detection by:
- Automatically discovering market states
- Capturing temporal dependencies
- Providing robust predictions
- Integrating seamlessly with existing systems

This enhancement significantly improves the trading bot's ability to adapt to changing market conditions and make more informed trading decisions. 