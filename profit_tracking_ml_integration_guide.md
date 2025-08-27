# Profit Tracking ML Integration Guide

## Overview

This guide explains how to integrate the new profit tracking information into your existing machine learning training pipelines. The profit tracking data provides rich information that can significantly enhance model performance and trading strategy optimization.

## Integration Methods

### 1. Sample Weighting in Classification

**Use Case**: Give higher importance to trades with higher profit potential during training.

```python
# Load your labeled data with profit tracking
labeled_data = pd.read_parquet("path/to/labeled_data.parquet")

# Filter out HOLD samples for binary classification
signal_data = labeled_data[labeled_data['label'] != 0].copy()

# Create sample weights based on profit magnitude
sample_weights = np.abs(signal_data['potential_profit_pct']) + 0.001

# Prepare features and labels
X = signal_data[feature_columns]
y = (signal_data['label'] == 1).astype(int)  # Binary classification

# Train with profit-weighted samples
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y, sample_weight=sample_weights)
```

**Benefits**:
- Models focus on high-value trades
- Better learning from profitable patterns
- Improved risk-adjusted performance

### 2. Direct Profit Prediction

**Use Case**: Predict the actual profit/loss percentage for each trade.

```python
# Prepare data for profit regression
X = signal_data[feature_columns]
y_profit = signal_data['potential_profit_pct']

# Train profit prediction model
from sklearn.ensemble import RandomForestRegressor
profit_model = RandomForestRegressor(n_estimators=100)
profit_model.fit(X, y_profit)

# Make predictions
profit_predictions = profit_model.predict(X_new)

# Use predictions for trade filtering
profitable_trades = profit_predictions > 0.01  # Only trades with >1% expected profit
```

**Benefits**:
- Quantify expected profit for each trade
- Filter trades based on profit potential
- Optimize position sizing based on profit expectations

### 3. Multi-Output Prediction

**Use Case**: Predict both trade direction and profit magnitude simultaneously.

```python
# Prepare multi-output targets
y_direction = (signal_data['label'] == 1).astype(int)
y_profit = signal_data['potential_profit_pct']

# Train separate models or use multi-output models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

direction_model = RandomForestClassifier(n_estimators=100)
profit_model = RandomForestRegressor(n_estimators=100)

direction_model.fit(X, y_direction)
profit_model.fit(X, y_profit)

# Make predictions
direction_pred = direction_model.predict(X_new)
profit_pred = profit_model.predict(X_new)

# Combine predictions for decision making
high_confidence_trades = (
    (direction_pred == 1) & (profit_pred > 0.02)  # BUY with >2% expected profit
) | (
    (direction_pred == 0) & (profit_pred < -0.01)  # SELL with >1% expected profit
)
```

**Benefits**:
- Comprehensive trade analysis
- Better risk management
- Enhanced decision-making framework

### 4. Profit-Based Feature Engineering

**Use Case**: Create new features based on profit patterns and relationships.

```python
# Create profit-based features
enhanced_data = signal_data.copy()

# Profit magnitude features
enhanced_data['profit_abs'] = np.abs(signal_data['potential_profit_pct'])
enhanced_data['profit_squared'] = signal_data['potential_profit_pct'] ** 2
enhanced_data['profit_sign'] = np.sign(signal_data['potential_profit_pct'])

# Interaction features
for feature in ['rsi', 'macd', 'bollinger_upper']:
    enhanced_data[f'{feature}_profit_interaction'] = (
        signal_data[feature] * signal_data['potential_profit_pct']
    )

# Risk-reward features
enhanced_data['risk_reward_ratio'] = (
    np.abs(signal_data['potential_profit_pct']) / 
    (1 + signal_data['volatility'])  # Volatility-adjusted
)

# Profit categories
enhanced_data['profit_category'] = pd.cut(
    signal_data['potential_profit_pct'],
    bins=[-np.inf, -0.02, -0.01, 0, 0.01, 0.02, np.inf],
    labels=['high_loss', 'medium_loss', 'small_loss', 'small_profit', 'medium_profit', 'high_profit']
)

# Use enhanced features for training
X_enhanced = enhanced_data[original_features + profit_features]
```

**Benefits**:
- Richer feature set for model training
- Better capture of profit patterns
- Improved model performance

### 5. Profit Threshold Optimization

**Use Case**: Find optimal profit thresholds for different market conditions.

```python
# Train profit prediction model
profit_model = RandomForestRegressor(n_estimators=100)
profit_model.fit(X_train, y_profit_train)

# Predict profits on validation set
profit_predictions = profit_model.predict(X_val)

# Test different thresholds
thresholds = np.arange(-0.03, 0.06, 0.005)
results = []

for threshold in thresholds:
    above_threshold = profit_predictions > threshold
    
    if above_threshold.sum() > 0:
        avg_profit = y_profit_val[above_threshold].mean()
        trade_count = above_threshold.sum()
        win_rate = (y_profit_val[above_threshold] > 0).mean()
        
        results.append({
            'threshold': threshold,
            'avg_profit': avg_profit,
            'trade_count': trade_count,
            'win_rate': win_rate,
            'total_profit': avg_profit * trade_count
        })

# Find optimal threshold
results_df = pd.DataFrame(results)
optimal_threshold = results_df.loc[results_df['total_profit'].idxmax(), 'threshold']

print(f"Optimal profit threshold: {optimal_threshold:.3f}")
```

**Benefits**:
- Data-driven threshold optimization
- Maximize profit potential
- Adapt to different market conditions

## Integration with Existing Pipeline

### Step 1: Update Data Loading

```python
# In your existing data loading code
def load_labeled_data_with_profits(data_path: str) -> pd.DataFrame:
    """Load labeled data that includes profit tracking information."""
    data = pd.read_parquet(data_path)
    
    # Verify profit tracking column exists
    if 'potential_profit_pct' not in data.columns:
        raise ValueError("Profit tracking data not found. Run triple barrier with include_profit_tracking=True")
    
    return data
```

### Step 2: Enhance Feature Engineering

```python
# In your feature engineering pipeline
def create_profit_enhanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features enhanced with profit tracking information."""
    enhanced = data.copy()
    
    # Add profit-based features
    enhanced['profit_magnitude'] = np.abs(data['potential_profit_pct'])
    enhanced['profit_direction'] = np.sign(data['potential_profit_pct'])
    
    # Create interaction features with existing technical indicators
    for col in ['rsi', 'macd', 'bollinger_position']:
        if col in data.columns:
            enhanced[f'{col}_profit_interaction'] = data[col] * data['potential_profit_pct']
    
    return enhanced
```

### Step 3: Update Model Training

```python
# In your model training code
def train_profit_enhanced_model(X: pd.DataFrame, y: pd.Series, profits: pd.Series):
    """Train model with profit tracking enhancement."""
    
    # Method 1: Profit-weighted training
    sample_weights = np.abs(profits) + 0.001
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y, sample_weight=sample_weights)
    
    return model

def train_profit_prediction_model(X: pd.DataFrame, profits: pd.Series):
    """Train model to predict profit magnitude."""
    
    profit_model = RandomForestRegressor(n_estimators=100)
    profit_model.fit(X, profits)
    
    return profit_model
```

### Step 4: Enhanced Model Evaluation

```python
# In your model evaluation code
def evaluate_profit_enhanced_model(model, X_test, y_test, profits_test):
    """Evaluate model with profit tracking metrics."""
    
    # Standard metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Profit-based metrics
    correct_predictions = (y_pred == y_test)
    avg_profit_correct = profits_test[correct_predictions].mean()
    avg_profit_incorrect = profits_test[~correct_predictions].mean()
    
    # High-profit trade accuracy
    high_profit_trades = profits_test > 0.02  # >2% profit
    high_profit_accuracy = accuracy_score(
        y_test[high_profit_trades], 
        y_pred[high_profit_trades]
    )
    
    return {
        'accuracy': accuracy,
        'avg_profit_correct': avg_profit_correct,
        'avg_profit_incorrect': avg_profit_incorrect,
        'high_profit_accuracy': high_profit_accuracy
    }
```

## Advanced Integration Techniques

### 1. Ensemble Methods with Profit Tracking

```python
# Combine multiple models with profit weighting
def create_profit_weighted_ensemble(models, X, y, profits):
    """Create ensemble with profit-weighted predictions."""
    
    predictions = []
    weights = []
    
    for model in models:
        pred = model.predict_proba(X)[:, 1]  # Probability of BUY
        predictions.append(pred)
        
        # Weight based on model's performance on high-profit trades
        high_profit_mask = profits > profits.quantile(0.8)
        weight = model.score(X[high_profit_mask], y[high_profit_mask])
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Weighted ensemble prediction
    ensemble_pred = np.average(predictions, weights=weights, axis=0)
    
    return ensemble_pred
```

### 2. Dynamic Threshold Adjustment

```python
# Adjust thresholds based on profit predictions
def dynamic_threshold_adjustment(profit_predictions, base_threshold=0.5):
    """Adjust classification threshold based on profit potential."""
    
    # Higher profit potential = lower threshold (more aggressive)
    profit_factor = np.clip(profit_predictions * 10, -0.5, 0.5)
    adjusted_threshold = base_threshold - profit_factor
    
    return adjusted_threshold
```

### 3. Risk-Adjusted Position Sizing

```python
# Size positions based on profit predictions
def calculate_position_size(profit_prediction, base_size=1.0, max_size=3.0):
    """Calculate position size based on expected profit."""
    
    # Scale position size with profit potential
    profit_factor = np.clip(profit_prediction * 20, 0.5, max_size)
    position_size = base_size * profit_factor
    
    return position_size
```

## Configuration Examples

### Configuration for Profit-Weighted Training

```python
config = {
    "model_training": {
        "use_profit_weighting": True,
        "profit_weight_power": 1.0,  # Linear weighting
        "min_profit_weight": 0.001,  # Minimum weight
    },
    "feature_engineering": {
        "include_profit_features": True,
        "profit_interaction_features": True,
        "risk_reward_features": True,
    },
    "threshold_optimization": {
        "enable_profit_thresholds": True,
        "threshold_search_range": [-0.03, 0.06],
        "optimization_metric": "total_profit",  # or "win_rate"
    }
}
```

### Configuration for Multi-Output Models

```python
config = {
    "multi_output_training": {
        "predict_direction": True,
        "predict_profit": True,
        "direction_model": "RandomForestClassifier",
        "profit_model": "RandomForestRegressor",
        "ensemble_method": "weighted_average",
    },
    "decision_logic": {
        "min_profit_threshold": 0.01,  # 1% minimum profit
        "max_loss_threshold": -0.02,   # 2% maximum loss
        "confidence_threshold": 0.7,   # 70% confidence
    }
}
```

## Best Practices

### 1. Data Quality
- Ensure profit tracking data is accurate and complete
- Handle missing profit values appropriately
- Validate profit ranges are reasonable

### 2. Model Validation
- Use time-series cross-validation for realistic evaluation
- Test on different market conditions
- Monitor profit prediction accuracy over time

### 3. Risk Management
- Set appropriate profit thresholds based on risk tolerance
- Monitor model performance on high-profit trades
- Implement stop-loss mechanisms based on profit predictions

### 4. Performance Monitoring
- Track profit prediction accuracy
- Monitor feature importance changes
- Evaluate model performance on different profit categories

## Conclusion

Integrating profit tracking into your ML training pipeline provides significant opportunities for improved model performance and better trading decisions. The key is to start with simple integrations (like sample weighting) and gradually add more sophisticated techniques based on your specific needs and performance requirements.

Remember to validate all changes thoroughly and monitor performance metrics to ensure the enhancements are providing the expected benefits.