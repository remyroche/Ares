# Confidence Calculation Explanation

## Overview

The enhanced confidence scoring system calculates confidence scores based on multi-output predictions (direction, profit, and price) to achieve more accurate trading decisions with threshold-based filtering.

## How Confidence is Calculated

### 1. **Individual Confidence Components**

#### **Direction Confidence**
```python
# Base confidence from probability
base_confidence = np.abs(direction_probability - 0.5) * 2  # Convert to 0-1 scale

# Apply threshold filtering
threshold_mask = base_confidence >= direction_threshold
direction_confidence = base_confidence * threshold_mask

# Add uncertainty penalty for predictions near 0.5
uncertainty_penalty = np.exp(-10 * np.abs(direction_probability - 0.5))
direction_confidence *= uncertainty_penalty
```

**Formula**: `direction_confidence = |probability - 0.5| * 2 * threshold_mask * uncertainty_penalty`

#### **Profit Confidence**
```python
# Base confidence from absolute profit magnitude
profit_abs = np.abs(profit_prediction)
base_confidence = np.tanh(profit_abs * 100)  # Sigmoid-like function

# Apply minimum profit threshold
threshold_mask = profit_abs >= profit_threshold
profit_confidence = base_confidence * threshold_mask
```

**Formula**: `profit_confidence = tanh(|profit_prediction| * 100) * threshold_mask`

#### **Price Confidence**
```python
# Calculate price movement percentage
price_movement = |predicted_price - current_price| / current_price

# Base confidence from price movement magnitude
base_confidence = tanh(price_movement * 50)  # Sigmoid-like function

# Apply minimum price movement threshold
threshold_mask = price_movement >= price_threshold
price_confidence = base_confidence * threshold_mask
```

**Formula**: `price_confidence = tanh(|price_movement| * 50) * threshold_mask`

### 2. **Simple Average Combination**

Since all predictions (direction, profit, price) come from the same model, we use a simple average instead of arbitrary weighting:

```python
simple_confidence = (
    direction_confidence + profit_confidence + price_confidence
) / 3.0
```

**Rationale**: No weighting is applied since all predictions are from the same model and are inherently related.

### 3. **No Risk Adjustments**

**All risk adjustments have been removed** as requested:

- ❌ **Sharpe ratio adjustment** - REMOVED
- ❌ **Drawdown adjustment** - REMOVED  
- ❌ **Volatility adjustment** - REMOVED
- ❌ **Market regime adjustment** - REMOVED
- ❌ **Risk-free rate adjustment** - REMOVED

The confidence calculation is now purely based on the model predictions without any external risk factors.

### 4. **Final Confidence Calculation**

```python
# Apply minimum ensemble confidence threshold
final_confidence = np.where(
    simple_confidence >= min_ensemble_confidence,
    simple_confidence,
    0.0
)

# Clip to valid range
final_confidence = np.clip(final_confidence, 0, 1)
```

## Configuration Parameters

### **Direction Settings**
- `direction_threshold`: 0.6 (Minimum direction confidence)
- `direction_weight`: 0.4 (Weight for direction in overall confidence)

### **Profit Settings**
- `profit_threshold`: 0.001 (Minimum expected profit - 0.1%)
- `profit_weight`: 0.3 (Weight for profit in overall confidence)

### **Price Settings**
- `price_threshold`: 0.005 (Minimum price movement - 0.5%)
- `price_weight`: 0.3 (Weight for price prediction in overall confidence)

### **Risk Settings**
- `risk_free_rate`: 0.02 (Risk-free rate - 2% annual)
- `sharpe_threshold`: 0.5 (Minimum Sharpe ratio)
- `max_drawdown_threshold`: 0.1 (Maximum acceptable drawdown - 10%)

### **Ensemble Settings**
- `min_ensemble_confidence`: 0.7 (Minimum ensemble confidence)
- `ensemble_method`: "weighted_average" (Ensemble combination method)

## Trading Signal Generation

Based on the final confidence score and direction prediction:

```python
signals = np.where(
    (final_confidence >= threshold) & (direction_prediction == 1),
    1,  # Long signal
    np.where(
        (final_confidence >= threshold) & (direction_prediction == 0),
        -1,  # Short signal
        0  # No signal
    )
)
```

## Key Features

### **✅ What's Included**
1. **Multi-dimensional confidence** (direction, profit, price)
2. **Threshold-based filtering** for quality trades
3. **Simple average combination** (no arbitrary weighting)
4. **Ensemble confidence** from multiple models
5. **Uncertainty penalty** for ambiguous predictions

### **❌ What's Removed (as requested)**
1. **All risk adjustments** - No Sharpe, drawdown, volatility, or market regime adjustments
2. **Arbitrary weighting** - No weighting since all predictions come from the same model

## Example Calculation

```python
# Input predictions
direction_probability = 0.8  # 80% confidence in upward direction
profit_prediction = 0.02    # 2% expected profit
current_price = 100.0
predicted_price = 102.0     # 2% price increase

# Calculate individual confidences
direction_confidence = |0.8 - 0.5| * 2 * 1.0 * exp(-10 * |0.8 - 0.5|) = 0.6 * 1.0 * 0.05 = 0.03
profit_confidence = tanh(0.02 * 100) * 1.0 = 0.96 * 1.0 = 0.96
price_confidence = tanh(0.02 * 50) * 1.0 = 0.76 * 1.0 = 0.76

# Simple average combination
simple_confidence = (0.03 + 0.96 + 0.76) / 3.0 = 1.75 / 3.0 = 0.583

# No risk adjustments applied

# Final confidence (if above threshold)
final_confidence = 0.583  # Below 0.7 threshold = 0.0 (no trade signal)
```

## Usage

```python
from src.utils.confidence import calculate_multi_output_confidence_batch, get_confidence_threshold_signals

# Calculate confidence using existing utility
confidence_scores = calculate_multi_output_confidence_batch(
    direction_probabilities=direction_prob,
    direction_predictions=direction_pred,
    profit_predictions=profit_pred,
    current_prices=current_prices,
    predicted_prices=price_pred,
    direction_threshold=0.6,
    profit_threshold=0.001,
    price_threshold=0.005,
    min_ensemble_confidence=0.7
)

# Get trading signals
trading_signals = get_confidence_threshold_signals(
    confidence_scores, threshold=0.7
)
```

This confidence calculation system provides a robust, multi-dimensional approach to trading decision making while respecting the requested removal of volatility and market regime adjustments.