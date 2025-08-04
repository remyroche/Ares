# Hyperparameter Optimization Improvements

## Overview

This document outlines the comprehensive improvements made to the Ares trading system's hyperparameter optimization, addressing specific requirements and questions.

## 1. Multi-Objective Optimization Enhancements

### Updated Objectives
- **Primary Focus**: Sharpe ratio (50%), Win rate (30%), Profit factor (20%)
- **Removed**: Total return, max drawdown, and other secondary metrics
- **Rationale**: Focus on the three most important metrics for trading performance

### Key Changes
```python
# Updated objective weights
objective_weights = {
    "sharpe_ratio": 0.50,    # Risk-adjusted returns
    "win_rate": 0.30,         # Percentage of winning trades
    "profit_factor": 0.20     # Ratio of gross profit to gross loss
}
```

### Hyperparameter Optimization
The system now optimizes comprehensive hyperparameters including:

#### Model Hyperparameters
- `learning_rate`: Model learning rate (1e-4 to 1e-1)
- `max_depth`: Maximum tree depth (3 to 12)
- `n_estimators`: Number of estimators (50 to 500)
- `subsample`: Subsample ratio (0.6 to 1.0)
- `colsample_bytree`: Column subsample ratio (0.6 to 1.0)

#### Regularization Parameters
- `reg_alpha`: L1 regularization (1e-8 to 10.0)
- `reg_lambda`: L2 regularization (1e-8 to 10.0)

**Why include reg_alpha and reg_lambda?**

1. **L1 Regularization (reg_alpha)**:
   - Promotes sparsity by setting some feature weights to exactly zero
   - Helps with feature selection in high-dimensional data
   - Useful for identifying the most important features
   - Prevents overfitting by encouraging simpler models

2. **L2 Regularization (reg_lambda)**:
   - Prevents overfitting by penalizing large weights
   - Improves generalization to unseen data
   - Helps with multicollinearity issues
   - Stabilizes model training

Both parameters are crucial for:
- **Feature Selection**: L1 helps identify important features
- **Overfitting Prevention**: L2 prevents complex models from memorizing training data
- **Model Stability**: Both improve model robustness
- **Performance**: Proper regularization often improves out-of-sample performance

#### Trading Parameters
- `tp_multiplier`: Take profit multiplier (1.2 to 5.0)
- `sl_multiplier`: Stop loss multiplier (0.8 to 2.5)
- `position_size`: Position size (0.02 to 0.3)
- `confidence_threshold`: Confidence threshold (0.6 to 0.95)

## 2. Bayesian Optimization with Advanced Models

### New Model Types
Added support for:
- **TabNet**: Attention-based neural network for tabular data
- **Transformers**: Self-attention models for sequence data

### TabNet Parameters
```python
tabnet_attention_dim: 8-64      # Attention dimension
tabnet_num_steps: 1-10          # Number of decision steps
tabnet_gamma: 1.0-3.0           # Feature selection parameter
tabnet_momentum: 0.1-0.9        # Momentum for batch normalization
```

### Transformer Parameters
```python
transformer_n_heads: 2-8         # Number of attention heads
transformer_n_layers: 1-6        # Number of transformer layers
transformer_d_model: 32-256      # Model dimension
transformer_dropout: 0.1-0.5     # Dropout rate
```

## 3. Market Regime Detection

### Updated Regime Types
Replaced "volatile" with more specific regimes:

1. **Bull Market**: Strong upward trend
   - Higher TP multipliers (2.5-5.0)
   - Moderate SL multipliers (1.2-2.5)
   - Larger position sizes (0.10-0.25)

2. **Bear Market**: Strong downward trend
   - Moderate TP multipliers (2.0-4.5)
   - Higher SL multipliers (1.0-2.2)
   - Moderate position sizes (0.08-0.20)

3. **Sideways Market**: Range-bound movement
   - Lower TP multipliers (1.5-3.0)
   - Lower SL multipliers (0.8-1.5)
   - Smaller position sizes (0.05-0.15)

4. **Support/Resistance (SR)**: Key level trading
   - Balanced TP/SL ratios (1.8-3.5 TP, 0.9-1.8 SL)
   - Moderate position sizes (0.06-0.18)

5. **Candle Pattern**: Pattern-based trading
   - Conservative TP/SL (1.2-2.5 TP, 0.6-1.2 SL)
   - Small position sizes (0.03-0.12)

## 4. Confidence Threshold Integration

### How confidence_threshold is used throughout the codebase:

1. **Signal Generation** (`src/analyst/`):
   ```python
   # Only generate signals above confidence threshold
   if signal_confidence > self.confidence_threshold:
       return generate_signal()
   ```

2. **Position Sizing** (`src/tactician/`):
   ```python
   # Scale position size based on confidence
   position_size = base_size * (confidence / self.confidence_threshold)
   ```

3. **Risk Management** (`src/supervisor/`):
   ```python
   # Adjust risk parameters based on confidence
   if confidence < self.confidence_threshold:
       reduce_position_size()
   ```

4. **Ensemble Weighting** (`src/analyst/predictive_ensembles/`):
   ```python
   # Weight ensemble predictions by confidence
   weighted_prediction = sum(pred * conf for pred, conf in zip(predictions, confidences))
   ```

5. **Trade Execution** (`src/exchange/`):
   ```python
   # Only execute trades above confidence threshold
   if trade_confidence >= self.confidence_threshold:
       execute_trade()
   ```

### Configuration Integration
The optimized `confidence_threshold` is automatically propagated to:
- Signal generation modules
- Position sizing algorithms
- Risk management systems
- Ensemble weighting schemes
- Trade execution logic

## 5. Configuration Updates

### Enhanced Search Spaces
```yaml
model_hyperparameters:
  model_type: ["xgboost", "lightgbm", "catboost", "random_forest", "gradient_boosting", "tabnet", "transformer"]
  reg_alpha: {"type": "float", "low": 1e-8, "high": 10.0, "log": True}
  reg_lambda: {"type": "float", "low": 1e-8, "high": 10.0, "log": True}

trading_parameters:
  confidence_threshold: {"type": "float", "low": 0.6, "high": 0.95}
  reg_alpha: {"type": "float", "low": 1e-8, "high": 10.0, "log": True}
  reg_lambda: {"type": "float", "low": 1e-8, "high": 10.0, "log": True}
```

### Regime-Specific Constraints
```yaml
regime_specific_constraints:
  bull: {tp_multiplier_range: [2.5, 5.0], sl_multiplier_range: [1.2, 2.5]}
  bear: {tp_multiplier_range: [2.0, 4.5], sl_multiplier_range: [1.0, 2.2]}
  sideways: {tp_multiplier_range: [1.5, 3.0], sl_multiplier_range: [0.8, 1.5]}
  sr: {tp_multiplier_range: [1.8, 3.5], sl_multiplier_range: [0.9, 1.8]}
  candle: {tp_multiplier_range: [1.2, 2.5], sl_multiplier_range: [0.6, 1.2]}
```

## 6. Integration with Existing Pipeline

### Updated Step 5 HPO
The existing `step5_multi_stage_hpo.py` now includes:
- Enhanced hyperparameter ranges
- Multi-objective optimization
- Regime-specific parameter optimization
- Advanced model support (TabNet, Transformers)
- Confidence threshold optimization

### Benefits
1. **Better Risk Management**: Multi-objective optimization considers risk metrics
2. **Adaptive Performance**: Parameters adjust to market conditions
3. **Advanced Models**: Support for state-of-the-art neural networks
4. **Comprehensive Evaluation**: Multiple metrics ensure robust selection
5. **Automated Workflow**: Scheduled optimization keeps parameters current

## 7. Usage Examples

### Running Multi-Objective Optimization
```python
from src.training.multi_objective_optimizer import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(config)
results = optimizer.run_optimization(n_trials=500)
```

### Running Bayesian Optimization
```python
from src.training.bayesian_optimizer import AdvancedBayesianOptimizer

optimizer = AdvancedBayesianOptimizer(config)
results = optimizer.run_optimization()
```

### Running Adaptive Optimization
```python
from src.training.adaptive_optimizer import AdaptiveOptimizer

optimizer = AdaptiveOptimizer(config)
regime = optimizer.detect_market_regime(market_data)
results = optimizer.optimize_for_regime(regime, market_data)
```

## 8. Performance Monitoring

The system tracks:
- Optimization convergence metrics
- Parameter importance analysis
- Regime-specific performance
- Confidence threshold effectiveness
- Model type performance comparison

This comprehensive approach ensures that hyperparameter optimization is both effective and adaptive to changing market conditions. 