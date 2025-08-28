# Performance Metrics Guide: Win/Loss Amount Analysis

## 🎯 **Updated Performance Metrics Focus**

The hyperparameter optimization has been updated to focus on **actual win/loss amounts** rather than just ratios. This provides a more nuanced view of trading performance by considering:

1. **How much we win** on winning trades
2. **How much we lose** on losing trades
3. **The relationship** between win amounts and loss amounts

## 📊 **New Performance Metrics Breakdown**

### **Primary Metrics (70% of total weight)**

#### **1. Win Rate (35% weight)**
- **Definition**: Percentage of winning trades
- **Target**: >60% for bonus, >50% minimum
- **Formula**: `Winning Trades / Total Trades`
- **Impact**: High win rate reduces overall risk

#### **2. Win/Loss Amount Ratio (35% weight)**
- **Definition**: Average win amount ÷ Average loss amount
- **Target**: >2.0 for bonus, >1.5 minimum
- **Formula**: `Average Win Amount / Average Loss Amount`
- **Impact**: Ensures wins are significantly larger than losses

### **Secondary Metrics (30% of total weight)**

#### **3. Sharpe Ratio (15% weight)**
- **Definition**: Risk-adjusted return measure
- **Target**: >1.5 for optimal performance
- **Formula**: `(Return - Risk Free Rate) / Standard Deviation`
- **Impact**: Balances return with risk

#### **4. Max Drawdown (15% weight)**
- **Definition**: Maximum peak-to-trough decline
- **Target**: <30% for optimal risk management
- **Formula**: `(Peak Value - Trough Value) / Peak Value`
- **Impact**: Controls downside risk

## 🔍 **Detailed Win/Loss Analysis**

### **Win Amount Factors**
```python
def calculate_win_amount(params):
    base_win = 0.02  # 2% base win
    
    # Factors that increase win amount
    confidence_factor = params.get("analyst_confidence_threshold", 0.7) * 0.01
    position_size_factor = params.get("base_position_size", 0.05) * 0.5
    volatility_factor = params.get("target_volatility", 0.15) * 0.1
    
    return base_win + confidence_factor + position_size_factor + volatility_factor
```

### **Loss Amount Factors**
```python
def calculate_loss_amount(params):
    base_loss = 0.015  # 1.5% base loss
    
    # Factors that affect loss amount
    stop_loss_factor = params.get("stop_loss_atr_multiplier", 2.0) * 0.005
    position_size_factor = params.get("base_position_size", 0.05) * 0.3
    risk_factor = params.get("max_position_size", 0.25) * 0.1
    
    return base_loss + stop_loss_factor + position_size_factor + risk_factor
```

### **Win/Loss Ratio Calculation**
```python
def calculate_win_loss_ratio(avg_win, avg_loss):
    return avg_win / avg_loss

# Example:
# avg_win = 0.025 (2.5%)
# avg_loss = 0.015 (1.5%)
# win_loss_ratio = 0.025 / 0.015 = 1.67
```

## 🎯 **Optimization Strategy**

### **Multi-Objective Optimization**
The system now optimizes for 5 objectives simultaneously:

1. **Maximize Win Rate** (`win_rate`)
2. **Maximize Average Win** (`avg_win`)
3. **Minimize Average Loss** (`avg_loss`) - Note: negative in optimization
4. **Maximize Sharpe Ratio** (`sharpe_ratio`)
5. **Minimize Max Drawdown** (`max_drawdown`) - Note: negative in optimization

```python
# Multi-objective study creation
study = optuna.create_study(
    directions=["maximize", "maximize", "minimize", "maximize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.HyperbandPruner()
)
```

### **Composite Score Calculation**
```python
def calculate_composite_score(metrics):
    # Calculate actual win/loss amounts and ratios
    avg_win_amount = metrics.average_win
    avg_loss_amount = abs(metrics.average_loss)
    win_loss_amount_ratio = avg_win_amount / avg_loss_amount
    
    # Primary focus on win rate and actual win/loss amounts
    weights = {
        "win_rate": 0.35,                    # 35% weight on win rate
        "win_loss_amount_ratio": 0.35,       # 35% weight on actual win/loss amounts
        "sharpe_ratio": 0.15,                # 15% weight on risk-adjusted return
        "max_drawdown": 0.15                 # 15% weight on risk management
    }
    
    # Calculate weighted score
    composite_score = sum(
        weights[metric] * normalized_metrics[metric]
        for metric in weights.keys()
    )
    
    # Bonuses and penalties
    if metrics.win_rate > 0.6 and win_loss_amount_ratio > 2.0:
        composite_score *= 1.15  # 15% bonus for excellent performance
    
    if avg_win_amount > avg_loss_amount * 2.5 and metrics.win_rate > 0.5:
        composite_score *= 1.1  # 10% bonus for large wins relative to losses
    
    if avg_win_amount < avg_loss_amount * 0.5:
        composite_score *= 0.8  # 20% penalty for small wins vs large losses
    
    return composite_score
```

## 📈 **Performance Targets**

### **Excellent Performance**
- **Win Rate**: >70%
- **Win/Loss Ratio**: >2.5
- **Sharpe Ratio**: >2.0
- **Max Drawdown**: <20%

### **Good Performance**
- **Win Rate**: >60%
- **Win/Loss Ratio**: >2.0
- **Sharpe Ratio**: >1.5
- **Max Drawdown**: <30%

### **Acceptable Performance**
- **Win Rate**: >50%
- **Win/Loss Ratio**: >1.5
- **Sharpe Ratio**: >1.0
- **Max Drawdown**: <40%

## 🔧 **Parameter Impact on Win/Loss Amounts**

### **Parameters That Increase Win Amounts**
1. **Higher Confidence Thresholds**: More selective trades → larger wins
2. **Larger Position Sizes**: More capital at risk → larger absolute wins
3. **Higher Volatility Targets**: More volatile markets → larger moves
4. **Better Ensemble Agreement**: More consensus → higher quality signals

### **Parameters That Decrease Loss Amounts**
1. **Tighter Stop Losses**: `stop_loss_atr_multiplier` < 2.0
2. **Smaller Position Sizes**: Less capital at risk → smaller absolute losses
3. **Dynamic Stop Losses**: Adaptive risk management
4. **Regime-Specific Adjustments**: Adapt to market conditions

### **Parameters That Affect Both**
1. **Position Sizing**: Affects both win and loss amounts proportionally
2. **Risk Management**: Balances potential gains with potential losses
3. **Timing Parameters**: Entry/exit timing affects both
4. **Ensemble Weights**: Model combination affects signal quality

## 📊 **Example Scenarios**

### **Scenario 1: High Win Rate, Small Wins**
```
Win Rate: 80%
Average Win: 1.0%
Average Loss: 2.0%
Win/Loss Ratio: 0.5
Result: Penalty applied (small wins vs large losses)
```

### **Scenario 2: Moderate Win Rate, Large Wins**
```
Win Rate: 55%
Average Win: 3.0%
Average Loss: 1.5%
Win/Loss Ratio: 2.0
Result: Bonus applied (large wins relative to losses)
```

### **Scenario 3: Balanced Performance**
```
Win Rate: 65%
Average Win: 2.5%
Average Loss: 1.0%
Win/Loss Ratio: 2.5
Result: Excellent score with bonuses
```

## 🎯 **Optimization Recommendations**

### **1. Focus on Win/Loss Amount Ratio**
- Prioritize strategies that generate larger wins than losses
- Use position sizing to amplify good signals
- Implement tight stop losses to limit losses

### **2. Balance Win Rate and Win Size**
- Don't sacrifice win rate for larger wins
- Aim for >60% win rate with >2.0 win/loss ratio
- Use confidence thresholds to filter trades

### **3. Risk Management Integration**
- Use dynamic stop losses based on market conditions
- Implement regime-specific position sizing
- Monitor drawdown and adjust accordingly

### **4. Ensemble Optimization**
- Optimize model weights for better signal quality
- Use ensemble agreement to filter trades
- Combine multiple models for robust signals

## 📈 **Monitoring and Validation**

### **Key Metrics to Track**
1. **Win/Loss Amount Ratio**: Should be >2.0 consistently
2. **Win Rate Stability**: Should remain >50% across different market conditions
3. **Drawdown Control**: Should stay <30% even in adverse markets
4. **Sharpe Ratio**: Should be >1.5 for good risk-adjusted returns

### **Validation Process**
1. **Out-of-Sample Testing**: Validate on unseen data
2. **Walk-Forward Analysis**: Test across different time periods
3. **Regime Testing**: Validate across different market conditions
4. **Stress Testing**: Test under extreme market conditions

## 🚀 **Implementation Benefits**

### **Advantages of Win/Loss Amount Focus**
1. **More Realistic**: Reflects actual trading performance
2. **Better Risk Management**: Considers actual loss amounts
3. **Improved Optimization**: Balances frequency and magnitude
4. **Practical Application**: Directly applicable to live trading

### **Expected Improvements**
- **Better Risk/Reward**: More balanced win/loss profiles
- **Reduced Drawdown**: Better loss control
- **Improved Consistency**: More stable performance
- **Enhanced Profitability**: Better overall returns

This updated approach ensures that the hyperparameter optimization focuses on creating strategies that not only win frequently but also win significantly more than they lose, leading to more robust and profitable trading systems.