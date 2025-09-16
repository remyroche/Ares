# Triple Barrier Parameters - Detailed Explanation

## 🔍 **Updated Stage 1: Coarse Grid Search Parameters**

### **Search Space: 10 × 10 × 10 × 10 = 10,000 combinations**

---

## 📊 **Parameter Details**

### **1. `pt_mult` (Profit Target Multiplier)**
- **Range**: `0.005` to `0.02` (0.5% to 2.0%)
- **Grid Points**: 10 evenly spaced values
- **Example Values**: 
  ```
  [0.005, 0.0067, 0.0083, 0.01, 0.0117, 0.0133, 0.015, 0.0167, 0.0183, 0.02]
  ```
- **Purpose**: Defines the profit target as a percentage of entry price
- **Calculation**: `Profit Target Price = entry_price × (1 + pt_mult)`
- **Example**: If entry price = $100 and pt_mult = 0.01, profit target = $101

### **2. `sl_mult` (Stop Loss Multiplier)**
- **Range**: `0.002` to `0.01` (0.2% to 1.0%)
- **Grid Points**: 10 evenly spaced values
- **Example Values**:
  ```
  [0.002, 0.0031, 0.0042, 0.0053, 0.0064, 0.0075, 0.0086, 0.0097, 0.0108, 0.01]
  ```
- **Purpose**: Defines the stop loss as a percentage of entry price
- **Calculation**: `Stop Loss Price = entry_price × (1 - sl_mult)`
- **Example**: If entry price = $100 and sl_mult = 0.005, stop loss = $99.50

### **3. `time_barrier` (Time Barrier)**
- **Range**: `20` to `90` minutes
- **Grid Points**: 10 evenly spaced values
- **Example Values**:
  ```
  [20, 27, 35, 42, 50, 57, 65, 72, 80, 90]
  ```
- **Purpose**: Maximum time a position can be held before forced exit
- **Usage**: If neither profit target nor stop loss is hit within this time, position is closed
- **Example**: If time_barrier = 60, position must be closed within 60 minutes

### **4. `lookahead` (Max Lookahead)**
- **Range**: `50` to `300` bars
- **Grid Points**: 10 evenly spaced values
- **Example Values**:
  ```
  [50, 77, 105, 132, 160, 187, 215, 242, 270, 300]
  ```
- **Purpose**: Maximum number of bars to look ahead for barrier hits
- **Usage**: Prevents data leakage by limiting how far into the future to check
- **Example**: If lookahead = 100, only check next 100 bars for profit/stop loss

---

## 🔄 **Key Difference: `time_barrier` vs `lookahead`**

### **`time_barrier` (Time-based)**
- **Unit**: Minutes
- **Purpose**: Real-world time limit for position holding
- **Example**: "Close position within 60 minutes regardless of price"

### **`lookahead` (Data-based)**
- **Unit**: Bars (data points)
- **Purpose**: Prevents data leakage in backtesting
- **Example**: "Only check next 100 bars for barrier hits"

### **How They Work Together**
```python
# Example scenario:
time_barrier = 60 minutes    # Real-time limit
lookahead = 100 bars         # Data limit
bar_frequency = 1 minute     # 1 bar = 1 minute

# The system uses the MORE RESTRICTIVE limit:
# - time_barrier = 60 minutes
# - lookahead = 100 bars = 100 minutes
# → Use 60 minutes (time_barrier is more restrictive)

# Another scenario:
time_barrier = 120 minutes   # Real-time limit  
lookahead = 50 bars          # Data limit
bar_frequency = 1 minute     # 1 bar = 1 minute

# - time_barrier = 120 minutes
# - lookahead = 50 bars = 50 minutes
# → Use 50 minutes (lookahead is more restrictive)
```

---

## 📈 **Parameter Combinations Example**

### **Sample Combination**
```python
pt_mult = 0.01        # 1.0% profit target
sl_mult = 0.005       # 0.5% stop loss  
time_barrier = 60     # 60 minutes max holding
lookahead = 100       # Check next 100 bars
```

### **Trading Logic**
1. **Entry**: Buy at $100
2. **Profit Target**: $101 (1% profit)
3. **Stop Loss**: $99.50 (0.5% loss)
4. **Time Limit**: Close within 60 minutes
5. **Data Limit**: Only check next 100 bars

### **Exit Scenarios**
- **Profit Hit**: Price reaches $101 → Exit with profit
- **Stop Loss Hit**: Price reaches $99.50 → Exit with loss
- **Time Hit**: 60 minutes pass → Exit at current price
- **Lookahead Hit**: 100 bars pass → Exit at current price

---

## 🎯 **Optimization Process**

### **Stage 1: Coarse Grid (10,000 combinations)**
- Test all 10,000 parameter combinations
- Evaluate each combination using the objective function
- Select top 8 candidates with highest scores

### **Stage 2: Fine Grid (1,000 combinations)**
- Refine around best coarse candidates
- Use 30% refinement factor
- Select top 3 candidates

### **Stage 3: Bayesian (100 trials)**
- Fine-tune in optimal parameter space
- Use Optuna TPE sampler
- Select best single parameter set

---

## 🔧 **Configuration Example**

```python
from market_analysis.triple_barrier_labeling import (
    EnhancedOptimizedTripleBarrierLabeler,
    CoarseGridConfig
)

# Updated configuration
coarse_config = CoarseGridConfig(
    pt_mult_range=(0.005, 0.02),      # 0.5% to 2.0%
    sl_mult_range=(0.002, 0.01),      # 0.2% to 1.0%
    time_barrier_range=(20, 90),      # 20 to 90 minutes
    lookahead_range=(50, 300),        # 50 to 300 bars
    grid_size=10,                     # 10³ = 1,000 combinations
    top_k_candidates=8                # Top 8 candidates
)

# Initialize labeler
labeler = EnhancedOptimizedTripleBarrierLabeler({
    'coarse_grid_config': coarse_config
})
```

---

## 📊 **Search Space Visualization**

### **Parameter Ranges**
```
pt_mult:    [0.005, 0.02]     → 10 points
sl_mult:    [0.002, 0.01]     → 10 points  
time_barrier: [20, 90]        → 10 points
lookahead:  [50, 300]         → 10 points
```

### **Total Combinations**
```
10 × 10 × 10 × 10 = 10,000 combinations
```

### **Grid Spacing**
```
pt_mult:    0.0015 spacing (0.02 - 0.005) / 9
sl_mult:    0.0009 spacing (0.01 - 0.002) / 9
time_barrier: 7.8 spacing (90 - 20) / 9
lookahead:  27.8 spacing (300 - 50) / 9
```

---

## 🎯 **Why These Ranges?**

### **Profit Target (0.5% to 2.0%)**
- **Lower bound (0.5%)**: Ensures meaningful profit targets
- **Upper bound (2.0%)**: Prevents unrealistic profit expectations
- **Range**: Covers typical trading profit targets

### **Stop Loss (0.2% to 1.0%)**
- **Lower bound (0.2%)**: Allows for small stop losses
- **Upper bound (1.0%)**: Prevents excessive risk
- **Range**: Covers typical risk management levels

### **Time Barrier (20 to 90 minutes)**
- **Lower bound (20 min)**: Allows for short-term trades
- **Upper bound (90 min)**: Prevents very long holding periods
- **Range**: Covers typical intraday trading timeframes

### **Lookahead (50 to 300 bars)**
- **Lower bound (50 bars)**: Ensures sufficient data for analysis
- **Upper bound (300 bars)**: Prevents excessive data leakage
- **Range**: Balances data availability with leakage prevention

This configuration provides a focused search space that covers realistic trading scenarios while maintaining computational efficiency.