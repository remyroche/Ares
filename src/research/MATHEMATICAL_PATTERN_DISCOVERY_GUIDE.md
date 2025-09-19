# Mathematical Pattern Discovery & Definition Framework

## 🎯 **Core Innovation: Mathematical Precision**

### **The Problem with Traditional Pattern Analysis:**
- **Vague definitions**: "Look for momentum patterns"
- **Subjective interpretation**: Different analysts see different patterns
- **Not ML-ready**: Unclear how to convert to supervised learning targets
- **Not reproducible**: Same data, different results

### **Our Solution: Mathematical Precision**
- **Exact formulas**: `IF |momentum(t)| > 0.005 AND same_direction ≥70% for 10 periods THEN pattern=1`
- **Objective measurement**: Binary output (0 or 1) for each time period
- **ML-ready targets**: Direct use in supervised learning
- **Fully reproducible**: Same formula, same results, every time

## 📊 **Mathematical Pattern Definitions**

### **1. Momentum Persistence Pattern**

**Concept**: Price momentum continues in the same direction with gradual decay

**Mathematical Definition**:
```
Let momentum(t) = mean(returns[t-4:t])
Let threshold = 0.005
Let persistence_window = 10

Pattern exists at time t IF:
1. |momentum(t)| > threshold
2. sign(momentum(t+k)) == sign(momentum(t)) for ≥70% of k ∈ [1,10]
3. |momentum(t+k)| > 0.3 * |momentum(t)| for ≥60% of k ∈ [1,10]

OUTPUT: pattern_label[t] = 1 if conditions met, else 0
```

**Why This Formula**:
- **Condition 1**: Ensures significant momentum exists (not noise)
- **Condition 2**: Momentum direction persists (not random reversals)
- **Condition 3**: Magnitude decays gradually (not abrupt stops)

**Example Binary Output**: `[0,0,1,1,1,0,0,1,1,1,1,0,0,...]`

### **2. Mean Reversion Speed Pattern**

**Concept**: Price reverts toward moving average within specific timeframe

**Mathematical Definition**:
```
Let MA(t) = moving_average(prices[t-19:t])
Let deviation(t) = (price(t) - MA(t)) / MA(t)
Let threshold = 0.02
Let reversion_window = 10

Pattern exists at time t IF:
1. |deviation(t)| > threshold
2. ∃k ∈ [1,10]: |price(t+k) - MA(t)| < 0.7 * |price(t) - MA(t)|

OUTPUT: pattern_label[t] = 1 if conditions met, else 0
```

**Why This Formula**:
- **Condition 1**: Price significantly deviated from mean (oversold/overbought)
- **Condition 2**: Price moves at least 30% closer to mean within 10 periods

**Example Binary Output**: `[0,1,0,0,0,1,1,0,0,1,0,0,...]`

### **3. Volatility Expansion Pattern**

**Concept**: Low volatility periods followed by high volatility expansion

**Mathematical Definition**:
```
Let vol(t) = std(returns[t-19:t])
Let vol_percentile(t) = percentile_rank(vol(t), lookback=100)
Let expansion_window = 10

Pattern exists at time t IF:
1. vol_percentile(t) < 0.2  (bottom 20% volatility)
2. ∃k ∈ [1,10]: vol_percentile(t+k) > 0.8  (top 20% volatility)
3. Expansion rate ≥ 30% of future periods

OUTPUT: pattern_label[t] = 1 if conditions met, else 0
```

**Why This Formula**:
- **Condition 1**: Current volatility is low (quiet market)
- **Condition 2**: Future volatility becomes high (expansion occurs)
- **Condition 3**: Expansion is sustained, not just a single spike

**Example Binary Output**: `[0,0,0,1,0,0,1,1,0,0,0,1,...]`

### **4. Confirmed Breakout Pattern**

**Concept**: Price breaks technical level and continues in breakout direction

**Mathematical Definition**:
```
Let upper_band(t) = MA(t) + 2*STD(t)
Let lower_band(t) = MA(t) - 2*STD(t)
Let confirmation_window = 5

Pattern exists at time t IF:
1. price(t) > upper_band(t) OR price(t) < lower_band(t)
2. ≥60% of price(t+k) for k∈[1,5] continue beyond breakout level
3. Continuation magnitude > 1% of price

OUTPUT: pattern_label[t] = 1 if conditions met, else 0
```

**Why This Formula**:
- **Condition 1**: Clear breakout above/below Bollinger Bands
- **Condition 2**: Breakout is confirmed (not false breakout)
- **Condition 3**: Continuation is meaningful (not tiny moves)

**Example Binary Output**: `[0,0,1,0,0,0,1,0,0,1,1,0,...]`

### **5. Trend Continuation Pattern**

**Concept**: Established trend continues in same direction for multiple periods

**Mathematical Definition**:
```
Let MA_short(t) = moving_average(prices[t-9:t])
Let MA_long(t) = moving_average(prices[t-49:t])
Let trend_strength(t) = |MA_short(t) - MA_long(t)| / MA_long(t)
Let trend_direction(t) = sign(MA_short(t) - MA_long(t))

Pattern exists at time t IF:
1. trend_strength(t) > 0.005
2. trend_direction(t+k) == trend_direction(t) for ≥80% of k∈[1,20]
3. trend_strength(t+k) ≥ 0.7 * trend_strength(t) for ≥60% of k∈[1,20]

OUTPUT: pattern_label[t] = 1 if conditions met, else 0
```

**Why This Formula**:
- **Condition 1**: Established trend with minimum strength
- **Condition 2**: Direction consistency over 20 periods
- **Condition 3**: Strength maintained (trend doesn't weaken significantly)

**Example Binary Output**: `[1,1,1,1,0,0,0,1,1,1,1,1,1,0,...]`

## 🔬 **Pattern Discovery Process**

### **Step 1: Mathematical Definition**
```python
def momentum_persistence(prices, momentum_window=5, persistence_window=10, threshold=0.005):
    """
    EXACT MATHEMATICAL IMPLEMENTATION
    """
    returns = prices.pct_change().fillna(0)
    momentum = returns.rolling(momentum_window).mean()
    
    labels = []
    for i in range(len(momentum) - persistence_window):
        current_momentum = momentum.iloc[i]
        
        if abs(current_momentum) > threshold:
            future_momentum = momentum.iloc[i+1:i+persistence_window+1]
            
            # Direction persistence check
            same_direction = (np.sign(future_momentum) == np.sign(current_momentum))
            direction_persistence = same_direction.sum() / len(future_momentum)
            
            # Magnitude decay check
            magnitude_ratios = abs(future_momentum) / abs(current_momentum)
            gradual_decay = (magnitude_ratios > 0.3).sum() / len(magnitude_ratios)
            
            # Pattern exists if both conditions met
            pattern_exists = (direction_persistence >= 0.7) and (gradual_decay >= 0.6)
            labels.append(1 if pattern_exists else 0)
        else:
            labels.append(0)
    
    return pd.Series(labels, index=prices.index[:len(labels)])
```

### **Step 2: Statistical Validation**
```python
def validate_pattern(pattern_labels, prices):
    """
    STATISTICAL VALIDATION OF DISCOVERED PATTERN
    """
    validation = {}
    
    # Frequency check
    frequency = pattern_labels.sum() / len(pattern_labels)
    validation['frequency'] = frequency
    validation['frequent_enough'] = frequency >= 0.05  # 5% minimum
    
    # Predictability check (entropy-based)
    if frequency > 0 and frequency < 1:
        entropy = -frequency * np.log2(frequency) - (1-frequency) * np.log2(1-frequency)
        predictability = 1.0 - entropy  # Higher = more predictable
        validation['predictability'] = predictability
        validation['predictable_enough'] = predictability > 0.1
    
    # Statistical significance
    returns = prices.pct_change().fillna(0)
    pattern_returns = returns[pattern_labels == 1]
    no_pattern_returns = returns[pattern_labels == 0]
    
    if len(pattern_returns) > 5 and len(no_pattern_returns) > 5:
        t_stat, p_value = stats.ttest_ind(pattern_returns, no_pattern_returns)
        validation['p_value'] = p_value
        validation['statistically_significant'] = p_value < 0.05
    
    return validation
```

### **Step 3: ML Target Generation**
```python
def generate_ml_targets(prices):
    """
    GENERATE ML-READY TARGETS FROM MATHEMATICAL PATTERNS
    """
    targets = {}
    
    # Generate each pattern
    targets['momentum_persistence'] = momentum_persistence(prices)
    targets['mean_reversion_speed'] = mean_reversion_speed(prices)
    targets['volatility_expansion'] = volatility_expansion(prices)
    targets['confirmed_breakout'] = confirmed_breakout(prices)
    targets['trend_continuation'] = trend_continuation(prices)
    
    # Combine into DataFrame
    targets_df = pd.DataFrame(targets)
    
    # Add composite targets
    targets_df['any_momentum'] = targets_df[['momentum_persistence']].max(axis=1)
    targets_df['any_reversion'] = targets_df[['mean_reversion_speed']].max(axis=1)
    targets_df['any_volatility'] = targets_df[['volatility_expansion']].max(axis=1)
    
    return targets_df
```

## 🎯 **Key Advantages of Mathematical Precision**

### **1. Reproducibility**
```
Traditional: "Look for momentum patterns"
→ Analyst A finds 50 patterns
→ Analyst B finds 75 patterns
→ Results not comparable

Mathematical: IF |momentum(t)| > 0.005 AND...
→ Always finds exact same patterns
→ Fully reproducible results
→ Comparable across studies
```

### **2. ML Training Ready**
```
Traditional: Subjective pattern identification
→ How do you train ML model on subjective patterns?
→ No clear target variable

Mathematical: Binary pattern labels [0,1,0,1,0,...]
→ Direct use as supervised learning targets
→ Standard ML training: model.fit(X, y)
```

### **3. Parameter Optimization**
```
Traditional: "Adjust pattern sensitivity by feel"
→ No systematic way to optimize
→ Subjective parameter tuning

Mathematical: Systematic parameter optimization
→ threshold ∈ [0.001, 0.01]
→ persistence_window ∈ [5, 20]
→ direction_persistence_rate ∈ [0.6, 0.9]
→ Optimize for specific objectives (frequency, predictability, etc.)
```

### **4. Statistical Validation**
```
Traditional: "This pattern looks significant"
→ No statistical testing
→ No confidence measures

Mathematical: Rigorous statistical validation
→ Frequency thresholds (pattern must occur often enough)
→ Predictability scores (pattern must be non-random)
→ Statistical significance testing (t-tests, p-values)
→ Signal-to-noise ratio analysis
```

## 📈 **Implementation Example**

```python
from pattern_discovery_framework import PatternDiscoveryOrchestrator

# Initialize pattern discovery
orchestrator = PatternDiscoveryOrchestrator()

# Discover all patterns in price data
results = orchestrator.discover_all_patterns(price_series)

# Get mathematical definitions
definitions = orchestrator.get_pattern_definitions()

# Generate ML-ready targets
ml_targets = orchestrator.export_pattern_labels(results)

# Example output:
# ml_targets = DataFrame with columns:
# - momentum_persistence: [0,1,0,1,0,0,1,1,0,...]
# - mean_reversion_speed: [1,0,0,1,1,0,0,1,0,...]  
# - volatility_expansion: [0,0,1,0,0,0,1,0,0,...]
# - confirmed_breakout: [0,1,0,0,0,1,0,0,1,...]
# - trend_continuation: [1,1,1,0,0,0,1,1,1,...]

# Ready for supervised learning:
X = market_features  # Your market dimension features
y = ml_targets['momentum_persistence']  # Pattern target

model = RandomForestClassifier()
model.fit(X, y)  # Standard ML training!
```

## 🚀 **Benefits for Trading Strategy Development**

### **1. Clear Pattern Identification**
- No ambiguity about what constitutes a pattern
- Exact mathematical criteria
- Consistent pattern recognition across different markets/timeframes

### **2. Backtesting Precision**
- Historical pattern occurrences precisely identified
- Exact entry/exit conditions based on pattern mathematics
- Reproducible backtesting results

### **3. Strategy Optimization**
- Systematic parameter optimization
- Clear objective functions (frequency, predictability, profitability)
- Statistical validation of strategy performance

### **4. Risk Management**
- Pattern duration and magnitude statistics
- Probability-based position sizing
- Statistical confidence intervals for pattern predictions

## 🎯 **Summary: The Core Innovation**

**Traditional Approach**:
- "Look for momentum patterns" → Subjective, not reproducible
- Visual pattern recognition → Not ML-ready
- Ad-hoc parameter tuning → No systematic optimization

**Mathematical Approach**:
- `IF |momentum(t)| > 0.005 AND same_direction ≥70% THEN pattern=1` → Objective, reproducible
- Binary labels [0,1,0,1,...] → Direct ML training targets
- Systematic parameter optimization → Clear objectives and methods

**Result**: Transform vague pattern concepts into precise mathematical formulas that generate ML-ready targets with statistical validation.

This mathematical precision is the foundation that enables everything else - dimension relevance analysis, economic significance testing, and robust ML model training. Without precise pattern definitions, all subsequent analysis is built on subjective foundations.