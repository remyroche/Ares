# Low Sample Count Analysis - Critical Issue

## Problem Identified: Only 84 Events from 3 Years

### Current Evidence:
```
📊 Candidate OHLCV_VO: 205 events, 70 features, target range [-2.3394, 2.5679]
⚠️ WARNING: Only 84 events - high risk of overfitting
```

### Root Cause Analysis:

#### 1. **Event Generation Bottleneck**
- **Expected**: Thousands of events from 3 years of 15m data
- **Actual**: Only 84-205 events per geometry
- **Issue**: Overly restrictive event generation criteria

#### 2. **Triple Barrier Labeling Too Restrictive**
The triple barrier labeling is likely filtering out too many potential events:

```python
# Current likely settings (too restrictive):
- Minimum volatility threshold: Too high
- Maximum holding period: Too short  
- Risk budget: Too conservative
- Sample balance requirements: Too strict
```

#### 3. **Data Filtering Issues**
Possible data quality or filtering problems:
- Missing data periods
- Excessive data cleaning
- Market condition filtering
- Volume/liquidity thresholds

## Investigation Points:

### 1. **Event Generation Parameters**
```python
# Need to check these parameters in Layer 2:
- min_volatility: Current threshold?
- max_holding_period: Current limit?
- risk_budget: Current setting?
- sample_balance: Current requirement?
- liquidity_threshold: Current filter?
```

### 2. **Data Coverage Analysis**
```python
# Need to verify:
- Total data points: Should be ~105,120 (3 years × 365 days × 24 hours × 4 bars/hour)
- Actual data points: How many after filtering?
- Missing periods: Any gaps in data?
- Data quality: Any excessive cleaning?
```

### 3. **Labeling Strategy**
```python
# Triple Barrier settings to check:
- pt_mult (profit target multiplier): Current value?
- sl_mult (stop loss multiplier): Current value?
- horizon (maximum holding period): Current value?
- min_return (minimum return threshold): Current value?
```

## Expected vs Actual:

### **Expected Event Generation:**
- 15-minute data over 3 years: ~105,120 data points
- Reasonable event frequency: 1-5% of data points
- Expected events: 1,000 - 5,000 events
- Actual events: Only 84-205 events

### **Event Generation Efficiency:**
- Current efficiency: 0.08% (84/105,120)
- Expected efficiency: 1-5%
- Problem: 12-60x lower than expected

## Potential Solutions:

### 1. **Relax Event Generation Criteria**
```python
# Suggested parameter adjustments:
- min_volatility: Reduce by 50%
- max_holding_period: Increase from 48 to 96 bars
- risk_budget: Increase from 0.7 to 1.0
- liquidity_threshold: Reduce by 25%
```

### 2. **Adjust Triple Barrier Settings**
```python
# Suggested labeling adjustments:
- pt_mult: Increase from 1.5-2.0 to 2.0-3.0
- sl_mult: Increase from 0.75-1.0 to 1.0-1.5
- horizon: Increase from 12-48 to 24-96 bars
- min_return: Reduce threshold
```

### 3. **Data Pipeline Investigation**
```python
# Check data processing steps:
- Raw data count: Verify input data volume
- Resampling impact: Check 15m resampling efficiency
- Cleaning filters: Review excessive data removal
- Market hours: Verify correct time filtering
```

## Immediate Actions Needed:

### 1. **Diagnostic Analysis**
```python
# Add detailed logging to event generation:
- Log total input data points
- Log data points after each filter
- Log event generation attempts
- Log event rejection reasons
```

### 2. **Parameter Audit**
```python
# Review current parameter settings:
- Triple barrier parameters
- Event generation thresholds
- Data cleaning filters
- Sample balance requirements
```

### 3. **Data Quality Check**
```python
# Verify data integrity:
- Check for missing data periods
- Verify resampling correctness
- Check for excessive filtering
- Validate market hours handling
```

## Impact Assessment:

### **Current Impact:**
- **Model Quality**: 84 events insufficient for reliable training
- **Overfitting Risk**: Extremely high with small sample
- **Statistical Power**: Too low for meaningful validation
- **Generalization**: Poor expected performance

### **Expected Improvement:**
- **Target Events**: 1,000-2,000 events per geometry
- **Model Reliability**: Significantly improved
- **Validation**: Proper train/test splits possible
- **Statistical Power**: Adequate for robust modeling

## Next Steps:

1. **Immediate**: Add diagnostic logging to event generation
2. **Short-term**: Relax restrictive parameters
3. **Medium-term**: Optimize triple barrier settings
4. **Long-term**: Implement adaptive event generation

## Critical Priority:

This is the **most important issue** to resolve. All downstream modeling depends on having sufficient events. The current 84 events will lead to unreliable models and poor performance.
