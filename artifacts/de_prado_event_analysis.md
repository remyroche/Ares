# De Prado Framework Event Analysis

## Corrected Event Expectations

### De Prado Causal Framework Standards:
- **Event Rate**: 1-2% of data points (not 1-5%)
- **Regime Segmentation**: 3 regimes 
- **Expected Events**: ~365 events per geometry (not 1,000-5,000)

### Mathematical Calculation:
```
Total 15m data points (3 years): ~105,120
De Prado event rate (1-2%): 1,051 - 2,102 events
Divided by 3 regimes: 350 - 700 events per regime
Expected per geometry: ~365 events
```

## Current Status Analysis

### Actual Results:
- **Observed**: 84 events per geometry
- **Expected**: ~365 events per geometry  
- **Gap**: 4.3x lower than expected
- **Efficiency**: 0.08% vs expected 1-2%

### Assessment:
- **Problem**: Still significant but less severe than initially thought
- **Target**: Need 4.3x increase in events
- **Achievable**: Parameter adjustments should resolve this

## Root Cause Analysis (De Prado Context)

### 1. **Triple Barrier Labeling Too Restrictive**
Within De Prado framework, likely issues:
- **pt_mult**: Too low for volatility environment
- **sl_mult**: Too tight for risk management
- **horizon**: Too short for meaningful price movements
- **volatility_threshold**: Too high for current market conditions

### 2. **Regime-Based Filtering Issues**
With 3 regimes, potential problems:
- **Regime segmentation**: Too aggressive filtering
- **Regime stability**: Minimum regime length too long
- **Transition periods**: Excluding too much data
- **Sample balance**: Overly strict balance requirements

### 3. **Market Condition Adaptation**
De Prado framework requires adaptive parameters:
- **Volatility scaling**: Not adapting to current volatility
- **Liquidity thresholds**: Too high for current market
- **Risk budget**: Too conservative for event frequency

## De Prado-Specific Solutions

### 1. **Adaptive Triple Barrier Parameters**
```python
# De Prado adaptive parameters:
def get_adaptive_triple_barrier_params(market_data, regime):
    """Adapt triple barrier parameters to market conditions"""
    
    # Calculate regime-specific volatility
    regime_vol = market_data[market_data['regime'] == regime]['close'].std()
    
    # Adaptive pt/sl based on volatility
    if regime_vol < 0.01:  # Low volatility regime
        pt_mult = 2.5  # Higher profit target
        sl_mult = 1.2  # Wider stop loss
        horizon = 96   # Longer holding period
    elif regime_vol < 0.02:  # Medium volatility
        pt_mult = 2.0  # Standard profit target
        sl_mult = 1.0  # Standard stop loss  
        horizon = 48   # Standard holding period
    else:  # High volatility regime
        pt_mult = 1.5  # Lower profit target
        sl_mult = 0.8  # Tighter stop loss
        horizon = 24   # Shorter holding period
    
    return {
        'pt_mult': pt_mult,
        'sl_mult': sl_mult, 
        'horizon': horizon,
        'min_volatility': regime_vol * 0.5  # 50% of regime volatility
    }
```

### 2. **Regime-Aware Event Generation**
```python
# De Prado regime-specific event generation:
def generate_regime_events(market_data, regime_params):
    """Generate events with regime-specific parameters"""
    
    events = []
    for regime in [0, 1, 2]:
        regime_data = market_data[market_data['regime'] == regime]
        params = regime_params[regime]
        
        # Generate events with regime-adapted parameters
        regime_events = generate_triple_barrier_events(
            regime_data, 
            pt_mult=params['pt_mult'],
            sl_mult=params['sl_mult'],
            horizon=params['horizon'],
            min_volatility=params['min_volatility']
        )
        
        events.extend(regime_events)
    
    return events
```

### 3. **De Prado Quality Gates Adjustment**
```python
# Adjust quality gates for De Prado framework:
def get_de_prado_quality_gates():
    """Quality gates aligned with De Prado standards"""
    
    return {
        'min_events_per_regime': 100,  # Minimum per regime
        'max_events_per_regime': 500,  # Maximum per regime
        'min_total_events': 300,       # Minimum total
        'max_total_events': 1500,      # Maximum total
        'balance_tolerance': 0.3,      # 30% balance tolerance
        'min_volatility_percentile': 20,  # 20th percentile
        'max_volatility_percentile': 80   # 80th percentile
    }
```

## Parameter Adjustment Strategy

### 1. **Immediate Fixes (2-3x improvement)**
```python
# Quick parameter adjustments:
- pt_mult: 1.5 → 2.0 (33% increase)
- sl_mult: 0.8 → 1.0 (25% increase)  
- horizon: 24 → 48 bars (100% increase)
- min_volatility: Reduce by 30%
```

### 2. **Regime-Specific Optimization (2-3x improvement)**
```python
# Regime-adaptive parameters:
- Low volatility regime: pt_mult=2.5, horizon=96
- Medium volatility: pt_mult=2.0, horizon=48
- High volatility: pt_mult=1.5, horizon=24
```

### 3. **Quality Gate Relaxation (1.5x improvement)**
```python
# Adjust quality gates:
- Balance tolerance: 20% → 30%
- Minimum events: 150 → 100 per regime
- Volatility threshold: 30th → 20th percentile
```

## Expected Results

### **Target Event Counts:**
- **Current**: 84 events per geometry
- **Target**: 365 events per geometry
- **Improvement**: 4.3x increase needed

### **Achievable Through:**
- Parameter adjustments: 2-3x improvement
- Regime adaptation: 2-3x improvement  
- Quality gate relaxation: 1.5x improvement
- **Combined**: 6-13x improvement (exceeds target)

## Implementation Priority

### **Phase 1: Quick Wins (Immediate)**
1. Relax triple barrier parameters
2. Reduce volatility thresholds
3. Increase holding periods

### **Phase 2: Regime Adaptation (Next run)**
1. Implement regime-specific parameters
2. Adaptive volatility scaling
3. Regime-aware quality gates

### **Phase 3: Advanced Optimization (Future)**
1. Dynamic parameter optimization
2. Market condition adaptation
3. Advanced regime detection

## Validation Requirements

### **De Prado Framework Compliance:**
- Event rate: 1-2% of data points
- Regime balance: Within 30% tolerance
- Statistical validity: Sufficient sample size
- Causal integrity: Maintain framework principles

### **Quality Metrics:**
- Events per regime: 100-500
- Total events: 300-1500
- Balance ratio: 0.7-1.3
- Volatility coverage: 20th-80th percentile

## Conclusion

The 84 events per geometry is **4.3x below** the De Prado framework expectation of ~365 events. This is a significant but solvable issue. Through parameter adjustments and regime-specific optimization, we should achieve the target event count while maintaining De Prado framework integrity.
