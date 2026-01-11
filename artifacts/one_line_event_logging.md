# One-Line Event Pipeline Logging Implementation

Implement concise one-line logging showing event counts at every stage of the process.

## Implementation Overview

Create a simple, clean logging system that shows event counts flowing through each pipeline stage in a single line per stage.

## One-Line Logger Implementation

### Simple Event Counter Class
```python
class EventPipelineLogger:
    """Simple one-line logging for event pipeline stages"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.stages = []
    
    def log_stage(self, stage_name: str, count: int, total: int = None):
        """Log a single stage with event count"""
        if total:
            percentage = (count / total) * 100
            message = f"📊 {stage_name}: {count:,} events ({percentage:.1f}% of {total:,})"
        else:
            message = f"📊 {stage_name}: {count:,} events"
        
        if self.verbose:
            tprint_info(message)
        
        self.stages.append({
            'stage': stage_name,
            'count': count,
            'total': total,
            'percentage': percentage if total else None
        })
    
    def print_summary(self):
        """Print final summary line"""
        if not self.stages:
            return
        
        initial = self.stages[0]['count']
        final = self.stages[-1]['count']
        efficiency = (final / initial) * 100 if initial > 0 else 0
        
        tprint_info(f"🎯 Pipeline Summary: {initial:,} → {final:,} events ({efficiency:.1f}% efficiency)")
```

### Integration into Event Generation Pipeline

#### Modified Event Generation Function
```python
def generate_events_with_logging(self, market_data: pd.DataFrame, config: Dict) -> List[Dict]:
    """Generate events with one-line logging at each stage"""
    
    logger = EventPipelineLogger(verbose=True)
    
    # Stage 1: Raw data
    raw_count = len(market_data)
    logger.log_stage("Raw Data", raw_count)
    
    # Stage 2: Volatility filter
    vol_filtered = self._apply_volatility_filter(market_data, config.get('min_volatility', 0.01))
    vol_count = len(vol_filtered)
    logger.log_stage("Volatility Filter", vol_count, raw_count)
    
    # Stage 3: Liquidity filter
    liq_filtered = self._apply_liquidity_filter(vol_filtered, config.get('min_liquidity', 1000000))
    liq_count = len(liq_filtered)
    logger.log_stage("Liquidity Filter", liq_count, raw_count)
    
    # Stage 4: Regime filter
    reg_filtered = self._apply_regime_filter(liq_filtered, config)
    reg_count = len(reg_filtered)
    logger.log_stage("Regime Filter", reg_count, raw_count)
    
    # Stage 5: Triple barrier generation
    raw_events = self._generate_triple_barrier_events(reg_filtered, config)
    event_count = len(raw_events)
    logger.log_stage("Triple Barrier Events", event_count, raw_count)
    
    # Stage 6: Quality gate - sample balance
    balance_events = [e for e in raw_events if self._check_sample_balance(e, config)]
    balance_count = len(balance_events)
    logger.log_stage("Sample Balance Gate", balance_count, event_count)
    
    # Stage 7: Quality gate - volatility range
    vol_events = [e for e in balance_events if self._check_volatility_range(e, config)]
    vol_gate_count = len(vol_events)
    logger.log_stage("Volatility Range Gate", vol_gate_count, event_count)
    
    # Stage 8: Quality gate - time distribution
    time_events = [e for e in vol_events if self._check_time_distribution(e, config)]
    time_gate_count = len(time_events)
    logger.log_stage("Time Distribution Gate", time_gate_count, event_count)
    
    # Stage 9: Quality gate - regime balance
    final_events = [e for e in time_events if self._check_regime_balance(e, config)]
    final_count = len(final_events)
    logger.log_stage("Regime Balance Gate", final_count, event_count)
    
    # Final summary
    logger.print_summary()
    
    return final_events
```

### Expected Log Output

#### Clean One-Line Format
```
📊 Raw Data: 105,120 events
📊 Volatility Filter: 80,000 events (76.1% of 105,120)
📊 Liquidity Filter: 75,000 events (71.4% of 105,120)
📊 Regime Filter: 60,000 events (57.1% of 105,120)
📊 Triple Barrier Events: 500 events (0.5% of 105,120)
📊 Sample Balance Gate: 300 events (60.0% of 500)
📊 Volatility Range Gate: 250 events (50.0% of 500)
📊 Time Distribution Gate: 200 events (40.0% of 500)
📊 Regime Balance Gate: 168 events (33.6% of 500)
🎯 Pipeline Summary: 105,120 → 168 events (0.2% efficiency)
```

### Integration Points

#### File: `adaptive_event_driven_labeling.py`
```python
# Add to main event generation method
class AdaptiveEventDrivenLabeling:
    def __init__(self, horizon=48, **kwargs):
        self.horizon = horizon
        self.enable_pipeline_logging = kwargs.get('enable_pipeline_logging', True)
    
    def generate_events(self, market_data: pd.DataFrame, config: Dict) -> List[Dict]:
        if self.enable_pipeline_logging:
            return self.generate_events_with_logging(market_data, config)
        else:
            return self.generate_events_original(market_data, config)
```

#### File: `meta_labeling_hpo_experiment_step.py`
```python
# Enable logging in Layer 2 configuration
def get_layer2_config():
    return {
        'horizon': 48,
        'enable_pipeline_logging': True,
        'event_verbose_logging': True,
        # ... other parameters
    }
```

### Configuration Option

#### Add Logging Toggle
```python
# In configuration files
EVENT_GENERATION_CONFIG = {
    'horizon': 48,
    'enable_pipeline_logging': True,  # Master toggle
    'log_percentage': True,           # Show percentages
    'log_summary': True,              # Show final summary
    'compact_format': True            # One-line format
}
```

### Benefits

#### Immediate Visibility
- **Clear Pipeline Flow**: See exactly where events are lost
- **Quick Identification**: Spot bottlenecks instantly
- **Percentage Context**: Understand relative impact of each filter
- **Efficiency Tracking**: Monitor overall pipeline performance

#### Minimal Overhead
- **Simple Implementation**: Lightweight logging class
- **Fast Execution**: Minimal performance impact
- **Clean Output**: One line per stage, no clutter
- **Optional**: Can be disabled for production

## Implementation Priority

### Phase 1: Core Implementation
1. Create EventPipelineLogger class
2. Integrate into main event generation function
3. Update horizon parameter to 48

### Phase 2: Configuration Integration
1. Add logging toggle to configuration
2. Enable by default for debugging
3. Test with current pipeline

### Phase 3: Validation
1. Verify log output format
2. Confirm event count accuracy
3. Check performance impact

## Success Criteria

- [ ] One-line log output for each pipeline stage
- [ ] Clear event count progression visible
- [ ] Percentage calculations accurate
- [ ] Final summary line shows efficiency
- [ ] Horizon parameter updated to 48
- [ ] Minimal performance overhead (<2%)

This provides exactly what you need: clean, simple one-line logging showing event counts at every stage of the process.
