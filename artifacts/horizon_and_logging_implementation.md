# Horizon Increase and Event Refusal Logging Implementation

Increase horizon from 24 to 48 bars and add comprehensive logging to track where events are being refused.

## Implementation Overview

### Changes Required:
1. **Increase horizon parameter**: 24 → 48 bars (100% increase)
2. **Add event refusal logging**: Track where and why events are rejected
3. **Implement diagnostic counters**: Monitor event generation pipeline

## Target Files for Modification

### 1. Labeling Configuration Files
- `meta_labeling_hpo_experiment_step.py` - Main orchestration
- `adaptive_event_driven_labeling.py` - Event generation logic
- `composite_event_generators.py` - Composite event logic
- `triple_barrier_validator.py` - Triple barrier validation

### 2. Event Generation Pipeline
- Any files containing horizon=24 parameter
- Event filtering and validation logic
- Quality gate implementations

## Specific Implementation Plan

### 1. Horizon Parameter Update

#### File: `meta_labeling_hpo_experiment_step.py`
```python
# CURRENT (likely location):
def get_layer2_config():
    return {
        'horizon': 24,  # Current value
        'pt_mult': 1.5,
        'sl_mult': 0.8,
        # ... other parameters
    }

# UPDATED:
def get_layer2_config():
    return {
        'horizon': 48,  # INCREASED from 24 to 48
        'pt_mult': 1.5,
        'sl_mult': 0.8,
        # ... other parameters
    }
```

#### File: `adaptive_event_driven_labeling.py`
```python
# CURRENT:
class AdaptiveEventDrivenLabeling:
    def __init__(self, horizon=24, **kwargs):
        self.horizon = horizon

# UPDATED:
class AdaptiveEventDrivenLabeling:
    def __init__(self, horizon=48, **kwargs):  # INCREASED from 24 to 48
        self.horizon = horizon
```

### 2. Event Refusal Logging Implementation

#### Create Event Generation Logger
```python
# New class to track event generation pipeline
class EventGenerationLogger:
    """Comprehensive logging for event generation and refusal tracking"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.stats = {
            'total_data_points': 0,
            'volatility_filtered': 0,
            'liquidity_filtered': 0,
            'regime_filtered': 0,
            'triple_barrier_generated': 0,
            'quality_gate_passed': 0,
            'quality_gate_failed': 0,
            'final_events': 0,
            'refusal_reasons': {}
        }
    
    def log_data_start(self, data_size: int):
        """Log start of event generation"""
        self.stats['total_data_points'] = data_size
        if self.verbose:
            tprint_info(f"📊 Event Generation Starting: {data_size} data points")
    
    def log_volatility_filter(self, passed: int, failed: int, threshold: float):
        """Log volatility filtering results"""
        self.stats['volatility_filtered'] = failed
        if self.verbose:
            tprint_info(f"🔻 Volatility Filter: {passed} passed, {failed} failed (threshold: {threshold})")
    
    def log_liquidity_filter(self, passed: int, failed: int, threshold: float):
        """Log liquidity filtering results"""
        self.stats['liquidity_filtered'] = failed
        if self.verbose:
            tprint_info(f"💧 Liquidity Filter: {passed} passed, {failed} failed (threshold: {threshold})")
    
    def log_regime_filter(self, passed: int, failed: int, regime: str):
        """Log regime filtering results"""
        self.stats['regime_filtered'] = failed
        if self.verbose:
            tprint_info(f"🏛️ Regime Filter ({regime}): {passed} passed, {failed} failed")
    
    def log_triple_barrier_generation(self, events_generated: int):
        """Log triple barrier event generation"""
        self.stats['triple_barrier_generated'] = events_generated
        if self.verbose:
            tprint_info(f"🎯 Triple Barrier Generated: {events_generated} events")
    
    def log_quality_gate_results(self, passed: int, failed: int, gate_name: str):
        """Log quality gate results"""
        self.stats['quality_gate_passed'] += passed
        self.stats['quality_gate_failed'] += failed
        
        if gate_name not in self.stats['refusal_reasons']:
            self.stats['refusal_reasons'][gate_name] = 0
        self.stats['refusal_reasons'][gate_name] += failed
        
        if self.verbose:
            tprint_info(f"🚪 Quality Gate ({gate_name}): {passed} passed, {failed} failed")
    
    def log_final_events(self, final_count: int):
        """Log final event count"""
        self.stats['final_events'] = final_count
        if self.verbose:
            tprint_info(f"✅ Final Events: {final_count}")
    
    def print_summary(self):
        """Print comprehensive summary"""
        tprint_info("📈 Event Generation Summary:")
        tprint_info(f"   Total Data Points: {self.stats['total_data_points']}")
        tprint_info(f"   Volatility Filtered: {self.stats['volatility_filtered']}")
        tprint_info(f"   Liquidity Filtered: {self.stats['liquidity_filtered']}")
        tprint_info(f"   Regime Filtered: {self.stats['regime_filtered']}")
        tprint_info(f"   Triple Barrier Generated: {self.stats['triple_barrier_generated']}")
        tprint_info(f"   Quality Gate Passed: {self.stats['quality_gate_passed']}")
        tprint_info(f"   Quality Gate Failed: {self.stats['quality_gate_failed']}")
        tprint_info(f"   Final Events: {self.stats['final_events']}")
        
        if self.stats['refusal_reasons']:
            tprint_info("   Refusal Reasons:")
            for reason, count in self.stats['refusal_reasons'].items():
                tprint_info(f"     {reason}: {count}")
        
        # Calculate efficiency
        if self.stats['total_data_points'] > 0:
            efficiency = (self.stats['final_events'] / self.stats['total_data_points']) * 100
            tprint_info(f"   Event Generation Efficiency: {efficiency:.2f}%")
```

### 3. Integration into Event Generation Pipeline

#### File: `adaptive_event_driven_labeling.py`
```python
# Add logging to event generation
class AdaptiveEventDrivenLabeling:
    def __init__(self, horizon=48, **kwargs):  # Updated horizon
        self.horizon = horizon
        self.event_logger = EventGenerationLogger(verbose=True)
    
    def generate_events(self, market_data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Generate events with comprehensive logging"""
        
        # Start logging
        self.event_logger.log_data_start(len(market_data))
        
        # Step 1: Volatility filtering
        volatility_threshold = config.get('min_volatility', 0.01)
        vol_filtered_data = self._apply_volatility_filter(market_data, volatility_threshold)
        vol_passed = len(vol_filtered_data)
        vol_failed = len(market_data) - vol_passed
        self.event_logger.log_volatility_filter(vol_passed, vol_failed, volatility_threshold)
        
        # Step 2: Liquidity filtering
        liquidity_threshold = config.get('min_liquidity', 1000000)
        liq_filtered_data = self._apply_liquidity_filter(vol_filtered_data, liquidity_threshold)
        liq_passed = len(liq_filtered_data)
        liq_failed = vol_passed - liq_passed
        self.event_logger.log_liquidity_filter(liq_passed, liq_failed, liquidity_threshold)
        
        # Step 3: Regime filtering
        regime_filtered_data = self._apply_regime_filter(liq_filtered_data, config)
        reg_passed = len(regime_filtered_data)
        reg_failed = liq_passed - reg_passed
        self.event_logger.log_regime_filter(reg_passed, reg_failed, "current")
        
        # Step 4: Triple barrier generation
        raw_events = self._generate_triple_barrier_events(regime_filtered_data, config)
        self.event_logger.log_triple_barrier_generation(len(raw_events))
        
        # Step 5: Quality gates
        final_events = []
        for event in raw_events:
            if self._pass_quality_gates(event, config):
                final_events.append(event)
            else:
                # Log specific refusal reason
                reason = self._get_refusal_reason(event, config)
                self.event_logger.log_quality_gate_results(0, 1, reason)
        
        self.event_logger.log_final_events(len(final_events))
        self.event_logger.print_summary()
        
        return final_events
    
    def _generate_triple_barrier_events(self, data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Generate triple barrier events with updated horizon"""
        events = []
        
        for i in range(len(data) - self.horizon):  # Use updated horizon
            # Generate event with 48-bar horizon
            event = self._create_single_event(data, i, self.horizon, config)
            if event:
                events.append(event)
        
        return events
    
    def _pass_quality_gates(self, event: Dict, config: Dict) -> bool:
        """Check if event passes quality gates with logging"""
        gates = [
            ('sample_balance', self._check_sample_balance),
            ('volatility_range', self._check_volatility_range),
            ('time_distribution', self._check_time_distribution),
            ('regime_balance', self._check_regime_balance)
        ]
        
        for gate_name, gate_func in gates:
            if not gate_func(event, config):
                return False
        
        return True
    
    def _get_refusal_reason(self, event: Dict, config: Dict) -> str:
        """Get specific refusal reason for event"""
        if not self._check_sample_balance(event, config):
            return 'sample_balance'
        elif not self._check_volatility_range(event, config):
            return 'volatility_range'
        elif not self._check_time_distribution(event, config):
            return 'time_distribution'
        elif not self._check_regime_balance(event, config):
            return 'regime_balance'
        else:
            return 'unknown'
```

### 4. Configuration Updates

#### Update Default Parameters
```python
# File: labeling_components.py or similar
DEFAULT_LABELING_CONFIG = {
    'horizon': 48,  # Updated from 24 to 48
    'pt_mult': 1.5,
    'sl_mult': 0.8,
    'min_volatility': 0.01,
    'min_liquidity': 1000000,
    'enable_event_logging': True,  # New parameter
    'event_verbose_logging': True  # New parameter
}
```

## Expected Impact

### Event Count Improvement:
- **Current**: 84 events per geometry
- **Expected**: 168+ events per geometry (100% increase from horizon doubling)
- **Target**: 365 events per geometry
- **Gap after fix**: ~2.2x improvement still needed

### Logging Benefits:
- **Visibility**: Clear understanding of where events are lost
- **Optimization**: Data-driven parameter tuning
- **Debugging**: Easy identification of bottlenecks
- **Validation**: Verify event generation efficiency

## Implementation Priority

### Phase 1: Immediate (Critical)
1. Update horizon parameter from 24 to 48
2. Add basic event generation logging
3. Implement EventGenerationLogger class

### Phase 2: Secondary (Important)
1. Add detailed refusal reason logging
2. Implement quality gate tracking
3. Add efficiency metrics

### Phase 3: Validation (Post-implementation)
1. Verify event count increase
2. Analyze refusal patterns
3. Optimize additional parameters

## Validation Requirements

### Functional Validation:
- Horizon parameter correctly updated to 48
- Event generation logging functional
- Refusal reasons accurately tracked
- No regression in event quality

### Performance Validation:
- Event count increases by ~100%
- Logging overhead minimal (<5% performance impact)
- Memory usage stable
- Pipeline execution time acceptable

## Success Criteria

- [ ] Horizon parameter updated to 48 bars
- [ ] Event count increases to 150-200 per geometry
- [ ] Comprehensive logging shows refusal reasons
- [ ] Event generation efficiency clearly visible
- [ ] No functional regression in pipeline
