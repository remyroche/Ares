# Horizon Parameter Implementation Code

## Files to Modify

### 1. orthogonal_label_generation.py - Line 5094

**CURRENT CODE:**
```python
# Use fixed TP:SL for composites for now (Standard Institutional Params)
pt, actual_sl, horizon, risk_budget = 2.0, 1.0, 48, 0.7
```

**UPDATED CODE:**
```python
# Use fixed TP:SL for composites for now (Standard Institutional Params)
pt, actual_sl, horizon, risk_budget = 2.0, 1.0, 48, 0.7  # Horizon updated from 24 to 48
```

### 2. Event Pipeline Logger Implementation

**Create new logging system:**

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

### 3. Integration into Event Generation

**Add logging to orthogonal_label_generation.py:**

```python
def orthogonal_label_generation(
    df: pd.DataFrame,
    causal_graph: Optional[Dict[str, List[str]]] = None,
    config: Optional[Dict[str, Any]] = None,
    enable_pipeline_logging: bool = True
) -> List[Dict]:
    """
    Generate orthogonal labels with pipeline logging.
    """
    
    # Initialize logger
    if enable_pipeline_logging:
        logger = EventPipelineLogger(verbose=True)
        logger.log_stage("Raw Data", len(df))
    
    # ... existing code for event generation ...
    
    # Add logging at each major stage:
    
    # After OHLCV candidates
    if enable_pipeline_logging:
        total_ohlcv_events = sum(len(cand.get('events', [])) for cand in ohlcv_candidates)
        logger.log_stage("OHLCV Candidates", total_ohlcv_events, len(df))
    
    # After specialist candidates
    if enable_pipeline_logging:
        total_spec_events = sum(len(cand.get('events', [])) for cand in spec_candidates)
        logger.log_stage("Specialist Candidates", total_spec_events, len(df))
    
    # After horizon candidates
    if enable_pipeline_logging:
        total_horizon_events = sum(len(cand.get('events', [])) for cand in horizon_candidates)
        logger.log_stage("Horizon Candidates", total_horizon_events, len(df))
    
    # After filtering
    if enable_pipeline_logging:
        total_filtered_events = sum(len(cand.get('events', [])) for cand in filtered_candidates)
        logger.log_stage("Filtered Candidates", total_filtered_events, len(df))
    
    # After validation
    if enable_pipeline_logging:
        total_validated_events = sum(len(cand.get('events', [])) for cand in validated_candidates)
        logger.log_stage("Validated Candidates", total_validated_events, len(df))
    
    # Final summary
    if enable_pipeline_logging:
        logger.print_summary()
    
    return final_candidates_for_selection
```

## Implementation Steps

### Step 1: Update Horizon Parameter
- Change horizon from 24 to 48 in orthogonal_label_generation.py line 5094

### Step 2: Add Event Pipeline Logger
- Add EventPipelineLogger class to orthogonal_label_generation.py

### Step 3: Integrate Logging
- Add logging calls throughout the event generation pipeline

### Step 4: Enable Logging by Default
- Add enable_pipeline_logging parameter to configuration

## Expected Output

### Before Changes:
```
📊 Raw Data: 105,120 events
🎯 Pipeline Summary: 105,120 → 84 events (0.08% efficiency)
```

### After Changes:
```
📊 Raw Data: 105,120 events
📊 OHLCV Candidates: 15,000 events (14.3% of 105,120)
📊 Specialist Candidates: 8,000 events (7.6% of 105,120)
📊 Horizon Candidates: 4,000 events (3.8% of 105,120)
📊 Filtered Candidates: 500 events (0.5% of 105,120)
📊 Validated Candidates: 168 events (0.2% of 105,120)
🎯 Pipeline Summary: 105,120 → 168 events (0.16% efficiency)
```

## Success Metrics

- [ ] Horizon parameter updated to 48
- [ ] Event count increases from 84 to ~168
- [ ] Pipeline logging shows clear stage progression
- [ ] Efficiency doubles from 0.08% to 0.16%
- [ ] No functional regression in event quality
