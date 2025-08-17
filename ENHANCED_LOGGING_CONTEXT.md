# Enhanced Logging Context Implementation

## Problem Statement

Current logging lacks crucial context information:
- **Step identification**: Which training step we're in (step1_7, step2, etc.)
- **Model context**: Which model/regime we're optimizing
- **Asset context**: Which asset/timeframe we're working on
- **Progress tracking**: Clear indication of optimization progress

## Solution: Context-Aware Logging

### 1. **Enhanced Heartbeat Logging**

The `heartbeat` function in `src/utils/logger.py` will be enhanced to include context:

```python
@contextmanager
def heartbeat_with_context(
    logger: logging.Logger,
    name: str,
    context: dict[str, str] = None,
    interval_seconds: float = 15.0,
    details_provider: Callable[[], str] | None = None,
):
    """
    Enhanced heartbeat with context information.
    
    Args:
        context: Dict with keys like 'step', 'model', 'regime', 'asset', 'timeframe'
    """
```

### 2. **Context Propagation**

Add context to all major logging calls:

```python
# Example usage in autoencoder optimization
context = {
    'step': 'step3_feature_engineering',
    'model': 'autoencoder',
    'regime': 'combined',
    'asset': 'ETHUSDT',
    'timeframe': '1m'
}

with heartbeat_with_context(
    self.logger, 
    "AE optuna_optimization", 
    context=context,
    interval_seconds=60.0
):
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
```

### 3. **Enhanced Log Messages**

Current:
```
⏳ AE optuna_optimization still running... elapsed=1740.3s
```

Enhanced:
```
⏳ [step3_feature_engineering] AE optuna_optimization still running... 
   Model: autoencoder | Regime: combined | Asset: ETHUSDT/1m | elapsed=1740.3s
   Progress: Trial 45/100 | Best loss: 0.0039 | Current trial: 46
```

### 4. **Context Manager for Steps**

Create a context manager to automatically add step context:

```python
@contextmanager
def step_context(step_name: str, asset: str, timeframe: str):
    """Context manager for step-based logging."""
    context = {
        'step': step_name,
        'asset': asset,
        'timeframe': timeframe
    }
    # Set global context
    # Log step start/end
    # Provide context to all loggers
```

## Implementation Plan

### Phase 1: Enhanced Heartbeat Function
1. **Modify `heartbeat` function** in `src/utils/logger.py`
2. **Add context parameter** to include step/model/regime info
3. **Update log format** to show context in heartbeat messages

### Phase 2: Context Propagation
1. **Add context to autoencoder optimization**
2. **Add context to HMM training**
3. **Add context to feature engineering**
4. **Add context to model training**

### Phase 3: Step Context Manager
1. **Create step context manager**
2. **Integrate with existing step functions**
3. **Add automatic context injection**

## Benefits

### **Immediate Benefits**
- **Clear progress tracking**: Know exactly which step/model is running
- **Better debugging**: Identify which component is causing issues
- **Performance monitoring**: Track time per step/model/regime
- **Resource allocation**: Understand which components are resource-intensive

### **Long-term Benefits**
- **Automated reporting**: Generate step-by-step progress reports
- **Performance optimization**: Identify bottlenecks per step
- **Error correlation**: Link errors to specific steps/models
- **User experience**: Clear progress indication for users

## Example Enhanced Log Output

```
2025-08-15 16:03:54,599 INFO [step3_feature_engineering] ⏳ AE optuna_optimization still running...
   Model: autoencoder | Regime: combined | Asset: ETHUSDT/1m | elapsed=1800.3s
   Progress: Trial 67/100 | Best loss: 0.0039 | Current trial: 68
   Context: correlation_id=0980f7e7e6434788ac5cd148e3e2f93a

2025-08-15 16:03:58,355 INFO [step3_feature_engineering] 📈 Epoch progress:
   Model: autoencoder | Regime: combined | Asset: ETHUSDT/1m
   Epoch: 44 | Loss: 0.0196 | Val Loss: 0.0039 | Elapsed: 32.2s
```

## Configuration

Add logging context configuration:

```yaml
logging:
  context:
    enabled: true
    include_step: true
    include_model: true
    include_regime: true
    include_asset: true
    include_timeframe: true
    include_progress: true
```

## Files to Modify

1. **`src/utils/logger.py`** - Enhance heartbeat function
2. **`src/analyst/autoencoder_feature_generator.py`** - Add context to optimization
3. **`src/training/steps/step1_7_hmm_regime_discovery.py`** - Add context to HMM training
4. **`src/training/steps/vectorized_advanced_feature_engineering.py`** - Add context to feature engineering
5. **All step files** - Add step context managers

## Testing

1. **Run training pipeline** and verify context appears in logs
2. **Check log parsing** to ensure context is properly formatted
3. **Verify performance** - ensure logging doesn't impact performance
4. **Test error scenarios** - ensure context is preserved in error logs
