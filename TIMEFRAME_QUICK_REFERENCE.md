# Timeframe Configuration - Quick Reference

## TL;DR

✅ **All timeframe requirements are already implemented.**

- Default: **15m**
- Analyst: **60m** (automatic)
- All components accept timeframe parameters
- Global configuration available

## Quick Reference Table

| Component | Default | Analyst | Parameter |
|-----------|---------|---------|-----------|
| Sub-Pipeline | 15m | 60m (auto) | `timeframe=` |
| Multi-Horizon Profit Labeler | 15m | via pipeline | `config.timeframe` |
| Analyst Profit Labeler | 60m | 60m | `custom_params['timeframe']` |
| Tactician Entry Labeler | 15m | N/A | `config.timeframe` |
| Final Feature Selection | 15m | 60m (auto) | `timeframe=` |
| Interactive Features | 15m | via pipeline | `config.timeframe` |

## Usage Examples

### 1. Default (15m)
```python
from src.training.steps.pre_training.sub_pipeline import SubPipelineConfig

config = SubPipelineConfig(
    symbol="ETHUSDT",
    exchange="binance"
)
# Uses 15m automatically
```

### 2. Analyst Mode (60m)
```python
config = SubPipelineConfig(
    symbol="ETHUSDT",
    exchange="binance",
    custom_params={'role': 'analyst'}  # Auto-switches to 60m
)
```

### 3. Custom Timeframe
```python
config = SubPipelineConfig(
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="5m"  # Override to 5m
)
```

### 4. Via Pipeline State
```python
pipeline_state = {
    'timeframe': '30m',  # Override in pipeline state
    'symbol': 'ETHUSDT',
    'exchange': 'binance'
}
```

## Resolution Order

The timeframe is resolved in this priority order:

1. **Explicit parameter** - `timeframe="30m"`
2. **Custom params** - `custom_params={'timeframe': '30m'}`
3. **Pipeline overrides** - `pipeline={'timeframe': '30m'}`
4. **Global config** - `get_primary_timeframe()` → '15m'
5. **Final fallback** - `'15m'`
6. **Analyst override** - If Analyst role detected → `'60m'`

## Key Files

| File | Purpose |
|------|---------|
| `sub_pipeline.py` | Main resolution logic |
| `universal_timeframe_config.py` | Global default (15m) |
| `analyst_profit_labeler.py` | Analyst default (60m) |
| `TIMEFRAME_CONFIGURATION_SUMMARY.md` | Full documentation |

## Detecting Analyst Mode

The system automatically detects Analyst mode by checking for:

```python
custom_params['role'] == 'analyst'
custom_params['pipeline_role'] == 'analyst'
custom_params['execution_role'] == 'analyst'
custom_params['run_role'] == 'analyst'
custom_params['analyst_mode'] == True
custom_params['is_analyst_run'] == True
```

## Global Configuration

To change the global default:

```python
from src.utils.ml_common.config.universal_timeframe_config import (
    get_timeframe_config,
    set_timeframe_config,
    UniversalTimeframeConfig
)

# Create new config with different default
config = UniversalTimeframeConfig(primary_timeframe="30m")
set_timeframe_config(config)
```

## Verification

Run this to verify timeframe configuration:

```python
from src.training.steps.pre_training.sub_pipeline import SubPipelineConfig

# Test default
config1 = SubPipelineConfig()
print(f"Default: {config1.timeframe}")  # Should be '15m'

# Test Analyst
config2 = SubPipelineConfig(custom_params={'role': 'analyst'})
print(f"Analyst: {config2.timeframe}")  # Should be '60m'

# Test custom
config3 = SubPipelineConfig(timeframe='5m')
print(f"Custom: {config3.timeframe}")  # Should be '5m'
```

## Common Patterns

### Pattern 1: Component Configuration
```python
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig

config = MultiHorizonConfig(
    timeframe="15m",  # Explicit
    # ... other params
)
```

### Pattern 2: Custom Parameters
```python
config = ComponentConfig(
    custom_params={
        'timeframe': '30m',
        'other_param': value
    }
)
```

### Pattern 3: Pipeline State
```python
async def execute(self, data, pipeline_state):
    timeframe = pipeline_state.get('timeframe', '15m')
    # ... use timeframe
```

## Notes

- **Analyst components** default to 60m for strategic focus
- **Tactician components** default to 15m for tactical entry timing
- **All components** accept parameter overrides
- **Resolution is automatic** - no manual intervention needed

## Support

For detailed information, see:
- `/workspace/src/training/steps/pre_training/TIMEFRAME_CONFIGURATION_SUMMARY.md`
- `/workspace/TIMEFRAME_ADJUSTMENT_COMPLETE.md`

---

**Last Updated:** October 8, 2025