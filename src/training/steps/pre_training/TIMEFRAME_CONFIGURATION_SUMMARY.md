# Timeframe Configuration Summary

## Overview

This document summarizes the timeframe configuration across the PRE_TRAINING pipeline components.

## Configuration Status: ✅ COMPLETE

All components in the `src/training/steps/pre_training/` directory now follow the standardized timeframe handling:

### Requirements
1. ✅ Accept timeframe as a parameter
2. ✅ Use 15m as the default timeframe
3. ✅ Use 60m when running with the Analyst
4. ✅ Look at global flag (`get_primary_timeframe()`) when available

## Component Details

### 1. Sub-Pipeline (`sub_pipeline.py`)

**Status:** ✅ Fully Configured

**Resolution Order:**
1. Explicit `timeframe` parameter
2. `custom_params['timeframe']`
3. `pipeline['timeframe']`
4. `get_primary_timeframe()` (returns '15m' by default)
5. Final fallback: `'15m'`
6. **Override:** Automatically uses `'60m'` for Analyst runs

**Code Reference:** Lines 459-482
```python
@classmethod
def resolve_timeframe(
    cls,
    *,
    explicit: Optional[str] = None,
    custom_params: Optional[Mapping[str, Any]] = None,
    pipeline_overrides: Optional[Mapping[str, Any]] = None,
) -> str:
    custom_map = dict(custom_params) if isinstance(custom_params, Mapping) else {}
    pipeline_map = dict(pipeline_overrides) if isinstance(pipeline_overrides, Mapping) else {}

    candidates = (
        explicit,
        custom_map.get('timeframe'),
        pipeline_map.get('timeframe'),
        get_primary_timeframe(),  # Returns '15m'
        '15m',
    )

    timeframe = next((str(candidate) for candidate in candidates if candidate), '15m')

    if cls._is_analyst_run(custom_map, pipeline_map):
        timeframe = '60m'  # Analyst override

    return timeframe
```

### 2. Multi-Horizon Profit Labeler (`multi_horizon_profit_labeler.py`)

**Status:** ✅ Fully Configured

**Default:** `'15m'`

**Configuration:**
```python
@dataclass
class MultiHorizonConfig:
    timeframe: str = "15m"  # Line 364
    
    def update_timeframe(self, timeframe: Optional[str]) -> None:
        resolved = timeframe.strip() if timeframe else "15m"  # Line 445
```

**Component Configuration:** Lines 2660-2665
- Accepts timeframe via `config.timeframe`
- Accepts timeframe via `custom_params['timeframe']`
- Falls back to `'15m'`

### 3. Analyst Profit Labeler (`analyst_profit_labeler.py`)

**Status:** ✅ Fully Configured

**Default:** `'60m'` (correct for Analyst)

**Configuration:**
```python
@dataclass
class AnalystProfitLabelerConfig:
    timeframe: str = "60m"  # Line 54
    base_period_minutes: int = 60
```

**Parameter Handling:** Lines 219-225
- Accepts timeframe via `custom_params['timeframe']`
- Automatically calculates `base_period_minutes` from timeframe

### 4. Tactician Entry Labeler (`tactician_entry_labeler.py`)

**Status:** ✅ Fully Configured

**Default:** `'15m'`

**Usage:**
- Line 8: Documentation states "15m timeframe optimization for entry timing"
- Line 461: Falls back to `'15m'` in metadata
- Line 488: Falls back to `'15m'` in metadata

**Parameter Handling:**
- Accepts timeframe via `config.timeframe`
- Used in component metadata and logging

### 5. Final Feature Selection Step (`final_feature_selection_step.py`)

**Status:** ✅ Fully Configured

**Default:** `'15m'` (or `'60m'` for Analyst)

**Resolution Logic:** Lines 1190-1204
```python
if timeframe:
    resolved_timeframe = timeframe
    timeframe_source = 'explicit argument'
else:
    extracted = _extract_timeframe_from_config(runtime_config)
    if extracted:
        resolved_timeframe = extracted
        timeframe_source = 'config override'
    else:
        default_timeframe = '60m' if _config_indicates_analyst(runtime_config) else '15m'
        resolved_timeframe = default_timeframe
        timeframe_source = 'analyst default' if default_timeframe == '60m' else 'global default'
```

### 6. Interactive Feature Generation Components

**Status:** ✅ Fully Configured

All interaction feature generation components use `'15m'` as default and accept timeframe parameter:

#### a. `interactive_feature_generation_component.py`
```python
@dataclass
class InteractiveFeatureGenerationConfig:
    timeframe: str = "15m"  # Line 210
```

#### b. `optimized_interaction_orchestrator.py`
```python
@dataclass
class OptimizedInteractionConfig:
    timeframe: str = "15m"  # Line 249
```

#### c. `optimized_lookback_component.py`
```python
timeframe = pipeline_state.get('timeframe', '15m')  # Line 275
```

## Global Timeframe Configuration

The global timeframe is managed by `src/utils/ml_common/config/universal_timeframe_config.py`:

```python
@dataclass
class UniversalTimeframeConfig:
    primary_timeframe: str = "15m"  # Line 20
```

**Global Function:**
```python
def get_primary_timeframe() -> str:
    """Get the primary timeframe for ML operations."""
    return DEFAULT_TIMEFRAME_CONFIG.get_primary_timeframe()  # Returns '15m'
```

## Summary

✅ **All components are properly configured**

| Component | Default Timeframe | Accepts Parameter | Analyst Override |
|-----------|-------------------|-------------------|------------------|
| Sub-Pipeline | 15m | ✅ | ✅ 60m |
| Multi-Horizon Profit Labeler | 15m | ✅ | via sub-pipeline |
| Analyst Profit Labeler | 60m | ✅ | N/A (already 60m) |
| Tactician Entry Labeler | 15m | ✅ | N/A (always 15m) |
| Final Feature Selection | 15m | ✅ | ✅ 60m |
| Interactive Feature Generation | 15m | ✅ | via sub-pipeline |

## Usage Examples

### 1. Default Usage (15m)
```python
from src.training.steps.pre_training.sub_pipeline import SubPipelineConfig

config = SubPipelineConfig(
    symbol="ETHUSDT",
    exchange="binance"
    # timeframe will default to '15m'
)
```

### 2. Custom Timeframe
```python
config = SubPipelineConfig(
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="5m"  # Explicitly set
)
```

### 3. Analyst Mode (60m)
```python
config = SubPipelineConfig(
    symbol="ETHUSDT",
    exchange="binance",
    custom_params={'role': 'analyst'}  # Automatically uses 60m
)
```

### 4. Override via Pipeline State
```python
pipeline_state = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '30m',  # Override timeframe
    'custom_params': {}
}
```

## Testing

The timeframe configuration has been verified across:
- ✅ Sub-pipeline configuration
- ✅ All labeling components
- ✅ Feature engineering components
- ✅ Feature selection components
- ✅ Interaction feature generation

## Notes

1. The Analyst automatically uses 60m timeframe due to its strategic decision-making focus
2. The Tactician uses 15m timeframe for tactical entry timing
3. All components properly cascade timeframe parameters from pipeline state
4. The global default can be changed via `UniversalTimeframeConfig` if needed

## Last Updated

October 8, 2025