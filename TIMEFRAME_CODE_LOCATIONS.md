# Timeframe Configuration - Code Locations Reference

## Primary Timeframe Resolution

### Global Default Configuration
**File:** `src/utils/ml_common/config/universal_timeframe_config.py`
```python
# Line 20
@dataclass
class UniversalTimeframeConfig:
    primary_timeframe: str = "15m"  # ← GLOBAL DEFAULT
```

**Function:**
```python
# Line 339
def get_primary_timeframe() -> str:
    """Get the primary timeframe for ML operations."""
    return DEFAULT_TIMEFRAME_CONFIG.get_primary_timeframe()
```

---

## Sub-Pipeline Orchestration

### Main Timeframe Resolution Logic
**File:** `src/training/steps/pre_training/sub_pipeline.py`

**Configuration Class:**
```python
# Lines 376-388
@dataclass
class SubPipelineConfig:
    """Configuration for sub-pipeline execution."""
    timeframe: Optional[str] = None  # Resolved during initialization
    
    # Resolution priority:
    # 1. Explicit timeframe argument
    # 2. custom_params or pipeline metadata
    # 3. Global primary timeframe
    # 4. Fallback to '15m'
    # Analyst runs → '60m' override
```

**Resolution Method:**
```python
# Lines 459-482
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
        explicit,                      # 1. Explicit parameter
        custom_map.get('timeframe'),   # 2. Custom params
        pipeline_map.get('timeframe'), # 3. Pipeline overrides
        get_primary_timeframe(),       # 4. Global (15m)
        '15m',                         # 5. Final fallback
    )

    timeframe = next((str(candidate) for candidate in candidates if candidate), '15m')

    if cls._is_analyst_run(custom_map, pipeline_map):
        timeframe = '60m'  # 6. Analyst override

    return timeframe
```

**Analyst Detection:**
```python
# Lines 496-520
@classmethod
def _is_analyst_run(cls, *sources: Mapping[str, Any]) -> bool:
    role_keys = ('role', 'pipeline_role', 'execution_role', 'run_role')
    analyst_flags = ('analyst_mode', 'is_analyst_run')

    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in role_keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip().lower() == 'analyst':
                return True  # ← Triggers 60m override
        for key in analyst_flags:
            value = source.get(key)
            if cls._is_truthy_flag(value):
                return True  # ← Triggers 60m override
    return False
```

---

## Labeling Components

### 1. Multi-Horizon Profit Labeler
**File:** `src/training/steps/pre_training/multi_horizon_profit_labeler.py`

**Config:**
```python
# Lines 359-365
@dataclass
class MultiHorizonConfig:
    """Configuration for multi-horizon profit labeling."""
    
    # Timeframe settings
    timeframe: str = "15m"  # ← DEFAULT
    base_period_minutes: Optional[float] = None
```

**Update Method:**
```python
# Lines 443-447
def update_timeframe(self, timeframe: Optional[str]) -> None:
    """Update the configuration timeframe and derive its base period."""
    resolved = timeframe.strip() if timeframe else "15m"  # ← Fallback to 15m
    self.timeframe = resolved
    self.base_period_minutes = self._timeframe_to_minutes(resolved)
```

**Component Initialization:**
```python
# Lines 2658-2665
def __init__(self, config: Optional[ComponentConfig] = None):
    super().__init__(config)
    
    mh_config = MultiHorizonConfig()
    
    component_timeframe = self.config.timeframe if self.config and getattr(self.config, 'timeframe', None) else None
    custom_params = config.custom_params if config and config.custom_params else {}
    timeframe_override = custom_params.get('timeframe') if isinstance(custom_params, Mapping) else None
    resolved_timeframe = timeframe_override or component_timeframe or mh_config.timeframe  # ← Resolution chain
    mh_config.update_timeframe(resolved_timeframe)
```

**Execute Method:**
```python
# Lines 2703-2705
async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
    pipeline_timeframe = pipeline_state.get('timeframe')
    config_timeframe = self.config.timeframe if getattr(self.config, 'timeframe', None) else None
    timeframe = pipeline_timeframe or config_timeframe or self.labeler.config.timeframe or '15m'  # ← Fallback chain
```

### 2. Analyst Profit Labeler
**File:** `src/training/steps/pre_training/analyst_profit_labeler.py`

**Config:**
```python
# Lines 49-55
@dataclass
class AnalystProfitLabelerConfig:
    """Configuration for Analyst profit labeling."""
    
    # Timeframe settings (Analyst operates on 60m)
    timeframe: str = "60m"  # ← ANALYST DEFAULT
    base_period_minutes: int = 60
```

**Component Parameter Handling:**
```python
# Lines 215-225
def __init__(self, config: Optional[ComponentConfig] = None):
    super().__init__(config)
    
    analyst_config = AnalystProfitLabelerConfig()
    
    # Override with custom parameters if provided
    if self.config and self.config.custom_params:
        custom_params = self.config.custom_params
        
        # Update timeframe if provided
        if 'timeframe' in custom_params:
            analyst_config.timeframe = custom_params['timeframe']  # ← Parameter override
            # Update base period based on timeframe
            if analyst_config.timeframe.endswith('m'):
                analyst_config.base_period_minutes = int(analyst_config.timeframe[:-1])
            elif analyst_config.timeframe.endswith('h'):
                analyst_config.base_period_minutes = int(analyst_config.timeframe[:-1]) * 60
```

### 3. Tactician Entry Labeler
**File:** `src/training/steps/pre_training/tactician_entry_labeler.py`

**Documentation:**
```python
# Lines 7-8
"""
Key Features:
- 15m timeframe optimization for entry timing  # ← TACTICIAN DEFAULT
"""
```

**Component Usage:**
```python
# Lines 460-461
'metadata': {
    'symbol': self.config.symbol if self.config else 'UNKNOWN',
    'exchange': self.config.exchange if self.config else 'UNKNOWN',
    'timeframe': self.config.timeframe if self.config else '15m',  # ← Fallback to 15m
```

---

## Feature Engineering Components

### 1. Final Feature Selection Step
**File:** `src/training/steps/pre_training/final_feature_selection_step.py`

**Timeframe Resolution:**
```python
# Lines 1190-1204
def run_final_feature_selection_step(...):
    resolved_timeframe: str
    timeframe_source: str
    if timeframe:
        resolved_timeframe = timeframe
        timeframe_source = 'explicit argument'
    else:
        extracted = _extract_timeframe_from_config(runtime_config)
        if extracted:
            resolved_timeframe = extracted
            timeframe_source = 'config override'
        else:
            # ↓ ANALYST CHECK
            default_timeframe = '60m' if _config_indicates_analyst(runtime_config) else '15m'
            resolved_timeframe = default_timeframe
            timeframe_source = 'analyst default' if default_timeframe == '60m' else 'global default'
```

### 2. Interactive Feature Generation Component
**File:** `src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/interactive_feature_generation_component.py`

**Config:**
```python
# Lines 204-210
@dataclass
class InteractiveFeatureGenerationConfig:
    """Configuration for interactive feature generation component."""
    # Basic configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"  # ← DEFAULT
    data_dir: str = field(default_factory=_default_data_directory)
```

### 3. Optimized Interaction Orchestrator
**File:** `src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/optimized_interaction_orchestrator.py`

**Config:**
```python
# Lines 243-249
@dataclass
class OptimizedInteractionConfig:
    """Configuration for optimized interaction feature generation."""
    # Pipeline configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"  # ← DEFAULT
    data_dir: str = field(default_factory=_default_data_directory)
```

### 4. Optimized Lookback Component
**File:** `src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/optimized_lookback_component.py`

**Pipeline State Extraction:**
```python
# Lines 270-275
def _extract_data_from_pipeline_state(self, pipeline_state: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict], Dict]:
    """Extract market data and targets from pipeline state."""
    try:
        # Get symbol and timeframe
        symbol = pipeline_state.get('symbol', 'ETHUSDT')
        timeframe = pipeline_state.get('timeframe', '15m')  # ← Fallback to 15m
```

---

## Step Registry (Documentation)

**File:** `src/training/steps/pre_training/sub_pipeline.py`

```python
# Lines 63-87
STEP_REGISTRY: Dict[str, StepSpec] = {
    'analyst_profit_labeler': StepSpec(
        # ...
        description='Apply Analyst-specific multi-horizon profit labeling (60m timeframe).',  # ← 60m documented
    ),
    'tactician_entry_labeler': StepSpec(
        # ...
        description='Apply Tactician-specific entry timing labels (15m timeframe).',  # ← 15m documented
    ),
    'analyst_feature_lookback_optimization': StepSpec(
        # ...
        description='Optimize feature lookback periods for Analyst (60m timeframe, strategic).',  # ← 60m documented
    ),
    'tactician_feature_lookback_optimization': StepSpec(
        # ...
        description='Optimize feature lookback periods for Tactician (15m timeframe, tactical).',  # ← 15m documented
    ),
}
```

---

## Summary of Key Locations

| Component | File | Line(s) | Default |
|-----------|------|---------|---------|
| Global Config | `universal_timeframe_config.py` | 20 | 15m |
| Sub-Pipeline Resolution | `sub_pipeline.py` | 459-482 | 15m (60m for Analyst) |
| Multi-Horizon Config | `multi_horizon_profit_labeler.py` | 364 | 15m |
| Multi-Horizon Component | `multi_horizon_profit_labeler.py` | 2658-2665 | 15m |
| Multi-Horizon Execute | `multi_horizon_profit_labeler.py` | 2705 | 15m |
| Analyst Config | `analyst_profit_labeler.py` | 54 | 60m |
| Analyst Component | `analyst_profit_labeler.py` | 219-225 | 60m |
| Tactician Doc | `tactician_entry_labeler.py` | 8 | 15m |
| Tactician Component | `tactician_entry_labeler.py` | 461 | 15m |
| Feature Selection | `final_feature_selection_step.py` | 1202 | 15m (60m for Analyst) |
| Interactive Features | `interactive_feature_generation_component.py` | 210 | 15m |
| Optimized Orchestrator | `optimized_interaction_orchestrator.py` | 249 | 15m |
| Optimized Lookback | `optimized_lookback_component.py` | 275 | 15m |

---

**Last Updated:** October 8, 2025