# Step 2 & 3 Import and Module Fixes

## Issues Identified

### 1. Step 3 Function Call Issue
**Error**: `run_step() missing 1 required positional argument: 'self'`

**Root Cause**: The enhanced training manager was calling `run_step()` directly, but the function signature expected different parameter names.

**Fix**: Updated the enhanced training manager to call `run_step_enhanced()` instead of `run_step()` with the correct parameter names.

**Changes Made**:
```python
# Before (incorrect)
step3_success = await _step3.run_step(
    symbol=symbol,
    exchange=exchange,
    data_dir=data_dir,
    timeframe=timeframe,
    lookback_days=self.lookback_days,
    force=self.force_rerun,
)

# After (correct)
step3_success = await _step3.run_step_enhanced(
    symbol=symbol,
    exchange=exchange,
    data_dir=data_dir,
    timeframe=timeframe,
    lookback_days=self.lookback_days,
    force_rerun=self.force_rerun,
)
```

### 2. Missing Module Imports
**Error**: `cannot import name 'step3_processing_labeling_feature_engineering' from 'src.training.steps'`

**Root Cause**: The `__init__.py` file in `src/training/steps/` was missing imports for key step modules.

**Fix**: Added missing imports to `src/training/steps/__init__.py`.

**Changes Made**:
```python
# Added missing imports
try:
    from .step2_feature_engineering import *
except ImportError:
    pass

try:
    from .step3_hmm_regime_discovery import *
except ImportError:
    pass

try:
    from .step4_processing_labeling import *
except ImportError:
    pass
```

## Files Modified

### 1. `src/training/enhanced_training_manager.py`
- **Line ~1175**: Fixed step3 function call to use `run_step_enhanced()` with correct parameters
- **Parameter mapping**: `force` → `force_rerun` to match expected function signature

### 2. `src/training/steps/__init__.py`
- **Added**: Import for `step2_feature_engineering`
- **Added**: Import for `step3_hmm_regime_discovery`
- **Added**: Import for `step4_processing_labeling`

## Function Signatures

### Step 3 HMM Regime Discovery
```python
# Main function (for direct calls)
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    lookback_days: Optional[int] = None,
    force: bool = False,
    **kwargs: Any,
) -> bool:

# Enhanced function (for enhanced training manager)
async def run_step_enhanced(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    lookback_days: Optional[int] = None,
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
```

### Step 4 Processing & Labeling
```python
async def run_step(
    symbol: str,
    exchange_name: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    exchange: str = "BINANCE",
    force_rerun: bool = False,
    pipeline_config: dict[str, Any] | None = None,
) -> bool:
```

## Testing Recommendations

1. **Test Step 3**: Verify that HMM regime discovery runs without import errors
2. **Test Step 4**: Verify that processing & labeling runs without import errors
3. **Test Pipeline Flow**: Ensure steps 2 → 3 → 4 → 5 flow correctly
4. **Test HMM Data Generation**: Verify that `composite_cluster_id` columns are generated

## Expected Behavior After Fixes

1. **Step 2**: Feature engineering should complete successfully
2. **Step 3**: HMM regime discovery should run without "missing self argument" error
3. **Step 4**: Processing & labeling should run without import errors
4. **Step 5**: Regime data splitting should work with HMM composite clusters

## Error Handling

The fixes maintain the existing error handling:
- **Step 3**: Non-fatal failure (proceeds with warning if HMM data missing)
- **Step 4**: Fatal failure (pipeline stops if processing fails)
- **Step 5**: Fatal failure (pipeline stops if HMM composite clusters missing)

## Next Steps

1. **Run Pipeline**: Test the complete pipeline from step 2 onwards
2. **Verify HMM Data**: Ensure `composite_cluster_id` columns are generated
3. **Monitor Logs**: Check for any remaining import or function call issues
4. **Validate Integration**: Ensure step 5 can access HMM composite cluster data
