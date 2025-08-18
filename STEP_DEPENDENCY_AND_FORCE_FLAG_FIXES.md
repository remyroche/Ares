# Step Dependency and Force Flag Fixes

## Problem Analysis

The issue was that running `step3 --force` would restart the whole feature engineering process instead of using artifacts from step2. This was caused by several problems in the codebase:

### 1. **Incomplete Force Flag Handling**
- The `--force` flag was handled inconsistently across different components
- Artifacts from previous steps were not properly preserved
- The force flag logic was not properly integrated with step dependency validation

### 2. **Step Dependency Validation Issues**
- Dependency validation was not properly integrated with the force flag
- Previous step artifacts were not verified before starting a step
- Validation failures were treated as warnings instead of stopping the pipeline

### 3. **Artifact Management Problems**
- No clear policy on which artifacts to preserve vs. delete when using `--force`
- Missing verification that previous step artifacts exist before starting a step
- Inconsistent artifact clearing behavior

### 4. **Validation Integration Issues**
- Validators were called at the end of steps but didn't stop the pipeline on failure
- No verification of previous step artifacts before starting a step
- Validation was treated as informational rather than blocking

## Solution Implementation

### 1. **Enhanced Force Flag Handling**

#### Updated `EnhancedTrainingManager._execute_comprehensive_pipeline()`
- Added proper force_rerun handling that clears artifacts from the starting step and subsequent steps
- Preserves artifacts from previous steps
- Clears checkpoints to ensure fresh start

```python
# Handle force_rerun: clear artifacts from starting step and subsequent steps
if self.force_rerun:
    self.logger.info(f"🧹 Force rerun enabled - clearing artifacts from {start_step} and subsequent steps")
    await self._clear_artifacts_from_step_onward(start_step, symbol, exchange, timeframe)
    # Clear the checkpoint to ensure fresh start
    self._clear_checkpoint()
    self.logger.info(f"✅ Cleared artifacts and checkpoints from {start_step} onward")
```

#### Added `_clear_artifacts_from_step_onward()` Method
- Clears artifacts from the specified step and all subsequent steps
- Preserves artifacts from previous steps
- Uses comprehensive artifact patterns for each step

#### Added `_clear_step_artifacts()` Method
- Defines artifact patterns for each step
- Safely deletes files matching the patterns
- Provides detailed logging of cleared artifacts

### 2. **Enhanced Step Dependency Validation**

#### Updated `validate_step_dependencies()` Method
- Properly handles force_rerun flag by skipping dependency validation when enabled
- Provides clear logging of validation results
- Returns appropriate error messages for failed dependencies

```python
# If force_rerun is True, we're starting from this step, so skip dependency validation
if force_rerun:
    self.logger.info(f"✅ Force rerun enabled for {step_name}, skipping dependency validation")
    return True
```

#### Updated `StepDependencyValidator.validate_step_prerequisites()`
- Improved force_rerun handling
- Better error reporting for failed prerequisites
- More robust checkpoint file checking

### 3. **Previous Step Artifact Verification**

#### Added `verify_previous_step_artifacts()` Method
- Verifies that critical artifacts from the previous step exist before starting a step
- Defines critical artifact patterns for each step
- Stops the pipeline if previous step artifacts are missing

```python
async def verify_previous_step_artifacts(
    self,
    step_name: str,
    symbol: str,
    exchange: str,
    timeframe: str
) -> bool:
    """
    Verify that artifacts from the previous step exist before starting a step.
    """
    # Implementation details...
```

#### Critical Artifact Patterns
Defined critical artifacts for each step:
- `step1_data_collection`: Consolidated data files
- `step2_feature_engineering`: Engineered features parquet file
- `step3_hmm_regime_discovery`: Composite clusters parquet file
- And so on for all steps...

### 4. **Enhanced Validation Integration**

#### Updated Step Execution Logic
- Added artifact verification before each step execution
- Added dependency validation before each step execution
- Made validation failures stop the pipeline instead of just warning

```python
# Verify previous step artifacts BEFORE execution
if not await self.verify_previous_step_artifacts("step2_feature_engineering", symbol, exchange, timeframe):
    self.logger.error("❌ Previous step artifacts not found, stopping pipeline")
    return False

# Validate step dependencies BEFORE execution
if not await self.validate_step_dependencies("step2_feature_engineering", pipeline_state, self.force_rerun):
    self.logger.error("❌ Step 2 dependencies not met, stopping pipeline")
    return False
```

#### Updated Validator Integration
- Validators are called at the end of each step
- Validation failures now stop the pipeline instead of just warning
- Proper error handling for validator exceptions

```python
# Run validator for Step 2 (AFTER execution, for verification only)
try:
    step2_validation = await self._run_step_validator(
        "step2_feature_engineering", training_input, pipeline_state
    )
    if step2_validation and step2_validation.get("validation_passed", False):
        self.logger.info("🎉 Step 2: Feature Engineering completed successfully and validation passed")
    else:
        self.logger.error("❌ Step 2 validation failed - stopping pipeline")
        return False
except Exception as e:
    self.logger.error(f"❌ Step 2 validator failed: {e} - stopping pipeline")
    return False
```

### 5. **Updated Ares Launcher**

#### Updated `_force_fresh_start_from_step()` Method
- Now clears progress for the starting step and all subsequent steps
- Preserves progress from previous steps
- Provides detailed logging of cleared steps

#### Updated `_clear_checkpoint_files()` Method
- Clears both centralized and individual step checkpoint files
- Ensures complete cleanup for fresh start

## Key Improvements

### 1. **Proper Artifact Management**
- ✅ When using `--force`, artifacts from the starting step and subsequent steps are deleted
- ✅ Artifacts from previous steps are preserved
- ✅ Clear logging of what artifacts are being cleared

### 2. **Step Dependency Validation**
- ✅ When starting from a certain step, the system uses artifacts from previous steps
- ✅ Previous step artifacts are verified before starting a step
- ✅ Pipeline stops if previous step artifacts are missing

### 3. **Force Flag Behavior**
- ✅ `--force` deletes artifacts from that step and subsequent steps, not from previous steps
- ✅ Proper integration with step dependency validation
- ✅ Clear checkpoint clearing for fresh start

### 4. **Validation Integration**
- ✅ Validator is called at the end of each step
- ✅ Process stops if validation fails
- ✅ Previous step artifacts are verified before starting each step

## Usage Examples

### Starting from Step 3 with Force
```bash
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step3_hmm_regime_discovery --force
```

**Behavior:**
1. Verifies that step2 artifacts exist (engineered features)
2. Clears step3 artifacts and all subsequent step artifacts
3. Preserves step1 and step2 artifacts
4. Starts execution from step3 using step2 artifacts
5. Validates step3 at the end and stops if validation fails

### Starting from Step 2 without Force
```bash
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step2_feature_engineering
```

**Behavior:**
1. Verifies that step1 artifacts exist (consolidated data)
2. Uses existing step1 artifacts
3. Skips step1 execution
4. Starts execution from step2
5. Validates step2 at the end and stops if validation fails

## Testing Recommendations

1. **Test force flag behavior:**
   - Run step2, then step3 with --force
   - Verify step2 artifacts are preserved
   - Verify step3 artifacts are regenerated

2. **Test dependency validation:**
   - Delete step2 artifacts manually
   - Try to run step3
   - Verify pipeline stops with appropriate error

3. **Test validation integration:**
   - Modify a step to produce invalid output
   - Verify pipeline stops after validation fails

4. **Test artifact preservation:**
   - Run multiple steps
   - Use --force on a later step
   - Verify earlier step artifacts are preserved

## Files Modified

1. `src/training/enhanced_training_manager.py`
   - Added artifact clearing methods
   - Enhanced dependency validation
   - Added artifact verification
   - Updated validation integration

2. `src/utils/step_dependency_validator.py`
   - Improved force_rerun handling
   - Enhanced error reporting

3. `ares_launcher.py`
   - Updated force flag handling
   - Enhanced checkpoint clearing

## Conclusion

These fixes ensure that:
1. When starting from a certain step, the system uses artifacts from previous steps
2. The `--force` flag only deletes artifacts from that step and subsequent steps
3. Validators are called at the end of each step and stop the process if validation fails
4. Previous step artifacts are verified before starting each step

The system now properly handles step dependencies and force flag behavior, providing a robust and predictable training pipeline.