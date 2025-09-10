# ML Commons Review for Data Qualification Functions

## Overview
This document reviews what functions from the data qualification steps should be moved to `src/utils/ml_common/` for better reusability and organization.

## Current Status

### ✅ Already in ML Commons
1. **Feature Engineering** - `src/utils/ml_common/feature_selection.py`
2. **Feature Selection** - `src/utils/ml_common/feature_selection.py` 
3. **Matrix Operations** - `src/utils/ml_common/matrix_operations.py`
4. **SR Clustering** - `src/utils/sr_clustering/` (comprehensive SR utilities)
5. **HMM Management** - `src/utils/hmm_composite_manager.py`
6. **Triple Barrier Method** - `src/utils/step06_utilities/step06_labeling_components/`

### ❌ Missing from ML Commons
1. **Feature Generation Optimization** - Not found, needs to be created

## Recommended Functions to Move to ML Commons

### 1. Support/Resistance Detection (`step02_5_sr_optimization.py`)

**Functions to extract to `src/utils/ml_common/sr_detection.py`:**
- `analyze_feature_importance()` - Feature importance analysis
- `_parallel_train_model()` - Parallel model training utilities
- `generate_function_report()` - ML results reporting
- Error handling utilities (`classify_error`, `handle_error_with_recovery`)
- Data drift detection (`detect_data_drift`)

**Core SR Detection Logic:**
- SR level identification algorithms
- SR strength calculation methods
- SR validation and filtering
- SR optimization parameters

### 2. HMM Regime Discovery (`step03_hmm_regime_discovery.py`)

**Functions to extract to `src/utils/ml_common/hmm_regime_detection.py`:**
- HMM model training and optimization
- Regime state identification
- Regime transition probability calculations
- Regime validation and quality metrics
- Multi-timeframe HMM ensemble logic

**Integration with existing `hmm_composite_manager.py`:**
- Enhance the existing manager with additional regime detection capabilities
- Add regime quality assessment methods
- Add regime transition analysis

### 3. Regime Data Splitting (`step04_regime_data_splitting.py`)

**Functions to extract to `src/utils/ml_common/regime_data_processing.py`:**
- `AsyncFileProcessor` - Async file processing utilities
- `MemoryPoolManager` - Memory management for large datasets
- `DataTypeOptimizer` - Data type optimization utilities
- Regime-based data splitting algorithms
- Regime continuity validation
- Cross-regime data validation

### 4. Data Labeling (`step05_labeling.py`)

**Functions to extract to `src/utils/ml_common/data_labeling.py`:**
- `SimpleTimer` - Timing utilities
- `ComprehensiveValidationFramework` - Data validation framework
- Triple barrier method implementation
- Label quality assessment
- Label distribution analysis
- Label validation and correction

**Integration with existing triple barrier utilities:**
- Consolidate with existing `step06_labeling_components/`
- Add regime-aware labeling capabilities
- Add fractional barrier methods

### 5. Feature Generation Optimization (NEW)

**Create `src/utils/ml_common/feature_generation_optimization.py`:**
- Data-driven lookback period optimization
- Feature performance analysis
- Feature stability assessment
- Optimal feature window selection
- Feature decay analysis
- Cross-validation for feature parameters

## Proposed ML Commons Structure

```
src/utils/ml_common/
├── sr_detection.py                    # Support/Resistance detection utilities
├── hmm_regime_detection.py           # HMM regime discovery and analysis
├── regime_data_processing.py         # Regime-based data processing
├── data_labeling.py                  # Data labeling utilities (triple barrier, etc.)
├── feature_generation_optimization.py # Feature generation optimization
└── data_qualification_orchestrator.py # Orchestrates all data qualification steps
```

## Benefits of Moving to ML Commons

1. **Reusability** - Functions can be used across different steps and pipelines
2. **Maintainability** - Centralized location for core algorithms
3. **Testing** - Easier to unit test individual components
4. **Consistency** - Standardized implementations across the codebase
5. **Performance** - Optimized implementations can be shared
6. **Documentation** - Better documentation and examples

## Implementation Priority

1. **High Priority:**
   - `feature_generation_optimization.py` (missing, needed for feature optimization)
   - `sr_detection.py` (core SR algorithms)
   - `data_labeling.py` (consolidate existing triple barrier utilities)

2. **Medium Priority:**
   - `hmm_regime_detection.py` (enhance existing HMM manager)
   - `regime_data_processing.py` (data processing utilities)

3. **Low Priority:**
   - `data_qualification_orchestrator.py` (orchestration layer)

## Integration Points

- **Existing SR utilities** in `src/utils/sr_clustering/` should be integrated
- **Existing HMM manager** in `src/utils/hmm_composite_manager.py` should be enhanced
- **Existing triple barrier** utilities in `src/utils/step06_utilities/step06_labeling_components/` should be consolidated
- **Feature engineering** already exists in `src/utils/ml_common/feature_selection.py`

## Next Steps

1. Create `feature_generation_optimization.py` in ML commons
2. Extract core SR detection algorithms to `sr_detection.py`
3. Consolidate triple barrier utilities into `data_labeling.py`
4. Enhance existing HMM manager with additional regime detection capabilities
5. Create regime data processing utilities
6. Update data qualification steps to use ML commons functions