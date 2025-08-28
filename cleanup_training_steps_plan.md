# Training Steps Cleanup Plan

## Current Issues Identified

### 1. Duplicate/Conflicting Files
- `step6_hmm_based_training.py` (201KB) vs `step6_hmm_based_training_enhanced.py` (36KB)
  - **Decision**: Keep `step6_hmm_based_training_enhanced.py` as it's more recent and focused
  - **Action**: Remove `step6_hmm_based_training.py`

- `step1_data_collection.py` vs `enhanced_step1_data_collection.py`
  - **Decision**: Keep `enhanced_step1_data_collection.py` as it's more comprehensive
  - **Action**: Remove `step1_data_collection.py`

- `step1_5_data_converter.py` vs `enhanced_step1_5_data_converter.py`
  - **Decision**: Keep `enhanced_step1_5_data_converter.py` as it's more recent
  - **Action**: Remove `step1_5_data_converter.py`

- `step12_final_parameters_optimization.py` vs `step12_final_parameters_optimization_new.py`
  - **Decision**: Keep `step12_final_parameters_optimization_new.py` as it's more recent
  - **Action**: Remove `step12_final_parameters_optimization.py`

### 2. Empty/Dead Code Files
- `feature_artifact_loader.py` (0 bytes) - **Remove**
- `optimized_step_executor.py` (0 bytes) - **Remove**
- `step2_5_enhanced_matrix_operations.py` (85 bytes, just re-export) - **Remove**

### 3. Naming Inconsistencies to Fix
- `step4_triple_barrier_method.py` → `step5_triple_barrier_method.py`
- `step5_labeling.py` → `step6_labeling.py`
- `step6_feature_engineering.py` → `step7_feature_engineering.py`
- `step11_confidence_calibration.py` → `step10_confidence_calibration.py`

### 4. Files to Keep (Most Comprehensive Versions)
- `step6_hmm_based_training_enhanced.py` → `step8_hmm_based_training.py`
- `enhanced_step1_data_collection.py` → `step1_data_collection.py`
- `enhanced_step1_5_data_converter.py` → `step1_5_data_converter.py`
- `step12_final_parameters_optimization_new.py` → `step12_final_parameters_optimization.py`

### 5. Validator Files to Rename Accordingly
- All validator files should be renamed to match their corresponding step files

## Implementation Steps

1. **Remove dead code files**
2. **Rename files to match enhanced_training_manager expectations**
3. **Remove duplicate files (keep most comprehensive versions)**
4. **Update imports in enhanced_training_manager if needed**
5. **Verify all step files have corresponding validators**

## Expected Final Structure

```
src/training/steps/
├── step1_data_collection.py (renamed from enhanced_step1_data_collection.py)
├── step1_data_collection_validator.py
├── step1_5_data_converter.py (renamed from enhanced_step1_5_data_converter.py)
├── step1_5_data_converter_validator.py
├── step2_data_reading.py
├── step2_data_reading_validator.py
├── step3_hmm_regime_discovery.py
├── step3_hmm_regime_discovery_validator.py
├── step4_regime_data_splitting.py
├── step4_regime_data_splitting_validator.py
├── step5_triple_barrier_method.py (renamed from step4_triple_barrier_method.py)
├── step5_triple_barrier_method_validator.py
├── step6_labeling.py (renamed from step5_labeling.py)
├── step6_labeling_validator.py
├── step7_feature_engineering.py (renamed from step6_feature_engineering.py)
├── step7_feature_engineering_validator.py
├── step8_hmm_based_training.py (renamed from step6_hmm_based_training_enhanced.py)
├── step8_hmm_based_training_validator.py
├── step9_analyst_enhancement.py (renamed from step6_analyst_enhancement.py)
├── step9_analyst_enhancement_validator.py
├── step10_confidence_calibration.py (renamed from step11_confidence_calibration.py)
├── step10_confidence_calibration_validator.py
├── step11_tactician_labeling.py (renamed from step8_tactician_labeling.py)
├── step11_tactician_labeling_validator.py
├── step12_tactician_specialist_training.py (renamed from step9_tactician_specialist_training.py)
├── step12_tactician_specialist_training_validator.py
├── step13_final_parameters_optimization.py (renamed from step12_final_parameters_optimization_new.py)
├── step13_final_parameters_optimization_validator.py
├── step14_walk_forward_validation.py (renamed from step13_walk_forward_validation.py)
├── step14_walk_forward_validation_validator.py
├── step15_monte_carlo_validation.py (renamed from step14_monte_carlo_validation.py)
├── step15_monte_carlo_validation_validator.py
├── step16_ab_testing.py (renamed from step15_ab_testing.py)
├── step16_ab_testing_validator.py
├── step17_saving.py (renamed from step16_saving.py)
├── step17_saving_validator.py
└── [other utility files]
```