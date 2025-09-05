# Code Quality Unified Report

**Generated:** 2025-09-05T12:30:11.438199
**Project Root:** /workspace/src

## Overall Summary
- **Total Files Analyzed:** 767
- **Total Directories:** 78
- **Total Issues Found:** 160936
- **Issues Fixed:** 0

### Issue Breakdown
- **Undefined Names:** 15782
- **Undefined Imports:** 0
- **Unused Imports:** 0
- **Import Conflicts:** 0
- **Import Issues:** 31564
- **Function Issues:** 141832
- **Security Issues:** 146468
- **Syntax Errors:** 2008

## Critical Files (Most Issues)
| File | Total Issues | Fixed |
|------|--------------|-------|
| step03_hmm_regime_discovery.py | 2190 | 0 |
| enhanced_training_manager.py | 2118 | 0 |
| step12_analyst_enhancement.py | 2115 | 0 |
| step05_labeling.py | 2040 | 0 |
| autoencoder_feature_generator.py | 1897 | 0 |
| step07_enhanced_matrix_operations.py | 1557 | 0 |
| step01_5_data_converter.py | 1536 | 0 |
| advanced_feature_engineering.py | 1406 | 0 |
| raw_data_quality_checker.py | 1368 | 0 |
| step03_regime_discovery_features.py | 1368 | 0 |

## Directory Summary
| Directory | Files | Files with Issues | Total Issues | Fixed |
|-----------|-------|-------------------|--------------|-------|
| training | 144 | 142 | 38110 | 0 |
| utils | 180 | 172 | 31150 | 0 |
| analyst | 56 | 54 | 25972 | 0 |
| market_analysis | 74 | 74 | 23736 | 0 |
| model_training | 78 | 78 | 22734 | 0 |
| hmm_clustering | 40 | 38 | 19446 | 0 |
| data_collection | 68 | 64 | 16766 | 0 |
| sr_levels | 38 | 36 | 13804 | 0 |
| supervisor | 38 | 36 | 8766 | 0 |
| step17_final_parameters_optimization | 28 | 28 | 7832 | 0 |
| tactician | 44 | 40 | 6780 | 0 |
| data_preparation | 14 | 14 | 6488 | 0 |
| backtesting | 26 | 26 | 5862 | 0 |
| step1 | 24 | 22 | 4980 | 0 |
| simplified_architecture | 18 | 18 | 4818 | 0 |
| transition | 22 | 22 | 4686 | 0 |
| decorators | 22 | 22 | 4452 | 0 |
| explainability | 18 | 18 | 4286 | 0 |
| step06_labeling_components | 12 | 12 | 3844 | 0 |
| monitoring | 60 | 60 | 3816 | 0 |

## File Details (Top 20)

### step03_hmm_regime_discovery.py
**Path:** `/workspace/src/training/steps/market_analysis/hmm_clustering/step03_hmm_regime_discovery.py`
**Lines of Code:** 2085
**Total Issues:** 2190 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'error_handling' for function 'secure_step_execution'
- Unknown keyword argument 'rollback_on_failure' for function 'secure_step_execution'
- Unknown keyword argument 'data_validation' for function 'secure_step_execution'
- ... and 13 more

**Import Issues:**
- Undefined name: decorator
- Undefined name: decorator
- Undefined name: decorator
- ... and 51 more

**Function Issues:**
- Function 'handles_errors' is missing a docstring
- Function 'decorator' is missing a docstring
- Function 'monitor_feature_engineering' is missing a docstring
- ... and 1028 more

**Security Issues:**
- Potentially unsafe attribute access 'Path(__file__).parent.parent.parent' without null check
- Potentially unsafe attribute access 'Path(__file__).parent.parent' without null check
- Potentially unsafe attribute access 'Path(__file__).parent' without null check
- ... and 1086 more

### enhanced_training_manager.py
**Path:** `/workspace/src/training/enhanced_training_manager.py`
**Lines of Code:** 2800
**Total Issues:** 2118 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'interval' for function 'process.cpu_percent'
- Unknown keyword argument 'step_name' for function 'self.step_dependency_validator.validate_step_prerequisites'
- Unknown keyword argument 'pipeline_state' for function 'self.step_dependency_validator.validate_step_prerequisites'
- ... and 21 more

**Import Issues:**
- Undefined name: _safe_json_write
- Undefined name: _is_relative_to
- Undefined name: _sanitize_identifier
- ... and 82 more

**Function Issues:**
- Function '_timed_step' is missing a docstring
- Function '_execute_pipeline_step_with_validation' has 12 arguments (consider using a config object)
- Function '_should_run' is missing a docstring
- ... and 1061 more

**Security Issues:**
- Potentially unsafe attribute access 'warnings.filterwarnings' without null check
- Potentially unsafe attribute access 'path.resolve().relative_to' without null check
- Potentially unsafe attribute access 'target.parent.mkdir' without null check
- ... and 942 more

### step12_analyst_enhancement.py
**Path:** `/workspace/src/training/steps/model_training/step12_analyst_enhancement.py`
**Lines of Code:** 1841
**Total Issues:** 2115 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'timeout' for function 'result_queue.get'
- Unknown keyword argument 'exc_info' for function 'self.logger.error'
- Unknown keyword argument 'extra' for function 'self.logger.info'
- ... and 11 more

**Import Issues:**
- Undefined name: decorator
- Undefined name: decorator
- Undefined name: decorator
- ... and 211 more

**Function Issues:**
- Function 'handles_errors' is missing a docstring
- Function 'decorator' is missing a docstring
- Function 'traced' is missing a docstring
- ... and 1014 more

**Security Issues:**
- Potentially unsafe attribute access 'optuna.logging.set_verbosity' without null check
- Potentially unsafe attribute access 'optuna.logging' without null check
- Potentially unsafe attribute access 'optuna.logging.WARNING' without null check
- ... and 867 more

### step05_labeling.py
**Path:** `/workspace/src/training/steps/market_analysis/step05_labeling.py`
**Lines of Code:** 2028
**Total Issues:** 2040 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'regime_column' for function 'regime_labeler.generate_labels'
- Unknown keyword argument 'time_barrier_minutes' for function 'regime_labeler.generate_labels'
- Unknown keyword argument 'max_lookahead' for function 'regime_labeler.generate_labels'

**Import Issues:**
- Undefined name: _wrap
- Undefined name: _identity
- Undefined name: _identity
- ... and 163 more

**Function Issues:**
- Function '_identity' is missing a docstring
- Function '_wrap' is missing a docstring
- Function '__init__' is missing a docstring
- ... and 852 more

**Security Issues:**
- Potentially unsafe attribute access 'Path(__file__).parent.parent.parent' without null check
- Potentially unsafe attribute access 'Path(__file__).parent.parent' without null check
- Potentially unsafe attribute access 'Path(__file__).parent' without null check
- ... and 1013 more

### autoencoder_feature_generator.py
**Path:** `/workspace/src/analyst/autoencoder_feature_generator.py`
**Lines of Code:** 1400
**Total Issues:** 1897 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'validation_data' for function 'self.autoencoder.fit'
- Unknown keyword argument 'epochs' for function 'self.autoencoder.fit'
- Unknown keyword argument 'batch_size' for function 'self.autoencoder.fit'
- ... and 9 more

**Import Issues:**
- Undefined name: AutoencoderConfig
- Undefined name: pd
- Undefined name: exclude_pattern
- ... and 168 more

**Function Issues:**
- Function '__init__' is missing a docstring
- Function '__init__' is missing a docstring
- Function '__init__' is missing a docstring
- ... and 847 more

**Security Issues:**
- Potentially unsafe attribute access 'logging.getLogger' without null check
- Potentially unsafe attribute access 'Path(__file__).resolve().parent.parent.parent' without null check
- Potentially unsafe attribute access 'Path(__file__).resolve().parent.parent' without null check
- ... and 861 more

### step07_enhanced_matrix_operations.py
**Path:** `/workspace/src/training/steps/market_analysis/step07_enhanced_matrix_operations.py`
**Lines of Code:** 1422
**Total Issues:** 1557 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'interval' for function 'psutil.cpu_percent'

**Import Issues:**
- Undefined name: decorator
- Undefined name: logging
- Undefined name: arg
- ... and 85 more

**Function Issues:**
- Function 'create_fallback_logger' is missing a docstring
- Function 'create_fallback_decorator' is missing a docstring
- Function 'decorator' is missing a docstring
- ... and 828 more

**Security Issues:**
- Potentially unsafe attribute access 'Path(__file__).parent.parent.parent' without null check
- Potentially unsafe attribute access 'Path(__file__).parent.parent' without null check
- Potentially unsafe attribute access 'Path(__file__).parent' without null check
- ... and 634 more

### step01_5_data_converter.py
**Path:** `/workspace/src/training/steps/data_collection/data_preparation/step01_5_data_converter.py`
**Lines of Code:** 1420
**Total Issues:** 1536 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'context' for function 'log_step_metrics'
- Unknown keyword argument 'context' for function 'log_step_metrics'
- Unknown keyword argument 'symbol' for function 'converter.execute'
- ... and 4 more

**Import Issues:**
- Undefined name: Callable
- Undefined name: decorator
- Undefined name: create_fallback_logger
- ... and 235 more

**Function Issues:**
- Function 'create_fallback_logger' is missing a docstring
- Function 'create_fallback_decorator' is missing a docstring
- Function 'decorator' is missing a docstring
- ... and 668 more

**Security Issues:**
- Potentially unsafe attribute access 'Path(__file__).parent.parent.parent' without null check
- Potentially unsafe attribute access 'Path(__file__).parent.parent' without null check
- Potentially unsafe attribute access 'Path(__file__).parent' without null check
- ... and 617 more

### advanced_feature_engineering.py
**Path:** `/workspace/src/analyst/advanced_feature_engineering.py`
**Lines of Code:** 2430
**Total Issues:** 1406 (Fixed: 0)

**Import Issues:**
- Undefined name: pd
- Undefined name: initialization_error
- Undefined name: pd
- ... and 132 more

**Function Issues:**
- Function '__init__' is missing a docstring
- Function '__init__' is missing a docstring
- Function '__init__' is missing a docstring
- ... and 521 more

**Security Issues:**
- Potentially unsafe subscript access 'dict[(str, Any)]' without existence check
- Potentially unsafe attribute access 'logging.getLogger' without null check
- Potentially unsafe attribute access 'self.__class__.__name__' without null check
- ... and 744 more

### raw_data_quality_checker.py
**Path:** `/workspace/src/training/steps/data_collection/raw_data_quality_checker.py`
**Lines of Code:** 1358
**Total Issues:** 1368 (Fixed: 0)

**Import Issues:**
- Undefined name: Callable
- Undefined name: pd
- Undefined name: pd
- ... and 167 more

**Function Issues:**
- Function '__init__' is missing a docstring
- Function 'wrapper' is missing a docstring
- Function 'wrapper' is missing a docstring
- ... and 675 more

**Security Issues:**
- Potentially unsafe attribute access 'warnings.filterwarnings' without null check
- Potentially unsafe subscript access 'dict[(str, Any)]' without existence check
- Potentially unsafe attribute access 'system_logger.getChild' without null check
- ... and 517 more

### step03_regime_discovery_features.py
**Path:** `/workspace/src/training/steps/market_analysis/hmm_clustering/step03_regime_discovery_features.py`
**Lines of Code:** 768
**Total Issues:** 1368 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'min_duration' for function 'self._calculate_regime_persistence'
- Unknown keyword argument 'min_duration' for function 'self._calculate_regime_persistence'
- Unknown keyword argument 'axis' for function 'transition_matrix.sum'
- ... and 1 more

**Import Issues:**
- Undefined name: Dict
- Undefined name: Any
- Undefined name: Optional
- ... and 10 more

**Function Issues:**
- Function '__init__' is missing a docstring
- Function 'filterwarnings' is called but not defined, imported, or built-in
- Function 'fillna' is called but not defined, imported, or built-in
- ... and 579 more

**Security Issues:**
- Potentially unsafe attribute access 'warnings.filterwarnings' without null check
- Potentially unsafe subscript access 'Dict[(str, Any)]' without existence check
- Potentially unsafe subscript access 'Optional[np.ndarray]' without existence check
- ... and 766 more

### ml_confidence_predictor.py
**Path:** `/workspace/src/analyst/ml_confidence_predictor.py`
**Lines of Code:** 1718
**Total Issues:** 1226 (Fixed: 0)

**Import Issues:**
- Undefined name: np
- Undefined name: np
- Undefined name: np
- ... and 107 more

**Function Issues:**
- Function 'execute_order_with_strategy' has 8 arguments (consider using a config object)
- Function 'compute_mixture_scores' is missing a docstring
- Function 'compute_mixture_scores' has 11 arguments (consider using a config object)
- ... and 578 more

**Security Issues:**
- Potentially unsafe subscript access 'dict[(str, Any)]' without existence check
- Potentially unsafe subscript access 'dict[(str, Any)]' without existence check
- Potentially unsafe attribute access 'logging.getLogger' without null check
- ... and 532 more

### step01_5_data_converter_validator.py
**Path:** `/workspace/src/training/steps/data_collection/step01_5_data_converter_validator.py`
**Lines of Code:** 1128
**Total Issues:** 1174 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'interval' for function 'psutil.cpu_percent'

**Import Issues:**
- Undefined name: check_dependencies
- Undefined name: dep
- Undefined name: dep
- ... and 54 more

**Function Issues:**
- Function '__init__' is missing a docstring
- Function 'async_wrapper' is missing a docstring
- Function 'sync_wrapper' is missing a docstring
- ... and 582 more

**Security Issues:**
- Potentially unsafe subscript access 'Path(__file__).resolve().parents[2]' without existence check
- Potentially unsafe attribute access 'Path(__file__).resolve().parents' without null check
- Potentially unsafe attribute access 'Path(__file__).resolve' without null check
- ... and 528 more

### matrix_diverse_lookback_optimizer.py
**Path:** `/workspace/src/training/matrix_diverse_lookback_optimizer.py`
**Lines of Code:** 903
**Total Issues:** 1124 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'sampler' for function 'optuna.create_study'
- Unknown keyword argument 'n_trials' for function 'study.optimize'

**Import Issues:**
- Undefined name: i
- Undefined name: i
- Undefined name: i
- ... and 13 more

**Function Issues:**
- Function 'objective' is missing a docstring
- Function 'constraint' is missing a docstring
- Function 'objective' is missing a docstring
- ... and 444 more

**Security Issues:**
- Potentially unsafe subscript access 'dict[(str, Any)]' without existence check
- Potentially unsafe attribute access 'system_logger.getChild' without null check
- Potentially unsafe attribute access 'self.output_dir.mkdir' without null check
- ... and 656 more

### sr_ml_enhancer.py
**Path:** `/workspace/src/tactician/sr_levels/sr_ml_enhancer.py`
**Lines of Code:** 1223
**Total Issues:** 1080 (Fixed: 0)

**Import Issues:**
- Undefined name: MLFeatureSet
- Undefined name: MLFeatureSet
- Undefined name: target
- ... and 35 more

**Function Issues:**
- Function 'append' is called but not defined, imported, or built-in
- Function 'append' is called but not defined, imported, or built-in
- Function 'array' is called but not defined, imported, or built-in
- ... and 570 more

**Security Issues:**
- Potentially unsafe attribute access 'np.ndarray' without null check
- Potentially unsafe subscript access 'List[str]' without existence check
- Potentially unsafe attribute access 'np.ndarray' without null check
- ... and 466 more

### unified_regime_classifier.py
**Path:** `/workspace/src/analyst/unified_regime_classifier.py`
**Lines of Code:** 1208
**Total Issues:** 1062 (Fixed: 0)

**Import Issues:**
- Undefined name: Callable
- Undefined name: decorator
- Undefined name: Callable
- ... and 42 more

**Function Issues:**
- Function 'handles_errors' is missing a docstring
- Function 'decorator' is missing a docstring
- Function 'warning' is missing a docstring
- ... and 528 more

**Security Issues:**
- Potentially unsafe attribute access 'logging.getLogger' without null check
- Potentially unsafe subscript access 'dict[(str, Any)]' without existence check
- Potentially unsafe attribute access 'logging.getLogger' without null check
- ... and 483 more

### enhanced_matrix_operations.py
**Path:** `/workspace/src/training/enhanced_matrix_operations.py`
**Lines of Code:** 1647
**Total Issues:** 1030 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'features_df' for function 'autoencoder_generator.generate_features'
- Unknown keyword argument 'regime_name' for function 'autoencoder_generator.generate_features'
- Unknown keyword argument 'labels' for function 'autoencoder_generator.generate_features'
- ... and 1 more

**Import Issues:**
- Undefined name: MatrixOperationsConfig
- Undefined name: pd
- Undefined name: np
- ... and 165 more

**Function Issues:**
- Function 'select_features_step2' has 8 arguments (consider using a config object)
- Function 'MatrixOperationsConfig' is called but not defined, imported, or built-in
- Function 'all' is called but not defined, imported, or built-in
- ... and 384 more

**Security Issues:**
- Potentially unsafe subscript access 'dict[(str, Any)]' without existence check
- Potentially unsafe attribute access 'system_logger.getChild' without null check
- Potentially unsafe attribute access 'config.get('feature_reduction', {}).get' without null check
- ... and 468 more

### step02_5_sr_optimization.py
**Path:** `/workspace/src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py`
**Lines of Code:** 654
**Total Issues:** 1019 (Fixed: 0)

**Import Issues:**
- Undefined name: async_wrapper
- Undefined name: sync_wrapper
- Undefined name: async_wrapper
- ... and 36 more

**Function Issues:**
- Function 'async_wrapper' is missing a docstring
- Function 'sync_wrapper' is missing a docstring
- Function 'async_wrapper' is missing a docstring
- ... and 444 more

**Security Issues:**
- Potentially unsafe attribute access 'system_logger.getChild' without null check
- Potentially unsafe subscript access 'function_call_tracker['call_count']' without existence check
- Potentially unsafe subscript access 'function_call_tracker['call_count']' without existence check
- ... and 530 more

### multi_output_model_trainer.py
**Path:** `/workspace/src/training/multi_output_model_trainer.py`
**Lines of Code:** 930
**Total Issues:** 963 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'eval_set' for function 'direction_model.fit'
- Unknown keyword argument 'early_stopping_rounds' for function 'direction_model.fit'
- Unknown keyword argument 'verbose' for function 'direction_model.fit'
- ... and 23 more

**Import Issues:**
- Undefined name: MultiOutputModelConfig
- Undefined name: level
- Undefined name: level
- ... and 60 more

**Function Issues:**
- Function '__init__' is missing a docstring
- Function '__init__' has 18 arguments (consider using a config object)
- Function '__init__' is missing a docstring
- ... and 417 more

**Security Issues:**
- Potentially unsafe subscript access 'Optional[List[str]]' without existence check
- Potentially unsafe subscript access 'List[str]' without existence check
- Potentially unsafe subscript access 'List[str]' without existence check
- ... and 451 more

### feature_output_validator.py
**Path:** `/workspace/src/utils/feature_output_validator.py`
**Lines of Code:** 612
**Total Issues:** 876 (Fixed: 0)

**Import Issues:**
- Undefined name: OutputValidationLevel
- Undefined name: OutputValidationIssue
- Undefined name: pd
- ... and 201 more

**Function Issues:**
- Function '__init__' is missing a docstring
- Function 'filterwarnings' is called but not defined, imported, or built-in
- Function 'type' is called but not defined, imported, or built-in
- ... and 332 more

**Security Issues:**
- Potentially unsafe attribute access 'warnings.filterwarnings' without null check
- Potentially unsafe subscript access 'dict[(str, Any)]' without existence check
- Potentially unsafe subscript access 'dict[(str, Any)]' without existence check
- ... and 334 more

### common_operations.py
**Path:** `/workspace/src/utils/common_operations.py`
**Lines of Code:** 1069
**Total Issues:** 862 (Fixed: 0)

**Syntax Errors:**
- Unknown keyword argument 'deep' for function 'df.copy'

**Import Issues:**
- Undefined name: _PDStub
- Undefined name: ensure_directory
- Undefined name: os
- ... and 58 more

**Function Issues:**
- Class '_PDStub' is missing a docstring
- Class 'DataFrame' is missing a docstring
- Class 'Series' is missing a docstring
- ... and 473 more

**Security Issues:**
- Potentially unsafe subscript access 'Union[(str, Path)]' without existence check
- Potentially unsafe attribute access 'json.load' without null check
- Potentially unsafe subscript access 'Union[(pd.DataFrame, Dict[str, Any])]' without existence check
- ... and 321 more

## Clean Files
**39 files with no issues found**