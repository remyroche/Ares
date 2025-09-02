# Code Complexity Report

## High Complexity Functions Requiring Manual Refactoring


### src/training/steps/combined_fractional_system.py

- M 84:4 HMMFractionalIntegration.calculate_regime_quality: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 387:4 CombinedFractionalSystem.get_performance_summary: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 304:4 CombinedFractionalSystem._calculate_performance_metrics: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/data_preparation_components/aggtrades_data_formatting.py

- M 143:4 DataFileReformatter._process_format2: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/data_preparation_components/training_validation_config.py

- M 271:4 DataValidator._validate_klines_quality: Complexity = 12
  **MODERATE: Could benefit from simplification**
- F 429:0 validate_data_collection: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/enhanced_step1_5_data_converter.py

- M 125:4 OptimizedUnifiedDataProcessor._transform_to_unified_format: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/fractional_feature_selector.py

- M 604:4 FractionalFeatureSelector.get_selection_summary: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/hmm_feature_enhancer.py

- M 174:4 HMMFeatureEnhancer._add_missing_technical_indicators: Complexity = 29
  **HIGH: Consider refactoring for better maintainability**
- M 128:4 HMMFeatureEnhancer._add_regime_interaction_features: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 228:4 HMMFeatureEnhancer._add_regime_enhanced_features: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/integrated_data_quality_pipeline.py

- M 59:4 IntegratedDataQualityPipeline.run_comprehensive_quality_pipeline: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 344:4 IntegratedDataQualityPipeline.generate_quality_report: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/multi_timeframe_hmm_ensemble.py

- M 562:4 MultiTimeframeHMMEnsemble._stacking_ensemble: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 389:4 MultiTimeframeHMMEnsemble.predict: Complexity = 13
  **MODERATE: Could benefit from simplification**

### src/training/steps/raw_data_quality_checker.py

- M 1415:4 RawDataQualityChecker._validate_feature_engineering_requirements: Complexity = 36
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 1071:4 RawDataQualityChecker._fix_datetime_index: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 1183:4 RawDataQualityChecker._estimate_timeframe_from_data: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 2091:4 RawDataQualityChecker._generate_recommendations: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 274:4 RawDataQualityChecker.validate_raw_data: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 1225:4 RawDataQualityChecker._validate_data_completeness: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 690:4 RawDataQualityChecker._determine_timeframe_from_data: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/sr_outcome_model_trainer.py

- M 757:4 SROutcomeModelTrainer._evaluate_model: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 265:4 SROutcomeModelTrainer._extract_features: Complexity = 16
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1/aggtrades_validator.py

- M 91:4 AggtradesValidator.validate_file_format: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 286:4 AggtradesValidator.validate_all_aggtrades: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 191:4 AggtradesValidator.fix_file_format: Complexity = 14
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1/comprehensive_gap_filler.py

- M 571:4 ComprehensiveGapFiller.fill_gap_until_complete: Complexity = 33
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 897:4 ComprehensiveGapFiller.process_all_data_types: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 474:4 ComprehensiveGapFiller._fetch_klines_from_binance_vision: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 745:4 ComprehensiveGapFiller.regenerate_timeframe_files: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 259:4 ComprehensiveGapFiller._fetch_aggtrades_from_binance_vision: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1/data_quality_monitor.py

- M 406:4 DataQualityMonitor.get_alerts: Complexity = 15
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1/data_resampler.py

- M 563:4 DataPreparation.validate_resampled_data: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 732:4 DataPreparation.validate_resampled_data_quality: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 408:4 DataPreparation.resample_all_timeframes: Complexity = 14
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1/enhanced_data_quality_manager.py

- M 79:4 EnhancedDataQualityManager.comprehensive_quality_check: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 217:4 EnhancedDataQualityManager._fill_data_gaps: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1/gap_filler_pipeline.py

- M 173:4 GapFillerPipeline.fill_gap_until_complete: Complexity = 14
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1/missing_data_downloader_and_gap_filler.py

- M 614:4 MissingDataDownloaderAndGapFiller.download_all_missing_data: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1/run_step1.py

- F 33:0 main: Complexity = 27
  **HIGH: Consider refactoring for better maintainability**

### src/training/steps/step1/step1_orchestrator.py

- M 69:4 Step1Orchestrator.run_complete_step1: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- M 422:4 Step1Orchestrator.generate_comprehensive_report: Complexity = 18
  **MODERATE: Could benefit from simplification**
- C 36:0 Step1Orchestrator: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1/validate_and_fix_aggtrades_format.py

- M 126:4 AggtradesFormatValidator.validate_file_format: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 394:4 AggtradesFormatValidator.fix_file_format: Complexity = 14
  **MODERATE: Could benefit from simplification**

### src/training/steps/step10_unified_regime_intelligence.py

- M 989:4 UnifiedRegimeIntelligenceStep._create_sequences: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 414:4 UnifiedRegimeIntelligenceStep.train: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 1301:4 UnifiedRegimeIntelligenceStep._train_model: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 1144:4 UnifiedRegimeIntelligenceStep._log_feature_count_info: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 1196:4 UnifiedRegimeIntelligenceStep._detect_intensity_transition: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 725:4 UnifiedRegimeIntelligenceStep._create_cross_timeframe_correlations: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 552:4 UnifiedRegimeIntelligenceStep._prepare_training_data: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 644:4 UnifiedRegimeIntelligenceStep._generate_intensity_scores: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 1639:4 UnifiedRegimeIntelligenceStep._determine_position_action: Complexity = 13
  **MODERATE: Could benefit from simplification**
- F 2053:0 run_step: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step10_unified_regime_intelligence_validator.py

- M 283:4 UnifiedRegimeIntelligenceValidator.validate_training_process: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 202:4 UnifiedRegimeIntelligenceValidator.validate_model_architecture: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 429:4 UnifiedRegimeIntelligenceValidator.validate_predictions: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 111:4 UnifiedRegimeIntelligenceValidator.validate_data_quality: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 353:4 UnifiedRegimeIntelligenceValidator.validate_artifacts: Complexity = 12
  **MODERATE: Could benefit from simplification**
- C 28:0 UnifiedRegimeIntelligenceValidator: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step11_analyst_creation_validator.py

- M 24:4 Step11AnalystCreationValidator.validate_step11_analyst_creation: Complexity = 13
  **MODERATE: Could benefit from simplification**

### src/training/steps/step12_analyst_enhancement.py

- M 598:4 RegimeAwareAnalystEnhancementStep._load_regime_data: Complexity = 27
  **HIGH: Consider refactoring for better maintainability**
- M 3051:4 RegimeAwareAnalystEnhancementStep._apply_data_driven_feature_selection: Complexity = 20
  **MODERATE: Could benefit from simplification**
- M 536:4 RegimeAwareAnalystEnhancementStep._load_models: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 860:4 RegimeAwareAnalystEnhancementStep._create_target_from_data: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 2553:4 RegimeAwareAnalystEnhancementStep._try_stable_shap_feature_selection: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 304:4 RegimeAwareAnalystEnhancementStep.execute: Complexity = 16
  **MODERATE: Could benefit from simplification**
- F 3921:0 run_step: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 2891:4 RegimeAwareAnalystEnhancementStep._robust_feature_selection_single_bootstrap: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 3541:4 RegimeAwareAnalystEnhancementStep._apply_architecture_specific_feature_selection: Complexity = 15
  **MODERATE: Could benefit from simplification**
- F 103:0 _normalized_numpy_bitgen_ctor: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 1942:4 RegimeAwareAnalystEnhancementStep._log_feature_stability_warnings: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 2400:4 RegimeAwareAnalystEnhancementStep._perform_stability_selection: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 942:4 RegimeAwareAnalystEnhancementStep._enhance_single_model: Complexity = 11
  **MODERATE: Could benefit from simplification**
- M 1910:4 RegimeAwareAnalystEnhancementStep._log_mutual_information_warnings: Complexity = 11
  **MODERATE: Could benefit from simplification**
- M 2800:4 RegimeAwareAnalystEnhancementStep._robust_stable_feature_selection: Complexity = 11
  **MODERATE: Could benefit from simplification**
- M 2972:4 RegimeAwareAnalystEnhancementStep._categorize_features_by_tier: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step12_analyst_enhancement_validator.py

- M 346:4 Step6HMMBasedEnhancementValidator._validate_enhancement_quality: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 448:4 Step6HMMBasedEnhancementValidator._extract_estimator_from_artifact: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 33:4 Step6HMMBasedEnhancementValidator.validate: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 270:4 Step6HMMBasedEnhancementValidator._validate_performance_improvement: Complexity = 14
  **MODERATE: Could benefit from simplification**
- C 27:0 Step6HMMBasedEnhancementValidator: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 205:4 Step6HMMBasedEnhancementValidator._validate_enhanced_model_files: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step13_analyst_ensemble_creation.py

- M 145:4 AnalystEnsembleCreationStep._create_ensemble: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step14_tactician_labeling.py

- M 764:4 TacticianLabelingStep._load_analyst_ensembles: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step14_tactician_labeling_validator.py

- M 339:4 Step8TacticianLabelingValidator._validate_labeling_consistency: Complexity = 25
  **HIGH: Consider refactoring for better maintainability**
- M 168:4 Step8TacticianLabelingValidator._validate_signal_quality: Complexity = 23
  **HIGH: Consider refactoring for better maintainability**
- M 540:4 Step8TacticianLabelingValidator._validate_signal_distribution: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- C 29:0 Step8TacticianLabelingValidator: Complexity = 15
  **MODERATE: Could benefit from simplification**

### src/training/steps/step15_tactician_specialist_training.py

- M 503:4 RegimeAwareTacticianSpecialistTrainingStep._train_tactician_models: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 232:4 RegimeAwareTacticianSpecialistTrainingStep._enhance_training_data_with_sr_context: Complexity = 14
  **MODERATE: Could benefit from simplification**

### src/training/steps/step15_tactician_specialist_training_validator.py

- M 322:4 Step9TacticianSpecialistTrainingValidator._validate_tactician_model_quality: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- M 241:4 Step9TacticianSpecialistTrainingValidator._validate_tactician_training_metrics: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 445:4 Step9TacticianSpecialistTrainingValidator._unwrap_estimator: Complexity = 16
  **MODERATE: Could benefit from simplification**
- C 26:0 Step9TacticianSpecialistTrainingValidator: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 154:4 Step9TacticianSpecialistTrainingValidator._validate_tactician_model_performance: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step16_confidence_calibration.py

- M 93:4 RegimeAwareConfidenceCalibrationStep.execute: Complexity = 46
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- F 864:0 _calibrate_regime_aware_analyst_models: Complexity = 11
  **MODERATE: Could benefit from simplification**
- F 934:0 _calibrate_regime_aware_tactician_models: Complexity = 11
  **MODERATE: Could benefit from simplification**
- M 603:4 RegimeAwareConfidenceCalibrationStep._calibrate_analyst_ensembles: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step17_final_parameters_optimization/advanced_optimization_engine.py

- M 483:4 CrossValidationPruner._evaluate_parameter_value: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step17_final_parameters_optimization/evaluation_engine.py

- M 243:4 AdvancedEvaluationEngine._calculate_performance_metrics: Complexity = 24
  **HIGH: Consider refactoring for better maintainability**
- M 502:4 AdvancedEvaluationEngine.calculate_composite_score: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step17_final_parameters_optimization/hyperparameter_optimization_config.py

- M 589:4 HyperparameterOptimizationConfig.validate_search_space: Complexity = 13
  **MODERATE: Could benefit from simplification**

### src/training/steps/step17_final_parameters_optimization/optimized_optuna_optimization_enhanced.py

- M 631:4 VectorizedOptunaOptimizer.optimize: Complexity = 13
  **MODERATE: Could benefit from simplification**

### src/training/steps/step17_final_parameters_optimization/optimized_step17_implementation.py

- M 363:4 IntelligentParameterPruner._evaluate_single_parameter: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 659:4 HierarchicalOptimizer.run_hierarchical_optimization: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 1017:4 HierarchicalOptimizer._evaluate_parameter_group_advanced: Complexity = 13
  **MODERATE: Could benefit from simplification**

### src/training/steps/step17_final_parameters_optimization/sr_optuna_optimization.py

- M 548:4 SROptunaOptimizer._create_optimization_result: Complexity = 16
  **MODERATE: Could benefit from simplification**

### src/training/steps/step17_final_parameters_optimization.py

- M 1161:4 FinalParametersOptimizationStep._generate_optimization_recommendations: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 1692:4 FinalParametersOptimizationStep._evaluate_predictions: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step17_final_parameters_optimization_new.py

- M 636:4 FinalParametersOptimizationStepNew._evaluate_training_optimization_params: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 345:4 FinalParametersOptimizationStepNew._evaluate_configuration: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 556:4 FinalParametersOptimizationStepNew._evaluate_technical_indicators_params: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 686:4 FinalParametersOptimizationStepNew._evaluate_regime_transitions_params: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 985:4 FinalParametersOptimizationStepNew._extract_tactician_optimization_results: Complexity = 13
  **MODERATE: Could benefit from simplification**

### src/training/steps/step17_final_parameters_optimization_validator.py

- M 222:4 Step12FinalParametersOptimizationValidator._validate_optimization_convergence: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 312:4 Step12FinalParametersOptimizationValidator._validate_optimized_parameters: Complexity = 17
  **MODERATE: Could benefit from simplification**
- C 21:0 Step12FinalParametersOptimizationValidator: Complexity = 11
  **MODERATE: Could benefit from simplification**
- M 154:4 Step12FinalParametersOptimizationValidator._validate_optimization_quality: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step18_walk_forward_validation_validator.py

- M 164:4 Step13WalkForwardValidationValidator._validate_walk_forward_performance: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 254:4 Step13WalkForwardValidationValidator._validate_walk_forward_stability: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 331:4 Step13WalkForwardValidationValidator._validate_walk_forward_consistency: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step19_monte_carlo_validation_validator.py

- M 251:4 Step14MonteCarloValidationValidator._validate_performance_distribution: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- M 373:4 Step14MonteCarloValidationValidator._validate_monte_carlo_robustness: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 160:4 Step14MonteCarloValidationValidator._validate_statistical_significance: Complexity = 16
  **MODERATE: Could benefit from simplification**
- C 24:0 Step14MonteCarloValidationValidator: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1_5_data_converter.py

- M 608:4 ParquetDatasetManager.write_partitioned_dataset: Complexity = 34
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 516:4 ParquetDatasetManager.enforce_schema: Complexity = 23
  **HIGH: Consider refactoring for better maintainability**
- M 742:4 ParquetDatasetManager.scan_dataset: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- M 231:4 ColumnVerifier._check_calculation_feasibility: Complexity = 20
  **MODERATE: Could benefit from simplification**
- M 161:4 ColumnVerifier.verify_missing_columns: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 871:4 ParquetDatasetManager.update_manifest: Complexity = 15
  **MODERATE: Could benefit from simplification**
- C 493:0 ParquetDatasetManager: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 812:4 ParquetDatasetManager._build_filter_expression: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 1093:4 UnifiedDataConverter._process_incremental_updates: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 352:4 ColumnVerifier._calculate_vwap_features: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 964:4 UnifiedDataConverter.execute: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 1185:4 UnifiedDataConverter._process_data_incrementally: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 1520:4 UnifiedDataConverter._verify_unified_data_quality: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 1659:4 UnifiedDataConverter._download_klines_data: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1_5_data_converter_validator.py

- M 185:4 Step1_5DataConverterValidator._validate_single_unified_file: Complexity = 15
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1_data_collection.py

- M 677:4 DataCollectionStep._log_detailed_data_extract: Complexity = 41
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 288:4 DataCollectionStep._run_standardized_quality_check: Complexity = 20
  **MODERATE: Could benefit from simplification**
- F 871:0 run_step: Complexity = 19
  **MODERATE: Could benefit from simplification**
- C 88:0 DataCollectionStep: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 428:4 DataCollectionStep._validate_downloaded_data: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 616:4 DataCollectionStep._run_comprehensive_validation: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step1_data_collection_validator.py

- M 299:4 Step1DataCollectionValidator._validate_data_characteristics: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 33:4 Step1DataCollectionValidator.validate: Complexity = 12
  **MODERATE: Could benefit from simplification**
- C 20:0 Step1DataCollectionValidator: Complexity = 11
  **MODERATE: Could benefit from simplification**
- M 198:4 Step1DataCollectionValidator._validate_consolidated_data_quality: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step21_saving.py

- M 244:4 SavingStep._save_to_mlflow: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step21_saving_validator.py

- M 379:4 Step21SavingValidator._validate_final_model_quality: Complexity = 24
  **HIGH: Consider refactoring for better maintainability**
- M 348:4 Step21SavingValidator._unwrap_estimator: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 247:4 Step21SavingValidator._validate_file_integrity: Complexity = 15
  **MODERATE: Could benefit from simplification**
- C 26:0 Step21SavingValidator: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step2_5_sr_optimization.py

- M 974:4 SROptimizationStep._generate_comparison_insights: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 1111:4 SROptimizationStep._combine_optimization_results: Complexity = 14
  **MODERATE: Could benefit from simplification**

### src/training/steps/step2_5_sr_optimization_validator.py

- M 153:4 SROptimizationValidator._validate_optimized_parameters: Complexity = 20
  **MODERATE: Could benefit from simplification**
- M 92:4 SROptimizationValidator._validate_optimization_results: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step2_data_reading.py

- M 215:4 DataReadingStep.validate_data_quality: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step2_data_reading_validator.py

- F 39:0 run_validator: Complexity = 25
  **HIGH: Consider refactoring for better maintainability**

### src/training/steps/step2_feature_engineering_validator.py

- M 311:4 Step2FeatureEngineeringValidator._validate_feature_quality: Complexity = 23
  **HIGH: Consider refactoring for better maintainability**
- M 197:4 Step2FeatureEngineeringValidator._validate_labeling_quality: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 454:4 Step2FeatureEngineeringValidator._validate_minimum_relevant_features: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 567:4 Step2FeatureEngineeringValidator._validate_data_balance: Complexity = 15
  **MODERATE: Could benefit from simplification**
- C 22:0 Step2FeatureEngineeringValidator: Complexity = 13
  **MODERATE: Could benefit from simplification**

### src/training/steps/step3_hmm_regime_discovery.py

- M 375:4 HMMRegimeDiscoveryStep._log_step3_artifacts_to_mlflow: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 1318:4 HMMRegimeDiscoveryStep._perform_simple_regime_discovery: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 1059:4 HMMRegimeDiscoveryStep._log_feature_categories: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 197:4 HMMRegimeDiscoveryStep.execute: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step3_hmm_regime_discovery_validator.py

- F 23:0 run_validator: Complexity = 28
  **HIGH: Consider refactoring for better maintainability**

### src/training/steps/step3_parameter_optimization.py

- M 391:4 ParameterOptimizationStep._combine_optimization_results: Complexity = 14
  **MODERATE: Could benefit from simplification**

### src/training/steps/step3_parameter_optimization_validator.py

- M 105:4 Step3ParameterOptimizationValidator._validate_optimization_results: Complexity = 23
  **HIGH: Consider refactoring for better maintainability**
- M 281:4 Step3ParameterOptimizationValidator._validate_optimization_metrics: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 176:4 Step3ParameterOptimizationValidator._validate_optimization_config: Complexity = 15
  **MODERATE: Could benefit from simplification**
- C 21:0 Step3ParameterOptimizationValidator: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 29:4 Step3ParameterOptimizationValidator.validate_step3_parameter_optimization: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step4_analyst_labeling_feature_engineering_components/optimized_triple_barrier_labeling.py

- M 131:4 OptimizedTripleBarrierLabeling.apply_triple_barrier_labeling_vectorized: Complexity = 32
  **CRITICAL: Very high complexity, consider breaking into smaller functions**

### src/training/steps/step4_analyst_labeling_feature_engineering_components/profit_based_feature_engineering.py

- M 490:4 ProfitBasedFeatureEngineering.get_feature_summary: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- M 534:4 ProfitBasedFeatureEngineering.select_features: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 169:4 ProfitBasedFeatureEngineering.apply_all_features: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step4_analyst_labeling_feature_engineering_components/regime_aware_triple_barrier_labeling.py

- M 371:4 RegimeAwareTripleBarrierLabeling._apply_regime_specific_labeling: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 265:4 RegimeAwareTripleBarrierLabeling.apply_regime_aware_triple_barrier_labeling: Complexity = 13
  **MODERATE: Could benefit from simplification**
- C 106:0 RegimeTripleBarrierConfig: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step4_analyst_labeling_feature_engineering_components/regime_specific_triple_barrier_optimizer.py

- M 502:4 RegimeSpecificTripleBarrierOptimizer._calculate_regime_performance_score: Complexity = 33
  **CRITICAL: Very high complexity, consider breaking into smaller functions**

### src/training/steps/step4_triple_barrier_method_validator.py

- F 39:0 run_validator: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step5_hmm_based_training_validator.py

- M 425:4 Step5HMMBasedTrainingValidator._validate_model_quality: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- M 609:4 Step5HMMBasedTrainingValidator._unwrap_estimator: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 351:4 Step5HMMBasedTrainingValidator._validate_training_metrics: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 38:4 Step5HMMBasedTrainingValidator.validate: Complexity = 12
  **MODERATE: Could benefit from simplification**
- C 32:0 Step5HMMBasedTrainingValidator: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step5_labeling.py

- M 444:4 LabelingStep._generate_comprehensive_labels: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 296:4 LabelingStep._log_step5_artifacts_and_report: Complexity = 13
  **MODERATE: Could benefit from simplification**

### src/training/steps/step6_feature_engineering.py

- F 897:0 _add_sr_features: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- F 464:0 _create_comprehensive_features: Complexity = 17
  **MODERATE: Could benefit from simplification**
- F 350:0 _categorize_features: Complexity = 16
  **MODERATE: Could benefit from simplification**
- F 207:0 run_step: Complexity = 14
  **MODERATE: Could benefit from simplification**

### src/training/steps/step6_feature_interaction_engineering.py

- M 396:4 FeatureInteractionEngine.extract_optimal_technical_indicators: Complexity = 40
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 286:4 FeatureInteractionEngine._extract_optimized_periods: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 356:4 FeatureInteractionEngine._validate_lookback_periods: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step7_enhanced_matrix_operations.py

- M 1361:4 Step7EnhancedMatrixOperations._generate_detailed_quality_report: Complexity = 51
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 869:4 Step7EnhancedMatrixOperations._analyze_enhanced_sr_feature_stability: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- M 831:4 Step7EnhancedMatrixOperations._analyze_enhanced_sr_feature_clusters: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 984:4 Step7EnhancedMatrixOperations._analyze_sr_optimization_parameters: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 926:4 Step7EnhancedMatrixOperations._analyze_enhanced_sr_feature_importance: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 1105:4 Step7EnhancedMatrixOperations._analyze_sr_feature_stability: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 155:4 Step7EnhancedMatrixOperations.execute: Complexity = 12
  **MODERATE: Could benefit from simplification**
- C 110:0 Step7EnhancedMatrixOperations: Complexity = 11
  **MODERATE: Could benefit from simplification**
- M 314:4 Step7EnhancedMatrixOperations._log_step7_artifacts_and_report: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step7_enhanced_matrix_operations_validator.py

- M 250:4 Step7EnhancedMatrixOperationsValidator._validate_operation_results: Complexity = 18
  **MODERATE: Could benefit from simplification**
- M 127:4 Step7EnhancedMatrixOperationsValidator._validate_config_file: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 299:4 Step7EnhancedMatrixOperationsValidator._validate_summary_file: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step8_regime_data_splitting_validator.py

- F 39:0 run_validator: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**

### src/training/steps/step9_5_hmm_lm_generalist_training.py

- M 602:4 HMMLMGeneralistTrainingStep._calculate_enhanced_tpsl_outcomes: Complexity = 20
  **MODERATE: Could benefit from simplification**
- M 192:4 HMMLMGeneralistTrainingStep._log_step9_5_artifacts_and_report: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 792:4 HMMLMGeneralistTrainingStep._sequence_to_features: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step9_5_hmm_lm_generalist_training_validator.py

- M 107:4 Step9_5HMMLMGeneralistTrainingValidator._validate_metadata_file: Complexity = 12
  **MODERATE: Could benefit from simplification**

### src/training/steps/step9_5_multi_timeframe_hmm_ensemble.py

- F 579:0 run_step: Complexity = 15
  **MODERATE: Could benefit from simplification**

### src/training/steps/step9_5_multi_timeframe_hmm_ensemble_validator.py

- M 36:4 Step9_5MultiTimeframeHMMEnsembleValidator.validate_step_outputs: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 201:4 Step9_5MultiTimeframeHMMEnsembleValidator.validate_input_data: Complexity = 12
  **MODERATE: Could benefit from simplification**
- C 18:0 Step9_5MultiTimeframeHMMEnsembleValidator: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step9_hmm_based_training.py

- M 4445:4 TCNTrainer._pre_filter_features: Complexity = 29
  **HIGH: Consider refactoring for better maintainability**
- M 485:4 HMMBasedTrainingStep.execute: Complexity = 28
  **HIGH: Consider refactoring for better maintainability**
- M 804:4 HMMBasedTrainingStep._load_feature_data: Complexity = 28
  **HIGH: Consider refactoring for better maintainability**
- M 1669:4 HMMBasedTrainingStep._train_regime_specific_models: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- M 3134:4 HMMBasedTrainingStep._create_training_summary: Complexity = 22
  **HIGH: Consider refactoring for better maintainability**
- M 3036:4 HMMBasedTrainingStep._create_feature_analysis_report: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 4614:4 TCNTrainer._calculate_comprehensive_scores: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 2597:4 HMMBasedTrainingStep._save_models: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 3227:4 HMMBasedTrainingStep._extract_estimator_from_artifact: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 2781:4 HMMBasedTrainingStep._save_enhanced_artifacts: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 1390:4 HMMBasedTrainingStep._load_hmm_composite_regime_data: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 1974:4 HMMBasedTrainingStep._add_regime_change_features: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 1185:4 HMMBasedTrainingStep._resample_features_to_timeframe: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 4150:4 TCNTrainer._train_and_optionally_refit: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 995:4 HMMBasedTrainingStep._load_and_combine_split_features: Complexity = 11
  **MODERATE: Could benefit from simplification**
- M 1541:4 HMMBasedTrainingStep._train_timeframe_model: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/step9_hmm_based_training_enhanced.py

- M 501:4 EnhancedHMMBasedTrainingStep.prepare_enhanced_data: Complexity = 15
  **MODERATE: Could benefit from simplification**
- F 1039:0 run_enhanced_step: Complexity = 13
  **MODERATE: Could benefit from simplification**

### src/training/steps/step9_hmm_based_training_validator.py

- F 39:0 run_validator: Complexity = 27
  **HIGH: Consider refactoring for better maintainability**

### src/training/steps/unified_data_loader.py

- M 215:4 UnifiedDataLoader._validate_unified_data: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 302:4 UnifiedDataLoader._load_unified_data_fallback: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 396:4 UnifiedDataLoader.get_data_info: Complexity = 13
  **MODERATE: Could benefit from simplification**
- C 78:0 UnifiedDataLoader: Complexity = 11
  **MODERATE: Could benefit from simplification**
- M 126:4 UnifiedDataLoader.load_unified_data: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/vectorized_advanced_feature_engineering.py

- M 2289:4 VectorizedAdvancedFeatureEngineering.engineer_features: Complexity = 147
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 4476:4 VectorizedAdvancedFeatureEngineering._generate_cross_timeframe_features: Complexity = 71
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 5449:4 VectorizedAdvancedFeatureEngineering._generate_interaction_features: Complexity = 67
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 5065:4 VectorizedAdvancedFeatureEngineering._engineer_difference_and_acceleration_features: Complexity = 55
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 4335:4 VectorizedAdvancedFeatureEngineering._generate_timeframe_features: Complexity = 29
  **HIGH: Consider refactoring for better maintainability**
- M 5792:4 VectorizedAdvancedFeatureEngineering._validate_enhanced_features: Complexity = 23
  **HIGH: Consider refactoring for better maintainability**
- M 5850:4 VectorizedAdvancedFeatureEngineering._log_feature_engineering_summary: Complexity = 21
  **HIGH: Consider refactoring for better maintainability**
- M 524:4 WaveletFeatureCache._features_to_dataframe: Complexity = 20
  **MODERATE: Could benefit from simplification**
- M 2080:4 VectorizedAdvancedFeatureEngineering._handle_nan_values_inline: Complexity = 20
  **MODERATE: Could benefit from simplification**
- M 1487:4 VectorizedAdvancedFeatureEngineering.initialize: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 3160:4 VectorizedAdvancedFeatureEngineering._engineer_microstructure_features_vectorized: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 3529:4 VectorizedAdvancedFeatureEngineering._track_nan_origins: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 4096:4 VectorizedAdvancedFeatureEngineering._generate_traditional_multi_timeframe_features: Complexity = 19
  **MODERATE: Could benefit from simplification**
- M 5627:4 VectorizedAdvancedFeatureEngineering._generate_cross_timeframe_features: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 1964:4 VectorizedAdvancedFeatureEngineering._handle_nan_values_comprehensive: Complexity = 16
  **MODERATE: Could benefit from simplification**
- M 1215:4 VectorizedWaveletTransformAnalyzer.analyze_wavelet_transforms: Complexity = 15
  **MODERATE: Could benefit from simplification**
- M 1901:4 VectorizedAdvancedFeatureEngineering._generate_simple_timeframe_features: Complexity = 15
  **MODERATE: Could benefit from simplification**
- C 1348:0 VectorizedAdvancedFeatureEngineering: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 3758:4 VectorizedAdvancedFeatureEngineering._engineer_adaptive_indicators_vectorized: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 4767:4 VectorizedAdvancedFeatureEngineering._validate_and_clean_features: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 5992:4 VectorizedAdvancedFeatureEngineering._handle_irregular_time_intervals: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 3879:4 VectorizedAdvancedFeatureEngineering._select_optimal_features_vectorized: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 4846:4 VectorizedAdvancedFeatureEngineering._ensure_pickle_safe_features: Complexity = 11
  **MODERATE: Could benefit from simplification**

### src/training/steps/vectorized_labelling_orchestrator.py

- M 278:4 VectorizedLabellingOrchestrator.orchestrate_labeling_and_feature_engineering: Complexity = 69
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 1946:4 VectorizedFeatureSelector.select_optimal_features: Complexity = 52
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 1530:4 VectorizedLabellingOrchestrator._run_mutual_information_analysis: Complexity = 44
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 703:4 VectorizedLabellingOrchestrator._combine_features_and_labels_vectorized: Complexity = 31
  **CRITICAL: Very high complexity, consider breaking into smaller functions**
- M 1260:4 VectorizedLabellingOrchestrator._categorize_features: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 1335:4 VectorizedLabellingOrchestrator._list_other_features: Complexity = 17
  **MODERATE: Could benefit from simplification**
- M 149:4 VectorizedLabellingOrchestrator._log_feature_sample: Complexity = 15
  **MODERATE: Could benefit from simplification**
- C 1892:0 VectorizedFeatureSelector: Complexity = 15
  **MODERATE: Could benefit from simplification**
- C 64:0 VectorizedLabellingOrchestrator: Complexity = 14
  **MODERATE: Could benefit from simplification**
- M 956:4 VectorizedLabellingOrchestrator._remove_datetime_columns: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 1201:4 VectorizedLabellingOrchestrator._log_feature_dict_summary: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 1919:4 VectorizedFeatureSelector._remove_datetime_columns: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 2298:4 VectorizedDataNormalizer._remove_datetime_columns: Complexity = 13
  **MODERATE: Could benefit from simplification**
- M 1154:4 VectorizedLabellingOrchestrator._optimize_memory_usage_vectorized: Complexity = 12
  **MODERATE: Could benefit from simplification**
- M 914:4 VectorizedLabellingOrchestrator._ensure_ohlcv_data: Complexity = 11
  **MODERATE: Could benefit from simplification**