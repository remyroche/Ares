# Undefined Function Calls Report

This report identifies function calls that may be undefined (import issues).

## src/analyst/analyst.py

**Undefined function calls: 55**

- **Line 1467:** handles_errors - Direct call: handles_errors
- **Line 129:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 188:** handles_errors - Direct call: handles_errors
- **Line 200:** handles_errors - Direct call: handles_errors
- **Line 219:** handles_errors - Direct call: handles_errors
- **Line 236:** handles_errors - Direct call: handles_errors
- **Line 247:** handles_errors - Direct call: handles_errors
- **Line 258:** handles_errors - Direct call: handles_errors
- **Line 279:** handles_errors - Direct call: handles_errors
- **Line 304:** handles_errors - Direct call: handles_errors
- **Line 332:** handles_errors - Direct call: handles_errors
- **Line 353:** handles_errors - Direct call: handles_errors
- **Line 370:** handles_errors - Direct call: handles_errors
- **Line 389:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 538:** handles_errors - Direct call: handles_errors
- **Line 561:** handles_errors - Direct call: handles_errors
- **Line 935:** handles_errors - Direct call: handles_errors
- **Line 965:** handles_errors - Direct call: handles_errors
- **Line 1002:** validate_data_quality - Direct call: validate_data_quality
- **Line 1003:** traced - Direct call: traced
- **Line 1029:** validate_data_quality - Direct call: validate_data_quality
- **Line 1030:** traced - Direct call: traced
- **Line 1058:** validate_data_quality - Direct call: validate_data_quality
- **Line 1059:** traced - Direct call: traced
- **Line 1092:** validate_data_quality - Direct call: validate_data_quality
- **Line 1093:** traced - Direct call: traced
- **Line 1111:** validate_data_quality - Direct call: validate_data_quality
- **Line 1112:** traced - Direct call: traced
- **Line 1138:** validate_data_quality - Direct call: validate_data_quality
- **Line 1139:** traced - Direct call: traced
- **Line 1157:** validate_data_quality - Direct call: validate_data_quality
- **Line 1158:** traced - Direct call: traced
- **Line 1182:** validate_data_quality - Direct call: validate_data_quality
- **Line 1183:** traced - Direct call: traced
- **Line 1199:** handles_errors - Direct call: handles_errors
- **Line 1244:** handles_errors - Direct call: handles_errors
- **Line 1279:** handles_errors - Direct call: handles_errors
- **Line 1348:** handles_errors - Direct call: handles_errors
- **Line 1370:** handles_errors - Direct call: handles_errors
- **Line 1395:** handles_errors - Direct call: handles_errors
- **Line 1438:** handles_errors - Direct call: handles_errors
- **Line 378:** UnifiedRegimeClassifierFractal - Direct call: UnifiedRegimeClassifierFractal
- **Line 340:** FeatureEngineeringOrchestrator - Direct call: FeatureEngineeringOrchestrator
- **Line 268:** setup_dual_model_system - Direct call: setup_dual_model_system
- **Line 289:** setup_market_health_analyzer - Direct call: setup_market_health_analyzer
- **Line 314:** setup_liquidation_risk_model - Direct call: setup_liquidation_risk_model
- **Line 363:** MLConfidencePredictor - Direct call: MLConfidencePredictor
- **Line 272:** failed - Direct call: failed
- **Line 276:** initialization_error - Direct call: initialization_error
- **Line 295:** failed - Direct call: failed
- **Line 299:** initialization_error - Direct call: initialization_error
- **Line 320:** failed - Direct call: failed
- **Line 324:** failed - Direct call: failed
- **Line 329:** initialization_error - Direct call: initialization_error
- **Line 534:** failed - Direct call: failed

---

## src/analyst/autoencoder_feature_generator.py

**Undefined function calls: 24**

- **Line 17:** setup_logging - Direct call: setup_logging
- **Line 1132:** traced - Direct call: traced
- **Line 716:** Model - Direct call: Model
- **Line 717:** Model - Direct call: Model
- **Line 294:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 384:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 389:** TreeExplainer - Direct call: TreeExplainer
- **Line 564:** RobustScaler - Direct call: RobustScaler
- **Line 745:** EarlyStopping - Direct call: EarlyStopping
- **Line 745:** ReduceLROnPlateau - Direct call: ReduceLROnPlateau
- **Line 906:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 371:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 566:** StandardScaler - Direct call: StandardScaler
- **Line 568:** MinMaxScaler - Direct call: MinMaxScaler
- **Line 747:** TFKerasPruningCallback - Direct call: TFKerasPruningCallback
- **Line 910:** GradientBoostingClassifier - Direct call: GradientBoostingClassifier
- **Line 920:** train_test_split - Direct call: train_test_split
- **Line 921:** LogisticRegression - Direct call: LogisticRegression
- **Line 923:** permutation_importance - Direct call: permutation_importance
- **Line 19:** Path - Direct call: Path
- **Line 402:** TimeoutError - Direct call: TimeoutError
- **Line 874:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 876:** mutual_info_regression - Direct call: mutual_info_regression
- **Line 341:** train_test_split - Direct call: train_test_split

---

## src/analyst/candlestick_pattern_analyzer.py

**Undefined function calls: 2**

- **Line 32:** handles_errors - Direct call: handles_errors
- **Line 50:** handles_errors - Direct call: handles_errors

---

## src/analyst/data_utils.py

**Undefined function calls: 38**

- **Line 1093:** handles_errors - Direct call: handles_errors
- **Line 63:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 103:** handles_errors - Direct call: handles_errors
- **Line 135:** handles_errors - Direct call: handles_errors
- **Line 182:** handles_errors - Direct call: handles_errors
- **Line 214:** handles_errors - Direct call: handles_errors
- **Line 236:** handles_errors - Direct call: handles_errors
- **Line 258:** handles_errors - Direct call: handles_errors
- **Line 282:** handles_errors - Direct call: handles_errors
- **Line 304:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 370:** handles_errors - Direct call: handles_errors
- **Line 414:** handles_errors - Direct call: handles_errors
- **Line 468:** handles_errors - Direct call: handles_errors
- **Line 522:** handles_errors - Direct call: handles_errors
- **Line 579:** handles_errors - Direct call: handles_errors
- **Line 957:** handles_errors - Direct call: handles_errors
- **Line 981:** handles_errors - Direct call: handles_errors
- **Line 1010:** handles_errors - Direct call: handles_errors
- **Line 1063:** handles_errors - Direct call: handles_errors
- **Line 1409:** find_peaks - Direct call: find_peaks
- **Line 1186:** missing - Direct call: missing
- **Line 1626:** missing - Direct call: missing
- **Line 1209:** critical - Direct call: critical
- **Line 1266:** invalid - Direct call: invalid
- **Line 1289:** critical - Direct call: critical
- **Line 1298:** critical - Direct call: critical
- **Line 1642:** missing - Direct call: missing
- **Line 1697:** warning - Direct call: warning
- **Line 87:** invalid - Direct call: invalid
- **Line 99:** failed - Direct call: failed
- **Line 150:** invalid - Direct call: invalid
- **Line 156:** invalid - Direct call: invalid
- **Line 211:** initialization_error - Direct call: initialization_error
- **Line 279:** initialization_error - Direct call: initialization_error
- **Line 397:** invalid - Direct call: invalid
- **Line 402:** invalid - Direct call: invalid
- **Line 670:** missing - Direct call: missing
- **Line 1259:** critical - Direct call: critical

---

## src/analyst/di_analyst.py

**Undefined function calls: 15**

- **Line 136:** handles_errors - Direct call: handles_errors
- **Line 97:** DualModelSystem - Direct call: DualModelSystem
- **Line 104:** MarketHealthAnalyzer - Direct call: MarketHealthAnalyzer
- **Line 111:** LiquidationRiskModel - Direct call: LiquidationRiskModel
- **Line 118:** FeatureEngineeringOrchestrator - Direct call: FeatureEngineeringOrchestrator
- **Line 231:** AnalysisResult - Direct call: AnalysisResult
- **Line 147:** initialization_error - Direct call: initialization_error
- **Line 90:** failed - Direct call: failed
- **Line 173:** failed - Direct call: failed
- **Line 244:** failed - Direct call: failed
- **Line 263:** failed - Direct call: failed
- **Line 283:** AnalysisResult - Direct call: AnalysisResult
- **Line 299:** failed - Direct call: failed
- **Line 323:** failed - Direct call: failed
- **Line 347:** failed - Direct call: failed

---

## src/analyst/dynamic_regime_mapper.py

**Undefined function calls: 3**

- **Line 25:** handles_errors - Direct call: handles_errors
- **Line 38:** handles_errors - Direct call: handles_errors
- **Line 58:** handles_errors - Direct call: handles_errors

---

## src/analyst/enhanced_prediction_integrator.py

**Undefined function calls: 51**

- **Line 74:** handles_errors - Direct call: handles_errors
- **Line 77:** comprehensive_validation - Direct call: comprehensive_validation
- **Line 78:** performance_monitor - Direct call: performance_monitor
- **Line 112:** handles_errors - Direct call: handles_errors
- **Line 115:** traced - Direct call: traced
- **Line 116:** cached - Direct call: cached
- **Line 140:** handles_errors - Direct call: handles_errors
- **Line 172:** handles_errors - Direct call: handles_errors
- **Line 197:** handles_errors - Direct call: handles_errors
- **Line 222:** handles_errors - Direct call: handles_errors
- **Line 225:** traced - Direct call: traced
- **Line 252:** handles_errors - Direct call: handles_errors
- **Line 255:** validates - Direct call: validates
- **Line 256:** traced - Direct call: traced
- **Line 257:** performance_monitor - Direct call: performance_monitor
- **Line 329:** handles_errors - Direct call: handles_errors
- **Line 367:** handles_errors - Direct call: handles_errors
- **Line 408:** handles_errors - Direct call: handles_errors
- **Line 441:** handles_errors - Direct call: handles_errors
- **Line 473:** handles_errors - Direct call: handles_errors
- **Line 120:** Path - Direct call: Path
- **Line 145:** Path - Direct call: Path
- **Line 177:** Path - Direct call: Path
- **Line 202:** Path - Direct call: Path
- **Line 109:** failed - Direct call: failed
- **Line 122:** warning - Direct call: warning
- **Line 138:** error - Direct call: error
- **Line 147:** warning - Direct call: warning
- **Line 170:** error - Direct call: error
- **Line 179:** warning - Direct call: warning
- **Line 195:** error - Direct call: error
- **Line 204:** warning - Direct call: warning
- **Line 220:** error - Direct call: error
- **Line 249:** error - Direct call: error
- **Line 281:** error - Direct call: error
- **Line 326:** error - Direct call: error
- **Line 364:** error - Direct call: error
- **Line 405:** error - Direct call: error
- **Line 438:** error - Direct call: error
- **Line 470:** error - Direct call: error
- **Line 508:** error - Direct call: error
- **Line 540:** error - Direct call: error
- **Line 565:** error - Direct call: error
- **Line 590:** error - Direct call: error
- **Line 619:** error - Direct call: error
- **Line 135:** warning - Direct call: warning
- **Line 192:** warning - Direct call: warning
- **Line 217:** warning - Direct call: warning
- **Line 359:** warning - Direct call: warning
- **Line 400:** warning - Direct call: warning
- **Line 165:** warning - Direct call: warning

---

## src/analyst/enhanced_regime_predictor.py

**Undefined function calls: 20**

- **Line 37:** with_tracing_span - Direct call: with_tracing_span
- **Line 38:** handles_errors - Direct call: handles_errors
- **Line 68:** handles_errors - Direct call: handles_errors
- **Line 77:** handles_errors - Direct call: handles_errors
- **Line 88:** handles_errors - Direct call: handles_errors
- **Line 118:** handles_errors - Direct call: handles_errors
- **Line 133:** handles_errors - Direct call: handles_errors
- **Line 149:** handles_errors - Direct call: handles_errors
- **Line 169:** handles_errors - Direct call: handles_errors
- **Line 183:** handles_errors - Direct call: handles_errors
- **Line 205:** with_tracing_span - Direct call: with_tracing_span
- **Line 206:** handles_errors - Direct call: handles_errors
- **Line 251:** handles_errors - Direct call: handles_errors
- **Line 263:** with_tracing_span - Direct call: with_tracing_span
- **Line 264:** handles_errors - Direct call: handles_errors
- **Line 289:** handles_errors - Direct call: handles_errors
- **Line 273:** StandardScaler - Direct call: StandardScaler
- **Line 275:** DBSCAN - Direct call: DBSCAN
- **Line 162:** survival_func - Direct call: survival_func
- **Line 255:** pdf_func - Direct call: pdf_func

---

## src/analyst/feature_engineering_orchestrator.py

**Undefined function calls: 28**

- **Line 47:** handles_errors - Direct call: handles_errors
- **Line 105:** handles_errors - Direct call: handles_errors
- **Line 140:** handle_data_processing_errors - Direct call: handle_data_processing_errors
- **Line 155:** handles_errors - Direct call: handles_errors
- **Line 167:** handles_errors - Direct call: handles_errors
- **Line 182:** handle_data_processing_errors - Direct call: handle_data_processing_errors
- **Line 219:** handle_data_processing_errors - Direct call: handle_data_processing_errors
- **Line 242:** handle_data_processing_errors - Direct call: handle_data_processing_errors
- **Line 269:** handle_data_processing_errors - Direct call: handle_data_processing_errors
- **Line 285:** handle_data_processing_errors - Direct call: handle_data_processing_errors
- **Line 308:** handle_data_processing_errors - Direct call: handle_data_processing_errors
- **Line 323:** handles_errors - Direct call: handles_errors
- **Line 324:** handles_errors - Direct call: handles_errors
- **Line 333:** handles_errors - Direct call: handles_errors
- **Line 359:** handles_errors - Direct call: handles_errors
- **Line 366:** handles_errors - Direct call: handles_errors
- **Line 375:** handle_file_operations - Direct call: handle_file_operations
- **Line 384:** handle_data_processing_errors - Direct call: handle_data_processing_errors
- **Line 393:** handle_file_operations - Direct call: handle_file_operations
- **Line 33:** AdvancedFeatureEngineering - Direct call: AdvancedFeatureEngineering
- **Line 34:** AutoencoderFeatureGenerator - Direct call: AutoencoderFeatureGenerator
- **Line 35:** LimitedMicrostructureFeatures - Direct call: LimitedMicrostructureFeatures
- **Line 41:** get_parameter_value - Direct call: get_parameter_value
- **Line 42:** get_parameter_value - Direct call: get_parameter_value
- **Line 43:** get_parameter_value - Direct call: get_parameter_value
- **Line 44:** get_parameter_value - Direct call: get_parameter_value
- **Line 159:** AdvancedFeatureEngineering - Direct call: AdvancedFeatureEngineering
- **Line 172:** MetaLabelingSystem - Direct call: MetaLabelingSystem

---

## src/analyst/feature_engineering_utils.py

**Undefined function calls: 7**

- **Line 16:** handles_errors - Direct call: handles_errors
- **Line 27:** handles_errors - Direct call: handles_errors
- **Line 36:** handles_errors - Direct call: handles_errors
- **Line 50:** handles_errors - Direct call: handles_errors
- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 99:** handles_errors - Direct call: handles_errors
- **Line 110:** handles_errors - Direct call: handles_errors

---

## src/analyst/liquidation_risk_model.py

**Undefined function calls: 7**

- **Line 76:** handles_errors - Direct call: handles_errors
- **Line 110:** handles_errors - Direct call: handles_errors
- **Line 119:** handles_errors - Direct call: handles_errors
- **Line 147:** handles_errors - Direct call: handles_errors
- **Line 206:** validates - Direct call: validates
- **Line 207:** traced - Direct call: traced
- **Line 412:** handles_errors - Direct call: handles_errors

---

## src/analyst/location_classifier_optimization.py

**Undefined function calls: 2**

- **Line 17:** lru_cache - Direct call: lru_cache
- **Line 197:** ProcessPoolExecutor - Direct call: ProcessPoolExecutor

---

## src/analyst/market_health_analyzer.py

**Undefined function calls: 1**

- **Line 50:** handles_errors - Direct call: handles_errors

---

## src/analyst/meta_label_relevance.py

**Undefined function calls: 9**

- **Line 15:** handles_errors - Direct call: handles_errors
- **Line 51:** handles_errors - Direct call: handles_errors
- **Line 66:** handles_errors - Direct call: handles_errors
- **Line 114:** handles_errors - Direct call: handles_errors
- **Line 195:** handles_errors - Direct call: handles_errors
- **Line 41:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 47:** mutual_info_regression - Direct call: mutual_info_regression
- **Line 93:** LGBMClassifier - Direct call: LGBMClassifier
- **Line 97:** LGBMRegressor - Direct call: LGBMRegressor

---

## src/analyst/meta_labeling_system.py

**Undefined function calls: 15**

- **Line 57:** handles_errors - Direct call: handles_errors
- **Line 76:** handles_errors - Direct call: handles_errors
- **Line 729:** validate_data_quality - Direct call: validate_data_quality
- **Line 730:** traced - Direct call: traced
- **Line 765:** validate_data_quality - Direct call: validate_data_quality
- **Line 766:** traced - Direct call: traced
- **Line 795:** validate_data_quality - Direct call: validate_data_quality
- **Line 796:** traced - Direct call: traced
- **Line 890:** validate_data_quality - Direct call: validate_data_quality
- **Line 891:** traced - Direct call: traced
- **Line 955:** handles_errors - Direct call: handles_errors
- **Line 1052:** handles_errors - Direct call: handles_errors
- **Line 1146:** handles_errors - Direct call: handles_errors
- **Line 1224:** handles_errors - Direct call: handles_errors
- **Line 72:** initialization_error - Direct call: initialization_error

---

## src/analyst/ml_confidence_predictor.py

**Undefined function calls: 37**

- **Line 3257:** handles_errors - Direct call: handles_errors
- **Line 221:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 296:** handles_errors - Direct call: handles_errors
- **Line 346:** handles_errors - Direct call: handles_errors
- **Line 378:** handles_errors - Direct call: handles_errors
- **Line 431:** handles_errors - Direct call: handles_errors
- **Line 1013:** handles_errors - Direct call: handles_errors
- **Line 1076:** handles_errors - Direct call: handles_errors
- **Line 1140:** handles_errors - Direct call: handles_errors
- **Line 1266:** handles_errors - Direct call: handles_errors
- **Line 1521:** handles_errors - Direct call: handles_errors
- **Line 1541:** handles_errors - Direct call: handles_errors
- **Line 1561:** handle_file_operations - Direct call: handle_file_operations
- **Line 1579:** handles_errors - Direct call: handles_errors
- **Line 2911:** handles_errors - Direct call: handles_errors
- **Line 2996:** handles_errors - Direct call: handles_errors
- **Line 3131:** handles_errors - Direct call: handles_errors
- **Line 3154:** handles_errors - Direct call: handles_errors
- **Line 3235:** handles_errors - Direct call: handles_errors
- **Line 67:** get_parameter_value - Direct call: get_parameter_value
- **Line 138:** get_parameter_value - Direct call: get_parameter_value
- **Line 142:** get_parameter_value - Direct call: get_parameter_value
- **Line 1115:** AdvancedFeatureEngineering - Direct call: AdvancedFeatureEngineering
- **Line 1120:** MultiTimeframeFeatureEngineering - Direct call: MultiTimeframeFeatureEngineering
- **Line 1124:** FeatureEngineeringOrchestrator - Direct call: FeatureEngineeringOrchestrator
- **Line 1169:** CompositeHMMRegimeSystem - Direct call: CompositeHMMRegimeSystem
- **Line 2635:** ExecutionRequest - Direct call: ExecutionRequest
- **Line 1025:** create_training_manager - Direct call: create_training_manager
- **Line 2324:** setup_enhanced_order_manager - Direct call: setup_enhanced_order_manager
- **Line 2336:** AsyncOrderExecutor - Direct call: AsyncOrderExecutor
- **Line 3285:** failed - Direct call: failed
- **Line 1177:** initialization_error - Direct call: initialization_error
- **Line 2330:** failed - Direct call: failed
- **Line 2346:** initialization_error - Direct call: initialization_error
- **Line 1574:** failed - Direct call: failed
- **Line 1600:** missing - Direct call: missing
- **Line 2340:** failed - Direct call: failed

---

## src/analyst/multi_timeframe_feature_engineering.py

**Undefined function calls: 7**

- **Line 295:** handles_errors - Direct call: handles_errors
- **Line 30:** Path - Direct call: Path
- **Line 53:** FeatureEngineeringEngine - Direct call: FeatureEngineeringEngine
- **Line 71:** timedelta - Direct call: timedelta
- **Line 845:** timedelta - Direct call: timedelta
- **Line 838:** error - Direct call: error
- **Line 867:** error - Direct call: error

---

## src/analyst/order_book_analyzer.py

**Undefined function calls: 4**

- **Line 22:** validate_data_quality - Direct call: validate_data_quality
- **Line 23:** traced - Direct call: traced
- **Line 48:** validate_data_quality - Direct call: validate_data_quality
- **Line 49:** traced - Direct call: traced

---

## src/analyst/predictive_ensembles.py

**Undefined function calls: 30**

- **Line 1249:** handles_errors - Direct call: handles_errors
- **Line 70:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 116:** handles_errors - Direct call: handles_errors
- **Line 148:** handles_errors - Direct call: handles_errors
- **Line 196:** handles_errors - Direct call: handles_errors
- **Line 230:** handles_errors - Direct call: handles_errors
- **Line 252:** handles_errors - Direct call: handles_errors
- **Line 274:** handles_errors - Direct call: handles_errors
- **Line 298:** handles_errors - Direct call: handles_errors
- **Line 320:** handles_errors - Direct call: handles_errors
- **Line 344:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 409:** handles_errors - Direct call: handles_errors
- **Line 453:** handles_errors - Direct call: handles_errors
- **Line 503:** handles_errors - Direct call: handles_errors
- **Line 553:** handles_errors - Direct call: handles_errors
- **Line 605:** handles_errors - Direct call: handles_errors
- **Line 659:** handles_errors - Direct call: handles_errors
- **Line 1115:** handles_errors - Direct call: handles_errors
- **Line 1139:** handles_errors - Direct call: handles_errors
- **Line 1165:** handles_errors - Direct call: handles_errors
- **Line 1219:** handles_errors - Direct call: handles_errors
- **Line 95:** invalid - Direct call: invalid
- **Line 111:** failed - Direct call: failed
- **Line 163:** invalid - Direct call: invalid
- **Line 169:** invalid - Direct call: invalid
- **Line 295:** initialization_error - Direct call: initialization_error
- **Line 341:** initialization_error - Direct call: initialization_error
- **Line 436:** invalid - Direct call: invalid
- **Line 441:** invalid - Direct call: invalid
- **Line 430:** missing - Direct call: missing

---

## src/analyst/predictive_ensembles/ensemble_orchestrator.py

**Undefined function calls: 15**

- **Line 250:** LabelEncoder - Direct call: LabelEncoder
- **Line 252:** StratifiedKFold - Direct call: StratifiedKFold
- **Line 48:** VolatileRegimeEnsemble - Direct call: VolatileRegimeEnsemble
- **Line 260:** StandardScaler - Direct call: StandardScaler
- **Line 270:** LGBMClassifier - Direct call: LGBMClassifier
- **Line 324:** dump - Direct call: dump
- **Line 325:** dump - Direct call: dump
- **Line 326:** dump - Direct call: dump
- **Line 398:** dump - Direct call: dump
- **Line 408:** load - Direct call: load
- **Line 104:** VolatileRegimeEnsemble - Direct call: VolatileRegimeEnsemble
- **Line 265:** PCA - Direct call: PCA
- **Line 335:** load - Direct call: load
- **Line 336:** load - Direct call: load
- **Line 337:** load - Direct call: load

---

## src/analyst/predictive_ensembles/multi_timeframe_ensemble.py

**Undefined function calls: 8**

- **Line 296:** StratifiedKFold - Direct call: StratifiedKFold
- **Line 330:** MLPClassifier - Direct call: MLPClassifier
- **Line 353:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 528:** LabelEncoder - Direct call: LabelEncoder
- **Line 533:** StandardScaler - Direct call: StandardScaler
- **Line 210:** failed - Direct call: failed
- **Line 157:** failed - Direct call: failed
- **Line 206:** failed - Direct call: failed

---

## src/analyst/predictive_ensembles/regime_ensembles/base_ensemble.py

**Undefined function calls: 30**

- **Line 38:** handles_errors - Direct call: handles_errors
- **Line 61:** handles_errors - Direct call: handles_errors
- **Line 109:** handles_errors - Direct call: handles_errors
- **Line 134:** handles_errors - Direct call: handles_errors
- **Line 159:** handles_errors - Direct call: handles_errors
- **Line 215:** handles_errors - Direct call: handles_errors
- **Line 231:** handles_errors - Direct call: handles_errors
- **Line 236:** handles_errors - Direct call: handles_errors
- **Line 244:** handles_errors - Direct call: handles_errors
- **Line 286:** handles_errors - Direct call: handles_errors
- **Line 330:** handles_errors - Direct call: handles_errors
- **Line 340:** handles_errors - Direct call: handles_errors
- **Line 378:** handles_errors - Direct call: handles_errors
- **Line 415:** handles_errors - Direct call: handles_errors
- **Line 519:** handles_errors - Direct call: handles_errors
- **Line 580:** handles_errors - Direct call: handles_errors
- **Line 663:** handles_errors - Direct call: handles_errors
- **Line 684:** handles_errors - Direct call: handles_errors
- **Line 48:** StandardScaler - Direct call: StandardScaler
- **Line 50:** LabelEncoder - Direct call: LabelEncoder
- **Line 96:** StandardScaler - Direct call: StandardScaler
- **Line 99:** PCA - Direct call: PCA
- **Line 233:** LGBMClassifier - Direct call: LGBMClassifier
- **Line 182:** search_space_func - Direct call: search_space_func
- **Line 183:** model_class - Direct call: model_class
- **Line 220:** LogisticRegression - Direct call: LogisticRegression
- **Line 223:** LogisticRegression - Direct call: LogisticRegression
- **Line 215:** LogisticRegression - Direct call: LogisticRegression
- **Line 185:** PurgedKFoldTime - Direct call: PurgedKFoldTime
- **Line 188:** StratifiedKFold - Direct call: StratifiedKFold

---

## src/analyst/predictive_ensembles/regime_ensembles/volatile_regime_ensemble.py

**Undefined function calls: 30**

- **Line 35:** LGBMClassifier - Direct call: LGBMClassifier
- **Line 80:** Input - Direct call: Input
- **Line 89:** Model - Direct call: Model
- **Line 100:** Input - Direct call: Input
- **Line 113:** Model - Direct call: Model
- **Line 124:** TabNetClassifier - Direct call: TabNetClassifier
- **Line 135:** arch_model - Direct call: arch_model
- **Line 81:** LSTM - Direct call: LSTM
- **Line 82:** Dropout - Direct call: Dropout
- **Line 83:** LSTM - Direct call: LSTM
- **Line 84:** Dropout - Direct call: Dropout
- **Line 85:** Dense - Direct call: Dense
- **Line 86:** Dropout - Direct call: Dropout
- **Line 87:** Dense - Direct call: Dense
- **Line 88:** Dense - Direct call: Dense
- **Line 101:** MultiHeadAttention - Direct call: MultiHeadAttention
- **Line 102:** LayerNormalization - Direct call: LayerNormalization
- **Line 103:** Dropout - Direct call: Dropout
- **Line 104:** Dense - Direct call: Dense
- **Line 105:** Dropout - Direct call: Dropout
- **Line 106:** Dense - Direct call: Dense
- **Line 107:** LayerNormalization - Direct call: LayerNormalization
- **Line 108:** Flatten - Direct call: Flatten
- **Line 109:** Dense - Direct call: Dense
- **Line 110:** Dropout - Direct call: Dropout
- **Line 111:** Dense - Direct call: Dense
- **Line 112:** Dense - Direct call: Dense
- **Line 42:** failed - Direct call: failed
- **Line 128:** failed - Direct call: failed
- **Line 138:** failed - Direct call: failed

---

## src/analyst/regime_expert_orchestrator.py

**Undefined function calls: 7**

- **Line 51:** handles_errors - Direct call: handles_errors
- **Line 122:** handles_errors - Direct call: handles_errors
- **Line 142:** handles_errors - Direct call: handles_errors
- **Line 160:** handles_errors - Direct call: handles_errors
- **Line 204:** handles_errors - Direct call: handles_errors
- **Line 27:** RegimePredictiveEnsembles - Direct call: RegimePredictiveEnsembles
- **Line 38:** get_current_regime_info - Direct call: get_current_regime_info

---

## src/analyst/regime_runtime.py

**Undefined function calls: 1**

- **Line 128:** get_hmm_composite_manager - Direct call: get_hmm_composite_manager

---

## src/analyst/sr_relevance_optimizer.py

**Undefined function calls: 1**

- **Line 164:** differential_evolution - Direct call: differential_evolution

---

## src/analyst/unified_regime_classifier.py

**Undefined function calls: 9**

- **Line 275:** SRBreakoutPredictor - Direct call: SRBreakoutPredictor
- **Line 1694:** StandardScaler - Direct call: StandardScaler
- **Line 1758:** LabelEncoder - Direct call: LabelEncoder
- **Line 1769:** LGBMClassifier - Direct call: LGBMClassifier
- **Line 1827:** LabelEncoder - Direct call: LabelEncoder
- **Line 1850:** LGBMClassifier - Direct call: LGBMClassifier
- **Line 331:** original_ctor - Direct call: original_ctor
- **Line 334:** original_ctor - Direct call: original_ctor
- **Line 346:** bitgen_cls - Direct call: bitgen_cls

---

## src/analyst/unified_regime_classifier_fractal_enhanced.py

**Undefined function calls: 3**

- **Line 106:** handles_errors - Direct call: handles_errors
- **Line 100:** StandardScaler - Direct call: StandardScaler
- **Line 121:** SRBreakoutPredictor - Direct call: SRBreakoutPredictor

---

## src/analyst/unified_regime_classifier_fractal_simplified.py

**Undefined function calls: 3**

- **Line 107:** handles_errors - Direct call: handles_errors
- **Line 101:** StandardScaler - Direct call: StandardScaler
- **Line 122:** SRBreakoutPredictor - Direct call: SRBreakoutPredictor

---

## src/analyst/unified_regime_classifier_sr_focused.py

**Undefined function calls: 1**

- **Line 82:** StandardScaler - Direct call: StandardScaler

---

## src/analyst/unified_regime_classifier_sr_optimized.py

**Undefined function calls: 1**

- **Line 35:** SRRelevanceOptimizer - Direct call: SRRelevanceOptimizer

---

## src/analytics/bayesian_probability_updates.py

**Undefined function calls: 6**

- **Line 85:** handles_errors - Direct call: handles_errors
- **Line 115:** handles_errors - Direct call: handles_errors
- **Line 255:** handles_errors - Direct call: handles_errors
- **Line 312:** handles_errors - Direct call: handles_errors
- **Line 356:** handles_errors - Direct call: handles_errors
- **Line 75:** deque - Direct call: deque

---

## src/analytics/copula_dependency_models.py

**Undefined function calls: 5**

- **Line 76:** handles_errors - Direct call: handles_errors
- **Line 106:** handles_errors - Direct call: handles_errors
- **Line 163:** handles_errors - Direct call: handles_errors
- **Line 415:** handles_errors - Direct call: handles_errors
- **Line 486:** handles_errors - Direct call: handles_errors

---

## src/analytics/limited_microstructure_features.py

**Undefined function calls: 4**

- **Line 47:** handles_errors - Direct call: handles_errors
- **Line 75:** handles_errors - Direct call: handles_errors
- **Line 421:** handles_errors - Direct call: handles_errors
- **Line 39:** deque - Direct call: deque

---

## src/analytics/performance_attribution.py

**Undefined function calls: 9**

- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 115:** handles_errors - Direct call: handles_errors
- **Line 206:** handles_errors - Direct call: handles_errors
- **Line 312:** handles_errors - Direct call: handles_errors
- **Line 418:** handles_errors - Direct call: handles_errors
- **Line 57:** deque - Direct call: deque
- **Line 83:** deque - Direct call: deque
- **Line 91:** deque - Direct call: deque
- **Line 102:** deque - Direct call: deque

---

## src/ares_pipeline.py

**Undefined function calls: 33**

- **Line 68:** handles_errors - Direct call: handles_errors
- **Line 91:** handles_errors - Direct call: handles_errors
- **Line 110:** handles_errors - Direct call: handles_errors
- **Line 196:** handles_errors - Direct call: handles_errors
- **Line 263:** handles_errors - Direct call: handles_errors
- **Line 283:** handles_errors - Direct call: handles_errors
- **Line 298:** handles_errors - Direct call: handles_errors
- **Line 376:** handles_errors - Direct call: handles_errors
- **Line 557:** handles_errors - Direct call: handles_errors
- **Line 637:** setup_logging - Direct call: setup_logging
- **Line 638:** init_observability - Direct call: init_observability
- **Line 32:** Path - Direct call: Path
- **Line 52:** DependencyContainer - Direct call: DependencyContainer
- **Line 53:** ServiceLocator - Direct call: ServiceLocator
- **Line 669:** main - Direct call: main
- **Line 625:** get_dual_model_config - Direct call: get_dual_model_config
- **Line 635:** Path - Direct call: Path
- **Line 99:** ConfigurationService - Direct call: ConfigurationService
- **Line 592:** setup_dual_model_system - Direct call: setup_dual_model_system
- **Line 609:** setup_performance_monitor - Direct call: setup_performance_monitor
- **Line 653:** failed - Direct call: failed
- **Line 659:** failed - Direct call: failed
- **Line 665:** error - Direct call: error
- **Line 192:** warning - Direct call: warning
- **Line 259:** warning - Direct call: warning
- **Line 310:** warning - Direct call: warning
- **Line 368:** critical - Direct call: critical
- **Line 488:** warning - Direct call: warning
- **Line 612:** setup_performance_dashboard - Direct call: setup_performance_dashboard
- **Line 128:** get_exchange_name - Direct call: get_exchange_name
- **Line 351:** error - Direct call: error
- **Line 355:** warning - Direct call: warning
- **Line 346:** warning - Direct call: warning

---

## src/components/modular_analyst.py

**Undefined function calls: 62**

- **Line 54:** handles_errors - Direct call: handles_errors
- **Line 93:** handles_errors - Direct call: handles_errors
- **Line 120:** handles_errors - Direct call: handles_errors
- **Line 153:** handles_errors - Direct call: handles_errors
- **Line 180:** handles_errors - Direct call: handles_errors
- **Line 201:** handles_errors - Direct call: handles_errors
- **Line 222:** handles_errors - Direct call: handles_errors
- **Line 241:** handles_errors - Direct call: handles_errors
- **Line 262:** handles_errors - Direct call: handles_errors
- **Line 325:** handles_errors - Direct call: handles_errors
- **Line 364:** handles_errors - Direct call: handles_errors
- **Line 414:** handles_errors - Direct call: handles_errors
- **Line 464:** handles_errors - Direct call: handles_errors
- **Line 510:** handles_errors - Direct call: handles_errors
- **Line 752:** handles_errors - Direct call: handles_errors
- **Line 771:** handles_errors - Direct call: handles_errors
- **Line 794:** handles_errors - Direct call: handles_errors
- **Line 841:** handles_errors - Direct call: handles_errors
- **Line 130:** invalid - Direct call: invalid
- **Line 135:** invalid - Direct call: invalid
- **Line 147:** error - Direct call: error
- **Line 78:** invalid - Direct call: invalid
- **Line 89:** failed - Direct call: failed
- **Line 118:** error - Direct call: error
- **Line 177:** initialization_error - Direct call: initialization_error
- **Line 198:** initialization_error - Direct call: initialization_error
- **Line 219:** initialization_error - Direct call: initialization_error
- **Line 238:** initialization_error - Direct call: initialization_error
- **Line 259:** initialization_error - Direct call: initialization_error
- **Line 321:** error - Direct call: error
- **Line 351:** invalid - Direct call: invalid
- **Line 355:** invalid - Direct call: invalid
- **Line 361:** error - Direct call: error
- **Line 411:** error - Direct call: error
- **Line 461:** error - Direct call: error
- **Line 507:** error - Direct call: error
- **Line 547:** error - Direct call: error
- **Line 559:** error - Direct call: error
- **Line 568:** error - Direct call: error
- **Line 577:** error - Direct call: error
- **Line 590:** error - Direct call: error
- **Line 605:** error - Direct call: error
- **Line 617:** error - Direct call: error
- **Line 628:** error - Direct call: error
- **Line 637:** error - Direct call: error
- **Line 646:** error - Direct call: error
- **Line 655:** error - Direct call: error
- **Line 664:** error - Direct call: error
- **Line 673:** error - Direct call: error
- **Line 684:** error - Direct call: error
- **Line 693:** error - Direct call: error
- **Line 702:** error - Direct call: error
- **Line 711:** error - Direct call: error
- **Line 722:** error - Direct call: error
- **Line 731:** error - Direct call: error
- **Line 740:** error - Direct call: error
- **Line 749:** error - Direct call: error
- **Line 769:** error - Direct call: error
- **Line 791:** error - Direct call: error
- **Line 814:** error - Direct call: error
- **Line 859:** error - Direct call: error
- **Line 345:** missing - Direct call: missing

---

## src/components/modular_strategist.py

**Undefined function calls: 59**

- **Line 57:** handles_errors - Direct call: handles_errors
- **Line 91:** handles_errors - Direct call: handles_errors
- **Line 110:** handles_errors - Direct call: handles_errors
- **Line 143:** handles_errors - Direct call: handles_errors
- **Line 170:** handles_errors - Direct call: handles_errors
- **Line 189:** handles_errors - Direct call: handles_errors
- **Line 208:** handles_errors - Direct call: handles_errors
- **Line 227:** handles_errors - Direct call: handles_errors
- **Line 246:** handles_errors - Direct call: handles_errors
- **Line 321:** handles_errors - Direct call: handles_errors
- **Line 371:** handles_errors - Direct call: handles_errors
- **Line 425:** handles_errors - Direct call: handles_errors
- **Line 479:** handles_errors - Direct call: handles_errors
- **Line 535:** handles_errors - Direct call: handles_errors
- **Line 886:** handles_errors - Direct call: handles_errors
- **Line 905:** handles_errors - Direct call: handles_errors
- **Line 928:** handles_errors - Direct call: handles_errors
- **Line 975:** handles_errors - Direct call: handles_errors
- **Line 844:** timedelta - Direct call: timedelta
- **Line 80:** invalid - Direct call: invalid
- **Line 120:** invalid - Direct call: invalid
- **Line 125:** invalid - Direct call: invalid
- **Line 137:** error - Direct call: error
- **Line 843:** timedelta - Direct call: timedelta
- **Line 167:** initialization_error - Direct call: initialization_error
- **Line 186:** initialization_error - Direct call: initialization_error
- **Line 205:** initialization_error - Direct call: initialization_error
- **Line 224:** initialization_error - Direct call: initialization_error
- **Line 243:** initialization_error - Direct call: initialization_error
- **Line 317:** error - Direct call: error
- **Line 358:** invalid - Direct call: invalid
- **Line 362:** invalid - Direct call: invalid
- **Line 368:** error - Direct call: error
- **Line 422:** error - Direct call: error
- **Line 476:** error - Direct call: error
- **Line 531:** error - Direct call: error
- **Line 584:** error - Direct call: error
- **Line 604:** error - Direct call: error
- **Line 620:** error - Direct call: error
- **Line 636:** error - Direct call: error
- **Line 650:** error - Direct call: error
- **Line 668:** error - Direct call: error
- **Line 684:** error - Direct call: error
- **Line 700:** error - Direct call: error
- **Line 717:** error - Direct call: error
- **Line 737:** error - Direct call: error
- **Line 760:** error - Direct call: error
- **Line 783:** error - Direct call: error
- **Line 806:** error - Direct call: error
- **Line 830:** error - Direct call: error
- **Line 848:** error - Direct call: error
- **Line 864:** error - Direct call: error
- **Line 882:** error - Direct call: error
- **Line 903:** error - Direct call: error
- **Line 925:** error - Direct call: error
- **Line 948:** error - Direct call: error
- **Line 993:** error - Direct call: error
- **Line 343:** missing - Direct call: missing
- **Line 352:** missing - Direct call: missing

---

## src/components/modular_supervisor.py

**Undefined function calls: 66**

- **Line 60:** handles_errors - Direct call: handles_errors
- **Line 103:** handles_errors - Direct call: handles_errors
- **Line 132:** handles_errors - Direct call: handles_errors
- **Line 172:** handles_errors - Direct call: handles_errors
- **Line 199:** handles_errors - Direct call: handles_errors
- **Line 220:** handles_errors - Direct call: handles_errors
- **Line 241:** handles_errors - Direct call: handles_errors
- **Line 262:** handles_errors - Direct call: handles_errors
- **Line 281:** handles_errors - Direct call: handles_errors
- **Line 356:** handles_errors - Direct call: handles_errors
- **Line 406:** handles_errors - Direct call: handles_errors
- **Line 468:** handles_errors - Direct call: handles_errors
- **Line 524:** handles_errors - Direct call: handles_errors
- **Line 584:** handles_errors - Direct call: handles_errors
- **Line 974:** handles_errors - Direct call: handles_errors
- **Line 993:** handles_errors - Direct call: handles_errors
- **Line 1016:** handles_errors - Direct call: handles_errors
- **Line 1063:** handles_errors - Direct call: handles_errors
- **Line 85:** invalid - Direct call: invalid
- **Line 99:** failed - Direct call: failed
- **Line 130:** error - Direct call: error
- **Line 143:** invalid - Direct call: invalid
- **Line 148:** invalid - Direct call: invalid
- **Line 161:** error - Direct call: error
- **Line 169:** error - Direct call: error
- **Line 196:** initialization_error - Direct call: initialization_error
- **Line 217:** initialization_error - Direct call: initialization_error
- **Line 238:** initialization_error - Direct call: initialization_error
- **Line 259:** initialization_error - Direct call: initialization_error
- **Line 278:** initialization_error - Direct call: initialization_error
- **Line 352:** error - Direct call: error
- **Line 393:** invalid - Direct call: invalid
- **Line 397:** invalid - Direct call: invalid
- **Line 403:** error - Direct call: error
- **Line 464:** error - Direct call: error
- **Line 521:** error - Direct call: error
- **Line 581:** error - Direct call: error
- **Line 633:** error - Direct call: error
- **Line 652:** error - Direct call: error
- **Line 665:** error - Direct call: error
- **Line 678:** error - Direct call: error
- **Line 691:** error - Direct call: error
- **Line 704:** error - Direct call: error
- **Line 717:** error - Direct call: error
- **Line 732:** error - Direct call: error
- **Line 745:** error - Direct call: error
- **Line 758:** error - Direct call: error
- **Line 771:** error - Direct call: error
- **Line 784:** error - Direct call: error
- **Line 797:** error - Direct call: error
- **Line 816:** error - Direct call: error
- **Line 833:** error - Direct call: error
- **Line 850:** error - Direct call: error
- **Line 867:** error - Direct call: error
- **Line 884:** error - Direct call: error
- **Line 901:** error - Direct call: error
- **Line 920:** error - Direct call: error
- **Line 937:** error - Direct call: error
- **Line 954:** error - Direct call: error
- **Line 971:** error - Direct call: error
- **Line 991:** error - Direct call: error
- **Line 1013:** error - Direct call: error
- **Line 1036:** error - Direct call: error
- **Line 1081:** error - Direct call: error
- **Line 378:** missing - Direct call: missing
- **Line 387:** missing - Direct call: missing

---

## src/components/modular_tactician.py

**Undefined function calls: 59**

- **Line 53:** handles_errors - Direct call: handles_errors
- **Line 87:** handles_errors - Direct call: handles_errors
- **Line 114:** handles_errors - Direct call: handles_errors
- **Line 152:** handles_errors - Direct call: handles_errors
- **Line 179:** handles_errors - Direct call: handles_errors
- **Line 198:** handles_errors - Direct call: handles_errors
- **Line 217:** handles_errors - Direct call: handles_errors
- **Line 236:** handles_errors - Direct call: handles_errors
- **Line 255:** handles_errors - Direct call: handles_errors
- **Line 330:** handles_errors - Direct call: handles_errors
- **Line 380:** handles_errors - Direct call: handles_errors
- **Line 434:** handles_errors - Direct call: handles_errors
- **Line 488:** handles_errors - Direct call: handles_errors
- **Line 542:** handles_errors - Direct call: handles_errors
- **Line 877:** handles_errors - Direct call: handles_errors
- **Line 896:** handles_errors - Direct call: handles_errors
- **Line 919:** handles_errors - Direct call: handles_errors
- **Line 966:** handles_errors - Direct call: handles_errors
- **Line 76:** invalid - Direct call: invalid
- **Line 112:** error - Direct call: error
- **Line 125:** invalid - Direct call: invalid
- **Line 130:** invalid - Direct call: invalid
- **Line 142:** error - Direct call: error
- **Line 149:** error - Direct call: error
- **Line 176:** initialization_error - Direct call: initialization_error
- **Line 195:** initialization_error - Direct call: initialization_error
- **Line 214:** initialization_error - Direct call: initialization_error
- **Line 233:** initialization_error - Direct call: initialization_error
- **Line 252:** initialization_error - Direct call: initialization_error
- **Line 326:** error - Direct call: error
- **Line 367:** invalid - Direct call: invalid
- **Line 371:** invalid - Direct call: invalid
- **Line 377:** error - Direct call: error
- **Line 431:** error - Direct call: error
- **Line 485:** error - Direct call: error
- **Line 539:** error - Direct call: error
- **Line 593:** error - Direct call: error
- **Line 613:** error - Direct call: error
- **Line 630:** error - Direct call: error
- **Line 647:** error - Direct call: error
- **Line 664:** error - Direct call: error
- **Line 683:** error - Direct call: error
- **Line 700:** error - Direct call: error
- **Line 717:** error - Direct call: error
- **Line 734:** error - Direct call: error
- **Line 753:** error - Direct call: error
- **Line 770:** error - Direct call: error
- **Line 787:** error - Direct call: error
- **Line 804:** error - Direct call: error
- **Line 823:** error - Direct call: error
- **Line 840:** error - Direct call: error
- **Line 857:** error - Direct call: error
- **Line 874:** error - Direct call: error
- **Line 894:** error - Direct call: error
- **Line 916:** error - Direct call: error
- **Line 939:** error - Direct call: error
- **Line 984:** error - Direct call: error
- **Line 352:** missing - Direct call: missing
- **Line 361:** missing - Direct call: missing

---

## src/config.py

**Undefined function calls: 28**

- **Line 27:** get_env_settings - Direct call: get_env_settings
- **Line 99:** handles_errors - Direct call: handles_errors
- **Line 128:** handles_errors - Direct call: handles_errors
- **Line 140:** handles_errors - Direct call: handles_errors
- **Line 160:** handles_errors - Direct call: handles_errors
- **Line 170:** handles_errors - Direct call: handles_errors
- **Line 179:** handles_errors - Direct call: handles_errors
- **Line 198:** handles_errors - Direct call: handles_errors
- **Line 210:** handles_errors - Direct call: handles_errors
- **Line 219:** handles_errors - Direct call: handles_errors
- **Line 232:** handles_errors - Direct call: handles_errors
- **Line 240:** handles_errors - Direct call: handles_errors
- **Line 164:** get_environment_config - Direct call: get_environment_config
- **Line 164:** get_system_config_section - Direct call: get_system_config_section
- **Line 164:** get_trading_config_section - Direct call: get_trading_config_section
- **Line 164:** get_training_config_section - Direct call: get_training_config_section
- **Line 111:** invalid - Direct call: invalid
- **Line 150:** invalid - Direct call: invalid
- **Line 154:** failed - Direct call: failed
- **Line 157:** failed - Direct call: failed
- **Line 176:** failed - Direct call: failed
- **Line 195:** failed - Direct call: failed
- **Line 208:** failed - Direct call: failed
- **Line 217:** failed - Direct call: failed
- **Line 230:** failed - Direct call: failed
- **Line 238:** failed - Direct call: failed
- **Line 248:** failed - Direct call: failed
- **Line 225:** warning - Direct call: warning

---

## src/config/__init__.py

**Undefined function calls: 5**

- **Line 28:** get_environment_settings - Direct call: get_environment_settings
- **Line 29:** get_system_config - Direct call: get_system_config
- **Line 30:** get_trading_config - Direct call: get_trading_config
- **Line 31:** get_training_config - Direct call: get_training_config
- **Line 64:** validate_complete_config - Direct call: validate_complete_config

---

## src/config/config_manager.py

**Undefined function calls: 25**

- **Line 58:** get_confidence_config - Direct call: get_confidence_config
- **Line 59:** get_intensity_config - Direct call: get_intensity_config
- **Line 60:** get_position_sizing_config - Direct call: get_position_sizing_config
- **Line 61:** get_leverage_config - Direct call: get_leverage_config
- **Line 62:** get_tpsl_config - Direct call: get_tpsl_config
- **Line 63:** get_ensemble_config - Direct call: get_ensemble_config
- **Line 64:** get_sr_config - Direct call: get_sr_config
- **Line 65:** get_two_tier_config - Direct call: get_two_tier_config
- **Line 66:** get_technical_indicators_config - Direct call: get_technical_indicators_config
- **Line 67:** get_system_monitoring_config - Direct call: get_system_monitoring_config
- **Line 68:** get_training_optimization_config - Direct call: get_training_optimization_config
- **Line 69:** get_regime_transition_config - Direct call: get_regime_transition_config
- **Line 74:** get_confidence_search_space - Direct call: get_confidence_search_space
- **Line 75:** get_intensity_search_space - Direct call: get_intensity_search_space
- **Line 76:** get_position_sizing_search_space - Direct call: get_position_sizing_search_space
- **Line 77:** get_leverage_search_space - Direct call: get_leverage_search_space
- **Line 78:** get_tpsl_search_space - Direct call: get_tpsl_search_space
- **Line 79:** get_ensemble_search_space - Direct call: get_ensemble_search_space
- **Line 80:** get_sr_search_space - Direct call: get_sr_search_space
- **Line 81:** get_two_tier_search_space - Direct call: get_two_tier_search_space
- **Line 82:** get_technical_indicators_search_space - Direct call: get_technical_indicators_search_space
- **Line 83:** get_system_monitoring_search_space - Direct call: get_system_monitoring_search_space
- **Line 84:** get_training_optimization_search_space - Direct call: get_training_optimization_search_space
- **Line 85:** get_regime_transition_search_space - Direct call: get_regime_transition_search_space
- **Line 115:** asdict - Direct call: asdict

---

## src/config/enhanced_feature_selection_config.py

**Undefined function calls: 34**

- **Line 22:** Field - Direct call: Field
- **Line 25:** Field - Direct call: Field
- **Line 28:** Field - Direct call: Field
- **Line 33:** Field - Direct call: Field
- **Line 36:** Field - Direct call: Field
- **Line 39:** field - Direct call: field
- **Line 47:** Field - Direct call: Field
- **Line 52:** Field - Direct call: Field
- **Line 56:** Field - Direct call: Field
- **Line 59:** Field - Direct call: Field
- **Line 64:** field - Direct call: field
- **Line 72:** field - Direct call: field
- **Line 82:** Field - Direct call: Field
- **Line 85:** field - Direct call: field
- **Line 102:** Field - Direct call: Field
- **Line 105:** Field - Direct call: Field
- **Line 108:** field - Direct call: field
- **Line 115:** Field - Direct call: Field
- **Line 121:** Field - Direct call: Field
- **Line 124:** Field - Direct call: Field
- **Line 128:** Field - Direct call: Field
- **Line 134:** Field - Direct call: Field
- **Line 137:** Field - Direct call: Field
- **Line 143:** Field - Direct call: Field
- **Line 146:** Field - Direct call: Field
- **Line 149:** Field - Direct call: Field
- **Line 154:** Field - Direct call: Field
- **Line 157:** Field - Direct call: Field
- **Line 162:** Field - Direct call: Field
- **Line 165:** Field - Direct call: Field
- **Line 166:** Field - Direct call: Field
- **Line 171:** Field - Direct call: Field
- **Line 174:** Field - Direct call: Field
- **Line 177:** Field - Direct call: Field

---

## src/config/enhanced_matrix_config.py

**Undefined function calls: 1**

- **Line 21:** get_m1_gpu_config - Direct call: get_m1_gpu_config

---

## src/config/fractional_implementations_config.py

**Undefined function calls: 6**

- **Line 33:** field - Direct call: field
- **Line 78:** field - Direct call: field
- **Line 81:** field - Direct call: field
- **Line 82:** field - Direct call: field
- **Line 124:** field - Direct call: field
- **Line 127:** field - Direct call: field

---

## src/config/label_model_mapping.py

**Undefined function calls: 9**

- **Line 42:** CatBoostClassifier - Direct call: CatBoostClassifier
- **Line 42:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 42:** SGDClassifier - Direct call: SGDClassifier
- **Line 42:** SGDClassifier - Direct call: SGDClassifier
- **Line 42:** LogisticRegression - Direct call: LogisticRegression
- **Line 75:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 77:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 49:** GaussianHMM - Direct call: GaussianHMM
- **Line 50:** LogisticRegression - Direct call: LogisticRegression

---

## src/config/regime_specific_optimization_config.py

**Undefined function calls: 19**

- **Line 22:** field - Direct call: field
- **Line 23:** field - Direct call: field
- **Line 24:** field - Direct call: field
- **Line 25:** field - Direct call: field
- **Line 26:** field - Direct call: field
- **Line 27:** field - Direct call: field
- **Line 28:** field - Direct call: field
- **Line 29:** field - Direct call: field
- **Line 30:** field - Direct call: field
- **Line 31:** field - Direct call: field
- **Line 32:** field - Direct call: field
- **Line 33:** field - Direct call: field
- **Line 34:** field - Direct call: field
- **Line 35:** field - Direct call: field
- **Line 45:** field - Direct call: field
- **Line 46:** field - Direct call: field
- **Line 47:** field - Direct call: field
- **Line 55:** field - Direct call: field
- **Line 56:** field - Direct call: field

---

## src/config/sr_comprehensive_config_loader.py

**Undefined function calls: 5**

- **Line 46:** Path - Direct call: Path
- **Line 53:** get_sr_config - Direct call: get_sr_config
- **Line 164:** asdict - Direct call: asdict
- **Line 195:** asdict - Direct call: asdict
- **Line 113:** get_sr_config - Direct call: get_sr_config

---

## src/config/sr_config_loader.py

**Undefined function calls: 4**

- **Line 98:** handles_errors - Direct call: handles_errors
- **Line 403:** get_sr_config_loader - Direct call: get_sr_config_loader
- **Line 106:** Path - Direct call: Path
- **Line 318:** Path - Direct call: Path

---

## src/config/sr_optimization_config.py

**Undefined function calls: 8**

- **Line 26:** field - Direct call: field
- **Line 37:** field - Direct call: field
- **Line 47:** field - Direct call: field
- **Line 58:** field - Direct call: field
- **Line 66:** field - Direct call: field
- **Line 76:** field - Direct call: field
- **Line 92:** field - Direct call: field
- **Line 103:** field - Direct call: field

---

## src/config/system.py

**Undefined function calls: 1**

- **Line 17:** get_environment_settings - Direct call: get_environment_settings

---

## src/config/typed_config.py

**Undefined function calls: 5**

- **Line 17:** TypeValidator - Direct call: TypeValidator
- **Line 36:** Path - Direct call: Path
- **Line 178:** Path - Direct call: Path
- **Line 179:** validate_config - Direct call: validate_config
- **Line 84:** RuntimeTypeError - Direct call: RuntimeTypeError

---

## src/core/config_service.py

**Undefined function calls: 12**

- **Line 164:** handles_errors - Direct call: handles_errors
- **Line 189:** handles_errors - Direct call: handles_errors
- **Line 209:** handles_errors - Direct call: handles_errors
- **Line 213:** Path - Direct call: Path
- **Line 380:** asdict - Direct call: asdict
- **Line 328:** Observer - Direct call: Observer
- **Line 175:** failed - Direct call: failed
- **Line 207:** error - Direct call: error
- **Line 325:** warning - Direct call: warning
- **Line 344:** warning - Direct call: warning
- **Line 378:** error - Direct call: error
- **Line 299:** error - Direct call: error

---

## src/core/decorators.py

**Undefined function calls: 8**

- **Line 19:** func - Direct call: func
- **Line 27:** func - Direct call: func
- **Line 35:** func - Direct call: func
- **Line 43:** func - Direct call: func
- **Line 51:** func - Direct call: func
- **Line 59:** func - Direct call: func
- **Line 67:** func - Direct call: func
- **Line 8:** func - Direct call: func

---

## src/core/decorators/auth.py

**Undefined function calls: 34**

- **Line 16:** ContextVar - Direct call: ContextVar
- **Line 133:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 173:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 213:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 286:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 303:** defaultdict - Direct call: defaultdict
- **Line 333:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 127:** func - Direct call: func
- **Line 161:** func - Direct call: func
- **Line 201:** func - Direct call: func
- **Line 259:** func - Direct call: func
- **Line 313:** time - Direct call: time
- **Line 321:** func - Direct call: func
- **Line 324:** time - Direct call: time
- **Line 132:** func - Direct call: func
- **Line 156:** AuthenticationError - Direct call: AuthenticationError
- **Line 160:** AuthorizationError - Direct call: AuthorizationError
- **Line 166:** AuthenticationError - Direct call: AuthenticationError
- **Line 170:** AuthorizationError - Direct call: AuthorizationError
- **Line 171:** func - Direct call: func
- **Line 196:** AuthenticationError - Direct call: AuthenticationError
- **Line 200:** AuthorizationError - Direct call: AuthorizationError
- **Line 206:** AuthenticationError - Direct call: AuthenticationError
- **Line 210:** AuthorizationError - Direct call: AuthorizationError
- **Line 211:** func - Direct call: func
- **Line 238:** AuthenticationError - Direct call: AuthenticationError
- **Line 258:** AuthorizationError - Direct call: AuthorizationError
- **Line 264:** AuthenticationError - Direct call: AuthenticationError
- **Line 284:** AuthorizationError - Direct call: AuthorizationError
- **Line 285:** func - Direct call: func
- **Line 308:** key_func - Direct call: key_func
- **Line 319:** RateLimitError - Direct call: RateLimitError
- **Line 330:** RateLimitError - Direct call: RateLimitError
- **Line 332:** func - Direct call: func

---

## src/core/decorators/cache.py

**Undefined function calls: 9**

- **Line 14:** ContextVar - Direct call: ContextVar
- **Line 268:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 154:** get_correlation_id - Direct call: get_correlation_id
- **Line 201:** key_func - Direct call: key_func
- **Line 214:** func - Direct call: func
- **Line 236:** key_func - Direct call: key_func
- **Line 249:** func - Direct call: func
- **Line 218:** condition - Direct call: condition
- **Line 253:** condition - Direct call: condition

---

## src/core/decorators/compose.py

**Undefined function calls: 9**

- **Line 13:** ParamSpec - Direct call: ParamSpec
- **Line 14:** TypeVar - Direct call: TypeVar
- **Line 131:** cast - Direct call: cast
- **Line 157:** cast - Direct call: cast
- **Line 71:** cast - Direct call: cast
- **Line 64:** cast - Direct call: cast
- **Line 68:** sync_handler - Direct call: sync_handler
- **Line 61:** async_handler - Direct call: async_handler
- **Line 155:** func - Direct call: func

---

## src/core/decorators/enhanced_error_handling.py

**Undefined function calls: 19**

- **Line 31:** ContextVar - Direct call: ContextVar
- **Line 32:** ContextVar - Direct call: ContextVar
- **Line 74:** field - Direct call: field
- **Line 75:** field - Direct call: field
- **Line 76:** field - Direct call: field
- **Line 77:** field - Direct call: field
- **Line 80:** field - Direct call: field
- **Line 81:** field - Direct call: field
- **Line 90:** field - Direct call: field
- **Line 91:** field - Direct call: field
- **Line 96:** field - Direct call: field
- **Line 97:** field - Direct call: field
- **Line 98:** field - Direct call: field
- **Line 527:** func - Direct call: func
- **Line 534:** func - Direct call: func
- **Line 553:** recovery_func - Direct call: recovery_func
- **Line 344:** func - Direct call: func
- **Line 471:** Path - Direct call: Path
- **Line 342:** func - Direct call: func

---

## src/core/decorators/errors.py

**Undefined function calls: 3**

- **Line 8:** func - Direct call: func
- **Line 20:** func - Direct call: func
- **Line 32:** func - Direct call: func

---

## src/core/decorators/function_monitor.py

**Undefined function calls: 15**

- **Line 36:** ContextVar - Direct call: ContextVar
- **Line 37:** ContextVar - Direct call: ContextVar
- **Line 53:** field - Direct call: field
- **Line 57:** field - Direct call: field
- **Line 58:** field - Direct call: field
- **Line 59:** field - Direct call: field
- **Line 71:** field - Direct call: field
- **Line 72:** field - Direct call: field
- **Line 73:** field - Direct call: field
- **Line 74:** field - Direct call: field
- **Line 75:** field - Direct call: field
- **Line 77:** field - Direct call: field
- **Line 398:** func - Direct call: func
- **Line 475:** func - Direct call: func
- **Line 316:** Path - Direct call: Path

---

## src/core/decorators/logging.py

**Undefined function calls: 14**

- **Line 18:** ContextVar - Direct call: ContextVar
- **Line 235:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 301:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 457:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 183:** log_method - Direct call: log_method
- **Line 214:** log_method - Direct call: log_method
- **Line 186:** func - Direct call: func
- **Line 193:** log_method - Direct call: log_method
- **Line 224:** log_method - Direct call: log_method
- **Line 265:** func - Direct call: func
- **Line 368:** func - Direct call: func
- **Line 217:** func - Direct call: func
- **Line 286:** func - Direct call: func
- **Line 424:** func - Direct call: func

---

## src/core/decorators/retry_timeout.py

**Undefined function calls: 24**

- **Line 177:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 204:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 238:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 256:** compose - Direct call: compose
- **Line 289:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 163:** ServiceUnavailableError - Direct call: ServiceUnavailableError
- **Line 176:** ServiceUnavailableError - Direct call: ServiceUnavailableError
- **Line 196:** func - Direct call: func
- **Line 49:** func - Direct call: func
- **Line 274:** func - Direct call: func
- **Line 47:** ServiceUnavailableError - Direct call: ServiceUnavailableError
- **Line 63:** ServiceUnavailableError - Direct call: ServiceUnavailableError
- **Line 65:** func - Direct call: func
- **Line 117:** callback - Direct call: callback
- **Line 156:** func - Direct call: func
- **Line 203:** AppTimeoutError - Direct call: AppTimeoutError
- **Line 276:** callable - Direct call: callable
- **Line 282:** func - Direct call: func
- **Line 284:** callable - Direct call: callable
- **Line 169:** func - Direct call: func
- **Line 200:** func - Direct call: func
- **Line 277:** fallback_value - Direct call: fallback_value
- **Line 287:** fallback_value - Direct call: fallback_value
- **Line 286:** fallback_value - Direct call: fallback_value

---

## src/core/decorators/trace.py

**Undefined function calls: 11**

- **Line 22:** ContextVar - Direct call: ContextVar
- **Line 54:** field - Direct call: field
- **Line 57:** field - Direct call: field
- **Line 58:** field - Direct call: field
- **Line 92:** field - Direct call: field
- **Line 93:** field - Direct call: field
- **Line 275:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 112:** get_correlation_id - Direct call: get_correlation_id
- **Line 199:** func - Direct call: func
- **Line 252:** func - Direct call: func
- **Line 357:** callable - Direct call: callable

---

## src/core/decorators/validate.py

**Undefined function calls: 46**

- **Line 159:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 267:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 448:** uniform_wrapper - Direct call: uniform_wrapper
- **Line 58:** get_type_hints - Direct call: get_type_hints
- **Line 107:** func - Direct call: func
- **Line 114:** get_type_hints - Direct call: get_type_hints
- **Line 223:** func - Direct call: func
- **Line 373:** func - Direct call: func
- **Line 101:** ValidationError - Direct call: ValidationError
- **Line 152:** ValidationError - Direct call: ValidationError
- **Line 157:** func - Direct call: func
- **Line 216:** func - Direct call: func
- **Line 220:** func - Direct call: func
- **Line 265:** func - Direct call: func
- **Line 313:** ValidationError - Direct call: ValidationError
- **Line 319:** ValidationError - Direct call: ValidationError
- **Line 359:** ValidationError - Direct call: ValidationError
- **Line 367:** ValidationError - Direct call: ValidationError
- **Line 385:** ValidationError - Direct call: ValidationError
- **Line 392:** ValidationError - Direct call: ValidationError
- **Line 432:** ValidationError - Direct call: ValidationError
- **Line 440:** ValidationError - Direct call: ValidationError
- **Line 446:** func - Direct call: func
- **Line 484:** ValidationError - Direct call: ValidationError
- **Line 495:** target_type - Direct call: target_type
- **Line 516:** ValidationError - Direct call: ValidationError
- **Line 66:** ValidationError - Direct call: ValidationError
- **Line 121:** ValidationError - Direct call: ValidationError
- **Line 200:** ValidationError - Direct call: ValidationError
- **Line 244:** ValidationError - Direct call: ValidationError
- **Line 258:** func - Direct call: func
- **Line 262:** func - Direct call: func
- **Line 330:** ValidationError - Direct call: ValidationError
- **Line 403:** ValidationError - Direct call: ValidationError
- **Line 466:** expected_type - Direct call: expected_type
- **Line 475:** ValidationError - Direct call: ValidationError
- **Line 512:** schema - Direct call: schema
- **Line 522:** ValidationError - Direct call: ValidationError
- **Line 532:** ValidationError - Direct call: ValidationError
- **Line 564:** ValidationError - Direct call: ValidationError
- **Line 576:** ValidationError - Direct call: ValidationError
- **Line 469:** ValidationError - Direct call: ValidationError
- **Line 344:** ValidationError - Direct call: ValidationError
- **Line 350:** ValidationError - Direct call: ValidationError
- **Line 417:** ValidationError - Direct call: ValidationError
- **Line 423:** ValidationError - Direct call: ValidationError

---

## src/core/dependency_injection.py

**Undefined function calls: 4**

- **Line 17:** TypeVar - Direct call: TypeVar
- **Line 218:** factory_func - Direct call: factory_func
- **Line 222:** factory_func - Direct call: factory_func
- **Line 225:** factory_func - Direct call: factory_func

---

## src/core/di_integration.py

**Undefined function calls: 3**

- **Line 39:** DependencyContainer - Direct call: DependencyContainer
- **Line 40:** ServiceRegistry - Direct call: ServiceRegistry
- **Line 43:** TradingSystemFactory - Direct call: TradingSystemFactory

---

## src/core/di_launcher.py

**Undefined function calls: 9**

- **Line 25:** DependencyContainer - Direct call: DependencyContainer
- **Line 26:** ServiceRegistry - Direct call: ServiceRegistry
- **Line 27:** TradingSystemFactory - Direct call: TradingSystemFactory
- **Line 46:** BinanceClient - Direct call: BinanceClient
- **Line 47:** StateManager - Direct call: StateManager
- **Line 48:** PerformanceReporter - Direct call: PerformanceReporter
- **Line 73:** BinanceClient - Direct call: BinanceClient
- **Line 74:** StateManager - Direct call: StateManager
- **Line 75:** PerformanceReporter - Direct call: PerformanceReporter

---

## src/core/domain.py

**Undefined function calls: 2**

- **Line 7:** func - Direct call: func
- **Line 15:** func - Direct call: func

---

## src/core/domain/__init__.py

**Undefined function calls: 14**

- **Line 10:** traced - Direct call: traced
- **Line 14:** handles_errors - Direct call: handles_errors
- **Line 45:** compose - Direct call: compose
- **Line 53:** compose - Direct call: compose
- **Line 61:** compose - Direct call: compose
- **Line 177:** compose - Direct call: compose
- **Line 30:** _mem - Direct call: _mem
- **Line 49:** handles_errors - Direct call: handles_errors
- **Line 52:** traced - Direct call: traced
- **Line 68:** wraps - Direct call: wraps
- **Line 42:** validate_feature_engineering_pipeline - Direct call: validate_feature_engineering_pipeline
- **Line 51:** cached - Direct call: cached
- **Line 59:** timeout - Direct call: timeout
- **Line 80:** func - Direct call: func

---

## src/core/domain/decorators.py

**Undefined function calls: 35**

- **Line 8:** TypeVar - Direct call: TypeVar
- **Line 292:** compose - Direct call: compose
- **Line 32:** wraps - Direct call: wraps
- **Line 109:** wraps - Direct call: wraps
- **Line 156:** wraps - Direct call: wraps
- **Line 176:** wraps - Direct call: wraps
- **Line 194:** wraps - Direct call: wraps
- **Line 233:** wraps - Direct call: wraps
- **Line 251:** wraps - Direct call: wraps
- **Line 52:** func - Direct call: func
- **Line 111:** func - Direct call: func
- **Line 141:** compose - Direct call: compose
- **Line 158:** func - Direct call: func
- **Line 180:** func - Direct call: func
- **Line 198:** func - Direct call: func
- **Line 219:** compose - Direct call: compose
- **Line 237:** func - Direct call: func
- **Line 254:** validator - Direct call: validator
- **Line 255:** validated_func - Direct call: validated_func
- **Line 276:** compose - Direct call: compose
- **Line 283:** handles_errors - Direct call: handles_errors
- **Line 285:** timeout - Direct call: timeout
- **Line 291:** cached - Direct call: cached
- **Line 142:** log_execution_time - Direct call: log_execution_time
- **Line 143:** log_call - Direct call: log_call
- **Line 144:** traced - Direct call: traced
- **Line 220:** validates - Direct call: validates
- **Line 221:** handles_errors - Direct call: handles_errors
- **Line 276:** traced - Direct call: traced
- **Line 47:** ValidationError - Direct call: ValidationError
- **Line 241:** ValidationError - Direct call: ValidationError
- **Line 164:** BusinessRuleError - Direct call: BusinessRuleError
- **Line 260:** DataIntegrityError - Direct call: DataIntegrityError
- **Line 264:** DataIntegrityError - Direct call: DataIntegrityError
- **Line 267:** DataIntegrityError - Direct call: DataIntegrityError

---

## src/core/domain/decorators_extended.py

**Undefined function calls: 34**

- **Line 10:** TypeVar - Direct call: TypeVar
- **Line 253:** cached - Direct call: cached
- **Line 23:** wraps - Direct call: wraps
- **Line 38:** wraps - Direct call: wraps
- **Line 67:** wraps - Direct call: wraps
- **Line 86:** wraps - Direct call: wraps
- **Line 112:** wraps - Direct call: wraps
- **Line 142:** wraps - Direct call: wraps
- **Line 185:** wraps - Direct call: wraps
- **Line 206:** wraps - Direct call: wraps
- **Line 226:** wraps - Direct call: wraps
- **Line 240:** wraps - Direct call: wraps
- **Line 40:** func - Direct call: func
- **Line 77:** func - Direct call: func
- **Line 88:** func - Direct call: func
- **Line 115:** func - Direct call: func
- **Line 146:** func - Direct call: func
- **Line 177:** compose - Direct call: compose
- **Line 187:** func - Direct call: func
- **Line 210:** func - Direct call: func
- **Line 218:** compose - Direct call: compose
- **Line 231:** func - Direct call: func
- **Line 25:** base_validator - Direct call: base_validator
- **Line 177:** traced - Direct call: traced
- **Line 177:** cached - Direct call: cached
- **Line 177:** validates - Direct call: validates
- **Line 218:** traced - Direct call: traced
- **Line 218:** handles_errors - Direct call: handles_errors
- **Line 245:** func - Direct call: func
- **Line 119:** ValidationError - Direct call: ValidationError
- **Line 72:** ValidationError - Direct call: ValidationError
- **Line 76:** ValidationError - Direct call: ValidationError
- **Line 124:** validator - Direct call: validator
- **Line 126:** ValidationError - Direct call: ValidationError

---

## src/core/enhanced_dependency_injection.py

**Undefined function calls: 2**

- **Line 13:** TypeVar - Direct call: TypeVar
- **Line 26:** _DependencyContainer - Direct call: _DependencyContainer

---

## src/core/enhanced_factories.py

**Undefined function calls: 6**

- **Line 114:** ExchangeFactory - Direct call: ExchangeFactory
- **Line 140:** FirestoreManager - Direct call: FirestoreManager
- **Line 151:** InfluxDBManager - Direct call: InfluxDBManager
- **Line 172:** StateManager - Direct call: StateManager
- **Line 196:** PerformanceReporter - Direct call: PerformanceReporter
- **Line 84:** failed - Direct call: failed

---

## src/core/errors/base.py

**Undefined function calls: 1**

- **Line 36:** field - Direct call: field

---

## src/core/errors/handlers/http.py

**Undefined function calls: 8**

- **Line 44:** JSONResponse - Direct call: JSONResponse
- **Line 24:** jsonify - Direct call: jsonify
- **Line 70:** JsonResponse - Direct call: JsonResponse
- **Line 112:** app - Direct call: app
- **Line 66:** JsonResponse - Direct call: JsonResponse
- **Line 88:** handler - Direct call: handler
- **Line 118:** start_response - Direct call: start_response
- **Line 127:** start_response - Direct call: start_response

---

## src/core/errors/mapping.py

**Undefined function calls: 14**

- **Line 41:** AppError - Direct call: AppError
- **Line 78:** issubclass - Direct call: issubclass
- **Line 16:** ValidationError - Direct call: ValidationError
- **Line 16:** NotFoundError - Direct call: NotFoundError
- **Line 16:** ValidationError - Direct call: ValidationError
- **Line 16:** ValidationError - Direct call: ValidationError
- **Line 16:** ServiceUnavailableError - Direct call: ServiceUnavailableError
- **Line 16:** AppTimeoutError - Direct call: AppTimeoutError
- **Line 16:** AppError - Direct call: AppError
- **Line 79:** mapper - Direct call: mapper
- **Line 37:** mapper - Direct call: mapper
- **Line 18:** ValidationError - Direct call: ValidationError
- **Line 18:** ValidationError - Direct call: ValidationError
- **Line 22:** ValidationError - Direct call: ValidationError

---

## src/core/examples/decorator_usage.py

**Undefined function calls: 46**

- **Line 36:** validates - Direct call: validates
- **Line 37:** handles_errors - Direct call: handles_errors
- **Line 47:** compose - Direct call: compose
- **Line 65:** retry - Direct call: retry
- **Line 66:** circuit_breaker - Direct call: circuit_breaker
- **Line 67:** timeout - Direct call: timeout
- **Line 68:** traced - Direct call: traced
- **Line 85:** authenticated - Direct call: authenticated
- **Line 86:** requires_role - Direct call: requires_role
- **Line 87:** log_call - Direct call: log_call
- **Line 88:** handles_errors - Direct call: handles_errors
- **Line 116:** validate_schema - Direct call: validate_schema
- **Line 117:** cached - Direct call: cached
- **Line 118:** traced - Direct call: traced
- **Line 158:** register_exception_mapping - Direct call: register_exception_mapping
- **Line 164:** handles_errors - Direct call: handles_errors
- **Line 174:** traced - Direct call: traced
- **Line 175:** cached - Direct call: cached
- **Line 176:** retry - Direct call: retry
- **Line 199:** trace_method - Direct call: trace_method
- **Line 216:** authenticated - Direct call: authenticated
- **Line 217:** validates - Direct call: validates
- **Line 218:** cached - Direct call: cached
- **Line 219:** traced - Direct call: traced
- **Line 220:** handles_errors - Direct call: handles_errors
- **Line 48:** log_call - Direct call: log_call
- **Line 49:** validates - Direct call: validates
- **Line 50:** handles_errors - Direct call: handles_errors
- **Line 51:** cached - Direct call: cached
- **Line 132:** validate_dataframe - Direct call: validate_dataframe
- **Line 137:** handles_errors - Direct call: handles_errors
- **Line 138:** cached - Direct call: cached
- **Line 186:** span_event - Direct call: span_event
- **Line 190:** span_attribute - Direct call: span_attribute
- **Line 191:** span_event - Direct call: span_event
- **Line 203:** cached - Direct call: cached
- **Line 208:** requires_permission - Direct call: requires_permission
- **Line 209:** validates - Direct call: validates
- **Line 298:** cache_stats - Direct call: cache_stats
- **Line 303:** get_current_trace - Direct call: get_current_trace
- **Line 79:** ConnectionError - Direct call: ConnectionError
- **Line 160:** ValidationError - Direct call: ValidationError
- **Line 288:** get_user_from_db - Direct call: get_user_from_db
- **Line 307:** get_trace_summary - Direct call: get_trace_summary
- **Line 313:** main - Direct call: main
- **Line 259:** fetch_external_data - Direct call: fetch_external_data

---

## src/core/generic_base.py

**Undefined function calls: 5**

- **Line 22:** TypeVar - Direct call: TypeVar
- **Line 23:** TypeVar - Direct call: TypeVar
- **Line 24:** TypeVar - Direct call: TypeVar
- **Line 25:** TypeVar - Direct call: TypeVar
- **Line 26:** TypeVar - Direct call: TypeVar

---

## src/core/reporting/step03_execution_reporter.py

**Undefined function calls: 4**

- **Line 60:** field - Direct call: field
- **Line 61:** field - Direct call: field
- **Line 136:** field - Direct call: field
- **Line 157:** Path - Direct call: Path

---

## src/core/sr_error_handlers.py

**Undefined function calls: 4**

- **Line 167:** wraps - Direct call: wraps
- **Line 191:** wraps - Direct call: wraps
- **Line 198:** func - Direct call: func
- **Line 174:** func - Direct call: func

---

## src/custom_types/base_types.py

**Undefined function calls: 16**

- **Line 14:** NewType - Direct call: NewType
- **Line 15:** NewType - Direct call: NewType
- **Line 16:** NewType - Direct call: NewType
- **Line 17:** NewType - Direct call: NewType
- **Line 18:** NewType - Direct call: NewType
- **Line 19:** NewType - Direct call: NewType
- **Line 20:** NewType - Direct call: NewType
- **Line 23:** NewType - Direct call: NewType
- **Line 24:** NewType - Direct call: NewType
- **Line 25:** NewType - Direct call: NewType
- **Line 26:** NewType - Direct call: NewType
- **Line 27:** NewType - Direct call: NewType
- **Line 28:** NewType - Direct call: NewType
- **Line 31:** NewType - Direct call: NewType
- **Line 32:** NewType - Direct call: NewType
- **Line 33:** NewType - Direct call: NewType

---

## src/custom_types/protocol_types.py

**Undefined function calls: 4**

- **Line 17:** TypeVar - Direct call: TypeVar
- **Line 18:** TypeVar - Direct call: TypeVar
- **Line 19:** TypeVar - Direct call: TypeVar
- **Line 20:** TypeVar - Direct call: TypeVar

---

## src/custom_types/validation.py

**Undefined function calls: 14**

- **Line 11:** TypeVar - Direct call: TypeVar
- **Line 101:** wraps - Direct call: wraps
- **Line 145:** Symbol - Direct call: Symbol
- **Line 151:** Price - Direct call: Price
- **Line 157:** Volume - Direct call: Volume
- **Line 48:** get_origin - Direct call: get_origin
- **Line 49:** get_args - Direct call: get_args
- **Line 113:** func - Direct call: func
- **Line 130:** wraps - Direct call: wraps
- **Line 132:** func - Direct call: func
- **Line 134:** validator_func - Direct call: validator_func
- **Line 118:** validation_error - Direct call: validation_error
- **Line 136:** validation_error - Direct call: validation_error
- **Line 112:** validation_error - Direct call: validation_error

---

## src/database/efficient_features_database.py

**Undefined function calls: 16**

- **Line 53:** handles_errors - Direct call: handles_errors
- **Line 117:** handles_errors - Direct call: handles_errors
- **Line 180:** handles_errors - Direct call: handles_errors
- **Line 230:** handles_errors - Direct call: handles_errors
- **Line 307:** handles_errors - Direct call: handles_errors
- **Line 349:** handles_errors - Direct call: handles_errors
- **Line 445:** handles_errors - Direct call: handles_errors
- **Line 528:** handles_errors - Direct call: handles_errors
- **Line 669:** handles_errors - Direct call: handles_errors
- **Line 177:** error - Direct call: error
- **Line 218:** warning - Direct call: warning
- **Line 304:** error - Direct call: error
- **Line 319:** missing - Direct call: missing
- **Line 346:** error - Direct call: error
- **Line 466:** warning - Direct call: warning
- **Line 525:** error - Direct call: error

---

## src/database/firestore_manager.py

**Undefined function calls: 18**

- **Line 35:** handles_errors - Direct call: handles_errors
- **Line 60:** handles_errors - Direct call: handles_errors
- **Line 73:** handles_errors - Direct call: handles_errors
- **Line 83:** handles_errors - Direct call: handles_errors
- **Line 97:** handles_errors - Direct call: handles_errors
- **Line 107:** handles_errors - Direct call: handles_errors
- **Line 130:** handles_errors - Direct call: handles_errors
- **Line 154:** handles_errors - Direct call: handles_errors
- **Line 175:** handles_errors - Direct call: handles_errors
- **Line 200:** handles_errors - Direct call: handles_errors
- **Line 104:** partial - Direct call: partial
- **Line 41:** error_context - Direct call: error_context
- **Line 42:** get_environment_settings - Direct call: get_environment_settings
- **Line 51:** error_context - Direct call: error_context
- **Line 65:** get_environment_settings - Direct call: get_environment_settings
- **Line 87:** error - Direct call: error
- **Line 101:** warning - Direct call: warning
- **Line 151:** missing - Direct call: missing

---

## src/database/migration_utils.py

**Undefined function calls: 12**

- **Line 178:** SQLiteManager - Direct call: SQLiteManager
- **Line 187:** SQLiteManager - Direct call: SQLiteManager
- **Line 196:** SQLiteManager - Direct call: SQLiteManager
- **Line 240:** main - Direct call: main
- **Line 35:** SQLiteManager - Direct call: SQLiteManager
- **Line 161:** timedelta - Direct call: timedelta
- **Line 214:** export_database_for_trading - Direct call: export_database_for_trading
- **Line 45:** failed - Direct call: failed
- **Line 69:** missing - Direct call: missing
- **Line 83:** failed - Direct call: failed
- **Line 225:** import_database_for_trading - Direct call: import_database_for_trading
- **Line 235:** validate_migration_file - Direct call: validate_migration_file

---

## src/database/precomputed_features_manager.py

**Undefined function calls: 6**

- **Line 47:** handles_errors - Direct call: handles_errors
- **Line 98:** handles_errors - Direct call: handles_errors
- **Line 127:** handles_errors - Direct call: handles_errors
- **Line 39:** InfluxDBManager - Direct call: InfluxDBManager
- **Line 112:** warning - Direct call: warning
- **Line 144:** warning - Direct call: warning

---

## src/database/sqlite_manager.py

**Undefined function calls: 74**

- **Line 24:** handles_errors - Direct call: handles_errors
- **Line 34:** handles_errors - Direct call: handles_errors
- **Line 43:** handles_errors - Direct call: handles_errors
- **Line 63:** handles_errors - Direct call: handles_errors
- **Line 110:** handles_errors - Direct call: handles_errors
- **Line 140:** handles_errors - Direct call: handles_errors
- **Line 167:** handles_errors - Direct call: handles_errors
- **Line 200:** handles_errors - Direct call: handles_errors
- **Line 212:** handles_errors - Direct call: handles_errors
- **Line 249:** handles_errors - Direct call: handles_errors
- **Line 264:** handles_errors - Direct call: handles_errors
- **Line 299:** handles_errors - Direct call: handles_errors
- **Line 335:** handles_errors - Direct call: handles_errors
- **Line 373:** handles_errors - Direct call: handles_errors
- **Line 399:** handles_errors - Direct call: handles_errors
- **Line 433:** handles_errors - Direct call: handles_errors
- **Line 468:** handles_errors - Direct call: handles_errors
- **Line 497:** handles_errors - Direct call: handles_errors
- **Line 528:** handles_errors - Direct call: handles_errors
- **Line 561:** handles_errors - Direct call: handles_errors
- **Line 587:** handles_errors - Direct call: handles_errors
- **Line 604:** handles_errors - Direct call: handles_errors
- **Line 651:** handles_errors - Direct call: handles_errors
- **Line 106:** defaultdict - Direct call: defaultdict
- **Line 107:** defaultdict - Direct call: defaultdict
- **Line 122:** invalid - Direct call: invalid
- **Line 131:** failed - Direct call: failed
- **Line 134:** failed - Direct call: failed
- **Line 137:** failed - Direct call: failed
- **Line 163:** error - Direct call: error
- **Line 165:** error - Direct call: error
- **Line 177:** invalid - Direct call: invalid
- **Line 180:** invalid - Direct call: invalid
- **Line 183:** invalid - Direct call: invalid
- **Line 186:** invalid - Direct call: invalid
- **Line 189:** invalid - Direct call: invalid
- **Line 194:** error - Direct call: error
- **Line 197:** error - Direct call: error
- **Line 208:** connection_error - Direct call: connection_error
- **Line 210:** connection_error - Direct call: connection_error
- **Line 226:** failed - Direct call: failed
- **Line 240:** initialization_error - Direct call: initialization_error
- **Line 243:** initialization_error - Direct call: initialization_error
- **Line 246:** initialization_error - Direct call: initialization_error
- **Line 260:** error - Direct call: error
- **Line 262:** error - Direct call: error
- **Line 283:** failed - Direct call: failed
- **Line 295:** error - Direct call: error
- **Line 318:** failed - Direct call: failed
- **Line 331:** error - Direct call: error
- **Line 350:** failed - Direct call: failed
- **Line 369:** error - Direct call: error
- **Line 384:** failed - Direct call: failed
- **Line 395:** error - Direct call: error
- **Line 413:** failed - Direct call: failed
- **Line 429:** error - Direct call: error
- **Line 452:** failed - Direct call: failed
- **Line 464:** error - Direct call: error
- **Line 482:** failed - Direct call: failed
- **Line 493:** error - Direct call: error
- **Line 512:** failed - Direct call: failed
- **Line 524:** error - Direct call: error
- **Line 544:** failed - Direct call: failed
- **Line 557:** error - Direct call: error
- **Line 575:** error - Direct call: error
- **Line 602:** connection_error - Direct call: connection_error
- **Line 621:** failed - Direct call: failed
- **Line 632:** error - Direct call: error
- **Line 648:** error - Direct call: error
- **Line 661:** error - Direct call: error
- **Line 279:** missing - Direct call: missing
- **Line 314:** missing - Direct call: missing
- **Line 448:** missing - Direct call: missing
- **Line 584:** error - Direct call: error

---

## src/exchange/binance.py

**Undefined function calls: 1**

- **Line 255:** urlencode - Direct call: urlencode

---

## src/explainability/analyst_explainer.py

**Undefined function calls: 5**

- **Line 111:** ExplanationResult - Direct call: ExplanationResult
- **Line 182:** ExplanationResult - Direct call: ExplanationResult
- **Line 255:** ExplanationResult - Direct call: ExplanationResult
- **Line 329:** ExplanationResult - Direct call: ExplanationResult
- **Line 468:** ExplanationResult - Direct call: ExplanationResult

---

## src/explainability/base_explainer.py

**Undefined function calls: 4**

- **Line 116:** Path - Direct call: Path
- **Line 308:** Path - Direct call: Path
- **Line 177:** LimeTabularExplainer - Direct call: LimeTabularExplainer
- **Line 207:** explainer - Direct call: explainer

---

## src/explainability/explainability_orchestrator.py

**Undefined function calls: 9**

- **Line 38:** TacticianExplainer - Direct call: TacticianExplainer
- **Line 39:** HMMExplainer - Direct call: HMMExplainer
- **Line 40:** SRExplainer - Direct call: SRExplainer
- **Line 41:** AnalystExplainer - Direct call: AnalystExplainer
- **Line 44:** TradeDecisionTracer - Direct call: TradeDecisionTracer
- **Line 440:** Path - Direct call: Path
- **Line 454:** TradeDecisionTrace - Direct call: TradeDecisionTrace
- **Line 513:** Path - Direct call: Path
- **Line 514:** Path - Direct call: Path

---

## src/explainability/hmm_explainer.py

**Undefined function calls: 4**

- **Line 105:** ExplanationResult - Direct call: ExplanationResult
- **Line 183:** ExplanationResult - Direct call: ExplanationResult
- **Line 254:** ExplanationResult - Direct call: ExplanationResult
- **Line 388:** ExplanationResult - Direct call: ExplanationResult

---

## src/explainability/integration_decorators.py

**Undefined function calls: 15**

- **Line 30:** ExplainabilityOrchestrator - Direct call: ExplainabilityOrchestrator
- **Line 211:** feature_extractor - Direct call: feature_extractor
- **Line 273:** explainable_func - Direct call: explainable_func
- **Line 292:** explainable_func - Direct call: explainable_func
- **Line 311:** explainable_func - Direct call: explainable_func
- **Line 330:** explainable_func - Direct call: explainable_func
- **Line 353:** explainable_func - Direct call: explainable_func
- **Line 385:** preprocessing - Direct call: preprocessing
- **Line 439:** extractor_func - Direct call: extractor_func
- **Line 63:** func - Direct call: func
- **Line 128:** func - Direct call: func
- **Line 193:** func - Direct call: func
- **Line 88:** func - Direct call: func
- **Line 162:** func - Direct call: func
- **Line 197:** func - Direct call: func

---

## src/explainability/sr_explainer.py

**Undefined function calls: 5**

- **Line 106:** ExplanationResult - Direct call: ExplanationResult
- **Line 179:** ExplanationResult - Direct call: ExplanationResult
- **Line 252:** ExplanationResult - Direct call: ExplanationResult
- **Line 325:** ExplanationResult - Direct call: ExplanationResult
- **Line 465:** ExplanationResult - Direct call: ExplanationResult

---

## src/explainability/tactician_explainer.py

**Undefined function calls: 4**

- **Line 99:** ExplanationResult - Direct call: ExplanationResult
- **Line 167:** ExplanationResult - Direct call: ExplanationResult
- **Line 235:** ExplanationResult - Direct call: ExplanationResult
- **Line 368:** ExplanationResult - Direct call: ExplanationResult

---

## src/explainability/visualization_tools.py

**Undefined function calls: 2**

- **Line 43:** Path - Direct call: Path
- **Line 586:** Path - Direct call: Path

---

## src/integration/paper_trading_integration.py

**Undefined function calls: 34**

- **Line 450:** handles_errors - Direct call: handles_errors
- **Line 68:** performance_monitor - Direct call: performance_monitor
- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 126:** handles_errors - Direct call: handles_errors
- **Line 153:** performance_monitor - Direct call: performance_monitor
- **Line 155:** comprehensive_validation - Direct call: comprehensive_validation
- **Line 156:** handles_errors - Direct call: handles_errors
- **Line 260:** performance_monitor - Direct call: performance_monitor
- **Line 261:** handles_errors - Direct call: handles_errors
- **Line 344:** performance_monitor - Direct call: performance_monitor
- **Line 367:** performance_monitor - Direct call: performance_monitor
- **Line 368:** handles_errors - Direct call: handles_errors
- **Line 427:** performance_monitor - Direct call: performance_monitor
- **Line 428:** handles_errors - Direct call: handles_errors
- **Line 85:** setup_paper_trader - Direct call: setup_paper_trader
- **Line 480:** error - Direct call: error
- **Line 87:** failed - Direct call: failed
- **Line 113:** failed - Direct call: failed
- **Line 135:** initialization_error - Direct call: initialization_error
- **Line 141:** warning - Direct call: warning
- **Line 150:** error - Direct call: error
- **Line 186:** initialization_error - Direct call: initialization_error
- **Line 241:** get_comprehensive_logger - Direct call: get_comprehensive_logger
- **Line 257:** error - Direct call: error
- **Line 273:** error - Direct call: error
- **Line 308:** error - Direct call: error
- **Line 319:** error - Direct call: error
- **Line 341:** error - Direct call: error
- **Line 364:** error - Direct call: error
- **Line 413:** error - Direct call: error
- **Line 448:** error - Direct call: error
- **Line 97:** _setup_reporter - Direct call: _setup_reporter
- **Line 231:** invalid - Direct call: invalid
- **Line 105:** warning - Direct call: warning

---

## src/launcher/ares_launcher_refactored.py

**Undefined function calls: 23**

- **Line 595:** setup_signal_handlers - Direct call: setup_signal_handlers
- **Line 54:** setup_comprehensive_logging - Direct call: setup_comprehensive_logging
- **Line 55:** ensure_comprehensive_logging_available - Direct call: ensure_comprehensive_logging_available
- **Line 76:** ConfigurationManager - Direct call: ConfigurationManager
- **Line 79:** StepOrchestratorWrapper - Direct call: StepOrchestratorWrapper
- **Line 82:** ValidationFactory - Direct call: ValidationFactory
- **Line 98:** setup_signal_handlers - Direct call: setup_signal_handlers
- **Line 377:** Path - Direct call: Path
- **Line 608:** parse_arguments - Direct call: parse_arguments
- **Line 609:** validate_arguments - Direct call: validate_arguments
- **Line 45:** Path - Direct call: Path
- **Line 59:** init_observability - Direct call: init_observability
- **Line 173:** SQLiteManager - Direct call: SQLiteManager
- **Line 177:** get_training_config_dict - Direct call: get_training_config_dict
- **Line 183:** EnhancedTrainingManager - Direct call: EnhancedTrainingManager
- **Line 190:** get_training_input_dict - Direct call: get_training_input_dict
- **Line 340:** WaveletFeaturePrecomputer - Direct call: WaveletFeaturePrecomputer
- **Line 466:** UnifiedRegimeClassifier - Direct call: UnifiedRegimeClassifier
- **Line 475:** load_klines_data - Direct call: load_klines_data
- **Line 91:** format_datetime - Direct call: format_datetime
- **Line 194:** format_datetime - Direct call: format_datetime
- **Line 91:** get_current_datetime - Direct call: get_current_datetime
- **Line 194:** get_current_datetime - Direct call: get_current_datetime

---

## src/launcher/configuration_manager.py

**Undefined function calls: 6**

- **Line 116:** get_intensity_comparison - Direct call: get_intensity_comparison
- **Line 130:** list_available_modes - Direct call: list_available_modes
- **Line 131:** get_mode_recommendations - Direct call: get_mode_recommendations
- **Line 83:** get_training_mode_config - Direct call: get_training_mode_config
- **Line 92:** get_intensity_percentage - Direct call: get_intensity_percentage
- **Line 137:** get_intensity_percentage - Direct call: get_intensity_percentage

---

## src/launcher/enhanced_trading_launcher.py

**Undefined function calls: 46**

- **Line 560:** handles_errors - Direct call: handles_errors
- **Line 82:** handles_errors - Direct call: handles_errors
- **Line 90:** performance_monitor - Direct call: performance_monitor
- **Line 121:** handles_errors - Direct call: handles_errors
- **Line 140:** handles_errors - Direct call: handles_errors
- **Line 145:** performance_monitor - Direct call: performance_monitor
- **Line 179:** handles_errors - Direct call: handles_errors
- **Line 187:** performance_monitor - Direct call: performance_monitor
- **Line 229:** handles_errors - Direct call: handles_errors
- **Line 237:** performance_monitor - Direct call: performance_monitor
- **Line 277:** handles_errors - Direct call: handles_errors
- **Line 285:** performance_monitor - Direct call: performance_monitor
- **Line 338:** handles_errors - Direct call: handles_errors
- **Line 346:** performance_monitor - Direct call: performance_monitor
- **Line 445:** performance_monitor - Direct call: performance_monitor
- **Line 472:** handles_errors - Direct call: handles_errors
- **Line 477:** performance_monitor - Direct call: performance_monitor
- **Line 536:** handles_errors - Direct call: handles_errors
- **Line 541:** performance_monitor - Direct call: performance_monitor
- **Line 269:** warning - Direct call: warning
- **Line 104:** invalid - Direct call: invalid
- **Line 131:** error - Direct call: error
- **Line 137:** error - Direct call: error
- **Line 151:** setup_paper_trading_integration - Direct call: setup_paper_trading_integration
- **Line 177:** initialization_error - Direct call: initialization_error
- **Line 203:** initialization_error - Direct call: initialization_error
- **Line 207:** error - Direct call: error
- **Line 226:** error - Direct call: error
- **Line 253:** initialization_error - Direct call: initialization_error
- **Line 257:** error - Direct call: error
- **Line 274:** error - Direct call: error
- **Line 305:** initialization_error - Direct call: initialization_error
- **Line 309:** error - Direct call: error
- **Line 335:** error - Direct call: error
- **Line 372:** initialization_error - Direct call: initialization_error
- **Line 386:** execution_error - Direct call: execution_error
- **Line 394:** error - Direct call: error
- **Line 410:** error - Direct call: error
- **Line 424:** error - Direct call: error
- **Line 442:** error - Direct call: error
- **Line 469:** error - Direct call: error
- **Line 520:** error - Direct call: error
- **Line 558:** error - Direct call: error
- **Line 167:** _setup_backtester - Direct call: _setup_backtester
- **Line 174:** failed - Direct call: failed
- **Line 169:** failed - Direct call: failed

---

## src/launcher/gui_manager.py

**Undefined function calls: 1**

- **Line 91:** Path - Direct call: Path

---

## src/launcher/pipeline_managers.py

**Undefined function calls: 22**

- **Line 319:** get_backtesting_logger - Direct call: get_backtesting_logger
- **Line 431:** initialize_report_manager - Direct call: initialize_report_manager
- **Line 432:** initialize_report_collector - Direct call: initialize_report_collector
- **Line 117:** safe_file_exists - Direct call: safe_file_exists
- **Line 76:** format_datetime - Direct call: format_datetime
- **Line 109:** safe_file_exists - Direct call: safe_file_exists
- **Line 111:** ensure_directory - Direct call: ensure_directory
- **Line 141:** format_datetime - Direct call: format_datetime
- **Line 179:** safe_file_exists - Direct call: safe_file_exists
- **Line 181:** ensure_directory - Direct call: ensure_directory
- **Line 193:** safe_file_exists - Direct call: safe_file_exists
- **Line 227:** format_datetime - Direct call: format_datetime
- **Line 265:** safe_file_exists - Direct call: safe_file_exists
- **Line 267:** ensure_directory - Direct call: ensure_directory
- **Line 279:** safe_file_exists - Direct call: safe_file_exists
- **Line 76:** get_current_datetime - Direct call: get_current_datetime
- **Line 141:** get_current_datetime - Direct call: get_current_datetime
- **Line 227:** get_current_datetime - Direct call: get_current_datetime
- **Line 359:** safe_file_exists - Direct call: safe_file_exists
- **Line 382:** format_datetime - Direct call: format_datetime
- **Line 89:** Path - Direct call: Path
- **Line 382:** get_current_datetime - Direct call: get_current_datetime

---

## src/launcher/step_orchestrator_wrapper.py

**Undefined function calls: 3**

- **Line 26:** ValidationFactory - Direct call: ValidationFactory
- **Line 64:** StepOrchestrator - Direct call: StepOrchestrator
- **Line 212:** Path - Direct call: Path

---

## src/launcher/validation_utilities.py

**Undefined function calls: 16**

- **Line 267:** ValidatorOrchestrator - Direct call: ValidatorOrchestrator
- **Line 429:** ValidatorOrchestrator - Direct call: ValidatorOrchestrator
- **Line 64:** safe_file_exists - Direct call: safe_file_exists
- **Line 56:** safe_file_exists - Direct call: safe_file_exists
- **Line 58:** ensure_directory - Direct call: ensure_directory
- **Line 90:** safe_file_exists - Direct call: safe_file_exists
- **Line 92:** ensure_directory - Direct call: ensure_directory
- **Line 104:** safe_file_exists - Direct call: safe_file_exists
- **Line 126:** safe_file_exists - Direct call: safe_file_exists
- **Line 152:** safe_file_exists - Direct call: safe_file_exists
- **Line 154:** ensure_directory - Direct call: ensure_directory
- **Line 166:** safe_file_exists - Direct call: safe_file_exists
- **Line 188:** safe_file_exists - Direct call: safe_file_exists
- **Line 214:** safe_file_exists - Direct call: safe_file_exists
- **Line 216:** ensure_directory - Direct call: ensure_directory
- **Line 228:** safe_file_exists - Direct call: safe_file_exists

---

## src/monitoring/advanced_tracer.py

**Undefined function calls: 9**

- **Line 57:** field - Direct call: field
- **Line 59:** field - Direct call: field
- **Line 74:** field - Direct call: field
- **Line 75:** field - Direct call: field
- **Line 122:** log_execution_time - Direct call: log_execution_time
- **Line 123:** handles_errors - Direct call: handles_errors
- **Line 145:** handles_errors - Direct call: handles_errors
- **Line 164:** handles_errors - Direct call: handles_errors
- **Line 180:** handles_errors - Direct call: handles_errors

---

## src/monitoring/correlation_manager.py

**Undefined function calls: 3**

- **Line 70:** handles_errors - Direct call: handles_errors
- **Line 86:** handles_errors - Direct call: handles_errors
- **Line 110:** handles_errors - Direct call: handles_errors

---

## src/monitoring/csv_export_manager.py

**Undefined function calls: 5**

- **Line 66:** handles_errors - Direct call: handles_errors
- **Line 543:** handles_errors - Direct call: handles_errors
- **Line 715:** handles_errors - Direct call: handles_errors
- **Line 848:** handles_errors - Direct call: handles_errors
- **Line 57:** Path - Direct call: Path

---

## src/monitoring/csv_exporter.py

**Undefined function calls: 7**

- **Line 49:** log_execution_time - Direct call: log_execution_time
- **Line 50:** cached - Direct call: cached
- **Line 51:** handles_errors - Direct call: handles_errors
- **Line 74:** log_execution_time - Direct call: log_execution_time
- **Line 75:** cached - Direct call: cached
- **Line 76:** handles_errors - Direct call: handles_errors
- **Line 43:** Path - Direct call: Path

---

## src/monitoring/daily_summary_tracker.py

**Undefined function calls: 13**

- **Line 118:** handles_errors - Direct call: handles_errors
- **Line 427:** handles_errors - Direct call: handles_errors
- **Line 455:** handles_errors - Direct call: handles_errors
- **Line 468:** handles_errors - Direct call: handles_errors
- **Line 486:** handles_errors - Direct call: handles_errors
- **Line 500:** handles_errors - Direct call: handles_errors
- **Line 105:** defaultdict - Direct call: defaultdict
- **Line 109:** Path - Direct call: Path
- **Line 114:** defaultdict - Direct call: defaultdict
- **Line 204:** defaultdict - Direct call: defaultdict
- **Line 432:** asdict - Direct call: asdict
- **Line 478:** timedelta - Direct call: timedelta
- **Line 519:** asdict - Direct call: asdict

---

## src/monitoring/enhanced_ml_monitoring.py

**Undefined function calls: 9**

- **Line 231:** handles_errors - Direct call: handles_errors
- **Line 256:** handles_errors - Direct call: handles_errors
- **Line 274:** handles_errors - Direct call: handles_errors
- **Line 305:** handles_errors - Direct call: handles_errors
- **Line 493:** handles_errors - Direct call: handles_errors
- **Line 530:** handles_errors - Direct call: handles_errors
- **Line 206:** Path - Direct call: Path
- **Line 222:** SHAPAnalyzer - Direct call: SHAPAnalyzer
- **Line 223:** LIMEAnalyzer - Direct call: LIMEAnalyzer

---

## src/monitoring/enhanced_ml_tracker.py

**Undefined function calls: 1**

- **Line 42:** handles_errors - Direct call: handles_errors

---

## src/monitoring/ensemble_monitor.py

**Undefined function calls: 10**

- **Line 117:** handles_errors - Direct call: handles_errors
- **Line 342:** handles_errors - Direct call: handles_errors
- **Line 427:** handles_errors - Direct call: handles_errors
- **Line 107:** defaultdict - Direct call: defaultdict
- **Line 108:** defaultdict - Direct call: defaultdict
- **Line 109:** defaultdict - Direct call: defaultdict
- **Line 113:** defaultdict - Direct call: defaultdict
- **Line 107:** deque - Direct call: deque
- **Line 108:** deque - Direct call: deque
- **Line 109:** deque - Direct call: deque

---

## src/monitoring/error_detection_system.py

**Undefined function calls: 1**

- **Line 54:** handles_errors - Direct call: handles_errors

---

## src/monitoring/explainability_integration.py

**Undefined function calls: 5**

- **Line 106:** handles_errors - Direct call: handles_errors
- **Line 377:** handles_errors - Direct call: handles_errors
- **Line 363:** hash - Direct call: hash
- **Line 88:** SHAPAnalyzer - Direct call: SHAPAnalyzer
- **Line 94:** LIMEAnalyzer - Direct call: LIMEAnalyzer

---

## src/monitoring/fractional_performance_tracker.py

**Undefined function calls: 3**

- **Line 21:** Path - Direct call: Path
- **Line 23:** get_logger - Direct call: get_logger
- **Line 214:** timedelta - Direct call: timedelta

---

## src/monitoring/fractional_system_monitor.py

**Undefined function calls: 3**

- **Line 27:** Path - Direct call: Path
- **Line 30:** get_logger - Direct call: get_logger
- **Line 381:** Path - Direct call: Path

---

## src/monitoring/gui/data_visualization.py

**Undefined function calls: 2**

- **Line 33:** Figure - Direct call: Figure
- **Line 34:** FigureCanvasTkAgg - Direct call: FigureCanvasTkAgg

---

## src/monitoring/gui/enhanced_dashboard.py

**Undefined function calls: 8**

- **Line 89:** VisualizationControlPanel - Direct call: VisualizationControlPanel
- **Line 95:** MonitoringVisualization - Direct call: MonitoringVisualization
- **Line 156:** VisualizationControlPanel - Direct call: VisualizationControlPanel
- **Line 162:** MonitoringVisualization - Direct call: MonitoringVisualization
- **Line 220:** VisualizationControlPanel - Direct call: VisualizationControlPanel
- **Line 226:** MonitoringVisualization - Direct call: MonitoringVisualization
- **Line 262:** VisualizationControlPanel - Direct call: VisualizationControlPanel
- **Line 268:** MonitoringVisualization - Direct call: MonitoringVisualization

---

## src/monitoring/gui/launch_dashboard.py

**Undefined function calls: 2**

- **Line 100:** create_enhanced_monitoring_dashboard - Direct call: create_enhanced_monitoring_dashboard
- **Line 14:** Path - Direct call: Path

---

## src/monitoring/gui/monitoring_dashboard.py

**Undefined function calls: 1**

- **Line 353:** Path - Direct call: Path

---

## src/monitoring/integration_manager.py

**Undefined function calls: 2**

- **Line 41:** log_execution_time - Direct call: log_execution_time
- **Line 42:** handles_errors - Direct call: handles_errors

---

## src/monitoring/metrics_dashboard.py

**Undefined function calls: 1**

- **Line 65:** handles_errors - Direct call: handles_errors

---

## src/monitoring/ml_monitor.py

**Undefined function calls: 3**

- **Line 95:** log_execution_time - Direct call: log_execution_time
- **Line 96:** handles_errors - Direct call: handles_errors
- **Line 111:** handles_errors - Direct call: handles_errors

---

## src/monitoring/performance_dashboard.py

**Undefined function calls: 5**

- **Line 73:** log_execution_time - Direct call: log_execution_time
- **Line 74:** log_execution_time - Direct call: log_execution_time
- **Line 75:** cached - Direct call: cached
- **Line 76:** handles_errors - Direct call: handles_errors
- **Line 70:** Path - Direct call: Path

---

## src/monitoring/performance_monitor.py

**Undefined function calls: 6**

- **Line 26:** field - Direct call: field
- **Line 67:** log_execution_time - Direct call: log_execution_time
- **Line 68:** log_execution_time - Direct call: log_execution_time
- **Line 69:** cached - Direct call: cached
- **Line 70:** handles_errors - Direct call: handles_errors
- **Line 63:** deque - Direct call: deque

---

## src/monitoring/regime_monitoring_dashboard.py

**Undefined function calls: 11**

- **Line 38:** RegimePerformanceTracker - Direct call: RegimePerformanceTracker
- **Line 41:** defaultdict - Direct call: defaultdict
- **Line 42:** deque - Direct call: deque
- **Line 44:** defaultdict - Direct call: defaultdict
- **Line 48:** Path - Direct call: Path
- **Line 153:** log_func - Direct call: log_func
- **Line 249:** main - Direct call: main
- **Line 86:** Path - Direct call: Path
- **Line 197:** defaultdict - Direct call: defaultdict
- **Line 248:** start_regime_monitoring - Direct call: start_regime_monitoring
- **Line 41:** deque - Direct call: deque

---

## src/monitoring/regime_performance_tracker.py

**Undefined function calls: 6**

- **Line 27:** defaultdict - Direct call: defaultdict
- **Line 184:** main - Direct call: main
- **Line 25:** Path - Direct call: Path
- **Line 65:** timedelta - Direct call: timedelta
- **Line 27:** defaultdict - Direct call: defaultdict
- **Line 149:** timedelta - Direct call: timedelta

---

## src/monitoring/regime_sr_tracker.py

**Undefined function calls: 1**

- **Line 31:** handles_errors - Direct call: handles_errors

---

## src/monitoring/report_scheduler.py

**Undefined function calls: 2**

- **Line 72:** handles_errors - Direct call: handles_errors
- **Line 69:** Path - Direct call: Path

---

## src/monitoring/surrogate_optimization_monitor.py

**Undefined function calls: 1**

- **Line 285:** asdict - Direct call: asdict

---

## src/monitoring/tracking_system.py

**Undefined function calls: 1**

- **Line 31:** handles_errors - Direct call: handles_errors

---

## src/monitoring/trade_conditions_monitor.py

**Undefined function calls: 1**

- **Line 32:** handles_errors - Direct call: handles_errors

---

## src/monitoring/trading_integration.py

**Undefined function calls: 19**

- **Line 61:** handles_errors - Direct call: handles_errors
- **Line 88:** handles_errors - Direct call: handles_errors
- **Line 115:** handles_errors - Direct call: handles_errors
- **Line 239:** handles_errors - Direct call: handles_errors
- **Line 596:** handles_errors - Direct call: handles_errors
- **Line 51:** EnhancedMLMonitor - Direct call: EnhancedMLMonitor
- **Line 52:** ExplainabilityIntegrator - Direct call: ExplainabilityIntegrator
- **Line 53:** EnsembleMonitor - Direct call: EnsembleMonitor
- **Line 268:** TradeDecision - Direct call: TradeDecision
- **Line 317:** TradeContext - Direct call: TradeContext
- **Line 391:** EnsembleDecision - Direct call: EnsembleDecision
- **Line 406:** EnsembleDecision - Direct call: EnsembleDecision
- **Line 464:** MLModelDecision - Direct call: MLModelDecision
- **Line 159:** original_execute_trade - Direct call: original_execute_trade
- **Line 178:** original_get_prediction - Direct call: original_get_prediction
- **Line 199:** original_execute_trade - Direct call: original_execute_trade
- **Line 226:** original_execute_trade - Direct call: original_execute_trade
- **Line 353:** TradingIndicator - Direct call: TradingIndicator
- **Line 466:** ModelType - Direct call: ModelType

---

## src/optimization/hmm_regime_ab_testing.py

**Undefined function calls: 4**

- **Line 65:** handles_errors - Direct call: handles_errors
- **Line 291:** handles_errors - Direct call: handles_errors
- **Line 416:** handles_errors - Direct call: handles_errors
- **Line 362:** hash - Direct call: hash

---

## src/optimization/ml_optimized_barriers.py

**Undefined function calls: 5**

- **Line 75:** handles_errors - Direct call: handles_errors
- **Line 116:** handles_errors - Direct call: handles_errors
- **Line 459:** handles_errors - Direct call: handles_errors
- **Line 481:** handles_errors - Direct call: handles_errors
- **Line 302:** minimize - Direct call: minimize

---

## src/optimization/regime_parameter_optimizer.py

**Undefined function calls: 5**

- **Line 48:** WalkForwardValidator - Direct call: WalkForwardValidator
- **Line 49:** Path - Direct call: Path
- **Line 240:** main - Direct call: main
- **Line 132:** OptimizedTripleBarrierLabeling - Direct call: OptimizedTripleBarrierLabeling
- **Line 235:** optimize_regime_parameters - Direct call: optimize_regime_parameters

---

## src/paper_trader.py

**Undefined function calls: 44**

- **Line 747:** handles_errors - Direct call: handles_errors
- **Line 93:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 132:** handles_errors - Direct call: handles_errors
- **Line 161:** handles_errors - Direct call: handles_errors
- **Line 203:** handles_errors - Direct call: handles_errors
- **Line 229:** handles_errors - Direct call: handles_errors
- **Line 230:** log_execution_time - Direct call: log_execution_time
- **Line 231:** log_call - Direct call: log_call
- **Line 232:** traced - Direct call: traced
- **Line 336:** handles_errors - Direct call: handles_errors
- **Line 337:** log_execution_time - Direct call: log_execution_time
- **Line 338:** log_call - Direct call: log_call
- **Line 339:** traced - Direct call: traced
- **Line 451:** handles_errors - Direct call: handles_errors
- **Line 500:** handles_errors - Direct call: handles_errors
- **Line 550:** handles_errors - Direct call: handles_errors
- **Line 569:** handles_errors - Direct call: handles_errors
- **Line 588:** handles_errors - Direct call: handles_errors
- **Line 614:** handles_errors - Direct call: handles_errors
- **Line 720:** handles_errors - Direct call: handles_errors
- **Line 91:** get_trade_tracker - Direct call: get_trade_tracker
- **Line 117:** invalid - Direct call: invalid
- **Line 128:** initialization_error - Direct call: initialization_error
- **Line 158:** initialization_error - Direct call: initialization_error
- **Line 176:** invalid - Direct call: invalid
- **Line 181:** invalid - Direct call: invalid
- **Line 186:** invalid - Direct call: invalid
- **Line 191:** invalid - Direct call: invalid
- **Line 199:** validation_error - Direct call: validation_error
- **Line 226:** initialization_error - Direct call: initialization_error
- **Line 333:** execution_error - Direct call: execution_error
- **Line 448:** execution_error - Direct call: execution_error
- **Line 471:** invalid - Direct call: invalid
- **Line 476:** invalid - Direct call: invalid
- **Line 481:** invalid - Direct call: invalid
- **Line 497:** validation_error - Direct call: validation_error
- **Line 520:** execution_error - Direct call: execution_error
- **Line 533:** execution_error - Direct call: execution_error
- **Line 548:** execution_error - Direct call: execution_error
- **Line 566:** execution_error - Direct call: execution_error
- **Line 585:** execution_error - Direct call: execution_error
- **Line 611:** execution_error - Direct call: execution_error
- **Line 698:** execution_error - Direct call: execution_error
- **Line 740:** execution_error - Direct call: execution_error

---

## src/pipelines/base_pipeline.py

**Undefined function calls: 10**

- **Line 28:** field - Direct call: field
- **Line 29:** field - Direct call: field
- **Line 30:** field - Direct call: field
- **Line 31:** field - Direct call: field
- **Line 78:** log_execution_time - Direct call: log_execution_time
- **Line 79:** cached - Direct call: cached
- **Line 80:** handles_errors - Direct call: handles_errors
- **Line 92:** log_execution_time - Direct call: log_execution_time
- **Line 93:** cached - Direct call: cached
- **Line 94:** handles_errors - Direct call: handles_errors

---

## src/strategist/enhanced_regime_classifier.py

**Undefined function calls: 3**

- **Line 68:** handles_errors - Direct call: handles_errors
- **Line 208:** handles_errors - Direct call: handles_errors
- **Line 54:** StandardScaler - Direct call: StandardScaler

---

## src/strategist/strategist.py

**Undefined function calls: 19**

- **Line 86:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 147:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 156:** create_strategy_validator - Direct call: create_strategy_validator
- **Line 574:** handles_errors - Direct call: handles_errors
- **Line 60:** StrategistConfig - Direct call: StrategistConfig
- **Line 63:** PerformanceOptimizer - Direct call: PerformanceOptimizer
- **Line 70:** StrategyComponentExtractor - Direct call: StrategyComponentExtractor
- **Line 247:** validate_required_columns - Direct call: validate_required_columns
- **Line 248:** validate_data_sufficiency - Direct call: validate_data_sufficiency
- **Line 287:** MarketIndicators - Direct call: MarketIndicators
- **Line 130:** log_error - Direct call: log_error
- **Line 144:** log_error - Direct call: log_error
- **Line 233:** log_error - Direct call: log_error
- **Line 236:** log_error - Direct call: log_error
- **Line 299:** CalculationError - Direct call: CalculationError
- **Line 316:** StrategyResult - Direct call: StrategyResult
- **Line 495:** log_error - Direct call: log_error
- **Line 590:** log_error - Direct call: log_error
- **Line 116:** _EnhancedRegimeClassifier - Direct call: _EnhancedRegimeClassifier

---

## src/strategist/strategist_backup.py

**Undefined function calls: 17**

- **Line 92:** handles_errors - Direct call: handles_errors
- **Line 126:** handles_errors - Direct call: handles_errors
- **Line 145:** handles_errors - Direct call: handles_errors
- **Line 172:** handles_errors - Direct call: handles_errors
- **Line 232:** handles_errors - Direct call: handles_errors
- **Line 261:** handles_errors - Direct call: handles_errors
- **Line 300:** handles_errors - Direct call: handles_errors
- **Line 317:** handles_errors - Direct call: handles_errors
- **Line 354:** handles_errors - Direct call: handles_errors
- **Line 401:** handles_errors - Direct call: handles_errors
- **Line 439:** handles_errors - Direct call: handles_errors
- **Line 498:** handles_errors - Direct call: handles_errors
- **Line 113:** invalid - Direct call: invalid
- **Line 123:** failed - Direct call: failed
- **Line 163:** invalid - Direct call: invalid
- **Line 509:** failed - Direct call: failed
- **Line 156:** missing - Direct call: missing

---

## src/supervisor/coordinator/circuit_breaker.py

**Undefined function calls: 2**

- **Line 31:** handles_errors - Direct call: handles_errors
- **Line 55:** func - Direct call: func

---

## src/supervisor/coordinator/component_monitor.py

**Undefined function calls: 8**

- **Line 34:** handles_errors - Direct call: handles_errors
- **Line 73:** handles_errors - Direct call: handles_errors
- **Line 111:** handles_errors - Direct call: handles_errors
- **Line 150:** handles_errors - Direct call: handles_errors
- **Line 70:** error - Direct call: error
- **Line 108:** error - Direct call: error
- **Line 147:** error - Direct call: error
- **Line 186:** error - Direct call: error

---

## src/supervisor/coordinator/health_monitor.py

**Undefined function calls: 2**

- **Line 43:** handles_errors - Direct call: handles_errors
- **Line 77:** error - Direct call: error

---

## src/supervisor/coordinator/online_learning_manager.py

**Undefined function calls: 5**

- **Line 37:** handles_errors - Direct call: handles_errors
- **Line 64:** handles_errors - Direct call: handles_errors
- **Line 30:** defaultdict - Direct call: defaultdict
- **Line 62:** error - Direct call: error
- **Line 107:** error - Direct call: error

---

## src/supervisor/coordinator/recovery_manager.py

**Undefined function calls: 7**

- **Line 50:** handles_errors - Direct call: handles_errors
- **Line 33:** defaultdict - Direct call: defaultdict
- **Line 67:** warning - Direct call: warning
- **Line 71:** failed - Direct call: failed
- **Line 86:** error - Direct call: error
- **Line 139:** recovery_func - Direct call: recovery_func
- **Line 147:** error - Direct call: error

---

## src/supervisor/coordinator/system_coordinator.py

**Undefined function calls: 14**

- **Line 81:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 165:** handles_errors - Direct call: handles_errors
- **Line 61:** OnlineLearningManager - Direct call: OnlineLearningManager
- **Line 64:** ComponentMonitor - Direct call: ComponentMonitor
- **Line 65:** HealthMonitor - Direct call: HealthMonitor
- **Line 66:** RecoveryManager - Direct call: RecoveryManager
- **Line 144:** CircuitBreaker - Direct call: CircuitBreaker
- **Line 155:** EnhancedPredictionService - Direct call: EnhancedPredictionService
- **Line 222:** error - Direct call: error
- **Line 252:** error - Direct call: error
- **Line 273:** error - Direct call: error
- **Line 292:** error - Direct call: error
- **Line 311:** error - Direct call: error
- **Line 334:** failed - Direct call: failed

---

## src/supervisor/dependency_container.py

**Undefined function calls: 4**

- **Line 106:** ModelManager - Direct call: ModelManager
- **Line 127:** ModelManager - Direct call: ModelManager
- **Line 149:** ModelManager - Direct call: ModelManager
- **Line 172:** Sentinel - Direct call: Sentinel

---

## src/supervisor/dynamic_weighter.py

**Undefined function calls: 28**

- **Line 81:** handles_errors - Direct call: handles_errors
- **Line 120:** handles_errors - Direct call: handles_errors
- **Line 149:** handles_errors - Direct call: handles_errors
- **Line 188:** handles_errors - Direct call: handles_errors
- **Line 217:** handles_errors - Direct call: handles_errors
- **Line 233:** handles_errors - Direct call: handles_errors
- **Line 249:** handles_errors - Direct call: handles_errors
- **Line 265:** handles_errors - Direct call: handles_errors
- **Line 281:** handles_errors - Direct call: handles_errors
- **Line 297:** handles_errors - Direct call: handles_errors
- **Line 363:** handles_errors - Direct call: handles_errors
- **Line 392:** handles_errors - Direct call: handles_errors
- **Line 418:** handles_errors - Direct call: handles_errors
- **Line 444:** handles_errors - Direct call: handles_errors
- **Line 796:** handles_errors - Direct call: handles_errors
- **Line 982:** handles_errors - Direct call: handles_errors
- **Line 1066:** handles_errors - Direct call: handles_errors
- **Line 1085:** handles_errors - Direct call: handles_errors
- **Line 1107:** handles_errors - Direct call: handles_errors
- **Line 1164:** handles_errors - Direct call: handles_errors
- **Line 1224:** handles_errors - Direct call: handles_errors
- **Line 1268:** handles_errors - Direct call: handles_errors
- **Line 1305:** handles_errors - Direct call: handles_errors
- **Line 1422:** handles_errors - Direct call: handles_errors
- **Line 1442:** handles_errors - Direct call: handles_errors
- **Line 410:** method_func - Direct call: method_func
- **Line 436:** method_func - Direct call: method_func
- **Line 1191:** deque - Direct call: deque

---

## src/supervisor/enhanced_model_monitor.py

**Undefined function calls: 5**

- **Line 110:** handles_errors - Direct call: handles_errors
- **Line 125:** handles_errors - Direct call: handles_errors
- **Line 133:** handles_errors - Direct call: handles_errors
- **Line 141:** handles_errors - Direct call: handles_errors
- **Line 149:** handles_errors - Direct call: handles_errors

---

## src/supervisor/enhanced_prediction_service.py

**Undefined function calls: 65**

- **Line 75:** handles_errors - Direct call: handles_errors
- **Line 76:** with_tracing_span - Direct call: with_tracing_span
- **Line 92:** handles_errors - Direct call: handles_errors
- **Line 93:** with_tracing_span - Direct call: with_tracing_span
- **Line 94:** intelligent_caching - Direct call: intelligent_caching
- **Line 123:** handles_errors - Direct call: handles_errors
- **Line 124:** with_tracing_span - Direct call: with_tracing_span
- **Line 125:** intelligent_caching - Direct call: intelligent_caching
- **Line 154:** handles_errors - Direct call: handles_errors
- **Line 155:** with_tracing_span - Direct call: with_tracing_span
- **Line 174:** handles_errors - Direct call: handles_errors
- **Line 175:** with_tracing_span - Direct call: with_tracing_span
- **Line 197:** with_tracing_span - Direct call: with_tracing_span
- **Line 198:** validate_data_quality - Direct call: validate_data_quality
- **Line 246:** with_tracing_span - Direct call: with_tracing_span
- **Line 272:** with_tracing_span - Direct call: with_tracing_span
- **Line 346:** with_tracing_span - Direct call: with_tracing_span
- **Line 391:** handles_errors - Direct call: handles_errors
- **Line 392:** with_tracing_span - Direct call: with_tracing_span
- **Line 418:** with_tracing_span - Direct call: with_tracing_span
- **Line 456:** handles_errors - Direct call: handles_errors
- **Line 457:** with_tracing_span - Direct call: with_tracing_span
- **Line 64:** get_logger - Direct call: get_logger
- **Line 63:** get_enhanced_prediction_service_config - Direct call: get_enhanced_prediction_service_config
- **Line 159:** Path - Direct call: Path
- **Line 179:** Path - Direct call: Path
- **Line 38:** Path - Direct call: Path
- **Line 38:** Path - Direct call: Path
- **Line 38:** Path - Direct call: Path
- **Line 89:** error - Direct call: error
- **Line 98:** Path - Direct call: Path
- **Line 120:** error - Direct call: error
- **Line 129:** Path - Direct call: Path
- **Line 151:** error - Direct call: error
- **Line 172:** error - Direct call: error
- **Line 194:** error - Direct call: error
- **Line 242:** error - Direct call: error
- **Line 268:** error - Direct call: error
- **Line 294:** error - Direct call: error
- **Line 342:** error - Direct call: error
- **Line 379:** warning - Direct call: warning
- **Line 381:** warning - Direct call: warning
- **Line 384:** warning - Direct call: warning
- **Line 388:** error - Direct call: error
- **Line 406:** warning - Direct call: warning
- **Line 414:** error - Direct call: error
- **Line 450:** warning - Direct call: warning
- **Line 453:** error - Direct call: error
- **Line 466:** warning - Direct call: warning
- **Line 468:** warning - Direct call: warning
- **Line 472:** warning - Direct call: warning
- **Line 476:** error - Direct call: error
- **Line 486:** error - Direct call: error
- **Line 368:** warning - Direct call: warning
- **Line 372:** warning - Direct call: warning
- **Line 170:** warning - Direct call: warning
- **Line 192:** warning - Direct call: warning
- **Line 256:** warning - Direct call: warning
- **Line 265:** warning - Direct call: warning
- **Line 282:** warning - Direct call: warning
- **Line 291:** warning - Direct call: warning
- **Line 111:** warning - Direct call: warning
- **Line 116:** warning - Direct call: warning
- **Line 142:** warning - Direct call: warning
- **Line 147:** warning - Direct call: warning

---

## src/supervisor/global_portfolio_manager.py

**Undefined function calls: 65**

- **Line 1229:** handles_errors - Direct call: handles_errors
- **Line 67:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 113:** handles_errors - Direct call: handles_errors
- **Line 150:** handles_errors - Direct call: handles_errors
- **Line 193:** handles_errors - Direct call: handles_errors
- **Line 230:** handles_errors - Direct call: handles_errors
- **Line 253:** handles_errors - Direct call: handles_errors
- **Line 273:** handles_errors - Direct call: handles_errors
- **Line 294:** handles_errors - Direct call: handles_errors
- **Line 317:** handles_errors - Direct call: handles_errors
- **Line 338:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 412:** handles_errors - Direct call: handles_errors
- **Line 452:** handles_errors - Direct call: handles_errors
- **Line 504:** handles_errors - Direct call: handles_errors
- **Line 553:** handles_errors - Direct call: handles_errors
- **Line 605:** handles_errors - Direct call: handles_errors
- **Line 661:** handles_errors - Direct call: handles_errors
- **Line 1098:** handles_errors - Direct call: handles_errors
- **Line 1121:** handles_errors - Direct call: handles_errors
- **Line 1148:** handles_errors - Direct call: handles_errors
- **Line 1200:** handles_errors - Direct call: handles_errors
- **Line 95:** invalid - Direct call: invalid
- **Line 148:** error - Direct call: error
- **Line 165:** invalid - Direct call: invalid
- **Line 170:** invalid - Direct call: invalid
- **Line 183:** error - Direct call: error
- **Line 190:** error - Direct call: error
- **Line 227:** initialization_error - Direct call: initialization_error
- **Line 250:** initialization_error - Direct call: initialization_error
- **Line 271:** initialization_error - Direct call: initialization_error
- **Line 292:** initialization_error - Direct call: initialization_error
- **Line 314:** initialization_error - Direct call: initialization_error
- **Line 336:** initialization_error - Direct call: initialization_error
- **Line 408:** error - Direct call: error
- **Line 439:** invalid - Direct call: invalid
- **Line 443:** invalid - Direct call: invalid
- **Line 449:** error - Direct call: error
- **Line 501:** error - Direct call: error
- **Line 550:** error - Direct call: error
- **Line 602:** error - Direct call: error
- **Line 658:** error - Direct call: error
- **Line 710:** error - Direct call: error
- **Line 730:** error - Direct call: error
- **Line 754:** error - Direct call: error
- **Line 777:** error - Direct call: error
- **Line 800:** error - Direct call: error
- **Line 823:** error - Direct call: error
- **Line 841:** error - Direct call: error
- **Line 859:** error - Direct call: error
- **Line 880:** error - Direct call: error
- **Line 899:** error - Direct call: error
- **Line 917:** error - Direct call: error
- **Line 936:** error - Direct call: error
- **Line 957:** error - Direct call: error
- **Line 976:** error - Direct call: error
- **Line 998:** error - Direct call: error
- **Line 1017:** error - Direct call: error
- **Line 1038:** error - Direct call: error
- **Line 1057:** error - Direct call: error
- **Line 1076:** error - Direct call: error
- **Line 1095:** error - Direct call: error
- **Line 1119:** error - Direct call: error
- **Line 1145:** error - Direct call: error
- **Line 1172:** error - Direct call: error
- **Line 1222:** error - Direct call: error

---

## src/supervisor/loss_functions/base.py

**Undefined function calls: 4**

- **Line 48:** handles_errors - Direct call: handles_errors
- **Line 78:** handles_errors - Direct call: handles_errors
- **Line 94:** handles_errors - Direct call: handles_errors
- **Line 114:** handles_errors - Direct call: handles_errors

---

## src/supervisor/loss_functions/loss_calculator.py

**Undefined function calls: 3**

- **Line 29:** handles_errors - Direct call: handles_errors
- **Line 81:** handles_errors - Direct call: handles_errors
- **Line 142:** handles_errors - Direct call: handles_errors

---

## src/supervisor/loss_functions/optimization_metrics.py

**Undefined function calls: 4**

- **Line 29:** handles_errors - Direct call: handles_errors
- **Line 75:** handles_errors - Direct call: handles_errors
- **Line 117:** handles_errors - Direct call: handles_errors
- **Line 155:** handles_errors - Direct call: handles_errors

---

## src/supervisor/loss_functions/performance_metrics.py

**Undefined function calls: 4**

- **Line 29:** handles_errors - Direct call: handles_errors
- **Line 91:** handles_errors - Direct call: handles_errors
- **Line 129:** handles_errors - Direct call: handles_errors
- **Line 169:** handles_errors - Direct call: handles_errors

---

## src/supervisor/loss_functions/pnl_calculator.py

**Undefined function calls: 3**

- **Line 30:** handles_errors - Direct call: handles_errors
- **Line 84:** handles_errors - Direct call: handles_errors
- **Line 118:** handles_errors - Direct call: handles_errors

---

## src/supervisor/loss_functions/risk_metrics.py

**Undefined function calls: 4**

- **Line 30:** handles_errors - Direct call: handles_errors
- **Line 63:** handles_errors - Direct call: handles_errors
- **Line 101:** handles_errors - Direct call: handles_errors
- **Line 158:** handles_errors - Direct call: handles_errors

---

## src/supervisor/main.py

**Undefined function calls: 18**

- **Line 92:** handles_errors - Direct call: handles_errors
- **Line 125:** handles_errors - Direct call: handles_errors
- **Line 173:** handles_errors - Direct call: handles_errors
- **Line 192:** handles_errors - Direct call: handles_errors
- **Line 203:** handles_errors - Direct call: handles_errors
- **Line 218:** handles_errors - Direct call: handles_errors
- **Line 232:** handles_errors - Direct call: handles_errors
- **Line 244:** handles_errors - Direct call: handles_errors
- **Line 30:** DependencyContainer - Direct call: DependencyContainer
- **Line 31:** ComponentBuilder - Direct call: ComponentBuilder
- **Line 32:** RiskAllocator - Direct call: RiskAllocator
- **Line 33:** PerformanceReporter - Direct call: PerformanceReporter
- **Line 34:** ABTester - Direct call: ABTester
- **Line 35:** Monitoring - Direct call: Monitoring
- **Line 36:** get_environment_settings - Direct call: get_environment_settings
- **Line 48:** ModelManager - Direct call: ModelManager
- **Line 38:** PaperTrader - Direct call: PaperTrader
- **Line 180:** initialize_sr_parameters - Direct call: initialize_sr_parameters

---

## src/supervisor/model_behavior_tracker.py

**Undefined function calls: 29**

- **Line 90:** handles_errors - Direct call: handles_errors
- **Line 105:** handles_errors - Direct call: handles_errors
- **Line 114:** handles_errors - Direct call: handles_errors
- **Line 123:** handles_errors - Direct call: handles_errors
- **Line 132:** handles_errors - Direct call: handles_errors
- **Line 141:** handles_errors - Direct call: handles_errors
- **Line 154:** handles_errors - Direct call: handles_errors
- **Line 165:** handles_errors - Direct call: handles_errors
- **Line 268:** handles_errors - Direct call: handles_errors
- **Line 395:** error - Direct call: error
- **Line 112:** error - Direct call: error
- **Line 121:** initialization_error - Direct call: initialization_error
- **Line 130:** initialization_error - Direct call: initialization_error
- **Line 139:** initialization_error - Direct call: initialization_error
- **Line 151:** failed - Direct call: failed
- **Line 187:** error - Direct call: error
- **Line 197:** error - Direct call: error
- **Line 207:** error - Direct call: error
- **Line 228:** error - Direct call: error
- **Line 238:** error - Direct call: error
- **Line 249:** error - Direct call: error
- **Line 279:** error - Direct call: error
- **Line 302:** error - Direct call: error
- **Line 318:** error - Direct call: error
- **Line 333:** error - Direct call: error
- **Line 352:** error - Direct call: error
- **Line 374:** error - Direct call: error
- **Line 162:** error - Direct call: error
- **Line 368:** asdict - Direct call: asdict

---

## src/supervisor/monitoring.py

**Undefined function calls: 20**

- **Line 198:** handles_errors - Direct call: handles_errors
- **Line 38:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 60:** handles_errors - Direct call: handles_errors
- **Line 75:** handles_errors - Direct call: handles_errors
- **Line 94:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 114:** handles_errors - Direct call: handles_errors
- **Line 132:** handles_errors - Direct call: handles_errors
- **Line 151:** handles_errors - Direct call: handles_errors
- **Line 165:** handles_errors - Direct call: handles_errors
- **Line 52:** invalid - Direct call: invalid
- **Line 57:** failed - Direct call: failed
- **Line 73:** error - Direct call: error
- **Line 83:** invalid - Direct call: invalid
- **Line 86:** invalid - Direct call: invalid
- **Line 91:** error - Direct call: error
- **Line 110:** error - Direct call: error
- **Line 130:** error - Direct call: error
- **Line 149:** error - Direct call: error
- **Line 163:** error - Direct call: error
- **Line 177:** error - Direct call: error

---

## src/supervisor/optimizer.py

**Undefined function calls: 9**

- **Line 34:** handles_errors - Direct call: handles_errors
- **Line 56:** handles_errors - Direct call: handles_errors
- **Line 67:** handles_errors - Direct call: handles_errors
- **Line 82:** handles_errors - Direct call: handles_errors
- **Line 102:** handles_errors - Direct call: handles_errors
- **Line 116:** handles_errors - Direct call: handles_errors
- **Line 131:** handles_errors - Direct call: handles_errors
- **Line 142:** handles_errors - Direct call: handles_errors
- **Line 167:** handles_errors - Direct call: handles_errors

---

## src/supervisor/performance_monitor.py

**Undefined function calls: 30**

- **Line 830:** handles_errors - Direct call: handles_errors
- **Line 78:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 115:** handles_errors - Direct call: handles_errors
- **Line 137:** handles_errors - Direct call: handles_errors
- **Line 156:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 176:** handles_errors - Direct call: handles_errors
- **Line 194:** handles_errors - Direct call: handles_errors
- **Line 214:** handles_errors - Direct call: handles_errors
- **Line 244:** handles_errors - Direct call: handles_errors
- **Line 414:** handles_errors - Direct call: handles_errors
- **Line 461:** handles_errors - Direct call: handles_errors
- **Line 548:** handles_errors - Direct call: handles_errors
- **Line 629:** handles_errors - Direct call: handles_errors
- **Line 748:** handles_errors - Direct call: handles_errors
- **Line 93:** invalid - Direct call: invalid
- **Line 101:** failed - Direct call: failed
- **Line 109:** failed - Direct call: failed
- **Line 131:** error - Direct call: error
- **Line 134:** error - Direct call: error
- **Line 145:** invalid - Direct call: invalid
- **Line 148:** invalid - Direct call: invalid
- **Line 153:** error - Direct call: error
- **Line 172:** error - Direct call: error
- **Line 192:** error - Direct call: error
- **Line 212:** error - Direct call: error
- **Line 229:** warning - Direct call: warning
- **Line 238:** warning - Direct call: warning
- **Line 242:** error - Direct call: error
- **Line 256:** error - Direct call: error
- **Line 368:** warning - Direct call: warning

---

## src/supervisor/performance_reporter.py

**Undefined function calls: 21**

- **Line 26:** handles_errors - Direct call: handles_errors
- **Line 59:** handles_errors - Direct call: handles_errors
- **Line 93:** handles_errors - Direct call: handles_errors
- **Line 126:** handles_errors - Direct call: handles_errors
- **Line 150:** handles_errors - Direct call: handles_errors
- **Line 173:** handles_errors - Direct call: handles_errors
- **Line 404:** handles_errors - Direct call: handles_errors
- **Line 450:** handles_errors - Direct call: handles_errors
- **Line 477:** handles_errors - Direct call: handles_errors
- **Line 513:** handles_errors - Direct call: handles_errors
- **Line 525:** handles_errors - Direct call: handles_errors
- **Line 538:** handles_errors - Direct call: handles_errors
- **Line 551:** handles_errors - Direct call: handles_errors
- **Line 576:** handles_errors - Direct call: handles_errors
- **Line 601:** handles_errors - Direct call: handles_errors
- **Line 621:** handles_errors - Direct call: handles_errors
- **Line 648:** handles_errors - Direct call: handles_errors
- **Line 680:** handles_errors - Direct call: handles_errors
- **Line 695:** handles_errors - Direct call: handles_errors
- **Line 710:** handles_errors - Direct call: handles_errors
- **Line 750:** handles_errors - Direct call: handles_errors

---

## src/supervisor/pnl_loss_functions.py

**Undefined function calls: 8**

- **Line 64:** handles_errors - Direct call: handles_errors
- **Line 101:** handles_errors - Direct call: handles_errors
- **Line 220:** handles_errors - Direct call: handles_errors
- **Line 42:** PnLCalculator - Direct call: PnLCalculator
- **Line 43:** RiskMetricsCalculator - Direct call: RiskMetricsCalculator
- **Line 44:** PerformanceMetricsCalculator - Direct call: PerformanceMetricsCalculator
- **Line 45:** OptimizationMetricsCalculator - Direct call: OptimizationMetricsCalculator
- **Line 46:** LossCalculator - Direct call: LossCalculator

---

## src/supervisor/pnl_loss_functions_backup.py

**Undefined function calls: 20**

- **Line 65:** handles_errors - Direct call: handles_errors
- **Line 86:** handles_errors - Direct call: handles_errors
- **Line 108:** handles_errors - Direct call: handles_errors
- **Line 132:** handles_errors - Direct call: handles_errors
- **Line 150:** handles_errors - Direct call: handles_errors
- **Line 159:** handles_errors - Direct call: handles_errors
- **Line 168:** handles_errors - Direct call: handles_errors
- **Line 177:** handles_errors - Direct call: handles_errors
- **Line 186:** handles_errors - Direct call: handles_errors
- **Line 195:** handles_errors - Direct call: handles_errors
- **Line 231:** handles_errors - Direct call: handles_errors
- **Line 257:** handles_errors - Direct call: handles_errors
- **Line 275:** handles_errors - Direct call: handles_errors
- **Line 293:** handles_errors - Direct call: handles_errors
- **Line 315:** handles_errors - Direct call: handles_errors
- **Line 337:** handles_errors - Direct call: handles_errors
- **Line 515:** handles_errors - Direct call: handles_errors
- **Line 527:** handles_errors - Direct call: handles_errors
- **Line 546:** handles_errors - Direct call: handles_errors
- **Line 575:** handles_errors - Direct call: handles_errors

---

## src/supervisor/risk_allocator.py

**Undefined function calls: 8**

- **Line 47:** handles_errors - Direct call: handles_errors
- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 80:** handles_errors - Direct call: handles_errors
- **Line 95:** handles_errors - Direct call: handles_errors
- **Line 115:** handles_errors - Direct call: handles_errors
- **Line 129:** handles_errors - Direct call: handles_errors
- **Line 144:** handles_errors - Direct call: handles_errors
- **Line 159:** handles_errors - Direct call: handles_errors

---

## src/supervisor/system_coordinator_backup.py

**Undefined function calls: 108**

- **Line 1059:** handles_errors - Direct call: handles_errors
- **Line 24:** handles_errors - Direct call: handles_errors
- **Line 58:** handles_errors - Direct call: handles_errors
- **Line 70:** handles_errors - Direct call: handles_errors
- **Line 141:** handles_errors - Direct call: handles_errors
- **Line 160:** handles_errors - Direct call: handles_errors
- **Line 175:** handles_errors - Direct call: handles_errors
- **Line 196:** handles_errors - Direct call: handles_errors
- **Line 205:** handles_errors - Direct call: handles_errors
- **Line 214:** handles_errors - Direct call: handles_errors
- **Line 224:** handles_errors - Direct call: handles_errors
- **Line 235:** handles_errors - Direct call: handles_errors
- **Line 251:** handles_errors - Direct call: handles_errors
- **Line 252:** with_tracing_span - Direct call: with_tracing_span
- **Line 273:** handles_errors - Direct call: handles_errors
- **Line 274:** with_tracing_span - Direct call: with_tracing_span
- **Line 296:** handles_errors - Direct call: handles_errors
- **Line 297:** with_tracing_span - Direct call: with_tracing_span
- **Line 335:** handles_errors - Direct call: handles_errors
- **Line 336:** with_tracing_span - Direct call: with_tracing_span
- **Line 422:** handles_errors - Direct call: handles_errors
- **Line 450:** handles_errors - Direct call: handles_errors
- **Line 477:** handles_errors - Direct call: handles_errors
- **Line 498:** handles_errors - Direct call: handles_errors
- **Line 525:** handles_errors - Direct call: handles_errors
- **Line 570:** handles_errors - Direct call: handles_errors
- **Line 601:** handles_errors - Direct call: handles_errors
- **Line 636:** handles_errors - Direct call: handles_errors
- **Line 645:** handles_errors - Direct call: handles_errors
- **Line 655:** handles_errors - Direct call: handles_errors
- **Line 720:** handles_errors - Direct call: handles_errors
- **Line 732:** handles_errors - Direct call: handles_errors
- **Line 744:** handles_errors - Direct call: handles_errors
- **Line 759:** handles_errors - Direct call: handles_errors
- **Line 778:** handles_errors - Direct call: handles_errors
- **Line 804:** handles_errors - Direct call: handles_errors
- **Line 825:** handles_errors - Direct call: handles_errors
- **Line 856:** handles_errors - Direct call: handles_errors
- **Line 876:** handles_errors - Direct call: handles_errors
- **Line 897:** handles_errors - Direct call: handles_errors
- **Line 908:** handles_errors - Direct call: handles_errors
- **Line 919:** handles_errors - Direct call: handles_errors
- **Line 930:** handles_errors - Direct call: handles_errors
- **Line 941:** handles_errors - Direct call: handles_errors
- **Line 952:** handles_errors - Direct call: handles_errors
- **Line 963:** handles_errors - Direct call: handles_errors
- **Line 974:** handles_errors - Direct call: handles_errors
- **Line 984:** handles_errors - Direct call: handles_errors
- **Line 997:** handles_errors - Direct call: handles_errors
- **Line 1029:** handles_errors - Direct call: handles_errors
- **Line 1064:** Supervisor - Direct call: Supervisor
- **Line 52:** defaultdict - Direct call: defaultdict
- **Line 127:** defaultdict - Direct call: defaultdict
- **Line 240:** EnhancedPredictionService - Direct call: EnhancedPredictionService
- **Line 345:** EnhancedExecutionManager - Direct call: EnhancedExecutionManager
- **Line 34:** func - Direct call: func
- **Line 68:** error - Direct call: error
- **Line 93:** error - Direct call: error
- **Line 147:** invalid - Direct call: invalid
- **Line 157:** failed - Direct call: failed
- **Line 173:** error - Direct call: error
- **Line 179:** invalid - Direct call: invalid
- **Line 182:** invalid - Direct call: invalid
- **Line 185:** invalid - Direct call: invalid
- **Line 188:** invalid - Direct call: invalid
- **Line 193:** error - Direct call: error
- **Line 203:** initialization_error - Direct call: initialization_error
- **Line 212:** error - Direct call: error
- **Line 222:** error - Direct call: error
- **Line 233:** error - Direct call: error
- **Line 261:** error - Direct call: error
- **Line 267:** error - Direct call: error
- **Line 270:** error - Direct call: error
- **Line 284:** error - Direct call: error
- **Line 290:** error - Direct call: error
- **Line 293:** error - Direct call: error
- **Line 311:** error - Direct call: error
- **Line 332:** error - Direct call: error
- **Line 364:** error - Direct call: error
- **Line 385:** error - Direct call: error
- **Line 447:** error - Direct call: error
- **Line 474:** error - Direct call: error
- **Line 495:** error - Direct call: error
- **Line 522:** error - Direct call: error
- **Line 567:** error - Direct call: error
- **Line 598:** error - Direct call: error
- **Line 633:** error - Direct call: error
- **Line 669:** error - Direct call: error
- **Line 730:** error - Direct call: error
- **Line 757:** error - Direct call: error
- **Line 776:** error - Direct call: error
- **Line 802:** error - Direct call: error
- **Line 823:** error - Direct call: error
- **Line 854:** error - Direct call: error
- **Line 874:** error - Direct call: error
- **Line 894:** error - Direct call: error
- **Line 905:** failed - Direct call: failed
- **Line 916:** failed - Direct call: failed
- **Line 927:** failed - Direct call: failed
- **Line 938:** failed - Direct call: failed
- **Line 949:** failed - Direct call: failed
- **Line 960:** failed - Direct call: failed
- **Line 971:** failed - Direct call: failed
- **Line 982:** error - Direct call: error
- **Line 995:** error - Direct call: error
- **Line 1004:** error - Direct call: error
- **Line 1055:** error - Direct call: error
- **Line 662:** failed - Direct call: failed

---

## src/tactician/async_order_executor.py

**Undefined function calls: 29**

- **Line 77:** field - Direct call: field
- **Line 97:** field - Direct call: field
- **Line 98:** field - Direct call: field
- **Line 149:** handles_errors - Direct call: handles_errors
- **Line 207:** handles_errors - Direct call: handles_errors
- **Line 165:** EnhancedOrderManager - Direct call: EnhancedOrderManager
- **Line 318:** OrderRequest - Direct call: OrderRequest
- **Line 223:** uuid4 - Direct call: uuid4
- **Line 388:** OrderRequest - Direct call: OrderRequest
- **Line 485:** OrderRequest - Direct call: OrderRequest
- **Line 173:** invalid - Direct call: invalid
- **Line 181:** failed - Direct call: failed
- **Line 194:** invalid - Direct call: invalid
- **Line 198:** invalid - Direct call: invalid
- **Line 204:** failed - Direct call: failed
- **Line 295:** failed - Direct call: failed
- **Line 351:** failed - Direct call: failed
- **Line 425:** failed - Direct call: failed
- **Line 449:** failed - Direct call: failed
- **Line 523:** failed - Direct call: failed
- **Line 560:** failed - Direct call: failed
- **Line 610:** failed - Direct call: failed
- **Line 626:** missing - Direct call: missing
- **Line 644:** failed - Direct call: failed
- **Line 666:** failed - Direct call: failed
- **Line 324:** uuid4 - Direct call: uuid4
- **Line 397:** uuid4 - Direct call: uuid4
- **Line 496:** uuid4 - Direct call: uuid4
- **Line 263:** invalid - Direct call: invalid

---

## src/tactician/dynamic_barrier_calculator.py

**Undefined function calls: 5**

- **Line 115:** handles_errors - Direct call: handles_errors
- **Line 116:** traced - Direct call: traced
- **Line 24:** get_logger - Direct call: get_logger
- **Line 39:** Path - Direct call: Path
- **Line 58:** Path - Direct call: Path

---

## src/tactician/fully_migrated_tactician.py

**Undefined function calls: 2**

- **Line 16:** func - Direct call: func
- **Line 61:** EnhancedScenarioBasedPredictor - Direct call: EnhancedScenarioBasedPredictor

---

## src/tactician/leverage_sizer.py

**Undefined function calls: 2**

- **Line 20:** _handles_errors - Direct call: _handles_errors
- **Line 79:** LinearConfidenceScaler - Direct call: LinearConfidenceScaler

---

## src/tactician/ml_tactics_manager.py

**Undefined function calls: 67**

- **Line 1868:** core_handles_errors - Direct call: core_handles_errors
- **Line 160:** core_handles_errors - Direct call: core_handles_errors
- **Line 195:** core_handles_errors - Direct call: core_handles_errors
- **Line 323:** core_handles_errors - Direct call: core_handles_errors
- **Line 359:** core_handles_errors - Direct call: core_handles_errors
- **Line 433:** core_handles_errors - Direct call: core_handles_errors
- **Line 457:** core_handles_errors - Direct call: core_handles_errors
- **Line 541:** validates - Direct call: validates
- **Line 542:** core_handles_errors - Direct call: core_handles_errors
- **Line 574:** core_handles_errors - Direct call: core_handles_errors
- **Line 627:** core_handles_errors - Direct call: core_handles_errors
- **Line 680:** core_handles_errors - Direct call: core_handles_errors
- **Line 723:** core_handles_errors - Direct call: core_handles_errors
- **Line 769:** core_handles_errors - Direct call: core_handles_errors
- **Line 815:** core_handles_errors - Direct call: core_handles_errors
- **Line 858:** core_handles_errors - Direct call: core_handles_errors
- **Line 906:** core_handles_errors - Direct call: core_handles_errors
- **Line 944:** core_handles_errors - Direct call: core_handles_errors
- **Line 1028:** core_handles_errors - Direct call: core_handles_errors
- **Line 1039:** core_handles_errors - Direct call: core_handles_errors
- **Line 1050:** core_handles_errors - Direct call: core_handles_errors
- **Line 1790:** core_handles_errors - Direct call: core_handles_errors
- **Line 374:** VersionManager - Direct call: VersionManager
- **Line 402:** _scan_and_load - Direct call: _scan_and_load
- **Line 403:** _scan_and_load - Direct call: _scan_and_load
- **Line 1887:** failed - Direct call: failed
- **Line 181:** invalid - Direct call: invalid
- **Line 192:** failed - Direct call: failed
- **Line 205:** invalid - Direct call: invalid
- **Line 209:** invalid - Direct call: invalid
- **Line 213:** invalid - Direct call: invalid
- **Line 217:** invalid - Direct call: invalid
- **Line 240:** invalid - Direct call: invalid
- **Line 251:** failed - Direct call: failed
- **Line 356:** failed - Direct call: failed
- **Line 382:** Path - Direct call: Path
- **Line 430:** failed - Direct call: failed
- **Line 454:** failed - Direct call: failed
- **Line 489:** warning - Direct call: warning
- **Line 538:** failed - Direct call: failed
- **Line 565:** invalid - Direct call: invalid
- **Line 571:** failed - Direct call: failed
- **Line 624:** failed - Direct call: failed
- **Line 720:** failed - Direct call: failed
- **Line 766:** failed - Direct call: failed
- **Line 812:** failed - Direct call: failed
- **Line 855:** failed - Direct call: failed
- **Line 941:** failed - Direct call: failed
- **Line 979:** failed - Direct call: failed
- **Line 1037:** failed - Direct call: failed
- **Line 1120:** failed - Direct call: failed
- **Line 1174:** failed - Direct call: failed
- **Line 1232:** failed - Direct call: failed
- **Line 1302:** failed - Direct call: failed
- **Line 1351:** failed - Direct call: failed
- **Line 1373:** failed - Direct call: failed
- **Line 1420:** failed - Direct call: failed
- **Line 1439:** failed - Direct call: failed
- **Line 1483:** failed - Direct call: failed
- **Line 1551:** failed - Direct call: failed
- **Line 1857:** failed - Direct call: failed
- **Line 223:** invalid - Direct call: invalid
- **Line 229:** invalid - Direct call: invalid
- **Line 234:** invalid - Direct call: invalid
- **Line 245:** invalid - Direct call: invalid
- **Line 393:** ModelSerializer - Direct call: ModelSerializer
- **Line 397:** ModelSerializer - Direct call: ModelSerializer

---

## src/tactician/position_closing.py

**Undefined function calls: 16**

- **Line 67:** handles_errors - Direct call: handles_errors
- **Line 171:** handles_errors - Direct call: handles_errors
- **Line 285:** handles_errors - Direct call: handles_errors
- **Line 84:** invalid - Direct call: invalid
- **Line 92:** failed - Direct call: failed
- **Line 105:** invalid - Direct call: invalid
- **Line 110:** invalid - Direct call: invalid
- **Line 115:** invalid - Direct call: invalid
- **Line 121:** failed - Direct call: failed
- **Line 216:** failed - Direct call: failed
- **Line 257:** failed - Direct call: failed
- **Line 282:** failed - Direct call: failed
- **Line 330:** failed - Direct call: failed
- **Line 359:** failed - Direct call: failed
- **Line 422:** failed - Direct call: failed
- **Line 442:** failed - Direct call: failed

---

## src/tactician/position_sizer.py

**Undefined function calls: 16**

- **Line 23:** _handles_errors - Direct call: _handles_errors
- **Line 149:** validate_data_quality - Direct call: validate_data_quality
- **Line 70:** LinearConfidenceScaler - Direct call: LinearConfidenceScaler
- **Line 290:** calculate_correct_kelly_position_size - Direct call: calculate_correct_kelly_position_size
- **Line 460:** normalize_dual_confidence - Direct call: normalize_dual_confidence
- **Line 182:** initialization_error - Direct call: initialization_error
- **Line 641:** error - Direct call: error
- **Line 110:** error - Direct call: error
- **Line 116:** error - Direct call: error
- **Line 280:** error - Direct call: error
- **Line 399:** error - Direct call: error
- **Line 472:** error - Direct call: error
- **Line 520:** error - Direct call: error
- **Line 582:** error - Direct call: error
- **Line 602:** error - Direct call: error
- **Line 100:** missing - Direct call: missing

---

## src/tactician/sr_detection_optimization.py

**Undefined function calls: 24**

- **Line 44:** field - Direct call: field
- **Line 45:** field - Direct call: field
- **Line 46:** field - Direct call: field
- **Line 47:** field - Direct call: field
- **Line 48:** field - Direct call: field
- **Line 188:** sr_error_handler - Direct call: sr_error_handler
- **Line 267:** handles_errors - Direct call: handles_errors
- **Line 268:** sr_error_handler - Direct call: sr_error_handler
- **Line 93:** get_sr_config - Direct call: get_sr_config
- **Line 201:** SRBreakoutPredictor - Direct call: SRBreakoutPredictor
- **Line 292:** validate_sr_data - Direct call: validate_sr_data
- **Line 351:** validate_sr_data - Direct call: validate_sr_data
- **Line 223:** SRRegimeOptimizer - Direct call: SRRegimeOptimizer
- **Line 228:** SRMLEnhancer - Direct call: SRMLEnhancer
- **Line 233:** SRComputationalOptimizer - Direct call: SRComputationalOptimizer
- **Line 238:** EnhancedSRBreakoutPredictor - Direct call: EnhancedSRBreakoutPredictor
- **Line 295:** SROptimizationError - Direct call: SROptimizationError
- **Line 298:** SROptimizationError - Direct call: SROptimizationError
- **Line 353:** SROptimizationError - Direct call: SROptimizationError
- **Line 703:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 760:** hash - Direct call: hash
- **Line 760:** hash - Direct call: hash
- **Line 948:** setup_sr_parameter_optimizer - Direct call: setup_sr_parameter_optimizer
- **Line 763:** id - Direct call: id

---

## src/tactician/sr_levels/enhanced_sr_confluence.py

**Undefined function calls: 2**

- **Line 72:** handles_errors - Direct call: handles_errors
- **Line 77:** traced - Direct call: traced

---

## src/tactician/sr_levels/enhanced_sr_detection.py

**Undefined function calls: 4**

- **Line 66:** handles_errors - Direct call: handles_errors
- **Line 71:** traced - Direct call: traced
- **Line 446:** find_peaks - Direct call: find_peaks
- **Line 454:** find_peaks - Direct call: find_peaks

---

## src/tactician/sr_levels/enhanced_sr_optimization.py

**Undefined function calls: 13**

- **Line 33:** field - Direct call: field
- **Line 34:** field - Direct call: field
- **Line 37:** field - Direct call: field
- **Line 47:** field - Direct call: field
- **Line 48:** field - Direct call: field
- **Line 54:** field - Direct call: field
- **Line 88:** handles_errors - Direct call: handles_errors
- **Line 138:** handles_errors - Direct call: handles_errors
- **Line 143:** traced - Direct call: traced
- **Line 66:** EnhancedSRDetector - Direct call: EnhancedSRDetector
- **Line 67:** EnhancedSRValidator - Direct call: EnhancedSRValidator
- **Line 68:** EnhancedSRConfluenceDetector - Direct call: EnhancedSRConfluenceDetector
- **Line 375:** EnhancedSRDetector - Direct call: EnhancedSRDetector

---

## src/tactician/sr_levels/enhanced_sr_validation.py

**Undefined function calls: 4**

- **Line 77:** handles_errors - Direct call: handles_errors
- **Line 82:** traced - Direct call: traced
- **Line 518:** handles_errors - Direct call: handles_errors
- **Line 523:** traced - Direct call: traced

---

## src/tactician/sr_levels/sr_breakout_predictor.py

**Undefined function calls: 13**

- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 125:** handles_errors - Direct call: handles_errors
- **Line 189:** handles_errors - Direct call: handles_errors
- **Line 228:** handles_errors - Direct call: handles_errors
- **Line 483:** handles_errors - Direct call: handles_errors
- **Line 569:** handles_errors - Direct call: handles_errors
- **Line 583:** handles_errors - Direct call: handles_errors
- **Line 80:** SRLevelDetector - Direct call: SRLevelDetector
- **Line 81:** SRMetricsCalculator - Direct call: SRMetricsCalculator
- **Line 82:** SRFeatureExtractor - Direct call: SRFeatureExtractor
- **Line 511:** SRProbabilityCalculator - Direct call: SRProbabilityCalculator
- **Line 86:** SRReportGenerator - Direct call: SRReportGenerator
- **Line 91:** SRAnalyzer - Direct call: SRAnalyzer

---

## src/tactician/sr_levels/sr_breakout_predictor_enhanced.py

**Undefined function calls: 2**

- **Line 126:** sr_error_handler - Direct call: sr_error_handler
- **Line 315:** VectorizedAdvancedFeatureEngineeringRefactored - Direct call: VectorizedAdvancedFeatureEngineeringRefactored

---

## src/tactician/sr_levels/sr_comprehensive_integration.py

**Undefined function calls: 14**

- **Line 125:** EnhancedSRDetector - Direct call: EnhancedSRDetector
- **Line 136:** SRStrengthOptimizer - Direct call: SRStrengthOptimizer
- **Line 147:** SRBreakoutPredictor - Direct call: SRBreakoutPredictor
- **Line 158:** SRContextAwareCalculator - Direct call: SRContextAwareCalculator
- **Line 169:** EnhancedSRConfluenceDetector - Direct call: EnhancedSRConfluenceDetector
- **Line 180:** EnhancedSRValidator - Direct call: EnhancedSRValidator
- **Line 191:** EnhancedSROptimizer - Direct call: EnhancedSROptimizer
- **Line 202:** SRDataIntegration - Direct call: SRDataIntegration
- **Line 213:** SREnsemblePredictor - Direct call: SREnsemblePredictor
- **Line 224:** SRParameterOptimizer - Direct call: SRParameterOptimizer
- **Line 235:** SRPerformanceMonitor - Direct call: SRPerformanceMonitor
- **Line 246:** SRWeightOptimizer - Direct call: SRWeightOptimizer
- **Line 257:** SRLevelsManager - Direct call: SRLevelsManager
- **Line 96:** init_func - Direct call: init_func

---

## src/tactician/sr_levels/sr_computational_optimizer.py

**Undefined function calls: 8**

- **Line 85:** sr_error_handler - Direct call: sr_error_handler
- **Line 541:** lru_cache - Direct call: lru_cache
- **Line 74:** ThreadPoolExecutor - Direct call: ThreadPoolExecutor
- **Line 589:** jit - Direct call: jit
- **Line 613:** jit - Direct call: jit
- **Line 594:** prange - Direct call: prange
- **Line 623:** prange - Direct call: prange
- **Line 80:** ProcessPoolExecutor - Direct call: ProcessPoolExecutor

---

## src/tactician/sr_levels/sr_context_aware_calculator.py

**Undefined function calls: 15**

- **Line 82:** handles_errors - Direct call: handles_errors
- **Line 378:** handles_errors - Direct call: handles_errors
- **Line 383:** traced - Direct call: traced
- **Line 443:** SRProbabilityCalculator - Direct call: SRProbabilityCalculator
- **Line 444:** SRLevelIdentifier - Direct call: SRLevelIdentifier
- **Line 70:** time - Direct call: time
- **Line 70:** time - Direct call: time
- **Line 71:** time - Direct call: time
- **Line 71:** time - Direct call: time
- **Line 72:** time - Direct call: time
- **Line 72:** time - Direct call: time
- **Line 423:** asdict - Direct call: asdict
- **Line 424:** asdict - Direct call: asdict
- **Line 654:** asdict - Direct call: asdict
- **Line 655:** asdict - Direct call: asdict

---

## src/tactician/sr_levels/sr_data_integration.py

**Undefined function calls: 5**

- **Line 17:** Path - Direct call: Path
- **Line 70:** UnifiedDataLoader - Direct call: UnifiedDataLoader
- **Line 249:** timedelta - Direct call: timedelta
- **Line 188:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation
- **Line 296:** Path - Direct call: Path

---

## src/tactician/sr_levels/sr_ensemble_predictor.py

**Undefined function calls: 3**

- **Line 382:** handles_errors - Direct call: handles_errors
- **Line 387:** traced - Direct call: traced
- **Line 55:** SRLevelIdentifier - Direct call: SRLevelIdentifier

---

## src/tactician/sr_levels/sr_levels_manager.py

**Undefined function calls: 3**

- **Line 42:** cls - Direct call: cls
- **Line 80:** Path - Direct call: Path
- **Line 97:** SRBreakoutPredictor - Direct call: SRBreakoutPredictor

---

## src/tactician/sr_levels/sr_ml_enhancer.py

**Undefined function calls: 12**

- **Line 111:** sr_error_handler - Direct call: sr_error_handler
- **Line 96:** StandardScaler - Direct call: StandardScaler
- **Line 1023:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 1703:** Path - Direct call: Path
- **Line 1731:** Path - Direct call: Path
- **Line 228:** VectorizedAdvancedFeatureEngineeringRefactored - Direct call: VectorizedAdvancedFeatureEngineeringRefactored
- **Line 924:** GradientBoostingRegressor - Direct call: GradientBoostingRegressor
- **Line 938:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 957:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 963:** permutation_importance - Direct call: permutation_importance
- **Line 1005:** cross_val_score - Direct call: cross_val_score
- **Line 1055:** VectorizedAdvancedFeatureEngineeringRefactored - Direct call: VectorizedAdvancedFeatureEngineeringRefactored

---

## src/tactician/sr_levels/sr_modules/sr_feature_extractor.py

**Undefined function calls: 1**

- **Line 23:** handles_errors - Direct call: handles_errors

---

## src/tactician/sr_levels/sr_modules/sr_level_detector.py

**Undefined function calls: 2**

- **Line 43:** handles_errors - Direct call: handles_errors
- **Line 330:** DBSCAN - Direct call: DBSCAN

---

## src/tactician/sr_levels/sr_modules/sr_metrics_calculator.py

**Undefined function calls: 1**

- **Line 25:** handles_errors - Direct call: handles_errors

---

## src/tactician/sr_levels/sr_modules/sr_probability_calculator.py

**Undefined function calls: 1**

- **Line 77:** handles_errors - Direct call: handles_errors

---

## src/tactician/sr_levels/sr_parameter_optimizer.py

**Undefined function calls: 4**

- **Line 118:** handles_errors - Direct call: handles_errors
- **Line 123:** traced - Direct call: traced
- **Line 769:** asdict - Direct call: asdict
- **Line 782:** asdict - Direct call: asdict

---

## src/tactician/sr_levels/sr_performance_monitor.py

**Undefined function calls: 13**

- **Line 104:** handles_errors - Direct call: handles_errors
- **Line 138:** handles_errors - Direct call: handles_errors
- **Line 278:** traced - Direct call: traced
- **Line 82:** deque - Direct call: deque
- **Line 83:** deque - Direct call: deque
- **Line 84:** deque - Direct call: deque
- **Line 91:** defaultdict - Direct call: defaultdict
- **Line 98:** timedelta - Direct call: timedelta
- **Line 419:** defaultdict - Direct call: defaultdict
- **Line 442:** defaultdict - Direct call: defaultdict
- **Line 460:** defaultdict - Direct call: defaultdict
- **Line 397:** asdict - Direct call: asdict
- **Line 584:** asdict - Direct call: asdict

---

## src/tactician/sr_levels/sr_regime_optimizer.py

**Undefined function calls: 2**

- **Line 78:** sr_error_handler - Direct call: sr_error_handler
- **Line 100:** SRDataError - Direct call: SRDataError

---

## src/tactician/sr_levels/sr_strength_optimizer.py

**Undefined function calls: 4**

- **Line 164:** handles_errors - Direct call: handles_errors
- **Line 169:** traced - Direct call: traced
- **Line 1160:** asdict - Direct call: asdict
- **Line 1178:** asdict - Direct call: asdict

---

## src/tactician/sr_levels/sr_weight_optimizer.py

**Undefined function calls: 23**

- **Line 81:** handles_errors - Direct call: handles_errors
- **Line 145:** handles_errors - Direct call: handles_errors
- **Line 98:** ensure_optimized_sr_config - Direct call: ensure_optimized_sr_config
- **Line 99:** setup_sr_breakout_predictor - Direct call: setup_sr_breakout_predictor
- **Line 107:** invalid - Direct call: invalid
- **Line 115:** failed - Direct call: failed
- **Line 128:** invalid - Direct call: invalid
- **Line 132:** invalid - Direct call: invalid
- **Line 136:** invalid - Direct call: invalid
- **Line 142:** failed - Direct call: failed
- **Line 204:** failed - Direct call: failed
- **Line 225:** failed - Direct call: failed
- **Line 315:** failed - Direct call: failed
- **Line 369:** failed - Direct call: failed
- **Line 451:** failed - Direct call: failed
- **Line 494:** failed - Direct call: failed
- **Line 523:** failed - Direct call: failed
- **Line 552:** failed - Direct call: failed
- **Line 573:** failed - Direct call: failed
- **Line 588:** warning - Direct call: warning
- **Line 616:** failed - Direct call: failed
- **Line 661:** failed - Direct call: failed
- **Line 679:** failed - Direct call: failed

---

## src/tactician/step17_optimized_tactician.py

**Undefined function calls: 2**

- **Line 31:** func - Direct call: func
- **Line 254:** ComprehensiveEnhancedScenarioPredictor - Direct call: ComprehensiveEnhancedScenarioPredictor

---

## src/tactician/tactician.py

**Undefined function calls: 22**

- **Line 16:** _handles_errors - Direct call: _handles_errors
- **Line 194:** TacticsOrchestrator - Direct call: TacticsOrchestrator
- **Line 199:** PositionSizer - Direct call: PositionSizer
- **Line 204:** LeverageSizer - Direct call: LeverageSizer
- **Line 209:** PositionDivisionStrategy - Direct call: PositionDivisionStrategy
- **Line 133:** invalid - Direct call: invalid
- **Line 141:** failed - Direct call: failed
- **Line 169:** invalid - Direct call: invalid
- **Line 173:** invalid - Direct call: invalid
- **Line 179:** failed - Direct call: failed
- **Line 216:** EnhancedScenarioBasedPredictor - Direct call: EnhancedScenarioBasedPredictor
- **Line 228:** failed - Direct call: failed
- **Line 266:** failed - Direct call: failed
- **Line 271:** failed - Direct call: failed
- **Line 299:** invalid - Direct call: invalid
- **Line 305:** failed - Direct call: failed
- **Line 341:** failed - Direct call: failed
- **Line 372:** failed - Direct call: failed
- **Line 1124:** failed - Direct call: failed
- **Line 1133:** failed - Direct call: failed
- **Line 1165:** failed - Direct call: failed
- **Line 294:** missing - Direct call: missing

---

## src/tactician/tactics_orchestrator.py

**Undefined function calls: 57**

- **Line 54:** handles_errors - Direct call: handles_errors
- **Line 133:** handles_errors - Direct call: handles_errors
- **Line 377:** handles_errors - Direct call: handles_errors
- **Line 446:** handles_errors - Direct call: handles_errors
- **Line 502:** handles_errors - Direct call: handles_errors
- **Line 522:** handles_errors - Direct call: handles_errors
- **Line 100:** PositionSizer - Direct call: PositionSizer
- **Line 102:** LeverageSizer - Direct call: LeverageSizer
- **Line 107:** SRBreakoutPredictor - Direct call: SRBreakoutPredictor
- **Line 109:** MLTacticsManager - Direct call: MLTacticsManager
- **Line 389:** PositionMonitor - Direct call: PositionMonitor
- **Line 391:** PositionCloser - Direct call: PositionCloser
- **Line 393:** EnhancedOrderManager - Direct call: EnhancedOrderManager
- **Line 395:** PositionDivisionStrategy - Direct call: PositionDivisionStrategy
- **Line 66:** invalid - Direct call: invalid
- **Line 71:** failed - Direct call: failed
- **Line 112:** failed - Direct call: failed
- **Line 123:** invalid - Direct call: invalid
- **Line 126:** invalid - Direct call: invalid
- **Line 130:** failed - Direct call: failed
- **Line 158:** failed - Direct call: failed
- **Line 179:** failed - Direct call: failed
- **Line 200:** failed - Direct call: failed
- **Line 221:** failed - Direct call: failed
- **Line 241:** failed - Direct call: failed
- **Line 299:** failed - Direct call: failed
- **Line 330:** failed - Direct call: failed
- **Line 349:** failed - Direct call: failed
- **Line 398:** invalid - Direct call: invalid
- **Line 403:** failed - Direct call: failed
- **Line 439:** invalid - Direct call: invalid
- **Line 443:** failed - Direct call: failed
- **Line 463:** invalid - Direct call: invalid
- **Line 466:** invalid - Direct call: invalid
- **Line 499:** failed - Direct call: failed
- **Line 512:** warning - Direct call: warning
- **Line 519:** failed - Direct call: failed
- **Line 532:** warning - Direct call: warning
- **Line 542:** failed - Direct call: failed
- **Line 558:** failed - Direct call: failed
- **Line 572:** failed - Direct call: failed
- **Line 594:** failed - Direct call: failed
- **Line 606:** failed - Direct call: failed
- **Line 619:** failed - Direct call: failed
- **Line 640:** failed - Direct call: failed
- **Line 656:** failed - Direct call: failed
- **Line 694:** failed - Direct call: failed
- **Line 716:** failed - Direct call: failed
- **Line 742:** failed - Direct call: failed
- **Line 768:** failed - Direct call: failed
- **Line 814:** failed - Direct call: failed
- **Line 833:** failed - Direct call: failed
- **Line 842:** failed - Direct call: failed
- **Line 864:** failed - Direct call: failed
- **Line 881:** failed - Direct call: failed
- **Line 907:** failed - Direct call: failed
- **Line 929:** failed - Direct call: failed

---

## src/tasks.py

**Undefined function calls: 7**

- **Line 7:** Celery - Direct call: Celery
- **Line 17:** AresPipeline - Direct call: AresPipeline
- **Line 44:** crontab - Direct call: crontab
- **Line 29:** SQLiteManager - Direct call: SQLiteManager
- **Line 31:** TrainingManager - Direct call: TrainingManager
- **Line 32:** get_environment_settings - Direct call: get_environment_settings
- **Line 41:** run_training - Direct call: run_training

---

## src/tracking/trade_tracker.py

**Undefined function calls: 9**

- **Line 155:** handles_errors - Direct call: handles_errors
- **Line 183:** handles_errors - Direct call: handles_errors
- **Line 197:** handles_errors - Direct call: handles_errors
- **Line 207:** handles_errors - Direct call: handles_errors
- **Line 180:** failed - Direct call: failed
- **Line 221:** missing - Direct call: missing
- **Line 232:** failed - Direct call: failed
- **Line 363:** asdict - Direct call: asdict
- **Line 369:** asdict - Direct call: asdict

---

## src/trading/live_wavelet_analyzer.py

**Undefined function calls: 9**

- **Line 86:** handles_errors - Direct call: handles_errors
- **Line 162:** handles_errors - Direct call: handles_errors
- **Line 78:** deque - Direct call: deque
- **Line 79:** deque - Direct call: deque
- **Line 99:** deque - Direct call: deque
- **Line 100:** deque - Direct call: deque
- **Line 117:** warning - Direct call: warning
- **Line 121:** warning - Direct call: warning
- **Line 111:** initialization_error - Direct call: initialization_error

---

## src/trading/live_wavelet_demo.py

**Undefined function calls: 2**

- **Line 23:** LiveWaveletIntegration - Direct call: LiveWaveletIntegration
- **Line 198:** main - Direct call: main

---

## src/trading/live_wavelet_integration.py

**Undefined function calls: 3**

- **Line 46:** handles_errors - Direct call: handles_errors
- **Line 84:** handles_errors - Direct call: handles_errors
- **Line 58:** LiveWaveletAnalyzer - Direct call: LiveWaveletAnalyzer

---

## src/trading/sr_trading_intelligence.py

**Undefined function calls: 3**

- **Line 286:** Path - Direct call: Path
- **Line 302:** Path - Direct call: Path
- **Line 64:** create_sr_levels_manager - Direct call: create_sr_levels_manager

---

## src/training/adaptive_optimizer.py

**Undefined function calls: 2**

- **Line 34:** handles_errors - Direct call: handles_errors
- **Line 105:** handles_errors - Direct call: handles_errors

---

## src/training/advanced_neural_models.py

**Undefined function calls: 7**

- **Line 220:** check_X_y - Direct call: check_X_y
- **Line 221:** unique_labels - Direct call: unique_labels
- **Line 255:** check_is_fitted - Direct call: check_is_fitted
- **Line 256:** check_array - Direct call: check_array
- **Line 267:** check_is_fitted - Direct call: check_is_fitted
- **Line 268:** check_array - Direct call: check_array
- **Line 238:** criterion - Direct call: criterion

---

## src/training/base_step.py

**Undefined function calls: 1**

- **Line 72:** execute_logic - Direct call: execute_logic

---

## src/training/bayesian_optimizer.py

**Undefined function calls: 5**

- **Line 47:** handles_errors - Direct call: handles_errors
- **Line 261:** handles_errors - Direct call: handles_errors
- **Line 286:** handles_errors - Direct call: handles_errors
- **Line 315:** handles_errors - Direct call: handles_errors
- **Line 403:** handles_errors - Direct call: handles_errors

---

## src/training/calibration_manager.py

**Undefined function calls: 31**

- **Line 576:** handles_errors - Direct call: handles_errors
- **Line 52:** handles_errors - Direct call: handles_errors
- **Line 86:** handles_errors - Direct call: handles_errors
- **Line 116:** handles_errors - Direct call: handles_errors
- **Line 147:** handles_errors - Direct call: handles_errors
- **Line 215:** handles_errors - Direct call: handles_errors
- **Line 259:** handles_errors - Direct call: handles_errors
- **Line 303:** handles_errors - Direct call: handles_errors
- **Line 347:** handles_errors - Direct call: handles_errors
- **Line 418:** handles_errors - Direct call: handles_errors
- **Line 449:** handles_errors - Direct call: handles_errors
- **Line 480:** handles_errors - Direct call: handles_errors
- **Line 511:** handles_errors - Direct call: handles_errors
- **Line 562:** handles_errors - Direct call: handles_errors
- **Line 126:** MLConfidencePredictor - Direct call: MLConfidencePredictor
- **Line 73:** invalid - Direct call: invalid
- **Line 83:** failed - Direct call: failed
- **Line 107:** error - Direct call: error
- **Line 113:** failed - Direct call: failed
- **Line 211:** failed - Direct call: failed
- **Line 238:** error - Direct call: error
- **Line 243:** error - Direct call: error
- **Line 250:** error - Direct call: error
- **Line 256:** failed - Direct call: failed
- **Line 300:** failed - Direct call: failed
- **Line 344:** failed - Direct call: failed
- **Line 446:** failed - Direct call: failed
- **Line 477:** failed - Direct call: failed
- **Line 508:** failed - Direct call: failed
- **Line 536:** failed - Direct call: failed
- **Line 574:** failed - Direct call: failed

---

## src/training/cleanup_duplicates.py

**Undefined function calls: 2**

- **Line 76:** Path - Direct call: Path
- **Line 104:** Path - Direct call: Path

---

## src/training/comprehensive_feature_optimizer.py

**Undefined function calls: 1**

- **Line 616:** Path - Direct call: Path

---

## src/training/comprehensive_pipeline_executor.py

**Undefined function calls: 8**

- **Line 35:** Steps1To7ComprehensiveExecutor - Direct call: Steps1To7ComprehensiveExecutor
- **Line 36:** DataQualityMonitor - Direct call: DataQualityMonitor
- **Line 206:** main - Direct call: main
- **Line 13:** Path - Direct call: Path
- **Line 104:** log_step_metrics - Direct call: log_step_metrics
- **Line 117:** log_step_report - Direct call: log_step_report
- **Line 136:** log_step_report - Direct call: log_step_report
- **Line 138:** log_step_metrics - Direct call: log_step_metrics

---

## src/training/comprehensive_sr_training_pipeline.py

**Undefined function calls: 4**

- **Line 59:** handles_errors - Direct call: handles_errors
- **Line 41:** MultiOutputModelTrainer - Direct call: MultiOutputModelTrainer
- **Line 41:** MultiOutputModelConfig - Direct call: MultiOutputModelConfig
- **Line 345:** Path - Direct call: Path

---

## src/training/core/training_manager.py

**Undefined function calls: 1**

- **Line 34:** SimplifiedTrainingManager - Direct call: SimplifiedTrainingManager

---

## src/training/data_access_utils.py

**Undefined function calls: 1**

- **Line 32:** UnifiedDataManager - Direct call: UnifiedDataManager

---

## src/training/data_manager.py

**Undefined function calls: 2**

- **Line 160:** timedelta - Direct call: timedelta
- **Line 142:** error - Direct call: error

---

## src/training/data_quality_monitor.py

**Undefined function calls: 4**

- **Line 572:** main - Direct call: main
- **Line 10:** Path - Direct call: Path
- **Line 525:** asdict - Direct call: asdict
- **Line 528:** log_step_metrics - Direct call: log_step_metrics

---

## src/training/data_sharing_manager.py

**Undefined function calls: 11**

- **Line 162:** validates - Direct call: validates
- **Line 173:** secure_data_processing - Direct call: secure_data_processing
- **Line 179:** prevent_data_leakage - Direct call: prevent_data_leakage
- **Line 184:** log_execution_time - Direct call: log_execution_time
- **Line 191:** cached - Direct call: cached
- **Line 197:** log_call - Direct call: log_call
- **Line 203:** circuit_breaker - Direct call: circuit_breaker
- **Line 209:** validates - Direct call: validates
- **Line 218:** quality_gate - Direct call: quality_gate
- **Line 223:** handles_errors - Direct call: handles_errors
- **Line 49:** get_unified_data_loader - Direct call: get_unified_data_loader

---

## src/training/di_training_manager.py

**Undefined function calls: 20**

- **Line 188:** handles_errors - Direct call: handles_errors
- **Line 110:** TrainingPipeline - Direct call: TrainingPipeline
- **Line 139:** __import__ - Direct call: __import__
- **Line 211:** warning - Direct call: warning
- **Line 90:** failed - Direct call: failed
- **Line 152:** step_class - Direct call: step_class
- **Line 166:** invalid - Direct call: invalid
- **Line 171:** invalid - Direct call: invalid
- **Line 185:** failed - Direct call: failed
- **Line 252:** failed - Direct call: failed
- **Line 261:** initialization_error - Direct call: initialization_error
- **Line 304:** failed - Direct call: failed
- **Line 343:** failed - Direct call: failed
- **Line 380:** failed - Direct call: failed
- **Line 415:** failed - Direct call: failed
- **Line 179:** missing - Direct call: missing
- **Line 283:** warning - Direct call: warning
- **Line 294:** failed - Direct call: failed
- **Line 335:** failed - Direct call: failed
- **Line 372:** failed - Direct call: failed

---

## src/training/enhanced_coarse_optimizer.py

**Undefined function calls: 33**

- **Line 224:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 61:** cpu_count - Direct call: cpu_count
- **Line 120:** ProcessPoolExecutor - Direct call: ProcessPoolExecutor
- **Line 125:** as_completed - Direct call: as_completed
- **Line 210:** failed - Direct call: failed
- **Line 465:** model_class - Direct call: model_class
- **Line 188:** CatBoostClassifier - Direct call: CatBoostClassifier
- **Line 199:** TreeExplainer - Direct call: TreeExplainer
- **Line 468:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 74:** failed - Direct call: failed
- **Line 96:** failed - Direct call: failed
- **Line 155:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 178:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 219:** failed - Direct call: failed
- **Line 233:** accuracy_score - Direct call: accuracy_score
- **Line 234:** precision_score - Direct call: precision_score
- **Line 235:** recall_score - Direct call: recall_score
- **Line 236:** f1_score - Direct call: f1_score
- **Line 268:** error - Direct call: error
- **Line 358:** error - Direct call: error
- **Line 368:** error - Direct call: error
- **Line 378:** error - Direct call: error
- **Line 388:** error - Direct call: error
- **Line 422:** error - Direct call: error
- **Line 467:** failed - Direct call: failed
- **Line 503:** SuccessiveHalvingPruner - Direct call: SuccessiveHalvingPruner
- **Line 547:** error - Direct call: error
- **Line 558:** error - Direct call: error
- **Line 581:** failed - Direct call: failed
- **Line 208:** failed - Direct call: failed
- **Line 240:** failed - Direct call: failed
- **Line 501:** failed - Direct call: failed
- **Line 540:** warning - Direct call: warning

---

## src/training/enhanced_dynamic_feature_selection.py

**Undefined function calls: 10**

- **Line 65:** handles_errors - Direct call: handles_errors
- **Line 245:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 301:** linkage - Direct call: linkage
- **Line 308:** fcluster - Direct call: fcluster
- **Line 340:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 346:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 355:** f_classif - Direct call: f_classif
- **Line 301:** squareform - Direct call: squareform
- **Line 567:** RFE - Direct call: RFE
- **Line 635:** fcluster - Direct call: fcluster

---

## src/training/enhanced_feature_engineering_optimizer.py

**Undefined function calls: 11**

- **Line 48:** handles_errors - Direct call: handles_errors
- **Line 166:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 243:** FeatureEngineeringOptimizer - Direct call: FeatureEngineeringOptimizer
- **Line 283:** Path - Direct call: Path
- **Line 444:** Path - Direct call: Path
- **Line 314:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 329:** cross_val_score - Direct call: cross_val_score
- **Line 365:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 329:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 107:** TPESampler - Direct call: TPESampler
- **Line 107:** MedianPruner - Direct call: MedianPruner

---

## src/training/enhanced_lm_config.py

**Undefined function calls: 43**

- **Line 19:** Field - Direct call: Field
- **Line 20:** Field - Direct call: Field
- **Line 21:** Field - Direct call: Field
- **Line 22:** Field - Direct call: Field
- **Line 23:** Field - Direct call: Field
- **Line 24:** Field - Direct call: Field
- **Line 25:** Field - Direct call: Field
- **Line 26:** Field - Direct call: Field
- **Line 28:** validator - Direct call: validator
- **Line 39:** Field - Direct call: Field
- **Line 40:** Field - Direct call: Field
- **Line 41:** Field - Direct call: Field
- **Line 42:** Field - Direct call: Field
- **Line 43:** Field - Direct call: Field
- **Line 45:** validator - Direct call: validator
- **Line 57:** Field - Direct call: Field
- **Line 58:** Field - Direct call: Field
- **Line 59:** Field - Direct call: Field
- **Line 60:** Field - Direct call: Field
- **Line 61:** Field - Direct call: Field
- **Line 62:** Field - Direct call: Field
- **Line 63:** Field - Direct call: Field
- **Line 65:** validator - Direct call: validator
- **Line 74:** Field - Direct call: Field
- **Line 75:** Field - Direct call: Field
- **Line 76:** Field - Direct call: Field
- **Line 77:** Field - Direct call: Field
- **Line 81:** Field - Direct call: Field
- **Line 82:** Field - Direct call: Field
- **Line 83:** Field - Direct call: Field
- **Line 84:** Field - Direct call: Field
- **Line 85:** Field - Direct call: Field
- **Line 89:** Field - Direct call: Field
- **Line 90:** Field - Direct call: Field
- **Line 91:** Field - Direct call: Field
- **Line 92:** Field - Direct call: Field
- **Line 93:** Field - Direct call: Field
- **Line 94:** Field - Direct call: Field
- **Line 95:** Field - Direct call: Field
- **Line 96:** Field - Direct call: Field
- **Line 97:** Field - Direct call: Field
- **Line 98:** Field - Direct call: Field
- **Line 113:** cls - Direct call: cls

---

## src/training/enhanced_matrix_gpu_integration.py

**Undefined function calls: 23**

- **Line 58:** secure_data_processing - Direct call: secure_data_processing
- **Line 59:** prevent_data_leakage - Direct call: prevent_data_leakage
- **Line 60:** log_execution_time - Direct call: log_execution_time
- **Line 61:** cached - Direct call: cached
- **Line 62:** log_call - Direct call: log_call
- **Line 63:** circuit_breaker - Direct call: circuit_breaker
- **Line 64:** validates - Direct call: validates
- **Line 65:** quality_gate - Direct call: quality_gate
- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 209:** secure_data_processing - Direct call: secure_data_processing
- **Line 210:** cached - Direct call: cached
- **Line 211:** log_call - Direct call: log_call
- **Line 212:** quality_gate - Direct call: quality_gate
- **Line 213:** handles_errors - Direct call: handles_errors
- **Line 313:** secure_data_processing - Direct call: secure_data_processing
- **Line 314:** cached - Direct call: cached
- **Line 315:** log_call - Direct call: log_call
- **Line 316:** quality_gate - Direct call: quality_gate
- **Line 317:** handles_errors - Direct call: handles_errors
- **Line 49:** EnhancedMatrixOperations - Direct call: EnhancedMatrixOperations
- **Line 52:** M1GPUAcceleration - Direct call: M1GPUAcceleration
- **Line 570:** demonstrate_gpu_integration - Direct call: demonstrate_gpu_integration
- **Line 407:** LinearRegression - Direct call: LinearRegression

---

## src/training/enhanced_matrix_operations.py

**Undefined function calls: 71**

- **Line 115:** secure_data_processing - Direct call: secure_data_processing
- **Line 116:** prevent_data_leakage - Direct call: prevent_data_leakage
- **Line 117:** log_execution_time - Direct call: log_execution_time
- **Line 118:** cached - Direct call: cached
- **Line 119:** log_call - Direct call: log_call
- **Line 120:** circuit_breaker - Direct call: circuit_breaker
- **Line 121:** validates - Direct call: validates
- **Line 122:** quality_gate - Direct call: quality_gate
- **Line 123:** handles_errors - Direct call: handles_errors
- **Line 205:** secure_data_processing - Direct call: secure_data_processing
- **Line 206:** cached - Direct call: cached
- **Line 207:** log_call - Direct call: log_call
- **Line 208:** quality_gate - Direct call: quality_gate
- **Line 209:** handles_errors - Direct call: handles_errors
- **Line 285:** secure_data_processing - Direct call: secure_data_processing
- **Line 286:** cached - Direct call: cached
- **Line 287:** log_call - Direct call: log_call
- **Line 288:** quality_gate - Direct call: quality_gate
- **Line 289:** handles_errors - Direct call: handles_errors
- **Line 357:** secure_data_processing - Direct call: secure_data_processing
- **Line 358:** log_execution_time - Direct call: log_execution_time
- **Line 359:** cached - Direct call: cached
- **Line 360:** log_call - Direct call: log_call
- **Line 361:** quality_gate - Direct call: quality_gate
- **Line 362:** handles_errors - Direct call: handles_errors
- **Line 460:** secure_data_processing - Direct call: secure_data_processing
- **Line 461:** cached - Direct call: cached
- **Line 462:** log_call - Direct call: log_call
- **Line 463:** quality_gate - Direct call: quality_gate
- **Line 464:** handles_errors - Direct call: handles_errors
- **Line 528:** secure_data_processing - Direct call: secure_data_processing
- **Line 529:** cached - Direct call: cached
- **Line 530:** log_call - Direct call: log_call
- **Line 531:** quality_gate - Direct call: quality_gate
- **Line 532:** handles_errors - Direct call: handles_errors
- **Line 639:** secure_data_processing - Direct call: secure_data_processing
- **Line 640:** log_execution_time - Direct call: log_execution_time
- **Line 641:** cached - Direct call: cached
- **Line 642:** log_call - Direct call: log_call
- **Line 643:** quality_gate - Direct call: quality_gate
- **Line 644:** handles_errors - Direct call: handles_errors
- **Line 723:** secure_data_processing - Direct call: secure_data_processing
- **Line 724:** cached - Direct call: cached
- **Line 725:** log_call - Direct call: log_call
- **Line 726:** quality_gate - Direct call: quality_gate
- **Line 727:** handles_errors - Direct call: handles_errors
- **Line 841:** secure_data_processing - Direct call: secure_data_processing
- **Line 842:** cached - Direct call: cached
- **Line 843:** log_call - Direct call: log_call
- **Line 844:** quality_gate - Direct call: quality_gate
- **Line 845:** handles_errors - Direct call: handles_errors
- **Line 980:** handles_errors - Direct call: handles_errors
- **Line 1204:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 147:** StandardScaler - Direct call: StandardScaler
- **Line 554:** StandardScaler - Direct call: StandardScaler
- **Line 1082:** AutoencoderFeatureGenerator - Direct call: AutoencoderFeatureGenerator
- **Line 1311:** RegularizationManager - Direct call: RegularizationManager
- **Line 385:** FastICA - Direct call: FastICA
- **Line 408:** FactorAnalysis - Direct call: FactorAnalysis
- **Line 428:** KernelPCA - Direct call: KernelPCA
- **Line 490:** IterativeImputer - Direct call: IterativeImputer
- **Line 559:** SpectralClustering - Direct call: SpectralClustering
- **Line 575:** KMeans - Direct call: KMeans
- **Line 580:** euclidean_distances - Direct call: euclidean_distances
- **Line 604:** DBSCAN - Direct call: DBSCAN
- **Line 669:** Lasso - Direct call: Lasso
- **Line 689:** Ridge - Direct call: Ridge
- **Line 750:** PolynomialFeatures - Direct call: PolynomialFeatures
- **Line 1368:** RFE - Direct call: RFE
- **Line 1529:** cross_val_score - Direct call: cross_val_score
- **Line 1530:** LogisticRegression - Direct call: LogisticRegression

---

## src/training/enhanced_multi_timeframe_optimizer.py

**Undefined function calls: 1**

- **Line 386:** Path - Direct call: Path

---

## src/training/enhanced_optimization_orchestrator.py

**Undefined function calls: 10**

- **Line 66:** handles_errors - Direct call: handles_errors
- **Line 405:** handles_errors - Direct call: handles_errors
- **Line 53:** MultiObjectiveOptimizer - Direct call: MultiObjectiveOptimizer
- **Line 58:** AdvancedBayesianOptimizer - Direct call: AdvancedBayesianOptimizer
- **Line 63:** AdaptiveOptimizer - Direct call: AdaptiveOptimizer
- **Line 250:** AdvancedBayesianOptimizer - Direct call: AdvancedBayesianOptimizer
- **Line 126:** error - Direct call: error
- **Line 147:** failed - Direct call: failed
- **Line 158:** failed - Direct call: failed
- **Line 169:** failed - Direct call: failed

---

## src/training/enhanced_training_manager_optimized.py

**Undefined function calls: 9**

- **Line 530:** handles_errors - Direct call: handles_errors
- **Line 139:** ProcessPoolExecutor - Direct call: ProcessPoolExecutor
- **Line 92:** hash - Direct call: hash
- **Line 206:** hash - Direct call: hash
- **Line 265:** Path - Direct call: Path
- **Line 117:** evaluator_func - Direct call: evaluator_func
- **Line 403:** Path - Direct call: Path
- **Line 580:** Path - Direct call: Path
- **Line 597:** Path - Direct call: Path

---

## src/training/ensemble_manager.py

**Undefined function calls: 38**

- **Line 694:** handles_errors - Direct call: handles_errors
- **Line 61:** handles_errors - Direct call: handles_errors
- **Line 97:** handles_errors - Direct call: handles_errors
- **Line 125:** handles_errors - Direct call: handles_errors
- **Line 151:** handles_errors - Direct call: handles_errors
- **Line 152:** retry - Direct call: retry
- **Line 153:** log_execution_time - Direct call: log_execution_time
- **Line 154:** traced - Direct call: traced
- **Line 229:** handles_errors - Direct call: handles_errors
- **Line 273:** handles_errors - Direct call: handles_errors
- **Line 339:** handles_errors - Direct call: handles_errors
- **Line 393:** handles_errors - Direct call: handles_errors
- **Line 441:** handles_errors - Direct call: handles_errors
- **Line 491:** handles_errors - Direct call: handles_errors
- **Line 539:** handles_errors - Direct call: handles_errors
- **Line 584:** handles_errors - Direct call: handles_errors
- **Line 627:** handles_errors - Direct call: handles_errors
- **Line 677:** handles_errors - Direct call: handles_errors
- **Line 59:** get_trade_tracker - Direct call: get_trade_tracker
- **Line 136:** EnsembleCreator - Direct call: EnsembleCreator
- **Line 82:** invalid - Direct call: invalid
- **Line 94:** failed - Direct call: failed
- **Line 114:** error - Direct call: error
- **Line 122:** failed - Direct call: failed
- **Line 148:** failed - Direct call: failed
- **Line 225:** failed - Direct call: failed
- **Line 252:** error - Direct call: error
- **Line 257:** error - Direct call: error
- **Line 262:** error - Direct call: error
- **Line 270:** failed - Direct call: failed
- **Line 305:** warning - Direct call: warning
- **Line 336:** failed - Direct call: failed
- **Line 371:** warning - Direct call: warning
- **Line 390:** failed - Direct call: failed
- **Line 438:** failed - Direct call: failed
- **Line 536:** failed - Direct call: failed
- **Line 651:** failed - Direct call: failed
- **Line 691:** failed - Direct call: failed

---

## src/training/examples/simplified_pipeline_example.py

**Undefined function calls: 5**

- **Line 23:** create_training_manager - Direct call: create_training_manager
- **Line 54:** create_training_manager - Direct call: create_training_manager
- **Line 75:** create_training_manager - Direct call: create_training_manager
- **Line 91:** create_training_manager - Direct call: create_training_manager
- **Line 107:** Path - Direct call: Path

---

## src/training/factory.py

**Undefined function calls: 6**

- **Line 23:** get_optimization_config - Direct call: get_optimization_config
- **Line 36:** EnhancedTrainingManagerOptimized - Direct call: EnhancedTrainingManagerOptimized
- **Line 57:** MemoryProfiler - Direct call: MemoryProfiler
- **Line 68:** MemoryLeakDetector - Direct call: MemoryLeakDetector
- **Line 88:** OptimizedStepExecutor - Direct call: OptimizedStepExecutor
- **Line 134:** get_performance_expectations - Direct call: get_performance_expectations

---

## src/training/feature_engineering.py

**Undefined function calls: 13**

- **Line 25:** validates - Direct call: validates
- **Line 26:** traced - Direct call: traced
- **Line 40:** validates - Direct call: validates
- **Line 41:** traced - Direct call: traced
- **Line 53:** validates - Direct call: validates
- **Line 54:** traced - Direct call: traced
- **Line 66:** validates - Direct call: validates
- **Line 67:** traced - Direct call: traced
- **Line 78:** validates - Direct call: validates
- **Line 79:** traced - Direct call: traced
- **Line 91:** validates - Direct call: validates
- **Line 92:** traced - Direct call: traced
- **Line 30:** func - Direct call: func

---

## src/training/feature_selection_manager.py

**Undefined function calls: 7**

- **Line 40:** handles_errors - Direct call: handles_errors
- **Line 262:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 499:** RFE - Direct call: RFE
- **Line 617:** AutoencoderFeatureGenerator - Direct call: AutoencoderFeatureGenerator
- **Line 665:** RegularizationManager - Direct call: RegularizationManager
- **Line 738:** cross_val_score - Direct call: cross_val_score
- **Line 739:** LogisticRegression - Direct call: LogisticRegression

---

## src/training/gpu_acceleration_m1.py

**Undefined function calls: 9**

- **Line 51:** validates - Direct call: validates
- **Line 52:** quality_gate - Direct call: quality_gate
- **Line 56:** handles_errors - Direct call: handles_errors
- **Line 120:** secure_data_processing - Direct call: secure_data_processing
- **Line 121:** cached - Direct call: cached
- **Line 122:** log_call - Direct call: log_call
- **Line 123:** quality_gate - Direct call: quality_gate
- **Line 124:** handles_errors - Direct call: handles_errors
- **Line 198:** handles_errors - Direct call: handles_errors

---

## src/training/hmm_regime_barrier_optimizer.py

**Undefined function calls: 1**

- **Line 129:** Path - Direct call: Path

---

## src/training/matrix_diverse_lookback_optimizer.py

**Undefined function calls: 8**

- **Line 38:** handles_errors - Direct call: handles_errors
- **Line 33:** Path - Direct call: Path
- **Line 240:** minimize - Direct call: minimize
- **Line 362:** Path - Direct call: Path
- **Line 381:** Path - Direct call: Path
- **Line 383:** Path - Direct call: Path
- **Line 129:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 254:** TPESampler - Direct call: TPESampler

---

## src/training/matrix_enhancement_manager.py

**Undefined function calls: 11**

- **Line 71:** handles_errors - Direct call: handles_errors
- **Line 138:** handles_errors - Direct call: handles_errors
- **Line 201:** handles_errors - Direct call: handles_errors
- **Line 290:** handles_errors - Direct call: handles_errors
- **Line 373:** handles_errors - Direct call: handles_errors
- **Line 428:** handles_errors - Direct call: handles_errors
- **Line 90:** StandardScaler - Direct call: StandardScaler
- **Line 162:** NMF - Direct call: NMF
- **Line 220:** StandardScaler - Direct call: StandardScaler
- **Line 229:** SpectralClustering - Direct call: SpectralClustering
- **Line 252:** euclidean_distances - Direct call: euclidean_distances

---

## src/training/memory_profiler.py

**Undefined function calls: 4**

- **Line 33:** deque - Direct call: deque
- **Line 36:** defaultdict - Direct call: defaultdict
- **Line 37:** defaultdict - Direct call: defaultdict
- **Line 122:** defaultdict - Direct call: defaultdict

---

## src/training/model_probability_generator.py

**Undefined function calls: 3**

- **Line 22:** ClassificationProbabilityCalculator - Direct call: ClassificationProbabilityCalculator
- **Line 23:** RegressionProbabilityCalculator - Direct call: RegressionProbabilityCalculator
- **Line 42:** get_probability_calculator - Direct call: get_probability_calculator

---

## src/training/model_saving_utils.py

**Undefined function calls: 1**

- **Line 407:** ModelProbabilityGenerator - Direct call: ModelProbabilityGenerator

---

## src/training/model_specific_pruning.py

**Undefined function calls: 14**

- **Line 58:** handles_errors - Direct call: handles_errors
- **Line 152:** handles_errors - Direct call: handles_errors
- **Line 230:** handles_errors - Direct call: handles_errors
- **Line 299:** handles_errors - Direct call: handles_errors
- **Line 343:** handles_errors - Direct call: handles_errors
- **Line 401:** handles_errors - Direct call: handles_errors
- **Line 445:** handles_errors - Direct call: handles_errors
- **Line 627:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 675:** Lasso - Direct call: Lasso
- **Line 697:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 111:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 200:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 715:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 727:** mutual_info_classif - Direct call: mutual_info_classif

---

## src/training/model_trainer.py

**Undefined function calls: 32**

- **Line 992:** traced - Direct call: traced
- **Line 993:** handles_errors - Direct call: handles_errors
- **Line 138:** handles_errors - Direct call: handles_errors
- **Line 169:** handles_errors - Direct call: handles_errors
- **Line 203:** handles_errors - Direct call: handles_errors
- **Line 243:** handles_errors - Direct call: handles_errors
- **Line 456:** handles_errors - Direct call: handles_errors
- **Line 490:** validates - Direct call: validates
- **Line 491:** traced - Direct call: traced
- **Line 492:** handles_errors - Direct call: handles_errors
- **Line 802:** handles_errors - Direct call: handles_errors
- **Line 971:** handles_errors - Direct call: handles_errors
- **Line 542:** handle_missing_data - Direct call: handle_missing_data
- **Line 714:** StandardScaler - Direct call: StandardScaler
- **Line 736:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 737:** cross_val_score - Direct call: cross_val_score
- **Line 306:** log_enhanced_training_metadata - Direct call: log_enhanced_training_metadata
- **Line 719:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 721:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 729:** accuracy_score - Direct call: accuracy_score
- **Line 730:** precision_score - Direct call: precision_score
- **Line 731:** recall_score - Direct call: recall_score
- **Line 732:** f1_score - Direct call: f1_score
- **Line 338:** AdvancedOptunaManager - Direct call: AdvancedOptunaManager
- **Line 582:** MultiOutputModelConfig - Direct call: MultiOutputModelConfig
- **Line 588:** create_multi_output_trainer - Direct call: create_multi_output_trainer
- **Line 960:** StandardScaler - Direct call: StandardScaler
- **Line 349:** log_params_with_metadata - Direct call: log_params_with_metadata
- **Line 372:** log_metrics_with_metadata - Direct call: log_metrics_with_metadata
- **Line 390:** log_artifacts_with_metadata - Direct call: log_artifacts_with_metadata
- **Line 405:** log_artifacts_with_metadata - Direct call: log_artifacts_with_metadata
- **Line 431:** log_artifacts_with_metadata - Direct call: log_artifacts_with_metadata

---

## src/training/model_training_integrator.py

**Undefined function calls: 29**

- **Line 127:** handles_errors - Direct call: handles_errors
- **Line 637:** handles_errors - Direct call: handles_errors
- **Line 36:** get_component_logger - Direct call: get_component_logger
- **Line 138:** get_data_optimizer - Direct call: get_data_optimizer
- **Line 125:** error - Direct call: error
- **Line 149:** initialization_error - Direct call: initialization_error
- **Line 185:** error - Direct call: error
- **Line 255:** error - Direct call: error
- **Line 298:** model_class - Direct call: model_class
- **Line 307:** accuracy_score - Direct call: accuracy_score
- **Line 308:** precision_score - Direct call: precision_score
- **Line 309:** recall_score - Direct call: recall_score
- **Line 310:** f1_score - Direct call: f1_score
- **Line 313:** cross_val_score - Direct call: cross_val_score
- **Line 390:** error - Direct call: error
- **Line 407:** error - Direct call: error
- **Line 418:** failed - Direct call: failed
- **Line 425:** failed - Direct call: failed
- **Line 437:** error - Direct call: error
- **Line 478:** error - Direct call: error
- **Line 521:** error - Direct call: error
- **Line 542:** error - Direct call: error
- **Line 613:** error - Direct call: error
- **Line 634:** error - Direct call: error
- **Line 659:** error - Direct call: error
- **Line 508:** failed - Direct call: failed
- **Line 178:** error - Direct call: error
- **Line 346:** error - Direct call: error
- **Line 572:** error - Direct call: error

---

## src/training/multi_objective_optimizer.py

**Undefined function calls: 4**

- **Line 71:** handles_errors - Direct call: handles_errors
- **Line 251:** handles_errors - Direct call: handles_errors
- **Line 42:** StandardScaler - Direct call: StandardScaler
- **Line 66:** OptimizedBacktester - Direct call: OptimizedBacktester

---

## src/training/multi_output_model_trainer.py

**Undefined function calls: 88**

- **Line 250:** handles_errors - Direct call: handles_errors
- **Line 311:** handles_errors - Direct call: handles_errors
- **Line 660:** handles_errors - Direct call: handles_errors
- **Line 960:** handles_errors - Direct call: handles_errors
- **Line 1632:** handles_errors - Direct call: handles_errors
- **Line 1657:** handles_errors - Direct call: handles_errors
- **Line 214:** ProfitBasedFeatureEngineering - Direct call: ProfitBasedFeatureEngineering
- **Line 840:** accuracy_score - Direct call: accuracy_score
- **Line 924:** accuracy_score - Direct call: accuracy_score
- **Line 1003:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 1211:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 1221:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 1767:** trainer_func - Direct call: trainer_func
- **Line 1837:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 1888:** CNNModel - Direct call: CNNModel
- **Line 1895:** CNNTrainer - Direct call: CNNTrainer
- **Line 1938:** TCNModel - Direct call: TCNModel
- **Line 1945:** TCNTrainer - Direct call: TCNTrainer
- **Line 1988:** TransformerModel - Direct call: TransformerModel
- **Line 1997:** TransformerTrainer - Direct call: TransformerTrainer
- **Line 245:** ProbabilityTargetGenerator - Direct call: ProbabilityTargetGenerator
- **Line 841:** mean_squared_error - Direct call: mean_squared_error
- **Line 846:** f1_score - Direct call: f1_score
- **Line 847:** precision_score - Direct call: precision_score
- **Line 848:** recall_score - Direct call: recall_score
- **Line 853:** mean_absolute_error - Direct call: mean_absolute_error
- **Line 854:** r2_score - Direct call: r2_score
- **Line 925:** mean_squared_error - Direct call: mean_squared_error
- **Line 930:** f1_score - Direct call: f1_score
- **Line 931:** precision_score - Direct call: precision_score
- **Line 932:** recall_score - Direct call: recall_score
- **Line 937:** mean_absolute_error - Direct call: mean_absolute_error
- **Line 938:** r2_score - Direct call: r2_score
- **Line 1019:** StandardScaler - Direct call: StandardScaler
- **Line 1160:** accuracy_score - Direct call: accuracy_score
- **Line 1161:** precision_score - Direct call: precision_score
- **Line 1162:** recall_score - Direct call: recall_score
- **Line 1163:** f1_score - Direct call: f1_score
- **Line 1167:** mean_squared_error - Direct call: mean_squared_error
- **Line 1168:** mean_absolute_error - Direct call: mean_absolute_error
- **Line 1169:** r2_score - Direct call: r2_score
- **Line 1236:** accuracy_score - Direct call: accuracy_score
- **Line 1237:** precision_score - Direct call: precision_score
- **Line 1238:** recall_score - Direct call: recall_score
- **Line 1239:** f1_score - Direct call: f1_score
- **Line 1243:** mean_squared_error - Direct call: mean_squared_error
- **Line 1244:** mean_absolute_error - Direct call: mean_absolute_error
- **Line 1245:** r2_score - Direct call: r2_score
- **Line 1311:** model - Direct call: model
- **Line 1313:** criterion_direction - Direct call: criterion_direction
- **Line 1314:** criterion_profit - Direct call: criterion_profit
- **Line 1323:** model - Direct call: model
- **Line 1330:** accuracy_score - Direct call: accuracy_score
- **Line 1331:** precision_score - Direct call: precision_score
- **Line 1332:** recall_score - Direct call: recall_score
- **Line 1333:** f1_score - Direct call: f1_score
- **Line 1337:** mean_squared_error - Direct call: mean_squared_error
- **Line 1338:** mean_absolute_error - Direct call: mean_absolute_error
- **Line 1339:** r2_score - Direct call: r2_score
- **Line 1795:** compute_class_weight - Direct call: compute_class_weight
- **Line 1811:** accuracy_score - Direct call: accuracy_score
- **Line 1812:** f1_score - Direct call: f1_score
- **Line 1813:** precision_score - Direct call: precision_score
- **Line 1814:** recall_score - Direct call: recall_score
- **Line 1853:** accuracy_score - Direct call: accuracy_score
- **Line 1854:** f1_score - Direct call: f1_score
- **Line 1855:** precision_score - Direct call: precision_score
- **Line 1856:** recall_score - Direct call: recall_score
- **Line 1903:** accuracy_score - Direct call: accuracy_score
- **Line 1904:** f1_score - Direct call: f1_score
- **Line 1905:** precision_score - Direct call: precision_score
- **Line 1906:** recall_score - Direct call: recall_score
- **Line 1953:** accuracy_score - Direct call: accuracy_score
- **Line 1954:** f1_score - Direct call: f1_score
- **Line 1955:** precision_score - Direct call: precision_score
- **Line 1956:** recall_score - Direct call: recall_score
- **Line 2005:** accuracy_score - Direct call: accuracy_score
- **Line 2006:** f1_score - Direct call: f1_score
- **Line 2007:** precision_score - Direct call: precision_score
- **Line 2008:** recall_score - Direct call: recall_score
- **Line 269:** Path - Direct call: Path
- **Line 330:** Path - Direct call: Path
- **Line 707:** Step6HMMBasedTraining - Direct call: Step6HMMBasedTraining
- **Line 1170:** mean_squared_error - Direct call: mean_squared_error
- **Line 1246:** mean_squared_error - Direct call: mean_squared_error
- **Line 1340:** mean_squared_error - Direct call: mean_squared_error
- **Line 1404:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 1411:** RandomForestRegressor - Direct call: RandomForestRegressor

---

## src/training/multi_output_probability_trainer.py

**Undefined function calls: 10**

- **Line 140:** handles_errors - Direct call: handles_errors
- **Line 180:** handles_errors - Direct call: handles_errors
- **Line 256:** handles_errors - Direct call: handles_errors
- **Line 257:** validates - Direct call: validates
- **Line 273:** log_execution_time - Direct call: log_execution_time
- **Line 295:** handles_errors - Direct call: handles_errors
- **Line 78:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 172:** minimize - Direct call: minimize
- **Line 105:** compute_class_weight - Direct call: compute_class_weight
- **Line 120:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV

---

## src/training/optimization_manager.py

**Undefined function calls: 30**

- **Line 655:** handles_errors - Direct call: handles_errors
- **Line 58:** handles_errors - Direct call: handles_errors
- **Line 92:** handles_errors - Direct call: handles_errors
- **Line 122:** handles_errors - Direct call: handles_errors
- **Line 150:** handles_errors - Direct call: handles_errors
- **Line 227:** handles_errors - Direct call: handles_errors
- **Line 271:** handles_errors - Direct call: handles_errors
- **Line 345:** handles_errors - Direct call: handles_errors
- **Line 394:** handles_errors - Direct call: handles_errors
- **Line 445:** handles_errors - Direct call: handles_errors
- **Line 508:** handles_errors - Direct call: handles_errors
- **Line 549:** handles_errors - Direct call: handles_errors
- **Line 589:** handles_errors - Direct call: handles_errors
- **Line 640:** handles_errors - Direct call: handles_errors
- **Line 79:** invalid - Direct call: invalid
- **Line 89:** failed - Direct call: failed
- **Line 113:** error - Direct call: error
- **Line 119:** failed - Direct call: failed
- **Line 223:** failed - Direct call: failed
- **Line 250:** error - Direct call: error
- **Line 255:** error - Direct call: error
- **Line 262:** error - Direct call: error
- **Line 268:** failed - Direct call: failed
- **Line 342:** failed - Direct call: failed
- **Line 442:** failed - Direct call: failed
- **Line 505:** failed - Direct call: failed
- **Line 546:** failed - Direct call: failed
- **Line 586:** failed - Direct call: failed
- **Line 614:** failed - Direct call: failed
- **Line 652:** failed - Direct call: failed

---

## src/training/optimized_feature_selection_manager.py

**Undefined function calls: 11**

- **Line 97:** handles_errors - Direct call: handles_errors
- **Line 261:** StandardScaler - Direct call: StandardScaler
- **Line 522:** Lasso - Direct call: Lasso
- **Line 362:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 303:** variance_inflation_factor - Direct call: variance_inflation_factor
- **Line 407:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 546:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 554:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 266:** LedoitWolf - Direct call: LedoitWolf
- **Line 449:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 435:** mutual_info_classif - Direct call: mutual_info_classif

---

## src/training/probabilistic_bayesian_optimizer.py

**Undefined function calls: 5**

- **Line 249:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 254:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 107:** brier_score_loss - Direct call: brier_score_loss
- **Line 122:** roc_auc_score - Direct call: roc_auc_score
- **Line 143:** model_factory - Direct call: model_factory

---

## src/training/probabilistic_model_integration.py

**Undefined function calls: 5**

- **Line 51:** ProbabilisticOptimizationConfig - Direct call: ProbabilisticOptimizationConfig
- **Line 52:** ProbabilisticBayesianOptimizer - Direct call: ProbabilisticBayesianOptimizer
- **Line 248:** main - Direct call: main
- **Line 121:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 132:** RandomForestClassifier - Direct call: RandomForestClassifier

---

## src/training/probability_calculators.py

**Undefined function calls: 1**

- **Line 76:** accuracy_score - Direct call: accuracy_score

---

## src/training/progress_manager.py

**Undefined function calls: 3**

- **Line 40:** handles_errors - Direct call: handles_errors
- **Line 90:** handles_errors - Direct call: handles_errors
- **Line 34:** Path - Direct call: Path

---

## src/training/run_feature_pipeline.py

**Undefined function calls: 5**

- **Line 53:** run_feature_pipeline - Direct call: run_feature_pipeline
- **Line 9:** Path - Direct call: Path
- **Line 24:** run_step06 - Direct call: run_step06
- **Line 30:** run_step07 - Direct call: run_step07
- **Line 36:** run_step08 - Direct call: run_step08

---

## src/training/run_pipeline_with_step08.py

**Undefined function calls: 5**

- **Line 78:** EnhancedTrainingManager - Direct call: EnhancedTrainingManager
- **Line 150:** run_pipeline_with_step08 - Direct call: run_pipeline_with_step08
- **Line 167:** main - Direct call: main
- **Line 10:** Path - Direct call: Path
- **Line 107:** Path - Direct call: Path

---

## src/training/simplified_architecture/config_driven_architecture.py

**Undefined function calls: 4**

- **Line 107:** Path - Direct call: Path
- **Line 197:** Path - Direct call: Path
- **Line 237:** step_class - Direct call: step_class
- **Line 210:** asdict - Direct call: asdict

---

## src/training/simplified_architecture/dependency_injection.py

**Undefined function calls: 9**

- **Line 18:** TypeVar - Direct call: TypeVar
- **Line 32:** field - Direct call: field
- **Line 35:** field - Direct call: field
- **Line 415:** DIContainer - Direct call: DIContainer
- **Line 417:** Path - Direct call: Path
- **Line 281:** func - Direct call: func
- **Line 332:** original_init - Direct call: original_init
- **Line 423:** MemoryCache - Direct call: MemoryCache
- **Line 113:** factory - Direct call: factory

---

## src/training/simplified_architecture/enhanced_config_system.py

**Undefined function calls: 17**

- **Line 38:** field - Direct call: field
- **Line 39:** field - Direct call: field
- **Line 40:** field - Direct call: field
- **Line 41:** field - Direct call: field
- **Line 42:** field - Direct call: field
- **Line 43:** field - Direct call: field
- **Line 56:** field - Direct call: field
- **Line 57:** field - Direct call: field
- **Line 59:** field - Direct call: field
- **Line 60:** field - Direct call: field
- **Line 61:** field - Direct call: field
- **Line 70:** field - Direct call: field
- **Line 71:** field - Direct call: field
- **Line 72:** field - Direct call: field
- **Line 116:** Path - Direct call: Path
- **Line 166:** Path - Direct call: Path
- **Line 99:** Path - Direct call: Path

---

## src/training/simplified_architecture/enhanced_interfaces.py

**Undefined function calls: 12**

- **Line 27:** TypeVar - Direct call: TypeVar
- **Line 52:** field - Direct call: field
- **Line 53:** field - Direct call: field
- **Line 54:** field - Direct call: field
- **Line 109:** field - Direct call: field
- **Line 110:** field - Direct call: field
- **Line 112:** field - Direct call: field
- **Line 113:** field - Direct call: field
- **Line 114:** field - Direct call: field
- **Line 521:** step_class - Direct call: step_class
- **Line 607:** Path - Direct call: Path
- **Line 414:** TimeoutError - Direct call: TimeoutError

---

## src/training/simplified_architecture/enhanced_pipeline_orchestrator.py

**Undefined function calls: 12**

- **Line 44:** field - Direct call: field
- **Line 45:** field - Direct call: field
- **Line 46:** field - Direct call: field
- **Line 47:** field - Direct call: field
- **Line 48:** field - Direct call: field
- **Line 522:** ConfigurationManager - Direct call: ConfigurationManager
- **Line 114:** ConfigurationManager - Direct call: ConfigurationManager
- **Line 478:** Path - Direct call: Path
- **Line 548:** example_usage - Direct call: example_usage
- **Line 113:** EnhancedDIContainer - Direct call: EnhancedDIContainer
- **Line 276:** StepConfig - Direct call: StepConfig
- **Line 462:** StepResult - Direct call: StepResult

---

## src/training/simplified_architecture/example_new_exchange.py

**Undefined function calls: 4**

- **Line 102:** CatBoostClassifier - Direct call: CatBoostClassifier
- **Line 136:** example_usage - Direct call: example_usage
- **Line 123:** datetime - Direct call: datetime
- **Line 123:** datetime - Direct call: datetime

---

## src/training/simplified_architecture/integrated_example.py

**Undefined function calls: 15**

- **Line 23:** inject - Direct call: inject
- **Line 54:** inject - Direct call: inject
- **Line 77:** inject - Direct call: inject
- **Line 95:** inject - Direct call: inject
- **Line 198:** Path - Direct call: Path
- **Line 126:** DIContainer - Direct call: DIContainer
- **Line 206:** create_example_config - Direct call: create_example_config
- **Line 222:** main - Direct call: main
- **Line 159:** StepConfig - Direct call: StepConfig
- **Line 135:** SchemaValidator - Direct call: SchemaValidator
- **Line 135:** DataQualityValidator - Direct call: DataQualityValidator
- **Line 136:** PriceFeatureCalculator - Direct call: PriceFeatureCalculator
- **Line 136:** VolumeFeatureCalculator - Direct call: VolumeFeatureCalculator
- **Line 163:** step_class - Direct call: step_class
- **Line 131:** LocalDataSource - Direct call: LocalDataSource

---

## src/training/simplified_architecture/migrated_components/data_components.py

**Undefined function calls: 3**

- **Line 114:** Path - Direct call: Path
- **Line 97:** timedelta - Direct call: timedelta
- **Line 300:** Path - Direct call: Path

---

## src/training/simplified_architecture/modular_components.py

**Undefined function calls: 10**

- **Line 170:** Path - Direct call: Path
- **Line 570:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 630:** StandardScaler - Direct call: StandardScaler
- **Line 675:** ModelTrainer - Direct call: ModelTrainer
- **Line 723:** Path - Direct call: Path
- **Line 760:** example_usage - Direct call: example_usage
- **Line 656:** model - Direct call: model
- **Line 657:** criterion - Direct call: criterion
- **Line 755:** datetime - Direct call: datetime
- **Line 755:** datetime - Direct call: datetime

---

## src/training/simplified_architecture/standard_interfaces.py

**Undefined function calls: 10**

- **Line 18:** TypeVar - Direct call: TypeVar
- **Line 35:** field - Direct call: field
- **Line 36:** field - Direct call: field
- **Line 37:** field - Direct call: field
- **Line 62:** field - Direct call: field
- **Line 63:** field - Direct call: field
- **Line 324:** step_class - Direct call: step_class
- **Line 346:** example_usage - Direct call: example_usage
- **Line 282:** Path - Direct call: Path
- **Line 211:** TimeoutError - Direct call: TimeoutError

---

## src/training/simplified_training_manager.py

**Undefined function calls: 10**

- **Line 70:** handles_errors - Direct call: handles_errors
- **Line 53:** ProgressManager - Direct call: ProgressManager
- **Line 54:** StepDependencyValidator - Direct call: StepDependencyValidator
- **Line 352:** get_all_steps - Direct call: get_all_steps
- **Line 84:** validate_step_sequence - Direct call: validate_step_sequence
- **Line 130:** get_step_execution_order_full_names - Direct call: get_step_execution_order_full_names
- **Line 228:** get_step_config - Direct call: get_step_config
- **Line 322:** step_class - Direct call: step_class
- **Line 151:** get_step_number_from_full_name - Direct call: get_step_number_from_full_name
- **Line 152:** get_step_config - Direct call: get_step_config

---

## src/training/step_config.py

**Undefined function calls: 4**

- **Line 26:** field - Direct call: field
- **Line 27:** field - Direct call: field
- **Line 28:** field - Direct call: field
- **Line 29:** field - Direct call: field

---

## src/training/step_orchestrator.py

**Undefined function calls: 7**

- **Line 35:** ProgressManager - Direct call: ProgressManager
- **Line 328:** apply_mode_parameters_to_config - Direct call: apply_mode_parameters_to_config
- **Line 331:** get_step_specific_parameters - Direct call: get_step_specific_parameters
- **Line 206:** apply_mode_parameters_to_config - Direct call: apply_mode_parameters_to_config
- **Line 209:** get_step_specific_parameters - Direct call: get_step_specific_parameters
- **Line 92:** create_training_manager - Direct call: create_training_manager
- **Line 187:** validate_step_dependencies - Direct call: validate_step_dependencies

---

## src/training/steps/backtesting/__init__.py

**Undefined function calls: 39**

- **Line 277:** compose - Direct call: compose
- **Line 283:** validate_pipeline_step - Direct call: validate_pipeline_step
- **Line 291:** get_backtesting_logger - Direct call: get_backtesting_logger
- **Line 278:** error_boundary - Direct call: error_boundary
- **Line 279:** traced - Direct call: traced
- **Line 281:** timeout - Direct call: timeout
- **Line 569:** get_current_datetime - Direct call: get_current_datetime
- **Line 582:** generate_backtesting_report - Direct call: generate_backtesting_report
- **Line 596:** generate_detailed_regime_metrics_report - Direct call: generate_detailed_regime_metrics_report
- **Line 172:** Path - Direct call: Path
- **Line 362:** get_current_datetime - Direct call: get_current_datetime
- **Line 573:** Path - Direct call: Path
- **Line 577:** Path - Direct call: Path
- **Line 581:** Path - Direct call: Path
- **Line 595:** Path - Direct call: Path
- **Line 626:** Path - Direct call: Path
- **Line 189:** safe_file_exists - Direct call: safe_file_exists
- **Line 384:** WalkForwardValidationPerRegimeStep - Direct call: WalkForwardValidationPerRegimeStep
- **Line 434:** MonteCarloValidationPerRegimeStep - Direct call: MonteCarloValidationPerRegimeStep
- **Line 484:** ABTestingPerRegimeStep - Direct call: ABTestingPerRegimeStep
- **Line 534:** SavingStep - Direct call: SavingStep
- **Line 595:** format_datetime - Direct call: format_datetime
- **Line 409:** generate_step_report - Direct call: generate_step_report
- **Line 459:** generate_step_report - Direct call: generate_step_report
- **Line 509:** generate_step_report - Direct call: generate_step_report
- **Line 549:** generate_step_report - Direct call: generate_step_report
- **Line 595:** get_current_datetime - Direct call: get_current_datetime
- **Line 408:** Path - Direct call: Path
- **Line 458:** Path - Direct call: Path
- **Line 508:** Path - Direct call: Path
- **Line 548:** Path - Direct call: Path
- **Line 408:** format_datetime - Direct call: format_datetime
- **Line 458:** format_datetime - Direct call: format_datetime
- **Line 508:** format_datetime - Direct call: format_datetime
- **Line 548:** format_datetime - Direct call: format_datetime
- **Line 408:** get_current_datetime - Direct call: get_current_datetime
- **Line 458:** get_current_datetime - Direct call: get_current_datetime
- **Line 508:** get_current_datetime - Direct call: get_current_datetime
- **Line 548:** get_current_datetime - Direct call: get_current_datetime

---

## src/training/steps/backtesting/comprehensive_reporting.py

**Undefined function calls: 13**

- **Line 370:** ComprehensiveReporter - Direct call: ComprehensiveReporter
- **Line 379:** get_current_datetime - Direct call: get_current_datetime
- **Line 22:** Path - Direct call: Path
- **Line 29:** ensure_directory - Direct call: ensure_directory
- **Line 70:** get_current_datetime - Direct call: get_current_datetime
- **Line 465:** safe_json_dump - Direct call: safe_json_dump
- **Line 56:** safe_json_dump - Direct call: safe_json_dump
- **Line 90:** safe_json_dump - Direct call: safe_json_dump
- **Line 476:** format_datetime - Direct call: format_datetime
- **Line 674:** safe_file_exists - Direct call: safe_file_exists
- **Line 883:** format_datetime - Direct call: format_datetime
- **Line 476:** get_current_datetime - Direct call: get_current_datetime
- **Line 883:** get_current_datetime - Direct call: get_current_datetime

---

## src/training/steps/backtesting/enhanced_logging.py

**Undefined function calls: 7**

- **Line 26:** Path - Direct call: Path
- **Line 40:** format_datetime - Direct call: format_datetime
- **Line 40:** get_current_datetime - Direct call: get_current_datetime
- **Line 433:** format_datetime - Direct call: format_datetime
- **Line 434:** format_datetime - Direct call: format_datetime
- **Line 433:** get_current_datetime - Direct call: get_current_datetime
- **Line 434:** get_current_datetime - Direct call: get_current_datetime

---

## src/training/steps/backtesting/step18_backtesting_main.py

**Undefined function calls: 37**

- **Line 39:** get_logger - Direct call: get_logger
- **Line 41:** compose - Direct call: compose
- **Line 47:** validate_pipeline_step - Direct call: validate_pipeline_step
- **Line 51:** monitor_step_execution - Direct call: monitor_step_execution
- **Line 68:** get_backtesting_logger - Direct call: get_backtesting_logger
- **Line 42:** error_boundary - Direct call: error_boundary
- **Line 43:** traced - Direct call: traced
- **Line 45:** timeout - Direct call: timeout
- **Line 430:** main - Direct call: main
- **Line 406:** main - Direct call: main
- **Line 21:** Path - Direct call: Path
- **Line 123:** Path - Direct call: Path
- **Line 162:** run_backtesting_pipeline - Direct call: run_backtesting_pipeline
- **Line 202:** safe_json_dump - Direct call: safe_json_dump
- **Line 215:** safe_json_dump - Direct call: safe_json_dump
- **Line 248:** safe_json_dump - Direct call: safe_json_dump
- **Line 279:** safe_json_dump - Direct call: safe_json_dump
- **Line 74:** format_datetime - Direct call: format_datetime
- **Line 195:** format_datetime - Direct call: format_datetime
- **Line 196:** format_datetime - Direct call: format_datetime
- **Line 201:** Path - Direct call: Path
- **Line 206:** Path - Direct call: Path
- **Line 212:** format_datetime - Direct call: format_datetime
- **Line 219:** Path - Direct call: Path
- **Line 244:** format_datetime - Direct call: format_datetime
- **Line 247:** Path - Direct call: Path
- **Line 252:** Path - Direct call: Path
- **Line 275:** format_datetime - Direct call: format_datetime
- **Line 278:** Path - Direct call: Path
- **Line 283:** Path - Direct call: Path
- **Line 74:** get_current_datetime - Direct call: get_current_datetime
- **Line 140:** safe_file_exists - Direct call: safe_file_exists
- **Line 195:** get_current_datetime - Direct call: get_current_datetime
- **Line 196:** get_current_datetime - Direct call: get_current_datetime
- **Line 212:** get_current_datetime - Direct call: get_current_datetime
- **Line 244:** get_current_datetime - Direct call: get_current_datetime
- **Line 275:** get_current_datetime - Direct call: get_current_datetime

---

## src/training/steps/backtesting/step18_walk_forward_validation_per_regime.py

**Undefined function calls: 9**

- **Line 825:** traced - Direct call: traced
- **Line 826:** validates - Direct call: validates
- **Line 38:** traced - Direct call: traced
- **Line 39:** per_regime_step - Direct call: per_regime_step
- **Line 889:** test - Direct call: test
- **Line 881:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 128:** Path - Direct call: Path
- **Line 812:** Path - Direct call: Path
- **Line 132:** Path - Direct call: Path

---

## src/training/steps/backtesting/step18_walk_forward_validation_validator.py

**Undefined function calls: 19**

- **Line 37:** handles_errors - Direct call: handles_errors
- **Line 128:** handles_errors - Direct call: handles_errors
- **Line 170:** handles_errors - Direct call: handles_errors
- **Line 258:** handles_errors - Direct call: handles_errors
- **Line 331:** handles_errors - Direct call: handles_errors
- **Line 19:** Path - Direct call: Path
- **Line 193:** safe_json_load - Direct call: safe_json_load
- **Line 279:** safe_json_load - Direct call: safe_json_load
- **Line 352:** safe_json_load - Direct call: safe_json_load
- **Line 438:** run_validator - Direct call: run_validator
- **Line 440:** test_validator - Direct call: test_validator
- **Line 66:** validation_error - Direct call: validation_error
- **Line 77:** failed - Direct call: failed
- **Line 88:** failed - Direct call: failed
- **Line 99:** failed - Direct call: failed
- **Line 110:** failed - Direct call: failed
- **Line 121:** validation_error - Direct call: validation_error
- **Line 286:** error - Direct call: error
- **Line 288:** error - Direct call: error

---

## src/training/steps/backtesting/step19_monte_carlo_validation_per_regime.py

**Undefined function calls: 8**

- **Line 145:** traced - Direct call: traced
- **Line 146:** validates - Direct call: validates
- **Line 27:** traced - Direct call: traced
- **Line 28:** per_regime_step - Direct call: per_regime_step
- **Line 172:** test - Direct call: test
- **Line 170:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 70:** Path - Direct call: Path
- **Line 136:** Path - Direct call: Path

---

## src/training/steps/backtesting/step19_monte_carlo_validation_validator.py

**Undefined function calls: 19**

- **Line 36:** handles_errors - Direct call: handles_errors
- **Line 125:** handles_errors - Direct call: handles_errors
- **Line 167:** handles_errors - Direct call: handles_errors
- **Line 254:** handles_errors - Direct call: handles_errors
- **Line 374:** handles_errors - Direct call: handles_errors
- **Line 18:** Path - Direct call: Path
- **Line 188:** safe_json_load - Direct call: safe_json_load
- **Line 277:** safe_json_load - Direct call: safe_json_load
- **Line 395:** safe_json_load - Direct call: safe_json_load
- **Line 508:** run_validator - Direct call: run_validator
- **Line 510:** test_validator - Direct call: test_validator
- **Line 65:** validation_error - Direct call: validation_error
- **Line 76:** failed - Direct call: failed
- **Line 87:** failed - Direct call: failed
- **Line 98:** failed - Direct call: failed
- **Line 109:** failed - Direct call: failed
- **Line 118:** validation_error - Direct call: validation_error
- **Line 244:** error - Direct call: error
- **Line 407:** error - Direct call: error

---

## src/training/steps/backtesting/step20_ab_testing_per_regime.py

**Undefined function calls: 10**

- **Line 36:** get_logger - Direct call: get_logger
- **Line 280:** traced - Direct call: traced
- **Line 281:** validates - Direct call: validates
- **Line 87:** traced - Direct call: traced
- **Line 88:** per_regime_step - Direct call: per_regime_step
- **Line 308:** test - Direct call: test
- **Line 306:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 20:** Path - Direct call: Path
- **Line 121:** Path - Direct call: Path
- **Line 271:** Path - Direct call: Path

---

## src/training/steps/backtesting/step20_ab_testing_validator.py

**Undefined function calls: 5**

- **Line 17:** get_logger - Direct call: get_logger
- **Line 24:** get_logger - Direct call: get_logger
- **Line 39:** Path - Direct call: Path
- **Line 48:** safe_json_load - Direct call: safe_json_load
- **Line 11:** Path - Direct call: Path

---

## src/training/steps/backtesting/step21_saving_per_regime.py

**Undefined function calls: 8**

- **Line 1025:** traced - Direct call: traced
- **Line 1026:** validates - Direct call: validates
- **Line 38:** traced - Direct call: traced
- **Line 39:** per_regime_step - Direct call: per_regime_step
- **Line 1089:** test - Direct call: test
- **Line 1081:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 127:** Path - Direct call: Path
- **Line 1012:** Path - Direct call: Path

---

## src/training/steps/backtesting/step21_saving_validator.py

**Undefined function calls: 31**

- **Line 524:** test_validator - Direct call: test_validator
- **Line 21:** Path - Direct call: Path
- **Line 353:** callable - Direct call: callable
- **Line 522:** run_validator - Direct call: run_validator
- **Line 67:** error - Direct call: error
- **Line 77:** failed - Direct call: failed
- **Line 87:** failed - Direct call: failed
- **Line 93:** failed - Direct call: failed
- **Line 99:** failed - Direct call: failed
- **Line 109:** error - Direct call: error
- **Line 223:** safe_json_load - Direct call: safe_json_load
- **Line 369:** callable - Direct call: callable
- **Line 373:** callable - Direct call: callable
- **Line 401:** safe_json_load - Direct call: safe_json_load
- **Line 155:** missing - Direct call: missing
- **Line 162:** error - Direct call: error
- **Line 305:** safe_json_load - Direct call: safe_json_load
- **Line 327:** safe_json_load - Direct call: safe_json_load
- **Line 344:** validation_error - Direct call: validation_error
- **Line 281:** callable - Direct call: callable
- **Line 359:** callable - Direct call: callable
- **Line 282:** missing - Direct call: missing
- **Line 297:** error - Direct call: error
- **Line 319:** error - Direct call: error
- **Line 336:** error - Direct call: error
- **Line 423:** error - Direct call: error
- **Line 431:** invalid - Direct call: invalid
- **Line 456:** error - Direct call: error
- **Line 425:** error - Direct call: error
- **Line 472:** error - Direct call: error
- **Line 363:** callable - Direct call: callable

---

## src/training/steps/data_collection/data_downloader.py

**Undefined function calls: 5**

- **Line 15:** handles_errors - Direct call: handles_errors
- **Line 48:** OptimizedDownloadConfig - Direct call: OptimizedDownloadConfig
- **Line 55:** OptimizedDataDownloader - Direct call: OptimizedDataDownloader
- **Line 69:** CleanDownloadConfig - Direct call: CleanDownloadConfig
- **Line 76:** CleanDataDownloader - Direct call: CleanDataDownloader

---

## src/training/steps/data_collection/data_preparation/step01_5_data_converter.py

**Undefined function calls: 39**

- **Line 1316:** handles_errors - Direct call: handles_errors
- **Line 1322:** handles_errors - Direct call: handles_errors
- **Line 369:** validates - Direct call: validates
- **Line 370:** traced - Direct call: traced
- **Line 419:** handles_errors - Direct call: handles_errors
- **Line 511:** handles_errors - Direct call: handles_errors
- **Line 587:** handles_errors - Direct call: handles_errors
- **Line 600:** handles_errors - Direct call: handles_errors
- **Line 674:** handles_errors - Direct call: handles_errors
- **Line 845:** validates - Direct call: validates
- **Line 906:** handles_errors - Direct call: handles_errors
- **Line 907:** validate_aggtrades_data - Direct call: validate_aggtrades_data
- **Line 908:** format_aggtrades_data - Direct call: format_aggtrades_data
- **Line 909:** log_step_metrics - Direct call: log_step_metrics
- **Line 939:** handles_errors - Direct call: handles_errors
- **Line 940:** validate_futures_data - Direct call: validate_futures_data
- **Line 941:** format_futures_data - Direct call: format_futures_data
- **Line 942:** log_step_metrics - Direct call: log_step_metrics
- **Line 971:** validates - Direct call: validates
- **Line 1243:** handles_errors - Direct call: handles_errors
- **Line 422:** ensure_directory - Direct call: ensure_directory
- **Line 590:** ensure_directory - Direct call: ensure_directory
- **Line 656:** ensure_directory - Direct call: ensure_directory
- **Line 657:** ensure_directory - Direct call: ensure_directory
- **Line 16:** Path - Direct call: Path
- **Line 629:** safe_json_dump - Direct call: safe_json_dump
- **Line 680:** ensure_directory - Direct call: ensure_directory
- **Line 681:** ensure_directory - Direct call: ensure_directory
- **Line 744:** EnhancedDataQualityManager - Direct call: EnhancedDataQualityManager
- **Line 866:** ensure_directory - Direct call: ensure_directory
- **Line 1068:** safe_json_dump - Direct call: safe_json_dump
- **Line 1404:** run_step - Direct call: run_step
- **Line 1409:** _main - Direct call: _main
- **Line 640:** safe_json_load - Direct call: safe_json_load
- **Line 720:** validate_step1_5_quality - Direct call: validate_step1_5_quality
- **Line 891:** timedelta - Direct call: timedelta
- **Line 876:** timedelta - Direct call: timedelta
- **Line 897:** timedelta - Direct call: timedelta
- **Line 786:** date - Direct call: date

---

## src/training/steps/data_collection/data_preparation/step01_5_data_converter_refactored.py

**Undefined function calls: 5**

- **Line 105:** DataFormatConverter - Direct call: DataFormatConverter
- **Line 106:** DataValidator - Direct call: DataValidator
- **Line 107:** DataCleaner - Direct call: DataCleaner
- **Line 380:** main - Direct call: main
- **Line 22:** Path - Direct call: Path

---

## src/training/steps/data_collection/data_preparation/step01_5_data_converter_wrapper.py

**Undefined function calls: 3**

- **Line 29:** handles_errors - Direct call: handles_errors
- **Line 73:** UnifiedDataConverter - Direct call: UnifiedDataConverter
- **Line 62:** run_step_15 - Direct call: run_step_15

---

## src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py

**Undefined function calls: 15**

- **Line 537:** handles_errors - Direct call: handles_errors
- **Line 329:** func - Direct call: func
- **Line 382:** PipelineStandards - Direct call: PipelineStandards
- **Line 967:** train_test_split - Direct call: train_test_split
- **Line 970:** train_test_split - Direct call: train_test_split
- **Line 983:** StandardScaler - Direct call: StandardScaler
- **Line 996:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 999:** accuracy_score - Direct call: accuracy_score
- **Line 1009:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 1065:** Path - Direct call: Path
- **Line 1179:** test - Direct call: test
- **Line 191:** func - Direct call: func
- **Line 302:** func - Direct call: func
- **Line 73:** func - Direct call: func
- **Line 71:** func - Direct call: func

---

## src/training/steps/data_collection/data_preparation/step02_data_reading.py

**Undefined function calls: 4**

- **Line 95:** handles_errors - Direct call: handles_errors
- **Line 130:** ParquetUtils - Direct call: ParquetUtils
- **Line 131:** Path - Direct call: Path
- **Line 85:** Path - Direct call: Path

---

## src/training/steps/data_collection/data_preparation/step03_hmm_regime_discovery.py

**Undefined function calls: 3**

- **Line 49:** PipelineStandards - Direct call: PipelineStandards
- **Line 268:** Path - Direct call: Path
- **Line 14:** Path - Direct call: Path

---

## src/training/steps/data_collection/data_preparation_components/aggtrades_data_formatting.py

**Undefined function calls: 1**

- **Line 75:** processor - Direct call: processor

---

## src/training/steps/data_collection/data_preparation_components/data_format_converter.py

**Undefined function calls: 3**

- **Line 59:** validates - Direct call: validates
- **Line 60:** traced - Direct call: traced
- **Line 560:** safe_json_load - Direct call: safe_json_load

---

## src/training/steps/data_collection/data_preparation_components/data_integrity_checker.py

**Undefined function calls: 2**

- **Line 203:** timedelta - Direct call: timedelta
- **Line 219:** timedelta - Direct call: timedelta

---

## src/training/steps/data_collection/data_preparation_components/training_validation_config.py

**Undefined function calls: 1**

- **Line 129:** __import__ - Direct call: __import__

---

## src/training/steps/data_collection/data_quality_components/data_downloader.py

**Undefined function calls: 2**

- **Line 73:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation
- **Line 132:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation

---

## src/training/steps/data_collection/data_quality_components/data_integrity_checker.py

**Undefined function calls: 2**

- **Line 247:** timedelta - Direct call: timedelta
- **Line 286:** timedelta - Direct call: timedelta

---

## src/training/steps/data_collection/data_quality_components/data_preprocessor.py

**Undefined function calls: 1**

- **Line 294:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation

---

## src/training/steps/data_collection/data_quality_components/error_handler.py

**Undefined function calls: 1**

- **Line 355:** func - Direct call: func

---

## src/training/steps/data_collection/data_quality_components/result_builder.py

**Undefined function calls: 3**

- **Line 145:** calculate_interval_statistics - Direct call: calculate_interval_statistics
- **Line 179:** detect_data_gaps - Direct call: detect_data_gaps
- **Line 206:** calculate_volume_statistics - Direct call: calculate_volume_statistics

---

## src/training/steps/data_collection/data_quality_components/validation_decorators.py

**Undefined function calls: 8**

- **Line 61:** func - Direct call: func
- **Line 158:** func - Direct call: func
- **Line 233:** func - Direct call: func
- **Line 50:** fix_datetime_index - Direct call: fix_datetime_index
- **Line 83:** func - Direct call: func
- **Line 119:** func - Direct call: func
- **Line 224:** func - Direct call: func
- **Line 231:** func - Direct call: func

---

## src/training/steps/data_collection/data_quality_components/validation_strategies.py

**Undefined function calls: 6**

- **Line 417:** timedelta - Direct call: timedelta
- **Line 256:** timedelta - Direct call: timedelta
- **Line 355:** timedelta - Direct call: timedelta
- **Line 482:** timedelta - Direct call: timedelta
- **Line 424:** timedelta - Direct call: timedelta
- **Line 491:** timedelta - Direct call: timedelta

---

## src/training/steps/data_collection/enhanced_data_collection_demo.py

**Undefined function calls: 28**

- **Line 432:** traced - Direct call: traced
- **Line 62:** handles_errors - Direct call: handles_errors
- **Line 63:** traced - Direct call: traced
- **Line 119:** handles_errors - Direct call: handles_errors
- **Line 120:** traced - Direct call: traced
- **Line 204:** handles_errors - Direct call: handles_errors
- **Line 205:** traced - Direct call: traced
- **Line 267:** handles_errors - Direct call: handles_errors
- **Line 268:** traced - Direct call: traced
- **Line 340:** handles_errors - Direct call: handles_errors
- **Line 341:** traced - Direct call: traced
- **Line 392:** handles_errors - Direct call: handles_errors
- **Line 393:** traced - Direct call: traced
- **Line 70:** list_supported_exchanges - Direct call: list_supported_exchanges
- **Line 147:** validate_data_batch - Direct call: validate_data_batch
- **Line 167:** validate_data_batch - Direct call: validate_data_batch
- **Line 179:** validate_data_batch - Direct call: validate_data_batch
- **Line 251:** get_validator - Direct call: get_validator
- **Line 348:** EnhancedAPIAgnosticDataCollector - Direct call: EnhancedAPIAgnosticDataCollector
- **Line 467:** main - Direct call: main
- **Line 196:** validate_data_batch - Direct call: validate_data_batch
- **Line 22:** Path - Direct call: Path
- **Line 105:** get_exchange_mapper - Direct call: get_exchange_mapper
- **Line 278:** collect_incremental_data - Direct call: collect_incremental_data
- **Line 298:** timedelta - Direct call: timedelta
- **Line 301:** collect_data_for_period - Direct call: collect_data_for_period
- **Line 322:** detect_and_fill_gaps - Direct call: detect_and_fill_gaps
- **Line 355:** timedelta - Direct call: timedelta

---

## src/training/steps/data_collection/enhanced_data_collector.py

**Undefined function calls: 5**

- **Line 45:** get_validator - Direct call: get_validator
- **Line 425:** test_enhanced_collection - Direct call: test_enhanced_collection
- **Line 350:** DataType - Direct call: DataType
- **Line 416:** collect_all_data_with_validation - Direct call: collect_all_data_with_validation
- **Line 21:** Path - Direct call: Path

---

## src/training/steps/data_collection/enhanced_data_validation_framework.py

**Undefined function calls: 5**

- **Line 80:** field - Direct call: field
- **Line 100:** field - Direct call: field
- **Line 101:** field - Direct call: field
- **Line 823:** test_validation - Direct call: test_validation
- **Line 27:** Path - Direct call: Path

---

## src/training/steps/data_collection/enhanced_step01_5_data_converter.py

**Undefined function calls: 9**

- **Line 40:** get_validator - Direct call: get_validator
- **Line 41:** get_validator - Direct call: get_validator
- **Line 42:** get_validator - Direct call: get_validator
- **Line 43:** get_validator - Direct call: get_validator
- **Line 717:** main - Direct call: main
- **Line 260:** get_validator - Direct call: get_validator
- **Line 705:** run_enhanced_step01_5_data_converter - Direct call: run_enhanced_step01_5_data_converter
- **Line 21:** Path - Direct call: Path
- **Line 67:** __import__ - Direct call: __import__

---

## src/training/steps/data_collection/enhanced_step01_data_collection.py

**Undefined function calls: 11**

- **Line 136:** EnhancedDataCollectionManager - Direct call: EnhancedDataCollectionManager
- **Line 585:** run_enhanced_step01_data_collection - Direct call: run_enhanced_step01_data_collection
- **Line 597:** main - Direct call: main
- **Line 20:** Path - Direct call: Path
- **Line 47:** __import__ - Direct call: __import__
- **Line 265:** timedelta - Direct call: timedelta
- **Line 280:** timedelta - Direct call: timedelta
- **Line 294:** timedelta - Direct call: timedelta
- **Line 305:** timedelta - Direct call: timedelta
- **Line 175:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation
- **Line 335:** DataType - Direct call: DataType

---

## src/training/steps/data_collection/enhanced_step1_data_collection.py

**Undefined function calls: 14**

- **Line 92:** retry_with_backoff - Direct call: retry_with_backoff
- **Line 93:** categorize_errors - Direct call: categorize_errors
- **Line 39:** MemoryMonitor - Direct call: MemoryMonitor
- **Line 40:** EnhancedDataQualityValidator - Direct call: EnhancedDataQualityValidator
- **Line 198:** optimize_dataframe_dtypes - Direct call: optimize_dataframe_dtypes
- **Line 270:** Step1Config - Direct call: Step1Config
- **Line 15:** Path - Direct call: Path
- **Line 37:** Step1Config - Direct call: Step1Config
- **Line 39:** MemoryConfig - Direct call: MemoryConfig
- **Line 40:** QualityThresholds - Direct call: QualityThresholds
- **Line 284:** main - Direct call: main
- **Line 123:** NonRetryableError - Direct call: NonRetryableError
- **Line 200:** ThreadPoolExecutor - Direct call: ThreadPoolExecutor
- **Line 109:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation

---

## src/training/steps/data_collection/enhanced_validation_framework_with_decorators.py

**Undefined function calls: 23**

- **Line 861:** traced - Direct call: traced
- **Line 67:** field - Direct call: field
- **Line 104:** field - Direct call: field
- **Line 105:** field - Direct call: field
- **Line 145:** handles_errors - Direct call: handles_errors
- **Line 146:** traced - Direct call: traced
- **Line 204:** handles_errors - Direct call: handles_errors
- **Line 205:** traced - Direct call: traced
- **Line 206:** memory_efficient - Direct call: memory_efficient
- **Line 285:** handles_errors - Direct call: handles_errors
- **Line 306:** handles_errors - Direct call: handles_errors
- **Line 357:** handles_errors - Direct call: handles_errors
- **Line 376:** handles_errors - Direct call: handles_errors
- **Line 501:** handles_errors - Direct call: handles_errors
- **Line 538:** handles_errors - Direct call: handles_errors
- **Line 565:** handles_errors - Direct call: handles_errors
- **Line 585:** handles_errors - Direct call: handles_errors
- **Line 598:** handles_errors - Direct call: handles_errors
- **Line 616:** handles_errors - Direct call: handles_errors
- **Line 649:** handles_errors - Direct call: handles_errors
- **Line 916:** test_enhanced_validation - Direct call: test_enhanced_validation
- **Line 119:** get_exchange_mapper - Direct call: get_exchange_mapper
- **Line 24:** Path - Direct call: Path

---

## src/training/steps/data_collection/feature_engineering/step06_advanced_features.py

**Undefined function calls: 6**

- **Line 80:** handles_errors - Direct call: handles_errors
- **Line 95:** Path - Direct call: Path
- **Line 134:** Path - Direct call: Path
- **Line 43:** _TechnicalIndicatorCalculator - Direct call: _TechnicalIndicatorCalculator
- **Line 66:** Path - Direct call: Path
- **Line 240:** Path - Direct call: Path

---

## src/training/steps/data_collection/feature_engineering/step06_feature_engineering.py

**Undefined function calls: 3**

- **Line 193:** handles_errors - Direct call: handles_errors
- **Line 76:** nullcontext - Direct call: nullcontext
- **Line 727:** Path - Direct call: Path

---

## src/training/steps/data_collection/feature_engineering/step08_advanced_feature_selection.py

**Undefined function calls: 14**

- **Line 65:** jit - Direct call: jit
- **Line 94:** jit - Direct call: jit
- **Line 73:** prange - Direct call: prange
- **Line 83:** prange - Direct call: prange
- **Line 211:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 218:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 955:** run_step - Direct call: run_step
- **Line 140:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 142:** mutual_info_regression - Direct call: mutual_info_regression
- **Line 551:** BorutaPy - Direct call: BorutaPy
- **Line 536:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 543:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 739:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 745:** RandomForestRegressor - Direct call: RandomForestRegressor

---

## src/training/steps/data_collection/feature_engineering/step08_advanced_feature_selection_wrapper.py

**Undefined function calls: 2**

- **Line 68:** Step08AdvancedFeatureSelection - Direct call: Step08AdvancedFeatureSelection
- **Line 30:** get_logger - Direct call: get_logger

---

## src/training/steps/data_collection/integrated_data_quality_pipeline.py

**Undefined function calls: 18**

- **Line 388:** handles_errors - Direct call: handles_errors
- **Line 50:** traced - Direct call: traced
- **Line 51:** handles_errors - Direct call: handles_errors
- **Line 188:** traced - Direct call: traced
- **Line 207:** traced - Direct call: traced
- **Line 231:** traced - Direct call: traced
- **Line 256:** traced - Direct call: traced
- **Line 281:** traced - Direct call: traced
- **Line 313:** traced - Direct call: traced
- **Line 332:** traced - Direct call: traced
- **Line 34:** Path - Direct call: Path
- **Line 17:** Path - Direct call: Path
- **Line 45:** EnhancedDataQualityManager - Direct call: EnhancedDataQualityManager
- **Line 453:** run_integrated_pipeline - Direct call: run_integrated_pipeline
- **Line 473:** main - Direct call: main
- **Line 212:** run_step1 - Direct call: run_step1
- **Line 237:** run_step1_5 - Direct call: run_step1_5
- **Line 262:** run_step3 - Direct call: run_step3

---

## src/training/steps/data_collection/monitoring/pipeline_monitor.py

**Undefined function calls: 16**

- **Line 296:** Path - Direct call: Path
- **Line 409:** asdict - Direct call: asdict
- **Line 264:** format_datetime - Direct call: format_datetime
- **Line 265:** format_datetime - Direct call: format_datetime
- **Line 356:** format_datetime - Direct call: format_datetime
- **Line 357:** format_datetime - Direct call: format_datetime
- **Line 138:** format_datetime - Direct call: format_datetime
- **Line 148:** format_datetime - Direct call: format_datetime
- **Line 158:** format_datetime - Direct call: format_datetime
- **Line 264:** get_current_datetime - Direct call: get_current_datetime
- **Line 265:** get_current_datetime - Direct call: get_current_datetime
- **Line 356:** get_current_datetime - Direct call: get_current_datetime
- **Line 357:** get_current_datetime - Direct call: get_current_datetime
- **Line 138:** get_current_datetime - Direct call: get_current_datetime
- **Line 148:** get_current_datetime - Direct call: get_current_datetime
- **Line 158:** get_current_datetime - Direct call: get_current_datetime

---

## src/training/steps/data_collection/raw_data_quality_checker.py

**Undefined function calls: 17**

- **Line 152:** validates - Direct call: validates
- **Line 1357:** func - Direct call: func
- **Line 43:** func - Direct call: func
- **Line 64:** func - Direct call: func
- **Line 143:** func - Direct call: func
- **Line 832:** timedelta - Direct call: timedelta
- **Line 74:** func - Direct call: func
- **Line 91:** func - Direct call: func
- **Line 115:** func - Direct call: func
- **Line 1039:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation
- **Line 806:** timedelta - Direct call: timedelta
- **Line 1349:** func - Direct call: func
- **Line 1356:** func - Direct call: func
- **Line 378:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation
- **Line 886:** timedelta - Direct call: timedelta
- **Line 837:** timedelta - Direct call: timedelta
- **Line 894:** timedelta - Direct call: timedelta

---

## src/training/steps/data_collection/raw_data_quality_checker_refactored.py

**Undefined function calls: 5**

- **Line 32:** QualityMetricsCalculator - Direct call: QualityMetricsCalculator
- **Line 33:** DataIntegrityChecker - Direct call: DataIntegrityChecker
- **Line 34:** AnomalyDetector - Direct call: AnomalyDetector
- **Line 67:** func - Direct call: func
- **Line 110:** func - Direct call: func

---

## src/training/steps/data_collection/raw_data_quality_checker_simplified.py

**Undefined function calls: 15**

- **Line 48:** QualityCheckConfig - Direct call: QualityCheckConfig
- **Line 51:** DataPreprocessor - Direct call: DataPreprocessor
- **Line 52:** DataDownloader - Direct call: DataDownloader
- **Line 53:** DataIntegrityChecker - Direct call: DataIntegrityChecker
- **Line 54:** QualityMetricsCalculator - Direct call: QualityMetricsCalculator
- **Line 55:** AnomalyDetector - Direct call: AnomalyDetector
- **Line 56:** ErrorHandler - Direct call: ErrorHandler
- **Line 94:** ValidationResultBuilder - Direct call: ValidationResultBuilder
- **Line 345:** calculate_interval_statistics - Direct call: calculate_interval_statistics
- **Line 60:** StructureValidationStrategy - Direct call: StructureValidationStrategy
- **Line 61:** CompletenessValidationStrategy - Direct call: CompletenessValidationStrategy
- **Line 62:** IntegrityValidationStrategy - Direct call: IntegrityValidationStrategy
- **Line 63:** MarketSpecificValidationStrategy - Direct call: MarketSpecificValidationStrategy
- **Line 64:** FeatureEngineeringValidationStrategy - Direct call: FeatureEngineeringValidationStrategy
- **Line 294:** ValidationResultBuilder - Direct call: ValidationResultBuilder

---

## src/training/steps/data_collection/setup_step01_5_enhanced.py

**Undefined function calls: 4**

- **Line 107:** Path - Direct call: Path
- **Line 145:** test_basic_functionality - Direct call: test_basic_functionality
- **Line 128:** Step1_5DataConverterValidator - Direct call: Step1_5DataConverterValidator
- **Line 131:** HealthCheckSystem - Direct call: HealthCheckSystem

---

## src/training/steps/data_collection/standalone_enhanced_pipeline.py

**Undefined function calls: 4**

- **Line 282:** Path - Direct call: Path
- **Line 355:** main - Direct call: main
- **Line 336:** Path - Direct call: Path
- **Line 348:** run_standalone_enhanced_data_collection_pipeline - Direct call: run_standalone_enhanced_data_collection_pipeline

---

## src/training/steps/data_collection/standalone_main.py

**Undefined function calls: 4**

- **Line 123:** main - Direct call: main
- **Line 55:** run_standalone_enhanced_data_collection_pipeline - Direct call: run_standalone_enhanced_data_collection_pipeline
- **Line 16:** Path - Direct call: Path
- **Line 86:** Path - Direct call: Path

---

## src/training/steps/data_collection/step01_5_data_converter_validator.py

**Undefined function calls: 13**

- **Line 106:** __import__ - Direct call: __import__
- **Line 119:** __import__ - Direct call: __import__
- **Line 1635:** safe_json_load - Direct call: safe_json_load
- **Line 1856:** run_validator - Direct call: run_validator
- **Line 1872:** run_validator - Direct call: run_validator
- **Line 1888:** run_validator - Direct call: run_validator
- **Line 1919:** test_enhanced_validator - Direct call: test_enhanced_validator
- **Line 79:** Path - Direct call: Path
- **Line 180:** id - Direct call: id
- **Line 221:** func - Direct call: func
- **Line 219:** func - Direct call: func
- **Line 669:** __import__ - Direct call: __import__
- **Line 482:** check_function - Direct call: check_function

---

## src/training/steps/data_collection/step01_comprehensive_monitoring.py

**Undefined function calls: 38**

- **Line 107:** get_function_call_monitor - Direct call: get_function_call_monitor
- **Line 108:** get_function_validator - Direct call: get_function_validator
- **Line 109:** get_error_handler - Direct call: get_error_handler
- **Line 760:** validate_function_entry - Direct call: validate_function_entry
- **Line 761:** validate_function_output - Direct call: validate_function_output
- **Line 762:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 123:** validate_function_entry - Direct call: validate_function_entry
- **Line 124:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 148:** validate_function_entry - Direct call: validate_function_entry
- **Line 149:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 164:** validate_function_entry - Direct call: validate_function_entry
- **Line 165:** validate_function_output - Direct call: validate_function_output
- **Line 166:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 233:** validate_function_entry - Direct call: validate_function_entry
- **Line 234:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 274:** validate_function_entry - Direct call: validate_function_entry
- **Line 275:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 307:** validate_function_entry - Direct call: validate_function_entry
- **Line 308:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 348:** validate_function_entry - Direct call: validate_function_entry
- **Line 349:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 435:** validate_function_entry - Direct call: validate_function_entry
- **Line 436:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 467:** validate_function_entry - Direct call: validate_function_entry
- **Line 468:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 563:** validate_function_entry - Direct call: validate_function_entry
- **Line 564:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 578:** validate_function_entry - Direct call: validate_function_entry
- **Line 579:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 670:** validate_function_entry - Direct call: validate_function_entry
- **Line 671:** handle_errors_with_tracking - Direct call: handle_errors_with_tracking
- **Line 751:** log_function_call_summary - Direct call: log_function_call_summary
- **Line 752:** log_error_summary - Direct call: log_error_summary
- **Line 865:** run_comprehensive_step01 - Direct call: run_comprehensive_step01
- **Line 879:** main - Direct call: main
- **Line 83:** Path - Direct call: Path
- **Line 476:** timedelta - Direct call: timedelta
- **Line 284:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation

---

## src/training/steps/data_collection/step01_data_collection.py

**Undefined function calls: 16**

- **Line 574:** monitor_data_collection - Direct call: monitor_data_collection
- **Line 575:** handles_errors - Direct call: handles_errors
- **Line 7:** Path - Direct call: Path
- **Line 34:** callable - Direct call: callable
- **Line 691:** run_step - Direct call: run_step
- **Line 700:** main - Direct call: main
- **Line 348:** timedelta - Direct call: timedelta
- **Line 277:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation
- **Line 410:** validate_step1_file - Direct call: validate_step1_file
- **Line 408:** Path - Direct call: Path
- **Line 456:** Path - Direct call: Path
- **Line 609:** Path - Direct call: Path
- **Line 651:** mtc - Direct call: mtc
- **Line 619:** Path - Direct call: Path
- **Line 564:** Path - Direct call: Path
- **Line 460:** Path - Direct call: Path

---

## src/training/steps/data_collection/step01_data_collection_main.py

**Undefined function calls: 4**

- **Line 109:** main - Direct call: main
- **Line 57:** run_standalone_enhanced_data_collection_pipeline - Direct call: run_standalone_enhanced_data_collection_pipeline
- **Line 18:** Path - Direct call: Path
- **Line 79:** Path - Direct call: Path

---

## src/training/steps/data_collection/step01_data_collection_validator.py

**Undefined function calls: 3**

- **Line 559:** test_validator - Direct call: test_validator
- **Line 12:** Path - Direct call: Path
- **Line 557:** run_validator - Direct call: run_validator

---

## src/training/steps/data_collection/step01_enhanced_with_monitoring.py

**Undefined function calls: 9**

- **Line 140:** log_function_call_summary - Direct call: log_function_call_summary
- **Line 151:** log_function_call_summary - Direct call: log_function_call_summary
- **Line 223:** log_function_call_summary - Direct call: log_function_call_summary
- **Line 773:** log_function_call_summary - Direct call: log_function_call_summary
- **Line 796:** run_enhanced_step01_with_monitoring - Direct call: run_enhanced_step01_with_monitoring
- **Line 810:** main - Direct call: main
- **Line 82:** Path - Direct call: Path
- **Line 457:** timedelta - Direct call: timedelta
- **Line 274:** download_all_data_with_consolidation - Direct call: download_all_data_with_consolidation

---

## src/training/steps/data_collection/step02_5_sr_optimization_validator.py

**Undefined function calls: 12**

- **Line 401:** func - Direct call: func
- **Line 1051:** Path - Direct call: Path
- **Line 1119:** run_validation - Direct call: run_validation
- **Line 1180:** test_validator - Direct call: test_validator
- **Line 25:** Path - Direct call: Path
- **Line 245:** func - Direct call: func
- **Line 365:** func - Direct call: func
- **Line 628:** Path - Direct call: Path
- **Line 641:** Path - Direct call: Path
- **Line 748:** Path - Direct call: Path
- **Line 928:** Path - Direct call: Path
- **Line 127:** func - Direct call: func

---

## src/training/steps/data_collection/step02_data_reading.py

**Undefined function calls: 52**

- **Line 39:** field - Direct call: field
- **Line 40:** field - Direct call: field
- **Line 46:** field - Direct call: field
- **Line 48:** field - Direct call: field
- **Line 58:** field - Direct call: field
- **Line 59:** field - Direct call: field
- **Line 60:** field - Direct call: field
- **Line 61:** field - Direct call: field
- **Line 702:** create_fallback_logger - Direct call: create_fallback_logger
- **Line 704:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 705:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 706:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 707:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 708:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 709:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 710:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 711:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 712:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 724:** create_fallback_decorator - Direct call: create_fallback_decorator
- **Line 759:** create_fallback_logger - Direct call: create_fallback_logger
- **Line 807:** traced - Direct call: traced
- **Line 808:** validates - Direct call: validates
- **Line 861:** traced - Direct call: traced
- **Line 862:** validates - Direct call: validates
- **Line 1001:** traced - Direct call: traced
- **Line 1040:** traced - Direct call: traced
- **Line 1255:** run_step_enhanced - Direct call: run_step_enhanced
- **Line 1265:** test - Direct call: test
- **Line 680:** Path - Direct call: Path
- **Line 1096:** create_detailed_step_report - Direct call: create_detailed_step_report
- **Line 1097:** log_step_report - Direct call: log_step_report
- **Line 1102:** log_step_report - Direct call: log_step_report
- **Line 1104:** log_step_metrics - Direct call: log_step_metrics
- **Line 1125:** ensure_directory - Direct call: ensure_directory
- **Line 1164:** safe_json_dump - Direct call: safe_json_dump
- **Line 1263:** run_step_enhanced - Direct call: run_step_enhanced
- **Line 475:** func - Direct call: func
- **Line 667:** func - Direct call: func
- **Line 695:** func - Direct call: func
- **Line 815:** Path - Direct call: Path
- **Line 827:** safe_read_parquet - Direct call: safe_read_parquet
- **Line 1065:** Path - Direct call: Path
- **Line 1100:** log_step_dataframe_with_standardized_name - Direct call: log_step_dataframe_with_standardized_name
- **Line 642:** func - Direct call: func
- **Line 1008:** Path - Direct call: Path
- **Line 1063:** Path - Direct call: Path
- **Line 428:** _validate_function_inputs - Direct call: _validate_function_inputs
- **Line 437:** func - Direct call: func
- **Line 441:** _validate_function_outputs - Direct call: _validate_function_outputs
- **Line 1125:** Path - Direct call: Path
- **Line 433:** func - Direct call: func
- **Line 451:** _retry_function_call - Direct call: _retry_function_call

---

## src/training/steps/data_collection/step02_data_reading_validator.py

**Undefined function calls: 17**

- **Line 32:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 69:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 107:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 225:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 309:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 329:** ensure_directory - Direct call: ensure_directory
- **Line 442:** test - Direct call: test
- **Line 14:** Path - Direct call: Path
- **Line 252:** _validate_directory_structure - Direct call: _validate_directory_structure
- **Line 257:** _validate_data_files - Direct call: _validate_data_files
- **Line 264:** _validate_data_content - Direct call: _validate_data_content
- **Line 271:** Path - Direct call: Path
- **Line 432:** run_validator - Direct call: run_validator
- **Line 276:** safe_json_load - Direct call: safe_json_load
- **Line 437:** generate_validation_function_report - Direct call: generate_validation_function_report
- **Line 41:** Path - Direct call: Path
- **Line 329:** Path - Direct call: Path

---

## src/training/steps/data_collection/test_step02_enhanced_monitoring.py

**Undefined function calls: 20**

- **Line 50:** Path - Direct call: Path
- **Line 496:** main - Direct call: main
- **Line 25:** Path - Direct call: Path
- **Line 101:** DataReadingStep - Direct call: DataReadingStep
- **Line 154:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 167:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 179:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 235:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 300:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 192:** test_parent_function - Direct call: test_parent_function
- **Line 249:** test_failing_function - Direct call: test_failing_function
- **Line 313:** test_performance_function - Direct call: test_performance_function
- **Line 314:** test_performance_function - Direct call: test_performance_function
- **Line 315:** test_performance_function - Direct call: test_performance_function
- **Line 368:** run_validator - Direct call: run_validator
- **Line 371:** generate_validation_function_report - Direct call: generate_validation_function_report
- **Line 163:** test_child_function_1 - Direct call: test_child_function_1
- **Line 164:** test_child_function_2 - Direct call: test_child_function_2
- **Line 253:** test_failing_function - Direct call: test_failing_function
- **Line 426:** test_method - Direct call: test_method

---

## src/training/steps/data_collection/test_step02_simple.py

**Undefined function calls: 19**

- **Line 39:** FunctionCallMonitor - Direct call: FunctionCallMonitor
- **Line 383:** main - Direct call: main
- **Line 17:** Path - Direct call: Path
- **Line 52:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 110:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 123:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 135:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 191:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 257:** comprehensive_function_monitoring - Direct call: comprehensive_function_monitoring
- **Line 65:** test_function - Direct call: test_function
- **Line 148:** parent_function - Direct call: parent_function
- **Line 205:** failing_function - Direct call: failing_function
- **Line 270:** performance_function - Direct call: performance_function
- **Line 271:** performance_function - Direct call: performance_function
- **Line 272:** performance_function - Direct call: performance_function
- **Line 119:** child_function_1 - Direct call: child_function_1
- **Line 120:** child_function_2 - Direct call: child_function_2
- **Line 209:** failing_function - Direct call: failing_function
- **Line 323:** test_method - Direct call: test_method

---

## src/training/steps/data_collection/test_step02_standalone.py

**Undefined function calls: 27**

- **Line 40:** field - Direct call: field
- **Line 41:** field - Direct call: field
- **Line 47:** field - Direct call: field
- **Line 49:** field - Direct call: field
- **Line 59:** field - Direct call: field
- **Line 60:** field - Direct call: field
- **Line 61:** field - Direct call: field
- **Line 62:** field - Direct call: field
- **Line 1020:** main - Direct call: main
- **Line 476:** func - Direct call: func
- **Line 658:** func - Direct call: func
- **Line 702:** test_function - Direct call: test_function
- **Line 785:** parent_function - Direct call: parent_function
- **Line 842:** failing_function - Direct call: failing_function
- **Line 907:** performance_function - Direct call: performance_function
- **Line 908:** performance_function - Direct call: performance_function
- **Line 909:** performance_function - Direct call: performance_function
- **Line 633:** func - Direct call: func
- **Line 756:** child_function_1 - Direct call: child_function_1
- **Line 757:** child_function_2 - Direct call: child_function_2
- **Line 846:** failing_function - Direct call: failing_function
- **Line 960:** test_method - Direct call: test_method
- **Line 429:** _validate_function_inputs - Direct call: _validate_function_inputs
- **Line 438:** func - Direct call: func
- **Line 442:** _validate_function_outputs - Direct call: _validate_function_outputs
- **Line 434:** func - Direct call: func
- **Line 452:** _retry_function_call - Direct call: _retry_function_call

---

## src/training/steps/data_collection/utils/data_operations_utils.py

**Undefined function calls: 9**

- **Line 517:** format_datetime - Direct call: format_datetime
- **Line 553:** Path - Direct call: Path
- **Line 621:** Path - Direct call: Path
- **Line 517:** get_current_datetime - Direct call: get_current_datetime
- **Line 655:** safe_json_load - Direct call: safe_json_load
- **Line 711:** format_datetime - Direct call: format_datetime
- **Line 339:** format_datetime - Direct call: format_datetime
- **Line 711:** get_current_datetime - Direct call: get_current_datetime
- **Line 339:** get_current_datetime - Direct call: get_current_datetime

---

## src/training/steps/data_collection/validators/pipeline_validators.py

**Undefined function calls: 32**

- **Line 57:** monitor_step_execution - Direct call: monitor_step_execution
- **Line 192:** monitor_step_execution - Direct call: monitor_step_execution
- **Line 193:** validate_data_quality - Direct call: validate_data_quality
- **Line 363:** monitor_step_execution - Direct call: monitor_step_execution
- **Line 364:** validate_klines_data_quality - Direct call: validate_klines_data_quality
- **Line 581:** monitor_step_execution - Direct call: monitor_step_execution
- **Line 74:** Path - Direct call: Path
- **Line 217:** Path - Direct call: Path
- **Line 383:** Path - Direct call: Path
- **Line 170:** format_datetime - Direct call: format_datetime
- **Line 475:** format_datetime - Direct call: format_datetime
- **Line 609:** format_datetime - Direct call: format_datetime
- **Line 81:** format_datetime - Direct call: format_datetime
- **Line 170:** get_current_datetime - Direct call: get_current_datetime
- **Line 186:** format_datetime - Direct call: format_datetime
- **Line 226:** format_datetime - Direct call: format_datetime
- **Line 329:** format_datetime - Direct call: format_datetime
- **Line 357:** format_datetime - Direct call: format_datetime
- **Line 475:** get_current_datetime - Direct call: get_current_datetime
- **Line 491:** format_datetime - Direct call: format_datetime
- **Line 609:** get_current_datetime - Direct call: get_current_datetime
- **Line 622:** format_datetime - Direct call: format_datetime
- **Line 81:** get_current_datetime - Direct call: get_current_datetime
- **Line 186:** get_current_datetime - Direct call: get_current_datetime
- **Line 226:** get_current_datetime - Direct call: get_current_datetime
- **Line 329:** get_current_datetime - Direct call: get_current_datetime
- **Line 344:** format_datetime - Direct call: format_datetime
- **Line 357:** get_current_datetime - Direct call: get_current_datetime
- **Line 491:** get_current_datetime - Direct call: get_current_datetime
- **Line 622:** get_current_datetime - Direct call: get_current_datetime
- **Line 111:** Path - Direct call: Path
- **Line 344:** get_current_datetime - Direct call: get_current_datetime

---

## src/training/steps/feature_engineering/step06_advanced_features.py

**Undefined function calls: 5**

- **Line 80:** Path - Direct call: Path
- **Line 186:** WaveletFeaturePrecomputer - Direct call: WaveletFeaturePrecomputer
- **Line 207:** OptimizedTimeframeConfig - Direct call: OptimizedTimeframeConfig
- **Line 208:** EnhancedMultiTimeframeOptimizer - Direct call: EnhancedMultiTimeframeOptimizer
- **Line 115:** Path - Direct call: Path

---

## src/training/steps/market_analysis/combined_fractional_system.py

**Undefined function calls: 6**

- **Line 237:** handles_errors - Direct call: handles_errors
- **Line 40:** get_logger - Direct call: get_logger
- **Line 219:** FractionalTripleBarrierLabeling - Direct call: FractionalTripleBarrierLabeling
- **Line 223:** FractionalFeatureGenerator - Direct call: FractionalFeatureGenerator
- **Line 233:** get_logger - Direct call: get_logger
- **Line 437:** Path - Direct call: Path

---

## src/training/steps/market_analysis/cross_timeframe_interaction_features.py

**Undefined function calls: 4**

- **Line 126:** ThreadPoolExecutor - Direct call: ThreadPoolExecutor
- **Line 134:** as_completed - Direct call: as_completed
- **Line 420:** ThreadPoolExecutor - Direct call: ThreadPoolExecutor
- **Line 430:** as_completed - Direct call: as_completed

---

## src/training/steps/market_analysis/decorators.py

**Undefined function calls: 32**

- **Line 105:** wraps - Direct call: wraps
- **Line 110:** wraps - Direct call: wraps
- **Line 121:** wraps - Direct call: wraps
- **Line 126:** wraps - Direct call: wraps
- **Line 137:** wraps - Direct call: wraps
- **Line 142:** wraps - Direct call: wraps
- **Line 153:** wraps - Direct call: wraps
- **Line 158:** wraps - Direct call: wraps
- **Line 169:** wraps - Direct call: wraps
- **Line 174:** wraps - Direct call: wraps
- **Line 185:** wraps - Direct call: wraps
- **Line 190:** wraps - Direct call: wraps
- **Line 201:** wraps - Direct call: wraps
- **Line 206:** wraps - Direct call: wraps
- **Line 217:** wraps - Direct call: wraps
- **Line 222:** wraps - Direct call: wraps
- **Line 113:** func - Direct call: func
- **Line 129:** func - Direct call: func
- **Line 145:** func - Direct call: func
- **Line 161:** func - Direct call: func
- **Line 177:** func - Direct call: func
- **Line 193:** func - Direct call: func
- **Line 209:** func - Direct call: func
- **Line 225:** func - Direct call: func
- **Line 108:** func - Direct call: func
- **Line 124:** func - Direct call: func
- **Line 140:** func - Direct call: func
- **Line 156:** func - Direct call: func
- **Line 172:** func - Direct call: func
- **Line 188:** func - Direct call: func
- **Line 204:** func - Direct call: func
- **Line 220:** func - Direct call: func

---

## src/training/steps/market_analysis/dependencies.py

**Undefined function calls: 5**

- **Line 15:** Path - Direct call: Path
- **Line 90:** __import__ - Direct call: __import__
- **Line 132:** Path - Direct call: Path
- **Line 55:** __import__ - Direct call: __import__
- **Line 65:** __import__ - Direct call: __import__

---

## src/training/steps/market_analysis/enhanced_logging_metrics.py

**Undefined function calls: 2**

- **Line 130:** Path - Direct call: Path
- **Line 620:** asdict - Direct call: asdict

---

## src/training/steps/market_analysis/enhanced_market_analysis_orchestrator.py

**Undefined function calls: 63**

- **Line 115:** handles_errors - Direct call: handles_errors
- **Line 116:** traced - Direct call: traced
- **Line 118:** audit_log - Direct call: audit_log
- **Line 257:** handles_errors - Direct call: handles_errors
- **Line 258:** validates - Direct call: validates
- **Line 321:** handles_errors - Direct call: handles_errors
- **Line 322:** timeout - Direct call: timeout
- **Line 323:** retry - Direct call: retry
- **Line 324:** circuit_breaker - Direct call: circuit_breaker
- **Line 325:** traced - Direct call: traced
- **Line 405:** handles_errors - Direct call: handles_errors
- **Line 422:** handles_errors - Direct call: handles_errors
- **Line 453:** comprehensive_pipeline_protection - Direct call: comprehensive_pipeline_protection
- **Line 460:** handles_errors - Direct call: handles_errors
- **Line 461:** traced - Direct call: traced
- **Line 542:** comprehensive_pipeline_protection - Direct call: comprehensive_pipeline_protection
- **Line 549:** handles_errors - Direct call: handles_errors
- **Line 550:** traced - Direct call: traced
- **Line 583:** comprehensive_pipeline_protection - Direct call: comprehensive_pipeline_protection
- **Line 590:** handles_errors - Direct call: handles_errors
- **Line 591:** traced - Direct call: traced
- **Line 624:** comprehensive_pipeline_protection - Direct call: comprehensive_pipeline_protection
- **Line 631:** handles_errors - Direct call: handles_errors
- **Line 632:** traced - Direct call: traced
- **Line 798:** comprehensive_pipeline_protection - Direct call: comprehensive_pipeline_protection
- **Line 805:** handles_errors - Direct call: handles_errors
- **Line 806:** traced - Direct call: traced
- **Line 842:** comprehensive_pipeline_protection - Direct call: comprehensive_pipeline_protection
- **Line 849:** handles_errors - Direct call: handles_errors
- **Line 850:** traced - Direct call: traced
- **Line 883:** handles_errors - Direct call: handles_errors
- **Line 52:** get_logger - Direct call: get_logger
- **Line 53:** EnhancedPipelineLogger - Direct call: EnhancedPipelineLogger
- **Line 54:** DataQualityFramework - Direct call: DataQualityFramework
- **Line 55:** ValidatorOrchestrator - Direct call: ValidatorOrchestrator
- **Line 56:** StepDependencyValidator - Direct call: StepDependencyValidator
- **Line 57:** EnhancedStepValidator - Direct call: EnhancedStepValidator
- **Line 142:** set_correlation_id - Direct call: set_correlation_id
- **Line 144:** get_current_datetime - Direct call: get_current_datetime
- **Line 275:** Path - Direct call: Path
- **Line 978:** main - Direct call: main
- **Line 234:** get_current_datetime - Direct call: get_current_datetime
- **Line 309:** validate_data_quality - Direct call: validate_data_quality
- **Line 564:** RegimeDataSplittingStep - Direct call: RegimeDataSplittingStep
- **Line 605:** LabelingStep - Direct call: LabelingStep
- **Line 646:** FeatureEngineeringStep - Direct call: FeatureEngineeringStep
- **Line 820:** EnhancedMatrixOperationsStep - Direct call: EnhancedMatrixOperationsStep
- **Line 864:** AdvancedFeatureSelectionStep - Direct call: AdvancedFeatureSelectionStep
- **Line 965:** run_enhanced_market_analysis_pipeline - Direct call: run_enhanced_market_analysis_pipeline
- **Line 82:** RegimeDataSplittingValidator - Direct call: RegimeDataSplittingValidator
- **Line 89:** LabelingValidator - Direct call: LabelingValidator
- **Line 96:** FeatureEngineeringValidator - Direct call: FeatureEngineeringValidator
- **Line 103:** MatrixOperationsValidator - Direct call: MatrixOperationsValidator
- **Line 246:** get_current_datetime - Direct call: get_current_datetime
- **Line 288:** safe_file_exists - Direct call: safe_file_exists
- **Line 357:** step_func - Direct call: step_func
- **Line 475:** run_enhanced_step - Direct call: run_enhanced_step
- **Line 776:** Path - Direct call: Path
- **Line 894:** Path - Direct call: Path
- **Line 904:** format_datetime - Direct call: format_datetime
- **Line 904:** get_current_datetime - Direct call: get_current_datetime
- **Line 491:** Path - Direct call: Path
- **Line 662:** Path - Direct call: Path

---

## src/training/steps/market_analysis/enhanced_pipeline_decorators.py

**Undefined function calls: 21**

- **Line 67:** get_logger - Direct call: get_logger
- **Line 312:** get_logger - Direct call: get_logger
- **Line 462:** get_logger - Direct call: get_logger
- **Line 535:** Path - Direct call: Path
- **Line 628:** log_execution_time - Direct call: log_execution_time
- **Line 660:** example_function - Direct call: example_function
- **Line 99:** func - Direct call: func
- **Line 502:** func - Direct call: func
- **Line 556:** format_datetime - Direct call: format_datetime
- **Line 557:** get_correlation_id - Direct call: get_correlation_id
- **Line 626:** handles_errors - Direct call: handles_errors
- **Line 627:** traced - Direct call: traced
- **Line 629:** audit_log - Direct call: audit_log
- **Line 84:** func - Direct call: func
- **Line 483:** func - Direct call: func
- **Line 556:** get_current_datetime - Direct call: get_current_datetime
- **Line 376:** func - Direct call: func
- **Line 333:** func - Direct call: func
- **Line 366:** TimeoutError - Direct call: TimeoutError
- **Line 372:** func - Direct call: func
- **Line 329:** func - Direct call: func

---

## src/training/steps/market_analysis/enhanced_step_validator.py

**Undefined function calls: 18**

- **Line 140:** handles_errors - Direct call: handles_errors
- **Line 141:** traced - Direct call: traced
- **Line 226:** handles_errors - Direct call: handles_errors
- **Line 227:** traced - Direct call: traced
- **Line 312:** handles_errors - Direct call: handles_errors
- **Line 390:** handles_errors - Direct call: handles_errors
- **Line 483:** handles_errors - Direct call: handles_errors
- **Line 484:** traced - Direct call: traced
- **Line 566:** handles_errors - Direct call: handles_errors
- **Line 47:** get_logger - Direct call: get_logger
- **Line 48:** DataQualityFramework - Direct call: DataQualityFramework
- **Line 683:** main - Direct call: main
- **Line 183:** Path - Direct call: Path
- **Line 269:** Path - Direct call: Path
- **Line 192:** safe_file_exists - Direct call: safe_file_exists
- **Line 278:** safe_file_exists - Direct call: safe_file_exists
- **Line 411:** validator - Direct call: validator
- **Line 341:** Path - Direct call: Path

---

## src/training/steps/market_analysis/fractional_differentiation.py

**Undefined function calls: 5**

- **Line 170:** handles_errors - Direct call: handles_errors
- **Line 171:** traced - Direct call: traced
- **Line 38:** get_logger - Direct call: get_logger
- **Line 168:** get_logger - Direct call: get_logger
- **Line 98:** adfuller - Direct call: adfuller

---

## src/training/steps/market_analysis/fractional_feature_selector.py

**Undefined function calls: 6**

- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 65:** get_logger - Direct call: get_logger
- **Line 655:** Path - Direct call: Path
- **Line 742:** f_regression - Direct call: f_regression
- **Line 758:** mutual_info_regression - Direct call: mutual_info_regression
- **Line 774:** RandomForestRegressor - Direct call: RandomForestRegressor

---

## src/training/steps/market_analysis/hmm_clustering/step03_5_final_regime_clustering.py

**Undefined function calls: 38**

- **Line 932:** handles_errors - Direct call: handles_errors
- **Line 87:** handles_errors - Direct call: handles_errors
- **Line 107:** validates - Direct call: validates
- **Line 108:** handles_errors - Direct call: handles_errors
- **Line 149:** handles_errors - Direct call: handles_errors
- **Line 154:** validates - Direct call: validates
- **Line 210:** handles_errors - Direct call: handles_errors
- **Line 215:** monitor_feature_engineering - Direct call: monitor_feature_engineering
- **Line 216:** validates - Direct call: validates
- **Line 276:** handles_errors - Direct call: handles_errors
- **Line 344:** handles_errors - Direct call: handles_errors
- **Line 400:** handles_errors - Direct call: handles_errors
- **Line 442:** handles_errors - Direct call: handles_errors
- **Line 455:** handles_errors - Direct call: handles_errors
- **Line 475:** handles_errors - Direct call: handles_errors
- **Line 504:** handles_errors - Direct call: handles_errors
- **Line 526:** handles_errors - Direct call: handles_errors
- **Line 571:** handles_errors - Direct call: handles_errors
- **Line 599:** handles_errors - Direct call: handles_errors
- **Line 615:** handles_errors - Direct call: handles_errors
- **Line 633:** handles_errors - Direct call: handles_errors
- **Line 650:** handles_errors - Direct call: handles_errors
- **Line 684:** handles_errors - Direct call: handles_errors
- **Line 728:** handles_errors - Direct call: handles_errors
- **Line 785:** handles_errors - Direct call: handles_errors
- **Line 869:** handles_errors - Direct call: handles_errors
- **Line 883:** handles_errors - Direct call: handles_errors
- **Line 895:** handles_errors - Direct call: handles_errors
- **Line 914:** handles_errors - Direct call: handles_errors
- **Line 488:** StandardScaler - Direct call: StandardScaler
- **Line 517:** KMeans - Direct call: KMeans
- **Line 988:** run_step - Direct call: run_step
- **Line 17:** Path - Direct call: Path
- **Line 69:** Path - Direct call: Path
- **Line 797:** Path - Direct call: Path
- **Line 801:** Path - Direct call: Path
- **Line 168:** Path - Direct call: Path
- **Line 307:** StandardScaler - Direct call: StandardScaler

---

## src/training/steps/market_analysis/hmm_clustering/step03_bayesian_parameter_optimization.py

**Undefined function calls: 34**

- **Line 749:** validates - Direct call: validates
- **Line 750:** handles_errors - Direct call: handles_errors
- **Line 91:** handles_errors - Direct call: handles_errors
- **Line 120:** validates - Direct call: validates
- **Line 121:** handles_errors - Direct call: handles_errors
- **Line 418:** handles_errors - Direct call: handles_errors
- **Line 422:** validates - Direct call: validates
- **Line 478:** handles_errors - Direct call: handles_errors
- **Line 479:** monitor_feature_engineering - Direct call: monitor_feature_engineering
- **Line 480:** validates - Direct call: validates
- **Line 627:** handles_errors - Direct call: handles_errors
- **Line 683:** handles_errors - Direct call: handles_errors
- **Line 705:** handles_errors - Direct call: handles_errors
- **Line 86:** TPESampler - Direct call: TPESampler
- **Line 87:** MedianPruner - Direct call: MedianPruner
- **Line 238:** StandardScaler - Direct call: StandardScaler
- **Line 864:** run_bayesian_optimization - Direct call: run_bayesian_optimization
- **Line 16:** Path - Direct call: Path
- **Line 303:** KMeans - Direct call: KMeans
- **Line 634:** StandardScaler - Direct call: StandardScaler
- **Line 690:** Path - Direct call: Path
- **Line 712:** Path - Direct call: Path
- **Line 305:** GaussianMixture - Direct call: GaussianMixture
- **Line 307:** SpectralClustering - Direct call: SpectralClustering
- **Line 366:** silhouette_score - Direct call: silhouette_score
- **Line 371:** calinski_harabasz_score - Direct call: calinski_harabasz_score
- **Line 376:** davies_bouldin_score - Direct call: davies_bouldin_score
- **Line 436:** Path - Direct call: Path
- **Line 657:** KMeans - Direct call: KMeans
- **Line 668:** silhouette_score - Direct call: silhouette_score
- **Line 669:** calinski_harabasz_score - Direct call: calinski_harabasz_score
- **Line 670:** davies_bouldin_score - Direct call: davies_bouldin_score
- **Line 659:** GaussianMixture - Direct call: GaussianMixture
- **Line 661:** SpectralClustering - Direct call: SpectralClustering

---

## src/training/steps/market_analysis/hmm_clustering/step03_dynamic_regime_optimization.py

**Undefined function calls: 13**

- **Line 200:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 629:** adjusted_rand_score - Direct call: adjusted_rand_score
- **Line 431:** KMeans - Direct call: KMeans
- **Line 465:** KMeans - Direct call: KMeans
- **Line 510:** KMeans - Direct call: KMeans
- **Line 544:** KMeans - Direct call: KMeans
- **Line 128:** GaussianHMM - Direct call: GaussianHMM
- **Line 277:** GaussianHMM - Direct call: GaussianHMM
- **Line 332:** GaussianHMM - Direct call: GaussianHMM
- **Line 479:** KMeans - Direct call: KMeans
- **Line 514:** silhouette_score - Direct call: silhouette_score
- **Line 589:** KMeans - Direct call: KMeans
- **Line 214:** GaussianHMM - Direct call: GaussianHMM

---

## src/training/steps/market_analysis/hmm_clustering/step03_economic_significance_validator.py

**Undefined function calls: 2**

- **Line 134:** ks_2samp - Direct call: ks_2samp
- **Line 137:** mannwhitneyu - Direct call: mannwhitneyu

---

## src/training/steps/market_analysis/hmm_clustering/step03_enhanced_hmm_regime_discovery.py

**Undefined function calls: 33**

- **Line 777:** enhanced_validates - Direct call: enhanced_validates
- **Line 778:** enhanced_traced - Direct call: enhanced_traced
- **Line 779:** validates - Direct call: validates
- **Line 780:** handles_errors - Direct call: handles_errors
- **Line 113:** enhanced_validates - Direct call: enhanced_validates
- **Line 114:** enhanced_traced - Direct call: enhanced_traced
- **Line 115:** handles_errors - Direct call: handles_errors
- **Line 128:** enhanced_validates - Direct call: enhanced_validates
- **Line 129:** enhanced_traced - Direct call: enhanced_traced
- **Line 130:** validates - Direct call: validates
- **Line 131:** traced - Direct call: traced
- **Line 132:** handles_errors - Direct call: handles_errors
- **Line 274:** enhanced_validates - Direct call: enhanced_validates
- **Line 275:** enhanced_traced - Direct call: enhanced_traced
- **Line 276:** traced - Direct call: traced
- **Line 277:** handles_errors - Direct call: handles_errors
- **Line 332:** handles_errors - Direct call: handles_errors
- **Line 333:** validates - Direct call: validates
- **Line 405:** handles_errors - Direct call: handles_errors
- **Line 438:** handles_errors - Direct call: handles_errors
- **Line 463:** handles_errors - Direct call: handles_errors
- **Line 491:** handles_errors - Direct call: handles_errors
- **Line 511:** handles_errors - Direct call: handles_errors
- **Line 544:** handles_errors - Direct call: handles_errors
- **Line 660:** handles_errors - Direct call: handles_errors
- **Line 95:** OptimizedBayesianParameterOptimization - Direct call: OptimizedBayesianParameterOptimization
- **Line 98:** RegimeDiscoveryFeatureEngineer - Direct call: RegimeDiscoveryFeatureEngineer
- **Line 101:** EconomicSignificanceValidator - Direct call: EconomicSignificanceValidator
- **Line 104:** EnsembleClusteringRegimeDetector - Direct call: EnsembleClusteringRegimeDetector
- **Line 107:** EnhancedMLRegimeTransitionDetector - Direct call: EnhancedMLRegimeTransitionDetector
- **Line 921:** run_enhanced_step - Direct call: run_enhanced_step
- **Line 23:** Path - Direct call: Path
- **Line 296:** Path - Direct call: Path

---

## src/training/steps/market_analysis/hmm_clustering/step03_enhanced_ml_transition_detector.py

**Undefined function calls: 12**

- **Line 408:** compute_class_weight - Direct call: compute_class_weight
- **Line 420:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 427:** permutation_importance - Direct call: permutation_importance
- **Line 463:** train_test_split - Direct call: train_test_split
- **Line 604:** train_test_split - Direct call: train_test_split
- **Line 609:** StandardScaler - Direct call: StandardScaler
- **Line 473:** StandardScaler - Direct call: StandardScaler
- **Line 494:** f1_score - Direct call: f1_score
- **Line 631:** f1_score - Direct call: f1_score
- **Line 632:** roc_auc_score - Direct call: roc_auc_score
- **Line 536:** StandardScaler - Direct call: StandardScaler
- **Line 553:** f1_score - Direct call: f1_score

---

## src/training/steps/market_analysis/hmm_clustering/step03_ensemble_clustering.py

**Undefined function calls: 19**

- **Line 106:** RobustScaler - Direct call: RobustScaler
- **Line 99:** IncrementalPCA - Direct call: IncrementalPCA
- **Line 125:** ThreadPoolExecutor - Direct call: ThreadPoolExecutor
- **Line 128:** as_completed - Direct call: as_completed
- **Line 459:** silhouette_score - Direct call: silhouette_score
- **Line 460:** calinski_harabasz_score - Direct call: calinski_harabasz_score
- **Line 461:** davies_bouldin_score - Direct call: davies_bouldin_score
- **Line 644:** pdist - Direct call: pdist
- **Line 806:** silhouette_score - Direct call: silhouette_score
- **Line 807:** calinski_harabasz_score - Direct call: calinski_harabasz_score
- **Line 808:** davies_bouldin_score - Direct call: davies_bouldin_score
- **Line 212:** KMeans - Direct call: KMeans
- **Line 319:** silhouette_score - Direct call: silhouette_score
- **Line 353:** silhouette_score - Direct call: silhouette_score
- **Line 548:** KMeans - Direct call: KMeans
- **Line 668:** silhouette_score - Direct call: silhouette_score
- **Line 701:** silhouette_score - Direct call: silhouette_score
- **Line 256:** DBSCAN - Direct call: DBSCAN
- **Line 594:** DBSCAN - Direct call: DBSCAN

---

## src/training/steps/market_analysis/hmm_clustering/step03_hierarchical_regime_detection.py

**Undefined function calls: 4**

- **Line 214:** GaussianHMM - Direct call: GaussianHMM
- **Line 192:** KMeans - Direct call: KMeans
- **Line 230:** KMeans - Direct call: KMeans
- **Line 196:** silhouette_score - Direct call: silhouette_score

---

## src/training/steps/market_analysis/hmm_clustering/step03_hmm_regime_discovery.py

**Undefined function calls: 31**

- **Line 96:** create_fallback_logger - Direct call: create_fallback_logger
- **Line 130:** FeatureCalculator - Direct call: FeatureCalculator
- **Line 131:** RegimeAnalyzer - Direct call: RegimeAnalyzer
- **Line 18:** Path - Direct call: Path
- **Line 154:** EnhancedDataQualityManager - Direct call: EnhancedDataQualityManager
- **Line 583:** StandardScaler - Direct call: StandardScaler
- **Line 597:** Path - Direct call: Path
- **Line 626:** StandardScaler - Direct call: StandardScaler
- **Line 629:** KMeans - Direct call: KMeans
- **Line 866:** StandardScaler - Direct call: StandardScaler
- **Line 868:** DBSCAN - Direct call: DBSCAN
- **Line 1598:** identify_market_condition_columns - Direct call: identify_market_condition_columns
- **Line 1603:** HMMRegimeOptimizer - Direct call: HMMRegimeOptimizer
- **Line 1687:** ensure_directory - Direct call: ensure_directory
- **Line 1693:** safe_json_dump - Direct call: safe_json_dump
- **Line 1701:** ensure_directory - Direct call: ensure_directory
- **Line 1763:** run_step - Direct call: run_step
- **Line 1773:** main - Direct call: main
- **Line 490:** Path - Direct call: Path
- **Line 650:** Path - Direct call: Path
- **Line 1156:** silhouette_score - Direct call: silhouette_score
- **Line 1160:** calinski_harabasz_score - Direct call: calinski_harabasz_score
- **Line 1164:** davies_bouldin_score - Direct call: davies_bouldin_score
- **Line 1623:** Path - Direct call: Path
- **Line 443:** run_step1 - Direct call: run_step1
- **Line 454:** run_step1_5 - Direct call: run_step1_5
- **Line 1673:** Path - Direct call: Path
- **Line 1687:** Path - Direct call: Path
- **Line 1701:** Path - Direct call: Path
- **Line 948:** pdf_func - Direct call: pdf_func
- **Line 1588:** Path - Direct call: Path

---

## src/training/steps/market_analysis/hmm_clustering/step03_hmm_regime_discovery_1h.py

**Undefined function calls: 12**

- **Line 35:** handles_errors - Direct call: handles_errors
- **Line 36:** traced - Direct call: traced
- **Line 113:** StandardScaler - Direct call: StandardScaler
- **Line 227:** ensure_directory - Direct call: ensure_directory
- **Line 231:** safe_json_dump - Direct call: safe_json_dump
- **Line 246:** main - Direct call: main
- **Line 15:** Path - Direct call: Path
- **Line 63:** Path - Direct call: Path
- **Line 203:** Path - Direct call: Path
- **Line 226:** Path - Direct call: Path
- **Line 243:** run_enhanced_regime_discovery - Direct call: run_enhanced_regime_discovery
- **Line 70:** Path - Direct call: Path

---

## src/training/steps/market_analysis/hmm_clustering/step03_hmm_regime_discovery_validator.py

**Undefined function calls: 4**

- **Line 222:** run_validator - Direct call: run_validator
- **Line 158:** Path - Direct call: Path
- **Line 80:** Path - Direct call: Path
- **Line 130:** safe_json_load - Direct call: safe_json_load

---

## src/training/steps/market_analysis/hmm_clustering/step03_microservices_regime_discovery.py

**Undefined function calls: 6**

- **Line 112:** OptimizedBayesianParameterOptimization - Direct call: OptimizedBayesianParameterOptimization
- **Line 144:** RegimeDiscoveryFeatureEngineer - Direct call: RegimeDiscoveryFeatureEngineer
- **Line 175:** EnsembleClusteringRegimeDetector - Direct call: EnsembleClusteringRegimeDetector
- **Line 211:** EconomicSignificanceValidator - Direct call: EconomicSignificanceValidator
- **Line 242:** EnhancedMLRegimeTransitionDetector - Direct call: EnhancedMLRegimeTransitionDetector
- **Line 273:** RegimePersistenceForecaster - Direct call: RegimePersistenceForecaster

---

## src/training/steps/market_analysis/hmm_clustering/step03_ml_transition_detector.py

**Undefined function calls: 9**

- **Line 69:** train_test_split - Direct call: train_test_split
- **Line 337:** compute_class_weight - Direct call: compute_class_weight
- **Line 349:** StandardScaler - Direct call: StandardScaler
- **Line 390:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 398:** GradientBoostingClassifier - Direct call: GradientBoostingClassifier
- **Line 428:** confusion_matrix - Direct call: confusion_matrix
- **Line 436:** roc_auc_score - Direct call: roc_auc_score
- **Line 405:** LogisticRegression - Direct call: LogisticRegression
- **Line 411:** MLPClassifier - Direct call: MLPClassifier

---

## src/training/steps/market_analysis/hmm_clustering/step03_optimized_bayesian_optimization.py

**Undefined function calls: 27**

- **Line 207:** KMeans - Direct call: KMeans
- **Line 243:** DBSCAN - Direct call: DBSCAN
- **Line 546:** KMeans - Direct call: KMeans
- **Line 564:** DBSCAN - Direct call: DBSCAN
- **Line 648:** KMeans - Direct call: KMeans
- **Line 675:** DBSCAN - Direct call: DBSCAN
- **Line 290:** TPESampler - Direct call: TPESampler
- **Line 291:** SuccessiveHalvingPruner - Direct call: SuccessiveHalvingPruner
- **Line 330:** CmaEsSampler - Direct call: CmaEsSampler
- **Line 331:** MedianPruner - Direct call: MedianPruner
- **Line 397:** TPESampler - Direct call: TPESampler
- **Line 398:** MedianPruner - Direct call: MedianPruner
- **Line 556:** silhouette_score - Direct call: silhouette_score
- **Line 571:** silhouette_score - Direct call: silhouette_score
- **Line 627:** pdist - Direct call: pdist
- **Line 658:** silhouette_score - Direct call: silhouette_score
- **Line 659:** calinski_harabasz_score - Direct call: calinski_harabasz_score
- **Line 660:** davies_bouldin_score - Direct call: davies_bouldin_score
- **Line 682:** silhouette_score - Direct call: silhouette_score
- **Line 683:** calinski_harabasz_score - Direct call: calinski_harabasz_score
- **Line 684:** davies_bouldin_score - Direct call: davies_bouldin_score
- **Line 865:** KMeans - Direct call: KMeans
- **Line 898:** DBSCAN - Direct call: DBSCAN
- **Line 801:** silhouette_score - Direct call: silhouette_score
- **Line 833:** silhouette_score - Direct call: silhouette_score
- **Line 876:** silhouette_score - Direct call: silhouette_score
- **Line 909:** silhouette_score - Direct call: silhouette_score

---

## src/training/steps/market_analysis/hmm_clustering/step03_parameter_optimization.py

**Undefined function calls: 36**

- **Line 722:** handles_errors - Direct call: handles_errors
- **Line 61:** handles_errors - Direct call: handles_errors
- **Line 81:** validates - Direct call: validates
- **Line 82:** handles_errors - Direct call: handles_errors
- **Line 126:** handles_errors - Direct call: handles_errors
- **Line 130:** validates - Direct call: validates
- **Line 186:** handles_errors - Direct call: handles_errors
- **Line 187:** monitor_feature_engineering - Direct call: monitor_feature_engineering
- **Line 188:** validates - Direct call: validates
- **Line 259:** handles_errors - Direct call: handles_errors
- **Line 276:** handles_errors - Direct call: handles_errors
- **Line 295:** handles_errors - Direct call: handles_errors
- **Line 309:** handles_errors - Direct call: handles_errors
- **Line 345:** handles_errors - Direct call: handles_errors
- **Line 360:** handles_errors - Direct call: handles_errors
- **Line 379:** handles_errors - Direct call: handles_errors
- **Line 393:** handles_errors - Direct call: handles_errors
- **Line 429:** handles_errors - Direct call: handles_errors
- **Line 445:** handles_errors - Direct call: handles_errors
- **Line 463:** handles_errors - Direct call: handles_errors
- **Line 483:** handles_errors - Direct call: handles_errors
- **Line 525:** handles_errors - Direct call: handles_errors
- **Line 545:** handles_errors - Direct call: handles_errors
- **Line 566:** handles_errors - Direct call: handles_errors
- **Line 585:** handles_errors - Direct call: handles_errors
- **Line 605:** handles_errors - Direct call: handles_errors
- **Line 629:** handles_errors - Direct call: handles_errors
- **Line 676:** handles_errors - Direct call: handles_errors
- **Line 686:** handles_errors - Direct call: handles_errors
- **Line 694:** handles_errors - Direct call: handles_errors
- **Line 709:** handles_errors - Direct call: handles_errors
- **Line 776:** run_step - Direct call: run_step
- **Line 17:** Path - Direct call: Path
- **Line 613:** Path - Direct call: Path
- **Line 637:** Path - Direct call: Path
- **Line 144:** Path - Direct call: Path

---

## src/training/steps/market_analysis/hmm_clustering/step03_realtime_streaming_pipeline.py

**Undefined function calls: 3**

- **Line 241:** RegimePersistenceForecaster - Direct call: RegimePersistenceForecaster
- **Line 519:** callback - Direct call: callback
- **Line 517:** callback - Direct call: callback

---

## src/training/steps/market_analysis/hmm_clustering/step03_regime_persistence_forecasting.py

**Undefined function calls: 3**

- **Line 534:** survival_func - Direct call: survival_func
- **Line 397:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 435:** EnhancedRegimePredictor - Direct call: EnhancedRegimePredictor

---

## src/training/steps/market_analysis/hmm_clustering/step03_streaming_regime_discovery.py

**Undefined function calls: 4**

- **Line 33:** deque - Direct call: deque
- **Line 34:** deque - Direct call: deque
- **Line 229:** KMeans - Direct call: KMeans
- **Line 80:** MemoryError - Direct call: MemoryError

---

## src/training/steps/market_analysis/hmm_components.py

**Undefined function calls: 3**

- **Line 44:** StandardScaler - Direct call: StandardScaler
- **Line 70:** StandardScaler - Direct call: StandardScaler
- **Line 72:** KMeans - Direct call: KMeans

---

## src/training/steps/market_analysis/hmm_feature_enhancer.py

**Undefined function calls: 2**

- **Line 15:** traced - Direct call: traced
- **Line 16:** validates - Direct call: validates

---

## src/training/steps/market_analysis/integrate_regime_processing.py

**Undefined function calls: 6**

- **Line 46:** Path - Direct call: Path
- **Line 112:** main - Direct call: main
- **Line 15:** Path - Direct call: Path
- **Line 95:** Path - Direct call: Path
- **Line 37:** Path - Direct call: Path
- **Line 38:** Path - Direct call: Path

---

## src/training/steps/market_analysis/labeling_components.py

**Undefined function calls: 4**

- **Line 80:** RegimeAwareTripleBarrierLabeling - Direct call: RegimeAwareTripleBarrierLabeling
- **Line 44:** RegimeSpecificTripleBarrierOptimizer - Direct call: RegimeSpecificTripleBarrierOptimizer
- **Line 139:** MetaLabelingSystem - Direct call: MetaLabelingSystem
- **Line 35:** get_regime_column - Direct call: get_regime_column

---

## src/training/steps/market_analysis/model_persistence_components/metadata_tracker.py

**Undefined function calls: 2**

- **Line 32:** handles_errors - Direct call: handles_errors
- **Line 194:** handles_errors - Direct call: handles_errors

---

## src/training/steps/market_analysis/model_persistence_components/model_persistence_step.py

**Undefined function calls: 8**

- **Line 62:** handles_errors - Direct call: handles_errors
- **Line 29:** ModelSerializer - Direct call: ModelSerializer
- **Line 30:** VersionManager - Direct call: VersionManager
- **Line 31:** MetadataTracker - Direct call: MetadataTracker
- **Line 32:** ModelRegistry - Direct call: ModelRegistry
- **Line 256:** Path - Direct call: Path
- **Line 300:** Path - Direct call: Path
- **Line 235:** Path - Direct call: Path

---

## src/training/steps/market_analysis/model_persistence_components/model_registry.py

**Undefined function calls: 7**

- **Line 36:** handles_errors - Direct call: handles_errors
- **Line 98:** handles_errors - Direct call: handles_errors
- **Line 127:** handles_errors - Direct call: handles_errors
- **Line 139:** handles_errors - Direct call: handles_errors
- **Line 168:** handles_errors - Direct call: handles_errors
- **Line 193:** handles_errors - Direct call: handles_errors
- **Line 21:** Path - Direct call: Path

---

## src/training/steps/market_analysis/model_persistence_components/model_serializer.py

**Undefined function calls: 8**

- **Line 39:** handles_errors - Direct call: handles_errors
- **Line 176:** handles_errors - Direct call: handles_errors
- **Line 36:** Path - Direct call: Path
- **Line 187:** Path - Direct call: Path
- **Line 59:** handler - Direct call: handler
- **Line 228:** loader - Direct call: loader
- **Line 132:** convert_sklearn - Direct call: convert_sklearn
- **Line 130:** FloatTensorType - Direct call: FloatTensorType

---

## src/training/steps/market_analysis/model_persistence_components/version_manager.py

**Undefined function calls: 6**

- **Line 37:** handles_errors - Direct call: handles_errors
- **Line 129:** handles_errors - Direct call: handles_errors
- **Line 154:** handles_errors - Direct call: handles_errors
- **Line 169:** handles_errors - Direct call: handles_errors
- **Line 185:** handles_errors - Direct call: handles_errors
- **Line 23:** Path - Direct call: Path

---

## src/training/steps/market_analysis/monitoring/error_handler.py

**Undefined function calls: 4**

- **Line 246:** wraps - Direct call: wraps
- **Line 260:** wraps - Direct call: wraps
- **Line 263:** func - Direct call: func
- **Line 249:** func - Direct call: func

---

## src/training/steps/market_analysis/monitoring/function_call_monitor.py

**Undefined function calls: 17**

- **Line 35:** field - Direct call: field
- **Line 36:** field - Direct call: field
- **Line 42:** field - Direct call: field
- **Line 43:** field - Direct call: field
- **Line 44:** field - Direct call: field
- **Line 57:** field - Direct call: field
- **Line 58:** field - Direct call: field
- **Line 59:** field - Direct call: field
- **Line 60:** field - Direct call: field
- **Line 261:** wraps - Direct call: wraps
- **Line 276:** wraps - Direct call: wraps
- **Line 298:** wraps - Direct call: wraps
- **Line 304:** wraps - Direct call: wraps
- **Line 308:** func - Direct call: func
- **Line 280:** func - Direct call: func
- **Line 302:** func - Direct call: func
- **Line 265:** func - Direct call: func

---

## src/training/steps/market_analysis/monitoring/performance_monitor.py

**Undefined function calls: 4**

- **Line 442:** wraps - Direct call: wraps
- **Line 460:** wraps - Direct call: wraps
- **Line 469:** func - Direct call: func
- **Line 451:** func - Direct call: func

---

## src/training/steps/market_analysis/monitoring/validation_framework.py

**Undefined function calls: 7**

- **Line 758:** wraps - Direct call: wraps
- **Line 793:** wraps - Direct call: wraps
- **Line 194:** Path - Direct call: Path
- **Line 809:** func - Direct call: func
- **Line 774:** func - Direct call: func
- **Line 598:** rule - Direct call: rule
- **Line 640:** rule - Direct call: rule

---

## src/training/steps/market_analysis/multi_timeframe_training/multi_timeframe_training_manager.py

**Undefined function calls: 46**

- **Line 1295:** handles_errors - Direct call: handles_errors
- **Line 67:** handles_errors - Direct call: handles_errors
- **Line 119:** handles_errors - Direct call: handles_errors
- **Line 164:** handles_errors - Direct call: handles_errors
- **Line 211:** handles_errors - Direct call: handles_errors
- **Line 244:** handles_errors - Direct call: handles_errors
- **Line 273:** handles_errors - Direct call: handles_errors
- **Line 302:** handles_errors - Direct call: handles_errors
- **Line 331:** handles_errors - Direct call: handles_errors
- **Line 458:** handles_errors - Direct call: handles_errors
- **Line 541:** handles_errors - Direct call: handles_errors
- **Line 593:** handles_errors - Direct call: handles_errors
- **Line 644:** handles_errors - Direct call: handles_errors
- **Line 704:** handles_errors - Direct call: handles_errors
- **Line 755:** handles_errors - Direct call: handles_errors
- **Line 1097:** handles_errors - Direct call: handles_errors
- **Line 1129:** handles_errors - Direct call: handles_errors
- **Line 1160:** handles_errors - Direct call: handles_errors
- **Line 1216:** handles_errors - Direct call: handles_errors
- **Line 179:** invalid - Direct call: invalid
- **Line 184:** invalid - Direct call: invalid
- **Line 208:** error - Direct call: error
- **Line 455:** error - Direct call: error
- **Line 537:** error - Direct call: error
- **Line 578:** invalid - Direct call: invalid
- **Line 582:** invalid - Direct call: invalid
- **Line 641:** error - Direct call: error
- **Line 701:** error - Direct call: error
- **Line 752:** error - Direct call: error
- **Line 816:** error - Direct call: error
- **Line 834:** error - Direct call: error
- **Line 851:** error - Direct call: error
- **Line 868:** error - Direct call: error
- **Line 885:** error - Direct call: error
- **Line 903:** error - Direct call: error
- **Line 920:** error - Direct call: error
- **Line 937:** error - Direct call: error
- **Line 954:** validation_error - Direct call: validation_error
- **Line 972:** error - Direct call: error
- **Line 989:** error - Direct call: error
- **Line 1006:** error - Direct call: error
- **Line 1023:** error - Direct call: error
- **Line 1041:** error - Direct call: error
- **Line 1058:** error - Direct call: error
- **Line 1075:** error - Direct call: error
- **Line 1093:** validation_error - Direct call: validation_error

---

## src/training/steps/market_analysis/precompute_wavelet_features.py

**Undefined function calls: 9**

- **Line 496:** main - Direct call: main
- **Line 65:** VectorizedAdvancedFeatureEngineering - Direct call: VectorizedAdvancedFeatureEngineering
- **Line 69:** WaveletFeatureCache - Direct call: WaveletFeatureCache
- **Line 126:** Path - Direct call: Path
- **Line 135:** ParquetDatasetManager - Direct call: ParquetDatasetManager
- **Line 136:** ohlcv_columns - Direct call: ohlcv_columns
- **Line 159:** log_io_operation - Direct call: log_io_operation
- **Line 144:** log_io_operation - Direct call: log_io_operation
- **Line 154:** log_io_operation - Direct call: log_io_operation

---

## src/training/steps/market_analysis/progress_monitor.py

**Undefined function calls: 2**

- **Line 133:** timedelta - Direct call: timedelta
- **Line 288:** func - Direct call: func

---

## src/training/steps/market_analysis/regime_continuity_decorator.py

**Undefined function calls: 6**

- **Line 232:** func - Direct call: func
- **Line 268:** Path - Direct call: Path
- **Line 185:** _aggregate_regime_results - Direct call: _aggregate_regime_results
- **Line 146:** func - Direct call: func
- **Line 64:** _execute_per_regime_step - Direct call: _execute_per_regime_step
- **Line 68:** _execute_standard_step - Direct call: _execute_standard_step

---

## src/training/steps/market_analysis/regime_continuity_manager.py

**Undefined function calls: 17**

- **Line 30:** get_logger - Direct call: get_logger
- **Line 118:** traced - Direct call: traced
- **Line 162:** traced - Direct call: traced
- **Line 198:** traced - Direct call: traced
- **Line 281:** traced - Direct call: traced
- **Line 353:** traced - Direct call: traced
- **Line 393:** traced - Direct call: traced
- **Line 417:** traced - Direct call: traced
- **Line 472:** traced - Direct call: traced
- **Line 526:** traced - Direct call: traced
- **Line 89:** get_logger - Direct call: get_logger
- **Line 503:** safe_json_dump - Direct call: safe_json_dump
- **Line 519:** safe_json_dump - Direct call: safe_json_dump
- **Line 183:** Path - Direct call: Path
- **Line 489:** Path - Direct call: Path
- **Line 496:** asdict - Direct call: asdict
- **Line 511:** asdict - Direct call: asdict

---

## src/training/steps/market_analysis/regime_continuity_validator.py

**Undefined function calls: 12**

- **Line 59:** traced - Direct call: traced
- **Line 149:** traced - Direct call: traced
- **Line 221:** traced - Direct call: traced
- **Line 310:** traced - Direct call: traced
- **Line 413:** traced - Direct call: traced
- **Line 500:** traced - Direct call: traced
- **Line 611:** traced - Direct call: traced
- **Line 46:** get_logger - Direct call: get_logger
- **Line 243:** Path - Direct call: Path
- **Line 332:** Path - Direct call: Path
- **Line 435:** Path - Direct call: Path
- **Line 715:** Path - Direct call: Path

---

## src/training/steps/market_analysis/regime_handler.py

**Undefined function calls: 15**

- **Line 36:** traced - Direct call: traced
- **Line 82:** traced - Direct call: traced
- **Line 100:** traced - Direct call: traced
- **Line 171:** traced - Direct call: traced
- **Line 284:** traced - Direct call: traced
- **Line 352:** traced - Direct call: traced
- **Line 31:** get_logger - Direct call: get_logger
- **Line 279:** processing_func - Direct call: processing_func
- **Line 310:** ensure_directory - Direct call: ensure_directory
- **Line 388:** safe_json_load - Direct call: safe_json_load
- **Line 57:** Path - Direct call: Path
- **Line 74:** safe_json_load - Direct call: safe_json_load
- **Line 377:** Path - Direct call: Path
- **Line 310:** Path - Direct call: Path
- **Line 400:** safe_json_load - Direct call: safe_json_load

---

## src/training/steps/market_analysis/regime_processing_decorator.py

**Undefined function calls: 2**

- **Line 232:** processing_func - Direct call: processing_func
- **Line 70:** func - Direct call: func

---

## src/training/steps/market_analysis/step03_hmm_clustering.py

**Undefined function calls: 14**

- **Line 216:** validates - Direct call: validates
- **Line 217:** traced - Direct call: traced
- **Line 67:** validates - Direct call: validates
- **Line 68:** traced - Direct call: traced
- **Line 82:** validates - Direct call: validates
- **Line 83:** traced - Direct call: traced
- **Line 181:** validates - Direct call: validates
- **Line 182:** traced - Direct call: traced
- **Line 344:** run_step - Direct call: run_step
- **Line 372:** main - Direct call: main
- **Line 24:** Path - Direct call: Path
- **Line 128:** run_enhanced_step - Direct call: run_enhanced_step
- **Line 280:** run_validator - Direct call: run_validator
- **Line 146:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step03_market_analysis_main.py

**Undefined function calls: 4**

- **Line 127:** main - Direct call: main
- **Line 69:** run_enhanced_market_analysis_pipeline - Direct call: run_enhanced_market_analysis_pipeline
- **Line 18:** Path - Direct call: Path
- **Line 91:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step04_5_triple_barrier_method_validator.py

**Undefined function calls: 6**

- **Line 17:** traced - Direct call: traced
- **Line 18:** validates - Direct call: validates
- **Line 144:** test - Direct call: test
- **Line 11:** Path - Direct call: Path
- **Line 142:** run_validator - Direct call: run_validator
- **Line 101:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step04_regime_data_splitting.py

**Undefined function calls: 22**

- **Line 614:** traced - Direct call: traced
- **Line 615:** validates - Direct call: validates
- **Line 616:** handles_errors - Direct call: handles_errors
- **Line 617:** cached - Direct call: cached
- **Line 618:** log_execution_time - Direct call: log_execution_time
- **Line 410:** traced - Direct call: traced
- **Line 411:** validates - Direct call: validates
- **Line 412:** cached - Direct call: cached
- **Line 673:** test - Direct call: test
- **Line 50:** Path - Direct call: Path
- **Line 70:** func - Direct call: func
- **Line 279:** func - Direct call: func
- **Line 535:** safe_json_dump - Direct call: safe_json_dump
- **Line 548:** ensure_directory - Direct call: ensure_directory
- **Line 608:** safe_json_dump - Direct call: safe_json_dump
- **Line 671:** run_step - Direct call: run_step
- **Line 266:** func - Direct call: func
- **Line 455:** Path - Direct call: Path
- **Line 459:** Path - Direct call: Path
- **Line 460:** Path - Direct call: Path
- **Line 607:** Path - Direct call: Path
- **Line 548:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step04_regime_data_splitting_validator.py

**Undefined function calls: 7**

- **Line 420:** run_validator - Direct call: run_validator
- **Line 238:** Path - Direct call: Path
- **Line 276:** Path - Direct call: Path
- **Line 84:** Path - Direct call: Path
- **Line 320:** Path - Direct call: Path
- **Line 321:** Path - Direct call: Path
- **Line 322:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step05_labeling_original_backup.py

**Undefined function calls: 55**

- **Line 115:** field - Direct call: field
- **Line 116:** field - Direct call: field
- **Line 122:** field - Direct call: field
- **Line 123:** field - Direct call: field
- **Line 124:** field - Direct call: field
- **Line 136:** field - Direct call: field
- **Line 137:** field - Direct call: field
- **Line 138:** field - Direct call: field
- **Line 139:** field - Direct call: field
- **Line 2189:** traced - Direct call: traced
- **Line 2190:** validates - Direct call: validates
- **Line 2191:** handles_errors - Direct call: handles_errors
- **Line 2192:** cached - Direct call: cached
- **Line 2193:** log_execution_time - Direct call: log_execution_time
- **Line 342:** wraps - Direct call: wraps
- **Line 357:** wraps - Direct call: wraps
- **Line 378:** wraps - Direct call: wraps
- **Line 384:** wraps - Direct call: wraps
- **Line 624:** wraps - Direct call: wraps
- **Line 638:** wraps - Direct call: wraps
- **Line 1077:** wraps - Direct call: wraps
- **Line 1095:** wraps - Direct call: wraps
- **Line 1856:** wraps - Direct call: wraps
- **Line 1891:** wraps - Direct call: wraps
- **Line 2931:** test - Direct call: test
- **Line 62:** Path - Direct call: Path
- **Line 388:** func - Direct call: func
- **Line 2152:** RegimeSpecificTripleBarrierOptimizer - Direct call: RegimeSpecificTripleBarrierOptimizer
- **Line 2203:** ensure_directory - Direct call: ensure_directory
- **Line 2270:** safe_json_dump - Direct call: safe_json_dump
- **Line 2681:** create_detailed_step_report - Direct call: create_detailed_step_report
- **Line 2682:** log_step_report - Direct call: log_step_report
- **Line 2690:** log_step_metrics - Direct call: log_step_metrics
- **Line 2838:** RegimeAwareTripleBarrierLabeling - Direct call: RegimeAwareTripleBarrierLabeling
- **Line 2928:** run_step - Direct call: run_step
- **Line 361:** func - Direct call: func
- **Line 382:** func - Direct call: func
- **Line 641:** func - Direct call: func
- **Line 1104:** func - Direct call: func
- **Line 1293:** Path - Direct call: Path
- **Line 1907:** func - Direct call: func
- **Line 2138:** get_regime_column - Direct call: get_regime_column
- **Line 2222:** ensure_regime_labels - Direct call: ensure_regime_labels
- **Line 2229:** get_regime_column - Direct call: get_regime_column
- **Line 2685:** log_step_dataframe_with_standardized_name - Direct call: log_step_dataframe_with_standardized_name
- **Line 2688:** log_step_artifact_with_standardized_name - Direct call: log_step_artifact_with_standardized_name
- **Line 346:** func - Direct call: func
- **Line 627:** func - Direct call: func
- **Line 1086:** func - Direct call: func
- **Line 1872:** func - Direct call: func
- **Line 2198:** Path - Direct call: Path
- **Line 2329:** Path - Direct call: Path
- **Line 2203:** Path - Direct call: Path
- **Line 1697:** rule - Direct call: rule
- **Line 1739:** rule - Direct call: rule

---

## src/training/steps/market_analysis/step05_labeling_per_regime.py

**Undefined function calls: 10**

- **Line 269:** traced - Direct call: traced
- **Line 270:** validates - Direct call: validates
- **Line 324:** per_regime_processing - Direct call: per_regime_processing
- **Line 35:** traced - Direct call: traced
- **Line 342:** LabelingStep - Direct call: LabelingStep
- **Line 364:** test - Direct call: test
- **Line 356:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 60:** RegimeProcessingContext - Direct call: RegimeProcessingContext
- **Line 104:** aggregate_regime_results - Direct call: aggregate_regime_results
- **Line 107:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step05_labeling_validator.py

**Undefined function calls: 11**

- **Line 34:** validates - Direct call: validates
- **Line 97:** smart_validation_cache - Direct call: smart_validation_cache
- **Line 151:** smart_validation_cache - Direct call: smart_validation_cache
- **Line 382:** run_validator - Direct call: run_validator
- **Line 162:** safe_json_load - Direct call: safe_json_load
- **Line 211:** Path - Direct call: Path
- **Line 246:** Path - Direct call: Path
- **Line 53:** Path - Direct call: Path
- **Line 284:** Path - Direct call: Path
- **Line 285:** Path - Direct call: Path
- **Line 286:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step06_feature_engineering.py

**Undefined function calls: 7**

- **Line 124:** StandardScaler - Direct call: StandardScaler
- **Line 47:** nullcontext - Direct call: nullcontext
- **Line 94:** DiverseLookbackOptimizer - Direct call: DiverseLookbackOptimizer
- **Line 634:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 673:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 106:** MatrixDiverseLookbackOptimizer - Direct call: MatrixDiverseLookbackOptimizer
- **Line 181:** Counter - Direct call: Counter

---

## src/training/steps/market_analysis/step06_feature_engineering_per_regime.py

**Undefined function calls: 8**

- **Line 74:** get_logger - Direct call: get_logger
- **Line 914:** test - Direct call: test
- **Line 906:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 864:** Path - Direct call: Path
- **Line 593:** Path - Direct call: Path
- **Line 683:** Path - Direct call: Path
- **Line 597:** Path - Direct call: Path
- **Line 190:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step06_feature_engineering_validator.py

**Undefined function calls: 5**

- **Line 108:** smart_validation_cache - Direct call: smart_validation_cache
- **Line 348:** run_validator - Direct call: run_validator
- **Line 181:** Path - Direct call: Path
- **Line 216:** Path - Direct call: Path
- **Line 61:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step07_enhanced_matrix_operations.py

**Undefined function calls: 25**

- **Line 835:** log_execution_time - Direct call: log_execution_time
- **Line 836:** cached - Direct call: cached
- **Line 837:** log_call - Direct call: log_call
- **Line 838:** circuit_breaker - Direct call: circuit_breaker
- **Line 839:** validates - Direct call: validates
- **Line 840:** handles_errors - Direct call: handles_errors
- **Line 664:** ensure_directory - Direct call: ensure_directory
- **Line 1706:** safe_json_dump - Direct call: safe_json_dump
- **Line 1709:** safe_json_dump - Direct call: safe_json_dump
- **Line 1712:** safe_json_dump - Direct call: safe_json_dump
- **Line 1722:** safe_json_dump - Direct call: safe_json_dump
- **Line 1747:** get_training_config - Direct call: get_training_config
- **Line 52:** Path - Direct call: Path
- **Line 1079:** create_detailed_step_report - Direct call: create_detailed_step_report
- **Line 1080:** log_step_report - Direct call: log_step_report
- **Line 1088:** log_step_metrics - Direct call: log_step_metrics
- **Line 266:** func - Direct call: func
- **Line 781:** rankdata - Direct call: rankdata
- **Line 1083:** log_step_report - Direct call: log_step_report
- **Line 1086:** log_step_report - Direct call: log_step_report
- **Line 234:** func - Direct call: func
- **Line 744:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 784:** rankdata - Direct call: rankdata
- **Line 232:** func - Direct call: func
- **Line 730:** mutual_info_classif - Direct call: mutual_info_classif

---

## src/training/steps/market_analysis/step07_enhanced_matrix_operations_per_regime.py

**Undefined function calls: 17**

- **Line 23:** get_logger - Direct call: get_logger
- **Line 605:** traced - Direct call: traced
- **Line 606:** validates - Direct call: validates
- **Line 35:** traced - Direct call: traced
- **Line 36:** per_regime_step - Direct call: per_regime_step
- **Line 554:** KMeans - Direct call: KMeans
- **Line 574:** DBSCAN - Direct call: DBSCAN
- **Line 595:** GaussianMixture - Direct call: GaussianMixture
- **Line 670:** test - Direct call: test
- **Line 369:** StandardScaler - Direct call: StandardScaler
- **Line 373:** PCA - Direct call: PCA
- **Line 417:** StandardScaler - Direct call: StandardScaler
- **Line 662:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 125:** Path - Direct call: Path
- **Line 530:** Path - Direct call: Path
- **Line 129:** Path - Direct call: Path
- **Line 140:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step07_enhanced_matrix_operations_simplified.py

**Undefined function calls: 26**

- **Line 149:** comprehensive_function_tracker - Direct call: comprehensive_function_tracker
- **Line 150:** log_execution_time - Direct call: log_execution_time
- **Line 151:** cached - Direct call: cached
- **Line 152:** log_call - Direct call: log_call
- **Line 153:** circuit_breaker - Direct call: circuit_breaker
- **Line 154:** validates - Direct call: validates
- **Line 155:** handles_errors - Direct call: handles_errors
- **Line 115:** FunctionCallTracker - Direct call: FunctionCallTracker
- **Line 116:** EnhancedErrorHandler - Direct call: EnhancedErrorHandler
- **Line 117:** ComprehensiveValidator - Direct call: ComprehensiveValidator
- **Line 118:** PerformanceMonitor - Direct call: PerformanceMonitor
- **Line 119:** MatrixOperations - Direct call: MatrixOperations
- **Line 120:** QualityMetricsCalculator - Direct call: QualityMetricsCalculator
- **Line 121:** FeatureFiltering - Direct call: FeatureFiltering
- **Line 137:** ensure_directory - Direct call: ensure_directory
- **Line 448:** safe_json_dump - Direct call: safe_json_dump
- **Line 453:** safe_json_dump - Direct call: safe_json_dump
- **Line 458:** safe_json_dump - Direct call: safe_json_dump
- **Line 487:** safe_json_dump - Direct call: safe_json_dump
- **Line 691:** get_training_config - Direct call: get_training_config
- **Line 18:** Path - Direct call: Path
- **Line 591:** create_detailed_step_report - Direct call: create_detailed_step_report
- **Line 601:** log_step_report - Direct call: log_step_report
- **Line 651:** log_step_metrics - Direct call: log_step_metrics
- **Line 620:** log_step_report - Direct call: log_step_report
- **Line 636:** log_step_report - Direct call: log_step_report

---

## src/training/steps/market_analysis/step07_enhanced_matrix_operations_validator.py

**Undefined function calls: 3**

- **Line 52:** Path - Direct call: Path
- **Line 71:** Path - Direct call: Path
- **Line 98:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step08_advanced_feature_selection.py

**Undefined function calls: 34**

- **Line 84:** jit - Direct call: jit
- **Line 113:** jit - Direct call: jit
- **Line 129:** jit - Direct call: jit
- **Line 223:** handles_errors - Direct call: handles_errors
- **Line 92:** prange - Direct call: prange
- **Line 102:** prange - Direct call: prange
- **Line 158:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 172:** ensure_directory - Direct call: ensure_directory
- **Line 514:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 782:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 989:** squareform - Direct call: squareform
- **Line 992:** linkage - Direct call: linkage
- **Line 1416:** safe_json_dump - Direct call: safe_json_dump
- **Line 1476:** safe_json_dump - Direct call: safe_json_dump
- **Line 1484:** safe_json_dump - Direct call: safe_json_dump
- **Line 1573:** run_step - Direct call: run_step
- **Line 425:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 464:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 521:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 604:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 645:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 652:** BorutaPy - Direct call: BorutaPy
- **Line 801:** roc_auc_score - Direct call: roc_auc_score
- **Line 1001:** fcluster - Direct call: fcluster
- **Line 1003:** fcluster - Direct call: fcluster
- **Line 1430:** safe_json_dump - Direct call: safe_json_dump
- **Line 571:** Parallel - Direct call: Parallel
- **Line 692:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 830:** cross_val_score - Direct call: cross_val_score
- **Line 932:** roc_auc_score - Direct call: roc_auc_score
- **Line 933:** accuracy_score - Direct call: accuracy_score
- **Line 934:** f1_score - Direct call: f1_score
- **Line 296:** get_regime_column - Direct call: get_regime_column
- **Line 572:** delayed - Direct call: delayed

---

## src/training/steps/market_analysis/step08_advanced_feature_selection_per_regime.py

**Undefined function calls: 10**

- **Line 21:** get_logger - Direct call: get_logger
- **Line 615:** traced - Direct call: traced
- **Line 616:** validates - Direct call: validates
- **Line 33:** traced - Direct call: traced
- **Line 34:** per_regime_step - Direct call: per_regime_step
- **Line 680:** test - Direct call: test
- **Line 672:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 123:** Path - Direct call: Path
- **Line 602:** Path - Direct call: Path
- **Line 127:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step1/comprehensive_gap_filler.py

**Undefined function calls: 4**

- **Line 1107:** run_comprehensive_gap_filling_pipeline - Direct call: run_comprehensive_gap_filling_pipeline
- **Line 230:** timedelta - Direct call: timedelta
- **Line 27:** Path - Direct call: Path
- **Line 723:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step1/data_gap_detector.py

**Undefined function calls: 11**

- **Line 45:** validates - Direct call: validates
- **Line 46:** traced - Direct call: traced
- **Line 47:** handles_errors - Direct call: handles_errors
- **Line 301:** traced - Direct call: traced
- **Line 302:** handles_errors - Direct call: handles_errors
- **Line 367:** traced - Direct call: traced
- **Line 32:** Path - Direct call: Path
- **Line 40:** MissingDataDownloaderAndGapFiller - Direct call: MissingDataDownloaderAndGapFiller
- **Line 186:** timedelta - Direct call: timedelta
- **Line 23:** Path - Direct call: Path
- **Line 78:** timedelta - Direct call: timedelta

---

## src/training/steps/market_analysis/step1/data_quality_dashboard.py

**Undefined function calls: 30**

- **Line 146:** traced - Direct call: traced
- **Line 167:** traced - Direct call: traced
- **Line 180:** traced - Direct call: traced
- **Line 192:** traced - Direct call: traced
- **Line 204:** traced - Direct call: traced
- **Line 216:** traced - Direct call: traced
- **Line 228:** traced - Direct call: traced
- **Line 239:** traced - Direct call: traced
- **Line 251:** traced - Direct call: traced
- **Line 263:** traced - Direct call: traced
- **Line 287:** traced - Direct call: traced
- **Line 304:** traced - Direct call: traced
- **Line 49:** Path - Direct call: Path
- **Line 77:** FastAPI - Direct call: FastAPI
- **Line 329:** main - Direct call: main
- **Line 61:** EnhancedDataQualityManager - Direct call: EnhancedDataQualityManager
- **Line 66:** DataQualityMonitor - Direct call: DataQualityMonitor
- **Line 81:** StaticFiles - Direct call: StaticFiles
- **Line 324:** start_data_quality_dashboard - Direct call: start_data_quality_dashboard
- **Line 12:** Path - Direct call: Path
- **Line 197:** HTTPException - Direct call: HTTPException
- **Line 202:** HTTPException - Direct call: HTTPException
- **Line 209:** HTTPException - Direct call: HTTPException
- **Line 214:** HTTPException - Direct call: HTTPException
- **Line 221:** HTTPException - Direct call: HTTPException
- **Line 226:** HTTPException - Direct call: HTTPException
- **Line 244:** HTTPException - Direct call: HTTPException
- **Line 249:** HTTPException - Direct call: HTTPException
- **Line 256:** HTTPException - Direct call: HTTPException
- **Line 261:** HTTPException - Direct call: HTTPException

---

## src/training/steps/market_analysis/step1/data_quality_monitor.py

**Undefined function calls: 21**

- **Line 60:** traced - Direct call: traced
- **Line 61:** handles_errors - Direct call: handles_errors
- **Line 85:** traced - Direct call: traced
- **Line 91:** traced - Direct call: traced
- **Line 101:** traced - Direct call: traced
- **Line 111:** traced - Direct call: traced
- **Line 134:** traced - Direct call: traced
- **Line 167:** traced - Direct call: traced
- **Line 184:** traced - Direct call: traced
- **Line 200:** traced - Direct call: traced
- **Line 219:** traced - Direct call: traced
- **Line 235:** traced - Direct call: traced
- **Line 248:** traced - Direct call: traced
- **Line 283:** traced - Direct call: traced
- **Line 303:** traced - Direct call: traced
- **Line 323:** traced - Direct call: traced
- **Line 328:** traced - Direct call: traced
- **Line 51:** Path - Direct call: Path
- **Line 139:** EnhancedDataQualityManager - Direct call: EnhancedDataQualityManager
- **Line 17:** Path - Direct call: Path
- **Line 228:** callback - Direct call: callback

---

## src/training/steps/market_analysis/step1/data_resampler.py

**Undefined function calls: 28**

- **Line 71:** traced - Direct call: traced
- **Line 85:** validates - Direct call: validates
- **Line 87:** traced - Direct call: traced
- **Line 88:** handles_errors - Direct call: handles_errors
- **Line 167:** validates - Direct call: validates
- **Line 168:** traced - Direct call: traced
- **Line 169:** handles_errors - Direct call: handles_errors
- **Line 253:** traced - Direct call: traced
- **Line 254:** handles_errors - Direct call: handles_errors
- **Line 334:** traced - Direct call: traced
- **Line 335:** handles_errors - Direct call: handles_errors
- **Line 388:** validates - Direct call: validates
- **Line 390:** traced - Direct call: traced
- **Line 391:** handles_errors - Direct call: handles_errors
- **Line 533:** validates - Direct call: validates
- **Line 535:** traced - Direct call: traced
- **Line 536:** handles_errors - Direct call: handles_errors
- **Line 644:** validates - Direct call: validates
- **Line 646:** traced - Direct call: traced
- **Line 647:** handles_errors - Direct call: handles_errors
- **Line 709:** validates - Direct call: validates
- **Line 710:** traced - Direct call: traced
- **Line 711:** handles_errors - Direct call: handles_errors
- **Line 817:** traced - Direct call: traced
- **Line 818:** handles_errors - Direct call: handles_errors
- **Line 68:** Path - Direct call: Path
- **Line 263:** Path - Direct call: Path
- **Line 32:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step1/enhanced_data_quality_manager.py

**Undefined function calls: 29**

- **Line 569:** traced - Direct call: traced
- **Line 581:** traced - Direct call: traced
- **Line 68:** traced - Direct call: traced
- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 157:** traced - Direct call: traced
- **Line 203:** traced - Direct call: traced
- **Line 252:** traced - Direct call: traced
- **Line 253:** validates - Direct call: validates
- **Line 307:** traced - Direct call: traced
- **Line 308:** validates - Direct call: validates
- **Line 345:** traced - Direct call: traced
- **Line 346:** handles_errors - Direct call: handles_errors
- **Line 364:** traced - Direct call: traced
- **Line 365:** handles_errors - Direct call: handles_errors
- **Line 404:** traced - Direct call: traced
- **Line 405:** handles_errors - Direct call: handles_errors
- **Line 424:** traced - Direct call: traced
- **Line 488:** traced - Direct call: traced
- **Line 515:** traced - Direct call: traced
- **Line 516:** handles_errors - Direct call: handles_errors
- **Line 541:** traced - Direct call: traced
- **Line 542:** handles_errors - Direct call: handles_errors
- **Line 39:** Path - Direct call: Path
- **Line 52:** DataGapDetector - Direct call: DataGapDetector
- **Line 58:** ComprehensiveGapFiller - Direct call: ComprehensiveGapFiller
- **Line 64:** AggtradesValidator - Direct call: AggtradesValidator
- **Line 17:** Path - Direct call: Path
- **Line 523:** run_step1 - Direct call: run_step1
- **Line 550:** run_step1_5 - Direct call: run_step1_5

---

## src/training/steps/market_analysis/step1/gap_filler_pipeline.py

**Undefined function calls: 3**

- **Line 23:** Path - Direct call: Path
- **Line 213:** run_gap_filling_pipeline - Direct call: run_gap_filling_pipeline
- **Line 15:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step1/missing_data_downloader_and_gap_filler.py

**Undefined function calls: 18**

- **Line 103:** handles_errors - Direct call: handles_errors
- **Line 126:** traced - Direct call: traced
- **Line 127:** handles_errors - Direct call: handles_errors
- **Line 215:** traced - Direct call: traced
- **Line 216:** handles_errors - Direct call: handles_errors
- **Line 307:** traced - Direct call: traced
- **Line 308:** handles_errors - Direct call: handles_errors
- **Line 401:** traced - Direct call: traced
- **Line 402:** handles_errors - Direct call: handles_errors
- **Line 87:** Path - Direct call: Path
- **Line 154:** timedelta - Direct call: timedelta
- **Line 419:** timedelta - Direct call: timedelta
- **Line 13:** Path - Direct call: Path
- **Line 91:** BinanceExchange - Direct call: BinanceExchange
- **Line 272:** timedelta - Direct call: timedelta
- **Line 274:** timedelta - Direct call: timedelta
- **Line 364:** timedelta - Direct call: timedelta
- **Line 366:** timedelta - Direct call: timedelta

---

## src/training/steps/market_analysis/step1/run_step1.py

**Undefined function calls: 6**

- **Line 84:** Step1Orchestrator - Direct call: Step1Orchestrator
- **Line 126:** DataGapDetector - Direct call: DataGapDetector
- **Line 29:** Path - Direct call: Path
- **Line 149:** AggtradesValidator - Direct call: AggtradesValidator
- **Line 168:** DataPreparation - Direct call: DataPreparation
- **Line 209:** MissingDataDownloaderAndGapFiller - Direct call: MissingDataDownloaderAndGapFiller

---

## src/training/steps/market_analysis/step1/step1_orchestrator.py

**Undefined function calls: 15**

- **Line 47:** handles_errors - Direct call: handles_errors
- **Line 300:** traced - Direct call: traced
- **Line 301:** handles_errors - Direct call: handles_errors
- **Line 370:** traced - Direct call: traced
- **Line 371:** handles_errors - Direct call: handles_errors
- **Line 497:** traced - Direct call: traced
- **Line 498:** handles_errors - Direct call: handles_errors
- **Line 37:** Path - Direct call: Path
- **Line 41:** DataGapDetector - Direct call: DataGapDetector
- **Line 42:** AggtradesValidator - Direct call: AggtradesValidator
- **Line 43:** DataPreparation - Direct call: DataPreparation
- **Line 44:** MissingDataDownloaderAndGapFiller - Direct call: MissingDataDownloaderAndGapFiller
- **Line 45:** ComprehensiveGapFiller - Direct call: ComprehensiveGapFiller
- **Line 349:** Path - Direct call: Path
- **Line 28:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step1/validate_and_fix_aggtrades_format.py

**Undefined function calls: 11**

- **Line 81:** traced - Direct call: traced
- **Line 93:** validates - Direct call: validates
- **Line 94:** traced - Direct call: traced
- **Line 95:** handles_errors - Direct call: handles_errors
- **Line 337:** traced - Direct call: traced
- **Line 338:** handles_errors - Direct call: handles_errors
- **Line 415:** traced - Direct call: traced
- **Line 416:** handles_errors - Direct call: handles_errors
- **Line 474:** traced - Direct call: traced
- **Line 78:** Path - Direct call: Path
- **Line 15:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/advanced_optimization_engine.py

**Undefined function calls: 3**

- **Line 262:** KFold - Direct call: KFold
- **Line 87:** NSGAIISampler - Direct call: NSGAIISampler
- **Line 87:** MedianPruner - Direct call: MedianPruner

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/comprehensive_parameter_integration.py

**Undefined function calls: 3**

- **Line 239:** Path - Direct call: Path
- **Line 72:** method - Direct call: method
- **Line 126:** method - Direct call: method

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/confidence_based_entry_logic.py

**Undefined function calls: 3**

- **Line 73:** handles_errors - Direct call: handles_errors
- **Line 78:** traced - Direct call: traced
- **Line 315:** LinearConfidenceScaler - Direct call: LinearConfidenceScaler

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/efficiency_optimizer.py

**Undefined function calls: 10**

- **Line 59:** handles_errors - Direct call: handles_errors
- **Line 70:** handles_errors - Direct call: handles_errors
- **Line 521:** run_efficiency_test - Direct call: run_efficiency_test
- **Line 64:** ProcessPoolExecutor - Direct call: ProcessPoolExecutor
- **Line 66:** ThreadPoolExecutor - Direct call: ThreadPoolExecutor
- **Line 339:** objective_function - Direct call: objective_function
- **Line 355:** hash - Direct call: hash
- **Line 153:** hash - Direct call: hash
- **Line 358:** hash - Direct call: hash
- **Line 335:** objective_function - Direct call: objective_function

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/evaluation_engine.py

**Undefined function calls: 15**

- **Line 147:** error - Direct call: error
- **Line 242:** error - Direct call: error
- **Line 388:** error - Direct call: error
- **Line 403:** error - Direct call: error
- **Line 420:** error - Direct call: error
- **Line 434:** error - Direct call: error
- **Line 449:** error - Direct call: error
- **Line 465:** error - Direct call: error
- **Line 486:** error - Direct call: error
- **Line 510:** error - Direct call: error
- **Line 527:** warning - Direct call: warning
- **Line 550:** error - Direct call: error
- **Line 614:** error - Direct call: error
- **Line 651:** error - Direct call: error
- **Line 227:** timedelta - Direct call: timedelta

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/hyperparameter_optimization_config.py

**Undefined function calls: 3**

- **Line 49:** field - Direct call: field
- **Line 54:** field - Direct call: field
- **Line 55:** field - Direct call: field

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/optimized_optuna_optimization.py

**Undefined function calls: 4**

- **Line 19:** setup_logging - Direct call: setup_logging
- **Line 200:** StratifiedKFold - Direct call: StratifiedKFold
- **Line 223:** cross_val_score - Direct call: cross_val_score
- **Line 209:** cross_val_score - Direct call: cross_val_score

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/optimized_optuna_optimization_enhanced.py

**Undefined function calls: 15**

- **Line 188:** lru_cache - Direct call: lru_cache
- **Line 120:** setup_logging - Direct call: setup_logging
- **Line 129:** SROptimizationParameters - Direct call: SROptimizationParameters
- **Line 460:** main - Direct call: main
- **Line 135:** validate_sr_optimization_config - Direct call: validate_sr_optimization_config
- **Line 137:** SROptimizationParameters - Direct call: SROptimizationParameters
- **Line 153:** _RFC - Direct call: _RFC
- **Line 212:** jit - Direct call: jit
- **Line 379:** model_cls - Direct call: model_cls
- **Line 278:** HyperbandPruner - Direct call: HyperbandPruner
- **Line 278:** TPESampler - Direct call: TPESampler
- **Line 391:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 394:** StratifiedKFold - Direct call: StratifiedKFold
- **Line 284:** custom_space - Direct call: custom_space
- **Line 283:** custom_objective - Direct call: custom_objective

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/regime_specific_triple_barrier_optimization.py

**Undefined function calls: 14**

- **Line 38:** setup_logging - Direct call: setup_logging
- **Line 94:** field - Direct call: field
- **Line 95:** field - Direct call: field
- **Line 96:** field - Direct call: field
- **Line 388:** precision_score - Direct call: precision_score
- **Line 395:** recall_score - Direct call: recall_score
- **Line 547:** setup_regime_specific_optimizer - Direct call: setup_regime_specific_optimizer
- **Line 189:** get_regime_column - Direct call: get_regime_column
- **Line 262:** OptimizedTripleBarrierLabeling - Direct call: OptimizedTripleBarrierLabeling
- **Line 472:** Path - Direct call: Path
- **Line 443:** TPESampler - Direct call: TPESampler
- **Line 502:** plot_param_importances - Direct call: plot_param_importances
- **Line 505:** plot_optimization_history - Direct call: plot_optimization_history
- **Line 443:** HyperbandPruner - Direct call: HyperbandPruner

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/sr_optuna_optimization.py

**Undefined function calls: 12**

- **Line 16:** setup_logging - Direct call: setup_logging
- **Line 393:** main - Direct call: main
- **Line 83:** ensure_optimized_sr_config - Direct call: ensure_optimized_sr_config
- **Line 88:** SRWeightOptimizer - Direct call: SRWeightOptimizer
- **Line 353:** plot_optimization_history - Direct call: plot_optimization_history
- **Line 358:** plot_param_importances - Direct call: plot_param_importances
- **Line 379:** setup_sr_optuna_optimizer - Direct call: setup_sr_optuna_optimizer
- **Line 84:** setup_sr_breakout_predictor - Direct call: setup_sr_breakout_predictor
- **Line 138:** HyperbandPruner - Direct call: HyperbandPruner
- **Line 138:** TPESampler - Direct call: TPESampler
- **Line 140:** HyperbandPruner - Direct call: HyperbandPruner
- **Line 140:** TPESampler - Direct call: TPESampler

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/step17_probabilistic_bayesian_optimization.py

**Undefined function calls: 9**

- **Line 68:** ProbabilisticOptimizationConfig - Direct call: ProbabilisticOptimizationConfig
- **Line 92:** ProbabilisticBayesianOptimizer - Direct call: ProbabilisticBayesianOptimizer
- **Line 93:** ProbabilisticBayesianOptimizer - Direct call: ProbabilisticBayesianOptimizer
- **Line 94:** ProbabilisticModelIntegrator - Direct call: ProbabilisticModelIntegrator
- **Line 98:** AdvancedOptunaManager - Direct call: AdvancedOptunaManager
- **Line 99:** AdvancedOptunaManager - Direct call: AdvancedOptunaManager
- **Line 232:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 247:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 400:** Path - Direct call: Path

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/triple_barrier_optimizer.py

**Undefined function calls: 2**

- **Line 58:** handles_errors - Direct call: handles_errors
- **Line 63:** traced - Direct call: traced

---

## src/training/steps/market_analysis/utils/feature_filtering.py

**Undefined function calls: 4**

- **Line 160:** rankdata - Direct call: rankdata
- **Line 123:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 163:** rankdata - Direct call: rankdata
- **Line 109:** mutual_info_classif - Direct call: mutual_info_classif

---

## src/training/steps/market_analysis/utils/function_call_tracker.py

**Undefined function calls: 3**

- **Line 198:** func - Direct call: func
- **Line 166:** func - Direct call: func
- **Line 164:** func - Direct call: func

---

## src/training/steps/model_training/__init__.py

**Undefined function calls: 61**

- **Line 47:** handles_errors - Direct call: handles_errors
- **Line 100:** handles_errors - Direct call: handles_errors
- **Line 101:** validates - Direct call: validates
- **Line 181:** handles_errors - Direct call: handles_errors
- **Line 241:** handles_errors - Direct call: handles_errors
- **Line 382:** handles_errors - Direct call: handles_errors
- **Line 450:** handles_errors - Direct call: handles_errors
- **Line 483:** handles_errors - Direct call: handles_errors
- **Line 509:** handles_errors - Direct call: handles_errors
- **Line 597:** handles_errors - Direct call: handles_errors
- **Line 626:** handles_errors - Direct call: handles_errors
- **Line 655:** handles_errors - Direct call: handles_errors
- **Line 684:** handles_errors - Direct call: handles_errors
- **Line 710:** handles_errors - Direct call: handles_errors
- **Line 736:** handles_errors - Direct call: handles_errors
- **Line 762:** handles_errors - Direct call: handles_errors
- **Line 788:** handles_errors - Direct call: handles_errors
- **Line 852:** handles_errors - Direct call: handles_errors
- **Line 119:** safe_file_exists - Direct call: safe_file_exists
- **Line 189:** ValidatorOrchestrator - Direct call: ValidatorOrchestrator
- **Line 190:** StepDependencyValidator - Direct call: StepDependencyValidator
- **Line 258:** safe_read_parquet - Direct call: safe_read_parquet
- **Line 268:** optimize_dataframe_dtypes - Direct call: optimize_dataframe_dtypes
- **Line 273:** validate_dataframe_schema - Direct call: validate_dataframe_schema
- **Line 392:** step_class - Direct call: step_class
- **Line 809:** safe_file_exists - Direct call: safe_file_exists
- **Line 932:** ModelExplainer - Direct call: ModelExplainer
- **Line 1014:** _monitor_memory_usage - Direct call: _monitor_memory_usage
- **Line 1019:** _validate_pipeline_inputs - Direct call: _validate_pipeline_inputs
- **Line 1030:** _validate_step_dependencies - Direct call: _validate_step_dependencies
- **Line 1041:** _validate_data_quality - Direct call: _validate_data_quality
- **Line 1051:** _monitor_memory_usage - Direct call: _monitor_memory_usage
- **Line 1170:** safe_json_dump - Direct call: safe_json_dump
- **Line 1175:** safe_log_metric - Direct call: safe_log_metric
- **Line 1176:** safe_log_metric - Direct call: safe_log_metric
- **Line 1177:** safe_log_metric - Direct call: safe_log_metric
- **Line 1178:** safe_log_metric - Direct call: safe_log_metric
- **Line 1179:** safe_log_metric - Direct call: safe_log_metric
- **Line 1180:** safe_log_metric - Direct call: safe_log_metric
- **Line 1181:** safe_log_metric - Direct call: safe_log_metric
- **Line 1182:** safe_log_params - Direct call: safe_log_params
- **Line 132:** safe_file_exists - Direct call: safe_file_exists
- **Line 810:** safe_read_parquet - Direct call: safe_read_parquet
- **Line 837:** safe_file_exists - Direct call: safe_file_exists
- **Line 882:** _extract_models_and_data - Direct call: _extract_models_and_data
- **Line 938:** _load_model_specific_data - Direct call: _load_model_specific_data
- **Line 1081:** _execute_training_step - Direct call: _execute_training_step
- **Line 1126:** _monitor_memory_usage - Direct call: _monitor_memory_usage
- **Line 838:** safe_read_parquet - Direct call: safe_read_parquet
- **Line 894:** safe_file_exists - Direct call: safe_file_exists
- **Line 1135:** format_datetime - Direct call: format_datetime
- **Line 426:** _run_model_interpretability_analysis - Direct call: _run_model_interpretability_analysis
- **Line 895:** safe_read_parquet - Direct call: safe_read_parquet
- **Line 906:** safe_file_exists - Direct call: safe_file_exists
- **Line 1098:** _monitor_memory_usage - Direct call: _monitor_memory_usage
- **Line 1135:** get_current_datetime - Direct call: get_current_datetime
- **Line 1153:** safe_file_exists - Direct call: safe_file_exists
- **Line 1153:** format_bytes - Direct call: format_bytes
- **Line 907:** safe_read_parquet - Direct call: safe_read_parquet
- **Line 139:** Path - Direct call: Path
- **Line 1153:** Path - Direct call: Path

---

## src/training/steps/model_training/analyst_enhancement_components/analyst_enhancement_step.py

**Undefined function calls: 6**

- **Line 67:** handles_errors - Direct call: handles_errors
- **Line 30:** HyperparameterOptimizer - Direct call: HyperparameterOptimizer
- **Line 31:** FeatureSelector - Direct call: FeatureSelector
- **Line 32:** ModelOptimizer - Direct call: ModelOptimizer
- **Line 33:** EnsembleCreator - Direct call: EnsembleCreator
- **Line 138:** accuracy_score - Direct call: accuracy_score

---

## src/training/steps/model_training/analyst_enhancement_components/ensemble_creator.py

**Undefined function calls: 6**

- **Line 26:** handles_errors - Direct call: handles_errors
- **Line 88:** VotingClassifier - Direct call: VotingClassifier
- **Line 102:** StackingClassifier - Direct call: StackingClassifier
- **Line 148:** VotingClassifier - Direct call: VotingClassifier
- **Line 99:** LogisticRegression - Direct call: LogisticRegression
- **Line 101:** LogisticRegression - Direct call: LogisticRegression

---

## src/training/steps/model_training/analyst_enhancement_components/feature_selector.py

**Undefined function calls: 6**

- **Line 29:** handles_errors - Direct call: handles_errors
- **Line 65:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 94:** RFE - Direct call: RFE
- **Line 147:** accuracy_score - Direct call: accuracy_score
- **Line 79:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 154:** accuracy_score - Direct call: accuracy_score

---

## src/training/steps/model_training/analyst_enhancement_components/hyperparameter_optimizer.py

**Undefined function calls: 3**

- **Line 32:** handles_errors - Direct call: handles_errors
- **Line 106:** model_class - Direct call: model_class
- **Line 58:** cross_val_score - Direct call: cross_val_score

---

## src/training/steps/model_training/analyst_enhancement_components/model_optimizer.py

**Undefined function calls: 3**

- **Line 26:** handles_errors - Direct call: handles_errors
- **Line 62:** model_class - Direct call: model_class
- **Line 87:** accuracy_score - Direct call: accuracy_score

---

## src/training/steps/model_training/analyst_ensemble_components/analyst_ensemble_creation_step.py

**Undefined function calls: 5**

- **Line 65:** handles_errors - Direct call: handles_errors
- **Line 29:** EnsembleAggregator - Direct call: EnsembleAggregator
- **Line 30:** VotingMechanism - Direct call: VotingMechanism
- **Line 31:** WeightOptimizer - Direct call: WeightOptimizer
- **Line 32:** EnsembleEvaluator - Direct call: EnsembleEvaluator

---

## src/training/steps/model_training/analyst_ensemble_components/ensemble_aggregator.py

**Undefined function calls: 9**

- **Line 27:** handles_errors - Direct call: handles_errors
- **Line 52:** handles_errors - Direct call: handles_errors
- **Line 74:** handles_errors - Direct call: handles_errors
- **Line 157:** handles_errors - Direct call: handles_errors
- **Line 48:** VotingClassifier - Direct call: VotingClassifier
- **Line 70:** StackingClassifier - Direct call: StackingClassifier
- **Line 170:** VotingClassifier - Direct call: VotingClassifier
- **Line 184:** LogisticRegression - Direct call: LogisticRegression
- **Line 187:** LogisticRegression - Direct call: LogisticRegression

---

## src/training/steps/model_training/analyst_ensemble_components/ensemble_evaluator.py

**Undefined function calls: 13**

- **Line 28:** handles_errors - Direct call: handles_errors
- **Line 47:** handles_errors - Direct call: handles_errors
- **Line 120:** handles_errors - Direct call: handles_errors
- **Line 166:** handles_errors - Direct call: handles_errors
- **Line 112:** KFold - Direct call: KFold
- **Line 113:** cross_val_score - Direct call: cross_val_score
- **Line 181:** confusion_matrix - Direct call: confusion_matrix
- **Line 183:** classification_report - Direct call: classification_report
- **Line 78:** accuracy_score - Direct call: accuracy_score
- **Line 80:** f1_score - Direct call: f1_score
- **Line 82:** precision_score - Direct call: precision_score
- **Line 84:** recall_score - Direct call: recall_score
- **Line 87:** roc_auc_score - Direct call: roc_auc_score

---

## src/training/steps/model_training/analyst_ensemble_components/voting_mechanism.py

**Undefined function calls: 5**

- **Line 25:** handles_errors - Direct call: handles_errors
- **Line 129:** handles_errors - Direct call: handles_errors
- **Line 63:** VotingClassifier - Direct call: VotingClassifier
- **Line 89:** VotingClassifier - Direct call: VotingClassifier
- **Line 122:** VotingClassifier - Direct call: VotingClassifier

---

## src/training/steps/model_training/analyst_ensemble_components/weight_optimizer.py

**Undefined function calls: 6**

- **Line 28:** handles_errors - Direct call: handles_errors
- **Line 198:** handles_errors - Direct call: handles_errors
- **Line 120:** minimize - Direct call: minimize
- **Line 161:** accuracy_score - Direct call: accuracy_score
- **Line 111:** accuracy_score - Direct call: accuracy_score
- **Line 114:** log_loss - Direct call: log_loss

---

## src/training/steps/model_training/analyst_training_components/regime_specific_tpsl_optimizer.py

**Undefined function calls: 15**

- **Line 74:** handle_specific_errors - Direct call: handle_specific_errors
- **Line 133:** handles_errors - Direct call: handles_errors
- **Line 162:** handles_errors - Direct call: handles_errors
- **Line 266:** handles_errors - Direct call: handles_errors
- **Line 25:** Path - Direct call: Path
- **Line 55:** MetaLabelingSystem - Direct call: MetaLabelingSystem
- **Line 85:** failed - Direct call: failed
- **Line 109:** initialization_error - Direct call: initialization_error
- **Line 121:** warning - Direct call: warning
- **Line 131:** failed - Direct call: failed
- **Line 145:** warning - Direct call: warning
- **Line 159:** error - Direct call: error
- **Line 191:** error - Direct call: error
- **Line 229:** error - Direct call: error
- **Line 288:** error - Direct call: error

---

## src/training/steps/model_training/hmm_training_components.py

**Undefined function calls: 22**

- **Line 226:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 228:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 109:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 111:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 286:** precision_score - Direct call: precision_score
- **Line 293:** recall_score - Direct call: recall_score
- **Line 89:** accuracy_score - Direct call: accuracy_score
- **Line 89:** f1_score - Direct call: f1_score
- **Line 91:** mean_squared_error - Direct call: mean_squared_error
- **Line 91:** r2_score - Direct call: r2_score
- **Line 115:** accuracy_score - Direct call: accuracy_score
- **Line 115:** f1_score - Direct call: f1_score
- **Line 117:** mean_squared_error - Direct call: mean_squared_error
- **Line 117:** r2_score - Direct call: r2_score
- **Line 235:** accuracy_score - Direct call: accuracy_score
- **Line 235:** f1_score - Direct call: f1_score
- **Line 237:** mean_squared_error - Direct call: mean_squared_error
- **Line 237:** r2_score - Direct call: r2_score
- **Line 273:** mean_squared_error - Direct call: mean_squared_error
- **Line 273:** r2_score - Direct call: r2_score
- **Line 278:** accuracy_score - Direct call: accuracy_score
- **Line 278:** f1_score - Direct call: f1_score

---

## src/training/steps/model_training/matrix_components.py

**Undefined function calls: 4**

- **Line 143:** DiverseLookbackOptimizer - Direct call: DiverseLookbackOptimizer
- **Line 246:** matrix_func - Direct call: matrix_func
- **Line 263:** matrix_func - Direct call: matrix_func
- **Line 267:** matrix_func - Direct call: matrix_func

---

## src/training/steps/model_training/multi_timeframe_hmm_ensemble.py

**Undefined function calls: 4**

- **Line 131:** handles_errors - Direct call: handles_errors
- **Line 217:** handles_errors - Direct call: handles_errors
- **Line 273:** handles_errors - Direct call: handles_errors
- **Line 393:** handles_errors - Direct call: handles_errors

---

## src/training/steps/model_training/per_regime_pipeline_integration.py

**Undefined function calls: 7**

- **Line 312:** per_regime_processing - Direct call: per_regime_processing
- **Line 75:** get_logger - Direct call: get_logger
- **Line 400:** test - Direct call: test
- **Line 317:** original_step_func - Direct call: original_step_func
- **Line 198:** Path - Direct call: Path
- **Line 334:** regime_processor - Direct call: regime_processor
- **Line 343:** original_step_func - Direct call: original_step_func

---

## src/training/steps/model_training/per_regime_pipeline_orchestrator.py

**Undefined function calls: 15**

- **Line 570:** traced - Direct call: traced
- **Line 571:** validates - Direct call: validates
- **Line 116:** traced - Direct call: traced
- **Line 393:** traced - Direct call: traced
- **Line 62:** get_logger - Direct call: get_logger
- **Line 71:** PerRegimePipelineIntegrator - Direct call: PerRegimePipelineIntegrator
- **Line 635:** test - Direct call: test
- **Line 381:** asdict - Direct call: asdict
- **Line 627:** run_per_regime_pipeline - Direct call: run_per_regime_pipeline
- **Line 285:** step_function - Direct call: step_function
- **Line 377:** Path - Direct call: Path
- **Line 419:** Path - Direct call: Path
- **Line 424:** safe_json_load - Direct call: safe_json_load
- **Line 431:** safe_json_load - Direct call: safe_json_load
- **Line 105:** Path - Direct call: Path

---

## src/training/steps/model_training/sr_outcome_model_trainer.py

**Undefined function calls: 16**

- **Line 54:** handles_errors - Direct call: handles_errors
- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 214:** validates - Direct call: validates
- **Line 47:** SRBreakoutPredictor - Direct call: SRBreakoutPredictor
- **Line 48:** StandardScaler - Direct call: StandardScaler
- **Line 49:** LabelEncoder - Direct call: LabelEncoder
- **Line 267:** compute_class_weight - Direct call: compute_class_weight
- **Line 270:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 288:** compute_class_weight - Direct call: compute_class_weight
- **Line 291:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 317:** VotingClassifier - Direct call: VotingClassifier
- **Line 404:** classification_report - Direct call: classification_report
- **Line 405:** confusion_matrix - Direct call: confusion_matrix
- **Line 406:** roc_auc_score - Direct call: roc_auc_score
- **Line 343:** roc_auc_score - Direct call: roc_auc_score
- **Line 374:** roc_auc_score - Direct call: roc_auc_score

---

## src/training/steps/model_training/step04_5_triple_barrier_method.py

**Undefined function calls: 17**

- **Line 86:** traced - Direct call: traced
- **Line 87:** validates - Direct call: validates
- **Line 92:** handles_errors - Direct call: handles_errors
- **Line 93:** validates - Direct call: validates
- **Line 134:** handles_errors - Direct call: handles_errors
- **Line 190:** handles_errors - Direct call: handles_errors
- **Line 230:** OptimizedTripleBarrierLabeling - Direct call: OptimizedTripleBarrierLabeling
- **Line 290:** test - Direct call: test
- **Line 19:** Path - Direct call: Path
- **Line 62:** OptimizedTripleBarrierLabeling - Direct call: OptimizedTripleBarrierLabeling
- **Line 119:** ensure_directory - Direct call: ensure_directory
- **Line 181:** ensure_directory - Direct call: ensure_directory
- **Line 288:** run_step - Direct call: run_step
- **Line 118:** Path - Direct call: Path
- **Line 206:** Path - Direct call: Path
- **Line 99:** Path - Direct call: Path
- **Line 193:** Path - Direct call: Path

---

## src/training/steps/model_training/step05_labeling.py

**Undefined function calls: 3**

- **Line 365:** Path - Direct call: Path
- **Line 47:** TripleBarrierLabeler - Direct call: TripleBarrierLabeler
- **Line 51:** MetaLabelingSystem - Direct call: MetaLabelingSystem

---

## src/training/steps/model_training/step07_enhanced_matrix_operations.py

**Undefined function calls: 9**

- **Line 573:** handles_errors - Direct call: handles_errors
- **Line 485:** MatrixProcessor - Direct call: MatrixProcessor
- **Line 495:** MatrixOptimizer - Direct call: MatrixOptimizer
- **Line 1171:** Path - Direct call: Path
- **Line 1215:** Path - Direct call: Path
- **Line 227:** func - Direct call: func
- **Line 492:** DiverseLookbackIntegrator - Direct call: DiverseLookbackIntegrator
- **Line 195:** func - Direct call: func
- **Line 193:** func - Direct call: func

---

## src/training/steps/model_training/step09_5_hmm_lm_generalist_training.py

**Undefined function calls: 22**

- **Line 1280:** deterministic_seed - Direct call: deterministic_seed
- **Line 1281:** idempotent_step - Direct call: idempotent_step
- **Line 1283:** validates - Direct call: validates
- **Line 1285:** timeout - Direct call: timeout
- **Line 1286:** validates - Direct call: validates
- **Line 1298:** circuit_breaker_protection - Direct call: circuit_breaker_protection
- **Line 1302:** validates - Direct call: validates
- **Line 1307:** handles_errors - Direct call: handles_errors
- **Line 108:** handles_errors - Direct call: handles_errors
- **Line 115:** traced - Direct call: traced
- **Line 116:** validates - Direct call: validates
- **Line 118:** handles_errors - Direct call: handles_errors
- **Line 298:** traced - Direct call: traced
- **Line 1136:** DataLoader - Direct call: DataLoader
- **Line 1356:** test - Direct call: test
- **Line 235:** create_detailed_step_report - Direct call: create_detailed_step_report
- **Line 246:** log_step_report - Direct call: log_step_report
- **Line 280:** log_step_metrics - Direct call: log_step_metrics
- **Line 359:** load_timeframe_data - Direct call: load_timeframe_data
- **Line 1354:** run_step - Direct call: run_step
- **Line 264:** log_step_report - Direct call: log_step_report
- **Line 795:** standardize_price_action_probabilities - Direct call: standardize_price_action_probabilities

---

## src/training/steps/model_training/step09_5_hmm_lm_generalist_training_validator.py

**Undefined function calls: 6**

- **Line 163:** validates - Direct call: validates
- **Line 23:** validates - Direct call: validates
- **Line 77:** validates - Direct call: validates
- **Line 106:** validates - Direct call: validates
- **Line 113:** safe_json_load - Direct call: safe_json_load
- **Line 42:** Path - Direct call: Path

---

## src/training/steps/model_training/step09_5_multi_timeframe_hmm_ensemble.py

**Undefined function calls: 22**

- **Line 262:** validates - Direct call: validates
- **Line 263:** log_execution_time - Direct call: log_execution_time
- **Line 264:** cached - Direct call: cached
- **Line 265:** circuit_breaker - Direct call: circuit_breaker
- **Line 266:** log_call - Direct call: log_call
- **Line 267:** monitor_feature_engineering - Direct call: monitor_feature_engineering
- **Line 268:** handles_errors - Direct call: handles_errors
- **Line 375:** handles_errors - Direct call: handles_errors
- **Line 146:** get_multi_timeframe_hmm_ensemble_config - Direct call: get_multi_timeframe_hmm_ensemble_config
- **Line 289:** get_multi_timeframe_hmm_ensemble_config - Direct call: get_multi_timeframe_hmm_ensemble_config
- **Line 298:** EnsembleConfig - Direct call: EnsembleConfig
- **Line 323:** MultiTimeframeHMMEnsemble - Direct call: MultiTimeframeHMMEnsemble
- **Line 115:** MultiTimeframeHMMEnsemble - Direct call: MultiTimeframeHMMEnsemble
- **Line 354:** get_regime_column - Direct call: get_regime_column
- **Line 363:** MultiTimeframeHMMEnsemble - Direct call: MultiTimeframeHMMEnsemble
- **Line 404:** safe_json_load - Direct call: safe_json_load
- **Line 297:** TimeframeConfig - Direct call: TimeframeConfig
- **Line 356:** get_regime_ids - Direct call: get_regime_ids
- **Line 241:** ensure_directory - Direct call: ensure_directory
- **Line 243:** safe_json_dump - Direct call: safe_json_dump
- **Line 309:** safe_json_load - Direct call: safe_json_load
- **Line 246:** safe_json_dump - Direct call: safe_json_dump

---

## src/training/steps/model_training/step09_5_multi_timeframe_hmm_ensemble_validator.py

**Undefined function calls: 10**

- **Line 31:** handles_errors - Direct call: handles_errors
- **Line 191:** handles_errors - Direct call: handles_errors
- **Line 298:** handles_errors - Direct call: handles_errors
- **Line 156:** get_multi_timeframe_hmm_ensemble_config - Direct call: get_multi_timeframe_hmm_ensemble_config
- **Line 223:** Path - Direct call: Path
- **Line 63:** Path - Direct call: Path
- **Line 84:** safe_json_load - Direct call: safe_json_load
- **Line 251:** safe_json_load - Direct call: safe_json_load
- **Line 331:** Path - Direct call: Path
- **Line 336:** safe_json_load - Direct call: safe_json_load

---

## src/training/steps/model_training/step09_hmm_based_training.py

**Undefined function calls: 59**

- **Line 134:** handles_errors - Direct call: handles_errors
- **Line 135:** validates - Direct call: validates
- **Line 168:** handles_errors - Direct call: handles_errors
- **Line 209:** handles_errors - Direct call: handles_errors
- **Line 236:** handles_errors - Direct call: handles_errors
- **Line 281:** handles_errors - Direct call: handles_errors
- **Line 282:** validates - Direct call: validates
- **Line 347:** handles_errors - Direct call: handles_errors
- **Line 348:** validates - Direct call: validates
- **Line 430:** handles_errors - Direct call: handles_errors
- **Line 477:** handles_errors - Direct call: handles_errors
- **Line 512:** handles_errors - Direct call: handles_errors
- **Line 513:** validates - Direct call: validates
- **Line 834:** handles_errors - Direct call: handles_errors
- **Line 946:** handles_errors - Direct call: handles_errors
- **Line 1267:** handles_errors - Direct call: handles_errors
- **Line 70:** safe_copy - Direct call: safe_copy
- **Line 73:** SRBreakoutPredictor - Direct call: SRBreakoutPredictor
- **Line 80:** ProfitBasedFeatureEngineering - Direct call: ProfitBasedFeatureEngineering
- **Line 912:** safe_copy - Direct call: safe_copy
- **Line 1525:** safe_read_parquet - Direct call: safe_read_parquet
- **Line 219:** EnhancedDataQualityValidator - Direct call: EnhancedDataQualityValidator
- **Line 223:** FeatureEngineeringValidator - Direct call: FeatureEngineeringValidator
- **Line 227:** ModelPerformanceMonitor - Direct call: ModelPerformanceMonitor
- **Line 316:** safe_read_parquet - Direct call: safe_read_parquet
- **Line 322:** validate_dataframe_schema - Direct call: validate_dataframe_schema
- **Line 327:** validate_data_quality - Direct call: validate_data_quality
- **Line 538:** safe_copy - Direct call: safe_copy
- **Line 974:** format_datetime - Direct call: format_datetime
- **Line 1021:** standardize_price_action_probabilities - Direct call: standardize_price_action_probabilities
- **Line 1152:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 1214:** StandardScaler - Direct call: StandardScaler
- **Line 1379:** ensure_directory - Direct call: ensure_directory
- **Line 1415:** safe_json_dump - Direct call: safe_json_dump
- **Line 1536:** get_regime_column - Direct call: get_regime_column
- **Line 298:** safe_file_exists - Direct call: safe_file_exists
- **Line 305:** safe_file_exists - Direct call: safe_file_exists
- **Line 309:** safe_file_exists - Direct call: safe_file_exists
- **Line 552:** hash - Direct call: hash
- **Line 701:** format_datetime - Direct call: format_datetime
- **Line 872:** EnhancedMatrixOperations - Direct call: EnhancedMatrixOperations
- **Line 974:** get_current_datetime - Direct call: get_current_datetime
- **Line 1161:** StandardScaler - Direct call: StandardScaler
- **Line 1384:** ensure_directory - Direct call: ensure_directory
- **Line 1394:** ensure_directory - Direct call: ensure_directory
- **Line 1539:** split_train_val_test_by_regime - Direct call: split_train_val_test_by_regime
- **Line 1564:** ensure_directory - Direct call: ensure_directory
- **Line 701:** get_current_datetime - Direct call: get_current_datetime
- **Line 814:** ensure_directory - Direct call: ensure_directory
- **Line 1206:** average_precision_score - Direct call: average_precision_score
- **Line 1210:** accuracy_score - Direct call: accuracy_score
- **Line 1226:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 1567:** ensure_directory - Direct call: ensure_directory
- **Line 1032:** average_precision_score - Direct call: average_precision_score
- **Line 1182:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 1234:** LogisticRegression - Direct call: LogisticRegression
- **Line 1240:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV
- **Line 1191:** LogisticRegression - Direct call: LogisticRegression
- **Line 1197:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV

---

## src/training/steps/model_training/step09_hmm_based_training_per_regime.py

**Undefined function calls: 26**

- **Line 21:** get_logger - Direct call: get_logger
- **Line 845:** traced - Direct call: traced
- **Line 846:** validates - Direct call: validates
- **Line 33:** traced - Direct call: traced
- **Line 34:** per_regime_step - Direct call: per_regime_step
- **Line 910:** test - Direct call: test
- **Line 457:** train_test_split - Direct call: train_test_split
- **Line 468:** accuracy_score - Direct call: accuracy_score
- **Line 516:** train_test_split - Direct call: train_test_split
- **Line 519:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 527:** accuracy_score - Direct call: accuracy_score
- **Line 574:** train_test_split - Direct call: train_test_split
- **Line 577:** StandardScaler - Direct call: StandardScaler
- **Line 684:** train_test_split - Direct call: train_test_split
- **Line 687:** LogisticRegression - Direct call: LogisticRegression
- **Line 695:** accuracy_score - Direct call: accuracy_score
- **Line 739:** train_test_split - Direct call: train_test_split
- **Line 902:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 636:** model - Direct call: model
- **Line 640:** accuracy_score - Direct call: accuracy_score
- **Line 757:** accuracy_score - Direct call: accuracy_score
- **Line 123:** Path - Direct call: Path
- **Line 628:** model - Direct call: model
- **Line 629:** criterion - Direct call: criterion
- **Line 832:** Path - Direct call: Path
- **Line 127:** Path - Direct call: Path

---

## src/training/steps/model_training/step09_hmm_based_training_validator.py

**Undefined function calls: 10**

- **Line 20:** handles_errors - Direct call: handles_errors
- **Line 21:** validates - Direct call: validates
- **Line 73:** handles_errors - Direct call: handles_errors
- **Line 105:** handles_errors - Direct call: handles_errors
- **Line 132:** handles_errors - Direct call: handles_errors
- **Line 155:** safe_file_exists - Direct call: safe_file_exists
- **Line 96:** safe_file_exists - Direct call: safe_file_exists
- **Line 145:** safe_file_exists - Direct call: safe_file_exists
- **Line 159:** validate_dataframe_schema - Direct call: validate_dataframe_schema
- **Line 148:** Path - Direct call: Path

---

## src/training/steps/model_training/step09_model_training_main.py

**Undefined function calls: 32**

- **Line 42:** handles_errors - Direct call: handles_errors
- **Line 43:** validates - Direct call: validates
- **Line 133:** handles_errors - Direct call: handles_errors
- **Line 242:** handles_errors - Direct call: handles_errors
- **Line 298:** safe_json_dump - Direct call: safe_json_dump
- **Line 303:** safe_log_metric - Direct call: safe_log_metric
- **Line 304:** safe_log_metric - Direct call: safe_log_metric
- **Line 305:** safe_log_metric - Direct call: safe_log_metric
- **Line 306:** safe_log_metric - Direct call: safe_log_metric
- **Line 307:** safe_log_metric - Direct call: safe_log_metric
- **Line 308:** safe_log_params - Direct call: safe_log_params
- **Line 91:** safe_file_exists - Direct call: safe_file_exists
- **Line 179:** safe_file_exists - Direct call: safe_file_exists
- **Line 192:** safe_read_parquet - Direct call: safe_read_parquet
- **Line 200:** optimize_dataframe_dtypes - Direct call: optimize_dataframe_dtypes
- **Line 306:** safe_int - Direct call: safe_int
- **Line 420:** validate_training_config - Direct call: validate_training_config
- **Line 431:** validate_data_availability - Direct call: validate_data_availability
- **Line 167:** safe_file_exists - Direct call: safe_file_exists
- **Line 258:** format_datetime - Direct call: format_datetime
- **Line 270:** safe_int - Direct call: safe_int
- **Line 271:** safe_int - Direct call: safe_int
- **Line 272:** safe_int - Direct call: safe_int
- **Line 450:** run_model_training_pipeline - Direct call: run_model_training_pipeline
- **Line 537:** main - Direct call: main
- **Line 19:** Path - Direct call: Path
- **Line 258:** get_current_datetime - Direct call: get_current_datetime
- **Line 483:** create_training_summary - Direct call: create_training_summary
- **Line 505:** create_training_summary - Direct call: create_training_summary
- **Line 528:** create_training_summary - Direct call: create_training_summary
- **Line 120:** safe_int - Direct call: safe_int
- **Line 122:** safe_int - Direct call: safe_int

---

## src/training/steps/model_training/step10_unified_regime_intelligence.py

**Undefined function calls: 34**

- **Line 2158:** deterministic_seed - Direct call: deterministic_seed
- **Line 2159:** idempotent_step - Direct call: idempotent_step
- **Line 2161:** validates - Direct call: validates
- **Line 2164:** validates - Direct call: validates
- **Line 2177:** log_execution_time - Direct call: log_execution_time
- **Line 2184:** cached - Direct call: cached
- **Line 2187:** log_call - Direct call: log_call
- **Line 2193:** circuit_breaker - Direct call: circuit_breaker
- **Line 2199:** validates - Direct call: validates
- **Line 473:** handles_errors - Direct call: handles_errors
- **Line 517:** handles_errors - Direct call: handles_errors
- **Line 1929:** handles_errors - Direct call: handles_errors
- **Line 335:** StandardScaler - Direct call: StandardScaler
- **Line 351:** ensure_directory - Direct call: ensure_directory
- **Line 492:** LabelEncoder - Direct call: LabelEncoder
- **Line 493:** LabelEncoder - Direct call: LabelEncoder
- **Line 494:** LabelEncoder - Direct call: LabelEncoder
- **Line 1474:** TensorDataset - Direct call: TensorDataset
- **Line 1480:** DataLoader - Direct call: DataLoader
- **Line 1484:** TensorDataset - Direct call: TensorDataset
- **Line 1490:** DataLoader - Direct call: DataLoader
- **Line 1665:** safe_json_dump - Direct call: safe_json_dump
- **Line 61:** Path - Direct call: Path
- **Line 1757:** standardize_price_action_probabilities - Direct call: standardize_price_action_probabilities
- **Line 1430:** ModelSpecificPruning - Direct call: ModelSpecificPruning
- **Line 1553:** criterion - Direct call: criterion
- **Line 1554:** criterion - Direct call: criterion
- **Line 1555:** criterion - Direct call: criterion
- **Line 1593:** criterion - Direct call: criterion
- **Line 1594:** criterion - Direct call: criterion
- **Line 1597:** criterion - Direct call: criterion
- **Line 1537:** criterion - Direct call: criterion
- **Line 1538:** criterion - Direct call: criterion
- **Line 1539:** criterion - Direct call: criterion

---

## src/training/steps/model_training/step10_unified_regime_intelligence_per_regime.py

**Undefined function calls: 10**

- **Line 22:** get_logger - Direct call: get_logger
- **Line 931:** traced - Direct call: traced
- **Line 932:** validates - Direct call: validates
- **Line 34:** traced - Direct call: traced
- **Line 35:** per_regime_step - Direct call: per_regime_step
- **Line 996:** test - Direct call: test
- **Line 988:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 124:** Path - Direct call: Path
- **Line 918:** Path - Direct call: Path
- **Line 128:** Path - Direct call: Path

---

## src/training/steps/model_training/step10_unified_regime_intelligence_validator.py

**Undefined function calls: 12**

- **Line 695:** handles_errors - Direct call: handles_errors
- **Line 59:** handles_errors - Direct call: handles_errors
- **Line 110:** handles_errors - Direct call: handles_errors
- **Line 208:** handles_errors - Direct call: handles_errors
- **Line 291:** handles_errors - Direct call: handles_errors
- **Line 369:** handles_errors - Direct call: handles_errors
- **Line 448:** handles_errors - Direct call: handles_errors
- **Line 526:** handles_errors - Direct call: handles_errors
- **Line 579:** handles_errors - Direct call: handles_errors
- **Line 746:** ensure_directory - Direct call: ensure_directory
- **Line 685:** ensure_directory - Direct call: ensure_directory
- **Line 687:** safe_json_dump - Direct call: safe_json_dump

---

## src/training/steps/model_training/step11_analyst_creation.py

**Undefined function calls: 13**

- **Line 19:** Path - Direct call: Path
- **Line 63:** func - Direct call: func
- **Line 71:** func - Direct call: func
- **Line 583:** accuracy_score - Direct call: accuracy_score
- **Line 619:** accuracy_score - Direct call: accuracy_score
- **Line 648:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 653:** accuracy_score - Direct call: accuracy_score
- **Line 52:** func - Direct call: func
- **Line 699:** model - Direct call: model
- **Line 700:** criterion - Direct call: criterion
- **Line 707:** model - Direct call: model
- **Line 709:** accuracy_score - Direct call: accuracy_score
- **Line 375:** create_regime_analysts - Direct call: create_regime_analysts

---

## src/training/steps/model_training/step11_analyst_creation_per_regime.py

**Undefined function calls: 10**

- **Line 22:** get_logger - Direct call: get_logger
- **Line 806:** traced - Direct call: traced
- **Line 807:** validates - Direct call: validates
- **Line 34:** traced - Direct call: traced
- **Line 35:** per_regime_step - Direct call: per_regime_step
- **Line 871:** test - Direct call: test
- **Line 863:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 124:** Path - Direct call: Path
- **Line 793:** Path - Direct call: Path
- **Line 128:** Path - Direct call: Path

---

## src/training/steps/model_training/step11_analyst_creation_validator.py

**Undefined function calls: 6**

- **Line 167:** validates - Direct call: validates
- **Line 23:** validates - Direct call: validates
- **Line 93:** validates - Direct call: validates
- **Line 122:** validates - Direct call: validates
- **Line 129:** safe_json_load - Direct call: safe_json_load
- **Line 42:** Path - Direct call: Path

---

## src/training/steps/model_training/step12_analyst_enhancement.py

**Undefined function calls: 40**

- **Line 158:** _NP_ORIGINAL_BITGEN_CTOR - Direct call: _NP_ORIGINAL_BITGEN_CTOR
- **Line 706:** accuracy_score - Direct call: accuracy_score
- **Line 900:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 917:** KFold - Direct call: KFold
- **Line 1434:** model - Direct call: model
- **Line 1461:** TensorDataset - Direct call: TensorDataset
- **Line 1462:** DataLoader - Direct call: DataLoader
- **Line 714:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 729:** SVC - Direct call: SVC
- **Line 732:** MLPClassifier - Direct call: MLPClassifier
- **Line 810:** accuracy_score - Direct call: accuracy_score
- **Line 161:** _NP_ORIGINAL_BITGEN_CTOR - Direct call: _NP_ORIGINAL_BITGEN_CTOR
- **Line 1264:** SelectKBest - Direct call: SelectKBest
- **Line 1271:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 1333:** calculate_vif_robust - Direct call: calculate_vif_robust
- **Line 1471:** student_model - Direct call: student_model
- **Line 1488:** CatBoostClassifier - Direct call: CatBoostClassifier
- **Line 1579:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 1603:** LogisticRegression - Direct call: LogisticRegression
- **Line 1626:** make_pipeline - Direct call: make_pipeline
- **Line 362:** enhance_regime_models - Direct call: enhance_regime_models
- **Line 791:** TimeoutError - Direct call: TimeoutError
- **Line 796:** StringIO - Direct call: StringIO
- **Line 820:** log_loss - Direct call: log_loss
- **Line 1193:** TreeExplainer - Direct call: TreeExplainer
- **Line 1355:** compute_shap_importance - Direct call: compute_shap_importance
- **Line 1470:** teacher_model - Direct call: teacher_model
- **Line 1626:** StandardScaler - Direct call: StandardScaler
- **Line 1626:** RBFSampler - Direct call: RBFSampler
- **Line 1626:** LinearSVC - Direct call: LinearSVC
- **Line 822:** log_loss - Direct call: log_loss
- **Line 1211:** KernelExplainer - Direct call: KernelExplainer
- **Line 1218:** permutation_importance - Direct call: permutation_importance
- **Line 1344:** mutual_info_classif - Direct call: mutual_info_classif
- **Line 1346:** mutual_info_regression - Direct call: mutual_info_regression
- **Line 1367:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 1369:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 172:** bitgen_cls - Direct call: bitgen_cls
- **Line 1207:** permutation_importance - Direct call: permutation_importance
- **Line 1540:** mutual_info_classif - Direct call: mutual_info_classif

---

## src/training/steps/model_training/step12_analyst_enhancement_per_regime.py

**Undefined function calls: 10**

- **Line 23:** get_logger - Direct call: get_logger
- **Line 899:** traced - Direct call: traced
- **Line 900:** validates - Direct call: validates
- **Line 35:** traced - Direct call: traced
- **Line 36:** per_regime_step - Direct call: per_regime_step
- **Line 964:** test - Direct call: test
- **Line 956:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 125:** Path - Direct call: Path
- **Line 886:** Path - Direct call: Path
- **Line 129:** Path - Direct call: Path

---

## src/training/steps/model_training/step12_analyst_enhancement_validator.py

**Undefined function calls: 22**

- **Line 551:** test_validator - Direct call: test_validator
- **Line 22:** Path - Direct call: Path
- **Line 253:** safe_json_load - Direct call: safe_json_load
- **Line 321:** safe_json_load - Direct call: safe_json_load
- **Line 397:** safe_json_load - Direct call: safe_json_load
- **Line 478:** callable - Direct call: callable
- **Line 549:** run_validator - Direct call: run_validator
- **Line 308:** safe_json_load - Direct call: safe_json_load
- **Line 496:** callable - Direct call: callable
- **Line 501:** callable - Direct call: callable
- **Line 91:** error - Direct call: error
- **Line 112:** failed - Direct call: failed
- **Line 133:** failed - Direct call: failed
- **Line 154:** failed - Direct call: failed
- **Line 246:** missing - Direct call: missing
- **Line 267:** failed - Direct call: failed
- **Line 317:** missing - Direct call: missing
- **Line 393:** missing - Direct call: missing
- **Line 412:** failed - Direct call: failed
- **Line 439:** missing - Direct call: missing
- **Line 485:** callable - Direct call: callable
- **Line 489:** callable - Direct call: callable

---

## src/training/steps/model_training/step13_analyst_ensemble_creation.py

**Undefined function calls: 2**

- **Line 54:** handles_errors - Direct call: handles_errors
- **Line 161:** OptimizedFeatureSelectionManager - Direct call: OptimizedFeatureSelectionManager

---

## src/training/steps/model_training/step13_analyst_ensemble_creation_per_regime.py

**Undefined function calls: 10**

- **Line 22:** get_logger - Direct call: get_logger
- **Line 817:** traced - Direct call: traced
- **Line 818:** validates - Direct call: validates
- **Line 34:** traced - Direct call: traced
- **Line 35:** per_regime_step - Direct call: per_regime_step
- **Line 882:** test - Direct call: test
- **Line 874:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 124:** Path - Direct call: Path
- **Line 804:** Path - Direct call: Path
- **Line 128:** Path - Direct call: Path

---

## src/training/steps/model_training/step13_analyst_ensemble_creation_validator.py

**Undefined function calls: 16**

- **Line 24:** handles_errors - Direct call: handles_errors
- **Line 146:** safe_json_load - Direct call: safe_json_load
- **Line 71:** success - Direct call: success
- **Line 76:** failed - Direct call: failed
- **Line 83:** error - Direct call: error
- **Line 113:** missing - Direct call: missing
- **Line 115:** missing - Direct call: missing
- **Line 122:** error - Direct call: error
- **Line 141:** missing - Direct call: missing
- **Line 160:** failed - Direct call: failed
- **Line 165:** failed - Direct call: failed
- **Line 175:** failed - Direct call: failed
- **Line 180:** failed - Direct call: failed
- **Line 189:** warning - Direct call: warning
- **Line 194:** warning - Direct call: warning
- **Line 203:** error - Direct call: error

---

## src/training/steps/model_training/step14_tactician_labeling.py

**Undefined function calls: 11**

- **Line 1164:** test - Direct call: test
- **Line 1162:** run_step - Direct call: run_step
- **Line 917:** Path - Direct call: Path
- **Line 811:** Path - Direct call: Path
- **Line 820:** Path - Direct call: Path
- **Line 926:** ParquetDatasetManager - Direct call: ParquetDatasetManager
- **Line 965:** ParquetDatasetManager - Direct call: ParquetDatasetManager
- **Line 936:** log_io_operation - Direct call: log_io_operation
- **Line 946:** log_dataframe_overview - Direct call: log_dataframe_overview
- **Line 975:** log_io_operation - Direct call: log_io_operation
- **Line 985:** log_dataframe_overview - Direct call: log_dataframe_overview

---

## src/training/steps/model_training/step14_tactician_labeling_per_regime.py

**Undefined function calls: 10**

- **Line 22:** get_logger - Direct call: get_logger
- **Line 857:** traced - Direct call: traced
- **Line 858:** validates - Direct call: validates
- **Line 34:** traced - Direct call: traced
- **Line 35:** per_regime_step - Direct call: per_regime_step
- **Line 922:** test - Direct call: test
- **Line 914:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 124:** Path - Direct call: Path
- **Line 844:** Path - Direct call: Path
- **Line 128:** Path - Direct call: Path

---

## src/training/steps/model_training/step14_tactician_labeling_validator.py

**Undefined function calls: 31**

- **Line 685:** test_validator - Direct call: test_validator
- **Line 21:** Path - Direct call: Path
- **Line 683:** run_validator - Direct call: run_validator
- **Line 62:** error - Direct call: error
- **Line 72:** failed - Direct call: failed
- **Line 82:** failed - Direct call: failed
- **Line 92:** failed - Direct call: failed
- **Line 102:** failed - Direct call: failed
- **Line 112:** error - Direct call: error
- **Line 319:** missing - Direct call: missing
- **Line 511:** missing - Direct call: missing
- **Line 159:** error - Direct call: error
- **Line 324:** validation_error - Direct call: validation_error
- **Line 191:** ParquetDatasetManager - Direct call: ParquetDatasetManager
- **Line 263:** error - Direct call: error
- **Line 267:** error - Direct call: error
- **Line 355:** ParquetDatasetManager - Direct call: ParquetDatasetManager
- **Line 409:** error - Direct call: error
- **Line 418:** error - Direct call: error
- **Line 504:** missing - Direct call: missing
- **Line 300:** error - Direct call: error
- **Line 303:** error - Direct call: error
- **Line 614:** error - Direct call: error
- **Line 218:** log_io_operation - Direct call: log_io_operation
- **Line 229:** log_dataframe_overview - Direct call: log_dataframe_overview
- **Line 234:** log_io_operation - Direct call: log_io_operation
- **Line 381:** log_io_operation - Direct call: log_io_operation
- **Line 391:** log_dataframe_overview - Direct call: log_dataframe_overview
- **Line 396:** log_io_operation - Direct call: log_io_operation
- **Line 432:** log_io_operation - Direct call: log_io_operation
- **Line 444:** log_io_operation - Direct call: log_io_operation

---

## src/training/steps/model_training/step15_tactician_specialist_training.py

**Undefined function calls: 19**

- **Line 1333:** timeout - Direct call: timeout
- **Line 1335:** model_validation - Direct call: model_validation
- **Line 1342:** pipeline_checkpoint - Direct call: pipeline_checkpoint
- **Line 1347:** intelligent_caching - Direct call: intelligent_caching
- **Line 1353:** adaptive_resource_allocation - Direct call: adaptive_resource_allocation
- **Line 1358:** comprehensive_validation - Direct call: comprehensive_validation
- **Line 112:** Path - Direct call: Path
- **Line 1429:** TacticianSpecialistTrainingStep - Direct call: TacticianSpecialistTrainingStep
- **Line 1611:** test - Direct call: test
- **Line 991:** LogisticRegression - Direct call: LogisticRegression
- **Line 998:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV
- **Line 1226:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 1609:** run_step - Direct call: run_step
- **Line 109:** Path - Direct call: Path
- **Line 1008:** accuracy_score - Direct call: accuracy_score
- **Line 1144:** accuracy_score - Direct call: accuracy_score
- **Line 1239:** accuracy_score - Direct call: accuracy_score
- **Line 35:** func - Direct call: func
- **Line 130:** __import__ - Direct call: __import__

---

## src/training/steps/model_training/step15_tactician_specialist_training_per_regime.py

**Undefined function calls: 6**

- **Line 963:** test - Direct call: test
- **Line 955:** run_per_regime_step - Direct call: run_per_regime_step
- **Line 34:** func - Direct call: func
- **Line 152:** Path - Direct call: Path
- **Line 885:** Path - Direct call: Path
- **Line 156:** Path - Direct call: Path

---

## src/training/steps/model_training/step15_tactician_specialist_training_validator.py

**Undefined function calls: 27**

- **Line 554:** test_validator - Direct call: test_validator
- **Line 19:** Path - Direct call: Path
- **Line 196:** safe_json_load - Direct call: safe_json_load
- **Line 286:** safe_json_load - Direct call: safe_json_load
- **Line 486:** callable - Direct call: callable
- **Line 552:** run_validator - Direct call: run_validator
- **Line 64:** error - Direct call: error
- **Line 74:** failed - Direct call: failed
- **Line 84:** failed - Direct call: failed
- **Line 94:** failed - Direct call: failed
- **Line 104:** failed - Direct call: failed
- **Line 374:** safe_json_load - Direct call: safe_json_load
- **Line 502:** callable - Direct call: callable
- **Line 506:** callable - Direct call: callable
- **Line 157:** missing - Direct call: missing
- **Line 165:** error - Direct call: error
- **Line 434:** callable - Direct call: callable
- **Line 440:** callable - Direct call: callable
- **Line 292:** error - Direct call: error
- **Line 300:** error - Direct call: error
- **Line 340:** error - Direct call: error
- **Line 492:** callable - Direct call: callable
- **Line 294:** error - Direct call: error
- **Line 437:** missing - Direct call: missing
- **Line 443:** missing - Direct call: missing
- **Line 466:** error - Direct call: error
- **Line 496:** callable - Direct call: callable

---

## src/training/steps/model_training/test_refactored_components.py

**Undefined function calls: 12**

- **Line 11:** datetime - Direct call: datetime
- **Line 63:** DataValidator - Direct call: DataValidator
- **Line 70:** DataCleaner - Direct call: DataCleaner
- **Line 86:** DataFormatConverter - Direct call: DataFormatConverter
- **Line 109:** QualityMetricsCalculator - Direct call: QualityMetricsCalculator
- **Line 119:** DataIntegrityChecker - Direct call: DataIntegrityChecker
- **Line 128:** AnomalyDetector - Direct call: AnomalyDetector
- **Line 158:** DataCleaner - Direct call: DataCleaner
- **Line 163:** DataValidator - Direct call: DataValidator
- **Line 167:** DataIntegrityChecker - Direct call: DataIntegrityChecker
- **Line 171:** AnomalyDetector - Direct call: AnomalyDetector
- **Line 175:** QualityMetricsCalculator - Direct call: QualityMetricsCalculator

---

## src/training/steps/model_training/tests/test_step10_unified_regime_intelligence.py

**Undefined function calls: 5**

- **Line 35:** RegimeIntelligenceAnalyzer - Direct call: RegimeIntelligenceAnalyzer
- **Line 105:** RegimeMetricsCalculator - Direct call: RegimeMetricsCalculator
- **Line 170:** RegimeTransitionAnalyzer - Direct call: RegimeTransitionAnalyzer
- **Line 250:** UnifiedRegimeIntelligenceStep - Direct call: UnifiedRegimeIntelligenceStep
- **Line 326:** Mock - Direct call: Mock

---

## src/training/steps/model_training/tests/test_step11_analyst_creation.py

**Undefined function calls: 9**

- **Line 76:** patch - Direct call: patch
- **Line 100:** patch - Direct call: patch
- **Line 34:** AnalystModelBuilder - Direct call: AnalystModelBuilder
- **Line 84:** MagicMock - Direct call: MagicMock
- **Line 108:** MagicMock - Direct call: MagicMock
- **Line 150:** MultiOutputAnalystBuilder - Direct call: MultiOutputAnalystBuilder
- **Line 231:** AnalystCreationStep - Direct call: AnalystCreationStep
- **Line 387:** Mock - Direct call: Mock
- **Line 436:** Mock - Direct call: Mock

---

## src/training/steps/model_training/tests/test_step12_analyst_enhancement.py

**Undefined function calls: 13**

- **Line 37:** AnalystEnhancer - Direct call: AnalystEnhancer
- **Line 129:** Mock - Direct call: Mock
- **Line 156:** FeatureAugmenter - Direct call: FeatureAugmenter
- **Line 229:** ModelOptimizer - Direct call: ModelOptimizer
- **Line 236:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 266:** PerformanceAnalyzer - Direct call: PerformanceAnalyzer
- **Line 344:** AnalystEnhancementStep - Direct call: AnalystEnhancementStep
- **Line 79:** Mock - Direct call: Mock
- **Line 442:** Mock - Direct call: Mock
- **Line 472:** Mock - Direct call: Mock
- **Line 473:** Mock - Direct call: Mock
- **Line 67:** Mock - Direct call: Mock
- **Line 539:** Mock - Direct call: Mock

---

## src/training/steps/model_training/validation/__init__.py

**Undefined function calls: 1**

- **Line 10:** Path - Direct call: Path

---

## src/training/steps/model_training/validation/step16_confidence_calibration.py

**Undefined function calls: 34**

- **Line 418:** deterministic_seed - Direct call: deterministic_seed
- **Line 419:** idempotent_step - Direct call: idempotent_step
- **Line 420:** validates - Direct call: validates
- **Line 421:** timeout - Direct call: timeout
- **Line 422:** validates - Direct call: validates
- **Line 423:** log_execution_time - Direct call: log_execution_time
- **Line 424:** cached - Direct call: cached
- **Line 425:** log_call - Direct call: log_call
- **Line 426:** circuit_breaker - Direct call: circuit_breaker
- **Line 427:** validates - Direct call: validates
- **Line 42:** handles_errors - Direct call: handles_errors
- **Line 48:** handles_errors - Direct call: handles_errors
- **Line 558:** test - Direct call: test
- **Line 377:** accuracy_score - Direct call: accuracy_score
- **Line 378:** f1_score - Direct call: f1_score
- **Line 512:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV
- **Line 557:** run_step - Direct call: run_step
- **Line 293:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV
- **Line 295:** accuracy_score - Direct call: accuracy_score
- **Line 296:** f1_score - Direct call: f1_score
- **Line 323:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV
- **Line 325:** accuracy_score - Direct call: accuracy_score
- **Line 326:** f1_score - Direct call: f1_score
- **Line 346:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV
- **Line 348:** accuracy_score - Direct call: accuracy_score
- **Line 349:** f1_score - Direct call: f1_score
- **Line 70:** heartbeat - Direct call: heartbeat
- **Line 93:** heartbeat - Direct call: heartbeat
- **Line 107:** heartbeat - Direct call: heartbeat
- **Line 117:** heartbeat - Direct call: heartbeat
- **Line 211:** error - Direct call: error
- **Line 270:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV
- **Line 272:** accuracy_score - Direct call: accuracy_score
- **Line 273:** f1_score - Direct call: f1_score

---

## src/training/steps/model_training/validation/step17_final_parameters_optimization.py

**Undefined function calls: 10**

- **Line 800:** test - Direct call: test
- **Line 797:** run_step - Direct call: run_step
- **Line 90:** __import__ - Direct call: __import__
- **Line 132:** heartbeat - Direct call: heartbeat
- **Line 139:** heartbeat - Direct call: heartbeat
- **Line 151:** heartbeat - Direct call: heartbeat
- **Line 158:** heartbeat - Direct call: heartbeat
- **Line 162:** heartbeat - Direct call: heartbeat
- **Line 166:** heartbeat - Direct call: heartbeat
- **Line 15:** Path - Direct call: Path

---

## src/training/steps/model_training/validation/step18_walk_forward_validation.py

**Undefined function calls: 14**

- **Line 301:** deterministic_seed - Direct call: deterministic_seed
- **Line 302:** idempotent_step - Direct call: idempotent_step
- **Line 304:** validates - Direct call: validates
- **Line 306:** timeout - Direct call: timeout
- **Line 307:** validates - Direct call: validates
- **Line 326:** log_execution_time - Direct call: log_execution_time
- **Line 333:** cached - Direct call: cached
- **Line 336:** log_call - Direct call: log_call
- **Line 342:** circuit_breaker - Direct call: circuit_breaker
- **Line 348:** validates - Direct call: validates
- **Line 407:** test - Direct call: test
- **Line 405:** run_step - Direct call: run_step
- **Line 57:** __import__ - Direct call: __import__
- **Line 174:** validation_error - Direct call: validation_error

---

## src/training/steps/model_training/validation/step19_monte_carlo_validation.py

**Undefined function calls: 11**

- **Line 225:** deterministic_seed - Direct call: deterministic_seed
- **Line 226:** idempotent_step - Direct call: idempotent_step
- **Line 227:** timeout - Direct call: timeout
- **Line 228:** validates - Direct call: validates
- **Line 253:** log_execution_time - Direct call: log_execution_time
- **Line 260:** cached - Direct call: cached
- **Line 263:** log_call - Direct call: log_call
- **Line 269:** circuit_breaker - Direct call: circuit_breaker
- **Line 322:** test - Direct call: test
- **Line 320:** run_step - Direct call: run_step
- **Line 145:** ParquetDatasetManager - Direct call: ParquetDatasetManager

---

## src/training/steps/model_training/validation/step20_ab_testing.py

**Undefined function calls: 6**

- **Line 24:** get_logger - Direct call: get_logger
- **Line 43:** ensure_directory - Direct call: ensure_directory
- **Line 57:** safe_json_dump - Direct call: safe_json_dump
- **Line 84:** _test - Direct call: _test
- **Line 82:** run_step - Direct call: run_step
- **Line 10:** Path - Direct call: Path

---

## src/training/steps/model_training/validation_components/ab_testing_step.py

**Undefined function calls: 2**

- **Line 26:** handles_errors - Direct call: handles_errors
- **Line 35:** Impl - Direct call: Impl

---

## src/training/steps/model_training/validation_components/confidence_calibration_step.py

**Undefined function calls: 9**

- **Line 40:** handles_errors - Direct call: handles_errors
- **Line 59:** TimeSeriesSplit - Direct call: TimeSeriesSplit
- **Line 194:** minimize - Direct call: minimize
- **Line 107:** brier_score_loss - Direct call: brier_score_loss
- **Line 108:** log_loss - Direct call: log_loss
- **Line 111:** CalibratedClassifierCV - Direct call: CalibratedClassifierCV
- **Line 127:** brier_score_loss - Direct call: brier_score_loss
- **Line 128:** log_loss - Direct call: log_loss
- **Line 193:** log_loss - Direct call: log_loss

---

## src/training/steps/model_training/validation_components/monte_carlo_validation_step.py

**Undefined function calls: 7**

- **Line 37:** handles_errors - Direct call: handles_errors
- **Line 96:** train_test_split - Direct call: train_test_split
- **Line 92:** train_test_split - Direct call: train_test_split
- **Line 104:** accuracy_score - Direct call: accuracy_score
- **Line 104:** precision_score - Direct call: precision_score
- **Line 104:** recall_score - Direct call: recall_score
- **Line 104:** f1_score - Direct call: f1_score

---

## src/training/steps/model_training/validation_components/walk_forward_validation_step.py

**Undefined function calls: 5**

- **Line 35:** handles_errors - Direct call: handles_errors
- **Line 139:** accuracy_score - Direct call: accuracy_score
- **Line 139:** precision_score - Direct call: precision_score
- **Line 139:** recall_score - Direct call: recall_score
- **Line 139:** f1_score - Direct call: f1_score

---

## src/training/steps/optimisation/optimisation_pipeline_validator.py

**Undefined function calls: 15**

- **Line 36:** validates - Direct call: validates
- **Line 139:** validates - Direct call: validates
- **Line 211:** validates - Direct call: validates
- **Line 293:** validates - Direct call: validates
- **Line 363:** traced - Direct call: traced
- **Line 364:** log_execution_time - Direct call: log_execution_time
- **Line 32:** DataQualityFramework - Direct call: DataQualityFramework
- **Line 376:** get_current_datetime - Direct call: get_current_datetime
- **Line 378:** format_datetime - Direct call: format_datetime
- **Line 430:** get_current_datetime - Direct call: get_current_datetime
- **Line 431:** format_datetime - Direct call: format_datetime
- **Line 179:** safe_file_exists - Direct call: safe_file_exists
- **Line 59:** safe_file_exists - Direct call: safe_file_exists
- **Line 310:** safe_file_exists - Direct call: safe_file_exists
- **Line 318:** safe_json_load - Direct call: safe_json_load

---

## src/training/steps/optimisation/step16_confidence_calibration_per_regime.py

**Undefined function calls: 11**

- **Line 305:** Path - Direct call: Path
- **Line 1240:** run_step_enhanced - Direct call: run_step_enhanced
- **Line 1258:** test - Direct call: test
- **Line 69:** Path - Direct call: Path
- **Line 1250:** run_step_enhanced - Direct call: run_step_enhanced
- **Line 19:** Path - Direct call: Path
- **Line 65:** Path - Direct call: Path
- **Line 68:** Path - Direct call: Path
- **Line 459:** Path - Direct call: Path
- **Line 1146:** Path - Direct call: Path
- **Line 463:** Path - Direct call: Path

---

## src/training/steps/optimisation/step16_confidence_calibration_validator.py

**Undefined function calls: 2**

- **Line 14:** Path - Direct call: Path
- **Line 122:** missing - Direct call: missing

---

## src/training/steps/optimisation/step17_enhanced_multi_objective_optimization.py

**Undefined function calls: 17**

- **Line 257:** handles_errors - Direct call: handles_errors
- **Line 272:** handles_errors - Direct call: handles_errors
- **Line 83:** get_config_manager - Direct call: get_config_manager
- **Line 419:** NSGAIISampler - Direct call: NSGAIISampler
- **Line 425:** TPESampler - Direct call: TPESampler
- **Line 433:** SuccessiveHalvingPruner - Direct call: SuccessiveHalvingPruner
- **Line 439:** MedianPruner - Direct call: MedianPruner
- **Line 469:** get_search_space - Direct call: get_search_space
- **Line 490:** update_optimizable_config - Direct call: update_optimizable_config
- **Line 594:** TPESampler - Direct call: TPESampler
- **Line 595:** MedianPruner - Direct call: MedianPruner
- **Line 626:** get_search_space - Direct call: get_search_space
- **Line 694:** update_optimizable_config - Direct call: update_optimizable_config
- **Line 650:** update_optimizable_config - Direct call: update_optimizable_config
- **Line 488:** get_search_space - Direct call: get_search_space
- **Line 692:** get_search_space - Direct call: get_search_space
- **Line 648:** get_search_space - Direct call: get_search_space

---

## src/training/steps/optimisation/step17_final_parameters_optimization_new.py

**Undefined function calls: 13**

- **Line 34:** handles_errors - Direct call: handles_errors
- **Line 52:** handles_errors - Direct call: handles_errors
- **Line 31:** get_config_manager - Direct call: get_config_manager
- **Line 32:** get_optimizable_parameters - Direct call: get_optimizable_parameters
- **Line 242:** get_search_space - Direct call: get_search_space
- **Line 316:** update_optimizable_config - Direct call: update_optimizable_config
- **Line 76:** heartbeat - Direct call: heartbeat
- **Line 91:** heartbeat - Direct call: heartbeat
- **Line 103:** heartbeat - Direct call: heartbeat
- **Line 114:** heartbeat - Direct call: heartbeat
- **Line 128:** heartbeat - Direct call: heartbeat
- **Line 141:** heartbeat - Direct call: heartbeat
- **Line 219:** update_optimizable_config - Direct call: update_optimizable_config

---

## src/training/steps/optimisation/step17_final_parameters_optimization_validator.py

**Undefined function calls: 16**

- **Line 434:** test_validator - Direct call: test_validator
- **Line 16:** Path - Direct call: Path
- **Line 432:** run_validator - Direct call: run_validator
- **Line 62:** error - Direct call: error
- **Line 72:** failed - Direct call: failed
- **Line 78:** failed - Direct call: failed
- **Line 88:** failed - Direct call: failed
- **Line 98:** failed - Direct call: failed
- **Line 181:** safe_json_load - Direct call: safe_json_load
- **Line 246:** safe_json_load - Direct call: safe_json_load
- **Line 333:** safe_json_load - Direct call: safe_json_load
- **Line 151:** missing - Direct call: missing
- **Line 156:** error - Direct call: error
- **Line 338:** error - Direct call: error
- **Line 264:** error - Direct call: error
- **Line 340:** error - Direct call: error

---

## src/training/steps/optimisation/step17_parameter_optimization_wrapper.py

**Undefined function calls: 3**

- **Line 24:** handles_errors - Direct call: handles_errors
- **Line 37:** FinalParametersOptimizationStepNew - Direct call: FinalParametersOptimizationStepNew
- **Line 43:** FinalParametersOptimizationStep - Direct call: FinalParametersOptimizationStep

---

## src/training/steps/optimisation/step_validators.py

**Undefined function calls: 10**

- **Line 29:** validates - Direct call: validates
- **Line 188:** validates - Direct call: validates
- **Line 332:** validates - Direct call: validates
- **Line 27:** DataQualityFramework - Direct call: DataQualityFramework
- **Line 186:** DataQualityFramework - Direct call: DataQualityFramework
- **Line 89:** safe_file_exists - Direct call: safe_file_exists
- **Line 167:** Path - Direct call: Path
- **Line 247:** safe_file_exists - Direct call: safe_file_exists
- **Line 107:** safe_file_exists - Direct call: safe_file_exists
- **Line 150:** safe_file_exists - Direct call: safe_file_exists

---

## src/training/steps/run_all_pipelines.py

**Undefined function calls: 36**

- **Line 124:** handle_errors - Direct call: handle_errors
- **Line 125:** monitor_pipeline_step - Direct call: monitor_pipeline_step
- **Line 168:** handle_errors - Direct call: handle_errors
- **Line 169:** monitor_pipeline_step - Direct call: monitor_pipeline_step
- **Line 197:** handle_errors - Direct call: handle_errors
- **Line 198:** monitor_pipeline_step - Direct call: monitor_pipeline_step
- **Line 212:** handle_errors - Direct call: handle_errors
- **Line 213:** validate_data_quality - Direct call: validate_data_quality
- **Line 214:** monitor_pipeline_step - Direct call: monitor_pipeline_step
- **Line 282:** handle_errors - Direct call: handle_errors
- **Line 283:** monitor_pipeline_step - Direct call: monitor_pipeline_step
- **Line 309:** handle_errors - Direct call: handle_errors
- **Line 310:** monitor_pipeline_step - Direct call: monitor_pipeline_step
- **Line 391:** handle_errors - Direct call: handle_errors
- **Line 392:** monitor_pipeline_step - Direct call: monitor_pipeline_step
- **Line 99:** ValidatorOrchestrator - Direct call: ValidatorOrchestrator
- **Line 100:** DataQualityFramework - Direct call: DataQualityFramework
- **Line 101:** DataFormattingFramework - Direct call: DataFormattingFramework
- **Line 699:** main - Direct call: main
- **Line 30:** Path - Direct call: Path
- **Line 104:** Path - Direct call: Path
- **Line 202:** safe_file_exists - Direct call: safe_file_exists
- **Line 556:** get_report_manager - Direct call: get_report_manager
- **Line 174:** format_datetime - Direct call: format_datetime
- **Line 203:** safe_json_load - Direct call: safe_json_load
- **Line 331:** pipeline_func - Direct call: pipeline_func
- **Line 586:** format_datetime - Direct call: format_datetime
- **Line 606:** get_report_collector - Direct call: get_report_collector
- **Line 142:** safe_file_exists - Direct call: safe_file_exists
- **Line 142:** safe_file_exists - Direct call: safe_file_exists
- **Line 174:** get_current_datetime - Direct call: get_current_datetime
- **Line 224:** safe_file_exists - Direct call: safe_file_exists
- **Line 586:** get_current_datetime - Direct call: get_current_datetime
- **Line 624:** Path - Direct call: Path
- **Line 624:** format_datetime - Direct call: format_datetime
- **Line 624:** get_current_datetime - Direct call: get_current_datetime

---

## src/training/steps/step06_enhanced_validation_framework.py

**Undefined function calls: 15**

- **Line 60:** field - Direct call: field
- **Line 61:** field - Direct call: field
- **Line 66:** field - Direct call: field
- **Line 67:** field - Direct call: field
- **Line 68:** field - Direct call: field
- **Line 78:** field - Direct call: field
- **Line 79:** field - Direct call: field
- **Line 87:** deque - Direct call: deque
- **Line 88:** defaultdict - Direct call: defaultdict
- **Line 89:** defaultdict - Direct call: defaultdict
- **Line 90:** defaultdict - Direct call: defaultdict
- **Line 698:** func - Direct call: func
- **Line 279:** rule - Direct call: rule
- **Line 629:** func - Direct call: func
- **Line 624:** TimeoutError - Direct call: TimeoutError

---

## src/training/steps/step06_labeling_components/fractional_triple_barrier_labeling.py

**Undefined function calls: 4**

- **Line 69:** handles_errors - Direct call: handles_errors
- **Line 70:** traced - Direct call: traced
- **Line 46:** OptimizedTripleBarrierLabeling - Direct call: OptimizedTripleBarrierLabeling
- **Line 67:** get_logger - Direct call: get_logger

---

## src/training/steps/step06_labeling_components/optimized_triple_barrier_labeling.py

**Undefined function calls: 2**

- **Line 39:** nullcontext - Direct call: nullcontext
- **Line 325:** callable - Direct call: callable

---

## src/training/steps/step06_labeling_components/profit_based_feature_engineering.py

**Undefined function calls: 13**

- **Line 662:** handles_errors - Direct call: handles_errors
- **Line 26:** jit - Direct call: jit
- **Line 37:** jit - Direct call: jit
- **Line 48:** jit - Direct call: jit
- **Line 154:** handles_errors - Direct call: handles_errors
- **Line 249:** handles_errors - Direct call: handles_errors
- **Line 272:** handles_errors - Direct call: handles_errors
- **Line 309:** handles_errors - Direct call: handles_errors
- **Line 352:** handles_errors - Direct call: handles_errors
- **Line 391:** handles_errors - Direct call: handles_errors
- **Line 435:** handles_errors - Direct call: handles_errors
- **Line 476:** handles_errors - Direct call: handles_errors
- **Line 630:** mutual_info_regression - Direct call: mutual_info_regression

---

## src/training/steps/step06_labeling_components/regime_aware_triple_barrier_labeling.py

**Undefined function calls: 6**

- **Line 185:** handles_errors - Direct call: handles_errors
- **Line 186:** traced - Direct call: traced
- **Line 132:** get_logger - Direct call: get_logger
- **Line 412:** OptimizedTripleBarrierLabeling - Direct call: OptimizedTripleBarrierLabeling
- **Line 301:** callable - Direct call: callable
- **Line 504:** Path - Direct call: Path

---

## src/training/steps/step06_labeling_components/regime_specific_triple_barrier_optimizer.py

**Undefined function calls: 2**

- **Line 60:** RegimeAwareTripleBarrierLabeling - Direct call: RegimeAwareTripleBarrierLabeling
- **Line 62:** OptimizedTripleBarrierLabeling - Direct call: OptimizedTripleBarrierLabeling

---

## src/training/steps/step06_validation_orchestrator.py

**Undefined function calls: 3**

- **Line 101:** Path - Direct call: Path
- **Line 554:** main - Direct call: main
- **Line 543:** run_step06_comprehensive_validation - Direct call: run_step06_comprehensive_validation

---

## src/training/steps/step08_regime_data_splitting.py

**Undefined function calls: 35**

- **Line 573:** deterministic_seed - Direct call: deterministic_seed
- **Line 574:** idempotent_step - Direct call: idempotent_step
- **Line 575:** artifact_write_lock - Direct call: artifact_write_lock
- **Line 576:** nan_inf_and_constant_guard - Direct call: nan_inf_and_constant_guard
- **Line 577:** artifact_versioning - Direct call: artifact_versioning
- **Line 578:** time_budget_watchdog - Direct call: time_budget_watchdog
- **Line 579:** validate_step_prerequisites - Direct call: validate_step_prerequisites
- **Line 590:** secure_data_processing - Direct call: secure_data_processing
- **Line 596:** prevent_data_leakage - Direct call: prevent_data_leakage
- **Line 601:** resource_monitor - Direct call: resource_monitor
- **Line 608:** memory_efficient - Direct call: memory_efficient
- **Line 614:** debug_training_step - Direct call: debug_training_step
- **Line 620:** circuit_breaker_protection - Direct call: circuit_breaker_protection
- **Line 626:** validate_step_output - Direct call: validate_step_output
- **Line 632:** quality_gate - Direct call: quality_gate
- **Line 637:** handle_errors - Direct call: handle_errors
- **Line 137:** with_tracing_span - Direct call: with_tracing_span
- **Line 138:** handle_errors - Direct call: handle_errors
- **Line 146:** with_enhanced_mlflow_logging - Direct call: with_enhanced_mlflow_logging
- **Line 147:** with_tracing_span - Direct call: with_tracing_span
- **Line 148:** handle_errors - Direct call: handle_errors
- **Line 435:** with_tracing_span - Direct call: with_tracing_span
- **Line 436:** handle_errors - Direct call: handle_errors
- **Line 543:** with_tracing_span - Direct call: with_tracing_span
- **Line 544:** handle_errors - Direct call: handle_errors
- **Line 53:** func - Direct call: func
- **Line 121:** PipelineStandards - Direct call: PipelineStandards
- **Line 11:** Path - Direct call: Path
- **Line 368:** create_detailed_step_report - Direct call: create_detailed_step_report
- **Line 379:** log_step_report - Direct call: log_step_report
- **Line 415:** log_step_metrics - Direct call: log_step_metrics
- **Line 655:** run_step - Direct call: run_step
- **Line 661:** _test - Direct call: _test
- **Line 398:** log_step_report - Direct call: log_step_report
- **Line 664:** _test - Direct call: _test

---

## src/training/steps/step5_labeling.py

**Undefined function calls: 4**

- **Line 90:** Path - Direct call: Path
- **Line 96:** Path - Direct call: Path
- **Line 114:** PipelineStandards - Direct call: PipelineStandards
- **Line 215:** Path - Direct call: Path

---

## src/training/steps_1_7_comprehensive_executor.py

**Undefined function calls: 12**

- **Line 377:** main - Direct call: main
- **Line 6:** Path - Direct call: Path
- **Line 49:** DataCollectionStep - Direct call: DataCollectionStep
- **Line 49:** DataConverterStep - Direct call: DataConverterStep
- **Line 49:** DataReadingStep - Direct call: DataReadingStep
- **Line 49:** EnhancedHMMRegimeDiscoveryStep - Direct call: EnhancedHMMRegimeDiscoveryStep
- **Line 49:** RegimeDataSplittingStep - Direct call: RegimeDataSplittingStep
- **Line 49:** LabelingStep - Direct call: LabelingStep
- **Line 49:** FeatureEngineeringStep - Direct call: FeatureEngineeringStep
- **Line 49:** EnhancedMatrixOperationsStep - Direct call: EnhancedMatrixOperationsStep
- **Line 349:** log_step_report - Direct call: log_step_report
- **Line 172:** validate_step_dependencies - Direct call: validate_step_dependencies

---

## src/training/timeframe_relevance_analyzer.py

**Undefined function calls: 3**

- **Line 31:** handles_errors - Direct call: handles_errors
- **Line 399:** Path - Direct call: Path
- **Line 409:** Path - Direct call: Path

---

## src/training/tpsl_optimizer.py

**Undefined function calls: 2**

- **Line 31:** get_logger - Direct call: get_logger
- **Line 214:** LogisticRegression - Direct call: LogisticRegression

---

## src/training/training_manager.py

**Undefined function calls: 58**

- **Line 1032:** handles_errors - Direct call: handles_errors
- **Line 57:** handles_errors - Direct call: handles_errors
- **Line 95:** handles_errors - Direct call: handles_errors
- **Line 122:** handles_errors - Direct call: handles_errors
- **Line 161:** handles_errors - Direct call: handles_errors
- **Line 204:** handles_errors - Direct call: handles_errors
- **Line 223:** handles_errors - Direct call: handles_errors
- **Line 242:** handles_errors - Direct call: handles_errors
- **Line 261:** handles_errors - Direct call: handles_errors
- **Line 280:** handles_errors - Direct call: handles_errors
- **Line 340:** handles_errors - Direct call: handles_errors
- **Line 369:** handles_errors - Direct call: handles_errors
- **Line 421:** handles_errors - Direct call: handles_errors
- **Line 483:** handles_errors - Direct call: handles_errors
- **Line 533:** handles_errors - Direct call: handles_errors
- **Line 912:** handles_errors - Direct call: handles_errors
- **Line 933:** handles_errors - Direct call: handles_errors
- **Line 958:** handles_errors - Direct call: handles_errors
- **Line 1007:** handles_errors - Direct call: handles_errors
- **Line 196:** FeatureIntegrationManager - Direct call: FeatureIntegrationManager
- **Line 360:** invalid - Direct call: invalid
- **Line 364:** invalid - Direct call: invalid
- **Line 80:** invalid - Direct call: invalid
- **Line 92:** failed - Direct call: failed
- **Line 120:** error - Direct call: error
- **Line 132:** invalid - Direct call: invalid
- **Line 137:** invalid - Direct call: invalid
- **Line 149:** error - Direct call: error
- **Line 158:** error - Direct call: error
- **Line 189:** initialization_error - Direct call: initialization_error
- **Line 221:** initialization_error - Direct call: initialization_error
- **Line 259:** initialization_error - Direct call: initialization_error
- **Line 277:** initialization_error - Direct call: initialization_error
- **Line 355:** missing - Direct call: missing
- **Line 418:** error - Direct call: error
- **Line 480:** error - Direct call: error
- **Line 530:** error - Direct call: error
- **Line 574:** error - Direct call: error
- **Line 594:** error - Direct call: error
- **Line 646:** error - Direct call: error
- **Line 669:** error - Direct call: error
- **Line 688:** validation_error - Direct call: validation_error
- **Line 707:** error - Direct call: error
- **Line 726:** validation_error - Direct call: validation_error
- **Line 745:** error - Direct call: error
- **Line 764:** error - Direct call: error
- **Line 785:** error - Direct call: error
- **Line 804:** error - Direct call: error
- **Line 823:** validation_error - Direct call: validation_error
- **Line 842:** error - Direct call: error
- **Line 859:** error - Direct call: error
- **Line 875:** error - Direct call: error
- **Line 893:** error - Direct call: error
- **Line 909:** error - Direct call: error
- **Line 931:** error - Direct call: error
- **Line 955:** error - Direct call: error
- **Line 980:** error - Direct call: error
- **Line 1027:** error - Direct call: error

---

## src/training/training_orchestrator.py

**Undefined function calls: 32**

- **Line 691:** handles_errors - Direct call: handles_errors
- **Line 40:** handles_errors - Direct call: handles_errors
- **Line 78:** handles_errors - Direct call: handles_errors
- **Line 382:** handles_errors - Direct call: handles_errors
- **Line 422:** handles_errors - Direct call: handles_errors
- **Line 463:** handles_errors - Direct call: handles_errors
- **Line 512:** handles_errors - Direct call: handles_errors
- **Line 547:** handles_errors - Direct call: handles_errors
- **Line 621:** handles_errors - Direct call: handles_errors
- **Line 665:** handles_errors - Direct call: handles_errors
- **Line 90:** StepDependencyValidator - Direct call: StepDependencyValidator
- **Line 98:** BaseValidator - Direct call: BaseValidator
- **Line 393:** ModelTrainer - Direct call: ModelTrainer
- **Line 399:** OptimizationManager - Direct call: OptimizationManager
- **Line 405:** EnsembleManager - Direct call: EnsembleManager
- **Line 411:** CalibrationManager - Direct call: CalibrationManager
- **Line 63:** invalid - Direct call: invalid
- **Line 419:** failed - Direct call: failed
- **Line 452:** invalid - Direct call: invalid
- **Line 460:** failed - Direct call: failed
- **Line 500:** failed - Direct call: failed
- **Line 508:** failed - Direct call: failed
- **Line 538:** invalid - Direct call: invalid
- **Line 544:** failed - Direct call: failed
- **Line 571:** failed - Direct call: failed
- **Line 581:** failed - Direct call: failed
- **Line 591:** failed - Direct call: failed
- **Line 601:** failed - Direct call: failed
- **Line 618:** failed - Direct call: failed
- **Line 640:** failed - Direct call: failed
- **Line 689:** failed - Direct call: failed
- **Line 532:** missing - Direct call: missing

---

## src/training/unified_data_orchestrator.py

**Undefined function calls: 70**

- **Line 142:** validates - Direct call: validates
- **Line 149:** secure_data_processing - Direct call: secure_data_processing
- **Line 155:** log_execution_time - Direct call: log_execution_time
- **Line 162:** log_call - Direct call: log_call
- **Line 168:** circuit_breaker - Direct call: circuit_breaker
- **Line 174:** validates - Direct call: validates
- **Line 179:** quality_gate - Direct call: quality_gate
- **Line 183:** handles_errors - Direct call: handles_errors
- **Line 220:** validates - Direct call: validates
- **Line 224:** secure_data_processing - Direct call: secure_data_processing
- **Line 230:** log_execution_time - Direct call: log_execution_time
- **Line 236:** log_call - Direct call: log_call
- **Line 242:** circuit_breaker - Direct call: circuit_breaker
- **Line 248:** validates - Direct call: validates
- **Line 253:** quality_gate - Direct call: quality_gate
- **Line 257:** handles_errors - Direct call: handles_errors
- **Line 290:** validates - Direct call: validates
- **Line 301:** secure_data_processing - Direct call: secure_data_processing
- **Line 307:** prevent_data_leakage - Direct call: prevent_data_leakage
- **Line 313:** log_execution_time - Direct call: log_execution_time
- **Line 320:** cached - Direct call: cached
- **Line 326:** log_call - Direct call: log_call
- **Line 332:** circuit_breaker - Direct call: circuit_breaker
- **Line 338:** validates - Direct call: validates
- **Line 347:** quality_gate - Direct call: quality_gate
- **Line 351:** handles_errors - Direct call: handles_errors
- **Line 513:** validates - Direct call: validates
- **Line 524:** secure_data_processing - Direct call: secure_data_processing
- **Line 530:** prevent_data_leakage - Direct call: prevent_data_leakage
- **Line 536:** log_execution_time - Direct call: log_execution_time
- **Line 543:** cached - Direct call: cached
- **Line 549:** log_call - Direct call: log_call
- **Line 555:** circuit_breaker - Direct call: circuit_breaker
- **Line 561:** validates - Direct call: validates
- **Line 570:** quality_gate - Direct call: quality_gate
- **Line 577:** handles_errors - Direct call: handles_errors
- **Line 740:** validates - Direct call: validates
- **Line 748:** secure_data_processing - Direct call: secure_data_processing
- **Line 754:** prevent_data_leakage - Direct call: prevent_data_leakage
- **Line 755:** log_execution_time - Direct call: log_execution_time
- **Line 761:** cached - Direct call: cached
- **Line 767:** log_call - Direct call: log_call
- **Line 773:** circuit_breaker - Direct call: circuit_breaker
- **Line 779:** validates - Direct call: validates
- **Line 788:** quality_gate - Direct call: quality_gate
- **Line 792:** handles_errors - Direct call: handles_errors
- **Line 989:** validates - Direct call: validates
- **Line 994:** secure_data_processing - Direct call: secure_data_processing
- **Line 1000:** prevent_data_leakage - Direct call: prevent_data_leakage
- **Line 1001:** log_execution_time - Direct call: log_execution_time
- **Line 1007:** cached - Direct call: cached
- **Line 1013:** log_call - Direct call: log_call
- **Line 1019:** circuit_breaker - Direct call: circuit_breaker
- **Line 1025:** validates - Direct call: validates
- **Line 1030:** quality_gate - Direct call: quality_gate
- **Line 1034:** handles_errors - Direct call: handles_errors
- **Line 1208:** validates - Direct call: validates
- **Line 1216:** secure_data_processing - Direct call: secure_data_processing
- **Line 1222:** prevent_data_leakage - Direct call: prevent_data_leakage
- **Line 1223:** log_execution_time - Direct call: log_execution_time
- **Line 1230:** cached - Direct call: cached
- **Line 1236:** log_call - Direct call: log_call
- **Line 1242:** circuit_breaker - Direct call: circuit_breaker
- **Line 1248:** validates - Direct call: validates
- **Line 1257:** quality_gate - Direct call: quality_gate
- **Line 1261:** handles_errors - Direct call: handles_errors
- **Line 73:** UnifiedDataLoader - Direct call: UnifiedDataLoader
- **Line 75:** DataSharingManager - Direct call: DataSharingManager
- **Line 1366:** Path - Direct call: Path
- **Line 1460:** hash - Direct call: hash

---

## src/training/utils/feature_calculators.py

**Undefined function calls: 8**

- **Line 270:** calculator - Direct call: calculator
- **Line 272:** calculator - Direct call: calculator
- **Line 274:** calculator - Direct call: calculator
- **Line 276:** calculator - Direct call: calculator
- **Line 278:** calculator - Direct call: calculator
- **Line 280:** calculator - Direct call: calculator
- **Line 282:** calculator - Direct call: calculator
- **Line 284:** calculator - Direct call: calculator

---

## src/training/validator.py

**Undefined function calls: 3**

- **Line 57:** can_proceed_to_step - Direct call: can_proceed_to_step
- **Line 123:** get_validation_config - Direct call: get_validation_config
- **Line 79:** get_progression_rules - Direct call: get_progression_rules

---

## src/training/vectorized_training_pipeline.py

**Undefined function calls: 9**

- **Line 66:** handles_errors - Direct call: handles_errors
- **Line 89:** handles_errors - Direct call: handles_errors
- **Line 160:** handles_errors - Direct call: handles_errors
- **Line 182:** handles_errors - Direct call: handles_errors
- **Line 213:** handles_errors - Direct call: handles_errors
- **Line 243:** handles_errors - Direct call: handles_errors
- **Line 277:** handles_errors - Direct call: handles_errors
- **Line 56:** MatrixEnhancementManager - Direct call: MatrixEnhancementManager
- **Line 60:** VectorizedAdvancedFeatureEngineering - Direct call: VectorizedAdvancedFeatureEngineering

---

## src/training/wavelet_caching_workflow.py

**Undefined function calls: 18**

- **Line 26:** handles_errors - Direct call: handles_errors
- **Line 39:** handles_errors - Direct call: handles_errors
- **Line 82:** handles_errors - Direct call: handles_errors
- **Line 92:** WaveletFeaturePrecomputer - Direct call: WaveletFeaturePrecomputer
- **Line 103:** Path - Direct call: Path
- **Line 136:** BacktestingWithCachedFeatures - Direct call: BacktestingWithCachedFeatures
- **Line 201:** BacktestingWithCachedFeatures - Direct call: BacktestingWithCachedFeatures
- **Line 213:** BacktestingWithCachedFeatures - Direct call: BacktestingWithCachedFeatures
- **Line 246:** WaveletFeatureCache - Direct call: WaveletFeatureCache
- **Line 322:** main - Direct call: main
- **Line 96:** create_sample_data - Direct call: create_sample_data
- **Line 292:** step1_precompute_features - Direct call: step1_precompute_features
- **Line 298:** step2_run_backtests - Direct call: step2_run_backtests
- **Line 304:** step3_performance_comparison - Direct call: step3_performance_comparison
- **Line 310:** step4_cache_management - Direct call: step4_cache_management
- **Line 289:** load_config - Direct call: load_config
- **Line 142:** ohlcv_columns - Direct call: ohlcv_columns
- **Line 266:** Path - Direct call: Path

---

## src/training/wavelet_feature_selection_workflow.py

**Undefined function calls: 33**

- **Line 1107:** TreeExplainer - Direct call: TreeExplainer
- **Line 135:** handles_errors - Direct call: handles_errors
- **Line 176:** handles_errors - Direct call: handles_errors
- **Line 231:** handles_errors - Direct call: handles_errors
- **Line 357:** handles_errors - Direct call: handles_errors
- **Line 477:** handles_errors - Direct call: handles_errors
- **Line 541:** handles_errors - Direct call: handles_errors
- **Line 603:** handles_errors - Direct call: handles_errors
- **Line 713:** handles_errors - Direct call: handles_errors
- **Line 75:** Path - Direct call: Path
- **Line 152:** VectorizedAdvancedFeatureEngineering - Direct call: VectorizedAdvancedFeatureEngineering
- **Line 159:** WaveletFeaturePrecomputer - Direct call: WaveletFeaturePrecomputer
- **Line 308:** cross_val_score - Direct call: cross_val_score
- **Line 387:** permutation_importance - Direct call: permutation_importance
- **Line 659:** cross_val_score - Direct call: cross_val_score
- **Line 278:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 320:** classification_report - Direct call: classification_report
- **Line 633:** GradientBoostingClassifier - Direct call: GradientBoostingClassifier
- **Line 671:** classification_report - Direct call: classification_report
- **Line 155:** failed - Direct call: failed
- **Line 162:** failed - Direct call: failed
- **Line 173:** initialization_error - Direct call: initialization_error
- **Line 228:** error - Direct call: error
- **Line 293:** GradientBoostingClassifier - Direct call: GradientBoostingClassifier
- **Line 354:** error - Direct call: error
- **Line 445:** error - Direct call: error
- **Line 493:** error - Direct call: error
- **Line 538:** error - Direct call: error
- **Line 600:** error - Direct call: error
- **Line 644:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 710:** error - Direct call: error
- **Line 789:** error - Direct call: error
- **Line 1018:** error - Direct call: error

---

## src/transition/baseline_rf.py

**Undefined function calls: 2**

- **Line 90:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 102:** classification_report - Direct call: classification_report

---

## src/transition/event_trigger_indexer.py

**Undefined function calls: 2**

- **Line 55:** TrainingManager - Direct call: TrainingManager
- **Line 154:** CompositeHMMRegimeSystem - Direct call: CompositeHMMRegimeSystem

---

## src/transition/event_window_dataset.py

**Undefined function calls: 2**

- **Line 58:** StateSequenceBuilder - Direct call: StateSequenceBuilder
- **Line 125:** hash - Direct call: hash

---

## src/transition/multitask_rf.py

**Undefined function calls: 12**

- **Line 110:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 122:** classification_report - Direct call: classification_report
- **Line 161:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 268:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 84:** f1_score - Direct call: f1_score
- **Line 172:** classification_report - Direct call: classification_report
- **Line 219:** RandomForestClassifier - Direct call: RandomForestClassifier
- **Line 279:** classification_report - Direct call: classification_report
- **Line 313:** RandomForestRegressor - Direct call: RandomForestRegressor
- **Line 229:** classification_report - Direct call: classification_report
- **Line 323:** mean_absolute_error - Direct call: mean_absolute_error
- **Line 205:** Counter - Direct call: Counter

---

## src/transition/rolling_window_dataset.py

**Undefined function calls: 2**

- **Line 67:** StateSequenceBuilder - Direct call: StateSequenceBuilder
- **Line 72:** PathTargetEngineer - Direct call: PathTargetEngineer

---

## src/transition/seq2seq_trainer.py

**Undefined function calls: 7**

- **Line 180:** DataLoader - Direct call: DataLoader
- **Line 181:** DataLoader - Direct call: DataLoader
- **Line 106:** self - Direct call: self
- **Line 130:** self - Direct call: self
- **Line 199:** model - Direct call: model
- **Line 170:** block - Direct call: block
- **Line 242:** ModelCheckpoint - Direct call: ModelCheckpoint

---

## src/transition/state_sequence_builder.py

**Undefined function calls: 3**

- **Line 49:** UnifiedRegimeClassifier - Direct call: UnifiedRegimeClassifier
- **Line 87:** hash - Direct call: hash
- **Line 120:** StandardScaler - Direct call: StandardScaler

---

## src/utils/async_utils.py

**Undefined function calls: 30**

- **Line 46:** handles_errors - Direct call: handles_errors
- **Line 75:** handles_errors - Direct call: handles_errors
- **Line 92:** handles_errors - Direct call: handles_errors
- **Line 113:** handles_errors - Direct call: handles_errors
- **Line 147:** handles_errors - Direct call: handles_errors
- **Line 180:** handles_errors - Direct call: handles_errors
- **Line 203:** handles_errors - Direct call: handles_errors
- **Line 227:** handles_errors - Direct call: handles_errors
- **Line 246:** handles_errors - Direct call: handles_errors
- **Line 267:** handles_errors - Direct call: handles_errors
- **Line 302:** handles_errors - Direct call: handles_errors
- **Line 331:** handles_errors - Direct call: handles_errors
- **Line 346:** handles_errors - Direct call: handles_errors
- **Line 367:** handles_errors - Direct call: handles_errors
- **Line 417:** handles_errors - Direct call: handles_errors
- **Line 442:** handles_errors - Direct call: handles_errors
- **Line 474:** handles_errors - Direct call: handles_errors
- **Line 69:** invalid - Direct call: invalid
- **Line 102:** invalid - Direct call: invalid
- **Line 107:** invalid - Direct call: invalid
- **Line 325:** invalid - Direct call: invalid
- **Line 356:** invalid - Direct call: invalid
- **Line 361:** invalid - Direct call: invalid
- **Line 429:** missing - Direct call: missing
- **Line 553:** warning - Direct call: warning
- **Line 575:** missing - Direct call: missing
- **Line 406:** failed - Direct call: failed
- **Line 410:** failed - Direct call: failed
- **Line 565:** failed - Direct call: failed
- **Line 586:** failed - Direct call: failed

---

## src/utils/caching.py

**Undefined function calls: 1**

- **Line 8:** func - Direct call: func

---

## src/utils/common_operations.py

**Undefined function calls: 15**

- **Line 1226:** Path - Direct call: Path
- **Line 1274:** Path - Direct call: Path
- **Line 1340:** defaultdict - Direct call: defaultdict
- **Line 1344:** Counter - Direct call: Counter
- **Line 1348:** deque - Direct call: deque
- **Line 479:** Path - Direct call: Path
- **Line 512:** Path - Direct call: Path
- **Line 542:** Path - Direct call: Path
- **Line 576:** Path - Direct call: Path
- **Line 1260:** deepcopy - Direct call: deepcopy
- **Line 1388:** ThreadPoolExecutor - Direct call: ThreadPoolExecutor
- **Line 1064:** func - Direct call: func
- **Line 1268:** Path - Direct call: Path
- **Line 1360:** func - Direct call: func
- **Line 80:** Path - Direct call: Path

---

## src/utils/compat.py

**Undefined function calls: 5**

- **Line 24:** _handles_errors - Direct call: _handles_errors
- **Line 59:** wraps - Direct call: wraps
- **Line 44:** wraps - Direct call: wraps
- **Line 62:** func - Direct call: func
- **Line 47:** func - Direct call: func

---

## src/utils/comprehensive_logger.py

**Undefined function calls: 1**

- **Line 20:** Path - Direct call: Path

---

## src/utils/config_loader.py

**Undefined function calls: 3**

- **Line 29:** missing - Direct call: missing
- **Line 95:** error - Direct call: error
- **Line 37:** error - Direct call: error

---

## src/utils/configuration_security.py

**Undefined function calls: 8**

- **Line 40:** handles_errors - Direct call: handles_errors
- **Line 242:** handles_errors - Direct call: handles_errors
- **Line 270:** handles_errors - Direct call: handles_errors
- **Line 37:** Path - Direct call: Path
- **Line 51:** Path - Direct call: Path
- **Line 313:** Path - Direct call: Path
- **Line 264:** Path - Direct call: Path
- **Line 294:** Path - Direct call: Path

---

## src/utils/cross_step_validation.py

**Undefined function calls: 14**

- **Line 34:** ValidationResult - Direct call: ValidationResult
- **Line 150:** ValidationResult - Direct call: ValidationResult
- **Line 38:** ValidationIssue - Direct call: ValidationIssue
- **Line 45:** ValidationIssue - Direct call: ValidationIssue
- **Line 55:** ValidationIssue - Direct call: ValidationIssue
- **Line 58:** ValidationIssue - Direct call: ValidationIssue
- **Line 88:** ValidationIssue - Direct call: ValidationIssue
- **Line 102:** ValidationIssue - Direct call: ValidationIssue
- **Line 106:** ValidationIssue - Direct call: ValidationIssue
- **Line 155:** ValidationIssue - Direct call: ValidationIssue
- **Line 66:** ValidationIssue - Direct call: ValidationIssue
- **Line 68:** ValidationIssue - Direct call: ValidationIssue
- **Line 71:** ValidationIssue - Direct call: ValidationIssue
- **Line 83:** ValidationIssue - Direct call: ValidationIssue

---

## src/utils/cross_step_validator.py

**Undefined function calls: 8**

- **Line 22:** field - Direct call: field
- **Line 23:** field - Direct call: field
- **Line 24:** field - Direct call: field
- **Line 25:** field - Direct call: field
- **Line 27:** field - Direct call: field
- **Line 36:** field - Direct call: field
- **Line 46:** PipelineStandards - Direct call: PipelineStandards
- **Line 162:** check_func - Direct call: check_func

---

## src/utils/data_loader.py

**Undefined function calls: 5**

- **Line 32:** with_tracing_span - Direct call: with_tracing_span
- **Line 236:** lru_cache - Direct call: lru_cache
- **Line 56:** hash - Direct call: hash
- **Line 56:** hash - Direct call: hash
- **Line 131:** Path - Direct call: Path

---

## src/utils/data_preprocessing.py

**Undefined function calls: 1**

- **Line 73:** timedelta - Direct call: timedelta

---

## src/utils/data_streaming_manager.py

**Undefined function calls: 5**

- **Line 39:** PipelineStandards - Direct call: PipelineStandards
- **Line 327:** Path - Direct call: Path
- **Line 225:** processing_func - Direct call: processing_func
- **Line 149:** timedelta - Direct call: timedelta
- **Line 240:** progress_callback - Direct call: progress_callback

---

## src/utils/database_security.py

**Undefined function calls: 4**

- **Line 214:** handles_errors - Direct call: handles_errors
- **Line 82:** id - Direct call: id
- **Line 173:** Path - Direct call: Path
- **Line 194:** MongoClient - Direct call: MongoClient

---

## src/utils/decorator_registry.py

**Undefined function calls: 2**

- **Line 39:** callable - Direct call: callable
- **Line 197:** callable - Direct call: callable

---

## src/utils/decorators.py

**Undefined function calls: 10**

- **Line 27:** func - Direct call: func
- **Line 35:** func - Direct call: func
- **Line 43:** func - Direct call: func
- **Line 51:** func - Direct call: func
- **Line 59:** func - Direct call: func
- **Line 67:** func - Direct call: func
- **Line 75:** func - Direct call: func
- **Line 83:** func - Direct call: func
- **Line 91:** func - Direct call: func
- **Line 99:** func - Direct call: func

---

## src/utils/decorators/__init__.py

**Undefined function calls: 1**

- **Line 8:** Path - Direct call: Path

---

## src/utils/decorators/errors.py

**Undefined function calls: 6**

- **Line 123:** callable - Direct call: callable
- **Line 25:** callable - Direct call: callable
- **Line 28:** default_return - Direct call: default_return
- **Line 113:** f - Direct call: f
- **Line 30:** default_return - Direct call: default_return
- **Line 102:** f - Direct call: f

---

## src/utils/enhanced_config_management.py

**Undefined function calls: 8**

- **Line 132:** field - Direct call: field
- **Line 133:** field - Direct call: field
- **Line 61:** asdict - Direct call: asdict
- **Line 66:** cls - Direct call: cls
- **Line 122:** asdict - Direct call: asdict
- **Line 127:** cls - Direct call: cls
- **Line 164:** cls - Direct call: cls
- **Line 170:** Path - Direct call: Path

---

## src/utils/enhanced_data_quality_validator.py

**Undefined function calls: 3**

- **Line 33:** field - Direct call: field
- **Line 34:** field - Direct call: field
- **Line 35:** field - Direct call: field

---

## src/utils/enhanced_error_handler.py

**Undefined function calls: 3**

- **Line 43:** field - Direct call: field
- **Line 212:** func - Direct call: func
- **Line 198:** func - Direct call: func

---

## src/utils/enhanced_memory_management.py

**Undefined function calls: 3**

- **Line 194:** processor_func - Direct call: processor_func
- **Line 218:** processor_func - Direct call: processor_func
- **Line 128:** func - Direct call: func

---

## src/utils/enhanced_missing_value_handler.py

**Undefined function calls: 2**

- **Line 54:** handles_errors - Direct call: handles_errors
- **Line 233:** DataDownloader - Direct call: DataDownloader

---

## src/utils/enhanced_mlflow_integration.py

**Undefined function calls: 38**

- **Line 1109:** handles_errors - Direct call: handles_errors
- **Line 1150:** handles_errors - Direct call: handles_errors
- **Line 1193:** handles_errors - Direct call: handles_errors
- **Line 467:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 517:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 59:** wraps - Direct call: wraps
- **Line 222:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 235:** log_artifacts_with_metadata - Direct call: log_artifacts_with_metadata
- **Line 326:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 361:** log_artifacts_with_metadata - Direct call: log_artifacts_with_metadata
- **Line 558:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 598:** log_artifacts_with_metadata - Direct call: log_artifacts_with_metadata
- **Line 639:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 651:** log_model_with_metadata - Direct call: log_model_with_metadata
- **Line 687:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 698:** log_metrics_with_metadata - Direct call: log_metrics_with_metadata
- **Line 724:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 1085:** validate_run_metadata - Direct call: validate_run_metadata
- **Line 1127:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 1130:** log_enhanced_training_metadata - Direct call: log_enhanced_training_metadata
- **Line 1170:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 1173:** log_metrics_with_metadata - Direct call: log_metrics_with_metadata
- **Line 1209:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 1212:** log_enhanced_training_metadata - Direct call: log_enhanced_training_metadata
- **Line 1338:** get_report_manager - Direct call: get_report_manager
- **Line 69:** extract_training_metadata - Direct call: extract_training_metadata
- **Line 823:** log_model_with_metadata - Direct call: log_model_with_metadata
- **Line 875:** log_metrics_with_metadata - Direct call: log_metrics_with_metadata
- **Line 918:** log_params_with_metadata - Direct call: log_params_with_metadata
- **Line 965:** log_artifacts_with_metadata - Direct call: log_artifacts_with_metadata
- **Line 1099:** get_enhanced_run_metadata - Direct call: get_enhanced_run_metadata
- **Line 772:** log_enhanced_training_metadata - Direct call: log_enhanced_training_metadata
- **Line 96:** log_enhanced_training_metadata - Direct call: log_enhanced_training_metadata
- **Line 119:** log_params_with_metadata - Direct call: log_params_with_metadata
- **Line 149:** log_enhanced_training_metadata - Direct call: log_enhanced_training_metadata
- **Line 133:** func - Direct call: func
- **Line 192:** func - Direct call: func
- **Line 169:** log_metrics_with_metadata - Direct call: log_metrics_with_metadata

---

## src/utils/enhanced_outlier_handler.py

**Undefined function calls: 3**

- **Line 139:** handles_errors - Direct call: handles_errors
- **Line 245:** IsolationForest - Direct call: IsolationForest
- **Line 275:** LocalOutlierFactor - Direct call: LocalOutlierFactor

---

## src/utils/enhanced_step_wrapper.py

**Undefined function calls: 5**

- **Line 44:** PipelineStandards - Direct call: PipelineStandards
- **Line 45:** DataStreamingManager - Direct call: DataStreamingManager
- **Line 46:** CrossStepValidator - Direct call: CrossStepValidator
- **Line 47:** AdvancedQualityMetrics - Direct call: AdvancedQualityMetrics
- **Line 378:** __import__ - Direct call: __import__

---

## src/utils/error_handler.py

**Undefined function calls: 1**

- **Line 8:** func - Direct call: func

---

## src/utils/fallback_monitoring.py

**Undefined function calls: 11**

- **Line 57:** field - Direct call: field
- **Line 59:** field - Direct call: field
- **Line 60:** field - Direct call: field
- **Line 61:** field - Direct call: field
- **Line 62:** field - Direct call: field
- **Line 63:** field - Direct call: field
- **Line 172:** func - Direct call: func
- **Line 202:** func - Direct call: func
- **Line 228:** func - Direct call: func
- **Line 156:** func - Direct call: func
- **Line 217:** func - Direct call: func

---

## src/utils/feature_engineering_validation.py

**Undefined function calls: 12**

- **Line 35:** ValidationResult - Direct call: ValidationResult
- **Line 38:** ValidationIssue - Direct call: ValidationIssue
- **Line 46:** ValidationIssue - Direct call: ValidationIssue
- **Line 61:** ValidationIssue - Direct call: ValidationIssue
- **Line 64:** ValidationIssue - Direct call: ValidationIssue
- **Line 76:** ValidationIssue - Direct call: ValidationIssue
- **Line 81:** ValidationIssue - Direct call: ValidationIssue
- **Line 107:** ValidationIssue - Direct call: ValidationIssue
- **Line 136:** calc_func - Direct call: calc_func
- **Line 51:** ValidationIssue - Direct call: ValidationIssue
- **Line 57:** ValidationIssue - Direct call: ValidationIssue
- **Line 102:** ValidationIssue - Direct call: ValidationIssue

---

## src/utils/feature_output_validator.py

**Undefined function calls: 1**

- **Line 200:** critical - Direct call: critical

---

## src/utils/function_call_monitor.py

**Undefined function calls: 8**

- **Line 77:** field - Direct call: field
- **Line 79:** field - Direct call: field
- **Line 80:** field - Direct call: field
- **Line 81:** field - Direct call: field
- **Line 82:** field - Direct call: field
- **Line 83:** field - Direct call: field
- **Line 467:** func - Direct call: func
- **Line 460:** func - Direct call: func

---

## src/utils/function_validation_framework.py

**Undefined function calls: 7**

- **Line 61:** field - Direct call: field
- **Line 62:** field - Direct call: field
- **Line 63:** field - Direct call: field
- **Line 64:** field - Direct call: field
- **Line 67:** field - Direct call: field
- **Line 549:** func - Direct call: func
- **Line 560:** func - Direct call: func

---

## src/utils/hmm_composite_manager.py

**Undefined function calls: 8**

- **Line 113:** handles_errors - Direct call: handles_errors
- **Line 164:** handles_errors - Direct call: handles_errors
- **Line 230:** handles_errors - Direct call: handles_errors
- **Line 280:** handles_errors - Direct call: handles_errors
- **Line 331:** handles_errors - Direct call: handles_errors
- **Line 383:** handles_errors - Direct call: handles_errors
- **Line 450:** handles_errors - Direct call: handles_errors
- **Line 431:** run_step3 - Direct call: run_step3

---

## src/utils/intelligent_feature_cache.py

**Undefined function calls: 8**

- **Line 32:** Path - Direct call: Path
- **Line 337:** wraps - Direct call: wraps
- **Line 87:** callable - Direct call: callable
- **Line 321:** wraps - Direct call: wraps
- **Line 347:** func - Direct call: func
- **Line 331:** func - Direct call: func
- **Line 66:** hash - Direct call: hash
- **Line 66:** hash - Direct call: hash

---

## src/utils/logger.py

**Undefined function calls: 9**

- **Line 521:** get_comprehensive_logger - Direct call: get_comprehensive_logger
- **Line 578:** get_comprehensive_logger - Direct call: get_comprehensive_logger
- **Line 442:** Path - Direct call: Path
- **Line 222:** get_json_formatter - Direct call: get_json_formatter
- **Line 260:** RotatingFileHandler - Direct call: RotatingFileHandler
- **Line 264:** CorrelationIdFilter - Direct call: CorrelationIdFilter
- **Line 552:** get_comprehensive_logger - Direct call: get_comprehensive_logger
- **Line 363:** info - Direct call: info
- **Line 719:** details_provider - Direct call: details_provider

---

## src/utils/lookahead_bias_detector.py

**Undefined function calls: 2**

- **Line 41:** handles_errors - Direct call: handles_errors
- **Line 593:** handles_errors - Direct call: handles_errors

---

## src/utils/mlflow_utils.py

**Undefined function calls: 2**

- **Line 58:** wraps - Direct call: wraps
- **Line 63:** func - Direct call: func

---

## src/utils/observability.py

**Undefined function calls: 8**

- **Line 87:** LoggerProvider - Direct call: LoggerProvider
- **Line 88:** OTLPLogExporter - Direct call: OTLPLogExporter
- **Line 89:** BatchLogRecordProcessor - Direct call: BatchLogRecordProcessor
- **Line 44:** LoggingIntegration - Direct call: LoggingIntegration
- **Line 68:** failed - Direct call: failed
- **Line 93:** failed - Direct call: failed
- **Line 51:** AioHttpIntegration - Direct call: AioHttpIntegration
- **Line 54:** FastApiIntegration - Direct call: FastApiIntegration

---

## src/utils/parallel_processing_optimizer.py

**Undefined function calls: 8**

- **Line 122:** partial - Direct call: partial
- **Line 221:** wraps - Direct call: wraps
- **Line 120:** func - Direct call: func
- **Line 129:** executor_cls - Direct call: executor_cls
- **Line 131:** as_completed - Direct call: as_completed
- **Line 179:** partial - Direct call: partial
- **Line 236:** func - Direct call: func
- **Line 239:** func - Direct call: func

---

## src/utils/parquet_utils.py

**Undefined function calls: 4**

- **Line 20:** handles_errors - Direct call: handles_errors
- **Line 75:** handles_errors - Direct call: handles_errors
- **Line 76:** handles_errors - Direct call: handles_errors
- **Line 121:** handles_errors - Direct call: handles_errors

---

## src/utils/performance.py

**Undefined function calls: 1**

- **Line 7:** func - Direct call: func

---

## src/utils/pipeline_enhancement_integration.py

**Undefined function calls: 1**

- **Line 241:** enhanced_step_class - Direct call: enhanced_step_class

---

## src/utils/pipeline_standards.py

**Undefined function calls: 7**

- **Line 45:** field - Direct call: field
- **Line 46:** field - Direct call: field
- **Line 47:** field - Direct call: field
- **Line 49:** field - Direct call: field
- **Line 13:** Path - Direct call: Path
- **Line 75:** __import__ - Direct call: __import__
- **Line 97:** __import__ - Direct call: __import__

---

## src/utils/prometheus_metrics.py

**Undefined function calls: 15**

- **Line 44:** Histogram - Direct call: Histogram
- **Line 45:** Counter - Direct call: Counter
- **Line 46:** Counter - Direct call: Counter
- **Line 47:** Gauge - Direct call: Gauge
- **Line 48:** Gauge - Direct call: Gauge
- **Line 49:** Gauge - Direct call: Gauge
- **Line 50:** Gauge - Direct call: Gauge
- **Line 51:** Gauge - Direct call: Gauge
- **Line 52:** Gauge - Direct call: Gauge
- **Line 53:** Gauge - Direct call: Gauge
- **Line 54:** Counter - Direct call: Counter
- **Line 55:** Counter - Direct call: Counter
- **Line 134:** generate_latest - Direct call: generate_latest
- **Line 73:** start_http_server - Direct call: start_http_server
- **Line 77:** failed - Direct call: failed

---

## src/utils/quality_alert_system.py

**Undefined function calls: 16**

- **Line 231:** AlertConfig - Direct call: AlertConfig
- **Line 156:** defaultdict - Direct call: defaultdict
- **Line 15:** Path - Direct call: Path
- **Line 141:** timedelta - Direct call: timedelta
- **Line 176:** Alert - Direct call: Alert
- **Line 49:** Alert - Direct call: Alert
- **Line 51:** Alert - Direct call: Alert
- **Line 53:** Alert - Direct call: Alert
- **Line 55:** Alert - Direct call: Alert
- **Line 57:** Alert - Direct call: Alert
- **Line 59:** Alert - Direct call: Alert
- **Line 112:** MIMEMultipart - Direct call: MIMEMultipart
- **Line 41:** Alert - Direct call: Alert
- **Line 47:** Alert - Direct call: Alert
- **Line 116:** MIMEText - Direct call: MIMEText
- **Line 43:** Alert - Direct call: Alert

---

## src/utils/regime_data_access.py

**Undefined function calls: 1**

- **Line 68:** get_hmm_composite_manager - Direct call: get_hmm_composite_manager

---

## src/utils/report_collector.py

**Undefined function calls: 13**

- **Line 31:** Path - Direct call: Path
- **Line 43:** format_datetime - Direct call: format_datetime
- **Line 43:** get_current_datetime - Direct call: get_current_datetime
- **Line 133:** __import__ - Direct call: __import__
- **Line 202:** Path - Direct call: Path
- **Line 143:** original_func - Direct call: original_func
- **Line 363:** __import__ - Direct call: __import__
- **Line 113:** format_datetime - Direct call: format_datetime
- **Line 233:** format_datetime - Direct call: format_datetime
- **Line 113:** get_current_datetime - Direct call: get_current_datetime
- **Line 233:** get_current_datetime - Direct call: get_current_datetime
- **Line 277:** format_datetime - Direct call: format_datetime
- **Line 277:** get_current_datetime - Direct call: get_current_datetime

---

## src/utils/security_framework.py

**Undefined function calls: 6**

- **Line 427:** handles_errors - Direct call: handles_errors
- **Line 43:** Path - Direct call: Path
- **Line 58:** Fernet - Direct call: Fernet
- **Line 183:** Fernet - Direct call: Fernet
- **Line 367:** Path - Direct call: Path
- **Line 304:** timedelta - Direct call: timedelta

---

## src/utils/signal_handler.py

**Undefined function calls: 41**

- **Line 287:** handles_errors - Direct call: handles_errors
- **Line 61:** handles_errors - Direct call: handles_errors
- **Line 76:** handles_errors - Direct call: handles_errors
- **Line 94:** handles_errors - Direct call: handles_errors
- **Line 111:** handles_errors - Direct call: handles_errors
- **Line 127:** handles_errors - Direct call: handles_errors
- **Line 142:** handles_errors - Direct call: handles_errors
- **Line 157:** handles_errors - Direct call: handles_errors
- **Line 189:** handles_errors - Direct call: handles_errors
- **Line 207:** handles_errors - Direct call: handles_errors
- **Line 228:** handles_errors - Direct call: handles_errors
- **Line 245:** handles_errors - Direct call: handles_errors
- **Line 271:** handles_errors - Direct call: handles_errors
- **Line 170:** load_configuration - Direct call: load_configuration
- **Line 306:** failed - Direct call: failed
- **Line 137:** warning - Direct call: warning
- **Line 152:** warning - Direct call: warning
- **Line 202:** initialization_error - Direct call: initialization_error
- **Line 309:** failed - Direct call: failed
- **Line 51:** invalid - Direct call: invalid
- **Line 58:** failed - Direct call: failed
- **Line 74:** error - Direct call: error
- **Line 86:** invalid - Direct call: invalid
- **Line 91:** error - Direct call: error
- **Line 109:** error - Direct call: error
- **Line 125:** error - Direct call: error
- **Line 140:** error - Direct call: error
- **Line 155:** error - Direct call: error
- **Line 177:** failed - Direct call: failed
- **Line 179:** error - Direct call: error
- **Line 187:** error - Direct call: error
- **Line 205:** initialization_error - Direct call: initialization_error
- **Line 226:** error - Direct call: error
- **Line 241:** warning - Direct call: warning
- **Line 243:** error - Direct call: error
- **Line 258:** missing - Direct call: missing
- **Line 260:** error - Direct call: error
- **Line 284:** error - Direct call: error
- **Line 220:** callback - Direct call: callback
- **Line 218:** callback - Direct call: callback
- **Line 223:** failed - Direct call: failed

---

## src/utils/simple_signal_handler.py

**Undefined function calls: 1**

- **Line 49:** callback - Direct call: callback

---

## src/utils/sr_parameter_loader.py

**Undefined function calls: 2**

- **Line 138:** Path - Direct call: Path
- **Line 36:** Path - Direct call: Path

---

## src/utils/standardized_config_manager.py

**Undefined function calls: 3**

- **Line 14:** Path - Direct call: Path
- **Line 139:** Path - Direct call: Path
- **Line 143:** Path - Direct call: Path

---

## src/utils/standardized_model_manager.py

**Undefined function calls: 9**

- **Line 78:** handles_errors - Direct call: handles_errors
- **Line 125:** handles_errors - Direct call: handles_errors
- **Line 165:** handles_errors - Direct call: handles_errors
- **Line 37:** cls - Direct call: cls
- **Line 51:** Path - Direct call: Path
- **Line 53:** Path - Direct call: Path
- **Line 144:** Path - Direct call: Path
- **Line 238:** Path - Direct call: Path
- **Line 100:** callable - Direct call: callable

---

## src/utils/state_manager.py

**Undefined function calls: 16**

- **Line 52:** handles_errors - Direct call: handles_errors
- **Line 87:** handles_errors - Direct call: handles_errors
- **Line 92:** handles_errors - Direct call: handles_errors
- **Line 117:** handles_errors - Direct call: handles_errors
- **Line 131:** handles_errors - Direct call: handles_errors
- **Line 151:** handles_errors - Direct call: handles_errors
- **Line 173:** handles_errors - Direct call: handles_errors
- **Line 190:** handles_errors - Direct call: handles_errors
- **Line 204:** handles_errors - Direct call: handles_errors
- **Line 213:** handles_errors - Direct call: handles_errors
- **Line 74:** invalid - Direct call: invalid
- **Line 102:** invalid - Direct call: invalid
- **Line 107:** invalid - Direct call: invalid
- **Line 114:** error - Direct call: error
- **Line 121:** Path - Direct call: Path
- **Line 160:** Path - Direct call: Path

---

## src/utils/statistical_distribution_validation.py

**Undefined function calls: 21**

- **Line 38:** ValidationResult - Direct call: ValidationResult
- **Line 124:** jarque_bera - Direct call: jarque_bera
- **Line 131:** anderson - Direct call: anderson
- **Line 198:** kstest - Direct call: kstest
- **Line 204:** ValidationResult - Direct call: ValidationResult
- **Line 121:** shapiro - Direct call: shapiro
- **Line 128:** normaltest - Direct call: normaltest
- **Line 168:** adfuller - Direct call: adfuller
- **Line 186:** acorr_ljungbox - Direct call: acorr_ljungbox
- **Line 41:** ValidationIssue - Direct call: ValidationIssue
- **Line 176:** kpss - Direct call: kpss
- **Line 206:** ValidationIssue - Direct call: ValidationIssue
- **Line 49:** ValidationIssue - Direct call: ValidationIssue
- **Line 53:** ValidationIssue - Direct call: ValidationIssue
- **Line 59:** ValidationIssue - Direct call: ValidationIssue
- **Line 71:** ValidationIssue - Direct call: ValidationIssue
- **Line 80:** ValidationIssue - Direct call: ValidationIssue
- **Line 65:** ValidationIssue - Direct call: ValidationIssue
- **Line 76:** ValidationIssue - Direct call: ValidationIssue
- **Line 84:** ValidationIssue - Direct call: ValidationIssue
- **Line 220:** ValidationIssue - Direct call: ValidationIssue

---

## src/utils/step_validation_initializer.py

**Undefined function calls: 5**

- **Line 19:** PipelineStandards - Direct call: PipelineStandards
- **Line 181:** __import__ - Direct call: __import__
- **Line 193:** original_init - Direct call: original_init
- **Line 195:** PipelineStandards - Direct call: PipelineStandards
- **Line 224:** original_execute - Direct call: original_execute

---

## src/utils/step_validation_updater.py

**Undefined function calls: 5**

- **Line 19:** PipelineStandards - Direct call: PipelineStandards
- **Line 37:** original_init - Direct call: original_init
- **Line 229:** __import__ - Direct call: __import__
- **Line 39:** PipelineStandards - Direct call: PipelineStandards
- **Line 65:** original_execute - Direct call: original_execute

---

## src/utils/step_validation_wrapper.py

**Undefined function calls: 1**

- **Line 19:** PipelineStandards - Direct call: PipelineStandards

---

## src/utils/steps_1_7_compatibility_framework.py

**Undefined function calls: 4**

- **Line 36:** handles_errors - Direct call: handles_errors
- **Line 117:** handles_errors - Direct call: handles_errors
- **Line 161:** handles_errors - Direct call: handles_errors
- **Line 194:** handles_errors - Direct call: handles_errors

---

## src/utils/structured_logging.py

**Undefined function calls: 1**

- **Line 82:** call_next - Direct call: call_next

---

## src/utils/tracing.py

**Undefined function calls: 1**

- **Line 7:** func - Direct call: func

---

## src/utils/validated_step_factory.py

**Undefined function calls: 3**

- **Line 19:** PipelineStandards - Direct call: PipelineStandards
- **Line 293:** __import__ - Direct call: __import__
- **Line 71:** PipelineStandards - Direct call: PipelineStandards

---

## src/utils/validation_decorators.py

**Undefined function calls: 1**

- **Line 7:** func - Direct call: func

---

## src/utils/validator_orchestrator.py

**Undefined function calls: 6**

- **Line 10:** Path - Direct call: Path
- **Line 31:** PrometheusMetrics - Direct call: PrometheusMetrics
- **Line 323:** run_validator_func - Direct call: run_validator_func
- **Line 316:** callable - Direct call: callable
- **Line 318:** missing - Direct call: missing
- **Line 321:** run_validator_func - Direct call: run_validator_func

---

## src/utils/vif_calculator.py

**Undefined function calls: 4**

- **Line 70:** comprehensive_vif_validation - Direct call: comprehensive_vif_validation
- **Line 124:** StandardScaler - Direct call: StandardScaler
- **Line 52:** LinearRegression - Direct call: LinearRegression
- **Line 129:** LedoitWolf - Direct call: LedoitWolf

---

## src/validation/critical_path_validators.py

**Undefined function calls: 28**

- **Line 29:** TypeVar - Direct call: TypeVar
- **Line 203:** wraps - Direct call: wraps
- **Line 214:** wraps - Direct call: wraps
- **Line 227:** wraps - Direct call: wraps
- **Line 243:** wraps - Direct call: wraps
- **Line 256:** wraps - Direct call: wraps
- **Line 205:** func - Direct call: func
- **Line 216:** func - Direct call: func
- **Line 235:** func - Direct call: func
- **Line 245:** func - Direct call: func
- **Line 258:** func - Direct call: func
- **Line 331:** func - Direct call: func
- **Line 247:** validate_market_data - Direct call: validate_market_data
- **Line 260:** validate_model_input - Direct call: validate_model_input
- **Line 47:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 57:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 81:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 91:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 139:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 146:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 156:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 180:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 187:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 282:** get_correlation_id - Direct call: get_correlation_id
- **Line 334:** failed - Direct call: failed
- **Line 337:** error - Direct call: error
- **Line 104:** RuntimeTypeError - Direct call: RuntimeTypeError
- **Line 115:** RuntimeTypeError - Direct call: RuntimeTypeError

---

## src/validation/walk_forward_validator.py

**Undefined function calls: 11**

- **Line 43:** Path - Direct call: Path
- **Line 271:** main - Direct call: main
- **Line 53:** timedelta - Direct call: timedelta
- **Line 65:** timedelta - Direct call: timedelta
- **Line 97:** ProcessPoolExecutor - Direct call: ProcessPoolExecutor
- **Line 118:** model_trainer - Direct call: model_trainer
- **Line 158:** model_trainer - Direct call: model_trainer
- **Line 54:** timedelta - Direct call: timedelta
- **Line 57:** timedelta - Direct call: timedelta
- **Line 80:** timedelta - Direct call: timedelta
- **Line 80:** timedelta - Direct call: timedelta

---

## Summary

- **Total files affected:** 590
- **Total undefined calls:** 7415
