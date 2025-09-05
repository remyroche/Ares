# Functions with Too Many Arguments Report

This report identifies functions that have more than 5 arguments and may need refactoring.

## src/analyst/autoencoder_feature_generator.py

**Functions with too many arguments: 2**

### fit (Line 740)
- **Argument count:** 6
- **Arguments:** self, X_train, y_train, X_val, y_val, trial

### generate_features (Line 1133)
- **Argument count:** 6
- **Arguments:** self, features_df, regime_name, labels, regime_labels, enable_analysis

---

## src/analyst/data_utils.py

**Functions with too many arguments: 2**

### _filter_reasonable_data (Line 1348)
- **Argument count:** 6
- **Arguments:** klines_df, min_price, max_price, close_col, high_col, low_col

### _create_volume_profile (Line 1369)
- **Argument count:** 7
- **Arguments:** klines_df, min_price, max_price, high_col, low_col, volume_col, num_bins

---

## src/analyst/meta_label_relevance.py

**Functions with too many arguments: 2**

### __init__ (Line 162)
- **Argument count:** 6
- **Arguments:** self, artifacts_dir, mi_threshold, sharpe_min_delta, synergy_mi_threshold, max_pairs

### evaluate_from_frame (Line 200)
- **Argument count:** 6
- **Arguments:** self, df, label_names, thresholds, returns_col, risk_free_rate

---

## src/analyst/ml_confidence_predictor.py

**Functions with too many arguments: 2**

### _build_prediction_result (Line 703)
- **Argument count:** 6
- **Arguments:** self, price_target_confidences, adversarial_confidences, directional_analysis, ensemble_predictions, current_price

### compute_mixture_scores (Line 3087)
- **Argument count:** 11
- **Arguments:** self, intensities, confidences, reliability, alpha, beta, gamma, top_k, w_min, w_max, normalize

---

## src/analyst/predictive_ensembles/regime_ensembles/base_ensemble.py

**Functions with too many arguments: 2**

### _tune_hyperparameters (Line 175)
- **Argument count:** 6
- **Arguments:** self, model_class, search_space_func, X, y, n_trials

### _calculate_sr_distances (Line 447)
- **Argument count:** 7
- **Arguments:** self, sr_features, row_idx, current_price, pivot_levels, hvn_levels, current_location

---

## src/core/decorators/enhanced_error_handling.py

**Functions with too many arguments: 3**

### handle_errors_enhanced (Line 565)
- **Argument count:** 8
- **Arguments:** enable_automatic_recovery, enable_error_pattern_detection, enable_performance_impact_analysis, max_recovery_attempts, recovery_timeout_seconds, log_level, generate_error_reports, error_report_path

### __init__ (Line 103)
- **Argument count:** 9
- **Arguments:** self, enable_automatic_recovery, enable_error_pattern_detection, enable_performance_impact_analysis, max_recovery_attempts, recovery_timeout_seconds, log_level, generate_error_reports, error_report_path

### _create_error_context (Line 232)
- **Argument count:** 6
- **Arguments:** self, error, func, args, kwargs, local_vars

---

## src/core/decorators/function_monitor.py

**Functions with too many arguments: 2**

### monitor_function_calls (Line 511)
- **Argument count:** 8
- **Arguments:** enable_performance_monitoring, enable_memory_monitoring, enable_cpu_monitoring, enable_parameter_validation, enable_nested_call_tracking, log_level, generate_detailed_report, report_file_path

### __init__ (Line 82)
- **Argument count:** 9
- **Arguments:** self, enable_performance_monitoring, enable_memory_monitoring, enable_cpu_monitoring, enable_parameter_validation, enable_nested_call_tracking, log_level, generate_detailed_report, report_file_path

---

## src/core/decorators/logging.py

**Functions with too many arguments: 1**

### prepare_log_data (Line 135)
- **Argument count:** 6
- **Arguments:** func, args, kwargs, result, error, duration

---

## src/core/dependency_injection.py

**Functions with too many arguments: 1**

### register (Line 59)
- **Argument count:** 8
- **Arguments:** self, service_name, service_type, implementation, singleton, config, dependencies, lifetime

---

## src/core/domain/decorators.py

**Functions with too many arguments: 3**

### validate_data_quality (Line 23)
- **Argument count:** 14
- **Arguments:** validation_level, required_columns, min_rows, max_null_ratio, check_duplicates, check_timestamps, check_nan, check_infinite, check_constant, check_correlation, max_correlation_threshold, min_unique_values, context, fail_on_issues

### _validate_dataframe (Line 56)
- **Argument count:** 12
- **Arguments:** df, required_columns, min_rows, max_null_ratio, check_duplicates, check_timestamps, check_nan, check_infinite, check_constant, check_correlation, max_correlation_threshold, min_unique_values

### create_step_decorator (Line 279)
- **Argument count:** 6
- **Arguments:** step_name, validate_inputs, monitor_performance, handle_errors, cache_results, timeout_seconds

---

## src/core/reporting/step03_execution_reporter.py

**Functions with too many arguments: 1**

### __init__ (Line 141)
- **Argument count:** 6
- **Arguments:** self, output_directory, enable_html_reports, enable_pdf_reports, enable_csv_exports, log_level

---

## src/core/sr_error_handlers.py

**Functions with too many arguments: 1**

### sr_error_handler (Line 157)
- **Argument count:** 6
- **Arguments:** exceptions, default_return, context, reraise, max_retries, retry_delay

---

## src/launcher/ares_launcher_refactored.py

**Functions with too many arguments: 1**

### _run_unified_training (Line 120)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, training_mode, lookback_days, with_gui

---

## src/launcher/command_handlers.py

**Functions with too many arguments: 2**

### execute (Line 33)
- **Argument count:** 6
- **Arguments:** self, training_mode, symbol, exchange, lookback_days, with_gui

### execute (Line 62)
- **Argument count:** 7
- **Arguments:** self, start_step, symbol, exchange, training_mode, force_rerun, with_gui

---

## src/launcher/configuration_manager.py

**Functions with too many arguments: 1**

### setup_training_environment (Line 179)
- **Argument count:** 6
- **Arguments:** self, training_mode, symbol, exchange, lookback_days, force_rerun

---

## src/monitoring/advanced_tracer.py

**Functions with too many arguments: 1**

### start_span (Line 146)
- **Argument count:** 6
- **Arguments:** self, correlation_id, component_type, operation_name, parent_span_id, metadata

---

## src/monitoring/fractional_system_monitor.py

**Functions with too many arguments: 2**

### track_performance (Line 46)
- **Argument count:** 6
- **Arguments:** self, features, labels, hmm_regime, processing_time, error_occurred

### _calculate_performance_metrics (Line 71)
- **Argument count:** 6
- **Arguments:** self, features, labels, hmm_regime, processing_time, error_occurred

---

## src/strategist/strategist.py

**Functions with too many arguments: 1**

### _apply_regime_adjustments (Line 509)
- **Argument count:** 6
- **Arguments:** self, strategy, regime, regime_confidence, regime_params, regime_metadata

---

## src/supervisor/main.py

**Functions with too many arguments: 1**

### __init__ (Line 22)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange_name, exchange_client, state_manager, db_manager

---

## src/tactician/fully_migrated_tactician.py

**Functions with too many arguments: 1**

### _generate_decision_reasoning (Line 260)
- **Argument count:** 6
- **Arguments:** self, entry_signal, exit_signal, scenario_analysis, model_confidence, analyst_confidence

---

## src/tactician/leverage_sizer.py

**Functions with too many arguments: 1**

### _generate_leverage_reason (Line 502)
- **Argument count:** 7
- **Arguments:** self, final_leverage, ml_leverage, liquidation_leverage, price_target_confidences, adversarial_confidences, combined_confidence

---

## src/tactician/position_sizer.py

**Functions with too many arguments: 2**

### _generate_sizing_reason (Line 475)
- **Argument count:** 7
- **Arguments:** self, final_position_size, kelly_position_size, ml_position_size, price_target_confidences, adversarial_confidences, combined_confidence

### _generate_dual_confidence_sizing_reason (Line 523)
- **Argument count:** 9
- **Arguments:** self, final_position_size, final_confidence, normalized_confidence, analyst_confidence, tactician_confidence, p_avg, b_avg, fractional_kelly_pct

---

## src/tactician/sr_levels/enhanced_sr_validation.py

**Functions with too many arguments: 1**

### _calculate_validation_score (Line 477)
- **Argument count:** 7
- **Arguments:** self, bounce_rate, false_breakout_rate, volume_confirmation_rate, touch_count, failure_count, statistical_significance

---

## src/tactician/sr_levels/sr_computational_optimizer.py

**Functions with too many arguments: 1**

### _vectorized_strength_calculation (Line 467)
- **Argument count:** 6
- **Arguments:** self, prices, volumes, level_prices, level_types, touch_counts

---

## src/tactician/sr_levels/sr_levels_manager.py

**Functions with too many arguments: 1**

### __init__ (Line 17)
- **Argument count:** 14
- **Arguments:** self, price, level_type, method, data_source, timestamp, strength, volume, touch_count, age_hours, bounce_rate, isolation_score, confidence, metadata

---

## src/tactician/sr_levels/sr_ml_enhancer.py

**Functions with too many arguments: 1**

### _calculate_test_strength (Line 1980)
- **Argument count:** 6
- **Arguments:** self, volume_ratio, momentum_strength, test_duration, wick_penetration, step06_penetration_features

---

## src/tactician/sr_levels/sr_parameter_optimizer.py

**Functions with too many arguments: 1**

### _calculate_point_probabilities (Line 280)
- **Argument count:** 6
- **Arguments:** self, market_data, idx, nearest_support, nearest_resistance, params

---

## src/tactician/sr_levels/sr_strength_optimizer.py

**Functions with too many arguments: 3**

### _analyze_sr_level (Line 419)
- **Argument count:** 7
- **Arguments:** self, market_data, level_price, level_type, origin_idx, params, features

### _calculate_enhanced_level_strength (Line 575)
- **Argument count:** 11
- **Arguments:** self, touches, bounces, failures, volumes, wick_touches, body_touches, origin_idx, total_bars, params, features

### _calculate_level_strength (Line 750)
- **Argument count:** 9
- **Arguments:** self, touches, bounces, failures, volumes, origin_idx, total_bars, params, features

---

## src/tactician/sr_levels/sr_weight_optimizer.py

**Functions with too many arguments: 1**

### _normalize_weights (Line 257)
- **Argument count:** 6
- **Arguments:** self, fractal_w, volume_w, pivot_w, atr_w, total_weight

---

## src/tactician/step17_optimized_tactician.py

**Functions with too many arguments: 4**

### _calculate_step17_optimized_position_management (Line 522)
- **Argument count:** 6
- **Arguments:** self, scenario_predictions, trading_decisions, analyst_barriers, analyst_confidence, market_data

### _calculate_step17_optimized_direction (Line 717)
- **Argument count:** 6
- **Arguments:** self, profit_zone_prob, risk_zone_prob, neutral_prob, confidence, dominant_zone

### _calculate_step17_optimized_confidence (Line 762)
- **Argument count:** 6
- **Arguments:** self, scenario_analysis, model_confidence, analyst_confidence, volatility, volume_ratio

### _generate_step17_optimized_reasoning (Line 885)
- **Argument count:** 8
- **Arguments:** self, entry_signal, exit_signal, scenario_analysis, model_confidence, analyst_confidence, volatility, volume_ratio

---

## src/tactician/tactician.py

**Functions with too many arguments: 1**

### _generate_decision_reasoning (Line 913)
- **Argument count:** 6
- **Arguments:** self, entry_signal, exit_signal, scenario_analysis, model_confidence, analyst_confidence

---

## src/tactician/tactics_orchestrator.py

**Functions with too many arguments: 1**

### _aggregate_decisions (Line 244)
- **Argument count:** 7
- **Arguments:** self, sizing_decision, leverage_decision, sr_decision, ml_decision, analyst_confidence, tactician_confidence

---

## src/training/adaptive_optimizer.py

**Functions with too many arguments: 1**

### __init__ (Line 9)
- **Argument count:** 6
- **Arguments:** self, name, volatility, trend_strength, regime_type, optimal_params

---

## src/training/advanced_neural_models.py

**Functions with too many arguments: 7**

### __init__ (Line 20)
- **Argument count:** 6
- **Arguments:** self, input_size, num_channels, kernel_size, dropout, num_classes

### __init__ (Line 46)
- **Argument count:** 8
- **Arguments:** self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout

### __init__ (Line 72)
- **Argument count:** 6
- **Arguments:** self, input_size, num_filters, kernel_sizes, dropout, num_classes

### __init__ (Line 103)
- **Argument count:** 7
- **Arguments:** self, input_size, d_model, nhead, num_layers, dropout, num_classes

### __init__ (Line 147)
- **Argument count:** 7
- **Arguments:** self, input_size, hidden_size, num_layers, dropout, bidirectional, num_classes

### __init__ (Line 172)
- **Argument count:** 7
- **Arguments:** self, input_size, hidden_size, num_layers, dropout, bidirectional, num_classes

### __init__ (Line 197)
- **Argument count:** 8
- **Arguments:** self, model_class, model_params, device, batch_size, epochs, learning_rate, early_stopping_patience

---

## src/training/data_manager.py

**Functions with too many arguments: 1**

### create_unified_database (Line 65)
- **Argument count:** 6
- **Arguments:** self, labeled_data, strategic_signals, train_ratio, validation_ratio, test_ratio

---

## src/training/data_sharing_manager.py

**Functions with too many arguments: 2**

### _generate_cache_key (Line 59)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, timeframe, lookback_days, data_type

### cache_data (Line 345)
- **Argument count:** 7
- **Arguments:** self, symbol, exchange, timeframe, lookback_days, data, data_type

---

## src/training/enhanced_coarse_optimizer.py

**Functions with too many arguments: 1**

### __init__ (Line 40)
- **Argument count:** 9
- **Arguments:** self, db_manager, symbol, timeframe, optimal_target_params, klines_data, agg_trades_data, futures_data, blank_training_mode

---

## src/training/enhanced_dynamic_feature_selection.py

**Functions with too many arguments: 1**

### select_features_dynamically (Line 66)
- **Argument count:** 6
- **Arguments:** self, features_df, target, symbol, exchange, data_dir

---

## src/training/enhanced_matrix_operations.py

**Functions with too many arguments: 1**

### select_features_step2 (Line 984)
- **Argument count:** 8
- **Arguments:** self, features_df, target, symbol, exchange, data_dir, use_autoencoder_features, use_regularization

---

## src/training/enhanced_training_manager_optimized.py

**Functions with too many arguments: 2**

### write_partitioned_dataset (Line 743)
- **Argument count:** 9
- **Arguments:** self, df, base_dir, partition_cols, schema_name, compression, metadata, min_rows_per_group, max_rows_per_file

### materialize_projection (Line 772)
- **Argument count:** 10
- **Arguments:** self, base_dir, filters, columns, output_dir, partition_cols, schema_name, compression, batch_size, metadata

---

## src/training/feature_selection_manager.py

**Functions with too many arguments: 1**

### select_features_step2 (Line 41)
- **Argument count:** 8
- **Arguments:** self, features_df, target, symbol, exchange, data_dir, use_autoencoder_features, use_regularization

---

## src/training/model_probability_generator.py

**Functions with too many arguments: 8**

### generate_model_probabilities (Line 184)
- **Argument count:** 6
- **Arguments:** model, X_test, y_test, market_data, model_type
- **Has **kwargs:** Yes

### generate_price_action_probabilities (Line 25)
- **Argument count:** 7
- **Arguments:** self, model, X_test, y_test, market_data, model_type
- **Has **kwargs:** Yes

### _calculate_triple_barrier_probability (Line 52)
- **Argument count:** 6
- **Arguments:** self, calculator, model, X_test, market_data
- **Has **kwargs:** Yes

### _calculate_direction_probability (Line 65)
- **Argument count:** 6
- **Arguments:** self, calculator, model, X_test, y_test
- **Has **kwargs:** Yes

### _calculate_magnitude_probability (Line 73)
- **Argument count:** 6
- **Arguments:** self, calculator, model, X_test, market_data
- **Has **kwargs:** Yes

### _calculate_barrier_avoidance_probability (Line 82)
- **Argument count:** 6
- **Arguments:** self, calculator, model, X_test, market_data
- **Has **kwargs:** Yes

### generate_ensemble_probabilities (Line 117)
- **Argument count:** 8
- **Arguments:** self, models, model_types, X_test, y_test, market_data, weights
- **Has **kwargs:** Yes

### generate_calibrated_probabilities (Line 159)
- **Argument count:** 8
- **Arguments:** self, model, X_test, y_test, market_data, model_type, calibration_method
- **Has **kwargs:** Yes

---

## src/training/model_saving_utils.py

**Functions with too many arguments: 1**

### generate_and_save_model_probabilities (Line 383)
- **Argument count:** 6
- **Arguments:** model_data, model_path, X_test, y_test, market_data, save_format

---

## src/training/multi_output_model_trainer.py

**Functions with too many arguments: 15**

### __init__ (Line 80)
- **Argument count:** 18
- **Arguments:** self, model_type, direction_target, profit_target, use_profit_features, profit_feature_columns, direction_threshold, profit_scaling, ensemble_method, validation_method, n_splits, test_size, random_state, use_enhanced_feature_selection, supported_model_types, enable_probability_outputs, probability_targets, probability_config

### __init__ (Line 154)
- **Argument count:** 6
- **Arguments:** self, input_size, hidden_sizes, dropout_rate, direction_output_size, profit_output_size

### _train_xgboost_multi_output (Line 789)
- **Argument count:** 8
- **Arguments:** self, X_train, X_val, y_dir_train, y_dir_val, y_prof_train, y_prof_val, feature_names

### _train_catboost_multi_output (Line 876)
- **Argument count:** 8
- **Arguments:** self, X_train, X_val, y_dir_train, y_dir_val, y_prof_train, y_prof_val, feature_names

### _train_lightgbm_multi_output (Line 1108)
- **Argument count:** 8
- **Arguments:** self, X_train, X_val, y_dir_train, y_dir_val, y_prof_train, y_prof_val, feature_names

### _train_randomforest_multi_output (Line 1198)
- **Argument count:** 8
- **Arguments:** self, X_train, X_val, y_dir_train, y_dir_val, y_prof_train, y_prof_val, feature_names

### _train_neural_network_multi_output (Line 1274)
- **Argument count:** 8
- **Arguments:** self, X_train, X_val, y_dir_train, y_dir_val, y_prof_train, y_prof_val, feature_names

### train_with_probability_targets (Line 1658)
- **Argument count:** 7
- **Arguments:** self, X_train, X_val, y_train, y_val, market_data, feature_names

### _train_probability_model_by_type (Line 1725)
- **Argument count:** 7
- **Arguments:** self, X_train, X_val, y_train, y_val, feature_names, prob_type

### _train_lightgbm_probability_model (Line 1769)
- **Argument count:** 7
- **Arguments:** self, X_train, X_val, y_train, y_val, feature_names, prob_type

### _train_randomforest_probability_model (Line 1824)
- **Argument count:** 7
- **Arguments:** self, X_train, X_val, y_train, y_val, feature_names, prob_type

### _train_cnn_probability_model (Line 1866)
- **Argument count:** 7
- **Arguments:** self, X_train, X_val, y_train, y_val, feature_names, prob_type

### _train_tcn_probability_model (Line 1916)
- **Argument count:** 7
- **Arguments:** self, X_train, X_val, y_train, y_val, feature_names, prob_type

### _train_transformer_probability_model (Line 1966)
- **Argument count:** 7
- **Arguments:** self, X_train, X_val, y_train, y_val, feature_names, prob_type

### _train_standard_multi_output (Line 2067)
- **Argument count:** 6
- **Arguments:** self, X_train, X_val, y_train, y_val, feature_names

---

## src/training/optimized_feature_selection_manager.py

**Functions with too many arguments: 1**

### select_features_optimized (Line 102)
- **Argument count:** 6
- **Arguments:** self, features_df, target, model_type, step_name
- **Has **kwargs:** Yes

---

## src/training/probability_calculators.py

**Functions with too many arguments: 2**

### calculate_triple_barrier_probability (Line 28)
- **Argument count:** 7
- **Arguments:** self, model, X_test, market_data, profit_target, stop_loss, volatility_window

### calculate_triple_barrier_probability (Line 146)
- **Argument count:** 6
- **Arguments:** self, model, X_test, market_data, profit_target, stop_loss

---

## src/training/simplified_architecture/config_driven_architecture.py

**Functions with too many arguments: 1**

### add_dependency (Line 180)
- **Argument count:** 6
- **Arguments:** self, name, class_name, module, type
- **Has **kwargs:** Yes

---

## src/training/simplified_architecture/dependency_injection.py

**Functions with too many arguments: 5**

### register_singleton (Line 72)
- **Argument count:** 6
- **Arguments:** self, name, service_type, factory, dependencies, metadata

### register_transient (Line 76)
- **Argument count:** 6
- **Arguments:** self, name, service_type, factory, dependencies, metadata

### register_scoped (Line 80)
- **Argument count:** 6
- **Arguments:** self, name, service_type, factory, dependencies, metadata

### register_factory (Line 84)
- **Argument count:** 6
- **Arguments:** self, name, factory, lifetime, dependencies, metadata

### _register_service (Line 101)
- **Argument count:** 7
- **Arguments:** self, name, service_type, factory, lifetime, dependencies, metadata

---

## src/training/simplified_architecture/modular_components.py

**Functions with too many arguments: 3**

### __init__ (Line 80)
- **Argument count:** 7
- **Arguments:** self, exchange_name, symbols, timeframes, price_range, volume_range
- **Has **kwargs:** Yes

### __init__ (Line 203)
- **Argument count:** 7
- **Arguments:** self, required_columns, column_types, max_null_percentage, max_duplicate_percentage, expected_frequency, max_gaps

### __init__ (Line 690)
- **Argument count:** 6
- **Arguments:** self, data_source, validators, feature_calculators, model_trainer, logger

---

## src/training/steps/backtesting/comprehensive_reporting.py

**Functions with too many arguments: 3**

### generate_step_report (Line 361)
- **Argument count:** 6
- **Arguments:** step_name, step_results, symbol, timeframe, data_dir, output_file

### generate_backtesting_report (Line 1284)
- **Argument count:** 7
- **Arguments:** symbol, exchange, timeframe, data_dir, pipeline_results, logger_data, output_file

### generate_step_report (Line 60)
- **Argument count:** 7
- **Arguments:** self, step_name, step_results, symbol, timeframe, data_dir, output_file

---

## src/training/steps/backtesting/step20_ab_testing_per_regime.py

**Functions with too many arguments: 1**

### _create_ab_testing_context (Line 45)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, timeframe, data_dir, regime_id

---

## src/training/steps/data_collection/data_preparation/step01_5_data_converter.py

**Functions with too many arguments: 3**

### write_partitioned_dataset (Line 420)
- **Argument count:** 13
- **Arguments:** self, df, base_dir, partition_cols, schema_name, compression, use_dictionary, min_rows_per_group, max_rows_per_file, use_threads, update_manifest, metadata, auto_add_date_columns

### scan_dataset (Line 512)
- **Argument count:** 8
- **Arguments:** self, base_dir, filters, columns, batch_size, to_pandas, use_threads, ignore_hidden_temp

### write_flat_parquet (Line 588)
- **Argument count:** 9
- **Arguments:** self, df, file_path, schema_name, compression, use_dictionary, row_group_size, write_statistics, metadata

---

## src/training/steps/data_collection/data_preparation_components/data_cleaner.py

**Functions with too many arguments: 2**

### fill_missing_values (Line 63)
- **Argument count:** 6
- **Arguments:** self, df, method, numeric_fill, string_fill, custom_fills

### clean_time_series (Line 258)
- **Argument count:** 6
- **Arguments:** self, df, timestamp_col, remove_weekends, remove_holidays, ensure_regular_intervals

---

## src/training/steps/data_collection/data_preparation_components/data_format_converter.py

**Functions with too many arguments: 3**

### write_partitioned_dataset (Line 171)
- **Argument count:** 13
- **Arguments:** self, df, base_dir, partition_cols, schema_name, compression, use_dictionary, min_rows_per_group, max_rows_per_file, use_threads, update_manifest, metadata, auto_add_date_columns

### scan_dataset (Line 334)
- **Argument count:** 8
- **Arguments:** self, base_dir, filters, columns, batch_size, to_pandas, use_threads, ignore_hidden_temp

### write_flat_parquet (Line 452)
- **Argument count:** 9
- **Arguments:** self, df, file_path, schema_name, compression, use_dictionary, row_group_size, write_statistics, metadata

---

## src/training/steps/data_collection/data_quality_components/data_downloader.py

**Functions with too many arguments: 2**

### download_missing_data_for_timeframe (Line 47)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, timeframe, start_time, end_time

### download_data_for_timeframe (Line 105)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, timeframe, start_time, end_time

---

## src/training/steps/data_collection/data_quality_components/data_preprocessor.py

**Functions with too many arguments: 2**

### enhanced_preprocess_market_data (Line 115)
- **Argument count:** 7
- **Arguments:** self, data, symbol, exchange, expected_interval_seconds, max_forward_fill_seconds, download_missing_data

### _load_and_filter_downloaded_data (Line 351)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, timeframe, start_time, end_time

---

## src/training/steps/data_collection/raw_data_quality_checker.py

**Functions with too many arguments: 4**

### enhanced_preprocess_market_data (Line 1283)
- **Argument count:** 7
- **Arguments:** data, symbol, exchange, expected_interval_seconds, max_forward_fill_seconds, download_missing_data, config

### enhanced_preprocess_market_data (Line 285)
- **Argument count:** 7
- **Arguments:** self, data, symbol, exchange, expected_interval_seconds, max_forward_fill_seconds, download_missing_data

### _load_and_filter_downloaded_data (Line 422)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, timeframe, start_time, end_time

### download_data_for_timeframe (Line 1020)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, timeframe, start_time, end_time

---

## src/training/steps/data_collection/raw_data_quality_checker_refactored.py

**Functions with too many arguments: 1**

### validate_raw_data (Line 151)
- **Argument count:** 6
- **Arguments:** self, data, symbol, exchange, auto_fix, timeframe

---

## src/training/steps/data_collection/raw_data_quality_checker_simplified.py

**Functions with too many arguments: 2**

### enhanced_preprocess_market_data (Line 402)
- **Argument count:** 7
- **Arguments:** data, symbol, exchange, expected_interval_seconds, max_forward_fill_seconds, download_missing_data, config

### validate_raw_data (Line 70)
- **Argument count:** 6
- **Arguments:** self, data, symbol, exchange, auto_fix, auto_download_missing

---

## src/training/steps/data_collection/step01_data_collection_validator.py

**Functions with too many arguments: 1**

### validate_dataframe_quality (Line 158)
- **Argument count:** 8
- **Arguments:** self, df, min_rows, required_columns, check_data_types, check_value_ranges, check_duplicates, check_temporal_consistency

---

## src/training/steps/data_collection/step02_data_reading.py

**Functions with too many arguments: 1**

### comprehensive_function_monitoring (Line 404)
- **Argument count:** 6
- **Arguments:** validate_inputs, validate_outputs, track_performance, track_memory, timeout_seconds, retry_attempts

---

## src/training/steps/data_collection/test_step02_standalone.py

**Functions with too many arguments: 1**

### comprehensive_function_monitoring (Line 405)
- **Argument count:** 6
- **Arguments:** validate_inputs, validate_outputs, track_performance, track_memory, timeout_seconds, retry_attempts

---

## src/training/steps/data_collection/utils/data_operations_utils.py

**Functions with too many arguments: 2**

### _log_data_access (Line 507)
- **Argument count:** 6
- **Arguments:** self, user_id, data_type, symbol, exchange, granted

### save_data (Line 541)
- **Argument count:** 6
- **Arguments:** self, data, file_path, format, compression, metadata

---

## src/training/steps/market_analysis/cross_timeframe_interaction_features.py

**Functions with too many arguments: 2**

### _calculate_hl_momentum (Line 174)
- **Argument count:** 6
- **Arguments:** self, high, low, close, tf1, tf2

### _calculate_range_pair (Line 228)
- **Argument count:** 6
- **Arguments:** self, high, low, close, tf1, tf2

---

## src/training/steps/market_analysis/enhanced_logging_metrics.py

**Functions with too many arguments: 1**

### end_step (Line 241)
- **Argument count:** 7
- **Arguments:** self, step_name, success, error_message, input_shape, output_shape, memory_usage_mb

---

## src/training/steps/market_analysis/enhanced_pipeline_decorators.py

**Functions with too many arguments: 1**

### comprehensive_pipeline_protection (Line 601)
- **Argument count:** 11
- **Arguments:** required_columns, data_types, validation_rules, max_memory_mb, max_execution_time, allowed_operations, forbidden_operations, allowed_paths, forbidden_paths, require_authentication, audit_access

---

## src/training/steps/market_analysis/fractional_feature_selector.py

**Functions with too many arguments: 2**

### get_fractional_feature_selector_config (Line 680)
- **Argument count:** 9
- **Arguments:** min_features, max_features, target_feature_count, selection_methods, method_weights, correlation_threshold, vif_threshold, alignment_window, alignment_threshold

### _track_selection_history (Line 561)
- **Argument count:** 6
- **Arguments:** self, original_features, selected_features, metrics, hmm_regime, processing_time

---

## src/training/steps/market_analysis/hmm_clustering/step03_bayesian_parameter_optimization.py

**Functions with too many arguments: 1**

### _perform_clustering (Line 282)
- **Argument count:** 6
- **Arguments:** self, features_processed, hmm_states, hmm_probs, trial_params, scaler

---

## src/training/steps/market_analysis/hmm_clustering/step03_dynamic_regime_optimization.py

**Functions with too many arguments: 1**

### _combine_optimization_results (Line 632)
- **Argument count:** 9
- **Arguments:** self, ic_results, cv_results, economic_results, persistence_results, market_adaptation, enhanced_results, balance_results, stability_results

---

## src/training/steps/market_analysis/hmm_clustering/step03_hmm_regime_discovery.py

**Functions with too many arguments: 2**

### _create_meta_information (Line 1282)
- **Argument count:** 6
- **Arguments:** self, hmm_model, kmeans_model, composite_analysis, cluster_metrics, reports

### _should_run_optimization (Line 1573)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, timeframe, data_dir, force_rerun

---

## src/training/steps/market_analysis/hmm_clustering/step03_ml_transition_detector.py

**Functions with too many arguments: 1**

### _train_single_model (Line 344)
- **Argument count:** 6
- **Arguments:** self, model_type, X_train, X_test, y_train, y_test

---

## src/training/steps/market_analysis/hmm_clustering/step03_optimized_bayesian_optimization.py

**Functions with too many arguments: 1**

### _evaluate_parameter_combination (Line 426)
- **Argument count:** 6
- **Arguments:** self, trial, data, features, parameter_space, coarse

---

## src/training/steps/market_analysis/hmm_feature_enhancer.py

**Functions with too many arguments: 1**

### _calculate_mfi (Line 312)
- **Argument count:** 6
- **Arguments:** self, high, low, close, volume, period

---

## src/training/steps/market_analysis/progress_monitor.py

**Functions with too many arguments: 1**

### update_step_progress (Line 88)
- **Argument count:** 8
- **Arguments:** self, step_name, progress, message, status, details, step_number, total_steps

---

## src/training/steps/market_analysis/step04_regime_data_splitting.py

**Functions with too many arguments: 3**

### _save_unified_dataset (Line 494)
- **Argument count:** 6
- **Arguments:** self, data, training_dir, exchange, symbol, timeframe

### _save_regime_statistics (Line 506)
- **Argument count:** 7
- **Arguments:** self, data, regime_ids, training_dir, exchange, symbol, timeframe

### _save_regime_labels (Line 521)
- **Argument count:** 7
- **Arguments:** self, data, regime_ids, training_dir, exchange, symbol, timeframe

---

## src/training/steps/market_analysis/step07_enhanced_matrix_operations_simplified.py

**Functions with too many arguments: 1**

### _update_pipeline_state (Line 493)
- **Argument count:** 12
- **Arguments:** self, pipeline_state, start_time, output_files, matrix_results, quality_metrics, feature_optimization_results, timeframe_analysis_results, filtering_metadata, symbol, exchange, timeframe

---

## src/training/steps/market_analysis/step08_advanced_feature_selection.py

**Functions with too many arguments: 1**

### _select_features_with_redundancy_advanced (Line 1192)
- **Argument count:** 6
- **Arguments:** self, feature_importance, all_redundancy_groups, target_size, confirmed_features, boruta_selector

---

## src/training/steps/market_analysis/step1/data_quality_monitor.py

**Functions with too many arguments: 2**

### __init__ (Line 28)
- **Argument count:** 9
- **Arguments:** self, alert_type, severity, message, symbol, exchange, timeframe, timestamp, details

### get_alerts (Line 249)
- **Argument count:** 8
- **Arguments:** self, symbol, exchange, severity, alert_type, start_time, end_time, limit

---

## src/training/steps/market_analysis/step1/data_resampler.py

**Functions with too many arguments: 2**

### save_resampled_data (Line 266)
- **Argument count:** 6
- **Arguments:** self, df, symbol, exchange, timeframe, output_format

### resample_all_timeframes (Line 414)
- **Argument count:** 7
- **Arguments:** self, symbol, exchange, timeframes, start_date, end_date, create_partitions

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/confidence_based_entry_logic.py

**Functions with too many arguments: 1**

### _generate_entry_reasoning (Line 362)
- **Argument count:** 7
- **Arguments:** self, should_enter, all_above_minimum, combined_meets_threshold, consistent_confidence, confidence_spread, combined_confidence

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/optimized_optuna_optimization.py

**Functions with too many arguments: 1**

### optimize (Line 140)
- **Argument count:** 9
- **Arguments:** self, model_type, X, y, n_trials, n_jobs, cv_folds, early_stopping_patience, subsample_fraction

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/optimized_optuna_optimization_enhanced.py

**Functions with too many arguments: 3**

### __init__ (Line 108)
- **Argument count:** 7
- **Arguments:** self, storage_url, study_name_prefix, config, enable_gpu, enable_jit, cache_size

### optimize (Line 253)
- **Argument count:** 12
- **Arguments:** self, model_type, X, y, n_trials, n_jobs, cv_folds, early_stopping_patience, subsample_fraction, custom_objective, custom_space, batch_size

### _evaluate_ml_model_vectorized (Line 370)
- **Argument count:** 7
- **Arguments:** self, trial, model_type, X, y, cv_folds, subsample_fraction

---

## src/training/steps/market_analysis/step17_final_parameters_optimization/sr_optuna_optimization.py

**Functions with too many arguments: 1**

### _calculate_performance_metrics (Line 196)
- **Argument count:** 7
- **Arguments:** self, sr_features, target_returns, level_params, breakout_params, zone_params, confidence_params

---

## src/training/steps/market_analysis/utils/feature_filtering.py

**Functions with too many arguments: 1**

### apply_combined_filtering (Line 300)
- **Argument count:** 6
- **Arguments:** self, features_df, labels_df, regime_labels, variance_threshold, correlation_threshold

---

## src/training/steps/model_training/step09_5_hmm_lm_generalist_training.py

**Functions with too many arguments: 1**

### __init__ (Line 951)
- **Argument count:** 6
- **Arguments:** self, input_dim, num_regimes, d_model, nhead, num_layers

---

## src/training/steps/model_training/step14_tactician_labeling.py

**Functions with too many arguments: 1**

### _save_results (Line 914)
- **Argument count:** 6
- **Arguments:** self, labeled_data, signals, data_dir, exchange, symbol

---

## src/training/steps/optimisation/step16_confidence_calibration_per_regime.py

**Functions with too many arguments: 1**

### _create_enhanced_results (Line 313)
- **Argument count:** 6
- **Arguments:** self, calibration_results, symbol, exchange, timeframe, regime_id

---

## src/training/steps/step06_enhanced_validation_framework.py

**Functions with too many arguments: 1**

### start_call (Line 93)
- **Argument count:** 6
- **Arguments:** self, function_name, module_name, call_id, args, kwargs

---

## src/training/steps/step06_labeling_components/fractional_triple_barrier_labeling.py

**Functions with too many arguments: 1**

### __init__ (Line 29)
- **Argument count:** 6
- **Arguments:** self, profit_take_multiplier, stop_loss_multiplier, time_barrier_minutes, max_lookahead, fractional_config

---

## src/training/steps/step06_labeling_components/optimized_triple_barrier_labeling.py

**Functions with too many arguments: 2**

### _numba_triple_barrier_labels (Line 110)
- **Argument count:** 6
- **Arguments:** close, high, low, pt_mult, sl_mult, end_idx_arr

### __init__ (Line 169)
- **Argument count:** 6
- **Arguments:** self, profit_take_multiplier, stop_loss_multiplier, time_barrier_minutes, max_lookahead, binary_classification

---

## src/training/steps/step06_labeling_components/profit_based_feature_engineering.py

**Functions with too many arguments: 1**

### __init__ (Line 99)
- **Argument count:** 6
- **Arguments:** self, profit_column, volume_column, price_column, use_numba, memory_efficient

---

## src/training/steps/step06_labeling_components/regime_aware_triple_barrier_labeling.py

**Functions with too many arguments: 3**

### apply_regime_aware_triple_barrier_labeling_with_barriers (Line 482)
- **Argument count:** 6
- **Arguments:** data, barrier_map_or_path, regime_column, binary_classification, default_time_barrier_minutes, default_max_lookahead

### _numba_regime_aware_triple_barrier_labels (Line 28)
- **Argument count:** 7
- **Arguments:** close, high, low, regime_ids, pt_multipliers, sl_multipliers, end_idx_arr

### set_regime_parameters (Line 139)
- **Argument count:** 9
- **Arguments:** self, regime_name, profit_take_multiplier, stop_loss_multiplier, time_barrier_minutes, max_lookahead, tp_multiplier, sl_multiplier, position_size

---

## src/training/steps/step06_labeling_components/regime_specific_triple_barrier_optimizer.py

**Functions with too many arguments: 1**

### _calculate_regime_performance_score (Line 181)
- **Argument count:** 6
- **Arguments:** self, regime_name, barrier_params, labeling_params, position_params, risk_params

---

## src/training/tpsl_optimizer.py

**Functions with too many arguments: 2**

### _numba_backtest (Line 38)
- **Argument count:** 12
- **Arguments:** close_prices, low_prices, high_prices, signals, ml_buy_confidence, ml_sell_confidence, tp_long, sl_long, tp_short, sl_short, enable_ml_early_exit, early_exit_confidence

### _run_backtest (Line 236)
- **Argument count:** 7
- **Arguments:** self, tp_long, sl_long, tp_short, sl_short, enable_ml_early_exit, early_exit_confidence

---

## src/training/unified_data_orchestrator.py

**Functions with too many arguments: 1**

### _generate_resampling_cache_key (Line 1440)
- **Argument count:** 6
- **Arguments:** self, data, from_timeframe, to_timeframe, symbol, exchange

---

## src/training/wavelet_feature_selection_workflow.py

**Functions with too many arguments: 1**

### _generate_summary_report (Line 1021)
- **Argument count:** 7
- **Arguments:** self, analysis_results, discovery_model_results, feature_results, winner_features, production_model_results, live_configs

---

## src/transition/event_trigger_indexer.py

**Functions with too many arguments: 1**

### build_event_index (Line 233)
- **Argument count:** 7
- **Arguments:** self, combined_df, price_data, volume_data, candidate_labels, timeframe, instrument_id

---

## src/transition/seq2seq_trainer.py

**Functions with too many arguments: 4**

### build_dataloaders (Line 175)
- **Argument count:** 6
- **Arguments:** samples, numeric_dim, label_index, post_len, batch_size, seed

### train_seq2seq (Line 223)
- **Argument count:** 16
- **Arguments:** samples, label_index, numeric_feature_names, post_window, d_model, nhead, num_layers, max_epochs, lr, path_class_weights, focal_gamma, precision, artifact_dir_models, cv_folds, pt_mult, model_type

### __init__ (Line 65)
- **Argument count:** 10
- **Arguments:** self, hmm_vocab, num_features, d_model, nhead, num_layers, dropout, lr, path_class_weights, focal_gamma

### __init__ (Line 158)
- **Argument count:** 9
- **Arguments:** self, hmm_vocab, num_features, d_model, layers, dropout, lr, path_class_weights, focal_gamma

---

## src/utils/confidence.py

**Functions with too many arguments: 2**

### calculate_multi_output_confidence (Line 212)
- **Argument count:** 9
- **Arguments:** direction_probability, direction_prediction, profit_prediction, current_price, predicted_price, direction_threshold, profit_threshold, price_threshold, min_ensemble_confidence

### calculate_multi_output_confidence_batch (Line 260)
- **Argument count:** 9
- **Arguments:** direction_probabilities, direction_predictions, profit_predictions, current_prices, predicted_prices, direction_threshold, profit_threshold, price_threshold, min_ensemble_confidence

---

## src/utils/cross_step_validation.py

**Functions with too many arguments: 2**

### validate_step_transition (Line 19)
- **Argument count:** 6
- **Arguments:** self, previous_step_output, current_step_input, previous_step_name, current_step_name, tolerance

### _store_validation_metadata (Line 129)
- **Argument count:** 6
- **Arguments:** self, prev_step, curr_step, result, prev_rows, curr_rows

---

## src/utils/cross_step_validator.py

**Functions with too many arguments: 7**

### validate_step_transition (Line 113)
- **Argument count:** 6
- **Arguments:** self, from_step, to_step, input_data, output_data, step_metadata

### _check_timestamp_continuity (Line 239)
- **Argument count:** 6
- **Arguments:** self, from_step, to_step, input_data, output_data, metadata

### _check_volume_consistency (Line 282)
- **Argument count:** 6
- **Arguments:** self, from_step, to_step, input_data, output_data, metadata

### _check_price_range_validation (Line 327)
- **Argument count:** 6
- **Arguments:** self, from_step, to_step, input_data, output_data, metadata

### _check_column_preservation (Line 382)
- **Argument count:** 6
- **Arguments:** self, from_step, to_step, input_data, output_data, metadata

### _check_data_shape_consistency (Line 418)
- **Argument count:** 6
- **Arguments:** self, from_step, to_step, input_data, output_data, metadata

### _check_step_specific_rules (Line 459)
- **Argument count:** 6
- **Arguments:** self, from_step, to_step, input_data, output_data, metadata

---

## src/utils/data_loader.py

**Functions with too many arguments: 2**

### load_partitioned_data (Line 305)
- **Argument count:** 8
- **Arguments:** exchange, symbol, data_type, timeframe, base_dir, max_rows, use_streaming, logger

### load_partitioned_data (Line 33)
- **Argument count:** 14
- **Arguments:** self, base_dir, exchange, symbol, data_type, timeframe, filters, columns, max_rows, use_streaming, enable_partition_pruning, use_cache, cache_key
- **Has **kwargs:** Yes

---

## src/utils/decorator_registry.py

**Functions with too many arguments: 3**

### register_decorator (Line 467)
- **Argument count:** 6
- **Arguments:** name, version, description, tags, deprecated, aliases

### __init__ (Line 31)
- **Argument count:** 7
- **Arguments:** self, name, decorator, version, description, tags, deprecated

### register (Line 188)
- **Argument count:** 8
- **Arguments:** self, name, decorator, version, description, tags, deprecated, aliases

---

## src/utils/enhanced_missing_value_handler.py

**Functions with too many arguments: 3**

### handle_missing_values_intelligently (Line 55)
- **Argument count:** 6
- **Arguments:** self, data, timestamp_column, symbol, exchange, timeframe

### _handle_large_gap_with_download (Line 182)
- **Argument count:** 7
- **Arguments:** self, data, gap, timestamp_column, symbol, exchange, timeframe

### _download_missing_data (Line 213)
- **Argument count:** 6
- **Arguments:** self, symbol, exchange, timeframe, start_time, end_time

---

## src/utils/enhanced_mlflow_integration.py

**Functions with too many arguments: 8**

### log_step_artifact (Line 199)
- **Argument count:** 6
- **Arguments:** config, step_name, artifact_path, artifact_type, run_id, additional_metadata

### generate_standardized_artifact_name (Line 254)
- **Argument count:** 6
- **Arguments:** exchange, token, step_number, artifact_type, extension, timestamp

### log_step_dataframe (Line 317)
- **Argument count:** 6
- **Arguments:** config, step_name, df, artifact_name, run_id, additional_metadata

### log_step_dataframe_with_standardized_name (Line 446)
- **Argument count:** 6
- **Arguments:** config, step_name, df, artifact_type, run_id, additional_metadata

### log_step_artifact_with_standardized_name (Line 496)
- **Argument count:** 6
- **Arguments:** config, step_name, artifact_path, artifact_type, run_id, additional_metadata

### log_step_report (Line 549)
- **Argument count:** 6
- **Arguments:** config, step_name, report_data, report_type, run_id, additional_metadata

### log_step_model (Line 618)
- **Argument count:** 7
- **Arguments:** config, step_name, model, model_name, model_type, run_id, additional_metadata

### create_detailed_step_report (Line 1233)
- **Argument count:** 7
- **Arguments:** step_name, step_data, training_input, execution_metadata, artifacts_generated, metrics_calculated, errors_encountered

---

## src/utils/enhanced_outlier_handler.py

**Functions with too many arguments: 4**

### __init__ (Line 27)
- **Argument count:** 6
- **Arguments:** self, name, required_columns, optional_columns, data_types, constraints

### __init__ (Line 95)
- **Argument count:** 7
- **Arguments:** self, column, indices, values, method, severity, threshold

### detect_outliers (Line 140)
- **Argument count:** 6
- **Arguments:** self, data, method, threshold, columns, raise_errors

### create_custom_schema (Line 377)
- **Argument count:** 6
- **Arguments:** self, name, required_columns, optional_columns, data_types, constraints

---

## src/utils/enhanced_step_wrapper.py

**Functions with too many arguments: 2**

### __init__ (Line 20)
- **Argument count:** 6
- **Arguments:** self, step_class, step_name, enable_streaming, enable_cross_step_validation, enable_advanced_quality

### create_enhanced_step (Line 323)
- **Argument count:** 6
- **Arguments:** self, step_class, step_name, enable_streaming, enable_cross_step_validation, enable_advanced_quality

---

## src/utils/feature_engineering_validation.py

**Functions with too many arguments: 2**

### validate_engineered_features (Line 20)
- **Argument count:** 6
- **Arguments:** self, original_df, features_df, feature_config, validate_calculations, check_dependencies

### _validate_feature_completeness (Line 72)
- **Argument count:** 6
- **Arguments:** self, original_df, features_df, feature_config, result, summary

---

## src/utils/hmm_composite_manager.py

**Functions with too many arguments: 1**

### load_composite_clusters (Line 165)
- **Argument count:** 6
- **Arguments:** self, exchange, symbol, timeframe, data_dir, auto_create

---

## src/utils/logger.py

**Functions with too many arguments: 1**

### log_step_progress (Line 757)
- **Argument count:** 7
- **Arguments:** logger, step_name, step_number, total_steps, status, details, context

---

## src/utils/lookahead_bias_detector.py

**Functions with too many arguments: 1**

### validate_train_test_split (Line 594)
- **Argument count:** 6
- **Arguments:** self, X_train, X_test, y_train, y_test, timestamp_col

---

## src/utils/mlflow_utils.py

**Functions with too many arguments: 5**

### log_enhanced_training_metadata (Line 104)
- **Argument count:** 7
- **Arguments:** asset, exchange, lookback_period, project_version, training_date, additional_metadata, run_id

### log_model_with_metadata (Line 139)
- **Argument count:** 9
- **Arguments:** model, model_name, asset, exchange, lookback_period, project_version, training_date, additional_metadata, run_id

### log_artifacts_with_metadata (Line 170)
- **Argument count:** 9
- **Arguments:** local_path, artifact_path, asset, exchange, lookback_period, project_version, training_date, additional_metadata, run_id

### log_metrics_with_metadata (Line 201)
- **Argument count:** 9
- **Arguments:** metrics, asset, exchange, lookback_period, project_version, training_date, additional_metadata, run_id, step

### log_params_with_metadata (Line 240)
- **Argument count:** 8
- **Arguments:** params, asset, exchange, lookback_period, project_version, training_date, additional_metadata, run_id

---

## src/utils/regime_data_access.py

**Functions with too many arguments: 1**

### split_train_val_test_by_regime (Line 138)
- **Argument count:** 6
- **Arguments:** df, regime_column, train_ratio, val_ratio, test_ratio, min_samples_per_split

---

## src/utils/report_collector.py

**Functions with too many arguments: 3**

### collect_report (Line 391)
- **Argument count:** 6
- **Arguments:** report_content, report_name, report_type, symbol, exchange, step_name

### collect_report (Line 57)
- **Argument count:** 7
- **Arguments:** self, report_content, report_name, report_type, symbol, exchange, step_name

### copy_existing_report (Line 179)
- **Argument count:** 7
- **Arguments:** self, source_path, report_name, report_type, symbol, exchange, step_name

---

## src/utils/security_framework.py

**Functions with too many arguments: 1**

### log_security_event (Line 377)
- **Argument count:** 6
- **Arguments:** self, event_type, user_id, action, details, severity

---

## src/utils/statistical_distribution_validation.py

**Functions with too many arguments: 1**

### validate_distribution (Line 23)
- **Argument count:** 6
- **Arguments:** self, df, columns, expected_distribution, outlier_threshold, check_stationarity

---

## Summary

- **Total files affected:** 116
- **Total functions needing refactoring:** 231
