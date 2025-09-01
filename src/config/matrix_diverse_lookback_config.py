# src/config/matrix_diverse_lookback_config.py

"""
Matrix-Based Diverse Lookback Period Configuration

Configuration settings for matrix/vector-based optimization of diverse lookback periods.
"""



def get_matrix_diverse_lookback_config() -> dict[str, Any]:
    """
Get matrix-based diverse lookback period optimization configuration.

Returns:
        dict: Configuration dictionary
"""
    return {
"matrix_diverse_lookback_optimization": {
"target_periods_per_feature": 3,
"min_periods_per_feature": 2,
"max_periods_per_feature": 3,
"diversity_threshold": 0.3,
"meaningful_threshold": 0.1,
"correlation_threshold": 0.7,
"quality_thresholds": {
"min_diversity_score": 0.2,
"min_information_score": 0.05,
"max_correlation": 0.8,
"min_periods_for_3": 2  # Minimum meaningful periods needed for 3-period selection
},
"matrix_optimization": {
"enabled": True,
"method": "scipy",  # "scipy", "optuna", "greedy"
"max_iterations": 1000,
"tolerance": 1e-6,
"optimization_objectives": {
"information_weight": 0.6,
"diversity_weight": 0.4,
"correlation_penalty": 0.5
}
},
"vector_operations": {
"enabled": True,
"batch_size": 1000,
"parallel_processing": True,
"memory_efficient": True,
"chunk_size": 500
},
"file_output": {
"output_directory": "data/matrix_diverse_lookback_optimization",
"step_parameters_directory": "data/optimized_feature_parameters",
"save_detailed_results": True,
"save_summary": True,
"save_matrix_details": True,
"save_optimized_parameters": True,
"compression": "gzip",
"file_format": "json"
},
"lookback_ranges": {
"RSI": {
"min": 5,
"max": 50,
"step": 2,
"description": "RSI lookback periods for momentum analysis",
"expected_insights": ["Short-term momentum", "Medium-term trend", "Long-term trend"]
},
"MACD_fast": {
"min": 5,
"max": 25,
"step": 1,
"description": "MACD fast period for quick signal generation",
"expected_insights": ["Quick momentum", "Fast trend changes", "Short-term signals"]
},
"MACD_slow": {
"min": 20,
"max": 40,
"step": 2,
"description": "MACD slow period for trend confirmation",
"expected_insights": ["Trend confirmation", "Medium-term trend", "Signal filtering"]
},
"Bollinger_Bands": {
"min": 10,
"max": 50,
"step": 2,
"description": "Bollinger Bands lookback for volatility analysis",
"expected_insights": ["Volatility regime", "Price extremes", "Mean reversion"]
},
"SMA_short": {
"min": 3,
"max": 20,
"step": 1,
"description": "Short SMA for immediate trend detection",
"expected_insights": ["Immediate trend", "Quick reversals", "Short-term support/resistance"]
},
"SMA_long": {
"min": 20,
"max": 100,
"step": 5,
"description": "Long SMA for major trend identification",
"expected_insights": ["Major trend", "Long-term support/resistance", "Trend strength"]
},
"EMA_short": {
"min": 3,
"max": 20,
"step": 1,
"description": "Short EMA for responsive trend detection",
"expected_insights": ["Responsive trend", "Quick signals", "Short-term momentum"]
},
"EMA_long": {
"min": 20,
"max": 100,
"step": 5,
"description": "Long EMA for major trend confirmation",
"expected_insights": ["Major trend confirmation", "Long-term momentum", "Trend persistence"]
},
"ATR": {
"min": 5,
"max": 30,
"step": 1,
"description": "ATR for volatility measurement",
"expected_insights": ["Volatility regime", "Risk assessment", "Position sizing"]
},
"Stochastic_k": {
"min": 5,
"max": 30,
"step": 1,
"description": "Stochastic %K for momentum analysis",
"expected_insights": ["Momentum extremes", "Overbought/oversold", "Divergence detection"]
},
"Stochastic_d": {
"min": 3,
"max": 10,
"step": 1,
"description": "Stochastic %D for signal confirmation",
"expected_insights": ["Signal confirmation", "Momentum smoothing", "Trend confirmation"]
},
"ADX": {
"min": 5,
"max": 30,
"step": 1,
"description": "ADX for trend strength measurement",
"expected_insights": ["Trend strength", "Market regime", "Directional movement"]
},
"CCI": {
"min": 5,
"max": 30,
"step": 1,
"description": "CCI for cyclical analysis",
"expected_insights": ["Cyclical extremes", "Mean reversion", "Momentum cycles"]
},
"Williams_R": {
"min": 5,
"max": 30,
"step": 1,
"description": "Williams %R for momentum extremes",
"expected_insights": ["Overbought/oversold levels", "Momentum reversal", "Divergence signals"]
},
"MFI": {
"min": 5,
"max": 30,
"step": 1,
"description": "Money Flow Index for volume-price analysis",
"expected_insights": ["Volume-price divergence", "Money flow trends", "Market sentiment"]
},
"ROC": {
"min": 5,
"max": 30,
"step": 1,
"description": "Rate of Change for momentum measurement",
"expected_insights": ["Momentum acceleration", "Trend strength", "Price momentum"]
},
"MOM": {
"min": 5,
"max": 30,
"step": 1,
"description": "Momentum indicator for price momentum",
"expected_insights": ["Price momentum", "Trend continuation", "Momentum shifts"]
},
"TSI": {
"min": 5,
"max": 30,
"step": 1,
"description": "True Strength Index for trend analysis",
"expected_insights": ["Trend strength", "Momentum confirmation", "Signal generation"]
},
"UO": {
"min": 5,
"max": 30,
"step": 1,
"description": "Ultimate Oscillator for multi-timeframe analysis",
"expected_insights": ["Multi-timeframe momentum", "Overbought/oversold", "Divergence detection"]
},
"AO": {
"min": 5,
"max": 30,
"step": 1,
"description": "Awesome Oscillator for momentum analysis",
"expected_insights": ["Momentum shifts", "Trend changes", "Signal generation"]
},
"CMF": {
"min": 5,
"max": 30,
"step": 1,
"description": "Chaikin Money Flow for volume analysis",
"expected_insights": ["Money flow trends", "Volume confirmation", "Market sentiment"]
},
"VWAP": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Weighted Average Price for price analysis",
"expected_insights": ["Fair value", "Price efficiency", "Volume-weighted trends"]
},
"Pivot_Points": {
"min": 5,
"max": 30,
"step": 1,
"description": "Pivot Points for support/resistance",
"expected_insights": ["Support/resistance levels", "Price reversals", "Trading ranges"]
},
"Ichimoku": {
"min": 5,
"max": 30,
"step": 1,
"description": "Ichimoku Cloud for trend analysis",
"expected_insights": ["Trend direction", "Support/resistance", "Cloud analysis"]
},
"Parabolic_SAR": {
"min": 5,
"max": 30,
"step": 1,
"description": "Parabolic SAR for trend following",
"expected_insights": ["Trend following", "Stop loss levels", "Trend reversals"]
},
"Keltner_Channels": {
"min": 5,
"max": 30,
"step": 1,
"description": "Keltner Channels for volatility analysis",
"expected_insights": ["Volatility bands", "Price channels", "Breakout signals"]
},
"Donchian_Channels": {
"min": 5,
"max": 30,
"step": 1,
"description": "Donchian Channels for breakout analysis",
"expected_insights": ["Breakout levels", "Trading ranges", "Trend channels"]
},
"Price_Channels": {
"min": 5,
"max": 30,
"step": 1,
"description": "Price Channels for range analysis",
"expected_insights": ["Price ranges", "Channel breakouts", "Range trading"]
},
"Volume_Profile": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Profile for volume analysis",
"expected_insights": ["Volume distribution", "Price levels", "Market structure"]
},
"OBV": {
"min": 5,
"max": 30,
"step": 1,
"description": "On Balance Volume for volume analysis",
"expected_insights": ["Volume trends", "Price confirmation", "Divergence detection"]
},
"AD": {
"min": 5,
"max": 30,
"step": 1,
"description": "Accumulation/Distribution for money flow",
"expected_insights": ["Money flow", "Price-volume relationship", "Market sentiment"]
},
"Chaikin_Money_Flow": {
"min": 5,
"max": 30,
"step": 1,
"description": "Chaikin Money Flow for volume analysis",
"expected_insights": ["Money flow trends", "Volume confirmation", "Market sentiment"]
},
"Money_Flow_Index": {
"min": 5,
"max": 30,
"step": 1,
"description": "Money Flow Index for volume-price analysis",
"expected_insights": ["Volume-price divergence", "Money flow trends", "Market sentiment"]
},
"Volume_RSI": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume RSI for volume momentum",
"expected_insights": ["Volume momentum", "Volume trends", "Volume divergence"]
},
"Volume_Stochastic": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Stochastic for volume analysis",
"expected_insights": ["Volume extremes", "Volume trends", "Volume patterns"]
},
"Volume_Price_Trend": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Trend for volume-price analysis",
"expected_insights": ["Volume-price relationship", "Trend confirmation", "Divergence detection"]
},
"Accumulation_Distribution": {
"min": 5,
"max": 30,
"step": 1,
"description": "Accumulation/Distribution for money flow",
"expected_insights": ["Money flow", "Price-volume relationship", "Market sentiment"]
},
"On_Balance_Volume": {
"min": 5,
"max": 30,
"step": 1,
"description": "On Balance Volume for volume analysis",
"expected_insights": ["Volume trends", "Price confirmation", "Divergence detection"]
},
"Volume_Weighted_Average_Price": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Weighted Average Price for price analysis",
"expected_insights": ["Fair value", "Price efficiency", "Volume-weighted trends"]
},
"Volume_Price_Oscillator": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Oscillator for volume-price analysis",
"expected_insights": ["Volume-price divergence", "Trend confirmation", "Signal generation"]
},
"Volume_Price_Confirmation": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Confirmation for trend validation",
"expected_insights": ["Trend validation", "Volume confirmation", "Signal strength"]
},
"Volume_Price_Trend_Indicator": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Trend Indicator for trend analysis",
"expected_insights": ["Trend analysis", "Volume confirmation", "Trend strength"]
},
"Volume_Price_Oscillator_Histogram": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Oscillator Histogram for signal analysis",
"expected_insights": ["Signal analysis", "Histogram patterns", "Trend changes"]
},
"Volume_Price_Oscillator_Signal": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Oscillator Signal for signal generation",
"expected_insights": ["Signal generation", "Trend confirmation", "Entry/exit signals"]
},
"Volume_Price_Oscillator_Trigger": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Oscillator Trigger for trigger signals",
"expected_insights": ["Trigger signals", "Signal timing", "Entry/exit points"]
},
"Volume_Price_Oscillator_Zero_Line": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Oscillator Zero Line for baseline",
"expected_insights": ["Baseline reference", "Zero crossing", "Trend neutrality"]
},
"Volume_Price_Oscillator_Upper_Band": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Oscillator Upper Band for overbought",
"expected_insights": ["Overbought levels", "Upper resistance", "Sell signals"]
},
"Volume_Price_Oscillator_Lower_Band": {
"min": 5,
"max": 30,
"step": 1,
"description": "Volume Price Oscillator Lower Band for oversold",
"expected_insights": ["Oversold levels", "Lower support", "Buy signals"]
},
"VWAP_Momentum": {
"min": 3,
"max": 50,
"step": 1,
"description": "VWAP momentum for trend analysis",
"expected_insights": ["VWAP trend strength", "Volume-weighted momentum", "Trend confirmation"]
},
"VWAP_Acceleration": {
"min": 3,
"max": 50,
"step": 1,
"description": "VWAP acceleration for momentum changes",
"expected_insights": ["Momentum acceleration", "Trend changes", "Signal generation"]
},
"VWAP_Volatility": {
"min": 5,
"max": 50,
"step": 1,
"description": "VWAP volatility for risk assessment",
"expected_insights": ["Volume-weighted volatility", "Risk measurement", "Volatility regime"]
},
"VWAP_Momentum_Volatility": {
"min": 5,
"max": 50,
"step": 1,
"description": "VWAP momentum volatility for momentum stability",
"expected_insights": ["Momentum stability", "Trend reliability", "Signal quality"]
},
"VWAP_Returns": {
"min": 5,
"max": 50,
"step": 1,
"description": "VWAP returns for volume-weighted analysis",
"expected_insights": ["Volume-weighted returns", "Fair value analysis", "Price efficiency"]
},
"VWAP_Log_Returns": {
"min": 5,
"max": 50,
"step": 1,
"description": "VWAP log returns for continuous compounding",
"expected_insights": ["Continuous returns", "Compounding effects", "Return analysis"]
},
"Price_VWAP_Ratio": {
"min": 5,
"max": 50,
"step": 1,
"description": "Price to VWAP ratio for relative valuation",
"expected_insights": ["Relative valuation", "Over/under valued", "Mean reversion"]
},
"Price_VWAP_Deviation": {
"min": 5,
"max": 50,
"step": 1,
"description": "Price to VWAP deviation for mispricing",
"expected_insights": ["Mispricing detection", "Deviation from fair value", "Reversion signals"]
},
"Price_VWAP_Spread": {
"min": 5,
"max": 50,
"step": 1,
"description": "Price to VWAP spread for absolute deviation",
"expected_insights": ["Absolute deviation", "Spread analysis", "Price efficiency"]
}
}
},
"matrix_optimization_methods": {
"scipy": {
"method": "SLSQP",
"constraints": {
"type": "eq",
"fun": "period_count_constraint"
},
"bounds": [(0, 1)],
"options": {
"maxiter": 1000,
"ftol": 1e-6,
"eps": 1e-8
}
},
"optuna": {
"n_trials": 100,
"sampler": "TPESampler",
"pruner": "MedianPruner",
"storage": "sqlite:///matrix_optimization.db"
},
"greedy": {
"start_with_best": True,
"diversity_threshold": 0.3,
"max_iterations": 100
}
},
"vector_optimization_settings": {
"feature_calculation": {
"vectorized": True,
"parallel": True,
"chunk_size": 1000,
"memory_limit": "2GB"
},
"correlation_analysis": {
"method": "pearson",
"vectorized": True,
"handle_nan": "drop",
"min_periods": 100
},
"information_scoring": {
"method": "shap",
"n_estimators": 100,
"random_state": 42,
"vectorized": True
}
},
"file_naming_convention": {
"main_results": "{exchange}_{symbol}_{timeframe}_matrix_diverse_lookback_periods.json",
"summary": "{exchange}_{symbol}_{timeframe}_diverse_periods_summary.json",
"matrix_details": "{exchange}_{symbol}_{timeframe}_matrix_optimization_details.json",
"optimized_parameters": "{exchange}_{symbol}_{timeframe}_optimized_feature_parameters.json",
"regime_specific": "{exchange}_{symbol}_{timeframe}_regime_specific_periods.json"
},
"logging_settings": {
"log_file_paths": True,
"log_optimization_progress": True,
"log_matrix_operations": True,
"log_performance_metrics": True,
"log_file_sizes": True
}
}


def get_matrix_optimization_objectives() -> dict[str, Any]:
    """
Get matrix optimization objective definitions.

Returns:
        dict: Objective definitions
"""
    return {
"objectives": {
"information_maximization": {
"description": "Maximize information content of selected periods",
"weight": 0.6,
"method": "shap_importance",
"normalization": "min_max"
},
"diversity_maximization": {
"description": "Maximize diversity between selected periods",
"weight": 0.4,
"method": "correlation_diversity",
"normalization": "min_max"
},
"correlation_penalization": {
"description": "Penalize high correlations between periods",
"weight": 0.5,
"method": "correlation_penalty",
"normalization": "linear"
}
},
"constraints": {
"period_count": {
"type": "equality",
"value": 3,
"description": "Exactly 3 periods per feature"
},
"meaningful_threshold": {
"type": "inequality",
"value": 0.1,
"description": "Minimum information score"
},
"correlation_threshold": {
"type": "inequality",
"value": 0.7,
"description": "Maximum correlation between periods"
}
}
}


def get_vector_operation_settings() -> dict[str, Any]:
    """
Get vector operation settings for efficient computation.

Returns:
        dict: Vector operation settings
"""
    return {
"vector_operations": {
"feature_calculation": {
"batch_processing": True,
"batch_size": 1000,
"parallel_workers": 4,
"memory_efficient": True,
"chunk_size": 500
},
"correlation_analysis": {
"vectorized_correlation": True,
"numpy_corrcoef": True,
"handle_nan": "drop",
"min_periods": 100,
"efficient_memory": True
},
"information_scoring": {
"vectorized_shap": True,
"batch_shap": True,
"shap_batch_size": 1000,
"parallel_shap": True,
"memory_optimized": True
},
"matrix_operations": {
"numpy_operations": True,
"scipy_optimization": True,
"efficient_matrix": True,
"sparse_matrices": False,
"memory_mapping": False
}
},
"performance_optimization": {
"memory_management": {
"max_memory_usage": "4GB",
"garbage_collection": True,
"memory_monitoring": True,
"chunk_processing": True
},
"parallel_processing": {
"enabled": True,
"n_jobs": -1,
"backend": "multiprocessing",
"batch_size": 1000,
"memory_efficient": True
},
"caching": {
"enable_caching": True,
"cache_directory": "cache/matrix_optimization",
"cache_size": "1GB",
"cache_ttl": 3600
}
}
}


def get_file_output_settings() -> dict[str, Any]:
    """
Get file output settings for saving optimization results.

Returns:
        dict: File output settings
"""
    return {
"output_directories": {
"main_output": "data/matrix_diverse_lookback_optimization",
"step_parameters": "data/optimized_feature_parameters",
"cache": "cache/matrix_optimization",
"logs": "logs/matrix_optimization"
},
"file_formats": {
"main_results": "json",
"summary": "json",
"matrix_details": "json",
"optimized_parameters": "json",
"logs": "txt"
},
"compression": {
"enabled": True,
"method": "gzip",
"level": 6,
"extensions": [".json", ".txt"]
},
"file_organization": {
"create_date_folders": True,
"create_symbol_folders": True,
"backup_previous_results": True,
"max_backup_count": 5
},
"logging": {
"log_file_paths": True,
"log_file_sizes": True,
"log_optimization_progress": True,
"log_performance_metrics": True,
"log_memory_usage": True
}
}


def get_integration_settings() -> dict[str, Any]:
    """
Get integration settings for subsequent steps.

Returns:
        dict: Integration settings
"""
    return {
"subsequent_steps": {
"step7": {
"load_optimized_parameters": True,
"parameter_file_path": "data/optimized_feature_parameters/{exchange}_{symbol}_{timeframe}_optimized_feature_parameters.json",
"fallback_parameters": "default_feature_parameters",
"validate_parameters": True
},
"step8": {
"load_optimized_parameters": True,
"use_optimized_periods": True,
"feature_generation": "optimized",
"parameter_validation": True
},
"step9": {
"load_optimized_parameters": True,
"feature_engineering": "optimized",
"period_optimization": True
}
},
"parameter_validation": {
"validate_periods": True,
"validate_diversity": True,
"validate_information_scores": True,
"validate_file_integrity": True,
"fallback_strategy": "use_defaults"
},
"feature_generation": {
"use_optimized_periods": True,
"generate_all_periods": False,
"period_combination": "individual",
"feature_naming": "include_period",
"validation": True
}
}