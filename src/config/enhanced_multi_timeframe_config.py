"""
Configuration for Enhanced Multi-Timeframe Optimizer

This module provides configuration settings for the enhanced multi-timeframe
optimizer that uses optimized lookback periods from the matrix optimization system.
"""

from typing import Dict, List, Any

def get_enhanced_multi_timeframe_config() -> Dict[str, Any]:
    """Get configuration for enhanced multi-timeframe optimization."""
return {
"enhanced_multi_timeframe_optimization": {
"enabled": True,
"base_timeframes": ["1m", "5m", "15m", "30m", "1h"],
"cross_timeframe_enabled": True,
"regime_specific": True,
"quality_thresholds": {
"min_correlation": 0.3,
"max_correlation": 0.8,
"min_information_score": 0.05,
"min_diversity_score": 0.2
},
"optimization_settings": {
"max_cross_timeframe_pairs": 20,
"min_period_difference_ratio": 1.5,  # At least 50% difference
"diversity_selection_method": "greedy",
"quality_filtering": True,
"regime_aware_optimization": True
},
"feature_generation": {
"momentum_features": True,
"volatility_features": True,
"volume_features": True,
"range_features": True,
"regime_features": True,
"cross_timeframe_features": True
},
"performance_settings": {
"parallel_processing": True,
"batch_size": 1000,
"memory_limit_mb": 2048,
"cache_results": True
}
},
"timeframe_specific_settings": {
"1m": {
"min_data_points": 100,
"resampling_method": "none",
"feature_types": ["all"]
},
"5m": {
"min_data_points": 50,
"resampling_method": "ohlcv",
"feature_types": ["momentum", "volatility", "volume"]
},
"15m": {
"min_data_points": 30,
"resampling_method": "ohlcv",
"feature_types": ["momentum", "volatility", "volume"]
},
"30m": {
"min_data_points": 20,
"resampling_method": "ohlcv",
"feature_types": ["momentum", "volatility"]
},
"1h": {
"min_data_points": 10,
"resampling_method": "ohlcv",
"feature_types": ["momentum", "volatility"]
}
},
"cross_timeframe_optimization": {
"period_pair_selection": {
"method": "diverse_optimization",
"max_pairs": 20,
"min_difference_ratio": 1.5,
"diversity_metric": "correlation_distance"
},
"feature_types": {
"momentum": {
"enabled": True,
"methods": ["difference", "ratio", "high_low_momentum"]
},
"volatility": {
"enabled": True,
"methods": ["ratio", "difference", "volatility_of_volatility"]
},
"volume": {
"enabled": True,
"methods": ["ratio", "difference", "momentum"]
},
"range": {
"enabled": True,
"methods": ["ratio", "difference"]
}
}
},
"regime_specific_optimization": {
"enabled": True,
"regime_detection": {
"method": "hmm",
"min_regime_data_points": 100,
"regime_stability_threshold": 0.7
},
"regime_feature_generation": {
"use_regime_specific_periods": True,
"fallback_to_general_periods": True,
"regime_feature_prefix": "regime_"
}
},
"quality_validation": {
"variance_threshold": 1e-12,
"correlation_thresholds": {
"min_with_target": 0.3,
"max_between_features": 0.8
},
"information_score_threshold": 0.05,
"diversity_score_threshold": 0.2,
"stability_threshold": 0.7
},
"output_settings": {
"save_optimization_results": True,
"save_feature_metadata": True,
"log_optimization_details": True,
"output_format": "json",
"output_directory": "data/optimization_results"
},
"integration_settings": {
"matrix_optimization_integration": {
"enabled": True,
"use_optimized_periods": True,
"fallback_to_default": True,
"period_extraction_method": "diverse_lookback"
},
"pipeline_integration": {
"replace_existing_multi_timeframe": True,
"enhance_existing_features": True,
"backward_compatibility": True
}
}
}

def get_timeframe_period_mapping() -> Dict[str, Dict[str, List[int]]]:
    """Get mapping of timeframes to optimized periods for different indicators."""
return {
"1m": {
"RSI": [7, 14, 21],
"MACD_fast": [8, 12, 16],
"Bollinger_Bands": [10, 20, 30],
"SMA": [5, 20, 50],
"EMA": [5, 20, 50],
"ATR": [10, 20, 30],
"VWAP": [5, 10, 20],
"VWAP_Momentum": [5, 10, 20],
"VWAP_Volatility": [5, 10, 20]
},
"5m": {
"RSI": [5, 10, 15],
"MACD_fast": [6, 10, 14],
"Bollinger_Bands": [8, 15, 25],
"SMA": [4, 15, 40],
"EMA": [4, 15, 40],
"ATR": [8, 15, 25],
"VWAP": [4, 8, 15],
"VWAP_Momentum": [4, 8, 15],
"VWAP_Volatility": [4, 8, 15]
},
"15m": {
"RSI": [4, 8, 12],
"MACD_fast": [5, 8, 12],
"Bollinger_Bands": [6, 12, 20],
"SMA": [3, 12, 30],
"EMA": [3, 12, 30],
"ATR": [6, 12, 20],
"VWAP": [3, 6, 12],
"VWAP_Momentum": [3, 6, 12],
"VWAP_Volatility": [3, 6, 12]
},
"30m": {
"RSI": [3, 6, 10],
"MACD_fast": [4, 6, 10],
"Bollinger_Bands": [5, 10, 15],
"SMA": [2, 10, 25],
"EMA": [2, 10, 25],
"ATR": [5, 10, 15],
"VWAP": [2, 5, 10],
"VWAP_Momentum": [2, 5, 10],
"VWAP_Volatility": [2, 5, 10]
},
"1h": {
"RSI": [2, 5, 8],
"MACD_fast": [3, 5, 8],
"Bollinger_Bands": [4, 8, 12],
"SMA": [2, 8, 20],
"EMA": [2, 8, 20],
"ATR": [4, 8, 12],
"VWAP": [2, 4, 8],
"VWAP_Momentum": [2, 4, 8],
"VWAP_Volatility": [2, 4, 8]
}
}

def get_cross_timeframe_period_pairs() -> List[tuple]:
    """Get optimized period pairs for cross-timeframe analysis."""
return [
(3, 5), (3, 8), (3, 12), (3, 20),
(5, 8), (5, 12), (5, 20), (5, 30),
(8, 12), (8, 20), (8, 30), (8, 50),
(12, 20), (12, 30), (12, 50),
(20, 30), (20, 50),
(30, 50)
]

def get_regime_specific_config() -> Dict[str, Any]:
    """Get configuration for regime-specific optimization."""
return {
"regime_detection": {
"method": "hmm",
"n_regimes": 3,
"min_regime_duration": 100,
"regime_stability_threshold": 0.7
},
"regime_feature_generation": {
"use_regime_specific_periods": True,
"regime_feature_prefix": "regime_",
"regime_specific_indicators": [
"RSI", "MACD_fast", "Bollinger_Bands", "SMA", "EMA",
"ATR", "VWAP", "VWAP_Momentum", "VWAP_Volatility"
]
},
"regime_period_optimization": {
"regime_0": {  # Low volatility regime
"RSI": [5, 10, 15],
"MACD_fast": [6, 10, 14],
"Bollinger_Bands": [8, 15, 25],
"VWAP": [4, 8, 15]
},
"regime_1": {  # Medium volatility regime
"RSI": [7, 14, 21],
"MACD_fast": [8, 12, 16],
"Bollinger_Bands": [10, 20, 30],
"VWAP": [5, 10, 20]
},
"regime_2": {  # High volatility regime
"RSI": [10, 20, 30],
"MACD_fast": [12, 18, 24],
"Bollinger_Bands": [15, 25, 40],
"VWAP": [8, 15, 25]
}
}
}

def get_quality_validation_config() -> Dict[str, Any]:
    """Get configuration for quality validation."""
return {
"variance_threshold": 1e-12,
"correlation_thresholds": {
"min_with_target": 0.3,
"max_between_features": 0.8
},
"information_score_threshold": 0.05,
"diversity_score_threshold": 0.2,
"stability_threshold": 0.7,
"validation_methods": [
"variance_check",
"correlation_analysis",
"information_score",
"diversity_analysis",
"stability_check"
]
}

def get_performance_config() -> Dict[str, Any]:
    """Get configuration for performance optimization."""
return {
"parallel_processing": True,
"batch_size": 1000,
"memory_limit_mb": 2048,
"cache_results": True,
"optimization_levels": {
"data_preprocessing": True,
"feature_calculation": True,
"quality_validation": True,
"result_filtering": True
}
}