# src/config/enhanced_prediction_service_config.py

"""
Enhanced Prediction Service Configuration
Configuration for the Universal ML Profit Integration System
"""

from typing import Dict, Any

def get_enhanced_prediction_service_config() -> Dict[str, Any]:
    """
Get configuration for the Enhanced Prediction Service.

Returns:
        Dict containing configuration settings
"""
    return {
"enhanced_prediction_service": {
# Service configuration
"data_dir": "data/training",
"models_dir": "models",

# Prediction thresholds
"confidence_threshold": 0.7,
"price_prediction_threshold": 0.6,

# ML Profit Integration thresholds
"profit_threshold": 0.02,  # 2% default profit target
"barrier_threshold": 0.01,  # 1% default barrier
"direction_confidence_threshold": 0.65,  # Minimum confidence for directional trades

# Timeframe configuration
"timeframes": ["1m", "5m", "15m", "30m"],
"primary_timeframe": "1m",
"secondary_timeframe": "5m",

# Model loading configuration
"load_hmm_models": True,
"load_analyst_models": True,
"load_tactician_models": True,
"load_ensemble_models": True,

# Confidence calculation parameters
"confidence_calculation": {
"volatility_scaling_factor": 10.0,
"direction_strength_factor": 1.0,
"magnitude_scaling_factor": 5.0,
"barrier_distance_scaling": 10.0,
"volatility_penalty_factor": 8.0,
"direction_boost": 0.1,
"volatility_adjustment_scale": 20.0,
"volatility_adjustment_offset": 5.0
},

# Risk management parameters
"risk_management": {
"max_position_size": 100.0,  # Maximum position size as percentage
"min_position_size": 5.0,    # Minimum position size as percentage
"max_leverage": 3.0,         # Maximum leverage
"min_leverage": 1.0,         # Minimum leverage
"execution_delay_immediate": 0,    # Seconds for immediate execution
"execution_delay_normal": 30,      # Seconds for normal execution
"execution_delay_cautious": 120,   # Seconds for cautious execution
"high_confidence_threshold": 0.8,  # Threshold for high confidence
"medium_confidence_threshold": 0.6, # Threshold for medium confidence
"low_confidence_threshold": 0.4     # Threshold for low confidence
},

# Performance monitoring
"performance_monitoring": {
"enable_prediction_tracking": True,
"enable_confidence_tracking": True,
"enable_risk_tracking": True,
"prediction_history_size": 1000,
"confidence_history_size": 1000,
"risk_history_size": 1000
},

# Integration settings
"integration": {
"enable_analyst_integration": True,
"enable_tactician_integration": True,
"enable_supervisor_integration": True,
"integration_timeout": 30,  # Seconds
"retry_attempts": 3,
"retry_delay": 1  # Seconds
},

# Caching configuration
"caching": {
"enable_caching": True,
"cache_ttl": 300,  # Seconds
"max_cache_size": 1000,
"cache_cleanup_interval": 60  # Seconds
},

# Validation settings
"validation": {
"enable_data_validation": True,
"enable_prediction_validation": True,
"validation_level": "WARNING",  # DEBUG, INFO, WARNING, ERROR
"strict_validation": False
},

# Logging configuration
"logging": {
"log_level": "INFO",
"enable_debug_logging": False,
"log_predictions": True,
"log_confidence_calculations": True,
"log_risk_metrics": True
}
}
}

def get_ml_profit_integration_config() -> Dict[str, Any]:
    """
Get specific configuration for ML Profit Integration.

Returns:
        Dict containing ML profit integration settings
"""
    return {
"ml_profit_integration": {
# Model types to integrate
"model_types": [
"hmm_profit",
"analyst_profit",
"tactician_profit",
"ensemble_profit"
],

# Prediction processing
"prediction_processing": {
"extract_direction": True,
"extract_magnitude": True,
"normalize_predictions": True,
"apply_confidence_calibration": True,
"apply_optimization_weights": True
},

# Confidence enhancement
"confidence_enhancement": {
"enable_barrier_analysis": True,
"enable_volatility_adjustment": True,
"enable_directional_probability": True,
"enable_magnitude_probability": True,
"enable_barrier_avoidance": True,
"confidence_bounds": {
"min_confidence": 0.0,
"max_confidence": 1.0,
"neutral_confidence": 0.5
}
},

# Risk calculation
"risk_calculation": {
"enable_risk_reward_calculation": True,
"enable_expected_value_calculation": True,
"enable_barrier_metrics": True,
"risk_adjustment_factors": {
"volatility_penalty_max": 0.4,
"direction_boost": 0.1,
"magnitude_boost_max": 0.3
}
},

# Integration thresholds
"integration_thresholds": {
"minimum_confidence": 0.3,
"minimum_magnitude": 0.001,
"minimum_risk_reward": 0.5,
"maximum_volatility": 0.1
}
}
}

def get_enhanced_confidence_config() -> Dict[str, Any]:
    """
Get configuration for enhanced confidence calculation.

Returns:
        Dict containing enhanced confidence settings
"""
    return {
"enhanced_confidence": {
# Directional probability calculation
"directional_probability": {
"volatility_scaling": 10.0,
"direction_strength_scaling": 1.0,
"min_probability": 0.1,
"max_probability": 0.95
},

# Magnitude probability calculation
"magnitude_probability": {
"magnitude_scaling": 5.0,
"volatility_boost_max": 0.3,
"min_probability": 0.05,
"max_probability": 0.9
},

# Barrier avoidance probability calculation
"barrier_avoidance": {
"distance_scaling": 10.0,
"volatility_penalty_max": 0.4,
"direction_boost": 0.1,
"min_probability": 0.1,
"max_probability": 0.95
},

# Volatility adjustment
"volatility_adjustment": {
"scaling_factor": 20.0,
"offset": 5.0,
"min_adjustment": 0.5,
"max_adjustment": 1.2
},

# Confidence bounds
"confidence_bounds": {
"min_confidence": 0.0,
"max_confidence": 1.0,
"neutral_confidence": 0.5
}
}
}

def get_integration_config() -> Dict[str, Any]:
    """
Get complete integration configuration.

Returns:
        Dict containing all integration settings
"""
base_config , get_enhanced_prediction_service_config()
ml_profit_config = get_ml_profit_integration_config()
confidence_config , get_enhanced_confidence_config()

# Merge configurations
complete_config = base_config.copy()
complete_config.update(ml_profit_config)
complete_config.update(confidence_config)

    return complete_config