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
