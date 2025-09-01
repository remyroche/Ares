#!/usr/bin/env python3
"""Multi-Output Prediction Configuration.

This module provides configuration settings for enabling intelligent multi-output
prediction for both direction and profit using the triple barrier method and
profit-based feature engineering.
"""

from typing import Dict, Any


def get_multi_output_config() -> Dict[str, Any]:
    """Get configuration for multi-output prediction features.

    Returns:
        Dictionary containing multi-output configuration settings
    """
    return {
        # Enable multi-output prediction
        "enable_multi_output": True,

        # Multi-output model configuration
        "multi_output_models": {
            "model_type": "LightGBM",  # "LightGBM", "RandomForest", "NeuralNetwork"
            "use_profit_features": True,
            "direction_target": "direction",
            "profit_target": "potential_profit_pct",
            "direction_threshold": 0.0,
            "profit_scaling": "standard",  # "standard", "robust", "minmax"
            "ensemble_method": "stacking",  # "stacking", "voting", "blending"
            "validation_method": "time_series_cv",
            "n_splits": 5,
            "test_size": 0.2,
            "random_state": 42,
        },

        # Profit-based feature engineering configuration
        "profit_feature_engineering": {
            "profit_column": "potential_profit_pct",
            "volume_column": "volume",
            "price_column": "close",
            "use_numba": True,
            "memory_efficient": True,
            "feature_categories": [
                "basic_profit",
                "categorical",
                "risk_reward",
                "momentum",
                "volatility",
                "volume",
                "rolling"
            ],
            "profit_bins": [-float('inf'), -0.005, -0.002, -0.001, 0, 0.001, 0.002, 0.005, float('inf')],
            "profit_labels": [
                "Large Loss", "Medium Loss", "Small Loss", "Tiny Loss",
                "No Profit", "Tiny Profit", "Small Profit", "Large Profit"
            ]
        },

        # Enhanced training configuration
        "enhanced_training": {
            "enable_enhanced_hmm_training": True,
            "enable_regime_specific_models": True,
            "enable_profit_based_features": True,
            "enable_direction_profit_prediction": True,
            "validation_config": {
                "n_splits": 5,
                "test_size": 0.2,
                "validation_size": 0.2,
                "min_samples_per_split": 1000,
                "regime_aware_splitting": True,
            }
        },

        # Enhanced data-driven feature selection configuration
        "feature_reduction": {
            "target_features": 100,
            "vif_threshold": 10.0,
            "mi_threshold": 0.01,
            "correlation_threshold": 0.95,
            "variance_threshold": 0.01,
            "method_weights": {
                "vif": 0.2,
                "mutual_info": 0.25,
                "shap": 0.25,
                "random_forest": 0.2,
                "rfe": 0.1
            },
            "enable_enhanced_selection": True,
            "enable_vif_filtering": True,
            "enable_mi_filtering": True,
            "enable_shap_filtering": True,
            "enable_rf_filtering": True,
            "enable_ensemble_selection": True,
            "enable_final_rfe": True
        },

        # Model trainer configuration
        "model_trainer": {
            "enable_multi_output": True,
            "enable_analyst_models": True,
            "enable_tactician_models": True,
            "model_directory": "models",
            "multi_output_model_directory": "models/multi_output_models",
        },

        # Triple barrier method configuration for multi-output
        "triple_barrier_multi_output": {
            "enable_direction_labeling": True,
            "enable_profit_labeling": True,
            "direction_threshold": 0.0,
            "profit_calculation": "percentage",  # "percentage", "absolute", "log_return"
            "barrier_method": "triple_barrier",  # "triple_barrier", "fixed_horizon", "dynamic"
            "upper_barrier": 0.02,  # 2% profit target
            "lower_barrier": -0.01,  # 1% stop loss
            "time_horizon": 100,  # Maximum bars to hold position
        },

        # Fractional labeling configuration
        "fractional_labeling": {
            "enable_fractional_labels": True,
            "enable_confidence_scoring": True,
            "enable_regime_adaptation": False,
            "component_weights": {
                "distance_weight": 0.4,
                "time_weight": 0.3,
                "volatility_weight": 0.3,
            },
            "confidence_thresholds": {
                "min_confidence": 0.1,
                "max_confidence": 0.95,
            },
            "regime_specific_configs": {
                "trending": {
                    "distance_weight": 0.5,
                    "time_weight": 0.3,
                    "volatility_weight": 0.2,
                    "min_confidence": 0.15,
                },
                "ranging": {
                    "distance_weight": 0.3,
                    "time_weight": 0.4,
                    "volatility_weight": 0.3,
                    "min_confidence": 0.1,
                },
                "volatile": {
                    "distance_weight": 0.2,
                    "time_weight": 0.2,
                    "volatility_weight": 0.6,
                    "min_confidence": 0.2,
                }
            }
        },

        # Performance monitoring
        "performance_monitoring": {
            "enable_metrics_tracking": True,
            "track_direction_accuracy": True,
            "track_profit_prediction": True,
            "track_combined_metrics": True,
            "enable_feature_importance": True,
            "enable_model_comparison": True,
        },

        # Validation and testing
        "validation": {
            "enable_cross_validation": True,
            "enable_walk_forward_validation": True,
            "enable_monte_carlo_validation": True,
            "enable_ab_testing": True,
            "validation_metrics": [
                "direction_accuracy",
                "direction_precision",
                "direction_recall",
                "direction_f1",
                "profit_mse",
                "profit_mae",
                "profit_r2",
                "profit_rmse",
                "combined_correlation",
                "profit_accuracy"
            ]
        },

        # Logging and reporting
        "logging": {
            "enable_detailed_logging": True,
            "log_training_progress": True,
            "log_validation_results": True,
            "log_feature_importance": True,
            "log_model_comparison": True,
            "log_performance_metrics": True,
        }
    }


def get_multi_output_model_config(model_type: str = "LightGBM") -> Dict[str, Any]:
    """Get specific configuration for multi-output model type.

    Args:
        model_type: Type of model ("LightGBM", "RandomForest", "NeuralNetwork")

    Returns:
        Model-specific configuration
    """
    base_config = get_multi_output_config()

    if model_type == "LightGBM":
        model_config = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42,
            "verbose": -1,
            "early_stopping_rounds": 10,
            "eval_metric": "binary_logloss",  # for direction
            "eval_metric_profit": "rmse",  # for profit
        }
    elif model_type == "RandomForest":
        model_config = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
            "criterion": "gini",  # for direction
            "criterion_profit": "mse",  # for profit
        }
    elif model_type == "NeuralNetwork":
        model_config = {
            "hidden_sizes": [128, 64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "early_stopping_patience": 10,
            "loss_direction": "binary_crossentropy",
            "loss_profit": "mse",
        }
    else:
        model_config = {}

    return {**base_config, "model_config": model_config}


def get_enhanced_training_pipeline_config() -> Dict[str, Any]:
    """Get configuration for the enhanced training pipeline with multi-output support.

    Returns:
        Enhanced training pipeline configuration
    """
    multi_output_config = get_multi_output_config()

    return {
        # Pipeline configuration
        "pipeline": {
            "enable_enhanced_steps": True,
            "enable_multi_output_training": True,
            "enable_profit_based_features": True,
            "enable_direction_profit_prediction": True,
        },

        # Step-specific configurations
        "steps": {
            "step04_triple_barrier_method": {
                "enable_multi_output_labeling": True,
                "enable_direction_labeling": True,
                "enable_profit_labeling": True,
            },
            "step05_labeling": {
                "enable_multi_output_targets": True,
                "enable_direction_targets": True,
                "enable_profit_targets": True,
            },
            "step06_feature_engineering": {
                "enable_profit_based_features": True,
                "enable_enhanced_features": True,
            },
            "step08_enhanced_hmm_based_training": {
                "enable_multi_output_models": True,
                "enable_regime_specific_models": True,
                "enable_profit_based_training": True,
            },
        },

        # Include multi-output configuration
        **multi_output_config
    }


def validate_multi_output_config(config: Dict[str, Any]) -> bool:
    """Validate multi-output configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = [
        "enable_multi_output",
        "multi_output_models",
        "profit_feature_engineering",
        "enhanced_training"
    ]

    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required configuration key: {key}")
            return False

    # Validate model type
    model_type = config["multi_output_models"].get("model_type")
    valid_model_types = ["LightGBM", "RandomForest", "NeuralNetwork"]
    if model_type not in valid_model_types:
        print(f"❌ Invalid model type: {model_type}. Must be one of {valid_model_types}")
        return False

    # Validate profit feature engineering
    profit_config = config["profit_feature_engineering"]
    if not profit_config.get("use_numba", True):
        print("⚠️ Numba acceleration is recommended for profit feature engineering")

    print("✅ Multi-output configuration validation passed")
    return True


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = get_multi_output_config()
    print("Multi-output configuration:")
    print(f"  - Enable multi-output: {config['enable_multi_output']}")
    print(f"  - Model type: {config['multi_output_models']['model_type']}")
    print(f"  - Use profit features: {config['multi_output_models']['use_profit_features']}")

    # Validate configuration
    validate_multi_output_config(config)

    # Test model-specific configuration
    lightgbm_config = get_multi_output_model_config("LightGBM")
    print(f"\nLightGBM configuration:")
    print(f"  - N estimators: {lightgbm_config['model_config']['n_estimators']}")
    print(f"  - Learning rate: {lightgbm_config['model_config']['learning_rate']}")

    # Test enhanced pipeline configuration
    pipeline_config = get_enhanced_training_pipeline_config()
    print(f"\nEnhanced pipeline configuration:")
    print(f"  - Enable enhanced steps: {pipeline_config['pipeline']['enable_enhanced_steps']}")
    print(f"  - Enable multi-output training: {pipeline_config['pipeline']['enable_multi_output_training']}")