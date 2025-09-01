#!/usr/bin/env python3
"""Multi-Output Prediction Configuration.

This module provides configuration settings for enabling intelligent multi-output
prediction for both direction and profit using the triple barrier method and
profit-based feature engineering.
"""

from typing import Dict, Any




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