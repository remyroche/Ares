# src/config/training.py

from typing import Any

from src.config.constants import DEFAULT_LOOKBACK_DAYS
from src.config.environment import get_environment_settings


def get_training_config() -> dict[str, Any]:
    """Get the complete training configuration.

    Returns:
        dict: Complete training configuration

    """
    get_environment_settings()

    return {
        # --- Model Training Configuration ---
        "training_pipeline": {
            "n_splits": 5,  # Number of folds for walk-forward validation
            "test_size": 0.2,  # Test set size for each fold
            "validation_size": 0.2,  # Validation set size for each fold
            "min_train_size": 1000,  # Minimum training samples required
            "max_train_size": 50000,  # Maximum training samples to use
        },
        # --- Model Training Parameters ---
        "MODEL_TRAINING": {
            "regularization": {
                "lightgbm": {"l1_alpha": 0.01, "l2_alpha": 0.1, "dropout_rate": 0.1},
                "tensorflow": {
                    "l1_alpha": 0.001,
                    "l2_alpha": 0.01,
                    "dropout_rate": 0.2,
                },
                "sklearn": {
                    "l1_alpha": 0.01,
                    "l2_alpha": 0.1,
                    "elastic_net_ratio": 0.5,
                },
                "tabnet": {
                    "lambda_sparse": 0.001,
                    "reg_lambda": 0.01,
                    "dropout_rate": 0.15,
                },
            },
            "optimization": {
                "hyperparameter_trials": 500,
                "cross_validation_folds": 5,
                "early_stopping_patience": 20,
                "ensemble_weight_optimization": True,
                "feature_selection_method": "recursive_feature_elimination",
                "model_selection_criteria": "sharpe_ratio",
            },
            "advanced_features": {
                "enable_market_regime_detection": True,
                "enable_volatility_regime_modeling": True,
                "enable_correlation_analysis": True,
                "enable_momentum_analysis": True,
                "enable_liquidity_analysis": True,
            },
        },
        # --- Global Data Configuration ---
        "DATA_CONFIG": {
            "default_lookback_days": DEFAULT_LOOKBACK_DAYS,  # Default lookback period for all timeframes (3 years)
            "exclude_recent_days": 2,  # Exclude the most recent N days from the lookback window
        },
        # --- Enhanced Training Configuration ---
        "ENHANCED_TRAINING": {
            "enable_efficiency_optimizations": True,
            "segment_days": 30,  # Days per segment for large datasets
            "chunk_size": 10000,  # Chunk size for memory-efficient processing
            "enable_feature_caching": True,  # Cache computed features in database
            "memory_threshold": 0.8,  # Memory usage threshold for cleanup (80%)
            "cache_expiry_hours": 24,  # Cache expiry time in hours
            "database_cleanup_threshold_mb": 1000,  # Database size threshold for cleanup
            "enable_checkpointing": True,  # Enable training checkpoint/resume
            "max_segment_size": 50000,  # Maximum rows per segment
            "enable_computational_optimization": True,  # Enable computational optimization strategies
            "enable_validators": True,  # Enable step validators
        },
        # --- Step 6 Feature Engineering Configuration ---
        "step6_feature_engineering": {
            "enable_basic_features": True,
            "enable_advanced_features": True,
            "enable_regime_aware_features": True,
            "enable_technical_indicators": True,
            "enable_statistical_features": True,
            "feature_selection_method": "variance_threshold",
            "correlation_threshold": 0.8,
            "min_variance": 0.01,
            "max_features": 500,
            "output_dir": "data/training",
        },
        # --- Step 7 Enhanced Matrix Operations Configuration ---
        "step7_enhanced_matrix_operations": {
            "enable_gpu_acceleration": False,
            "enable_sparse_optimizations": True,
            "enable_memory_optimization": True,
            "enable_parallel_processing": True,
            "condition_number_threshold": 1e12,
            "min_eigenvalue_threshold": 1e-10,
            "correlation_threshold": 0.8,
            "memory_threshold_gb": 8.0,
            "batch_size": 1000,
            "max_iterations": 1000,
            "tolerance": 1e-6,
            "output_dir": "data/matrix_operations",
        },
        # --- HMM-LM Model Configuration ---
        "HMM_LM": {
            "generalist": {
                "enabled": True,
                "hmm_states": 5,
                "sequence_length": 20,
                "model_type": "hmm_lm_hybrid",
                "timeframes": ["1m", "5m", "15m", "30m"],
                "d_model": 256,
                "nhead": 8,
                "num_layers": 6,
                "dropout_rate": 0.1,
                "learning_rate": 0.0001,
                "batch_size": 32,
                "epochs": 100,
                "early_stopping_patience": 10,
            },
            "specialist_models": {
                "1m": {
                    "architecture": "CNN",
                    "filters": [32, 64, 128],
                    "kernel_size": 3,
                    "sequence_length": 60,
                    "dropout_rate": 0.3,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 50,
                },
                "5m": {
                    "architecture": "TCN",
                    "channels": [64, 128, 256],
                    "kernel_size": 3,
                    "sequence_length": 40,
                    "dropout_rate": 0.2,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 50,
                },
                "15m": {
                    "architecture": "LSTM",
                    "hidden_size": 128,
                    "num_layers": 2,
                    "sequence_length": 30,
                    "dropout_rate": 0.2,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 50,
                },
                "30m": {
                    "architecture": "Transformer",
                    "d_model": 128,
                    "nhead": 4,
                    "num_layers": 4,
                    "sequence_length": 20,
                    "dropout_rate": 0.1,
                    "learning_rate": 0.0005,
                    "batch_size": 32,
                    "epochs": 50,
                },
            },
        },
        # --- Feature Engineering Configuration ---
        "FEATURE_ENGINEERING": {
            "enable_technical_indicators": True,
            "enable_price_features": True,
            "enable_volume_features": True,
            "enable_volatility_features": True,
            "enable_momentum_features": True,
            "enable_trend_features": True,
            "enable_regime_features": True,
            "enable_interaction_features": True,
            "enable_context_features": True,
            "enable_difference_features": True,
            "enable_acceleration_features": True,
            "enable_sr_features": True,
        },
        # --- Validation Configuration ---
        "VALIDATION": {
            "enable_walk_forward_validation": True,
            "enable_monte_carlo_validation": True,
            "enable_ab_testing": True,
            "enable_confidence_calibration": True,
            "enable_final_parameters_optimization": True,
        },
    }


def get_training_pipeline_config() -> dict[str, Any]:
    """Get training pipeline configuration.

    Returns:
        dict: Training pipeline configuration

    """
    training_config = get_training_config()
    return training_config.get("training_pipeline", {})


def get_model_training_config() -> dict[str, Any]:
    """Get model training configuration.

    Returns:
        dict: Model training configuration

    """
    training_config = get_training_config()
    return training_config.get("MODEL_TRAINING", {})


def get_data_config() -> dict[str, Any]:
    """Get data configuration.

    Returns:
        dict: Data configuration

    """
    training_config = get_training_config()
    return training_config.get("DATA_CONFIG", {})


def get_enhanced_training_config() -> dict[str, Any]:
    """Get enhanced training configuration.

    Returns:
        dict: Enhanced training configuration

    """
    training_config = get_training_config()
    return training_config.get("ENHANCED_TRAINING", {})


def get_hmm_lm_config() -> dict[str, Any]:
    """Get HMM-LM model configuration.

    Returns:
        dict: HMM-LM model configuration

    """
    training_config = get_training_config()
    return training_config.get("HMM_LM", {})


def get_feature_engineering_config() -> dict[str, Any]:
    """Get feature engineering configuration.

    Returns:
        dict: Feature engineering configuration

    """
    training_config = get_training_config()
    return training_config.get("FEATURE_ENGINEERING", {})


def get_validation_config() -> dict[str, Any]:
    """Get validation configuration.

    Returns:
        dict: Validation configuration

    """
    training_config = get_training_config()
    return training_config.get("VALIDATION", {})
