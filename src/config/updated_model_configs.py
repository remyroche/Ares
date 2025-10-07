"""
Updated Model Configurations for Analyst and Tactician

This module contains the updated model configurations as requested:
- Analyst models: LGBM, LGBM + PatchTST features, CatBoost with stacker_lgbm_calibrated meta-learner
- Tactician models: LGBM + small GRU embedding, CatBoost, Causal Dilated TCN with stacker_lgbm_calibrated meta-learner
- Updated hyperparameters and feature limits
- Updated fee structure
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Model types for the updated configurations."""
    LGBM = "LGBM"
    LGBM_PATCHTST = "LGBM_PatchTST"
    CATBOOST = "CatBoost"
    LGBM_GRU = "LGBM_GRU"
    CAUSAL_TCN = "Causal_TCN"
    STACKER_LGBM_CALIBRATED = "Stacker_LGBM_Calibrated"


@dataclass
class PatchTSTConfig:
    """PatchTST configuration for analyst models."""
    lookback_hours: int = 16  # 8-24h range, using 16h as middle
    d_model: int = 96  # 64-128 range, using 96 as middle
    heads: int = 3  # 2-4 range, using 3 as middle
    layers: int = 2
    export_dims: int = 10  # 8-12 range, using 10 as middle
    include_confidence: bool = True
    include_oof_predictions: bool = True


@dataclass
class GRUConfig:
    """GRU configuration for tactician models."""
    lookback_hours: int = 3  # 2-4h range, using 3h as middle
    hidden_size: int = 48  # 32-64 range, using 48 as middle
    num_layers: int = 1
    dropout: float = 0.05  # ≤0.1, using 0.05
    pca_dims: int = 10  # 8-12 range, using 10 as middle
    fit_pca_on_train_only: bool = True


@dataclass
class LGBMConfig:
    """LightGBM configuration with updated hyperparameters."""
    # Updated hyperparameters as requested
    max_depth: int = 3  # 3-4 range, using 3
    num_leaves: int = 12  # 8-16 range, using 12
    min_child_samples: int = 800  # 600-1000 range, using 800
    lambda_l2: float = 30.0  # 10-50 range, using 30
    feature_fraction: float = 0.7  # 0.6-0.8 range, using 0.7
    
    # Additional parameters
    learning_rate: float = 0.05
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1


@dataclass
class CatBoostConfig:
    """CatBoost configuration."""
    depth: int = 4
    learning_rate: float = 0.05
    l2_leaf_reg: float = 8.0
    iterations: int = 500
    subsample: float = 0.8
    colsample_bylevel: float = 0.8
    random_seed: int = 42
    verbose: bool = False


@dataclass
class CausalTCNConfig:
    """Causal Dilated TCN configuration for tactician."""
    num_filters: int = 64
    kernel_size: int = 3
    dilation_base: int = 2
    num_layers: int = 4
    dropout: float = 0.1
    activation: str = "relu"
    use_skip_connections: bool = True


@dataclass
class StackerLGBMCalibratedConfig:
    """Stacker LGBM Calibrated meta-learner configuration."""
    base_lgbm_config: LGBMConfig = None
    calibration_method: str = "isotonic"  # or "sigmoid"
    cv_folds: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        if self.base_lgbm_config is None:
            self.base_lgbm_config = LGBMConfig()


# Analyst Models Configuration
ANALYST_MODELS_CONFIG = {
    "base_models": {
        ModelType.LGBM: {
            "config": LGBMConfig(),
            "enabled": True,
            "description": "LightGBM with updated hyperparameters"
        },
        ModelType.LGBM_PATCHTST: {
            "config": {
                "lgbm": LGBMConfig(),
                "patchtst": PatchTSTConfig()
            },
            "enabled": True,
            "description": "LightGBM enhanced with PatchTST features"
        },
        ModelType.CATBOOST: {
            "config": CatBoostConfig(),
            "enabled": True,
            "description": "CatBoost classifier"
        }
    },
    "meta_learner": {
        ModelType.STACKER_LGBM_CALIBRATED: {
            "config": StackerLGBMCalibratedConfig(),
            "enabled": True,
            "description": "Stacker LGBM with calibration"
        }
    }
}

# Tactician Models Configuration
TACTICIAN_MODELS_CONFIG = {
    "base_models": {
        ModelType.LGBM_GRU: {
            "config": {
                "lgbm": LGBMConfig(),
                "gru": GRUConfig()
            },
            "enabled": True,
            "description": "LightGBM with small GRU embedding"
        },
        ModelType.CATBOOST: {
            "config": CatBoostConfig(),
            "enabled": True,
            "description": "CatBoost classifier"
        },
        ModelType.CAUSAL_TCN: {
            "config": CausalTCNConfig(),
            "enabled": True,
            "description": "Causal Dilated TCN"
        }
    },
    "meta_learner": {
        ModelType.STACKER_LGBM_CALIBRATED: {
            "config": StackerLGBMCalibratedConfig(),
            "enabled": True,
            "description": "Stacker LGBM with calibration"
        }
    }
}

# Feature Configuration
FEATURE_CONFIG = {
    "max_features": 60,  # Reduced from previous value
    "feature_fraction_range": (0.6, 0.8),  # For random feature subsampling
    "stability_selection": {
        "n_bootstrap": 75,  # 50-100 range, using 75
        "threshold": 0.6,  # Stability selection threshold
        "random_state": 42
    },
    "cluster_correlation": {
        "max_features_per_cluster": 1,
        "correlation_threshold": 0.8
    }
}

# Trading Configuration
TRADING_CONFIG = {
    "fees": {
        "assumed_fee_rate": 0.001,  # Updated from 0.0008 to 0.001 (0.1%)
        "description": "Updated assumed trading fees"
    }
}

# Complete Configuration
UPDATED_MODEL_CONFIGURATIONS = {
    "analyst": ANALYST_MODELS_CONFIG,
    "tactician": TACTICIAN_MODELS_CONFIG,
    "features": FEATURE_CONFIG,
    "trading": TRADING_CONFIG
}


def get_analyst_config() -> Dict[str, Any]:
    """Get analyst model configuration."""
    return ANALYST_MODELS_CONFIG


def get_tactician_config() -> Dict[str, Any]:
    """Get tactician model configuration."""
    return TACTICIAN_MODELS_CONFIG


def get_feature_config() -> Dict[str, Any]:
    """Get feature configuration."""
    return FEATURE_CONFIG


def get_trading_config() -> Dict[str, Any]:
    """Get trading configuration."""
    return TRADING_CONFIG


def get_complete_config() -> Dict[str, Any]:
    """Get complete updated configuration."""
    return UPDATED_MODEL_CONFIGURATIONS