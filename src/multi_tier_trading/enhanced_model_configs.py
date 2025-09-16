"""
Enhanced Model Configurations for Multi-Tier Trading System

This module provides the exact model configurations as specified:
- HMM: CatBoost, Elastic Net (base) + XGBoost (meta-learner)
- Analyst: TCN, CatBoost, LightGBM (base) + Elastic Net (meta-learner)  
- Tactician: XGBoost, Random Forest, CatBoost, Elastic Net (base) + LightGBM (meta-learner)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from src.utils.tprint import tprint


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    is_meta_learner: bool = False


@dataclass
class TierModelConfig:
    """Configuration for all models in a tier."""
    base_models: List[ModelConfig]
    meta_learner: ModelConfig
    n_features: int
    target_threshold: float
    run_interval: str
    base_timeframe: str


class MultiTierModelConfigs:
    """Enhanced model configurations for the multi-tier trading system."""
    
    @staticmethod
    def get_hmm_config() -> TierModelConfig:
        """Get HMM system configuration (1h base, runs every 15 minutes)."""
        base_models = [
            ModelConfig(
                name="catboost",
                model_type="CatBoost",
                hyperparameters={
                    "iterations": 1000,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "random_seed": 42,
                    "verbose": False
                }
            ),
            ModelConfig(
                name="elastic_net",
                model_type="Elastic Net",
                hyperparameters={
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 42,
                    "max_iter": 1000
                }
            )
        ]
        
        meta_learner = ModelConfig(
            name="xgboost",
            model_type="XGBoost",
            hyperparameters={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42,
                "n_jobs": -1
            },
            is_meta_learner=True
        )
        
        return TierModelConfig(
            base_models=base_models,
            meta_learner=meta_learner,
            n_features=100,
            target_threshold=0.5,  # 0.5% price change
            run_interval="15min",
            base_timeframe="1h"
        )
    
    @staticmethod
    def get_analyst_config() -> TierModelConfig:
        """Get Analyst system configuration (5m base, runs every 2 minutes)."""
        base_models = [
            ModelConfig(
                name="tcn",
                model_type="Temporal Convolutions Network",
                hyperparameters={
                    "sequence_length": 20,
                    "n_filters": 64,
                    "kernel_size": 3,
                    "dilation_rates": [1, 2, 4, 8],
                    "dropout": 0.2,
                    "epochs": 100,
                    "batch_size": 32
                }
            ),
            ModelConfig(
                name="catboost",
                model_type="CatBoostRegressor",
                hyperparameters={
                    "iterations": 1000,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "random_seed": 42,
                    "verbose": False
                }
            ),
            ModelConfig(
                name="lightgbm",
                model_type="LGBMRegressor",
                hyperparameters={
                    "n_estimators": 1000,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "num_leaves": 31,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1
                }
            )
        ]
        
        meta_learner = ModelConfig(
            name="elastic_net",
            model_type="Elastic Net",
            hyperparameters={
                "alpha": 0.1,
                "l1_ratio": 0.5,
                "random_state": 42,
                "max_iter": 1000
            },
            is_meta_learner=True
        )
        
        return TierModelConfig(
            base_models=base_models,
            meta_learner=meta_learner,
            n_features=300,  # 300+ features
            target_threshold=0.5,  # 0.5% price change
            run_interval="2min",
            base_timeframe="5m"
        )
    
    @staticmethod
    def get_tactician_config() -> TierModelConfig:
        """Get Tactician system configuration (1m base, runs every 30 seconds)."""
        base_models = [
            ModelConfig(
                name="xgboost",
                model_type="XGBoost",
                hyperparameters={
                    "n_estimators": 1000,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1
                }
            ),
            ModelConfig(
                name="randomforest",
                model_type="RandomForestRegressor",
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                    "n_jobs": -1
                }
            ),
            ModelConfig(
                name="catboost",
                model_type="CatBoostRegressor",
                hyperparameters={
                    "iterations": 1000,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "random_seed": 42,
                    "verbose": False
                }
            ),
            ModelConfig(
                name="elastic_net",
                model_type="Elastic Net",
                hyperparameters={
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 42,
                    "max_iter": 1000
                }
            )
        ]
        
        meta_learner = ModelConfig(
            name="lightgbm",
            model_type="LGBMRegressor",
            hyperparameters={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "num_leaves": 31,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            },
            is_meta_learner=True
        )
        
        return TierModelConfig(
            base_models=base_models,
            meta_learner=meta_learner,
            n_features=50,  # 50+ features
            target_threshold=0.5,  # 0.5% price change
            run_interval="30sec",
            base_timeframe="1m"
        )
    
    @staticmethod
    def get_all_configs() -> Dict[str, TierModelConfig]:
        """Get all tier configurations."""
        return {
            "hmm": MultiTierModelConfigs.get_hmm_config(),
            "analyst": MultiTierModelConfigs.get_analyst_config(),
            "tactician": MultiTierModelConfigs.get_tactician_config()
        }
    
    @staticmethod
    def validate_config(config: TierModelConfig) -> bool:
        """Validate a tier configuration."""
        try:
            # Check base models
            if not config.base_models:
                tprint("❌ No base models configured")
                return False
            
            # Check meta-learner
            if not config.meta_learner:
                tprint("❌ No meta-learner configured")
                return False
            
            # Check feature count
            if config.n_features <= 0:
                tprint("❌ Invalid feature count")
                return False
            
            # Check target threshold
            if config.target_threshold <= 0:
                tprint("❌ Invalid target threshold")
                return False
            
            tprint(f"✅ Configuration validated: {len(config.base_models)} base models + 1 meta-learner")
            return True
            
        except Exception as e:
            tprint(f"❌ Configuration validation failed: {e}")
            return False
    
    @staticmethod
    def print_config_summary() -> None:
        """Print a summary of all configurations."""
        configs = MultiTierModelConfigs.get_all_configs()
        
        tprint("=" * 80)
        tprint("MULTI-TIER TRADING SYSTEM MODEL CONFIGURATIONS")
        tprint("=" * 80)
        
        for tier_name, config in configs.items():
            tprint(f"\n{tier_name.upper()} SYSTEM:")
            tprint(f"  Base Timeframe: {config.base_timeframe}")
            tprint(f"  Run Interval: {config.run_interval}")
            tprint(f"  Features: {config.n_features}")
            tprint(f"  Target Threshold: {config.target_threshold}%")
            
            tprint(f"  Base Models ({len(config.base_models)}):")
            for model in config.base_models:
                tprint(f"    - {model.name}: {model.model_type}")
            
            tprint(f"  Meta-Learner:")
            tprint(f"    - {config.meta_learner.name}: {config.meta_learner.model_type}")
        
        tprint("\n" + "=" * 80)


# Example usage and validation
if __name__ == "__main__":
    # Print configuration summary
    MultiTierModelConfigs.print_config_summary()
    
    # Validate all configurations
    configs = MultiTierModelConfigs.get_all_configs()
    for tier_name, config in configs.items():
        tprint(f"\nValidating {tier_name} configuration...")
        MultiTierModelConfigs.validate_config(config)