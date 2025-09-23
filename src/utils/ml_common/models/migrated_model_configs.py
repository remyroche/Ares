"""
Migrated Model Configurations for HMM, Analyst, and Tactician

This module contains the comprehensive model configurations for the migrated ML models
across all three components, with proper regularization, overfitting prevention,
and regime-aware training capabilities.

Key Features:
- HMM models for regime detection on 15m timeframe
- Analyst models for trading opportunities on 5m timeframe  
- Tactician models for entry timing on 1m timeframe
- Comprehensive regularization and overfitting prevention
- Regime-aware training and parameter optimization
- Support for all required model types with proper configurations
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelArchitecture(Enum):
    """Model architecture types."""
    # Tree-based models
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    
    # Neural network models
    FINANCIAL_RESNET = "financial_resnet"
    DEEPSCALER = "deepscaler"
    DEEPSCALER_1M = "deepscaler_1m"
    NBEATS = "nbeats"
    ADVANCED_MAMBA_HYBRID = "advanced_mamba_hybrid"
    
    # Mobile and efficient architectures
    MOBILENET = "mobilenet"
    EFFICIENTNET = "efficientnet"


@dataclass
class RegimeCharacteristics:
    """Regime characteristics for regime-aware training."""
    volume: float
    volatility: float
    momentum: float
    trend: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "volume": self.volume,
            "volatility": self.volatility,
            "momentum": self.momentum,
            "trend": self.trend
        }


@dataclass
class FinancialResNetConfig:
    """Configuration for FinancialResNet architecture."""
    architecture: str = "FinancialResNet"
    blocks: List[int] = field(default_factory=lambda: [32, 64, 128])
    temporal_conv_layers: int = 3
    attention_heads: int = 4
    dropout: float = 0.15
    regime_aware: bool = True
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    l2_regularization: float = 0.01


@dataclass
class DeepScalerConfig:
    """Configuration for DeepScaler architecture."""
    architecture: str = "DeepScaler"
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.2
    batch_norm: bool = True
    activation: str = "relu"
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    l2_regularization: float = 0.01


@dataclass
class NBEATSConfig:
    """Configuration for N-BEATS architecture with regime-aware parameters."""
    architecture: str = "N-BEATS"
    block_type: str = "generic"  # generic, trend, seasonality
    num_blocks: int = 10
    num_layers: int = 4
    layer_widths: List[int] = field(default_factory=lambda: [512, 512, 256, 256])
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    l2_regularization: float = 0.01
    
    # Regime-aware parameters
    regime_aware: bool = True
    regime_parameter_mapping: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "high_volume": {
            "learning_rate_multiplier": 1.2,
            "dropout_multiplier": 0.8,
            "layer_width_multiplier": 1.1
        },
        "high_volatility": {
            "learning_rate_multiplier": 0.8,
            "dropout_multiplier": 1.2,
            "layer_width_multiplier": 0.9
        },
        "high_momentum": {
            "learning_rate_multiplier": 1.1,
            "dropout_multiplier": 0.9,
            "layer_width_multiplier": 1.05
        },
        "strong_trend": {
            "learning_rate_multiplier": 0.9,
            "dropout_multiplier": 1.1,
            "layer_width_multiplier": 0.95
        }
    })


@dataclass
class AdvancedMambaHybridConfig:
    """Configuration for AdvancedMambaHybrid architecture."""
    architecture: str = "AdvancedMambaHybrid"
    mamba_layers: int = 2
    conv_layers: int = 4
    attention_heads: int = 8
    hidden_dim: int = 128
    state_expansion: int = 4
    multi_timeframe_fusion: bool = True
    dropout: float = 0.1
    activation: str = "GELU"
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    l2_regularization: float = 0.01
    
    # Tactician-specific optimizations
    execution_optimization: bool = False
    micro_timing_attention: bool = False
    latency_aware: bool = False


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str
    architecture: ModelArchitecture
    timeframe: str
    role: str  # "regime_detection", "trading_opportunities", "entry_timing"
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    
    # Regularization
    dropout: float = 0.1
    l2_regularization: float = 0.01
    l1_regularization: float = 0.0
    
    # Overfitting prevention
    enable_early_stopping: bool = True
    enable_dropout: bool = True
    enable_batch_norm: bool = True
    enable_data_augmentation: bool = True
    
    # Validation
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    enable_purged_cv: bool = True
    
    # Regime awareness
    regime_aware: bool = False
    regime_characteristics: Optional[RegimeCharacteristics] = None
    
    # Model-specific config
    model_specific_config: Optional[Dict[str, Any]] = None


class MigratedModelConfigs:
    """Comprehensive model configurations for migrated ML models."""
    
    @staticmethod
    def get_hmm_models() -> Dict[str, ModelConfig]:
        """Get HMM model configurations for regime detection on 15m timeframe."""
        return {
            "lgbm": ModelConfig(
                name="LightGBM_HMM",
                architecture=ModelArchitecture.LIGHTGBM,
                timeframe="15m",
                role="regime_detection",
                model_specific_config={
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "num_leaves": 31,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_samples": 20,
                    "early_stopping_rounds": 50,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": -1
                }
            ),
            "xgboost": ModelConfig(
                name="XGBoost_HMM",
                architecture=ModelArchitecture.XGBOOST,
                timeframe="15m",
                role="regime_detection",
                model_specific_config={
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "min_child_weight": 3,
                    "early_stopping_rounds": 50,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": 0
                }
            ),
            "financial_resnet": ModelConfig(
                name="FinancialResNet_HMM",
                architecture=ModelArchitecture.FINANCIAL_RESNET,
                timeframe="15m",
                role="regime_detection",
                regime_aware=True,
                model_specific_config=FinancialResNetConfig(
                    blocks=[32, 64, 128],
                    temporal_conv_layers=3,
                    attention_heads=4,
                    dropout=0.15,
                    regime_aware=True
                ).__dict__
            )
        }
    
    @staticmethod
    def get_analyst_models() -> Dict[str, ModelConfig]:
        """Get Analyst model configurations for trading opportunities on 5m timeframe."""
        return {
            "deepscaler": ModelConfig(
                name="DeepScaler_Analyst",
                architecture=ModelArchitecture.DEEPSCALER,
                timeframe="5m",
                role="trading_opportunities",
                model_specific_config=DeepScalerConfig(
                    hidden_layers=[512, 256, 128],
                    dropout=0.2,
                    batch_norm=True,
                    activation="relu"
                ).__dict__
            ),
            "catboost": ModelConfig(
                name="CatBoost_Analyst",
                architecture=ModelArchitecture.CATBOOST,
                timeframe="5m",
                role="trading_opportunities",
                model_specific_config={
                    "iterations": 1000,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "l2_leaf_reg": 3.0,
                    "bagging_temperature": 1.0,
                    "subsample": 0.8,
                    "colsample_bylevel": 0.8,
                    "early_stopping_rounds": 50,
                    "random_seed": 42,
                    "verbose": False
                }
            ),
            "xgboost": ModelConfig(
                name="XGBoost_Analyst",
                architecture=ModelArchitecture.XGBOOST,
                timeframe="5m",
                role="trading_opportunities",
                model_specific_config={
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "min_child_weight": 3,
                    "early_stopping_rounds": 50,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": 0
                }
            ),
            "nbeats": ModelConfig(
                name="NBEATS_Analyst",
                architecture=ModelArchitecture.NBEATS,
                timeframe="5m",
                role="trading_opportunities",
                regime_aware=True,
                model_specific_config=NBEATSConfig(
                    num_blocks=10,
                    num_layers=4,
                    layer_widths=[512, 512, 256, 256],
                    dropout=0.1,
                    regime_aware=True
                ).__dict__
            ),
            "advanced_mamba_hybrid": ModelConfig(
                name="AdvancedMambaHybrid_Analyst",
                architecture=ModelArchitecture.ADVANCED_MAMBA_HYBRID,
                timeframe="5m",
                role="trading_opportunities",
                model_specific_config=AdvancedMambaHybridConfig(
                    mamba_layers=2,
                    conv_layers=4,
                    attention_heads=8,
                    hidden_dim=128,
                    state_expansion=4,
                    multi_timeframe_fusion=True,
                    dropout=0.1,
                    activation="GELU"
                ).__dict__
            )
        }
    
    @staticmethod
    def get_tactician_models() -> Dict[str, ModelConfig]:
        """Get Tactician model configurations for entry timing on 1m timeframe."""
        return {
            "xgboost": ModelConfig(
                name="XGBoost_Tactician",
                architecture=ModelArchitecture.XGBOOST,
                timeframe="1m",
                role="entry_timing",
                model_specific_config={
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "min_child_weight": 3,
                    "early_stopping_rounds": 50,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": 0
                }
            ),
            "lightgbm": ModelConfig(
                name="LightGBM_Tactician",
                architecture=ModelArchitecture.LIGHTGBM,
                timeframe="1m",
                role="entry_timing",
                model_specific_config={
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "num_leaves": 31,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_samples": 20,
                    "early_stopping_rounds": 50,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": -1
                }
            ),
            "deepscaler_1m": ModelConfig(
                name="DeepScaler1m_Tactician",
                architecture=ModelArchitecture.DEEPSCALER_1M,
                timeframe="1m",
                role="entry_timing",
                model_specific_config=DeepScalerConfig(
                    hidden_layers=[256, 128, 64],
                    dropout=0.1,
                    batch_norm=True,
                    activation="relu",
                    batch_size=16,
                    epochs=50
                ).__dict__
            ),
            "financial_resnet": ModelConfig(
                name="FinancialResNet_Tactician",
                architecture=ModelArchitecture.FINANCIAL_RESNET,
                timeframe="1m",
                role="entry_timing",
                model_specific_config=FinancialResNetConfig(
                    blocks=[16, 32, 64],
                    temporal_conv_layers=2,
                    attention_heads=2,
                    dropout=0.1,
                    regime_aware=True,
                    batch_size=16,
                    epochs=50
                ).__dict__
            ),
            "advanced_mamba_hybrid": ModelConfig(
                name="AdvancedMambaHybrid_Tactician",
                architecture=ModelArchitecture.ADVANCED_MAMBA_HYBRID,
                timeframe="1m",
                role="entry_timing",
                model_specific_config=AdvancedMambaHybridConfig(
                    mamba_layers=3,
                    conv_layers=4,
                    attention_heads=6,
                    hidden_dim=128,
                    state_expansion=5,
                    execution_optimization=True,
                    micro_timing_attention=True,
                    latency_aware=True,
                    dropout=0.1,
                    activation="GELU",
                    batch_size=16,
                    epochs=50
                ).__dict__
            )
        }
    
    @staticmethod
    def get_all_models() -> Dict[str, Dict[str, ModelConfig]]:
        """Get all model configurations."""
        return {
            "hmm_models": MigratedModelConfigs.get_hmm_models(),
            "analyst_models": MigratedModelConfigs.get_analyst_models(),
            "tactician_models": MigratedModelConfigs.get_tactician_models()
        }
    
    @staticmethod
    def get_regime_aware_models() -> List[str]:
        """Get list of regime-aware model names."""
        all_models = MigratedModelConfigs.get_all_models()
        regime_aware_models = []
        
        for component, models in all_models.items():
            for model_name, config in models.items():
                if config.regime_aware:
                    regime_aware_models.append(config.name)
        
        return regime_aware_models
    
    @staticmethod
    def get_model_by_name(model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        all_models = MigratedModelConfigs.get_all_models()
        
        for component, models in all_models.items():
            for model_key, config in models.items():
                if config.name == model_name:
                    return config
        
        return None
    
    @staticmethod
    def get_models_by_timeframe(timeframe: str) -> List[ModelConfig]:
        """Get all models for a specific timeframe."""
        all_models = MigratedModelConfigs.get_all_models()
        timeframe_models = []
        
        for component, models in all_models.items():
            for model_key, config in models.items():
                if config.timeframe == timeframe:
                    timeframe_models.append(config)
        
        return timeframe_models
    
    @staticmethod
    def get_models_by_role(role: str) -> List[ModelConfig]:
        """Get all models for a specific role."""
        all_models = MigratedModelConfigs.get_all_models()
        role_models = []
        
        for component, models in all_models.items():
            for model_key, config in models.items():
                if config.role == role:
                    role_models.append(config)
        
        return role_models


class RegimeAwareParameterOptimizer:
    """Optimizes model parameters based on regime characteristics."""
    
    @staticmethod
    def optimize_nbeats_parameters(config: NBEATSConfig, regime: RegimeCharacteristics) -> NBEATSConfig:
        """Optimize N-BEATS parameters based on regime characteristics."""
        optimized_config = config
        
        # Determine regime type
        is_high_volume = regime.volume > 0.7
        is_high_volatility = regime.volatility > 0.7
        is_high_momentum = regime.momentum > 0.7
        is_strong_trend = regime.trend > 0.7
        
        # Apply regime-specific parameter adjustments
        if is_high_volume:
            mapping = config.regime_parameter_mapping["high_volume"]
            optimized_config.learning_rate *= mapping["learning_rate_multiplier"]
            optimized_config.dropout *= mapping["dropout_multiplier"]
            optimized_config.layer_widths = [int(w * mapping["layer_width_multiplier"]) for w in config.layer_widths]
        
        if is_high_volatility:
            mapping = config.regime_parameter_mapping["high_volatility"]
            optimized_config.learning_rate *= mapping["learning_rate_multiplier"]
            optimized_config.dropout *= mapping["dropout_multiplier"]
            optimized_config.layer_widths = [int(w * mapping["layer_width_multiplier"]) for w in config.layer_widths]
        
        if is_high_momentum:
            mapping = config.regime_parameter_mapping["high_momentum"]
            optimized_config.learning_rate *= mapping["learning_rate_multiplier"]
            optimized_config.dropout *= mapping["dropout_multiplier"]
            optimized_config.layer_widths = [int(w * mapping["layer_width_multiplier"]) for w in config.layer_widths]
        
        if is_strong_trend:
            mapping = config.regime_parameter_mapping["strong_trend"]
            optimized_config.learning_rate *= mapping["learning_rate_multiplier"]
            optimized_config.dropout *= mapping["dropout_multiplier"]
            optimized_config.layer_widths = [int(w * mapping["layer_width_multiplier"]) for w in config.layer_widths]
        
        return optimized_config
    
    @staticmethod
    def optimize_financial_resnet_parameters(config: FinancialResNetConfig, regime: RegimeCharacteristics) -> FinancialResNetConfig:
        """Optimize FinancialResNet parameters based on regime characteristics."""
        optimized_config = config
        
        # Adjust attention heads based on volatility
        if regime.volatility > 0.7:
            optimized_config.attention_heads = min(config.attention_heads + 2, 8)
        elif regime.volatility < 0.3:
            optimized_config.attention_heads = max(config.attention_heads - 1, 2)
        
        # Adjust dropout based on momentum
        if regime.momentum > 0.7:
            optimized_config.dropout = max(config.dropout - 0.05, 0.05)
        elif regime.momentum < 0.3:
            optimized_config.dropout = min(config.dropout + 0.05, 0.3)
        
        # Adjust temporal conv layers based on trend
        if regime.trend > 0.7:
            optimized_config.temporal_conv_layers = min(config.temporal_conv_layers + 1, 5)
        elif regime.trend < 0.3:
            optimized_config.temporal_conv_layers = max(config.temporal_conv_layers - 1, 1)
        
        return optimized_config