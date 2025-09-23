"""
Enhanced Migrated Model Factory

This factory creates and manages all migrated ML models for HMM, Analyst, and Tactician
components with comprehensive support for:
- All required model architectures
- Regime-aware training and parameter optimization
- Comprehensive regularization and overfitting prevention
- Integration with existing ML training pipeline
- Support for validation and HPO
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import warnings
from dataclasses import dataclass

# Enhanced dependency management
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0] if args else ''}")
    def tprint_success(*args, **kwargs): print(f"SUCCESS: {args[0] if args else ''}")

# Import model configurations and implementations
try:
    from .migrated_model_configs import (
        MigratedModelConfigs, ModelConfig, ModelArchitecture,
        RegimeCharacteristics, RegimeAwareParameterOptimizer
    )
    from .advanced_model_implementations import (
        FinancialResNet, DeepScaler, NBEATS, AdvancedMambaHybrid,
        MobileNet, EfficientNet, ModelWrapper
    )
    MIGRATED_MODELS_AVAILABLE = True
except ImportError:
    MIGRATED_MODELS_AVAILABLE = False
    tprint_warning("⚠️ Migrated model configurations not available")

# Import existing model factory
try:
    from .model_factory import EnhancedModelFactory, ModelType, ModelConfig as BaseModelConfig
    EXISTING_FACTORY_AVAILABLE = True
except ImportError:
    EXISTING_FACTORY_AVAILABLE = False
    tprint_warning("⚠️ Existing model factory not available")

# Import common utilities
try:
    from src.utils.common_operations import safe_json_dump, safe_json_load
    from src.utils.math_validation import validate_finite, validate_positive
    COMMON_UTILS_AVAILABLE = True
except ImportError:
    COMMON_UTILS_AVAILABLE = False
    tprint_warning("⚠️ Common utilities not available")

logger = logging.getLogger(__name__)


class EnhancedMigratedModelFactory:
    """Enhanced factory for creating migrated ML models with comprehensive support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced migrated model factory."""
        self.logger = logger.getChild('EnhancedMigratedModelFactory')
        self.logger.info("🚀 Initializing Enhanced Migrated Model Factory...")
        start_time = time.time()
        
        self.config = config or {}
        
        # Initialize base factory if available
        self.base_factory = None
        if EXISTING_FACTORY_AVAILABLE:
            self.base_factory = EnhancedModelFactory(config)
            self.logger.info("✅ Base model factory initialized")
        
        # Model registry for created models
        self.model_registry: Dict[str, Any] = {}
        
        # Regime characteristics cache
        self.regime_cache: Dict[str, RegimeCharacteristics] = {}
        
        # Initialize dependencies
        self.dependencies = self._check_dependencies()
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ Enhanced Migrated Model Factory initialized in {init_time:.3f}s")
        self.logger.info(f"📊 Available dependencies: {list(self.dependencies.keys())}")
        self.logger.info(f"🧠 Migrated models available: {MIGRATED_MODELS_AVAILABLE}")
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Check availability of required dependencies."""
        dependencies = {}
        
        # PyTorch
        try:
            import torch
            dependencies['torch'] = True
            self.logger.debug("✅ PyTorch available")
        except ImportError:
            dependencies['torch'] = False
            self.logger.warning("⚠️ PyTorch not available")
        
        # Scikit-learn
        try:
            import sklearn
            dependencies['sklearn'] = True
            self.logger.debug("✅ Scikit-learn available")
        except ImportError:
            dependencies['sklearn'] = False
            self.logger.warning("⚠️ Scikit-learn not available")
        
        # LightGBM
        try:
            import lightgbm
            dependencies['lightgbm'] = True
            self.logger.debug("✅ LightGBM available")
        except ImportError:
            dependencies['lightgbm'] = False
            self.logger.warning("⚠️ LightGBM not available")
        
        # XGBoost
        try:
            import xgboost
            dependencies['xgboost'] = True
            self.logger.debug("✅ XGBoost available")
        except ImportError:
            dependencies['xgboost'] = False
            self.logger.warning("⚠️ XGBoost not available")
        
        # CatBoost
        try:
            import catboost
            dependencies['catboost'] = True
            self.logger.debug("✅ CatBoost available")
        except ImportError:
            dependencies['catboost'] = False
            self.logger.warning("⚠️ CatBoost not available")
        
        return dependencies
    
    def create_hmm_model(self, model_name: str, input_dim: int, output_dim: int,
                        regime_characteristics: Optional[RegimeCharacteristics] = None) -> Any:
        """Create HMM model for regime detection on 15m timeframe."""
        self.logger.info(f"🔄 Creating HMM model: {model_name}")
        
        if not MIGRATED_MODELS_AVAILABLE:
            raise ImportError("Migrated model configurations not available")
        
        # Get model configuration
        hmm_models = MigratedModelConfigs.get_hmm_models()
        if model_name not in hmm_models:
            raise ValueError(f"Unknown HMM model: {model_name}")
        
        config = hmm_models[model_name]
        
        # Create model based on architecture
        if config.architecture == ModelArchitecture.LIGHTGBM:
            model = self._create_lightgbm_model(config, input_dim, output_dim)
        elif config.architecture == ModelArchitecture.XGBOOST:
            model = self._create_xgboost_model(config, input_dim, output_dim)
        elif config.architecture == ModelArchitecture.FINANCIAL_RESNET:
            model = self._create_financial_resnet_model(config, input_dim, output_dim, regime_characteristics)
        else:
            raise ValueError(f"Unsupported HMM model architecture: {config.architecture}")
        
        # Register model
        self.model_registry[f"hmm_{model_name}"] = model
        
        self.logger.info(f"✅ HMM model {model_name} created successfully")
        return model
    
    def create_analyst_model(self, model_name: str, input_dim: int, output_dim: int,
                           regime_characteristics: Optional[RegimeCharacteristics] = None) -> Any:
        """Create Analyst model for trading opportunities on 5m timeframe."""
        self.logger.info(f"🔄 Creating Analyst model: {model_name}")
        
        if not MIGRATED_MODELS_AVAILABLE:
            raise ImportError("Migrated model configurations not available")
        
        # Get model configuration
        analyst_models = MigratedModelConfigs.get_analyst_models()
        if model_name not in analyst_models:
            raise ValueError(f"Unknown Analyst model: {model_name}")
        
        config = analyst_models[model_name]
        
        # Create model based on architecture
        if config.architecture == ModelArchitecture.DEEPSCALER:
            model = self._create_deepscaler_model(config, input_dim, output_dim)
        elif config.architecture == ModelArchitecture.CATBOOST:
            model = self._create_catboost_model(config, input_dim, output_dim)
        elif config.architecture == ModelArchitecture.XGBOOST:
            model = self._create_xgboost_model(config, input_dim, output_dim)
        elif config.architecture == ModelArchitecture.NBEATS:
            model = self._create_nbeats_model(config, input_dim, output_dim, regime_characteristics)
        elif config.architecture == ModelArchitecture.ADVANCED_MAMBA_HYBRID:
            model = self._create_advanced_mamba_hybrid_model(config, input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported Analyst model architecture: {config.architecture}")
        
        # Register model
        self.model_registry[f"analyst_{model_name}"] = model
        
        self.logger.info(f"✅ Analyst model {model_name} created successfully")
        return model
    
    def create_tactician_model(self, model_name: str, input_dim: int, output_dim: int,
                             regime_characteristics: Optional[RegimeCharacteristics] = None) -> Any:
        """Create Tactician model for entry timing on 1m timeframe."""
        self.logger.info(f"🔄 Creating Tactician model: {model_name}")
        
        if not MIGRATED_MODELS_AVAILABLE:
            raise ImportError("Migrated model configurations not available")
        
        # Get model configuration
        tactician_models = MigratedModelConfigs.get_tactician_models()
        if model_name not in tactician_models:
            raise ValueError(f"Unknown Tactician model: {model_name}")
        
        config = tactician_models[model_name]
        
        # Create model based on architecture
        if config.architecture == ModelArchitecture.XGBOOST:
            model = self._create_xgboost_model(config, input_dim, output_dim)
        elif config.architecture == ModelArchitecture.LIGHTGBM:
            model = self._create_lightgbm_model(config, input_dim, output_dim)
        elif config.architecture == ModelArchitecture.DEEPSCALER_1M:
            model = self._create_deepscaler_1m_model(config, input_dim, output_dim)
        elif config.architecture == ModelArchitecture.FINANCIAL_RESNET:
            model = self._create_financial_resnet_model(config, input_dim, output_dim, regime_characteristics)
        elif config.architecture == ModelArchitecture.ADVANCED_MAMBA_HYBRID:
            model = self._create_advanced_mamba_hybrid_model(config, input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported Tactician model architecture: {config.architecture}")
        
        # Register model
        self.model_registry[f"tactician_{model_name}"] = model
        
        self.logger.info(f"✅ Tactician model {model_name} created successfully")
        return model
    
    def _create_lightgbm_model(self, config: ModelConfig, input_dim: int, output_dim: int) -> Any:
        """Create LightGBM model."""
        if not self.dependencies.get('lightgbm', False):
            raise ImportError("LightGBM not available")
        
        import lightgbm as lgb
        
        # Get model-specific parameters
        params = config.model_specific_config or {}
        
        # Create model based on role
        if config.role == "regime_detection":
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        
        return model
    
    def _create_xgboost_model(self, config: ModelConfig, input_dim: int, output_dim: int) -> Any:
        """Create XGBoost model."""
        if not self.dependencies.get('xgboost', False):
            raise ImportError("XGBoost not available")
        
        import xgboost as xgb
        
        # Get model-specific parameters
        params = config.model_specific_config or {}
        
        # Create model based on role
        if config.role == "regime_detection":
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)
        
        return model
    
    def _create_catboost_model(self, config: ModelConfig, input_dim: int, output_dim: int) -> Any:
        """Create CatBoost model."""
        if not self.dependencies.get('catboost', False):
            raise ImportError("CatBoost not available")
        
        from catboost import CatBoostRegressor, CatBoostClassifier
        
        # Get model-specific parameters
        params = config.model_specific_config or {}
        
        # Create model based on role
        if config.role == "regime_detection":
            model = CatBoostClassifier(**params)
        else:
            model = CatBoostRegressor(**params)
        
        return model
    
    def _create_financial_resnet_model(self, config: ModelConfig, input_dim: int, output_dim: int,
                                     regime_characteristics: Optional[RegimeCharacteristics] = None) -> Any:
        """Create FinancialResNet model."""
        if not self.dependencies.get('torch', False):
            raise ImportError("PyTorch not available")
        
        # Get model-specific configuration
        model_config = config.model_specific_config or {}
        
        # Optimize parameters based on regime characteristics
        if regime_characteristics and config.regime_aware:
            from .migrated_model_configs import FinancialResNetConfig
            resnet_config = FinancialResNetConfig(**model_config)
            optimized_config = RegimeAwareParameterOptimizer.optimize_financial_resnet_parameters(
                resnet_config, regime_characteristics
            )
            model_config = optimized_config.__dict__
        
        # Create model wrapper
        model = ModelWrapper(
            model_class=FinancialResNet,
            model_config=model_config,
            input_dim=input_dim,
            output_dim=output_dim,
            device=self.config.get('device', 'cpu')
        )
        
        return model
    
    def _create_deepscaler_model(self, config: ModelConfig, input_dim: int, output_dim: int) -> Any:
        """Create DeepScaler model."""
        if not self.dependencies.get('torch', False):
            raise ImportError("PyTorch not available")
        
        # Get model-specific configuration
        model_config = config.model_specific_config or {}
        
        # Create model wrapper
        model = ModelWrapper(
            model_class=DeepScaler,
            model_config=model_config,
            input_dim=input_dim,
            output_dim=output_dim,
            device=self.config.get('device', 'cpu')
        )
        
        return model
    
    def _create_deepscaler_1m_model(self, config: ModelConfig, input_dim: int, output_dim: int) -> Any:
        """Create DeepScaler1m model (optimized for 1m timeframe)."""
        if not self.dependencies.get('torch', False):
            raise ImportError("PyTorch not available")
        
        # Get model-specific configuration
        model_config = config.model_specific_config or {}
        
        # Create model wrapper
        model = ModelWrapper(
            model_class=DeepScaler,
            model_config=model_config,
            input_dim=input_dim,
            output_dim=output_dim,
            device=self.config.get('device', 'cpu')
        )
        
        return model
    
    def _create_nbeats_model(self, config: ModelConfig, input_dim: int, output_dim: int,
                           regime_characteristics: Optional[RegimeCharacteristics] = None) -> Any:
        """Create N-BEATS model with regime-aware parameter optimization."""
        if not self.dependencies.get('torch', False):
            raise ImportError("PyTorch not available")
        
        # Get model-specific configuration
        model_config = config.model_specific_config or {}
        
        # Optimize parameters based on regime characteristics
        if regime_characteristics and config.regime_aware:
            from .migrated_model_configs import NBEATSConfig
            nbeats_config = NBEATSConfig(**model_config)
            optimized_config = RegimeAwareParameterOptimizer.optimize_nbeats_parameters(
                nbeats_config, regime_characteristics
            )
            model_config = optimized_config.__dict__
        
        # Create model wrapper
        model = ModelWrapper(
            model_class=NBEATS,
            model_config=model_config,
            input_dim=input_dim,
            output_dim=output_dim,
            device=self.config.get('device', 'cpu')
        )
        
        return model
    
    def _create_advanced_mamba_hybrid_model(self, config: ModelConfig, input_dim: int, output_dim: int) -> Any:
        """Create AdvancedMambaHybrid model."""
        if not self.dependencies.get('torch', False):
            raise ImportError("PyTorch not available")
        
        # Get model-specific configuration
        model_config = config.model_specific_config or {}
        
        # Create model wrapper
        model = ModelWrapper(
            model_class=AdvancedMambaHybrid,
            model_config=model_config,
            input_dim=input_dim,
            output_dim=output_dim,
            device=self.config.get('device', 'cpu')
        )
        
        return model
    
    def create_all_hmm_models(self, input_dim: int, output_dim: int,
                            regime_characteristics: Optional[RegimeCharacteristics] = None) -> Dict[str, Any]:
        """Create all HMM models."""
        self.logger.info("🔄 Creating all HMM models...")
        
        models = {}
        hmm_model_names = ["lgbm", "xgboost", "financial_resnet"]
        
        for model_name in hmm_model_names:
            try:
                model = self.create_hmm_model(model_name, input_dim, output_dim, regime_characteristics)
                models[model_name] = model
                self.logger.info(f"✅ Created HMM model: {model_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to create HMM model {model_name}: {e}")
        
        self.logger.info(f"✅ Created {len(models)} HMM models")
        return models
    
    def create_all_analyst_models(self, input_dim: int, output_dim: int,
                                regime_characteristics: Optional[RegimeCharacteristics] = None) -> Dict[str, Any]:
        """Create all Analyst models."""
        self.logger.info("🔄 Creating all Analyst models...")
        
        models = {}
        analyst_model_names = ["deepscaler", "catboost", "xgboost", "nbeats", "advanced_mamba_hybrid"]
        
        for model_name in analyst_model_names:
            try:
                model = self.create_analyst_model(model_name, input_dim, output_dim, regime_characteristics)
                models[model_name] = model
                self.logger.info(f"✅ Created Analyst model: {model_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to create Analyst model {model_name}: {e}")
        
        self.logger.info(f"✅ Created {len(models)} Analyst models")
        return models
    
    def create_all_tactician_models(self, input_dim: int, output_dim: int,
                                  regime_characteristics: Optional[RegimeCharacteristics] = None) -> Dict[str, Any]:
        """Create all Tactician models."""
        self.logger.info("🔄 Creating all Tactician models...")
        
        models = {}
        tactician_model_names = ["xgboost", "lightgbm", "deepscaler_1m", "financial_resnet", "advanced_mamba_hybrid"]
        
        for model_name in tactician_model_names:
            try:
                model = self.create_tactician_model(model_name, input_dim, output_dim, regime_characteristics)
                models[model_name] = model
                self.logger.info(f"✅ Created Tactician model: {model_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to create Tactician model {model_name}: {e}")
        
        self.logger.info(f"✅ Created {len(models)} Tactician models")
        return models
    
    def create_all_models(self, input_dim: int, output_dim: int,
                         regime_characteristics: Optional[RegimeCharacteristics] = None) -> Dict[str, Dict[str, Any]]:
        """Create all models for all components."""
        self.logger.info("🔄 Creating all migrated models...")
        
        all_models = {
            "hmm_models": self.create_all_hmm_models(input_dim, output_dim, regime_characteristics),
            "analyst_models": self.create_all_analyst_models(input_dim, output_dim, regime_characteristics),
            "tactician_models": self.create_all_tactician_models(input_dim, output_dim, regime_characteristics)
        }
        
        total_models = sum(len(models) for models in all_models.values())
        self.logger.info(f"✅ Created {total_models} total migrated models")
        
        return all_models
    
    def get_model(self, component: str, model_name: str) -> Optional[Any]:
        """Get a model from the registry."""
        registry_key = f"{component}_{model_name}"
        return self.model_registry.get(registry_key)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.model_registry.keys())
    
    def remove_model(self, component: str, model_name: str) -> bool:
        """Remove a model from the registry."""
        registry_key = f"{component}_{model_name}"
        if registry_key in self.model_registry:
            del self.model_registry[registry_key]
            self.logger.info(f"🗑️ Removed model: {registry_key}")
            return True
        return False
    
    def clear_registry(self) -> None:
        """Clear all models from the registry."""
        self.model_registry.clear()
        self.logger.info("🗑️ Cleared model registry")
    
    def get_model_info(self, component: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        if not MIGRATED_MODELS_AVAILABLE:
            return None
        
        # Get model configuration
        if component == "hmm":
            config = MigratedModelConfigs.get_hmm_models().get(model_name)
        elif component == "analyst":
            config = MigratedModelConfigs.get_analyst_models().get(model_name)
        elif component == "tactician":
            config = MigratedModelConfigs.get_tactician_models().get(model_name)
        else:
            return None
        
        if config is None:
            return None
        
        return {
            "name": config.name,
            "architecture": config.architecture.value,
            "timeframe": config.timeframe,
            "role": config.role,
            "regime_aware": config.regime_aware,
            "model_specific_config": config.model_specific_config
        }


def create_migrated_model_factory(config: Optional[Dict[str, Any]] = None) -> EnhancedMigratedModelFactory:
    """Create an enhanced migrated model factory instance."""
    return EnhancedMigratedModelFactory(config)