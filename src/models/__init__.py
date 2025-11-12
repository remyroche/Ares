"""
Model Registry and Factory for Ares Trading System

This module provides a centralized registry and factory for all machine learning models
used in Ares trading system. It supports various model types including CatBoost,
TabR, and provides utilities for model creation and management.
"""

from typing import Dict, Any, Type, Optional, List
from enum import Enum
import logging
import importlib

# Import model classes
from .catboost_regressor import CatBoostRegressor
from .tcn_regressor import TabRRegressor, DepthwiseSeparableCNNRegressor  # TabR replaces DepthWiseCNN

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('ModelRegistry')

class ModelType(Enum):
    """Enumeration of supported model types."""
    CATBOOST = "catboost"
    TABR = "tabr"
    # Legacy aliases for backward compatibility
    DEPTWISE_SEPARABLE_CNN = "depthwise_separable_cnn"  # Maps to TabR
    TCN = "tcn"  # Maps to TabR

    @classmethod
    def from_string(cls, model_type_str: str) -> 'ModelType':
        """Convert string to ModelType enum with backward compatibility."""
        model_type_str = model_type_str.lower().strip()
        
        # Handle legacy names
        if model_type_str in ["depthwise_separable_cnn", "depthwisecnn", "dwcnn"]:
            tprint_warning("⚠️ DepthwiseSeparableCNN is deprecated, using TabR instead")
            return cls.TABR
        elif model_type_str in ["tcn", "temporal_convolutional_network"]:
            tprint_warning("⚠️ TCN is deprecated, using TabR instead")
            return cls.TABR
        
        # Standard mapping
        for model_type in cls:
            if model_type.value == model_type_str:
                return model_type
        
        raise ValueError(f"Unknown model type: {model_type_str}")

    def get_actual_model_type(self) -> 'ModelType':
        """Get actual model type (resolves aliases)."""
        if self in [ModelType.DEPTWISE_SEPARABLE_CNN, ModelType.TCN]:
            return ModelType.TABR
        return self

class ModelRegistry:
    """Registry for model classes and factory functions."""
    
    _models: Dict[ModelType, Type] = {
        ModelType.CATBOOST: CatBoostRegressor,
        ModelType.TABR: TabRRegressor,
        # Legacy mappings
        ModelType.DEPTWISE_SEPARABLE_CNN: TabRRegressor,
        ModelType.TCN: TabRRegressor,
    }
    
    _factory_functions: Dict[ModelType, callable] = {
        ModelType.CATBOOST: None,  # Use default constructor
        ModelType.TABR: None,  # Use default constructor
        # Legacy mappings
        ModelType.DEPTWISE_SEPARABLE_CNN: None,
        ModelType.TCN: None,
    }

    @classmethod
    def register_model(cls, model_type: ModelType, model_class: Type, factory_func: callable = None):
        """Register a new model type."""
        cls._models[model_type] = model_class
        cls._factory_functions[model_type] = factory_func
        logger.info(f"Registered model type: {model_type.value}")

    @classmethod
    def get_model_class(cls, model_type: ModelType) -> Type:
        """Get model class for given type."""
        actual_type = model_type.get_actual_model_type()
        if actual_type not in cls._models:
            raise ValueError(f"Model type {actual_type.value} not registered")
        return cls._models[actual_type]

    @classmethod
    def get_factory_function(cls, model_type: ModelType) -> Optional[callable]:
        """Get factory function for given type."""
        actual_type = model_type.get_actual_model_type()
        return cls._factory_functions.get(actual_type)

    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model types."""
        return [model_type.value for model_type in cls._models.keys()]

class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(model_type: ModelType, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            config: Configuration parameters for model
            
        Returns:
            Model instance
        """
        # Resolve aliases
        actual_type = model_type.get_actual_model_type()
        
        # Show deprecation warnings for legacy models
        if model_type != actual_type:
            tprint_warning(f"⚠️ {model_type.value} is deprecated, using {actual_type.value} instead")
        
        # Get factory function or model class
        factory_func = ModelRegistry.get_factory_function(actual_type)
        model_class = ModelRegistry.get_model_class(actual_type)
        
        try:
            if factory_func is not None:
                # Use factory function if available
                model = factory_func(**(config or {}))
            else:
                # Use direct constructor
                model = model_class(**(config or {}))
            
            logger.info(f"Created {actual_type.value} model with config: {config}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create {actual_type.value} model: {e}")
            raise

    @staticmethod
    def create_model_from_string(model_type_str: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create model from string type.
        
        Args:
            model_type_str: String representation of model type
            config: Configuration parameters
            
        Returns:
            Model instance
        """
        try:
            model_type = ModelType.from_string(model_type_str)
            return ModelFactory.create_model(model_type, config)
        except ValueError as e:
            logger.error(f"Invalid model type '{model_type_str}': {e}")
            raise

# Convenience functions for creating specific models
def create_catboost_regressor(**kwargs) -> CatBoostRegressor:
    """Create CatBoost regressor with default configuration."""
    return ModelFactory.create_model(ModelType.CATBOOST, kwargs)

def create_tabr_regressor(**kwargs) -> TabRRegressor:
    """Create TabR regressor with default configuration."""
    return ModelFactory.create_model(ModelType.TABR, kwargs)

def create_depthwise_cnn_regressor(**kwargs) -> TabRRegressor:
    """Create DepthwiseSeparableCNN regressor (deprecated - creates TabR)."""
    tprint_warning("⚠️ DepthwiseSeparableCNN is deprecated, using TabR instead")
    return ModelFactory.create_model(ModelType.DEPTWISE_SEPARABLE_CNN, kwargs)

def create_tcn_regressor(**kwargs) -> TabRRegressor:
    """Create TCN regressor (deprecated - creates TabR)."""
    tprint_warning("⚠️ TCN is deprecated, using TabR instead")
    return ModelFactory.create_model(ModelType.TCN, kwargs)

def get_available_models() -> List[str]:
    """Get list of available model types."""
    return ModelRegistry.list_available_models()

def is_model_available(model_type: str) -> bool:
    """Check if a model type is available."""
    try:
        ModelType.from_string(model_type)
        return True
    except ValueError:
        return False

# Initialize and validate model registry
def initialize_registry():
    """Initialize model registry and validate all models."""
    tprint_info("🔧 Initializing model registry...")
    
    available_models = []
    unavailable_models = []
    
    for model_type in ModelType:
        try:
            actual_type = model_type.get_actual_model_type()
            model_class = ModelRegistry.get_model_class(model_type)
            
            # Test instantiation with minimal parameters
            if actual_type == ModelType.CATBOOST:
                # Test with verbose=False to avoid output
                test_model = model_class(verbose=False)
            else:
                # TabR models require more setup, just check class
                test_model = None
            
            available_models.append(model_type.value)
            logger.info(f"✅ {model_type.value} -> {actual_type.value}: Available")
            
        except Exception as e:
            unavailable_models.append(model_type.value)
            logger.error(f"❌ {model_type.value}: {e}")
    
    if available_models:
        tprint_success(f"✅ Available models: {', '.join(available_models)}")
    
    if unavailable_models:
        tprint_warning(f"⚠️ Unavailable models: {', '.join(unavailable_models)}")
    
    return len(available_models) > 0

# Auto-initialize on import
try:
    initialize_registry()
except Exception as e:
    logger.error(f"Failed to initialize model registry: {e}")

# Export main components
__all__ = [
    'ModelType',
    'ModelRegistry', 
    'ModelFactory',
    'create_catboost_regressor',
    'create_tabr_regressor',
    'create_depthwise_cnn_regressor',
    'create_tcn_regressor',
    'get_available_models',
    'is_model_available',
    'initialize_registry',
    # Model classes
    'CatBoostRegressor',
    'TabRRegressor',
    'DepthwiseSeparableCNNRegressor',
]
