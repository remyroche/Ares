"""
Weighted Loss Integration for ML Model Trainer

This module provides integration utilities for applying weighted losses
across all models in the ML Model Trainer pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_data_format, tprint_data_preview, LogLevel
)

# Import math validation utilities
from src.utils.math_validation import (
    validate_finite, validate_array_finite, safe_divide, safe_log, safe_sqrt
)

# Import common operations utilities
from src.utils.common_operations import (
    safe_dataframe_operation, memory_managed, MemoryStrategy
)

# Import hardware optimization decorators
from src.utils.hardware.optimization_decorators import (
    performance_tracked, memory_efficient, auto_optimize
)

from .weighted_loss_framework import (
    WeightedLossManager, WeightedLossConfig, WeightingStrategy, FailureContextType
)
from .error_handling import (
    handle_errors, validate_data,
    MLModelTrainerError, DataValidationError, ModelTrainingError
)

logger = logging.getLogger(__name__)

@dataclass
class WeightedLossIntegrationConfig:
    """Configuration for weighted loss integration."""
    # Core settings
    enable_weighted_loss: bool = True
    weighting_strategy: WeightingStrategy = WeightingStrategy.ADAPTIVE
    
    # Model-specific settings
    enable_for_lightgbm: bool = True
    enable_for_catboost: bool = True
    enable_for_xgboost: bool = True
    enable_for_sklearn: bool = True
    enable_for_neural_networks: bool = True
    
    # Failure context detection
    volatility_threshold: float = 0.02
    chop_threshold: float = 0.5
    spread_threshold: float = 0.01
    liquidity_threshold: float = 1000.0
    
    # Weighting parameters
    base_weight: float = 1.0
    max_weight: float = 5.0
    min_weight: float = 0.1
    weight_smoothing: float = 0.1
    
    # Adaptive parameters
    adaptation_rate: float = 0.01
    memory_decay: float = 0.95
    stability_threshold: float = 0.1
    
    # Performance monitoring
    enable_monitoring: bool = True
    log_frequency: int = 100
    save_weight_statistics: bool = True

class WeightedLossIntegrator:
    """Integrates weighted losses across all model types."""
    
    def __init__(self, config: WeightedLossIntegrationConfig):
        self.config = config
        self.weighted_loss_managers = {}
        self.weight_statistics = {}
        self.is_initialized = False
        
    def initialize(self, model_types: List[str], feature_names: Optional[List[str]] = None):
        """Initialize weighted loss managers for each model type."""
        tprint_info("Initializing weighted loss integrator...")
        
        for model_type in model_types:
            # Create weighted loss config for this model type
            model_config = WeightedLossConfig(
                enable_weighted_loss=self.config.enable_weighted_loss,
                weighting_strategy=self.config.weighting_strategy,
                volatility_threshold=self.config.volatility_threshold,
                chop_threshold=self.config.chop_threshold,
                spread_threshold=self.config.spread_threshold,
                liquidity_threshold=self.config.liquidity_threshold,
                base_weight=self.config.base_weight,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight,
                weight_smoothing=self.config.weight_smoothing,
                adaptation_rate=self.config.adaptation_rate,
                memory_decay=self.config.memory_decay,
                stability_threshold=self.config.stability_threshold,
                enable_monitoring=self.config.enable_monitoring,
                log_frequency=self.config.log_frequency
            )
            
            # Create weighted loss manager
            self.weighted_loss_managers[model_type] = WeightedLossManager(model_config)
            self.weight_statistics[model_type] = []
            
        self.is_initialized = True
        tprint_success(f"Weighted loss integrator initialized for {len(model_types)} model types")
        
    @memory_managed(MemoryStrategy.MODERATE)
    @performance_tracked(log_performance=True, track_memory=True)
    def fit(self, model_type: str, X: np.ndarray, y: np.ndarray, 
           market_data: Optional[Dict[str, np.ndarray]] = None):
        """Fit weighted loss manager for specific model type."""
        if not self.is_initialized:
            raise ModelTrainingError("Integrator must be initialized before fitting")
            
        if model_type not in self.weighted_loss_managers:
            raise ModelTrainingError(f"Model type {model_type} not found in integrator")
        
        tprint_info(f"Fitting weighted loss manager for {model_type}")
        tprint_data_format(X, f"Input features for {model_type}", LogLevel.DEBUG)
        tprint_data_format(y, f"Target values for {model_type}", LogLevel.DEBUG)
        
        # Validate inputs using math validation utilities
        X = validate_array_finite(X, f"X_{model_type}")
        y = validate_array_finite(y, f"y_{model_type}")
        
        if market_data:
            tprint_data_preview(market_data, f"Market data for {model_type}", LogLevel.DEBUG)
        
        self.weighted_loss_managers[model_type].fit(X, y, market_data)
        
        tprint_success(f"Weighted loss manager fitted for {model_type}")
        tprint_data_format(self.weight_statistics[model_type], f"Weight statistics for {model_type}", LogLevel.DEBUG)
        
    @memory_managed(MemoryStrategy.MODERATE)
    def get_sample_weights(self, model_type: str, X: np.ndarray, y: np.ndarray,
                          predictions: Optional[np.ndarray] = None,
                          market_data: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Get sample weights for specific model type with enhanced logging."""
        tprint_debug(f"Getting sample weights for {model_type}")
        tprint_data_format(X, f"Input features for {model_type} weights", LogLevel.DEBUG)
        tprint_data_format(y, f"Target values for {model_type} weights", LogLevel.DEBUG)
        
        if model_type not in self.weighted_loss_managers:
            tprint_warning(f"No weighted loss manager for {model_type}, returning uniform weights")
            return np.ones(len(y))
        
        # Validate inputs
        X = validate_array_finite(X, f"X_{model_type}_weights")
        y = validate_array_finite(y, f"y_{model_type}_weights")
        
        weights = self.weighted_loss_managers[model_type].get_sample_weights(
            X, y, predictions, market_data
        )
        
        # Store weight statistics with enhanced logging
        if self.config.save_weight_statistics:
            weight_stats = {
                'mean_weight': float(np.mean(weights)),
                'std_weight': float(np.std(weights)),
                'min_weight': float(np.min(weights)),
                'max_weight': float(np.max(weights)),
                'samples': len(weights)
            }
            self.weight_statistics[model_type].append(weight_stats)
            
            tprint_data_format(weight_stats, f"Weight statistics for {model_type}", LogLevel.DEBUG)
        
        tprint_data_format(weights, f"Calculated sample weights for {model_type}", LogLevel.DEBUG)
        return weights
    
    def calculate_weighted_loss(self, model_type: str, y_true: np.ndarray, y_pred: np.ndarray,
                              X: np.ndarray, loss_type: str = "mse",
                              market_data: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Calculate weighted loss for specific model type."""
        if model_type not in self.weighted_loss_managers:
            tprint_warning(f"No weighted loss manager for {model_type}, using standard loss")
            # Fallback to standard loss calculation
            if loss_type == "mse":
                return np.mean((y_true - y_pred) ** 2)
            elif loss_type == "mae":
                return np.mean(np.abs(y_true - y_pred))
            else:
                return 0.0
                
        return self.weighted_loss_managers[model_type].calculate_weighted_loss(
            y_true, y_pred, X, loss_type, market_data
        )
    
    def get_weight_statistics(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get weight statistics for monitoring."""
        if model_type is not None:
            if model_type not in self.weight_statistics:
                return {}
            return self.weight_statistics[model_type]
        
        # Return statistics for all model types
        all_stats = {}
        for model_type, stats in self.weight_statistics.items():
            if stats:
                all_stats[model_type] = {
                    'recent_mean': np.mean([s['mean_weight'] for s in stats[-10:]]),
                    'recent_std': np.mean([s['std_weight'] for s in stats[-10:]]),
                    'total_samples': sum([s['samples'] for s in stats]),
                    'weight_entries': len(stats)
                }
        
        return all_stats
    
    def update_model_parameters(self, model_type: str, model, X: np.ndarray, y: np.ndarray,
                              predictions: Optional[np.ndarray] = None,
                              market_data: Optional[Dict[str, np.ndarray]] = None):
        """Update model parameters with weighted loss information."""
        if not self.config.enable_weighted_loss:
            return
            
        if model_type not in self.weighted_loss_managers:
            return
            
        # Get sample weights
        sample_weights = self.get_sample_weights(model_type, X, y, predictions, market_data)
        
        # Update model parameters based on model type
        if hasattr(model, 'sample_weight'):
            model.sample_weight = sample_weights
        elif hasattr(model, 'set_params'):
            # For sklearn models, try to set sample_weight parameter
            try:
                model.set_params(sample_weight=sample_weights)
            except:
                pass
        elif hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
            # Store weights for use in fit method
            model._sample_weights = sample_weights
        
        tprint_debug(f"Updated {model_type} model with weighted loss parameters")
    
    def create_weighted_loss_callback(self, model_type: str):
        """Create a callback function for weighted loss updates."""
        def weighted_loss_callback(model, X, y, predictions=None, market_data=None):
            self.update_model_parameters(model_type, model, X, y, predictions, market_data)
        
        return weighted_loss_callback

class WeightedLossModelWrapper:
    """Wrapper to add weighted loss functionality to any model."""
    
    def __init__(self, base_model, model_type: str, integrator: WeightedLossIntegrator):
        self.base_model = base_model
        self.model_type = model_type
        self.integrator = integrator
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        """Fit the model with weighted loss."""
        tprint_info(f"Fitting {self.model_type} model with weighted loss...")
        
        # Get sample weights
        sample_weights = self.integrator.get_sample_weights(
            self.model_type, X, y, None, fit_params.get('market_data')
        )
        
        # Add sample weights to fit parameters
        fit_params['sample_weight'] = sample_weights
        
        # Fit the base model
        self.base_model.fit(X, y, **fit_params)
        self.is_fitted = True
        
        tprint_success(f"{self.model_type} model fitted with weighted loss")
        
        return self
    
    def predict(self, X: np.ndarray):
        """Make predictions."""
        if not self.is_fitted:
            raise ModelTrainingError("Model must be fitted before prediction")
        return self.base_model.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        """Make probability predictions."""
        if not self.is_fitted:
            raise ModelTrainingError("Model must be fitted before prediction")
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_type} model does not support predict_proba")
    
    def get_params(self, deep=True):
        """Get model parameters."""
        return self.base_model.get_params(deep=deep)
    
    def set_params(self, **params):
        """Set model parameters."""
        return self.base_model.set_params(**params)

def create_weighted_loss_integrator(config: Optional[WeightedLossIntegrationConfig] = None) -> WeightedLossIntegrator:
    """Create a weighted loss integrator with configuration."""
    if config is None:
        config = WeightedLossIntegrationConfig()
    
    return WeightedLossIntegrator(config)

def wrap_model_with_weighted_loss(model, model_type: str, integrator: WeightedLossIntegrator) -> WeightedLossModelWrapper:
    """Wrap a model with weighted loss functionality."""
    return WeightedLossModelWrapper(model, model_type, integrator)