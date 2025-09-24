"""
Enhanced ML Common Integration for Perfect NAS Regime System

Integrates with utils/ml_common/ for comprehensive ML utilities.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass

# Import ML common utilities with fallback
try:
    from src.utils.ml_common.common_operations import get_ml_common_operations
    from src.utils.ml_common.validation import get_validation_framework
    from src.utils.ml_common.optimization.grid_utils import (
        build_coarse_grid_from_search_space, build_fine_grid_around_best
    )
    from src.utils.ml_common.feature_selection import get_feature_selection_utils
    from src.utils.ml_common.ensembles import get_ensemble_utils
    from src.utils.ml_common.evaluation import get_evaluation_utils
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MLCommonConfig:
    """Configuration for ML common integration."""
    enable_validation: bool = True
    enable_feature_selection: bool = True
    enable_ensemble_methods: bool = True
    enable_evaluation: bool = True
    enable_optimization: bool = True
    validation_threshold: float = 0.8
    feature_selection_threshold: float = 0.1
    ensemble_method: str = 'voting'

class EnhancedMLCommonIntegration:
    """
    Enhanced ML Common Integration for Perfect NAS Regime System.
    
    Integrates with existing ML common utilities for:
    - Data validation
    - Feature selection
    - Ensemble methods
    - Model evaluation
    - Optimization utilities
    """
    
    def __init__(self, config: MLCommonConfig = None):
        """Initialize enhanced ML common integration.
        
        Args:
            config: ML common configuration
        """
        self.config = config or MLCommonConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ML common utilities if available
        if ML_COMMON_AVAILABLE:
            try:
                self.ml_common_ops = get_ml_common_operations()
                self.validation_framework = get_validation_framework()
                self.feature_selection_utils = get_feature_selection_utils()
                self.ensemble_utils = get_ensemble_utils()
                self.evaluation_utils = get_evaluation_utils()
                self.logger.info("✅ Enhanced ML common integration initialized with full utilities")
            except Exception as e:
                self.logger.warning(f"ML common utilities initialization failed: {e}")
                self._initialize_fallback_utilities()
        else:
            self.logger.warning("ML common utilities not available - using fallback implementations")
            self._initialize_fallback_utilities()
    
    def _initialize_fallback_utilities(self):
        """Initialize fallback utilities when ML common is not available."""
        self.ml_common_ops = None
        self.validation_framework = None
        self.feature_selection_utils = None
        self.ensemble_utils = None
        self.evaluation_utils = None
    
    def validate_data(self, data: np.ndarray, data_type: str = 'market_data') -> Dict[str, Any]:
        """Validate data using ML common validation framework."""
        try:
            if self.validation_framework:
                return self.validation_framework.validate_data(data, data_type)
            else:
                # Fallback validation
                validation_result = {
                    'is_valid': True,
                    'data_quality_score': 1.0,
                    'missing_values': 0,
                    'outliers': 0,
                    'data_distribution': 'normal',
                    'recommendations': []
                }
                
                # Basic validation checks
                if np.isnan(data).any():
                    validation_result['is_valid'] = False
                    validation_result['missing_values'] = np.isnan(data).sum()
                    validation_result['recommendations'].append('Handle missing values')
                
                if np.isinf(data).any():
                    validation_result['is_valid'] = False
                    validation_result['recommendations'].append('Handle infinite values')
                
                return validation_result
                
        except Exception as e:
            self.logger.warning(f"Data validation failed: {e}")
            return {'is_valid': False, 'error': str(e)}
    
    def select_features(self, data: np.ndarray, target: np.ndarray = None, 
                       method: str = 'correlation') -> Dict[str, Any]:
        """Select features using ML common feature selection utilities."""
        try:
            if self.feature_selection_utils:
                return self.feature_selection_utils.select_features(data, target, method)
            else:
                # Fallback feature selection
                n_features = data.shape[1]
                selected_features = list(range(n_features))  # Select all features
                
                return {
                    'selected_features': selected_features,
                    'feature_importance': np.ones(n_features),
                    'selection_method': 'fallback',
                    'n_selected': n_features
                }
                
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return {'selected_features': [], 'error': str(e)}
    
    def create_ensemble(self, models: List[Any], method: str = 'voting') -> Any:
        """Create ensemble using ML common ensemble utilities."""
        try:
            if self.ensemble_utils:
                return self.ensemble_utils.create_ensemble(models, method)
            else:
                # Fallback ensemble (simple voting)
                class FallbackEnsemble:
                    def __init__(self, models):
                        self.models = models
                    
                    def predict(self, X):
                        predictions = []
                        for model in self.models:
                            if hasattr(model, 'predict'):
                                pred = model.predict(X)
                            else:
                                pred = model(X)
                            predictions.append(pred)
                        
                        # Simple voting
                        if isinstance(predictions[0], np.ndarray):
                            return np.mean(predictions, axis=0)
                        else:
                            return torch.mean(torch.stack(predictions), dim=0)
                
                return FallbackEnsemble(models)
                
        except Exception as e:
            self.logger.warning(f"Ensemble creation failed: {e}")
            return None
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                      metrics: List[str] = None) -> Dict[str, float]:
        """Evaluate model using ML common evaluation utilities."""
        try:
            if self.evaluation_utils:
                return self.evaluation_utils.evaluate_model(model, X, y, metrics)
            else:
                # Fallback evaluation
                if hasattr(model, 'predict'):
                    predictions = model.predict(X)
                else:
                    predictions = model(X)
                
                # Basic metrics
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.detach().cpu().numpy()
                
                if isinstance(y, torch.Tensor):
                    y = y.detach().cpu().numpy()
                
                # Calculate basic metrics
                mse = np.mean((predictions - y) ** 2)
                mae = np.mean(np.abs(predictions - y))
                r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
                
                return {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
        except Exception as e:
            self.logger.warning(f"Model evaluation failed: {e}")
            return {'error': str(e)}
    
    def optimize_hyperparameters(self, model_class: Any, X: np.ndarray, y: np.ndarray,
                                param_grid: Dict[str, List] = None) -> Dict[str, Any]:
        """Optimize hyperparameters using ML common optimization utilities."""
        try:
            if self.ml_common_ops and hasattr(self.ml_common_ops, 'optimize_hyperparameters'):
                return self.ml_common_ops.optimize_hyperparameters(model_class, X, y, param_grid)
            else:
                # Fallback hyperparameter optimization
                if param_grid is None:
                    param_grid = {'learning_rate': [0.001, 0.01, 0.1]}
                
                best_params = {}
                best_score = float('inf')
                
                # Simple grid search
                for param_name, param_values in param_grid.items():
                    for param_value in param_values:
                        # Create model with parameter
                        if hasattr(model_class, '__init__'):
                            try:
                                model = model_class(**{param_name: param_value})
                                
                                # Simple evaluation
                                if hasattr(model, 'fit'):
                                    model.fit(X, y)
                                
                                if hasattr(model, 'predict'):
                                    predictions = model.predict(X)
                                    score = np.mean((predictions - y) ** 2)
                                    
                                    if score < best_score:
                                        best_score = score
                                        best_params[param_name] = param_value
                            except Exception:
                                continue
                
                return {
                    'best_params': best_params,
                    'best_score': best_score,
                    'optimization_method': 'fallback_grid_search'
                }
                
        except Exception as e:
            self.logger.warning(f"Hyperparameter optimization failed: {e}")
            return {'error': str(e)}
    
    def create_search_space(self, param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Create search space for optimization."""
        try:
            if self.ml_common_ops and hasattr(self.ml_common_ops, 'create_search_space'):
                return self.ml_common_ops.create_search_space(param_ranges)
            else:
                # Fallback search space
                search_space = {}
                for param_name, (min_val, max_val) in param_ranges.items():
                    search_space[param_name] = {
                        'min': min_val,
                        'max': max_val,
                        'type': 'continuous'
                    }
                return search_space
                
        except Exception as e:
            self.logger.warning(f"Search space creation failed: {e}")
            return {}
    
    def build_optimization_grid(self, search_space: Dict[str, Any], 
                               grid_size: int = 10) -> List[Dict[str, Any]]:
        """Build optimization grid using ML common utilities."""
        try:
            if self.ml_common_ops and hasattr(self.ml_common_ops, 'build_optimization_grid'):
                return self.ml_common_ops.build_optimization_grid(search_space, grid_size)
            else:
                # Fallback grid building
                grid_points = []
                
                for param_name, param_config in search_space.items():
                    if param_config.get('type') == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        values = np.linspace(min_val, max_val, grid_size)
                        
                        if not grid_points:
                            grid_points = [{param_name: val} for val in values]
                        else:
                            new_points = []
                            for point in grid_points:
                                for val in values:
                                    new_point = point.copy()
                                    new_point[param_name] = val
                                    new_points.append(new_point)
                            grid_points = new_points
                
                return grid_points
                
        except Exception as e:
            self.logger.warning(f"Optimization grid building failed: {e}")
            return []
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics from ML common integration."""
        try:
            if self.validation_framework:
                return {
                    'validation_enabled': True,
                    'validation_threshold': self.config.validation_threshold,
                    'validation_method': 'ml_common'
                }
            else:
                return {
                    'validation_enabled': False,
                    'validation_threshold': self.config.validation_threshold,
                    'validation_method': 'fallback'
                }
        except Exception as e:
            self.logger.warning(f"Validation metrics collection failed: {e}")
            return {}
    
    def get_feature_selection_metrics(self) -> Dict[str, Any]:
        """Get feature selection metrics from ML common integration."""
        try:
            if self.feature_selection_utils:
                return {
                    'feature_selection_enabled': True,
                    'selection_threshold': self.config.feature_selection_threshold,
                    'selection_method': 'ml_common'
                }
            else:
                return {
                    'feature_selection_enabled': False,
                    'selection_threshold': self.config.feature_selection_threshold,
                    'selection_method': 'fallback'
                }
        except Exception as e:
            self.logger.warning(f"Feature selection metrics collection failed: {e}")
            return {}
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get ensemble metrics from ML common integration."""
        try:
            if self.ensemble_utils:
                return {
                    'ensemble_enabled': True,
                    'ensemble_method': self.config.ensemble_method,
                    'ensemble_type': 'ml_common'
                }
            else:
                return {
                    'ensemble_enabled': False,
                    'ensemble_method': self.config.ensemble_method,
                    'ensemble_type': 'fallback'
                }
        except Exception as e:
            self.logger.warning(f"Ensemble metrics collection failed: {e}")
            return {}
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics from ML common integration."""
        try:
            if self.evaluation_utils:
                return {
                    'evaluation_enabled': True,
                    'evaluation_method': 'ml_common',
                    'metrics_available': True
                }
            else:
                return {
                    'evaluation_enabled': False,
                    'evaluation_method': 'fallback',
                    'metrics_available': False
                }
        except Exception as e:
            self.logger.warning(f"Evaluation metrics collection failed: {e}")
            return {}
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics from ML common integration."""
        try:
            if self.ml_common_ops:
                return {
                    'optimization_enabled': True,
                    'optimization_method': 'ml_common',
                    'grid_search_available': True
                }
            else:
                return {
                    'optimization_enabled': False,
                    'optimization_method': 'fallback',
                    'grid_search_available': False
                }
        except Exception as e:
            self.logger.warning(f"Optimization metrics collection failed: {e}")
            return {}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics from ML common integration."""
        return {
            'validation': self.get_validation_metrics(),
            'feature_selection': self.get_feature_selection_metrics(),
            'ensemble': self.get_ensemble_metrics(),
            'evaluation': self.get_evaluation_metrics(),
            'optimization': self.get_optimization_metrics(),
            'ml_common_available': ML_COMMON_AVAILABLE
        }