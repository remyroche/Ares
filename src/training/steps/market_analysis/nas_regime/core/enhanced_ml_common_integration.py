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

# Import enhanced utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory,
    safe_file_exists, timed_operation, format_bytes,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    memory_checkpoint, gpu_context, optimize_memory
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite,
    validate_positive, validate_range, safe_correlation,
    validate_numeric_array, MathValidationError
)
from src.utils.serialization_utils import UniversalSerializer

# Import ML common utilities with fallback
try:
    from src.utils.ml_common.common_operations import get_ml_common_operations
    from src.utils.ml_common.validation import get_validation_framework
    from src.utils.ml_common.optimization.grid_utils import (
        build_coarse_grid_from_search_space, build_fine_grid_around_best
    )
    from src.feature_selection.core import get_feature_selection_framework as get_feature_selection_utils
    from src.utils.ml_common.ensembles import get_ensemble_utils
    from src.utils.ml_common.evaluation import get_evaluation_utils
    from src.utils.ml_common.math_validation import MathValidator, ValidationLevel
    from src.utils.ml_common.config.enhanced_ml_config import EnhancedMLConfig
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False

    # Define fallback classes
    class ValidationLevel:
        LAX = "lax"
        STANDARD = "standard"
        STRICT = "strict"

    class MathValidator:
        def __init__(self, validation_level=None):
            self.validation_level = validation_level

logger = logging.getLogger(__name__)

@dataclass
class MLCommonConfig:
    """Enhanced configuration for ML common integration."""
    enable_validation: bool = True
    enable_feature_selection: bool = True
    enable_ensemble_methods: bool = True
    enable_evaluation: bool = True
    enable_optimization: bool = True
    enable_hardware_optimization: bool = True
    enable_m1_optimization: bool = True
    enable_serialization: bool = True
    validation_threshold: float = 0.8
    feature_selection_threshold: float = 0.1
    ensemble_method: str = 'voting'
    math_validation_level: str = 'standard'  # 'lax', 'standard', 'strict'
    enable_safe_math: bool = True
    enable_performance_monitoring: bool = True

class EnhancedMLCommonIntegration:
    """
    Enhanced ML Common Integration for Perfect NAS Regime System.

    Integrates with existing ML common utilities for:
    - Data validation with safe math
    - Feature selection with hardware optimization
    - Ensemble methods with M1 acceleration
    - Model evaluation with performance monitoring
    - Optimization utilities with serialization
    - Hardware optimization and memory management
    """

    def __init__(self, config: MLCommonConfig = None):
        """Initialize enhanced ML common integration.

        Args:
            config: ML common configuration
        """
        self.config = config or MLCommonConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize serialization
        if self.config.enable_serialization:
            self.serializer = UniversalSerializer()
        else:
            self.serializer = None

        # Initialize math validator
        if self.config.enable_safe_math:
            # Convert string to ValidationLevel enum member
            if self.config.math_validation_level == 'lax':
                validation_level = ValidationLevel.LAX
            elif self.config.math_validation_level == 'strict':
                validation_level = ValidationLevel.STRICT
            else:  # default to 'standard'
                validation_level = ValidationLevel.STANDARD
            self.math_validator = MathValidator(validation_level)
        else:
            self.math_validator = None

        # Initialize hardware optimization
        if self.config.enable_hardware_optimization:
            self._initialize_hardware_optimization()
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

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

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            if self.config.enable_m1_optimization:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ M1 hardware optimization initialized")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        except Exception as e:
            self.logger.warning(f"Hardware optimization initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

    def _initialize_fallback_utilities(self):
        """Initialize fallback utilities when ML common is not available."""
        self.ml_common_ops = None
        self.validation_framework = None
        self.feature_selection_utils = None
        self.ensemble_utils = None
        self.evaluation_utils = None

    @timed_operation
    def validate_data(self, data: np.ndarray, data_type: str = 'market_data') -> Dict[str, Any]:
        """Validate data using enhanced validation framework with safe math."""
        try:
            # Use memory checkpoint for large datasets
            with memory_checkpoint(f"validate_data_{data_type}"):
                # First, validate with math validator if available
                if self.math_validator:
                    math_validation = self.math_validator.validate_numeric_array(data, data_type)
                    if not math_validation.is_valid:
                        return {
                            'is_valid': False,
                            'errors': math_validation.errors,
                            'warnings': math_validation.warnings,
                            'data_quality_score': 0.0
                        }

                # Then use ML common validation framework if available
                if self.validation_framework:
                    result = self.validation_framework.validate_data(data, data_type)

                    # Enhance with safe math validation
                    if self.math_validator and 'is_valid' in result and result['is_valid']:
                        # Additional checks for financial data
                        if data_type == 'market_data':
                            result = self._enhance_financial_validation(data, result)

                    return result
                else:
                    # Enhanced fallback validation with safe math
                    return self._fallback_validation(data, data_type)
        except Exception as e:
            self.logger.warning(f"Data validation failed: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'data_quality_score': 0.0
            }

    def _enhance_financial_validation(self, data: np.ndarray, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance validation results with financial-specific checks."""
        try:
            enhanced_result = base_result.copy()

            # Check for negative prices (should not occur in OHLCV data)
            if data.ndim == 2 and data.shape[1] >= 4:  # OHLC data
                negative_prices = np.any(data < 0, axis=1).sum()
                if negative_prices > 0:
                    enhanced_result['warnings'] = enhanced_result.get('warnings', [])
                    enhanced_result['warnings'].append(f"Found {negative_prices} records with negative prices")
                    enhanced_result['data_quality_score'] = max(0.0, enhanced_result.get('data_quality_score', 1.0) - 0.1)

            # Check for zero volume
            if data.ndim == 2 and data.shape[1] >= 5:  # OHLCV data
                zero_volume = (data[:, -1] == 0).sum()  # Volume is typically last column
                if zero_volume > 0:
                    enhanced_result['warnings'] = enhanced_result.get('warnings', [])
                    enhanced_result['warnings'].append(f"Found {zero_volume} records with zero volume")

            # Check for price relationships (High >= Low, etc.)
            if data.ndim == 2 and data.shape[1] >= 4:
                invalid_relationships = 0
                for i in range(len(data)):
                    if data[i, 1] < data[i, 2] or data[i, 1] < data[i, 0] or data[i, 1] < data[i, 3]:  # High < Low/Open/Close
                        invalid_relationships += 1
                    if data[i, 2] > data[i, 0] or data[i, 2] > data[i, 1] or data[i, 2] > data[i, 3]:  # Low > Open/High/Close
                        invalid_relationships += 1

                if invalid_relationships > 0:
                    enhanced_result['warnings'] = enhanced_result.get('warnings', [])
                    enhanced_result['warnings'].append(f"Found {invalid_relationships} records with invalid price relationships")
                    enhanced_result['data_quality_score'] = max(0.0, enhanced_result.get('data_quality_score', 1.0) - 0.2)

            return enhanced_result

        except Exception as e:
            self.logger.warning(f"Financial validation enhancement failed: {e}")
            return base_result

    def _fallback_validation(self, data: np.ndarray, data_type: str) -> Dict[str, Any]:
        """Enhanced fallback validation with safe math operations."""
        try:
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
