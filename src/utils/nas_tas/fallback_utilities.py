"""
Centralized Fallback Utility Helpers for NAS/TAS Modules

This module provides centralized fallback utilities for confidence and uncertainty modules,
ensuring graceful degradation when dependencies are missing.

Key Features:
- Centralized fallback implementations for common utilities
- Graceful degradation when dependencies are missing
- Consistent error handling and logging
- Memory-efficient fallback operations
- Hardware optimization fallbacks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle

# Setup logging
logger = logging.getLogger(__name__)

# Fallback utility flags
FALLBACK_UTILITIES_AVAILABLE = True

@dataclass
class FallbackConfig:
    """Configuration for fallback utilities."""
    enable_logging: bool = True
    enable_warnings: bool = True
    fallback_accuracy: float = 0.5
    fallback_efficiency: float = 0.5
    fallback_interpretability: float = 0.5
    fallback_robustness: float = 0.5
    memory_limit_gb: Optional[float] = None
    timeout_seconds: int = 300


class FallbackMathUtils:
    """Fallback math utilities when advanced math validation is not available."""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._logged_events: Set[str] = set()

    def _log_fallback_event(self, key: str, message: str):
        """Log fallback events once to avoid noisy output."""
        if not self.config.enable_logging or key in self._logged_events:
            return

        log_method = self.logger.warning if self.config.enable_warnings else self.logger.info
        log_method(message)
        self._logged_events.add(key)
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with fallback."""
        try:
            if denominator == 0 or not np.isfinite(denominator):
                self._log_fallback_event(
                    "safe_divide_non_finite_denominator",
                    "Division encountered a non-finite denominator; returning default value.",
                )
                return default
            result = numerator / denominator
            if not np.isfinite(result):
                self._log_fallback_event(
                    "safe_divide_non_finite_result",
                    "Division result was non-finite; returning default value.",
                )
                return default
            return result
        except Exception as exc:
            self._log_fallback_event(
                "safe_divide_exception",
                f"Division failed with error '{exc}'; returning default value.",
            )
            return default
    
    def safe_log(self, value: float, default: float = 0.0) -> float:
        """Safe logarithm with fallback."""
        try:
            if value <= 0 or not np.isfinite(value):
                self._log_fallback_event(
                    "safe_log_invalid_input",
                    "Logarithm received non-positive or non-finite input; returning default value.",
                )
                return default
            result = np.log(value)
            if not np.isfinite(result):
                self._log_fallback_event(
                    "safe_log_non_finite_result",
                    "Logarithm result was non-finite; returning default value.",
                )
                return default
            return result
        except Exception as exc:
            self._log_fallback_event(
                "safe_log_exception",
                f"Logarithm failed with error '{exc}'; returning default value.",
            )
            return default

    def safe_sqrt(self, value: float, default: float = 0.0) -> float:
        """Safe square root with fallback."""
        try:
            if value < 0 or not np.isfinite(value):
                self._log_fallback_event(
                    "safe_sqrt_invalid_input",
                    "Square root received negative or non-finite input; returning default value.",
                )
                return default
            result = np.sqrt(value)
            if not np.isfinite(result):
                self._log_fallback_event(
                    "safe_sqrt_non_finite_result",
                    "Square root result was non-finite; returning default value.",
                )
                return default
            return result
        except Exception as exc:
            self._log_fallback_event(
                "safe_sqrt_exception",
                f"Square root failed with error '{exc}'; returning default value.",
            )
            return default
    
    def safe_mean(self, values: Union[List[float], np.ndarray], default: float = 0.0) -> float:
        """Safe mean calculation with fallback."""
        try:
            if len(values) == 0:
                self._log_fallback_event(
                    "safe_mean_empty",
                    "Mean calculation received empty values; returning default value.",
                )
                return default
            values = np.array(values)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                self._log_fallback_event(
                    "safe_mean_non_finite",
                    "Mean calculation had no finite values; returning default value.",
                )
                return default
            return float(np.mean(values))
        except Exception as exc:
            self._log_fallback_event(
                "safe_mean_exception",
                f"Mean calculation failed with error '{exc}'; returning default value.",
            )
            return default

    def safe_std(self, values: Union[List[float], np.ndarray], default: float = 0.0) -> float:
        """Safe standard deviation calculation with fallback."""
        try:
            if len(values) == 0:
                self._log_fallback_event(
                    "safe_std_empty",
                    "Standard deviation received empty values; returning default value.",
                )
                return default
            values = np.array(values)
            values = values[np.isfinite(values)]
            if len(values) <= 1:
                self._log_fallback_event(
                    "safe_std_insufficient",
                    "Standard deviation requires more than one finite value; returning default value.",
                )
                return default
            return float(np.std(values))
        except Exception as exc:
            self._log_fallback_event(
                "safe_std_exception",
                f"Standard deviation failed with error '{exc}'; returning default value.",
            )
            return default
    
    def safe_correlation(self, x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
        """Safe correlation calculation with fallback."""
        try:
            if len(x) != len(y) or len(x) <= 1:
                self._log_fallback_event(
                    "safe_correlation_length_mismatch",
                    "Correlation inputs had mismatched lengths or insufficient samples; returning default value.",
                )
                return default
            x, y = np.array(x), np.array(y)
            x, y = x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)]
            if len(x) <= 1:
                self._log_fallback_event(
                    "safe_correlation_non_finite",
                    "Correlation inputs had insufficient finite samples; returning default value.",
                )
                return default
            corr = np.corrcoef(x, y)[0, 1]
            if not np.isfinite(corr):
                self._log_fallback_event(
                    "safe_correlation_non_finite_result",
                    "Correlation result was non-finite; returning default value.",
                )
                return default
            return corr
        except Exception as exc:
            self._log_fallback_event(
                "safe_correlation_exception",
                f"Correlation calculation failed with error '{exc}'; returning default value.",
            )
            return default

    def safe_covariance(self, x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray], default: float = 0.0) -> float:
        """Safe covariance calculation with fallback."""
        try:
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            if x_arr.size == 0 or y_arr.size == 0:
                self._log_fallback_event(
                    "safe_covariance_empty",
                    "Covariance calculation received empty values; returning default value.",
                )
                return default
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if mask.sum() <= 1:
                self._log_fallback_event(
                    "safe_covariance_insufficient",
                    "Covariance calculation requires at least two finite samples; returning default value.",
                )
                return default
            return float(np.cov(x_arr[mask], y_arr[mask], ddof=1)[0, 1])
        except Exception as exc:
            self._log_fallback_event(
                "safe_covariance_exception",
                f"Covariance calculation failed with error '{exc}'; returning default value.",
            )
            return default

    def safe_percentile(self, values: Union[List[float], np.ndarray], percentile: float, default: float = 0.0) -> float:
        """Safe percentile calculation with fallback."""
        try:
            if len(values) == 0:
                self._log_fallback_event(
                    "safe_percentile_empty",
                    "Percentile calculation received empty values; returning default value.",
                )
                return default
            values = np.array(values)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                self._log_fallback_event(
                    "safe_percentile_non_finite",
                    "Percentile calculation had no finite values; returning default value.",
                )
                return default
            return float(np.percentile(values, percentile))
        except Exception as exc:
            self._log_fallback_event(
                "safe_percentile_exception",
                f"Percentile calculation failed with error '{exc}'; returning default value.",
            )
            return default

    def safe_weighted_average(
        self,
        values: Union[List[float], np.ndarray],
        weights: Optional[Union[List[float], np.ndarray]] = None,
        default: float = 0.0,
    ) -> float:
        """Safe weighted average with fallback behaviour."""
        try:
            values_arr = np.asarray(values, dtype=float)
            if values_arr.size == 0:
                self._log_fallback_event(
                    "safe_weighted_average_empty",
                    "Weighted average received empty values; returning default value.",
                )
                return default

            if weights is None:
                mask = np.isfinite(values_arr)
                if not np.any(mask):
                    self._log_fallback_event(
                        "safe_weighted_average_non_finite",
                        "Weighted average received no finite values; returning default value.",
                    )
                    return default
                return float(np.mean(values_arr[mask]))

            weights_arr = np.asarray(weights, dtype=float)
            if weights_arr.size != values_arr.size:
                self._log_fallback_event(
                    "safe_weighted_average_mismatch",
                    "Weighted average received mismatched value/weight lengths; returning default value.",
                )
                return default
            mask = np.isfinite(values_arr) & np.isfinite(weights_arr)
            if not np.any(mask):
                self._log_fallback_event(
                    "safe_weighted_average_non_finite_weighted",
                    "Weighted average received no finite weighted samples; returning default value.",
                )
                return default
            weights_sum = np.sum(weights_arr[mask])
            if weights_sum == 0:
                self._log_fallback_event(
                    "safe_weighted_average_zero_weights",
                    "Weighted average weights sum to zero; returning default value.",
                )
                return default
            return float(np.average(values_arr[mask], weights=weights_arr[mask]))
        except Exception as exc:
            self._log_fallback_event(
                "safe_weighted_average_exception",
                f"Weighted average calculation failed with error '{exc}'; returning default value.",
            )
            return default

    def safe_power(self, value: float, exponent: float, default: float = 0.0) -> float:
        """Safe power operation with fallback."""
        try:
            if not np.isfinite(value) or not np.isfinite(exponent):
                self._log_fallback_event(
                    "safe_power_non_finite",
                    "Power received non-finite value or exponent; returning default value.",
                )
                return default
            result = float(np.power(value, exponent))
            if not np.isfinite(result):
                self._log_fallback_event(
                    "safe_power_result_non_finite",
                    "Power result was non-finite; returning default value.",
                )
                return default
            return result
        except Exception as exc:
            self._log_fallback_event(
                "safe_power_exception",
                f"Power calculation failed with error '{exc}'; returning default value.",
            )
            return default

    def safe_percentage_change(self, current: float, previous: float, default: float = 0.0) -> float:
        """Safe percentage change calculation with fallback."""
        try:
            if not np.isfinite(current) or not np.isfinite(previous):
                self._log_fallback_event(
                    "safe_percentage_change_non_finite",
                    "Percentage change received non-finite values; returning default value.",
                )
                return default
            if previous == 0:
                self._log_fallback_event(
                    "safe_percentage_change_divide_by_zero",
                    "Percentage change encountered zero previous value; returning default value.",
                )
                return default
            result = (current - previous) / abs(previous)
            if not np.isfinite(result):
                self._log_fallback_event(
                    "safe_percentage_change_non_finite_result",
                    "Percentage change result was non-finite; returning default value.",
                )
                return default
            return float(result)
        except Exception as exc:
            self._log_fallback_event(
                "safe_percentage_change_exception",
                f"Percentage change calculation failed with error '{exc}'; returning default value.",
            )
            return default

    def safe_kelly_calculation(self, win_probability: float, win_loss_ratio: float, default: float = 0.0) -> float:
        """Safe Kelly criterion calculation with fallback."""
        try:
            if not all(np.isfinite(v) for v in (win_probability, win_loss_ratio)):
                self._log_fallback_event(
                    "safe_kelly_non_finite",
                    "Kelly calculation received non-finite inputs; returning default value.",
                )
                return default
            if not 0 <= win_probability <= 1:
                self._log_fallback_event(
                    "safe_kelly_probability_range",
                    "Kelly calculation received probability outside [0, 1]; returning default value.",
                )
                return default
            denominator = win_loss_ratio
            if denominator == 0:
                self._log_fallback_event(
                    "safe_kelly_zero_denominator",
                    "Kelly calculation encountered zero win/loss ratio; returning default value.",
                )
                return default
            edge = win_probability * (win_loss_ratio + 1) - 1
            fraction = edge / denominator
            if not np.isfinite(fraction):
                self._log_fallback_event(
                    "safe_kelly_non_finite_result",
                    "Kelly calculation result was non-finite; returning default value.",
                )
                return default
            return max(default, float(fraction))
        except Exception as exc:
            self._log_fallback_event(
                "safe_kelly_exception",
                f"Kelly calculation failed with error '{exc}'; returning default value.",
            )
            return default

    def safe_matrix_inverse(self, matrix: Union[List[List[float]], np.ndarray], default: Optional[np.ndarray] = None):
        """Safe matrix inverse with fallback."""
        try:
            matrix_arr = np.asarray(matrix, dtype=float)
            if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
                self._log_fallback_event(
                    "safe_matrix_inverse_shape",
                    "Matrix inverse received a non-square matrix; returning default value.",
                )
                return default
            result = np.linalg.inv(matrix_arr)
            if not np.all(np.isfinite(result)):
                self._log_fallback_event(
                    "safe_matrix_inverse_non_finite",
                    "Matrix inverse produced non-finite values; returning default value.",
                )
                return default
            return result
        except Exception as exc:
            self._log_fallback_event(
                "safe_matrix_inverse_exception",
                f"Matrix inversion failed with error '{exc}'; returning default value.",
            )
            return default


class FallbackHardwareUtils:
    """Fallback hardware utilities when M1 optimizations are not available."""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.memory_usage = 0.0
        self.gpu_available = False
        self._logged_events: Set[str] = set()

    def _log_fallback_event(self, key: str, message: str):
        if not self.config.enable_logging or key in self._logged_events:
            return

        log_method = self.logger.warning if self.config.enable_warnings else self.logger.info
        log_method(message)
        self._logged_events.add(key)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_memory': memory.total,
                'available_memory': memory.available,
                'used_memory': memory.used,
                'memory_percent': memory.percent
            }
        except Exception as exc:
            self._log_fallback_event(
                "hardware_memory_usage_fallback",
                f"psutil unavailable or failed with '{exc}'; using fallback memory metrics.",
            )
            return {
                'total_memory': 8.0 * 1024**3,  # 8GB fallback
                'available_memory': 4.0 * 1024**3,  # 4GB fallback
                'used_memory': 4.0 * 1024**3,  # 4GB fallback
                'memory_percent': 50.0
            }
    
    def optimize_memory(self, data: Any) -> Any:
        """Optimize memory usage (fallback implementation)."""
        try:
            if isinstance(data, np.ndarray):
                # Convert to float32 if possible
                if data.dtype == np.float64:
                    return data.astype(np.float32)
            elif isinstance(data, pd.DataFrame):
                # Optimize DataFrame dtypes
                for col in data.columns:
                    if data[col].dtype == 'float64':
                        data[col] = data[col].astype('float32')
            return data
        except Exception as exc:
            self._log_fallback_event(
                "hardware_optimize_memory_exception",
                f"Memory optimization failed with '{exc}'; returning original data.",
            )
            return data
    
    def memory_checkpoint(self, name: str = "fallback"):
        """Memory checkpoint context manager (fallback implementation)."""
        from contextlib import contextmanager
        
        @contextmanager
        def checkpoint_context():
            start_memory = self.get_memory_usage()['used_memory']
            try:
                yield
            finally:
                end_memory = self.get_memory_usage()['used_memory']
                if self.config.enable_logging:
                    self.logger.debug(f"Memory checkpoint {name}: {end_memory - start_memory:.2f} MB")
        
        return checkpoint_context()
    
    def gpu_context(self, name: str = "fallback"):
        """GPU context manager (fallback implementation)."""
        from contextlib import contextmanager
        
        @contextmanager
        def gpu_context_manager():
            start_time = time.time()
            self._log_fallback_event(
                "hardware_gpu_context_fallback",
                "GPU context executed with fallback implementation (no GPU acceleration).",
            )
            try:
                yield
            finally:
                duration = time.time() - start_time
                if self.config.enable_logging:
                    self.logger.debug(f"GPU context {name}: duration {duration:.2f}s (fallback)")

        return gpu_context_manager()


class FallbackSerializationUtils:
    """Fallback serialization utilities when advanced serialization is not available."""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_json(self, data: Any, filepath: Union[str, Path]) -> bool:
        """Save data as JSON (fallback implementation)."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")
            return False
    
    def load_json(self, filepath: Union[str, Path]) -> Optional[Any]:
        """Load data from JSON (fallback implementation)."""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return None
            
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON: {e}")
            return None
    
    def save_pickle(self, data: Any, filepath: Union[str, Path]) -> bool:
        """Save data as pickle (fallback implementation)."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save pickle: {e}")
            return False
    
    def load_pickle(self, filepath: Union[str, Path]) -> Optional[Any]:
        """Load data from pickle (fallback implementation)."""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return None
            
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load pickle: {e}")
            return None


class FallbackTPrintUtils:
    """Fallback tprint utilities when advanced logging is not available."""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def tprint(self, message: str, color: str = "white", bold: bool = False):
        """Fallback tprint implementation."""
        if self.config.enable_logging:
            self.logger.info(message)
        else:
            print(message)
    
    def tprint_info(self, message: str):
        """Fallback tprint_info implementation."""
        if self.config.enable_logging:
            self.logger.info(f"ℹ️ {message}")
        else:
            print(f"ℹ️ {message}")
    
    def tprint_warning(self, message: str):
        """Fallback tprint_warning implementation."""
        if self.config.enable_logging:
            self.logger.warning(f"⚠️ {message}")
        else:
            print(f"⚠️ {message}")
    
    def tprint_error(self, message: str):
        """Fallback tprint_error implementation."""
        if self.config.enable_logging:
            self.logger.error(f"❌ {message}")
        else:
            print(f"❌ {message}")
    
    def tprint_success(self, message: str):
        """Fallback tprint_success implementation."""
        if self.config.enable_logging:
            self.logger.info(f"✅ {message}")
        else:
            print(f"✅ {message}")


class FallbackOptimizationUtils:
    """Fallback optimization utilities when advanced optimization is not available."""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize_parameters(self, objective_function: Callable, 
                          parameter_space: Dict[str, Any], 
                          n_trials: int = 10) -> Dict[str, Any]:
        """Fallback parameter optimization."""
        try:
            best_params = None
            best_score = -np.inf
            
            for trial in range(n_trials):
                # Random parameter sampling
                params = {}
                for param_name, param_config in parameter_space.items():
                    if isinstance(param_config, dict):
                        if 'min' in param_config and 'max' in param_config:
                            params[param_name] = np.random.uniform(
                                param_config['min'], param_config['max']
                            )
                        elif 'choices' in param_config:
                            params[param_name] = np.random.choice(param_config['choices'])
                    else:
                        params[param_name] = param_config
                
                # Evaluate objective
                try:
                    score = objective_function(params)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                except Exception as e:
                    self.logger.warning(f"Trial {trial} failed: {e}")
                    continue
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': n_trials,
                'method': 'random_fallback'
            }
            
        except Exception as e:
            self.logger.error(f"Fallback optimization failed: {e}")
            return {
                'best_params': {},
                'best_score': self.config.fallback_accuracy,
                'n_trials': 0,
                'method': 'fallback_failed'
            }


class CentralizedFallbackUtils:
    """Centralized fallback utilities for NAS/TAS modules."""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize fallback utilities
        self.math_utils = FallbackMathUtils(config)
        self.hardware_utils = FallbackHardwareUtils(config)
        self.serialization_utils = FallbackSerializationUtils(config)
        self.tprint_utils = FallbackTPrintUtils(config)
        self.optimization_utils = FallbackOptimizationUtils(config)
        
        if self.config.enable_logging:
            self.logger.info("✅ Centralized fallback utilities initialized")
    
    def get_math_utils(self) -> FallbackMathUtils:
        """Get math utilities."""
        return self.math_utils
    
    def get_hardware_utils(self) -> FallbackHardwareUtils:
        """Get hardware utilities."""
        return self.hardware_utils
    
    def get_serialization_utils(self) -> FallbackSerializationUtils:
        """Get serialization utilities."""
        return self.serialization_utils
    
    def get_tprint_utils(self) -> FallbackTPrintUtils:
        """Get tprint utilities."""
        return self.tprint_utils
    
    def get_optimization_utils(self) -> FallbackOptimizationUtils:
        """Get optimization utilities."""
        return self.optimization_utils
    
    def create_fallback_result(self, result_type: str = "default", **kwargs) -> Any:
        """Create a fallback result object."""
        try:
            if result_type == "confidence":
                return type('FallbackConfidenceResult', (), {
                    'confidence_score': kwargs.get('confidence_score', self.config.fallback_accuracy),
                    'reliability_score': kwargs.get('reliability_score', self.config.fallback_robustness),
                    'calibration_score': kwargs.get('calibration_score', self.config.fallback_interpretability),
                    'method': 'fallback',
                    'is_fallback': True
                })()
            elif result_type == "uncertainty":
                return type('FallbackUncertaintyResult', (), {
                    'uncertainty_score': kwargs.get('uncertainty_score', 1.0 - self.config.fallback_accuracy),
                    'confidence_interval': kwargs.get('confidence_interval', [0.25, 0.75]),
                    'prediction_interval': kwargs.get('prediction_interval', [0.1, 0.9]),
                    'method': 'fallback',
                    'is_fallback': True
                })()
            else:
                return type('FallbackResult', (), {
                    'accuracy': kwargs.get('accuracy', self.config.fallback_accuracy),
                    'efficiency': kwargs.get('efficiency', self.config.fallback_efficiency),
                    'interpretability': kwargs.get('interpretability', self.config.fallback_interpretability),
                    'robustness': kwargs.get('robustness', self.config.fallback_robustness),
                    'method': 'fallback',
                    'is_fallback': True
                })()
        except Exception as e:
            self.logger.error(f"Failed to create fallback result: {e}")
            return None


# Global fallback utilities instance
_fallback_utils = None

def get_fallback_utils(config: Optional[FallbackConfig] = None) -> CentralizedFallbackUtils:
    """Get global fallback utilities instance."""
    global _fallback_utils
    if _fallback_utils is None:
        _fallback_utils = CentralizedFallbackUtils(config)
    return _fallback_utils

def create_fallback_config(**kwargs) -> FallbackConfig:
    """Create fallback configuration."""
    return FallbackConfig(**kwargs)