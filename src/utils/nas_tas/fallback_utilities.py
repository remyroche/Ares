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
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
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
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with fallback."""
        try:
            if denominator == 0 or not np.isfinite(denominator):
                return default
            result = numerator / denominator
            return result if np.isfinite(result) else default
        except Exception:
            return default
    
    def safe_log(self, value: float, default: float = 0.0) -> float:
        """Safe logarithm with fallback."""
        try:
            if value <= 0 or not np.isfinite(value):
                return default
            result = np.log(value)
            return result if np.isfinite(result) else default
        except Exception:
            return default
    
    def safe_sqrt(self, value: float, default: float = 0.0) -> float:
        """Safe square root with fallback."""
        try:
            if value < 0 or not np.isfinite(value):
                return default
            result = np.sqrt(value)
            return result if np.isfinite(result) else default
        except Exception:
            return default
    
    def safe_mean(self, values: Union[List[float], np.ndarray], default: float = 0.0) -> float:
        """Safe mean calculation with fallback."""
        try:
            if len(values) == 0:
                return default
            values = np.array(values)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return default
            return float(np.mean(values))
        except Exception:
            return default
    
    def safe_std(self, values: Union[List[float], np.ndarray], default: float = 0.0) -> float:
        """Safe standard deviation calculation with fallback."""
        try:
            if len(values) == 0:
                return default
            values = np.array(values)
            values = values[np.isfinite(values)]
            if len(values) <= 1:
                return default
            return float(np.std(values))
        except Exception:
            return default
    
    def safe_correlation(self, x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
        """Safe correlation calculation with fallback."""
        try:
            if len(x) != len(y) or len(x) <= 1:
                return default
            x, y = np.array(x), np.array(y)
            x, y = x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)]
            if len(x) <= 1:
                return default
            corr = np.corrcoef(x, y)[0, 1]
            return corr if np.isfinite(corr) else default
        except Exception:
            return default
    
    def safe_percentile(self, values: Union[List[float], np.ndarray], percentile: float, default: float = 0.0) -> float:
        """Safe percentile calculation with fallback."""
        try:
            if len(values) == 0:
                return default
            values = np.array(values)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return default
            return float(np.percentile(values, percentile))
        except Exception:
            return default


class FallbackHardwareUtils:
    """Fallback hardware utilities when M1 optimizations are not available."""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.memory_usage = 0.0
        self.gpu_available = False
    
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
        except Exception:
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
        except Exception:
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
            if self.config.enable_logging:
                self.logger.debug(f"GPU context {name}: GPU not available, using CPU")
            try:
                yield
            finally:
                if self.config.enable_logging:
                    self.logger.debug(f"GPU context {name} completed")
        
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