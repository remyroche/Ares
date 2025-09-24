from src.utils.tprint import tprint

"""
Base Feature Selection Framework

This module provides the core base class for the feature selection framework,
handling initialization, configuration, and common utilities.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import logging
from functools import partial

from src.utils.unified_cache import get_unified_cache
from concurrent.futures import ThreadPoolExecutor
import warnings
import time
import sys

# Import utilities from the original location
try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        safe_correlation, safe_covariance, safe_mean, safe_std, safe_percentile
    )
    from src.utils.common_operations import create_fallback_logger, safe_dataframe_operation
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.parallel_processing_optimizer import ParallelProcessor
    from src.utils.matrix_operations import (
        safe_correlation_matrix, safe_matrix_multiply, get_unified_matrix_operations
    )
    from src.utils.performance_utils import PerformanceMonitor, performance_timer, memory_monitor
    from src.utils.ml_common.utils.memory_optimization import M1MemoryOptimizer, MemoryEfficientProcessor

    from src.utils.ml_common.validation.validation_utils import validate_data_quality, validate_feature_matrix
    from src.utils.ml_common.validation.stability import StabilityAnalyzer
    from src.utils.ml_common.validation.thresholding import AdaptiveThresholding
except ImportError as e:
    tprint(f"⚠️ Some utilities not available: {e}")
    # Create fallback implementations
    def safe_divide(a, b): return a / b if b != 0 else 0
    def safe_log(x): return np.log(np.maximum(x, 1e-10))
    def safe_sqrt(x): return np.sqrt(np.maximum(x, 0))
    def safe_power(x, p): return np.power(np.maximum(x, 0), p)
    def validate_finite(x): return np.isfinite(x).all()
    def safe_correlation(x, y): return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    def safe_covariance(x, y): return np.cov(x, y)[0, 1] if len(x) > 1 else 0
    def safe_mean(x): return np.mean(x) if len(x) > 0 else 0
    def safe_std(x): return np.std(x) if len(x) > 1 else 0
    def safe_percentile(x, p): return np.percentile(x, p) if len(x) > 0 else 0

    # Create fallback classes for failed imports
    class PerformanceMonitor:
        def __init__(self, max_history=1000):
            self.max_history = max_history
            self.history = []
            self.logger = logging.getLogger(__name__)

        def log_performance(self, operation, time_taken, memory_used=None):
            self.history.append({
                'operation': operation,
                'time': time_taken,
                'memory': memory_used,
                'timestamp': time.time()
            })
            if len(self.history) > self.max_history:
                self.history.pop(0)

        def get_stats(self):
            return {'total_operations': len(self.history)}

    class M1GPUManager:
        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def detect_m1(self):
            return False

        def check_mps_availability(self):
            return False

    def performance_timer():
        return lambda func: func

    def memory_monitor():
        return lambda func: func

    class M1MemoryOptimizer:
        pass

    class MemoryEfficientProcessor:
        pass

    def validate_data_quality(data):
        return True

    def validate_feature_matrix(X, y=None):
        return True

    class StabilityAnalyzer:
        pass

    class AdaptiveThresholding:
        pass

    class ParallelProcessor:
        def __init__(self, max_workers=4, chunk_size=10000):
            self.max_workers = max_workers
            self.chunk_size = chunk_size
            self.logger = logging.getLogger(__name__)

        def process_parallel(self, func, items, *args, **kwargs):
            """Process items in parallel using ThreadPoolExecutor"""
            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(func, item, *args, **kwargs) for item in items]
                    results = [future.result() for future in futures]
                return results
            except Exception as e:
                self.logger.warning(f"Parallel processing failed: {e}, falling back to sequential")
                return [func(item, *args, **kwargs) for item in items]

# Enhanced dependency management with fast fail
try:
    from src.utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.BaseFramework")
    tprint("✅ Custom logger available for FeatureSelection.BaseFramework")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("FeatureSelection.BaseFramework")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr, spearmanr
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited feature selection functionality")


class BaseFeatureSelectionFramework:
    """Base class for feature selection framework with core functionality."""
    
    # Model-specific optimal feature counts
    MODEL_FEATURE_TARGETS = {
        # Linear models - work well with moderate feature counts
        'linear_regression': 60,
        'ridge_regression': 80,
        'lasso_regression': 50,
        'elastic_net': 70,
        'logistic_regression': 60,
        
        # Tree-based models - can handle more features
        'random_forest': 100,
        'gradient_boosting': 120,
        'xgboost': 100,
        'lightgbm': 100,
        'catboost': 100,
        'extra_trees': 100,
        
        # SVM models - sensitive to feature count
        'svm_linear': 50,
        'svm_rbf': 80,
        'svm_poly': 60,
        
        # Neural networks - can handle many features
        'neural_network': 150,
        'deep_learning': 200,
        
        # Ensemble methods
        'voting_classifier': 100,
        'stacking_classifier': 120,
        'bagging_classifier': 100,
        
        # Default fallback
        'default': 80
    }
    
    # Minimum feature count for intermediate stages
    MIN_FEATURES_INTERMEDIATE = 100

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base feature selection framework with comprehensive optimization tools."""
        self.config = config or {}
        self.logger = logger.getChild('BaseFramework')
        
        _LOGGER.info("🚀 Initializing BaseFeatureSelectionFramework with comprehensive optimizations...")

        # Configuration defaults
        self.enable_gpu = self.config.get('enable_gpu', True)
        self.enable_parallel = self.config.get('enable_parallel', True)
        self.max_workers = self.config.get('max_workers', 4)
        
        # Initialize comprehensive optimization tools
        self._initialize_optimization_tools()
        self.memory_threshold = self.config.get('memory_threshold', 0.8)
        self.random_state = self.config.get('random_state', 42)

        _LOGGER.info(f"⚙️ Configuration - GPU enabled: {self.enable_gpu}")
        _LOGGER.info(f"⚙️ Configuration - Parallel processing: {self.enable_parallel}")
        _LOGGER.info(f"⚙️ Configuration - Max workers: {self.max_workers}")
        _LOGGER.info(f"⚙️ Configuration - Memory threshold: {self.memory_threshold}")
        _LOGGER.info(f"⚙️ Configuration - Random state: {self.random_state}")

        # Initialize utilities
        self.gpu_manager = M1GPUManager() if self.enable_gpu else None
        self.parallel_processor = ParallelProcessor() if self.enable_parallel else None

        # Method configurations
        _LOGGER.debug("🔧 Initializing method configurations...")
        self.method_configs = {
            'mrmr': {
                'relevance_method': 'mutual_info',
                'redundancy_method': 'correlation',
                'n_neighbors': 3
            },
            'importance': {
                'n_estimators': 100,
                'max_depth': 10,
                'bootstrap': True
            },
            'rfe': {
                'step': 0.1,
                'cv': 3,
                'scoring': 'accuracy'
            },
            'stability': {
                'n_bootstraps': 50,
                'bootstrap_fraction': 0.8,
                'stability_threshold': 0.6
            },
            'lasso': {
                'alpha_range': (0.001, 1.0),
                'cv_folds': 5,
                'max_iter': 1000,
                'tol': 1e-4,
                'random_state': 42
            },
            'elastic_net_stability': {
                'n_bootstraps': 50,  # Capped at 50
                'bootstrap_fraction': 0.8,
                'stability_threshold': 0.6,
                'alpha_range': (0.001, 1.0),
                'l1_ratio_range': (0.1, 0.9),  # Balance between L1 and L2 regularization
                'cv_folds': 5
            },
            'tree_ensemble': {
                'cv_folds': 5,
                'permutation_importance_repeats': 10,
                'correlation_threshold': 0.8,
                'hyperparameter_search': True,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None]
                },
                'random_state': 42
            }
        }

        # Update with user config
        if 'method_configs' in self.config:
            _LOGGER.debug("🔧 Updating method configurations with user config...")
            self.method_configs.update(self.config['method_configs'])
        
        _LOGGER.info("✅ BaseFeatureSelectionFramework initialized successfully")

    def _initialize_optimization_tools(self):
        """Initialize comprehensive optimization tools from src/utils/ and src/utils/ml_common/."""
        try:
            # Performance monitoring
            self.performance_monitor = PerformanceMonitor(max_history=1000)
            _LOGGER.info("📊 PerformanceMonitor initialized")
            
            # Memory optimization
            self.memory_optimizer = M1MemoryOptimizer()
            self.memory_processor = MemoryEfficientProcessor()
            _LOGGER.info("🧠 Memory optimization tools initialized")
            
            # Caching and shared resources (Unified)
            self.shared_cache = get_unified_cache(namespace="ml_common_feature_selection")
            _LOGGER.info("💾 Unified cache initialized for feature selection")
            
            # Stability and thresholding
            self.stability_analyzer = StabilityAnalyzer()
            self.adaptive_thresholding = AdaptiveThresholding()
            _LOGGER.info("📈 Stability and thresholding tools initialized")
            
            # Setup optimization settings
            self._setup_optimization_settings()
            
            # Add comprehensive optimization hooks
            self._add_safe_math_operations()
            self._add_memory_optimization_hooks()
            self._add_performance_monitoring_hooks()
            self._add_caching_hooks()
            
            # Enhance all existing methods
            self._enhance_existing_methods()
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Some optimization tools failed to initialize: {e}")
            # Initialize fallback tools
            self.performance_monitor = None
            self.memory_optimizer = None
            self.memory_processor = None
            self.shared_cache = None
            self.stability_analyzer = None
            self.adaptive_thresholding = None

    def _setup_optimization_settings(self):
        """Setup optimization-specific settings with validation."""
        # Core optimization settings
        self.cache_enabled = self._validate_boolean_setting('cache_enabled', True)
        self.memory_efficient_mode = self._validate_boolean_setting('memory_efficient_mode', True)
        self.performance_monitoring = self._validate_boolean_setting('performance_monitoring', True)
        self.stability_analysis = self._validate_boolean_setting('stability_analysis', True)
        
        # Memory management settings with validation
        self.chunk_size = self._validate_positive_int('chunk_size', 10000, min_val=1000, max_val=100000)
        self.memory_limit_gb = self._validate_positive_float('memory_limit_gb', 8.0, min_val=1.0, max_val=128.0)
        self.gc_frequency = self._validate_positive_int('gc_frequency', 100, min_val=10, max_val=1000)
        
        # Additional production settings
        self.max_features_per_method = self._validate_positive_int('max_features_per_method', 1000, min_val=10, max_val=10000)
        self.min_samples_for_analysis = self._validate_positive_int('min_samples_for_analysis', 10, min_val=5, max_val=1000)
        self.correlation_threshold = self._validate_float_range('correlation_threshold', 0.95, 0.0, 1.0)
        self.stability_threshold = self._validate_float_range('stability_threshold', 0.6, 0.0, 1.0)
        
        _LOGGER.info(f"⚙️ Optimization settings - Cache: {self.cache_enabled}")
        _LOGGER.info(f"⚙️ Optimization settings - Memory efficient: {self.memory_efficient_mode}")
        _LOGGER.info(f"⚙️ Optimization settings - Performance monitoring: {self.performance_monitoring}")
        _LOGGER.info(f"⚙️ Optimization settings - Chunk size: {self.chunk_size}")
        _LOGGER.info(f"⚙️ Optimization settings - Memory limit: {self.memory_limit_gb} GB")

    def _validate_boolean_setting(self, key: str, default: bool) -> bool:
        """Validate boolean configuration setting."""
        value = self.config.get(key, default)
        if not isinstance(value, bool):
            _LOGGER.warning(f"⚠️ Invalid {key}: {value}, using default: {default}")
            return default
        return value

    def _validate_positive_int(self, key: str, default: int, min_val: int = 1, max_val: int = None) -> int:
        """Validate positive integer configuration setting."""
        value = self.config.get(key, default)
        if not isinstance(value, int) or value < min_val:
            _LOGGER.warning(f"⚠️ Invalid {key}: {value}, using default: {default}")
            return default
        if max_val is not None and value > max_val:
            _LOGGER.warning(f"⚠️ {key} too large: {value}, using default: {default}")
            return default
        return value

    def _validate_positive_float(self, key: str, default: float, min_val: float = 0.0, max_val: float = None) -> float:
        """Validate positive float configuration setting."""
        value = self.config.get(key, default)
        if not isinstance(value, (int, float)) or value < min_val:
            _LOGGER.warning(f"⚠️ Invalid {key}: {value}, using default: {default}")
            return default
        if max_val is not None and value > max_val:
            _LOGGER.warning(f"⚠️ {key} too large: {value}, using default: {default}")
            return default
        return value

    def _validate_float_range(self, key: str, default: float, min_val: float, max_val: float) -> float:
        """Validate float configuration setting within range."""
        value = self.config.get(key, default)
        if not isinstance(value, (int, float)) or value < min_val or value > max_val:
            _LOGGER.warning(f"⚠️ Invalid {key}: {value}, using default: {default}")
            return default
        return value

    def _optimize_method_execution(self, method_name: str, func: callable, *args, **kwargs):
        """Optimize method execution with comprehensive monitoring and caching."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Check cache first
            if self.cache_enabled and hasattr(self, 'shared_cache'):
                cache_key = self._generate_cache_key(method_name, args, kwargs)
                try:
                    cached_result = self.shared_cache.get(cache_key)
                    if cached_result is not None:
                        _LOGGER.debug(f"🎯 Cache hit for {method_name}")
                        return cached_result
                except Exception as e:
                    _LOGGER.debug(f"Cache lookup failed: {e}")
            
            # Execute method with monitoring
            if self.performance_monitoring and hasattr(self, 'performance_monitor'):
                with self.performance_monitor.monitor(method_name):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            if self.cache_enabled and hasattr(self, 'shared_cache'):
                try:
                    self.shared_cache.set(cache_key, result)
                    _LOGGER.debug(f"💾 Cached result for {method_name}")
                except Exception as e:
                    _LOGGER.debug(f"Cache storage failed: {e}")
            
            # Log performance
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory
            
            _LOGGER.info(f"⏱️ {method_name} completed in {execution_time:.3f}s, memory delta: {memory_delta:.2f}MB")
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in {method_name}: {e}")
            raise

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _generate_cache_key(self, method_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for method execution."""
        import hashlib
        key_data = f"{method_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _safe_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Safely compute correlation between two arrays."""
        try:
            return safe_correlation(x, y)
        except Exception as e:
            _LOGGER.warning(f"⚠️ Correlation computation failed: {e}")
            return 0.0

    def _safe_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Safely compute mutual information between two arrays."""
        try:
            if SKLEARN_AVAILABLE and len(x) > 1:
                return mutual_info_regression(x.reshape(-1, 1), y)[0]
            return 0.0
        except Exception as e:
            _LOGGER.warning(f"⚠️ Mutual information computation failed: {e}")
            return 0.0

    def _memory_efficient_correlation_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute correlation matrix with memory efficiency."""
        try:
            if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
                return self.memory_optimizer.compute_correlation_matrix(X)
            else:
                return np.corrcoef(X.T)
        except Exception as e:
            _LOGGER.warning(f"⚠️ Memory efficient correlation failed: {e}")
            return np.corrcoef(X.T)

    def _adaptive_threshold_selection(self, scores: Dict[str, float], 
                                    method: str = 'percentile') -> float:
        """Select adaptive threshold based on score distribution."""
        try:
            if hasattr(self, 'adaptive_thresholding') and self.adaptive_thresholding:
                return self.adaptive_thresholding.select_threshold(scores, method)
            else:
                # Fallback to simple percentile
                score_values = list(scores.values())
                return np.percentile(score_values, 75)
        except Exception as e:
            _LOGGER.warning(f"⚠️ Adaptive threshold selection failed: {e}")
            return 0.5

    def _validate_data_quality(self, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Validate data quality and return quality metrics."""
        try:
            if hasattr(self, 'stability_analyzer') and self.stability_analyzer:
                return self.stability_analyzer.validate_data_quality(X, y)
            else:
                # Fallback validation
                return {
                    'is_valid': True,
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'has_nan': np.isnan(X).any(),
                    'has_inf': np.isinf(X).any(),
                    'constant_features': self._detect_constant_features(X)
                }
        except Exception as e:
            _LOGGER.warning(f"⚠️ Data quality validation failed: {e}")
            return {'is_valid': False, 'error': str(e)}

    def _detect_constant_features(self, X: np.ndarray) -> List[int]:
        """Detect constant features in the dataset."""
        try:
            constant_features = []
            for i in range(X.shape[1]):
                if np.std(X[:, i]) == 0:
                    constant_features.append(i)
            return constant_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ Constant feature detection failed: {e}")
            return []

    def _detect_high_correlation_features(self, X: np.ndarray, threshold: float = 0.99) -> List[Tuple[int, int, float]]:
        """Detect highly correlated feature pairs."""
        try:
            corr_matrix = self._memory_efficient_correlation_matrix(X)
            high_corr_pairs = []
            
            for i in range(corr_matrix.shape[0]):
                for j in range(i + 1, corr_matrix.shape[1]):
                    corr = abs(corr_matrix[i, j])
                    if corr > threshold:
                        high_corr_pairs.append((i, j, corr))
            
            return high_corr_pairs
        except Exception as e:
            _LOGGER.warning(f"⚠️ High correlation detection failed: {e}")
            return []

    def _detect_suspicious_target_correlations(self, X: np.ndarray, y: np.ndarray, 
                                             threshold: float = 0.99) -> List[Tuple[int, float]]:
        """Detect suspiciously high correlations with target."""
        try:
            suspicious_features = []
            for i in range(X.shape[1]):
                corr = abs(self._safe_correlation(X[:, i], y))
                if corr > threshold:
                    suspicious_features.append((i, corr))
            return suspicious_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ Suspicious correlation detection failed: {e}")
            return []

    def _detect_nan_inf_features(self, X: np.ndarray) -> List[int]:
        """Detect features with NaN or Inf values."""
        try:
            problematic_features = []
            for i in range(X.shape[1]):
                if np.isnan(X[:, i]).any() or np.isinf(X[:, i]).any():
                    problematic_features.append(i)
            return problematic_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ NaN/Inf detection failed: {e}")
            return []

    def _detect_zero_variance_features(self, X: np.ndarray) -> List[int]:
        """Detect features with zero variance."""
        try:
            zero_var_features = []
            for i in range(X.shape[1]):
                if np.var(X[:, i]) == 0:
                    zero_var_features.append(i)
            return zero_var_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ Zero variance detection failed: {e}")
            return []

    def _detect_perfect_correlations(self, X: np.ndarray, threshold: float = 0.98) -> List[Tuple[int, int, float]]:
        """Detect perfectly correlated feature pairs."""
        return self._detect_high_correlation_features(X, threshold)

    def _detect_suspicious_mutual_information(self, X: np.ndarray, y: np.ndarray, 
                                            threshold: float = 0.99) -> List[Tuple[int, float]]:
        """Detect suspiciously high mutual information with target."""
        try:
            suspicious_features = []
            for i in range(X.shape[1]):
                mi = self._safe_mutual_information(X[:, i], y)
                if mi > threshold:
                    suspicious_features.append((i, mi))
            return suspicious_features
        except Exception as e:
            _LOGGER.warning(f"⚠️ Suspicious MI detection failed: {e}")
            return []

    def _enhance_existing_methods(self):
        """Enhance existing methods with optimization hooks."""
        try:
            # Get all methods of this class
            methods = [method for method in dir(self) if callable(getattr(self, method)) and not method.startswith('_')]
            
            for method_name in methods:
                original_method = getattr(self, method_name)
                
                def create_enhanced_wrapper(original_func, name):
                    def enhanced_wrapper(*args, **kwargs):
                        return self._optimize_method_execution(name, original_func, *args, **kwargs)
                    return enhanced_wrapper
                
                # Replace method with enhanced version
                setattr(self, method_name, create_enhanced_wrapper(original_method, method_name))
            
            _LOGGER.info(f"🔧 Enhanced {len(methods)} methods with optimization hooks")
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Method enhancement failed: {e}")

    def _add_safe_math_operations(self):
        """Add safe mathematical operations to the framework."""
        try:
            self.safe_divide = safe_divide
            self.safe_log = safe_log
            self.safe_sqrt = safe_sqrt
            self.safe_power = safe_power
            self.validate_finite = validate_finite
            _LOGGER.info("🔢 Safe math operations added")
        except Exception as e:
            _LOGGER.warning(f"⚠️ Safe math operations setup failed: {e}")

    def _add_memory_optimization_hooks(self):
        """Add memory optimization hooks."""
        try:
            if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
                def memory_optimization_hook():
                    self.memory_optimizer.optimize_memory_usage()
                
                self.memory_optimization_hook = memory_optimization_hook
                _LOGGER.info("🧠 Memory optimization hooks added")
        except Exception as e:
            _LOGGER.warning(f"⚠️ Memory optimization hooks setup failed: {e}")

    def _add_performance_monitoring_hooks(self):
        """Add performance monitoring hooks."""
        try:
            if hasattr(self, 'performance_monitor') and self.performance_monitor:
                def performance_hook(method_name, start_time, end_time, memory_usage):
                    self.performance_monitor.record_execution(method_name, end_time - start_time, memory_usage)
                
                self.performance_hook = performance_hook
                _LOGGER.info("📊 Performance monitoring hooks added")
        except Exception as e:
            _LOGGER.warning(f"⚠️ Performance monitoring hooks setup failed: {e}")

    def _add_caching_hooks(self):
        """Add caching hooks."""
        try:
            if hasattr(self, 'shared_cache') and self.shared_cache:
                def cache_hook(operation, cache_key, hit):
                    if hit:
                        _LOGGER.debug(f"🎯 Cache hit for {operation}")
                    else:
                        _LOGGER.debug(f"💾 Cache miss for {operation}")
                
                self.cache_hook = cache_hook
                _LOGGER.info("💾 Caching hooks added")
        except Exception as e:
            _LOGGER.warning(f"⚠️ Caching hooks setup failed: {e}")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            'framework_initialized': True,
            'gpu_enabled': self.enable_gpu,
            'parallel_enabled': self.enable_parallel,
            'max_workers': self.max_workers,
            'memory_threshold': self.memory_threshold,
            'random_state': self.random_state,
            'cache_enabled': getattr(self, 'cache_enabled', False),
            'memory_efficient_mode': getattr(self, 'memory_efficient_mode', False),
            'performance_monitoring': getattr(self, 'performance_monitoring', False),
            'stability_analysis': getattr(self, 'stability_analysis', False)
        }
        
        # Add performance monitor stats if available
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            stats['performance_stats'] = self.performance_monitor.get_stats()
        
        # Add memory optimizer stats if available
        if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
            stats['memory_stats'] = self.memory_optimizer.get_stats()
        
        # Add cache stats if available
        try:
            stats['cache_stats'] = self.shared_cache.get_stats() if hasattr(self, 'shared_cache') else {'status': 'disabled'}
        except Exception as e:
            stats['cache_stats'] = {'error': str(e)}
        
        return stats

    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements and capabilities."""
        requirements = {
            'python_version': sys.version,
            'numpy_available': True,
            'pandas_available': True,
            'sklearn_available': SKLEARN_AVAILABLE,
            'gpu_available': False,
            'memory_available_gb': 0,
            'cpu_count': 1
        }
        
        try:
            requirements['memory_available_gb'] = psutil.virtual_memory().available / (1024**3)
            requirements['cpu_count'] = psutil.cpu_count()
        except ImportError:
            pass
        
        try:
            if hasattr(self, 'gpu_manager') and self.gpu_manager:
                requirements['gpu_available'] = self.gpu_manager.is_available()
        except Exception:
            pass
        
        return requirements

    def generate_error_report(self, error_context: Dict[str, Any], requirements: Dict[str, Any] = None) -> str:
        """Generate comprehensive error report."""
        report = f"""
=== Feature Selection Framework Error Report ===
Timestamp: {datetime.now().isoformat()}
Error: {error_context.get('error', 'Unknown error')}

System Information:
- Python Version: {sys.version}
- NumPy Available: {requirements.get('numpy_available', False)}
- Pandas Available: {requirements.get('pandas_available', False)}
- Scikit-learn Available: {SKLEARN_AVAILABLE}

Framework Configuration:
- GPU Enabled: {self.enable_gpu}
- Parallel Processing: {self.enable_parallel}
- Max Workers: {self.max_workers}
- Memory Threshold: {self.memory_threshold}

Data Information:
- Data Shape: {error_context.get('data_shape', 'Unknown')}
- Feature Count: {error_context.get('feature_count', 'Unknown')}
- Sample Count: {error_context.get('sample_count', 'Unknown')}

Error Context:
{error_context.get('context', 'No additional context')}

Stack Trace:
{error_context.get('traceback', 'No stack trace available')}
"""
        return report

    def log_error_with_context(self, error_context: Dict[str, Any], level: str = "ERROR"):
        """Log error with comprehensive context."""
        requirements = self.check_system_requirements()
        error_report = self.generate_error_report(error_context, requirements)
        
        if level.upper() == "ERROR":
            _LOGGER.error(error_report)
        elif level.upper() == "WARNING":
            _LOGGER.warning(error_report)
        else:
            _LOGGER.info(error_report)

    def get_model_target_features(self, model_type: str) -> int:
        """Get target feature count for specific model type."""
        return self.MODEL_FEATURE_TARGETS.get(model_type, self.MODEL_FEATURE_TARGETS['default'])

    def _auto_detect_model_type(self, model: Any) -> str:
        """Auto-detect model type from model object."""
        if model is None:
            return 'default'
        
        model_name = str(type(model)).lower()
        
        if 'linear' in model_name or 'logistic' in model_name:
            return 'linear_regression' if 'regression' in model_name else 'logistic_regression'
        elif 'lasso' in model_name:
            return 'lasso_regression'
        elif 'elastic' in model_name:
            return 'elastic_net'
        elif 'randomforest' in model_name or 'random_forest' in model_name:
            return 'random_forest'
        elif 'gradient' in model_name or 'xgboost' in model_name:
            return 'gradient_boosting'
        elif 'svm' in model_name:
            return 'svm_linear'
        elif 'neural' in model_name or 'mlp' in model_name:
            return 'neural_network'
        else:
            return 'default'


# Alias for backward compatibility
FeatureSelectionFramework = BaseFeatureSelectionFramework