"""
Enhanced Matrix Operations for Perfect NAS Regime System

Integrates with utils/matrix_operations/ for optimized computations.
Now includes full integration with common utilities and hardware optimization.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from contextlib import contextmanager

# Import enhanced utilities
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, 
    validate_finite, validate_positive, validate_range,
    safe_mean, safe_std, safe_percentage_change,
    math_safe, timed_operation, format_bytes,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, memory_checkpoint, gpu_context
)
from src.utils.math_validation import (
    safe_correlation, safe_covariance, safe_percentile,
    validate_numeric_array, MathValidationError
)
from src.utils.serialization_utils import UniversalSerializer

# Import matrix operations with fallback
try:
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    from src.utils.matrix_operations.vectorized_core import get_vectorized_processing_core
    from src.utils.matrix_operations.batch_operations import BatchMatrixOperations
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedMatrixOperations:
    """
    Enhanced matrix operations for Perfect NAS Regime System.
    
    Integrates with existing matrix operations infrastructure for:
    - Optimized computations with common utilities
    - Hardware acceleration (M1 GPU/CPU)
    - Memory management and validation
    - Vectorized operations with safe math
    - Serialization and persistence
    """
    
    def __init__(self, enable_gpu: bool = True, enable_optimization: bool = True, 
                 enable_m1_optimization: bool = True):
        """Initialize enhanced matrix operations.
        
        Args:
            enable_gpu: Enable GPU acceleration
            enable_optimization: Enable optimization features
            enable_m1_optimization: Enable M1-specific optimizations
        """
        self.enable_gpu = enable_gpu
        self.enable_optimization = enable_optimization
        self.enable_m1_optimization = enable_m1_optimization
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize serialization
        self.serializer = UniversalSerializer()
        
        # Initialize M1 optimizations
        if enable_m1_optimization:
            try:
                self.m1_integration = integrate_with_m1_optimizers()
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ M1 optimizations initialized")
            except Exception as e:
                self.logger.warning(f"M1 optimization initialization failed: {e}")
                self.m1_integration = None
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.m1_integration = None
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        # Initialize unified matrix operations if available
        if MATRIX_OPS_AVAILABLE:
            try:
                self.matrix_ops = UnifiedMatrixOperations(
                    enable_gpu=enable_gpu,
                    enable_memory_optimization=True,
                    enable_parallel=True
                )
                self.vectorized_core = get_vectorized_processing_core()
                self.batch_ops = BatchMatrixOperations()
                self.hardware_ops = HardwareOptimizedOperations()
                self.logger.info("✅ Enhanced matrix operations initialized with full integration")
            except Exception as e:
                self.logger.warning(f"Matrix operations initialization failed: {e}")
                self.matrix_ops = None
                self.vectorized_core = None
                self.batch_ops = None
                self.hardware_ops = None
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.batch_ops = None
            self.hardware_ops = None
            self.logger.warning("Matrix operations not available - using fallback implementations")
    
    @timed_operation
    def normalize_data(self, data: np.ndarray, method: str = 'z_score') -> np.ndarray:
        """Normalize data using optimized operations with safe math validation."""
        try:
            # Validate input data
            validate_numeric_array(data, "input_data")
            
            # Use memory checkpoint for large operations
            with memory_checkpoint(f"normalize_data_{method}"):
                if self.matrix_ops:
                    return self.matrix_ops.normalize_data(data, method=method)
                else:
                    # Enhanced fallback normalization with safe math
                    if method == 'z_score':
                        mean_vals = safe_mean(data)
                        std_vals = safe_std(data)
                        return safe_divide(data - mean_vals, std_vals + 1e-8)
                    elif method == 'min_max':
                        min_vals = np.min(data, axis=0)
                        max_vals = np.max(data, axis=0)
                        range_vals = max_vals - min_vals + 1e-8
                        return safe_divide(data - min_vals, range_vals)
                    elif method == 'robust':
                        # Robust normalization using median and IQR
                        median_vals = np.median(data, axis=0)
                        q75 = np.percentile(data, 75, axis=0)
                        q25 = np.percentile(data, 25, axis=0)
                        iqr = q75 - q25 + 1e-8
                        return safe_divide(data - median_vals, iqr)
                    else:
                        return data
        except Exception as e:
            self.logger.warning(f"Data normalization failed: {e}")
            return data
    
    @timed_operation
    def calculate_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix with optimizations and safe math."""
        try:
            # Validate input data
            validate_numeric_array(data, "correlation_data")
            
            # Use GPU context if available
            with gpu_context("correlation_calculation"):
                if self.matrix_ops:
                    return self.matrix_ops.calculate_correlation_matrix(data)
                else:
                    # Enhanced fallback correlation calculation with safe math
                    if data.shape[1] == 1:
                        return np.array([[1.0]])
                    
                    # Calculate pairwise correlations safely
                    n_features = data.shape[1]
                    corr_matrix = np.eye(n_features)
                    
                    for i in range(n_features):
                        for j in range(i + 1, n_features):
                            corr = safe_correlation(data[:, i], data[:, j])
                            corr_matrix[i, j] = corr
                            corr_matrix[j, i] = corr
                    
                    return corr_matrix
        except Exception as e:
            self.logger.warning(f"Correlation matrix calculation failed: {e}")
            return np.eye(data.shape[1])
    
    def calculate_regime_stability(self, regime_predictions: np.ndarray, 
                                  timestamps: np.ndarray) -> np.ndarray:
        """Calculate regime stability with optimizations."""
        try:
            if self.matrix_ops:
                return self.matrix_ops.calculate_regime_stability(regime_predictions, timestamps)
            else:
                # Fallback stability calculation
                stability_scores = np.zeros(len(regime_predictions))
                
                for i in range(len(regime_predictions)):
                    current_regime = regime_predictions[i]
                    
                    # Look ahead and behind for regime consistency
                    lookback = min(10, i)
                    lookahead = min(10, len(regime_predictions) - i - 1)
                    
                    if lookback > 0:
                        past_regimes = regime_predictions[i-lookback:i]
                        past_consistency = np.mean(past_regimes == current_regime)
                    else:
                        past_consistency = 1.0
                    
                    if lookahead > 0:
                        future_regimes = regime_predictions[i+1:i+1+lookahead]
                        future_consistency = np.mean(future_regimes == current_regime)
                    else:
                        future_consistency = 1.0
                    
                    stability_scores[i] = (past_consistency + future_consistency) / 2.0
                
                return stability_scores
                
        except Exception as e:
            self.logger.warning(f"Regime stability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def calculate_transition_probabilities(self, regime_predictions: np.ndarray, 
                                         n_regimes: int) -> np.ndarray:
        """Calculate regime transition probabilities with optimizations."""
        try:
            if self.matrix_ops:
                return self.matrix_ops.calculate_transition_probabilities(regime_predictions, n_regimes)
            else:
                # Fallback transition calculation
                transition_matrix = np.zeros((n_regimes, n_regimes))
                
                for i in range(len(regime_predictions) - 1):
                    current_regime = regime_predictions[i]
                    next_regime = regime_predictions[i + 1]
                    transition_matrix[current_regime, next_regime] += 1
                
                # Normalize transition matrix
                row_sums = transition_matrix.sum(axis=1)
                transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
                
                return transition_matrix
                
        except Exception as e:
            self.logger.warning(f"Transition probability calculation failed: {e}")
            return np.eye(n_regimes) / n_regimes
    
    def calculate_volatility_features(self, data: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate volatility features with optimizations."""
        try:
            if self.matrix_ops:
                return self.matrix_ops.calculate_volatility_features(data, window)
            else:
                # Fallback volatility calculation
                volatility_features = np.zeros_like(data)
                
                for i in range(window, len(data)):
                    window_data = data[i-window:i]
                    volatility_features[i] = np.std(window_data, axis=0)
                
                return volatility_features
                
        except Exception as e:
            self.logger.warning(f"Volatility features calculation failed: {e}")
            return np.zeros_like(data)
    
    def calculate_momentum_features(self, data: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate momentum features with optimizations."""
        try:
            if self.matrix_ops:
                return self.matrix_ops.calculate_momentum_features(data, window)
            else:
                # Fallback momentum calculation
                momentum_features = np.zeros_like(data)
                
                for i in range(window, len(data)):
                    current_data = data[i]
                    past_data = data[i-window]
                    momentum_features[i] = current_data - past_data
                
                return momentum_features
                
        except Exception as e:
            self.logger.warning(f"Momentum features calculation failed: {e}")
            return np.zeros_like(data)
    
    def calculate_technical_indicators(self, data: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        """Calculate technical indicators with optimizations."""
        try:
            if self.matrix_ops:
                return self.matrix_ops.calculate_technical_indicators(data, window)
            else:
                # Fallback technical indicators
                indicators = {}
                
                # Simple moving average
                sma = np.zeros_like(data)
                for i in range(window, len(data)):
                    sma[i] = np.mean(data[i-window:i], axis=0)
                indicators['sma'] = sma
                
                # Relative strength index (simplified)
                rsi = np.zeros_like(data)
                for i in range(window, len(data)):
                    window_data = data[i-window:i]
                    gains = np.maximum(0, np.diff(window_data, axis=0))
                    losses = np.maximum(0, -np.diff(window_data, axis=0))
                    avg_gain = np.mean(gains, axis=0)
                    avg_loss = np.mean(losses, axis=0)
                    rs = avg_gain / (avg_loss + 1e-8)
                    rsi[i] = 100 - (100 / (1 + rs))
                indicators['rsi'] = rsi
                
                return indicators
                
        except Exception as e:
            self.logger.warning(f"Technical indicators calculation failed: {e}")
            return {}
    
    def optimize_for_inference(self):
        """Optimize matrix operations for inference."""
        try:
            if self.matrix_ops:
                self.matrix_ops.optimize_for_inference()
                self.logger.info("✅ Matrix operations optimized for inference")
        except Exception as e:
            self.logger.warning(f"Inference optimization failed: {e}")
    
    def optimize_for_training(self):
        """Optimize matrix operations for training."""
        try:
            if self.matrix_ops:
                self.matrix_ops.optimize_for_training()
                self.logger.info("✅ Matrix operations optimized for training")
        except Exception as e:
            self.logger.warning(f"Training optimization failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from matrix operations."""
        try:
            if self.matrix_ops:
                return {
                    'operations_count': getattr(self.matrix_ops, 'operations_count', 0),
                    'optimization_level': 'aggressive',
                    'gpu_acceleration': self.enable_gpu,
                    'memory_optimization': True,
                    'parallel_processing': True
                }
            else:
                return {
                    'operations_count': 0,
                    'optimization_level': 'fallback',
                    'gpu_acceleration': False,
                    'memory_optimization': False,
                    'parallel_processing': False
                }
        except Exception as e:
            self.logger.warning(f"Performance metrics collection failed: {e}")
            return {}
    
    @contextmanager
    def optimization_context(self, context_type: str = 'inference'):
        """Context manager for optimization."""
        try:
            if context_type == 'inference':
                self.optimize_for_inference()
            elif context_type == 'training':
                self.optimize_for_training()
            
            yield
            
        finally:
            # Cleanup if needed
            pass
    
    def calculate_enhanced_features(self, data: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        """Calculate enhanced features using safe math operations."""
        try:
            validate_numeric_array(data, "feature_data")
            
            features = {}
            
            # Volatility features
            features['volatility'] = self.calculate_volatility_features(data, window)
            
            # Momentum features
            features['momentum'] = self.calculate_momentum_features(data, window)
            
            # Technical indicators
            technical_indicators = self.calculate_technical_indicators(data, window)
            features.update(technical_indicators)
            
            # Statistical features
            features['skewness'] = self._calculate_rolling_skewness(data, window)
            features['kurtosis'] = self._calculate_rolling_kurtosis(data, window)
            
            # Regime-specific features
            features['regime_strength'] = self._calculate_regime_strength(data, window)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Enhanced feature calculation failed: {e}")
            return {}
    
    def _calculate_rolling_skewness(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling skewness with safe math."""
        try:
            skewness = np.zeros_like(data)
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                mean_val = safe_mean(window_data)
                std_val = safe_std(window_data)
                if std_val > 0:
                    normalized = safe_divide(window_data - mean_val, std_val)
                    skewness[i] = safe_mean(normalized ** 3)
            return skewness
        except Exception as e:
            self.logger.warning(f"Rolling skewness calculation failed: {e}")
            return np.zeros_like(data)
    
    def _calculate_rolling_kurtosis(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling kurtosis with safe math."""
        try:
            kurtosis = np.zeros_like(data)
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                mean_val = safe_mean(window_data)
                std_val = safe_std(window_data)
                if std_val > 0:
                    normalized = safe_divide(window_data - mean_val, std_val)
                    kurtosis[i] = safe_mean(normalized ** 4) - 3  # Excess kurtosis
            return kurtosis
        except Exception as e:
            self.logger.warning(f"Rolling kurtosis calculation failed: {e}")
            return np.zeros_like(data)
    
    def _calculate_regime_strength(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime strength indicator."""
        try:
            strength = np.zeros_like(data)
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                # Calculate consistency within window
                mean_val = safe_mean(window_data)
                deviations = np.abs(window_data - mean_val)
                consistency = 1.0 / (1.0 + safe_mean(deviations))
                strength[i] = consistency
            return strength
        except Exception as e:
            self.logger.warning(f"Regime strength calculation failed: {e}")
            return np.ones_like(data)
    
    def save_operations_state(self, filepath: str) -> bool:
        """Save current operations state using serialization utils."""
        try:
            state = {
                'matrix_ops_available': MATRIX_OPS_AVAILABLE,
                'm1_integration': self.m1_integration,
                'performance_metrics': self.get_performance_metrics(),
                'optimization_settings': {
                    'enable_gpu': self.enable_gpu,
                    'enable_optimization': self.enable_optimization,
                    'enable_m1_optimization': self.enable_m1_optimization
                }
            }
            
            return self.serializer.save(state, filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save operations state: {e}")
            return False
    
    def load_operations_state(self, filepath: str) -> bool:
        """Load operations state using serialization utils."""
        try:
            state = self.serializer.load(filepath)
            if state is None:
                return False
            
            # Restore settings if available
            if 'optimization_settings' in state:
                settings = state['optimization_settings']
                self.enable_gpu = settings.get('enable_gpu', True)
                self.enable_optimization = settings.get('enable_optimization', True)
                self.enable_m1_optimization = settings.get('enable_m1_optimization', True)
            
            self.logger.info("✅ Operations state loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load operations state: {e}")
            return False