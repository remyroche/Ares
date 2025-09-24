"""
Enhanced Matrix Operations for Perfect NAS Regime System

Integrates with utils/matrix_operations/ for optimized computations.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from contextlib import contextmanager

# Import matrix operations with fallback
try:
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    from src.utils.matrix_operations.vectorized_core import get_vectorized_processing_core
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedMatrixOperations:
    """
    Enhanced matrix operations for Perfect NAS Regime System.
    
    Integrates with existing matrix operations infrastructure for:
    - Optimized computations
    - Hardware acceleration
    - Memory management
    - Vectorized operations
    """
    
    def __init__(self, enable_gpu: bool = True, enable_optimization: bool = True):
        """Initialize enhanced matrix operations.
        
        Args:
            enable_gpu: Enable GPU acceleration
            enable_optimization: Enable optimization features
        """
        self.enable_gpu = enable_gpu
        self.enable_optimization = enable_optimization
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize unified matrix operations if available
        if MATRIX_OPS_AVAILABLE:
            try:
                self.matrix_ops = UnifiedMatrixOperations(
                    enable_gpu=enable_gpu,
                    enable_memory_optimization=True,
                    enable_parallel_processing=True,
                    optimization_level='aggressive'
                )
                self.vectorized_core = get_vectorized_processing_core()
                self.logger.info("✅ Enhanced matrix operations initialized with full integration")
            except Exception as e:
                self.logger.warning(f"Matrix operations initialization failed: {e}")
                self.matrix_ops = None
                self.vectorized_core = None
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.logger.warning("Matrix operations not available - using fallback implementations")
    
    def normalize_data(self, data: np.ndarray, method: str = 'z_score') -> np.ndarray:
        """Normalize data using optimized operations."""
        try:
            if self.matrix_ops:
                return self.matrix_ops.normalize_data(data, method=method)
            else:
                # Fallback normalization
                if method == 'z_score':
                    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
                elif method == 'min_max':
                    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-8)
                else:
                    return data
        except Exception as e:
            self.logger.warning(f"Data normalization failed: {e}")
            return data
    
    def calculate_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix with optimizations."""
        try:
            if self.matrix_ops:
                return self.matrix_ops.calculate_correlation_matrix(data)
            else:
                # Fallback correlation calculation
                return np.corrcoef(data.T)
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