"""
VectorBT-Optimized Correlation Calculations for Feature Lookback Optimization.

This module provides high-performance correlation and mutual information calculations
using VectorBT's optimized C++ backend, replacing the custom vectorized operations
with significantly faster and more memory-efficient implementations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from functools import lru_cache

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    from vectorbt.indicators.basic import RSI, MA, BBANDS
    from vectorbt.generic import nb
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None
    RSI = None
    MA = None
    BBANDS = None
    nb = None

# Import VectorBT Rolling Optimizer and Unified Vectorization Manager
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, 
        get_vectorbt_rolling_optimizer,
        optimized_rolling_corr,
        optimized_rolling_cov
    )
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager,
        get_unified_vectorization_manager,
        OperationType,
        OptimizationStrategy as UnifiedOptimizationStrategy
    )
    VECTORBT_UTILS_AVAILABLE = True
except ImportError:
    VECTORBT_UTILS_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    optimized_rolling_corr = None
    optimized_rolling_cov = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    OperationType = None
    UnifiedOptimizationStrategy = None

from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug, tprint_info
from src.utils.logger import get_logger
from ..error_handling.error_handler import safe_operation, get_error_handler

logger = get_logger('VectorBTCorrelation')


@dataclass
class VectorBTCorrelationConfig:
    """Configuration for VectorBT correlation calculations."""
    use_gpu: bool = False
    batch_size: int = 1000
    memory_efficient: bool = True
    correlation_method: str = 'pearson'  # 'pearson', 'spearman', 'kendall'
    mi_approximation: bool = True
    parallel_processing: bool = True
    cache_size: int = 1000
    
    # Enhanced optimization settings
    use_rolling_optimizer: bool = True
    use_unified_vectorization: bool = True
    rolling_optimizer_config: Optional[Dict[str, Any]] = None
    unified_vectorization_config: Optional[Dict[str, Any]] = None


class VectorBTCorrelationCalculator:
    """
    High-performance correlation calculator using VectorBT.
    
    This class provides optimized correlation and mutual information calculations
    using VectorBT's C++ backend, offering significant performance improvements
    over custom NumPy implementations.
    """
    
    def __init__(self, config: Optional[VectorBTCorrelationConfig] = None):
        """Initialize VectorBT correlation calculator."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")
        
        self.config = config or VectorBTCorrelationConfig()
        self.logger = get_logger('VectorBTCorrelationCalculator')
        self.error_handler = get_error_handler()
        
        # Configure VectorBT settings
        self._configure_vectorbt()
        
        # Initialize caches
        self._correlation_cache = {}
        self._mi_cache = {}
        
        # Initialize enhanced optimization components
        self._initialize_enhanced_components()
        
        tprint_success("✅ VectorBT Correlation Calculator initialized with enhanced optimizations")
    
    def _configure_vectorbt(self):
        """Configure VectorBT global settings for optimal performance."""
        try:
            # Configure VectorBT for maximum performance
            vbt.settings.set_theme('dark')
            vbt.settings['array_wrapper']['freq_precision'] = 0
            vbt.settings['array_wrapper']['freq_shorten'] = True
            
            # Enable parallel processing if available
            if self.config.parallel_processing:
                vbt.settings['array_wrapper']['parallel'] = True
            
            # Configure memory settings
            if self.config.memory_efficient:
                vbt.settings['array_wrapper']['memory_efficient'] = True
            
            self.logger.debug("VectorBT configuration applied successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not configure VectorBT settings: {e}")
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced optimization components."""
        try:
            # Initialize VectorBT Rolling Optimizer
            if self.config.use_rolling_optimizer and VECTORBT_UTILS_AVAILABLE:
                rolling_config = self.config.rolling_optimizer_config or {}
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=rolling_config.get('enable_gpu', False),
                    enable_parallel=rolling_config.get('enable_parallel', True),
                    memory_efficient=rolling_config.get('memory_efficient', True)
                )
                self.logger.debug("VectorBT Rolling Optimizer initialized for correlation calculations")
            else:
                self.rolling_optimizer = None
                self.logger.warning("VectorBT Rolling Optimizer not available for correlation calculations")
            
            # Initialize Unified Vectorization Manager
            if self.config.use_unified_vectorization and VECTORBT_UTILS_AVAILABLE:
                unified_config = self.config.unified_vectorization_config or {}
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.debug("Unified Vectorization Manager initialized for correlation calculations")
            else:
                self.unified_manager = None
                self.logger.warning("Unified Vectorization Manager not available for correlation calculations")
            
            self.logger.debug("Enhanced optimization components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Enhanced component initialization failed: {e}")
            # Don't raise - these are optional enhancements
    
    @safe_operation
    def calculate_correlations_vectorbt(
        self, 
        features_list: List[np.ndarray], 
        returns_list: List[np.ndarray]
    ) -> List[float]:
        """
        Calculate correlations using VectorBT's optimized backend.
        
        Args:
            features_list: List of feature arrays
            returns_list: List of corresponding return arrays
            
        Returns:
            List of correlation coefficients
        """
        if not features_list or not returns_list:
            return []
        
        tprint_debug(f"🔄 Calculating correlations for {len(features_list)} pairs using VectorBT")
        start_time = time.time()
        
        try:
            # Use enhanced optimization if available
            if self.rolling_optimizer and self.unified_manager:
                correlations = self._calculate_correlations_enhanced(features_list, returns_list)
            else:
                # Convert to VectorBT arrays for optimized processing
                features_vbt = self._prepare_vectorbt_arrays(features_list)
                returns_vbt = self._prepare_vectorbt_arrays(returns_list)
                
                # Use VectorBT's optimized correlation calculation
                correlations = self._vectorbt_correlation_batch(features_vbt, returns_vbt)
                correlations = correlations.tolist()
            
            computation_time = time.time() - start_time
            tprint_success(f"✅ VectorBT correlation calculation completed in {computation_time:.3f}s")
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"VectorBT correlation calculation failed: {e}")
            # Fallback to individual calculations
            return self._fallback_correlation_calculation(features_list, returns_list)
    
    def _calculate_correlations_enhanced(
        self, 
        features_list: List[np.ndarray], 
        returns_list: List[np.ndarray]
    ) -> List[float]:
        """Calculate correlations using enhanced VectorBT optimizations."""
        try:
            correlations = []
            
            # Use Unified Vectorization Manager for intelligent optimization
            if self.unified_manager:
                # Configure operation for correlation calculation
                operation_config = self.unified_manager.create_operation_config(
                    operation_type=OperationType.STATISTICAL_COMPUTATION,
                    data_size=len(features_list[0]) if features_list else 0,
                    data_dimensions=(len(features_list[0]),) if features_list else (0,),
                    memory_budget_mb=512.0,
                    time_budget_seconds=30.0
                )
                
                # Get optimal strategy
                strategy = self.unified_manager.select_optimization_strategy(operation_config)
                self.logger.debug(f"Selected correlation strategy: {strategy}")
            
            for i, (feature, returns) in enumerate(zip(features_list, returns_list)):
                try:
                    # Use VectorBTRollingOptimizer for rolling correlation
                    if self.rolling_optimizer and len(feature) > 50:
                        # Calculate rolling correlation for more robust results
                        rolling_corr = self.rolling_optimizer.rolling_corr(
                            pd.Series(feature), 
                            pd.Series(returns), 
                            window=min(20, len(feature) // 4)
                        )
                        
                        # Use mean of rolling correlations
                        valid_corr = rolling_corr.dropna()
                        if len(valid_corr) > 0:
                            corr = valid_corr.mean()
                        else:
                            # Fallback to simple correlation
                            corr = self._calculate_single_correlation_vectorbt(feature, returns)
                    else:
                        # Use standard VectorBT correlation
                        corr = self._calculate_single_correlation_vectorbt(feature, returns)
                    
                    correlations.append(corr)
                    
                except Exception as e:
                    self.logger.warning(f"Enhanced correlation calculation failed for feature {i}: {e}")
                    # Fallback to standard method
                    corr = self._calculate_single_correlation_vectorbt(feature, returns)
                    correlations.append(corr)
            
            return correlations
            
        except Exception as e:
            self.logger.warning(f"Enhanced correlation calculation failed: {e}")
            # Fallback to standard method
            return self._fallback_correlation_calculation(features_list, returns_list)
    
    def _prepare_vectorbt_arrays(self, arrays_list: List[np.ndarray]) -> np.ndarray:
        """Prepare arrays for VectorBT processing."""
        if not arrays_list:
            return np.array([])
        
        # Find the minimum length for alignment
        min_length = min(len(arr) for arr in arrays_list if len(arr) > 0)
        
        if min_length < 10:
            return np.array([])
        
        # Align all arrays to the same length
        aligned_arrays = []
        for arr in arrays_list:
            if len(arr) >= min_length:
                aligned_arrays.append(arr[:min_length])
        
        if not aligned_arrays:
            return np.array([])
        
        # Convert to 2D array for batch processing
        return np.array(aligned_arrays)
    
    def _vectorbt_correlation_batch(self, features: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Calculate correlations using VectorBT's batch processing."""
        if features.size == 0 or returns.size == 0:
            return np.array([])
        
        try:
            # Use VectorBT's optimized correlation function
            if self.config.correlation_method == 'pearson':
                correlations = vbt.generic.nb.corr_1d_2d(features, returns)
            elif self.config.correlation_method == 'spearman':
                # For Spearman, we need to rank the data first
                features_ranked = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, features)
                returns_ranked = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, returns)
                correlations = vbt.generic.nb.corr_1d_2d(features_ranked, returns_ranked)
            else:  # kendall
                # Kendall correlation is more complex, fall back to scipy
                from scipy.stats import kendalltau
                correlations = np.array([
                    kendalltau(f, r)[0] for f, r in zip(features, returns)
                ])
            
            return correlations
            
        except Exception as e:
            self.logger.warning(f"VectorBT batch correlation failed: {e}")
            # Fallback to individual calculations
            return self._individual_correlation_calculation(features, returns)
    
    def _individual_correlation_calculation(self, features: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Fallback to individual correlation calculations."""
        correlations = []
        for i in range(len(features)):
            try:
                if self.config.correlation_method == 'pearson':
                    corr = np.corrcoef(features[i], returns[i])[0, 1]
                elif self.config.correlation_method == 'spearman':
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(features[i], returns[i])
                else:  # kendall
                    from scipy.stats import kendalltau
                    corr, _ = kendalltau(features[i], returns[i])
                
                correlations.append(corr if not np.isnan(corr) else 0.0)
            except Exception:
                correlations.append(0.0)
        
        return np.array(correlations)
    
    @safe_operation
    def calculate_mutual_information_vectorbt(
        self, 
        features_list: List[np.ndarray], 
        returns_list: List[np.ndarray]
    ) -> List[float]:
        """
        Calculate mutual information using VectorBT-optimized correlation approximation.
        
        Args:
            features_list: List of feature arrays
            returns_list: List of corresponding return arrays
            
        Returns:
            List of mutual information scores
        """
        if not self.config.mi_approximation:
            # Use exact MI calculation if approximation is disabled
            return self._calculate_exact_mi(features_list, returns_list)
        
        tprint_debug(f"🔄 Calculating MI for {len(features_list)} pairs using VectorBT approximation")
        start_time = time.time()
        
        try:
            # Get correlations using VectorBT
            correlations = self.calculate_correlations_vectorbt(features_list, returns_list)
            
            # Convert correlations to MI approximations using VectorBT
            mi_scores = self._correlation_to_mi_vectorbt(correlations)
            
            computation_time = time.time() - start_time
            tprint_success(f"✅ VectorBT MI calculation completed in {computation_time:.3f}s")
            
            return mi_scores
            
        except Exception as e:
            self.logger.error(f"VectorBT MI calculation failed: {e}")
            return self._fallback_mi_calculation(features_list, returns_list)
    
    def _correlation_to_mi_vectorbt(self, correlations: List[float]) -> List[float]:
        """Convert correlations to mutual information using VectorBT operations."""
        correlations_array = np.array(correlations)
        
        # Use VectorBT's optimized log operations
        # MI approximation: MI ≈ -0.5 * log(1 - r²)
        squared_correlations = correlations_array ** 2
        
        # Avoid log(0) and log(negative) issues
        safe_squared = np.clip(squared_correlations, 0, 0.999)
        
        # Calculate MI using VectorBT's optimized log function
        mi_scores = vbt.generic.nb.log_1p(-safe_squared) * -0.5
        
        # Ensure non-negative values
        mi_scores = np.maximum(mi_scores, 0.0)
        
        return mi_scores.tolist()
    
    def _calculate_exact_mi(self, features_list: List[np.ndarray], returns_list: List[np.ndarray]) -> List[float]:
        """Calculate exact mutual information using scipy."""
        from scipy.stats import entropy
        from sklearn.feature_selection import mutual_info_regression
        
        mi_scores = []
        for features, returns in zip(features_list, returns_list):
            try:
                # Align arrays
                min_length = min(len(features), len(returns))
                if min_length < 10:
                    mi_scores.append(0.0)
                    continue
                
                features_aligned = features[:min_length].reshape(-1, 1)
                returns_aligned = returns[:min_length]
                
                # Calculate mutual information
                mi = mutual_info_regression(features_aligned, returns_aligned, random_state=42)[0]
                mi_scores.append(max(0.0, mi))
                
            except Exception as e:
                self.logger.warning(f"Exact MI calculation failed: {e}")
                mi_scores.append(0.0)
        
        return mi_scores
    
    def _fallback_correlation_calculation(self, features_list: List[np.ndarray], returns_list: List[np.ndarray]) -> List[float]:
        """Fallback correlation calculation when VectorBT fails."""
        correlations = []
        for features, returns in zip(features_list, returns_list):
            try:
                min_length = min(len(features), len(returns))
                if min_length < 10:
                    correlations.append(0.0)
                    continue
                
                features_aligned = features[:min_length]
                returns_aligned = returns[:min_length]
                
                corr = np.corrcoef(features_aligned, returns_aligned)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0.0)
                
            except Exception:
                correlations.append(0.0)
        
        return correlations
    
    def _fallback_mi_calculation(self, features_list: List[np.ndarray], returns_list: List[np.ndarray]) -> List[float]:
        """Fallback MI calculation when VectorBT fails."""
        return self._calculate_exact_mi(features_list, returns_list)
    
    @lru_cache(maxsize=1000)
    def _cached_correlation(self, features_hash: str, returns_hash: str) -> float:
        """Cached correlation calculation for repeated computations."""
        # This is a placeholder for caching - in practice, you'd implement
        # proper caching based on data hashes
        return 0.0
    
    def clear_caches(self):
        """Clear all caches to free memory."""
        self._correlation_cache.clear()
        self._mi_cache.clear()
        self._cached_correlation.cache_clear()
        tprint_debug("🧹 VectorBT correlation caches cleared")


class VectorBTBatchProcessor:
    """
    Batch processor for VectorBT operations.
    
    This class handles batch processing of multiple feature-return pairs
    using VectorBT's optimized operations.
    """
    
    def __init__(self, config: Optional[VectorBTCorrelationConfig] = None):
        """Initialize VectorBT batch processor."""
        self.config = config or VectorBTCorrelationConfig()
        self.calculator = VectorBTCorrelationCalculator(config)
        self.logger = get_logger('VectorBTBatchProcessor')
    
    def process_batch(
        self, 
        features_batch: List[np.ndarray], 
        returns_batch: List[np.ndarray],
        operation: str = 'correlation'
    ) -> List[float]:
        """
        Process a batch of feature-return pairs.
        
        Args:
            features_batch: List of feature arrays
            returns_batch: List of return arrays
            operation: Type of operation ('correlation' or 'mi')
            
        Returns:
            List of results
        """
        if operation == 'correlation':
            return self.calculator.calculate_correlations_vectorbt(features_batch, returns_batch)
        elif operation == 'mi':
            return self.calculator.calculate_mutual_information_vectorbt(features_batch, returns_batch)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def process_large_batch(
        self, 
        features_batch: List[np.ndarray], 
        returns_batch: List[np.ndarray],
        operation: str = 'correlation'
    ) -> List[float]:
        """
        Process large batches by splitting into smaller chunks.
        
        Args:
            features_batch: List of feature arrays
            returns_batch: List of return arrays
            operation: Type of operation
            
        Returns:
            List of results
        """
        batch_size = self.config.batch_size
        results = []
        
        for i in range(0, len(features_batch), batch_size):
            end_idx = min(i + batch_size, len(features_batch))
            chunk_features = features_batch[i:end_idx]
            chunk_returns = returns_batch[i:end_idx]
            
            chunk_results = self.process_batch(chunk_features, chunk_returns, operation)
            results.extend(chunk_results)
        
        return results


# Convenience functions
def create_vectorbt_correlation_calculator(
    use_gpu: bool = False,
    batch_size: int = 1000,
    memory_efficient: bool = True
) -> VectorBTCorrelationCalculator:
    """Create a VectorBT correlation calculator with specified configuration."""
    config = VectorBTCorrelationConfig(
        use_gpu=use_gpu,
        batch_size=batch_size,
        memory_efficient=memory_efficient
    )
    return VectorBTCorrelationCalculator(config)


def create_vectorbt_batch_processor(
    use_gpu: bool = False,
    batch_size: int = 1000,
    memory_efficient: bool = True
) -> VectorBTBatchProcessor:
    """Create a VectorBT batch processor with specified configuration."""
    config = VectorBTCorrelationConfig(
        use_gpu=use_gpu,
        batch_size=batch_size,
        memory_efficient=memory_efficient
    )
    return VectorBTBatchProcessor(config)


# Test function
def test_vectorbt_correlation():
    """Test VectorBT correlation calculations."""
    if not VECTORBT_AVAILABLE:
        tprint_error("❌ VectorBT not available for testing")
        return False
    
    tprint("🧪 Testing VectorBT Correlation Calculator...")
    
    try:
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        features_list = [np.random.randn(n_samples) for _ in range(n_features)]
        returns_list = [np.random.randn(n_samples) for _ in range(n_features)]
        
        # Test correlation calculator
        calculator = create_vectorbt_correlation_calculator()
        correlations = calculator.calculate_correlations_vectorbt(features_list, returns_list)
        
        tprint_success(f"✅ Calculated {len(correlations)} correlations")
        tprint_info(f"📊 Correlation range: {min(correlations):.4f} to {max(correlations):.4f}")
        
        # Test MI calculation
        mi_scores = calculator.calculate_mutual_information_vectorbt(features_list, returns_list)
        tprint_success(f"✅ Calculated {len(mi_scores)} MI scores")
        tprint_info(f"📊 MI range: {min(mi_scores):.4f} to {max(mi_scores):.4f}")
        
        # Test batch processor
        batch_processor = create_vectorbt_batch_processor()
        batch_correlations = batch_processor.process_batch(features_list, returns_list, 'correlation')
        
        tprint_success(f"✅ Batch processing completed: {len(batch_correlations)} results")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ VectorBT correlation test failed: {e}")
        return False


if __name__ == "__main__":
    test_vectorbt_correlation()