"""
VectorBT Rolling Operations for Time Series Feature Selection

This module provides VectorBT-optimized rolling operations specifically designed
for time series feature selection with enhanced performance and memory efficiency.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Import utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.math_validation import validate_numeric_array, validate_finite

# Import VectorBTRollingOptimizer for enhanced performance
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Import UnifiedVectorizationManager for unified vectorization
try:
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None

# Import hardware optimization tools
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager,
        get_unified_hardware_manager,
        WorkloadType,
        OptimizationLevel,
        HardwareConfig
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    UnifiedHardwareManager = None
    get_unified_hardware_manager = None
    WorkloadType = None
    OptimizationLevel = None
    HardwareConfig = None

from .vectorbt_config import VectorBTFeatureSelectionConfig

logger = logging.getLogger(__name__)

@dataclass
class RollingConfig:
    """Configuration for VectorBT rolling operations."""
    # Rolling window settings
    default_window: int = 100
    min_periods: int = 1
    center: bool = False

    # Memory optimization
    enable_memory_optimization: bool = True
    chunk_size: int = 1000
    overlap: int = 100

    # Performance settings
    enable_parallel: bool = True
    max_workers: Optional[int] = None

    # Financial data specific
    enable_financial_optimization: bool = True
    business_day_only: bool = True

class VectorBTRollingOperations:
    """
    VectorBT-optimized rolling operations for time series feature selection.

    This class provides:
    - Rolling correlation analysis for feature relationships
    - Rolling variance analysis for feature stability
    - Rolling mutual information for time-dependent relevance
    - Memory-efficient processing for large time series
    - Financial data optimization
    """

    def __init__(self, config: Optional[VectorBTFeatureSelectionConfig] = None):
        """Initialize VectorBT rolling operations with enhanced optimizations."""
        self.config = config or VectorBTFeatureSelectionConfig()
        self.rolling_config = RollingConfig()
        self.logger = logger.getChild('VectorBTRollingOperations')

        # Check VectorBT availability
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Please install vectorbt.")

        # Initialize VectorBTRollingOptimizer for enhanced performance
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.rolling_optimizer = VectorBTRollingOptimizer(
                    enable_parallel=self.rolling_config.enable_parallel,
                    memory_efficient=self.rolling_config.enable_memory_optimization,
                    chunk_size=self.rolling_config.chunk_size,
                    enable_hardware_optimization=HARDWARE_AVAILABLE,
                    workload_type=WorkloadType.FEATURE_ENGINEERING if HARDWARE_AVAILABLE else None
                )
                tprint_success("✅ VectorBTRollingOptimizer integrated")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBTRollingOptimizer initialization failed: {e}")
                self.rolling_optimizer = None
        else:
            self.rolling_optimizer = None
            tprint_warning("⚠️ VectorBTRollingOptimizer not available. Using fallback methods.")

        # Initialize UnifiedVectorizationManager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_success("✅ UnifiedVectorizationManager integrated")
            except Exception as e:
                tprint_warning(f"⚠️ UnifiedVectorizationManager initialization failed: {e}")
                self.vectorization_manager = None
        else:
            self.vectorization_manager = None
            tprint_warning("⚠️ UnifiedVectorizationManager not available. Using fallback methods.")

        # Initialize hardware manager
        if HARDWARE_AVAILABLE:
            try:
                self.hardware_manager = get_unified_hardware_manager()
                self.hardware_manager.optimize_for_workload(
                    WorkloadType.FEATURE_ENGINEERING,
                    OptimizationLevel.BALANCED
                )
                tprint_success("✅ Hardware manager integrated")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware manager initialization failed: {e}")
                self.hardware_manager = None
        else:
            self.hardware_manager = None

        # Enhanced performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'rolling_operations': 0,
            'vectorbt_optimizer_operations': 0,
            'unified_vectorization_operations': 0,
            'hardware_optimized_operations': 0,
            'total_time': 0.0,
            'rolling_time': 0.0,
            'features_processed': 0,
            'memory_saved_mb': 0.0,
            'speedup_vs_naive': 0.0
        }

        tprint_success("🚀 VectorBTRollingOperations initialized with enhanced optimizations")

    def _create_time_series_dataframe(self, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Create VectorBT-optimized time series DataFrame."""
        try:
            # Use VectorBT's optimized DataFrame creation
            df = vbt.PandasDataFrame(X, columns=feature_names)

            # Enhanced financial time series indexing
            if self.rolling_config.enable_financial_optimization:
                # Use proper financial time series indexing
                if self.rolling_config.business_day_only:
                    df.index = pd.bdate_range(start='2020-01-01', periods=len(df), freq='1min')
                else:
                    df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='1min')

                # Enable VectorBT's financial data optimizations
                try:
                    df = df.vbt.freq_infer()  # Infer optimal frequency
                    df = df.vbt.resample_apply('1H', 'last')  # Resample for efficiency
                    df = df.vbt.validate()  # Validate financial data integrity
                except Exception as freq_e:
                    self.logger.debug(f"Financial optimization skipped: {freq_e}")

            return df

        except Exception as e:
            self.logger.warning(f"Enhanced DataFrame creation failed: {e}")
            # Fallback to standard DataFrame
            df = pd.DataFrame(X, columns=feature_names)
            if self.rolling_config.enable_financial_optimization:
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            return df

    def rolling_correlation_analysis(self, X: np.ndarray, feature_names: List[str],
                                   window: int = None, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Perform rolling correlation analysis using VectorBT optimization.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            window: Rolling window size
            threshold: Correlation threshold for filtering

        Returns:
            Dictionary with correlation analysis results
        """
        window = window or self.rolling_config.default_window

        def _rolling_correlation():
            try:
                # Validate inputs
                X = validate_numeric_array(X, name="Feature matrix X")
                if not validate_finite(X):
                    raise ValueError("Feature matrix X contains non-finite values")

                # Create VectorBT DataFrame
                df = self._create_time_series_dataframe(X, feature_names)

                # Use VectorBTRollingOptimizer if available for enhanced performance
                if self.rolling_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
                    try:
                        # Use VectorBTRollingOptimizer for enhanced rolling correlation
                        rolling_corr = self.rolling_optimizer.rolling_corr(
                            df, window=window, min_periods=self.rolling_config.min_periods
                        )
                        self.performance_stats['vectorbt_optimizer_operations'] += 1
                        tprint_debug("🚀 Using VectorBTRollingOptimizer for correlation analysis")
                    except Exception as opt_e:
                        tprint_warning(f"⚠️ VectorBTRollingOptimizer failed, falling back: {opt_e}")
                        # Fallback to standard VectorBT
                        if hasattr(df, 'vbt'):
                            rolling_corr = df.vbt.rolling_corr(
                                window=window,
                                min_periods=self.rolling_config.min_periods,
                            )
                        else:
                            rolling_corr = df.rolling(window=window, min_periods=self.rolling_config.min_periods).corr()
                elif hasattr(df, 'vbt'):
                    try:
                        # VectorBT rolling correlation with memory optimization
                        rolling_corr = df.vbt.rolling_corr(
                            window=window,
                            min_periods=self.rolling_config.min_periods,
                            pairwise=True,
                            chunked=self.rolling_config.enable_memory_optimization,
                            parallel=self.rolling_config.enable_parallel
                        )

                        # Get final correlation matrix
                        final_corr = rolling_corr.iloc[-1]

                        # Apply VectorBT optimizations
                        final_corr = final_corr.vbt.fillna(0)
                        final_corr = final_corr.vbt.clip(-1, 1)

                        tprint_debug(f"📊 VectorBT rolling correlation completed for window {window}")

                    except Exception as vbt_e:
                        self.logger.debug(f"VectorBT rolling correlation failed: {vbt_e}")
                        # Fallback to standard rolling correlation
                        rolling_corr = df.rolling(window=window, min_periods=self.rolling_config.min_periods).corr()
                        final_corr = rolling_corr.iloc[-1]
                else:
                    # Standard rolling correlation
                    rolling_corr = df.rolling(window=window, min_periods=self.rolling_config.min_periods).corr()
                    final_corr = rolling_corr.iloc[-1]

                # Find highly correlated pairs
                corr_values = final_corr.values
                high_corr_pairs = []

                for i in range(len(corr_values)):
                    for j in range(i + 1, len(corr_values)):
                        if abs(corr_values[i, j]) > threshold:
                            high_corr_pairs.append({
                                'feature1': feature_names[i],
                                'feature2': feature_names[j],
                                'correlation': float(corr_values[i, j]),
                                'window': window
                            })

                # Update performance stats
                self.performance_stats['rolling_operations'] += 1
                self.performance_stats['features_processed'] += X.shape[1]

                return {
                    'success': True,
                    'correlation_matrix': corr_values,
                    'high_corr_pairs': high_corr_pairs,
                    'n_high_corr_pairs': len(high_corr_pairs),
                    'window': window,
                    'threshold': threshold,
                    'method': 'vectorbt_rolling_correlation'
                }

            except Exception as e:
                self.logger.error(f"Rolling correlation analysis failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'method': 'vectorbt_rolling_correlation'
                }

        start_time = time.time()
        result = _rolling_correlation()
        execution_time = time.time() - start_time

        self.performance_stats['total_time'] += execution_time
        self.performance_stats['rolling_time'] += execution_time

        if self.config.log_performance:
            tprint_performance(f"⏱️ Rolling Correlation Analysis: {execution_time:.3f}s")

        return result

    def rolling_variance_analysis(self, X: np.ndarray, feature_names: List[str],
                                window: int = None, threshold: float = 0.01) -> Dict[str, Any]:
        """
        Perform rolling variance analysis using VectorBT optimization.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            window: Rolling window size
            threshold: Variance threshold for filtering

        Returns:
            Dictionary with variance analysis results
        """
        window = window or self.rolling_config.default_window

        def _rolling_variance():
            try:
                # Validate inputs
                X = validate_numeric_array(X, name="Feature matrix X")
                if not validate_finite(X):
                    raise ValueError("Feature matrix X contains non-finite values")

                # Create VectorBT DataFrame
                df = self._create_time_series_dataframe(X, feature_names)

                # Use VectorBT's optimized rolling variance
                if hasattr(df, 'vbt'):
                    try:
                        # VectorBT rolling variance with memory optimization
                        rolling_var = df.vbt.rolling_apply(
                            'var',
                            window=window,
                            min_periods=self.rolling_config.min_periods,
                            chunked=self.rolling_config.enable_memory_optimization,
                            parallel=self.rolling_config.enable_parallel
                        )

                        # Get final variance values
                        final_var = rolling_var.iloc[-1]

                        # Apply VectorBT optimizations
                        final_var = final_var.vbt.fillna(0)

                        tprint_debug(f"📊 VectorBT rolling variance completed for window {window}")

                    except Exception as vbt_e:
                        self.logger.debug(f"VectorBT rolling variance failed: {vbt_e}")
                        # Fallback to standard rolling variance
                        rolling_var = df.rolling(window=window, min_periods=self.rolling_config.min_periods).var()
                        final_var = rolling_var.iloc[-1]
                else:
                    # Standard rolling variance
                    rolling_var = df.rolling(window=window, min_periods=self.rolling_config.min_periods).var()
                    final_var = rolling_var.iloc[-1]

                # Find features with low variance
                var_values = final_var.values if hasattr(final_var, 'values') else final_var
                low_var_mask = var_values < threshold
                low_var_features = [feature_names[i] for i in range(len(feature_names)) if low_var_mask[i]]

                # Update performance stats
                self.performance_stats['rolling_operations'] += 1
                self.performance_stats['features_processed'] += X.shape[1]

                return {
                    'success': True,
                    'variance_values': var_values.tolist(),
                    'low_var_features': low_var_features,
                    'n_low_var_features': len(low_var_features),
                    'window': window,
                    'threshold': threshold,
                    'method': 'vectorbt_rolling_variance'
                }

            except Exception as e:
                self.logger.error(f"Rolling variance analysis failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'method': 'vectorbt_rolling_variance'
                }

        start_time = time.time()
        result = _rolling_variance()
        execution_time = time.time() - start_time

        self.performance_stats['total_time'] += execution_time
        self.performance_stats['rolling_time'] += execution_time

        if self.config.log_performance:
            tprint_performance(f"⏱️ Rolling Variance Analysis: {execution_time:.3f}s")

        return result

    def rolling_mutual_information_analysis(self, X: np.ndarray, y: np.ndarray,
                                          feature_names: List[str], window: int = None) -> Dict[str, Any]:
        """
        Perform rolling mutual information analysis using VectorBT optimization.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            feature_names: List of feature names
            window: Rolling window size

        Returns:
            Dictionary with mutual information analysis results
        """
        window = window or self.rolling_config.default_window

        def _rolling_mutual_info():
            try:
                from sklearn.feature_selection import mutual_info_regression

                # Validate inputs
                X = validate_numeric_array(X, name="Feature matrix X")
                y = validate_numeric_array(y, name="Target variable y")
                if not validate_finite(X) or not validate_finite(y):
                    raise ValueError("Input data contains non-finite values")

                # Create VectorBT DataFrame
                df = self._create_time_series_dataframe(X, feature_names)

                # Use VectorBT's optimized rolling operations
                if hasattr(df, 'vbt'):
                    try:
                        # VectorBT rolling mutual information with memory optimization
                        rolling_mi = df.vbt.rolling_apply(
                            lambda chunk: mutual_info_regression(chunk, y, random_state=42),
                            window=window,
                            min_periods=self.rolling_config.min_periods,
                            chunked=self.rolling_config.enable_memory_optimization,
                            parallel=self.rolling_config.enable_parallel
                        )

                        # Get final mutual information values
                        final_mi = rolling_mi.iloc[-1]

                        # Apply VectorBT optimizations
                        final_mi = final_mi.vbt.fillna(0)

                        tprint_debug(f"📊 VectorBT rolling mutual information completed for window {window}")

                    except Exception as vbt_e:
                        self.logger.debug(f"VectorBT rolling mutual information failed: {vbt_e}")
                        # Fallback to chunked processing
                        final_mi = self._compute_rolling_mi_fallback(df, y, window)
                else:
                    # Fallback to chunked processing
                    final_mi = self._compute_rolling_mi_fallback(df, y, window)

                # Get mutual information values
                mi_values = final_mi.values if hasattr(final_mi, 'values') else final_mi

                # Rank features by mutual information
                ranked_indices = np.argsort(mi_values)[::-1]
                ranked_features = [feature_names[i] for i in ranked_indices]
                ranked_scores = [float(mi_values[i]) for i in ranked_indices]

                # Update performance stats
                self.performance_stats['rolling_operations'] += 1
                self.performance_stats['features_processed'] += X.shape[1]

                return {
                    'success': True,
                    'mutual_info_values': mi_values.tolist(),
                    'ranked_features': ranked_features,
                    'ranked_scores': ranked_scores,
                    'window': window,
                    'method': 'vectorbt_rolling_mutual_information'
                }

            except Exception as e:
                self.logger.error(f"Rolling mutual information analysis failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'method': 'vectorbt_rolling_mutual_information'
                }

        start_time = time.time()
        result = _rolling_mutual_info()
        execution_time = time.time() - start_time

        self.performance_stats['total_time'] += execution_time
        self.performance_stats['rolling_time'] += execution_time

        if self.config.log_performance:
            tprint_performance(f"⏱️ Rolling Mutual Information Analysis: {execution_time:.3f}s")

        return result

    def _compute_rolling_mi_fallback(self, df: pd.DataFrame, y: np.ndarray, window: int) -> pd.Series:
        """Fallback method for rolling mutual information computation."""
        try:
            from sklearn.feature_selection import mutual_info_regression

            mi_values = []

            for i in range(len(df)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1

                if end_idx - start_idx >= self.rolling_config.min_periods:
                    chunk_X = df.iloc[start_idx:end_idx].values
                    chunk_y = y[start_idx:end_idx]

                    if len(chunk_X) > 0:
                        mi_scores = mutual_info_regression(chunk_X, chunk_y, random_state=42)
                        mi_values.append(mi_scores)
                    else:
                        mi_values.append(np.zeros(df.shape[1]))
                else:
                    mi_values.append(np.zeros(df.shape[1]))

            # Convert to DataFrame and get final values
            mi_df = pd.DataFrame(mi_values, columns=df.columns)
            return mi_df.iloc[-1]

        except Exception as e:
            self.logger.warning(f"Rolling MI fallback failed: {e}")
            return pd.Series(np.zeros(df.shape[1]), index=df.columns)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        if stats['rolling_operations'] > 0:
            stats['avg_time_per_operation'] = stats['rolling_time'] / stats['rolling_operations']
        else:
            stats['avg_time_per_operation'] = 0.0

        if stats['features_processed'] > 0:
            stats['avg_features_per_second'] = stats['features_processed'] / stats['rolling_time']
        else:
            stats['avg_features_per_second'] = 0.0

        tprint_performance(f"📊 VectorBT Rolling Operations Stats: {stats['rolling_operations']} operations, "
                         f"{stats['avg_time_per_operation']:.3f}s avg, "
                         f"{stats['avg_features_per_second']:.1f} features/sec")

        return stats

def create_vectorbt_rolling_operations(config: Optional[VectorBTFeatureSelectionConfig] = None) -> VectorBTRollingOperations:
    """Create a VectorBT rolling operations instance."""
    return VectorBTRollingOperations(config)
