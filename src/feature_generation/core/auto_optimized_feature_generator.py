from src.utils.tprint import tprint
from typing import Dict, Any, Optional, Union
import logging
import pandas as pd
import time
import threading

from .feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory, FeatureResult
from .optimization_mixin import OptimizationMixin
from .rolling_operations_mixin import RollingOperationsMixin
from .vectorbt_optimization_mixin import VectorBTOptimizationMixin
from .auto_optimization_config import AutoOptimizationConfig, OptimizationLevel
from .optimization_strategies import (
    OptimizationStrategy,
    ConservativeOptimizationStrategy,
    BalancedOptimizationStrategy,
    AggressiveOptimizationStrategy
)

logger = logging.getLogger(__name__)

class AutoOptimizedFeatureGenerator(FeatureGenerator,
                                   OptimizationMixin,
                                   RollingOperationsMixin,
                                   VectorBTOptimizationMixin):
    """Base class with automatic optimization enabled by default."""

    def __init__(self, config: FeatureConfig,
                 auto_optimization_config: Optional[AutoOptimizationConfig] = None,
                 **kwargs):
        try:
            # Initialize base classes
            super().__init__(config)

            # Initialize mixins
            OptimizationMixin.__init__(self)
            RollingOperationsMixin.__init__(self)
            VectorBTOptimizationMixin.__init__(self)

            # Auto-optimization configuration
            self.auto_optimization_config = auto_optimization_config or AutoOptimizationConfig()

            # Apply level-specific settings
            self._apply_level_settings()

            # Initialize optimization strategy
            self.optimization_strategy = self._create_optimization_strategy()

            # Performance tracking
            self.auto_optimization_stats = {
                'total_optimizations': 0,
                'total_optimization_time': 0.0,
                'memory_savings_mb': 0.0,
                'strategy_used': self.auto_optimization_config.optimization_level.value
            }
            
            # Thread safety for stats updates
            self._stats_lock = threading.Lock()

            self.logger = logger.getChild(f'AutoOptimized.{self.__class__.__name__}')

            if self.auto_optimization_config.enable_optimization_logging:
                self.logger.info(f"Auto-optimization enabled with {self.auto_optimization_config.optimization_level.value} strategy")

            self.logger.debug(f"AutoOptimizedFeatureGenerator '{config.name}' initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing AutoOptimizedFeatureGenerator '{config.name}': {e}")
            raise

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate the feature. This is a base implementation that provides auto-optimization wrapper functionality.
        
        This method provides a default implementation that can be overridden by subclasses.
        It applies basic feature generation with automatic optimization.

        Args:
            data: Input data DataFrame
            **kwargs: Additional keyword arguments

        Returns:
            pd.Series: Generated feature series
        """
        try:
            # Apply auto-optimization if enabled
            if self.auto_optimization_config.enable_auto_optimization:
                data = self._auto_optimize_data(data)
            
            # Default implementation: return a simple feature based on available data
            if data.empty:
                self.logger.warning("Input data is empty, returning empty series")
                return pd.Series(dtype=float, index=data.index)
            
            # Try to generate a meaningful feature based on available columns
            feature_series = self._generate_default_feature(data, **kwargs)
            
            # Apply post-processing optimization
            if self.auto_optimization_config.enable_auto_optimization:
                feature_series = self._optimize_feature_series(feature_series)
            
            return feature_series
            
        except Exception as e:
            self.logger.error(f"Error generating feature: {e}")
            # Return a safe fallback
            return pd.Series(dtype=float, index=data.index)

    def _generate_default_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate a default feature when no specific implementation is provided.
        
        Args:
            data: Input data DataFrame
            **kwargs: Additional keyword arguments
            
        Returns:
            pd.Series: Generated feature series
        """
        try:
            # Look for common price columns
            price_columns = ['close', 'Close', 'price', 'Price', 'last', 'Last']
            price_col = None
            for col in price_columns:
                if col in data.columns:
                    price_col = col
                    break
            
            if price_col is not None:
                # Generate a simple moving average feature
                window = kwargs.get('window', 20)
                if len(data) >= window:
                    feature = data[price_col].rolling(window=window, min_periods=1).mean()
                else:
                    feature = data[price_col].mean()
                return feature
            
            # Look for volume columns
            volume_columns = ['volume', 'Volume', 'vol', 'Vol']
            volume_col = None
            for col in volume_columns:
                if col in data.columns:
                    volume_col = col
                    break
            
            if volume_col is not None:
                # Generate a volume-based feature
                window = kwargs.get('window', 10)
                if len(data) >= window:
                    feature = data[volume_col].rolling(window=window, min_periods=1).sum()
                else:
                    feature = data[volume_col].sum()
                return feature
            
            # Fallback: use the first numeric column
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                col = numeric_columns[0]
                window = kwargs.get('window', 5)
                if len(data) >= window:
                    feature = data[col].rolling(window=window, min_periods=1).mean()
                else:
                    feature = data[col].mean()
                return feature
            
            # Last resort: return zeros
            self.logger.warning("No suitable columns found for feature generation, returning zeros")
            return pd.Series(0.0, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Error in default feature generation: {e}")
            return pd.Series(dtype=float, index=data.index)

    def _optimize_feature_series(self, feature_series: pd.Series) -> pd.Series:
        """
        Apply optimization to the generated feature series.
        
        Args:
            feature_series: The feature series to optimize
            
        Returns:
            pd.Series: Optimized feature series
        """
        try:
            if feature_series.empty:
                return feature_series
            
            # Apply data type optimization
            if self.auto_optimization_config.enable_dtype_optimization:
                # Convert to more memory-efficient dtype if possible
                if feature_series.dtype == 'float64':
                    # Check if we can use float32
                    if feature_series.min() >= np.finfo(np.float32).min and feature_series.max() <= np.finfo(np.float32).max:
                        feature_series = feature_series.astype(np.float32)
            
            # Apply NaN handling optimization
            if self.auto_optimization_config.enable_nan_optimization:
                if feature_series.isna().any():
                    # Forward fill then backward fill
                    feature_series = feature_series.fillna(method='ffill').fillna(method='bfill')
                    # If still NaN, fill with 0
                    feature_series = feature_series.fillna(0.0)
            
            # Apply outlier handling
            if self.auto_optimization_config.enable_outlier_optimization:
                # Winsorize outliers (cap at 99th percentile)
                q99 = feature_series.quantile(0.99)
                q01 = feature_series.quantile(0.01)
                feature_series = feature_series.clip(lower=q01, upper=q99)
            
            return feature_series
            
        except Exception as e:
            self.logger.warning(f"Feature optimization failed: {e}")
            return feature_series

    def _apply_level_settings(self) -> None:
        """Apply settings based on optimization level."""
        try:
            level_settings = self.auto_optimization_config.get_settings_for_level()

            applied_count = 0
            for key, value in level_settings.items():
                if hasattr(self.auto_optimization_config, key):
                    setattr(self.auto_optimization_config, key, value)
                    applied_count += 1
                else:
                    self.logger.warning(f"Setting {key} not found in config, skipping")

            self.logger.debug(f"Applied {applied_count} level settings for {self.auto_optimization_config.optimization_level.value}")

        except Exception as e:
            self.logger.error(f"Error applying level settings: {e}")
            raise

    def _create_optimization_strategy(self) -> OptimizationStrategy:
        """Create optimization strategy based on configuration."""
        try:
            if self.auto_optimization_config.optimization_level == OptimizationLevel.CONSERVATIVE:
                strategy = ConservativeOptimizationStrategy(self.auto_optimization_config)
            elif self.auto_optimization_config.optimization_level == OptimizationLevel.BALANCED:
                strategy = BalancedOptimizationStrategy(self.auto_optimization_config)
            elif self.auto_optimization_config.optimization_level == OptimizationLevel.AGGRESSIVE:
                strategy = AggressiveOptimizationStrategy(self.auto_optimization_config)
            else:
                self.logger.warning(f"Unknown optimization level {self.auto_optimization_config.optimization_level.value}, defaulting to balanced")
                strategy = BalancedOptimizationStrategy(self.auto_optimization_config)

            self.logger.debug(f"Created optimization strategy: {strategy.__class__.__name__}")
            return strategy

        except Exception as e:
            self.logger.error(f"Error creating optimization strategy: {e}")
            self.logger.info("Falling back to balanced strategy")
            return BalancedOptimizationStrategy(self.auto_optimization_config)

    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """Generate feature with automatic optimization."""
        try:
            start_time = time.time()

            # DEBUG: Check data quality at the start of generate
            import numpy as np
            if self.auto_optimization_config.enable_optimization_logging:
                self.logger.debug(f"AutoOptimizedFeatureGenerator.generate - Data shape: {data.shape}")
                self.logger.debug(f"AutoOptimizedFeatureGenerator.generate - Non-finite values: {(~np.isfinite(data.select_dtypes(include=[np.number])).values).sum()}")
                for col in data.select_dtypes(include=[np.number]).columns:
                    non_finite = (~np.isfinite(data[col])).sum()
                    if non_finite > 0:
                        self.logger.debug(f"AutoOptimizedFeatureGenerator.generate - {col}: {non_finite} non-finite values")

            # Log optimization start
            if self.auto_optimization_config.enable_optimization_logging:
                self.logger.debug(f"Starting auto-optimization for {self.config.name}")

            # Apply automatic optimization
            if self.auto_optimization_config.enable_auto_optimization:
                self.logger.debug(f"Applying auto-optimization ({self.auto_optimization_config.optimization_level.value})")
                data = self._auto_optimize_data(data)
                # DEBUG: Check data quality after optimization
                if self.auto_optimization_config.enable_optimization_logging:
                    self.logger.debug(f"After _auto_optimize_data - Data shape: {data.shape}")
                    self.logger.debug(f"After _auto_optimize_data - Non-finite values: {(~np.isfinite(data.select_dtypes(include=[np.number])).values).sum()}")
                    for col in data.select_dtypes(include=[np.number]).columns:
                        non_finite = (~np.isfinite(data[col])).sum()
                        if non_finite > 0:
                            self.logger.debug(f"After _auto_optimize_data - {col}: {non_finite} non-finite values")
            else:
                self.logger.debug("Auto-optimization disabled, using original data")

            # Call parent generate method
            result = super().generate(data, **kwargs)

            # Update auto-optimization stats
            optimization_time = time.time() - start_time
            with self._stats_lock:
                self.auto_optimization_stats['total_optimizations'] += 1
                self.auto_optimization_stats['total_optimization_time'] += optimization_time

            self.logger.debug(f"Feature generation completed in {optimization_time:.3f}s, success: {result.success}")

            # Add optimization info to result metadata
            if result.metadata is None:
                result.metadata = {}

            result.metadata.update({
                'auto_optimization_enabled': self.auto_optimization_config.enable_auto_optimization,
                'optimization_strategy': self.auto_optimization_config.optimization_level.value,
                'optimization_time': optimization_time,
                'optimization_stats': self.optimization_strategy.get_stats()
            })

            # Add auto-optimization error if present
            if hasattr(self, '_last_auto_opt_error') and self._last_auto_opt_error:
                result.metadata['auto_optimization_error'] = self._last_auto_opt_error

            return result

        except Exception as e:
            self.logger.error(f"Error generating feature '{self.config.name}': {e}")
            # Create a failed result
            return FeatureResult(
                name=self.config.name,
                data=pd.Series(dtype=float, index=data.index),
                config=self.config,
                computation_time=0.0,
                success=False,
                error_message=str(e)
            )

    def _auto_optimize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply automatic optimization using the configured strategy."""
        try:
            if not self.auto_optimization_config.enable_auto_optimization:
                self.logger.debug("Auto-optimization disabled, returning original data")
                return data

            self.logger.debug(f"Applying {self.optimization_strategy.__class__.__name__} optimization, input shape: {data.shape}")

            # Use the configured optimization strategy
            optimized_data = self.optimization_strategy.optimize_data(data, self)

            self.logger.debug(f"Optimization strategy completed, output shape: {optimized_data.shape}")

            # Update memory savings stats
            if hasattr(self, 'get_optimization_stats'):
                opt_stats = self.get_optimization_stats()
                memory_saved = opt_stats.get('memory_saved_mb', 0.0)
                with self._stats_lock:
                    self.auto_optimization_stats['memory_savings_mb'] += memory_saved
                self.logger.debug(f"Memory saved this optimization: {memory_saved:.2f}MB")

            if self.auto_optimization_config.enable_optimization_logging:
                self.logger.debug(f"Auto-optimization completed for {self.config.name}")

            return optimized_data

        except Exception as e:
            self.logger.warning(f"Auto-optimization failed: {e}, using original data")
            self._last_auto_opt_error = str(e)
            return data

    def set_optimization_strategy(self, level: Union[str, OptimizationLevel]) -> None:
        """Change optimization strategy at runtime."""
        if isinstance(level, str):
            level = OptimizationLevel(level)

        self.auto_optimization_config.optimization_level = level
        self.optimization_strategy = self._create_optimization_strategy()
        self.auto_optimization_stats['strategy_used'] = level.value

        if self.auto_optimization_config.enable_optimization_logging:
            self.logger.info(f"Optimization strategy changed to {level.value}")

    def get_auto_optimization_stats(self) -> Dict[str, Any]:
        """Get automatic optimization statistics."""
        with self._stats_lock:
            stats = self.auto_optimization_stats.copy()

        if stats['total_optimizations'] > 0:
            stats['average_optimization_time'] = (
                stats['total_optimization_time'] / stats['total_optimizations']
            )
        else:
            stats['average_optimization_time'] = 0.0

        # Add strategy-specific stats
        stats['strategy_stats'] = self.optimization_strategy.get_stats()

        return stats

    def reset_auto_optimization_stats(self) -> None:
        """Reset automatic optimization statistics."""
        with self._stats_lock:
            self.auto_optimization_stats = {
                'total_optimizations': 0,
                'total_optimization_time': 0.0,
                'memory_savings_mb': 0.0,
                'strategy_used': self.auto_optimization_config.optimization_level.value
            }
        self.optimization_strategy.reset_stats()


    def enable_auto_optimization(self, enabled: bool = True) -> None:
        """Enable or disable automatic optimization."""
        self.auto_optimization_config.enable_auto_optimization = enabled

        if self.auto_optimization_config.enable_optimization_logging:
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"Auto-optimization {status}")

    def get_optimization_config(self) -> AutoOptimizationConfig:
        """Get current optimization configuration."""
        return self.auto_optimization_config

    def update_optimization_config(self, **kwargs) -> None:
        """Update optimization configuration."""
        for key, value in kwargs.items():
            if hasattr(self.auto_optimization_config, key):
                setattr(self.auto_optimization_config, key, value)

        # Recreate strategy if optimization level changed
        if 'optimization_level' in kwargs:
            self.optimization_strategy = self._create_optimization_strategy()

        if self.auto_optimization_config.enable_optimization_logging:
            self.logger.info(f"Optimization configuration updated: {kwargs}")
