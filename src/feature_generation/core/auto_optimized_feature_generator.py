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
        Generate the feature. This is an abstract method that must be implemented by subclasses.
        This base implementation provides auto-optimization wrapper functionality.

        Args:
            data: Input data DataFrame
            **kwargs: Additional keyword arguments

        Returns:
            pd.Series: Generated feature series

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement _generate_feature method"
        )

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
