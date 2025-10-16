from src.utils.tprint import tprint
from typing import Dict, Any, Optional, Union
import logging
import pandas as pd
import time

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
            tprint(f"🔧 Initializing AutoOptimizedFeatureGenerator: {config.name}")

            # Initialize base classes
            tprint("📦 Initializing base classes...")
            super().__init__(config)

            # Initialize mixins
            tprint("🔧 Initializing optimization mixins...")
            OptimizationMixin.__init__(self)
            RollingOperationsMixin.__init__(self)
            VectorBTOptimizationMixin.__init__(self)
            tprint("✅ All mixins initialized")

            # Auto-optimization configuration
            tprint("⚙️ Setting up auto-optimization configuration...")
            self.auto_optimization_config = auto_optimization_config or AutoOptimizationConfig()
            tprint(f"✅ Auto-optimization config: {self.auto_optimization_config.optimization_level.value}")

            # Apply level-specific settings
            tprint("🔧 Applying level-specific settings...")
            self._apply_level_settings()

            # Initialize optimization strategy
            tprint("🎯 Creating optimization strategy...")
            self.optimization_strategy = self._create_optimization_strategy()
            tprint(f"✅ Strategy created: {self.optimization_strategy.__class__.__name__}")

            # Performance tracking
            tprint("📊 Initializing performance tracking...")
            self.auto_optimization_stats = {
                'total_optimizations': 0,
                'total_optimization_time': 0.0,
                'memory_savings_mb': 0.0,
                'strategy_used': self.auto_optimization_config.optimization_level.value
            }

            self.logger = logger.getChild(f'AutoOptimized{self.__class__.__name__}')

            if self.auto_optimization_config.enable_optimization_logging:
                self.logger.info(f"Auto-optimization enabled with {self.auto_optimization_config.optimization_level.value} strategy")
                tprint(f"📝 Optimization logging enabled for {config.name}")

            tprint(f"✅ AutoOptimizedFeatureGenerator '{config.name}' initialized successfully")

        except Exception as e:
            tprint(f"❌ Error initializing AutoOptimizedFeatureGenerator '{config.name}': {e}")
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

    def _apply_level_settings(self):
        """Apply settings based on optimization level."""
        try:
            tprint(f"🔧 Applying level settings for {self.auto_optimization_config.optimization_level.value}...")
            level_settings = self.auto_optimization_config.get_settings_for_level()

            applied_count = 0
            for key, value in level_settings.items():
                if hasattr(self.auto_optimization_config, key):
                    setattr(self.auto_optimization_config, key, value)
                    applied_count += 1
                    tprint(f"   ✅ Applied {key} = {value}")
                else:
                    tprint(f"   ⚠️ Setting {key} not found in config, skipping")

            tprint(f"✅ Applied {applied_count} level settings")

        except Exception as e:
            tprint(f"❌ Error applying level settings: {e}")
            raise

    def _create_optimization_strategy(self) -> OptimizationStrategy:
        """Create optimization strategy based on configuration."""
        try:
            tprint(f"🎯 Creating optimization strategy for {self.auto_optimization_config.optimization_level.value}...")

            if self.auto_optimization_config.optimization_level == OptimizationLevel.CONSERVATIVE:
                tprint("📊 Creating conservative strategy...")
                strategy = ConservativeOptimizationStrategy(self.auto_optimization_config)
            elif self.auto_optimization_config.optimization_level == OptimizationLevel.BALANCED:
                tprint("📊 Creating balanced strategy...")
                strategy = BalancedOptimizationStrategy(self.auto_optimization_config)
            elif self.auto_optimization_config.optimization_level == OptimizationLevel.AGGRESSIVE:
                tprint("📊 Creating aggressive strategy...")
                strategy = AggressiveOptimizationStrategy(self.auto_optimization_config)
            else:
                tprint("⚠️ Unknown optimization level, defaulting to balanced...")
                strategy = BalancedOptimizationStrategy(self.auto_optimization_config)

            tprint(f"✅ Strategy created: {strategy.__class__.__name__}")
            return strategy

        except Exception as e:
            tprint(f"❌ Error creating optimization strategy: {e}")
            tprint("🔄 Falling back to balanced strategy...")
            return BalancedOptimizationStrategy(self.auto_optimization_config)

    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """Generate feature with automatic optimization."""
        try:
            tprint(f"🚀 Starting feature generation for '{self.config.name}' with auto-optimization...")
            start_time = time.time()

            # Log optimization start
            if self.auto_optimization_config.enable_optimization_logging:
                self.logger.debug(f"Starting auto-optimization for {self.config.name}")
                tprint(f"📝 Optimization logging enabled for {self.config.name}")

            # Apply automatic optimization
            if self.auto_optimization_config.enable_auto_optimization:
                tprint(f"🔧 Applying auto-optimization ({self.auto_optimization_config.optimization_level.value})...")
                data = self._auto_optimize_data(data)
                tprint("✅ Auto-optimization completed")
            else:
                tprint("⚠️ Auto-optimization disabled, using original data")

            # Call parent generate method
            tprint("📊 Generating feature...")
            result = super().generate(data, **kwargs)

            # Update auto-optimization stats
            optimization_time = time.time() - start_time
            self.auto_optimization_stats['total_optimizations'] += 1
            self.auto_optimization_stats['total_optimization_time'] += optimization_time

            tprint(f"✅ Feature generation completed in {optimization_time:.3f}s")
            tprint(f"📊 Success: {result.success}")

            # Add optimization info to result metadata
            if result.metadata is None:
                result.metadata = {}

            result.metadata.update({
                'auto_optimization_enabled': self.auto_optimization_config.enable_auto_optimization,
                'optimization_strategy': self.auto_optimization_config.optimization_level.value,
                'optimization_time': optimization_time,
                'optimization_stats': self.optimization_strategy.get_stats()
            })

            tprint(f"📈 Feature '{self.config.name}' generated successfully")
            return result

        except Exception as e:
            tprint(f"❌ Error generating feature '{self.config.name}': {e}")
            self.logger.error(f"Error generating feature '{self.config.name}': {e}")
            # Create a failed result
            from .feature_generator import FeatureResult
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
                tprint("⚠️ Auto-optimization disabled, returning original data")
                return data

            tprint(f"🔧 Applying {self.optimization_strategy.__class__.__name__} optimization...")
            tprint(f"📊 Input data shape: {data.shape}")

            # Use the configured optimization strategy
            optimized_data = self.optimization_strategy.optimize_data(data, self)

            tprint(f"✅ Optimization strategy completed")
            tprint(f"📊 Output data shape: {optimized_data.shape}")

            # Update memory savings stats
            if hasattr(self, 'get_optimization_stats'):
                opt_stats = self.get_optimization_stats()
                memory_saved = opt_stats.get('memory_saved_mb', 0.0)
                self.auto_optimization_stats['memory_savings_mb'] += memory_saved
                tprint(f"💾 Memory saved this optimization: {memory_saved:.2f}MB")

            if self.auto_optimization_config.enable_optimization_logging:
                self.logger.debug(f"Auto-optimization completed for {self.config.name}")
                tprint(f"📝 Optimization logging completed for {self.config.name}")

            tprint("✅ Auto-optimization completed successfully")
            return optimized_data

        except Exception as e:
            tprint(f"❌ Auto-optimization failed: {e}")
            self.logger.warning(f"Auto-optimization failed: {e}, using original data")
            tprint("🔄 Falling back to original data")
            return data

    def set_optimization_strategy(self, level: Union[str, OptimizationLevel]):
        """Change optimization strategy at runtime."""
        if isinstance(level, str):
            level = OptimizationLevel(level)

        self.auto_optimization_config.optimization_level = level
        self.optimization_strategy = self._create_optimization_strategy()

        if self.auto_optimization_config.enable_optimization_logging:
            self.logger.info(f"Optimization strategy changed to {level.value}")

    def get_auto_optimization_stats(self) -> Dict[str, Any]:
        """Get automatic optimization statistics."""
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

    def reset_auto_optimization_stats(self):
        """Reset automatic optimization statistics."""
        self.auto_optimization_stats = {
            'total_optimizations': 0,
            'total_optimization_time': 0.0,
            'memory_savings_mb': 0.0,
            'strategy_used': self.auto_optimization_config.optimization_level.value
        }
        self.optimization_strategy.reset_stats()

    def _should_use_vectorbt(self, data: pd.DataFrame) -> bool:
        """Determine if VectorBT optimization should be used."""
        if not self.auto_optimization_config.enable_vectorbt_optimization:
            return False

        # Check if data size exceeds threshold
        if len(data) < self.auto_optimization_config.vectorbt_threshold:
            return False

        # Check if data has numeric columns suitable for VectorBT
        numeric_columns = data.select_dtypes(include=['number']).columns
        if len(numeric_columns) == 0:
            return False

        return True

    def enable_auto_optimization(self, enabled: bool = True):
        """Enable or disable automatic optimization."""
        self.auto_optimization_config.enable_auto_optimization = enabled

        if self.auto_optimization_config.enable_optimization_logging:
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"Auto-optimization {status}")

    def get_optimization_config(self) -> AutoOptimizationConfig:
        """Get current optimization configuration."""
        return self.auto_optimization_config

    def update_optimization_config(self, **kwargs):
        """Update optimization configuration."""
        for key, value in kwargs.items():
            if hasattr(self.auto_optimization_config, key):
                setattr(self.auto_optimization_config, key, value)

        # Recreate strategy if optimization level changed
        if 'optimization_level' in kwargs:
            self.optimization_strategy = self._create_optimization_strategy()

        if self.auto_optimization_config.enable_optimization_logging:
            self.logger.info(f"Optimization configuration updated: {kwargs}")
