from typing import Dict, Any, Optional, Union
import pandas as pd
import logging
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
        
        self.logger = logger.getChild(f'AutoOptimized{self.__class__.__name__}')
        
        if self.auto_optimization_config.enable_optimization_logging:
            self.logger.info(f"Auto-optimization enabled with {self.auto_optimization_config.optimization_level.value} strategy")
    
    def _apply_level_settings(self):
        """Apply settings based on optimization level."""
        level_settings = self.auto_optimization_config.get_settings_for_level()
        
        for key, value in level_settings.items():
            if hasattr(self.auto_optimization_config, key):
                setattr(self.auto_optimization_config, key, value)
    
    def _create_optimization_strategy(self) -> OptimizationStrategy:
        """Create optimization strategy based on configuration."""
        if self.auto_optimization_config.optimization_level == OptimizationLevel.CONSERVATIVE:
            return ConservativeOptimizationStrategy(self.auto_optimization_config)
        elif self.auto_optimization_config.optimization_level == OptimizationLevel.BALANCED:
            return BalancedOptimizationStrategy(self.auto_optimization_config)
        elif self.auto_optimization_config.optimization_level == OptimizationLevel.AGGRESSIVE:
            return AggressiveOptimizationStrategy(self.auto_optimization_config)
        else:
            return BalancedOptimizationStrategy(self.auto_optimization_config)
    
    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """Generate feature with automatic optimization."""
        start_time = time.time()
        
        # Log optimization start
        if self.auto_optimization_config.enable_optimization_logging:
            self.logger.debug(f"Starting auto-optimization for {self.config.name}")
        
        # Apply automatic optimization
        if self.auto_optimization_config.enable_auto_optimization:
            data = self._auto_optimize_data(data)
        
        # Call parent generate method
        result = super().generate(data, **kwargs)
        
        # Update auto-optimization stats
        optimization_time = time.time() - start_time
        self.auto_optimization_stats['total_optimizations'] += 1
        self.auto_optimization_stats['total_optimization_time'] += optimization_time
        
        # Add optimization info to result metadata
        if result.metadata is None:
            result.metadata = {}
        
        result.metadata.update({
            'auto_optimization_enabled': self.auto_optimization_config.enable_auto_optimization,
            'optimization_strategy': self.auto_optimization_config.optimization_level.value,
            'optimization_time': optimization_time,
            'optimization_stats': self.optimization_strategy.get_stats()
        })
        
        return result
    
    def _auto_optimize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply automatic optimization using the configured strategy."""
        if not self.auto_optimization_config.enable_auto_optimization:
            return data
        
        try:
            # Use the configured optimization strategy
            optimized_data = self.optimization_strategy.optimize_data(data, self)
            
            # Update memory savings stats
            if hasattr(self, 'get_optimization_stats'):
                opt_stats = self.get_optimization_stats()
                self.auto_optimization_stats['memory_savings_mb'] += opt_stats.get('memory_saved_mb', 0.0)
            
            if self.auto_optimization_config.enable_optimization_logging:
                self.logger.debug(f"Auto-optimization completed for {self.config.name}")
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"Auto-optimization failed: {e}, using original data")
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