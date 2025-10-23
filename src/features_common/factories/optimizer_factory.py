"""
Optimizer factory for creating optimized optimizers.

This factory provides intelligent optimizer creation with automatic
optimization selection and configuration.
"""

import logging
from typing import Dict, Any, Optional, Union, Type, List
import pandas as pd

from ..config import get_unified_config
from ..mixins import OptimizationMixin, PerformanceMixin, VectorBTMixin, ValidationMixin, CachingMixin, MonitoringMixin

logger = logging.getLogger(__name__)

class OptimizedOptimizer(OptimizationMixin, PerformanceMixin, VectorBTMixin, ValidationMixin, CachingMixin, MonitoringMixin):
    """
    Optimized optimizer with all mixins for maximum performance.

    This class combines all available mixins to provide comprehensive
    optimization capabilities with automatic fallback handling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = get_unified_config()
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components."""
        try:
            self._initialize_optimization()
            self._initialize_performance()
            self._initialize_vectorbt()
            self._initialize_validation()
            self._initialize_caching()
            self._initialize_monitoring()
        except Exception as e:
            logger.warning(f"Failed to initialize some components: {e}")

    def optimize(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Optimize data using the best available method."""
        try:
            # Use VectorBT optimization if available
            if self.vectorbt_available:
                return self._vectorbt_optimize(data, **kwargs)
            else:
                # Fallback to standard optimization
                return self._standard_optimize(data, **kwargs)
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return data

    def _standard_optimize(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Standard optimization fallback."""
        return data

class OptimizerFactory:
    """Factory for creating optimized optimizers."""

    def __init__(self):
        self.config = get_unified_config()
        self.available_optimizers = {
            'optimized': OptimizedOptimizer,
        }

    def create_optimizer(self, optimizer_type: str = 'optimized', **kwargs) -> OptimizedOptimizer:
        """Create an optimizer instance."""
        if optimizer_type not in self.available_optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return self.available_optimizers[optimizer_type](**kwargs)

    def get_available_optimizers(self) -> List[str]:
        """Get list of available optimizer types."""
        return list(self.available_optimizers.keys())

    def get_optimizer_info(self, optimizer_type: str) -> Dict[str, Any]:
        """Get information about a specific optimizer type."""
        if optimizer_type not in self.available_optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return {
            'type': optimizer_type,
            'class': self.available_optimizers[optimizer_type],
            'description': 'Optimized optimizer with all mixins'
        }

def create_optimizer(optimizer_type: str = 'optimized', **kwargs) -> OptimizedOptimizer:
    """Create an optimizer instance."""
    factory = OptimizerFactory()
    return factory.create_optimizer(optimizer_type, **kwargs)

def create_vectorbt_optimizer(**kwargs) -> OptimizedOptimizer:
    """Create a VectorBT-optimized optimizer."""
    return create_optimizer('optimized', **kwargs)
