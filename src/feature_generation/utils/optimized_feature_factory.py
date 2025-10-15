
"""
import warnings
Optimized Feature Factory

This module provides an optimized feature factory that automatically
uses the new optimization utilities for all feature generation.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory
from ..core.feature_bank import get_global_feature_bank
from .optimized_feature_pipeline import get_optimized_feature_pipeline, PipelineConfig
from .vectorization_optimizer import get_vectorization_optimizer, VectorizationConfig

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

logger = logging.getLogger(__name__)

class OptimizedFeatureFactory:
    """Factory for creating optimized feature generators."""
    
    def __init__(self, 
                 enable_pipeline_optimization: bool = True,
                 enable_vectorization_optimization: bool = True,
                 pipeline_config: Optional[PipelineConfig] = None,
                 vectorization_config: Optional[VectorizationConfig] = None):
        """
        Initialize the optimized feature factory.
        
        Args:
            enable_pipeline_optimization: Whether to enable pipeline optimization
            enable_vectorization_optimization: Whether to enable vectorization optimization
            pipeline_config: Optional pipeline configuration
            vectorization_config: Optional vectorization configuration
        """
        self.enable_pipeline_optimization = enable_pipeline_optimization
        self.enable_vectorization_optimization = enable_vectorization_optimization
        
        # Initialize optimization components
        if enable_pipeline_optimization:
            self.pipeline = get_optimized_feature_pipeline(pipeline_config)
        else:
            self.pipeline = None
            
        if enable_vectorization_optimization:
            self.vectorization_optimizer = get_vectorization_optimizer(vectorization_config)
        else:
            self.vectorization_optimizer = None
        
        # Get feature bank
        self.feature_bank = get_global_feature_bank()
        
        logger.info("✅ Optimized Feature Factory initialized")
    
    def generate_features_optimized(self, 
                                   data: pd.DataFrame,
                                   categories: Optional[List[Union[str, FeatureCategory]]] = None,
                                   features: Optional[List[str]] = None,
                                   target_column: Optional[str] = None,
                                   **kwargs) -> pd.DataFrame:
        """
        Generate features using the optimized pipeline.
        
        Args:
            data: Input DataFrame
            categories: List of feature categories
            features: List of specific features
            target_column: Target column for optimization
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with generated features
        """
        if self.pipeline:
            # Convert categories to strings if needed
            category_strings = []
            if categories:
                for cat in categories:
                    if isinstance(cat, FeatureCategory):
                        category_strings.append(cat.value)
                    else:
                        category_strings.append(cat)
            
            result = self.pipeline.process_features(
                data=data,
                categories=category_strings if category_strings else None,
                features=features,
                target_column=target_column,
                **kwargs
            )
            
            if result.success:
                logger.info(f"✅ Optimized feature generation completed in {result.processing_time:.3f}s")
                return result.features
            else:
                logger.warning(f"Optimized pipeline failed: {result.error_message}")
                # Fall back to standard feature bank
        else:
            logger.warning("Pipeline optimization not available, using standard feature bank")
        
        # Fallback to standard feature bank
        return self.feature_bank.generate_features(
            data=data,
            categories=categories,
            features=features,
            target_column=target_column,
            use_optimized_pipeline=False,  # Avoid recursion
            **kwargs
        )
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for processing."""
        if self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'factory_status': {
                'pipeline_optimization_enabled': self.pipeline is not None,
                'vectorization_optimization_enabled': self.vectorization_optimizer is not None
            }
        }
        
        if self.pipeline:
            report['pipeline_performance'] = self.pipeline.get_performance_report()
        
        if self.vectorization_optimizer:
            report['vectorization_performance'] = self.vectorization_optimizer.get_performance_report()
        
        return report

# Global factory instance
_optimized_factory: Optional[OptimizedFeatureFactory] = None

def get_optimized_feature_factory() -> OptimizedFeatureFactory:
    """Get or create the global optimized feature factory."""
    global _optimized_factory
    
    if _optimized_factory is None:
        _optimized_factory = OptimizedFeatureFactory()
    
    return _optimized_factory

def generate_features_optimized(data: pd.DataFrame,
                              categories: Optional[List[Union[str, FeatureCategory]]] = None,
                              features: Optional[List[str]] = None,
                              target_column: Optional[str] = None,
                              **kwargs) -> pd.DataFrame:
    """Convenience function for optimized feature generation."""
    factory = get_optimized_feature_factory()
    return factory.generate_features_optimized(data, categories, features, target_column, **kwargs)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
