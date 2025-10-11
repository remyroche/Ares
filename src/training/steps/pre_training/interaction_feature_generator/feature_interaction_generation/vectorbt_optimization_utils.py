"""
VectorBT Optimization Utilities for Interactive Feature Generation

This module provides VectorBT-based optimizations that can be used alongside
the existing feature generation pipeline without modifying the core logic.

Key Features:
- VectorBT-optimized data processing utilities
- Memory-efficient data structure optimizations
- Performance monitoring and validation
- Integration helpers for existing pipeline
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.indicators import RSI, MACD, BollingerBands, SMA, EMA
    from vectorbt.utils import checks
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    RSI = None
    MACD = None
    BollingerBands = None
    SMA = None
    EMA = None

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class VectorBTOptimizationConfig:
    """Configuration for VectorBT optimization utilities."""
    # Performance settings
    use_gpu: bool = True
    chunk_size: int = 50000
    memory_limit_gb: float = 8.0
    enable_parallel: bool = True
    
    # Data optimization
    optimize_dtypes: bool = True
    enable_memory_monitoring: bool = True
    
    # Validation
    min_valid_ratio: float = 0.8
    max_constant_ratio: float = 0.1


class VectorBTOptimizer:
    """VectorBT optimization utilities for data processing and validation."""
    
    def __init__(self, config: Optional[VectorBTOptimizationConfig] = None):
        """Initialize the VectorBT optimizer."""
        self.config = config or VectorBTOptimizationConfig()
        
        if not VECTORBT_AVAILABLE:
            tprint_warning("⚠️ VectorBT not available - optimizations disabled")
            self.enabled = False
        else:
            self.enabled = True
            self._setup_vectorbt()
            tprint_success("🚀 VectorBT Optimizer initialized")
    
    def _setup_vectorbt(self):
        """Setup VectorBT configuration for optimal performance."""
        try:
            # Configure VectorBT for performance
            vbt.settings.set_theme("dark")
            vbt.settings['array_wrapper']['freq_precision'] = 0
            vbt.settings['array_wrapper']['freq_rep'] = 'auto'
            
            # Enable parallel processing if requested
            if self.config.enable_parallel:
                vbt.settings['array_wrapper']['freq_rep'] = 'auto'
            
            # Configure memory settings
            if hasattr(vbt.settings, 'memory'):
                vbt.settings['memory']['limit'] = self.config.memory_limit_gb * 1024**3
            
            tprint_debug("✅ VectorBT configuration applied")
            
        except Exception as e:
            tprint_warning(f"⚠️ Could not configure VectorBT settings: {e}")
    
    def optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes for memory efficiency using VectorBT utilities.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame with reduced memory usage
        """
        if not self.enabled:
            return self._basic_dtype_optimization(df)
        
        tprint_debug("🔧 Optimizing DataFrame dtypes with VectorBT...")
        
        try:
            # Convert to VectorBT format for optimization
            optimized_df = df.copy()
            
            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=['float64']).columns:
                # Use VectorBT's optimized numeric conversion
                optimized_df[col] = vbt.ArrayWrapper.from_1d(optimized_df[col]).values.astype(np.float32)
            
            for col in optimized_df.select_dtypes(include=['int64']).columns:
                # Use VectorBT's optimized integer conversion
                optimized_df[col] = vbt.ArrayWrapper.from_1d(optimized_df[col]).values.astype(np.int32)
            
            # Calculate memory savings
            original_memory = df.memory_usage(deep=True).sum()
            optimized_memory = optimized_df.memory_usage(deep=True).sum()
            savings = (original_memory - optimized_memory) / original_memory * 100
            
            tprint_success(f"✅ Memory optimization: {savings:.1f}% reduction")
            
            return optimized_df
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT dtype optimization failed: {e}")
            return self._basic_dtype_optimization(df)
    
    def _basic_dtype_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic dtype optimization when VectorBT is not available."""
        tprint_debug("🔧 Applying basic dtype optimization...")
        
        optimized_df = df.copy()
        
        # Optimize dtypes
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
        
        return optimized_df
    
    def validate_features_vectorbt(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate features using VectorBT utilities for enhanced performance.
        
        Args:
            features: DataFrame with features to validate
            
        Returns:
            Validation results dictionary
        """
        if not self.enabled:
            return self._basic_validation(features)
        
        tprint_debug("🔍 Validating features with VectorBT...")
        
        try:
            issues = []
            quality_metrics = {}
            
            # Convert to VectorBT format for validation
            vbt_features = vbt.ArrayWrapper.from_2d(features.values)
            
            # Check for infinite values using VectorBT
            inf_mask = vbt_features.isinf()
            inf_count = inf_mask.sum()
            if inf_count > 0:
                issues.append(f"Found {inf_count} infinite values")
            quality_metrics['infinite_ratio'] = inf_count / features.size
            
            # Check for NaN values using VectorBT
            nan_mask = vbt_features.isnan()
            nan_count = nan_mask.sum()
            nan_ratio = nan_count / features.size
            if nan_ratio > (1 - self.config.min_valid_ratio):
                issues.append(f"Too many NaN values: {nan_ratio:.1%}")
            quality_metrics['nan_ratio'] = nan_ratio
            
            # Check for constant features using VectorBT
            constant_cols = []
            for i in range(features.shape[1]):
                col_data = vbt_features[:, i]
                if col_data.nunique() <= 1:
                    constant_cols.append(features.columns[i])
            
            constant_ratio = len(constant_cols) / len(features.columns)
            if constant_ratio > self.config.max_constant_ratio:
                issues.append(f"Too many constant features: {constant_ratio:.1%}")
            quality_metrics['constant_ratio'] = constant_ratio
            
            # Calculate overall quality score
            quality_score = (1 - quality_metrics['infinite_ratio']) * (1 - quality_metrics['nan_ratio']) * (1 - quality_metrics['constant_ratio'])
            
            tprint_success(f"✅ VectorBT validation completed: {quality_score:.3f} quality score")
            
            return {
                'passed': len(issues) == 0,
                'quality_score': quality_score,
                'issues': issues,
                'metrics': quality_metrics
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT validation failed: {e}")
            return self._basic_validation(features)
    
    def _basic_validation(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Basic validation when VectorBT is not available."""
        if features.empty:
            return {
                'passed': False,
                'quality_score': 0.0,
                'issues': ['No features to validate']
            }
        
        issues = []
        quality_metrics = {}
        
        # Check for infinite values
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")
        quality_metrics['infinite_ratio'] = inf_count / (features.size or 1)
        
        # Check for NaN values
        nan_count = features.isnull().sum().sum()
        nan_ratio = nan_count / (features.size or 1)
        if nan_ratio > (1 - self.config.min_valid_ratio):
            issues.append(f"Too many NaN values: {nan_ratio:.1%}")
        quality_metrics['nan_ratio'] = nan_ratio
        
        # Check for constant features
        constant_cols = features.nunique() <= 1
        constant_count = constant_cols.sum()
        constant_ratio = constant_count / len(features.columns)
        if constant_ratio > self.config.max_constant_ratio:
            issues.append(f"Too many constant features: {constant_ratio:.1%}")
        quality_metrics['constant_ratio'] = constant_ratio
        
        # Calculate overall quality score
        quality_score = (1 - quality_metrics['infinite_ratio']) * (1 - quality_metrics['nan_ratio']) * (1 - quality_metrics['constant_ratio'])
        
        return {
            'passed': len(issues) == 0,
            'quality_score': quality_score,
            'issues': issues,
            'metrics': quality_metrics
        }
    
    def optimize_rolling_operations(self, data: pd.Series, window: int, operation: str = 'mean') -> pd.Series:
        """
        Optimize rolling operations using VectorBT for better performance.
        
        Args:
            data: Input series
            window: Rolling window size
            operation: Operation to perform ('mean', 'std', 'min', 'max', 'median')
            
        Returns:
            Optimized rolling result
        """
        if not self.enabled:
            return self._basic_rolling_operation(data, window, operation)
        
        try:
            # Use VectorBT for optimized rolling operations
            rolling = vbt.Rolling.from_1d(data, window=window)
            
            if operation == 'mean':
                result = rolling.mean()
            elif operation == 'std':
                result = rolling.std()
            elif operation == 'min':
                result = rolling.min()
            elif operation == 'max':
                result = rolling.max()
            elif operation == 'median':
                result = rolling.median()
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            return result.values
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT rolling operation failed: {e}")
            return self._basic_rolling_operation(data, window, operation)
    
    def _basic_rolling_operation(self, data: pd.Series, window: int, operation: str) -> pd.Series:
        """Basic rolling operation when VectorBT is not available."""
        if operation == 'mean':
            return data.rolling(window).mean()
        elif operation == 'std':
            return data.rolling(window).std()
        elif operation == 'min':
            return data.rolling(window).min()
        elif operation == 'max':
            return data.rolling(window).max()
        elif operation == 'median':
            return data.rolling(window).median()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def optimize_matrix_operations(self, data: pd.DataFrame, operation: str = 'correlation') -> pd.DataFrame:
        """
        Optimize matrix operations using VectorBT for better performance.
        
        Args:
            data: Input DataFrame
            operation: Operation to perform ('correlation', 'covariance')
            
        Returns:
            Optimized matrix result
        """
        if not self.enabled:
            return self._basic_matrix_operation(data, operation)
        
        try:
            # Use VectorBT for optimized matrix operations
            vbt_data = vbt.ArrayWrapper.from_2d(data.values)
            
            if operation == 'correlation':
                result = vbt_data.corr()
            elif operation == 'covariance':
                result = vbt_data.cov()
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            return pd.DataFrame(result, index=data.columns, columns=data.columns)
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT matrix operation failed: {e}")
            return self._basic_matrix_operation(data, operation)
    
    def _basic_matrix_operation(self, data: pd.DataFrame, operation: str) -> pd.DataFrame:
        """Basic matrix operation when VectorBT is not available."""
        if operation == 'correlation':
            return data.corr()
        elif operation == 'covariance':
            return data.cov()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the optimizer."""
        return {
            'vectorbt_available': VECTORBT_AVAILABLE,
            'gpu_available': CUPY_AVAILABLE,
            'optimizer_enabled': self.enabled,
            'config': self.config.__dict__
        }


# Convenience functions
def create_vectorbt_optimizer(config: Optional[VectorBTOptimizationConfig] = None) -> VectorBTOptimizer:
    """Create a VectorBT optimizer instance."""
    return VectorBTOptimizer(config)


def optimize_dataframe_with_vectorbt(df: pd.DataFrame, config: Optional[VectorBTOptimizationConfig] = None) -> pd.DataFrame:
    """
    Optimize a DataFrame using VectorBT utilities.
    
    Args:
        df: Input DataFrame
        config: VectorBT configuration
        
    Returns:
        Optimized DataFrame
    """
    optimizer = create_vectorbt_optimizer(config)
    return optimizer.optimize_dataframe_dtypes(df)


def validate_features_with_vectorbt(features: pd.DataFrame, config: Optional[VectorBTOptimizationConfig] = None) -> Dict[str, Any]:
    """
    Validate features using VectorBT utilities.
    
    Args:
        features: DataFrame with features to validate
        config: VectorBT configuration
        
    Returns:
        Validation results
    """
    optimizer = create_vectorbt_optimizer(config)
    return optimizer.validate_features_vectorbt(features)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'feature1': np.random.randn(n_samples).astype(np.float64),
        'feature2': np.random.randn(n_samples).astype(np.float64),
        'feature3': np.random.randn(n_samples).astype(np.float64),
        'feature4': np.random.randint(0, 100, n_samples).astype(np.int64),
        'feature5': np.random.randint(0, 10, n_samples).astype(np.int64)
    })
    
    print(f"Original data shape: {data.shape}")
    print(f"Original memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Test VectorBT optimization
    config = VectorBTOptimizationConfig(
        use_gpu=False,  # Use CPU for testing
        enable_parallel=True
    )
    
    optimizer = create_vectorbt_optimizer(config)
    
    # Test dtype optimization
    optimized_data = optimizer.optimize_dataframe_dtypes(data)
    print(f"Optimized memory usage: {optimized_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Test validation
    validation_result = optimizer.validate_features_vectorbt(optimized_data)
    print(f"Validation passed: {validation_result['passed']}")
    print(f"Quality score: {validation_result['quality_score']:.3f}")
    
    # Test rolling operations
    rolling_mean = optimizer.optimize_rolling_operations(data['feature1'], window=20, operation='mean')
    print(f"Rolling mean shape: {rolling_mean.shape}")
    
    # Test matrix operations
    correlation_matrix = optimizer.optimize_matrix_operations(data, operation='correlation')
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    
    # Get performance metrics
    metrics = optimizer.get_performance_metrics()
    print(f"Performance metrics: {metrics}")