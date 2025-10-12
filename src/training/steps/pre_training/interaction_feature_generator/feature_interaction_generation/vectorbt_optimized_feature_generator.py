"""
VectorBT Optimized Feature Generator

This module provides optimized feature generation using VectorBT rolling operations
and unified vectorization management for maximum performance.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import warnings

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
        optimized_rolling_mean, optimized_rolling_std, optimized_rolling_var,
        optimized_rolling_min, optimized_rolling_max, optimized_rolling_sum,
        optimized_rolling_quantile, optimized_rolling_apply,
        optimized_rolling_corr, optimized_rolling_cov
    )
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager,
        OperationType, OptimizationStrategy, optimize_financial_operation
    )
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    warnings.warn(f"VectorBT optimizations not available: {e}")

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress
)


@dataclass
class VectorBTFeatureConfig:
    """Configuration for VectorBT optimized feature generation."""
    # Rolling operations settings
    enable_vectorbt_rolling: bool = True
    vectorbt_window_threshold: int = 1000  # Use VectorBT for windows >= this size
    vectorbt_correlation_threshold: int = 500  # Use VectorBT for correlation with >= this data points
    
    # Performance settings
    enable_gpu: bool = True
    enable_parallel: bool = True
    chunk_size: int = 50000
    memory_limit_gb: float = 8.0
    
    # Feature generation settings
    rolling_windows: List[int] = None
    correlation_windows: List[int] = None
    quantile_levels: List[float] = None
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50, 100, 200]
        if self.correlation_windows is None:
            self.correlation_windows = [20, 50, 100]
        if self.quantile_levels is None:
            self.quantile_levels = [0.25, 0.5, 0.75, 0.9, 0.95]


class VectorBTOptimizedFeatureGenerator:
    """
    VectorBT optimized feature generator with intelligent operation selection.
    
    This generator automatically selects the best optimization strategy for each
    operation based on data size, available hardware, and operation type.
    """
    
    def __init__(self, config: Optional[VectorBTFeatureConfig] = None):
        """Initialize the VectorBT optimized feature generator."""
        self.config = config or VectorBTFeatureConfig()
        
        # Initialize VectorBT components
        self._initialize_vectorbt_components()
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'total_operations': 0,
            'total_time': 0.0,
            'memory_optimizations': 0
        }
        
        tprint_success("🚀 VectorBT Optimized Feature Generator initialized")
        tprint_info(f"📊 VectorBT rolling: {'✅' if self.config.enable_vectorbt_rolling else '❌'}")
        tprint_info(f"📊 GPU acceleration: {'✅' if self.config.enable_gpu else '❌'}")
        tprint_info(f"📊 Parallel processing: {'✅' if self.config.enable_parallel else '❌'}")
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT optimization components."""
        if not VECTORBT_OPTIMIZATIONS_AVAILABLE:
            tprint_warning("⚠️ VectorBT optimizations not available, using fallback methods")
            self.vectorbt_rolling_optimizer = None
            self.unified_vectorization_manager = None
            return

        try:
            # Initialize VectorBT rolling optimizer
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel
            )
            tprint_success("✅ VectorBT rolling optimizer initialized")

            # Initialize unified vectorization manager
            self.unified_vectorization_manager = get_unified_vectorization_manager()
            tprint_success("✅ Unified vectorization manager initialized")

            # Configure settings
            if hasattr(self.vectorbt_rolling_optimizer, 'chunk_size'):
                self.vectorbt_rolling_optimizer.chunk_size = self.config.chunk_size

        except Exception as e:
            tprint_error(f"❌ Failed to initialize VectorBT components: {e}")
            self.vectorbt_rolling_optimizer = None
            self.unified_vectorization_manager = None
    
    def generate_rolling_features(self, data: pd.DataFrame, 
                                target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Generate rolling features using VectorBT optimizations.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_column: Optional target column for correlation features
            
        Returns:
            DataFrame with generated rolling features
        """
        tprint_debug("🔧 Generating rolling features with VectorBT optimizations...")
        start_time = time.time()
        
        features = {}
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            tprint_warning(f"⚠️ Missing required columns: {missing_cols}")
            return pd.DataFrame(index=data.index)
        
        # Generate rolling features for each window
        for window in self.config.rolling_windows:
            tprint_debug(f"🔧 Processing rolling window: {window}")
            
            # Price-based rolling features
            features.update(self._generate_price_rolling_features(data, window))
            
            # Volume-based rolling features
            features.update(self._generate_volume_rolling_features(data, window))
            
            # Volatility rolling features
            features.update(self._generate_volatility_rolling_features(data, window))
            
            # Correlation features if target column provided
            if target_column and target_column in data.columns:
                features.update(self._generate_correlation_rolling_features(data, target_column, window))
        
        # Create result DataFrame
        if features:
            result = pd.DataFrame(features, index=data.index)
            result = result.dropna(axis=1, how='all')
            
            # Update performance stats
            self.performance_stats['total_operations'] += len(self.config.rolling_windows)
            self.performance_stats['total_time'] += time.time() - start_time
            
            tprint_success(f"✅ Generated {len(result.columns)} rolling features in {time.time() - start_time:.3f}s")
            return result
        else:
            tprint_warning("⚠️ No rolling features generated")
            return pd.DataFrame(index=data.index)
    
    def _generate_price_rolling_features(self, data: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        """Generate price-based rolling features."""
        features = {}
        
        # Price columns
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col not in data.columns:
                continue
                
            series = data[col]
            
            # Use VectorBT if available and data size is above threshold
            if (self.vectorbt_rolling_optimizer and 
                self.config.enable_vectorbt_rolling and 
                len(series) >= self.config.vectorbt_window_threshold):
                
                try:
                    # Rolling mean
                    features[f'{col}_rolling_mean_{window}'] = self.vectorbt_rolling_optimizer.rolling_mean(series, window)
                    
                    # Rolling std
                    features[f'{col}_rolling_std_{window}'] = self.vectorbt_rolling_optimizer.rolling_std(series, window)
                    
                    # Rolling min/max
                    features[f'{col}_rolling_min_{window}'] = self.vectorbt_rolling_optimizer.rolling_min(series, window)
                    features[f'{col}_rolling_max_{window}'] = self.vectorbt_rolling_optimizer.rolling_max(series, window)
                    
                    # Rolling quantiles
                    for q in self.config.quantile_levels:
                        features[f'{col}_rolling_q{int(q*100)}_{window}'] = self.vectorbt_rolling_optimizer.rolling_quantile(series, window, q=q)
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    
                except Exception as e:
                    tprint_warning(f"⚠️ VectorBT operation failed for {col}, using pandas fallback: {e}")
                    features.update(self._pandas_rolling_features(series, col, window))
                    self.performance_stats['pandas_fallbacks'] += 1
            else:
                # Use pandas for smaller datasets
                features.update(self._pandas_rolling_features(series, col, window))
                self.performance_stats['pandas_fallbacks'] += 1
        
        return features
    
    def _generate_volume_rolling_features(self, data: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        """Generate volume-based rolling features."""
        features = {}
        
        if 'volume' not in data.columns:
            return features
        
        volume = data['volume']
        
        # Use VectorBT if available and data size is above threshold
        if (self.vectorbt_rolling_optimizer and 
            self.config.enable_vectorbt_rolling and 
            len(volume) >= self.config.vectorbt_window_threshold):
            
            try:
                # Volume rolling statistics
                features[f'volume_rolling_mean_{window}'] = self.vectorbt_rolling_optimizer.rolling_mean(volume, window)
                features[f'volume_rolling_std_{window}'] = self.vectorbt_rolling_optimizer.rolling_std(volume, window)
                features[f'volume_rolling_sum_{window}'] = self.vectorbt_rolling_optimizer.rolling_sum(volume, window)
                
                # Volume quantiles
                for q in self.config.quantile_levels:
                    features[f'volume_rolling_q{int(q*100)}_{window}'] = self.vectorbt_rolling_optimizer.rolling_quantile(volume, window, q=q)
                
                self.performance_stats['vectorbt_operations'] += 1
                
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT volume operation failed, using pandas fallback: {e}")
                features.update(self._pandas_volume_features(volume, window))
                self.performance_stats['pandas_fallbacks'] += 1
        else:
            # Use pandas for smaller datasets
            features.update(self._pandas_volume_features(volume, window))
            self.performance_stats['pandas_fallbacks'] += 1
        
        return features
    
    def _generate_volatility_rolling_features(self, data: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        """Generate volatility-based rolling features."""
        features = {}
        
        # Calculate returns for volatility features
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            
            # Use VectorBT if available and data size is above threshold
            if (self.vectorbt_rolling_optimizer and 
                self.config.enable_vectorbt_rolling and 
                len(returns) >= self.config.vectorbt_window_threshold):
                
                try:
                    # Rolling volatility (std of returns)
                    features[f'volatility_rolling_std_{window}'] = self.vectorbt_rolling_optimizer.rolling_std(returns, window)
                    
                    # Rolling variance
                    features[f'volatility_rolling_var_{window}'] = self.vectorbt_rolling_optimizer.rolling_var(returns, window)
                    
                    # Rolling skewness and kurtosis
                    features[f'volatility_rolling_skew_{window}'] = self.vectorbt_rolling_optimizer.rolling_skew(returns, window)
                    features[f'volatility_rolling_kurt_{window}'] = self.vectorbt_rolling_optimizer.rolling_kurt(returns, window)
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    
                except Exception as e:
                    tprint_warning(f"⚠️ VectorBT volatility operation failed, using pandas fallback: {e}")
                    features.update(self._pandas_volatility_features(returns, window))
                    self.performance_stats['pandas_fallbacks'] += 1
            else:
                # Use pandas for smaller datasets
                features.update(self._pandas_volatility_features(returns, window))
                self.performance_stats['pandas_fallbacks'] += 1
        
        return features
    
    def _generate_correlation_rolling_features(self, data: pd.DataFrame, target_column: str, window: int) -> Dict[str, pd.Series]:
        """Generate correlation-based rolling features."""
        features = {}
        
        if target_column not in data.columns:
            return features
        
        target = data[target_column]
        
        # Use VectorBT if available and data size is above threshold
        if (self.vectorbt_rolling_optimizer and 
            self.config.enable_vectorbt_rolling and 
            len(target) >= self.config.vectorbt_correlation_threshold):
            
            try:
                # Correlate with price columns
                for col in ['open', 'high', 'low', 'close']:
                    if col in data.columns and col != target_column:
                        corr = self.vectorbt_rolling_optimizer.rolling_corr(target, data[col], window)
                        features[f'corr_{target_column}_{col}_{window}'] = corr
                
                # Correlate with volume
                if 'volume' in data.columns:
                    corr = self.vectorbt_rolling_optimizer.rolling_corr(target, data['volume'], window)
                    features[f'corr_{target_column}_volume_{window}'] = corr
                
                self.performance_stats['vectorbt_operations'] += 1
                
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT correlation operation failed, using pandas fallback: {e}")
                features.update(self._pandas_correlation_features(data, target_column, window))
                self.performance_stats['pandas_fallbacks'] += 1
        else:
            # Use pandas for smaller datasets
            features.update(self._pandas_correlation_features(data, target_column, window))
            self.performance_stats['pandas_fallbacks'] += 1
        
        return features
    
    def _pandas_rolling_features(self, series: pd.Series, col: str, window: int) -> Dict[str, pd.Series]:
        """Generate rolling features using pandas (fallback)."""
        features = {}
        
        try:
            rolling = series.rolling(window=window)
            
            features[f'{col}_rolling_mean_{window}'] = rolling.mean()
            features[f'{col}_rolling_std_{window}'] = rolling.std()
            features[f'{col}_rolling_min_{window}'] = rolling.min()
            features[f'{col}_rolling_max_{window}'] = rolling.max()
            
            # Quantiles
            for q in self.config.quantile_levels:
                features[f'{col}_rolling_q{int(q*100)}_{window}'] = rolling.quantile(q)
                
        except Exception as e:
            tprint_warning(f"⚠️ Pandas rolling operation failed for {col}: {e}")
        
        return features
    
    def _pandas_volume_features(self, volume: pd.Series, window: int) -> Dict[str, pd.Series]:
        """Generate volume features using pandas (fallback)."""
        features = {}
        
        try:
            rolling = volume.rolling(window=window)
            
            features[f'volume_rolling_mean_{window}'] = rolling.mean()
            features[f'volume_rolling_std_{window}'] = rolling.std()
            features[f'volume_rolling_sum_{window}'] = rolling.sum()
            
            # Quantiles
            for q in self.config.quantile_levels:
                features[f'volume_rolling_q{int(q*100)}_{window}'] = rolling.quantile(q)
                
        except Exception as e:
            tprint_warning(f"⚠️ Pandas volume operation failed: {e}")
        
        return features
    
    def _pandas_volatility_features(self, returns: pd.Series, window: int) -> Dict[str, pd.Series]:
        """Generate volatility features using pandas (fallback)."""
        features = {}
        
        try:
            rolling = returns.rolling(window=window)
            
            features[f'volatility_rolling_std_{window}'] = rolling.std()
            features[f'volatility_rolling_var_{window}'] = rolling.var()
            features[f'volatility_rolling_skew_{window}'] = rolling.skew()
            features[f'volatility_rolling_kurt_{window}'] = rolling.kurt()
            
        except Exception as e:
            tprint_warning(f"⚠️ Pandas volatility operation failed: {e}")
        
        return features
    
    def _pandas_correlation_features(self, data: pd.DataFrame, target_column: str, window: int) -> Dict[str, pd.Series]:
        """Generate correlation features using pandas (fallback)."""
        features = {}
        
        try:
            target = data[target_column]
            
            # Correlate with price columns
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns and col != target_column:
                    corr = target.rolling(window=window).corr(data[col])
                    features[f'corr_{target_column}_{col}_{window}'] = corr
            
            # Correlate with volume
            if 'volume' in data.columns:
                corr = target.rolling(window=window).corr(data['volume'])
                features[f'corr_{target_column}_volume_{window}'] = corr
                
        except Exception as e:
            tprint_warning(f"⚠️ Pandas correlation operation failed: {e}")
        
        return features
    
    def generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate interaction features using VectorBT optimizations.
        
        Args:
            data: Input DataFrame with features
            
        Returns:
            DataFrame with generated interaction features
        """
        tprint_debug("🔧 Generating interaction features with VectorBT optimizations...")
        start_time = time.time()
        
        features = {}
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            tprint_warning("⚠️ Not enough numeric columns for interaction features")
            return pd.DataFrame(index=data.index)
        
        # Generate pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                try:
                    # Ratio interaction
                    features[f'{col1}_div_{col2}'] = data[col1] / (data[col2] + 1e-8)
                    
                    # Product interaction
                    features[f'{col1}_mul_{col2}'] = data[col1] * data[col2]
                    
                    # Difference interaction
                    features[f'{col1}_sub_{col2}'] = data[col1] - data[col2]
                    
                    # Sum interaction
                    features[f'{col1}_add_{col2}'] = data[col1] + data[col2]
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Interaction feature generation failed for {col1} and {col2}: {e}")
                    continue
        
        # Create result DataFrame
        if features:
            result = pd.DataFrame(features, index=data.index)
            result = result.dropna(axis=1, how='all')
            
            # Update performance stats
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_time'] += time.time() - start_time
            
            tprint_success(f"✅ Generated {len(result.columns)} interaction features in {time.time() - start_time:.3f}s")
            return result
        else:
            tprint_warning("⚠️ No interaction features generated")
            return pd.DataFrame(index=data.index)
    
    def generate_cross_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-timeframe features using VectorBT optimizations.
        
        Args:
            data: Input DataFrame with features
            
        Returns:
            DataFrame with generated cross-timeframe features
        """
        tprint_debug("🔧 Generating cross-timeframe features with VectorBT optimizations...")
        start_time = time.time()
        
        features = {}
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            tprint_warning("⚠️ No numeric columns for cross-timeframe features")
            return pd.DataFrame(index=data.index)
        
        # Generate cross-timeframe features for each column
        for col in numeric_cols:
            series = data[col]
            
            # Use VectorBT if available and data size is above threshold
            if (self.vectorbt_rolling_optimizer and 
                self.config.enable_vectorbt_rolling and 
                len(series) >= self.config.vectorbt_window_threshold):
                
                try:
                    # Cross-timeframe aggregations
                    for window in self.config.rolling_windows:
                        features[f'ctf_{col}_mean_{window}'] = self.vectorbt_rolling_optimizer.rolling_mean(series, window)
                        features[f'ctf_{col}_std_{window}'] = self.vectorbt_rolling_optimizer.rolling_std(series, window)
                        features[f'ctf_{col}_min_{window}'] = self.vectorbt_rolling_optimizer.rolling_min(series, window)
                        features[f'ctf_{col}_max_{window}'] = self.vectorbt_rolling_optimizer.rolling_max(series, window)
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    
                except Exception as e:
                    tprint_warning(f"⚠️ VectorBT cross-timeframe operation failed for {col}, using pandas fallback: {e}")
                    features.update(self._pandas_cross_timeframe_features(series, col))
                    self.performance_stats['pandas_fallbacks'] += 1
            else:
                # Use pandas for smaller datasets
                features.update(self._pandas_cross_timeframe_features(series, col))
                self.performance_stats['pandas_fallbacks'] += 1
        
        # Create result DataFrame
        if features:
            result = pd.DataFrame(features, index=data.index)
            result = result.dropna(axis=1, how='all')
            
            # Update performance stats
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_time'] += time.time() - start_time
            
            tprint_success(f"✅ Generated {len(result.columns)} cross-timeframe features in {time.time() - start_time:.3f}s")
            return result
        else:
            tprint_warning("⚠️ No cross-timeframe features generated")
            return pd.DataFrame(index=data.index)
    
    def _pandas_cross_timeframe_features(self, series: pd.Series, col: str) -> Dict[str, pd.Series]:
        """Generate cross-timeframe features using pandas (fallback)."""
        features = {}
        
        try:
            for window in self.config.rolling_windows:
                rolling = series.rolling(window=window)
                
                features[f'ctf_{col}_mean_{window}'] = rolling.mean()
                features[f'ctf_{col}_std_{window}'] = rolling.std()
                features[f'ctf_{col}_min_{window}'] = rolling.min()
                features[f'ctf_{col}_max_{window}'] = rolling.max()
                
        except Exception as e:
            tprint_warning(f"⚠️ Pandas cross-timeframe operation failed for {col}: {e}")
        
        return features
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['pandas_fallback_rate'] = stats['pandas_fallbacks'] / stats['total_operations']
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'total_operations': 0,
            'total_time': 0.0,
            'memory_optimizations': 0
        }


# Convenience functions
def create_vectorbt_feature_generator(config: Optional[VectorBTFeatureConfig] = None) -> VectorBTOptimizedFeatureGenerator:
    """Create a VectorBT optimized feature generator."""
    return VectorBTOptimizedFeatureGenerator(config)


def generate_vectorbt_features(data: pd.DataFrame, 
                             config: Optional[VectorBTFeatureConfig] = None,
                             target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Generate features using VectorBT optimizations.
    
    Args:
        data: Input DataFrame with OHLCV data
        config: Optional configuration
        target_column: Optional target column for correlation features
        
    Returns:
        DataFrame with generated features
    """
    generator = create_vectorbt_feature_generator(config)
    
    # Generate all types of features
    rolling_features = generator.generate_rolling_features(data, target_column)
    interaction_features = generator.generate_interaction_features(data)
    cross_timeframe_features = generator.generate_cross_timeframe_features(data)
    
    # Combine all features
    all_features = [rolling_features, interaction_features, cross_timeframe_features]
    valid_features = [f for f in all_features if not f.empty]
    
    if valid_features:
        result = pd.concat(valid_features, axis=1)
        # Remove duplicate columns if any
        result = result.loc[:, ~result.columns.duplicated(keep='first')]
        return result
    else:
        return pd.DataFrame(index=data.index)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.random.rand(n_samples) * 2,
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.random.rand(n_samples) * 2,
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.lognormal(10, 1, n_samples),
        'target': np.random.randn(n_samples).cumsum()
    })
    
    # Test VectorBT feature generator
    config = VectorBTFeatureConfig(
        enable_vectorbt_rolling=True,
        enable_gpu=True,
        enable_parallel=True,
        rolling_windows=[10, 20, 50],
        quantile_levels=[0.25, 0.5, 0.75]
    )
    
    generator = create_vectorbt_feature_generator(config)
    
    # Generate features
    features = generate_vectorbt_features(data, config, target_column='target')
    
    print(f"Generated {len(features.columns)} features")
    print(f"Performance stats: {generator.get_performance_stats()}")