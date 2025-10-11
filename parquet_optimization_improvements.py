#!/usr/bin/env python3
"""
Parquet Optimization Improvements for KlinesParquetManager

This script demonstrates additional optimizations that can be applied to improve
parquet storage efficiency and performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

class OptimizedKlinesParquetManager:
    """
    Enhanced KlinesParquetManager with additional optimizations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Enhanced compression options
        self.compression_options = {
            'zstd': {'compression': 'zstd', 'compression_level': 3},  # Better compression than snappy
            'lz4': {'compression': 'lz4'},  # Faster than snappy
            'snappy': {'compression': 'snappy'},  # Current default
            'gzip': {'compression': 'gzip'},  # Best compression, slower
        }
        
        # Optimal compression based on data characteristics
        self.compression = self.config.get('compression', 'zstd')
        
        # Partitioning strategy
        self.partition_strategy = self.config.get('partition_strategy', 'hive')
        
        # Row group size optimization
        self.row_group_size = self.config.get('row_group_size', 100000)  # 100k rows per group
        
        # Dictionary encoding for categorical columns
        self.use_dictionary_encoding = self.config.get('use_dictionary_encoding', True)
        
        # Column-specific optimizations
        self.column_optimizations = {
            'timestamp': {'dtype': 'datetime64[ns]', 'nullable': False},
            'open': {'dtype': 'float32', 'nullable': False},
            'high': {'dtype': 'float32', 'nullable': False},
            'low': {'dtype': 'float32', 'nullable': False},
            'close': {'dtype': 'float32', 'nullable': False},
            'volume': {'dtype': 'float32', 'nullable': False},
            'symbol': {'dtype': 'category', 'nullable': False},
            'exchange': {'dtype': 'category', 'nullable': False},
            'interval': {'dtype': 'category', 'nullable': False},
        }
    
    def optimize_dataframe_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive optimizations to DataFrame before parquet storage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        optimized_df = df.copy()
        
        # 1. Optimize data types
        optimized_df = self._optimize_dtypes(optimized_df)
        
        # 2. Sort data for better compression
        optimized_df = self._sort_for_compression(optimized_df)
        
        # 3. Remove unnecessary columns
        optimized_df = self._remove_unnecessary_columns(optimized_df)
        
        # 4. Optimize categorical data
        optimized_df = self._optimize_categorical_data(optimized_df)
        
        # 5. Handle missing values efficiently
        optimized_df = self._handle_missing_values(optimized_df)
        
        return optimized_df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for parquet storage."""
        optimized_df = df.copy()
        
        for col, config in self.column_optimizations.items():
            if col in optimized_df.columns:
                target_dtype = config['dtype']
                
                try:
                    if target_dtype == 'category':
                        # Use category for high-cardinality string columns
                        optimized_df[col] = optimized_df[col].astype('category')
                    elif target_dtype == 'datetime64[ns]':
                        # Ensure proper datetime format
                        if not pd.api.types.is_datetime64_any_dtype(optimized_df[col]):
                            optimized_df[col] = pd.to_datetime(optimized_df[col], utc=True)
                    else:
                        # Convert to target numeric type
                        optimized_df[col] = optimized_df[col].astype(target_dtype)
                        
                except Exception as e:
                    print(f"Warning: Could not optimize {col} to {target_dtype}: {e}")
        
        return optimized_df
    
    def _sort_for_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort data for better compression efficiency."""
        if 'timestamp' in df.columns:
            return df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that don't add value for storage."""
        # Keep only essential columns
        essential_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'symbol', 'exchange', 'interval'
        ]
        
        # Add any additional columns that exist
        existing_columns = [col for col in essential_columns if col in df.columns]
        additional_columns = [col for col in df.columns if col not in essential_columns]
        
        return df[existing_columns + additional_columns]
    
    def _optimize_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize categorical columns for better compression."""
        optimized_df = df.copy()
        
        categorical_columns = ['symbol', 'exchange', 'interval']
        for col in categorical_columns:
            if col in optimized_df.columns:
                # Use category type for better compression
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values efficiently."""
        optimized_df = df.copy()
        
        # For OHLCV data, forward fill missing values
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            if col in optimized_df.columns:
                if optimized_df[col].isnull().any():
                    optimized_df[col] = optimized_df[col].fillna(method='ffill')
        
        return optimized_df
    
    def get_optimal_parquet_kwargs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get optimal parquet write parameters based on data characteristics.
        
        Args:
            df: DataFrame to be stored
            
        Returns:
            Dictionary of optimal parquet write parameters
        """
        # Base parameters
        kwargs = {
            'engine': 'pyarrow',
            'index': False,  # Don't store index as separate column
            'compression': self.compression,
        }
        
        # Add compression-specific parameters
        if self.compression in self.compression_options:
            kwargs.update(self.compression_options[self.compression])
        
        # Row group size optimization
        if len(df) > 0:
            # Calculate optimal row group size based on data
            optimal_row_group_size = min(
                self.row_group_size,
                max(1000, len(df) // 10)  # At least 10 row groups
            )
            kwargs['row_group_size'] = optimal_row_group_size
        
        # Dictionary encoding for categorical columns
        if self.use_dictionary_encoding:
            categorical_columns = df.select_dtypes(include=['category']).columns.tolist()
            if categorical_columns:
                kwargs['use_dictionary'] = True
        
        # Schema optimization
        kwargs['schema'] = self._get_optimized_schema(df)
        
        return kwargs
    
    def _get_optimized_schema(self, df: pd.DataFrame) -> Optional[Any]:
        """Get optimized parquet schema."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Create optimized schema
            fields = []
            for col in df.columns:
                if col in self.column_optimizations:
                    config = self.column_optimizations[col]
                    if config['dtype'] == 'datetime64[ns]':
                        fields.append(pa.field(col, pa.timestamp('ns')))
                    elif config['dtype'] == 'float32':
                        fields.append(pa.field(col, pa.float32()))
                    elif config['dtype'] == 'category':
                        fields.append(pa.field(col, pa.dictionary(pa.int32(), pa.string())))
                    else:
                        fields.append(pa.field(col, pa.string()))
                else:
                    # Default to string for unknown columns
                    fields.append(pa.field(col, pa.string()))
            
            return pa.schema(fields)
            
        except ImportError:
            return None
    
    def calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        if original_size == 0:
            return 0.0
        return (1 - compressed_size / original_size) * 100
    
    def get_storage_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get storage recommendations based on data characteristics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with storage recommendations
        """
        recommendations = {
            'compression': self.compression,
            'estimated_size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
        
        # Compression recommendation based on data size
        if len(df) > 1000000:  # > 1M rows
            recommendations['compression'] = 'zstd'
            recommendations['reason'] = 'Large dataset - use zstd for better compression'
        elif len(df) > 100000:  # > 100K rows
            recommendations['compression'] = 'lz4'
            recommendations['reason'] = 'Medium dataset - use lz4 for good balance'
        else:
            recommendations['compression'] = 'snappy'
            recommendations['reason'] = 'Small dataset - use snappy for speed'
        
        # Row group size recommendation
        if len(df) > 0:
            optimal_row_groups = max(1, len(df) // 50000)  # ~50k rows per group
            recommendations['row_group_size'] = min(100000, len(df) // optimal_row_groups)
        
        return recommendations


def demonstrate_optimizations():
    """Demonstrate the optimization improvements."""
    print("🚀 Parquet Optimization Improvements Demo")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100000, freq='1min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(3000, 3100, 100000),
        'high': np.random.uniform(3100, 3200, 100000),
        'low': np.random.uniform(2900, 3000, 100000),
        'close': np.random.uniform(3000, 3100, 100000),
        'volume': np.random.uniform(100, 1000, 100000),
        'symbol': ['ETHUSDT'] * 100000,
        'exchange': ['binance'] * 100000,
        'interval': ['1m'] * 100000
    })
    
    print(f"📊 Sample data: {len(sample_data)} rows, {len(sample_data.columns)} columns")
    print(f"💾 Memory usage: {sample_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Initialize optimized manager
    manager = OptimizedKlinesParquetManager()
    
    # Get recommendations
    recommendations = manager.get_storage_recommendations(sample_data)
    print(f"\n📋 Storage Recommendations:")
    for key, value in recommendations.items():
        print(f"   {key}: {value}")
    
    # Optimize DataFrame
    optimized_data = manager.optimize_dataframe_for_parquet(sample_data)
    print(f"\n🔧 After optimization:")
    print(f"   Memory usage: {optimized_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"   Data types: {optimized_data.dtypes.to_dict()}")
    
    # Get optimal parquet parameters
    parquet_kwargs = manager.get_optimal_parquet_kwargs(optimized_data)
    print(f"\n⚙️ Optimal parquet parameters:")
    for key, value in parquet_kwargs.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    demonstrate_optimizations()