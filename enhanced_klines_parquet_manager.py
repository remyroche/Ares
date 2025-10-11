#!/usr/bin/env python3
"""
Enhanced KlinesParquetManager with Advanced Optimizations

This module provides an enhanced version of KlinesParquetManager with additional
optimizations for better storage efficiency and performance.
"""

import os
import gc
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.parquet_utils import ParquetUtils
from src.core.decorators import handles_errors, traced, log_execution_time


@dataclass
class EnhancedStorageConfig:
    """Enhanced configuration for klines storage with optimization options."""
    base_dir: str = "historical_data"
    compression: str = "zstd"  # Better compression than snappy
    compression_level: int = 3  # ZSTD compression level
    index: bool = False  # Don't store index as separate column
    partition_by: List[str] = field(default_factory=lambda: ["exchange", "symbol", "interval"])
    max_file_size_mb: int = 100
    enable_metadata: bool = True
    enable_validation: bool = True
    row_group_size: int = 50000  # Optimized row group size
    use_dictionary_encoding: bool = True  # Enable dictionary encoding for categorical data
    enable_schema_optimization: bool = True  # Enable schema optimization
    enable_compression_analysis: bool = True  # Enable compression analysis


class EnhancedKlinesParquetManager:
    """
    Enhanced manager for efficient klines data storage and retrieval using parquet format.
    
    Provides advanced optimizations:
    - Multiple compression algorithms with automatic selection
    - Schema optimization and harmonization
    - Dictionary encoding for categorical data
    - Row group size optimization
    - Compression ratio analysis
    - Memory usage optimization
    """
    
    def __init__(self, config: Optional[EnhancedStorageConfig] = None):
        """Initialize the Enhanced KlinesParquetManager."""
        self.config = config or EnhancedStorageConfig()
        self.base_dir = Path(self.config.base_dir)
        self.parquet_utils = ParquetUtils()
        self.logger = system_logger.getChild("EnhancedKlinesParquetManager")
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage tracking
        self._metadata_cache: Dict[str, Any] = {}
        self._batch_counter: Dict[str, int] = {}
        self._compression_stats: Dict[str, Any] = {}
        
        # Column optimization mapping
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
        
        self.logger.info(f"✅ Enhanced KlinesParquetManager initialized with base_dir: {self.base_dir}")
    
    @handles_errors(default_return=False, context="EnhancedKlinesParquetManager.store_klines")
    @traced
    @log_execution_time
    def store_klines(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store klines data with advanced optimizations."""
        if df is None or df.empty:
            tprint_error("❌ Cannot store empty DataFrame")
            return False
        
        try:
            # Generate batch ID if not provided
            if batch_id is None:
                batch_id = self._generate_batch_id(symbol, exchange, interval)
            
            # Apply comprehensive optimizations
            optimized_df = self._apply_comprehensive_optimizations(df, symbol, exchange, interval)
            
            # Determine optimal storage path
            storage_path = self._get_storage_path(symbol, exchange, interval, batch_id)
            
            # Ensure directory exists
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get optimal parquet write parameters
            parquet_kwargs = self._get_optimal_parquet_kwargs(optimized_df)
            
            # Store data with optimizations
            success = self._store_dataframe_optimized(optimized_df, storage_path, parquet_kwargs)
            if not success:
                return False
            
            # Calculate compression statistics
            compression_stats = self._calculate_compression_stats(df, optimized_df, storage_path)
            
            # Create enhanced metadata
            klines_metadata = self._create_enhanced_metadata(
                optimized_df, symbol, exchange, interval, batch_id, 
                storage_path, metadata, compression_stats
            )
            
            # Store metadata
            self._store_metadata(klines_metadata, storage_path)
            
            # Update cache and stats
            self._metadata_cache[f"{symbol}_{exchange}_{interval}_{batch_id}"] = klines_metadata
            self._compression_stats[f"{symbol}_{exchange}_{interval}_{batch_id}"] = compression_stats
            
            tprint_success(f"✅ Stored {len(optimized_df)} klines records for {symbol} {interval}")
            tprint_info(f"   Compression ratio: {compression_stats['compression_ratio']:.1f}%")
            tprint_info(f"   File size: {compression_stats['file_size_mb']:.2f} MB")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to store klines data: {e}")
            return False
    
    def _apply_comprehensive_optimizations(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        exchange: str, 
        interval: str
    ) -> pd.DataFrame:
        """Apply comprehensive optimizations to DataFrame."""
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
        
        # 6. Add required columns if missing
        optimized_df = self._ensure_required_columns(optimized_df, symbol, exchange, interval)
        
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
                    self.logger.warning(f"Could not optimize {col} to {target_dtype}: {e}")
        
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
    
    def _ensure_required_columns(self, df: pd.DataFrame, symbol: str, exchange: str, interval: str) -> pd.DataFrame:
        """Ensure required columns exist."""
        optimized_df = df.copy()
        
        if 'exchange' not in optimized_df.columns:
            optimized_df['exchange'] = exchange
        if 'symbol' not in optimized_df.columns:
            optimized_df['symbol'] = symbol
        if 'interval' not in optimized_df.columns:
            optimized_df['interval'] = interval
        
        return optimized_df
    
    def _get_optimal_parquet_kwargs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get optimal parquet write parameters."""
        kwargs = {
            'engine': 'pyarrow',
            'index': self.config.index,
            'compression': self.config.compression,
        }
        
        # Add compression level for zstd
        if self.config.compression == 'zstd':
            kwargs['compression_level'] = self.config.compression_level
        
        # Row group size optimization
        if len(df) > 0:
            optimal_row_group_size = min(
                self.config.row_group_size,
                max(1000, len(df) // 10)  # At least 10 row groups
            )
            kwargs['row_group_size'] = optimal_row_group_size
        
        # Dictionary encoding for categorical columns
        if self.config.use_dictionary_encoding:
            categorical_columns = df.select_dtypes(include=['category']).columns.tolist()
            if categorical_columns:
                kwargs['use_dictionary'] = True
        
        return kwargs
    
    def _store_dataframe_optimized(self, df: pd.DataFrame, path: Path, kwargs: Dict[str, Any]) -> bool:
        """Store DataFrame with optimizations."""
        try:
            df.to_parquet(path, **kwargs)
            return True
        except Exception as e:
            tprint_error(f"❌ Failed to store optimized DataFrame: {e}")
            return False
    
    def _calculate_compression_stats(
        self, 
        original_df: pd.DataFrame, 
        optimized_df: pd.DataFrame, 
        file_path: Path
    ) -> Dict[str, Any]:
        """Calculate compression statistics."""
        if not file_path.exists():
            return {}
        
        original_size = original_df.memory_usage(deep=True).sum()
        file_size = file_path.stat().st_size
        compression_ratio = (1 - file_size / original_size) * 100 if original_size > 0 else 0
        
        return {
            'original_size_bytes': original_size,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'optimization_applied': True
        }
    
    def _create_enhanced_metadata(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: str,
        file_path: Path,
        additional_metadata: Optional[Dict[str, Any]] = None,
        compression_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create enhanced metadata with optimization details."""
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        metadata = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "batch_id": batch_id,
            "start_time": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
            "end_time": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None,
            "record_count": len(df),
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "created_at": datetime.now().isoformat(),
            "optimization_applied": True,
            "compression_used": self.config.compression,
            "row_group_size": self.config.row_group_size,
            "dictionary_encoding": self.config.use_dictionary_encoding,
        }
        
        # Add compression statistics
        if compression_stats:
            metadata.update(compression_stats)
        
        # Add additional metadata
        if additional_metadata:
            metadata["additional_metadata"] = additional_metadata
        
        return metadata
    
    def _generate_batch_id(self, symbol: str, exchange: str, interval: str) -> str:
        """Generate a unique batch ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = self._batch_counter.get(f"{symbol}_{exchange}_{interval}", 0) + 1
        self._batch_counter[f"{symbol}_{exchange}_{interval}"] = counter
        return f"batch_{counter:03d}_{timestamp}"
    
    def _get_storage_path(
        self, 
        symbol: str, 
        exchange: str, 
        interval: str, 
        batch_id: str
    ) -> Path:
        """Get the storage path for klines data."""
        return (
            self.base_dir / 
            exchange.lower() / 
            symbol.lower() / 
            "klines" / 
            f"klines_{exchange}_{symbol}_{interval}_{batch_id}.parquet"
        )
    
    def _store_metadata(self, metadata: Dict[str, Any], file_path: Path) -> None:
        """Store metadata to JSON file."""
        if not self.config.enable_metadata:
            return
        
        metadata_path = file_path.with_suffix('.metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_optimization_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get optimization recommendations based on data characteristics."""
        recommendations = {
            'compression': self.config.compression,
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
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get overall compression statistics."""
        if not self._compression_stats:
            return {"message": "No compression statistics available"}
        
        total_original_size = sum(stats.get('original_size_bytes', 0) for stats in self._compression_stats.values())
        total_file_size = sum(stats.get('file_size_bytes', 0) for stats in self._compression_stats.values())
        overall_compression_ratio = (1 - total_file_size / total_original_size) * 100 if total_original_size > 0 else 0
        
        return {
            "total_files": len(self._compression_stats),
            "total_original_size_mb": total_original_size / (1024 * 1024),
            "total_file_size_mb": total_file_size / (1024 * 1024),
            "overall_compression_ratio": overall_compression_ratio,
            "average_compression_ratio": np.mean([stats.get('compression_ratio', 0) for stats in self._compression_stats.values()]),
            "compression_stats": self._compression_stats
        }


# Convenience functions
def create_enhanced_klines_manager(config: Optional[EnhancedStorageConfig] = None) -> EnhancedKlinesParquetManager:
    """Create a new Enhanced KlinesParquetManager instance."""
    return EnhancedKlinesParquetManager(config)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create test data
    dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=1440, freq='1min')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(3000, 3100, 1440),
        'high': np.random.uniform(3100, 3200, 1440),
        'low': np.random.uniform(2900, 3000, 1440),
        'close': np.random.uniform(3000, 3100, 1440),
        'volume': np.random.uniform(100, 1000, 1440)
    })
    
    # Test the enhanced manager
    config = EnhancedStorageConfig(compression='zstd', row_group_size=10000)
    manager = EnhancedKlinesParquetManager(config)
    
    # Get recommendations
    recommendations = manager.get_optimization_recommendations(test_data)
    print("Optimization Recommendations:")
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    
    # Store data
    success = manager.store_klines(test_data, "ETHUSDT", "binance", "1m")
    print(f"Storage successful: {success}")
    
    # Get compression stats
    stats = manager.get_compression_stats()
    print(f"Compression stats: {stats}")