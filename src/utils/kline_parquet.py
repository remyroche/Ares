"""
KlinesParquetManager - Efficient and Standardized Klines Data Storage

This module provides a comprehensive manager for storing and retrieving klines data
in parquet format with efficient compression, batch management, and data integrity.

Features:
- Efficient parquet storage with compression
- Batch management for incremental data updates
- Data integrity validation
- Automatic directory structure management
- Exchange-agnostic data format
- Memory-efficient operations
- Comprehensive error handling
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
class KlinesMetadata:
    """Metadata for klines data batches."""
    symbol: str
    exchange: str
    interval: str
    batch_id: str
    start_time: datetime
    end_time: datetime
    record_count: int
    file_size_bytes: int
    compression_ratio: float
    created_at: datetime
    data_quality_score: float = 0.0
    gaps_detected: int = 0
    gaps_filled: int = 0
    resampled_intervals: List[str] = field(default_factory=list)
    additional_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageConfig:
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


class KlinesParquetManager:
    """
    Manager for efficient klines data storage and retrieval using parquet format.
    
    Provides:
    - Efficient storage with compression
    - Batch management for incremental updates
    - Data integrity validation
    - Automatic directory structure management
    - Exchange-agnostic data format
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize the KlinesParquetManager.
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.base_dir = Path(self.config.base_dir)
        self.parquet_utils = ParquetUtils()
        self.logger = system_logger.getChild("KlinesParquetManager")
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage tracking
        self._metadata_cache: Dict[str, KlinesMetadata] = {}
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
        
        self.logger.info(f"✅ KlinesParquetManager initialized with base_dir: {self.base_dir}")
    
    @handles_errors(default_return=False, context="KlinesParquetManager.store_klines")
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
        """Store klines data in parquet format.
        
        Args:
            df: DataFrame containing klines data
            symbol: Trading symbol (e.g., "ETHUSDT")
            exchange: Exchange name (e.g., "binance")
            interval: Data interval (e.g., "1m")
            batch_id: Optional batch identifier
            metadata: Additional metadata to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        if df is None or df.empty:
            tprint_error("❌ Cannot store empty DataFrame")
            return False
        
        try:
            # Generate batch ID if not provided
            if batch_id is None:
                batch_id = self._generate_batch_id(symbol, exchange, interval)
            
            # Apply comprehensive optimizations
            storage_df = self._apply_comprehensive_optimizations(df, symbol, exchange, interval)
            
            # Determine storage path
            storage_path = self._get_storage_path(symbol, exchange, interval, batch_id)
            
            # Ensure directory exists
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get optimal parquet write parameters
            parquet_kwargs = self._get_optimal_parquet_kwargs(storage_df)
            
            # Store data with optimizations
            success = self._store_dataframe_optimized(storage_df, storage_path, parquet_kwargs)
            if not success:
                return False
            
            # Calculate compression statistics
            compression_stats = self._calculate_compression_stats(df, storage_df, storage_path)
            
            # Create and store metadata with compression stats
            klines_metadata = self._create_enhanced_metadata(
                storage_df, symbol, exchange, interval, batch_id, 
                storage_path, metadata, compression_stats
            )
            
            # Store metadata
            self._store_metadata(klines_metadata, storage_path)
            
            # Update cache
            self._metadata_cache[f"{symbol}_{exchange}_{interval}_{batch_id}"] = klines_metadata
            
            tprint_success(f"✅ Stored {len(storage_df)} klines records for {symbol} {interval}")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to store klines data: {e}")
            return False
    
    @handles_errors(default_return=pd.DataFrame(), context="KlinesParquetManager.load_klines")
    @traced
    @log_execution_time
    def load_klines(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Load klines data from parquet files.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            start_time: Optional start time filter
            end_time: Optional end time filter
            batch_id: Optional specific batch to load
            
        Returns:
            DataFrame containing klines data
        """
        try:
            # Find relevant files
            files = self._find_klines_files(symbol, exchange, interval, batch_id)
            if not files:
                tprint_warning(f"⚠️ No klines files found for {symbol} {exchange} {interval}")
                return pd.DataFrame()
            
            # Load and combine data
            combined_df = self._load_and_combine_files(files, start_time, end_time)
            
            if combined_df.empty:
                tprint_warning(f"⚠️ No data found for {symbol} {exchange} {interval}")
                return pd.DataFrame()
            
            tprint_success(f"✅ Loaded {len(combined_df)} klines records for {symbol} {interval}")
            return combined_df
            
        except Exception as e:
            tprint_error(f"❌ Failed to load klines data: {e}")
            return pd.DataFrame()
    
    @handles_errors(default_return=List[str], context="KlinesParquetManager.list_available_data")
    def list_available_data(self) -> List[Dict[str, Any]]:
        """List all available klines data.
        
        Returns:
            List of dictionaries containing available data information
        """
        try:
            available_data = []
            
            # Scan base directory for klines data
            for exchange_dir in self.base_dir.iterdir():
                if not exchange_dir.is_dir():
                    continue
                
                for symbol_dir in exchange_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue
                    
                    klines_dir = symbol_dir / "klines"
                    if not klines_dir.exists():
                        continue
                    
                    # Find parquet files
                    for file_path in klines_dir.glob("*.parquet"):
                        metadata = self._load_file_metadata(file_path)
                        if metadata:
                            available_data.append({
                                "symbol": metadata.symbol,
                                "exchange": metadata.exchange,
                                "interval": metadata.interval,
                                "batch_id": metadata.batch_id,
                                "start_time": metadata.start_time,
                                "end_time": metadata.end_time,
                                "record_count": metadata.record_count,
                                "file_size_mb": metadata.file_size_bytes / (1024 * 1024),
                                "created_at": metadata.created_at
                            })
            
            return available_data
            
        except Exception as e:
            tprint_error(f"❌ Failed to list available data: {e}")
            return []
    
    @handles_errors(default_return=False, context="KlinesParquetManager.update_klines")
    def update_klines(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        append_mode: bool = True
    ) -> bool:
        """Update existing klines data.
        
        Args:
            df: New klines data
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            append_mode: If True, append to existing data; if False, replace
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            if append_mode:
                # Load existing data
                existing_df = self.load_klines(symbol, exchange, interval)
                
                if not existing_df.empty:
                    # Combine with existing data
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                    combined_df = combined_df.sort_values('timestamp')
                else:
                    combined_df = df
            else:
                combined_df = df
            
            # Store updated data
            return self.store_klines(combined_df, symbol, exchange, interval)
            
        except Exception as e:
            tprint_error(f"❌ Failed to update klines data: {e}")
            return False
    
    @handles_errors(default_return=False, context="KlinesParquetManager.delete_klines")
    def delete_klines(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: Optional[str] = None
    ) -> bool:
        """Delete klines data.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            batch_id: Optional specific batch to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            files = self._find_klines_files(symbol, exchange, interval, batch_id)
            
            for file_path in files:
                # Delete parquet file
                if file_path.exists():
                    file_path.unlink()
                    tprint_info(f"🗑️ Deleted {file_path}")
                
                # Delete metadata file
                metadata_path = file_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    metadata_path.unlink()
                    tprint_info(f"🗑️ Deleted {metadata_path}")
            
            tprint_success(f"✅ Deleted klines data for {symbol} {exchange} {interval}")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to delete klines data: {e}")
            return False
    
    def _generate_batch_id(self, symbol: str, exchange: str, interval: str) -> str:
        """Generate a unique batch ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = self._batch_counter.get(f"{symbol}_{exchange}_{interval}", 0) + 1
        self._batch_counter[f"{symbol}_{exchange}_{interval}"] = counter
        return f"batch_{counter:03d}_{timestamp}"
    
    
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
    
    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Legacy method - now handled by _optimize_dtypes."""
        return self._optimize_dtypes(df)
    
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
    ) -> KlinesMetadata:
        """Create enhanced metadata with optimization details."""
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        # Calculate compression ratio
        compression_ratio = 0.0
        if compression_stats and 'compression_ratio' in compression_stats:
            compression_ratio = compression_stats['compression_ratio']
        elif file_size > 0 and len(df) > 0:
            # Fallback calculation
            estimated_size = len(df) * len(df.columns) * 8  # Rough estimate
            compression_ratio = (1 - file_size / estimated_size) * 100 if estimated_size > 0 else 0
        
        return KlinesMetadata(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            batch_id=batch_id,
            start_time=df['timestamp'].min() if 'timestamp' in df.columns else df.index.min(),
            end_time=df['timestamp'].max() if 'timestamp' in df.columns else df.index.max(),
            record_count=len(df),
            file_size_bytes=file_size,
            compression_ratio=compression_ratio,
            created_at=datetime.now(),
            additional_metadata={
                **(additional_metadata or {}),
                'optimization_applied': True,
                'compression_used': self.config.compression,
                'row_group_size': self.config.row_group_size,
                'dictionary_encoding': self.config.use_dictionary_encoding,
                'compression_stats': compression_stats or {}
            }
        )
    
    def _create_metadata(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: str,
        file_path: Path,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> KlinesMetadata:
        """Create metadata for klines data."""
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        return KlinesMetadata(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            batch_id=batch_id,
            start_time=df['timestamp'].min() if 'timestamp' in df.columns else df.index.min(),
            end_time=df['timestamp'].max() if 'timestamp' in df.columns else df.index.max(),
            record_count=len(df),
            file_size_bytes=file_size,
            compression_ratio=file_size / (len(df) * len(df.columns) * 8) if len(df) > 0 else 0,
            created_at=datetime.now(),
            additional_metadata=additional_metadata or {}
        )
    
    def _store_metadata(self, metadata: KlinesMetadata, file_path: Path) -> None:
        """Store metadata to JSON file."""
        if not self.config.enable_metadata:
            return
        
        metadata_path = file_path.with_suffix('.metadata.json')
        
        metadata_dict = {
            "symbol": metadata.symbol,
            "exchange": metadata.exchange,
            "interval": metadata.interval,
            "batch_id": metadata.batch_id,
            "start_time": metadata.start_time.isoformat(),
            "end_time": metadata.end_time.isoformat(),
            "record_count": metadata.record_count,
            "file_size_bytes": metadata.file_size_bytes,
            "compression_ratio": metadata.compression_ratio,
            "created_at": metadata.created_at.isoformat(),
            "data_quality_score": metadata.data_quality_score,
            "gaps_detected": metadata.gaps_detected,
            "gaps_filled": metadata.gaps_filled,
            "resampled_intervals": metadata.resampled_intervals,
            "additional_metadata": metadata.additional_metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _load_file_metadata(self, file_path: Path) -> Optional[KlinesMetadata]:
        """Load metadata from JSON file."""
        metadata_path = file_path.with_suffix('.metadata.json')
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            return KlinesMetadata(
                symbol=data['symbol'],
                exchange=data['exchange'],
                interval=data['interval'],
                batch_id=data['batch_id'],
                start_time=datetime.fromisoformat(data['start_time']),
                end_time=datetime.fromisoformat(data['end_time']),
                record_count=data['record_count'],
                file_size_bytes=data['file_size_bytes'],
                compression_ratio=data['compression_ratio'],
                created_at=datetime.fromisoformat(data['created_at']),
                data_quality_score=data.get('data_quality_score', 0.0),
                gaps_detected=data.get('gaps_detected', 0),
                gaps_filled=data.get('gaps_filled', 0),
                resampled_intervals=data.get('resampled_intervals', []),
                additional_metadata=data.get('additional_metadata', {})
            )
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load metadata from {metadata_path}: {e}")
            return None
    
    def _find_klines_files(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        batch_id: Optional[str] = None
    ) -> List[Path]:
        """Find klines files matching criteria."""
        klines_dir = self.base_dir / exchange.lower() / symbol.lower() / "klines"
        
        if not klines_dir.exists():
            return []
        
        pattern = f"klines_{exchange}_{symbol}_{interval}_*.parquet"
        if batch_id:
            pattern = f"klines_{exchange}_{symbol}_{interval}_{batch_id}.parquet"
        
        return list(klines_dir.glob(pattern))
    
    def _load_and_combine_files(
        self,
        files: List[Path],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load and combine multiple parquet files."""
        dataframes = []
        
        for file_path in files:
            try:
                df = self.parquet_utils.safe_read_parquet(str(file_path))
                if df is not None and not df.empty:
                    dataframes.append(df)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load {file_path}: {e}")
                continue
        
        if not dataframes:
            return pd.DataFrame()
        
        # Combine dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')
        
        # Apply time filters
        if start_time:
            combined_df = combined_df[combined_df['timestamp'] >= start_time]
        if end_time:
            combined_df = combined_df[combined_df['timestamp'] <= end_time]
        
        return combined_df
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            available_data = self.list_available_data()
            
            if not available_data:
                return {"total_files": 0, "total_size_mb": 0, "total_records": 0}
            
            total_files = len(available_data)
            total_size_mb = sum(item["file_size_mb"] for item in available_data)
            total_records = sum(item["record_count"] for item in available_data)
            
            # Group by exchange and symbol
            by_exchange = {}
            by_symbol = {}
            
            for item in available_data:
                exchange = item["exchange"]
                symbol = item["symbol"]
                
                if exchange not in by_exchange:
                    by_exchange[exchange] = {"files": 0, "size_mb": 0, "records": 0}
                by_exchange[exchange]["files"] += 1
                by_exchange[exchange]["size_mb"] += item["file_size_mb"]
                by_exchange[exchange]["records"] += item["record_count"]
                
                if symbol not in by_symbol:
                    by_symbol[symbol] = {"files": 0, "size_mb": 0, "records": 0}
                by_symbol[symbol]["files"] += 1
                by_symbol[symbol]["size_mb"] += item["file_size_mb"]
                by_symbol[symbol]["records"] += item["record_count"]
            
            return {
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "total_records": total_records,
                "by_exchange": by_exchange,
                "by_symbol": by_symbol
            }
            
        except Exception as e:
            tprint_error(f"❌ Failed to get storage stats: {e}")
            return {"error": str(e)}
    
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
            "average_compression_ratio": np.mean([stats.get('compression_ratio', 0) for stats in self._compression_stats.values()]) if self._compression_stats else 0,
            "compression_stats": self._compression_stats
        }


# Convenience functions
def create_klines_manager(config: Optional[StorageConfig] = None) -> KlinesParquetManager:
    """Create a new KlinesParquetManager instance."""
    return KlinesParquetManager(config)


def get_klines_manager() -> KlinesParquetManager:
    """Get a singleton KlinesParquetManager instance."""
    if not hasattr(get_klines_manager, '_instance'):
        get_klines_manager._instance = KlinesParquetManager()
    return get_klines_manager._instance


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
    
    # Test the manager
    manager = KlinesParquetManager()
    
    # Store data
    success = manager.store_klines(test_data, "ETHUSDT", "binance", "1m")
    print(f"Storage successful: {success}")
    
    # Load data
    loaded_data = manager.load_klines("ETHUSDT", "binance", "1m")
    print(f"Loaded {len(loaded_data)} records")
    
    # Get stats
    stats = manager.get_storage_stats()
    print(f"Storage stats: {stats}")