#!/usr/bin/env python3
"""
Common Utilities for Data Operations

This module provides comprehensive utilities for data formatting, analysis,
access control, and error handling in the data collection pipeline.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import gzip
import shutil

from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    ensure_directory,
    safe_json_dump,
    safe_json_load
)


class DataFormat(Enum):
    """Supported data formats."""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    PICKLE = "pickle"
    HDF5 = "hdf5"


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZ4 = "lz4"


@dataclass
class DataOperationResult:
    """Result of a data operation."""
    success: bool
    message: str
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    total_rows: int
    total_columns: int
    null_counts: Dict[str, int]
    duplicate_count: int
    data_types: Dict[str, str]
    memory_usage: int
    file_size: int
    quality_score: float
    issues: List[str]
    timestamp: str


class DataFormatter:
    """Utility class for data formatting operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def format_klines_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str = "1m"
    ) -> DataOperationResult:
        """Format klines data to standard structure."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Formatting klines data for {symbol} on {exchange}")
            
            # Required columns for klines data
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Check if required columns exist
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return DataOperationResult(
                    success=False,
                    message=f"Missing required columns: {missing_columns}",
                    execution_time=time.time() - start_time,
                    errors=[f"Missing columns: {missing_columns}"]
                )
            
            # Create a copy to avoid modifying original
            formatted_df = df.copy()
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(formatted_df['timestamp']):
                formatted_df['timestamp'] = pd.to_datetime(formatted_df['timestamp'])
            
            # Ensure numeric columns are numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in formatted_df.columns:
                    formatted_df[col] = pd.to_numeric(formatted_df[col], errors='coerce')
            
            # Sort by timestamp
            formatted_df = formatted_df.sort_values('timestamp').reset_index(drop=True)
            
            # Add metadata columns
            formatted_df['symbol'] = symbol
            formatted_df['exchange'] = exchange
            formatted_df['timeframe'] = timeframe
            
            # Validate OHLC integrity
            ohlc_issues = self._validate_ohlc_integrity(formatted_df)
            
            execution_time = time.time() - start_time
            
            return DataOperationResult(
                success=True,
                message="Klines data formatted successfully",
                data=formatted_df,
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "rows": len(formatted_df),
                    "columns": list(formatted_df.columns),
                    "ohlc_issues": ohlc_issues
                },
                execution_time=execution_time,
                warnings=ohlc_issues
            )
            
        except Exception as e:
            self.logger.exception(f"Error formatting klines data: {e}")
            return DataOperationResult(
                success=False,
                message=f"Error formatting klines data: {e}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def format_aggtrades_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> DataOperationResult:
        """Format aggtrades data to standard structure."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Formatting aggtrades data for {symbol} on {exchange}")
            
            # Required columns for aggtrades data
            required_columns = ['timestamp', 'price', 'quantity', 'trade_count']
            
            # Check if required columns exist
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return DataOperationResult(
                    success=False,
                    message=f"Missing required columns: {missing_columns}",
                    execution_time=time.time() - start_time,
                    errors=[f"Missing columns: {missing_columns}"]
                )
            
            # Create a copy to avoid modifying original
            formatted_df = df.copy()
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(formatted_df['timestamp']):
                formatted_df['timestamp'] = pd.to_datetime(formatted_df['timestamp'])
            
            # Ensure numeric columns are numeric
            numeric_columns = ['price', 'quantity', 'trade_count']
            for col in numeric_columns:
                if col in formatted_df.columns:
                    formatted_df[col] = pd.to_numeric(formatted_df[col], errors='coerce')
            
            # Sort by timestamp
            formatted_df = formatted_df.sort_values('timestamp').reset_index(drop=True)
            
            # Add metadata columns
            formatted_df['symbol'] = symbol
            formatted_df['exchange'] = exchange
            
            # Validate price and quantity integrity
            validation_issues = self._validate_aggtrades_integrity(formatted_df)
            
            execution_time = time.time() - start_time
            
            return DataOperationResult(
                success=True,
                message="Aggtrades data formatted successfully",
                data=formatted_df,
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "rows": len(formatted_df),
                    "columns": list(formatted_df.columns),
                    "validation_issues": validation_issues
                },
                execution_time=execution_time,
                warnings=validation_issues
            )
            
        except Exception as e:
            self.logger.exception(f"Error formatting aggtrades data: {e}")
            return DataOperationResult(
                success=False,
                message=f"Error formatting aggtrades data: {e}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _validate_ohlc_integrity(self, df: pd.DataFrame) -> List[str]:
        """Validate OHLC data integrity."""
        issues = []
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                negative_prices = (df[col] <= 0).sum()
                if negative_prices > 0:
                    issues.append(f"Found {negative_prices} negative/zero prices in {col}")
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= max(open, close)
            invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
            if invalid_high.any():
                issues.append(f"Found {invalid_high.sum()} rows where high < max(open, close)")
            
            # Low should be <= min(open, close)
            invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
            if invalid_low.any():
                issues.append(f"Found {invalid_low.sum()} rows where low > min(open, close)")
            
            # High should be >= low
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                issues.append(f"Found {invalid_hl.sum()} rows where high < low")
        
        return issues
    
    def _validate_aggtrades_integrity(self, df: pd.DataFrame) -> List[str]:
        """Validate aggtrades data integrity."""
        issues = []
        
        # Check for negative prices
        if 'price' in df.columns:
            negative_prices = (df['price'] <= 0).sum()
            if negative_prices > 0:
                issues.append(f"Found {negative_prices} negative/zero prices")
        
        # Check for negative quantities
        if 'quantity' in df.columns:
            negative_quantities = (df['quantity'] < 0).sum()
            if negative_quantities > 0:
                issues.append(f"Found {negative_quantities} negative quantities")
        
        # Check for negative trade counts
        if 'trade_count' in df.columns:
            negative_trades = (df['trade_count'] < 0).sum()
            if negative_trades > 0:
                issues.append(f"Found {negative_trades} negative trade counts")
        
        return issues


class DataAnalyzer:
    """Utility class for data analysis operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_data_quality(
        self,
        df: pd.DataFrame,
        data_type: str = "klines"
    ) -> DataQualityMetrics:
        """Analyze data quality and return metrics."""
        try:
            self.logger.info(f"Analyzing data quality for {data_type} data")
            
            # Basic metrics
            total_rows = len(df)
            total_columns = len(df.columns)
            
            # Null counts
            null_counts = df.isnull().sum().to_dict()
            
            # Duplicate count
            duplicate_count = df.duplicated().sum()
            
            # Data types
            data_types = df.dtypes.astype(str).to_dict()
            
            # Memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            
            # Quality score calculation
            quality_score = self._calculate_quality_score(
                total_rows, null_counts, duplicate_count, data_types
            )
            
            # Identify issues
            issues = self._identify_data_issues(df, data_type)
            
            return DataQualityMetrics(
                total_rows=total_rows,
                total_columns=total_columns,
                null_counts=null_counts,
                duplicate_count=duplicate_count,
                data_types=data_types,
                memory_usage=memory_usage,
                file_size=0,  # Will be set by caller if needed
                quality_score=quality_score,
                issues=issues,
                timestamp=format_datetime(get_current_datetime())
            )
            
        except Exception as e:
            self.logger.exception(f"Error analyzing data quality: {e}")
            raise
    
    def _calculate_quality_score(
        self,
        total_rows: int,
        null_counts: Dict[str, int],
        duplicate_count: int,
        data_types: Dict[str, str]
    ) -> float:
        """Calculate overall data quality score."""
        if total_rows == 0:
            return 0.0
        
        # Start with perfect score
        score = 1.0
        
        # Penalize for null values
        total_nulls = sum(null_counts.values())
        null_penalty = (total_nulls / (total_rows * len(null_counts))) * 0.3
        score -= null_penalty
        
        # Penalize for duplicates
        duplicate_penalty = (duplicate_count / total_rows) * 0.2
        score -= duplicate_penalty
        
        # Penalize for wrong data types
        type_penalty = 0.0
        for col, dtype in data_types.items():
            if 'object' in dtype and col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                type_penalty += 0.1
        
        score -= min(type_penalty, 0.3)
        
        return max(0.0, min(1.0, score))
    
    def _identify_data_issues(self, df: pd.DataFrame, data_type: str) -> List[str]:
        """Identify specific data issues."""
        issues = []
        
        # Check for empty DataFrame
        if len(df) == 0:
            issues.append("Empty DataFrame")
            return issues
        
        # Check for all-null columns
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            issues.append(f"All-null columns: {all_null_cols}")
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")
        
        # Data type specific checks
        if data_type == "klines":
            issues.extend(self._check_klines_issues(df))
        elif data_type == "aggtrades":
            issues.extend(self._check_aggtrades_issues(df))
        
        return issues
    
    def _check_klines_issues(self, df: pd.DataFrame) -> List[str]:
        """Check for klines-specific issues."""
        issues = []
        
        # Check for missing OHLC columns
        ohlc_cols = ['open', 'high', 'low', 'close']
        missing_ohlc = set(ohlc_cols) - set(df.columns)
        if missing_ohlc:
            issues.append(f"Missing OHLC columns: {missing_ohlc}")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                negative_prices = (df[col] <= 0).sum()
                if negative_prices > 0:
                    issues.append(f"Negative prices in {col}: {negative_prices}")
        
        # Check for negative volume
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Negative volume: {negative_volume}")
        
        return issues
    
    def _check_aggtrades_issues(self, df: pd.DataFrame) -> List[str]:
        """Check for aggtrades-specific issues."""
        issues = []
        
        # Check for missing required columns
        required_cols = ['price', 'quantity', 'trade_count']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for negative prices
        if 'price' in df.columns:
            negative_prices = (df['price'] <= 0).sum()
            if negative_prices > 0:
                issues.append(f"Negative prices: {negative_prices}")
        
        # Check for negative quantities
        if 'quantity' in df.columns:
            negative_quantities = (df['quantity'] < 0).sum()
            if negative_quantities > 0:
                issues.append(f"Negative quantities: {negative_quantities}")
        
        return issues


class DataAccessManager:
    """Utility class for managing data access and security."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.access_log: List[Dict[str, Any]] = []
    
    def check_data_access_permissions(
        self,
        user_id: str,
        data_type: str,
        symbol: str,
        exchange: str
    ) -> bool:
        """Check if user has permission to access specific data."""
        try:
            # Get user permissions from config
            user_permissions = self.config.get('user_permissions', {})
            user_config = user_permissions.get(user_id, {})
            
            # Check if user has access to this data type
            allowed_data_types = user_config.get('data_types', [])
            if data_type not in allowed_data_types:
                self.logger.warning(f"User {user_id} denied access to {data_type} data")
                return False
            
            # Check if user has access to this symbol
            allowed_symbols = user_config.get('symbols', [])
            if allowed_symbols and symbol not in allowed_symbols:
                self.logger.warning(f"User {user_id} denied access to {symbol}")
                return False
            
            # Check if user has access to this exchange
            allowed_exchanges = user_config.get('exchanges', [])
            if allowed_exchanges and exchange not in allowed_exchanges:
                self.logger.warning(f"User {user_id} denied access to {exchange}")
                return False
            
            # Log successful access
            self._log_data_access(user_id, data_type, symbol, exchange, True)
            return True
            
        except Exception as e:
            self.logger.exception(f"Error checking data access permissions: {e}")
            return False
    
    def _log_data_access(
        self,
        user_id: str,
        data_type: str,
        symbol: str,
        exchange: str,
        granted: bool
    ) -> None:
        """Log data access attempt."""
        log_entry = {
            "timestamp": format_datetime(get_current_datetime()),
            "user_id": user_id,
            "data_type": data_type,
            "symbol": symbol,
            "exchange": exchange,
            "granted": granted
        }
        
        self.access_log.append(log_entry)
        
        level = logging.INFO if granted else logging.WARNING
        self.logger.log(
            level,
            f"Data access: {user_id} | {data_type} | {symbol} | {exchange} | {'GRANTED' if granted else 'DENIED'}"
        )


class DataStorageManager:
    """Utility class for managing data storage operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def save_data(
        self,
        data: Any,
        file_path: Union[str, Path],
        format: DataFormat = DataFormat.PARQUET,
        compression: CompressionType = CompressionType.NONE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataOperationResult:
        """Save data to file with specified format and compression."""
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            ensure_directory(file_path.parent)
            
            self.logger.info(f"Saving data to {file_path} in {format.value} format")
            
            # Save based on format
            if format == DataFormat.PARQUET:
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(file_path, compression=compression.value if compression != CompressionType.NONE else None)
                else:
                    raise ValueError("Parquet format only supports pandas DataFrames")
            
            elif format == DataFormat.CSV:
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, index=False, compression=compression.value if compression != CompressionType.NONE else None)
                else:
                    raise ValueError("CSV format only supports pandas DataFrames")
            
            elif format == DataFormat.JSON:
                if isinstance(data, (dict, list)):
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    raise ValueError("JSON format only supports dict or list")
            
            elif format == DataFormat.PICKLE:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Save metadata if provided
            if metadata:
                metadata_path = file_path.with_suffix('.metadata.json')
                safe_json_dump(metadata, metadata_path)
            
            execution_time = time.time() - start_time
            
            return DataOperationResult(
                success=True,
                message=f"Data saved successfully to {file_path}",
                metadata={
                    "file_path": str(file_path),
                    "format": format.value,
                    "compression": compression.value,
                    "file_size": file_path.stat().st_size,
                    "metadata_saved": metadata is not None
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.exception(f"Error saving data: {e}")
            return DataOperationResult(
                success=False,
                message=f"Error saving data: {e}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def load_data(
        self,
        file_path: Union[str, Path],
        format: Optional[DataFormat] = None
    ) -> DataOperationResult:
        """Load data from file."""
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return DataOperationResult(
                    success=False,
                    message=f"File not found: {file_path}",
                    execution_time=time.time() - start_time,
                    errors=[f"File not found: {file_path}"]
                )
            
            # Auto-detect format if not specified
            if format is None:
                format = self._detect_format(file_path)
            
            self.logger.info(f"Loading data from {file_path} in {format.value} format")
            
            # Load based on format
            if format == DataFormat.PARQUET:
                data = pd.read_parquet(file_path)
            elif format == DataFormat.CSV:
                data = pd.read_csv(file_path)
            elif format == DataFormat.JSON:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif format == DataFormat.PICKLE:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Load metadata if available
            metadata = None
            metadata_path = file_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                metadata = safe_json_load(metadata_path)
            
            execution_time = time.time() - start_time
            
            return DataOperationResult(
                success=True,
                message=f"Data loaded successfully from {file_path}",
                data=data,
                metadata=metadata,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.exception(f"Error loading data: {e}")
            return DataOperationResult(
                success=False,
                message=f"Error loading data: {e}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _detect_format(self, file_path: Path) -> DataFormat:
        """Auto-detect file format from extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.parquet':
            return DataFormat.PARQUET
        elif suffix == '.csv':
            return DataFormat.CSV
        elif suffix == '.json':
            return DataFormat.JSON
        elif suffix in ['.pkl', '.pickle']:
            return DataFormat.PICKLE
        else:
            # Default to parquet for unknown extensions
            return DataFormat.PARQUET


class ErrorHandler:
    """Utility class for comprehensive error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_log: List[Dict[str, Any]] = []
    
    def handle_operation_error(
        self,
        operation: str,
        error: Exception,
        context: Dict[str, Any],
        retry_count: int = 0
    ) -> DataOperationResult:
        """Handle operation errors with comprehensive logging and recovery."""
        try:
            error_info = {
                "timestamp": format_datetime(get_current_datetime()),
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "retry_count": retry_count
            }
            
            self.error_log.append(error_info)
            
            # Log error
            self.logger.error(
                f"Operation error: {operation} | {type(error).__name__}: {error} | "
                f"Context: {context} | Retry: {retry_count}"
            )
            
            # Determine if error is recoverable
            recoverable = self._is_recoverable_error(error)
            
            # Create error result
            result = DataOperationResult(
                success=False,
                message=f"Operation failed: {operation}",
                execution_time=0.0,
                errors=[f"{type(error).__name__}: {error}"]
            )
            
            # Add recovery suggestions
            if recoverable:
                result.warnings.append("Error may be recoverable with retry")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Error in error handler: {e}")
            return DataOperationResult(
                success=False,
                message="Error handler failed",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable."""
        recoverable_errors = [
            "ConnectionError",
            "TimeoutError",
            "TemporaryFailure",
            "RateLimitError"
        ]
        
        return type(error).__name__ in recoverable_errors
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        if not self.error_log:
            return {"message": "No errors logged"}
        
        error_types = {}
        for error in self.error_log:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "recent_errors": self.error_log[-10:]  # Last 10 errors
        }


# Export main classes
__all__ = [
    'DataFormat',
    'CompressionType',
    'DataOperationResult',
    'DataQualityMetrics',
    'DataFormatter',
    'DataAnalyzer',
    'DataAccessManager',
    'DataStorageManager',
    'ErrorHandler'
]