"""
Kline Parquet Utilities

This module provides utilities for loading, saving, and processing kline data
in Parquet format with enhanced error handling and M1 optimization.

Key Features:
- Efficient kline data loading and saving
- Data validation and quality checks
- M1 hardware optimization
- Memory-efficient operations
- Error handling and recovery
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import logging

# Import utilities with error handling
try:
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_info
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, ensure_directory,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        safe_dataframe_operation, validate_dataframe_columns, optimize_dataframe_dtypes
    )
    from src.utils.math_validation import (
        validate_finite, validate_positive, safe_divide, safe_log, safe_sqrt,
        safe_correlation, safe_covariance, safe_mean, safe_std
    )
    from src.utils.serialization_utils import UniversalSerializer
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Some utilities not available: {e}")
    UTILS_AVAILABLE = False
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_info

logger = logging.getLogger(__name__)

class KlineParquetManager:
    """
    Manager for kline data operations with Parquet format.
    
    Provides efficient loading, saving, and processing of kline data
    with enhanced error handling and M1 optimization.
    """
    
    def __init__(self):
        """Initialize the kline parquet manager."""
        try:
            tprint("🚀 [INIT] Starting KlineParquetManager initialization...")
            self.utils_available = UTILS_AVAILABLE
            self.serializer = None
            self.m1_optimizers = None
            
            if self.utils_available:
                try:
                    self.serializer = UniversalSerializer()
                    self.m1_optimizers = integrate_with_m1_optimizers()
                    tprint_success("✅ [INIT] Utility systems initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ [INIT] Utility systems initialization failed: {e}")
                    self.utils_available = False
            
            tprint_success("✅ [INIT] KlineParquetManager initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ [INIT] KlineParquetManager initialization failed: {e}")
            raise
    
    def load_kline_data(
        self, 
        file_path: Union[str, Path], 
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load kline data from Parquet file with validation.
        
        Args:
            file_path: Path to the Parquet file
            validate: Whether to validate the data
            
        Returns:
            DataFrame with kline data
        """
        try:
            tprint(f"📊 [LOAD_KLINE] Loading kline data from: {file_path}")
            
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Kline file not found: {file_path}")
            
            # Load data
            if self.utils_available and self.serializer:
                try:
                    data = self.serializer.load(str(file_path), format='parquet')
                    tprint_success("✅ [LOAD_KLINE] Data loaded with serializer")
                except Exception as e:
                    tprint_warning(f"⚠️ [LOAD_KLINE] Serializer failed, falling back to pandas: {e}")
                    data = pd.read_parquet(file_path)
            else:
                data = pd.read_parquet(file_path)
            
            # Validate data if requested
            if validate:
                self._validate_kline_data(data)
                tprint_success("✅ [LOAD_KLINE] Data validation passed")
            
            tprint_success(f"✅ [LOAD_KLINE] Successfully loaded {len(data)} kline records")
            return data
            
        except Exception as e:
            tprint_error(f"❌ [LOAD_KLINE] Failed to load kline data: {e}")
            raise
    
    def save_kline_data(
        self, 
        data: pd.DataFrame, 
        file_path: Union[str, Path],
        optimize: bool = True
    ) -> bool:
        """
        Save kline data to Parquet file with optimization.
        
        Args:
            data: DataFrame with kline data
            file_path: Path to save the file
            optimize: Whether to optimize the data before saving
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tprint(f"💾 [SAVE_KLINE] Saving kline data to: {file_path}")
            
            # Ensure directory exists
            if self.utils_available:
                try:
                    ensure_directory(Path(file_path).parent)
                except Exception as e:
                    tprint_warning(f"⚠️ [SAVE_KLINE] Directory creation failed: {e}")
            
            # Optimize data if requested
            if optimize and self.utils_available:
                try:
                    data = optimize_dataframe_dtypes(data)
                    tprint_success("✅ [SAVE_KLINE] Data optimized successfully")
                except Exception as e:
                    tprint_warning(f"⚠️ [SAVE_KLINE] Data optimization failed: {e}")
            
            # Save data
            if self.utils_available and self.serializer:
                try:
                    success = self.serializer.save(data, str(file_path), format='parquet')
                    if success:
                        tprint_success("✅ [SAVE_KLINE] Data saved with serializer")
                        return True
                    else:
                        raise Exception("Serialization failed")
                except Exception as e:
                    tprint_warning(f"⚠️ [SAVE_KLINE] Serializer failed, falling back to pandas: {e}")
                    data.to_parquet(file_path, index=False)
            else:
                data.to_parquet(file_path, index=False)
            
            tprint_success(f"✅ [SAVE_KLINE] Successfully saved {len(data)} kline records")
            return True
            
        except Exception as e:
            tprint_error(f"❌ [SAVE_KLINE] Failed to save kline data: {e}")
            return False
    
    def _validate_kline_data(self, data: pd.DataFrame) -> None:
        """
        Validate kline data for required columns and data quality.
        
        Args:
            data: DataFrame to validate
        """
        try:
            tprint("🔍 [VALIDATE_KLINE] Validating kline data...")
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(f"Column {col} must be numeric")
            
            # Check for negative values in price columns
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if (data[col] <= 0).any():
                    raise ValueError(f"Column {col} contains non-positive values")
            
            # Check for negative volume
            if (data['volume'] < 0).any():
                raise ValueError("Volume contains negative values")
            
            # Check OHLC relationships
            if not (data['high'] >= data['low']).all():
                raise ValueError("High prices must be >= low prices")
            
            if not (data['high'] >= data['open']).all():
                raise ValueError("High prices must be >= open prices")
            
            if not (data['high'] >= data['close']).all():
                raise ValueError("High prices must be >= close prices")
            
            if not (data['low'] <= data['open']).all():
                raise ValueError("Low prices must be <= open prices")
            
            if not (data['low'] <= data['close']).all():
                raise ValueError("Low prices must be <= close prices")
            
            tprint_success("✅ [VALIDATE_KLINE] Kline data validation passed")
            
        except Exception as e:
            tprint_error(f"❌ [VALIDATE_KLINE] Kline data validation failed: {e}")
            raise
    
    def process_kline_data(
        self, 
        data: pd.DataFrame,
        operations: List[str] = None
    ) -> pd.DataFrame:
        """
        Process kline data with various operations.
        
        Args:
            data: DataFrame with kline data
            operations: List of operations to perform
            
        Returns:
            Processed DataFrame
        """
        try:
            tprint("⚙️ [PROCESS_KLINE] Processing kline data...")
            
            if operations is None:
                operations = ['add_returns', 'add_volatility', 'add_technical_indicators']
            
            processed_data = data.copy()
            
            for operation in operations:
                try:
                    if operation == 'add_returns':
                        processed_data = self._add_returns(processed_data)
                    elif operation == 'add_volatility':
                        processed_data = self._add_volatility(processed_data)
                    elif operation == 'add_technical_indicators':
                        processed_data = self._add_technical_indicators(processed_data)
                    else:
                        tprint_warning(f"⚠️ [PROCESS_KLINE] Unknown operation: {operation}")
                except Exception as e:
                    tprint_warning(f"⚠️ [PROCESS_KLINE] Operation {operation} failed: {e}")
                    continue
            
            tprint_success(f"✅ [PROCESS_KLINE] Successfully processed kline data with {len(operations)} operations")
            return processed_data
            
        except Exception as e:
            tprint_error(f"❌ [PROCESS_KLINE] Kline data processing failed: {e}")
            raise
    
    def _add_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add return calculations to the data."""
        try:
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            return data
        except Exception as e:
            tprint_warning(f"⚠️ [ADD_RETURNS] Failed to add returns: {e}")
            return data
    
    def _add_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add volatility calculations to the data."""
        try:
            data['volatility'] = data['returns'].rolling(window=window).std()
            data['realized_volatility'] = data['returns'].rolling(window=window).std() * np.sqrt(252)
            return data
        except Exception as e:
            tprint_warning(f"⚠️ [ADD_VOLATILITY] Failed to add volatility: {e}")
            return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        try:
            # Simple Moving Averages
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            return data
        except Exception as e:
            tprint_warning(f"⚠️ [ADD_TECHNICAL_INDICATORS] Failed to add technical indicators: {e}")
            return data

# Convenience functions
def load_kline_data(file_path: Union[str, Path], validate: bool = True) -> pd.DataFrame:
    """Convenience function to load kline data."""
    manager = KlineParquetManager()
    return manager.load_kline_data(file_path, validate)

def save_kline_data(data: pd.DataFrame, file_path: Union[str, Path], optimize: bool = True) -> bool:
    """Convenience function to save kline data."""
    manager = KlineParquetManager()
    return manager.save_kline_data(data, file_path, optimize)

def process_kline_data(data: pd.DataFrame, operations: List[str] = None) -> pd.DataFrame:
    """Convenience function to process kline data."""
    manager = KlineParquetManager()
    return manager.process_kline_data(data, operations)