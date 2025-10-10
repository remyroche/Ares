"""
Centralized Exchange Data Standardizer

This module provides a centralized data standardization system for all exchange
pipelines, ensuring consistent data format across different exchanges.

Features:
- Unified data format conversion
- Exchange-agnostic standardization
- Built-in data quality validation
- Automatic data type optimization
- Comprehensive error handling
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.data.processing.data_processing import DataProcessor
from src.utils.data.quality.data_quality import DataQualityFramework


class ExchangeDataStandardizer:
    """Centralized data standardizer for all exchange pipelines."""
    
    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the exchange data standardizer.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.logger = system_logger.getChild("ExchangeDataStandardizer")
        self.data_processor = DataProcessor()
        self.quality_framework = DataQualityFramework()
        
        # Standardized data format specifications
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        self.metadata_columns = ['exchange', 'symbol', 'interval']
        self.optional_columns = ['open_time', 'close_time', 'quote_volume', 'trades', 
                                'taker_buy_base', 'taker_buy_quote', 'number_of_trades']
        
        # Exchange-specific configurations
        self.exchange_configs = {
            'binance': {
                'timestamp_column': 'open_time',
                'timestamp_unit': 'ms',
                'expected_columns': ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume']
            },
            'bingx': {
                'timestamp_column': 'open_time', 
                'timestamp_unit': 'ms',
                'expected_columns': ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume']
            },
            'mexc': {
                'timestamp_column': 'open_time',
                'timestamp_unit': 'ms', 
                'expected_columns': ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume']
            },
            'okx': {
                'timestamp_column': 'timestamp',
                'timestamp_unit': 'ms',
                'expected_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            },
            'gateio': {
                'timestamp_column': 'timestamp',
                'timestamp_unit': 's',
                'expected_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            }
        }
        
        self.logger.info("✅ ExchangeDataStandardizer initialized")
    
    def standardize_data(
        self, 
        df: pd.DataFrame, 
        exchange: str, 
        symbol: str, 
        interval: str,
        validate_quality: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Standardize exchange data to unified format.
        
        Args:
            df: Raw DataFrame from exchange
            exchange: Exchange name (binance, bingx, mexc, etc.)
            symbol: Trading symbol
            interval: Data interval
            validate_quality: Whether to validate data quality
            
        Returns:
            Tuple of (standardized_dataframe, standardization_report)
        """
        if df is None or df.empty:
            self.logger.warning("Empty DataFrame provided for standardization")
            return df, {'error': 'Empty DataFrame', 'success': False}
        
        start_time = datetime.now()
        report = {
            'exchange': exchange,
            'symbol': symbol,
            'interval': interval,
            'original_shape': df.shape,
            'original_columns': list(df.columns),
            'success': False,
            'errors': [],
            'warnings': [],
            'processing_time': 0.0
        }
        
        try:
            self.logger.info(f"🔄 Standardizing {exchange} data: {symbol} {interval} ({len(df)} records)")
            
            # Step 1: Create working copy
            standardized_df = df.copy()
            
            # Step 2: Handle timestamp column
            standardized_df = self._standardize_timestamp(standardized_df, exchange, report)
            
            # Step 3: Ensure required OHLCV columns
            standardized_df = self._standardize_ohlcv_columns(standardized_df, report)
            
            # Step 4: Set timestamp as index
            standardized_df = self._set_timestamp_index(standardized_df, report)
            
            # Step 5: Add exchange metadata
            standardized_df = self._add_exchange_metadata(standardized_df, exchange, symbol, interval)
            
            # Step 6: Apply data quality fixes
            if validate_quality:
                standardized_df = self._apply_quality_fixes(standardized_df, report)
            
            # Step 7: Optimize data types
            standardized_df = self._optimize_data_types(standardized_df, report)
            
            # Step 8: Final validation
            if validate_quality:
                self._validate_standardized_data(standardized_df, report)
            
            # Update report
            report['final_shape'] = standardized_df.shape
            report['final_columns'] = list(standardized_df.columns)
            report['success'] = True
            report['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ Standardization completed: {len(standardized_df)} records in {report['processing_time']:.2f}s")
            
            return standardized_df, report
            
        except Exception as e:
            error_msg = f"Standardization failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            report['errors'].append(error_msg)
            report['processing_time'] = (datetime.now() - start_time).total_seconds()
            return df, report
    
    def _standardize_timestamp(self, df: pd.DataFrame, exchange: str, report: Dict[str, Any]) -> pd.DataFrame:
        """Standardize timestamp column based on exchange configuration."""
        try:
            config = self.exchange_configs.get(exchange, {})
            timestamp_col = config.get('timestamp_column', 'timestamp')
            
            # Find the best timestamp column
            if timestamp_col in df.columns:
                df['timestamp'] = df[timestamp_col]
            elif 'open_time' in df.columns:
                df['timestamp'] = df['open_time']
                report['warnings'].append("Using 'open_time' as timestamp column")
            elif 'timestamp' in df.columns:
                # Already exists
                pass
            elif df.index.name == 'timestamp':
                df = df.reset_index()
            else:
                # Use index as timestamp
                df['timestamp'] = df.index
                report['warnings'].append("Using DataFrame index as timestamp")
            
            # Convert to datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                timestamp_unit = config.get('timestamp_unit', 'ms')
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit=timestamp_unit, utc=True)
                except (ValueError, TypeError):
                    try:
                        # Try alternative units
                        for unit in ['ms', 's', 'us']:
                            try:
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit, utc=True)
                                break
                            except (ValueError, TypeError):
                                continue
                        else:
                            # Direct conversion
                            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    except Exception as e:
                        raise ValueError(f"Could not convert timestamp column: {e}")
            
            return df
            
        except Exception as e:
            report['errors'].append(f"Timestamp standardization failed: {e}")
            return df
    
    def _standardize_ohlcv_columns(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Ensure all required OHLCV columns exist and are numeric."""
        try:
            # Check for missing required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                report['errors'].append(f"Missing required columns: {missing_columns}")
                return df
            
            # Convert OHLCV columns to numeric
            for col in self.required_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check for conversion issues
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        report['warnings'].append(f"Column '{col}' has {nan_count} NaN values after conversion")
            
            return df
            
        except Exception as e:
            report['errors'].append(f"OHLCV standardization failed: {e}")
            return df
    
    def _set_timestamp_index(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Set timestamp as DataFrame index."""
        try:
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
                report['warnings'].append("Set timestamp as DataFrame index")
            elif not isinstance(df.index, pd.DatetimeIndex):
                report['warnings'].append("No timestamp column found for indexing")
            
            return df
            
        except Exception as e:
            report['errors'].append(f"Index setting failed: {e}")
            return df
    
    def _add_exchange_metadata(self, df: pd.DataFrame, exchange: str, symbol: str, interval: str) -> pd.DataFrame:
        """Add exchange metadata columns."""
        try:
            df['exchange'] = exchange
            df['symbol'] = symbol
            df['interval'] = interval
            return df
            
        except Exception as e:
            self.logger.warning(f"Failed to add metadata: {e}")
            return df
    
    def _apply_quality_fixes(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Apply data quality fixes using DataProcessor."""
        try:
            fixed_df, fix_report = self.data_processor.fix_data_quality_issues(df)
            
            # Update report with fix details
            if 'issues_fixed' in fix_report:
                report['quality_fixes'] = fix_report['issues_fixed']
            
            return fixed_df
            
        except Exception as e:
            report['warnings'].append(f"Quality fixes failed: {e}")
            return df
    
    def _optimize_data_types(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        try:
            optimized_df = self.data_processor.optimize_dataframe_dtypes(df)
            
            # Log memory optimization
            original_memory = df.memory_usage(deep=True).sum()
            optimized_memory = optimized_df.memory_usage(deep=True).sum()
            memory_reduction = (original_memory - optimized_memory) / original_memory * 100
            
            report['memory_optimization'] = {
                'original_mb': original_memory / 1024 / 1024,
                'optimized_mb': optimized_memory / 1024 / 1024,
                'reduction_percent': memory_reduction
            }
            
            return optimized_df
            
        except Exception as e:
            report['warnings'].append(f"Data type optimization failed: {e}")
            return df
    
    def _validate_standardized_data(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate the standardized data using DataQualityFramework."""
        try:
            quality_result = self.quality_framework.validate_dataframe_quality(df, f"{report['exchange']} standardization")
            
            report['quality_validation'] = {
                'passed': quality_result.passed,
                'quality_score': quality_result.quality_score,
                'issues': quality_result.issues,
                'warnings': quality_result.warnings,
                'metrics': quality_result.metrics
            }
            
            if not quality_result.passed:
                report['warnings'].extend(quality_result.issues)
            
        except Exception as e:
            report['warnings'].append(f"Quality validation failed: {e}")
    
    def get_standardization_schema(self, exchange: str) -> Dict[str, Any]:
        """Get the expected schema for a specific exchange.
        
        Args:
            exchange: Exchange name
            
        Returns:
            Dictionary containing expected schema information
        """
        config = self.exchange_configs.get(exchange, {})
        
        return {
            'exchange': exchange,
            'required_columns': self.required_columns,
            'metadata_columns': self.metadata_columns,
            'optional_columns': self.optional_columns,
            'timestamp_column': config.get('timestamp_column', 'timestamp'),
            'timestamp_unit': config.get('timestamp_unit', 'ms'),
            'expected_columns': config.get('expected_columns', self.required_columns)
        }
    
    def validate_exchange_data(self, df: pd.DataFrame, exchange: str) -> Dict[str, Any]:
        """Validate that raw exchange data can be standardized.
        
        Args:
            df: Raw DataFrame from exchange
            exchange: Exchange name
            
        Returns:
            Validation results
        """
        config = self.exchange_configs.get(exchange, {})
        expected_columns = config.get('expected_columns', self.required_columns)
        
        validation_result = {
            'exchange': exchange,
            'can_standardize': True,
            'missing_columns': [],
            'extra_columns': [],
            'data_quality_issues': [],
            'recommendations': []
        }
        
        # Check for missing expected columns
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            validation_result['missing_columns'] = missing_columns
            validation_result['can_standardize'] = False
            validation_result['recommendations'].append(f"Add missing columns: {missing_columns}")
        
        # Check for extra columns
        extra_columns = [col for col in df.columns if col not in expected_columns + self.optional_columns]
        if extra_columns:
            validation_result['extra_columns'] = extra_columns
            validation_result['recommendations'].append(f"Consider handling extra columns: {extra_columns}")
        
        # Check data quality
        if df.empty:
            validation_result['data_quality_issues'].append("DataFrame is empty")
            validation_result['can_standardize'] = False
        
        # Check for required OHLCV columns
        missing_ohlcv = [col for col in self.required_columns if col not in df.columns]
        if missing_ohlcv:
            validation_result['data_quality_issues'].append(f"Missing OHLCV columns: {missing_ohlcv}")
            validation_result['can_standardize'] = False
        
        return validation_result


# Convenience functions for easy usage
def standardize_exchange_data(
    df: pd.DataFrame, 
    exchange: str, 
    symbol: str, 
    interval: str,
    data_dir: str = "historical_data",
    validate_quality: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function to standardize exchange data.
    
    Args:
        df: Raw DataFrame from exchange
        exchange: Exchange name
        symbol: Trading symbol
        interval: Data interval
        data_dir: Base directory for data storage
        validate_quality: Whether to validate data quality
        
    Returns:
        Tuple of (standardized_dataframe, standardization_report)
    """
    standardizer = ExchangeDataStandardizer(data_dir)
    return standardizer.standardize_data(df, exchange, symbol, interval, validate_quality)


def get_exchange_schema(exchange: str) -> Dict[str, Any]:
    """Get the expected schema for a specific exchange.
    
    Args:
        exchange: Exchange name
        
    Returns:
        Dictionary containing expected schema information
    """
    standardizer = ExchangeDataStandardizer()
    return standardizer.get_standardization_schema(exchange)


def validate_exchange_data(df: pd.DataFrame, exchange: str) -> Dict[str, Any]:
    """Validate that raw exchange data can be standardized.
    
    Args:
        df: Raw DataFrame from exchange
        exchange: Exchange name
        
    Returns:
        Validation results
    """
    standardizer = ExchangeDataStandardizer()
    return standardizer.validate_exchange_data(df, exchange)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create mock data
    mock_data = pd.DataFrame({
        'open_time': [int((datetime.now() - timedelta(minutes=i)).timestamp() * 1000) for i in range(10)],
        'close_time': [int((datetime.now() - timedelta(minutes=i-1)).timestamp() * 1000) for i in range(10)],
        'open': np.random.uniform(50000, 51000, 10),
        'high': np.random.uniform(51000, 52000, 10),
        'low': np.random.uniform(49000, 50000, 10),
        'close': np.random.uniform(50000, 51000, 10),
        'volume': np.random.uniform(100, 1000, 10)
    })
    
    # Test standardization
    standardizer = ExchangeDataStandardizer()
    standardized_df, report = standardizer.standardize_data(mock_data, "binance", "BTCUSDT", "1m")
    
    print(f"Standardization successful: {report['success']}")
    print(f"Final shape: {standardized_df.shape}")
    print(f"Columns: {list(standardized_df.columns)}")
    print(f"Index type: {type(standardized_df.index)}")