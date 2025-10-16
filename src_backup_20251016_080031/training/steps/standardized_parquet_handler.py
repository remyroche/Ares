"""
Standardized Parquet File Handler for Training Pipeline

This module provides a centralized, consistent interface for all Parquet file operations
across the training pipeline. It ensures:
- Consistent file paths and naming conventions
- Standardized column names and data types
- Unified schema validation and enforcement
- Proper error handling and logging
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.logger import system_logger
import time


class StandardizedParquetHandler:
    """Centralized handler for all Parquet file operations in the training pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the standardized Parquet handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = system_logger.getChild('StandardizedParquetHandler')
        self.standards = pipeline_standards
        
        # Standardized column mappings
        self.column_mappings = {
            'timestamp': 'timestamp',
            'time': 'timestamp',  # Map 'time' to 'timestamp'
            'datetime': 'timestamp',  # Map 'datetime' to 'timestamp'
            'ts': 'timestamp',  # Map 'ts' to 'timestamp'
            'price': 'close',  # Map generic 'price' to 'close'
            'amount': 'volume',  # Map 'amount' to 'volume'
            'vol': 'volume',  # Map 'vol' to 'volume'
            'qty': 'quantity',  # Map 'qty' to 'quantity'
        }
        
        # Standardized data types
        self.standard_dtypes = {
            'timestamp': 'int64',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64',
            'exchange': 'string',
            'symbol': 'string',
            'timeframe': 'string',
            'year': 'int32',
            'month': 'int8',
            'day': 'int8',
            'trade_volume': 'float64',
            'trade_count': 'int64',
            'avg_price': 'float64',
            'min_price': 'float64',
            'max_price': 'float64',
            'volume_ratio': 'float64',
            'is_buyer_maker': 'bool',
            'first_trade_id': 'int64',
            'last_trade_id': 'int64',
            'trade_time': 'int64',
            'quote_asset_volume': 'float64',
            'number_of_trades': 'int64',
            'taker_buy_base_asset_volume': 'float64',
            'taker_buy_quote_asset_volume': 'float64',
        }
        
        self.logger.debug('✅ StandardizedParquetHandler initialized')
    
    def add_partition_columns(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add partition columns (year, month, day) to DataFrame based on timestamp.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with added partition columns
        """
        if 'timestamp' not in df.columns:
            self.logger.warning('No timestamp column found, cannot add partition columns')
            return df
        
        df_copy = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if df_copy['timestamp'].dtype == 'int64':
            # Assume timestamp is in milliseconds
            df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='ms')
        else:
            df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'])
        
        # Add partition columns
        df_copy['year'] = df_copy['datetime'].dt.year.astype('int16')
        df_copy['month'] = df_copy['datetime'].dt.month.astype('int8')
        df_copy['day'] = df_copy['datetime'].dt.day.astype('int8')
        
        # Remove temporary datetime column
        df_copy = df_copy.drop('datetime', axis=1)
        
        self.logger.debug(f'✅ Added partition columns: year, month, day')
        return df_copy
    
    def write_partitioned_parquet(
        self, 
        df: 'pd.DataFrame', 
        base_path: str, 
        schema_name: str = 'unified',
        partition_cols: List[str] = None,
        **kwargs
    ) -> bool:
        """Write DataFrame as partitioned Parquet dataset.
        
        Args:
            df: DataFrame to write
            base_path: Base directory path for the dataset
            schema_name: Schema to enforce
            partition_cols: Columns to partition by (default: ['year', 'month', 'day'])
            **kwargs: Additional arguments for write_parquet_standardized
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add partition columns if not present
            if partition_cols is None:
                partition_cols = ['year', 'month', 'day']
            
            # Check if partition columns exist, add them if not
            missing_cols = [col for col in partition_cols if col not in df.columns]
            if missing_cols:
                self.logger.debug(f'Adding missing partition columns: {missing_cols}')
                df = self.add_partition_columns(df)
            
            # Ensure directory exists
            Path(base_path).mkdir(parents=True, exist_ok=True)
            
            # Use PyArrow for partitioned writing if available
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                import pyarrow.dataset as ds
                
                # Convert to PyArrow table
                table = pa.Table.from_pandas(df)
                
                # Create partitioning schema
                partition_fields = []
                for col in partition_cols:
                    if col in df.columns:
                        if col in ['year', 'month', 'day']:
                            partition_fields.append(pa.field(col, pa.int32() if col == 'year' else pa.int8()))
                        else:
                            partition_fields.append(pa.field(col, pa.string()))
                
                if partition_fields:
                    partition_schema = pa.schema(partition_fields)
                    partitioning = ds.partitioning(partition_schema, flavor='hive')
                    
                    # Write partitioned dataset
                    ds.write_dataset(
                        table,
                        base_path,
                        format='parquet',
                        partitioning=partitioning,
                        basename_template='part-{i}.parquet',
                        existing_data_behavior='overwrite_or_ignore'
                    )
                    
                    self.logger.debug(f'✅ Wrote partitioned dataset to {base_path}')
                    return True
                else:
                    # Fallback to single file
                    self.logger.warning('No valid partition columns, writing as single file')
                    return self.write_parquet_standardized(df, base_path, schema_name, **kwargs)
                    
            except ImportError:
                self.logger.warning('PyArrow not available, falling back to single file writing')
                return self.write_parquet_standardized(df, base_path, schema_name, **kwargs)
                
        except Exception as e:
            self.logger.error(f'❌ Failed to write partitioned parquet: {e}')
            return False
    
    def read_partitioned_parquet(
        self, 
        base_path: str, 
        schema_name: str = 'unified',
        filters: List[Tuple] = None,
        columns: List[str] = None,
        **kwargs
    ) -> Optional['pd.DataFrame']:
        """Read partitioned Parquet dataset.
        
        Args:
            base_path: Base directory path of the dataset
            schema_name: Schema to validate against
            filters: PyArrow filters for reading specific partitions
            columns: Specific columns to read
            **kwargs: Additional arguments
            
        Returns:
            DataFrame or None if failed
        """
        try:
            # Use PyArrow for partitioned reading if available
            try:
                import pyarrow.dataset as ds
                
                # Read partitioned dataset
                dataset = ds.dataset(base_path, format='parquet')
                
                # Apply filters if provided
                if filters:
                    dataset = dataset.filter(filters)
                
                # Convert to pandas
                df = dataset.to_table(columns=columns).to_pandas()
                
                if df.empty:
                    self.logger.warning(f'No data found in partitioned dataset at {base_path}')
                    return None
                
                # Validate and standardize
                df = self.standardize_dtypes(df, schema_name)
                
                self.logger.debug(f'✅ Read partitioned dataset from {base_path}: {len(df)} rows')
                return df
                
            except ImportError:
                self.logger.warning('PyArrow not available, falling back to single file reading')
                # Fallback: try to read as single file
                return self.read_parquet_standardized(base_path, schema_name, **kwargs)
                
        except Exception as e:
            self.logger.error(f'❌ Failed to read partitioned parquet: {e}')
            return None
    
    def get_partitioned_path(
        self, 
        path_type: str, 
        exchange: str, 
        symbol: str, 
        timeframe: str = '1m',
        **kwargs
    ) -> str:
        """Get standardized path for partitioned data.
        
        Args:
            path_type: Type of path (unified_partitioned, etc.)
            exchange: Exchange name
            symbol: Asset symbol
            timeframe: Timeframe
            **kwargs: Additional path parameters
            
        Returns:
            Standardized partitioned path string
        """
        if path_type == 'unified_partitioned':
            # Use the partitioned path structure
            base_path = self.standards.build_path('unified_data', exchange, symbol, timeframe)
            return f"{base_path}/partitioned"
        else:
            # Fallback to regular path
            return self.get_standardized_path(path_type, exchange, symbol, timeframe, **kwargs)
    
    def get_standardized_path(
        self,
        path_type: str,
        exchange: str,
        symbol: str,
        timeframe: str = '1m',
        **kwargs
    ) -> str:
        """Get standardized file path using pipeline standards.

        Args:
            path_type: Type of path (raw_data, unified_data, processed_data, etc.)
            exchange: Exchange name
            symbol: Asset symbol
            timeframe: Timeframe
            **kwargs: Additional path parameters

        Returns:
            Standardized path string
        """
        try:
            # Handle training-specific paths to use correct data directory
            if path_type in ['training', 'unified_data', 'processed_data', 'raw_data']:
                # Override training paths to use data/training instead of data_cache
                base_path = f"data/training/{exchange.lower()}/{symbol.lower()}"
                if path_type == 'unified_data':
                    base_path += f"/{timeframe}"
                elif path_type == 'processed_data':
                    base_path += "/processed"
                elif path_type == 'raw_data':
                    base_path += "/raw"
            else:
                # Use pipeline standards for other path types
                base_path = self.standards.build_path(path_type, exchange, symbol, timeframe=timeframe, **kwargs)

            # Ensure the path exists
            Path(base_path).mkdir(parents=True, exist_ok=True)

            return base_path

        except Exception as e:
            self.logger.error(f"Error building standardized path: {e}")
            # Fallback to training data directory structure
            fallback_path = f"data/training/{exchange.lower()}/{symbol.lower()}"
            if path_type == 'unified_data':
                fallback_path += f"/{timeframe}"
            Path(fallback_path).mkdir(parents=True, exist_ok=True)
            return fallback_path
    
    def get_standardized_filename(
        self, 
        file_type: str, 
        exchange: str, 
        symbol: str, 
        timeframe: str = '1m',
        **kwargs
    ) -> str:
        """Get standardized filename using pipeline standards.
        
        Args:
            file_type: Type of file (klines, aggtrades, unified, etc.)
            exchange: Exchange name
            symbol: Asset symbol
            timeframe: Timeframe
            **kwargs: Additional filename parameters
            
        Returns:
            Standardized filename
        """
        try:
            return self.standards.generate_file_name(file_type, exchange, symbol, timeframe, **kwargs)
        except Exception as e:
            self.logger.error(f"Error generating standardized filename: {e}")
            # Fallback filename pattern
            return f"{file_type}_{exchange}_{symbol}_{timeframe}.parquet"
    
    def standardize_columns(self, df: 'pd.DataFrame', schema_name: str = 'unified') -> 'pd.DataFrame':
        """Standardize column names in DataFrame.

        Args:
            df: DataFrame to standardize
            schema_name: Schema name to determine appropriate column mappings

        Returns:
            DataFrame with standardized column names
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        # Apply column mappings
        column_renames = {}
        for old_name, new_name in self.column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                column_renames[old_name] = new_name

        # Special handling for aggtrades: map 'close' back to 'price' if needed
        if schema_name == 'aggtrades':
            if 'close' in df.columns and 'price' not in df.columns:
                column_renames['close'] = 'price'
                self.logger.debug("🔄 Mapping 'close' to 'price' for aggtrades schema")

        if column_renames:
            df = df.rename(columns=column_renames)
            self.logger.debug(f"Renamed columns: {column_renames}")

        # Validate required columns for unified schema
        if schema_name == 'unified':
            required_columns = pipeline_standards.SCHEMAS['unified']['required_columns']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                self.logger.warning(f"⚠️ Missing required columns for unified schema: {missing_columns}")
                self.logger.debug(f"🔧 Attempting to add missing columns automatically...")

                # Handle missing columns for processed data files
                if 'timestamp' in missing_columns and 'open_time' in df.columns:
                    # Clean open_time values before conversion
                    open_time_clean = df['open_time'].copy()

                    # Remove NaN and infinite values
                    valid_mask = pd.notna(open_time_clean) & np.isfinite(open_time_clean)
                    if not valid_mask.all():
                        invalid_count = len(open_time_clean) - valid_mask.sum()
                        self.logger.warning(f"⚠️ Found {invalid_count} invalid open_time values, cleaning them")
                        open_time_clean = open_time_clean[valid_mask]

                    # Convert open_time from milliseconds to datetime
                    df['timestamp'] = pd.to_datetime(open_time_clean, unit='ms')
                    missing_columns.remove('timestamp')
                    self.logger.debug(f"✅ Added 'timestamp' column from 'open_time'")

                if 'exchange' in missing_columns and 'symbol' in df.columns:
                    # Extract exchange from symbol (e.g., 'ETHUSDT' -> 'binance')
                    # This is a heuristic - in practice, exchange should be passed as a parameter
                    df['exchange'] = 'binance'  # Default to binance for now
                    missing_columns.remove('exchange')
                    self.logger.debug(f"✅ Added 'exchange' column (default: binance)")

                if 'timeframe' in missing_columns and 'interval' in df.columns:
                    # Convert interval to timeframe format (e.g., '15m' -> '15m')
                    df['timeframe'] = df['interval']
                    missing_columns.remove('timeframe')
                    self.logger.debug(f"✅ Added 'timeframe' column from 'interval'")

                # Check if we still have missing columns
                if missing_columns:
                    error_msg = f"Could not automatically resolve missing columns: {missing_columns}"
                    self.logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)
                else:
                    self.logger.debug(f"✅ Successfully added all missing columns automatically")

        return df
    
    def standardize_dtypes(self, df: 'pd.DataFrame', schema_name: str = 'unified') -> 'pd.DataFrame':
        """Standardize data types in DataFrame.
        
        Args:
            df: DataFrame to standardize
            schema_name: Schema name for type enforcement
            
        Returns:
            DataFrame with standardized data types
        """
        if df is None or df.empty:
            return df
            
        try:
            # Use pipeline standards for schema enforcement
            df = self.standards.enforce_schema(df, schema_name)
            self.logger.debug(f"Applied schema enforcement for {schema_name}")
            return df
            
        except Exception as e:
            self.logger.warning(f"Schema enforcement failed: {e}, using fallback type conversion")
            
            # Fallback: manual type conversion
            df = df.copy()
            for column, dtype in self.standard_dtypes.items():
                if column in df.columns:
                    try:
                        if dtype == 'int64':
                            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype('int64')
                        elif dtype == 'float64':
                            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0).astype('float64')
                        elif dtype == 'string':
                            df[column] = df[column].astype('string')
                        elif dtype == 'bool':
                            df[column] = df[column].astype('boolean')
                        elif dtype in ['int16', 'int8']:
                            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(dtype)
                    except Exception as col_error:
                        self.logger.warning(f"Failed to convert column {column} to {dtype}: {col_error}")
            
            return df
    
    def standardize_timestamp(self, df: 'pd.DataFrame', column: str = 'timestamp') -> 'pd.DataFrame':
        """Standardize timestamp column.
        
        Args:
            df: DataFrame to process
            column: Timestamp column name
            
        Returns:
            DataFrame with standardized timestamp
        """
        if df is None or df.empty or column not in df.columns:
            return df
            
        try:
            # Use pipeline standards for timestamp standardization
            df = self.standards.standardize_timestamp(df, column, 'int64')
            self.logger.debug(f"Standardized timestamp column: {column}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error standardizing timestamp: {e}")
            return df
    
    def validate_data_quality(self, df: 'pd.DataFrame', schema_name: str = 'unified') -> Dict[str, Any]:
        """Validate data quality using pipeline standards.

        Args:
            df: DataFrame to validate
            schema_name: Schema name for validation

        Returns:
            Validation results dictionary
        """
        try:
            # Standardize columns before validation
            df = self.standardize_columns(df, schema_name)
            validation_result = self.standards.validate_data_quality(df, schema_name)
            
            return {
                'passed': validation_result.passed,
                'quality_score': validation_result.quality_score,
                'issues': [issue.message for issue in validation_result.issues],
                'warnings': [warning.message for warning in validation_result.warnings],
                'metadata': validation_result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error validating data quality: {e}")
            return {
                'passed': False,
                'quality_score': 0.0,
                'issues': [f'Validation error: {str(e)}'],
                'warnings': [],
                'metadata': {}
            }
    
    def read_parquet(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Optional['pd.DataFrame']:
        """Convenience method for read_parquet_standardized.

        This method provides backward compatibility for any code that calls read_parquet.
        """
        return self.read_parquet_standardized(file_path, **kwargs)

    def read_parquet_standardized(
        self,
        file_path: Union[str, Path],
        schema_name: str = 'unified',
        validate_quality: bool = True
    ) -> Optional['pd.DataFrame']:
        """Read Parquet file with standardized processing.
        
        Args:
            file_path: Path to Parquet file
            schema_name: Schema name for validation
            validate_quality: Whether to validate data quality
            
        Returns:
            Standardized DataFrame or None if failed
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return None
            
            # Read the file
            df = pd.read_parquet(file_path)
            
            if df is None or df.empty:
                self.logger.warning(f"File is empty: {file_path}")
                return None
            
            # Standardize the data
            df = self.standardize_columns(df, schema_name)
            df = self.standardize_timestamp(df)
            df = self.standardize_dtypes(df, schema_name)
            
            # Validate quality if requested
            if validate_quality:
                validation_result = self.validate_data_quality(df, schema_name)
                if not validation_result['passed']:
                    self.logger.warning(f"Data quality issues in {file_path}: {validation_result['issues']}")
                else:
                    self.logger.debug(f"Data quality validation passed for {file_path} (score: {validation_result['quality_score']:.2f})")
            
            self.logger.debug(f"Successfully read and standardized {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading Parquet file {file_path}: {e}")
            return None
    
    def write_parquet(
        self,
        df: 'pd.DataFrame',
        file_path: Union[str, Path],
        **kwargs
    ) -> bool:
        """Convenience method for write_parquet_standardized.

        This method provides backward compatibility for any code that calls write_parquet.
        """
        return self.write_parquet_standardized(df, file_path, **kwargs)

    def write_parquet_standardized(
        self,
        df: 'pd.DataFrame',
        file_path: Union[str, Path],
        schema_name: str = 'unified',
        validate_quality: bool = True,
        create_metadata: bool = True,
        index: bool = None,
        **kwargs
    ) -> bool:
        """Write DataFrame to Parquet file with standardized processing.
        
        Args:
            df: DataFrame to write
            file_path: Output file path
            schema_name: Schema name for validation
            validate_quality: Whether to validate data quality before writing
            create_metadata: Whether to create metadata file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if df is None or df.empty:
                self.logger.error("Cannot write empty DataFrame")
                return False
            
            file_path = Path(file_path)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Standardize the data before writing
            df = self.standardize_columns(df, schema_name)
            df = self.standardize_timestamp(df)
            df = self.standardize_dtypes(df, schema_name)
            
            # Validate quality if requested
            if validate_quality:
                validation_result = self.validate_data_quality(df, schema_name)
                if not validation_result['passed']:
                    self.logger.warning(f"Data quality issues before writing: {validation_result['issues']}")
                    # Continue anyway, but log the issues
            
            # Write the file - use index parameter if provided, otherwise default to False
            index_param = index if index is not None else False
            df.to_parquet(file_path, index=index_param, compression='snappy')
            
            # Create metadata file if requested
            if create_metadata:
                self._create_metadata_file(df, file_path, schema_name)
            
            self.logger.debug(f"Successfully wrote {len(df)} rows to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing Parquet file {file_path}: {e}")
            return False
    
    def _create_metadata_file(self, df: 'pd.DataFrame', file_path: Path, schema_name: str) -> None:
        """Create metadata file for the Parquet file.
        
        Args:
            df: DataFrame that was written
            file_path: Path to the Parquet file
            schema_name: Schema name used
        """
        try:
            metadata_path = file_path.with_suffix('.metadata.json')
            
            metadata = {
                'file_path': str(file_path),
                'schema_name': schema_name,
                'created_at': datetime.now().isoformat(),
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
                'pipeline_version': '1.0.0'
            }
            
            # Add timestamp range if available
            if 'timestamp' in df.columns:
                try:
                    timestamps = pd.to_datetime(df['timestamp'], unit='ms')
                    metadata['timestamp_range'] = {
                        'start': timestamps.min().isoformat(),
                        'end': timestamps.max().isoformat()
                    }
                except Exception:
                    pass
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.debug(f"Created metadata file: {metadata_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create metadata file: {e}")
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a Parquet file.
        
        Args:
            file_path: Path to Parquet file
            
        Returns:
            Dictionary with file information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {'exists': False, 'error': 'File does not exist'}
            
            # Get basic file info
            stat = file_path.stat()
            file_info = {
                'exists': True,
                'file_path': str(file_path),
                'file_size_bytes': stat.st_size,
                'file_size_mb': stat.st_size / (1024**2),
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
            # Try to get Parquet-specific info
            try:
                df_sample = pd.read_parquet(file_path, nrows=1)
                file_info.update({
                    'columns': list(df_sample.columns),
                    'column_count': len(df_sample.columns),
                    'dtypes': df_sample.dtypes.to_dict()
                })
                
                # Get full row count (this might be expensive for large files)
                if stat.st_size < 100 * 1024 * 1024:  # Only for files < 100MB
                    df_full = pd.read_parquet(file_path)
                    file_info['row_count'] = len(df_full)
                
            except Exception as e:
                file_info['parquet_info_error'] = str(e)
            
            # Check for metadata file
            metadata_path = file_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                try:
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    file_info['metadata'] = metadata
                except Exception as e:
                    file_info['metadata_error'] = str(e)
            
            return file_info
            
        except Exception as e:
            return {'exists': False, 'error': str(e)}
    
    def list_parquet_files(
        self, 
        directory: Union[str, Path], 
        pattern: str = "*.parquet",
        recursive: bool = True
    ) -> List[Path]:
        """List Parquet files in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of Parquet file paths
        """
        try:
            directory = Path(directory)
            
            if not directory.exists():
                self.logger.warning(f"Directory does not exist: {directory}")
                return []
            
            if recursive:
                parquet_files = list(directory.rglob(pattern))
            else:
                parquet_files = list(directory.glob(pattern))
            
            # Sort by modification time (newest first)
            parquet_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            self.logger.debug(f"Found {len(parquet_files)} Parquet files in {directory}")
            return parquet_files
            
        except Exception as e:
            self.logger.error(f"Error listing Parquet files in {directory}: {e}")
            return []


# Global instance for easy access
standardized_parquet_handler = StandardizedParquetHandler()


# Convenience functions for backward compatibility
def read_parquet(file_path: Union[str, Path], **kwargs) -> Optional['pd.DataFrame']:
    """Convenience function to read Parquet file."""
    return standardized_parquet_handler.read_parquet(file_path, **kwargs)


def read_parquet_standardized(file_path: Union[str, Path], **kwargs) -> Optional['pd.DataFrame']:
    """Convenience function to read Parquet file with standardization."""
    return standardized_parquet_handler.read_parquet_standardized(file_path, **kwargs)


def write_parquet(df: 'pd.DataFrame', file_path: Union[str, Path], **kwargs) -> bool:
    """Convenience function to write Parquet file."""
    return standardized_parquet_handler.write_parquet(df, file_path, **kwargs)


def write_parquet_standardized(df: 'pd.DataFrame', file_path: Union[str, Path], **kwargs) -> bool:
    """Convenience function to write Parquet file with standardization."""
    return standardized_parquet_handler.write_parquet_standardized(df, file_path, **kwargs)


def get_standardized_path(path_type: str, exchange: str, symbol: str, **kwargs) -> str:
    """Convenience function to get standardized path."""
    return standardized_parquet_handler.get_standardized_path(path_type, exchange, symbol, **kwargs)


def get_standardized_filename(file_type: str, exchange: str, symbol: str, **kwargs) -> str:
    """Convenience function to get standardized filename."""
    return standardized_parquet_handler.get_standardized_filename(file_type, exchange, symbol, **kwargs)