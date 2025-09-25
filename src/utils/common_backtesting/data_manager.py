"""
Unified Data Manager for Backtesting

This module provides unified data management functionality for backtesting
across TAS, NAS, and hybrid systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import common utilities
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
        calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
        safe_apply_function, create_summary_statistics, safe_drop_columns,
        safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
        get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
        optimize_dataframe_dtypes, safe_to_parquet, safe_read_parquet,
        align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, memory_checkpoint, gpu_context,
        optimize_memory, get_memory_usage, validate_file_path, get_file_size,
        check_disk_space, CommonUtilities
    )
    COMMON_UTILS_AVAILABLE = True
except ImportError:
    COMMON_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DataManagerConfig:
    """Configuration for backtesting data manager."""
    
    # Data source parameters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    frequency: str = "1H"  # 1H, 1D, etc.
    
    # Data quality parameters
    min_data_points: int = 1000
    max_data_points: int = 10000
    enable_validation: bool = True
    quality_threshold: float = 0.95
    fill_missing_data: bool = True
    
    # Data processing parameters
    enable_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_outlier_detection: bool = True
    outlier_threshold: float = 3.0
    
    # Storage parameters
    enable_caching: bool = True
    cache_directory: Optional[str] = None
    cache_format: str = "parquet"  # parquet, csv, pickle
    
    # Performance parameters
    enable_memory_optimization: bool = True
    chunk_size: Optional[int] = None
    parallel_processing: bool = True


class BacktestingDataManager:
    """
    Unified data manager for backtesting operations.
    
    Handles data loading, validation, preprocessing, and caching
    for all backtesting systems.
    """
    
    def __init__(self, config: DataManagerConfig):
        """Initialize the data manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache_directory = None
        
        # Initialize cache directory
        if self.config.enable_caching:
            self._initialize_cache()
        
        # Initialize memory optimization if available
        if self.config.enable_memory_optimization and COMMON_UTILS_AVAILABLE:
            self.memory_optimizer = get_m1_memory_optimizer()
        else:
            self.memory_optimizer = None
    
    def _initialize_cache(self):
        """Initialize cache directory."""
        if self.config.cache_directory:
            self.cache_directory = Path(self.config.cache_directory)
        else:
            self.cache_directory = Path("data_cache/backtesting")
        
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Cache directory initialized: {self.cache_directory}")
    
    def load_data(
        self,
        data_source: Optional[Union[str, pd.DataFrame]] = None,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data for backtesting.
        
        Args:
            data_source: Path to data file or DataFrame
            symbol: Trading symbol (for cache key generation)
            
        Returns:
            Processed DataFrame ready for backtesting
        """
        self.logger.info("Loading data for backtesting")
        
        # Check cache first
        if self.config.enable_caching and symbol:
            cached_data = self._load_from_cache(symbol)
            if cached_data is not None:
                self.logger.info("Data loaded from cache")
                return cached_data
        
        # Load data from source
        if isinstance(data_source, str):
            data = self._load_from_file(data_source)
        elif isinstance(data_source, pd.DataFrame):
            data = data_source.copy()
        else:
            # Try to load default data source
            data = self._load_default_data()
        
        # Process data
        data = self._process_data(data)
        
        # Cache processed data
        if self.config.enable_caching and symbol:
            self._save_to_cache(data, symbol)
        
        return data
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        if not self.cache_directory:
            return None
        
        cache_file = self.cache_directory / f"{symbol}_{self.config.frequency}.{self.config.cache_format}"
        
        if not cache_file.exists():
            return None
        
        try:
            if self.config.cache_format == "parquet":
                return pd.read_parquet(cache_file)
            elif self.config.cache_format == "csv":
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            elif self.config.cache_format == "pickle":
                return pd.read_pickle(cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, symbol: str):
        """Save data to cache."""
        if not self.cache_directory:
            return
        
        cache_file = self.cache_directory / f"{symbol}_{self.config.frequency}.{self.config.cache_format}"
        
        try:
            if self.config.cache_format == "parquet":
                data.to_parquet(cache_file)
            elif self.config.cache_format == "csv":
                data.to_csv(cache_file)
            elif self.config.cache_format == "pickle":
                data.to_pickle(cache_file)
            
            self.logger.info(f"Data cached to {cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")
    
    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            if file_path.suffix == '.parquet':
                return pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                return pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif file_path.suffix == '.json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def _load_default_data(self) -> pd.DataFrame:
        """Load default data for testing."""
        # Generate synthetic data for testing
        self.logger.info("Generating synthetic data for testing")
        
        start_date = self.config.start_date or datetime.now() - timedelta(days=365)
        end_date = self.config.end_date or datetime.now()
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=self.config.frequency)
        
        # Generate synthetic OHLCV data
        np.random.seed(42)  # For reproducible results
        n_points = len(date_range)
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0001, 0.02, n_points)  # Small positive drift
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(date_range, prices)):
            # Generate OHLC from price
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data for backtesting."""
        self.logger.info("Processing data for backtesting")
        
        # Make a copy to avoid modifying original
        processed_data = data.copy()
        
        # Validate data
        if self.config.enable_validation:
            processed_data = self._validate_data(processed_data)
        
        # Preprocess data
        if self.config.enable_preprocessing:
            processed_data = self._preprocess_data(processed_data)
        
        # Feature engineering
        if self.config.enable_feature_engineering:
            processed_data = self._engineer_features(processed_data)
        
        # Outlier detection and handling
        if self.config.enable_outlier_detection:
            processed_data = self._handle_outliers(processed_data)
        
        # Memory optimization
        if self.config.enable_memory_optimization:
            processed_data = self._optimize_memory(processed_data)
        
        return processed_data
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and structure."""
        self.logger.info("Validating data quality")
        
        # Check minimum data points
        if len(data) < self.config.min_data_points:
            raise ValueError(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
        
        # Check maximum data points
        if len(data) > self.config.max_data_points:
            self.logger.warning(f"Truncating data: {len(data)} > {self.config.max_data_points}")
            data = data.tail(self.config.max_data_points)
        
        # Validate data quality if utils available
        if COMMON_UTILS_AVAILABLE:
            try:
                quality_score = calculate_data_quality_metrics(data)
                if quality_score < self.config.quality_threshold:
                    self.logger.warning(f"Data quality score {quality_score:.3f} below threshold {self.config.quality_threshold}")
                
                # Validate columns
                data = validate_dataframe_columns(data)
                
                # Validate timestamps
                if 'timestamp' in data.columns:
                    data = validate_timestamp_column(data, 'timestamp')
                
            except Exception as e:
                self.logger.warning(f"Data validation failed: {e}")
        
        return data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for backtesting."""
        self.logger.info("Preprocessing data")
        
        # Fill missing data
        if self.config.fill_missing_data:
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure proper data types
        if COMMON_UTILS_AVAILABLE:
            try:
                data = safe_convert_dtypes(data)
                data = optimize_dataframe_dtypes(data)
            except Exception as e:
                self.logger.warning(f"Data type optimization failed: {e}")
        
        # Handle infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for backtesting."""
        self.logger.info("Engineering features")
        
        # Add basic technical indicators if OHLCV data is available
        if all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            # Price returns
            data['returns'] = data['close'].pct_change()
            
            # Moving averages
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            
            # Volatility
            data['volatility_20'] = data['returns'].rolling(window=20).std()
            
            # RSI (simplified)
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        self.logger.info("Handling outliers")
        
        # Apply outlier detection to numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['open', 'high', 'low', 'close', 'volume', 'returns']:
                # Calculate z-scores
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                
                # Identify outliers
                outliers = z_scores > self.config.outlier_threshold
                
                if outliers.any():
                    self.logger.info(f"Found {outliers.sum()} outliers in {column}")
                    
                    # Replace outliers with interpolated values
                    data.loc[outliers, column] = np.nan
                    data[column] = data[column].interpolate(method='linear')
        
        return data
    
    def _optimize_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage."""
        if not self.memory_optimizer:
            return data
        
        try:
            # Use memory optimizer if available
            return self.memory_optimizer.optimize_dataframe(data)
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
            return data
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get information about the data."""
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'missing_values': data.isnull().sum().to_dict(),
            'date_range': {
                'start': data.index.min() if hasattr(data.index, 'min') else None,
                'end': data.index.max() if hasattr(data.index, 'max') else None
            }
        }
        
        if COMMON_UTILS_AVAILABLE:
            try:
                info['quality_score'] = calculate_data_quality_metrics(data)
                info['summary_stats'] = create_summary_statistics(data)
            except Exception as e:
                self.logger.warning(f"Failed to calculate additional info: {e}")
        
        return info
    
    def save_data(self, data: pd.DataFrame, file_path: str, format: str = "parquet"):
        """Save processed data to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "parquet":
                data.to_parquet(file_path)
            elif format == "csv":
                data.to_csv(file_path)
            elif format == "json":
                data.to_json(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            raise
    
    def clear_cache(self):
        """Clear the cache directory."""
        if self.cache_directory and self.cache_directory.exists():
            import shutil
            shutil.rmtree(self.cache_directory)
            self.cache_directory.mkdir(parents=True, exist_ok=True)
            self.logger.info("Cache cleared")