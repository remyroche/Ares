"""
Data Ingestion for TAS

Comprehensive data ingestion system for tree architecture search including
historical data loading, multiple data sources, and data format standardization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import existing utilities from the codebase
try:
    from src.utils.data.klines_parquet import get_klines_manager
    KLINES_MANAGER_AVAILABLE = True
except ImportError:
    KLINES_MANAGER_AVAILABLE = False

try:
    from src.utils.data.processing.data_processing import DataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    DATA_PROCESSOR_AVAILABLE = False

try:
    from src.utils.parquet_utils import ParquetUtils
    PARQUET_UTILS_AVAILABLE = True
except ImportError:
    PARQUET_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Data sources for ingestion."""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    BINANCE_API = "binance_api"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    MEMORY = "memory"

class DataFormat(Enum):
    """Data formats."""
    OHLCV = "ohlcv"
    PRICE_ONLY = "price_only"
    FEATURES = "features"
    REGIME_LABELS = "regime_labels"
    CUSTOM = "custom"

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""

    # Data source configuration
    data_source: DataSource = DataSource.PARQUET
    data_format: DataFormat = DataFormat.OHLCV

    # Data paths and directories
    data_directory: str = "historical_data"
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1h"  # 1h base timeframe for regime detection

    # Date range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_data_points: int = 100000

    # Data source specific configuration
    parquet_config: Dict[str, Any] = field(default_factory=dict)
    csv_config: Dict[str, Any] = field(default_factory=dict)
    api_config: Dict[str, Any] = field(default_factory=dict)

    # Data validation
    validate_data: bool = True
    min_data_points: int = 100
    required_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'volume'])

    # Data processing
    enable_data_cleaning: bool = True
    enable_timestamp_regularization: bool = True
    enable_outlier_detection: bool = True

    # Output configuration
    save_ingested_data: bool = True
    output_directory: str = "ingested_data"
    cache_data: bool = True

@dataclass
class DataIngestionResult:
    """Result of data ingestion."""

    # Data information
    data: pd.DataFrame
    data_shape: Tuple[int, int]
    data_columns: List[str]
    data_types: Dict[str, str]
    data_range: Tuple[datetime, datetime]

    # Data quality
    data_quality_score: float
    missing_values: Dict[str, int]
    outlier_count: int
    irregular_intervals: int

    # Metadata
    ingestion_time: float
    data_source: str
    data_format: str
    config: DataIngestionConfig

    # Additional information
    preprocessing_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class DataIngestionManager:
    """
    Comprehensive data ingestion manager for TAS.

    Provides data loading from multiple sources, format standardization,
    and data quality validation for tree architecture search.
    """

    def __init__(self, config: DataIngestionConfig):
        """Initialize data ingestion manager.

        Args:
            config: Data ingestion configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize data processors
        self.klines_manager = None
        self.data_processor = None
        self.parquet_utils = None

        # Initialize available processors
        self._initialize_processors()

        self.logger.info("✅ Data Ingestion Manager initialized")
        self.logger.info(f"📊 Data source: {config.data_source.value}")
        self.logger.info(f"📊 Data format: {config.data_format.value}")
        self.logger.info(f"📊 Symbol: {config.symbol}")
        self.logger.info(f"📊 Timeframe: {config.timeframe}")

    def _initialize_processors(self):
        """Initialize available data processors."""
        # Initialize klines manager if available
        if KLINES_MANAGER_AVAILABLE:
            try:
                self.klines_manager = get_klines_manager()
                self.logger.info("✅ Klines manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Klines manager not available: {e}")

        # Initialize data processor if available
        if DATA_PROCESSOR_AVAILABLE:
            try:
                self.data_processor = DataProcessor()
                self.logger.info("✅ Data processor initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Data processor not available: {e}")

        # Initialize parquet utils if available
        if PARQUET_UTILS_AVAILABLE:
            try:
                self.parquet_utils = ParquetUtils()
                self.logger.info("✅ Parquet utils initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Parquet utils not available: {e}")

    def ingest_data(self,
                   data: Optional[Union[pd.DataFrame, str, Dict[str, Any]]] = None,
                   custom_loader: Optional[Callable] = None) -> DataIngestionResult:
        """
        Ingest data from various sources.

        Args:
            data: Optional data to ingest (DataFrame, file path, or data dict)
            custom_loader: Optional custom data loader function

        Returns:
            Data ingestion result
        """
        self.logger.info("🚀 Starting data ingestion")
        start_time = datetime.now()

        try:
            # Load data based on source
            if data is not None:
                raw_data = self._load_provided_data(data)
            elif custom_loader is not None:
                raw_data = self._load_custom_data(custom_loader)
            else:
                raw_data = self._load_from_config()

            # Validate data
            if self.config.validate_data:
                self._validate_data(raw_data)

            # Process data
            processed_data = self._process_data(raw_data)

            # Calculate data quality metrics
            quality_metrics = self._calculate_quality_metrics(processed_data)

            # Create comprehensive result
            result = DataIngestionResult(
                # Data information
                data=processed_data,
                data_shape=processed_data.shape,
                data_columns=list(processed_data.columns),
                data_types=processed_data.dtypes.to_dict(),
                data_range=(processed_data.index[0], processed_data.index[-1]),

                # Data quality
                data_quality_score=quality_metrics['data_quality_score'],
                missing_values=quality_metrics['missing_values'],
                outlier_count=quality_metrics['outlier_count'],
                irregular_intervals=quality_metrics['irregular_intervals'],

                # Metadata
                ingestion_time=(datetime.now() - start_time).total_seconds(),
                data_source=self.config.data_source.value,
                data_format=self.config.data_format.value,
                config=self.config,

                # Additional information
                preprocessing_applied=quality_metrics['preprocessing_applied'],
                warnings=quality_metrics['warnings'],
                errors=quality_metrics['errors']
            )

            # Save ingested data if configured
            if self.config.save_ingested_data:
                self._save_ingested_data(result)

            self.logger.info(f"✅ Data ingestion completed in {result.ingestion_time:.2f}s")
            self.logger.info(f"📊 Data shape: {result.data_shape}")
            self.logger.info(f"📊 Data quality score: {result.data_quality_score:.3f}")
            self.logger.info(f"📊 Missing values: {sum(result.missing_values.values())}")
            self.logger.info(f"📊 Outliers: {result.outlier_count}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Data ingestion failed: {e}")
            raise

    def _load_provided_data(self, data: Union[pd.DataFrame, str, Dict[str, Any]]) -> pd.DataFrame:
        """Load data from provided source."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, str):
            return self._load_from_file(data)
        elif isinstance(data, dict):
            return self._load_from_dict(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _load_custom_data(self, custom_loader: Callable) -> pd.DataFrame:
        """Load data using custom loader."""
        try:
            return custom_loader()
        except Exception as e:
            self.logger.error(f"❌ Custom data loader failed: {e}")
            raise

    def _load_from_config(self) -> pd.DataFrame:
        """Load data from configuration."""
        if self.config.data_source == DataSource.PARQUET:
            return self._load_parquet_data()
        elif self.config.data_source == DataSource.CSV:
            return self._load_csv_data()
        elif self.config.data_source == DataSource.JSON:
            return self._load_json_data()
        elif self.config.data_source == DataSource.BINANCE_API:
            return self._load_binance_api_data()
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")

    def _load_parquet_data(self) -> pd.DataFrame:
        """Load data from parquet files."""
        try:
            if self.klines_manager is not None:
                # Use klines manager for parquet data
                data_info = self.klines_manager.get_data_info(
                    symbol=self.config.symbol,
                    interval=self.config.timeframe,
                    data_type="processed"
                )

                if not data_info.get("available", False):
                    raise ValueError(f"No parquet data available for {self.config.symbol} {self.config.timeframe}")

                # Load data using klines manager
                data = self.klines_manager.load_data(
                    symbol=self.config.symbol,
                    interval=self.config.timeframe,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date
                )

                return data
            else:
                # Fallback to direct parquet loading
                return self._load_parquet_direct()

        except Exception as e:
            self.logger.error(f"❌ Failed to load parquet data: {e}")
            raise

    def _load_parquet_direct(self) -> pd.DataFrame:
        """Load parquet data directly."""
        data_dir = Path(self.config.data_directory)
        symbol = self.config.symbol.lower()
        timeframe = self.config.timeframe

        # Look for parquet files
        pattern = f"{symbol}_{timeframe}*.parquet"
        files = list(data_dir.glob(pattern))

        if not files:
            raise ValueError(f"No parquet files found for {symbol} {timeframe}")

        # Load and combine files
        dataframes = []
        for file_path in files:
            try:
                if self.parquet_utils:
                    df = self.parquet_utils.safe_read_parquet(str(file_path))
                else:
                    df = pd.read_parquet(file_path)

                if df is not None and not df.empty:
                    dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")

        if not dataframes:
            raise ValueError("No valid parquet files could be loaded")

        # Combine dataframes
        data = pd.concat(dataframes, ignore_index=False)
        data = data.sort_index()

        return data

    def _load_csv_data(self) -> pd.DataFrame:
        """Load data from CSV files."""
        data_dir = Path(self.config.data_directory)
        symbol = self.config.symbol.lower()
        timeframe = self.config.timeframe

        # Look for CSV files
        pattern = f"{symbol}_{timeframe}*.csv"
        files = list(data_dir.glob(pattern))

        if not files:
            raise ValueError(f"No CSV files found for {symbol} {timeframe}")

        # Load and combine files
        dataframes = []
        for file_path in files:
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not df.empty:
                    dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")

        if not dataframes:
            raise ValueError("No valid CSV files could be loaded")

        # Combine dataframes
        data = pd.concat(dataframes, ignore_index=False)
        data = data.sort_index()

        return data

    def _load_json_data(self) -> pd.DataFrame:
        """Load data from JSON files."""
        data_dir = Path(self.config.data_directory)
        symbol = self.config.symbol.lower()
        timeframe = self.config.timeframe

        # Look for JSON files
        pattern = f"{symbol}_{timeframe}*.json"
        files = list(data_dir.glob(pattern))

        if not files:
            raise ValueError(f"No JSON files found for {symbol} {timeframe}")

        # Load and combine files
        dataframes = []
        for file_path in files:
            try:
                df = pd.read_json(file_path, orient='index', date_unit='s')
                if not df.empty:
                    dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")

        if not dataframes:
            raise ValueError("No valid JSON files could be loaded")

        # Combine dataframes
        data = pd.concat(dataframes, ignore_index=False)
        data = data.sort_index()

        return data

    def _load_binance_api_data(self) -> pd.DataFrame:
        """Load data from Binance API (placeholder)."""
        # This would implement actual API calls to Binance
        # For now, return empty DataFrame
        self.logger.warning("⚠️ Binance API loading not implemented, returning empty DataFrame")
        return pd.DataFrame()

    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.parquet':
            if self.parquet_utils:
                return self.parquet_utils.safe_read_parquet(str(file_path))
            else:
                return pd.read_parquet(file_path)
        elif file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path, orient='index', date_unit='s')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_from_dict(self, data_dict: Dict[str, Any]) -> pd.DataFrame:
        """Load data from dictionary."""
        # Convert dictionary to DataFrame
        df = pd.DataFrame(data_dict)

        # Set index if datetime column exists
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        return df

    def _validate_data(self, data: pd.DataFrame):
        """Validate data for ingestion."""
        # Check required columns
        for col in self.config.required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check data length
        if len(data) < self.config.min_data_points:
            raise ValueError(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")

        if len(data) > self.config.max_data_points:
            self.logger.warning(f"Large dataset: {len(data)} points, using first {self.config.max_data_points}")
            data = data.head(self.config.max_data_points)

        # Check for missing values
        missing_values = data[self.config.required_columns].isnull().sum()
        if missing_values.any():
            self.logger.warning(f"Missing values detected: {missing_values.to_dict()}")

        # Check data types
        for col in self.config.required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                self.logger.warning(f"Non-numeric data in column {col}")

        self.logger.info(f"✅ Data validation passed: {len(data)} data points")

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data for ingestion."""
        processed_data = data.copy()

        # Apply data cleaning if configured
        if self.config.enable_data_cleaning and self.data_processor:
            try:
                processed_data, cleaning_metadata = self.data_processor.clean_data(processed_data)
                self.logger.info(f"✅ Data cleaning applied: {cleaning_metadata.get('cleaning_steps', [])}")
            except Exception as e:
                self.logger.warning(f"⚠️ Data cleaning failed: {e}")

        # Apply timestamp regularization if configured
        if self.config.enable_timestamp_regularization and self.data_processor:
            try:
                processed_data = self.data_processor.regularize_timestamps(processed_data)
                self.logger.info("✅ Timestamp regularization applied")
            except Exception as e:
                self.logger.warning(f"⚠️ Timestamp regularization failed: {e}")

        # Apply outlier detection if configured
        if self.config.enable_outlier_detection and self.data_processor:
            try:
                processed_data, outlier_metadata = self.data_processor.detect_outliers(processed_data)
                self.logger.info(f"✅ Outlier detection applied: {outlier_metadata.get('outlier_count', 0)} outliers detected")
            except Exception as e:
                self.logger.warning(f"⚠️ Outlier detection failed: {e}")

        return processed_data

    def _calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        # Missing values
        missing_values = data.isnull().sum().to_dict()

        # Outlier count (simplified)
        outlier_count = 0
        for col in data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outlier_count += (z_scores > 3).sum()

        # Irregular intervals
        irregular_intervals = 0
        if isinstance(data.index, pd.DatetimeIndex):
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                irregular_intervals = (abs(time_diffs - expected_interval) > timedelta(seconds=30)).sum()

        # Data quality score
        total_observations = len(data) * len(data.columns)
        missing_observations = sum(missing_values.values())
        outlier_observations = outlier_count

        data_quality_score = 1.0 - (missing_observations + outlier_observations) / total_observations

        return {
            'data_quality_score': data_quality_score,
            'missing_values': missing_values,
            'outlier_count': outlier_count,
            'irregular_intervals': irregular_intervals,
            'preprocessing_applied': ['data_cleaning', 'timestamp_regularization', 'outlier_detection'],
            'warnings': [],
            'errors': []
        }

    def _save_ingested_data(self, result: DataIngestionResult):
        """Save ingested data to file."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.symbol.lower()}_{self.config.timeframe}_ingested_{timestamp}.parquet"
            filepath = output_dir / filename

            result.data.to_parquet(filepath)

            # Save metadata
            metadata_file = output_dir / f"{filename.replace('.parquet', '_metadata.json')}"
            import json
            metadata = {
                'data_shape': result.data_shape,
                'data_columns': result.data_columns,
                'data_quality_score': result.data_quality_score,
                'missing_values': result.missing_values,
                'outlier_count': result.outlier_count,
                'ingestion_time': result.ingestion_time,
                'data_source': result.data_source,
                'data_format': result.data_format
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"📁 Ingested data saved to {filepath}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save ingested data: {e}")

    def get_data_info(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """Get information about available data."""
        if symbol is None:
            symbol = self.config.symbol
        if timeframe is None:
            timeframe = self.config.timeframe

        try:
            if self.klines_manager is not None:
                return self.klines_manager.get_data_info(
                    symbol=symbol,
                    interval=timeframe,
                    data_type="processed"
                )
            else:
                # Fallback to directory scanning
                data_dir = Path(self.config.data_directory)
                pattern = f"{symbol.lower()}_{timeframe}*.parquet"
                files = list(data_dir.glob(pattern))

                return {
                    "available": len(files) > 0,
                    "files_count": len(files),
                    "total_records": 0,  # Would need to read files to calculate
                    "date_range": None,
                    "file_size_mb": 0
                }

        except Exception as e:
            self.logger.error(f"❌ Failed to get data info: {e}")
            return {
                "available": False,
                "files_count": 0,
                "total_records": 0,
                "date_range": None,
                "file_size_mb": 0
            }

    def export_data(self, result: DataIngestionResult, filepath: str):
        """Export ingested data to file."""
        try:
            result.data.to_csv(filepath)
            self.logger.info(f"📁 Data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to export data: {e}")
