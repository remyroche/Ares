"""
Data Manager for TAS Backtesting

Comprehensive data management for tree architecture search backtesting including
data ingestion, preprocessing, validation, and storage.
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
import json
warnings.filterwarnings('ignore')

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Data sources."""
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    API = "api"
    DATABASE = "database"
    MEMORY = "memory"

@dataclass
class DataConfig:
    """Configuration for data management."""

    # Data source
    data_source: DataSource = DataSource.CSV
    data_path: Optional[str] = None
    data_url: Optional[str] = None

    # Data parameters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    data_frequency: str = "1H"  # 1H, 1D, etc.

    # Data validation
    required_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'volume'])
    min_data_points: int = 100
    max_data_points: int = 10000

    # Data preprocessing
    enable_data_cleaning: bool = True
    enable_outlier_detection: bool = True
    enable_missing_data_handling: bool = True
    outlier_threshold: float = 3.0  # Standard deviations

    # Data transformation
    enable_log_returns: bool = True
    enable_technical_indicators: bool = True
    enable_regime_features: bool = True

    # Data storage
    save_processed_data: bool = True
    data_directory: str = "backtesting_data"
    cache_data: bool = True

    # Advanced parameters
    enable_data_validation: bool = True
    enable_data_quality_checks: bool = True
    data_quality_threshold: float = 0.95

@dataclass
class DataResult:
    """Result of data management operations."""

    # Data information
    data_shape: Tuple[int, int]
    data_columns: List[str]
    data_types: Dict[str, str]
    data_range: Tuple[datetime, datetime]

    # Data quality
    missing_values: Dict[str, int]
    outlier_count: int
    data_quality_score: float

    # Processed data
    processed_data: pd.DataFrame
    feature_data: pd.DataFrame
    regime_data: Optional[Dict[str, Any]] = None

    # Metadata
    processing_time: float
    config: DataConfig

class BacktestingDataManager:
    """
    Comprehensive data manager for TAS backtesting.

    Provides data ingestion, preprocessing, validation, and storage
    for tree architecture search backtesting.
    """

    def __init__(self, config: DataConfig):
        """Initialize data manager.

        Args:
            config: Data configuration
        """
        tprint_info("📊 Initializing Backtesting Data Manager")
        tprint_debug(f"Configuration: {config}")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Data state
        tprint_debug("📊 Initializing data state...")
        self.raw_data = None
        self.processed_data = None
        self.feature_data = None
        self.regime_data = None

        tprint_success("✅ Backtesting Data Manager initialized")
        tprint_info(f"📊 Data source: {config.data_source.value}")
        tprint_info(f"📊 Data frequency: {config.data_frequency}")
        self.logger.info("✅ Backtesting Data Manager initialized")
        self.logger.info(f"📊 Data source: {config.data_source.value}")
        self.logger.info(f"📊 Data frequency: {config.data_frequency}")

    def load_data(self, data: Optional[Union[pd.DataFrame, str, Dict[str, Any]]] = None) -> DataResult:
        """
        Load data from various sources.

        Args:
            data: Optional data to load (DataFrame, file path, or data dict)

        Returns:
            Data management result
        """
        tprint_info("🚀 Loading data for backtesting")
        self.logger.info("🚀 Loading data for backtesting")
        start_time = datetime.now()

        try:
            # Load data based on source
            tprint_debug("📥 Loading data based on source...")
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    tprint_debug("📊 Loading from DataFrame...")
                    raw_data = data.copy()
                elif isinstance(data, str):
                    tprint_debug(f"📁 Loading from file: {data}")
                    raw_data = self._load_from_file(data)
                elif isinstance(data, dict):
                    tprint_debug("📋 Loading from dictionary...")
                    raw_data = self._load_from_dict(data)
                else:
                    tprint_error(f"❌ Unsupported data type: {type(data)}")
                    raise ValueError(f"Unsupported data type: {type(data)}")
            else:
                tprint_debug("⚙️ Loading from configuration...")
                raw_data = self._load_from_config()

            tprint_success(f"✅ Data loaded with shape: {raw_data.shape}")

            # Validate data
            tprint_debug("🔍 Validating data...")
            self._validate_data(raw_data)
            tprint_success("✅ Data validation passed")

            # Preprocess data
            tprint_debug("🔧 Preprocessing data...")
            processed_data = self._preprocess_data(raw_data)
            tprint_success(f"✅ Data preprocessed with shape: {processed_data.shape}")

            # Generate features
            tprint_debug("🎯 Generating features...")
            feature_data = self._generate_features(processed_data)
            tprint_success(f"✅ Features generated with shape: {feature_data.shape}")

            # Generate regime data
            tprint_debug("📊 Generating regime data...")
            regime_data = self._generate_regime_data(processed_data)
            tprint_success("✅ Regime data generated")

            # Calculate data quality metrics
            tprint_debug("📈 Calculating data quality metrics...")
            data_quality = self._calculate_data_quality(processed_data)
            tprint_success(f"✅ Data quality score: {data_quality['data_quality_score']:.3f}")

            # Create comprehensive result
            tprint_debug("📋 Creating comprehensive result...")
            result = DataResult(
                # Data information
                data_shape=processed_data.shape,
                data_columns=list(processed_data.columns),
                data_types=processed_data.dtypes.to_dict(),
                data_range=(processed_data.index[0], processed_data.index[-1]),

                # Data quality
                missing_values=data_quality['missing_values'],
                outlier_count=data_quality['outlier_count'],
                data_quality_score=data_quality['data_quality_score'],

                # Processed data
                processed_data=processed_data,
                feature_data=feature_data,
                regime_data=regime_data,

                # Metadata
                processing_time=(datetime.now() - start_time).total_seconds(),
                config=self.config
            )

            # Save processed data if configured
            if self.config.save_processed_data:
                tprint_debug("💾 Saving processed data...")
                self._save_processed_data(result)
                tprint_success("✅ Processed data saved")

            # Store data in manager
            tprint_debug("📊 Storing data in manager...")
            self.raw_data = raw_data
            self.processed_data = processed_data
            self.feature_data = feature_data
            self.regime_data = regime_data

            tprint_success(f"✅ Data loading completed in {result.processing_time:.2f}s")
            tprint_info(f"📊 Data shape: {result.data_shape}")
            tprint_info(f"📊 Data quality score: {result.data_quality_score:.3f}")
            tprint_info(f"📊 Missing values: {sum(result.missing_values.values())}")
            tprint_info(f"📊 Outliers: {result.outlier_count}")
            self.logger.info(f"✅ Data loading completed in {result.processing_time:.2f}s")
            self.logger.info(f"📊 Data shape: {result.data_shape}")
            self.logger.info(f"📊 Data quality score: {result.data_quality_score:.3f}")
            self.logger.info(f"📊 Missing values: {sum(result.missing_values.values())}")
            self.logger.info(f"📊 Outliers: {result.outlier_count}")

            return result

        except Exception as e:
            tprint_error(f"❌ Data loading failed: {e}")
            self.logger.error(f"❌ Data loading failed: {e}")
            raise

    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        if self.config.data_source == DataSource.CSV:
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif self.config.data_source == DataSource.PARQUET:
            return pd.read_parquet(file_path)
        elif self.config.data_source == DataSource.JSON:
            return pd.read_json(file_path, orient='index', date_unit='s')
        else:
            raise ValueError(f"Unsupported file format for source: {self.config.data_source}")

    def _load_from_dict(self, data_dict: Dict[str, Any]) -> pd.DataFrame:
        """Load data from dictionary."""
        # Convert dictionary to DataFrame
        df = pd.DataFrame(data_dict)

        # Set index if datetime column exists
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

        return df

    def _load_from_config(self) -> pd.DataFrame:
        """Load data from configuration."""
        if self.config.data_path:
            return self._load_from_file(self.config.data_path)
        elif self.config.data_url:
            # Load from URL (simplified)
            return pd.read_csv(self.config.data_url, index_col=0, parse_dates=True)
        else:
            raise ValueError("No data source specified in configuration")

    def _validate_data(self, data: pd.DataFrame):
        """Validate data for backtesting."""
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

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for backtesting."""
        processed_data = data.copy()

        # Handle missing data
        if self.config.enable_missing_data_handling:
            processed_data = self._handle_missing_data(processed_data)

        # Detect and handle outliers
        if self.config.enable_outlier_detection:
            processed_data = self._handle_outliers(processed_data)

        # Clean data
        if self.config.enable_data_cleaning:
            processed_data = self._clean_data(processed_data)

        # Filter by date range
        if self.config.start_date:
            processed_data = processed_data[processed_data.index >= self.config.start_date]

        if self.config.end_date:
            processed_data = processed_data[processed_data.index <= self.config.end_date]

        # Resample data if needed
        if self.config.data_frequency != '1H':
            processed_data = self._resample_data(processed_data)

        return processed_data

    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data."""
        # Forward fill for price data
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')

        # Interpolate for volume data
        if 'volume' in data.columns:
            data['volume'] = data['volume'].interpolate()

        # Drop remaining missing values
        data = data.dropna()

        return data

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in data."""
        for col in self.config.required_columns:
            if col in data.columns:
                # Calculate z-scores
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())

                # Identify outliers
                outliers = z_scores > self.config.outlier_threshold

                if outliers.any():
                    self.logger.warning(f"Found {outliers.sum()} outliers in column {col}")

                    # Cap outliers at threshold
                    data.loc[outliers, col] = data[col].mean() + np.sign(data[col] - data[col].mean()) * self.config.outlier_threshold * data[col].std()

        return data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data for backtesting."""
        # Remove duplicate timestamps
        data = data[~data.index.duplicated(keep='first')]

        # Sort by timestamp
        data = data.sort_index()

        # Remove negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]

        # Remove negative volume
        if 'volume' in data.columns:
            data = data[data['volume'] >= 0]

        return data

    def _resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample data to specified frequency."""
        # Resample OHLCV data
        resampled_data = data.resample(self.config.data_frequency).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Remove rows with missing data
        resampled_data = resampled_data.dropna()

        return resampled_data

    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for backtesting."""
        feature_data = data.copy()

        if self.config.enable_log_returns:
            # Calculate log returns
            feature_data['log_return'] = np.log(data['close'] / data['close'].shift(1))
            feature_data['log_return'] = feature_data['log_return'].fillna(0)

        if self.config.enable_technical_indicators:
            # Calculate technical indicators
            feature_data = self._calculate_technical_indicators(feature_data)

        if self.config.enable_regime_features:
            # Calculate regime features
            feature_data = self._calculate_regime_features(feature_data)

        return feature_data

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['ema_20'] = data['close'].ewm(span=20).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()

        # Volatility
        data['volatility_20'] = data['close'].rolling(window=20).std()
        data['volatility_50'] = data['close'].rolling(window=50).std()

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)

        # MACD
        data['macd'] = data['ema_12'] - data['ema_26'] if 'ema_12' in data.columns and 'ema_26' in data.columns else 0
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']

        return data

    def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime features."""
        # Volatility regime
        data['volatility_regime'] = (data['volatility_20'] > data['volatility_20'].rolling(window=50).mean()).astype(int)

        # Trend regime
        data['trend_regime'] = (data['close'] > data['sma_20']).astype(int)

        # Volume regime
        data['volume_regime'] = (data['volume'] > data['volume'].rolling(window=20).mean()).astype(int)

        # Combined regime
        data['combined_regime'] = data['volatility_regime'] * 4 + data['trend_regime'] * 2 + data['volume_regime']

        return data

    def _generate_regime_data(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate regime data for backtesting."""
        if not self.config.enable_regime_features:
            return None

        # Simple regime detection based on features
        regime_labels = data['combined_regime'].values if 'combined_regime' in data.columns else np.zeros(len(data))

        # Calculate regime statistics
        unique_regimes = np.unique(regime_labels)
        regime_stats = {}

        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id
            regime_data = data[regime_mask]

            regime_stats[f'regime_{regime_id}'] = {
                'regime_id': regime_id,
                'count': np.sum(regime_mask),
                'percentage': np.sum(regime_mask) / len(regime_labels),
                'mean_return': regime_data['log_return'].mean() if 'log_return' in regime_data.columns else 0.0,
                'volatility': regime_data['volatility_20'].mean() if 'volatility_20' in regime_data.columns else 0.0,
                'mean_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0
            }

        return {
            'regime_labels': regime_labels,
            'regime_stats': regime_stats,
            'qualified_regimes': regime_stats
        }

    def _calculate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        # Missing values
        missing_values = data.isnull().sum().to_dict()

        # Outlier count
        outlier_count = 0
        for col in self.config.required_columns:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_count += (z_scores > self.config.outlier_threshold).sum()

        # Data quality score
        total_observations = len(data) * len(self.config.required_columns)
        missing_observations = sum(missing_values.values())
        outlier_observations = outlier_count

        data_quality_score = 1.0 - (missing_observations + outlier_observations) / total_observations

        return {
            'missing_values': missing_values,
            'outlier_count': outlier_count,
            'data_quality_score': data_quality_score
        }

    def _save_processed_data(self, result: DataResult):
        """Save processed data to file."""
        try:
            data_dir = Path(self.config.data_directory)
            data_dir.mkdir(parents=True, exist_ok=True)

            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_file = data_dir / f"processed_data_{timestamp}.csv"
            result.processed_data.to_csv(processed_file)

            # Save feature data
            feature_file = data_dir / f"feature_data_{timestamp}.csv"
            result.feature_data.to_csv(feature_file)

            # Save regime data
            if result.regime_data:
                regime_file = data_dir / f"regime_data_{timestamp}.json"
                import json
                with open(regime_file, 'w') as f:
                    json.dump(result.regime_data, f, indent=2, default=str)

            self.logger.info(f"📁 Processed data saved to {data_dir}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save processed data: {e}")

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed data."""
        return self.processed_data

    def get_feature_data(self) -> Optional[pd.DataFrame]:
        """Get feature data."""
        return self.feature_data

    def get_regime_data(self) -> Optional[Dict[str, Any]]:
        """Get regime data."""
        return self.regime_data

    def export_data(self, filepath: str):
        """Export data to file."""
        if self.processed_data is None:
            self.logger.warning("⚠️ No processed data to export")
            return

        try:
            # Export processed data
            self.processed_data.to_csv(filepath)

            # Export feature data
            feature_file = filepath.replace('.csv', '_features.csv')
            if self.feature_data is not None:
                self.feature_data.to_csv(feature_file)

            # Export regime data
            regime_file = filepath.replace('.csv', '_regime.json')
            if self.regime_data is not None:
                with open(regime_file, 'w') as f:
                    json.dump(self.regime_data, f, indent=2, default=str)

            self.logger.info(f"📁 Data exported to {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to export data: {e}")
