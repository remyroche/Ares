from src.utils.tprint import tprint
import warnings

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import joblib
from sklearn.ensemble import RandomForestClassifier
import torch
from sklearn.preprocessing import StandardScaler
import asyncio
import json

class IDataSource(ABC):
    """Interface for data sources."""

    @abstractmethod
    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch data from the source."""

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to data source."""

class IExchangeDataSource(IDataSource):
    """Base interface for exchange data sources."""

    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Name of the exchange."""

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""

    @abstractmethod
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""

class BaseExchangeDataSource(IExchangeDataSource):
    """Base implementation for exchange data sources with common functionality."""

    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._connection_validated = False

    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Must be implemented by subclasses."""

    def validate_connection(self) -> bool:
        """Base connection validation - can be overridden."""
        if self._connection_validated:
            return True
        try:
            self._connection_validated = self._perform_connection_test()
            return self._connection_validated
        except Exception:
            return False

    def _perform_connection_test(self) -> bool:
        """Override in subclasses for exchange-specific connection tests."""
        return True

    def _standardize_ohlcv_data(self, raw_data: Any) -> pd.DataFrame:
        """Standardize OHLCV data format across exchanges."""
        try:
            if isinstance(raw_data, pd.DataFrame):
                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in raw_data.columns]

                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")

                # Standardize column names to lowercase
                standardized_data = raw_data.copy()
                standardized_data.columns = standardized_data.columns.str.lower()

                # Ensure numeric types
                for col in required_columns:
                    if col in standardized_data.columns:
                        standardized_data[col] = pd.to_numeric(standardized_data[col], errors='coerce')

                # Remove any rows with NaN values in required columns
                standardized_data = standardized_data.dropna(subset=required_columns)

                # Ensure timestamp index if not already
                if not isinstance(standardized_data.index, pd.DatetimeIndex):
                    if 'timestamp' in standardized_data.columns:
                        standardized_data = standardized_data.set_index('timestamp')
                    elif 'time' in standardized_data.columns:
                        standardized_data = standardized_data.set_index('time')
                    else:
                        # Create a default timestamp index
                        standardized_data.index = pd.date_range(
                            start='2023-01-01',
                            periods=len(standardized_data),
                            freq='1H'
                        )

                # Sort by timestamp
                standardized_data = standardized_data.sort_index()

                return standardized_data[required_columns]

            elif isinstance(raw_data, dict):
                # Convert dictionary to DataFrame
                df = pd.DataFrame(raw_data)
                return self._standardize_ohlcv_data(df)

            else:
                raise ValueError(f"Unsupported data type: {type(raw_data)}")

        except Exception as e:
            self.logger.error(f"Failed to standardize OHLCV data: {e}")
            raise ValueError(f"Data standardization failed: {e}")

class ExchangeDataSource(BaseExchangeDataSource):
    """Generic exchange data source with configurable parameters."""

    def __init__(self, exchange_name: str, symbols: List[str], timeframes: List[str],
                 price_range: Tuple[float, float], volume_range: Tuple[int, int],
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._exchange_name = exchange_name
        self._symbols = symbols
        self._timeframes = timeframes
        self._price_range = price_range
        self._volume_range = volume_range

    @property
    def exchange_name(self) -> str:
        return self._exchange_name

    def get_supported_symbols(self) -> List[str]:
        return self._symbols

    def get_supported_timeframes(self) -> List[str]:
        return self._timeframes

    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data with configurable price and volume ranges."""
        hours = int((end - start).total_seconds() / 3600)
        base_price = np.random.uniform(*self._price_range)
        price_volatility = base_price * 0.05

        data = pd.DataFrame({
            'timestamp': pd.date_range(start, end, freq='1H')[:hours],
            'open': np.random.randn(hours) * price_volatility + base_price,
            'high': np.random.randn(hours) * price_volatility + base_price * 1.05,
            'low': np.random.randn(hours) * price_volatility + base_price * 0.95,
            'close': np.random.randn(hours) * price_volatility + base_price,
            'volume': np.random.randint(*self._volume_range, hours)
        })
        return data.set_index('timestamp')

    def _perform_connection_test(self) -> bool:
        return True

class ExchangeDataSourceFactory:
    """Factory for creating exchange data sources."""

    # Predefined exchange configurations
    EXCHANGE_CONFIGS = {
        'binance': {
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
            'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
            'price_range': (50, 200),
            'volume_range': (1000, 10000)
        },
        'coinbase': {
            'symbols': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'MATIC-USD'],
            'timeframes': ['1m', '5m', '15m', '1h', '6h', '1d'],
            'price_range': (10000, 100000),
            'volume_range': (100, 1000)
        },
        'kraken': {
            'symbols': ['XXBTZUSD', 'XETHZUSD', 'XLTCZUSD', 'XXRPZUSD'],
            'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
            'price_range': (20000, 80000),
            'volume_range': (500, 5000)
        }
    }

    @classmethod
    def create(cls, exchange: str, **kwargs) -> IExchangeDataSource:
        """Create data source for specified exchange."""
        exchange_lower = exchange.lower()
        if exchange_lower not in cls.EXCHANGE_CONFIGS:
            raise ValueError(f'Unknown exchange: {exchange}. Available: {list(cls.EXCHANGE_CONFIGS.keys())}')

        config = cls.EXCHANGE_CONFIGS[exchange_lower]
        return ExchangeDataSource(
            exchange_name = exchange_lower,
            symbols = config['symbols'],
            timeframes = config['timeframes'],
            price_range = config['price_range'],
            volume_range = config['volume_range'],
            **kwargs
        )

    @classmethod
    def get_available_exchanges(cls) -> List[str]:
        """Get list of available exchanges."""
        return list(cls.EXCHANGE_CONFIGS.keys())

class LocalDataSource(IDataSource):
    """Responsible ONLY for loading data from local files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Load data from local parquet file."""
        file_path = self.data_dir / f'{symbol}.parquet'
        if not file_path.exists():
            raise FileNotFoundError(f'Data file not found: {file_path}')
        data = pd.read_parquet(file_path)
        mask = (data.index >= start) & (data.index <= end)
        return data[mask]

    def validate_connection(self) -> bool:
        """Check if data directory exists."""
        return self.data_dir.exists()

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

class IDataValidator(ABC):
    """Interface for data validators."""

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate the data."""

class DataValidator(IDataValidator):
    """Comprehensive data validator that handles schema, quality, and time series validation."""

    def __init__(self, required_columns: List[str] = None, column_types: Dict[str, type] = None,
                 max_null_percentage: float = 0.1, max_duplicate_percentage: float = 0.01,
                 expected_frequency: str = None, max_gaps: int = 0) -> None:
        self.required_columns = required_columns or []
        self.column_types = column_types or {}
        self.max_null_percentage = max_null_percentage
        self.max_duplicate_percentage = max_duplicate_percentage
        self.expected_frequency = expected_frequency
        self.max_gaps = max_gaps

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Comprehensive data validation."""
        errors = []
        warnings = []
        metrics = {'num_columns': len(data.columns), 'num_rows': len(data)}

        # Schema validation
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            errors.append(f'Missing required columns: {missing_columns}')

        for col, expected_type in self.column_types.items():
            if col in data.columns:
                actual_type = data[col].dtype
                if not np.issubdtype(actual_type, expected_type):
                    warnings.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")

        # Data quality validation
        null_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
        metrics['null_percentage'] = null_percentage
        if null_percentage > self.max_null_percentage:
            errors.append(f'Too many null values: {null_percentage:.2%} > {self.max_null_percentage:.2%}')

        duplicate_count = data.duplicated().sum()
        duplicate_percentage = duplicate_count / len(data)
        metrics['duplicate_percentage'] = duplicate_percentage
        if duplicate_percentage > self.max_duplicate_percentage:
            warnings.append(f'High duplicate rate: {duplicate_percentage:.2%} > {self.max_duplicate_percentage:.2%}')

        # Time series validation
        if isinstance(data.index, pd.DatetimeIndex):
            if not data.index.is_monotonic_increasing:
                errors.append('Time series is not sorted')

            if self.expected_frequency:
                inferred_freq = pd.infer_freq(data.index)
                if inferred_freq != self.expected_frequency:
                    warnings.append(f'Unexpected frequency: {inferred_freq} != {self.expected_frequency}')

            time_diffs = data.index.to_series().diff()
            gaps = time_diffs[time_diffs > time_diffs.mode()[0]]
            metrics['num_gaps'] = len(gaps)
            if len(gaps) > self.max_gaps:
                errors.append(f'Too many gaps in time series: {len(gaps)} > {self.max_gaps}')
        else:
            errors.append('Data index is not DatetimeIndex')

        return ValidationResult(is_valid = len(errors) == 0, errors = errors, warnings = warnings, metrics = metrics)

class IFeatureCalculator(ABC):
    """Interface for feature calculators."""

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from data."""

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of features this calculator produces."""

class FeatureCalculator(IFeatureCalculator):
    """Comprehensive feature calculator that handles price, volume, and technical indicators."""

    def __init__(self, window: int = 20, indicators: List[Dict[str, Any]] = None) -> None:
        self.window = window
        self.indicators = indicators or []

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive features."""
        features = pd.DataFrame(index = data.index)

        # Price-based features
        if 'close' in data.columns:
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log1p(features['returns'])

        if all(col in data.columns for col in ['high', 'low']):
            features['high_low_ratio'] = data['high'] / data['low']
            features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])

        if all(col in data.columns for col in ['close', 'open']):
            features['close_open_ratio'] = data['close'] / data['open']

        # Volume-based features
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(self.window).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['volume_volatility'] = data['volume'].pct_change().rolling(self.window).std()

            if 'close' in data.columns:
                features['price_volume_corr'] = data['close'].pct_change().rolling(self.window).corr(data['volume'].pct_change())

        # Technical indicators
        for indicator in self.indicators:
            name = indicator['name']
            params = indicator.get('params', {})
            period = params.get('period', 14 if name == 'RSI' else 20)

            if name == 'RSI' and 'close' in data.columns:
                features[f"rsi_{period}"] = self._calculate_rsi(data['close'], period)
            elif name == 'SMA' and 'close' in data.columns:
                features[f'sma_{period}'] = data['close'].rolling(period).mean()
            elif name == 'EMA' and 'close' in data.columns:
                features[f'ema_{period}'] = data['close'].ewm(span = period).mean()

        return features

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def get_feature_names(self) -> List[str]:
        """Get list of all features this calculator produces."""
        names = ['returns', 'log_returns', 'high_low_ratio', 'close_open_ratio', 'price_position',
                'volume_sma', 'volume_ratio', 'volume_volatility', 'price_volume_corr']

        for indicator in self.indicators:
            name = indicator['name']
            params = indicator.get('params', {})
            period = params.get('period', 14 if name == 'RSI' else 20)
            if name in ['RSI', 'SMA', 'EMA']:
                names.append(f'{name.lower()}_{period}')

        return names

class IModel(ABC):
    """Base interface for trained models."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classifiers)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""

class IModelTrainer(ABC):
    """Interface for model trainers."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Type/name of the model."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series]=None) -> IModel:
        """Train a model on the data."""

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""

    @abstractmethod
    def set_hyperparameters(self, **hyperparameters) -> None:
        """Update hyperparameters."""

    @abstractmethod
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""

class BaseModelTrainer(IModelTrainer):
    """Base implementation with common functionality for model trainers."""

    def __init__(self, **hyperparameters) -> None:
        self.hyperparameters = self._get_default_hyperparameters()
        self.hyperparameters.update(hyperparameters)
        self.model = None
        self.feature_importance_ = None

    @abstractmethod
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for this model type."""

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return self.hyperparameters.copy()

    def set_hyperparameters(self, **hyperparameters) -> None:
        """Update hyperparameters."""
        self.hyperparameters.update(hyperparameters)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        return self.feature_importance_

class ModelWrapper(IModel):
    """Generic wrapper for any model with standard interface."""

    def __init__(self, model: Any, model_type: str, scaler: Any = None) -> None:
        self.model = model
        self.model_type = model_type
        self.scaler = scaler

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        if self.model_type == 'neural_network':
            X_tensor = torch.FloatTensor(X_scaled)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions = (outputs > 0.5).float().numpy().squeeze()
            return predictions
        else:
            return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        if self.model_type == 'neural_network':
            X_tensor = torch.FloatTensor(X_scaled)
            self.model.eval()
            with torch.no_grad():
                outputs = torch.sigmoid(self.model(X_tensor)).numpy()
            proba = np.column_stack([1 - outputs, outputs])
            return proba
        else:
            return self.model.predict_proba(X_scaled)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents = True, exist_ok = True)

        if self.model_type == 'neural_network':
            torch.save(self.model.state_dict(), path.with_suffix('.pth'))
            if self.scaler:
                joblib.dump(self.scaler, path.with_suffix('.scaler'))
        elif self.model_type == 'xgboost':
            self.model.save_model(str(path))
        else:
            joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model = joblib.load(path)

class LightGBMTrainer(BaseModelTrainer):
    """LightGBM model trainer."""

    @property
    def model_type(self) -> str:
        return self._model_type

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters based on model type."""
        defaults = {
            'lightgbm': {'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 128,
                        'learning_rate': 0.05, 'n_estimators': 1000, 'random_state': 42, 'verbosity': -1,
                        'max_depth': 12, 'min_child_samples': 10},
            'xgboost': {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 10,
                       'learning_rate': 0.05, 'n_estimators': 1000, 'random_state': 42, 'verbosity': 0},
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2,
                             'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1},
            'neural_network': {'hidden_layers': [64, 32], 'activation': 'relu', 'dropout_rate': 0.2,
                              'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100, 'early_stopping_patience': 10}
        }
        return defaults.get(self._model_type, {})
    
    def __init__(self, model_type: str, **hyperparameters) -> None:
        self._model_type = model_type
        super().__init__(**hyperparameters)

    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> IModel:
        """Train model based on type."""
        if self._model_type == 'lightgbm':
            return self._train_lightgbm(X, y, validation_data)
        elif self._model_type == 'xgboost':
            return self._train_xgboost(X, y, validation_data)
        elif self._model_type == 'random_forest':
            return self._train_random_forest(X, y, validation_data)
        elif self._model_type == 'neural_network':
            return self._train_neural_network(X, y, validation_data)
        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")

    def _train_lightgbm(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> IModel:
        """Train LightGBM model."""
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(**self.hyperparameters)
        eval_set = [(X, y)]
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_set.append((X_val, y_val))
        self.model.fit(X, y, eval_set = eval_set, callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({'feature': X.columns, 'importance': self.model.feature_importances_}).sort_values('importance', ascending = False)
        return ModelWrapper(self.model, 'lightgbm')

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> IModel:
        """Train XGBoost model."""
        import xgboost as xgb

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> IModel:
        """Train XGBoost model."""
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**self.hyperparameters)
        eval_set = [(X, y)]
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_set.append((X_val, y_val))
        self.model.fit(X, y, eval_set = eval_set, early_stopping_rounds = 10, verbose = False)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({'feature': X.columns, 'importance': self.model.feature_importances_}).sort_values('importance', ascending = False)
        return ModelWrapper(self.model, 'xgboost')

class RandomForestModel(IModel):
    """Random Forest model wrapper."""
    
    def __init__(self, rf_model: Any) -> None:
        self.model = rf_model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents = True, exist_ok = True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model = joblib.load(path)

class RandomForestTrainer(BaseModelTrainer):
    """Random Forest model trainer."""

    @property
    def model_type(self) -> str:
        return 'random_forest'

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get Random Forest default hyperparameters."""
        return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1}

    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series]=None) -> IModel:
        """Train Random Forest model."""
        self.model = RandomForestClassifier(**self.hyperparameters)
        self.model.fit(X, y)
        self.feature_importance_ = pd.DataFrame({'feature': X.columns, 'importance': self.model.feature_importances_}).sort_values('importance', ascending = False)
        return ModelWrapper(self.model, 'random_forest')

class NeuralNetworkModel(IModel):
    """Wrapper for Neural Network model with standard interface."""

    def __init__(self, nn_model: Any, scaler: Any = None) -> None:
        self.model = nn_model
        self.scaler = scaler

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X) if self.scaler else X.values
        X_tensor = torch.FloatTensor(X_scaled)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float().numpy().squeeze()
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X) if self.scaler else X.values
        X_tensor = torch.FloatTensor(X_scaled)
        self.model.eval()
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(X_tensor)).numpy()
        proba = np.column_stack([1 - outputs, outputs])
        return proba

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents = True, exist_ok = True)
        torch.save(self.model.state_dict(), path.with_suffix('.pth'))
        if self.scaler:
            joblib.dump(self.scaler, path.with_suffix('.scaler'))

    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model.load_state_dict(torch.load(path.with_suffix('.pth')))
        scaler_path = path.with_suffix('.scaler')
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

class NeuralNetworkTrainer(BaseModelTrainer):
    """Neural Network model trainer using PyTorch."""

    @property
    def model_type(self) -> str:
        return 'neural_network'

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get Neural Network default hyperparameters."""
        return {'hidden_layers': [64, 32], 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100, 'early_stopping_patience': 10}

    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series]=None) -> IModel:
        """Train Neural Network model."""

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_size = X.shape[1]
        hidden_layers = self.hyperparameters['hidden_layers']

        import torch.nn as nn
        import torch.optim as optim
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if self.hyperparameters['activation'] == 'relu' else nn.Tanh(),
                nn.Dropout(self.hyperparameters['dropout_rate'])
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))

        model = nn.Sequential(*layers)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr = self.hyperparameters['learning_rate'])

        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)

        model.train()
        for epoch in range(self.hyperparameters['epochs']):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.model = model
        return ModelWrapper(model, 'neural_network', scaler)

class ModelTrainerFactory:
    """Factory for creating model trainers."""

    SUPPORTED_MODELS = ['lightgbm', 'xgboost', 'random_forest', 'neural_network']

    @classmethod
    def create(cls, model_type: str, **hyperparameters) -> IModelTrainer:
        """Create model trainer for specified type."""
        model_type_lower = model_type.lower()
        if model_type_lower not in cls.SUPPORTED_MODELS:
            raise ValueError(f'Unknown model type: {model_type}. Available: {cls.SUPPORTED_MODELS}')
        
        if model_type_lower == 'lightgbm':
            return LightGBMTrainer(**hyperparameters)
        elif model_type_lower == 'xgboost':
            return LightGBMTrainer(**hyperparameters)  # Using LightGBMTrainer as base
        elif model_type_lower == 'random_forest':
            return RandomForestTrainer(**hyperparameters)
        elif model_type_lower == 'neural_network':
            return NeuralNetworkTrainer(**hyperparameters)
        else:
            return LightGBMTrainer(**hyperparameters)

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return cls.SUPPORTED_MODELS.copy()

class SimplifiedPipeline:
    """
    Orchestrator that combines single-responsibility components.

    This class itself has a single responsibility: orchestration.
    It doesn't implement any business logic, just coordinates components.
    """

    def __init__(self, data_source: IDataSource, validators: List[IDataValidator], feature_calculators: List[IFeatureCalculator], model_trainer: IModelTrainer, logger: logging.Logger = None) -> None:
        self.data_source = data_source
        self.validators = validators
        self.feature_calculators = feature_calculators
        self.model_trainer = model_trainer
        self.logger = logger or logging.getLogger(__name__)

    async def run(self, symbol: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """Run the pipeline."""
        results = {}
        self.logger.info(f'Loading data for {symbol}')
        data = await self.data_source.fetch_data(symbol, start, end)
        results['data'] = data
        self.logger.info('Validating data')
        for validator in self.validators:
            validation_result = validator.validate(data)
            if not validation_result.is_valid:
                raise ValueError(f'Validation failed: {validation_result.errors}')
        self.logger.info('Calculating features')
        features = pd.DataFrame(index = data.index)
        for calculator in self.feature_calculators:
            calculated_features = calculator.calculate(data)
            features = pd.concat([features, calculated_features], axis = 1)
        results['features'] = features
        self.logger.info('Training model')
        labels = (data['close'].pct_change() > 0).astype(int)
        model = self.model_trainer.train(features.fillna(0), labels)
        results['model'] = model
        return results

async def example_usage() -> None:
    """Example of using simplified modular components."""
    # Create data source
    data_source = LocalDataSource(Path('data/cache'))

    # Create comprehensive validator
    validator = DataValidator(
        required_columns=['open', 'high', 'low', 'close', 'volume'],
        column_types={'volume': np.number},
        max_null_percentage = 0.05,
        expected_frequency='H',
        max_gaps = 5
    )

    # Create comprehensive feature calculator
    feature_calculator = FeatureCalculator(
        window = 20,
        indicators=[
            {'name': 'RSI', 'params': {'period': 14}},
            {'name': 'SMA', 'params': {'period': 20}},
            {'name': 'EMA', 'params': {'period': 12}}
        ]
    )

    # Create model trainer
    model_trainer = ModelTrainerFactory.create('lightgbm', num_leaves = 31, learning_rate = 0.05, n_estimators = 100)

    # Create and run pipeline
    pipeline = SimplifiedPipeline(
        data_source = data_source,
        validators=[validator],
        feature_calculators=[feature_calculator],
        model_trainer = model_trainer
    )

    results = await pipeline.run(symbol='BTCUSDT', start = datetime(2023, 1, 1), end = datetime(2023, 12, 31))
    tprint('Pipeline completed successfully!')
    tprint(f"Features calculated: {results['features'].columns.tolist()}")

if __name__ == '__main__':
    asyncio.run(example_usage())

class VectorBTOptimizedFeatureCalculator(FeatureCalculator):
    """Feature calculator with VectorBT optimization."""
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
