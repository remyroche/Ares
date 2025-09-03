"""
Modular Components with Single Responsibility

This module demonstrates the Single Responsibility Principle (SRP) by breaking down
complex pipeline components into focused, single-purpose modules.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path


# ============================================================================
# Data Loading Components - Single Responsibility: Load data from sources
# ============================================================================

class IDataSource(ABC):
    """Interface for data sources."""
    
    @abstractmethod
    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to data source."""
        pass


class IExchangeDataSource(IDataSource):
    """Base interface for exchange data sources."""
    
    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Name of the exchange."""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""
        pass
    
    @abstractmethod
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        pass


class BaseExchangeDataSource(IExchangeDataSource):
    """Base implementation for exchange data sources with common functionality."""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._connection_validated = False
    
    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Must be implemented by subclasses."""
        pass
    
    def validate_connection(self) -> bool:
        """Base connection validation - can be overridden."""
        if self._connection_validated:
            return True
        
        # Attempt basic validation
        try:
            # Subclasses can override for specific validation
            self._connection_validated = self._perform_connection_test()
            return self._connection_validated
        except Exception:
            return False
    
    def _perform_connection_test(self) -> bool:
        """Override in subclasses for exchange-specific connection tests."""
        return True
    
    def _standardize_ohlcv_data(self, raw_data: Any) -> pd.DataFrame:
        """Standardize OHLCV data format across exchanges."""
        # Subclasses should convert their specific format to standard format
        # Standard columns: timestamp, open, high, low, close, volume
        raise NotImplementedError("Subclasses must implement data standardization")


class BinanceDataSource(BaseExchangeDataSource):
    """Binance-specific data source implementation."""
    
    @property
    def exchange_name(self) -> str:
        return "binance"
    
    def get_supported_symbols(self) -> List[str]:
        """Get Binance trading pairs."""
        # In real implementation, would fetch from API
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    
    def get_supported_timeframes(self) -> List[str]:
        """Get Binance supported timeframes."""
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    
    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from Binance."""
        # Real implementation would use ccxt or binance-python
        # This is a mock implementation
        hours = int((end - start).total_seconds() / 3600)
        data = pd.DataFrame({
            'timestamp': pd.date_range(start, end, freq='1H')[:hours],
            'open': np.random.randn(hours) * 10 + 100,
            'high': np.random.randn(hours) * 10 + 105,
            'low': np.random.randn(hours) * 10 + 95,
            'close': np.random.randn(hours) * 10 + 100,
            'volume': np.random.randint(1000, 10000, hours)
        })
        return data.set_index('timestamp')
    
    def _perform_connection_test(self) -> bool:
        """Test Binance API connectivity."""
        # Would implement actual API ping
        return True


class CoinbaseDataSource(BaseExchangeDataSource):
    """Coinbase-specific data source implementation."""
    
    @property
    def exchange_name(self) -> str:
        return "coinbase"
    
    def get_supported_symbols(self) -> List[str]:
        """Get Coinbase trading pairs."""
        return ["BTC-USD", "ETH-USD", "SOL-USD", "MATIC-USD"]
    
    def get_supported_timeframes(self) -> List[str]:
        """Get Coinbase supported timeframes."""
        return ["1m", "5m", "15m", "1h", "6h", "1d"]
    
    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from Coinbase."""
        # Convert symbol format if needed (BTCUSDT -> BTC-USD)
        # Real implementation would use Coinbase API
        hours = int((end - start).total_seconds() / 3600)
        data = pd.DataFrame({
            'timestamp': pd.date_range(start, end, freq='1H')[:hours],
            'open': np.random.randn(hours) * 15 + 50000,
            'high': np.random.randn(hours) * 15 + 50500,
            'low': np.random.randn(hours) * 15 + 49500,
            'close': np.random.randn(hours) * 15 + 50000,
            'volume': np.random.randint(100, 1000, hours)
        })
        return data.set_index('timestamp')


class KrakenDataSource(BaseExchangeDataSource):
    """Kraken-specific data source implementation."""
    
    @property
    def exchange_name(self) -> str:
        return "kraken"
    
    def get_supported_symbols(self) -> List[str]:
        """Get Kraken trading pairs."""
        return ["XXBTZUSD", "XETHZUSD", "XLTCZUSD", "XXRPZUSD"]
    
    def get_supported_timeframes(self) -> List[str]:
        """Get Kraken supported timeframes."""
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    
    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from Kraken."""
        # Real implementation would use Kraken API
        hours = int((end - start).total_seconds() / 3600)
        data = pd.DataFrame({
            'timestamp': pd.date_range(start, end, freq='1H')[:hours],
            'open': np.random.randn(hours) * 12 + 45000,
            'high': np.random.randn(hours) * 12 + 45500,
            'low': np.random.randn(hours) * 12 + 44500,
            'close': np.random.randn(hours) * 12 + 45000,
            'volume': np.random.randint(500, 5000, hours)
        })
        return data.set_index('timestamp')


class ExchangeDataSourceFactory:
    """Factory for creating exchange data sources."""
    
    _registry: Dict[str, Type[IExchangeDataSource]] = {
        'binance': BinanceDataSource,
        'coinbase': CoinbaseDataSource,
        'kraken': KrakenDataSource,
    }
    
    @classmethod
    def register_exchange(cls, name: str, data_source_class: Type[IExchangeDataSource]):
        """Register a new exchange data source."""
        cls._registry[name.lower()] = data_source_class
    
    @classmethod
    def create(cls, exchange: str, **kwargs) -> IExchangeDataSource:
        """Create data source for specified exchange."""
        exchange_lower = exchange.lower()
        
        if exchange_lower not in cls._registry:
            raise ValueError(
                f"Unknown exchange: {exchange}. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        return cls._registry[exchange_lower](**kwargs)
    
    @classmethod
    def get_available_exchanges(cls) -> List[str]:
        """Get list of available exchanges."""
        return list(cls._registry.keys())


class LocalDataSource(IDataSource):
    """Responsible ONLY for loading data from local files."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
    
    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Load data from local parquet file."""
        file_path = self.data_dir / f"{symbol}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = pd.read_parquet(file_path)
        # Filter by date range
        mask = (data.index >= start) & (data.index <= end)
        return data[mask]
    
    def validate_connection(self) -> bool:
        """Check if data directory exists."""
        return self.data_dir.exists()


# ============================================================================
# Data Validation Components - Single Responsibility: Validate data quality
# ============================================================================

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
        pass


class SchemaValidator(IDataValidator):
    """Responsible ONLY for validating data schema."""
    
    def __init__(self, required_columns: List[str], column_types: Dict[str, type] = None):
        self.required_columns = required_columns
        self.column_types = column_types or {}
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data has required columns and types."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check column types
        for col, expected_type in self.column_types.items():
            if col in data.columns:
                actual_type = data[col].dtype
                if not np.issubdtype(actual_type, expected_type):
                    warnings.append(
                        f"Column '{col}' has type {actual_type}, expected {expected_type}"
                    )
        
        metrics['num_columns'] = len(data.columns)
        metrics['num_rows'] = len(data)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )


class DataQualityValidator(IDataValidator):
    """Responsible ONLY for validating data quality metrics."""
    
    def __init__(
        self,
        max_null_percentage: float = 0.1,
        max_duplicate_percentage: float = 0.01
    ):
        self.max_null_percentage = max_null_percentage
        self.max_duplicate_percentage = max_duplicate_percentage
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data quality metrics."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check null values
        null_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
        metrics['null_percentage'] = null_percentage
        
        if null_percentage > self.max_null_percentage:
            errors.append(
                f"Too many null values: {null_percentage:.2%} > {self.max_null_percentage:.2%}"
            )
        
        # Check duplicates
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = duplicate_count / len(data)
        metrics['duplicate_percentage'] = duplicate_percentage
        
        if duplicate_percentage > self.max_duplicate_percentage:
            warnings.append(
                f"High duplicate rate: {duplicate_percentage:.2%} > {self.max_duplicate_percentage:.2%}"
            )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )


class TimeSeriesValidator(IDataValidator):
    """Responsible ONLY for validating time series properties."""
    
    def __init__(self, expected_frequency: str = None, max_gaps: int = 0):
        self.expected_frequency = expected_frequency
        self.max_gaps = max_gaps
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate time series specific properties."""
        errors = []
        warnings = []
        metrics = {}
        
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Data index is not DatetimeIndex")
            return ValidationResult(False, errors, warnings, metrics)
        
        # Check if sorted
        if not data.index.is_monotonic_increasing:
            errors.append("Time series is not sorted")
        
        # Check frequency
        if self.expected_frequency:
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq != self.expected_frequency:
                warnings.append(
                    f"Unexpected frequency: {inferred_freq} != {self.expected_frequency}"
                )
        
        # Check gaps
        time_diffs = data.index.to_series().diff()
        gaps = time_diffs[time_diffs > time_diffs.mode()[0]]
        metrics['num_gaps'] = len(gaps)
        
        if len(gaps) > self.max_gaps:
            errors.append(f"Too many gaps in time series: {len(gaps)} > {self.max_gaps}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )


# ============================================================================
# Feature Engineering Components - Single Responsibility: Calculate features
# ============================================================================

class IFeatureCalculator(ABC):
    """Interface for feature calculators."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from data."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of features this calculator produces."""
        pass


class PriceFeatureCalculator(IFeatureCalculator):
    """Responsible ONLY for calculating price-based features."""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features."""
        features = pd.DataFrame(index=data.index)
        
        # Returns
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log1p(features['returns'])
        
        # Price ratios
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        
        # Price position
        features['price_position'] = (
            (data['close'] - data['low']) / (data['high'] - data['low'])
        )
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of price features."""
        return ['returns', 'log_returns', 'high_low_ratio', 
                'close_open_ratio', 'price_position']


class VolumeFeatureCalculator(IFeatureCalculator):
    """Responsible ONLY for calculating volume-based features."""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        features = pd.DataFrame(index=data.index)
        
        # Volume metrics
        features['volume_sma'] = data['volume'].rolling(self.window).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma']
        features['volume_volatility'] = (
            data['volume'].pct_change().rolling(self.window).std()
        )
        
        # Price-volume interaction
        features['price_volume_corr'] = (
            data['close'].pct_change()
            .rolling(self.window)
            .corr(data['volume'].pct_change())
        )
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of volume features."""
        return ['volume_sma', 'volume_ratio', 'volume_volatility', 
                'price_volume_corr']


class TechnicalIndicatorCalculator(IFeatureCalculator):
    """Responsible ONLY for calculating technical indicators."""
    
    def __init__(self, indicators: List[Dict[str, Any]]):
        self.indicators = indicators
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate specified technical indicators."""
        features = pd.DataFrame(index=data.index)
        
        for indicator in self.indicators:
            name = indicator['name']
            params = indicator.get('params', {})
            
            if name == 'RSI':
                features[f'rsi_{params.get("period", 14)}'] = self._calculate_rsi(
                    data['close'], params.get('period', 14)
                )
            elif name == 'SMA':
                period = params.get('period', 20)
                features[f'sma_{period}'] = data['close'].rolling(period).mean()
            elif name == 'EMA':
                period = params.get('period', 20)
                features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            # Add more indicators as needed
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_feature_names(self) -> List[str]:
        """Get list of indicator features."""
        names = []
        for indicator in self.indicators:
            name = indicator['name']
            params = indicator.get('params', {})
            
            if name in ['RSI', 'SMA', 'EMA']:
                period = params.get('period', 14 if name == 'RSI' else 20)
                names.append(f'{name.lower()}_{period}')
        
        return names


# ============================================================================
# Model Training Components - Single Responsibility: Train models
# ============================================================================

class IModel(ABC):
    """Base interface for trained models."""
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classifiers)."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass


class IModelTrainer(ABC):
    """Interface for model trainers."""
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Type/name of the model."""
        pass
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> IModel:
        """Train a model on the data."""
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        pass
    
    @abstractmethod
    def set_hyperparameters(self, **hyperparameters) -> None:
        """Update hyperparameters."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        pass


class BaseModelTrainer(IModelTrainer):
    """Base implementation with common functionality for model trainers."""
    
    def __init__(self, **hyperparameters):
        self.hyperparameters = self._get_default_hyperparameters()
        self.hyperparameters.update(hyperparameters)
        self.model = None
        self.feature_importance_ = None
    
    @abstractmethod
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for this model type."""
        pass
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return self.hyperparameters.copy()
    
    def set_hyperparameters(self, **hyperparameters) -> None:
        """Update hyperparameters."""
        self.hyperparameters.update(hyperparameters)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        return self.feature_importance_


class LightGBMModel(IModel):
    """Wrapper for LightGBM model with standard interface."""
    
    def __init__(self, lgb_model):
        self.model = lgb_model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        import joblib
        self.model = joblib.load(path)


class LightGBMTrainer(BaseModelTrainer):
    """LightGBM model trainer."""
    
    @property
    def model_type(self) -> str:
        return "lightgbm"
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get LightGBM default hyperparameters."""
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'random_state': 42,
            'verbosity': -1
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> IModel:
        """Train LightGBM model."""
        import lightgbm as lgb
        
        # Create model
        self.model = lgb.LGBMClassifier(**self.hyperparameters)
        
        # Prepare validation data if provided
        eval_set = [(X, y)]
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return LightGBMModel(self.model)


class XGBoostModel(IModel):
    """Wrapper for XGBoost model with standard interface."""
    
    def __init__(self, xgb_model):
        self.model = xgb_model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model.load_model(str(path))


class XGBoostTrainer(BaseModelTrainer):
    """XGBoost model trainer."""
    
    @property
    def model_type(self) -> str:
        return "xgboost"
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get XGBoost default hyperparameters."""
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'random_state': 42,
            'verbosity': 0
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> IModel:
        """Train XGBoost model."""
        import xgboost as xgb
        
        # Create model
        self.model = xgb.XGBClassifier(**self.hyperparameters)
        
        # Prepare validation data if provided
        eval_set = [(X, y)]
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return XGBoostModel(self.model)


class RandomForestModel(IModel):
    """Wrapper for Random Forest model with standard interface."""
    
    def __init__(self, rf_model):
        self.model = rf_model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        import joblib
        self.model = joblib.load(path)


class RandomForestTrainer(BaseModelTrainer):
    """Random Forest model trainer."""
    
    @property
    def model_type(self) -> str:
        return "random_forest"
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get Random Forest default hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> IModel:
        """Train Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create and train model
        self.model = RandomForestClassifier(**self.hyperparameters)
        self.model.fit(X, y)
        
        # Store feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return RandomForestModel(self.model)


class NeuralNetworkModel(IModel):
    """Wrapper for Neural Network model with standard interface."""
    
    def __init__(self, nn_model, scaler=None):
        self.model = nn_model
        self.scaler = scaler
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        import torch
        
        # Scale input if scaler provided
        X_scaled = self.scaler.transform(X) if self.scaler else X.values
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float().numpy().squeeze()
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        import torch
        
        # Scale input if scaler provided
        X_scaled = self.scaler.transform(X) if self.scaler else X.values
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Predict probabilities
        self.model.eval()
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(X_tensor)).numpy()
        
        # Return probabilities for both classes
        proba = np.column_stack([1 - outputs, outputs])
        return proba
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        import torch
        import joblib
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), path.with_suffix('.pth'))
        
        # Save scaler if exists
        if self.scaler:
            joblib.dump(self.scaler, path.with_suffix('.scaler'))
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        import torch
        import joblib
        
        # Load model
        self.model.load_state_dict(torch.load(path.with_suffix('.pth')))
        
        # Load scaler if exists
        scaler_path = path.with_suffix('.scaler')
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)


class NeuralNetworkTrainer(BaseModelTrainer):
    """Neural Network model trainer using PyTorch."""
    
    @property
    def model_type(self) -> str:
        return "neural_network"
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get Neural Network default hyperparameters."""
        return {
            'hidden_layers': [64, 32],
            'activation': 'relu',
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> IModel:
        """Train Neural Network model."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create model architecture
        input_size = X.shape[1]
        hidden_layers = self.hyperparameters['hidden_layers']
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if self.hyperparameters['activation'] == 'relu' else nn.Tanh(),
                nn.Dropout(self.hyperparameters['dropout_rate'])
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))  # Binary classification
        
        model = nn.Sequential(*layers)
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.hyperparameters['learning_rate'])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
        
        # Training loop (simplified)
        model.train()
        for epoch in range(self.hyperparameters['epochs']):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        self.model = model
        return NeuralNetworkModel(model, scaler)


class ModelTrainerFactory:
    """Factory for creating model trainers."""
    
    _registry: Dict[str, Type[IModelTrainer]] = {
        'lightgbm': LightGBMTrainer,
        'xgboost': XGBoostTrainer,
        'random_forest': RandomForestTrainer,
        'neural_network': NeuralNetworkTrainer,
    }
    
    @classmethod
    def register_trainer(cls, name: str, trainer_class: Type[IModelTrainer]):
        """Register a new model trainer."""
        cls._registry[name.lower()] = trainer_class
    
    @classmethod
    def create(cls, model_type: str, **hyperparameters) -> IModelTrainer:
        """Create model trainer for specified type."""
        model_type_lower = model_type.lower()
        
        if model_type_lower not in cls._registry:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        return cls._registry[model_type_lower](**hyperparameters)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls._registry.keys())


# ============================================================================
# Orchestrator - Combines single-responsibility components
# ============================================================================

class SimplifiedPipeline:
    """
    Orchestrator that combines single-responsibility components.
    
    This class itself has a single responsibility: orchestration.
    It doesn't implement any business logic, just coordinates components.
    """
    
    def __init__(
        self,
        data_source: IDataSource,
        validators: List[IDataValidator],
        feature_calculators: List[IFeatureCalculator],
        model_trainer: IModelTrainer,
        logger: logging.Logger = None
    ):
        self.data_source = data_source
        self.validators = validators
        self.feature_calculators = feature_calculators
        self.model_trainer = model_trainer
        self.logger = logger or logging.getLogger(__name__)
    
    async def run(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> Dict[str, Any]:
        """Run the pipeline."""
        results = {}
        
        # Step 1: Load data (delegated to data source)
        self.logger.info(f"Loading data for {symbol}")
        data = await self.data_source.fetch_data(symbol, start, end)
        results['data'] = data
        
        # Step 2: Validate data (delegated to validators)
        self.logger.info("Validating data")
        for validator in self.validators:
            validation_result = validator.validate(data)
            if not validation_result.is_valid:
                raise ValueError(f"Validation failed: {validation_result.errors}")
        
        # Step 3: Calculate features (delegated to feature calculators)
        self.logger.info("Calculating features")
        features = pd.DataFrame(index=data.index)
        for calculator in self.feature_calculators:
            calculated_features = calculator.calculate(data)
            features = pd.concat([features, calculated_features], axis=1)
        results['features'] = features
        
        # Step 4: Train model (delegated to model trainer)
        self.logger.info("Training model")
        # For demo, create simple labels
        labels = (data['close'].pct_change() > 0).astype(int)
        
        model = self.model_trainer.train(features.fillna(0), labels)
        results['model'] = model
        
        return results


# ============================================================================
# Usage Example
# ============================================================================

async def example_usage():
    """Example of using modular components."""
    
    # Create single-responsibility components
    data_source = LocalDataSource("data/cache")
    
    validators = [
        SchemaValidator(
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            column_types={'volume': np.number}
        ),
        DataQualityValidator(max_null_percentage=0.05),
        TimeSeriesValidator(expected_frequency='H', max_gaps=5)
    ]
    
    feature_calculators = [
        PriceFeatureCalculator(),
        VolumeFeatureCalculator(window=20),
        TechnicalIndicatorCalculator([
            {'name': 'RSI', 'params': {'period': 14}},
            {'name': 'SMA', 'params': {'period': 20}},
            {'name': 'EMA', 'params': {'period': 12}}
        ])
    ]
    
    model_trainer = LightGBMTrainer(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100
    )
    
    # Create pipeline by composing components
    pipeline = SimplifiedPipeline(
        data_source=data_source,
        validators=validators,
        feature_calculators=feature_calculators,
        model_trainer=model_trainer
    )
    
    # Run pipeline
    results = await pipeline.run(
        symbol="BTCUSDT",
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31)
    )
    
    print("Pipeline completed successfully!")
    print(f"Features calculated: {results['features'].columns.tolist()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())