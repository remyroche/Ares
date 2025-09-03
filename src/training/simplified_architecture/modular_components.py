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


class BinanceDataSource(IDataSource):
    """Responsible ONLY for fetching data from Binance."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
    
    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from Binance."""
        # Implementation would connect to Binance API
        # This is a mock implementation
        data = pd.DataFrame({
            'timestamp': pd.date_range(start, end, freq='1H'),
            'open': np.random.randn(100) * 10 + 100,
            'high': np.random.randn(100) * 10 + 105,
            'low': np.random.randn(100) * 10 + 95,
            'close': np.random.randn(100) * 10 + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        return data.set_index('timestamp')
    
    def validate_connection(self) -> bool:
        """Check if Binance API is accessible."""
        # Would implement actual connection check
        return True


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

class IModelTrainer(ABC):
    """Interface for model trainers."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train a model on the data."""
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        pass


class LightGBMTrainer(IModelTrainer):
    """Responsible ONLY for training LightGBM models."""
    
    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self.model = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train LightGBM model."""
        import lightgbm as lgb
        
        # Default parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100
        }
        params.update(self.hyperparameters)
        
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X, y)
        
        return self.model
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return self.hyperparameters


class NeuralNetworkTrainer(IModelTrainer):
    """Responsible ONLY for training neural network models."""
    
    def __init__(self, architecture: List[int], **hyperparameters):
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.model = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train neural network model."""
        # Would implement actual NN training
        # This is a placeholder
        return f"NN with architecture {self.architecture}"
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            'architecture': self.architecture,
            **self.hyperparameters
        }


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