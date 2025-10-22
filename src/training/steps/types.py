"""
Type definitions for training steps.

This module provides comprehensive type definitions for all training step components,
ensuring type safety and better IDE support throughout the codebase.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Protocol, TypedDict, Literal
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================================
# BASIC TYPE ALIASES
# ============================================================================

# Data types
DataFrameType = pd.DataFrame
SeriesType = pd.Series
NDArrayType = np.ndarray
PathType = Union[str, Path]

# Configuration types
ConfigDict = Dict[str, Any]
MetadataDict = Dict[str, Any]
MetricsDict = Dict[str, Union[int, float, str, bool]]

# Execution types
ExecutionMode = Literal['light', 'full', 'blank']
SignalType = Literal['long', 'short', 'both']
ModelType = Literal['analyst', 'tactician', 'ensemble']
DirectionType = Literal['longs', 'shorts', 'both']

# ============================================================================
# TYPED DICTIONARIES
# ============================================================================

class StepConfig(TypedDict, total=False):
    """Configuration for training steps."""
    symbol: str
    exchange: str
    timeframe: str
    execution_mode: ExecutionMode
    lookback_years: int
    direction: DirectionType
    model: ModelType
    information: Optional[str]
    initial_capital: float
    commission: float
    slippage: float
    strategy_config: Optional[ConfigDict]
    risk_config: Optional[ConfigDict]

class ExecutionResult(TypedDict, total=False):
    """Result of step execution."""
    success: bool
    artifacts: List[str]
    metrics: MetricsDict
    error: Optional[str]
    execution_time: float
    data: Optional[DataFrameType]
    model: Optional[Any]
    predictions: Optional[DataFrameType]

class ValidationResult(TypedDict, total=False):
    """Data validation result."""
    is_valid: bool
    quality_score: float
    errors: List[str]
    warnings: List[str]
    metadata: MetadataDict

class FeatureSelectionResult(TypedDict, total=False):
    """Feature selection result."""
    success: bool
    selected_features: List[str]
    feature_scores: Dict[str, float]
    n_features: int
    signal_type: SignalType
    error: Optional[str]

class ModelTrainingResult(TypedDict, total=False):
    """Model training result."""
    success: bool
    model: Optional[Any]
    accuracy: float
    model_type: str
    signal_type: SignalType
    model_key: str
    error: Optional[str]
    training_metrics: MetricsDict

class DataLoadResult(TypedDict, total=False):
    """Data loading result."""
    success: bool
    data: Optional[DataFrameType]
    error: Optional[str]
    source: str
    rows: int
    columns: int

# ============================================================================
# PROTOCOLS
# ============================================================================

class DataProcessor(Protocol):
    """Protocol for data processing objects."""
    def process(self, data: DataFrameType) -> DataFrameType: ...
    def validate(self, data: DataFrameType) -> ValidationResult: ...

class ModelTrainer(Protocol):
    """Protocol for model training objects."""
    async def train(self, X: DataFrameType, y: SeriesType, config: StepConfig) -> ModelTrainingResult: ...
    def save_model(self, model: Any, path: PathType) -> bool: ...
    def load_model(self, path: PathType) -> Optional[Any]: ...

class FeatureSelector(Protocol):
    """Protocol for feature selection objects."""
    async def select_features(self, X: DataFrameType, y: SeriesType, config: StepConfig) -> FeatureSelectionResult: ...
    def get_feature_importance(self, X: DataFrameType, y: SeriesType) -> Dict[str, float]: ...

class DataLoader(Protocol):
    """Protocol for data loading objects."""
    async def load_data(self, config: StepConfig) -> DataLoadResult: ...
    def validate_data(self, data: DataFrameType) -> ValidationResult: ...

# ============================================================================
# EXCEPTION CLASSES
# ============================================================================

class TrainingStepError(Exception):
    """Base exception for training step errors."""
    pass

class ValidationError(TrainingStepError):
    """Exception raised when data validation fails."""
    pass

class DataLoadError(TrainingStepError):
    """Exception raised when data loading fails."""
    pass

class ModelTrainingError(TrainingStepError):
    """Exception raised when model training fails."""
    pass

class FeatureSelectionError(TrainingStepError):
    """Exception raised when feature selection fails."""
    pass

class ConfigurationError(TrainingStepError):
    """Exception raised when configuration is invalid."""
    pass

class ArtifactError(TrainingStepError):
    """Exception raised when artifact operations fail."""
    pass

# ============================================================================
# UTILITY TYPE FUNCTIONS
# ============================================================================

def validate_config(config: Dict[str, Any]) -> StepConfig:
    """Validate and convert configuration dictionary to StepConfig."""
    required_fields = ['symbol', 'exchange']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ConfigurationError(f"Missing required configuration fields: {missing_fields}")
    
    # Set defaults
    validated_config: StepConfig = {
        'symbol': str(config['symbol']).upper(),
        'exchange': str(config['exchange']).lower(),
        'timeframe': config.get('timeframe', '15m'),
        'execution_mode': config.get('execution_mode', 'light'),
        'lookback_years': config.get('lookback_years', 2),
        'direction': config.get('direction', 'longs'),
        'model': config.get('model', 'analyst'),
        'information': config.get('information'),
        'initial_capital': float(config.get('initial_capital', 100000.0)),
        'commission': float(config.get('commission', 0.001)),
        'slippage': float(config.get('slippage', 0.0005)),
        'strategy_config': config.get('strategy_config'),
        'risk_config': config.get('risk_config')
    }
    
    # Validate execution mode
    if validated_config['execution_mode'] not in ['light', 'full', 'blank']:
        raise ConfigurationError(f"Invalid execution_mode: {validated_config['execution_mode']}")
    
    # Validate direction
    if validated_config['direction'] not in ['longs', 'shorts', 'both']:
        raise ConfigurationError(f"Invalid direction: {validated_config['direction']}")
    
    # Validate model type
    if validated_config['model'] not in ['analyst', 'tactician', 'ensemble']:
        raise ConfigurationError(f"Invalid model: {validated_config['model']}")
    
    return validated_config

def create_error_result(error: Exception, context: str = "") -> ExecutionResult:
    """Create a standardized error result."""
    return ExecutionResult(
        success=False,
        error=f"{type(error).__name__}: {str(error)}" + (f" (Context: {context})" if context else ""),
        artifacts=[],
        metrics={},
        execution_time=0.0
    )

def create_success_result(artifacts: List[str] = None, metrics: MetricsDict = None, **kwargs) -> ExecutionResult:
    """Create a standardized success result."""
    return ExecutionResult(
        success=True,
        artifacts=artifacts or [],
        metrics=metrics or {},
        execution_time=0.0,
        **kwargs
    )

# ============================================================================
# TYPE GUARDS
# ============================================================================

def is_dataframe(data: Any) -> bool:
    """Check if data is a pandas DataFrame."""
    return isinstance(data, pd.DataFrame)

def is_series(data: Any) -> bool:
    """Check if data is a pandas Series."""
    return isinstance(data, pd.Series)

def is_valid_config(config: Any) -> bool:
    """Check if config is a valid configuration dictionary."""
    return isinstance(config, dict) and 'symbol' in config and 'exchange' in config

def is_execution_result(result: Any) -> bool:
    """Check if result is a valid execution result."""
    return (isinstance(result, dict) and 
            'success' in result and 
            isinstance(result['success'], bool))

# ============================================================================
# CONSTANTS
# ============================================================================

# Default values
DEFAULT_TIMEFRAME = '15m'
DEFAULT_EXECUTION_MODE: ExecutionMode = 'light'
DEFAULT_LOOKBACK_YEARS = 2
DEFAULT_DIRECTION: DirectionType = 'longs'
DEFAULT_MODEL: ModelType = 'analyst'

# Required columns for different data types
REQUIRED_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
REQUIRED_TIMESTAMP_COLUMNS = ['timestamp']
REQUIRED_FEATURE_COLUMNS = ['feature_1', 'feature_2']  # Minimum for feature data

# File extensions
PARQUET_EXTENSION = '.parquet'
CSV_EXTENSION = '.csv'
JSON_EXTENSION = '.json'
PKL_EXTENSION = '.pkl'

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Type aliases
    'DataFrameType', 'SeriesType', 'NDArrayType', 'PathType',
    'ConfigDict', 'MetadataDict', 'MetricsDict',
    'ExecutionMode', 'SignalType', 'ModelType', 'DirectionType',
    
    # TypedDict classes
    'StepConfig', 'ExecutionResult', 'ValidationResult',
    'FeatureSelectionResult', 'ModelTrainingResult', 'DataLoadResult',
    
    # Protocols
    'DataProcessor', 'ModelTrainer', 'FeatureSelector', 'DataLoader',
    
    # Exception classes
    'TrainingStepError', 'ValidationError', 'DataLoadError',
    'ModelTrainingError', 'FeatureSelectionError', 'ConfigurationError', 'ArtifactError',
    
    # Utility functions
    'validate_config', 'create_error_result', 'create_success_result',
    'is_dataframe', 'is_series', 'is_valid_config', 'is_execution_result',
    
    # Constants
    'DEFAULT_TIMEFRAME', 'DEFAULT_EXECUTION_MODE', 'DEFAULT_LOOKBACK_YEARS',
    'DEFAULT_DIRECTION', 'DEFAULT_MODEL',
    'REQUIRED_OHLCV_COLUMNS', 'REQUIRED_TIMESTAMP_COLUMNS', 'REQUIRED_FEATURE_COLUMNS',
    'PARQUET_EXTENSION', 'CSV_EXTENSION', 'JSON_EXTENSION', 'PKL_EXTENSION'
]