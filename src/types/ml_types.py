# src/types/ml_types.py

"""
Machine learning type definitions for model inputs, outputs, and metrics.
"""

from typing import Dict, List, Literal, Optional, TypedDict, Union

import numpy as np
import pandas as pd

from .base_types import ConfidenceLevel, ModelId, Score, Symbol, Timestamp

# Type aliases for ML data structures
FeatureArray = np.ndarray
TargetArray = np.ndarray
PredictionArray = np.ndarray

# Feature engineering types
class FeatureDict(TypedDict, total=False):
    """Type-safe feature dictionary."""
    technical_indicators: Dict[str, float]
    market_microstructure: Dict[str, float]
    sentiment_features: Dict[str, float]
    regime_features: Dict[str, float]
    volatility_features: Dict[str, float]
    volume_features: Dict[str, float]


class ModelInput(TypedDict):
    """Type-safe model input structure."""
    features: FeatureArray
    timestamps: List[Timestamp]
    symbols: List[Symbol]
    metadata: Dict[str, Union[str, int, float]]


class PredictionResult(TypedDict):
    """Type-safe prediction result."""
    prediction: Union[float, int, List[float]]
    confidence: ConfidenceLevel
    probabilities: Optional[List[float]]
    feature_importance: Optional[Dict[str, float]]
    model_id: ModelId
    timestamp: Timestamp


class ModelOutput(TypedDict):
    """Type-safe model output structure."""
    predictions: List[PredictionResult]
    model_metadata: Dict[str, Union[str, int, float]]
    processing_time_ms: float


class ModelMetrics(TypedDict):
    """Type-safe model performance metrics."""
    accuracy: Score
    precision: Score
    recall: Score
    f1_score: Score
    auc_roc: Optional[Score]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[Score]
    profit_factor: Optional[float]


class TrainingData(TypedDict):
    """Type-safe training data structure."""
    X_train: FeatureArray
    y_train: TargetArray
    X_val: FeatureArray
    y_val: TargetArray
    feature_names: List[str]
    target_name: str
    data_split_info: Dict[str, Union[str, int, float]]


class ValidationData(TypedDict):
    """Type-safe validation data structure."""
    X_test: FeatureArray
    y_test: TargetArray
    predictions: PredictionArray
    metrics: ModelMetrics
    validation_timestamp: Timestamp


class ModelConfig(TypedDict, total=False):
    """Type-safe model configuration."""
    model_type: Literal["classification", "regression", "time_series"]
    algorithm: Literal["xgboost", "lightgbm", "neural_network", "ensemble"]
    hyperparameters: Dict[str, Union[int, float, str, bool]]
    feature_selection: Dict[str, Union[bool, int, float]]
    preprocessing: Dict[str, Union[bool, str, List[str]]]


class EnsembleConfig(TypedDict):
    """Type-safe ensemble configuration."""
    ensemble_method: Literal["voting", "stacking", "blending", "boosting"]
    base_models: List[ModelConfig]
    meta_model: Optional[ModelConfig]
    weights: Optional[List[float]]
    cross_validation_folds: int


# Regime and market state types
class RegimeClassification(TypedDict):
    """Type-safe regime classification result."""
    regime: Literal["bullish", "bearish", "sideways", "volatile", "trending"]
    confidence: ConfidenceLevel
    regime_probabilities: Dict[str, float]
    features_used: List[str]
    timestamp: Timestamp


class MarketState(TypedDict):
    """Type-safe market state information."""
    regime: RegimeClassification
    volatility_level: Literal["low", "medium", "high", "extreme"]
    trend_direction: Literal["up", "down", "sideways"]
    momentum_score: Score
    support_resistance: Dict[str, float]