# src/types/ml_types.py

"""
Machine learning type definitions for model inputs, outputs, and metrics.
"""

from typing import Literal, TypedDict

import numpy as np

from .base_types import ConfidenceLevel, ModelId, Score, Symbol, Timestamp

# Type aliases for ML data structures
FeatureArray = np.ndarray
TargetArray = np.ndarray
PredictionArray = np.ndarray


# Feature engineering types
class FeatureDict(TypedDict, total=False):
    """Type-safe feature dictionary."""

    technical_indicators: dict[str, float]
    market_microstructure: dict[str, float]
    sentiment_features: dict[str, float]
    regime_features: dict[str, float]
    volatility_features: dict[str, float]
    volume_features: dict[str, float]


class ModelInput(TypedDict):
    """Type-safe model input structure."""

    features: FeatureArray
    timestamps: list[Timestamp]
    symbols: list[Symbol]
    metadata: dict[str, str | int | float]


class PredictionResult(TypedDict):
    """Type-safe prediction result."""

    prediction: float | int | list[float]
    confidence: ConfidenceLevel
    probabilities: list[float] | None
    feature_importance: dict[str, float] | None
    model_id: ModelId
    timestamp: Timestamp


class ModelOutput(TypedDict):
    """Type-safe model output structure."""

    predictions: list[PredictionResult]
    model_metadata: dict[str, str | int | float]
    processing_time_ms: float


class ModelMetrics(TypedDict):
    """Type-safe model performance metrics."""

    accuracy: Score
    precision: Score
    recall: Score
    f1_score: Score
    auc_roc: Score | None
    sharpe_ratio: float | None
    max_drawdown: float | None
    win_rate: Score | None
    profit_factor: float | None


class TrainingData(TypedDict):
    """Type-safe training data structure."""

    X_train: FeatureArray
    y_train: TargetArray
    X_val: FeatureArray
    y_val: TargetArray
    feature_names: list[str]
    target_name: str
    data_split_info: dict[str, str | int | float]


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
    hyperparameters: dict[str, int | float | str | bool]
    feature_selection: dict[str, bool | int | float]
    preprocessing: dict[str, bool | str | list[str]]


class EnsembleConfig(TypedDict):
    """Type-safe ensemble configuration."""

    ensemble_method: Literal["voting", "stacking", "blending", "boosting"]
    base_models: list[ModelConfig]
    meta_model: ModelConfig | None
    weights: list[float] | None
    cross_validation_folds: int


# Regime and market state types
class RegimeClassification(TypedDict):
    """Type-safe regime classification result."""

    regime: Literal["bullish", "bearish", "sideways", "volatile", "trending"]
    confidence: ConfidenceLevel
    regime_probabilities: dict[str, float]
    features_used: list[str]
    timestamp: Timestamp


class MarketState(TypedDict):
    """Type-safe market state information."""

    regime: RegimeClassification
    volatility_level: Literal["low", "medium", "high", "extreme"]
    trend_direction: Literal["up", "down", "sideways"]
    momentum_score: Score
    support_resistance: dict[str, float]
