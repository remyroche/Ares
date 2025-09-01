#!/usr/bin/env python3
"""
Comprehensive Trade Tracking System

This module provides detailed tracking of trades with model ensemble data = regime analysis, feature importance, decision paths, and model behavior monitoring.
"""

from dataclasses_json import dataclass_json
from datetime import datetime
from src.utils.logger import system_logger
from typing import Any
import json
import time

from dataclasses import asdict, dataclass
from enum import Enum
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import failed, missing
import numpy as np
import pandas as pd

class TradeStatus(Enum):
    """Trade status enumeration."""

    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class ModelType(Enum):
    """Model type enumeration."""

    XGBOOST = "xgboost"
    LSTM = "lstm"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"
    META_LEARNER = "meta_learner"

@dataclass_json
@dataclass

class FeatureImportance:
    """Feature importance tracking."""

    feature_name: str
    importance_score: float
    importance_rank: int
    model_type: str
    timeframe: str
    regime: str

@dataclass_json
@dataclass

class ModelPrediction:
    """Individual model prediction tracking."""

    model_type: str
    model_id: str
    prediction: str  # "buy", "sell", "hold"
    confidence: float
    probability: dict[str, float]
    features_used: list[str]
    feature_importance: list[FeatureImportance]
    prediction_time: datetime
    model_version: str

@dataclass_json
@dataclass

class EnsembleDecision:
    """Ensemble decision tracking."""

    ensemble_id: str
    ensemble_type: str
    primary_prediction: str
    primary_confidence: float
    individual_predictions: list[ModelPrediction]
    ensemble_weights: dict[str, float]
    meta_learner_prediction: str | None = None
    meta_learner_confidence: float | None = None

@dataclass_json
@dataclass

class RegimeAnalysis:
    """Market regime analysis tracking."""

    regime_type: str
    regime_confidence: float
    regime_probabilities: dict[str, float]
    regime_features: list[str]
    regime_indicators: dict[str, float]
    regime_transition_probability: float
    regime_duration: int | None = None

@dataclass_json
@dataclass

class DecisionPath:
    """Decision path analysis tracking."""

    decision_steps: list[str]
    decision_reasons: list[str]
    decision_weights: list[float]
    decision_thresholds: dict[str, float]
    decision_metadata: dict[str, Any]

@dataclass_json
@dataclass

class ModelBehavior:
    """Model behavior monitoring."""

    model_type: str
    prediction_consistency: float
    confidence_trend: list[float]
    feature_importance_stability: float
    prediction_drift: float
    model_performance_metrics: dict[str, float]
    last_retraining: datetime | None = None

@dataclass_json
@dataclass

class TradeRecord:
    """Comprehensive trade record."""

    trade_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    timestamp: datetime
    status: TradeStatus
    order_type: str

    # Model ensemble data
    ensemble_decision: EnsembleDecision

    # Regime analysis
    regime_analysis: RegimeAnalysis

    # Decision path
    decision_path: DecisionPath

    # Model behavior
    model_behaviors: list[ModelBehavior]

    # Additional metadata
    market_conditions: dict[str, Any]
    risk_metrics: dict[str, float]
    execution_metadata: dict[str, Any]

    # Optional fields
    stop_loss: float | None = None
    take_profit: float | None = None
    pnl: float | None = None
    close_timestamp: datetime | None = None
    close_price: float | None = None
    close_reason: str | None = None

class TradeTracker:
    """
    Comprehensive trade tracking system with model ensemble = regime analysis,
    feature importance, decision path, and model behavior monitoring.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize trade tracker.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("TradeTracker")

        # Storage
        self.trades: dict[str, TradeRecord] = {}
        self.trade_history: list[TradeRecord] = []
        self.model_performance_history: dict[str, list[dict[str, Any]]] = {}

        # Configuration
        self.tracking_config = config.get("trade_tracking", {})
        self.enable_feature_importance_tracking = self.tracking_config.get(
            "enable_feature_importance_tracking",
            True,
        )
        self.enable_decision_path_tracking = self.tracking_config.get(
            "enable_decision_path_tracking",
            True,
        )
        self.enable_model_behavior_tracking = self.tracking_config.get(
            "enable_model_behavior_tracking",
            True,
        )
        self.max_history_size = self.tracking_config.get("max_history_size", 10000)

        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
        }

        self.logger.info("🚀 Trade Tracker initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: ("Invalid trade data", False),
            KeyError: ("Missing required trade fields", False),
        },
        default_return=False,
        context="trade recording",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance metrics update",
    )
    def _update_performance_metrics(self, trade_record: TradeRecord) -> None:
        """Update performance metrics with new trade."""
        self.performance_metrics["total_trades"] += 1

        # Update PnL if trade is closed
        if trade_record.pnl is not None:
            self.performance_metrics["total_pnl"] += trade_record.pnl

            if trade_record.pnl > 0:
                self.performance_metrics["winning_trades"] += 1
            else:
                self.performance_metrics["losing_trades"] += 1

            # Update win rate
            total_trades = self.performance_metrics["total_trades"]
            winning_trades = self.performance_metrics["winning_trades"]
            self.performance_metrics["win_rate"] = (
                winning_trades / total_trades if total_trades > 0 else 0.0
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model performance tracking",
    )
    async def _track_model_performance(self, trade_record: TradeRecord) -> None:
        """Track individual model performance."""
        for model_behavior in trade_record.model_behaviors:
            model_type = model_behavior.model_type

            if model_type not in self.model_performance_history:
                self.model_performance_history[model_type] = []

            # Record model performance
            performance_record = {
                "timestamp": trade_record.timestamp,
                "trade_id": trade_record.trade_id,
                "prediction_consistency": model_behavior.prediction_consistency,
                "confidence_trend": model_behavior.confidence_trend,
                "feature_importance_stability": model_behavior.feature_importance_stability,
                "prediction_drift": model_behavior.prediction_drift,
                "performance_metrics": model_behavior.model_performance_metrics,
            }

            self.model_performance_history[model_type].append(performance_record)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trade update",
    )
    def _flatten_trade_dict(self, trade_dict: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested trade dictionary for CSV export."""
        flattened = {}

        for key, value in trade_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, list):
                flattened[f"{key}_count"] = len(value)
            else:
                flattened[key] = value

        return flattened
