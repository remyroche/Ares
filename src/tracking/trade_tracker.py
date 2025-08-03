#!/usr/bin/env python3
"""
Comprehensive Trade Tracking System

This module provides detailed tracking of trades with model ensemble data,
regime analysis, feature importance, decision paths, and model behavior monitoring.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from dataclasses_json import dataclass_json

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


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
    probability: Dict[str, float]
    features_used: List[str]
    feature_importance: List[FeatureImportance]
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
    individual_predictions: List[ModelPrediction]
    ensemble_weights: Dict[str, float]
    meta_learner_prediction: Optional[str] = None
    meta_learner_confidence: Optional[float] = None


@dataclass_json
@dataclass
class RegimeAnalysis:
    """Market regime analysis tracking."""
    regime_type: str
    regime_confidence: float
    regime_probabilities: Dict[str, float]
    regime_features: List[str]
    regime_indicators: Dict[str, float]
    regime_transition_probability: float
    regime_duration: Optional[int] = None


@dataclass_json
@dataclass
class DecisionPath:
    """Decision path analysis tracking."""
    decision_steps: List[str]
    decision_reasons: List[str]
    decision_weights: List[float]
    decision_thresholds: Dict[str, float]
    decision_metadata: Dict[str, Any]


@dataclass_json
@dataclass
class ModelBehavior:
    """Model behavior monitoring."""
    model_type: str
    prediction_consistency: float
    confidence_trend: List[float]
    feature_importance_stability: float
    prediction_drift: float
    model_performance_metrics: Dict[str, float]
    last_retraining: Optional[datetime] = None


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
    model_behaviors: List[ModelBehavior]
    
    # Additional metadata
    market_conditions: Dict[str, Any]
    risk_metrics: Dict[str, float]
    execution_metadata: Dict[str, Any]
    
    # Optional fields
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: Optional[float] = None
    close_timestamp: Optional[datetime] = None
    close_price: Optional[float] = None
    close_reason: Optional[str] = None


class TradeTracker:
    """
    Comprehensive trade tracking system with model ensemble, regime analysis,
    feature importance, decision path, and model behavior monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trade tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("TradeTracker")
        
        # Storage
        self.trades: Dict[str, TradeRecord] = {}
        self.trade_history: List[TradeRecord] = []
        self.model_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Configuration
        self.tracking_config = config.get("trade_tracking", {})
        self.enable_feature_importance_tracking = self.tracking_config.get(
            "enable_feature_importance_tracking", True
        )
        self.enable_decision_path_tracking = self.tracking_config.get(
            "enable_decision_path_tracking", True
        )
        self.enable_model_behavior_tracking = self.tracking_config.get(
            "enable_model_behavior_tracking", True
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
            ValueError: (False, "Invalid trade data"),
            KeyError: (False, "Missing required trade fields"),
        },
        default_return=False,
        context="trade recording",
    )
    async def record_trade(
        self,
        trade_data: Dict[str, Any],
        ensemble_decision: Dict[str, Any],
        regime_analysis: Dict[str, Any],
        decision_path: Dict[str, Any],
        model_behaviors: List[Dict[str, Any]],
    ) -> bool:
        """
        Record a comprehensive trade with all tracking data.
        
        Args:
            trade_data: Basic trade information
            ensemble_decision: Model ensemble decision data
            regime_analysis: Market regime analysis
            decision_path: Decision path analysis
            model_behaviors: Model behavior monitoring data
            
        Returns:
            bool: True if recording successful
        """
        try:
            # Generate trade ID
            trade_id = f"trade_{int(time.time() * 1000)}"
            
            # Create trade record
            trade_record = TradeRecord(
                trade_id=trade_id,
                symbol=trade_data["symbol"],
                side=trade_data["side"],
                quantity=trade_data["quantity"],
                price=trade_data["price"],
                timestamp=datetime.fromisoformat(trade_data["timestamp"]),
                status=TradeStatus(trade_data["status"]),
                order_type=trade_data["order_type"],
                
                # Model ensemble data
                ensemble_decision=EnsembleDecision(**ensemble_decision),
                
                # Regime analysis
                regime_analysis=RegimeAnalysis(**regime_analysis),
                
                # Decision path
                decision_path=DecisionPath(**decision_path),
                
                # Model behaviors
                model_behaviors=[ModelBehavior(**mb) for mb in model_behaviors],
                
                # Additional metadata
                market_conditions=trade_data.get("market_conditions", {}),
                risk_metrics=trade_data.get("risk_metrics", {}),
                execution_metadata=trade_data.get("execution_metadata", {}),
                
                # Optional fields
                stop_loss=trade_data.get("stop_loss"),
                take_profit=trade_data.get("take_profit"),
            )
            
            # Store trade
            self.trades[trade_id] = trade_record
            self.trade_history.append(trade_record)
            
            # Update performance metrics
            self._update_performance_metrics(trade_record)
            
            # Track model performance
            await self._track_model_performance(trade_record)
            
            # Log trade
            self.logger.info(f"📊 Trade recorded: {trade_id} - {trade_record.symbol} {trade_record.side} @ {trade_record.price}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to record trade: {e}")
            return False

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
            self.performance_metrics["win_rate"] = winning_trades / total_trades if total_trades > 0 else 0.0

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
    async def update_trade(
        self,
        trade_id: str,
        update_data: Dict[str, Any],
    ) -> bool:
        """
        Update an existing trade record.
        
        Args:
            trade_id: Trade ID to update
            update_data: Data to update
            
        Returns:
            bool: True if update successful
        """
        try:
            if trade_id not in self.trades:
                self.logger.error(f"Trade {trade_id} not found")
                return False
            
            trade_record = self.trades[trade_id]
            
            # Update fields
            for key, value in update_data.items():
                if hasattr(trade_record, key):
                    setattr(trade_record, key, value)
            
            # Update performance metrics if PnL changed
            if "pnl" in update_data:
                self._update_performance_metrics(trade_record)
            
            self.logger.info(f"📝 Trade {trade_id} updated")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update trade {trade_id}: {e}")
            return False

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Get a specific trade record."""
        return self.trades.get(trade_id)

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[TradeRecord]:
        """
        Get trade history with optional filtering.
        
        Args:
            symbol: Filter by symbol
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Limit number of results
            
        Returns:
            List of trade records
        """
        filtered_trades = self.trade_history
        
        # Apply filters
        if symbol:
            filtered_trades = [t for t in filtered_trades if t.symbol == symbol]
        
        if start_time:
            filtered_trades = [t for t in filtered_trades if t.timestamp >= start_time]
        
        if end_time:
            filtered_trades = [t for t in filtered_trades if t.timestamp <= end_time]
        
        # Apply limit
        if limit:
            filtered_trades = filtered_trades[-limit:]
        
        return filtered_trades

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get model performance summary."""
        summary = {}
        
        for model_type, history in self.model_performance_history.items():
            if not history:
                continue
            
            # Calculate average metrics
            avg_consistency = np.mean([h["prediction_consistency"] for h in history])
            avg_stability = np.mean([h["feature_importance_stability"] for h in history])
            avg_drift = np.mean([h["prediction_drift"] for h in history])
            
            summary[model_type] = {
                "total_predictions": len(history),
                "avg_prediction_consistency": avg_consistency,
                "avg_feature_importance_stability": avg_stability,
                "avg_prediction_drift": avg_drift,
                "last_prediction": history[-1]["timestamp"] if history else None,
            }
        
        return summary

    def get_feature_importance_analysis(
        self,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze feature importance across trades.
        
        Args:
            model_type: Filter by model type
            timeframe: Filter by timeframe
            regime: Filter by regime
            
        Returns:
            Feature importance analysis
        """
        feature_importance_data = []
        
        for trade in self.trade_history:
            for model_behavior in trade.model_behaviors:
                if model_type and model_behavior.model_type != model_type:
                    continue
                
                # Extract feature importance from ensemble decision
                for prediction in trade.ensemble_decision.individual_predictions:
                    if model_type and prediction.model_type != model_type:
                        continue
                    
                    for feature_imp in prediction.feature_importance:
                        if timeframe and feature_imp.timeframe != timeframe:
                            continue
                        if regime and feature_imp.regime != regime:
                            continue
                        
                        feature_importance_data.append({
                            "trade_id": trade.trade_id,
                            "timestamp": trade.timestamp,
                            "model_type": feature_imp.model_type,
                            "timeframe": feature_imp.timeframe,
                            "regime": feature_imp.regime,
                            "feature_name": feature_imp.feature_name,
                            "importance_score": feature_imp.importance_score,
                            "importance_rank": feature_imp.importance_rank,
                        })
        
        # Analyze feature importance
        if not feature_importance_data:
            return {}
        
        df = pd.DataFrame(feature_importance_data)
        
        analysis = {
            "total_features_tracked": len(df["feature_name"].unique()),
            "top_features": df.groupby("feature_name")["importance_score"].mean().nlargest(10).to_dict(),
            "feature_stability": df.groupby("feature_name")["importance_score"].std().to_dict(),
            "model_performance_by_feature": {},
        }
        
        # Analyze by model type
        for model in df["model_type"].unique():
            model_data = df[df["model_type"] == model]
            analysis["model_performance_by_feature"][model] = {
                "top_features": model_data.groupby("feature_name")["importance_score"].mean().nlargest(5).to_dict(),
                "feature_count": len(model_data["feature_name"].unique()),
            }
        
        return analysis

    def get_decision_path_analysis(self) -> Dict[str, Any]:
        """Analyze decision paths across trades."""
        decision_paths = [trade.decision_path for trade in self.trade_history]
        
        if not decision_paths:
            return {}
        
        analysis = {
            "total_decisions": len(decision_paths),
            "common_decision_steps": {},
            "decision_weights_distribution": {},
            "decision_thresholds_analysis": {},
        }
        
        # Analyze common decision steps
        all_steps = []
        for dp in decision_paths:
            all_steps.extend(dp.decision_steps)
        
        step_counts = pd.Series(all_steps).value_counts()
        analysis["common_decision_steps"] = step_counts.head(10).to_dict()
        
        # Analyze decision weights
        all_weights = []
        for dp in decision_paths:
            all_weights.extend(dp.decision_weights)
        
        if all_weights:
            analysis["decision_weights_distribution"] = {
                "mean": np.mean(all_weights),
                "std": np.std(all_weights),
                "min": np.min(all_weights),
                "max": np.max(all_weights),
            }
        
        return analysis

    def get_regime_analysis_summary(self) -> Dict[str, Any]:
        """Get regime analysis summary."""
        regime_data = [trade.regime_analysis for trade in self.trade_history]
        
        if not regime_data:
            return {}
        
        analysis = {
            "total_regime_analyses": len(regime_data),
            "regime_distribution": {},
            "regime_confidence_stats": {},
            "regime_transition_analysis": {},
        }
        
        # Analyze regime distribution
        regime_types = [ra.regime_type for ra in regime_data]
        regime_counts = pd.Series(regime_types).value_counts()
        analysis["regime_distribution"] = regime_counts.to_dict()
        
        # Analyze confidence
        confidences = [ra.regime_confidence for ra in regime_data]
        analysis["regime_confidence_stats"] = {
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
        }
        
        # Analyze transitions
        transition_probs = [ra.regime_transition_probability for ra in regime_data if ra.regime_transition_probability is not None]
        if transition_probs:
            analysis["regime_transition_analysis"] = {
                "mean_transition_probability": np.mean(transition_probs),
                "high_transition_periods": len([p for p in transition_probs if p > 0.5]),
            }
        
        return analysis

    def export_trade_data(
        self,
        format: str = "json",
        filepath: Optional[str] = None,
    ) -> str:
        """
        Export trade data to file.
        
        Args:
            format: Export format ("json", "csv")
            filepath: Output file path
            
        Returns:
            str: File path
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"trade_data_{timestamp}.{format}"
        
        if format == "json":
            # Convert to JSON-serializable format
            export_data = {
                "trades": [asdict(trade) for trade in self.trade_history],
                "performance_metrics": self.performance_metrics,
                "model_performance": self.model_performance_history,
            }
            
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == "csv":
            # Export as CSV
            trade_data = []
            for trade in self.trade_history:
                trade_dict = asdict(trade)
                # Flatten nested structures for CSV
                trade_data.append(self._flatten_trade_dict(trade_dict))
            
            df = pd.DataFrame(trade_data)
            df.to_csv(filepath, index=False)
        
        self.logger.info(f"📊 Trade data exported to {filepath}")
        return filepath

    def _flatten_trade_dict(self, trade_dict: Dict[str, Any]) -> Dict[str, Any]:
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

    async def cleanup_old_records(self, max_age_days: int = 30) -> int:
        """
        Clean up old trade records.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            int: Number of records cleaned up
        """
        cutoff_time = datetime.now() - pd.Timedelta(days=max_age_days)
        
        # Filter out old records
        old_count = len(self.trade_history)
        self.trade_history = [
            trade for trade in self.trade_history
            if trade.timestamp > cutoff_time
        ]
        new_count = len(self.trade_history)
        
        cleaned_count = old_count - new_count
        
        if cleaned_count > 0:
            self.logger.info(f"🧹 Cleaned up {cleaned_count} old trade records")
        
        return cleaned_count 