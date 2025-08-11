#!/usr/bin/env python3
"""
Trade Conditions Monitor

This module provides comprehensive monitoring for trade conditions, decisions, and execution details.
It captures what we did, when (time, regime, S/R levels), why (model predictions),
and provides in-depth multi-timeframe analysis for ML model improvement.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
)


class TradeAction(Enum):
    """Trade action types."""

    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    CANCEL_ORDER = "cancel_order"


class RegimeType(Enum):
    """Market regime types."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class SupportResistanceType(Enum):
    """Support/Resistance level types."""

    SUPPORT = "support"
    RESISTANCE = "resistance"
    PIVOT = "pivot"
    FIBONACCI = "fibonacci"
    VOLUME_LEVEL = "volume_level"


@dataclass
class MultiTimeframeFeatures:
    """Multi-timeframe feature analysis."""

    timeframe: str  # "30m", "15m", "5m", "1m"
    timestamp: datetime
    price: float
    volume: float

    # Technical indicators
    rsi: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    bollinger_position: float | None = None

    # Volume indicators
    volume_sma: float | None = None
    volume_ratio: float | None = None
    vwap: float | None = None

    # Volatility indicators
    atr: float | None = None
    volatility: float | None = None

    # Trend indicators
    ema_9: float | None = None
    ema_21: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    trend_strength: float | None = None

    # Momentum indicators
    momentum: float | None = None
    rate_of_change: float | None = None

    # Custom features
    regime_probability: dict[str, float] = field(default_factory=dict)
    market_microstructure: dict[str, float] = field(default_factory=dict)
    liquidity_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SupportResistanceLevel:
    """Support/Resistance level information."""

    level_type: SupportResistanceType
    price: float
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    distance_from_current: float  # Percentage distance
    touch_count: int
    last_touch_time: datetime | None = None
    volume_at_level: float | None = None
    breakout_probability: float | None = None


@dataclass
class ModelPrediction:
    """Individual model prediction details."""

    model_id: str
    model_type: str  # "xgboost", "catboost", "neural_network", etc.
    ensemble_type: str  # "bear_trend", "bull_trend", "sideways", etc.
    prediction: float
    confidence: float
    probability_distribution: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    prediction_reasoning: str = ""
    model_version: str = ""
    last_training_date: datetime | None = None


@dataclass
class EnsemblePrediction:
    """Ensemble prediction aggregation."""

    ensemble_id: str
    regime_type: RegimeType
    individual_predictions: list[ModelPrediction]
    aggregated_prediction: float
    ensemble_confidence: float
    consensus_level: float  # How much models agree
    disagreement_score: float
    weighted_average: float
    voting_result: str
    meta_learner_prediction: float | None = None
    meta_learner_confidence: float | None = None


@dataclass
class TradeDecisionContext:
    """Complete context for a trade decision."""

    decision_id: str
    timestamp: datetime
    symbol: str
    exchange: str

    # Market conditions
    current_price: float
    current_regime: RegimeType
    regime_confidence: float
    regime_duration_minutes: float

    # Support/Resistance analysis
    nearby_sr_levels: list[SupportResistanceLevel]
    closest_support: SupportResistanceLevel | None = None
    closest_resistance: SupportResistanceLevel | None = None
    sr_zone_strength: float = 0.0

    # Multi-timeframe analysis
    timeframe_features: dict[str, MultiTimeframeFeatures] = field(default_factory=dict)

    # ML Model predictions
    ensemble_predictions: list[EnsemblePrediction]
    final_prediction: float
    final_confidence: float

    # Trade decision
    recommended_action: TradeAction
    position_size: float | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_reward_ratio: float | None = None

    # Risk assessment
    risk_score: float
    max_position_risk: float
    portfolio_risk: float
    correlation_risk: float

    # Execution details
    order_type: str | None = None
    leverage: float | None = None
    execution_strategy: str | None = None

    # Additional metadata
    market_session: str = ""  # "london", "new_york", "tokyo", "sydney"
    economic_events: list[str] = field(default_factory=list)
    sentiment_score: float | None = None
    funding_rate: float | None = None


@dataclass
class TradeExecution:
    """Trade execution monitoring."""

    execution_id: str
    decision_id: str
    timestamp: datetime

    # Order details
    order_id: str | None = None
    symbol: str
    side: str  # "buy", "sell"
    order_type: str
    quantity: float
    price: float | None = None
    leverage: float | None = None

    # Execution results
    executed_quantity: float = 0.0
    average_execution_price: float = 0.0
    execution_time_ms: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0

    # Status tracking
    status: str = "pending"  # "pending", "filled", "partial", "cancelled", "failed"
    error_message: str | None = None

    # Performance tracking
    pnl_unrealized: float = 0.0
    pnl_realized: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Post-execution analysis
    execution_quality_score: float | None = None
    timing_analysis: dict[str, float] = field(default_factory=dict)


@dataclass
class TradeOutcome:
    """Final trade outcome analysis."""

    trade_id: str
    decision_id: str
    execution_id: str
    symbol: str

    # Trade summary
    entry_time: datetime
    exit_time: datetime | None = None
    duration_minutes: float | None = None

    # Performance
    entry_price: float
    exit_price: float | None = None
    quantity: float
    pnl_percentage: float | None = None
    pnl_absolute: float | None = None

    # Risk metrics
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    risk_adjusted_return: float | None = None

    # Model performance validation
    prediction_accuracy: float | None = None
    confidence_calibration: float | None = None
    regime_prediction_accuracy: float | None = None

    # Lessons learned
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)


class TradeConditionsMonitor:
    """
    Comprehensive trade conditions monitor that tracks every aspect of trading decisions
    and execution for ML model improvement and debugging.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize trade conditions monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("TradeConditionsMonitor")

        # Configuration
        self.monitor_config = config.get("trade_conditions_monitor", {})
        self.enable_detailed_logging = self.monitor_config.get(
            "enable_detailed_logging",
            True,
        )
        self.enable_feature_analysis = self.monitor_config.get(
            "enable_feature_analysis",
            True,
        )
        self.enable_model_tracking = self.monitor_config.get(
            "enable_model_tracking",
            True,
        )
        self.storage_backend = self.monitor_config.get("storage_backend", "sqlite")

        # Storage
        self.trade_decisions: dict[str, TradeDecisionContext] = {}
        self.trade_executions: dict[str, TradeExecution] = {}
        self.trade_outcomes: dict[str, TradeOutcome] = {}

        # Multi-timeframe data cache
        self.timeframe_cache: dict[str, dict[str, pd.DataFrame]] = {}

        # Performance tracking
        self.monitoring_stats = {
            "decisions_tracked": 0,
            "executions_tracked": 0,
            "outcomes_tracked": 0,
            "errors_detected": 0,
            "last_update": datetime.now(),
        }

        # Initialize storage
        self.storage_manager = None
        self.is_initialized = False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid trade monitor configuration"),
            AttributeError: (False, "Missing required monitor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="trade monitor initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the trade conditions monitor.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Trade Conditions Monitor...")

            # Initialize storage backend
            await self._initialize_storage()

            # Initialize timeframe cache
            await self._initialize_timeframe_cache()

            # Load historical data if needed
            await self._load_historical_data()

            self.is_initialized = True
            self.logger.info("✅ Trade Conditions Monitor initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Trade Conditions Monitor initialization failed: {e}",
            )
            return False

    async def _initialize_storage(self) -> None:
        """Initialize storage backend."""
        try:
            if self.storage_backend == "sqlite":
                from src.database.sqlite_manager import SQLiteManager

                self.storage_manager = SQLiteManager(self.config)
                await self.storage_manager.initialize()

                # Create tables for trade monitoring
                await self._create_monitoring_tables()

            elif self.storage_backend == "influxdb":
                from src.database.influxdb_manager import InfluxDBManager

                self.storage_manager = InfluxDBManager(self.config)
                await self.storage_manager.initialize()

            self.logger.info(f"Storage backend '{self.storage_backend}' initialized")

        except Exception:
            self.print(failed("Failed to initialize storage backend: {e}"))
            raise

    async def _create_monitoring_tables(self) -> None:
        """Create database tables for trade monitoring."""
        try:
            tables = [
                """
                CREATE TABLE IF NOT EXISTS trade_decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    symbol TEXT,
                    exchange TEXT,
                    current_price REAL,
                    current_regime TEXT,
                    regime_confidence REAL,
                    final_prediction REAL,
                    final_confidence REAL,
                    recommended_action TEXT,
                    risk_score REAL,
                    decision_context TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS trade_executions (
                    execution_id TEXT PRIMARY KEY,
                    decision_id TEXT,
                    timestamp DATETIME,
                    order_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    executed_quantity REAL,
                    average_price REAL,
                    slippage REAL,
                    commission REAL,
                    status TEXT,
                    execution_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (decision_id) REFERENCES trade_decisions (decision_id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    trade_id TEXT PRIMARY KEY,
                    decision_id TEXT,
                    execution_id TEXT,
                    symbol TEXT,
                    entry_time DATETIME,
                    exit_time DATETIME,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_percentage REAL,
                    pnl_absolute REAL,
                    max_drawdown REAL,
                    prediction_accuracy REAL,
                    outcome_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (decision_id) REFERENCES trade_decisions (decision_id),
                    FOREIGN KEY (execution_id) REFERENCES trade_executions (execution_id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS timeframe_features (
                    feature_id TEXT PRIMARY KEY,
                    decision_id TEXT,
                    timeframe TEXT,
                    timestamp DATETIME,
                    price REAL,
                    volume REAL,
                    features_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (decision_id) REFERENCES trade_decisions (decision_id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS model_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    decision_id TEXT,
                    model_id TEXT,
                    model_type TEXT,
                    ensemble_type TEXT,
                    prediction REAL,
                    confidence REAL,
                    feature_importance TEXT,
                    prediction_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (decision_id) REFERENCES trade_decisions (decision_id)
                )
                """,
            ]

            for table_sql in tables:
                await self.storage_manager.execute_query(table_sql)

            self.logger.info("Trade monitoring tables created successfully")

        except Exception:
            self.print(failed("Failed to create monitoring tables: {e}"))
            raise

    async def _initialize_timeframe_cache(self) -> None:
        """Initialize multi-timeframe data cache."""
        try:
            timeframes = ["30m", "15m", "5m", "1m"]
            for tf in timeframes:
                self.timeframe_cache[tf] = {}

            self.logger.info("Timeframe cache initialized")

        except Exception:
            self.print(failed("Failed to initialize timeframe cache: {e}"))
            raise

    async def _load_historical_data(self) -> None:
        """Load historical monitoring data if needed."""
        try:
            # Load recent trade decisions for analysis
            if self.storage_manager:
                recent_decisions = await self._load_recent_decisions()
                self.logger.info(
                    f"Loaded {len(recent_decisions)} recent trade decisions",
                )

        except Exception:
            self.print(failed("Failed to load historical data: {e}"))
            # Non-critical error, continue

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="trade decision recording",
    )
    async def record_trade_decision(
        self,
        decision_context: TradeDecisionContext,
    ) -> bool:
        """
        Record a comprehensive trade decision with all context.

        Args:
            decision_context: Complete trade decision context

        Returns:
            bool: True if recorded successfully, False otherwise
        """
        try:
            self.logger.info(
                f"Recording trade decision: {decision_context.decision_id}",
            )

            # Store in memory
            self.trade_decisions[decision_context.decision_id] = decision_context

            # Store in database
            if self.storage_manager:
                await self._store_decision_in_db(decision_context)

            # Update monitoring stats
            self.monitoring_stats["decisions_tracked"] += 1
            self.monitoring_stats["last_update"] = datetime.now()

            # Log detailed decision information
            if self.enable_detailed_logging:
                await self._log_decision_details(decision_context)

            return True

        except Exception:
            self.print(failed("Failed to record trade decision: {e}"))
            self.monitoring_stats["errors_detected"] += 1
            return False

    async def _store_decision_in_db(
        self,
        decision_context: TradeDecisionContext,
    ) -> None:
        """Store trade decision in database."""
        try:
            # Store main decision
            decision_data = {
                "decision_id": decision_context.decision_id,
                "timestamp": decision_context.timestamp,
                "symbol": decision_context.symbol,
                "exchange": decision_context.exchange,
                "current_price": decision_context.current_price,
                "current_regime": decision_context.current_regime.value,
                "regime_confidence": decision_context.regime_confidence,
                "final_prediction": decision_context.final_prediction,
                "final_confidence": decision_context.final_confidence,
                "recommended_action": decision_context.recommended_action.value,
                "risk_score": decision_context.risk_score,
                "decision_context": json.dumps(asdict(decision_context), default=str),
            }

            await self.storage_manager.insert_data("trade_decisions", decision_data)

            # Store timeframe features
            for timeframe, features in decision_context.timeframe_features.items():
                feature_data = {
                    "feature_id": f"{decision_context.decision_id}_{timeframe}",
                    "decision_id": decision_context.decision_id,
                    "timeframe": timeframe,
                    "timestamp": features.timestamp,
                    "price": features.price,
                    "volume": features.volume,
                    "features_json": json.dumps(asdict(features), default=str),
                }

                await self.storage_manager.insert_data(
                    "timeframe_features",
                    feature_data,
                )

            # Store model predictions
            for ensemble in decision_context.ensemble_predictions:
                for prediction in ensemble.individual_predictions:
                    prediction_data = {
                        "prediction_id": f"{decision_context.decision_id}_{prediction.model_id}",
                        "decision_id": decision_context.decision_id,
                        "model_id": prediction.model_id,
                        "model_type": prediction.model_type,
                        "ensemble_type": prediction.ensemble_type,
                        "prediction": prediction.prediction,
                        "confidence": prediction.confidence,
                        "feature_importance": json.dumps(prediction.feature_importance),
                        "prediction_details": json.dumps(
                            asdict(prediction),
                            default=str,
                        ),
                    }

                    await self.storage_manager.insert_data(
                        "model_predictions",
                        prediction_data,
                    )

        except Exception:
            self.print(failed("Failed to store decision in database: {e}"))
            raise

    async def _log_decision_details(
        self,
        decision_context: TradeDecisionContext,
    ) -> None:
        """Log detailed decision information for debugging."""
        try:
            details = [
                "🎯 Trade Decision Details:",
                f"   Decision ID: {decision_context.decision_id}",
                f"   Symbol: {decision_context.symbol}",
                f"   Current Price: ${decision_context.current_price:.4f}",
                f"   Regime: {decision_context.current_regime.value} (confidence: {decision_context.regime_confidence:.2f})",
                f"   Final Prediction: {decision_context.final_prediction:.4f}",
                f"   Confidence: {decision_context.final_confidence:.2f}",
                f"   Recommended Action: {decision_context.recommended_action.value}",
                f"   Risk Score: {decision_context.risk_score:.2f}",
            ]

            # Support/Resistance levels
            if decision_context.nearby_sr_levels:
                details.append(
                    f"   Nearby S/R Levels: {len(decision_context.nearby_sr_levels)}",
                )
                for sr in decision_context.nearby_sr_levels[:3]:  # Top 3
                    details.append(
                        f"     - {sr.level_type.value}: ${sr.price:.4f} (strength: {sr.strength:.2f})",
                    )

            # Model predictions summary
            details.append(
                f"   Ensemble Predictions: {len(decision_context.ensemble_predictions)}",
            )
            for ensemble in decision_context.ensemble_predictions:
                details.append(
                    f"     - {ensemble.regime_type.value}: {ensemble.aggregated_prediction:.4f} (consensus: {ensemble.consensus_level:.2f})",
                )

            # Multi-timeframe features
            details.append(
                f"   Timeframes Analyzed: {list(decision_context.timeframe_features.keys())}",
            )

            self.logger.info("\n".join(details))

        except Exception:
            self.print(failed("Failed to log decision details: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="trade execution recording",
    )
    async def record_trade_execution(self, execution: TradeExecution) -> bool:
        """
        Record trade execution details.

        Args:
            execution: Trade execution details

        Returns:
            bool: True if recorded successfully, False otherwise
        """
        try:
            self.logger.info(f"Recording trade execution: {execution.execution_id}")

            # Store in memory
            self.trade_executions[execution.execution_id] = execution

            # Store in database
            if self.storage_manager:
                await self._store_execution_in_db(execution)

            # Update monitoring stats
            self.monitoring_stats["executions_tracked"] += 1
            self.monitoring_stats["last_update"] = datetime.now()

            # Analyze execution quality
            await self._analyze_execution_quality(execution)

            return True

        except Exception:
            self.print(failed("Failed to record trade execution: {e}"))
            self.monitoring_stats["errors_detected"] += 1
            return False

    async def _store_execution_in_db(self, execution: TradeExecution) -> None:
        """Store trade execution in database."""
        try:
            execution_data = {
                "execution_id": execution.execution_id,
                "decision_id": execution.decision_id,
                "timestamp": execution.timestamp,
                "order_id": execution.order_id,
                "symbol": execution.symbol,
                "side": execution.side,
                "quantity": execution.quantity,
                "executed_quantity": execution.executed_quantity,
                "average_price": execution.average_execution_price,
                "slippage": execution.slippage,
                "commission": execution.commission,
                "status": execution.status,
                "execution_details": json.dumps(asdict(execution), default=str),
            }

            await self.storage_manager.insert_data("trade_executions", execution_data)

        except Exception:
            self.print(failed("Failed to store execution in database: {e}"))
            raise

    async def _analyze_execution_quality(self, execution: TradeExecution) -> None:
        """Analyze execution quality and timing."""
        try:
            # Calculate execution quality score based on slippage, timing, etc.
            quality_factors = []

            # Slippage factor (lower is better)
            if execution.slippage is not None:
                slippage_factor = max(
                    0,
                    1 - abs(execution.slippage) / 0.01,
                )  # Normalize to 1% slippage
                quality_factors.append(slippage_factor)

            # Execution speed factor (faster is better)
            if execution.execution_time_ms > 0:
                speed_factor = max(
                    0,
                    1 - execution.execution_time_ms / 5000,
                )  # Normalize to 5 seconds
                quality_factors.append(speed_factor)

            # Fill ratio factor (complete fills are better)
            if execution.quantity > 0:
                fill_factor = execution.executed_quantity / execution.quantity
                quality_factors.append(fill_factor)

            # Calculate overall quality score
            if quality_factors:
                execution.execution_quality_score = sum(quality_factors) / len(
                    quality_factors,
                )

            # Log quality analysis
            self.logger.debug(
                f"Execution quality for {execution.execution_id}: {execution.execution_quality_score:.2f}",
            )

        except Exception:
            self.print(failed("Failed to analyze execution quality: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="trade outcome recording",
    )
    async def record_trade_outcome(self, outcome: TradeOutcome) -> bool:
        """
        Record final trade outcome and analysis.

        Args:
            outcome: Trade outcome details

        Returns:
            bool: True if recorded successfully, False otherwise
        """
        try:
            self.logger.info(f"Recording trade outcome: {outcome.trade_id}")

            # Store in memory
            self.trade_outcomes[outcome.trade_id] = outcome

            # Store in database
            if self.storage_manager:
                await self._store_outcome_in_db(outcome)

            # Update monitoring stats
            self.monitoring_stats["outcomes_tracked"] += 1
            self.monitoring_stats["last_update"] = datetime.now()

            # Perform post-trade analysis
            await self._perform_post_trade_analysis(outcome)

            return True

        except Exception:
            self.print(failed("Failed to record trade outcome: {e}"))
            self.monitoring_stats["errors_detected"] += 1
            return False

    async def _store_outcome_in_db(self, outcome: TradeOutcome) -> None:
        """Store trade outcome in database."""
        try:
            outcome_data = {
                "trade_id": outcome.trade_id,
                "decision_id": outcome.decision_id,
                "execution_id": outcome.execution_id,
                "symbol": outcome.symbol,
                "entry_time": outcome.entry_time,
                "exit_time": outcome.exit_time,
                "entry_price": outcome.entry_price,
                "exit_price": outcome.exit_price,
                "pnl_percentage": outcome.pnl_percentage,
                "pnl_absolute": outcome.pnl_absolute,
                "max_drawdown": outcome.max_drawdown,
                "prediction_accuracy": outcome.prediction_accuracy,
                "outcome_details": json.dumps(asdict(outcome), default=str),
            }

            await self.storage_manager.insert_data("trade_outcomes", outcome_data)

        except Exception:
            self.print(failed("Failed to store outcome in database: {e}"))
            raise

    async def _perform_post_trade_analysis(self, outcome: TradeOutcome) -> None:
        """Perform comprehensive post-trade analysis."""
        try:
            # Get original decision context
            decision = self.trade_decisions.get(outcome.decision_id)
            if not decision:
                self.logger.warning(
                    f"No decision context found for trade {outcome.trade_id}",
                )
                return

            # Analyze prediction accuracy
            if (
                outcome.pnl_percentage is not None
                and decision.final_prediction is not None
            ):
                # Simple accuracy: did the prediction direction match the outcome?
                predicted_positive = decision.final_prediction > 0
                actual_positive = outcome.pnl_percentage > 0
                outcome.prediction_accuracy = (
                    1.0 if predicted_positive == actual_positive else 0.0
                )

            # Analyze regime prediction accuracy
            if decision.current_regime and outcome.exit_time:
                # This would require checking if the regime continued as predicted
                # For now, placeholder implementation
                outcome.regime_prediction_accuracy = 0.8  # Placeholder

            # Generate improvement suggestions
            await self._generate_improvement_suggestions(outcome, decision)

            # Log analysis results
            self.logger.info(f"Post-trade analysis completed for {outcome.trade_id}")

        except Exception:
            self.print(failed("Failed to perform post-trade analysis: {e}"))

    async def _generate_improvement_suggestions(
        self,
        outcome: TradeOutcome,
        decision: TradeDecisionContext,
    ) -> None:
        """Generate improvement suggestions based on trade outcome."""
        try:
            suggestions = []

            # Analyze what worked
            if outcome.pnl_percentage and outcome.pnl_percentage > 0:
                outcome.what_worked.append("Trade was profitable")
                if decision.regime_confidence > 0.8:
                    outcome.what_worked.append("High regime confidence led to success")
                if decision.final_confidence > 0.8:
                    outcome.what_worked.append(
                        "High prediction confidence was justified",
                    )

            # Analyze what failed
            if outcome.pnl_percentage and outcome.pnl_percentage < 0:
                outcome.what_failed.append("Trade was unprofitable")
                if decision.regime_confidence < 0.6:
                    outcome.what_failed.append(
                        "Low regime confidence should have been avoided",
                    )
                if outcome.max_drawdown > 0.05:  # 5%
                    outcome.what_failed.append("Excessive drawdown experienced")

            # Generate suggestions
            if (
                decision.risk_score > 0.7
                and outcome.pnl_percentage
                and outcome.pnl_percentage < 0
            ):
                suggestions.append("Consider avoiding trades with high risk scores")

            if decision.final_confidence < 0.6:
                suggestions.append(
                    "Require higher prediction confidence for trade entry",
                )

            if len(decision.ensemble_predictions) < 3:
                suggestions.append(
                    "Use more ensemble models for better prediction diversity",
                )

            outcome.improvement_suggestions = suggestions

        except Exception:
            self.print(failed("Failed to generate improvement suggestions: {e}"))

    async def get_multi_timeframe_features(
        self,
        symbol: str,
        timestamp: datetime,
        timeframes: list[str] = None,
    ) -> dict[str, MultiTimeframeFeatures]:
        """
        Get multi-timeframe feature analysis for a given symbol and time.

        Args:
            symbol: Trading symbol
            timestamp: Analysis timestamp
            timeframes: List of timeframes to analyze (default: ["30m", "15m", "5m", "1m"])

        Returns:
            dict: Multi-timeframe features
        """
        try:
            if timeframes is None:
                timeframes = ["30m", "15m", "5m", "1m"]

            features = {}

            for tf in timeframes:
                # Get cached data or fetch new data
                tf_data = await self._get_timeframe_data(symbol, tf, timestamp)

                if tf_data is not None and not tf_data.empty:
                    # Calculate features for this timeframe
                    tf_features = await self._calculate_timeframe_features(
                        tf_data,
                        tf,
                        timestamp,
                    )
                    features[tf] = tf_features

            return features

        except Exception:
            self.print(failed("Failed to get multi-timeframe features: {e}"))
            return {}

    async def _get_timeframe_data(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
    ) -> pd.DataFrame | None:
        """Get timeframe data from cache or fetch from source."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.timeframe_cache[timeframe]:
                data = self.timeframe_cache[timeframe][cache_key]

                # Check if data is recent enough
                if not data.empty and data.index[-1] >= timestamp - timedelta(
                    minutes=int(timeframe.replace("m", "")),
                ):
                    return data

            # Fetch new data (placeholder - integrate with your data source)
            # This would typically call your data downloader or exchange API
            data = await self._fetch_timeframe_data(symbol, timeframe, timestamp)

            # Cache the data
            if data is not None:
                self.timeframe_cache[timeframe][cache_key] = data

            return data

        except Exception as e:
            self.logger.exception(
                f"Failed to get timeframe data for {symbol} {timeframe}: {e}",
            )
            return None

    async def _fetch_timeframe_data(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
    ) -> pd.DataFrame | None:
        """Fetch timeframe data from data source."""
        try:
            # Placeholder implementation - integrate with your data source
            # This should call your exchange API or data downloader

            # For now, return None to indicate no data available
            # In a real implementation, you would:
            # 1. Call your exchange API
            # 2. Get the last N candles for this timeframe
            # 3. Return as DataFrame with OHLCV data

            self.logger.debug(f"Fetching {timeframe} data for {symbol} at {timestamp}")
            return None

        except Exception:
            self.print(failed("Failed to fetch timeframe data: {e}"))
            return None

    async def _calculate_timeframe_features(
        self,
        data: pd.DataFrame,
        timeframe: str,
        timestamp: datetime,
    ) -> MultiTimeframeFeatures:
        """Calculate features for a specific timeframe."""
        try:
            # Get the latest values
            latest = data.iloc[-1]

            features = MultiTimeframeFeatures(
                timeframe=timeframe,
                timestamp=timestamp,
                price=latest.get("close", 0.0),
                volume=latest.get("volume", 0.0),
            )

            # Calculate technical indicators
            if len(data) >= 14:  # Minimum for RSI
                # RSI
                delta = data["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features.rsi = 100 - (100 / (1 + rs.iloc[-1]))

            if len(data) >= 26:  # Minimum for MACD
                # MACD
                ema12 = data["close"].ewm(span=12).mean()
                ema26 = data["close"].ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                features.macd_signal = signal.iloc[-1]
                features.macd_histogram = (macd - signal).iloc[-1]

            if len(data) >= 20:  # Minimum for Bollinger Bands
                # Bollinger Bands
                sma20 = data["close"].rolling(window=20).mean()
                std20 = data["close"].rolling(window=20).std()
                features.bollinger_upper = (sma20 + 2 * std20).iloc[-1]
                features.bollinger_lower = (sma20 - 2 * std20).iloc[-1]
                features.bollinger_position = (
                    latest["close"] - features.bollinger_lower
                ) / (features.bollinger_upper - features.bollinger_lower)

            # EMAs
            if len(data) >= 9:
                features.ema_9 = data["close"].ewm(span=9).mean().iloc[-1]
            if len(data) >= 21:
                features.ema_21 = data["close"].ewm(span=21).mean().iloc[-1]
            if len(data) >= 50:
                features.ema_50 = data["close"].ewm(span=50).mean().iloc[-1]
            if len(data) >= 200:
                features.ema_200 = data["close"].ewm(span=200).mean().iloc[-1]

            # Volume indicators
            if len(data) >= 20:
                features.volume_sma = data["volume"].rolling(window=20).mean().iloc[-1]
                features.volume_ratio = (
                    latest["volume"] / features.volume_sma
                    if features.volume_sma > 0
                    else 1.0
                )

            # Volatility (ATR)
            if len(data) >= 14:
                high_low = data["high"] - data["low"]
                high_close = abs(data["high"] - data["close"].shift())
                low_close = abs(data["low"] - data["close"].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                    axis=1,
                )
                features.atr = true_range.rolling(window=14).mean().iloc[-1]

            return features

        except Exception:
            self.print(failed("Failed to calculate timeframe features: {e}"))
            return MultiTimeframeFeatures(
                timeframe=timeframe,
                timestamp=timestamp,
                price=0.0,
                volume=0.0,
            )

    async def get_monitoring_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive monitoring statistics.

        Returns:
            dict: Monitoring statistics
        """
        try:
            stats = self.monitoring_stats.copy()

            # Add additional statistics
            stats.update(
                {
                    "active_decisions": len(self.trade_decisions),
                    "active_executions": len(self.trade_executions),
                    "completed_outcomes": len(self.trade_outcomes),
                    "cache_size": sum(
                        len(cache) for cache in self.timeframe_cache.values()
                    ),
                    "is_initialized": self.is_initialized,
                    "storage_backend": self.storage_backend,
                },
            )

            # Calculate success rates if we have outcomes
            if self.trade_outcomes:
                profitable_trades = sum(
                    1
                    for outcome in self.trade_outcomes.values()
                    if outcome.pnl_percentage and outcome.pnl_percentage > 0
                )
                stats["win_rate"] = profitable_trades / len(self.trade_outcomes)

                avg_pnl = np.mean(
                    [
                        outcome.pnl_percentage
                        for outcome in self.trade_outcomes.values()
                        if outcome.pnl_percentage is not None
                    ],
                )
                stats["average_pnl_percentage"] = avg_pnl

            return stats

        except Exception:
            self.print(failed("Failed to get monitoring statistics: {e}"))
            return {}

    async def generate_monitoring_report(self, days: int = 7) -> dict[str, Any]:
        """
        Generate comprehensive monitoring report.

        Args:
            days: Number of days to include in report

        Returns:
            dict: Monitoring report
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Filter recent data
            recent_decisions = {
                k: v
                for k, v in self.trade_decisions.items()
                if v.timestamp >= cutoff_date
            }

            recent_executions = {
                k: v
                for k, v in self.trade_executions.items()
                if v.timestamp >= cutoff_date
            }

            recent_outcomes = {
                k: v
                for k, v in self.trade_outcomes.items()
                if v.entry_time >= cutoff_date
            }

            # Generate report
            report = {
                "report_period_days": days,
                "report_generated": datetime.now(),
                "summary": {
                    "total_decisions": len(recent_decisions),
                    "total_executions": len(recent_executions),
                    "total_outcomes": len(recent_outcomes),
                },
                "performance_metrics": {},
                "model_analysis": {},
                "regime_analysis": {},
                "timeframe_analysis": {},
                "improvement_recommendations": [],
            }

            # Performance metrics
            if recent_outcomes:
                pnl_values = [
                    outcome.pnl_percentage
                    for outcome in recent_outcomes.values()
                    if outcome.pnl_percentage is not None
                ]

                if pnl_values:
                    report["performance_metrics"] = {
                        "win_rate": sum(1 for pnl in pnl_values if pnl > 0)
                        / len(pnl_values),
                        "average_pnl": np.mean(pnl_values),
                        "max_profit": max(pnl_values),
                        "max_loss": min(pnl_values),
                        "std_pnl": np.std(pnl_values),
                        "sharpe_ratio": np.mean(pnl_values) / np.std(pnl_values)
                        if np.std(pnl_values) > 0
                        else 0,
                    }

            # Model analysis
            if recent_decisions:
                model_predictions = []
                for decision in recent_decisions.values():
                    for ensemble in decision.ensemble_predictions:
                        for prediction in ensemble.individual_predictions:
                            model_predictions.append(
                                {
                                    "model_type": prediction.model_type,
                                    "prediction": prediction.prediction,
                                    "confidence": prediction.confidence,
                                },
                            )

                if model_predictions:
                    model_df = pd.DataFrame(model_predictions)
                    report["model_analysis"] = {
                        "total_predictions": len(model_predictions),
                        "unique_models": model_df["model_type"].nunique(),
                        "average_confidence": model_df["confidence"].mean(),
                        "model_distribution": model_df["model_type"]
                        .value_counts()
                        .to_dict(),
                    }

            # Regime analysis
            if recent_decisions:
                regimes = [
                    decision.current_regime.value
                    for decision in recent_decisions.values()
                ]
                regime_counts = pd.Series(regimes).value_counts()

                report["regime_analysis"] = {
                    "regime_distribution": regime_counts.to_dict(),
                    "most_common_regime": regime_counts.index[0]
                    if not regime_counts.empty
                    else None,
                }

            return report

        except Exception:
            self.print(failed("Failed to generate monitoring report: {e}"))
            return {}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.logger.info("Cleaning up Trade Conditions Monitor...")

            # Clear caches
            self.timeframe_cache.clear()

            # Close storage connections
            if self.storage_manager:
                await self.storage_manager.close()

            self.logger.info("Trade Conditions Monitor cleanup completed")

        except Exception:
            self.print(failed("Failed to cleanup Trade Conditions Monitor: {e}"))
