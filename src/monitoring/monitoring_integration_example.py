#!/usr/bin/env python3
"""
Monitoring Integration Example

This module demonstrates how to integrate the comprehensive monitoring system
with the trading pipeline for complete observability and trade analysis.
"""

import asyncio
from datetime import datetime
from typing import Any

from src.monitoring import (
    EnhancedMLTracker,
    ErrorDetectionSystem,
    MonitoringIntegrationManager,
    RegimeSRTracker,
    TradeConditionsMonitor,
)
from src.monitoring.enhanced_ml_tracker import ModelType, PredictionType
from src.monitoring.error_detection_system import AlertSeverity, ErrorCategory
from src.monitoring.trade_conditions_monitor import (
    EnsemblePrediction,
    ModelPrediction,
    MultiTimeframeFeatures,
    RegimeType,
    TradeAction,
    TradeDecisionContext,
    TradeExecution,
    TradeOutcome,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    warning,
)


class MonitoringIntegrationExample:
    """
    Example integration of the comprehensive monitoring system with
    the trading pipeline to demonstrate complete trade analysis capabilities.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize monitoring integration example."""
        self.config = config
        self.logger = system_logger.getChild("MonitoringIntegrationExample")

        # Initialize monitoring components
        self.monitoring_manager: MonitoringIntegrationManager | None = None
        self.trade_monitor: TradeConditionsMonitor | None = None
        self.ml_tracker: EnhancedMLTracker | None = None
        self.regime_tracker: RegimeSRTracker | None = None
        self.error_detector: ErrorDetectionSystem | None = None

    async def initialize(self) -> bool:
        """Initialize all monitoring components."""
        try:
            self.logger.info("Initializing comprehensive monitoring system...")

            # Initialize the main monitoring integration manager
            self.monitoring_manager = MonitoringIntegrationManager(self.config)
            if not await self.monitoring_manager.initialize():
                self.print(failed("Failed to initialize monitoring manager"))
                return False

            # Get individual components
            components = self.monitoring_manager.components
            self.trade_monitor = components.trade_conditions_monitor
            self.ml_tracker = components.enhanced_ml_tracker
            self.regime_tracker = components.regime_sr_tracker
            self.error_detector = components.error_detection_system

            self.logger.info(
                "✅ Comprehensive monitoring system initialized successfully",
            )
            return True

        except Exception:
            self.print(failed("Failed to initialize monitoring system: {e}"))
            return False

    async def demonstrate_trade_analysis_workflow(self) -> None:
        """Demonstrate complete trade analysis workflow."""
        try:
            self.logger.info(
                "🚀 Starting comprehensive trade analysis demonstration...",
            )

            # 1. Collect multi-timeframe market data and features
            symbol = "ETHUSDT"
            timeframes = ["30m", "15m", "5m", "1m"]

            # Get multi-timeframe features
            timeframe_features = await self._collect_multi_timeframe_features(
                symbol,
                timeframes,
            )

            # 2. Detect current market regime and S/R levels
            current_regime = await self._detect_market_conditions(symbol)
            sr_levels = await self._identify_sr_levels(symbol)

            # 3. Generate ML model predictions
            model_predictions = await self._generate_model_predictions(
                timeframe_features,
            )
            ensemble_predictions = await self._aggregate_ensemble_predictions(
                model_predictions,
            )

            # 4. Make trading decision with complete context
            trade_decision = await self._make_trading_decision(
                symbol,
                timeframe_features,
                current_regime,
                sr_levels,
                ensemble_predictions,
            )

            # 5. Execute trade and monitor
            execution_result = await self._execute_and_monitor_trade(trade_decision)

            # 6. Track performance and generate insights
            await self._analyze_trade_performance(trade_decision, execution_result)

            # 7. Demonstrate error detection and alerting
            await self._demonstrate_error_detection()

            self.logger.info("✅ Trade analysis demonstration completed successfully")

        except Exception as e:
            self.print(failed("Trade analysis demonstration failed: {e}"))
            # Record error for monitoring
            if self.error_detector:
                await self.error_detector.record_error_event(
                    severity=AlertSeverity.ERROR,
                    category=ErrorCategory.SYSTEM,
                    error_message=f"Trade analysis demonstration failed: {e}",
                    component="MonitoringIntegrationExample",
                    function="demonstrate_trade_analysis_workflow",
                )

    async def _collect_multi_timeframe_features(
        self,
        symbol: str,
        timeframes: list[str],
    ) -> dict[str, MultiTimeframeFeatures]:
        """Collect features across multiple timeframes."""
        try:
            self.logger.info(f"📊 Collecting multi-timeframe features for {symbol}")

            if not self.trade_monitor:
                return {}

            # This would integrate with your data source
            # For demonstration, we'll create example features
            features = {}
            current_time = datetime.now()

            for tf in timeframes:
                # Example feature calculation for each timeframe
                tf_features = MultiTimeframeFeatures(
                    timeframe=tf,
                    timestamp=current_time,
                    price=3500.0 + (hash(tf) % 100),  # Example price
                    volume=1000000.0 + (hash(tf) % 500000),  # Example volume
                    # Technical indicators (example values)
                    rsi=50.0 + (hash(tf + "rsi") % 40),
                    macd_signal=hash(tf + "macd") % 10 - 5,
                    bollinger_upper=3550.0 + (hash(tf) % 50),
                    bollinger_lower=3450.0 + (hash(tf) % 50),
                    bollinger_position=(hash(tf) % 100) / 100.0,
                    # Volume indicators
                    volume_sma=800000.0 + (hash(tf) % 400000),
                    volume_ratio=0.8 + (hash(tf) % 40) / 100.0,
                    vwap=3480.0 + (hash(tf) % 40),
                    # Volatility indicators
                    atr=15.0 + (hash(tf) % 20),
                    volatility=0.02 + (hash(tf) % 8) / 1000.0,
                    # Trend indicators
                    ema_9=3490.0 + (hash(tf + "ema9") % 30),
                    ema_21=3485.0 + (hash(tf + "ema21") % 30),
                    ema_50=3480.0 + (hash(tf + "ema50") % 30),
                    ema_200=3400.0 + (hash(tf + "ema200") % 100),
                    trend_strength=(hash(tf + "trend") % 100) / 100.0,
                    # Momentum indicators
                    momentum=(hash(tf + "momentum") % 20 - 10) / 100.0,
                    rate_of_change=(hash(tf + "roc") % 10 - 5) / 100.0,
                    # Custom features
                    regime_probability={
                        "bull_trend": (hash(tf + "bull") % 100) / 100.0,
                        "bear_trend": (hash(tf + "bear") % 100) / 100.0,
                        "sideways": (hash(tf + "sideways") % 100) / 100.0,
                    },
                    market_microstructure={
                        "bid_ask_spread": (hash(tf + "spread") % 5) / 1000.0,
                        "order_book_imbalance": (hash(tf + "imbalance") % 20 - 10)
                        / 100.0,
                    },
                    liquidity_metrics={
                        "market_depth": 50000.0 + (hash(tf + "depth") % 200000),
                        "liquidity_ratio": 0.7 + (hash(tf + "liquidity") % 30) / 100.0,
                    },
                )

                features[tf] = tf_features

            # Get features using the trade monitor
            monitor_features = await self.trade_monitor.get_multi_timeframe_features(
                symbol,
                current_time,
                timeframes,
            )

            # Merge with example features (in practice, you'd use real data)
            features.update(monitor_features)

            self.logger.info(f"✅ Collected features for {len(features)} timeframes")
            return features

        except Exception:
            self.print(failed("Failed to collect multi-timeframe features: {e}"))
            return {}

    async def _detect_market_conditions(self, symbol: str) -> dict[str, Any]:
        """Detect current market regime and conditions."""
        try:
            self.logger.info(f"🔍 Detecting market conditions for {symbol}")

            if not self.regime_tracker:
                return {}

            # Detect regime for multiple timeframes
            timeframes = ["1h", "4h"]
            regime_data = {}

            for tf in timeframes:
                regime = await self.regime_tracker.detect_current_regime(symbol, tf)
                if regime:
                    regime_data[tf] = {
                        "regime_type": regime.current_regime.value,
                        "confidence": regime.confidence,
                        "duration_minutes": regime.duration_minutes,
                        "price_action_score": regime.price_action_score,
                        "volume_score": regime.volume_score,
                        "volatility_score": regime.volatility_score,
                    }

            self.logger.info(f"✅ Detected regimes for {len(regime_data)} timeframes")
            return regime_data

        except Exception:
            self.print(failed("Failed to detect market conditions: {e}"))
            return {}

    async def _identify_sr_levels(self, symbol: str) -> dict[str, list[dict[str, Any]]]:
        """Identify support and resistance levels."""
        try:
            self.logger.info(f"📈 Identifying S/R levels for {symbol}")

            if not self.regime_tracker:
                return {}

            # Identify S/R levels for multiple timeframes
            timeframes = ["1h", "4h"]
            sr_data = {}

            for tf in timeframes:
                sr_levels = await self.regime_tracker.identify_sr_levels(symbol, tf)
                sr_data[tf] = [
                    {
                        "level_type": level.level_type.value,
                        "price": level.price,
                        "strength": level.strength,
                        "confidence": level.confidence,
                        "touch_count": level.touch_count,
                        "distance_from_current": level.distance_from_current,
                    }
                    for level in sr_levels
                ]

            self.logger.info(f"✅ Identified S/R levels for {len(sr_data)} timeframes")
            return sr_data

        except Exception:
            self.print(failed("Failed to identify S/R levels: {e}"))
            return {}

    async def _generate_model_predictions(
        self,
        timeframe_features: dict[str, MultiTimeframeFeatures],
    ) -> list[ModelPrediction]:
        """Generate predictions from multiple ML models."""
        try:
            self.logger.info("🤖 Generating ML model predictions")

            predictions = []

            # Example predictions from different model types
            model_configs = [
                {
                    "id": "xgb_1h",
                    "type": ModelType.XGBOOST,
                    "ensemble": "trend_following",
                },
                {
                    "id": "cat_1h",
                    "type": ModelType.CATBOOST,
                    "ensemble": "mean_reversion",
                },
                {
                    "id": "nn_1h",
                    "type": ModelType.NEURAL_NETWORK,
                    "ensemble": "volatility_breakout",
                },
                {
                    "id": "rf_1h",
                    "type": ModelType.RANDOM_FOREST,
                    "ensemble": "ensemble_meta",
                },
            ]

            for _i, model_config in enumerate(model_configs):
                # Generate example prediction
                prediction_value = (
                    0.1 + (hash(model_config["id"]) % 80) / 100.0 - 0.4
                )  # -0.4 to 0.4
                confidence = (
                    0.6 + (hash(model_config["id"] + "conf") % 40) / 100.0
                )  # 0.6 to 1.0

                # Create feature importance (example)
                feature_importance = {}
                for tf in timeframe_features:
                    feature_importance[f"{tf}_rsi"] = (
                        hash(f"{model_config['id']}_{tf}_rsi") % 100
                    ) / 100.0
                    feature_importance[f"{tf}_macd"] = (
                        hash(f"{model_config['id']}_{tf}_macd") % 100
                    ) / 100.0
                    feature_importance[f"{tf}_volume_ratio"] = (
                        hash(f"{model_config['id']}_{tf}_vol") % 100
                    ) / 100.0

                prediction = ModelPrediction(
                    model_id=model_config["id"],
                    model_type=model_config["type"].value,
                    ensemble_type=model_config["ensemble"],
                    prediction=prediction_value,
                    confidence=confidence,
                    probability_distribution={
                        "buy": max(0, prediction_value),
                        "sell": max(0, -prediction_value),
                        "hold": 1 - abs(prediction_value),
                    },
                    feature_importance=feature_importance,
                    prediction_reasoning=f"Model {model_config['id']} based on {len(timeframe_features)} timeframes",
                    model_version="v1.0",
                    last_training_date=datetime.now(),
                )

                predictions.append(prediction)

                # Track prediction with ML tracker
                if self.ml_tracker:
                    features_dict = {}
                    for tf, tf_features in timeframe_features.items():
                        features_dict.update(
                            {
                                f"{tf}_price": tf_features.price,
                                f"{tf}_volume": tf_features.volume,
                                f"{tf}_rsi": tf_features.rsi or 0.0,
                                f"{tf}_macd": tf_features.macd_signal or 0.0,
                            },
                        )

                    await self.ml_tracker.track_model_prediction(
                        model_id=model_config["id"],
                        model_type=model_config["type"],
                        ensemble_name=model_config["ensemble"],
                        prediction=prediction_value,
                        confidence=confidence,
                        features=features_dict,
                        feature_importance=feature_importance,
                        prediction_type=PredictionType.REGRESSION,
                        model_version="v1.0",
                    )

            self.logger.info(f"✅ Generated {len(predictions)} model predictions")
            return predictions

        except Exception:
            self.print(failed("Failed to generate model predictions: {e}"))
            return []

    async def _aggregate_ensemble_predictions(
        self,
        model_predictions: list[ModelPrediction],
    ) -> list[EnsemblePrediction]:
        """Aggregate model predictions into ensemble predictions."""
        try:
            self.logger.info("🔗 Aggregating ensemble predictions")

            # Group predictions by ensemble
            ensemble_groups = {}
            for pred in model_predictions:
                ensemble = pred.ensemble_type
                if ensemble not in ensemble_groups:
                    ensemble_groups[ensemble] = []
                ensemble_groups[ensemble].append(pred)

            ensemble_predictions = []

            for ensemble_name, predictions in ensemble_groups.items():
                # Calculate ensemble metrics
                pred_values = [p.prediction for p in predictions]
                confidences = [p.confidence for p in predictions]

                # Weighted average
                weights = [c / sum(confidences) for c in confidences]
                final_prediction = sum(
                    p * w for p, w in zip(pred_values, weights, strict=False)
                )
                ensemble_confidence = sum(confidences) / len(confidences)

                # Consensus metrics
                prediction_variance = sum(
                    (p - final_prediction) ** 2 for p in pred_values
                ) / len(pred_values)
                consensus_level = 1.0 - (
                    prediction_variance / (abs(final_prediction) + 0.1)
                )
                disagreement_score = prediction_variance

                # Create ensemble prediction
                ensemble_pred = EnsemblePrediction(
                    ensemble_id=f"{ensemble_name}_{int(datetime.now().timestamp())}",
                    regime_type=RegimeType.BULL_TREND,  # Example
                    individual_predictions=predictions,
                    aggregated_prediction=final_prediction,
                    ensemble_confidence=ensemble_confidence,
                    consensus_level=max(0, min(1, consensus_level)),
                    disagreement_score=disagreement_score,
                    weighted_average=final_prediction,
                    voting_result="buy"
                    if final_prediction > 0.1
                    else "sell"
                    if final_prediction < -0.1
                    else "hold",
                )

                ensemble_predictions.append(ensemble_pred)

                # Track ensemble performance
                if self.ml_tracker:
                    await self.ml_tracker.track_ensemble_performance(
                        ensemble_name=ensemble_name,
                        individual_predictions=[],  # Simplified for example
                        final_prediction=final_prediction,
                        ensemble_confidence=ensemble_confidence,
                        aggregation_method="weighted_average",
                    )

            self.logger.info(
                f"✅ Aggregated {len(ensemble_predictions)} ensemble predictions",
            )
            return ensemble_predictions

        except Exception:
            self.print(failed("Failed to aggregate ensemble predictions: {e}"))
            return []

    async def _make_trading_decision(
        self,
        symbol: str,
        timeframe_features: dict[str, MultiTimeframeFeatures],
        regime_data: dict[str, Any],
        sr_data: dict[str, list[dict[str, Any]]],
        ensemble_predictions: list[EnsemblePrediction],
    ) -> TradeDecisionContext:
        """Make comprehensive trading decision with full context."""
        try:
            self.logger.info(f"⚖️ Making trading decision for {symbol}")

            # Calculate final prediction (ensemble of ensembles)
            if ensemble_predictions:
                final_prediction = sum(
                    e.aggregated_prediction for e in ensemble_predictions
                ) / len(ensemble_predictions)
                final_confidence = sum(
                    e.ensemble_confidence for e in ensemble_predictions
                ) / len(ensemble_predictions)
            else:
                final_prediction = 0.0
                final_confidence = 0.0

            # Determine recommended action
            if final_prediction > 0.2 and final_confidence > 0.7:
                recommended_action = TradeAction.ENTER_LONG
            elif final_prediction < -0.2 and final_confidence > 0.7:
                recommended_action = TradeAction.ENTER_SHORT
            else:
                recommended_action = TradeAction.HOLD

            # Get current price from features
            current_price = 3500.0  # Example
            if "1m" in timeframe_features:
                current_price = timeframe_features["1m"].price

            # Create comprehensive decision context
            decision_context = TradeDecisionContext(
                decision_id=f"decision_{symbol}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                symbol=symbol,
                exchange="BINANCE",
                current_price=current_price,
                current_regime=RegimeType.BULL_TREND,  # From regime_data
                regime_confidence=0.8,
                regime_duration_minutes=120.0,
                nearby_sr_levels=[],  # Would convert from sr_data
                timeframe_features=timeframe_features,
                ensemble_predictions=ensemble_predictions,
                final_prediction=final_prediction,
                final_confidence=final_confidence,
                recommended_action=recommended_action,
                position_size=0.1 if recommended_action != TradeAction.HOLD else 0.0,
                entry_price=current_price
                if recommended_action != TradeAction.HOLD
                else None,
                stop_loss=current_price * 0.98
                if recommended_action == TradeAction.ENTER_LONG
                else current_price * 1.02
                if recommended_action == TradeAction.ENTER_SHORT
                else None,
                take_profit=current_price * 1.04
                if recommended_action == TradeAction.ENTER_LONG
                else current_price * 0.96
                if recommended_action == TradeAction.ENTER_SHORT
                else None,
                risk_reward_ratio=2.0,
                risk_score=0.3,
                max_position_risk=0.02,
                portfolio_risk=0.05,
                correlation_risk=0.1,
                order_type="LIMIT",
                leverage=1.0,
                execution_strategy="IMMEDIATE",
                market_session="london",
                sentiment_score=0.1,
            )

            # Record decision with trade monitor
            if self.trade_monitor:
                await self.trade_monitor.record_trade_decision(decision_context)

            self.logger.info(
                f"✅ Trading decision: {recommended_action.value} with confidence {final_confidence:.2f}",
            )
            return decision_context

        except Exception:
            self.print(failed("Failed to make trading decision: {e}"))
            # Return default decision
            return TradeDecisionContext(
                decision_id="error_decision",
                timestamp=datetime.now(),
                symbol=symbol,
                exchange="BINANCE",
                current_price=0.0,
                current_regime=RegimeType.SIDEWAYS,
                regime_confidence=0.0,
                regime_duration_minutes=0.0,
                timeframe_features={},
                ensemble_predictions=[],
                final_prediction=0.0,
                final_confidence=0.0,
                recommended_action=TradeAction.HOLD,
                risk_score=1.0,
                max_position_risk=0.0,
                portfolio_risk=0.0,
                correlation_risk=0.0,
            )

    async def _execute_and_monitor_trade(
        self,
        decision: TradeDecisionContext,
    ) -> TradeExecution:
        """Execute trade and monitor execution."""
        try:
            self.logger.info(
                f"💱 Executing trade decision: {decision.recommended_action.value}",
            )

            if decision.recommended_action == TradeAction.HOLD:
                self.logger.info("No trade execution required (HOLD)")
                return TradeExecution(
                    execution_id="no_execution",
                    decision_id=decision.decision_id,
                    timestamp=datetime.now(),
                    symbol=decision.symbol,
                    side="none",
                    order_type="none",
                    quantity=0.0,
                    status="skipped",
                )

            # Simulate trade execution
            execution = TradeExecution(
                execution_id=f"exec_{decision.decision_id}",
                decision_id=decision.decision_id,
                timestamp=datetime.now(),
                order_id=f"order_{int(datetime.now().timestamp())}",
                symbol=decision.symbol,
                side="buy"
                if decision.recommended_action == TradeAction.ENTER_LONG
                else "sell",
                order_type=decision.order_type or "LIMIT",
                quantity=decision.position_size or 0.1,
                price=decision.entry_price,
                executed_quantity=decision.position_size or 0.1,
                average_execution_price=decision.entry_price or 0.0,
                execution_time_ms=150.0,
                slippage=0.0005,  # 0.05%
                commission=0.001,  # 0.1%
                status="filled",
            )

            # Record execution with trade monitor
            if self.trade_monitor:
                await self.trade_monitor.record_trade_execution(execution)

            self.logger.info(
                f"✅ Trade executed: {execution.side} {execution.executed_quantity} {execution.symbol}",
            )
            return execution

        except Exception as e:
            self.print(failed("Failed to execute trade: {e}"))
            return TradeExecution(
                execution_id="error_execution",
                decision_id=decision.decision_id,
                timestamp=datetime.now(),
                symbol=decision.symbol,
                side="error",
                order_type="error",
                quantity=0.0,
                status="failed",
                error_message=str(e),
            )

    async def _analyze_trade_performance(
        self,
        decision: TradeDecisionContext,
        execution: TradeExecution,
    ) -> None:
        """Analyze trade performance and generate insights."""
        try:
            self.logger.info("📊 Analyzing trade performance")

            if execution.status in ("failed", "skipped"):
                self.logger.info(
                    "Skipping performance analysis for failed/skipped trade",
                )
                return

            # Simulate trade outcome (in practice, this would be done when trade closes)
            await asyncio.sleep(1)  # Simulate time passage

            # Example outcome
            exit_price = execution.average_execution_price * (
                1.02 if execution.side == "buy" else 0.98
            )
            pnl_percentage = (
                (exit_price / execution.average_execution_price - 1)
                if execution.side == "buy"
                else (execution.average_execution_price / exit_price - 1)
            )

            outcome = TradeOutcome(
                trade_id=f"trade_{execution.execution_id}",
                decision_id=decision.decision_id,
                execution_id=execution.execution_id,
                symbol=execution.symbol,
                entry_time=execution.timestamp,
                exit_time=datetime.now(),
                duration_minutes=1.0,  # Example short trade
                entry_price=execution.average_execution_price,
                exit_price=exit_price,
                quantity=execution.executed_quantity,
                pnl_percentage=pnl_percentage,
                pnl_absolute=pnl_percentage
                * execution.average_execution_price
                * execution.executed_quantity,
                max_drawdown=0.005,  # 0.5%
                max_profit=pnl_percentage,
                prediction_accuracy=1.0 if pnl_percentage > 0 else 0.0,
                confidence_calibration=decision.final_confidence,
                what_worked=["Good timing", "Strong conviction"]
                if pnl_percentage > 0
                else [],
                what_failed=["Poor entry", "Market reversal"]
                if pnl_percentage <= 0
                else [],
                improvement_suggestions=[
                    "Consider tighter stops",
                    "Improve entry timing",
                ],
            )

            # Record outcome with trade monitor
            if self.trade_monitor:
                await self.trade_monitor.record_trade_outcome(outcome)

            # Record actual outcome for ML models (for training/validation)
            if self.ml_tracker:
                for ensemble in decision.ensemble_predictions:
                    for pred in ensemble.individual_predictions:
                        # This would be the prediction ID from the ML tracker
                        prediction_id = (
                            f"{pred.model_id}_{int(decision.timestamp.timestamp())}"
                        )
                        await self.ml_tracker.record_actual_outcome(
                            prediction_id,
                            pnl_percentage,
                            outcome.exit_time,
                        )

            self.logger.info(f"✅ Trade analysis complete: PnL {pnl_percentage:.2%}")

        except Exception:
            self.print(failed("Failed to analyze trade performance: {e}"))

    async def _demonstrate_error_detection(self) -> None:
        """Demonstrate error detection and alerting capabilities."""
        try:
            self.logger.info("🚨 Demonstrating error detection capabilities")

            if not self.error_detector:
                self.print(warning("Error detector not available"))
                return

            # Record various types of errors
            await self.error_detector.record_error_event(
                severity=AlertSeverity.WARNING,
                category=ErrorCategory.PERFORMANCE,
                error_message="High latency detected in market data feed",
                component="DataFeed",
                function="get_market_data",
                system_state={"latency_ms": 850, "normal_latency_ms": 150},
            )

            await self.error_detector.record_error_event(
                severity=AlertSeverity.ERROR,
                category=ErrorCategory.MODEL,
                error_message="Model prediction confidence dropped below threshold",
                component="MLPredictor",
                function="predict",
                user_context={
                    "model_id": "xgb_1h",
                    "confidence": 0.35,
                    "threshold": 0.5,
                },
            )

            # Test anomaly detection
            await self.error_detector.detect_anomaly(
                metric_name="prediction_accuracy",
                current_value=0.35,  # Low accuracy
                trend_direction="decreasing",
            )

            await self.error_detector.detect_anomaly(
                metric_name="response_time_ms",
                current_value=2500.0,  # High latency
                related_metrics={"cpu_usage": 85.0, "memory_usage": 92.0},
            )

            # Collect system health metrics
            health_metrics = await self.error_detector.collect_system_health_metrics()
            self.logger.info(
                f"System health: CPU {health_metrics.cpu_usage_percent:.1f}%, Memory {health_metrics.memory_usage_percent:.1f}%",
            )

            self.logger.info("✅ Error detection demonstration completed")

        except Exception:
            self.print(failed("Failed to demonstrate error detection: {e}"))

    async def generate_monitoring_report(self) -> dict[str, Any]:
        """Generate comprehensive monitoring report."""
        try:
            self.logger.info("📈 Generating comprehensive monitoring report")

            report = {
                "timestamp": datetime.now(),
                "trade_monitoring": {},
                "ml_tracking": {},
                "regime_analysis": {},
                "error_detection": {},
                "system_health": {},
            }

            # Trade monitoring statistics
            if self.trade_monitor:
                trade_stats = await self.trade_monitor.get_monitoring_statistics()
                report["trade_monitoring"] = trade_stats

                # Generate detailed trade report
                trade_report = await self.trade_monitor.generate_monitoring_report(
                    days=7,
                )
                report["trade_monitoring"]["detailed_report"] = trade_report

            # ML tracking statistics
            if self.ml_tracker:
                ml_stats = await self.ml_tracker.get_tracking_statistics()
                report["ml_tracking"] = ml_stats

                # Generate model comparison report
                comparison_report = (
                    await self.ml_tracker.generate_model_comparison_report()
                )
                if comparison_report:
                    report["ml_tracking"]["model_comparison"] = comparison_report

            # Regime analysis statistics
            if self.regime_tracker:
                regime_stats = await self.regime_tracker.get_tracking_statistics()
                report["regime_analysis"] = regime_stats

            # Error detection statistics
            if self.error_detector:
                error_stats = await self.error_detector.get_detection_statistics()
                report["error_detection"] = error_stats

            self.logger.info("✅ Comprehensive monitoring report generated")
            return report

        except Exception:
            self.print(failed("Failed to generate monitoring report: {e}"))
            return {}

    async def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        try:
            self.logger.info("🧹 Cleaning up monitoring resources")

            if self.trade_monitor:
                await self.trade_monitor.cleanup()

            if self.ml_tracker:
                await self.ml_tracker.cleanup()

            if self.regime_tracker:
                await self.regime_tracker.cleanup()

            if self.error_detector:
                await self.error_detector.cleanup()

            self.logger.info("✅ Monitoring cleanup completed")

        except Exception:
            self.print(failed("Failed to cleanup monitoring: {e}"))


# Example configuration for comprehensive monitoring
EXAMPLE_MONITORING_CONFIG = {
    "monitoring": {"storage_backend": "sqlite", "database_path": "monitoring.db"},
    "trade_conditions_monitor": {
        "enable_detailed_logging": True,
        "enable_feature_analysis": True,
        "enable_model_tracking": True,
        "storage_backend": "sqlite",
    },
    "enhanced_ml_tracker": {
        "enable_real_time_tracking": True,
        "enable_ensemble_analysis": True,
        "enable_model_comparison": True,
        "performance_window_days": 7,
        "min_predictions_for_analysis": 50,
    },
    "regime_sr_tracker": {
        "enable_regime_tracking": True,
        "enable_sr_tracking": True,
        "enable_performance_analysis": True,
        "regime_detection_interval": 60,
        "sr_update_interval": 300,
        "min_regime_duration": 30,
        "sr_touch_threshold": 0.002,
    },
    "error_detection": {
        "enable_anomaly_detection": True,
        "enable_predictive_alerts": True,
        "enable_email_alerts": False,
        "enable_slack_alerts": False,
        "monitoring_interval": 30,
        "anomaly_sensitivity": 0.95,
        "min_data_points": 100,
        "lookback_window_hours": 24,
        "default_rules": {
            "high_error_rate": {
                "name": "High Error Rate",
                "category": "system",
                "metric_name": "error_rate_percent",
                "condition": "greater_than",
                "threshold": 5.0,
                "severity": "critical",
                "evaluation_window_minutes": 5,
            },
            "low_prediction_accuracy": {
                "name": "Low Prediction Accuracy",
                "category": "model",
                "metric_name": "prediction_accuracy",
                "condition": "less_than",
                "threshold": 0.4,
                "severity": "error",
                "evaluation_window_minutes": 30,
            },
        },
    },
    "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your-email@gmail.com",
        "password": "your-app-password",
        "default_recipients": ["admin@yourcompany.com"],
    },
    "slack": {"webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"},
}


async def main():
    """Main demonstration function."""
    logger = system_logger.getChild("MonitoringDemo")

    try:
        logger.info("🚀 Starting comprehensive monitoring demonstration")

        # Initialize monitoring integration
        monitoring_example = MonitoringIntegrationExample(EXAMPLE_MONITORING_CONFIG)

        if not await monitoring_example.initialize():
            print(failed("Failed to initialize monitoring system"))
            return

        # Demonstrate complete trade analysis workflow
        await monitoring_example.demonstrate_trade_analysis_workflow()

        # Generate comprehensive report
        report = await monitoring_example.generate_monitoring_report()
        logger.info(f"Generated monitoring report with {len(report)} sections")

        # Cleanup
        await monitoring_example.cleanup()

        logger.info("✅ Monitoring demonstration completed successfully")

    except Exception:
        print(failed("Monitoring demonstration failed: {e}"))


if __name__ == "__main__":
    asyncio.run(main())
