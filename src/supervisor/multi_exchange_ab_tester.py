#!/usr/bin/env python3
"""
Multi-Exchange A/B Testing Framework

This module enables simultaneous testing of the same model across different exchanges
to compare performance, validate transfer learning, and identify exchange-specific characteristics.
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
    warning,
)

if TYPE_CHECKING:
    from src.supervisor.exchange_volume_adapter import ExchangeVolumeAdapter


@dataclass
class ABTestConfig:
    """Configuration for A/B testing across exchanges."""

    test_name: str
    model_id: str
    exchanges: list[str]
    test_duration_hours: int = 24
    sample_interval_seconds: int = 60
    min_confidence_threshold: float = 0.6
    max_position_size: float = 0.05
    enable_volume_adaptation: bool = True
    metrics_to_track: list[str] = None

    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = [
                "prediction_accuracy",
                "execution_quality",
                "slippage",
                "spread_costs",
                "market_impact",
                "volume_utilization",
                "profit_loss",
                "sharpe_ratio",
                "max_drawdown",
            ]


@dataclass
class ExchangeTestResult:
    """Results for a single exchange in A/B test."""

    exchange: str
    timestamp: datetime
    prediction: float
    confidence: float
    position_size: float
    executed: bool
    execution_price: float | None = None
    slippage: float | None = None
    spread_cost: float | None = None
    market_impact: float | None = None
    volume_utilization: float | None = None
    profit_loss: float | None = None
    error_message: str | None = None


class MultiExchangeABTester:
    """
    A/B testing framework for comparing model performance across exchanges.

    Features:
    - Simultaneous testing on multiple exchanges
    - Real-time performance comparison
    - Volume adaptation integration
    - Comprehensive metrics tracking
    - Statistical significance testing
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MultiExchangeABTester")

        # A/B test state
        self.current_test: ABTestConfig | None = None
        self.test_results: dict[str, list[ExchangeTestResult]] = {}
        self.test_start_time: datetime | None = None
        self.test_end_time: datetime | None = None
        self.is_running: bool = False

        # Volume adapter for exchange-specific adjustments
        self.volume_adapter: ExchangeVolumeAdapter | None = None

        # Performance tracking
        self.performance_metrics: dict[str, dict[str, Any]] = {}
        self.statistical_tests: dict[str, dict[str, Any]] = {}

        # Configuration
        self.ab_config: dict[str, Any] = self.config.get("multi_exchange_ab_tester", {})
        self.max_concurrent_tests: int = self.ab_config.get("max_concurrent_tests", 3)
        self.result_storage_path: str = self.ab_config.get(
            "result_storage_path",
            "ab_test_results",
        )
        self.enable_real_time_monitoring: bool = self.ab_config.get(
            "enable_real_time_monitoring",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid A/B test configuration"),
            AttributeError: (False, "Missing required A/B test parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="multi-exchange A/B tester initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the multi-exchange A/B tester."""
        try:
            self.logger.info("Initializing Multi-Exchange A/B Tester...")

            # Load configuration
            await self._load_ab_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.print(
                    invalid("Invalid configuration for multi-exchange A/B tester"),
                )
                return False

            # Initialize volume adapter
            await self._initialize_volume_adapter()

            # Initialize result storage
            await self._initialize_result_storage()

            self.logger.info(
                "✅ Multi-Exchange A/B Tester initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Multi-Exchange A/B Tester initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="A/B test configuration loading",
    )
    async def _load_ab_configuration(self) -> None:
        """Load A/B test configuration."""
        try:
            # Set defaults
            self.ab_config.setdefault("max_concurrent_tests", 3)
            self.ab_config.setdefault("result_storage_path", "ab_test_results")
            self.ab_config.setdefault("enable_real_time_monitoring", True)
            self.ab_config.setdefault("min_sample_size", 100)
            self.ab_config.setdefault("confidence_level", 0.95)

            self.logger.info("A/B test configuration loaded successfully")

        except Exception:
            self.print(error("Error loading A/B test configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate A/B test configuration."""
        try:
            if self.max_concurrent_tests <= 0:
                self.print(invalid("Invalid max concurrent tests"))
                return False

            if not self.result_storage_path:
                self.print(error("No result storage path specified"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="volume adapter initialization",
    )
    async def _initialize_volume_adapter(self) -> None:
        """Initialize volume adapter for exchange-specific adjustments."""
        try:
            from src.supervisor.exchange_volume_adapter import (
                setup_exchange_volume_adapter,
            )

            self.volume_adapter = await setup_exchange_volume_adapter(self.config)
            if self.volume_adapter:
                self.logger.info("Volume adapter initialized successfully")
            else:
                self.logger.warning(
                    "Volume adapter initialization failed, continuing without volume adaptation",
                )

        except Exception:
            self.print(initialization_error("Error initializing volume adapter: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="result storage initialization",
    )
    async def _initialize_result_storage(self) -> None:
        """Initialize result storage directory."""
        try:
            import os

            # Create result storage directory
            os.makedirs(self.result_storage_path, exist_ok=True)
            self.logger.info(
                f"Result storage initialized at {self.result_storage_path}",
            )

        except Exception:
            self.print(initialization_error("Error initializing result storage: {e}"))

    async def start_ab_test(self, test_config: ABTestConfig) -> bool:
        """
        Start a new A/B test across multiple exchanges.

        Args:
            test_config: A/B test configuration

        Returns:
            bool: True if test started successfully, False otherwise
        """
        try:
            if self.is_running:
                self.print(error("A/B test already running"))
                return False

            # Validate test configuration
            if not self._validate_test_config(test_config):
                return False

            # Initialize test state
            self.current_test = test_config
            self.test_start_time = datetime.now()
            self.test_end_time = self.test_start_time + timedelta(
                hours=test_config.test_duration_hours,
            )

            # Initialize results for each exchange
            for exchange in test_config.exchanges:
                self.test_results[exchange] = []

            # Initialize performance metrics
            for exchange in test_config.exchanges:
                self.performance_metrics[exchange] = {
                    "total_predictions": 0,
                    "total_executions": 0,
                    "total_profit_loss": 0.0,
                    "accuracy": 0.0,
                    "avg_slippage": 0.0,
                    "avg_spread_cost": 0.0,
                    "avg_market_impact": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                }

            self.is_running = True
            self.logger.info(
                f"🚀 Started A/B test '{test_config.test_name}' across {len(test_config.exchanges)} exchanges",
            )

            # Start monitoring
            if self.enable_real_time_monitoring:
                asyncio.create_task(self._monitor_test_progress())

            return True

        except Exception:
            self.print(error("Error starting A/B test: {e}"))
            return False

    def _validate_test_config(self, test_config: ABTestConfig) -> bool:
        """Validate A/B test configuration."""
        try:
            if not test_config.exchanges or len(test_config.exchanges) < 2:
                self.print(error("A/B test requires at least 2 exchanges"))
                return False

            if test_config.test_duration_hours <= 0:
                self.print(invalid("Invalid test duration"))
                return False

            if test_config.sample_interval_seconds <= 0:
                self.print(invalid("Invalid sample interval"))
                return False

            if not (0 <= test_config.min_confidence_threshold <= 1):
                self.print(invalid("Invalid confidence threshold"))
                return False

            return True

        except Exception:
            self.print(error("Error validating test config: {e}"))
            return False

    async def process_model_prediction(
        self,
        exchange: str,
        prediction: float,
        confidence: float,
        market_data: dict[str, Any],
        model_id: str = None,
    ) -> ExchangeTestResult:
        """
        Process a model prediction for a specific exchange.

        Args:
            exchange: Exchange name
            prediction: Model prediction
            confidence: Model confidence score
            market_data: Current market data
            model_id: Model identifier

        Returns:
            ExchangeTestResult: Test result for this prediction
        """
        try:
            if not self.is_running or self.current_test is None:
                msg = "No A/B test currently running"
                raise ValueError(msg)

            if exchange not in self.current_test.exchanges:
                msg = f"Exchange {exchange} not in current test"
                raise ValueError(msg)

            # Get base position size from model
            base_position_size = self.current_test.max_position_size

            # Apply volume adaptation if enabled
            adjusted_position_size = base_position_size
            adjusted_confidence = confidence

            if self.volume_adapter and self.current_test.enable_volume_adaptation:
                current_volume = market_data.get("volume", 0)

                adjusted_position_size = (
                    self.volume_adapter.calculate_position_size_adjustment(
                        exchange=exchange,
                        base_position_size=base_position_size,
                        current_volume=current_volume,
                        confidence_score=confidence,
                    )
                )

                adjusted_confidence = self.volume_adapter.adjust_model_confidence(
                    exchange=exchange,
                    base_confidence=confidence,
                )

            # Check if trade should be executed
            should_execute = False
            execution_price = None
            slippage = None
            spread_cost = None
            market_impact = None
            volume_utilization = None
            profit_loss = None
            error_message = None

            if adjusted_confidence >= self.current_test.min_confidence_threshold:
                if self.volume_adapter:
                    should_execute, reason = self.volume_adapter.should_execute_trade(
                        exchange=exchange,
                        position_size=adjusted_position_size,
                        current_volume=market_data.get("volume", 0),
                    )

                    if not should_execute:
                        error_message = reason
                else:
                    should_execute = True

            # Simulate execution (in real implementation, this would execute actual trades)
            if should_execute:
                execution_price = market_data.get("price", 0)
                slippage = self._simulate_slippage(
                    exchange,
                    adjusted_position_size,
                    market_data,
                )
                spread_cost = self._simulate_spread_cost(exchange, market_data)
                market_impact = self._simulate_market_impact(
                    exchange,
                    adjusted_position_size,
                    market_data,
                )
                volume_utilization = self._calculate_volume_utilization(
                    adjusted_position_size,
                    market_data,
                )
                profit_loss = self._simulate_profit_loss(
                    prediction,
                    execution_price,
                    adjusted_position_size,
                )

            # Create test result
            result = ExchangeTestResult(
                exchange=exchange,
                timestamp=datetime.now(),
                prediction=prediction,
                confidence=adjusted_confidence,
                position_size=adjusted_position_size,
                executed=should_execute,
                execution_price=execution_price,
                slippage=slippage,
                spread_cost=spread_cost,
                market_impact=market_impact,
                volume_utilization=volume_utilization,
                profit_loss=profit_loss,
                error_message=error_message,
            )

            # Store result
            self.test_results[exchange].append(result)

            # Update performance metrics
            await self._update_performance_metrics(exchange, result)

            self.logger.info(
                f"📊 {exchange}: prediction={prediction:.4f}, "
                f"confidence={adjusted_confidence:.3f}, "
                f"position={adjusted_position_size:.4f}, "
                f"executed={should_execute}",
            )

            return result

        except Exception as e:
            self.logger.exception(
                f"Error processing model prediction for {exchange}: {e}",
            )
            return ExchangeTestResult(
                exchange=exchange,
                timestamp=datetime.now(),
                prediction=0.0,
                confidence=0.0,
                position_size=0.0,
                executed=False,
                error_message=str(e),
            )

    def _simulate_slippage(
        self,
        exchange: str,
        position_size: float,
        market_data: dict[str, Any],
    ) -> float:
        """Simulate slippage based on exchange characteristics."""
        try:
            base_slippage = 0.001  # 0.1% base slippage

            if self.volume_adapter:
                return self.volume_adapter.calculate_slippage_adjustment(
                    exchange,
                    base_slippage,
                )

            # Simple simulation based on exchange
            exchange_multipliers = {"BINANCE": 1.0, "MEXC": 3.0, "GATEIO": 3.5}

            multiplier = exchange_multipliers.get(exchange.upper(), 2.0)
            return base_slippage * multiplier

        except Exception:
            self.print(error("Error simulating slippage: {e}"))
            return 0.002  # Conservative fallback

    def _simulate_spread_cost(
        self,
        exchange: str,
        market_data: dict[str, Any],
    ) -> float:
        """Simulate spread cost based on exchange characteristics."""
        try:
            base_spread = 0.0005  # 0.05% base spread

            if self.volume_adapter:
                return self.volume_adapter.calculate_spread_adjustment(
                    exchange,
                    base_spread,
                )

            # Simple simulation based on exchange
            exchange_multipliers = {"BINANCE": 1.0, "MEXC": 2.5, "GATEIO": 3.0}

            multiplier = exchange_multipliers.get(exchange.upper(), 2.0)
            return base_spread * multiplier

        except Exception:
            self.print(error("Error simulating spread cost: {e}"))
            return 0.001  # Conservative fallback

    def _simulate_market_impact(
        self,
        exchange: str,
        position_size: float,
        market_data: dict[str, Any],
    ) -> float:
        """Simulate market impact based on position size and exchange."""
        try:
            volume = market_data.get("volume", 1000000)
            impact_ratio = position_size / volume

            # Exchange-specific impact multipliers
            impact_multipliers = {"BINANCE": 1.0, "MEXC": 5.0, "GATEIO": 8.0}

            multiplier = impact_multipliers.get(exchange.upper(), 3.0)
            return impact_ratio * multiplier

        except Exception:
            self.print(error("Error simulating market impact: {e}"))
            return 0.001  # Conservative fallback

    def _calculate_volume_utilization(
        self,
        position_size: float,
        market_data: dict[str, Any],
    ) -> float:
        """Calculate volume utilization ratio."""
        try:
            volume = market_data.get("volume", 1000000)
            return position_size / volume if volume > 0 else 0.0

        except Exception:
            self.print(error("Error calculating volume utilization: {e}"))
            return 0.0

    def _simulate_profit_loss(
        self,
        prediction: float,
        execution_price: float,
        position_size: float,
    ) -> float:
        """Simulate profit/loss based on prediction accuracy."""
        try:
            # Simple simulation: if prediction is positive, assume some profit
            # In real implementation, this would track actual P&L
            if prediction > 0:
                return position_size * prediction * 0.1  # 10% of prediction
            return position_size * prediction * 0.1  # Loss

        except Exception:
            self.print(error("Error simulating profit/loss: {e}"))
            return 0.0

    async def _update_performance_metrics(
        self,
        exchange: str,
        result: ExchangeTestResult,
    ) -> None:
        """Update performance metrics for an exchange."""
        try:
            metrics = self.performance_metrics[exchange]

            # Update basic counts
            metrics["total_predictions"] += 1
            if result.executed:
                metrics["total_executions"] += 1

            # Update running totals
            if result.profit_loss is not None:
                metrics["total_profit_loss"] += result.profit_loss

            if result.slippage is not None:
                # Update average slippage
                current_avg = metrics["avg_slippage"]
                count = metrics["total_executions"]
                metrics["avg_slippage"] = (
                    current_avg * (count - 1) + result.slippage
                ) / count

            if result.spread_cost is not None:
                # Update average spread cost
                current_avg = metrics["avg_spread_cost"]
                count = metrics["total_executions"]
                metrics["avg_spread_cost"] = (
                    current_avg * (count - 1) + result.spread_cost
                ) / count

            if result.market_impact is not None:
                # Update average market impact
                current_avg = metrics["avg_market_impact"]
                count = metrics["total_executions"]
                metrics["avg_market_impact"] = (
                    current_avg * (count - 1) + result.market_impact
                ) / count

            # Calculate accuracy (simplified)
            if result.executed and result.profit_loss is not None:
                if result.profit_loss > 0:
                    metrics["accuracy"] = (
                        metrics["accuracy"] * (metrics["total_executions"] - 1) + 1
                    ) / metrics["total_executions"]
                else:
                    metrics["accuracy"] = (
                        metrics["accuracy"] * (metrics["total_executions"] - 1) + 0
                    ) / metrics["total_executions"]

        except Exception:
            self.print(error("Error updating performance metrics: {e}"))

    async def _monitor_test_progress(self) -> None:
        """Monitor A/B test progress in real-time."""
        try:
            while self.is_running and self.current_test:
                # Check if test should end
                if datetime.now() >= self.test_end_time:
                    await self.stop_ab_test()
                    break

                # Log progress
                await self._log_test_progress()

                # Wait for next check
                await asyncio.sleep(60)  # Check every minute

        except Exception:
            self.print(error("Error monitoring test progress: {e}"))

    async def _log_test_progress(self) -> None:
        """Log current test progress."""
        try:
            if not self.current_test:
                return

            elapsed = datetime.now() - self.test_start_time
            total_duration = timedelta(hours=self.current_test.test_duration_hours)
            progress = (elapsed / total_duration) * 100

            self.logger.info(f"📈 A/B Test Progress: {progress:.1f}% complete")

            # Log performance summary
            for exchange in self.current_test.exchanges:
                if exchange in self.performance_metrics:
                    metrics = self.performance_metrics[exchange]
                    self.logger.info(
                        f"📊 {exchange}: "
                        f"Predictions={metrics['total_predictions']}, "
                        f"Executions={metrics['total_executions']}, "
                        f"P&L={metrics['total_profit_loss']:.4f}, "
                        f"Accuracy={metrics['accuracy']:.3f}",
                    )

        except Exception:
            self.print(error("Error logging test progress: {e}"))

    async def stop_ab_test(self) -> bool:
        """Stop the current A/B test and generate results."""
        try:
            if not self.is_running:
                self.print(warning("No A/B test currently running"))
                return False

            self.is_running = False
            self.test_end_time = datetime.now()

            self.logger.info("🛑 Stopping A/B test...")

            # Generate final results
            await self._generate_final_results()

            # Save results
            await self._save_test_results()

            # Perform statistical analysis
            await self._perform_statistical_analysis()

            self.logger.info("✅ A/B test completed successfully")
            return True

        except Exception:
            self.print(error("Error stopping A/B test: {e}"))
            return False

    async def _generate_final_results(self) -> None:
        """Generate final test results and summary."""
        try:
            if not self.current_test:
                return

            self.logger.info("📊 Generating final A/B test results...")

            # Calculate final metrics for each exchange
            for exchange in self.current_test.exchanges:
                if exchange in self.test_results:
                    results = self.test_results[exchange]

                    if results:
                        # Calculate additional metrics
                        profits = [
                            r.profit_loss for r in results if r.profit_loss is not None
                        ]
                        if profits:
                            self.performance_metrics[exchange]["sharpe_ratio"] = (
                                self._calculate_sharpe_ratio(profits)
                            )
                            self.performance_metrics[exchange]["max_drawdown"] = (
                                self._calculate_max_drawdown(profits)
                            )

            # Generate comparison summary
            await self._generate_comparison_summary()

        except Exception:
            self.print(error("Error generating final results: {e}"))

    def _calculate_sharpe_ratio(self, returns: list[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        try:
            if not returns:
                return 0.0

            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return == 0:
                return 0.0

            return mean_return / std_return

        except Exception:
            self.print(error("Error calculating Sharpe ratio: {e}"))
            return 0.0

    def _calculate_max_drawdown(self, returns: list[float]) -> float:
        """Calculate maximum drawdown from returns."""
        try:
            if not returns:
                return 0.0

            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max

            return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        except Exception:
            self.print(error("Error calculating max drawdown: {e}"))
            return 0.0

    async def _generate_comparison_summary(self) -> None:
        """Generate comparison summary across exchanges."""
        try:
            if not self.current_test:
                return

            self.logger.info("📈 Generating exchange comparison summary...")

            # Create comparison DataFrame
            comparison_data = []
            for exchange in self.current_test.exchanges:
                if exchange in self.performance_metrics:
                    metrics = self.performance_metrics[exchange]
                    comparison_data.append(
                        {
                            "exchange": exchange,
                            "total_predictions": metrics["total_predictions"],
                            "total_executions": metrics["total_executions"],
                            "execution_rate": metrics["total_executions"]
                            / max(metrics["total_predictions"], 1),
                            "total_profit_loss": metrics["total_profit_loss"],
                            "accuracy": metrics["accuracy"],
                            "avg_slippage": metrics["avg_slippage"],
                            "avg_spread_cost": metrics["avg_spread_cost"],
                            "avg_market_impact": metrics["avg_market_impact"],
                            "sharpe_ratio": metrics["sharpe_ratio"],
                            "max_drawdown": metrics["max_drawdown"],
                        },
                    )

            if comparison_data:
                df = pd.DataFrame(comparison_data)

                # Log summary
                self.logger.info("🏆 Exchange Performance Summary:")
                for _, row in df.iterrows():
                    self.logger.info(
                        f"  {row['exchange']}: "
                        f"P&L={row['total_profit_loss']:.4f}, "
                        f"Accuracy={row['accuracy']:.3f}, "
                        f"Sharpe={row['sharpe_ratio']:.3f}, "
                        f"ExecRate={row['execution_rate']:.3f}",
                    )

                # Find best performing exchange
                best_pnl = df.loc[df["total_profit_loss"].idxmax()]
                best_sharpe = df.loc[df["sharpe_ratio"].idxmax()]
                best_accuracy = df.loc[df["accuracy"].idxmax()]

                self.logger.info(
                    f"🥇 Best P&L: {best_pnl['exchange']} ({best_pnl['total_profit_loss']:.4f})",
                )
                self.logger.info(
                    f"📈 Best Sharpe: {best_sharpe['exchange']} ({best_sharpe['sharpe_ratio']:.3f})",
                )
                self.logger.info(
                    f"🎯 Best Accuracy: {best_accuracy['exchange']} ({best_accuracy['accuracy']:.3f})",
                )

        except Exception:
            self.print(error("Error generating comparison summary: {e}"))

    async def _perform_statistical_analysis(self) -> None:
        """Perform statistical analysis on test results."""
        try:
            if not self.current_test or len(self.current_test.exchanges) < 2:
                return

            self.logger.info("📊 Performing statistical analysis...")

            # Compare performance between exchanges
            exchanges = list(self.current_test.exchanges)

            for i, exchange1 in enumerate(exchanges):
                for exchange2 in exchanges[i + 1 :]:
                    if (
                        exchange1 in self.performance_metrics
                        and exchange2 in self.performance_metrics
                    ):
                        await self._compare_exchanges_statistically(
                            exchange1,
                            exchange2,
                        )

        except Exception:
            self.print(error("Error performing statistical analysis: {e}"))

    async def _compare_exchanges_statistically(
        self,
        exchange1: str,
        exchange2: str,
    ) -> None:
        """Compare two exchanges statistically."""
        try:
            metrics1 = self.performance_metrics[exchange1]
            metrics2 = self.performance_metrics[exchange2]

            # Compare key metrics
            pnl_diff = metrics1["total_profit_loss"] - metrics2["total_profit_loss"]
            accuracy_diff = metrics1["accuracy"] - metrics2["accuracy"]
            sharpe_diff = metrics1["sharpe_ratio"] - metrics2["sharpe_ratio"]

            self.logger.info(f"📊 {exchange1} vs {exchange2}:")
            self.logger.info(f"  P&L Difference: {pnl_diff:.4f}")
            self.logger.info(f"  Accuracy Difference: {accuracy_diff:.3f}")
            self.logger.info(f"  Sharpe Difference: {sharpe_diff:.3f}")

            # Determine winner
            if pnl_diff > 0:
                self.logger.info(f"  🏆 {exchange1} has better P&L")
            elif pnl_diff < 0:
                self.logger.info(f"  🏆 {exchange2} has better P&L")
            else:
                self.logger.info(f"  🤝 {exchange1} and {exchange2} have equal P&L")

        except Exception:
            self.print(error("Error comparing exchanges: {e}"))

    async def _save_test_results(self) -> None:
        """Save test results to file."""
        try:
            if not self.current_test:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.result_storage_path}/ab_test_{self.current_test.test_name}_{timestamp}.json"

            # Prepare data for saving
            save_data = {
                "test_config": asdict(self.current_test),
                "test_start_time": self.test_start_time.isoformat()
                if self.test_start_time
                else None,
                "test_end_time": self.test_end_time.isoformat()
                if self.test_end_time
                else None,
                "performance_metrics": self.performance_metrics,
                "results": {
                    exchange: [asdict(result) for result in results]
                    for exchange, results in self.test_results.items()
                },
            }

            # Save to file
            with open(filename, "w") as f:
                json.dump(save_data, f, indent=2, default=str)

            self.logger.info(f"💾 Test results saved to {filename}")

        except Exception:
            self.print(error("Error saving test results: {e}"))

    def get_test_status(self) -> dict[str, Any]:
        """Get current test status."""
        try:
            return {
                "is_running": self.is_running,
                "current_test": asdict(self.current_test)
                if self.current_test
                else None,
                "test_start_time": self.test_start_time.isoformat()
                if self.test_start_time
                else None,
                "test_end_time": self.test_end_time.isoformat()
                if self.test_end_time
                else None,
                "performance_metrics": self.performance_metrics,
                "total_results": {
                    exchange: len(results)
                    for exchange, results in self.test_results.items()
                },
            }

        except Exception as e:
            self.print(error("Error getting test status: {e}"))
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.logger.info("Cleaning up Multi-Exchange A/B Tester...")

            if self.is_running:
                await self.stop_ab_test()

            # Clear results
            self.test_results.clear()
            self.performance_metrics.clear()
            self.statistical_tests.clear()

            # Cleanup volume adapter
            if self.volume_adapter:
                await self.volume_adapter.cleanup()

            self.logger.info("✅ Multi-Exchange A/B Tester cleanup completed")

        except Exception:
            self.print(error("Error during cleanup: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="multi-exchange A/B tester setup",
)
async def setup_multi_exchange_ab_tester(
    config: dict[str, Any] | None = None,
) -> MultiExchangeABTester | None:
    """Setup multi-exchange A/B tester."""
    try:
        if config is None:
            config = {}

        tester = MultiExchangeABTester(config)
        if await tester.initialize():
            return tester
        return None

    except Exception:
        system_logger.exception(error("Error setting up multi-exchange A/B tester: {e}"))
        return None
