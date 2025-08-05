#!/usr/bin/env python3
"""
Exchange A/B Testing Framework

Simplified A/B testing framework for comparing model performance across exchanges.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.utils.logger import system_logger


@dataclass
class ABTestConfig:
    """A/B test configuration."""

    test_name: str
    model_id: str
    exchanges: list[str]
    test_duration_hours: int = 24
    sample_interval_seconds: int = 60
    min_confidence_threshold: float = 0.6
    max_position_size: float = 0.05


@dataclass
class ExchangeResult:
    """Single exchange test result."""

    exchange: str
    timestamp: datetime
    prediction: float
    confidence: float
    position_size: float
    executed: bool
    profit_loss: float | None = None
    slippage: float | None = None
    error_message: str | None = None


class ExchangeABTester:
    """A/B testing framework for comparing model performance across exchanges."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("ExchangeABTester")

        # Test state
        self.current_test: ABTestConfig | None = None
        self.test_results: dict[str, list[ExchangeResult]] = {}
        self.test_start_time: datetime | None = None
        self.is_running: bool = False

        # Performance tracking
        self.performance_metrics: dict[str, dict[str, Any]] = {}

        # Configuration
        self.ab_config = self.config.get("exchange_ab_tester", {})
        self.result_storage_path = self.ab_config.get(
            "result_storage_path",
            "ab_test_results",
        )

    async def initialize(self) -> bool:
        """Initialize the A/B tester."""
        try:
            self.logger.info("Initializing Exchange A/B Tester...")

            # Create result storage directory
            import os

            os.makedirs(self.result_storage_path, exist_ok=True)

            self.logger.info("✅ Exchange A/B Tester initialization completed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Exchange A/B Tester initialization failed: {e}")
            return False

    async def start_ab_test(self, test_config: ABTestConfig) -> bool:
        """Start a new A/B test."""
        try:
            if self.is_running:
                self.logger.error("A/B test already running")
                return False

            # Validate test configuration
            if len(test_config.exchanges) < 2:
                self.logger.error("A/B test requires at least 2 exchanges")
                return False

            # Initialize test state
            self.current_test = test_config
            self.test_start_time = datetime.now()

            # Initialize results for each exchange
            for exchange in test_config.exchanges:
                self.test_results[exchange] = []
                self.performance_metrics[exchange] = {
                    "total_predictions": 0,
                    "total_executions": 0,
                    "total_profit_loss": 0.0,
                    "accuracy": 0.0,
                    "avg_slippage": 0.0,
                }

            self.is_running = True
            self.logger.info(
                f"🚀 Started A/B test '{test_config.test_name}' across {len(test_config.exchanges)} exchanges",
            )

            return True

        except Exception as e:
            self.logger.error(f"Error starting A/B test: {e}")
            return False

    async def process_prediction(
        self,
        exchange: str,
        prediction: float,
        confidence: float,
        market_data: dict[str, Any],
    ) -> ExchangeResult:
        """Process a model prediction for a specific exchange."""
        try:
            if not self.is_running or self.current_test is None:
                raise ValueError("No A/B test currently running")

            if exchange not in self.current_test.exchanges:
                raise ValueError(f"Exchange {exchange} not in current test")

            # Determine if trade should be executed
            should_execute = confidence >= self.current_test.min_confidence_threshold

            # Calculate position size (with exchange-specific adjustments)
            position_size = self.current_test.max_position_size
            if exchange.upper() in ["MEXC", "GATEIO"]:
                position_size *= 0.4  # Reduce position size for smaller exchanges

            # Simulate execution results
            profit_loss = None
            slippage = None

            if should_execute:
                # Simulate slippage based on exchange
                slippage_multipliers = {"BINANCE": 1.0, "MEXC": 3.0, "GATEIO": 3.5}
                base_slippage = 0.001
                slippage = base_slippage * slippage_multipliers.get(
                    exchange.upper(),
                    2.0,
                )

                # Simulate profit/loss
                if prediction > 0:
                    profit_loss = position_size * prediction * 0.1
                else:
                    profit_loss = position_size * prediction * 0.1

            # Create result
            result = ExchangeResult(
                exchange=exchange,
                timestamp=datetime.now(),
                prediction=prediction,
                confidence=confidence,
                position_size=position_size,
                executed=should_execute,
                profit_loss=profit_loss,
                slippage=slippage,
            )

            # Store result
            self.test_results[exchange].append(result)

            # Update performance metrics
            await self._update_metrics(exchange, result)

            self.logger.info(
                f"📊 {exchange}: prediction={prediction:.4f}, "
                f"confidence={confidence:.3f}, executed={should_execute}",
            )

            return result

        except Exception as e:
            self.logger.error(f"Error processing prediction for {exchange}: {e}")
            return ExchangeResult(
                exchange=exchange,
                timestamp=datetime.now(),
                prediction=0.0,
                confidence=0.0,
                position_size=0.0,
                executed=False,
                error_message=str(e),
            )

    async def _update_metrics(self, exchange: str, result: ExchangeResult) -> None:
        """Update performance metrics for an exchange."""
        try:
            metrics = self.performance_metrics[exchange]

            metrics["total_predictions"] += 1
            if result.executed:
                metrics["total_executions"] += 1

                if result.profit_loss is not None:
                    metrics["total_profit_loss"] += result.profit_loss

                if result.slippage is not None:
                    # Update average slippage
                    current_avg = metrics["avg_slippage"]
                    count = metrics["total_executions"]
                    metrics["avg_slippage"] = (
                        current_avg * (count - 1) + result.slippage
                    ) / count

                # Update accuracy (simplified)
                if result.profit_loss is not None:
                    if result.profit_loss > 0:
                        metrics["accuracy"] = (
                            metrics["accuracy"] * (metrics["total_executions"] - 1) + 1
                        ) / metrics["total_executions"]
                    else:
                        metrics["accuracy"] = (
                            metrics["accuracy"] * (metrics["total_executions"] - 1) + 0
                        ) / metrics["total_executions"]

        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    async def stop_ab_test(self) -> bool:
        """Stop the current A/B test and generate results."""
        try:
            if not self.is_running:
                return False

            self.is_running = False
            self.logger.info("🛑 Stopping A/B test...")

            # Generate final results
            await self._generate_results()

            # Save results
            await self._save_results()

            self.logger.info("✅ A/B test completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping A/B test: {e}")
            return False

    async def _generate_results(self) -> None:
        """Generate final test results."""
        try:
            if not self.current_test:
                return

            self.logger.info("📊 Generating A/B test results...")

            # Create comparison summary
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
                        f"ExecRate={row['execution_rate']:.3f}",
                    )

                # Find best performing exchange
                best_pnl = df.loc[df["total_profit_loss"].idxmax()]
                best_accuracy = df.loc[df["accuracy"].idxmax()]

                self.logger.info(
                    f"🥇 Best P&L: {best_pnl['exchange']} ({best_pnl['total_profit_loss']:.4f})",
                )
                self.logger.info(
                    f"🎯 Best Accuracy: {best_accuracy['exchange']} ({best_accuracy['accuracy']:.3f})",
                )

        except Exception as e:
            self.logger.error(f"Error generating results: {e}")

    async def _save_results(self) -> None:
        """Save test results to file."""
        try:
            if not self.current_test:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.result_storage_path}/ab_test_{self.current_test.test_name}_{timestamp}.json"

            save_data = {
                "test_config": asdict(self.current_test),
                "test_start_time": self.test_start_time.isoformat()
                if self.test_start_time
                else None,
                "performance_metrics": self.performance_metrics,
                "results": {
                    exchange: [asdict(result) for result in results]
                    for exchange, results in self.test_results.items()
                },
            }

            with open(filename, "w") as f:
                json.dump(save_data, f, indent=2, default=str)

            self.logger.info(f"💾 Test results saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

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
                "performance_metrics": self.performance_metrics,
                "total_results": {
                    exchange: len(results)
                    for exchange, results in self.test_results.items()
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting test status: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.is_running:
                await self.stop_ab_test()

            self.test_results.clear()
            self.performance_metrics.clear()
            self.logger.info("✅ Exchange A/B Tester cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def setup_exchange_ab_tester(
    config: dict[str, Any] = None,
) -> ExchangeABTester | None:
    """Setup exchange A/B tester."""
    try:
        if config is None:
            config = {}

        tester = ExchangeABTester(config)
        if await tester.initialize():
            return tester
        return None

    except Exception as e:
        system_logger.error(f"Failed to setup exchange A/B tester: {e}")
        return None
