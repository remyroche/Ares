# src/supervisor/risk_allocator.py

import asyncio
from datetime import datetime
from typing import Any

import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    invalid,
)


class RiskAllocator:
    """
    Enhanced Risk Allocator component with DI, type hints, and robust error handling.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("RiskAllocator")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.risk_config: dict[str, Any] = self.config.get("risk_allocator", {})
        self.allocation_interval: int = self.risk_config.get("allocation_interval", 60)
        self.max_history: int = self.risk_config.get("max_history", 100)
        self.risk_allocations: dict[str, Any] = {}
        self.risk_limits: dict[str, Any] = {}

        # VaR and ES monitoring
        self.var_config: dict[str, Any] = self.risk_config.get("var_monitoring", {})
        self.var_confidence_level: float = self.var_config.get("confidence_level", 0.95)
        self.var_time_horizon: int = self.var_config.get("time_horizon", 1)  # days
        self.es_confidence_level: float = self.var_config.get(
            "es_confidence_level",
            0.95,
        )
        self.var_history: list[dict[str, Any]] = []
        self.es_history: list[dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid risk allocator configuration"),
            AttributeError: (False, "Missing required risk allocator parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="risk allocator initialization",
    )
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Risk Allocator...")
            await self._load_risk_configuration()
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for risk allocator"))
                return False
            self.logger.info("✅ Risk Allocator initialization completed successfully")
            return True
        except Exception:
            self.print(failed("❌ Risk Allocator initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk configuration loading",
    )
    async def _load_risk_configuration(self) -> None:
        try:
            self.risk_config.setdefault("allocation_interval", 60)
            self.risk_config.setdefault("max_history", 100)
            self.allocation_interval = self.risk_config["allocation_interval"]
            self.max_history = self.risk_config["max_history"]
            self.logger.info("Risk allocator configuration loaded successfully")
        except Exception:
            self.print(error("Error loading risk configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.allocation_interval <= 0:
                self.print(invalid("Invalid allocation interval"))
                return False
            if self.max_history <= 0:
                self.print(invalid("Invalid max history"))
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Risk allocator run failed"),
        },
        default_return=False,
        context="risk allocator run",
    )
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info("🚦 Risk Allocator started.")
            while self.is_running:
                await self._perform_risk_allocation()
                await asyncio.sleep(self.allocation_interval)
            return True
        except Exception:
            self.print(error("Error in risk allocator run: {e}"))
            self.is_running = False
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="risk allocation step",
    )
    async def _perform_risk_allocation(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {"timestamp": now, "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            await self._calculate_risk_allocations()
            await self._update_risk_limits()
            self.logger.info(f"Risk allocation tick at {now}")
        except Exception:
            self.print(error("Error in risk allocation step: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="risk allocation calculation",
    )
    async def _calculate_risk_allocations(self) -> None:
        try:
            # Simulate risk allocation calculations
            allocations = {
                "equity_allocation": 0.6,
                "fixed_income_allocation": 0.3,
                "commodities_allocation": 0.1,
                "risk_score": 0.75,
            }
            self.risk_allocations.update(allocations)
            self.logger.info("Risk allocation calculation completed")
        except Exception:
            self.print(error("Error calculating risk allocations: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="risk limits update",
    )
    async def _update_risk_limits(self) -> None:
        try:
            # Update risk limits
            limits = {
                "max_position_size": 0.1,
                "max_drawdown": 0.15,
                "max_leverage": 2.0,
                "stop_loss_threshold": 0.05,
            }
            self.risk_limits.update(limits)
            self.logger.info("Risk limits updated successfully")
        except Exception:
            self.print(error("Error updating risk limits: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="risk allocator stop",
    )
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Risk Allocator...")
        try:
            self.is_running = False
            self.status = {"timestamp": datetime.now().isoformat(), "status": "stopped"}
            self.logger.info("✅ Risk Allocator stopped successfully")
        except Exception:
            self.print(error("Error stopping risk allocator: {e}"))

    def get_status(self) -> dict[str, Any]:
        return self.status.copy()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_risk_allocations(self) -> dict[str, Any]:
        return self.risk_allocations.copy()

    def calculate_var(
        self,
        returns: list[float],
        confidence_level: float = None,
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: List of portfolio returns
            confidence_level: Confidence level for VaR calculation (default: 0.95)

        Returns:
            float: VaR value
        """
        try:
            if not returns:
                return 0.0

            confidence_level = confidence_level or self.var_confidence_level
            percentile = (1 - confidence_level) * 100

            var = np.percentile(returns, percentile)
            return abs(var)  # Return absolute value for risk measurement

        except Exception:
            self.print(error("Error calculating VaR: {e}"))
            return 0.0

    def calculate_expected_shortfall(
        self,
        returns: list[float],
        confidence_level: float = None,
    ) -> float:
        """
        Calculate Expected Shortfall (ES) / Conditional VaR.

        Args:
            returns: List of portfolio returns
            confidence_level: Confidence level for ES calculation (default: 0.95)

        Returns:
            float: Expected Shortfall value
        """
        try:
            if not returns:
                return 0.0

            confidence_level = confidence_level or self.es_confidence_level
            var = self.calculate_var(returns, confidence_level)

            # Calculate ES as the mean of returns below VaR
            returns_array = np.array(returns)
            tail_returns = returns_array[returns_array <= -var]

            if len(tail_returns) == 0:
                return var  # If no tail returns, ES equals VaR

            es = np.mean(tail_returns)
            return abs(es)  # Return absolute value

        except Exception:
            self.print(error("Error calculating Expected Shortfall: {e}"))
            return 0.0

    def calculate_multi_timeframe_var(
        self,
        portfolio_data: dict[str, Any],
    ) -> dict[str, float]:
        """
        Calculate VaR across multiple timeframes.

        Args:
            portfolio_data: Portfolio data containing returns for different timeframes

        Returns:
            dict: VaR values for different timeframes
        """
        try:
            var_results = {}

            # Calculate VaR for different timeframes
            timeframes = ["1d", "1w", "1m", "3m"]

            for timeframe in timeframes:
                returns = portfolio_data.get(f"returns_{timeframe}", [])
                if returns:
                    var = self.calculate_var(returns)
                    var_results[f"var_{timeframe}"] = var

            return var_results

        except Exception:
            self.print(error("Error calculating multi-timeframe VaR: {e}"))
            return {}

    def monitor_risk_limits(
        self,
        current_var: float,
        current_es: float,
    ) -> dict[str, Any]:
        """
        Monitor risk limits and generate alerts.

        Args:
            current_var: Current VaR value
            current_es: Current Expected Shortfall value

        Returns:
            dict: Risk monitoring results and alerts
        """
        try:
            risk_limits = self.risk_config.get("risk_limits", {})
            var_limit = risk_limits.get("max_var", 0.02)  # 2% VaR limit
            es_limit = risk_limits.get("max_es", 0.03)  # 3% ES limit

            alerts = []
            risk_status = "normal"

            # Check VaR limit
            if current_var > var_limit:
                alerts.append(
                    {
                        "type": "var_limit_exceeded",
                        "severity": "high"
                        if current_var > var_limit * 1.5
                        else "medium",
                        "message": f"VaR ({current_var:.4f}) exceeds limit ({var_limit:.4f})",
                        "value": current_var,
                        "limit": var_limit,
                    },
                )
                risk_status = "elevated"

            # Check ES limit
            if current_es > es_limit:
                alerts.append(
                    {
                        "type": "es_limit_exceeded",
                        "severity": "high" if current_es > es_limit * 1.5 else "medium",
                        "message": f"Expected Shortfall ({current_es:.4f}) exceeds limit ({es_limit:.4f})",
                        "value": current_es,
                        "limit": es_limit,
                    },
                )
                risk_status = "elevated"

            # Store risk metrics
            risk_metrics = {
                "current_var": current_var,
                "current_es": current_es,
                "var_limit": var_limit,
                "es_limit": es_limit,
                "risk_status": risk_status,
                "alerts": alerts,
                "timestamp": datetime.now().isoformat(),
            }

            self.var_history.append(risk_metrics)
            if len(self.var_history) > self.max_history:
                self.var_history.pop(0)

            return risk_metrics

        except Exception:
            self.print(error("Error monitoring risk limits: {e}"))
            return {}

    def get_risk_metrics(self, timeframe: str = "all") -> dict[str, Any]:
        """
        Get historical risk metrics.

        Args:
            timeframe: Timeframe for metrics ("all", "1d", "1w", "1m")

        Returns:
            dict: Risk metrics for the specified timeframe
        """
        try:
            if not self.var_history:
                return {}

            if timeframe == "all":
                return {
                    "var_history": self.var_history.copy(),
                    "latest_metrics": self.var_history[-1] if self.var_history else {},
                    "summary": self._calculate_risk_summary(),
                }
            # Filter by timeframe (simplified implementation)
            return {
                "latest_metrics": self.var_history[-1] if self.var_history else {},
                "timeframe": timeframe,
            }

        except Exception:
            self.print(error("Error getting risk metrics: {e}"))
            return {}

    def _calculate_risk_summary(self) -> dict[str, Any]:
        """Calculate summary statistics for risk metrics."""
        try:
            if not self.var_history:
                return {}

            var_values = [entry["current_var"] for entry in self.var_history]
            es_values = [entry["current_es"] for entry in self.var_history]

            return {
                "avg_var": np.mean(var_values),
                "max_var": np.max(var_values),
                "min_var": np.min(var_values),
                "var_volatility": np.std(var_values),
                "avg_es": np.mean(es_values),
                "max_es": np.max(es_values),
                "min_es": np.min(es_values),
                "es_volatility": np.std(es_values),
                "risk_events": len(
                    [
                        entry
                        for entry in self.var_history
                        if entry["risk_status"] == "elevated"
                    ],
                ),
            }

        except Exception:
            self.print(error("Error calculating risk summary: {e}"))
            return {}


risk_allocator: RiskAllocator | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="risk allocator setup",
)
async def setup_risk_allocator(
    config: dict[str, Any] | None = None,
) -> RiskAllocator | None:
    try:
        global risk_allocator
        if config is None:
            config = {"risk_allocator": {"allocation_interval": 60, "max_history": 100}}
        risk_allocator = RiskAllocator(config)
        success = await risk_allocator.initialize()
        if success:
            return risk_allocator
        return None
    except Exception as e:
        print(f"Error setting up risk allocator: {e}")
        return None
