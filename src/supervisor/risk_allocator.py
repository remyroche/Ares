# src/supervisor/risk_allocator.py

from datetime import datetime
from src.utils.logger import system_logger
from typing import Any
import asyncio
import numpy as np

from src.utils.error_handler import handle_errors, handle_specific_errors

class RiskAllocator:
    """
    Portfolio-Level Risk Allocator component responsible for:
    - Portfolio-level risk management (excluding position sizing)
    - Global portfolio guards and kill-switches
    - VaR and ES monitoring
    - Portfolio-level risk limits and allocations

    Note: Position sizing is handled by the Tactician component
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
        except Exception as e:
            self.logger.error(f"Error loading risk configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.allocation_interval <= 0:
                self.logger.error("Invalid allocation interval")
                return False
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
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
        except Exception as e:
            self.logger.error(f"Error in risk allocator run: {e}")
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
        except Exception as e:
            self.logger.error(f"Error in risk allocation step: {e}")

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
        except Exception as e:
            self.logger.error(f"Error calculating risk allocations: {e}")

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
        except Exception as e:
            self.logger.error(f"Error updating risk limits: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="risk allocator stop",
    )
    def calculate_var(
        self, returns: list[float],
        confidence_level: float = None
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

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.0

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

        except Exception as e:
            self.logger.error(f"Error calculating risk summary: {e}")
            return {}

risk_allocator: RiskAllocator | None = None

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="risk allocator setup",
)