"""
PnL Loss Functions Module - Backward Compatibility Layer.

This module provides backward compatibility for the refactored loss functions.
The actual implementations are now in the src/supervisor/loss_functions/ package.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from src.utils.logger import system_logger

from .loss_functions.loss_calculator import LossCalculator
from .loss_functions.optimization_metrics import OptimizationMetricsCalculator
from .loss_functions.performance_metrics import PerformanceMetricsCalculator

# Import the original factory function
from .loss_functions.pnl_aware import create_pnl_aware_loss

# Import all calculator classes
from .loss_functions.pnl_calculator import PnLCalculator
from .loss_functions.risk_metrics import RiskMetricsCalculator
from copy import copy


class PnLLossFunctions:
    """
    Unified PnL Loss Functions class that combines all calculators.
    This provides backward compatibility with the original monolithic class.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize PnL loss functions with all calculator components.

        Args:
            config: Configuration dictionary
        """
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("PnLLossFunctions")

        # Initialize all calculator components
        self.pnl_calculator = PnLCalculator(config)
        self.risk_metrics_calculator = RiskMetricsCalculator(config)
        self.performance_metrics_calculator = PerformanceMetricsCalculator(config)
        self.optimization_metrics_calculator = OptimizationMetricsCalculator(config)
        self.loss_calculator = LossCalculator(config)

        # Shared state
        self.is_calculating: bool = False
        self.calculation_results: Dict[str, Any] = {}
        self.calculation_history: list[Dict[str, Any]] = []

        # Configuration
        self.pnl_config: Dict[str, Any] = self.config.get("pnl_loss_functions", {})
        self.calculation_interval: int = self.pnl_config.get(
            "calculation_interval",
            3600,
        )
        self.max_calculation_history: int = self.pnl_config.get(
            "max_calculation_history",
            100,
        )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid PnL configuration"),
            AttributeError: (False, "Missing required PnL parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="pnl loss functions initialization",
    )
    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            self.logger.info("Initializing PnL Loss Functions...")
            
            # Initialize all calculators
            results = await asyncio.gather(
                self.pnl_calculator.initialize(),
                self.risk_metrics_calculator.initialize(),
                self.performance_metrics_calculator.initialize(),
                self.optimization_metrics_calculator.initialize(),
                self.loss_calculator.initialize(),
                return_exceptions=True
            )
            
            # Check if all initializations succeeded
            for i, result in enumerate(results):
                if isinstance(result, Exception) or not result:
                    self.logger.error(f"Failed to initialize component {i}: {result}")
                    return False
            
            self.logger.info("✅ PnL Loss Functions initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ PnL Loss Functions initialization failed: {e}")
            return False

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid calculation inputs"),
            KeyError: (False, "Missing required calculation parameters"),
            Exception: (False, "Calculation execution failed"),
        },
        default_return=False,
        context="pnl calculation execution",
    )
    async def execute_calculation(self, calculation_input: Dict[str, Any]) -> bool:
        """
        Execute PnL calculations using all calculators.

        Args:
            calculation_input: Dictionary containing calculation inputs

        Returns:
            Success status
        """
        try:
            self.is_calculating = True
            self.logger.info("Executing PnL calculations...")

            # Clear previous results
            self.calculation_results = {}

            # PnL calculations
            if self.pnl_calculator.enable_pnl_calculation:
                trades = calculation_input.get("trades", [])
                self.calculation_results["total_pnl"] = self.pnl_calculator.calculate_total_pnl(
                    {"trades": trades}
                )
                
                # Calculate additional PnL metrics if we have returns data
                returns = calculation_input.get("returns")
                if returns is not None:
                    self.calculation_results["sharpe_ratio"] = self.pnl_calculator.calculate_sharpe_ratio(returns)
                    
                    equity_curve = calculation_input.get("equity_curve")
                    if equity_curve is not None:
                        self.calculation_results["max_drawdown"] = self.pnl_calculator.calculate_max_drawdown(equity_curve)

            # Risk metrics
            if self.risk_metrics_calculator.enable_risk_metrics and "returns" in calculation_input:
                returns = calculation_input["returns"]
                self.calculation_results["var_95"] = self.risk_metrics_calculator.calculate_var(returns, 0.95)
                self.calculation_results["var_99"] = self.risk_metrics_calculator.calculate_var(returns, 0.99)
                self.calculation_results["cvar_95"] = self.risk_metrics_calculator.calculate_cvar(returns, 0.95)
                self.calculation_results["cvar_99"] = self.risk_metrics_calculator.calculate_cvar(returns, 0.99)
                self.calculation_results["tail_risk"] = self.risk_metrics_calculator.calculate_tail_risk(returns)

            # Performance metrics
            if self.performance_metrics_calculator.enable_performance_metrics:
                trades = calculation_input.get("trades", [])
                if trades:
                    self.calculation_results["win_rate"] = self.performance_metrics_calculator.calculate_win_rate(trades)
                
                returns = calculation_input.get("returns")
                if returns is not None:
                    self.calculation_results["sortino_ratio"] = self.performance_metrics_calculator.calculate_sortino_ratio(returns)

            # Optimization metrics
            if self.optimization_metrics_calculator.enable_optimization_metrics:
                win_rate_data = self.calculation_results.get("win_rate", {})
                if win_rate_data:
                    self.calculation_results["kelly_criterion"] = self.optimization_metrics_calculator.calculate_kelly_criterion(
                        win_rate_data.get("win_rate", 0.5),
                        win_rate_data.get("average_win", 1.0),
                        win_rate_data.get("average_loss", 1.0)
                    )

            # Update history
            self._update_calculation_history()
            
            self.is_calculating = False
            self.logger.info("PnL calculations completed successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error executing PnL calculations: {e}")
            self.is_calculating = False
            return False

    # Delegate methods to maintain backward compatibility
    def _perform_total_pnl(self, calculation_input: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper for total PnL calculation."""
        return self.pnl_calculator.calculate_total_pnl(calculation_input)

    def _perform_var_95(self, calculation_input: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper for 95% VaR calculation."""
        returns = calculation_input.get("returns", [])
        return self.risk_metrics_calculator.calculate_var(returns, 0.95)

    def _perform_var_99(self, calculation_input: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper for 99% VaR calculation."""
        returns = calculation_input.get("returns", [])
        return self.risk_metrics_calculator.calculate_var(returns, 0.99)

    def _perform_cvar_95(self, calculation_input: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper for 95% CVaR calculation."""
        returns = calculation_input.get("returns", [])
        return self.risk_metrics_calculator.calculate_cvar(returns, 0.95)

    def _perform_cvar_99(self, calculation_input: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper for 99% CVaR calculation."""
        returns = calculation_input.get("returns", [])
        return self.risk_metrics_calculator.calculate_cvar(returns, 0.99)

    def _perform_tail_risk(self, calculation_input: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper for tail risk calculation."""
        returns = calculation_input.get("returns", [])
        return self.risk_metrics_calculator.calculate_tail_risk(returns)

    def _perform_risk_budget(self, calculation_input: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper for risk budget calculation."""
        weights = calculation_input.get("portfolio_weights", [])
        covariances = calculation_input.get("asset_covariances", [])
        return self.risk_metrics_calculator.calculate_risk_budget(weights, covariances)

    @handles_errors(fallback=None)
    def _update_calculation_history(self) -> None:
        """Update calculation history."""
        try:
            now = datetime.now()
            history_entry = {
                "timestamp": now.isoformat(),
                "results": self.calculation_results.copy(),
                "is_calculating": self.is_calculating,
            }
            self.calculation_history.append(history_entry)
            if len(self.calculation_history) > self.max_calculation_history:
                self.calculation_history.pop(0)
        except Exception as e:
            self.logger.exception(f"Error updating calculation history: {e}")

    def get_calculation_history(self, limit: int | None = None) -> list[Dict[str, Any]]:
        """Get calculation history."""
        history = self.calculation_history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_calculation_status(self) -> Dict[str, Any]:
        """Get current calculation status."""
        return {
            "is_calculating": self.is_calculating,
            "last_calculation": (
                self.calculation_history[-1]["timestamp"]
                if self.calculation_history
                else None
            ),
            "results": self.calculation_results.copy(),
        }

    async def stop(self) -> None:
        """Stop the PnL loss functions component."""
        self.logger.info("Stopping PnL Loss Functions...")
        self.is_calculating = False
        self.logger.info("PnL Loss Functions stopped successfully")


# Import asyncio for the async methods
import asyncio

# Export the main components
__all__ = [
    "create_pnl_aware_loss",
    "PnLLossFunctions",
    "PnLCalculator",
    "RiskMetricsCalculator",
    "PerformanceMetricsCalculator",
    "OptimizationMetricsCalculator",
    "LossCalculator",
]