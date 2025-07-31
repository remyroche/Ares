# src/components/modular_supervisor.py

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import time
import subprocess
import mlflow

from src.config import CONFIG
from src.interfaces import ISupervisor, EventType
from src.interfaces.base_interfaces import IExchangeClient, IStateManager, IEventBus
from src.utils.logger import system_logger
from src.config import settings
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
)


class ModularSupervisor(ISupervisor):
    """
    Modular implementation of the Supervisor that implements the ISupervisor interface.
    Uses dependency injection and event-driven communication.
    """

    def __init__(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        event_bus: IEventBus,
    ):
        """
        Initialize the modular supervisor.

        Args:
            exchange_client: Exchange client for data access
            state_manager: State manager for persistence
            event_bus: Event bus for communication
        """
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.config = settings.get("supervisor", {})
        self.running = False
        self._main_task: Optional[asyncio.Task] = None  # <-- ADD
        self.retraining_interval = self.config.get(
            "retraining_interval_seconds", 2592000
        )  # Default: 30 days <-- ADD
        mlflow.set_tracking_uri(CONFIG.get("MLFLOW_TRACKING_URI"))  # <-- ADD
        self.logger = system_logger.getChild("ModularSupervisor")
        self.performance_metrics = {}
        self.risk_alerts = []
        self.system_health = {
            "status": "HEALTHY",
            "last_check": datetime.now(),
            "components": {},
        }

        self.logger.info("ModularSupervisor initialized")

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="modular_supervisor_start"
    )
    async def start(self) -> None:
        """Start the modular supervisor"""
        self.logger.info("Starting ModularSupervisor")
        self.running = True

        # Subscribe to system events
        await self.event_bus.subscribe(
            EventType.TRADE_EXECUTED, self._handle_trade_executed
        )

        await self.event_bus.subscribe(
            EventType.SYSTEM_ERROR, self._handle_system_error
        )

        await self.event_bus.subscribe(
            EventType.COMPONENT_STARTED, self._handle_component_started
        )

        await self.event_bus.subscribe(
            EventType.COMPONENT_STOPPED, self._handle_component_stopped
        )

        self._main_task = asyncio.create_task(self._run_periodic_checks())
        self.logger.info("ModularSupervisor started and periodic checks are running.")

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="modular_supervisor_stop"
    )
    async def _run_periodic_checks(self):
        """The main loop for running periodic supervisory tasks."""
        while self.running:
            try:
                self.logger.info(
                    "Supervisor running periodic checks for model management..."
                )

                # 1. Check if it's time to trigger a new training run
                self._trigger_retraining_if_needed()

                # 2. Check for new candidate models and promote if better
                self._check_and_promote_model()

                # Wait for the next check cycle (e.g., 1 hour)
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                self.logger.info("Supervisor periodic check task cancelled.")
                break
            except Exception as e:
                self.logger.error(
                    f"Error in supervisor periodic check loop: {e}", exc_info=True
                )
                # Wait before retrying to avoid spamming errors
                await asyncio.sleep(60)

    def _trigger_retraining_if_needed(self):
        """Triggers the training CLI script if the defined interval has passed."""
        last_training_time_str = self.state_manager.get_state("last_retrain_timestamp")
        last_training_time = datetime.fromisoformat(last_training_time_str).timestamp()

        if time.time() - last_training_time > self.retraining_interval:
            self.logger.info(
                f"Retraining interval of {self.retraining_interval}s has elapsed. Triggering new training run."
            )
            try:
                symbol = settings.trade_symbol
                # Execute the training script
                process = subprocess.run(
                    ["python", "scripts/training_cli.py", "train", symbol],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.logger.info(f"Training script stdout: {process.stdout}")
                # Update the timestamp in the state
                self.state_manager.set_state(
                    "last_retrain_timestamp", datetime.now().isoformat()
                )
                self.logger.info("Training pipeline script executed successfully.")
            except subprocess.CalledProcessError as e:
                self.logger.error(
                    f"Training script failed with exit code {e.returncode}. Stderr: {e.stderr}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to execute training pipeline: {e}", exc_info=True
                )

    def _check_and_promote_model(self):
        """Checks MLflow for new candidate models and promotes the best one."""
        self.logger.info("Checking for candidate models for promotion...")
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(
                CONFIG.get("MLFLOW_EXPERIMENT_NAME")
            )
            if not experiment:
                self.logger.warning(
                    f"MLflow experiment '{CONFIG.get('MLFLOW_EXPERIMENT_NAME')}' not found."
                )
                return

            # Find the best performing candidate model
            candidate_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.model_status = 'candidate'",
                order_by=["metrics.accuracy DESC"],
                max_results=1,
            )
            if not candidate_runs:
                self.logger.info("No new candidate models found.")
                return

            best_candidate_run = candidate_runs[0]
            candidate_run_id = best_candidate_run.info.run_id
            candidate_accuracy = best_candidate_run.data.metrics.get("accuracy", 0)

            # Get current production model's performance
            prod_run_id = self.state_manager.get_state("production_model_run_id")
            prod_accuracy = 0
            if prod_run_id:
                try:
                    prod_run = client.get_run(prod_run_id)
                    prod_accuracy = prod_run.data.metrics.get("accuracy", 0)
                except mlflow.exceptions.MlflowException:
                    self.logger.warning(
                        f"Could not find previous production run {prod_run_id}. Assuming 0 accuracy."
                    )
                    prod_run_id = None

            self.logger.info(
                f"Candidate: {candidate_run_id} (Acc: {candidate_accuracy:.4f}). Production: {prod_run_id} (Acc: {prod_accuracy:.4f})."
            )

            # Promote if candidate is better
            if candidate_accuracy > prod_accuracy:
                self.logger.warning(
                    f"Promoting new model {candidate_run_id} to production."
                )
                self.state_manager.set_state(
                    "production_model_run_id", candidate_run_id
                )
                client.set_tag(candidate_run_id, "model_status", "production")
                if prod_run_id:
                    client.set_tag(prod_run_id, "model_status", "archived")
            else:
                self.logger.info(
                    "Current production model remains superior. Archiving candidate."
                )
                client.set_tag(candidate_run_id, "model_status", "evaluated_inferior")

        except Exception as e:
            self.logger.error(
                f"Failed during model promotion check: {e}", exc_info=True
            )

    async def stop(self) -> None:
        """Stop the modular supervisor"""
        self.logger.info("Stopping ModularSupervisor")
        self.running = False

        # Unsubscribe from events
        await self.event_bus.unsubscribe(
            EventType.TRADE_EXECUTED, self._handle_trade_executed
        )

        await self.event_bus.unsubscribe(
            EventType.SYSTEM_ERROR, self._handle_system_error
        )

        await self.event_bus.unsubscribe(
            EventType.COMPONENT_STARTED, self._handle_component_started
        )

        await self.event_bus.unsubscribe(
            EventType.COMPONENT_STOPPED, self._handle_component_stopped
        )

        self.logger.info("ModularSupervisor stopped")

    @handle_errors(
        exceptions=(Exception,), default_return={}, context="monitor_performance"
    )
    async def monitor_performance(self) -> Dict[str, Any]:
        """
        Monitor system performance.

        Returns:
            Performance monitoring results
        """
        if not self.running:
            return {}

        self.logger.debug("Monitoring system performance")

        try:
            # Get account information
            account_info = await self.exchange.get_account_info()

            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(
                account_info
            )

            # Check for performance alerts
            alerts = await self._check_performance_alerts(performance_metrics)

            # Update performance history
            self.performance_metrics = performance_metrics

            # Publish performance update
            await self.event_bus.publish(
                EventType.PERFORMANCE_UPDATE, performance_metrics, "ModularSupervisor"
            )

            return {
                "metrics": performance_metrics,
                "alerts": alerts,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}", exc_info=True)
            return {}

    @handle_errors(exceptions=(Exception,), default_return={}, context="manage_risk")
    async def manage_risk(self) -> Dict[str, Any]:
        """
        Manage risk across all components.

        Returns:
            Risk management results
        """
        if not self.running:
            return {}

        self.logger.debug("Managing system risk")

        try:
            # Get current positions
            positions = await self.exchange.get_position_risk(settings.trade_symbol)

            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(positions)

            # Check for risk alerts
            risk_alerts = await self._check_risk_alerts(risk_metrics)

            # Update risk state
            self.state_manager.set_state("current_risk_metrics", risk_metrics)

            # Publish risk alerts if any
            if risk_alerts:
                await self.event_bus.publish(
                    EventType.RISK_ALERT, risk_alerts, "ModularSupervisor"
                )

            return {
                "risk_metrics": risk_metrics,
                "alerts": risk_alerts,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Risk management failed: {e}", exc_info=True)
            return {}

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="coordinate_components"
    )
    async def coordinate_components(self) -> None:
        """Coordinate all trading components"""
        if not self.running:
            return

        self.logger.debug("Coordinating trading components")

        try:
            # Monitor component health
            await self._monitor_component_health()

            # Check for component failures
            await self._check_component_failures()

            # Update system health
            await self._update_system_health()

            # Publish system status
            await self.event_bus.publish(
                EventType.PERFORMANCE_UPDATE, self.system_health, "ModularSupervisor"
            )

        except Exception as e:
            self.logger.error(f"Component coordination failed: {e}", exc_info=True)

    async def _handle_trade_executed(self, event) -> None:
        """Handle trade executed events"""
        trade_data = event.data
        self.logger.info(f"Trade executed: {trade_data.symbol} {trade_data.action}")

        # Update performance metrics
        await self.monitor_performance()

        # Check risk levels
        await self.manage_risk()

    async def _handle_system_error(self, event) -> None:
        """Handle system error events"""
        error_data = event.data
        self.logger.error(f"System error: {error_data}")

        # Update system health
        self.system_health["status"] = "ERROR"
        self.system_health["last_error"] = error_data

        # Publish error alert
        await self.event_bus.publish(
            EventType.RISK_ALERT,
            {"type": "SYSTEM_ERROR", "data": error_data},
            "ModularSupervisor",
        )

    async def _handle_component_started(self, event) -> None:
        """Handle component started events"""
        component_data = event.data
        component_name = component_data.get("component", "unknown")

        self.system_health["components"][component_name] = {
            "status": "RUNNING",
            "start_time": datetime.now(),
            "last_update": datetime.now(),
        }

        self.logger.info(f"Component started: {component_name}")

    async def _handle_component_stopped(self, event) -> None:
        """Handle component stopped events"""
        component_data = event.data
        component_name = component_data.get("component", "unknown")

        if component_name in self.system_health["components"]:
            self.system_health["components"][component_name]["status"] = "STOPPED"
            self.system_health["components"][component_name]["stop_time"] = (
                datetime.now()
            )

        self.logger.info(f"Component stopped: {component_name}")

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return={},
        context="calculate_performance_metrics",
    )
    async def _calculate_performance_metrics(
        self, account_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance metrics from account information"""
        try:
            total_balance = account_info.get("totalWalletBalance", 0)
            available_balance = account_info.get("availableBalance", 0)
            total_margin_balance = account_info.get("totalMarginBalance", 0)

            # Calculate basic metrics
            metrics = {
                "total_balance": total_balance,
                "available_balance": available_balance,
                "total_margin_balance": total_margin_balance,
                "utilization_rate": (total_balance - available_balance) / total_balance
                if total_balance > 0
                else 0,
                "margin_utilization": (total_margin_balance - available_balance)
                / total_margin_balance
                if total_margin_balance > 0
                else 0,
            }

            # Get historical performance if available
            historical_performance = self.state_manager.get_state(
                "historical_performance", {}
            )
            if historical_performance:
                metrics["historical_pnl"] = historical_performance.get("total_pnl", 0)
                metrics["win_rate"] = historical_performance.get("win_rate", 0)
                metrics["sharpe_ratio"] = historical_performance.get("sharpe_ratio", 0)

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=[],
        context="check_performance_alerts",
    )
    async def _check_performance_alerts(
        self, performance_metrics: Dict[str, Any]
    ) -> list:
        """Check for performance alerts"""
        alerts = []

        try:
            # Check balance thresholds
            total_balance = performance_metrics.get("total_balance", 0)
            min_balance = self.config.get("min_balance", 1000)

            if total_balance < min_balance:
                alerts.append(
                    {
                        "type": "LOW_BALANCE",
                        "severity": "HIGH",
                        "message": f"Account balance ({total_balance}) below minimum threshold ({min_balance})",
                    }
                )

            # Check utilization rates
            utilization_rate = performance_metrics.get("utilization_rate", 0)
            max_utilization = self.config.get("max_utilization", 0.8)

            if utilization_rate > max_utilization:
                alerts.append(
                    {
                        "type": "HIGH_UTILIZATION",
                        "severity": "MEDIUM",
                        "message": f"Account utilization ({utilization_rate:.2%}) above threshold ({max_utilization:.2%})",
                    }
                )

            # Check historical performance
            win_rate = performance_metrics.get("win_rate", 0)
            min_win_rate = self.config.get("min_win_rate", 0.5)

            if win_rate < min_win_rate:
                alerts.append(
                    {
                        "type": "LOW_WIN_RATE",
                        "severity": "MEDIUM",
                        "message": f"Win rate ({win_rate:.2%}) below threshold ({min_win_rate:.2%})",
                    }
                )

            return alerts

        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
            return []

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return={},
        context="calculate_risk_metrics",
    )
    async def _calculate_risk_metrics(
        self, positions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate risk metrics from positions"""
        try:
            risk_metrics = {
                "total_position_value": 0,
                "total_unrealized_pnl": 0,
                "max_drawdown": 0,
                "liquidation_risk": 0,
                "position_count": 0,
            }

            if positions and "positions" in positions:
                for position in positions["positions"]:
                    if position.get("size", 0) != 0:  # Active position
                        risk_metrics["position_count"] += 1
                        risk_metrics["total_position_value"] += abs(
                            position.get("notional", 0)
                        )
                        risk_metrics["total_unrealized_pnl"] += position.get(
                            "unrealizedPnl", 0
                        )

                        # Calculate liquidation risk
                        liquidation_price = position.get("liquidationPrice", 0)
                        current_price = position.get("markPrice", 0)

                        if liquidation_price > 0 and current_price > 0:
                            if position.get("side") == "LONG":
                                distance_to_liquidation = (
                                    current_price - liquidation_price
                                ) / current_price
                            else:
                                distance_to_liquidation = (
                                    liquidation_price - current_price
                                ) / current_price

                            risk_metrics["liquidation_risk"] = max(
                                risk_metrics["liquidation_risk"],
                                1 - distance_to_liquidation,
                            )

            return risk_metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=[],
        context="check_risk_alerts",
    )
    async def _check_risk_alerts(self, risk_metrics: Dict[str, Any]) -> list:
        """Check for risk alerts"""
        alerts = []

        try:
            # Check liquidation risk
            liquidation_risk = risk_metrics.get("liquidation_risk", 0)
            max_liquidation_risk = self.config.get("max_liquidation_risk", 0.3)

            if liquidation_risk > max_liquidation_risk:
                alerts.append(
                    {
                        "type": "HIGH_LIQUIDATION_RISK",
                        "severity": "HIGH",
                        "message": f"Liquidation risk ({liquidation_risk:.2%}) above threshold ({max_liquidation_risk:.2%})",
                    }
                )

            # Check position count
            position_count = risk_metrics.get("position_count", 0)
            max_positions = self.config.get("max_positions", 5)

            if position_count > max_positions:
                alerts.append(
                    {
                        "type": "TOO_MANY_POSITIONS",
                        "severity": "MEDIUM",
                        "message": f"Number of positions ({position_count}) above limit ({max_positions})",
                    }
                )

            # Check unrealized P&L
            unrealized_pnl = risk_metrics.get("total_unrealized_pnl", 0)
            max_loss = self.config.get("max_unrealized_loss", -1000)

            if unrealized_pnl < max_loss:
                alerts.append(
                    {
                        "type": "HIGH_UNREALIZED_LOSS",
                        "severity": "HIGH",
                        "message": f"Unrealized loss ({unrealized_pnl}) below threshold ({max_loss})",
                    }
                )

            return alerts

        except Exception as e:
            self.logger.error(f"Error checking risk alerts: {e}")
            return []

    async def _monitor_component_health(self) -> None:
        """Monitor health of all components"""
        try:
            current_time = datetime.now()

            for component_name, component_info in self.system_health[
                "components"
            ].items():
                last_update = component_info.get("last_update", current_time)

                # Check if component is stale (no updates in last 5 minutes)
                if (current_time - last_update).total_seconds() > 300:
                    component_info["status"] = "STALE"
                    self.logger.warning(f"Component {component_name} is stale")

        except Exception as e:
            self.logger.error(f"Error monitoring component health: {e}")

    async def _check_component_failures(self) -> None:
        """Check for component failures"""
        try:
            failed_components = [
                name
                for name, info in self.system_health["components"].items()
                if info.get("status") in ["STOPPED", "ERROR"]
            ]

            if failed_components:
                self.system_health["status"] = "DEGRADED"
                self.logger.warning(f"Failed components: {failed_components}")

                # Publish system error
                await self.event_bus.publish(
                    EventType.SYSTEM_ERROR,
                    {"type": "COMPONENT_FAILURE", "components": failed_components},
                    "ModularSupervisor",
                )
            else:
                self.system_health["status"] = "HEALTHY"

        except Exception as e:
            self.logger.error(f"Error checking component failures: {e}")

    async def _update_system_health(self) -> None:
        """Update system health status"""
        try:
            current_time = datetime.now()
            self.system_health["last_check"] = current_time

            # Update component status
            for component_info in self.system_health["components"].values():
                if component_info.get("status") == "RUNNING":
                    component_info["last_update"] = current_time

        except Exception as e:
            self.logger.error(f"Error updating system health: {e}")
