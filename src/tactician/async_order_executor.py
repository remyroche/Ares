# src/tactician/async_order_executor.py

"""
Async Order Executor with Advanced Analytics and Dynamic Parameter Optimization
Integrates with Enhanced Order Manager, Performance Reporter, and Optuna for optimization.
"""

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np
import optuna

from src.supervisor.performance_reporter import (
    AdvancedReportingEngine,
    PerformanceReporter,
)
from src.tactician.enhanced_order_manager import (
    EnhancedOrderManager,
    OrderRequest,
    OrderSide,
    OrderType,
)
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.prometheus_metrics import metrics
from src.utils.warning_symbols import (
    error,
    execution_error,
    failed,
    initialization_error,
    warning,
)


class ExecutionStrategy(Enum):
    """Order execution strategies."""

    IMMEDIATE = "immediate"
    BATCH = "batch"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"
    ADAPTIVE = "adaptive"


class ExecutionStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    EXECUTING = "executing"
    PARTIALLY_FILLED = "partially_filled"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionRequest:
    """Execution request data structure."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    leverage: float | None = None
    strategy_type: str | None = None
    execution_strategy: ExecutionStrategy = ExecutionStrategy.IMMEDIATE
    max_slippage: float = 0.001  # 0.1% max slippage
    timeout_seconds: int = 30
    batch_size: float | None = None
    batch_interval: int | None = None
    priority: int = 1  # Higher number = higher priority
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Execution result data structure."""

    execution_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    requested_quantity: float
    executed_quantity: float
    average_price: float
    total_cost: float
    slippage: float
    execution_time: float
    status: ExecutionStatus
    orders_placed: list[str]
    performance_metrics: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class AsyncOrderExecutor:
    """
    Async order executor with advanced analytics and dynamic parameter optimization.
    Integrates with Enhanced Order Manager, Performance Reporter, and Optuna.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("AsyncOrderExecutor")

        # Core components
        self.order_manager: EnhancedOrderManager | None = None
        self.performance_reporter: PerformanceReporter | None = None
        self.advanced_reporter: AdvancedReportingEngine | None = None

        # Execution configuration
        self.execution_config = config.get("async_order_executor", {})
        self.max_concurrent_orders = self.execution_config.get(
            "max_concurrent_orders",
            10,
        )
        self.execution_timeout = self.execution_config.get("execution_timeout", 300)
        self.retry_attempts = self.execution_config.get("retry_attempts", 3)
        self.retry_delay = self.execution_config.get("retry_delay", 1)

        # Performance tracking
        self.execution_history: list[ExecutionResult] = []
        self.active_executions: dict[str, ExecutionRequest] = {}
        self.performance_metrics: dict[str, Any] = {}

        # Optuna integration for parameter optimization
        self.optuna_study: optuna.Study | None = None
        self.optimization_config = config.get("optuna_optimization", {})
        self.optimization_enabled = self.optimization_config.get("enabled", True)

        # Execution queue and semaphore
        # Bounded priority queue to apply backpressure and honor priorities
        self.execution_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.execution_config.get("queue_maxsize", 1000),
        )
        self.execution_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self.max_concurrent_orders,
        )

        # Optional rate limiter (token bucket) semaphore per endpoint
        self.rate_limiter: asyncio.Semaphore = asyncio.Semaphore(
            self.execution_config.get("max_requests_per_window", 20),
        )

        # Background tasks
        self.execution_task: asyncio.Task | None = None
        self.optimization_task: asyncio.Task | None = None
        self.analytics_task: asyncio.Task | None = None

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="async order executor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the async order executor."""
        try:
            self.logger.info("🚀 Initializing Async Order Executor...")

            # Initialize order manager
            await self._initialize_order_manager()

            # Initialize performance reporter
            await self._initialize_performance_reporter()

            # Initialize Optuna study
            if self.optimization_enabled:
                await self._initialize_optuna_study()

            # Start background tasks
            await self._start_background_tasks()

            self.is_initialized = True
            self.logger.info("✅ Async Order Executor initialized successfully")
            return True

        except Exception:
            self.print(
                initialization_error("❌ Error initializing async order executor: {e}"),
            )
            return False

    async def _initialize_order_manager(self) -> None:
        """Initialize the enhanced order manager."""
        try:
            from src.tactician.enhanced_order_manager import (
                setup_enhanced_order_manager,
            )

            self.order_manager = await setup_enhanced_order_manager(self.config)

            if self.order_manager:
                self.logger.info("✅ Order manager initialized successfully")
            else:
                self.print(failed("Failed to initialize order manager"))

        except Exception:
            self.print(initialization_error("Error initializing order manager: {e}"))

    async def _initialize_performance_reporter(self) -> None:
        """Initialize the performance reporter."""
        try:
            from src.supervisor.performance_reporter import setup_performance_reporter

            self.performance_reporter = await setup_performance_reporter(self.config)

            if self.performance_reporter:
                self.logger.info("✅ Performance reporter initialized successfully")

                # Initialize advanced reporting engine
                self.advanced_reporter = AdvancedReportingEngine(self.config)
                self.logger.info("✅ Advanced reporting engine initialized")
            else:
                self.print(failed("Failed to initialize performance reporter"))

        except Exception:
            self.print(
                initialization_error("Error initializing performance reporter: {e}"),
            )

    async def _initialize_optuna_study(self) -> None:
        """Initialize Optuna study for parameter optimization."""
        try:
            study_name = self.optimization_config.get(
                "study_name",
                "order_execution_optimization",
            )
            storage_url = self.optimization_config.get(
                "storage_url",
                "sqlite:///optuna_studies.db",
            )

            # Create or load existing study
            self.optuna_study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                direction="maximize",  # Maximize execution success rate
                sampler=optuna.samplers.TPESampler(seed=42),
            )

            self.logger.info(f"✅ Optuna study initialized: {study_name}")

        except Exception:
            self.print(initialization_error("Error initializing Optuna study: {e}"))

    async def _start_background_tasks(self) -> None:
        """Start background tasks for execution, optimization, and analytics."""
        try:
            # Start execution task
            self.execution_task = asyncio.create_task(self._execution_worker())

            # Start optimization task if enabled
            if self.optimization_enabled:
                self.optimization_task = asyncio.create_task(
                    self._optimization_worker(),
                )

            # Start analytics task
            self.analytics_task = asyncio.create_task(self._analytics_worker())

            self.logger.info("✅ Background tasks started successfully")

        except Exception:
            self.print(error("Error starting background tasks: {e}"))

    async def execute_order_async(self, execution_request: ExecutionRequest) -> str:
        """
        Submit an order for async execution.

        Args:
            execution_request: Order execution request

        Returns:
            str: Execution ID
        """
        try:
            # Generate execution ID (idempotency friendly caller can reuse)
            execution_id = f"exec_{uuid4().hex[:8]}_{execution_request.symbol}"

            # Add to execution queue
            priority = max(
                0,
                10 - int(execution_request.priority),
            )  # lower is higher priority
            await self.execution_queue.put((priority, execution_id, execution_request))
            try:
                # Use data_size_gauge to track queue depth under a label
                metrics.data_size_gauge.set_function(
                    lambda: self.execution_queue.qsize(),
                )
            except Exception:
                pass

            self.logger.info(f"Order submitted for async execution: {execution_id}")
            return execution_id

        except Exception:
            self.print(execution_error("Error submitting order for execution: {e}"))
            raise

    async def _execution_worker(self) -> None:
        """Background worker for processing execution requests."""
        try:
            while True:
                # Get execution request from queue
                _, execution_id, execution_request = await self.execution_queue.get()
                with contextlib.suppress(Exception):
                    metrics.data_size_gauge.set_function(
                        lambda: self.execution_queue.qsize(),
                    )

                # Acquire semaphore for concurrent execution control
                async with self.execution_semaphore:
                    try:
                        # Optional simple rate limit backoff
                        async with self.rate_limiter:
                            pass
                        # Execute the order
                        result = await self._execute_single_order(
                            execution_id,
                            execution_request,
                        )

                        # Store result
                        self.execution_history.append(result)

                        # Update performance metrics
                        await self._update_performance_metrics(result)
                        try:
                            status_label = str(
                                result.status.value
                                if hasattr(result.status, "value")
                                else result.status,
                            )
                            metrics.step_execution_duration.labels(
                                step_name="order_execution",
                                status=status_label,
                            ).observe(result.execution_time)
                            if status_label == "completed":
                                metrics.step_success_counter.labels(
                                    step_name="order_execution",
                                ).inc()
                            else:
                                metrics.step_failure_counter.labels(
                                    step_name="order_execution",
                                    error_type=status_label,
                                ).inc()
                        except Exception:
                            pass

                        self.logger.info(f"Order execution completed: {execution_id}")

                    except Exception as e:
                        self.logger.exception(
                            f"Error executing order {execution_id}: {e}",
                        )

                    finally:
                        # Mark task as done
                        self.execution_queue.task_done()

        except asyncio.CancelledError:
            self.logger.info("Execution worker cancelled")
        except Exception:
            self.print(execution_error("Error in execution worker: {e}"))

    async def _execute_single_order(
        self,
        execution_id: str,
        execution_request: ExecutionRequest,
    ) -> ExecutionResult:
        """Execute a single order with the specified strategy."""
        try:
            start_time = time.time()

            # Choose execution strategy
            if execution_request.execution_strategy == ExecutionStrategy.IMMEDIATE:
                result = await self._execute_immediate(execution_id, execution_request)
            elif execution_request.execution_strategy == ExecutionStrategy.BATCH:
                result = await self._execute_batch(execution_id, execution_request)
            elif execution_request.execution_strategy == ExecutionStrategy.TWAP:
                result = await self._execute_twap(execution_id, execution_request)
            elif execution_request.execution_strategy == ExecutionStrategy.VWAP:
                result = await self._execute_vwap(execution_id, execution_request)
            elif execution_request.execution_strategy == ExecutionStrategy.ICEBERG:
                result = await self._execute_iceberg(execution_id, execution_request)
            elif execution_request.execution_strategy == ExecutionStrategy.ADAPTIVE:
                result = await self._execute_adaptive(execution_id, execution_request)
            else:
                result = await self._execute_immediate(execution_id, execution_request)

            # Calculate execution metrics
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            return result

        except Exception:
            self.print(execution_error("Error executing order {execution_id}: {e}"))
            raise

    async def _execute_immediate(
        self,
        execution_id: str,
        execution_request: ExecutionRequest,
    ) -> ExecutionResult:
        """Execute order immediately."""
        try:
            if not self.order_manager:
                msg = "Order manager not initialized"
                raise ValueError(msg)

            time.time()

            # Create order request
            order_request = OrderRequest(
                symbol=execution_request.symbol,
                side=execution_request.side,
                order_type=execution_request.order_type,
                quantity=execution_request.quantity,
                price=execution_request.price,
                stop_price=execution_request.stop_price,
                leverage=execution_request.leverage,
                time_in_force=self.execution_config.get("default_time_in_force", "GTC"),
                reduce_only=bool(self.execution_config.get("reduce_only", False)),
                close_on_trigger=bool(
                    self.execution_config.get("close_on_trigger", False),
                ),
                order_link_id=execution_id,
                strategy_type=execution_request.strategy_type,
            )

            # Place order
            order_state = await self.order_manager._place_order(order_request)

            if not order_state:
                msg = "Failed to place order"
                raise RuntimeError(msg)

            # Calculate metrics
            slippage = self._calculate_slippage(
                execution_request.price,
                order_state.average_price,
            )
            total_cost = order_state.executed_quantity * order_state.average_price

            return ExecutionResult(
                execution_id=execution_id,
                symbol=execution_request.symbol,
                side=execution_request.side,
                order_type=execution_request.order_type,
                requested_quantity=execution_request.quantity,
                executed_quantity=order_state.executed_quantity,
                average_price=order_state.average_price,
                total_cost=total_cost,
                slippage=slippage,
                execution_time=0,  # Will be set by caller
                status=ExecutionStatus.COMPLETED
                if order_state.status.value == "filled"
                else ExecutionStatus.PARTIALLY_FILLED,
                orders_placed=[order_state.order_id],
                performance_metrics={},
            )

        except Exception:
            self.print(execution_error("Error in immediate execution: {e}"))
            raise

    async def _execute_batch(
        self,
        execution_id: str,
        execution_request: ExecutionRequest,
    ) -> ExecutionResult:
        """Execute order using batch strategy."""
        try:
            if not self.order_manager:
                msg = "Order manager not initialized"
                raise ValueError(msg)

            if not execution_request.batch_size or not execution_request.batch_interval:
                msg = "Batch size and interval required for batch execution"
                raise ValueError(msg)

            total_quantity = execution_request.quantity
            batch_size = execution_request.batch_size
            batch_interval = execution_request.batch_interval

            executed_quantity = 0.0
            total_cost = 0.0
            orders_placed = []

            num_batches = int(total_quantity / batch_size) + (
                1 if total_quantity % batch_size != 0 else 0
            )
            for i in range(num_batches):
                batched_qty = (
                    batch_size
                    if i < num_batches - 1
                    else total_quantity - (num_batches - 1) * batch_size
                )
                orq = OrderRequest(
                    symbol=execution_request.symbol,
                    side=execution_request.side,
                    order_type=execution_request.order_type,
                    quantity=float(batched_qty),
                    price=execution_request.price,
                    stop_price=execution_request.stop_price,
                    leverage=execution_request.leverage,
                    time_in_force=self.execution_config.get(
                        "default_time_in_force",
                        "GTC",
                    ),
                    reduce_only=bool(self.execution_config.get("reduce_only", False)),
                    close_on_trigger=bool(
                        self.execution_config.get("close_on_trigger", False),
                    ),
                    order_link_id=f"{execution_id}_b{i}",
                    strategy_type=execution_request.strategy_type,
                )
                st = await self.order_manager._place_order(orq)
                if st:
                    orders_placed.append(st.order_id)
                    # Accumulate executed quantity and cost using average price
                    executed_quantity += float(st.executed_quantity)
                    if st.executed_quantity and st.average_price:
                        total_cost += float(st.executed_quantity) * float(
                            st.average_price,
                        )

                # Wait for next batch
                if i < num_batches - 1:
                    await asyncio.sleep(batch_interval)

            # Calculate average price
            average_price = (
                (total_cost / executed_quantity) if executed_quantity > 0 else 0.0
            )
            slippage = self._calculate_slippage(execution_request.price, average_price)

            return ExecutionResult(
                execution_id=execution_id,
                symbol=execution_request.symbol,
                side=execution_request.side,
                order_type=execution_request.order_type,
                requested_quantity=float(execution_request.quantity),
                executed_quantity=float(executed_quantity),
                average_price=average_price,
                total_cost=float(total_cost),
                slippage=slippage,
                execution_time=0,  # Will be set by caller
                status=ExecutionStatus.COMPLETED,
                orders_placed=orders_placed,
                performance_metrics={},
            )

        except Exception:
            self.print(execution_error("Error in batch execution: {e}"))
            raise

    async def _execute_twap(
        self,
        execution_id: str,
        execution_request: ExecutionRequest,
    ) -> ExecutionResult:
        """Execute order using TWAP strategy."""
        try:
            if not self.order_manager:
                msg = "Order manager not initialized"
                raise ValueError(msg)

            # TWAP divides the order into equal parts over time
            total_quantity = execution_request.quantity
            execution_duration = execution_request.timeout_seconds
            num_slices = max(1, execution_duration // 10)  # Slice every 10 seconds
            slice_quantity = total_quantity / num_slices
            slice_interval = execution_duration / num_slices

            executed_quantity = 0
            total_cost = 0
            orders_placed = []

            for i in range(num_slices):
                # Create slice order request
                slice_request = ExecutionRequest(
                    symbol=execution_request.symbol,
                    side=execution_request.side,
                    order_type=execution_request.order_type,
                    quantity=slice_quantity,
                    price=execution_request.price,
                    stop_price=execution_request.stop_price,
                    leverage=execution_request.leverage,
                    strategy_type=execution_request.strategy_type,
                    execution_strategy=ExecutionStrategy.IMMEDIATE,
                )

                # Execute slice
                slice_result = await self._execute_immediate(
                    f"{execution_id}_twap_{i}",
                    slice_request,
                )

                executed_quantity += slice_result.executed_quantity
                total_cost += slice_result.total_cost
                orders_placed.extend(slice_result.orders_placed)

                # Wait for next slice
                if i < num_slices - 1:
                    await asyncio.sleep(slice_interval)

            # Calculate average price
            average_price = (
                total_cost / executed_quantity if executed_quantity > 0 else 0
            )
            slippage = self._calculate_slippage(execution_request.price, average_price)

            return ExecutionResult(
                execution_id=execution_id,
                symbol=execution_request.symbol,
                side=execution_request.side,
                order_type=execution_request.order_type,
                requested_quantity=execution_request.quantity,
                executed_quantity=executed_quantity,
                average_price=average_price,
                total_cost=total_cost,
                slippage=slippage,
                execution_time=0,  # Will be set by caller
                status=ExecutionStatus.COMPLETED,
                orders_placed=orders_placed,
                performance_metrics={},
            )

        except Exception:
            self.print(execution_error("Error in TWAP execution: {e}"))
            raise

    async def _execute_vwap(
        self,
        execution_id: str,
        execution_request: ExecutionRequest,
    ) -> ExecutionResult:
        """Execute order using VWAP strategy."""
        try:
            # VWAP execution - simplified implementation
            # In a real implementation, this would use actual volume data
            return await self._execute_twap(execution_id, execution_request)

        except Exception:
            self.print(execution_error("Error in VWAP execution: {e}"))
            raise

    async def _execute_iceberg(
        self,
        execution_id: str,
        execution_request: ExecutionRequest,
    ) -> ExecutionResult:
        """Execute order using iceberg strategy."""
        try:
            # Iceberg execution - simplified implementation
            # In a real implementation, this would use actual order book data
            return await self._execute_batch(execution_id, execution_request)

        except Exception:
            self.print(execution_error("Error in iceberg execution: {e}"))
            raise

    async def _execute_adaptive(
        self,
        execution_id: str,
        execution_request: ExecutionRequest,
    ) -> ExecutionResult:
        """Execute order using adaptive strategy based on market conditions."""
        try:
            # Get optimal parameters from Optuna
            if self.optuna_study:
                optimal_params = self._get_optimal_execution_params(execution_request)

                # Use optimal parameters for execution
                execution_request.max_slippage = optimal_params.get(
                    "max_slippage",
                    execution_request.max_slippage,
                )
                execution_request.timeout_seconds = optimal_params.get(
                    "timeout_seconds",
                    execution_request.timeout_seconds,
                )

                if optimal_params.get("use_batch", False):
                    execution_request.execution_strategy = ExecutionStrategy.BATCH
                    execution_request.batch_size = optimal_params.get(
                        "batch_size",
                        execution_request.quantity * 0.1,
                    )
                    execution_request.batch_interval = optimal_params.get(
                        "batch_interval",
                        5,
                    )

            # Execute with adaptive strategy
            return await self._execute_immediate(execution_id, execution_request)

        except Exception:
            self.print(execution_error("Error in adaptive execution: {e}"))
            raise

    def _calculate_slippage(
        self,
        target_price: float | None,
        actual_price: float,
    ) -> float:
        """Calculate slippage percentage."""
        if not target_price or target_price == 0:
            return 0.0
        return abs(actual_price - target_price) / target_price

    async def _optimization_worker(self) -> None:
        """Background worker for parameter optimization."""
        try:
            while True:
                # Run optimization every hour
                await asyncio.sleep(3600)

                if self.optuna_study and len(self.execution_history) > 10:
                    await self._run_parameter_optimization()

        except asyncio.CancelledError:
            self.logger.info("Optimization worker cancelled")
        except Exception:
            self.print(error("Error in optimization worker: {e}"))

    async def _run_parameter_optimization(self) -> None:
        """Run parameter optimization using Optuna."""
        try:

            def objective(trial):
                # Define hyperparameters to optimize
                trial.suggest_float("max_slippage", 0.0001, 0.01)
                trial.suggest_int("timeout_seconds", 10, 300)
                trial.suggest_categorical("use_batch", [True, False])
                trial.suggest_float("batch_size_ratio", 0.05, 0.5)
                trial.suggest_int("batch_interval", 1, 30)

                # Calculate objective value based on recent execution history
                recent_executions = self.execution_history[-50:]  # Last 50 executions

                if not recent_executions:
                    return 0.0

                # Calculate success rate and average slippage
                success_count = sum(
                    1
                    for e in recent_executions
                    if e.status == ExecutionStatus.COMPLETED
                )
                success_rate = success_count / len(recent_executions)

                avg_slippage = np.mean([e.slippage for e in recent_executions])

                # Objective: maximize success rate while minimizing slippage
                return success_rate - avg_slippage

            # Run optimization
            self.optuna_study.optimize(objective, n_trials=10)

            self.logger.info(
                f"Parameter optimization completed. Best value: {self.optuna_study.best_value}",
            )

        except Exception:
            self.print(error("Error in parameter optimization: {e}"))

    def _get_optimal_execution_params(
        self,
        execution_request: ExecutionRequest,
    ) -> dict[str, Any]:
        """Get optimal execution parameters from Optuna study."""
        try:
            if not self.optuna_study or not self.optuna_study.best_params:
                return {}

            return self.optuna_study.best_params

        except Exception:
            self.print(error("Error getting optimal parameters: {e}"))
            return {}

    async def _analytics_worker(self) -> None:
        """Background worker for analytics and reporting."""
        try:
            while True:
                # Generate analytics every 5 minutes
                await asyncio.sleep(300)

                if self.advanced_reporter and self.execution_history:
                    await self._generate_execution_analytics()

        except asyncio.CancelledError:
            self.logger.info("Analytics worker cancelled")
        except Exception:
            self.print(error("Error in analytics worker: {e}"))

    async def _generate_execution_analytics(self) -> None:
        """Generate execution analytics and reports."""
        try:
            # Prepare performance data
            performance_data = {
                "executions": self.execution_history,
                "metrics": self.performance_metrics,
                "timestamp": datetime.now().isoformat(),
            }

            # Generate real-time report
            if self.advanced_reporter:
                report = await self.advanced_reporter.generate_real_time_report(
                    performance_data,
                )

                # Log key metrics
                if report:
                    self.logger.info(
                        f"Analytics Report - Success Rate: {report.get('real_time_metrics', {}).get('success_rate', 0):.2%}",
                    )

        except Exception:
            self.print(execution_error("Error generating execution analytics: {e}"))

    async def _update_performance_metrics(
        self,
        execution_result: ExecutionResult,
    ) -> None:
        """Update performance metrics based on execution result."""
        try:
            # Calculate basic metrics
            success_rate = len(
                [
                    e
                    for e in self.execution_history
                    if e.status == ExecutionStatus.COMPLETED
                ],
            ) / len(self.execution_history)
            avg_slippage = np.mean([e.slippage for e in self.execution_history])
            avg_execution_time = np.mean(
                [e.execution_time for e in self.execution_history],
            )

            self.performance_metrics.update(
                {
                    "success_rate": success_rate,
                    "avg_slippage": avg_slippage,
                    "avg_execution_time": avg_execution_time,
                    "total_executions": len(self.execution_history),
                    "last_updated": datetime.now().isoformat(),
                },
            )

        except Exception:
            self.print(error("Error updating performance metrics: {e}"))

    def get_execution_status(self, execution_id: str) -> ExecutionResult | None:
        """Get the status of an execution."""
        try:
            for result in self.execution_history:
                if result.execution_id == execution_id:
                    return result
            return None

        except Exception:
            self.print(execution_error("Error getting execution status: {e}"))
            return None

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def get_execution_history(
        self,
        limit: int | None = None,
    ) -> list[ExecutionResult]:
        """Get execution history."""
        try:
            if limit:
                return self.execution_history[-limit:]
            return self.execution_history.copy()

        except Exception:
            self.print(execution_error("Error getting execution history: {e}"))
            return []

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="async order executor cleanup",
    )
    async def stop(self) -> None:
        """Clean up the async order executor."""
        try:
            self.logger.info("🛑 Stopping Async Order Executor...")

            # Stop background tasks
            tasks = [
                t
                for t in [
                    self.execution_task,
                    self.optimization_task,
                    self.analytics_task,
                ]
                if t
            ]
            for t in tasks:
                t.cancel()
            for t in tasks:
                with contextlib.suppress(Exception):
                    await t

            self.logger.info("✅ Async Order Executor stopped successfully")

        except Exception:
            self.print(error("Error stopping async order executor: {e}"))


# Factory function for creating async order executor
@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="async order executor setup",
)
async def setup_async_order_executor(
    config: dict[str, Any] | None = None,
) -> AsyncOrderExecutor | None:
    """
    Setup and initialize async order executor.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized AsyncOrderExecutor instance
    """
    try:
        if config is None:
            config = {}

        executor = AsyncOrderExecutor(config)
        success = await executor.initialize()

        if success:
            return executor
        return None

    except Exception:
        print(warning("Error setting up async order executor: {e}"))
        return None
