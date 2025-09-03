from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from src.config import CONFIG, get_environment_settings
from src.core.decorators import handles_errors
from src.paper_trader import PaperTrader
from src.sentinel.sentinel import Sentinel
from src.supervisor.ab_tester import ABTester
from src.supervisor.monitoring import Monitoring
from src.supervisor.performance_reporter import PerformanceReporter
from src.supervisor.risk_allocator import RiskAllocator
from src.utils.logger import system_logger
from src.utils.model_manager import ModelManager
from src.utils.state_manager import StateManager
from src.supervisor.dependency_container import DependencyContainer, ComponentBuilder

# src/supervisor/main.py


class Supervisor:
    """
    The central real-time orchestrator of the Ares Trading Bot.
    It initializes, manages, and connects all the core components of the
    trading pipeline, ensuring they run concurrently and communicate efficiently.
    """

    def __init__(
        self,
        symbol: str,
        exchange_name: str,
        exchange_client: Any,
        state_manager: StateManager,
        db_manager: Any,
    ):  # Accept the exchange client from main.py
        self.logger = system_logger.getChild("Supervisor")
        self.state_manager = state_manager  # Use the passed state_manager
        self.symbol = symbol
        self.exchange_name = exchange_name
        self.state = (
            self.state_manager.get_state(  # Use get_state() to load current state
                "global_trading_status",
            )
        )  # Use get_state() to load current state
        self.config = CONFIG  # Use the global CONFIG dictionary for general settings
        self.db_manager = db_manager  # Store the database manager

        # Initialize dependency container and component builder
        self.dependency_container = DependencyContainer(self.config)
        self.component_builder = ComponentBuilder(self.dependency_container)

        # Initialize Supervisor sub-components, passing necessary dependencies
        self.risk_allocator = RiskAllocator(self.config)
        self.performance_reporter = PerformanceReporter(
            self.config,
            self.db_manager,
        )  # Pass db_manager
        self.ab_tester = ABTester(self.config, self.performance_reporter)
        self.monitoring = Monitoring(self.db_manager)

        # Determine the actual trading client (PaperTrader or live exchange_client)
        env_settings = get_environment_settings()
        if env_settings.trading_environment == "PAPER":
            self.trader = PaperTrader(
                symbol=self.symbol, exchange_name=self.exchange_name, config=self.config
            )
            self.logger.info("Paper Trader initialized for simulation.")
        elif env_settings.trading_environment == "LIVE":
            self.trader = (
                exchange_client  # Use the live exchange client passed from main
            )
            self.logger.info(
                "Live Trader (BinanceExchange) initialized for live operations.",
            )
        else:
            self.trader = None
            self.logger.error(
                f"Unknown trading environment: '{env_settings.trading_environment}'. Trading will be disabled.",
            )
            msg = f"Invalid TRADING_ENVIRONMENT: {env_settings.trading_environment}"
            raise ValueError(
                msg,
            )  # Halt if invalid

        # Initialize ModelManager first, which will load the champion models
        # Pass performance_reporter to ModelManager so it can pass it to Tactician

        self.model_manager = ModelManager(
            database_manager=self.db_manager,
            performance_reporter=self.performance_reporter,
        )

        # Register component factories in the dependency container
        if self.trader:
            # Register component factories
            self.dependency_container.register(
                "sentinel", 
                self.component_builder.build_sentinel(self.trader, self.state_manager)
            )
            self.dependency_container.register(
                "analyst",
                self.component_builder.build_analyst(self.trader, self.state_manager)
            )
            self.dependency_container.register(
                "strategist",
                self.component_builder.build_strategist(self.trader, self.state_manager)
            )
            self.dependency_container.register(
                "tactician",
                self.component_builder.build_tactician(self.trader, self.state_manager, self.performance_reporter)
            )
            
            # Initialize components through dependency container
            self.sentinel = self.dependency_container.get("sentinel")
            self.analyst = self.dependency_container.get("analyst")
            self.strategist = self.dependency_container.get("strategist")
            self.tactician = self.dependency_container.get("tactician")

        else:
            self.sentinel = None
            self.analyst = None
            self.strategist = None
            self.tactician = None
            self.logger.critical(
                "Core trading components not initialized due to invalid trading environment.",
            )

        self.running = False

        # Initialize communication queues
        self.market_data_queue = asyncio.Queue(maxsize=100)
        self.analysis_queue = asyncio.Queue(maxsize=100)
        self.signal_queue = asyncio.Queue(maxsize=50)
        
        # Wire up queue connections between components
        self._wire_component_queues()

    def _wire_component_queues(self):
        """
        Explicitly wire up communication queues between components.
        This makes the data flow between components clear and traceable.
        """
        if not (self.sentinel and self.analyst and self.strategist and self.tactician):
            self.logger.warning("Cannot wire queues: Not all components are initialized")
            return
            
        # Wire Sentinel -> Analyst (market data flow)
        if hasattr(self.sentinel, 'output_queue'):
            self.sentinel.output_queue = self.market_data_queue
        if hasattr(self.analyst, 'input_queue'):
            self.analyst.input_queue = self.market_data_queue
            
        # Wire Analyst -> Strategist (analysis results flow)
        if hasattr(self.analyst, 'output_queue'):
            self.analyst.output_queue = self.analysis_queue
        if hasattr(self.strategist, 'input_queue'):
            self.strategist.input_queue = self.analysis_queue
            
        # Wire Strategist -> Tactician (signals flow)
        if hasattr(self.strategist, 'output_queue'):
            self.strategist.output_queue = self.signal_queue
        if hasattr(self.tactician, 'input_queue'):
            self.tactician.input_queue = self.signal_queue
            
        self.logger.info("Component queues wired successfully")

    @handles_errors(fallback=None)
    async def start(self):
        """
        Starts all bot components and the main processing loop.
        """
        self.logger.info("Supervisor starting all components...")
        self.running = True

        if hasattr(self.db_manager, "initialize") and asyncio.iscoroutinefunction(
            self.db_manager.initialize
        ):
            await self.db_manager.initialize()

        tasks = []
        if (
            self.trader
            and self.sentinel
            and self.analyst
            and self.strategist
            and self.tactician
        ):
            tasks.extend(
                [
                    asyncio.create_task(self.sentinel.start(), name="Sentinel_Task"),
                    asyncio.create_task(self.analyst.start(), name="Analyst_Task"),
                    asyncio.create_task(
                        self.strategist.start(),
                        name="Strategist_Task",
                    ),
                    asyncio.create_task(self.tactician.start(), name="Tactician_Task"),
                ],
            )
            if isinstance(self.trader, PaperTrader):
                tasks.append(
                    asyncio.create_task(
                        self.trader.run_simulation(),
                        name="PaperTrader_Simulation_Task",
                    ),
                )
        else:
            self.logger.error(
                "Cannot start supervisor: Core trading components are not initialized.",
            )
            self.running = False
            return

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info(
                "Supervisor tasks cancelled. Beginning graceful shutdown...",
            )
        finally:
            self.running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            if self.trader and hasattr(self.trader, "close"):
                await self.trader.close()
            self.state_manager._save_state_to_file()  # Call internal save method
            self.logger.info(
                "All components have been shut down and state has been saved.",
            )

    @handles_errors(fallback=None)
    async def _synchronize_exchange_state(self):
        """
        Fetches the current account equity and open positions from the exchange
        and updates the persistent state. This is key for crash recovery.
        """
        try:
            # 1. Update account equity and peak equity
            account_info = await self.trader.get_account_info()  # Use self.trader
            current_equity = float(account_info.get("totalWalletBalance", 0))

            if current_equity > 0:
                self.state_manager.set_state("account_equity", current_equity)
                self.logger.debug(f"Updated account equity: ${current_equity:,.2f}")

                peak_equity = self.state_manager.get_state(
                    "global_peak_equity",
                )  # Use global_peak_equity from state
                if current_equity > peak_equity:
                    self.state_manager.set_state("global_peak_equity", current_equity)
                    self.logger.info(f"New peak equity reached: ${current_equity:,.2f}")
            else:
                self.logger.warning("Could not retrieve a valid account balance.")

            # 2. Update open positions state for crash recovery
            open_positions = await self.trader.get_open_positions()  # Use self.trader
            symbol = self.symbol
            active_position_on_exchange = None

            for position in open_positions:
                if (
                    position.get("symbol") == symbol
                    and float(position.get("positionAmt", 0)) != 0
                ):
                    # Capture more details for active_position
                    active_position_on_exchange = {
                        "symbol": position["symbol"],
                        "amount": float(position["positionAmt"]),
                        "entry_price": float(position["entryPrice"]),
                        "leverage": int(position.get("leverage", 1)),
                        "direction": (
                            "LONG" if float(position["positionAmt"]) > 0 else "SHORT"
                        ),
                        "trade_id": self.state_manager.get_state(
                            "current_position",
                            {},
                        ).get(
                            "trade_id",
                        ),  # Attempt to recover trade_id
                        "entry_timestamp": self.state_manager.get_state(
                            "current_position",
                            {},
                        ).get(
                            "entry_timestamp",
                        ),  # Attempt to recover timestamp
                        "stop_loss": self.state_manager.get_state(
                            "current_position",
                            {},
                        ).get("stop_loss"),
                        "take_profit": self.state_manager.get_state(
                            "current_position",
                            {},
                        ).get("take_profit"),
                        "entry_fees_usd": self.state_manager.get_state(
                            "current_position",
                            {},
                        ).get("entry_fees_usd", 0.0),
                        "entry_context": self.state_manager.get_state(
                            "current_position",
                            {},
                        ).get("entry_context", {}),
                    }
                    self.logger.debug(
                        f"Found active position on exchange for {symbol}.",
                    )
                    break

            # Synchronize the state file with what's on the exchange
            current_state_position = self.state_manager.get_state(
                "current_position",
            )  # Use 'current_position'

            # Only update if there's a meaningful change or new position found
            if active_position_on_exchange != current_state_position:
                self.logger.info(
                    f"State mismatch or update: Synchronizing position state with exchange. New state: {active_position_on_exchange}",
                )
                self.state_manager.set_state(
                    "current_position",
                    active_position_on_exchange,
                )  # Update 'current_position'

        except Exception as e:
            self.logger.error(
                f"Failed to synchronize state with exchange: {e}", exc_info=True
            )


class MainSupervisor:
    """
    Main Supervisor Entrypoint with DI, type hints, and robust error handling.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MainSupervisor")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.supervisor_config: dict[str, Any] = self.config.get("main_supervisor", {})
        self.run_interval: int = self.supervisor_config.get("run_interval", 60)
        self.max_history: int = self.supervisor_config.get("max_history", 100)

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid main supervisor configuration"),
            AttributeError: (False, "Missing required main supervisor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="main supervisor initialization",
    )
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Main Supervisor...")
            await self._load_supervisor_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for main supervisor")
                return False
            self.logger.info("✅ Main Supervisor initialization completed successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Main Supervisor initialization failed: {e}")
            return False

    @handles_errors(fallback=None)
    async def _load_supervisor_configuration(self) -> None:
        try:
            self.supervisor_config.setdefault("run_interval", 60)
            self.supervisor_config.setdefault("max_history", 100)
            self.run_interval = self.supervisor_config["run_interval"]
            self.max_history = self.supervisor_config["max_history"]
            self.logger.info("Main supervisor configuration loaded successfully")
        except Exception as e:
            self.logger.exception(f"Error loading supervisor configuration: {e}")

    @handles_errors(fallback=False)
    def _validate_configuration(self) -> bool:
        try:
            if self.run_interval <= 0:
                self.logger.error("Invalid run interval")
                return False
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception as e:
            self.logger.exception(f"Error validating configuration: {e}")
            return False

    @handles_errors(
        error_handlers={
            Exception: (False, "Supervisor run failed"),
        },
        default_return=False,
        context="main supervisor run",
    )
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info("🚦 Main Supervisor started.")
            while self.is_running:
                await self._supervise()
                await asyncio.sleep(self.run_interval)
            return True
        except Exception as e:
            self.logger.exception(f"Error in main supervisor run: {e}")
            self.is_running = False
            return False

    @handles_errors(fallback=None)
    async def _supervise(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {"timestamp": now, "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            self.logger.info(f"Main Supervisor tick at {now}")
        except Exception as e:
            self.logger.exception(f"Error in supervise step: {e}")

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Main Supervisor...")
        try:
            self.is_running = False
            self.status = {"timestamp": datetime.now().isoformat(), "status": "stopped"}
            self.logger.info("✅ Main Supervisor stopped successfully")
        except Exception as e:
            self.logger.exception(f"Error stopping main supervisor: {e}")

    def get_status(self) -> dict[str, Any]:
        return self.status.copy()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history


main_supervisor: MainSupervisor | None = None


@handles_errors(fallback=None)
async def setup_main_supervisor(
    config: dict[str, Any] | None = None,
) -> MainSupervisor | None:
    try:
        global main_supervisor
        if config is None:
            config = {"main_supervisor": {"run_interval": 60, "max_history": 100}}
        main_supervisor = MainSupervisor(config)
        success = await main_supervisor.initialize()
        if success:
            return main_supervisor
        return None
    except Exception as e:
        print(f"Error setting up main supervisor: {e}")
        return None
