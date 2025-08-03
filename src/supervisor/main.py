# src/supervisor/main.py

import asyncio
from datetime import datetime
from typing import Any

# Import necessary modules
from src.config import (
    CONFIG,
    settings,
)  # Import settings for trading_mode, and CONFIG for other params
from src.paper_trader import PaperTrader  # Import PaperTrader for paper trading mode
from src.sentinel.sentinel import Sentinel
from src.supervisor.ab_tester import ABTester
from src.supervisor.monitoring import Monitoring
from src.supervisor.performance_reporter import PerformanceReporter
from src.supervisor.risk_allocator import RiskAllocator
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.state_manager import StateManager


class Supervisor:
    """
    The central, real-time orchestrator of the Ares Trading Bot.
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

        # Initialize Supervisor sub-components, passing necessary dependencies
        self.risk_allocator = RiskAllocator(self.config)
        self.performance_reporter = PerformanceReporter(
            self.config,
            self.db_manager,
        )  # Pass db_manager
        self.ab_tester = ABTester(self.config, self.performance_reporter)
        self.monitoring = Monitoring(self.db_manager)

        # Determine the actual trading client (PaperTrader or live exchange_client)
        if settings.trading_environment == "PAPER":
            self.trader = PaperTrader(
                symbol=self.symbol,
                exchange_name=self.exchange_name,
                config=self.config,
            )
            self.logger.info("Paper Trader initialized for simulation.")
        elif settings.trading_environment == "LIVE":
            self.trader = (
                exchange_client  # Use the live exchange client passed from main
            )
            self.logger.info(
                "Live Trader (BinanceExchange) initialized for live operations.",
            )
        else:
            self.trader = None
            self.logger.error(
                f"Unknown trading environment: '{settings.trading_environment}'. Trading will be disabled.",
            )
            raise ValueError(
                f"Invalid TRADING_ENVIRONMENT: {settings.trading_environment}",
            )  # Halt if invalid

        # Initialize ModelManager first, which will load the champion models
        # Pass performance_reporter to ModelManager so it can pass it to Tactician
        from src.utils.model_manager import ModelManager

        self.model_manager = ModelManager(
            database_manager=self.db_manager,
            performance_reporter=self.performance_reporter,
        )

        # Initialize the core real-time components, getting instances from ModelManager
        if self.trader:
            self.sentinel = Sentinel(
                self.trader,
                self.state_manager,
            )  # Sentinel needs the real trader
            self.analyst = (
                self.model_manager.get_analyst()
            )  # Get Analyst instance from ModelManager
            self.strategist = (
                self.model_manager.get_strategist()
            )  # Get Strategist instance from ModelManager
            # Tactician instance is already created by ModelManager with performance_reporter
            self.tactician = self.model_manager.get_tactician()

            # Ensure the Analyst, Strategist, Tactician instances from ModelManager
            # have their exchange_client and state_manager set if they need it for live ops.
            # This is a critical point for dependency injection.
            # For the training pipeline, these are mostly placeholders.
            if hasattr(self.analyst, "exchange") and self.analyst.exchange is None:
                self.analyst.exchange = self.trader
            if (
                hasattr(self.analyst, "state_manager")
                and self.analyst.state_manager is None
            ):
                self.analyst.state_manager = self.state_manager

            if (
                hasattr(self.strategist, "exchange")
                and self.strategist.exchange is None
            ):
                self.strategist.exchange = self.trader
            if (
                hasattr(self.strategist, "state_manager")
                and self.strategist.state_manager is None
            ):
                self.strategist.state_manager = self.state_manager

            if hasattr(self.tactician, "exchange") and self.tactician.exchange is None:
                self.tactician.exchange = self.trader
            if (
                hasattr(self.tactician, "state_manager")
                and self.tactician.state_manager is None
            ):
                self.tactician.state_manager = self.state_manager

        else:
            self.sentinel = None
            self.analyst = None
            self.strategist = None
            self.tactician = None
            self.logger.critical(
                "Core trading components not initialized due to invalid trading environment.",
            )

        self.running = False

        self.market_data_queue = asyncio.Queue(maxsize=100)
        self.analysis_queue = asyncio.Queue(maxsize=100)
        self.signal_queue = asyncio.Queue(maxsize=50)

    async def start(self):
        """
        Starts all bot components and the main processing loop.
        """
        self.logger.info("Supervisor starting all components...")
        self.running = True

        if hasattr(self.db_manager, "initialize") and asyncio.iscoroutinefunction(
            self.db_manager.initialize,
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
                        "direction": "LONG"
                        if float(position["positionAmt"]) > 0
                        else "SHORT",
                        "trade_id": self.state_manager.get_state(
                            "current_position",
                            {},
                        ).get("trade_id"),  # Attempt to recover trade_id
                        "entry_timestamp": self.state_manager.get_state(
                            "current_position",
                            {},
                        ).get("entry_timestamp"),  # Attempt to recover timestamp
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
                f"Failed to synchronize state with exchange: {e}",
                exc_info=True,
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

    @handle_specific_errors(
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
            self.logger.error(f"❌ Main Supervisor initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="supervisor configuration loading",
    )
    async def _load_supervisor_configuration(self) -> None:
        try:
            self.supervisor_config.setdefault("run_interval", 60)
            self.supervisor_config.setdefault("max_history", 100)
            self.run_interval = self.supervisor_config["run_interval"]
            self.max_history = self.supervisor_config["max_history"]
            self.logger.info("Main supervisor configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading supervisor configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
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
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
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
            self.logger.error(f"Error in main supervisor run: {e}")
            self.is_running = False
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="supervise step",
    )
    async def _supervise(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {"timestamp": now, "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            self.logger.info(f"Main Supervisor tick at {now}")
        except Exception as e:
            self.logger.error(f"Error in supervise step: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="main supervisor stop",
    )
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Main Supervisor...")
        try:
            self.is_running = False
            self.status = {"timestamp": datetime.now().isoformat(), "status": "stopped"}
            self.logger.info("✅ Main Supervisor stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping main supervisor: {e}")

    def get_status(self) -> dict[str, Any]:
        return self.status.copy()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history


main_supervisor: MainSupervisor | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="main supervisor setup",
)
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
