import asyncio
from datetime import datetime
from typing import Any
from ..config.config import CONFIG, get_environment_settings
from ..paper_trader import PaperTrader
# from .ab_tester import ABTester  # Module not found, using mock class

class ABTester:
    """Mock ABTester class until the actual module is implemented."""
    def __init__(self, config: dict, performance_reporter: Any) -> None:
        self.config = config
        self.performance_reporter = performance_reporter
from .dependency_container import ComponentBuilder, DependencyContainer
from .monitoring import Monitoring
from .performance_reporter import PerformanceReporter
from .risk_allocator import RiskAllocator
from ..utils.logger import system_logger
from ..utils.model_manager import ModelManager
from ..utils.state_manager import StateManager
from ..utils.config.loaders import initialize_sr_parameters
# Enhanced error handling and performance monitoring
from src.utils.enhanced_error_handler import handle_errors_with_tracking
from src.utils.warning_symbols import failed, initialization_error, warning
from src.utils.performance_utils import PerformanceMonitor, global_monitor
from src.utils.caching import intelligent_caching
# Live trading utilities
from src.utils.model_manager import ModelManager

from src.core.decorators import handles_errors
import logging
import time

class Supervisor:
    """
    The central real-time orchestrator of the Ares Trading Bot.
    It initializes, manages, and connects all the core components of the
    trading pipeline, ensuring they run concurrently and communicate efficiently.
    """

    def __init__(self, symbol: str, exchange_name: str, exchange_client: Any, state_manager: StateManager, db_manager: Any) -> None:
        self.logger = system_logger.getChild('Supervisor')
        self.state_manager = state_manager
        self.symbol = symbol
        self.exchange_name = exchange_name
        self.state = self.state_manager.get_state('global_trading_status')
        self.config = CONFIG
        self.db_manager = db_manager
        self.dependency_container = DependencyContainer(self.config)
        self.component_builder = ComponentBuilder(self.dependency_container)
        self.risk_allocator = RiskAllocator(self.config)
        self.performance_reporter = PerformanceReporter(self.config, self.db_manager)
        self.ab_tester = ABTester(self.config, self.performance_reporter)
        self.monitoring = Monitoring(self.db_manager)
        env_settings = get_environment_settings()
        if env_settings.trading_environment == 'PAPER':
            self.trader = PaperTrader(symbol = self.symbol, exchange_name = self.exchange_name, config = self.config)
            self.logger.info('Paper Trader initialized for simulation.')
        elif env_settings.trading_environment == 'LIVE':
            self.trader = exchange_client
            self.logger.info('Live Trader (BinanceExchange) initialized for live operations.')
        else:
            self.trader = None
            self.logger.error(f"Unknown trading environment: '{env_settings.trading_environment}'. Trading will be disabled.")
            msg = f'Invalid TRADING_ENVIRONMENT: {env_settings.trading_environment}'
            raise ValueError(msg)
        self.model_manager = ModelManager(database_manager = self.db_manager, performance_reporter = self.performance_reporter)
        
        # Live trading utilities
        self.model_manager: ModelManager | None = None
        self.selected_models: dict[str, str] = {}
        self.model_cache: dict[str, Any] = {}
        
        # Performance monitoring for live trading
        self.performance_monitor: PerformanceMonitor | None = None
        self.global_monitor = global_monitor
        self.supervision_cache: dict[str, Any] = {}
        
        if self.trader:
            self.dependency_container.register('sentinel', self.component_builder.build_sentinel(self.trader, self.state_manager))
            self.dependency_container.register('analyst', self.component_builder.build_analyst(self.trader, self.state_manager))
            self.dependency_container.register('strategist', self.component_builder.build_strategist(self.trader, self.state_manager))
            self.dependency_container.register('tactician', self.component_builder.build_tactician(self.trader, self.state_manager, self.performance_reporter))
            self.sentinel = self.dependency_container.get('sentinel')
            self.analyst = self.dependency_container.get('analyst')
            self.strategist = self.dependency_container.get('strategist')
            self.tactician = self.dependency_container.get('tactician')
        else:
            self.sentinel = None
            self.analyst = None
            self.strategist = None
            self.tactician = None
            self.logger.critical('Core trading components not initialized due to invalid trading environment.')
        self.running = False
        self.market_data_queue = asyncio.Queue(maxsize = 100)
        self.analysis_queue = asyncio.Queue(maxsize = 100)
        self.signal_queue = asyncio.Queue(maxsize = 50)
        self._wire_component_queues()

    def _wire_component_queues(self) -> None:
        """
        Explicitly wire up communication queues between components.
        This makes the data flow between components clear and traceable.
        """
        if not (self.sentinel and self.analyst and self.strategist and self.tactician):
            self.logger.warning('Cannot wire queues: Not all components are initialized')
            return
        if hasattr(self.sentinel, 'output_queue'):
            self.sentinel.output_queue = self.market_data_queue
        if hasattr(self.analyst, 'input_queue'):
            self.analyst.input_queue = self.market_data_queue
        if hasattr(self.analyst, 'output_queue'):
            self.analyst.output_queue = self.analysis_queue
        if hasattr(self.strategist, 'input_queue'):
            self.strategist.input_queue = self.analysis_queue
        if hasattr(self.strategist, 'output_queue'):
            self.strategist.output_queue = self.signal_queue
        if hasattr(self.tactician, 'input_queue'):
            self.tactician.input_queue = self.signal_queue
        self.logger.info('Component queues wired successfully')

    @handles_errors(fallback = None)
    async def start(self) -> None:
        """
        Starts all bot components and the main processing loop.
        """
        self.logger.info('Supervisor starting all components...')
        self.running = True
        if hasattr(self.db_manager, 'initialize') and asyncio.iscoroutinefunction(self.db_manager.initialize):
            await self.db_manager.initialize()
        tasks = []
        if self.trader and self.sentinel and self.analyst and self.strategist and self.tactician:
            tasks.extend([asyncio.create_task(self.sentinel.start(), name='Sentinel_Task'), asyncio.create_task(self.analyst.start(), name='Analyst_Task'), asyncio.create_task(self.strategist.start(), name='Strategist_Task'), asyncio.create_task(self.tactician.start(), name='Tactician_Task')])
            if isinstance(self.trader, PaperTrader):
                tasks.append(asyncio.create_task(self.trader.run_simulation(), name='PaperTrader_Simulation_Task'))
        else:
            self.logger.error('Cannot start supervisor: Core trading components are not initialized.')
            self.running = False
            return
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info('Supervisor tasks cancelled. Beginning graceful shutdown...')
        finally:
            self.running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions = True)
            if self.trader and hasattr(self.trader, 'close'):
                await self.trader.close()
            self.state_manager._save_state_to_file()
            self.logger.info('All components have been shut down and state has been saved.')

    @handles_errors(fallback = None)
    async def _synchronize_exchange_state(self) -> None:
        """
        Fetches the current account equity and open positions from the exchange
        and updates the persistent state. This is key for crash recovery.
        """
        try:
            account_info = await self.trader.get_account_info()
            current_equity = float(account_info.get('totalWalletBalance', 0))
            if current_equity > 0:
                self.state_manager.set_state('account_equity', current_equity)
                self.logger.debug(f'Updated account equity: ${current_equity:,.2f}')
                peak_equity = self.state_manager.get_state('global_peak_equity')
                if current_equity > peak_equity:
                    self.state_manager.set_state('global_peak_equity', current_equity)
                    self.logger.info(f'New peak equity reached: ${current_equity:,.2f}')
            else:
                self.logger.warning('Could not retrieve a valid account balance.')
            open_positions = await self.trader.get_open_positions()
            symbol = self.symbol
            active_position_on_exchange = None
            for position in open_positions:
                if position.get('symbol') == symbol and float(position.get('positionAmt', 0)) != 0:
                    active_position_on_exchange = {'symbol': position['symbol'], 'amount': float(position['positionAmt']), 'entry_price': float(position['entryPrice']), 'leverage': int(position.get('leverage', 1)), 'direction': 'LONG' if float(position['positionAmt']) > 0 else 'SHORT', 'trade_id': self.state_manager.get_state('current_position', {}).get('trade_id'), 'entry_timestamp': self.state_manager.get_state('current_position', {}).get('entry_timestamp'), 'stop_loss': self.state_manager.get_state('current_position', {}).get('stop_loss'), 'take_profit': self.state_manager.get_state('current_position', {}).get('take_profit'), 'entry_fees_usd': self.state_manager.get_state('current_position', {}).get('entry_fees_usd', 0.0), 'entry_context': self.state_manager.get_state('current_position', {}).get('entry_context', {})}
                    self.logger.debug(f'Found active position on exchange for {symbol}.')
                    break
            current_state_position = self.state_manager.get_state('current_position')
            if active_position_on_exchange != current_state_position:
                self.logger.info(f'State mismatch or update: Synchronizing position state with exchange. New state: {active_position_on_exchange}')
                self.state_manager.set_state('current_position', active_position_on_exchange)
        except Exception as e:
            self.logger.error(f'Failed to synchronize state with exchange: {e}', exc_info = True)

class MainSupervisor:
    """
    Main Supervisor Entrypoint with DI, type hints, and robust error handling.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild('MainSupervisor')
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.supervisor_config: dict[str, Any] = self.config.get('main_supervisor', {})
        self.run_interval: int = self.supervisor_config.get('run_interval', 60)
        self.max_history: int = self.supervisor_config.get('max_history', 100)

    @handles_errors(error_handlers={ValueError: (False, 'Invalid main supervisor configuration'), AttributeError: (False, 'Missing required main supervisor parameters'), KeyError: (False, 'Missing configuration keys')}, default_return = False, context='main supervisor initialization')
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Main Supervisor...")
            
            # Load optimized S/R parameters from step 2.5
            self.logger.info("Loading optimized S/R parameters...")
            initialize_sr_parameters(self.config)
            
            await self._load_supervisor_configuration()
            if not self._validate_configuration():
                self.logger.error('Invalid configuration for main supervisor')
                return False
            
            # Initialize live trading utilities
            await self._initialize_live_trading_utilities()
            
            # Initialize performance monitoring
            await self._initialize_performance_monitoring()
            
            self.logger.info('✅ Main Supervisor initialization completed successfully')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Main Supervisor initialization failed: {e}')
            return False

    @handles_errors(fallback = None)
    async def _load_supervisor_configuration(self) -> None:
        try:
            self.supervisor_config.setdefault('run_interval', 60)
            self.supervisor_config.setdefault('max_history', 100)
            self.run_interval = self.supervisor_config['run_interval']
            self.max_history = self.supervisor_config['max_history']
            self.logger.info('Main supervisor configuration loaded successfully')
        except Exception as e:
            self.logger.exception(f'Error loading supervisor configuration: {e}')

    @handles_errors(fallback = False)
    def _validate_configuration(self) -> bool:
        try:
            if self.run_interval <= 0:
                self.logger.error('Invalid run interval')
                return False
            if self.max_history <= 0:
                self.logger.error('Invalid max history')
                return False
            self.logger.info('Configuration validation successful')
            return True
        except Exception as e:
            self.logger.exception(f'Error validating configuration: {e}')
            return False

    @handles_errors(error_handlers={Exception: (False, 'Supervisor run failed')}, default_return = False, context='main supervisor run')
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info('🚦 Main Supervisor started.')
            while self.is_running:
                await self._supervise()
                await asyncio.sleep(self.run_interval)
            return True
        except Exception as e:
            self.logger.exception(f'Error in main supervisor run: {e}')
            self.is_running = False
            return False

    @handles_errors(fallback = None)
    async def _supervise(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {'timestamp': now, 'status': 'running'}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            self.logger.info(f'Main Supervisor tick at {now}')
        except Exception as e:
            self.logger.exception(f'Error in supervise step: {e}')

    @handles_errors(fallback = None)
    async def stop(self) -> None:
        self.logger.info('🛑 Stopping Main Supervisor...')
        try:
            self.is_running = False
            self.status = {'timestamp': datetime.now().isoformat(), 'status': 'stopped'}
            self.logger.info('✅ Main Supervisor stopped successfully')
        except Exception as e:
            self.logger.exception(f'Error stopping main supervisor: {e}')

    def get_status(self) -> dict[str, Any]:
        return self.status.copy()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    @handle_errors_with_tracking(
        context="live trading utilities initialization",
        log_level="INFO",
        print_errors=True
    )
    async def _initialize_live_trading_utilities(self) -> bool:
        """Initialize live trading utilities."""
        try:
            self.logger.info("Initializing live trading utilities...")
            print("Initializing live trading utilities...")
            
            # Initialize Model Manager for model selection and loading
            self.model_manager = ModelManager()
            self.logger.info("✅ Model Manager initialized")
            print("✅ Model Manager initialized")
            
            # Set default model selections for each component
            self.selected_models = {
                "analyst": "analyst_regime_classifier",
                "strategist": "strategist_market_analysis_model",
                "tactician": "tactician_position_sizing_model"
            }
            self.logger.info("✅ Default model selections configured")
            print("✅ Default model selections configured")
            
            # Initialize caches
            self.model_cache = {}
            self.supervision_cache = {}
            self.logger.info("✅ Model and supervision caches initialized")
            print("✅ Model and supervision caches initialized")
            
            return True
        except Exception as e:
            error_msg = f"❌ Error initializing live trading utilities: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            return False

    @handles_errors(fallback = False)
    async def _initialize_performance_monitoring(self) -> bool:
        """Initialize performance monitoring."""
        try:
            self.logger.info("Initializing performance monitoring...")
            
            # Initialize Performance Monitor
            self.performance_monitor = PerformanceMonitor()
            self.logger.info("✅ Performance Monitor initialized")
            
            # Enable global monitoring
            self.global_monitor.enable()
            self.logger.info("✅ Global monitoring enabled")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing performance monitoring: {e}")
            return False

    @handle_errors_with_tracking(
        context="supervisor model management",
        log_level="INFO",
        print_errors=True
    )
    async def manage_component_models(self, component: str, model_name: str) -> bool:
        """
        Manage model selection for specific components in live trading.
        
        Args:
            component: Component name (analyst, strategist, tactician)
            model_name: Name of the pre-trained model to select
            
        Returns:
            bool: True if model selection successful
        """
        if not self.model_manager:
            error_msg = "Model Manager not available"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False
        
        try:
            self.logger.info(f"Managing model for component {component}: {model_name}")
            print(f"Managing model for component {component}: {model_name}")
            
            # Check if model is available
            available_models = await self.model_manager.list_available_models()
            if model_name not in available_models:
                error_msg = f"Model {model_name} not available for live trading"
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                return False
            
            # Update selected model for component
            self.selected_models[component] = model_name
            
            # Load and cache the model
            model = await self.model_manager.load_model(model_name)
            if model:
                self.model_cache[model_name] = model
                self.logger.info(f"✅ Model {model_name} selected and cached for {component}")
                print(f"✅ Model {model_name} selected and cached for {component}")
                return True
            else:
                error_msg = f"Failed to load model: {model_name}"
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                return False
            
        except Exception as e:
            error_msg = f"Error managing model for component {component}: {e}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False

    @handle_errors_with_tracking(
        context="supervisor model coordination",
        log_level="INFO",
        print_errors=True
    )
    async def coordinate_models(self, market_conditions: dict[str, Any]) -> dict[str, bool]:
        """
        Coordinate model selection across all components based on market conditions.
        
        Args:
            market_conditions: Current market conditions
            
        Returns:
            dict: Results of model coordination for each component
        """
        try:
            self.logger.info("Coordinating models across all components...")
            print("Coordinating models across all components...")
            
            results = {}
            
            # Determine appropriate models based on market conditions
            volatility = market_conditions.get("volatility", "normal")
            trend = market_conditions.get("trend", "sideways")
            
            # Model selection logic for each component
            if volatility == "high":
                analyst_model = "analyst_high_volatility_model"
                strategist_model = "strategist_high_volatility_model"
                tactician_model = "tactician_high_volatility_model"
            elif trend == "strong_trend":
                analyst_model = "analyst_trend_following_model"
                strategist_model = "strategist_trend_following_model"
                tactician_model = "tactician_trend_following_model"
            else:
                analyst_model = "analyst_regime_classifier"
                strategist_model = "strategist_market_analysis_model"
                tactician_model = "tactician_position_sizing_model"
            
            # Update models for each component
            results["analyst"] = await self.manage_component_models("analyst", analyst_model)
            results["strategist"] = await self.manage_component_models("strategist", strategist_model)
            results["tactician"] = await self.manage_component_models("tactician", tactician_model)
            
            success_count = sum(results.values())
            self.logger.info(f"✅ Model coordination completed: {success_count}/3 components updated")
            print(f"✅ Model coordination completed: {success_count}/3 components updated")
            
            return results
            
        except Exception as e:
            error_msg = f"Error coordinating models: {e}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            return {"error": error_msg}

main_supervisor: MainSupervisor | None = None

async def setup_main_supervisor(config: dict[str, Any] | None = None) -> MainSupervisor | None:
    try:
        global main_supervisor
        if config is None:
            config = {'main_supervisor': {'run_interval': 60, 'max_history': 100}}
        main_supervisor = MainSupervisor(config)
        success = await main_supervisor.initialize()
        if success:
            return main_supervisor
        return None
    except Exception as e:
        print(f'Error setting up main supervisor: {e}')
        return None