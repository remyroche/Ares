from src.utils.tprint import tprint

import argparse
import asyncio
from datetime import datetime
import os
from pathlib import Path
import signal
import sys
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.validation.regime_consensus_validator import RegimeConsensusValidator

from analyst.analyst import Analyst
from config.environment import get_environment_settings
from core.config_service import ConfigurationService
from core.decorators import handles_errors
from core.dependency_injection import DependencyContainer
from database.sqlite_manager import SQLiteManager
from exchanges.binance import BinanceExchange as RootExchangeFactory
from interfaces.base_interfaces import IAnalyst
from interfaces.base_interfaces import IEventBus
from interfaces.base_interfaces import IStateManager
from interfaces.base_interfaces import IStrategist
from interfaces.base_interfaces import ISupervisor
from interfaces.base_interfaces import ITactician
from interfaces.event_bus import EventBus
from monitoring.auto_monitoring_launcher import get_auto_monitoring
from monitoring.auto_monitoring_launcher import launch_auto_monitoring
from monitoring.enhanced_monitoring_orchestrator import EnhancedMonitoringOrchestrator
from monitoring.performance_dashboard import PerformanceDashboard
from monitoring.performance_dashboard import setup_performance_dashboard
from monitoring.performance_monitor import PerformanceMonitor
from monitoring.performance_monitor import setup_performance_monitor
import pandas as pd
from strategist import Strategist
from supervisor import Supervisor
from tactician import Tactician
# Note: dual_model_system has been refactored into training steps
# Using training steps components instead
try:
    from training.steps.model_training import GeneralModelTrainer, AnalystModelTrainer, TacticianModelTrainer
    TRAINING_STEPS_AVAILABLE = True
except ImportError:
    TRAINING_STEPS_AVAILABLE = False
    GeneralModelTrainer = None
    AnalystModelTrainer = None
    TacticianModelTrainer = None
from utils.dependency_manager import get_dependency_manager
from utils.dependency_manager import optional_package
from utils.logger import setup_logging
from utils.logger import system_logger
from utils.observability import init_observability
from src.utils.lookahead_bias_detector import get_global_detector
from utils.regime_transition_handler import RegimeTransitionHandler
from utils.regime_transition_handler import handle_regime_transition
from utils.regime_transition_handler import set_global_handler
from utils.service_discovery import discover_and_register_services
from utils.state_manager import StateManager
from utils.warning_symbols import critical
from utils.warning_symbols import error
from utils.warning_symbols import failed
from utils.warning_symbols import warning

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

class AresPipeline:
    """
    Enhanced main pipeline with dependency injection and comprehensive error handling.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize Ares pipeline with enhanced type safety and DI.

        Args:
            config: Optional configuration dictionary
        """
        self.logger = system_logger.getChild('AresPipeline')
        self.config = config or {}
        self.container: DependencyContainer = DependencyContainer(config or {})
        self.service_locator: DependencyContainer = self.container
        self.analyst: IAnalyst | None = None
        self.strategist: IStrategist | None = None
        self.tactician: ITactician | None = None
        self.supervisor: ISupervisor | None = None
        self.state_manager: IStateManager | None = None
        self.event_bus: IEventBus | None = None
        self.dual_model_system = None
        self.performance_monitor: PerformanceMonitor | None = None
        self.performance_dashboard: PerformanceDashboard | None = None
        self.enhanced_monitoring: EnhancedMonitoringOrchestrator | None = None
        self.auto_monitoring_launcher = None
        self.regime_transition_handler: RegimeTransitionHandler | None = None
        self.regime_consensus_validator: RegimeConsensusValidator | None = None
        self.is_running: bool = False
        self.start_time: datetime | None = None
        self.cycle_count: int = 0
        self.last_cycle_time: datetime | None = None

    @handles_errors(error_handlers={ValueError: (False, 'Invalid pipeline configuration'), AttributeError: (False, 'Missing required pipeline components'), KeyError: (False, 'Missing configuration keys')}, default_return = False, context='pipeline initialization')
    async def initialize(self) -> bool:
        """
        Initialize pipeline with enhanced error handling and DI.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info('Initializing Ares Pipeline...')
            await self._initialize_configuration_service()
            await self._register_core_services()
            await self._resolve_pipeline_components()
            await self._initialize_components()
            await self._initialize_dual_model_system()
            await self._initialize_performance_monitoring()
            await self._initialize_enhanced_monitoring()
            await self._launch_auto_monitoring()
            await self._initialize_regime_transition_handler()
            await self._initialize_regime_consensus_validator()
            await self._discover_and_register_services()
            await self._check_optional_dependencies()
            self._setup_signal_handlers()
            self.logger.info('✅ Ares Pipeline initialization completed successfully')
            return True
        except Exception:
            self.logger.exception('❌ Ares Pipeline initialization failed')
            return False

    @handles_errors(default_return = None, context='configuration service initialization')
    async def _initialize_configuration_service(self) -> None:
        """Initialize configuration service."""
        try:
            tprint('   ⚙️ Initializing ConfigurationService...')
            self.logger.info('   ⚙️ Initializing ConfigurationService...')

            def _config_service_factory(container: DependencyContainer) -> ConfigurationService:
                return ConfigurationService(container.get_config('root_config', {}))
            if self.container.get_config('root_config') is None:
                self.container.set_config('root_config', self.container._config)
            self.container.register_factory('ConfigurationService', _config_service_factory)
            tprint('   ✅ ConfigurationService initialized successfully')
            self.logger.info('   ✅ ConfigurationService initialized successfully')
        except Exception as e:
            tprint(f'   ❌ Error initializing configuration service: {e}')
            self.logger.exception('Error initializing configuration service')
            raise

    @handles_errors(default_return = None, context='core service registration')
    async def _register_core_services(self) -> None:
        """Register core services in DI container with comprehensive logging."""
        try:
            tprint('🔧 Registering core services...')
            self.logger.info('🔧 Registering core services...')
            tprint('   💾 Registering DatabaseManager...')
            self.logger.info('   💾 Registering DatabaseManager...')
            try:
                self.container.register('DatabaseManager', SQLiteManager)
                tprint('   ✅ DatabaseManager registered successfully')
                self.logger.info('   ✅ DatabaseManager registered successfully')
            except Exception as e:
                tprint(f'   ❌ Failed to register DatabaseManager: {e}')
                self.logger.exception('   ❌ Failed to register DatabaseManager')
            tprint('   🏢 Registering ExchangeClient...')
            self.logger.info('   🏢 Registering ExchangeClient...')
            try:
                env_settings = get_environment_settings()
                exchange_instance = RootExchangeFactory.get_exchange(env_settings.exchange_name.lower())
                self.container.register_instance('ExchangeClient', exchange_instance)
                tprint('   ✅ ExchangeClient registered successfully')
                self.logger.info('   ✅ ExchangeClient registered successfully')
            except Exception as e:
                tprint(f'   ❌ Failed to register ExchangeClient: {e}')
                self.logger.exception('   ❌ Failed to register ExchangeClient')
            tprint('   📊 Registering Analyst...')
            self.logger.info('   📊 Registering Analyst...')
            try:
                self.container.register('Analyst', Analyst, config={'analyst': {}})
                tprint('   ✅ Analyst registered successfully')
                self.logger.info('   ✅ Analyst registered successfully')
            except Exception as e:
                tprint(f'   ❌ Failed to register Analyst: {e}')
                self.logger.exception('   ❌ Failed to register Analyst')
            tprint('   🧠 Registering Strategist...')
            self.logger.info('   🧠 Registering Strategist...')
            try:
                self.container.register('Strategist', Strategist, config={'strategist': {}})
                tprint('   ✅ Strategist registered successfully')
                self.logger.info('   ✅ Strategist registered successfully')
            except Exception as e:
                tprint(f'   ❌ Failed to register Strategist: {e}')
                self.logger.exception('   ❌ Failed to register Strategist')
            tprint('   🎯 Registering Tactician...')
            self.logger.info('   🎯 Registering Tactician...')
            try:
                self.container.register('Tactician', Tactician, config={'tactician': {}})
                tprint('   ✅ Tactician registered successfully')
                self.logger.info('   ✅ Tactician registered successfully')
            except Exception as e:
                tprint(f'   ❌ Failed to register Tactician: {e}')
                self.logger.exception('   ❌ Failed to register Tactician')
            tprint('   👁️ Registering Supervisor...')
            self.logger.info('   👁️ Registering Supervisor...')
            try:
                self.container.register('Supervisor', Supervisor, config={'supervisor': {}})
                tprint('   ✅ Supervisor registered successfully')
                self.logger.info('   ✅ Supervisor registered successfully')
            except Exception as e:
                tprint(f'   ❌ Failed to register Supervisor: {e}')
                self.logger.exception('   ❌ Failed to register Supervisor')
            tprint('   💾 Registering StateManager...')
            self.logger.info('   💾 Registering StateManager...')
            try:
                self.container.register('StateManager', StateManager, config={'state_manager': {}})
                tprint('   ✅ StateManager registered successfully')
                self.logger.info('   ✅ StateManager registered successfully')
            except Exception as e:
                tprint(f'   ❌ Failed to register StateManager: {e}')
                self.logger.exception('   ❌ Failed to register StateManager')
            tprint('   📡 Registering EventBus...')
            self.logger.info('   📡 Registering EventBus...')
            try:
                self.container.register('EventBus', EventBus, config={'event_bus': {}})
                tprint('   ✅ EventBus registered successfully')
                self.logger.info('   ✅ EventBus registered successfully')
            except Exception as e:
                tprint(f'   ❌ Failed to register EventBus: {e}')
                self.logger.exception('   ❌ Failed to register EventBus')
            tprint('✅ Core services registered successfully')
            self.logger.info('✅ Core services registered successfully')
        except Exception:
            tprint(warning('Error registering core services'))
            self.logger.exception('Error registering core services')
            raise

    @handles_errors(default_return = None, context='pipeline component resolution')
    async def _resolve_pipeline_components(self) -> None:
        """Resolve pipeline components through DI container with comprehensive logging."""
        try:
            tprint('🔧 Resolving pipeline components...')
            self.logger.info('🔧 Resolving pipeline components...')
            tprint('   📊 Resolving Analyst component...')
            self.logger.info('   📊 Resolving Analyst component...')
            self.analyst = self.container.resolve('Analyst')
            if self.analyst:
                tprint('   ✅ Analyst component resolved successfully')
                self.logger.info('   ✅ Analyst component resolved successfully')
            else:
                tprint('   ❌ Failed to resolve Analyst component')
                self.logger.error('   ❌ Failed to resolve Analyst component')
            tprint('   🧠 Resolving Strategist component...')
            self.logger.info('   🧠 Resolving Strategist component...')
            self.strategist = self.container.resolve('Strategist')
            if self.strategist:
                tprint('   ✅ Strategist component resolved successfully')
                self.logger.info('   ✅ Strategist component resolved successfully')
            else:
                tprint('   ❌ Failed to resolve Strategist component')
                self.logger.error('   ❌ Failed to resolve Strategist component')
            tprint('   🎯 Resolving Tactician component...')
            self.logger.info('   🎯 Resolving Tactician component...')
            self.tactician = self.container.resolve('Tactician')
            if self.tactician:
                tprint('   ✅ Tactician component resolved successfully')
                self.logger.info('   ✅ Tactician component resolved successfully')
            else:
                tprint('   ❌ Failed to resolve Tactician component')
                self.logger.error('   ❌ Failed to resolve Tactician component')
            tprint('   👁️ Resolving Supervisor component...')
            self.logger.info('   👁️ Resolving Supervisor component...')
            self.supervisor = self.container.resolve('Supervisor')
            if self.supervisor:
                tprint('   ✅ Supervisor component resolved successfully')
                self.logger.info('   ✅ Supervisor component resolved successfully')
            else:
                tprint('   ❌ Failed to resolve Supervisor component')
                self.logger.error('   ❌ Failed to resolve Supervisor component')
            tprint('   💾 Resolving StateManager component...')
            self.logger.info('   💾 Resolving StateManager component...')
            self.state_manager = self.container.resolve('StateManager')
            if self.state_manager:
                tprint('   ✅ StateManager component resolved successfully')
                self.logger.info('   ✅ StateManager component resolved successfully')
            else:
                tprint('   ❌ Failed to resolve StateManager component')
                self.logger.error('   ❌ Failed to resolve StateManager component')
            tprint('   📡 Resolving EventBus component...')
            self.logger.info('   📡 Resolving EventBus component...')
            self.event_bus = self.container.resolve('EventBus')
            if self.event_bus:
                tprint('   ✅ EventBus component resolved successfully')
                self.logger.info('   ✅ EventBus component resolved successfully')
            else:
                tprint('   ❌ Failed to resolve EventBus component')
                self.logger.error('   ❌ Failed to resolve EventBus component')
            tprint('✅ Pipeline components resolved successfully')
            self.logger.info('✅ Pipeline components resolved successfully')
        except Exception:
            tprint(warning('Error resolving pipeline components'))
            self.logger.exception('Error resolving pipeline components')
            raise

    @handles_errors(default_return = None, context='component initialization')
    async def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            if self.state_manager:
                await self.state_manager.initialize()
            if self.event_bus:
                await self.event_bus.initialize()
            if self.analyst:
                await self.analyst.initialize()
            if self.strategist:
                await self.strategist.initialize()
            if self.tactician:
                await self.tactician.initialize()
            if self.supervisor:
                await self.supervisor.initialize()
            self.logger.info('All pipeline components initialized successfully')
        except Exception:
            self.logger.exception('Error initializing components')

    @handles_errors(default_return = None, context='signal handler setup')
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.info('Signal handlers configured')
        except Exception:
            self.logger.exception('Error setting up signal handlers')

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        """Handle shutdown signals."""
        self.logger.info(f'Received signal {signum}, initiating graceful shutdown...')
        asyncio.create_task(self.stop())

    @handles_errors(error_handlers={ConnectionError: (None, 'Failed to connect to exchange'), TimeoutError: (None, 'Pipeline operation timed out'), ValueError: (None, 'Invalid pipeline state')}, default_return = None, context='pipeline execution')
    async def run(self) -> dict[str, Any] | None:
        """
        Run the Ares pipeline with comprehensive logging and timeout protection.

        Returns:
            dict[str, Any] | None: Pipeline execution results or None if failed
        """
        try:
            tprint('🔄 Starting Ares Pipeline execution...')
            self.logger.info('🔄 Starting Ares Pipeline execution...')
            if self.is_running:
                tprint(warning('Pipeline already running'))
                self.logger.warning('Pipeline already running')
                return None
            tprint('🚀 Starting Ares Pipeline...')
            self.logger.info('🚀 Starting Ares Pipeline...')
            self.is_running = True
            self.start_time = datetime.now()
            tprint(f'📅 Pipeline start time: {self.start_time}')
            self.logger.info(f'📅 Pipeline start time: {self.start_time}')
            max_cycles = 10
            max_duration = 300
            while self.is_running:
                try:
                    current_time = datetime.now()
                    elapsed_time = (current_time - self.start_time).total_seconds()
                    if self.cycle_count >= max_cycles:
                        tprint(f'⏰ Reached maximum cycles ({max_cycles}), stopping pipeline')
                        self.logger.info(f'⏰ Reached maximum cycles ({max_cycles}), stopping pipeline')
                        break
                    if elapsed_time >= max_duration:
                        tprint(f'⏰ Reached maximum duration ({max_duration}s), stopping pipeline')
                        self.logger.info(f'⏰ Reached maximum duration ({max_duration}s), stopping pipeline')
                        break
                    tprint(f'🔄 Executing pipeline cycle {self.cycle_count + 1}... (Time: {elapsed_time:.1f}s)')
                    self.logger.info(f'🔄 Executing pipeline cycle {self.cycle_count + 1}... (Time: {elapsed_time:.1f}s)')
                    await self._execute_cycle()
                    self.cycle_count += 1
                    self.last_cycle_time = datetime.now()
                    tprint(f'✅ Cycle {self.cycle_count} completed successfully')
                    self.logger.info(f'✅ Cycle {self.cycle_count} completed successfully')
                    try:
                        config_service = self.container.resolve('ConfigurationService')
                        cycle_interval = config_service.get_value('pipeline.loop_interval_seconds', 10)
                        tprint(f'⏱️ Waiting {cycle_interval} seconds before next cycle...')
                        self.logger.info(f'⏱️ Waiting {cycle_interval} seconds before next cycle...')
                    except Exception as e:
                        tprint(warning('Error getting cycle interval, using default'))
                        self.logger.warning(f'Error getting cycle interval, using default: {e}')
                        cycle_interval = 10
                    await asyncio.sleep(cycle_interval)
                except asyncio.CancelledError:
                    tprint(error('Pipeline cancelled'))
                    self.logger.info('Pipeline cancelled')
                    break
                except Exception as e:
                    tprint(warning(f'Error in pipeline cycle: {e}'))
                    self.logger.exception('Error in pipeline cycle')
                    await asyncio.sleep(5)
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            tprint('✅ Pipeline completed successfully!')
            tprint(f'📊 Total cycles executed: {self.cycle_count}')
            tprint(f'⏱️ Total duration: {duration:.2f} seconds')
            self.logger.info('✅ Pipeline completed successfully!')
            self.logger.info(f'📊 Total cycles executed: {self.cycle_count}')
            self.logger.info(f'⏱️ Total duration: {duration:.2f} seconds')
            return {'status': 'completed', 'cycles_executed': self.cycle_count, 'start_time': self.start_time, 'end_time': end_time, 'duration_seconds': duration}
        except Exception as e:
            tprint(critical(f'Fatal error running pipeline: {e}'))
            self.logger.exception('Error running pipeline')
            return None
        finally:
            self.is_running = False
            tprint('🧹 Pipeline cleanup completed')
            self.logger.info('🧹 Pipeline cleanup completed')

    @handles_errors(default_return = None, context='pipeline cycle execution')
    async def _execute_cycle(self) -> None:
        """Execute a single pipeline cycle with comprehensive logging."""
        try:
            cycle_start = datetime.now()
            tprint(f'🔄 Starting pipeline cycle {self.cycle_count + 1}')
            self.logger.info(f'🔄 Starting pipeline cycle {self.cycle_count + 1}')
            tprint('📊 Step 1: Market Analysis')
            self.logger.info('📊 Step 1: Market Analysis')
            if self.analyst:
                tprint('   🔍 Executing market analysis...')
                self.logger.info('   🔍 Executing market analysis...')

                # Set current timestamp for lookahead bias detection
                current_time = datetime.now()
                bias_detector = get_global_detector()
                bias_detector.set_current_timestamp(current_time)

                analysis_input = {'symbol': 'ETHUSDT', 'timeframe': '1h', 'limit': 100, 'analysis_type': 'technical', 'include_indicators': True, 'include_patterns': True}
                analysis_result = await self.analyst.execute_analysis(analysis_input)

                # Handle regime transitions with position protection
                if analysis_result and self.regime_transition_handler:
                    regime_info = analysis_result.get('regime_analysis', {})
                    current_regime = regime_info.get('current_regime', 'unknown')
                    regime_confidence = regime_info.get('confidence', 0.5)
                    market_volatility = regime_info.get('volatility', 0.02)

                    transition_result = handle_regime_transition(
                        current_regime=current_regime,
                        regime_confidence=regime_confidence,
                        market_volatility=market_volatility
                    )

                    if transition_result['transition_detected']:
                        self.logger.info(f"🔄 Regime transition detected: {transition_result['transition_type']}")
                        if transition_result['should_exit']:
                            self.logger.warning("⚠️ Emergency position exit recommended due to regime transition")
                if analysis_result:
                    tprint('   ✅ Market analysis completed successfully')
                    self.logger.info('   ✅ Market analysis completed successfully')
                else:
                    tprint('   ⚠️ Market analysis had issues')
                    self.logger.warning('   ⚠️ Market analysis had issues')
            else:
                tprint('   ❌ Analyst component not available')
                self.logger.error('   ❌ Analyst component not available')
            tprint('🧠 Step 2: Strategy Development')
            self.logger.info('🧠 Step 2: Strategy Development')
            if self.strategist:
                tprint('   🎯 Developing trading strategy...')
                self.logger.info('   🎯 Developing trading strategy...')
                strategy_market_data = pd.DataFrame({'open': [100.0] * 100, 'high': [101.0] * 100, 'low': [99.0] * 100, 'close': [100.5] * 100, 'volume': [1000.0] * 100})
                strategy_current_price = 100.5
                strategy_result = await self.strategist.generate_strategy(market_data = strategy_market_data, current_price = strategy_current_price)
                if strategy_result:
                    tprint('   ✅ Strategy development completed successfully')
                    self.logger.info('   ✅ Strategy development completed successfully')
                else:
                    tprint('   ⚠️ Strategy development had issues')
                    self.logger.warning('   ⚠️ Strategy development had issues')
            else:
                tprint('   ❌ Strategist component not available')
                self.logger.error('   ❌ Strategist component not available')
            tprint('🎯 Step 3: Tactical Execution')
            self.logger.info('🎯 Step 3: Tactical Execution')
            if self.tactician:
                tprint('   ⚡ Executing tactical decisions...')
                self.logger.info('   ⚡ Executing tactical decisions...')
                tactical_result = await self.tactician.run()
                if tactical_result:
                    tprint('   ✅ Tactical execution completed successfully')
                    self.logger.info('   ✅ Tactical execution completed successfully')
                else:
                    tprint('   ⚠️ Tactical execution had issues')
                    self.logger.warning('   ⚠️ Tactical execution had issues')
            else:
                tprint('   ❌ Tactician component not available')
                self.logger.error('   ❌ Tactician component not available')
            tprint('👁️ Step 5: Supervision and Monitoring')
            self.logger.info('👁️ Step 5: Supervision and Monitoring')
            if self.supervisor:
                tprint('   📊 Monitoring system performance...')
                self.logger.info('   📊 Monitoring system performance...')
                supervision_result = True
                if supervision_result:
                    tprint('   ✅ Supervision completed successfully')
                    self.logger.info('   ✅ Supervision completed successfully')
                else:
                    tprint('   ⚠️ Supervision had issues')
                    self.logger.warning('   ⚠️ Supervision had issues')
            else:
                tprint('   ❌ Supervisor component not available')
                self.logger.error('   ❌ Supervisor component not available')
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            tprint(f'✅ Pipeline cycle completed in {cycle_duration:.2f}s')
            self.logger.info(f'✅ Pipeline cycle completed in {cycle_duration:.2f}s')
        except Exception as e:
            tprint(warning(f'Error executing pipeline cycle: {e}'))
            self.logger.exception('Error executing pipeline cycle')
            raise

    def get_pipeline_status(self) -> dict[str, Any]:
        """
        Get current pipeline status.

        Returns:
            Dict[str, Any]: Pipeline status information
        """
        status = {'is_running': self.is_running, 'start_time': self.start_time, 'cycle_count': self.cycle_count, 'last_cycle_time': self.last_cycle_time, 'components': {'analyst': self.analyst is not None, 'strategist': self.strategist is not None, 'tactician': self.tactician is not None, 'supervisor': self.supervisor is not None, 'state_manager': self.state_manager is not None, 'event_bus': self.event_bus is not None, 'enhanced_monitoring': self.enhanced_monitoring is not None, 'auto_monitoring_launcher': self.auto_monitoring_launcher is not None}}
        if self.performance_monitor:
            try:
                performance_status = self.performance_monitor.get_performance_summary()
                status['performance_monitoring_status'] = performance_status
            except Exception as e:
                status['performance_monitoring_status'] = {'error': str(e)}
        if self.performance_dashboard:
            try:
                dashboard_status = self.performance_dashboard.get_dashboard_summary()
                status['performance_dashboard_status'] = dashboard_status
            except Exception as e:
                status['performance_dashboard_status'] = {'error': str(e)}
        if self.enhanced_monitoring:
            try:
                monitoring_status = self.enhanced_monitoring.get_system_status()
                status['enhanced_monitoring_status'] = monitoring_status
            except Exception as e:
                status['enhanced_monitoring_status'] = {'error': str(e)}
        if self.auto_monitoring_launcher:
            try:
                auto_monitoring_status = self.auto_monitoring_launcher.get_system_status()
                status['auto_monitoring_launcher_status'] = auto_monitoring_status
            except Exception as e:
                status['auto_monitoring_launcher_status'] = {'error': str(e)}
        return status

    @handles_errors(default_return = None, context='pipeline cleanup')
    async def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self.logger.info('🛑 Stopping Ares Pipeline...')
        try:
            self.is_running = False
            if self.auto_monitoring_launcher:
                await self.auto_monitoring_launcher.stop()
            if self.enhanced_monitoring:
                await self.enhanced_monitoring.stop()
            if self.performance_dashboard:
                await self.performance_dashboard.stop()
            if self.performance_monitor:
                await self.performance_monitor.stop()
            if self.supervisor:
                await self.supervisor.stop()
            if self.tactician:
                await self.tactician.stop()
            if self.strategist:
                await self.strategist.stop()
            if self.analyst:
                await self.analyst.stop()
            if self.event_bus:
                await self.event_bus.stop()
            if self.state_manager:
                await self.state_manager.stop()
            db_manager = self.container.resolve('DatabaseManager')
            if db_manager:
                await db_manager.close()
            self.logger.info('✅ Ares Pipeline stopped successfully')
        except Exception:
            self.logger.exception('Error stopping pipeline')

    async def _initialize_dual_model_system(self) -> None:
        """Initialize training steps system."""
        try:
            if TRAINING_STEPS_AVAILABLE and GeneralModelTrainer is not None:
                # Initialize the new training steps components
                self.dual_model_system = GeneralModelTrainer(self.config)
                self.logger.info('✅ Training Steps System initialized successfully')
                self.logger.info("   📊 Using new training steps architecture")
                self.logger.info("   📊 General Model Trainer available")
                self.logger.info("   📊 Analyst Model Trainer available")
                self.logger.info("   📊 Tactician Model Trainer available")
            else:
                self.logger.warning('Dual Model System not available - training steps not loaded')
        except Exception:
            self.logger.exception('Error initializing dual model system')

    async def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring."""
        try:
            self.logger.info('📊 Initializing Performance Monitoring...')
            self.performance_monitor = await setup_performance_monitor(self.config)
            if self.performance_monitor:
                self.logger.info('✅ Performance Monitor initialized successfully')
                self.performance_dashboard = await setup_performance_dashboard(self.config, self.performance_monitor)
                if self.performance_dashboard:
                    self.logger.info('✅ Performance Dashboard initialized successfully')
                else:
                    self.logger.warning('⚠️ Failed to initialize Performance Dashboard')
            else:
                self.logger.warning('⚠️ Failed to initialize Performance Monitor')
        except Exception:
            self.logger.exception('Error initializing performance monitoring')

    async def _initialize_enhanced_monitoring(self) -> None:
        """Initialize enhanced monitoring system."""
        try:
            self.logger.info('🔍 Initializing Enhanced Monitoring System...')

            # Get trading mode from environment
            trading_mode = os.environ.get('TRADING_MODE', 'PAPER').upper()

            # Initialize enhanced monitoring orchestrator
            self.enhanced_monitoring = EnhancedMonitoringOrchestrator()
            await self.enhanced_monitoring.initialize()

            if self.enhanced_monitoring:
                self.logger.info('✅ Enhanced Monitoring System initialized successfully')
                self.logger.info(f'   📊 Trading Mode: {trading_mode}')
                self.logger.info('   🔍 SHAP/LIME explanations enabled')
                self.logger.info('   📈 Ensemble and ML model tracking active')
                self.logger.info('   📋 Daily and monthly CSV exports configured')
            else:
                self.logger.warning('⚠️ Failed to initialize Enhanced Monitoring System')
        except Exception:
            self.logger.exception('Error initializing enhanced monitoring system')

    async def _launch_auto_monitoring(self) -> None:
        """Launch auto monitoring system."""
        try:
            self.logger.info('🚀 Launching Auto Enhanced Monitoring System...')

            # Launch the auto monitoring system
            success = await launch_auto_monitoring()

            if success:
                self.auto_monitoring_launcher = await get_auto_monitoring()
                self.logger.info('✅ Auto Enhanced Monitoring System launched successfully')
                self.logger.info('   🎯 Automatic trade decision capture enabled')
                self.logger.info('   📊 Performance tracking active')
                self.logger.info('   🔍 SHAP/LIME explanations ready')
            else:
                self.logger.warning('⚠️ Failed to launch Auto Enhanced Monitoring System')
        except Exception:
            self.logger.exception('Error launching auto monitoring system')

    async def _initialize_regime_transition_handler(self) -> None:
        """Initialize regime transition handler with position protection."""
        try:
            self.logger.info('🔄 Initializing regime transition handler...')
            regime_config = self.config.get('regime_transition_handler', {})
            self.regime_transition_handler = RegimeTransitionHandler(regime_config)
            set_global_handler(self.regime_transition_handler)
            self.logger.info('✅ Regime transition handler initialized successfully')
        except Exception as e:
            self.logger.exception(f'Error initializing regime transition handler: {e}')
            raise

    @handles_errors(default_return=None, context='regime consensus validator initialization')
    async def _initialize_regime_consensus_validator(self) -> None:
        """Initialize regime consensus validator for semantic consensus validation."""
        try:
            self.logger.info('🧠 Initializing regime consensus validator...')

            # Import the regime consensus validator
            from src.validation.regime_consensus_validator import RegimeConsensusValidator

            # Initialize with configuration
            validator_config = self.config.get('regime_consensus_validator', {
                'enable_semantic_consensus': True,
                'consensus_threshold': 0.6,
                'disagreement_tolerance': 0.3,
                'enable_regime_mapping': True,
                'enable_feature_based_mapping': True
            })

            self.regime_consensus_validator = RegimeConsensusValidator(validator_config)
            self.logger.info('✅ Regime consensus validator initialized successfully')

        except ImportError:
            self.logger.warning('⚠️ Regime consensus validator not available - semantic consensus validation disabled')
            self.regime_consensus_validator = None
        except Exception as e:
            self.logger.exception(f'Error initializing regime consensus validator: {e}')
            self.regime_consensus_validator = None

    async def _discover_and_register_services(self) -> None:
        """Discover and register services automatically."""
        try:
            self.logger.info('🔍 Discovering and registering services automatically...')
            discover_and_register_services(self.container, "src")
            self.logger.info('✅ Service discovery and registration completed')
        except Exception as e:
            self.logger.exception(f'Error in service discovery: {e}')
            raise

    @optional_package('numpy', 'pandas', 'scipy', 'sklearn')
    async def _check_optional_dependencies(self) -> None:
        """Check and report on optional dependencies."""
        try:
            self.logger.info('📦 Checking optional dependencies...')
            dep_manager = get_dependency_manager()
            available_packages = dep_manager.get_available_packages()

            self.logger.info(f'Available packages: {len(available_packages)}')
            for package in sorted(available_packages):
                self.logger.debug(f'  ✅ {package}')

            # Check for critical missing packages
            critical_packages = ['numpy', 'pandas']
            missing_critical = dep_manager.get_missing_packages(critical_packages)
            if missing_critical:
                self.logger.warning(f'⚠️ Missing critical packages: {missing_critical}')
            else:
                self.logger.info('✅ All critical packages available')

        except Exception as e:
            self.logger.exception(f'Error checking dependencies: {e}')
            raise

async def main() -> None:
    """Main entry point for the Ares Pipeline."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    setup_logging()
    init_observability({})
    logger = system_logger.getChild('AresPipelineMain')
    parser = argparse.ArgumentParser(description='Ares Trading Pipeline')
    parser.add_argument('symbol', help='Trading symbol (e.g., ETHUSDT)')
    parser.add_argument('exchange', help='Exchange name (e.g., BINANCE)')
    parser.add_argument('--config', help='Path to configuration file')
    args = parser.parse_args()
    trading_mode = os.environ.get('TRADING_MODE', 'PAPER').upper()
    logger.info(f'🚀 Starting Ares Pipeline in {trading_mode} mode')
    logger.info(f'📊 Symbol: {args.symbol}')
    logger.info(f'🏢 Exchange: {args.exchange}')
    logger.info(f'🔧 Trading Mode: {trading_mode}')
    pipeline = AresPipeline()
    try:
        if not await pipeline.initialize():
            tprint(failed('❌ Failed to initialize pipeline'))
            sys.exit(1)
        result = await pipeline.run()
        if result:
            logger.info('✅ Pipeline completed successfully')
        else:
            tprint(failed('❌ Pipeline failed'))
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info('🛑 Received interrupt signal, shutting down gracefully...')
        await pipeline.stop()
    except Exception as e:
        tprint(error(f'💥 Unexpected error: {e}'))
        await pipeline.stop()
        sys.exit(1)
if __name__ == '__main__':
    asyncio.run( main())
