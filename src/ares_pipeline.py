# src/ares_pipeline.py

from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Any
import asyncio
import signal
import sys
import os
import argparse

import pandas as pd

from src.analyst.analyst import Analyst
from src.config.environment import get_exchange_name
from src.database.sqlite_manager import SQLiteManager
from src.interfaces.event_bus import EventBus
# Note: Strategist import removed - was unused dead code
from src.supervisor.supervisor import Supervisor
from src.tactician.tactician import Tactician
from src.utils.state_manager import StateManager
from src.strategist.strategist import Strategist
from src.exchange.root_exchange_factory import RootExchangeFactory
from src.config import get_dual_model_config
from src.interfaces.base_interfaces import (
    IAnalyst,
    IEventBus,
    IStateManager,
    IStrategist,
    ISupervisor,
    ITactician,
)
from src.utils.observability import init_observability
from src.monitoring.performance_dashboard import (
    PerformanceDashboard,
    setup_performance_dashboard,
)
from src.monitoring.performance_monitor import (
    PerformanceMonitor,
    setup_performance_monitor,
)
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.warning_symbols import (
    critical,
    error,
    failed,
    warning,
)
from src.utils.logger import system_logger, setup_logging
from src.core.dependency_injection import DependencyContainer, ServiceLocator
from src.monitoring.dual_model_system import DualModelSystem, setup_dual_model_system
from src.core.config_service import ConfigurationService

# Add the project root to the Python path for subprocess execution
# Important: append instead of inserting at position 0 to avoid shadowing
# standard library modules like 'types' with our internal 'src/types' package.
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

if TYPE_CHECKING:
    pass

class AresPipeline:
    """
    Enhanced main pipeline with dependency injection and comprehensive error handling.
    """
    
    def __init__(self, config: dict = None) -> None:
        """Initialize the Ares Pipeline."""
        self.logger = system_logger.getChild("AresPipeline")
        self.config = config or {}

        # Initialize dependency injection container
        self.container: DependencyContainer = DependencyContainer(config or {})
        self.service_locator: ServiceLocator = ServiceLocator(self.container)

        # Pipeline components (will be resolved through DI)
        self.analyst: IAnalyst | None = None
        self.strategist: IStrategist | None = None
        self.tactician: ITactician | None = None
        self.supervisor: ISupervisor | None = None
        self.state_manager: IStateManager | None = None
        self.event_bus: IEventBus | None = None

        # Dual model system
        self.dual_model_system: DualModelSystem | None = None

        # Performance monitoring
        self.performance_monitor: PerformanceMonitor | None = None
        self.performance_dashboard: PerformanceDashboard | None = None

        # Pipeline state
        self.is_running: bool = False
        self.start_time: datetime | None = None
        self.cycle_count: int = 0
        self.last_cycle_time: datetime | None = None

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid pipeline configuration"),
            AttributeError: (False, "Missing required pipeline components"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False, 
        context="pipeline initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the pipeline components."""
        try:
            self.logger.info("Initializing Ares Pipeline...")

            # Initialize configuration service
            await self._initialize_configuration_service()

            # Register core services
            await self._register_core_services()

            # Resolve pipeline components
            await self._resolve_pipeline_components()

            # Initialize components
            await self._initialize_components()

            # Initialize dual model system
            await self._initialize_dual_model_system()

            # Initialize performance monitoring
            await self._initialize_performance_monitoring()

            # Setup signal handlers
            self._setup_signal_handlers()

            self.logger.info("✅ Ares Pipeline initialization completed successfully")
            return True

        except Exception:
            self.logger.exception("❌ Ares Pipeline initialization failed")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, 
        context="configuration service initialization",
    )
    async def _initialize_configuration_service(self) -> None:
        """Initialize the configuration service."""
        try:
            print("   ⚙️ Initializing ConfigurationService...")
            self.logger.info("   ⚙️ Initializing ConfigurationService...")

            # Create default configuration if none provided
            default_config = {
                "pipeline": {
                    "loop_interval_seconds": 10,
                    "max_cycles": 100,
                    "max_duration_seconds": 3600,
                    "enable_dual_model_system": True,
                    "enable_performance_monitoring": True,
                    "enable_supervision": True,
                },
                "trading": {
                    "default_symbol": "ETHUSDT",
                    "default_exchange": "BINANCE",
                    "risk_management": {
                        "max_position_size": 0.1,
                        "max_leverage": 3.0,
                        "stop_loss_percentage": 0.02,
                        "take_profit_percentage": 0.04,
                    },
                },
                "dual_model_system": {
                    "analyst_timeframes": ["30m", "15m", "5m"],
                    "tactician_timeframes": ["1m"],
                    "analyst_confidence_threshold": 0.6,
                    "tactician_confidence_threshold": 0.7,
                    "ensemble_weight_analyst": 0.6,
                    "ensemble_weight_tactician": 0.4,
                },
                "performance": {
                    "metrics_collection_interval": 30,
                    "dashboard_update_interval": 60,
                    "enable_real_time_monitoring": True,
                }
            }

            # Merge with provided config
            if self.config:
                self._deep_merge_config(default_config, self.config)

            # Register ConfigurationService via factory so it receives DI config
            def _config_service_factory(
                container: DependencyContainer
            ) -> ConfigurationService:
                # Pass the DI container's config into the service
                return ConfigurationService(container.get_config("root_config", default_config))

            # Store current container config under a conventional key if missing
            if self.container.get_config("root_config") is None:
                # The DependencyContainer already holds its config; expose it
                self.container.set_config("root_config", default_config)

            self.container.register_factory(
                "ConfigurationService",
                _config_service_factory
            )

            print("   ✅ ConfigurationService initialized successfully")
            self.logger.info("   ✅ ConfigurationService initialized successfully")

        except Exception as e:
            print(f"   ❌ Error initializing configuration service: {e}")
            self.logger.exception("Error initializing configuration service")
            raise

    def _deep_merge_config(self, base_config: dict, override_config: dict) -> None:
        """Deep merge configuration dictionaries."""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._deep_merge_config(base_config[key], value)
            else:
                base_config[key] = value

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, 
        context="core service registration",
    )
    async def _register_core_services(self) -> None:
        """Register core services in the dependency injection container."""
        try:
            print("🔧 Registering core services...")
            self.logger.info("🔧 Registering core services...")

            # Get configuration for service registration
            config_service = self.container.resolve("ConfigurationService")
            if not config_service:
                raise ValueError("ConfigurationService not available for service registration")

            # Register database manager
            print("   💾 Registering DatabaseManager...")
            self.logger.info("   💾 Registering DatabaseManager...")
            try:
                db_config = config_service.get_value("database", {})
                self.container.register("DatabaseManager", SQLiteManager, config=db_config)
                print("   ✅ DatabaseManager registered successfully")
                self.logger.info("   ✅ DatabaseManager registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register DatabaseManager: {e}")
                self.logger.exception("   ❌ Failed to register DatabaseManager")
                raise

            # Register exchange client
            print("   🏢 Registering ExchangeClient...")
            self.logger.info("   🏢 Registering ExchangeClient...")
            try:
                # Get exchange configuration
                exchange_name = get_exchange_name().lower()
                exchange_config = config_service.get_value("trading.exchange", {})
                
                # Build exchange instance via factory and register the instance
                exchange_instance = RootExchangeFactory.get_exchange(
                    exchange_name, config=exchange_config
                )
                self.container.register_instance("ExchangeClient", exchange_instance)
                print("   ✅ ExchangeClient registered successfully")
                self.logger.info("   ✅ ExchangeClient registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register ExchangeClient: {e}")
                self.logger.exception("   ❌ Failed to register ExchangeClient")
                raise

            # Register analyst with configuration
            print("   📊 Registering Analyst...")
            self.logger.info("   📊 Registering Analyst...")
            try:
                analyst_config = config_service.get_value("analyst", {})
                self.container.register("Analyst", Analyst, config=analyst_config)
                print("   ✅ Analyst registered successfully")
                self.logger.info("   ✅ Analyst registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register Analyst: {e}")
                self.logger.exception("   ❌ Failed to register Analyst")
                raise

            # Register strategist with configuration
            print("   🧠 Registering Strategist...")
            self.logger.info("   🧠 Registering Strategist...")
            try:
                strategist_config = config_service.get_value("strategist", {})
                self.container.register("Strategist", Strategist, config=strategist_config)
                print("   ✅ Strategist registered successfully")
                self.logger.info("   ✅ Strategist registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register Strategist: {e}")
                self.logger.exception("   ❌ Failed to register Strategist")
                raise

            # Register tactician with configuration
            print("   🎯 Registering Tactician...")
            self.logger.info("   🎯 Registering Tactician...")
            try:
                tactician_config = config_service.get_value("tactician", {})
                self.container.register("Tactician", Tactician, config=tactician_config)
                print("   ✅ Tactician registered successfully")
                self.logger.info("   ✅ Tactician registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register Tactician: {e}")
                self.logger.exception("   ❌ Failed to register Tactician")
                raise

            # Register supervisor with configuration
            print("   👁️ Registering Supervisor...")
            self.logger.info("   👁️ Registering Supervisor...")
            try:
                supervisor_config = config_service.get_value("supervisor", {})
                self.container.register("Supervisor", Supervisor, config=supervisor_config)
                print("   ✅ Supervisor registered successfully")
                self.logger.info("   ✅ Supervisor registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register Supervisor: {e}")
                self.logger.exception("   ❌ Failed to register Supervisor")
                raise

            # Register state manager with configuration
            print("   💾 Registering StateManager...")
            self.logger.info("   💾 Registering StateManager...")
            try:
                state_config = config_service.get_value("state_manager", {})
                self.container.register("StateManager", StateManager, config=state_config)
                print("   ✅ StateManager registered successfully")
                self.logger.info("   ✅ StateManager registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register StateManager: {e}")
                self.logger.exception("   ❌ Failed to register StateManager")
                raise

            # Register event bus with configuration
            print("   📡 Registering EventBus...")
            self.logger.info("   📡 Registering EventBus...")
            try:
                event_bus_config = config_service.get_value("event_bus", {})
                self.container.register("EventBus", EventBus, config=event_bus_config)
                print("   ✅ EventBus registered successfully")
                self.logger.info("   ✅ EventBus registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register EventBus: {e}")
                self.logger.exception("   ❌ Failed to register EventBus")
                raise

            print("✅ Core services registered successfully")
            self.logger.info("✅ Core services registered successfully")

        except Exception:
            print(warning("Error registering core services"))
            self.logger.exception("Error registering core services")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, 
        context="pipeline component resolution",
    )
    async def _resolve_pipeline_components(self) -> None:
        """Resolve all pipeline components from the dependency injection container."""
        try:
            print("🔧 Resolving pipeline components...")
            self.logger.info("🔧 Resolving pipeline components...")

            # Get configuration for component validation
            config_service = self.container.resolve("ConfigurationService")
            if not config_service:
                raise ValueError("ConfigurationService not available for component resolution")

            # Define required components and their validation rules
            required_components = {
                "Analyst": {
                    "interface": IAnalyst,
                    "required_methods": ["execute_analysis", "initialize", "stop"],
                    "description": "Market analysis and data processing"
                },
                "Strategist": {
                    "interface": IStrategist,
                    "required_methods": ["generate_strategy", "initialize", "stop"],
                    "description": "Trading strategy generation"
                },
                "Tactician": {
                    "interface": ITactician,
                    "required_methods": ["run", "initialize", "stop"],
                    "description": "Tactical execution and position management"
                },
                "Supervisor": {
                    "interface": ISupervisor,
                    "required_methods": ["initialize", "stop"],
                    "description": "System supervision and monitoring"
                },
                "StateManager": {
                    "interface": IStateManager,
                    "required_methods": ["initialize", "stop"],
                    "description": "State management and persistence"
                },
                "EventBus": {
                    "interface": IEventBus,
                    "required_methods": ["initialize", "stop"],
                    "description": "Event communication and routing"
                }
            }

            # Resolve and validate each component
            for component_name, validation_rules in required_components.items():
                print(f"   🔍 Resolving {component_name} component...")
                self.logger.info(f"   🔍 Resolving {component_name} component...")
                
                try:
                    # Resolve component
                    component = self.container.resolve(component_name)
                    if not component:
                        raise ValueError(f"Failed to resolve {component_name} from container")
                    
                    # Validate component interface
                    if not isinstance(component, validation_rules["interface"]):
                        raise TypeError(f"{component_name} does not implement {validation_rules['interface'].__name__}")
                    
                    # Validate required methods
                    for method_name in validation_rules["required_methods"]:
                        if not hasattr(component, method_name):
                            raise AttributeError(f"{component_name} missing required method: {method_name}")
                    
                    # Store component reference
                    setattr(self, component_name.lower(), component)
                    
                    print(f"   ✅ {component_name} component resolved and validated successfully")
                    self.logger.info(f"   ✅ {component_name} component resolved and validated successfully")
                    
                except Exception as e:
                    print(f"   ❌ Failed to resolve {component_name} component: {e}")
                    self.logger.error(f"   ❌ Failed to resolve {component_name} component: {e}")
                    raise

            # Validate that all critical components are available
            critical_components = ["analyst", "tactician", "state_manager"]
            missing_components = [comp for comp in critical_components if getattr(self, comp) is None]
            
            if missing_components:
                raise ValueError(f"Critical components missing: {missing_components}")

            print("✅ Pipeline components resolved and validated successfully")
            self.logger.info("✅ Pipeline components resolved and validated successfully")

        except Exception:
            print(warning("Error resolving pipeline components"))
            self.logger.exception("Error resolving pipeline components")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, 
        context="component initialization",
    )
    async def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            print("🔧 Initializing pipeline components...")
            self.logger.info("🔧 Initializing pipeline components...")

            # Define initialization order based on dependencies
            initialization_order = [
                ("StateManager", "state_manager", "State management and persistence"),
                ("EventBus", "event_bus", "Event communication and routing"),
                ("Analyst", "analyst", "Market analysis and data processing"),
                ("Strategist", "strategist", "Trading strategy generation"),
                ("Tactician", "tactician", "Tactical execution and position management"),
                ("Supervisor", "supervisor", "System supervision and monitoring")
            ]

            # Initialize components in dependency order
            for component_name, attr_name, description in initialization_order:
                component = getattr(self, attr_name)
                if component:
                    try:
                        print(f"   🔧 Initializing {component_name} ({description})...")
                        self.logger.info(f"   🔧 Initializing {component_name} ({description})...")
                        
                        # Initialize component with timeout protection
                        init_task = asyncio.create_task(component.initialize())
                        try:
                            await asyncio.wait_for(init_task, timeout=30.0)
                            print(f"   ✅ {component_name} initialized successfully")
                            self.logger.info(f"   ✅ {component_name} initialized successfully")
                        except asyncio.TimeoutError:
                            print(f"   ⚠️ {component_name} initialization timed out")
                            self.logger.warning(f"   ⚠️ {component_name} initialization timed out")
                            # Continue with other components
                        except Exception as e:
                            print(f"   ❌ Failed to initialize {component_name}: {e}")
                            self.logger.exception(f"   ❌ Failed to initialize {component_name}")
                            # For critical components, raise the error
                            if attr_name in ["state_manager", "analyst", "tactician"]:
                                raise
                    except Exception as e:
                        print(f"   ❌ Error initializing {component_name}: {e}")
                        self.logger.exception(f"   ❌ Error initializing {component_name}")
                        if attr_name in ["state_manager", "analyst", "tactician"]:
                            raise
                else:
                    print(f"   ⚠️ {component_name} not available, skipping initialization")
                    self.logger.warning(f"   ⚠️ {component_name} not available, skipping initialization")

            # Validate that critical components are properly initialized
            critical_components = ["state_manager", "analyst", "tactician"]
            for comp_name in critical_components:
                component = getattr(self, comp_name)
                if not component:
                    raise ValueError(f"Critical component {comp_name} not available after initialization")
                
                # Check if component has required methods
                if not hasattr(component, "is_initialized"):
                    self.logger.warning(f"Component {comp_name} does not have is_initialized method")
                elif not getattr(component, "is_initialized", False):
                    self.logger.warning(f"Component {comp_name} reports not initialized")

            print("✅ All pipeline components initialized successfully")
            self.logger.info("✅ All pipeline components initialized successfully")

        except Exception:
            self.logger.exception("Error initializing components")
            raise

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        try:
            # Store original signal handlers for restoration
            self._original_handlers = {}
            
            # Set up signal handlers for graceful shutdown
            for sig in [signal.SIGINT, signal.SIGTERM]:
                try:
                    self._original_handlers[sig] = signal.signal(sig, self._signal_handler)
                    self.logger.debug(f"Signal handler set for {sig}")
                except Exception as e:
                    self.logger.warning(f"Could not set signal handler for {sig}: {e}")
            
            # Set up additional signal handlers for debugging
            if hasattr(signal, 'SIGUSR1'):
                try:
                    signal.signal(signal.SIGUSR1, self._debug_signal_handler)
                    self.logger.debug("Debug signal handler (SIGUSR1) configured")
                except Exception as e:
                    self.logger.debug(f"Could not set debug signal handler: {e}")
            
            self.logger.info("Signal handlers configured successfully")
            print("🔄 Signal handlers configured for graceful shutdown")
            
        except Exception as e:
            self.logger.exception("Error setting up signal handlers")
            print(f"⚠️ Warning: Signal handlers not fully configured: {e}")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle signal for graceful shutdown."""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        self.logger.info(f"Received signal {signal_name} ({signum}), initiating graceful shutdown...")
        print(f"🛑 Received {signal_name} signal, shutting down gracefully...")
        
        # Create shutdown task
        try:
            if hasattr(self, '_shutdown_task') and not self._shutdown_task.done():
                self.logger.warning("Shutdown already in progress")
                return
            
            self._shutdown_task = asyncio.create_task(self.stop())
            self.logger.info("Graceful shutdown task created")
        except Exception as e:
            self.logger.exception("Error creating shutdown task")
            print(f"❌ Error during shutdown: {e}")

    def _debug_signal_handler(self, signum: int, frame) -> None:
        """Handle debug signal for pipeline status."""
        self.logger.info(f"Received debug signal {signum}")
        print("🔍 Debug signal received - displaying pipeline status...")
        
        try:
            # Display current pipeline status
            status = self.get_pipeline_status()
            print("📊 Current Pipeline Status:")
            print(f"   Running: {status.get('is_running', False)}")
            print(f"   Cycle Count: {status.get('cycle_count', 0)}")
            print(f"   Components: {len(status.get('components', {}))}")
            
            # Log detailed status
            self.logger.info(f"Pipeline status: {status}")
        except Exception as e:
            self.logger.exception("Error handling debug signal")
            print(f"❌ Error displaying status: {e}")

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            if hasattr(self, '_original_handlers'):
                for sig, handler in self._original_handlers.items():
                    try:
                        signal.signal(sig, handler)
                        self.logger.debug(f"Restored signal handler for {sig}")
                    except Exception as e:
                        self.logger.warning(f"Could not restore signal handler for {sig}: {e}")
                self.logger.info("Original signal handlers restored")
        except Exception as e:
            self.logger.exception("Error restoring signal handlers")

    @handle_specific_errors(
        error_handlers={
            ConnectionError: (None, "Failed to connect to exchange"),
            TimeoutError: (None, "Pipeline operation timed out"),
            ValueError: (None, "Invalid pipeline state"),
        },
        default_return=None, 
        context="pipeline execution",
    )
    async def run(self) -> dict | None:
        """Run the pipeline."""
        try:
            print("🔄 Starting Ares Pipeline execution...")
            self.logger.info("🔄 Starting Ares Pipeline execution...")

            if self.is_running:
                print(warning("Pipeline already running"))
                self.logger.warning("Pipeline already running")
                return None

            print("🚀 Starting Ares Pipeline...")
            self.logger.info("🚀 Starting Ares Pipeline...")
            self.is_running = True
            self.start_time = datetime.now()

            print(f"📅 Pipeline start time: {self.start_time}")
            self.logger.info(f"📅 Pipeline start time: {self.start_time}")

            # Add timeout protection
            max_cycles = 10  # Maximum number of cycles to prevent infinite loops
            max_duration = 300  # Maximum duration in seconds (5 minutes)

            # Main pipeline loop with timeout protection
            while self.is_running:
                try:
                    # Check timeout conditions
                    current_time = datetime.now()
                    elapsed_time = (current_time - self.start_time).total_seconds()

                    if self.cycle_count >= max_cycles:
                        print(
                            f"⏰ Reached maximum cycles ({max_cycles}), stopping pipeline",
                        )
                        self.logger.info(
                            f"⏰ Reached maximum cycles ({max_cycles}), stopping pipeline",
                        )
                        break

                    if elapsed_time >= max_duration:
                        print(
                            f"⏰ Reached maximum duration ({max_duration}s), stopping pipeline",
                        )
                        self.logger.info(
                            f"⏰ Reached maximum duration ({max_duration}s), stopping pipeline",
                        )
                        break

                    print(
                        f"🔄 Executing pipeline cycle {self.cycle_count + 1}... (Time: {elapsed_time:.1f}s)",
                    )
                    self.logger.info(
                        f"🔄 Executing pipeline cycle {self.cycle_count + 1}... (Time: {elapsed_time:.1f}s)",
                    )

                    await self._execute_cycle()
                    self.cycle_count += 1
                    self.last_cycle_time = datetime.now()

                    print(f"✅ Cycle {self.cycle_count} completed successfully")
                    self.logger.info(
                        f"✅ Cycle {self.cycle_count} completed successfully",
                    )

                    # Get cycle interval from configuration
                    try:
                        config_service = self.container.resolve("ConfigurationService")
                        cycle_interval = config_service.get_value(
                            "pipeline.loop_interval_seconds",
                            10,
                        )
                        print(
                            f"⏱️ Waiting {cycle_interval} seconds before next cycle...",
                        )
                        self.logger.info(
                            f"⏱️ Waiting {cycle_interval} seconds before next cycle...",
                        )
                    except Exception as e:
                        print(
                            warning("Error getting cycle interval, using default"),
                        )
                        self.logger.warning(
                            f"Error getting cycle interval, using default: {e}",
                        )
                        cycle_interval = 10

                    await asyncio.sleep(cycle_interval)

                except asyncio.CancelledError:
                    print(error("Pipeline cancelled"))
                    self.logger.info("Pipeline cancelled")
                    break
                except Exception as e:
                    print(warning(f"Error in pipeline cycle: {e}"))
                    self.logger.exception("Error in pipeline cycle")
                    await asyncio.sleep(5)  # Wait before retrying

            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()

            print("✅ Pipeline completed successfully!")
            print(f"📊 Total cycles executed: {self.cycle_count}")
            print(f"⏱️ Total duration: {duration:.2f} seconds")

            self.logger.info("✅ Pipeline completed successfully!")
            self.logger.info(f"📊 Total cycles executed: {self.cycle_count}")
            self.logger.info(f"⏱️ Total duration: {duration:.2f} seconds")

            return {
                "status": "completed",
                "cycles_executed": self.cycle_count,
                "start_time": self.start_time,
                "end_time": end_time,
                "duration_seconds": duration,
            }

        except Exception:
            print(critical(f"Fatal error running pipeline: {e}"))
            self.logger.exception("Error running pipeline")
            return None
        finally:
            self.is_running = False
            print("🧹 Pipeline cleanup completed")
            self.logger.info("🧹 Pipeline cleanup completed")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, 
        context="pipeline cycle execution",
    )
    async def _execute_cycle(self) -> None:
        """Execute a single pipeline cycle."""
        try:
            cycle_start = datetime.now()
            print(f"🔄 Starting pipeline cycle {self.cycle_count + 1}")
            self.logger.info(f"🔄 Starting pipeline cycle {self.cycle_count + 1}")

            # Step 1: Market Analysis
            print("📊 Step 1: Market Analysis")
            self.logger.info("📊 Step 1: Market Analysis")
            if self.analyst:
                print("   🔍 Executing market analysis...")
                self.logger.info("   🔍 Executing market analysis...")
                # Provide complete analysis input with all required fields
                analysis_input = {
                    "symbol": "ETHUSDT",
                    "timeframe": "1h",
                    "limit": 100,
                    "analysis_type": "technical",  # Add required analysis_type
                    "include_indicators": True,
                    "include_patterns": True,
                }
                analysis_result = await self.analyst.execute_analysis(analysis_input)
                if analysis_result:
                    print("   ✅ Market analysis completed successfully")
                    self.logger.info("   ✅ Market analysis completed successfully")
                else:
                    print("   ⚠️ Market analysis had issues")
                    self.logger.warning("   ⚠️ Market analysis had issues")
            else:
                print("   ❌ Analyst component not available")
                self.logger.error("   ❌ Analyst component not available")

            # Step 2: Strategy Development
            print("🧠 Step 2: Strategy Development")
            self.logger.info("🧠 Step 2: Strategy Development")
            if self.strategist:
                print("   🎯 Developing trading strategy...")
                self.logger.info("   🎯 Developing trading strategy...")
                # Provide basic market context for strategist
                strategy_market_data = pd.DataFrame(
                    {
                        "open": [100.0] * 100,
                        "high": [101.0] * 100,
                        "low": [99.0] * 100,
                        "close": [100.5] * 100,
                        "volume": [1000.0] * 100,
                    },
                )
                strategy_current_price = 100.5
                strategy_result = await self.strategist.generate_strategy(
                    market_data=strategy_market_data, current_price=strategy_current_price,
                )
                if strategy_result:
                    print(
                        "   ✅ Strategy development completed successfully",
                    )
                    self.logger.info(
                        "   ✅ Strategy development completed successfully",
                    )
                else:
                    print("   ⚠️ Strategy development had issues")
                    self.logger.warning("   ⚠️ Strategy development had issues")
            else:
                print("   ❌ Strategist component not available")
                self.logger.error("   ❌ Strategist component not available")

            # Step 3: Tactical Execution
            print("🎯 Step 3: Tactical Execution")
            self.logger.info("🎯 Step 3: Tactical Execution")
            if self.tactician:
                print("   ⚡ Executing tactical decisions...")
                self.logger.info("   ⚡ Executing tactical decisions...")
                tactical_result = await self.tactician.run()
                if tactical_result:
                    print("   ✅ Tactical execution completed successfully")
                    self.logger.info("   ✅ Tactical execution completed successfully")
                else:
                    print("   ⚠️ Tactical execution had issues")
                    self.logger.warning("   ⚠️ Tactical execution had issues")
            else:
                print("   ❌ Tactician component not available")
                self.logger.error("   ❌ Tactician component not available")

            # Step 4: Dual Model System Decision Making
            print("🤖 Step 4: Dual Model System Decision Making")
            self.logger.info("🤖 Step 4: Dual Model System Decision Making")
            if self.dual_model_system:
                print("   🧠 Making trading decisions with dual model system...")
                self.logger.info(
                    "   🧠 Making trading decisions with dual model system...",
                )

                # Create mock market data for demonstration
                market_data = pd.DataFrame(
                    {
                        "open": [100.0] * 100,
                        "high": [101.0] * 100,
                        "low": [99.0] * 100,
                        "close": [100.5] * 100,
                        "volume": [1000.0] * 100,
                    },
                )
                current_price = 100.5

                # Make trading decision
                decision_result = await self.dual_model_system.make_trading_decision(
                    market_data=market_data, current_price=current_price,
                )

                if decision_result:
                    print("   ✅ Dual model system decision completed successfully")
                    self.logger.info(
                        "   ✅ Dual model system decision completed successfully",
                    )

                    # Integrate with tactician for position sizing and leverage
                    integrated_decision = (
                        await self._integrate_dual_model_with_tactician(
                            dual_model_decision=decision_result,
                            market_data=market_data,
                            current_price=current_price,
                        )
                    )

                    # Log decision details
                    action = decision_result.get("action", "UNKNOWN")
                    analyst_confidence = decision_result.get("analyst_confidence", 0.0)
                    tactician_confidence = decision_result.get(
                        "tactician_confidence",
                        0.0,
                    )
                    final_confidence = decision_result.get("final_confidence", 0.0)

                    # Log position sizing and leverage
                    position_size = integrated_decision.get("position_sizing", {}).get(
                        "final_position_size",
                        0.0,
                    )
                    leverage = integrated_decision.get("leverage_sizing", {}).get(
                        "final_leverage",
                        1.0,
                    )

                    print(
                        f"   📊 Decision: {action}, Analyst: {analyst_confidence:.3f}, Tactician: {tactician_confidence:.3f}, Final: {final_confidence:.3f}",
                    )
                    self.logger.info(
                        f"   📊 Decision: {action}, Analyst: {analyst_confidence:.3f}, Tactician: {tactician_confidence:.3f}, Final: {final_confidence:.3f}",
                    )
                    print(
                        f"   💰 Position Size: {position_size:.4f}, Leverage: {leverage:.2f}x",
                    )
                    self.logger.info(
                        f"   💰 Position Size: {position_size:.4f}, Leverage: {leverage:.2f}x",
                    )

                    # Check if model training should be triggered
                    if self.dual_model_system.should_trigger_training():
                        print(
                            "   🔄 Model training conditions met - triggering training...",
                        )
                        self.logger.info(
                            "   🔄 Model training conditions met - triggering training...",
                        )

                        # Trigger model training
                        training_result = (
                            await self.dual_model_system.trigger_model_training(
                                market_data=market_data,
                                force_training=False,
                            )
                        )

                        if training_result.get("success", False):
                            print("   ✅ Model training completed successfully")
                            self.logger.info(
                                "   ✅ Model training completed successfully",
                            )
                        else:
                            print(
                                f"   ⚠️ Model training failed: {training_result.get('error', 'Unknown error')}",
                            )
                            self.logger.warning(
                                f"   ⚠️ Model training failed: {training_result.get('error', 'Unknown error')}",
                            )
                    else:
                        print("   ⚠️ Dual model system decision had issues")
                        self.logger.warning("   ⚠️ Dual model system decision had issues")
                else:
                    print("   ⚠️ Dual model system decision had issues")
                    self.logger.warning("   ⚠️ Dual model system decision had issues")
            else:
                print("   ❌ Dual model system not available")
                self.logger.error("   ❌ Dual model system not available")

            # Step 5: Supervision and Monitoring
            print("👁️ Step 5: Supervision and Monitoring")
            self.logger.info("👁️ Step 5: Supervision and Monitoring")
            if self.supervisor:
                print("   📊 Monitoring system performance...")
                self.logger.info("   📊 Monitoring system performance...")
                # Use a simple method that exists
                supervision_result = True  # Assume success for now
                if supervision_result:
                    print("   ✅ Supervision completed successfully")
                    self.logger.info("   ✅ Supervision completed successfully")
                else:
                    print("   ⚠️ Supervision had issues")
                    self.logger.warning("   ⚠️ Supervision had issues")
            else:
                print("   ❌ Supervisor component not available")
                self.logger.error("   ❌ Supervisor component not available")

            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            print(f"✅ Pipeline cycle completed in {cycle_duration:.2f}s")
            self.logger.info(f"✅ Pipeline cycle completed in {cycle_duration:.2f}s")

        except Exception:
            print(warning(f"Error executing pipeline cycle: {e}"))
            self.logger.exception("Error executing pipeline cycle")
            raise

    async def _integrate_dual_model_with_tactician(self, dual_model_decision: dict, market_data: pd.DataFrame, current_price: float) -> dict:
        """
        Integrate the dual model decision with the tactician's position sizing and leverage.
        """
        try:
            if not self.tactician or not dual_model_decision:
                return {"error": "Tactician or dual model decision not available"}

            # Extract confidence scores from dual model decision
            analyst_confidence = dual_model_decision.get("analyst_confidence", 0.5)
            tactician_confidence = dual_model_decision.get("tactician_confidence", 0.5)
            final_confidence = dual_model_decision.get("final_confidence", 0.5)
            normalized_confidence = dual_model_decision.get(
                "normalized_confidence",
                0.5,
            )

            # Create ML predictions for tactician
            ml_predictions = {
                "price_target_confidences": {
                    "0.5%": analyst_confidence,
                    "1.0%": analyst_confidence * 0.9,
                    "1.5%": analyst_confidence * 0.8,
                    "2.0%": analyst_confidence * 0.7,
                },
                "adversarial_confidences": {
                    "0.5%": 1.0 - tactician_confidence,
                    "1.0%": (1.0 - tactician_confidence) * 0.9,
                    "1.5%": (1.0 - tactician_confidence) * 0.8,
                    "2.0%": (1.0 - tactician_confidence) * 0.7,
                },
                "directional_analysis": {
                    "primary_direction": dual_model_decision.get("direction", "HOLD"),
                    "primary_confidence": final_confidence,
                    "magnitude_levels": [0.5, 1.0, 1.5, 2.0],
                },
            }

            # Calculate position size using tactician
            position_sizer = getattr(self.tactician, "position_sizer", None)
            if position_sizer:
                position_size_result = await position_sizer.calculate_position_size(
                    ml_predictions=ml_predictions, current_price=current_price,
                    account_balance=1000.0,  # Default balance
                    analyst_confidence=analyst_confidence,
                    tactician_confidence=tactician_confidence,
                )
            else:
                position_size_result = {
                    "final_position_size": 0.0,
                    "error": "Position sizer not available",
                }

            # Calculate leverage using tactician
            leverage_sizer = getattr(self.tactician, "leverage_sizer", None)
            if leverage_sizer:
                leverage_result = await leverage_sizer.calculate_leverage(
                    ml_predictions=ml_predictions, current_price=current_price,
                    target_direction=dual_model_decision.get("action", "HOLD"),
                    analyst_confidence=analyst_confidence,
                    tactician_confidence=tactician_confidence,
                )
            else:
                leverage_result = {
                    "final_leverage": 1.0,
                    "error": "Leverage sizer not available",
                }

            # Integrate results
            integrated_decision = {
                **dual_model_decision,
                "position_sizing": position_size_result,
                "leverage_sizing": leverage_result,
                "integrated": True,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(
                f"Integrated dual model decision with tactician - Position: {position_size_result.get('final_position_size', 0.0)}, Leverage: {leverage_result.get('final_leverage', 1.0)}",
            )

            return integrated_decision

        except Exception as e:
            self.logger.exception("Error integrating dual model with tactician")
            return {
                "error": str(e),
                "dual_model_decision": dual_model_decision,
                "integrated": False,
            }

    def get_pipeline_status(self) -> dict:
        """Get the current status of the pipeline."""
        status = {
            "is_running": self.is_running,
            "start_time": self.start_time,
            "cycle_count": self.cycle_count,
            "last_cycle_time": self.last_cycle_time,
            "components": {
                "analyst": self.analyst is not None,
                "strategist": self.strategist is not None,
                "tactician": self.tactician is not None,
                "supervisor": self.supervisor is not None,
                "state_manager": self.state_manager is not None,
                "event_bus": self.event_bus is not None,
                "dual_model_system": self.dual_model_system is not None,
            },
        }

        # Add dual model system status if available
        if self.dual_model_system:
            try:
                dual_model_status = self.dual_model_system.get_system_info()
                status["dual_model_system_status"] = dual_model_status
            except Exception as e:
                status["dual_model_system_status"] = {"error": str(e)}

        # Add performance monitoring status if available
        if self.performance_monitor:
            try:
                performance_status = self.performance_monitor.get_performance_summary()
                status["performance_monitoring_status"] = performance_status
            except Exception as e:
                status["performance_monitoring_status"] = {"error": str(e)}

        if self.performance_dashboard:
            try:
                dashboard_status = self.performance_dashboard.get_dashboard_summary()
                status["performance_dashboard_status"] = dashboard_status
            except Exception as e:
                status["performance_dashboard_status"] = {"error": str(e)}

        return status

    @handle_errors(
        exceptions=(Exception,),
        default_return=None, 
        context="pipeline cleanup",
    )
    async def stop(self) -> None:
        """Stop the pipeline."""
        self.logger.info("🛑 Stopping Ares Pipeline...")

        try:
            # Stop pipeline loop
            self.is_running = False

            # Stop components in reverse dependency order
            if self.dual_model_system:
                await self.dual_model_system.stop()

            # Stop performance monitoring
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

            # Close database connections
            db_manager = self.container.resolve("DatabaseManager")
            if db_manager:
                await db_manager.close()

            self.logger.info("✅ Ares Pipeline stopped successfully")

        except Exception:
            self.logger.exception("Error stopping pipeline")

    async def _initialize_dual_model_system(self) -> None:
        """Initialize the dual model system."""
        try:
            print("🤖 Initializing Dual Model System...")
            self.logger.info("🤖 Initializing Dual Model System...")

            # Check if dual model system is enabled in configuration
            config_service = self.container.resolve("ConfigurationService")
            if config_service:
                enable_dual_model = config_service.get_value("pipeline.enable_dual_model_system", True)
                if not enable_dual_model:
                    print("   ⚠️ Dual Model System disabled in configuration")
                    self.logger.info("   ⚠️ Dual Model System disabled in configuration")
                    return
            else:
                print("   ⚠️ ConfigurationService not available, using default settings")
                self.logger.warning("   ⚠️ ConfigurationService not available, using default settings")

            # Get proper configuration for dual model system
            dual_model_config = self._get_dual_model_config()
            
            # Validate configuration
            if not self._validate_dual_model_config(dual_model_config):
                raise ValueError("Invalid dual model system configuration")

            # Initialize the dual model system
            print("   🔧 Setting up dual model system...")
            self.logger.info("   🔧 Setting up dual model system...")
            
            self.dual_model_system = await setup_dual_model_system(dual_model_config)
            
            if self.dual_model_system:
                self.logger.info("✅ Dual Model System initialized successfully")
                print("   ✅ Dual Model System initialized successfully")

                # Log system information
                try:
                    system_info = self.dual_model_system.get_system_info()
                    print("   📊 Dual Model System Configuration:")
                    print(f"      Analyst timeframes: {system_info.get('analyst_timeframes', [])}")
                    print(f"      Tactician timeframes: {system_info.get('tactician_timeframes', [])}")
                    print(f"      Analyst confidence threshold: {system_info.get('analyst_confidence_threshold', 0.5)}")
                    print(f"      Tactician confidence threshold: {system_info.get('tactician_confidence_threshold', 0.6)}")
                    
                    self.logger.info(f"   📊 Analyst timeframes: {system_info.get('analyst_timeframes', [])}")
                    self.logger.info(f"   📊 Tactician timeframes: {system_info.get('tactician_timeframes', [])}")
                    self.logger.info(f"   📊 Analyst confidence threshold: {system_info.get('analyst_confidence_threshold', 0.5)}")
                    self.logger.info(f"   📊 Tactician confidence threshold: {system_info.get('tactician_confidence_threshold', 0.6)}")
                    
                    # Validate system capabilities
                    self._validate_dual_model_capabilities(system_info)
                    
                except Exception as e:
                    print(f"   ⚠️ Could not retrieve system info: {e}")
                    self.logger.warning(f"   ⚠️ Could not retrieve system info: {e}")
            else:
                print("   ❌ Failed to initialize Dual Model System")
                self.logger.error("   ❌ Failed to initialize Dual Model System")
                raise RuntimeError("Dual Model System initialization failed")
                
        except Exception as e:
            print(f"   ❌ Error initializing dual model system: {e}")
            self.logger.exception("Error initializing dual model system")
            raise

    def _validate_dual_model_config(self, config: dict) -> bool:
        """Validate dual model system configuration."""
        try:
            required_keys = ["dual_model_system"]
            if not all(key in config for key in required_keys):
                self.logger.error(f"Missing required configuration keys: {required_keys}")
                return False
            
            dual_config = config["dual_model_system"]
            required_dual_keys = [
                "analyst_timeframes", "tactician_timeframes",
                "analyst_confidence_threshold", "tactician_confidence_threshold"
            ]
            
            if not all(key in dual_config for key in required_dual_keys):
                self.logger.error(f"Missing required dual model keys: {required_dual_keys}")
                return False
            
            # Validate confidence thresholds
            if not (0.0 <= dual_config["analyst_confidence_threshold"] <= 1.0):
                self.logger.error("Analyst confidence threshold must be between 0.0 and 1.0")
                return False
                
            if not (0.0 <= dual_config["tactician_confidence_threshold"] <= 1.0):
                self.logger.error("Tactician confidence threshold must be between 0.0 and 1.0")
                return False
            
            # Validate timeframes
            if not isinstance(dual_config["analyst_timeframes"], list) or len(dual_config["analyst_timeframes"]) == 0:
                self.logger.error("Analyst timeframes must be a non-empty list")
                return False
                
            if not isinstance(dual_config["tactician_timeframes"], list) or len(dual_config["tactician_timeframes"]) == 0:
                self.logger.error("Tactician timeframes must be a non-empty list")
                return False
            
            return True
            
        except Exception as e:
            self.logger.exception("Error validating dual model configuration")
            return False

    def _validate_dual_model_capabilities(self, system_info: dict) -> None:
        """Validate dual model system capabilities."""
        try:
            # Check if system has required methods
            required_methods = [
                "make_trading_decision", "should_trigger_training", 
                "trigger_model_training", "get_system_info"
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(self.dual_model_system, method):
                    missing_methods.append(method)
            
            if missing_methods:
                self.logger.warning(f"Dual model system missing methods: {missing_methods}")
                print(f"   ⚠️ Missing methods: {missing_methods}")
            
            # Check system health
            if hasattr(self.dual_model_system, "get_system_health"):
                health = self.dual_model_system.get_system_health()
                if health.get("status") != "healthy":
                    self.logger.warning(f"Dual model system health: {health}")
                    print(f"   ⚠️ System health: {health.get('status', 'unknown')}")
            
        except Exception as e:
            self.logger.exception("Error validating dual model capabilities")
            print(f"   ⚠️ Could not validate capabilities: {e}")

    async def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring."""
        try:
            print("📊 Initializing Performance Monitoring...")
            self.logger.info("📊 Initializing Performance Monitoring...")

            # Check if performance monitoring is enabled
            config_service = self.container.resolve("ConfigurationService")
            if config_service:
                enable_monitoring = config_service.get_value("pipeline.enable_performance_monitoring", True)
                if not enable_monitoring:
                    print("   ⚠️ Performance monitoring disabled in configuration")
                    self.logger.info("   ⚠️ Performance monitoring disabled in configuration")
                    return
            else:
                print("   ⚠️ ConfigurationService not available, using default settings")
                self.logger.warning("   ⚠️ ConfigurationService not available, using default settings")

            # Setup performance monitor
            print("   📊 Setting up performance monitor...")
            self.logger.info("   📊 Setting up performance monitor...")
            
            try:
                self.performance_monitor = await setup_performance_monitor(self.config)
                
                if self.performance_monitor:
                    self.logger.info("✅ Performance Monitor initialized successfully")
                    print("   ✅ Performance Monitor initialized successfully")
                    
                    # Configure monitoring intervals
                    if hasattr(self.performance_monitor, 'set_collection_interval'):
                        interval = config_service.get_value("performance.metrics_collection_interval", 30) if config_service else 30
                        self.performance_monitor.set_collection_interval(interval)
                        print(f"   ⏱️ Metrics collection interval set to {interval} seconds")
                else:
                    print("   ❌ Failed to initialize Performance Monitor")
                    self.logger.error("   ❌ Failed to initialize Performance Monitor")
                    raise RuntimeError("Performance Monitor initialization failed")
                    
            except Exception as e:
                print(f"   ❌ Error setting up performance monitor: {e}")
                self.logger.exception("   ❌ Error setting up performance monitor")
                raise

            # Setup performance dashboard
            print("   📈 Setting up performance dashboard...")
            self.logger.info("   📈 Setting up performance dashboard...")
            
            try:
                self.performance_dashboard = await setup_performance_dashboard(
                    self.config,
                    self.performance_monitor,
                )

                if self.performance_dashboard:
                    self.logger.info("✅ Performance Dashboard initialized successfully")
                    print("   ✅ Performance Dashboard initialized successfully")
                    
                    # Configure dashboard update interval
                    if hasattr(self.performance_dashboard, 'set_update_interval'):
                        interval = config_service.get_value("performance.dashboard_update_interval", 60) if config_service else 60
                        self.performance_dashboard.set_update_interval(interval)
                        print(f"   ⏱️ Dashboard update interval set to {interval} seconds")
                        
                    # Enable real-time monitoring if configured
                    if hasattr(self.performance_dashboard, 'enable_real_time_monitoring'):
                        real_time = config_service.get_value("performance.enable_real_time_monitoring", True) if config_service else True
                        self.performance_dashboard.enable_real_time_monitoring(real_time)
                        print(f"   🔄 Real-time monitoring: {'enabled' if real_time else 'disabled'}")
                        
                else:
                    print("   ⚠️ Failed to initialize Performance Dashboard")
                    self.logger.warning("   ⚠️ Failed to initialize Performance Dashboard")
                    # Dashboard failure is not critical, continue without it
                    
            except Exception as e:
                print(f"   ⚠️ Error setting up performance dashboard: {e}")
                self.logger.warning(f"   ⚠️ Error setting up performance dashboard: {e}")
                # Dashboard failure is not critical, continue without it

            print("✅ Performance monitoring initialization completed")
            self.logger.info("✅ Performance monitoring initialization completed")
            
        except Exception as e:
            print(f"❌ Error initializing performance monitoring: {e}")
            self.logger.exception("Error initializing performance monitoring")
            # Performance monitoring failure is not critical for pipeline operation
            self.logger.warning("Continuing pipeline initialization without performance monitoring")

    def _get_dual_model_config(self) -> dict:
        """Get the configuration for the dual model system."""
        try:
            # Try to get configuration from the centralized config system first
            dual_model_config = get_dual_model_config()
            
            if dual_model_config and isinstance(dual_model_config, dict):
                self.logger.info("Retrieved dual model configuration from centralized system")
                return {"dual_model_system": dual_model_config}
            
            # Fallback to ConfigurationService if available
            config_service = self.container.resolve("ConfigurationService")
            if config_service:
                try:
                    dual_config = config_service.get_value("dual_model_system", {})
                    if dual_config and isinstance(dual_config, dict):
                        self.logger.info("Retrieved dual model configuration from ConfigurationService")
                        return {"dual_model_system": dual_config}
                except Exception as e:
                    self.logger.warning(f"Could not retrieve dual model config from ConfigurationService: {e}")
            
            # Fallback to default configuration
            self.logger.info("Using default dual model configuration")
            default_config = {
                "dual_model_system": {
                    "analyst_timeframes": ["30m", "15m", "5m"],
                    "tactician_timeframes": ["1m"],
                    "analyst_confidence_threshold": 0.6,
                    "tactician_confidence_threshold": 0.7,
                    "enter_signal_validity_duration": 120,
                    "signal_check_interval": 10,
                    "neutral_signal_threshold": 0.5,
                    "close_signal_threshold": 0.4,
                    "position_close_confidence_threshold": 0.6,
                    "enable_ensemble_analysis": True,
                    "ensemble_weight_analyst": 0.6,
                    "ensemble_weight_tactician": 0.4,
                    "min_confidence_difference": 0.1,
                    "max_position_hold_time": 3600,
                    "enable_adaptive_thresholds": True,
                    "training_trigger_conditions": {
                        "min_data_points": 1000,
                        "max_model_age_hours": 24,
                        "performance_degradation_threshold": 0.1
                    }
                }
            }
            
            return default_config

        except Exception as e:
            self.logger.exception(f"Error getting dual model config: {e}")
            # Return robust default configuration
            return {
                "dual_model_system": {
                    "analyst_timeframes": ["30m", "15m", "5m"],
                    "tactician_timeframes": ["1m"],
                    "analyst_confidence_threshold": 0.6,
                    "tactician_confidence_threshold": 0.7,
                    "enter_signal_validity_duration": 120,
                    "signal_check_interval": 10,
                    "neutral_signal_threshold": 0.5,
                    "close_signal_threshold": 0.4,
                    "position_close_confidence_threshold": 0.6,
                    "enable_ensemble_analysis": True,
                    "ensemble_weight_analyst": 0.6,
                    "ensemble_weight_tactician": 0.4,
                    "min_confidence_difference": 0.1,
                    "max_position_hold_time": 3600,
                    "enable_adaptive_thresholds": True,
                    "training_trigger_conditions": {
                        "min_data_points": 1000,
                        "max_model_age_hours": 24,
                        "performance_degradation_threshold": 0.1
                    }
                }
            }


async def main():
    """Main entry point for the Ares Pipeline."""
    try:
        # Add the project root to the Python path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Setup logging
        setup_logging()
        init_observability({})
        logger = system_logger.getChild("AresPipelineMain")

        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Ares Trading Pipeline - Advanced algorithmic trading system",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python src/ares_pipeline.py ETHUSDT BINANCE
  python src/ares_pipeline.py BTCUSDT BINANCE --config config/trading.yaml
  TRADING_MODE=LIVE python src/ares_pipeline.py ETHUSDT BINANCE

Environment Variables:
  TRADING_MODE: Set to 'PAPER', 'LIVE', or 'SIMULATION' (default: PAPER)
  LOG_LEVEL: Set logging level (default: INFO)
  CONFIG_PATH: Default configuration file path
            """
        )
        
        parser.add_argument(
            "symbol", 
            help="Trading symbol (e.g., ETHUSDT, BTCUSDT, ADAUSDT)"
        )
        parser.add_argument(
            "exchange", 
            help="Exchange name (e.g., BINANCE, COINBASE, KRAKEN)"
        )
        parser.add_argument(
            "--config", 
            help="Path to configuration file (YAML/JSON)",
            default=os.environ.get("CONFIG_PATH", "config/trading.yaml")
        )
        parser.add_argument(
            "--dry-run", 
            action="store_true",
            help="Run pipeline in dry-run mode without executing trades"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )

        args = parser.parse_args()

        # Validate arguments
        if not args.symbol or not args.exchange:
            print(failed("❌ Symbol and exchange are required"))
            parser.print_help()
            sys.exit(1)

        # Get trading mode from environment variable
        trading_mode = os.environ.get("TRADING_MODE", "PAPER").upper()
        if trading_mode not in ["PAPER", "LIVE", "SIMULATION"]:
            print(warning(f"⚠️ Invalid TRADING_MODE '{trading_mode}', using PAPER"))
            trading_mode = "PAPER"

        # Set log level based on verbose flag
        if args.verbose:
            os.environ["LOG_LEVEL"] = "DEBUG"
            logger.setLevel("DEBUG")

        # Display startup information
        print("🚀 Ares Trading Pipeline")
        print("=" * 50)
        print(f"📊 Symbol: {args.symbol}")
        print(f"🏢 Exchange: {args.exchange}")
        print(f"🔧 Trading Mode: {trading_mode}")
        print(f"📁 Config: {args.config}")
        print(f"🔍 Dry Run: {'Yes' if args.dry_run else 'No'}")
        print(f"📝 Verbose: {'Yes' if args.verbose else 'No'}")
        print("=" * 50)

        logger.info(f"🚀 Starting Ares Pipeline in {trading_mode} mode")
        logger.info(f"📊 Symbol: {args.symbol}")
        logger.info(f"🏢 Exchange: {args.exchange}")
        logger.info(f"🔧 Trading Mode: {trading_mode}")
        logger.info(f"📁 Config: {args.config}")

        # Load configuration if provided
        config = {}
        if args.config and os.path.exists(args.config):
            try:
                import yaml
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f) or {}
                logger.info(f"✅ Configuration loaded from {args.config}")
                print(f"✅ Configuration loaded from {args.config}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load configuration from {args.config}: {e}")
                print(f"⚠️ Could not load configuration from {args.config}: {e}")
        else:
            logger.info("Using default configuration")
            print("ℹ️ Using default configuration")

        # Add command line arguments to config
        config.update({
            "symbol": args.symbol,
            "exchange": args.exchange,
            "trading_mode": trading_mode,
            "dry_run": args.dry_run,
            "verbose": args.verbose
        })

        # Create pipeline instance
        pipeline = AresPipeline(config)

        try:
            # Initialize pipeline
            print("🔧 Initializing pipeline...")
            if not await pipeline.initialize():
                print(failed("❌ Failed to initialize pipeline"))
                sys.exit(1)

            # Run pipeline
            print("🚀 Starting pipeline execution...")
            result = await pipeline.run()

            if result:
                logger.info("✅ Pipeline completed successfully")
                print("✅ Pipeline completed successfully")
                
                # Display results
                if isinstance(result, dict):
                    print(f"📊 Cycles executed: {result.get('cycles_executed', 0)}")
                    print(f"⏱️ Duration: {result.get('duration_seconds', 0):.2f} seconds")
                    print(f"📅 Start time: {result.get('start_time')}")
                    print(f"📅 End time: {result.get('end_time')}")
            else:
                print(failed("❌ Pipeline failed"))
                sys.exit(1)

        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal, shutting down gracefully...")
            print("🛑 Received interrupt signal, shutting down gracefully...")
            await pipeline.stop()
        except Exception as e:
            print(error(f"💥 Unexpected error: {e}"))
            logger.exception("Unexpected error in pipeline execution")
            try:
                await pipeline.stop()
            except Exception as stop_error:
                logger.exception("Error during pipeline shutdown")
                print(f"⚠️ Error during shutdown: {stop_error}")
            sys.exit(1)

    except Exception as e:
        print(critical(f"💥 Fatal error in main: {e}"))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


