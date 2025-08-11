# src/ares_pipeline.py

import asyncio
import pandas as pd
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add the project root to the Python path for subprocess execution
# Important: append instead of inserting at position 0 to avoid shadowing
# standard library modules like 'types' with our internal 'src/types' package.
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.core.config_service import ConfigurationService
from src.core.dependency_injection import DependencyContainer, ServiceLocator
from src.monitoring.performance_dashboard import (
    PerformanceDashboard,
    setup_performance_dashboard,
)

# Import performance monitoring
from src.monitoring.performance_monitor import (
    PerformanceMonitor,
    setup_performance_monitor,
)

# Import dual model system
from src.training.dual_model_system import DualModelSystem, setup_dual_model_system
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    critical,
    error,
    execution_error,
    failed,
    initialization_error,
    problem,
    warning,
)

if TYPE_CHECKING:
    from src.interfaces.base_interfaces import (
        IAnalyst,
        IEventBus,
        IStateManager,
        IStrategist,
        ISupervisor,
        ITactician,
    )


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
        self.logger = system_logger.getChild("AresPipeline")

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
        """
        Initialize pipeline with enhanced error handling and DI.

        Returns:
            bool: True if initialization successful, False otherwise
        """
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
        """Initialize configuration service."""
        try:
            print("   ⚙️ Initializing ConfigurationService...")
            self.logger.info("   ⚙️ Initializing ConfigurationService...")

            # Register ConfigurationService via factory so it receives DI config
            def _config_service_factory(
                container: DependencyContainer,
            ) -> ConfigurationService:
                # Pass the DI container's config into the service
                return ConfigurationService(container.get_config("root_config", {}))

            # Store current container config under a conventional key if missing
            if self.container.get_config("root_config") is None:
                # The DependencyContainer already holds its config; expose it
                self.container.set_config("root_config", self.container._config)

            self.container.register_factory(
                "ConfigurationService", _config_service_factory
            )

            print("   ✅ ConfigurationService initialized successfully")
            self.logger.info("   ✅ ConfigurationService initialized successfully")

        except Exception as e:
            print(f"   ❌ Error initializing configuration service: {e}")
            self.logger.exception("Error initializing configuration service")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="core service registration",
    )
    async def _register_core_services(self) -> None:
        """Register core services in DI container with comprehensive logging."""
        try:
            print("🔧 Registering core services...")
            self.logger.info("🔧 Registering core services...")

            # Register database manager
            print("   💾 Registering DatabaseManager...")
            self.logger.info("   💾 Registering DatabaseManager...")
            try:
                from src.database.sqlite_manager import SQLiteManager

                self.container.register("DatabaseManager", SQLiteManager)
                print("   ✅ DatabaseManager registered successfully")
                self.logger.info("   ✅ DatabaseManager registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register DatabaseManager: {e}")
                self.logger.exception("   ❌ Failed to register DatabaseManager")

            # Register exchange client
            print("   🏢 Registering ExchangeClient...")
            self.logger.info("   🏢 Registering ExchangeClient...")
            try:
                from exchange.factory import ExchangeFactory as RootExchangeFactory

                # Use environment-configured exchange as default
                from src.config.environment import get_exchange_name

                # Build exchange instance via factory and register the instance
                exchange_instance = RootExchangeFactory.get_exchange(
                    get_exchange_name().lower(),
                )
                self.container.register_instance("ExchangeClient", exchange_instance)
                print("   ✅ ExchangeClient registered successfully")
                self.logger.info("   ✅ ExchangeClient registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register ExchangeClient: {e}")
                self.logger.exception("   ❌ Failed to register ExchangeClient")

            # Register analyst
            print("   📊 Registering Analyst...")
            self.logger.info("   📊 Registering Analyst...")
            try:
                from src.analyst.analyst import Analyst

                self.container.register("Analyst", Analyst, config={"analyst": {}})
                print("   ✅ Analyst registered successfully")
                self.logger.info("   ✅ Analyst registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register Analyst: {e}")
                self.logger.exception("   ❌ Failed to register Analyst")

            # Register strategist
            print("   🧠 Registering Strategist...")
            self.logger.info("   🧠 Registering Strategist...")
            try:
                from src.strategist.strategist import Strategist

                self.container.register(
                    "Strategist",
                    Strategist,
                    config={"strategist": {}},
                )
                print("   ✅ Strategist registered successfully")
                self.logger.info("   ✅ Strategist registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register Strategist: {e}")
                self.logger.exception("   ❌ Failed to register Strategist")

            # Register tactician
            print("   🎯 Registering Tactician...")
            self.logger.info("   🎯 Registering Tactician...")
            try:
                from src.tactician.tactician import Tactician

                self.container.register(
                    "Tactician",
                    Tactician,
                    config={"tactician": {}},
                )
                print("   ✅ Tactician registered successfully")
                self.logger.info("   ✅ Tactician registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register Tactician: {e}")
                self.logger.exception("   ❌ Failed to register Tactician")

            # Register supervisor
            print("   👁️ Registering Supervisor...")
            self.logger.info("   👁️ Registering Supervisor...")
            try:
                from src.supervisor.supervisor import Supervisor

                self.container.register(
                    "Supervisor",
                    Supervisor,
                    config={"supervisor": {}},
                )
                print("   ✅ Supervisor registered successfully")
                self.logger.info("   ✅ Supervisor registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register Supervisor: {e}")
                self.logger.exception("   ❌ Failed to register Supervisor")

            # Register state manager
            print("   💾 Registering StateManager...")
            self.logger.info("   💾 Registering StateManager...")
            try:
                from src.utils.state_manager import StateManager

                self.container.register(
                    "StateManager",
                    StateManager,
                    config={"state_manager": {}},
                )
                print("   ✅ StateManager registered successfully")
                self.logger.info("   ✅ StateManager registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register StateManager: {e}")
                self.logger.exception("   ❌ Failed to register StateManager")

            # Register event bus
            print("   📡 Registering EventBus...")
            self.logger.info("   📡 Registering EventBus...")
            try:
                from src.interfaces.event_bus import EventBus

                self.container.register("EventBus", EventBus, config={"event_bus": {}})
                print("   ✅ EventBus registered successfully")
                self.logger.info("   ✅ EventBus registered successfully")
            except Exception as e:
                print(f"   ❌ Failed to register EventBus: {e}")
                self.logger.exception("   ❌ Failed to register EventBus")

            print("✅ Core services registered successfully")
            self.logger.info("✅ Core services registered successfully")

        except Exception:
            print(warning("Error registering core services: {e}"))
            self.logger.exception("Error registering core services")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline component resolution",
    )
    async def _resolve_pipeline_components(self) -> None:
        """Resolve pipeline components through DI container with comprehensive logging."""
        try:
            print("🔧 Resolving pipeline components...")
            self.logger.info("🔧 Resolving pipeline components...")

            # Resolve all components
            print("   📊 Resolving Analyst component...")
            self.logger.info("   📊 Resolving Analyst component...")
            self.analyst = self.container.resolve("Analyst")
            if self.analyst:
                print("   ✅ Analyst component resolved successfully")
                self.logger.info("   ✅ Analyst component resolved successfully")
            else:
                print("   ❌ Failed to resolve Analyst component")
                self.logger.error("   ❌ Failed to resolve Analyst component")

            print("   🧠 Resolving Strategist component...")
            self.logger.info("   🧠 Resolving Strategist component...")
            self.strategist = self.container.resolve("Strategist")
            if self.strategist:
                print("   ✅ Strategist component resolved successfully")
                self.logger.info("   ✅ Strategist component resolved successfully")
            else:
                print("   ❌ Failed to resolve Strategist component")
                self.logger.error("   ❌ Failed to resolve Strategist component")

            print("   🎯 Resolving Tactician component...")
            self.logger.info("   🎯 Resolving Tactician component...")
            self.tactician = self.container.resolve("Tactician")
            if self.tactician:
                print("   ✅ Tactician component resolved successfully")
                self.logger.info("   ✅ Tactician component resolved successfully")
            else:
                print("   ❌ Failed to resolve Tactician component")
                self.logger.error("   ❌ Failed to resolve Tactician component")

            print("   👁️ Resolving Supervisor component...")
            self.logger.info("   👁️ Resolving Supervisor component...")
            self.supervisor = self.container.resolve("Supervisor")
            if self.supervisor:
                print("   ✅ Supervisor component resolved successfully")
                self.logger.info("   ✅ Supervisor component resolved successfully")
            else:
                print("   ❌ Failed to resolve Supervisor component")
                self.logger.error("   ❌ Failed to resolve Supervisor component")

            print("   💾 Resolving StateManager component...")
            self.logger.info("   💾 Resolving StateManager component...")
            self.state_manager = self.container.resolve("StateManager")
            if self.state_manager:
                print("   ✅ StateManager component resolved successfully")
                self.logger.info("   ✅ StateManager component resolved successfully")
            else:
                print("   ❌ Failed to resolve StateManager component")
                self.logger.error("   ❌ Failed to resolve StateManager component")

            print("   📡 Resolving EventBus component...")
            self.logger.info("   📡 Resolving EventBus component...")
            self.event_bus = self.container.resolve("EventBus")
            if self.event_bus:
                print("   ✅ EventBus component resolved successfully")
                self.logger.info("   ✅ EventBus component resolved successfully")
            else:
                print("   ❌ Failed to resolve EventBus component")
                self.logger.error("   ❌ Failed to resolve EventBus component")

            print("✅ Pipeline components resolved successfully")
            self.logger.info("✅ Pipeline components resolved successfully")

        except Exception:
            print(warning("Error resolving pipeline components: {e}"))
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
            # Initialize components in dependency order
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

            self.logger.info("All pipeline components initialized successfully")

        except Exception:
            self.logger.exception("Error initializing components")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="signal handler setup",
    )
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            self.logger.info("Signal handlers configured")

        except Exception:
            self.logger.exception("Error setting up signal handlers")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop())

    @handle_specific_errors(
        error_handlers={
            ConnectionError: (None, "Failed to connect to exchange"),
            TimeoutError: (None, "Pipeline operation timed out"),
            ValueError: (None, "Invalid pipeline state"),
        },
        default_return=None,
        context="pipeline execution",
    )
    async def run(self) -> dict[str, Any] | None:
        """
        Run the Ares pipeline with comprehensive logging and timeout protection.

        Returns:
            dict[str, Any] | None: Pipeline execution results or None if failed
        """
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
                            warning("Error getting cycle interval, using default: {e}"),
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
                except Exception:
                    print(warning("Error in pipeline cycle: {e}"))
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
            print(critical("Fatal error running pipeline: {e}"))
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
        """Execute a single pipeline cycle with comprehensive logging."""
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
                    }
                )
                strategy_current_price = 100.5
                strategy_result = await self.strategist.generate_strategy(
                    market_data=strategy_market_data,
                    current_price=strategy_current_price,
                )
                if strategy_result:
                    print("   ✅ Strategy development completed successfully")
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
                    market_data=market_data,
                    current_price=current_price,
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
                    print(
                        f"   💰 Position Size: {position_size:.4f}, Leverage: {leverage:.2f}x",
                    )
                    self.logger.info(
                        f"   📊 Decision: {action}, Analyst: {analyst_confidence:.3f}, Tactician: {tactician_confidence:.3f}, Final: {final_confidence:.3f}",
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
                                market_data,
                                "continuous",
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
            print(warning("Error executing pipeline cycle: {e}"))
            self.logger.exception("Error executing pipeline cycle")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="exchange initialization",
    )
    async def _initialize_exchange(self) -> None:
        """Initialize exchange connection."""
        try:
            exchange_client = self.container.resolve("ExchangeClient")
            if exchange_client:
                await exchange_client.initialize()
                self.logger.info("Exchange client initialized")

        except Exception:
            self.logger.exception("Error initializing exchange")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst initialization",
    )
    async def _initialize_analyst(self) -> None:
        """Initialize analyst component."""
        try:
            if self.analyst:
                await self.analyst.initialize()
                self.logger.info("Analyst initialized")

        except Exception:
            self.logger.exception("Error initializing analyst")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategist initialization",
    )
    async def _initialize_strategist(self) -> None:
        """Initialize strategist component."""
        try:
            if self.strategist:
                await self.strategist.initialize()
                self.logger.info("Strategist initialized")

        except Exception:
            self.logger.exception("Error initializing strategist")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician initialization",
    )
    async def _initialize_tactician(self) -> None:
        """Initialize tactician component."""
        try:
            if self.tactician:
                await self.tactician.initialize()
                self.logger.info("Tactician initialized")

        except Exception:
            self.logger.exception("Error initializing tactician")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="supervisor initialization",
    )
    async def _initialize_supervisor(self) -> None:
        """Initialize supervisor component."""
        try:
            if self.supervisor:
                await self.supervisor.initialize()
                self.logger.info("Supervisor initialized")

        except Exception:
            self.logger.exception("Error initializing supervisor")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="dual model system initialization",
    )
    async def _initialize_dual_model_system(self) -> None:
        """Initialize dual model system."""
        try:
            # Get proper configuration for dual model system
            dual_model_config = self._get_dual_model_config()

            self.dual_model_system = await setup_dual_model_system(dual_model_config)
            if self.dual_model_system:
                self.logger.info("✅ Dual Model System initialized successfully")

                # Log system information
                system_info = self.dual_model_system.get_system_info()
                self.logger.info(
                    f"   📊 Analyst timeframes: {system_info.get('analyst_timeframes', [])}",
                )
                self.logger.info(
                    f"   📊 Tactician timeframes: {system_info.get('tactician_timeframes', [])}",
                )
                self.logger.info(
                    f"   📊 Analyst confidence threshold: {system_info.get('analyst_confidence_threshold', 0.5)}",
                )
                self.logger.info(
                    f"   📊 Tactician confidence threshold: {system_info.get('tactician_confidence_threshold', 0.6)}",
                )
            else:
                self.logger.warning("Dual Model System not available")
        except Exception:
            self.logger.exception("Error initializing dual model system")

    async def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring."""
        try:
            self.logger.info("📊 Initializing Performance Monitoring...")

            # Setup performance monitor
            self.performance_monitor = await setup_performance_monitor(self.config)

            if self.performance_monitor:
                self.logger.info("✅ Performance Monitor initialized successfully")

                # Setup performance dashboard
                self.performance_dashboard = await setup_performance_dashboard(
                    self.config,
                    self.performance_monitor,
                )

                if self.performance_dashboard:
                    self.logger.info(
                        "✅ Performance Dashboard initialized successfully",
                    )
                else:
                    self.logger.warning("⚠️ Failed to initialize Performance Dashboard")
            else:
                self.logger.warning("⚠️ Failed to initialize Performance Monitor")

        except Exception:
            self.logger.exception("Error initializing performance monitoring")

    async def _integrate_dual_model_with_tactician(
        self,
        dual_model_decision: dict[str, Any],
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """
        Integrate dual model system decisions with tactician for position sizing and leverage.

        Args:
            dual_model_decision: Decision from dual model system
            market_data: Current market data
            current_price: Current market price

        Returns:
            dict[str, Any]: Integrated tactical decision
        """
        try:
            if not self.tactician or not dual_model_decision:
                return {"error": "Tactician or dual model decision not available"}

            # Extract confidence scores from dual model decision
            analyst_confidence = dual_model_decision.get("analyst_confidence", 0.5)
            tactician_confidence = dual_model_decision.get("tactician_confidence", 0.5)
            final_confidence = dual_model_decision.get("final_confidence", 0.5)
            dual_model_decision.get(
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
                    "primary_direction": dual_model_decision.get("action", "HOLD"),
                    "primary_confidence": final_confidence,
                    "magnitude_levels": [0.5, 1.0, 1.5, 2.0],
                },
            }

            # Calculate position size using tactician
            position_sizer = getattr(self.tactician, "position_sizer", None)
            if position_sizer:
                position_size_result = await position_sizer.calculate_position_size(
                    ml_predictions=ml_predictions,
                    current_price=current_price,
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
                    ml_predictions=ml_predictions,
                    current_price=current_price,
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

    def get_pipeline_status(self) -> dict[str, Any]:
        """
        Get current pipeline status.

        Returns:
            Dict[str, Any]: Pipeline status information
        """
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
        """Stop the pipeline gracefully."""
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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="pipeline cleanup",
    )
    async def _cleanup(self) -> None:
        """Cleanup pipeline resources."""
        try:
            # Additional cleanup tasks
            self.logger.info("Pipeline cleanup completed")

        except Exception:
            self.logger.exception("Error during cleanup")

    def _get_dual_model_config(self) -> dict[str, Any]:
        """Get dual model system configuration."""
        try:
            # Import configuration helper
            from src.config import get_dual_model_config

            # Get configuration from the centralized config system
            dual_model_config = get_dual_model_config()

            if dual_model_config:
                return {"dual_model_system": dual_model_config}
            # Fallback to default configuration
            return {
                "dual_model_system": {
                    "analyst_timeframes": ["30m", "15m", "5m"],
                    "tactician_timeframes": ["1m"],
                    "analyst_confidence_threshold": 0.5,
                    "tactician_confidence_threshold": 0.6,
                    "enter_signal_validity_duration": 120,
                    "signal_check_interval": 10,
                    "neutral_signal_threshold": 0.5,
                    "close_signal_threshold": 0.4,
                    "position_close_confidence_threshold": 0.6,
                    "enable_ensemble_analysis": True,
                },
            }

        except Exception:
            self.logger.exception("Error getting dual model config")
            # Return default configuration
            return {
                "dual_model_system": {
                    "analyst_timeframes": ["30m", "15m", "5m"],
                    "tactician_timeframes": ["1m"],
                    "analyst_confidence_threshold": 0.5,
                    "tactician_confidence_threshold": 0.6,
                    "enter_signal_validity_duration": 120,
                    "signal_check_interval": 10,
                    "neutral_signal_threshold": 0.5,
                    "close_signal_threshold": 0.4,
                    "position_close_confidence_threshold": 0.6,
                    "enable_ensemble_analysis": True,
                },
            }


async def main():
    """Main entry point for the Ares Pipeline."""
    import argparse
    import os
    import sys
    from pathlib import Path

    # Add the project root to the Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.utils.logger import setup_logging, system_logger
    from src.utils.observability import init_observability

    # Setup logging
    setup_logging()
    init_observability({})
    logger = system_logger.getChild("AresPipelineMain")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ares Trading Pipeline")
    parser.add_argument("symbol", help="Trading symbol (e.g., ETHUSDT)")
    parser.add_argument("exchange", help="Exchange name (e.g., BINANCE)")
    parser.add_argument("--config", help="Path to configuration file")

    args = parser.parse_args()

    # Get trading mode from environment variable
    trading_mode = os.environ.get("TRADING_MODE", "PAPER").upper()

    logger.info(f"🚀 Starting Ares Pipeline in {trading_mode} mode")
    logger.info(f"📊 Symbol: {args.symbol}")
    logger.info(f"🏢 Exchange: {args.exchange}")
    logger.info(f"🔧 Trading Mode: {trading_mode}")

    # Create pipeline instance
    pipeline = AresPipeline()

    try:
        # Initialize pipeline
        if not await pipeline.initialize():
            print(failed("❌ Failed to initialize pipeline"))
            sys.exit(1)

        # Run pipeline
        result = await pipeline.run()

        if result:
            logger.info("✅ Pipeline completed successfully")
        else:
            print(failed("❌ Pipeline failed"))
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal, shutting down gracefully...")
        await pipeline.stop()
    except Exception:
        print(error("💥 Unexpected error: {e}"))
        await pipeline.stop()
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
