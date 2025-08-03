"""
Base pipeline framework for Ares trading bot.

This module provides the abstract base classes and common functionality
for all pipeline implementations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


@dataclass
class PipelineConfig:
    """
    Configuration for pipeline execution.

    This class contains all the configuration parameters needed to
    run a pipeline, including environment settings, component
    configurations, and execution parameters.
    """

    name: str
    symbol: str
    exchange: str
    environment: str  # "live", "backtest", "training"

    # Pipeline settings
    checkpoint_enabled: bool = True
    email_notifications: bool = True
    pid_file_enabled: bool = True
    loop_interval_seconds: int = 10
    max_retries: int = 3
    timeout_seconds: int = 3600

    # Component configurations
    data_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    risk_config: dict[str, Any] = field(default_factory=dict)
    notification_config: dict[str, Any] = field(default_factory=dict)

    # Execution parameters
    parallel_execution: bool = False
    max_workers: int = 4
    continue_on_failure: bool = False

    def validate(self) -> list[str]:
        """
        Validate the pipeline configuration.

        Returns:
            List of validation error messages
        """
        errors = []

        if not self.name:
            errors.append("Pipeline name is required")

        if not self.symbol:
            errors.append("Symbol is required")

        if not self.exchange:
            errors.append("Exchange is required")

        if self.environment not in ["live", "backtest", "training"]:
            errors.append("Environment must be 'live', 'backtest', or 'training'")

        if self.loop_interval_seconds <= 0:
            errors.append("Loop interval must be positive")

        if self.max_retries < 0:
            errors.append("Max retries must be non-negative")

        if self.timeout_seconds <= 0:
            errors.append("Timeout must be positive")

        return errors


@dataclass
class PipelineMetrics:
    """
    Metrics and statistics for pipeline execution.
    """

    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float | None = None
    stages_completed: int = 0
    stages_failed: int = 0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None

    def update_duration(self):
        """Update duration based on start and end times."""
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def get_success_rate(self) -> float:
        """Get the success rate of operations."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations

    def get_stage_success_rate(self) -> float:
        """Get the success rate of stages."""
        total_stages = self.stages_completed + self.stages_failed
        if total_stages == 0:
            return 0.0
        return self.stages_completed / total_stages


class BasePipeline:
    """
    Enhanced base pipeline with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize base pipeline with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("BasePipeline")

        # Pipeline state
        self.is_running: bool = False
        self.pipeline_results: dict[str, Any] = {}
        self.pipeline_history: list[dict[str, Any]] = []

        # Configuration
        self.pipeline_config: dict[str, Any] = self.config.get("base_pipeline", {})
        self.pipeline_interval: int = self.pipeline_config.get("pipeline_interval", 60)
        self.max_pipeline_history: int = self.pipeline_config.get(
            "max_pipeline_history",
            100,
        )
        self.enable_data_processing: bool = self.pipeline_config.get(
            "enable_data_processing",
            True,
        )
        self.enable_analysis_processing: bool = self.pipeline_config.get(
            "enable_analysis_processing",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid base pipeline configuration"),
            AttributeError: (False, "Missing required pipeline parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="base pipeline initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize base pipeline with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Base Pipeline...")

            # Load pipeline configuration
            await self._load_pipeline_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for base pipeline")
                return False

            # Initialize pipeline modules
            await self._initialize_pipeline_modules()

            self.logger.info("✅ Base Pipeline initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Base Pipeline initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline configuration loading",
    )
    async def _load_pipeline_configuration(self) -> None:
        """Load pipeline configuration."""
        try:
            # Set default pipeline parameters
            self.pipeline_config.setdefault("pipeline_interval", 60)
            self.pipeline_config.setdefault("max_pipeline_history", 100)
            self.pipeline_config.setdefault("enable_data_processing", True)
            self.pipeline_config.setdefault("enable_analysis_processing", True)
            self.pipeline_config.setdefault("enable_strategy_processing", False)
            self.pipeline_config.setdefault("enable_execution_processing", True)

            # Update configuration
            self.pipeline_interval = self.pipeline_config["pipeline_interval"]
            self.max_pipeline_history = self.pipeline_config["max_pipeline_history"]
            self.enable_data_processing = self.pipeline_config["enable_data_processing"]
            self.enable_analysis_processing = self.pipeline_config[
                "enable_analysis_processing"
            ]

            self.logger.info("Pipeline configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading pipeline configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate pipeline configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate pipeline interval
            if self.pipeline_interval <= 0:
                self.logger.error("Invalid pipeline interval")
                return False

            # Validate max pipeline history
            if self.max_pipeline_history <= 0:
                self.logger.error("Invalid max pipeline history")
                return False

            # Validate that at least one processing type is enabled
            if not any(
                [
                    self.enable_data_processing,
                    self.enable_analysis_processing,
                    self.pipeline_config.get("enable_strategy_processing", False),
                    self.pipeline_config.get("enable_execution_processing", True),
                ],
            ):
                self.logger.error("At least one processing type must be enabled")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline modules initialization",
    )
    async def _initialize_pipeline_modules(self) -> None:
        """Initialize pipeline modules."""
        try:
            # Initialize data processing module
            if self.enable_data_processing:
                await self._initialize_data_processing()

            # Initialize analysis processing module
            if self.enable_analysis_processing:
                await self._initialize_analysis_processing()

            # Initialize strategy processing module
            if self.pipeline_config.get("enable_strategy_processing", False):
                await self._initialize_strategy_processing()

            # Initialize execution processing module
            if self.pipeline_config.get("enable_execution_processing", True):
                await self._initialize_execution_processing()

            self.logger.info("Pipeline modules initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing pipeline modules: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data processing initialization",
    )
    async def _initialize_data_processing(self) -> None:
        """Initialize data processing module."""
        try:
            # Initialize data processing components
            self.data_processing_components = {
                "data_collection": True,
                "data_validation": True,
                "data_transformation": True,
                "data_storage": True,
            }

            self.logger.info("Data processing module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing data processing: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analysis processing initialization",
    )
    async def _initialize_analysis_processing(self) -> None:
        """Initialize analysis processing module."""
        try:
            # Initialize analysis processing components
            self.analysis_processing_components = {
                "technical_analysis": True,
                "fundamental_analysis": True,
                "sentiment_analysis": True,
                "risk_analysis": True,
            }

            self.logger.info("Analysis processing module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing analysis processing: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategy processing initialization",
    )
    async def _initialize_strategy_processing(self) -> None:
        """Initialize strategy processing module."""
        try:
            # Initialize strategy processing components
            self.strategy_processing_components = {
                "signal_generation": True,
                "position_sizing": True,
                "risk_management": True,
                "portfolio_optimization": True,
            }

            self.logger.info("Strategy processing module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing strategy processing: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="execution processing initialization",
    )
    async def _initialize_execution_processing(self) -> None:
        """Initialize execution processing module."""
        try:
            # Initialize execution processing components
            self.execution_processing_components = {
                "order_management": True,
                "position_management": True,
                "performance_tracking": True,
                "risk_monitoring": True,
            }

            self.logger.info("Execution processing module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing execution processing: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid pipeline parameters"),
            AttributeError: (False, "Missing pipeline components"),
            KeyError: (False, "Missing required pipeline data"),
        },
        default_return=False,
        context="pipeline execution",
    )
    async def execute_pipeline(self, input_data: dict[str, Any]) -> bool:
        """
        Execute pipeline processing.

        Args:
            input_data: Input data dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_pipeline_inputs(input_data):
                return False

            self.is_running = True
            self.logger.info("🔄 Starting pipeline processing...")

            # Perform data processing
            if self.enable_data_processing:
                data_results = await self._perform_data_processing(input_data)
                self.pipeline_results["data"] = data_results

            # Perform analysis processing
            if self.enable_analysis_processing:
                analysis_results = await self._perform_analysis_processing(input_data)
                self.pipeline_results["analysis"] = analysis_results

            # Perform strategy processing
            if self.pipeline_config.get("enable_strategy_processing", False):
                strategy_results = await self._perform_strategy_processing(input_data)
                self.pipeline_results["strategy"] = strategy_results

            # Perform execution processing
            if self.pipeline_config.get("enable_execution_processing", True):
                execution_results = await self._perform_execution_processing(input_data)
                self.pipeline_results["execution"] = execution_results

            # Store pipeline results
            await self._store_pipeline_results()

            self.is_running = False
            self.logger.info("✅ Pipeline processing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error executing pipeline: {e}")
            self.is_running = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="pipeline inputs validation",
    )
    def _validate_pipeline_inputs(self, input_data: dict[str, Any]) -> bool:
        """
        Validate pipeline inputs.

        Args:
            input_data: Input data dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required input data fields
            required_fields = ["symbol", "timestamp", "data_type"]
            for field in required_fields:
                if field not in input_data:
                    self.logger.error(f"Missing required input data field: {field}")
                    return False

            # Validate data types
            if not isinstance(input_data["symbol"], str):
                self.logger.error("Invalid symbol data type")
                return False

            if not isinstance(input_data["timestamp"], (str, datetime)):
                self.logger.error("Invalid timestamp data type")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating pipeline inputs: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data processing",
    )
    async def _perform_data_processing(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform data processing.

        Args:
            input_data: Input data dictionary

        Returns:
            Dict[str, Any]: Data processing results
        """
        try:
            results = {}

            # Perform data collection
            if self.data_processing_components.get("data_collection", False):
                results["data_collection"] = self._perform_data_collection(input_data)

            # Perform data validation
            if self.data_processing_components.get("data_validation", False):
                results["data_validation"] = self._perform_data_validation(input_data)

            # Perform data transformation
            if self.data_processing_components.get("data_transformation", False):
                results["data_transformation"] = self._perform_data_transformation(
                    input_data,
                )

            # Perform data storage
            if self.data_processing_components.get("data_storage", False):
                results["data_storage"] = self._perform_data_storage(input_data)

            self.logger.info("Data processing completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing data processing: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analysis processing",
    )
    async def _perform_analysis_processing(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform analysis processing.

        Args:
            input_data: Input data dictionary

        Returns:
            Dict[str, Any]: Analysis processing results
        """
        try:
            results = {}

            # Perform technical analysis
            if self.analysis_processing_components.get("technical_analysis", False):
                results["technical_analysis"] = self._perform_technical_analysis(
                    input_data,
                )

            # Perform fundamental analysis
            if self.analysis_processing_components.get("fundamental_analysis", False):
                results["fundamental_analysis"] = self._perform_fundamental_analysis(
                    input_data,
                )

            # Perform sentiment analysis
            if self.analysis_processing_components.get("sentiment_analysis", False):
                results["sentiment_analysis"] = self._perform_sentiment_analysis(
                    input_data,
                )

            # Perform risk analysis
            if self.analysis_processing_components.get("risk_analysis", False):
                results["risk_analysis"] = self._perform_risk_analysis(input_data)

            self.logger.info("Analysis processing completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing analysis processing: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategy processing",
    )
    async def _perform_strategy_processing(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform strategy processing.

        Args:
            input_data: Input data dictionary

        Returns:
            Dict[str, Any]: Strategy processing results
        """
        try:
            results = {}

            # Perform signal generation
            if self.strategy_processing_components.get("signal_generation", False):
                results["signal_generation"] = self._perform_signal_generation(
                    input_data,
                )

            # Perform position sizing
            if self.strategy_processing_components.get("position_sizing", False):
                results["position_sizing"] = self._perform_position_sizing(input_data)

            # Perform risk management
            if self.strategy_processing_components.get("risk_management", False):
                results["risk_management"] = self._perform_risk_management(input_data)

            # Perform portfolio optimization
            if self.strategy_processing_components.get("portfolio_optimization", False):
                results["portfolio_optimization"] = (
                    self._perform_portfolio_optimization(input_data)
                )

            self.logger.info("Strategy processing completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing strategy processing: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="execution processing",
    )
    async def _perform_execution_processing(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform execution processing.

        Args:
            input_data: Input data dictionary

        Returns:
            Dict[str, Any]: Execution processing results
        """
        try:
            results = {}

            # Perform order management
            if self.execution_processing_components.get("order_management", False):
                results["order_management"] = self._perform_order_management(input_data)

            # Perform position management
            if self.execution_processing_components.get("position_management", False):
                results["position_management"] = self._perform_position_management(
                    input_data,
                )

            # Perform performance tracking
            if self.execution_processing_components.get("performance_tracking", False):
                results["performance_tracking"] = self._perform_performance_tracking(
                    input_data,
                )

            # Perform risk monitoring
            if self.execution_processing_components.get("risk_monitoring", False):
                results["risk_monitoring"] = self._perform_risk_monitoring(input_data)

            self.logger.info("Execution processing completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing execution processing: {e}")
            return {}

    # Data processing methods
    def _perform_data_collection(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform data collection."""
        try:
            # Simulate data collection
            symbol = input_data.get("symbol", "UNKNOWN")
            data_type = input_data.get("data_type", "klines")

            return {
                "symbol": symbol,
                "data_type": data_type,
                "records_collected": 1000,
                "collection_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing data collection: {e}")
            return {}

    def _perform_data_validation(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform data validation."""
        try:
            # Simulate data validation
            return {
                "validation_passed": True,
                "invalid_records": 0,
                "validation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing data validation: {e}")
            return {}

    def _perform_data_transformation(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform data transformation."""
        try:
            # Simulate data transformation
            return {
                "transformation_completed": True,
                "transformed_records": 1000,
                "transformation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing data transformation: {e}")
            return {}

    def _perform_data_storage(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform data storage."""
        try:
            # Simulate data storage
            return {
                "storage_completed": True,
                "stored_records": 1000,
                "storage_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing data storage: {e}")
            return {}

    # Analysis processing methods
    def _perform_technical_analysis(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform technical analysis."""
        try:
            # Simulate technical analysis
            return {
                "technical_indicators": ["SMA", "EMA", "RSI", "MACD"],
                "analysis_completed": True,
                "analysis_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing technical analysis: {e}")
            return {}

    def _perform_fundamental_analysis(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform fundamental analysis."""
        try:
            # Simulate fundamental analysis
            return {
                "fundamental_metrics": ["P/E", "P/B", "ROE", "Debt/Equity"],
                "analysis_completed": True,
                "analysis_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing fundamental analysis: {e}")
            return {}

    def _perform_sentiment_analysis(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform sentiment analysis."""
        try:
            # Simulate sentiment analysis
            return {
                "sentiment_score": 0.65,
                "sentiment_label": "Positive",
                "analysis_completed": True,
                "analysis_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing sentiment analysis: {e}")
            return {}

    def _perform_risk_analysis(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform risk analysis."""
        try:
            # Simulate risk analysis
            return {
                "risk_score": 0.35,
                "risk_level": "Medium",
                "analysis_completed": True,
                "analysis_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing risk analysis: {e}")
            return {}

    # Strategy processing methods
    def _perform_signal_generation(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform signal generation."""
        try:
            # Simulate signal generation
            return {
                "signal": "BUY",
                "confidence": 0.75,
                "signal_strength": "Strong",
                "generation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing signal generation: {e}")
            return {}

    def _perform_position_sizing(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform position sizing."""
        try:
            # Simulate position sizing
            return {
                "position_size": 0.1,
                "leverage": 2.0,
                "risk_per_trade": 0.02,
                "sizing_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing position sizing: {e}")
            return {}

    def _perform_risk_management(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform risk management."""
        try:
            # Simulate risk management
            return {
                "stop_loss": 0.05,
                "take_profit": 0.10,
                "max_drawdown": 0.20,
                "risk_management_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing risk management: {e}")
            return {}

    def _perform_portfolio_optimization(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform portfolio optimization."""
        try:
            # Simulate portfolio optimization
            return {
                "optimization_completed": True,
                "optimal_allocation": {"BTC": 0.6, "ETH": 0.4},
                "expected_return": 0.15,
                "optimization_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing portfolio optimization: {e}")
            return {}

    # Execution processing methods
    def _perform_order_management(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform order management."""
        try:
            # Simulate order management
            return {
                "order_placed": True,
                "order_id": "12345",
                "order_status": "FILLED",
                "order_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing order management: {e}")
            return {}

    def _perform_position_management(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform position management."""
        try:
            # Simulate position management
            return {
                "position_opened": True,
                "position_id": "67890",
                "position_size": 0.1,
                "position_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing position management: {e}")
            return {}

    def _perform_performance_tracking(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform performance tracking."""
        try:
            # Simulate performance tracking
            return {
                "pnl": 150.0,
                "return_pct": 0.015,
                "sharpe_ratio": 1.2,
                "tracking_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing performance tracking: {e}")
            return {}

    def _perform_risk_monitoring(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Perform risk monitoring."""
        try:
            # Simulate risk monitoring
            return {
                "current_risk": 0.25,
                "risk_limit": 0.30,
                "risk_status": "OK",
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing risk monitoring: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline results storage",
    )
    async def _store_pipeline_results(self) -> None:
        """Store pipeline results."""
        try:
            # Add timestamp
            self.pipeline_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.pipeline_history.append(self.pipeline_results.copy())

            # Limit history size
            if len(self.pipeline_history) > self.max_pipeline_history:
                self.pipeline_history.pop(0)

            self.logger.info("Pipeline results stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing pipeline results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline results getting",
    )
    def get_pipeline_results(
        self,
        pipeline_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get pipeline results.

        Args:
            pipeline_type: Optional pipeline type filter

        Returns:
            Dict[str, Any]: Pipeline results
        """
        try:
            if pipeline_type:
                return self.pipeline_results.get(pipeline_type, {})
            return self.pipeline_results.copy()

        except Exception as e:
            self.logger.error(f"Error getting pipeline results: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline history getting",
    )
    def get_pipeline_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get pipeline history.

        Args:
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: Pipeline history
        """
        try:
            history = self.pipeline_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception as e:
            self.logger.error(f"Error getting pipeline history: {e}")
            return []

    def get_pipeline_status(self) -> dict[str, Any]:
        """
        Get pipeline status information.

        Returns:
            Dict[str, Any]: Pipeline status
        """
        return {
            "is_running": self.is_running,
            "pipeline_interval": self.pipeline_interval,
            "max_pipeline_history": self.max_pipeline_history,
            "enable_data_processing": self.enable_data_processing,
            "enable_analysis_processing": self.enable_analysis_processing,
            "enable_strategy_processing": self.pipeline_config.get(
                "enable_strategy_processing",
                False,
            ),
            "enable_execution_processing": self.pipeline_config.get(
                "enable_execution_processing",
                True,
            ),
            "pipeline_history_count": len(self.pipeline_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="base pipeline cleanup",
    )
    async def stop(self) -> None:
        """Stop the base pipeline."""
        self.logger.info("🛑 Stopping Base Pipeline...")

        try:
            # Stop pipeline
            self.is_running = False

            # Clear results
            self.pipeline_results.clear()

            # Clear history
            self.pipeline_history.clear()

            self.logger.info("✅ Base Pipeline stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping base pipeline: {e}")


# Global base pipeline instance
base_pipeline: BasePipeline | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="base pipeline setup",
)
async def setup_base_pipeline(
    config: dict[str, Any] | None = None,
) -> BasePipeline | None:
    """
    Setup global base pipeline.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[BasePipeline]: Global base pipeline instance
    """
    try:
        global base_pipeline

        if config is None:
            config = {
                "base_pipeline": {
                    "pipeline_interval": 60,
                    "max_pipeline_history": 100,
                    "enable_data_processing": True,
                    "enable_analysis_processing": True,
                    "enable_strategy_processing": False,
                    "enable_execution_processing": True,
                },
            }

        # Create base pipeline
        base_pipeline = BasePipeline(config)

        # Initialize base pipeline
        success = await base_pipeline.initialize()
        if success:
            return base_pipeline
        return None

    except Exception as e:
        print(f"Error setting up base pipeline: {e}")
        return None
