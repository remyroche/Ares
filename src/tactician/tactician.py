"""Tactician module for trading strategy execution."""

from datetime import datetime
from typing import Any

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import failed, invalid, missing

class Tactician:
    """
    Refactored Tactician component with modular architecture.
    This module orchestrates the tactics pipeline using specialized managers.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize refactored tactician.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Tactician")

        # Tactician state
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.tactics_results: dict[str, Any] = {}

        # Configuration
        self.tactician_config: dict[str, Any] = self.config.get("tactician", {})
        self.tactics_interval: int = self.tactician_config.get("tactics_interval", 30)
        self.max_history: int = self.tactician_config.get("max_history", 100)

        # Component managers (will be initialized)
        self.tactics_orchestrator = None
        self.position_sizer = None
        self.leverage_sizer = None
        self.position_division_strategy = None

        # Enhanced predictions from supervisor
        self.enable_enhanced_predictions: bool = self.tactician_config.get(
            "enable_enhanced_predictions",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid tactician configuration"),
            AttributeError: (False, "Missing required tactician parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="tactician initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="component managers initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate tactician configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate required configuration sections
            required_sections = ["tactician", "tactics_orchestrator"]

            for section in required_sections:
                if section not in self.config:
                    self.logger.error(
                        f"Missing required configuration section: {section}",
                    )
                    return False

            # Validate tactician specific settings
            if self.tactics_interval <= 0:
                self.logger.error(invalid("Invalid tactics_interval configuration"))
                return False

            if self.max_history <= 0:
                self.logger.error(invalid("Invalid max_history configuration"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"Configuration validation failed: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid tactics parameters"),
            AttributeError: (False, "Missing tactics components"),
            KeyError: (False, "Missing required tactics data"),
        },
        default_return=False,
        context="tactics execution",
    )
    async def execute_tactics(
        self, tactics_input: dict[str, Any]
    ) -> bool:
        """
        Execute the complete tactics pipeline.

        Args:
            tactics_input: Tactics input parameters

        Returns:
            bool: True if tactics successful, False otherwise
        """
        try:
            self.logger.info("🚀 Starting tactics pipeline execution...")

            # Validate tactics input
            if not self._validate_tactics_input(tactics_input):
                return False

            # Execute tactics using the orchestrator
            success = await self.tactics_orchestrator.execute_tactics(tactics_input)

            if success:
                self.logger.info("✅ Tactics pipeline completed successfully")
                await self._store_tactics_results(tactics_input)
            else:
                self.logger.error(failed("❌ Tactics pipeline failed"))

            return success

        except Exception as e:
            self.logger.error(failed(f"❌ Tactics execution failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="tactics input validation",
    )
    def _validate_tactics_input(self, tactics_input: dict[str, Any]) -> bool:
        """
        Validate tactics input parameters.

        Args:
            tactics_input: Tactics input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["symbol", "exchange", "timeframe", "current_price"]

            for field in required_fields:
                if field not in tactics_input:
                    self.logger.error(missing(f"Missing required tactics input field: {field}"))
                    return False

            # Validate specific field values
            if tactics_input.get("current_price", 0) <= 0:
                self.logger.error(invalid("Invalid current_price value"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"Tactics input validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactics results storage",
    )
    async def _store_tactics_results(self, tactics_input: dict[str, Any]) -> None:
        """
        Store tactics results for later retrieval.

        Args:
            tactics_input: Tactics input parameters
        """
        try:
            # Get results from orchestrator
            self.tactics_results = self.tactics_orchestrator.get_tactics_results()

            # Add to history
            history_entry = {
                "timestamp": datetime.now(),
                "tactics_input": tactics_input, "tactics_results": self.tactics_results.copy(),
            }

            self.history.append(history_entry)

            # Limit history size
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]

            self.logger.info(
                f"📁 Stored tactics results (history: {len(self.history)} entries)",
            )

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to store tactics results: {e}"))

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Tactician run failed"),
        },
        default_return=False,
        context="tactician run",
    )
    async def run(self) -> bool:
        """
        Run the tactician.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("🚀 Starting Tactician...")
            self.is_running = True

            # Update status
            self.status = {
                "is_running": True, "start_time": datetime.now(),
                "component_count": 4,  # tactics_orchestrator = position_sizer, leverage_sizer = position_division_strategy
            }

            self.logger.info("✅ Tactician run completed successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Tactician run failed: {e}"))
            return False

    # Enhanced predictions are now handled by the supervisor
    # No local methods needed

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tactician stop",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tactician cleanup",
    )
@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="tactician setup",
)