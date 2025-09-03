from __future__ import annotations

from typing import Any

from src.core.decorators import handles_errors
from src.utils.logger import system_logger


class StageRegistry:
    """Stage registry with comprehensive error handling and type safety."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize stage registry with enhanced type safety.

        Args:
            config: Configuration dictionary

        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("StageRegistry")

        # Stage registry state
        self.is_registered: bool = False
        self.stage_results: dict[str, Any] = {}
        self.stage_history: list[dict[str, Any]] = []

        # Configuration
        self.stage_config: dict[str, Any] = self.config.get("stage_registry", {})
        self.stage_interval: int = self.stage_config.get("stage_interval", 3600)
        self.max_stage_history: int = self.stage_config.get("max_stage_history", 100)
        self.enable_stage_registration: bool = self.stage_config.get(
            "enable_stage_registration",
            True,
        )
        self.enable_stage_validation: bool = self.stage_config.get(
            "enable_stage_validation",
            True,
        )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid stage registry configuration"),
            AttributeError: (False, "Missing required stage registry parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="stage registry initialization",
    )
    async def initialize(self) -> bool:
        """Initialize stage registry with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise

        """
        try:
            self.logger.info("Initializing Stage Registry...")

            # Load stage configuration
            await self._load_stage_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for stage registry")
                return False

            # Initialize stage modules
            await self._initialize_stage_modules()

            self.logger.info("✅ Stage Registry initialization completed successfully")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Stage Registry initialization failed: {e}")
            return False

    @handles_errors(fallback=None)
    async def _load_stage_configuration(self) -> None:
        """Load stage configuration."""
        try:
            # Set default stage parameters
            self.stage_config.setdefault("stage_interval", 3600)
            self.stage_config.setdefault("max_stage_history", 100)
            self.stage_config.setdefault("enable_stage_registration", True)
            self.stage_config.setdefault("enable_stage_validation", True)
            self.stage_config.setdefault("enable_stage_execution", True)
            self.stage_config.setdefault("enable_stage_monitoring", True)

            # Update configuration
            self.stage_interval = self.stage_config["stage_interval"]
            self.max_stage_history = self.stage_config["max_stage_history"]
            self.enable_stage_registration = self.stage_config[
                "enable_stage_registration"
            ]
            self.enable_stage_validation = self.stage_config["enable_stage_validation"]

            self.logger.info("Stage configuration loaded successfully")

        except Exception as e:
            self.logger.exception(f"Error loading stage configuration: {e}")
