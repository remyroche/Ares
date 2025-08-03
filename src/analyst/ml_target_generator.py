from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class MLTargetGenerator:
    """
    Enhanced ML target generator with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ML target generator with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MLTargetGenerator")

        # Target generation state
        self.current_targets: dict[str, Any] | None = None
        self.target_history: list[dict[str, Any]] = []
        self.last_target_update: datetime | None = None

        # Configuration
        self.target_config: dict[str, Any] = self.config.get("ml_target_generator", {})
        self.min_samples: int = self.target_config.get("min_samples", 100)
        # Import the centralized lookback window function
        from src.config import get_lookback_window
        self.lookback_window: int = get_lookback_window()
        self.confidence_threshold: float = self.target_config.get(
            "confidence_threshold",
            0.6,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ML target generator configuration"),
            AttributeError: (False, "Missing required target parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="ML target generator initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize ML target generator with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing ML Target Generator...")

            # Load target configuration
            await self._load_target_configuration()

            # Initialize target parameters
            await self._initialize_target_parameters()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for ML target generator")
                return False

            self.logger.info(
                "✅ ML Target Generator initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ ML Target Generator initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="target configuration loading",
    )
    async def _load_target_configuration(self) -> None:
        """Load target generation configuration."""
        try:
            # Set default target parameters
            self.target_config.setdefault("min_samples", 100)
            self.target_config.setdefault("confidence_threshold", 0.6)
            self.target_config.setdefault("min_tp_distance", 0.005)
            self.target_config.setdefault("max_tp_distance", 0.05)
            self.target_config.setdefault("min_sl_distance", 0.002)
            self.target_config.setdefault("max_sl_distance", 0.03)

            # Use centralized lookback window
            from src.config import get_lookback_window
            lookback_days = get_lookback_window()
            self.logger.info(f"📊 Using lookback window: {lookback_days} days")

            self.logger.info("Target configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading target configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="target parameters initialization",
    )
    async def _initialize_target_parameters(self) -> None:
        """Initialize target generation parameters."""
        try:
            # Initialize target parameters
            self.min_samples = self.target_config["min_samples"]
            # Use centralized lookback window
            from src.config import get_lookback_window
            self.lookback_window = get_lookback_window()
            self.confidence_threshold = self.target_config["confidence_threshold"]

            self.logger.info("Target parameters initialized")

        except Exception as e:
            self.logger.error(f"Error initializing target parameters: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate ML target generator configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_keys = ["min_samples", "confidence_threshold"]
            for key in required_keys:
                if key not in self.target_config:
                    self.logger.error(
                        f"Missing required target configuration key: {key}",
                    )
                    return False

            # Validate parameter ranges
            if self.min_samples < 10:
                self.logger.error("min_samples must be at least 10")
                return False

            # Validate lookback window (now handled centrally)
            from src.config import get_lookback_window
            lookback_days = get_lookback_window()
            if lookback_days < 5:
                self.logger.error("lookback_window must be at least 5")
                return False

            if not 0 < self.confidence_threshold < 1:
                self.logger.error("confidence_threshold must be between 0 and 1")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ConnectionError: (None, "Failed to connect to data source"),
            TimeoutError: (None, "Target generation timed out"),
            ValueError: (None, "Invalid market data"),
        },
        default_return=None,
        context="target generation",
    )
    async def generate_targets(
        self,
        data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        Generate ML targets with enhanced error handling.

        Args:
            data: Historical market data
            current_price: Current market price

        Returns:
            Optional[Dict[str, Any]]: Generated targets or None if failed
        """
        try:
            if data.empty:
                self.logger.error("Empty data provided for target generation")
                return None

            self.logger.info("Generating ML targets...")

            # Validate data structure
            if not self._validate_data_structure(data):
                self.logger.error("Invalid data structure for target generation")
                return None

            # Generate target components
            take_profit_targets = await self._generate_take_profit_targets(
                data,
                current_price,
            )
            stop_loss_targets = await self._generate_stop_loss_targets(
                data,
                current_price,
            )

            # Calculate confidence scores
            tp_confidence = self._calculate_target_confidence(take_profit_targets, data)
            sl_confidence = self._calculate_target_confidence(stop_loss_targets, data)

            # Generate target results
            targets = {
                "take_profit_targets": take_profit_targets,
                "stop_loss_targets": stop_loss_targets,
                "tp_confidence": tp_confidence,
                "sl_confidence": sl_confidence,
                "current_price": current_price,
                "generation_time": datetime.now(),
                "valid_until": datetime.now() + timedelta(minutes=15),
            }

            # Validate targets
            if not self._validate_targets(targets):
                self.logger.error("Generated targets validation failed")
                return None

            # Update state
            self.current_targets = targets
            self.target_history.append(targets)
            self.last_target_update = datetime.now()

            self.logger.info("✅ ML targets generated successfully")
            return targets

        except Exception as e:
            self.logger.error(f"Error generating ML targets: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="data structure validation",
    )
    def _validate_data_structure(self, data: pd.DataFrame) -> bool:
        """
        Validate data structure for target generation.

        Args:
            data: DataFrame to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]

            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check for sufficient data
            if len(data) < self.lookback_window:
                self.logger.warning(
                    f"Insufficient data: {len(data)} < {self.lookback_window}",
                )
                return False

            # Check for valid price data
            if (data[["open", "high", "low", "close"]] <= 0).any().any():
                self.logger.error("Invalid price data (non-positive values)")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating data structure: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="take profit target generation",
    )
    async def _generate_take_profit_targets(
        self,
        data: pd.DataFrame,
        current_price: float,
    ) -> list[dict[str, Any]] | None:
        """
        Generate take profit targets.

        Args:
            data: Market data
            current_price: Current price

        Returns:
            Optional[List[Dict[str, Any]]]: Take profit targets or None
        """
        try:
            targets = []

            # Use recent data for analysis
            recent_data = data.tail(self.lookback_window)

            # Calculate volatility-based targets
            volatility = recent_data["close"].pct_change().std()

            # Generate multiple TP levels
            tp_levels = [
                {"distance": 0.01, "weight": 0.4},  # 1% - 40% weight
                {"distance": 0.02, "weight": 0.3},  # 2% - 30% weight
                {"distance": 0.03, "weight": 0.2},  # 3% - 20% weight
                {"distance": 0.05, "weight": 0.1},  # 5% - 10% weight
            ]

            for level in tp_levels:
                # Adjust distance based on volatility
                adjusted_distance = level["distance"] * (1 + volatility * 10)

                # Ensure distance is within bounds
                adjusted_distance = max(
                    self.target_config["min_tp_distance"],
                    min(adjusted_distance, self.target_config["max_tp_distance"]),
                )

                target_price = current_price * (1 + adjusted_distance)

                targets.append(
                    {
                        "price": target_price,
                        "distance": adjusted_distance,
                        "weight": level["weight"],
                        "type": "take_profit",
                        "confidence": self._calculate_level_confidence(
                            adjusted_distance,
                            volatility,
                        ),
                    },
                )

            return targets

        except Exception as e:
            self.logger.error(f"Error generating take profit targets: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stop loss target generation",
    )
    async def _generate_stop_loss_targets(
        self,
        data: pd.DataFrame,
        current_price: float,
    ) -> list[dict[str, Any]] | None:
        """
        Generate stop loss targets.

        Args:
            data: Market data
            current_price: Current price

        Returns:
            Optional[List[Dict[str, Any]]]: Stop loss targets or None
        """
        try:
            targets = []

            # Use recent data for analysis
            recent_data = data.tail(self.lookback_window)

            # Calculate volatility-based targets
            volatility = recent_data["close"].pct_change().std()

            # Generate multiple SL levels
            sl_levels = [
                {"distance": 0.005, "weight": 0.5},  # 0.5% - 50% weight
                {"distance": 0.01, "weight": 0.3},  # 1% - 30% weight
                {"distance": 0.02, "weight": 0.2},  # 2% - 20% weight
            ]

            for level in sl_levels:
                # Adjust distance based on volatility
                adjusted_distance = level["distance"] * (1 + volatility * 5)

                # Ensure distance is within bounds
                adjusted_distance = max(
                    self.target_config["min_sl_distance"],
                    min(adjusted_distance, self.target_config["max_sl_distance"]),
                )

                target_price = current_price * (1 - adjusted_distance)

                targets.append(
                    {
                        "price": target_price,
                        "distance": adjusted_distance,
                        "weight": level["weight"],
                        "type": "stop_loss",
                        "confidence": self._calculate_level_confidence(
                            adjusted_distance,
                            volatility,
                        ),
                    },
                )

            return targets

        except Exception as e:
            self.logger.error(f"Error generating stop loss targets: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=0.5,
        context="level confidence calculation",
    )
    def _calculate_level_confidence(self, distance: float, volatility: float) -> float:
        """
        Calculate confidence for a target level.

        Args:
            distance: Target distance
            volatility: Market volatility

        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            # Base confidence
            confidence = 0.5

            # Adjust based on distance appropriateness
            if 0.005 <= distance <= 0.02:  # Optimal range
                confidence += 0.2
            elif distance > 0.05:  # Too far
                confidence -= 0.1

            # Adjust based on volatility
            if volatility < 0.01:  # Low volatility
                confidence += 0.1
            elif volatility > 0.03:  # High volatility
                confidence -= 0.1

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating level confidence: {e}")
            return 0.5

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=0.0,
        context="target confidence calculation",
    )
    def _calculate_target_confidence(
        self,
        targets: list[dict[str, Any]],
        data: pd.DataFrame,
    ) -> float:
        """
        Calculate overall confidence for target set.

        Args:
            targets: List of targets
            data: Market data

        Returns:
            float: Overall confidence score
        """
        try:
            if not targets:
                return 0.0

            # Calculate average confidence
            avg_confidence = np.mean([target["confidence"] for target in targets])

            # Adjust based on data quality
            data_quality = min(len(data) / self.min_samples, 1.0)

            # Weighted combination
            final_confidence = (avg_confidence * 0.7) + (data_quality * 0.3)

            return max(0.0, min(1.0, final_confidence))

        except Exception as e:
            self.logger.error(f"Error calculating target confidence: {e}")
            return 0.0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="target validation",
    )
    def _validate_targets(self, targets: dict[str, Any]) -> bool:
        """
        Validate generated targets.

        Args:
            targets: Targets to validate

        Returns:
            bool: True if targets are valid, False otherwise
        """
        try:
            required_keys = [
                "take_profit_targets",
                "stop_loss_targets",
                "tp_confidence",
                "sl_confidence",
            ]
            for key in required_keys:
                if key not in targets:
                    self.logger.error(f"Missing required target key: {key}")
                    return False

            # Validate confidence scores
            tp_confidence = targets.get("tp_confidence", 0)
            sl_confidence = targets.get("sl_confidence", 0)

            if (
                tp_confidence < self.confidence_threshold
                or sl_confidence < self.confidence_threshold
            ):
                self.logger.warning(
                    f"Target confidence too low: TP={tp_confidence}, SL={sl_confidence}",
                )
                return False

            # Validate target prices
            current_price = targets.get("current_price", 0)
            if current_price <= 0:
                self.logger.error("Invalid current price")
                return False

            # Validate TP targets
            for tp in targets.get("take_profit_targets", []):
                if tp["price"] <= current_price:
                    self.logger.error("Take profit target must be above current price")
                    return False

            # Validate SL targets
            for sl in targets.get("stop_loss_targets", []):
                if sl["price"] >= current_price:
                    self.logger.error("Stop loss target must be below current price")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating targets: {e}")
            return False

    def get_current_targets(self) -> dict[str, Any] | None:
        """
        Get current ML targets.

        Returns:
            Optional[Dict[str, Any]]: Current targets or None
        """
        return self.current_targets.copy() if self.current_targets else None

    def get_target_history(self) -> list[dict[str, Any]]:
        """
        Get target generation history.

        Returns:
            List[Dict[str, Any]]: Target history
        """
        return self.target_history.copy()

    def get_last_target_update(self) -> datetime | None:
        """
        Get last target update time.

        Returns:
            Optional[datetime]: Last update time or None
        """
        return self.last_target_update

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML target generator cleanup",
    )
    async def stop(self) -> None:
        """Stop the ML target generator component."""
        self.logger.info("🛑 Stopping ML Target Generator...")

        try:
            # Save target history
            if self.target_history:
                self.logger.info(f"Saving {len(self.target_history)} target records")

            # Clear current state
            self.current_targets = None
            self.last_target_update = None

            self.logger.info("✅ ML Target Generator stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping ML target generator: {e}")
