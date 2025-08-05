# src/analyst/ml_target_generator.py

import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class MLTargetGenerator:
    """
    ML Target Generator for creating training targets from market data.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the ML Target Generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MLTargetGenerator")

        # Target configuration
        self.target_config = config.get("ml_target_generator", {})
        self.lookback_window = self.target_config.get("lookback_window", 730)  # days
        self.target_threshold = self.target_config.get("target_threshold", 0.02)  # 2%
        self.stop_loss_threshold = self.target_config.get(
            "stop_loss_threshold",
            -0.01,
        )  # -1%
        self.take_profit_threshold = self.target_config.get(
            "take_profit_threshold",
            0.03,
        )  # 3%

        self.logger.info("Initializing ML Target Generator...")
        self.logger.info(f"📊 Using lookback window: {self.lookback_window} days")

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError),
        default_return=False,
        context="target generator initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the target generator.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid target generator configuration")
                return False

            self.logger.info("Target configuration loaded successfully")
            self.logger.info("Target parameters initialized")
            self.logger.info("Configuration validation successful")
            self.logger.info(
                "✅ ML Target Generator initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize target generator: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """
        Validate the target generator configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Check required parameters
            if self.lookback_window <= 0:
                self.logger.error("Lookback window must be positive")
                return False

            if self.target_threshold <= 0:
                self.logger.error("Target threshold must be positive")
                return False

            if self.stop_loss_threshold >= 0:
                self.logger.error("Stop loss threshold must be negative")
                return False

            if self.take_profit_threshold <= 0:
                self.logger.error("Take profit threshold must be positive")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError),
        default_return=pd.DataFrame(),
        context="target generation",
    )
    async def generate_targets(
        self,
        data: pd.DataFrame,
        current_price: float,
    ) -> pd.DataFrame:
        """
        Generate ML targets from market data.

        Args:
            data: DataFrame with market data
            current_price: Current market price

        Returns:
            pd.DataFrame: DataFrame with generated targets
        """
        try:
            self.logger.info("Generating ML targets...")

            if data.empty:
                self.logger.warning("Empty data provided, returning empty DataFrame")
                return pd.DataFrame()

            # Create a copy of the data
            result_data = data.copy()

            # Generate basic targets
            result_data = self._generate_basic_targets(result_data)

            # Generate advanced targets
            result_data = self._generate_advanced_targets(result_data, current_price)

            # Clean up any NaN values
            result_data = result_data.dropna()

            self.logger.info("✅ ML targets generated successfully")
            return result_data

        except Exception as e:
            self.logger.error(f"Failed to generate targets: {e}")
            return data.copy()

    def _generate_basic_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate basic binary targets based on price movement.

        Args:
            data: DataFrame with market data

        Returns:
            pd.DataFrame: DataFrame with basic targets
        """
        try:
            if "close" not in data.columns:
                self.logger.warning(
                    "No 'close' column found, skipping basic target generation",
                )
                return data

            # Calculate price changes
            data["price_change"] = data["close"].pct_change()

            # Generate binary target based on price movement
            data["target"] = (data["price_change"] > self.target_threshold).astype(int)

            # Generate multi-class target
            data["target_multi"] = 0  # Hold
            data.loc[
                data["price_change"] > self.take_profit_threshold,
                "target_multi",
            ] = 1  # Buy
            data.loc[
                data["price_change"] < self.stop_loss_threshold,
                "target_multi",
            ] = -1  # Sell

            return data

        except Exception as e:
            self.logger.error(f"Failed to generate basic targets: {e}")
            return data

    def _generate_advanced_targets(
        self,
        data: pd.DataFrame,
        current_price: float,
    ) -> pd.DataFrame:
        """
        Generate advanced targets including take profit and stop loss levels.

        Args:
            data: DataFrame with market data
            current_price: Current market price

        Returns:
            pd.DataFrame: DataFrame with advanced targets
        """
        try:
            if "close" not in data.columns:
                self.logger.warning(
                    "No 'close' column found, skipping advanced target generation",
                )
                return data

            # Calculate volatility-based targets
            returns = data["close"].pct_change()
            volatility = returns.rolling(window=20).std()

            # Dynamic thresholds based on volatility
            data["dynamic_tp_threshold"] = (
                volatility * 2
            )  # 2x volatility for take profit
            data["dynamic_sl_threshold"] = (
                -volatility * 1.5
            )  # 1.5x volatility for stop loss

            # Generate take profit and stop loss targets
            data["tp_target"] = data["close"] * (1 + data["dynamic_tp_threshold"])
            data["sl_target"] = data["close"] * (1 + data["dynamic_sl_threshold"])

            # Generate momentum-based targets
            data["momentum"] = data["close"].pct_change(5)  # 5-period momentum
            data["momentum_target"] = (data["momentum"] > 0).astype(int)

            # Generate trend-based targets
            sma_20 = data["close"].rolling(window=20).mean()
            sma_50 = data["close"].rolling(window=50).mean()
            data["trend_target"] = (sma_20 > sma_50).astype(int)

            # Generate volatility regime targets
            data["volatility_regime"] = 0  # Normal
            data.loc[volatility > volatility.quantile(0.75), "volatility_regime"] = (
                1  # High
            )
            data.loc[
                volatility < volatility.quantile(0.25),
                "volatility_regime",
            ] = -1  # Low

            return data

        except Exception as e:
            self.logger.error(f"Failed to generate advanced targets: {e}")
            return data

    def get_target_statistics(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Get statistics about the generated targets.

        Args:
            data: DataFrame with targets

        Returns:
            Dict[str, Any]: Target statistics
        """
        try:
            stats = {}

            if "target" in data.columns:
                stats["binary_target_distribution"] = (
                    data["target"].value_counts().to_dict()
                )
                stats["binary_target_ratio"] = data["target"].mean()

            if "target_multi" in data.columns:
                stats["multi_target_distribution"] = (
                    data["target_multi"].value_counts().to_dict()
                )

            if "momentum_target" in data.columns:
                stats["momentum_target_ratio"] = data["momentum_target"].mean()

            if "trend_target" in data.columns:
                stats["trend_target_ratio"] = data["trend_target"].mean()

            if "volatility_regime" in data.columns:
                stats["volatility_regime_distribution"] = (
                    data["volatility_regime"].value_counts().to_dict()
                )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get target statistics: {e}")
            return {}

    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        try:
            self.logger.info("Cleaning up ML Target Generator...")
            # Add any cleanup logic here if needed
            self.logger.info("✅ ML Target Generator cleanup completed")
        except Exception as e:
            self.logger.error(f"Failed to cleanup target generator: {e}")
