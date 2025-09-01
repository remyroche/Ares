# src/analyst/multi_timeframe_regime_integration.py

"""
Multi-Timeframe Regime Integration

This module integrates the HMM regime classifier (which operates only on 1h timeframe)
with the multi-timeframe system. It ensures that:

1. Regime classification is done ONLY on 1h timeframe (strategic level)
2. The regime information is propagated to all other timeframes
3. Each timeframe can use the regime information for its specific predictions
4. The regime information is consistent across all timeframes

This follows the principle that there should be only ONE regime classification
based on the 1-hour timeframe, which represents the macro trend.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.config import CONFIG
from src.training.steps.analyst_training_components.regime_specific_tpsl_optimizer import (
    RegimeSpecificTPSLOptimizer,
)
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
)


class MultiTimeframeRegimeIntegration:
    """
    Integrates HMM regime classification with multi-timeframe system.

    This class ensures that:
    - Regime classification is done only on 1h timeframe
    - Regime information is propagated to all timeframes
    - Each timeframe can access consistent regime information
    - Regime-specific optimizations are available across timeframes
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the multi-timeframe regime integration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MultiTimeframeRegimeIntegration")

        # Initialize Unified regime classifier (1h only)
        self.regime_classifier = UnifiedRegimeClassifier(config)

        # Initialize regime-specific TP/SL optimizer
        self.regime_tpsl_optimizer = RegimeSpecificTPSLOptimizer(config)

        # Timeframe configuration
        self.timeframes = CONFIG.get("TIMEFRAMES", {})
        self.timeframe_set = CONFIG.get("DEFAULT_TIMEFRAME_SET", "intraday")
        self.active_timeframes = CONFIG.get("TIMEFRAME_SETS", {}).get(
            self.timeframe_set,
            [],
        )

        # Regime cache
        self.current_regime: str | None = None
        self.regime_confidence: float = 0.0
        self.regime_info: dict[str, Any] = {}
        self.last_regime_update: datetime | None = None
        self.regime_cache_duration = timedelta(
            minutes=15,
        )  # Cache regime for 15 minutes

        # Regime propagation settings
        self.regime_propagation_config = config.get(
            "multi_timeframe_regime_integration",
            {},
        )
        self.enable_regime_propagation = self.regime_propagation_config.get(
            "enable_propagation",
            True,
        )
        self.regime_smoothing_window = self.regime_propagation_config.get(
            "smoothing_window",
            5,
        )

        self.logger.info("🚀 Initialized MultiTimeframeRegimeIntegration")
        self.logger.info(f"📊 Active timeframes: {self.active_timeframes}")
        self.logger.info("⏰ Strategic timeframe: 1h (regime classification only)")

    @handle_specific_errors(
        error_handlers={
            ValueError: (
                False,
                "Invalid multi-timeframe regime integration configuration",
            ),
            AttributeError: (False, "Missing required integration parameters"),
        },
        default_return=False,
        context="multi-timeframe regime integration initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime classification",
    )
    async def classify_regime_1h(
        self,
        data_1h: pd.DataFrame,
    ) -> tuple[str, float, dict[str, Any]]:
        """
        Classify market regime using 1h timeframe data only.

        Args:
            data_1h: 1-hour timeframe data

        Returns:
            Tuple of (regime, confidence, additional_info)
        """
        try:
            # Validate that we have 1h data
            if not self._validate_1h_data(data_1h):
                self.logger.warning(
                    "Invalid 1h data provided for regime classification",
                )
                return (
                    "SIDEWAYS_RANGE",
                    0.5,
                    {"method": "fallback", "reason": "invalid_data"},
                )

            # Check if we need to update regime (cache management)
            if self._should_update_regime():
                regime, confidence, info = self.regime_classifier.predict_regime(
                    data_1h,
                )

                # Update cache
                self.current_regime = regime
                self.regime_confidence = confidence
                self.regime_info = info
                self.last_regime_update = datetime.now()

                self.logger.info(
                    f"🔄 Updated regime classification: {regime} (confidence: {confidence:.2f})",
                )
            else:
                self.logger.info(
                    f"📋 Using cached regime: {self.current_regime} (confidence: {self.regime_confidence:.2f})",
                )

            return self.current_regime, self.regime_confidence, self.regime_info

        except Exception as e:
            self.print(error("Error in regime classification: {e}"))
            return "SIDEWAYS_RANGE", 0.5, {"method": "fallback", "error": str(e)}

    def _validate_1h_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data is from 1h timeframe.

        Args:
            data: DataFrame to validate

        Returns:
            bool: True if valid 1h data, False otherwise
        """
        if data.empty:
            return False

        if not isinstance(data.index, pd.DatetimeIndex):
            return False

        if len(data) < 2:
            return False

        # Check timeframe
        time_diff = data.index[1] - data.index[0]
        hours_diff = time_diff.total_seconds() / 3600

        # Allow tolerance (0.8 to 1.2 hours)
        return 0.8 <= hours_diff <= 1.2

    def _should_update_regime(self) -> bool:
        """
        Check if regime should be updated based on cache duration.

        Returns:
            bool: True if regime should be updated, False otherwise
        """
        if self.last_regime_update is None:
            return True

        time_since_update = datetime.now() - self.last_regime_update
        return time_since_update > self.regime_cache_duration

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime propagation",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime-specific optimization",
    )
    async def train_regime_classifier(self, historical_data_1h: pd.DataFrame) -> bool:
        """
        Train the HMM classifier using 1h historical data.

        Args:
            historical_data_1h: Historical 1h data for training

        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("🎓 Training HMM regime classifier with 1h data...")

            if not self._validate_1h_data(historical_data_1h):
                self.print(invalid("Invalid 1h data provided for training"))
                return False

            success = await self.regime_classifier.train_complete_system(
                historical_data_1h,
            )

            if success:
                self.logger.info("✅ HMM regime classifier trained successfully")
                # Save the model
                # Model saving is handled automatically by UnifiedRegimeClassifier
                return True
            self.print(failed("❌ Failed to train HMM regime classifier"))
            return False

        except Exception:
            self.print(error("Error training HMM classifier: {e}"))
            return False
