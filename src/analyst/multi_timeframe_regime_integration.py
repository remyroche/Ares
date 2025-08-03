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

from src.analyst.hmm_regime_classifier import HMMRegimeClassifier
from src.config import CONFIG
from src.training.regime_specific_tpsl_optimizer import RegimeSpecificTPSLOptimizer
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


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

        # Initialize HMM classifier (1h only)
        self.hmm_classifier = HMMRegimeClassifier(config)

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
    async def initialize(self) -> bool:
        """
        Initialize the multi-timeframe regime integration.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Multi-Timeframe Regime Integration...")

            # Initialize HMM classifier
            if not await self._initialize_hmm_classifier():
                self.logger.error("Failed to initialize HMM classifier")
                return False

            # Initialize regime-specific TP/SL optimizer
            if not await self.regime_tpsl_optimizer.initialize():
                self.logger.error(
                    "Failed to initialize regime-specific TP/SL optimizer",
                )
                return False

            self.logger.info(
                "✅ Multi-Timeframe Regime Integration initialized successfully",
            )
            return True

        except Exception as e:
            self.logger.error(
                f"❌ Failed to initialize Multi-Timeframe Regime Integration: {e}",
            )
            return False

    async def _initialize_hmm_classifier(self) -> bool:
        """
        Initialize the HMM regime classifier.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Try to load existing HMM model
            model_path = os.path.join(
                CONFIG["CHECKPOINT_DIR"],
                "analyst_models",
                "hmm_regime_classifier_1h.joblib",
            )

            if os.path.exists(model_path):
                if self.hmm_classifier.load_model(model_path):
                    self.logger.info("✅ Loaded existing HMM regime classifier")
                    return True
                self.logger.warning("Failed to load existing HMM model")

            self.logger.info(
                "HMM classifier not trained yet, will be trained when 1h data is available",
            )
            return True

        except Exception as e:
            self.logger.error(f"Error initializing HMM classifier: {e}")
            return False

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
                regime, confidence, info = self.hmm_classifier.predict_regime(data_1h)

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
            self.logger.error(f"Error in regime classification: {e}")
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
    async def get_regime_for_timeframe(
        self,
        timeframe: str,
        current_data: pd.DataFrame,
        data_1h: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Get regime information for a specific timeframe.

        This ensures that all timeframes use the same regime classification
        (from 1h) but can access it in a timeframe-specific context.

        Args:
            timeframe: Target timeframe (1m, 5m, 15m, 1h)
            current_data: Current data for the target timeframe
            data_1h: 1-hour data for regime classification

        Returns:
            Dictionary with regime information for the timeframe
        """
        try:
            # Get regime classification from 1h data
            regime, confidence, regime_info = await self.classify_regime_1h(data_1h)

            # Create timeframe-specific regime information
            timeframe_regime_info = {
                "regime": regime,
                "confidence": confidence,
                "regime_info": regime_info,
                "timeframe": timeframe,
                "strategic_timeframe": "1h",
                "regime_source": "1h_hmm_classifier",
                "timestamp": datetime.now().isoformat(),
            }

            # Add timeframe-specific adjustments if needed
            if timeframe != "1h":
                timeframe_regime_info.update(
                    {
                        "timeframe_adjustment": self._get_timeframe_adjustment(
                            timeframe,
                            regime,
                        ),
                        "propagation_method": "cached_from_1h",
                    },
                )

            self.logger.info(
                f"📊 Regime for {timeframe}: {regime} (confidence: {confidence:.2f})",
            )
            return timeframe_regime_info

        except Exception as e:
            self.logger.error(f"Error getting regime for timeframe {timeframe}: {e}")
            return {
                "regime": "SIDEWAYS_RANGE",
                "confidence": 0.5,
                "regime_info": {"method": "fallback", "error": str(e)},
                "timeframe": timeframe,
                "strategic_timeframe": "1h",
                "regime_source": "fallback",
            }

    def _get_timeframe_adjustment(self, timeframe: str, regime: str) -> dict[str, Any]:
        """
        Get timeframe-specific adjustments for regime information.

        Args:
            timeframe: Target timeframe
            regime: Current regime

        Returns:
            Dictionary with timeframe-specific adjustments
        """
        # Define timeframe-specific adjustments based on regime
        adjustments = {
            "1m": {
                "BULL_TREND": {"volatility_multiplier": 1.5, "momentum_threshold": 0.8},
                "BEAR_TREND": {"volatility_multiplier": 1.5, "momentum_threshold": 0.8},
                "SIDEWAYS_RANGE": {
                    "volatility_multiplier": 1.0,
                    "momentum_threshold": 0.5,
                },
                "SR_ZONE_ACTION": {
                    "volatility_multiplier": 1.2,
                    "momentum_threshold": 0.7,
                },
                "HIGH_IMPACT_CANDLE": {
                    "volatility_multiplier": 2.0,
                    "momentum_threshold": 0.9,
                },
            },
            "5m": {
                "BULL_TREND": {"volatility_multiplier": 1.3, "momentum_threshold": 0.7},
                "BEAR_TREND": {"volatility_multiplier": 1.3, "momentum_threshold": 0.7},
                "SIDEWAYS_RANGE": {
                    "volatility_multiplier": 1.0,
                    "momentum_threshold": 0.5,
                },
                "SR_ZONE_ACTION": {
                    "volatility_multiplier": 1.1,
                    "momentum_threshold": 0.6,
                },
                "HIGH_IMPACT_CANDLE": {
                    "volatility_multiplier": 1.8,
                    "momentum_threshold": 0.8,
                },
            },
            "15m": {
                "BULL_TREND": {"volatility_multiplier": 1.1, "momentum_threshold": 0.6},
                "BEAR_TREND": {"volatility_multiplier": 1.1, "momentum_threshold": 0.6},
                "SIDEWAYS_RANGE": {
                    "volatility_multiplier": 1.0,
                    "momentum_threshold": 0.5,
                },
                "SR_ZONE_ACTION": {
                    "volatility_multiplier": 1.05,
                    "momentum_threshold": 0.55,
                },
                "HIGH_IMPACT_CANDLE": {
                    "volatility_multiplier": 1.5,
                    "momentum_threshold": 0.7,
                },
            },
        }

        return adjustments.get(timeframe, {}).get(
            regime,
            {"volatility_multiplier": 1.0, "momentum_threshold": 0.5},
        )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime-specific optimization",
    )
    async def get_regime_specific_optimization(
        self,
        timeframe: str,
        current_data: pd.DataFrame,
        data_1h: pd.DataFrame,
        historical_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Get regime-specific optimization parameters for a timeframe.

        Args:
            timeframe: Target timeframe
            current_data: Current data for the timeframe
            data_1h: 1-hour data for regime classification
            historical_data: Historical data for optimization

        Returns:
            Dictionary with regime-specific optimization parameters
        """
        try:
            # Get regime information
            regime_info = await self.get_regime_for_timeframe(
                timeframe,
                current_data,
                data_1h,
            )

            # Get regime-specific TP/SL optimization
            tpsl_params = await self.regime_tpsl_optimizer.get_optimized_tpsl(
                data_1h,
                historical_data,
            )

            # Combine regime info with TP/SL parameters
            optimization_params = {
                **regime_info,
                **tpsl_params,
                "timeframe": timeframe,
                "optimization_timestamp": datetime.now().isoformat(),
            }

            self.logger.info(
                f"🎯 Regime-specific optimization for {timeframe}: {regime_info['regime']}",
            )
            return optimization_params

        except Exception as e:
            self.logger.error(
                f"Error getting regime-specific optimization for {timeframe}: {e}",
            )
            return {
                "regime": "SIDEWAYS_RANGE",
                "confidence": 0.5,
                "timeframe": timeframe,
                "target_pct": 0.5,
                "stop_pct": 0.3,
                "risk_reward_ratio": 1.67,
                "method": "fallback",
                "error": str(e),
            }

    async def train_hmm_classifier(self, historical_data_1h: pd.DataFrame) -> bool:
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
                self.logger.error("Invalid 1h data provided for training")
                return False

            success = self.hmm_classifier.train_classifier(historical_data_1h)

            if success:
                self.logger.info("✅ HMM regime classifier trained successfully")
                # Save the model
                self.hmm_classifier.save_model()
                return True
            self.logger.error("❌ Failed to train HMM regime classifier")
            return False

        except Exception as e:
            self.logger.error(f"Error training HMM classifier: {e}")
            return False

    def get_integration_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the multi-timeframe regime integration.

        Returns:
            Dictionary with integration statistics
        """
        return {
            "current_regime": self.current_regime,
            "regime_confidence": self.regime_confidence,
            "last_regime_update": self.last_regime_update,
            "hmm_trained": self.hmm_classifier.trained,
            "active_timeframes": self.active_timeframes,
            "strategic_timeframe": "1h",
            "regime_cache_duration_minutes": self.regime_cache_duration.total_seconds()
            / 60,
            "regime_tpsl_optimizer_stats": self.regime_tpsl_optimizer.get_regime_statistics(),
        }
