# src/analyst/multi_timeframe_regime_integration.py

"""
Multi-Timeframe Regime Integration

This module integrates the Meta-Labeling System for regime context (dominant meta-label)
with the multi-timeframe system. It ensures that:

1. Regime context selection is performed on the strategic timeframe (default 1h)
2. The selected label context is propagated to other timeframes
3. Each timeframe can access consistent regime context for its specific predictions
4. The regime context is consistent across all timeframes
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

from src.analyst.meta_labeling_system import MetaLabelingSystem
from src.analyst.regime_runtime import get_current_regime_info
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
    Integrates meta-label regime context with multi-timeframe system.

    This class ensures that:
    - Regime context selection is done on the strategic timeframe (default 1h)
    - Context is propagated to all timeframes
    - Each timeframe can access consistent context information
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
        self.print = self.logger.info

        # Initialize Meta-Labeling system (strategic level)
        self.meta_labeling_system = MetaLabelingSystem(config)

        # Initialize regime-specific TP/SL optimizer
        self.regime_tpsl_optimizer = RegimeSpecificTPSLOptimizer(config)

        # Timeframe configuration
        self.timeframes = CONFIG.get("TIMEFRAMES", {})
        self.timeframe_set = CONFIG.get("DEFAULT_TIMEFRAME_SET", "intraday")
        self.active_timeframes = CONFIG.get("TIMEFRAME_SETS", {}).get(
            self.timeframe_set,
            [],
        )

        # Regime context selection settings
        self.regime_propagation_config = config.get(
            "multi_timeframe_regime_integration",
            {},
        )
        self.analysis_timeframe: str = self.regime_propagation_config.get(
            "analysis_timeframe",
            "1h",
        )
        self.candidate_labels: list[str] = self.regime_propagation_config.get(
            "candidate_labels",
            [
                "STRONG_TREND_CONTINUATION",
                "EXHAUSTION_REVERSAL",
                "RANGE_MEAN_REVERSION",
                "BREAKOUT_SUCCESS",
                "BREAKOUT_FAILURE",
                "MOMENTUM_IGNITION",
                "VOLATILITY_COMPRESSION",
                "VOLATILITY_EXPANSION",
                "SR_TOUCH",
                "SR_BOUNCE",
                "SR_BREAK",
                "IGNITION_BAR",
            ],
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
        self.logger.info(f"⏰ Strategic timeframe: {self.analysis_timeframe} (regime context)")

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

            # Initialize Meta-Labeling system
            if not await self._initialize_meta_label_system():
                self.print(failed("Failed to initialize Meta-Labeling system"))
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
            self.logger.exception(
                f"❌ Failed to initialize Multi-Timeframe Regime Integration: {e}",
            )
            return False

    async def _initialize_meta_label_system(self) -> bool:
        """
        Initialize the MetaLabelingSystem.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            ok = await self.meta_labeling_system.initialize()
            if ok:
                self.logger.info("✅ Meta-Labeling system initialized for regime context")
                return True
            self.print(failed("Meta-Labeling system failed to initialize"))
            return False
        except Exception:
            self.print(initialization_error("Error initializing Meta-Labeling system: {e}"))
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
        Select dominant meta-label regime context using strategic timeframe data.

        Args:
            data_1h: 1-hour timeframe data (used as strategic timeframe by default)

        Returns:
            Tuple of (regime_label, confidence, additional_info)
        """
        try:
            # Validate that we have 1h-like data
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
                labels = await self.meta_labeling_system.generate_analyst_labels(
                    price_data=data_1h,
                    volume_data=data_1h,
                    timeframe=self.analysis_timeframe,
                )
                # Use HMM composite cluster as regime; fetch runtime intensities and probabilities
                cluster_id = int(labels.get("HMM_COMPOSITE_CLUSTER", -1))
                # Fetch runtime calibrated signals
                try:
                    rr = get_current_regime_info(self.exchange, self.symbol, self.analysis_timeframe)
                except Exception:
                    rr = {"cluster_id": cluster_id, "intensities": {}, "p_emerge": {}, "exit_hazard": None}
                confidence = float(rr.get("intensities", {}).get(cluster_id, 0.0)) if cluster_id >= 0 else 0.0
                info = {
                    "method": "hmm_composite",
                    "timeframe": self.analysis_timeframe,
                    "cluster_id": cluster_id,
                    "intensity": confidence,
                    "intensities": rr.get("intensities", {}),
                    "p_emerge": rr.get("p_emerge", {}),
                    "exit_hazard": rr.get("exit_hazard", None),
                }

                # Update cache
                self.current_regime = f"CLUSTER_{cluster_id}" if cluster_id >= 0 else "SIDEWAYS_RANGE"
                self.regime_confidence = confidence
                self.regime_info = info
                self.last_regime_update = datetime.now()

                self.logger.info(
                    f"🔄 Updated regime context: CLUSTER_{cluster_id} (intensity: {confidence:.2f})",
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
    async def get_regime_for_timeframe(
        self,
        timeframe: str,
        current_data: pd.DataFrame,
        data_1h: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Get regime information for a specific timeframe.

        This ensures that all timeframes use the same regime context
        (selected from strategic timeframe) but can access it in a timeframe-specific context.

        Args:
            timeframe: Target timeframe (1m, 5m, 15m, 1h)
            current_data: Current data for the target timeframe
            data_1h: 1-hour data for regime context selection

        Returns:
            Dictionary with regime information for the timeframe
        """
        try:
            # Get regime classification from strategic timeframe data
            regime, confidence, regime_info = await self.classify_regime_1h(data_1h)

            # Create timeframe-specific regime information
            timeframe_regime_info = {
                "regime": regime,
                "confidence": confidence,
                "regime_info": regime_info,
                "timeframe": timeframe,
                "strategic_timeframe": self.analysis_timeframe,
                "regime_source": "meta_labeling_system",
                "timestamp": datetime.now().isoformat(),
            }

            # Add timeframe-specific adjustments if needed
            if timeframe != self.analysis_timeframe:
                timeframe_regime_info.update(
                    {
                        "timeframe_adjustment": self._get_timeframe_adjustment(
                            timeframe,
                            regime,
                        ),
                        "propagation_method": "cached_from_strategic",
                    },
                )

            self.logger.info(
                f"📊 Regime for {timeframe}: {regime} (confidence: {confidence:.2f})",
            )
            return timeframe_regime_info

        except Exception as e:
            self.logger.exception(
                f"Error getting regime for timeframe {timeframe}: {e}",
            )
            return {
                "regime": "SIDEWAYS_RANGE",
                "confidence": 0.5,
                "regime_info": {"method": "fallback", "error": str(e)},
                "timeframe": timeframe,
                "strategic_timeframe": self.analysis_timeframe,
                "regime_source": "fallback",
            }

    def _get_timeframe_adjustment(self, timeframe: str, regime: str) -> dict[str, Any]:
        """
        Get timeframe-specific adjustments for regime information.

        Args:
            timeframe: Target timeframe
            regime: Current regime (dominant meta-label)

        Returns:
            Dictionary with timeframe-specific adjustments
        """
        # Define timeframe-specific adjustments based on meta-label regime
        adjustments = {
            "1m": {
                "STRONG_TREND_CONTINUATION": {"volatility_multiplier": 1.5, "momentum_threshold": 0.85},
                "EXHAUSTION_REVERSAL": {"volatility_multiplier": 1.4, "momentum_threshold": 0.6},
                "RANGE_MEAN_REVERSION": {"volatility_multiplier": 0.9, "momentum_threshold": 0.45},
                "BREAKOUT_SUCCESS": {"volatility_multiplier": 1.6, "momentum_threshold": 0.9},
                "BREAKOUT_FAILURE": {"volatility_multiplier": 1.2, "momentum_threshold": 0.55},
                "MOMENTUM_IGNITION": {"volatility_multiplier": 1.8, "momentum_threshold": 0.95},
                "VOLATILITY_COMPRESSION": {"volatility_multiplier": 0.8, "momentum_threshold": 0.4},
                "VOLATILITY_EXPANSION": {"volatility_multiplier": 1.7, "momentum_threshold": 0.85},
                "SR_TOUCH": {"volatility_multiplier": 1.1, "momentum_threshold": 0.6},
                "SR_BOUNCE": {"volatility_multiplier": 1.2, "momentum_threshold": 0.65},
                "SR_BREAK": {"volatility_multiplier": 1.4, "momentum_threshold": 0.8},
                "IGNITION_BAR": {"volatility_multiplier": 2.0, "momentum_threshold": 0.95},
            },
            "5m": {
                "STRONG_TREND_CONTINUATION": {"volatility_multiplier": 1.3, "momentum_threshold": 0.75},
                "EXHAUSTION_REVERSAL": {"volatility_multiplier": 1.25, "momentum_threshold": 0.55},
                "RANGE_MEAN_REVERSION": {"volatility_multiplier": 0.95, "momentum_threshold": 0.5},
                "BREAKOUT_SUCCESS": {"volatility_multiplier": 1.4, "momentum_threshold": 0.8},
                "BREAKOUT_FAILURE": {"volatility_multiplier": 1.15, "momentum_threshold": 0.5},
                "MOMENTUM_IGNITION": {"volatility_multiplier": 1.6, "momentum_threshold": 0.9},
                "VOLATILITY_COMPRESSION": {"volatility_multiplier": 0.85, "momentum_threshold": 0.45},
                "VOLATILITY_EXPANSION": {"volatility_multiplier": 1.5, "momentum_threshold": 0.8},
                "SR_TOUCH": {"volatility_multiplier": 1.05, "momentum_threshold": 0.55},
                "SR_BOUNCE": {"volatility_multiplier": 1.1, "momentum_threshold": 0.6},
                "SR_BREAK": {"volatility_multiplier": 1.3, "momentum_threshold": 0.75},
                "IGNITION_BAR": {"volatility_multiplier": 1.8, "momentum_threshold": 0.9},
            },
            "15m": {
                "STRONG_TREND_CONTINUATION": {"volatility_multiplier": 1.15, "momentum_threshold": 0.65},
                "EXHAUSTION_REVERSAL": {"volatility_multiplier": 1.1, "momentum_threshold": 0.5},
                "RANGE_MEAN_REVERSION": {"volatility_multiplier": 1.0, "momentum_threshold": 0.5},
                "BREAKOUT_SUCCESS": {"volatility_multiplier": 1.25, "momentum_threshold": 0.7},
                "BREAKOUT_FAILURE": {"volatility_multiplier": 1.05, "momentum_threshold": 0.5},
                "MOMENTUM_IGNITION": {"volatility_multiplier": 1.5, "momentum_threshold": 0.85},
                "VOLATILITY_COMPRESSION": {"volatility_multiplier": 0.9, "momentum_threshold": 0.5},
                "VOLATILITY_EXPANSION": {"volatility_multiplier": 1.35, "momentum_threshold": 0.7},
                "SR_TOUCH": {"volatility_multiplier": 1.0, "momentum_threshold": 0.55},
                "SR_BOUNCE": {"volatility_multiplier": 1.05, "momentum_threshold": 0.6},
                "SR_BREAK": {"volatility_multiplier": 1.2, "momentum_threshold": 0.7},
                "IGNITION_BAR": {"volatility_multiplier": 1.6, "momentum_threshold": 0.85},
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
            data_1h: 1-hour data for regime context selection
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
            self.logger.exception(
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

    async def train_regime_classifier(self, historical_data_1h: pd.DataFrame) -> bool:
        """
        Deprecated: HMM regime classifier training.
        The meta-labeling system does not require training here.
        """
        try:
            self.logger.info(
                "HMM regime classifier is deprecated; using Meta-Labeling system for regime context",
            )
            return True
        except Exception:
            self.print(error("Error in deprecated training stub: {e}"))
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
            "meta_label_system_initialized": getattr(self.meta_labeling_system, "is_initialized", False),
            "candidate_labels": self.candidate_labels,
            "strategic_timeframe": self.analysis_timeframe,
            "regime_cache_duration_minutes": self.regime_cache_duration.total_seconds() / 60,
            "regime_tpsl_optimizer_stats": self.regime_tpsl_optimizer.get_regime_statistics(),
        }
