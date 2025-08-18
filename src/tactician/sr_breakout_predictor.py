# src/tactician/sr_breakout_predictor.py

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import os  # Added for model loading

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    invalid,
    missing,
    warning,
)


class SRBreakoutPredictor:
    """
    SR Breakout Predictor responsible for predicting support/resistance breakouts.
    This module handles all SR breakout prediction logic and feature engineering.
    Centralized S/R detection using multiple methods:
    - Fractal analysis for swing highs/lows
    - Volume-weighted price levels
    - Traditional pivot points (fallback)
    - ATR-based activation ranges
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize SR breakout predictor.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("SRBreakoutPredictor")

        # SR predictor state
        self.is_initialized: bool = False
        self.sr_predictions: dict[str, Any] = {}

        # Configuration
        self.sr_config: dict[str, Any] = self.config.get("sr_breakout_predictor", {})
        self.enable_sr_breakout_tactics: bool = self.sr_config.get(
            "enable_sr_breakout_tactics",
            True,
        )
        self.sr_proximity_threshold: float = self.sr_config.get(
            "sr_proximity_threshold",
            0.02,
        )
        self.breakout_confidence_threshold: float = self.sr_config.get(
            "breakout_confidence_threshold",
            0.7,
        )

        # Enhanced S/R Detection Configuration
        self.enhanced_sr_config: dict[str, Any] = self.sr_config.get(
            "enhanced_sr_detection", {}
        )
        self.enable_fractal_analysis: bool = self.enhanced_sr_config.get(
            "enable_fractal_analysis", True
        )
        self.enable_volume_weighted_levels: bool = self.enhanced_sr_config.get(
            "enable_volume_weighted_levels", True
        )
        self.enable_atr_based_activation: bool = self.enhanced_sr_config.get(
            "enable_atr_based_activation", True
        )

        # Fractal Analysis Parameters
        self.fractal_config: dict[str, Any] = self.enhanced_sr_config.get(
            "fractal_analysis", {}
        )
        self.fractal_lookback_periods: int = self.fractal_config.get(
            "lookback_periods", 5
        )
        self.fractal_min_swing_strength: float = self.fractal_config.get(
            "min_swing_strength", 0.6
        )
        self.fractal_volume_threshold: float = self.fractal_config.get(
            "volume_threshold", 1.5
        )

        # Volume-Weighted Level Parameters
        self.vw_config: dict[str, Any] = self.enhanced_sr_config.get(
            "volume_weighted_levels", {}
        )
        self.vw_window_size: int = self.vw_config.get("window_size", 100)
        self.vw_price_bins: int = self.vw_config.get("price_bins", 50)
        self.vw_min_volume_ratio: float = self.vw_config.get("min_volume_ratio", 0.1)

        # ATR-Based Activation Parameters
        self.atr_config: dict[str, Any] = self.enhanced_sr_config.get(
            "atr_based_activation", {}
        )
        self.atr_period: int = self.atr_config.get("atr_period", 14)
        self.atr_multiplier: float = self.atr_config.get("atr_multiplier", 1.5)
        self.atr_fallback_multiplier: float = self.atr_config.get(
            "fallback_multiplier", 0.03
        )

        # LM Model Selection Configuration
        self.lm_config: dict[str, Any] = self.sr_config.get("lm_model_selection", {})
        self.enable_specialist_models: bool = self.lm_config.get(
            "enable_specialist_models", True
        )
        self.sr_proximity_trigger_base: float = self.lm_config.get(
            "sr_proximity_trigger_base", 0.006
        )  # 0.6% base proximity
        self.sr_proximity_trigger_min: float = self.lm_config.get(
            "sr_proximity_trigger_min", 0.003
        )  # 0.3% minimum
        self.sr_proximity_trigger_max: float = self.lm_config.get(
            "sr_proximity_trigger_max", 0.01
        )  # 1.0% maximum

        # Single unified model for S/R outcome prediction
        self.sr_outcome_model_type: str = self.lm_config.get(
            "sr_outcome_model_type", "unified_transformer"
        )
        self.sr_outcome_threshold: float = self.lm_config.get(
            "sr_outcome_threshold", 0.6
        )

        # S/R outcome classes
        self.sr_outcome_classes = {"breakout": 0, "rebounce": 1, "consolidation": 2}

        # Model confidence thresholds for each outcome
        self.outcome_confidence_thresholds = {
            "breakout": self.lm_config.get("breakout_confidence_threshold", 0.65),
            "rebounce": self.lm_config.get("rebounce_confidence_threshold", 0.65),
            "consolidation": self.lm_config.get(
                "consolidation_confidence_threshold", 0.6
            ),
        }

        # Specialist model types for different S/R scenarios
        self.specialist_model_types = {
            "breakout": ["breakout_lgbm", "breakout_transformer", "breakout_cnn"],
            "rebounce": ["rebounce_lgbm", "rebounce_transformer", "rebounce_cnn"],
            "consolidation": ["consolidation_lgbm", "consolidation_transformer"],
            "default": ["generalist_lgbm", "generalist_transformer"],
        }

        # Unified regime classifier for S/R levels
        self.regime_classifier: UnifiedRegimeClassifier | None = None

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid SR breakout predictor configuration"),
            AttributeError: (
                False,
                "Missing required SR breakout predictor parameters",
            ),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="SR breakout predictor initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize SR breakout predictor with enhanced S/R detection capabilities.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing enhanced SR breakout predictor...")

            # Initialize unified regime classifier
            self.regime_classifier = UnifiedRegimeClassifier(
                self.config,
                "UNKNOWN",
                "UNKNOWN",
            )

            # Validate enhanced S/R detection configuration
            if not self._validate_enhanced_sr_config():
                self.logger.error("❌ Invalid enhanced S/R detection configuration")
                return False

            self.is_initialized = True
            self.logger.info(
                "✅ Enhanced SR breakout predictor initialized successfully"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing SR breakout predictor: {e}")
            return False

    def _validate_enhanced_sr_config(self) -> bool:
        """Validate enhanced S/R detection configuration."""
        try:
            # Validate fractal analysis parameters
            if self.enable_fractal_analysis:
                if self.fractal_lookback_periods < 3:
                    self.logger.warning(
                        "⚠️ Fractal lookback periods too small, setting to 3"
                    )
                    self.fractal_lookback_periods = 3
                if (
                    self.fractal_min_swing_strength < 0.1
                    or self.fractal_min_swing_strength > 1.0
                ):
                    self.logger.warning(
                        "⚠️ Invalid fractal swing strength, setting to 0.6"
                    )
                    self.fractal_min_swing_strength = 0.6

            # Validate volume-weighted parameters
            if self.enable_volume_weighted_levels:
                if self.vw_window_size < 20:
                    self.logger.warning(
                        "⚠️ Volume-weighted window size too small, setting to 20"
                    )
                    self.vw_window_size = 20
                if self.vw_price_bins < 10:
                    self.logger.warning(
                        "⚠️ Volume-weighted price bins too small, setting to 10"
                    )
                    self.vw_price_bins = 10

            # Validate ATR parameters
            if self.enable_atr_based_activation:
                if self.atr_period < 5:
                    self.logger.warning("⚠️ ATR period too small, setting to 5")
                    self.atr_period = 5
                if self.atr_multiplier < 0.1:
                    self.logger.warning("⚠️ ATR multiplier too small, setting to 0.1")
                    self.atr_multiplier = 0.1

            return True

        except Exception as e:
            self.logger.error(f"❌ Error validating enhanced S/R config: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="enhanced S/R level detection",
    )
    def detect_enhanced_sr_levels(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """
        Detect enhanced S/R levels using multiple methods:
        1. Fractal analysis for swing highs/lows
        2. Volume-weighted price levels
        3. Traditional pivot points (fallback)

        Args:
            price_data: OHLCV price data

        Returns:
            Dictionary containing enhanced S/R levels with strength metrics
        """
        try:
            if not self.is_initialized:
                self.logger.error("❌ SR breakout predictor not initialized")
                return {"support_levels": [], "resistance_levels": []}

            enhanced_levels = {
                "support_levels": [],
                "resistance_levels": [],
                "level_metadata": {},
            }

            # 1. Fractal Analysis for Swing Highs/Lows
            if self.enable_fractal_analysis:
                fractal_levels = self._detect_fractal_swing_levels(price_data)
                enhanced_levels["support_levels"].extend(
                    fractal_levels.get("support_levels", [])
                )
                enhanced_levels["resistance_levels"].extend(
                    fractal_levels.get("resistance_levels", [])
                )
                enhanced_levels["level_metadata"]["fractal"] = fractal_levels.get(
                    "metadata", {}
                )

            # 2. Volume-Weighted Price Levels
            if self.enable_volume_weighted_levels:
                vw_levels = self._detect_volume_weighted_levels(price_data)
                enhanced_levels["support_levels"].extend(
                    vw_levels.get("support_levels", [])
                )
                enhanced_levels["resistance_levels"].extend(
                    vw_levels.get("resistance_levels", [])
                )
                enhanced_levels["level_metadata"]["volume_weighted"] = vw_levels.get(
                    "metadata", {}
                )

            # 3. Traditional Pivot Points (fallback)
            pivot_levels = self._detect_pivot_point_levels(price_data)
            enhanced_levels["support_levels"].extend(
                pivot_levels.get("support_levels", [])
            )
            enhanced_levels["resistance_levels"].extend(
                pivot_levels.get("resistance_levels", [])
            )
            enhanced_levels["level_metadata"]["pivot_points"] = pivot_levels.get(
                "metadata", {}
            )

            # 4. Consolidate and deduplicate levels
            consolidated_levels = self._consolidate_sr_levels(enhanced_levels)

            # 5. Calculate ATR-based activation ranges
            if self.enable_atr_based_activation:
                consolidated_levels = self._add_atr_based_activation_ranges(
                    price_data, consolidated_levels
                )

            self.logger.info(
                f"✅ Detected {len(consolidated_levels['support_levels'])} support and {len(consolidated_levels['resistance_levels'])} resistance levels"
            )

            return consolidated_levels

        except Exception as e:
            self.logger.error(f"❌ Error detecting enhanced S/R levels: {e}")
            return {"support_levels": [], "resistance_levels": []}

    def _detect_fractal_swing_levels(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """
        Detect swing highs and lows using fractal analysis.

        Args:
            price_data: OHLCV price data

        Returns:
            Dictionary containing fractal swing levels
        """
        try:
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            close = price_data["close"].astype(float)
            volume = price_data["volume"].astype(float)

            swing_highs = []
            swing_lows = []
            metadata = {
                "swing_highs_count": 0,
                "swing_lows_count": 0,
                "avg_swing_strength": 0.0,
            }

            # Detect swing highs
            for i in range(
                self.fractal_lookback_periods, len(high) - self.fractal_lookback_periods
            ):
                # Check if current high is a swing high
                left_window = high.iloc[i - self.fractal_lookback_periods : i]
                right_window = high.iloc[i + 1 : i + self.fractal_lookback_periods + 1]
                current_high = high.iloc[i]

                # Fractal condition: current high is higher than surrounding highs
                if (
                    current_high > left_window.max()
                    and current_high > right_window.max()
                ):
                    # Calculate swing strength based on volume and price movement
                    price_movement = current_high - min(
                        left_window.min(), right_window.min()
                    )
                    avg_volume = volume.iloc[
                        i - self.fractal_lookback_periods : i
                        + self.fractal_lookback_periods
                        + 1
                    ].mean()
                    current_volume = volume.iloc[i]
                    volume_ratio = (
                        current_volume / avg_volume if avg_volume > 0 else 1.0
                    )

                    # Normalize price movement
                    price_range = high.max() - low.min()
                    normalized_movement = (
                        price_movement / price_range if price_range > 0 else 0
                    )

                    # Calculate swing strength (0.0 to 1.0)
                    swing_strength = min(
                        (
                            normalized_movement * 0.6
                            + min(volume_ratio / self.fractal_volume_threshold, 1.0)
                            * 0.4
                        ),
                        1.0,
                    )

                    if swing_strength >= self.fractal_min_swing_strength:
                        swing_highs.append(
                            {
                                "price": float(current_high),
                                "strength": float(swing_strength),
                                "timestamp": price_data.index[i],
                                "volume_ratio": float(volume_ratio),
                                "price_movement": float(price_movement),
                                "type": "fractal_swing_high",
                            }
                        )

            # Detect swing lows
            for i in range(
                self.fractal_lookback_periods, len(low) - self.fractal_lookback_periods
            ):
                # Check if current low is a swing low
                left_window = low.iloc[i - self.fractal_lookback_periods : i]
                right_window = low.iloc[i + 1 : i + self.fractal_lookback_periods + 1]
                current_low = low.iloc[i]

                # Fractal condition: current low is lower than surrounding lows
                if current_low < left_window.min() and current_low < right_window.min():
                    # Calculate swing strength based on volume and price movement
                    price_movement = (
                        max(left_window.max(), right_window.max()) - current_low
                    )
                    avg_volume = volume.iloc[
                        i - self.fractal_lookback_periods : i
                        + self.fractal_lookback_periods
                        + 1
                    ].mean()
                    current_volume = volume.iloc[i]
                    volume_ratio = (
                        current_volume / avg_volume if avg_volume > 0 else 1.0
                    )

                    # Normalize price movement
                    price_range = high.max() - low.min()
                    normalized_movement = (
                        price_movement / price_range if price_range > 0 else 0
                    )

                    # Calculate swing strength (0.0 to 1.0)
                    swing_strength = min(
                        (
                            normalized_movement * 0.6
                            + min(volume_ratio / self.fractal_volume_threshold, 1.0)
                            * 0.4
                        ),
                        1.0,
                    )

                    if swing_strength >= self.fractal_min_swing_strength:
                        swing_lows.append(
                            {
                                "price": float(current_low),
                                "strength": float(swing_strength),
                                "timestamp": price_data.index[i],
                                "volume_ratio": float(volume_ratio),
                                "price_movement": float(price_movement),
                                "type": "fractal_swing_low",
                            }
                        )

            # Classify as support/resistance based on current price
            current_price = close.iloc[-1]
            support_levels = []
            resistance_levels = []

            for swing in swing_lows:
                if swing["price"] < current_price:
                    support_levels.append(swing)
                else:
                    resistance_levels.append(swing)

            for swing in swing_highs:
                if swing["price"] < current_price:
                    support_levels.append(swing)
                else:
                    resistance_levels.append(swing)

            # Update metadata
            metadata["swing_highs_count"] = len(swing_highs)
            metadata["swing_lows_count"] = len(swing_lows)
            all_strengths = [s["strength"] for s in swing_highs + swing_lows]
            metadata["avg_swing_strength"] = (
                float(np.mean(all_strengths)) if all_strengths else 0.0
            )

            self.logger.info(
                f"🔍 Fractal analysis: {len(swing_highs)} swing highs, {len(swing_lows)} swing lows"
            )

            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "metadata": metadata,
            }

        except Exception as e:
            self.logger.error(f"❌ Error in fractal swing detection: {e}")
            return {"support_levels": [], "resistance_levels": [], "metadata": {}}

    def _detect_volume_weighted_levels(
        self, price_data: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Detect volume-weighted price levels using volume profile analysis.

        Args:
            price_data: OHLCV price data

        Returns:
            Dictionary containing volume-weighted levels
        """
        try:
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            close = price_data["close"].astype(float)
            volume = price_data["volume"].astype(float)

            # Use the last vw_window_size periods
            window_data = price_data.tail(self.vw_window_size)
            window_high = window_data["high"].astype(float)
            window_low = window_data["low"].astype(float)
            window_close = window_data["close"].astype(float)
            window_volume = window_data["volume"].astype(float)

            # Create price bins
            price_min = window_low.min()
            price_max = window_high.max()
            price_range = price_max - price_min

            if price_range <= 0:
                return {"support_levels": [], "resistance_levels": [], "metadata": {}}

            bin_size = price_range / self.vw_price_bins
            volume_profile = np.zeros(self.vw_price_bins)

            # Calculate volume profile
            for i in range(len(window_data)):
                price = window_close.iloc[i]
                vol = window_volume.iloc[i]

                # Determine which bin this price falls into
                bin_index = int((price - price_min) / bin_size)
                bin_index = max(0, min(bin_index, self.vw_price_bins - 1))

                volume_profile[bin_index] += vol

            # Find significant volume nodes
            total_volume = volume_profile.sum()
            if total_volume <= 0:
                return {"support_levels": [], "resistance_levels": [], "metadata": {}}

            # Calculate volume ratios and find peaks
            volume_ratios = volume_profile / total_volume
            avg_volume_ratio = volume_ratios.mean()
            threshold = avg_volume_ratio * self.vw_min_volume_ratio

            # Find local maxima in volume profile
            significant_levels = []
            for i in range(1, len(volume_ratios) - 1):
                if (
                    volume_ratios[i] > threshold
                    and volume_ratios[i] > volume_ratios[i - 1]
                    and volume_ratios[i] > volume_ratios[i + 1]
                ):
                    # Calculate price level for this bin
                    price_level = price_min + (i + 0.5) * bin_size
                    volume_ratio = volume_ratios[i]

                    # Calculate strength based on volume ratio
                    strength = min(volume_ratio / avg_volume_ratio, 2.0) / 2.0

                    significant_levels.append(
                        {
                            "price": float(price_level),
                            "strength": float(strength),
                            "volume_ratio": float(volume_ratio),
                            "type": "volume_weighted",
                        }
                    )

            # Classify as support/resistance based on current price
            current_price = close.iloc[-1]
            support_levels = []
            resistance_levels = []

            for level in significant_levels:
                if level["price"] < current_price:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)

            metadata = {
                "total_volume": float(total_volume),
                "avg_volume_ratio": float(avg_volume_ratio),
                "significant_levels_count": len(significant_levels),
                "price_range": float(price_range),
                "bin_size": float(bin_size),
            }

            self.logger.info(
                f"📊 Volume-weighted analysis: {len(significant_levels)} significant levels"
            )

            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "metadata": metadata,
            }

        except Exception as e:
            self.logger.error(f"❌ Error in volume-weighted level detection: {e}")
            return {"support_levels": [], "resistance_levels": [], "metadata": {}}

    def _detect_pivot_point_levels(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """
        Detect traditional pivot point levels using the existing regime classifier.

        Args:
            price_data: OHLCV price data

        Returns:
            Dictionary containing pivot point levels
        """
        try:
            if not self.regime_classifier:
                return {"support_levels": [], "resistance_levels": [], "metadata": {}}

            # Use the last 24 periods for pivot calculation
            window_data = price_data.tail(24)
            if len(window_data) < 5:
                return {"support_levels": [], "resistance_levels": [], "metadata": {}}

            # Calculate pivot points using the regime classifier
            pivots = self.regime_classifier._calculate_rolling_pivots(window_data)

            support_levels = []
            resistance_levels = []

            # Extract support levels
            for level_name in ["s1", "s2"]:
                if pivots[level_name] > 0:
                    support_levels.append(
                        {
                            "price": float(pivots[level_name]),
                            "strength": float(
                                pivots["strengths"][level_name]["strength"]
                            ),
                            "touches": int(pivots["strengths"][level_name]["touches"]),
                            "volume": float(pivots["strengths"][level_name]["volume"]),
                            "age": int(pivots["strengths"][level_name]["age"]),
                            "type": "pivot_point",
                        }
                    )

            # Extract resistance levels
            for level_name in ["r1", "r2"]:
                if pivots[level_name] > 0:
                    resistance_levels.append(
                        {
                            "price": float(pivots[level_name]),
                            "strength": float(
                                pivots["strengths"][level_name]["strength"]
                            ),
                            "touches": int(pivots["strengths"][level_name]["touches"]),
                            "volume": float(pivots["strengths"][level_name]["volume"]),
                            "age": int(pivots["strengths"][level_name]["age"]),
                            "type": "pivot_point",
                        }
                    )

            metadata = {
                "pivot": float(pivots["pivot"]),
                "support_count": len(support_levels),
                "resistance_count": len(resistance_levels),
            }

            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "metadata": metadata,
            }

        except Exception as e:
            self.logger.error(f"❌ Error in pivot point detection: {e}")
            return {"support_levels": [], "resistance_levels": [], "metadata": {}}

    def _consolidate_sr_levels(self, enhanced_levels: dict[str, Any]) -> dict[str, Any]:
        """
        Consolidate and deduplicate S/R levels from multiple detection methods.

        Args:
            enhanced_levels: Dictionary containing levels from multiple methods

        Returns:
            Consolidated levels with deduplication
        """
        try:
            all_support = enhanced_levels.get("support_levels", [])
            all_resistance = enhanced_levels.get("resistance_levels", [])

            # Group levels by proximity
            consolidated_support = self._group_levels_by_proximity(all_support)
            consolidated_resistance = self._group_levels_by_proximity(all_resistance)

            # Sort by strength
            consolidated_support.sort(key=lambda x: x["strength"], reverse=True)
            consolidated_resistance.sort(key=lambda x: x["strength"], reverse=True)

            # Keep only the strongest levels (limit to prevent overcrowding)
            max_levels_per_side = 10
            consolidated_support = consolidated_support[:max_levels_per_side]
            consolidated_resistance = consolidated_resistance[:max_levels_per_side]

            return {
                "support_levels": consolidated_support,
                "resistance_levels": consolidated_resistance,
                "level_metadata": enhanced_levels.get("level_metadata", {}),
            }

        except Exception as e:
            self.logger.error(f"❌ Error consolidating S/R levels: {e}")
            return {"support_levels": [], "resistance_levels": []}

    def _group_levels_by_proximity(self, levels: list[dict]) -> list[dict]:
        """
        Group levels that are close to each other and merge their properties.

        Args:
            levels: List of level dictionaries

        Returns:
            Consolidated list of levels
        """
        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x["price"])

        # Group levels within proximity threshold
        proximity_threshold = 0.005  # 0.5% of price
        grouped_levels = []
        current_group = [sorted_levels[0]]

        for i in range(1, len(sorted_levels)):
            current_level = sorted_levels[i]
            last_level = current_group[-1]

            # Check if levels are close enough to group
            price_diff = (
                abs(current_level["price"] - last_level["price"]) / last_level["price"]
            )

            if price_diff <= proximity_threshold:
                current_group.append(current_level)
            else:
                # Merge current group
                merged_level = self._merge_level_group(current_group)
                grouped_levels.append(merged_level)
                current_group = [current_level]

        # Merge the last group
        if current_group:
            merged_level = self._merge_level_group(current_group)
            grouped_levels.append(merged_level)

        return grouped_levels

    def _merge_level_group(self, level_group: list[dict]) -> dict:
        """
        Merge a group of nearby levels into a single level.

        Args:
            level_group: List of levels to merge

        Returns:
            Merged level dictionary
        """
        if not level_group:
            return {}

        # Calculate weighted average price
        total_weight = sum(level["strength"] for level in level_group)
        weighted_price = (
            sum(level["price"] * level["strength"] for level in level_group)
            / total_weight
        )

        # Take the maximum strength
        max_strength = max(level["strength"] for level in level_group)

        # Combine types
        types = list(set(level.get("type", "unknown") for level in level_group))

        # Create merged level
        merged_level = {
            "price": float(weighted_price),
            "strength": float(max_strength),
            "type": "+".join(types),
            "group_size": len(level_group),
        }

        # Add additional properties if available
        if "timestamp" in level_group[0]:
            merged_level["timestamp"] = level_group[0]["timestamp"]
        if "volume_ratio" in level_group[0]:
            merged_level["volume_ratio"] = float(
                np.mean([level.get("volume_ratio", 0) for level in level_group])
            )

        return merged_level

    def _add_atr_based_activation_ranges(
        self, price_data: pd.DataFrame, levels: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Add ATR-based activation ranges to S/R levels.

        Args:
            price_data: OHLCV price data
            levels: Dictionary containing S/R levels

        Returns:
            Levels with ATR-based activation ranges
        """
        try:
            # Calculate ATR
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            close = price_data["close"].astype(float)

            # True Range calculation
            close_prev = close.shift(1)
            tr1 = high - low
            tr2 = np.abs(high - close_prev)
            tr3 = np.abs(low - close_prev)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR
            atr = true_range.rolling(window=self.atr_period, min_periods=1).mean()
            current_atr = atr.iloc[-1]

            # Add activation ranges to levels
            for level_type in ["support_levels", "resistance_levels"]:
                for level in levels.get(level_type, []):
                    level_strength = level.get("strength", 1.0)

                    # Calculate ATR-based activation range
                    activation_range = (
                        current_atr * self.atr_multiplier * level_strength
                    )

                    # Fallback to percentage-based if ATR is too small
                    if activation_range < level["price"] * self.atr_fallback_multiplier:
                        activation_range = (
                            level["price"]
                            * self.atr_fallback_multiplier
                            * level_strength
                        )

                    level["activation_range"] = float(activation_range)
                    level["atr_value"] = float(current_atr)

            return levels

        except Exception as e:
            self.logger.error(f"❌ Error adding ATR-based activation ranges: {e}")
            return levels

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="enhanced S/R breakout prediction",
    )
    async def predict_breakouts(
        self,
        prediction_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Enhanced S/R breakout prediction using multiple detection methods.

        Args:
            prediction_input: Prediction input parameters

        Returns:
            Dictionary containing enhanced S/R breakout predictions
        """
        try:
            if not self.is_initialized:
                self.logger.error("❌ SR breakout predictor not initialized")
                return {}

            # Validate prediction input
            if not self._validate_prediction_input(prediction_input):
                return {}

            # Extract data from input
            df = prediction_input.get("dataframe")
            current_price = prediction_input.get("current_price")

            if df is None or current_price is None:
                self.logger.error("Missing required prediction data")
                return {}

            # Detect enhanced S/R levels
            enhanced_levels = self.detect_enhanced_sr_levels(df)

            # Calculate distances and proximity scores
            support_distances = []
            resistance_distances = []

            for level in enhanced_levels.get("support_levels", []):
                distance = (current_price - level["price"]) / current_price
                if distance > 0:  # Price is above support
                    support_distances.append(
                        {
                            "level": level,
                            "distance": distance,
                            "proximity_score": self._calculate_proximity_score(
                                distance, level
                            ),
                        }
                    )

            for level in enhanced_levels.get("resistance_levels", []):
                distance = (level["price"] - current_price) / current_price
                if distance > 0:  # Price is below resistance
                    resistance_distances.append(
                        {
                            "level": level,
                            "distance": distance,
                            "proximity_score": self._calculate_proximity_score(
                                distance, level
                            ),
                        }
                    )

            # Sort by proximity
            support_distances.sort(key=lambda x: x["distance"])
            resistance_distances.sort(key=lambda x: x["distance"])

            # Generate predictions
            predictions = {
                "nearest_support": support_distances[0] if support_distances else None,
                "nearest_resistance": resistance_distances[0]
                if resistance_distances
                else None,
                "support_levels": enhanced_levels.get("support_levels", []),
                "resistance_levels": enhanced_levels.get("resistance_levels", []),
                "current_price": float(current_price),
                "level_metadata": enhanced_levels.get("level_metadata", {}),
            }

            # Add breakout probability estimates
            if predictions["nearest_support"] and predictions["nearest_resistance"]:
                support_proximity = predictions["nearest_support"]["proximity_score"]
                resistance_proximity = predictions["nearest_resistance"][
                    "proximity_score"
                ]

                # Breakout probability based on proximity and strength
                support_strength = predictions["nearest_support"]["level"]["strength"]
                resistance_strength = predictions["nearest_resistance"]["level"][
                    "strength"
                ]

                # Higher proximity and lower strength of opposing level = higher breakout probability
                breakout_up_prob = min(
                    resistance_proximity * (1 - support_strength) * 0.8 + 0.2, 1.0
                )
                breakout_down_prob = min(
                    support_proximity * (1 - resistance_strength) * 0.8 + 0.2, 1.0
                )

                predictions["breakout_probabilities"] = {
                    "breakout_up": float(breakout_up_prob),
                    "breakout_down": float(breakout_down_prob),
                }

            self.sr_predictions = predictions
            self.logger.info(
                "✅ Enhanced SR breakout prediction completed successfully"
            )

            return predictions

        except Exception as e:
            self.logger.error(f"❌ Error in enhanced breakout prediction: {e}")
            return {}

    def _validate_prediction_input(self, prediction_input: dict[str, Any]) -> bool:
        """
        Validate prediction input parameters.

        Args:
            prediction_input: Prediction input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["dataframe", "current_price"]

            for field in required_fields:
                if field not in prediction_input:
                    self.logger.error(
                        f"Missing required prediction input field: {field}"
                    )
                    return False

            # Validate specific field values
            if prediction_input.get("current_price", 0) <= 0:
                self.logger.error("Invalid current_price value")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Prediction input validation failed: {e}")
            return False

    def _calculate_proximity_score(self, distance: float, level: dict) -> float:
        """
        Calculate proximity score based on distance and level properties.

        Args:
            distance: Distance to level (as percentage)
            level: Level dictionary

        Returns:
            Proximity score (0.0 to 1.0)
        """
        try:
            # Base proximity using exponential decay
            base_proximity = np.exp(-distance / 0.02)  # 2% scale

            # Adjust for level strength
            strength_factor = level.get("strength", 1.0)

            # Adjust for activation range if available
            activation_range = level.get("activation_range", None)
            if activation_range:
                # Normalize activation range
                normalized_range = activation_range / level["price"]
                range_factor = 1.0 / (
                    1.0 + normalized_range * 10
                )  # Penalize wide ranges
            else:
                range_factor = 1.0

            # Combine factors
            proximity_score = base_proximity * strength_factor * range_factor

            return float(np.clip(proximity_score, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"❌ Error calculating proximity score: {e}")
            return 0.0

    async def stop(self) -> None:
        """Stop SR breakout predictor."""
        try:
            self.logger.info("🛑 Stopping SR breakout predictor...")
            self.is_initialized = False
            self.logger.info("✅ SR breakout predictor stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Error stopping SR breakout predictor: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR breakout predictor cleanup",
    )
    async def stop(self) -> None:
        """Stop the SR breakout predictor and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping SR Breakout Predictor...")
            self.is_initialized = False
            self.logger.info("✅ SR Breakout Predictor stopped successfully")

        except Exception:
            self.print(failed("❌ Failed to stop SR Breakout Predictor: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=0.0,
        context="SR confidence score calculation",
    )
    def calculate_sr_confidence_score(
        self,
        sr_context: dict[str, Any],
        current_price: float,
        market_data: pd.DataFrame,
    ) -> float:
        """
        Calculate comprehensive S/R confidence score for risk management.

        This method integrates all S/R detection methods and provides a unified
        confidence score based on:
        1. Level strength (touches, volume, age)
        2. Proximity to levels
        3. Breakout/bounce probabilities
        4. Market context and volatility

        Args:
            sr_context: S/R context from get_sr_context
            current_price: Current market price
            market_data: Recent market data for context

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        try:
            confidence_components = {}

            # 1. Level Strength Component (40% weight)
            level_strength_score = self._calculate_level_strength_score(sr_context)
            confidence_components["level_strength"] = level_strength_score

            # 2. Proximity Component (25% weight)
            proximity_score = self._calculate_proximity_score(sr_context, current_price)
            confidence_components["proximity"] = proximity_score

            # 3. Breakout/Bounce Probability Component (20% weight)
            probability_score = self._calculate_probability_score(sr_context)
            confidence_components["probability"] = probability_score

            # 4. Market Context Component (15% weight)
            context_score = self._calculate_market_context_score(
                market_data, sr_context
            )
            confidence_components["market_context"] = context_score

            # Calculate weighted confidence score
            weights = {
                "level_strength": 0.40,
                "proximity": 0.25,
                "probability": 0.20,
                "market_context": 0.15,
            }

            final_confidence = sum(
                confidence_components[component] * weights[component]
                for component in weights.keys()
            )

            # Log confidence breakdown for debugging
            self.logger.debug(f"SR Confidence Components: {confidence_components}")
            self.logger.debug(f"Final SR Confidence Score: {final_confidence:.3f}")

            return max(0.0, min(1.0, final_confidence))

        except Exception as e:
            self.logger.error(f"Error calculating SR confidence score: {e}")
            return 0.5  # Neutral confidence as fallback

    def _calculate_level_strength_score(self, sr_context: dict[str, Any]) -> float:
        """Calculate confidence based on S/R level strength."""
        try:
            strength_scores = []

            # Analyze pivot levels
            pivot_levels = sr_context.get("pivot_levels", {})
            if pivot_levels:
                pivot_strengths = pivot_levels.get("strengths", {})
                for strength_data in pivot_strengths.values():
                    if isinstance(strength_data, dict):
                        # Combine touches, volume, and age
                        touches = strength_data.get("touches", 0)
                        volume_strength = strength_data.get("strength", 0.0)
                        age = strength_data.get("age", 0)

                        # Normalize components
                        touch_score = min(touches / 10.0, 1.0)  # Max 10 touches
                        age_score = min(age / 100.0, 1.0)  # Max 100 periods

                        # Weighted strength score
                        level_strength = (
                            touch_score * 0.4 + volume_strength * 0.4 + age_score * 0.2
                        )
                        strength_scores.append(level_strength)

            # Analyze HVN levels
            hvn_levels = sr_context.get("hvn_levels", {})
            if hvn_levels:
                hvn_strengths = hvn_levels.get("strengths", {})
                for strength_data in hvn_strengths.values():
                    if isinstance(strength_data, dict):
                        strength_scores.append(strength_data.get("strength", 0.0))

            # Return average strength score
            if strength_scores:
                return sum(strength_scores) / len(strength_scores)
            else:
                return 0.3  # Default low strength

        except Exception as e:
            self.logger.error(f"Error calculating level strength score: {e}")
            return 0.3

    def _calculate_proximity_score(
        self, sr_context: dict[str, Any], current_price: float
    ) -> float:
        """Calculate confidence based on proximity to S/R levels."""
        try:
            nearest_support = sr_context.get("nearest_support", current_price)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)

            # Calculate proximity percentages
            support_proximity = abs(current_price - nearest_support) / current_price
            resistance_proximity = (
                abs(nearest_resistance - current_price) / current_price
            )

            # Convert to proximity scores (closer = higher score)
            # Optimal proximity is 1-3%, too close (<0.5%) or far (>5%) reduces confidence
            def proximity_to_score(proximity: float) -> float:
                if proximity < 0.005:  # Too close
                    return 0.3
                elif 0.01 <= proximity <= 0.03:  # Optimal range
                    return 1.0
                elif 0.03 < proximity <= 0.05:  # Good range
                    return 0.8
                else:  # Too far
                    return 0.2

            support_score = proximity_to_score(support_proximity)
            resistance_score = proximity_to_score(resistance_proximity)

            # Return the higher score (closer to a level)
            return max(support_score, resistance_score)

        except Exception as e:
            self.logger.error(f"Error calculating proximity score: {e}")
            return 0.5

    def _calculate_probability_score(self, sr_context: dict[str, Any]) -> float:
        """Calculate confidence based on breakout/bounce probabilities."""
        try:
            breakout_prob = sr_context.get("breakout_probability", 0.5)
            bounce_prob = sr_context.get("bounce_probability", 0.5)

            # Higher confidence when probabilities are more extreme (closer to 0 or 1)
            # Lower confidence when probabilities are around 0.5 (uncertain)
            breakout_confidence = (
                1.0 - abs(breakout_prob - 0.5) * 2
            )  # 0.5 -> 0.0, 0.0/1.0 -> 1.0
            bounce_confidence = 1.0 - abs(bounce_prob - 0.5) * 2

            # Return average of both confidences
            return (breakout_confidence + bounce_confidence) / 2

        except Exception as e:
            self.logger.error(f"Error calculating probability score: {e}")
            return 0.5

    def _calculate_market_context_score(
        self, market_data: pd.DataFrame, sr_context: dict[str, Any]
    ) -> float:
        """Calculate confidence based on market context and volatility."""
        try:
            if market_data.empty:
                return 0.5

            # Calculate volatility
            price_volatility = (
                market_data["close"].pct_change().rolling(20).std().iloc[-1]
            )

            # Calculate volume trend
            volume_ratio = (
                market_data["volume"].iloc[-5:].mean()
                / market_data["volume"].iloc[-20:].mean()
            )

            # Calculate price momentum
            price_momentum = (
                market_data["close"].iloc[-1] - market_data["close"].iloc[-10]
            ) / market_data["close"].iloc[-10]

            # Volatility score: moderate volatility is best for S/R
            if price_volatility < 0.01:  # Too low volatility
                volatility_score = 0.3
            elif 0.01 <= price_volatility <= 0.03:  # Good volatility
                volatility_score = 1.0
            elif 0.03 < price_volatility <= 0.05:  # High but acceptable
                volatility_score = 0.7
            else:  # Too volatile
                volatility_score = 0.2

            # Volume score: higher volume near levels is better
            volume_score = min(volume_ratio, 2.0) / 2.0  # Normalize to 0-1

            # Momentum score: strong momentum can break levels
            momentum_score = (
                1.0 - min(abs(price_momentum), 0.1) / 0.1
            )  # Lower momentum = higher confidence

            # Combine context scores
            context_score = (
                volatility_score * 0.5 + volume_score * 0.3 + momentum_score * 0.2
            )

            return context_score

        except Exception as e:
            self.logger.error(f"Error calculating market context score: {e}")
            return 0.5

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="enhanced SR prediction with confidence",
    )
    async def predict_breakout_with_confidence(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """
        Enhanced SR breakout prediction with comprehensive confidence scoring.

        Args:
            market_data: Recent market data
            current_price: Current market price

        Returns:
            dict: Enhanced prediction with confidence scores
        """
        try:
            # Get base SR context
            sr_context = await self.get_sr_context(market_data, current_price)

            # Calculate comprehensive confidence score
            sr_confidence = self.calculate_sr_confidence_score(
                sr_context, current_price, market_data
            )

            # Get breakout prediction
            breakout_prediction = await self.predict_breakout(
                market_data, current_price
            )

            # Enhance prediction with confidence
            enhanced_prediction = {
                **breakout_prediction,
                "sr_confidence_score": sr_confidence,
                "confidence_breakdown": {
                    "level_strength": self._calculate_level_strength_score(sr_context),
                    "proximity": self._calculate_proximity_score(
                        sr_context, current_price
                    ),
                    "probability": self._calculate_probability_score(sr_context),
                    "market_context": self._calculate_market_context_score(
                        market_data, sr_context
                    ),
                },
                "risk_assessment": {
                    "high_confidence": sr_confidence >= 0.8,
                    "medium_confidence": 0.6 <= sr_confidence < 0.8,
                    "low_confidence": sr_confidence < 0.6,
                    "recommended_position_size": min(sr_confidence, 0.8),  # Cap at 80%
                    "stop_loss_multiplier": 1.0
                    + (1.0 - sr_confidence) * 0.5,  # Wider stops for low confidence
                },
            }

            return enhanced_prediction

        except Exception as e:
            self.logger.error(f"Error in enhanced SR prediction: {e}")
            return {
                "sr_confidence_score": 0.5,
                "confidence_breakdown": {},
                "risk_assessment": {
                    "high_confidence": False,
                    "medium_confidence": True,
                    "low_confidence": False,
                    "recommended_position_size": 0.5,
                    "stop_loss_multiplier": 1.25,
                },
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="S/R proximity detection",
    )
    def is_near_sr_level(
        self, current_price: float, sr_context: dict[str, Any]
    ) -> bool:
        """
        Determine if current price is near a significant S/R level.
        Uses dynamic proximity thresholds based on S/R level strength.

        Args:
            current_price: Current market price
            sr_context: S/R context from get_sr_context

        Returns:
            bool: True if near S/R level, False otherwise
        """
        try:
            if not sr_context:
                return False

            # Get nearest support and resistance levels
            nearest_support = sr_context.get("nearest_support", current_price)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)

            # Calculate proximity to each level
            support_proximity = abs(current_price - nearest_support) / current_price
            resistance_proximity = (
                abs(nearest_resistance - current_price) / current_price
            )

            # Get S/R level strength to adjust proximity threshold
            support_strength = sr_context.get("support_strength", 0.5)
            resistance_strength = sr_context.get("resistance_strength", 0.5)

            # Calculate dynamic proximity thresholds based on level strength
            # Stronger levels get wider proximity (easier to detect)
            # Weaker levels get tighter proximity (harder to detect)
            support_threshold = self._calculate_dynamic_proximity_threshold(
                support_strength
            )
            resistance_threshold = self._calculate_dynamic_proximity_threshold(
                resistance_strength
            )

            # Check if within dynamic proximity thresholds
            near_support = support_proximity <= support_threshold
            near_resistance = resistance_proximity <= resistance_threshold

            return near_support or near_resistance

        except Exception as e:
            self.logger.error(f"Error detecting S/R proximity: {e}")
            return False

    def _calculate_dynamic_proximity_threshold(self, level_strength: float) -> float:
        """
        Calculate dynamic proximity threshold based on S/R level strength.

        Args:
            level_strength: Strength of the S/R level (0.0 to 1.0)

        Returns:
            float: Dynamic proximity threshold (0.3% to 1.0%)
        """
        try:
            # Normalize level strength to 0-1 range
            normalized_strength = max(0.0, min(1.0, level_strength))

            # Calculate dynamic threshold
            # Stronger levels (higher strength) get wider proximity
            # Weaker levels (lower strength) get tighter proximity
            threshold_range = (
                self.sr_proximity_trigger_max - self.sr_proximity_trigger_min
            )
            dynamic_threshold = self.sr_proximity_trigger_min + (
                normalized_strength * threshold_range
            )

            # Ensure within bounds
            dynamic_threshold = max(
                self.sr_proximity_trigger_min,
                min(self.sr_proximity_trigger_max, dynamic_threshold),
            )

            return dynamic_threshold

        except Exception as e:
            self.logger.error(f"Error calculating dynamic proximity threshold: {e}")
            return self.sr_proximity_trigger_base  # Fallback to base threshold

    def get_sr_proximity_details(
        self, current_price: float, sr_context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get detailed S/R proximity information including dynamic thresholds.

        Args:
            current_price: Current market price
            sr_context: S/R context from get_sr_context

        Returns:
            dict: Detailed proximity information
        """
        try:
            if not sr_context:
                return {
                    "is_near_sr": False,
                    "reason": "No S/R context available",
                    "proximity_details": {},
                }

            # Get nearest levels
            nearest_support = sr_context.get("nearest_support", current_price)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)

            # Calculate proximities
            support_proximity = abs(current_price - nearest_support) / current_price
            resistance_proximity = (
                abs(nearest_resistance - current_price) / current_price
            )

            # Get level strengths
            support_strength = sr_context.get("support_strength", 0.5)
            resistance_strength = sr_context.get("resistance_strength", 0.5)

            # Calculate dynamic thresholds
            support_threshold = self._calculate_dynamic_proximity_threshold(
                support_strength
            )
            resistance_threshold = self._calculate_dynamic_proximity_threshold(
                resistance_strength
            )

            # Check proximity
            near_support = support_proximity <= support_threshold
            near_resistance = resistance_proximity <= resistance_threshold
            is_near_sr = near_support or near_resistance

            # Build detailed response
            proximity_details = {
                "support": {
                    "level": nearest_support,
                    "proximity": support_proximity,
                    "threshold": support_threshold,
                    "strength": support_strength,
                    "is_near": near_support,
                    "distance_pct": support_proximity * 100,
                },
                "resistance": {
                    "level": nearest_resistance,
                    "proximity": resistance_proximity,
                    "threshold": resistance_threshold,
                    "strength": resistance_strength,
                    "is_near": near_resistance,
                    "distance_pct": resistance_proximity * 100,
                },
                "configuration": {
                    "base_threshold": self.sr_proximity_trigger_base,
                    "min_threshold": self.sr_proximity_trigger_min,
                    "max_threshold": self.sr_proximity_trigger_max,
                    "current_price": current_price,
                },
            }

            return {
                "is_near_sr": is_near_sr,
                "reason": f"Near {'support' if near_support else 'resistance' if near_resistance else 'neither'} level",
                "proximity_details": proximity_details,
            }

        except Exception as e:
            self.logger.error(f"Error getting S/R proximity details: {e}")
            return {
                "is_near_sr": False,
                "reason": f"Error: {e}",
                "proximity_details": {},
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return="consolidation",
        context="S/R outcome prediction",
    )
    async def predict_sr_outcome(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        sr_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Predict S/R outcome using unified model: breakout, rebounce, or consolidation.

        Args:
            market_data: Recent market data
            current_price: Current market price
            sr_context: S/R context

        Returns:
            dict: Prediction with outcome, confidence, and probabilities
        """
        try:
            # Check if we're near S/R level
            is_near_level = self.is_near_sr_level(current_price, sr_context)

            if not is_near_level:
                return {
                    "outcome": "consolidation",
                    "confidence": 0.8,
                    "probabilities": {
                        "breakout": 0.1,
                        "rebounce": 0.1,
                        "consolidation": 0.8,
                    },
                    "reason": "Not near S/R level",
                }

            # Prepare features for outcome prediction
            features = self._prepare_sr_outcome_features(
                market_data, current_price, sr_context
            )

            # Get model prediction (this would use the actual trained model)
            prediction_result = await self._get_sr_outcome_prediction(features)

            # Determine outcome based on highest probability
            probabilities = prediction_result.get("probabilities", {})
            outcome = (
                max(probabilities, key=probabilities.get)
                if probabilities
                else "consolidation"
            )
            confidence = prediction_result.get("confidence", 0.5)

            # Check if confidence meets threshold for the predicted outcome
            threshold = self.outcome_confidence_thresholds.get(outcome, 0.6)
            if confidence < threshold:
                outcome = "consolidation"  # Default to consolidation if low confidence
                confidence = max(confidence, 0.6)

            return {
                "outcome": outcome,
                "confidence": confidence,
                "probabilities": probabilities,
                "is_near_sr_level": True,
                "sr_context": sr_context,
            }

        except Exception as e:
            self.logger.error(f"Error predicting S/R outcome: {e}")
            return {
                "outcome": "consolidation",
                "confidence": 0.5,
                "probabilities": {
                    "breakout": 0.33,
                    "rebounce": 0.33,
                    "consolidation": 0.34,
                },
                "reason": f"Prediction error: {e}",
            }

    def _prepare_sr_outcome_features(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        sr_context: dict[str, Any],
    ) -> dict[str, float]:
        """
        Prepare features for S/R outcome prediction.

        Args:
            market_data: Market data
            current_price: Current price
            sr_context: S/R context

        Returns:
            dict: Feature dictionary for outcome prediction
        """
        try:
            features = {}

            # Price-based features
            features["price_change_1m"] = (
                market_data["close"].pct_change().iloc[-1]
                if len(market_data) > 1
                else 0
            )
            features["price_change_5m"] = (
                market_data["close"].pct_change(5).iloc[-1]
                if len(market_data) > 5
                else 0
            )
            features["price_volatility"] = (
                market_data["close"].rolling(20).std().iloc[-1]
                if len(market_data) >= 20
                else 0
            )

            # Volume-based features
            features["volume_ratio"] = (
                (
                    market_data["volume"].iloc[-1]
                    / market_data["volume"].rolling(20).mean().iloc[-1]
                )
                if len(market_data) >= 20
                else 1.0
            )
            features["volume_momentum"] = (
                market_data["volume"].pct_change().iloc[-1]
                if len(market_data) > 1
                else 0
            )

            # Technical indicators
            features["rsi"] = (
                self._calculate_rsi(market_data["close"]).iloc[-1]
                if len(market_data) >= 14
                else 50
            )
            features["macd"] = (
                self._calculate_macd(market_data["close"]).iloc[-1]
                if len(market_data) >= 26
                else 0
            )

            # S/R-specific features
            if sr_context:
                nearest_support = sr_context.get("nearest_support", current_price)
                nearest_resistance = sr_context.get("nearest_resistance", current_price)

                # Distance to levels (normalized)
                features["distance_to_support"] = (
                    current_price - nearest_support
                ) / current_price
                features["distance_to_resistance"] = (
                    nearest_resistance - current_price
                ) / current_price

                # Level strength features
                features["support_strength"] = sr_context.get("support_strength", 0.5)
                features["resistance_strength"] = sr_context.get(
                    "resistance_strength", 0.5
                )

                # Pivot level features
                pivot_levels = sr_context.get("pivot_levels", {})
                if pivot_levels:
                    features["nearest_pivot_strength"] = pivot_levels.get(
                        "nearest_strength", 0.5
                    )
                    features["pivot_touches"] = pivot_levels.get("nearest_touches", 0)
                else:
                    features["nearest_pivot_strength"] = 0.5
                    features["pivot_touches"] = 0

            # Market context features
            features["market_trend"] = self._calculate_market_trend(market_data)
            features["momentum_strength"] = self._calculate_momentum_strength(
                market_data
            )

            return features

        except Exception as e:
            self.logger.error(f"Error preparing S/R outcome features: {e}")
            return {}

    async def _get_sr_outcome_prediction(
        self, features: dict[str, float]
    ) -> dict[str, Any]:
        """
        Get S/R outcome prediction from the unified model.
        Loads and uses trained ML models for breakout/rebounce/consolidation prediction.

        Args:
            features: Prepared features

        Returns:
            dict: Model prediction with probabilities and confidence
        """
        try:
            # Try to load and use trained model first
            model_prediction = await self._get_trained_model_prediction(features)
            if model_prediction:
                return model_prediction

            # Fallback to rule-based prediction if no trained model available
            return self._get_rule_based_prediction(features)

        except Exception as e:
            self.logger.error(f"Error getting S/R outcome prediction: {e}")
            return {
                "probabilities": {
                    "breakout": 0.33,
                    "rebounce": 0.33,
                    "consolidation": 0.34,
                },
                "confidence": 0.5,
            }

    async def _get_trained_model_prediction(
        self, features: dict[str, float]
    ) -> Optional[dict[str, Any]]:
        """
        Get prediction from trained ML model.

        Args:
            features: Prepared features

        Returns:
            dict: Model prediction or None if model not available
        """
        try:
            # Check if we have a trained model available
            model_path = self.config.get(
                "sr_outcome_model_path", "models/sr_outcome_model.pkl"
            )
            if not os.path.exists(model_path):
                self.logger.debug(
                    f"Trained S/R outcome model not found at {model_path}"
                )
                return None

            # Load the trained model
            import pickle

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Prepare features for model input
            feature_names = [
                "price_change_1m",
                "price_change_5m",
                "price_volatility",
                "volume_ratio",
                "volume_momentum",
                "rsi",
                "macd",
                "distance_to_support",
                "distance_to_resistance",
                "support_strength",
                "resistance_strength",
                "nearest_pivot_strength",
                "pivot_touches",
                "market_trend",
                "momentum_strength",
            ]

            # Create feature vector
            feature_vector = []
            for feature_name in feature_names:
                feature_vector.append(features.get(feature_name, 0.0))

            # Make prediction
            if hasattr(model, "predict_proba"):
                # For sklearn models
                probabilities = model.predict_proba([feature_vector])[0]
                prediction = model.predict([feature_vector])[0]

                # Map prediction to outcome
                outcome_mapping = {0: "breakout", 1: "rebounce", 2: "consolidation"}
                outcome = outcome_mapping.get(prediction, "consolidation")

                # Create probability dict
                prob_dict = {
                    "breakout": probabilities[0],
                    "rebounce": probabilities[1],
                    "consolidation": probabilities[2],
                }

                # Calculate confidence as max probability
                confidence = max(probabilities)

            elif hasattr(model, "forward"):
                # For PyTorch models
                import torch

                model.eval()
                with torch.no_grad():
                    feature_tensor = torch.FloatTensor([feature_vector])
                    outputs = model(feature_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0].numpy()

                    outcome_mapping = {0: "breakout", 1: "rebounce", 2: "consolidation"}
                    prediction = torch.argmax(outputs, dim=1).item()
                    outcome = outcome_mapping.get(prediction, "consolidation")

                    prob_dict = {
                        "breakout": float(probabilities[0]),
                        "rebounce": float(probabilities[1]),
                        "consolidation": float(probabilities[2]),
                    }
                    confidence = float(max(probabilities))
            else:
                self.logger.warning(
                    "Unknown model type, falling back to rule-based prediction"
                )
                return None

            return {
                "probabilities": prob_dict,
                "confidence": confidence,
                "outcome": outcome,
                "model_type": "trained_ml_model",
            }

        except Exception as e:
            self.logger.debug(f"Error loading trained model: {e}")
            return None

    def _get_rule_based_prediction(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Get rule-based prediction as fallback when trained model is not available.

        Args:
            features: Prepared features

        Returns:
            dict: Rule-based prediction
        """
        try:
            # Extract key features
            momentum = features.get("momentum_strength", 0)
            volume_ratio = features.get("volume_ratio", 1.0)
            rsi = features.get("rsi", 50)
            market_trend = features.get("market_trend", 0)
            distance_to_support = features.get("distance_to_support", 0)
            distance_to_resistance = features.get("distance_to_resistance", 0)
            support_strength = features.get("support_strength", 0.5)
            resistance_strength = features.get("resistance_strength", 0.5)

            # Rule-based logic for outcome prediction
            breakout_score = 0.0
            rebounce_score = 0.0
            consolidation_score = 0.0

            # Breakout conditions
            if momentum > 0.5 and volume_ratio > 1.2 and market_trend > 0.3:
                breakout_score += 0.4
            if rsi > 70 and momentum > 0.3:
                breakout_score += 0.3
            if distance_to_resistance < 0.01 and resistance_strength < 0.7:
                breakout_score += 0.3

            # Rebounce conditions
            if momentum < -0.5 and volume_ratio > 1.2 and market_trend < -0.3:
                rebounce_score += 0.4
            if rsi < 30 and momentum < -0.3:
                rebounce_score += 0.3
            if distance_to_support < 0.01 and support_strength < 0.7:
                rebounce_score += 0.3

            # Consolidation conditions (default)
            consolidation_score = 0.4  # Base consolidation probability

            # Adjust based on volatility and trend strength
            if abs(momentum) < 0.2 and volume_ratio < 1.1:
                consolidation_score += 0.3
                breakout_score *= 0.7
                rebounce_score *= 0.7

            # Normalize scores to probabilities
            total_score = breakout_score + rebounce_score + consolidation_score
            if total_score > 0:
                probabilities = {
                    "breakout": breakout_score / total_score,
                    "rebounce": rebounce_score / total_score,
                    "consolidation": consolidation_score / total_score,
                }
            else:
                probabilities = {
                    "breakout": 0.33,
                    "rebounce": 0.33,
                    "consolidation": 0.34,
                }

            # Determine outcome and confidence
            outcome = max(probabilities, key=probabilities.get)
            confidence = probabilities[outcome]

            # Boost confidence for clear signals
            if max(probabilities.values()) > 0.6:
                confidence = min(0.9, confidence * 1.2)

            return {
                "probabilities": probabilities,
                "confidence": confidence,
                "outcome": outcome,
                "model_type": "rule_based",
            }

        except Exception as e:
            self.logger.error(f"Error in rule-based prediction: {e}")
            return {
                "probabilities": {
                    "breakout": 0.33,
                    "rebounce": 0.33,
                    "consolidation": 0.34,
                },
                "confidence": 0.5,
                "outcome": "consolidation",
                "model_type": "rule_based",
            }

    def _calculate_market_trend(self, market_data: pd.DataFrame) -> float:
        """Calculate market trend strength."""
        try:
            if len(market_data) < 20:
                return 0.0

            # Simple trend calculation using linear regression slope
            prices = market_data["close"].values
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]

            # Normalize slope
            avg_price = np.mean(prices)
            normalized_slope = slope / avg_price if avg_price > 0 else 0

            return np.clip(normalized_slope * 100, -1, 1)  # Clip to [-1, 1]

        except Exception as e:
            self.logger.error(f"Error calculating market trend: {e}")
            return 0.0

    def _calculate_momentum_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate momentum strength."""
        try:
            if len(market_data) < 10:
                return 0.0

            # Calculate momentum using price change over different periods
            short_momentum = (
                market_data["close"].pct_change(5).iloc[-1]
                if len(market_data) > 5
                else 0
            )
            long_momentum = (
                market_data["close"].pct_change(20).iloc[-1]
                if len(market_data) > 20
                else 0
            )

            # Combine momentums
            momentum = short_momentum * 0.7 + long_momentum * 0.3

            return np.clip(momentum * 100, -1, 1)  # Clip to [-1, 1]

        except Exception as e:
            self.logger.error(f"Error calculating momentum strength: {e}")
            return 0.0

    def calculate_comprehensive_sr_features(
        self, price_data: pd.DataFrame, sr_levels: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Calculate SR_Score and ΔSR_Score for HMM regime discovery.
        
        This method generates only the two essential SR features for HMM:
        - SR_Score: Combined directional pressure and strength score
        - ΔSR_Score: Change in SR_Score from previous period
        
        All other calculations (distances, normalized distances, etc.) are internal
        and only used to compute the final SR_Score.
        
        Args:
            price_data: OHLCV price data
            sr_levels: Optional pre-calculated SR levels
            
        Returns:
            Dictionary containing only SR_Score and ΔSR_Score
        """
        try:
            if price_data.empty or "close" not in price_data.columns:
                self.logger.warning("⚠️ Invalid price data for SR feature calculation")
                return {}
            
            close = price_data["close"].astype(float)
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            volume = price_data["volume"].astype(float)
            
            # Get SR levels if not provided
            if sr_levels is None:
                sr_levels = self.detect_enhanced_sr_levels(price_data)
            
            # Calculate ATR for normalization
            atr = self._calculate_atr(price_data, period=14)
            
            # Extract support and resistance levels
            support_levels = sr_levels.get("support_levels", [])
            resistance_levels = sr_levels.get("resistance_levels", [])
            
            # Convert to price arrays
            support_prices = np.array([level.get("price", 0) for level in support_levels if level.get("price", 0) > 0])
            resistance_prices = np.array([level.get("price", 0) for level in resistance_levels if level.get("price", 0) > 0])
            
            # Calculate SR_Score for each price point
            sr_scores = []
            
            for i, current_price in enumerate(close):
                if pd.isna(current_price):
                    sr_scores.append(np.nan)
                    continue
                
                # Calculate distances to nearest levels
                if len(support_prices) > 0:
                    support_distances = abs(support_prices - current_price)
                    nearest_support_idx = np.argmin(support_distances)
                    nearest_support_distance = support_distances[nearest_support_idx]
                    nearest_support_strength = support_levels[nearest_support_idx].get("strength", 0.5)
                else:
                    nearest_support_distance = current_price * 0.1  # 10% fallback
                    nearest_support_strength = 0.3
                
                if len(resistance_prices) > 0:
                    resistance_distances = abs(resistance_prices - current_price)
                    nearest_resistance_idx = np.argmin(resistance_distances)
                    nearest_resistance_distance = resistance_distances[nearest_resistance_idx]
                    nearest_resistance_strength = resistance_levels[nearest_resistance_idx].get("strength", 0.5)
                else:
                    nearest_resistance_distance = current_price * 0.1  # 10% fallback
                    nearest_resistance_strength = 0.3
                
                # Calculate normalized distances (by ATR)
                current_atr = atr.iloc[i] if i < len(atr) and not pd.isna(atr.iloc[i]) else current_price * 0.02
                normalized_distance_to_support = nearest_support_distance / current_atr if current_atr > 0 else 0
                normalized_distance_to_resistance = nearest_resistance_distance / current_atr if current_atr > 0 else 0
                
                # Calculate directional pressure
                if normalized_distance_to_resistance + normalized_distance_to_support > 0:
                    directional_pressure = (normalized_distance_to_resistance - normalized_distance_to_support) / (normalized_distance_to_resistance + normalized_distance_to_support)
                else:
                    directional_pressure = 0
                
                # Calculate strength score (replaces clarity factor)
                strength_score = self._calculate_strength_score(
                    nearest_support_strength, nearest_resistance_strength,
                    len(support_prices), len(resistance_prices),
                    current_price, volume.iloc[i] if i < len(volume) else 0
                )
                
                # Calculate SR_Score = Directional_Pressure * Strength_Score
                sr_score = directional_pressure * strength_score
                sr_scores.append(sr_score)
            
            # Convert to pandas Series
            sr_score_series = pd.Series(sr_scores, index=close.index).fillna(0)
            
            # Calculate ΔSR_Score (change from previous period)
            delta_sr_score = sr_score_series.diff().fillna(0)
            
            # Return only the two essential features for HMM
            features = {
                "sr_score": sr_score_series,
                "delta_sr_score": delta_sr_score
            }
            
            self.logger.info(f"✅ Generated SR_Score and ΔSR_Score for HMM regime discovery")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error in SR feature calculation: {e}")
            return {}

    def _calculate_strength_score(
        self, support_strength: float, resistance_strength: float,
        support_count: int, resistance_count: int,
        current_price: float, volume: float
    ) -> float:
        """
        Calculate strength score for SR levels.
        
        This replaces the clarity factor and provides a measure of how strong
        the current S/R context is based on level strength, count, and market conditions.
        
        Args:
            support_strength: Strength of nearest support level
            resistance_strength: Strength of nearest resistance level
            support_count: Number of support levels
            resistance_count: Number of resistance levels
            current_price: Current market price
            volume: Current volume
            
        Returns:
            float: Strength score between 0.0 and 1.0
        """
        try:
            # Average strength of nearby levels
            avg_strength = (support_strength + resistance_strength) / 2
            
            # Level density factor (more levels = higher strength)
            total_levels = support_count + resistance_count
            level_density = min(total_levels / 10.0, 1.0)  # Normalize to 0-1
            
            # Volume factor (higher volume = stronger levels)
            volume_factor = min(volume / 10000.0, 1.0)  # Normalize to 0-1
            
            # Combine factors with weights
            strength_score = (
                avg_strength * 0.5 +      # 50% weight to level strength
                level_density * 0.3 +     # 30% weight to level density
                volume_factor * 0.2       # 20% weight to volume
            )
            
            return float(np.clip(strength_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating strength score: {e}")
            return 0.5

    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            close = price_data["close"].astype(float)
            
            # True Range calculation
            close_prev = close.shift(1)
            tr1 = high - low
            tr2 = np.abs(high - close_prev)
            tr3 = np.abs(low - close_prev)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR
            atr = true_range.rolling(window=period, min_periods=1).mean()
            return atr
        except Exception as e:
            self.logger.error(f"❌ Error calculating ATR: {e}")
            return pd.Series([0.02] * len(price_data), index=price_data.index)

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="SR breakout predictor setup",
)
async def setup_sr_breakout_predictor(
    config: dict[str, Any] | None = None,
) -> SRBreakoutPredictor | None:
    """
    Setup and return a configured SRBreakoutPredictor instance.

    Args:
        config: Configuration dictionary

    Returns:
        SRBreakoutPredictor: Configured SR breakout predictor instance
    """
    try:
        predictor = SRBreakoutPredictor(config or {})
        if await predictor.initialize():
            return predictor
        return None
    except Exception:
        system_logger.exception(failed("Failed to setup SR Breakout Predictor: {e}"))
        return None
