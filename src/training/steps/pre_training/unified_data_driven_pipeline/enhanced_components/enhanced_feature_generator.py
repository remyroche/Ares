"""
Enhanced Feature Generator for UnifiedDataDrivenPipeline

This module provides comprehensive feature generation including:
- Cross timeframe features with optimized lookback period
- Interaction (2-3) features with optimized lookback period
- Feature creation in multiple ways (addition, subtraction, log, multiplication, division)
- No features with optimized lookback period
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
import warnings
from collections import defaultdict
from itertools import combinations, product
import math

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        scale, rank, zscore, winsorize, clip, quantile
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)

@dataclass
class FeatureGenerationConfig:
    """Configuration for enhanced feature generation."""
    enable_cross_timeframe: bool = True
    enable_interaction_features: bool = True
    enable_multiple_creation_methods: bool = True
    enable_no_features: bool = True
    enable_feature_comparisons: bool = True
    max_cross_timeframe_features: int = 20
    max_interaction_features: int = 30
    max_no_features: int = 15
    max_comparison_features: int = 20
    cross_timeframe_periods: List[int] = None
    interaction_orders: List[int] = None
    creation_methods: List[str] = None
    base_timeframe_minutes: int = 15  # Default 15-minute timeframe
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True
    max_workers: int = 4

    def __post_init__(self):
        if self.cross_timeframe_periods is None:
            # Generate periods up to 600 minutes, respecting base timeframe
            base_periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 600]
            self.cross_timeframe_periods = [p * self.base_timeframe_minutes for p in base_periods]
        if self.interaction_orders is None:
            self.interaction_orders = [2]  # Only 2-way interactions (X² max)
        if self.creation_methods is None:
            self.creation_methods = [
                'add', 'subtract', 'multiply', 'divide', 'log', 'sqrt', 'power', 'ratio',
                'log_add', 'log_subtract', 'log_divide', 'log_multiply', 'log_ratio',  # Added log relationships
                'exp_add', 'exp_multiply',
                'abs_add', 'abs_multiply', 'square_add', 'square_multiply',
                'cube_add', 'cube_multiply', 'sin_add', 'cos_multiply', 'tan_divide'
            ]

@dataclass
class GeneratedFeature:
    """Generated feature with metadata."""
    name: str
    feature_type: str  # 'cross_timeframe', 'interaction', 'no_feature', 'comparison'
    formula: str
    parent_features: List[str]
    feature_series: pd.Series
    utility_score: float
    lookback_period: Optional[int] = None
    creation_method: Optional[str] = None
    base_timeframe_minutes: Optional[int] = None
    source_features: Optional[List[Dict[str, Any]]] = None  # For interaction features
    comparison_type: Optional[str] = None  # 'base', 'vwap', 'volatility_adjusted', 'zscore_volume'
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.source_features is None:
            self.source_features = []

@dataclass
class FeatureGenerationResult:
    """Result of feature generation."""
    cross_timeframe_features: List[GeneratedFeature]
    interaction_features: List[GeneratedFeature]
    no_features: List[GeneratedFeature]
    comparison_features: List[GeneratedFeature]
    all_features: List[GeneratedFeature]
    generation_time: float
    success: bool
    error_message: Optional[str] = None

class EnhancedFeatureGenerator:
    """
    Enhanced feature generator with comprehensive feature types.

    Features:
    - Cross timeframe features with optimized lookback period
    - Interaction (2-3) features with optimized lookback period
    - Feature creation in multiple ways (addition, subtraction, log, multiplication, division)
    - No features with optimized lookback period
    """

    def __init__(self, config: Optional[FeatureGenerationConfig] = None):
        """Initialize the enhanced feature generator."""
        self.config = config or FeatureGenerationConfig()

        # Performance tracking
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_execution_time': 0.0,
            'cross_timeframe_features_generated': 0,
            'interaction_features_generated': 0,
            'no_features_generated': 0,
            'comparison_features_generated': 0,
            'vectorbt_operations': 0
        }

        tprint_success("✅ Enhanced Feature Generator initialized")

    def generate_features(
        self,
        data: pd.DataFrame,
        targets: Optional[pd.Series] = None,
        base_features: Optional[pd.DataFrame] = None
    ) -> FeatureGenerationResult:
        """
        Generate comprehensive features including cross-timeframe, interactions, and no features.

        Args:
            data: Input OHLCV data
            targets: Optional target series for utility scoring
            base_features: Optional base features for interaction generation

        Returns:
            FeatureGenerationResult with all generated features
        """
        tprint_info("🚀 Starting enhanced feature generation")
        tprint_info(f"📊 Data shape: {data.shape}")

        start_time = time.time()

        try:
            # Initialize result containers
            cross_timeframe_features = []
            interaction_features = []
            no_features = []
            comparison_features = []

            # Generate cross-timeframe features
            if self.config.enable_cross_timeframe:
                tprint_info("Step 1: Generating cross-timeframe features")
                cross_timeframe_features = self._generate_cross_timeframe_features(data, targets)
                tprint_success(f"✅ Generated {len(cross_timeframe_features)} cross-timeframe features")

            # Generate interaction features
            if self.config.enable_interaction_features and base_features is not None:
                tprint_info("Step 2: Generating interaction features")
                interaction_features = self._generate_interaction_features(base_features, targets)
                tprint_success(f"✅ Generated {len(interaction_features)} interaction features")

            # Generate no features (features without lookback optimization)
            if self.config.enable_no_features:
                tprint_info("Step 3: Generating no features")
                no_features = self._generate_no_features(data, targets)
                tprint_success(f"✅ Generated {len(no_features)} no features")

            # Generate comparison features
            if self.config.enable_feature_comparisons:
                tprint_info("Step 4: Generating comparison features")
                comparison_features = self._generate_comparison_features(data, targets)
                tprint_success(f"✅ Generated {len(comparison_features)} comparison features")

            # Combine all features
            all_features = cross_timeframe_features + interaction_features + no_features + comparison_features

            execution_time = time.time() - start_time

            # Update performance stats
            self.performance_stats.update({
                'total_generations': 1,
                'successful_generations': 1,
                'total_execution_time': execution_time,
                'cross_timeframe_features_generated': len(cross_timeframe_features),
                'interaction_features_generated': len(interaction_features),
                'no_features_generated': len(no_features),
                'comparison_features_generated': len(comparison_features)
            })

            tprint_success(f"✅ Enhanced feature generation completed in {execution_time:.3f}s")
            tprint_info(f"🏆 Total features generated: {len(all_features)}")

            return FeatureGenerationResult(
                cross_timeframe_features=cross_timeframe_features,
                interaction_features=interaction_features,
                no_features=no_features,
                comparison_features=comparison_features,
                all_features=all_features,
                generation_time=execution_time,
                success=True
            )

        except Exception as e:
            tprint_error(f"❌ Enhanced feature generation failed: {e}")
            return FeatureGenerationResult(
                cross_timeframe_features=[],
                interaction_features=[],
                no_features=[],
                comparison_features=[],
                all_features=[],
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _generate_cross_timeframe_features(
        self,
        data: pd.DataFrame,
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate cross-timeframe features with optimized lookback periods."""
        tprint_debug("Generating cross-timeframe features")

        features = []

        try:
            # Ensure we have OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                tprint_warning("⚠️ Missing OHLCV columns, using available data")
                available_cols = [col for col in required_cols if col in data.columns]
                if not available_cols:
                    return features
            else:
                available_cols = required_cols

            # Generate features for each timeframe period
            for period in self.config.cross_timeframe_periods:
                # Skip if period is too large for data
                if period >= len(data) // 4:
                    continue

                # Generate different types of cross-timeframe features
                period_features = self._generate_period_cross_timeframe_features(
                    data, period, available_cols, targets
                )
                features.extend(period_features)

            # Limit to max features
            if len(features) > self.config.max_cross_timeframe_features:
                # Sort by utility score and take top features
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:self.config.max_cross_timeframe_features]

            return features

        except Exception as e:
            tprint_error(f"❌ Cross-timeframe feature generation failed: {e}")
            return []

    def _generate_period_cross_timeframe_features(
        self,
        data: pd.DataFrame,
        period: int,
        available_cols: List[str],
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate cross-timeframe features for a specific period."""
        features = []

        try:
            # Price-based cross-timeframe features
            if 'close' in available_cols:
                close = data['close']

                # Multi-timeframe momentum
                momentum_features = self._create_momentum_cross_timeframe_features(close, period)
                features.extend(momentum_features)

                # Multi-timeframe volatility
                volatility_features = self._create_volatility_cross_timeframe_features(close, period)
                features.extend(volatility_features)

                # Multi-timeframe trend
                trend_features = self._create_trend_cross_timeframe_features(close, period)
                features.extend(trend_features)

            # Volume-based cross-timeframe features
            if 'volume' in available_cols:
                volume = data['volume']
                volume_features = self._create_volume_cross_timeframe_features(volume, period)
                features.extend(volume_features)

            # OHLC-based cross-timeframe features
            if all(col in available_cols for col in ['high', 'low', 'close']):
                ohlc_features = self._create_ohlc_cross_timeframe_features(
                    data[['high', 'low', 'close']], period
                )
                features.extend(ohlc_features)

            # Calculate utility scores for all features
            for feature in features:
                feature.utility_score = self._calculate_utility_score(feature.feature_series, targets)
                feature.lookback_period = period
                feature.base_timeframe_minutes = self.config.base_timeframe_minutes
                feature.metadata.update({
                    'timeframe_period': period,
                    'feature_category': 'cross_timeframe',
                    'base_timeframe_minutes': self.config.base_timeframe_minutes,
                    'period_in_base_units': period // self.config.base_timeframe_minutes
                })

            return features

        except Exception as e:
            tprint_debug(f"Error generating period {period} cross-timeframe features: {e}")
            return []

    def _create_momentum_cross_timeframe_features(
        self,
        close: pd.Series,
        period: int
    ) -> List[GeneratedFeature]:
        """Create momentum-based cross-timeframe features with comprehensive creation methods."""
        features = []

        try:
            # Short-term vs long-term momentum
            short_momentum = close.pct_change(period)
            long_momentum = close.pct_change(period * 2)

            # Momentum divergence (subtraction)
            momentum_div = short_momentum - long_momentum
            features.append(GeneratedFeature(
                name=f"momentum_divergence_{period}",
                feature_type="cross_timeframe",
                formula=f"pct_change({period}) - pct_change({period * 2})",
                parent_features=["close"],
                feature_series=momentum_div,
                utility_score=0.0,
                creation_method="subtract",
                lookback_period=period
            ))

            # Momentum ratio (division)
            momentum_ratio = short_momentum / (long_momentum + 1e-8)
            features.append(GeneratedFeature(
                name=f"momentum_ratio_{period}",
                feature_type="cross_timeframe",
                formula=f"pct_change({period}) / pct_change({period * 2})",
                parent_features=["close"],
                feature_series=momentum_ratio,
                utility_score=0.0,
                creation_method="divide",
                lookback_period=period
            ))

            # Momentum acceleration (difference)
            momentum_accel = short_momentum.diff()
            features.append(GeneratedFeature(
                name=f"momentum_acceleration_{period}",
                feature_type="cross_timeframe",
                formula=f"diff(pct_change({period}))",
                parent_features=["close"],
                feature_series=momentum_accel,
                utility_score=0.0,
                creation_method="diff",
                lookback_period=period
            ))

            # Momentum addition (sum of short and long)
            momentum_sum = short_momentum + long_momentum
            features.append(GeneratedFeature(
                name=f"momentum_sum_{period}",
                feature_type="cross_timeframe",
                formula=f"pct_change({period}) + pct_change({period * 2})",
                parent_features=["close"],
                feature_series=momentum_sum,
                utility_score=0.0,
                creation_method="add",
                lookback_period=period
            ))

            # Momentum multiplication
            momentum_mult = short_momentum * long_momentum
            features.append(GeneratedFeature(
                name=f"momentum_mult_{period}",
                feature_type="cross_timeframe",
                formula=f"pct_change({period}) * pct_change({period * 2})",
                parent_features=["close"],
                feature_series=momentum_mult,
                utility_score=0.0,
                creation_method="multiply",
                lookback_period=period
            ))

            # Log momentum (logarithmic transformation)
            log_momentum = np.log(np.abs(short_momentum) + 1e-8) * np.sign(short_momentum)
            features.append(GeneratedFeature(
                name=f"log_momentum_{period}",
                feature_type="cross_timeframe",
                formula=f"log(abs(pct_change({period})) + 1e-8) * sign(pct_change({period}))",
                parent_features=["close"],
                feature_series=log_momentum,
                utility_score=0.0,
                creation_method="log",
                lookback_period=period
            ))

            # Momentum power (exponential transformation)
            momentum_power = np.power(np.abs(short_momentum) + 1e-8, 0.5) * np.sign(short_momentum)
            features.append(GeneratedFeature(
                name=f"momentum_power_{period}",
                feature_type="cross_timeframe",
                formula=f"pow(abs(pct_change({period})) + 1e-8, 0.5) * sign(pct_change({period}))",
                parent_features=["close"],
                feature_series=momentum_power,
                utility_score=0.0,
                creation_method="power",
                lookback_period=period
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating momentum cross-timeframe features: {e}")
            return []

    def _create_volatility_cross_timeframe_features(
        self,
        close: pd.Series,
        period: int
    ) -> List[GeneratedFeature]:
        """Create volatility-based cross-timeframe features with comprehensive creation methods."""
        features = []

        try:
            # Short-term vs long-term volatility
            short_vol = close.rolling(period).std()
            long_vol = close.rolling(period * 2).std()

            # Volatility ratio (division)
            vol_ratio = short_vol / (long_vol + 1e-8)
            features.append(GeneratedFeature(
                name=f"volatility_ratio_{period}",
                feature_type="cross_timeframe",
                formula=f"std({period}) / std({period * 2})",
                parent_features=["close"],
                feature_series=vol_ratio,
                utility_score=0.0,
                creation_method="divide",
                lookback_period=period
            ))

            # Volatility spread (subtraction)
            vol_spread = short_vol - long_vol
            features.append(GeneratedFeature(
                name=f"volatility_spread_{period}",
                feature_type="cross_timeframe",
                formula=f"std({period}) - std({period * 2})",
                parent_features=["close"],
                feature_series=vol_spread,
                utility_score=0.0,
                creation_method="subtract",
                lookback_period=period
            ))

            # Volatility regime (comparison)
            vol_regime = (short_vol > long_vol).astype(int)
            features.append(GeneratedFeature(
                name=f"volatility_regime_{period}",
                feature_type="cross_timeframe",
                formula=f"std({period}) > std({period * 2})",
                parent_features=["close"],
                feature_series=vol_regime,
                utility_score=0.0,
                creation_method="compare",
                lookback_period=period
            ))

            # Volatility sum (addition)
            vol_sum = short_vol + long_vol
            features.append(GeneratedFeature(
                name=f"volatility_sum_{period}",
                feature_type="cross_timeframe",
                formula=f"std({period}) + std({period * 2})",
                parent_features=["close"],
                feature_series=vol_sum,
                utility_score=0.0,
                creation_method="add",
                lookback_period=period
            ))

            # Volatility multiplication
            vol_mult = short_vol * long_vol
            features.append(GeneratedFeature(
                name=f"volatility_mult_{period}",
                feature_type="cross_timeframe",
                formula=f"std({period}) * std({period * 2})",
                parent_features=["close"],
                feature_series=vol_mult,
                utility_score=0.0,
                creation_method="multiply",
                lookback_period=period
            ))

            # Log volatility (logarithmic transformation)
            log_vol = np.log(short_vol + 1e-8)
            features.append(GeneratedFeature(
                name=f"log_volatility_{period}",
                feature_type="cross_timeframe",
                formula=f"log(std({period}) + 1e-8)",
                parent_features=["close"],
                feature_series=log_vol,
                utility_score=0.0,
                creation_method="log",
                lookback_period=period
            ))

            # Volatility power (exponential transformation)
            vol_power = np.power(short_vol + 1e-8, 0.5)
            features.append(GeneratedFeature(
                name=f"volatility_power_{period}",
                feature_type="cross_timeframe",
                formula=f"pow(std({period}) + 1e-8, 0.5)",
                parent_features=["close"],
                feature_series=vol_power,
                utility_score=0.0,
                creation_method="power",
                lookback_period=period
            ))

            # Volatility difference (rate of change)
            vol_diff = short_vol.diff()
            features.append(GeneratedFeature(
                name=f"volatility_diff_{period}",
                feature_type="cross_timeframe",
                formula=f"diff(std({period}))",
                parent_features=["close"],
                feature_series=vol_diff,
                utility_score=0.0,
                creation_method="diff",
                lookback_period=period
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating volatility cross-timeframe features: {e}")
            return []

    def _create_trend_cross_timeframe_features(
        self,
        close: pd.Series,
        period: int
    ) -> List[GeneratedFeature]:
        """Create trend-based cross-timeframe features."""
        features = []

        try:
            # Short-term vs long-term trend
            short_trend = close.rolling(period).mean()
            long_trend = close.rolling(period * 2).mean()

            # Trend strength
            trend_strength = (close - short_trend) / (short_trend + 1e-8)
            features.append(GeneratedFeature(
                name=f"trend_strength_{period}",
                feature_type="cross_timeframe",
                formula=f"(close - mean({period})) / mean({period})",
                parent_features=["close"],
                feature_series=trend_strength,
                utility_score=0.0,
                creation_method="ratio"
            ))

            # Trend alignment
            trend_alignment = ((close > short_trend) & (short_trend > long_trend)).astype(int)
            features.append(GeneratedFeature(
                name=f"trend_alignment_{period}",
                feature_type="cross_timeframe",
                formula=f"(close > mean({period})) & (mean({period}) > mean({period * 2}))",
                parent_features=["close"],
                feature_series=trend_alignment,
                utility_score=0.0,
                creation_method="logical_and"
            ))

            # Trend divergence
            trend_div = short_trend - long_trend
            features.append(GeneratedFeature(
                name=f"trend_divergence_{period}",
                feature_type="cross_timeframe",
                formula=f"mean({period}) - mean({period * 2})",
                parent_features=["close"],
                feature_series=trend_div,
                utility_score=0.0,
                creation_method="subtract"
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating trend cross-timeframe features: {e}")
            return []

    def _create_volume_cross_timeframe_features(
        self,
        volume: pd.Series,
        period: int
    ) -> List[GeneratedFeature]:
        """Create volume-based cross-timeframe features."""
        features = []

        try:
            # Volume momentum
            vol_momentum = volume.pct_change(period)
            features.append(GeneratedFeature(
                name=f"volume_momentum_{period}",
                feature_type="cross_timeframe",
                formula=f"volume.pct_change({period})",
                parent_features=["volume"],
                feature_series=vol_momentum,
                utility_score=0.0,
                creation_method="pct_change"
            ))

            # Volume vs price correlation
            if hasattr(self, '_last_close_series'):
                vol_price_corr = volume.rolling(period).corr(self._last_close_series)
                features.append(GeneratedFeature(
                    name=f"volume_price_correlation_{period}",
                    feature_type="cross_timeframe",
                    formula=f"volume.corr(close, {period})",
                    parent_features=["volume", "close"],
                    feature_series=vol_price_corr,
                    utility_score=0.0,
                    creation_method="correlation"
                ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating volume cross-timeframe features: {e}")
            return []

    def _create_ohlc_cross_timeframe_features(
        self,
        ohlc: pd.DataFrame,
        period: int
    ) -> List[GeneratedFeature]:
        """Create OHLC-based cross-timeframe features."""
        features = []

        try:
            high, low, close = ohlc['high'], ohlc['low'], ohlc['close']

            # True range cross-timeframe
            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            tr_short = tr.rolling(period).mean()
            tr_long = tr.rolling(period * 2).mean()

            # True range ratio
            tr_ratio = tr_short / (tr_long + 1e-8)
            features.append(GeneratedFeature(
                name=f"true_range_ratio_{period}",
                feature_type="cross_timeframe",
                formula=f"tr_mean({period}) / tr_mean({period * 2})",
                parent_features=["high", "low", "close"],
                feature_series=tr_ratio,
                utility_score=0.0,
                creation_method="divide"
            ))

            # Price position in range
            price_position = (close - low.rolling(period).min()) / (high.rolling(period).max() - low.rolling(period).min() + 1e-8)
            features.append(GeneratedFeature(
                name=f"price_position_{period}",
                feature_type="cross_timeframe",
                formula=f"(close - min(low, {period})) / (max(high, {period}) - min(low, {period}))",
                parent_features=["high", "low", "close"],
                feature_series=price_position,
                utility_score=0.0,
                creation_method="ratio"
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating OHLC cross-timeframe features: {e}")
            return []

    def _generate_interaction_features(
        self,
        base_features: pd.DataFrame,
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate interaction features (2-3 way) with optimized lookback periods."""
        tprint_debug("Generating interaction features")

        features = []

        try:
            feature_names = list(base_features.columns)

            # Generate 2-way interactions
            if 2 in self.config.interaction_orders:
                tprint_debug("Generating 2-way interactions")
                two_way_features = self._generate_two_way_interactions(base_features, targets)
                features.extend(two_way_features)

            # Generate 3-way interactions
            if 3 in self.config.interaction_orders:
                tprint_debug("Generating 3-way interactions")
                three_way_features = self._generate_three_way_interactions(base_features, targets)
                features.extend(three_way_features)

            # Limit to max features
            if len(features) > self.config.max_interaction_features:
                # Sort by utility score and take top features
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:self.config.max_interaction_features]

            return features

        except Exception as e:
            tprint_error(f"❌ Interaction feature generation failed: {e}")
            return []

    def _generate_two_way_interactions(
        self,
        base_features: pd.DataFrame,
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate 2-way interaction features."""
        features = []

        try:
            feature_names = list(base_features.columns)

            # Generate all possible 2-way combinations
            for i, feat1 in enumerate(feature_names):
                for j, feat2 in enumerate(feature_names[i+1:], i+1):
                    # Skip if same feature
                    if feat1 == feat2:
                        continue

                    # Generate different types of interactions
                    interaction_features = self._create_feature_interactions(
                        base_features, feat1, feat2, targets
                    )
                    features.extend(interaction_features)

            return features

        except Exception as e:
            tprint_debug(f"Error generating 2-way interactions: {e}")
            return []

    def _generate_three_way_interactions(
        self,
        base_features: pd.DataFrame,
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate 3-way interaction features."""
        features = []

        try:
            feature_names = list(base_features.columns)

            # Limit to avoid too many combinations
            max_features = min(20, len(feature_names))
            selected_features = feature_names[:max_features]

            # Generate 3-way combinations
            for combo in combinations(selected_features, 3):
                feat1, feat2, feat3 = combo

                # Generate different types of 3-way interactions
                interaction_features = self._create_three_way_feature_interactions(
                    base_features, feat1, feat2, feat3, targets
                )
                features.extend(interaction_features)

            return features

        except Exception as e:
            tprint_debug(f"Error generating 3-way interactions: {e}")
            return []

    def _create_feature_interactions(
        self,
        base_features: pd.DataFrame,
        feat1: str,
        feat2: str,
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Create interaction features between two features using multiple methods."""
        features = []

        try:
            series1 = base_features[feat1]
            series2 = base_features[feat2]

            # Generate interactions using different creation methods
            for method in self.config.creation_methods:
                try:
                    if method == 'add':
                        interaction_series = series1 + series2
                        formula = f"{feat1} + {feat2}"
                    elif method == 'subtract':
                        interaction_series = series1 - series2
                        formula = f"{feat1} - {feat2}"
                    elif method == 'multiply':
                        interaction_series = series1 * series2
                        formula = f"{feat1} * {feat2}"
                    elif method == 'divide':
                        interaction_series = series1 / (series2 + 1e-8)
                        formula = f"{feat1} / ({feat2} + 1e-8)"
                    elif method == 'log':
                        interaction_series = np.log(np.abs(series1) + 1e-8) * np.log(np.abs(series2) + 1e-8)
                        formula = f"log(|{feat1}|) * log(|{feat2}|)"
                    elif method == 'sqrt':
                        interaction_series = np.sqrt(np.abs(series1)) * np.sqrt(np.abs(series2))
                        formula = f"sqrt(|{feat1}|) * sqrt(|{feat2}|)"
                    elif method == 'power':
                        interaction_series = np.power(np.abs(series1), 0.5) * np.power(np.abs(series2), 0.5)
                        formula = f"pow(|{feat1}|, 0.5) * pow(|{feat2}|, 0.5)"
                    elif method == 'ratio':
                        interaction_series = series1 / (series2 + 1e-8) * series2 / (series1 + 1e-8)
                        formula = f"({feat1} / {feat2}) * ({feat2} / {feat1})"
                    elif method == 'log_add':
                        interaction_series = np.log(np.abs(series1) + 1e-8) + np.log(np.abs(series2) + 1e-8)
                        formula = f"log(|{feat1}|) + log(|{feat2}|)"
                    elif method == 'log_subtract':
                        interaction_series = np.log(np.abs(series1) + 1e-8) - np.log(np.abs(series2) + 1e-8)
                        formula = f"log(|{feat1}|) - log(|{feat2}|)"
                    elif method == 'log_divide':
                        interaction_series = np.log(np.abs(series1) + 1e-8) / (np.log(np.abs(series2) + 1e-8) + 1e-8)
                        formula = f"log(|{feat1}|) / log(|{feat2}|)"
                    elif method == 'exp_add':
                        interaction_series = np.exp(series1) + np.exp(series2)
                        formula = f"exp({feat1}) + exp({feat2})"
                    elif method == 'exp_multiply':
                        interaction_series = np.exp(series1) * np.exp(series2)
                        formula = f"exp({feat1}) * exp({feat2})"
                    elif method == 'abs_add':
                        interaction_series = np.abs(series1) + np.abs(series2)
                        formula = f"abs({feat1}) + abs({feat2})"
                    elif method == 'abs_multiply':
                        interaction_series = np.abs(series1) * np.abs(series2)
                        formula = f"abs({feat1}) * abs({feat2})"
                    elif method == 'square_add':
                        interaction_series = np.square(series1) + np.square(series2)
                        formula = f"{feat1}^2 + {feat2}^2"
                    elif method == 'square_multiply':
                        interaction_series = np.square(series1) * np.square(series2)
                        formula = f"{feat1}^2 * {feat2}^2"
                    elif method == 'cube_add':
                        interaction_series = np.power(series1, 3) + np.power(series2, 3)
                        formula = f"{feat1}^3 + {feat2}^3"
                    elif method == 'cube_multiply':
                        interaction_series = np.power(series1, 3) * np.power(series2, 3)
                        formula = f"{feat1}^3 * {feat2}^3"
                    elif method == 'sin_add':
                        interaction_series = np.sin(series1) + np.sin(series2)
                        formula = f"sin({feat1}) + sin({feat2})"
                    elif method == 'cos_multiply':
                        interaction_series = np.cos(series1) * np.cos(series2)
                        formula = f"cos({feat1}) * cos({feat2})"
                    elif method == 'tan_divide':
                        interaction_series = np.tan(series1) / (np.tan(series2) + 1e-8)
                        formula = f"tan({feat1}) / tan({feat2})"
                    else:
                        continue

                    # Create feature
                    feature = GeneratedFeature(
                        name=f"{feat1}_{feat2}_{method}",
                        feature_type="interaction",
                        formula=formula,
                        parent_features=[feat1, feat2],
                        feature_series=interaction_series,
                        utility_score=0.0,
                        creation_method=method,
                        source_features=[
                            {'name': feat1, 'lookback_period': None, 'feature_type': 'base'},
                            {'name': feat2, 'lookback_period': None, 'feature_type': 'base'}
                        ]
                    )

                    # Calculate utility score
                    feature.utility_score = self._calculate_utility_score(interaction_series, targets)
                    feature.metadata.update({
                        'interaction_order': 2,
                        'feature_category': 'interaction',
                        'base_timeframe_minutes': self.config.base_timeframe_minutes
                    })

                    features.append(feature)

                except Exception as e:
                    tprint_debug(f"Error creating {method} interaction between {feat1} and {feat2}: {e}")
                    continue

            return features

        except Exception as e:
            tprint_debug(f"Error creating feature interactions: {e}")
            return []

    def _create_three_way_feature_interactions(
        self,
        base_features: pd.DataFrame,
        feat1: str,
        feat2: str,
        feat3: str,
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Create 3-way interaction features."""
        features = []

        try:
            series1 = base_features[feat1]
            series2 = base_features[feat2]
            series3 = base_features[feat3]

            # Generate 3-way interactions using different methods
            for method in ['multiply', 'add', 'ratio']:
                try:
                    if method == 'multiply':
                        interaction_series = series1 * series2 * series3
                        formula = f"{feat1} * {feat2} * {feat3}"
                    elif method == 'add':
                        interaction_series = series1 + series2 + series3
                        formula = f"{feat1} + {feat2} + {feat3}"
                    elif method == 'ratio':
                        interaction_series = (series1 * series2) / (series3 + 1e-8)
                        formula = f"({feat1} * {feat2}) / ({feat3} + 1e-8)"
                    else:
                        continue

                    # Create feature
                    feature = GeneratedFeature(
                        name=f"{feat1}_{feat2}_{feat3}_{method}",
                        feature_type="interaction",
                        formula=formula,
                        parent_features=[feat1, feat2, feat3],
                        feature_series=interaction_series,
                        utility_score=0.0,
                        creation_method=method,
                        source_features=[
                            {'name': feat1, 'lookback_period': None, 'feature_type': 'base'},
                            {'name': feat2, 'lookback_period': None, 'feature_type': 'base'},
                            {'name': feat3, 'lookback_period': None, 'feature_type': 'base'}
                        ]
                    )

                    # Calculate utility score
                    feature.utility_score = self._calculate_utility_score(interaction_series, targets)
                    feature.metadata.update({
                        'interaction_order': 3,
                        'feature_category': 'interaction',
                        'base_timeframe_minutes': self.config.base_timeframe_minutes
                    })

                    features.append(feature)

                except Exception as e:
                    tprint_debug(f"Error creating 3-way {method} interaction: {e}")
                    continue

            return features

        except Exception as e:
            tprint_debug(f"Error creating 3-way feature interactions: {e}")
            return []

    def _generate_no_features(
        self,
        data: pd.DataFrame,
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate features with optimized lookback periods (no features with lookback optimization)."""
        tprint_debug("Generating no features with optimized lookback periods")

        features = []

        try:
            # Generate features for different lookback periods
            lookback_periods = [5, 10, 20, 30, 50, 100]  # Common lookback periods

            # Price-based no features with lookback optimization
            if 'close' in data.columns:
                close = data['close']

                for period in lookback_periods:
                    if period < len(data) // 4:  # Ensure sufficient data
                        price_features = self._create_price_no_features_with_lookback(close, period)
                        features.extend(price_features)

            # Volume-based no features with lookback optimization
            if 'volume' in data.columns:
                volume = data['volume']

                for period in lookback_periods:
                    if period < len(data) // 4:  # Ensure sufficient data
                        volume_features = self._create_volume_no_features_with_lookback(volume, period)
                        features.extend(volume_features)

            # OHLC-based no features with lookback optimization
            if all(col in data.columns for col in ['high', 'low', 'close']):
                for period in lookback_periods:
                    if period < len(data) // 4:  # Ensure sufficient data
                        ohlc_features = self._create_ohlc_no_features_with_lookback(
                            data[['high', 'low', 'close']], period
                        )
                        features.extend(ohlc_features)

            # Calculate utility scores and optimize lookback periods
            for feature in features:
                feature.utility_score = self._calculate_utility_score(feature.feature_series, targets)
                feature.metadata.update({
                    'feature_category': 'no_feature',
                    'optimization_type': 'lookback_optimized',
                    'base_timeframe_minutes': self.config.base_timeframe_minutes
                })

            # Group features by type and select best lookback period for each
            features = self._optimize_no_feature_lookbacks(features, targets)

            # Limit to max features
            if len(features) > self.config.max_no_features:
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:self.config.max_no_features]

            return features

        except Exception as e:
            tprint_error(f"❌ No features generation failed: {e}")
            return []

    def _create_price_no_features(self, close: pd.Series) -> List[GeneratedFeature]:
        """Create price-based no features."""
        features = []

        try:
            # Price change
            price_change = close.pct_change()
            features.append(GeneratedFeature(
                name="price_change",
                feature_type="no_feature",
                formula="close.pct_change()",
                parent_features=["close"],
                feature_series=price_change,
                utility_score=0.0,
                creation_method="pct_change"
            ))

            # Price log return
            log_return = np.log(close / close.shift(1))
            features.append(GeneratedFeature(
                name="log_return",
                feature_type="no_feature",
                formula="log(close / close.shift(1))",
                parent_features=["close"],
                feature_series=log_return,
                utility_score=0.0,
                creation_method="log"
            ))

            # Price rank
            price_rank = close.rank(pct=True)
            features.append(GeneratedFeature(
                name="price_rank",
                feature_type="no_feature",
                formula="close.rank(pct=True)",
                parent_features=["close"],
                feature_series=price_rank,
                utility_score=0.0,
                creation_method="rank"
            ))

            # Price z-score
            price_zscore = (close - close.mean()) / close.std()
            features.append(GeneratedFeature(
                name="price_zscore",
                feature_type="no_feature",
                formula="(close - close.mean()) / close.std()",
                parent_features=["close"],
                feature_series=price_zscore,
                utility_score=0.0,
                creation_method="zscore"
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating price no features: {e}")
            return []

    def _create_price_no_features_with_lookback(self, close: pd.Series, period: int) -> List[GeneratedFeature]:
        """Create price-based no features with lookback optimization."""
        features = []

        try:
            # Price change with lookback
            price_change = close.pct_change(period)
            features.append(GeneratedFeature(
                name=f"price_change_{period}",
                feature_type="no_feature",
                formula=f"close.pct_change({period})",
                parent_features=["close"],
                feature_series=price_change,
                utility_score=0.0,
                creation_method="pct_change",
                lookback_period=period
            ))

            # Price log return with lookback
            log_return = np.log(close / close.shift(period))
            features.append(GeneratedFeature(
                name=f"log_return_{period}",
                feature_type="no_feature",
                formula=f"log(close / close.shift({period}))",
                parent_features=["close"],
                feature_series=log_return,
                utility_score=0.0,
                creation_method="log",
                lookback_period=period
            ))

            # Price rank with lookback
            price_rank = close.rolling(period).rank(pct=True)
            features.append(GeneratedFeature(
                name=f"price_rank_{period}",
                feature_type="no_feature",
                formula=f"close.rolling({period}).rank(pct=True)",
                parent_features=["close"],
                feature_series=price_rank,
                utility_score=0.0,
                creation_method="rank",
                lookback_period=period
            ))

            # Price z-score with lookback
            rolling_mean = close.rolling(period).mean()
            rolling_std = close.rolling(period).std()
            price_zscore = (close - rolling_mean) / (rolling_std + 1e-8)
            features.append(GeneratedFeature(
                name=f"price_zscore_{period}",
                feature_type="no_feature",
                formula=f"(close - close.rolling({period}).mean()) / close.rolling({period}).std()",
                parent_features=["close"],
                feature_series=price_zscore,
                utility_score=0.0,
                creation_method="zscore",
                lookback_period=period
            ))

            # Price momentum with lookback
            momentum = close / close.shift(period) - 1
            features.append(GeneratedFeature(
                name=f"momentum_{period}",
                feature_type="no_feature",
                formula=f"close / close.shift({period}) - 1",
                parent_features=["close"],
                feature_series=momentum,
                utility_score=0.0,
                creation_method="momentum",
                lookback_period=period
            ))

            # Price volatility with lookback
            volatility = close.rolling(period).std()
            features.append(GeneratedFeature(
                name=f"volatility_{period}",
                feature_type="no_feature",
                formula=f"close.rolling({period}).std()",
                parent_features=["close"],
                feature_series=volatility,
                utility_score=0.0,
                creation_method="volatility",
                lookback_period=period
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating price no features with lookback {period}: {e}")
            return []

    def _create_volume_no_features(self, volume: pd.Series) -> List[GeneratedFeature]:
        """Create volume-based no features."""
        features = []

        try:
            # Volume change
            volume_change = volume.pct_change()
            features.append(GeneratedFeature(
                name="volume_change",
                feature_type="no_feature",
                formula="volume.pct_change()",
                parent_features=["volume"],
                feature_series=volume_change,
                utility_score=0.0,
                creation_method="pct_change"
            ))

            # Volume rank
            volume_rank = volume.rank(pct=True)
            features.append(GeneratedFeature(
                name="volume_rank",
                feature_type="no_feature",
                formula="volume.rank(pct=True)",
                parent_features=["volume"],
                feature_series=volume_rank,
                utility_score=0.0,
                creation_method="rank"
            ))

            # Volume z-score
            volume_zscore = (volume - volume.mean()) / volume.std()
            features.append(GeneratedFeature(
                name="volume_zscore",
                feature_type="no_feature",
                formula="(volume - volume.mean()) / volume.std()",
                parent_features=["volume"],
                feature_series=volume_zscore,
                utility_score=0.0,
                creation_method="zscore"
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating volume no features: {e}")
            return []

    def _create_volume_no_features_with_lookback(self, volume: pd.Series, period: int) -> List[GeneratedFeature]:
        """Create volume-based no features with lookback optimization."""
        features = []

        try:
            # Volume change with lookback
            volume_change = volume.pct_change(period)
            features.append(GeneratedFeature(
                name=f"volume_change_{period}",
                feature_type="no_feature",
                formula=f"volume.pct_change({period})",
                parent_features=["volume"],
                feature_series=volume_change,
                utility_score=0.0,
                creation_method="pct_change",
                lookback_period=period
            ))

            # Volume rank with lookback
            volume_rank = volume.rolling(period).rank(pct=True)
            features.append(GeneratedFeature(
                name=f"volume_rank_{period}",
                feature_type="no_feature",
                formula=f"volume.rolling({period}).rank(pct=True)",
                parent_features=["volume"],
                feature_series=volume_rank,
                utility_score=0.0,
                creation_method="rank",
                lookback_period=period
            ))

            # Volume z-score with lookback
            rolling_mean = volume.rolling(period).mean()
            rolling_std = volume.rolling(period).std()
            volume_zscore = (volume - rolling_mean) / (rolling_std + 1e-8)
            features.append(GeneratedFeature(
                name=f"volume_zscore_{period}",
                feature_type="no_feature",
                formula=f"(volume - volume.rolling({period}).mean()) / volume.rolling({period}).std()",
                parent_features=["volume"],
                feature_series=volume_zscore,
                utility_score=0.0,
                creation_method="zscore",
                lookback_period=period
            ))

            # Volume momentum with lookback
            volume_momentum = volume / volume.shift(period) - 1
            features.append(GeneratedFeature(
                name=f"volume_momentum_{period}",
                feature_type="no_feature",
                formula=f"volume / volume.shift({period}) - 1",
                parent_features=["volume"],
                feature_series=volume_momentum,
                utility_score=0.0,
                creation_method="momentum",
                lookback_period=period
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating volume no features with lookback {period}: {e}")
            return []

    def _create_ohlc_no_features(self, ohlc: pd.DataFrame) -> List[GeneratedFeature]:
        """Create OHLC-based no features."""
        features = []

        try:
            high, low, close = ohlc['high'], ohlc['low'], ohlc['close']

            # True range
            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            features.append(GeneratedFeature(
                name="true_range",
                feature_type="no_feature",
                formula="max(high - low, max(abs(high - close.shift(1)), abs(low - close.shift(1))))",
                parent_features=["high", "low", "close"],
                feature_series=tr,
                utility_score=0.0,
                creation_method="max"
            ))

            # Price position in daily range
            daily_range = high - low
            price_position = (close - low) / (daily_range + 1e-8)
            features.append(GeneratedFeature(
                name="price_position_daily",
                feature_type="no_feature",
                formula="(close - low) / (high - low)",
                parent_features=["high", "low", "close"],
                feature_series=price_position,
                utility_score=0.0,
                creation_method="ratio"
            ))

            # Body size
            body_size = abs(close - ohlc.get('open', close.shift(1)))
            features.append(GeneratedFeature(
                name="body_size",
                feature_type="no_feature",
                formula="abs(close - open)",
                parent_features=["open", "close"],
                feature_series=body_size,
                utility_score=0.0,
                creation_method="abs"
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating OHLC no features: {e}")
            return []

    def _create_ohlc_no_features_with_lookback(self, ohlc: pd.DataFrame, period: int) -> List[GeneratedFeature]:
        """Create OHLC-based no features with lookback optimization."""
        features = []

        try:
            high, low, close = ohlc['high'], ohlc['low'], ohlc['close']

            # True range with lookback
            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            tr_rolling = tr.rolling(period).mean()
            features.append(GeneratedFeature(
                name=f"true_range_{period}",
                feature_type="no_feature",
                formula=f"true_range.rolling({period}).mean()",
                parent_features=["high", "low", "close"],
                feature_series=tr_rolling,
                utility_score=0.0,
                creation_method="rolling_mean",
                lookback_period=period
            ))

            # Price position in range with lookback
            high_max = high.rolling(period).max()
            low_min = low.rolling(period).min()
            price_position = (close - low_min) / (high_max - low_min + 1e-8)
            features.append(GeneratedFeature(
                name=f"price_position_{period}",
                feature_type="no_feature",
                formula=f"(close - low.rolling({period}).min()) / (high.rolling({period}).max() - low.rolling({period}).min())",
                parent_features=["high", "low", "close"],
                feature_series=price_position,
                utility_score=0.0,
                creation_method="ratio",
                lookback_period=period
            ))

            # Range volatility with lookback
            range_vol = (high - low).rolling(period).std()
            features.append(GeneratedFeature(
                name=f"range_volatility_{period}",
                feature_type="no_feature",
                formula=f"(high - low).rolling({period}).std()",
                parent_features=["high", "low"],
                feature_series=range_vol,
                utility_score=0.0,
                creation_method="volatility",
                lookback_period=period
            ))

            # High-low ratio with lookback
            hl_ratio = high.rolling(period).max() / (low.rolling(period).min() + 1e-8)
            features.append(GeneratedFeature(
                name=f"hl_ratio_{period}",
                feature_type="no_feature",
                formula=f"high.rolling({period}).max() / low.rolling({period}).min()",
                parent_features=["high", "low"],
                feature_series=hl_ratio,
                utility_score=0.0,
                creation_method="ratio",
                lookback_period=period
            ))

            return features

        except Exception as e:
            tprint_debug(f"Error creating OHLC no features with lookback {period}: {e}")
            return []

    def _optimize_no_feature_lookbacks(self, features: List[GeneratedFeature], targets: Optional[pd.Series] = None) -> List[GeneratedFeature]:
        """Optimize lookback periods for no features by selecting the most informative period for each feature type."""
        if not features:
            return features

        try:
            # Group features by base name (without period suffix)
            feature_groups = {}
            for feature in features:
                # Extract base name by removing period suffix
                base_name = feature.name.rsplit('_', 1)[0] if '_' in feature.name and feature.name.split('_')[-1].isdigit() else feature.name
                if base_name not in feature_groups:
                    feature_groups[base_name] = []
                feature_groups[base_name].append(feature)

            optimized_features = []

            # For each group, select the feature with the highest utility score
            for base_name, group_features in feature_groups.items():
                if len(group_features) == 1:
                    optimized_features.append(group_features[0])
                else:
                    # Sort by utility score and take the best one
                    best_feature = max(group_features, key=lambda x: x.utility_score)
                    optimized_features.append(best_feature)

                    # Also add the second best if it's significantly different
                    if len(group_features) > 1:
                        group_features.sort(key=lambda x: x.utility_score, reverse=True)
                        second_best = group_features[1]
                        if second_best.utility_score > best_feature.utility_score * 0.8:  # 80% threshold
                            optimized_features.append(second_best)

            tprint_debug(f"Optimized {len(features)} features to {len(optimized_features)} features")
            return optimized_features

        except Exception as e:
            tprint_debug(f"Error optimizing no feature lookbacks: {e}")
            return features

    def _calculate_utility_score(
        self,
        feature_series: pd.Series,
        targets: Optional[pd.Series] = None
    ) -> float:
        """Calculate utility score for a feature."""
        try:
            if targets is None:
                # Use variance as utility score
                return float(feature_series.var())

            # Align series
            aligned_feature = feature_series.dropna()
            aligned_targets = targets.reindex(aligned_feature.index).dropna()

            if len(aligned_feature) < 10 or len(aligned_targets) < 10:
                return 0.0

            # Calculate correlation
            correlation = np.corrcoef(aligned_feature, aligned_targets)[0, 1]

            if np.isnan(correlation):
                return 0.0

            return abs(correlation)

        except Exception as e:
            tprint_debug(f"Error calculating utility score: {e}")
            return 0.0

    def _generate_comparison_features(
        self,
        data: pd.DataFrame,
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate comparison features between base, VWAP-based, volatility-adjusted, and z-score volume adjusted features."""
        features = []

        try:
            # Ensure we have OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in data.columns]
            if not available_cols:
                return features

            # Generate comparison features for different periods
            periods = [5, 10, 15, 30, 60, 120, 240]  # minutes

            for period in periods:
                # Skip if period is too large for data
                if period >= len(data) // 4:
                    continue

                # Generate different types of comparison features
                period_features = self._generate_period_comparison_features(
                    data, period, available_cols, targets
                )
                features.extend(period_features)

            # Limit to max features
            if len(features) > self.config.max_comparison_features:
                # Sort by utility score and take top features
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:self.config.max_comparison_features]

            return features

        except Exception as e:
            tprint_error(f"❌ Comparison feature generation failed: {e}")
            return []

    def _generate_period_comparison_features(
        self,
        data: pd.DataFrame,
        period: int,
        available_cols: List[str],
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate comparison features for a specific period."""
        features = []

        try:
            if 'close' in available_cols and 'volume' in available_cols:
                close = data['close']
                volume = data['volume']

                # Base features
                base_sma = close.rolling(period).mean()
                base_vol = close.rolling(period).std()

                # VWAP-based features
                vwap = (close * volume).rolling(period).sum() / volume.rolling(period).sum()
                vwap_sma = vwap.rolling(period).mean()
                vwap_vol = vwap.rolling(period).std()

                # Volatility-adjusted features
                vol_adjusted_close = close / (base_vol + 1e-8)
                vol_adjusted_sma = vol_adjusted_close.rolling(period).mean()
                vol_adjusted_vol = vol_adjusted_close.rolling(period).std()

                # Z-score volume adjusted features
                volume_zscore = (volume - volume.rolling(period).mean()) / (volume.rolling(period).std() + 1e-8)
                zscore_vol_adjusted_close = close * volume_zscore
                zscore_vol_adjusted_sma = zscore_vol_adjusted_close.rolling(period).mean()
                zscore_vol_adjusted_vol = zscore_vol_adjusted_close.rolling(period).std()

                # Generate comparison features
                comparison_types = [
                    ('base', base_sma, base_vol),
                    ('vwap', vwap_sma, vwap_vol),
                    ('volatility_adjusted', vol_adjusted_sma, vol_adjusted_vol),
                    ('zscore_volume', zscore_vol_adjusted_sma, zscore_vol_adjusted_vol)
                ]

                # Compare each type with others
                for i, (type1, sma1, vol1) in enumerate(comparison_types):
                    for j, (type2, sma2, vol2) in enumerate(comparison_types[i+1:], i+1):
                        # SMA comparison
                        sma_ratio = sma1 / (sma2 + 1e-8)
                        features.append(GeneratedFeature(
                            name=f"sma_ratio_{type1}_vs_{type2}_{period}",
                            feature_type="comparison",
                            formula=f"sma_{type1}({period}) / sma_{type2}({period})",
                            parent_features=["close", "volume"],
                            feature_series=sma_ratio,
                            utility_score=0.0,
                            lookback_period=period,
                            base_timeframe_minutes=self.config.base_timeframe_minutes,
                            comparison_type=f"{type1}_vs_{type2}",
                            metadata={
                                'feature_category': 'comparison',
                                'comparison_types': [type1, type2],
                                'base_timeframe_minutes': self.config.base_timeframe_minutes,
                                'period_in_base_units': period // self.config.base_timeframe_minutes
                            }
                        ))

                        # Volatility comparison
                        vol_ratio = vol1 / (vol2 + 1e-8)
                        features.append(GeneratedFeature(
                            name=f"vol_ratio_{type1}_vs_{type2}_{period}",
                            feature_type="comparison",
                            formula=f"vol_{type1}({period}) / vol_{type2}({period})",
                            parent_features=["close", "volume"],
                            feature_series=vol_ratio,
                            utility_score=0.0,
                            lookback_period=period,
                            base_timeframe_minutes=self.config.base_timeframe_minutes,
                            comparison_type=f"{type1}_vs_{type2}",
                            metadata={
                                'feature_category': 'comparison',
                                'comparison_types': [type1, type2],
                                'base_timeframe_minutes': self.config.base_timeframe_minutes,
                                'period_in_base_units': period // self.config.base_timeframe_minutes
                            }
                        ))

                        # Divergence features
                        sma_divergence = sma1 - sma2
                        features.append(GeneratedFeature(
                            name=f"sma_divergence_{type1}_vs_{type2}_{period}",
                            feature_type="comparison",
                            formula=f"sma_{type1}({period}) - sma_{type2}({period})",
                            parent_features=["close", "volume"],
                            feature_series=sma_divergence,
                            utility_score=0.0,
                            lookback_period=period,
                            base_timeframe_minutes=self.config.base_timeframe_minutes,
                            comparison_type=f"{type1}_vs_{type2}",
                            metadata={
                                'feature_category': 'comparison',
                                'comparison_types': [type1, type2],
                                'base_timeframe_minutes': self.config.base_timeframe_minutes,
                                'period_in_base_units': period // self.config.base_timeframe_minutes
                            }
                        ))

                # Calculate utility scores for all features
                for feature in features:
                    feature.utility_score = self._calculate_utility_score(feature.feature_series, targets)

            return features

        except Exception as e:
            tprint_debug(f"Error generating period {period} comparison features: {e}")
            return []

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_execution_time': 0.0,
            'cross_timeframe_features_generated': 0,
            'interaction_features_generated': 0,
            'no_features_generated': 0,
            'comparison_features_generated': 0,
            'vectorbt_operations': 0
        }

def create_enhanced_feature_generator(config: Optional[FeatureGenerationConfig] = None) -> EnhancedFeatureGenerator:
    """Create an enhanced feature generator with default configuration."""
    return EnhancedFeatureGenerator(config)
