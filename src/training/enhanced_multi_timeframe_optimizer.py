"""
Enhanced Multi-Timeframe Optimizer

This module enhances multi-timeframe and cross-timeframe features by using
optimized lookback periods from the matrix optimization system instead of
fixed periods. It integrates with the existing matrix optimization results
to provide more effective multi-timeframe analysis.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class OptimizedTimeframeConfig:
    """Configuration for optimized timeframe features."""
    base_timeframes: List[str] = None
    optimized_periods: Dict[str, List[int]] = None
    cross_timeframe_enabled: bool = True
    regime_specific: bool = True
    quality_thresholds: Dict[str, float] = None

class EnhancedMultiTimeframeOptimizer:
    """
    Enhanced Multi-Timeframe Optimizer that uses optimized lookback periods
    from the matrix optimization system instead of fixed periods.
    """

    def __init__(self, config: OptimizedTimeframeConfig, matrix_optimization_results: Dict[str, Any] = None):
        self.config = config
        self.matrix_results = matrix_optimization_results or {}
        self.logger = logging.getLogger(__name__)

        # Extract optimized periods from matrix results
        self.optimized_periods = self._extract_optimized_periods()

    def _extract_optimized_periods(self) -> Dict[str, List[int]]:
        """Extract optimized lookback periods from matrix optimization results."""
        optimized_periods = {}

        if not self.matrix_results:
            self.logger.warning("⚠️ No matrix optimization results provided, using default periods")
            return self._get_default_periods()

        # Extract periods from matrix optimization results
        if "diverse_lookback_periods" in self.matrix_results:
            for feature_name, result in self.matrix_results["diverse_lookback_periods"].items():
                if "selected_periods" in result:
                    optimized_periods[feature_name] = result["selected_periods"]

        # Also check for regime-specific periods
        if "regime_specific_periods" in self.matrix_results:
            for regime, regime_results in self.matrix_results["regime_specific_periods"].items():
                for feature_name, result in regime_results.items():
                    if "selected_periods" in result:
                        key = f"{regime}_{feature_name}"
                        optimized_periods[key] = result["selected_periods"]

        self.logger.info(f"✅ Extracted {len(optimized_periods)} optimized period sets")
        return optimized_periods

    async def _generate_base_timeframe_features(
        self,
        data: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """Generate base timeframe features using optimized periods."""
        features = {}

        for timeframe in self.config.base_timeframes:
            self.logger.info(f"🔍 Generating {timeframe} timeframe features with optimized periods...")

            # Resample data to timeframe
            resampled_data = self._resample_data(data, timeframe)
            if resampled_data is None or resampled_data.empty:
                continue

            # Generate features for each optimized indicator
            for indicator_name, periods in self.optimized_periods.items():
                for period in periods:
                    feature_name = f"{indicator_name}_{period}_{timeframe}"

                    try:
                        # Calculate indicator with optimized period
                        indicator_value = self._calculate_indicator(
                            resampled_data, indicator_name, period
                        )

                        if indicator_value is not None and not indicator_value.isna().all():
                            # Align back to original timeframe
                            aligned_value = self._align_to_base_timeframe(
                                indicator_value, data.index, timeframe
                            )

                            if aligned_value is not None:
                                features[feature_name] = aligned_value

                    except Exception as e:
                        self.logger.debug(f"⚠️ Failed to calculate {feature_name}: {e}")
                        continue

        return features

    async def _generate_optimized_cross_timeframe_features(
        self,
        data: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """Generate cross-timeframe features using optimized periods."""
        features = {}

        # Get optimized periods for cross-timeframe analysis
        cross_periods = self._get_cross_timeframe_periods()

        self.logger.info(f"🔍 Generating cross-timeframe features with {len(cross_periods)} optimized period pairs...")

        for i, (period1, period2) in enumerate(cross_periods):
            if period1 >= period2:
                continue

            try:
                # 1. Cross-timeframe momentum features
                momentum_features = self._calculate_cross_momentum_features(data, period1, period2)
                features.update(momentum_features)

                # 2. Cross-timeframe volatility features
                volatility_features = self._calculate_cross_volatility_features(data, period1, period2)
                features.update(volatility_features)

                # 3. Cross-timeframe volume features
                if "volume" in data.columns:
                    volume_features = self._calculate_cross_volume_features(data, period1, period2)
                    features.update(volume_features)

                # 4. Cross-timeframe price range features
                range_features = self._calculate_cross_range_features(data, period1, period2)
                features.update(range_features)

            except Exception as e:
                self.logger.debug(f"⚠️ Failed to calculate cross-timeframe features for {period1}-{period2}: {e}")
                continue

        return features

    def _select_diverse_period_pairs(self, period_pairs: List[Tuple[int, int]], max_pairs: int = 20) -> List[Tuple[int, int]]:
        """Select diverse period pairs to avoid redundancy."""
        if len(period_pairs) <= max_pairs:
            return period_pairs

        # Sort by diversity (larger difference first)
        sorted_pairs = sorted(period_pairs, key=lambda x: x[1] - x[0], reverse=True)

        # Select diverse pairs
        selected_pairs = []
        used_periods = set()

        for period1, period2 in sorted_pairs:
            if len(selected_pairs) >= max_pairs:
                break

            # Check if this pair adds diversity
            if period1 not in used_periods or period2 not in used_periods:
                selected_pairs.append((period1, period2))
                used_periods.add(period1)
                used_periods.add(period2)

        return selected_pairs

    def _calculate_cross_momentum_features(
        self,
        data: pd.DataFrame,
        period1: int,
        period2: int
    ) -> Dict[str, Any]:
        """Calculate cross-timeframe momentum features."""
        features = {}
        close = data["close"]

        # Price momentum differences
        momentum1 = close.pct_change(period1)
        momentum2 = close.pct_change(period2)

        # Momentum difference
        momentum_diff = momentum1 - momentum2
        if momentum_diff.var() > 1e-12:
            features[f"momentum_diff_{period1}_{period2}"] = momentum_diff

        # Momentum ratio
        momentum_ratio = momentum1 / (momentum2 + 1e-8)
        if momentum_ratio.var() > 1e-12:
            features[f"momentum_ratio_{period1}_{period2}"] = momentum_ratio

        # High-Low momentum
        if len(close) >= max(period1, period2) * 2:
            high = data["high"]
            low = data["low"]

            hl_momentum1 = (high.rolling(period1).max() - low.rolling(period1).min()) / (close.rolling(period1).mean() + 1e-8)
            hl_momentum2 = (high.rolling(period2).max() - low.rolling(period2).min()) / (close.rolling(period2).mean() + 1e-8)

            hl_diff = hl_momentum1 - hl_momentum2
            if hl_diff.var() > 1e-12:
                features[f"hl_momentum_diff_{period1}_{period2}"] = hl_diff

        return features

    def _calculate_cross_volatility_features(
        self,
        data: pd.DataFrame,
        period1: int,
        period2: int
    ) -> Dict[str, Any]:
        """Calculate cross-timeframe volatility features."""
        features = {}
        close = data["close"]
        returns = close.pct_change().fillna(0)

        # Volatility calculations
        vol1 = returns.rolling(period1).std()
        vol2 = returns.rolling(period2).std()

        # Volatility ratio
        vol_ratio = vol1 / (vol2 + 1e-8)
        if vol_ratio.var() > 1e-12:
            features[f"volatility_ratio_{period1}_{period2}"] = vol_ratio

        # Volatility difference
        vol_diff = vol1 - vol2
        if vol_diff.var() > 1e-12:
            features[f"volatility_diff_{period1}_{period2}"] = vol_diff

        # Volatility of volatility
        if len(vol1) >= 20:
            vol_of_vol = (vol1 - vol2).rolling(20).std()
            if vol_of_vol.var() > 1e-12:
                features[f"volatility_of_vol_{period1}_{period2}"] = vol_of_vol

        return features

    def _calculate_cross_volume_features(
        self,
        data: pd.DataFrame,
        period1: int,
        period2: int
    ) -> Dict[str, Any]:
        """Calculate cross-timeframe volume features."""
        features = {}
        volume = data["volume"]

        # Volume averages
        vol1 = volume.rolling(period1).mean()
        vol2 = volume.rolling(period2).mean()

        # Volume ratio
        vol_ratio = vol1 / (vol2 + 1e-8)
        if vol_ratio.var() > 1e-12:
            features[f"volume_ratio_{period1}_{period2}"] = vol_ratio

        # Volume difference
        vol_diff = vol1 - vol2
        if vol_diff.var() > 1e-12:
            features[f"volume_diff_{period1}_{period2}"] = vol_diff

        # Volume momentum
        vol_momentum1 = volume.pct_change(period1)
        vol_momentum2 = volume.pct_change(period2)
        vol_momentum_diff = vol_momentum1 - vol_momentum2

        if vol_momentum_diff.var() > 1e-12:
            features[f"volume_momentum_diff_{period1}_{period2}"] = vol_momentum_diff

        return features

    def _calculate_cross_range_features(
        self,
        data: pd.DataFrame,
        period1: int,
        period2: int
    ) -> Dict[str, Any]:
        """Calculate cross-timeframe price range features."""
        features = {}
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Price ranges
        range1 = (high.rolling(period1).max() - low.rolling(period1).min()) / (close.rolling(period1).mean() + 1e-8)
        range2 = (high.rolling(period2).max() - low.rolling(period2).min()) / (close.rolling(period2).mean() + 1e-8)

        # Range ratio
        range_ratio = range1 / (range2 + 1e-8)
        if range_ratio.var() > 1e-12:
            features[f"range_ratio_{period1}_{period2}"] = range_ratio

        # Range difference
        range_diff = range1 - range2
        if range_diff.var() > 1e-12:
            features[f"range_diff_{period1}_{period2}"] = range_diff

        return features

    async def _generate_regime_specific_features(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        regime_labels: pd.Series
    ) -> Dict[str, Any]:
        """Generate regime-specific multi-timeframe features."""
        features = {}

        unique_regimes = regime_labels.unique()
        self.logger.info(f"🔍 Generating regime-specific features for {len(unique_regimes)} regimes...")

        for regime in unique_regimes:
            if pd.isna(regime):
                continue

            # Create regime mask
            regime_mask = regime_labels == regime
            regime_data = data[regime_mask].copy()

            if len(regime_data) < 100:  # Need sufficient data
                continue

            # Get regime-specific optimized periods
            regime_periods = self._get_regime_specific_periods(regime)

            # Generate features for this regime
            for indicator_name, periods in regime_periods.items():
                for period in periods:
                    feature_name = f"regime_{regime}_{indicator_name}_{period}"

                    try:
                        indicator_value = self._calculate_indicator(
                            regime_data, indicator_name, period
                        )

                        if indicator_value is not None:
                            # Align to full dataset
                            full_series = pd.Series(index=data.index, dtype=float)
                            full_series[regime_mask] = indicator_value
                            full_series = full_series.fillna(method="ffill").fillna(0)

                            features[feature_name] = full_series

                    except Exception as e:
                        self.logger.debug(f"⚠️ Failed to calculate {feature_name}: {e}")
                        continue

        return features

    async def _validate_and_filter_features(
        self,
        features: Dict[str, Any],
        target: pd.Series
    ) -> Dict[str, Any]:
        """Validate and filter features based on quality thresholds."""
        filtered_features = {}

        for feature_name, feature_series in features.items():
            if not isinstance(feature_series, pd.Series):
                continue

            # Check for sufficient variance
            if feature_series.var() < 1e-12:
                continue

            # Check for correlation with target
            correlation = abs(feature_series.corr(target))
            if correlation < self.config.quality_thresholds["min_correlation"]:
                continue

            # Check for excessive correlation with existing features
            max_corr = 0
            for existing_name, existing_series in filtered_features.items():
                if isinstance(existing_series, pd.Series):
                    corr = abs(feature_series.corr(existing_series))
                    max_corr = max(max_corr, corr)

            if max_corr > self.config.quality_thresholds["max_correlation"]:
                continue

            filtered_features[feature_name] = feature_series

        self.logger.info(f"✅ Filtered {len(features)} features to {len(filtered_features)} high-quality features")
        return filtered_features

    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Resample data to specified timeframe."""
        try:
            if timeframe == "1m":
                return data

            # Convert timeframe to pandas offset
            timeframe_map = {
                "5m": "5T",
                "15m": "15T",
                "30m": "30T",
                "1h": "1H"
            }

            offset = timeframe_map.get(timeframe)
            if offset is None:
                return None

            # Resample OHLCV data
            resampled = data.resample(offset).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            return resampled.dropna()

        except Exception as e:
            self.logger.debug(f"⚠️ Failed to resample to {timeframe}: {e}")
            return None

    def _calculate_indicator(self, data: pd.DataFrame, indicator_name: str, period: int) -> Optional[pd.Series]:
        """Calculate technical indicator with specified period."""
        try:
            if indicator_name == "RSI":
                return self._calculate_rsi(data["close"], period)
            elif indicator_name == "SMA":
                return data["close"].rolling(period).mean()
            elif indicator_name == "EMA":
                return data["close"].ewm(span=period).mean()
            elif indicator_name == "ATR":
                return self._calculate_atr(data, period)
            elif indicator_name == "VWAP":
                return self._calculate_vwap(data, period)
            elif indicator_name == "VWAP_Momentum":
                vwap = self._calculate_vwap(data, period)
                return vwap / vwap.shift(period) - 1
            elif indicator_name == "VWAP_Volatility":
                vwap = self._calculate_vwap(data, period)
                returns = vwap.pct_change()
                return returns.rolling(period).std()
            else:
                # Add more indicators as needed
                return None

        except Exception as e:
            self.logger.debug(f"⚠️ Failed to calculate {indicator_name}: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with specified period."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR with specified period."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def _calculate_vwap(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP with specified period."""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        vwap = (typical_price * data["volume"]).rolling(window=period).sum() / data["volume"].rolling(window=period).sum()
        return vwap

    def _align_to_base_timeframe(self, series: pd.Series, target_index: pd.DatetimeIndex, timeframe: str) -> Optional[pd.Series]:
        """Align series to base timeframe (1m)."""
        try:
            if timeframe == "1m":
                return series

            # Forward fill and align to target index
            aligned = series.reindex(target_index, method="ffill")
            aligned = aligned.fillna(method="bfill").fillna(0)
            return aligned

        except Exception as e:
            self.logger.debug(f"⚠️ Failed to align {timeframe} series: {e}")
            return None
