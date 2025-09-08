
import pandas as pd
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from ..standardized_parquet_handler import standardized_parquet_handler

"""Feature engineering components.

This module contains specialized components for feature engineering
including technical indicators, interactions, regime-aware features, and S/R features.
"""
import numpy as np
import logging
import typing
from typing import Dict, List, Any, Optional, Tuple

class TechnicalIndicatorEngine:
    """Engine for creating technical indicators."""

    def __init__(self, lookback_periods: Dict[str, List[int]]) -> None:
        """Initialize technical indicator engine.
        
        Args:
            lookback_periods: Dictionary of lookback periods by type
        """
        self.lookback_periods = lookback_periods
        self.logger = system_logger.getChild('TechnicalIndicatorEngine')
        self.default_periods = {'short': [5, 10, 20], 'medium': [50, 100], 'long': [200]}

    def apply_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply technical indicators to data.

        Args:
            data: Market data

        Returns:
            Data with technical indicators
        """
        data = self._add_moving_averages(data)
        data = self._add_price_channels(data)
        data = self._add_momentum_indicators(data)
        data = self._add_volatility_indicators(data)
        if 'volume' in data.columns:
            data = self._add_volume_indicators(data)
        data = self._add_pattern_features(data)
        data = self._add_entropy_features(data)
        return data
    @log_all_calls

    def _add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add moving average indicators."""
        returns = data['close'].pct_change()
        all_periods = []
        for period_type in ['short', 'medium', 'long']:
            periods = self.lookback_periods.get(period_type, self.default_periods[period_type])
            all_periods.extend(periods)

        # Standard price-based moving averages
        for period in all_periods:
            data[f'feature_sma_{period}'] = data['close'].rolling(period).mean()
            data[f'feature_sma_{period}_ratio'] = data['close'] / data[f'feature_sma_{period}']
            data[f'feature_ema_{period}'] = data['close'].ewm(span = period).mean()
            data[f'feature_ema_{period}_ratio'] = data['close'] / data[f'feature_ema_{period}']

        # Returns-based moving averages
        for period in all_periods:
            data[f'feature_sma_returns_{period}'] = returns.rolling(period).mean()
            data[f'feature_ema_returns_{period}'] = returns.ewm(span = period).mean()

        # Acceleration-based moving averages (second derivatives)
        for period in all_periods:
            sma_returns = data[f'feature_sma_returns_{period}']
            ema_returns = data[f'feature_ema_returns_{period}']
            data[f'feature_sma_returns_acceleration_{period}'] = sma_returns.diff().rolling(5).mean()
            data[f'feature_ema_returns_acceleration_{period}'] = ema_returns.diff().rolling(5).mean()

        short_periods = self.lookback_periods.get('short', self.default_periods['short'])
        medium_periods = self.lookback_periods.get('medium', self.default_periods['medium'])
        if short_periods and medium_periods:
            short_ma = data[f'feature_sma_{short_periods[0]}']
            long_ma = data[f'feature_sma_{medium_periods[0]}']
            data['feature_ma_crossover'] = (short_ma > long_ma).astype(int)
            data['feature_ma_spread'] = (short_ma - long_ma) / long_ma

            # Returns-based crossover signals
            short_ma_returns = data[f'feature_sma_returns_{short_periods[0]}']
            long_ma_returns = data[f'feature_sma_returns_{medium_periods[0]}']
            data['feature_ma_returns_crossover'] = (short_ma_returns > long_ma_returns).astype(int)
            data['feature_ma_returns_spread'] = (short_ma_returns - long_ma_returns)

        return data
    @log_all_calls

    def _add_price_channels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price channel indicators."""
        # Calculate VWAP first
        vwap = self._calculate_vwap(data)
        data['feature_vwap'] = vwap
        data['feature_price_vwap_ratio'] = data['close'] / vwap
        data['feature_price_vwap_deviation'] = (data['close'] - vwap) / vwap * 100

        # Bollinger Bands on price
        for period in [20, 50]:
            sma = data['close'].rolling(period).mean()
            std = data['close'].rolling(period).std()
            data[f'feature_bb_upper_{period}'] = sma + 2 * std
            data[f'feature_bb_lower_{period}'] = sma - 2 * std
            data[f'feature_bb_width_{period}'] = (data[f'feature_bb_upper_{period}'] - data[f'feature_bb_lower_{period}']) / sma
            data[f'feature_bb_position_{period}'] = (data['close'] - data[f'feature_bb_lower_{period}']) / (data[f'feature_bb_upper_{period}'] - data[f'feature_bb_lower_{period}'])

        # Bollinger Bands on VWAP (instead of price-based momentum)
        for period in [20, 50]:
            vwap_sma = vwap.rolling(period).mean()
            vwap_std = vwap.rolling(period).std()
            data[f'feature_bb_vwap_upper_{period}'] = vwap_sma + 2 * vwap_std
            data[f'feature_bb_vwap_lower_{period}'] = vwap_sma - 2 * vwap_std
            data[f'feature_bb_vwap_position_{period}'] = (vwap - data[f'feature_bb_vwap_lower_{period}']) / (data[f'feature_bb_vwap_upper_{period}'] - data[f'feature_bb_vwap_lower_{period}'])

        if all((col in data.columns for col in ['high', 'low'])):
            for period in [20]:
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                ema = typical_price.ewm(span = period).mean()
                atr = self._calculate_atr(data, period)
                data[f'feature_kc_upper_{period}'] = ema + 2 * atr
                data[f'feature_kc_lower_{period}'] = ema - 2 * atr
                data[f'feature_kc_position_{period}'] = (data['close'] - data[f'feature_kc_lower_{period}']) / (data[f'feature_kc_upper_{period}'] - data[f'feature_kc_lower_{period}'])

                # VWAP-based Keltner Channels
                vwap_ema = vwap.ewm(span = period).mean()
                data[f'feature_kc_vwap_upper_{period}'] = vwap_ema + 2 * atr
                data[f'feature_kc_vwap_lower_{period}'] = vwap_ema - 2 * atr
                data[f'feature_kc_vwap_position_{period}'] = (vwap - data[f'feature_kc_vwap_lower_{period}']) / (data[f'feature_kc_vwap_upper_{period}'] - data[f'feature_kc_vwap_lower_{period}'])

        # VWAP-based momentum features (replacing price-based momentum)
        data['feature_vwap_momentum_5'] = vwap.pct_change(5)
        data['feature_vwap_momentum_10'] = vwap.pct_change(10)
        data['feature_vwap_momentum_20'] = vwap.pct_change(20)
        data['feature_vwap_acceleration'] = data['feature_vwap_momentum_5'].diff()

        return data
    @log_all_calls

    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # Calculate returns first
        returns = data['close'].pct_change()

        # Standard price-based indicators
        for period in [14, 21]:
            data[f'feature_rsi_{period}'] = self._calculate_rsi(data['close'], period)
        data['feature_macd'] = data['close'].ewm(span = 12).mean() - data['close'].ewm(span = 26).mean()
        data['feature_macd_signal'] = data['feature_macd'].ewm(span = 9).mean()
        data['feature_macd_histogram'] = data['feature_macd'] - data['feature_macd_signal']
        for period in [10, 20]:
            data[f'feature_roc_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period) * 100
        if all((col in data.columns for col in ['high', 'low'])):
            for period in [14]:
                lowest_low = data['low'].rolling(period).min()
                highest_high = data['high'].rolling(period).max()
                data[f'feature_stoch_k_{period}'] = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
                data[f'feature_stoch_d_{period}'] = data[f'feature_stoch_k_{period}'].rolling(3).mean()

        # Returns-based indicators
        for period in [14, 21]:
            data[f'feature_rsi_returns_{period}'] = self._calculate_rsi(returns, period)
        data['feature_macd_returns'] = returns.ewm(span = 12).mean() - returns.ewm(span = 26).mean()
        data['feature_macd_returns_signal'] = data['feature_macd_returns'].ewm(span = 9).mean()
        data['feature_macd_returns_histogram'] = data['feature_macd_returns'] - data['feature_macd_returns_signal']
        for period in [10, 20]:
            data[f'feature_roc_returns_{period}'] = (returns - returns.shift(period)) / (returns.shift(period) + 1e-10) * 100

        # Acceleration-based indicators (second derivatives)
        for period in [14, 21]:
            rsi_returns = data[f'feature_rsi_returns_{period}']
            data[f'feature_rsi_acceleration_{period}'] = rsi_returns.diff().rolling(5).mean()
        macd_returns = data['feature_macd_returns']
        data['feature_macd_returns_acceleration'] = macd_returns.diff().rolling(5).mean()
        for period in [10, 20]:
            roc_returns = data[f'feature_roc_returns_{period}']
            data[f'feature_roc_returns_acceleration_{period}'] = roc_returns.diff().rolling(5).mean()

        return data
    @log_all_calls

    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        returns = data['close'].pct_change()

        # Standard volatility indicators
        for period in [10, 20, 50]:
            data[f'feature_volatility_{period}'] = returns.rolling(period).std()
            data[f'feature_volatility_ratio_{period}'] = data[f'feature_volatility_{period}'] / data[f'feature_volatility_{period}'].rolling(period).mean()
        if all((col in data.columns for col in ['high', 'low'])):
            for period in [14, 20]:
                data[f'feature_atr_{period}'] = self._calculate_atr(data, period)
                data[f'feature_atr_ratio_{period}'] = data[f'feature_atr_{period}'] / data['close']
        if all((col in data.columns for col in ['high', 'low'])):
            hl_ratio = np.log(data['high'] / data['low'])
            # Vectorized Parkinson volatility calculation
            hl_ratio_squared = hl_ratio ** 2
            parkinson_vol = np.sqrt(hl_ratio_squared.rolling(20).mean() / (4 * np.log(2)))
            data['feature_parkinson_vol'] = parkinson_vol

        # Returns-based volatility indicators
        for period in [10, 20, 50]:
            returns_volatility = returns.rolling(period).std()
            data[f'feature_volatility_returns_{period}'] = returns_volatility
            data[f'feature_volatility_returns_ratio_{period}'] = returns_volatility / returns_volatility.rolling(period).mean()

        # Acceleration-based volatility indicators
        for period in [10, 20, 50]:
            volatility_returns = data[f'feature_volatility_returns_{period}']
            data[f'feature_volatility_returns_acceleration_{period}'] = volatility_returns.diff().rolling(5).mean()
            if all((col in data.columns for col in ['high', 'low'])):
                atr_series = data[f'feature_atr_{period}']
                data[f'feature_atr_acceleration_{period}'] = atr_series.diff().rolling(5).mean()

        return data
    @log_all_calls

    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        returns = data['close'].pct_change()
        vwap = data.get('feature_vwap', self._calculate_vwap(data))

        for period in [10, 20]:
            data[f'feature_volume_sma_{period}'] = data['volume'].rolling(period).mean()
            data[f'feature_volume_ratio_{period}'] = data['volume'] / data[f'feature_volume_sma_{period}']
        data['feature_obv'] = (np.sign(data['close'].diff()) * data['volume']).cumsum()
        data['feature_obv_sma'] = data['feature_obv'].rolling(20).mean()
        data['feature_vpt'] = (data['close'].diff() / data['close'].shift() * data['volume']).cumsum()

        # Returns-based volume indicators
        for period in [10, 20]:
            volume_returns = data['volume'].pct_change().rolling(period).mean()
            data[f'feature_volume_returns_{period}'] = volume_returns
            data[f'feature_volume_returns_ratio_{period}'] = volume_returns / volume_returns.rolling(period).mean()

        # VWAP-based volume indicators
        if not vwap.empty:
            data['feature_vwap_volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            data['feature_vwap_volume_momentum'] = data['feature_vwap_volume_ratio'].pct_change(5)

        # Acceleration-based volume indicators
        for period in [10, 20]:
            volume_returns = data[f'feature_volume_returns_{period}']
            data[f'feature_volume_returns_acceleration_{period}'] = volume_returns.diff().rolling(5).mean()

        if all((col in data.columns for col in ['high', 'low'])):
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_price * data['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
            positive_mf = positive_flow.rolling(14).sum()
            negative_mf = negative_flow.rolling(14).sum()
            mfi_ratio = positive_mf / negative_mf
            data['feature_mfi'] = 100 - 100 / (1 + mfi_ratio)

            # VWAP-based MFI
            if not vwap.empty:
                vwap_money_flow = vwap * data['volume']
                vwap_positive_flow = vwap_money_flow.where(vwap > vwap.shift(), 0)
                vwap_negative_flow = vwap_money_flow.where(vwap < vwap.shift(), 0)
                vwap_positive_mf = vwap_positive_flow.rolling(14).sum()
                vwap_negative_mf = vwap_negative_flow.rolling(14).sum()
                vwap_mfi_ratio = vwap_positive_mf / vwap_negative_mf
                data['feature_mfi_vwap'] = 100 - 100 / (1 + vwap_mfi_ratio)

        return data
    @log_all_calls

    def _add_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        vwap = data.get('feature_vwap', self._calculate_vwap(data))

        # Standard price-based patterns
        data['feature_higher_high'] = ((data['high'] > data['high'].shift(1)) & (data['high'].shift(1) > data['high'].shift(2))).astype(int)
        data['feature_lower_low'] = ((data['low'] < data['low'].shift(1)) & (data['low'].shift(1) < data['low'].shift(2))).astype(int)

        # Returns-based pattern features
        returns = data['close'].pct_change()
        data['feature_higher_returns'] = ((returns > returns.shift(1)) & (returns.shift(1) > returns.shift(2))).astype(int)
        data['feature_lower_returns'] = ((returns < returns.shift(1)) & (returns.shift(1) < returns.shift(2))).astype(int)

        for period in [20, 50]:
            data[f'feature_resistance_{period}'] = data['high'].rolling(period).max()
            data[f'feature_support_{period}'] = data['low'].rolling(period).min()
            data[f'feature_sr_position_{period}'] = (data['close'] - data[f'feature_support_{period}']) / (data[f'feature_resistance_{period}'] - data[f'feature_support_{period}'])

        # VWAP-based pattern features
        if not vwap.empty:
            data['feature_vwap_higher'] = ((vwap > vwap.shift(1)) & (vwap.shift(1) > vwap.shift(2))).astype(int)
            data['feature_vwap_lower'] = ((vwap < vwap.shift(1)) & (vwap.shift(1) < vwap.shift(2))).astype(int)
            for period in [20, 50]:
                data[f'feature_vwap_resistance_{period}'] = vwap.rolling(period).max()
                data[f'feature_vwap_support_{period}'] = vwap.rolling(period).min()
                data[f'feature_vwap_sr_position_{period}'] = (vwap - data[f'feature_vwap_support_{period}']) / (data[f'feature_vwap_resistance_{period}'] - data[f'feature_vwap_support_{period}'])

        return data
    @log_all_calls

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)
    @log_all_calls

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
        return true_range.rolling(period).mean()

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        if 'volume' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap
    @log_all_calls

    def _add_entropy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add entropy-based features."""
        # Shannon entropy for returns
        returns = data['close'].pct_change()
        for period in [20, 50, 100]:
            if len(returns.dropna()) >= period:
                data[f'feature_shannon_entropy_returns_{period}'] = self._calculate_shannon_entropy(
                    returns, period
                )

        # Sample entropy for price complexity
        for period in [20, 50]:
            if len(data['close'].dropna()) >= period + 10:  # Need sufficient data
                data[f'feature_sample_entropy_price_{period}'] = self._calculate_sample_entropy(
                    data['close'], period
                )

        # Permutation entropy for ordinal patterns
        for period in [20, 50]:
            if len(data['close'].dropna()) >= period:
                data[f'feature_permutation_entropy_price_{period}'] = self._calculate_permutation_entropy(
                    data['close'], period
                )

        # Spectral entropy for frequency analysis
        for period in [20, 50]:
            if len(data['close'].dropna()) >= period:
                data[f'feature_spectral_entropy_price_{period}'] = self._calculate_spectral_entropy(
                    data['close'], period
                )

        # Conditional entropy for volatility regimes
        for period in [20, 50]:
            if len(returns.dropna()) >= period:
                data[f'feature_conditional_entropy_volatility_{period}'] = self._calculate_conditional_entropy(
                    returns, period
                )

        # Entropy-based volatility measures
        for period in [20, 50]:
            if len(returns.dropna()) >= period:
                data[f'feature_entropy_volatility_{period}'] = self._calculate_entropy_volatility(
                    returns, period
                )

        # Multi-scale entropy features
        for period in [20, 50]:
            if len(data['close'].dropna()) >= period * 2:
                data[f'feature_multiscale_entropy_price_{period}'] = self._calculate_multiscale_entropy(
                    data['close'], period
                )

        return data
    @log_all_calls

    def _calculate_shannon_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Shannon entropy for a rolling window using vectorized operations."""
        # Vectorized entropy calculation using numba for performance
        try:
            from numba import jit

            @jit(nopython=True)
            def vectorized_entropy_calc(values, window_size):
                """Vectorized entropy calculation."""
                n = len(values)
                result = np.zeros(n)
                result[:] = np.nan  # Initialize with NaN

                for i in range(window_size - 1, n):
                    window_start = i - window_size + 1
                    window_data = values[window_start:i + 1]

                    # Remove NaN values
                    valid_data = window_data[~np.isnan(window_data)]
                    if len(valid_data) < 2:
                        result[i] = 0.0
                        continue

                    # Create histogram with adaptive bins
                    n_bins = min(int(np.sqrt(len(valid_data))), 20)
                    if n_bins < 2:
                        result[i] = 0.0
                        continue

                    # Calculate histogram
                    hist_min = np.min(valid_data)
                    hist_max = np.max(valid_data)
                    if hist_min == hist_max:
                        result[i] = 0.0
                        continue

                    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)
                    hist = np.zeros(n_bins)

                    for val in valid_data:
                        bin_idx = np.searchsorted(bin_edges, val, side='left') - 1
                        if 0 <= bin_idx < n_bins:
                            hist[bin_idx] += 1

                    # Normalize to density
                    hist = hist / np.sum(hist)
                    hist = hist[hist > 0]  # Remove zeros

                    if len(hist) > 0:
                        # Calculate entropy
                        result[i] = -np.sum(hist * np.log2(hist))
                    else:
                        result[i] = 0.0

                return result

            # Apply vectorized calculation
            values = series.values
            entropy_values = vectorized_entropy_calc(values, window)

            return pd.Series(entropy_values, index=series.index)

        except ImportError:
            # Fallback to original method if numba not available
            self.logger.warning("Numba not available, falling back to pandas apply for entropy calculation")
            def entropy_calc(x):
                x = x.dropna()
                if len(x) < 2:
                    return 0.0

                n_bins = min(int(np.sqrt(len(x))), 20)
                if n_bins < 2:
                    return 0.0

                hist, _ = np.histogram(x, bins=n_bins, density=True)
                hist = hist[hist > 0]

                if len(hist) == 0:
                    return 0.0

                entropy = -np.sum(hist * np.log2(hist))
                return entropy

            return series.rolling(window).apply(entropy_calc, raw=False)
    @log_all_calls

    def _calculate_sample_entropy(self, series: pd.Series, window: int, m: int = 2, r: float = 0.2) -> pd.Series:
        """Calculate sample entropy for complexity analysis."""
        def sample_entropy_calc(x):
            x = x.dropna()
            if len(x) < m + 1:
                return 0.0

            # Normalize data
            x = (x - np.mean(x)) / (np.std(x) + 1e-10)

            def _phi(m_val):
                """Calculate phi for given embedding dimension."""
                patterns = []
                for i in range(len(x) - m_val + 1):
                    pattern = tuple(x[i:i+m_val])
                    patterns.append(pattern)

                if not patterns:
                    return 0

                # Count matches within tolerance r
                matches = 0
                for i in range(len(patterns)):
                    for j in range(len(patterns)):
                        if i != j:
                            # Chebyshev distance
                            dist = max(abs(a - b) for a, b in zip(patterns[i], patterns[j]))
                            if dist <= r:
                                matches += 1

                if matches == 0:
                    return 0

                return matches / (len(patterns) * (len(patterns) - 1))

            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)

            if phi_m == 0 or phi_m1 == 0:
                return 0.0

            return -np.log(phi_m1 / phi_m)

            # Try Numba-optimized version first, fallback to pandas apply
            try:
                return self._calculate_sample_entropy_numba_optimized(series, window, m, r)
            except (ImportError, Exception) as e:
                self.logger.debug(f"Numba sample entropy failed: {e}, using pandas apply")
                return series.rolling(window).apply(sample_entropy_calc, raw=False)
    @log_all_calls

    def _calculate_permutation_entropy(self, series: pd.Series, window: int, order: int = 3) -> pd.Series:
        """Calculate permutation entropy for ordinal pattern analysis."""
        def permutation_entropy_calc(x):
            x = x.dropna()
            if len(x) < order + 1:
                return 0.0

            # Create ordinal patterns
            patterns = []
            for i in range(len(x) - order + 1):
                window_data = x[i:i+order]
                # Get permutation indices
                perm = np.argsort(window_data)
                patterns.append(tuple(perm))

            if not patterns:
                return 0.0

            # Count pattern frequencies
            unique_patterns = {}
            for pattern in patterns:
                unique_patterns[pattern] = unique_patterns.get(pattern, 0) + 1

            # Calculate entropy
            n_patterns = len(patterns)
            entropy = 0.0
            for count in unique_patterns.values():
                p = count / n_patterns
                if p > 0:
                    entropy -= p * np.log2(p)

            # Normalize by maximum possible entropy
            try:
                # Use math.factorial for better compatibility
                import math
                max_entropy = np.log2(math.factorial(min(order, 10)))  # Cap at 10 to avoid overflow
            except (OverflowError, ValueError):
                max_entropy = np.log2(2 ** order)  # Approximation for large factorials
            return entropy / max_entropy if max_entropy > 0 else 0.0

        return series.rolling(window).apply(permutation_entropy_calc, raw=False)
    @log_all_calls

    def _calculate_spectral_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate spectral entropy using Fourier transform."""
        def spectral_entropy_calc(x):
            x = x.dropna()
            if len(x) < 4:
                return 0.0

            # Remove linear trend
            x_detrended = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))

            # Apply FFT
            fft = np.fft.fft(x_detrended)
            power_spectrum = np.abs(fft) ** 2

            # Only use positive frequencies
            power_spectrum = power_spectrum[:len(power_spectrum)//2]

            # Normalize power spectrum
            power_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-10)
            power_spectrum = power_spectrum[power_spectrum > 0]

            if len(power_spectrum) == 0:
                return 0.0

            # Calculate spectral entropy
            entropy = -np.sum(power_spectrum * np.log2(power_spectrum))
            return entropy

        return series.rolling(window).apply(spectral_entropy_calc, raw=False)
    @log_all_calls

    def _calculate_conditional_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate conditional entropy for volatility regimes."""
        def conditional_entropy_calc(x):
            x = x.dropna()
            if len(x) < 4:
                return 0.0

            # Create volatility regimes based on standard deviations
            mean_val = np.mean(x)
            std_val = np.std(x)

            # Classify into regimes
            regimes = []
            for val in x:
                if val > mean_val + std_val:
                    regimes.append(2)  # High volatility
                elif val < mean_val - std_val:
                    regimes.append(0)  # Low volatility
                else:
                    regimes.append(1)  # Normal volatility

            # Calculate transition probabilities
            transitions = {}
            for i in range(len(regimes) - 1):
                transition = (regimes[i], regimes[i+1])
                transitions[transition] = transitions.get(transition, 0) + 1

            # Calculate conditional entropy
            entropy = 0.0
            for (from_state, to_state), count in transitions.items():
                # Find total transitions from this state
                total_from_state = sum(c for (f, t), c in transitions.items() if f == from_state)
                if total_from_state > 0:
                    p_transition = count / total_from_state
                    if p_transition > 0:
                        entropy -= p_transition * np.log2(p_transition)

            return entropy

        return series.rolling(window).apply(conditional_entropy_calc, raw=False)
    @log_all_calls

    def _calculate_entropy_volatility(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate entropy-based volatility measure."""
        def entropy_volatility_calc(x):
            x = x.dropna()
            if len(x) < 4:
                return 0.0

            # Calculate absolute returns
            abs_returns = np.abs(x)

            # Create volatility bins
            bins = np.linspace(0, np.max(abs_returns) + 1e-10, 11)
            hist, _ = np.histogram(abs_returns, bins=bins, density=True)
            hist = hist[hist > 0]

            if len(hist) == 0:
                return 0.0

            # Calculate entropy of volatility distribution
            entropy = -np.sum(hist * np.log2(hist))

            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(bins) - 1)
            return entropy / max_entropy if max_entropy > 0 else 0.0

        return series.rolling(window).apply(entropy_volatility_calc, raw=False)
    @log_all_calls

    def _calculate_multiscale_entropy(self, series: pd.Series, window: int, scales: List[int] = None) -> pd.Series:
        """Calculate multi-scale entropy for complexity analysis."""
        if scales is None:
            scales = [1, 2, 3, 4]

        def multiscale_entropy_calc(x):
            x = x.dropna()
            if len(x) < max(scales) * 10:
                return 0.0

            # Calculate sample entropy at different scales
            scale_entropies = []
            for scale in scales:
                if len(x) < scale * 2:
                    continue

                # Coarse-grain the time series
                coarse_grained = []
                for i in range(0, len(x) - scale + 1, scale):
                    coarse_grained.append(np.mean(x[i:i+scale]))

                if len(coarse_grained) < 3:
                    continue

                # Calculate sample entropy for this scale
                coarse_series = pd.Series(coarse_grained)
                entropy = self._calculate_sample_entropy(coarse_series, len(coarse_series), m=1, r=0.2)
                scale_entropies.append(entropy.iloc[-1] if not entropy.empty else 0.0)

            # Return average entropy across scales
            return np.mean(scale_entropies) if scale_entropies else 0.0

        return series.rolling(window).apply(multiscale_entropy_calc, raw=False)


class EntropyFeatureEngine:
    """
    Engine for creating entropy-based features for market complexity analysis.

    This engine provides various entropy measures that can capture different aspects of
    market dynamics:

    1. Shannon Entropy: Measures information content and randomness in returns
    2. Sample Entropy: Detects complexity and predictability in price series
    3. Permutation Entropy: Analyzes ordinal patterns in price movements
    4. Spectral Entropy: Examines frequency domain characteristics
    5. Conditional Entropy: Models volatility regime transitions
    6. Multi-scale Entropy: Analyzes complexity across different time scales

    These features are particularly useful for:
    - Detecting market regime changes
    - Identifying trending vs. ranging markets
    - Measuring market efficiency/volatility
    - Predicting potential breakout points
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize entropy feature engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('EntropyFeatureEngine')

        # Extract entropy parameters from config
        entropy_params = config.get('feature_engineering_parameters', {}).get('entropy_parameters', {})
        self.shannon_windows = entropy_params.get('shannon_windows', [20, 50, 100])
        self.sample_windows = entropy_params.get('sample_windows', [20, 50])
        self.permutation_windows = entropy_params.get('permutation_windows', [20, 50])
        self.spectral_windows = entropy_params.get('spectral_windows', [20, 50])
        self.conditional_windows = entropy_params.get('conditional_windows', [20, 50])
        self.multiscale_windows = entropy_params.get('multiscale_windows', [20, 50])
        self.sample_entropy_m = entropy_params.get('sample_entropy_m', 2)
        self.sample_entropy_r = entropy_params.get('sample_entropy_r', 0.2)
        self.permutation_order = entropy_params.get('permutation_order', 3)

    def create_entropy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive entropy-based features.

        Args:
            data: Market data with price columns

        Returns:
            Data with entropy features
        """
        if data.empty or 'close' not in data.columns:
            return data

        # Shannon entropy features
        data = self._add_shannon_entropy_features(data)

        # Sample entropy features
        data = self._add_sample_entropy_features(data)

        # Permutation entropy features
        data = self._add_permutation_entropy_features(data)

        # Spectral entropy features
        data = self._add_spectral_entropy_features(data)

        # Advanced entropy features
        data = self._add_advanced_entropy_features(data)

        self.logger.info(f"✅ Generated {len([col for col in data.columns if 'entropy_' in col])} entropy features")

        return data

    def _add_shannon_entropy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Shannon entropy features."""
        returns = data['close'].pct_change()

        # Returns entropy
        for period in self.shannon_windows:
            if len(returns.dropna()) >= period:
                data[f'entropy_shannon_returns_{period}'] = self._calculate_shannon_entropy(
                    returns, period
                )

        # Price entropy
        for period in self.shannon_windows:
            if len(data['close'].dropna()) >= period:
                data[f'entropy_shannon_price_{period}'] = self._calculate_shannon_entropy(
                    data['close'], period
                )

        # Volume entropy (if available)
        if 'volume' in data.columns:
            volume_returns = data['volume'].pct_change()
            for period in self.shannon_windows:
                if len(volume_returns.dropna()) >= period:
                    data[f'entropy_shannon_volume_{period}'] = self._calculate_shannon_entropy(
                        volume_returns, period
                    )

        return data

    def _add_sample_entropy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sample entropy features."""
        for period in self.sample_windows:
            if len(data['close'].dropna()) >= period + 10:
                data[f'entropy_sample_price_{period}'] = self._calculate_sample_entropy(
                    data['close'], period, m=self.sample_entropy_m, r=self.sample_entropy_r
                )

        # Returns sample entropy
        returns = data['close'].pct_change()
        for period in self.sample_windows:
            if len(returns.dropna()) >= period + 10:
                data[f'entropy_sample_returns_{period}'] = self._calculate_sample_entropy(
                    returns, period, m=self.sample_entropy_m, r=self.sample_entropy_r * 0.75  # Slightly tighter for returns
                )

        return data

    def _add_permutation_entropy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add permutation entropy features."""
        for period in self.permutation_windows:
            if len(data['close'].dropna()) >= period:
                data[f'entropy_permutation_price_{period}'] = self._calculate_permutation_entropy(
                    data['close'], period, order=self.permutation_order
                )

        # Returns permutation entropy
        returns = data['close'].pct_change()
        for period in self.permutation_windows:
            if len(returns.dropna()) >= period:
                data[f'entropy_permutation_returns_{period}'] = self._calculate_permutation_entropy(
                    returns, period, order=self.permutation_order
                )

        return data

    def _add_spectral_entropy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add spectral entropy features."""
        for period in self.spectral_windows:
            if len(data['close'].dropna()) >= period:
                data[f'entropy_spectral_price_{period}'] = self._calculate_spectral_entropy(
                    data['close'], period
                )

        # Returns spectral entropy
        returns = data['close'].pct_change()
        for period in self.spectral_windows:
            if len(returns.dropna()) >= period:
                data[f'entropy_spectral_returns_{period}'] = self._calculate_spectral_entropy(
                    returns, period
                )

        return data

    def _add_advanced_entropy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced entropy-based features."""
        returns = data['close'].pct_change()

        # Conditional entropy
        for period in self.conditional_windows:
            if len(returns.dropna()) >= period:
                data[f'entropy_conditional_volatility_{period}'] = self._calculate_conditional_entropy(
                    returns, period
                )

        # Entropy-based volatility
        for period in self.conditional_windows:
            if len(returns.dropna()) >= period:
                data[f'entropy_volatility_measure_{period}'] = self._calculate_entropy_volatility(
                    returns, period
                )

        # Multi-scale entropy
        for period in self.multiscale_windows:
            if len(data['close'].dropna()) >= period * 2:
                data[f'entropy_multiscale_price_{period}'] = self._calculate_multiscale_entropy(
                    data['close'], period
                )

        # Entropy trends
        entropy_cols = [col for col in data.columns if 'entropy_' in col and '_20' in col]
        for col in entropy_cols:
            if col in data.columns:
                data[f'{col}_trend'] = data[col].pct_change(5)
                data[f'{col}_acceleration'] = data[f'{col}_trend'].diff()

        return data

    # Include all the entropy calculation methods from TechnicalIndicatorEngine
    def _calculate_shannon_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Shannon entropy for a rolling window."""
        def entropy_calc(x):
            x = x.dropna()
            if len(x) < 2:
                return 0.0

            n_bins = min(int(np.sqrt(len(x))), 20)
            if n_bins < 2:
                return 0.0

            hist, _ = np.histogram(x, bins=n_bins, density=True)
            hist = hist[hist > 0]

            if len(hist) == 0:
                return 0.0

            entropy = -np.sum(hist * np.log2(hist))
            return entropy

        return series.rolling(window).apply(entropy_calc, raw=False)

    def _calculate_sample_entropy(self, series: pd.Series, window: int, m: int = 2, r: float = 0.2) -> pd.Series:
        """Calculate sample entropy for complexity analysis."""
        def sample_entropy_calc(x):
            x = x.dropna()
            if len(x) < m + 1:
                return 0.0

            x = (x - np.mean(x)) / (np.std(x) + 1e-10)

            def _phi(m_val):
                patterns = []
                for i in range(len(x) - m_val + 1):
                    pattern = tuple(x[i:i+m_val])
                    patterns.append(pattern)

                if not patterns:
                    return 0

                matches = 0
                for i in range(len(patterns)):
                    for j in range(len(patterns)):
                        if i != j:
                            dist = max(abs(a - b) for a, b in zip(patterns[i], patterns[j]))
                            if dist <= r:
                                matches += 1

                if matches == 0:
                    return 0

                return matches / (len(patterns) * (len(patterns) - 1))

            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)

            if phi_m == 0 or phi_m1 == 0:
                return 0.0

            return -np.log(phi_m1 / phi_m)

        return series.rolling(window).apply(sample_entropy_calc, raw=False)

    def _calculate_permutation_entropy(self, series: pd.Series, window: int, order: int = 3) -> pd.Series:
        """Calculate permutation entropy for ordinal pattern analysis."""
        def permutation_entropy_calc(x):
            x = x.dropna()
            if len(x) < order + 1:
                return 0.0

            patterns = []
            for i in range(len(x) - order + 1):
                window_data = x[i:i+order]
                perm = np.argsort(window_data)
                patterns.append(tuple(perm))

            if not patterns:
                return 0.0

            unique_patterns = {}
            for pattern in patterns:
                unique_patterns[pattern] = unique_patterns.get(pattern, 0) + 1

            n_patterns = len(patterns)
            entropy = 0.0
            for count in unique_patterns.values():
                p = count / n_patterns
                if p > 0:
                    entropy -= p * np.log2(p)

            try:
                # Use math.factorial for better compatibility
                import math
                max_entropy = np.log2(math.factorial(min(order, 10)))  # Cap at 10 to avoid overflow
            except (OverflowError, ValueError):
                max_entropy = np.log2(2 ** order)  # Approximation for large factorials
            return entropy / max_entropy if max_entropy > 0 else 0.0

        return series.rolling(window).apply(permutation_entropy_calc, raw=False)

    def _calculate_spectral_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate spectral entropy using Fourier transform."""
        def spectral_entropy_calc(x):
            x = x.dropna()
            if len(x) < 4:
                return 0.0

            x_detrended = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))

            fft = np.fft.fft(x_detrended)
            power_spectrum = np.abs(fft) ** 2
            power_spectrum = power_spectrum[:len(power_spectrum)//2]

            power_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-10)
            power_spectrum = power_spectrum[power_spectrum > 0]

            if len(power_spectrum) == 0:
                return 0.0

            entropy = -np.sum(power_spectrum * np.log2(power_spectrum))
            return entropy

        return series.rolling(window).apply(spectral_entropy_calc, raw=False)

    def _calculate_conditional_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate conditional entropy for volatility regimes."""
        def conditional_entropy_calc(x):
            x = x.dropna()
            if len(x) < 4:
                return 0.0

            mean_val = np.mean(x)
            std_val = np.std(x)

            regimes = []
            for val in x:
                if val > mean_val + std_val:
                    regimes.append(2)
                elif val < mean_val - std_val:
                    regimes.append(0)
                else:
                    regimes.append(1)

            transitions = {}
            for i in range(len(regimes) - 1):
                transition = (regimes[i], regimes[i+1])
                transitions[transition] = transitions.get(transition, 0) + 1

            entropy = 0.0
            for (from_state, to_state), count in transitions.items():
                total_from_state = sum(c for (f, t), c in transitions.items() if f == from_state)
                if total_from_state > 0:
                    p_transition = count / total_from_state
                    if p_transition > 0:
                        entropy -= p_transition * np.log2(p_transition)

            return entropy

        return series.rolling(window).apply(conditional_entropy_calc, raw=False)

    def _calculate_entropy_volatility(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate entropy-based volatility measure."""
        def entropy_volatility_calc(x):
            x = x.dropna()
            if len(x) < 4:
                return 0.0

            abs_returns = np.abs(x)
            bins = np.linspace(0, np.max(abs_returns) + 1e-10, 11)
            hist, _ = np.histogram(abs_returns, bins=bins, density=True)
            hist = hist[hist > 0]

            if len(hist) == 0:
                return 0.0

            entropy = -np.sum(hist * np.log2(hist))
            max_entropy = np.log2(len(bins) - 1)
            return entropy / max_entropy if max_entropy > 0 else 0.0

        return series.rolling(window).apply(entropy_volatility_calc, raw=False)

    def _calculate_multiscale_entropy(self, series: pd.Series, window: int, scales: List[int] = None) -> pd.Series:
        """Calculate multi-scale entropy for complexity analysis."""
        if scales is None:
            scales = [1, 2, 3, 4]

        def multiscale_entropy_calc(x):
            x = x.dropna()
            if len(x) < max(scales) * 10:
                return 0.0

            scale_entropies = []
            for scale in scales:
                if len(x) < scale * 2:
                    continue

                coarse_grained = []
                for i in range(0, len(x) - scale + 1, scale):
                    coarse_grained.append(np.mean(x[i:i+scale]))

                if len(coarse_grained) < 3:
                    continue

                coarse_series = pd.Series(coarse_grained)
                entropy = self._calculate_sample_entropy(coarse_series, len(coarse_series), m=1, r=0.2)
                scale_entropies.append(entropy.iloc[-1] if not entropy.empty else 0.0)

            return np.mean(scale_entropies) if scale_entropies else 0.0

        return series.rolling(window).apply(multiscale_entropy_calc, raw=False)


class FeatureInteractionEngine:
    """Engine for creating feature interactions."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize interaction engine.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.logger = system_logger.getChild('FeatureInteractionEngine')

    async def create_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create feature interactions.
        
        Args:
            data: Data with features
            
        Returns:
            Data with interaction features
        """
        data = self._create_price_volume_interactions(data)
        data = self._create_momentum_volatility_interactions(data)
        data = self._create_indicator_interactions(data)
        data = self._create_cross_timeframe_interactions(data)
        return data
    @log_all_calls

    def _create_price_volume_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price-volume interaction features."""
        if 'volume' in data.columns:
            if 'feature_returns' not in data.columns:
                data['feature_returns'] = data['close'].pct_change()
            data['feature_price_volume_interaction'] = data['feature_returns'] * np.log1p(data['volume'])
            if 'feature_volume_ratio_20' in data.columns:
                data['feature_volume_weighted_momentum'] = data['feature_returns'].rolling(10).mean() * data['feature_volume_ratio_20']
        return data
    @log_all_calls

    def _create_momentum_volatility_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-volatility interactions."""
        if 'feature_rsi_14' in data.columns and 'feature_volatility_20' in data.columns:
            data['feature_rsi_volatility_interaction'] = data['feature_rsi_14'] * data['feature_volatility_20']
        if 'feature_macd' in data.columns and 'feature_atr_14' in data.columns:
            data['feature_macd_atr_interaction'] = data['feature_macd'] / (data['feature_atr_14'] + 1e-08)
        return data
    @log_all_calls

    def _create_indicator_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interactions between technical indicators."""
        if 'feature_rsi_14' in data.columns and 'feature_stoch_k_14' in data.columns:
            data['feature_rsi_stoch_interaction'] = data['feature_rsi_14'] * data['feature_stoch_k_14'] / 100
        if all((col in data.columns for col in ['feature_bb_width_20', 'feature_kc_upper_20'])):
            bb_width = data['feature_bb_width_20']
            kc_width = (data['feature_kc_upper_20'] - data.get('feature_kc_lower_20', 0)) / data['close']
            data['feature_bb_kc_squeeze'] = bb_width / (kc_width + 1e-08)
        ma_pairs = [('feature_sma_10', 'feature_sma_50'), ('feature_ema_10', 'feature_ema_50'), ('feature_sma_20', 'feature_sma_100')]
        for fast_ma, slow_ma in ma_pairs:
            if fast_ma in data.columns and slow_ma in data.columns:
                interaction_name = f'feature_{fast_ma}_{slow_ma}_divergence'
                data[interaction_name] = (data[fast_ma] - data[slow_ma]) / data[slow_ma]
        return data
    @log_all_calls

    def _create_cross_timeframe_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create cross-timeframe interactions."""
        if 'feature_roc_10' in data.columns and 'feature_roc_20' in data.columns:
            data['feature_momentum_divergence'] = data['feature_roc_10'] - data['feature_roc_20']
        vol_pairs = [('feature_volatility_10', 'feature_volatility_50'), ('feature_volatility_20', 'feature_volatility_50')]
        for short_vol, long_vol in vol_pairs:
            if short_vol in data.columns and long_vol in data.columns:
                interaction_name = f'{short_vol}_{long_vol}_ratio'
                data[interaction_name] = data[short_vol] / (data[long_vol] + 1e-08)
        return data

class RegimeAwareFeatureEngine:
    """Engine for creating regime-aware features."""

    def __init__(self) -> None:
        """Initialize regime-aware feature engine."""
        self.logger = system_logger.getChild('RegimeAwareFeatureEngine')

    def create_regime_features(self, data: pd.DataFrame, regime_characteristics: Dict[str, Any]) -> pd.DataFrame:
        """Create regime-aware features.
        
        Args:
            data: Data with regime labels
            regime_characteristics: Characteristics of each regime
            
        Returns:
            Data with regime features
        """
        if 'regime_label' not in data.columns:
            return data
        regime_dummies = pd.get_dummies(data['regime_label'], prefix='regime')
        data = pd.concat([data, regime_dummies], axis = 1)
        data = self._add_regime_transition_features(data)
        data = self._add_regime_statistics(data, regime_characteristics)
        data = self._add_regime_persistence_features(data)
        return data
    @log_all_calls

    def _add_regime_transition_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime transition features."""
        data['feature_regime_changed'] = (data['regime_label'] != data['regime_label'].shift(1)).astype(int)
        data['feature_prev_regime'] = data['regime_label'].shift(1)
        regime_groups = (data['regime_label'] != data['regime_label'].shift()).cumsum()
        data['feature_time_in_regime'] = data.groupby(regime_groups).cumcount()
        regime_durations = data.groupby(regime_groups).size()
        data['feature_regime_duration'] = data.groupby(regime_groups).transform(lambda x: np.arange(len(x), 0, -1))
        return data
    @log_all_calls

    def _add_regime_statistics(self, data: pd.DataFrame, regime_characteristics: Dict[str, Any]) -> pd.DataFrame:
        """Add regime-specific statistics."""
        for regime_key, chars in regime_characteristics.items():
            if isinstance(chars, dict) and regime_key.startswith('regime_'):
                regime_id = int(regime_key.split('_')[1])
                if 'volatility_20_mean' in chars:
                    mask = data['regime_label'] == regime_id
                    data.loc[mask, 'feature_regime_volatility'] = chars['volatility_20_mean']
                if 'returns_mean' in chars:
                    mask = data['regime_label'] == regime_id
                    data.loc[mask, 'feature_regime_return_expectation'] = chars['returns_mean']
        if 'feature_regime_volatility' in data.columns:
            data['feature_regime_volatility'].fillna(data['feature_volatility_20'].mean(), inplace = True)
        return data
    @log_all_calls

    def _add_regime_persistence_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime persistence features."""
        window = 20
        regime_changes = data['feature_regime_changed'].rolling(window).sum()
        data['feature_regime_stability'] = 1 - regime_changes / window
        regime_counts = data['regime_label'].rolling(window).apply(lambda x: pd.Series(x).value_counts().iloc[0] / len(x))
        data['feature_regime_concentration'] = regime_counts
        return data


class DataResampler:
    """Engine for resampling data to multiple timeframes."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize data resampler.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('DataResampler')
        self.timeframe_multipliers = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440
        }

    def create_multi_timeframe_features(self, data: pd.DataFrame, base_timeframe: str,
                                             target_timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Create features from multiple timeframes.

        Args:
            data: Base timeframe data
            base_timeframe: Base timeframe (e.g., '1m')
            target_timeframes: List of target timeframes

        Returns:
            Dictionary of resampled data by timeframe
        """
        mtf_data = {}

        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Data index is not DatetimeIndex, attempting to convert")
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
                data.index = pd.to_datetime(data.index)

        for tf in target_timeframes:
            if tf == base_timeframe:
                continue

            try:
                resampled = self._resample_data(data, base_timeframe, tf)
                mtf_data[tf] = resampled
                self.logger.info(f"✅ Resampled data to {tf}: {len(resampled)} rows")
            except Exception as e:
                self.logger.error(f"❌ Failed to resample to {tf}: {e}")
                continue

        return mtf_data

    def _resample_data(self, data: pd.DataFrame, from_tf: str, to_tf: str) -> pd.DataFrame:
        """Resample data from one timeframe to another.

        Args:
            data: Source data
            from_tf: Source timeframe
            to_tf: Target timeframe

        Returns:
            Resampled data
        """
        # Calculate resampling rule
        from_minutes = self.timeframe_multipliers.get(from_tf, 1)
        to_minutes = self.timeframe_multipliers.get(to_tf, 60)

        if to_minutes % from_minutes != 0:
            raise ValueError(f"Cannot resample from {from_tf} to {to_tf}: incompatible timeframes")

        periods = to_minutes // from_minutes
        rule = f"{periods}min" if periods > 1 else "min"

        # OHLCV resampling
        resampled = data.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled


class WaveletAnalyzer:
    """Engine for wavelet-based feature extraction."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize wavelet analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('WaveletAnalyzer')

        # Try to import PyWavelets
        try:
            import pywt
            self.pywt = pywt
            self.available = True
        except ImportError:
            self.logger.warning("PyWavelets not available, wavelet features disabled")
            self.pywt = None
            self.available = False

    def extract_wavelet_features(self, data: pd.DataFrame, price_column: str = 'close',
                               symbol: str = '', timeframe: str = '') -> pd.DataFrame:
        """Extract wavelet-based features from price data.

        Args:
            data: Price data
            price_column: Column name for price data
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            DataFrame with wavelet features
        """
        if not self.available or self.pywt is None:
            self.logger.warning("Wavelet analysis not available")
            return pd.DataFrame(index=data.index)

        try:
            features = pd.DataFrame(index=data.index)
            price_data = data[price_column].values

            # Remove NaN values for wavelet analysis
            clean_data = pd.Series(price_data).dropna()
            if len(clean_data) < 32:  # Minimum length for wavelet analysis
                self.logger.warning("Insufficient data for wavelet analysis")
                return features

            # Perform wavelet decomposition
            wavelet = self.config.get('wavelet', 'db4')
            level = self.config.get('wavelet_level', 4)

            coeffs = self.pywt.wavedec(clean_data.values, wavelet, level=level)

            # Extract features from wavelet coefficients
            for i, coeff in enumerate(coeffs):
                if i == 0:  # Approximation coefficients
                    features[f'wavelet_approx_mean'] = self._reindex_coeff(coeff, clean_data.index, 'mean')
                    features[f'wavelet_approx_std'] = self._reindex_coeff(coeff, clean_data.index, 'std')
                    features[f'wavelet_approx_energy'] = self._reindex_coeff(coeff, clean_data.index, 'energy')
                else:  # Detail coefficients
                    features[f'wavelet_detail_{i}_mean'] = self._reindex_coeff(coeff, clean_data.index, 'mean')
                    features[f'wavelet_detail_{i}_std'] = self._reindex_coeff(coeff, clean_data.index, 'std')
                    features[f'wavelet_detail_{i}_energy'] = self._reindex_coeff(coeff, clean_data.index, 'energy')

            # Additional wavelet features
            features['wavelet_smoothness'] = self._calculate_smoothness(price_data)
            features['wavelet_entropy'] = self._calculate_entropy(price_data)

            self.logger.info(f"✅ Extracted {len(features.columns)} wavelet features")
            return features

        except Exception as e:
            self.logger.error(f"❌ Wavelet feature extraction failed: {e}")
            return pd.DataFrame(index=data.index)

    def _reindex_coeff(self, coeff: np.ndarray, original_index: pd.Index, method: str) -> pd.Series:
        """Reindex wavelet coefficients to match original data length.

        Args:
            coeff: Wavelet coefficients
            original_index: Original data index
            method: Aggregation method

        Returns:
            Reindexed series
        """
        if method == 'mean':
            value = np.mean(coeff)
        elif method == 'std':
            value = np.std(coeff)
        elif method == 'energy':
            value = np.sum(coeff ** 2)
        else:
            value = np.mean(coeff)

        return pd.Series([value] * len(original_index), index=original_index)

    def _calculate_smoothness(self, data: np.ndarray) -> pd.Series:
        """Calculate smoothness using wavelet analysis."""
        try:
            # Use simple difference-based smoothness if wavelet fails
            diff = np.diff(data)
            smoothness = 1 / (1 + np.std(diff))
            return pd.Series([smoothness] * len(data))
        except:
            return pd.Series([0.5] * len(data))

    def _calculate_entropy(self, data: np.ndarray) -> pd.Series:
        """Calculate entropy from wavelet coefficients."""
        try:
            # Simple entropy calculation
            normalized = (data - np.mean(data)) / (np.std(data) + 1e-10)
            hist, _ = np.histogram(normalized, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zeros
            entropy = -np.sum(hist * np.log2(hist))
            return pd.Series([entropy] * len(data))
        except:
            return pd.Series([1.0] * len(data))


class EnhancedFeatureInteractionEngine:
    """Enhanced engine for creating sophisticated feature interactions."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced feature interaction engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('EnhancedFeatureInteractionEngine')

    async def create_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated feature interactions.

        Args:
            data: Data with base features

        Returns:
            Data with interaction features
        """
        if data.empty:
            return data

        try:
            # Polynomial interactions
            data = self._add_polynomial_interactions(data)

            # Conditional interactions
            data = self._add_conditional_interactions(data)

            # Ratio-based interactions
            data = self._add_ratio_interactions(data)

            # Momentum interactions
            data = self._add_momentum_interactions(data)

            # Volatility interactions
            data = self._add_volatility_interactions(data)

            # Count actual interaction features created
            original_cols = len(set(data.columns))
            # All interaction features have specific patterns
            interaction_patterns = ['_interaction', '_ratio', '_cross', '_momentum_squared', '_acceleration']
            interaction_cols = [col for col in data.columns if any(pattern in col for pattern in interaction_patterns)]
            self.logger.info(f"✅ Created {len(interaction_cols)} interaction features")
            return data

        except Exception as e:
            self.logger.error(f"❌ Feature interaction creation failed: {e}")
            return data

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        if 'volume' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap

    def _add_polynomial_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial feature interactions with limits."""
        vwap = data.get('feature_vwap', self._calculate_vwap(data))
        max_interactions = self.config.get('step06_feature_engineering', {}).get('max_interactions', 200) // 4  # Use config, divide by 4 for this method
        interaction_count = 0

        # Price momentum interactions
        if 'close' in data.columns and interaction_count < max_interactions:
            returns = data['close'].pct_change()
            data['price_momentum_squared'] = returns ** 2
            data['price_acceleration'] = returns.diff()
            interaction_count += 2

        # VWAP momentum interactions
        if not vwap.empty and interaction_count < max_interactions:
            vwap_returns = vwap.pct_change()
            data['vwap_momentum_squared'] = vwap_returns ** 2
            data['vwap_acceleration'] = vwap_returns.diff()
            interaction_count += 2

        # RSI and momentum interactions (limited)
        rsi_cols = [col for col in data.columns if 'rsi' in col.lower()][:1]  # Limit to 1 RSI column
        momentum_cols = [col for col in data.columns if 'momentum' in col.lower() or 'roc' in col.lower()][:2]  # Limit to 2 momentum columns

        for rsi_col in rsi_cols:
            if interaction_count >= max_interactions:
                break
            for mom_col in momentum_cols[:2]:  # Max 2 momentum interactions per RSI
                if interaction_count >= max_interactions:
                    break
                if rsi_col in data.columns and mom_col in data.columns:
                    data[f'{rsi_col}_{mom_col}_interaction'] = data[rsi_col] * data[mom_col]
                    interaction_count += 1

        # VWAP-based momentum interactions (limited)
        if interaction_count < max_interactions:
            vwap_momentum_cols = [col for col in data.columns if 'vwap_momentum' in col.lower()][:1]
            for rsi_col in rsi_cols[:1]:
                if interaction_count >= max_interactions:
                    break
                for vwap_mom_col in vwap_momentum_cols:
                    if interaction_count >= max_interactions:
                        break
                    if rsi_col in data.columns and vwap_mom_col in data.columns:
                        data[f'{rsi_col}_{vwap_mom_col}_interaction'] = data[rsi_col] * data[vwap_mom_col]
                        interaction_count += 1

        return data

    def _add_conditional_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add conditional feature interactions."""
        # Volume conditionals
        if 'volume' in data.columns and 'close' in data.columns:
            data['high_volume_price_change'] = np.where(
                data['volume'] > data['volume'].rolling(20).mean(),
                data['close'].pct_change(),
                0
            )

        # RSI conditionals
        rsi_cols = [col for col in data.columns if 'rsi' in col.lower()]
        if rsi_cols and 'close' in data.columns:
            rsi_col = rsi_cols[0]
            data['rsi_overbought_momentum'] = np.where(
                data[rsi_col] > 70,
                data['close'].pct_change(),
                0
            )
            data['rsi_oversold_momentum'] = np.where(
                data[rsi_col] < 30,
                data['close'].pct_change(),
                0
            )

        return data

    def _add_ratio_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add ratio-based feature interactions with limits."""
        max_ratios = self.config.get('step06_feature_engineering', {}).get('max_interactions', 200) // 6  # Use config, divide by 6 for this method
        ratio_count = 0

        # Moving average ratios (limited)
        ma_cols = [col for col in data.columns if 'sma' in col.lower() or 'ema' in col.lower()][:4]  # Limit to 4 MA columns
        if len(ma_cols) >= 2 and 'close' in data.columns and ratio_count < max_ratios:
            for i in range(len(ma_cols)):
                if ratio_count >= max_ratios:
                    break
                for j in range(i+1, min(i+3, len(ma_cols))):  # Max 2 ratios per MA
                    if ratio_count >= max_ratios:
                        break
                    col1, col2 = ma_cols[i], ma_cols[j]
                    if col1 in data.columns and col2 in data.columns:
                        data[f'{col1}_{col2}_ratio'] = data[col1] / (data[col2] + 1e-10)
                        ratio_count += 1

        return data

    def _add_momentum_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based interactions with limits."""
        max_momentum_interactions = self.config.get('step06_feature_engineering', {}).get('max_interactions', 200) // 10  # Use config, divide by 10 for this method
        momentum_count = 0

        momentum_cols = [col for col in data.columns if 'momentum' in col.lower() or 'roc' in col.lower()][:3]  # Limit to 3 momentum columns

        if len(momentum_cols) >= 2 and momentum_count < max_momentum_interactions:
            for i in range(len(momentum_cols)):
                if momentum_count >= max_momentum_interactions:
                    break
                for j in range(i+1, min(i+2, len(momentum_cols))):  # Max 1 cross per momentum pair
                    if momentum_count >= max_momentum_interactions:
                        break
                    col1, col2 = momentum_cols[i], momentum_cols[j]
                    if col1 in data.columns and col2 in data.columns:
                        data[f'{col1}_{col2}_cross'] = np.sign(data[col1]) * np.sign(data[col2])
                        momentum_count += 1

        return data

    def _add_volatility_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based interactions."""
        vol_cols = [col for col in data.columns if 'volatility' in col.lower() or 'std' in col.lower()]

        if vol_cols and 'close' in data.columns:
            vol_col = vol_cols[0]
            returns = data['close'].pct_change()
            data['vol_adjusted_returns'] = returns / (data[vol_col] + 1e-10)

        return data


class EnhancedRegimeAwareFeatureEngine:
    """Enhanced engine for creating advanced regime-aware features."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced regime-aware feature engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('EnhancedRegimeAwareFeatureEngine')

    async def create_regime_features(self, data: pd.DataFrame, regime_characteristics: Dict[str, Any]) -> pd.DataFrame:
        """Create advanced regime-aware features.

        Args:
            data: Market data with regime labels
            regime_characteristics: Regime characteristics from clustering

        Returns:
            Data with advanced regime-aware features
        """
        if 'regime_label' not in data.columns and 'composite_cluster_id' not in data.columns:
            self.logger.warning("No regime labels found in data - creating default regime labels")
            # Create default regime labels based on simple price movement patterns
            data = self._create_default_regime_labels(data)
            regime_col = 'regime_label'
        else:
            # Get regime column
            regime_col = 'regime_label' if 'regime_label' in data.columns else 'composite_cluster_id'

        try:

            # Basic regime features
            data = self._add_basic_regime_features(data, regime_col)

            # Regime transition features
            data = self._add_regime_transition_features(data, regime_col)

            # Regime-specific indicators
            data = self._add_regime_specific_indicators(data, regime_col)

            # Regime performance features
            data = self._add_regime_performance_features(data, regime_col)

            self.logger.info("✅ Created enhanced regime-aware features")
            return data

        except Exception as e:
            self.logger.error(f"❌ Enhanced regime feature creation failed: {e}")
            return data

    def _create_default_regime_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create default regime labels based on simple price movement patterns.

        Args:
            data: Market data without regime labels

        Returns:
            Data with default regime labels added
        """
        if 'close' not in data.columns:
            self.logger.error("Cannot create default regime labels: 'close' column not found")
            data['regime_label'] = 0  # Default single regime
            return data

        try:
            # Calculate returns and volatility
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()

            # Create simple regime classification based on volatility and trend
            # Regime 0: Low volatility trending
            # Regime 1: High volatility ranging
            # Regime 2: High volatility trending

            # Use rolling statistics to classify regimes
            vol_threshold = volatility.median()
            trend_strength = abs(returns.rolling(20).mean())

            # Classify each period
            conditions = [
                (volatility <= vol_threshold) & (trend_strength <= trend_strength.median()),  # Low vol, weak trend
                (volatility > vol_threshold) & (trend_strength <= trend_strength.median()),   # High vol, weak trend
                (volatility > vol_threshold) & (trend_strength > trend_strength.median())     # High vol, strong trend
            ]
            choices = [0, 1, 2]

            data['regime_label'] = np.select(conditions, choices, default=0)

            # Smooth the labels to avoid too frequent switching
            data['regime_label'] = data['regime_label'].rolling(10, center=True).median().fillna(method='bfill').fillna(method='ffill').astype(int)

            self.logger.info(f"Created default regime labels: {data['regime_label'].value_counts().to_dict()}")

        except Exception as e:
            self.logger.error(f"Error creating default regime labels: {e}")
            data['regime_label'] = 0  # Fallback to single regime

        return data

    def _add_basic_regime_features(self, data: pd.DataFrame, regime_col: str) -> pd.DataFrame:
        """Add basic regime features."""
        # Regime dummies
        regime_dummies = pd.get_dummies(data[regime_col], prefix='regime')
        data = pd.concat([data, regime_dummies], axis=1)

        # Regime duration
        data['regime_changed'] = (data[regime_col] != data[regime_col].shift(1)).astype(int)
        data['regime_duration'] = data.groupby(
            (data[regime_col] != data[regime_col].shift()).cumsum()
        ).cumcount()

        return data

    def _add_regime_transition_features(self, data: pd.DataFrame, regime_col: str) -> pd.DataFrame:
        """Add regime transition features."""
        # Transition probabilities (simplified)
        data['regime_transition'] = data[regime_col].astype(str) + '_to_' + data[regime_col].shift(-1).astype(str)
        data['regime_transition'] = data['regime_transition'].fillna('unknown')

        # Transition frequency
        transition_counts = data['regime_transition'].value_counts()
        data['transition_frequency'] = data['regime_transition'].map(transition_counts)

        return data

    def _add_regime_specific_indicators(self, data: pd.DataFrame, regime_col: str) -> pd.DataFrame:
        """Add regime-specific technical indicators."""
        if 'close' in data.columns:
            # Regime-specific moving averages
            for regime in data[regime_col].unique():
                mask = data[regime_col] == regime
                regime_data = data[mask].copy()

                if len(regime_data) > 20:
                    data.loc[mask, f'regime_{regime}_sma_20'] = regime_data['close'].rolling(20).mean()
                    data.loc[mask, f'regime_{regime}_volatility'] = regime_data['close'].pct_change().rolling(20).std()

        return data

    def _add_regime_performance_features(self, data: pd.DataFrame, regime_col: str) -> pd.DataFrame:
        """Add regime performance features."""
        if 'close' in data.columns:
            # Rolling performance within regime
            data['regime_cumulative_return'] = data.groupby(
                (data[regime_col] != data[regime_col].shift()).cumsum()
            )['close'].pct_change().cumsum()

            # Regime-specific Sharpe ratio approximation
            returns = data.groupby(regime_col)['close'].pct_change()
            data['regime_sharpe'] = returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-10)

        return data


class MarketProfileFeatureEngine:
    """Engine for creating market profile features including POC, Value Area, and Volume Profile."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize market profile feature engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('MarketProfileFeatureEngine')

        # Extract market profile parameters with regime-specific optimization
        mp_params = config.get('feature_engineering_parameters', {}).get('market_profile_parameters', {})
        step17_mp = config.get('step17_optimization', {}).get('market_profile', {})

        # Use regime-specific parameters if available, otherwise use defaults
        self.profile_periods = step17_mp.get('profile_periods', mp_params.get('profile_periods', [20, 50, 100]))
        self.value_area_percentage = step17_mp.get('value_area_percentage', mp_params.get('value_area_percentage', 0.7))
        self.initial_balance_periods = step17_mp.get('initial_balance_periods', mp_params.get('initial_balance_periods', [5, 10, 20]))

    def create_market_profile_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive market profile features.

        Args:
            data: Market data with OHLC and volume

        Returns:
            Data with market profile features
        """
        try:
            # Point of Control (POC) features
            for period in self.profile_periods:
                data = self._add_poc_features(data, period)

            # Value Area features
            for period in self.profile_periods:
                data = self._add_value_area_features(data, period)

            # Initial Balance features
            for period in self.initial_balance_periods:
                data = self._add_initial_balance_features(data, period)

            # Volume Profile features
            for period in self.profile_periods:
                data = self._add_volume_profile_features(data, period)

            self.logger.info("✅ Generated market profile features")
            return data

        except Exception as e:
            self.logger.error(f"❌ Market profile feature creation failed: {e}")
            return data

    def _add_poc_features(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add Point of Control features."""
        # POC - Price level with maximum volume
        for i in range(period, len(data)):
            window = data.iloc[i-period:i]
            volume_profile = {}

            # Build volume profile
            for idx, row in window.iterrows():
                price_range = np.linspace(row['low'], row['high'], 10)
                volume_per_price = row['volume'] / len(price_range)

                for price in price_range:
                    price_level = round(price, 2)
                    volume_profile[price_level] = volume_profile.get(price_level, 0) + volume_per_price

            # Find POC
            if volume_profile:
                poc = max(volume_profile, key=volume_profile.get)
                data.loc[data.index[i-1], f'market_profile_poc_{period}'] = poc
                data.loc[data.index[i-1], f'market_profile_poc_distance_{period}'] = data.loc[data.index[i-1], 'close'] - poc

        return data

    def _add_value_area_features(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add Value Area features."""
        for i in range(period, len(data)):
            window = data.iloc[i-period:i]
            volume_profile = {}

            # Build volume profile
            for idx, row in window.iterrows():
                price_range = np.linspace(row['low'], row['high'], 10)
                volume_per_price = row['volume'] / len(price_range)

                for price in price_range:
                    price_level = round(price, 2)
                    volume_profile[price_level] = volume_profile.get(price_level, 0) + volume_per_price

            if volume_profile:
                total_volume = sum(volume_profile.values())
                target_volume = total_volume * self.value_area_percentage

                # Sort by volume
                sorted_prices = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)

                cumulative_volume = 0
                value_area_prices = []

                for price, volume in sorted_prices:
                    cumulative_volume += volume
                    value_area_prices.append(price)
                    if cumulative_volume >= target_volume:
                        break

                if value_area_prices:
                    vah = max(value_area_prices)
                    val = min(value_area_prices)
                    data.loc[data.index[i-1], f'market_profile_vah_{period}'] = vah
                    data.loc[data.index[i-1], f'market_profile_val_{period}'] = val
                    data.loc[data.index[i-1], f'market_profile_value_area_width_{period}'] = vah - val

        return data

    def _add_initial_balance_features(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add Initial Balance features."""
        for i in range(period, len(data)):
            window = data.iloc[i-period:i]

            if len(window) >= period:
                ib_high = window['high'].max()
                ib_low = window['low'].min()
                ib_mid = (ib_high + ib_low) / 2

                current_price = data.loc[data.index[i-1], 'close']

                data.loc[data.index[i-1], f'market_profile_ib_high_{period}'] = ib_high
                data.loc[data.index[i-1], f'market_profile_ib_low_{period}'] = ib_low
                data.loc[data.index[i-1], f'market_profile_ib_mid_{period}'] = ib_mid
                data.loc[data.index[i-1], f'market_profile_ib_range_{period}'] = ib_high - ib_low
                data.loc[data.index[i-1], f'market_profile_ib_position_{period}'] = (current_price - ib_low) / (ib_high - ib_low + 1e-10)

        return data

    def _add_volume_profile_features(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add Volume Profile features."""
        for i in range(period, len(data)):
            window = data.iloc[i-period:i]

            # Volume concentration
            volume_concentration = window['volume'].std() / (window['volume'].mean() + 1e-10)
            data.loc[data.index[i-1], f'market_profile_volume_concentration_{period}'] = volume_concentration

            # Volume profile skewness
            volume_skew = window['volume'].skew()
            data.loc[data.index[i-1], f'market_profile_volume_skew_{period}'] = volume_skew

        return data


class IchimokuFeatureEngine:
    """Engine for creating Ichimoku Cloud features."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Ichimoku feature engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('IchimokuFeatureEngine')

        # Extract Ichimoku parameters with regime-specific optimization
        ichimoku_params = config.get('feature_engineering_parameters', {}).get('ichimoku_parameters', {})
        step17_ichimoku = config.get('step17_optimization', {}).get('ichimoku', {})

        # Use regime-specific parameters if available, otherwise use defaults
        self.tenkan_period = step17_ichimoku.get('tenkan_period', ichimoku_params.get('tenkan_period', 9))
        self.kijun_period = step17_ichimoku.get('kijun_period', ichimoku_params.get('kijun_period', 26))
        self.senkou_span_b_period = step17_ichimoku.get('senkou_span_b_period', ichimoku_params.get('senkou_span_b_period', 52))
        self.displacement = step17_ichimoku.get('displacement', ichimoku_params.get('displacement', 26))

    def create_ichimoku_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive Ichimoku Cloud features.

        Args:
            data: Market data with OHLC

        Returns:
            Data with Ichimoku features
        """
        try:
            # Tenkan-sen (Conversion Line)
            data['ichimoku_tenkan'] = self._calculate_tenkan(data)

            # Kijun-sen (Base Line)
            data['ichimoku_kijun'] = self._calculate_kijun(data)

            # Senkou Span A (Leading Span A)
            data['ichimoku_senkou_a'] = ((data['ichimoku_tenkan'] + data['ichimoku_kijun']) / 2).shift(self.displacement)

            # Senkou Span B (Leading Span B)
            data['ichimoku_senkou_b'] = self._calculate_senkou_b(data).shift(self.displacement)

            # Chikou Span (Lagging Span)
            data['ichimoku_chikou'] = data['close'].shift(-self.displacement)

            # Cloud features
            data = self._add_cloud_features(data)

            # Signal features
            data = self._add_ichimoku_signals(data)

            self.logger.info("✅ Generated Ichimoku features")
            return data

        except Exception as e:
            self.logger.error(f"❌ Ichimoku feature creation failed: {e}")
            return data

    def _calculate_tenkan(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Tenkan-sen (Conversion Line)."""
        high_max = data['high'].rolling(self.tenkan_period).max()
        low_min = data['low'].rolling(self.tenkan_period).min()
        return (high_max + low_min) / 2

    def _calculate_kijun(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Kijun-sen (Base Line)."""
        high_max = data['high'].rolling(self.kijun_period).max()
        low_min = data['low'].rolling(self.kijun_period).min()
        return (high_max + low_min) / 2

    def _calculate_senkou_b(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Senkou Span B."""
        high_max = data['high'].rolling(self.senkou_span_b_period).max()
        low_min = data['low'].rolling(self.senkou_span_b_period).min()
        return (high_max + low_min) / 2

    def _add_cloud_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cloud-based features."""
        # Cloud thickness
        data['ichimoku_cloud_thickness'] = abs(data['ichimoku_senkou_a'] - data['ichimoku_senkou_b'])

        # Cloud position relative to price
        data['ichimoku_cloud_position'] = data['close'] - ((data['ichimoku_senkou_a'] + data['ichimoku_senkou_b']) / 2)

        # Cloud color (green when Senkou A > Senkou B)
        data['ichimoku_cloud_green'] = (data['ichimoku_senkou_a'] > data['ichimoku_senkou_b']).astype(int)

        return data

    def _add_ichimoku_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku signal features."""
        # Tenkan-Kijun cross
        tenkan_kijun_diff = data['ichimoku_tenkan'] - data['ichimoku_kijun']
        data['ichimoku_tenkan_kijun_cross'] = (tenkan_kijun_diff > 0).astype(int).diff().fillna(0)

        # Price vs Cloud
        data['ichimoku_price_above_cloud'] = (data['close'] > data[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)).astype(int)
        data['ichimoku_price_below_cloud'] = (data['close'] < data[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1)).astype(int)

        # Chikou vs Price
        data['ichimoku_chikou_above_price'] = (data['ichimoku_chikou'] > data['close']).astype(int)

        return data


class HarmonicPatternFeatureEngine:
    """Engine for creating harmonic pattern features."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize harmonic pattern feature engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('HarmonicPatternFeatureEngine')

        # Extract harmonic parameters with regime-specific optimization
        harmonic_params = config.get('feature_engineering_parameters', {}).get('harmonic_parameters', {})
        step17_harmonic = config.get('step17_optimization', {}).get('harmonic_patterns', {})

        # Use regime-specific parameters if available, otherwise use defaults
        self.fib_levels = step17_harmonic.get('fib_levels', harmonic_params.get('fib_levels', [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]))
        self.pattern_tolerance = step17_harmonic.get('pattern_tolerance', harmonic_params.get('pattern_tolerance', 0.05))
        self.min_pattern_length = step17_harmonic.get('min_pattern_length', harmonic_params.get('min_pattern_length', 5))
        self.max_pattern_length = step17_harmonic.get('max_pattern_length', harmonic_params.get('max_pattern_length', 50))

    def create_harmonic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive harmonic pattern features.

        Args:
            data: Market data with OHLC

        Returns:
            Data with harmonic pattern features
        """
        try:
            # Fibonacci retracement levels
            data = self._add_fibonacci_features(data)

            # Pattern detection features
            data = self._add_pattern_detection_features(data)

            # Wave relationship features
            data = self._add_wave_relationship_features(data)

            self.logger.info("✅ Generated harmonic pattern features")
            return data

        except Exception as e:
            self.logger.error(f"❌ Harmonic pattern feature creation failed: {e}")
            return data

    def _add_fibonacci_features(self, data: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Add Fibonacci retracement and extension features."""
        for i in range(lookback, len(data)):
            window = data.iloc[i-lookback:i]

            if len(window) >= 10:
                swing_high = window['high'].max()
                swing_low = window['low'].min()
                price_range = swing_high - swing_low

                current_price = data.loc[data.index[i-1], 'close']

                # Fibonacci retracement levels
                for fib_level in self.fib_levels:
                    if fib_level <= 1.0:  # Retracement
                        fib_price = swing_high - (price_range * fib_level)
                        distance = abs(current_price - fib_price) / (price_range + 1e-10)
                        data.loc[data.index[i-1], f'harmonic_fib_ret_{fib_level:.3f}'] = fib_price
                        data.loc[data.index[i-1], f'harmonic_fib_ret_dist_{fib_level:.3f}'] = distance
                    else:  # Extension
                        fib_price = swing_high + (price_range * (fib_level - 1.0))
                        distance = abs(current_price - fib_price) / (price_range + 1e-10)
                        data.loc[data.index[i-1], f'harmonic_fib_ext_{fib_level:.3f}'] = fib_price
                        data.loc[data.index[i-1], f'harmonic_fib_ext_dist_{fib_level:.3f}'] = distance

        return data

    def _add_pattern_detection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add harmonic pattern detection features."""
        # Simple pattern ratios (for demonstration - can be expanded)
        data['harmonic_golden_ratio'] = data['high'].rolling(20).max() / (data['low'].rolling(20).min() + 1e-10)

        # Pattern strength based on volume
        data['harmonic_pattern_strength'] = data['volume'].rolling(10).mean() / (data['volume'].rolling(50).mean() + 1e-10)

        # Wave symmetry features
        price_change = data['close'].pct_change()
        data['harmonic_wave_symmetry'] = abs(price_change.rolling(10).mean()) / (abs(price_change.rolling(20).mean()) + 1e-10)

        return data

    def _add_wave_relationship_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add wave relationship features."""
        # Wave length ratios
        for period in [5, 10, 20]:
            wave_length = data['close'].rolling(period).std()
            data[f'harmonic_wave_length_{period}'] = wave_length

            if period >= 10:
                short_wave = data[f'harmonic_wave_length_{period//2}']
                data[f'harmonic_wave_ratio_{period}'] = short_wave / (wave_length + 1e-10)

        return data


class SentimentFeatureEngine:
    """Engine for creating sentiment-based features including Greed/Fear Index and Momentum Crowding."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize sentiment feature engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('SentimentFeatureEngine')

        # Extract sentiment parameters with regime-specific optimization
        sentiment_params = config.get('feature_engineering_parameters', {}).get('sentiment_parameters', {})
        step17_sentiment = config.get('step17_optimization', {}).get('sentiment_features', {})

        # Use regime-specific parameters if available, otherwise use defaults
        self.greed_fear_lookback = step17_sentiment.get('greed_fear_lookback', sentiment_params.get('greed_fear_lookback', 30))
        self.momentum_crowding_window = step17_sentiment.get('momentum_crowding_window', sentiment_params.get('momentum_crowding_window', 20))
        self.sentiment_smoothing = step17_sentiment.get('sentiment_smoothing', sentiment_params.get('sentiment_smoothing', 5))

    def create_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive sentiment-based features.

        Args:
            data: Market data with OHLC and volume

        Returns:
            Data with sentiment features
        """
        try:
            # Greed/Fear Index features
            data = self._add_greed_fear_features(data)

            # Momentum Crowding features
            data = self._add_momentum_crowding_features(data)

            # Market sentiment indicators
            data = self._add_market_sentiment_features(data)

            self.logger.info("✅ Generated sentiment features")
            return data

        except Exception as e:
            self.logger.error(f"❌ Sentiment feature creation failed: {e}")
            return data

    def _add_greed_fear_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Greed/Fear Index features."""
        # Calculate returns for different periods
        returns_1d = data['close'].pct_change()
        returns_7d = data['close'].pct_change(7)
        returns_30d = data['close'].pct_change(30)

        # Volatility component
        volatility = returns_1d.rolling(self.greed_fear_lookback).std()

        # Volume component
        volume_ma = data['volume'].rolling(self.greed_fear_lookback).mean()
        volume_ratio = data['volume'] / volume_ma

        # Momentum component
        momentum = returns_7d.rolling(self.sentiment_smoothing).mean()

        # Composite Greed/Fear score (simplified version)
        # In practice, this would integrate with actual Greed/Fear Index API
        greed_fear_score = (
            0.4 * (1 - volatility.rank(pct=True)) +  # Low volatility = greed
            0.3 * volume_ratio.rank(pct=True) +     # High volume = greed
            0.3 * momentum.rank(pct=True)           # Positive momentum = greed
        )

        data['sentiment_greed_fear_score'] = greed_fear_score
        data['sentiment_greed_fear_extreme'] = ((greed_fear_score > 0.8) | (greed_fear_score < 0.2)).astype(int)

        return data

    def _add_momentum_crowding_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Momentum Crowding features."""
        # Calculate momentum for different timeframes
        momentum_short = data['close'].pct_change(5)
        momentum_medium = data['close'].pct_change(20)
        momentum_long = data['close'].pct_change(50)

        # Crowding measure based on momentum alignment
        momentum_alignment = (
            np.sign(momentum_short) +
            np.sign(momentum_medium) +
            np.sign(momentum_long)
        ) / 3

        data['sentiment_momentum_crowding'] = momentum_alignment.abs()

        # Extreme crowding signals
        data['sentiment_extreme_bull_crowding'] = (momentum_alignment > 0.8).astype(int)
        data['sentiment_extreme_bear_crowding'] = (momentum_alignment < -0.8).astype(int)

        # Crowding divergence
        momentum_divergence = abs(momentum_short - momentum_long)
        data['sentiment_momentum_divergence'] = momentum_divergence

        return data

    def _add_market_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add general market sentiment features."""
        # Put/Call ratio proxy (using volume and price action)
        price_volatility = data['close'].pct_change().rolling(20).std()
        volume_volatility = data['volume'].pct_change().rolling(20).std()

        # High volume + low price volatility might indicate accumulation (bullish sentiment)
        sentiment_score = volume_volatility / (price_volatility + 1e-10)
        data['sentiment_accumulation_score'] = sentiment_score.rolling(5).mean()

        # Extreme sentiment readings
        data['sentiment_extreme_optimism'] = (sentiment_score > sentiment_score.quantile(0.9)).astype(int)
        data['sentiment_extreme_pessimism'] = (sentiment_score < sentiment_score.quantile(0.1)).astype(int)

        return data


class SupportResistanceFeatureEngine:
    """Engine for creating Support/Resistance (S/R) features with ML-optimized encoding."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize S/R feature engine.
        
        Args:
            config: Configuration dictionary with S/R parameters
        """
        self.config = config
        self.logger = system_logger.getChild('SupportResistanceFeatureEngine')
        
        # ATR and normalization parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 1.0)
        
        # S/R detection parameters
        self.pivot_period = config.get('pivot_period', 4)
        self.prominence_threshold = config.get('prominence_threshold', 0.5)
        self.width_threshold = config.get('width_threshold', 1)
        
        # Feature encoding parameters
        self.max_levels_per_type = config.get('max_levels_per_type', 10)
        self.tolerance_atr_multiplier = config.get('tolerance_atr_multiplier', 0.5)
        
    @log_all_calls
    def create_sr_features(self, data: pd.DataFrame, sr_levels: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Create comprehensive S/R features from market data and detected levels.
        
        Args:
            data: Market data with OHLCV columns
            sr_levels: Optional pre-detected S/R levels
            
        Returns:
            DataFrame with S/R features
        """
        try:
            self.logger.info('🔧 Creating S/R features...')
            
            # Initialize features DataFrame
            sr_features = pd.DataFrame(index=data.index)
            
            # Calculate ATR for normalization
            atr = self._calculate_atr(data)
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else atr.mean()
            
            # Detect S/R levels if not provided
            if sr_levels is None:
                sr_levels = self._detect_sr_levels(data)
            
            # Create enhanced S/R features
            sr_features = self._create_distance_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_strength_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_multiplicity_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_top_k_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_binary_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_time_features(sr_features, data, sr_levels)
            sr_features = self._create_volume_features(sr_features, data, sr_levels)
            
            # Create advanced S/R features
            sr_features = self._create_signed_distance_change_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_velocity_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_proximity_direction_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_breakout_rejection_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_relative_momentum_features(sr_features, data, sr_levels, current_atr)
            sr_features = self._create_time_since_approach_features(sr_features, data, sr_levels, current_atr)
            
            self.logger.info(f'✅ Created {len(sr_features.columns)} S/R features')
            return sr_features
            
        except Exception as e:
            self.logger.error(f'❌ Failed to create S/R features: {e}')
            return pd.DataFrame(index=data.index)
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range (ATR) for normalization."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as rolling mean of True Range
            atr = true_range.rolling(window=self.atr_period).mean()
            
            return atr
        except Exception as e:
            self.logger.warning(f'ATR calculation failed: {e}')
            # Fallback to simple price range
            return (data['high'] - data['low']).rolling(window=self.atr_period).mean()
    
    def _detect_sr_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect S/R levels using pivot-based detection."""
        try:
            from scipy.signal import find_peaks
            
            # Detect pivot highs (resistance)
            pivot_highs, _ = find_peaks(
                data['high'].values,
                prominence=data['high'].std() * self.prominence_threshold,
                width=self.width_threshold,
                distance=self.pivot_period
            )
            
            # Detect pivot lows (support)
            pivot_lows, _ = find_peaks(
                -data['low'].values,  # Invert for valleys
                prominence=data['low'].std() * self.prominence_threshold,
                width=self.width_threshold,
                distance=self.pivot_period
            )
            
            # Create level dictionaries
            resistance_levels = []
            for idx in pivot_highs:
                if idx < len(data):
                    resistance_levels.append({
                        'price': data['high'].iloc[idx],
                        'strength': 0.7,  # Default strength
                        'touch_count': 1,
                        'timestamp': data.index[idx]
                    })
            
            support_levels = []
            for idx in pivot_lows:
                if idx < len(data):
                    support_levels.append({
                        'price': data['low'].iloc[idx],
                        'strength': 0.7,  # Default strength
                        'touch_count': 1,
                        'timestamp': data.index[idx]
                    })
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            self.logger.warning(f'S/R level detection failed: {e}')
            return {'support_levels': [], 'resistance_levels': []}
    
    def _create_distance_features(self, features: pd.DataFrame, data: pd.DataFrame, 
                                sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create distance-based features using percentage returns and ATR normalization."""
        try:
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])
            current_prices = data['close'].values
            
            # Distance to nearest support/resistance (percentage returns + ATR-normalized)
            nearest_support_distances_pct = []
            nearest_resistance_distances_pct = []
            nearest_support_distances_atr = []
            nearest_resistance_distances_atr = []
            
            for price in current_prices:
                # Find nearest support
                support_distances = [abs(price - level.get('price', level)) for level in support_levels 
                                   if isinstance(level.get('price', level), (int, float))]
                nearest_support_dist = min(support_distances) if support_distances else float('inf')
                
                # Calculate percentage return: (level - price) / price
                if support_distances:
                    nearest_support_price = min([level.get('price', level) for level in support_levels 
                                               if isinstance(level.get('price', level), (int, float))], 
                                              key=lambda x: abs(price - x))
                    support_pct = (nearest_support_price - price) / price
                    nearest_support_distances_pct.append(support_pct)
                else:
                    nearest_support_distances_pct.append(0.0)
                
                nearest_support_distances_atr.append(nearest_support_dist / atr if atr > 0 else 0)
                
                # Find nearest resistance
                resistance_distances = [abs(price - level.get('price', level)) for level in resistance_levels 
                                      if isinstance(level.get('price', level), (int, float))]
                nearest_resistance_dist = min(resistance_distances) if resistance_distances else float('inf')
                
                # Calculate percentage return: (level - price) / price
                if resistance_distances:
                    nearest_resistance_price = min([level.get('price', level) for level in resistance_levels 
                                                  if isinstance(level.get('price', level), (int, float))], 
                                                 key=lambda x: abs(price - x))
                    resistance_pct = (nearest_resistance_price - price) / price
                    nearest_resistance_distances_pct.append(resistance_pct)
                else:
                    nearest_resistance_distances_pct.append(0.0)
                
                nearest_resistance_distances_atr.append(nearest_resistance_dist / atr if atr > 0 else 0)
            
            features['sr_dist_to_nearest_support_pct'] = nearest_support_distances_pct
            features['sr_dist_to_nearest_resistance_pct'] = nearest_resistance_distances_pct
            features['sr_dist_to_nearest_support_atr'] = nearest_support_distances_atr
            features['sr_dist_to_nearest_resistance_atr'] = nearest_resistance_distances_atr
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Distance features creation failed: {e}')
            return features
    
    def _create_strength_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create strength-based features."""
        try:
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])
            current_prices = data['close'].values
            
            # Strength of nearest support/resistance
            nearest_support_strengths = []
            nearest_resistance_strengths = []
            
            for i, price in enumerate(current_prices):
                # Find nearest support strength
                support_strengths = []
                for level in support_levels:
                    if isinstance(level.get('price', level), (int, float)):
                        dist = abs(price - level.get('price', level))
                        if dist == features['sr_dist_to_nearest_support_atr'].iloc[i] * atr:
                            support_strengths.append(level.get('strength', 0.5))
                nearest_support_strengths.append(max(support_strengths) if support_strengths else 0)
                
                # Find nearest resistance strength
                resistance_strengths = []
                for level in resistance_levels:
                    if isinstance(level.get('price', level), (int, float)):
                        dist = abs(price - level.get('price', level))
                        if dist == features['sr_dist_to_nearest_resistance_atr'].iloc[i] * atr:
                            resistance_strengths.append(level.get('strength', 0.5))
                nearest_resistance_strengths.append(max(resistance_strengths) if resistance_strengths else 0)
            
            features['sr_strength_of_nearest_support'] = nearest_support_strengths
            features['sr_strength_of_nearest_resistance'] = nearest_resistance_strengths
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Strength features creation failed: {e}')
            return features
    
    def _create_multiplicity_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                    sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create multiplicity features (count of levels within ATR ranges)."""
        try:
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])
            current_prices = data['close'].values
            
            # Count levels within 1×ATR, 2×ATR, 5×ATR
            for atr_multiplier in [1, 2, 5]:
                support_counts = []
                resistance_counts = []
                
                for price in current_prices:
                    threshold = atr * atr_multiplier
                    
                    # Count support levels within threshold
                    support_count = sum(1 for level in support_levels 
                                      if isinstance(level.get('price', level), (int, float)) and 
                                      abs(price - level.get('price', level)) <= threshold)
                    support_counts.append(support_count)
                    
                    # Count resistance levels within threshold
                    resistance_count = sum(1 for level in resistance_levels 
                                         if isinstance(level.get('price', level), (int, float)) and 
                                         abs(price - level.get('price', level)) <= threshold)
                    resistance_counts.append(resistance_count)
                
                features[f'sr_support_levels_within_{atr_multiplier}atr'] = support_counts
                features[f'sr_resistance_levels_within_{atr_multiplier}atr'] = resistance_counts
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Multiplicity features creation failed: {e}')
            return features
    
    def _create_top_k_features(self, features: pd.DataFrame, data: pd.DataFrame,
                             sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create top-k level features (distances and strengths for top-3 nearest levels)."""
        try:
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])
            current_prices = data['close'].values
            
            # Top-k levels: distances and strengths for top-3 nearest levels
            for k in range(1, 4):  # Top 3 levels
                top_support_distances = []
                top_support_strengths = []
                top_resistance_distances = []
                top_resistance_strengths = []
                
                for price in current_prices:
                    # Get top-k support levels
                    support_data = [(abs(price - level.get('price', level)), level.get('strength', 0.5)) 
                                  for level in support_levels 
                                  if isinstance(level.get('price', level), (int, float))]
                    support_data.sort(key=lambda x: x[0])
                    
                    if len(support_data) >= k:
                        dist, strength = support_data[k-1]
                        top_support_distances.append(dist / atr if atr > 0 else 0)
                        top_support_strengths.append(strength)
                    else:
                        top_support_distances.append(float('inf'))
                        top_support_strengths.append(0)
                    
                    # Get top-k resistance levels
                    resistance_data = [(abs(price - level.get('price', level)), level.get('strength', 0.5)) 
                                     for level in resistance_levels 
                                     if isinstance(level.get('price', level), (int, float))]
                    resistance_data.sort(key=lambda x: x[0])
                    
                    if len(resistance_data) >= k:
                        dist, strength = resistance_data[k-1]
                        top_resistance_distances.append(dist / atr if atr > 0 else 0)
                        top_resistance_strengths.append(strength)
                    else:
                        top_resistance_distances.append(float('inf'))
                        top_resistance_strengths.append(0)
                
                features[f'sr_top_{k}_support_dist_atr'] = top_support_distances
                features[f'sr_top_{k}_support_strength'] = top_support_strengths
                features[f'sr_top_{k}_resistance_dist_atr'] = top_resistance_distances
                features[f'sr_top_{k}_resistance_strength'] = top_resistance_strengths
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Top-k features creation failed: {e}')
            return features
    
    def _create_binary_features(self, features: pd.DataFrame, data: pd.DataFrame,
                              sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create binary flag features."""
        try:
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])
            current_prices = data['close'].values
            
            # Binary flags: within tolerance, recent break
            tolerance = atr * self.tolerance_atr_multiplier
            within_tolerance = []
            recent_break = []
            
            for price in current_prices:
                # Check if within tolerance of any level
                within_support = any(abs(price - level.get('price', level)) <= tolerance 
                                   for level in support_levels 
                                   if isinstance(level.get('price', level), (int, float)))
                within_resistance = any(abs(price - level.get('price', level)) <= tolerance 
                                     for level in resistance_levels 
                                     if isinstance(level.get('price', level), (int, float)))
                within_tolerance.append(1 if (within_support or within_resistance) else 0)
                
                # Recent break (simplified - would need historical data for proper implementation)
                recent_break.append(0)  # Placeholder
            
            features['sr_within_tolerance'] = within_tolerance
            features['sr_recent_break'] = recent_break
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Binary features creation failed: {e}')
            return features
    
    def _create_time_features(self, features: pd.DataFrame, data: pd.DataFrame,
                            sr_levels: Dict[str, Any]) -> pd.DataFrame:
        """Create time-based features."""
        try:
            # Time-since-touch: scalar for nearest level (simplified)
            time_since_touch = []
            for i in range(len(data)):
                # Simplified: use index position as proxy for time
                time_since_touch.append(i % 100)  # Placeholder
            
            features['sr_time_since_last_touch'] = time_since_touch
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Time features creation failed: {e}')
            return features
    
    def _create_volume_features(self, features: pd.DataFrame, data: pd.DataFrame,
                              sr_levels: Dict[str, Any]) -> pd.DataFrame:
        """Create volume-based S/R features."""
        try:
            if 'volume' not in data.columns:
                return features
            
            # Volume at S/R levels (simplified)
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])
            
            # Calculate volume ratios (normalized by average volume)
            volume_at_support = []
            volume_at_resistance = []
            
            # Calculate average volume for normalization
            avg_volume = data['volume'].rolling(20).mean()
            
            for i in range(len(data)):
                current_price = data['close'].iloc[i]
                current_volume = data['volume'].iloc[i]
                current_avg_volume = avg_volume.iloc[i] if not pd.isna(avg_volume.iloc[i]) else 1
                
                # Normalize volume by average volume (percentage of average)
                volume_ratio = current_volume / current_avg_volume if current_avg_volume > 0 else 0
                
                # Check if near support levels (using percentage-based tolerance)
                near_support = any(abs(current_price - level.get('price', level)) / current_price <= 0.01 
                                 for level in support_levels 
                                 if isinstance(level.get('price', level), (int, float)))
                volume_at_support.append(volume_ratio if near_support else 0)
                
                # Check if near resistance levels (using percentage-based tolerance)
                near_resistance = any(abs(current_price - level.get('price', level)) / current_price <= 0.01 
                                    for level in resistance_levels 
                                    if isinstance(level.get('price', level), (int, float)))
                volume_at_resistance.append(volume_ratio if near_resistance else 0)
            
            features['sr_volume_at_support'] = volume_at_support
            features['sr_volume_at_resistance'] = volume_at_resistance
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Volume features creation failed: {e}')
            return features

    def _create_signed_distance_change_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                              sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create signed distance change features using percentage returns (Δdist = dist_t - dist_{t-1})."""
        try:
            # Calculate distance changes using percentage returns
            if 'sr_dist_to_nearest_support_pct' in features.columns:
                features['sr_delta_dist_support_pct'] = features['sr_dist_to_nearest_support_pct'].diff()
                features['sr_delta_dist_support_positive'] = (features['sr_delta_dist_support_pct'] > 0).astype(int)
                features['sr_delta_dist_support_negative'] = (features['sr_delta_dist_support_pct'] < 0).astype(int)
            
            if 'sr_dist_to_nearest_resistance_pct' in features.columns:
                features['sr_delta_dist_resistance_pct'] = features['sr_dist_to_nearest_resistance_pct'].diff()
                features['sr_delta_dist_resistance_positive'] = (features['sr_delta_dist_resistance_pct'] > 0).astype(int)
                features['sr_delta_dist_resistance_negative'] = (features['sr_delta_dist_resistance_pct'] < 0).astype(int)
            
            # Also calculate ATR-normalized changes for comparison
            if 'sr_dist_to_nearest_support_atr' in features.columns:
                features['sr_delta_dist_support_atr'] = features['sr_dist_to_nearest_support_atr'].diff()
            
            if 'sr_dist_to_nearest_resistance_atr' in features.columns:
                features['sr_delta_dist_resistance_atr'] = features['sr_dist_to_nearest_resistance_atr'].diff()
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Signed distance change features creation failed: {e}')
            return features

    def _create_velocity_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create velocity toward S/R features using percentage returns (normalized by ATR)."""
        try:
            # Calculate velocity using percentage returns (distance change per time unit)
            if 'sr_delta_dist_support_pct' in features.columns:
                # Use price change as time proxy (1 bar = 1 time unit)
                price_change = data['close'].pct_change().abs()
                features['sr_velocity_toward_support_pct'] = features['sr_delta_dist_support_pct'] / (price_change + 1e-8)
                features['sr_velocity_toward_support_atr'] = features['sr_velocity_toward_support_pct'] / atr
            
            if 'sr_delta_dist_resistance_pct' in features.columns:
                price_change = data['close'].pct_change().abs()
                features['sr_velocity_toward_resistance_pct'] = features['sr_delta_dist_resistance_pct'] / (price_change + 1e-8)
                features['sr_velocity_toward_resistance_atr'] = features['sr_velocity_toward_resistance_pct'] / atr
            
            # Also calculate velocity using ATR-normalized distances
            if 'sr_delta_dist_support_atr' in features.columns:
                price_change = data['close'].pct_change().abs()
                features['sr_velocity_toward_support_atr_raw'] = features['sr_delta_dist_support_atr'] / (price_change + 1e-8)
            
            if 'sr_delta_dist_resistance_atr' in features.columns:
                price_change = data['close'].pct_change().abs()
                features['sr_velocity_toward_resistance_atr_raw'] = features['sr_delta_dist_resistance_atr'] / (price_change + 1e-8)
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Velocity features creation failed: {e}')
            return features

    def _create_proximity_direction_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                           sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create proximity + direction categorical features."""
        try:
            tolerance = atr * self.tolerance_atr_multiplier
            
            # Determine proximity and direction states
            proximity_states = []
            
            for i in range(len(data)):
                current_price = data['close'].iloc[i]
                
                # Get distances to nearest levels (use percentage returns for better normalization)
                dist_support_pct = abs(features['sr_dist_to_nearest_support_pct'].iloc[i]) if 'sr_dist_to_nearest_support_pct' in features.columns else float('inf')
                dist_resistance_pct = abs(features['sr_dist_to_nearest_resistance_pct'].iloc[i]) if 'sr_dist_to_nearest_resistance_pct' in features.columns else float('inf')
                
                # Convert to ATR units for tolerance comparison
                tolerance_pct = (atr * self.tolerance_atr_multiplier) / current_price
                
                # Determine state using percentage-based tolerance
                if dist_support_pct <= tolerance_pct and dist_resistance_pct <= tolerance_pct:
                    # Near both - determine which is closer
                    if dist_support_pct < dist_resistance_pct:
                        state = 'approaching_support'
                    else:
                        state = 'approaching_resistance'
                elif dist_support_pct <= tolerance_pct:
                    # Check direction of movement using percentage returns
                    if i > 0:
                        price_change_pct = (current_price - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                        if price_change_pct < 0:  # Price falling
                            state = 'approaching_support'
                        else:  # Price rising
                            state = 'moving_away_from_support'
                    else:
                        state = 'approaching_support'
                elif dist_resistance_pct <= tolerance_pct:
                    # Check direction of movement using percentage returns
                    if i > 0:
                        price_change_pct = (current_price - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                        if price_change_pct > 0:  # Price rising
                            state = 'approaching_resistance'
                        else:  # Price falling
                            state = 'moving_away_from_resistance'
                    else:
                        state = 'approaching_resistance'
                else:
                    state = 'neutral'
                
                proximity_states.append(state)
            
            # Create one-hot encoded features
            state_dummies = pd.get_dummies(proximity_states, prefix='sr_proximity_state')
            for col in state_dummies.columns:
                features[col] = state_dummies[col].values
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Proximity direction features creation failed: {e}')
            return features

    def _create_breakout_rejection_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                          sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create breakout vs rejection flags."""
        try:
            tolerance = atr * self.tolerance_atr_multiplier
            breakout_support = []
            breakout_resistance = []
            rejection_support = []
            rejection_resistance = []
            
            for i in range(len(data)):
                if i < 2:  # Need at least 2 previous bars
                    breakout_support.append(0)
                    breakout_resistance.append(0)
                    rejection_support.append(0)
                    rejection_resistance.append(0)
                    continue
                
                current_price = data['close'].iloc[i]
                prev_price = data['close'].iloc[i-1]
                prev2_price = data['close'].iloc[i-2]
                
                # Get nearest levels
                support_levels = sr_levels.get('support_levels', [])
                resistance_levels = sr_levels.get('resistance_levels', [])
                
                # Find nearest support and resistance
                nearest_support = None
                nearest_resistance = None
                
                if support_levels:
                    support_prices = [level.get('price', level) for level in support_levels if isinstance(level.get('price', level), (int, float))]
                    if support_prices:
                        nearest_support = min(support_prices, key=lambda x: abs(current_price - x))
                
                if resistance_levels:
                    resistance_prices = [level.get('price', level) for level in resistance_levels if isinstance(level.get('price', level), (int, float))]
                    if resistance_prices:
                        nearest_resistance = min(resistance_prices, key=lambda x: abs(current_price - x))
                
                # Check for breakouts and rejections
                support_breakout = 0
                support_rejection = 0
                resistance_breakout = 0
                resistance_rejection = 0
                
                if nearest_support is not None:
                    # Check if price broke below support
                    if prev_price > nearest_support and current_price < nearest_support:
                        support_breakout = 1
                    # Check if price bounced off support
                    elif abs(prev_price - nearest_support) <= tolerance and current_price > prev_price:
                        support_rejection = 1
                
                if nearest_resistance is not None:
                    # Check if price broke above resistance
                    if prev_price < nearest_resistance and current_price > nearest_resistance:
                        resistance_breakout = 1
                    # Check if price bounced off resistance
                    elif abs(prev_price - nearest_resistance) <= tolerance and current_price < prev_price:
                        resistance_rejection = 1
                
                breakout_support.append(support_breakout)
                breakout_resistance.append(resistance_breakout)
                rejection_support.append(support_rejection)
                rejection_resistance.append(resistance_rejection)
            
            features['sr_breakout_support'] = breakout_support
            features['sr_breakout_resistance'] = breakout_resistance
            features['sr_rejection_support'] = rejection_support
            features['sr_rejection_resistance'] = rejection_resistance
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Breakout rejection features creation failed: {e}')
            return features

    def _create_relative_momentum_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                         sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create relative momentum to level features."""
        try:
            # Calculate momentum
            momentum_short = data['close'].pct_change(5)  # 5-period momentum
            momentum_medium = data['close'].pct_change(10)  # 10-period momentum
            
            # Relative momentum features
            relative_momentum_support = []
            relative_momentum_resistance = []
            
            for i in range(len(data)):
                current_momentum = momentum_short.iloc[i] if not pd.isna(momentum_short.iloc[i]) else 0
                
                # Get distance to nearest levels (use percentage returns for better normalization)
                dist_support_pct = abs(features['sr_dist_to_nearest_support_pct'].iloc[i]) if 'sr_dist_to_nearest_support_pct' in features.columns else float('inf')
                dist_resistance_pct = abs(features['sr_dist_to_nearest_resistance_pct'].iloc[i]) if 'sr_dist_to_nearest_resistance_pct' in features.columns else float('inf')
                
                # Calculate relative momentum using percentage returns
                # Positive momentum approaching resistance = high breakout likelihood
                # Negative momentum approaching support = high breakout likelihood
                if dist_support_pct < dist_resistance_pct:
                    # Closer to support
                    relative_momentum_support.append(current_momentum * (1 / (dist_support_pct + 1e-8)))
                    relative_momentum_resistance.append(0)
                else:
                    # Closer to resistance
                    relative_momentum_support.append(0)
                    relative_momentum_resistance.append(current_momentum * (1 / (dist_resistance_pct + 1e-8)))
            
            features['sr_relative_momentum_support'] = relative_momentum_support
            features['sr_relative_momentum_resistance'] = relative_momentum_resistance
            
            # Momentum direction features using percentage returns
            features['sr_momentum_approaching_support'] = ((momentum_short < 0) & (features['sr_delta_dist_support_pct'] < 0)).astype(int)
            features['sr_momentum_approaching_resistance'] = ((momentum_short > 0) & (features['sr_delta_dist_resistance_pct'] < 0)).astype(int)
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Relative momentum features creation failed: {e}')
            return features

    def _create_time_since_approach_features(self, features: pd.DataFrame, data: pd.DataFrame,
                                           sr_levels: Dict[str, Any], atr: float) -> pd.DataFrame:
        """Create time since last approach features."""
        try:
            tolerance = atr * self.tolerance_atr_multiplier
            
            time_since_approach_support = []
            time_since_approach_resistance = []
            
            for i in range(len(data)):
                current_price = data['close'].iloc[i]
                
                # Find last time price was within tolerance of support (using percentage returns)
                last_support_approach = -1
                tolerance_pct = tolerance / current_price
                for j in range(max(0, i-100), i):  # Look back up to 100 bars
                    price_change_pct = abs(data['close'].iloc[j] - current_price) / current_price
                    if price_change_pct <= tolerance_pct:
                        # Check if there was a support level nearby
                        support_levels = sr_levels.get('support_levels', [])
                        for level in support_levels:
                            if isinstance(level.get('price', level), (int, float)):
                                level_dist_pct = abs(data['close'].iloc[j] - level.get('price', level)) / data['close'].iloc[j]
                                if level_dist_pct <= tolerance_pct:
                                    last_support_approach = i - j
                                    break
                        if last_support_approach != -1:
                            break
                
                # Find last time price was within tolerance of resistance (using percentage returns)
                last_resistance_approach = -1
                for j in range(max(0, i-100), i):  # Look back up to 100 bars
                    price_change_pct = abs(data['close'].iloc[j] - current_price) / current_price
                    if price_change_pct <= tolerance_pct:
                        # Check if there was a resistance level nearby
                        resistance_levels = sr_levels.get('resistance_levels', [])
                        for level in resistance_levels:
                            if isinstance(level.get('price', level), (int, float)):
                                level_dist_pct = abs(data['close'].iloc[j] - level.get('price', level)) / data['close'].iloc[j]
                                if level_dist_pct <= tolerance_pct:
                                    last_resistance_approach = i - j
                                    break
                        if last_resistance_approach != -1:
                            break
                
                time_since_approach_support.append(last_support_approach if last_support_approach != -1 else 999)
                time_since_approach_resistance.append(last_resistance_approach if last_resistance_approach != -1 else 999)
            
            features['sr_time_since_approach_support'] = time_since_approach_support
            features['sr_time_since_approach_resistance'] = time_since_approach_resistance
            
            # Categorical features for fresh vs stale levels
            features['sr_fresh_support_test'] = (features['sr_time_since_approach_support'] <= 5).astype(int)
            features['sr_fresh_resistance_test'] = (features['sr_time_since_approach_resistance'] <= 5).astype(int)
            features['sr_stale_support_level'] = (features['sr_time_since_approach_support'] > 20).astype(int)
            features['sr_stale_resistance_level'] = (features['sr_time_since_approach_resistance'] > 20).astype(int)
            
            return features
            
        except Exception as e:
            self.logger.warning(f'Time since approach features creation failed: {e}')
            return features