from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

class AgnosticCusumFilter:
    """
    An agnostic CUSUM filter that self-adjusts to volatility and target event frequency.
    """
    def __init__(self, target_events_per_day: float = 7.5, vol_window: int = 100):
        self.target_events_per_day = target_events_per_day
        self.vol_window = vol_window
        self.threshold = 1.0  # Initial threshold, will be calibrated

    def calibrate(self, feature_series: pd.Series, time_span_days: float) -> float:
        """
        Calibrate threshold to achieve target event frequency.
        """
        if time_span_days <= 0:
            return self.threshold

        target_events = self.target_events_per_day * time_span_days

        # Binary search for threshold
        low, high = 0.01, 10.0
        best_h = self.threshold
        min_diff = float('inf')

        for _ in range(20):
            mid = (low + high) / 2
            events = self._run_cusum(feature_series, mid)
            n_events = len(events)

            diff = abs(n_events - target_events)
            if diff < min_diff:
                min_diff = diff
                best_h = mid

            if n_events > target_events:
                low = mid  # Need higher threshold to reduce events (if feature is normalized?)
                # Wait, usually higher threshold = fewer events.
                # If n_events > target, we have too many events. We need to INCREASE threshold.
                # So low = mid is correct? No, mid should be the new lower bound?
                # If mid gave too many events, mid is too low. So we search in [mid, high].
                # Yes, low = mid.
            else:
                high = mid

        self.threshold = best_h
        return best_h

    def _run_cusum(self, series: pd.Series, h: float) -> pd.DatetimeIndex:
        """Run CUSUM on normalized series."""
        # Simple symmetric CUSUM on absolute values or raw series?
        # The user said "Trigger represents a standard deviation move in that specific feature's space".
        # Assuming series is already z-scored or normalized.
        # We accumulate deviations.

        # "accumulate Standardized Volume Shock: (Vt - median(V))/sigmaV"
        # So we accumulate the series values themselves.

        s_pos, s_neg = 0.0, 0.0
        events = []
        idx = series.index
        vals = series.values

        for i in range(len(vals)):
            x = vals[i]
            if np.isnan(x): continue

            s_pos = max(0.0, s_pos + x)
            s_neg = min(0.0, s_neg + x)

            if s_pos > h:
                s_pos = 0.0
                events.append(idx[i])
            elif s_neg < -h:
                s_neg = 0.0
                events.append(idx[i])

        return pd.DatetimeIndex(events)

    def generate_signals(self, feature_series: pd.Series) -> pd.DatetimeIndex:
        """
        Generate CUSUM signals.
        Assumes feature_series is already normalized (e.g. z-score).
        """
        if len(feature_series) < 2:
            return pd.DatetimeIndex([])

        time_span_days = (feature_series.index[-1] - feature_series.index[0]).total_seconds() / 86400.0
        self.calibrate(feature_series, time_span_days)

        return self._run_cusum(feature_series, self.threshold)


class CausalFeatureGenerator:
    """
    Generates specific causal features as requested.
    """

    @staticmethod
    def volatility_main_feature(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Volatility Main Feature: Garman-Klass or EWMA of squared returns.
        Using Garman-Klass for low lag.
        """
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            # Garman-Klass
            log_hl = np.log(df['high'] / df['low']) ** 2
            log_co = np.log(df['close'] / df['open']) ** 2
            gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
            return np.sqrt(gk_var)
        else:
            # Fallback: EWMA of squared returns
            ret = df['close'].pct_change()
            return np.sqrt((ret**2).ewm(span=window).mean())

    @staticmethod
    def path_smoothness_feature(df: pd.DataFrame, window: int = 8) -> pd.Series:
        """
        Path Smoothness: Kaufman's Adaptive Efficiency Ratio with Gaussian smoothing.
        """
        change = df['close'].diff(window).abs()
        volatility = df['close'].diff().abs().rolling(window).sum()
        er = change / (volatility + 1e-9)

        # Gaussian smoothing (approx with minimal lag using EWM)
        return er.ewm(span=3).mean()

    @staticmethod
    def volume_main_feature(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Volume Main Feature: Standardized Volume Shock.
        (Vt - median(V))/sigmaV
        """
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)

        vol = df['volume']
        median = vol.rolling(window).median()
        sigma = vol.rolling(window).std()

        return (vol - median) / (sigma + 1e-9)

    @staticmethod
    def kalman_trend_feature(df: pd.DataFrame) -> pd.Series:
        """
        Market Reversion / Trend: Constant Acceleration Kalman Filter Innovation.
        """
        # Simplified Constant Velocity/Acceleration Kalman Filter
        # State: [x, v, a]
        # We track innovation (prediction error)

        price = df['close'].values
        n = len(price)

        # Parameters
        dt = 1.0
        # State transition matrix F
        F = np.array([[1, dt, 0.5*dt**2],
                      [0, 1, dt],
                      [0, 0, 1]])
        # Measurement matrix H
        H = np.array([[1, 0, 0]])
        # Process noise Q
        Q = np.eye(3) * 1e-5
        # Measurement noise R
        R = 0.01

        x_hat = np.zeros((3, 1))
        x_hat[0] = price[0]
        P = np.eye(3)

        innovations = np.zeros(n)

        for i in range(n):
            z = price[i]

            # Predict
            x_pred = F @ x_hat
            P_pred = F @ P @ F.T + Q

            # Update
            y = z - (H @ x_pred) # Innovation
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            x_hat = x_pred + K @ y
            P = (np.eye(3) - K @ H) @ P_pred

            innovations[i] = y[0, 0]

        return pd.Series(innovations, index=df.index)

    @staticmethod
    def momentum_persistence_feature(df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Momentum Persistence: Z-score of run length / Bayesian update.
        Simplification: Rolling sum of signs.
        """
        ret = df['close'].diff()
        signs = np.sign(ret)

        # Run length proxy: rolling sum of signs over window
        run_strength = signs.rolling(window).sum().abs()

        # Probability of continuation (Bayesian update proxy)
        # If run_strength is high, persistence is high.
        # Normalize to z-score
        mean = run_strength.rolling(100).mean()
        std = run_strength.rolling(100).std()

        return (run_strength - mean) / (std + 1e-9)

    @staticmethod
    def liquidity_feature(df: pd.DataFrame) -> pd.Series:
        """
        Liquidity: Corwin-Schultz Spread Estimator.
        """
        if not all(c in df.columns for c in ['high', 'low']):
            return pd.Series(0, index=df.index)

        high = df['high']
        low = df['low']

        # Beta calculation
        # sum of squared log high/low ranges over 2 days
        hl_ratio = np.log(high / low) ** 2
        beta = hl_ratio.rolling(2).sum()

        # Gamma calculation
        # log(max(H2)/min(L2))^2
        h2 = high.rolling(2).max()
        l2 = low.rolling(2).min()
        gamma = np.log(h2 / l2) ** 2

        # Alpha (Corrected Corwin-Schultz formula)
        # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
        den = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)

        # Spread
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

        return spread.fillna(0.0)

    @staticmethod
    def shannon_entropy_feature(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Shannon Entropy: Approximate Entropy on binary price changes (in bits).
        """
        ret = df['close'].diff()
        binary = (ret > 0).astype(int)

        # Sliding window entropy
        # Simplified: Entropy of binary distribution in window
        def calc_entropy(x):
            counts = np.bincount(x.astype(int))
            probs = counts / len(x)
            return entropy(probs, base=2)

        return binary.rolling(window).apply(calc_entropy, raw=True)

    @staticmethod
    def order_imbalance_feature(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Order Imbalance: Tick Rule Proxy.
        """
        if not all(c in df.columns for c in ['close', 'high', 'low', 'volume']):
            return pd.Series(0, index=df.index)

        # Tick rule proxy using OHLC
        # If Close > (High + Low) / 2 -> Buyer Aggressive
        mid = (df['high'] + df['low']) / 2
        is_buy = df['close'] > mid

        signed_vol = np.where(is_buy, df['volume'], -df['volume'])
        imbalance = pd.Series(signed_vol, index=df.index).rolling(window).sum()

        # Volume-Weighted Imbalance normalized by total volume
        total_vol = df['volume'].rolling(window).sum()

        return imbalance / (total_vol + 1e-9)

    @staticmethod
    def time_of_day_features(index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Sessionality: Sin/Cos of hour of day.
        Captures cyclic nature of market sessions (Asian, London, NY).
        """
        # Ensure index is datetime
        if not isinstance(index, pd.DatetimeIndex):
            try:
                index = pd.to_datetime(index)
            except Exception:
                return pd.DataFrame(0.0, index=index, columns=['sin_time', 'cos_time'])

        # Extract hour (and minute fraction)
        # We normalize to [0, 2pi]
        # Using 24 hours cycle

        # If naive, assume UTC or local as is
        # Calculate time in hours (0-24)
        time_hours = index.hour + index.minute / 60.0

        # Transform
        sin_time = np.sin(2 * np.pi * time_hours / 24.0)
        cos_time = np.cos(2 * np.pi * time_hours / 24.0)

        return pd.DataFrame({
            'sin_time': sin_time,
            'cos_time': cos_time
        }, index=index)

    @staticmethod
    def volatility_of_volatility_feature(volatility_series: pd.Series, window: int = 10) -> pd.Series:
        """
        Volatility-of-Volatility: Std dev of volatility over window.
        W (Nuisance) variable.
        """
        return volatility_series.rolling(window).std().fillna(0.0)
