"""
Gate Model Implementation.

This module defines the GateModel class, which acts as a filter for trading signals.
It uses a sparse, interpretable ElasticNet model to reject trades in unfavorable regimes.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

class GateModel:
    """
    Gate Model for filtering trading signals.

    This model predicts the probability of a trade being profitable (Option A target)
    based on market regime and trade history features.
    It is designed to be sparse, interpretable, and fast.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GateModel.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.pipeline = None
        self.threshold = 0.5  # Default threshold, updated after training
        self.feature_names = []

        # Hyperparameters for LogisticRegressionCV
        self.l1_ratios = self.config.get('l1_ratios', [0.1, 0.3, 0.5, 0.7, 0.9])
        self.cv_splits = self.config.get('cv_splits', 5)
        self.max_iter = self.config.get('max_iter', 5000)
        self.n_jobs = self.config.get('n_jobs', -1)

    def _compute_regime_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute self-contained OHLCV-based regime features.

        Features:
        - Short-term volatility (RV, ATR, BB Width)
        - Trend strength (ADX proxy, Slope, R2)
        - Momentum

        Args:
            ohlcv: DataFrame with 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            DataFrame of regime features.
        """
        df = ohlcv.copy()
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']

        # Log returns
        log_ret = np.log(close / close.shift(1))

        # 1. Volatility Features
        # Realized Volatility (Short & Med)
        features['rv_short'] = log_ret.rolling(window=12).std() * np.sqrt(12)
        features['rv_med'] = log_ret.rolling(window=48).std() * np.sqrt(48)

        # ATR Proxy (High-Low range / Close) - faster than full ATR
        tr = (high - low) / close
        features['atr_short'] = tr.rolling(window=12).mean()

        # Bollinger Band Width
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        features['bb_width'] = (4 * rolling_std) / rolling_mean

        # RV Z-Score (Short vs Long)
        rv_long = log_ret.rolling(window=200).std()
        # Avoid division by zero
        features['rv_z_short'] = (features['rv_short'] - rv_long) / (rv_long + 1e-8)

        # 2. Trend Strength Features
        # Slope of log price (Short)
        log_price = np.log(close)
        features['slope_short'] = log_price.diff(12) # Simple proxy for slope

        # Trend Strength (ADX-like proxy using High-Low expansion)
        # Proper ADX is complex, using a simplified directional movement proxy
        up_move = high.diff()
        down_move = low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smoothing
        tr_smooth = tr.rolling(window=14).sum()
        plus_di = pd.Series(plus_dm).rolling(window=14).sum() / (tr_smooth + 1e-8)
        minus_di = pd.Series(minus_dm).rolling(window=14).sum() / (tr_smooth + 1e-8)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        features['adx_proxy'] = dx.rolling(window=14).mean()

        # 3. Momentum
        features['momentum_short'] = close.diff(12) / close.shift(12)

        # Signal-to-Noise Ratio (Momentum / Volatility)
        features['snr'] = features['momentum_short'].abs() / (features['rv_short'] + 1e-8)

        # 4. New Features (Vol Spike, Large Candle, Time)
        # Volatility Spike (Z-score > 2.0)
        # Using simple rolling mean/std for Z-score of rv_short
        rv_mean = features['rv_short'].rolling(window=100, min_periods=20).mean()
        rv_std = features['rv_short'].rolling(window=100, min_periods=20).std()
        rv_z = (features['rv_short'] - rv_mean) / (rv_std + 1e-8)

        # Binary spike mask
        is_vol_spike = (rv_z > 2.0).astype(int)

        # Time since last vol spike
        # Use cumulative sum to identify groups reset by spike
        # Or simpler: get indices where spike=1, reindex and ffill
        # Create a Series with values equal to index where spike occurs
        spike_times = pd.Series(np.nan, index=df.index)
        spike_indices = df.index[is_vol_spike == 1]

        # Cast to object to hold timestamps safely
        spike_times = spike_times.astype('object')

        if len(spike_indices) > 0:
            spike_times.loc[spike_indices] = spike_indices
            spike_times = spike_times.ffill()

            # Calculate bars since (not time) as requested "bars"
            # Since index might not be uniform, we can use rank/position
            # Efficient way for bars:
            # Create an integer index series
            int_index = pd.Series(np.arange(len(df)), index=df.index)
            last_spike_int_idx = int_index.where(is_vol_spike == 1).ffill()
            features['time_since_last_vol_spike'] = int_index - last_spike_int_idx
            features['time_since_last_vol_spike'] = features['time_since_last_vol_spike'].fillna(1000)
        else:
            features['time_since_last_vol_spike'] = 1000.0

        # Large Candle (Range > 2.5 * ATR)
        candle_range = high - low
        # Use existing atr_short (12 period)
        is_large_candle = (candle_range > 2.5 * features['atr_short']).astype(int)

        large_candle_int_idx = int_index.where(is_large_candle == 1).ffill()
        features['time_since_last_large_candle'] = int_index - large_candle_int_idx
        features['time_since_last_large_candle'] = features['time_since_last_large_candle'].fillna(1000)

        # Time Features (Cyclical)
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        else:
            # Fallback if index is not datetime (e.g. integer index)
            features['hour_sin'] = 0.0
            features['hour_cos'] = 0.0
            features['is_weekend'] = 0

        return features

    def _compute_trade_history_features(self, ohlcv: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trade history features based on an ungated trade log.

        This assumes the trade log contains ALL candidate trades (Main Model signals).
        Features are computed 'as of' the candidate timestamp.

        Args:
            ohlcv: DataFrame index (timestamps) to align features.
            trade_log: DataFrame with 'exit_time', 'profit', 'realized_return'.
                       Must be sorted by exit_time.

        Returns:
            DataFrame of trade history features aligned to ohlcv index.
        """
        # Initialize features with zeros/defaults
        features = pd.DataFrame(index=ohlcv.index)

        # If no trades, return defaults
        if trade_log.empty:
            features['time_since_last_trade'] = 1000.0 # High value
            features['num_trades_last_100h'] = 0
            features['rolling_winrate_20'] = 0.5
            features['consecutive_losses'] = 0
            features['rolling_avg_pl'] = 0.0
            return features

        # We need to map trade history state to each timestamp in ohlcv
        # This is efficiently done by 'asof' merging or resampling the trade log state

        # 1. Pre-calculate trade outcome metrics cumulatively/rolling on the trade log itself
        tl = trade_log.sort_values('exit_time').copy()
        tl['win'] = (tl['profit'] > 0).astype(int)
        tl['loss'] = (tl['profit'] <= 0).astype(int)

        # Rolling Winrate (last 20 trades)
        tl['rolling_winrate_20'] = tl['win'].rolling(20, min_periods=1).mean()

        # Rolling Avg PnL (last 20 trades)
        tl['rolling_avg_pl'] = tl['profit'].rolling(20, min_periods=1).mean()

        # Consecutive Losses
        # Vectorized way to count consecutive groups
        # We want the count of consecutive losses ENDING at this row
        # Identify streaks
        y = tl['loss']
        tl['consecutive_losses'] = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

        # 2. Map these "post-trade" states to the timeline
        # A trade's outcome is known at 'exit_time'. So from 'exit_time' onwards, the history features update.
        # We index the trade log by 'exit_time'

        state_df = tl[['exit_time', 'rolling_winrate_20', 'rolling_avg_pl', 'consecutive_losses']].set_index('exit_time')

        # Remove duplicate index (multiple trades exiting same time), keep last
        state_df = state_df[~state_df.index.duplicated(keep='last')]

        # Reindex to full OHLCV timeline using forward fill (asof)
        # This propagates the state after the last known trade exit to future bars
        aligned_state = state_df.reindex(ohlcv.index, method='ffill')

        # Fill NaNs (before first trade)
        aligned_state['rolling_winrate_20'] = aligned_state['rolling_winrate_20'].fillna(0.5)
        aligned_state['rolling_avg_pl'] = aligned_state['rolling_avg_pl'].fillna(0.0)
        aligned_state['consecutive_losses'] = aligned_state['consecutive_losses'].fillna(0)

        features['rolling_winrate_20'] = aligned_state['rolling_winrate_20']
        features['rolling_avg_pl'] = aligned_state['rolling_avg_pl']
        features['consecutive_losses'] = aligned_state['consecutive_losses']

        # 3. Time-based features (computed directly on timeline)
        # Time since last trade exit
        # Create a series with timestamps where trades exited
        last_exit_time = pd.Series(np.nan, index=ohlcv.index)
        # Cast to object or datetime compatible type before assignment if necessary,
        # but intersection usually preserves type. The issue is likely initializing with NaN (float)
        # then assigning datetimes.
        last_exit_time = last_exit_time.astype('object')

        common_indices = state_df.index.intersection(ohlcv.index)
        last_exit_time.loc[common_indices] = common_indices
        last_exit_time = last_exit_time.ffill()

        # Calculate hours difference
        # Assuming index is DatetimeIndex
        # 'ohlcv.index - last_exit_time' results in a TimedeltaIndex or Series of Timedeltas
        diff_series = ohlcv.index.to_series() - last_exit_time

        # Handle potential dtype issues if subtraction fails or returns object
        if hasattr(diff_series, 'dt'):
             time_diff = diff_series.dt.total_seconds() / 3600.0
        else:
             # Fallback: iterate or force coercion (slow but safe)
             # Ideally diff_series is already timedelta64[ns]
             time_diff = pd.to_timedelta(diff_series).dt.total_seconds() / 3600.0

        features['time_since_last_trade'] = time_diff.fillna(1000.0) # Default large value

        # Num trades last 100h
        # Create a binary series of trade exits (1 at exit time, 0 else)
        trade_exits = pd.Series(0, index=ohlcv.index)
        trade_exits.loc[state_df.index.intersection(ohlcv.index)] = 1

        # Rolling sum over 100h
        # Need to know samples per hour for window size
        # Estimate from index freq or assume 15m (4 per hour) -> 400 bars
        # Better: use '100h' offset string if index is proper datetime
        try:
            features['num_trades_last_100h'] = trade_exits.rolling('100h').sum()
        except ValueError:
            # Fallback for non-time-aware index or error
            window_size = 4 * 100 # Approx for 15m
            features['num_trades_last_100h'] = trade_exits.rolling(window_size, min_periods=1).sum()

        return features

    def prepare_features(self, ohlcv: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and combine all features.

        Args:
            ohlcv: OHLCV DataFrame.
            trade_log: Ungated trade log DataFrame.

        Returns:
            Feature matrix X (DataFrame).
        """
        regime_feats = self._compute_regime_features(ohlcv)
        history_feats = self._compute_trade_history_features(ohlcv, trade_log)

        X = pd.concat([regime_feats, history_feats], axis=1)

        # Save feature names
        self.feature_names = X.columns.tolist()

        return X

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """
        Train the Gate Model.

        Args:
            X: Feature DataFrame.
            y: Target Series (1 = Profitable, 0 = Loss).
            sample_weight: Optional sample weights.
        """
        # Pipeline: Impute -> Scale -> LogisticRegressionCV
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')), # Handle NaNs from rolling windows
            ('scaler', StandardScaler()),                  # Essential for ElasticNet
            ('clf', LogisticRegressionCV(
                Cs=20,
                cv=TimeSeriesSplit(n_splits=self.cv_splits),
                penalty='elasticnet',
                solver='saga', # Required for elasticnet
                l1_ratios=self.l1_ratios,
                scoring='roc_auc',
                max_iter=self.max_iter,
                n_jobs=self.n_jobs,
                random_state=42
            ))
        ])

        print(f"Training GateModel with {X.shape[0]} samples and {X.shape[1]} features...")
        self.pipeline.fit(X, y, clf__sample_weight=sample_weight)

        # Log coefficients
        clf = self.pipeline.named_steps['clf']
        best_l1 = clf.l1_ratio_[0]
        best_C = clf.C_[0]
        print(f"Best L1 Ratio: {best_l1}, Best C: {best_C}")

        # Check sparsity
        coefs = clf.coef_.flatten()
        n_zero = np.sum(coefs == 0)
        print(f"Sparsity: {n_zero}/{len(coefs)} features set to zero")

    def calibrate_threshold(self, X: pd.DataFrame, percentile: int = 40):
        """
        Set the threshold based on score percentile.

        Args:
            X: Feature DataFrame (can be training set or validation set).
            percentile: Percentile of scores to reject.
                        e.g., 40 means reject bottom 40% (accept top 60%).
                        User req: "threshold = percentile(scores, 60%)" implies keeping top X%.
                        Wait, requirements say: "percentile(past_500_scores, 60%) ... ensure only trade when confidence is among top X%"
                        If I set threshold at 60th percentile, I keep top 40%.
                        If I want to keep 60% of trades, I set threshold at 40th percentile.

                        Let's clarify: "threshold = percentile(..., 60%)".
                        Usually np.percentile(x, 60) gives value below which 60% of data falls.
                        So score > threshold means top 40%.

                        Requirements: "Keep ~60–80% of trades"
                        So we need to cut bottom 20-40%.
                        So threshold should be at 20th-40th percentile.

                        Let's default to 40th percentile (keep top 60%).
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")

        scores = self.pipeline.predict_proba(X)[:, 1]
        self.threshold = np.percentile(scores, percentile)
        print(f"Threshold calibrated at {percentile}th percentile: {self.threshold:.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary decision based on threshold."""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def save(self, filepath: str):
        """Save the model to disk."""
        joblib.dump({
            'pipeline': self.pipeline,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'config': self.config
        }, filepath)
        print(f"GateModel saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'GateModel':
        """Load model from disk."""
        data = joblib.load(filepath)
        model = cls(config=data.get('config'))
        model.pipeline = data['pipeline']
        model.threshold = data['threshold']
        model.feature_names = data['feature_names']
        return model
