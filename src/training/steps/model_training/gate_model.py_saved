"""
Gate Model Implementation.

This module defines the GateModel class, which acts as a filter for trading signals.
It uses a sparse, interpretable ElasticNet regression model to reject trades in unfavorable regimes.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
import math
import itertools
from scipy.stats import entropy

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

class GateModel:
    """
    Gate Model for filtering trading signals.

    This model predicts the expected PnL of a trade (Regression)
    based on market regime and trade history features.
    It is designed to be sparse, interpretable, and fast.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the GateModel.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}

        # Metadata defaults
        self.exchange = self.config.get('exchange', 'unknown')
        self.symbol = self.config.get('symbol', 'unknown')
        self.timeframe = self.config.get('timeframe', 'unknown')
        self.direction = self.config.get('direction', 'long')

        # Default behavior: if no explicit calibration settings are provided,
        # we aim to block the worst trades (e.g. bottom 25%).
        # target_coverage=0.75 means we keep the top 75% of trades.
        if (
            'min_predicted_pnl' not in self.config
            and 'calibration_percentile' not in self.config
            and 'target_coverage' not in self.config
        ):
            self.config['target_coverage'] = 0.75

        self.pipeline = None
        self.threshold = 0.0  # Default threshold (breakeven), updated after training
        self.feature_names = []

        # Hyperparameters for ElasticNetCV
        self.l1_ratios = self.config.get('l1_ratios', [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
        self.cv_splits = self.config.get('cv_splits', 5)
        self.max_iter = self.config.get('max_iter', 5000)
        self.n_jobs = self.config.get('n_jobs', -1)

        # ExtraTrees-based gate configuration ("Dumb Manager")
        self.n_estimators = self.config.get('n_estimators', 500)
        self.max_depth = self.config.get('max_depth', 3)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 0.05)
        self.max_features = self.config.get('max_features', 'sqrt')
        self.bootstrap = self.config.get('bootstrap', True)
        self.class_weight = self.config.get('class_weight', 'balanced')
        self.random_state = self.config.get('random_state', 42)

        # Optional probability calibration
        self.calibration_model = None
        self.calibration_method = self.config.get('calibration_method', 'isotonic')

        self.enable_shap = bool(self.config.get('enable_shap', True))
        self.max_shap_samples = int(self.config.get('max_shap_samples', 5000))
        self.shap_explainer = None
        self.shap_values = None
        self.shap_feature_importance_ = None
        self.shap_feature_names_ = None

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
        rv_med = log_ret.rolling(window=48).std() * np.sqrt(48)
        features['rv_short_over_med'] = features['rv_short'] / (rv_med + 1e-8)

        # ATR Proxy (High-Low range / Close) - faster than full ATR
        tr = (high - low) / close
        atr_short = tr.rolling(window=12).mean()

        # Bollinger Band Width
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        bb_width = (4 * rolling_std) / rolling_mean

        # RV Z-Score (Short vs Long)
        rv_long = log_ret.rolling(window=200).std()
        # Avoid division by zero
        features['rv_z_short'] = (features['rv_short'] - rv_long) / (rv_long + 1e-8)

        # 2. Trend Strength Features
        # Slope of log price (Short)
        log_price = np.log(close)
        features['slope_short'] = log_price.diff(12).abs() # Simple proxy for slope

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
        features['momentum_short'] = (close.diff(12) / close.shift(12)).abs()

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

        int_index = pd.Series(np.arange(len(df)), index=df.index)

        if len(spike_indices) > 0:
            spike_times.loc[spike_indices] = spike_indices
            spike_times = spike_times.ffill()

            # Calculate bars since (not time) as requested "bars"
            # Since index might not be uniform, we can use rank/position
            # Efficient way for bars:
            # Create an integer index series
            last_spike_int_idx = int_index.where(is_vol_spike == 1).ffill()
            features['time_since_last_vol_spike'] = int_index - last_spike_int_idx
            features['time_since_last_vol_spike'] = features['time_since_last_vol_spike'].fillna(1000)
        else:
            features['time_since_last_vol_spike'] = 1000.0

        # Large Candle (Range > 2.5 * ATR)
        candle_range = high - low
        # Use existing atr_short (12 period)
        is_large_candle = (candle_range > 2.5 * atr_short).astype(int)

        large_candle_int_idx = int_index.where(is_large_candle == 1).ffill()
        features['time_since_last_large_candle'] = int_index - large_candle_int_idx
        features['time_since_last_large_candle'] = features['time_since_last_large_candle'].fillna(1000)

        # 5. Advanced Regime Features (Choppiness, Variance Ratio, Permutation Entropy)

        # Choppiness Index (Bill Dreiss)
        # 100 * LOG10( SUM(ATR(1), n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
        chop_window = 20
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        sum_tr = tr.rolling(chop_window).sum()
        max_hi = high.rolling(chop_window).max()
        min_lo = low.rolling(chop_window).min()
        range_hl = max_hi - min_lo

        features['choppiness_index'] = 100 * np.log10(sum_tr / (range_hl + 1e-8)) / np.log10(chop_window)

        # Variance Ratio Test (Lo & MacKinlay)
        # Var(20-period log returns) / (2 * Var(10-period log returns))
        # Note: 20/10 = 2, so we divide by 2 to normalize to 1.0
        vr_window = 50
        r_20 = log_ret.rolling(20).sum()
        r_10 = log_ret.rolling(10).sum()
        var_20 = r_20.rolling(vr_window).var()
        var_10 = r_10.rolling(vr_window).var()
        features['variance_ratio'] = var_20 / (2 * var_10 + 1e-8)

        # Permutation Entropy (Efficient Implementation)
        pe_window = 50
        pe_dim = 3
        pe_values = close.values
        pe_n = len(pe_values)

        if pe_n >= pe_window + pe_dim:
            # Generate all patterns for the entire series using stride tricks
            # Requires careful handling to avoid heavy dependencies if possible,
            # but standard numpy stride tricks are efficient.
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(pe_values, window_shape=pe_dim)
            except ImportError:
                # Fallback for older numpy
                shape = (pe_n - pe_dim + 1, pe_dim)
                strides = (pe_values.strides[0], pe_values.strides[0])
                windows = np.lib.stride_tricks.as_strided(pe_values, shape=shape, strides=strides)

            # Convert to pattern codes (rank order)
            patterns = np.argsort(windows, axis=1)
            # Map permutations to unique integers 0..dim!-1
            perms = list(itertools.permutations(range(pe_dim)))
            perm_to_code = {p: i for i, p in enumerate(perms)}
            codes = np.apply_along_axis(lambda x: perm_to_code[tuple(x)], 1, patterns)

            # Rolling entropy on codes
            code_series = pd.Series(codes, index=df.index[pe_dim - 1:])

            def calc_ent(x):
                # x is array of codes
                counts = np.unique(x, return_counts=True)[1]
                probs = counts / counts.sum()
                # Normalize by log2(factorial(dim))
                max_ent = np.log2(math.factorial(pe_dim))
                ent = entropy(probs, base=2)
                return ent / max_ent

            # Use pandas rolling apply
            rolling_ent = code_series.rolling(pe_window).apply(calc_ent, raw=True)
            features['permutation_entropy'] = rolling_ent.reindex(df.index)
        else:
            features['permutation_entropy'] = np.nan

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
        tl['rolling_winrate_5'] = tl['win'].rolling(5, min_periods=1).mean()
        tl['rolling_winrate_10'] = tl['win'].rolling(10, min_periods=1).mean()
        tl['rolling_winrate_15'] = tl['win'].rolling(15, min_periods=1).mean()

        # Rolling Avg PnL (last 20 trades)
        tl['rolling_avg_pl'] = tl['profit'].rolling(20, min_periods=1).mean()
        tl['rolling_avg_pl_5'] = tl['profit'].rolling(5, min_periods=1).mean()
        tl['rolling_avg_pl_10'] = tl['profit'].rolling(10, min_periods=1).mean()
        tl['rolling_avg_pl_15'] = tl['profit'].rolling(15, min_periods=1).mean()

        # Consecutive Losses
        # Vectorized way to count consecutive groups
        # We want the count of consecutive losses ENDING at this row
        # Identify streaks
        y = tl['loss']
        tl['consecutive_losses'] = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

        # 2. Map these "post-trade" states to the timeline
        # A trade's outcome is known at 'exit_time'. So from 'exit_time' onwards, the history features update.
        # We index the trade log by 'exit_time'

        state_df = tl[
            [
                'exit_time',
                'rolling_winrate_20',
                'rolling_winrate_5',
                'rolling_winrate_10',
                'rolling_winrate_15',
                'rolling_avg_pl',
                'rolling_avg_pl_5',
                'rolling_avg_pl_10',
                'rolling_avg_pl_15',
                'consecutive_losses',
            ]
        ].set_index('exit_time')

        # Remove duplicate index (multiple trades exiting same time), keep last
        state_df = state_df[~state_df.index.duplicated(keep='last')]

        # Reindex to full OHLCV timeline using forward fill (asof)
        # This propagates the state after the last known trade exit to future bars
        aligned_state = state_df.reindex(ohlcv.index, method='ffill')

        # Fill NaNs (before first trade)
        aligned_state['rolling_winrate_20'] = aligned_state['rolling_winrate_20'].fillna(0.5)
        aligned_state['rolling_winrate_5'] = aligned_state['rolling_winrate_5'].fillna(0.5)
        aligned_state['rolling_winrate_10'] = aligned_state['rolling_winrate_10'].fillna(0.5)
        aligned_state['rolling_winrate_15'] = aligned_state['rolling_winrate_15'].fillna(0.5)
        aligned_state['rolling_avg_pl'] = aligned_state['rolling_avg_pl'].fillna(0.0)
        aligned_state['rolling_avg_pl_5'] = aligned_state['rolling_avg_pl_5'].fillna(0.0)
        aligned_state['rolling_avg_pl_10'] = aligned_state['rolling_avg_pl_10'].fillna(0.0)
        aligned_state['rolling_avg_pl_15'] = aligned_state['rolling_avg_pl_15'].fillna(0.0)
        aligned_state['consecutive_losses'] = aligned_state['consecutive_losses'].fillna(0)

        features['rolling_winrate_20'] = aligned_state['rolling_winrate_20']
        features['rolling_winrate_5'] = aligned_state['rolling_winrate_5']
        features['rolling_winrate_10'] = aligned_state['rolling_winrate_10']
        features['rolling_winrate_15'] = aligned_state['rolling_winrate_15']
        features['rolling_avg_pl'] = aligned_state['rolling_avg_pl']
        features['rolling_avg_pl_5'] = aligned_state['rolling_avg_pl_5']
        features['rolling_avg_pl_10'] = aligned_state['rolling_avg_pl_10']
        features['rolling_avg_pl_15'] = aligned_state['rolling_avg_pl_15']
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

    def prepare_features(self, ohlcv: pd.DataFrame, trade_log: pd.DataFrame, preds: Optional[pd.Series] = None, base_model_preds: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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

        feature_blocks: List[pd.DataFrame] = [regime_feats, history_feats]

        if preds is not None:
            preds_series = preds
            if isinstance(preds_series, pd.DataFrame):
                preds_series = preds_series.iloc[:, 0]
            preds_series = preds_series.astype(float)
            preds_aligned = preds_series.reindex(ohlcv.index)

            analyst_feats = pd.DataFrame(
                {
                    'analyst_prediction': preds_aligned,
                },
                index=ohlcv.index,
            )
            feature_blocks.append(analyst_feats)

        if base_model_preds is not None:
            try:
                numeric_preds = base_model_preds.select_dtypes(include=[np.number])
                if isinstance(numeric_preds, pd.DataFrame) and numeric_preds.shape[1] > 0:
                    aligned_numeric = numeric_preds.reindex(ohlcv.index)
                    disagreement_feats = pd.DataFrame(index=ohlcv.index)
                    disagreement_feats['base_pred_mean'] = aligned_numeric.mean(axis=1)
                    disagreement_feats['base_pred_std'] = aligned_numeric.std(axis=1)
                    disagreement_feats['base_pred_range'] = aligned_numeric.max(axis=1) - aligned_numeric.min(axis=1)
                    feature_blocks.append(disagreement_feats)
            except Exception:
                pass

        X = pd.concat(feature_blocks, axis=1)

        # Save feature names
        self.feature_names = X.columns.tolist()

        return X

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """
        Train the Gate Model.

        Args:
            X: Feature DataFrame.
            y: Target labels (e.g., binary trade outcomes).
            sample_weight: Optional sample weights.
        """
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )),
        ])

        print(f"Training GateModel (ExtraTreesClassifier) with {X.shape[0]} samples and {X.shape[1]} features...")

        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs['model__sample_weight'] = sample_weight

        self.pipeline.fit(X, y, **fit_kwargs)

        model = self.pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            n_nonzero = int(np.sum(importances > 0))
            print(f"Trained ExtraTrees gate; non-zero feature importances: {n_nonzero}/{len(importances)}")

        if self.enable_shap and shap is not None:
            try:
                imputer = self.pipeline.named_steps.get('imputer')
                scaler = self.pipeline.named_steps.get('scaler')

                if isinstance(X, pd.DataFrame):
                    X_array = X.values
                    feature_names = X.columns.tolist()
                else:
                    X_array = np.asarray(X)
                    feature_names = [f"f{i}" for i in range(X_array.shape[1])]

                self.shap_feature_names_ = feature_names

                if imputer is not None:
                    X_array = imputer.transform(X_array)
                if scaler is not None:
                    X_array = scaler.transform(X_array)

                n_samples = X_array.shape[0]
                max_samples = max(1, int(self.max_shap_samples))
                if n_samples > max_samples:
                    rng = np.random.RandomState(self.random_state)
                    idx = rng.choice(n_samples, size=max_samples, replace=False)
                    X_shap = X_array[idx]
                else:
                    X_shap = X_array

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)

                self.shap_explainer = explainer
                self.shap_values = shap_values

                if isinstance(shap_values, list) and len(shap_values) >= 2:
                    sv = shap_values[1]
                else:
                    sv = shap_values

                mean_abs_shap = np.mean(np.abs(sv), axis=0)
                self.shap_feature_importance_ = mean_abs_shap

                order = np.argsort(mean_abs_shap)[::-1]
                top_k = min(10, len(order))
                print("Top SHAP features (by mean |SHAP|):")
                for i in order[:top_k]:
                    print(f"  {feature_names[i]}: {mean_abs_shap[i]:.6f}")
            except Exception as e:
                print(f"SHAP computation failed: {e}")

    def predict_raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict uncalibrated gate score from the underlying classifier."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")
        if hasattr(self.pipeline, 'predict_proba'):
            probs = self.pipeline.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            return probs.ravel()
        return self.pipeline.predict(X)

    def fit_calibrator(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Fit optional probability calibration model on a validation window."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")
        if X_val is None or y_val is None or len(X_val) == 0:
            return

        y_array = np.asarray(y_val).ravel()
        # Need at least two classes to calibrate
        if np.unique(y_array).size < 2:
            return

        try:
            raw_scores = self.predict_raw_score(X_val)
        except Exception:
            return

        method = str(self.calibration_method).lower()
        # For now, support isotonic regression; other values fall back to isotonic
        if method not in ("isotonic", "isotonic_regression"):
            method = "isotonic"

        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_scores, y_array)
            self.calibration_model = calibrator
        except Exception:
            # If calibration fails, fall back to raw scores
            self.calibration_model = None

    def calibrate_threshold(self, X: pd.DataFrame, percentile: int = 25):
        """Calibrate the gating threshold based on predicted PnL distribution.

        Logic:
        1) If ``min_predicted_pnl`` is in config, use that (e.g., 0.0 for breakeven).
        2) Else, if ``min_win_probability`` is set (e.g. 0.5), we set threshold to this.
        3) Else, if ``target_coverage`` is set (e.g. 0.75), we set threshold to the
           corresponding quantile (25th percentile) to block the bottom 25%.
        4) Fallback to ``percentile`` arg (default 25).
        """

        if self.pipeline is None:
            raise ValueError("Model not trained yet.")

        scores = self.predict_score(X)

        # 1) Direct PnL thresholding
        min_pnl = self.config.get('min_predicted_pnl', None)
        if isinstance(min_pnl, (int, float)):
            self.threshold = float(min_pnl)
            print(
                f"Threshold set from min_predicted_pnl={self.threshold:.6f} "
                "(blocking regions where predicted PnL < this)."
            )
            return

        # 1b) Direct probability threshold on predicted win probability
        min_win_prob = self.config.get('min_win_probability', None)
        if isinstance(min_win_prob, (int, float)):
            self.threshold = float(min_win_prob)
            print(
                f"Threshold set from min_win_probability={self.threshold:.6f} "
                "(blocking trades with predicted win probability below this)."
            )
            return

        # 2) Coverage-based calibration: keep top X% (target_coverage)
        target_cov = self.config.get('target_coverage', None)
        if isinstance(target_cov, (int, float)):
            tc = float(target_cov)
            # Clamp to a sensible range
            tc = max(0.01, min(0.99, tc))
            # If coverage is 0.75, we want to block the bottom 0.25
            calib_pct = 100.0 * (1.0 - tc)
            self.threshold = np.percentile(scores, calib_pct)
            print(
                f"Threshold calibrated from target_coverage={tc:.2f} "
                f"-> percentile={calib_pct:.1f}: {self.threshold:.6f}"
            )
            return

        # 3) Percentile-based calibration (default 25th percentile -> block bottom 25%)
        calib_pct = float(self.config.get('calibration_percentile', percentile))
        calib_pct = max(0.0, min(100.0, calib_pct))
        self.threshold = np.percentile(scores, calib_pct)
        print(f"Threshold calibrated at {calib_pct:.1f}th percentile: {self.threshold:.6f}")

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict continuous gate score (e.g., probability of accepting a trade)."""
        raw_scores = self.predict_raw_score(X)

        if self.calibration_model is not None:
            try:
                calibrated = self.calibration_model.predict(raw_scores)
                return np.clip(calibrated, 0.0, 1.0)
            except Exception:
                return raw_scores

        return raw_scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary decision: 1 (Trade) if score >= threshold, else 0 (Block)."""
        scores = self.predict_score(X)
        return (scores >= self.threshold).astype(int)

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
