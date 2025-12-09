"""
Gate Model Implementation.

This module defines the GateModel class, which acts as a filter for trading signals.
It uses a sparse, interpretable classifier to reject trades in unfavorable regimes.
Supported models: ExtraTreesClassifier, RidgeClassifier.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
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

    This model predicts the Probability of Success (Binary Classification)
    based on market regime and trade history features.
    It supports 'ExtraTreesClassifier' and 'RidgeClassifier'.
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

        # Model Type: 'extratrees' or 'ridge'
        self.model_type = self.config.get('model_type', 'extratrees').lower()

        # Thresholding defaults
        if 'min_win_probability' not in self.config:
            self.config['min_win_probability'] = 0.55

        self.pipeline = None
        self.threshold = float(self.config['min_win_probability'])
        self.feature_names = []

        # ExtraTrees-based gate configuration
        self.n_estimators = self.config.get('n_estimators', 500)
        self.max_depth = self.config.get('max_depth', 5)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 0.05)
        self.max_features = self.config.get('max_features', 'sqrt')
        self.bootstrap = self.config.get('bootstrap', True)
        self.class_weight = self.config.get('class_weight', 'balanced')

        # RidgeClassifier configuration
        self.alpha = self.config.get('alpha', 1.0)
        self.tol = self.config.get('tol', 1e-3)

        self.n_jobs = self.config.get('n_jobs', -1)
        self.random_state = self.config.get('random_state', 42)

        # Probability calibration
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
        (Same logic as previous implementation)
        """
        df = ohlcv.copy()
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']

        # Log returns
        log_ret = np.log(close / close.shift(1))

        # 1. Volatility Features
        features['rv_short'] = log_ret.rolling(window=12).std() * np.sqrt(12)
        rv_med = log_ret.rolling(window=48).std() * np.sqrt(48)
        features['rv_short_over_med'] = features['rv_short'] / (rv_med + 1e-8)

        tr = (high - low) / close
        atr_short = tr.rolling(window=12).mean()

        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()

        rv_long = log_ret.rolling(window=200).std()
        features['rv_z_short'] = (features['rv_short'] - rv_long) / (rv_long + 1e-8)

        # 2. Trend Strength Features
        log_price = np.log(close)
        features['slope_short'] = log_price.diff(12).abs()

        up_move = high.diff()
        down_move = low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = tr.rolling(window=14).sum()
        plus_di = pd.Series(plus_dm).rolling(window=14).sum() / (tr_smooth + 1e-8)
        minus_di = pd.Series(minus_dm).rolling(window=14).sum() / (tr_smooth + 1e-8)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        features['adx_proxy'] = dx.rolling(window=14).mean()

        # 3. Momentum
        features['momentum_short'] = (close.diff(12) / close.shift(12)).abs()
        features['snr'] = features['momentum_short'].abs() / (features['rv_short'] + 1e-8)

        # 4. New Features (Vol Spike, Large Candle, Time)
        rv_mean = features['rv_short'].rolling(window=100, min_periods=20).mean()
        rv_std = features['rv_short'].rolling(window=100, min_periods=20).std()
        rv_z = (features['rv_short'] - rv_mean) / (rv_std + 1e-8)

        is_vol_spike = (rv_z > 2.0).astype(int)

        int_index = pd.Series(np.arange(len(df)), index=df.index)
        last_spike_int_idx = int_index.where(is_vol_spike == 1).ffill()
        features['time_since_last_vol_spike'] = int_index - last_spike_int_idx
        features['time_since_last_vol_spike'] = features['time_since_last_vol_spike'].fillna(1000)

        candle_range = high - low
        is_large_candle = (candle_range > 2.5 * atr_short).astype(int)
        large_candle_int_idx = int_index.where(is_large_candle == 1).ffill()
        features['time_since_last_large_candle'] = int_index - large_candle_int_idx
        features['time_since_last_large_candle'] = features['time_since_last_large_candle'].fillna(1000)

        # 5. Advanced Regime Features
        chop_window = 20
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        sum_tr = tr.rolling(chop_window).sum()
        max_hi = high.rolling(chop_window).max()
        min_lo = low.rolling(chop_window).min()
        range_hl = max_hi - min_lo

        features['choppiness_index'] = 100 * np.log10(sum_tr / (range_hl + 1e-8)) / np.log10(chop_window)

        vr_window = 50
        r_20 = log_ret.rolling(20).sum()
        r_10 = log_ret.rolling(10).sum()
        var_20 = r_20.rolling(vr_window).var()
        var_10 = r_10.rolling(vr_window).var()
        features['variance_ratio'] = var_20 / (2 * var_10 + 1e-8)

        # Permutation Entropy
        pe_window = 50
        pe_dim = 3
        pe_values = close.values
        pe_n = len(pe_values)

        if pe_n >= pe_window + pe_dim:
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(pe_values, window_shape=pe_dim)
            except ImportError:
                shape = (pe_n - pe_dim + 1, pe_dim)
                strides = (pe_values.strides[0], pe_values.strides[0])
                windows = np.lib.stride_tricks.as_strided(pe_values, shape=shape, strides=strides)

            patterns = np.argsort(windows, axis=1)
            perms = list(itertools.permutations(range(pe_dim)))
            perm_to_code = {p: i for i, p in enumerate(perms)}
            codes = np.apply_along_axis(lambda x: perm_to_code[tuple(x)], 1, patterns)

            code_series = pd.Series(codes, index=df.index[pe_dim - 1:])

            def calc_ent(x):
                counts = np.unique(x, return_counts=True)[1]
                probs = counts / counts.sum()
                max_ent = np.log2(math.factorial(pe_dim))
                ent = entropy(probs, base=2)
                return ent / max_ent

            rolling_ent = code_series.rolling(pe_window).apply(calc_ent, raw=True)
            features['permutation_entropy'] = rolling_ent.reindex(df.index)
        else:
            features['permutation_entropy'] = np.nan

        # Time Features
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        else:
            features['hour_sin'] = 0.0
            features['hour_cos'] = 0.0
            features['is_weekend'] = 0

        return features

    def _compute_trade_history_features(self, ohlcv: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trade history features based on an ungated trade log.
        """
        features = pd.DataFrame(index=ohlcv.index)

        if trade_log.empty:
            features['time_since_last_trade'] = 1000.0
            features['num_trades_last_100h'] = 0
            features['rolling_winrate_20'] = 0.5
            features['consecutive_losses'] = 0
            features['rolling_avg_pl'] = 0.0
            return features

        tl = trade_log.sort_values('exit_time').copy()
        tl['win'] = (tl['profit'] > 0).astype(int)
        tl['loss'] = (tl['profit'] <= 0).astype(int)

        tl['rolling_winrate_20'] = tl['win'].rolling(20, min_periods=1).mean()
        tl['rolling_winrate_5'] = tl['win'].rolling(5, min_periods=1).mean()
        tl['rolling_winrate_10'] = tl['win'].rolling(10, min_periods=1).mean()
        tl['rolling_winrate_15'] = tl['win'].rolling(15, min_periods=1).mean()

        tl['rolling_avg_pl'] = tl['profit'].rolling(20, min_periods=1).mean()
        tl['rolling_avg_pl_5'] = tl['profit'].rolling(5, min_periods=1).mean()
        tl['rolling_avg_pl_10'] = tl['profit'].rolling(10, min_periods=1).mean()
        tl['rolling_avg_pl_15'] = tl['profit'].rolling(15, min_periods=1).mean()

        y = tl['loss']
        tl['consecutive_losses'] = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

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

        state_df = state_df[~state_df.index.duplicated(keep='last')]
        aligned_state = state_df.reindex(ohlcv.index, method='ffill')

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

        last_exit_time = pd.Series(np.nan, index=ohlcv.index)
        last_exit_time = last_exit_time.astype('object')
        common_indices = state_df.index.intersection(ohlcv.index)
        last_exit_time.loc[common_indices] = common_indices
        last_exit_time = last_exit_time.ffill()

        diff_series = ohlcv.index.to_series() - last_exit_time
        if hasattr(diff_series, 'dt'):
             time_diff = diff_series.dt.total_seconds() / 3600.0
        else:
             time_diff = pd.to_timedelta(diff_series).dt.total_seconds() / 3600.0

        features['time_since_last_trade'] = time_diff.fillna(1000.0)

        trade_exits = pd.Series(0, index=ohlcv.index)
        trade_exits.loc[state_df.index.intersection(ohlcv.index)] = 1

        try:
            features['num_trades_last_100h'] = trade_exits.rolling('100h').sum()
        except ValueError:
            window_size = 4 * 100
            features['num_trades_last_100h'] = trade_exits.rolling(window_size, min_periods=1).sum()

        return features

    def prepare_features(self, ohlcv: pd.DataFrame, trade_log: pd.DataFrame, preds: Optional[pd.Series] = None, base_model_preds: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compute and combine all features."""
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
        self.feature_names = X.columns.tolist()
        return X

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """
        Train the Gate Model Classifier.

        Args:
            X: Feature DataFrame.
            y: Target labels (Binary: 1=Success, 0=Fail).
            sample_weight: Weights for training samples (based on PnL magnitude).
        """
        model = None
        if self.model_type == 'ridge':
            # RidgeClassifier
            model = RidgeClassifier(
                alpha=self.alpha,
                tol=self.tol,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            print(f"Training GateModel (RidgeClassifier) with {X.shape[0]} samples...")
        else:
            # ExtraTreesClassifier (default)
            model = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            print(f"Training GateModel (ExtraTreesClassifier) with {X.shape[0]} samples...")

        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', model),
        ])

        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs['model__sample_weight'] = sample_weight

        self.pipeline.fit(X, y, **fit_kwargs)

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            n_nonzero = int(np.sum(importances > 0))
            print(f"Trained ExtraTrees gate; non-zero feature importances: {n_nonzero}/{len(importances)}")
        elif hasattr(model, 'coef_'):
            coef = model.coef_.ravel()
            n_nonzero = int(np.sum(np.abs(coef) > 1e-5))
            print(f"Trained Ridge gate; approx non-zero coefs: {n_nonzero}/{len(coef)}")

        if self.enable_shap and shap is not None and self.model_type == 'extratrees':
            try:
                self._compute_shap_importance(X)
            except Exception as e:
                print(f"SHAP computation failed: {e}")

    def _compute_shap_importance(self, X):
        model = self.pipeline.named_steps['model']
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

    def predict_raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw score or probability from the underlying classifier."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")

        model = self.pipeline.named_steps['model']

        if hasattr(self.pipeline, 'predict_proba'):
            probs = self.pipeline.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            return probs.ravel()
        elif hasattr(model, 'decision_function'):
            # RidgeClassifier has decision_function
            # We need to transform X first since pipeline.decision_function might not exist directly
            # or behaves differently depending on sklearn version/pipeline structure
            # Safest is to use pipeline to transform, then model.decision_function
            Xt = X
            for name, transform in self.pipeline.steps[:-1]:
                Xt = transform.transform(Xt)
            scores = model.decision_function(Xt)
            if scores.ndim > 1:
                scores = scores[:, 0]
            # Since we don't have probabilities, return raw scores.
            # Calibration will map these to [0,1]
            return scores

        return self.pipeline.predict(X)

    def fit_calibrator(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Fit probability calibration model on a validation window."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")
        if X_val is None or y_val is None or len(X_val) == 0:
            return

        y_array = np.asarray(y_val).ravel()
        if np.unique(y_array).size < 2:
            return

        try:
            raw_scores = self.predict_raw_score(X_val)
        except Exception:
            return

        method = str(self.calibration_method).lower()
        if method not in ("isotonic", "isotonic_regression"):
            method = "isotonic"

        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_scores, y_array)
            self.calibration_model = calibrator
        except Exception:
            self.calibration_model = None

    def calibrate_threshold(self, X: pd.DataFrame, percentile: int = 25):
        """Calibrate the gating threshold based on predicted probability distribution."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")

        scores = self.predict_score(X)

        # 1) Direct probability threshold
        min_win_prob = self.config.get('min_win_probability', None)
        if isinstance(min_win_prob, (int, float)):
            self.threshold = float(min_win_prob)
            print(
                f"Threshold set from min_win_probability={self.threshold:.6f} "
                "(blocking trades with predicted win probability below this)."
            )
            return

        # 2) Coverage-based calibration
        target_cov = self.config.get('target_coverage', None)
        if isinstance(target_cov, (int, float)):
            tc = float(target_cov)
            tc = max(0.01, min(0.99, tc))
            calib_pct = 100.0 * (1.0 - tc)
            self.threshold = np.percentile(scores, calib_pct)
            print(
                f"Threshold calibrated from target_coverage={tc:.2f} "
                f"-> percentile={calib_pct:.1f}: {self.threshold:.6f}"
            )
            return

        # 3) Percentile-based calibration
        calib_pct = float(self.config.get('calibration_percentile', percentile))
        calib_pct = max(0.0, min(100.0, calib_pct))
        self.threshold = np.percentile(scores, calib_pct)
        print(f"Threshold calibrated at {calib_pct:.1f}th percentile: {self.threshold:.6f}")

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probability of success."""
        raw_scores = self.predict_raw_score(X)

        if self.calibration_model is not None:
            try:
                calibrated = self.calibration_model.predict(raw_scores)
                return np.clip(calibrated, 0.0, 1.0)
            except Exception:
                return raw_scores

        # If uncalibrated Ridge, map decision function roughly to prob?
        # Better to rely on fit_calibrator. If not called, raw scores might be outside [0,1].
        # For ExtraTrees, raw_scores are already probs.
        if self.model_type == 'ridge':
             # Sigmoid approximation if no calibration available
             return 1 / (1 + np.exp(-raw_scores))

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
            'config': self.config,
            'calibration_model': self.calibration_model
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
        model.calibration_model = data.get('calibration_model')
        return model
