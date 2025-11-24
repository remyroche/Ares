"""Mean-reversion regime detection step (OU/Hurst teacher + XGB student).

This step mirrors other market_analysis ML steps:
- loads OHLCV via BaseStep
- builds statistical teacher features for mean reversion
- trains an XGBoost regressor student
- calibrates outputs with an anchored z-score transform
- saves artifacts and a lightweight Markdown/CSV report in outcomes/.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score

try:
    from statsmodels.tsa.stattools import adfuller
    STATIONARITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    STATIONARITY_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

from scipy.stats import norm

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.features_common.transforms.scaling_normalization import (
    winsorized_zscore_normalize,
)
from src.training.steps.market_analysis.shared_utils.balanced_feature_extractor import (
    BalancedFeatureExtractor,
    BalancedFeatureConfig,
    FeatureCategory as BFCategory,
)
from src.utils.ml_common.trading_grid_backtester import run_simple_long_grid_backtest

logger = logging.getLogger(__name__)


def mean_reversion_objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Asymmetric Linex loss for distance-to-mean regression.

    scikit-learn XGBRegressor custom objective signature:
    obj(y_true, y_pred) -> (grad, hess).

    We define residual r = pred - true with a < 0 so that
    r < 0 (predicting *too small* distance → early reversion call)
    is penalised more heavily than r > 0 (late call).
    """
    residual = y_pred - y_true
    a = -5.0  # a < 0 → stronger penalty for r < 0 (early mean-reversion call)
    exp_term = np.exp(a * residual)
    # Linex loss L = exp(a r) - a r - 1 → dL/dpred = a (exp(a r) - 1)
    grad = a * (exp_term - 1.0)
    hess = (a ** 2) * exp_term
    return grad, hess


class MLMeanReversionRegimeStep(BaseStep):
    """Ornstein–Uhlenbeck / Hurst teacher → XGB student for mean reversion."""

    def __init__(self, step_name: str = "ml_mean_reversion_step") -> None:
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLMeanReversionRegimeStep") if hasattr(logger, "getChild") else logger
        self._cached_market_data: Optional[pd.DataFrame] = None
        self._cached_market_source: Optional[str] = None
        self._cached_market_cache_key: Optional[Tuple[str, str, str, str]] = None
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        start_time = time.time()
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for MLMeanReversionRegimeStep")

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            regime_timeframe = str(config.get("regime_timeframe", config.get("timeframe", "15m")))
            direction = str(config.get("direction", "long"))
            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            tprint_info(
                f"🚀 Starting {self.step_name} for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # 1) Load OHLCV (no light-mode filter, with caching)
            exec_mode_cfg = str(config.get("execution_mode", "")).lower()
            cache_key = (symbol, exchange, regime_timeframe, exec_mode_cfg)
            if self._cached_market_data is not None and self._cached_market_cache_key == cache_key:
                market_data = self._cached_market_data.copy()
                market_source = self._cached_market_source
                tprint_info("♻️ Reusing cached market data for mean-reversion step")
            else:
                market_data, market_source = self.load_market_data_or_fail(
                    {**config, "timeframe": regime_timeframe},
                    pipeline_state={},
                    allow_config_override=True,
                    light_mode_filter=False,
                    skip_artifacts=True,
                )
                self._cached_market_data = market_data.copy() if isinstance(market_data, pd.DataFrame) else market_data
                self._cached_market_source = market_source
                self._cached_market_cache_key = cache_key

            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")

            if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data = market_data.copy()
                market_data.index = pd.to_datetime(market_data.index)
                if market_data.index.tz is not None:
                    market_data.index = market_data.index.tz_convert(None)

            tprint_info(
                f"✅ Loaded market data from {market_source}: {market_data.shape} "
                f"({market_data.index.min()} → {market_data.index.max()})"
            )

            market_data = market_data.sort_index()
            required_cols = {"open", "high", "low", "close", "volume"}
            missing = [c for c in required_cols if c not in market_data.columns]
            if missing:
                raise ValueError(f"Market data missing OHLCV columns: {missing}")

            # 2) Teacher features + GMM labels + continuous reversion score
            teacher_df = self._build_teacher_features(market_data, config)
            (
                gmm,
                teacher_clusters,
                teacher_binary,
                teacher_score,
                teacher_metrics,
            ) = self._train_teacher_gmm(teacher_df, config)

            # 3) Student features + XGB regressor (predicting distance-to-mean)
            student_df = self._build_student_features(market_data, config)
            common_idx = teacher_score.index.intersection(student_df.index).sort_values()
            if len(common_idx) < 500:
                raise ValueError(f"Not enough aligned samples for training ({len(common_idx)} < 500)")

            X_all = student_df.loc[common_idx]

            # Distance-to-mean regression target: average absolute normalized
            # distance to slow MA and VWAP (bounded for robustness).
            dist_ma = X_all["z_price_ma_slow"].abs()
            dist_vwap = X_all["z_price_vwap"].abs()
            distance_target = 0.5 * (dist_ma + dist_vwap)
            max_dist = float(config.get("mr_distance_clip", 5.0))
            distance_target = distance_target.clip(lower=0.0, upper=max_dist)
            y_target_all = distance_target.astype(float)

            # Binary label from tightened teacher thresholds for supervision & metrics
            y_binary_all = teacher_binary.loc[common_idx].astype(int)

            model, student_metrics, raw_scores, calibrated_scores, calib_params = self._train_xgb_student(
                X_all,
                y_target_all,
                config,
                y_binary=y_binary_all,
            )

            # 4) Attach outputs to main frame
            output_df = market_data.copy()
            for c in teacher_df.columns:
                output_df[c] = teacher_df[c]
            output_df["mr_teacher_cluster"] = teacher_clusters
            output_df["mr_teacher_mean_reversion"] = teacher_binary
            output_df["mr_teacher_score"] = teacher_score
            for c in student_df.columns:
                output_df[c] = student_df[c]
            output_df.loc[X_all.index, "mr_raw_score"] = raw_scores
            output_df.loc[X_all.index, "mr_probability"] = calibrated_scores
            output_df.loc[X_all.index, "mr_distance_to_mean_target"] = y_target_all.values

            # Forward-return diagnostics at multiple horizons
            horizons_cfg = config.get("mr_forward_horizons", [5, 10, 20])
            fwd_metrics: Dict[int, Dict[str, Any]] = {}
            for h in horizons_cfg:
                try:
                    h_int = int(h)
                except (TypeError, ValueError):
                    continue
                m = self._compute_forward_metrics(
                    output_df, prob_col="mr_probability", horizon=h_int
                )
                if m:
                    fwd_metrics[h_int] = m

            # 5) Persist artifacts + reports
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="mean_reversion",
            )

            artifacts, reports = self._save_artifacts_and_reports(
                output_df=output_df,
                X_all=X_all,
                y_binary=y_binary_all,
                model=model,
                teacher_metrics=teacher_metrics,
                student_metrics=student_metrics,
                calib_params=calib_params,
                fwd_metrics=fwd_metrics,
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                market_source=str(market_source),
            )

            exec_time = time.time() - start_time
            tprint_success(f"✅ {self.step_name} completed in {exec_time:.2f}s with {len(X_all)} samples")

            return {
                "success": True,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "n_samples": int(len(X_all)),
                "metrics": {
                    "teacher": teacher_metrics,
                    "student": student_metrics,
                    "forward": fwd_metrics,
                },
                "artifacts": artifacts,
                "reports": reports,
                "execution_time": exec_time,
            }

        except Exception as exc:  # noqa: BLE001
            tprint_error(f"❌ {self.step_name} failed: {exc}")
            logger.exception("Mean reversion step failed")
            return {"success": False, "error": str(exc)}

    # ---------------- Teacher -----------------
    def _build_teacher_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        close = df["close"].astype(float)
        log_price = np.log(close.replace(0.0, np.nan)).ffill()
        returns = log_price.diff().fillna(0.0)

        hurst_window = int(config.get("mr_hurst_window", 200))
        ou_window = int(config.get("mr_ou_window", 200))
        vr_window = int(config.get("mr_variance_ratio_window", 200))
        vr_h = int(config.get("mr_variance_ratio_horizon", 5))

        hurst = self._rolling_hurst(log_price.values, hurst_window)
        ou_half_life, ou_theta = self._rolling_ou_params(log_price.values, ou_window)

        # Simple rolling variance ratio VR(k) using log returns
        vr = np.full(len(returns), np.nan)
        if vr_window > vr_h + 5:
            for i in range(vr_window, len(returns)):
                win = returns.iloc[i - vr_window : i]
                if win.isna().all():
                    continue
                var1 = float(win.var(ddof=1))
                if not np.isfinite(var1) or var1 <= 0:
                    continue
                win_k = win.rolling(vr_h).sum().dropna()
                if len(win_k) < 10:
                    continue
                var_k = float(win_k.var(ddof=1))
                if not np.isfinite(var_k) or var_k <= 0:
                    continue
                vr[i] = var_k / (vr_h * var1)

        adf_p = np.full(len(close), np.nan)
        if STATIONARITY_AVAILABLE:
            adf_w = int(config.get("mr_adf_window", 200))
            for i in range(adf_w, len(returns)):
                seg = returns.iloc[i - adf_w : i]
                try:
                    adf_p[i] = float(adfuller(seg.values, maxlag=0, autolag=None)[1])
                except Exception:
                    adf_p[i] = np.nan
        teacher_df = pd.DataFrame(
            {
                "mr_hurst": hurst,
                "mr_ou_half_life": ou_half_life,
                "mr_ou_theta": ou_theta,
                "mr_variance_ratio": vr,
                "mr_adf_pvalue": adf_p,
            },
            index=df.index,
        )
        return teacher_df

    @staticmethod
    def _rolling_hurst(series: np.ndarray, window: int) -> np.ndarray:
        h = np.full(len(series), np.nan)
        for i in range(window, len(series)):
            x = series[i - window : i]
            x = x[~np.isnan(x)]
            if len(x) < 10:
                continue
            r = np.diff(x)
            if len(r) < 5:
                continue
            n = len(r)
            mean_r = r.mean()
            dev = r - mean_r
            cum = np.cumsum(dev)
            R = cum.max() - cum.min()
            S = r.std()
            if S <= 0 or R <= 0:
                h[i] = 0.5
            else:
                h[i] = max(0.0, min(1.0, np.log(R / S) / np.log(n)))
        return h

    @staticmethod
    def _rolling_ou_params(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        half = np.full(len(series), np.nan)
        theta = np.full(len(series), np.nan)
        for i in range(window, len(series)):
            x = series[i - window : i]
            x = x[~np.isnan(x)]
            if len(x) < 10:
                continue
            x0, x1 = x[:-1], x[1:]
            x0c = x0 - x0.mean()
            x1c = x1 - x1.mean()
            denom = np.dot(x0c, x0c)
            if denom <= 0:
                continue
            phi = float(np.dot(x0c, x1c) / denom)
            if phi <= 0 or phi >= 1:
                continue
            hl = -np.log(2.0) / np.log(phi)
            half[i] = hl
            theta[i] = 1.0 / max(hl, 1e-6)
        return half, theta

    def _train_teacher_gmm(
        self, teacher_df: pd.DataFrame, config: Dict[str, Any]
    ) -> Tuple[GaussianMixture, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
        feat_cols = [
            "mr_hurst",
            "mr_ou_half_life",
            "mr_ou_theta",
            "mr_variance_ratio",
            "mr_adf_pvalue",
        ]
        df = teacher_df[feat_cols].copy()

        # Require only core OU/Hurst features for GMM and teacher validity; VR/ADF
        # may have more NaNs and are used as soft gating signals rather than
        # hard validity requirements.
        core_gmm_cols = ["mr_hurst", "mr_ou_half_life", "mr_ou_theta"]
        mask = teacher_df[core_gmm_cols].notna().all(axis=1)

        min_teacher = int(config.get("mr_min_teacher_samples", 100))
        if mask.sum() < min_teacher:
            raise ValueError("Not enough valid teacher samples")

        X = winsorized_zscore_normalize(teacher_df.loc[mask, core_gmm_cols]).values.astype(float)
        n_comp = int(config.get("mr_teacher_n_components", 3))
        gmm = GaussianMixture(n_components=n_comp, covariance_type="full", random_state=42)
        gmm.fit(X)
        clusters_clean = gmm.predict(X)
        clusters = pd.Series(-1, index=teacher_df.index, dtype=int)
        clusters.loc[mask.index[mask]] = clusters_clean

        # Identify mean-reversion cluster: high theta, low hurst
        stats = (
            teacher_df.loc[mask, ["mr_hurst", "mr_ou_theta"]]
            .groupby(clusters_clean)
            .mean()
        )
        if stats.empty:
            raise ValueError("GMM stats empty")
        # Normalize then score
        h_norm = (stats["mr_hurst"] - stats["mr_hurst"].mean()) / (stats["mr_hurst"].std() + 1e-8)
        th_norm = (stats["mr_ou_theta"] - stats["mr_ou_theta"].mean()) / (stats["mr_ou_theta"].std() + 1e-8)
        score = -h_norm + th_norm
        mr_cluster = int(score.idxmax())

        # Tightened teacher definition with explicit thresholds
        h_thr = float(config.get("mr_hurst_threshold", 0.4))
        hl_thr = float(config.get("mr_half_life_threshold", 5.0))
        adf_thr = float(config.get("mr_adf_p_threshold", 0.1))
        vr_thr = float(config.get("mr_vr_threshold", 0.9))

        h_arr = teacher_df.loc[mask, "mr_hurst"].astype(float).values
        hl_arr = teacher_df.loc[mask, "mr_ou_half_life"].astype(float).values
        vr_arr = teacher_df.loc[mask, "mr_variance_ratio"].astype(float).values
        adf_arr = teacher_df.loc[mask, "mr_adf_pvalue"].astype(float).values

        h_finite = np.isfinite(h_arr)
        hl_finite = np.isfinite(hl_arr)
        vr_finite = np.isfinite(vr_arr)
        adf_finite = np.isfinite(adf_arr)

        cond_h = np.zeros_like(h_arr, dtype=bool)
        cond_h[h_finite] = h_arr[h_finite] < h_thr
        cond_hl = np.zeros_like(hl_arr, dtype=bool)
        cond_hl[hl_finite] = hl_arr[hl_finite] < hl_thr
        cond_vr = np.zeros_like(vr_arr, dtype=bool)
        if vr_finite.any():
            cond_vr[vr_finite] = vr_arr[vr_finite] < vr_thr
        else:
            # If variance ratio is entirely unavailable, do not gate on it.
            cond_vr[:] = True
        cond_adf = np.zeros_like(adf_arr, dtype=bool)
        if adf_finite.any():
            cond_adf[adf_finite] = adf_arr[adf_finite] < adf_thr
        else:
            # If ADF p-values are entirely unavailable, do not gate on them.
            cond_adf[:] = True

        cond_cluster = clusters_clean == mr_cluster
        cond_all = cond_cluster & cond_h & cond_hl & cond_vr & cond_adf

        binary = pd.Series(0, index=teacher_df.index, dtype=int)
        binary.loc[mask.index[mask]] = cond_all.astype(int)

        # Continuous teacher reversion score in [0, 1]
        h_score = np.zeros_like(h_arr, dtype=float)
        h_score[h_finite] = np.clip((h_thr - h_arr[h_finite]) / max(h_thr, 1e-6), 0.0, 1.0)
        hl_score = np.zeros_like(hl_arr, dtype=float)
        hl_score[hl_finite] = np.clip((hl_thr - hl_arr[hl_finite]) / max(hl_thr, 1e-6), 0.0, 1.0)
        vr_score = np.zeros_like(vr_arr, dtype=float)
        vr_score[vr_finite] = np.clip((vr_thr - vr_arr[vr_finite]) / max(vr_thr, 1e-6), 0.0, 1.0)
        adf_score = np.zeros_like(adf_arr, dtype=float)
        adf_score[adf_finite] = np.clip((adf_thr - adf_arr[adf_finite]) / max(adf_thr, 1e-6), 0.0, 1.0)

        comp_stack = np.vstack([h_score, hl_score, vr_score, adf_score])
        base_score = np.nanmean(comp_stack, axis=0)
        # Gate by MR cluster membership
        base_score = base_score * cond_cluster.astype(float)

        teacher_score = pd.Series(0.0, index=teacher_df.index, dtype=float)
        teacher_score.loc[mask.index[mask]] = base_score

        metrics: Dict[str, Any] = {
            "n_components": n_comp,
            "mean_reversion_cluster": mr_cluster,
            "cluster_counts": clusters.value_counts().to_dict(),
            "cluster_stats": stats.to_dict(),
            "thresholds": {
                "hurst": h_thr,
                "half_life": hl_thr,
                "adf_p": adf_thr,
                "variance_ratio": vr_thr,
            },
        }
        return gmm, clusters, binary, teacher_score, metrics

    # ---------------- Student -----------------
    def _build_student_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        vol = df["volume"].astype(float)

        ma_fast = int(config.get("mr_ma_fast_window", 20))
        ma_slow = int(config.get("mr_ma_slow_window", 50))
        vwap_w = int(config.get("mr_vwap_window", 30))

        ma_f = close.rolling(ma_fast, min_periods=ma_fast // 2).mean()
        ma_s = close.rolling(ma_slow, min_periods=ma_slow // 2).mean()
        vwap = (close * vol).rolling(vwap_w, min_periods=vwap_w // 2).sum() / (
            vol.rolling(vwap_w, min_periods=vwap_w // 2).sum() + 1e-8
        )

        dist_ma = (close - ma_s) / (ma_s.replace(0.0, np.nan))
        dist_vwap = (close - vwap) / (vwap.replace(0.0, np.nan))

        # RSI
        rsi_w = int(config.get("mr_rsi_window", 14))
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rsi_w).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_w).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # Bollinger width
        bb_w = int(config.get("mr_bb_window", ma_slow))
        mid = close.rolling(bb_w, min_periods=bb_w // 2).mean()
        std = close.rolling(bb_w, min_periods=bb_w // 2).std()
        bb_width = (2.0 * std) / (mid.replace(0.0, np.nan))

        # Simple volatility + volume state
        ret = close.pct_change().fillna(0.0)
        vol_std = ret.rolling(20, min_periods=10).std()
        vol_atr = (high - low).rolling(20, min_periods=10).mean() / close.replace(0.0, np.nan)

        vol_ma = vol.rolling(30, min_periods=10).mean()
        vol_std_ = vol.rolling(30, min_periods=10).std()
        vol_cv = vol_std_ / (vol_ma + 1e-8)
        vol_rel = vol / (vol_ma + 1e-8)
        log_vol = np.log1p(vol)

        feats = pd.DataFrame(
            {
                "z_price_ma_slow": dist_ma,
                "z_price_vwap": dist_vwap,
                "rsi": rsi,
                "bb_width": bb_width,
                "ret_std_20": vol_std,
                "atr_rel_20": vol_atr,
                "log_volume": log_vol,
                "volume_rel_ma": vol_rel,
                "volume_cv_30": vol_cv,
            },
            index=df.index,
        )
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats = feats.dropna()

        # Optional: augment with balanced feature extractor (most relevant categories)
        if bool(config.get("mr_enable_balanced_features", True)):
            try:
                bf_config = BalancedFeatureConfig()
                bf_config.enabled_categories = [
                    BFCategory.PRICE,
                    BFCategory.VOLUME,
                    BFCategory.VOLATILITY,
                    BFCategory.MOMENTUM,
                    BFCategory.TREND,
                    BFCategory.REGIME,
                ]
                bf_config.enable_temporal_features = False
                bf_config.enable_micro_regime_features = False
                bf_config.enable_feature_selection = True
                bf_config.total_max_features = int(
                    config.get("mr_balanced_total_max_features", 64)
                )
                bf_config.max_features_per_category = int(
                    config.get("mr_balanced_max_features_per_category", 12)
                )

                extractor = BalancedFeatureExtractor(bf_config)
                bf_result = extractor.extract_balanced_features(
                    df[["open", "high", "low", "close", "volume"]]
                )
                if bf_result.success and bf_result.features.size > 0:
                    bf_df = pd.DataFrame(
                        bf_result.features,
                        index=df.index,
                        columns=[f"bf_{name}" for name in bf_result.feature_names],
                    )
                    # Align with existing feature index after dropna
                    bf_df = bf_df.loc[feats.index]
                    feats = pd.concat([feats, bf_df], axis=1)
            except Exception as e:  # noqa: BLE001
                tprint_warning(f"Balanced feature extraction failed: {e}")

        # Normalise most features with winsorized z-score (keep core ones raw)
        exclude = {"z_price_ma_slow", "z_price_vwap", "rsi", "bb_width"}
        norm_cols = [c for c in feats.columns if c not in exclude]
        if norm_cols:
            feats[norm_cols] = winsorized_zscore_normalize(feats[norm_cols])
        return feats

    def _train_xgb_student(
        self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any], y_binary: Optional[pd.Series] = None
    ) -> Tuple[xgb.XGBRegressor, Dict[str, Any], np.ndarray, np.ndarray, Dict[str, Any]]:
        X_np = X.astype(np.float32).values
        y_np = y.astype(np.float32).values
        n = len(X_np)
        train_frac = float(config.get("mr_train_fraction", 0.6))
        val_frac = float(config.get("mr_val_fraction", 0.2))
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_train = max(100, min(n_train, n - 200))
        n_val = max(50, min(n_val, n - n_train - 50))
        n_test = n - n_train - n_val
        if n_test < 50:
            n_test = 50
            n_train = n - n_val - n_test
        idx_train = slice(0, n_train)
        idx_val = slice(n_train, n_train + n_val)
        idx_test = slice(n_train + n_val, n)

        params = dict(
            tree_method="hist",
            learning_rate=float(config.get("mr_learning_rate", 0.01)),
            max_depth=int(config.get("mr_max_depth", 3)),
            min_child_weight=float(config.get("mr_min_child_weight", 20.0)),
            subsample=float(config.get("mr_subsample", 0.6)),
            colsample_bytree=float(config.get("mr_colsample_bytree", 0.5)),
            gamma=float(config.get("mr_gamma", 0.2)),
            reg_alpha=float(config.get("mr_reg_alpha", 2.0)),
            reg_lambda=float(config.get("mr_reg_lambda", 1.0)),
            n_estimators=int(config.get("mr_n_estimators", 800)),
        )
        model = xgb.XGBRegressor(objective=mean_reversion_objective, **params)
        model.fit(X_np[idx_train], y_np[idx_train], eval_set=[(X_np[idx_val], y_np[idx_val])], verbose=False)

        raw = model.predict(X_np)
        raw_train, raw_val, raw_test = raw[idx_train], raw[idx_val], raw[idx_test]
        y_train, y_val, y_test = y_np[idx_train], y_np[idx_val], y_np[idx_test]

        def _metrics(a, b) -> Dict[str, float]:
            return {
                "r2": float(r2_score(a, b)) if len(a) > 1 else float("nan"),
                "rmse": float(np.sqrt(mean_squared_error(a, b))) if len(a) > 1 else float("nan"),
            }

        metrics: Dict[str, Any] = {
            "train": _metrics(y_train, raw_train),
            "val": _metrics(y_val, raw_val),
            "test": _metrics(y_test, raw_test),
        }

        # Optional walk-forward validation for stability across folds
        try:
            wf_metrics = self._run_walkforward_validation(
                X_np,
                y_np,
                y_binary.values.astype(int) if y_binary is not None else None,
                config,
            )
            if wf_metrics:
                metrics["walkforward"] = wf_metrics
        except Exception:
            # Walk-forward is diagnostic only; ignore failures
            pass

        # Basic classification view using tightened teacher binary labels (if provided)
        # We treat *low* predicted distance-to-mean as the positive MR signal and
        # adaptively choose a threshold that maximises F1 on the training set,
        # unless an explicit mr_classification_threshold is provided.
        if y_binary is not None:
            try:
                y_bin_np = y_binary.astype(int).values
                y_train_bin = y_bin_np[idx_train]
                y_val_bin = y_bin_np[idx_val]
                y_test_bin = y_bin_np[idx_test]

                raw_train_np = np.asarray(raw_train, dtype=float)
                raw_val_np = np.asarray(raw_val, dtype=float)
                raw_test_np = np.asarray(raw_test, dtype=float)

                thr_cfg = config.get("mr_classification_threshold", "auto")
                thr_auto = isinstance(thr_cfg, str) and str(thr_cfg).lower() in {"auto", "adaptive"}

                if thr_auto:
                    # Search over quantiles of the training predictions and
                    # pick the threshold that maximises F1 when using
                    #   pred = 1{raw <= thr}
                    pos_mask = y_train_bin == 1
                    if int(pos_mask.sum()) >= 20:
                        finite_mask = np.isfinite(raw_train_np)
                        base_scores = raw_train_np[finite_mask]
                        base_labels = y_train_bin[finite_mask]
                        if base_scores.size > 0:
                            qs = np.linspace(0.01, 0.99, 25)
                            cand = np.unique(np.quantile(base_scores, qs))
                            best_f1 = -1.0
                            best_thr = None
                            for t in cand:
                                preds = (base_scores <= t).astype(int)
                                f1 = f1_score(base_labels, preds, zero_division=0.0)
                                if f1 > best_f1:
                                    best_f1 = f1
                                    best_thr = float(t)
                            thr = float(best_thr) if best_thr is not None else float(np.median(base_scores))
                        else:
                            thr = float(np.nan)
                    else:
                        # Too few positives to tune threshold; fall back to
                        # median of training predictions.
                        thr = float(np.median(raw_train_np))
                else:
                    thr = float(thr_cfg)

                metrics["classification_threshold"] = float(thr)

                pred_train_bin = (raw_train_np <= thr).astype(int)
                pred_val_bin = (raw_val_np <= thr).astype(int)
                pred_test_bin = (raw_test_np <= thr).astype(int)

                metrics["train"]["acc"] = float(
                    accuracy_score(y_train_bin, pred_train_bin)
                )
                metrics["train"]["f1"] = float(
                    f1_score(y_train_bin, pred_train_bin, zero_division="warn")
                )
                metrics["val"]["acc"] = float(
                    accuracy_score(y_val_bin, pred_val_bin)
                )
                metrics["val"]["f1"] = float(
                    f1_score(y_val_bin, pred_val_bin, zero_division="warn")
                )
                metrics["test"]["acc"] = float(
                    accuracy_score(y_test_bin, pred_test_bin)
                )
                metrics["test"]["f1"] = float(
                    f1_score(y_test_bin, pred_test_bin, zero_division="warn")
                )
            except Exception:
                # Keep regression metrics only if classification view fails
                pass

        # Anchored z-score calibration with noise floor
        mu_long = float(np.mean(raw_val)) if len(raw_val) > 0 else float(np.mean(raw))
        sigma_long = float(np.std(raw_val)) if len(raw_val) > 0 else float(np.std(raw))
        if not np.isfinite(sigma_long) or sigma_long <= 0:
            sigma_long = 1.0
        min_std = float(config.get("mr_min_std", 0.6 * sigma_long))
        sigma_eff = max(sigma_long, min_std)
        z = (raw - mu_long) / sigma_eff
        calibrated = norm.cdf(z)

        calib_params = {
            "mu_long": mu_long,
            "sigma_long": sigma_long,
            "min_std": min_std,
        }
        return model, metrics, raw, calibrated, calib_params

    # -------------- Diagnostics / persistence --------------
    def _run_walkforward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_binary: Optional[np.ndarray],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rolling walk-forward validation with time-ordered folds.

        Uses expanding-window training and forward test segments to assess
        stability over time. Trains lightweight XGB models per fold.
        """

        n = int(X.shape[0])
        if n < 600:
            return {}

        n_folds = int(config.get("mr_walkforward_folds", 5))
        if n_folds < 2:
            return {}

        min_train = int(config.get("mr_walkforward_min_train_size", max(200, n // 4)))
        if min_train >= n - 100:
            return {}

        step = (n - min_train) // n_folds
        if step < 50:
            return {}

        r2_list: List[float] = []
        rmse_list: List[float] = []
        acc_list: List[float] = []
        f1_list: List[float] = []

        thr = float(config.get("mr_classification_threshold", 0.5))

        # Use slightly reduced estimators for WF models to control cost
        base_estimators = int(config.get("mr_n_estimators", 800))
        wf_estimators = max(200, min(base_estimators, 600))

        for fold in range(n_folds):
            train_end = min_train + fold * step
            test_start = train_end
            test_end = min(train_end + step, n)
            if test_end - test_start < 50:
                continue

            X_tr, y_tr = X[:train_end], y[:train_end]
            X_te, y_te = X[test_start:test_end], y[test_start:test_end]

            params = dict(
                tree_method="hist",
                learning_rate=float(config.get("mr_learning_rate", 0.01)),
                max_depth=int(config.get("mr_max_depth", 3)),
                min_child_weight=float(config.get("mr_min_child_weight", 20.0)),
                subsample=float(config.get("mr_subsample", 0.6)),
                colsample_bytree=float(config.get("mr_colsample_bytree", 0.5)),
                gamma=float(config.get("mr_gamma", 0.2)),
                reg_alpha=float(config.get("mr_reg_alpha", 2.0)),
                reg_lambda=float(config.get("mr_reg_lambda", 1.0)),
                n_estimators=wf_estimators,
            )
            try:
                model = xgb.XGBRegressor(objective=mean_reversion_objective, **params)
                model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
                y_pred = model.predict(X_te)
            except Exception:
                continue

            try:
                r2_val = float(r2_score(y_te, y_pred))
                rmse_val = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                r2_list.append(r2_val)
                rmse_list.append(rmse_val)
            except Exception:
                pass

            if y_binary is not None:
                try:
                    yb_te = y_binary[test_start:test_end]
                    yb_pred = (y_pred >= thr).astype(int)
                    acc_val = float(accuracy_score(yb_te, yb_pred))
                    f1_val = float(f1_score(yb_te, yb_pred, zero_division="warn"))
                    acc_list.append(acc_val)
                    f1_list.append(f1_val)
                except Exception:
                    pass

        if not r2_list:
            return {}

        wf_result: Dict[str, Any] = {
            "folds": len(r2_list),
            "r2_mean": float(np.mean(r2_list)),
            "r2_std": float(np.std(r2_list)),
            "rmse_mean": float(np.mean(rmse_list)) if rmse_list else float("nan"),
            "rmse_std": float(np.std(rmse_list)) if rmse_list else float("nan"),
            "r2_per_fold": r2_list,
            "rmse_per_fold": rmse_list,
        }

        if acc_list and f1_list:
            wf_result.update(
                {
                    "acc_mean": float(np.mean(acc_list)),
                    "acc_std": float(np.std(acc_list)),
                    "f1_mean": float(np.mean(f1_list)),
                    "f1_std": float(np.std(f1_list)),
                    "acc_per_fold": acc_list,
                    "f1_per_fold": f1_list,
                }
            )

        return wf_result

    @staticmethod
    def _compute_forward_metrics(df: pd.DataFrame, prob_col: str, horizon: int) -> Dict[str, Any]:
        if "close" not in df.columns or prob_col not in df.columns:
            return {}
        close = df["close"].astype(float).values
        fwd = np.full(len(close), np.nan)
        for i in range(len(close) - horizon):
            if close[i] > 0 and close[i + horizon] > 0:
                fwd[i] = np.log(close[i + horizon] / close[i])
        probs = df[prob_col].values
        mask = np.isfinite(fwd) & np.isfinite(probs)
        if mask.sum() < 50:
            return {}
        corr = float(np.corrcoef(probs[mask], fwd[mask])[0, 1])
        return {
            "horizon": horizon,
            "n_samples": int(mask.sum()),
            "mean_fwd_return": float(np.mean(fwd[mask])),
            "std_fwd_return": float(np.std(fwd[mask])),
            "corr_prob_fwd": corr,
        }

    def _save_artifacts_and_reports(
        self,
        output_df: pd.DataFrame,
        X_all: pd.DataFrame,
        y_binary: pd.Series,
        model: xgb.XGBRegressor,
        teacher_metrics: Dict[str, Any],
        student_metrics: Dict[str, Any],
        calib_params: Dict[str, Any],
        fwd_metrics: Dict[Any, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        market_source: str,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        artifacts: Dict[str, str] = {}
        reports: Dict[str, str] = {}

        to_save = output_df[[
            "mr_teacher_cluster",
            "mr_teacher_mean_reversion",
            "mr_teacher_score",
            "mr_raw_score",
            "mr_probability",
        ]].copy()
        to_save = to_save.reset_index().rename(columns={output_df.index.name or "index": "timestamp"})
        try:
            artifacts["training_data"] = self._save_artifact(
                data=to_save,
                artifact_name=f"ml_mean_reversion_training_data_{timeframe}",
                artifact_type="data",
                metadata={"symbol": symbol, "exchange": exchange, "timeframe": timeframe, "source_market_data": market_source},
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save training data artifact: {exc}")

        try:
            artifacts["model"] = self._save_artifact(
                data=model,
                artifact_name=f"ml_mean_reversion_model_{timeframe}",
                artifact_type="model",
                metadata={"symbol": symbol, "exchange": exchange, "timeframe": timeframe, "model_type": "xgboost_regressor"},
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save mean-reversion model artifact: {exc}")

        try:
            artifacts["calibration"] = self._save_artifact(
                data={"calibration": calib_params, "student_metrics": student_metrics},
                artifact_name=f"ml_mean_reversion_calibration_{timeframe}",
                artifact_type="metadata",
                metadata={"symbol": symbol, "exchange": exchange, "timeframe": timeframe},
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save calibration artifact: {exc}")

        # Lightweight Markdown and CSV reports in outcomes/
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            md_path = f"outcomes/ml_mean_reversion_summary_{symbol}_{timeframe}_{ts}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# ML Mean-Reversion Regime Summary for {symbol} ({timeframe})\n\n")
                f.write("## Teacher (OU/Hurst GMM)\n\n")
                f.write(f"- Components: {teacher_metrics.get('n_components')}\n")
                f.write(f"- Mean-reversion cluster: {teacher_metrics.get('mean_reversion_cluster')}\n")
                f.write(f"- Cluster counts: {teacher_metrics.get('cluster_counts')}\n\n")
                f.write("## Student (XGB Regressor)\n\n")
                for split, m in student_metrics.items():
                    # Skip non-split entries such as the global
                    # 'classification_threshold', and the walk-forward block
                    # which is documented separately.
                    if split == "walkforward" or not isinstance(m, dict):
                        continue
                    f.write(
                        f"- {split}: R2={m.get('r2'):.4f}, RMSE={m.get('rmse'):.6f}"
                    )
                    acc = m.get("acc")
                    f1 = m.get("f1")
                    if acc is not None and f1 is not None:
                        f.write(f", ACC={acc:.4f}, F1={f1:.4f}")
                    f.write("\n")
                f.write("\n## Calibration\n\n")
                f.write(f"- mu_long={calib_params.get('mu_long'):.6f}, sigma_long={calib_params.get('sigma_long'):.6f}, min_std={calib_params.get('min_std'):.6f}\n\n")

                # Walk-forward stability metrics if available
                wf = student_metrics.get("walkforward")
                if isinstance(wf, dict) and wf.get("folds", 0) > 0:
                    f.write("## Walk-Forward Stability\n\n")
                    f.write(f"- folds={wf.get('folds')}\n")
                    f.write(
                        f"- R2 mean={wf.get('r2_mean'):.4f}, std={wf.get('r2_std'):.4f}\n"
                    )
                    f.write(
                        f"- RMSE mean={wf.get('rmse_mean'):.6f}, std={wf.get('rmse_std'):.6f}\n"
                    )
                    if wf.get("acc_mean") is not None and wf.get("f1_mean") is not None:
                        f.write(
                            f"- ACC mean={wf.get('acc_mean'):.4f}, std={wf.get('acc_std'):.4f}\n"
                        )
                        f.write(
                            f"- F1 mean={wf.get('f1_mean'):.4f}, std={wf.get('f1_std'):.4f}\n"
                        )
                    f.write("\n")

                # Feature WCoV (weighted by teacher mean-reversion labels)
                try:
                    f.write(
                        "## Feature WCoV (weighted by teacher mean-reversion labels)\n\n"
                    )
                    w_cov: Dict[str, float] = {}
                    y_vals = y_binary.loc[X_all.index].astype(float).values
                    for col in X_all.columns:
                        vals = X_all[col].astype(float).values
                        mask = (
                            np.isfinite(vals)
                            & np.isfinite(y_vals)
                            & (y_vals > 0)
                        )
                        if mask.sum() < 20 or float(y_vals[mask].sum()) <= 0.0:
                            continue
                        w = y_vals[mask]
                        v = vals[mask]
                        w_norm = w / (w.sum() + 1e-8)
                        mean_w = float(np.sum(w_norm * v))
                        std_w = float(np.sqrt(np.sum(w_norm * (v - mean_w) ** 2)))
                        if abs(mean_w) > 1e-8:
                            w_cov[col] = std_w / abs(mean_w)
                    for col, val in sorted(
                        w_cov.items(), key=lambda kv: kv[1], reverse=True
                    )[:15]:
                        f.write(f"- {col}: WCoV={val:.4f}\n")
                    f.write("\n")
                except Exception:
                    pass

                if fwd_metrics:
                    f.write("## Forward-Return Diagnostics\n\n")
                    for h, m in sorted(fwd_metrics.items()):
                        f.write(f"### Horizon {h} bars\n\n")
                        f.write(f"- n_samples={m.get('n_samples')}\n")
                        f.write(
                            f"- mean_fwd_return={m.get('mean_fwd_return'):.6f}\n"
                        )
                        f.write(
                            f"- std_fwd_return={m.get('std_fwd_return'):.6f}\n"
                        )
                        f.write(
                            f"- corr_prob_fwd={m.get('corr_prob_fwd'):.4f}\n\n"
                        )
            reports["markdown"] = md_path
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to write Markdown report: {exc}")

        try:
            csv_df = pd.DataFrame(
                {
                    "timestamp": X_all.index,
                    "mr_teacher_score": output_df.loc[X_all.index, "mr_teacher_score"],
                    "mr_probability": output_df.loc[X_all.index, "mr_probability"],
                }
            )
            csv_path = f"outcomes/ml_mean_reversion_probabilities_{symbol}_{timeframe}_{ts}.csv"
            csv_df.to_csv(csv_path, index=False)
            reports["probabilities_csv"] = csv_path
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to write probabilities CSV: {exc}")

        # Grid backtest conditioned on mr_probability deciles. We use the
        # student's MR classification (low predicted distance-to-mean →
        # positive MR signal) as the directional prediction input, and keep
        # the teacher binary label as a regime column for diagnostics.
        #
        # If available, align TP/SL to the meta-labeling triple-barrier
        # configuration (realized_return) by loading the latest
        # meta_labeling_hpo_best_params file for this symbol/timeframe and
        # deriving (profit_thr_base, stop_to_profit_ratio).
        try:
            idx = X_all.index
            close = output_df.loc[idx, "close"].astype(float)
            high = output_df.loc[idx, "high"].astype(float)
            low = output_df.loc[idx, "low"].astype(float)
            raw_returns = close.pct_change().fillna(0.0)

            prob = output_df.loc[idx, "mr_probability"].astype(float)

            # Classification threshold discovered during training (adaptive by
            # default). If unavailable, fall back to the median raw score.
            thr_cls = student_metrics.get("classification_threshold")
            raw_scores_idx = output_df.loc[idx, "mr_raw_score"].astype(float)
            if thr_cls is None or not np.isfinite(thr_cls):
                thr_cls = float(np.median(raw_scores_idx.values))

            teacher_mr = y_binary.loc[idx].astype(int)
            z_ma = output_df.loc[idx, "z_price_ma_slow"].astype(float)
            z_vwap = output_df.loc[idx, "z_price_vwap"].astype(float)
            teacher_score_series = output_df.loc[idx, "mr_teacher_score"].astype(float)
            mr_signal = raw_scores_idx <= float(thr_cls)
            score_thr = 0.3
            teacher_mask = teacher_score_series >= score_thr
            below_mean = (z_ma < 0.0) | (z_vwap < 0.0)
            preds = (mr_signal & teacher_mask & below_mean).astype(float)

            ml_df_grid = pd.DataFrame(
                {
                    "mr_teacher_mean_reversion": teacher_mr,
                    "mr_teacher_score": output_df.loc[idx, "mr_teacher_score"].astype(float),
                    "mr_probability": prob,
                },
                index=idx,
            )

            # Attempt to load meta-labeling HPO parameters to derive a
            # triple-barrier-like TP/SL pair. If anything fails, fall back
            # to the default TP/SL grid inside run_simple_long_grid_backtest.
            tp_override = None
            sl_override = None
            try:
                from pathlib import Path
                import json

                base_dir = Path("outcomes")
                hpo_pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json"
                hpo_candidates = sorted(base_dir.glob(hpo_pattern))
                if not hpo_candidates:
                    # Fallback: use the latest HPO file for this symbol across
                    # any timeframe.
                    fallback_pattern = f"meta_labeling_hpo_best_params_{symbol}_*_*.json"
                    hpo_candidates = sorted(base_dir.glob(fallback_pattern))
                if hpo_candidates:
                    hpo_path = hpo_candidates[-1]
                    with open(hpo_path, "r", encoding="utf-8") as f_hpo:
                        hpo_cfg = json.load(f_hpo)
                    params = {}
                    if isinstance(hpo_cfg, dict):
                        knee = hpo_cfg.get("knee_params")
                        best = hpo_cfg.get("best_params")
                        if isinstance(knee, dict) and knee:
                            params = knee
                        elif isinstance(best, dict) and best:
                            params = best
                    profit_thr = float(params.get("profit_thr_base")) if params.get("profit_thr_base") is not None else None
                    stop_ratio = float(params.get("stop_to_profit_ratio")) if params.get("stop_to_profit_ratio") is not None else None
                    if profit_thr is not None and stop_ratio is not None:
                        tp_override = max(0.0005, profit_thr)
                        sl_override = max(0.0005, profit_thr * stop_ratio)
            except Exception:
                tp_override = None
                sl_override = None

            if tp_override is not None and sl_override is not None:
                grid_df = run_simple_long_grid_backtest(
                    close=close,
                    high=high,
                    low=low,
                    raw_returns=raw_returns,
                    predictions=preds,
                    confidence=prob,
                    ml_df=ml_df_grid,
                    timeframe=timeframe,
                    regime_col="mr_teacher_mean_reversion",
                    tp_values=[tp_override],
                    sl_values=[sl_override],
                )
            else:
                grid_df = run_simple_long_grid_backtest(
                    close=close,
                    high=high,
                    low=low,
                    raw_returns=raw_returns,
                    predictions=preds,
                    confidence=prob,
                    ml_df=ml_df_grid,
                    timeframe=timeframe,
                    regime_col="mr_teacher_mean_reversion",
                )

            if isinstance(grid_df, pd.DataFrame):
                grid_path = f"outcomes/ml_mean_reversion_grid_backtest_{symbol}_{timeframe}_{ts}.csv"
                tprint_info(
                    f"Writing grid backtest CSV with shape={grid_df.shape} to {grid_path}"
                )
                grid_df.to_csv(grid_path, index=False)
                reports["grid_backtest_csv"] = grid_path
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to run/write grid backtest: {exc}")

        return artifacts, reports
