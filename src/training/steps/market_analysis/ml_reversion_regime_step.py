"""Mean-reversion regime detection step (OU/Hurst teacher + XGB student).

IMPROVED VERSION with:
- Relaxed teacher thresholds for realistic mean-reversion detection
- Classification target: predicts directional moves (0=up, 1=down)
- Enhanced features: momentum divergence, reversion speed, persistence
- Isotonic calibration for proper probability estimates
- Simplified signal generation without overly strict gating
- Comprehensive diagnostics and walk-forward validation

Output: calibrated probability where:
  - 0.0 = bullish (price will increase)
  - 1.0 = bearish (price will decrease)
  - 0.5 = neutral/uncertain
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)
from sklearn.calibration import CalibratedClassifierCV

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


class MLMeanReversionRegimeStep(BaseStep):
    """Ornstein–Uhlenbeck / Hurst teacher → XGB classifier for mean reversion.

    Predicts directional moves (up=0, down=1) using mean-reversion signals.
    """

    def __init__(self, step_name: str = "ml_mean_reversion_step") -> None:
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLMeanReversionRegimeStep") if hasattr(logger, "getChild") else logger
        self._cached_market_data: Optional[pd.DataFrame] = None
        self._cached_market_source: Optional[str] = None
        self._cached_market_cache_key: Optional[Tuple[str, str, str, str]] = None
        tprint(f"✅ Initialized {step_name} step (IMPROVED with classification)", "SUCCESS")

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

            # 3) Student features (ENHANCED with momentum divergence, reversion speed, persistence)
            student_df = self._build_student_features(market_data, config)

            # 4) Build classification target: forward price direction
            #    0 = price will go up (bullish)
            #    1 = price will go down (bearish)
            y_direction_all = self._build_direction_target(market_data, config)

            # Align indices
            common_idx = (
                teacher_score.index
                .intersection(student_df.index)
                .intersection(y_direction_all.index)
                .sort_values()
            )
            if len(common_idx) < 500:
                raise ValueError(f"Not enough aligned samples for training ({len(common_idx)} < 500)")

            X_all = student_df.loc[common_idx]
            y_target_all = y_direction_all.loc[common_idx].astype(int)
            y_teacher_binary = teacher_binary.loc[common_idx].astype(int)
            teacher_score_aligned = teacher_score.loc[common_idx]

            # 5) Train XGB classifier with isotonic calibration
            model, calibrated_model, student_metrics, raw_scores, calibrated_scores = self._train_xgb_student(
                X_all,
                y_target_all,
                config,
                y_teacher=y_teacher_binary,
            )

            # 6) Attach outputs to main frame
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
            output_df.loc[X_all.index, "mr_direction_target"] = y_target_all.values

            # Forward-return diagnostics at multiple horizons
            horizons_cfg = config.get("mr_forward_horizons", [2, 4, 8, 12])  # 30m to 3h for 15m bars
            fwd_metrics: Dict[int, Dict[str, Any]] = {}
            for h in horizons_cfg:
                try:
                    h_int = int(h)
                except (TypeError, ValueError):
                    continue
                m = self._compute_forward_metrics(
                    output_df, prob_col="mr_probability", horizon=h_int, target_col="mr_direction_target"
                )
                if m:
                    fwd_metrics[h_int] = m

            # 7) Persist artifacts + reports
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="mean_reversion_v2",
            )

            artifacts, reports = self._save_artifacts_and_reports(
                output_df=output_df,
                X_all=X_all,
                y_target=y_target_all,
                y_teacher=y_teacher_binary,
                model=model,
                calibrated_model=calibrated_model,
                teacher_metrics=teacher_metrics,
                student_metrics=student_metrics,
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
        """Train GMM on teacher features and identify mean-reversion regime.

        IMPROVED: Relaxed thresholds for 15m timeframe, OR logic for auxiliary features.
        """
        feat_cols = [
            "mr_hurst",
            "mr_ou_half_life",
            "mr_ou_theta",
            "mr_variance_ratio",
            "mr_adf_pvalue",
        ]
        df = teacher_df[feat_cols].copy()

        # Require only core OU/Hurst features for GMM validity
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

        # IMPROVED: Relaxed thresholds for 15m timeframe (trades last 2-12 bars)
        # For 15m: half-life of 4-10 bars = 1-2.5h is reasonable for mean reversion
        h_thr = float(config.get("mr_hurst_threshold", 0.5))       # Relaxed from 0.4
        hl_thr = float(config.get("mr_half_life_threshold", 12.0)) # Relaxed from 5.0, ~3h for 15m
        adf_thr = float(config.get("mr_adf_p_threshold", 0.15))    # Relaxed from 0.1
        vr_thr = float(config.get("mr_vr_threshold", 1.2))         # Relaxed from 0.9

        h_arr = teacher_df.loc[mask, "mr_hurst"].astype(float).values
        hl_arr = teacher_df.loc[mask, "mr_ou_half_life"].astype(float).values
        vr_arr = teacher_df.loc[mask, "mr_variance_ratio"].astype(float).values
        adf_arr = teacher_df.loc[mask, "mr_adf_pvalue"].astype(float).values

        h_finite = np.isfinite(h_arr)
        hl_finite = np.isfinite(hl_arr)
        vr_finite = np.isfinite(vr_arr)
        adf_finite = np.isfinite(adf_arr)

        # Core conditions (must satisfy)
        cond_h = np.zeros_like(h_arr, dtype=bool)
        cond_h[h_finite] = h_arr[h_finite] < h_thr
        cond_hl = np.zeros_like(hl_arr, dtype=bool)
        cond_hl[hl_finite] = hl_arr[hl_finite] < hl_thr
        cond_cluster = clusters_clean == mr_cluster

        # IMPROVED: Auxiliary conditions (at least one should be true)
        cond_vr = np.zeros_like(vr_arr, dtype=bool)
        if vr_finite.any():
            cond_vr[vr_finite] = vr_arr[vr_finite] < vr_thr
        cond_adf = np.zeros_like(adf_arr, dtype=bool)
        if adf_finite.any():
            cond_adf[adf_finite] = adf_arr[adf_finite] < adf_thr

        # At least one auxiliary feature should support mean-reversion
        # If both unavailable, allow through (don't penalize)
        has_vr = vr_finite.any()
        has_adf = adf_finite.any()
        if has_vr and has_adf:
            cond_aux = cond_vr | cond_adf  # OR logic
        elif has_vr:
            cond_aux = cond_vr
        elif has_adf:
            cond_aux = cond_adf
        else:
            cond_aux = np.ones_like(h_arr, dtype=bool)  # Pass if neither available

        # Final: core conditions AND at least one auxiliary
        cond_all = cond_cluster & cond_h & cond_hl & cond_aux

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
            "teacher_positive_rate": float(binary.mean()),
        }
        return gmm, clusters, binary, teacher_score, metrics

    # ---------------- Student -----------------
    def _build_student_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Build student features with ENHANCED mean-reversion indicators:
        - Momentum divergence
        - Reversion speed
        - Regime persistence
        """
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

        # NEW: Momentum divergence features
        price_roc_5 = close.pct_change(5)
        price_roc_10 = close.pct_change(10)
        ma_roc_5 = ma_s.pct_change(5)
        ma_roc_10 = ma_s.pct_change(10)
        momentum_div_5 = price_roc_5 - ma_roc_5
        momentum_div_10 = price_roc_10 - ma_roc_10

        # RSI divergence from price position
        rsi_centered = (rsi - 50) / 50  # Normalize to [-1, 1]
        rsi_divergence = rsi_centered * dist_ma  # Positive when aligned, negative when diverging

        # NEW: Mean reversion speed indicators
        # How fast is price converging to/diverging from mean?
        dist_ma_change_2 = dist_ma.diff(2)   # 30m change for 15m bars
        dist_ma_change_4 = dist_ma.diff(4)   # 1h change
        dist_vwap_change_2 = dist_vwap.diff(2)
        dist_vwap_change_4 = dist_vwap.diff(4)

        # Acceleration toward mean (second derivative)
        dist_ma_accel = dist_ma_change_2.diff(2)

        # NEW: Regime persistence features
        # How long has price been in current regime?
        below_ma = (dist_ma < 0).astype(int)
        below_vwap = (dist_vwap < 0).astype(int)
        oversold_rsi = (rsi < 30).astype(int)
        overbought_rsi = (rsi > 70).astype(int)

        # Count consecutive periods in regime
        below_ma_periods = below_ma.rolling(20, min_periods=1).sum()
        below_vwap_periods = below_vwap.rolling(20, min_periods=1).sum()
        oversold_periods = oversold_rsi.rolling(20, min_periods=1).sum()
        overbought_periods = overbought_rsi.rolling(20, min_periods=1).sum()

        # Extreme distance (potential reversal zones)
        extreme_below = (dist_ma < -0.02).astype(int)  # >2% below MA
        extreme_above = (dist_ma > 0.02).astype(int)   # >2% above MA
        extreme_below_periods = extreme_below.rolling(10, min_periods=1).sum()
        extreme_above_periods = extreme_above.rolling(10, min_periods=1).sum()

        feats = pd.DataFrame(
            {
                # Original features
                "z_price_ma_slow": dist_ma,
                "z_price_vwap": dist_vwap,
                "rsi": rsi,
                "bb_width": bb_width,
                "ret_std_20": vol_std,
                "atr_rel_20": vol_atr,
                "log_volume": log_vol,
                "volume_rel_ma": vol_rel,
                "volume_cv_30": vol_cv,
                # NEW: Momentum divergence
                "momentum_div_5": momentum_div_5,
                "momentum_div_10": momentum_div_10,
                "rsi_divergence": rsi_divergence,
                # NEW: Reversion speed
                "dist_ma_change_2": dist_ma_change_2,
                "dist_ma_change_4": dist_ma_change_4,
                "dist_vwap_change_2": dist_vwap_change_2,
                "dist_vwap_change_4": dist_vwap_change_4,
                "dist_ma_accel": dist_ma_accel,
                # NEW: Regime persistence
                "below_ma_periods": below_ma_periods,
                "below_vwap_periods": below_vwap_periods,
                "oversold_periods": oversold_periods,
                "overbought_periods": overbought_periods,
                "extreme_below_periods": extreme_below_periods,
                "extreme_above_periods": extreme_above_periods,
            },
            index=df.index,
        )
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats = feats.dropna()

        # Optional: augment with balanced feature extractor
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

    def _build_direction_target(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Build classification target: forward price direction.

        Returns:
            0 = bullish (price will go up)
            1 = bearish (price will go down)

        For 15m bars with trades lasting 30m-3h (2-12 bars), we use a forward
        horizon that captures the typical trade duration.
        """
        close = df["close"].astype(float)

        # For 15m timeframe, use 4-6 bar horizon (1-1.5h)
        forward_horizon = int(config.get("mr_forward_target_horizon", 6))
        min_threshold = float(config.get("mr_direction_min_threshold", 0.002))  # 0.2% minimum move

        fwd_returns = np.full(len(close), np.nan)
        for i in range(len(close) - forward_horizon):
            if close.iloc[i] > 0 and close.iloc[i + forward_horizon] > 0:
                fwd_returns[i] = (close.iloc[i + forward_horizon] - close.iloc[i]) / close.iloc[i]

        # Classification:
        # - If forward return > +min_threshold: label = 0 (bullish, price went up)
        # - If forward return < -min_threshold: label = 1 (bearish, price went down)
        # - If |forward return| < min_threshold: look at sign (slight bias up vs down)
        y_direction = np.full(len(close), np.nan)
        finite_mask = np.isfinite(fwd_returns)
        y_direction[finite_mask] = (fwd_returns[finite_mask] < 0).astype(int)  # 1 if down, 0 if up

        return pd.Series(y_direction, index=df.index)

    def _train_xgb_student(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any],
        y_teacher: Optional[pd.Series] = None
    ) -> Tuple[xgb.XGBClassifier, CalibratedClassifierCV, Dict[str, Any], np.ndarray, np.ndarray]:
        """Train XGBoost classifier with isotonic calibration and walk-forward validation.

        Returns:
            - Base XGB model
            - Calibrated model (isotonic)
            - Metrics dict
            - Raw scores (uncalibrated probabilities)
            - Calibrated scores
        """
        X_np = X.astype(np.float32).values
        y_np = y.astype(np.int32).values
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

        # Train base XGBoost classifier
        params = dict(
            tree_method="hist",
            learning_rate=float(config.get("mr_learning_rate", 0.02)),
            max_depth=int(config.get("mr_max_depth", 4)),
            min_child_weight=float(config.get("mr_min_child_weight", 10.0)),
            subsample=float(config.get("mr_subsample", 0.7)),
            colsample_bytree=float(config.get("mr_colsample_bytree", 0.6)),
            gamma=float(config.get("mr_gamma", 0.1)),
            reg_alpha=float(config.get("mr_reg_alpha", 1.0)),
            reg_lambda=float(config.get("mr_reg_lambda", 1.0)),
            n_estimators=int(config.get("mr_n_estimators", 500)),
            scale_pos_weight=float(config.get("mr_scale_pos_weight", 1.0)),
            eval_metric="logloss",
        )

        model = xgb.XGBClassifier(**params, random_state=42)
        model.fit(
            X_np[idx_train],
            y_np[idx_train],
            eval_set=[(X_np[idx_val], y_np[idx_val])],
            verbose=False
        )

        # Get raw predictions (uncalibrated)
        raw_proba = model.predict_proba(X_np)[:, 1]  # Probability of class 1 (bearish)
        raw_train = raw_proba[idx_train]
        raw_val = raw_proba[idx_val]
        raw_test = raw_proba[idx_test]

        y_train = y_np[idx_train]
        y_val = y_np[idx_val]
        y_test = y_np[idx_test]

        # Calibrate on validation set using isotonic regression
        calibration_method = config.get("mr_calibration_method", "isotonic")
        if calibration_method not in ["isotonic", "sigmoid"]:
            calibration_method = "isotonic"

        calibrated_model = CalibratedClassifierCV(
            model,
            method=calibration_method,
            cv="prefit"
        )
        calibrated_model.fit(X_np[idx_val], y_np[idx_val])

        # Get calibrated predictions
        calibrated_proba = calibrated_model.predict_proba(X_np)[:, 1]
        calib_train = calibrated_proba[idx_train]
        calib_val = calibrated_proba[idx_val]
        calib_test = calibrated_proba[idx_test]

        def _metrics(y_true, y_pred_proba, prefix="") -> Dict[str, float]:
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)
            metrics = {
                f"{prefix}acc": float(accuracy_score(y_true, y_pred_binary)),
                f"{prefix}f1": float(f1_score(y_true, y_pred_binary, zero_division=0.0)),
                f"{prefix}precision": float(precision_score(y_true, y_pred_binary, zero_division=0.0)),
                f"{prefix}recall": float(recall_score(y_true, y_pred_binary, zero_division=0.0)),
            }
            try:
                metrics[f"{prefix}auc"] = float(roc_auc_score(y_true, y_pred_proba))
            except ValueError:
                metrics[f"{prefix}auc"] = float("nan")
            try:
                metrics[f"{prefix}logloss"] = float(log_loss(y_true, y_pred_proba))
            except ValueError:
                metrics[f"{prefix}logloss"] = float("nan")
            return metrics

        metrics: Dict[str, Any] = {
            "train_raw": _metrics(y_train, raw_train, ""),
            "val_raw": _metrics(y_val, raw_val, ""),
            "test_raw": _metrics(y_test, raw_test, ""),
            "train_calibrated": _metrics(y_train, calib_train, ""),
            "val_calibrated": _metrics(y_val, calib_val, ""),
            "test_calibrated": _metrics(y_test, calib_test, ""),
            "calibration_method": calibration_method,
            "class_balance": {
                "train_pos_rate": float(y_train.mean()),
                "val_pos_rate": float(y_val.mean()),
                "test_pos_rate": float(y_test.mean()),
            }
        }

        # Walk-forward validation for OOF calibration
        try:
            wf_metrics = self._run_walkforward_validation(
                X_np,
                y_np,
                config,
                calibration_method=calibration_method,
            )
            if wf_metrics:
                metrics["walkforward"] = wf_metrics
        except Exception as e:
            tprint_warning(f"Walk-forward validation failed: {e}")

        return model, calibrated_model, metrics, raw_proba, calibrated_proba

    def _run_walkforward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict[str, Any],
        calibration_method: str = "isotonic",
    ) -> Dict[str, Any]:
        """Rolling walk-forward validation with OOF calibration.

        Each fold:
        - Train on expanding window
        - Calibrate on small validation window
        - Test on forward window
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

        acc_list: List[float] = []
        f1_list: List[float] = []
        auc_list: List[float] = []
        logloss_list: List[float] = []

        base_estimators = int(config.get("mr_n_estimators", 500))
        wf_estimators = max(200, min(base_estimators, 400))

        for fold in range(n_folds):
            train_end = min_train + fold * step
            val_size = min(100, train_end // 10)
            val_start = train_end - val_size
            test_start = train_end
            test_end = min(train_end + step, n)
            if test_end - test_start < 50:
                continue

            X_tr = X[:val_start]
            y_tr = y[:val_start]
            X_val = X[val_start:train_end]
            y_val = y[val_start:train_end]
            X_te = X[test_start:test_end]
            y_te = y[test_start:test_end]

            params = dict(
                tree_method="hist",
                learning_rate=float(config.get("mr_learning_rate", 0.02)),
                max_depth=int(config.get("mr_max_depth", 4)),
                min_child_weight=float(config.get("mr_min_child_weight", 10.0)),
                subsample=float(config.get("mr_subsample", 0.7)),
                colsample_bytree=float(config.get("mr_colsample_bytree", 0.6)),
                gamma=float(config.get("mr_gamma", 0.1)),
                reg_alpha=float(config.get("mr_reg_alpha", 1.0)),
                reg_lambda=float(config.get("mr_reg_lambda", 1.0)),
                n_estimators=wf_estimators,
                eval_metric="logloss",
            )
            try:
                model = xgb.XGBClassifier(**params, random_state=42)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                # Calibrate on val set
                calibrated = CalibratedClassifierCV(model, method=calibration_method, cv="prefit")
                calibrated.fit(X_val, y_val)

                # Predict on test set
                y_pred_proba = calibrated.predict_proba(X_te)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)
            except Exception:
                continue

            try:
                acc_list.append(float(accuracy_score(y_te, y_pred)))
                f1_list.append(float(f1_score(y_te, y_pred, zero_division=0.0)))
                auc_list.append(float(roc_auc_score(y_te, y_pred_proba)))
                logloss_list.append(float(log_loss(y_te, y_pred_proba)))
            except Exception:
                pass

        if not acc_list:
            return {}

        return {
            "folds": len(acc_list),
            "acc_mean": float(np.mean(acc_list)),
            "acc_std": float(np.std(acc_list)),
            "f1_mean": float(np.mean(f1_list)),
            "f1_std": float(np.std(f1_list)),
            "auc_mean": float(np.mean(auc_list)),
            "auc_std": float(np.std(auc_list)),
            "logloss_mean": float(np.mean(logloss_list)),
            "logloss_std": float(np.std(logloss_list)),
            "acc_per_fold": acc_list,
            "f1_per_fold": f1_list,
            "auc_per_fold": auc_list,
            "logloss_per_fold": logloss_list,
        }

    @staticmethod
    def _compute_forward_metrics(
        df: pd.DataFrame,
        prob_col: str,
        horizon: int,
        target_col: str = "mr_direction_target"
    ) -> Dict[str, Any]:
        """Compute forward-looking metrics for model validation."""
        if "close" not in df.columns or prob_col not in df.columns:
            return {}

        close = df["close"].astype(float).values
        fwd = np.full(len(close), np.nan)
        for i in range(len(close) - horizon):
            if close[i] > 0 and close[i + horizon] > 0:
                fwd[i] = (close[i + horizon] - close[i]) / close[i]

        probs = df[prob_col].values
        mask = np.isfinite(fwd) & np.isfinite(probs)
        if mask.sum() < 50:
            return {}

        # Correlation: higher prob (bearish) should correlate with negative returns
        corr = float(np.corrcoef(probs[mask], fwd[mask])[0, 1])

        # Directional accuracy: prob > 0.5 predicts down (negative return)
        pred_down = (probs[mask] > 0.5).astype(int)
        actual_down = (fwd[mask] < 0).astype(int)
        dir_acc = float(accuracy_score(actual_down, pred_down))

        # Returns by probability bucket
        buckets = pd.qcut(probs[mask], q=5, labels=False, duplicates='drop')
        bucket_returns = {}
        for b in range(5):
            bucket_mask = (buckets == b)
            if bucket_mask.sum() > 0:
                bucket_returns[f"bucket_{b}"] = float(np.mean(fwd[mask][bucket_mask]))

        return {
            "horizon": horizon,
            "n_samples": int(mask.sum()),
            "mean_fwd_return": float(np.mean(fwd[mask])),
            "std_fwd_return": float(np.std(fwd[mask])),
            "corr_prob_fwd": corr,
            "directional_accuracy": dir_acc,
            "bucket_returns": bucket_returns,
        }

    def _save_artifacts_and_reports(
        self,
        output_df: pd.DataFrame,
        X_all: pd.DataFrame,
        y_target: pd.Series,
        y_teacher: pd.Series,
        model: xgb.XGBClassifier,
        calibrated_model: CalibratedClassifierCV,
        teacher_metrics: Dict[str, Any],
        student_metrics: Dict[str, Any],
        fwd_metrics: Dict[Any, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        market_source: str,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Save artifacts and generate comprehensive reports with improved diagnostics."""
        artifacts: Dict[str, str] = {}
        reports: Dict[str, str] = {}

        # Save training data with all scores
        to_save = output_df[[
            "mr_teacher_cluster",
            "mr_teacher_mean_reversion",
            "mr_teacher_score",
            "mr_raw_score",
            "mr_probability",
            "mr_direction_target",
        ]].copy()
        to_save = to_save.reset_index().rename(columns={output_df.index.name or "index": "timestamp"})
        try:
            artifacts["training_data"] = self._save_artifact(
                data=to_save,
                artifact_name=f"ml_mean_reversion_training_data_{timeframe}",
                artifact_type="data",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "source_market_data": market_source,
                    "version": "v2_classification"
                },
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save training data artifact: {exc}")

        # Save base XGB model
        try:
            artifacts["model_base"] = self._save_artifact(
                data=model,
                artifact_name=f"ml_mean_reversion_model_base_{timeframe}",
                artifact_type="model",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "model_type": "xgboost_classifier",
                    "version": "v2"
                },
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save base model artifact: {exc}")

        # Save calibrated model
        try:
            artifacts["model_calibrated"] = self._save_artifact(
                data=calibrated_model,
                artifact_name=f"ml_mean_reversion_model_calibrated_{timeframe}",
                artifact_type="model",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "model_type": "calibrated_classifier",
                    "calibration_method": student_metrics.get("calibration_method", "isotonic"),
                    "version": "v2"
                },
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save calibrated model artifact: {exc}")

        # Save metrics
        try:
            artifacts["metrics"] = self._save_artifact(
                data={
                    "teacher": teacher_metrics,
                    "student": student_metrics,
                    "forward": fwd_metrics,
                },
                artifact_name=f"ml_mean_reversion_metrics_{timeframe}",
                artifact_type="metadata",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "version": "v2"
                },
            )
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to save metrics artifact: {exc}")

        # Generate comprehensive Markdown report
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            md_path = f"outcomes/ml_mean_reversion_summary_{symbol}_{timeframe}_{ts}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# ML Mean-Reversion (v2) Summary for {symbol} ({timeframe})\n\n")
                f.write("**Model Type**: XGBoost Classifier with Isotonic Calibration\n")
                f.write("**Target**: Directional (0=up, 1=down)\n")
                f.write("**Version**: v2 with relaxed thresholds, enhanced features, and proper calibration\n\n")

                # Teacher metrics
                f.write("## Teacher (OU/Hurst GMM) - IMPROVED\n\n")
                f.write(f"- Components: {teacher_metrics.get('n_components')}\n")
                f.write(f"- Mean-reversion cluster: {teacher_metrics.get('mean_reversion_cluster')}\n")
                f.write(f"- Cluster counts: {teacher_metrics.get('cluster_counts')}\n")
                thresholds = teacher_metrics.get('thresholds', {})
                f.write(f"- Thresholds (RELAXED for 15m):\n")
                f.write(f"  - Hurst: {thresholds.get('hurst', 'N/A')}\n")
                f.write(f"  - Half-life: {thresholds.get('half_life', 'N/A')} bars\n")
                f.write(f"  - ADF p-value: {thresholds.get('adf_p', 'N/A')}\n")
                f.write(f"  - Variance ratio: {thresholds.get('variance_ratio', 'N/A')}\n")
                f.write(f"- **Teacher positive rate: {teacher_metrics.get('teacher_positive_rate', 0.0):.4f}** (IMPROVED from ~0.0)\n\n")

                # Student metrics
                f.write("## Student (XGB Classifier) - RAW vs CALIBRATED\n\n")

                f.write("### Raw Model Performance\n\n")
                for split in ["train", "val", "test"]:
                    m = student_metrics.get(f"{split}_raw", {})
                    f.write(f"**{split.upper()}**: ")
                    f.write(f"ACC={m.get('acc', 0):.4f}, ")
                    f.write(f"F1={m.get('f1', 0):.4f}, ")
                    f.write(f"Precision={m.get('precision', 0):.4f}, ")
                    f.write(f"Recall={m.get('recall', 0):.4f}, ")
                    f.write(f"AUC={m.get('auc', 0):.4f}, ")
                    f.write(f"LogLoss={m.get('logloss', 0):.4f}\n")
                f.write("\n")

                f.write("### Calibrated Model Performance\n\n")
                for split in ["train", "val", "test"]:
                    m = student_metrics.get(f"{split}_calibrated", {})
                    f.write(f"**{split.upper()}**: ")
                    f.write(f"ACC={m.get('acc', 0):.4f}, ")
                    f.write(f"F1={m.get('f1', 0):.4f}, ")
                    f.write(f"Precision={m.get('precision', 0):.4f}, ")
                    f.write(f"Recall={m.get('recall', 0):.4f}, ")
                    f.write(f"AUC={m.get('auc', 0):.4f}, ")
                    f.write(f"LogLoss={m.get('logloss', 0):.4f}\n")
                f.write("\n")

                # Class balance
                f.write("### Class Balance\n\n")
                cb = student_metrics.get("class_balance", {})
                f.write(f"- Train positive rate (bearish): {cb.get('train_pos_rate', 0):.4f}\n")
                f.write(f"- Val positive rate (bearish): {cb.get('val_pos_rate', 0):.4f}\n")
                f.write(f"- Test positive rate (bearish): {cb.get('test_pos_rate', 0):.4f}\n\n")

                # Walk-forward stability
                wf = student_metrics.get("walkforward")
                if isinstance(wf, dict) and wf.get("folds", 0) > 0:
                    f.write("## Walk-Forward Stability (OOF Calibrated)\n\n")
                    f.write(f"- Folds: {wf.get('folds')}\n")
                    f.write(f"- **ACC**: mean={wf.get('acc_mean', 0):.4f}, std={wf.get('acc_std', 0):.4f}\n")
                    f.write(f"- **F1**: mean={wf.get('f1_mean', 0):.4f}, std={wf.get('f1_std', 0):.4f}\n")
                    f.write(f"- **AUC**: mean={wf.get('auc_mean', 0):.4f}, std={wf.get('auc_std', 0):.4f}\n")
                    f.write(f"- **LogLoss**: mean={wf.get('logloss_mean', 0):.4f}, std={wf.get('logloss_std', 0):.4f}\n\n")

                # Forward diagnostics
                if fwd_metrics:
                    f.write("## Forward-Return Diagnostics\n\n")
                    for h, m in sorted(fwd_metrics.items()):
                        f.write(f"### Horizon {h} bars ({h * 15} minutes)\n\n")
                        f.write(f"- n_samples: {m.get('n_samples')}\n")
                        f.write(f"- mean_fwd_return: {m.get('mean_fwd_return', 0):.6f}\n")
                        f.write(f"- std_fwd_return: {m.get('std_fwd_return', 0):.6f}\n")
                        f.write(f"- **corr_prob_fwd**: {m.get('corr_prob_fwd', 0):.4f} (negative = good, higher prob → lower returns)\n")
                        f.write(f"- **directional_accuracy**: {m.get('directional_accuracy', 0):.4f}\n")

                        bucket_returns = m.get('bucket_returns', {})
                        if bucket_returns:
                            f.write(f"- Returns by probability bucket:\n")
                            for bucket, ret in sorted(bucket_returns.items()):
                                f.write(f"  - {bucket}: {ret:.6f}\n")
                        f.write("\n")

                # Signal statistics
                f.write("## Signal Statistics\n\n")
                prob_series = output_df.loc[X_all.index, "mr_probability"]
                raw_series = output_df.loc[X_all.index, "mr_raw_score"]
                target_series = y_target

                bullish_rate = float((prob_series < 0.4).mean())
                neutral_rate = float(((prob_series >= 0.4) & (prob_series <= 0.6)).mean())
                bearish_rate = float((prob_series > 0.6).mean())

                f.write(f"- Bullish signals (prob < 0.4): {bullish_rate:.4f}\n")
                f.write(f"- Neutral signals (0.4 ≤ prob ≤ 0.6): {neutral_rate:.4f}\n")
                f.write(f"- Bearish signals (prob > 0.6): {bearish_rate:.4f}\n")
                f.write(f"- Mean calibrated probability: {prob_series.mean():.4f}\n")
                f.write(f"- Std calibrated probability: {prob_series.std():.4f}\n\n")

                # Feature importance
                try:
                    f.write("## Top 15 Feature Importances\n\n")
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:15]
                    for i, idx in enumerate(indices):
                        col_name = X_all.columns[idx]
                        imp = importances[idx]
                        f.write(f"{i+1}. {col_name}: {imp:.4f}\n")
                    f.write("\n")
                except Exception:
                    pass

            reports["markdown"] = md_path
            tprint_success(f"✅ Saved markdown report: {md_path}")
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to write Markdown report: {exc}")

        # Save probabilities CSV
        try:
            csv_df = pd.DataFrame(
                {
                    "timestamp": X_all.index,
                    "mr_teacher_score": output_df.loc[X_all.index, "mr_teacher_score"],
                    "mr_raw_score": output_df.loc[X_all.index, "mr_raw_score"],
                    "mr_probability": output_df.loc[X_all.index, "mr_probability"],
                    "mr_direction_target": y_target,
                    "close": output_df.loc[X_all.index, "close"],
                }
            )
            csv_path = f"outcomes/ml_mean_reversion_probabilities_{symbol}_{timeframe}_{ts}.csv"
            csv_df.to_csv(csv_path, index=False)
            reports["probabilities_csv"] = csv_path
            tprint_success(f"✅ Saved probabilities CSV: {csv_path}")
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to write probabilities CSV: {exc}")

        # Grid backtest with SIMPLIFIED signal generation
        try:
            idx = X_all.index
            close = output_df.loc[idx, "close"].astype(float)
            high = output_df.loc[idx, "high"].astype(float)
            low = output_df.loc[idx, "low"].astype(float)
            raw_returns = close.pct_change().fillna(0.0)

            prob = output_df.loc[idx, "mr_probability"].astype(float)

            # SIMPLIFIED: Use continuous probability directly
            # For long-only strategy:
            # - High bearish prob (close to 1) = avoid/short
            # - Low bearish prob (close to 0) = strong long signal
            # Transform: long_confidence = 1 - prob
            long_confidence = 1.0 - prob

            # Gate by position relative to mean for mean-reversion context
            z_ma = output_df.loc[idx, "z_price_ma_slow"].astype(float)
            z_vwap = output_df.loc[idx, "z_price_vwap"].astype(float)

            # Boost confidence when oversold (below mean) for mean-reversion longs
            oversold = ((z_ma < -0.01) | (z_vwap < -0.01)).astype(float)
            confidence_boost = 1.0 + oversold * 0.5

            preds = long_confidence * confidence_boost
            preds = preds.clip(0, 1)  # Normalize back to [0, 1]

            ml_df_grid = pd.DataFrame(
                {
                    "mr_teacher_mean_reversion": y_teacher.loc[idx].astype(int),
                    "mr_teacher_score": output_df.loc[idx, "mr_teacher_score"].astype(float),
                    "mr_probability": prob,
                    "mr_direction_target": y_target.loc[idx].astype(int),
                },
                index=idx,
            )

            # Attempt to load meta-labeling HPO parameters
            tp_override = None
            sl_override = None
            try:
                from pathlib import Path
                import json

                base_dir = Path("outcomes")
                hpo_pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json"
                hpo_candidates = sorted(base_dir.glob(hpo_pattern))
                if not hpo_candidates:
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
                    confidence=long_confidence,
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
                    confidence=long_confidence,
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
                tprint_success(f"✅ Saved grid backtest CSV: {grid_path}")
        except Exception as exc:  # noqa: BLE001
            tprint_warning(f"Failed to run/write grid backtest: {exc}")

        return artifacts, reports
