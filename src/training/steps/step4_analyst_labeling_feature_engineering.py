# src/training/steps/step4_analyst_labeling_feature_engineering.py

import asyncio
import json
import os
import pickle
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
from src.training.steps.unified_data_loader import get_unified_data_loader

# Optional stationarity test
try:
    from statsmodels.tsa.stattools import adfuller as _adfuller
    _ADF_AVAILABLE = True
except Exception:
    _adfuller = None
    _ADF_AVAILABLE = False


class AnalystLabelingFeatureEngineeringStep:
    """Step 4: Analyst Labeling and Feature Engineering using Vectorized Orchestrator."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("AnalystLabelingFeatureEngineeringStep")
        self.orchestrator = None

        # Standardized column groups
        self._RAW_PRICE_COLUMNS = [
            "open",
            "high",
            "low",
            "close",
            "avg_price",
            "min_price",
            "max_price",
        ]
        self._RAW_VOLUME_COLUMNS = [
            "volume",
            "trade_volume",
            "trade_count",
            "volume_ratio",
        ]
        self._RAW_MICROSTRUCTURE_COLUMNS = [
            "market_depth",
            "bid_ask_spread",
        ]
        # Any raw-like context that must never leak into features directly
        self._RAW_CONTEXT_COLUMNS = (
            self._RAW_PRICE_COLUMNS
            + self._RAW_VOLUME_COLUMNS
            + self._RAW_MICROSTRUCTURE_COLUMNS
            + ["funding_rate"]
        )
        # Metadata/non-feature columns
        self._METADATA_COLUMNS = [
            "year",
            "month",
            "day",
            "day_of_week",
            "day_of_month",
            "quarter",
            "exchange",
            "symbol",
            "timeframe",
            "split",
        ]
        self._LABEL_NAME = "label"

    # -----------------
    # Feature validators
    # -----------------
    def validate_feature_output(self, func):
        def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.Series:
            ref_index = df.index
            out = func(df, *args, **kwargs)
            if not isinstance(out, pd.Series):
                raise TypeError(f"Feature function {func.__name__} must return pandas Series, got {type(out)}")
            if not out.index.equals(ref_index):
                raise ValueError(f"Feature {getattr(out,'name',func.__name__)} index misaligned with input index")
            if not isinstance(getattr(out, "name", None), str):
                raise ValueError(f"Feature from {func.__name__} must have a string name")
            return out
        return wrapper

    # -----------------
    # Decorated feature helpers
    # -----------------
    @property
    def _feat_close_returns(self):
        @self.validate_feature_output
        def _impl(df: pd.DataFrame) -> pd.Series:
            s = self._get_series_for_calc(df, "close").pct_change()
            s.name = "close_returns"
            return s
        return _impl

    def _feat_pct_change(self, col: str, name: str):
        @self.validate_feature_output
        def _impl(df: pd.DataFrame) -> pd.Series:
            s = self._get_series_for_calc(df, col).pct_change()
            s.name = name
            return s
        return _impl

    def _feat_diff(self, col: str, name: str):
        @self.validate_feature_output
        def _impl(df: pd.DataFrame) -> pd.Series:
            s = self._get_series_for_calc(df, col).diff()
            s.name = name
            return s
        return _impl

    @property
    def _feat_gk_vol_returns(self):
        @self.validate_feature_output
        def _impl(df: pd.DataFrame) -> pd.Series:
            def _compute_group(g: pd.DataFrame) -> pd.Series:
                open_ = g["open"].astype(float)
                high = g["high"].astype(float)
                low = g["low"].astype(float)
                close = g["close"].astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_hl = np.log(high / low)
                    log_co = np.log(close / open_)
                gk_var = (log_hl ** 2) / (2 * np.log(2)) - (2 * np.log(2) - 1) * (log_co ** 2)
                gk_vol = np.sqrt(gk_var.clip(lower=0))
                with np.errstate(divide="ignore", invalid="ignore"):
                    ret = (gk_vol / gk_vol.shift(1)) - 1.0
                return ret
            if "symbol" in df.columns and df["symbol"].nunique() > 1:
                ret = df.groupby("symbol", group_keys=False).apply(_compute_group)
            else:
                ret = _compute_group(df)
            ret.name = "gk_vol_returns"
            return ret
        return _impl

    def _feat_sma_distance(self, window: int, name: str):
        @self.validate_feature_output
        def _impl(df: pd.DataFrame) -> pd.Series:
            def _compute_sma_distance(g: pd.DataFrame) -> pd.Series:
                close = g["close"].astype(float)
                sma = close.rolling(window, min_periods=1).mean()
                with np.errstate(divide="ignore", invalid="ignore"):
                    s_local = (close / sma) - 1.0
                return s_local
            if "symbol" in df.columns and df["symbol"].nunique() > 1:
                s = df.groupby("symbol", group_keys=False).apply(_compute_sma_distance)
            else:
                s = _compute_sma_distance(df)
            s.name = name
            return s
        return _impl

    def _feat_vwap_distance(self, window: int, name: str):
        @self.validate_feature_output
        def _impl(df: pd.DataFrame) -> pd.Series:
            required = ["close", "volume"]
            if not all(c in df.columns for c in required):
                return pd.Series(np.nan, index=df.index, name=name)
            def _compute_group(g: pd.DataFrame) -> pd.Series:
                c = g["close"].astype(float)
                v = g["volume"].astype(float).clip(lower=0)
                num = (c * v).rolling(window, min_periods=max(2, window // 2)).sum()
                den = v.rolling(window, min_periods=max(2, window // 2)).sum()
                with np.errstate(divide="ignore", invalid="ignore"):
                    vwap = num / den.replace(0, np.nan)
                    dist = (c / vwap) - 1.0
                return dist
            if "symbol" in df.columns and df["symbol"].nunique() > 1:
                s = df.groupby("symbol", group_keys=False).apply(_compute_group)
            else:
                s = _compute_group(df)
            s = s.replace([np.inf, -np.inf], np.nan)
            s.name = name
            return s
        return _impl

    def _feat_choppiness_index(self, period: int, name: str):
        @self.validate_feature_output
        def _impl(df: pd.DataFrame) -> pd.Series:
            required = ["high", "low", "close"]
            if not all(c in df.columns for c in required):
                return pd.Series(np.nan, index=df.index, name=name)
            def _compute_group(g: pd.DataFrame) -> pd.Series:
                high = g["high"].astype(float)
                low = g["low"].astype(float)
                close = g["close"].astype(float)
                prev_close = close.shift(1)
                tr = pd.concat([
                    (high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ], axis=1).max(axis=1)
                atr_sum = tr.rolling(period, min_periods=max(3, period // 2)).sum()
                hh = high.rolling(period, min_periods=max(3, period // 2)).max()
                ll = low.rolling(period, min_periods=max(3, period // 2)).min()
                with np.errstate(divide="ignore", invalid="ignore"):
                    denom = (hh - ll).replace(0, np.nan)
                    chop = 100.0 * (np.log10(atr_sum / denom)) / np.log10(float(period))
                return chop
            if "symbol" in df.columns and df["symbol"].nunique() > 1:
                s = df.groupby("symbol", group_keys=False).apply(_compute_group)
            else:
                s = _compute_group(df)
            s = s.clip(lower=0.0, upper=100.0).replace([np.inf, -np.inf], np.nan)
            s.name = name
            return s
        return _impl

    def _group_aware_rolling_corr(self, s1: pd.Series, s2: pd.Series, df: pd.DataFrame, window: int) -> pd.Series:
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            def _corr_group(g: pd.DataFrame) -> pd.Series:
                return g[s1.name].rolling(window, min_periods=max(5, window // 2)).corr(g[s2.name])
            tmp = pd.concat({"sym": df["symbol"], s1.name: s1, s2.name: s2}, axis=1)
            out = tmp.groupby("sym", group_keys=False).apply(_corr_group)
            return out.reset_index(level=0, drop=True)
        return s1.rolling(window, min_periods=max(5, window // 2)).corr(s2)

    def _feat_engulf_strength_z(self, kind: str):
        @self.validate_feature_output
        def _impl(df: pd.DataFrame) -> pd.Series:
            close = df["close"].astype(float)
            open_ = df["open"].astype(float)
            body = (close - open_).abs()
            body_prev = body.shift(1)
            if kind == "bull":
                is_current_bullish = close > open_
                is_previous_bearish = df["close"].shift(1) < df["open"].shift(1)
                is_engulf = (open_ < df["close"].shift(1)) & (close > df["open"].shift(1))
                cond = is_current_bullish & is_previous_bearish & is_engulf
                name = "bullish_engulf_strength_z"
            else:
                is_current_bearish = close < open_
                is_previous_bullish = df["close"].shift(1) > df["open"].shift(1)
                is_engulf = (open_ > df["close"].shift(1)) & (close < df["open"].shift(1))
                cond = is_current_bearish & is_previous_bullish & is_engulf
                name = "bearish_engulf_strength_z"
            strength = (body / (body_prev.replace(0, np.nan))).where(cond, 0.0)
            # Group-aware rolling z-score to prevent cross-asset leakage
            if "symbol" in df.columns and df["symbol"].nunique() > 1:
                def _z(g: pd.Series) -> pd.Series:
                    r = g.rolling(50, min_periods=5)
                    return ((g - r.mean()) / r.std().replace(0, np.nan))
                z = strength.groupby(df["symbol"]).transform(_z)
            else:
                z = self._zscore(strength, window=50)
            z = z.fillna(0)
            z.name = name
            return z
        return _impl

    def _ensure_continuous_time_index(self, df: pd.DataFrame, freq: str = "1T") -> tuple[pd.DataFrame, int]:
        """Reindex to continuous time grid and return number of inserted rows."""
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to set from 'timestamp' if available
            if "timestamp" in df.columns:
                try:
                    df = df.copy()
                    df.index = pd.to_datetime(df["timestamp"], errors="coerce")
                    df = df.sort_index()
                    df = df.drop(columns=["timestamp"], errors="ignore")
                except Exception:
                    return df, 0
            else:
                return df, 0
        if df.index.size == 0:
            return df, 0
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        before = len(df)
        df = df.reindex(full_index)
        inserted = len(df) - before
        return df, inserted

    def _sanitize_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf with NaN, forward-fill NaNs; log fill counts by column."""
        df = df.replace([np.inf, -np.inf], np.nan)
        # Count NaNs per column before
        nan_before = df.isna().sum()
        df = df.fillna(method="ffill")
        # Log fills
        try:
            filled = nan_before - df.isna().sum()
            filled = filled[filled > 0]
            if not filled.empty:
                self.logger.warning(f"Raw data forward-filled values: {filled.to_dict()}")
        except Exception:
            pass
        return df

    def _sanitize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-merge engineered features sanitization: handle inf/NaNs.
        Strategy: replace inf with NaN, then fill NaN with median per column; fallback to 0 if median is NaN.
        """
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        medians: dict[str, float] = {}
        for c in numeric_cols:
            med = df[c].median()
            if pd.isna(med):
                med = 0.0
            medians[c] = med
        if medians:
            df[numeric_cols] = df[numeric_cols].fillna(value=medians)
        # Non-numeric: forward-fill if any (rare in features)
        non_num = [c for c in df.columns if c not in numeric_cols and c != "label"]
        if non_num:
            df[non_num] = df[non_num].fillna(method="ffill")
        # Log strategy summary
        try:
            total_inf = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
            total_nan = int(df.isna().sum().sum())
            self.logger.info(f"Feature sanitization completed: inf_residual={total_inf}, nan_residual={total_nan}")
        except Exception:
            pass
        return df

    def _zscore(self, series: pd.Series, window: int = 50) -> pd.Series:
        roll = series.rolling(window=window, min_periods=max(5, window // 5))
        z = (series - roll.mean()) / roll.std().replace(0, np.nan)
        return z.replace([np.inf, -np.inf], np.nan)

    async def _build_pipeline_a_stationary(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Pipeline A: create stationary features directly from raw series made stationary."""
        try:
            df = price_data.copy()
            features = pd.DataFrame(index=df.index)

            # Core stationary transforms
            if "close" in df.columns:
                features[self._feat_close_returns(df).name] = self._feat_close_returns(df)

            # Prefer trade_volume if present, else volume (readable control flow)
            if "trade_volume" in df.columns:
                vol_col = "trade_volume"
            elif "volume" in df.columns:
                vol_col = "volume"
            else:
                vol_col = None
            if vol_col is not None:
                vol_ret = self._feat_pct_change(vol_col, "volume_returns")(df)
                features[vol_ret.name] = vol_ret

            # Funding rates dynamics
            if "funding_rate" in df.columns:
                fr_chg = self._feat_diff("funding_rate", "funding_rate_change")(df)
                fr_ret = self._feat_pct_change("funding_rate", "funding_rate_returns")(df)
                features[fr_chg.name] = fr_chg
                features[fr_ret.name] = fr_ret

            # Volatility of stationary series (group-aware)
            cr = self._feat_close_returns(df)
            if cr is not None:
                features["returns_volatility_20"] = self._group_aware_rolling_std(cr, df, 20)
            if "volume_returns" in features.columns:
                vr = features["volume_returns"]
                features["volume_returns_volatility_20"] = self._group_aware_rolling_std(vr, df, 20)

            # Correlation between returns and volume changes (market microstructure/correlation)
            if "close_returns" in features.columns and "volume_returns" in features.columns:
                try:
                    s1 = features["close_returns"].astype(float)
                    s2 = features["volume_returns"].astype(float)
                    features["returns_volume_correlation_20"] = self._group_aware_rolling_corr(s1, s2, df, 20)
                except Exception:
                    pass

            # Simple interactive features
            if cr is not None and "volume_returns" in features.columns:
                features["returns_x_volume_returns"] = (
                    cr * features["volume_returns"]
                )

            # Additional upstream standardization to returns/changes for available raw series
            if "trade_count" in df.columns:
                tc_ret = self._feat_pct_change("trade_count", "trade_count_returns")(df)
                features[tc_ret.name] = tc_ret
            if "trade_volume" in df.columns:
                tv_ret = self._feat_pct_change("trade_volume", "trade_volume_returns")(df)
                features[tv_ret.name] = tv_ret
            if "market_depth" in df.columns:
                md_ret = self._feat_pct_change("market_depth", "market_depth_returns")(df)
                features[md_ret.name] = md_ret
            if "bid_ask_spread" in df.columns:
                bas_ret = self._feat_pct_change("bid_ask_spread", "bid_ask_spread_returns")(df)
                features[bas_ret.name] = bas_ret

            # Cleanup
            num = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            features[num.columns] = num
            # Do not fill with zeros here; allow NaN to surface for later imputation
            features = features.dropna(axis=1, how="all")
            return features
        except Exception as e:
            self.logger.warning(f"Pipeline A failed: {e}")
            return pd.DataFrame(index=price_data.index)

    async def _build_pipeline_b_ohlcv(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Pipeline B: compute raw OHLCV indicators then transform to stationary features."""
        try:
            required = ["open", "high", "low", "close"]
            if not all(c in price_data.columns for c in required):
                self.logger.warning("Pipeline B skipped due to missing OHLCV columns")
                return pd.DataFrame(index=price_data.index)

            df = price_data.copy()
            feats = pd.DataFrame(index=df.index)

            close = df["close"].astype(float)
            open_ = df["open"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            # SMA distances (stationary)
            sma20d = self._feat_sma_distance(20, "sma20_distance")(df)
            sma50d = self._feat_sma_distance(50, "sma50_distance")(df)
            feats[sma20d.name] = sma20d
            feats[sma50d.name] = sma50d

            # Garman–Klass volatility -> returns of vol
            gk = self._feat_gk_vol_returns(df)
            feats[gk.name] = gk

            # Price distance from VWAP (stationary)
            try:
                vwapd20 = self._feat_vwap_distance(20, "vwap_distance_20")(df)
                feats[vwapd20.name] = vwapd20
            except Exception:
                pass

            # Choppiness Index (stationary, bounded [0,100])
            try:
                chop14 = self._feat_choppiness_index(14, "choppiness_index_14")(df)
                feats[chop14.name] = chop14
            except Exception:
                pass

            # Simple candlestick strength (engulfing bullish/bearish) -> z-scored
            body = (close - open_).abs()
            body_prev = body.shift(1)
            # Bullish engulfing
            bullz = self._feat_engulf_strength_z("bull")(df)
            feats[bullz.name] = bullz
            # Bearish engulfing
            bearz = self._feat_engulf_strength_z("bear")(df)
            feats[bearz.name] = bearz

            # SR location features (pivot-based), computed without lookahead
            try:
                sr_feats = self._build_sr_location_features(df)
                for name, series in sr_feats.items():
                    feats[name] = series
                self.logger.info(
                    f"Step4 SR location features added: {list(sr_feats.keys())}"
                )
                # SR event labels for S/R model training (touch-only moments)
                try:
                    labels = self._build_sr_event_labels(pd.concat([df, pd.DataFrame(sr_feats, index=df.index)], axis=1))
                    feats["sr_event_label"] = labels
                    feats["sr_touch"] = (labels != 0).astype(int)
                except Exception as _e:
                    self.logger.warning(f"Failed to create SR event labels: {_e}")
            except Exception as _e:
                self.logger.warning(f"Failed to add SR location features: {_e}")

            # Cleanup
            num = feats.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            feats[num.columns] = num
            # Do not fill with zeros here; allow NaN to surface for later imputation
            feats = feats.dropna(axis=1, how="all")
            return feats
        except Exception as e:
            self.logger.warning(f"Pipeline B failed: {e}")
            return pd.DataFrame(index=price_data.index)

    def _build_sr_location_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Construct SR 'location' features using classic floor pivots and nearest S/R distances.

        - Uses previous bar pivots (shifted) to avoid lookahead.
        - Returns stationary relative distances (distance normalized by close).
        """
        required = ["open", "high", "low", "close"]
        for c in required:
            if c not in df.columns:
                return {}
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        # Classic floor pivots from prior bar
        pivot = ((high.shift(1) + low.shift(1) + close.shift(1)) / 3.0)
        r1 = 2 * pivot - low.shift(1)
        s1 = 2 * pivot - high.shift(1)
        r2 = pivot + (high.shift(1) - low.shift(1))
        s2 = pivot - (high.shift(1) - low.shift(1))

        # Relative distances (stationary)
        def _reldist(level: pd.Series) -> pd.Series:
            with np.errstate(divide='ignore', invalid='ignore'):
                d = (close - level).abs() / close.replace(0, np.nan)
            return d.replace([np.inf, -np.inf], np.nan)

        dist_s1 = _reldist(s1).rename("distance_to_pivot_support_s1")
        dist_s2 = _reldist(s2).rename("distance_to_pivot_support_s2")
        dist_r1 = _reldist(r1).rename("distance_to_pivot_resistance_r1")
        dist_r2 = _reldist(r2).rename("distance_to_pivot_resistance_r2")

        # Nearest support/resistance among available pivot levels
        nearest_support = pd.concat([dist_s1, dist_s2], axis=1).min(axis=1).rename("nearest_support_distance")
        nearest_resistance = pd.concat([dist_r1, dist_r2], axis=1).min(axis=1).rename("nearest_resistance_distance")

        # Binary flags: at/near support or resistance (within 0.2%)
        near_thresh = 0.002
        is_at_support = (nearest_support <= near_thresh).astype(int).rename("is_at_support")
        is_at_resistance = (nearest_resistance <= near_thresh).astype(int).rename("is_at_resistance")

        # Unified S/R zones (Phases 1.2–3): candidates, scoring, clustering
        zones = self._unified_sr_zones(df)
        # Final features (Phase 4): distances/position/breakout/bounce
        final_loc = self._final_sr_location_features(df, zones)

        return {
            dist_s1.name: dist_s1.fillna(method="ffill").fillna(0),
            dist_s2.name: dist_s2.fillna(method="ffill").fillna(0),
            dist_r1.name: dist_r1.fillna(method="ffill").fillna(0),
            dist_r2.name: dist_r2.fillna(method="ffill").fillna(0),
            nearest_support.name: nearest_support.fillna(method="ffill").fillna(0),
            nearest_resistance.name: nearest_resistance.fillna(method="ffill").fillna(0),
            is_at_support.name: is_at_support.fillna(0),
            is_at_resistance.name: is_at_resistance.fillna(0),
            **final_loc,
        }

    def _unified_sr_zones(
        self,
        df: pd.DataFrame,
        lookback_bars: int = 2000,
        profile_bins: int = 50,
        top_hvns: int = 10,
        touch_tol_pct: float = 0.002,
        recency_half_life_bars: int = 1000,
        w_volume: float = 0.5,
        w_recency: float = 0.3,
        w_touches: float = 0.2,
    ) -> dict[str, Any]:
        """Detect pivot/HVN S/R candidates, score them, and cluster via DBSCAN into zones.

        Returns dict with keys: 'levels' (pd.DataFrame of candidates with price, side, score), 'clusters' (list of dicts)
        """
        try:
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                return {"levels": pd.DataFrame(), "clusters": []}
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df.get("volume", pd.Series(1.0, index=df.index)).astype(float)

            # Phase 1.1: Price pivots (use peaks on returns magnitude)
            returns = close.pct_change().fillna(0).abs()
            prominence = returns.rolling(20, min_periods=1).mean()
            peaks, _ = find_peaks(high.values, prominence=prominence.values)
            troughs, _ = find_peaks((-low).values, prominence=prominence.values)

            pivot_levels = []  # (price, idx, side)
            for i in peaks:
                pivot_levels.append((float(high.iloc[i]), i, "resistance"))
            for i in troughs:
                pivot_levels.append((float(low.iloc[i]), i, "support"))

            # Phase 1.2: HVNs using simple volume profile over recent window
            recent_slice = df.tail(min(lookback_bars, len(df)))
            ch = recent_slice["close"].astype(float)
            vol = recent_slice.get("volume", pd.Series(1.0, index=recent_slice.index)).astype(float)
            if len(recent_slice) > 5:
                pr_min, pr_max = float(ch.min()), float(ch.max())
                bins = np.linspace(pr_min, pr_max, num=profile_bins + 1)
                # Assign each bar to bin of its close
                bin_idx = np.digitize(ch.values, bins) - 1
                vol_by_bin = {}
                for bi, v in zip(bin_idx, vol.values):
                    if 0 <= bi < profile_bins:
                        vol_by_bin[bi] = vol_by_bin.get(bi, 0.0) + float(v)
                # Top HVN bin centers
                top_bins = sorted(vol_by_bin.items(), key=lambda x: x[1], reverse=True)[:top_hvns]
                hvm_levels = [((bins[b] + bins[b + 1]) / 2.0, vol_by_bin[b]) for b, _ in top_bins]
            else:
                hvm_levels = []

            # Phase 2: Multi-factor scoring for each candidate level
            # Build dataframe of candidates
            rows = []
            # Average volume for normalization
            avg_vol = float(volume.rolling(100, min_periods=1).mean().iloc[-1] or 1.0)
            now_idx = np.arange(len(df))[-1] if len(df) > 0 else 0
            lam = np.log(2) / max(1, recency_half_life_bars)

            # Pivots
            for price_level, i, side in pivot_levels:
                # Volume confirmation at pivot
                vol_conf = float(volume.iloc[i] / avg_vol) if avg_vol > 0 else 1.0
                # Recency decay
                recency = float(np.exp(-lam * max(0, now_idx - i)))
                # Touches count within tolerance
                tol = touch_tol_pct * float(close.iloc[i])
                touches = int(((close - price_level).abs() <= tol).sum())
                score = w_volume * vol_conf + w_recency * recency + w_touches * (touches / max(1, len(df) / 100))
                rows.append({"price": price_level, "idx": i, "side": side, "score": score})

            # HVNs
            for price_level, bin_vol in hvm_levels:
                vol_conf = float(bin_vol / (avg_vol * 10.0))  # scale: bins accumulate many bars
                recency = 1.0  # HVN computed on recent window
                tol = touch_tol_pct * price_level
                touches = int(((close - price_level).abs() <= tol).sum())
                score = w_volume * vol_conf + w_recency * recency + w_touches * (touches / max(1, len(df) / 100))
                side = "support" if price_level <= float(close.median()) else "resistance"
                rows.append({"price": price_level, "idx": now_idx, "side": side, "score": score})

            levels_df = pd.DataFrame(rows)
            if levels_df.empty:
                return {"levels": levels_df, "clusters": []}

            # Phase 3: DBSCAN clustering in price space
            median_price = float(close.median() or 1.0)
            eps = max(1e-8, 0.005 * median_price)  # 0.5%
            db = DBSCAN(eps=eps, min_samples=1)
            labels = db.fit_predict(levels_df[["price"]])
            levels_df["cluster_id"] = labels

            clusters = []
            for cid, group in levels_df.groupby("cluster_id"):
                cluster_score = float(group["score"].sum())
                # Weighted center
                w = group["score"].values
                p = group["price"].values
                center = float(np.average(p, weights=w)) if w.sum() > 0 else float(p.mean())
                # Determine side by majority
                side = group["side"].mode().iloc[0] if not group["side"].mode().empty else "support"
                # Cluster width as price range
                width = float(np.max(p) - np.min(p)) if p.size else 0.0
                clusters.append({"cluster_id": int(cid), "center": center, "score": cluster_score, "side": side})

            return {"levels": levels_df, "clusters": clusters}
        except Exception as e:
            self.logger.warning(f"Unified SR zones failed: {e}")
            return {"levels": pd.DataFrame(), "clusters": []}

    def _final_sr_location_features(self, df: pd.DataFrame, zones: dict[str, Any], top_n: int = 3) -> dict[str, pd.Series]:
        """Compute final SR features from top-N support/resistance clusters.

        Returns features: dist_to_support_pct, dist_to_resistance_pct, sr_zone_position,
        sr_breakout_up, sr_breakout_down, sr_bounce_up, sr_bounce_down
        """
        try:
            close = df["close"].astype(float)
            clusters = zones.get("clusters", [])
            if not clusters:
                zeros = pd.Series(0, index=df.index)
                return {
                    "dist_to_support_pct": zeros.astype(float),
                    "dist_to_resistance_pct": zeros.astype(float),
                    "sr_zone_position": pd.Series(0.5, index=df.index).astype(float),
                    "sr_breakout_up": zeros,
                    "sr_breakout_down": zeros,
                    "sr_bounce_up": zeros,
                    "sr_bounce_down": zeros,
                    "nearest_support_center": pd.Series(np.nan, index=df.index),
                    "nearest_resistance_center": pd.Series(np.nan, index=df.index),
                    "nearest_support_score": pd.Series(0.0, index=df.index),
                    "nearest_resistance_score": pd.Series(0.0, index=df.index),
                    "nearest_support_band_pct": pd.Series(0.0, index=df.index),
                    "nearest_resistance_band_pct": pd.Series(0.0, index=df.index),
                }

            # Separate and pick top-N by score
            sup = [c for c in clusters if c["side"] == "support"]
            res = [c for c in clusters if c["side"] == "resistance"]
            sup = sorted(sup, key=lambda x: x["score"], reverse=True)[:top_n]
            res = sorted(res, key=lambda x: x["score"], reverse=True)[:top_n]
            sup_levels = np.array([c["center"] for c in sup]) if sup else np.array([])
            res_levels = np.array([c["center"] for c in res]) if res else np.array([])
            # Build maps for score and band width (percentage of price)
            sup_score_map = {c["center"]: c.get("score", 0.0) for c in sup}
            res_score_map = {c["center"]: c.get("score", 0.0) for c in res}
            # If widths were computed, attach; else fallback tiny
            sup_width_map = {c["center"]: abs(c.get("width", 0.0)) for c in sup}
            res_width_map = {c["center"]: abs(c.get("width", 0.0)) for c in res}

            # For each bar, nearest support below and resistance above
            sup_arr = np.full(len(df), np.nan)
            res_arr = np.full(len(df), np.nan)
            sup_score_arr = np.zeros(len(df))
            res_score_arr = np.zeros(len(df))
            sup_band_pct_arr = np.zeros(len(df))
            res_band_pct_arr = np.zeros(len(df))
            for i, cp in enumerate(close.values):
                if sup_levels.size:
                    below = sup_levels[sup_levels <= cp]
                    if below.size:
                        sel = below[np.argmin(cp - below)]
                        sup_arr[i] = sel
                        sup_score_arr[i] = float(sup_score_map.get(sel, 0.0))
                        width = float(sup_width_map.get(sel, 0.0))
                        sup_band_pct_arr[i] = float(width / max(1e-8, cp))
                if res_levels.size:
                    above = res_levels[res_levels >= cp]
                    if above.size:
                        sel = above[np.argmin(above - cp)]
                        res_arr[i] = sel
                        res_score_arr[i] = float(res_score_map.get(sel, 0.0))
                        width = float(res_width_map.get(sel, 0.0))
                        res_band_pct_arr[i] = float(width / max(1e-8, cp))

            sup_series = pd.Series(sup_arr, index=df.index)
            res_series = pd.Series(res_arr, index=df.index)
            sup_score_series = pd.Series(sup_score_arr, index=df.index)
            res_score_series = pd.Series(res_score_arr, index=df.index)
            sup_band_pct_series = pd.Series(sup_band_pct_arr, index=df.index)
            res_band_pct_series = pd.Series(res_band_pct_arr, index=df.index)

            with np.errstate(divide='ignore', invalid='ignore'):
                dist_sup = ((close - sup_series) / close).clip(lower=0).fillna(1.0)
                dist_res = ((res_series - close) / close).clip(lower=0).fillna(1.0)
                pos = ((close - sup_series) / (res_series - sup_series)).clip(0, 1).fillna(0.5)

            # Breakout/bounce detectors (simple):
            # breakout_up: cross above resistance by > 0.1% from below
            # bounce_up: touch support (within 0.1%) then next bar up move
            tol = 0.001
            prev_close = close.shift(1)
            breakout_up = ((prev_close <= res_series * (1 + tol)) & (close > res_series * (1 + tol))).astype(int)
            breakout_down = ((prev_close >= sup_series * (1 - tol)) & (close < sup_series * (1 - tol))).astype(int)
            bounce_up = (((close - sup_series).abs() / close) <= tol) & (close.pct_change().fillna(0) > 0)
            bounce_down = (((res_series - close).abs() / close) <= tol) & (close.pct_change().fillna(0) < 0)

            return {
                "dist_to_support_pct": dist_sup.astype(float),
                "dist_to_resistance_pct": dist_res.astype(float),
                "sr_zone_position": pos.astype(float),
                "sr_breakout_up": breakout_up.astype(int),
                "sr_breakout_down": breakout_down.astype(int),
                "sr_bounce_up": bounce_up.astype(int),
                "sr_bounce_down": bounce_down.astype(int),
                "nearest_support_center": sup_series,
                "nearest_resistance_center": res_series,
                "nearest_support_score": sup_score_series,
                "nearest_resistance_score": res_score_series,
                "nearest_support_band_pct": sup_band_pct_series,
                "nearest_resistance_band_pct": res_band_pct_series,
            }
        except Exception as e:
            self.logger.warning(f"Final SR location features failed: {e}")
            zeros = pd.Series(0, index=df.index)
            return {
                "dist_to_support_pct": zeros.astype(float),
                "dist_to_resistance_pct": zeros.astype(float),
                "sr_zone_position": pd.Series(0.5, index=df.index).astype(float),
                "sr_breakout_up": zeros,
                "sr_breakout_down": zeros,
                "sr_bounce_up": zeros,
                "sr_bounce_down": zeros,
                "nearest_support_center": pd.Series(np.nan, index=df.index),
                "nearest_resistance_center": pd.Series(np.nan, index=df.index),
                "nearest_support_score": pd.Series(0.0, index=df.index),
                "nearest_resistance_score": pd.Series(0.0, index=df.index),
                "nearest_support_band_pct": pd.Series(0.0, index=df.index),
                "nearest_resistance_band_pct": pd.Series(0.0, index=df.index),
            }

    def _build_sr_event_labels(
        self,
        df: pd.DataFrame,
        touch_tol_pct: float = 0.001,
        breakout_thresh: float = 0.003,  # 0.3%
        bounce_thresh: float = 0.005,    # 0.5%
        horizon: int = 20,
        min_consecutive: int = 2,
        dedup_window_bars: int = 5,
        min_zone_score: float = 0.0,
        min_hist_touches: int = 1,
    ) -> pd.Series:
        """Build SR-event labels using nearest centers and OHLC data.
        -1 breakout, +1 bounce, 0 none. Triggered only on touches at t.
        """
        try:
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            sup_center = df.get("nearest_support_center", pd.Series(np.nan, index=df.index)).astype(float)
            res_center = df.get("nearest_resistance_center", pd.Series(np.nan, index=df.index)).astype(float)

            labels = pd.Series(0, index=df.index, dtype=int)
            # Support touches
            sup_band = df.get("nearest_support_band_pct", pd.Series(0.0, index=df.index)).astype(float)
            res_band = df.get("nearest_resistance_band_pct", pd.Series(0.0, index=df.index)).astype(float)
            sup_tol = np.maximum(touch_tol_pct, 0.5 * sup_band).astype(float)
            res_tol = np.maximum(touch_tol_pct, 0.5 * res_band).astype(float)
            # Touch conditions
            touch_sup = (sup_center.notna()) & (((low - sup_center).abs() / close) <= sup_tol)
            touch_res = (res_center.notna()) & (((res_center - high).abs() / close) <= res_tol)
            # Require min consecutive bars in-zone
            sup_roll = touch_sup.rolling(min_consecutive, min_periods=min_consecutive).sum().fillna(0) >= min_consecutive
            res_roll = touch_res.rolling(min_consecutive, min_periods=min_consecutive).sum().fillna(0) >= min_consecutive
            touch_sup = sup_roll
            touch_res = res_roll

            last_labeled = -dedup_window_bars - 1
            ambiguous_count = 0
            total_touches = len(touch_sup)
            for i in touch_sup.index:
                if i - last_labeled < dedup_window_bars:
                    continue
                end = min(len(df) - 1, i + horizon)
                # Determine side
                side = "support" if bool(touch_sup.iloc[i]) else "resistance"
                level = float(sup_center.iloc[i] if side == "support" else res_center.iloc[i])
                # Zone score curation
                score = float(df.get("nearest_support_score", pd.Series(0.0, index=df.index)).iloc[i]) if side == "support" else float(df.get("nearest_resistance_score", pd.Series(0.0, index=df.index)).iloc[i])
                if score < min_zone_score:
                    continue
                # Historical touches count
                start_hist = max(0, i - horizon)
                past = close.iloc[start_hist:i]
                if len(past) > 0:
                    hist_touches = int(((past - level).abs() / np.maximum(1e-8, past)) <= (touch_tol_pct)).sum()
                    if hist_touches < min_hist_touches:
                        continue
                # scan ahead
                window_close = close.iloc[i + 1 : end + 1]
                window_high = high.iloc[i + 1 : end + 1]
                window_low = low.iloc[i + 1 : end + 1]
                if window_close.empty:
                    continue

                breakout = False
                bounce = False
                if side == "support":
                    # Breakout down if closes below level by breakout_thresh
                    breakout = bool(((window_close - level) / level < -breakout_thresh).any())
                    # Bounce up if price moves away by bounce_thresh without breakout first
                    away = bool(((window_high - level) / level > bounce_thresh).any())
                    if away and breakout:
                        ambiguous_count += 1
                    bounce = (away and not breakout)
                else:
                    # Breakout up if closes above level by breakout_thresh
                    breakout = bool(((window_close - level) / level > breakout_thresh).any())
                    # Bounce down if price moves away by bounce_thresh without breakout first
                    away = bool(((level - window_low) / level > bounce_thresh).any())
                    if away and breakout:
                        ambiguous_count += 1
                    bounce = (away and not breakout)

                labels.iloc[i] = -1 if breakout else (1 if bounce else 0)
                if labels.iloc[i] != 0:
                    last_labeled = i

            try:
                self.logger.info(
                    f"SR-event labeling diagnostics: touches={total_touches}, ambiguous={ambiguous_count}"
                )
            except Exception:
                pass
            return labels
        except Exception:
            return pd.Series(0, index=df.index, dtype=int)

    def _compute_vif_scores(self, X: pd.DataFrame) -> dict[str, float]:
        from sklearn.linear_model import LinearRegression
        vif_scores: dict[str, float] = {}
        cols = X.columns.tolist()
        X_imputed = X.copy()
        for c in cols:
            med = X_imputed[c].median()
            if pd.isna(med):
                med = 0.0
            X_imputed[c] = X_imputed[c].fillna(med)
        for col in cols:
            others = [c for c in cols if c != col]
            if not others:
                vif_scores[col] = 1.0
                continue
            reg = LinearRegression()
            try:
                reg.fit(X_imputed[others], X_imputed[col])
                y = X_imputed[col].values
                y_pred = reg.predict(X_imputed[others])
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
                vif = float(np.inf) if r2 >= 0.999999 else (1.0 / max(1e-12, 1.0 - r2))
            except Exception:
                vif = 1.0
            vif_scores[col] = vif
        return vif_scores

    def _build_cpa_transform(self, X: pd.DataFrame, cluster_cols: list[str], name_prefix: str) -> tuple[pd.Series, dict]:
        means = X[cluster_cols].mean()
        stds = X[cluster_cols].std().replace(0, np.nan)
        Z = (X[cluster_cols] - means) / stds
        Z = Z.fillna(0.0)
        try:
            U, S, VT = np.linalg.svd(Z.values, full_matrices=False)
            weights = VT[0, :]
        except Exception:
            k = len(cluster_cols)
            weights = np.ones(k) / max(1, k)
        pc1 = pd.Series(np.dot(Z.values, weights), index=X.index, name=f"cpa_{name_prefix}_pc1").astype(np.float32)
        transform = {
            "name": pc1.name,
            "cols": cluster_cols,
            "means": means.to_dict(),
            "stds": stds.to_dict(),
            "weights": weights.tolist(),
        }
        return pc1, transform

    def _apply_cpa_transforms(self, X: pd.DataFrame, transforms: list[dict]) -> pd.DataFrame:
        X_new = X.copy()
        for t in transforms:
            cols = [c for c in t["cols"] if c in X_new.columns]
            if len(cols) < 1:
                continue
            means = pd.Series(t["means"]).reindex(cols).fillna(0.0)
            stds = pd.Series(t["stds"]).reindex(cols).replace(0, np.nan)
            Z = (X_new[cols] - means) / stds
            Z = Z.fillna(0.0)
            w = np.array(t["weights"])[: len(cols)]
            comp = pd.Series(np.dot(Z.values, w), index=X_new.index, name=t["name"]).astype(np.float32)
            X_new[t["name"]] = comp
            X_new = X_new.drop(columns=cols, errors="ignore")
        return X_new

    def _vif_reduce_train_val_test(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        removed: list[str] = []
        cpa_transforms: list[dict] = []
        cpa_clusters: list[list[str]] = []
        # Separate features and labels
        feat_cols = [c for c in train_df.columns if c != self._LABEL_NAME]
        X_tr = train_df[feat_cols].copy()
        X_vl = val_df[feat_cols].copy() if not val_df.empty else val_df.copy()
        X_te = test_df[feat_cols].copy() if not test_df.empty else test_df.copy()
        # Remove zero-variance columns
        zero_var = [c for c in X_tr.columns if X_tr[c].nunique(dropna=True) <= 1]
        if zero_var:
            removed.extend(zero_var)
            X_tr = X_tr.drop(columns=zero_var, errors="ignore")
            X_vl = X_vl.drop(columns=zero_var, errors="ignore") if not X_vl.empty else X_vl
            X_te = X_te.drop(columns=zero_var, errors="ignore") if not X_te.empty else X_te
        # Main VIF loop
        max_iters = 50
        for _ in range(max_iters):
            if X_tr.shape[1] <= 2:
                break
            vif = self._compute_vif_scores(X_tr)
            if not vif:
                break
            worst_feature, worst_vif = max(vif.items(), key=lambda kv: (float("inf") if np.isinf(kv[1]) else kv[1]))
            if worst_vif is None or (not np.isfinite(worst_vif)):
                removed.append(worst_feature)
                X_tr = X_tr.drop(columns=[worst_feature], errors="ignore")
                X_vl = X_vl.drop(columns=[worst_feature], errors="ignore") if not X_vl.empty else X_vl
                X_te = X_te.drop(columns=[worst_feature], errors="ignore") if not X_te.empty else X_te
                continue
            if worst_vif > 20.0:
                try:
                    corr = X_tr.corr().abs()
                    candidates = corr.columns[(corr[worst_feature] >= 0.7)].tolist()
                    cluster = list(dict.fromkeys([worst_feature] + [c for c in candidates if c != worst_feature]))
                    if len(cluster) >= 2:
                        pc1, transform = self._build_cpa_transform(X_tr, cluster, name_prefix=str(len(cpa_transforms)+1))
                        X_tr[pc1.name] = pc1
                        cpa_transforms.append(transform)
                        cpa_clusters.append(cluster)
                        X_tr = X_tr.drop(columns=cluster, errors="ignore")
                        if not X_vl.empty:
                            X_vl = self._apply_cpa_transforms(X_vl.drop(columns=cluster, errors="ignore"), [transform])
                        if not X_te.empty:
                            X_te = self._apply_cpa_transforms(X_te.drop(columns=cluster, errors="ignore"), [transform])
                        continue
                    else:
                        removed.append(worst_feature)
                        X_tr = X_tr.drop(columns=[worst_feature], errors="ignore")
                        X_vl = X_vl.drop(columns=[worst_feature], errors="ignore") if not X_vl.empty else X_vl
                        X_te = X_te.drop(columns=[worst_feature], errors="ignore") if not X_te.empty else X_te
                        continue
                except Exception:
                    removed.append(worst_feature)
                    X_tr = X_tr.drop(columns=[worst_feature], errors="ignore")
                    X_vl = X_vl.drop(columns=[worst_feature], errors="ignore") if not X_vl.empty else X_vl
                    X_te = X_te.drop(columns=[worst_feature], errors="ignore") if not X_te.empty else X_te
                    continue
            elif worst_vif > 10.0:
                removed.append(worst_feature)
                X_tr = X_tr.drop(columns=[worst_feature], errors="ignore")
                X_vl = X_vl.drop(columns=[worst_feature], errors="ignore") if not X_vl.empty else X_vl
                X_te = X_te.drop(columns=[worst_feature], errors="ignore") if not X_te.empty else X_te
                continue
            else:
                break
        # Reassemble DataFrames with label column
        out_train = pd.concat([X_tr, train_df[[self._LABEL_NAME]]], axis=1)
        out_val = pd.concat([X_vl, val_df[[self._LABEL_NAME]]], axis=1) if not val_df.empty else val_df
        out_test = pd.concat([X_te, test_df[[self._LABEL_NAME]]], axis=1) if not test_df.empty else test_df
        summary = {
            "removed_features": removed,
            "cpa_clusters": cpa_clusters,
            "cpa_count": len(cpa_clusters),
        }
        if removed:
            self.logger.info(f"Step 4 VIF: removed {len(removed)} features (>10): {removed[:50]}{' ...' if len(removed)>50 else ''}")
        if cpa_clusters:
            self.logger.info(f"Step 4 VIF: created {len(cpa_clusters)} CPA components (>20)")
        return out_train, out_val, out_test, summary

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="analyst labeling and feature engineering step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the analyst labeling and feature engineering step."""
        self.logger.info(
            "Initializing Analyst Labeling and Feature Engineering Step...",
        )

        # Initialize the vectorized labeling orchestrator
        from src.training.steps.vectorized_labelling_orchestrator import (
            VectorizedLabellingOrchestrator,
        )

        self.orchestrator = VectorizedLabellingOrchestrator(self.config)
        await self.orchestrator.initialize()

        self.logger.info(
            "Analyst Labeling and Feature Engineering Step initialized successfully",
        )

    async def _validate_and_enhance_features(self, labeled_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and enhance features for the 240+ feature set."""
        try:
            self.logger.info("🔍 Validating and enhancing features for 240+ feature set...")
            
            # Separate features from labels
            feature_columns = [col for col in labeled_data.columns if col != 'label']
            features_df = labeled_data[feature_columns]
            
            # CRITICAL: Enhanced feature validation with detailed reporting
            constant_features = []
            low_variance_features = []
            valid_features = []
            problematic_features = []
            
            self.logger.info(f"📊 Starting feature validation for {len(feature_columns)} features...")
            
            for col in feature_columns:
                try:
                    # Skip non-numeric columns during variance checks (e.g., datetime/timestamp/strings)
                    if not np.issubdtype(features_df[col].dtype, np.number):
                        # Coerce categoricals to string and skip validation for them
                        try:
                            if str(features_df[col].dtype).startswith("category"):
                                features_df[col] = features_df[col].astype(str)
                        except Exception:
                            pass
                        continue
                    feature_values = features_df[col].dropna()
                    
                    if len(feature_values) == 0:
                        constant_features.append(f"{col} (all NaN values)")
                        continue
                    
                    # Check for constant features
                    unique_values = feature_values.unique()
                    if len(unique_values) <= 1:
                        constant_value = unique_values[0] if len(unique_values) == 1 else "NaN"
                        constant_features.append(f"{col} (constant value: {constant_value})")
                        continue
                    
                    # Check for very low variance features
                    variance = feature_values.var()
                    if variance < 1e-8:  # Very low variance threshold
                        low_variance_features.append(f"{col} (variance: {variance:.2e})")
                        continue
                    
                    # Check for binary features with only one value
                    if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                        # Binary feature with both 0 and 1 - this is perfect!
                        valid_features.append(col)
                    elif len(unique_values) == 2:
                        # Binary feature with unexpected values
                        problematic_features.append(f"{col} (binary feature with values: {unique_values})")
                        continue
                    else:
                        # Continuous feature with sufficient variation
                        valid_features.append(col)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error validating feature '{col}': {e}")
                    problematic_features.append(f"{col} (validation error: {e})")
                    continue
            
            # Log comprehensive feature quality metrics
            self.logger.info(f"📊 Feature quality analysis:")
            self.logger.info(f"   📊 Total features: {len(feature_columns)}")
            self.logger.info(f"   ✅ Valid features: {len(valid_features)}")
            self.logger.info(f"   {'✅' if len(constant_features) == 0 else '🚨'} Constant features: {len(constant_features)} (SHOULD BE 0)")
            self.logger.info(f"   {'✅' if len(low_variance_features) == 0 else '⚠️'} Low-variance features: {len(low_variance_features)}")
            self.logger.info(f"   {'✅' if len(problematic_features) == 0 else '⚠️'} Problematic features: {len(problematic_features)}")
            
            # CRITICAL: If we have constant features, this indicates a serious issue
            if constant_features:
                self.logger.error(
                    f"🚨 CRITICAL: {len(constant_features)} constant features: "
                    + ", ".join([cf.split(" (" )[0] for cf in constant_features[:50]])
                    + (" ..." if len(constant_features) > 50 else "")
                )
                
                # Provide diagnostic information
                self.logger.error(f"🚨 DIAGNOSTIC INFORMATION:")
                self.logger.error(f"   - Check if time-series data is being processed correctly")
                self.logger.error(f"   - Verify feature engineering is not flattening arrays")
                self.logger.error(f"   - Ensure proper data length validation")
                self.logger.error(f"   - Review the orchestrator feature combination logic")
                
                # Remove constant features to prevent model failure
                features_to_remove = [col.split(" (constant value:")[0] for col in constant_features]
                labeled_data = labeled_data.drop(columns=features_to_remove)
                self.logger.warning(f"🗑️ Removed {len(features_to_remove)} constant features to prevent model failure")
            
            # Remove low variance features
            if low_variance_features:
                low_var_features_to_remove = [col.split(" (variance:")[0] for col in low_variance_features]
                labeled_data = labeled_data.drop(columns=low_var_features_to_remove)
                self.logger.info(f"🗑️ Removed {len(low_var_features_to_remove)} low-variance features")
            
            # Remove problematic features
            if problematic_features:
                names = [col.split(" (")[0] for col in problematic_features]
                labeled_data = labeled_data.drop(columns=names)
                self.logger.warning(
                    f"🗑️ Removed {len(names)} problematic features: " + ", ".join(names[:50]) + (" ..." if len(names) > 50 else "")
                )
            
            # Check for NaN values and handle them
            nan_counts = labeled_data.isnull().sum()
            features_with_nans = nan_counts[nan_counts > 0]
            
            if len(features_with_nans) > 0:
                self.logger.warning(f"⚠️ Found {len(features_with_nans)} features with NaN values")
                # Fill NaN values with 0 for numerical features
                numeric_columns = labeled_data.select_dtypes(include=[np.number]).columns
                labeled_data[numeric_columns] = labeled_data[numeric_columns].fillna(0)
                self.logger.info("✅ Filled NaN values with 0")
            
            # Check for infinite values
            inf_counts = np.isinf(labeled_data.select_dtypes(include=[np.number])).sum()
            features_with_infs = inf_counts[inf_counts > 0]
            
            if len(features_with_infs) > 0:
                self.logger.warning(f"⚠️ Found {len(features_with_infs)} features with infinite values")
                # Replace infinite values with large finite values
                labeled_data = labeled_data.replace([np.inf, -np.inf], [1e6, -1e6])
                self.logger.info("✅ Replaced infinite values with finite bounds")
            
            # Remove raw OHLCV columns to prevent data leakage
            raw_ohlcv_columns = [c for c in self._RAW_CONTEXT_COLUMNS if c in ["open","high","low","close","volume","avg_price","min_price","max_price"]]
            ohlcv_columns_found = [col for col in raw_ohlcv_columns if col in labeled_data.columns]
            if ohlcv_columns_found:
                labeled_data = labeled_data.drop(columns=ohlcv_columns_found)
                self.logger.warning(f"🚨 CRITICAL: Found raw OHLCV columns in features: {ohlcv_columns_found}")
                self.logger.warning(f"🚨 Removed raw OHLCV columns to prevent data leakage!")
                self.logger.warning(f"🚨 This indicates the orchestrator is including raw price data!")
            
            # Final validation check
            remaining_features = [col for col in labeled_data.columns if col != 'label']
            self.logger.info(f"✅ Final feature validation complete:")
            self.logger.info(f"   📊 Remaining features: {len(remaining_features)}")
            self.logger.info(f"   📊 Total samples: {len(labeled_data)}")
            self.logger.info(f"   🗑️ Raw OHLCV columns removed: {ohlcv_columns_found if ohlcv_columns_found else 'None'}")
            
            if len(remaining_features) < 10:
                self.logger.error(f"🚨 CRITICAL: Only {len(remaining_features)} features remaining!")
                self.logger.error(f"🚨 This indicates a severe feature engineering failure!")
                self.logger.error(f"🚨 The model will likely fail to train properly!")
            
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"❌ Feature validation failed: {e}")
            return labeled_data

    async def _log_feature_engineering_summary(self, labeled_data: pd.DataFrame) -> None:
        """Log concise feature engineering summary."""
        try:
            feature_columns = [col for col in labeled_data.columns if col != 'label']
            
            self.logger.info(f"📊 Feature engineering completed: {len(feature_columns)} features, {len(labeled_data)} samples")
            
            # Log only basic label distribution if available
            if 'label' in labeled_data.columns:
                label_distribution = labeled_data['label'].value_counts()
                self.logger.info(f"🎯 Label distribution: {dict(label_distribution)}")
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering summary logging failed: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="analyst labeling and feature engineering step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute analyst labeling and feature engineering step using vectorized orchestrator.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            dict: Updated pipeline state
        """
        self.logger.info("Starting analyst labeling and feature engineering...")

        try:
            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            timeframe = training_input.get("timeframe", "1m")

            # Use unified data loader to get comprehensive data for feature engineering
            self.logger.info("🔄 Loading unified data...")
            data_loader = get_unified_data_loader(self.config)

            # Load unified data with optimizations for ML training (use configured lookback)
            lookback_days = self.config.get("lookback_days", 180)
            price_data = await data_loader.load_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=lookback_days,
                use_streaming=True,  # Enable streaming for large datasets
            )

            if price_data is None or price_data.empty:
                self.logger.error(f"No unified data found for {symbol} on {exchange}")
                return {
                    "status": "FAILED",
                    "error": f"No unified data found for {symbol} on {exchange}",
                }

            # Log data information
            data_info = data_loader.get_data_info(price_data)
            self.logger.info(f"✅ Loaded unified data: {data_info['rows']} rows")

            # Ensure price_data is a DataFrame
            if not isinstance(price_data, pd.DataFrame):
                price_data = pd.DataFrame(price_data)

            # Initial metadata separation for later rejoin
            metadata_cols_present = [c for c in self._METADATA_COLUMNS if c in price_data.columns]
            metadata_df = pd.DataFrame(index=None)
            try:
                # Prepare datetime index for integrity checks
                if not isinstance(price_data.index, pd.DatetimeIndex) and "timestamp" in price_data.columns:
                    price_data = price_data.copy()
                    price_data.index = pd.to_datetime(price_data["timestamp"], errors="coerce")
                    price_data = price_data.sort_index()
                # Extract metadata aligned to index
                if metadata_cols_present:
                    metadata_df = price_data[metadata_cols_present].copy()
            except Exception:
                metadata_df = pd.DataFrame(index=price_data.index)

            # Validate that we have proper OHLCV data for triple barrier labeling
            required_ohlcv_columns = ["open", "high", "low", "close", "volume"]
            missing_ohlcv = [
                col for col in required_ohlcv_columns if col not in price_data.columns
            ]

            if missing_ohlcv:
                self.logger.error(f"Missing required OHLCV columns: {missing_ohlcv}")
                self.logger.error(
                    "Cannot perform proper triple barrier labeling without OHLCV data."
                )
                self.logger.error(f"Available columns: {list(price_data.columns)}")
                return {
                    "status": "FAILED",
                    "error": f"Missing OHLCV columns: {missing_ohlcv}",
                }

            self.logger.info("✅ Validated OHLCV data")

            # Check 1: Missing timestamps -> reindex to continuous grid
            price_data, inserted_rows = self._ensure_continuous_time_index(price_data, freq="1T")
            if inserted_rows > 0:
                self.logger.warning(f"Time index had gaps; inserted {inserted_rows} missing rows during reindex")

            # Dual-pipeline feature engineering
            self.logger.info("🔀 Building Pipeline A (Stationary) and Pipeline B (OHLCV->Stationary)...")
            features_a = await self._build_pipeline_a_stationary(price_data)
            # Check 2: Sanitize raw OHLCV before Pipeline B calculations
            price_data_sanitized = self._sanitize_raw_data(price_data[[c for c in price_data.columns if c in set(self._RAW_PRICE_COLUMNS + ["volume"]) ]].join(
                price_data[[c for c in price_data.columns if c not in set(self._RAW_PRICE_COLUMNS + ["volume"]) ]], how="left"
            )) if isinstance(price_data.index, pd.DatetimeIndex) else self._sanitize_raw_data(price_data.copy())
            features_b = await self._build_pipeline_b_ohlcv(price_data_sanitized)
            combined_features = pd.concat([features_a, features_b], axis=1)

            # Add correlation features (multi-window, cross-timeframe) using raw df context
            try:
                corr_feats = self._build_correlation_features(price_data_sanitized, combined_features)
                if not corr_feats.empty:
                    combined_features = pd.concat([combined_features, corr_feats], axis=1)
            except Exception as e:
                self.logger.warning(f"Correlation features block failed: {e}")

            # Multi-timeframe: previous week's close merged to daily with forward-fill (group-aware)
            try:
                def _weekly_prev_close(g: pd.DataFrame) -> pd.Series:
                    weekly = g["close"].resample("W").last().shift(1)
                    aligned = weekly.reindex(g.index, method="ffill")
                    return aligned
                if isinstance(price_data.index, pd.DatetimeIndex):
                    if "symbol" in price_data.columns and price_data["symbol"].nunique() > 1:
                        wpc = price_data.groupby("symbol", group_keys=False).apply(_weekly_prev_close)
                    else:
                        wpc = _weekly_prev_close(price_data)
                    wpc.name = "weekly_prev_close"
                    combined_features[wpc.name] = wpc
            except Exception as e:
                self.logger.warning(f"Weekly prev close feature generation failed: {e}")

            # Label using optimized triple barrier on raw OHLCV, then align features to labeled index
            try:
                from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
                    OptimizedTripleBarrierLabeling,
                )
                tb_config = self.config.get("vectorized_labelling_orchestrator", {})
                labeler = OptimizedTripleBarrierLabeling(
                    profit_take_multiplier=tb_config.get("profit_take_multiplier", 0.002),
                    stop_loss_multiplier=tb_config.get("stop_loss_multiplier", 0.001),
                    time_barrier_minutes=tb_config.get("time_barrier_minutes", 30),
                    max_lookahead=tb_config.get("max_lookahead", 100),
                )
                labeled_ohlcv = labeler.apply_triple_barrier_labeling_vectorized(
                    price_data[["open", "high", "low", "close", "volume"]].copy()
                )
                # Align feature rows to labeled rows (binary classification removes HOLD rows)
                combined_features = combined_features.loc[labeled_ohlcv.index]
                labeled_data = pd.concat([combined_features, labeled_ohlcv[["label"]]], axis=1)
            except Exception as e:
                self.logger.warning(f"Triple barrier labeling failed, using fallback labels: {e}")
                fallback = self._create_fallback_labeled_data(price_data)
                labeled_data = fallback.get("data", price_data)

            # Final feature sanity: drop raw OHLCV if somehow present
            raw_cols = list(set(self._RAW_CONTEXT_COLUMNS))
            labeled_data = labeled_data.drop(columns=[c for c in raw_cols if c in labeled_data.columns], errors="ignore")

            # Stationarity enforcement (decide using training portion to avoid lookahead)
            try:
                total_rows_tmp = len(labeled_data)
                train_end_tmp = int(total_rows_tmp * 0.8)
                train_mask_tmp = pd.Series(False, index=labeled_data.index)
                if train_end_tmp > 0:
                    train_mask_tmp.iloc[:train_end_tmp] = True
                labeled_data, _transformed_cols = self._enforce_stationarity(labeled_data, train_mask=train_mask_tmp)
            except Exception as e:
                self.logger.warning(f"Stationarity enforcement skipped due to error: {e}")

            # Check 5: Feature Matrix Sanitization (post-merge)
            labeled_data = self._sanitize_features(labeled_data)

            # Drop datetime/timestamp and metadata columns
            try:
                datetime_cols = [
                    c for c in labeled_data.columns if str(labeled_data[c].dtype).startswith("datetime64") or "timestamp" in c.lower()
                ]
                if datetime_cols:
                    self.logger.info(f"Removing datetime columns prior to saving/validation: {datetime_cols}")
                    labeled_data = labeled_data.drop(columns=datetime_cols)
            except Exception:
                pass

            # Keep metadata separate from features (explicit column management)
            meta_cols = [c for c in self._METADATA_COLUMNS if c in labeled_data.columns]
            if meta_cols:
                self.logger.info(f"Separating metadata columns from features: {meta_cols}")
                metadata_df = (metadata_df.join(labeled_data[meta_cols], how="left") if not metadata_df.empty else labeled_data[meta_cols].copy())
                labeled_data = labeled_data.drop(columns=meta_cols)

            # Final guard: drop metadata columns again prior to splitting/saving
            try:
                drop_meta = [c for c in self._METADATA_COLUMNS if c in labeled_data.columns]
                if drop_meta:
                    labeled_data = labeled_data.drop(columns=drop_meta)
            except Exception:
                pass

            # Split the data into train/validation/test sets (80/10/10 split)
            total_rows = len(labeled_data)
            train_end = int(total_rows * 0.8)
            val_end = int(total_rows * 0.9)

            train_data = labeled_data.iloc[:train_end]
            validation_data = labeled_data.iloc[train_end:val_end]
            test_data = labeled_data.iloc[val_end:]

            # Apply VIF loop with CPA on training split; transform val/test accordingly
            try:
                train_data, validation_data, test_data, vif_summary = self._vif_reduce_train_val_test(
                    train_data, validation_data, test_data
                )
            except Exception as e:
                self.logger.warning(f"Step 4 VIF reduction skipped due to error: {e}")

            # Enforce minimum features per category before any global thresholding
            try:
                train_data, validation_data, test_data, cat_log = self._enforce_min_features_per_category(
                    train_data, validation_data, test_data, labeled_data
                )
                if cat_log:
                    self.logger.info(f"Category enforcement summary: {cat_log}")
            except Exception as e:
                self.logger.warning(f"Per-category minimum enforcement skipped: {e}")

            # Persist and log selected feature lists per split (traceability)
            try:
                selected_features = {
                    "train": [c for c in train_data.columns if c != "label"],
                    "validation": [c for c in validation_data.columns if c != "label"],
                    "test": [c for c in test_data.columns if c != "label"],
                }
                trace_path = f"{data_dir}/{exchange}_{symbol}_selected_features.json"
                with open(trace_path, "w") as jf:
                    json.dump(selected_features, jf, indent=2)
                self.logger.info(f"🔎 Saved feature lists to {trace_path}")
            except Exception as e:
                self.logger.warning(f"Could not persist selected feature lists: {e}")

            # Save feature files that the validator expects
            feature_files = [
                (f"{data_dir}/{exchange}_{symbol}_features_train.pkl", train_data),
                (
                    f"{data_dir}/{exchange}_{symbol}_features_validation.pkl",
                    validation_data,
                ),
                (f"{data_dir}/{exchange}_{symbol}_features_test.pkl", test_data),
            ]

            # For features files, drop raw OHLCV/trade columns to avoid leakage
            raw_cols = list(set(self._RAW_CONTEXT_COLUMNS))
            for file_path, data in feature_files:
                features_df = data.drop(columns=[c for c in raw_cols if c in data.columns], errors="ignore")
                with open(file_path, "wb") as f:
                    pickle.dump(features_df, f)
            self.logger.info(f"✅ Saved feature data files")

            # Also save Parquet versions with downcasting for efficiency
            try:
                from src.training.enhanced_training_manager_optimized import (
                    MemoryEfficientDataManager,
                )

                mem_mgr = MemoryEfficientDataManager()
                parquet_files = [
                    (
                        f"{data_dir}/{exchange}_{symbol}_features_train.parquet",
                        mem_mgr.optimize_dataframe(train_data.copy()),
                    ),
                    (
                        f"{data_dir}/{exchange}_{symbol}_features_validation.parquet",
                        mem_mgr.optimize_dataframe(validation_data.copy()),
                    ),
                    (
                        f"{data_dir}/{exchange}_{symbol}_features_test.parquet",
                        mem_mgr.optimize_dataframe(test_data.copy()),
                    ),
                ]
                for file_path, df in parquet_files:
                    try:
                        from src.training.enhanced_training_manager_optimized import (
                            ParquetDatasetManager,
                        )
                        import json as _json

                        feature_cols = list(df.columns)
                        if "label" in feature_cols:
                            feature_cols.remove("label")
                        metadata = {
                            "schema_version": "1",
                            "feature_list": _json.dumps(feature_cols),
                            "feature_hash": str(hash(tuple(sorted(feature_cols))))[:16],
                            "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                            "generator_commit": training_input.get(
                                "generator_commit", ""
                            ),
                        }
                        # Arrow-native cast on timestamp before write
                        import pyarrow as _pa, pyarrow.compute as pc

                        table = _pa.Table.from_pandas(df, preserve_index=False)
                        if "timestamp" in table.schema.names and not _pa.types.is_int64(
                            table.schema.field("timestamp").type
                        ):
                            table = table.set_column(
                                table.schema.get_field_index("timestamp"),
                                "timestamp",
                                pc.cast(table.column("timestamp"), _pa.int64()),
                            )
                        df = table.to_pandas(types_mapper=pd.ArrowDtype)
                        # Drop raw OHLCV for feature parquet
                        drop_cols = [c for c in self._RAW_CONTEXT_COLUMNS if c in df.columns]
                        if drop_cols:
                            df = df.drop(columns=drop_cols)
                        ParquetDatasetManager(self.logger).write_flat_parquet(
                            df,
                            file_path,
                            compression="snappy",
                        )
                    except Exception:
                        from src.utils.logger import (
                            log_io_operation,
                            log_dataframe_overview,
                        )

                        with log_io_operation(
                            self.logger, "to_parquet", file_path, compression="snappy"
                        ):
                            df.to_parquet(file_path, compression="snappy", index=False)
                        try:
                            log_dataframe_overview(
                                self.logger, df, name=f"features_{split_name}"
                            )
                        except Exception:
                            pass
                    pass  # Reduced logging
                # Also write partitioned dataset for features
                try:
                    from src.training.enhanced_training_manager_optimized import (
                        ParquetDatasetManager,
                    )

                    pdm = ParquetDatasetManager(logger=self.logger)
                    base_dir = os.path.join(data_dir, "parquet", "features")
                    for split_name, df in (
                        ("train", mem_mgr.optimize_dataframe(train_data.copy())),
                        (
                            "validation",
                            mem_mgr.optimize_dataframe(validation_data.copy()),
                        ),
                        ("test", mem_mgr.optimize_dataframe(test_data.copy())),
                    ):
                        df = df.copy()
                        df["exchange"] = exchange
                        df["symbol"] = symbol
                        df["timeframe"] = training_input.get("timeframe", "1m")
                        df["split"] = split_name
                        import json as _json

                        feat_cols = list(df.columns)
                        if "label" in feat_cols:
                            feat_cols.remove("label")
                        # Drop raw OHLCV for feature parquet (partitioned)
                        drop_cols = [c for c in self._RAW_CONTEXT_COLUMNS if c in df.columns]
                        if drop_cols:
                            df = df.drop(columns=drop_cols)
                        pdm.write_partitioned_dataset(
                            df=df,
                            base_dir=base_dir,
                            partition_cols=[
                                "exchange",
                                "symbol",
                                "timeframe",
                                "split",
                                "year",
                                "month",
                                "day",
                            ],
                            schema_name="features",
                            compression="snappy",
                            metadata={
                                "schema_version": "1",
                                "feature_list": _json.dumps(feat_cols),
                                "feature_hash": str(hash(tuple(sorted(feat_cols))))[
                                    :16
                                ],
                                "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                                "generator_commit": training_input.get(
                                    "generator_commit", ""
                                ),
                            },
                        )
                except Exception as _e:
                    self.logger.warning(f"Partitioned features write skipped: {_e}")
            except Exception as e:
                self.logger.error(f"Could not save Parquet features: {e}")

            # Also save labeled data files for compatibility
            labeled_files = [
                (f"{data_dir}/{exchange}_{symbol}_labeled_train.pkl", train_data),
                (
                    f"{data_dir}/{exchange}_{symbol}_labeled_validation.pkl",
                    validation_data,
                ),
                (f"{data_dir}/{exchange}_{symbol}_labeled_test.pkl", test_data),
            ]

            for file_path, data in labeled_files:
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
            self.logger.info(f"✅ Saved labeled data files")

            # Parquet for labeled data too
            try:
                # Rejoin metadata for labeled parquet outputs
                def _rejoin_metadata(df_part: pd.DataFrame) -> pd.DataFrame:
                    if isinstance(metadata_df, pd.DataFrame) and not metadata_df.empty:
                        # Align indices if needed
                        aligned = metadata_df.reindex(df_part.index).copy()
                        merged = df_part.join(aligned, how="left")
                        return merged
                    return df_part

                parquet_labeled = [
                    (
                        f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
                        mem_mgr.optimize_dataframe(_rejoin_metadata(train_data.copy())),
                    ),
                    (
                        f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
                        mem_mgr.optimize_dataframe(_rejoin_metadata(validation_data.copy())),
                    ),
                    (
                        f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet",
                        mem_mgr.optimize_dataframe(_rejoin_metadata(test_data.copy())),
                    ),
                ]
                for file_path, df in parquet_labeled:
                    try:
                        from src.training.enhanced_training_manager_optimized import (
                            ParquetDatasetManager,
                        )
                        import json as _json

                        metadata = {
                            "schema_version": "1",
                            "feature_list": _json.dumps(
                                [c for c in df.columns if c != "label"]
                            ),
                            "feature_hash": str(
                                hash(
                                    tuple(
                                        sorted([c for c in df.columns if c != "label"])
                                    )
                                )
                            )[:16],
                            "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                            "generator_commit": training_input.get(
                                "generator_commit", ""
                            ),
                        }
                        ParquetDatasetManager(self.logger).write_flat_parquet(
                            df,
                            file_path,
                            compression="snappy",
                        )
                    except Exception:
                        from src.utils.logger import log_io_operation

                        with log_io_operation(
                            self.logger, "to_parquet", file_path, compression="snappy"
                        ):
                            df.to_parquet(file_path, compression="snappy", index=False)
                    pass  # Reduced logging
                # Also write partitioned dataset for labeled data
                try:
                    from src.training.enhanced_training_manager_optimized import (
                        ParquetDatasetManager,
                    )

                    pdm = ParquetDatasetManager(logger=self.logger)
                    base_dir = os.path.join(data_dir, "parquet", "labeled")
                    for split_name, df in (
                        ("train", mem_mgr.optimize_dataframe(_rejoin_metadata(train_data.copy()))),
                        (
                            "validation",
                            mem_mgr.optimize_dataframe(_rejoin_metadata(validation_data.copy())),
                        ),
                        ("test", mem_mgr.optimize_dataframe(_rejoin_metadata(test_data.copy()))),
                    ):
                        df = df.copy()
                        df["exchange"] = exchange
                        df["symbol"] = symbol
                        df["timeframe"] = training_input.get("timeframe", "1m")
                        df["split"] = split_name
                        import json as _json

                        pdm.write_partitioned_dataset(
                            df=df,
                            base_dir=base_dir,
                            partition_cols=[
                                "exchange",
                                "symbol",
                                "timeframe",
                                "split",
                                "year",
                                "month",
                                "day",
                            ],
                            schema_name="labeled",
                            compression="snappy",
                            metadata={
                                "schema_version": "1",
                                "feature_list": _json.dumps(
                                    [c for c in df.columns if c != "label"]
                                ),
                                "feature_hash": str(
                                    hash(
                                        tuple(
                                            sorted(
                                                [c for c in df.columns if c != "label"]
                                            )
                                        )
                                    )
                                )[:16],
                                "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                                "generator_commit": training_input.get(
                                    "generator_commit", ""
                                ),
                            },
                        )

                    # Materialize model-specific feature projection per split for faster downstream reads
                    try:
                        import json as _json

                        feat_cols = self.config.get(
                            "model_feature_columns"
                        ) or self.config.get("feature_columns")
                        label_col = self.config.get("label_column", "label")
                        if isinstance(feat_cols, list) and len(feat_cols) > 0:
                            cols = ["timestamp", *feat_cols, label_col]
                            model_name = self.config.get("model_name", "default")
                            out_dir = os.path.join(
                                "data_cache", "parquet", f"proj_features_{model_name}"
                            )
                            for split_name in ("train", "validation", "test"):
                                filters = [
                                    ("exchange", "==", exchange),
                                    ("symbol", "==", symbol),
                                    (
                                        "timeframe",
                                        "==",
                                        training_input.get("timeframe", "1m"),
                                    ),
                                    ("split", "==", split_name),
                                ]
                                meta = {
                                    "schema_version": "1",
                                    "feature_list": _json.dumps(feat_cols),
                                    "feature_hash": str(hash(tuple(sorted(feat_cols))))[
                                        :16
                                    ],
                                    "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                                    "generator_commit": training_input.get(
                                        "generator_commit", ""
                                    ),
                                    "split": split_name,
                                }
                                pdm.materialize_projection(
                                    base_dir=base_dir,
                                    filters=filters,
                                    columns=cols,
                                    output_dir=out_dir,
                                    partition_cols=[
                                        "exchange",
                                        "symbol",
                                        "timeframe",
                                        "split",
                                        "year",
                                        "month",
                                        "day",
                                    ],
                                    compression="snappy",
                                    metadata=meta,
                                )
                            self.logger.info(f"✅ Materialized model projections")
                    except Exception as _proj_err:
                        self.logger.warning(
                            f"Feature projection materialization skipped: {_proj_err}"
                        )
                except Exception as _e:
                    self.logger.warning(f"Partitioned labeled write skipped: {_e}")
            except Exception as e:
                self.logger.error(f"Could not save Parquet labeled data: {e}")

            # Integrate UnifiedDataManager to create time-based train/validation/test splits
            try:
                from src.training.data_manager import UnifiedDataManager

                lookback_days = training_input.get(
                    "lookback_days",
                    self.config.get("lookback_days", 180),
                )

                labeled_full = labeled_data.copy()
                # Ensure datetime index for time-based splits
                if "timestamp" in labeled_full.columns:
                    labeled_full["timestamp"] = pd.to_datetime(
                        labeled_full["timestamp"],
                        errors="coerce",
                    )
                    labeled_full = labeled_full.dropna(
                        subset=["timestamp"],
                    )  # drop rows with invalid timestamps
                    labeled_full = labeled_full.set_index("timestamp").sort_index()
                elif not pd.api.types.is_datetime64_any_dtype(labeled_full.index):
                    # Fallback: create a synthetic datetime index to preserve ordering
                    self.logger.warning(
                        "No timestamp column found; creating synthetic datetime index for splits",
                    )
                    labeled_full = labeled_full.copy()
                    labeled_full.index = pd.date_range(
                        end=pd.Timestamp.utcnow(),
                        periods=len(labeled_full),
                        freq="T",
                    )

                data_manager = UnifiedDataManager(
                    data_dir=data_dir,
                    symbol=symbol,
                    exchange=exchange,
                    lookback_days=lookback_days,
                )
                db_result = data_manager.create_unified_database(labeled_full)
                pipeline_state["unified_database"] = db_result
                self.logger.info("✅ Created unified database splits")
            except Exception as e:
                self.logger.exception(
                    f"❌ UnifiedDataManager failed to create splits: {e}",
                )

            # Update pipeline state with results
            pipeline_state.update(
                {
                    "labeled_data": labeled_data,
                    "feature_engineering_metadata": {},
                    "feature_engineering_completed": True,
                    "labeling_completed": True,
                },
            )

            self.logger.info(
                "✅ Analyst labeling and feature engineering completed successfully",
            )
            self.logger.info("Training specialist models for regime: combined (single unified feature set)")
            return {"status": "SUCCESS", "data": {"data": labeled_data}}

        except Exception as e:
            self.logger.exception(
                f"❌ Error in analyst labeling and feature engineering: {e}",
            )
            return {"status": "FAILED", "error": str(e)}

    def _create_fallback_labeled_data(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Create fallback labeled data when orchestrator fails."""
        try:
            # Create simple labeled data with basic features
            labeled_data = price_data.copy()

            # Add basic features
            if "close" in labeled_data.columns:
                labeled_data["returns"] = labeled_data["close"].pct_change()
                labeled_data["volatility"] = (
                    labeled_data["returns"].rolling(window=20).std()
                )
                labeled_data["sma_20"] = labeled_data["close"].rolling(window=20).mean()
                labeled_data["sma_50"] = labeled_data["close"].rolling(window=50).mean()

            # Add simple labels (binary classification: -1 for sell, 1 for buy)
            labeled_data["label"] = -1  # Default to sell signal

            # Create simple buy/sell signals based on moving averages
            if "sma_20" in labeled_data.columns and "sma_50" in labeled_data.columns:
                # Use -1 for sell signal, 1 for buy signal (binary classification)
                labeled_data.loc[
                    labeled_data["sma_20"] > labeled_data["sma_50"],
                    "label",
                ] = 1  # Buy signal
                # Keep -1 for when sma_20 <= sma_50 (sell signal)

            # Remove raw OHLCV columns to prevent data leakage
            raw_ohlcv_columns = [c for c in self._RAW_CONTEXT_COLUMNS if c in ["open","high","low","close","volume","avg_price","min_price","max_price"]]
            columns_to_remove = [col for col in raw_ohlcv_columns if col in labeled_data.columns]
            if columns_to_remove:
                labeled_data = labeled_data.drop(columns=columns_to_remove)
                self.logger.info(f"🗑️ Removed raw OHLCV columns to prevent data leakage: {columns_to_remove}")

            # Remove NaN values
            labeled_data = labeled_data.dropna()

            return {
                "data": labeled_data,
                "metadata": {
                    "labeling_method": "fallback_simple",
                    "features_added": ["returns", "volatility", "sma_20", "sma_50"],
                    "raw_ohlcv_removed": columns_to_remove,
                    "label_distribution": labeled_data["label"]
                    .value_counts()
                    .to_dict(),
                },
            }

        except Exception as e:
            self.logger.error(f"Error creating fallback labeled data: {e}")
            # Return original data with basic label (binary classification)
            price_data_copy = price_data.copy()
            price_data_copy["label"] = -1  # Default to sell signal
            
            # Remove raw OHLCV columns even in error case
            raw_ohlcv_columns = [c for c in self._RAW_CONTEXT_COLUMNS if c in ["open","high","low","close","volume","avg_price","min_price","max_price"]]
            columns_to_remove = [col for col in raw_ohlcv_columns if col in price_data_copy.columns]
            if columns_to_remove:
                price_data_copy = price_data_copy.drop(columns=columns_to_remove)
                self.logger.info(f"🗑️ Removed raw OHLCV columns in error case: {columns_to_remove}")
            
            return {
                "data": price_data_copy,
                "metadata": {"labeling_method": "fallback_basic", "error": str(e), "raw_ohlcv_removed": columns_to_remove},
            }

    def _run_post_merge_feature_checks(self, features_df: pd.DataFrame) -> None:
        """Run ADF stationarity and distribution/outlier sanity checks.
        Logs warnings for non-stationary and pathological distributions.
        """
        if features_df is None or features_df.empty:
            return
        cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            return
        # ADF tests
        if _ADF_AVAILABLE:
            for c in cols:
                try:
                    s = features_df[c].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                    if len(s) < 30:
                        continue
                    pval = float(_adfuller(s, autolag='AIC')[1])
                    if pval >= 0.05:
                        self.logger.warning(f"WARNING: Feature '{c}' is not stationary (ADF p-value: {pval:.4f})!")
                except Exception:
                    continue
        else:
            self.logger.warning("ADF test not available; skipping stationarity checks")
        # Distribution sanity checks
        for c in cols:
            try:
                s = features_df[c].astype(float)
                desc = s.describe()
                skew = float(s.skew()) if np.isfinite(s.skew()) else 0.0
                kurt = float(s.kurtosis()) if np.isfinite(s.kurtosis()) else 0.0
                if abs(skew) > 100 or abs(kurt) > 100:
                    self.logger.warning(f"Feature '{c}' has extreme skew/kurtosis (skew={skew:.2f}, kurt={kurt:.2f})")
                std = float(desc.get('std', 0.0) or 0.0)
                if std < 1e-12:
                    self.logger.warning(f"Feature '{c}' has near-zero variance (std={std:.2e})")
            except Exception:
                continue

    def _log_mutual_information_warnings_step4(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute MI between each feature and label; warn on near-zero MI.
        Blank mode (env BLANK_TRAINING_MODE=1): threshold 1e-5; Full: bottom 20% percentile.
        """
        if X is None or X.empty or y is None or y.empty:
            return
        numeric = X.select_dtypes(include=[np.number])
        if numeric.empty:
            return
        # Blank mode detection
        blank_mode = False
        try:
            blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        except Exception:
            pass
        mi = mutual_info_classif(numeric.values, y.values, discrete_features=False, random_state=42)
        mi_series = pd.Series(mi, index=numeric.columns)
        if blank_mode:
            low = mi_series[mi_series <= 1e-5]
            threshold_txt = "1e-5"
        else:
            thr = mi_series.quantile(0.20)
            low = mi_series[mi_series <= thr]
            threshold_txt = f"{thr:.4g}"
        if not low.empty:
            names = low.sort_values().index.tolist()
            self.logger.warning(
                f"MI (Step4): {len(names)} features show near-zero uni-variate predictive power (<= {threshold_txt}): {names[:50]}{' ...' if len(names)>50 else ''}"
            )

    def _log_feature_stability_warnings_step4(self, X: pd.DataFrame) -> None:
        """4-fold CV stability check on features; warn if std of fold means exceeds 3x expected standard error."""
        if X is None or X.empty:
            return
        numeric = X.select_dtypes(include=[np.number])
        if numeric.empty:
            return
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        unstable: list[str] = []
        for col in numeric.columns:
            try:
                vals = numeric[col].astype(float).values
                gstd = float(np.nanstd(vals))
                if not np.isfinite(gstd) or gstd == 0.0:
                    continue
                fold_means = []
                for train_idx, _ in kf.split(vals):
                    fm = float(np.nanmean(vals[train_idx]))
                    if np.isfinite(fm):
                        fold_means.append(fm)
                if len(fold_means) < 2:
                    continue
                std_of_means = float(np.nanstd(fold_means))
                expected_se = gstd / np.sqrt(4)
                if std_of_means > 3.0 * expected_se:
                    unstable.append(col)
            except Exception:
                continue
        if unstable:
            self.logger.warning(
                f"Stability (Step4): {len(unstable)} features are unstable across folds: {unstable[:50]}{' ...' if len(unstable)>50 else ''}"
            )

    def _adf_pvalue(self, series: pd.Series) -> float:
        """Compute ADF p-value; return 1.0 on failure or insufficient data."""
        try:
            s = series.dropna().astype(float)
            if len(s) < 30 or not _ADF_AVAILABLE:
                return 1.0
            result = _adfuller(s, autolag="AIC")
            return float(result[1])
        except Exception:
            return 1.0

    def _fracdiff_weights(self, d: float, size: int, cutoff: float = 1e-5) -> np.ndarray:
        """Compute fractional differencing weights up to size with magnitude cutoff."""
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - (k - 1)) / k
            if abs(w_k) < cutoff:
                break
            w.append(w_k)
        return np.array(w, dtype=float)

    def _fracdiff_series(self, s: pd.Series, d: float = 0.4, cutoff: float = 1e-5) -> pd.Series:
        """Apply fractional differencing to a series with parameter d."""
        try:
            s = s.astype(float)
            w = self._fracdiff_weights(d, len(s), cutoff)
            if len(w) <= 1:
                return s.diff()
            # Convolution-like operation (finite weights)
            vals = s.values
            out = np.full_like(vals, fill_value=np.nan, dtype=float)
            for i in range(len(w) - 1, len(vals)):
                window = vals[i - (len(w) - 1) : i + 1]
                if np.any(~np.isfinite(window)):
                    continue
                out[i] = np.dot(w[::-1], window)
            return pd.Series(out, index=s.index)
        except Exception:
            return s.diff()

    def _enforce_stationarity(self, df: pd.DataFrame, train_mask: pd.Series | None = None) -> tuple[pd.DataFrame, list[str]]:
        """Enforce stationarity on non-stationary numeric features using training portion for detection.
        Uses percentage change only (group-aware). Returns transformed DataFrame and list of transformed columns.
        """
        if df.empty:
            return df, []
        data = df.copy()
        numeric_cols = [c for c in data.columns if c != self._LABEL_NAME and np.issubdtype(data[c].dtype, np.number)]
        if not numeric_cols:
            return data, []
        # Heuristics: skip obviously stationary/bounded metrics
        def is_likely_stationary_name(name: str) -> bool:
            keywords = ["return", "z", "distance", "ratio", "imbalance", "volatility", "pattern", "ofi"]
            name_l = name.lower()
            return any(k in name_l for k in keywords)
        transformed: list[str] = []
        # Determine baseline for ADF from training mask
        train_idx = data.index if train_mask is None else data.index[train_mask]
        # Group-aware transformation
        has_symbol = "symbol" in data.columns and data["symbol"].nunique() > 1
        for col in numeric_cols:
            if is_likely_stationary_name(col):
                continue
            # ADF over training portion only
            try:
                train_series = data.loc[train_idx, col]
                pval = self._adf_pvalue(train_series)
            except Exception:
                pval = 1.0
            if pval > 0.05:
                # Non-stationary: transform using pct_change (group-aware)
                try:
                    if has_symbol:
                        data[col] = data.groupby("symbol", group_keys=False)[col].pct_change()
                    else:
                        data[col] = data[col].pct_change()
                    transformed.append(col)
                except Exception:
                    # Fallback to simple diff
                    try:
                        if has_symbol:
                            data[col] = data.groupby("symbol", group_keys=False)[col].diff()
                        else:
                            data[col] = data[col].diff()
                        transformed.append(col)
                    except Exception:
                        pass
        if transformed:
            self.logger.info(f"Stationarity enforcement (pct_change) applied to {len(transformed)} columns: {transformed[:30]}{' ...' if len(transformed)>30 else ''}")
        return data, transformed

    # -----------------
    # Group-aware helpers
    # -----------------
    def _get_series_for_calc(self, df: pd.DataFrame, col: str):
        """Return a Series or grouped Series for group-aware calculations."""
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            return df.groupby("symbol")[col]
        return df[col]

    def _group_aware_rolling_std(self, series: pd.Series, df: pd.DataFrame, window: int) -> pd.Series:
        """Compute rolling std per symbol group when applicable, preserving index."""
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            return (
                series.groupby(df["symbol"]).rolling(window).std().reset_index(level=0, drop=True)
            )
        return series.rolling(window).std()

    # --------------
    # Category utils
    # --------------
    def _feature_category(self, name: str) -> str | None:
        n = name.lower()
        if any(k in n for k in ["wavelet", "cwt_", "dwt_", "wl_"]):
            return "wavelet"
        if any(k in n for k in ["market_depth", "bid_ask_spread", "ofi", "imbalance", "liquidity", "spread", "queue", "slippage"]):
            return "microstructure"
        if any(k in n for k in ["engulf", "doji", "hammer", "marubozu", "tweezer", "shooting", "candle", "pattern"]):
            return "candlestick"
        if any(k in n for k in ["return", "momentum", "roc", "sma", "ema", "vwap_distance", "distance"]):
            return "momentum"
        if any(k in n for k in ["volatility", "atr", "gk_vol", "bb_width", "variance"]):
            return "volatility"
        if any(k in n for k in ["corr", "correlation", "cov", "beta"]):
            return "correlation"
        if any(k in n for k in ["support", "resistance", "sr_", "pivot"]):
            return "sr"
        return None

    def _enforce_min_features_per_category(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        original_labeled_df: pd.DataFrame,
        min_per_category: dict[str, int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
        """Ensure at least N features per category by re-adding top-MI candidates from the original set.

        Returns updated (train, val, test) and a log dict with counts added per category.
        """
        if min_per_category is None:
            min_per_category = {
                "wavelet": 2,
                "microstructure": 2,
                "candlestick": 2,
                "momentum": 3,
                "volatility": 3,
                "correlation": 2,
                "sr": 3,
            }
        # Current features
        current_cols = [c for c in train_df.columns if c != self._LABEL_NAME]
        cat_to_cols: dict[str, list[str]] = {}
        for c in current_cols:
            cat = self._feature_category(c)
            if cat:
                cat_to_cols.setdefault(cat, []).append(c)
        y = train_df[self._LABEL_NAME].astype(int)
        added_counts: dict[str, dict[str, int]] = {}
        # Candidate pool from original labeled features
        original_pool = [
            c for c in original_labeled_df.columns
            if c != self._LABEL_NAME and c not in self._RAW_CONTEXT_COLUMNS
        ]
        for cat, min_n in min_per_category.items():
            have = len(cat_to_cols.get(cat, []))
            need = max(0, min_n - have)
            if need <= 0:
                continue
            # Candidates for this category not currently included
            candidates = [
                c for c in original_pool
                if c not in current_cols and self._feature_category(c) == cat and np.issubdtype(original_labeled_df[c].dtype, np.number)
            ]
            if not candidates:
                # No available candidates; skip gracefully
                continue
            # Compute MI on training split for candidates
            try:
                X_cand = original_labeled_df.loc[train_df.index, candidates].copy()
                X_cand = X_cand.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                mi = mutual_info_classif(X_cand.values, y.values, discrete_features=False, random_state=42)
                order = np.argsort(mi)[::-1]
                chosen = [candidates[i] for i in order[:need] if np.isfinite(mi[i])]
            except Exception:
                chosen = candidates[:need]
            if not chosen:
                continue
            # Add back to all splits from original_labeled_df
            for col in chosen:
                train_df[col] = original_labeled_df.loc[train_df.index, col]
                if not val_df.empty:
                    val_df[col] = original_labeled_df.loc[val_df.index, col]
                if not test_df.empty:
                    test_df[col] = original_labeled_df.loc[test_df.index, col]
            added_counts[cat] = {"needed": need, "added": len(chosen)}
        return train_df, val_df, test_df, added_counts

    # -----------------------------
    # Correlation feature utilities
    # -----------------------------
    def _compute_simple_ofi(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Simple Order Flow Imbalance proxy using tick rule and volume normalization.
        OFI ≈ sum(sign(Δclose) * volume) / sum(volume) over a rolling window.
        """
        required = ["close", "volume"]
        name = f"order_flow_imbalance_{window}"
        if not all(c in df.columns for c in required):
            return pd.Series(np.nan, index=df.index, name=name)
        def _impl_group(g: pd.DataFrame) -> pd.Series:
            c = g["close"].astype(float)
            v = g.get("volume", pd.Series(0, index=g.index)).astype(float).clip(lower=0)
            sign = np.sign(c.diff().fillna(0.0))
            num = (sign * v).rolling(window, min_periods=max(3, window // 3)).sum()
            den = v.rolling(window, min_periods=max(3, window // 3)).sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                ofi = (num / den.replace(0, np.nan)).clip(-1, 1)
            return ofi
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            s = df.groupby("symbol", group_keys=False).apply(_impl_group)
        else:
            s = _impl_group(df)
        return s.replace([np.inf, -np.inf], np.nan).rename(name)

    def _compute_realized_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        name = f"realized_volatility_{window}"
        if "close" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=name)
        def _impl_group(g: pd.DataFrame) -> pd.Series:
            r = g["close"].astype(float).pct_change()
            return r.rolling(window, min_periods=max(5, window // 2)).std()
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            s = df.groupby("symbol", group_keys=False).apply(_impl_group)
        else:
            s = _impl_group(df)
        return s.rename(name)

    def _compute_momentum(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        name = f"momentum_{lookback}"
        if "close" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=name)
        def _impl_group(g: pd.DataFrame) -> pd.Series:
            c = g["close"].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                m = (c / c.shift(lookback)) - 1.0
            return m
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            s = df.groupby("symbol", group_keys=False).apply(_impl_group)
        else:
            s = _impl_group(df)
        return s.replace([np.inf, -np.inf], np.nan).rename(name)

    def _compute_avg_trade_size(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        name = f"average_trade_size_{window}"
        if not {"trade_volume", "trade_count"}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index, name=name)
        def _impl_group(g: pd.DataFrame) -> pd.Series:
            tv = g["trade_volume"].astype(float)
            tc = g["trade_count"].astype(float).replace(0, np.nan)
            ats = (tv / tc).rolling(window, min_periods=max(3, window // 3)).mean()
            return ats
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            s = df.groupby("symbol", group_keys=False).apply(_impl_group)
        else:
            s = _impl_group(df)
        return s.rename(name)

    def _compute_price_minus_vwap(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        name = f"price_minus_vwap_{window}"
        if not {"close", "volume"}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index, name=name)
        def _impl_group(g: pd.DataFrame) -> pd.Series:
            c = g["close"].astype(float)
            v = g["volume"].astype(float).clip(lower=0)
            num = (c * v).rolling(window, min_periods=max(2, window // 2)).sum()
            den = v.rolling(window, min_periods=max(2, window // 2)).sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap = num / den.replace(0, np.nan)
                diff = c - vwap
            return diff
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            s = df.groupby("symbol", group_keys=False).apply(_impl_group)
        else:
            s = _impl_group(df)
        return s.replace([np.inf, -np.inf], np.nan).rename(name)

    def _build_cross_timeframe_returns(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Build cross-timeframe returns (1m base; 5m and 15m resampled) aligned to base index."""
        if not isinstance(df.index, pd.DatetimeIndex) or "close" not in df.columns:
            return {}
        out: dict[str, pd.Series] = {}
        base_idx = df.index
        close = df["close"].astype(float)
        # 1m returns
        ret_1m = close.pct_change().rename("returns_1m")
        out[ret_1m.name] = ret_1m
        # Higher TF returns using last price in bucket, then align with ffill (no lookahead)
        for tf_label, rule in [("5m", "5T"), ("15m", "15T")]:
            res = close.resample(rule).last()
            ret = res.pct_change().rename(f"returns_{tf_label}")
            ret_aligned = ret.reindex(base_idx, method="ffill")
            out[ret_aligned.name] = ret_aligned
        return out

    def _build_correlation_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
        """Construct requested rolling correlation features with multi-window support."""
        out = pd.DataFrame(index=df.index)
        windows = [30, 60, 120]
        # Prepare prerequisite series
        mom20 = self._compute_momentum(df, 20)
        mom50 = self._compute_momentum(df, 50)
        ofi20 = self._compute_simple_ofi(df, 20)
        rv20 = self._compute_realized_volatility(df, 20)
        # Prefer realized_volatility_20 existing column if present
        if "realized_volatility_20" in base_features.columns:
            rv20 = base_features["realized_volatility_20"].astype(float)
            rv20.name = "realized_volatility_20"
        # Volume
        vol = df.get("volume", pd.Series(np.nan, index=df.index)).astype(float).rename("volume")
        # Price returns (1m)
        ret_1m = base_features.get("close_returns", self._feat_close_returns(df)).astype(float).rename("price_returns")
        # VWAP deviation
        pmvwap20 = self._compute_price_minus_vwap(df, 20)
        # Average trade size
        ats20 = self._compute_avg_trade_size(df, 20)
        # Cross-timeframe returns
        tf_returns = self._build_cross_timeframe_returns(df)
        # Correlation builders
        def add_corr(a: pd.Series, b: pd.Series, w: int, key: str):
            try:
                a_local = a.astype(float)
                b_local = b.astype(float)
                a_local.name = "a"; b_local.name = "b"
                s = self._group_aware_rolling_corr(a_local, b_local, df, w)
                out[f"{key}_corr_{w}"] = s
            except Exception:
                pass
        # Momentum vs OFI
        for w in windows:
            add_corr(mom20, ofi20, w, "momentum20_ofi20")
        # Volatility vs Volume
        for w in windows:
            add_corr(rv20, vol, w, "realized_volatility20_volume")
        # Price vs Microstructure (price returns vs OFI)
        for w in windows:
            add_corr(ret_1m, ofi20.rename("order_flow_imbalance"), w, "price_returns_ofi")
        # Momentum vs Volatility
        for w in windows:
            add_corr(mom50, rv20, w, "momentum50_realized_volatility20")
        # Cross-Timeframe Correlation (returns_1m vs returns_5m/15m)
        r1 = tf_returns.get("returns_1m")
        for tf in ("returns_5m", "returns_15m"):
            rt = tf_returns.get(tf)
            if r1 is not None and rt is not None:
                for w in windows:
                    add_corr(r1, rt, w, f"returns1m_{tf}")
        # VWAP deviation vs Volume
        for w in windows:
            add_corr(pmvwap20, vol, w, "price_minus_vwap20_volume")
        # Trade size vs momentum
        for w in windows:
            add_corr(ats20, mom20, w, "avg_trade_size20_momentum20")
        # Cleanup
        if not out.empty:
            out = out.replace([np.inf, -np.inf], np.nan)
        return out


class DeprecatedAnalystLabelingFeatureEngineeringStep:
    """Step 4: Analyst Labeling and Feature Engineering (DEPRECATED)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild(
            "DeprecatedAnalystLabelingFeatureEngineeringStep",
        )

    async def initialize(self) -> None:
        """Initialize the analyst labeling and feature engineering step."""
        self.logger.info(
            "Initializing Deprecated Analyst Labeling and Feature Engineering Step...",
        )
        self.logger.info(
            "Deprecated Analyst Labeling and Feature Engineering Step initialized successfully",
        )

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute analyst labeling and feature engineering step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            dict: Updated pipeline state
        """
        self.logger.info(
            "Executing deprecated analyst labeling and feature engineering step...",
        )

        # This step is deprecated - return current state
        return pipeline_state

    async def _apply_triple_barrier_labeling(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> pd.DataFrame:
        """
        Apply Triple Barrier Method for labeling.

        Args:
            data: Market data for the regime
            regime_name: Name of the regime

        Returns:
            DataFrame with labels added
        """
        self.logger.info(
            f"Applying Triple Barrier Method for regime: {regime_name}",
        )

        # Ensure we have required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                msg = f"Missing required column: {col}"
                raise ValueError(msg)

        # Calculate features for labeling
        data = self._calculate_features(data)

        # Apply Triple Barrier Method
        labeled_data = data.copy()

        # Define barrier parameters based on regime
        profit_take_multiplier = 0.002  # 0.2%
        stop_loss_multiplier = 0.001  # 0.1%
        # Use environment-aware default consistent with vectorized orchestrator
        try:
            if os.environ.get("BLANK_TRAINING_MODE", "0") == "1":
                time_barrier_minutes = 90
            elif os.environ.get("FULL_TRAINING_MODE", "0") == "1":
                time_barrier_minutes = 360
            else:
                time_barrier_minutes = 30
        except Exception:
            time_barrier_minutes = 30

        # Apply triple barrier labeling
        labels = []
        for i in range(len(data)):
            if i >= len(data) - 1:  # Skip last point
                labels.append(0)
                continue

            entry_price = data.iloc[i]["close"]
            entry_time = data.index[i]

            # Calculate barriers
            profit_take_barrier = entry_price * (1 + profit_take_multiplier)
            stop_loss_barrier = entry_price * (1 - stop_loss_multiplier)
            time_barrier = entry_time + timedelta(minutes=time_barrier_minutes)

            # Check if any barrier is hit
            label = -1  # Default to sell signal for binary classification

            for j in range(
                i + 1,
                min(i + 100, len(data)),
            ):  # Look ahead up to 100 points
                current_time = data.index[j]
                current_price = data.iloc[j]["high"]  # Use high for profit take
                current_low = data.iloc[j]["low"]  # Use low for stop loss

                # Check time barrier
                if current_time > time_barrier:
                    label = -1  # Time barrier hit - default to sell signal
                    break

                # Check profit take barrier
                if current_price >= profit_take_barrier:
                    label = 1  # Profit take hit - positive
                    break

                # Check stop loss barrier
                if current_low <= stop_loss_barrier:
                    label = -1  # Stop loss hit - negative
                    break

            labels.append(label)

        labeled_data["label"] = labels

        # Calculate label distribution
        label_counts = pd.Series(labels).value_counts()
        self.logger.info(
            f"Label distribution for {regime_name}: {dict(label_counts)}",
        )

        return labeled_data

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features for the data.

        Args:
            data: Market data

        Returns:
            DataFrame with features added
        """
        try:
            # Calculate RSI
            data["rsi"] = self._calculate_rsi(data["close"])

            # Calculate MACD
            macd, signal = self._calculate_macd(data["close"])
            data["macd"] = macd
            data["macd_signal"] = signal
            data["macd_histogram"] = macd - signal

            # Calculate Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(data["close"])
            data["bb_upper"] = bb_upper
            data["bb_lower"] = bb_lower
            data["bb_width"] = bb_upper - bb_lower
            data["bb_position"] = (data["close"] - bb_lower) / (bb_upper - bb_lower)

            # Calculate ATR
            data["atr"] = self._calculate_atr(data)

            # Calculate price-based features
            data["price_change"] = data["close"].pct_change()
            data["price_change_abs"] = data["price_change"].abs()
            data["high_low_ratio"] = data["high"] / data["low"]
            data["volume_price_ratio"] = data["volume"] / data["close"]

            # Calculate moving averages
            data["sma_5"] = data["close"].rolling(window=5).mean()
            data["sma_10"] = data["close"].rolling(window=10).mean()
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["ema_12"] = data["close"].ewm(span=12).mean()
            data["ema_26"] = data["close"].ewm(span=26).mean()

            # Calculate momentum features
            data["momentum_5"] = data["close"] / data["close"].shift(5) - 1
            data["momentum_10"] = data["close"] / data["close"].shift(10) - 1
            data["momentum_20"] = data["close"] / data["close"].shift(20) - 1

            # Add candlestick pattern features using advanced feature engineering
            # Legacy S/R/Candle code removed - using simplified approach
            data = data  # Keep original data for now

            # Fill NaN values
            return data.fillna(method="bfill").fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    async def _add_candlestick_pattern_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add candlestick pattern features using advanced feature engineering.

        Args:
            data: Market data with OHLCV

        Returns:
            DataFrame with candlestick pattern features added
        """
        try:
            self.logger.info("Adding candlestick pattern features...")

            # Import advanced feature engineering
            from src.analyst.advanced_feature_engineering import (
                AdvancedFeatureEngineering,
            )

            # Initialize advanced feature engineering
            config = {
                "advanced_features": {
                    # Legacy S/R/Candle code removed,
                    "enable_volatility_regime_modeling": True,
                    "enable_correlation_analysis": True,
                    "enable_momentum_analysis": True,
                    "enable_liquidity_analysis": True,
                },
                # Legacy S/R/Candle code removed
            }

            # Initialize advanced feature engineering
            advanced_fe = AdvancedFeatureEngineering(config)
            await advanced_fe.initialize()

            # Prepare data for feature engineering
            price_data = data[["open", "high", "low", "close"]].copy()
            volume_data = data[["volume"]].copy()

            # Get advanced features including candlestick patterns
            advanced_features = await advanced_fe.engineer_features(
                price_data=price_data,
                volume_data=volume_data,
                order_flow_data=None,
            )

            # Convert features to DataFrame and align with original data
            if advanced_features:
                # Align per-row features to data length; skip scalars
                aligned = pd.DataFrame(index=data.index)
                n = len(data)
                for name, val in advanced_features.items():
                    arr = None
                    if isinstance(val, pd.Series):
                        arr = val.values
                    elif isinstance(val, np.ndarray):
                        if val.ndim == 1:
                            arr = val
                        elif val.ndim == 2 and (val.shape[0] == 1 or val.shape[1] == 1):
                            arr = val.reshape(-1)
                    elif isinstance(val, list):
                        tmp = np.asarray(val)
                        if tmp.ndim == 1:
                            arr = tmp
                        elif tmp.ndim == 2 and (tmp.shape[0] == 1 or tmp.shape[1] == 1):
                            arr = tmp.reshape(-1)
                    if arr is None:
                        continue
                    if len(arr) > n:
                        arr = arr[-n:]
                    elif len(arr) < n:
                        pad = n - len(arr)
                        arr = np.concatenate([np.full(pad, np.nan), arr])
                    try:
                        aligned[name] = pd.to_numeric(arr, errors="coerce")
                    except Exception:
                        pass

                # Drop fully-NaN and constant columns
                if not aligned.empty:
                    aligned = aligned.dropna(axis=1, how="all")
                    nunique = aligned.nunique(dropna=True)
                    const_cols = nunique[nunique <= 1].index.tolist()
                    if const_cols:
                        self.logger.warning(f"Dropping {len(const_cols)} constant candlestick features")
                        aligned = aligned.drop(columns=const_cols)

                # Add aligned features to original data
                for col in aligned.columns:
                    if col not in data.columns:
                        data[col] = aligned[col]

                self.logger.info(
                    f"Added {len(aligned.columns)} candlestick pattern features",
                )
            else:
                self.logger.error("No candlestick pattern features generated")

            return data

        except Exception as e:
            self.logger.error(f"Error adding candlestick pattern features: {e}")
            return data

    async def _perform_feature_engineering(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> pd.DataFrame:
        """
        Perform feature engineering for the regime.

        Args:
            data: Labeled data with features
            regime_name: Name of the regime

        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info(
                f"Performing feature engineering for regime: {regime_name}",
            )

            # Select feature columns including candlestick patterns
            feature_columns = [
                "rsi",
                "macd",
                "macd_signal",
                "macd_histogram",
                "bb_upper",
                "bb_lower",
                "bb_width",
                "bb_position",
                "atr",
                "price_change",
                "price_change_abs",
                "high_low_ratio",
                "volume_price_ratio",
                "sma_5",
                "sma_10",
                "sma_20",
                "ema_12",
                "ema_26",
                "momentum_5",
                "momentum_10",
                "momentum_20",
            ]

            # Add candlestick pattern features
            # Legacy S/R/Candle code removed

            # Combine all feature columns
            # Legacy S/R/Candle code removed

            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]

            if not available_features:
                msg = f"No features available for regime {regime_name}"
                raise ValueError(msg)

            # Create feature matrix
            feature_data = data[available_features].copy()

            # Remove rows with NaN values
            feature_data = feature_data.dropna()

            # Standardize features
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)
            feature_data_scaled = pd.DataFrame(
                feature_data_scaled,
                columns=available_features,
                index=feature_data.index,
            )

            # Add label column
            feature_data_scaled["label"] = data.loc[feature_data.index, "label"]

            self.logger.info(
                f"Feature engineering completed for {regime_name}: {len(feature_data_scaled)} samples, {len(available_features)} features",
            )

            return feature_data_scaled

        except Exception as e:
            self.logger.exception(
                f"Error in feature engineering for {regime_name}: {e}",
            )
            raise

    async def _train_regime_encoders(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> dict[str, Any]:
        """
        Train regime-specific encoders.

        Args:
            data: Feature-engineered data
            regime_name: Name of the regime

        Returns:
            Dict containing trained encoders
        """
        try:
            self.logger.info(f"Training encoders for regime: {regime_name}")

            # Separate features and labels
            feature_columns = [col for col in data.columns if col != "label"]
            X = data[feature_columns]
            y = data["label"]

            # Train PCA encoder for dimensionality reduction
            pca = PCA(n_components=min(10, len(feature_columns)))
            pca.fit_transform(X)

            # Train autoencoder for feature learning
            from sklearn.neural_network import MLPRegressor

            # Ensure we have enough data and features for autoencoder
            if len(X) < 10 or len(feature_columns) < 2:
                self.logger.warning(
                    f"Insufficient data for autoencoder training: {len(X)} samples, {len(feature_columns)} features"
                )
                # Create a simple identity encoder as fallback
                autoencoder = None
            else:
                try:
                    # Use a simpler architecture for better convergence
                    hidden_size = max(2, min(len(feature_columns) // 2, 10))
                    autoencoder = MLPRegressor(
                        hidden_layer_sizes=(hidden_size,),
                        max_iter=500,  # Reduced for faster training
                        random_state=42,
                        alpha=0.01,  # L2 regularization
                        learning_rate_init=0.001,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                    )
                    autoencoder.fit(X, X)
                    self.logger.info(
                        f"Autoencoder trained successfully with {hidden_size} hidden units"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Autoencoder training failed: {e}, using fallback"
                    )
                    autoencoder = None

            # Store encoders
            encoders = {
                "pca": pca,
                "autoencoder": autoencoder,
                "feature_columns": feature_columns,
                "n_features": len(feature_columns),
                "n_samples": len(X),
            }

            # Log encoder information
            pca_info = f"PCA components={pca.n_components_}"
            autoencoder_info = f"Autoencoder layers={autoencoder.n_layers_ if autoencoder else 'None (fallback)'}"
            self.logger.info(
                f"Encoders trained for {regime_name}: {pca_info}, {autoencoder_info}",
            )

            return encoders

        except Exception as e:
            self.logger.error(f"Error training encoders for {regime_name}: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange_name: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    exchange: str = "BINANCE",
    force_rerun: bool = False,
) -> bool:
    """
    Run analyst labeling and feature engineering step using vectorized orchestrator.

    Args:
        symbol: Trading symbol
        exchange_name: Exchange name (deprecated, use exchange)
        data_dir: Data directory path
        timeframe: Timeframe for data
        exchange: Exchange name

    Returns:
        bool: True if successful, False otherwise
    """
    print(
        "🚀 Running analyst labeling and feature engineering step with vectorized orchestrator..."
    )

    # Use exchange parameter if provided, otherwise use exchange_name for backward compatibility
    actual_exchange = exchange if exchange != "BINANCE" else exchange_name

    try:
        # Create step instance
        config = {
            "symbol": symbol,
            "exchange": actual_exchange,
            "data_dir": data_dir,
            "timeframe": timeframe,
        }
        step = AnalystLabelingFeatureEngineeringStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": actual_exchange,
            "data_dir": data_dir,
            "timeframe": timeframe,
            "force_rerun": force_rerun,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS" if isinstance(result, dict) else True

    except Exception as e:
        print(f"Analyst labeling and feature engineering failed: {e}")
        return False


# For backward compatibility with existing step structure
async def deprecated_run_step(
    symbol: str,
    exchange_name: str = "BINANCE",
    data_dir: str = "data/training",
) -> bool:
    """
    DEPRECATED: Run analyst labeling and feature engineering step.

    This function is deprecated and should not be used in new training pipelines.
    """
    print(
        "⚠️  WARNING: This step is deprecated and should not be used in new training pipelines.",
    )
    return True


if __name__ == "__main__":
    # Test the step
    async def test():
        """Test the analyst labeling and feature engineering step."""
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Step test result: {result}")

    asyncio.run(test())
