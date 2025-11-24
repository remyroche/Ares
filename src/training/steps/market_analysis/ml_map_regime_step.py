"""
ML Map Regime Step

Market Structure Map step that consumes 1h OHLCV data to construct
structure-based features (volume profile HVN/LVN, pivots, FVG, SFP,
PDH/PDL proximity, anchored VWAP, wick ratios) and symmetric long/short
alpha context.
"""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import pandas as pd

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


logger = logging.getLogger(__name__)


class MLMapRegimeStep(BaseStep):
    def __init__(self, step_name: str = "ml_map_regime_step"):
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLMapRegimeStep") if hasattr(logger, "getChild") else logger
        self._cached_market_data_15m = None
        self._cached_market_source_15m = None
        self._cached_market_cache_key_15m = None
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        artifacts: List[str] = []
        metrics: Dict[str, Any] = {}
        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            regime_timeframe = str(config.get("regime_timeframe", config.get("timeframe", "15m")))
            direction = str(config.get("direction", "long"))
            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")
            tprint_info(
                f"🚀 Starting {self.step_name} (Market Structure Map) for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )
            exec_mode_cfg = str(config.get("execution_mode", "")).lower()
            cache_key = (symbol, exchange, regime_timeframe, exec_mode_cfg, "map")
            market_data = None
            market_source = None
            if (
                getattr(self, "_cached_market_data_15m", None) is not None
                and getattr(self, "_cached_market_cache_key_15m", None) == cache_key
            ):
                try:
                    market_data = self._cached_market_data_15m.copy()
                except Exception:
                    market_data = self._cached_market_data_15m
                market_source = self._cached_market_source_15m
                tprint_info("♻️ Reusing cached 15m market data for ML map regimes")
            else:
                market_data, market_source = self.load_market_data_or_fail(
                    {**config, "timeframe": regime_timeframe},
                    pipeline_state={},
                    allow_config_override=True,
                    light_mode_filter=False,
                    skip_artifacts=True,
                )
                if isinstance(market_data, pd.DataFrame):
                    self._cached_market_data_15m = market_data.copy()
                else:
                    self._cached_market_data_15m = market_data
                self._cached_market_source_15m = market_source
                self._cached_market_cache_key_15m = cache_key
            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")
            if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data = market_data.copy()
                market_data.index = pd.to_datetime(market_data.index)
                if market_data.index.tz is not None:
                    market_data.index = market_data.index.tz_convert(None)
            market_data = market_data.sort_index()
            tprint_info(
                f"✅ Loaded market data from {market_source}: {market_data.shape} "
                f"({market_data.index.min()} → {market_data.index.max()})"
            )
            profile_enabled = bool(config.get("map_profile_features", False))
            feature_profile: Dict[str, float] = {}
            structure_df = self._generate_map_features(
                market_data,
                config,
                feature_profile if profile_enabled else None,
            )
            structure_df = self._compute_map_alpha_signals(structure_df, config)

            if bool(config.get("map_v2_enable_unsup", False)):
                try:
                    structure_df, map_v2_metrics = self._compute_map_v2_unsupervised_state(
                        structure_df,
                        config,
                    )
                    if map_v2_metrics:
                        metrics.update(map_v2_metrics)
                except Exception as unsup_exc:
                    tprint_warning(
                        f"Map v2 unsupervised state computation failed (non-fatal): {unsup_exc}"
                    )

            if bool(config.get("map_xgb_enable_training", False)):
                try:
                    xgb_metrics, xgb_artifacts = self._train_map_xgb_model(
                        structure_df,
                        config,
                    )
                    if xgb_metrics:
                        metrics.update(xgb_metrics)
                    if xgb_artifacts:
                        artifacts.extend(xgb_artifacts)
                except Exception as xgb_exc:
                    tprint_warning(
                        f"Map XGB training failed (non-fatal): {xgb_exc}"
                    )

            # Robustly construct a timestamp column without clashing with existing columns
            idx_name = structure_df.index.name or "index"
            if idx_name in structure_df.columns:
                # Index name already exists as a column (e.g. 'open_time'); keep columns
                # and add a dedicated 'timestamp' column from the index values.
                to_save = structure_df.copy()
                if "timestamp" not in to_save.columns:
                    to_save.insert(0, "timestamp", structure_df.index.to_numpy())
            else:
                to_save = structure_df.reset_index().rename(columns={idx_name: "timestamp"})
            map_features_path = self._save_artifact(
                data=to_save,
                artifact_name="ml_map_structure_features_15m",
                artifact_type="data",
                data_category="features",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "direction": direction,
                    "source_market_data": market_source,
                },
            )
            artifacts.append(map_features_path)
            hvn_series = structure_df.get("map_is_hvn") if "map_is_hvn" in structure_df.columns else None
            lvn_series = structure_df.get("map_is_lvn") if "map_is_lvn" in structure_df.columns else None
            fast_long_series = (
                structure_df.get("map_alpha_fast_profit_long")
                if "map_alpha_fast_profit_long" in structure_df.columns
                else None
            )
            trap_long_series = (
                structure_df.get("map_alpha_trap_grind_long")
                if "map_alpha_trap_grind_long" in structure_df.columns
                else None
            )
            fast_short_series = (
                structure_df.get("map_alpha_fast_profit_short")
                if "map_alpha_fast_profit_short" in structure_df.columns
                else None
            )
            trap_short_series = (
                structure_df.get("map_alpha_trap_grind_short")
                if "map_alpha_trap_grind_short" in structure_df.columns
                else None
            )

            hvn_frac = float(hvn_series.mean()) if hvn_series is not None else 0.0
            lvn_frac = float(lvn_series.mean()) if lvn_series is not None else 0.0
            fast_long_frac = float(fast_long_series.mean()) if fast_long_series is not None else 0.0
            trap_long_frac = float(trap_long_series.mean()) if trap_long_series is not None else 0.0
            fast_short_frac = float(fast_short_series.mean()) if fast_short_series is not None else 0.0
            trap_short_frac = float(trap_short_series.mean()) if trap_short_series is not None else 0.0
            metrics.update(
                {
                    "hvn_fraction": hvn_frac,
                    "lvn_fraction": lvn_frac,
                    "alpha_fast_profit_long_fraction": fast_long_frac,
                    "alpha_trap_grind_long_fraction": trap_long_frac,
                    "alpha_fast_profit_short_fraction": fast_short_frac,
                    "alpha_trap_grind_short_fraction": trap_short_frac,
                    "n_samples": int(len(structure_df)),
                }
            )

            # Optional Map regime WCoV diagnostics and report
            regime_col_cfg = str(config.get("map_report_regime_col", "auto"))
            if regime_col_cfg == "auto":
                if "map_xgb_regime" in structure_df.columns:
                    regime_col = "map_xgb_regime"
                elif bool(config.get("map_xgb_enable_training", False)):
                    raise RuntimeError(
                        "Map XGB reporting is in 'auto' mode with map_xgb_enable_training=True, "
                        "but 'map_xgb_regime' is missing from structure_df. XGB training likely "
                        "failed before attaching regimes."
                    )
                else:
                    regime_col = "map_trend_state"
            else:
                regime_col = regime_col_cfg

            try:
                if regime_col in structure_df.columns:
                    wcov_metrics = self._generate_map_regime_reports(
                        structure_df,
                        regime_col=regime_col,
                        symbol=symbol,
                        exchange=exchange,
                        regime_timeframe=regime_timeframe,
                    )
                    if wcov_metrics:
                        metrics.update(wcov_metrics)
                else:
                    tprint_warning(
                        f"Map regime report skipped: regime column '{regime_col}' not found in structure_df"
                    )
            except Exception as report_exc:  # pragma: no cover - defensive
                tprint_warning(f"Map regime report generation failed (non-fatal): {report_exc}")
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            if profile_enabled and feature_profile:
                metrics.update(feature_profile)
            tprint_info(
                f"✅ {self.step_name} (Market Structure Map) completed in {execution_time:.2f}s "
                f"with {len(structure_df)} samples"
            )
            return {
                "success": True,
                "artifacts": artifacts,
                "metrics": metrics,
                "error": None,
                "execution_time": execution_time,
            }
        except Exception as exc:
            execution_time = time.time() - start_time
            tprint_error(f"❌ {self.step_name} failed: {exc}")
            return {
                "success": False,
                "artifacts": artifacts,
                "metrics": metrics,
                "error": str(exc),
                "execution_time": execution_time,
            }

    def _generate_map_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        profiling: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        result = df.copy()
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in result.columns]
        if missing:
            raise ValueError(f"Missing columns for market structure features: {missing}")

        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)
        result = result.sort_index()

        o = result["open"].astype(float)
        h = result["high"].astype(float)
        l = result["low"].astype(float)
        c = result["close"].astype(float)
        v = result["volume"].astype(float)

        # ------------------------------------------------------------------
        # Core price/volatility features (1h returns, ATR, wick ratios)
        # ------------------------------------------------------------------
        if profiling is not None:
            t0 = time.perf_counter()

        result["returns_1h"] = np.log(c / c.shift(4))

        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_window = int(config.get("map_atr_window", 56))
        atr = true_range.rolling(window=atr_window).mean()
        result["true_range"] = true_range
        result["atr_14"] = atr

        # ADX trend strength (Wilder-style approximation)
        try:
            adx_period = int(config.get("map_adx_period", 56))
            if adx_period > 1 and len(result) > adx_period + 2:
                high_vals = h.values.astype(float)
                low_vals = l.values.astype(float)
                close_vals = c.values.astype(float)

                up_move = high_vals[1:] - high_vals[:-1]
                down_move = low_vals[:-1] - low_vals[1:]

                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

                plus_dm_series = pd.Series(
                    np.concatenate([[np.nan], plus_dm]), index=result.index
                )
                minus_dm_series = pd.Series(
                    np.concatenate([[np.nan], minus_dm]), index=result.index
                )
                tr_series = true_range.astype(float)

                tr_smooth = tr_series.rolling(adx_period).sum()
                plus_dm_smooth = plus_dm_series.rolling(adx_period).sum()
                minus_dm_smooth = minus_dm_series.rolling(adx_period).sum()

                plus_di = 100.0 * (plus_dm_smooth / (tr_smooth + 1e-9))
                minus_di = 100.0 * (minus_dm_smooth / (tr_smooth + 1e-9))
                dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100.0
                adx = dx.rolling(adx_period).mean()

                regime_tf = str(config.get("regime_timeframe", config.get("timeframe", "15m")))
                adx_col_name = f"adx_{regime_tf}"
                result[adx_col_name] = adx.astype(float)
        except Exception as adx_exc:
            tprint_warning(f"Map: ADX computation failed, skipping: {adx_exc}")

        if profiling is not None:
            profiling["map_profile_atr_returns_s"] = time.perf_counter() - t0

        body = (c - o).abs()
        upper_wick = h - np.maximum(c, o)
        lower_wick = np.minimum(c, o) - l
        result["upper_wick_ratio"] = upper_wick / (body + 1e-9)
        result["lower_wick_ratio"] = lower_wick / (body + 1e-9)
        result["total_wick_ratio"] = (upper_wick + lower_wick) / (body + 1e-9)

        # ------------------------------------------------------------------
        # Volume-by-Price profile over last N days (HVN/LVN zones)
        # ------------------------------------------------------------------
        if profiling is not None:
            t1 = time.perf_counter()

        lookback_days = float(config.get("map_vbp_lookback_days", 30.0))
        if len(result.index) > 0:
            cutoff = result.index.max() - pd.Timedelta(days=lookback_days)
        else:
            cutoff = result.index.min()

        mask_vbp = result.index >= cutoff
        tp_all = (h + l + c) / 3.0
        tp_vbp = tp_all[mask_vbp]
        v_vbp = v[mask_vbp]

        if len(tp_vbp) > 0:
            price_min = float(tp_vbp.min())
            price_max = float(tp_vbp.max())
            if (not np.isfinite(price_min)) or (not np.isfinite(price_max)) or price_min == price_max:
                price_min = float(c.min()) * 0.999
                price_max = float(c.max()) * 1.001

            n_bins = int(config.get("map_vbp_n_bins", 60))
            price_edges = np.linspace(price_min, price_max, n_bins + 1)
            price_centers = 0.5 * (price_edges[:-1] + price_edges[1:])

            bin_idx_vbp = np.searchsorted(price_edges, tp_vbp.values, side="right") - 1
            bin_idx_vbp = np.clip(bin_idx_vbp, 0, n_bins - 1)

            profile_vol = np.bincount(
                bin_idx_vbp,
                weights=v_vbp.values.astype(float),
                minlength=n_bins,
            )

            mean_v = profile_vol.mean()
            std_v = profile_vol.std()
            if std_v <= 0.0:
                z_bins = np.zeros_like(profile_vol, dtype=float)
            else:
                z_bins = (profile_vol - mean_v) / (std_v + 1e-9)

            z_all = np.full(len(result), np.nan, dtype=float)
            price_center_all = np.full(len(result), np.nan, dtype=float)
            mask_arr = np.asarray(mask_vbp, dtype=bool)
            z_all[mask_arr] = z_bins[bin_idx_vbp]
            price_center_all[mask_arr] = price_centers[bin_idx_vbp]

            result["map_volume_profile_z"] = z_all
            result["map_vbp_price_center"] = price_center_all

            hvn_z = float(config.get("map_hvn_z", 1.0))
            lvn_z = float(config.get("map_lvn_z", -1.0))
            finite_mask = np.isfinite(z_all)
            result["map_is_hvn"] = ((z_all >= hvn_z) & finite_mask).astype(int)
            result["map_is_lvn"] = ((z_all <= lvn_z) & finite_mask).astype(int)
        else:
            result["map_volume_profile_z"] = np.nan
            result["map_vbp_price_center"] = np.nan
            result["map_is_hvn"] = 0
            result["map_is_lvn"] = 0

        if profiling is not None:
            profiling["map_profile_volume_profile_s"] = time.perf_counter() - t1

        # ------------------------------------------------------------------
        # Fractal swing highs/lows and distance to nearest pivot
        # ------------------------------------------------------------------
        sh = (
            (h.shift(2) < h.shift(1))
            & (h.shift(1) < h)
            & (h > h.shift(-1))
            & (h.shift(-1) > h.shift(-2))
        )
        sl = (
            (l.shift(2) > l.shift(1))
            & (l.shift(1) > l)
            & (l < l.shift(-1))
            & (l.shift(-1) < l.shift(-2))
        )

        result["map_swing_high"] = sh.astype(int)
        result["map_swing_low"] = sl.astype(int)

        pivot_prices = pd.concat([h[sh], l[sl]]).dropna()
        if len(pivot_prices) > 0:
            piv = np.sort(pivot_prices.values.astype(float))
            closes = c.values.astype(float)
            pos = np.searchsorted(piv, closes)
            left = np.where(pos > 0, piv[np.clip(pos - 1, 0, len(piv) - 1)], np.nan)
            right = np.where(pos < len(piv), piv[np.clip(pos, 0, len(piv) - 1)], np.nan)
            dist_left = np.abs(closes - left)
            dist_right = np.abs(closes - right)
            dist = np.where(
                np.isnan(dist_left),
                dist_right,
                np.where(np.isnan(dist_right), dist_left, np.minimum(dist_left, dist_right)),
            )
            result["map_distance_to_nearest_pivot"] = dist
            result["map_distance_to_nearest_pivot_atr"] = dist / (atr.values + 1e-9)
        else:
            result["map_distance_to_nearest_pivot"] = np.nan
            result["map_distance_to_nearest_pivot_atr"] = np.nan

        # ------------------------------------------------------------------
        # Fair Value Gap (FVG) / imbalance
        # ------------------------------------------------------------------
        prev_high = h.shift(1)
        prev_low = l.shift(1)
        fvg_up = c > prev_high
        fvg_down = c < prev_low
        gap_up = (c - prev_high).clip(lower=0.0)
        gap_down = (prev_low - c).clip(lower=0.0)
        fvg_size = gap_up.where(fvg_up, 0.0) + gap_down.where(fvg_down, 0.0)
        result["map_fvg_up"] = fvg_up.astype(int)
        result["map_fvg_down"] = fvg_down.astype(int)
        result["map_fvg_size_atr"] = fvg_size / (atr + 1e-9)

        # ------------------------------------------------------------------
        # Swing Failure Pattern (SFP) using most recent swing high/low
        # ------------------------------------------------------------------
        tol = float(config.get("map_sfp_tolerance", 0.0005))
        last_sh = h.where(sh).shift(1).ffill()
        last_sl = l.where(sl).shift(1).ffill()
        sfp_short = last_sh.notna() & (h > last_sh * (1.0 + tol)) & (c < last_sh)
        sfp_long = last_sl.notna() & (l < last_sl * (1.0 - tol)) & (c > last_sl)
        result["map_sfp_short"] = sfp_short.astype(int)
        result["map_sfp_long"] = sfp_long.astype(int)

        # ------------------------------------------------------------------
        # Previous day high/low proximity (PDH/PDL) and anchored VWAP
        # ------------------------------------------------------------------
        if profiling is not None:
            t2 = time.perf_counter()

        day_index = result.index.normalize()
        daily = result.groupby(day_index).agg(high=("high", "max"), low=("low", "min"))
        prev_day_high = daily["high"].shift(1)
        prev_day_low = daily["low"].shift(1)
        pdh = prev_day_high.reindex(day_index).to_numpy()
        pdl = prev_day_low.reindex(day_index).to_numpy()
        result["map_pdh"] = pdh
        result["map_pdl"] = pdl
        result["map_dist_to_pdh"] = (c.values - pdh) / (atr.values + 1e-9)
        result["map_dist_to_pdl"] = (c.values - pdl) / (atr.values + 1e-9)

        # Positional continuum within previous day's range: 0 at PDH, 1 at PDL
        denom = pdl - pdh
        pos = np.full_like(c.values, np.nan, dtype=float)
        valid_pos_mask = np.isfinite(pdh) & np.isfinite(pdl) & (np.abs(denom) > 0)
        pos[valid_pos_mask] = (c.values[valid_pos_mask] - pdh[valid_pos_mask]) / (
            denom[valid_pos_mask] + 1e-9
        )
        result["map_pdh_pdl_position"] = pos

        tp = tp_all
        pv = tp * v

        day = day_index
        cum_pv_day = pv.groupby(day).cumsum()
        cum_v_day = v.groupby(day).cumsum()
        vwap_day = cum_pv_day / (cum_v_day + 1e-9)
        result["map_vwap_day"] = vwap_day
        result["map_price_vs_vwap_day"] = (c - vwap_day) / (atr + 1e-9)

        week = result.index.to_period("W").to_timestamp()
        cum_pv_week = pv.groupby(week).cumsum()
        cum_v_week = v.groupby(week).cumsum()
        vwap_week = cum_pv_week / (cum_v_week + 1e-9)
        result["map_vwap_week"] = vwap_week
        result["map_price_vs_vwap_week"] = (c - vwap_week) / (atr + 1e-9)

        if profiling is not None:
            profiling["map_profile_groupby_vwap_s"] = time.perf_counter() - t2

        return result

    def _compute_map_alpha_signals(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        c = result["close"].astype(float)
        atr = result["atr_14"] if "atr_14" in result.columns else None
        if atr is None:
            tr = (result["high"] - result["low"]).abs()
            atr = tr.rolling(window=56).mean()
            result["atr_14"] = atr
        fast_win = int(config.get("map_trend_fast_window", 24))
        slow_win = int(config.get("map_trend_slow_window", 96))
        ma_fast = c.rolling(fast_win).mean()
        ma_slow = c.rolling(slow_win).mean()
        eps = 1e-9
        trend_state = np.where(
            ma_fast > ma_slow * (1.0 + eps),
            1,
            np.where(ma_fast < ma_slow * (1.0 - eps), -1, 0),
        )
        result["map_trend_state"] = trend_state
        hvn_flag = result["map_is_hvn"].astype(bool).values if "map_is_hvn" in result.columns else np.zeros(len(result), dtype=bool)
        lvn_flag = result["map_is_lvn"].astype(bool).values if "map_is_lvn" in result.columns else np.zeros(len(result), dtype=bool)

        trend_up = trend_state > 0
        trend_down = trend_state < 0

        # Long-side fast profit: trend up, trading in low-volume pocket (LVN)
        fast_profit_long = trend_up & lvn_flag
        result["map_alpha_fast_profit_long"] = fast_profit_long.astype(int)

        # Short-side fast profit: trend down, trading in low-volume pocket (LVN)
        fast_profit_short = trend_down & lvn_flag
        result["map_alpha_fast_profit_short"] = fast_profit_short.astype(int)
        if "map_vbp_price_center" in result.columns:
            centers = result["map_vbp_price_center"].values.astype(float)
            atr_vals = atr.values if isinstance(atr, pd.Series) else np.asarray(atr)
            trap_dist_atr = float(config.get("map_trap_distance_atr", 1.0))

            # Long-side trap/grind: uptrend into HVN from below
            below_hvn = hvn_flag & (c.values <= centers)
            close_enough = np.abs(centers - c.values) <= (atr_vals * trap_dist_atr + eps)
            trap_long = trend_up & below_hvn & close_enough
            result["map_alpha_trap_grind_long"] = trap_long.astype(int)

            # Short-side trap/grind: downtrend into HVN from above
            above_hvn = hvn_flag & (c.values >= centers)
            trap_short = trend_down & above_hvn & close_enough
            result["map_alpha_trap_grind_short"] = trap_short.astype(int)
        else:
            result["map_alpha_trap_grind_long"] = 0
            result["map_alpha_trap_grind_short"] = 0
        return result

    def _compute_map_v2_unsupervised_state(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        feature_candidates = [
            "map_distance_to_nearest_pivot_atr",
            "map_price_vs_vwap_day",
            "map_price_vs_vwap_week",
            "map_dist_to_pdh",
            "map_dist_to_pdl",
            "map_fvg_size_atr",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "total_wick_ratio",
            "map_volume_profile_z",
            "map_is_hvn",
            "map_is_lvn",
            "returns_1h",
            "map_trend_state",
        ]

        available_cols = [c for c in feature_candidates if c in df.columns]
        if len(available_cols) < 4:
            return df, {}

        feature_df = df[available_cols].astype(float)

        lower_q = float(config.get("map_v2_winsor_lower_q", 0.01))
        upper_q = float(config.get("map_v2_winsor_upper_q", 0.99))
        try:
            feature_df_norm = winsorized_zscore_normalize(
                feature_df,
                lower_quantile=lower_q,
                upper_quantile=upper_q,
            )
        except Exception as norm_exc:
            tprint_warning(f"Map v2 normalization failed, skipping unsupervised state: {norm_exc}")
            return df, {}

        if isinstance(feature_df_norm, pd.Series):
            feature_df_norm = feature_df_norm.to_frame()

        features_clean = feature_df_norm.dropna()
        min_samples = int(config.get("map_v2_min_samples", 200))
        if len(features_clean) < min_samples:
            return df, {}

        frac = float(config.get("map_v2_unsup_subsample_frac", 0.4))
        max_rows = int(config.get("map_v2_unsup_max_rows", 20000))
        n_target = min(max_rows, max(int(len(features_clean) * frac), 0))
        if n_target <= 0 or n_target > len(features_clean):
            n_target = len(features_clean)

        if n_target < min_samples:
            n_target = len(features_clean)

        sample_index = features_clean.index
        if n_target < len(features_clean):
            sample_index = (
                features_clean.index.to_series()
                .sample(n=n_target, random_state=42)
                .sort_index()
            )

        try:
            import umap  # type: ignore
        except ImportError:
            tprint_warning("UMAP not installed, skipping Map v2 unsupervised state")
            return df, {}

        n_components_cfg = int(config.get("map_v2_umap_components", 3))
        n_neighbors = int(config.get("map_v2_umap_n_neighbors", 64))
        min_dist = float(config.get("map_v2_umap_min_dist", 0.05))
        metric = str(config.get("map_v2_umap_metric", "euclidean"))

        max_components = max(1, features_clean.shape[1] - 2)
        n_components = min(n_components_cfg, max_components)

        try:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=42,
                verbose=False,
            )
        except Exception as umap_init_exc:
            tprint_warning(f"Map v2 UMAP initialization failed, skipping: {umap_init_exc}")
            return df, {}

        try:
            sample_X = features_clean.loc[sample_index].values
            if hasattr(reducer, "transform"):
                reducer.fit(sample_X)
                embedding_full = reducer.transform(features_clean.values)
            else:
                embedding_full = reducer.fit_transform(features_clean.values)
        except Exception as umap_exc:
            tprint_warning(f"Map v2 UMAP failed, skipping unsupervised state: {umap_exc}")
            return df, {}

        Z_partial = pd.DataFrame(
            embedding_full,
            index=features_clean.index,
            columns=[f"map_v2_umap_{i + 1}" for i in range(embedding_full.shape[1])],
        )
        Z_df = pd.DataFrame(
            index=df.index,
            columns=Z_partial.columns,
            dtype=float,
        )
        Z_df.loc[Z_partial.index, :] = Z_partial

        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            tprint_warning("sklearn.mixture.GaussianMixture not available, skipping Map v2 GMM")
            out_df = df.copy().join(Z_df)
            return out_df, {}

        Z_for_gmm = Z_partial
        if len(Z_for_gmm) < min_samples:
            tprint_warning("Insufficient samples for Map v2 GMM, skipping")
            out_df = df.copy().join(Z_df)
            return out_df, {}

        n_regimes = int(config.get("map_v2_gmm_n_regimes", 4))
        covariance_type = str(config.get("map_v2_gmm_covariance_type", "full"))
        n_init = int(config.get("map_v2_gmm_n_init", 3))
        reg_covar = float(config.get("map_v2_gmm_reg_covar", 1e-6))

        try:
            gmm = GaussianMixture(
                n_components=n_regimes,
                covariance_type=covariance_type,
                n_init=n_init,
                random_state=42,
                reg_covar=reg_covar,
            )
            gmm.fit(Z_for_gmm.values)
            probs_partial = gmm.predict_proba(Z_partial.values)
        except Exception as gmm_exc:
            tprint_warning(f"Map v2 GMM failed, skipping unsupervised state: {gmm_exc}")
            out_df = df.copy().join(Z_df)
            return out_df, {}

        balance_soft = bool(config.get("map_v2_balance_soft_assign", True))
        if balance_soft and probs_partial.shape[1] == n_regimes:
            try:
                global_probs = probs_partial.mean(axis=0)
                min_global = float(config.get("map_v2_balance_min_global_prob", 0.01))
                global_probs_clipped = np.clip(global_probs, min_global, None)
                reweight = 1.0 / (global_probs_clipped + 1e-8)
                balanced_probs = probs_partial * reweight[None, :]
                row_sums = balanced_probs.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
                probs_partial = balanced_probs / row_sums
            except Exception as balance_exc:
                tprint_warning(f"Map v2 GMM soft-assignment balancing failed, using raw probabilities: {balance_exc}")

        labels_partial = probs_partial.argmax(axis=1)

        regime_series = pd.Series(-1, index=df.index, dtype=int)
        regime_series.loc[Z_partial.index] = labels_partial.astype(int)

        prob_cols = [f"map_v2_regime_p{k}" for k in range(n_regimes)]
        probs_df = pd.DataFrame(0.0, index=df.index, columns=prob_cols, dtype=float)
        probs_partial_df = pd.DataFrame(
            probs_partial,
            index=Z_partial.index,
            columns=prob_cols,
        )
        probs_df.loc[Z_partial.index, :] = probs_partial_df

        out_df = df.copy()
        out_df = out_df.join(Z_df)
        out_df["map_v2_regime"] = regime_series
        out_df = out_df.join(probs_df)

        metrics: Dict[str, Any] = {}
        valid_mask = regime_series >= 0
        if valid_mask.any():
            labels_valid = regime_series[valid_mask].to_numpy()
            unique, counts = np.unique(labels_valid, return_counts=True)
            total = float(counts.sum())
            for ridx, cnt in zip(unique, counts):
                key = f"map_v2_regime_{int(ridx)}_fraction"
                metrics[key] = float(cnt) / total if total > 0 else 0.0

        return out_df, metrics

    def _train_map_xgb_model(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        metrics: Dict[str, Any] = {}
        artifacts: List[str] = []

        try:
            import xgboost as xgb  # type: ignore
        except Exception:
            tprint_warning("xgboost not installed, skipping Map XGB training")
            metrics["map_xgb_early_exit_reason"] = "xgboost_not_installed"
            return metrics, artifacts

        if "close" not in df.columns:
            tprint_warning("Map XGB: 'close' column missing; skipping training")
            metrics["map_xgb_early_exit_reason"] = "missing_close_column"
            return metrics, artifacts

        c = df["close"].astype(float)
        if "atr_14" in df.columns:
            atr = df["atr_14"].astype(float)
        else:
            tr = (df["high"] - df["low"]).abs()
            atr = tr.rolling(window=14).mean()

        horizon = int(config.get("map_xgb_target_horizon_bars", 4))
        if horizon <= 0 or horizon >= len(c):
            tprint_warning(
                f"Map XGB: invalid target horizon (horizon={horizon}, len={len(c)})"
            )
            metrics["map_xgb_early_exit_reason"] = "invalid_horizon"
            return metrics, artifacts

        fwd_log_ret = np.log(c.shift(-horizon) / c)

        use_adx_direction = bool(config.get("map_xgb_use_adx_direction", False))

        if use_adx_direction:
            # Resolve ADX column to use
            adx_col_cfg = str(config.get("map_xgb_adx_column", "")).strip()
            adx_col = adx_col_cfg if adx_col_cfg else ""

            if not adx_col:
                tf = str(config.get("regime_timeframe", config.get("timeframe", "")))
                if tf:
                    candidate = f"adx_{tf}"
                    if candidate in df.columns:
                        adx_col = candidate

            if not adx_col:
                # Fallback: first adx_* column if any
                for col in df.columns:
                    if col.startswith("adx_"):
                        adx_col = col
                        break

            if adx_col and adx_col in df.columns:
                adx = df[adx_col].astype(float)
                adx_thr = float(config.get("map_xgb_adx_trend_threshold", 20.0))
                ret_thr = float(config.get("map_xgb_label_threshold_ret", 0.0))

                y = np.full(len(df), -1, dtype=int)
                directional_mask = (adx >= adx_thr) & np.isfinite(fwd_log_ret)
                y[directional_mask & (fwd_log_ret > ret_thr)] = 2
                y[directional_mask & (fwd_log_ret < -ret_thr)] = 0

                neutral_mask = (y == -1) & np.isfinite(fwd_log_ret)
                y[neutral_mask] = 1

                tprint_info(
                    "Map XGB: using ADX-based directional labeling "
                    f"(horizon={horizon}, adx_col='{adx_col}', adx_thr={adx_thr}, ret_thr={ret_thr})"
                )
            else:
                tprint_warning(
                    "Map XGB: map_xgb_use_adx_direction=True but no suitable ADX column "
                    "found; falling back to ATR-based labeling"
                )
                use_adx_direction = False

        if not use_adx_direction:
            fwd_norm = fwd_log_ret / (atr + 1e-9)
            threshold = float(config.get("map_xgb_label_threshold_atr", 0.3))

            y = np.full(len(df), -1, dtype=int)
            y[fwd_norm > threshold] = 2
            y[fwd_norm < -threshold] = 0
            neutral_mask = (np.abs(fwd_norm) <= threshold) & np.isfinite(fwd_norm)
            y[neutral_mask] = 1

        valid_mask = y >= 0
        min_samples = int(config.get("map_xgb_min_samples", 800))
        n_valid = int(valid_mask.sum())
        if n_valid < min_samples:
            tprint_warning(
                f"Map XGB: insufficient labeled samples for training (valid={n_valid}, min={min_samples})"
            )
            metrics["map_xgb_early_exit_reason"] = f"insufficient_samples_valid_{n_valid}_min_{min_samples}"
            return metrics, artifacts

        df_clean = df.loc[valid_mask].copy()
        y_clean = y[valid_mask]

        # Global class-diversity check: require enough non-neutral samples
        min_per_class = int(config.get("map_xgb_min_per_class", 50))
        try:
            classes_all, counts_all = np.unique(y_clean, return_counts=True)
            class_counts = {int(c): int(cnt) for c, cnt in zip(classes_all, counts_all)}
        except Exception:
            class_counts = {}
        non_neutral_classes = [0, 2]
        shortages = {c: int(class_counts.get(c, 0)) for c in non_neutral_classes}
        if any(shortages[c] < min_per_class for c in non_neutral_classes):
            tprint_warning(
                "Map XGB: insufficient global class diversity for training "
                f"(min_per_class={min_per_class}, non_neutral_counts={shortages})"
            )
            metrics["map_xgb_early_exit_reason"] = (
                f"insufficient_class_diversity_global_min_{min_per_class}_"
                f"counts_{shortages}"
            )
            return metrics, artifacts

        numeric_df = df_clean.select_dtypes(include=[np.number])
        if numeric_df.empty:
            tprint_warning("Map XGB: no numeric features available after cleaning; skipping training")
            metrics["map_xgb_early_exit_reason"] = "no_numeric_features_after_cleaning"
            return metrics, artifacts

        exclude_prefixes = [
            "map_xgb_",
        ]
        feature_cols = [
            col
            for col in numeric_df.columns
            if not any(col.startswith(pfx) for pfx in exclude_prefixes)
        ]
        if len(feature_cols) < 4:
            tprint_warning(
                f"Map XGB: too few numeric feature columns after filtering (n_features={len(feature_cols)})"
            )
            metrics["map_xgb_early_exit_reason"] = f"too_few_features_{len(feature_cols)}"
            return metrics, artifacts

        max_feats = int(config.get("map_xgb_max_features", 40))
        if max_feats > 0 and len(feature_cols) > max_feats:
            feature_cols = feature_cols[:max_feats]

        X_raw = numeric_df[feature_cols]

        try:
            from src.features_common.transforms.scaling_normalization import (  # type: ignore
                ScalingNormalizer,
            )
        except Exception:
            X_scaled = X_raw.astype(np.float32)
        else:
            normalizer_config = {
                "default_strategy": "robust",
                "auto_select": False,
                "handle_outliers": True,
                "outlier_threshold": 3.0,
                "use_vectorbt": False,
            }
            scaler = ScalingNormalizer(normalizer_config)
            X_scaled = scaler.fit_transform(X_raw, strategy="robust")
            if hasattr(X_scaled, "astype"):
                X_scaled = X_scaled.astype("float32")

        if not hasattr(X_scaled, "index"):
            tprint_warning("Map XGB: scaled feature matrix has no index attribute; skipping training")
            metrics["map_xgb_early_exit_reason"] = "scaled_matrix_missing_index"
            return metrics, artifacts

        X_scaled = X_scaled.loc[df_clean.index]
        y_series = pd.Series(y_clean, index=df_clean.index)

        hpo_frac = float(config.get("map_xgb_hpo_subsample_frac", 0.25))
        hpo_max_rows = int(config.get("map_xgb_hpo_max_rows", 50000))
        n_total = len(df_clean)
        n_target = min(hpo_max_rows, max(int(n_total * hpo_frac), min_samples))
        if n_target >= n_total:
            hpo_index = df_clean.index
        else:
            hpo_index = (
                df_clean.index.to_series()
                .sample(n=n_target, random_state=42)
                .sort_index()
            )

        X_hpo = X_scaled.loc[hpo_index]
        y_hpo = y_series.loc[hpo_index].to_numpy()

        # HPO setup diagnostics and class-diversity check on the subsample
        try:
            unique_labels, label_counts = np.unique(y_hpo, return_counts=True)
            label_dist = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, label_counts)}
        except Exception:
            unique_labels, label_counts, label_dist = np.array([]), np.array([]), {}

        tprint_info(
            "Map XGB: starting HPO "
            f"(n_total={n_total}, n_hpo={len(X_hpo)}, n_features={len(feature_cols)}, "
            f"label_dist={label_dist})"
        )

        # Require at least one sample of each non-neutral class in the HPO subset
        shortages_hpo = {
            c: int(label_dist.get(c, 0)) for c in non_neutral_classes
        }
        if any(shortages_hpo[c] < 1 for c in non_neutral_classes):
            tprint_warning(
                "Map XGB: HPO subset lacks directional classes; skipping XGB training "
                f"(non_neutral_counts={shortages_hpo})"
            )
            metrics["map_xgb_early_exit_reason"] = (
                f"hpo_insufficient_class_diversity_counts_{shortages_hpo}"
            )
            return metrics, artifacts

        n_hpo = len(X_hpo)
        train_frac = float(config.get("map_xgb_train_fraction", 0.8))
        split_idx = max(1, min(n_hpo - 1, int(n_hpo * train_frac)))

        X_train = X_hpo.iloc[:split_idx]
        y_train = y_hpo[:split_idx]
        X_val = X_hpo.iloc[split_idx:]
        y_val = y_hpo[split_idx:]
        if len(X_val) < 100:
            tprint_warning(
                f"Map XGB: validation set too small for HPO (len_val={len(X_val)})"
            )
            metrics["map_xgb_early_exit_reason"] = f"val_too_small_{len(X_val)}"
            return metrics, artifacts

        core_cols = [
            "map_distance_to_nearest_pivot_atr",
            "map_fvg_size_atr",
            "map_volume_profile_z",
            "map_pdh_pdl_position",
        ]
        core_cols = [c for c in core_cols if c in df_clean.columns]
        if len(core_cols) < 2 or "returns_1h" not in df_clean.columns:
            tprint_warning(
                f"Map XGB: missing core features or returns_1h for WCoV objective "
                f"(n_core={len(core_cols)}, has_returns={bool('returns_1h' in df_clean.columns)})"
            )
            metrics["map_xgb_early_exit_reason"] = (
                f"missing_core_features_n_core_{len(core_cols)}_"
                f"has_returns_{bool('returns_1h' in df_clean.columns)}"
            )
            return metrics, artifacts

        core_df_all = df_clean[core_cols].astype(float)
        ret_all = df_clean["returns_1h"].astype(float)
        val_index = X_val.index
        core_val = core_df_all.loc[val_index]
        ret_val = ret_all.loc[val_index]

        n_classes = 3
        base_params = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "tree_method": "hist",
            "n_jobs": -1,
            "max_depth": 5,
            "min_child_weight": 20,
            "learning_rate": 0.05,
            "n_estimators": 600,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 1.0,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 40,
            "random_state": 42,
        }

        max_depth_candidates = [4, 6, 8]
        lr_candidates = [0.03, 0.05, 0.1]
        subsample_candidates = [0.7, 0.9]
        colsample_candidates = [0.7, 0.9]
        min_child_candidates = [5, 20]
        n_estimators_candidates = [300, 800]

        best_score = float("-inf")
        best_params = base_params.copy()

        max_trials = int(config.get("map_xgb_hpo_max_trials", 20))
        trial_count = 0

        for md in max_depth_candidates:
            for lr in lr_candidates:
                for ss in subsample_candidates:
                    for cs in colsample_candidates:
                        for mc in min_child_candidates:
                            for ne in n_estimators_candidates:
                                if trial_count >= max_trials:
                                    break
                                trial_count += 1
                                params = base_params.copy()
                                params["max_depth"] = int(md)
                                params["learning_rate"] = float(lr)
                                params["subsample"] = float(ss)
                                params["colsample_bytree"] = float(cs)
                                params["min_child_weight"] = int(mc)
                                params["n_estimators"] = int(ne)

                                try:
                                    model = xgb.XGBClassifier(**params)
                                    model.fit(
                                        X_train,
                                        y_train,
                                        eval_set=[(X_val, y_val)],
                                        verbose=False,
                                    )
                                    probs_val = model.predict_proba(X_val)
                                except Exception as train_exc:
                                    tprint_warning(
                                        "Map XGB: HPO trial failed during model fit/predict_proba "
                                        f"(trial={trial_count}, params={{'max_depth': {md}, 'lr': {lr}, "
                                        f"'subsample': {ss}, 'colsample_bytree': {cs}, "
                                        f"'min_child_weight': {mc}, 'n_estimators': {ne}}}): {train_exc}"
                                    )
                                    continue

                                if probs_val.shape[1] != n_classes:
                                    tprint_warning(
                                        "Map XGB: HPO trial produced unexpected prob shape "
                                        f"(trial={trial_count}, shape={probs_val.shape}, n_classes={n_classes})"
                                    )
                                    continue

                                model_labels = probs_val.argmax(axis=1)
                                labels_arr = model_labels.astype(int)

                                # Default fallback score; WCoV failures should not skip the trial.
                                score = 0.0

                                core_between_list: List[float] = []
                                core_within_list: List[float] = []
                                try:
                                    # Per-feature WCoV for core Map features
                                    for col in core_val.columns:
                                        feat_series = core_val[col].astype(float)
                                        bf = self._calculate_winsorized_cv_between(
                                            labels_arr,
                                            feat_series,
                                        )
                                        wf = self._calculate_winsorized_cv_within(
                                            labels_arr,
                                            feat_series,
                                        )
                                        core_between_list.append(float(bf))
                                        core_within_list.append(float(wf))

                                    if core_between_list:
                                        core_between = float(np.mean(core_between_list))
                                        core_within = float(np.mean(core_within_list))
                                    else:
                                        core_between = core_within = 0.0

                                    # Returns WCoV remains scalar
                                    ret_between = self._calculate_winsorized_cv_between(
                                        labels_arr,
                                        ret_val,
                                    )
                                    ret_within = self._calculate_winsorized_cv_within(
                                        labels_arr,
                                        ret_val,
                                    )
                                except Exception as wcov_exc:
                                    tprint_warning(
                                        f"Map XGB: WCoV computation failed in HPO trial: {wcov_exc}"
                                    )
                                    core_between_list = []
                                    core_within_list = []
                                    core_between = core_within = 0.0
                                    ret_between = ret_within = 0.0

                                try:
                                    # Aggregate per-feature WCoV contributions with log transforms
                                    core_terms: List[float] = []
                                    for bf, wf in zip(core_between_list, core_within_list):
                                        bf_clipped = max(float(bf), 0.0)
                                        bf_log = float(np.log1p(bf_clipped))
                                        wf_inv = 1.0 / (float(wf) + 1e-8)
                                        wf_inv_clipped = max(wf_inv, 0.0)
                                        wf_rev_log = float(np.log1p(wf_inv_clipped))
                                        feat_term = bf_log + wf_rev_log
                                        if np.isfinite(feat_term):
                                            core_terms.append(feat_term)

                                    if core_terms:
                                        core_term = float(np.mean(core_terms))
                                    else:
                                        core_term = 0.0

                                    # Returns term (single feature) with log-style transform
                                    ret_between_clipped = max(float(ret_between), 0.0)
                                    ret_between_log = float(np.log1p(ret_between_clipped))
                                    ret_inv_within = 1.0 / (float(ret_within) + 1e-8)
                                    ret_inv_within_clipped = max(ret_inv_within, 0.0)
                                    ret_within_reverse_log = float(np.log1p(ret_inv_within_clipped))
                                    ret_term = ret_between_log + ret_within_reverse_log

                                    wcov_term = core_term + 0.5 * ret_term
                                    ratio_cap = float(config.get("map_xgb_wcov_ratio_cap", 10.0))
                                    wcov_term_capped = float(min(wcov_term, ratio_cap))
                                except Exception as obj_exc:
                                    tprint_warning(
                                        f"Map XGB: WCoV objective computation failed in HPO trial: {obj_exc}"
                                    )
                                    wcov_term_capped = 0.0

                                if not np.isfinite(wcov_term_capped):
                                    tprint_warning(
                                        "Map XGB: non-finite WCoV objective in HPO trial "
                                        f"(core_between={core_between}, core_within={core_within}, "
                                        f"ret_between={ret_between}, ret_within={ret_within}, "
                                        f"wcov_term={wcov_term_capped})"
                                    )
                                    wcov_term_capped = 0.0

                                within_spread_penalty = 0.0
                                try:
                                    unique_regimes = [int(r) for r in np.unique(labels_arr) if r >= 0]
                                    per_regime_cvs: List[float] = []
                                    for rid in unique_regimes:
                                        regime_mask = labels_arr == rid
                                        regime_data = core_val.loc[regime_mask]
                                        if regime_data.shape[0] < 2:
                                            continue

                                        feature_cvs: List[float] = []
                                        for col in core_val.columns:
                                            col_data = regime_data[col].dropna()
                                            if len(col_data) > 1:
                                                lower_bound = col_data.quantile(0.05)
                                                upper_bound = col_data.quantile(0.95)
                                                col_winsorized = col_data.clip(
                                                    lower=lower_bound,
                                                    upper=upper_bound,
                                                )
                                                cv_val = col_winsorized.std() / (
                                                    np.abs(col_winsorized.mean()) + 1e-8
                                                )
                                                feature_cvs.append(float(cv_val))

                                        if feature_cvs:
                                            per_regime_cvs.append(float(np.mean(feature_cvs)))

                                    if len(per_regime_cvs) >= 2:
                                        spread = float(max(per_regime_cvs) - min(per_regime_cvs))
                                        spread_weight = float(
                                            config.get("map_xgb_within_spread_penalty_weight", 0.2)
                                        )
                                        within_spread_penalty = spread_weight * spread
                                except Exception:
                                    within_spread_penalty = 0.0

                                # Class-balance penalty based on regime frequency vector p_k
                                balance_penalty = 0.0
                                try:
                                    valid_labels = labels_arr[labels_arr >= 0]
                                    if valid_labels.size > 0:
                                        unique, counts = np.unique(valid_labels, return_counts=True)
                                        counts = counts.astype(float)
                                        p = counts / counts.sum()

                                        # Imbalance as RMSE to uniform distribution
                                        k = float(len(p))
                                        if k > 0:
                                            p_target = np.full_like(p, 1.0 / k, dtype=float)
                                            imbalance = float(
                                                np.sqrt(np.mean((p - p_target) ** 2))
                                            )

                                            balance_strength = float(
                                                config.get("map_xgb_balance_strength", 0.0)
                                            )
                                            min_regime_pct = float(
                                                config.get("map_xgb_min_regime_pct", 0.0)
                                            )

                                            balance_penalty = balance_strength * imbalance

                                            # Optional extra penalty if any regime is below the minimum
                                            if min_regime_pct > 0.0 and np.min(p) < min_regime_pct:
                                                balance_penalty *= 1.5
                                except Exception:
                                    balance_penalty = 0.0

                                score = float(wcov_term_capped - within_spread_penalty - balance_penalty)
                                if not np.isfinite(score):
                                    score = 0.0
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                            if trial_count >= max_trials:
                                break
                    if trial_count >= max_trials:
                        break
                if trial_count >= max_trials:
                    break
            if trial_count >= max_trials:
                break

        tprint_info(
            f"Map XGB: completed HPO loop with {trial_count} trials, best_score={best_score}"
        )

        if not np.isfinite(best_score):
            tprint_warning("Map XGB: HPO did not find a finite WCoV score; skipping XGB training")
            metrics["map_xgb_early_exit_reason"] = "hpo_no_finite_score"
            return metrics, artifacts

        feature_pruning_enabled = bool(config.get("map_xgb_enable_feature_pruning", False))
        final_feature_cols = list(feature_cols)

        if feature_pruning_enabled and len(feature_cols) > 4:
            try:
                current_features = list(feature_cols)
                X_train_base = X_train[current_features]
                X_val_base = X_val[current_features]

                base_model = xgb.XGBClassifier(**best_params)
                base_model.fit(
                    X_train_base,
                    y_train,
                    eval_set=[(X_val_base, y_val)],
                    verbose=False,
                )
                base_probs = base_model.predict_proba(X_val_base)
                base_labels = base_probs.argmax(axis=1).astype(int)

                best_prune_score, _, _, _ = self._compute_xgb_wcov_objective(
                    base_labels,
                    core_val,
                    ret_val,
                    config,
                )

                improved = True
                min_features = int(config.get("map_xgb_min_features", 8))
                if min_features < 2:
                    min_features = 2

                while improved and len(current_features) > min_features:
                    improved = False
                    try:
                        booster = base_model.get_booster()
                        gain_dict = booster.get_score(importance_type="gain")
                    except Exception:
                        break

                    feat_gains = []
                    for j, fname in enumerate(current_features):
                        key = f"f{j}"
                        gain_val = float(gain_dict.get(key, 0.0))
                        feat_gains.append((fname, gain_val))

                    if not feat_gains:
                        break

                    feat_gains.sort(key=lambda x: x[1])
                    max_candidates = int(config.get("map_xgb_feature_pruning_max_candidates", 5))
                    if max_candidates <= 0 or max_candidates > len(feat_gains):
                        max_candidates = len(feat_gains)
                    candidates = [name for name, _ in feat_gains[:max_candidates]]

                    for feat_to_drop in candidates:
                        if len(current_features) <= min_features:
                            break

                        trial_features = [f for f in current_features if f != feat_to_drop]
                        if not trial_features:
                            continue

                        X_train_trial = X_train[trial_features]
                        X_val_trial = X_val[trial_features]

                        try:
                            trial_model = xgb.XGBClassifier(**best_params)
                            trial_model.fit(
                                X_train_trial,
                                y_train,
                                eval_set=[(X_val_trial, y_val)],
                                verbose=False,
                            )
                            trial_probs = trial_model.predict_proba(X_val_trial)
                            trial_labels = trial_probs.argmax(axis=1).astype(int)

                            trial_score, _, _, _ = self._compute_xgb_wcov_objective(
                                trial_labels,
                                core_val,
                                ret_val,
                                config,
                            )
                        except Exception:
                            continue

                        if np.isfinite(trial_score) and trial_score > best_prune_score:
                            best_prune_score = float(trial_score)
                            current_features = trial_features
                            base_model = trial_model
                            improved = True
                            break

                final_feature_cols = current_features
                metrics["map_xgb_n_features_pruned"] = int(len(feature_cols) - len(final_feature_cols))
                metrics["map_xgb_feature_pruning_score"] = float(best_prune_score)
            except Exception as prune_exc:
                tprint_warning(f"Map XGB: feature pruning failed, using all features: {prune_exc}")
                final_feature_cols = list(feature_cols)

        X_full = X_scaled[final_feature_cols]
        y_full = y_series.to_numpy()
        n_full = len(X_full)
        split_full = max(1, min(n_full - 1, int(n_full * train_frac)))
        X_train_full = X_full.iloc[:split_full]
        y_train_full = y_full[:split_full]
        X_val_full = X_full.iloc[split_full:]
        y_val_full = y_full[split_full:]
        if len(X_val_full) == 0:
            tprint_warning("Map XGB: no validation samples in full-sample split; skipping XGB training")
            metrics["map_xgb_early_exit_reason"] = "no_val_samples_full_split"
            return metrics, artifacts

        temperature = 1.0
        try:
            final_model = xgb.XGBClassifier(**best_params)
            final_model.fit(
                X_train_full,
                y_train_full,
                eval_set=[(X_val_full, y_val_full)],
                verbose=False,
            )
            final_probs_val = final_model.predict_proba(X_val_full)
        except Exception as final_exc:
            tprint_warning(f"Map XGB: final model training failed, skipping: {final_exc}")
            metrics["map_xgb_early_exit_reason"] = "final_model_training_failed"
            return metrics, artifacts

        try:
            temperature = self._fit_temperature_scaling(final_probs_val, y_val_full)
        except Exception:
            temperature = 1.0

        val_pred = final_probs_val.argmax(axis=1)
        val_acc = float((val_pred == y_val_full).mean())

        core_full = df_clean[core_cols].astype(float).loc[X_val_full.index]
        ret_full = df_clean["returns_1h"].astype(float).loc[X_val_full.index]

        try:
            core_between_full = self._calculate_winsorized_cv_between(
                val_pred.astype(int),
                core_full,
            )
            core_within_full = self._calculate_winsorized_cv_within(
                val_pred.astype(int),
                core_full,
            )
            core_ratio_full = core_between_full / (core_within_full + 1e-8)

            ret_between_full = self._calculate_winsorized_cv_between(
                val_pred.astype(int),
                ret_full,
            )
            ret_within_full = self._calculate_winsorized_cv_within(
                val_pred.astype(int),
                ret_full,
            )
            ret_ratio_full = ret_between_full / (ret_within_full + 1e-8)
        except Exception:
            core_between_full = core_within_full = core_ratio_full = 0.0
            ret_between_full = ret_within_full = ret_ratio_full = 0.0

        metrics["map_xgb_hpo_best_score"] = float(best_score)
        metrics["map_xgb_val_accuracy"] = float(val_acc)
        metrics["map_xgb_temperature"] = float(temperature)
        metrics["map_xgb_n_features"] = int(len(final_feature_cols))
        metrics["map_xgb_n_samples"] = int(n_full)
        metrics["map_xgb_core_wcov_between"] = float(core_between_full)
        metrics["map_xgb_core_wcov_within"] = float(core_within_full)
        metrics["map_xgb_core_wcov_ratio"] = float(core_ratio_full)
        metrics["map_xgb_returns_wcov_between"] = float(ret_between_full)
        metrics["map_xgb_returns_wcov_within"] = float(ret_within_full)
        metrics["map_xgb_returns_wcov_ratio"] = float(ret_ratio_full)

        metrics["map_xgb_best_max_depth"] = int(best_params.get("max_depth", 0))
        metrics["map_xgb_best_learning_rate"] = float(best_params.get("learning_rate", 0.0))
        metrics["map_xgb_best_subsample"] = float(best_params.get("subsample", 0.0))
        metrics["map_xgb_best_colsample_bytree"] = float(best_params.get("colsample_bytree", 0.0))
        metrics["map_xgb_best_min_child_weight"] = int(best_params.get("min_child_weight", 0))
        metrics["map_xgb_best_n_estimators"] = int(best_params.get("n_estimators", 0))

        # Attach XGB-implied regimes and probabilities to the main DataFrame for reporting
        try:
            full_probs_raw = final_model.predict_proba(X_full)
            if full_probs_raw.shape[1] == n_classes:
                full_probs = self._apply_temperature_scaling(full_probs_raw, temperature)
                full_labels = full_probs.argmax(axis=1).astype(int)

                regime_series = pd.Series(np.nan, index=df.index)
                regime_series.loc[df_clean.index] = full_labels
                df["map_xgb_regime"] = regime_series

                for k in range(n_classes):
                    col_name = f"map_xgb_regime_p{k}"
                    prob_series = pd.Series(np.nan, index=df.index)
                    prob_series.loc[df_clean.index] = full_probs[:, k]
                    df[col_name] = prob_series
        except Exception:
            pass

        return metrics, artifacts

    def _fit_temperature_scaling(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        if probs.size == 0 or y_true.size == 0:
            return 1.0
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1.0)
        logits = np.log(probs_clipped)
        nll_best = np.inf
        log_t_best = 0.0

        log_t_values = np.linspace(-2.0, 2.0, 25)
        indices = np.arange(len(y_true), dtype=int)
        y_int = y_true.astype(int)
        for log_t in log_t_values:
            t = float(np.exp(log_t))
            if not np.isfinite(t) or t <= 0.0:
                continue
            scaled_logits = logits / t
            scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits)
            sum_exp = exp_logits.sum(axis=1, keepdims=True)
            sum_exp = np.clip(sum_exp, eps, np.inf)
            probs_t = exp_logits / sum_exp
            p_true = probs_t[indices, y_int]
            p_true = np.clip(p_true, eps, 1.0)
            nll = float(-np.mean(np.log(p_true)))
            if np.isfinite(nll) and nll < nll_best:
                nll_best = nll
                log_t_best = log_t

        if not np.isfinite(nll_best):
            return 1.0
        return float(np.exp(log_t_best))

    def _apply_temperature_scaling(
        self,
        probs: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        if probs.size == 0:
            return probs
        if not np.isfinite(temperature) or temperature <= 0.0:
            return probs
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1.0)
        logits = np.log(probs_clipped)
        scaled_logits = logits / float(temperature)
        scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits)
        sum_exp = exp_logits.sum(axis=1, keepdims=True)
        sum_exp = np.clip(sum_exp, eps, np.inf)
        return exp_logits / sum_exp

    def _compute_xgb_wcov_objective(
        self,
        labels_arr: np.ndarray,
        core_val: pd.DataFrame,
        ret_val: pd.Series,
        config: Dict[str, Any],
    ) -> Tuple[float, float, float, float]:
        score = 0.0

        core_between_list: List[float] = []
        core_within_list: List[float] = []
        try:
            for col in core_val.columns:
                feat_series = core_val[col].astype(float)
                bf = self._calculate_winsorized_cv_between(
                    labels_arr,
                    feat_series,
                )
                wf = self._calculate_winsorized_cv_within(
                    labels_arr,
                    feat_series,
                )
                core_between_list.append(float(bf))
                core_within_list.append(float(wf))

            if core_between_list:
                core_between = float(np.mean(core_between_list))
                core_within = float(np.mean(core_within_list))
            else:
                core_between = core_within = 0.0

            ret_between = self._calculate_winsorized_cv_between(
                labels_arr,
                ret_val,
            )
            ret_within = self._calculate_winsorized_cv_within(
                labels_arr,
                ret_val,
            )
        except Exception as wcov_exc:
            tprint_warning(
                f"Map XGB: WCoV computation failed in objective evaluation: {wcov_exc}"
            )
            core_between_list = []
            core_within_list = []
            core_between = core_within = 0.0
            ret_between = ret_within = 0.0

        try:
            core_terms: List[float] = []
            for bf, wf in zip(core_between_list, core_within_list):
                bf_clipped = max(float(bf), 0.0)
                bf_log = float(np.log1p(bf_clipped))
                wf_inv = 1.0 / (float(wf) + 1e-8)
                wf_inv_clipped = max(wf_inv, 0.0)
                wf_rev_log = float(np.log1p(wf_inv_clipped))
                feat_term = bf_log + wf_rev_log
                if np.isfinite(feat_term):
                    core_terms.append(feat_term)

            if core_terms:
                core_term = float(np.mean(core_terms))
            else:
                core_term = 0.0

            ret_between_clipped = max(float(ret_between), 0.0)
            ret_between_log = float(np.log1p(ret_between_clipped))
            ret_inv_within = 1.0 / (float(ret_within) + 1e-8)
            ret_inv_within_clipped = max(ret_inv_within, 0.0)
            ret_within_reverse_log = float(np.log1p(ret_inv_within_clipped))
            ret_term = ret_between_log + ret_within_reverse_log

            wcov_term = core_term + 0.5 * ret_term
            ratio_cap = float(config.get("map_xgb_wcov_ratio_cap", 10.0))
            wcov_term_capped = float(min(wcov_term, ratio_cap))
        except Exception as obj_exc:
            tprint_warning(
                f"Map XGB: WCoV objective computation failed in objective evaluation: {obj_exc}"
            )
            wcov_term_capped = 0.0

        if not np.isfinite(wcov_term_capped):
            tprint_warning(
                "Map XGB: non-finite WCoV objective in objective evaluation "
                f"(core_between={core_between}, core_within={core_within}, "
                f"ret_between={ret_between}, ret_within={ret_within}, "
                f"wcov_term={wcov_term_capped})"
            )
            wcov_term_capped = 0.0

        within_spread_penalty = 0.0
        try:
            unique_regimes = [int(r) for r in np.unique(labels_arr) if r >= 0]
            per_regime_cvs: List[float] = []
            for rid in unique_regimes:
                regime_mask = labels_arr == rid
                regime_data = core_val.loc[regime_mask]
                if regime_data.shape[0] < 2:
                    continue

                feature_cvs: List[float] = []
                for col in core_val.columns:
                    col_data = regime_data[col].dropna()
                    if len(col_data) > 1:
                        lower_bound = col_data.quantile(0.05)
                        upper_bound = col_data.quantile(0.95)
                        col_winsorized = col_data.clip(
                            lower=lower_bound,
                            upper=upper_bound,
                        )
                        cv_val = col_winsorized.std() / (
                            np.abs(col_winsorized.mean()) + 1e-8
                        )
                        feature_cvs.append(float(cv_val))

                if feature_cvs:
                    per_regime_cvs.append(float(np.mean(feature_cvs)))

            if len(per_regime_cvs) >= 2:
                spread = float(max(per_regime_cvs) - min(per_regime_cvs))
                spread_weight = float(
                    config.get("map_xgb_within_spread_penalty_weight", 0.2)
                )
                within_spread_penalty = spread_weight * spread
        except Exception:
            within_spread_penalty = 0.0

        balance_penalty = 0.0
        try:
            valid_labels = labels_arr[labels_arr >= 0]
            if valid_labels.size > 0:
                unique, counts = np.unique(valid_labels, return_counts=True)
                counts = counts.astype(float)
                p = counts / counts.sum()

                k = float(len(p))
                if k > 0:
                    p_target = np.full_like(p, 1.0 / k, dtype=float)
                    imbalance = float(
                        np.sqrt(np.mean((p - p_target) ** 2))
                    )

                    balance_strength = float(
                        config.get("map_xgb_balance_strength", 0.0)
                    )
                    min_regime_pct = float(
                        config.get("map_xgb_min_regime_pct", 0.0)
                    )

                    balance_penalty = balance_strength * imbalance

                    if min_regime_pct > 0.0 and np.min(p) < min_regime_pct:
                        balance_penalty *= 1.5
        except Exception:
            balance_penalty = 0.0

        score = float(wcov_term_capped - within_spread_penalty - balance_penalty)
        if not np.isfinite(score):
            score = 0.0
        return score, float(wcov_term_capped), float(within_spread_penalty), float(balance_penalty)

    def _calculate_winsorized_cv_between(
        self,
        regime_labels: np.ndarray,
        features: Union[pd.DataFrame, pd.Series],
        lower_pct: float = 0.05,
        upper_pct: float = 0.95,
    ) -> float:
        """Calculate between-regime CV using winsorized means."""
        if isinstance(features, pd.Series):
            features = features.to_frame()

        regime_means: List[float] = []
        regime_sizes: List[int] = []
        for regime_id in np.unique(regime_labels):
            if regime_id < 0:  # Skip invalid labels
                continue

            regime_mask = regime_labels == regime_id
            regime_data = features.loc[regime_mask]
            regime_sizes.append(int(regime_mask.sum()))

            regime_means_winsorized: List[float] = []
            for col in features.columns:
                col_data = regime_data[col].dropna()
                if len(col_data) > 0:
                    lower_bound = col_data.quantile(lower_pct)
                    upper_bound = col_data.quantile(upper_pct)
                    col_winsorized = col_data.clip(lower=lower_bound, upper=upper_bound)
                    regime_means_winsorized.append(float(col_winsorized.mean()))

            if regime_means_winsorized:
                regime_means.append(float(np.mean(regime_means_winsorized)))

        if len(regime_means) < 2:
            return 0.0

        regime_means_array = np.asarray(regime_means, dtype=float)
        regime_sizes_array = np.asarray(regime_sizes[: len(regime_means)], dtype=float)
        if regime_sizes_array.sum() <= 0:
            weights = None
        else:
            weights = regime_sizes_array / regime_sizes_array.sum()

        mean_weighted = float(np.average(regime_means_array, weights=weights))
        var_weighted = float(np.average((regime_means_array - mean_weighted) ** 2, weights=weights))
        std_weighted = float(np.sqrt(max(var_weighted, 0.0)))

        cv_between = std_weighted / (np.abs(mean_weighted) + 1e-8)
        return float(cv_between)

    def _calculate_winsorized_cv_within(
        self,
        regime_labels: np.ndarray,
        features: Union[pd.DataFrame, pd.Series],
        lower_pct: float = 0.05,
        upper_pct: float = 0.95,
    ) -> float:
        """Calculate within-regime CV using winsorized standard deviations."""
        if isinstance(features, pd.Series):
            features = features.to_frame()

        within_cvs: List[float] = []
        regime_sizes: List[int] = []

        for regime_id in np.unique(regime_labels):
            if regime_id < 0:
                continue

            regime_mask = regime_labels == regime_id
            regime_data = features.loc[regime_mask]
            regime_size = int(regime_mask.sum())
            if regime_size < 2:
                continue

            feature_cvs: List[float] = []
            for col in features.columns:
                col_data = regime_data[col].dropna()
                if len(col_data) > 1:
                    lower_bound = col_data.quantile(lower_pct)
                    upper_bound = col_data.quantile(upper_pct)
                    col_winsorized = col_data.clip(lower=lower_bound, upper=upper_bound)
                    cv = col_winsorized.std() / (np.abs(col_winsorized.mean()) + 1e-8)
                    feature_cvs.append(float(cv))

            if feature_cvs:
                within_cvs.append(float(np.mean(feature_cvs)))
                regime_sizes.append(regime_size)

        if not within_cvs:
            return 1.0

        within_cvs_array = np.asarray(within_cvs, dtype=float)
        regime_sizes_array = np.asarray(regime_sizes[: len(within_cvs)], dtype=float)
        if regime_sizes_array.sum() <= 0:
            weights = None
        else:
            # Use higher-order regime weights to emphasize larger, more reliable regimes
            w = regime_sizes_array / regime_sizes_array.sum()
            weights = w ** 3

        # Shapes of within_cvs_array and weights are now aligned (one CV per regime)
        weighted_cv = float(np.average(within_cvs_array, weights=weights))
        return float(weighted_cv)

    def _generate_map_regime_reports(
        self,
        df: pd.DataFrame,
        regime_col: str,
        symbol: str,
        exchange: str,
        regime_timeframe: str,
    ) -> Dict[str, float]:
        """Generate per-regime and global WCoV diagnostics for Map regimes.

        Uses the specified regime column (default: map_trend_state) and a
        curated set of core Map structure features together with returns_1h.

        Outputs a CSV and Markdown report under outcomes/ and returns a small
        dict of global WCoV metrics to be merged into the step metrics.
        """
        if regime_col not in df.columns:
            return {}

        regime_series = df[regime_col].dropna()
        if regime_series.empty:
            return {}

        labels = regime_series.to_numpy()
        unique_regimes = np.unique(labels)
        if len(unique_regimes) < 2:
            return {}

        # Core Map structure features used for separation diagnostics
        important_features = [
            "map_distance_to_nearest_pivot_atr",
            "map_price_vs_vwap_day",
            "map_price_vs_vwap_week",
            "map_dist_to_pdh",
            "map_dist_to_pdl",
            "map_fvg_size_atr",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "map_volume_profile_z",
        ]
        core_features = [f for f in important_features if f in df.columns]

        valid_index = regime_series.index
        core_between = core_within = core_ratio = np.nan
        if core_features:
            core_df = df.loc[valid_index, core_features].astype(float)
            core_between = self._calculate_winsorized_cv_between(labels, core_df)
            core_within = self._calculate_winsorized_cv_within(labels, core_df)
            core_ratio = core_between / (core_within + 1e-8)

        returns_between = returns_within = returns_ratio = np.nan
        returns_series = None
        if "returns_1h" in df.columns:
            returns_series = df.loc[valid_index, "returns_1h"].astype(float)
            returns_between = self._calculate_winsorized_cv_between(labels, returns_series)
            returns_within = self._calculate_winsorized_cv_within(labels, returns_series)
            returns_ratio = returns_between / (returns_within + 1e-8)

        # Per-regime summary: counts, mean/std of returns and core features, Sharpe
        rows: List[Dict[str, Any]] = []
        total_n = len(labels)
        returns_all = returns_series.to_numpy() if returns_series is not None else None
        core_df_for_regimes = df.loc[valid_index, core_features].astype(float) if core_features else None

        for regime_id in unique_regimes:
            regime_mask = labels == regime_id
            n_reg = int(regime_mask.sum())
            if n_reg == 0:
                continue

            row: Dict[str, Any] = {
                "regime_id": int(regime_id),
                "n_samples": n_reg,
                "sample_fraction": float(n_reg) / float(total_n),
            }

            if returns_all is not None:
                vals = returns_all[regime_mask]
                mean_r = float(np.nanmean(vals))
                std_r = float(np.nanstd(vals))
                sharpe_r = mean_r / (std_r + 1e-8)
                row["mean_returns_1h"] = mean_r
                row["std_returns_1h"] = std_r
                row["sharpe_1h"] = sharpe_r

            if core_df_for_regimes is not None:
                for feat in core_features:
                    feat_vals = core_df_for_regimes[feat].to_numpy()[regime_mask]
                    row[f"mean_{feat}"] = float(np.nanmean(feat_vals))
                    row[f"std_{feat}"] = float(np.nanstd(feat_vals))

            rows.append(row)

        regime_stats_df = pd.DataFrame(rows)

        # Global summary row with WCoV diagnostics
        global_row: Dict[str, Any] = {
            "regime_id": "GLOBAL",
            "n_samples": int(total_n),
            "sample_fraction": 1.0,
        }
        if returns_all is not None:
            mean_r_all = float(np.nanmean(returns_all))
            std_r_all = float(np.nanstd(returns_all))
            global_row["mean_returns_1h"] = mean_r_all
            global_row["std_returns_1h"] = std_r_all
            global_row["sharpe_1h"] = mean_r_all / (std_r_all + 1e-8)

        global_row["wcov_core_between"] = float(core_between) if not np.isnan(core_between) else np.nan
        global_row["wcov_core_within"] = float(core_within) if not np.isnan(core_within) else np.nan
        global_row["wcov_core_ratio"] = float(core_ratio) if not np.isnan(core_ratio) else np.nan
        global_row["wcov_returns_between"] = float(returns_between) if not np.isnan(returns_between) else np.nan
        global_row["wcov_returns_within"] = float(returns_within) if not np.isnan(returns_within) else np.nan
        global_row["wcov_returns_ratio"] = float(returns_ratio) if not np.isnan(returns_ratio) else np.nan

        full_df = pd.concat([pd.DataFrame([global_row]), regime_stats_df], ignore_index=True)

        out_dir = Path("outcomes")
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_name = f"ml_map_regime_wcov_metrics_{symbol}_{regime_timeframe}_{ts}.csv"
        csv_path = out_dir / csv_name
        full_df.to_csv(csv_path, index=False)

        md_name = f"ml_map_regime_wcov_report_{symbol}_{regime_timeframe}_{ts}.md"
        md_path = out_dir / md_name

        lines: List[str] = []
        lines.append("# ML Map Regime WCoV Report")
        lines.append("")
        lines.append(f"- Symbol: **{symbol}**")
        lines.append(f"- Exchange: **{exchange}**")
        lines.append(f"- Regime timeframe: **{regime_timeframe}**")
        lines.append(f"- Regime column: **{regime_col}**")
        lines.append(f"- Generated at: **{ts}**")
        lines.append("")
        lines.append("## Global WCoV metrics")
        lines.append("")
        lines.append("| metric | between_cv | within_cv | ratio |")
        lines.append("| --- | --- | --- | --- |")

        if not np.isnan(core_between) and not np.isnan(core_within):
            lines.append(
                f"| core_features | {core_between:.6f} | {core_within:.6f} | {core_ratio:.6f} |"
            )
        else:
            lines.append("| core_features | nan | nan | nan |")

        if not np.isnan(returns_between) and not np.isnan(returns_within):
            lines.append(
                f"| returns_1h | {returns_between:.6f} | {returns_within:.6f} | {returns_ratio:.6f} |"
            )
        else:
            lines.append("| returns_1h | nan | nan | nan |")

        lines.append("")
        lines.append("## Per-regime summary")
        lines.append("")

        if not regime_stats_df.empty:
            cols = list(regime_stats_df.columns)
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, row in regime_stats_df.iterrows():
                values = [row[c] for c in cols]
                lines.append("| " + " | ".join(str(v) for v in values) + " |")
        else:
            lines.append("No per-regime statistics available (insufficient data).")

        md_path.write_text("\n".join(lines), encoding="utf-8")

        tprint_info(
            f"💾 Saved Map regime WCoV metrics CSV: {csv_path} and Markdown report: {md_path}"
        )

        return {
            "map_core_wcov_between": float(core_between) if not np.isnan(core_between) else 0.0,
            "map_core_wcov_within": float(core_within) if not np.isnan(core_within) else 0.0,
            "map_core_wcov_ratio": float(core_ratio) if not np.isnan(core_ratio) else 0.0,
            "map_returns_wcov_between": float(returns_between) if not np.isnan(returns_between) else 0.0,
            "map_returns_wcov_within": float(returns_within) if not np.isnan(returns_within) else 0.0,
            "map_returns_wcov_ratio": float(returns_ratio) if not np.isnan(returns_ratio) else 0.0,
        }
