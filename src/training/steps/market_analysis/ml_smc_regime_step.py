"""
ML SMC Regime Step

Smart Money Concepts (SMC) step that consumes 15m OHLCV data to construct
SMC-based features (liquidity scalars, FVG/inefficiency, premium/discount,
displacement, volume profile, time categories) and trains an XGBoost regressor
to predict future price position (target ratio) with conformal prediction calibration.
"""
import logging
import time
import pickle
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

logger = logging.getLogger(__name__)


class MLSMCRegimeStep(BaseStep):
    def __init__(self, step_name: str = "ml_smc_regime_step"):
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLSMCRegimeStep") if hasattr(logger, "getChild") else logger
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

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            tprint_info(
                f"🚀 Starting {self.step_name} (SMC Regime) for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # Load market data with caching
            exec_mode_cfg = str(config.get("execution_mode", "")).lower()
            cache_key = (symbol, exchange, regime_timeframe, exec_mode_cfg, "smc")
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
                tprint_info("♻️ Reusing cached 15m market data for ML SMC regimes")
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

            # Generate SMC features
            smc_df = self._generate_smc_features(market_data, config)

            # Train XGBoost model if enabled
            if bool(config.get("smc_xgb_enable_training", True)):
                try:
                    xgb_metrics, xgb_artifacts = self._train_smc_xgb_model(
                        smc_df,
                        config,
                        symbol,
                        exchange,
                        regime_timeframe,
                    )
                    if xgb_metrics:
                        metrics.update(xgb_metrics)
                    if xgb_artifacts:
                        artifacts.extend(xgb_artifacts)
                except Exception as xgb_exc:
                    tprint_error(f"SMC XGB training failed: {xgb_exc}")
                    raise

            # Save SMC features
            idx_name = smc_df.index.name or "index"
            if idx_name in smc_df.columns:
                to_save = smc_df.copy()
                if "timestamp" not in to_save.columns:
                    to_save.insert(0, "timestamp", smc_df.index.to_numpy())
            else:
                to_save = smc_df.reset_index().rename(columns={idx_name: "timestamp"})

            smc_features_path = self._save_artifact(
                data=to_save,
                artifact_name="ml_smc_features_15m",
                artifact_type="data",
                data_category="features",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "source_market_data": market_source,
                },
            )
            artifacts.append(smc_features_path)

            metrics.update({
                "n_samples": int(len(smc_df)),
            })

            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time

            tprint_success(
                f"✅ {self.step_name} (SMC Regime) completed in {execution_time:.2f}s "
                f"with {len(smc_df)} samples"
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

    def _generate_smc_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Generate all SMC features."""
        result = df.copy()

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in result.columns]
        if missing:
            raise ValueError(f"Missing columns for SMC features: {missing}")

        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)
        result = result.sort_index()

        o = result["open"].astype(float)
        h = result["high"].astype(float)
        l = result["low"].astype(float)
        c = result["close"].astype(float)
        v = result["volume"].astype(float)

        # Calculate ATR (core normalization factor for SMC)
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_window = int(config.get("smc_atr_window", 14))
        atr = true_range.rolling(window=atr_window).mean()
        result["atr"] = atr

        # 1. Liquidity & Reference Point Scalars
        tprint_info("Generating liquidity & reference point features...")
        result = self._add_liquidity_features(result, config)

        # 2. Inefficiency (FVG/Gap) Scalars
        tprint_info("Generating FVG/inefficiency features...")
        result = self._add_fvg_features(result, config)

        # 3. Premium / Discount & Structure Scalars
        tprint_info("Generating premium/discount features...")
        result = self._add_premium_discount_features(result, config)

        # 4. Momentum & Displacement Scalars
        tprint_info("Generating momentum/displacement features...")
        result = self._add_momentum_features(result, config)

        # 5. Volatility & Time Scalars
        tprint_info("Generating volatility/time features...")
        result = self._add_volatility_time_features(result, config)

        # 6. Multi-Timeframe (MTF) Scalars
        tprint_info("Generating multi-timeframe features...")
        result = self._add_mtf_features(result, config)

        # 7. Volume Profile Scalars
        tprint_info("Generating volume profile features...")
        result = self._add_volume_profile_features(result, config)

        # 8. Time Categories
        tprint_info("Generating time category features...")
        result = self._add_time_categories(result, config)

        tprint_success(f"✅ Generated {len(result.columns)} SMC features")

        return result

    def _add_liquidity_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add liquidity and reference point features."""
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        atr = df["atr"].astype(float)

        # Previous day high/low (PDH/PDL)
        day_index = df.index.normalize()
        daily = df.groupby(day_index).agg(high=("high", "max"), low=("low", "min"), open=("open", "first"))
        prev_day_high = daily["high"].shift(1)
        prev_day_low = daily["low"].shift(1)
        day_open = daily["open"]

        pdh = prev_day_high.reindex(day_index).to_numpy()
        pdl = prev_day_low.reindex(day_index).to_numpy()
        day_open_vals = day_open.reindex(day_index).to_numpy()

        df["pdh"] = pdh
        df["pdl"] = pdl
        df["dist_to_pdh_atr"] = (c.values - pdh) / (atr.values + 1e-9)
        df["dist_to_pdl_atr"] = (c.values - pdl) / (atr.values + 1e-9)

        # NY midnight open (00:00 NYC time = 05:00 UTC)
        # Approximate as the first candle of each day
        df["day_open"] = day_open_vals
        df["dist_to_day_open"] = (c.values - day_open_vals) / (atr.values + 1e-9)

        # Week open
        week_index = df.index.to_period('W').to_timestamp()
        weekly = df.groupby(week_index).agg(open=("open", "first"), close=("close", "last"))
        week_open = weekly["open"]
        prev_week_close = weekly["close"].shift(1)

        week_open_vals = week_open.reindex(week_index).to_numpy()
        prev_week_close_vals = prev_week_close.reindex(week_index).to_numpy()

        df["week_open"] = week_open_vals
        df["dist_to_week_open"] = (c.values - week_open_vals) / (atr.values + 1e-9)

        # New Week Opening Gap (NWOG)
        nwog_gap = week_open_vals - prev_week_close_vals
        df["nwog_gap_size"] = nwog_gap / (atr.values + 1e-9)

        return df

    def _add_fvg_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add Fair Value Gap (FVG) and inefficiency features."""
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        o = df["open"].astype(float)
        atr = df["atr"].astype(float)

        # FVG detection: gap between current candle and candle 2 bars ago
        # Bullish FVG: low[i] > high[i-2]
        # Bearish FVG: high[i] < low[i-2]

        high_2 = h.shift(2)
        low_2 = l.shift(2)

        bullish_fvg = l > high_2
        bearish_fvg = h < low_2

        # FVG size
        bullish_fvg_size = (l - high_2).clip(lower=0.0)
        bearish_fvg_size = (low_2 - h).clip(lower=0.0)

        fvg_size = bullish_fvg_size.where(bullish_fvg, bearish_fvg_size.where(bearish_fvg, 0.0))
        df["current_fvg_size"] = fvg_size / (atr + 1e-9)

        # Calculate FVG midpoint for nearest distance calculation
        bullish_fvg_mid = (l + high_2) / 2.0
        bearish_fvg_mid = (h + low_2) / 2.0

        fvg_mid = bullish_fvg_mid.where(bullish_fvg, bearish_fvg_mid.where(bearish_fvg, np.nan))

        # Nearest FVG distance (simplified: just use most recent FVG)
        fvg_mid_filled = fvg_mid.ffill()
        df["nearest_fvg_dist"] = (c - fvg_mid_filled) / (atr + 1e-9)

        # Consequent encroachment (CE) - position within the FVG
        # For simplicity, calculate for current FVG only
        fvg_high = l.where(bullish_fvg, low_2.where(bearish_fvg, np.nan))
        fvg_low = high_2.where(bullish_fvg, h.where(bearish_fvg, np.nan))

        fvg_range = fvg_high - fvg_low
        ce_position = (c - fvg_low) / (fvg_range + 1e-9)
        df["consequent_encroachment"] = ce_position.fillna(0.5)

        # Volume imbalance (gaps between candle bodies)
        volume_imb = (o - c.shift(1)).abs()
        df["volume_imbalance_size"] = volume_imb / (atr + 1e-9)

        # Gap fill ratio (how much of the FVG has been filled)
        # Simplified: if price has moved into the FVG, calculate fill percentage
        fvg_fill = ((h - fvg_low) / (fvg_range + 1e-9)).clip(0.0, 1.0)
        df["gap_fill_ratio"] = fvg_fill.fillna(0.0)

        return df

    def _add_premium_discount_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add premium/discount and structure features."""
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        atr = df["atr"].astype(float)

        # Swing highs and lows (5-bar fractal)
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

        swing_high = h.where(sh).ffill()
        swing_low = l.where(sl).ffill()

        # Range position (0.0 to 0.5 = Discount, 0.5 to 1.0 = Premium)
        range_height = swing_high - swing_low
        range_pos = (c - swing_low) / (range_height + 1e-9)
        df["range_position"] = range_pos.clip(0.0, 1.0)

        # Distance to swing high/low
        df["dist_to_swing_high"] = (swing_high - c) / (atr + 1e-9)
        df["dist_to_swing_low"] = (c - swing_low) / (atr + 1e-9)

        # Fibonacci retracement level (as a ratio)
        fib_level = (swing_high - c) / (range_height + 1e-9)
        df["fib_retracement_level"] = fib_level.clip(0.0, 1.0)

        # Break of structure magnitude
        # Detection: when price breaks above previous swing high
        prev_swing_high = swing_high.shift(1)
        bos_magnitude = (c - prev_swing_high) / (atr + 1e-9)
        df["break_of_structure_mag"] = bos_magnitude.clip(lower=0.0)

        return df

    def _add_momentum_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add momentum and displacement features."""
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)

        # Body size
        body = (c - o).abs()
        avg_body = body.rolling(window=20).mean()

        # Displacement strength
        df["displacement_strength"] = body / (avg_body + 1e-9)

        # Wick to body ratio
        upper_wick = h - np.maximum(c, o)
        lower_wick = np.minimum(c, o) - l
        total_wick = upper_wick + lower_wick
        df["wick_body_ratio"] = total_wick / (body + 1e-9)

        # Close position in candle (0.0 = close at low, 1.0 = close at high)
        candle_range = h - l
        close_pos = (c - l) / (candle_range + 1e-9)
        df["close_position_in_candle"] = close_pos.clip(0.0, 1.0)

        # Velocity / Rate of Change
        roc = (c - c.shift(3)) / 3.0
        df["velocity_roc"] = roc / (c.shift(3) + 1e-9)

        # Consecutive candles (streak)
        candle_direction = np.sign(c - o)

        # Calculate streaks
        def calculate_streaks(series):
            streaks = np.zeros(len(series))
            current_streak = 0
            for i in range(len(series)):
                if i == 0:
                    current_streak = series.iloc[i]
                elif series.iloc[i] == series.iloc[i-1] and series.iloc[i] != 0:
                    current_streak += series.iloc[i]
                else:
                    current_streak = series.iloc[i]
                streaks[i] = current_streak
            return streaks

        df["consecutive_candles"] = calculate_streaks(pd.Series(candle_direction))

        return df

    def _add_volatility_time_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add volatility and time-based features."""
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        v = df["volume"].astype(float)
        atr = df["atr"].astype(float)

        # Average Daily Range (ADR)
        day_index = df.index.normalize()
        daily_range = df.groupby(day_index).apply(lambda x: x["high"].max() - x["low"].min())
        adr = daily_range.rolling(window=20).mean()

        # Today's range filled percentage
        today_high = df.groupby(day_index)["high"].transform("max")
        today_low = df.groupby(day_index)["low"].transform("min")
        today_range = today_high - today_low
        adr_reindexed = adr.reindex(day_index).to_numpy()
        df["adr_filled_pct"] = today_range / (adr_reindexed + 1e-9)

        # Relative volume
        avg_vol = v.rolling(window=20).mean()
        df["rel_volume"] = v / (avg_vol + 1e-9)

        # Time elapsed in session (minutes since midnight)
        df["time_elapsed_session"] = df.index.hour * 60 + df.index.minute

        # ATR compression/expansion
        atr_20 = atr.rolling(window=20).mean()
        df["atr_compression"] = atr / (atr_20 + 1e-9)

        return df

    def _add_mtf_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add multi-timeframe features."""
        c = df["close"].astype(float)
        atr = df["atr"].astype(float)

        # Resample to 1H for HTF features
        try:
            df_1h = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if len(df_1h) > 50:
                # HTF trend (EMA slope)
                ema_20 = df_1h["close"].ewm(span=20).mean()
                ema_50 = df_1h["close"].ewm(span=50).mean()

                # Calculate HTF ATR
                h_1h = df_1h["high"]
                l_1h = df_1h["low"]
                c_1h = df_1h["close"]
                tr1_1h = h_1h - l_1h
                tr2_1h = (h_1h - c_1h.shift(1)).abs()
                tr3_1h = (l_1h - c_1h.shift(1)).abs()
                tr_1h = pd.concat([tr1_1h, tr2_1h, tr3_1h], axis=1).max(axis=1)
                atr_1h = tr_1h.rolling(window=14).mean()

                htf_trend_slope = (ema_20 - ema_50) / (atr_1h + 1e-9)

                # Reindex to 15m
                htf_trend_slope_15m = htf_trend_slope.reindex(df.index, method='ffill')
                df["htf_trend_slope"] = htf_trend_slope_15m.fillna(0.0)
            else:
                df["htf_trend_slope"] = 0.0

        except Exception as e:
            tprint_warning(f"MTF feature calculation failed: {e}")
            df["htf_trend_slope"] = 0.0

        # Placeholder for daily FVG distance (would need daily data)
        df["dist_to_daily_fvg"] = 0.0

        # Daily wick rejection (using previous day data)
        day_index = df.index.normalize()
        daily_stats = df.groupby(day_index).agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last")
        )
        prev_day_high = daily_stats["high"].shift(1)
        prev_day_low = daily_stats["low"].shift(1)
        prev_day_close = daily_stats["close"].shift(1)
        prev_day_range = prev_day_high - prev_day_low

        daily_wick_rej = (prev_day_high - prev_day_close) / (prev_day_range + 1e-9)
        df["daily_wick_rejection"] = daily_wick_rej.reindex(day_index).fillna(0.0).to_numpy()

        return df

    def _add_volume_profile_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add volume profile features."""
        c = df["close"].astype(float)
        v = df["volume"].astype(float)
        atr = df["atr"].astype(float)

        lookback = int(config.get("smc_vp_lookback", 100))
        bins = int(config.get("smc_vp_bins", 50))

        # Calculate rolling volume profile
        hvn_gravity_list = []
        poc_dist_list = []
        is_in_value_area_list = []
        profile_skew_list = []

        for i in range(len(df)):
            if i < lookback:
                hvn_gravity_list.append(0.5)
                poc_dist_list.append(0.0)
                is_in_value_area_list.append(1)
                profile_skew_list.append(0.0)
                continue

            window_close = c.iloc[i-lookback:i]
            window_volume = v.iloc[i-lookback:i]
            current_price = c.iloc[i]

            try:
                hist, bin_edges = np.histogram(
                    window_close,
                    bins=bins,
                    weights=window_volume,
                )

                # Find current bin
                bin_index = np.digitize(current_price, bin_edges) - 1
                bin_index = max(0, min(bins-1, bin_index))

                vol_at_price = hist[bin_index]
                max_vol = np.max(hist)

                # HVN gravity
                hvn_gravity = vol_at_price / (max_vol + 1e-9)
                hvn_gravity_list.append(float(hvn_gravity))

                # POC distance
                poc_price = bin_edges[np.argmax(hist)]
                poc_dist = (current_price - poc_price) / (atr.iloc[i] + 1e-9)
                poc_dist_list.append(float(poc_dist))

                # Value area (70% of volume)
                sorted_indices = np.argsort(hist)[::-1]
                cumsum = 0
                value_area_bins = []
                for idx in sorted_indices:
                    cumsum += hist[idx]
                    value_area_bins.append(idx)
                    if cumsum >= 0.7 * hist.sum():
                        break

                is_in_va = 1 if bin_index in value_area_bins else 0
                is_in_value_area_list.append(is_in_va)

                # Profile skew
                volume_above = hist[bin_index:].sum()
                volume_below = hist[:bin_index].sum()
                skew = (volume_above - volume_below) / (hist.sum() + 1e-9)
                profile_skew_list.append(float(skew))

            except Exception:
                hvn_gravity_list.append(0.5)
                poc_dist_list.append(0.0)
                is_in_value_area_list.append(1)
                profile_skew_list.append(0.0)

        df["hvn_gravity"] = hvn_gravity_list
        df["poc_dist_atr"] = poc_dist_list
        df["is_in_value_area"] = is_in_value_area_list
        df["profile_skew"] = profile_skew_list

        return df

    def _add_time_categories(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add time category features."""

        # Session (Kill Zone) - UTC times
        # Asia: 00:00-08:00, London: 08:00-16:00, NY_AM: 13:00-17:00, NY_PM: 17:00-21:00, Dead: 21:00-00:00
        hour = df.index.hour

        session_kz = pd.Series("Dead", index=df.index)
        session_kz[(hour >= 0) & (hour < 8)] = "Asia"
        session_kz[(hour >= 8) & (hour < 13)] = "London"
        session_kz[(hour >= 13) & (hour < 17)] = "NY_AM"
        session_kz[(hour >= 17) & (hour < 21)] = "NY_PM"

        # One-hot encode sessions
        for session in ["Asia", "London", "NY_AM", "NY_PM", "Dead"]:
            df[f"session_{session}"] = (session_kz == session).astype(int)

        # Day of week
        dow = df.index.dayofweek  # Monday=0, Sunday=6
        for day_num, day_name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            df[f"dow_{day_name}"] = (dow == day_num).astype(int)

        # Market structure (simplified: based on recent swing direction)
        h = df["high"].astype(float)
        l = df["low"].astype(float)

        hh = (h > h.shift(1)) & (h.shift(1) > h.shift(2))
        ll = (l < l.shift(1)) & (l.shift(1) < l.shift(2))
        lh = (h < h.shift(1)) & (h.shift(1) < h.shift(2))
        hl = (l > l.shift(1)) & (l.shift(1) > l.shift(2))

        market_structure = pd.Series("Range", index=df.index)
        market_structure[hh & hl] = "Uptrend"
        market_structure[lh & ll] = "Downtrend"

        df["market_structure_Uptrend"] = (market_structure == "Uptrend").astype(int)
        df["market_structure_Downtrend"] = (market_structure == "Downtrend").astype(int)
        df["market_structure_Range"] = (market_structure == "Range").astype(int)

        # Inside FVG (use existing FVG size as proxy)
        df["is_inside_fvg"] = (df["current_fvg_size"] > 0).astype(int)

        # Sweep confirmed (wick beyond PDH/PDL but close inside)
        c = df["close"].astype(float)
        pdh = df["pdh"]
        pdl = df["pdl"]

        sweep_high = (h > pdh) & (c < pdh)
        sweep_low = (l < pdl) & (c > pdl)
        df["sweep_confirmed"] = (sweep_high | sweep_low).astype(int)

        return df

    def _train_smc_xgb_model(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        symbol: str,
        exchange: str,
        regime_timeframe: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Train XGBoost regression model with conformal prediction calibration."""
        metrics: Dict[str, Any] = {}
        artifacts: List[str] = []

        try:
            import xgboost as xgb
        except Exception:
            tprint_warning("xgboost not installed, skipping SMC XGB training")
            metrics["smc_xgb_early_exit_reason"] = "xgboost_not_installed"
            return metrics, artifacts

        if "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
            tprint_warning("SMC XGB: missing price columns; skipping training")
            metrics["smc_xgb_early_exit_reason"] = "missing_price_columns"
            return metrics, artifacts

        # Create target ratio
        lookahead = int(config.get("smc_lookahead", 16))  # 4 hours on 15m chart

        future_close = df["close"].shift(-lookahead)
        current_range = (df["high"] - df["low"]).replace(0, 0.0001)
        target_ratio = (future_close - df["low"]) / current_range

        df_with_target = df.copy()
        df_with_target["target_ratio"] = target_ratio
        df_with_target = df_with_target.dropna(subset=["target_ratio"])

        tprint_info(f"SMC XGB: target ratio stats - mean: {target_ratio.mean():.3f}, std: {target_ratio.std():.3f}")

        # Select features (exclude target and non-numeric)
        exclude_cols = ["target_ratio", "pdh", "pdl", "day_open", "week_open"]
        numeric_df = df_with_target.select_dtypes(include=[np.number])
        feature_cols = [col for col in numeric_df.columns if col not in exclude_cols]

        if len(feature_cols) < 5:
            tprint_warning(f"SMC XGB: insufficient features (n={len(feature_cols)})")
            metrics["smc_xgb_early_exit_reason"] = f"insufficient_features_{len(feature_cols)}"
            return metrics, artifacts

        X = numeric_df[feature_cols].astype(np.float32)
        y = df_with_target["target_ratio"].astype(np.float32)

        # Handle infinities and NaNs
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)

        min_samples = int(config.get("smc_xgb_min_samples", 800))
        if len(X) < min_samples:
            tprint_warning(f"SMC XGB: insufficient samples (n={len(X)}, min={min_samples})")
            metrics["smc_xgb_early_exit_reason"] = f"insufficient_samples_{len(X)}"
            return metrics, artifacts

        # Time-series split (80/20)
        train_frac = float(config.get("smc_xgb_train_fraction", 0.80))
        split_idx = int(len(X) * train_frac)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        tprint_info(f"SMC XGB: train={len(X_train)}, test={len(X_test)}, features={len(feature_cols)}")

        # Train XGBoost model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=int(config.get("smc_xgb_n_estimators", 500)),
            learning_rate=float(config.get("smc_xgb_learning_rate", 0.05)),
            max_depth=int(config.get("smc_xgb_max_depth", 6)),
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

        early_stop = int(config.get("smc_xgb_early_stopping", 50))

        tprint_info("Training XGBoost regression model...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        metrics.update({
            "smc_xgb_train_rmse": float(train_rmse),
            "smc_xgb_test_rmse": float(test_rmse),
            "smc_xgb_train_mae": float(train_mae),
            "smc_xgb_test_mae": float(test_mae),
            "smc_xgb_train_r2": float(train_r2),
            "smc_xgb_test_r2": float(test_r2),
        })

        # Directional accuracy (breakout prediction)
        breakout_mask_test = y_test_pred > 1.0
        if breakout_mask_test.sum() > 0:
            breakout_accuracy = (y_test[breakout_mask_test] > 1.0).mean()
            metrics["smc_xgb_breakout_accuracy"] = float(breakout_accuracy)
            tprint_info(f"Breakout prediction accuracy: {breakout_accuracy:.2%}")

        breakdown_mask_test = y_test_pred < 0.0
        if breakdown_mask_test.sum() > 0:
            breakdown_accuracy = (y_test[breakdown_mask_test] < 0.0).mean()
            metrics["smc_xgb_breakdown_accuracy"] = float(breakdown_accuracy)
            tprint_info(f"Breakdown prediction accuracy: {breakdown_accuracy:.2%}")

        tprint_success(
            f"✅ XGBoost trained: train_rmse={train_rmse:.4f}, test_rmse={test_rmse:.4f}, "
            f"train_r2={train_r2:.4f}, test_r2={test_r2:.4f}"
        )

        # Conformal prediction calibration
        tprint_info("Performing conformal prediction calibration...")
        calibration_results = self._calibrate_conformal_prediction(
            model, X_train, y_train, X_test, y_test
        )

        if calibration_results:
            metrics.update(calibration_results["metrics"])

        # Generate comprehensive reports
        tprint_info("Generating comprehensive reports...")
        report_artifacts = self._generate_smc_reports(
            df_with_target,
            model,
            X,
            y,
            feature_cols,
            calibration_results,
            symbol,
            exchange,
            regime_timeframe,
        )
        artifacts.extend(report_artifacts)

        # Save model
        tprint_info("Saving XGBoost model...")
        model_path = self._save_artifact(
            data=model,
            artifact_name="smc_xgb_model",
            artifact_type="model",
            data_category="models",
            metadata={
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "n_features": len(feature_cols),
                "test_rmse": float(test_rmse),
                "test_r2": float(test_r2),
            },
        )
        artifacts.append(model_path)

        # Save calibration
        if calibration_results:
            calibration_path = self._save_artifact(
                data=calibration_results["calibration"],
                artifact_name="smc_conformal_calibration",
                artifact_type="model",
                data_category="models",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                },
            )
            artifacts.append(calibration_path)

        # Save predictions and confidence scores
        predictions_df = pd.DataFrame({
            "timestamp": df_with_target.index,
            "actual": y.values,
            "predicted": np.concatenate([y_train_pred, y_test_pred]),
            "is_test": [False] * len(y_train) + [True] * len(y_test),
        })

        if calibration_results:
            all_preds = np.concatenate([y_train_pred, y_test_pred])
            predictions_df["confidence_50"] = calibration_results["confidence_scores"]["50%"]
            predictions_df["confidence_80"] = calibration_results["confidence_scores"]["80%"]
            predictions_df["confidence_90"] = calibration_results["confidence_scores"]["90%"]

        predictions_path = self._save_artifact(
            data=predictions_df,
            artifact_name="smc_predictions_with_confidence",
            artifact_type="data",
            data_category="predictions",
            metadata={
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
            },
        )
        artifacts.append(predictions_path)

        return metrics, artifacts

    def _calibrate_conformal_prediction(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """Perform conformal prediction calibration."""
        try:
            # Split training data into proper train and calibration sets
            cal_frac = 0.2
            split_idx = int(len(X_train) * (1 - cal_frac))

            X_proper_train = X_train.iloc[:split_idx]
            y_proper_train = y_train.iloc[:split_idx]
            X_cal = X_train.iloc[split_idx:]
            y_cal = y_train.iloc[split_idx:]

            # Retrain model on proper training set
            model.fit(X_proper_train, y_proper_train, verbose=False)

            # Calculate non-conformity scores on calibration set
            y_cal_pred = model.predict(X_cal)
            nonconformity_scores = np.abs(y_cal - y_cal_pred)

            # Calculate quantiles for different confidence levels
            confidence_levels = [0.50, 0.80, 0.90, 0.95, 0.99]
            quantiles = {}

            for alpha in confidence_levels:
                q = np.percentile(nonconformity_scores, alpha * 100)
                quantiles[f"{int(alpha*100)}%"] = float(q)

            tprint_info(f"Conformal prediction quantiles: {quantiles}")

            # Calculate prediction intervals for all data
            all_X = pd.concat([X_train, X_test])
            all_preds = model.predict(all_X)

            confidence_scores = {}
            for level, q in quantiles.items():
                # Normalized confidence: how much margin we have
                # Higher is better (prediction is far from boundary)
                conf_score = q / (np.abs(all_preds - 0.5) + 1e-9)
                confidence_scores[level] = conf_score.clip(0.0, 10.0)

            return {
                "calibration": {
                    "quantiles": quantiles,
                    "nonconformity_scores": nonconformity_scores.tolist(),
                },
                "confidence_scores": confidence_scores,
                "metrics": {
                    "conformal_quantile_50": quantiles["50%"],
                    "conformal_quantile_80": quantiles["80%"],
                    "conformal_quantile_90": quantiles["90%"],
                },
            }

        except Exception as e:
            tprint_warning(f"Conformal calibration failed: {e}")
            return None

    def _generate_smc_reports(
        self,
        df: pd.DataFrame,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        calibration_results: Optional[Dict[str, Any]],
        symbol: str,
        exchange: str,
        regime_timeframe: str,
    ) -> List[str]:
        """Generate comprehensive reports (MD/CSV with WCoV, correlations, F1, etc.)."""
        artifacts = []

        out_dir = Path("outcomes")
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Feature importance report
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importance,
        }).sort_values("importance", ascending=False)

        importance_csv = out_dir / f"smc_feature_importance_{symbol}_{regime_timeframe}_{ts}.csv"
        importance_df.to_csv(importance_csv, index=False)
        artifacts.append(str(importance_csv))

        # 2. Correlation with forward returns
        y_pred = model.predict(X)

        # Calculate actual forward returns for correlation
        df_aligned = df.loc[X.index]
        forward_returns = (df_aligned["close"].shift(-16) / df_aligned["close"] - 1)

        corr_with_target = {}
        for feat in feature_cols[:30]:  # Top 30 features
            if feat in X.columns:
                try:
                    corr = X[feat].corr(y)
                    corr_with_target[feat] = float(corr) if not np.isnan(corr) else 0.0
                except Exception:
                    corr_with_target[feat] = 0.0

        corr_df = pd.DataFrame({
            "feature": list(corr_with_target.keys()),
            "correlation_with_target": list(corr_with_target.values()),
        }).sort_values("correlation_with_target", key=abs, ascending=False)

        corr_csv = out_dir / f"smc_feature_correlations_{symbol}_{regime_timeframe}_{ts}.csv"
        corr_df.to_csv(corr_csv, index=False)
        artifacts.append(str(corr_csv))

        # 3. Prediction distribution analysis
        prediction_bins = np.linspace(y.min(), y.max(), 20)
        pred_hist, pred_edges = np.histogram(y_pred, bins=prediction_bins)
        actual_hist, _ = np.histogram(y, bins=prediction_bins)

        dist_df = pd.DataFrame({
            "bin_center": (pred_edges[:-1] + pred_edges[1:]) / 2,
            "predicted_count": pred_hist,
            "actual_count": actual_hist,
        })

        dist_csv = out_dir / f"smc_prediction_distribution_{symbol}_{regime_timeframe}_{ts}.csv"
        dist_df.to_csv(dist_csv, index=False)
        artifacts.append(str(dist_csv))

        # 4. Confidence analysis at different levels
        if calibration_results:
            conf_data = []
            for level, scores in calibration_results["confidence_scores"].items():
                mean_conf = np.mean(scores)
                std_conf = np.std(scores)
                conf_data.append({
                    "confidence_level": level,
                    "mean_score": float(mean_conf),
                    "std_score": float(std_conf),
                })

            conf_df = pd.DataFrame(conf_data)
            conf_csv = out_dir / f"smc_confidence_analysis_{symbol}_{regime_timeframe}_{ts}.csv"
            conf_df.to_csv(conf_csv, index=False)
            artifacts.append(str(conf_csv))

        # 5. Comprehensive Markdown report
        md_lines = []
        md_lines.append("# SMC XGBoost Model Report")
        md_lines.append("")
        md_lines.append(f"- **Symbol**: {symbol}")
        md_lines.append(f"- **Exchange**: {exchange}")
        md_lines.append(f"- **Timeframe**: {regime_timeframe}")
        md_lines.append(f"- **Generated**: {ts}")
        md_lines.append("")

        md_lines.append("## Model Performance")
        md_lines.append("")
        md_lines.append("| Metric | Value |")
        md_lines.append("| --- | --- |")

        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        md_lines.append(f"| RMSE | {rmse:.4f} |")
        md_lines.append(f"| R² Score | {r2:.4f} |")
        md_lines.append(f"| Samples | {len(y)} |")
        md_lines.append(f"| Features | {len(feature_cols)} |")
        md_lines.append("")

        md_lines.append("## Top 15 Features by Importance")
        md_lines.append("")
        md_lines.append("| Rank | Feature | Importance |")
        md_lines.append("| --- | --- | --- |")
        for i, row in importance_df.head(15).iterrows():
            md_lines.append(f"| {i+1} | {row['feature']} | {row['importance']:.4f} |")
        md_lines.append("")

        md_lines.append("## Feature Correlations with Target (Top 15)")
        md_lines.append("")
        md_lines.append("| Rank | Feature | Correlation |")
        md_lines.append("| --- | --- | --- |")
        for i, row in corr_df.head(15).iterrows():
            md_lines.append(f"| {i+1} | {row['feature']} | {row['correlation_with_target']:.4f} |")
        md_lines.append("")

        if calibration_results:
            md_lines.append("## Conformal Prediction Calibration")
            md_lines.append("")
            md_lines.append("| Confidence Level | Quantile |")
            md_lines.append("| --- | --- |")
            for level, q in calibration_results["calibration"]["quantiles"].items():
                md_lines.append(f"| {level} | {q:.4f} |")
            md_lines.append("")

        md_lines.append("## Prediction Statistics")
        md_lines.append("")
        md_lines.append("| Statistic | Predicted | Actual |")
        md_lines.append("| --- | --- | --- |")
        md_lines.append(f"| Mean | {y_pred.mean():.4f} | {y.mean():.4f} |")
        md_lines.append(f"| Std | {y_pred.std():.4f} | {y.std():.4f} |")
        md_lines.append(f"| Min | {y_pred.min():.4f} | {y.min():.4f} |")
        md_lines.append(f"| Max | {y_pred.max():.4f} | {y.max():.4f} |")
        md_lines.append("")

        # Directional accuracy
        breakout_pred = y_pred > 1.0
        breakout_actual = y > 1.0
        if breakout_pred.sum() > 0:
            breakout_acc = (breakout_actual[breakout_pred]).mean()
            md_lines.append(f"**Breakout Prediction Accuracy** (when model predicts > 1.0): {breakout_acc:.2%}")
            md_lines.append("")

        md_path = out_dir / f"smc_xgb_report_{symbol}_{regime_timeframe}_{ts}.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        artifacts.append(str(md_path))

        tprint_success(f"✅ Generated {len(artifacts)} report artifacts")

        return artifacts
