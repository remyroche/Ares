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
from src.features_common.transforms.scaling_normalization import (
    winsorized_zscore_normalize,
    rolling_winsorized_zscore_normalize,
)
from src.feature_generation.categories.cross_timeframe import CrossTimeframeFeatureGenerator
from sklearn.isotonic import IsotonicRegression
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
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
            direction = str(config.get("direction", "long"))

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            # Route all SMC artifacts into a dedicated SMC regime namespace
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="smc_regime",
            )

            tprint_info(
                f"🚀 Starting {self.step_name} (SMC Regime) for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # Load market data with caching (using BaseStep pattern)
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

            # Create temporal split config with 6-month burn-in for indicator stabilization
            split_config = create_temporal_split_config_for_pipeline(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                data_start=market_data.index.min(),
                data_end=market_data.index.max(),
                enable_burnin=True,
                # Use default burnin_pct=1/12 (3 months)
            )
            tprint_info(
                f"📊 Temporal split config created with burn-in: "
                f"Burn-in {split_config.burnin.start if split_config.burnin else 'N/A'} → "
                f"{split_config.burnin.effective_end if split_config.burnin else 'N/A'}, "
                f"Train {split_config.training.start} → {split_config.training.effective_end}, "
                f"Val {split_config.validation.start} → {split_config.validation.effective_end}, "
                f"Test {split_config.test.start} → {split_config.test.effective_end}"
            )

            # Generate SMC features with proper normalization
            smc_df = self._generate_smc_features(market_data, config)

            # Train XGBoost model if enabled
            if bool(config.get("smc_xgb_enable_training", True)):
                try:
                    xgb_metrics, xgb_artifacts = self._train_smc_xgb_model(
                        smc_df,
                        config,
                        split_config,
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

            # Save SMC features using BaseStep pattern
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
        """Generate all SMC features with proper normalization."""
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

        # 9. Optional cross-timeframe features from feature_generation categories
        if bool(config.get("smc_enable_cross_timeframe_features", True)):
            tprint_info("Generating cross-timeframe features...")
            result = self._add_cross_timeframe_features(result, config)

        # Apply normalization to features (except categorical/binary features)
        tprint_info("Normalizing SMC features...")
        result = self._normalize_smc_features(result, config)

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

        df["smc_pdh"] = pdh
        df["smc_pdl"] = pdl
        df["smc_dist_to_pdh_atr"] = (c.values - pdh) / (atr.values + 1e-9)
        df["smc_dist_to_pdl_atr"] = (c.values - pdl) / (atr.values + 1e-9)

        # Day open
        df["smc_day_open"] = day_open_vals
        df["smc_dist_to_day_open"] = (c.values - day_open_vals) / (atr.values + 1e-9)

        # Week open
        week_index = df.index.to_period('W').to_timestamp()
        weekly = df.groupby(week_index).agg(open=("open", "first"), close=("close", "last"))
        week_open = weekly["open"]
        prev_week_close = weekly["close"].shift(1)

        week_open_vals = week_open.reindex(week_index).to_numpy()
        prev_week_close_vals = prev_week_close.reindex(week_index).to_numpy()

        df["smc_week_open"] = week_open_vals
        df["smc_dist_to_week_open"] = (c.values - week_open_vals) / (atr.values + 1e-9)

        # New Week Opening Gap (NWOG)
        nwog_gap = week_open_vals - prev_week_close_vals
        df["smc_nwog_gap_size"] = nwog_gap / (atr.values + 1e-9)

        return df

    def _add_fvg_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add Fair Value Gap (FVG) and inefficiency features."""
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        o = df["open"].astype(float)
        atr = df["atr"].astype(float)

        # FVG detection
        high_2 = h.shift(2)
        low_2 = l.shift(2)

        bullish_fvg = l > high_2
        bearish_fvg = h < low_2

        # FVG size
        bullish_fvg_size = (l - high_2).clip(lower=0.0)
        bearish_fvg_size = (low_2 - h).clip(lower=0.0)

        fvg_size = bullish_fvg_size.where(bullish_fvg, bearish_fvg_size.where(bearish_fvg, 0.0))
        df["smc_current_fvg_size"] = fvg_size / (atr + 1e-9)

        # FVG midpoint
        bullish_fvg_mid = (l + high_2) / 2.0
        bearish_fvg_mid = (h + low_2) / 2.0

        fvg_mid = bullish_fvg_mid.where(bullish_fvg, bearish_fvg_mid.where(bearish_fvg, np.nan))
        fvg_mid_filled = fvg_mid.ffill()
        df["smc_nearest_fvg_dist"] = (c - fvg_mid_filled) / (atr + 1e-9)

        # Consequent encroachment
        fvg_high = l.where(bullish_fvg, low_2.where(bearish_fvg, np.nan))
        fvg_low = high_2.where(bullish_fvg, h.where(bearish_fvg, np.nan))

        fvg_range = fvg_high - fvg_low
        ce_position = (c - fvg_low) / (fvg_range + 1e-9)
        df["smc_consequent_encroachment"] = ce_position.fillna(0.5)

        # Volume imbalance
        volume_imb = (o - c.shift(1)).abs()
        df["smc_volume_imbalance_size"] = volume_imb / (atr + 1e-9)

        # Gap fill ratio
        fvg_fill = ((h - fvg_low) / (fvg_range + 1e-9)).clip(0.0, 1.0)
        df["smc_gap_fill_ratio"] = fvg_fill.fillna(0.0)

        return df

    def _add_premium_discount_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add premium/discount and structure features."""
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        atr = df["atr"].astype(float)

        # Swing highs and lows
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

        # Range position
        range_height = swing_high - swing_low
        range_pos = (c - swing_low) / (range_height + 1e-9)
        df["smc_range_position"] = range_pos.clip(0.0, 1.0)

        # Distance to swing high/low
        df["smc_dist_to_swing_high"] = (swing_high - c) / (atr + 1e-9)
        df["smc_dist_to_swing_low"] = (c - swing_low) / (atr + 1e-9)

        # Fibonacci retracement level
        fib_level = (swing_high - c) / (range_height + 1e-9)
        df["smc_fib_retracement_level"] = fib_level.clip(0.0, 1.0)

        # Break of structure magnitude
        prev_swing_high = swing_high.shift(1)
        bos_magnitude = (c - prev_swing_high) / (atr + 1e-9)
        df["smc_break_of_structure_mag"] = bos_magnitude.clip(lower=0.0)

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
        df["smc_displacement_strength"] = body / (avg_body + 1e-9)

        # Wick to body ratio
        upper_wick = h - np.maximum(c, o)
        lower_wick = np.minimum(c, o) - l
        total_wick = upper_wick + lower_wick
        df["smc_wick_body_ratio"] = total_wick / (body + 1e-9)

        # Close position in candle
        candle_range = h - l
        close_pos = (c - l) / (candle_range + 1e-9)
        df["smc_close_position_in_candle"] = close_pos.clip(0.0, 1.0)

        # Velocity / Rate of Change
        roc = (c - c.shift(3)) / 3.0
        df["smc_velocity_roc"] = roc / (c.shift(3) + 1e-9)

        # Consecutive candles
        candle_direction = np.sign(c - o)
        streaks = np.zeros(len(candle_direction))
        current_streak = 0
        for i in range(len(candle_direction)):
            if i == 0:
                current_streak = candle_direction.iloc[i]
            elif candle_direction.iloc[i] == candle_direction.iloc[i-1] and candle_direction.iloc[i] != 0:
                current_streak += candle_direction.iloc[i]
            else:
                current_streak = candle_direction.iloc[i]
            streaks[i] = current_streak

        df["smc_consecutive_candles"] = streaks

        return df

    def _add_volatility_time_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add volatility and time-based features."""
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        v = df["volume"].astype(float)
        atr = df["atr"].astype(float)

        # Average Daily Range
        day_index = df.index.normalize()
        daily_range = df.groupby(day_index).apply(lambda x: x["high"].max() - x["low"].min())
        adr = daily_range.rolling(window=20).mean()

        # Today's range filled percentage
        today_high = df.groupby(day_index)["high"].transform("max")
        today_low = df.groupby(day_index)["low"].transform("min")
        today_range = today_high - today_low
        adr_reindexed = adr.reindex(day_index).to_numpy()
        df["smc_adr_filled_pct"] = today_range / (adr_reindexed + 1e-9)

        # Relative volume (use log1p for volume normalization later)
        avg_vol = v.rolling(window=20).mean()
        df["smc_rel_volume"] = v / (avg_vol + 1e-9)

        # Time elapsed in session
        df["smc_time_elapsed_session"] = df.index.hour * 60 + df.index.minute

        # ATR compression
        atr_20 = atr.rolling(window=20).mean()
        df["smc_atr_compression"] = atr / (atr_20 + 1e-9)

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
                # HTF trend
                ema_20 = df_1h["close"].ewm(span=20).mean()
                ema_50 = df_1h["close"].ewm(span=50).mean()

                # HTF ATR
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
                df["smc_htf_trend_slope"] = htf_trend_slope_15m.fillna(0.0)
            else:
                df["smc_htf_trend_slope"] = 0.0

        except Exception as e:
            tprint_warning(f"MTF feature calculation failed: {e}")
            df["smc_htf_trend_slope"] = 0.0

        # Daily wick rejection
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
        df["smc_daily_wick_rejection"] = daily_wick_rej.reindex(day_index).fillna(0.0).to_numpy()

        return df

    def _add_cross_timeframe_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Optionally augment with cross-timeframe features from the shared generator."""
        try:
            generator = CrossTimeframeFeatureGenerator()
        except Exception as e:
            tprint_warning(f"SMC cross-timeframe feature generator init failed: {e}")
            return df

        base_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        if not base_cols:
            return df

        try:
            data = df[base_cols].copy()
            features = generator.generate_enhanced_cross_timeframe_features(data)
            if not features:
                return df

            for name, series in features.items():
                col_name = f"smc_ctf_{name}"
                ser = pd.Series(series)
                ser.index = data.index
                df[col_name] = ser.reindex(df.index).astype(float)

        except Exception as e:
            tprint_warning(f"SMC cross-timeframe feature generation failed: {e}")

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

                # Value area
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

        df["smc_hvn_gravity"] = hvn_gravity_list
        df["smc_poc_dist_atr"] = poc_dist_list
        df["smc_is_in_value_area"] = is_in_value_area_list
        df["smc_profile_skew"] = profile_skew_list

        return df

    def _add_time_categories(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add time category features."""

        # Session (Kill Zone)
        hour = df.index.hour

        session_kz = pd.Series("Dead", index=df.index)
        session_kz[(hour >= 0) & (hour < 8)] = "Asia"
        session_kz[(hour >= 8) & (hour < 13)] = "London"
        session_kz[(hour >= 13) & (hour < 17)] = "NY_AM"
        session_kz[(hour >= 17) & (hour < 21)] = "NY_PM"

        # One-hot encode sessions
        for session in ["Asia", "London", "NY_AM", "NY_PM", "Dead"]:
            df[f"smc_session_{session}"] = (session_kz == session).astype(int)

        # Day of week
        dow = df.index.dayofweek
        for day_num, day_name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            df[f"smc_dow_{day_name}"] = (dow == day_num).astype(int)

        # Market structure (simplified)
        h = df["high"].astype(float)
        l = df["low"].astype(float)

        hh = (h > h.shift(1)) & (h.shift(1) > h.shift(2))
        ll = (l < l.shift(1)) & (l.shift(1) < l.shift(2))
        lh = (h < h.shift(1)) & (h.shift(1) < h.shift(2))
        hl = (l > l.shift(1)) & (l.shift(1) > l.shift(2))

        market_structure = pd.Series("Range", index=df.index)
        market_structure[hh & hl] = "Uptrend"
        market_structure[lh & ll] = "Downtrend"

        df["smc_market_structure_Uptrend"] = (market_structure == "Uptrend").astype(int)
        df["smc_market_structure_Downtrend"] = (market_structure == "Downtrend").astype(int)
        df["smc_market_structure_Range"] = (market_structure == "Range").astype(int)

        # Inside FVG
        df["smc_is_inside_fvg"] = (df["smc_current_fvg_size"] > 0).astype(int)

        # Sweep confirmed
        c = df["close"].astype(float)
        pdh = df["smc_pdh"]
        pdl = df["smc_pdl"]

        sweep_high = (h > pdh) & (c < pdh)
        sweep_low = (l < pdl) & (c > pdl)
        df["smc_sweep_confirmed"] = (sweep_high | sweep_low).astype(int)

        return df

    def _normalize_smc_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply winsorized z-score normalization to continuous features, log1p to volume features."""

        # Identify continuous features (exclude binary/categorical and raw price levels)
        exclude_patterns = [
            "_session_", "_dow_", "_market_structure_", "_is_", "_confirmed",
            "smc_pdh", "smc_pdl", "smc_day_open", "smc_week_open", "atr",
            "open", "high", "low", "close", "volume"
        ]

        continuous_features = []
        volume_features = []

        for col in df.columns:
            if any(pattern in col for pattern in exclude_patterns):
                continue
            if col.startswith("smc_"):
                if "volume" in col.lower() or "rel_volume" in col.lower():
                    volume_features.append(col)
                else:
                    continuous_features.append(col)

        # Apply rolling winsorized z-score to continuous features to prevent look-ahead bias
        if continuous_features:
            window_size = int(config.get("smc_normalization_window", 500))
            tprint_info(f"Applying rolling winsorized z-score normalization to {len(continuous_features)} features (window={window_size})")
            for feat in continuous_features:
                try:
                    normalized = rolling_winsorized_zscore_normalize(
                        df[feat],
                        window=window_size,
                        min_periods=window_size // 2,
                        lower_quantile=0.05,
                        upper_quantile=0.95
                    )
                    df[feat] = normalized
                except Exception as e:
                    tprint_warning(f"Failed to normalize {feat}: {e}")

        # Apply log1p to volume features
        if volume_features:
            tprint_info(f"Applying log1p transformation to {len(volume_features)} volume features")
            for feat in volume_features:
                try:
                    df[feat] = np.log1p(df[feat].clip(lower=0.0))
                except Exception as e:
                    tprint_warning(f"Failed to log1p transform {feat}: {e}")

        return df

    def _train_smc_xgb_model(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        split_config: TemporalSplitConfig,
        symbol: str,
        exchange: str,
        regime_timeframe: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Train XGBoost classifier with HPO and conformal prediction calibration using temporal splits."""
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
        lookahead = int(config.get("smc_lookahead", 16))

        future_close = df["close"].shift(-lookahead)
        current_range = (df["high"] - df["low"]).replace(0, 0.0001)
        target_ratio = (future_close - df["low"]) / current_range

        df_with_target = df.copy()
        df_with_target["target_ratio"] = target_ratio

        # Calculate actual forward returns for metrics
        df_with_target["forward_return"] = (future_close / df["close"] - 1.0)

        df_with_target = df_with_target.dropna(subset=["target_ratio", "forward_return"])

        tprint_info(
            f"SMC XGB: target ratio stats - mean: {df_with_target['target_ratio'].mean():.3f}, "
            f"std: {df_with_target['target_ratio'].std():.3f}"
        )

        # Derive 3-class labels: 0=breakdown, 1=neutral, 2=breakout
        # More conservative defaults: require stronger moves for breakout/breakdown.
        breakout_threshold = float(config.get("smc_breakout_threshold", 1.50))
        breakdown_threshold = float(config.get("smc_breakdown_threshold", -0.50))

        ratio_clean = df_with_target["target_ratio"]
        target_class = np.where(
            ratio_clean > breakout_threshold,
            2,
            np.where(ratio_clean < breakdown_threshold, 0, 1),
        ).astype(np.int32)
        df_with_target["target_class"] = target_class

        class_counts = {
            "breakdown": int((target_class == 0).sum()),
            "neutral": int((target_class == 1).sum()),
            "breakout": int((target_class == 2).sum()),
        }
        tprint_info(
            "SMC XGB: class distribution (0=breakdown,1=neutral,2=breakout): "
            f"{class_counts}"
        )

        # Select features
        exclude_cols = [
            "target_ratio",
            "forward_return",
            "target_class",
            "smc_pdh",
            "smc_pdl",
            "smc_day_open",
            "smc_week_open",
        ]
        numeric_df = df_with_target.select_dtypes(include=[np.number])
        feature_cols = [col for col in numeric_df.columns if col not in exclude_cols and col.startswith("smc_")]

        if len(feature_cols) < 5:
            tprint_warning(f"SMC XGB: insufficient features (n={len(feature_cols)})")
            metrics["smc_xgb_early_exit_reason"] = f"insufficient_features_{len(feature_cols)}"
            return metrics, artifacts

        max_features = int(config.get("smc_xgb_max_features", 48))
        if len(feature_cols) > max_features:
            ratio_for_corr = df_with_target["target_ratio"]
            corr_scores: List[Tuple[str, float]] = []
            for col in feature_cols:
                try:
                    corr_val = numeric_df[col].corr(ratio_for_corr)
                except Exception:
                    corr_val = 0.0
                if corr_val is None or not np.isfinite(corr_val):
                    corr_val = 0.0
                corr_scores.append((col, float(abs(corr_val))))
            corr_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [name for name, _ in corr_scores[:max_features]]
            tprint_info(
                "SMC XGB: reducing feature set from "
                f"{len(feature_cols)} to {len(selected)} based on correlation with target_ratio"
            )
            feature_cols = selected

        X = numeric_df[feature_cols].astype(np.float32)
        y_ratio = df_with_target["target_ratio"].astype(np.float32)
        y_class = df_with_target["target_class"].astype(np.int32)
        forward_returns = df_with_target["forward_return"].astype(np.float32)

        # Handle infinities and NaNs
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)

        min_samples = int(config.get("smc_xgb_min_samples", 800))
        if len(X) < min_samples:
            tprint_warning(f"SMC XGB: insufficient samples (n={len(X)}, min={min_samples})")
            metrics["smc_xgb_early_exit_reason"] = f"insufficient_samples_{len(X)}"
            return metrics, artifacts

        # Time-series split into train / validation / test
        # Use temporal split config for proper train/val/test separation
        train_mask = (df_with_target.index >= split_config.training.start) & \
                     (df_with_target.index <= split_config.training.effective_end)
        val_mask = (df_with_target.index >= split_config.validation.start) & \
                   (df_with_target.index <= split_config.validation.effective_end)
        test_mask = (df_with_target.index >= split_config.test.start) & \
                    (df_with_target.index <= split_config.test.effective_end)

        X_train = X.loc[train_mask]
        X_val = X.loc[val_mask]
        X_test = X.loc[test_mask]

        y_train_ratio = y_ratio.loc[train_mask]
        y_val_ratio = y_ratio.loc[val_mask]
        y_test_ratio = y_ratio.loc[test_mask]

        y_train_cls = y_class.loc[train_mask]
        y_val_cls = y_class.loc[val_mask]
        y_test_cls = y_class.loc[test_mask]

        forward_returns_train = forward_returns.loc[train_mask]
        forward_returns_val = forward_returns.loc[val_mask]
        forward_returns_test = forward_returns.loc[test_mask]

        tprint_info(
            f"📊 Using temporal splits: Train {len(X_train)} samples "
            f"({split_config.training.start} → {split_config.training.effective_end}), "
            f"Val {len(X_val)} samples ({split_config.validation.start} → {split_config.validation.effective_end}), "
            f"Test {len(X_test)} samples ({split_config.test.start} → {split_config.test.effective_end}), "
            f"Features {len(feature_cols)}"
        )

        # HPO using Bayesian TPE optimizer (classification). Always enabled; the
        # _run_hpo helper handles falling back to sensible defaults if Optuna
        # is unavailable.
        tprint_info("Starting Bayesian TPE hyperparameter optimization (classification)...")
        best_params = self._run_hpo(X_train, y_train_cls, config)

        # Ensure core classification params
        best_params.setdefault("objective", "multi:softprob")
        best_params.setdefault("num_class", 3)
        best_params.setdefault("random_state", 42)
        best_params.setdefault("n_jobs", -1)

        # Train final classifier
        tprint_info("Training final XGBoost classifier with best params...")
        model = xgb.XGBClassifier(**best_params)

        model.fit(
            X_train,
            y_train_cls,
            eval_set=[(X_val, y_val_cls)] if len(X_val) > 0 else [(X_train, y_train_cls)],
            verbose=False,
        )

        # Predictions and probabilities
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
            accuracy_score,
            f1_score,
            log_loss,
            brier_score_loss,
        )

        proba_train = model.predict_proba(X_train)
        proba_val = model.predict_proba(X_val) if len(X_val) > 0 else np.empty((0, 3), dtype=np.float32)
        proba_test = model.predict_proba(X_test) if len(X_test) > 0 else np.empty((0, 3), dtype=np.float32)

        class_to_scalar = np.array([0.0, 0.5, 1.0], dtype=np.float32)

        y_train_pred = proba_train.dot(class_to_scalar)
        y_val_pred = proba_val.dot(class_to_scalar) if len(X_val) > 0 else np.array([], dtype=np.float32)
        y_test_pred = proba_test.dot(class_to_scalar) if len(X_test) > 0 else np.array([], dtype=np.float32)

        y_train_pred_cls = np.argmax(proba_train, axis=1)
        y_val_pred_cls = np.argmax(proba_val, axis=1) if len(X_val) > 0 else np.array([], dtype=np.int32)
        y_test_pred_cls = np.argmax(proba_test, axis=1) if len(X_test) > 0 else np.array([], dtype=np.int32)

        # Regression-style metrics vs continuous target_ratio for continuity
        train_rmse = np.sqrt(mean_squared_error(y_train_ratio, y_train_pred))
        train_mae = mean_absolute_error(y_train_ratio, y_train_pred)
        train_r2 = r2_score(y_train_ratio, y_train_pred)

        if len(X_val) > 0:
            val_rmse = np.sqrt(mean_squared_error(y_val_ratio, y_val_pred))
            val_mae = mean_absolute_error(y_val_ratio, y_val_pred)
            val_r2 = r2_score(y_val_ratio, y_val_pred)
        else:
            val_rmse = float("nan")
            val_mae = float("nan")
            val_r2 = float("nan")

        if len(X_test) > 0:
            test_rmse = np.sqrt(mean_squared_error(y_test_ratio, y_test_pred))
            test_mae = mean_absolute_error(y_test_ratio, y_test_pred)
            test_r2 = r2_score(y_test_ratio, y_test_pred)
        else:
            test_rmse = float("nan")
            test_mae = float("nan")
            test_r2 = float("nan")

        # Classification metrics
        train_accuracy = accuracy_score(y_train_cls, y_train_pred_cls)
        train_f1 = f1_score(y_train_cls, y_train_pred_cls, average="macro")
        train_logloss = log_loss(y_train_cls, proba_train, labels=[0, 1, 2])

        if len(X_val) > 0:
            val_accuracy = accuracy_score(y_val_cls, y_val_pred_cls)
            val_f1 = f1_score(y_val_cls, y_val_pred_cls, average="macro")
            val_logloss = log_loss(y_val_cls, proba_val, labels=[0, 1, 2])
        else:
            val_accuracy = float("nan")
            val_f1 = float("nan")
            val_logloss = float("nan")

        if len(X_test) > 0:
            test_accuracy = accuracy_score(y_test_cls, y_test_pred_cls)
            test_f1 = f1_score(y_test_cls, y_test_pred_cls, average="macro")
            test_logloss = log_loss(y_test_cls, proba_test, labels=[0, 1, 2])
        else:
            test_accuracy = float("nan")
            test_f1 = float("nan")
            test_logloss = float("nan")

        # Brier score for breakout event (class 2)
        try:
            event_train = (y_train_cls == 2).astype(int)
            train_brier = brier_score_loss(event_train, proba_train[:, 2])
        except Exception:
            train_brier = float("nan")

        if len(X_val) > 0:
            try:
                event_val = (y_val_cls == 2).astype(int)
                val_brier = brier_score_loss(event_val, proba_val[:, 2])
            except Exception:
                val_brier = float("nan")
        else:
            val_brier = float("nan")

        if len(X_test) > 0:
            try:
                event_test = (y_test_cls == 2).astype(int)
                test_brier = brier_score_loss(event_test, proba_test[:, 2])
            except Exception:
                test_brier = float("nan")
        else:
            test_brier = float("nan")

        # Directional IC vs forward returns
        def _safe_ic(pred, ret):
            arr_pred = np.asarray(pred, dtype=float)
            arr_ret = np.asarray(ret, dtype=float)
            mask = np.isfinite(arr_pred) & np.isfinite(arr_ret)
            if mask.sum() < 50:
                return float("nan")
            try:
                return float(np.corrcoef(arr_pred[mask], arr_ret[mask])[0, 1])
            except Exception:
                return float("nan")

        train_ic = _safe_ic(y_train_pred, forward_returns_train)
        val_ic = _safe_ic(y_val_pred, forward_returns_val) if len(X_val) > 0 else float("nan")
        test_ic = _safe_ic(y_test_pred, forward_returns_test) if len(X_test) > 0 else float("nan")

        metrics.update({
            "smc_xgb_train_rmse": float(train_rmse),
            "smc_xgb_val_rmse": float(val_rmse) if np.isfinite(val_rmse) else float("nan"),
            "smc_xgb_test_rmse": float(test_rmse),
            "smc_xgb_train_mae": float(train_mae),
            "smc_xgb_val_mae": float(val_mae) if np.isfinite(val_mae) else float("nan"),
            "smc_xgb_test_mae": float(test_mae),
            "smc_xgb_train_r2": float(train_r2),
            "smc_xgb_val_r2": float(val_r2) if np.isfinite(val_r2) else float("nan"),
            "smc_xgb_test_r2": float(test_r2),
            "smc_xgb_train_brier": float(train_brier) if not np.isnan(train_brier) else 0.0,
            "smc_xgb_val_brier": float(val_brier) if not np.isnan(val_brier) else 0.0,
            "smc_xgb_test_brier": float(test_brier) if not np.isnan(test_brier) else 0.0,
            "smc_xgb_train_accuracy": float(train_accuracy),
            "smc_xgb_val_accuracy": float(val_accuracy) if np.isfinite(val_accuracy) else float("nan"),
            "smc_xgb_test_accuracy": float(test_accuracy) if np.isfinite(test_accuracy) else float("nan"),
            "smc_xgb_train_f1": float(train_f1),
            "smc_xgb_val_f1": float(val_f1) if np.isfinite(val_f1) else float("nan"),
            "smc_xgb_test_f1": float(test_f1) if np.isfinite(test_f1) else float("nan"),
            "smc_xgb_train_logloss": float(train_logloss) if np.isfinite(train_logloss) else float("nan"),
            "smc_xgb_val_logloss": float(val_logloss) if np.isfinite(val_logloss) else float("nan"),
            "smc_xgb_test_logloss": float(test_logloss) if np.isfinite(test_logloss) else float("nan"),
            "smc_xgb_train_ic": float(train_ic) if not np.isnan(train_ic) else 0.0,
            "smc_xgb_val_ic": float(val_ic) if not np.isnan(val_ic) else 0.0,
            "smc_xgb_test_ic": float(test_ic) if not np.isnan(test_ic) else 0.0,
        })

        # Model acceptance based on validation R² or directional IC
        min_val_r2 = float(config.get("smc_min_val_r2", 0.03))
        min_val_ic = float(config.get("smc_min_val_directional_ic", 0.05))

        accept_by_r2 = np.isfinite(val_r2) and val_r2 >= min_val_r2
        accept_by_ic = not np.isnan(val_ic) and abs(val_ic) >= min_val_ic
        model_accepted = bool(accept_by_r2 or accept_by_ic)
        metrics["smc_xgb_model_accepted"] = model_accepted

        if not model_accepted:
            tprint_warning(
                "SMC XGB model rejected based on validation metrics: "
                f"val_r2={val_r2:.4f}, val_directional_ic={val_ic:.4f}"
            )

        tprint_success(
            "✅ XGBoost classifier trained: "
            f"test_rmse={test_rmse:.4f}, test_r2={test_r2:.4f}, "
            f"test_accuracy={test_accuracy:.4f}, test_brier={test_brier:.4f}, "
            f"val_directional_ic={val_ic:.4f}"
        )

        if model_accepted and bool(config.get("smc_enable_isotonic_calibration", True)):
            try:
                scalar_train = y_train_pred
                scalar_val = y_val_pred if len(X_val) > 0 else np.array([], dtype=np.float32)
                returns_train = forward_returns_train.to_numpy()
                returns_val = (
                    forward_returns_val.to_numpy() if len(X_val) > 0 else np.array([], dtype=np.float32)
                )

                scalar_all = np.concatenate([scalar_train, scalar_val])
                returns_all = np.concatenate([returns_train, returns_val])

                mask_iso = np.isfinite(scalar_all) & np.isfinite(returns_all)
                min_samples_iso = int(config.get("smc_iso_min_samples", 500))

                if mask_iso.sum() >= min_samples_iso:
                    ret_clip = float(config.get("smc_iso_return_clip", 0.10))
                    if ret_clip > 0.0:
                        target_iso = np.clip(returns_all[mask_iso], -ret_clip, ret_clip)
                    else:
                        target_iso = returns_all[mask_iso]

                    iso_model = IsotonicRegression(out_of_bounds="clip")
                    iso_model.fit(scalar_all[mask_iso], target_iso)
                    # Store the return clip on the model for use at runtime.
                    iso_model.return_clip_ = ret_clip

                    iso_rmse = float("nan")
                    iso_r2 = float("nan")

                    if len(y_test_pred) > 0:
                        scalar_test = y_test_pred
                        returns_test = forward_returns_test.to_numpy()
                        mask_test_iso = np.isfinite(scalar_test) & np.isfinite(returns_test)
                        if mask_test_iso.sum() >= 50:
                            pred_test_iso = iso_model.predict(scalar_test[mask_test_iso])
                            iso_rmse = float(
                                np.sqrt(mean_squared_error(returns_test[mask_test_iso], pred_test_iso))
                            )
                            try:
                                iso_r2 = float(r2_score(returns_test[mask_test_iso], pred_test_iso))
                            except Exception:
                                iso_r2 = float("nan")

                    metrics["smc_iso_test_rmse"] = iso_rmse
                    metrics["smc_iso_test_r2"] = iso_r2

                    iso_path = self._save_artifact(
                        data=iso_model,
                        artifact_name="smc_scalar_isotonic_calibrator",
                        artifact_type="model",
                        data_category="models",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "return_clip": ret_clip,
                            "n_calibration_samples": int(mask_iso.sum()),
                        },
                    )
                    artifacts.append(iso_path)
            except Exception as e:
                tprint_warning(f"Isotonic regression calibration failed: {e}")

        # Conformal prediction calibration on breakout probability
        tprint_info("Performing conformal prediction calibration...")
        non_test_X = X_train
        train_bin = (y_train_cls == 2).astype(int)
        if len(X_val) > 0:
            non_test_X = pd.concat([X_train, X_val])
            non_test_y = pd.concat([
                pd.Series(train_bin, index=X_train.index),
                pd.Series((y_val_cls == 2).astype(int), index=X_val.index),
            ])
        else:
            non_test_y = pd.Series(train_bin, index=X_train.index)

        y_test_bin = pd.Series((y_test_cls == 2).astype(int), index=X_test.index)

        class _ProbaWrapper:
            def __init__(self, base_model):
                self._base_model = base_model

            def predict(self, X_input):
                proba = self._base_model.predict_proba(X_input)
                if proba.shape[1] == 0:
                    return np.zeros(len(X_input), dtype=float)
                return proba[:, -1]

        calibration_results = self._calibrate_conformal_prediction(
            _ProbaWrapper(model),
            non_test_X,
            non_test_y,
            X_test,
            y_test_bin,
        )

        if calibration_results:
            metrics.update(calibration_results["metrics"])

        # Generate comprehensive reports
        tprint_info("Generating comprehensive reports...")
        report_artifacts = self._generate_smc_reports(
            df_with_target,
            model,
            X,
            y_ratio,
            forward_returns,
            feature_cols,
            calibration_results,
            symbol,
            exchange,
            regime_timeframe,
            train_rmse,
            val_rmse,
            test_rmse,
            train_r2,
            val_r2,
            test_r2,
            train_brier,
            val_brier,
            test_brier,
            len(X_train),
            len(X_val),
            len(X_test),
        )
        artifacts.extend(report_artifacts)

        # Save model using BaseStep
        tprint_info("Saving XGBoost model...")
        # Prepare metadata with temporal split information
        model_metadata = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": regime_timeframe,
            "n_features": len(feature_cols),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
            "version": "v1_with_burnin",
            "training_start": str(split_config.training.start),
            "training_end": str(split_config.training.effective_end),
            "validation_start": str(split_config.validation.start),
            "validation_end": str(split_config.validation.effective_end),
            "test_start": str(split_config.test.start),
            "test_end": str(split_config.test.effective_end),
        }
        if split_config.burnin is not None:
            model_metadata["burnin_start"] = str(split_config.burnin.start)
            model_metadata["burnin_end"] = str(split_config.burnin.effective_end)

        model_path = self._save_artifact(
            data=model,
            artifact_name="smc_xgb_model",
            artifact_type="model",
            data_category="models",
            metadata=model_metadata,
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

        # Save predictions using BaseStep. The artifact is shaped to mirror what
        # the live model will expose: a single scalar score in [0,1] per bar.
        predictions_list = []
        index_array = df_with_target.index

        n_train_samples = len(y_train_ratio)
        n_val_samples = len(y_val_ratio)
        n_test_samples = len(y_test_ratio)

        for i in range(n_train_samples):
            idx = i
            predictions_list.append(
                {
                    "timestamp": index_array[idx],
                    "predicted": float(y_train_pred[i]),
                }
            )

        offset_val = n_train_samples
        for i in range(n_val_samples):
            idx = offset_val + i
            predictions_list.append(
                {
                    "timestamp": index_array[idx],
                    "predicted": float(y_val_pred[i]),
                }
            )

        offset_test = n_train_samples + n_val_samples
        for i in range(n_test_samples):
            idx = offset_test + i
            predictions_list.append(
                {
                    "timestamp": index_array[idx],
                    "predicted": float(y_test_pred[i]),
                }
            )

        predictions_df = pd.DataFrame(predictions_list)

        predictions_path = self._save_artifact(
            data=predictions_df,
            artifact_name="smc_predictions_with_confidence",
            artifact_type="data",
            data_category="predictions",
            metadata=model_metadata,  # Reuse metadata with temporal split info
        )
        artifacts.append(predictions_path)

        # Optional lightweight walk-forward validation for stability diagnostics
        try:
            wf_metrics = self._run_smc_walkforward_validation(
                X,
                y_ratio,
                best_params,
                config,
            )
            if wf_metrics:
                metrics["smc_xgb_walkforward"] = wf_metrics
        except Exception as e:
            tprint_warning(f"SMC XGB walk-forward validation failed: {e}")

        return metrics, artifacts

    def _run_smc_walkforward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_params: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simple expanding-window walk-forward validation for stability diagnostics.

        Trains lightweight models on expanding prefixes of the data and evaluates
        on forward test windows, returning per-fold and aggregate metrics.
        """
        try:
            import xgboost as xgb
        except Exception:
            return {}

        n = int(len(X))
        if n < 1200:
            return {}

        n_folds = int(config.get("smc_xgb_walkforward_folds", 3))
        n_folds = max(2, min(n_folds, 6))

        min_train = int(config.get("smc_xgb_walkforward_min_train", 800))
        min_test = int(config.get("smc_xgb_walkforward_min_test", 200))

        if n < min_train + min_test:
            return {}

        fold_metrics: List[Dict[str, float]] = []

        from sklearn.metrics import mean_squared_error, r2_score

        # Determine step size for expanding window
        remaining = n - (min_train + min_test)
        step = remaining // max(n_folds - 1, 1) if remaining > 0 else 0

        for k in range(n_folds):
            train_end = min_train + k * step
            test_end = train_end + min_test

            if test_end > n:
                break

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test_fold = X.iloc[train_end:test_end]
            y_test_fold = y.iloc[train_end:test_end]

            if len(X_test_fold) < max(50, min_test // 2):
                continue

            try:
                wf_params = dict(base_params)
                wf_params.pop("num_class", None)
                wf_params["objective"] = "reg:squarederror"

                model = xgb.XGBRegressor(**wf_params)
                model.fit(X_train, y_train, verbose=False)
                y_pred_fold = model.predict(X_test_fold)

                rmse_fold = float(np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)))
                r2_fold = float(r2_score(y_test_fold, y_pred_fold)) if len(y_test_fold) > 1 else float("nan")

                fold_metrics.append(
                    {
                        "fold": k,
                        "train_end_index": int(train_end),
                        "test_start_index": int(train_end),
                        "test_end_index": int(test_end),
                        "test_rmse": rmse_fold,
                        "test_r2": r2_fold,
                    }
                )
            except Exception as e:
                tprint_warning(f"SMC XGB walk-forward fold {k} failed: {e}")
                continue

        if not fold_metrics:
            return {}

        test_rmses = np.array([m["test_rmse"] for m in fold_metrics], dtype=float)
        test_r2s = np.array([m["test_r2"] for m in fold_metrics], dtype=float)

        summary = {
            "n_folds": len(fold_metrics),
            "mean_test_rmse": float(np.nanmean(test_rmses)),
            "std_test_rmse": float(np.nanstd(test_rmses)),
            "mean_test_r2": float(np.nanmean(test_r2s)),
            "std_test_r2": float(np.nanstd(test_r2s)),
            "folds": fold_metrics,
        }

        return summary

    def _run_hpo(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Bayesian TPE hyperparameter optimization for multi-class classification."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            tprint_warning("Optuna not available, using default params")
            return {
                "objective": "multi:softprob",
                "num_class": 3,
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 4,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "gamma": 0.5,
                "reg_alpha": 1.0,
                "reg_lambda": 2.0,
                "min_child_weight": 15,
                "random_state": 42,
                "n_jobs": -1,
            }

        # Split for validation
        val_frac = 0.2
        split_idx = int(len(X_train) * (1 - val_frac))
        X_tr = X_train.iloc[:split_idx]
        y_tr = y_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]

        def objective(trial):
            import xgboost as xgb
            from sklearn.metrics import log_loss, f1_score

            # Emphasize directional IC more strongly by default.
            ic_weight = float(config.get("smc_hpo_ic_weight", 0.9))

            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "n_estimators": trial.suggest_int("n_estimators", 300, 900),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "gamma": trial.suggest_float("gamma", 0.1, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 3.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
                "random_state": 42,
                "n_jobs": -1,
            }

            try:
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                proba_val = model.predict_proba(X_val)

                # Base objective: minimize log loss (multi-class)
                loss = log_loss(y_val, proba_val, labels=[0, 1, 2])

                # Lightly regularize against low F1 by adding a penalty
                y_pred_cls = np.argmax(proba_val, axis=1)
                f1 = f1_score(y_val, y_pred_cls, average="macro")
                penalty = max(0.0, 0.6 - f1)

                # Directional IC-like term on scalarized predictions vs label direction
                class_to_scalar = np.array([0.0, 0.5, 1.0], dtype=np.float32)
                pred_scalar = proba_val.dot(class_to_scalar)
                y_dir = np.where(y_val == 2, 1.0, np.where(y_val == 0, -1.0, 0.0)).astype(np.float32)

                mask = np.isfinite(pred_scalar) & np.isfinite(y_dir)
                if mask.sum() > 50:
                    try:
                        ic = float(np.corrcoef(pred_scalar[mask], y_dir[mask])[0, 1])
                    except Exception:
                        ic = 0.0
                else:
                    ic = 0.0

                # Minimize logloss + F1 penalty - directional IC term
                return loss + penalty - ic_weight * ic

            except Exception as e:
                tprint_warning(f"HPO trial failed: {e}")
                return float("inf")

        # Create and run study
        n_trials = int(config.get("smc_xgb_hpo_trials", 30))
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        tprint_info(f"HPO completed: best objective value={study.best_value:.4f}")

        best_params = study.best_params
        best_params.update(
            {
                "objective": "multi:softprob",
                "num_class": 3,
                "random_state": 42,
                "n_jobs": -1,
            }
        )

        return best_params

    def _calibrate_conformal_prediction(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """Perform conformal prediction calibration with extended confidence levels."""
        try:
            cal_frac = 0.2
            split_idx = int(len(X_train) * (1 - cal_frac))

            X_cal = X_train.iloc[split_idx:]
            y_cal = y_train.iloc[split_idx:]

            if len(X_cal) == 0:
                return None

            y_cal_pred = model.predict(X_cal)
            nonconformity_scores = np.abs(y_cal - y_cal_pred)

            confidence_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
            quantiles = {}
            coverage_metrics = {}

            for alpha in confidence_levels:
                q = np.percentile(nonconformity_scores, alpha * 100)
                key = f"{int(alpha * 100)}%"
                quantiles[key] = float(q)
                coverage = float((nonconformity_scores <= q).mean())
                coverage_metrics[f"conformal_coverage_{key}"] = coverage

            tprint_info(f"Conformal prediction quantiles: {quantiles}")

            all_X = pd.concat([X_train, X_test])
            all_preds = model.predict(all_X)

            confidence_scores = {}
            for level, q in quantiles.items():
                conf_score = q / (np.abs(all_preds - 0.5) + q + 1e-9)
                confidence_scores[level] = conf_score.clip(0.0, 1.0)

            metrics_dict = {
                "conformal_quantile_50": quantiles.get("50%", float("nan")),
                "conformal_quantile_70": quantiles.get("70%", float("nan")),
                "conformal_quantile_90": quantiles.get("90%", float("nan")),
            }
            metrics_dict.update(coverage_metrics)

            return {
                "calibration": {
                    "quantiles": quantiles,
                    "nonconformity_scores": nonconformity_scores.tolist(),
                },
                "confidence_scores": confidence_scores,
                "metrics": metrics_dict,
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
        forward_returns: pd.Series,
        feature_cols: List[str],
        calibration_results: Optional[Dict[str, Any]],
        symbol: str,
        exchange: str,
        regime_timeframe: str,
        train_rmse: float,
        val_rmse: float,
        test_rmse: float,
        train_r2: float,
        val_r2: float,
        test_r2: float,
        train_brier: float,
        val_brier: float,
        test_brier: float,
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> List[str]:
        """Generate consolidated comprehensive reports."""
        artifacts = []

        out_dir = Path("outcomes")
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Predictions (scalarized classifier output 0.0/0.5/1.0 where possible)
        try:
            proba_all = model.predict_proba(X)
            if proba_all.shape[1] >= 3:
                class_to_scalar = np.array([0.0, 0.5, 1.0], dtype=np.float32)
                y_pred_scalar = proba_all.dot(class_to_scalar)
                y_pred_class = np.argmax(proba_all, axis=1)
            else:
                y_pred_scalar = model.predict(X)
                y_pred_class = None
        except Exception:
            y_pred_scalar = model.predict(X)
            y_pred_class = None

        # 1. CONSOLIDATED CSV: Feature importance + correlations
        feature_importance = model.feature_importances_

        feature_data = []
        for i, feat in enumerate(feature_cols):
            corr = X[feat].corr(y) if feat in X.columns else 0.0
            feature_data.append({
                "feature": feat,
                "importance": float(feature_importance[i]) if i < len(feature_importance) else 0.0,
                "correlation_with_target": float(corr) if not np.isnan(corr) else 0.0,
            })

        features_df = pd.DataFrame(feature_data).sort_values("importance", ascending=False)

        features_csv = out_dir / f"smc_features_analysis_{symbol}_{regime_timeframe}_{ts}.csv"
        features_df.to_csv(features_csv, index=False)
        artifacts.append(str(features_csv))

        # 2. COMPREHENSIVE MARKDOWN REPORT with integrated distributions
        md_lines = []
        md_lines.append("# SMC XGBoost Model Report")
        md_lines.append("")
        md_lines.append(f"- **Symbol**: {symbol}")
        md_lines.append(f"- **Exchange**: {exchange}")
        md_lines.append(f"- **Timeframe**: {regime_timeframe}")
        md_lines.append(f"- **Generated**: {ts}")
        md_lines.append("")

        # Model Performance
        md_lines.append("## Model Performance")
        md_lines.append("")
        md_lines.append("| Metric | Train | Val | Test |")
        md_lines.append("| --- | --- | --- | --- |")
        md_lines.append(
            f"| RMSE | {train_rmse:.4f} | {val_rmse:.4f} | {test_rmse:.4f} |"
        )
        md_lines.append(
            f"| R² Score | {train_r2:.4f} | {val_r2:.4f} | {test_r2:.4f} |"
        )
        md_lines.append(
            f"| Brier Score | {train_brier:.4f} | {val_brier:.4f} | {test_brier:.4f} |"
        )
        md_lines.append(
            f"| Samples | {n_train} | {n_val} | {n_test} |"
        )
        md_lines.append(f"| Features | {len(feature_cols)} | {len(feature_cols)} | {len(feature_cols)} |")
        md_lines.append("")

        # Mean returns by signal
        if y_pred_class is not None:
            breakout_pred = y_pred_class == 2
            breakdown_pred = y_pred_class == 0
        else:
            breakout_pred = y_pred_scalar > 0.75
            breakdown_pred = y_pred_scalar < 0.25
        if breakout_pred.sum() > 0:
            mean_ret_breakout = forward_returns[breakout_pred].mean()
            md_lines.append(f"**Mean Return (Breakout Signals)**: {mean_ret_breakout:.4f} ({breakout_pred.sum()} signals)")
        if breakdown_pred.sum() > 0:
            mean_ret_breakdown = forward_returns[breakdown_pred].mean()
            md_lines.append(f"**Mean Return (Breakdown Signals)**: {mean_ret_breakdown:.4f} ({breakdown_pred.sum()} signals)")
        md_lines.append("")

        # Top features
        md_lines.append("## Top 15 Features by Importance")
        md_lines.append("")
        md_lines.append("| Rank | Feature | Importance | Correlation |")
        md_lines.append("| --- | --- | --- | --- |")
        for i, row in features_df.head(15).iterrows():
            md_lines.append(f"| {i+1} | {row['feature']} | {row['importance']:.4f} | {row['correlation_with_target']:.4f} |")
        md_lines.append("")

        # Conformal prediction
        if calibration_results:
            md_lines.append("## Conformal Prediction Calibration")
            md_lines.append("")
            md_lines.append("Prediction intervals for uncertainty quantification:")
            md_lines.append("")
            md_lines.append("| Confidence Level | Quantile |")
            md_lines.append("| --- | --- |")
            for level, q in calibration_results["calibration"]["quantiles"].items():
                md_lines.append(f"| {level} | ±{q:.4f} |")
            md_lines.append("")

        # Prediction distribution (integrated)
        md_lines.append("## Prediction Distribution Analysis")
        md_lines.append("")
        md_lines.append("| Statistic | Predicted | Actual |")
        md_lines.append("| --- | --- | --- |")
        md_lines.append(f"| Mean | {y_pred_scalar.mean():.4f} | {y.mean():.4f} |")
        md_lines.append(f"| Std | {y_pred_scalar.std():.4f} | {y.std():.4f} |")
        md_lines.append(f"| Min | {y_pred_scalar.min():.4f} | {y.min():.4f} |")
        md_lines.append(
            f"| 25th Percentile | {np.percentile(y_pred_scalar, 25):.4f} | {np.percentile(y, 25):.4f} |"
        )
        md_lines.append(f"| Median | {np.median(y_pred_scalar):.4f} | {np.median(y):.4f} |")
        md_lines.append(
            f"| 75th Percentile | {np.percentile(y_pred_scalar, 75):.4f} | {np.percentile(y, 75):.4f} |"
        )
        md_lines.append(f"| Max | {y_pred_scalar.max():.4f} | {y.max():.4f} |")
        md_lines.append("")

        # Scalar band performance (deciles of scalar prediction)
        md_lines.append("## Scalar Band Performance (Deciles)")
        md_lines.append("")
        try:
            arr_pred = np.asarray(y_pred_scalar, dtype=float)
            arr_ret = np.asarray(forward_returns.to_numpy(), dtype=float)
            mask = np.isfinite(arr_pred) & np.isfinite(arr_ret)
            if mask.sum() >= 100:
                band_df = pd.DataFrame({"pred": arr_pred[mask], "ret": arr_ret[mask]})
                band_df["band"] = pd.qcut(band_df["pred"], q=10, duplicates="drop")

                def _band_stats(g: pd.DataFrame) -> pd.Series:
                    pred_vals = g["pred"].to_numpy()
                    ret_vals = g["ret"].to_numpy()
                    direction = np.sign(pred_vals - 0.5)
                    ret_sign = np.sign(ret_vals)
                    dir_mask = direction != 0
                    if dir_mask.any():
                        hit_rate = float(
                            (direction[dir_mask] * ret_sign[dir_mask] > 0).mean()
                        )
                    else:
                        hit_rate = float("nan")
                    return pd.Series(
                        {
                            "count": int(len(g)),
                            "pred_mean": float(pred_vals.mean()),
                            "ret_mean": float(ret_vals.mean()),
                            "hit_rate": hit_rate,
                        }
                    )

                stats = band_df.groupby("band").apply(_band_stats)

                md_lines.append(
                    "| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |"
                )
                md_lines.append("| --- | --- | --- | --- | --- |")
                for band_label, row in stats.iterrows():
                    md_lines.append(
                        f"| {band_label} | {int(row['count'])} | "
                        f"{row['pred_mean']:.4f} | {row['ret_mean']:.4f} | "
                        f"{row['hit_rate']:.2%} |"
                    )
                md_lines.append("")
            else:
                md_lines.append("_Not enough samples for scalar band analysis._")
                md_lines.append("")
        except Exception:
            md_lines.append("_Scalar band performance analysis failed; see logs for details._")
            md_lines.append("")

        # Directional accuracy
        breakout_actual = y > 1.0
        breakdown_actual = y < 0.0
        if breakout_pred.sum() > 0:
            breakout_acc = (breakout_actual[breakout_pred]).mean()
            md_lines.append(f"**Breakout Prediction Accuracy**: {breakout_acc:.2%} ({breakout_pred.sum()} predictions)")
        if breakdown_pred.sum() > 0:
            breakdown_acc = (breakdown_actual[breakdown_pred]).mean()
            md_lines.append(f"**Breakdown Prediction Accuracy**: {breakdown_acc:.2%} ({breakdown_pred.sum()} predictions)")
        md_lines.append("")

        # Confidence analysis (integrated)
        if calibration_results:
            md_lines.append("## Confidence Score Analysis")
            md_lines.append("")
            md_lines.append("| Confidence Level | Mean Score | Std Score |")
            md_lines.append("| --- | --- | --- |")
            for level, scores in calibration_results["confidence_scores"].items():
                mean_conf = np.mean(scores)
                std_conf = np.std(scores)
                md_lines.append(f"| {level} | {mean_conf:.4f} | {std_conf:.4f} |")
            md_lines.append("")

        md_path = out_dir / f"smc_xgb_report_{symbol}_{regime_timeframe}_{ts}.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        artifacts.append(str(md_path))

        tprint_success(f"✅ Generated {len(artifacts)} report artifacts")

        return artifacts
