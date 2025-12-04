"""
Level Generators for Market Structure Analysis

This module provides pluggable strategies for generating Support & Resistance (SR) levels
from OHLCV data.

Strategies:
1. RollingKDELevelGenerator: Uses Kernel Density Estimation on swing points and volume nodes.
2. FractalLevelGenerator: Uses discrete swing highs/lows (fractals).
3. HTFLevelGenerator: Uses High-Timeframe (Daily/Weekly) pivots.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, peak_prominences, argrelextrema

class BaseLevelGenerator:
    """Base interface for level generators."""

    def compute_levels(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute levels for the entire dataframe.

        Returns:
            DataFrame aligned with input index containing:
            - primary_level_price
            - primary_level_type ('resistance', 'support')
            - primary_level_source
            - opposing_level_price
            - opposing_level_type
            ... and optional strength metrics.
        """
        raise NotImplementedError


class RollingKDELevelGenerator(BaseLevelGenerator):
    """
    Generates levels using Rolling Kernel Density Estimation (KDE) on swing points and volume nodes.
    Good for finding 'clusters' of price action.
    """
    def __init__(
        self,
        lookback_days: int = 30,
        peaks_per_side: int = 3,
        price_grid_size: int = 400,
        min_history_bars: int = 200,
        sr_min_touch_count: int = 2,
        sr_min_volume_depth_ratio: float = 0.8,
        sr_min_prominence: float = 0.5,
    ) -> None:
        self.lookback_days = int(max(1, lookback_days))
        self.peaks_per_side = int(max(1, peaks_per_side))
        self.price_grid_size = int(max(100, price_grid_size))
        self.min_history_bars = int(max(50, min_history_bars))
        self.sr_min_touch_count = int(max(1, sr_min_touch_count))
        self.sr_min_volume_depth_ratio = float(sr_min_volume_depth_ratio)
        self.sr_min_prominence = float(sr_min_prominence)

    def _compute_pivots(
        self,
        highs: pd.Series,
        lows: pd.Series,
        period: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        arr_high = np.asarray(highs, dtype=float)
        arr_low = np.asarray(lows, dtype=float)
        n = len(arr_high)
        if n < 2 * period + 1:
            return np.array([], dtype=int), np.array([], dtype=int)

        pivot_high_idx: List[int] = []
        pivot_low_idx: List[int] = []

        for i in range(period, n - period):
            h = arr_high[i]
            l = arr_low[i]
            window_high = arr_high[i - period : i + period + 1]
            window_low = arr_low[i - period : i + period + 1]
            if h >= window_high.max() and h > window_high[period - 1] and h >= window_high[period + 1]:
                pivot_high_idx.append(i)
            if l <= window_low.min() and l < window_low[period - 1] and l <= window_low[period + 1]:
                pivot_low_idx.append(i)

        return np.asarray(pivot_high_idx, dtype=int), np.asarray(pivot_low_idx, dtype=int)

    def _build_kde_levels(
        self,
        prices: np.ndarray,
        kind: str,
    ) -> List[Dict[str, Any]]:
        prices = np.asarray(prices, dtype=float)
        prices = prices[np.isfinite(prices)]
        if prices.size < 10:
            return []

        p_min = float(prices.min())
        p_max = float(prices.max())
        if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
            return []

        grid = np.linspace(p_min, p_max, self.price_grid_size)
        try:
            kde = gaussian_kde(prices)
            density = kde(grid)
        except Exception:
            return []

        try:
            peak_idx, _ = find_peaks(density)
        except Exception:
            return []

        if peak_idx.size == 0:
            return []

        try:
            prominences, _, _ = peak_prominences(density, peak_idx)
        except Exception:
            prominences = np.ones_like(peak_idx, dtype=float)

        levels: List[Dict[str, Any]] = []
        for idx, prom in zip(peak_idx, prominences):
            price = float(grid[int(idx)])
            dens = float(density[int(idx)])
            levels.append(
                {
                    "price": price,
                    "density": dens,
                    "prominence": float(prom),
                    "source_type": kind,
                }
            )
        return levels

    def _compute_level_stats(
        self,
        level_price: float,
        history: pd.DataFrame,
    ) -> Tuple[int, Optional[pd.Timestamp], Optional[pd.Timestamp], float]:
        if history.empty:
            return 0, None, None, float("nan")

        high = np.asarray(history["high"], dtype=float)
        low = np.asarray(history["low"], dtype=float)
        vol = np.asarray(history["volume"], dtype=float)
        tol = level_price * 0.001
        mask = (np.abs(high - level_price) <= tol) | (np.abs(low - level_price) <= tol)
        touch_count = int(mask.sum())

        if touch_count > 0:
            idx = history.index[mask]
            first_ts = idx[0]
            last_ts = idx[-1]
            vol_at_level = float(np.nanmean(vol[mask]))
        else:
            first_ts = None
            last_ts = None
            vol_at_level = float("nan")

        median_vol = float(np.nanmedian(vol)) if vol.size > 0 else float("nan")
        if np.isfinite(vol_at_level) and np.isfinite(median_vol) and median_vol > 0.0:
            depth_ratio = vol_at_level / median_vol
        else:
            depth_ratio = float("nan")

        return touch_count, first_ts, last_ts, depth_ratio

    def compute_levels(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise ValueError("RollingKDELevelGenerator requires DatetimeIndex input")

        data = ohlcv.sort_index()
        index = data.index
        if data.empty:
            return pd.DataFrame(index=index)

        dates = index.normalize()
        unique_days = dates.unique()
        result = pd.DataFrame(
            index=index,
            data={
                "primary_level_price": np.nan,
                "primary_level_type": np.nan,
                "primary_level_source": np.nan,
                "primary_level_touch_count": np.nan,
                "primary_level_first_touch_ts": pd.NaT,
                "primary_level_last_touch_ts": pd.NaT,
                "primary_level_prominence": np.nan,
                "primary_level_volume_depth_ratio": np.nan,
                "opposing_level_price": np.nan,
                "opposing_level_type": np.nan,
                "opposing_level_source": np.nan,
                "opposing_level_touch_count": np.nan,
                "opposing_level_first_touch_ts": pd.NaT,
                "opposing_level_last_touch_ts": pd.NaT,
                "opposing_level_prominence": np.nan,
                "opposing_level_volume_depth_ratio": np.nan,
            },
        )

        if len(unique_days) <= 1:
            return result

        for day_idx in range(1, len(unique_days)):
            day_start = unique_days[day_idx]
            window_start = day_start - pd.Timedelta(days=self.lookback_days)
            history_mask = (index >= window_start) & (index < day_start)
            history = data.loc[history_mask]
            if history.shape[0] < self.min_history_bars:
                continue

            pivot_high_idx, pivot_low_idx = self._compute_pivots(
                history["high"], history["low"], period=3
            )

            swing_high_prices: np.ndarray
            swing_low_prices: np.ndarray
            if pivot_high_idx.size:
                swing_high_prices = history["high"].to_numpy()[pivot_high_idx]
            else:
                swing_high_prices = history["high"].to_numpy()

            if pivot_low_idx.size:
                swing_low_prices = history["low"].to_numpy()[pivot_low_idx]
            else:
                swing_low_prices = history["low"].to_numpy()

            vol = history["volume"].to_numpy()
            closes = history["close"].to_numpy()
            if vol.size > 0:
                vol_threshold = float(np.nanpercentile(vol, 80.0))
                vol_mask = vol >= vol_threshold
                volume_node_prices = closes[vol_mask]
            else:
                volume_node_prices = np.array([], dtype=float)

            candidate_levels: List[Dict[str, Any]] = []
            candidate_levels.extend(self._build_kde_levels(swing_high_prices, "swing_high"))
            candidate_levels.extend(self._build_kde_levels(swing_low_prices, "swing_low"))
            candidate_levels.extend(self._build_kde_levels(volume_node_prices, "volume_node"))

            if not candidate_levels:
                continue

            for level in candidate_levels:
                touch_count, first_ts, last_ts, depth_ratio = self._compute_level_stats(
                    float(level["price"]), history
                )
                level["touch_count"] = touch_count
                level["first_touch_ts"] = first_ts
                level["last_touch_ts"] = last_ts
                level["volume_depth_ratio"] = depth_ratio

            # Filter levels by quality/strength
            filtered_levels: List[Dict[str, Any]] = []
            for level in candidate_levels:
                touch_count = level.get("touch_count", 0)
                volume_depth = level.get("volume_depth_ratio", 0.0)
                prominence = level.get("prominence", 0.0)

                min_touches = int(self.sr_min_touch_count)
                min_volume_depth = float(self.sr_min_volume_depth_ratio)
                min_prominence = float(self.sr_min_prominence)

                if (
                    touch_count >= min_touches
                    and volume_depth >= min_volume_depth
                    and prominence >= min_prominence
                ):
                    filtered_levels.append(level)

            if not filtered_levels:
                # Fallback to median quality
                qualities: List[float] = []
                for level in candidate_levels:
                    touch_count = float(level.get("touch_count", 0.0))
                    volume_depth = float(level.get("volume_depth_ratio", 0.0))
                    prominence = float(level.get("prominence", 0.0))
                    qualities.append(touch_count + volume_depth + prominence)

                if qualities:
                    threshold = float(np.nanpercentile(qualities, 50.0))
                    for level, quality in zip(candidate_levels, qualities):
                        if quality >= threshold:
                            filtered_levels.append(level)

            if not filtered_levels:
                continue

            day_mask = dates == day_start
            day_index = index[day_mask]
            if day_index.empty:
                continue

            for ts in day_index:
                close_price = float(data.at[ts, "close"])
                above: List[Tuple[Dict[str, Any], float]] = []
                below: List[Tuple[Dict[str, Any], float]] = []
                for level in filtered_levels:
                    lp = float(level["price"])
                    dist = abs(lp - close_price)
                    if lp >= close_price:
                        above.append((level, dist))
                    else:
                        below.append((level, dist))

                above_sorted = sorted(above, key=lambda x: x[1])
                below_sorted = sorted(below, key=lambda x: x[1])

                primary_level: Optional[Dict[str, Any]] = None
                opposing_level: Optional[Dict[str, Any]] = None

                best_above = above_sorted[0][0] if above_sorted else None
                best_below = below_sorted[0][0] if below_sorted else None

                if best_above is not None and best_below is not None:
                    if abs(float(best_above["price"]) - close_price) <= abs(
                        float(best_below["price"]) - close_price
                    ):
                        primary_level = best_above
                        opposing_level = best_below
                    else:
                        primary_level = best_below
                        opposing_level = best_above
                elif best_above is not None:
                    primary_level = best_above
                elif best_below is not None:
                    primary_level = best_below

                if primary_level is None:
                    continue

                # Store primary level
                result.at[ts, "primary_level_price"] = float(primary_level["price"])
                result.at[ts, "primary_level_type"] = "resistance" if float(primary_level["price"]) >= close_price else "support"
                result.at[ts, "primary_level_source"] = primary_level.get("source_type")
                result.at[ts, "primary_level_touch_count"] = float(primary_level.get("touch_count", np.nan))
                result.at[ts, "primary_level_first_touch_ts"] = primary_level.get("first_touch_ts")
                result.at[ts, "primary_level_last_touch_ts"] = primary_level.get("last_touch_ts")
                result.at[ts, "primary_level_prominence"] = float(primary_level.get("prominence", np.nan))
                result.at[ts, "primary_level_volume_depth_ratio"] = float(primary_level.get("volume_depth_ratio", np.nan))

                # Store opposing level
                if opposing_level is not None:
                    result.at[ts, "opposing_level_price"] = float(opposing_level["price"])
                    result.at[ts, "opposing_level_type"] = "resistance" if float(opposing_level["price"]) >= close_price else "support"
                    result.at[ts, "opposing_level_source"] = opposing_level.get("source_type")
                    result.at[ts, "opposing_level_touch_count"] = float(opposing_level.get("touch_count", np.nan))
                    result.at[ts, "opposing_level_first_touch_ts"] = opposing_level.get("first_touch_ts")
                    result.at[ts, "opposing_level_last_touch_ts"] = opposing_level.get("last_touch_ts")
                    result.at[ts, "opposing_level_prominence"] = float(opposing_level.get("prominence", np.nan))
                    result.at[ts, "opposing_level_volume_depth_ratio"] = float(opposing_level.get("volume_depth_ratio", np.nan))

        return result


class FractalLevelGenerator(BaseLevelGenerator):
    """
    Generates levels based on discrete Swing Highs/Lows (Fractals).
    Strict structural approach: Last un-breached swing point is the level.
    """
    def __init__(self, pivot_period: int = 5, lookback_bars: int = 500) -> None:
        self.pivot_period = int(max(2, pivot_period))
        self.lookback_bars = int(max(100, lookback_bars))

    def compute_levels(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise ValueError("FractalLevelGenerator requires DatetimeIndex input")

        data = ohlcv.sort_index()
        highs = data["high"].values
        lows = data["low"].values
        closes = data["close"].values
        n = len(data)

        result = pd.DataFrame(
            index=data.index,
            columns=[
                "primary_level_price", "primary_level_type", "primary_level_source",
                "primary_level_touch_count", "primary_level_prominence",
                "opposing_level_price", "opposing_level_type", "opposing_level_source"
            ]
        )

        # Find pivots
        # argrelextrema uses order (period) to find local max/min
        high_idx = argrelextrema(highs, np.greater, order=self.pivot_period)[0]
        low_idx = argrelextrema(lows, np.less, order=self.pivot_period)[0]

        for i in range(self.pivot_period, n):
            start_search = max(0, i - self.lookback_bars)
            limit_search = i - self.pivot_period # Can't see future pivots

            close_price = closes[i]

            # Recent Swing Highs
            relevant_highs = [p for p in high_idx if start_search <= p <= limit_search and highs[p] > close_price]
            # Recent Swing Lows
            relevant_lows = [p for p in low_idx if start_search <= p <= limit_search and lows[p] < close_price]

            best_res = None
            if relevant_highs:
                best_res_idx = relevant_highs[-1]
                best_res = highs[best_res_idx]

            best_sup = None
            if relevant_lows:
                best_sup_idx = relevant_lows[-1]
                best_sup = lows[best_sup_idx]

            ts = data.index[i]

            if best_res is not None and best_sup is not None:
                # Primary is closer one
                dist_res = abs(best_res - close_price)
                dist_sup = abs(best_sup - close_price)
                if dist_res <= dist_sup:
                    result.at[ts, "primary_level_price"] = best_res
                    result.at[ts, "primary_level_type"] = "resistance"
                    result.at[ts, "primary_level_source"] = "fractal_high"
                    result.at[ts, "opposing_level_price"] = best_sup
                    result.at[ts, "opposing_level_type"] = "support"
                else:
                    result.at[ts, "primary_level_price"] = best_sup
                    result.at[ts, "primary_level_type"] = "support"
                    result.at[ts, "primary_level_source"] = "fractal_low"
                    result.at[ts, "opposing_level_price"] = best_res
                    result.at[ts, "opposing_level_type"] = "resistance"

            elif best_res is not None:
                result.at[ts, "primary_level_price"] = best_res
                result.at[ts, "primary_level_type"] = "resistance"
                result.at[ts, "primary_level_source"] = "fractal_high"

            elif best_sup is not None:
                result.at[ts, "primary_level_price"] = best_sup
                result.at[ts, "primary_level_type"] = "support"
                result.at[ts, "primary_level_source"] = "fractal_low"

            # Fill dummy metrics to match interface
            if pd.notna(result.at[ts, "primary_level_price"]):
                result.at[ts, "primary_level_touch_count"] = 1.0
                result.at[ts, "primary_level_prominence"] = 1.0 # High confidence discrete level

        return result


class HTFLevelGenerator(BaseLevelGenerator):
    """
    Generates levels based on High Timeframe (Daily/Weekly) OHLC.
    Previous Day High/Low/Close, Weekly High/Low.
    """
    def __init__(self, use_weekly: bool = False) -> None:
        self.use_weekly = use_weekly

    def compute_levels(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise ValueError("HTFLevelGenerator requires DatetimeIndex input")

        data = ohlcv.sort_index()

        # Resample to Daily
        daily = data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        # Shift to get "Previous Day" levels
        prev_daily = daily.shift(1)

        # Reindex to match original timeframe (forward fill)
        aligned = prev_daily.reindex(data.index, method='ffill')

        result = pd.DataFrame(index=data.index)

        pdh = aligned['high']
        pdl = aligned['low']

        close = data['close']

        dist_pdh = (pdh - close).abs()
        dist_pdl = (pdl - close).abs()

        closer_to_high = dist_pdh <= dist_pdl

        # Primary Level
        result["primary_level_price"] = np.where(closer_to_high, pdh, pdl)

        is_res = close < result["primary_level_price"]
        result["primary_level_type"] = np.where(is_res, "resistance", "support")
        result["primary_level_source"] = np.where(closer_to_high, "pdh", "pdl")

        # Opposing Level
        result["opposing_level_price"] = np.where(closer_to_high, pdl, pdh)
        is_opp_res = close < result["opposing_level_price"]
        result["opposing_level_type"] = np.where(is_opp_res, "resistance", "support")
        result["opposing_level_source"] = np.where(closer_to_high, "pdl", "pdh")

        result["primary_level_touch_count"] = 1.0
        result["primary_level_prominence"] = 1.0

        return result
