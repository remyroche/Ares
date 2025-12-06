#!/usr/bin/env python3
"""SR Level Strength Feature Research

This script is a research tool to answer two questions:

1. What features best characterize a "strong" support/resistance (S/R) level?
2. What features best characterize the "push" required to go through a level
   (volume, momentum, volatility, trend strength, etc.) and the magnitude of
   the resulting move against the level?

It:
- Loads OHLCV data for a symbol/exchange/timeframe.
- Generates S/R levels using multiple generators (KDE, Fractal, HTF-like).
- Builds event samples around level "touches".
- Constructs two feature blocks:
  * Level-strength features (ex-ante quality of the level).
  * Push/move features (what the price/volume/volatility did around the touch).
- Defines regression and classification targets based on forward returns
  relative to the level.
- Trains simple XGBoost models and reports feature importances and a
  bucketed backtest (by predicted strength).

This script is intentionally self-contained and read-only with respect to
artifacts: it does not register anything in the versioned artifact system.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so that `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.market_analysis.components.level_generators import (  # type: ignore  # noqa: E501
    RollingKDELevelGenerator,
    FractalLevelGenerator,
    HTFLevelGenerator,
)
from src.utils.tprint import tprint_info, tprint_warning, tprint_error  # type: ignore  # noqa: E501
from src.utils.data.real_data_loader import RealDataLoader  # type: ignore  # noqa: E501
from src.utils.ml_common.confidence_metrics import (  # type: ignore  # noqa: E501
    calculate_calibration_metrics,
)

try:
    import xgboost as xgb  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("xgboost is required for this research script") from exc

try:
    import scipy  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("scipy is required for this research script") from exc

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:  # Optional dependency for explicit probability calibration
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except ImportError:  # pragma: no cover - optional at runtime
    IsotonicRegression = None  # type: ignore

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research S/R level strength and push features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        help="Timeframe for analysis (e.g. 15m, 1h)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD) to restrict history",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) to restrict history",
    )
    parser.add_argument(
        "--horizon-bars",
        type=int,
        default=48,
        help="Forward horizon in bars for measuring move strength",
    )
    parser.add_argument(
        "--min-ret",
        type=float,
        default=0.005,
        help="Absolute return threshold to define a 'strong move' event",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Optional cap on number of events (for speed)",
    )
    parser.add_argument(
        "--output-features",
        type=str,
        default="sr_research_best_features.yaml",
        help="Path to save the best features list",
    )

    parser.add_argument(
        "--strong-quantile",
        type=float,
        default=0.7,
        help=(
            "Quantile (0-1) of volatility-normalized |forward return| used to "
            "define 'strong' events. Combined with min_ret as an absolute floor."
        ),
    )
    parser.add_argument(
        "--bounce-tolerance-pct",
        type=float,
        default=0.003,
        help=(
            "Relative move away from the level required to count as a bounce "
            "in path-aware bounce/break labeling."
        ),
    )
    parser.add_argument(
        "--break-tolerance-pct",
        type=float,
        default=0.003,
        help=(
            "Relative move through the level required to count as a break "
            "in path-aware bounce/break labeling."
        ),
    )

    return parser.parse_args()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, Plus DI, Minus DI."""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join="outer").max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1 / period).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di


def load_market_data(
    symbol: str,
    exchange: str,
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    """Load OHLCV data via RealDataLoader and apply optional date filter.

    This uses the same underlying parquet/real-data infrastructure as the
    training pipeline (KlinesParquetManager), but in a lightweight way.
    """

    tprint_info(f"Loading OHLCV for {symbol} on {exchange} @ {timeframe}")

    loader = RealDataLoader()

    async def _load_async() -> pd.DataFrame:
        return await loader.load_market_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            lookback_days=None,
            start_date=start,
            end_date=end,
            force_download=False,
            use_cache=True,
        )

    df = asyncio.run(_load_async())

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    tprint_info(f"Loaded {len(df)} bars of OHLCV data")
    return df


def generate_sr_levels(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Generate S/R levels from multiple generators and aggregate.

    We combine several level sources:
    - RollingKDELevelGenerator (cluster-based levels)
    - HTFLevelGenerator (previous-day high/low style pivots)

    The output is a single DataFrame aligned to `ohlcv.index` with columns:
    - primary_level_* and opposing_level_* (from unioned generators)
    - is_support / is_resistance flags
    - source indicators (kde/htf)
    """

    idx = ohlcv.index
    base = pd.DataFrame(index=idx)

    # --- KDE levels ---
    try:
        kde_gen = RollingKDELevelGenerator()
        kde_levels = kde_gen.compute_levels(ohlcv)

        # Optionally filter out the weakest KDE levels by volume depth ratio
        if not kde_levels.empty and "primary_level_volume_depth_ratio" in kde_levels.columns:
            strength = kde_levels["primary_level_volume_depth_ratio"].astype(float)
            strength = strength.replace([np.inf, -np.inf], np.nan)
            if strength.notna().sum() >= 20:
                try:
                    q25 = float(strength.quantile(0.25))
                    weak_mask = strength < q25
                    # Drop weak KDE levels by nulling their primary-level fields
                    cols_to_null = [
                        "primary_level_price",
                        "primary_level_type",
                        "primary_level_source",
                        "primary_level_touch_count",
                        "primary_level_first_touch_ts",
                        "primary_level_last_touch_ts",
                        "primary_level_prominence",
                        "primary_level_volume_depth_ratio",
                    ]
                    existing_cols = [c for c in cols_to_null if c in kde_levels.columns]
                    if existing_cols:
                        kde_levels.loc[weak_mask, existing_cols] = np.nan
                    tprint_info(
                        f"Filtered KDE levels: removed bottom 25% by depth ratio (threshold={q25:.3f})"
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    tprint_warning(f"Failed to apply KDE quantile filter: {exc}")

        kde_levels = kde_levels.add_prefix("kde_")
        base = base.join(kde_levels, how="left")
        tprint_info("Added KDE-based S/R levels")
    except Exception as exc:
        tprint_warning(f"Failed to compute KDE levels: {exc}")

    # --- HTF levels (daily pivots) ---
    try:
        htf_gen = HTFLevelGenerator(use_weekly=False)
        htf_levels = htf_gen.compute_levels(ohlcv)
        htf_levels = htf_levels.add_prefix("htf_")
        base = base.join(htf_levels, how="left")
        tprint_info("Added HTF-based S/R levels")
    except Exception as exc:
        tprint_warning(f"Failed to compute HTF levels: {exc}")

    # For simplicity, define a canonical primary level by taking the nearest
    # available level among the three generators (priority: KDE, Fractal, HTF).
    close = ohlcv["close"].astype(float)

    def process_row_for_confluence(row: pd.Series) -> Dict[str, Any]:
        """
        Process a single row to:
        1. Pick the primary level (nearest).
        2. Recover metadata for that primary level.
        3. Calculate confluence score.
        """
        price = float(close.at[row.name]) if row.name in close.index else float("nan")
        if not np.isfinite(price):
            return {
                "primary_level_price": float("nan"),
                "primary_level_type": np.nan,
                "primary_level_source": np.nan,
                "primary_level_touch_count": 0,
                "primary_level_prominence": 0.0,
                "primary_level_volume_depth_ratio": 0.0,
                "primary_level_first_touch_ts": pd.NaT,
                "primary_level_last_touch_ts": pd.NaT,
                "confluence_score": 0,
                "weighted_confluence_score": 0.0,
            }

        # 1. Gather all candidates on this row
        candidates: list[dict] = []
        # NOTE: We treat fractal-only levels as weaker/noisier and therefore
        # exclude them from primary/confluence selection here. They are still
        # computed above and available in `base` if needed elsewhere.
        for prefix, src_tag in [("kde_", "kde"), ("htf_", "htf")]:
            p_col = f"{prefix}primary_level_price"
            lp = row.get(p_col, np.nan)
            if np.isfinite(lp):
                cand = {
                    "prefix": prefix,
                    "source": src_tag,
                    "price": float(lp),
                    "type": row.get(f"{prefix}primary_level_type", np.nan),
                    "dist": abs(float(lp) - price),
                    "touch_count": row.get(f"{prefix}primary_level_touch_count", 0),
                    "prominence": row.get(f"{prefix}primary_level_prominence", 0.0),
                    "volume_depth_ratio": row.get(f"{prefix}primary_level_volume_depth_ratio", 0.0),
                    "first_touch_ts": row.get(f"{prefix}primary_level_first_touch_ts", pd.NaT),
                    "last_touch_ts": row.get(f"{prefix}primary_level_last_touch_ts", pd.NaT),
                }
                candidates.append(cand)

        if not candidates:
            return {
                "primary_level_price": float("nan"),
                "primary_level_type": np.nan,
                "primary_level_source": np.nan,
                "primary_level_touch_count": 0,
                "primary_level_prominence": 0.0,
                "primary_level_volume_depth_ratio": 0.0,
                "primary_level_first_touch_ts": pd.NaT,
                "primary_level_last_touch_ts": pd.NaT,
                "confluence_score": 0,
                "weighted_confluence_score": 0.0,
            }

        # Pick nearest as primary
        best = min(candidates, key=lambda x: x["dist"])

        # 2. Calculate Confluence
        # Count other levels within X% of the PRIMARY level price (not just current price)
        # Use 0.2% as a tight confluence band
        confluence_band = 0.002
        confluence_score = 0
        weighted_confluence_score = 0.0

        for cand in candidates:
            # Distance from PRIMARY level
            d_pct = abs(cand["price"] - best["price"]) / best["price"]
            if d_pct <= confluence_band:
                confluence_score += 1

                # Weighting logic
                base_weight = 1.0
                multiplier = 1.0

                if cand["source"] == "htf":
                    base_weight = 2.0  # HTF is stronger
                elif cand["source"] == "kde":
                    base_weight = 1.0
                    # Scale by volume depth if available (often > 1.0 for strong levels)
                    # Clip to reasonable range to avoid exploding scores
                    vol_scale = cand.get("volume_depth_ratio", 1.0)
                    if np.isfinite(vol_scale):
                        multiplier = max(0.5, min(vol_scale, 5.0))
                elif cand["source"] == "fractal":
                    base_weight = 1.0
                    # Fractal prominence is usually 1.0, but if we had it, we'd use it
                    prom = cand.get("prominence", 1.0)
                    if np.isfinite(prom):
                        multiplier = max(0.5, min(prom, 3.0))

                weighted_confluence_score += base_weight * multiplier

        return {
            "primary_level_price": best["price"],
            "primary_level_type": best["type"],
            "primary_level_source": best["source"],
            "primary_level_touch_count": best.get("touch_count", 0),
            "primary_level_prominence": best.get("prominence", 0.0),
            "primary_level_volume_depth_ratio": best.get("volume_depth_ratio", 0.0),
            "primary_level_first_touch_ts": best.get("first_touch_ts", pd.NaT),
            "primary_level_last_touch_ts": best.get("last_touch_ts", pd.NaT),
            "confluence_score": confluence_score,
            "weighted_confluence_score": weighted_confluence_score,
        }

    results = []
    for ts, row in base.iterrows():
        results.append(process_row_for_confluence(row))

    sr = pd.DataFrame(results, index=idx)

    sr["is_support"] = sr["primary_level_type"].astype(str).str.contains("support", case=False, na=False)
    sr["is_resistance"] = sr["primary_level_type"].astype(str).str.contains("resistance", case=False, na=False)

    return sr


def compute_path_aware_bounce_break_labels(
    df: pd.DataFrame,
    event_df: pd.DataFrame,
    horizon_bars: int,
    break_tolerance_pct: float,
    bounce_tolerance_pct: float,
) -> pd.Series:
    """Compute path-aware bounce/break labels for each event.

    For each touch event, we look forward up to ``horizon_bars`` and determine
    whether price first makes a meaningful move *through* the level (break) or
    *away* from it (bounce), with simple tolerances:

    - For support levels:
        * Break: price trades below ``level * (1 - break_tolerance_pct)``.
        * Bounce: price trades above ``level * (1 + bounce_tolerance_pct)``.
    - For resistance levels: symmetric conditions.

    The label is:
        1 = break, 0 = bounce, NaN = undecided within horizon.
    """

    if horizon_bars <= 0:
        return pd.Series(index=event_df.index, dtype=float)

    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values

    labels = pd.Series(index=event_df.index, dtype=float)

    for ts, row in event_df.iterrows():
        try:
            loc = df.index.get_loc(ts)
        except KeyError:
            labels.loc[ts] = np.nan
            continue

        level_price = float(row["primary_level_price"])
        if not np.isfinite(level_price) or level_price <= 0:
            labels.loc[ts] = np.nan
            continue

        is_support = bool(row.get("is_support", False))
        is_resistance = bool(row.get("is_resistance", False))
        if not (is_support or is_resistance):
            labels.loc[ts] = np.nan
            continue

        start = loc + 1
        end = min(loc + horizon_bars, len(df) - 1)
        if start > end:
            labels.loc[ts] = np.nan
            continue

        win_slice = slice(start, end + 1)
        w_close = close[win_slice]
        w_high = high[win_slice]
        w_low = low[win_slice]

        if is_support:
            break_cond = w_low <= level_price * (1.0 - break_tolerance_pct)
            bounce_cond = w_close >= level_price * (1.0 + bounce_tolerance_pct)
        else:  # resistance
            break_cond = w_high >= level_price * (1.0 + break_tolerance_pct)
            bounce_cond = w_close <= level_price * (1.0 - bounce_tolerance_pct)

        break_idx = np.where(break_cond)[0]
        bounce_idx = np.where(bounce_cond)[0]

        if break_idx.size == 0 and bounce_idx.size == 0:
            labels.loc[ts] = np.nan
        elif break_idx.size == 0:
            labels.loc[ts] = 0.0
        elif bounce_idx.size == 0:
            labels.loc[ts] = 1.0
        else:
            labels.loc[ts] = 1.0 if break_idx[0] <= bounce_idx[0] else 0.0

    return labels


def build_event_dataset(
    ohlcv: pd.DataFrame,
    sr: pd.DataFrame,
    horizon_bars: int,
    min_ret: float,
    max_samples: int,
    strong_quantile: float,
    bounce_tolerance_pct: float,
    break_tolerance_pct: float,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Construct event-level dataset around S/R touches.

    Returns:
        X: feature dataframe (level-strength + push features).
        y_reg: continuous forward return vs level at ``horizon_bars``.
        y_cls: binary strong-move indicator (|ret| >= min_ret).
        y_reg_1h: continuous forward return vs level at ~1h horizon.
    """
    # ------------------------------------------------------------------
    # 0. Pre-calculate HTF & Regime Features (Full History)
    # ------------------------------------------------------------------
    # Resample to 1H and 4H
    # NOTE: We shift(1) after resampling to ensure we are using COMPLETED candles
    # to avoid look-ahead bias when reindexing back to 15m.
    df_1h = ohlcv.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    df_4h = ohlcv.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    # Calculate HTF Indicators
    for label, frame in [("1h", df_1h), ("4h", df_4h)]:
        # Trend (SMA)
        frame[f"sma_20_{label}"] = frame["close"].rolling(20).mean()
        frame[f"trend_{label}"] = (frame["close"] / frame[f"sma_20_{label}"] - 1.0)
        # RSI
        frame[f"rsi_{label}"] = calculate_rsi(frame["close"], 14)
        # Volatility (ATR-ish on close)
        frame[f"vol_{label}"] = frame["close"].pct_change().rolling(20).std()

    # Reindex HTF features back to 15m (Forward Fill)
    htf_feats = pd.DataFrame(index=ohlcv.index)

    # 1H Features: Shift 1 to use only completed candles
    aligned_1h = df_1h.shift(1).reindex(ohlcv.index, method="ffill")
    htf_feats["trend_1h"] = aligned_1h["trend_1h"]
    htf_feats["rsi_1h"] = aligned_1h["rsi_1h"]
    htf_feats["vol_1h"] = aligned_1h["vol_1h"]

    # 4H Features: Shift 1 to use only completed candles
    aligned_4h = df_4h.shift(1).reindex(ohlcv.index, method="ffill")
    htf_feats["trend_4h"] = aligned_4h["trend_4h"]
    htf_feats["rsi_4h"] = aligned_4h["rsi_4h"]
    htf_feats["vol_4h"] = aligned_4h["vol_4h"]

    # Market Regime (15m)
    adx, plus_di, minus_di = calculate_adx(
        ohlcv["high"], ohlcv["low"], ohlcv["close"]
    )
    htf_feats["adx_15m"] = adx
    htf_feats["di_spread_15m"] = plus_di - minus_di

    sma_50 = ohlcv["close"].rolling(50).mean()
    sma_200 = ohlcv["close"].rolling(200).mean()
    htf_feats["trend_regime_sma"] = (sma_50 > sma_200).astype(float)
    htf_feats["regime_interaction"] = htf_feats["adx_15m"] * (htf_feats["trend_regime_sma"] * 2 - 1) # +/- ADX based on trend

    # ------------------------------------------------------------------
    # 1. Standard Setup
    # ------------------------------------------------------------------
    # Calculate ATR first for dynamic touch definition
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=10).mean()

    # Join S/R
    df = ohlcv.join(sr, how="inner")
    df = df.dropna(subset=["primary_level_price"])  # require a level

    # Align ATR and price data to the filtered df index
    atr = atr_14.loc[df.index]
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    level = df["primary_level_price"].astype(float)

    # Dynamic Touch Band: 0.5 * ATR
    # If ATR is nan (start of data), fallback to 0.4%
    touch_dist_price = 0.5 * atr
    fallback_dist = 0.004 * level
    touch_dist_price = touch_dist_price.fillna(fallback_dist)

    # Improved Touch Logic: Check if level is within High-Low range OR close proximity
    # This captures "wick tests" better than just close proximity.
    # We still use the threshold for the 'close' check, but strictly use range for wicks.
    is_within_range = (low <= level) & (high >= level)
    abs_diff = (close - level).abs()
    is_close_proximity = abs_diff <= touch_dist_price

    touch_mask = is_within_range | is_close_proximity

    # Pre-calculate Volume Consumption Features on the full dataframe
    # "Volume Consumed": Volume of bars that are touching the active level
    vol_touching = df["volume"].where(touch_mask, 0.0)
    # Recent volume consumed (last 10 bars)
    df["recent_vol_consumed"] = vol_touching.rolling(10).sum()
    # Intensity: Volume consumed / Average Volume
    vol_ma = df["volume"].rolling(50).mean()
    df["recent_vol_intensity"] = df["recent_vol_consumed"] / (vol_ma * 10).replace(0, np.nan)

    # "Time Since Last Touch" refinement:
    # We want exact bars since last True in touch_mask
    # This is a bit complex to vectorise perfectly for "active level" switches,
    # but using the existing `hours_since_last_test` from the generator is good.
    # We will add "Bars Since Last Touch" based on the boolean mask for the current regime
    # A simple expanding group count can proxy this if levels were constant, but they aren't.
    # We will rely on generator metadata for 'last_touch_ts' but add a 'local' recency.

    event_df = df.loc[touch_mask].copy()
    if event_df.empty:
        raise ValueError("No S/R touch events found with current configuration")

    if len(event_df) > max_samples:
        event_df = event_df.iloc[-max_samples:]

    # Forward returns vs level (signed so that positive is away from level, negative is through).
    fwd_close_full = close.shift(-horizon_bars)
    # Align all components on the event index to avoid boolean indexer misalignment.
    fwd_close_evt = fwd_close_full.loc[event_df.index]
    level_evt = level.loc[event_df.index]

    # For support, a break is downward; for resistance, a break is upward.
    is_support = event_df["is_support"].astype(bool)
    is_resistance = event_df["is_resistance"].astype(bool)

    fwd_ret = pd.Series(index=event_df.index, dtype=float)
    # For support: move away = positive, break through = negative
    fwd_ret.loc[is_support] = ((fwd_close_evt - level_evt) / level_evt).loc[is_support]
    # For resistance: move away = positive, break through = negative
    fwd_ret.loc[is_resistance] = ((level_evt - fwd_close_evt) / level_evt).loc[is_resistance]

    fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan).dropna()
    event_df = event_df.loc[fwd_ret.index]

    # Re-align support/resistance flags to the filtered event set
    is_support = event_df["is_support"].astype(bool)
    is_resistance = event_df["is_resistance"].astype(bool)

    y_reg = fwd_ret

    # ------------------------------------------------------------------
    # Strong/Weak Label: volatility-normalized + quantile-based
    # ------------------------------------------------------------------
    # Volatility proxy: ATR relative to level
    level_evt_final = event_df["primary_level_price"].astype(float)
    atr_evt_final = atr.loc[event_df.index]
    vol_unit = (atr_evt_final / level_evt_final.replace(0.0, np.nan)).abs()
    vol_unit = vol_unit.replace([np.inf, -np.inf], np.nan)

    abs_ret = y_reg.abs()
    if vol_unit.notna().any():
        norm_abs_ret = (abs_ret / vol_unit).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        norm_thr = float(norm_abs_ret.quantile(strong_quantile))
    else:
        norm_abs_ret = abs_ret.copy()
        norm_thr = float(norm_abs_ret.quantile(strong_quantile))

    abs_thr = float(abs_ret.quantile(strong_quantile))
    # Ensure thresholds are positive and provide a log for transparency
    if norm_thr <= 0:
        norm_thr = max(abs_thr, min_ret)
    tprint_info(
        f"Strong label thresholds: min_ret={min_ret:.4f}, "
        f"strong_quantile={strong_quantile:.2f}, norm_thr={norm_thr:.4f}, abs_thr={abs_thr:.4f}"
    )

    # Strong if: (i) abs return clears min_ret AND (ii) vol-normalized return in top quantile
    strong_mask = (abs_ret >= min_ret) & (norm_abs_ret >= norm_thr)
    y_cls = strong_mask.astype(int)

    # Secondary ~1h horizon (time-based, independent of horizon_bars)
    bar_deltas = ohlcv.index.to_series().diff().dropna()
    if bar_deltas.empty:
        bars_per_hour = 0
    else:
        approx_step_seconds = bar_deltas.median().total_seconds()
        bars_per_hour = int(round(3600.0 / approx_step_seconds)) if approx_step_seconds > 0 else 0

    if bars_per_hour <= 0:
        y_reg_1h = pd.Series(index=y_reg.index, dtype=float)
        y_reg_1h[:] = np.nan
    else:
        idx_evt = event_df.index
        fwd_close_full_1h = close.shift(-bars_per_hour)
        fwd_close_evt_1h = fwd_close_full_1h.loc[idx_evt]
        level_evt_1h = level.loc[idx_evt]

        # Base signed return (same convention as primary horizon before side adjustment)
        base_ret_1h = (fwd_close_evt_1h - level_evt_1h) / level_evt_1h

        fwd_ret_1h = pd.Series(index=idx_evt, dtype=float)
        # For support: move away = positive
        fwd_ret_1h.loc[is_support] = base_ret_1h.loc[is_support]
        # For resistance: move away = positive (invert sign)
        fwd_ret_1h.loc[is_resistance] = (-base_ret_1h).loc[is_resistance]
        fwd_ret_1h = fwd_ret_1h.replace([np.inf, -np.inf], np.nan)

        # Align 1h target to the same event set as y_reg
        y_reg_1h = fwd_ret_1h.loc[y_reg.index]

    # Path-aware bounce vs break labels (1 = break, 0 = bounce, NaN = undecided)
    y_bb = compute_path_aware_bounce_break_labels(
        df=df,
        event_df=event_df,
        horizon_bars=horizon_bars,
        break_tolerance_pct=break_tolerance_pct,
        bounce_tolerance_pct=bounce_tolerance_pct,
    )

    # ------------------------------------------------------------------
    # Feature Assembly
    # ------------------------------------------------------------------
    lvl_feats = pd.DataFrame(index=event_df.index)

    # --- 1. Basic Level Features ---
    lvl_feats["dist_to_level_pct"] = ((close - level) / level).loc[event_df.index]

    # Generator/type/source indicators (one-hot-ish flags)
    src_series = event_df["primary_level_source"].astype(str)
    lvl_feats["src_is_kde"] = src_series.str.contains("kde", case=False, na=False).astype(float)
    lvl_feats["src_is_fractal"] = src_series.str.contains("fractal", case=False, na=False).astype(float)
    lvl_feats["src_is_htf"] = src_series.str.contains("htf|pdh|pdl", case=False, na=False).astype(float)

    # More granular level-type flags:
    lvl_feats["lvl_is_hvn"] = src_series.str.contains("volume_node", case=False, na=False).astype(float)
    lvl_feats["lvl_is_swing"] = src_series.str.contains("swing_high|swing_low|fractal", case=False, na=False).astype(float)
    lvl_feats["lvl_is_pivot"] = src_series.str.contains("pdh|pdl", case=False, na=False).astype(float)

    lvl_feats["is_support"] = is_support.astype(float)
    lvl_feats["is_resistance"] = is_resistance.astype(float)

    # NEW METADATA FEATURES (Blind Spot Fixes)
    # 1. Raw Metadata
    lvl_feats["meta_touch_count"] = event_df["primary_level_touch_count"].astype(float)
    lvl_feats["meta_prominence"] = event_df["primary_level_prominence"].astype(float)
    lvl_feats["meta_vol_depth"] = event_df["primary_level_volume_depth_ratio"].astype(float)

    # 2. Confluence Scores
    lvl_feats["confluence_score"] = event_df["confluence_score"].astype(float)
    lvl_feats["weighted_confluence_score"] = event_df["weighted_confluence_score"].astype(float)

    # 3. Age / Decay
    # We need to convert timestamps to "bars ago"
    # Current index is datetime
    # We can use get_indexer to find integer locations if needed, or simple timedelta math
    # Since ohlcv is equidistant (mostly), timedelta / freq is easiest approximation

    # Convert index to Series for subtraction
    current_ts = event_df.index.to_series()

    first_touch = pd.to_datetime(event_df["primary_level_first_touch_ts"])
    last_touch = pd.to_datetime(event_df["primary_level_last_touch_ts"])

    # Approximate bars using Minutes (assuming 15m timeframe if not passed, but we can just use total seconds)
    # The script uses 'timeframe' arg but it's string.
    # Let's assume standard 15m for the denominator or just use raw seconds as a proxy for age
    # Or better: simply (current_ts - timestamp).dt.total_seconds() / (15 * 60)
    # Since we might be on 1h, let's just use hours as the unit for "Age"

    lvl_feats["level_age_hours"] = (current_ts - first_touch).dt.total_seconds() / 3600.0
    lvl_feats["hours_since_last_test"] = (current_ts - last_touch).dt.total_seconds() / 3600.0

    # Fill NaT with 0.0 (fresh level)
    lvl_feats["level_age_hours"] = lvl_feats["level_age_hours"].fillna(0.0)
    lvl_feats["hours_since_last_test"] = lvl_feats["hours_since_last_test"].fillna(0.0)

    # --- 2. Advanced Level Physics (Resistance Testing) ---
    # Volume Consumed Features (Joined from pre-calculation)
    lvl_feats["recent_vol_consumed_ratio"] = event_df["recent_vol_intensity"]

    # Decay/Reinforcement Interactions
    # High touch count + Low Age = Frequent testing (Weakening?)
    # High touch count + High Age = Historic Level (Strong?)
    lvl_feats["touch_frequency"] = lvl_feats["meta_touch_count"] / (lvl_feats["level_age_hours"] + 1.0)

    # Volume Depth per Touch (Efficiency)
    lvl_feats["vol_depth_per_touch"] = lvl_feats["meta_vol_depth"] / (lvl_feats["meta_touch_count"] + 1.0)

    lvl_feats["age_log_hours"] = np.log1p(lvl_feats["level_age_hours"].clip(lower=0.0))
    lvl_feats["recency_ratio"] = lvl_feats["hours_since_last_test"] / (lvl_feats["level_age_hours"] + 1.0)

    # ------------------------------------------------------------------
    # 2b. Level-specific historical behaviour (online, past-only)
    # ------------------------------------------------------------------
    # For each distinct level (side + price), track:
    # - number of past events
    # - past strong-move rate
    # - past break vs bounce fractions
    level_side = np.where(is_support, "support", "resistance")
    level_ids = pd.Series(
        [f"{side}_{price:.6f}" for side, price in zip(level_side, level_evt_final)],
        index=event_df.index,
    )

    hist_event_count = pd.Series(0.0, index=event_df.index)
    hist_strong_rate = pd.Series(0.5, index=event_df.index)
    hist_break_rate = pd.Series(0.5, index=event_df.index)
    hist_bounce_rate = pd.Series(0.5, index=event_df.index)

    history: Dict[str, Dict[str, float]] = {}
    for ts in event_df.index:
        key = str(level_ids.loc[ts])
        state = history.get(key, {"n": 0.0, "strong": 0.0, "break": 0.0, "bounce": 0.0})
        n_prev = state["n"]
        if n_prev > 0:
            hist_event_count.loc[ts] = n_prev
            hist_strong_rate.loc[ts] = state["strong"] / n_prev
            total_bb = state["break"] + state["bounce"]
            if total_bb > 0:
                hist_break_rate.loc[ts] = state["break"] / total_bb
                hist_bounce_rate.loc[ts] = state["bounce"] / total_bb
        # Update state with current outcome for future events
        state["n"] = n_prev + 1.0
        if y_cls.loc[ts] == 1:
            state["strong"] += 1.0
        if y_reg.loc[ts] < 0:
            state["break"] += 1.0
        elif y_reg.loc[ts] > 0:
            state["bounce"] += 1.0
        history[key] = state

    lvl_feats["level_hist_event_count"] = hist_event_count
    lvl_feats["level_hist_strong_rate"] = hist_strong_rate
    lvl_feats["level_hist_break_rate"] = hist_break_rate
    lvl_feats["level_hist_bounce_rate"] = hist_bounce_rate

    # ------------------------------------------------------------------
    # 3. Push / Move Features (Momentum & Volatility)
    # ------------------------------------------------------------------
    push_feats = pd.DataFrame(index=event_df.index)

    # Recent volatility (rolling std of returns)
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    vol_20 = ret.rolling(20, min_periods=10).std()
    vol_60 = ret.rolling(60, min_periods=20).std()
    push_feats["vol_20"] = vol_20.loc[event_df.index]
    push_feats["vol_60"] = vol_60.loc[event_df.index]

    # Recent momentum (price vs moving averages)
    ma_fast = close.rolling(20, min_periods=10).mean()
    ma_slow = close.rolling(60, min_periods=20).mean()
    push_feats["ma_fast_rel"] = (close / ma_fast - 1.0).loc[event_df.index]
    push_feats["ma_slow_rel"] = (close / ma_slow - 1.0).loc[event_df.index]
    push_feats["trend_slope_fast"] = ma_fast.pct_change(5).loc[event_df.index]

    # Volume push: current vs rolling median
    vol = df["volume"].astype(float)
    vol_med_60 = vol.rolling(60, min_periods=20).median()
    push_feats["vol_rel_60"] = (vol / vol_med_60.replace(0.0, np.nan)).loc[event_df.index]

    # True range as instantaneous volatility proxy
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_20 = tr.rolling(20, min_periods=10).mean()
    push_feats["atr20_rel_level"] = (atr_20 / level.replace(0.0, np.nan)).loc[event_df.index]

    # Immediate candle structure: wick lengths relative to body
    body = (close - df["open"].astype(float)).abs()
    upper_wick = (high - close).clip(lower=0.0)
    lower_wick = (close - low).clip(lower=0.0)
    denom = body.replace(0.0, np.nan)
    push_feats["upper_wick_rel"] = (upper_wick / denom).loc[event_df.index]
    push_feats["lower_wick_rel"] = (lower_wick / denom).loc[event_df.index]

    # --- ADVANCED PUSH FEATURES ---

    # 1. Approach Velocity (Momentum into the level)
    # Calculate return over last 3 bars normalized by ATR
    # Positive means moving UP towards level (if resistance) or DOWN towards level (if support) ?
    # Actually, we want velocity *towards* the level.
    # We already have `dist` = abs(close - level).
    # Velocity = change in distance / time
    # Let's use simple signed momentum relative to the level direction.
    # If Support: Price is falling. We want to know how fast.
    # If Resistance: Price is rising. We want to know how fast.

    ret_3 = close.diff(3)
    atr_val = atr.loc[event_df.index]

    # For support (price falling), negative return is high velocity towards level.
    # For resistance (price rising), positive return is high velocity towards level.
    # Let's normalize so positive = fast approach towards level.

    velocity_proxy = pd.Series(index=event_df.index, dtype=float)

    # Extract ret_3 values for the events to ensure alignment with boolean masks
    ret_3_events = ret_3.loc[event_df.index]

    velocity_proxy.loc[is_support] = -ret_3_events.loc[is_support] # Falling = +velocity
    velocity_proxy.loc[is_resistance] = ret_3_events.loc[is_resistance] # Rising = +velocity

    push_feats["approach_velocity_3_atr"] = (velocity_proxy / atr_val).loc[event_df.index]

    # 2. Volume Trend (Slope of volume)
    # Simple linear regression slope of last 5 volume bars
    from scipy.stats import linregress

    def calc_slope(series):
        if len(series) < 5: return 0.0
        # Normalize volume by its mean to get comparable slopes
        y = series.values / (series.mean() + 1e-9)
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        return slope

    # Rolling apply is slow, but we only need it for event indices.
    # Optimization: Extract windows for events only
    vol_slope = []
    vol_series = df["volume"].astype(float)
    for idx_ts in event_df.index:
        loc = df.index.get_loc(idx_ts)
        if loc < 5:
            vol_slope.append(0.0)
            continue
        window = vol_series.iloc[loc-4:loc+1]
        vol_slope.append(calc_slope(window))

    push_feats["volume_trend_slope_5"] = vol_slope

    # 3. Candle Compression (Coiling)
    # Ratio of short-term ATR (3) to medium-term ATR (14)
    # Low values (< 1) imply coiling/tightening range
    atr_3 = tr.rolling(3).mean()
    push_feats["candle_compression_ratio"] = (atr_3 / atr_14).loc[event_df.index]

    push_feats["vol_ratio_20_60"] = push_feats["vol_20"] / push_feats["vol_60"].replace(0.0, np.nan)
    comp_series = push_feats["candle_compression_ratio"]
    if comp_series.notna().sum() >= 20:
        comp_thr = float(comp_series.quantile(0.25))
    else:
        comp_thr = 1.0
    push_feats["is_compressed"] = (comp_series <= comp_thr).astype(float)

    # 4. Close Location (Buying/Selling Pressure)
    # (Close - Low) / (High - Low)
    # 1.0 = Close at High, 0.0 = Close at Low
    # Averaged over last 3 bars
    clv = (close - low) / (high - low).replace(0.0, np.nan)
    clv_avg = clv.rolling(3).mean()
    push_feats["close_location_score_3"] = clv_avg.loc[event_df.index]

    # 5. Consecutive Tests (Grinding)
    # How many of the last 10 bars touched the level band?
    # touch_mask is the full boolean series defined earlier
    recent_touches = touch_mask.rolling(10).sum()
    push_feats["consecutive_test_count_10"] = recent_touches.loc[event_df.index]

    # Distance to opposing level (if available)
    # We re-use the HTF generator's opposing level if present; otherwise this
    # remains NaN and will be filled.
    opp_cols = [c for c in df.columns if c.endswith("opposing_level_price")]
    if opp_cols:
        opp_price = df[opp_cols[0]].astype(float)
        push_feats["dist_to_opp_level_pct"] = ((opp_price - level) / level).loc[event_df.index]

    # ------------------------------------------------------------------
    # 4. HTF & Regime Context Features (New)
    # ------------------------------------------------------------------
    context_feats = htf_feats.loc[event_df.index].copy()

    # Cross-Timeframe Interactions
    # "Push" Alignment: Is 15m momentum aligned with 1H/4H trend?
    context_feats["trend_alignment_1h"] = np.sign(push_feats["trend_slope_fast"]) == np.sign(context_feats["trend_1h"])
    context_feats["trend_alignment_1h"] = context_feats["trend_alignment_1h"].astype(float)

    context_feats["trend_alignment_4h"] = np.sign(push_feats["trend_slope_fast"]) == np.sign(context_feats["trend_4h"])
    context_feats["trend_alignment_4h"] = context_feats["trend_alignment_4h"].astype(float)

    # Interaction: Confluence * HTF Trend Strength
    # Strong level + Strong HTF trend into it = Breakout?
    # Strong level + Overextended HTF trend (High RSI) = Bounce?
    context_feats["confluence_x_rsi1h"] = lvl_feats["confluence_score"] * context_feats["rsi_1h"]
    context_feats["confluence_x_trend4h"] = lvl_feats["confluence_score"] * context_feats["trend_4h"].abs()

    # Additional MTF interaction features
    context_feats["vol_ratio_1h_4h"] = context_feats["vol_1h"] / context_feats["vol_4h"].replace(0.0, np.nan)
    context_feats["trend_1h_x_trend4h"] = context_feats["trend_1h"] * context_feats["trend_4h"]

    trend_in_1h = context_feats["trend_1h"].copy()
    trend_in_1h[is_support] = -trend_in_1h[is_support]
    context_feats["trend_into_level_1h"] = trend_in_1h

    trend_in_4h = context_feats["trend_4h"].copy()
    trend_in_4h[is_support] = -trend_in_4h[is_support]
    context_feats["trend_into_level_4h"] = trend_in_4h

    local_trend_in = push_feats["trend_slope_fast"].copy()
    local_trend_in[is_support] = -local_trend_in[is_support]
    context_feats["local_trend_into_level"] = local_trend_in

    # Final feature frame
    features = pd.concat([lvl_feats, push_feats, context_feats], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return features, y_reg, y_cls, y_reg_1h, y_bb


def train_and_report(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_cls: pd.Series,
    output_features_path: str,
    y_reg_1h: Optional[pd.Series] | None = None,
    y_bb: Optional[pd.Series] | None = None,
) -> None:
    """Train simple XGB models and print feature importances and bucketed PnL."""

    if len(X) < 200:
        tprint_warning(f"Very few events ({len(X)}); results will be noisy")

    # Time-based split: 60% train, 20% val, 20% test
    n = len(X)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    idx = X.index

    X_train = X.iloc[:train_end]
    y_reg_train = y_reg.loc[X_train.index]
    y_cls_train = y_cls.loc[X_train.index]

    X_val = X.iloc[train_end:val_end]
    y_reg_val = y_reg.loc[X_val.index]
    y_cls_val = y_cls.loc[X_val.index]

    X_test = X.iloc[val_end:]
    y_reg_test = y_reg.loc[X_test.index]
    y_cls_test = y_cls.loc[X_test.index]

    # Helper: simple time-based walk-forward CV to obtain OOF probabilities
    def _time_series_oof_cv(
        X_all: pd.DataFrame,
        y_all: pd.Series,
        n_splits: int = 3,
    ) -> Tuple[pd.Series, Dict[int, Dict[str, float]]]:
        n_all = len(X_all)
        oof = pd.Series(index=X_all.index, dtype=float)
        fold_metrics: Dict[int, Dict[str, float]] = {}
        if n_all < 600:
            return oof, fold_metrics

        # Expanding train, rolling test windows
        fractions = [0.4, 0.6, 0.8, 1.0]
        for k in range(1, len(fractions)):
            train_end = int(fractions[k - 1] * n_all)
            test_end = int(fractions[k] * n_all)
            if train_end < 200 or (test_end - train_end) < 100:
                continue

            X_tr = X_all.iloc[:train_end]
            y_tr = y_all.loc[X_tr.index]
            X_te = X_all.iloc[train_end:test_end]
            y_te = y_all.loc[X_te.index]

            clf_cv = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
            )
            clf_cv.fit(X_tr.values, y_tr.values)
            proba_te = clf_cv.predict_proba(X_te.values)[:, 1]
            oof.loc[X_te.index] = proba_te

            try:
                prob_mat = np.column_stack([1.0 - proba_te, proba_te])
                cal = calculate_calibration_metrics(y_te.values, prob_mat)
                fold_metrics[k] = {
                    "brier": float(cal.get("brier_score", np.nan)),
                    "ece": float(cal.get("expected_calibration_error", np.nan)),
                }
            except Exception:
                continue

        return oof, fold_metrics

    # Regression model for magnitude of move
    reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
    )

    tprint_info("Training XGBRegressor for move magnitude...")
    reg.fit(X_train.values, y_reg_train.values)

    # Classification model for strong vs weak events
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
    )

    tprint_info("Training XGBClassifier for strong vs weak moves...")
    clf.fit(X_train.values, y_cls_train.values)

    # Walk-forward OOF calibration diagnostics for the strong/weak classifier
    oof_proba, cv_metrics = _time_series_oof_cv(X, y_cls)
    if oof_proba.notna().any():
        idx_valid = oof_proba.notna()
        try:
            prob_mat_oof = np.column_stack([
                1.0 - oof_proba.loc[idx_valid].values,
                oof_proba.loc[idx_valid].values,
            ])
            cal_oof = calculate_calibration_metrics(y_cls.loc[idx_valid].values, prob_mat_oof)
            tprint_info(
                "Walk-forward OOF calibration (strong/weak, uncalibrated): "
                f"Brier={cal_oof.get('brier_score', np.nan):.4f}, "
                f"ECE={cal_oof.get('expected_calibration_error', np.nan):.4f}, "
                f"quality={cal_oof.get('calibration_quality', 'unknown')}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            tprint_warning(f"Failed to compute OOF calibration metrics: {exc}")

        if cv_metrics:
            for k, m in cv_metrics.items():
                tprint_info(
                    f"  Fold {k}: Brier={m['brier']:.4f}, ECE={m['ece']:.4f}"
                )

    # ------------------------------------------------------------------
    # Feature importances & Selection
    # ------------------------------------------------------------------
    feature_names = list(X.columns)

    def _get_top_features(model) -> Dict[str, float]:
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            return {feature_names[i]: float(importances[i]) for i in indices[:30]}
        except Exception:
            return {}

    reg_top = _get_top_features(reg)
    clf_top = _get_top_features(clf)

    def _print_importances(top_dict, title):
        tprint_info(f"Top 20 features for {title}:")
        for name, imp in list(top_dict.items())[:20]:
            tprint_info(f"  {name:30s}  {imp:.4f}")

    _print_importances(reg_top, "XGBRegressor (magnitude)")
    _print_importances(clf_top, "XGBClassifier (strong/weak)")

    # Save best features to YAML
    # We union the top 20 from both models
    best_features = sorted(list(set(list(reg_top.keys())[:20] + list(clf_top.keys())[:20])))

    if yaml is not None:
        try:
            with open(output_features_path, "w") as f:
                yaml.dump({"selected_features": best_features}, f)
            tprint_info(f"Saved {len(best_features)} unique best features to {output_features_path}")
        except Exception as e:
            tprint_error(f"Failed to save features to YAML: {e}")
    else:
        tprint_warning("PyYAML not installed; skipping feature export to YAML.")

    # ------------------------------------------------------------------
    # Bucketed backtest by predicted strength (classifier probabilities)
    # ------------------------------------------------------------------
    tprint_info("Computing bucketed backtest by predicted strong-move probability...")
    proba_test = clf.predict_proba(X_test.values)[:, 1]
    # Also compute validation probabilities for calibration
    try:
        proba_val = clf.predict_proba(X_val.values)[:, 1]
    except Exception:  # pragma: no cover - defensive
        proba_val = None

    # Global calibration diagnostics (uncalibrated)
    try:
        prob_matrix_raw = np.column_stack([1.0 - proba_test, proba_test])
        cal_raw = calculate_calibration_metrics(y_cls_test.values, prob_matrix_raw)
        if cal_raw.get("brier_score") is not None:
            tprint_info(
                "Global classifier calibration (uncalibrated): "
                f"Brier={cal_raw['brier_score']:.4f}, "
                f"ECE={cal_raw.get('expected_calibration_error', 0.0):.4f}, "
                f"quality={cal_raw.get('calibration_quality', 'unknown')}"
            )
    except Exception as exc:  # pragma: no cover - defensive
        tprint_warning(f"Failed to compute global calibration metrics: {exc}")

    # Optional isotonic regression calibration using validation set
    if IsotonicRegression is not None and proba_val is not None:
        try:
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True)  # type: ignore[call-arg]
            iso.fit(proba_val, y_cls_val.values)
            proba_test_iso = iso.predict(proba_test)
            proba_test_iso = np.clip(proba_test_iso, 0.0, 1.0)
            prob_matrix_iso = np.column_stack([1.0 - proba_test_iso, proba_test_iso])
            cal_iso = calculate_calibration_metrics(y_cls_test.values, prob_matrix_iso)
            if cal_iso.get("brier_score") is not None:
                tprint_info(
                    "Global classifier calibration (isotonic): "
                    f"Brier={cal_iso['brier_score']:.4f}, "
                    f"ECE={cal_iso.get('expected_calibration_error', 0.0):.4f}, "
                    f"quality={cal_iso.get('calibration_quality', 'unknown')}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            tprint_warning(f"Isotonic calibration failed: {exc}")
    elif IsotonicRegression is None:
        tprint_warning("sklearn.isotonic.IsotonicRegression not available; skipping isotonic calibration.")

    # Backtest frame
    df_bt = pd.DataFrame(index=X_test.index)
    df_bt["pred_strong_prob"] = proba_test
    df_bt["fwd_ret"] = y_reg_test
    if y_reg_1h is not None:
        y_reg_1h_test = y_reg_1h.loc[X_test.index]
        df_bt["fwd_ret_1h"] = y_reg_1h_test

    # Carry over side flags for side-specific analysis on the test set
    if "is_support" in X.columns:
        df_bt["is_support"] = X_test["is_support"]
    else:
        df_bt["is_support"] = 0.0
    if "is_resistance" in X.columns:
        df_bt["is_resistance"] = X_test["is_resistance"]
    else:
        df_bt["is_resistance"] = 0.0

    # Quantile buckets: bottom 20%, mid 60%, top 20%
    q_low = df_bt["pred_strong_prob"].quantile(0.2)
    q_high = df_bt["pred_strong_prob"].quantile(0.8)

    low_bucket = df_bt[df_bt["pred_strong_prob"] <= q_low]
    mid_bucket = df_bt[(df_bt["pred_strong_prob"] > q_low) & (df_bt["pred_strong_prob"] < q_high)]
    high_bucket = df_bt[df_bt["pred_strong_prob"] >= q_high]

    def _bucket_stats(name: str, bucket: pd.DataFrame, col: str = "fwd_ret") -> Dict[str, float]:
        if bucket.empty:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "sharpe": float("nan")}
        r = bucket[col].values
        mean = float(np.nanmean(r))
        std = float(np.nanstd(r))
        sharpe = float(mean / std) if std > 0 else float("nan")
        return {"n": int(len(r)), "mean": mean, "std": std, "sharpe": sharpe}

    stats_low = _bucket_stats("low", low_bucket)
    stats_mid = _bucket_stats("mid", mid_bucket)
    stats_high = _bucket_stats("high", high_bucket)

    tprint_info("Bucketed forward-return stats (test set, horizon_bars target):")
    tprint_info(f"  LOW  bucket: n={stats_low['n']:5d}, mean={stats_low['mean']:.6f}, std={stats_low['std']:.6f}, sharpe={stats_low['sharpe']:.3f}")
    tprint_info(f"  MID  bucket: n={stats_mid['n']:5d}, mean={stats_mid['mean']:.6f}, std={stats_mid['std']:.6f}, sharpe={stats_mid['sharpe']:.3f}")
    tprint_info(f"  HIGH bucket: n={stats_high['n']:5d}, mean={stats_high['mean']:.6f}, std={stats_high['std']:.6f}, sharpe={stats_high['sharpe']:.3f}")

    if "fwd_ret_1h" in df_bt.columns:
        stats_low_1h = _bucket_stats("low_1h", low_bucket, "fwd_ret_1h")
        stats_mid_1h = _bucket_stats("mid_1h", mid_bucket, "fwd_ret_1h")
        stats_high_1h = _bucket_stats("high_1h", high_bucket, "fwd_ret_1h")

        tprint_info("Bucketed forward-return stats (test set, ~1h target):")
        tprint_info(f"  LOW  bucket: n={stats_low_1h['n']:5d}, mean={stats_low_1h['mean']:.6f}, std={stats_low_1h['std']:.6f}, sharpe={stats_low_1h['sharpe']:.3f}")
        tprint_info(f"  MID  bucket: n={stats_mid_1h['n']:5d}, mean={stats_mid_1h['mean']:.6f}, std={stats_mid_1h['std']:.6f}, sharpe={stats_mid_1h['sharpe']:.3f}")
        tprint_info(f"  HIGH bucket: n={stats_high_1h['n']:5d}, mean={stats_high_1h['mean']:.6f}, std={stats_high_1h['std']:.6f}, sharpe={stats_high_1h['sharpe']:.3f}")

    # ------------------------------------------------------------------
    # Side-specific bucket stats (support vs resistance)
    # ------------------------------------------------------------------
    support_mask = df_bt["is_support"] > 0.5
    resistance_mask = df_bt["is_resistance"] > 0.5

    def _bucket_stats_side(bucket: pd.DataFrame, side_mask: pd.Series, col: str = "fwd_ret") -> Dict[str, float]:
        if bucket.empty:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "sharpe": float("nan")}
        # Align side mask to bucket index
        local_mask = side_mask.loc[bucket.index]
        sub = bucket.loc[local_mask]
        if sub.empty:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "sharpe": float("nan")}
        return _bucket_stats("side", sub, col)

    tprint_info("Side-specific bucket stats (horizon_bars target):")
    for side_name, side_mask in [("support", support_mask), ("resistance", resistance_mask)]:
        stats_low_s = _bucket_stats_side(low_bucket, side_mask)
        stats_mid_s = _bucket_stats_side(mid_bucket, side_mask)
        stats_high_s = _bucket_stats_side(high_bucket, side_mask)
        tprint_info(
            f"  {side_name:10s} LOW  n={stats_low_s['n']:5d}, mean={stats_low_s['mean']:.6f}, std={stats_low_s['std']:.6f}, sharpe={stats_low_s['sharpe']:.3f}"
        )
        tprint_info(
            f"  {side_name:10s} MID  n={stats_mid_s['n']:5d}, mean={stats_mid_s['mean']:.6f}, std={stats_mid_s['std']:.6f}, sharpe={stats_mid_s['sharpe']:.3f}"
        )
        tprint_info(
            f"  {side_name:10s} HIGH n={stats_high_s['n']:5d}, mean={stats_high_s['mean']:.6f}, std={stats_high_s['std']:.6f}, sharpe={stats_high_s['sharpe']:.3f}"
        )

    if "fwd_ret_1h" in df_bt.columns:
        tprint_info("Side-specific bucket stats (~1h target):")
        for side_name, side_mask in [("support", support_mask), ("resistance", resistance_mask)]:
            stats_low_s_1h = _bucket_stats_side(low_bucket, side_mask, col="fwd_ret_1h")
            stats_mid_s_1h = _bucket_stats_side(mid_bucket, side_mask, col="fwd_ret_1h")
            stats_high_s_1h = _bucket_stats_side(high_bucket, side_mask, col="fwd_ret_1h")
            tprint_info(
                f"  {side_name:10s} LOW  n={stats_low_s_1h['n']:5d}, mean={stats_low_s_1h['mean']:.6f}, std={stats_low_s_1h['std']:.6f}, sharpe={stats_low_s_1h['sharpe']:.3f}"
            )
            tprint_info(
                f"  {side_name:10s} MID  n={stats_mid_s_1h['n']:5d}, mean={stats_mid_s_1h['mean']:.6f}, std={stats_mid_s_1h['std']:.6f}, sharpe={stats_mid_s_1h['sharpe']:.3f}"
            )
            tprint_info(
                f"  {side_name:10s} HIGH n={stats_high_s_1h['n']:5d}, mean={stats_high_s_1h['mean']:.6f}, std={stats_high_s_1h['std']:.6f}, sharpe={stats_high_s_1h['sharpe']:.3f}"
            )

    def _run_bucket_stats_subset(df_subset: pd.DataFrame, label: str) -> None:
        if df_subset.empty:
            tprint_warning(f"No test events in subset '{label}'; skipping subset bucket stats.")
            return

        q_low_s = df_subset["pred_strong_prob"].quantile(0.2)
        q_high_s = df_subset["pred_strong_prob"].quantile(0.8)

        low_s = df_subset[df_subset["pred_strong_prob"] <= q_low_s]
        mid_s = df_subset[(df_subset["pred_strong_prob"] > q_low_s) & (df_subset["pred_strong_prob"] < q_high_s)]
        high_s = df_subset[df_subset["pred_strong_prob"] >= q_high_s]

        stats_low_s = _bucket_stats("low", low_s)
        stats_mid_s = _bucket_stats("mid", mid_s)
        stats_high_s = _bucket_stats("high", high_s)

        tprint_info(f"Subset '{label}' bucketed forward-return stats (horizon_bars target):")
        tprint_info(
            f"  LOW  bucket: n={stats_low_s['n']:5d}, mean={stats_low_s['mean']:.6f}, std={stats_low_s['std']:.6f}, sharpe={stats_low_s['sharpe']:.3f}"
        )
        tprint_info(
            f"  MID  bucket: n={stats_mid_s['n']:5d}, mean={stats_mid_s['mean']:.6f}, std={stats_mid_s['std']:.6f}, sharpe={stats_mid_s['sharpe']:.3f}"
        )
        tprint_info(
            f"  HIGH bucket: n={stats_high_s['n']:5d}, mean={stats_high_s['mean']:.6f}, std={stats_high_s['std']:.6f}, sharpe={stats_high_s['sharpe']:.3f}"
        )

        if "fwd_ret_1h" in df_subset.columns:
            stats_low_s_1h = _bucket_stats("low_1h", low_s, "fwd_ret_1h")
            stats_mid_s_1h = _bucket_stats("mid_1h", mid_s, "fwd_ret_1h")
            stats_high_s_1h = _bucket_stats("high_1h", high_s, "fwd_ret_1h")

            tprint_info(f"Subset '{label}' bucketed forward-return stats (~1h target):")
            tprint_info(
                f"  LOW  bucket: n={stats_low_s_1h['n']:5d}, mean={stats_low_s_1h['mean']:.6f}, std={stats_low_s_1h['std']:.6f}, sharpe={stats_low_s_1h['sharpe']:.3f}"
            )
            tprint_info(
                f"  MID  bucket: n={stats_mid_s_1h['n']:5d}, mean={stats_mid_s_1h['mean']:.6f}, std={stats_mid_s_1h['std']:.6f}, sharpe={stats_mid_s_1h['sharpe']:.3f}"
            )
            tprint_info(
                f"  HIGH bucket: n={stats_high_s_1h['n']:5d}, mean={stats_high_s_1h['mean']:.6f}, std={stats_high_s_1h['std']:.6f}, sharpe={stats_high_s_1h['sharpe']:.3f}"
            )

        support_mask_subset = df_subset["is_support"] > 0.5
        resistance_mask_subset = df_subset["is_resistance"] > 0.5

        tprint_info(f"Subset '{label}' side-specific bucket stats (horizon_bars target):")
        for side_name, side_mask in [("support", support_mask_subset), ("resistance", resistance_mask_subset)]:
            stats_low_side = _bucket_stats_side(low_s, side_mask)
            stats_mid_side = _bucket_stats_side(mid_s, side_mask)
            stats_high_side = _bucket_stats_side(high_s, side_mask)
            tprint_info(
                f"  {side_name:10s} LOW  n={stats_low_side['n']:5d}, mean={stats_low_side['mean']:.6f}, std={stats_low_side['std']:.6f}, sharpe={stats_low_side['sharpe']:.3f}"
            )
            tprint_info(
                f"  {side_name:10s} MID  n={stats_mid_side['n']:5d}, mean={stats_mid_side['mean']:.6f}, std={stats_mid_side['std']:.6f}, sharpe={stats_mid_side['sharpe']:.3f}"
            )
            tprint_info(
                f"  {side_name:10s} HIGH n={stats_high_side['n']:5d}, mean={stats_high_side['mean']:.6f}, std={stats_high_side['std']:.6f}, sharpe={stats_high_side['sharpe']:.3f}"
            )

        if "fwd_ret_1h" in df_subset.columns:
            tprint_info(f"Subset '{label}' side-specific bucket stats (~1h target):")
            for side_name, side_mask in [("support", support_mask_subset), ("resistance", resistance_mask_subset)]:
                stats_low_side_1h = _bucket_stats_side(low_s, side_mask, col="fwd_ret_1h")
                stats_mid_side_1h = _bucket_stats_side(mid_s, side_mask, col="fwd_ret_1h")
                stats_high_side_1h = _bucket_stats_side(high_s, side_mask, col="fwd_ret_1h")
                tprint_info(
                    f"  {side_name:10s} LOW  n={stats_low_side_1h['n']:5d}, mean={stats_low_side_1h['mean']:.6f}, std={stats_low_side_1h['std']:.6f}, sharpe={stats_low_side_1h['sharpe']:.3f}"
                )
                tprint_info(
                    f"  {side_name:10s} MID  n={stats_mid_side_1h['n']:5d}, mean={stats_mid_side_1h['mean']:.6f}, std={stats_mid_side_1h['std']:.6f}, sharpe={stats_mid_side_1h['sharpe']:.3f}"
                )
                tprint_info(
                    f"  {side_name:10s} HIGH n={stats_high_side_1h['n']:5d}, mean={stats_high_side_1h['mean']:.6f}, std={stats_high_side_1h['std']:.6f}, sharpe={stats_high_side_1h['sharpe']:.3f}"
                )

    if "level_hist_event_count" in X.columns:
        df_bt["level_hist_event_count"] = X_test["level_hist_event_count"]
        hist_vals = df_bt["level_hist_event_count"].replace([np.inf, -np.inf], np.nan)
        if hist_vals.notna().sum() >= 50 and hist_vals.max() > 0:
            hist_thr = float(hist_vals.quantile(0.75))
            mask_high_hist = hist_vals >= hist_thr
            n_high_hist = int(mask_high_hist.sum())
            tprint_info(
                f"High-history subset: level_hist_event_count >= {hist_thr:.1f} (n={n_high_hist}) on test set"
            )
            _run_bucket_stats_subset(df_bt.loc[mask_high_hist], "high level_hist_event_count")

    if "weighted_confluence_score" in X.columns:
        df_bt["weighted_confluence_score"] = X_test["weighted_confluence_score"]
        conf_vals = df_bt["weighted_confluence_score"].replace([np.inf, -np.inf], np.nan)
        if conf_vals.notna().sum() >= 50 and conf_vals.max() > 0:
            conf_thr = float(conf_vals.quantile(0.75))
            mask_high_conf = conf_vals >= conf_thr
            n_high_conf = int(mask_high_conf.sum())
            tprint_info(
                f"High-confluence subset: weighted_confluence_score >= {conf_thr:.3f} (n={n_high_conf}) on test set"
            )
            _run_bucket_stats_subset(df_bt.loc[mask_high_conf], "high weighted_confluence_score")

    # ------------------------------------------------------------------
    # Decile-level calibration by predicted strong-move probability
    # ------------------------------------------------------------------
    try:
        df_bt["prob_decile"] = pd.qcut(
            df_bt["pred_strong_prob"],
            q=10,
            labels=False,
            duplicates="drop",
        )
    except Exception as exc:  # pragma: no cover - defensive
        tprint_warning(f"Failed to compute decile calibration: {exc}")
        df_bt["prob_decile"] = np.nan

    if df_bt["prob_decile"].notna().any():
        tprint_info("Decile calibration (horizon_bars target, all sides):")
        for dec in sorted(df_bt["prob_decile"].dropna().unique()):
            mask = df_bt["prob_decile"] == dec
            n = int(mask.sum())
            if n == 0:
                continue
            sub = df_bt.loc[mask]
            avg_prob = float(sub["pred_strong_prob"].mean())
            mean_abs_ret = float(sub["fwd_ret"].abs().mean())
            # Bounce vs break fractions (sign of fwd_ret encodes away-from-level vs through-level)
            bounce_frac = float((sub["fwd_ret"] > 0).mean())
            break_frac = float((sub["fwd_ret"] < 0).mean())
            sup_mask = sub["is_support"] > 0.5
            res_mask = sub["is_resistance"] > 0.5
            mean_sup = float(sub.loc[sup_mask, "fwd_ret"].mean()) if sup_mask.any() else float("nan")
            mean_res = float(sub.loc[res_mask, "fwd_ret"].mean()) if res_mask.any() else float("nan")
            tprint_info(
                f"  decile={int(dec):2d} n={n:5d}, avg_prob={avg_prob:.3f}, "
                f"mean|ret|={mean_abs_ret:.6f}, bounce_frac={bounce_frac:.3f}, break_frac={break_frac:.3f}, "
                f"mean_ret_support={mean_sup:.6f}, mean_ret_resistance={mean_res:.6f}"
            )

        if "fwd_ret_1h" in df_bt.columns:
            tprint_info("Decile calibration (~1h target, all sides):")
            for dec in sorted(df_bt["prob_decile"].dropna().unique()):
                mask = df_bt["prob_decile"] == dec
                n = int(mask.sum())
                if n == 0:
                    continue
                sub = df_bt.loc[mask]
                mean_abs_ret_1h = float(sub["fwd_ret_1h"].abs().mean())
                bounce_frac_1h = float((sub["fwd_ret_1h"] > 0).mean())
                break_frac_1h = float((sub["fwd_ret_1h"] < 0).mean())
                sup_mask = sub["is_support"] > 0.5
                res_mask = sub["is_resistance"] > 0.5
                mean_sup_1h = float(sub.loc[sup_mask, "fwd_ret_1h"].mean()) if sup_mask.any() else float("nan")
                mean_res_1h = float(sub.loc[res_mask, "fwd_ret_1h"].mean()) if res_mask.any() else float("nan")
                tprint_info(
                    f"  decile={int(dec):2d} n={n:5d}, mean|ret_1h|={mean_abs_ret_1h:.6f}, "
                    f"bounce_frac_1h={bounce_frac_1h:.3f}, break_frac_1h={break_frac_1h:.3f}, "
                    f"mean_ret_1h_support={mean_sup_1h:.6f}, mean_ret_1h_resistance={mean_res_1h:.6f}"
                )

            def _discrete_mi(x: pd.Series, y: pd.Series) -> float:
                valid = x.notna() & y.notna()
                if not valid.any():
                    return float("nan")
                xv = x.loc[valid]
                yv = y.loc[valid]
                joint = pd.crosstab(xv, yv, normalize=True)
                px = joint.sum(axis=1)
                py = joint.sum(axis=0)
                mi_val = 0.0
                for xi in joint.index:
                    for yi in joint.columns:
                        pxy = float(joint.loc[xi, yi])
                        if pxy <= 0.0:
                            continue
                        denom = float(px[xi] * py[yi])
                        if denom <= 0.0:
                            continue
                        mi_val += pxy * np.log(pxy / denom)
                return float(mi_val)

            dir_1h = np.sign(df_bt["fwd_ret_1h"]).astype(int)
            mi_prob_dir_nats = _discrete_mi(df_bt["prob_decile"], dir_1h)
            mi_prob_dir_bits = (
                mi_prob_dir_nats / np.log(2.0) if np.isfinite(mi_prob_dir_nats) else float("nan")
            )
            mi_strong_dir_nats = _discrete_mi(y_cls_test, dir_1h.loc[y_cls_test.index])
            mi_strong_dir_bits = (
                mi_strong_dir_nats / np.log(2.0) if np.isfinite(mi_strong_dir_nats) else float("nan")
            )
            tprint_info(
                f"Mutual information (prob_decile -> sign(fwd_ret_1h)): {mi_prob_dir_bits:.4f} bits"
            )
            tprint_info(
                f"Mutual information (strong_label -> sign(fwd_ret_1h)): {mi_strong_dir_bits:.4f} bits"
            )

            if "level_hist_event_count" in df_bt.columns:
                hist_vals_mi = df_bt["level_hist_event_count"].replace([np.inf, -np.inf], np.nan)
                if hist_vals_mi.notna().sum() >= 50 and hist_vals_mi.max() > 0:
                    hist_thr_mi = float(hist_vals_mi.quantile(0.75))
                    mask_high_hist_mi = hist_vals_mi >= hist_thr_mi
                    x_prob_hist = df_bt.loc[mask_high_hist_mi, "prob_decile"]
                    y_dir_hist = dir_1h.loc[mask_high_hist_mi]
                    mi_prob_dir_hist_nats = _discrete_mi(x_prob_hist, y_dir_hist)
                    mi_prob_dir_hist_bits = (
                        mi_prob_dir_hist_nats / np.log(2.0) if np.isfinite(mi_prob_dir_hist_nats) else float("nan")
                    )
                    strong_hist = y_cls_test.loc[x_prob_hist.index]
                    mi_strong_dir_hist_nats = _discrete_mi(strong_hist, y_dir_hist.loc[strong_hist.index])
                    mi_strong_dir_hist_bits = (
                        mi_strong_dir_hist_nats / np.log(2.0) if np.isfinite(mi_strong_dir_hist_nats) else float("nan")
                    )
                    tprint_info(
                        f"Mutual information (prob_decile -> sign(fwd_ret_1h)) [high level_hist_event_count]: {mi_prob_dir_hist_bits:.4f} bits"
                    )
                    tprint_info(
                        f"Mutual information (strong_label -> sign(fwd_ret_1h)) [high level_hist_event_count]: {mi_strong_dir_hist_bits:.4f} bits"
                    )

            if "weighted_confluence_score" in df_bt.columns:
                conf_vals_mi = df_bt["weighted_confluence_score"].replace([np.inf, -np.inf], np.nan)
                if conf_vals_mi.notna().sum() >= 50 and conf_vals_mi.max() > 0:
                    conf_thr_mi = float(conf_vals_mi.quantile(0.75))
                    mask_high_conf_mi = conf_vals_mi >= conf_thr_mi
                    x_prob_conf = df_bt.loc[mask_high_conf_mi, "prob_decile"]
                    y_dir_conf = dir_1h.loc[mask_high_conf_mi]
                    mi_prob_dir_conf_nats = _discrete_mi(x_prob_conf, y_dir_conf)
                    mi_prob_dir_conf_bits = (
                        mi_prob_dir_conf_nats / np.log(2.0) if np.isfinite(mi_prob_dir_conf_nats) else float("nan")
                    )
                    strong_conf = y_cls_test.loc[x_prob_conf.index]
                    mi_strong_dir_conf_nats = _discrete_mi(strong_conf, y_dir_conf.loc[strong_conf.index])
                    mi_strong_dir_conf_bits = (
                        mi_strong_dir_conf_nats / np.log(2.0) if np.isfinite(mi_strong_dir_conf_nats) else float("nan")
                    )
                    tprint_info(
                        f"Mutual information (prob_decile -> sign(fwd_ret_1h)) [high weighted_confluence_score]: {mi_prob_dir_conf_bits:.4f} bits"
                    )
                    tprint_info(
                        f"Mutual information (strong_label -> sign(fwd_ret_1h)) [high weighted_confluence_score]: {mi_strong_dir_conf_bits:.4f} bits"
                    )

    # ------------------------------------------------------------------
    # 1h directional classifier (sign of fwd_ret_1h)
    # ------------------------------------------------------------------
    if y_reg_1h is not None:
        y_dir_full = np.sign(y_reg_1h.loc[X.index])
        dir_mask = y_dir_full != 0
        n_dir = int(dir_mask.sum())
        if n_dir >= 500:
            tprint_info(f"Training 1h directional classifier on {n_dir} non-flat events...")
            X_dir = X.loc[dir_mask]
            # 1 = move away from level (bounce), 0 = move through (break) under our sign convention
            y_dir = (y_dir_full.loc[dir_mask] > 0).astype(int)

            n_d = len(X_dir)
            train_end_d = int(0.6 * n_d)
            val_end_d = int(0.8 * n_d)

            Xd_train = X_dir.iloc[:train_end_d]
            yd_train = y_dir.loc[Xd_train.index]
            Xd_val = X_dir.iloc[train_end_d:val_end_d]
            yd_val = y_dir.loc[Xd_val.index]
            Xd_test = X_dir.iloc[val_end_d:]
            yd_test = y_dir.loc[Xd_test.index]

            clf_dir = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
            )

            clf_dir.fit(Xd_train.values, yd_train.values)
            proba_dir_test = clf_dir.predict_proba(Xd_test.values)[:, 1]
            preds_dir = (proba_dir_test >= 0.5).astype(int)
            acc_dir = float((preds_dir == yd_test.values).mean())
            pos_rate = float(yd_test.mean())
            tprint_info(
                f"1h directional classifier (sign(fwd_ret_1h)): test n={len(yd_test)}, "
                f"pos_rate={pos_rate:.3f}, acc@0.5={acc_dir:.3f}"
            )

            try:
                prob_matrix_dir = np.column_stack([1.0 - proba_dir_test, proba_dir_test])
                cal_dir = calculate_calibration_metrics(yd_test.values, prob_matrix_dir)
                if cal_dir.get("brier_score") is not None:
                    tprint_info(
                        "1h directional classifier calibration: "
                        f"Brier={cal_dir['brier_score']:.4f}, "
                        f"ECE={cal_dir.get('expected_calibration_error', 0.0):.4f}, "
                        f"quality={cal_dir.get('calibration_quality', 'unknown')}"
                    )
            except Exception as exc:  # pragma: no cover - defensive
                tprint_warning(f"Directional calibration metrics failed: {exc}")

            # Feature importances for directional classifier
            dir_top = _get_top_features(clf_dir)
            _print_importances(dir_top, "1h directional classifier (sign(fwd_ret_1h))")
        else:
            tprint_warning(f"Too few non-flat 1h events ({n_dir}) for directional classifier; skipping.")

    # ------------------------------------------------------------------
    # Side-specific strong/weak classifiers (support vs resistance)
    # ------------------------------------------------------------------
    if "is_support" in X.columns and "is_resistance" in X.columns:
        for side_name, side_col in [("support", "is_support"), ("resistance", "is_resistance")]:
            side_mask_all = X[side_col] > 0.5
            n_side = int(side_mask_all.sum())
            if n_side < 200:
                tprint_warning(f"Too few {side_name} events ({n_side}) for side-specific classifier; skipping.")
                continue

            X_side = X.loc[side_mask_all]
            y_cls_side = y_cls.loc[X_side.index]

            n_s = len(X_side)
            train_end_s = int(0.6 * n_s)
            val_end_s = int(0.8 * n_s)

            Xs_train = X_side.iloc[:train_end_s]
            y_train_s = y_cls_side.loc[Xs_train.index]
            Xs_val = X_side.iloc[train_end_s:val_end_s]
            y_val_s = y_cls_side.loc[Xs_val.index]
            Xs_test = X_side.iloc[val_end_s:]
            y_test_s = y_cls_side.loc[Xs_test.index]

            clf_side = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
            )

            tprint_info(f"Training side-specific XGBClassifier for {side_name} strong vs weak...")
            clf_side.fit(Xs_train.values, y_train_s.values)
            proba_side_test = clf_side.predict_proba(Xs_test.values)[:, 1]
            preds_side = (proba_side_test >= 0.5).astype(int)
            acc_side = float((preds_side == y_test_s.values).mean())
            strong_rate_test = float(y_test_s.mean())
            tprint_info(
                f"{side_name.capitalize()} classifier: test n={len(y_test_s)}, "
                f"strong_rate={strong_rate_test:.3f}, acc@0.5={acc_side:.3f}"
            )

            # Calibration metrics for side-specific classifier
            try:
                prob_matrix_side = np.column_stack([1.0 - proba_side_test, proba_side_test])
                cal_side = calculate_calibration_metrics(y_test_s.values, prob_matrix_side)
                if cal_side.get("brier_score") is not None:
                    tprint_info(
                        f"{side_name.capitalize()} classifier calibration: "
                        f"Brier={cal_side['brier_score']:.4f}, "
                        f"ECE={cal_side.get('expected_calibration_error', 0.0):.4f}, "
                        f"quality={cal_side.get('calibration_quality', 'unknown')}"
                    )
            except Exception as exc:  # pragma: no cover - defensive
                tprint_warning(f"Calibration metrics failed for {side_name} classifier: {exc}")

    # ------------------------------------------------------------------
    # Strong bounce vs strong break classifier (path-aware if y_bb provided)
    # ------------------------------------------------------------------
    if y_bb is not None:
        strong_mask_all = (y_cls == 1) & y_bb.notna()
        n_strong = int(strong_mask_all.sum())
        if n_strong >= 200:
            tprint_info(f"Training bounce-vs-break classifier on {n_strong} strong events (path-aware)...")
            X_strong = X.loc[strong_mask_all]
            y_bb_full = y_bb.loc[X_strong.index].astype(int)

            n_s = len(X_strong)
            train_end_s = int(0.6 * n_s)
            val_end_s = int(0.8 * n_s)

            Xs_train = X_strong.iloc[:train_end_s]
            ybb_train = y_bb_full.loc[Xs_train.index]
            Xs_val = X_strong.iloc[train_end_s:val_end_s]
            ybb_val = y_bb_full.loc[Xs_val.index]
            Xs_test = X_strong.iloc[val_end_s:]
            ybb_test = y_bb_full.loc[Xs_test.index]

            clf_bb = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
            )

            clf_bb.fit(Xs_train.values, ybb_train.values)
            proba_bb_test = clf_bb.predict_proba(Xs_test.values)[:, 1]
            preds_bb = (proba_bb_test >= 0.5).astype(int)
            acc_bb = float((preds_bb == ybb_test.values).mean())
            break_rate_test = float(ybb_test.mean())
            tprint_info(
                f"Bounce/break classifier (strong events, global): test n={len(ybb_test)}, "
                f"break_rate={break_rate_test:.3f}, acc@0.5={acc_bb:.3f}"
            )

            try:
                prob_matrix_bb = np.column_stack([1.0 - proba_bb_test, proba_bb_test])
                cal_bb = calculate_calibration_metrics(ybb_test.values, prob_matrix_bb)
                if cal_bb.get("brier_score") is not None:
                    tprint_info(
                        "Bounce/break classifier calibration (global): "
                        f"Brier={cal_bb['brier_score']:.4f}, "
                        f"ECE={cal_bb.get('expected_calibration_error', 0.0):.4f}, "
                        f"quality={cal_bb.get('calibration_quality', 'unknown')}"
                    )
            except Exception as exc:  # pragma: no cover - defensive
                tprint_warning(f"Bounce/break calibration metrics failed: {exc}")

            try:
                proba_strong_on_bb = clf.predict_proba(Xs_test.values)[:, 1]
            except Exception:
                proba_strong_on_bb = None

            if proba_strong_on_bb is not None:
                df_quad = pd.DataFrame(index=Xs_test.index)
                df_quad["p_strong"] = proba_strong_on_bb
                df_quad["p_break"] = proba_bb_test
                df_quad["score_strong_break"] = df_quad["p_strong"] * df_quad["p_break"]
                df_quad["score_strong_bounce"] = df_quad["p_strong"] * (1.0 - df_quad["p_break"])
                df_quad["fwd_ret"] = y_reg.loc[Xs_test.index]
                if y_reg_1h is not None:
                    df_quad["fwd_ret_1h"] = y_reg_1h.loc[Xs_test.index]
                if "is_support" in X.columns:
                    df_quad["is_support"] = X.loc[Xs_test.index, "is_support"]
                else:
                    df_quad["is_support"] = 0.0
                if "is_resistance" in X.columns:
                    df_quad["is_resistance"] = X.loc[Xs_test.index, "is_resistance"]
                else:
                    df_quad["is_resistance"] = 0.0

                def _quad_side_stats(df_subset: pd.DataFrame, label: str) -> None:
                    if df_subset.empty:
                        tprint_warning(f"No events in quadrant subset '{label}'; skipping.")
                        return

                    for side_name, side_col in [("support", "is_support"), ("resistance", "is_resistance")]:
                        side_mask = df_subset[side_col] > 0.5
                        df_side = df_subset.loc[side_mask]
                        if df_side.empty:
                            continue
                        stats_main = _bucket_stats(label, df_side)
                        tprint_info(
                            f"Quadrant '{label}', {side_name}: n={stats_main['n']:5d}, "
                            f"mean={stats_main['mean']:.6f}, std={stats_main['std']:.6f}, sharpe={stats_main['sharpe']:.3f}"
                        )
                        if "fwd_ret_1h" in df_side.columns:
                            stats_1h = _bucket_stats(label + "_1h", df_side, col="fwd_ret_1h")
                            tprint_info(
                                f"Quadrant '{label}', {side_name} (~1h): n={stats_1h['n']:5d}, "
                                f"mean={stats_1h['mean']:.6f}, std={stats_1h['std']:.6f}, sharpe={stats_1h['sharpe']:.3f}"
                            )

                        if "trend_regime_sma" in df_side.columns:
                            mask_tr = df_side["trend_regime_sma"] > 0.5
                            df_tr = df_side.loc[mask_tr]
                            if not df_tr.empty:
                                stats_tr = _bucket_stats(label, df_tr)
                                tprint_info(
                                    f"Quadrant '{label}', {side_name}, trend_regime_sma=1: n={stats_tr['n']:5d}, "
                                    f"mean={stats_tr['mean']:.6f}, std={stats_tr['std']:.6f}, sharpe={stats_tr['sharpe']:.3f}"
                                )
                                if "fwd_ret_1h" in df_tr.columns:
                                    stats_tr_1h = _bucket_stats(label + "_1h", df_tr, col="fwd_ret_1h")
                                    tprint_info(
                                        f"Quadrant '{label}', {side_name}, trend_regime_sma=1 (~1h): n={stats_tr_1h['n']:5d}, "
                                        f"mean={stats_tr_1h['mean']:.6f}, std={stats_tr_1h['std']:.6f}, sharpe={stats_tr_1h['sharpe']:.3f}"
                                    )

                        if "trend_into_level_4h" in df_side.columns:
                            mask_in = df_side["trend_into_level_4h"] > 0.0
                            df_in = df_side.loc[mask_in]
                            if not df_in.empty:
                                stats_in = _bucket_stats(label, df_in)
                                tprint_info(
                                    f"Quadrant '{label}', {side_name}, trend_into_level_4h>0: n={stats_in['n']:5d}, "
                                    f"mean={stats_in['mean']:.6f}, std={stats_in['std']:.6f}, sharpe={stats_in['sharpe']:.3f}"
                                )
                                if "fwd_ret_1h" in df_in.columns:
                                    stats_in_1h = _bucket_stats(label + "_1h", df_in, col="fwd_ret_1h")
                                    tprint_info(
                                        f"Quadrant '{label}', {side_name}, trend_into_level_4h>0 (~1h): n={stats_in_1h['n']:5d}, "
                                        f"mean={stats_in_1h['mean']:.6f}, std={stats_in_1h['std']:.6f}, sharpe={stats_in_1h['sharpe']:.3f}"
                                    )

                        if "is_compressed" in df_side.columns:
                            mask_comp = df_side["is_compressed"] > 0.5
                            df_comp = df_side.loc[mask_comp]
                            if not df_comp.empty:
                                stats_comp = _bucket_stats(label, df_comp)
                                tprint_info(
                                    f"Quadrant '{label}', {side_name}, is_compressed=1: n={stats_comp['n']:5d}, "
                                    f"mean={stats_comp['mean']:.6f}, std={stats_comp['std']:.6f}, sharpe={stats_comp['sharpe']:.3f}"
                                )
                                if "fwd_ret_1h" in df_comp.columns:
                                    stats_comp_1h = _bucket_stats(label + "_1h", df_comp, col="fwd_ret_1h")
                                    tprint_info(
                                        f"Quadrant '{label}', {side_name}, is_compressed=1 (~1h): n={stats_comp_1h['n']:5d}, "
                                        f"mean={stats_comp_1h['mean']:.6f}, std={stats_comp_1h['std']:.6f}, sharpe={stats_comp_1h['sharpe']:.3f}"
                                    )

                        if "level_hist_event_count" in df_side.columns:
                            hist_vals_q = df_side["level_hist_event_count"].replace([np.inf, -np.inf], np.nan)
                            if hist_vals_q.notna().sum() >= 20 and hist_vals_q.max() > 0:
                                hist_thr_q = float(hist_vals_q.quantile(0.75))
                                mask_hist = hist_vals_q >= hist_thr_q
                                df_hist = df_side.loc[mask_hist]
                                if not df_hist.empty:
                                    stats_hist = _bucket_stats(label, df_hist)
                                    tprint_info(
                                        f"Quadrant '{label}', {side_name}, high_history: n={stats_hist['n']:5d}, "
                                        f"mean={stats_hist['mean']:.6f}, std={stats_hist['std']:.6f}, sharpe={stats_hist['sharpe']:.3f}"
                                    )
                                    if "fwd_ret_1h" in df_hist.columns:
                                        stats_hist_1h = _bucket_stats(label + "_1h", df_hist, col="fwd_ret_1h")
                                        tprint_info(
                                            f"Quadrant '{label}', {side_name}, high_history (~1h): n={stats_hist_1h['n']:5d}, "
                                            f"mean={stats_hist_1h['mean']:.6f}, std={stats_hist_1h['std']:.6f}, sharpe={stats_hist_1h['sharpe']:.3f}"
                                        )

                        if "weighted_confluence_score" in df_side.columns:
                            conf_vals_q = df_side["weighted_confluence_score"].replace([np.inf, -np.inf], np.nan)
                            if conf_vals_q.notna().sum() >= 20 and conf_vals_q.max() > 0:
                                conf_thr_q = float(conf_vals_q.quantile(0.75))
                                mask_conf = conf_vals_q >= conf_thr_q
                                df_conf = df_side.loc[mask_conf]
                                if not df_conf.empty:
                                    stats_conf = _bucket_stats(label, df_conf)
                                    tprint_info(
                                        f"Quadrant '{label}', {side_name}, high_confluence: n={stats_conf['n']:5d}, "
                                        f"mean={stats_conf['mean']:.6f}, std={stats_conf['std']:.6f}, sharpe={stats_conf['sharpe']:.3f}"
                                    )
                                    if "fwd_ret_1h" in df_conf.columns:
                                        stats_conf_1h = _bucket_stats(label + "_1h", df_conf, col="fwd_ret_1h")
                                        tprint_info(
                                            f"Quadrant '{label}', {side_name}, high_confluence (~1h): n={stats_conf_1h['n']:5d}, "
                                            f"mean={stats_conf_1h['mean']:.6f}, std={stats_conf_1h['std']:.6f}, sharpe={stats_conf_1h['sharpe']:.3f}"
                                        )

                for quad_label, score_col in [("strong_break", "score_strong_break"), ("strong_bounce", "score_strong_bounce")]:
                    scores = df_quad[score_col].replace([np.inf, -np.inf], np.nan)
                    if scores.notna().sum() < 50:
                        continue
                    thr = float(scores.quantile(0.8))
                    mask_top = scores >= thr
                    df_top = df_quad.loc[mask_top]
                    tprint_info(
                        f"Top-quantile subset for {quad_label}: score_col={score_col}, threshold={thr:.4f}, n={len(df_top)}"
                    )
                    _quad_side_stats(df_top, quad_label)

            # Side-specific bounce/break models
            if "is_support" in X.columns and "is_resistance" in X.columns:
                for side_name, side_col in [("support", "is_support"), ("resistance", "is_resistance")]:
                    side_mask = strong_mask_all & (X[side_col] > 0.5)
                    n_side = int(side_mask.sum())
                    if n_side < 200:
                        tprint_warning(
                            f"Too few strong {side_name} events ({n_side}) for side-specific bounce/break classifier; skipping."
                        )
                        continue

                    X_side = X.loc[side_mask]
                    y_bb_side = y_bb.loc[X_side.index].astype(int)

                    n_ss = len(X_side)
                    train_end_ss = int(0.6 * n_ss)
                    val_end_ss = int(0.8 * n_ss)

                    Xs_train_s = X_side.iloc[:train_end_ss]
                    ybb_train_s = y_bb_side.loc[Xs_train_s.index]
                    Xs_val_s = X_side.iloc[train_end_ss:val_end_ss]
                    ybb_val_s = y_bb_side.loc[Xs_val_s.index]
                    Xs_test_s = X_side.iloc[val_end_ss:]
                    ybb_test_s = y_bb_side.loc[Xs_test_s.index]

                    clf_bb_s = xgb.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        tree_method="hist",
                    )

                    clf_bb_s.fit(Xs_train_s.values, ybb_train_s.values)
                    proba_bb_test_s = clf_bb_s.predict_proba(Xs_test_s.values)[:, 1]
                    preds_bb_s = (proba_bb_test_s >= 0.5).astype(int)
                    acc_bb_s = float((preds_bb_s == ybb_test_s.values).mean())
                    break_rate_test_s = float(ybb_test_s.mean())
                    tprint_info(
                        f"Bounce/break classifier ({side_name}, strong events): test n={len(ybb_test_s)}, "
                        f"break_rate={break_rate_test_s:.3f}, acc@0.5={acc_bb_s:.3f}"
                    )

                    try:
                        prob_matrix_bb_s = np.column_stack([
                            1.0 - proba_bb_test_s,
                            proba_bb_test_s,
                        ])
                        cal_bb_s = calculate_calibration_metrics(ybb_test_s.values, prob_matrix_bb_s)
                        if cal_bb_s.get("brier_score") is not None:
                            tprint_info(
                                f"Bounce/break classifier calibration ({side_name}): "
                                f"Brier={cal_bb_s['brier_score']:.4f}, "
                                f"ECE={cal_bb_s.get('expected_calibration_error', 0.0):.4f}, "
                                f"quality={cal_bb_s.get('calibration_quality', 'unknown')}"
                            )
                    except Exception as exc:  # pragma: no cover - defensive
                        tprint_warning(f"Bounce/break calibration metrics failed for {side_name}: {exc}")

            # ------------------------------------------------------------------
            # 3-class classifier: {weak, strong_bounce, strong_break}
            # ------------------------------------------------------------------
            # 0 = weak, 1 = strong_bounce, 2 = strong_break
            y_multi = pd.Series(index=X.index, dtype=float)
            y_multi[:] = np.nan
            y_multi.loc[y_cls == 0] = 0.0
            if y_bb is not None:
                strong_idx = (y_cls == 1) & y_bb.notna()
                y_multi.loc[strong_idx & (y_bb == 0)] = 1.0
                y_multi.loc[strong_idx & (y_bb == 1)] = 2.0

            multi_mask = y_multi.notna()
            n_multi = int(multi_mask.sum())
            if n_multi >= 500:
                tprint_info(f"Training 3-class classifier on {n_multi} events (weak/strong_bounce/strong_break)...")
                X_multi = X.loc[multi_mask]
                y_multi_int = y_multi.loc[multi_mask].astype(int)

                n_m = len(X_multi)
                train_end_m = int(0.6 * n_m)
                val_end_m = int(0.8 * n_m)

                Xm_train = X_multi.iloc[:train_end_m]
                ym_train = y_multi_int.loc[Xm_train.index]
                Xm_val = X_multi.iloc[train_end_m:val_end_m]
                ym_val = y_multi_int.loc[Xm_val.index]
                Xm_test = X_multi.iloc[val_end_m:]
                ym_test = y_multi_int.loc[Xm_test.index]

                clf_multi = xgb.XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    num_class=3,
                )

                clf_multi.fit(Xm_train.values, ym_train.values)
                proba_multi_test = clf_multi.predict_proba(Xm_test.values)
                preds_multi = proba_multi_test.argmax(axis=1)
                acc_multi = float((preds_multi == ym_test.values).mean())
                class_counts = ym_test.value_counts(normalize=True).to_dict()
                class_dist_pct = {int(k): int(v * 100) for k, v in class_counts.items()}
                tprint_info(
                    "3-class classifier (weak/strong_bounce/strong_break): "
                    f"test n={len(ym_test)}, acc={acc_multi:.3f}, "
                    f"class_dist={class_dist_pct}"
                )

                # Feature importances for 3-class classifier
                multi_top = _get_top_features(clf_multi)
                _print_importances(multi_top, "3-class weak/strong_bounce/strong_break classifier")
            else:
                tprint_warning(
                    f"Too few events ({n_multi}) for 3-class classifier; skipping."
                )
        else:
            tprint_warning(
                f"Too few strong events with path-aware labels ({n_strong}) to train bounce/break classifier; skipping."
            )
    else:
        tprint_warning("No path-aware bounce/break labels (y_bb) provided; skipping bounce/break models.")

    # ------------------------------------------------------------------
    # Full reporting by level type and side (all events)
    # ------------------------------------------------------------------
    full = pd.DataFrame(index=X.index)
    full["fwd_ret"] = y_reg
    # Carry over key indicators from feature matrix
    for col in [
        "is_support",
        "is_resistance",
        "lvl_is_hvn",
        "lvl_is_swing",
        "lvl_is_pivot",
    ]:
        if col in X.columns:
            full[col] = X[col]

    def _print_group(name: str, mask: pd.Series) -> None:
        if col not in full.columns:
            return
        bucket = full[mask]
        stats = _bucket_stats(name, bucket)
        tprint_info(
            f"  {name:18s} n={stats['n']:5d}, "
            f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, sharpe={stats['sharpe']:.3f}"
        )

    tprint_info("Per-level-type forward-return stats (all events):")
    if "lvl_is_hvn" in full.columns:
        _print_group("lvl_is_hvn==1", full["lvl_is_hvn"] > 0.5)
    if "lvl_is_swing" in full.columns:
        _print_group("lvl_is_swing==1", full["lvl_is_swing"] > 0.5)
    if "lvl_is_pivot" in full.columns:
        _print_group("lvl_is_pivot==1", full["lvl_is_pivot"] > 0.5)

    tprint_info("Per-side forward-return stats (all events):")
    if "is_support" in full.columns:
        _print_group("support", full["is_support"] > 0.5)
    if "is_resistance" in full.columns:
        _print_group("resistance", full["is_resistance"] > 0.5)


def main() -> None:
    args = parse_args()

    ohlcv = load_market_data(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
    )

    sr = generate_sr_levels(ohlcv)

    features, y_reg, y_cls, y_reg_1h, y_bb = build_event_dataset(
        ohlcv=ohlcv,
        sr=sr,
        horizon_bars=args.horizon_bars,
        min_ret=args.min_ret,
        max_samples=args.max_samples,
        strong_quantile=args.strong_quantile,
        bounce_tolerance_pct=args.bounce_tolerance_pct,
        break_tolerance_pct=args.break_tolerance_pct,
    )

    train_and_report(features, y_reg, y_cls, args.output_features, y_reg_1h=y_reg_1h, y_bb=y_bb)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tprint_warning("Interrupted by user")
        raise SystemExit(1)
    except Exception as exc:
        tprint_error(f"SR strength research script failed: {exc}")
        raise SystemExit(1)
