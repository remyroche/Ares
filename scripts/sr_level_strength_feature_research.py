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

try:
    import xgboost as xgb  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("xgboost is required for this research script") from exc

try:
    import scipy  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("scipy is required for this research script") from exc


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

    return parser.parse_args()


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

    We combine several level sources to intentionally over-generate levels:
    - RollingKDELevelGenerator (cluster-based levels)
    - FractalLevelGenerator (swing highs/lows)
    - HTFLevelGenerator (previous-day high/low style pivots)

    The output is a single DataFrame aligned to `ohlcv.index` with columns:
    - primary_level_* and opposing_level_* (from unioned generators)
    - is_support / is_resistance flags
    - source indicators (kde/fractal/htf)
    """

    idx = ohlcv.index
    base = pd.DataFrame(index=idx)

    # --- KDE levels ---
    try:
        kde_gen = RollingKDELevelGenerator()
        kde_levels = kde_gen.compute_levels(ohlcv)
        kde_levels = kde_levels.add_prefix("kde_")
        base = base.join(kde_levels, how="left")
        tprint_info("Added KDE-based S/R levels")
    except Exception as exc:
        tprint_warning(f"Failed to compute KDE levels: {exc}")

    # --- Fractal levels ---
    try:
        frac_gen = FractalLevelGenerator()
        frac_levels = frac_gen.compute_levels(ohlcv)
        frac_levels = frac_levels.add_prefix("fractal_")
        base = base.join(frac_levels, how="left")
        tprint_info("Added fractal-based S/R levels")
    except Exception as exc:
        tprint_warning(f"Failed to compute fractal levels: {exc}")

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
        for prefix, src_tag in [("kde_", "kde"), ("fractal_", "fractal"), ("htf_", "htf")]:
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


def build_event_dataset(
    ohlcv: pd.DataFrame,
    sr: pd.DataFrame,
    horizon_bars: int,
    min_ret: float,
    max_samples: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Construct event-level dataset around S/R touches.

    Returns:
        X: feature dataframe (level-strength + push features).
        y_reg: continuous forward return vs level.
        y_cls: binary strong-move indicator (|ret| >= min_ret).
    """

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

    # Align ATR
    atr = atr_14.loc[df.index]

    close = df["close"].astype(float)
    level = df["primary_level_price"].astype(float)

    # Dynamic Touch Band: 0.5 * ATR
    # If ATR is nan (start of data), fallback to 0.4%
    touch_dist_price = 0.5 * atr
    fallback_dist = 0.004 * level
    touch_dist_price = touch_dist_price.fillna(fallback_dist)

    abs_diff = (close - level).abs()
    touch_mask = abs_diff <= touch_dist_price

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
    y_cls = (fwd_ret.abs() >= min_ret).astype(int)

    # ------------------------------------------------------------------
    # Level-strength features
    # ------------------------------------------------------------------
    lvl_feats = pd.DataFrame(index=event_df.index)

    # Basic proximity
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

    # ------------------------------------------------------------------
    # Push / move features around the touch
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

    # Final feature frame
    X = pd.concat([lvl_feats, push_feats], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y_reg, y_cls


def train_and_report(X: pd.DataFrame, y_reg: pd.Series, y_cls: pd.Series) -> None:
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

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------
    def _sorted_importances(model, feature_names: list[str], title: str) -> None:
        try:
            importances = model.feature_importances_
        except Exception:
            tprint_warning(f"Model {title} has no feature_importances_")
            return
        order = np.argsort(importances)[::-1]
        tprint_info(f"Top 20 features for {title}:")
        for idx in order[:20]:
            tprint_info(f"  {feature_names[idx]:30s}  {importances[idx]:.4f}")

    feature_names = list(X.columns)
    _sorted_importances(reg, feature_names, "XGBRegressor (magnitude)")
    _sorted_importances(clf, feature_names, "XGBClassifier (strong/weak)")

    # ------------------------------------------------------------------
    # Bucketed backtest by predicted strength (classifier probabilities)
    # ------------------------------------------------------------------
    tprint_info("Computing bucketed backtest by predicted strong-move probability...")
    proba_test = clf.predict_proba(X_test.values)[:, 1]
    df_bt = pd.DataFrame(index=X_test.index)
    df_bt["pred_strong_prob"] = proba_test
    df_bt["fwd_ret"] = y_reg_test

    # Quantile buckets: bottom 20%, mid 60%, top 20%
    q_low = df_bt["pred_strong_prob"].quantile(0.2)
    q_high = df_bt["pred_strong_prob"].quantile(0.8)

    low_bucket = df_bt[df_bt["pred_strong_prob"] <= q_low]
    mid_bucket = df_bt[(df_bt["pred_strong_prob"] > q_low) & (df_bt["pred_strong_prob"] < q_high)]
    high_bucket = df_bt[df_bt["pred_strong_prob"] >= q_high]

    def _bucket_stats(name: str, bucket: pd.DataFrame) -> Dict[str, float]:
        if bucket.empty:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "sharpe": float("nan")}
        r = bucket["fwd_ret"].values
        mean = float(np.nanmean(r))
        std = float(np.nanstd(r))
        sharpe = float(mean / std) if std > 0 else float("nan")
        return {"n": int(len(r)), "mean": mean, "std": std, "sharpe": sharpe}

    stats_low = _bucket_stats("low", low_bucket)
    stats_mid = _bucket_stats("mid", mid_bucket)
    stats_high = _bucket_stats("high", high_bucket)

    tprint_info("Bucketed forward-return stats (test set):")
    tprint_info(f"  LOW  bucket: n={stats_low['n']:5d}, mean={stats_low['mean']:.6f}, std={stats_low['std']:.6f}, sharpe={stats_low['sharpe']:.3f}")
    tprint_info(f"  MID  bucket: n={stats_mid['n']:5d}, mean={stats_mid['mean']:.6f}, std={stats_mid['std']:.6f}, sharpe={stats_mid['sharpe']:.3f}")
    tprint_info(f"  HIGH bucket: n={stats_high['n']:5d}, mean={stats_high['mean']:.6f}, std={stats_high['std']:.6f}, sharpe={stats_high['sharpe']:.3f}")

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

    X, y_reg, y_cls = build_event_dataset(
        ohlcv=ohlcv,
        sr=sr,
        horizon_bars=args.horizon_bars,
        min_ret=args.min_ret,
        max_samples=args.max_samples,
    )

    train_and_report(X, y_reg, y_cls)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tprint_warning("Interrupted by user")
        raise SystemExit(1)
    except Exception as exc:
        tprint_error(f"SR strength research script failed: {exc}")
        raise SystemExit(1)
