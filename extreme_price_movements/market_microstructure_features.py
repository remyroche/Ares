"""Causal native-L2 continuation features.

The repository contains a large historical ``orderbook_hourly`` surface that
is generated from OHLCV summaries.  Those rows are useful as explicitly named
proxies, but they are not native order-book observations.  This module is
deliberately strict: it accepts only rows carrying an admitted native-L2
source tag and never silently mixes proxy rows into a research or production
feature panel.

All derived fields are vectorized trailing transforms.  A lagged field is
valid only when the preceding snapshot for the same symbol is no more than two
hours old; large gaps become missing rather than being forward-filled.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


NATIVE_L2_ALLOWED_SOURCES = frozenset({"kraken_futures_l2_snapshot"})

NATIVE_L2_CONTINUATION_FEATURE_KEYS = [
    "l2_spread_bps",
    "l2_top_depth_imbalance",
    "l2_depth_imbalance_l10",
    "l2_depth_imbalance_l20",
    "l2_depth_notional_l20_log",
    "l2_depth_imbalance_notional_l20",
    "l2_bid_shape_l20",
    "l2_ask_shape_l20",
    "l2_depth_shape_imbalance_l20",
    "l2_snapshot_gap_seconds",
    "l2_mid_return_prev_snapshot",
    "l2_spread_delta_prev_snapshot",
    "l2_depth_imbalance_delta_prev_snapshot",
    "l2_depth_total_log_delta_prev_snapshot",
    "l2_depth_imbalance_reversion_prev_snapshot",
    "l2_spread_widening_prev_snapshot",
    "l2_depth_depletion_prev_snapshot",
]

NATIVE_L2_REQUIRED_COLUMNS = (
    "best_bid", "best_ask", "mid", "bid_qty_1", "ask_qty_1",
    "cum_bid_qty_l10", "cum_ask_qty_l10", "cum_bid_qty_l20", "cum_ask_qty_l20",
    "snapshot_ts", "source",
)
NATIVE_L2_OPTIONAL_COLUMNS = ("l2_bid_notional_l20", "l2_ask_notional_l20")

NATIVE_L2_RAW_COLUMNS = (
    "observed_ts",
    "symbol",
    "side",
    "level",
    "price",
    "qty",
    "source",
    "timestamp",
)


class NativeL2FeatureContractError(ValueError):
    """Raised when a panel cannot prove native-L2 provenance."""


def summarize_native_l2_snapshot_rows(
    frame: pd.DataFrame,
    *,
    max_level: int = 20,
) -> pd.DataFrame:
    """Aggregate raw native L2 levels into the causal sidecar schema.

    Raw Kraken snapshots contain one row per product/side/level.  This
    implementation keeps only the declared native source, deduplicates a
    repeated product/side/level within a snapshot, and uses vectorized
    grouped reductions for best prices, depth, and notionals.  The observed
    timestamp is retained as the feature-availability time; the bucket time
    is never substituted for it.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "snapshot_ts", "source", *NATIVE_L2_REQUIRED_COLUMNS])
    work = frame.reset_index() if "timestamp" not in frame.columns else frame.copy()
    missing = sorted(set(NATIVE_L2_RAW_COLUMNS).difference(work.columns))
    if missing:
        raise NativeL2FeatureContractError(f"raw native-L2 snapshot is missing columns: {missing}")
    source = work["source"].astype("string")
    invalid = sorted(set(source.dropna().unique()).difference({"kraken_futures_l2_snapshot"}))
    if invalid:
        raise NativeL2FeatureContractError(
            f"raw native-L2 aggregation refuses non-native source tags: {invalid}"
        )
    work = work.loc[source.eq("kraken_futures_l2_snapshot")].copy()
    if work.empty:
        return pd.DataFrame()
    work["symbol"] = work["symbol"].astype("string")
    work["side"] = work["side"].astype("string").str.lower()
    work["level"] = pd.to_numeric(work["level"], errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["qty"] = pd.to_numeric(work["qty"], errors="coerce")
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.floor("h")
    work["observed_ts"] = pd.to_datetime(work["observed_ts"], utc=True, errors="coerce")
    work = work.loc[
        work["symbol"].notna()
        & work["symbol"].ne("")
        & work["side"].isin(["bid", "ask"])
        & work["level"].between(1, int(max_level))
        & work["price"].gt(0.0)
        & work["qty"].ge(0.0)
        & work["timestamp"].notna()
        & work["observed_ts"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame()
    keys = ["timestamp", "symbol", "side", "level"]
    work = work.sort_values([*keys, "observed_ts"], kind="stable")
    work = work.drop_duplicates(keys, keep="last")
    group_keys = ["timestamp", "symbol"]

    observed = work.groupby(group_keys, sort=True, observed=True)["observed_ts"].max().rename("snapshot_ts")

    def _side_summary(side: str, prefix: str) -> pd.DataFrame:
        side_rows = work.loc[work["side"].eq(side)].copy()
        if side_rows.empty:
            return pd.DataFrame()
        side_rows = side_rows.sort_values([*group_keys, "level"], kind="stable")
        first = side_rows.groupby(group_keys, sort=True, observed=True).head(1).set_index(group_keys)
        grouped = side_rows.groupby(group_keys, sort=True, observed=True)
        qty_l10 = side_rows.loc[side_rows["level"].le(10)].groupby(group_keys, observed=True)["qty"].sum()
        qty_l20 = grouped["qty"].sum()
        notional = side_rows["price"] * side_rows["qty"]
        notional_l20 = notional.groupby([side_rows["timestamp"], side_rows["symbol"]], observed=True).sum()
        output = pd.DataFrame(index=first.index)
        output[f"{prefix}_price_1"] = first["price"]
        output[f"{prefix}_qty_1"] = first["qty"]
        output[f"{prefix}_qty_l10"] = qty_l10
        output[f"{prefix}_qty_l20"] = qty_l20
        output[f"{prefix}_notional_l20"] = notional_l20
        return output

    bids = _side_summary("bid", "bid")
    asks = _side_summary("ask", "ask")
    if bids.empty or asks.empty:
        return pd.DataFrame()
    combined = bids.join(asks, how="inner").join(observed, how="inner")
    combined["mid"] = (combined["bid_price_1"] + combined["ask_price_1"]) * 0.5
    output = combined.reset_index().rename(
        columns={
            "bid_price_1": "best_bid",
            "ask_price_1": "best_ask",
            "bid_qty_1": "bid_qty_1",
            "ask_qty_1": "ask_qty_1",
            "bid_qty_l10": "cum_bid_qty_l10",
            "ask_qty_l10": "cum_ask_qty_l10",
            "bid_qty_l20": "cum_bid_qty_l20",
            "ask_qty_l20": "cum_ask_qty_l20",
            "bid_notional_l20": "l2_bid_notional_l20",
            "ask_notional_l20": "l2_ask_notional_l20",
        }
    )
    output["source"] = "kraken_futures_l2_snapshot"
    keep = [
        "symbol", "best_bid", "best_ask", "mid", "bid_qty_1", "ask_qty_1",
        "cum_bid_qty_l10", "cum_ask_qty_l10", "cum_bid_qty_l20", "cum_ask_qty_l20",
        "snapshot_ts", "l2_bid_notional_l20", "l2_ask_notional_l20", "source",
    ]
    output = output.loc[:, keep]
    output = output.sort_values(["symbol", "snapshot_ts"], kind="stable").reset_index(drop=True)
    if output.duplicated(["symbol", "snapshot_ts"]).any():
        raise NativeL2FeatureContractError("raw native-L2 aggregation produced duplicate symbol/snapshot keys")
    return output


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").astype("float64")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.abs().where(denominator.abs() > 1e-12)


def validate_native_l2_frame(
    frame: pd.DataFrame,
    *,
    symbol_column: str = "symbol",
    timestamp_column: str = "snapshot_ts",
    source_column: str = "source",
    allowed_sources: Iterable[str] = NATIVE_L2_ALLOWED_SOURCES,
) -> pd.DataFrame:
    """Validate and return a stable, UTC-sorted native-L2 input frame."""
    required = set(NATIVE_L2_REQUIRED_COLUMNS) | {symbol_column, timestamp_column, source_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise NativeL2FeatureContractError(f"native-L2 frame is missing columns: {missing}")
    allowed = frozenset(str(value) for value in allowed_sources)
    output = frame.copy()
    source = output[source_column].astype(str)
    invalid_sources = sorted(set(source.unique()).difference(allowed))
    if invalid_sources:
        raise NativeL2FeatureContractError(
            "native-L2 feature generation refuses non-native source tags: "
            f"{invalid_sources}; proxy rows must be filtered before this call"
        )
    output[timestamp_column] = pd.to_datetime(output[timestamp_column], utc=True, errors="coerce")
    if output[timestamp_column].isna().any():
        raise NativeL2FeatureContractError("native-L2 snapshot timestamps must be valid UTC values")
    output[symbol_column] = output[symbol_column].astype(str)
    if output.duplicated([symbol_column, timestamp_column]).any():
        raise NativeL2FeatureContractError("native-L2 input has duplicate symbol/snapshot timestamps")
    positive = ["best_bid", "best_ask", "mid"]
    numeric = {name: _numeric(output, name) for name in positive}
    if any(series.isna().any() or (series <= 0.0).any() for series in numeric.values()):
        raise NativeL2FeatureContractError("native-L2 bid/ask/mid values must be finite and positive")
    if (numeric["best_ask"] < numeric["best_bid"]).any():
        raise NativeL2FeatureContractError("native-L2 best ask is below best bid")
    return output.sort_values([symbol_column, timestamp_column], kind="stable").reset_index(drop=True)


def materialize_native_l2_continuation_features(
    frame: pd.DataFrame,
    *,
    symbol_column: str = "symbol",
    timestamp_column: str = "snapshot_ts",
    source_column: str = "source",
    max_lag_seconds: float = 2.0 * 3600.0,
) -> pd.DataFrame:
    """Generate native-L2 state and one-observation causal change fields."""
    ordered = validate_native_l2_frame(
        frame,
        symbol_column=symbol_column,
        timestamp_column=timestamp_column,
        source_column=source_column,
    )
    symbol = ordered[symbol_column]
    ts = ordered[timestamp_column]
    best_bid = _numeric(ordered, "best_bid")
    best_ask = _numeric(ordered, "best_ask")
    mid = _numeric(ordered, "mid")
    bid_qty_1 = _numeric(ordered, "bid_qty_1").clip(lower=0.0)
    ask_qty_1 = _numeric(ordered, "ask_qty_1").clip(lower=0.0)
    bid_l10 = _numeric(ordered, "cum_bid_qty_l10").clip(lower=0.0)
    ask_l10 = _numeric(ordered, "cum_ask_qty_l10").clip(lower=0.0)
    bid_l20 = _numeric(ordered, "cum_bid_qty_l20").clip(lower=0.0)
    ask_l20 = _numeric(ordered, "cum_ask_qty_l20").clip(lower=0.0)
    if "l2_bid_notional_l20" in ordered:
        bid_notional_l20 = _numeric(ordered, "l2_bid_notional_l20").clip(lower=0.0)
    else:
        bid_notional_l20 = mid * bid_l20
    if "l2_ask_notional_l20" in ordered:
        ask_notional_l20 = _numeric(ordered, "l2_ask_notional_l20").clip(lower=0.0)
    else:
        ask_notional_l20 = mid * ask_l20

    spread_bps = _safe_ratio(best_ask - best_bid, mid) * 10_000.0
    top_imbalance = _safe_ratio(bid_qty_1 - ask_qty_1, bid_qty_1 + ask_qty_1)
    imbalance_l10 = _safe_ratio(bid_l10 - ask_l10, bid_l10 + ask_l10)
    imbalance_l20 = _safe_ratio(bid_l20 - ask_l20, bid_l20 + ask_l20)
    total_notional = bid_notional_l20 + ask_notional_l20
    notional_imbalance = _safe_ratio(bid_notional_l20 - ask_notional_l20, total_notional)
    bid_shape = _safe_ratio(bid_qty_1, bid_l20)
    ask_shape = _safe_ratio(ask_qty_1, ask_l20)
    shape_imbalance = bid_shape - ask_shape

    previous = {
        "mid": mid.groupby(symbol, observed=True).shift(1),
        "spread": spread_bps.groupby(symbol, observed=True).shift(1),
        "imbalance": imbalance_l20.groupby(symbol, observed=True).shift(1),
        "total_notional": total_notional.groupby(symbol, observed=True).shift(1),
    }
    gap_seconds = ts.groupby(symbol, observed=True).diff().dt.total_seconds()
    previous_valid = gap_seconds.le(float(max_lag_seconds)) & gap_seconds.gt(0.0)
    previous_mid = previous["mid"].where(previous_valid)
    previous_spread = previous["spread"].where(previous_valid)
    previous_imbalance = previous["imbalance"].where(previous_valid)
    previous_total = previous["total_notional"].where(previous_valid)
    mid_return = np.log(mid / previous_mid)
    spread_delta = spread_bps - previous_spread
    imbalance_delta = imbalance_l20 - previous_imbalance
    total_log_delta = np.log(total_notional / previous_total)
    reversion = previous_imbalance.abs() - imbalance_l20.abs()

    output = ordered.loc[:, [symbol_column, timestamp_column, source_column]].copy()
    output["feature_available_at"] = ts
    output["l2_spread_bps"] = spread_bps
    output["l2_top_depth_imbalance"] = top_imbalance
    output["l2_depth_imbalance_l10"] = imbalance_l10
    output["l2_depth_imbalance_l20"] = imbalance_l20
    output["l2_depth_notional_l20_log"] = np.log1p(total_notional.clip(lower=0.0))
    output["l2_depth_imbalance_notional_l20"] = notional_imbalance
    output["l2_bid_shape_l20"] = bid_shape
    output["l2_ask_shape_l20"] = ask_shape
    output["l2_depth_shape_imbalance_l20"] = shape_imbalance
    output["l2_snapshot_gap_seconds"] = gap_seconds
    output["l2_mid_return_prev_snapshot"] = mid_return
    output["l2_spread_delta_prev_snapshot"] = spread_delta
    output["l2_depth_imbalance_delta_prev_snapshot"] = imbalance_delta
    output["l2_depth_total_log_delta_prev_snapshot"] = total_log_delta
    output["l2_depth_imbalance_reversion_prev_snapshot"] = reversion
    # Indicators must carry the same availability contract as the continuous
    # change fields.  A boolean ``False`` at a gap would silently claim that
    # a prior snapshot was observed and would turn missingness into signal.
    output["l2_spread_widening_prev_snapshot"] = pd.Series(
        np.where(previous_valid, (spread_delta > 0.0).to_numpy(), np.nan),
        index=output.index,
    )
    output["l2_depth_depletion_prev_snapshot"] = pd.Series(
        np.where(previous_valid, (total_log_delta < 0.0).to_numpy(), np.nan),
        index=output.index,
    )
    numeric_columns = [name for name in NATIVE_L2_CONTINUATION_FEATURE_KEYS if name in output]
    output[numeric_columns] = output[numeric_columns].replace([np.inf, -np.inf], np.nan).astype("float32")
    return output
