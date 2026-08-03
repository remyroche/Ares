"""Causal raw-market-state inputs for execution-EV research.

This is deliberately a narrow adapter around the point-in-time feature store.
It never derives a row from the candidate population, ranks, outcomes, or a
future store row: every join is an as-of backward join bounded by a declared
staleness.  The source timestamp is retained so experiments can prove this.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


MARKET_STATE_SCHEMA_VERSION = "execution_ev_raw_market_state_v1"

# Keep one compact, interpretable representative set per requested family.  The
# source store computes these causal rolling/cross-sectional measures at each
# completed hourly bar; this adapter only reads them at or before decision time.
MARKET_STATE_FAMILIES: dict[str, tuple[str, ...]] = {
    "volatility_atr_term_structure": (
        "volatility_of_volatility_48",
        "atr_compression_ratio",
        "atr_slope",
        "atr_pct_change",
        "mkt_atr_expansion_4h",
    ),
    "trend_range_efficiency": (
        "efficiency_ratio_20",
        "path_efficiency_24",
        "trend_r2_24",
        "range_expansion_ratio",
        "breakout_efficiency_4h",
    ),
    "breadth": (
        "market_breadth_4h",
        "negative_breadth_pct",
        "breadth_dispersion",
        "market_breadth_recovery_from_24h_min",
    ),
    "correlation": (
        "avg_pair_corr_24h",
        "corr_concentration_24h",
        "market_downside_pairwise_corr_24h",
    ),
    "funding": (
        "mkt_funding_mean_z_30d",
        "mkt_funding_dispersion_z_30d",
        "mkt_funding_chg_4h",
        "mkt_funding_tail_concentration",
    ),
    "open_interest": (
        "mkt_median_oi_chg_4h_rz",
        "mkt_oi_flush_z_30d",
        "pct_assets_oi_down_4h",
        "mkt_oi_flush_breadth_accel_1h",
    ),
    # These are price/OI/funding-derived *pressure proxies*, not a raw
    # liquidation tape.  Keep that distinction in the feature name and report.
    "liquidation_pressure_proxy": (
        "liquidation_climax_score",
        "liquidation_onset_score",
        "post_liquidation_rebound_score",
        "asset_liquidation_phase_score",
    ),
}

# Actual historical L2 is only present from late July and has too little support
# for the May--July forward ablation.  Do not silently substitute the existing
# OHLCV-generated order-book proxy for this family.
UNAVAILABLE_HISTORICAL_FAMILIES: dict[str, str] = {
    "spread_depth": (
        "No adequate real-L2 historical coverage: the orderbook sidecar is an "
        "OHLCV proxy before late July. Excluded from the full-period ablation."
    ),
    "observed_liquidations": (
        "No raw liquidation time series is persisted; only the explicitly "
        "labelled price/OI/funding pressure proxy is available."
    ),
}

MARKET_STATE_SOURCE_COLUMNS: tuple[str, ...] = tuple(
    column for family in MARKET_STATE_FAMILIES.values() for column in family
)
MARKET_STATE_COLUMNS: tuple[str, ...] = tuple(
    f"mkt_state__{column}" for column in MARKET_STATE_SOURCE_COLUMNS
)


@dataclass(frozen=True)
class MarketStateJoinResult:
    frame: pd.DataFrame
    coverage: pd.DataFrame
    source_audit: pd.DataFrame


def feature_store_filename(symbol: object) -> str:
    """Map canonical ``BASE/USD:USD`` to the compact-store file name."""

    text = str(symbol)
    return f"symbol={text.replace('/', '_')}.parquet"


def _utc(values: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _existing_source_columns(path: Path, requested: Iterable[str]) -> list[str]:
    names = set(pq.read_schema(path).names)
    return [column for column in requested if column in names]


def attach_decision_time_market_state(
    candidates: pd.DataFrame,
    *,
    feature_store_root: Path,
    decision_time_col: str = "execution_decision_utc",
    symbol_col: str = "__symbol__",
    max_staleness: pd.Timedelta = pd.Timedelta("90min"),
    completed_bar_delay: pd.Timedelta = pd.Timedelta("1h"),
) -> MarketStateJoinResult:
    """Attach source-store values known at each decision time.

    Missing source files and feature columns remain explicit missing values;
    callers must not interpret them as zero.  `mkt_state_source_utc` is the
    actual row used, permitting a direct no-lookahead audit.
    """

    required = {decision_time_col, symbol_col}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidate frame lacks required columns: {missing}")
    if max_staleness <= pd.Timedelta(0):
        raise ValueError("max_staleness must be positive")
    if completed_bar_delay < pd.Timedelta(0):
        raise ValueError("completed_bar_delay cannot be negative")

    work = candidates.copy()
    work[decision_time_col] = _utc(work[decision_time_col])
    if work[decision_time_col].isna().any():
        raise ValueError("decision timestamps must be valid UTC values")
    work["__market_state_row_id__"] = np.arange(len(work), dtype=np.int64)
    output = work[["__market_state_row_id__"]].copy()
    output["mkt_state_source_utc"] = pd.Series(
        pd.NaT, index=output.index, dtype="datetime64[ns, UTC]"
    )
    output["mkt_state_source_file_found"] = False
    for column in MARKET_STATE_COLUMNS:
        output[column] = np.nan

    audits: list[dict[str, object]] = []
    for symbol, left in work.groupby(symbol_col, sort=False, dropna=False):
        path = Path(feature_store_root) / feature_store_filename(symbol)
        audit: dict[str, object] = {
            "symbol": str(symbol),
            "source_path": str(path),
            "candidate_rows": int(len(left)),
            "source_file_found": bool(path.exists()),
        }
        if not path.exists():
            audits.append(audit)
            continue
        available = _existing_source_columns(path, MARKET_STATE_SOURCE_COLUMNS)
        audit["available_source_columns"] = ",".join(available)
        audit["available_source_column_count"] = int(len(available))
        if not available:
            audits.append(audit)
            continue
        # ``ts`` is physically an index field in some compact-store files and
        # a normal field in others.  Asking Arrow for it by logical name fails
        # on the former; pandas restores the persisted index automatically.
        source = pd.read_parquet(path, columns=available)
        # Feature-store parquet persists ``ts`` as its named index.  Pandas
        # restores it as an index while Arrow exposes it in the schema.
        if "ts" not in source.columns:
            source = source.reset_index()
        if "ts" not in source.columns and "index" in source.columns:
            source = source.rename(columns={"index": "ts"})
        if "ts" not in source.columns:
            raise ValueError(f"feature store source has no usable ts column: {path}")
        source["ts"] = _utc(source["ts"])
        source = source.dropna(subset=["ts"]).drop_duplicates("ts", keep="last")
        source = source.sort_values("ts")
        audit["source_rows"] = int(len(source))
        # The raw panel timestamp is the open of a completed hour.  Its OHLCV
        # and derived rolling values therefore become observable at the next
        # hour boundary, not at the timestamp printed on the row.
        left_join = left[["__market_state_row_id__", decision_time_col]].copy()
        left_join["__market_state_cutoff_utc__"] = (
            left_join[decision_time_col] - completed_bar_delay
        )
        left_join = left_join.sort_values("__market_state_cutoff_utc__")
        joined = pd.merge_asof(
            left_join,
            source,
            left_on="__market_state_cutoff_utc__",
            right_on="ts",
            direction="backward",
            tolerance=max_staleness,
            allow_exact_matches=True,
        )
        if (joined["ts"].notna() & joined["ts"].gt(joined["__market_state_cutoff_utc__"])).any():
            raise RuntimeError("market-state join selected a future source timestamp")
        ix = joined["__market_state_row_id__"].to_numpy(dtype=np.int64)
        output.loc[ix, "mkt_state_source_utc"] = pd.array(
            joined["ts"], dtype="datetime64[ns, UTC]"
        )
        output.loc[ix, "mkt_state_source_file_found"] = True
        for source_column in available:
            target = f"mkt_state__{source_column}"
            output.loc[ix, target] = pd.to_numeric(
                joined[source_column], errors="coerce"
            ).to_numpy(dtype=float)
        audit["joined_rows"] = int(joined["ts"].notna().sum())
        audit["max_source_age_seconds"] = float(
            (joined["__market_state_cutoff_utc__"] - joined["ts"]).dt.total_seconds().max()
        ) if joined["ts"].notna().any() else np.nan
        audits.append(audit)

    result = work.merge(output, on="__market_state_row_id__", how="left", validate="one_to_one")
    source_ts = _utc(result["mkt_state_source_utc"])
    decision_ts = _utc(result[decision_time_col])
    valid_source = source_ts.notna()
    cutoff_ts = decision_ts - completed_bar_delay
    if (source_ts[valid_source] > cutoff_ts[valid_source]).any():
        raise RuntimeError("causal market-state invariant failed after join")
    source_age = (cutoff_ts - source_ts).dt.total_seconds()
    if (source_age[valid_source] > max_staleness.total_seconds()).any():
        raise RuntimeError("market-state staleness invariant failed after join")
    result["mkt_state_source_age_seconds"] = source_age.astype("float32")
    result["mkt_state_completed_bar_delay_seconds"] = np.float32(
        completed_bar_delay.total_seconds()
    )
    result["mkt_state_schema"] = MARKET_STATE_SCHEMA_VERSION
    result = result.drop(columns="__market_state_row_id__")

    coverage_rows: list[dict[str, object]] = []
    for family, columns in MARKET_STATE_FAMILIES.items():
        for source_column in columns:
            target = f"mkt_state__{source_column}"
            coverage_rows.append(
                {
                    "family": family,
                    "source_column": source_column,
                    "market_state_column": target,
                    "finite_fraction": float(pd.to_numeric(result[target], errors="coerce").notna().mean()),
                    "source_row_fraction": float(valid_source.mean()),
                    "first_source_utc": str(source_ts.min()),
                    "last_source_utc": str(source_ts.max()),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    return MarketStateJoinResult(
        frame=result,
        coverage=coverage,
        source_audit=pd.DataFrame(audits),
    )


__all__ = [
    "MARKET_STATE_COLUMNS",
    "MARKET_STATE_FAMILIES",
    "MARKET_STATE_SCHEMA_VERSION",
    "MARKET_STATE_SOURCE_COLUMNS",
    "UNAVAILABLE_HISTORICAL_FAMILIES",
    "MarketStateJoinResult",
    "attach_decision_time_market_state",
    "feature_store_filename",
]
