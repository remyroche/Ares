#!/usr/bin/env python3
"""Materialise target-only H12 path/support labels for the strict-R3 long panel.

The sidecar deliberately separates the decision-time candidate substrate from
future-path supervision.  It reopens the canonical 15-minute OHLCV path from
the frozen decision open for every valid row, derives an ATR from bars ending
*before* that open, and emits only labels.  No column written by this producer
is permitted in an inference feature contract.

It is the long counterpart to ``materialize_strict_r3_short_supportive_path_labels``
but uses the complete historical 15-minute source because one-minute coverage
is not available across the full requested long history.  The cadence and
proxy are explicit in the manifest so later exact-one-minute studies cannot
silently mix contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_archetype_labels import (  # noqa: E402
    PATH_SUMMARY_PREFIX,
    PathArchetypeLabelConfig,
    materialize_path_archetypes,
)


SCHEMA = "strict_r3_long_supportive_path_labels_v2_h12_15m_observed_entry_causal_atr"
SIDE = "long"
HORIZON_HOURS = 12.0
BAR_HOURS = 0.25
ATR_HOURLY_PERIODS = 14
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
DEFAULT_SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_source_panel_targetfree_long_"
    "2023_aug7_2026_raw15m_strictfull_20260812_v1/canonical_source_panel.parquet"
)
DEFAULT_BARS = ROOT / "15m_ohlcv_perp"

# Fractional horizons provide the requested early 15/30 minute view while the
# named 1/2/4/8/12 hour quantities retain stable, readable field names.
PATH_HORIZONS = (0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 12.0)
PATH_CONFIG = PathArchetypeLabelConfig(
    timestamp_col="__ts__",
    symbol_col="__symbol__",
    side_col="side_name",
    bar_timestamp_col="timestamp",
    bar_symbol_col="symbol",
    decision_delay_hours=1,
    bar_hours=BAR_HOURS,
    horizons_hours=PATH_HORIZONS,  # type: ignore[arg-type]
    prefix=PATH_SUMMARY_PREFIX,
)

SOURCE_COLUMNS = (
    *IDENTITY,
    "policy_path_valid",
    "policy_label_available_ts",
    "policy_entry_price",
    "policy_gross_bps",
    "policy_net_bps",
    "policy_exit_reason",
    "policy_cost_bps",
    "h12_label_valid",
    "h12_label_available_ts",
    "h12_tp6_sl4_gross_bps",
    "h12_tp6_sl4_net_bps",
)

DIRECT_LABEL_COLUMNS = (
    "supportive_peak_mfe_atr_h12",
    "supportive_mae_before_meaningful_atr_h12",
    "supportive_time_to_meaningful_mfe_h12",
    "supportive_final_return_atr_h12",
    "supportive_path_efficiency_h12",
    "supportive_reversal_count_h12",
    "supportive_peak_retention_h12",
    "supportive_early_mfe_15m_atr",
    "supportive_early_mfe_30m_atr",
    "supportive_early_mfe_1h_atr",
    "supportive_early_mfe_2h_atr",
    "supportive_early_mae_15m_atr",
    "supportive_early_mae_30m_atr",
    "supportive_early_mae_1h_atr",
    "supportive_early_mae_2h_atr",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _canonical_file_symbol(symbol: str) -> str:
    """Map ``AAVE/USD:USD`` to the historical ``aaveusd:usd`` stem."""
    return symbol.strip().lower().replace("/", "")


def _bar_path(root: Path, symbol: str) -> Path:
    path = root / f"{_canonical_file_symbol(symbol)}_15m.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_bars(root: Path, symbol: str, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read only the bounded bar interval plus ATR warm-up for one symbol."""
    # The historical files have ``ts`` as a parquet index; filters are pushed
    # into pyarrow row groups on the current source contract.
    warm_start = start - pd.Timedelta(hours=ATR_HOURLY_PERIODS + 2)
    path = _bar_path(root, symbol)
    physical_columns = set(pq.ParquetFile(path).schema.names)
    timestamp_field = "ts" if "ts" in physical_columns else "__index_level_0__"
    raw = pd.read_parquet(
        path,
        filters=[(timestamp_field, ">=", warm_start), (timestamp_field, "<=", end)],
    ).copy()
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "atr_fraction"])
    if not isinstance(raw.index, pd.DatetimeIndex):
        if "ts" not in raw:
            raise ValueError(f"15m OHLCV has no timestamp index/column for {symbol}")
        raw = raw.set_index("ts")
    raw.index = pd.to_datetime(raw.index, utc=True, errors="raise")
    raw = raw.loc[:, [column for column in ("open", "high", "low", "close") if column in raw]].copy()
    if set(raw.columns) != {"open", "high", "low", "close"}:
        raise ValueError(f"15m OHLCV lacks OHLC for {symbol}")
    raw = raw.apply(pd.to_numeric, errors="coerce").sort_index(kind="stable")
    if raw.index.duplicated().any():
        raise ValueError(f"duplicate 15m timestamps for {symbol}")
    # Bars are timestamped at their opening instant.  The hourly candle at H
    # closes exactly at H+1 and is therefore available at a decision at H+1.
    hourly = raw.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).dropna()
    prior_close = hourly["close"].shift(1)
    true_range = pd.concat(
        [hourly["high"] - hourly["low"], (hourly["high"] - prior_close).abs(), (hourly["low"] - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    # Wilder ATR is causal; its value for the hour beginning H is only used at
    # decisions at or after H+1 through the explicit one-hour shift below.
    hourly["atr"] = true_range.ewm(alpha=1.0 / ATR_HOURLY_PERIODS, adjust=False, min_periods=ATR_HOURLY_PERIODS).mean()
    atr_by_decision = hourly["atr"].copy()
    atr_by_decision.index = atr_by_decision.index + pd.Timedelta(hours=1)
    raw["atr"] = atr_by_decision.reindex(raw.index, method="ffill")
    raw["atr_fraction"] = raw["atr"] / raw["open"]
    output = raw.reset_index(names="timestamp")
    output["symbol"] = symbol
    return output


def _load_month(source: Path, month: pd.Timestamp) -> pd.DataFrame:
    next_month = month + pd.offsets.MonthBegin(1)
    frame = pd.read_parquet(
        source,
        columns=list(SOURCE_COLUMNS),
        filters=[("__ts__", ">=", month), ("__ts__", "<", next_month)],
    ).copy()
    for column in ("__ts__", "__decision_ts__", "policy_label_available_ts", "h12_label_available_ts"):
        frame[column] = _utc(frame[column])
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"duplicate long candidate IDs in {month:%Y-%m}")
    if not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise AssertionError("supportive path source is not long-only")
    if not frame["__decision_ts__"].eq(frame["__ts__"] + pd.Timedelta(hours=1)).all():
        raise AssertionError("long target entry must be signal close + one hour")
    return frame.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _direct_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Emit compact direct controls from P1 without exposing them as features."""
    atr = pd.to_numeric(frame["atr_fraction"], errors="coerce").to_numpy(float)
    eps = np.maximum(atr, 1e-8)
    result = pd.DataFrame(index=frame.index)
    result["supportive_peak_mfe_atr_h12"] = pd.to_numeric(frame["path_arch_peak_mfe_atr"], errors="coerce")
    result["supportive_mae_before_meaningful_atr_h12"] = pd.to_numeric(frame["path_arch_mae_before_meaningful_mfe_r"], errors="coerce")
    result["supportive_time_to_meaningful_mfe_h12"] = pd.to_numeric(frame["path_arch_time_to_first_meaningful_mfe_h"], errors="coerce")
    result["supportive_final_return_atr_h12"] = pd.to_numeric(frame["path_arch_final_return_r"], errors="coerce")
    result["supportive_path_efficiency_h12"] = pd.to_numeric(frame["path_arch_efficiency"], errors="coerce")
    result["supportive_reversal_count_h12"] = pd.to_numeric(frame["path_arch_reversal_count"], errors="coerce")
    result["supportive_peak_retention_h12"] = pd.to_numeric(frame["path_arch_peak_retention_ratio"], errors="coerce")
    for token, suffix in (("0.25", "15m"), ("0.5", "30m"), ("1.0", "1h"), ("2.0", "2h")):
        mfe = pd.to_numeric(frame[f"path_arch_mfe_{token}h_r"], errors="coerce").to_numpy(float)
        mae = pd.to_numeric(frame[f"path_arch_mae_{token}h_r"], errors="coerce").to_numpy(float)
        # risk_distance is exactly one decision-time ATR, so P1 R units equal
        # ATR units by construction.
        result[f"supportive_early_mfe_{suffix}_atr"] = mfe
        result[f"supportive_early_mae_{suffix}_atr"] = np.maximum(-mae, 0.0)
    return result.loc[:, list(DIRECT_LABEL_COLUMNS)].astype(np.float32)


def _materialize_month(source: pd.DataFrame, bars_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_parts: list[pd.DataFrame] = []
    # Path targets are independent supervision.  They must be materialised
    # from the target-free candidate substrate, not selected by whether an
    # older policy-label materialisation happened to have an executable entry
    # or a complete policy outcome.  The actual observed 15-minute decision
    # open below is the only entry used for target construction.  Policy
    # validity remains output-only provenance for downstream outcome joins.
    source_policy_valid = (
        source["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(source["policy_entry_price"], errors="coerce").gt(0.0)
        & source["policy_label_available_ts"].notna()
    )
    entry_parity_rows = 0
    entry_mismatch_rows = 0
    observed_decision_open_rows = 0
    missing_decision_open_rows = 0
    path_complete_rows = 0
    for symbol, original_group in source.groupby("__symbol__", sort=True, observed=True):
        group = original_group.copy().reset_index(drop=True)
        decision_start = group["__decision_ts__"].min()
        decision_end = group["__decision_ts__"].max() + pd.Timedelta(hours=HORIZON_HOURS)
        bars = _load_bars(bars_root, str(symbol), start=decision_start, end=decision_end)
        if bars.empty:
            group["entry_price"] = np.nan
            group["atr_fraction"] = np.nan
            group["risk_distance"] = np.nan
            group_label_eligible = np.zeros(len(group), dtype=bool)
            missing_decision_open_rows += int(len(group))
        else:
            decision_open = bars.set_index("timestamp")["open"].reindex(group["__decision_ts__"]).to_numpy(float)
            frozen_entry = pd.to_numeric(group["policy_entry_price"], errors="coerce").to_numpy(float)
            observed_open = np.isfinite(decision_open) & (decision_open > 0.0)
            # This is provenance only.  A mismatch must never erase a complete
            # path target: the observed decision open is canonical here.
            comparable_policy_entry = observed_open & np.isfinite(frozen_entry) & (frozen_entry > 0.0)
            exact_parity = comparable_policy_entry & np.isclose(decision_open, frozen_entry, rtol=0.0, atol=2e-4)
            group_label_eligible = observed_open
            observed_decision_open_rows += int(observed_open.sum())
            entry_parity_rows += int(exact_parity.sum())
            entry_mismatch_rows += int((comparable_policy_entry & ~exact_parity).sum())
            missing_decision_open_rows += int((~observed_open).sum())
            atr_fraction = bars.set_index("timestamp")["atr_fraction"].reindex(group["__decision_ts__"]).to_numpy(float)
            group["entry_price"] = decision_open
            group["atr_fraction"] = atr_fraction
            group["risk_distance"] = decision_open * atr_fraction
        # The kernel also handles the explicit no-bars case: it creates the
        # entire future-label schema as null rather than changing columns or
        # accidentally dropping a historical candidate identity.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            materialized = materialize_path_archetypes(group, bars, config=PATH_CONFIG)
        complete = materialized["path_arch_complete_24h"].fillna(0).astype(bool)
        # A true absent decision bar, ATR warm-up gap, or incomplete H12 path
        # remains invalid supervision.  It is never encoded as a zero/failure.
        valid = group_label_eligible & complete.to_numpy(bool)
        materialized["supportive_path_valid"] = valid.astype(np.int8)
        materialized["supportive_target_invalid"] = (~valid).astype(np.int8)
        materialized["supportive_label_available_ts"] = materialized["__decision_ts__"] + pd.Timedelta(hours=HORIZON_HOURS)
        # An incomplete market path remains invalid supervision rather than an
        # ordinary zero/negative example.  Policy-label invalidity does not
        # alter this independent target contract.
        numeric_path = [column for column in materialized if column.startswith(PATH_SUMMARY_PREFIX)]
        materialized.loc[~valid, numeric_path] = np.nan
        for column in ("path_shape_archetype", "path_realization_strength", "path_archetype"):
            materialized.loc[~valid, column] = pd.NA
        materialized = materialized.rename(columns={"path_arch_complete_24h": "path_arch_complete_h12"})
        direct = _direct_labels(materialized)
        direct.loc[~valid, :] = np.nan
        materialized = pd.concat([materialized, direct], axis=1, copy=False)
        path_complete_rows += int(valid.sum())
        output_parts.append(materialized)
    if not output_parts:
        raise ValueError("source month has no symbols")
    full = pd.concat(output_parts, ignore_index=True)
    # Persist only identities, future targets, and target validity/outcome
    # provenance.  The 120 decision-time base fields remain in the source panel.
    path_columns = [column for column in full.columns if column.startswith(PATH_SUMMARY_PREFIX)]
    keep = [
        *IDENTITY,
        "supportive_label_available_ts", "supportive_path_valid", "supportive_target_invalid",
        "entry_price",
        "path_shape_archetype", "path_realization_strength", "path_archetype", "path_arch_complete_h12",
        *path_columns,
        *DIRECT_LABEL_COLUMNS,
        "policy_path_valid", "policy_label_available_ts", "policy_gross_bps", "policy_net_bps", "policy_exit_reason", "policy_cost_bps",
        "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_gross_bps", "h12_tp6_sl4_net_bps",
    ]
    full = full.loc[:, list(dict.fromkeys(keep))].copy()
    invalid = ~full["supportive_path_valid"].astype(bool)
    forbidden = [*path_columns, *DIRECT_LABEL_COLUMNS]
    if full.loc[invalid, forbidden].notna().any().any():
        raise AssertionError("invalid long path labels were encoded as ordinary targets")
    if full.loc[~invalid, "supportive_label_available_ts"].lt(full.loc[~invalid, "__decision_ts__"] + pd.Timedelta(hours=HORIZON_HOURS)).any():
        raise AssertionError("supportive label availability precedes the H12 path")
    record = {
        "rows": int(len(full)),
        "source_policy_valid_rows": int(source_policy_valid.sum()),
        "source_policy_invalid_rows": int((~source_policy_valid).sum()),
        "candidate_rows": int(len(source)),
        "path_valid_rows": int(full["supportive_path_valid"].sum()),
        "entry_open_parity_rows": int(entry_parity_rows),
        "entry_open_mismatch_rows": int(entry_mismatch_rows),
        "observed_decision_open_rows": int(observed_decision_open_rows),
        "missing_decision_open_rows": int(missing_decision_open_rows),
        "path_completion_rows": int(path_complete_rows),
    }
    for column in DIRECT_LABEL_COLUMNS:
        values = pd.to_numeric(full.loc[~invalid, column], errors="coerce")
        record[f"{column}_finite_rows"] = int(values.notna().sum())
    return full, record


def run(*, source: Path, bars_root: Path, out: Path, start: pd.Timestamp, end: pd.Timestamp, verbose: bool = True) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if start.tzinfo is None or end.tzinfo is None or start.day != 1 or end.day != 1 or start.hour != 0 or end.hour != 0 or start >= end:
        raise ValueError("start/end must be increasing UTC month boundaries")
    records: list[dict[str, Any]] = []
    for month in pd.date_range(start, end, freq="MS", inclusive="left"):
        source_month = _load_month(source, month)
        sidecar, record = _materialize_month(source_month, bars_root)
        record["month"] = f"{month:%Y-%m}"
        destination = out / "parts" / f"month={month:%Y-%m}" / "side=long.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sidecar.to_parquet(destination, index=False, compression="zstd")
        records.append(record)
        if verbose:
            print(json.dumps(record, sort_keys=True), flush=True)
    coverage = pd.DataFrame(records)
    coverage.to_parquet(out / "coverage_by_month.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "side": SIDE,
        "months": [record["month"] for record in records],
        "identity": list(IDENTITY),
        "source_panel": str(source.resolve()),
        "source_panel_sha256": _sha256(source),
        "bars_root": str(bars_root.resolve()),
        "bar_contract": "complete post-decision 48x15-minute OHLC; entry at frozen decision open",
        "atr_contract": "Wilder-14 hourly ATR from bars completed before the decision open",
        "label_available_at": "decision + 12 hours",
        "target_family": "P1 causal-path summaries plus compact direct supportive controls",
        "inference": "all emitted path/direct values are supervised labels only and prohibited from inference features",
        "entry_contract": "observed 15-minute open at frozen decision timestamp; policy-entry parity is provenance only",
        "invalidity": "missing decision opens, ATR warm-up gaps, or incomplete H12 paths remain target-invalid with null future-path values; policy-label availability never selects path targets",
        "coverage": records,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bars-root", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2024-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--quiet", action="store_true", help="persist monthly receipts without emitting one large progress line per month")
    args = parser.parse_args()
    print(run(
        source=args.source.resolve(), bars_root=args.bars_root.resolve(), out=args.out.resolve(),
        start=pd.to_datetime(args.start, utc=True), end=pd.to_datetime(args.end, utc=True), verbose=not args.quiet,
    ))


if __name__ == "__main__":
    main()
