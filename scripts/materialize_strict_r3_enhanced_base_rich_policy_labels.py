#!/usr/bin/env python3
"""Materialise frozen rich-policy labels for every enhanced-base OOS identity.

This is an offline outcome producer.  Candidate identities originate from the
already target-free direct base predictions.  The producer never filters
those identities by future-path availability: a missing/incomplete 15-minute
path is persisted as ``policy_path_valid = false`` and can only be excluded
later from supervised fitting or a label-complete research replay.

The state machine and parameter JSON are the frozen rich 15-minute aggregate
contract used by the current research stack.  It consumes a complete local
48-bar H12 path; it is not imported by live execution code and makes no
exchange calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_rich_policy import (  # noqa: E402
    RichPolicyParams,
    simulate_rich_policy,
)


SCHEMA = "strict_r3_enhanced_base_full_rich_policy_labels_v1"
ARM = "S3_direct_efficiency_time_base_equal"
HORIZON_MINUTES = 12 * 60
BATCH_ROWS = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
    else:
        files = [path]
    for child in files:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _direct_identity(direct_root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    columns = ["arm", "candidate_id", "__decision_ts__"]
    for path in sorted((direct_root / "oof_prediction_parts").glob("fold=*/*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["arm"].eq(ARM), ["candidate_id", "__decision_ts__"]]
        if not frame.empty:
            pieces.append(frame)
    if not pieces:
        raise FileNotFoundError(f"no {ARM} OOF parts under {direct_root}")
    output = pd.concat(pieces, ignore_index=True)
    output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True, errors="raise")
    if output["candidate_id"].duplicated().any():
        raise ValueError("enhanced-base OOF source has duplicate candidate IDs")
    return output


def _path_identity(path_root: Path, months: list[str]) -> pd.DataFrame:
    """Return target-free candidate identities from the path-panel ledger.

    The path-panel files also carry realised-path columns, but this function
    deliberately reads only the stable decision-time identity columns.  A
    candidate is retained regardless of its future-path completeness; that is
    represented later by an explicit invalid policy label rather than by an
    eligibility filter.
    """
    requested = [pd.Timestamp(f"{token}-01", tz="UTC") for token in months]
    if not requested:
        raise ValueError("path identity source requires at least one requested month")
    source_months = sorted({
        f"{token:%Y-%m}"
        for month in requested
        for token in (month - pd.offsets.MonthBegin(1), month)
    })
    columns = ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]
    pieces: list[pd.DataFrame] = []
    for token in source_months:
        path = path_root / f"month={token}" / "side=long.parquet"
        if not path.exists():
            raise FileNotFoundError(f"path identity source is missing {path}")
        pieces.append(pd.read_parquet(path, columns=columns))
    output = pd.concat(pieces, ignore_index=True)
    output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True, errors="raise")
    if output["candidate_id"].duplicated().any():
        raise ValueError("path identity source has duplicate candidate IDs across adjacent partitions")
    start, end = min(requested), max(requested) + pd.offsets.MonthBegin(1)
    output = output.loc[
        output["__decision_ts__"].ge(start) & output["__decision_ts__"].lt(end)
    ].loc[:, ["candidate_id", "__decision_ts__"]].copy()
    if output.empty:
        raise ValueError("path identity source supplied no identities in requested decision months")
    return output


def _months(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["__decision_ts__"].dt.strftime("%Y-%m").unique().tolist())


def _source_month(
    identities: pd.DataFrame,
    raw_ledger: Path,
    path_root: Path,
    month: str,
) -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    requested = identities.loc[
        identities["__decision_ts__"].ge(start) & identities["__decision_ts__"].lt(end)
    ].copy()
    raw = pd.read_parquet(
        raw_ledger,
        columns=["candidate_id", "__decision_ts__", "__symbol__", "side_name"],
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    source = requested.merge(raw, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    if source[["__symbol__", "side_name"]].isna().any(axis=None):
        raise AssertionError(f"{month}: target-free raw ledger missed enhanced identities")
    if not source["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{month}: source is not long-only")
    # Candidate IDs encode the signal hour while ``__decision_ts__`` is the
    # next executable hour.  At a calendar boundary, a 00:00 decision thus
    # belongs to the preceding signal-month partition.  Load both adjacent
    # partitions by ID; this is lineage repair, never a future-path lookup.
    prior_month = (start - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    metadata_parts: list[pd.DataFrame] = []
    for source_month in (prior_month, month):
        metadata_path = path_root / f"month={source_month}" / "side=long.parquet"
        if metadata_path.exists():
            metadata_parts.append(pd.read_parquet(
                metadata_path,
                columns=["candidate_id", "entry_price", "path_arch_atr_fraction"],
            ))
    if metadata_parts:
        metadata = pd.concat(metadata_parts, ignore_index=True)
        if metadata["candidate_id"].duplicated().any():
            raise AssertionError(f"{month}: adjacent path metadata has duplicate candidate IDs")
        source = source.merge(metadata, on="candidate_id", how="left", validate="one_to_one")
    else:
        source["entry_price"] = np.nan
        source["path_arch_atr_fraction"] = np.nan
    source["entry_price"] = pd.to_numeric(source["entry_price"], errors="coerce")
    source["path_arch_atr_fraction"] = pd.to_numeric(source["path_arch_atr_fraction"], errors="coerce")
    source["atr_1h"] = source["entry_price"] * source["path_arch_atr_fraction"]
    return source.reset_index(drop=True)


def _path_source_month(
    identities: pd.DataFrame,
    path_root: Path,
    month: str,
) -> pd.DataFrame:
    """Read only decision-time fields from the path-panel source.

    This is an explicit recovery path for archived experiments whose original
    target-free raw ledger has been sparsified.  It is safe because identity,
    symbol, side, entry, and ATR provenance are selected before any future
    outcome field is read.  Missing entry/ATR values remain rows with invalid
    labels and never disappear from the population.
    """
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    requested = identities.loc[
        identities["__decision_ts__"].ge(start) & identities["__decision_ts__"].lt(end)
    ].copy()
    prior_month = (start - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "entry_price", "path_arch_atr_fraction",
    ]
    pieces: list[pd.DataFrame] = []
    for source_month in (prior_month, month):
        path = path_root / f"month={source_month}" / "side=long.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{month}: missing path identity partition {path}")
        pieces.append(pd.read_parquet(path, columns=columns))
    source = pd.concat(pieces, ignore_index=True)
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
    if source["candidate_id"].duplicated().any():
        raise AssertionError(f"{month}: adjacent path partitions duplicate candidate IDs")
    source = requested.merge(
        source,
        on=["candidate_id", "__decision_ts__"],
        how="left",
        validate="one_to_one",
    )
    if source[["__symbol__", "side_name"]].isna().any(axis=None):
        raise AssertionError(f"{month}: path identity source missed requested candidates")
    if not source["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{month}: path identity source is not long-only")
    source["entry_price"] = pd.to_numeric(source["entry_price"], errors="coerce")
    source["path_arch_atr_fraction"] = pd.to_numeric(source["path_arch_atr_fraction"], errors="coerce")
    source["atr_1h"] = source["entry_price"] * source["path_arch_atr_fraction"]
    return source.reset_index(drop=True)


def _invalid(frame: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = frame.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]].copy()
    out["policy_path_valid"] = False
    out["policy_gross_bps"] = np.nan
    out["policy_net_bps"] = np.nan
    out["policy_exit_bar_15m"] = -1
    out["policy_entry_price"] = pd.to_numeric(frame.get("entry_price"), errors="coerce")
    out["policy_exit_price"] = np.nan
    out["policy_exit_reason"] = reason
    out["policy_label_available_ts"] = out["__decision_ts__"] + pd.Timedelta(hours=12)
    out["policy_cost_bps"] = np.nan
    out["policy_outcome_source"] = "unavailable"
    out["label_source_complete_1m_path"] = False
    return out


def _coarse_15m_windows(
    bars: pd.DataFrame,
    decisions: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return complete 48-bar frozen rich-policy paths from local 15m OHLC."""
    decisions = pd.DatetimeIndex(pd.to_datetime(decisions, utc=True, errors="raise"))
    if bars.empty:
        empty = np.full((len(decisions), 48), np.nan, dtype=np.float32)
        return np.zeros(len(decisions), dtype=bool), empty, empty.copy(), empty.copy()
    bars = bars.copy()
    bars.index = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True, errors="coerce"))
    bars = bars.loc[~bars.index.isna() & ~bars.index.duplicated(keep="last")].sort_index()
    start = min(pd.Timestamp(decisions.min()), pd.Timestamp(bars.index.min())).floor("15min")
    end = max(pd.Timestamp(decisions.max()) + pd.Timedelta(minutes=HORIZON_MINUTES), pd.Timestamp(bars.index.max())).ceil("15min")
    grid = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    values = bars.reindex(grid).loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    offsets = ((decisions - start) / pd.Timedelta(minutes=15)).astype(np.int64)
    horizon_bars = 48
    valid = (offsets >= 0) & (offsets + horizon_bars <= len(grid))
    arrays = [np.full((len(decisions), horizon_bars), np.nan, dtype=np.float32) for _ in range(4)]
    rows = np.flatnonzero(valid)
    for row in rows:
        path = values[offsets[row]: offsets[row] + horizon_bars]
        for col in range(4):
            arrays[col][row] = path[:, col]
    good = (
        np.isfinite(arrays[0]).all(axis=1)
        & np.isfinite(arrays[1]).all(axis=1)
        & np.isfinite(arrays[2]).all(axis=1)
        & np.isfinite(arrays[3]).all(axis=1)
        & (arrays[0] > 0).all(axis=1)
        & (arrays[1] >= arrays[2]).all(axis=1)
    )
    return (
        good,
        arrays[1], arrays[2], arrays[3],
    )


def _load_15m_bars(bars_root: Path, symbol: str) -> pd.DataFrame:
    path = bars_root / f"{str(symbol).lower().replace('/', '')}_15m.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    frame = pd.read_parquet(path, columns=["open", "high", "low", "close"])
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"15m source index is not datetime: {path}")
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index = frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index.tz_convert("UTC")
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index()


def _materialize_month(
    source: pd.DataFrame,
    *,
    bars_root: Path,
    params: RichPolicyParams,
    median_atr_fraction: float,
) -> pd.DataFrame:
    output = _invalid(source, "missing_path_metadata")
    entry = source["entry_price"].to_numpy(float)
    atr = source["atr_1h"].to_numpy(float)
    usable = np.isfinite(entry) & (entry > 0.0) & np.isfinite(atr) & (atr > 0.0)
    output.loc[usable, "policy_exit_reason"] = "incomplete_frozen_15m_path"
    for symbol, locations in source.loc[usable].groupby("__symbol__", sort=True).groups.items():
        positions = np.asarray(list(locations), dtype=np.int64)
        rows = source.loc[positions].copy().reset_index().rename(columns={"index": "__output_index__"})
        start = pd.Timestamp(rows["__decision_ts__"].min())
        end = pd.Timestamp(rows["__decision_ts__"].max()) + pd.Timedelta(minutes=HORIZON_MINUTES)
        bars = _load_15m_bars(bars_root, str(symbol)).loc[start:end]
        complete, highs, lows, closes = _coarse_15m_windows(bars, rows["__decision_ts__"])
        for begin in range(0, len(rows), BATCH_ROWS):
            stop = min(begin + BATCH_ROWS, len(rows))
            # ``idx`` indexes the local 15-minute-path matrix, so each
            # chunk must carry a zero-based local index even after the first
            # chunk of a symbol.
            batch = rows.iloc[begin:stop].copy().reset_index(drop=True)
            local = complete[begin:stop]
            if not local.any():
                continue
            idx = np.flatnonzero(local)
            replay = simulate_rich_policy(
                entry=batch.loc[idx, "entry_price"].to_numpy(float),
                atr=batch.loc[idx, "atr_1h"].to_numpy(float),
                highs=highs[begin:stop][idx],
                lows=lows[begin:stop][idx],
                closes=closes[begin:stop][idx],
                params=params,
                median_atr_fraction=median_atr_fraction,
                side="long",
            )
            if not replay["path_valid"].all():
                raise AssertionError("complete exact-1m extraction produced an invalid rich-policy path")
            output_index = batch.loc[idx, "__output_index__"].to_numpy(np.int64)
            output.loc[output_index, "policy_path_valid"] = True
            output.loc[output_index, "policy_gross_bps"] = replay["gross_bps"]
            output.loc[output_index, "policy_net_bps"] = replay["net_bps"]
            output.loc[output_index, "policy_exit_bar_15m"] = replay["exit_bar"]
            output.loc[output_index, "policy_entry_price"] = batch.loc[idx, "entry_price"].to_numpy(float)
            output.loc[output_index, "policy_exit_price"] = (
                batch.loc[idx, "entry_price"].to_numpy(float)
                * (1.0 + np.asarray(replay["gross_bps"], dtype=float) / 10_000.0)
            )
            output.loc[output_index, "policy_exit_reason"] = replay["exit_reason"]
            output.loc[output_index, "policy_cost_bps"] = 100.0
            output.loc[output_index, "policy_outcome_source"] = "frozen_rich_15m_aggregate_local_15m"
            output.loc[output_index, "label_source_complete_1m_path"] = True
    valid = output["policy_path_valid"].to_numpy(bool)
    if valid.any() and not np.allclose(
        output.loc[valid, "policy_gross_bps"].to_numpy(float) - 100.0,
        output.loc[valid, "policy_net_bps"].to_numpy(float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise AssertionError("rich-policy label cost is not exactly once")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path)
    parser.add_argument("--raw-ledger", type=Path)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument(
        "--identity-source",
        choices=("direct_oof", "path_identity"),
        default="direct_oof",
        help="direct_oof is the original producer; path_identity recovers only decision-time identity/entry/ATR fields from the archived path ledger",
    )
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--bars-root", type=Path, default=Path("15m_ohlcv_perp"))
    parser.add_argument(
        "--months",
        help="optional comma-separated YYYY-MM subset; useful for an immutable smoke run",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="optional deterministic per-month cap for a smoke receipt; never use for full research labels",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        raise FileExistsError(f"immutable output already exists: {args.out}")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    if args.identity_source == "direct_oof":
        if args.direct_root is None or args.raw_ledger is None:
            raise ValueError("direct_oof identity source requires --direct-root and --raw-ledger")
    elif args.direct_root is not None or args.raw_ledger is not None:
        raise ValueError("path_identity source does not accept --direct-root or --raw-ledger")
    raw_policy = json.loads(args.policy_json.read_text())
    params = RichPolicyParams.from_mapping(raw_policy["params"])
    median = float(raw_policy["median_atr_fraction_fitted_on_complete_2024_development"])
    wanted_months = (
        [value.strip() for value in args.months.split(",") if value.strip()]
        if args.months else None
    )
    if args.identity_source == "direct_oof":
        identities = _direct_identity(args.direct_root)
        available_months = _months(identities)
        wanted_months = wanted_months or available_months
        unknown = sorted(set(wanted_months).difference(available_months))
        if unknown:
            raise ValueError(f"requested months not present in enhanced OOF source: {unknown}")
    else:
        if not wanted_months:
            raise ValueError("path_identity source requires an explicit --months list")
        identities = _path_identity(args.path_root, wanted_months)
    coverage: list[dict[str, object]] = []
    for month in wanted_months:
        target = args.out / "policy_parts" / f"month={month}.parquet"
        if target.exists():
            print(json.dumps({"event": "month_skip_existing", "month": month}), flush=True)
            continue
        source = (
            _source_month(identities, args.raw_ledger, args.path_root, month)
            if args.identity_source == "direct_oof"
            else _path_source_month(identities, args.path_root, month)
        )
        if args.max_rows is not None:
            if args.max_rows < 1:
                raise ValueError("--max-rows must be positive")
            source = source.sort_values(["__decision_ts__", "candidate_id"], kind="stable").head(args.max_rows).copy()
        labels = _materialize_month(source, bars_root=args.bars_root, params=params, median_atr_fraction=median)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        labels.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, target)
        valid = labels["policy_path_valid"].to_numpy(bool)
        coverage.append({
            "month": month, "rows": int(len(labels)), "valid_rows": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "missing_metadata": int(labels["policy_exit_reason"].eq("missing_path_metadata").sum()),
            "incomplete_frozen_15m": int(labels["policy_exit_reason"].eq("incomplete_frozen_15m_path").sum()),
        })
        print(json.dumps({"event": "month_complete", **coverage[-1]}), flush=True)
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_parquet(args.out / "coverage.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline full-population rich policy labels for enhanced-base research; no live mutation or exchange I/O",
        "identities": "all enhanced-base OOF candidate identities; missing paths remain explicit invalid labels",
        "source": {
            "identity_source": args.identity_source,
            "direct_root": str(args.direct_root) if args.direct_root else None,
            "raw_ledger": str(args.raw_ledger) if args.raw_ledger else None,
            "path_root": str(args.path_root), "policy_json": str(args.policy_json),
            "direct_sha256": _sha256(args.direct_root) if args.direct_root else None,
            "raw_sha256": _sha256(args.raw_ledger) if args.raw_ledger else None,
            "path_sha256": _sha256(args.path_root), "policy_sha256": _sha256(args.policy_json),
        },
        "policy": {"params": params.to_dict(), "median_atr_fraction": median, "horizon_minutes": HORIZON_MINUTES, "cost_bps_once": 100.0},
        "resolution": {
            "schema": "strict_r3_rich_policy_15m_local_proxy_v1",
            "source": "local_native_15m_ohlc",
            "horizon_bars": 48,
            "horizon_minutes": HORIZON_MINUTES,
            "same_bar_activation": "forbidden by simulate_rich_policy",
        },
        "months_requested": wanted_months,
        "coverage": coverage,
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out": str(args.out), "rows": int(len(identities))}), flush=True)


if __name__ == "__main__":
    main()
