#!/usr/bin/env python3
"""Materialise exact short R3 labels using the side-invariant causal ATR substrate.

The existing long 2024 exact-label artifact already records the causal
decision-time ATR(14) and exact next-hour entry for each point-in-time
candidate.  Those two quantities are side-invariant.  This utility reuses
only that causal substrate, then reopens the 2024 one-minute high/low path to
produce *all* directional short outcomes independently.  It must never reuse
the long event, MFE, net-return, or robust-clear outcome.

It exists to avoid the generic label materializer opening the full 2023
minute partition solely to recompute a 14-hour January warm-up that is
already present in the immutable long causal substrate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_packb_tp6_sl4_h12_labels import (
    COST_BPS,
    OUTPUT_COLUMNS,
    SCHEMA,
    _atomic_parquet,
    _label_candidates_with_minute,
    _minute_path_pruned,
    _packb_to_kraken_symbol,
)


DEFAULT_ARTIFACT = ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v1"
DEFAULT_CANDIDATES = DEFAULT_ARTIFACT / "short_eligible_candidates.parquet"
DEFAULT_LONG_LABELS = ROOT / "data_perp/artifacts/strict_r3_schema_v2_exact_tp6_r3_long_2024_20260809_v1"
DEFAULT_OUT = DEFAULT_ARTIFACT / "labels_q1_shared_atr"
DEFAULT_MINUTES = ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv"
START = pd.Timestamp("2024-01-01T00:00:00Z")
END = pd.Timestamp("2024-04-01T00:00:00Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_month(path: Path, month: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__"])
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    end = month + pd.offsets.MonthBegin(1)
    frame = frame.loc[frame["__ts__"].ge(month) & frame["__ts__"].lt(end)].copy()
    if frame.empty or frame.candidate_id.duplicated().any():
        raise ValueError(f"invalid short candidate month {month:%Y-%m}")
    if not frame.side_name.eq("short").all():
        raise ValueError("candidate source is not exclusively short")
    if not frame["__decision_ts__"].eq(frame["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("short candidate decision timestamp is not signal +1h")
    return frame


def _long_substrate(root: Path, month: pd.Timestamp) -> pd.DataFrame:
    path = root / "parts" / f"month={month:%Y-%m}" / "side=long.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    fields = [
        "__ts__", "__symbol__", "__decision_ts__", "label_valid", "target_invalid",
        "tp6_sl4_entry_price", "atr_1h", "atr_bps",
    ]
    frame = pd.read_parquet(path, columns=fields)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame.duplicated(["__ts__", "__symbol__"]).any():
        raise ValueError(f"long causal substrate duplicates keys for {month:%Y-%m}")
    return frame


def _materialise_month(
    candidates: pd.DataFrame, substrate: pd.DataFrame, minute_root: Path,
) -> pd.DataFrame:
    audit = candidates.merge(
        substrate,
        on=["__ts__", "__symbol__", "__decision_ts__"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_long"),
    )
    if audit.label_valid.isna().any():
        raise ValueError("a short candidate has no same-time long causal ATR substrate")
    outputs: list[pd.DataFrame] = []
    for symbol, group in audit.groupby("__symbol__", sort=True):
        # The path starts at the earliest actual entry.  Unlike recomputing
        # ATR, no pre-January minute data is needed here.
        start = group["__decision_ts__"].min()
        end = group["__decision_ts__"].max() + pd.Timedelta(hours=12)
        minute = _minute_path_pruned(minute_root, _packb_to_kraken_symbol(str(symbol)), start, end)
        atr = pd.Series(
            pd.to_numeric(group["atr_1h"], errors="coerce").to_numpy(float),
            index=pd.DatetimeIndex(group["__ts__"]),
        )
        labelled = _label_candidates_with_minute(
            group[["candidate_id", "__ts__", "__symbol__", "side_name"]], minute, atr_hourly=atr,
        )
        expected_valid = group["label_valid"].astype(bool).to_numpy()
        actual_valid = labelled["label_valid"].astype(bool).to_numpy()
        if not np.array_equal(actual_valid, expected_valid):
            raise AssertionError(
                f"shared causal substrate validity differs for {symbol}; "
                "direction must not affect entry/ATR/path completeness"
            )
        valid = actual_valid
        if valid.any():
            if not np.allclose(
                labelled.loc[valid, "tp6_sl4_entry_price"].to_numpy(float),
                pd.to_numeric(group.loc[valid, "tp6_sl4_entry_price"], errors="coerce").to_numpy(float),
                rtol=0.0, atol=1e-10,
            ):
                raise AssertionError("short exact entry differs from shared causal entry substrate")
            if not np.allclose(
                labelled.loc[valid, "atr_1h"].to_numpy(float),
                pd.to_numeric(group.loc[valid, "atr_1h"], errors="coerce").to_numpy(float),
                rtol=0.0, atol=1e-10,
            ):
                raise AssertionError("short ATR differs from shared causal ATR substrate")
        outputs.append(labelled)
    out = pd.concat(outputs, ignore_index=True).sort_values(["__ts__", "__symbol__", "candidate_id"], kind="stable")
    if len(out) != len(candidates) or out.candidate_id.duplicated().any():
        raise AssertionError("short label materialisation changed candidate identity")
    valid = out.label_valid.astype(bool)
    if out.loc[~valid, [column for column in OUTPUT_COLUMNS if column.startswith(("t2_", "t4_", "first_", "pre_", "lower_", "robust_"))]].notna().any().any():
        raise AssertionError("invalid short row was encoded as an economic failure")
    if not np.allclose(
        out.loc[valid, "t4_tp6_sl4_gross_bps"].to_numpy(float) - COST_BPS,
        out.loc[valid, "t4_tp6_sl4_net_bps"].to_numpy(float), rtol=0.0, atol=2e-3,
    ):
        raise AssertionError("short gross/net cost identity failed")
    return out.loc[:, list(OUTPUT_COLUMNS)].reset_index(drop=True)


def run(
    candidates_path: Path, long_labels: Path, minute_root: Path, out: Path,
    *, start: pd.Timestamp = START, end: pd.Timestamp = END,
) -> Path:
    if out.exists():
        raise FileExistsError(f"output must be new: {out}")
    records: list[dict[str, object]] = []
    if start.tzinfo is None or end.tzinfo is None or not start < end:
        raise ValueError("start/end must be increasing UTC timestamps")
    if start.day != 1 or start.hour != 0 or end.day != 1 or end.hour != 0:
        raise ValueError("shared-ATR label windows must use UTC month boundaries")
    for month in pd.date_range(start, end, freq="MS", inclusive="left"):
        candidates = _candidate_month(candidates_path, month)
        labelled = _materialise_month(candidates, _long_substrate(long_labels, month), minute_root)
        destination = out / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
        _atomic_parquet(labelled, destination)
        valid = labelled.label_valid.astype(bool)
        records.append({
            "month": f"{month:%Y-%m}", "rows": int(len(labelled)),
            "valid_rows": int(valid.sum()), "invalid_rows": int((~valid).sum()),
            "valid_fraction": float(valid.mean()),
            "duplicate_candidate_ids": int(labelled.candidate_id.duplicated().sum()),
            "invalid_target_rows": int(labelled.loc[~valid, "t4_tp6_sl4_net_bps"].notna().sum()),
        })
    coverage = pd.DataFrame(records)
    coverage.to_parquet(out / "coverage.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "side": "short",
        "months": [record["month"] for record in records],
        "candidate_source": str(candidates_path),
        "long_causal_atr_substrate": str(long_labels),
        "long_substrate_sha256": _sha(long_labels / "run_manifest.json"),
        "minute_root": str(minute_root),
        "entry_and_atr": "reused only from same timestamp/symbol causal long substrate; verified equal to reopened short decision entry/ATR",
        "directional_outcomes": "all short TP/SL, MFE, MAE, net, and robust-clear targets re-materialised from exact 1m high/low/close; no long outcome reused",
        "target_contract": "exact H12 TP +6 ATR / SL -4 ATR, adverse same-minute tie, robust clear B25/T50, 100 bps cost once",
        "coverage": records,
    }
    payload = json.dumps(manifest, indent=2, default=str) + "\n"
    (out / "run_manifest.json").write_text(payload)
    (out / "manifest.json").write_text(payload)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--long-labels", type=Path, default=DEFAULT_LONG_LABELS)
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default=START.isoformat())
    parser.add_argument("--end", default=END.isoformat(), help="exclusive UTC month boundary")
    args = parser.parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    print(run(args.candidates, args.long_labels, args.minute_root, args.out, start=start, end=end))


if __name__ == "__main__":
    main()
