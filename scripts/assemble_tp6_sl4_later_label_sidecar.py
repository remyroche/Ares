#!/usr/bin/env python3
"""Assemble an exact TP6/SL4 later-population sidecar from symbol checkpoints.

This utility is intentionally fail-closed.  Valid rows must come from the
canonical one-minute relabeler.  A symbol whose raw minute source is unreadable
is retained only as ``target_invalid`` coverage; no alternate exchange or
coarser bar source is silently substituted for an exact TP6/SL4 label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materialize_packb_tp6_sl4_h12_labels import OUTPUT_COLUMNS, SIDES


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="candidate_source directory")
    p.add_argument("--checkpoint-root", type=Path, required=True, help="exact_labels directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--month", default="2026-07")
    p.add_argument("--invalid-symbol", action="append", default=[])
    return p.parse_args()


def _invalid_rows(candidates: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(len(candidates)), columns=list(OUTPUT_COLUMNS))
    out["candidate_id"] = candidates["candidate_id"].to_numpy()
    out["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True).to_numpy()
    out["__symbol__"] = candidates["__symbol__"].astype(str).to_numpy()
    out["side_name"] = candidates["side_name"].astype(str).to_numpy()
    out["__decision_ts__"] = pd.to_datetime(out["__ts__"], utc=True) + pd.Timedelta(hours=1)
    out["__label_available_at__"] = out["__decision_ts__"] + pd.Timedelta(hours=12)
    out["kraken_minute_symbol"] = out["__symbol__"].str.replace("/", "_", regex=False)
    out["label_valid"] = False
    out["target_invalid"] = True
    out["invalid_reason"] = reason
    return out.loc[:, list(OUTPUT_COLUMNS)]


def _digest(ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def main() -> None:
    a = _args()
    month = str(a.month)
    source = a.source
    root = a.checkpoint_root
    a.out.mkdir(parents=True, exist_ok=True)
    parts = a.out / "parts" / f"month={month}"
    parts.mkdir(parents=True, exist_ok=True)

    all_rows = pd.concat(
        [pd.read_parquet(source / f"train_global_{side}_5_{month.replace('-', '_')}.parquet")
         for side in SIDES],
        ignore_index=True,
    )
    all_rows["__ts__"] = pd.to_datetime(all_rows["__ts__"], utc=True)
    if all_rows.candidate_id.duplicated().any():
        raise ValueError("candidate source contains duplicate candidate IDs")

    invalid_symbols = {str(x) for x in a.invalid_symbol}
    coverage: list[dict[str, object]] = []
    for side in SIDES:
        expected = all_rows.loc[all_rows.side_name.eq(side)].copy()
        valid_parts: list[pd.DataFrame] = []
        checkpoint_paths = list((root / "symbol_parts" / f"month={month}").glob("**/side=" + side + ".parquet"))
        checkpoint_paths += list((root / "extra_symbol_parts").glob("*/" + side + ".parquet"))
        for path in checkpoint_paths:
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            if set(frame.columns) != set(OUTPUT_COLUMNS):
                raise ValueError(f"checkpoint schema mismatch: {path}")
            frame = frame.loc[:, list(OUTPUT_COLUMNS)]
            if frame.side_name.astype(str).ne(side).any():
                raise ValueError(f"checkpoint side mismatch: {path}")
            valid_parts.append(frame)
        valid = pd.concat(valid_parts, ignore_index=True) if valid_parts else pd.DataFrame(columns=OUTPUT_COLUMNS)
        if valid.candidate_id.duplicated().any():
            raise ValueError(f"duplicate checkpoint candidate IDs for {side}")
        expected_ids = set(expected.candidate_id.astype(str))
        valid_ids = set(valid.candidate_id.astype(str))
        unknown = valid_ids - expected_ids
        if unknown:
            raise ValueError(f"checkpoint contains IDs outside source ({side}): {len(unknown)}")
        missing = expected.loc[~expected.candidate_id.astype(str).isin(valid_ids)].copy()
        missing_symbols = set(missing.__symbol__.astype(str).unique())
        if missing_symbols - invalid_symbols:
            raise ValueError(f"unaccounted missing symbols for {side}: {sorted(missing_symbols - invalid_symbols)}")
        invalid = _invalid_rows(missing, "minute_source_unreadable_exact_label_unavailable")
        out = pd.concat([valid, invalid], ignore_index=True)
        out = out.sort_values(["__ts__", "__symbol__", "candidate_id"], kind="mergesort").reset_index(drop=True)
        if len(out) != len(expected) or set(out.candidate_id.astype(str)) != expected_ids:
            raise ValueError(f"assembled candidate identity mismatch for {side}")
        if out.candidate_id.duplicated().any():
            raise ValueError(f"assembled candidate IDs are not unique for {side}")
        invalid_mask = out.target_invalid.astype(bool)
        economic = [c for c in OUTPUT_COLUMNS if c.startswith(("t2_", "t4_", "first_", "pre_", "lower_", "robust_"))]
        if out.loc[invalid_mask, economic].notna().any().any():
            raise ValueError(f"invalid rows acquired economic targets for {side}")
        good = ~invalid_mask
        if good.any() and not np.allclose(
            out.loc[good, "t4_tp6_sl4_gross_bps"].to_numpy(float) - 100.0,
            out.loc[good, "t4_tp6_sl4_net_bps"].to_numpy(float),
            atol=2e-3,
            rtol=0.0,
        ):
            raise ValueError(f"gross/net identity failed for {side}")
        destination = parts / f"side={side}.parquet"
        out.to_parquet(destination, index=False, compression="zstd")
        coverage.append({
            "month": month, "side": side, "rows": int(len(out)),
            "valid_rows": int(good.sum()), "invalid_rows": int(invalid_mask.sum()),
            "valid_fraction": float(good.mean()),
            "invalid_reason_counts": json.dumps(out.loc[invalid_mask, "invalid_reason"].value_counts().to_dict(), sort_keys=True),
            "candidate_ids_sha256": _digest(out.candidate_id),
        })

    coverage_frame = pd.DataFrame(coverage).sort_values("side")
    coverage_frame.to_parquet(a.out / "coverage.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "exact_tp6_sl4_later_population_sidecar_v1",
        "status": "complete",
        "month": month,
        "candidate_source": str(source),
        "checkpoint_root": str(root),
        "contract": {
            "entry": "signal close + 1h; exact next-minute open",
            "atr": "14 completed hourly candles from one-minute OHLC; Wilder alpha=1/14; signal-close causal",
            "geometry": "TP +6 ATR / SL -4 ATR / H12",
            "same_minute_conflict": "adverse (SL) precedence",
            "cost_bps": 100.0,
            "net_formula": "gross_bps - 100 exactly once",
            "invalid_rows": "target_invalid=true; all economic/R3 targets null",
        },
        "invalid_symbols": sorted(invalid_symbols),
        "coverage": coverage,
        "parts": [str(parts / f"side={side}.parquet") for side in SIDES],
    }
    (a.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(a.out), "rows": int(coverage_frame.rows.sum()), "valid_rows": int(coverage_frame.valid_rows.sum()), "invalid_rows": int(coverage_frame.invalid_rows.sum())}, indent=2))


if __name__ == "__main__":
    main()
