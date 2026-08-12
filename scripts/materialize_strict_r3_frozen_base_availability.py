#!/usr/bin/env python3
"""Materialise the forward frozen-base availability gate for OOF replay.

The forward scorer deliberately retains a score for diagnostic coverage even
when a raw base feature is missing, but the executable admission command then
fails that row closed.  Historical OOF admission must apply precisely the
same gate before reporting portfolio or threshold results.

This reader is bounded-memory: it scans only the declared frozen base fields
in parquet batches and retains rows already present in the supplied OOF
ledger.  It never reads outcomes to decide feature availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    load_monthly_upstream_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _availability(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float, copy=False))
    output = frame.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].copy()
    output["frozen_base_feature_count"] = finite.sum(axis=1).astype("int16")
    output["frozen_base_feature_fraction"] = finite.mean(axis=1).astype("float32")
    output["frozen_base_contract_complete"] = finite.all(axis=1).astype(bool)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--oof-ledger", type=Path, required=True)
    parser.add_argument("--upstream-bundle-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=50_000)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable availability output already exists: {args.out_dir}")
    if args.batch_rows < 1:
        raise ValueError("--batch-rows must be positive")

    bundle = load_monthly_upstream_bundle(args.upstream_bundle_dir)
    fields = tuple(bundle.base_fields)
    if len(fields) != 120 or len(set(fields)) != len(fields):
        raise ValueError("availability requires the exact unique 120-field frozen base contract")
    ledger = pd.read_parquet(args.oof_ledger, columns=["candidate_id"])
    if ledger["candidate_id"].isna().any() or ledger["candidate_id"].duplicated().any():
        raise ValueError("OOF ledger needs unique immutable candidate identities")
    wanted = set(ledger["candidate_id"].astype(str))
    columns = ["candidate_id", "__decision_ts__", "side_name", *fields]
    parquet = pq.ParquetFile(args.source_panel)
    missing = sorted(set(columns).difference(parquet.schema.names))
    if missing:
        raise ValueError(f"source panel lacks frozen availability fields: {missing}")

    parts: list[pd.DataFrame] = []
    scanned_rows = 0
    feature_finite_rows = np.zeros(len(fields), dtype=np.int64)
    feature_monthly_parts: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(columns=columns, batch_size=args.batch_rows):
        frame = batch.to_pandas()
        scanned_rows += len(frame)
        subset = frame.loc[frame["candidate_id"].astype(str).isin(wanted)].copy()
        if not subset.empty:
            numeric = subset.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
            finite = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
            feature_finite_rows += finite.sum(axis=0, dtype=np.int64)
            month = pd.to_datetime(subset["__decision_ts__"], utc=True).dt.to_period("M").astype(str)
            for name in month.drop_duplicates().tolist():
                index = np.flatnonzero(month.eq(name).to_numpy())
                feature_monthly_parts.append(pd.DataFrame({
                    "month": str(name),
                    "feature": list(fields),
                    "finite_rows": finite[index].sum(axis=0, dtype=np.int64),
                    "rows": int(len(index)),
                }))
            parts.append(_availability(subset, fields))
    if not parts:
        raise ValueError("source panel has no identities from the OOF ledger")
    output = pd.concat(parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if output["candidate_id"].duplicated().any():
        raise ValueError("source panel produced duplicate OOF candidate availability rows")
    missing_ids = sorted(wanted.difference(set(output["candidate_id"].astype(str))))
    if missing_ids:
        raise ValueError(
            f"source panel misses {len(missing_ids)} OOF candidate availability rows",
        )
    if len(output) != len(ledger):
        raise AssertionError("availability row count differs from OOF ledger")

    output["__decision_ts__"] = pd.to_datetime(
        output["__decision_ts__"], utc=True, errors="raise",
    )
    by_month = (
        output.assign(month=output["__decision_ts__"].dt.to_period("M").astype(str))
        .groupby("month", observed=True, sort=True)
        .agg(
            rows=("candidate_id", "size"),
            complete_rows=("frozen_base_contract_complete", "sum"),
            mean_feature_fraction=("frozen_base_feature_fraction", "mean"),
        )
        .reset_index()
    )
    by_month["complete_fraction"] = by_month["complete_rows"] / by_month["rows"]
    feature_coverage = pd.DataFrame({
        "feature": list(fields),
        "finite_rows": feature_finite_rows,
        "coverage_fraction": feature_finite_rows / max(len(output), 1),
    }).sort_values(["coverage_fraction", "feature"], kind="stable").reset_index(drop=True)
    feature_monthly_coverage = pd.concat(feature_monthly_parts, ignore_index=True).groupby(
        ["month", "feature"], observed=True, sort=True,
    ).agg(finite_rows=("finite_rows", "sum"), rows=("rows", "sum")).reset_index()
    feature_monthly_coverage["coverage_fraction"] = (
        feature_monthly_coverage["finite_rows"] / feature_monthly_coverage["rows"]
    )
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "frozen_base_availability.parquet", index=False, compression="zstd")
    by_month.to_parquet(args.out_dir / "monthly_coverage.parquet", index=False)
    feature_coverage.to_parquet(args.out_dir / "feature_coverage.parquet", index=False)
    feature_monthly_coverage.to_parquet(
        args.out_dir / "feature_monthly_coverage.parquet", index=False,
    )
    manifest = {
        "schema": "strict_r3_frozen_base_availability_v1",
        "source_panel": str(args.source_panel),
        "source_panel_sha256": _sha(args.source_panel),
        "oof_ledger": str(args.oof_ledger),
        "oof_ledger_sha256": _sha(args.oof_ledger),
        "upstream_bundle_dir": str(args.upstream_bundle_dir),
        "upstream_bundle_sha256": bundle.manifest["bundle_sha256"],
        "base_field_count": len(fields),
        "base_field_sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest(),
        "scanned_source_rows": int(scanned_rows),
        "oof_rows": int(len(output)),
        "complete_rows": int(output["frozen_base_contract_complete"].sum()),
        "complete_fraction": float(output["frozen_base_contract_complete"].mean()),
        "fields_below_90pct_coverage": int(feature_coverage["coverage_fraction"].lt(0.90).sum()),
        "contract": (
            "bit-compatible with score_strict_r3_forward._frozen_base_availability; "
            "decision-time raw feature availability only; no outcomes consumed"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
