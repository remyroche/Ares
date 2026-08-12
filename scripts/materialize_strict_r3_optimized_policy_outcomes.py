#!/usr/bin/env python3
"""Materialise source-aligned selected-policy outcomes from causal 15m bars.

This producer is label-only.  It reads the immutable target-free candidate
identities from a strict-R3 source panel and simulates the frozen
SimplePolicyOptimiser winner independently for every candidate.  It never
filters candidates by future-path availability: incomplete paths remain in
the output with ``policy_path_valid = false``.

No one-minute source is consulted.  ATR is either the decision-time value
present in the source panel or a Wilder-14 proxy computed from complete prior
15-minute bars.  Entry is the first 15-minute open at signal close + one hour;
the horizon is twelve hours and the declared 100-bps cost is deducted once.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_strict_r3_policy_labels_coarse import (  # noqa: E402
    COST_BPS,
    _policy,
    _replay_coarse_symbol,
)


IDENTITY_FIELDS = (
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    )


def _load_candidates(
    source: Path, *, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    columns = list(IDENTITY_FIELDS)
    schema = set(pq.ParquetFile(source).schema_arrow.names)
    if "atr_1h" in schema:
        columns.append("atr_1h")
    frame = pd.read_parquet(
        source,
        columns=columns,
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["__decision_ts__"] = pd.to_datetime(
        frame["__decision_ts__"], utc=True, errors="raise",
    )
    frame = frame.loc[frame["side_name"].astype(str).str.lower().eq("long")].copy()
    if "atr_1h" not in frame:
        frame["atr_1h"] = float("nan")
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("source-aligned long candidate population is empty or duplicated")
    if not frame["__decision_ts__"].eq(frame["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("candidate decision timestamp is not signal timestamp + one hour")
    return frame.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    start, end = _utc(args.start), _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("end-exclusive must be after start")
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_payload = json.loads(args.policy_json.read_text())
    policy = _policy(policy_payload)
    candidates = _load_candidates(args.source_panel, start=start, end=end)

    checkpoint_dir = args.out_dir / "symbol_parts"
    checkpoint_dir.mkdir(exist_ok=True)
    symbols = sorted(candidates["__symbol__"].astype(str).unique())
    audits: list[dict[str, object]] = []
    for number, symbol in enumerate(symbols, 1):
        group = candidates.loc[candidates["__symbol__"].astype(str).eq(symbol)].copy()
        checkpoint = checkpoint_dir / f"{hashlib.sha256(symbol.encode()).hexdigest()[:20]}.parquet"
        if checkpoint.exists():
            existing = pd.read_parquet(checkpoint, columns=["candidate_id"])
            if set(existing["candidate_id"]) != set(group["candidate_id"]):
                raise ValueError(f"checkpoint identity mismatch for {symbol}")
            result = pd.read_parquet(checkpoint)
            status = "resumed"
        else:
            result = _replay_coarse_symbol(group, policy)
            result.to_parquet(checkpoint, index=False, compression="zstd")
            status = "complete"
        audits.append({
            "symbol": symbol,
            "rows": int(len(result)),
            "valid_rows": int(result["policy_path_valid"].fillna(False).sum()),
            "status": status,
        })
        if number % 20 == 0 or number == len(symbols):
            gc.collect()
            pa.default_memory_pool().release_unused()
            print(json.dumps({
                "event": "policy_symbols_complete", "completed": number,
                "total": len(symbols),
            }), flush=True)

    pieces = [pd.read_parquet(path) for path in sorted(checkpoint_dir.glob("*.parquet"))]
    output = pd.concat(pieces, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if len(output) != len(candidates) or output["candidate_id"].duplicated().any():
        raise AssertionError("policy outcome assembly changed candidate identity/cardinality")
    valid = output["policy_path_valid"].fillna(False).astype(bool)
    if valid.any():
        cost = output.loc[valid, "policy_gross_bps"] - output.loc[valid, "policy_net_bps"]
        if not (cost.sub(COST_BPS).abs() <= 1e-8).all():
            raise AssertionError("selected-policy cost was not applied exactly once")
    output.to_parquet(
        args.out_dir / "candidate_policy_outcomes.parquet",
        index=False, compression="zstd",
    )
    pd.DataFrame(audits).to_parquet(args.out_dir / "symbol_audit.parquet", index=False)
    monthly = output.assign(
        month=output["__decision_ts__"].dt.strftime("%Y-%m"),
    ).groupby("month", as_index=False).agg(
        rows=("candidate_id", "size"), valid_rows=("policy_path_valid", "sum"),
    )
    monthly["coverage"] = monthly["valid_rows"] / monthly["rows"]
    monthly.to_parquet(args.out_dir / "monthly_coverage.parquet", index=False)
    manifest = {
        "schema": "strict_r3_source_aligned_optimized_policy_outcomes_v1",
        "side": "long",
        "source_panel": str(args.source_panel),
        "source_panel_sha256": _sha(args.source_panel),
        "policy_json": str(args.policy_json),
        "policy_json_sha256": _sha(args.policy_json),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "rows": int(len(output)),
        "valid_rows": int(valid.sum()),
        "coverage": float(valid.mean()),
        "entry": "first 15-minute open at signal close + one hour",
        "policy": policy,
        "timeout_hours": 12,
        "cost_bps_once": COST_BPS,
        "bar_source": "downloaded causal 15-minute only; no minute fallback",
        "invalid_path_contract": "retained with policy_path_valid=false; never encoded as failure",
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
