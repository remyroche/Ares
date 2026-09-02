#!/usr/bin/env python3
"""Assemble immutable, source-aligned strict-R3 policy outcome ledgers.

Every supplied fragment must use the same frozen SimplePolicyOptimiser
contract.  The assembled ledger must contain exactly one row for every
target-free candidate in the declared interval; invalid future paths remain
present and are never recoded as economic failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_COLUMNS = (
    "candidate_id", "__decision_ts__", "policy_path_valid",
    "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_exit_reason", "policy_entry_price", "policy_exit_price",
    "policy_label_available_ts", "policy_outcome_source", "policy_cost_bps",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _manifest_for(path: Path) -> tuple[Path, dict[str, object]]:
    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"outcome fragment lacks run_manifest.json: {path}")
    return manifest_path, json.loads(manifest_path.read_text())


def _assemble(
    *, source_panel: Path, fragments: list[Path], start: pd.Timestamp,
    end: pd.Timestamp, side: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = pd.read_parquet(
        source_panel,
        columns=["candidate_id", "__decision_ts__", "side_name"],
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True)
    source = source.loc[source["side_name"].astype(str).str.lower().eq(side)].copy()
    if source.empty or source["candidate_id"].duplicated().any():
        raise ValueError(f"{side} source interval is empty or has duplicate candidate IDs")

    pieces: list[pd.DataFrame] = []
    fragment_audit: list[dict[str, object]] = []
    policy_hashes: set[str] = set()
    policy_payloads: list[dict[str, object]] = []
    for path in fragments:
        manifest_path, manifest = _manifest_for(path)
        policy_hash = str(manifest.get("policy_json_sha256", ""))
        if not policy_hash:
            raise ValueError(f"outcome fragment does not declare policy_json_sha256: {path}")
        fragment_side = str(manifest.get("side") or "long").strip().lower()
        if fragment_side != side:
            raise ValueError(
                f"outcome fragment side mismatch: expected {side}, got {fragment_side}"
            )
        policy_hashes.add(policy_hash)
        policy_payloads.append(dict(manifest.get("policy", {})))
        piece = pd.read_parquet(
            path, columns=list(POLICY_COLUMNS),
            filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
        )
        piece["__decision_ts__"] = pd.to_datetime(piece["__decision_ts__"], utc=True)
        pieces.append(piece)
        fragment_audit.append({
            "path": str(path), "sha256": _sha(path),
            "manifest": str(manifest_path), "manifest_sha256": _sha(manifest_path),
            "rows_in_interval": int(len(piece)), "policy_json_sha256": policy_hash,
            "side": fragment_side,
        })
    if len(policy_hashes) != 1 or len({json.dumps(value, sort_keys=True) for value in policy_payloads}) != 1:
        raise ValueError("outcome fragments do not use one identical optimized-policy contract")

    output = pd.concat(pieces, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        duplicate = output.loc[output["candidate_id"].duplicated(False), "candidate_id"].iloc[0]
        raise ValueError(f"outcome fragments overlap on candidate_id: {duplicate}")
    expected = source.set_index("candidate_id")["__decision_ts__"]
    observed = output.set_index("candidate_id")["__decision_ts__"]
    missing = expected.index.difference(observed.index)
    extra = observed.index.difference(expected.index)
    if len(missing):
        raise ValueError(
            "assembled outcomes do not cover the complete source population: "
            f"missing={len(missing)}, ignored_superset_rows={len(extra)}"
        )
    # Outcome stores may have been produced before a stricter, target-free
    # feature-coverage screen and therefore legitimately contain a superset of
    # identities.  They can never expand the scoring population: retain only
    # the exact source IDs and persist the ignored count in lineage.
    output = output.loc[output["candidate_id"].isin(expected.index)].copy()
    observed = output.set_index("candidate_id")["__decision_ts__"]
    observed = observed.reindex(expected.index)
    if not np.array_equal(observed.to_numpy(), expected.to_numpy()):
        raise ValueError("assembled outcome timestamps disagree with source identity")
    output = output.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)

    valid = output["policy_path_valid"].fillna(False).astype(bool)
    finite = np.isfinite(pd.to_numeric(output["policy_net_bps"], errors="coerce"))
    if not finite.loc[valid].all():
        raise ValueError("valid policy paths contain non-finite net outcomes")
    costs = pd.to_numeric(output.loc[valid, "policy_cost_bps"], errors="coerce")
    if valid.any() and not costs.sub(100.0).abs().le(1e-8).all():
        raise ValueError("canonical 100-bps cost is not present exactly once on valid rows")
    audit = {
        "policy_json_sha256": next(iter(policy_hashes)),
        "policy": policy_payloads[0],
        "fragments": fragment_audit,
        "ignored_outcome_superset_rows": int(len(extra)),
        "rows": int(len(output)), "valid_rows": int(valid.sum()),
        "coverage": float(valid.mean()),
    }
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path, action="append", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), default="long")
    args = parser.parse_args()
    start, end = _utc(args.start), _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("end-exclusive must be after start")
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    output, audit = _assemble(
        source_panel=args.source_panel, fragments=args.outcome_ledger,
        start=start, end=end, side=args.side,
    )
    args.out_dir.mkdir(parents=True)
    output_path = args.out_dir / "candidate_policy_outcomes.parquet"
    output.to_parquet(output_path, index=False, compression="zstd")
    monthly = output.assign(
        month=output["__decision_ts__"].dt.strftime("%Y-%m"),
    ).groupby("month", as_index=False).agg(
        rows=("candidate_id", "size"), valid_rows=("policy_path_valid", "sum"),
    )
    monthly["coverage"] = monthly["valid_rows"] / monthly["rows"]
    monthly.to_parquet(args.out_dir / "monthly_coverage.parquet", index=False)
    manifest = {
        "schema": "strict_r3_source_aligned_optimized_policy_outcomes_assembled_v1",
        "side": args.side, "source_panel": str(args.source_panel),
        "source_panel_sha256": _sha(args.source_panel),
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "entry": "first available bar open at signal close + one hour",
        "timeout_hours": 12, "cost_bps_once": 100.0,
        "invalid_path_contract": "retained with policy_path_valid=false",
        **audit,
        "output_sha256": _sha(output_path),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
