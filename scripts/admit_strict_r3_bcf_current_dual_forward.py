#!/usr/bin/env python3
"""Apply the sealed BCF/current-v5 dual-MC1 live admission contract.

Both maps are independently fitted and scored.  The output is suitable for a
single common portfolio auction: both expected EVs must be >=30 bps, while
BCF MC1 expected EV is the only priority coordinate.
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

from extreme_price_movements.strict_r3_bcf_mc1_mapper import BCFMC1D2Bundle


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-admission", type=Path, required=True)
    parser.add_argument("--bcf-predictions", type=Path, required=True)
    parser.add_argument("--bcf-resolved-ledger", type=Path, required=True)
    parser.add_argument("--bcf-mc1-bundle-dir", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    decision = _utc(args.decision_ts)
    current = pd.read_parquet(args.current_admission)
    current["__decision_ts__"] = pd.to_datetime(current["__decision_ts__"], utc=True)
    current = current.loc[current["__decision_ts__"].eq(decision)].copy()
    if current["candidate_id"].duplicated().any():
        raise ValueError("current admission has duplicate candidate IDs")
    bcf = pd.read_parquet(args.bcf_predictions)
    bcf["__decision_ts__"] = pd.to_datetime(bcf["__decision_ts__"], utc=True)
    bcf = bcf.loc[bcf["__decision_ts__"].eq(decision)].copy()
    if bcf["candidate_id"].duplicated().any():
        raise ValueError("BCF score frame has duplicate candidate IDs")
    missing_current = sorted(set(current["candidate_id"]).difference(set(bcf["candidate_id"])))
    if missing_current:
        raise ValueError(f"BCF scorer omitted {len(missing_current)} current-route IDs")
    bcf = bcf.loc[bcf["candidate_id"].isin(set(current["candidate_id"]))].copy()
    controller = BCFMC1D2Bundle.load(args.bcf_mc1_bundle_dir)
    history = pd.read_parquet(args.bcf_resolved_ledger)
    mapped = controller.score(bcf, resolved_history=history, decision_ts=decision)
    out = current.merge(mapped, on="candidate_id", how="inner", validate="one_to_one")
    route_field = "base_route_timestamp_top30" if "base_route_timestamp_top30" in out else "base_route_timestamp_top20"
    complete = out.get("frozen_base_contract_complete", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    routed = out.get(route_field, pd.Series(False, index=out.index)).fillna(False).astype(bool)
    current_ev = pd.to_numeric(out.get("mc1_d2_expected_net_bps"), errors="coerce")
    current_available = out.get("mc1_d2_available", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["current_mc1_admitted_ge_30bps"] = current_available & np.isfinite(current_ev) & current_ev.ge(30.0)
    out["dual_bcf_current_admitted_ge_30bps"] = (
        complete & routed & out["current_mc1_admitted_ge_30bps"]
        & out["bcf_mc1_admitted_ge_30bps"].fillna(False).astype(bool)
    )
    # Standard aliases allow the unchanged execution-reporting path to retain
    # an explicit expected-EV value.  It is BCF MC1 because that is the frozen
    # common-auction priority and the map used for execution friction checks.
    out["causal_21d_side_expected_net_bps"] = out["bcf_mc1_expected_net_bps"]
    out["causal_21d_side_admitted_ge_50bps"] = out["dual_bcf_current_admitted_ge_30bps"]
    out["dual_auction_priority_bps"] = out["bcf_mc1_expected_net_bps"]
    out["dual_admission_rejection_reason"] = np.select(
        [
            ~complete, ~routed, ~current_available,
            ~out["bcf_mc1_available"].fillna(False).astype(bool),
            current_ev.lt(30.0),
            pd.to_numeric(out["bcf_mc1_expected_net_bps"], errors="coerce").lt(30.0),
        ],
        [
            "frozen_base_contract_incomplete", "stopped_after_base_below_route",
            "current_mc1_unavailable", "bcf_mc1_unavailable",
            "current_mc1_expected_net_below_30bps", "bcf_mc1_expected_net_below_30bps",
        ], default="",
    )
    args.out_dir.mkdir(parents=True)
    out.to_parquet(args.out_dir / "admitted_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_bcf_current_dual_mc1_admission_v1",
        "decision_ts": decision.isoformat(), "current_outcomes_consumed": [],
        "current_rows": int(len(current)), "bcf_rows": int(len(bcf)), "joined_rows": int(len(out)),
        "current_mc1_admitted_ge_30bps": int(out["current_mc1_admitted_ge_30bps"].sum()),
        "bcf_mc1_admitted_ge_30bps": int(out["bcf_mc1_admitted_ge_30bps"].sum()),
        "dual_admitted_ge_30bps": int(out["dual_bcf_current_admitted_ge_30bps"].sum()),
        "threshold_bps": 30.0, "auction_priority": "bcf_mc1_expected_net_bps",
        "route_field": route_field, "bcf_mc1_bundle_id": controller.manifest["bundle_id"],
        "bcf_mc1_bundle_manifest_sha256": _sha(args.bcf_mc1_bundle_dir / "run_manifest.json"),
        "inputs": {
            "current_admission": _sha(args.current_admission),
            "bcf_predictions": _sha(args.bcf_predictions),
            "bcf_resolved_ledger": _sha(args.bcf_resolved_ledger),
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
