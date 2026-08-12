#!/usr/bin/env python3
"""Create a bounded, strict-OOF Bayesian adjustment for auction ordering.

The sidecar never changes ``final_score``, the causal EV map, or admission.
It is merged only after causal admission and used solely to rank simultaneously
actionable admitted candidates in the portfolio auction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--admission-provenance", type=Path, required=True)
    parser.add_argument("--bayesian-oof", type=Path, required=True)
    parser.add_argument("--max-adjustment-bps", type=float, required=True)
    parser.add_argument("--scale-bps", type=float, default=100.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if not 0.0 < float(args.max_adjustment_bps) <= 100.0:
        raise ValueError("--max-adjustment-bps must lie in (0, 100]")
    if float(args.scale_bps) <= 0.0:
        raise ValueError("--scale-bps must be positive")

    primary = pd.read_parquet(args.scored_label_ledger, columns=["candidate_id"])
    admission = pd.read_parquet(args.admission_provenance)
    if "raw_expected_bps" in admission:
        admission = admission.loc[:, ["candidate_id", "raw_expected_bps", "mapped_ev_available"]].copy()
    elif "causal_21d_side_expected_net_bps" in admission:
        admission = admission.loc[:, ["candidate_id", "causal_21d_side_expected_net_bps"]].rename(
            columns={"causal_21d_side_expected_net_bps": "raw_expected_bps"},
        )
        admission["mapped_ev_available"] = np.isfinite(
            pd.to_numeric(admission["raw_expected_bps"], errors="coerce"),
        )
    else:
        raise ValueError(
            "admission provenance needs raw_expected_bps/mapped_ev_available or "
            "causal_21d_side_expected_net_bps",
        )
    bayes = pd.read_parquet(args.bayesian_oof, columns=[
        "candidate_id", "n5_available", "posterior_expected_bps",
    ])
    for name, frame in (("primary", primary), ("admission", admission), ("bayes", bayes)):
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate candidate_id")
    if len(primary) != len(admission) or set(primary["candidate_id"]) != set(admission["candidate_id"]):
        raise ValueError("admission provenance must cover primary candidates exactly")
    if len(primary) != len(bayes) or set(primary["candidate_id"]) != set(bayes["candidate_id"]):
        raise ValueError("Bayesian OOF ledger must cover primary candidates exactly")

    frame = primary.merge(admission, on="candidate_id", how="inner", validate="one_to_one")
    frame = frame.merge(bayes, on="candidate_id", how="inner", validate="one_to_one")
    parent = pd.to_numeric(frame["raw_expected_bps"], errors="coerce").to_numpy(float)
    local = pd.to_numeric(frame["posterior_expected_bps"], errors="coerce").to_numpy(float)
    available = (
        frame["n5_available"].fillna(False).to_numpy(bool)
        & frame["mapped_ev_available"].fillna(False).to_numpy(bool)
        & np.isfinite(parent)
        & np.isfinite(local)
    )
    adjustment = np.zeros(len(frame), dtype=np.float64)
    adjustment[available] = float(args.max_adjustment_bps) * np.tanh(
        (local[available] - parent[available]) / float(args.scale_bps),
    )
    frame["auction_rank_adjustment_bps"] = adjustment.astype(np.float32)
    out = frame.loc[:, ["candidate_id", "auction_rank_adjustment_bps"]]
    args.out_dir.mkdir(parents=True)
    out.to_parquet(args.out_dir / "bayesian_auction_adjustment.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_bayesian_auction_adjustment_v1",
        "role": "post-admission auction ordering only; causal EV map and admission unchanged",
        "max_adjustment_bps": float(args.max_adjustment_bps),
        "scale_bps": float(args.scale_bps),
        "rows": int(len(frame)),
        "available_rows": int(available.sum()),
        "unavailable_rows_are_exact_zero": bool(np.all(adjustment[~available] == 0.0)),
        "raw_k9_memberships_used": False,
        "source_hashes": {
            "scored_label_ledger": _sha(args.scored_label_ledger),
            "admission_provenance": _sha(args.admission_provenance),
            "bayesian_oof": _sha(args.bayesian_oof),
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
