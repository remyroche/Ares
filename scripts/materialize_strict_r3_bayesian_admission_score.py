#!/usr/bin/env python3
"""Materialize a causal Bayesian correction for the EV-admission score only.

The canonical final score and downstream auction ranking remain unchanged in
the source ledger.  This utility creates a separately versioned ledger whose
``final_score`` is used only to fit/apply the causal, prior-resolved EV map.
The correction is an OOF empirical-Bayes residual in fixed bps units, passed
through a bounded deterministic transform; it never uses held outcomes.
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
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--admission-provenance", type=Path, required=True)
    parser.add_argument("--bayesian-oof", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--scale-bps", type=float, default=100.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if not 0.0 < float(args.alpha) <= 0.25:
        raise ValueError("--alpha must lie in (0, 0.25]")
    if float(args.scale_bps) <= 0.0:
        raise ValueError("--scale-bps must be positive")

    ledger = pd.read_parquet(args.scored_label_ledger)
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
    for name, frame in (("ledger", ledger), ("admission", admission), ("bayesian", bayes)):
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate candidate_id")
    work = ledger.merge(admission, on="candidate_id", how="left", validate="one_to_one")
    work = work.merge(bayes, on="candidate_id", how="left", validate="one_to_one")
    if len(work) != len(ledger):
        raise AssertionError("Bayesian admission materialization changed candidate identity")
    if work["raw_expected_bps"].notna().sum() != admission["raw_expected_bps"].notna().sum():
        raise AssertionError("Bayesian admission materialization lost EV-map support")
    base = pd.to_numeric(work["final_score"], errors="coerce").to_numpy(float)
    local = pd.to_numeric(work["posterior_expected_bps"], errors="coerce").to_numpy(float)
    parent = pd.to_numeric(work["raw_expected_bps"], errors="coerce").to_numpy(float)
    available = work["n5_available"].fillna(False).to_numpy(bool) & np.isfinite(local) & np.isfinite(parent)
    bounded = np.zeros(len(work), dtype=float)
    bounded[available] = np.tanh((local[available] - parent[available]) / float(args.scale_bps))
    corrected = np.clip(base + float(args.alpha) * bounded, 0.0, 1.0)
    if not np.array_equal(corrected[~available], base[~available], equal_nan=True):
        raise AssertionError("Bayesian correction changed an unavailable/warm-up score")
    work["bayesian_admission_base_score"] = base.astype(np.float32)
    work["bayesian_admission_correction"] = bounded.astype(np.float32)
    work["bayesian_admission_score"] = corrected.astype(np.float32)
    # ``final_score`` is intentionally overwritten only in this separately
    # named map-input ledger.  Replay receives the original ledger for the
    # auction ranking and this ledger only for map provenance validation.
    work["final_score"] = corrected
    work = work.drop(columns=["raw_expected_bps", "mapped_ev_available", "n5_available", "posterior_expected_bps"])
    args.out_dir.mkdir(parents=True)
    work.to_parquet(args.out_dir / "bayesian_admission_map_ledger.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_bayesian_admission_score_v1",
        "role": "causal EV-map score only; downstream ranking remains raw canonical final_score",
        "alpha": float(args.alpha), "scale_bps": float(args.scale_bps),
        "rows": int(len(work)), "available_rows": int(available.sum()),
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
