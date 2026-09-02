#!/usr/bin/env python3
"""Compose one immutable policy contract for a shifted strict-R3 phase.

The historical parent contract supplies resolved pre-May outcomes for the
score-history rows used to fit a phase-local prequential mapper.  The phase
contract supplies the later, shifted candidate identities.  This utility keeps
the two populations explicit, proves their identities do not overlap, and
produces one authoritative post-score outcome source.  It never reads scores
or features and must only be used after target-free scoring is complete.
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

from scripts.materialize_strict_r3_current_v5_policy_ledger import POLICY_COLUMNS


REQUIRED = ("candidate_id", *POLICY_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(REQUIRED)).copy()
    missing = sorted(set(REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} misses policy fields: {missing}")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{source} has duplicate candidate IDs")
    valid = frame["policy_path_valid"].fillna(False).astype(bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    available = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
    if (valid & (~np.isfinite(net) | available.isna())).any():
        raise ValueError(f"{source} has a valid row without finite, available net outcome")
    return frame.loc[:, list(REQUIRED)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--phase-policy", required=True, type=Path)
    parser.add_argument("--phase", required=True, type=int, choices=(15, 30, 45))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")

    parent = _read(args.parent, "parent policy contract")
    phase = _read(args.phase_policy, "phase policy contract")
    overlap = parent.loc[parent["candidate_id"].isin(set(phase["candidate_id"]))]
    if not overlap.empty:
        raise ValueError("parent and phase policy contracts overlap candidate identities")

    combined = pd.concat([parent, phase], ignore_index=True)
    if combined["candidate_id"].duplicated().any():
        raise AssertionError("combined policy contract duplicated candidate IDs")
    combined = combined.sort_values("candidate_id", kind="stable").reset_index(drop=True)

    args.out_dir.mkdir(parents=True)
    output = args.out_dir / "canonical_policy_contract.parquet"
    combined.to_parquet(output, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_phase_combined_policy_contract_v1",
        "phase_minutes": args.phase,
        "purpose": "pre-May parent policy support plus shifted-phase labels for strict-prequential MC1 fitting",
        "post_score_only": True,
        "parent": {"path": str(args.parent), "sha256": _sha256(args.parent), "rows": int(len(parent))},
        "phase_policy": {"path": str(args.phase_policy), "sha256": _sha256(args.phase_policy), "rows": int(len(phase))},
        "identity_overlap_rows": 0,
        "combined": {"path": str(output), "sha256": _sha256(output), "rows": int(len(combined))},
        "invalid_semantics": "invalid/missing paths remain non-numerical and are excluded by downstream fitting/evaluation",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", **manifest["combined"]}, sort_keys=True))


if __name__ == "__main__":
    main()
