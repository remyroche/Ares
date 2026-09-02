#!/usr/bin/env python3
"""Compose pre-May and evaluation policy labels for one shifted phase.

Both inputs must be phase-native.  This is a post-score outcome overlay: it
does not read features or predictions and never creates a numerical target for
an invalid path.
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _read(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(REQUIRED)).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{name} duplicated candidate IDs")
    valid = frame["policy_path_valid"].fillna(False).astype(bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    available = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
    if (valid & (~np.isfinite(net) | available.isna())).any():
        raise ValueError(f"{name} contains a valid path without an available finite policy outcome")
    return frame.loc[:, list(REQUIRED)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-may", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--phase", type=int, choices=(15, 30, 45), required=True)
    parser.add_argument(
        "--evaluation-label-available-from", required=True,
        help=(
            "First resolved-label timestamp delegated to the later evaluation "
            "contract. Earlier rows are authoritatively supplied by --pre-may."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    pre_may = _read(args.pre_may, "pre-May phase policy")
    evaluation = _read(args.evaluation, "evaluation phase policy")
    boundary = pd.Timestamp(args.evaluation_label_available_from)
    boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    pre_available = pd.to_datetime(pre_may["policy_label_available_ts"], utc=True, errors="raise")
    evaluation_available = pd.to_datetime(evaluation["policy_label_available_ts"], utc=True, errors="raise")
    if pre_available.ge(boundary).any():
        raise ValueError("pre-May label input crosses the declared evaluation boundary")
    evaluation = evaluation.loc[evaluation_available.ge(boundary)].copy()
    if evaluation.empty:
        raise ValueError("evaluation policy has no rows after the declared boundary")
    if pre_may["candidate_id"].isin(set(evaluation["candidate_id"])).any():
        raise ValueError("pre-May and evaluation policy inputs overlap identities")
    output = pd.concat([pre_may, evaluation], ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("combined policy contract duplicated candidate identities")
    args.out_dir.mkdir(parents=True)
    path = args.out_dir / "canonical_policy_contract.parquet"
    output.to_parquet(path, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_phase_native_combined_policy_contract_v1",
        "phase_minutes": args.phase,
        "post_score_only": True,
        "evaluation_label_available_from": boundary.isoformat(),
        "pre_may": {"path": str(args.pre_may), "sha256": _sha(args.pre_may), "rows": int(len(pre_may))},
        "evaluation": {"path": str(args.evaluation), "sha256": _sha(args.evaluation), "rows": int(len(evaluation))},
        "combined": {"path": str(path), "sha256": _sha(path), "rows": int(len(output))},
        "invalid_semantics": "invalid paths are retained but excluded from numerical fitting and realised evaluation",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "phase": args.phase, "rows": len(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
