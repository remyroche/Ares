#!/usr/bin/env python3
"""Combine disjoint R5 fold decompositions into canonical OOF replay fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable R5 OOF output exists: {args.out_dir}")
    parts: list[pd.DataFrame] = []
    for path in args.decomposition:
        frame = pd.read_parquet(path)
        required = {
            "candidate_id", "__decision_ts__", "frozen_admission",
            "causal_21d_side_expected_net_bps", "trust_corroborated_ev_a10",
            "trust_risk_corroborated", "trust_authority_unit",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"R5 decomposition {path} lacks: {missing}")
        parts.append(frame)
    output = pd.concat(parts, ignore_index=True, sort=False)
    output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True)
    if output["candidate_id"].duplicated().any():
        raise ValueError("R5 OOF folds overlap candidate identities")
    output["auction_rank_adjustment_bps"] = (
        output["trust_corroborated_ev_a10"]
        - output["causal_21d_side_expected_net_bps"]
    )
    # The evaluator persists the corrected score as float32 while the anchor
    # remains float64.  Permit only sub-microeconomic (<1e-5 bps) roundoff and
    # clamp it to zero; any larger positive value is a real promotion failure.
    if output["auction_rank_adjustment_bps"].gt(1e-5).any():
        raise ValueError("R5 OOF adjustment is not demotion-only")
    output["auction_rank_adjustment_bps"] = output[
        "auction_rank_adjustment_bps"
    ].clip(upper=0.0)
    output = output.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    args.out_dir.mkdir(parents=True)
    path = args.out_dir / "cell_day_residual_trust_oof_predictions.parquet"
    fields = [
        "candidate_id", "__decision_ts__", "frozen_admission",
        "causal_21d_side_expected_net_bps", "trust_corroborated_ev_a10",
        "trust_risk_corroborated", "trust_authority_unit",
        "auction_rank_adjustment_bps",
    ]
    output.loc[:, fields].to_parquet(path, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_cell_day_residual_trust_oof_v1",
        "decompositions": [str(value) for value in args.decomposition],
        "decomposition_sha256": [_sha(value) for value in args.decomposition],
        "rows": int(len(output)),
        "start": output["__decision_ts__"].min().isoformat(),
        "end_exclusive": (output["__decision_ts__"].max() + pd.Timedelta(hours=1)).isoformat(),
        "admission_changes": False,
        "authority_cap": 0.10,
        "corroboration_required": True,
        "output": str(path),
        "output_sha256": _sha(path),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
