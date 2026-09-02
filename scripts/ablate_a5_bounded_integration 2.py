#!/usr/bin/env python3
"""Repair A5 as a bounded secondary layer over A0 top-15 admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extend_r5_domain_probability_sweep import timestamp_fraction
from scripts.ablate_r5_posterior_contract import _sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    frame = pd.read_parquet(args.source, columns=[
        "candidate_id", "__decision_ts__", "final_score",
        "a0_current__expected", "a0_current__admitted",
        "a5_calibrated__expected", "a5_calibrated__p_positive",
    ])
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["timestamp_score_fraction"] = timestamp_fraction(frame).astype(np.float32)
    domain = frame["timestamp_score_fraction"].lt(0.15)
    a0 = pd.to_numeric(frame["a0_current__expected"], errors="coerce")
    a5 = pd.to_numeric(frame["a5_calibrated__expected"], errors="coerce")
    a0_admitted = frame["a0_current__admitted"].fillna(False).astype(bool) & domain
    a5_admitted = a5.ge(50.0) & domain
    delta = a5 - a0
    contracts: dict[str, tuple[str, str]] = {}

    def add(arm: str, expected: np.ndarray | pd.Series, admitted: np.ndarray | pd.Series) -> None:
        token = arm.lower()
        frame[f"{token}__expected"] = np.asarray(expected, dtype=np.float32)
        frame[f"{token}__admitted"] = np.asarray(admitted, dtype=bool)
        contracts[arm] = (f"{token}__expected", f"{token}__admitted")

    add("F0_A0_top15", a0, a0_admitted)
    add("F1_A5_rerank_fixed_A0_top15", a5, a0_admitted)
    for alpha in (0.10, 0.15, 0.20, 0.25, 0.30):
        code = int(round(alpha * 100))
        add(f"F2_blend_a{code:02d}_fixed_A0_top15", a0 + alpha * delta, a0_admitted)
        add(
            f"F3_demotion_a{code:02d}_fixed_A0_top15",
            a0 + alpha * np.minimum(delta, 0.0), a0_admitted,
        )
    add(
        "F4_capped50_a20_fixed_A0_top15",
        a0 + 0.20 * np.clip(delta, -50.0, 50.0), a0_admitted,
    )
    # Promotion-only union: A5 may add candidates, but it can never remove an
    # A0 admission. Ranking uses a conservative 20% correction.
    union = a0_admitted | a5_admitted
    add("F5_A0_floor_A5_union_blend20", a0 + 0.20 * delta, union)
    strong_add = a0_admitted | (
        a5_admitted & frame["a5_calibrated__p_positive"].ge(0.575)
    )
    add("F6_A0_floor_A5_strong_union_blend20", a0 + 0.20 * delta, strong_add)

    args.out_dir.mkdir(parents=True)
    fields = ["candidate_id", "__decision_ts__"]
    for expected, admitted in contracts.values():
        fields.extend([expected, admitted])
    frame.loc[:, fields].to_parquet(args.out_dir / "selection_ledger.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "a5_bounded_integration_ablation_v1",
        "source": str(args.source), "source_sha256": _sha(args.source),
        "domain": "timestamp-local top15 using position < ceil(0.15*n)",
        "contracts": {arm: {"expected": pair[0], "admitted": pair[1]} for arm, pair in contracts.items()},
        "a0_admission_is_floor_for_all_F1_F6_arms": True,
        "winner_promoted": False,
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(frame), "arms": list(contracts)}))


if __name__ == "__main__":
    main()
