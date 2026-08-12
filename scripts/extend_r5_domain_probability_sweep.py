#!/usr/bin/env python3
"""Extend the frozen R5 ledger with domain-depth and probability sweeps.

All gates are functions of decision-time scores/predictions already present in
the immutable matched ledger. Outcomes are used only by the reporting helper
after each score and admission column has been constructed.
"""

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

from scripts.ablate_r5_posterior_contract import _sha


DOMAIN_FRACTIONS = (0.30, 0.25, 0.20, 0.15, 0.10)
PROBABILITY_THRESHOLDS = (0.50, 0.525, 0.55, 0.570, 0.5725, 0.575, 0.60)


def timestamp_fraction(frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    position = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    size = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    # Zero-based fraction plus a strict comparison reproduces
    # ``position < ceil(group_size * retained_fraction)`` exactly.
    fraction = position / np.maximum(size, 1)
    return pd.Series(fraction, index=ordered.index).reindex(frame.index).to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-ledger", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    frame = pd.read_parquet(args.selection_ledger, columns=[
        "candidate_id", "__decision_ts__", "final_score", "policy_path_valid",
        "policy_net_bps", "month", "a0_current__expected",
        "a0_current__admitted", "a5_calibrated__expected",
        "a5_calibrated__p_positive",
    ])
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("selection ledger contains duplicate candidates")
    frame["timestamp_score_fraction"] = timestamp_fraction(frame).astype(np.float32)
    contracts: dict[str, tuple[str, str]] = {}
    current_expected = "a0_current__expected"
    current_admitted = frame["a0_current__admitted"].fillna(False).astype(bool)
    for fraction in DOMAIN_FRACTIONS:
        pct = int(round(100 * fraction))
        token = f"a1_top{pct}"
        frame[f"{token}__expected"] = frame[current_expected]
        frame[f"{token}__admitted"] = current_admitted & frame["timestamp_score_fraction"].lt(fraction)
        contracts[f"A1_domain_top{pct}"] = (f"{token}__expected", f"{token}__admitted")
    calibrated_admitted = frame["a5_calibrated__expected"].ge(50.0)
    for threshold in PROBABILITY_THRESHOLDS:
        code = int(round(threshold * 1000))
        token = f"a6_p{code:03d}"
        frame[f"{token}__expected"] = frame["a5_calibrated__expected"]
        frame[f"{token}__admitted"] = calibrated_admitted & frame["a5_calibrated__p_positive"].ge(threshold)
        contracts[f"A6_calibrated_p{threshold:.3f}"] = (
            f"{token}__expected", f"{token}__admitted",
        )
    # Added only after the independent A1 sweep showed that top-10 improved
    # both raw admission economics and the matched portfolio replay. This is a
    # challenger combination, not a canonical promotion.
    frame["a10_a5_top10__expected"] = frame["a5_calibrated__expected"]
    frame["a10_a5_top10__admitted"] = (
        frame["a5_calibrated__expected"].ge(50.0)
        & frame["timestamp_score_fraction"].lt(0.10)
    )
    contracts["A10_A5_calibrated_domain_top10"] = (
        "a10_a5_top10__expected", "a10_a5_top10__admitted",
    )
    frame["a11_a5_top15__expected"] = frame["a5_calibrated__expected"]
    frame["a11_a5_top15__admitted"] = (
        frame["a5_calibrated__expected"].ge(50.0)
        & frame["timestamp_score_fraction"].lt(0.15)
    )
    contracts["A11_A5_calibrated_domain_top15"] = (
        "a11_a5_top15__expected", "a11_a5_top15__admitted",
    )
    args.out_dir.mkdir(parents=True)
    selection_fields = ["candidate_id", "__decision_ts__"]
    for expected, admitted in contracts.values():
        selection_fields.extend([expected, admitted])
    frame.loc[:, selection_fields].to_parquet(
        args.out_dir / "selection_ledger.parquet", index=False,
    )
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "r5_domain_probability_sweep_v1",
        "source": str(args.selection_ledger), "source_sha256": _sha(args.selection_ledger),
        "domain_fractions": list(DOMAIN_FRACTIONS),
        "probability_thresholds": list(PROBABILITY_THRESHOLDS),
        "contracts": {arm: {"expected": pair[0], "admitted": pair[1]} for arm, pair in contracts.items()},
        "winner_promoted": False,
        "outcomes_used_only_after_target_free_gate_construction": True,
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(frame), "arms": list(contracts)}))


if __name__ == "__main__":
    main()
