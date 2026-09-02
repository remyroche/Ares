#!/usr/bin/env python3
"""Reproduce sealed C1-LVA dual-MC1 outputs through the canonical assembler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from extreme_price_movements.inference.p8u_c1_lva_canonical_stack import C1LVACanonicalStack


DEFAULT_CONFIG = ROOT / "config/strict_r3_p8u_c1_lva_canonical_20260901_v1.json"
DEFAULT_PACKAGE = ROOT / "data_perp/artifacts/p8u_c1_full_coverage_dual_mc1_prequential_mayjul_20260901_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/c1_lva_canonical_parity_20260901_v1"
IDENTITY = ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]
CORE = [
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
]
C1 = [
    "sr_long_support_hold_strength", "sr_long_resistance_break_probability",
    "sr_long_downside_break_probability", "sr_long_resistance_rejection_strength",
    "sr_long_structure_balance", "sr_long_support_distance_atr",
    "sr_long_resistance_distance_atr", "sr_support_prior_strength",
    "sr_resistance_prior_strength", "sr_support_reaction_magnitude_q50",
    "sr_resistance_reaction_magnitude_q50", "sr_snapshot_available",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config, package_root, output = args.config.resolve(), args.package_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must be immutable: {output}")
    stack = C1LVACanonicalStack.load(config.relative_to(ROOT), root=ROOT)
    dual = pd.read_parquet(package_root / "dual_target_free_predictions.parquet")
    bcf_all = pd.read_parquet(package_root / "predictions_bcf_target_free.parquet")
    current_all = pd.read_parquet(package_root / "predictions_current_target_free.parquet")
    dual["__decision_ts__"] = pd.to_datetime(dual["__decision_ts__"], utc=True, errors="raise")
    records: list[dict[str, object]] = []
    for month, expected in dual.groupby(dual["__decision_ts__"].dt.strftime("%Y-%m"), sort=True):
        identifiers = expected["candidate_id"]
        bcf = bcf_all.loc[bcf_all["candidate_id"].isin(identifiers), [*IDENTITY, *CORE]]
        current = current_all.loc[current_all["candidate_id"].isin(identifiers), [*IDENTITY, *CORE]]
        observed = stack.score_coordinates(
            bcf_coordinates=bcf, current_coordinates=current,
            c1_snapshots=expected.loc[:, [*IDENTITY, *C1]],
        ).scores.sort_values("candidate_id", kind="stable").reset_index(drop=True)
        truth = expected.sort_values("candidate_id", kind="stable").reset_index(drop=True)
        if observed["candidate_id"].tolist() != truth["candidate_id"].tolist():
            raise AssertionError(f"{month}: canonical identity population differs from sealed package")
        bcf_delta = float(np.max(np.abs(observed["bcf_mc1_expected_bps"] - truth["bcf_mc1_expected_bps"])))
        current_delta = float(np.max(np.abs(observed["current_mc1_expected_bps"] - truth["current_mc1_expected_bps"])))
        admissions_equal = bool((observed["dual_mc1_admitted"].to_numpy() == truth["dual_mc1_admitted"].to_numpy()).all())
        if bcf_delta != 0.0 or current_delta != 0.0 or not admissions_equal:
            raise AssertionError(f"{month}: canonical C1 parity failed")
        records.append({
            "month": month, "rows": int(len(observed)), "bcf_max_abs_delta": bcf_delta,
            "current_max_abs_delta": current_delta, "admission_exact": admissions_equal,
        })
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "p8u_c1_lva_canonical_contract_parity_v1",
        "config": str(config.relative_to(ROOT)), "config_sha256": _sha256(config),
        "package_root": str(package_root.relative_to(ROOT)),
        "package_run_manifest_sha256": _sha256(package_root / "run_manifest.json"),
        "target_free_only": True,
        "result": "PASS", "months": records,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
