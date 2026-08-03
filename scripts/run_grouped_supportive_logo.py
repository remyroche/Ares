#!/usr/bin/env python3
"""Run leave-one-group-out support ablations for the grouped best target arm.

This is deliberately a small, candidate-level diagnostic.  It refits only the
best target formulation (by default T3) while removing one grouped support
stage at a time, then reports the same pooled-global top-k economics and saves
the strict-prequential candidate scores for independent evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.controlled_target_supportive_ablation import (
    AcceptanceGates,
    GROUPED_SUPPORT_LABELS,
    SUPPORT_STAGES,
)
from scripts.run_controlled_target_supportive_ablation import run_matrix


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *,
    ledger: Path,
    features_json: Path,
    support_oof: Path,
    output: Path,
    fold_column: str,
    hurdle_bps: float,
    target_arm: str,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    payload = json.loads(features_json.read_text())
    features = payload.get("raw_feature_columns") or payload.get("feature_columns") if isinstance(payload, dict) else payload
    if not isinstance(features, list):
        raise ValueError("features JSON must contain a feature list")
    frame = pd.read_parquet(ledger)
    frozen_support = pd.read_parquet(support_oof)
    prediction_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    contracts: dict[str, object] = {}
    for excluded_stage in SUPPORT_STAGES[1:]:
        labels = tuple(item for item in GROUPED_SUPPORT_LABELS if item[0] != excluded_stage)
        prediction, summary, contract = run_matrix(
            frame,
            feature_columns=features,
            fold_column=fold_column,
            gates=AcceptanceGates(),
            hurdle_bps=hurdle_bps,
            frozen_support_oof=frozen_support,
            support_labels=labels,
            support_spec=f"grouped_logo_exclude_{excluded_stage}",
            target_arms=(target_arm,),
        )
        prediction["logo_excluded_group"] = excluded_stage
        summary["logo_excluded_group"] = excluded_stage
        prediction_parts.append(prediction)
        summary_parts.append(summary)
        contracts[excluded_stage] = contract
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        predictions = pd.concat(prediction_parts, ignore_index=True)
        summary = pd.concat(summary_parts, ignore_index=True)
        predictions.to_parquet(stage / "logo_oof_predictions.parquet", index=False, compression="zstd")
        summary.to_parquet(stage / "logo_policy_summary.parquet", index=False, compression="zstd")
        manifest = {
            "schema": "grouped_supportive_logo_v1",
            "status": "RESEARCH_ONLY_NOT_PROMOTION",
            "target_arm": target_arm,
            "excluded_groups": list(SUPPORT_STAGES[1:]),
            "support_spec": "grouped cumulative S1--S5, one group removed per arm",
            "ledger": str(ledger),
            "ledger_sha256": _sha256(ledger),
            "features_json": str(features_json),
            "features_json_sha256": _sha256(features_json),
            "support_oof": str(support_oof),
            "support_oof_sha256": _sha256(support_oof),
            "hurdle_bps": float(hurdle_bps),
            "strict_oof": True,
            "candidate_policy": "one pooled global post-score top-10%; no timestamp/side/asset/portfolio quota",
            "contracts": contracts,
            "outputs_sha256": {
                "logo_oof_predictions.parquet": _sha256(stage / "logo_oof_predictions.parquet"),
                "logo_policy_summary.parquet": _sha256(stage / "logo_policy_summary.parquet"),
            },
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(stage, output)
        return manifest
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--support-oof", type=Path, required=True)
    parser.add_argument("--fold-column", default="oof_fold")
    parser.add_argument("--hurdle-bps", type=float, default=25.0)
    parser.add_argument("--target-arm", default="T3_competing_risk_expected_net")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        ledger=args.ledger,
        features_json=args.features_json,
        support_oof=args.support_oof,
        output=args.output,
        fold_column=args.fold_column,
        hurdle_bps=args.hurdle_bps,
        target_arm=args.target_arm,
    ), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
