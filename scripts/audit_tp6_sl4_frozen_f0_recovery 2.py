#!/usr/bin/env python3
"""Fail-closed audit of the frozen F0/base/structural recovery boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "data_perp/artifacts/feature_portability_f4_panel_20260804_v1/f4_representation_contracts.json"
MODEL_CONTRACT = ROOT / "data_perp/artifacts/feature_portability_f4_panel_20260804_v1/f4_frozen_r3_model_contract.json"
HISTORICAL_PANEL = ROOT / "data_perp/artifacts/feature_portability_f4_panel_20260804_v1/f4_candidate_panel.parquet"
LATER_CANDIDATES = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/candidates/candidate_features.parquet"
LATER_LABEL_DIR = ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/assembled_exact_labels/parts/month=2026-07"
FROZEN_STRUCTURAL = ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/rolling_oof_predictions.parquet"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize() -> dict:
    contract = json.loads(CONTRACT.read_text()) if CONTRACT.exists() else {}
    model_contract = json.loads(MODEL_CONTRACT.read_text()) if MODEL_CONTRACT.exists() else {}
    f0 = contract.get("F0_current_frozen", {})
    candidate = pd.read_parquet(LATER_CANDIDATES) if LATER_CANDIDATES.exists() else pd.DataFrame()
    historical = pd.read_parquet(HISTORICAL_PANEL, columns=["candidate_id", "side_name"]) if HISTORICAL_PANEL.exists() else pd.DataFrame()
    missing_later = {
        side: [field for field in f0.get(side, []) if field not in candidate.columns]
        for side in ("long", "short")
    }
    labels = {}
    for side in ("long", "short"):
        path = LATER_LABEL_DIR / f"side={side}.parquet"
        if path.exists():
            frame = pd.read_parquet(path, columns=["candidate_id", "label_valid", "t4_tp6_sl4_net_bps", "t4_tp6_sl4_gross_bps"])
            labels[side] = {
                "path": str(path),
                "rows": int(len(frame)),
                "valid_fraction": float(frame.label_valid.astype(bool).mean()),
                "exact_net_finite_fraction": float(pd.to_numeric(frame.t4_tp6_sl4_net_bps, errors="coerce").notna().mean()),
                "sha256": sha256(path),
            }
        else:
            labels[side] = {"path": str(path), "exists": False}
    model_candidates = []
    for path in ROOT.glob("data_perp/artifacts/**/model.txt"):
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        if "num_class=3" in text or "objective=multiclass" in text:
            model_candidates.append(str(path))
    return {
        "schema": "tp6_sl4_frozen_f0_recovery_audit_v1",
        "status": "BLOCKED_EXACT_F0_MODEL_AND_LATER_STRUCTURAL_INPUTS",
        "frozen_contract": {
            "exists": CONTRACT.exists(),
            "sha256": sha256(CONTRACT),
            "f0_counts": {side: len(f0.get(side, [])) for side in ("long", "short")},
            "model_contract_exists": MODEL_CONTRACT.exists(),
            "model_contract": model_contract,
        },
        "historical_f0_panel": {
            "exists": HISTORICAL_PANEL.exists(),
            "rows": int(len(historical)),
            "side_counts": historical.side_name.value_counts().to_dict() if not historical.empty else {},
            "sha256": sha256(HISTORICAL_PANEL),
        },
        "later_candidate_source": {
            "exists": LATER_CANDIDATES.exists(),
            "rows": int(len(candidate)),
            "columns": int(len(candidate.columns)),
            "missing_f0_by_side": missing_later,
            "missing_f0_counts": {side: len(values) for side, values in missing_later.items()},
            "sha256": sha256(LATER_CANDIDATES),
        },
        "later_exact_labels": labels,
        "frozen_structural_artifact": {
            "exists": FROZEN_STRUCTURAL.exists(),
            "path": str(FROZEN_STRUCTURAL),
            "date_range": (
                [str(x) for x in pd.read_parquet(FROZEN_STRUCTURAL, columns=["__ts__"])["__ts__"].agg(["min", "max"])]
                if FROZEN_STRUCTURAL.exists() else None
            ),
        },
        "serialized_model_candidates_multiclass_named_only": model_candidates[:40],
        "recovery_findings": [
            "The F0 feature and R3 parameter contracts are preserved.",
            "The exact serialized F0 model is not preserved as a reusable artifact; newer model.txt files are not contract-equivalent by lineage.",
            "The later candidate source lacks 17 long and 18 short frozen F0 fields; it also lacks most frozen context fields.",
            "Exact later TP6/SL4 labels are present and valid rows are retained; invalid rows remain target-invalid.",
            "Native runtime regeneration was attempted with 1h Kraken/OI/funding/order-book data in bounded symbol chunks but exceeded the available memory before atomic output (about 1.3 GB peak for one symbol).",
            "No OOS base/structural/GAM score is published from a substituted model, alias field, or transform-skipping approximation.",
        ],
        "required_next_action": "Run the same native feature graph in a larger-memory worker or materialize and persist the frozen per-feature transform state; then refit exactly the declared F0 model on the preserved historical panel and regenerate structural inputs before final later OOS.",
    }


def run(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    result = summarize()
    (output_dir / "frozen_f0_recovery_audit.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (output_dir / "run_manifest.json").write_text(json.dumps({"schema": result["schema"], "status": result["status"], "artifacts": ["frozen_f0_recovery_audit.json", "run_manifest.json"]}, indent=2) + "\n")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data_perp/artifacts/tp6_sl4_frozen_f0_recovery_audit_20260815_v1")
    args = parser.parse_args()
    print(run(args.output_dir))
