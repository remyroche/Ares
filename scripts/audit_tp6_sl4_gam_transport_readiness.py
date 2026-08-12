#!/usr/bin/env python3
"""Audit whether an untouched chronological population can accept the frozen GAM contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_gam_transport_readiness_20260815_v1"

REQUIRED = {
    "candidate_id", "__ts__", "side_name", "base_score", "base_expected_bps",
    "exact_net_bps", "exact_gross_bps", "archetype_matched_mass",
    "archetype_unmatched_mass", "rolling_transport_valid", "gam_delta_bps",
}

# These fields are deliberately not interchangeable.  In particular, a
# generic execution-net label or a Pack-B/base-adapter score is not a valid
# substitute for the frozen TP6/SL4 structural contract.
EXACT_LABEL_FIELDS = {"exact_net_bps", "exact_gross_bps"}
KNOWN_NON_EQUIVALENT_LABEL_FIELDS = {
    "execution_net_ev_12h",
    "execution_gross_ev_12h",
    "q25_net_bps",
    "q50_net_bps",
}

CANDIDATES = (
    ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/assembled_exact_labels/parts/month=2026-07/side=long.parquet",
    ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/assembled_exact_labels/parts/month=2026-07/side=short.parquet",
    ROOT / "data_perp/artifacts/cross_era_direct_net_transfer_adapter_current_audit_20260730_v2/current_scored_exact.parquet",
    ROOT / "data_perp/artifacts/execution_ev_context_field_cutoff_20260726_v1/joined_execution_ev_model_ablation_oof.parquet",
    ROOT / "data_perp/artifacts/july20_23_retrospective_allscore_bridge_20260730_v1/retrospective_allscore_bridge.parquet",
    ROOT / "data_perp/artifacts/july_exact_preentry_head_audit_20260730_v2/exact_head_audit_ledger.parquet",
)

FROZEN_STRUCTURAL = ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/rolling_oof_predictions.parquet"


def _summarize(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    frame = pd.read_parquet(path)
    columns = set(map(str, frame.columns))
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce") if "__ts__" in frame else pd.Series(dtype="datetime64[ns, UTC]")
    canonical_exact_labels = {
        "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
        "tp6_sl4_entry_price", "atr_1h", "label_valid", "target_invalid",
    } <= columns
    exact_labels = (EXACT_LABEL_FIELDS <= columns) or canonical_exact_labels
    non_equivalent_labels = sorted(KNOWN_NON_EQUIVALENT_LABEL_FIELDS & columns)
    frozen_contract = REQUIRED <= columns and exact_labels
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "columns": int(len(columns)),
        "date_min": str(ts.min()) if len(ts) else None,
        "date_max": str(ts.max()) if len(ts) else None,
        "side_values": sorted(frame.side_name.astype(str).unique().tolist()) if "side_name" in frame else [],
        "required_present": sorted(REQUIRED & columns),
        "required_missing": sorted(REQUIRED - columns),
        "has_exact_tp6_sl4_labels": bool(exact_labels),
        "label_contract": (
            "canonical_tp6_sl4_h12_sidecar" if canonical_exact_labels
            else ("exact_net_bps/exact_gross_bps" if EXACT_LABEL_FIELDS <= columns else None)
        ),
        "label_valid_fraction": (
            float(frame["label_valid"].astype(bool).mean())
            if "label_valid" in frame and len(frame) else None
        ),
        "non_equivalent_label_fields": non_equivalent_labels,
        "has_frozen_gam_fields": bool(frozen_contract),
        "contract_rejection_reasons": [
            *(["missing_required_fields"] if not (REQUIRED <= columns) else []),
            *(["not_exact_tp6_sl4_labels"] if not exact_labels else []),
        ],
    }


def run(*, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = _summarize(FROZEN_STRUCTURAL)
    candidates = [_summarize(p) for p in CANDIDATES]
    usable = [x for x in candidates if x.get("has_frozen_gam_fields")]
    result = {
        "schema": "tp6_sl4_gam_transport_readiness_v1",
        "status": "READY" if usable else "BLOCKED_MISSING_FROZEN_STRUCTURAL_INPUTS",
        "frozen_development_contract": frozen,
        "candidate_populations": candidates,
        "required_fields": sorted(REQUIRED),
        "unchanged_contract_rule": "Do not substitute a different base score, structural leaf contract, or label geometry for untouched OOS.",
        "next_materialization": "Exact TP6/SL4 labels are present for the later candidate population. Generate the frozen base outputs and structural leaf assignments for that same population, then rerun rolling archetype/GAM and the matched residual/meta gate without HPO.",
    }
    (output_dir / "transport_readiness.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(json.dumps({"schema": result["schema"], "status": result["status"], "artifacts": ["transport_readiness.json", "run_manifest.json"]}, indent=2) + "\n")
    print(json.dumps({"output": str(output_dir), "status": result["status"], "usable_candidates": len(usable)}, indent=2))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(output_dir=args.output_dir)
