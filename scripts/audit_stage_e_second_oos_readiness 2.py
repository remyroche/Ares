#!/usr/bin/env python3
"""Fail-closed Stage-E E5 readiness audit for the frozen v9 scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
V9 = ART / "stage_d_compact_action_model_20260731_v9"
V10 = ART / "stage_d_compact_action_model_20260731_v10"
JAN_PATHS = ART / "january2025_native_first_touch_full_12h_paths_20260729_v1"
FEBAPR_PATHS = ART / "febapr2025_native_first_touch_full_12h_paths_20260729_v2"
FEBAPR_CONTEXT = ART / "febapr2025_historical_path_head_context_20260727_v1"
DEFAULT_OUTPUT = ART / "stage_e_second_oos_readiness_20260731_v1"
MODEL_SUFFIXES = {".txt", ".json", ".pkl", ".pickle", ".joblib", ".bin", ".model", ".ubj", ".onnx"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def model_candidates(root: Path) -> list[str]:
    excluded = {"run_manifest.json", "stage_d_compact_feature_manifest.json", "stage_d_action_research_gate.json", "stage_d_action_replay_bootstrap.json"}
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name not in excluded and p.suffix.lower() in MODEL_SUFFIXES
    )


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    v9_manifest = json.loads((V9 / "run_manifest.json").read_text())
    v10_manifest = json.loads((V10 / "run_manifest.json").read_text())
    feature_manifest = json.loads((V9 / "stage_d_compact_feature_manifest.json").read_text())
    if v9_manifest["outputs_sha256"] != v10_manifest["outputs_sha256"]:
        raise ValueError("v9/v10 are not byte-identical companions")
    selected = sorted({name for side in feature_manifest["frozen_before_final_oos"] for name in side["selected_features"]})
    context_shards = sorted((FEBAPR_CONTEXT / "shards").glob("*.parquet"))
    context_columns = set(pq.read_schema(context_shards[0]).names) if context_shards else set()
    dynamic = {"known_row_cost_bps", "gross_return_at_action_bps", "estimated_spread_bps", "time_to_clear_minutes", "entry_price_log"}
    static_selected = sorted(set(selected) - dynamic)
    available_static = sorted(set(static_selected) & context_columns)
    missing_static = sorted(set(static_selected) - context_columns)
    v9_models = model_candidates(V9)
    v10_models = model_candidates(V10)
    frozen_model_available = bool(v9_models and v9_models == v10_models)
    status = "READY" if frozen_model_available else "NOT_RUN_FROZEN_MODEL_ARTIFACT_UNAVAILABLE"
    reason = None if frozen_model_available else (
        "The sealed v9/v10 packs contain predictions, preprocessing, feature lists and calibrators but no serialized "
        "LightGBM booster/tree structure. Recreating it from historical rows would be retraining, prohibited by E5."
    )
    manifest = {
        "schema": "stage_e_second_oos_manifest_v1",
        "status": status,
        "selected_period": "2025-01-01..2025-04-30",
        "selection_reason": "earliest later exact-1m 12h path material already present after consumed 2024-08..2024-11",
        "results_opened": False,
        "model_refit": False,
        "frozen_model_available": frozen_model_available,
        "blocking_reason": reason,
        "canonical_model": "stage_d_compact_action_model_20260731_v9",
        "reproducibility_companion": "stage_d_compact_action_model_20260731_v10",
        "serialized_model_candidates_v9": v9_models,
        "serialized_model_candidates_v10": v10_models,
        "selected_feature_count": len(selected),
        "later_context_static_feature_coverage": {
            "required": len(static_selected), "available": len(available_static),
            "available_features": available_static, "missing_features": missing_static,
        },
        "candidate_sources": {
            str(JAN_PATHS / "manifest.json"): sha(JAN_PATHS / "manifest.json"),
            str(FEBAPR_PATHS / "manifest.json"): sha(FEBAPR_PATHS / "manifest.json"),
            str(FEBAPR_CONTEXT / "manifest.json"): sha(FEBAPR_CONTEXT / "manifest.json"),
        },
        "frozen_inputs": {
            "v9_run_manifest": sha(V9 / "run_manifest.json"),
            "v10_run_manifest": sha(V10 / "run_manifest.json"),
            "feature_manifest": sha(V9 / "stage_d_compact_feature_manifest.json"),
            "margin_bps": v9_manifest["development_selected_margin_bps"],
            "runner_sha256": v9_manifest["runner_sha256"],
        },
        "acceptance_gates": {
            "positive_uplift_vs_continue": True, "positive_uplift_vs_exit": True,
            "both_sides_non_negative": True, "latest_non_negative": True,
            "months_positive_fraction_min": 0.75, "calibration_slope_range": [0.5, 1.5],
            "calibration_intercept_abs_max_bps": 75.0, "each_action_rate_min": 0.02,
            "paired_day_probability_positive_min": 0.95,
        },
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        dump(stage / "stage_e_second_oos_manifest.json", manifest)
        pd.DataFrame(columns=["status", "candidate_id", "side", "month", "predicted_delta_bps", "policy_net_bps"]).to_parquet(
            stage / "stage_e_second_oos_results.parquet", index=False, compression="zstd"
        )
        pd.DataFrame(columns=["status", "replicate", "uplift_vs_continue_bps", "uplift_vs_exit_bps"]).to_parquet(
            stage / "stage_e_second_oos_bootstrap.parquet", index=False, compression="zstd"
        )
        outputs = {p.name: sha(p) for p in stage.iterdir()}
        dump(stage / "run_manifest.json", {
            "schema": "stage_e_second_oos_readiness_run_v1", "status": status,
            "inputs": manifest["frozen_inputs"] | manifest["candidate_sources"],
            "code_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
            "outputs_sha256": outputs, "limitations": [reason] if reason else [],
        })
        (stage / "manifest.sha256").write_text(f"{sha(stage / 'run_manifest.json')}  run_manifest.json\n")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2, default=str))
