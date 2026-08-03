#!/usr/bin/env python3
"""Run checkpointed side-local Stage-I base feature selection."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements import lgbm_pipeline
from extreme_price_movements.stage_i_feature_selection import (
    StageIHeadContract,
    run_stage_i_head_selection,
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), action="append", default=[])
    parser.add_argument("--hpo-trials", type=int, default=60)
    parser.add_argument("--hpo-patience", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lgbm_pipeline.LGBM_HPO_TRIALS = max(1, int(args.hpo_trials))
    lgbm_pipeline.LGBM_HPO_EARLY_STOP_PATIENCE = max(1, int(args.hpo_patience))

    ledger = pd.read_parquet(args.selector_dir / "selector_ledger.parquet")
    features = pd.read_parquet(args.selector_dir / "selector_features.parquet")
    selector_sample_manifest_path = args.selector_dir / "manifest.json"
    selector_contract_path = args.selector_dir / "selector_feature_contract.json"
    selector_sample_manifest = json.loads(selector_sample_manifest_path.read_text())
    if selector_sample_manifest.get("status") != "complete":
        raise ValueError("selector sample manifest is incomplete")
    identity = ["candidate_id", "__ts__", "__symbol__"]
    if not ledger.loc[:, identity].reset_index(drop=True).equals(
        features.loc[:, identity].reset_index(drop=True)
    ):
        raise ValueError("selector ledger/features identity order differs")
    matrix = features.drop(columns=identity)
    sides = list(dict.fromkeys(args.side or ["long", "short"]))
    completed = []
    for side in sides:
        destination = args.output_dir / side
        manifest_path = destination / "manifest.json"
        if manifest_path.exists() and args.resume:
            completed.append(json.loads(manifest_path.read_text()))
            continue
        if destination.exists():
            raise FileExistsError(f"selection cell exists without --resume: {destination}")
        destination.mkdir(parents=True)
        mask = ledger.side_name.eq(side).to_numpy()
        local_ledger = ledger.loc[mask].reset_index(drop=True)
        local_frame = matrix.loc[mask].reset_index(drop=True)
        contract = StageIHeadContract("base", side, "R3_economic_simplex_b25")
        result = run_stage_i_head_selection(
            local_frame,
            local_ledger.r3_class.to_numpy(dtype=np.int8),
            contract=contract,
            cfg=CFG,
            report_root=destination / "mda",
            train_candidate=lgbm_pipeline.train_lgbm_stability_candidate,
            candidate_kwargs={
                "timestamps": local_ledger["__ts__"],
                "label_available_timestamps": local_ledger["label_available_ts"],
                "exact_net_bps": local_ledger["exact_net_bps"].to_numpy(dtype=np.float32),
                "exact_net_units": "bps",
                "r3_metric_target": local_ledger["r3_metric_target"].to_numpy(dtype=np.float32),
                "assets": local_ledger["__symbol__"].astype(str).to_numpy(),
                "sample_weight": np.ones(len(local_ledger), dtype=np.float32),
                "mode": "multiclass3",
                "hpo_objective_mode": "train_base",
                "reference_artifact_dir": destination / "reference",
                "cfg": {
                    "lgbm_feature_min_coverage": 0.90,
                    "lgbm_joint_complete_case_filter_enabled": False,
                },
            },
        )
        if result is None:
            raise RuntimeError(f"Stage-I base selection returned no result for {side}")
        probabilities = np.asarray(result.get("oof_probs"), dtype=np.float32)
        if probabilities.ndim != 2 or probabilities.shape != (len(local_ledger), 3):
            raise ValueError(f"{side} base selector did not emit aligned R3 OOF probabilities")
        prediction = local_ledger.loc[:, identity + ["side_name", "label_available_ts", "exact_net_bps"]].copy()
        prediction[["r3_p_adverse", "r3_p_weak", "r3_p_clear"]] = probabilities
        prediction.to_parquet(destination / "selector_base_oof.parquet", index=False, compression="zstd")
        selected = list(map(str, result.get("selected_feature_names", []) or []))
        warmup = selector_sample_manifest.get("causal_warmup_prefix_features", {})
        selected_readiness = {
            feature: dict(warmup[feature])
            for feature in selected if feature in warmup
        }
        payload = {
            "schema": "stage_i_base_feature_selection_v1",
            "status": "complete",
            "side": side,
            "rows": int(len(local_ledger)),
            "input_features": int(result.get("stage_i_input_feature_count", len(local_frame.columns))),
            "selected_features": selected,
            "selected_feature_contract": selected,
            "input_feature_contract": _json_safe(result.get("stage_i_input_features", [])),
            "selected_feature_count": len(selected),
            "best_params": _json_safe(result.get("best_params", {})),
            "metrics": _json_safe(result.get("metrics", {})),
            "pruning_history": _json_safe(result.get("pruning_history", [])),
            "stage_i_prefix_confirmation": _json_safe(result.get("stage_i_prefix_confirmation", {})),
            "hpo_trials": int(args.hpo_trials),
            "hpo_patience": int(args.hpo_patience),
            "selector_sample_manifest_sha256": _file_sha256(selector_sample_manifest_path),
            "selector_feature_contract_sha256": _file_sha256(selector_contract_path),
            "selected_feature_readiness": selected_readiness,
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
        completed.append(payload)
    print(json.dumps({"status": "complete", "cells": completed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
