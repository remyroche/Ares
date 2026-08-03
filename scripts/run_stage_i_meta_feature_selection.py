#!/usr/bin/env python3
"""Run checkpointed side-local Stage-I residual feature selection."""

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
from extreme_price_movements.prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)
from extreme_price_movements.stage_i_feature_selection import (
    StageIHeadContract,
    run_stage_i_head_selection,
)


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
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
    if not ledger[identity].reset_index(drop=True).equals(features[identity].reset_index(drop=True)):
        raise ValueError("selector ledger/features identity order differs")
    raw = features.drop(columns=identity)
    completed = []
    for side in list(dict.fromkeys(args.side or ["long", "short"])):
        destination = args.output_dir / side
        manifest_path = destination / "manifest.json"
        if manifest_path.exists() and args.resume:
            completed.append(json.loads(manifest_path.read_text()))
            continue
        if destination.exists():
            raise FileExistsError(f"meta selection cell exists without --resume: {destination}")
        destination.mkdir(parents=True)
        side_mask = ledger.side_name.eq(side).to_numpy()
        local_ledger = ledger.loc[side_mask].reset_index(drop=True)
        local_frame = raw.loc[side_mask].reset_index(drop=True)
        base = pd.read_parquet(args.base_selection_dir / side / "selector_base_oof.parquet")
        handoff_identity = [*identity, "side_name", "label_available_ts"]
        if (
            any(column not in base.columns for column in handoff_identity)
            or not base.loc[:, handoff_identity].reset_index(drop=True).equals(
                local_ledger.loc[:, handoff_identity].reset_index(drop=True)
            )
        ):
            raise ValueError(f"{side} base OOF differs from selector full same-side identity/order")
        if not base["side_name"].astype(str).str.lower().eq(side).all():
            raise ValueError(f"{side} base OOF contains cross-side rows")
        probability = base[["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].to_numpy(np.float32)
        valid = np.isfinite(probability).all(axis=1)
        if not valid.any():
            raise ValueError(f"{side} base selector emitted no finite R3 OOF handoff rows")
        if (
            (probability[valid] < 0.0).any()
            or not np.allclose(probability[valid].sum(axis=1), 1.0, atol=1e-5)
        ):
            raise ValueError(f"{side} base OOF handoff is not an R3 probability simplex")
        score = probability[:, 2] - probability[:, 0]
        mapped, map_audit, map_provenance = prequential_same_side_r3_value_map(
            exact_net_bps=local_ledger.loc[valid, "exact_net_bps"].to_numpy(np.float32),
            decision_timestamps=local_ledger.loc[valid, "__ts__"],
            label_available_timestamps=local_ledger.loc[valid, "label_available_ts"],
            side=side,
            score=score[valid],
            config=PrequentialR3ValueMapConfig(side=side),
        )
        keep = np.flatnonzero(valid)
        local_ledger = local_ledger.iloc[keep].reset_index(drop=True)
        local_frame = local_frame.iloc[keep].reset_index(drop=True)
        probability = probability[keep]
        mapped = np.asarray(mapped, dtype=np.float32)
        local_frame["r3_p_adverse"] = probability[:, 0]
        local_frame["r3_p_weak"] = probability[:, 1]
        local_frame["r3_p_clear"] = probability[:, 2]
        local_frame["r3_opportunity_score"] = probability[:, 2] - probability[:, 0]
        local_frame["prequential_base_expected_net_bps"] = mapped
        residual_target = local_ledger.exact_net_bps.to_numpy(np.float32) - mapped
        contract = StageIHeadContract("meta", side, "shared_exact_net_residual")
        result = run_stage_i_head_selection(
            local_frame, residual_target,
            contract=contract, cfg=CFG,
            report_root=destination / "mda",
            train_candidate=lgbm_pipeline.train_lgbm_stability_candidate,
            candidate_kwargs={
                "timestamps": local_ledger["__ts__"],
                "label_available_timestamps": local_ledger["label_available_ts"],
                "exact_net_bps": local_ledger["exact_net_bps"].to_numpy(np.float32),
                "exact_net_units": "bps",
                "frozen_base_expected_net_bps": mapped,
                "frozen_base_expected_net_units": "bps",
                "base_oof_provenance": {"side": side, "strict_oof": True, "source": "stage_i_base_selector_forward_burnin"},
                "assets": local_ledger["__symbol__"].astype(str).to_numpy(),
                "sample_weight": np.ones(len(local_ledger), dtype=np.float32),
                "mode": "regression",
                "hpo_objective_mode": "train_meta",
                "reference_artifact_dir": destination / "reference",
                "cfg": {"lgbm_feature_min_coverage": 0.90, "lgbm_joint_complete_case_filter_enabled": False},
            },
        )
        if result is None:
            raise RuntimeError(f"Stage-I meta selection returned no result for {side}")
        prediction = np.asarray(result.get("oof_probs"), dtype=np.float32).reshape(-1)
        if len(prediction) != len(local_ledger):
            raise ValueError(f"{side} meta selector did not emit aligned residual OOF")
        output = local_ledger.loc[:, identity + ["side_name", "label_available_ts", "exact_net_bps"]].copy()
        output["prequential_base_expected_net_bps"] = mapped
        output["residual_oof_bps"] = prediction
        output["reconstructed_expected_net_bps"] = mapped + prediction
        output.to_parquet(destination / "selector_meta_oof.parquet", index=False, compression="zstd")
        map_audit.to_parquet(destination / "prequential_value_map_audit.parquet", index=False, compression="zstd")
        selected = list(map(str, result.get("selected_feature_names", []) or []))
        warmup = selector_sample_manifest.get("causal_warmup_prefix_features", {})
        selected_readiness = {
            feature: dict(warmup[feature])
            for feature in selected if feature in warmup
        }
        payload = {
            "schema": "stage_i_meta_feature_selection_v1", "status": "complete", "side": side,
            "rows": int(len(local_ledger)), "selected_features": selected,
            "selected_feature_contract": _safe(result.get("stage_i_selected_feature_contract", selected)),
            "required_same_side_base_oof_handoff_features": _safe(
                result.get("stage_i_required_same_side_base_oof_handoff_features", [])
            ),
            "selected_feature_count": len(selected), "best_params": _safe(result.get("best_params", {})),
            "metrics": _safe(result.get("metrics", {})), "pruning_history": _safe(result.get("pruning_history", [])),
            "stage_i_prefix_confirmation": _safe(result.get("stage_i_prefix_confirmation", {})),
            "value_map_provenance": _safe(map_provenance),
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
