#!/usr/bin/env python3
"""Run the legacy pre-mapped residual meta selector as a diagnostic control.

This script predates the promotable direct-FQ3 contract.  It consumes a
prequential bps map and therefore must never be used to create an S/O/R3
direct-correctness winner.  The default is deliberately fail-closed; an
explicit control-only flag is required to reproduce historical diagnostics.
Promotable evaluation belongs to ``run_stage_i_target_specific_oos.py``, where
the same-side raw base output enters FQ3 before any causal bps mapping.
"""

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

from extreme_price_movements import lgbm_pipeline
from extreme_price_movements.config import CFG
from extreme_price_movements.prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_CORRELATION_POLICIES,
    STAGE_I_CORRELATION_POLICY_GROUPED_PRESERVE,
    StageIHeadContract,
    run_stage_i_head_selection,
)
from extreme_price_movements.stage_i_mda_support import build_stage_i_mda_training_support
from extreme_price_movements.stage_i_model_hpo import run_stage_i_model_hpo
from extreme_price_movements.stage_i_ranking import RANKING_POLICY, stable_stage_i_topk_positions
from extreme_price_movements.stage_i_target_adapter import (
    FOLD_QUANTILE_RESIDUAL3,
    StageITargetContract,
    bind_target_contract,
    file_sha256,
    generic_base_trust_features,
    load_base_target_winner_bundle,
)


SCHEMA = "stage_i_adapter_meta_feature_selection_v2"
IDENTITY = ["candidate_id", "__ts__", "__symbol__"]


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sha(value: Any) -> str:
    return sha256(json.dumps(_safe(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _winner_support_ledger(target: pd.DataFrame) -> pd.DataFrame:
    output = target.copy()
    event = pd.to_numeric(output.event, errors="raise").to_numpy(np.int8)
    output["r3_class"] = event
    value = pd.to_numeric(output.target_value, errors="raise").to_numpy(np.float32)
    output["robust_clear_soft_b25_t50"] = value if output.target_family.iloc[0] == "scalar_S" else value / 4.0
    output["t2_tp6_sl4_event"] = np.select(
        [event == 2, event == 0, event == 1], [0, 1, 2], default=-1,
    ).astype(np.int8)
    output["exact_net_bps"] = pd.to_numeric(output.net_bps, errors="raise").to_numpy(np.float32)
    output["label_valid"] = output.target_valid.astype(bool)
    return output


def _candidate_positions(ledger: pd.DataFrame, score: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(float(fraction) * len(ledger))))
    return stable_stage_i_topk_positions(
        score, candidate_ids=ledger.candidate_id, side_names=ledger.side_name,
        decision_timestamps=ledger.decision_ts, signal_timestamps=ledger["__ts__"],
        symbols=ledger["__symbol__"], count=count,
    ).astype(np.int32)


def _load_base(
    root: Path, *, side: str, selector_manifest_sha: str, winner_contract: StageITargetContract,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    manifest_path = root / side / "manifest.json"
    oof_path = root / side / "selector_base_oof.parquet"
    if not manifest_path.is_file() or not oof_path.is_file():
        raise ValueError(f"{side}: target-v2 base selector is incomplete")
    manifest = json.loads(manifest_path.read_text())
    if (
        # Target-specific S/O selectors intentionally use the v2 manifest;
        # accepting the legacy R3 v1 schema here would detach the direct meta
        # from its immutable target contract.
        manifest.get("schema") != "stage_i_base_feature_selection_v2"
        or manifest.get("status") != "complete"
        or manifest.get("side") != side
        or manifest.get("target_contract_sha256") != winner_contract.sha256
        or manifest.get("selector_sample_manifest_sha256") != selector_manifest_sha
        or manifest.get("selector_base_oof_sha256") != file_sha256(oof_path)
    ):
        raise ValueError(f"{side}: target-v2 base OOF lineage drift")
    frame = pd.read_parquet(oof_path)
    required = {*IDENTITY, "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "exact_gross_bps", "base_raw_score"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{side}: target-v2 base OOF lacks {missing}")
    if not frame.side_name.astype(str).str.lower().eq(side).all():
        raise ValueError("base OOF is not side-local")
    return frame, manifest, file_sha256(manifest_path)


def _align_by_identity(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    left = pd.MultiIndex.from_frame(source.loc[:, IDENTITY])
    right = pd.MultiIndex.from_frame(target.loc[:, IDENTITY])
    positions = left.get_indexer(right)
    if (positions < 0).any() or len(np.unique(positions)) != len(positions):
        raise ValueError("identity alignment failed")
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--target-winner-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), action="append", default=[])
    parser.add_argument("--base-candidate-fraction", type=float, default=0.30)
    parser.add_argument("--hpo-trials", type=int, default=60)
    parser.add_argument("--hpo-patience", type=int, default=15)
    parser.add_argument(
        "--correlation-policy", choices=sorted(STAGE_I_CORRELATION_POLICIES),
        default=STAGE_I_CORRELATION_POLICY_GROUPED_PRESERVE,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-legacy-premapped-residual-control", action="store_true",
        help="Reproduce the non-promotable mapped-bps residual selector only.",
    )
    args = parser.parse_args()
    if not args.allow_legacy_premapped_residual_control:
        raise ValueError(
            "fail-closed: this selector uses pre-mapped expected-net residuals and is not the "
            "promotable direct FQ3 contract. Use run_stage_i_target_specific_oos.py with a "
            "direct S/O base or immutable frozen R3 OOF handoff; pass the legacy-control flag "
            "only for explicitly labelled historical diagnostics."
        )
    if not 0 < args.base_candidate_fraction <= 1:
        raise ValueError("base candidate fraction must lie in (0,1]")
    selector_manifest_path = args.selector_dir / "manifest.json"
    selector_feature_contract_path = args.selector_dir / "selector_feature_contract.json"
    selector_manifest_sha = file_sha256(selector_manifest_path)
    selector_feature_contract_sha = file_sha256(selector_feature_contract_path)
    selector_ledger = pd.read_parquet(args.selector_dir / "selector_ledger.parquet")
    selector_features = pd.read_parquet(args.selector_dir / "selector_features.parquet")
    if not selector_ledger[IDENTITY].reset_index(drop=True).equals(selector_features[IDENTITY].reset_index(drop=True)):
        raise ValueError("selector ledger/features identity drift")
    raw_matrix = selector_features.drop(columns=IDENTITY)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for side in list(dict.fromkeys(args.side or ["long", "short"])):
        winner, base_contract, winner_audit = load_base_target_winner_bundle(args.target_winner_dir, side=side)
        base, base_manifest, base_manifest_sha = _load_base(
            args.base_selection_dir, side=side,
            selector_manifest_sha=selector_manifest_sha, winner_contract=base_contract,
        )
        destination = args.output_dir / side
        request = {
            "schema": SCHEMA, "side": side,
            "selector_manifest_sha256": selector_manifest_sha,
            "selector_feature_contract_sha256": selector_feature_contract_sha,
            "base_selector_manifest_sha256": base_manifest_sha,
            "base_target_contract_sha256": base_contract.sha256,
            "base_candidate_fraction": float(args.base_candidate_fraction),
            "target_family": FOLD_QUANTILE_RESIDUAL3,
            "correlation_policy": args.correlation_policy,
            "hpo_trials": int(args.hpo_trials), "hpo_patience": int(args.hpo_patience),
        }
        request_sha = _sha(request)
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file() and args.resume:
            prior = json.loads(manifest_path.read_text())
            inventory = prior.get("artifact_sha256", {})
            if prior.get("request_sha256") != request_sha or prior.get("status") != "complete":
                raise ValueError(f"{side}: meta selector resume request drift")
            for relative, expected in inventory.items():
                if file_sha256(destination / relative) != expected:
                    raise ValueError(f"{side}: meta selector resume artifact drift: {relative}")
            summaries.append(prior)
            continue
        if destination.exists():
            raise FileExistsError(f"meta adapter selection exists without --resume: {destination}")
        destination.mkdir(parents=True)

        # The target winner is the authoritative economic/validity population.
        base_positions = _align_by_identity(base, winner)
        base = base.iloc[base_positions].reset_index(drop=True)
        if not np.allclose(base.exact_net_bps, winner.net_bps, atol=1e-5) or not np.allclose(base.exact_gross_bps, winner.gross_bps, atol=1e-5):
            raise ValueError("base OOF economics are not the winner geometry")
        source_side = selector_ledger.side_name.astype(str).str.lower().eq(side)
        side_ledger = selector_ledger.loc[source_side].reset_index(drop=True)
        side_raw = raw_matrix.loc[source_side].reset_index(drop=True)
        selector_positions = _align_by_identity(side_ledger, winner)
        raw = side_raw.iloc[selector_positions].reset_index(drop=True)
        ledger = _winner_support_ledger(winner)
        ledger["decision_ts"] = pd.to_datetime(ledger.decision_ts, utc=True, errors="raise")
        ledger["label_available_ts"] = pd.to_datetime(ledger.label_available_ts, utc=True, errors="raise")

        score = pd.to_numeric(base.base_raw_score, errors="coerce").to_numpy(np.float32)
        finite = np.isfinite(score)
        ledger, raw, base, score = (
            ledger.loc[finite].reset_index(drop=True), raw.loc[finite].reset_index(drop=True),
            base.loc[finite].reset_index(drop=True), score[finite],
        )
        mapper_score = (2.0 * score - 1.0).astype(np.float32)
        mapped, map_audit, map_provenance = prequential_same_side_r3_value_map(
            exact_net_bps=ledger.exact_net_bps.to_numpy(np.float32),
            decision_timestamps=ledger.decision_ts,
            label_available_timestamps=ledger.label_available_ts,
            side=side, score=mapper_score,
            config=PrequentialR3ValueMapConfig(side=side),
        )
        state_columns = sorted(
            [column for column in base.columns if str(column).startswith("base_state_p")],
            key=lambda value: int(str(value).rsplit("p", 1)[1]),
        )
        simplex = base.loc[:, state_columns].to_numpy(np.float32) if state_columns else None
        raw["base_raw_score"] = score
        raw["prequential_base_expected_net_bps"] = mapped
        for column in state_columns:
            raw[column] = base[column].to_numpy(np.float32)
        trust = generic_base_trust_features(score, simplex, map_audit)
        for column in trust:
            raw[column] = trust[column].to_numpy(np.float32)
        handoff_features = tuple([
            "base_raw_score", "prequential_base_expected_net_bps", *state_columns, *trust.columns,
        ])
        residual = ledger.exact_net_bps.to_numpy(np.float32) - np.asarray(mapped, dtype=np.float32)
        candidate_positions = _candidate_positions(ledger, score, args.base_candidate_fraction)
        candidate_mask = np.zeros(len(ledger), dtype=bool)
        candidate_mask[candidate_positions] = True

        support = build_stage_i_mda_training_support(
            ledger, side=side, identity_columns=IDENTITY,
            decision_timestamps=ledger.decision_ts,
        )
        mda_reference = {
            "source": "full_valid_winner_geometry_base_oof_reference",
            "side": side, "X": raw, "target": residual, "metric_target": residual,
            "sample_weight": np.ones(len(ledger), np.float32),
            "timestamps": ledger.decision_ts,
            "label_available_timestamps": ledger.label_available_ts,
            "exact_net_bps": ledger.exact_net_bps.to_numpy(np.float32),
            "prediction_offset": np.asarray(mapped, np.float32),
            "assets": ledger["__symbol__"].astype(str).to_numpy(),
            "identity": ledger.loc[:, [*IDENTITY, "decision_ts"]],
            "archetype_labels": support["archetype_labels"],
            "archetype_label_audit": support["audit"],
        }
        candidate_ledger = ledger.iloc[candidate_positions].reset_index(drop=True)
        candidate_frame = raw.iloc[candidate_positions].reset_index(drop=True)
        candidate_residual = residual[candidate_positions]
        candidate_mapped = np.asarray(mapped, np.float32)[candidate_positions]
        candidate_support = build_stage_i_mda_training_support(
            candidate_ledger, side=side, identity_columns=IDENTITY,
            decision_timestamps=candidate_ledger.decision_ts,
        )
        contract_frame = candidate_ledger.loc[:, [*IDENTITY, "side_name"]].copy()
        contract_frame["meta_residual_basis_bps"] = candidate_residual
        contract_frame["gross_bps"] = candidate_ledger.gross_bps.to_numpy(np.float32)
        contract_frame["net_bps"] = candidate_ledger.net_bps.to_numpy(np.float32)
        contract_frame["target_valid"] = True
        contract_frame["sample_weight"] = 1.0
        meta_contract = bind_target_contract(
            contract_frame, family=FOLD_QUANTILE_RESIDUAL3, layer="meta",
            target_name=FOLD_QUANTILE_RESIDUAL3, geometry=base_contract.geometry,
            target_columns=("meta_residual_basis_bps",),
            metadata={
                "base_target_contract_sha256": base_contract.sha256,
                "fit_semantics": "fold_local_q33_q67_q05_q95_support50_clip200",
                "semantic_gate": "q33<0<=q67",
                "training_stream": "base_candidate_only",
                "mapping_reference": "full_valid_winner_geometry_population",
            },
        )
        contract = StageIHeadContract("meta", side, FOLD_QUANTILE_RESIDUAL3)
        result = run_stage_i_head_selection(
            candidate_frame, candidate_residual,
            contract=contract, cfg=CFG, report_root=destination / "mda",
            train_candidate=lgbm_pipeline.train_lgbm_stability_candidate,
            candidate_kwargs={
                "timestamps": candidate_ledger.decision_ts,
                "label_available_timestamps": candidate_ledger.label_available_ts,
                "exact_net_bps": candidate_ledger.exact_net_bps.to_numpy(np.float32),
                "exact_net_units": "bps",
                "frozen_base_expected_net_bps": candidate_mapped,
                "frozen_base_expected_net_units": "bps",
                "base_oof_provenance": {
                    "side": side, "strict_oof": True,
                    "source": "target_v2_base_selector",
                    "base_target_contract_sha256": base_contract.sha256,
                },
                "assets": candidate_ledger["__symbol__"].astype(str).to_numpy(),
                "label_context": candidate_support["label_context"],
                "stage_i_declared_single_side_scope": side,
                "sample_weight": np.ones(len(candidate_ledger), np.float32),
                "mode": "regressor", "hpo_objective_mode": "train_meta",
                "reference_artifact_dir": destination / "reference",
                "mda_reference": mda_reference,
                "cfg": {
                    "lgbm_feature_min_coverage": 0.90,
                    "lgbm_feature_coverage_scope": "all_post_warmup",
                    "lgbm_joint_complete_case_filter_enabled": False,
                    "stage_i_exact_readiness_coverage_prevalidated": False,
                },
            },
            correlation_policy=args.correlation_policy,
            target_contract=meta_contract,
            required_base_handoff_features=handoff_features,
        )
        if result is None:
            raise RuntimeError(f"{side}: meta adapter selector returned no result")
        selected = tuple(map(str, result.get("selected_feature_names", ())))
        hpo = run_stage_i_model_hpo(
            candidate_frame, candidate_residual,
            selected_feature_names=selected,
            candidate_ids=candidate_ledger.candidate_id,
            exact_net_bps=candidate_ledger.exact_net_bps,
            decision_timestamps=candidate_ledger.decision_ts,
            label_available_timestamps=candidate_ledger.label_available_ts,
            side=side, layer="meta", target_contract=meta_contract,
            prediction_offset_bps=candidate_mapped,
            sample_weight=np.ones(len(candidate_ledger), np.float32),
            hpo_trials=args.hpo_trials, hpo_patience=args.hpo_patience,
            successive_halving_checkpoint_dir=destination / "_hpo_halving",
        )
        probabilities = np.asarray(hpo.oof_probabilities, dtype=np.float32)
        if probabilities.shape != (len(candidate_ledger), 3):
            raise ValueError("meta HPO did not emit the three-class OOF simplex")
        correction = np.asarray(hpo.oof_score, dtype=np.float32)
        output = candidate_ledger.loc[:, [*IDENTITY, "side_name", "decision_ts", "label_available_ts", "gross_bps", "net_bps"]].copy()
        output["base_raw_score"] = score[candidate_positions]
        output["prequential_base_expected_net_bps"] = candidate_mapped
        output[["meta_p_overestimating", "meta_p_approximately_right", "meta_p_underestimating"]] = probabilities
        output["meta_correction_bps"] = correction
        output["reconstructed_expected_net_bps"] = candidate_mapped + correction
        output["candidate_selected"] = True
        oof_path = destination / "selector_meta_oof.parquet"
        map_path = destination / "prequential_value_map_audit.parquet"
        gate_path = destination / "base_candidate_handoff.parquet"
        output.to_parquet(oof_path, index=False, compression="zstd")
        map_audit.to_parquet(map_path, index=False, compression="zstd")
        ledger.loc[:, [*IDENTITY, "side_name", "decision_ts"]].assign(
            base_raw_score=score, selected_base_candidate=candidate_mask,
            ranking_policy=RANKING_POLICY,
        ).to_parquet(gate_path, index=False, compression="zstd")
        payload = {
            **request, "request_sha256": request_sha, "status": "complete",
            "rows": len(candidate_ledger), "full_mapping_reference_rows": len(ledger),
            "selected_features": list(selected), "selected_feature_contract": list(selected),
            "selected_feature_count": len(selected), "best_params": _safe(hpo.best_params),
            "required_same_side_base_oof_handoff_features": list(handoff_features),
            "target_contract": meta_contract.to_dict(),
            "target_contract_sha256": meta_contract.sha256,
            "base_target_contract": base_contract.to_dict(),
            "target_winner_bundle": winner_audit,
            "base_selector_manifest_sha256": base_manifest_sha,
            "value_map_provenance": _safe(map_provenance),
            "hpo_target_family": hpo.target_family,
            "hpo_actual_trials": hpo.actual_trials,
            "hpo_completed_trials": hpo.completed_trials,
            "hpo_best_metrics": _safe(hpo.best_metrics),
            "hpo_fold_audit": _safe(hpo.fold_audit),
            "hpo_oof_fold_audit": _safe(hpo.oof_fold_audit),
            "meta_training_stream": "candidate_only",
            "mapping_reference_stream": "full_valid_rows_reference_only_for_noncandidates",
            "reconstruction": "fixed_mapped_base_ev + clip((p-prior)@locations,+/-200bps)",
            "artifact_sha256": {
                oof_path.name: file_sha256(oof_path), map_path.name: file_sha256(map_path),
                gate_path.name: file_sha256(gate_path),
            },
        }
        manifest_path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
        summaries.append(payload)
    print(json.dumps({
        "status": "complete",
        "cells": [{"side": item["side"], "rows": item["rows"], "selected_feature_count": item["selected_feature_count"]} for item in summaries],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
