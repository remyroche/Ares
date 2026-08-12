#!/usr/bin/env python3
"""Run the fixed-contract Stage-I nested LightGBM diagnostic when explicitly invoked.

This is intentionally an execution entry point, not an auto-run job.  It reads
the corrected selector sample plus completed per-side base selections, builds
the diagnostic ladders, and writes a new immutable result directory.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec
from extreme_price_movements.stage_i_nested_feature_challenger import NESTED_SET_NAMES, checkpoint_nested_feature_plan, load_completed_stage_i_base_selection, materialize_nested_feature_challenge
from extreme_price_movements.stage_i_nested_lgbm_hooks import FixedLGBMContract, fixed_lgbm_base_predictor, fixed_lgbm_meta_predictor, fold_local_meta_feature_selector, require_side_meta_params, resolve_side_meta_context_universe
from extreme_price_movements.stage_i_nested_stack_execution import GuardedMetaArmSpec, NestedStackConfig, NestedStackInput, execute_matched_nested_stack, pooled_global_causal_tail


IDENTITY = ("candidate_id", "__ts__", "__symbol__")
DIRECT_META = {"r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score", "base_r3_max_probability", "base_r3_top2_margin", "base_r3_entropy"}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_side_input(selector_dir: Path, *, side: str, base_features: tuple[str, ...]) -> NestedStackInput:
    ledger_path, matrix_path = selector_dir / "selector_ledger.parquet", selector_dir / "selector_features.parquet"
    manifest_path, contract_path = selector_dir / "manifest.json", selector_dir / "selector_feature_contract.json"
    if not all(path.is_file() for path in (ledger_path, matrix_path, manifest_path, contract_path)):
        raise ValueError("corrected selector ledger/matrix/contract inputs are incomplete")
    manifest = _json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("selector sample is not complete")
    ledger, matrix = pd.read_parquet(ledger_path), pd.read_parquet(matrix_path)
    if not ledger.loc[:, list(IDENTITY)].reset_index(drop=True).equals(matrix.loc[:, list(IDENTITY)].reset_index(drop=True)):
        raise ValueError("corrected selector ledger and matrix identity order differs")
    local_ledger = ledger.loc[ledger.side_name.astype(str).str.lower().eq(side)].reset_index(drop=True)
    local_matrix = matrix.loc[ledger.side_name.astype(str).str.lower().eq(side)].reset_index(drop=True).drop(columns=list(IDENTITY))
    if not set(base_features).issubset(local_matrix.columns):
        raise ValueError(f"{side}: nested base plan features are absent from corrected selector matrix")
    decision = pd.to_datetime(local_ledger["__ts__"], utc=True, errors="raise") + pd.Timedelta(hours=1)
    availability = pd.to_datetime(local_ledger["label_available_ts"], utc=True, errors="raise")
    if not availability.eq(decision + pd.Timedelta(hours=12)).all():
        raise ValueError(f"{side}: nested execution requires signal+1h decision and +12h exact availability")
    raw = pd.concat([local_ledger.loc[:, list(IDENTITY) + ["side_name", "r3_class", "exact_net_bps"]], local_matrix], axis=1)
    raw["decision_ts"], raw["label_available_ts"] = decision, availability
    context, provenance = resolve_side_meta_context_universe(CFG, side=side, available_columns=list(local_matrix.columns), direct_columns=DIRECT_META)
    return NestedStackInput(side=side, frame=raw, base_feature_universe=tuple(local_matrix.columns), meta_context_features=context, meta_universe_provenance=provenance)


def _write_side(root: Path, result) -> None:
    root.mkdir(parents=True, exist_ok=False)
    result.metrics.to_parquet(root / "metrics.parquet", index=False)
    for name, frame in result.base_outputs.items():
        frame.to_parquet(root / f"base_{name}.parquet", index=False)
    for (name, arm), output in result.meta_outputs.items():
        output.frame.to_parquet(root / f"meta_{name}__{arm}.parquet", index=False)
        (root / f"meta_{name}__{arm}_fold_provenance.json").write_text(output.fold_provenance.to_json(orient="records", indent=2, date_format="iso"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", required=True, type=Path)
    parser.add_argument("--base-selection-dir", required=True, type=Path)
    parser.add_argument("--meta-params-json", required=True, type=Path, help="Fixed LightGBM meta parameter JSON object.")
    parser.add_argument("--meta-arms-json", required=True, type=Path, help="Predeclared guarded target-arm JSON array.")
    parser.add_argument("--required-protected-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-validation-folds", type=int, default=4)
    parser.add_argument("--min-base-train-rows", type=int, default=500)
    parser.add_argument("--min-meta-train-rows", type=int, default=500)
    parser.add_argument("--base-candidate-fraction", type=float, default=0.30)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite nested diagnostic output: {args.output_dir}")
    policy = _json(args.required_protected_json)
    meta_params_raw, arm_data = _json(args.meta_params_json), _json(args.meta_arms_json)
    if not isinstance(policy, dict) or not isinstance(meta_params_raw, dict) or not isinstance(arm_data, list):
        raise ValueError("nested execution JSON contracts are malformed")
    meta_params_by_side = require_side_meta_params(meta_params_raw)
    arms = tuple(GuardedMetaArmSpec(**item) for item in arm_data)
    if not arms:
        raise ValueError("at least one target-specific guarded meta arm is required")
    args.output_dir.mkdir(parents=True)
    results, side_inputs = {}, {}
    for side in ("long", "short"):
        source = load_completed_stage_i_base_selection(args.base_selection_dir / side, side=side)
        plan = materialize_nested_feature_challenge(source, required_features=policy.get("required_features", ()), protected_features=policy.get("protected_features", ()))
        checkpoint_nested_feature_plan(plan, args.output_dir / side / "plan")
        manifest = _json(args.base_selection_dir / side / "manifest.json")
        if (
            manifest.get("schema") != "stage_i_base_feature_selection_v1"
            or manifest.get("status") != "complete"
            or str(manifest.get("side", "")).lower() != side
            or manifest.get("selector_sample_manifest_sha256") != _sha(args.selector_dir / "manifest.json")
            or manifest.get("selector_feature_contract_sha256") != _sha(args.selector_dir / "selector_feature_contract.json")
            or not isinstance(manifest.get("best_params"), dict)
        ):
            raise ValueError(f"{side}: corrected base manifest does not bind to this selector ledger/contract")
        contract = FixedLGBMContract(base_params=manifest["best_params"], meta_params=meta_params_by_side[side])
        input_data = load_side_input(args.selector_dir, side=side, base_features=tuple(source.input_features))
        result = execute_matched_nested_stack(input_data, plan, base_predictor=fixed_lgbm_base_predictor(contract), meta_predictor=fixed_lgbm_meta_predictor(contract), meta_arms=arms, meta_feature_selector=fold_local_meta_feature_selector(contract), config=NestedStackConfig(n_validation_folds=args.n_validation_folds, min_base_train_rows=args.min_base_train_rows, min_meta_train_rows=args.min_meta_train_rows, base_candidate_fraction=args.base_candidate_fraction))
        _write_side(args.output_dir / side / "execution", result)
        results[side] = result
        side_inputs[side] = input_data
    pooled_root = args.output_dir / "pooled_global"
    pooled_root.mkdir()
    for feature_set in NESTED_SET_NAMES:
        for arm_id in ("direct_r3_base", *(arm.arm_id for arm in arms)):
            pooled_global_causal_tail(results["long"], results["short"], feature_set=feature_set, arm_id=arm_id, admission_spec=Causal21dAdmissionSpec()).to_parquet(pooled_root / f"{feature_set}__{arm_id}.parquet", index=False)
    side_contracts = {
        side: {
            "base_selection_manifest_sha256": _sha(args.base_selection_dir / side / "manifest.json"),
            "base_hpo_params_sha256": sha256(json.dumps(_json(args.base_selection_dir / side / "manifest.json")["best_params"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "meta_hpo_params_sha256": sha256(json.dumps(meta_params_by_side[side], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "meta_hpo_params_scope": f"meta/{side}/target_specific_fixed_contract",
            "declared_meta_universe": dict(side_inputs[side].meta_universe_provenance),
        }
        for side in ("long", "short")
    }
    (args.output_dir / "manifest.json").write_text(json.dumps({"schema": "stage_i_nested_lgbm_execution_v1", "status": "complete", "selector_manifest_sha256": _sha(args.selector_dir / "manifest.json"), "selector_feature_contract_sha256": _sha(args.selector_dir / "selector_feature_contract.json"), "meta_params_contract_sha256": _sha(args.meta_params_json), "meta_arms_sha256": _sha(args.meta_arms_json), "side_contracts": side_contracts, "sides": ["long", "short"], "nested_sets": list(NESTED_SET_NAMES), "base_candidate_fraction": float(args.base_candidate_fraction), "base_candidate_scope": "side_local_global_within_chronological_fold_or_prior_oof_pool; never_per_timestamp", "full_input_control_promotion_policy": "eligible_only_if_best_under_identical_strict_OOF_and_OOS_gates; no_post_test_tuning", "causal_mapping_scope": "post_oof_side_21d_then_pooled_global_only"}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
