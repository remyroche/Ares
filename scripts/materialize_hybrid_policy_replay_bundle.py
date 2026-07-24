#!/usr/bin/env python3
"""Materialize a read-only packaged bundle for hybrid historical replays.

The hybrid base/meta research run stores fold models separately from the live
artifact layout.  This adapter preserves their exact contracts so the common
historical backfill runner can score a forward period without substituting a
different raw candidate ledger.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import joblib


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    base_dir = args.hybrid_root / "models" / "2026-06-30_2026-07-30"
    meta_dir = (
        args.hybrid_root
        / "meta_packb_sideheads_hpo150_wf30"
        / "best_full_oos"
        / "models"
        / "2026-07-01_2026-07-31"
    )
    expert_path = (
        args.hybrid_root
        / "meta_side_residual_correcttarget_hpo150_wf30"
        / "final_side_residual_expert.joblib"
    )
    ae_state = Path(
        _read_json(args.hybrid_root / "meta_handoff_top30" / "manifest.json")
        ["frozen_ae_gmm_context_contract"]["state_path"]
    )
    required = [
        base_dir / "base_model.joblib",
        base_dir / "columns.json",
        meta_dir / "base_soft_label_long.joblib",
        meta_dir / "base_soft_label_short.joblib",
        meta_dir / "columns.json",
        expert_path,
        ae_state,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing hybrid replay inputs: " + ", ".join(missing))

    args.output_root.mkdir(parents=True, exist_ok=True)
    policy_dir = args.output_root / "policy_params"
    policy_dir.mkdir(exist_ok=True)
    for name in (
        "v9_tail95_predecessor_bundle.joblib",
        "residual_event_state.joblib",
        "composite_policy_regime_ev_calibration.json",
        "threshold_basis_policy_sidearch_ev70_trim10_21d.json",
        "threshold_basis_reference_sidearch_ev70_trim10_21d.parquet",
        "hit_surprise_archetype_portfolio_policy.json",
        # The packaged meta scorer rebuilds source tags and reliability fields
        # before it applies the hybrid residual reference.  Keep this frozen
        # policy contract alongside the V9/admission artifacts so a replay
        # cannot silently fall back to a different live feature path.
        "meta_reliability_priors.json",
    ):
        _link(args.policy_root / name, policy_dir / name)
    _link(expert_path, policy_dir / "side_residual_expert.joblib")

    policy_config = _read_json(args.policy_root / "optimized_portfolio_policy_config.json")
    policy_config["side_residual_expert_enabled"] = True
    policy_config["side_residual_expert_artifact_path"] = str(
        (policy_dir / "side_residual_expert.joblib").resolve()
    )
    policy_config["mlp_postprocessor_enabled"] = False
    policy_config["threshold_basis_policy_path"] = str(
        (policy_dir / "threshold_basis_policy_sidearch_ev70_trim10_21d.json").resolve()
    )
    (policy_dir / "optimized_portfolio_policy_config.json").write_text(
        json.dumps(policy_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    base_models = joblib.load(base_dir / "base_model.joblib")
    base_columns = _read_json(base_dir / "columns.json")
    meta_columns = _read_json(meta_dir / "columns.json")
    meta_models = {
        "long_s52_meta_threshold_handoff": joblib.load(
            meta_dir / "base_soft_label_long.joblib"
        ),
        "short_s52_meta_threshold_handoff": joblib.load(
            meta_dir / "base_soft_label_short.joblib"
        ),
    }
    alpha_models = {
        f"{side}_s52_meta_threshold_handoff": {
            "model": base_models[side],
            "feat_cols": base_columns["feature_names_by_side"][side],
        }
        for side in ("long", "short")
    }
    trained_state = {"bundle": {"alpha_models": alpha_models, "meta_models": meta_models}}
    model_dir = args.output_root / "models"
    model_dir.mkdir(exist_ok=True)
    with (model_dir / "trained_state.pkl").open("wb") as handle:
        pickle.dump(trained_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _link(ae_state, args.output_root / "ae_gmm_state" / "ae_gmm_state.pkl")

    feature_contract = {
        "meta_models": {
            f"{side}_s52_meta_threshold_handoff": {
                "feature_columns": meta_columns.get("feature_names_by_model", {}).get(
                    f"base_soft_label_{side}", meta_columns["feature_names"]
                ),
                "feature_contract_hash": meta_columns.get(
                    "feature_contract_hash_by_model", {}
                ).get(f"base_soft_label_{side}", meta_columns.get("feature_contract_hash")),
            }
            for side in ("long", "short")
        }
    }
    meta_oof = args.output_root / "meta_oof"
    meta_oof.mkdir(exist_ok=True)
    (meta_oof / "meta_feature_contract.json").write_text(
        json.dumps(feature_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "hybrid_policy_replay_bundle_v1",
        "hybrid_root": str(args.hybrid_root),
        "base_model_dir": str(base_dir),
        "meta_model_dir": str(meta_dir),
        "side_residual_expert": str(expert_path),
        "ae_gmm_state": str(ae_state),
        "policy_root": str(args.policy_root),
        "mlp_postprocessor_enabled": False,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
