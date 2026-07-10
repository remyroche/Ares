#!/usr/bin/env python3
"""Run the extended-pool train_meta cross-asset/context ablation.

This launcher intentionally keeps the full growing-window train set by default
(`model_train_max_rows=0`).  The experiment tests whether the enlarged meta
pool has enough capacity to learn from context/cross-asset features, so a small
row cap would change the question being asked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_s52_train_meta_regime_handoff_smoke import run_smoke


DEFAULT_RUN_ROOT = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_"
    "largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706"
)
DEFAULT_HANDOFF_DIR = DEFAULT_RUN_ROOT / "s52_trailing_regime_meta_handoff_top30_allsafe_aegmm_fixedtargets_oos15_20260706"
DEFAULT_REFERENCE_MANIFEST = (
    DEFAULT_RUN_ROOT
    / "train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5"
    / "best_full_oos_fixedfs_streamed_v1"
    / "manifest.json"
)
DEFAULT_OUT_ROOT = DEFAULT_RUN_ROOT / "train_meta_extended_pool_cross_asset_ablation_20260707_uncapped"

CROSS_ASSET_CONTEXT_FEATURES = {
    "cs_rank_oi_value_z_30d",
    "eth_ret_24h",
    "q_tail_width__ob_spread_z_x_rv_24h",
    "state_spectral_abs_pc3_z",
    "state_spectral_eig_lambda1_share",
    "xs_dispersion__ob_depth_to_qv_z_x_rvol_z",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _fixed_features(manifest: dict[str, Any]) -> list[str]:
    features = manifest.get("selected_feature_union") or manifest.get("selected_features") or []
    return [str(col) for col in features if str(col).strip()]


def _model_params(manifest: dict[str, Any]) -> dict[str, Any]:
    classifier = dict(manifest.get("classifier_params") or {})
    regressor = dict(manifest.get("regressor_params") or classifier)
    return {"classifier": classifier, "regressor": regressor}


def _arm_features(features: list[str], arm: str) -> list[str]:
    if arm == "all_context":
        return list(features)
    if arm == "no_cross_asset":
        return [col for col in features if col not in CROSS_ASSET_CONTEXT_FEATURES]
    raise ValueError(f"Unknown arm: {arm}")


def _arm_out_dir(root: Path, arm: str, model_train_max_rows: int) -> Path:
    cap_label = "uncapped" if int(model_train_max_rows) <= 0 else f"cap{int(model_train_max_rows)}"
    return root / f"{arm}_fixedfs_{cap_label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--arm",
        choices=("all_context", "no_cross_asset", "both"),
        default="both",
        help="Ablation arm to run. Defaults to both arms.",
    )
    parser.add_argument("--frontier", default="top30")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--max-oos-model-age-days", type=int, default=15)
    parser.add_argument("--validation-scope", default="all")
    parser.add_argument(
        "--model-train-max-rows",
        type=int,
        default=0,
        help="0 means no cap. Keep this uncapped for the full-capacity extended-pool test.",
    )
    parser.add_argument("--minimal-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.reference_manifest)
    features = _fixed_features(manifest)
    if not features:
        raise ValueError(f"No selected features found in {args.reference_manifest}")
    params = _model_params(manifest)
    arms = ("all_context", "no_cross_asset") if args.arm == "both" else (args.arm,)
    for arm in arms:
        arm_features = _arm_features(features, arm)
        out_dir = _arm_out_dir(args.out_root, arm, args.model_train_max_rows)
        print(
            json.dumps(
                {
                    "event": "extended_pool_meta_ablation_arm_start",
                    "arm": arm,
                    "out_dir": str(out_dir),
                    "model_train_max_rows": int(args.model_train_max_rows),
                    "selected_feature_count": len(arm_features),
                    "removed_cross_asset_context_features": sorted(set(features) - set(arm_features)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        run_smoke(
            handoff_dir=args.handoff_dir,
            ledger_path=args.ledger_path,
            out_dir=out_dir,
            frontier=str(args.frontier),
            seed=int(args.seed),
            train_scope="selected",
            enable_base_prior_features=True,
            enable_reliability_features=True,
            enable_support_drift_features=True,
            enable_hit_surprise_features=True,
            max_oos_model_age_days=int(args.max_oos_model_age_days),
            validation_scope=str(args.validation_scope),
            model_train_max_rows=int(args.model_train_max_rows),
            model_params=params,
            model_profile_name=f"{arm}_fixedfs_uncapped"
            if int(args.model_train_max_rows) <= 0
            else f"{arm}_fixedfs_cap{int(args.model_train_max_rows)}",
            meta_head_mode="single_base_soft_label",
            minimal_artifacts=bool(args.minimal_artifacts),
            fixed_selected_features=arm_features,
        )


if __name__ == "__main__":
    main()
