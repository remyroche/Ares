#!/usr/bin/env python3
"""Train/predict the April-May-June meta regime ablation matrix.

This script is an entrypoint only.  Importing it does not launch an ablation.
It reuses the fixed base model / candidate handoff and fixed baseline meta HPO
params, then varies only the regime/context features for each arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zlib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_regime_ablation import (  # noqa: E402
    CROSS_ASSET_CONTEXT_FEATURES,
    apply_regime_builder_fold,
    make_regime_builder,
    regime_feature_names,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import run_smoke  # noqa: E402


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
DEFAULT_OUT_ROOT = DEFAULT_RUN_ROOT / "train_meta_regime_ablation_matrix_apr_may_jun_20260707"

DEFAULT_ARMS = (
    "baseline_current_full_context",
    "baseline_no_cross_context",
    "current_archetype_meta_regimes",
    "meta_feature_only_regimes",
    "base_error_signature_regimes",
    "joint_feature_error_regimes",
    "side_archetype_local_regimes",
    "temporal_reliability_regimes",
    "supervised_embedding_regimes",
)

DEFAULT_EVAL_MONTHS = ("2026-04", "2026-05", "2026-06")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _selected_features(manifest: dict[str, Any]) -> list[str]:
    features = manifest.get("selected_feature_union") or manifest.get("selected_features") or []
    out = [str(col) for col in features if str(col).strip()]
    if not out:
        raise ValueError("Reference manifest does not contain selected_feature_union/selected_features.")
    return list(dict.fromkeys(out))


def _model_params(manifest: dict[str, Any]) -> dict[str, Any]:
    classifier = dict(manifest.get("classifier_params") or {})
    regressor = dict(manifest.get("regressor_params") or classifier)
    if not classifier:
        raise ValueError("Reference manifest does not contain classifier_params.")
    return {"classifier": classifier, "regressor": regressor}


def _arm_features(arm: str, base_features: list[str], *, seed: int) -> list[str]:
    features = list(base_features)
    if arm == "baseline_no_cross_context":
        features = [col for col in features if col not in CROSS_ASSET_CONTEXT_FEATURES]
    features.extend(regime_feature_names(arm, seed=seed))
    return list(dict.fromkeys(features))


def _fold_seed(seed: int, arm: str, fold: str) -> int:
    token = f"{arm}:{fold}".encode("utf-8")
    return int(seed) + int(zlib.crc32(token) % 100_000)


def _arm_complete(arm_out: Path, eval_months: list[str]) -> bool:
    if not (arm_out / "manifest.json").exists():
        return False
    shard_dir = arm_out / "prediction_shards"
    for idx, month in enumerate(eval_months, start=1):
        if not (shard_dir / f"predictions_{idx:04d}_{month}.parquet").exists():
            return False
    return True


def _make_fold_builder(arm: str, *, seed: int):
    if make_regime_builder(arm, seed=seed) is None:
        return None

    def _builder(
        *,
        train,
        valid,
        fold: str,
        month: str,
        valid_start,
        valid_end,
        selected_col: str,
    ):
        del month, valid_start, valid_end, selected_col
        return apply_regime_builder_fold(
            arm,
            train=train,
            valid=valid,
            seed=_fold_seed(seed, arm, fold),
        )

    return _builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--frontier", default="top30")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--eval-months", default=",".join(DEFAULT_EVAL_MONTHS))
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--minimal-artifacts", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_arms:
        print(json.dumps({"available_arms": list(DEFAULT_ARMS)}, indent=2))
        return

    reference = _load_manifest(args.reference_manifest)
    base_features = _selected_features(reference)
    params = _model_params(reference)
    arms = [part.strip() for part in str(args.arms).split(",") if part.strip()]
    unknown = sorted(set(arms) - set(DEFAULT_ARMS))
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}; available={list(DEFAULT_ARMS)}")
    eval_months = [part.strip() for part in str(args.eval_months).split(",") if part.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)

    matrix_manifest: dict[str, Any] = {
        "generated_by": "run_train_meta_regime_ablation_matrix",
        "handoff_dir": str(args.handoff_dir),
        "ledger_path": str(args.ledger_path) if args.ledger_path else None,
        "reference_manifest": str(args.reference_manifest),
        "out_root": str(args.out_root),
        "frontier": str(args.frontier),
        "eval_months": eval_months,
        "base_feature_count": len(base_features),
        "arms": [],
        "leakage_contract": {
            "base_model_retrained": False,
            "meta_hpo_params": "fixed_from_reference_manifest",
            "folds": "train rows strictly before OOS month start",
            "regime_oos_assignment": "frozen fold-local scaler/clusterer/classifier",
            "realized_error_descriptors": "train-only cluster labels/priors; never passed directly to OOS transforms",
        },
    }

    for arm in arms:
        arm_out = args.out_root / arm
        features = _arm_features(arm, base_features, seed=int(args.seed))
        generated = regime_feature_names(arm, seed=int(args.seed))
        if not args.rerun_complete and _arm_complete(arm_out, eval_months):
            print(
                json.dumps(
                    {
                        "event": "meta_regime_ablation_arm_skip_complete",
                        "arm": arm,
                        "out_dir": str(arm_out),
                        "eval_months": eval_months,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            matrix_manifest["arms"].append(
                {
                    "arm": arm,
                    "out_dir": str(arm_out),
                    "generated_feature_count": len(generated),
                    "fixed_feature_count": len(features),
                    "skipped_complete": True,
                }
            )
            (args.out_root / "matrix_manifest.json").write_text(
                json.dumps(_json_safe(matrix_manifest), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            continue
        print(
            json.dumps(
                {
                    "event": "meta_regime_ablation_arm_start",
                    "arm": arm,
                    "out_dir": str(arm_out),
                    "eval_months": eval_months,
                    "fixed_feature_count": len(features),
                    "generated_feature_count": len(generated),
                    "removed_cross_asset_context_features": sorted(set(base_features) - set(features)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        manifest = run_smoke(
            handoff_dir=args.handoff_dir,
            ledger_path=args.ledger_path,
            out_dir=arm_out,
            frontier=str(args.frontier),
            seed=int(args.seed),
            train_scope="selected",
            enable_base_prior_features=True,
            enable_reliability_features=True,
            enable_support_drift_features=True,
            enable_hit_surprise_features=True,
            feature_selection_top_n=0,
            feature_selection_target="ev_frontier",
            feature_selection_method="lgbm_pipeline",
            max_oos_model_age_days=0,
            validation_scope="chronological",
            model_train_max_rows=0,
            model_params=params,
            model_profile_name=f"{arm}_fixed_reference_hpo",
            meta_head_mode="single_base_soft_label",
            minimal_artifacts=bool(args.minimal_artifacts),
            fixed_selected_features=features,
            eval_months=eval_months,
            fold_feature_builder=_make_fold_builder(arm, seed=int(args.seed)),
            fold_feature_profile_name=arm,
            extra_prediction_columns=generated,
            force_prediction_shards=True,
            combine_prediction_shards=False,
        )
        arm_record = {
            "arm": arm,
            "out_dir": str(arm_out),
            "generated_feature_count": len(generated),
            "fixed_feature_count": len(features),
            "selected_feature_union_count": manifest.get("selected_feature_union_count"),
            "best_selector": (manifest.get("best_selector") or {}).get("selector"),
            "best_status": (manifest.get("best_selector") or {}).get("meta_smoke_status"),
        }
        matrix_manifest["arms"].append(arm_record)
        (args.out_root / "matrix_manifest.json").write_text(
            json.dumps(_json_safe(matrix_manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(json.dumps({"event": "meta_regime_ablation_matrix_done", "out_root": str(args.out_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
