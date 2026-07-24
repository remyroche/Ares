#!/usr/bin/env python3
"""Train/predict the April-May-June meta regime ablation matrix.

This script is an entrypoint only.  Importing it does not launch an ablation.
It reuses the fixed base model / candidate handoff and fixed baseline meta HPO
params, then varies only the regime/context features for each arm.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_regime_ablation import (  # noqa: E402
    CROSS_ASSET_CONTEXT_FEATURES,
    FrozenPhaseStateContext,
    SideArchetypeIdentityContext,
    apply_regime_builder_fold,
    drop_oos_outcome_columns,
    make_regime_builder,
    regime_feature_names,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import run_smoke  # noqa: E402

DEFAULT_RUN_ROOT = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_"
    "largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706"
)
DEFAULT_HANDOFF_DIR = (
    DEFAULT_RUN_ROOT
    / "s52_trailing_regime_meta_handoff_top30_allsafe_aegmm_fixedtargets_oos15_20260706"
)
DEFAULT_REFERENCE_MANIFEST = (
    DEFAULT_RUN_ROOT
    / "train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5"
    / "best_full_oos_fixedfs_streamed_v1"
    / "manifest.json"
)
DEFAULT_OUT_ROOT = (
    DEFAULT_RUN_ROOT / "train_meta_regime_ablation_matrix_apr_may_jun_20260707"
)
DEFAULT_PHASE_STATE_CONTEXT = Path(
    "data_perp/reports/global_residual_state_discovery_20260712_v2/"
    "global_side_latent_states_phase_relevance_only/side_timestamp_market_states.parquet"
)

DEFAULT_ARMS = (
    "baseline_current_full_context",
    "baseline_no_cross_context",
    "causal_phase_state_context",
    "side_archetype_identity_context",
    "causal_phase_side_archetype_context",
    "current_archetype_meta_regimes",
    "meta_feature_only_regimes",
    "base_error_signature_regimes",
    "joint_feature_error_regimes",
    "hit_surprise_failure_shock_regimes",
    "mlp_failure_shock_regimes",
    "residual_event_aegmm_precomputed",
    "residual_event_aegmm_local",
    "residual_event_aegmm_local_market",
    "side_archetype_local_regimes",
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
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    # pandas.Timestamp is intentionally handled without importing pandas in
    # this lightweight runner module.
    if type(value).__name__ == "Timestamp" and hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _selected_features(manifest: dict[str, Any]) -> list[str]:
    features = (
        manifest.get("selected_feature_union")
        or manifest.get("selected_features")
        or []
    )
    out = [str(col) for col in features if str(col).strip()]
    if not out:
        raise ValueError(
            "Reference manifest does not contain selected_feature_union/selected_features."
        )
    return list(dict.fromkeys(out))


def _model_params(manifest: dict[str, Any]) -> dict[str, Any]:
    classifier = dict(manifest.get("classifier_params") or {})
    regressor = dict(manifest.get("regressor_params") or classifier)
    if not classifier:
        raise ValueError("Reference manifest does not contain classifier_params.")
    return {"classifier": classifier, "regressor": regressor}


def _bundle_reference(path: Path) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Load a frozen lifecycle reference without a second model search."""

    import joblib

    bundle = joblib.load(path)
    features = [str(name) for name in getattr(bundle, "selected_features", [])]
    model = getattr(bundle, "lifecycle_model", None)
    if not features or model is None or not hasattr(model, "get_params"):
        raise ValueError(f"Incomplete lifecycle reference bundle: {path}")
    params = dict(model.get_params())
    return (
        list(dict.fromkeys(features)),
        {"classifier": dict(params), "regressor": dict(params)},
        {
            "source": "lifecycle_bundle",
            "path": str(path),
            "model_class": type(model).__name__,
            "selected_feature_count": len(features),
        },
    )


def _resolve_soft_label_paths(values: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("train_global_*_5_*.parquet")))
        elif path.exists():
            paths.append(path)
    return list(dict.fromkeys(paths))


def _arm_features(
    arm: str,
    base_features: list[str],
    *,
    seed: int,
    phase_context: FrozenPhaseStateContext | None = None,
    identity_context: SideArchetypeIdentityContext | None = None,
) -> list[str]:
    features = list(base_features)
    if arm == "baseline_no_cross_context":
        features = [col for col in features if col not in CROSS_ASSET_CONTEXT_FEATURES]
    features.extend(regime_feature_names(arm, seed=seed))
    if arm in {"causal_phase_state_context", "causal_phase_side_archetype_context"}:
        if phase_context is None:
            raise ValueError(f"{arm} requires a frozen phase-state context source.")
        features.extend(phase_context.feature_names())
    if arm in {
        "side_archetype_identity_context",
        "causal_phase_side_archetype_context",
    }:
        if identity_context is None:
            raise ValueError(
                f"{arm} requires a frozen side x archetype identity context."
            )
        features.extend(identity_context.feature_names())
    return list(dict.fromkeys(features))


def _minimal_fixed_handoff_columns(
    handoff_path: Path,
    selected_features: list[str],
) -> list[str]:
    """Resolve raw source columns for a frozen encoded meta feature contract.

    The baseline reference uses numeric features plus fold-derived reliability
    and OOD fields. This resolver avoids loading unrelated config-full columns
    while retaining every raw source that can produce a selected feature.
    """
    try:
        import pyarrow.parquet as pq

        schema = list(pq.read_schema(handoff_path).names)
    except Exception:
        import pandas as pd

        schema = list(pd.read_parquet(handoff_path).columns)
    schema_set = set(schema)
    required = {
        "row_id",
        "__ts__",
        "__symbol__",
        "side_name",
        "month",
        "score",
        "selected_top30",
        # Archetype identity is needed by base priors/reliability and reports.
        "archetype_label_family",
        "policy_archetype",
        "archetype_policy_key",
        "local_side_archetype",
        "source_archetype",
        "source_semantic_family",
        "source_semantic_family_base",
        "long_source_regime_split",
    }
    generated_prefixes = (
        "meta_sel_ood_",
        "rel_",
        "base_margin_",
        "base_signal_",
        "base_score_rank_pct_",
        "base_rank_band_",
        "ctx_phase__",
        "ctx_phase_",
        "ctx_identity__",
    )
    raw_schema = sorted(schema, key=lambda name: (-len(name), name))
    for feature in selected_features:
        name = str(feature)
        if name.startswith(generated_prefixes):
            continue
        if name in schema_set:
            required.add(name)
            continue
        # Selected categorical dummies use <raw_feature>_<category>. Preserve
        # the longest matching raw source to avoid misreading nested names.
        matched = next((raw for raw in raw_schema if name.startswith(f"{raw}_")), None)
        if matched is not None:
            required.add(matched)
    return sorted(required.intersection(schema_set))


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


def _make_fold_builder(
    arm: str,
    *,
    seed: int,
    phase_context: FrozenPhaseStateContext | None = None,
    identity_context: SideArchetypeIdentityContext | None = None,
):
    use_phase_context = arm in {
        "causal_phase_state_context",
        "causal_phase_side_archetype_context",
    }
    use_identity_context = arm in {
        "side_archetype_identity_context",
        "causal_phase_side_archetype_context",
    }
    if (
        not use_phase_context
        and not use_identity_context
        and make_regime_builder(arm, seed=seed) is None
    ):
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
        generated: list[str] = []
        metadata: dict[str, Any] = {}
        if use_identity_context:
            if identity_context is None:
                raise ValueError("Missing frozen side x archetype identity context.")
            train_identity = identity_context.transform_train(train)
            valid_identity = identity_context.transform_oos(
                drop_oos_outcome_columns(valid)
            )
            train = train.copy(deep=False)
            valid = valid.copy(deep=False)
            for col in identity_context.feature_names():
                train[col] = train_identity[col].to_numpy(dtype="float32", copy=False)
                valid[col] = valid_identity[col].to_numpy(dtype="float32", copy=False)
            generated.extend(identity_context.feature_names())
            metadata["side_archetype_identity_context"] = identity_context.manifest()
        if use_phase_context:
            if phase_context is None:
                raise ValueError(
                    "Missing frozen phase-state context for causal_phase_state_context arm."
                )
            train_phase = phase_context.transform_train(train)
            valid_phase = phase_context.transform_oos(drop_oos_outcome_columns(valid))
            train = train.copy(deep=False)
            valid = valid.copy(deep=False)
            for col in phase_context.feature_names():
                train[col] = train_phase[col].to_numpy(dtype="float32", copy=False)
                valid[col] = valid_phase[col].to_numpy(dtype="float32", copy=False)
            generated.extend(phase_context.feature_names())
            metadata["causal_phase_context"] = phase_context.manifest()

        if make_regime_builder(arm, seed=seed) is not None:
            train, valid, regime_features, regime_meta = apply_regime_builder_fold(
                arm,
                train=train,
                valid=valid,
                seed=_fold_seed(seed, arm, fold),
            )
            generated.extend(regime_features)
            metadata["regime_builder"] = regime_meta
        return train, valid, list(dict.fromkeys(generated)), metadata

    return _builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument(
        "--handoff-path",
        type=Path,
        default=None,
        help="Optional direct handoff parquet; avoids materializing a duplicate handoff directory.",
    )
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument(
        "--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST
    )
    parser.add_argument(
        "--reference-bundle",
        type=Path,
        default=None,
        help="Frozen lifecycle bundle whose feature list and LightGBM params are reused.",
    )
    parser.add_argument(
        "--soft-label-path",
        type=Path,
        action="append",
        default=[],
        help=(
            "Materialized base-label parquet or directory. The canonical soft target "
            "is joined only for the meta training loss."
        ),
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--phase-state-context",
        type=Path,
        default=DEFAULT_PHASE_STATE_CONTEXT,
        help=(
            "Point-in-time side timestamp state table. Only deterministic state_phase__ fields "
            "are used by the causal_phase_state_context arm."
        ),
    )
    parser.add_argument("--phase-asof-max-minutes", type=int, default=60)
    parser.add_argument("--frontier", default="top30")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument(
        "--colsample-bytree-override",
        type=float,
        default=None,
        help=(
            "Optional common LightGBM feature-subsampling fraction for every arm. "
            "Use 1.0 for a feature-addition ablation when added columns would "
            "otherwise change the reference model's column-subsampling draw."
        ),
    )
    parser.add_argument(
        "--subsample-override",
        type=float,
        default=None,
        help=(
            "Optional common LightGBM row-subsampling fraction for every arm. "
            "Use 1.0 with --colsample-bytree-override 1.0 for a deterministic "
            "feature-addition control."
        ),
    )
    parser.add_argument("--eval-months", default=",".join(DEFAULT_EVAL_MONTHS))
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--minimal-artifacts", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument(
        "--model-train-max-rows",
        type=int,
        default=0,
        help=(
            "Optional time-spread cap for training rows passed to the meta smoke "
            "runner. Use 0 for full folds; nonzero is intended for bounded code "
            "smokes only."
        ),
    )
    parser.add_argument(
        "--regime-builder-handoff-columns",
        choices=("all", "selected"),
        default="all",
        help=(
            "Input column policy for exploratory regime-builder arms. 'all' keeps "
            "the broad pre-selection universe for serious ablations; 'selected' "
            "loads only columns needed by the fixed selected feature contract and "
            "is intended for fast code-path smokes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_arms:
        print(json.dumps({"available_arms": list(DEFAULT_ARMS)}, indent=2))
        return

    if args.reference_bundle is not None:
        base_features, params, reference = _bundle_reference(args.reference_bundle)
    else:
        reference = _load_manifest(args.reference_manifest)
        base_features = _selected_features(reference)
        params = _model_params(reference)
    if args.colsample_bytree_override is not None:
        colsample = float(args.colsample_bytree_override)
        if not 0.0 < colsample <= 1.0:
            raise ValueError("--colsample-bytree-override must be in (0, 1].")
        # A common override prevents feature-count-dependent random subsampling
        # from masquerading as value from a newly appended regime feature block.
        for model_kind in ("classifier", "regressor"):
            params.setdefault(model_kind, {})["colsample_bytree"] = colsample
    if args.subsample_override is not None:
        subsample = float(args.subsample_override)
        if not 0.0 < subsample <= 1.0:
            raise ValueError("--subsample-override must be in (0, 1].")
        for model_kind in ("classifier", "regressor"):
            params.setdefault(model_kind, {})["subsample"] = subsample
    handoff_path = (
        Path(args.handoff_path)
        if args.handoff_path is not None
        else Path(args.handoff_dir) / "train_meta_regime_handoff.parquet"
    )
    soft_label_paths = _resolve_soft_label_paths(list(args.soft_label_path))
    arms = [part.strip() for part in str(args.arms).split(",") if part.strip()]
    unknown = sorted(set(arms) - set(DEFAULT_ARMS))
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}; available={list(DEFAULT_ARMS)}")
    eval_months = [
        part.strip() for part in str(args.eval_months).split(",") if part.strip()
    ]
    args.out_root.mkdir(parents=True, exist_ok=True)
    phase_context = None
    phase_arms = {"causal_phase_state_context", "causal_phase_side_archetype_context"}
    identity_arms = {
        "side_archetype_identity_context",
        "causal_phase_side_archetype_context",
    }
    if phase_arms.intersection(arms):
        phase_context = FrozenPhaseStateContext(
            source_path=Path(args.phase_state_context),
            max_lag_minutes=int(args.phase_asof_max_minutes),
        )
    identity_context = None
    if identity_arms.intersection(arms):
        identity_context = SideArchetypeIdentityContext.from_parquet(
            handoff_path
        )

    matrix_manifest: dict[str, Any] = {
        "generated_by": "run_train_meta_regime_ablation_matrix",
        "handoff_dir": str(args.handoff_dir),
        "handoff_path": str(handoff_path),
        "ledger_path": str(args.ledger_path) if args.ledger_path else None,
        "soft_label_paths": [str(path) for path in soft_label_paths],
        "reference_manifest": (
            str(args.reference_bundle)
            if args.reference_bundle is not None
            else str(args.reference_manifest)
        ),
        "out_root": str(args.out_root),
        "frontier": str(args.frontier),
        "colsample_bytree_override": (
            float(args.colsample_bytree_override)
            if args.colsample_bytree_override is not None
            else None
        ),
        "subsample_override": (
            float(args.subsample_override)
            if args.subsample_override is not None
            else None
        ),
        "eval_months": eval_months,
        "model_train_max_rows": int(args.model_train_max_rows),
        "regime_builder_handoff_columns": str(args.regime_builder_handoff_columns),
        "base_feature_count": len(base_features),
        "arms": [],
        "leakage_contract": {
            "base_model_retrained": False,
            "meta_hpo_params": "fixed_from_reference_manifest",
            "folds": "train rows strictly before OOS month start",
            "regime_oos_assignment": "frozen fold-local scaler/clusterer/classifier",
            "realized_error_descriptors": "train-only cluster labels/priors; never passed directly to OOS transforms",
            "causal_phase_context": (
                phase_context.manifest()
                if phase_context is not None
                else "not_requested"
            ),
            "side_archetype_identity_context": (
                identity_context.manifest()
                if identity_context is not None
                else "not_requested"
            ),
        },
    }

    for arm in arms:
        arm_out = args.out_root / arm
        features = _arm_features(
            arm,
            base_features,
            seed=int(args.seed),
            phase_context=phase_context,
            identity_context=identity_context,
        )
        generated = regime_feature_names(arm, seed=int(args.seed))
        if arm in phase_arms and phase_context is not None:
            generated = list(
                dict.fromkeys([*generated, *phase_context.feature_names()])
            )
        if arm in identity_arms and identity_context is not None:
            generated = list(
                dict.fromkeys([*generated, *identity_context.feature_names()])
            )
        # The two fixed-contract controls do not need every config-full input.
        # More exploratory regime builders retain the full pre-selection space.
        use_minimal_handoff_columns = arm in {
            "baseline_current_full_context",
            "baseline_no_cross_context",
            "causal_phase_state_context",
            "side_archetype_identity_context",
            "causal_phase_side_archetype_context",
        } or str(args.regime_builder_handoff_columns).lower() == "selected"
        handoff_columns = (
            _minimal_fixed_handoff_columns(
                handoff_path, features
            )
            if use_minimal_handoff_columns
            else None
        )
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
                    "removed_cross_asset_context_features": sorted(
                        set(base_features) - set(features)
                    ),
                    "handoff_input_column_count": len(handoff_columns)
                    if handoff_columns is not None
                    else "all",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        manifest = run_smoke(
            handoff_dir=args.handoff_dir,
            ledger_path=args.ledger_path,
            handoff_path=handoff_path,
            soft_label_paths=soft_label_paths,
            ood_reference_features=base_features,
            out_dir=arm_out,
            frontier=str(args.frontier),
            seed=int(args.seed),
            train_scope="selected",
            enable_base_prior_features=True,
            enable_reliability_features=True,
            # Neither family is present in the frozen 55-feature reference
            # contract. Avoid materializing discarded fold features while
            # keeping the selected model inputs identical.
            enable_support_drift_features=False,
            enable_hit_surprise_features=False,
            feature_selection_top_n=0,
            feature_selection_target="ev_frontier",
            feature_selection_method="lgbm_pipeline",
            max_oos_model_age_days=0,
            validation_scope="chronological",
            model_train_max_rows=int(args.model_train_max_rows),
            model_params=params,
            model_profile_name=f"{arm}_fixed_reference_hpo",
            meta_head_mode="single_base_soft_label",
            minimal_artifacts=bool(args.minimal_artifacts),
            fixed_selected_features=features,
            eval_months=eval_months,
            fold_feature_builder=_make_fold_builder(
                arm,
                seed=int(args.seed),
                phase_context=phase_context,
                identity_context=identity_context,
            ),
            fold_feature_profile_name=arm,
            extra_prediction_columns=generated,
            force_prediction_shards=True,
            combine_prediction_shards=False,
            handoff_columns=handoff_columns,
        )
        arm_record = {
            "arm": arm,
            "out_dir": str(arm_out),
            "generated_feature_count": len(generated),
            "fixed_feature_count": len(features),
            "selected_feature_union_count": manifest.get(
                "selected_feature_union_count"
            ),
            "best_selector": (manifest.get("best_selector") or {}).get("selector"),
            "best_status": (manifest.get("best_selector") or {}).get(
                "meta_smoke_status"
            ),
            "handoff_input_column_count": len(handoff_columns)
            if handoff_columns is not None
            else "all",
        }
        matrix_manifest["arms"].append(arm_record)
        (args.out_root / "matrix_manifest.json").write_text(
            json.dumps(_json_safe(matrix_manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "event": "meta_regime_ablation_matrix_done",
                "out_root": str(args.out_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
