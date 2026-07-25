#!/usr/bin/env python3
"""Train the strict OOF MAE competing-risk challenger only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_auxiliary_bundle_training import (  # noqa: E402
    MEANINGFUL_EVENT_ROLE,
    canonical_role_targets,
    selected_features_for_role,
)
from extreme_price_movements.path_auxiliary_lgbm import (  # noqa: E402
    fit_base_archetype_label_feature_contract,
    transform_base_archetype_label_features,
)
from extreme_price_movements.path_auxiliary_mae_competing_risk import (  # noqa: E402
    MAE_COMPETING_RISK_SCHEMA,
    fit_side_local_mae_competing_risk_family,
)
from extreme_price_movements.path_auxiliary_model_families import (  # noqa: E402
    MAE_COLUMN,
    MEANINGFUL_HIT_COLUMN,
)
from extreme_price_movements.path_auxiliary_role_training import (  # noqa: E402
    FIXED_MAY_JULY_OOF_MONTHS,
)
from scripts.run_path_auxiliary_lgbm_models import (  # noqa: E402
    ARCHETYPE_COLUMNS,
    DEFAULT_LABEL_RESOLUTION_COLUMN,
    STRICT_IDENTITY_COLUMNS,
    _complete_archetype_source,
    _join_archetype_context,
    _load_labels,
)
from scripts.run_path_auxiliary_role_bundles import (  # noqa: E402
    CANONICAL_REFERENCE_END,
    DEFAULT_CONTEXT,
    DEFAULT_FEATURE_DIR,
    DEFAULT_LABELS,
    _build_resource_guard,
    _file_sha256,
    _full_selected_matrix,
)

DEFAULT_SELECTIONS = (
    ROOT
    / "data_perp/artifacts/packb_path_auxiliary_role_bundles_20260725_v1_31_8/shared/selection_contracts.joblib"
)
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/path_auxiliary_mae_competing_risk_20260725_v1"
)
STOP_ATR_BY_SIDE = {"long": 4.0, "short": 3.525840972995973}
RISK_OUTPUTS = (
    "favorable_before_0_5r",
    "adverse_0_5r_before_mfe",
    "neither_before_horizon",
    "stop_1r_before_mfe",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    guard = _build_resource_guard(
        output_dir=args.output_dir,
        min_free_ram_gib=args.resource_min_free_ram_gib,
        max_process_rss_gib=args.resource_max_process_rss_gib,
        min_free_disk_gib=args.resource_min_free_disk_gib,
        check_interval_seconds=1.0,
        telemetry_path=args.output_dir / "training_resource_telemetry.jsonl",
    )
    labels, _ = _load_labels(
        args.labels_path,
        label_resolution_column=DEFAULT_LABEL_RESOLUTION_COLUMN,
        max_rows=0,
    )
    labels, context_report = _join_archetype_context(
        labels, args.context_path, labels_are_canonical_top40=False
    )
    reference = labels["__ts__"].lt(CANONICAL_REFERENCE_END) & labels[
        DEFAULT_LABEL_RESOLUTION_COLUMN
    ].lt(CANONICAL_REFERENCE_END)
    reference_labels = labels.loc[reference].reset_index(drop=True)
    sources = [
        column
        for column in ARCHETYPE_COLUMNS
        if _complete_archetype_source(reference_labels, column)
    ]
    contract = fit_base_archetype_label_feature_contract(
        reference_labels,
        source_columns=sources,
        canonical_source=sources[0],
    )
    archetype_features = transform_base_archetype_label_features(labels, contract)
    selections = joblib.load(args.selection_contracts)["selection_contracts"]
    selected = {
        side: {
            "event": selected_features_for_role(
                selections, MEANINGFUL_EVENT_ROLE, side
            ),
            "if_hit": selected_features_for_role(
                selections, "mae_before_meaningful_mfe_atr.if_hit", side
            ),
            "if_no_hit": selected_features_for_role(
                selections, "mae_before_meaningful_mfe_atr.if_no_hit", side
            ),
        }
        for side in ("long", "short")
    }
    feature_union = list(
        dict.fromkeys(
            feature
            for side_contract in selected.values()
            for stream in side_contract.values()
            for feature in stream
        )
    )
    matrix, availability = _full_selected_matrix(
        labels,
        selected_features=feature_union,
        feature_dir=args.feature_dir,
        handoff_feature_columns=context_report["handoff_model_feature_columns"],
        archetype_features=archetype_features,
        guard=guard,
        stage="mae_competing_risk",
    )
    role_target = canonical_role_targets(labels)["mae_before_meaningful_mfe_atr.if_hit"]
    family = fit_side_local_mae_competing_risk_family(
        matrix,
        labels[MAE_COLUMN].to_numpy(np.float32),
        labels[MEANINGFUL_HIT_COLUMN].to_numpy(np.float32),
        train_mask=role_target.valid_mask,
        sides=labels["side"].to_numpy(),
        selected_features=selected,
        timestamps=labels["__ts__"].to_numpy(),
        label_resolved_at=labels[DEFAULT_LABEL_RESOLUTION_COLUMN].to_numpy(),
        selection_hpo_reference_end=CANONICAL_REFERENCE_END,
        stop_atr_by_side=STOP_ATR_BY_SIDE,
        n_trials=args.n_trials,
        hpo_rows=args.hpo_rows,
        random_state=args.seed,
        progress_callback=lambda event, payload: guard.checkpoint(
            f"mae_competing_risk:{event}"
        ),
    )
    mask = family["oof_prediction_mask"]
    output = labels.loc[
        mask, [column for column in STRICT_IDENTITY_COLUMNS if column in labels]
    ].rename(columns={"side": "side_name"})
    predictions = family["oof_predictions"]
    for name in RISK_OUTPUTS:
        output[f"prediction_p_{name}"] = predictions[name][mask]
    adverse = output["prediction_p_adverse_0_5r_before_mfe"].to_numpy(float)
    stop = output["prediction_p_stop_1r_before_mfe"].to_numpy(float)
    output["prediction"] = 0.5 * (adverse - stop) + stop
    output["oof_fold"] = family["oof_fold_ids"][mask]
    fold_table = {
        (record["side"], record["fold_month"]): record
        for record in family["fold_provenance"]
    }
    months = output["__ts__"].dt.strftime("%Y-%m")
    records = [
        fold_table[(side, month)]
        for side, month in zip(output["side_name"].astype(str), months)
    ]
    output["validation_start"] = pd.to_datetime(
        [record["valid_start"] for record in records], utc=True
    )
    output["train_decision_cutoff"] = pd.to_datetime(
        [record["training_label_resolved_max"] for record in records], utc=True
    )
    output["label_resolution_available_at"] = output["train_decision_cutoff"]
    output["available_at"] = output["__ts__"]
    ordered = [
        "__ts__",
        "__symbol__",
        "side_name",
        "candidate_id",
        "prediction",
        "oof_fold",
        "validation_start",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "available_at",
        *(f"prediction_p_{name}" for name in RISK_OUTPUTS),
    ]
    parquet = args.output_dir / "mae_competing_risk.parquet"
    output.loc[:, ordered].to_parquet(parquet, index=False)
    joblib.dump(family, args.output_dir / "mae_competing_risk_family.joblib")
    declarations = {
        "prediction": {
            "head": "mae",
            "role": "pre_entry_auxiliary_oof_prediction",
            "source_prediction_column": "pred_expected_adverse_r",
            "target": False,
        },
        **{
            f"prediction_p_{name}": {
                "head": "mae",
                "role": "pre_entry_auxiliary_mae_competing_risk_probability_oof",
                "source_prediction_column": f"pred_p_{name}",
                "target": False,
            }
            for name in RISK_OUTPUTS
        },
    }
    manifest = {
        "schema": MAE_COMPETING_RISK_SCHEMA,
        "prediction_role": "mae_before_mfe_oof",
        "source_artifact_sha256": _file_sha256(parquet),
        "prediction_columns": declarations,
        "rows": int(len(output)),
        "source": {
            "labels_sha256": _file_sha256(args.labels_path),
            "selection_contracts_sha256": _file_sha256(args.selection_contracts),
            "selected_features_by_side_and_source_role": selected,
            "feature_availability": availability,
        },
        "training": {
            "reference_end": CANONICAL_REFERENCE_END.isoformat(),
            "oof_months": list(FIXED_MAY_JULY_OOF_MONTHS),
            "hpo_trials_per_model": args.n_trials,
            "hpo_rows": args.hpo_rows,
            "stop_atr_by_side": STOP_ATR_BY_SIDE,
            "constraint_contract": family["constraint_contract"],
            "side_states": {
                side: {
                    key: value
                    for key, value in state.items()
                    if key
                    in {
                        "selected_features",
                        "stop_atr",
                        "multiclass_params",
                        "stop_params",
                        "multiclass_hpo",
                        "stop_hpo",
                    }
                }
                for side, state in family["side_models"].items()
            },
        },
        "target": {
            "kind": "mae_competing_risk",
            "unit": "deployed_stop_R",
            "first_outcomes": list(family["risk_class_names"]),
            "severity": "P(stop_1R | adverse_0.5R before meaningful MFE)",
        },
    }
    manifest["prediction_role_manifest_sha256"] = _canonical_hash(manifest)
    (args.output_dir / "mae_competing_risk.manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--context-path", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--selection-contracts", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--hpo-rows", type=int, default=45_000)
    parser.add_argument("--seed", type=int, default=5252)
    parser.add_argument("--resource-min-free-ram-gib", type=float, default=2.0)
    parser.add_argument("--resource-max-process-rss-gib", type=float, default=12.0)
    parser.add_argument("--resource-min-free-disk-gib", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": "complete", "rows": result["rows"]}))
