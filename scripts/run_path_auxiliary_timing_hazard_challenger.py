#!/usr/bin/env python3
"""Train the strict OOF discrete-hazard timing challenger only."""

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
from extreme_price_movements.path_auxiliary_model_families import (  # noqa: E402
    MEANINGFUL_HIT_COLUMN,
    TIMING_COLUMN,
)
from extreme_price_movements.path_auxiliary_timing_hazard import (  # noqa: E402
    TIMING_HAZARD_SCHEMA,
    fit_side_local_timing_hazard_family,
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
    ROOT / "data_perp/artifacts/path_auxiliary_timing_hazard_challenger_20260725_v1"
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
        reference_labels, source_columns=sources, canonical_source=sources[0]
    )
    archetype_features = transform_base_archetype_label_features(labels, contract)
    payload = joblib.load(args.selection_contracts)
    selections = payload["selection_contracts"]
    by_side = {
        side: {
            2: selected_features_for_role(
                selections, "time_to_first_meaningful_mfe.hit_by_2h", side
            ),
            4: selected_features_for_role(
                selections, "time_to_first_meaningful_mfe.hit_by_4h", side
            ),
            8: selected_features_for_role(
                selections, "time_to_first_meaningful_mfe.hit_by_8h", side
            ),
            12: selected_features_for_role(selections, MEANINGFUL_EVENT_ROLE, side),
        }
        for side in ("long", "short")
    }
    feature_union = list(
        dict.fromkeys(
            feature
            for side in ("long", "short")
            for horizon in (2, 4, 8, 12)
            for feature in by_side[side][horizon]
        )
    )
    matrix, availability = _full_selected_matrix(
        labels,
        selected_features=feature_union,
        feature_dir=args.feature_dir,
        handoff_feature_columns=context_report["handoff_model_feature_columns"],
        archetype_features=archetype_features,
        guard=guard,
        stage="timing_hazard",
    )
    target = canonical_role_targets(labels)["time_to_first_meaningful_mfe.hit_by_2h"]
    family = fit_side_local_timing_hazard_family(
        matrix,
        labels[TIMING_COLUMN].to_numpy(np.float32),
        labels[MEANINGFUL_HIT_COLUMN].to_numpy(np.float32),
        timing_train_mask=target.train_mask,
        sides=labels["side"].to_numpy(),
        selected_features=by_side,
        timestamps=labels["__ts__"].to_numpy(),
        label_resolved_at=labels[DEFAULT_LABEL_RESOLUTION_COLUMN].to_numpy(),
        selection_hpo_reference_end=CANONICAL_REFERENCE_END,
        n_trials=args.n_trials,
        hpo_rows=args.hpo_rows,
        random_state=args.seed,
        progress_callback=lambda event, data: guard.checkpoint(
            f"timing_hazard:{event}"
        ),
    )
    mask = family["oof_prediction_mask"]
    output = labels.loc[
        mask, [column for column in STRICT_IDENTITY_COLUMNS if column in labels]
    ].copy()
    output = output.rename(columns={"side": "side_name"})
    if "candidate_id" not in output:
        output["candidate_id"] = (
            output["__symbol__"].astype(str)
            + "|"
            + output["__ts__"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            + "|1h|"
            + output["side_name"].astype(str)
        )
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
    output["label_resolution_available_at"] = pd.to_datetime(
        [record["training_label_resolved_max"] for record in records], utc=True
    )
    output["available_at"] = output["__ts__"]
    cdf = {
        hours: family["oof_predictions_by_horizon"][hours][mask]
        for hours in (2, 4, 8, 12)
    }
    masses = np.column_stack(
        [cdf[2], cdf[4] - cdf[2], cdf[8] - cdf[4], cdf[12] - cdf[8]]
    )
    output["prediction"] = (
        masses @ np.asarray([2.0, 4.0, 8.0, 12.0]) + (1.0 - cdf[12]) * 12.0
    )
    for hours in (2, 4, 8, 12):
        output[f"prediction_p_hit_by_{hours}h"] = cdf[hours]
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
        "prediction_p_hit_by_2h",
        "prediction_p_hit_by_4h",
        "prediction_p_hit_by_8h",
        "prediction_p_hit_by_12h",
    ]
    parquet = args.output_dir / "timing.parquet"
    output.loc[:, ordered].to_parquet(parquet, index=False)
    joblib.dump(family, args.output_dir / "timing_hazard_family.joblib")
    declarations = {
        "prediction": {
            "head": "timing",
            "role": "pre_entry_auxiliary_oof_prediction",
            "source_prediction_column": "discrete_hazard_expected_censored_time_hours",
            "target": False,
        },
        **{
            f"prediction_p_hit_by_{hours}h": {
                "head": "timing",
                "role": "pre_entry_auxiliary_timing_cdf_probability_oof",
                "source_prediction_column": f"pred_p_hit_by_{hours}h",
                "target": False,
            }
            for hours in (2, 4, 8, 12)
        },
    }
    manifest = {
        "schema": TIMING_HAZARD_SCHEMA,
        "prediction_role": "time_to_mfe_oof",
        "source_artifact_sha256": _file_sha256(parquet),
        "prediction_columns": declarations,
        "rows": int(len(output)),
        "source": {
            "labels_sha256": _file_sha256(args.labels_path),
            "selection_contracts_sha256": _file_sha256(args.selection_contracts),
            "selected_features_by_side_and_horizon": by_side,
            "feature_availability": availability,
        },
        "training": {
            "reference_end": CANONICAL_REFERENCE_END.isoformat(),
            "oof_months": ["2026-05", "2026-06", "2026-07"],
            "hpo_trials": args.n_trials,
            "hpo_rows": args.hpo_rows,
            "side_states": {
                side: {
                    "selected_features": state["selected_features"],
                    "best_params": state["best_params"],
                    "hpo": state["hpo"],
                }
                for side, state in family["side_models"].items()
            },
            "metrics": family["oof_metrics_by_horizon"],
            "constraint_contract": family["constraint_contract"],
        },
        "target": {
            "kind": "timing",
            "unit": "hours",
            "right_censor_hours": 12,
            "event": "first meaningful MFE",
        },
    }
    manifest["prediction_role_manifest_sha256"] = _canonical_hash(manifest)
    (args.output_dir / "timing.manifest.json").write_text(
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
    parser.add_argument("--hpo-rows", type=int, default=45000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--resource-min-free-ram-gib", type=float, default=2.0)
    parser.add_argument("--resource-max-process-rss-gib", type=float, default=12.0)
    parser.add_argument("--resource-min-free-disk-gib", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": "complete", "rows": result["rows"]}))
