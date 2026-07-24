#!/usr/bin/env python3
"""Freeze the sole conservative short-default uncertainty forward challenger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_short_default_uncertainty_ablation import (
    GROUP,
    RISK_COLUMNS,
    _add_uncertainty_components,
    _adjust_rank,
    _metrics,
    _uncertainty,
    _weight_templates,
)


CANDIDATE = {
    "family": "requested_core_equal",
    "threshold": 0.85,
    "alpha": 0.04,
    "minimum_effective_neighbor_support": 1.0,
    "missing_component_fallback": 0.5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _continuous_taxonomy(
    train: pd.DataFrame, evaluated: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = (
        ("ensemble_risk_std", 0.30, False),
        ("neighbor_weighted_outcome_entropy", 0.25, False),
        ("neighbor_weighted_ev_std", 0.30, False),
        ("neighbor_effective_count", 0.15, True),
    )
    train_score = np.zeros(len(train), dtype=np.float32)
    eval_score = np.zeros(len(evaluated), dtype=np.float32)
    references: dict[str, list[float]] = {}
    for column, weight, reverse in specs:
        reference = np.sort(
            pd.to_numeric(train[column], errors="coerce")
            .dropna()
            .to_numpy(np.float32)
        )
        references[column] = [float(reference[0]), float(np.median(reference)), float(reference[-1])]
        train_values = pd.to_numeric(train[column], errors="coerce").to_numpy(np.float32)
        eval_values = pd.to_numeric(evaluated[column], errors="coerce").to_numpy(np.float32)
        train_pct = np.searchsorted(reference, train_values, side="right") / max(len(reference), 1)
        eval_pct = np.searchsorted(reference, eval_values, side="right") / max(len(reference), 1)
        if reverse:
            train_pct = 1.0 - train_pct
            eval_pct = 1.0 - eval_pct
        train_score += np.float32(weight) * np.nan_to_num(train_pct, nan=0.5)
        eval_score += np.float32(weight) * np.nan_to_num(eval_pct, nan=0.5)
    threshold = float(np.quantile(train_score, 0.75))
    parts: list[pd.DataFrame] = []
    for stage, frame, score in (
        ("train_oof", train, train_score),
        ("eval_oos", evaluated, eval_score),
    ):
        adverse = frame["bad_residual_event_target"].astype(bool).to_numpy()
        high = score >= threshold
        taxonomy = np.where(
            adverse,
            np.where(high, "ambiguous_adverse", "predictable_adverse"),
            np.where(high, "ambiguous_favorable", "predictable_favorable"),
        )
        part = frame.loc[:, ["__ts__", "side_name", "archetype_policy_key", "bad_residual_event_target"]].copy()
        part["stage"] = stage
        part["historical_difficulty_score"] = score
        part["historical_difficulty_high"] = high.astype(np.int8)
        part["outcome_distinguishability_taxonomy"] = taxonomy
        parts.append(part)
    return pd.concat(parts, ignore_index=True, copy=False), {
        "weights": {column: weight for column, weight, _ in specs},
        "reverse_orientations": [column for column, _, reverse in specs if reverse],
        "train_q75_threshold": threshold,
        "reference_min_median_max": references,
        "status": "evaluation_taxonomy_not_supervised_target",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.diagnostics / "state_distinguishability_predictions.parquet"
    feature_schema_path = args.diagnostics / "feature_schemas.csv"
    neighbor_index_path = args.diagnostics / "neighbor_training_index.parquet"
    diagnostics = pd.read_parquet(diagnostics_path)
    diagnostics["__ts__"] = pd.to_datetime(diagnostics["__ts__"], utc=True)
    group = diagnostics["side_name"].eq(GROUP[0]) & diagnostics["archetype_policy_key"].eq(GROUP[1])
    train_state = diagnostics.loc[group & diagnostics["stage"].eq("train_oof")].copy()
    eval_state = diagnostics.loc[group & diagnostics["stage"].eq("eval_oos")].copy()

    # Preserve the exact tested normalization basis: each V11 train-OOF row
    # receives its frozen timestamp context before empirical percentiles are fit.
    train_parent_path = args.v11_dir / "train_oof_predictions.parquet"
    train_parent = pd.read_parquet(train_parent_path)
    train_parent["__ts__"] = pd.to_datetime(train_parent["__ts__"], utc=True)
    train_parent = train_parent.loc[
        train_parent["side_name"].eq(GROUP[0])
        & train_parent["archetype_policy_key"].eq(GROUP[1])
    ]
    keys = ["__ts__", "side_name", "archetype_policy_key"]
    train_context = train_state.loc[:, keys + list(RISK_COLUMNS)].drop_duplicates(
        keys, keep="last"
    )
    train = train_parent.merge(
        train_context, on=keys, how="left", validate="many_to_one"
    ).dropna(subset=list(RISK_COLUMNS))

    reference_arrays = {
        column: np.sort(pd.to_numeric(train[column], errors="coerce").dropna().to_numpy(np.float32))
        for column in RISK_COLUMNS
    }
    np.savez_compressed(args.output / "normalization_references.npz", **reference_arrays)
    normalization_path = args.output / "normalization_references.npz"
    train_components, eval_components = _add_uncertainty_components(train, eval_state)
    weights = _weight_templates()[CANDIDATE["family"]]
    state_uncertainty = _uncertainty(eval_components, weights)
    weak_support = eval_components["neighbor_effective_count"].lt(
        CANDIDATE["minimum_effective_neighbor_support"]
    ).to_numpy()
    state_uncertainty[weak_support] = np.float32(CANDIDATE["missing_component_fallback"])

    parent_path = args.v11_dir / "oos_predictions.parquet"
    parent_model_path = args.v11_dir / "model__short__short_default_clean_path.txt"
    parent = pd.read_parquet(parent_path)
    parent["__ts__"] = pd.to_datetime(parent["__ts__"], utc=True)
    context = eval_components.loc[:, keys].copy()
    context["short_default_uncertainty_score"] = state_uncertainty
    context = context.drop_duplicates(keys, keep="last")
    scored = parent.merge(context, on=keys, how="left", validate="many_to_one")
    scored["short_default_uncertainty_score"] = scored["short_default_uncertainty_score"].fillna(0.0).astype(np.float32)
    parent_rank = scored["parent_rank_v9_residual_error_overlay"].to_numpy(np.float32)
    adjusted = parent_rank.copy()
    local = scored["side_name"].eq(GROUP[0]) & scored["archetype_policy_key"].eq(GROUP[1])
    adjusted[local] = _adjust_rank(
        parent_rank[local],
        scored.loc[local, "short_default_uncertainty_score"].to_numpy(np.float32),
        CANDIDATE["threshold"],
        CANDIDATE["alpha"],
    )
    scored["frozen_short_default_uncertainty_rank"] = adjusted
    output_columns = [
        "__ts__", "__symbol__", "side_name", "archetype_policy_key",
        "parent_rank_v9_residual_error_overlay", "short_default_uncertainty_score",
        "frozen_short_default_uncertainty_rank", "ev_after_1pct", "clean_exec",
    ]
    scored.loc[:, output_columns].to_parquet(
        args.output / "oos_replication_predictions.parquet", index=False, compression="zstd"
    )

    parent_metrics = _metrics(scored, parent_rank)
    challenger_metrics = _metrics(scored, adjusted)
    local_parent_metrics = _metrics(scored.loc[local], parent_rank[local])
    local_challenger_metrics = _metrics(scored.loc[local], adjusted[local])
    ev = pd.to_numeric(scored["ev_after_1pct"], errors="coerce").to_numpy(np.float32)
    day = scored["__ts__"].dt.floor("D")
    daily = pd.DataFrame(
        {
            "day": day,
            "parent_ev": np.where(parent_rank >= 0.90, ev, 0.0),
            "challenger_ev": np.where(adjusted >= 0.90, ev, 0.0),
        }
    ).groupby("day", observed=True).sum().reset_index()
    daily["delta_ev"] = daily["challenger_ev"] - daily["parent_ev"]
    daily.to_csv(args.output / "oos_replication_daily_delta.csv", index=False)
    positive_day_delta = daily.loc[daily["delta_ev"].gt(0.0), "delta_ev"].sort_values(ascending=False)
    positive_delta_sum = float(positive_day_delta.sum())
    taxonomy, taxonomy_contract = _continuous_taxonomy(train_state, eval_state)
    taxonomy.to_parquet(
        args.output / "outcome_distinguishability_taxonomy.parquet",
        index=False,
        compression="zstd",
    )
    diagnostics_manifest = args.diagnostics / "manifest.json"
    parent_manifest = args.v11_dir / "manifest.json"
    schema_rows = pd.read_csv(feature_schema_path)
    schema_row = schema_rows.loc[
        schema_rows["side_name"].eq(GROUP[0])
        & schema_rows["archetype_policy_key"].eq(GROUP[1])
        & schema_rows["stage"].eq("eval_oos")
    ]
    if len(schema_row) != 1:
        raise ValueError("Expected one final short-default neighbor feature schema")
    schema = schema_row.iloc[0]
    manifest = {
        "schema": "short_default_uncertainty_forward_challenger_v1",
        "candidate_id": "v11_short_default_requested_core_equal_q85_alpha004",
        "status": "frozen_research_challenger_not_live",
        "candidate": CANDIDATE,
        "weights": dict(zip(RISK_COLUMNS, weights.tolist(), strict=True)),
        "neighbor_contract": {
            "count": 50,
            "metric": "euclidean_on_train_robust_scaled_features",
            "kernel": "exp(-(distance/row_median_distance)^2)",
            "shrinkage": 20.0,
            "reliability": "n_eff/(n_eff+20)",
            "prior": "train_side_archetype_adverse_rate",
        },
        "normalization": "full_sorted_train_oof_empirical_references in normalization_references.npz",
        "feature_schema": {
            "hash": str(schema["feature_schema_hash"]),
            "feature_order_json": str(schema["feature_order_json"]),
            "transform_schema": str(schema["transform_schema"]),
        },
        "parent_v11": str(args.v11_dir),
        "provenance_hashes": {
            "feature_schema_hash": str(schema["feature_schema_hash"]),
            "normalization_array_hash": _sha256(normalization_path),
            "neighbor_training_index_hash": _sha256(neighbor_index_path),
            "parent_model_hash": _sha256(parent_model_path),
            "diagnostic_source_hash": _sha256(diagnostics_path),
            "parent_predictions_sha256": _sha256(parent_path),
            "parent_train_oof_predictions_sha256": _sha256(train_parent_path),
            "parent_manifest_sha256": _sha256(parent_manifest),
            "diagnostics_predictions_sha256": _sha256(diagnostics_path),
            "diagnostics_manifest_sha256": _sha256(diagnostics_manifest),
        },
        "replication": {
            "parent": parent_metrics,
            "challenger": challenger_metrics,
            "delta_mean_ev": challenger_metrics["mean_ev"] - parent_metrics["mean_ev"],
            "delta_sum_ev": challenger_metrics["sum_ev"] - parent_metrics["sum_ev"],
            "delta_clean_precision": challenger_metrics["clean_precision"] - parent_metrics["clean_precision"],
            "activity_retained": challenger_metrics["selected_rows"] / max(parent_metrics["selected_rows"], 1),
            "short_default_parent": local_parent_metrics,
            "short_default_challenger": local_challenger_metrics,
            "short_default_activity_retained": local_challenger_metrics["selected_rows"]
            / max(local_parent_metrics["selected_rows"], 1),
            "positive_delta_largest_day_share": float(positive_day_delta.iloc[0] / positive_delta_sum)
            if positive_delta_sum > 0.0 else None,
            "positive_delta_top3_day_share": float(positive_day_delta.iloc[:3].sum() / positive_delta_sum)
            if positive_delta_sum > 0.0 else None,
        },
        "forward_pass_bar": {
            "delta_total_ev": ">0", "delta_ev_per_trade": ">0",
            "delta_clean_precision": ">=0", "activity_retained": ">=0.90",
            "concentration": "gain must not be attributable to one day or one episode",
        },
        "taxonomy_contract": taxonomy_contract,
        "rejected_target": "strict predictable_adverse binary label",
        "activation": "none",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest["replication"]), indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=Path("data_perp/reports/residual_distinguishability_20260713_v5_weighted_neighbor_contract"))
    parser.add_argument("--v11-dir", type=Path, default=Path("data_perp/reports/meta_residual_event_balanced_error_overlay_20260713_v11_predicted_damage"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/short_default_uncertainty_forward_challenger_20260713_v1"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
