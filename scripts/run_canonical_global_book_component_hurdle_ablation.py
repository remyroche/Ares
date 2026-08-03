#!/usr/bin/env python3
"""Two-part zero-inflated B1--B4 selected-book component ablation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from scripts.run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        build_expanding_folds,
        sha256,
    )
    from scripts.run_canonical_global_book_conversion_head_ablation import (
        PRIMARY_BOOK_FRACTION,
        PRIMARY_HORIZON,
        _features,
    )
    from scripts.run_canonical_global_book_reconciled_component_ablation import (
        BANDS,
        CONTEXT_SOURCE,
        DIRECT_SOURCE,
        GEOMETRY,
        LABEL_SOURCE,
        _classifier,
        _metric_bundle,
        _prepare_component,
        _prepare_global,
        _regressor,
        _source_contract,
    )
except ModuleNotFoundError:
    from run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        build_expanding_folds,
        sha256,
    )
    from run_canonical_global_book_conversion_head_ablation import (
        PRIMARY_BOOK_FRACTION,
        PRIMARY_HORIZON,
        _features,
    )
    from run_canonical_global_book_reconciled_component_ablation import (
        BANDS,
        CONTEXT_SOURCE,
        DIRECT_SOURCE,
        GEOMETRY,
        LABEL_SOURCE,
        _classifier,
        _metric_bundle,
        _prepare_component,
        _prepare_global,
        _regressor,
        _source_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
RAW_COMPONENT_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_reconciled_component_ablation_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_component_hurdle_ablation_20260729_v1"
)
SCHEMA = "canonical_global_book_component_hurdle_ablation_v1"
EPSILON = 1e-15


def _feature_arms(
    book_features: Iterable[str], band_features: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    book = tuple(book_features)
    band = tuple(
        column
        for column in band_features
        if column != "context__global_common_ev_band_ordinal"
    )
    if set(book).intersection(band):
        raise ValueError("global and band component feature blocks overlap")
    return {
        "global_only": book,
        "band_only": band,
        "combined": (*book, *band),
    }


def _fit_models(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    features: Iterable[str],
    min_train_rows: int,
    min_conditional_rows: int,
    seed: int,
    threads: int,
    required_variants: tuple[str, ...] = (
        "hurdle_signed_mean",
        "hurdle_sign_magnitude",
    ),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    unknown = set(required_variants).difference(
        {"hurdle_signed_mean", "hurdle_sign_magnitude"}
    )
    if unknown:
        raise ValueError(f"unknown hurdle variants: {sorted(unknown)}")
    features = tuple(features)
    zero = np.zeros(len(evaluation), dtype=float)
    result = {
        "hurdle_signed_mean": zero.copy(),
        "hurdle_sign_magnitude": zero.copy(),
        "nonzero_probability": zero.copy(),
        "positive_given_nonzero_probability": np.full(len(evaluation), np.nan),
        "conditional_signed_prediction": zero.copy(),
        "conditional_magnitude_prediction": zero.copy(),
    }
    status: dict[str, Any] = {
        "training_rows": int(len(train)),
        "nonzero_training_rows": 0,
        "occurrence_status": "zero_fallback",
        "conditional_signed_status": "zero_fallback",
        "conditional_magnitude_status": "zero_fallback",
        "conditional_sign_status": "zero_fallback",
    }
    if len(train) < int(min_train_rows):
        return result, status
    y = train["target_delta"].to_numpy(dtype=float)
    nonzero = np.abs(y) > EPSILON
    status["nonzero_training_rows"] = int(nonzero.sum())
    x_train = train.loc[:, list(features)]
    x_eval = evaluation.loc[:, list(features)]
    if np.unique(nonzero.astype(np.int8)).size == 2:
        occurrence = _classifier(seed=seed + 1, threads=threads).fit(
            x_train, nonzero.astype(np.int8)
        )
        p_nonzero = np.asarray(
            occurrence.predict_proba(x_eval)[:, 1], dtype=float
        )
        status["occurrence_status"] = "fixed_compact_classifier"
    else:
        p_nonzero = np.full(
            len(evaluation), float(nonzero.mean()), dtype=float
        )
        status["occurrence_status"] = "constant_single_class"
    result["nonzero_probability"] = p_nonzero
    if int(nonzero.sum()) < int(min_conditional_rows):
        return result, status
    conditional = train.loc[nonzero].copy()
    conditional_y = conditional["target_delta"].to_numpy(dtype=float)
    x_conditional = conditional.loc[:, list(features)]
    if "hurdle_signed_mean" in required_variants:
        signed = _regressor(seed=seed + 2, threads=threads).fit(
            x_conditional, conditional_y
        )
        signed_prediction = np.asarray(
            signed.predict(x_eval), dtype=float
        )
        result["conditional_signed_prediction"] = signed_prediction
        result["hurdle_signed_mean"] = p_nonzero * signed_prediction
        status["conditional_signed_status"] = "fixed_compact_regressor"

    if "hurdle_sign_magnitude" in required_variants:
        magnitude = _regressor(seed=seed + 3, threads=threads).fit(
            x_conditional, np.abs(conditional_y)
        )
        magnitude_prediction = np.clip(
            np.asarray(magnitude.predict(x_eval), dtype=float), 0.0, None
        )
        result["conditional_magnitude_prediction"] = magnitude_prediction
        status["conditional_magnitude_status"] = "fixed_compact_regressor"
        positive = (conditional_y > 0).astype(np.int8)
        if np.unique(positive).size == 2:
            sign = _classifier(seed=seed + 4, threads=threads).fit(
                x_conditional, positive
            )
            p_positive = np.asarray(
                sign.predict_proba(x_eval)[:, 1], dtype=float
            )
            status["conditional_sign_status"] = "fixed_compact_classifier"
        else:
            p_positive = np.full(
                len(evaluation), float(positive.mean()), dtype=float
            )
            status["conditional_sign_status"] = "constant_single_class"
        result["positive_given_nonzero_probability"] = p_positive
        result["hurdle_sign_magnitude"] = (
            p_nonzero * (2.0 * p_positive - 1.0) * magnitude_prediction
        )
    return result, status


def _fit_band_arm(
    population: pd.DataFrame,
    *,
    band: str,
    arm: str,
    features: Iterable[str],
    folds: Iterable[Mapping[str, Any]],
    min_train_rows: int,
    min_conditional_rows: int,
    random_state: int,
    threads: int,
    required_variants: tuple[str, ...] = (
        "hurdle_signed_mean",
        "hurdle_sign_magnitude",
    ),
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for fold in folds:
        start = pd.Timestamp(fold["validation_start_utc"])
        end = pd.Timestamp(fold["validation_end_utc"])
        evaluation = population.loc[
            population["cohort_anchor_utc"].ge(start)
            & population["cohort_anchor_utc"].lt(end)
            & population["target_valid"].astype(bool)
        ].copy()
        if evaluation.empty:
            continue
        train = population.loc[
            population["target_valid"].astype(bool)
            & population["label_available_utc"].lt(start)
            & population["after_target_available_utc"].lt(start)
        ].copy()
        model_predictions, status = _fit_models(
            train,
            evaluation,
            features=features,
            min_train_rows=int(min_train_rows),
            min_conditional_rows=int(min_conditional_rows),
            seed=(
                int(random_state)
                + 100_000 * int(fold["fold_id"])
                + 1_000 * BANDS.index(band)
                + len(arm)
            ),
            threads=int(threads),
            required_variants=required_variants,
        )
        common = evaluation.loc[
            :,
            [
                "cohort_anchor_utc",
                "horizon_hours",
                "book_fraction",
                "target_delta",
                "target_valid",
                "delta_direct_mean_net",
                "delta_mapped_score_mean",
                "label_available_utc",
                "after_target_available_utc",
            ],
        ].copy()
        common["component_band"] = band
        common["feature_arm"] = arm
        common["fold_id"] = int(fold["fold_id"])
        common["validation_start_utc"] = start
        common["validation_end_utc"] = end
        common["training_rows"] = int(len(train))
        common["nonzero_training_rows"] = int(
            status["nonzero_training_rows"]
        )
        common["constant_component_prediction"] = (
            float(train["target_delta"].mean()) if len(train) else 0.0
        )
        common["training_max_after_target_available_utc"] = (
            train["after_target_available_utc"].max()
            if len(train)
            else pd.NaT
        )
        for name, values in model_predictions.items():
            common[name] = values
        for name, value in status.items():
            if name not in common:
                common[name] = value
        records.append(common)
    return pd.concat(records, ignore_index=True)


def _sum_components(
    component_predictions: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    key = [
        "feature_arm",
        "cohort_anchor_utc",
        "horizon_hours",
        "book_fraction",
        "fold_id",
        "validation_start_utc",
        "validation_end_utc",
    ]
    result = (
        component_predictions.groupby(key, sort=True, observed=True)
        .agg(
            target_delta=("target_delta", "sum"),
            target_valid=("target_valid", "all"),
            delta_prediction=(variant, "sum"),
            constant_delta_prediction=(
                "constant_component_prediction",
                "sum",
            ),
            delta_direct_mean_net=("delta_direct_mean_net", "first"),
            delta_mapped_score_mean=("delta_mapped_score_mean", "first"),
            label_available_utc=("label_available_utc", "first"),
            after_target_available_utc=(
                "after_target_available_utc",
                "first",
            ),
            training_rows=("training_rows", "min"),
            nonzero_training_rows=("nonzero_training_rows", "min"),
            training_max_after_target_available_utc=(
                "training_max_after_target_available_utc",
                "max",
            ),
        )
        .reset_index()
    )
    result["variant"] = variant
    result["model_name"] = (
        result["feature_arm"] + "__" + variant + "__B1_B4_sum"
    )
    result["zero_delta_prediction"] = 0.0
    result["causal_persistence_prediction"] = np.nan
    result["sign_probability"] = np.nan
    result["constant_sign_probability"] = np.nan
    return result


def _raw_component_sum(source: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(source / "oof_predictions.parquet")
    result = predictions.loc[
        predictions["horizon_hours"].eq(PRIMARY_HORIZON)
        & predictions["model_name"].eq(
            "reconciled_component_sum_B1_B4"
        )
        & predictions["target_valid"].astype(bool)
    ].copy()
    result["feature_arm"] = "combined"
    result["variant"] = "raw_regression"
    result["model_name"] = "combined__raw_regression__B1_B4_sum"
    return result


def _aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model_name, group in predictions.groupby("model_name", sort=True):
        record: dict[str, Any] = {
            "model_name": model_name,
            "feature_arm": group["feature_arm"].iloc[0],
            "variant": group["variant"].iloc[0],
            "oof_rows": int(len(group)),
            "folds": int(group["fold_id"].nunique()),
        }
        record.update(_metric_bundle(group))
        full_folds = []
        for fold_id, fold in group.groupby("fold_id", sort=True):
            if (
                fold["validation_end_utc"].iloc[0]
                - fold["validation_start_utc"].iloc[0]
                >= pd.Timedelta(days=14)
            ):
                full_folds.append(int(fold_id))
        record["latest_full_fold_id"] = (
            max(full_folds) if full_folds else -1
        )
        if full_folds:
            latest = group.loc[
                group["fold_id"].eq(max(full_folds))
            ]
            record.update(
                {
                    f"latest_full_{name}": value
                    for name, value in _metric_bundle(latest).items()
                }
            )
            record["latest_full_rows"] = int(len(latest))
        positive_ic = 0
        positive_spread = 0
        for fold_id in full_folds:
            fold_metrics = _metric_bundle(
                group.loc[group["fold_id"].eq(fold_id)]
            )
            positive_ic += int(
                fold_metrics["model_regression_rank_ic"] > 0
            )
            positive_spread += int(
                fold_metrics["top_bottom_target_spread"] > 0
            )
        record["development_positive_ic_folds"] = positive_ic
        record["development_positive_spread_folds"] = positive_spread
        records.append(record)
    return pd.DataFrame(records)


def plan(
    context_source: Path,
    label_source: Path,
    direct_source: Path,
    raw_component_source: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifests, hashes = _source_contract(
        context_source, label_source, direct_source
    )
    raw_manifest, raw_hashes = _artifact_manifest(
        raw_component_source,
        "canonical_global_book_reconciled_component_ablation_v1",
    )
    if (
        raw_manifest.get("source_panel_identity_sha256")
        != manifests["context"].get("source_panel_identity_sha256")
    ):
        raise ValueError("raw component comparator has different identity")
    arms = _feature_arms(
        _features(manifests["context"], "book"),
        _features(manifests["context"], "band"),
    )
    return {
        "action": "PLAN_ONLY_NO_TRAINING_OR_MATERIALIZATION",
        "schema": SCHEMA,
        "output": str(output),
        "source_sha256": {**hashes, **raw_hashes},
        "primary_task": {
            "horizon_hours": PRIMARY_HORIZON,
            "book_fraction": PRIMARY_BOOK_FRACTION,
        },
        "components": list(BANDS),
        "feature_arms": {
            name: list(columns) for name, columns in arms.items()
        },
        "variants": [
            "raw_regression comparator",
            "P(nonzero) * E[signed contribution | nonzero]",
            "P(nonzero) * signed probability * E[absolute magnitude | nonzero]",
        ],
        "fixed_geometry": dict(GEOMETRY),
        "minimum_rows": {
            "all_component": int(args.min_train_rows),
            "conditional_nonzero": int(args.min_conditional_rows),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context_source = Path(args.context_source)
    label_source = Path(args.label_source)
    direct_source = Path(args.direct_source)
    raw_component_source = Path(args.raw_component_source)
    output = Path(args.output_dir)
    if args.plan_only:
        return plan(
            context_source,
            label_source,
            direct_source,
            raw_component_source,
            output,
            args,
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    manifests, hashes = _source_contract(
        context_source, label_source, direct_source
    )
    raw_manifest, raw_hashes = _artifact_manifest(
        raw_component_source,
        "canonical_global_book_reconciled_component_ablation_v1",
    )
    raw_predictions_path = raw_component_source / "oof_predictions.parquet"
    if (
        raw_manifest.get("outputs_sha256", {}).get(
            raw_predictions_path.name
        )
        != sha256(raw_predictions_path)
    ):
        raise ValueError("raw component comparator prediction hash mismatch")
    book_features = _features(manifests["context"], "book")
    band_features = _features(manifests["context"], "band")
    arms = _feature_arms(book_features, band_features)
    labels = pd.read_parquet(
        label_source / "global_book_transition_labels.parquet"
    )
    book_context = pd.read_parquet(
        context_source / "global_book_context.parquet"
    )
    band_context = pd.read_parquet(
        context_source / "global_ev_band_context.parquet"
    )
    global_population = _prepare_global(
        labels, book_context, book_features
    )
    global_population = global_population.loc[
        global_population["horizon_hours"].eq(PRIMARY_HORIZON)
    ].copy()
    folds = build_expanding_folds(
        global_population,
        min_train_days=int(args.min_train_days),
        validation_days=int(args.validation_days),
    )
    component_parts: list[pd.DataFrame] = []
    for band in BANDS:
        component = _prepare_component(
            global_population,
            band_context,
            band=band,
            book_features=book_features,
            band_features=tuple(
                column
                for column in band_features
                if column != "context__global_common_ev_band_ordinal"
            ),
        )
        for arm, features in arms.items():
            component_parts.append(
                _fit_band_arm(
                    component,
                    band=band,
                    arm=arm,
                    features=features,
                    folds=folds,
                    min_train_rows=int(args.min_train_rows),
                    min_conditional_rows=int(args.min_conditional_rows),
                    random_state=int(args.random_state),
                    threads=int(args.threads),
                )
            )
    components = pd.concat(component_parts, ignore_index=True)
    sums = [
        _sum_components(components, variant=variant)
        for variant in (
            "hurdle_signed_mean",
            "hurdle_sign_magnitude",
        )
    ]
    sum_predictions = pd.concat(
        [*sums, _raw_component_sum(raw_component_source)],
        ignore_index=True,
    )
    metrics = _aggregate(sum_predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    components.to_parquet(
        temporary / "component_oof_predictions.parquet",
        index=False,
        compression="zstd",
    )
    sum_predictions.to_parquet(
        temporary / "reconciled_sum_oof_predictions.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_parquet(
        temporary / "aggregate_metrics.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_ZERO_INFLATED_COMPONENT_ABLATION_NOT_PROMOTION_ELIGIBLE",
        "promotion_eligible": False,
        "source_artifacts_sha256": {
            **hashes,
            **raw_hashes,
            str(raw_predictions_path): sha256(raw_predictions_path),
        },
        "source_panel_identity_sha256": manifests["context"].get(
            "source_panel_identity_sha256"
        ),
        "primary_task": {
            "horizon_hours": PRIMARY_HORIZON,
            "book_fraction": PRIMARY_BOOK_FRACTION,
        },
        "components": list(BANDS),
        "structural_zero_component": "B0",
        "feature_arms": {
            name: list(columns) for name, columns in arms.items()
        },
        "variants": {
            "hurdle_signed_mean": "P(nonzero) * E[signed contribution | nonzero]",
            "hurdle_sign_magnitude": "P(nonzero) * (2P(positive|nonzero)-1) * E(abs(contribution)|nonzero)",
            "raw_regression": "same-geometry reconciled comparator from immutable prior artifact",
        },
        "fixed_geometry": dict(GEOMETRY),
        "fit_contract": {
            "minimum_prior_resolved_rows": int(args.min_train_rows),
            "minimum_conditional_nonzero_rows": int(
                args.min_conditional_rows
            ),
            "minimum_history_days": int(args.min_train_days),
            "validation_days": int(args.validation_days),
            "threads": int(args.threads),
            "random_state": int(args.random_state),
            "feature_selection": "disabled",
            "hyperparameter_optimization": "disabled",
        },
        "contracts": {
            "zero_fallback": "insufficient component or conditional support produces exactly zero predicted contribution; no imputed future band state",
            "reconciliation": "B1-B4 predictions are summed; B0 remains structural zero",
            "availability": "every fitted row has actual label availability strictly before validation start",
            "scope": "component learnability only; broad-band stacking, admission, action layer and policy replay remain blocked",
        },
        "rows": {
            "component_predictions": int(len(components)),
            "sum_predictions": int(len(sum_predictions)),
        },
        "outputs_sha256": {
            path.name: sha256(path)
            for path in sorted(temporary.glob("*.parquet"))
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "component_predictions": int(len(components)),
        "sum_predictions": int(len(sum_predictions)),
        "metric_rows": int(len(metrics)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--context-source", type=Path, default=CONTEXT_SOURCE)
    result.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument("--direct-source", type=Path, default=DIRECT_SOURCE)
    result.add_argument(
        "--raw-component-source",
        type=Path,
        default=RAW_COMPONENT_SOURCE,
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--min-train-days", type=int, default=28)
    result.add_argument("--validation-days", type=int, default=14)
    result.add_argument("--min-train-rows", type=int, default=500)
    result.add_argument("--min-conditional-rows", type=int, default=120)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--random-state", type=int, default=20260729)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
