#!/usr/bin/env python3
"""Feature-family attribution for the March/April hurdle-arm reversal."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from scripts.run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        build_expanding_folds,
        sha256,
    )
    from scripts.run_canonical_global_book_component_hurdle_ablation import (
        _aggregate,
        _fit_band_arm,
        _sum_components,
    )
    from scripts.run_canonical_global_book_conversion_head_ablation import (
        PRIMARY_HORIZON,
        _features,
    )
    from scripts.run_canonical_global_book_reconciled_component_ablation import (
        BANDS,
        CONTEXT_SOURCE,
        DIRECT_SOURCE,
        LABEL_SOURCE,
        _metric_bundle,
        _prepare_component,
        _prepare_global,
        _source_contract,
    )
except ModuleNotFoundError:
    from run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        build_expanding_folds,
        sha256,
    )
    from run_canonical_global_book_component_hurdle_ablation import (
        _aggregate,
        _fit_band_arm,
        _sum_components,
    )
    from run_canonical_global_book_conversion_head_ablation import (
        PRIMARY_HORIZON,
        _features,
    )
    from run_canonical_global_book_reconciled_component_ablation import (
        BANDS,
        CONTEXT_SOURCE,
        DIRECT_SOURCE,
        LABEL_SOURCE,
        _metric_bundle,
        _prepare_component,
        _prepare_global,
        _source_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
HURDLE_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_component_hurdle_ablation_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_hurdle_feature_family_ablation_20260729_v2"
)
SCHEMA = "canonical_global_book_hurdle_feature_family_ablation_v2"
FAMILIES = (
    "current_geometry",
    "score_and_mapping",
    "market_state",
    "transition",
    "regime",
    "trailing_geometry",
)


def feature_family(column: str) -> str:
    name = column.lower()
    if "trailing_" in name:
        return "trailing_geometry"
    if "preentry_transition" in name:
        return "transition"
    if "regime_source" in name:
        return "regime"
    if any(
        token in name
        for token in (
            "range_24h_pct",
            "volatility_zscore",
            "trend_r2_24",
            "jump_intensity",
            "chop_score",
        )
    ):
        return "market_state"
    if any(
        token in name
        for token in (
            "base_oof_score",
            "base_rank_pct",
            "base_score_z",
            "base_group_rows",
            "base_margin_to_top",
            "mapped_direct_net",
            "__map_",
            "causal_global_mapped_ev",
        )
    ):
        return "score_and_mapping"
    return "current_geometry"


def partition_features(
    columns: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    result = {family: [] for family in FAMILIES}
    values = tuple(columns)
    for column in values:
        result[feature_family(column)].append(column)
    partition = {
        family: tuple(group) for family, group in result.items()
    }
    flattened = [
        column for family in FAMILIES for column in partition[family]
    ]
    if len(flattened) != len(values) or set(flattened) != set(values):
        raise ValueError("feature-family partition is incomplete or overlaps")
    return partition


def _period_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["cohort_anchor_utc"] = pd.to_datetime(
        frame["cohort_anchor_utc"], utc=True, errors="raise"
    )
    frame["period"] = frame["cohort_anchor_utc"].dt.strftime("%Y-%m")
    records: list[dict[str, Any]] = []
    for (model_name, period), group in frame.groupby(
        ["model_name", "period"], sort=True
    ):
        records.append(
            {
                "model_name": model_name,
                "feature_arm": group["feature_arm"].iloc[0],
                "variant": group["variant"].iloc[0],
                "period": period,
                "rows": int(len(group)),
                **_metric_bundle(group),
            }
        )
    for (model_name, fold_id), group in frame.groupby(
        ["model_name", "fold_id"], sort=True
    ):
        records.append(
            {
                "model_name": model_name,
                "feature_arm": group["feature_arm"].iloc[0],
                "variant": group["variant"].iloc[0],
                "period": f"fold_{int(fold_id)}",
                "rows": int(len(group)),
                **_metric_bundle(group),
            }
        )
    return pd.DataFrame(records)


def _lofo_effects(
    metrics: pd.DataFrame,
    periods: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    metric_columns = (
        "model_regression_mae",
        "model_regression_rank_ic",
        "top_bottom_target_spread",
        "top_bottom_direct_net_spread",
    )
    for architecture in ("band_signed", "combined_signmag"):
        full_name = next(
            name
            for name in metrics["model_name"]
            if name.startswith(f"{architecture}__full__")
        )
        full = metrics.loc[metrics["model_name"].eq(full_name)].iloc[0]
        for family in FAMILIES:
            prefix = f"{architecture}__drop_{family}__"
            dropped_name = next(
                name
                for name in metrics["model_name"]
                if name.startswith(prefix)
            )
            dropped = metrics.loc[
                metrics["model_name"].eq(dropped_name)
            ].iloc[0]
            for metric in metric_columns:
                records.append(
                    {
                        "architecture": architecture,
                        "scope": "aggregate",
                        "dropped_family": family,
                        "metric": metric,
                        "full_value": full[metric],
                        "drop_value": dropped[metric],
                        "drop_minus_full": dropped[metric] - full[metric],
                    }
                )
            for period in sorted(periods["period"].unique()):
                full_period = periods.loc[
                    periods["model_name"].eq(full_name)
                    & periods["period"].eq(period)
                ]
                drop_period = periods.loc[
                    periods["model_name"].eq(dropped_name)
                    & periods["period"].eq(period)
                ]
                if full_period.empty or drop_period.empty:
                    continue
                for metric in metric_columns:
                    records.append(
                        {
                            "architecture": architecture,
                            "scope": period,
                            "dropped_family": family,
                            "metric": metric,
                            "full_value": full_period.iloc[0][metric],
                            "drop_value": drop_period.iloc[0][metric],
                            "drop_minus_full": (
                                drop_period.iloc[0][metric]
                                - full_period.iloc[0][metric]
                            ),
                        }
                    )
    return pd.DataFrame(records)


def plan(
    context_source: Path,
    label_source: Path,
    direct_source: Path,
    hurdle_source: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifests, hashes = _source_contract(
        context_source, label_source, direct_source
    )
    hurdle_manifest, hurdle_hashes = _artifact_manifest(
        hurdle_source,
        "canonical_global_book_component_hurdle_ablation_v1",
    )
    if (
        hurdle_manifest.get("source_panel_identity_sha256")
        != manifests["context"].get("source_panel_identity_sha256")
    ):
        raise ValueError("hurdle comparator uses a different panel identity")
    book = _features(manifests["context"], "book")
    band = tuple(
        column
        for column in _features(manifests["context"], "band")
        if column != "context__global_common_ev_band_ordinal"
    )
    architectures = {
        "band_signed": {
            "variant": "hurdle_signed_mean",
            "features": band,
        },
        "combined_signmag": {
            "variant": "hurdle_sign_magnitude",
            "features": (*book, *band),
        },
    }
    return {
        "action": "PLAN_ONLY_NO_TRAINING_OR_MATERIALIZATION",
        "schema": SCHEMA,
        "output": str(output),
        "source_sha256": {**hashes, **hurdle_hashes},
        "architectures": {
            name: {
                "variant": values["variant"],
                "full_feature_count": len(values["features"]),
                "families": {
                    family: list(columns)
                    for family, columns in partition_features(
                        values["features"]
                    ).items()
                },
            }
            for name, values in architectures.items()
        },
        "contract": {
            "primary_task": "H12/global-10% only",
            "selection": "no winner selection; full plus one-family-drop diagnostics only",
            "calendar": "calendar month is an evaluation slice, never a feature or router input",
            "hpo": "disabled",
        },
        "minimum_rows": {
            "all_component": int(args.min_train_rows),
            "conditional_nonzero": int(args.min_conditional_rows),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context_source = Path(args.context_source)
    label_source = Path(args.label_source)
    direct_source = Path(args.direct_source)
    hurdle_source = Path(args.hurdle_source)
    output = Path(args.output_dir)
    if args.plan_only:
        return plan(
            context_source,
            label_source,
            direct_source,
            hurdle_source,
            output,
            args,
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    manifests, hashes = _source_contract(
        context_source, label_source, direct_source
    )
    hurdle_manifest, hurdle_hashes = _artifact_manifest(
        hurdle_source,
        "canonical_global_book_component_hurdle_ablation_v1",
    )
    hurdle_predictions = (
        hurdle_source / "reconciled_sum_oof_predictions.parquet"
    )
    if hurdle_manifest.get("outputs_sha256", {}).get(
        hurdle_predictions.name
    ) != sha256(hurdle_predictions):
        raise ValueError("hurdle source prediction hash mismatch")
    book_features = _features(manifests["context"], "book")
    band_features = tuple(
        column
        for column in _features(manifests["context"], "band")
        if column != "context__global_common_ev_band_ordinal"
    )
    architectures = {
        "band_signed": {
            "variant": "hurdle_signed_mean",
            "features": band_features,
        },
        "combined_signmag": {
            "variant": "hurdle_sign_magnitude",
            "features": (*book_features, *band_features),
        },
    }
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
    prepared_components = {
        band: _prepare_component(
            global_population,
            band_context,
            band=band,
            book_features=book_features,
            band_features=band_features,
        )
        for band in BANDS
    }
    component_parts: list[pd.DataFrame] = []
    sum_parts: list[pd.DataFrame] = []
    family_manifest: dict[str, Any] = {}
    for architecture, values in architectures.items():
        variant = str(values["variant"])
        full_features = tuple(values["features"])
        families = partition_features(full_features)
        family_manifest[architecture] = {
            family: list(columns)
            for family, columns in families.items()
        }
        arms = {"full": full_features}
        arms.update(
            {
                f"drop_{family}": tuple(
                    column
                    for column in full_features
                    if column not in set(families[family])
                )
                for family in FAMILIES
            }
        )
        for arm, features in arms.items():
            arm_name = f"{architecture}__{arm}"
            arm_components: list[pd.DataFrame] = []
            for band in BANDS:
                predicted = _fit_band_arm(
                    prepared_components[band],
                    band=band,
                    arm=arm_name,
                    features=features,
                    folds=folds,
                    min_train_rows=int(args.min_train_rows),
                    min_conditional_rows=int(args.min_conditional_rows),
                    random_state=int(args.random_state),
                    threads=int(args.threads),
                    required_variants=(variant,),
                )
                arm_components.append(predicted)
                component_parts.append(predicted)
            sum_parts.append(
                _sum_components(
                    pd.concat(arm_components, ignore_index=True),
                    variant=variant,
                )
            )
    component_predictions = pd.concat(
        component_parts, ignore_index=True
    )
    sum_predictions = pd.concat(sum_parts, ignore_index=True)
    metrics = _aggregate(sum_predictions)
    period_metrics = _period_metrics(sum_predictions)
    effects = _lofo_effects(metrics, period_metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    component_predictions.to_parquet(
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
    period_metrics.to_parquet(
        temporary / "period_metrics.parquet",
        index=False,
        compression="zstd",
    )
    effects.to_parquet(
        temporary / "leave_one_family_out_effects.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_FEATURE_FAMILY_DIAGNOSTIC_NOT_ROUTER_OR_PROMOTION",
        "promotion_eligible": False,
        "source_artifacts_sha256": {
            **hashes,
            **hurdle_hashes,
            str(hurdle_predictions): sha256(hurdle_predictions),
        },
        "source_panel_identity_sha256": manifests["context"].get(
            "source_panel_identity_sha256"
        ),
        "architectures": {
            name: {
                "variant": values["variant"],
                "full_feature_count": len(values["features"]),
                "families": family_manifest[name],
            }
            for name, values in architectures.items()
        },
        "contracts": {
            "task": "H12/global-10% exact book only",
            "ablation": "full features plus one predeclared family removed at a time; no feature selection or HPO",
            "availability": "actual target availability strictly before every validation boundary",
            "calendar": "March/April and fold are reporting slices only, never features or router labels",
            "scope": "diagnose transfer reversal; no routing, admission, action layer or policy replay",
        },
        "fit_contract": {
            "minimum_prior_resolved_rows": int(args.min_train_rows),
            "minimum_conditional_nonzero_rows": int(
                args.min_conditional_rows
            ),
            "minimum_history_days": int(args.min_train_days),
            "validation_days": int(args.validation_days),
            "threads": int(args.threads),
            "random_state": int(args.random_state),
        },
        "rows": {
            "component_predictions": int(len(component_predictions)),
            "sum_predictions": int(len(sum_predictions)),
            "lofo_effects": int(len(effects)),
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
        "component_predictions": int(len(component_predictions)),
        "sum_predictions": int(len(sum_predictions)),
        "lofo_effect_rows": int(len(effects)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--context-source", type=Path, default=CONTEXT_SOURCE)
    result.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument("--direct-source", type=Path, default=DIRECT_SOURCE)
    result.add_argument("--hurdle-source", type=Path, default=HURDLE_SOURCE)
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
