#!/usr/bin/env python3
"""Diagnose hurdle-arm performance in fixed pre-March market states."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.bootstrap_canonical_global_book_component_attribution import (
        _rank_ic,
        _tail_spread,
    )
    from scripts.run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        sha256,
    )
except ModuleNotFoundError:
    from bootstrap_canonical_global_book_component_attribution import (
        _rank_ic,
        _tail_spread,
    )
    from run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SOURCE = (
    ROOT
    / "data_perp/artifacts/canonical_global_book_conversion_context_20260729_v1"
)
HURDLE_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_component_hurdle_ablation_20260729_v1"
)
FAMILY_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_hurdle_feature_family_ablation_20260729_v2"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_hurdle_market_state_diagnosis_20260729_v2"
)
SCHEMA = "canonical_global_book_hurdle_market_state_diagnosis_v2"
CONTEXT_SCHEMA = "canonical_global_book_conversion_context_v1"
HURDLE_SCHEMA = "canonical_global_book_component_hurdle_ablation_v1"
FAMILY_SCHEMA = "canonical_global_book_hurdle_feature_family_ablation_v2"
BANDS = ("B1", "B2", "B3", "B4")
MARKET_FEATURES = (
    "context__current_band__range_24h_pct__mean",
    "context__current_band__meta_raw__volatility_zscore__mean",
    "context__current_band__trend_r2_24__mean",
    "context__current_band__jump_intensity__mean",
    "context__current_band__meta_raw__chop_score__mean",
)
MODELS = {
    "raw": "combined__raw_regression__B1_B4_sum",
    "band": "band_only__hurdle_signed_mean__B1_B4_sum",
    "combined": "combined__hurdle_sign_magnitude__B1_B4_sum",
}


def _source_contract(
    context_source: Path,
    hurdle_source: Path,
    family_source: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    context, context_hashes = _artifact_manifest(
        context_source, CONTEXT_SCHEMA
    )
    hurdle, hurdle_hashes = _artifact_manifest(
        hurdle_source, HURDLE_SCHEMA
    )
    family, family_hashes = _artifact_manifest(
        family_source, FAMILY_SCHEMA
    )
    identities = {
        context.get("source_panel_identity_sha256"),
        hurdle.get("source_panel_identity_sha256"),
        family.get("source_panel_identity_sha256"),
    }
    if len(identities) != 1:
        raise ValueError("market-state sources have different panel identities")
    context_path = context_source / "global_ev_band_context.parquet"
    prediction_path = hurdle_source / "reconciled_sum_oof_predictions.parquet"
    effect_path = family_source / "leave_one_family_out_effects.parquet"
    for manifest, path in (
        (context, context_path),
        (hurdle, prediction_path),
        (family, effect_path),
    ):
        if manifest.get("outputs_sha256", {}).get(path.name) != sha256(path):
            raise ValueError(f"source output hash mismatch: {path}")
    return (
        {"context": context, "hurdle": hurdle, "family": family},
        {
            **context_hashes,
            **hurdle_hashes,
            **family_hashes,
            str(context_path): sha256(context_path),
            str(prediction_path): sha256(prediction_path),
            str(effect_path): sha256(effect_path),
        },
    )


def _state_features(context_source: Path) -> pd.DataFrame:
    context = pd.read_parquet(
        context_source / "global_ev_band_context.parquet",
        columns=[
            "cohort_anchor_utc",
            "horizon_hours",
            "global_common_ev_band",
            *MARKET_FEATURES,
        ],
    )
    context = context.loc[
        context["horizon_hours"].eq(12)
        & context["global_common_ev_band"].isin(BANDS)
    ].copy()
    context["cohort_anchor_utc"] = pd.to_datetime(
        context["cohort_anchor_utc"], utc=True, errors="raise"
    )
    if context.duplicated(
        ["cohort_anchor_utc", "global_common_ev_band"]
    ).any():
        raise ValueError("band-state context identity is not one-to-one")
    pivot = context.pivot(
        index="cohort_anchor_utc",
        columns="global_common_ev_band",
        values=list(MARKET_FEATURES),
    )
    pivot.columns = [
        f"{band}__{feature}"
        for feature, band in pivot.columns.to_flat_index()
    ]
    expected = {
        f"{band}__{feature}"
        for band in BANDS
        for feature in MARKET_FEATURES
    }
    if set(pivot.columns) != expected:
        raise ValueError("fixed market-state feature topology is incomplete")
    return pivot.reset_index().sort_values(
        "cohort_anchor_utc", kind="stable"
    )


def _paired_predictions(hurdle_source: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(
        hurdle_source / "reconciled_sum_oof_predictions.parquet"
    )
    predictions = predictions.loc[
        predictions["horizon_hours"].eq(12)
        & predictions["model_name"].isin(MODELS.values())
        & predictions["target_valid"].astype(bool)
    ].copy()
    key = [
        "cohort_anchor_utc",
        "horizon_hours",
        "book_fraction",
        "fold_id",
        "validation_start_utc",
        "validation_end_utc",
    ]
    reference = predictions.loc[
        predictions["model_name"].eq(MODELS["raw"]),
        [
            *key,
            "target_delta",
            "delta_direct_mean_net",
        ],
    ].copy()
    for role, model in MODELS.items():
        values = predictions.loc[
            predictions["model_name"].eq(model),
            [*key, "delta_prediction"],
        ].rename(columns={"delta_prediction": f"{role}_prediction"})
        reference = reference.merge(
            values, on=key, how="inner", validate="one_to_one"
        )
    for column in (
        "cohort_anchor_utc",
        "validation_start_utc",
        "validation_end_utc",
    ):
        reference[column] = pd.to_datetime(
            reference[column], utc=True, errors="raise"
        )
    reference["month"] = reference["cohort_anchor_utc"].dt.strftime("%Y-%m")
    reference["calendar_day"] = reference[
        "cohort_anchor_utc"
    ].dt.floor("D")
    reference["full_fold"] = (
        reference["validation_end_utc"]
        - reference["validation_start_utc"]
    ).ge(pd.Timedelta(days=14))
    return reference


def _fit_fixed_states(
    features: pd.DataFrame,
    *,
    training_end_utc: pd.Timestamp,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    columns = [
        column for column in features if column != "cohort_anchor_utc"
    ]
    training = features.loc[
        features["cohort_anchor_utc"].lt(training_end_utc)
    ].copy()
    if len(training) < 500:
        raise ValueError("insufficient pre-March rows for fixed state fit")
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    train_imputed = imputer.fit_transform(training[columns])
    train_scaled = scaler.fit_transform(train_imputed)
    model = KMeans(
        n_clusters=3,
        random_state=int(seed),
        n_init=20,
        max_iter=500,
    ).fit(train_scaled)
    all_scaled = scaler.transform(imputer.transform(features[columns]))
    raw_state = model.predict(all_scaled)
    # Canonicalize state IDs by average standardized range/volatility level;
    # this uses centroids only, never outcomes or calendar labels.
    range_volatility_positions = [
        index
        for index, column in enumerate(columns)
        if "range_24h_pct" in column or "volatility_zscore" in column
    ]
    order = np.argsort(
        model.cluster_centers_[:, range_volatility_positions].mean(axis=1)
    )
    remap = {int(old): int(new) for new, old in enumerate(order)}
    assigned = features.loc[:, ["cohort_anchor_utc"]].copy()
    assigned["market_state"] = [
        f"S{remap[int(value)]}" for value in raw_state
    ]
    centroids = pd.DataFrame(
        model.cluster_centers_,
        columns=columns,
    )
    centroids["raw_cluster"] = np.arange(3)
    centroids["market_state"] = centroids["raw_cluster"].map(
        lambda value: f"S{remap[int(value)]}"
    )
    centroids = centroids.sort_values("market_state", kind="stable")
    audit = {
        "training_rows": int(len(training)),
        "training_start_utc": training["cohort_anchor_utc"].min(),
        "training_end_exclusive_utc": training_end_utc,
        "feature_columns": columns,
        "imputer_medians": dict(
            zip(columns, imputer.statistics_.astype(float))
        ),
        "robust_scaler_center": dict(
            zip(columns, scaler.center_.astype(float))
        ),
        "robust_scaler_scale": dict(
            zip(columns, scaler.scale_.astype(float))
        ),
        "cluster_id_canonicalization": "ascending mean standardized range/volatility centroid",
    }
    return assigned, centroids, audit


def _state_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["target_delta"].to_numpy(dtype=float)
    zero_mae = float(np.abs(y).mean())
    result: dict[str, float] = {"zero_mae": zero_mae}
    for role in MODELS:
        prediction = frame[f"{role}_prediction"].to_numpy(dtype=float)
        mae = float(np.abs(y - prediction).mean())
        result[f"{role}_mae"] = mae
        result[f"{role}_minus_zero_mae"] = mae - zero_mae
        result[f"{role}_rank_ic"] = _rank_ic(y, prediction)
        result[f"{role}_target_quintile_spread"] = _tail_spread(
            frame, f"{role}_prediction", "target_delta"
        )
        result[f"{role}_direct_net_quintile_spread"] = _tail_spread(
            frame, f"{role}_prediction", "delta_direct_mean_net"
        )
    return result


def _metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scopes: list[tuple[str, list[str]]] = [
        ("state", ["market_state"]),
        ("state_month", ["market_state", "month"]),
        ("state_fold", ["market_state", "fold_id"]),
    ]
    for scope, groups in scopes:
        for key, group in frame.groupby(groups, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            records.append(
                {
                    "scope": scope,
                    **dict(zip(groups, values)),
                    "rows": int(len(group)),
                    "utc_days": int(group["calendar_day"].nunique()),
                    **_state_metrics(group),
                }
            )
    return pd.DataFrame(records)


def _transitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("cohort_anchor_utc", kind="stable").copy()
    prior_time = ordered["cohort_anchor_utc"].shift(1)
    contiguous = ordered["cohort_anchor_utc"].sub(prior_time).eq(
        pd.Timedelta(hours=1)
    )
    ordered["prior_market_state"] = ordered["market_state"].shift(1)
    ordered.loc[~contiguous, "prior_market_state"] = pd.NA
    transition = (
        ordered.dropna(subset=["prior_market_state"])
        .groupby(
            ["prior_market_state", "market_state"], sort=True, observed=True
        )
        .size()
        .rename("hours")
        .reset_index()
    )
    transition["from_total_hours"] = transition.groupby(
        "prior_market_state", observed=True
    )["hours"].transform("sum")
    transition["transition_probability"] = (
        transition["hours"] / transition["from_total_hours"]
    )
    new_run = (
        ~contiguous
        | ordered["market_state"].ne(ordered["market_state"].shift(1))
    )
    ordered["run_id"] = new_run.cumsum()
    dwell = (
        ordered.groupby("run_id", sort=True)
        .agg(
            market_state=("market_state", "first"),
            start_utc=("cohort_anchor_utc", "min"),
            end_utc=("cohort_anchor_utc", "max"),
            dwell_hours=("cohort_anchor_utc", "size"),
        )
        .reset_index(drop=True)
    )
    return transition, dwell


def run(args: argparse.Namespace) -> dict[str, Any]:
    context_source = Path(args.context_source)
    hurdle_source = Path(args.hurdle_source)
    family_source = Path(args.family_source)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    manifests, hashes = _source_contract(
        context_source, hurdle_source, family_source
    )
    features = _state_features(context_source)
    predictions = _paired_predictions(hurdle_source)
    training_end = predictions["validation_start_utc"].min()
    assignments, centroids, state_audit = _fit_fixed_states(
        features,
        training_end_utc=training_end,
        seed=int(args.seed),
    )
    joined = predictions.merge(
        assignments,
        on="cohort_anchor_utc",
        how="left",
        validate="one_to_one",
    )
    if joined["market_state"].isna().any():
        raise ValueError("one or more OOF predictions lack a market state")
    metrics = _metrics_table(joined)
    all_assignments = assignments.copy()
    all_assignments["month"] = all_assignments[
        "cohort_anchor_utc"
    ].dt.strftime("%Y-%m")
    all_assignments["phase"] = np.where(
        all_assignments["cohort_anchor_utc"].lt(training_end),
        "pre_validation_fit_period",
        "oof_period",
    )
    occupancy = (
        all_assignments.groupby(
            ["phase", "market_state", "month"], sort=True
        )
        .agg(
            rows=("cohort_anchor_utc", "size"),
            utc_days=("cohort_anchor_utc", lambda values: values.dt.floor("D").nunique()),
        )
        .reset_index()
    )
    transition, dwell = _transitions(all_assignments)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    joined.to_parquet(
        temporary / "oof_state_assignments_and_predictions.parquet",
        index=False,
        compression="zstd",
    )
    all_assignments.to_parquet(
        temporary / "all_state_assignments.parquet",
        index=False,
        compression="zstd",
    )
    centroids.to_parquet(
        temporary / "fixed_state_centroids.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_parquet(
        temporary / "conditional_performance.parquet",
        index=False,
        compression="zstd",
    )
    occupancy.to_parquet(
        temporary / "state_occupancy.parquet",
        index=False,
        compression="zstd",
    )
    transition.to_parquet(
        temporary / "state_transition_matrix.parquet",
        index=False,
        compression="zstd",
    )
    dwell.to_parquet(
        temporary / "state_dwell_episodes.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "FIXED_PRE_MARCH_UNSUPERVISED_STATE_DIAGNOSIS_NOT_ROUTER",
        "promotion_eligible": False,
        "source_artifacts_sha256": hashes,
        "source_panel_identity_sha256": manifests["context"].get(
            "source_panel_identity_sha256"
        ),
        "state_fit": state_audit,
        "state_features": {
            "family": "band-local core market level only",
            "bands": list(BANDS),
            "base_fields": list(MARKET_FEATURES),
            "expanded_feature_count": len(state_audit["feature_columns"]),
            "clusters": 3,
            "algorithm": "median imputation + RobustScaler + KMeans fixed before first OOF validation",
        },
        "contracts": {
            "causality": "state imputer, scaler and centroids use only pre-March/pre-first-validation context; assignments use contemporaneous decision-time fields",
            "no_calendar": "timestamp, month, week, fold and row order are not state inputs",
            "no_outcomes": "no label, residual, realized EV, exit, MFE/MAE or recent performance field is a state input",
            "models": MODELS,
            "scope": "conditional diagnosis only; no supervised router, hard branch, admission, action layer or policy replay",
        },
        "rows": {
            "state_feature_anchors": int(len(features)),
            "all_state_assignments": int(len(all_assignments)),
            "oof_assignments": int(len(joined)),
            "conditional_metrics": int(len(metrics)),
            "dwell_episodes": int(len(dwell)),
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
        "oof_assignments": int(len(joined)),
        "conditional_metric_rows": int(len(metrics)),
        "dwell_episodes": int(len(dwell)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--context-source", type=Path, default=CONTEXT_SOURCE)
    result.add_argument("--hurdle-source", type=Path, default=HURDLE_SOURCE)
    result.add_argument("--family-source", type=Path, default=FAMILY_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--seed", type=int, default=20260729)
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
