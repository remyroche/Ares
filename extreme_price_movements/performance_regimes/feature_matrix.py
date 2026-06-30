"""Timestamp-level market-state feature matrix construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketStateFeatureMatrix:
    X: pd.DataFrame
    feature_families: dict[str, list[str]]
    missing_families: dict[str, list[str]]
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class FeatureMatrixPipelineArtifact:
    timestamp_col: str
    feature_families: dict[str, list[str]]
    aggregation_config: dict[str, list[str]]
    feature_columns: tuple[str, ...]
    training_diagnostics: pd.DataFrame


_SUPPORTED_AGGREGATIONS = {
    "mean",
    "median",
    "weighted_mean",
    "std",
    "min",
    "max",
    "q05",
    "q10",
    "q25",
    "q75",
    "q90",
    "q95",
    "skew",
    "fraction_missing",
    "breadth_above_threshold",
    "breadth_below_threshold",
    "cross_sectional_dispersion",
}


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


def _weighted_mean(values: pd.Series, weights: pd.Series | None) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x)
    if weights is None:
        return float(np.nanmean(x)) if finite.any() else np.nan
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok = finite & np.isfinite(w) & (w > 0.0)
    if not ok.any():
        return float(np.nanmean(x)) if finite.any() else np.nan
    return float(np.average(x[ok], weights=w[ok]))


def _aggregate_feature(
    group: pd.DataFrame,
    feature: str,
    aggregation: str,
    *,
    weight_col: str | None,
) -> float:
    values = pd.to_numeric(group[feature], errors="coerce")
    if aggregation == "mean":
        return float(values.mean())
    if aggregation == "median":
        return float(values.median())
    if aggregation == "weighted_mean":
        weights = group[weight_col] if weight_col and weight_col in group.columns else None
        return _weighted_mean(values, weights)
    if aggregation == "std" or aggregation == "cross_sectional_dispersion":
        return float(values.std(ddof=0))
    if aggregation == "min":
        return float(values.min())
    if aggregation == "max":
        return float(values.max())
    if aggregation == "q05":
        return float(values.quantile(0.05))
    if aggregation == "q10":
        return float(values.quantile(0.10))
    if aggregation == "q25":
        return float(values.quantile(0.25))
    if aggregation == "q75":
        return float(values.quantile(0.75))
    if aggregation == "q90":
        return float(values.quantile(0.90))
    if aggregation == "q95":
        return float(values.quantile(0.95))
    if aggregation == "skew":
        return float(values.skew())
    if aggregation == "fraction_missing":
        return float(values.isna().mean())
    if aggregation == "breadth_above_threshold":
        return float((values > 0.0).mean())
    if aggregation == "breadth_below_threshold":
        return float((values < 0.0).mean())
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _grouped_quantile(
    numeric: pd.DataFrame,
    group_key: pd.Series,
    q: float,
) -> pd.DataFrame:
    return numeric.groupby(group_key, sort=True).quantile(float(q))


def _grouped_weighted_mean(
    numeric: pd.DataFrame,
    group_key: pd.Series,
    weights: pd.Series | None,
) -> pd.DataFrame:
    if weights is None:
        return numeric.groupby(group_key, sort=True).mean()
    w = pd.to_numeric(weights, errors="coerce").astype(float)
    w = w.where(np.isfinite(w) & (w > 0.0))
    numerator = numeric.mul(w, axis=0).groupby(group_key, sort=True).sum(min_count=1)
    denominator = (
        numeric.notna()
        .astype(float)
        .mul(w.fillna(0.0), axis=0)
        .groupby(group_key, sort=True)
        .sum(min_count=1)
    )
    return numerator.divide(denominator.replace(0.0, np.nan))


def _aggregate_family_vectorized(
    work: pd.DataFrame,
    *,
    all_timestamps: pd.Index,
    family: str,
    available: Sequence[str],
    aggregations: Sequence[str],
    weight_col: str | None,
) -> tuple[dict[str, pd.Series], list[str]]:
    """Aggregate all available features for one family with grouped vector ops."""

    if not available:
        return {}, []
    features = [str(feature) for feature in available]
    numeric = work.loc[:, features].apply(pd.to_numeric, errors="coerce")
    group_key = work["__timestamp__"]
    grouped = numeric.groupby(group_key, sort=True)
    agg_frames: dict[str, pd.DataFrame] = {}
    for aggregation in dict.fromkeys(str(a) for a in aggregations):
        if aggregation == "mean":
            agg_frames[aggregation] = grouped.mean()
        elif aggregation == "median":
            agg_frames[aggregation] = grouped.median()
        elif aggregation == "weighted_mean":
            weights = work[weight_col] if weight_col and weight_col in work.columns else None
            agg_frames[aggregation] = _grouped_weighted_mean(numeric, group_key, weights)
        elif aggregation in {"std", "cross_sectional_dispersion"}:
            agg_frames[aggregation] = grouped.std(ddof=0)
        elif aggregation == "min":
            agg_frames[aggregation] = grouped.min()
        elif aggregation == "max":
            agg_frames[aggregation] = grouped.max()
        elif aggregation == "q05":
            agg_frames[aggregation] = _grouped_quantile(numeric, group_key, 0.05)
        elif aggregation == "q10":
            agg_frames[aggregation] = _grouped_quantile(numeric, group_key, 0.10)
        elif aggregation == "q25":
            agg_frames[aggregation] = _grouped_quantile(numeric, group_key, 0.25)
        elif aggregation == "q75":
            agg_frames[aggregation] = _grouped_quantile(numeric, group_key, 0.75)
        elif aggregation == "q90":
            agg_frames[aggregation] = _grouped_quantile(numeric, group_key, 0.90)
        elif aggregation == "q95":
            agg_frames[aggregation] = _grouped_quantile(numeric, group_key, 0.95)
        elif aggregation == "skew":
            agg_frames[aggregation] = grouped.skew()
        elif aggregation == "fraction_missing":
            agg_frames[aggregation] = numeric.isna().groupby(group_key, sort=True).mean()
        elif aggregation == "breadth_above_threshold":
            agg_frames[aggregation] = (numeric > 0.0).groupby(group_key, sort=True).mean()
        elif aggregation == "breadth_below_threshold":
            agg_frames[aggregation] = (numeric < 0.0).groupby(group_key, sort=True).mean()
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

    columns: dict[str, pd.Series] = {}
    generated: list[str] = []
    for feature in features:
        for aggregation in aggregations:
            col = f"{_safe_name(str(family))}__{_safe_name(feature)}__{aggregation}"
            values = agg_frames[str(aggregation)][feature].reindex(all_timestamps)
            columns[col] = values.astype(np.float32)
            generated.append(col)
    return columns, generated


def build_market_state_feature_matrix(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    feature_families: Mapping[str, Sequence[str]],
    aggregation_config: Mapping[str, Sequence[str]],
) -> MarketStateFeatureMatrix:
    """Aggregate row-level features to a causal timestamp-level matrix."""

    if timestamp_col not in frame.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_col}")
    work = frame.copy()
    work["__timestamp__"] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work = work.loc[work["__timestamp__"].notna()].sort_values("__timestamp__", kind="mergesort")
    grouped = work.groupby("__timestamp__", sort=True)
    weight_col = "sample_weight" if "sample_weight" in work.columns else None
    columns: dict[str, pd.Series] = {}
    resolved_families: dict[str, list[str]] = {}
    missing_families: dict[str, list[str]] = {}
    diag_rows: list[dict[str, object]] = []
    all_timestamps = pd.Index(sorted(grouped.groups.keys()), name=timestamp_col)
    for family, requested_raw in feature_families.items():
        requested = list(dict.fromkeys(str(c) for c in requested_raw if str(c)))
        available = [feature for feature in requested if feature in work.columns]
        missing = [feature for feature in requested if feature not in work.columns]
        aggregations = [
            str(a)
            for a in aggregation_config.get(family, aggregation_config.get("*", ["mean"]))
        ]
        unsupported = sorted(set(aggregations).difference(_SUPPORTED_AGGREGATIONS))
        if unsupported:
            raise ValueError(f"Unsupported aggregations for family {family}: {unsupported}")
        family_columns, generated = _aggregate_family_vectorized(
            work,
            all_timestamps=all_timestamps,
            family=str(family),
            available=available,
            aggregations=aggregations,
            weight_col=weight_col,
        )
        columns.update(family_columns)
        resolved_families[str(family)] = generated
        missing_families[str(family)] = missing
        diag_rows.append(
            {
                "family": str(family),
                "requested_feature_count": int(len(requested)),
                "available_feature_count": int(len(available)),
                "missing_feature_count": int(len(missing)),
                "missing_features": ",".join(missing),
                "generated_timestamp_feature_count": int(len(generated)),
            }
        )
    X = pd.DataFrame(columns, index=all_timestamps, dtype=np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
    return MarketStateFeatureMatrix(
        X=X,
        feature_families=resolved_families,
        missing_families=missing_families,
        diagnostics=pd.DataFrame(diag_rows),
    )


def fit_market_state_feature_pipeline(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    feature_families: Mapping[str, Sequence[str]],
    aggregation_config: Mapping[str, Sequence[str]],
) -> tuple[MarketStateFeatureMatrix, FeatureMatrixPipelineArtifact]:
    """Fit a fold-local timestamp feature contract and freeze output columns."""

    matrix = build_market_state_feature_matrix(
        frame,
        timestamp_col=timestamp_col,
        feature_families=feature_families,
        aggregation_config=aggregation_config,
    )
    artifact = FeatureMatrixPipelineArtifact(
        timestamp_col=str(timestamp_col),
        feature_families={
            str(k): [str(v) for v in values]
            for k, values in feature_families.items()
        },
        aggregation_config={
            str(k): [str(v) for v in values]
            for k, values in aggregation_config.items()
        },
        feature_columns=tuple(str(c) for c in matrix.X.columns),
        training_diagnostics=matrix.diagnostics.copy(),
    )
    return matrix, artifact


def apply_frozen_feature_pipeline(
    frame: pd.DataFrame,
    artifact: FeatureMatrixPipelineArtifact,
) -> MarketStateFeatureMatrix:
    """Apply a train-fold feature artifact to validation/inference rows."""

    matrix = build_market_state_feature_matrix(
        frame,
        timestamp_col=artifact.timestamp_col,
        feature_families=artifact.feature_families,
        aggregation_config=artifact.aggregation_config,
    )
    X = matrix.X.reindex(columns=list(artifact.feature_columns))
    return MarketStateFeatureMatrix(
        X=X,
        feature_families=matrix.feature_families,
        missing_families=matrix.missing_families,
        diagnostics=matrix.diagnostics,
    )
