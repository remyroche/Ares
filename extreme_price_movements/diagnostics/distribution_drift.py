"""Retrospective distribution-shift and losing-trade neighbourhood diagnostics.

The functions in this module deliberately accept already materialized data.  They
do not fit predictive models, mutate their inputs, or infer an observation time.
For every comparison, bins and scaling parameters are fitted only on the supplied
reference sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # SciPy is an optional runtime dependency for this diagnostic.
    from scipy.stats import ks_2samp, wasserstein_distance
except ImportError:  # pragma: no cover - exercised only in reduced environments.
    ks_2samp = None
    wasserstein_distance = None

try:  # sklearn provides an efficient exact neighbour search when available.
    from sklearn.neighbors import NearestNeighbors
except ImportError:  # pragma: no cover - exercised only in reduced environments.
    NearestNeighbors = None


DEFAULT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_PSI_EPSILON = 1e-6
_MISSING_CATEGORY = "<MISSING>"


def benjamini_hochberg_qvalues(pvalues: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values, preserving missing values."""
    values = np.asarray(pvalues, dtype=float)
    qvalues = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return qvalues

    clipped = np.clip(values[finite], 0.0, 1.0)
    order = np.argsort(clipped)
    ranked = clipped[order]
    count = len(ranked)
    adjusted = ranked * count / np.arange(1, count + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(count, dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    qvalues[finite] = restored
    return qvalues


def numeric_distribution_shift(
    reference: Sequence[float] | np.ndarray | pd.Series,
    comparison: Sequence[float] | np.ndarray | pd.Series,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, float | int]:
    """Measure a numeric comparison using reference-fixed PSI deciles.

    `wasserstein_normalized` is scaled by the reference IQR.  A zero-IQR
    reference falls back to its standard deviation and then to one, so a
    constant reference remains comparable without division by zero.
    """
    left = _finite_values(reference)
    right = _finite_values(comparison)
    result: dict[str, float | int] = {
        "n_reference": int(len(left)),
        "n_comparison": int(len(right)),
        "ks_statistic": np.nan,
        "ks_pvalue": np.nan,
        "wasserstein": np.nan,
        "wasserstein_normalized": np.nan,
        "wasserstein_scale": np.nan,
        "mean_reference": np.nan,
        "mean_comparison": np.nan,
        "mean_shift": np.nan,
        "variance_reference": np.nan,
        "variance_comparison": np.nan,
        "variance_shift": np.nan,
        "psi": np.nan,
        "psi_bin_count": 0,
        "jensen_shannon_divergence": np.nan,
    }
    if len(left) == 0 or len(right) == 0:
        return result

    reference_mean = float(np.mean(left))
    comparison_mean = float(np.mean(right))
    reference_variance = float(np.var(left, ddof=1)) if len(left) > 1 else 0.0
    comparison_variance = float(np.var(right, ddof=1)) if len(right) > 1 else 0.0
    result.update(
        {
            "mean_reference": reference_mean,
            "mean_comparison": comparison_mean,
            "mean_shift": comparison_mean - reference_mean,
            "variance_reference": reference_variance,
            "variance_comparison": comparison_variance,
            "variance_shift": comparison_variance - reference_variance,
        }
    )

    for quantile in quantiles:
        label = _quantile_label(quantile)
        left_value = float(np.quantile(left, quantile))
        right_value = float(np.quantile(right, quantile))
        result[f"quantile_{label}_reference"] = left_value
        result[f"quantile_{label}_comparison"] = right_value
        result[f"quantile_{label}_shift"] = right_value - left_value

    if ks_2samp is not None:
        ks_result = ks_2samp(left, right, method="auto")
        result["ks_statistic"] = float(ks_result.statistic)
        result["ks_pvalue"] = float(ks_result.pvalue)
    else:
        result["ks_statistic"] = _empirical_ks_statistic(left, right)

    if wasserstein_distance is not None:
        wasserstein = float(wasserstein_distance(left, right))
    else:
        wasserstein = _quantile_wasserstein(left, right)
    reference_iqr = float(np.subtract(*np.quantile(left, [0.75, 0.25])))
    reference_scale = reference_iqr
    if not np.isfinite(reference_scale) or reference_scale <= 0.0:
        reference_scale = float(np.std(left, ddof=0))
    if not np.isfinite(reference_scale) or reference_scale <= 0.0:
        reference_scale = 1.0
    result["wasserstein"] = wasserstein
    result["wasserstein_scale"] = reference_scale
    result["wasserstein_normalized"] = wasserstein / reference_scale

    psi, bin_count = _population_stability_index(left, right)
    result["psi"] = psi
    result["psi_bin_count"] = bin_count
    pooled = np.concatenate((left, right))
    edges = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, 21)))
    if len(edges) >= 2:
        edges[0] = -np.inf
        edges[-1] = np.inf
        left_hist = np.histogram(left, bins=edges)[0].astype(np.float64)
        right_hist = np.histogram(right, bins=edges)[0].astype(np.float64)
        left_prob = left_hist / max(float(left_hist.sum()), 1.0)
        right_prob = right_hist / max(float(right_hist.sum()), 1.0)
        midpoint = 0.5 * (left_prob + right_prob)
        result["jensen_shannon_divergence"] = float(
            0.5 * (_base2_kl(left_prob, midpoint) + _base2_kl(right_prob, midpoint))
        )
    else:
        result["jensen_shannon_divergence"] = 0.0
    return result


def categorical_distribution_shift(
    reference: Sequence[Any] | pd.Series,
    comparison: Sequence[Any] | pd.Series,
) -> dict[str, float | int]:
    """Measure categorical shift with total variation and base-2 JS divergence."""
    left = _categorical_values(reference)
    right = _categorical_values(comparison)
    result: dict[str, float | int] = {
        "n_reference": int(len(left)),
        "n_comparison": int(len(right)),
        "category_count": 0,
        "total_variation": np.nan,
        "jensen_shannon_base2": np.nan,
    }
    if len(left) == 0 or len(right) == 0:
        return result

    categories = pd.Index(pd.unique(np.concatenate((left, right))))
    left_probabilities = (
        pd.Series(left).value_counts(normalize=True).reindex(categories, fill_value=0.0)
    ).to_numpy(dtype=float)
    right_probabilities = (
        pd.Series(right).value_counts(normalize=True).reindex(categories, fill_value=0.0)
    ).to_numpy(dtype=float)
    midpoint = 0.5 * (left_probabilities + right_probabilities)
    left_kl = _base2_kl(left_probabilities, midpoint)
    right_kl = _base2_kl(right_probabilities, midpoint)
    result.update(
        {
            "category_count": int(len(categories)),
            "total_variation": float(0.5 * np.abs(left_probabilities - right_probabilities).sum()),
            "jensen_shannon_base2": float(0.5 * (left_kl + right_kl)),
        }
    )
    return result


def feature_distribution_drift(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    numeric_features: Iterable[str] = (),
    categorical_features: Iterable[str] = (),
    reference_label: str = "reference",
    comparison_label: str = "comparison",
    scope: str = "all_rows",
) -> pd.DataFrame:
    """Return one row per requested feature and BH-adjust numeric KS p-values."""
    rows: list[dict[str, Any]] = []
    for feature in numeric_features:
        row: dict[str, Any] = _base_feature_row(
            feature, "numeric", reference_label, comparison_label, scope
        )
        if feature not in reference or feature not in comparison:
            row["status"] = "missing_feature"
        else:
            row["status"] = "ok"
            row.update(numeric_distribution_shift(reference[feature], comparison[feature]))
        rows.append(row)
    for feature in categorical_features:
        row = _base_feature_row(
            feature, "categorical", reference_label, comparison_label, scope
        )
        if feature not in reference or feature not in comparison:
            row["status"] = "missing_feature"
        else:
            row["status"] = "ok"
            row.update(categorical_distribution_shift(reference[feature], comparison[feature]))
        rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["ks_qvalue"] = np.nan
    numeric_mask = output["feature_type"].eq("numeric") & output["status"].eq("ok")
    output.loc[numeric_mask, "ks_qvalue"] = benjamini_hochberg_qvalues(
        output.loc[numeric_mask, "ks_pvalue"].to_numpy(dtype=float)
    )
    return output


def month_pair_feature_drift(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    reference_month: str | pd.Period | pd.Timestamp,
    comparison_month: str | pd.Period | pd.Timestamp,
    numeric_features: Iterable[str] = (),
    categorical_features: Iterable[str] = (),
    include_worst_day: bool = False,
    outcome_column: str | None = None,
) -> pd.DataFrame:
    """Compare two calendar months, optionally repeating the worst-day slice.

    The worst day is the day with the lowest sum of `outcome_column` within each
    month.  It is intentionally a retrospective diagnostic and is labelled as
    such in the returned `scope` column.
    """
    timestamps = _utc_timestamps(frame, timestamp_column)
    periods = timestamps.dt.tz_localize(None).dt.to_period("M")
    left_period = _as_month_period(reference_month)
    right_period = _as_month_period(comparison_month)
    left = frame.loc[periods.eq(left_period)]
    right = frame.loc[periods.eq(right_period)]
    reports = [
        feature_distribution_drift(
            left,
            right,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            reference_label=str(left_period),
            comparison_label=str(right_period),
        )
    ]
    if include_worst_day:
        if outcome_column is None:
            raise ValueError("outcome_column is required when include_worst_day=True")
        reports.append(
            feature_distribution_drift(
                _worst_day_rows(left, timestamp_column, outcome_column),
                _worst_day_rows(right, timestamp_column, outcome_column),
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                reference_label=str(left_period),
                comparison_label=str(right_period),
                scope="worst_day_only",
            )
        )
    return pd.concat(reports, ignore_index=True, sort=False)


def may_june_july_feature_drift(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    numeric_features: Iterable[str] = (),
    categorical_features: Iterable[str] = (),
    year: int | None = None,
    include_worst_day: bool = False,
    outcome_column: str | None = None,
) -> pd.DataFrame:
    """Return May-to-June, June-to-July, and May-to-July drift tables.

    If `year` is omitted, the data must contain exactly one year with all three
    requested months.  This avoids silently pooling different calendar years.
    """
    timestamps = _utc_timestamps(frame, timestamp_column)
    available = pd.DataFrame({"year": timestamps.dt.year, "month": timestamps.dt.month})
    complete_years = sorted(
        year_value
        for year_value, group in available.groupby("year", dropna=True)
        if {5, 6, 7}.issubset(set(group["month"].dropna().astype(int)))
    )
    if year is None:
        if len(complete_years) != 1:
            raise ValueError("year is required unless exactly one May-June-July span exists")
        year = int(complete_years[0])
    if year not in complete_years:
        raise ValueError(f"Missing at least one of May, June, or July for year {year}")

    months = {month: pd.Period(year=year, month=month, freq="M") for month in (5, 6, 7)}
    reports = [
        month_pair_feature_drift(
            frame,
            timestamp_column=timestamp_column,
            reference_month=months[left],
            comparison_month=months[right],
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            include_worst_day=include_worst_day,
            outcome_column=outcome_column,
        )
        for left, right in ((5, 6), (6, 7), (5, 7))
    ]
    return pd.concat(reports, ignore_index=True, sort=False)


@dataclass(frozen=True)
class NearestNeighborDiagnostic:
    """Per-losing-trade neighbour rows and a compact aggregate summary."""

    neighbors: pd.DataFrame
    summary: Mapping[str, float | int]


# The repository generally uses US spelling, but expose the wording in the report
# request as well for callers that prefer it.
NearestNeighbourDiagnostic = NearestNeighborDiagnostic


def nearest_neighbor_losing_trade_diagnostic(
    comparison_features: np.ndarray | pd.DataFrame,
    reference_features: np.ndarray | pd.DataFrame,
    *,
    reference_is_loss: Sequence[bool | int | float],
    comparison_is_loss: Sequence[bool | int | float] | None = None,
    reference_month: Sequence[Any] | None = None,
    comparison_month: Sequence[Any] | None = None,
    reference_episode: Sequence[Any] | None = None,
    comparison_episode: Sequence[Any] | None = None,
    reference_timestamps: Sequence[Any] | None = None,
    comparison_timestamps: Sequence[Any] | None = None,
    reference_ids: Sequence[Any] | None = None,
    comparison_ids: Sequence[Any] | None = None,
    k: int = 20,
    near_time_window: str | pd.Timedelta | None = "1h",
) -> NearestNeighborDiagnostic:
    """Describe neighbours of comparison losing trades using reference-only scaling.

    Features must be pre-entry numeric values supplied by the caller.  Median and
    IQR are fitted on finite reference rows only.  Matches with equal supplied IDs
    and timestamp pairs no farther apart than `near_time_window` are excluded.
    """
    reference = _numeric_matrix(reference_features, "reference_features")
    comparison = _numeric_matrix(comparison_features, "comparison_features")
    if reference.shape[1] != comparison.shape[1]:
        raise ValueError("reference_features and comparison_features must have equal width")
    if not 1 <= int(k) <= 20:
        raise ValueError("k must be between 1 and 20")
    k = int(k)

    reference_loss = _boolean_vector(reference_is_loss, len(reference), "reference_is_loss")
    comparison_loss = (
        np.ones(len(comparison), dtype=bool)
        if comparison_is_loss is None
        else _boolean_vector(comparison_is_loss, len(comparison), "comparison_is_loss")
    )
    ref_month = _optional_vector(reference_month, len(reference), "reference_month")
    comp_month = _optional_vector(comparison_month, len(comparison), "comparison_month")
    ref_episode = _optional_vector(reference_episode, len(reference), "reference_episode")
    comp_episode = _optional_vector(comparison_episode, len(comparison), "comparison_episode")
    ref_ids = _optional_vector(reference_ids, len(reference), "reference_ids")
    comp_ids = _optional_vector(comparison_ids, len(comparison), "comparison_ids")
    ref_times = _optional_timestamp_vector(reference_timestamps, len(reference), "reference_timestamps")
    comp_times = _optional_timestamp_vector(comparison_timestamps, len(comparison), "comparison_timestamps")

    reference_valid = np.isfinite(reference).all(axis=1)
    comparison_valid = np.isfinite(comparison).all(axis=1)
    query_positions = np.flatnonzero(comparison_valid & comparison_loss)
    reference_positions = np.flatnonzero(reference_valid)
    empty = _empty_neighbour_result(
        reference_count=len(reference),
        comparison_count=len(comparison),
        reference_valid_count=int(reference_valid.sum()),
        query_count=int(len(query_positions)),
        k=k,
    )
    if len(reference_positions) == 0 or len(query_positions) == 0:
        return empty

    reference_valid_matrix = reference[reference_positions]
    median = np.median(reference_valid_matrix, axis=0)
    iqr = np.subtract(
        np.quantile(reference_valid_matrix, 0.75, axis=0),
        np.quantile(reference_valid_matrix, 0.25, axis=0),
    )
    scale = np.where(np.isfinite(iqr) & (iqr > 0.0), iqr, 1.0)
    scaled_reference = (reference_valid_matrix - median) / scale
    scaled_queries = (comparison[query_positions] - median) / scale
    distances, local_indices = _nearest_distances(scaled_reference, scaled_queries)
    exclusion_ns = _timedelta_nanoseconds(near_time_window)

    rows: list[dict[str, Any]] = []
    for query_local, query_position in enumerate(query_positions):
        candidate_positions = reference_positions[local_indices[query_local]]
        candidate_distances = distances[query_local]
        keep = np.ones(len(candidate_positions), dtype=bool)
        if ref_ids is not None and comp_ids is not None:
            keep &= np.asarray(ref_ids[candidate_positions] != comp_ids[query_position])
        if exclusion_ns is not None and ref_times is not None and comp_times is not None:
            query_time = comp_times[query_position]
            candidate_times = ref_times[candidate_positions]
            if query_time != pd.NaT.value:
                valid_times = candidate_times != pd.NaT.value
                keep &= ~(
                    valid_times
                    & (np.abs(candidate_times - query_time) <= exclusion_ns)
                )
        candidate_positions = candidate_positions[keep][:k]
        candidate_distances = candidate_distances[keep][:k]
        rows.append(
            _neighbour_row(
                query_position,
                candidate_positions,
                candidate_distances,
                reference_loss,
                ref_month,
                comp_month[query_position] if comp_month is not None else None,
                ref_episode,
                comp_episode[query_position] if comp_episode is not None else None,
            )
        )

    detail = pd.DataFrame(rows)
    summary: dict[str, float | int] = {
        "n_reference_input": int(len(reference)),
        "n_reference_valid": int(reference_valid.sum()),
        "n_comparison_input": int(len(comparison)),
        "n_losing_queries": int(len(query_positions)),
        "k_requested": k,
        "mean_neighbor_count": float(detail["neighbor_count"].mean()),
        "mean_neighbor_loss_fraction": float(detail["neighbor_loss_fraction"].mean()),
        "mean_neighbor_same_month_fraction": float(
            detail["neighbor_same_month_fraction"].mean()
        ),
        "mean_neighbor_same_episode_fraction": float(
            detail["neighbor_same_episode_fraction"].mean()
        ),
        "mean_neighbor_distance": float(detail["mean_neighbor_distance"].mean()),
        "median_neighbor_distance": float(detail["median_neighbor_distance"].median()),
    }
    return NearestNeighborDiagnostic(neighbors=detail, summary=summary)


def _finite_values(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def _categorical_values(values: Sequence[Any] | pd.Series) -> np.ndarray:
    series = pd.Series(values, dtype="object")
    return series.where(series.notna(), _MISSING_CATEGORY).astype(str).to_numpy()


def _quantile_label(quantile: float) -> str:
    return f"p{int(round(100 * quantile)):02d}"


def _empirical_ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    grid = np.sort(np.unique(np.concatenate((left, right))))
    left_cdf = np.searchsorted(np.sort(left), grid, side="right") / len(left)
    right_cdf = np.searchsorted(np.sort(right), grid, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _quantile_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    count = max(len(left), len(right))
    probabilities = (np.arange(count, dtype=float) + 0.5) / count
    return float(np.mean(np.abs(np.quantile(left, probabilities) - np.quantile(right, probabilities))))


def _population_stability_index(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    edges = np.quantile(left, np.linspace(0.0, 1.0, 11))
    internal_edges = np.unique(edges[1:-1])
    bins = np.concatenate(([-np.inf], internal_edges, [np.inf]))
    left_counts = np.histogram(left, bins=bins)[0].astype(float)
    right_counts = np.histogram(right, bins=bins)[0].astype(float)
    left_proportions = np.clip(left_counts / len(left), _PSI_EPSILON, None)
    right_proportions = np.clip(right_counts / len(right), _PSI_EPSILON, None)
    psi = np.sum((right_proportions - left_proportions) * np.log(right_proportions / left_proportions))
    return float(psi), int(len(bins) - 1)


def _base2_kl(probabilities: np.ndarray, midpoint: np.ndarray) -> float:
    nonzero = probabilities > 0.0
    return float(np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero] / midpoint[nonzero])))


def _base_feature_row(
    feature: str,
    feature_type: str,
    reference_label: str,
    comparison_label: str,
    scope: str,
) -> dict[str, Any]:
    return {
        "feature": feature,
        "feature_type": feature_type,
        "reference_month": reference_label,
        "comparison_month": comparison_label,
        "scope": scope,
    }


def _utc_timestamps(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Missing timestamp column: {column}")
    return pd.to_datetime(frame[column], utc=True, errors="coerce")


def _as_month_period(value: str | pd.Period | pd.Timestamp) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    return pd.Period(pd.Timestamp(value), freq="M")


def _worst_day_rows(frame: pd.DataFrame, timestamp_column: str, outcome_column: str) -> pd.DataFrame:
    if outcome_column not in frame:
        raise ValueError(f"Missing outcome column: {outcome_column}")
    timestamps = _utc_timestamps(frame, timestamp_column)
    outcomes = pd.to_numeric(frame[outcome_column], errors="coerce")
    daily = pd.DataFrame({"day": timestamps.dt.floor("D"), "outcome": outcomes}).dropna()
    if daily.empty:
        return frame.iloc[0:0]
    worst_day = daily.groupby("day", sort=True)["outcome"].sum().idxmin()
    return frame.loc[timestamps.dt.floor("D").eq(worst_day)]


def _numeric_matrix(values: np.ndarray | pd.DataFrame, name: str) -> np.ndarray:
    matrix = values.to_numpy(dtype=float) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional numeric matrix")
    return matrix


def _boolean_vector(values: Sequence[bool | int | float], length: int, name: str) -> np.ndarray:
    series = pd.Series(values)
    if len(series) != length:
        raise ValueError(f"{name} length must match its feature matrix")
    return series.fillna(False).astype(bool).to_numpy()


def _optional_vector(values: Sequence[Any] | None, length: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    vector = np.asarray(values, dtype=object)
    if len(vector) != length:
        raise ValueError(f"{name} length must match its feature matrix")
    return vector


def _optional_timestamp_vector(
    values: Sequence[Any] | None, length: int, name: str
) -> np.ndarray | None:
    if values is None:
        return None
    series = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if len(series) != length:
        raise ValueError(f"{name} length must match its feature matrix")
    return series.astype("int64", copy=False).to_numpy()


def _timedelta_nanoseconds(value: str | pd.Timedelta | None) -> int | None:
    if value is None:
        return None
    delta = pd.Timedelta(value)
    if delta < pd.Timedelta(0):
        raise ValueError("near_time_window must be non-negative")
    return int(delta.value)


def _nearest_distances(reference: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(reference)
    if NearestNeighbors is not None:
        search = NearestNeighbors(n_neighbors=count, metric="euclidean")
        search.fit(reference)
        return search.kneighbors(queries, return_distance=True)
    differences = queries[:, None, :] - reference[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    indices = np.argsort(distances, axis=1)
    return np.take_along_axis(distances, indices, axis=1), indices


def _neighbour_row(
    query_position: int,
    positions: np.ndarray,
    distances: np.ndarray,
    reference_loss: np.ndarray,
    reference_month: np.ndarray | None,
    query_month: Any,
    reference_episode: np.ndarray | None,
    query_episode: Any,
) -> dict[str, Any]:
    count = len(positions)
    row: dict[str, Any] = {
        "comparison_row": int(query_position),
        "neighbor_count": int(count),
        "neighbor_loss_fraction": np.nan,
        "neighbor_same_month_fraction": np.nan,
        "neighbor_same_episode_fraction": np.nan,
        "mean_neighbor_distance": np.nan,
        "median_neighbor_distance": np.nan,
        "min_neighbor_distance": np.nan,
        "max_neighbor_distance": np.nan,
    }
    if count == 0:
        return row
    row.update(
        {
            "neighbor_loss_fraction": float(reference_loss[positions].mean()),
            "mean_neighbor_distance": float(distances.mean()),
            "median_neighbor_distance": float(np.median(distances)),
            "min_neighbor_distance": float(distances.min()),
            "max_neighbor_distance": float(distances.max()),
        }
    )
    if reference_month is not None and not pd.isna(query_month):
        row["neighbor_same_month_fraction"] = float(
            np.mean(reference_month[positions] == query_month)
        )
    if reference_episode is not None and not pd.isna(query_episode):
        row["neighbor_same_episode_fraction"] = float(
            np.mean(reference_episode[positions] == query_episode)
        )
    return row


def _empty_neighbour_result(
    *,
    reference_count: int,
    comparison_count: int,
    reference_valid_count: int,
    query_count: int,
    k: int,
) -> NearestNeighborDiagnostic:
    columns = [
        "comparison_row",
        "neighbor_count",
        "neighbor_loss_fraction",
        "neighbor_same_month_fraction",
        "neighbor_same_episode_fraction",
        "mean_neighbor_distance",
        "median_neighbor_distance",
        "min_neighbor_distance",
        "max_neighbor_distance",
    ]
    summary: dict[str, float | int] = {
        "n_reference_input": reference_count,
        "n_reference_valid": reference_valid_count,
        "n_comparison_input": comparison_count,
        "n_losing_queries": query_count,
        "k_requested": k,
        "mean_neighbor_count": np.nan,
        "mean_neighbor_loss_fraction": np.nan,
        "mean_neighbor_same_month_fraction": np.nan,
        "mean_neighbor_same_episode_fraction": np.nan,
        "mean_neighbor_distance": np.nan,
        "median_neighbor_distance": np.nan,
    }
    return NearestNeighborDiagnostic(pd.DataFrame(columns=columns), summary)


__all__ = [
    "DEFAULT_QUANTILES",
    "NearestNeighborDiagnostic",
    "NearestNeighbourDiagnostic",
    "benjamini_hochberg_qvalues",
    "categorical_distribution_shift",
    "feature_distribution_drift",
    "may_june_july_feature_drift",
    "month_pair_feature_drift",
    "nearest_neighbor_losing_trade_diagnostic",
    "numeric_distribution_shift",
]
