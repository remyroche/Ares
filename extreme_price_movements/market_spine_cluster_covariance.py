"""Pure, causal market-spine aggregation and frozen-cluster covariance features.

The module deliberately separates the one training-time operation (discovering
cluster membership) from transformation.  A :class:`FrozenSpineClusterModel`
contains all fitted state, so ``transform_market_spine_cluster_covariance``
never re-estimates clusters, signs, or weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


_EPS = 1e-12


@dataclass(frozen=True)
class MarketSpineClusterCovarianceConfig:
    """Parameters for the causal transformation.

    ``cluster_block_hours`` defaults to 60 days.  It may be set between 60
    and 90 days to use a longer consensus block without changing semantics.
    """

    level_window_hours: int = 30 * 24
    final_normalization_window_hours: int = 30 * 24
    innovation_ewma_span_hours: int = 24
    cluster_block_hours: int = 60 * 24
    min_block_observations: int = 72
    cluster_distance_threshold: float = 0.45
    cluster_count: int | None = None
    consensus_coassociation_threshold: float = 0.60
    min_stable_cluster_size: int = 2
    fast_windows_hours: tuple[int, ...] = (12, 24, 72)
    slow_window_hours: int = 30 * 24
    min_covariance_observations: int = 24

    def __post_init__(self) -> None:
        if self.level_window_hours < 2 or self.final_normalization_window_hours < 2:
            raise ValueError("normalization windows must be at least two hours")
        if self.innovation_ewma_span_hours < 1:
            raise ValueError("innovation_ewma_span_hours must be positive")
        if not 60 * 24 <= self.cluster_block_hours <= 90 * 24:
            raise ValueError("cluster_block_hours must be between 60 and 90 days")
        if not self.fast_windows_hours or any(w < 2 for w in self.fast_windows_hours):
            raise ValueError("fast_windows_hours must contain windows of at least two hours")
        if self.slow_window_hours <= max(self.fast_windows_hours):
            raise ValueError("slow_window_hours must exceed every fast window")
        if self.cluster_count is not None and self.cluster_count < 1:
            raise ValueError("cluster_count must be positive when supplied")
        if not 0.0 < self.consensus_coassociation_threshold <= 1.0:
            raise ValueError("consensus_coassociation_threshold must be in (0, 1]")
        if self.min_stable_cluster_size < 1:
            raise ValueError("min_stable_cluster_size must be positive")


@dataclass(frozen=True)
class FrozenSpineClusterModel:
    """Training-only cluster artifact required by the pure transform stage."""

    columns: tuple[str, ...]
    memberships: Mapping[str, tuple[str, ...]]
    orientations: Mapping[str, float]
    weights: Mapping[str, float]
    training_end: pd.Timestamp
    config: MarketSpineClusterCovarianceConfig


@dataclass(frozen=True)
class MarketSpineClusterCovarianceResult:
    """All intermediate causal panels plus final rolling-MAD-normalized features."""

    market_spine: pd.DataFrame
    normalized_levels: pd.DataFrame
    innovations: pd.DataFrame
    factors: pd.DataFrame
    raw_features: pd.DataFrame
    features: pd.DataFrame


def aggregate_hourly_market_spine(
    candidates: pd.DataFrame,
    source_columns: Sequence[str] | None = None,
    *,
    timestamp_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate candidate rows into robust hourly market-spine columns.

    Each source yields median, IQR, p10, p90, positive breadth, and robust
    dispersion (IQR / 1.349).  ``timestamp_col`` may be omitted for a
    DatetimeIndex.  The function does not mutate ``candidates``.
    """

    if candidates.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))
    frame = candidates.copy(deep=False)
    if timestamp_col is None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("candidates requires timestamp_col or a DatetimeIndex")
        timestamps = pd.DatetimeIndex(frame.index)
    else:
        if timestamp_col not in frame:
            raise KeyError(f"missing timestamp column: {timestamp_col}")
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame[timestamp_col], utc=True))
    if source_columns is None:
        excluded = {timestamp_col} if timestamp_col else set()
        source_columns = [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    columns = tuple(source_columns)
    if not columns:
        raise ValueError("source_columns must contain at least one numeric source")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise KeyError(f"missing market-spine sources: {sorted(missing)}")

    hour = timestamps.floor("h")
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    values = values.assign(__hour__=hour)
    grouped = values.groupby("__hour__", sort=True, observed=True)
    median = grouped[list(columns)].median()
    q25 = grouped[list(columns)].quantile(0.25)
    q75 = grouped[list(columns)].quantile(0.75)
    p10 = grouped[list(columns)].quantile(0.10)
    p90 = grouped[list(columns)].quantile(0.90)
    breadth = values.loc[:, columns].gt(0.0).where(values.loc[:, columns].notna()).assign(__hour__=hour)
    breadth = breadth.groupby("__hour__", sort=True, observed=True)[list(columns)].mean()

    pieces: dict[str, pd.Series] = {}
    for col in columns:
        prefix = f"mspine__{col}"
        iqr = q75[col] - q25[col]
        pieces[f"{prefix}__median"] = median[col]
        pieces[f"{prefix}__iqr"] = iqr
        pieces[f"{prefix}__p10"] = p10[col]
        pieces[f"{prefix}__p90"] = p90[col]
        pieces[f"{prefix}__breadth"] = breadth[col]
        pieces[f"{prefix}__dispersion"] = iqr / 1.349
    result = pd.DataFrame(pieces, index=median.index)
    result.index.name = "timestamp"
    return result.astype(float)


def causal_rolling_mad_normalize(
    frame: pd.DataFrame,
    window_hours: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Robust trailing normalization using only observations at or before t."""

    if frame.empty:
        return frame.copy()
    if window_hours < 2:
        raise ValueError("window_hours must be at least two")
    values = frame.astype(float).sort_index()
    required = min_periods or min(window_hours, max(12, window_hours // 5))
    median = values.rolling(window_hours, min_periods=required).median()

    # Pandas' rolling apply keeps this operation column-wise and bounded by a
    # single window, avoiding a (time x window x feature) temporary cube.
    mad = (values - median).abs().rolling(window_hours, min_periods=required).median()
    scale = (1.4826 * mad).where(lambda x: x > _EPS)
    return (values - median) / scale


def causal_normalized_innovations(
    market_spine: pd.DataFrame,
    config: MarketSpineClusterCovarianceConfig = MarketSpineClusterCovarianceConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return robust normalized levels and causal EWMA-smoothed innovations."""

    levels = causal_rolling_mad_normalize(market_spine, config.level_window_hours)
    innovations = levels.diff().ewm(
        span=config.innovation_ewma_span_hours,
        adjust=False,
        min_periods=1,
    ).mean()
    return levels, innovations


def fit_market_spine_cluster_model(
    market_spine: pd.DataFrame,
    training_end: pd.Timestamp | str,
    config: MarketSpineClusterCovarianceConfig = MarketSpineClusterCovarianceConfig(),
    *,
    cluster_columns: Sequence[str] | None = None,
) -> FrozenSpineClusterModel:
    """Fit consensus absolute-Spearman hierarchical memberships on training only."""

    _validate_hourly_frame(market_spine)
    cutoff = _as_timestamp(training_end, market_spine.index)
    available = market_spine.loc[:cutoff]
    if cluster_columns is None:
        cluster_columns = [c for c in available if c.endswith("__median")]
        if not cluster_columns:
            cluster_columns = list(available.columns)
    columns = tuple(cluster_columns)
    if len(columns) == 0:
        raise ValueError("no columns available for cluster discovery")
    if set(columns).difference(available.columns):
        raise KeyError("cluster_columns must be market_spine columns")

    _, innovations = causal_normalized_innovations(available.loc[:, columns], config)
    coassociation = _blockwise_consensus_coassociation(innovations, config)
    labels = _hierarchical_labels(
        coassociation,
        config,
        distance_threshold=1.0 - config.consensus_coassociation_threshold,
    )
    memberships: dict[str, tuple[str, ...]] = {}
    orientations: dict[str, float] = {}
    weights: dict[str, float] = {}
    for label in sorted(np.unique(labels)):
        members = tuple(columns[i] for i in np.flatnonzero(labels == label))
        cluster_name = f"cluster_{len(memberships):02d}"
        memberships[cluster_name] = members
        member_values = innovations.loc[:, list(members)]
        # The lowest-MAD member is a stable, training-only orientation anchor.
        scales = _median_absolute_deviation(member_values)
        anchor = scales.replace(0.0, np.nan).idxmin()
        if pd.isna(anchor):
            anchor = members[0]
        anchor_values = member_values[anchor]
        for member in members:
            corr = member_values[member].corr(anchor_values, method="spearman")
            orientations[member] = -1.0 if pd.notna(corr) and corr < 0.0 else 1.0
        inv_scale = 1.0 / scales.replace(0.0, np.nan)
        inv_scale = inv_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        inv_scale /= float(inv_scale.sum())
        weights.update({member: float(inv_scale[member]) for member in members})
    return FrozenSpineClusterModel(
        columns=columns,
        memberships=memberships,
        orientations=orientations,
        weights=weights,
        training_end=cutoff,
        config=config,
    )


def transform_market_spine_cluster_covariance(
    market_spine: pd.DataFrame,
    model: FrozenSpineClusterModel,
) -> MarketSpineClusterCovarianceResult:
    """Transform an hourly spine with frozen memberships; no fitting occurs here."""

    _validate_hourly_frame(market_spine)
    missing = set(model.columns).difference(market_spine.columns)
    if missing:
        raise KeyError(f"market_spine is missing frozen cluster columns: {sorted(missing)}")
    spine = market_spine.sort_index().copy()
    levels, all_innovations = causal_normalized_innovations(spine, model.config)
    innovations = all_innovations.loc[:, list(model.columns)]
    factors = _oriented_weighted_median_factors(innovations, model)
    raw_features = _covariance_features(factors, innovations, model)
    normalized_features = causal_rolling_mad_normalize(
        raw_features,
        model.config.final_normalization_window_hours,
    )
    return MarketSpineClusterCovarianceResult(
        market_spine=spine,
        normalized_levels=levels,
        innovations=innovations,
        factors=factors,
        raw_features=raw_features,
        features=normalized_features,
    )


def build_market_spine_cluster_covariance_features(
    candidates: pd.DataFrame,
    training_end: pd.Timestamp | str,
    source_columns: Sequence[str] | None = None,
    *,
    timestamp_col: str | None = None,
    config: MarketSpineClusterCovarianceConfig = MarketSpineClusterCovarianceConfig(),
    cluster_columns: Sequence[str] | None = None,
) -> tuple[FrozenSpineClusterModel, MarketSpineClusterCovarianceResult]:
    """Convenience pipeline: aggregate, fit frozen training clusters, transform all rows."""

    spine = aggregate_hourly_market_spine(candidates, source_columns, timestamp_col=timestamp_col)
    model = fit_market_spine_cluster_model(spine, training_end, config, cluster_columns=cluster_columns)
    return model, transform_market_spine_cluster_covariance(spine, model)


def _blockwise_consensus_coassociation(
    innovations: pd.DataFrame,
    config: MarketSpineClusterCovarianceConfig,
) -> np.ndarray:
    """Cluster each 60--90d block, then aggregate pair co-association rates.

    This is intentionally not an average of correlation magnitudes: a pair
    must repeatedly land in the same *block-level* hierarchical cluster to
    survive the consensus threshold used by the final clustering pass.
    """

    n_columns = innovations.shape[1]
    together = np.zeros((n_columns, n_columns), dtype=float)
    observed = np.zeros((n_columns, n_columns), dtype=float)
    for start in range(0, len(innovations), config.cluster_block_hours):
        block = innovations.iloc[start : start + config.cluster_block_hours]
        if len(block) < config.min_block_observations:
            continue
        corr = block.corr(method="spearman").abs().to_numpy(dtype=float)
        valid = np.isfinite(corr)
        # Missing pairwise correlations are never treated as evidence that a
        # pair is separate.  They simply receive no consensus vote.  The
        # finite zero placeholder is used only to make hierarchy construction
        # well-defined; `valid` below prevents it from influencing consensus.
        labels = _hierarchical_labels(np.where(valid, corr, 0.0), config)
        same_cluster = labels[:, None] == labels[None, :]
        together[valid] += same_cluster[valid]
        observed[valid] += 1.0
    coassociation = np.divide(together, observed, out=np.zeros_like(together), where=observed > 0)
    np.fill_diagonal(coassociation, 1.0)
    return np.clip((coassociation + coassociation.T) / 2.0, 0.0, 1.0)


def _hierarchical_labels(
    similarity: np.ndarray,
    config: MarketSpineClusterCovarianceConfig,
    *,
    distance_threshold: float | None = None,
) -> np.ndarray:
    n_columns = similarity.shape[0]
    if n_columns == 1:
        return np.zeros(1, dtype=int)
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:  # pragma: no cover - project depends on scipy
        raise ImportError("scipy is required for hierarchical cluster discovery") from exc
    distance = 1.0 - similarity
    distance = np.maximum(distance, 0.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    # ``cluster_count`` is useful for the individual block partitions.  The
    # final pass is deliberately thresholded by co-association stability.
    if config.cluster_count is not None and distance_threshold is None:
        return fcluster(tree, t=min(config.cluster_count, n_columns), criterion="maxclust") - 1
    threshold = config.cluster_distance_threshold if distance_threshold is None else distance_threshold
    labels = fcluster(tree, t=threshold, criterion="distance") - 1
    # Consensus cannot create a fragile pair: groups below the configured
    # stable size remain singleton factors rather than implicit clusters.
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        if len(positions) < config.min_stable_cluster_size:
            labels[positions] = np.arange(n_columns, n_columns + len(positions))
    _, normalized = np.unique(labels, return_inverse=True)
    return normalized


def _oriented_weighted_median_factors(
    innovations: pd.DataFrame,
    model: FrozenSpineClusterModel,
) -> pd.DataFrame:
    factors: dict[str, np.ndarray] = {}
    for cluster_name, members in model.memberships.items():
        values = innovations.loc[:, list(members)].to_numpy(dtype=float)
        signs = np.array([model.orientations[m] for m in members], dtype=float)
        weights = np.array([model.weights[m] for m in members], dtype=float)
        factors[f"mspine_factor__{cluster_name}"] = _weighted_median_rows(values * signs, weights)
    return pd.DataFrame(factors, index=innovations.index)


def _weighted_median_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=float)
    for row_no, row in enumerate(values):
        valid = np.isfinite(row)
        if not valid.any():
            continue
        x = row[valid]
        w = weights[valid]
        order = np.argsort(x, kind="stable")
        x, w = x[order], w[order]
        result[row_no] = x[np.searchsorted(np.cumsum(w), 0.5 * w.sum(), side="left")]
    return result


def _covariance_features(
    factors: pd.DataFrame,
    innovations: pd.DataFrame,
    model: FrozenSpineClusterModel,
) -> pd.DataFrame:
    if factors.empty:
        return pd.DataFrame(index=factors.index)
    config = model.config
    slow = _rolling_covariance_state(factors, config.slow_window_hours, config.min_covariance_observations)
    raw: dict[str, np.ndarray] = {}
    global_factor = _weighted_median_rows(factors.to_numpy(dtype=float), np.ones(factors.shape[1]))
    for fast_window in config.fast_windows_hours:
        fast = _rolling_covariance_state(factors, fast_window, min(config.min_covariance_observations, fast_window))
        suffix = f"{fast_window}h_vs_{config.slow_window_hours}h"
        raw[f"mspine_cov__global__scale_ratio__{suffix}"] = fast["scale"] / slow["scale"] - 1.0
        raw[f"mspine_cov__global__corr_frobenius__{suffix}"] = _matrix_difference_norm(fast["corr"], slow["corr"])
        raw[f"mspine_cov__global__coherence_drop__{suffix}"] = fast["coherence"] - slow["coherence"]
        raw[f"mspine_cov__global__pc1_evr_drop__{suffix}"] = fast["pc1_evr"] - slow["pc1_evr"]
        raw[f"mspine_cov__global__loading_angle__{suffix}"] = _loading_angle(fast["loading"], slow["loading"])
        raw[f"mspine_cov__global__effective_rank_drop__{suffix}"] = fast["effective_rank"] - slow["effective_rank"]
        for col_no, factor_name in enumerate(factors.columns):
            factor = factors.iloc[:, col_no]
            raw[f"mspine_cov__{factor_name}__scale_ratio__{suffix}"] = (
                np.sqrt(fast["variances"][:, col_no]) / np.sqrt(slow["variances"][:, col_no]) - 1.0
            )
            fast_corr = _rolling_pairwise_correlation(factor, pd.Series(global_factor, index=factors.index), fast_window, min(config.min_covariance_observations, fast_window))
            slow_corr = _rolling_pairwise_correlation(factor, pd.Series(global_factor, index=factors.index), config.slow_window_hours, config.min_covariance_observations)
            raw[f"mspine_cov__{factor_name}__global_corr_break__{suffix}"] = fast_corr - slow_corr
        for cluster_name, members in model.memberships.items():
            member_innovations = innovations.loc[:, list(members)].copy()
            member_innovations *= np.array([model.orientations[m] for m in members])
            member_fast = _rolling_covariance_state(
                member_innovations,
                fast_window,
                min(config.min_covariance_observations, fast_window),
            )
            member_slow = _rolling_covariance_state(
                member_innovations,
                config.slow_window_hours,
                config.min_covariance_observations,
            )
            prefix = f"mspine_cov__{cluster_name}__internal"
            raw[f"{prefix}_scale_ratio__{suffix}"] = member_fast["scale"] / member_slow["scale"] - 1.0
            raw[f"{prefix}_corr_frobenius__{suffix}"] = _matrix_difference_norm(member_fast["corr"], member_slow["corr"])
            raw[f"{prefix}_coherence_drop__{suffix}"] = member_fast["coherence"] - member_slow["coherence"]
            raw[f"{prefix}_pc1_evr_drop__{suffix}"] = member_fast["pc1_evr"] - member_slow["pc1_evr"]
            raw[f"{prefix}_loading_angle__{suffix}"] = _loading_angle(member_fast["loading"], member_slow["loading"])
            raw[f"{prefix}_effective_rank_drop__{suffix}"] = member_fast["effective_rank"] - member_slow["effective_rank"]
    return pd.DataFrame(raw, index=factors.index).replace([np.inf, -np.inf], np.nan)


def _rolling_covariance_state(frame: pd.DataFrame, window: int, min_periods: int) -> dict[str, np.ndarray]:
    x = frame.to_numpy(dtype=float)
    n_times, n_factors = x.shape
    covariance = np.full((n_times, n_factors, n_factors), np.nan, dtype=float)
    for i in range(n_factors):
        for j in range(i, n_factors):
            cov = _rolling_pairwise_covariance(frame.iloc[:, i], frame.iloc[:, j], window, min_periods)
            covariance[:, i, j] = covariance[:, j, i] = cov
    variances = np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0)
    denominator = np.sqrt(variances[:, :, None] * variances[:, None, :])
    correlation = covariance / denominator
    for i in range(n_factors):
        correlation[:, i, i] = np.where(variances[:, i] > _EPS, 1.0, np.nan)
    finite_variances = np.isfinite(variances)
    variance_count = finite_variances.sum(axis=1)
    mean_variance = np.divide(
        np.where(finite_variances, variances, 0.0).sum(axis=1),
        variance_count,
        out=np.full(n_times, np.nan),
        where=variance_count > 0,
    )
    scale = np.sqrt(mean_variance)
    coherence, evr, loading, effective_rank = _spectral_statistics(correlation)
    return {
        "variances": variances,
        "corr": correlation,
        "scale": scale,
        "coherence": coherence,
        "pc1_evr": evr,
        "loading": loading,
        "effective_rank": effective_rank,
    }


def _rolling_pairwise_covariance(x: pd.Series, y: pd.Series, window: int, min_periods: int) -> np.ndarray:
    valid = x.notna() & y.notna()
    xv, yv = x.where(valid, 0.0), y.where(valid, 0.0)
    count = valid.astype(float).rolling(window, min_periods=min_periods).sum()
    sx = xv.rolling(window, min_periods=min_periods).sum()
    sy = yv.rolling(window, min_periods=min_periods).sum()
    sxy = (xv * yv).rolling(window, min_periods=min_periods).sum()
    covariance = sxy / count - (sx / count) * (sy / count)
    return covariance.where(count > 1).to_numpy(dtype=float)


def _rolling_pairwise_correlation(x: pd.Series, y: pd.Series, window: int, min_periods: int) -> np.ndarray:
    cov = _rolling_pairwise_covariance(x, y, window, min_periods)
    vx = _rolling_pairwise_covariance(x, x, window, min_periods)
    vy = _rolling_pairwise_covariance(y, y, window, min_periods)
    return cov / np.sqrt(vx * vy)


def _spectral_statistics(correlation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_times, n_factors, _ = correlation.shape
    coherence = np.full(n_times, np.nan)
    pc1_evr = np.full(n_times, np.nan)
    loading = np.full((n_times, n_factors), np.nan)
    effective_rank = np.full(n_times, np.nan)
    for t, matrix in enumerate(correlation):
        if not np.isfinite(matrix).all():
            continue
        off_diagonal = matrix[~np.eye(n_factors, dtype=bool)]
        coherence[t] = float(np.mean(np.abs(off_diagonal))) if len(off_diagonal) else 0.0
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        total = eigenvalues.sum()
        if total <= _EPS:
            continue
        probabilities = eigenvalues / total
        pc1_evr[t] = probabilities[-1]
        loading[t] = eigenvectors[:, -1]
        positive = probabilities[probabilities > _EPS]
        effective_rank[t] = float(np.exp(-(positive * np.log(positive)).sum()))
    return coherence, pc1_evr, loading, effective_rank


def _matrix_difference_norm(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    result = np.full(len(fast), np.nan)
    for t, (left, right) in enumerate(zip(fast, slow)):
        if np.isfinite(left).all() and np.isfinite(right).all():
            result[t] = np.linalg.norm(left - right, ord="fro") / np.sqrt(left.size)
    return result


def _loading_angle(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    result = np.full(len(fast), np.nan)
    for t, (left, right) in enumerate(zip(fast, slow)):
        if np.isfinite(left).all() and np.isfinite(right).all():
            cosine = np.clip(abs(np.dot(left, right)) / (np.linalg.norm(left) * np.linalg.norm(right)), 0.0, 1.0)
            result[t] = np.arccos(cosine)
    return result


def _median_absolute_deviation(frame: pd.DataFrame) -> pd.Series:
    return (frame - frame.median()).abs().median()


def _validate_hourly_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("market_spine must have a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("market_spine index must be sorted ascending")
    if frame.index.has_duplicates:
        raise ValueError("market_spine index must be unique")


def _as_timestamp(value: pd.Timestamp | str, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if index.tz is not None and timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(index.tz)
    elif index.tz is None and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    elif index.tz is not None:
        timestamp = timestamp.tz_convert(index.tz)
    if timestamp < index.min():
        raise ValueError("training_end precedes all market_spine observations")
    return timestamp


__all__ = [
    "FrozenSpineClusterModel",
    "MarketSpineClusterCovarianceConfig",
    "MarketSpineClusterCovarianceResult",
    "aggregate_hourly_market_spine",
    "build_market_spine_cluster_covariance_features",
    "causal_normalized_innovations",
    "causal_rolling_mad_normalize",
    "fit_market_spine_cluster_model",
    "transform_market_spine_cluster_covariance",
]
