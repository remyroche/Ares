"""Causal, multi-horizon market-regime research features.

This module deliberately produces *observable state context*, not labels or a
trading decision.  It is intended as a reusable research sidecar before a
fold-local regime discovery/classification step.  In particular, realised
trade paths, target columns, policy PnL and any post-entry outcome are refused
as inputs.

Every transform is calculated inside an exact-cadence segment.  A missing bar
starts a new segment, so no rolling calculation, lag or covariance comparison
can bridge a data gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


MULTIVIEW_REGIME_SCHEMA = "causal_multiview_regime_features_v1"
BASE_HORIZON_MINUTES: tuple[int, ...] = (60, 180, 360, 720, 1440, 2880, 4320, 10080)

# These are deliberately broad.  An outcome-derived field is not a valid
# approximation to observable state, even when it happens to be available in
# an offline training table.
FORBIDDEN_INPUT_TOKENS: tuple[str, ...] = (
    "target",
    "label",
    "outcome",
    "post_entry",
    "postentry",
    "future",
    "realized_pnl",
    "realised_pnl",
    "realized_ev",
    "realised_ev",
    "realized_outcome",
    "realised_outcome",
    "mfe",
    "mae",
    "pnl",
    "net_ev",
    "gross_ev",
    "ev_after",
    "exit",
    "timeout",
    "time_to",
    "policy_return",
    "barrier",
)


@dataclass(frozen=True)
class MultiViewRegimeConfig:
    """Configuration for causal multi-view state features.

    ``feature_columns`` should contain portable, decision-time inputs.  The
    caller may provide a narrower dependency set when a panel has many raw
    columns; dense dependence metrics are intentionally bounded by
    ``max_dependence_columns``.
    """

    timestamp_col: str = "source_utc"
    group_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] | None = None
    dependence_columns: tuple[str, ...] | None = None
    max_dependence_columns: int = 12
    include_15m_when_supported: bool = True
    robust_iqr_floor: float = 1e-6
    minimum_dependence_columns: int = 2
    liquidity_tokens: tuple[str, ...] = (
        "liquid",
        "spread",
        "amihud",
        "amivest",
        "volume",
        "rvol",
        "dollar_volume",
        "turnover",
        "illiqu",
        "depth",
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _is_forbidden(name: str) -> bool:
    lower = str(name).lower()
    return any(token in lower for token in FORBIDDEN_INPUT_TOKENS)


def _validate_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    kind: str,
) -> list[str]:
    requested = list(dict.fromkeys(str(column) for column in columns))
    missing = [column for column in requested if column not in frame.columns]
    if missing:
        raise KeyError(f"{kind} columns missing from frame: {missing[:8]}")
    forbidden = [column for column in requested if _is_forbidden(column)]
    if forbidden:
        raise ValueError(
            f"{kind} contains forbidden outcome/post-entry fields: {forbidden[:8]}"
        )
    non_numeric = [
        column
        for column in requested
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"{kind} must be numeric: {non_numeric[:8]}")
    return requested


def _infer_cadence(frame: pd.DataFrame, config: MultiViewRegimeConfig) -> pd.Timedelta:
    timestamp = pd.to_datetime(frame[config.timestamp_col], utc=True, errors="raise")
    work = frame.loc[:, list(config.group_columns)].copy() if config.group_columns else pd.DataFrame(index=frame.index)
    work["__timestamp__"] = timestamp
    deltas: list[pd.Series] = []
    if config.group_columns:
        for _, local in work.groupby(list(config.group_columns), observed=True, sort=False):
            delta = local.sort_values("__timestamp__", kind="stable")["__timestamp__"].diff()
            deltas.append(delta.loc[delta > pd.Timedelta(0)])
    else:
        delta = work["__timestamp__"].sort_values(kind="stable").diff()
        deltas.append(delta.loc[delta > pd.Timedelta(0)])
    values = pd.concat(deltas, ignore_index=True) if deltas else pd.Series(dtype="timedelta64[ns]")
    if values.empty:
        raise ValueError("at least two timestamped rows are required to infer cadence")
    counts = values.value_counts()
    cadence = pd.Timedelta(counts.index[0])
    if cadence <= pd.Timedelta(0):
        raise ValueError("inferred cadence must be positive")
    return cadence


def _horizons(cadence: pd.Timedelta, config: MultiViewRegimeConfig) -> list[tuple[int, int, str]]:
    cadence_minutes = cadence.total_seconds() / 60.0
    if cadence_minutes <= 0:
        raise ValueError("cadence must be positive")
    minutes = list(BASE_HORIZON_MINUTES)
    # 15m is valid only when it is an integer number of input bars.  We do not
    # upsample a slower panel, and a faster cadence still has to divide 15m.
    if (
        config.include_15m_when_supported
        and cadence_minutes <= 15.0
        and np.isclose(15.0 / cadence_minutes, round(15.0 / cadence_minutes))
    ):
        minutes.insert(0, 15)
    result: list[tuple[int, int, str]] = []
    for horizon in minutes:
        bars_float = horizon / cadence_minutes
        if not np.isclose(bars_float, round(bars_float)):
            continue
        bars = int(round(bars_float))
        if bars < 1:
            continue
        label = "15m" if horizon == 15 else f"{horizon // 60}h"
        result.append((horizon, bars, label))
    return result


def _segment_ids(timestamp: pd.Series, cadence: pd.Timedelta) -> np.ndarray:
    """Return segment IDs; an unexpected timestamp delta starts a new one."""

    discontinuity = timestamp.diff().ne(cadence)
    discontinuity.iloc[0] = True
    return discontinuity.cumsum().to_numpy(dtype=np.int64)


def _distribution_and_dynamics(
    values: np.ndarray,
    *,
    bars: int,
    horizon_hours: float,
    iqr_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal per-series features inside one contiguous segment."""

    series = pd.Series(values, dtype="float64")
    current = series.to_numpy(dtype=float)
    lag = series.shift(bars).to_numpy(dtype=float)
    lag2 = series.shift(2 * bars).to_numpy(dtype=float)
    delta = current - lag
    slope = delta / max(horizon_hours, 1e-12)
    acceleration = (current - 2.0 * lag + lag2) / max(horizon_hours**2, 1e-12)
    window = bars + 1
    median = series.rolling(window=window, min_periods=window).median().to_numpy(dtype=float)
    q25 = series.rolling(window=window, min_periods=window).quantile(0.25).to_numpy(dtype=float)
    q75 = series.rolling(window=window, min_periods=window).quantile(0.75).to_numpy(dtype=float)
    iqr = q75 - q25
    robust_z = (current - median) / np.maximum(iqr, float(iqr_floor))
    # One-bar change volatility and its trailing volatility-of-volatility.
    diff = series.diff()
    realized_vol = diff.rolling(window=bars, min_periods=bars).std(ddof=0).to_numpy(dtype=float)
    vol_of_vol = pd.Series(realized_vol).rolling(window=bars, min_periods=bars).std(ddof=0).to_numpy(dtype=float)
    return delta, slope, acceleration, robust_z, iqr, realized_vol, vol_of_vol


def _dependence_metrics(
    matrix: np.ndarray,
    *,
    bars: int,
) -> dict[str, np.ndarray]:
    """Causal correlation/covariance geometry versus a preceding equal window."""

    n_rows, n_columns = matrix.shape
    names = (
        "mean_abs_corr",
        "corr_dispersion",
        "eig1_share",
        "effective_rank",
        "corr_frobenius_shift",
        "covariance_frobenius_shift",
    )
    output = {name: np.full(n_rows, np.nan, dtype=np.float32) for name in names}
    if n_columns < 2 or n_rows < 2 * bars + 1:
        return output
    returns = np.diff(matrix.astype(float, copy=False), axis=0)
    eps = 1e-12
    for end in range(2 * bars, n_rows):
        current = returns[end - bars : end]
        previous = returns[end - 2 * bars : end - bars]
        finite = np.isfinite(current).all(axis=0) & np.isfinite(previous).all(axis=0)
        if int(finite.sum()) < 2:
            continue
        now = current[:, finite]
        prior = previous[:, finite]
        if np.any(np.nanstd(now, axis=0) <= eps) or np.any(np.nanstd(prior, axis=0) <= eps):
            continue
        now_corr = np.corrcoef(now, rowvar=False)
        prior_corr = np.corrcoef(prior, rowvar=False)
        if not np.isfinite(now_corr).all() or not np.isfinite(prior_corr).all():
            continue
        upper = now_corr[np.triu_indices_from(now_corr, k=1)]
        eigenvalues = np.clip(np.linalg.eigvalsh(now_corr), 0.0, None)
        weight = eigenvalues / max(float(eigenvalues.sum()), eps)
        # Scale both covariance matrices with the same *past-and-current*
        # robust scale.  This preserves causal comparability across features.
        combined = np.vstack((prior, now))
        scale = np.nanquantile(combined, 0.75, axis=0) - np.nanquantile(combined, 0.25, axis=0)
        standardized_now = now / np.maximum(scale, eps)
        standardized_prior = prior / np.maximum(scale, eps)
        now_cov = np.cov(standardized_now, rowvar=False, ddof=0)
        prior_cov = np.cov(standardized_prior, rowvar=False, ddof=0)
        output["mean_abs_corr"][end] = float(np.mean(np.abs(upper)))
        output["corr_dispersion"][end] = float(np.std(upper, ddof=0))
        output["eig1_share"][end] = float(weight[-1])
        output["effective_rank"][end] = float(np.exp(-np.sum(weight * np.log(np.maximum(weight, eps)))))
        output["corr_frobenius_shift"][end] = float(np.linalg.norm(now_corr - prior_corr, ord="fro"))
        output["covariance_frobenius_shift"][end] = float(np.linalg.norm(now_cov - prior_cov, ord="fro"))
    return output


def _liquidity_direction(name: str) -> float:
    lower = name.lower()
    # Higher volume/depth/amivest means less stress.  The remaining proxies
    # (spread, Amihud, illiquidity) have direct stress direction.
    return -1.0 if any(token in lower for token in ("volume", "rvol", "depth", "amivest", "turnover")) else 1.0


def build_causal_multiview_regime_features(
    frame: pd.DataFrame,
    *,
    config: MultiViewRegimeConfig = MultiViewRegimeConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate causal multi-horizon regime features and audit metadata.

    The returned frame has the same index as ``frame``.  It contains only
    decision-time transformations.  The metadata records the inferred cadence,
    exact supported horizons, input fields and contiguous segment count so a
    downstream artifact can disclose this research contract.
    """

    if config.timestamp_col not in frame:
        raise KeyError(f"timestamp column missing: {config.timestamp_col}")
    group_columns = list(config.group_columns)
    missing_groups = [column for column in group_columns if column not in frame]
    if missing_groups:
        raise KeyError(f"group columns missing from frame: {missing_groups}")
    if frame.empty:
        return pd.DataFrame(index=frame.index), {
            "schema": MULTIVIEW_REGIME_SCHEMA,
            "research_only": True,
            "rows": 0,
            "feature_columns": [],
            "dependence_columns": [],
            "horizons": [],
            "segment_count": 0,
        }
    candidate_columns = (
        list(config.feature_columns)
        if config.feature_columns is not None
        else [
            column
            for column in frame.columns
            if column not in {config.timestamp_col, *group_columns}
            and pd.api.types.is_numeric_dtype(frame[column])
            and not _is_forbidden(column)
        ]
    )
    feature_columns = _validate_columns(frame, candidate_columns, kind="feature")
    if not feature_columns:
        raise ValueError("at least one observable numeric feature is required")
    dependence_candidates = (
        list(config.dependence_columns)
        if config.dependence_columns is not None
        else list(feature_columns)
    )
    dependence_columns = _validate_columns(frame, dependence_candidates, kind="dependence")
    dependence_columns = [column for column in dependence_columns if column in feature_columns]
    dependence_columns = dependence_columns[: max(0, int(config.max_dependence_columns))]

    prepared = frame.loc[:, [*group_columns, config.timestamp_col, *feature_columns]].copy()
    # Keep a positional identity rather than relying on a unique DataFrame
    # index.  Research panels often retain non-unique source indices after a
    # causal join, while their timestamp/group identity remains unique.
    prepared["__row_position__"] = np.arange(len(prepared), dtype=np.int64)
    prepared[config.timestamp_col] = pd.to_datetime(prepared[config.timestamp_col], utc=True, errors="raise")
    duplicate_key = [*group_columns, config.timestamp_col]
    if prepared.duplicated(duplicate_key).any():
        raise ValueError("timestamp/group identity must be unique for causal regime features")
    cadence = _infer_cadence(prepared, config)
    horizons = _horizons(cadence, config)
    if not horizons:
        raise ValueError("input cadence supports none of the required regime horizons")
    output_arrays: dict[str, np.ndarray] = {}

    def _write(name: str, positions: np.ndarray, values: np.ndarray) -> None:
        target = output_arrays.get(name)
        if target is None:
            target = np.full(len(frame), np.nan, dtype=np.float32)
            output_arrays[name] = target
        target[positions] = np.asarray(values, dtype=np.float32)

    liquidity_columns = [
        column for column in feature_columns
        if any(token in column.lower() for token in config.liquidity_tokens)
    ]
    segment_count = 0
    iterator = prepared.groupby(group_columns, observed=True, sort=False) if group_columns else [((), prepared)]
    for _, local in iterator:
        local = local.sort_values(config.timestamp_col, kind="stable")
        timestamp = local[config.timestamp_col].reset_index(drop=True)
        segment = _segment_ids(timestamp, cadence)
        for _, positions in pd.Series(np.arange(len(local)), index=segment).groupby(level=0, sort=False):
            pos = positions.to_numpy(dtype=np.int64)
            if not len(pos):
                continue
            segment_count += 1
            original_positions = local["__row_position__"].to_numpy(dtype=np.int64)[pos]
            values = local.iloc[pos].loc[:, feature_columns]
            dependence_values = local.iloc[pos].loc[:, dependence_columns].to_numpy(dtype=float) if dependence_columns else np.empty((len(pos), 0))
            for _minutes, bars, label in horizons:
                horizon_hours = _minutes / 60.0
                for column in feature_columns:
                    delta, slope, acceleration, robust_z, iqr, rv, vov = _distribution_and_dynamics(
                        values[column].to_numpy(dtype=float),
                        bars=bars,
                        horizon_hours=horizon_hours,
                        iqr_floor=config.robust_iqr_floor,
                    )
                    prefix = f"mv__{_safe_name(column)}"
                    _write(f"{prefix}__delta_{label}", original_positions, delta)
                    _write(f"{prefix}__slope_per_hour_{label}", original_positions, slope)
                    _write(f"{prefix}__acceleration_per_hour2_{label}", original_positions, acceleration)
                    _write(f"{prefix}__robust_z_{label}", original_positions, robust_z)
                    _write(f"{prefix}__iqr_{label}", original_positions, iqr)
                    _write(f"{prefix}__realized_vol_{label}", original_positions, rv)
                    _write(f"{prefix}__vol_of_vol_{label}", original_positions, vov)
                    if column in liquidity_columns:
                        stress = _liquidity_direction(column) * robust_z
                        _write(
                            f"mv__liquidity__{_safe_name(column)}__stress_{label}",
                            original_positions,
                            stress,
                        )
                        _write(
                            f"mv__liquidity__{_safe_name(column)}__change_{label}",
                            original_positions,
                            _liquidity_direction(column) * delta,
                        )
                metrics = _dependence_metrics(dependence_values, bars=bars)
                for name, array in metrics.items():
                    _write(f"mv__dependence__{name}_{label}", original_positions, array)
    output = pd.DataFrame(output_arrays, index=frame.index, dtype=np.float32)
    metadata = {
        "schema": MULTIVIEW_REGIME_SCHEMA,
        "research_only": True,
        "causality": "trailing exact-cadence segments only; no target, post-entry or realised outcome input",
        "timestamp_col": config.timestamp_col,
        "group_columns": group_columns,
        "cadence": str(cadence),
        "horizons": [label for _minutes, _bars, label in horizons],
        "feature_columns": feature_columns,
        "dependence_columns": dependence_columns,
        "liquidity_proxy_columns": liquidity_columns,
        "segment_count": int(segment_count),
        "output_feature_count": int(output.shape[1]),
        "rows": int(len(output)),
    }
    return output, metadata


__all__ = [
    "BASE_HORIZON_MINUTES",
    "FORBIDDEN_INPUT_TOKENS",
    "MULTIVIEW_REGIME_SCHEMA",
    "MultiViewRegimeConfig",
    "build_causal_multiview_regime_features",
]
