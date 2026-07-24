"""Chronological detectors for observable model-failure regimes.

The descriptive taxonomy may use realized outcomes.  This module consumes only
day-open observable state at scoring time.  Failure and mode labels are used on
strictly earlier training rows, and every threshold is derived from an inner
chronological validation slice rather than the scored interval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

from extreme_price_movements.unsupervised_regime_learning.failure_episodes import (
    validate_inference_feature_columns,
)

KEYS = ("day", "side_name", "archetype_policy_key")
LABEL_ONLY_COLUMNS = {
    "adverse_event",
    "event_block",
    "failure_mode",
    "failure_mode_available_day",
    "negative_pnl_day",
    "mean_ev_after_1pct",
    "selected_rows",
    "clean_exec_rate",
    "clean_exec_precision",
    "signed_surprise",
    "persistence_strength",
    "large_event_strength",
}
LABEL_AVAILABILITY_PREFIX = "availability__"
OUTCOME_RESOLUTION_DAYS = 1
FAILURE_MODE_RECOVERY_HORIZON_DAYS = 14
BATCH_LAYOUT_DEPENDENT_AE_GMM_TOKENS = (
    "cluster_speed",
    "cluster_acceleration",
    "gmm_posterior_delta_1",
    "gmm_posterior_accel_1",
    "dae_reconstruction_error_delta_1",
    "dae_reconstruction_error_accel_1",
    "latent_speed",
    "latent_acceleration",
    "cluster_entropy_delta_1",
    "cluster_entropy_accel_1",
)


@dataclass(frozen=True)
class ProspectiveFailureDetectorConfig:
    min_train_days: int = 120
    eval_days: int = 45
    inner_validation_days: int = 35
    min_positive_days: int = 5
    max_features: int = 20
    mi_bins: int = 8
    alert_quantile: float = 0.95
    probability_calibration: str = "platt"
    lead_days: tuple[int, ...] = (1, 3)
    embargo_days: int = 0
    evaluation_start: str = ""
    random_state: int = 20260719


@dataclass
class FrozenFailureDetector:
    """Serializable causal detector for one side x archetype cell.

    The chronological experiment intentionally only writes OOS predictions.
    This compact bundle is the corresponding forward-scoring contract: feature
    selection, robust scaling, Platt calibration and threshold are frozen from
    label-available history.  It never substitutes missing state inputs.
    """

    side_name: str
    archetype_policy_key: str
    target: str
    selected_features: list[str]
    median: np.ndarray
    scale: np.ndarray
    model_string: str
    threshold: float
    calibration_method: str
    platt_coef: float | None
    platt_intercept: float | None
    train_boundary: str
    train_rows: int
    train_positive_days: int

    def score(self, state: pd.DataFrame) -> pd.DataFrame:
        """Score fully observable state rows; incomplete rows stay unscored."""

        missing = [name for name in self.selected_features if name not in state]
        if missing:
            raise KeyError(
                "Frozen failure detector state is missing required features: "
                + ", ".join(missing[:12])
            )
        values = (
            state.loc[:, self.selected_features]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        complete = np.isfinite(values).all(axis=1)
        risk = np.full(len(state), np.nan, dtype=np.float32)
        if complete.any():
            normalized = np.clip(
                (values[complete] - self.median) / self.scale,
                -8.0,
                8.0,
            ).astype(np.float32, copy=False)
            model = lgb.Booster(model_str=self.model_string)
            raw = np.asarray(model.predict(normalized), dtype=np.float64)
            if self.calibration_method == "platt_logit_inner_validation":
                assert self.platt_coef is not None and self.platt_intercept is not None
                logits = _logit(raw)
                risk[complete] = (1.0 / (1.0 + np.exp(
                    -(self.platt_coef * logits + self.platt_intercept)
                ))).astype(np.float32)
            else:
                risk[complete] = raw.astype(np.float32)
        out = state.loc[:, [name for name in KEYS if name in state]].copy()
        out["risk"] = risk
        out["threshold"] = float(self.threshold)
        out["alert"] = pd.Series(risk >= float(self.threshold), index=out.index).where(
            np.isfinite(risk), pd.NA
        ).astype("boolean")
        out["state_complete"] = complete
        out["failure_mode"] = self.target.removeprefix("target__")
        return out


_TARGET_HORIZON_PATTERN = re.compile(r"^target__next(?P<days>\d+)d__")


def target_horizon_days(target: str) -> int:
    """Return the forward label horizon encoded in a target column name."""

    match = _TARGET_HORIZON_PATTERN.match(str(target))
    return int(match.group("days")) if match else 0


def is_batch_layout_dependent_ae_gmm_feature(name: str) -> bool:
    """Identify legacy temporal latent outputs that are not live portable."""

    normalized = str(name).casefold()
    return any(token in normalized for token in BATCH_LAYOUT_DEPENDENT_AE_GMM_TOKENS)


def purged_before_boundary(
    frame: pd.DataFrame,
    *,
    boundary: pd.Timestamp,
    target: str,
    embargo_days: int = 0,
) -> pd.DataFrame:
    """Keep labels whose full forward path ends before ``boundary``."""

    boundary = pd.Timestamp(boundary) - pd.Timedelta(
        days=max(0, int(embargo_days))
    )
    availability_column = f"{LABEL_AVAILABILITY_PREFIX}{target}"
    if availability_column in frame:
        available = pd.to_datetime(frame[availability_column], utc=True, errors="coerce")
    else:
        available = pd.to_datetime(frame["day"], utc=True) + pd.Timedelta(
            days=target_horizon_days(target) + OUTCOME_RESOLUTION_DAYS
        )
    return frame.loc[available.lt(boundary)].copy()


def _failure_onset(
    frame: pd.DataFrame,
    values: pd.Series,
) -> pd.Series:
    """Return first active day after a non-active or non-contiguous prior day."""
    active = values.astype("boolean").fillna(False).astype(bool)
    previous_active = active.groupby(
        [frame["side_name"], frame["archetype_policy_key"]],
        observed=True,
        sort=False,
    ).shift(1, fill_value=False)
    previous_day = (
        frame["day"]
        .groupby(
            [frame["side_name"], frame["archetype_policy_key"]],
            observed=True,
            sort=False,
        )
        .shift(1)
    )
    contiguous = frame["day"].sub(previous_day).eq(pd.Timedelta(days=1))
    return active & ~(previous_active & contiguous)


def _future_window_target(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    horizon_days: int,
) -> pd.Series:
    """Label whether an event occurs in days ``t+1`` through ``t+h``.

    The operation is target construction only. Rows without a complete future
    horizon are left missing so they cannot be counted as easy negatives.
    """
    horizon = int(horizon_days)
    if horizon <= 0:
        raise ValueError("horizon_days must be positive")
    output = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    active = values.astype("boolean").fillna(False).astype(bool)
    for positions in frame.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ).indices.values():
        index = np.asarray(positions, dtype=np.int64)
        days = pd.DatetimeIndex(frame.iloc[index]["day"])
        targets = active.iloc[index].to_numpy(dtype=bool, copy=False)
        max_day = days.max()
        day_to_target = {day: bool(target) for day, target in zip(days, targets)}
        local_values: list[object] = []
        for day in days:
            if day + pd.Timedelta(days=horizon) > max_day:
                local_values.append(pd.NA)
                continue
            local_values.append(
                any(
                    day_to_target.get(day + pd.Timedelta(days=offset), False)
                    for offset in range(1, horizon + 1)
                )
            )
        output.iloc[index] = pd.array(local_values, dtype="boolean")
    return output


def _future_window_maximum(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    horizon_days: int,
) -> pd.Series:
    """Return the maximum realized severity in days ``t+1`` through ``t+h``."""

    horizon = int(horizon_days)
    if horizon <= 0:
        raise ValueError("horizon_days must be positive")
    output = pd.Series(np.nan, index=frame.index, dtype=np.float32)
    numeric = pd.to_numeric(values, errors="coerce").clip(lower=0.0)
    for positions in frame.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ).indices.values():
        index = np.asarray(positions, dtype=np.int64)
        days = pd.DatetimeIndex(frame.iloc[index]["day"])
        severity = numeric.iloc[index].to_numpy(dtype=np.float32, copy=False)
        max_day = days.max()
        day_to_severity = {
            day: float(value) if np.isfinite(value) else 0.0
            for day, value in zip(days, severity)
        }
        local_values: list[float] = []
        for day in days:
            if day + pd.Timedelta(days=horizon) > max_day:
                local_values.append(np.nan)
                continue
            local_values.append(
                max(
                    day_to_severity.get(day + pd.Timedelta(days=offset), 0.0)
                    for offset in range(1, horizon + 1)
                )
            )
        output.iloc[index] = np.asarray(local_values, dtype=np.float32)
    return output


def _future_window_availability(
    frame: pd.DataFrame,
    event_requiring_classification: pd.Series,
    event_available: pd.Series,
    *,
    horizon_days: int,
) -> pd.Series:
    """Return the date on which a future-window mode label is fully known."""

    horizon = int(horizon_days)
    output = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    active = (
        event_requiring_classification.astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    available = pd.to_datetime(event_available, utc=True, errors="coerce")
    for positions in frame.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ).indices.values():
        index = np.asarray(positions, dtype=np.int64)
        days = pd.DatetimeIndex(frame.iloc[index]["day"])
        targets = active.iloc[index].to_numpy(dtype=bool, copy=False)
        availability = available.iloc[index].to_numpy(copy=False)
        for offset, day in enumerate(days):
            base = day + pd.Timedelta(days=horizon + OUTCOME_RESOLUTION_DAYS)
            future_availability = [
                pd.Timestamp(availability[position])
                for position in range(offset + 1, len(index))
                if days[position] <= day + pd.Timedelta(days=horizon)
                and targets[position]
                and not pd.isna(availability[position])
            ]
            output.iloc[index[offset]] = max([base, *future_availability])
    return output


_DISTRIBUTION_SHIFT_TOKENS = (
    "gmm_entropy",
    "posterior",
    "mahal",
    "reconstruction",
    "ood",
    "shock",
    "breadth",
    "oi_flush",
    "systemic_deleveraging",
    "funding",
    "pc1_variance",
    "downside_corr",
    "base_attr_abs_concentration",
)

_DYNAMIC_STATE_TOKENS = (
    "gmm_",
    "cluster_",
    "dae_",
    "ae_",
    "entropy",
    "mahal",
    "reconstruction",
    "ood",
    "shock",
    "breadth",
    "oi_",
    "funding",
    "liquidation",
    "flush",
    "recovery",
    "deleveraging",
    "volatility",
    "dispersion",
    "correlation",
    "corr_",
    "pc1_",
    "liquidity",
    "spread",
    "dislocation",
    "base_attr_",
    "model_",
    "residual",
    "support",
    "drift",
    "uncertainty",
)


def _dynamic_state_columns(
    columns: Sequence[str], *, maximum: int = 192
) -> list[str]:
    selected = [
        name
        for name in columns
        if any(token in name.casefold() for token in _DYNAMIC_STATE_TOKENS)
    ]
    return selected[: int(maximum)] or list(columns[: int(maximum)])


def _pairwise_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.maximum(np.mean(difference * difference, axis=2), 0.0))


def _rolling_distribution_shift(
    frame: pd.DataFrame,
    numeric: pd.DataFrame,
    *,
    recent_days: int = 3,
    reference_days: int = 30,
    max_features: int = 24,
) -> pd.DataFrame:
    """Compute compact causal recent-vs-reference distribution distances."""

    eligible = [
        name
        for name in numeric.columns
        if any(token in name.casefold() for token in _DISTRIBUTION_SHIFT_TOKENS)
    ][: int(max_features)]
    output = pd.DataFrame(
        {
            "state_energy_distance_3d_30d": np.nan,
            "state_mmd_rbf_3d_30d": np.nan,
            "state_wasserstein_proxy_3d_30d": np.nan,
            "state_distribution_shift_feature_count": float(len(eligible)),
        },
        index=frame.index,
        dtype=np.float32,
    )
    if not eligible:
        return output
    energy_column = output.columns.get_loc("state_energy_distance_3d_30d")
    mmd_column = output.columns.get_loc("state_mmd_rbf_3d_30d")
    wasserstein_column = output.columns.get_loc(
        "state_wasserstein_proxy_3d_30d"
    )
    values = numeric.loc[:, eligible].to_numpy(np.float32, copy=False)
    for positions in frame.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ).indices.values():
        index = np.asarray(positions, dtype=np.int64)
        for offset in range(len(index)):
            recent_start = max(0, offset - int(recent_days) + 1)
            reference_end = recent_start
            reference_start = max(0, reference_end - int(reference_days))
            if reference_end - reference_start < 10:
                continue
            recent = values[index[recent_start : offset + 1]].astype(
                np.float64, copy=False
            )
            reference = values[index[reference_start:reference_end]].astype(
                np.float64, copy=False
            )
            median = np.nanmedian(reference, axis=0)
            q25 = np.nanquantile(reference, 0.25, axis=0)
            q75 = np.nanquantile(reference, 0.75, axis=0)
            scale = np.maximum(q75 - q25, 1e-4)
            recent_z = np.clip(
                np.nan_to_num((recent - median) / scale, nan=0.0), -8.0, 8.0
            )
            reference_z = np.clip(
                np.nan_to_num((reference - median) / scale, nan=0.0), -8.0, 8.0
            )
            cross_distance = _pairwise_l2(recent_z, reference_z)
            recent_distance = _pairwise_l2(recent_z, recent_z)
            reference_distance = _pairwise_l2(reference_z, reference_z)
            energy = (
                2.0 * cross_distance.mean()
                - recent_distance.mean()
                - reference_distance.mean()
            )
            reference_nonzero = reference_distance[reference_distance > 0.0]
            bandwidth = (
                float(np.median(reference_nonzero))
                if len(reference_nonzero)
                else 1.0
            )
            gamma = 1.0 / max(2.0 * bandwidth * bandwidth, 1e-6)
            mmd = (
                np.exp(-gamma * recent_distance**2).mean()
                + np.exp(-gamma * reference_distance**2).mean()
                - 2.0 * np.exp(-gamma * cross_distance**2).mean()
            )
            recent_quantiles = np.nanquantile(recent_z, [0.25, 0.5, 0.75], axis=0)
            reference_quantiles = np.nanquantile(
                reference_z, [0.25, 0.5, 0.75], axis=0
            )
            wasserstein = np.mean(np.abs(recent_quantiles - reference_quantiles))
            row = index[offset]
            output.iat[row, energy_column] = np.float32(max(float(energy), 0.0))
            output.iat[row, mmd_column] = np.float32(max(float(mmd), 0.0))
            output.iat[row, wasserstein_column] = np.float32(wasserstein)
    return output


def add_causal_state_dynamics(
    daily_state: pd.DataFrame,
    *,
    lookback_days: int = 30,
    add_market_geometry: bool = True,
) -> pd.DataFrame:
    """Add causal transition and cross-sectional state geometry features.

    Every derived value uses the current day-open snapshot and earlier
    snapshots only. Robust reference medians/IQRs are shifted by one day so
    the current observation cannot alter its own normalization.
    """

    frame = daily_state.copy()
    frame["day"] = pd.to_datetime(frame["day"], utc=True).dt.floor("D")
    frame = frame.sort_values(
        ["side_name", "archetype_policy_key", "day"], kind="stable"
    )
    score_columns = {
        name: pd.to_numeric(frame[name], errors="coerce").astype(np.float32)
        for name in (
            "base_score",
            "score_meta_base_soft_label",
            "hit_probability",
            "historical_rank",
        )
        if name in frame
    }
    base_score = score_columns.get("base_score")
    meta_score = score_columns.get("score_meta_base_soft_label")
    hit_probability = score_columns.get("hit_probability")
    historical_rank = score_columns.get("historical_rank")
    if base_score is not None and meta_score is not None:
        frame["model_meta_minus_base_score"] = (meta_score - base_score).astype(
            np.float32
        )
        frame["model_base_meta_abs_disagreement"] = (
            meta_score - base_score
        ).abs().astype(np.float32)
    if meta_score is not None and hit_probability is not None:
        frame["model_meta_minus_hit_probability"] = (
            meta_score - hit_probability
        ).astype(np.float32)
        frame["model_meta_hit_abs_disagreement"] = (
            meta_score - hit_probability
        ).abs().astype(np.float32)
    if historical_rank is not None and hit_probability is not None:
        frame["model_rank_minus_hit_probability"] = (
            historical_rank - hit_probability
        ).astype(np.float32)
    if len(score_columns) >= 2:
        frame["model_layer_score_dispersion"] = pd.DataFrame(
            score_columns, index=frame.index
        ).std(axis=1, ddof=0).astype(np.float32)
    numeric_columns = [
        name
        for name in frame.columns
        if name not in KEYS
        and not name.startswith(("target__", "expost__"))
        and pd.api.types.is_numeric_dtype(frame[name])
    ]
    if not numeric_columns:
        return frame
    numeric = frame.loc[:, numeric_columns].astype(np.float32, copy=False)
    dynamic_columns = _dynamic_state_columns(numeric_columns)
    dynamic = numeric.loc[:, dynamic_columns]
    groups = frame.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    )
    lag1 = groups[dynamic_columns].shift(1)
    lag3 = groups[dynamic_columns].shift(3)
    prior = lag1.groupby(
        [frame["side_name"], frame["archetype_policy_key"]], observed=True, sort=False
    )
    rolling_median = prior.transform(
        lambda values: values.rolling(int(lookback_days), min_periods=10).median()
    )
    rolling_q25 = prior.transform(
        lambda values: values.rolling(int(lookback_days), min_periods=10).quantile(0.25)
    )
    rolling_q75 = prior.transform(
        lambda values: values.rolling(int(lookback_days), min_periods=10).quantile(0.75)
    )
    scale = (rolling_q75 - rolling_q25).clip(lower=1e-4)
    delta1 = dynamic - lag1
    delta3 = dynamic - lag3
    prior_rz = ((dynamic - rolling_median) / scale).clip(-8.0, 8.0)
    absolute_delta = delta1.abs()
    transition_summary = pd.DataFrame(
        {
            "state_transition_l1": absolute_delta.mean(axis=1),
            "state_transition_l2": np.sqrt(delta1.pow(2).mean(axis=1)),
            "state_transition_p90": absolute_delta.quantile(0.90, axis=1),
            "state_jump_fraction_rz2": prior_rz.abs().gt(2.0).mean(axis=1),
            "state_positive_shift_fraction_rz1": prior_rz.gt(1.0).mean(axis=1),
            "state_negative_shift_fraction_rz1": prior_rz.lt(-1.0).mean(axis=1),
            "state_directional_shift": prior_rz.clip(-3.0, 3.0).mean(axis=1),
            "state_dynamic_feature_count": float(len(dynamic_columns)),
        },
        index=frame.index,
        dtype=np.float32,
    )
    local_keys = [frame["side_name"], frame["archetype_policy_key"]]
    transition_summary["state_directional_shift_sum3"] = (
        transition_summary["state_directional_shift"]
        .groupby(local_keys, observed=True, sort=False)
        .transform(lambda values: values.rolling(3, min_periods=1).sum())
        .astype(np.float32)
    )
    transition_summary["state_directional_shift_sum7"] = (
        transition_summary["state_directional_shift"]
        .groupby(local_keys, observed=True, sort=False)
        .transform(lambda values: values.rolling(7, min_periods=1).sum())
        .astype(np.float32)
    )

    # One-sided Page-Hinkley/CUSUM-style pressure with a small deadband. The
    # recursion is computed independently for every side x archetype sequence
    # and never reads a later row.
    positive_cusum = np.zeros(len(frame), dtype=np.float32)
    negative_cusum = np.zeros(len(frame), dtype=np.float32)
    directional = transition_summary["state_directional_shift"].to_numpy(
        dtype=np.float32, copy=False
    )
    for positions in frame.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ).indices.values():
        pos_value = 0.0
        neg_value = 0.0
        for position in np.asarray(positions, dtype=np.int64):
            value = float(directional[position])
            if not np.isfinite(value):
                value = 0.0
            pos_value = max(0.0, pos_value + value - 0.05)
            neg_value = max(0.0, neg_value - value - 0.05)
            positive_cusum[position] = pos_value
            negative_cusum[position] = neg_value
    transition_summary["state_positive_cusum"] = positive_cusum
    transition_summary["state_negative_cusum"] = negative_cusum

    blocks = [
        delta1.rename(columns=lambda name: f"state_delta1__{name}"),
        delta3.rename(columns=lambda name: f"state_delta3__{name}"),
        prior_rz.rename(columns=lambda name: f"state_prior_rz30__{name}"),
        transition_summary,
        _rolling_distribution_shift(frame, dynamic),
    ]
    if (
        add_market_geometry
        and not (
            frame["side_name"].eq("global")
            & frame["archetype_policy_key"].eq("global_market")
        ).all()
    ):
        market = dynamic.groupby(frame["day"], observed=True).transform("median")
        blocks.extend(
            [
                market.rename(columns=lambda name: f"market_median__{name}"),
                (dynamic - market).rename(
                    columns=lambda name: f"local_minus_market__{name}"
                ),
            ]
        )
    return pd.concat(
        [
            frame.reset_index(drop=True),
            *[block.reset_index(drop=True) for block in blocks],
        ],
        axis=1,
        copy=False,
    )


def _quantile_codes(values: pd.Series, bins: int) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 8 or numeric.nunique(dropna=True) < 2:
        return np.zeros(len(numeric), dtype=np.int16)
    rank = numeric.rank(method="average", pct=True).fillna(0.5).to_numpy(np.float64)
    return np.minimum((rank * int(bins)).astype(np.int16), int(bins) - 1)


def nonlinear_feature_screen(
    train: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    maximum: int,
    bins: int,
) -> pd.DataFrame:
    """Rank observable features by binned nonlinear MI and tail separation."""

    validate_inference_feature_columns(features)
    y = train[target].fillna(False).astype(np.int8).to_numpy()
    if np.unique(y).size < 2:
        return pd.DataFrame(
            columns=["feature", "mutual_information", "tail_lift", "score"]
        )
    prevalence = max(float(y.mean()), 1e-6)
    numeric = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    minimum_finite = max(8, int(np.ceil(0.30 * len(numeric))))
    usable = [
        name
        for name in numeric.columns
        if int(numeric[name].notna().sum()) >= minimum_finite
        and int(numeric[name].nunique(dropna=True)) >= 2
    ]
    if not usable:
        return pd.DataFrame(
            columns=["feature", "mutual_information", "tail_lift", "score"]
        )
    numeric = numeric.loc[:, usable]
    features = usable
    values = numeric.to_numpy(np.float32, copy=False)
    ranks = numeric.rank(method="average", pct=True).fillna(0.5).to_numpy(np.float32)
    codes = np.minimum((ranks * int(bins)).astype(np.int16), int(bins) - 1)
    q10, q90 = np.nanquantile(values, [0.10, 0.90], axis=0)
    rows: list[dict[str, float | str]] = []
    for column_index, name in enumerate(features):
        local_codes = codes[:, column_index]
        if np.unique(local_codes).size < 2:
            continue
        contingency = (
            np.bincount(
                local_codes.astype(np.int64) * 2 + y,
                minlength=int(bins) * 2,
            )
            .reshape(int(bins), 2)
            .astype(np.float64)
        )
        total = contingency.sum()
        joint = contingency / max(total, 1.0)
        row_mass = joint.sum(axis=1, keepdims=True)
        col_mass = joint.sum(axis=0, keepdims=True)
        expected = row_mass @ col_mass
        valid = (joint > 0.0) & (expected > 0.0)
        mi = float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))
        raw = values[:, column_index]
        low_mask = np.isfinite(raw) & (raw <= q10[column_index])
        high_mask = np.isfinite(raw) & (raw >= q90[column_index])
        low_rate = float(y[low_mask].mean()) if low_mask.any() else prevalence
        high_rate = float(y[high_mask].mean()) if high_mask.any() else prevalence
        tail_lift = max(low_rate, high_rate) / prevalence
        rows.append(
            {
                "feature": name,
                "mutual_information": mi,
                "tail_lift": tail_lift,
                "score": mi * (1.0 + np.log1p(max(tail_lift - 1.0, 0.0))),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["feature", "mutual_information", "tail_lift", "score"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["score", "mutual_information"], ascending=False, kind="stable")
        .head(int(maximum))
        .reset_index(drop=True)
    )


def attach_failure_mode_targets(
    daily_state: pd.DataFrame,
    calendar: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    lead_days: Sequence[int] = (1, 3),
) -> pd.DataFrame:
    """Join descriptive train labels to observable daily state.

    The returned ``target__`` columns are labels only.  Callers must remove
    them from model inputs; :func:`chronological_failure_detection` enforces
    that boundary.
    """

    state = daily_state.copy()
    state["day"] = pd.to_datetime(state["day"], utc=True).dt.floor("D")
    label_columns = [*KEYS, "adverse_event", "event_block"]
    for optional in ("negative_pnl_day", "mean_ev_after_1pct"):
        if optional in calendar.columns:
            label_columns.append(optional)
    labels = calendar.loc[:, label_columns].copy()
    labels["day"] = pd.to_datetime(labels["day"], utc=True).dt.floor("D")
    mode_columns = [
        "side_name",
        "archetype_policy_key",
        "event_block",
        "method",
        "latent_dim",
        "clusters",
        "cluster_id",
        "semantic_label",
    ]
    available = [name for name in mode_columns if name in assignments.columns]
    modes = assignments.loc[:, available].drop_duplicates(
        ["side_name", "archetype_policy_key", "event_block"]
    )
    if not modes.empty:
        technical_mode = (
            modes["method"].astype(str)
            + "__d"
            + modes["latent_dim"].astype(str)
            + "__k"
            + modes["clusters"].astype(str)
            + "__c"
            + modes["cluster_id"].astype(str)
        )
        semantic = modes.get(
            "semantic_label", pd.Series(pd.NA, index=modes.index, dtype="string")
        ).astype("string")
        modes["failure_mode"] = semantic.where(
            semantic.notna() & semantic.str.strip().ne(""),
            "unresolved_failure_mode",
        ) + "::" + technical_mode
        labels = labels.merge(
            modes.loc[
                :, ["side_name", "archetype_policy_key", "event_block", "failure_mode"]
            ],
            on=["side_name", "archetype_policy_key", "event_block"],
            how="left",
            validate="many_to_one",
        )
        assignment_keys = [
            "side_name",
            "archetype_policy_key",
            "event_block",
        ]
        if "event_end" in assignments:
            assignment_end = assignments.loc[
                :, [*assignment_keys, "event_end"]
            ].drop_duplicates(assignment_keys)
        else:
            # Older assignment artifacts did not retain episode boundaries.
            # Recover the end from the labelled event calendar rather than
            # weakening the availability contract.
            assignment_end = (
                labels.loc[labels["event_block"].ne("normal"), [*assignment_keys, "day"]]
                .groupby(assignment_keys, observed=True, as_index=False)["day"]
                .max()
                .rename(columns={"day": "event_end"})
            )
        assignment_end["failure_mode_available_day"] = (
            pd.to_datetime(assignment_end["event_end"], utc=True, errors="coerce")
            + pd.Timedelta(
                days=FAILURE_MODE_RECOVERY_HORIZON_DAYS + OUTCOME_RESOLUTION_DAYS
            )
        )
        labels = labels.merge(
            assignment_end.drop(columns="event_end"),
            on=["side_name", "archetype_policy_key", "event_block"],
            how="left",
            validate="many_to_one",
        )
    else:
        labels["failure_mode"] = pd.NA
        labels["failure_mode_available_day"] = pd.NaT
    labels["target__any_failure"] = labels["adverse_event"].fillna(False).astype(bool)
    if "negative_pnl_day" in labels:
        labels["target__negative_ev_day"] = (
            labels["negative_pnl_day"].astype("boolean").fillna(False).astype(bool)
        )
    elif "mean_ev_after_1pct" in labels:
        labels["target__negative_ev_day"] = pd.to_numeric(
            labels["mean_ev_after_1pct"], errors="coerce"
        ).lt(0.0)
    joined = state.merge(labels, on=list(KEYS), how="left", validate="one_to_one")
    joined["target__any_failure"] = (
        joined["target__any_failure"].astype("boolean").fillna(False).astype(bool)
    )
    if "target__negative_ev_day" in joined:
        joined["target__negative_ev_day"] = (
            joined["target__negative_ev_day"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    joined["failure_mode"] = joined["failure_mode"].astype("string")
    joined = joined.sort_values(
        ["side_name", "archetype_policy_key", "day"], kind="stable"
    ).reset_index(drop=True)
    joined["target__failure_onset"] = _failure_onset(
        joined, joined["target__any_failure"]
    )
    if "mean_ev_after_1pct" in joined:
        joined["target__failure_severity"] = (
            -pd.to_numeric(joined["mean_ev_after_1pct"], errors="coerce")
        ).clip(lower=0.0)
    else:
        joined["target__failure_severity"] = np.nan
    if "target__negative_ev_day" in joined:
        joined["target__negative_ev_onset"] = _failure_onset(
            joined, joined["target__negative_ev_day"]
        )
    for horizon in sorted({int(value) for value in lead_days if int(value) > 0}):
        for name in (
            "any_failure",
            "failure_onset",
            "negative_ev_day",
            "negative_ev_onset",
        ):
            source = f"target__{name}"
            if source not in joined:
                continue
            joined[f"target__next{horizon}d__{name}"] = _future_window_target(
                joined,
                joined[source],
                horizon_days=horizon,
            )
        joined[f"target__next{horizon}d__failure_severity"] = (
            _future_window_maximum(
                joined,
                joined["target__failure_severity"],
                horizon_days=horizon,
            )
        )
    return joined


def _severity_column_for_target(target: str) -> str:
    horizon = target_horizon_days(target)
    return (
        f"target__next{horizon}d__failure_severity"
        if horizon > 0
        else "target__failure_severity"
    )


def _fill_scale(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train, axis=0)
    q25 = np.nanquantile(train, 0.25, axis=0)
    q75 = np.nanquantile(train, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    median = np.nan_to_num(median, nan=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    output: list[np.ndarray] = []
    for values in (train, other):
        normalized = (values - median) / scale
        output.append(
            np.clip(
                np.nan_to_num(normalized, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0
            ).astype(np.float32, copy=False)
        )
    return output[0], output[1]


def _fit_model(
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = (
        train.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    x_score = (
        score.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    x_train, x_score = _fill_scale(x_train, x_score)
    y = train[target].astype(np.int8).to_numpy()
    positives = max(int(y.sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    weights = np.where(y > 0, negatives / positives, 1.0).astype(np.float32)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "None",
            "learning_rate": 0.025,
            "max_depth": 2,
            "num_leaves": 4,
            "min_data_in_leaf": max(8, min(30, len(train) // 12)),
            "min_gain_to_split": 0.05,
            "lambda_l1": 4.0,
            "lambda_l2": 20.0,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "seed": int(seed),
            "num_threads": 1,
            "verbosity": -1,
            "force_col_wise": True,
        },
        lgb.Dataset(x_train, label=y, weight=weights, feature_name=list(features)),
        num_boost_round=140,
    )
    return (
        np.asarray(model.predict(x_train), dtype=np.float32),
        np.asarray(model.predict(x_score), dtype=np.float32),
    )


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _calibrate_probability_scores(
    validation_score: np.ndarray,
    validation_target: np.ndarray,
    score: np.ndarray,
    *,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Calibrate weighted binary-model scores on a chronological holdout.

    LightGBM is fitted with inverse-prevalence class weights, so its raw binary
    output is a useful risk ordering but not an empirical probability.  A
    one-dimensional Platt map preserves ordering while restoring probability
    and expected-severity semantics. Sparse or one-class validation slices use
    the raw score rather than fitting an unstable calibrator.
    """

    validation_score = np.asarray(validation_score, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(validation_target, dtype=np.int8)
    requested = str(method or "none").strip().casefold()
    if requested in {"", "none", "identity"}:
        return (
            validation_score.astype(np.float32),
            score.astype(np.float32),
            "identity",
        )
    if requested != "platt":
        raise ValueError(f"Unsupported probability calibration: {method!r}")
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives < 3 or negatives < 3 or np.unique(validation_score).size < 3:
        return (
            validation_score.astype(np.float32),
            score.astype(np.float32),
            "identity_insufficient_support",
        )
    calibrator = LogisticRegression(
        C=0.25,
        class_weight=None,
        random_state=int(seed),
        solver="lbfgs",
        max_iter=200,
    )
    calibrator.fit(_logit(validation_score).reshape(-1, 1), target)
    if float(calibrator.coef_[0, 0]) <= 1e-8:
        return (
            validation_score.astype(np.float32),
            score.astype(np.float32),
            "identity_nonmonotonic_platt",
        )
    calibrated_validation = calibrator.predict_proba(
        _logit(validation_score).reshape(-1, 1)
    )[:, 1]
    calibrated_score = calibrator.predict_proba(_logit(score).reshape(-1, 1))[:, 1]
    return (
        np.clip(calibrated_validation, 0.0, 1.0).astype(np.float32),
        np.clip(calibrated_score, 0.0, 1.0).astype(np.float32),
        "platt_logit_inner_validation",
    )


def fit_frozen_same_day_detector(
    labelled_state: pd.DataFrame,
    *,
    side_name: str,
    archetype_policy_key: str,
    boundary: pd.Timestamp,
    config: ProspectiveFailureDetectorConfig = ProspectiveFailureDetectorConfig(),
    feature_columns: Sequence[str] | None = None,
    target: str = "target__negative_ev_day",
) -> FrozenFailureDetector | None:
    """Fit one forward-scoring same-day detector from label-available history.

    ``boundary`` is the first date to score.  The target availability purge is
    applied before both feature screening and final fitting.  The inner tail
    threshold and Platt map are therefore causal with respect to every future
    score, including a score emitted after the original OOS experiment ended.
    """

    frame = labelled_state.loc[
        labelled_state["side_name"].astype(str).eq(str(side_name))
        & labelled_state["archetype_policy_key"].astype(str).eq(
            str(archetype_policy_key)
        )
    ].copy()
    if target not in frame or frame.empty:
        return None
    boundary = pd.Timestamp(boundary)
    if boundary.tzinfo is None:
        raise ValueError("Frozen detector boundary must be timezone-aware")
    frame["day"] = pd.to_datetime(frame["day"], utc=True).dt.floor("D")
    train = frame.loc[frame["day"].lt(boundary)].copy()
    train = train.loc[train[target].notna()].copy()
    train = purged_before_boundary(
        train,
        boundary=boundary,
        target=target,
        embargo_days=config.embargo_days,
    )
    if train.empty:
        return None
    train[target] = train[target].astype(bool)
    positives = int(train[target].sum())
    if positives < int(config.min_positive_days) or positives == len(train):
        return None
    excluded = {*KEYS, *LABEL_ONLY_COLUMNS}
    candidates = list(feature_columns) if feature_columns is not None else [
        name
        for name in train
        if name not in excluded
        and not name.startswith(("target__", LABEL_AVAILABILITY_PREFIX))
    ]
    validate_inference_feature_columns(candidates)
    inner_cutoff = boundary - pd.Timedelta(days=int(config.inner_validation_days))
    inner_train = purged_before_boundary(
        train,
        boundary=inner_cutoff,
        target=target,
        embargo_days=config.embargo_days,
    )
    validation_label_cutoff = boundary - pd.Timedelta(
        days=target_horizon_days(target) + max(0, config.embargo_days)
    )
    inner_valid = train.loc[
        train["day"].ge(inner_cutoff) & train["day"].lt(validation_label_cutoff)
    ].copy()
    if (
        inner_train.empty
        or inner_valid.empty
        or int(inner_train[target].sum()) < int(config.min_positive_days)
        or int(inner_valid[target].sum()) < 1
        or int((~inner_valid[target]).sum()) < 1
    ):
        return None
    selected_report = nonlinear_feature_screen(
        inner_train,
        candidates,
        target,
        maximum=config.max_features,
        bins=config.mi_bins,
    )
    selected = selected_report["feature"].tolist()
    if len(selected) < 2:
        return None
    _, inner_raw = _fit_model(
        inner_train,
        inner_valid,
        selected,
        target,
        seed=config.random_state + 7001,
    )
    x_train = (
        train.loc[:, selected].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    )
    median = np.nanmedian(x_train, axis=0)
    q25 = np.nanquantile(x_train, 0.25, axis=0)
    q75 = np.nanquantile(x_train, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    median = np.nan_to_num(median, nan=0.0).astype(np.float32)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    x_fit = np.clip(
        np.nan_to_num((x_train - median) / scale, nan=0.0, posinf=8.0, neginf=-8.0),
        -8.0,
        8.0,
    ).astype(np.float32)
    y = train[target].astype(np.int8).to_numpy()
    negatives = max(int((y == 0).sum()), 1)
    weights = np.where(y > 0, negatives / max(int(y.sum()), 1), 1.0).astype(np.float32)
    model = lgb.train(
        {
            "objective": "binary", "metric": "None", "learning_rate": 0.025,
            "max_depth": 2, "num_leaves": 4,
            "min_data_in_leaf": max(8, min(30, len(train) // 12)),
            "min_gain_to_split": 0.05, "lambda_l1": 4.0, "lambda_l2": 20.0,
            "feature_fraction": 0.80, "bagging_fraction": 0.85, "bagging_freq": 1,
            "seed": int(config.random_state + 8001), "num_threads": 1,
            "verbosity": -1, "force_col_wise": True,
        },
        lgb.Dataset(x_fit, label=y, weight=weights, feature_name=selected),
        num_boost_round=140,
    )
    calibration_method = "identity"
    coefficient: float | None = None
    intercept: float | None = None
    validation_target = inner_valid[target].astype(np.int8).to_numpy()
    if str(config.probability_calibration).casefold() == "platt":
        pos = int(validation_target.sum())
        neg = int(len(validation_target) - pos)
        if pos >= 3 and neg >= 3 and np.unique(inner_raw).size >= 3:
            calibrator = LogisticRegression(
                C=0.25, random_state=int(config.random_state + 9001),
                solver="lbfgs", max_iter=200,
            ).fit(_logit(inner_raw).reshape(-1, 1), validation_target)
            if float(calibrator.coef_[0, 0]) > 1e-8:
                coefficient = float(calibrator.coef_[0, 0])
                intercept = float(calibrator.intercept_[0])
                validation_score = 1.0 / (1.0 + np.exp(
                    -(coefficient * _logit(inner_raw) + intercept)
                ))
                calibration_method = "platt_logit_inner_validation"
            else:
                validation_score = inner_raw
        else:
            validation_score = inner_raw
    else:
        validation_score = inner_raw
    threshold = float(np.quantile(validation_score, float(config.alert_quantile)))
    return FrozenFailureDetector(
        side_name=str(side_name),
        archetype_policy_key=str(archetype_policy_key), target=target,
        selected_features=selected, median=median, scale=scale,
        model_string=model.model_to_string(), threshold=threshold,
        calibration_method=calibration_method, platt_coef=coefficient,
        platt_intercept=intercept, train_boundary=str(boundary),
        train_rows=int(len(train)), train_positive_days=positives,
    )


def _fit_positive_severity_model(
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: Sequence[str],
    target: str,
    severity_column: str,
    *,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Predict conditional severity after a failure with a shallow hurdle head."""

    severity = pd.to_numeric(train.get(severity_column), errors="coerce")
    positive = train[target].astype("boolean").fillna(False).astype(bool)
    usable = positive & severity.notna() & severity.ge(0.0)
    fallback = float(severity.loc[usable].mean()) if usable.any() else 0.0
    if int(usable.sum()) < 12 or float(severity.loc[usable].std(ddof=0)) < 1e-6:
        return np.full(len(score), fallback, dtype=np.float32), "support_mean"
    x_train = (
        train.loc[usable, features]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    x_score = (
        score.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    x_train, x_score = _fill_scale(x_train, x_score)
    y_train = severity.loc[usable].to_numpy(dtype=np.float32, copy=False)
    model = lgb.train(
        {
            "objective": "regression_l1",
            "metric": "None",
            "learning_rate": 0.025,
            "max_depth": 2,
            "num_leaves": 4,
            "min_data_in_leaf": max(6, min(20, len(y_train) // 8)),
            "min_gain_to_split": 0.02,
            "lambda_l1": 4.0,
            "lambda_l2": 25.0,
            "feature_fraction": 0.80,
            "seed": int(seed),
            "num_threads": 1,
            "verbosity": -1,
            "force_col_wise": True,
        },
        lgb.Dataset(x_train, label=y_train, feature_name=list(features)),
        num_boost_round=100,
    )
    upper = max(float(np.quantile(y_train, 0.99)), fallback, 1e-6)
    prediction = np.clip(model.predict(x_score), 0.0, upper).astype(
        np.float32, copy=False
    )
    return prediction, "shallow_l1_hurdle"


def _metrics(
    y: np.ndarray, score: np.ndarray, threshold: float
) -> dict[str, float | int]:
    selected = score >= float(threshold)
    prevalence = float(y.mean()) if len(y) else np.nan
    precision = float(y[selected].mean()) if selected.any() else np.nan
    return {
        "oos_days": int(len(y)),
        "oos_positive_days": int(y.sum()),
        "alert_days": int(selected.sum()),
        "alert_rate": float(selected.mean()) if len(y) else np.nan,
        "precision": precision,
        "recall": float((selected & (y > 0)).sum() / max(int(y.sum()), 1)),
        "lift": precision / max(prevalence, 1e-9) if np.isfinite(precision) else np.nan,
        "average_precision": float(average_precision_score(y, score))
        if np.unique(y).size > 1
        else np.nan,
        "brier": float(np.mean((score - y) ** 2)),
    }


def chronological_failure_detection(
    labelled_state: pd.DataFrame,
    *,
    config: ProspectiveFailureDetectorConfig = ProspectiveFailureDetectorConfig(),
    feature_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit expanding chronological any-failure and mode-specific detectors."""

    frame = labelled_state.copy()
    frame["day"] = pd.to_datetime(frame["day"], utc=True).dt.floor("D")
    excluded = {*KEYS, *LABEL_ONLY_COLUMNS}
    candidates = (
        list(feature_columns)
        if feature_columns is not None
        else [
            name
            for name in frame
            if name not in excluded
            and not name.startswith(("target__", LABEL_AVAILABILITY_PREFIX))
        ]
    )
    validate_inference_feature_columns(candidates)
    predictions: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    selections: list[pd.DataFrame] = []
    for (side, archetype), local in frame.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        local = local.sort_values("day", kind="stable").reset_index(drop=True)
        target_specs: dict[str, str] = {
            name.removeprefix("target__").replace("__", "_"): name
            for name in local.columns
            if name.startswith("target__")
            and "mode__" not in name
            and "failure_severity" not in name
        }
        for failure_mode in sorted(local["failure_mode"].dropna().unique().tolist()):
            active = local["failure_mode"].eq(failure_mode).fillna(False)
            onset = _failure_onset(local, active)
            onset_column = f"target__mode_onset__{failure_mode}"
            local[onset_column] = onset
            current_mode_available = pd.to_datetime(
                local.get("failure_mode_available_day"), utc=True, errors="coerce"
            )
            # Even a negative mode label on an adverse day is unknown until
            # that episode's ex-post mode classification has fully resolved.
            # Only confirmed non-event days use the one-day outcome delay.
            local[f"{LABEL_AVAILABILITY_PREFIX}{onset_column}"] = (
                current_mode_available.where(
                    local["target__any_failure"].astype(bool),
                    local["day"] + pd.Timedelta(days=OUTCOME_RESOLUTION_DAYS),
                )
            )
            target_specs[f"mode_onset::{failure_mode}"] = onset_column
            for horizon in config.lead_days:
                horizon_int = int(horizon)
                if horizon_int <= 0:
                    continue
                future_column = (
                    f"target__next{horizon_int}d__mode_onset__{failure_mode}"
                )
                local[future_column] = _future_window_target(
                    local,
                    onset,
                    horizon_days=horizon_int,
                )
                local[f"{LABEL_AVAILABILITY_PREFIX}{future_column}"] = (
                    _future_window_availability(
                        local,
                        local["target__any_failure"],
                        current_mode_available,
                        horizon_days=horizon_int,
                    )
                )
                target_specs[f"next{horizon_int}d_mode_onset::{failure_mode}"] = (
                    future_column
                )
        days = pd.DatetimeIndex(local["day"].drop_duplicates().sort_values())
        if len(days) <= config.min_train_days:
            continue
        fold_start = int(config.min_train_days)
        if str(config.evaluation_start).strip():
            evaluation_start = pd.Timestamp(config.evaluation_start)
            if evaluation_start.tzinfo is None:
                raise ValueError("evaluation_start must be timezone-aware")
            evaluation_start = evaluation_start.tz_convert("UTC")
            fold_start = max(
                fold_start,
                int(days.searchsorted(evaluation_start, side="left")),
            )
        fold_index = 0
        while fold_start < len(days):
            eval_days = days[fold_start : fold_start + int(config.eval_days)]
            if len(eval_days) == 0:
                break
            train_end = eval_days.min()
            eval_end = eval_days.max() + pd.Timedelta(days=1)
            train = local.loc[local["day"].lt(train_end)].copy()
            score = local.loc[
                local["day"].ge(train_end) & local["day"].lt(eval_end)
            ].copy()
            for mode, target in target_specs.items():
                train_mode = train.loc[train[target].notna()].copy()
                score_mode = score.loc[score[target].notna()].copy()
                train_mode = purged_before_boundary(
                    train_mode,
                    boundary=train_end,
                    target=target,
                    embargo_days=config.embargo_days,
                )
                if train_mode.empty or score_mode.empty:
                    continue
                train_mode[target] = train_mode[target].astype(bool)
                score_mode[target] = score_mode[target].astype(bool)
                positives = int(train_mode[target].sum())
                if positives < int(config.min_positive_days) or positives == len(
                    train_mode
                ):
                    continue
                inner_cutoff = train_end - pd.Timedelta(
                    days=int(config.inner_validation_days)
                )
                inner_train = purged_before_boundary(
                    train_mode,
                    boundary=inner_cutoff,
                    target=target,
                    embargo_days=config.embargo_days,
                )
                validation_label_cutoff = train_end - pd.Timedelta(
                    days=target_horizon_days(target) + max(0, config.embargo_days)
                )
                inner_valid = train_mode.loc[
                    train_mode["day"].ge(inner_cutoff)
                    & train_mode["day"].lt(validation_label_cutoff)
                ].copy()
                if (
                    int(inner_train[target].sum()) < config.min_positive_days
                    or inner_valid.empty
                    or int(inner_valid[target].sum()) == 0
                ):
                    continue
                selected_report = nonlinear_feature_screen(
                    inner_train,
                    candidates,
                    target,
                    maximum=config.max_features,
                    bins=config.mi_bins,
                )
                selected = selected_report["feature"].tolist()
                if len(selected) < 2:
                    continue
                _, inner_score_raw = _fit_model(
                    inner_train,
                    inner_valid,
                    selected,
                    target,
                    seed=config.random_state + fold_index,
                )
                _, oos_score_raw = _fit_model(
                    train_mode,
                    score_mode,
                    selected,
                    target,
                    seed=config.random_state + 1000 + fold_index,
                )
                inner_score, oos_score, calibration_method = (
                    _calibrate_probability_scores(
                        inner_score_raw,
                        inner_valid[target].astype(np.int8).to_numpy(),
                        oos_score_raw,
                        method=config.probability_calibration,
                        seed=config.random_state + 3000 + fold_index,
                    )
                )
                threshold = float(np.quantile(inner_score, config.alert_quantile))
                y_oos = score_mode[target].astype(np.int8).to_numpy()
                severity_column = _severity_column_for_target(target)
                severity = (
                    pd.to_numeric(train_mode.get(severity_column), errors="coerce")
                    if severity_column in train_mode
                    else pd.Series(np.nan, index=train_mode.index)
                )
                positive_severity = severity.loc[train_mode[target].astype(bool)]
                positive_severity = positive_severity.loc[
                    np.isfinite(positive_severity) & positive_severity.ge(0.0)
                ]
                conditional_severity_mean = (
                    float(positive_severity.mean())
                    if len(positive_severity)
                    else 0.0
                )
                conditional_severity, severity_model = _fit_positive_severity_model(
                    train_mode,
                    score_mode,
                    selected,
                    target,
                    severity_column,
                    seed=config.random_state + 2000 + fold_index,
                )
                expected_severity = oos_score * conditional_severity
                aleatoric_uncertainty = 4.0 * oos_score * (1.0 - oos_score)
                support_uncertainty = 1.0 / np.sqrt(max(positives, 1))
                realized_severity = (
                    pd.to_numeric(score_mode.get(severity_column), errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                    * y_oos
                )
                base = {
                    "fold_index": fold_index,
                    "train_end": train_end,
                    "eval_end": eval_end,
                    "side_name": str(side),
                    "archetype_policy_key": str(archetype),
                    "failure_mode": str(mode),
                    "train_days": int(len(train_mode)),
                    "train_positive_days": positives,
                    "target_horizon_days": target_horizon_days(target),
                    "embargo_days": int(config.embargo_days),
                    "train_label_max_day": train_mode["day"].max(),
                    "inner_train_label_max_day": inner_train["day"].max(),
                    "inner_valid_label_max_day": inner_valid["day"].max(),
                    "threshold": threshold,
                    "probability_calibration": calibration_method,
                    "train_positive_conditional_severity": conditional_severity_mean,
                    "severity_model": severity_model,
                    "oos_mean_conditional_failure_severity": float(
                        np.mean(conditional_severity)
                    ),
                    "train_positive_support_uncertainty": support_uncertainty,
                    "oos_mean_expected_failure_severity": float(
                        np.mean(expected_severity)
                    ),
                    "oos_mean_realized_failure_severity": float(
                        np.mean(realized_severity)
                    ),
                    "oos_failure_severity_mae": float(
                        np.mean(np.abs(expected_severity - realized_severity))
                    ),
                    "selected_features": "|".join(selected),
                }
                reports.append({**base, **_metrics(y_oos, oos_score, threshold)})
                prediction = score_mode.loc[:, [*KEYS, target]].rename(
                    columns={target: "target"}
                )
                prediction["risk"] = oos_score
                prediction["risk_raw"] = oos_score_raw
                prediction["expected_failure_severity"] = expected_severity
                prediction["conditional_failure_severity"] = conditional_severity
                prediction["target_failure_severity"] = realized_severity
                prediction["risk_aleatoric_uncertainty"] = aleatoric_uncertainty
                prediction["risk_support_uncertainty"] = support_uncertainty
                prediction["threshold"] = threshold
                prediction["alert"] = oos_score >= threshold
                predictions.append(prediction.assign(**base))
                selections.append(selected_report.assign(**base))
            fold_start += int(config.eval_days)
            fold_index += 1
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(reports),
        pd.concat(selections, ignore_index=True) if selections else pd.DataFrame(),
    )
