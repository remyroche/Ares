"""One shared, regime-aware exact-net residual expert.

This module is deliberately a *building block*, rather than an experiment
runner.  It implements the causal transformations required by the Stage-III
shared-expert design:

``base expected net -> soft side x regime prior -> candidate residual``.

There is exactly one residual model.  Soft regime probabilities are used for a
shrunk baseline, restricted interactions, and a later hierarchical score-to-bps
calibration.  They must never be converted into hard routes or per-regime
models.  All outcome-dependent quantities below are prequential: a row at
decision time ``t`` sees only labels with ``label_available_ts < t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .lgbm_pipeline import _fit_lgbm_model
from .shared_residual_funnel_contract import reconstruct_shared_common_bps


SCHEMA = "shared_regime_residual_expert_v1"
_EPS = 1e-8
_FORBIDDEN_FEATURE_TOKENS = (
    "exact_net", "realised_net", "realized_net", "outcome_resolved",
    "candidate_residual", "target", "label", "future_", "mfe", "mae",
)
_HARD_REGIME_FEATURE_TOKENS = (
    "regime_id", "regime_code", "regime_class", "hard_regime", "argmax_regime",
)


class SharedResidualExpertError(ValueError):
    """Raised when the shared-expert causal contract is not met."""


@dataclass(frozen=True)
class SharedResidualColumns:
    """Explicit narrow-ledger column contract.

    ``base_expected_net_bps`` must be a strict OOF/frozen, prequential mapping;
    same-fold or converted scores are intentionally outside this interface.
    """

    decision_timestamp: str = "decision_ts"
    label_available_timestamp: str = "label_available_ts"
    side: str = "side_name"
    exact_net_bps: str = "exact_net_bps"
    base_expected_net_bps: str = "prequential_base_expected_net_bps"


@dataclass(frozen=True)
class SoftRegimeResidualConfig:
    """Support/shrinkage settings for a prequential soft-regime baseline."""

    min_global_rows: int = 64
    side_shrink_rows: float = 1_500.0
    regime_shrink_rows: float = 3_000.0
    regime_weight_cap: float = 0.50
    residual_scale_floor_bps: float = 25.0
    target_clip_bps: float = 400.0

    def validate(self) -> None:
        if self.min_global_rows < 1:
            raise SharedResidualExpertError("min_global_rows must be positive")
        if self.side_shrink_rows <= 0 or self.regime_shrink_rows <= 0:
            raise SharedResidualExpertError("residual shrinkage constants must be positive")
        if not 0.0 < self.regime_weight_cap <= 1.0:
            raise SharedResidualExpertError("regime_weight_cap must be in (0, 1]")
        if self.residual_scale_floor_bps <= 0 or self.target_clip_bps <= 0:
            raise SharedResidualExpertError("residual scales and clips must be positive")


@dataclass(frozen=True)
class SharedResidualExpertFit:
    """Frozen one-model fit with the target unit needed for reconstruction."""

    model: Any
    feature_names: tuple[str, ...]
    target_mode: Literal["huber", "clipped", "regime_standardized"]
    training_cutoff_utc: pd.Timestamp
    max_label_available_utc: pd.Timestamp
    rows: int
    contract: str = SCHEMA

    def predict_candidate_residual_bps(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise SharedResidualExpertError(
                f"shared residual prediction is missing frozen features: {missing[:12]}"
            )
        raw = np.asarray(
            self.model.predict(frame.loc[:, self.feature_names]), dtype=np.float64
        ).reshape(-1)
        if not np.isfinite(raw).all():
            raise SharedResidualExpertError("shared residual model returned non-finite values")
        if self.target_mode == "regime_standardized":
            scale_name = "prequential_soft_regime_residual_scale_bps"
            if scale_name not in frame:
                raise SharedResidualExpertError(
                    f"standardized residual prediction requires {scale_name!r}"
                )
            scale = pd.to_numeric(frame[scale_name], errors="coerce").to_numpy(float)
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise SharedResidualExpertError("residual scale must be finite and positive")
            raw = raw * scale
        return raw.astype(np.float32, copy=False)


def _utc_series(frame: pd.DataFrame, column: str, *, name: str) -> pd.Series:
    if column not in frame:
        raise SharedResidualExpertError(f"frame lacks {name} column {column!r}")
    value = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if value.isna().any():
        raise SharedResidualExpertError(f"{name} contains invalid timestamps")
    return value


def _validate_soft_simplex(
    frame: pd.DataFrame, soft_regime_columns: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = tuple(dict.fromkeys(str(name) for name in soft_regime_columns))
    if len(columns) < 2:
        raise SharedResidualExpertError("at least two soft regime probability columns are required")
    missing = [name for name in columns if name not in frame]
    if missing:
        raise SharedResidualExpertError(f"frame lacks soft regime columns: {missing}")
    p = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(p).all() or (p < -1e-8).any():
        raise SharedResidualExpertError("soft regime probabilities must be finite and non-negative")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise SharedResidualExpertError("soft regime probabilities must sum to one per row")
    return np.clip(p, 0.0, 1.0), columns


def _validate_core(
    frame: pd.DataFrame,
    columns: SharedResidualColumns,
    soft_regime_columns: Sequence[str],
    *,
    require_outcome: bool = True,
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    decision = _utc_series(frame, columns.decision_timestamp, name="decision timestamp")
    available = _utc_series(frame, columns.label_available_timestamp, name="label availability")
    if (available <= decision).any():
        raise SharedResidualExpertError("labels must resolve strictly after their decision timestamp")
    if columns.side not in frame:
        raise SharedResidualExpertError(f"frame lacks side column {columns.side!r}")
    side = frame[columns.side].astype(str).str.lower().to_numpy(object)
    if (pd.Series(side).str.strip() == "").any():
        raise SharedResidualExpertError("side must be non-empty")
    if columns.base_expected_net_bps not in frame:
        raise SharedResidualExpertError("frame lacks strict OOF base expected-net bps")
    base = pd.to_numeric(frame[columns.base_expected_net_bps], errors="coerce").to_numpy(float)
    if not np.isfinite(base).all():
        raise SharedResidualExpertError("base expected net must be finite common-bps values")
    if require_outcome:
        if columns.exact_net_bps not in frame:
            raise SharedResidualExpertError("frame lacks exact net target")
        exact = pd.to_numeric(frame[columns.exact_net_bps], errors="coerce").to_numpy(float)
        if not np.isfinite(exact).all():
            raise SharedResidualExpertError("exact net target must be finite common-bps values")
    else:
        exact = np.full(len(frame), np.nan, dtype=float)
    p, names = _validate_soft_simplex(frame, soft_regime_columns)
    return decision, available, side, base, exact, names


def _shrink_weight(support: np.ndarray | float, shrink_rows: float, cap: float = 1.0) -> np.ndarray:
    values = np.asarray(support, dtype=float)
    return np.minimum(float(cap), np.divide(values, values + float(shrink_rows), out=np.zeros_like(values), where=values > 0))


def _blocks(values: pd.Series) -> list[np.ndarray]:
    order = np.argsort(values.to_numpy(dtype="datetime64[ns]"), kind="stable")
    ordered = values.to_numpy(dtype="datetime64[ns]")[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    stops = np.r_[starts[1:], len(order)]
    return [order[start:stop] for start, stop in zip(starts, stops, strict=False)]


def prequential_soft_side_regime_residual_baseline(
    frame: pd.DataFrame,
    *,
    soft_regime_columns: Sequence[str],
    columns: SharedResidualColumns = SharedResidualColumns(),
    config: SoftRegimeResidualConfig = SoftRegimeResidualConfig(),
    baseline_mode: Literal[
        "A0_current",
        "A1_side_centered",
        "A2_side_hard_regime_centered",
        "A3_soft_regime_centered",
    ] = "A3_soft_regime_centered",
    hard_regime_column: str | None = None,
) -> pd.DataFrame:
    """Return prior-resolved soft side×regime baseline and candidate residual.

    The function batches equal decision timestamps before admitting any newly
    resolved labels.  Therefore neither a row's own outcome nor another row at
    the same timestamp can influence its baseline.
    """
    config.validate()
    decision, available, side, base, exact, regime_names = _validate_core(
        frame, columns, soft_regime_columns
    )
    p, _ = _validate_soft_simplex(frame, regime_names)
    if baseline_mode not in {
        "A0_current", "A1_side_centered",
        "A2_side_hard_regime_centered", "A3_soft_regime_centered",
    }:
        raise SharedResidualExpertError(f"unsupported residual baseline mode {baseline_mode!r}")
    if baseline_mode == "A2_side_hard_regime_centered":
        if not hard_regime_column or hard_regime_column not in frame:
            raise SharedResidualExpertError(
                "A2 diagnostic baseline requires an explicit causal hard-regime column"
            )
        hard = frame[hard_regime_column].astype(str).str.strip()
        if hard.eq("").any():
            raise SharedResidualExpertError("hard-regime baseline values must be non-empty")
        hard_states = tuple(sorted(hard.unique().tolist()))
        hard_lookup = {name: position for position, name in enumerate(hard_states)}
        p = np.zeros((len(frame), len(hard_states)), dtype=float)
        p[np.arange(len(frame)), hard.map(hard_lookup).to_numpy(np.int32)] = 1.0
    residual = exact - base
    n, states = len(frame), p.shape[1]
    side_keys = tuple(sorted(pd.unique(side).tolist()))
    side_index = {key: pos for pos, key in enumerate(side_keys)}
    side_id = np.asarray([side_index[key] for key in side], dtype=np.int16)

    global_n = 0.0
    global_sum = 0.0
    global_sumsq = 0.0
    side_n = np.zeros(len(side_keys), dtype=float)
    side_sum = np.zeros(len(side_keys), dtype=float)
    side_sumsq = np.zeros(len(side_keys), dtype=float)
    state_n = np.zeros((len(side_keys), states), dtype=float)
    state_sum = np.zeros((len(side_keys), states), dtype=float)
    state_sumsq = np.zeros((len(side_keys), states), dtype=float)

    prior = np.full(n, np.nan, dtype=float)
    prior_scale = np.full(n, np.nan, dtype=float)
    global_support = np.zeros(n, dtype=np.int32)
    local_support = np.zeros(n, dtype=np.int32)
    expected_state_support = np.zeros(n, dtype=np.float32)
    fallback = np.empty(n, dtype=object)
    max_available = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

    if baseline_mode == "A0_current":
        result = pd.DataFrame(index=frame.index)
        result["prequential_soft_regime_prior_residual_bps"] = np.zeros(n, dtype=np.float32)
        result["prequential_soft_regime_residual_scale_bps"] = np.full(
            n, config.residual_scale_floor_bps, dtype=np.float32
        )
        result["candidate_residual_bps"] = residual.astype(np.float32)
        result["candidate_residual_clipped_bps"] = np.clip(
            residual, -config.target_clip_bps, config.target_clip_bps
        ).astype(np.float32)
        result["candidate_residual_standardized"] = (
            residual / config.residual_scale_floor_bps
        ).astype(np.float32)
        result["prior_resolved_global_support"] = np.zeros(n, dtype=np.int32)
        result["prior_resolved_side_support"] = np.zeros(n, dtype=np.int32)
        result["prior_resolved_expected_soft_regime_support"] = np.zeros(n, dtype=np.float32)
        result["prior_resolved_max_label_available_ts"] = pd.NaT
        result["soft_regime_prior_fallback"] = "no_regime_centering_control"
        return result

    by_available = np.argsort(available.to_numpy(dtype="datetime64[ns]"), kind="stable")
    pointer = 0

    def _admit(indices: np.ndarray) -> None:
        nonlocal global_n, global_sum, global_sumsq
        if not len(indices):
            return
        values = residual[indices]
        global_n += float(len(indices))
        global_sum += float(values.sum())
        global_sumsq += float(np.dot(values, values))
        for code in np.unique(side_id[indices]):
            take = indices[side_id[indices] == code]
            value = residual[take]
            side_n[code] += float(len(take))
            side_sum[code] += float(value.sum())
            side_sumsq[code] += float(np.dot(value, value))
            weights = p[take]
            state_n[code] += weights.sum(axis=0)
            state_sum[code] += weights.T @ value
            state_sumsq[code] += weights.T @ (value * value)

    for block in _blocks(decision):
        cutoff = decision.iloc[block[0]]
        admitted_start = pointer
        ordered_available = available.to_numpy(dtype="datetime64[ns]")[by_available]
        while pointer < n and ordered_available[pointer] < cutoff.to_datetime64():
            pointer += 1
        _admit(by_available[admitted_start:pointer])
        if global_n < float(config.min_global_rows):
            fallback[block] = "neutral_no_prior_resolved_support"
            continue
        global_mean = global_sum / global_n
        global_second = global_sumsq / global_n
        for code in np.unique(side_id[block]):
            take = block[side_id[block] == code]
            support = side_n[code]
            side_weight = float(_shrink_weight(support, config.side_shrink_rows))
            side_mean = global_mean if support <= 0 else global_mean + side_weight * (side_sum[code] / support - global_mean)
            side_second = global_second if support <= 0 else global_second + side_weight * (side_sumsq[code] / support - global_second)
            support_r = state_n[code]
            if baseline_mode == "A1_side_centered":
                prior[take] = side_mean
                second = np.full(len(take), side_second, dtype=float)
            else:
                state_weight = _shrink_weight(
                    support_r, config.regime_shrink_rows, config.regime_weight_cap
                )
                state_mean = np.where(
                    support_r > 0,
                    state_sum[code] / np.maximum(support_r, _EPS),
                    side_mean,
                )
                state_second = np.where(
                    support_r > 0,
                    state_sumsq[code] / np.maximum(support_r, _EPS),
                    side_second,
                )
                state_mean = side_mean + state_weight * (state_mean - side_mean)
                state_second = side_second + state_weight * (state_second - side_second)
                prior[take] = p[take] @ state_mean
                second = p[take] @ state_second
            prior_scale[take] = np.sqrt(np.maximum(second - prior[take] ** 2, config.residual_scale_floor_bps ** 2))
            global_support[take] = int(global_n)
            local_support[take] = int(support)
            expected_state_support[take] = (p[take] @ support_r).astype(np.float32)
            fallback[take] = {
                "A1_side_centered": "shrunk_side_prior",
                "A2_side_hard_regime_centered": "shrunk_side_hard_regime_prior_diagnostic",
                "A3_soft_regime_centered": "shrunk_soft_side_regime_prior",
            }[baseline_mode]
            if pointer:
                max_available[take] = ordered_available[pointer - 1]

    result = pd.DataFrame(index=frame.index)
    result["prequential_soft_regime_prior_residual_bps"] = prior.astype(np.float32)
    result["prequential_soft_regime_residual_scale_bps"] = prior_scale.astype(np.float32)
    result["candidate_residual_bps"] = (residual - prior).astype(np.float32)
    result["candidate_residual_clipped_bps"] = np.clip(
        result["candidate_residual_bps"], -config.target_clip_bps, config.target_clip_bps
    ).astype(np.float32)
    result["candidate_residual_standardized"] = (
        result["candidate_residual_bps"] / result["prequential_soft_regime_residual_scale_bps"]
    ).astype(np.float32)
    result["prior_resolved_global_support"] = global_support
    result["prior_resolved_side_support"] = local_support
    result["prior_resolved_expected_soft_regime_support"] = expected_state_support
    result["prior_resolved_max_label_available_ts"] = pd.to_datetime(max_available, utc=True)
    result["soft_regime_prior_fallback"] = fallback
    return result


def build_prequential_regime_relative_features(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    soft_regime_columns: Sequence[str],
    columns: SharedResidualColumns = SharedResidualColumns(),
    min_reference_rows: int = 64,
    side_shrink_rows: float = 1_500.0,
    regime_shrink_rows: float = 3_000.0,
    regime_weight_cap: float = 0.50,
    scale_floor: float = 1e-6,
    scale_estimator: Literal[
        "mean_absolute_deviation", "standard_deviation"
    ] = "mean_absolute_deviation",
    prefix: str = "__srre__",
) -> tuple[pd.DataFrame, list[str]]:
    """Create training-only soft-regime residual/z features from prior rows.

    These reference moments are driven solely by decision-time covariates and
    enforce ``reference decision_ts < current decision_ts``.  They do not use
    targets or outcome resolution, and so are valid both while fitting and in
    live/prequential scoring once state is carried forward.
    """
    if min_reference_rows < 1 or side_shrink_rows <= 0 or regime_shrink_rows <= 0:
        raise SharedResidualExpertError("relative feature support/shrinkage must be positive")
    if scale_estimator not in {"mean_absolute_deviation", "standard_deviation"}:
        raise SharedResidualExpertError(f"unknown regime-relative scale estimator {scale_estimator!r}")
    decision, _, side, _, _, regime_names = _validate_core(
        frame, columns, soft_regime_columns, require_outcome=False
    )
    names = list(dict.fromkeys(str(name) for name in feature_names))
    if not names:
        return pd.DataFrame(index=frame.index), []
    missing = [name for name in names if name not in frame]
    if missing:
        raise SharedResidualExpertError(f"relative feature source columns are missing: {missing[:12]}")
    x = frame.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    p, _ = _validate_soft_simplex(frame, regime_names)
    n, features, states = len(frame), len(names), p.shape[1]
    side_keys = tuple(sorted(pd.unique(side).tolist()))
    side_lookup = {name: index for index, name in enumerate(side_keys)}
    side_id = np.asarray([side_lookup[value] for value in side], dtype=np.int16)
    global_n = np.zeros(features, dtype=float); global_sum = np.zeros(features, dtype=float); global_sq = np.zeros(features, dtype=float)
    side_n = np.zeros((len(side_keys), features), dtype=float); side_sum = np.zeros_like(side_n); side_sq = np.zeros_like(side_n)
    state_n = np.zeros((len(side_keys), states, features), dtype=float)
    state_sum = np.zeros_like(state_n); state_sq = np.zeros_like(state_n)
    global_abs_n = np.zeros(features, dtype=float); global_abs_sum = np.zeros(features, dtype=float)
    side_abs_n = np.zeros_like(side_n); side_abs_sum = np.zeros_like(side_n)
    state_abs_n = np.zeros_like(state_n); state_abs_sum = np.zeros_like(state_n)
    residual_out = np.full((n, features), np.nan, dtype=np.float32)
    z_out = np.full((n, features), np.nan, dtype=np.float32)

    def _update(indices: np.ndarray) -> None:
        valid = np.isfinite(x[indices])
        xv = np.where(valid, x[indices], 0.0)
        if scale_estimator == "mean_absolute_deviation":
            ready = global_n > 0
            prior_mean = np.divide(
                global_sum, global_n, out=np.zeros(features), where=ready
            )
            global_abs_n[:] += (valid & ready[None, :]).sum(axis=0)
            global_abs_sum[:] += np.where(
                valid & ready[None, :], np.abs(xv - prior_mean), 0.0
            ).sum(axis=0)
        global_n[:] += valid.sum(axis=0)
        global_sum[:] += xv.sum(axis=0)
        global_sq[:] += (xv * xv).sum(axis=0)
        for code in np.unique(side_id[indices]):
            local = indices[side_id[indices] == code]
            good = np.isfinite(x[local]); value = np.where(good, x[local], 0.0)
            if scale_estimator == "mean_absolute_deviation":
                side_ready = side_n[code] > 0
                side_mean = np.divide(
                    side_sum[code], side_n[code],
                    out=np.zeros(features), where=side_ready,
                )
                side_abs_n[code] += (good & side_ready[None, :]).sum(axis=0)
                side_abs_sum[code] += np.where(
                    good & side_ready[None, :], np.abs(value - side_mean), 0.0
                ).sum(axis=0)
            side_n[code] += good.sum(axis=0); side_sum[code] += value.sum(axis=0); side_sq[code] += (value * value).sum(axis=0)
            weights = p[local]
            if scale_estimator == "mean_absolute_deviation":
                state_ready = state_n[code] > 0
                state_mean = np.divide(
                    state_sum[code], state_n[code],
                    out=np.zeros_like(state_sum[code]), where=state_ready,
                )
                weighted_valid = (
                    weights[:, :, None]
                    * good[:, None, :]
                    * state_ready[None, :, :]
                )
                state_abs_n[code] += weighted_valid.sum(axis=0)
                state_abs_sum[code] += (
                    weighted_valid
                    * np.abs(value[:, None, :] - state_mean[None, :, :])
                ).sum(axis=0)
            state_n[code] += np.einsum("rs,rf->sf", weights, good, optimize=True)
            state_sum[code] += np.einsum("rs,rf->sf", weights, value, optimize=True)
            state_sq[code] += np.einsum("rs,rf->sf", weights, value * value, optimize=True)

    for block in _blocks(decision):
        # Query first: equal-timestamp rows cannot normalize one another.
        for code in np.unique(side_id[block]):
            take = block[side_id[block] == code]
            okay = global_n >= float(min_reference_rows)
            if not okay.any():
                continue
            gmean = np.divide(global_sum, global_n, out=np.zeros(features), where=global_n > 0)
            gsecond = np.divide(global_sq, global_n, out=np.zeros(features), where=global_n > 0)
            sn = side_n[code]
            sw = _shrink_weight(sn, side_shrink_rows)
            smean = gmean + sw * (np.divide(side_sum[code], sn, out=gmean.copy(), where=sn > 0) - gmean)
            ssecond = gsecond + sw * (np.divide(side_sq[code], sn, out=gsecond.copy(), where=sn > 0) - gsecond)
            rn = state_n[code]
            rw = _shrink_weight(rn, regime_shrink_rows, regime_weight_cap)
            rmean = np.divide(state_sum[code], rn, out=np.broadcast_to(smean, rn.shape).copy(), where=rn > 0)
            rsecond = np.divide(state_sq[code], rn, out=np.broadcast_to(ssecond, rn.shape).copy(), where=rn > 0)
            rmean = smean[None, :] + rw * (rmean - smean[None, :])
            rsecond = ssecond[None, :] + rw * (rsecond - ssecond[None, :])
            expected = np.einsum("rs,sf->rf", p[take], rmean, optimize=True)
            if scale_estimator == "standard_deviation":
                expected_second = np.einsum("rs,sf->rf", p[take], rsecond, optimize=True)
                scale = np.sqrt(np.maximum(expected_second - expected * expected, scale_floor ** 2))
            else:
                global_abs = np.divide(
                    global_abs_sum, global_abs_n,
                    out=np.full(features, scale_floor), where=global_abs_n > 0,
                )
                side_abs_raw = np.divide(
                    side_abs_sum[code], side_abs_n[code],
                    out=global_abs.copy(), where=side_abs_n[code] > 0,
                )
                side_abs = global_abs + sw * (side_abs_raw - global_abs)
                state_abs_raw = np.divide(
                    state_abs_sum[code], state_abs_n[code],
                    out=np.broadcast_to(side_abs, state_abs_n[code].shape).copy(),
                    where=state_abs_n[code] > 0,
                )
                state_abs = side_abs[None, :] + rw * (state_abs_raw - side_abs[None, :])
                expected_abs = np.einsum("rs,sf->rf", p[take], state_abs, optimize=True)
                # For a Gaussian, sigma ~= sqrt(pi/2) * E|X-mu|.  Absolute
                # deviation has bounded first-order influence versus variance.
                scale = np.maximum(np.sqrt(np.pi / 2.0) * expected_abs, scale_floor)
            values = x[take]
            residual_out[take] = np.where(okay[None, :] & np.isfinite(values), values - expected, np.nan)
            z_out[take] = np.where(okay[None, :] & np.isfinite(values), (values - expected) / scale, np.nan)
        _update(block)

    output = pd.DataFrame(index=frame.index)
    generated: list[str] = []
    for pos, name in enumerate(names):
        residual_name = f"{prefix}{name}__soft_regime_residual"
        z_name = f"{prefix}{name}__soft_regime_z"
        output[residual_name] = residual_out[:, pos]
        output[z_name] = z_out[:, pos]
        generated.extend([residual_name, z_name])
    return output, generated


def build_restricted_soft_regime_interactions(
    frame: pd.DataFrame,
    *,
    soft_regime_columns: Sequence[str],
    base_feature_names: Sequence[str],
    prefix: str = "__srre_interaction__",
) -> tuple[pd.DataFrame, list[str]]:
    """Build only explicitly declared continuous feature × soft-state terms."""
    p, regime_names = _validate_soft_simplex(frame, soft_regime_columns)
    bases = list(dict.fromkeys(str(name) for name in base_feature_names))
    missing = [name for name in bases if name not in frame]
    if missing:
        raise SharedResidualExpertError(f"restricted interaction bases are missing: {missing[:12]}")
    output = pd.DataFrame(index=frame.index)
    generated: list[str] = []
    for feature in bases:
        value = pd.to_numeric(frame[feature], errors="coerce").to_numpy(np.float32)
        for state_pos, state in enumerate(regime_names):
            name = f"{prefix}{feature}__x__{state}"
            output[name] = value * p[:, state_pos].astype(np.float32)
            generated.append(name)
    return output, generated


def add_soft_regime_entropy(
    frame: pd.DataFrame, *, soft_regime_columns: Sequence[str], name: str = "soft_regime_entropy"
) -> pd.Series:
    """Return the observable uncertainty of the supplied soft regime state."""
    p, _ = _validate_soft_simplex(frame, soft_regime_columns)
    return pd.Series(-(p * np.log(np.maximum(p, _EPS))).sum(axis=1), index=frame.index, name=name, dtype=np.float32)


def mild_environment_weights(
    frame: pd.DataFrame,
    *,
    environment_column: str | None = None,
    soft_regime_columns: Sequence[str] = (),
    balance: Literal["natural", "era", "soft_regime"] = "natural",
    label_certainty_column: str | None = None,
    floor: float = 0.25,
    cap: float = 4.0,
) -> np.ndarray:
    """Return mean-one, capped square-root environment weights.

    Certainty is deliberately a training-loss weight only; this helper returns
    an array and never adds it to an inference feature frame.
    """
    if not 0 < floor <= cap:
        raise SharedResidualExpertError("weight floor/cap must be positive and ordered")
    n = len(frame)
    if n == 0:
        raise SharedResidualExpertError("cannot build weights for an empty frame")
    weight = np.ones(n, dtype=float)
    if balance == "era":
        if not environment_column or environment_column not in frame:
            raise SharedResidualExpertError("era balancing requires an explicit environment column")
        env = frame[environment_column].astype(str).to_numpy(object)
        if (pd.Series(env).str.strip() == "").any():
            raise SharedResidualExpertError("environment values must be non-empty")
        keys, counts = np.unique(env, return_counts=True)
        mapping = {key: np.sqrt(float(n) / (len(keys) * float(count))) for key, count in zip(keys, counts, strict=False)}
        weight *= np.asarray([mapping[key] for key in env], dtype=float)
    elif balance == "soft_regime":
        p, names = _validate_soft_simplex(frame, soft_regime_columns)
        effective = p.sum(axis=0)
        factors = np.sqrt(float(n) / (len(names) * np.maximum(effective, _EPS)))
        weight *= p @ factors
    elif balance != "natural":
        raise SharedResidualExpertError(f"unknown environment balance {balance!r}")
    if label_certainty_column is not None:
        if label_certainty_column not in frame:
            raise SharedResidualExpertError("label certainty column is missing")
        certainty = pd.to_numeric(frame[label_certainty_column], errors="coerce").to_numpy(float)
        if not np.isfinite(certainty).all() or (certainty < 0).any() or (certainty > 1).any():
            raise SharedResidualExpertError("label certainty must be finite on [0, 1]")
        weight *= 0.5 + 0.5 * certainty
    # Alternating normalization and projection preserves both parts of the
    # contract.  A single normalize-then-clip pass can leave the final mean
    # materially different from one when an imbalanced environment hits a
    # bound.
    weight = np.clip(weight, floor, cap)
    for _ in range(32):
        previous = weight.copy()
        weight = np.clip(weight / float(weight.mean()), floor, cap)
        if np.max(np.abs(weight - previous)) < 1e-12:
            break
    if not np.isclose(float(weight.mean()), 1.0, rtol=0.0, atol=1e-7):
        raise SharedResidualExpertError(
            "weight floor/cap make mean-one normalization infeasible"
        )
    return weight.astype(np.float32)


def robust_cross_era_selection_score(
    environment_scores: Mapping[str, float] | pd.Series,
    *,
    worst_penalty: float = 0.50,
    dispersion_penalty: float = 0.25,
) -> dict[str, float]:
    """Score a higher-is-better metric without allowing one era to dominate."""
    values = np.asarray(list(dict(environment_scores).values()), dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise SharedResidualExpertError("robust selection requires at least two finite environment scores")
    if worst_penalty < 0 or dispersion_penalty < 0:
        raise SharedResidualExpertError("robust selection penalties must be non-negative")
    mean = float(values.mean()); worst = float(values.min()); dispersion = float(values.std(ddof=0))
    return {
        "mean_environment_score": mean,
        "median_environment_score": float(np.median(values)),
        "worst_environment_score": worst,
        "environment_dispersion": dispersion,
        "positive_environment_count": float((values > 0).sum()),
        "environment_count": float(len(values)),
        "selection_score": mean - float(worst_penalty) * (mean - worst) - float(dispersion_penalty) * dispersion,
    }


def classify_cross_era_feature_transport(
    mda: pd.DataFrame,
    *,
    feature_column: str = "feature_group",
    environment_column: str = "environment",
    importance_column: str = "transport_mda",
    phantom_threshold: float = 0.0,
    conditioned_importance_column: str | None = None,
    latest_environment: str | None = None,
    min_positive_fraction: float = 0.70,
) -> pd.DataFrame:
    """Classify MDA groups for the one shared expert, not local experts.

    ``conditioned_importance_column`` is the same group's MDA *after* its
    declared restricted soft-regime interaction is present.  A group can then
    be admitted as ``REGIME_CONDITIONAL`` without promoting it to a local
    regime-only model.  Phantom thresholding is intentionally supplied by the
    caller's fold-local MDA artifact; this helper never estimates it from test
    eras.
    """
    required = [feature_column, environment_column, importance_column]
    missing = [name for name in required if name not in mda]
    if missing:
        raise SharedResidualExpertError(f"transport MDA lacks columns: {missing}")
    if not 0.0 < min_positive_fraction <= 1.0:
        raise SharedResidualExpertError("min_positive_fraction must be in (0, 1]")
    work = mda.loc[:, [*required, *([conditioned_importance_column] if conditioned_importance_column else [])]].copy()
    work[importance_column] = pd.to_numeric(work[importance_column], errors="coerce")
    if work[importance_column].isna().any():
        raise SharedResidualExpertError("transport MDA importance must be finite")
    if conditioned_importance_column:
        work[conditioned_importance_column] = pd.to_numeric(work[conditioned_importance_column], errors="coerce")
        if work[conditioned_importance_column].isna().any():
            raise SharedResidualExpertError("conditioned transport MDA must be finite")
    rows: list[dict[str, Any]] = []
    for group, local in work.groupby(feature_column, sort=True, observed=True):
        values = local[importance_column].to_numpy(float)
        positive_fraction = float((values > phantom_threshold).mean())
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        transport_mda = median - 0.5 * mad
        latest = local
        if latest_environment is not None:
            selected = local.loc[local[environment_column].astype(str).eq(str(latest_environment))]
            if not selected.empty:
                latest = selected
        else:
            final_name = sorted(local[environment_column].astype(str).unique())[-1]
            latest = local.loc[local[environment_column].astype(str).eq(final_name)]
        latest_mda = float(latest[importance_column].median())
        # A single small negative perturbation is not a severe reversal.  A
        # comparable negative effect is: that feature has changed meaning.
        severe_reversal = bool(values.min() < phantom_threshold and values.max() > phantom_threshold and abs(values.min()) >= abs(values.max()))
        invariant = (
            positive_fraction >= min_positive_fraction
            and transport_mda > phantom_threshold
            and latest_mda >= phantom_threshold
            and not severe_reversal
        )
        conditioned = False
        conditioned_transport_mda = float("nan")
        conditioned_latest_mda = float("nan")
        if conditioned_importance_column:
            c = local[conditioned_importance_column].to_numpy(float)
            c_median = float(np.median(c))
            c_mad = float(np.median(np.abs(c - c_median)))
            conditioned_transport_mda = c_median - 0.5 * c_mad
            conditioned_latest_mda = float(
                latest[conditioned_importance_column].median()
            )
            conditioned = bool(
                (c > phantom_threshold).mean() >= min_positive_fraction
                and conditioned_transport_mda > phantom_threshold
                and conditioned_latest_mda >= phantom_threshold
                and not (c.min() < phantom_threshold and c.max() > phantom_threshold and abs(c.min()) >= abs(c.max()))
            )
        if invariant:
            classification = "INVARIANT_CORE"
        elif conditioned:
            classification = "REGIME_CONDITIONAL"
        elif median <= phantom_threshold and float(np.max(values)) <= phantom_threshold:
            classification = "REDUNDANT"
        elif int((values > phantom_threshold).sum()) == 1 and not severe_reversal:
            classification = "REGIME_LOCAL_DIAGNOSTIC"
        else:
            classification = "UNSTABLE"
        rows.append({
            "feature_group": str(group), "transport_mda_median": median,
            "transport_mda_mad": mad, "transport_mda": transport_mda,
            "conditioned_transport_mda": conditioned_transport_mda,
            "conditioned_latest_environment_mda": conditioned_latest_mda,
            "positive_environment_fraction": positive_fraction,
            "latest_environment_mda": latest_mda, "severe_sign_reversal": severe_reversal,
            "classification": classification,
        })
    return pd.DataFrame(rows).sort_values("feature_group", kind="stable").reset_index(drop=True)


def prepare_shared_regime_residual_frame(
    frame: pd.DataFrame,
    *,
    soft_regime_columns: Sequence[str],
    regime_relative_feature_names: Sequence[str],
    restricted_interaction_feature_names: Sequence[str],
    columns: SharedResidualColumns = SharedResidualColumns(),
    baseline_config: SoftRegimeResidualConfig = SoftRegimeResidualConfig(),
    baseline_mode: Literal[
        "A0_current",
        "A1_side_centered",
        "A2_side_hard_regime_centered",
        "A3_soft_regime_centered",
    ] = "A3_soft_regime_centered",
    hard_regime_column: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Materialise all causal Stage-III fields, without fitting an expert."""
    baseline = prequential_soft_side_regime_residual_baseline(
        frame,
        soft_regime_columns=soft_regime_columns,
        columns=columns,
        config=baseline_config,
        baseline_mode=baseline_mode,
        hard_regime_column=hard_regime_column,
    )
    relative, relative_names = build_prequential_regime_relative_features(
        frame, feature_names=regime_relative_feature_names, soft_regime_columns=soft_regime_columns,
        columns=columns,
        min_reference_rows=baseline_config.min_global_rows,
        side_shrink_rows=baseline_config.side_shrink_rows,
        regime_shrink_rows=baseline_config.regime_shrink_rows,
        regime_weight_cap=baseline_config.regime_weight_cap,
    )
    interactions, interaction_names = build_restricted_soft_regime_interactions(
        frame, soft_regime_columns=soft_regime_columns,
        base_feature_names=restricted_interaction_feature_names,
    )
    prepared = pd.concat([frame.copy(), baseline, relative, interactions], axis=1)
    prepared["soft_regime_entropy"] = add_soft_regime_entropy(frame, soft_regime_columns=soft_regime_columns)
    prepared["shared_residual_side_is_long"] = (
        prepared[columns.side].astype(str).str.lower().eq("long").astype(np.float32)
    )
    generated = [
        "soft_regime_entropy", "shared_residual_side_is_long", *relative_names, *interaction_names,
    ]
    return prepared, generated


def fit_shared_regime_residual_expert(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    fit_before_utc: object,
    columns: SharedResidualColumns = SharedResidualColumns(),
    target_mode: Literal["huber", "clipped", "regime_standardized"] = "huber",
    sample_weight: Sequence[float] | None = None,
    params: Mapping[str, Any] | None = None,
) -> SharedResidualExpertFit:
    """Fit the single shared residual expert on a strictly prior-resolved set."""
    if target_mode not in {"huber", "clipped", "regime_standardized"}:
        raise SharedResidualExpertError(f"unsupported residual target mode {target_mode!r}")
    names = tuple(dict.fromkeys(str(name) for name in feature_names))
    if not names:
        raise SharedResidualExpertError("shared residual expert needs frozen feature names")
    missing = [name for name in names if name not in frame]
    if missing:
        raise SharedResidualExpertError(f"shared residual feature columns are missing: {missing[:12]}")
    suspicious = [name for name in names if any(token in name.lower() for token in _FORBIDDEN_FEATURE_TOKENS)]
    if suspicious:
        raise SharedResidualExpertError(f"outcome-derived fields cannot enter shared residual expert: {suspicious[:12]}")
    hard_regime = [
        name for name in names
        if any(token in name.lower() for token in _HARD_REGIME_FEATURE_TOKENS)
    ]
    if hard_regime:
        raise SharedResidualExpertError(
            "hard regime identifiers cannot enter the shared residual expert; "
            f"use causal soft probabilities instead: {hard_regime[:12]}"
        )
    available = _utc_series(frame, columns.label_available_timestamp, name="label availability")
    cutoff = pd.Timestamp(fit_before_utc)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    if not (available < cutoff).all():
        raise SharedResidualExpertError("shared expert fit includes unresolved/current/future labels")
    target_column = {
        "huber": "candidate_residual_bps",
        "clipped": "candidate_residual_clipped_bps",
        "regime_standardized": "candidate_residual_standardized",
    }[target_mode]
    if target_column not in frame:
        raise SharedResidualExpertError(f"prepared frame lacks target {target_column!r}")
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(np.float32)
    if not np.isfinite(target).all():
        raise SharedResidualExpertError("shared residual target has unavailable prequential burn-in rows")
    if sample_weight is None:
        weight = None
    else:
        weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        if len(weight) != len(frame) or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
            raise SharedResidualExpertError("sample weights must be aligned, finite, non-negative and non-empty")
    model_params = dict(params or {})
    model_params.setdefault("objective", "huber")
    model = _fit_lgbm_model(
        frame.loc[:, names], target, weight, classifier=False, params=model_params,
        objective_mode="shared_regime_residual_expert",
    )
    return SharedResidualExpertFit(
        model=model, feature_names=names, target_mode=target_mode,
        training_cutoff_utc=cutoff, max_label_available_utc=pd.Timestamp(available.max()), rows=len(frame),
    )


def reconstruct_shared_regime_expected_net_bps(
    frame: pd.DataFrame,
    candidate_residual_bps: Sequence[float],
    *,
    columns: SharedResidualColumns = SharedResidualColumns(),
) -> np.ndarray:
    """Reconstruct the shared model's comparable bps score before calibration."""
    required = [columns.base_expected_net_bps, "prequential_soft_regime_prior_residual_bps"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise SharedResidualExpertError(f"shared reconstruction lacks fields: {missing}")
    return reconstruct_shared_common_bps(
        pd.to_numeric(frame[columns.base_expected_net_bps], errors="coerce").to_numpy(float),
        pd.to_numeric(frame["prequential_soft_regime_prior_residual_bps"], errors="coerce").to_numpy(float),
        candidate_residual_bps,
    ).astype(np.float32, copy=False)


__all__ = [
    "SCHEMA", "SharedResidualColumns", "SharedResidualExpertError", "SharedResidualExpertFit",
    "SoftRegimeResidualConfig", "add_soft_regime_entropy", "build_prequential_regime_relative_features",
    "build_restricted_soft_regime_interactions", "fit_shared_regime_residual_expert",
    "mild_environment_weights", "prequential_soft_side_regime_residual_baseline",
    "prepare_shared_regime_residual_frame", "reconstruct_shared_regime_expected_net_bps",
    "robust_cross_era_selection_score", "classify_cross_era_feature_transport",
]
