"""Economic target construction for TBM label artifacts.

The functions in this module are deliberately label-side only: they consume
already materialized TBM/path outcome columns and create auditable target columns.
They do not generate features or look forward from the feature side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


ECONOMIC_TARGET_COLUMNS: tuple[str, ...] = (
    "__u_econ_source__",
    "__u_econ_net__",
    "__u_econ_adjusted_net__",
    "__u_econ_vol_norm__",
    "__y_econ_reg__",
    "__y_econ_pos__",
    "__y_econ_soft__",
    "__y_econ_bin__",
    "__u_econ_sideaware_net__",
    "__u_econ_sideaware_adjusted_net__",
    "__y_econ_sideaware_reg__",
    "__y_econ_sideaware_pos__",
    "__y_econ_sideaware_soft__",
    "__y_econ_sideaware_bin__",
    "__econ_sideaware_clean__",
    "__econ_sideaware_reason_code__",
    "__u_econ_side_resolution_net__",
    "__u_econ_side_resolution_adjusted_net__",
    "__y_econ_side_resolution_reg__",
    "__y_econ_side_resolution_pos__",
    "__y_econ_side_resolution_soft__",
    "__y_econ_side_resolution_bin__",
    "__econ_side_resolution_clean__",
    "__econ_side_resolution_dirty_positive__",
    "__econ_side_resolution_reason_code__",
    "__econ_side_resolution_geometry_bucket__",
    "__u_econ_sideaware_execres_net__",
    "__u_econ_sideaware_execres_adjusted_net__",
    "__y_econ_sideaware_execres_reg__",
    "__y_econ_sideaware_execres_pos__",
    "__y_econ_sideaware_execres_soft__",
    "__y_econ_sideaware_execres_bin__",
    "__econ_sideaware_execres_clean__",
    "__econ_sideaware_execres_dirty_positive__",
    "__econ_sideaware_execres_reason_code__",
    "__econ_sideaware_execres_geometry_bucket__",
    "__econ_cost__",
    "__econ_margin__",
    "__econ_sl_pct__",
    "__econ_tp_pct__",
    "__econ_vol_denom__",
    "__econ_feasible__",
    "__econ_target_weight__",
)


@dataclass(frozen=True)
class EconomicTargetSpec:
    """Parameters for one economically constrained target candidate."""

    name: str = "econ_y_ret_m50"
    utility_source: str = "y_ret"
    cost: float = 0.01
    margin: float = 0.005
    sl_buffer: float = 1.2
    vol_source: str = "barrier"
    temperature: float = 0.75
    clip_abs: float = 8.0
    mae_penalty: float = 0.0
    timeout_penalty: float = 0.0
    min_vol: float = 1e-5

    def with_name(self) -> "EconomicTargetSpec":
        if self.name:
            return self
        cost_bps = int(round(float(self.cost) * 10000.0))
        margin_bps = int(round(float(self.margin) * 10000.0))
        temp_tag = int(round(float(self.temperature) * 100.0))
        return replace(
            self,
            name=(
                f"econ_{self.utility_source}_{self.vol_source}_"
                f"c{cost_bps:04d}_"
                f"m{margin_bps:04d}_"
                f"mae{int(round(self.mae_penalty * 100.0)):02d}_"
                f"to{int(round(self.timeout_penalty * 100.0)):02d}_"
                f"t{temp_tag:03d}"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.with_name())


def _safe_numeric(values: Any, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(np.nan, index=index, dtype=np.float64)
    if isinstance(values, pd.Series):
        return pd.to_numeric(values.reindex(index), errors="coerce")
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _numeric_col(frame: pd.DataFrame, column: str) -> pd.Series:
    return _safe_numeric(frame[column] if column in frame.columns else None, frame.index)


def _first_positive(frame: pd.DataFrame, columns: Iterable[str], fallback: float) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    for column in columns:
        if column not in frame.columns:
            continue
        vals = _numeric_col(frame, column).abs()
        vals = vals.where(np.isfinite(vals) & (vals > 0.0))
        out = out.where(out.notna(), vals)
    return out.fillna(float(fallback)).clip(lower=1e-8)


def _first_finite(frame: pd.DataFrame, columns: Iterable[str], fallback: float) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    for column in columns:
        if column not in frame.columns:
            continue
        vals = _numeric_col(frame, column)
        vals = vals.where(np.isfinite(vals))
        out = out.where(out.notna(), vals)
    return out.fillna(float(fallback))


def infer_side(frame: pd.DataFrame) -> pd.Series:
    if "__side__" in frame.columns:
        raw = _numeric_col(frame, "__side__")
    elif "side" in frame.columns:
        raw = _numeric_col(frame, "side")
    else:
        raw = pd.Series(1.0, index=frame.index, dtype=np.float64)
    return pd.Series(np.where(raw.fillna(1.0) < 0.0, -1, 1), index=frame.index, dtype=np.int8)


def infer_sl_pct(frame: pd.DataFrame) -> pd.Series:
    return _first_positive(frame, ("__sl__", "sl", "__barrier_pct__"), fallback=0.01)


def infer_tp_pct(frame: pd.DataFrame) -> pd.Series:
    return _first_positive(frame, ("__tp__", "tp", "__barrier_pct__"), fallback=0.02)


def infer_vol_denom(frame: pd.DataFrame, spec: EconomicTargetSpec) -> pd.Series:
    source = str(spec.vol_source or "barrier").strip()
    lower = source.lower()
    if lower in {"sl", "stop", "stop_loss"}:
        vol = infer_sl_pct(frame)
    elif lower in {"tp", "take_profit"}:
        vol = infer_tp_pct(frame)
    elif lower in {"barrier", "__barrier_pct__"}:
        vol = _first_positive(frame, ("__barrier_pct__", "__tp__", "__sl__"), fallback=0.01)
    elif lower in {"max_sl_barrier", "barrier_or_sl"}:
        vol = pd.concat(
            [
                infer_sl_pct(frame),
                _first_positive(frame, ("__barrier_pct__", "__tp__"), fallback=0.01),
            ],
            axis=1,
        ).max(axis=1)
    elif source in frame.columns:
        vol = _numeric_col(frame, source).abs()
    else:
        vol = _first_positive(frame, ("__barrier_pct__", "__tp__", "__sl__"), fallback=0.01)
        source = "barrier_fallback"
    vol = vol.where(np.isfinite(vol) & (vol > 0.0))
    fallback = float(vol.dropna().median()) if vol.notna().any() else 0.01
    if not math.isfinite(fallback) or fallback <= 0.0:
        fallback = 0.01
    return vol.fillna(fallback).clip(lower=max(float(spec.min_vol), 1e-8))


def _mae_abs(frame: pd.DataFrame) -> pd.Series:
    raw = _numeric_col(frame, "__mae_ret__")
    finite = raw.dropna()
    if len(finite) and float(finite.median()) < 0.0:
        return (-raw).clip(lower=0.0).fillna(0.0)
    return raw.clip(lower=0.0).fillna(0.0)


def _mfe_abs(frame: pd.DataFrame) -> pd.Series:
    return _numeric_col(frame, "__mfe_ret__").clip(lower=0.0).fillna(0.0)


def utility_source(frame: pd.DataFrame, spec: EconomicTargetSpec) -> pd.Series:
    """Return a side-aware gross utility proxy before applying this spec's cost."""

    source = str(spec.utility_source or "y_ret").strip().lower()
    y_ret = _numeric_col(frame, "__y_ret__").fillna(0.0)
    policy = _numeric_col(frame, "__u_policy_net__")
    if source in {"policy_net", "u_policy_net", "__u_policy_net__"}:
        return policy.where(policy.notna(), y_ret).fillna(0.0)
    if source in {"conservative", "min_policy_ret", "min_policy_y_ret"}:
        base = policy.where(policy.notna(), y_ret)
        return pd.concat([base, y_ret], axis=1).min(axis=1).fillna(0.0)
    if source in {"path_adjusted", "path_adjusted_return"}:
        mae = _mae_abs(frame)
        mfe = _mfe_abs(frame)
        barrier = _first_positive(frame, ("__barrier_pct__", "__tp__", "__sl__"), fallback=0.01)
        path_bonus = 0.25 * np.minimum(mfe, barrier)
        path_drag = 0.25 * np.maximum(mae - infer_sl_pct(frame), 0.0)
        return (y_ret + path_bonus - path_drag).fillna(0.0)
    if source in {"mfe_mae", "excursion_net"}:
        return (_mfe_abs(frame) - _mae_abs(frame)).fillna(0.0)
    return y_ret.fillna(0.0)


def _build_sideaware_path_target(
    frame: pd.DataFrame,
    *,
    net: pd.Series,
    margin: float,
    vol: pd.Series,
    sl_pct: pd.Series,
    tp_pct: pd.Series,
    mae: pd.Series,
    is_timeout: pd.Series,
    spec: EconomicTargetSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a side-specific executable-path target.

    This leaves the generic economic target intact and adds a stricter target
    whose hard labels encode the current Gate 3 diagnosis:
    long positives decayed in high-barrier/late-resolution slices, while short
    positives were more stable in the 4-14 bar region but dirty in very fast and
    late/timeout-heavy paths.
    """

    idx = frame.index
    side = infer_side(frame).astype(np.int8)
    is_long = side > 0
    is_short = side < 0
    barrier = _first_positive(frame, ("__barrier_pct__", "__tp__", "__sl__"), fallback=0.01)
    mfe = _mfe_abs(frame).astype(np.float64)
    mae_norm = (mae.astype(np.float64) / barrier.astype(np.float64)).replace(
        [np.inf, -np.inf], np.nan
    )
    mfe_norm = (mfe / barrier.astype(np.float64)).replace([np.inf, -np.inf], np.nan)
    mae_norm = mae_norm.fillna(99.0)
    mfe_norm = mfe_norm.fillna(0.0)
    mfe_mae = (mfe / np.maximum(mae.astype(np.float64), 0.25 * barrier.astype(np.float64))).replace(
        [np.inf, -np.inf], np.nan
    )
    mfe_mae = mfe_mae.fillna(0.0)

    bars_policy = _first_finite(
        frame,
        (
            "__bars_policy__",
            "__exit_bars__",
            "__bars_to_exit__",
            "__holding_bars__",
            "__horizon_bars__",
        ),
        fallback=999.0,
    ).clip(lower=0.0)
    bars_to_mfe = _first_finite(
        frame,
        (
            "__bars_to_mfe__",
            "__time_to_mfe__",
            "__mfe_bars__",
            "__bars_mfe__",
            "__exit_bars__",
            "__bars_to_exit__",
            "__holding_bars__",
        ),
        fallback=999.0,
    ).clip(lower=0.0)

    timeout = pd.Series(is_timeout.to_numpy(dtype=bool), index=idx)
    feasible = (float(margin) + float(spec.cost)) > (float(spec.sl_buffer) * sl_pct.astype(np.float64))
    positive_net = net.astype(np.float64) > float(margin)
    late_path = bars_policy >= 15.0
    long_high_barrier = is_long & (barrier.astype(np.float64) > 0.014)
    short_too_early = is_short & (bars_policy <= 3.0)

    long_clean = (
        is_long
        & positive_net
        & feasible
        & ~timeout
        & ~long_high_barrier
        & ~late_path
        & (bars_policy <= 8.0)
        & (bars_to_mfe <= 8.0)
        & (mae_norm <= 0.75)
        & (mfe_norm >= 1.25)
        & (mfe_mae >= 1.50)
    )
    short_clean = (
        is_short
        & positive_net
        & feasible
        & ~timeout
        & ~late_path
        & (bars_policy >= 4.0)
        & (bars_policy <= 14.0)
        & (bars_to_mfe <= 14.0)
        & (mae_norm <= 0.75)
        & (mfe_norm >= 1.25)
        & (mfe_mae >= 1.50)
    )
    path_clean = (long_clean | short_clean).astype(bool)

    timeout_penalty = timeout.astype(np.float64) * np.maximum(sl_pct.astype(np.float64), barrier.astype(np.float64))
    bad_mae_penalty = np.maximum(mae_norm - 0.65, 0.0) * barrier.astype(np.float64)
    weak_mfe_penalty = np.maximum(1.25 - mfe_norm, 0.0) * barrier.astype(np.float64)
    late_penalty = late_path.astype(np.float64) * 0.50 * barrier.astype(np.float64)
    long_barrier_penalty = (
        long_high_barrier.astype(np.float64)
        * np.maximum(barrier.astype(np.float64) - 0.014, 0.0)
        * 1.50
    )
    short_early_penalty = short_too_early.astype(np.float64) * 0.35 * barrier.astype(np.float64)
    sideaware_adjusted = (
        net.astype(np.float64)
        - 0.75 * bad_mae_penalty
        - 0.50 * weak_mfe_penalty
        - timeout_penalty
        - late_penalty
        - long_barrier_penalty
        - short_early_penalty
    )
    sideaware_edge = sideaware_adjusted - float(margin)
    sideaware_reg = np.clip(
        sideaware_edge / vol.astype(np.float64).clip(lower=max(float(spec.min_vol), 1e-8)),
        -float(spec.clip_abs),
        float(spec.clip_abs),
    )
    sideaware_soft = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                sideaware_reg / max(float(spec.temperature), 1e-8),
                -60.0,
                60.0,
            )
        )
    )
    sideaware_soft = np.where(feasible, sideaware_soft, np.minimum(sideaware_soft, 0.05))
    sideaware_hard = path_clean & (sideaware_adjusted > float(margin))

    # The base model trains on the soft target. Keep profitable-but-dirty rows
    # visibly below clean executable paths so they cannot sit close together in
    # the supervised ranking objective.
    dirty_cap = np.full(len(frame), 0.18, dtype=np.float64)
    dirty_cap = np.where(timeout.to_numpy(dtype=bool), 0.04, dirty_cap)
    dirty_cap = np.where(late_path.to_numpy(dtype=bool), np.minimum(dirty_cap, 0.06), dirty_cap)
    dirty_cap = np.where((mae_norm > 0.75).to_numpy(dtype=bool), np.minimum(dirty_cap, 0.08), dirty_cap)
    dirty_cap = np.where((mfe_norm < 1.25).to_numpy(dtype=bool), np.minimum(dirty_cap, 0.10), dirty_cap)
    dirty_cap = np.where(
        (long_high_barrier | short_too_early).to_numpy(dtype=bool),
        np.minimum(dirty_cap, 0.12),
        dirty_cap,
    )
    dirty_cap = np.where(~positive_net.to_numpy(dtype=bool), np.minimum(dirty_cap, 0.02), dirty_cap)
    clean_floor = np.where(sideaware_hard.to_numpy(dtype=bool), 0.55, 0.0)
    sideaware_soft = np.where(
        path_clean.to_numpy(dtype=bool),
        np.maximum(sideaware_soft, clean_floor),
        np.minimum(sideaware_soft, dirty_cap),
    )

    reason = np.zeros(len(frame), dtype=np.int16)
    reason[~positive_net.to_numpy(dtype=bool)] = 1
    reason[timeout.to_numpy(dtype=bool)] = 2
    reason[(mae_norm > 0.75).to_numpy(dtype=bool)] = 3
    reason[(mfe_norm < 1.25).to_numpy(dtype=bool)] = 4
    reason[(mfe_mae < 1.50).to_numpy(dtype=bool)] = 5
    reason[late_path.to_numpy(dtype=bool)] = 6
    reason[long_high_barrier.to_numpy(dtype=bool)] = 7
    reason[short_too_early.to_numpy(dtype=bool)] = 8
    reason[~feasible.to_numpy(dtype=bool)] = 9
    reason[sideaware_hard.to_numpy(dtype=bool)] = 0

    out = pd.DataFrame(index=idx)
    out["__u_econ_sideaware_net__"] = net.astype(np.float32)
    out["__u_econ_sideaware_adjusted_net__"] = sideaware_adjusted.astype(np.float32)
    out["__y_econ_sideaware_reg__"] = sideaware_reg.astype(np.float32)
    out["__y_econ_sideaware_pos__"] = np.maximum(sideaware_reg, 0.0).astype(np.float32)
    out["__y_econ_sideaware_soft__"] = np.clip(sideaware_soft, 0.0, 1.0).astype(np.float32)
    out["__y_econ_sideaware_bin__"] = sideaware_hard.astype(np.int8)
    out["__econ_sideaware_clean__"] = path_clean.astype(np.int8)
    out["__econ_sideaware_reason_code__"] = reason.astype(np.int16)

    long_mask = is_long.to_numpy(dtype=bool)
    short_mask = is_short.to_numpy(dtype=bool)
    summary = {
        "sideaware_hard_rate": float(np.nanmean(out["__y_econ_sideaware_bin__"])),
        "sideaware_clean_rate": float(np.nanmean(out["__econ_sideaware_clean__"])),
        "sideaware_soft_mean": float(np.nanmean(out["__y_econ_sideaware_soft__"])),
        "sideaware_soft_std": float(np.nanstd(out["__y_econ_sideaware_soft__"])),
        "sideaware_long_hard_rate": float(np.nanmean(out.loc[long_mask, "__y_econ_sideaware_bin__"]))
        if long_mask.any()
        else float("nan"),
        "sideaware_short_hard_rate": float(np.nanmean(out.loc[short_mask, "__y_econ_sideaware_bin__"]))
        if short_mask.any()
        else float("nan"),
        "sideaware_timeout_block_rate": float(np.nanmean(timeout)),
        "sideaware_late_block_rate": float(np.nanmean(late_path)),
        "sideaware_long_high_barrier_block_rate": float(np.nanmean(long_high_barrier))
        if len(frame)
        else float("nan"),
        "sideaware_short_early_block_rate": float(np.nanmean(short_too_early))
        if len(frame)
        else float("nan"),
    }
    return out, summary


def _build_side_resolution_target(
    frame: pd.DataFrame,
    *,
    net: pd.Series,
    margin: float,
    vol: pd.Series,
    sl_pct: pd.Series,
    tp_pct: pd.Series,
    mae: pd.Series,
    is_timeout: pd.Series,
    spec: EconomicTargetSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a side-local executable-resolution target.

    This is a stricter Gate 3 repair candidate. It encodes the latest failure
    split directly: long candidates must resolve quickly to avoid timeout drag,
    while short candidates must have cleaner adverse excursion separation.
    """

    idx = frame.index
    side = infer_side(frame).astype(np.int8)
    is_long = side > 0
    is_short = side < 0
    barrier = _first_positive(frame, ("__barrier_pct__", "__tp__", "__sl__"), fallback=0.01)
    mfe = _mfe_abs(frame).astype(np.float64)
    mae_norm = (mae.astype(np.float64) / barrier.astype(np.float64)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(99.0)
    mfe_norm = (mfe / barrier.astype(np.float64)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mfe_mae = (
        mfe / np.maximum(mae.astype(np.float64), 0.25 * barrier.astype(np.float64))
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    bars_policy = _first_finite(
        frame,
        (
            "__bars_policy__",
            "__exit_bars__",
            "__bars_to_exit__",
            "__holding_bars__",
            "__horizon_bars__",
        ),
        fallback=999.0,
    ).clip(lower=0.0)
    bars_to_mfe = _first_finite(
        frame,
        (
            "__bars_to_mfe__",
            "__time_to_mfe__",
            "__mfe_bars__",
            "__bars_mfe__",
            "__exit_bars__",
            "__bars_to_exit__",
            "__holding_bars__",
        ),
        fallback=999.0,
    ).clip(lower=0.0)

    timeout = pd.Series(is_timeout.to_numpy(dtype=bool), index=idx)
    feasible = (float(margin) + float(spec.cost)) > (float(spec.sl_buffer) * sl_pct.astype(np.float64))
    positive_net = net.astype(np.float64) > float(margin)
    late_path = bars_policy >= 12.0
    long_slow = is_long & ((bars_policy > 6.0) | (bars_to_mfe > 6.0))
    short_dirty_mae = is_short & (mae_norm > 0.55)
    weak_mfe = mfe_norm < 1.25
    weak_ratio = mfe_mae < 1.75
    high_barrier = barrier.astype(np.float64) > 0.018

    long_clean = (
        is_long
        & positive_net
        & feasible
        & ~timeout
        & ~late_path
        & ~long_slow
        & (mae_norm <= 0.60)
        & (mfe_norm >= 1.35)
        & (mfe_mae >= 1.75)
        & (barrier.astype(np.float64) <= 0.014)
    )
    short_clean = (
        is_short
        & positive_net
        & feasible
        & ~timeout
        & ~late_path
        & (bars_policy >= 4.0)
        & (bars_policy <= 12.0)
        & (bars_to_mfe <= 10.0)
        & (mae_norm <= 0.55)
        & (mfe_norm >= 1.25)
        & (mfe_mae >= 1.75)
    )
    clean = (long_clean | short_clean).astype(bool)

    timeout_penalty = timeout.astype(np.float64) * np.maximum(sl_pct.astype(np.float64), barrier.astype(np.float64))
    long_slow_penalty = long_slow.astype(np.float64) * 0.70 * barrier.astype(np.float64)
    short_mae_penalty = short_dirty_mae.astype(np.float64) * 1.10 * barrier.astype(np.float64)
    weak_mfe_penalty = weak_mfe.astype(np.float64) * 0.55 * barrier.astype(np.float64)
    weak_ratio_penalty = weak_ratio.astype(np.float64) * 0.35 * barrier.astype(np.float64)
    late_penalty = late_path.astype(np.float64) * 0.80 * barrier.astype(np.float64)
    high_barrier_penalty = high_barrier.astype(np.float64) * 0.35 * barrier.astype(np.float64)
    adjusted = (
        net.astype(np.float64)
        - 1.40 * timeout_penalty
        - long_slow_penalty
        - short_mae_penalty
        - weak_mfe_penalty
        - weak_ratio_penalty
        - late_penalty
        - high_barrier_penalty
    )
    edge = adjusted - float(margin)
    reg = np.clip(
        edge / vol.astype(np.float64).clip(lower=max(float(spec.min_vol), 1e-8)),
        -float(spec.clip_abs),
        float(spec.clip_abs),
    )
    hard = clean & (adjusted > float(margin))

    base_soft = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                reg / max(float(spec.temperature), 1e-8),
                -60.0,
                60.0,
            )
        )
    )
    dirty_cap = np.full(len(frame), 0.12, dtype=np.float64)
    dirty_cap = np.where(timeout.to_numpy(dtype=bool), 0.02, dirty_cap)
    dirty_cap = np.where(late_path.to_numpy(dtype=bool), np.minimum(dirty_cap, 0.03), dirty_cap)
    dirty_cap = np.where(long_slow.to_numpy(dtype=bool), np.minimum(dirty_cap, 0.04), dirty_cap)
    dirty_cap = np.where(short_dirty_mae.to_numpy(dtype=bool), np.minimum(dirty_cap, 0.03), dirty_cap)
    dirty_cap = np.where(weak_mfe.to_numpy(dtype=bool), np.minimum(dirty_cap, 0.07), dirty_cap)
    dirty_cap = np.where(weak_ratio.to_numpy(dtype=bool), np.minimum(dirty_cap, 0.08), dirty_cap)
    dirty_cap = np.where(~positive_net.to_numpy(dtype=bool), np.minimum(dirty_cap, 0.01), dirty_cap)
    soft = np.where(
        clean.to_numpy(dtype=bool),
        np.maximum(base_soft, np.where(hard.to_numpy(dtype=bool), 0.65, 0.50)),
        np.minimum(base_soft, dirty_cap),
    )
    soft = np.where(feasible, soft, np.minimum(soft, 0.02))

    reason = np.zeros(len(frame), dtype=np.int16)
    reason[~positive_net.to_numpy(dtype=bool)] = 1
    reason[high_barrier.to_numpy(dtype=bool)] = 8
    reason[timeout.to_numpy(dtype=bool)] = 2
    reason[short_dirty_mae.to_numpy(dtype=bool)] = 3
    reason[long_slow.to_numpy(dtype=bool)] = 4
    reason[weak_mfe.to_numpy(dtype=bool)] = 5
    reason[weak_ratio.to_numpy(dtype=bool)] = 6
    reason[late_path.to_numpy(dtype=bool)] = 7
    reason[~feasible.to_numpy(dtype=bool)] = 9
    reason[hard.to_numpy(dtype=bool)] = 0

    dirty_positive = positive_net & ~hard
    geometry = np.zeros(len(frame), dtype=np.int16)
    geometry[is_long.to_numpy(dtype=bool) & (bars_policy.to_numpy(dtype=np.float64) <= 6.0)] = 1
    geometry[is_long.to_numpy(dtype=bool) & (bars_policy.to_numpy(dtype=np.float64) > 6.0)] = 2
    geometry[is_short.to_numpy(dtype=bool) & (bars_policy.to_numpy(dtype=np.float64) < 4.0)] = 3
    geometry[
        is_short.to_numpy(dtype=bool)
        & (bars_policy.to_numpy(dtype=np.float64) >= 4.0)
        & (bars_policy.to_numpy(dtype=np.float64) <= 12.0)
    ] = 4
    geometry[is_short.to_numpy(dtype=bool) & (bars_policy.to_numpy(dtype=np.float64) > 12.0)] = 5
    geometry[high_barrier.to_numpy(dtype=bool)] = 6
    geometry[timeout.to_numpy(dtype=bool)] = 7

    out = pd.DataFrame(index=idx)
    out["__u_econ_side_resolution_net__"] = net.astype(np.float32)
    out["__u_econ_side_resolution_adjusted_net__"] = adjusted.astype(np.float32)
    out["__y_econ_side_resolution_reg__"] = reg.astype(np.float32)
    out["__y_econ_side_resolution_pos__"] = np.maximum(reg, 0.0).astype(np.float32)
    out["__y_econ_side_resolution_soft__"] = np.clip(soft, 0.0, 1.0).astype(np.float32)
    out["__y_econ_side_resolution_bin__"] = hard.astype(np.int8)
    out["__econ_side_resolution_clean__"] = clean.astype(np.int8)
    out["__econ_side_resolution_dirty_positive__"] = dirty_positive.astype(np.int8)
    out["__econ_side_resolution_reason_code__"] = reason.astype(np.int16)
    out["__econ_side_resolution_geometry_bucket__"] = geometry.astype(np.int16)

    alias_pairs = {
        "__u_econ_sideaware_execres_net__": "__u_econ_side_resolution_net__",
        "__u_econ_sideaware_execres_adjusted_net__": "__u_econ_side_resolution_adjusted_net__",
        "__y_econ_sideaware_execres_reg__": "__y_econ_side_resolution_reg__",
        "__y_econ_sideaware_execres_pos__": "__y_econ_side_resolution_pos__",
        "__y_econ_sideaware_execres_soft__": "__y_econ_side_resolution_soft__",
        "__y_econ_sideaware_execres_bin__": "__y_econ_side_resolution_bin__",
        "__econ_sideaware_execres_clean__": "__econ_side_resolution_clean__",
        "__econ_sideaware_execres_dirty_positive__": "__econ_side_resolution_dirty_positive__",
        "__econ_sideaware_execres_reason_code__": "__econ_side_resolution_reason_code__",
        "__econ_sideaware_execres_geometry_bucket__": "__econ_side_resolution_geometry_bucket__",
    }
    for alias, source in alias_pairs.items():
        out[alias] = out[source].to_numpy(copy=False)

    long_mask = is_long.to_numpy(dtype=bool)
    short_mask = is_short.to_numpy(dtype=bool)
    summary = {
        "side_resolution_hard_rate": float(np.nanmean(out["__y_econ_side_resolution_bin__"])),
        "side_resolution_clean_rate": float(np.nanmean(out["__econ_side_resolution_clean__"])),
        "side_resolution_dirty_positive_rate": float(
            np.nanmean(out["__econ_side_resolution_dirty_positive__"])
        ),
        "side_resolution_soft_mean": float(np.nanmean(out["__y_econ_side_resolution_soft__"])),
        "side_resolution_soft_std": float(np.nanstd(out["__y_econ_side_resolution_soft__"])),
        "side_resolution_long_hard_rate": float(
            np.nanmean(out.loc[long_mask, "__y_econ_side_resolution_bin__"])
        )
        if long_mask.any()
        else float("nan"),
        "side_resolution_short_hard_rate": float(
            np.nanmean(out.loc[short_mask, "__y_econ_side_resolution_bin__"])
        )
        if short_mask.any()
        else float("nan"),
        "side_resolution_timeout_block_rate": float(np.nanmean(timeout)),
        "side_resolution_long_slow_block_rate": float(np.nanmean(long_slow))
        if len(frame)
        else float("nan"),
        "side_resolution_short_dirty_mae_block_rate": float(np.nanmean(short_dirty_mae))
        if len(frame)
        else float("nan"),
    }
    return out, summary


def build_economic_target(
    frame: pd.DataFrame,
    spec: EconomicTargetSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build target columns and diagnostics for one spec."""

    spec = spec.with_name()
    if len(frame) == 0:
        empty = pd.DataFrame(index=frame.index)
        for column in ECONOMIC_TARGET_COLUMNS:
            empty[column] = pd.Series(dtype=np.float32)
        return empty, {"rows": 0, "spec": spec.to_dict()}

    source_u = utility_source(frame, spec).astype(np.float64)
    sl_pct = infer_sl_pct(frame).astype(np.float64)
    tp_pct = infer_tp_pct(frame).astype(np.float64)
    vol = infer_vol_denom(frame, spec).astype(np.float64)
    mae = _mae_abs(frame).astype(np.float64)
    is_timeout = _numeric_col(frame, "__is_timeout__").fillna(0.0).astype(np.float64) > 0.5

    cost = float(spec.cost)
    margin = float(spec.margin)
    sl_buffer = float(spec.sl_buffer)
    net = source_u - cost
    adverse_excess = np.maximum(mae - sl_pct, 0.0)
    adjusted_net = (
        net
        - float(spec.mae_penalty) * adverse_excess
        - float(spec.timeout_penalty) * is_timeout.astype(np.float64) * sl_pct
    )
    feasible = (margin + cost) > (sl_buffer * sl_pct)
    edge = adjusted_net - margin
    reg = np.clip(edge / vol, -float(spec.clip_abs), float(spec.clip_abs))
    soft = 1.0 / (1.0 + np.exp(-np.clip(reg / max(float(spec.temperature), 1e-8), -60.0, 60.0)))
    soft = np.where(feasible, soft, np.minimum(soft, 0.05))
    hard = feasible & (net > margin) & (adjusted_net > margin)
    pos = np.maximum(reg, 0.0)
    weight = 1.0 + np.minimum(np.abs(reg), 3.0) / 3.0
    weight = np.where(feasible, weight, 0.25)
    sideaware, sideaware_summary = _build_sideaware_path_target(
        frame,
        net=net,
        margin=margin,
        vol=vol,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        mae=mae,
        is_timeout=pd.Series(is_timeout, index=frame.index),
        spec=spec,
    )
    side_resolution, side_resolution_summary = _build_side_resolution_target(
        frame,
        net=net,
        margin=margin,
        vol=vol,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        mae=mae,
        is_timeout=pd.Series(is_timeout, index=frame.index),
        spec=spec,
    )

    out = pd.DataFrame(index=frame.index)
    out["__u_econ_source__"] = source_u.astype(np.float32)
    out["__u_econ_net__"] = net.astype(np.float32)
    out["__u_econ_adjusted_net__"] = adjusted_net.astype(np.float32)
    out["__u_econ_vol_norm__"] = (net / vol).astype(np.float32)
    out["__y_econ_reg__"] = reg.astype(np.float32)
    out["__y_econ_pos__"] = pos.astype(np.float32)
    out["__y_econ_soft__"] = np.clip(soft, 0.0, 1.0).astype(np.float32)
    out["__y_econ_bin__"] = hard.astype(np.int8)
    for column in sideaware.columns:
        out[column] = sideaware[column].to_numpy(copy=False)
    for column in side_resolution.columns:
        out[column] = side_resolution[column].to_numpy(copy=False)
    out["__econ_cost__"] = np.full(len(frame), cost, dtype=np.float32)
    out["__econ_margin__"] = np.full(len(frame), margin, dtype=np.float32)
    out["__econ_sl_pct__"] = sl_pct.astype(np.float32)
    out["__econ_tp_pct__"] = tp_pct.astype(np.float32)
    out["__econ_vol_denom__"] = vol.astype(np.float32)
    out["__econ_feasible__"] = feasible.astype(np.int8)
    out["__econ_target_weight__"] = weight.astype(np.float32)

    finite_soft = np.isfinite(out["__y_econ_soft__"].to_numpy(dtype=np.float64))
    finite_net = np.isfinite(out["__u_econ_net__"].to_numpy(dtype=np.float64))
    summary = {
        "spec": spec.to_dict(),
        "rows": int(len(frame)),
        "finite_soft_frac": float(np.mean(finite_soft)),
        "finite_net_frac": float(np.mean(finite_net)),
        "soft_mean": float(np.nanmean(out["__y_econ_soft__"])),
        "soft_std": float(np.nanstd(out["__y_econ_soft__"])),
        "hard_rate": float(np.nanmean(out["__y_econ_bin__"])),
        "feasible_rate": float(np.nanmean(out["__econ_feasible__"])),
        "mean_net_utility": float(np.nanmean(out["__u_econ_net__"])),
        "median_net_utility": float(np.nanmedian(out["__u_econ_net__"])),
        "p90_net_utility": float(np.nanpercentile(out["__u_econ_net__"], 90)),
        "mean_sl_pct": float(np.nanmean(out["__econ_sl_pct__"])),
        "p90_sl_pct": float(np.nanpercentile(out["__econ_sl_pct__"], 90)),
        "cost": cost,
        "margin": margin,
        "sl_buffer": sl_buffer,
    }
    summary.update(sideaware_summary)
    summary.update(side_resolution_summary)
    return out, summary


def candidate_specs(
    *,
    utility_sources: Iterable[str] = ("y_ret", "conservative", "policy_net", "path_adjusted"),
    margins: Iterable[float] = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02),
    vol_sources: Iterable[str] = ("barrier", "sl", "max_sl_barrier"),
    costs: Iterable[float] = (0.01,),
    sl_buffer: float = 1.2,
    temperatures: Iterable[float] = (0.50, 0.75, 1.00),
    mae_penalties: Iterable[float] = (0.0, 0.25, 0.50),
    timeout_penalties: Iterable[float] = (0.0, 0.50),
) -> list[EconomicTargetSpec]:
    specs: list[EconomicTargetSpec] = []
    for cost in costs:
        for margin in margins:
            for source in utility_sources:
                for vol_source in vol_sources:
                    for temperature in temperatures:
                        for mae_penalty in mae_penalties:
                            for timeout_penalty in timeout_penalties:
                                spec = EconomicTargetSpec(
                                    name="",
                                    utility_source=str(source),
                                    cost=float(cost),
                                    margin=float(margin),
                                    sl_buffer=float(sl_buffer),
                                    vol_source=str(vol_source),
                                    temperature=float(temperature),
                                    mae_penalty=float(mae_penalty),
                                    timeout_penalty=float(timeout_penalty),
                                ).with_name()
                                specs.append(spec)
    return specs


def append_economic_target_columns(
    frame: pd.DataFrame,
    spec: EconomicTargetSpec,
    *,
    copy: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets, summary = build_economic_target(frame, spec)
    out = frame.copy() if copy else frame
    for column in ECONOMIC_TARGET_COLUMNS:
        out[column] = targets[column].to_numpy(copy=False)
    out["__econ_target_name__"] = str(spec.with_name().name)
    out["__econ_utility_source__"] = str(spec.utility_source)
    return out, summary


def economic_target_column_names() -> list[str]:
    return list(ECONOMIC_TARGET_COLUMNS) + ["__econ_target_name__", "__econ_utility_source__"]
