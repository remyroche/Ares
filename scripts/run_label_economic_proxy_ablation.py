#!/usr/bin/env python3
"""No-training label/economic proxy ablations.

This is a pre-training diagnostic. It does not fit LightGBM, Optuna, or policy
geometry. It asks whether current features can recover labels only after adding
an economic envelope learned causally from prior months.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    PROXY_TOP_K_FEATURES,
    ROUND_TRIP_COST,
    TOP_FRACS,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _make_targets,
    _path_metrics,
    _proxy_score,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _sigmoid,
    _spearman,
)
from scripts.run_label_quality_proxy_grid import GridSpec, _build_target


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_economic_proxy_ablation_v1")
LABEL_ARMS = (
    "S10_policy_net_soft",
    "S13_policy_net_risk_adjusted",
    "S14_policy_net_path_blend",
    "S13_grid_mild_no_time",
    "S13_grid_mild_time075",
    "S16_tail_utility_soft",
    "S17_tail_payoff_clean",
    "S18_tail_payoff_ts_rank",
    "S19_tail_mild_path_soft",
    "S20_tail_risk_adjusted_soft",
    "S21_tail_margin_ts_rank",
    "S22_policy_tail_balanced_blend",
    "S23_policy_tail_s16_lean",
    "S24_policy_tail_s14_lean",
    "S25_tail_fast_risk_soft",
    "S26_broad_policy_path_fast",
    "S27_tail_rank_risk_balanced",
    "S28_lowbarrier_broad_policy",
    "S29_lowbarrier_s24_blend",
    "S30_lowbarrier_tail_risk",
    "S31_clean_tail_economic",
    "S32_econ_limited_broad_policy",
    "S33_clean_margin_ts_rank",
    "S34_exec_guard_broad_policy",
    "S35_exec_margin_soft",
    "S36_exec_margin_ts_rank",
    "S37_lowdrawdown_tail_rank",
    "S38_conditional_clean_utility",
    "S39_conditional_clean_ts_rank",
    "S40_dirty_capped_broad_policy",
    "S41_lowmae_timeout_safe_tail",
    "S42_lowbarrier_lowmae_tail",
    "S43_lowbarrier_dirty_capped_broad",
    "S44_clean_masked_lowmae_rank",
    "S45_strict_clean_tail_rank",
    "S46_badmae_contrast_margin",
    "S47_dirty_capped_s41",
    "S48_clean_recoverable_margin_rank",
    "S49_clean_recoverable_tail_rank",
    "S50_s30_clean_recoverable_tail",
    "S51_clean_dirty_contrast_recoverable",
    "S52_timeout_barrier_cap_policy_soft",
    "S53_timeout_barrier_cap_path_blend",
    "S54_timeout_barrier_cap_clean_tail",
    "S55_timeout_barrier_cap_exec_guard",
    "S56_timeout_tpnet_cap_policy_soft",
    "S57_timeout_tpnet_cap_path_blend",
    "S58_timeout_tpnet_cap_clean_tail",
    "S59_timeout_tpnet_cap_exec_guard",
    "S60_tpnet_severe_adverse_veto_path",
    "S61_tpnet_strict_adverse_veto_rank",
    "S62_tpnet_clean_dirty_contrast_rank",
    "S63_exec_admissible_primary",
    "S64_exec_admissible_rank",
    "S65_profit_inside_exec_admissible",
    "S66_exec_admissible_contrast_rank",
)
ECONOMIC_ARMS = (
    "raw_u_policy_net",
    "risk_u_mild",
    "risk_u_strict_fast",
)
COMBINE_WEIGHTS = (0.35, 0.50, 0.65)
ECONOMIC_GATE_FRACS = (0.30, 0.10)


def _grid_label_targets(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    specs = [
        GridSpec(
            arm="S13_grid_mild_no_time",
            family="risk_adjusted_policy_net",
            mae_penalty=0.0025,
            mae_free=1.0,
            time_penalty=0.0,
            barrier_penalty=0.35,
            barrier_free=0.018,
            margin=0.0015,
            temperature=0.008,
        ),
        GridSpec(
            arm="S13_grid_mild_time075",
            family="risk_adjusted_policy_net",
            mae_penalty=0.0025,
            mae_free=1.0,
            time_penalty=0.00075,
            barrier_penalty=0.20,
            barrier_free=0.018,
            margin=0.0015,
            temperature=0.008,
        ),
    ]
    out: dict[str, pd.DataFrame] = {}
    for spec in specs:
        target, _risk_u = _build_target(metrics, spec)
        out[spec.arm] = target
    return out


def _label_targets(frame: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    targets = _make_targets(frame, metrics)
    targets.update(_grid_label_targets(metrics))
    u = metrics["u_policy_net"].fillna(-0.02)
    tail_soft = pd.Series(_sigmoid((u - 0.005) / 0.018), index=frame.index).clip(0.0, 1.0)
    mae_norm = metrics["mae_norm"].fillna(0.0)
    mfe_norm = metrics["mfe_norm"].fillna(0.0)
    bars_to_mfe = metrics["bars_to_mfe"].fillna(24.0)
    barrier = metrics["barrier"].fillna(0.0)
    mfe_mae = (mfe_norm / mae_norm.clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=10.0)
    mild_path_score = (
        0.35 * pd.Series(_sigmoid((3.50 - mae_norm) / 1.25), index=frame.index)
        + 0.25 * pd.Series(_sigmoid((0.060 - barrier) / 0.020), index=frame.index)
        + 0.20 * pd.Series(_sigmoid((18.0 - bars_to_mfe) / 8.0), index=frame.index)
        + 0.20 * pd.Series(_sigmoid((mfe_norm - 1.00) / 0.75), index=frame.index)
    ).clip(0.0, 1.0)
    risk_adjusted_u = (
        u
        - 0.0012 * (mae_norm - 1.50).clip(lower=0.0)
        - 0.00030 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.08 * (barrier - 0.025).clip(lower=0.0)
        - 0.0010 * metrics["is_timeout"].astype(float)
    )
    tail_risk_soft = pd.Series(
        _sigmoid((risk_adjusted_u - 0.003) / 0.018),
        index=frame.index,
    ).clip(0.0, 1.0)
    tail_rank = tail_soft.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    tail_rank = tail_rank.fillna(tail_soft.rank(method="average", pct=True)).clip(0.0, 1.0)
    payoff_clean = (
        tail_soft
        * pd.Series(_sigmoid((mfe_norm - 2.0) / 0.75), index=frame.index)
        * pd.Series(_sigmoid((2.50 - mae_norm) / 0.75), index=frame.index)
        * pd.Series(_sigmoid((10.0 - bars_to_mfe) / 5.0), index=frame.index)
        * pd.Series(_sigmoid((0.045 - barrier) / 0.015), index=frame.index)
    ).clip(0.0, 1.0)
    payoff_rank = payoff_clean.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    payoff_rank = payoff_rank.fillna(payoff_clean.rank(method="average", pct=True)).clip(0.0, 1.0)
    targets["S16_tail_utility_soft"] = pd.DataFrame(
        {
            "target_soft": tail_soft,
            "target_hard": (u > 0.005).fillna(False).astype(float),
        },
        index=frame.index,
    )
    s14_soft = targets["S14_policy_net_path_blend"]["target_soft"]
    targets["S17_tail_payoff_clean"] = pd.DataFrame(
        {
            "target_soft": payoff_clean,
            "target_hard": (
                (u > 0.005)
                & (mfe_norm >= 2.0)
                & (mae_norm <= 2.5)
                & (bars_to_mfe <= 10.0)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S18_tail_payoff_ts_rank"] = pd.DataFrame(
        {
            "target_soft": (0.50 * payoff_clean + 0.50 * payoff_rank).clip(0.0, 1.0),
            "target_hard": (payoff_rank >= 0.95).fillna(False).astype(float),
        },
        index=frame.index,
    )
    targets["S19_tail_mild_path_soft"] = pd.DataFrame(
        {
            "target_soft": (tail_soft * (0.75 + 0.25 * mild_path_score)).clip(0.0, 1.0),
            "target_hard": (
                (u > 0.005)
                & (mae_norm <= 4.0)
                & (barrier <= 0.075)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S20_tail_risk_adjusted_soft"] = pd.DataFrame(
        {
            "target_soft": (0.70 * tail_soft + 0.30 * tail_risk_soft).clip(0.0, 1.0),
            "target_hard": (risk_adjusted_u > 0.003).fillna(False).astype(float),
        },
        index=frame.index,
    )
    targets["S21_tail_margin_ts_rank"] = pd.DataFrame(
        {
            "target_soft": (0.75 * tail_risk_soft + 0.25 * tail_rank).clip(0.0, 1.0),
            "target_hard": ((risk_adjusted_u > 0.0) & (tail_rank >= 0.85))
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S22_policy_tail_balanced_blend"] = pd.DataFrame(
        {
            "target_soft": (0.50 * s14_soft + 0.50 * tail_soft).clip(0.0, 1.0),
            "target_hard": ((u > 0.0025) & ((s14_soft >= 0.45) | (tail_soft >= 0.65)))
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S23_policy_tail_s16_lean"] = pd.DataFrame(
        {
            "target_soft": (0.30 * s14_soft + 0.70 * tail_soft).clip(0.0, 1.0),
            "target_hard": ((u > 0.0035) & ((s14_soft >= 0.40) | (tail_soft >= 0.68)))
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S24_policy_tail_s14_lean"] = pd.DataFrame(
        {
            "target_soft": (0.70 * s14_soft + 0.30 * tail_soft).clip(0.0, 1.0),
            "target_hard": ((u > 0.0015) & ((s14_soft >= 0.48) | (tail_soft >= 0.62)))
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    fast_score = pd.Series(_sigmoid((14.0 - bars_to_mfe) / 6.0), index=frame.index)
    risk_clean_score = (
        pd.Series(_sigmoid((3.00 - mae_norm) / 1.00), index=frame.index)
        * pd.Series(_sigmoid((0.060 - barrier) / 0.020), index=frame.index)
    ).clip(0.0, 1.0)
    mfe_score = pd.Series(_sigmoid((mfe_norm - 1.20) / 0.75), index=frame.index)
    fast_risk_u = (
        u
        - 0.0010 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.0015 * (mae_norm - 1.50).clip(lower=0.0)
        - 0.12 * (barrier - 0.025).clip(lower=0.0)
    )
    fast_risk_soft = pd.Series(
        _sigmoid((fast_risk_u - 0.002) / 0.014),
        index=frame.index,
    ).clip(0.0, 1.0)
    path_fast = (0.35 * fast_score + 0.35 * risk_clean_score + 0.30 * mfe_score).clip(0.0, 1.0)
    targets["S25_tail_fast_risk_soft"] = pd.DataFrame(
        {
            "target_soft": (fast_risk_soft * (0.75 + 0.25 * path_fast)).clip(0.0, 1.0),
            "target_hard": (
                (fast_risk_u > 0.002)
                & (mae_norm <= 3.50)
                & (barrier <= 0.065)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    broad_soft = (
        0.35 * s14_soft
        + 0.30 * tail_risk_soft
        + 0.20 * tail_soft
        + 0.15 * path_fast
    ).clip(0.0, 1.0)
    targets["S26_broad_policy_path_fast"] = pd.DataFrame(
        {
            "target_soft": broad_soft,
            "target_hard": (
                (u > 0.0)
                & (mae_norm <= 4.00)
                & (barrier <= 0.075)
                & (path_fast >= 0.35)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S27_tail_rank_risk_balanced"] = pd.DataFrame(
        {
            "target_soft": (0.40 * tail_risk_soft + 0.30 * tail_rank + 0.30 * path_fast).clip(0.0, 1.0),
            "target_hard": (
                (risk_adjusted_u > 0.0)
                & (tail_rank >= 0.75)
                & (risk_clean_score >= 0.35)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    lowbarrier_score = pd.Series(_sigmoid((0.030 - barrier) / 0.008), index=frame.index).clip(0.0, 1.0)
    clean_lowbarrier = (0.55 * lowbarrier_score + 0.45 * path_fast).clip(0.0, 1.0)
    s24_soft = targets["S24_policy_tail_s14_lean"]["target_soft"]
    targets["S28_lowbarrier_broad_policy"] = pd.DataFrame(
        {
            "target_soft": (broad_soft * (0.45 + 0.55 * clean_lowbarrier)).clip(0.0, 1.0),
            "target_hard": (
                (u > 0.0)
                & (mae_norm <= 3.50)
                & (barrier <= 0.040)
                & (path_fast >= 0.35)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S29_lowbarrier_s24_blend"] = pd.DataFrame(
        {
            "target_soft": (
                (0.60 * s24_soft + 0.40 * broad_soft)
                * (0.40 + 0.60 * lowbarrier_score)
            ).clip(0.0, 1.0),
            "target_hard": (
                (u > 0.001)
                & (barrier <= 0.035)
                & (mae_norm <= 3.25)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S30_lowbarrier_tail_risk"] = pd.DataFrame(
        {
            "target_soft": (
                tail_risk_soft
                * (0.35 + 0.65 * lowbarrier_score)
                * (0.60 + 0.40 * path_fast)
            ).clip(0.0, 1.0),
            "target_hard": (
                (risk_adjusted_u > 0.001)
                & (barrier <= 0.035)
                & (mae_norm <= 3.00)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    clean_mae_score = pd.Series(_sigmoid((1.00 - mae_norm) / 0.35), index=frame.index).clip(0.0, 1.0)
    clean_barrier_score = pd.Series(_sigmoid((0.025 - barrier) / 0.006), index=frame.index).clip(0.0, 1.0)
    clean_speed_score = pd.Series(_sigmoid((12.0 - bars_to_mfe) / 5.0), index=frame.index).clip(0.0, 1.0)
    clean_mfe_score = pd.Series(_sigmoid((mfe_norm - 1.35) / 0.55), index=frame.index).clip(0.0, 1.0)
    economic_clean_score = (
        clean_mae_score * clean_barrier_score * (0.55 + 0.45 * clean_speed_score) * (0.50 + 0.50 * clean_mfe_score)
    ).clip(0.0, 1.0)
    clean_margin_u = (
        u
        - 0.0020
        - 0.0040 * (mae_norm - 0.75).clip(lower=0.0)
        - 0.45 * (barrier - 0.020).clip(lower=0.0)
        - 0.0010 * np.log1p(bars_to_mfe.clip(lower=0.0))
    )
    clean_margin_soft = pd.Series(_sigmoid(clean_margin_u / 0.010), index=frame.index).clip(0.0, 1.0)
    clean_margin_rank = clean_margin_soft.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    clean_margin_rank = clean_margin_rank.fillna(clean_margin_soft.rank(method="average", pct=True)).clip(0.0, 1.0)
    exec_mae_score = pd.Series(_sigmoid((0.85 - mae_norm) / 0.25), index=frame.index).clip(0.0, 1.0)
    exec_barrier_score = pd.Series(_sigmoid((0.022 - barrier) / 0.004), index=frame.index).clip(0.0, 1.0)
    exec_speed_score = pd.Series(_sigmoid((8.0 - bars_to_mfe) / 3.0), index=frame.index).clip(0.0, 1.0)
    exec_mfe_score = pd.Series(_sigmoid((mfe_norm - 1.15) / 0.45), index=frame.index).clip(0.0, 1.0)
    exec_clean_score = (
        exec_mae_score
        * exec_barrier_score
        * (0.40 + 0.60 * exec_speed_score)
        * (0.35 + 0.65 * exec_mfe_score)
    ).clip(0.0, 1.0)
    exec_margin_u = (
        u
        - 0.0015
        - 0.0060 * (mae_norm - 0.65).clip(lower=0.0)
        - 0.65 * (barrier - 0.018).clip(lower=0.0)
        - 0.0014 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.0020 * metrics["is_timeout"].astype(float)
    )
    exec_margin_soft = pd.Series(_sigmoid(exec_margin_u / 0.009), index=frame.index).clip(0.0, 1.0)
    exec_margin_rank = exec_margin_soft.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    exec_margin_rank = exec_margin_rank.fillna(exec_margin_soft.rank(method="average", pct=True)).clip(0.0, 1.0)
    lowdrawdown_tail = (
        tail_soft
        * (0.10 + 0.90 * exec_clean_score)
        * pd.Series(_sigmoid((exec_margin_u + 0.002) / 0.010), index=frame.index)
    ).clip(0.0, 1.0)
    lowdrawdown_tail_rank = lowdrawdown_tail.groupby(frame["__ts__"], dropna=False).rank(
        method="average",
        pct=True,
    )
    lowdrawdown_tail_rank = lowdrawdown_tail_rank.fillna(
        lowdrawdown_tail.rank(method="average", pct=True),
    ).clip(0.0, 1.0)
    timeout_float = metrics["is_timeout"].astype(float)

    def _frame_numeric(name: str, fallback: pd.Series) -> pd.Series:
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce")
        else:
            values = pd.Series(np.nan, index=frame.index)
        return values.fillna(fallback).astype(float)

    effective_tp = _frame_numeric(
        "__first_touch_effective_tp_abs__",
        _frame_numeric("__tp__", barrier * 0.75),
    ).abs()
    effective_sl = _frame_numeric(
        "__first_touch_effective_sl_abs__",
        _frame_numeric("__sl__", barrier * 1.50),
    ).abs()
    timeout_mask = timeout_float > 0.5
    timeout_positive = timeout_mask & (u > 0.0)
    barrier_cap = pd.concat([effective_tp, barrier], axis=1).max(axis=1).clip(lower=0.00075)
    tpnet_cap = (effective_tp - ROUND_TRIP_COST).clip(lower=0.00075)
    loss_floor = -(effective_sl + ROUND_TRIP_COST).clip(lower=0.00075)

    def _cap_timeout_positive(positive_cap: pd.Series) -> pd.Series:
        capped = u.copy()
        capped = capped.where(~timeout_positive, np.minimum(capped, positive_cap))
        capped = capped.where(~timeout_mask, np.maximum(capped, loss_floor))
        return pd.Series(capped, index=frame.index).replace([np.inf, -np.inf], np.nan).fillna(-0.02)

    u_timeout_barrier_cap = _cap_timeout_positive(barrier_cap)
    u_timeout_tpnet_cap = _cap_timeout_positive(tpnet_cap)
    base_policy_soft = targets["S10_policy_net_soft"]["target_soft"]
    base_path_blend = targets["S14_policy_net_path_blend"]["target_soft"]
    path_component_proxy = (2.0 * base_path_blend - base_policy_soft).clip(0.0, 1.0)

    def _capped_family(capped_u: pd.Series) -> dict[str, pd.Series]:
        capped_policy_soft = pd.Series(_sigmoid(capped_u / 0.012), index=frame.index).clip(0.0, 1.0)
        capped_tail_soft = pd.Series(_sigmoid((capped_u - 0.005) / 0.018), index=frame.index).clip(0.0, 1.0)
        capped_risk_u = (
            capped_u
            - 0.0012 * (mae_norm - 1.50).clip(lower=0.0)
            - 0.00030 * np.log1p(bars_to_mfe.clip(lower=0.0))
            - 0.08 * (barrier - 0.025).clip(lower=0.0)
            - 0.0010 * timeout_float
        )
        capped_tail_risk_soft = pd.Series(
            _sigmoid((capped_risk_u - 0.003) / 0.018),
            index=frame.index,
        ).clip(0.0, 1.0)
        capped_path_blend = (0.50 * capped_policy_soft + 0.50 * path_component_proxy).clip(0.0, 1.0)
        capped_broad_soft = (
            0.35 * capped_path_blend
            + 0.30 * capped_tail_risk_soft
            + 0.20 * capped_tail_soft
            + 0.15 * path_fast
        ).clip(0.0, 1.0)
        capped_clean_tail = (capped_tail_soft * (0.20 + 0.80 * economic_clean_score)).clip(0.0, 1.0)
        capped_exec_guard = (capped_broad_soft * (0.05 + 0.95 * exec_clean_score)).clip(0.0, 1.0)
        return {
            "policy_soft": capped_policy_soft,
            "tail_soft": capped_tail_soft,
            "path_blend": capped_path_blend,
            "clean_tail": capped_clean_tail,
            "exec_guard": capped_exec_guard,
        }

    barrier_capped = _capped_family(u_timeout_barrier_cap)
    tpnet_capped = _capped_family(u_timeout_tpnet_cap)
    survival_clean_score = (
        pd.Series(_sigmoid((0.95 - mae_norm) / 0.22), index=frame.index)
        * pd.Series(_sigmoid((0.026 - barrier) / 0.006), index=frame.index)
        * (0.55 + 0.45 * pd.Series(_sigmoid((12.0 - bars_to_mfe) / 4.0), index=frame.index))
        * (0.50 + 0.50 * pd.Series(_sigmoid((mfe_norm - 1.00) / 0.45), index=frame.index))
        * (1.0 - 0.75 * timeout_float)
    ).clip(0.0, 1.0)
    conditional_clean_u = (
        u
        - 0.0015
        - 0.0050 * (mae_norm - 0.75).clip(lower=0.0)
        - 0.45 * (barrier - 0.022).clip(lower=0.0)
        - 0.0012 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.0040 * timeout_float
    )
    conditional_clean_soft = pd.Series(
        _sigmoid(conditional_clean_u / 0.008),
        index=frame.index,
    ).clip(0.0, 1.0)
    conditional_clean_rank = conditional_clean_soft.groupby(
        frame["__ts__"],
        dropna=False,
    ).rank(method="average", pct=True)
    conditional_clean_rank = conditional_clean_rank.fillna(
        conditional_clean_soft.rank(method="average", pct=True),
    ).clip(0.0, 1.0)
    dirty_path = (
        (mae_norm > 1.0)
        | (barrier > 0.030)
        | (bars_to_mfe > 18.0)
        | (timeout_float > 0.5)
    )
    dirty_cap = (0.08 + 0.22 * survival_clean_score).clip(0.0, 0.30)
    dirty_capped_broad = broad_soft.where(~dirty_path, np.minimum(broad_soft, dirty_cap))
    lowmae_timeout_safe = (
        tail_soft
        * pd.Series(_sigmoid((0.85 - mae_norm) / 0.22), index=frame.index)
        * pd.Series(_sigmoid((14.0 - bars_to_mfe) / 4.5), index=frame.index)
        * pd.Series(_sigmoid((0.030 - barrier) / 0.006), index=frame.index)
        * (1.0 - 0.85 * timeout_float)
        * (0.50 + 0.50 * pd.Series(_sigmoid((mfe_norm - 1.10) / 0.45), index=frame.index))
    ).clip(0.0, 1.0)
    lowmae_timeout_rank = lowmae_timeout_safe.groupby(
        frame["__ts__"],
        dropna=False,
    ).rank(method="average", pct=True)
    lowmae_timeout_rank = lowmae_timeout_rank.fillna(
        lowmae_timeout_safe.rank(method="average", pct=True),
    ).clip(0.0, 1.0)
    strict_lowbarrier_score = pd.Series(
        _sigmoid((0.022 - barrier) / 0.004),
        index=frame.index,
    ).clip(0.0, 1.0)
    strict_lowmae_score = pd.Series(
        _sigmoid((0.90 - mae_norm) / 0.23),
        index=frame.index,
    ).clip(0.0, 1.0)
    strict_timeout_speed_score = (
        pd.Series(_sigmoid((12.0 - bars_to_mfe) / 4.0), index=frame.index)
        * (1.0 - 0.85 * timeout_float)
    ).clip(0.0, 1.0)
    strict_mfe_score = pd.Series(_sigmoid((mfe_norm - 1.10) / 0.45), index=frame.index).clip(0.0, 1.0)
    strict_exec_envelope = (
        strict_lowbarrier_score
        * strict_lowmae_score
        * (0.50 + 0.50 * strict_timeout_speed_score)
        * (0.45 + 0.55 * strict_mfe_score)
    ).clip(0.0, 1.0)
    strict_lowbarrier_u = (
        u
        - 0.0015
        - 0.0045 * (mae_norm - 0.75).clip(lower=0.0)
        - 0.80 * (barrier - 0.020).clip(lower=0.0)
        - 0.0012 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.0040 * timeout_float
    )
    strict_lowbarrier_soft = pd.Series(
        _sigmoid(strict_lowbarrier_u / 0.009),
        index=frame.index,
    ).clip(0.0, 1.0)
    lowbarrier_lowmae_tail = (
        (0.45 * tail_soft + 0.55 * strict_lowbarrier_soft)
        * (0.10 + 0.90 * strict_exec_envelope)
    ).clip(0.0, 1.0)
    lowbarrier_lowmae_rank = lowbarrier_lowmae_tail.groupby(
        frame["__ts__"],
        dropna=False,
    ).rank(method="average", pct=True)
    lowbarrier_lowmae_rank = lowbarrier_lowmae_rank.fillna(
        lowbarrier_lowmae_tail.rank(method="average", pct=True),
    ).clip(0.0, 1.0)
    strict_dirty_path = (
        (mae_norm > 1.0)
        | (barrier > 0.025)
        | (bars_to_mfe > 16.0)
        | (timeout_float > 0.5)
    )
    strict_dirty_cap = (0.04 + 0.18 * strict_exec_envelope).clip(0.0, 0.22)
    lowbarrier_dirty_capped_broad = broad_soft.where(
        ~strict_dirty_path,
        np.minimum(broad_soft, strict_dirty_cap),
    )
    lowbarrier_dirty_capped_rank = lowbarrier_dirty_capped_broad.groupby(
        frame["__ts__"],
        dropna=False,
    ).rank(method="average", pct=True)
    lowbarrier_dirty_capped_rank = lowbarrier_dirty_capped_rank.fillna(
        lowbarrier_dirty_capped_broad.rank(method="average", pct=True),
    ).clip(0.0, 1.0)
    clean_tail_mask = (
        (u > 0.001)
        & (mae_norm <= 0.95)
        & (barrier <= 0.030)
        & (bars_to_mfe <= 16.0)
        & (mfe_norm >= 1.00)
        & (timeout_float <= 0.5)
    ).fillna(False)
    strict_clean_tail_mask = (
        (u > 0.002)
        & (mae_norm <= 0.85)
        & (barrier <= 0.026)
        & (bars_to_mfe <= 12.0)
        & (mfe_norm >= 1.10)
        & (timeout_float <= 0.5)
    ).fillna(False)

    def _masked_rank(values: pd.Series, mask: pd.Series) -> pd.Series:
        out = pd.Series(0.0, index=frame.index)
        mask = mask.reindex(frame.index).fillna(False).astype(bool)
        if not bool(mask.any()):
            return out
        masked_values = pd.to_numeric(values.reindex(frame.index), errors="coerce").loc[mask]
        ranks = masked_values.groupby(frame.loc[mask, "__ts__"], dropna=False).rank(
            method="average",
            pct=True,
        )
        fallback = masked_values.rank(method="average", pct=True)
        out.loc[mask] = ranks.fillna(fallback).fillna(0.0).clip(0.0, 1.0)
        return out.clip(0.0, 1.0)

    clean_lowmae_rank = _masked_rank(lowmae_timeout_safe, clean_tail_mask)
    strict_clean_lowmae_rank = _masked_rank(lowmae_timeout_safe * strict_exec_envelope, strict_clean_tail_mask)
    recoverable_clean_mask = (
        (u > 0.0005)
        & (mae_norm <= 1.00)
        & (barrier <= 0.030)
        & (mfe_norm >= 1.00)
        & (bars_to_mfe <= 18.0)
        & (timeout_float <= 0.5)
    ).fillna(False)
    recoverable_strict_mask = (
        (u > 0.0020)
        & (mae_norm <= 0.85)
        & (barrier <= 0.026)
        & (mfe_norm >= 1.10)
        & (bars_to_mfe <= 12.0)
        & (timeout_float <= 0.5)
    ).fillna(False)
    recoverable_envelope = (
        pd.Series(_sigmoid((0.98 - mae_norm) / 0.28), index=frame.index)
        * pd.Series(_sigmoid((0.030 - barrier) / 0.006), index=frame.index)
        * (0.40 + 0.60 * pd.Series(_sigmoid((16.0 - bars_to_mfe) / 5.0), index=frame.index))
        * (0.35 + 0.65 * pd.Series(_sigmoid((mfe_norm - 1.00) / 0.45), index=frame.index))
        * (1.0 - 0.85 * timeout_float)
    ).clip(0.0, 1.0)
    recoverable_margin_u = (
        u
        - 0.0010
        - 0.0050 * (mae_norm - 0.85).clip(lower=0.0)
        - 0.60 * (barrier - 0.024).clip(lower=0.0)
        - 0.0012 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.0040 * timeout_float
    )
    recoverable_margin_soft = (
        pd.Series(_sigmoid(recoverable_margin_u / 0.008), index=frame.index)
        * recoverable_envelope
    ).clip(0.0, 1.0)
    recoverable_margin_rank = _masked_rank(recoverable_margin_u, recoverable_clean_mask)
    recoverable_tail_rank = _masked_rank(tail_risk_soft * recoverable_envelope, recoverable_clean_mask)
    s30_soft = targets["S30_lowbarrier_tail_risk"]["target_soft"]
    s30_clean_recoverable = (
        s30_soft
        * (0.08 + 0.92 * recoverable_envelope)
    ).clip(0.0, 1.0)
    s30_dirty = (
        (mae_norm > 1.00)
        | (barrier > 0.033)
        | (bars_to_mfe > 20.0)
        | (timeout_float > 0.5)
    ).fillna(True)
    s30_dirty_cap = (0.02 + 0.12 * recoverable_envelope).clip(0.0, 0.14)
    s30_clean_recoverable = s30_clean_recoverable.where(
        ~s30_dirty,
        np.minimum(s30_clean_recoverable, s30_dirty_cap),
    )
    s30_recoverable_rank = _masked_rank(s30_clean_recoverable, recoverable_clean_mask)
    contrast_dirty = (
        (u > 0.0)
        & (
            (mae_norm >= 1.00)
            | (barrier > 0.033)
            | (bars_to_mfe > 20.0)
            | (timeout_float > 0.5)
        )
    ).fillna(False)
    contrast_recoverable = (
        pd.Series(_sigmoid((u - 0.0005) / 0.010), index=frame.index)
        * recoverable_envelope
    ).clip(0.0, 1.0)
    contrast_recoverable = contrast_recoverable.where(
        ~contrast_dirty,
        np.minimum(contrast_recoverable, 0.04 + 0.10 * recoverable_envelope),
    )
    contrast_recoverable_rank = _masked_rank(contrast_recoverable, recoverable_clean_mask)
    s41_like_soft = (0.65 * lowmae_timeout_safe + 0.35 * lowmae_timeout_rank).clip(0.0, 1.0)
    dirty_for_s41 = (
        (mae_norm > 1.0)
        | (barrier > 0.032)
        | (bars_to_mfe > 18.0)
        | (timeout_float > 0.5)
    ).fillna(True)
    s41_dirty_cap = (0.03 + 0.14 * strict_exec_envelope).clip(0.0, 0.17)
    dirty_capped_s41 = s41_like_soft.where(~dirty_for_s41, np.minimum(s41_like_soft, s41_dirty_cap))
    badmae_contrast_u = (
        u
        - 0.0015
        - 0.0045 * (mae_norm - 0.75).clip(lower=0.0)
        - 0.0060 * (mae_norm >= 1.0).astype(float)
        - 0.55 * (barrier - 0.024).clip(lower=0.0)
        - 0.0012 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.0040 * timeout_float
    )
    badmae_contrast_soft = pd.Series(
        _sigmoid(badmae_contrast_u / 0.008),
        index=frame.index,
    ).clip(0.0, 1.0)
    targets["S31_clean_tail_economic"] = pd.DataFrame(
        {
            "target_soft": (tail_soft * (0.20 + 0.80 * economic_clean_score)).clip(0.0, 1.0),
            "target_hard": (
                (u > 0.002)
                & (barrier <= 0.025)
                & (mae_norm <= 1.25)
                & (mfe_norm >= 1.00)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S32_econ_limited_broad_policy"] = pd.DataFrame(
        {
            "target_soft": (broad_soft * (0.15 + 0.85 * economic_clean_score)).clip(0.0, 1.0),
            "target_hard": (
                (u > 0.0)
                & (barrier <= 0.025)
                & (mae_norm <= 1.50)
                & (path_fast >= 0.35)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S33_clean_margin_ts_rank"] = pd.DataFrame(
        {
            "target_soft": (0.65 * clean_margin_soft + 0.35 * clean_margin_rank).clip(0.0, 1.0),
            "target_hard": (
                (clean_margin_u > 0.0)
                & (clean_margin_rank >= 0.80)
                & (barrier <= 0.030)
                & (mae_norm <= 1.50)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S34_exec_guard_broad_policy"] = pd.DataFrame(
        {
            "target_soft": (broad_soft * (0.05 + 0.95 * exec_clean_score)).clip(0.0, 1.0),
            "target_hard": (
                (u > 0.001)
                & (barrier <= 0.022)
                & (mae_norm <= 1.05)
                & (mfe_norm >= 1.00)
                & (bars_to_mfe <= 12.0)
                & (path_fast >= 0.40)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S35_exec_margin_soft"] = pd.DataFrame(
        {
            "target_soft": (exec_margin_soft * (0.20 + 0.80 * exec_clean_score)).clip(0.0, 1.0),
            "target_hard": (
                (exec_margin_u > 0.0)
                & (barrier <= 0.024)
                & (mae_norm <= 1.10)
                & (mfe_norm >= 1.00)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S36_exec_margin_ts_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.55 * exec_margin_soft
                + 0.25 * exec_margin_rank
                + 0.20 * exec_clean_score
            ).clip(0.0, 1.0),
            "target_hard": (
                (exec_margin_u > 0.0)
                & (exec_margin_rank >= 0.80)
                & (barrier <= 0.026)
                & (mae_norm <= 1.15)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S37_lowdrawdown_tail_rank"] = pd.DataFrame(
        {
            "target_soft": (0.70 * lowdrawdown_tail + 0.30 * lowdrawdown_tail_rank).clip(0.0, 1.0),
            "target_hard": (
                (u > 0.002)
                & (lowdrawdown_tail_rank >= 0.80)
                & (barrier <= 0.024)
                & (mae_norm <= 1.00)
                & (bars_to_mfe <= 12.0)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S38_conditional_clean_utility"] = pd.DataFrame(
        {
            "target_soft": (
                conditional_clean_soft
                * (0.10 + 0.90 * survival_clean_score)
            ).clip(0.0, 1.0),
            "target_hard": (
                (conditional_clean_u > 0.0)
                & (mae_norm <= 1.00)
                & (barrier <= 0.028)
                & (mfe_norm >= 1.00)
                & (bars_to_mfe <= 16.0)
                & (timeout_float <= 0.5)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S39_conditional_clean_ts_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.40 * conditional_clean_soft
                + 0.40 * conditional_clean_rank
                + 0.20 * survival_clean_score
            ).clip(0.0, 1.0),
            "target_hard": (
                (conditional_clean_u > 0.0)
                & (conditional_clean_rank >= 0.80)
                & (mae_norm <= 1.10)
                & (barrier <= 0.030)
                & (timeout_float <= 0.5)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S40_dirty_capped_broad_policy"] = pd.DataFrame(
        {
            "target_soft": dirty_capped_broad.clip(0.0, 1.0),
            "target_hard": (
                (u > 0.0)
                & ~dirty_path
                & (mfe_norm >= 1.00)
                & (path_fast >= 0.35)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S41_lowmae_timeout_safe_tail"] = pd.DataFrame(
        {
            "target_soft": s41_like_soft,
            "target_hard": (
                (u > 0.001)
                & (lowmae_timeout_rank >= 0.80)
                & (mae_norm <= 0.95)
                & (barrier <= 0.030)
                & (timeout_float <= 0.5)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S44_clean_masked_lowmae_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.75 * lowmae_timeout_safe.where(clean_tail_mask, 0.0)
                + 0.25 * clean_lowmae_rank
            ).clip(0.0, 1.0),
            "target_hard": (
                clean_tail_mask
                & (clean_lowmae_rank >= 0.75)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S45_strict_clean_tail_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.55 * (lowmae_timeout_safe * strict_exec_envelope).where(strict_clean_tail_mask, 0.0)
                + 0.35 * strict_clean_lowmae_rank
                + 0.10 * strict_lowbarrier_soft.where(strict_clean_tail_mask, 0.0)
            ).clip(0.0, 1.0),
            "target_hard": (
                strict_clean_tail_mask
                & (strict_clean_lowmae_rank >= 0.70)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S46_badmae_contrast_margin"] = pd.DataFrame(
        {
            "target_soft": (
                badmae_contrast_soft
                * (0.20 + 0.50 * clean_lowmae_rank + 0.30 * strict_exec_envelope)
            ).clip(0.0, 1.0),
            "target_hard": (
                (badmae_contrast_u > 0.0)
                & clean_tail_mask
                & (clean_lowmae_rank >= 0.65)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S47_dirty_capped_s41"] = pd.DataFrame(
        {
            "target_soft": dirty_capped_s41.clip(0.0, 1.0),
            "target_hard": (
                clean_tail_mask
                & (dirty_capped_s41 >= 0.40)
                & (u > 0.001)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S48_clean_recoverable_margin_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.60 * recoverable_margin_soft.where(recoverable_clean_mask, 0.0)
                + 0.40 * recoverable_margin_rank
            ).clip(0.0, 1.0),
            "target_hard": (
                recoverable_clean_mask
                & (recoverable_margin_rank >= 0.70)
                & (recoverable_margin_u > 0.0)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S49_clean_recoverable_tail_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.40 * (tail_risk_soft * recoverable_envelope).where(recoverable_clean_mask, 0.0)
                + 0.35 * recoverable_tail_rank
                + 0.25 * recoverable_margin_soft.where(recoverable_clean_mask, 0.0)
            ).clip(0.0, 1.0),
            "target_hard": (
                recoverable_clean_mask
                & (recoverable_tail_rank >= 0.70)
                & (u > 0.001)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S50_s30_clean_recoverable_tail"] = pd.DataFrame(
        {
            "target_soft": (
                0.55 * s30_clean_recoverable.where(recoverable_clean_mask, 0.0)
                + 0.30 * s30_recoverable_rank
                + 0.15 * recoverable_margin_soft.where(recoverable_clean_mask, 0.0)
            ).clip(0.0, 1.0),
            "target_hard": (
                recoverable_clean_mask
                & (s30_recoverable_rank >= 0.65)
                & (u > 0.001)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S51_clean_dirty_contrast_recoverable"] = pd.DataFrame(
        {
            "target_soft": (
                0.50 * contrast_recoverable.where(recoverable_clean_mask, 0.0)
                + 0.30 * contrast_recoverable_rank
                + 0.20 * recoverable_margin_soft.where(recoverable_strict_mask, 0.0)
            ).clip(0.0, 1.0),
            "target_hard": (
                recoverable_clean_mask
                & (contrast_recoverable_rank >= 0.65)
                & (recoverable_margin_u > 0.0)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S52_timeout_barrier_cap_policy_soft"] = pd.DataFrame(
        {
            "target_soft": barrier_capped["policy_soft"],
            "target_hard": (u_timeout_barrier_cap > 0.0).fillna(False).astype(float),
        },
        index=frame.index,
    )
    targets["S53_timeout_barrier_cap_path_blend"] = pd.DataFrame(
        {
            "target_soft": barrier_capped["path_blend"],
            "target_hard": (
                (u_timeout_barrier_cap > 0.0)
                & (path_component_proxy >= 0.45)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S54_timeout_barrier_cap_clean_tail"] = pd.DataFrame(
        {
            "target_soft": barrier_capped["clean_tail"],
            "target_hard": (
                (u_timeout_barrier_cap > 0.002)
                & (barrier <= 0.025)
                & (mae_norm <= 1.25)
                & (mfe_norm >= 1.00)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S55_timeout_barrier_cap_exec_guard"] = pd.DataFrame(
        {
            "target_soft": barrier_capped["exec_guard"],
            "target_hard": (
                (u_timeout_barrier_cap > 0.001)
                & (barrier <= 0.022)
                & (mae_norm <= 1.05)
                & (mfe_norm >= 1.00)
                & (bars_to_mfe <= 12.0)
                & (path_fast >= 0.40)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S56_timeout_tpnet_cap_policy_soft"] = pd.DataFrame(
        {
            "target_soft": tpnet_capped["policy_soft"],
            "target_hard": (u_timeout_tpnet_cap > 0.0).fillna(False).astype(float),
        },
        index=frame.index,
    )
    targets["S57_timeout_tpnet_cap_path_blend"] = pd.DataFrame(
        {
            "target_soft": tpnet_capped["path_blend"],
            "target_hard": (
                (u_timeout_tpnet_cap > 0.0)
                & (path_component_proxy >= 0.45)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S58_timeout_tpnet_cap_clean_tail"] = pd.DataFrame(
        {
            "target_soft": tpnet_capped["clean_tail"],
            "target_hard": (
                (u_timeout_tpnet_cap > 0.0)
                & (barrier <= 0.025)
                & (mae_norm <= 1.25)
                & (mfe_norm >= 1.00)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S59_timeout_tpnet_cap_exec_guard"] = pd.DataFrame(
        {
            "target_soft": tpnet_capped["exec_guard"],
            "target_hard": (
                (u_timeout_tpnet_cap > 0.0)
                & (barrier <= 0.022)
                & (mae_norm <= 1.05)
                & (mfe_norm >= 1.00)
                & (bars_to_mfe <= 12.0)
                & (path_fast >= 0.40)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    severe_veto_envelope = (
        pd.Series(_sigmoid((0.85 - mae_norm) / 0.18), index=frame.index)
        * pd.Series(_sigmoid((0.024 - barrier) / 0.004), index=frame.index)
        * (0.35 + 0.65 * pd.Series(_sigmoid((mfe_mae - 1.35) / 0.35), index=frame.index))
        * (0.45 + 0.55 * pd.Series(_sigmoid((14.0 - bars_to_mfe) / 4.0), index=frame.index))
        * (1.0 - 0.95 * timeout_float)
    ).clip(0.0, 1.0)
    severe_dirty_path = (
        (mae_norm >= 1.00)
        | (barrier > 0.026)
        | (timeout_float > 0.5)
        | ((mae_norm > 0.75) & (mfe_mae < 1.25))
    ).fillna(True)
    severe_dirty_cap = (0.01 + 0.08 * severe_veto_envelope).clip(0.0, 0.09)
    severe_veto_path = (
        tpnet_capped["path_blend"]
        * (0.02 + 0.98 * severe_veto_envelope)
    ).clip(0.0, 1.0)
    severe_veto_path = severe_veto_path.where(
        ~severe_dirty_path,
        np.minimum(severe_veto_path, severe_dirty_cap),
    )
    severe_margin_u = (
        u_timeout_tpnet_cap
        - 0.0015
        - 0.0080 * (mae_norm - 0.65).clip(lower=0.0)
        - 0.85 * (barrier - 0.018).clip(lower=0.0)
        - 0.0014 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.0050 * timeout_float
        - 0.0060 * (mae_norm >= 1.00).astype(float)
    )
    severe_margin_soft = (
        pd.Series(_sigmoid(severe_margin_u / 0.007), index=frame.index)
        * severe_veto_envelope
    ).clip(0.0, 1.0)
    severe_clean_mask = (
        (u_timeout_tpnet_cap > 0.0005)
        & (mae_norm <= 0.85)
        & (barrier <= 0.024)
        & (mfe_mae >= 1.35)
        & (bars_to_mfe <= 14.0)
        & (timeout_float <= 0.5)
    ).fillna(False)
    severe_margin_rank = _masked_rank(severe_margin_u, severe_clean_mask)
    severe_contrast = severe_margin_soft.where(
        severe_clean_mask,
        np.minimum(severe_margin_soft, 0.02 + 0.05 * severe_veto_envelope),
    )
    severe_contrast_rank = _masked_rank(severe_contrast, severe_clean_mask)
    targets["S60_tpnet_severe_adverse_veto_path"] = pd.DataFrame(
        {
            "target_soft": severe_veto_path.clip(0.0, 1.0),
            "target_hard": (
                (u_timeout_tpnet_cap > 0.0)
                & ~severe_dirty_path
                & (mfe_mae >= 1.25)
                & (path_component_proxy >= 0.45)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S61_tpnet_strict_adverse_veto_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.55 * severe_margin_soft.where(severe_clean_mask, 0.0)
                + 0.35 * severe_margin_rank
                + 0.10 * severe_veto_envelope.where(severe_clean_mask, 0.0)
            ).clip(0.0, 1.0),
            "target_hard": (
                severe_clean_mask
                & (severe_margin_rank >= 0.65)
                & (severe_margin_u > 0.0)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S62_tpnet_clean_dirty_contrast_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.65 * severe_contrast.where(severe_clean_mask, 0.0)
                + 0.35 * severe_contrast_rank
            ).clip(0.0, 1.0),
            "target_hard": (
                severe_clean_mask
                & (severe_contrast_rank >= 0.60)
                & (u_timeout_tpnet_cap > 0.001)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    exec_admissible_score = (
        pd.Series(_sigmoid((0.78 - mae_norm) / 0.16), index=frame.index)
        * pd.Series(_sigmoid((0.024 - barrier) / 0.004), index=frame.index)
        * pd.Series(_sigmoid((12.0 - bars_to_mfe) / 3.0), index=frame.index)
        * pd.Series(_sigmoid((mfe_mae - 1.35) / 0.30), index=frame.index)
        * pd.Series(_sigmoid((mfe_norm - 1.05) / 0.30), index=frame.index)
        * (1.0 - timeout_float)
    ).clip(0.0, 1.0)
    exec_profit_soft = pd.Series(
        _sigmoid((u_timeout_tpnet_cap - 0.0020) / 0.006),
        index=frame.index,
    ).clip(0.0, 1.0)
    exec_margin_for_rank = (
        u_timeout_tpnet_cap
        - 0.0020
        - 0.0060 * (mae_norm - 0.65).clip(lower=0.0)
        - 0.75 * (barrier - 0.020).clip(lower=0.0)
        - 0.0015 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.0060 * timeout_float
        - 0.0050 * (mae_norm >= 1.0).astype(float)
    )
    exec_admissible_mask = (
        (timeout_float <= 0.5)
        & (u_timeout_tpnet_cap > 0.0005)
        & (mae_norm <= 0.85)
        & (barrier <= 0.026)
        & (mfe_norm >= 1.00)
        & (mfe_mae >= 1.25)
        & (bars_to_mfe <= 14.0)
    ).fillna(False)
    exec_strict_mask = (
        exec_admissible_mask
        & (u_timeout_tpnet_cap > 0.0020)
        & (mae_norm <= 0.75)
        & (barrier <= 0.024)
        & (mfe_mae >= 1.40)
        & (bars_to_mfe <= 10.0)
    ).fillna(False)
    exec_admissible_rank = _masked_rank(exec_admissible_score, exec_admissible_mask)
    exec_margin_rank = _masked_rank(exec_margin_for_rank, exec_admissible_mask)
    exec_strict_rank = _masked_rank(exec_margin_for_rank * exec_admissible_score, exec_strict_mask)
    exec_dirty_positive = (
        (u_timeout_tpnet_cap > 0.0)
        & (
            (timeout_float > 0.5)
            | (mae_norm > 0.95)
            | (barrier > 0.030)
            | (mfe_mae < 1.20)
            | (bars_to_mfe > 16.0)
        )
    ).fillna(False)
    exec_primary = (
        exec_admissible_score
        * (0.20 + 0.80 * exec_profit_soft)
    ).where(exec_admissible_mask, 0.0).clip(0.0, 1.0)
    exec_contrast = exec_primary.where(
        ~exec_dirty_positive,
        np.minimum(exec_primary, 0.015 + 0.035 * exec_admissible_score),
    ).clip(0.0, 1.0)
    exec_contrast_rank = _masked_rank(exec_contrast, exec_admissible_mask)
    targets["S63_exec_admissible_primary"] = pd.DataFrame(
        {
            "target_soft": exec_primary,
            "target_hard": (
                exec_admissible_mask
                & (exec_admissible_score >= 0.35)
                & (u_timeout_tpnet_cap > 0.0010)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S64_exec_admissible_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.60 * exec_primary
                + 0.40 * exec_admissible_rank
            ).clip(0.0, 1.0),
            "target_hard": (
                exec_admissible_mask
                & (exec_admissible_rank >= 0.65)
                & (u_timeout_tpnet_cap > 0.0010)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S65_profit_inside_exec_admissible"] = pd.DataFrame(
        {
            "target_soft": (
                0.55 * exec_margin_rank
                + 0.35 * exec_profit_soft.where(exec_admissible_mask, 0.0)
                + 0.10 * exec_strict_rank
            ).clip(0.0, 1.0),
            "target_hard": (
                exec_admissible_mask
                & (exec_margin_rank >= 0.65)
                & (u_timeout_tpnet_cap > 0.0020)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S66_exec_admissible_contrast_rank"] = pd.DataFrame(
        {
            "target_soft": (
                0.60 * exec_contrast.where(exec_admissible_mask, 0.0)
                + 0.40 * exec_contrast_rank
            ).clip(0.0, 1.0),
            "target_hard": (
                exec_admissible_mask
                & (exec_contrast_rank >= 0.60)
                & (u_timeout_tpnet_cap > 0.0015)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S42_lowbarrier_lowmae_tail"] = pd.DataFrame(
        {
            "target_soft": (
                0.75 * lowbarrier_lowmae_tail
                + 0.25 * lowbarrier_lowmae_rank
            ).clip(0.0, 1.0),
            "target_hard": (
                (strict_lowbarrier_u > 0.0)
                & (lowbarrier_lowmae_rank >= 0.80)
                & (mae_norm <= 0.95)
                & (barrier <= 0.025)
                & (timeout_float <= 0.5)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    targets["S43_lowbarrier_dirty_capped_broad"] = pd.DataFrame(
        {
            "target_soft": (
                0.80 * lowbarrier_dirty_capped_broad
                + 0.20 * lowbarrier_dirty_capped_rank
            ).clip(0.0, 1.0),
            "target_hard": (
                (u > 0.0)
                & ~strict_dirty_path
                & (path_fast >= 0.35)
                & (mfe_norm >= 1.00)
            )
            .fillna(False)
            .astype(float),
        },
        index=frame.index,
    )
    return {name: targets[name] for name in LABEL_ARMS}


def _economic_targets(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    u = metrics["u_policy_net"].fillna(-0.02)
    risk_mild = (
        u
        - 0.0025 * (metrics["mae_norm"] - 1.0).clip(lower=0.0)
        - 0.00075 * np.log1p(metrics["bars_to_mfe"].clip(lower=0.0))
        - 0.20 * (metrics["barrier"] - 0.018).clip(lower=0.0)
    )
    risk_strict_fast = (
        u
        - 0.0040 * (metrics["mae_norm"] - 0.75).clip(lower=0.0)
        - 0.00150 * np.log1p(metrics["bars_to_mfe"].clip(lower=0.0))
        - 0.35 * (metrics["barrier"] - 0.018).clip(lower=0.0)
        - 0.0030 * metrics["is_timeout"].astype(float)
    )
    return {
        "raw_u_policy_net": u,
        "risk_u_mild": risk_mild,
        "risk_u_strict_fast": risk_strict_fast,
    }


def _target_frame_from_series(series: pd.Series) -> pd.DataFrame:
    soft = pd.Series(_sigmoid(series.fillna(-0.02) / 0.008), index=series.index).clip(0.0, 1.0)
    hard = (series > 0.0).fillna(False).astype(float)
    return pd.DataFrame({"target_soft": soft, "target_hard": hard}, index=series.index)


def _top_gate(score: pd.Series, frac: float) -> pd.Series:
    score = pd.to_numeric(score, errors="coerce")
    ranks = score.rank(method="first", pct=True)
    return ranks >= (1.0 - float(frac))


def _baseline_row(valid_metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "period_baseline_mean_u": _safe_mean(valid_metrics["u_policy_net"]),
        "period_baseline_hit_u": _safe_mean(valid_metrics["u_policy_net"] > 0.0),
        "period_baseline_q10_u": _safe_quantile(valid_metrics["u_policy_net"], 0.10),
    }


def _add_delta_fields(row: dict[str, Any], baseline: dict[str, float]) -> None:
    mean_u = float(row["mean_u"])
    hit_u = float(row["hit_u"])
    q10_u = float(row["q10_u"])
    row.update(baseline)
    row["delta_mean_u_vs_period"] = (
        mean_u - baseline["period_baseline_mean_u"]
        if math.isfinite(mean_u) and math.isfinite(baseline["period_baseline_mean_u"])
        else float("nan")
    )
    row["delta_hit_u_vs_period"] = (
        hit_u - baseline["period_baseline_hit_u"]
        if math.isfinite(hit_u) and math.isfinite(baseline["period_baseline_hit_u"])
        else float("nan")
    )
    row["delta_q10_u_vs_period"] = (
        q10_u - baseline["period_baseline_q10_u"]
        if math.isfinite(q10_u) and math.isfinite(baseline["period_baseline_q10_u"])
        else float("nan")
    )


def _score_candidate(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
    label_arm: str,
    economic_arm: str,
    baseline: dict[str, float],
    label_proxy_features: str,
    economic_proxy_features: str,
    gate_frac: float | None,
    label_score: pd.Series | None,
    economic_score: pd.Series | None,
    economic_target: pd.Series | None,
) -> dict[str, Any]:
    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=arm,
        selector=selector,
        period=period,
        top_frac=top_frac,
    )
    _add_delta_fields(row, baseline)
    row.update(
        {
            "label_arm": label_arm,
            "economic_arm": economic_arm,
            "gate_frac": float(gate_frac) if gate_frac is not None else 1.0,
            "gate_rows": int(pd.to_numeric(score, errors="coerce").notna().sum()),
            "gate_valid_frac": float(pd.to_numeric(score, errors="coerce").notna().mean()),
            "label_proxy_features": label_proxy_features,
            "economic_proxy_features": economic_proxy_features,
            "score_ic_u": _spearman(score, metrics["u_policy_net"]),
            "score_ic_label": _spearman(score, label_score) if label_score is not None else float("nan"),
            "score_ic_economic": (
                _spearman(score, economic_target) if economic_target is not None else float("nan")
            ),
            "label_econ_score_ic": (
                _spearman(label_score, economic_score)
                if label_score is not None and economic_score is not None
                else float("nan")
            ),
        }
    )
    return row


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    economic_targets: dict[str, pd.Series],
    features: list[str],
    month: str,
) -> list[dict[str, Any]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    valid_mask = month_period == month
    if int(train_mask.sum()) < 100 or int(valid_mask.sum()) < 50:
        return []

    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    baseline = _baseline_row(valid_metrics)

    label_scores: dict[str, pd.Series] = {}
    label_feature_names: dict[str, str] = {}
    for arm in LABEL_ARMS:
        score, diag = _proxy_score(
            train,
            frame.loc[valid_mask].copy(),
            features,
            targets[arm].loc[train_mask, "target_soft"],
        )
        label_scores[arm] = score.reset_index(drop=True)
        label_feature_names[arm] = ",".join(diag.get("proxy_features", []))

    economic_scores: dict[str, pd.Series] = {}
    economic_feature_names: dict[str, str] = {}
    for arm in ECONOMIC_ARMS:
        score, diag = _proxy_score(
            train,
            frame.loc[valid_mask].copy(),
            features,
            economic_targets[arm].loc[train_mask],
        )
        economic_scores[arm] = score.reset_index(drop=True)
        economic_feature_names[arm] = ",".join(diag.get("proxy_features", []))

    rows: list[dict[str, Any]] = []
    for top_frac in TOP_FRACS:
        for label_arm, label_score in label_scores.items():
            target = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
            rows.append(
                _score_candidate(
                    frame=valid,
                    metrics=valid_metrics,
                    target=target,
                    score=label_score,
                    arm=f"label_only::{label_arm}",
                    selector="label_ic_proxy_oos",
                    period=month,
                    top_frac=top_frac,
                    label_arm=label_arm,
                    economic_arm="none",
                    baseline=baseline,
                    label_proxy_features=label_feature_names[label_arm],
                    economic_proxy_features="",
                    gate_frac=None,
                    label_score=target["target_soft"],
                    economic_score=None,
                    economic_target=None,
                )
            )

        for economic_arm, economic_score in economic_scores.items():
            economic_valid = economic_targets[economic_arm].loc[valid_mask].copy().reset_index(drop=True)
            target = _target_frame_from_series(economic_valid)
            rows.append(
                _score_candidate(
                    frame=valid,
                    metrics=valid_metrics,
                    target=target,
                    score=economic_score,
                    arm=f"economic_only::{economic_arm}",
                    selector="economic_ic_proxy_oos",
                    period=month,
                    top_frac=top_frac,
                    label_arm="none",
                    economic_arm=economic_arm,
                    baseline=baseline,
                    label_proxy_features="",
                    economic_proxy_features=economic_feature_names[economic_arm],
                    gate_frac=None,
                    label_score=None,
                    economic_score=economic_score,
                    economic_target=economic_valid,
                )
            )

        for label_arm, label_score in label_scores.items():
            target = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
            for economic_arm, economic_score in economic_scores.items():
                economic_valid = economic_targets[economic_arm].loc[valid_mask].copy().reset_index(drop=True)
                for label_weight in COMBINE_WEIGHTS:
                    combined = (label_weight * label_score) + ((1.0 - label_weight) * economic_score)
                    rows.append(
                        _score_candidate(
                            frame=valid,
                            metrics=valid_metrics,
                            target=target,
                            score=combined,
                            arm=f"combined_l{label_weight:.2f}::{label_arm}::{economic_arm}",
                            selector="combined_label_economic_proxy_oos",
                            period=month,
                            top_frac=top_frac,
                            label_arm=label_arm,
                            economic_arm=economic_arm,
                            baseline=baseline,
                            label_proxy_features=label_feature_names[label_arm],
                            economic_proxy_features=economic_feature_names[economic_arm],
                            gate_frac=None,
                            label_score=target["target_soft"],
                            economic_score=economic_score,
                            economic_target=economic_valid,
                        )
                    )
                for gate_frac in ECONOMIC_GATE_FRACS:
                    gated = label_score.where(_top_gate(economic_score, gate_frac))
                    rows.append(
                        _score_candidate(
                            frame=valid,
                            metrics=valid_metrics,
                            target=target,
                            score=gated,
                            arm=f"econ_gate{gate_frac:.2f}_label::{label_arm}::{economic_arm}",
                            selector="economic_gate_then_label_proxy_oos",
                            period=month,
                            top_frac=top_frac,
                            label_arm=label_arm,
                            economic_arm=economic_arm,
                            baseline=baseline,
                            label_proxy_features=label_feature_names[label_arm],
                            economic_proxy_features=economic_feature_names[economic_arm],
                            gate_frac=gate_frac,
                            label_score=target["target_soft"],
                            economic_score=economic_score,
                            economic_target=economic_valid,
                        )
                    )
    return rows


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    groups = monthly.groupby(
        ["arm", "selector", "label_arm", "economic_arm", "top_frac", "gate_frac"],
        dropna=False,
        observed=True,
    )
    for key, group in groups:
        arm, selector, label_arm, economic_arm, top_frac, gate_frac = key
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        rows.append(
            {
                "arm": arm,
                "selector": selector,
                "label_arm": label_arm,
                "economic_arm": economic_arm,
                "top_frac": float(top_frac),
                "gate_frac": float(gate_frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": _safe_quantile(mean_u, 0.0),
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "delta_mean_u_vs_period": _safe_mean(group["delta_mean_u_vs_period"]),
                "delta_hit_u_vs_period": _safe_mean(group["delta_hit_u_vs_period"]),
                "delta_q10_u_vs_period": _safe_mean(group["delta_q10_u_vs_period"]),
                "score_ic_u": _safe_mean(group["score_ic_u"]),
                "score_ic_label": _safe_mean(group["score_ic_label"]),
                "score_ic_economic": _safe_mean(group["score_ic_economic"]),
                "label_econ_score_ic": _safe_mean(group["label_econ_score_ic"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "mean_selected_rows": _safe_mean(selected_rows),
                "min_selected_rows": int(selected_rows.min()) if len(selected_rows.dropna()) else 0,
                "mean_gate_valid_frac": _safe_mean(group["gate_valid_frac"]),
                "label_proxy_features": str(group["label_proxy_features"].dropna().iloc[0])
                if group["label_proxy_features"].dropna().size
                else "",
                "economic_proxy_features": str(group["economic_proxy_features"].dropna().iloc[0])
                if group["economic_proxy_features"].dropna().size
                else "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_frac", "mean_u", "worst_month_mean_u"],
        ascending=[True, False, False],
    )


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_economic_proxy_ablation.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "arm",
        "selector",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "hit_u",
        "q10_u",
        "delta_mean_u_vs_period",
        "score_ic_u",
        "bad_mae_1r_rate",
        "mean_selected_rows",
        "min_selected_rows",
        "top_symbol_share",
    ]
    lines = [
        "# Label/Economic Proxy Ablation",
        "",
        "Scope: no model training, no Optuna, no policy geometry. Scores are built from prior-month feature IC only.",
        "",
        "Selectors:",
        "",
        "- `label_ic_proxy_oos`: feature proxy learned from the label target.",
        "- `economic_ic_proxy_oos`: feature proxy learned directly from economic utility or risk-adjusted utility.",
        "- `combined_label_economic_proxy_oos`: weighted blend of label proxy and economic proxy.",
        "- `economic_gate_then_label_proxy_oos`: label proxy only inside the top economic-proxy gate.",
        "",
    ]
    for frac in TOP_FRACS:
        subset = aggregate[aggregate["top_frac"].eq(frac)].sort_values(
            ["mean_u", "worst_month_mean_u"],
            ascending=[False, False],
        )
        lines.extend(
            [
                f"## Top {frac:.0%}",
                "",
                table(subset, cols, limit=25),
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Monthly: `{manifest['outputs']['monthly']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)

    metrics = _path_metrics(frame)
    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    economic_targets = _economic_targets(metrics)
    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())

    monthly_rows: list[dict[str, Any]] = []
    for month in months[1:]:
        monthly_rows.extend(
            _run_month(
                frame=frame,
                metrics=metrics,
                targets=targets,
                economic_targets=economic_targets,
                features=features,
                month=month,
            )
        )

    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly)

    paths = {
        "monthly": output_dir / "label_economic_proxy_monthly.csv",
        "aggregate": output_dir / "label_economic_proxy_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)

    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "features": features,
        "label_arms": list(LABEL_ARMS),
        "economic_arms": list(ECONOMIC_ARMS),
        "combine_weights": list(COMBINE_WEIGHTS),
        "economic_gate_fracs": list(ECONOMIC_GATE_FRACS),
        "proxy_top_k_features": int(PROXY_TOP_K_FEATURES),
        "top_fracs": list(TOP_FRACS),
        "feature_store": feature_store_report,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
