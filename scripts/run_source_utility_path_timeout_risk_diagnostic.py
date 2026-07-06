#!/usr/bin/env python3
"""Utility plus separate path-time risk-head economics diagnostic.

This is the economics screen after the source-archetype and timeout/holding
diagnostics. It trains a utility head and separate bad-MAE, wide-barrier, and
timeout/holding risk heads on prior months only, then selects future-month rows
with small predeclared soft-penalty combinations. Optional two-stage selection
specs first filter on predicted path risk, then rank the survivors by utility
or soft risk penalties.

The script is diagnostic-only. It does not modify production training,
policies, Optuna search spaces, or model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_feature_store_model_smoke import _fit_predict, _month_model_frame  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_feature_store_columns,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (  # noqa: E402
    DEFAULT_MONTHS,
    DEFAULT_QUALITY_LABELS,
    DEFAULT_SEEDS,
    _load_joined_frame,
    _parse_csv,
    _parse_float_csv,
    _parse_int_csv,
    _source_feature_columns,
)
from scripts.run_source_utility_label_rework_diagnostic import (  # noqa: E402
    DEFAULT_TOP_FRACS,
    _build_target,
    _safe_numeric,
)
from scripts.run_source_utility_risk_gate_diagnostic import (  # noqa: E402
    _assert_gate_columns_causal,
    _gate_mask,
    _gate_specs_by_name,
    _label_specs_by_name,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/utility_path_timeout_joint_risk"
)
DEFAULT_ARCHETYPES_V2 = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/source_archetypes_v2/"
    "candidate_source_archetypes_v2.parquet"
)
DEFAULT_LABELS = ("utility_linear_source_q80_v1",)
DEFAULT_RISK_HEADS = ("bad_mae_risk_v1", "wide_barrier_risk_v1", "timeout_risk_v1")
DEFAULT_FEATURE_SETS = ("base_plus_source", "base_plus_source_v2")
DEFAULT_SOURCE_BUCKETS = ("all_rows", "risk_adjusted_capture_candidate")
DEFAULT_CAUSAL_GATES = ("no_gate", "low_barrier_pressure_q50")
DEFAULT_SELECTIONS = (
    "utility_only",
    "utility_minus_bad_mae_0p50",
    "utility_minus_wide_barrier_0p50",
    "utility_minus_timeout_0p50",
    "utility_minus_bad_wide_0p50",
    "utility_minus_all_0p25",
    "utility_minus_all_0p50",
    "utility_minus_all_1p00",
)


@dataclass(frozen=True)
class RiskHeadSpec:
    name: str
    kind: str
    description: str


RISK_HEAD_SPECS = (
    RiskHeadSpec("bad_mae_risk_v1", "bad_mae", "MAE >= 1R plus scaled MAE severity."),
    RiskHeadSpec("wide_barrier_risk_v1", "wide_barrier", "Barrier width > 25 bps plus scaled width."),
    RiskHeadSpec("timeout_risk_v1", "timeout", "Policy timeout flag."),
    RiskHeadSpec("holding_risk_v1", "holding", "Holding duration above prior-train q80."),
    RiskHeadSpec(
        "bad_mae_wide25_risk_v1",
        "bad_mae_wide25",
        "Joint MAE >= 1R and barrier width > 25 bps risk.",
    ),
    RiskHeadSpec(
        "adverse_path_loss_risk_v1",
        "adverse_path_loss",
        "Loss-conditioned adverse path risk; avoids treating wide profitable captures as uniformly bad.",
    ),
    RiskHeadSpec(
        "bad_mae_loss_risk_v1",
        "bad_mae_loss",
        "MAE >= 1R with negative realized utility; separates recovered MAE from economic path loss.",
    ),
    RiskHeadSpec(
        "fast_bad_mae_loss_risk_v1",
        "fast_bad_mae_loss",
        "Fast MAE >= 1R with negative realized utility; targets immediate adverse-path failures.",
    ),
    RiskHeadSpec(
        "bad_mae_recovery_failure_risk_v1",
        "bad_mae_recovery_failure",
        "Contrastive MAE >= 1R head: positive class is negative bad-MAE; recovered bad-MAE rows are emphasized negatives.",
    ),
)


@dataclass(frozen=True)
class SelectionSpec:
    name: str
    bad_mae_lambda: float = 0.0
    wide_barrier_lambda: float = 0.0
    timeout_lambda: float = 0.0
    holding_lambda: float = 0.0
    bad_mae_wide25_lambda: float = 0.0
    adverse_path_loss_lambda: float = 0.0
    bad_mae_loss_lambda: float = 0.0
    fast_bad_mae_loss_lambda: float = 0.0
    bad_mae_recovery_failure_lambda: float = 0.0
    bad_mae_keep_frac: float | None = None
    wide_barrier_keep_frac: float | None = None
    timeout_keep_frac: float | None = None
    bad_mae_wide25_keep_frac: float | None = None
    adverse_path_loss_keep_frac: float | None = None
    bad_mae_loss_keep_frac: float | None = None
    fast_bad_mae_loss_keep_frac: float | None = None
    bad_mae_recovery_failure_keep_frac: float | None = None
    preserve_scope_budget: bool = False

    @property
    def required_heads(self) -> tuple[str, ...]:
        heads: list[str] = []
        if self.bad_mae_lambda or self.bad_mae_keep_frac is not None:
            heads.append("bad_mae_risk_v1")
        if self.wide_barrier_lambda or self.wide_barrier_keep_frac is not None:
            heads.append("wide_barrier_risk_v1")
        if self.timeout_lambda or self.timeout_keep_frac is not None:
            heads.append("timeout_risk_v1")
        if self.holding_lambda:
            heads.append("holding_risk_v1")
        if self.bad_mae_wide25_lambda or self.bad_mae_wide25_keep_frac is not None:
            heads.append("bad_mae_wide25_risk_v1")
        if self.adverse_path_loss_lambda or self.adverse_path_loss_keep_frac is not None:
            heads.append("adverse_path_loss_risk_v1")
        if self.bad_mae_loss_lambda or self.bad_mae_loss_keep_frac is not None:
            heads.append("bad_mae_loss_risk_v1")
        if self.fast_bad_mae_loss_lambda or self.fast_bad_mae_loss_keep_frac is not None:
            heads.append("fast_bad_mae_loss_risk_v1")
        if self.bad_mae_recovery_failure_lambda or self.bad_mae_recovery_failure_keep_frac is not None:
            heads.append("bad_mae_recovery_failure_risk_v1")
        return tuple(dict.fromkeys(heads))


SELECTION_SPECS = (
    SelectionSpec("utility_only"),
    SelectionSpec("utility_minus_bad_mae_0p25", bad_mae_lambda=0.25),
    SelectionSpec("utility_minus_bad_mae_0p50", bad_mae_lambda=0.50),
    SelectionSpec("utility_minus_bad_mae_1p00", bad_mae_lambda=1.00),
    SelectionSpec("utility_minus_wide_barrier_0p25", wide_barrier_lambda=0.25),
    SelectionSpec("utility_minus_wide_barrier_0p50", wide_barrier_lambda=0.50),
    SelectionSpec("utility_minus_wide_barrier_1p00", wide_barrier_lambda=1.00),
    SelectionSpec("utility_minus_timeout_0p25", timeout_lambda=0.25),
    SelectionSpec("utility_minus_timeout_0p50", timeout_lambda=0.50),
    SelectionSpec("utility_minus_timeout_1p00", timeout_lambda=1.00),
    SelectionSpec("utility_minus_bad_wide_0p25", bad_mae_lambda=0.25, wide_barrier_lambda=0.25),
    SelectionSpec("utility_minus_bad_wide_0p50", bad_mae_lambda=0.50, wide_barrier_lambda=0.50),
    SelectionSpec("utility_minus_bad_wide_1p00", bad_mae_lambda=1.00, wide_barrier_lambda=1.00),
    SelectionSpec(
        "utility_minus_all_0p25",
        bad_mae_lambda=0.25,
        wide_barrier_lambda=0.25,
        timeout_lambda=0.25,
    ),
    SelectionSpec(
        "utility_minus_all_0p50",
        bad_mae_lambda=0.50,
        wide_barrier_lambda=0.50,
        timeout_lambda=0.50,
    ),
    SelectionSpec(
        "utility_minus_all_1p00",
        bad_mae_lambda=1.00,
        wide_barrier_lambda=1.00,
        timeout_lambda=1.00,
    ),
    SelectionSpec(
        "utility_minus_bad_wide_timeout_holding_0p50",
        bad_mae_lambda=0.50,
        wide_barrier_lambda=0.50,
        timeout_lambda=0.50,
        holding_lambda=0.50,
    ),
    SelectionSpec("stage1_badmae_q30_then_utility", bad_mae_keep_frac=0.30, preserve_scope_budget=True),
    SelectionSpec("stage1_badmae_q40_then_utility", bad_mae_keep_frac=0.40, preserve_scope_budget=True),
    SelectionSpec("stage1_badmae_q50_then_utility", bad_mae_keep_frac=0.50, preserve_scope_budget=True),
    SelectionSpec("stage1_badmae_q60_then_utility", bad_mae_keep_frac=0.60, preserve_scope_budget=True),
    SelectionSpec("stage1_badmae_q70_then_utility", bad_mae_keep_frac=0.70, preserve_scope_budget=True),
    SelectionSpec(
        "stage1_badmae_q30_then_timeout_0p50",
        timeout_lambda=0.50,
        bad_mae_keep_frac=0.30,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_badmae_q40_then_timeout_0p50",
        timeout_lambda=0.50,
        bad_mae_keep_frac=0.40,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_badmae_q50_then_timeout_0p50",
        timeout_lambda=0.50,
        bad_mae_keep_frac=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_badmae_q60_then_timeout_0p50",
        timeout_lambda=0.50,
        bad_mae_keep_frac=0.60,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_badmae_q70_then_timeout_0p50",
        timeout_lambda=0.50,
        bad_mae_keep_frac=0.70,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_badmae_q50_then_timeout_wide_0p25",
        wide_barrier_lambda=0.25,
        timeout_lambda=0.25,
        bad_mae_keep_frac=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_badmae_q30_wide_q80_then_timeout_0p50",
        timeout_lambda=0.50,
        bad_mae_keep_frac=0.30,
        wide_barrier_keep_frac=0.80,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_badmae_q40_wide_q80_then_timeout_0p50",
        timeout_lambda=0.50,
        bad_mae_keep_frac=0.40,
        wide_barrier_keep_frac=0.80,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_badmae_q50_wide_q80_then_timeout_0p50",
        timeout_lambda=0.50,
        bad_mae_keep_frac=0.50,
        wide_barrier_keep_frac=0.80,
        preserve_scope_budget=True,
    ),
    SelectionSpec("utility_minus_adverse_path_0p25", adverse_path_loss_lambda=0.25),
    SelectionSpec("utility_minus_adverse_path_0p50", adverse_path_loss_lambda=0.50),
    SelectionSpec(
        "utility_minus_adverse_path_timeout_0p50",
        adverse_path_loss_lambda=0.50,
        timeout_lambda=0.50,
    ),
    SelectionSpec("utility_minus_bad_mae_loss_0p25", bad_mae_loss_lambda=0.25),
    SelectionSpec("utility_minus_bad_mae_loss_0p50", bad_mae_loss_lambda=0.50),
    SelectionSpec(
        "utility_minus_bad_mae_loss_timeout_0p50",
        bad_mae_loss_lambda=0.50,
        timeout_lambda=0.50,
    ),
    SelectionSpec(
        "utility_minus_fast_bad_mae_loss_timeout_0p50",
        fast_bad_mae_loss_lambda=0.50,
        timeout_lambda=0.50,
    ),
    SelectionSpec("utility_minus_recovery_failure_0p50", bad_mae_recovery_failure_lambda=0.50),
    SelectionSpec(
        "utility_minus_recovery_failure_timeout_0p50",
        bad_mae_recovery_failure_lambda=0.50,
        timeout_lambda=0.50,
    ),
    SelectionSpec(
        "stage1_adverse_path_q30_then_utility",
        adverse_path_loss_keep_frac=0.30,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_adverse_path_q40_then_utility",
        adverse_path_loss_keep_frac=0.40,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_adverse_path_q50_then_utility",
        adverse_path_loss_keep_frac=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_adverse_path_q30_then_timeout_0p50",
        adverse_path_loss_keep_frac=0.30,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_adverse_path_q40_then_timeout_0p50",
        adverse_path_loss_keep_frac=0.40,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_adverse_path_q50_then_timeout_0p50",
        adverse_path_loss_keep_frac=0.50,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_adverse_path_q30_wide_q80_then_timeout_0p50",
        adverse_path_loss_keep_frac=0.30,
        wide_barrier_keep_frac=0.80,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_bad_mae_wide_q50_then_utility",
        bad_mae_wide25_keep_frac=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_bad_mae_wide_q50_then_timeout_0p50",
        bad_mae_wide25_keep_frac=0.50,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_bad_mae_loss_q30_then_timeout_0p50",
        bad_mae_loss_keep_frac=0.30,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_bad_mae_loss_q40_then_timeout_0p50",
        bad_mae_loss_keep_frac=0.40,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_bad_mae_loss_q50_then_timeout_0p50",
        bad_mae_loss_keep_frac=0.50,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_fast_bad_mae_loss_q30_then_timeout_0p50",
        fast_bad_mae_loss_keep_frac=0.30,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_fast_bad_mae_loss_q40_then_timeout_0p50",
        fast_bad_mae_loss_keep_frac=0.40,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_fast_bad_mae_loss_q50_then_timeout_0p50",
        fast_bad_mae_loss_keep_frac=0.50,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_recovery_failure_q30_then_timeout_0p50",
        bad_mae_recovery_failure_keep_frac=0.30,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_recovery_failure_q40_then_timeout_0p50",
        bad_mae_recovery_failure_keep_frac=0.40,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
    SelectionSpec(
        "stage1_recovery_failure_q50_then_timeout_0p50",
        bad_mae_recovery_failure_keep_frac=0.50,
        timeout_lambda=0.50,
        preserve_scope_budget=True,
    ),
)


def _risk_heads_by_name(names: list[str]) -> list[RiskHeadSpec]:
    available = {spec.name: spec for spec in RISK_HEAD_SPECS}
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Unknown risk head(s): {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def _selection_specs_by_name(names: list[str]) -> list[SelectionSpec]:
    available = {spec.name: spec for spec in SELECTION_SPECS}
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Unknown selection(s): {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def _bool_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    lowered = values.astype(str).str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _source_bucket_mask(frame: pd.DataFrame, source_bucket: str) -> pd.Series:
    if source_bucket == "all_rows":
        return pd.Series(True, index=frame.index)
    if "primary_source_tag" in frame.columns:
        primary = frame["primary_source_tag"].fillna("").astype(str).eq(str(source_bucket))
        if bool(primary.any()):
            return primary
    tag_col = f"tag_{source_bucket}"
    if tag_col in frame.columns:
        return _bool_series(frame, tag_col)
    tag_col_v2 = f"tag_{source_bucket}_archetype"
    if tag_col_v2 in frame.columns:
        return _bool_series(frame, tag_col_v2)
    return pd.Series(False, index=frame.index)


def _rank_top_fraction_indices(score: pd.Series, top_frac: float) -> np.ndarray:
    score_s = _safe_numeric(score).reset_index(drop=True)
    valid = score_s.notna().to_numpy()
    if not bool(valid.any()):
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = min(max(1, int(math.ceil(float(top_frac) * len(valid_idx)))), len(valid_idx))
    return _rank_top_k_indices(score_s, k)


def _rank_top_k_indices(score: pd.Series, k: int) -> np.ndarray:
    score_s = _safe_numeric(score).reset_index(drop=True)
    valid = score_s.notna().to_numpy()
    if not bool(valid.any()) or int(k) <= 0:
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = min(int(k), len(valid_idx))
    order = np.argsort(-score_s.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False)


def _low_risk_keep_mask(risk_pred: pd.Series, keep_frac: float | None, eligible: pd.Series) -> pd.Series:
    out = pd.Series(True, index=risk_pred.index)
    if keep_frac is None:
        return out
    risk = _safe_numeric(risk_pred).reset_index(drop=True)
    base = pd.Series(eligible.to_numpy(dtype=bool, copy=False), index=risk.index)
    valid = base & risk.notna()
    valid_idx = np.flatnonzero(valid.to_numpy())
    keep = min(max(1, int(math.ceil(float(keep_frac) * len(valid_idx)))), len(valid_idx))
    mask = pd.Series(False, index=risk.index)
    if keep <= 0:
        return mask
    order = np.argsort(risk.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    mask.iloc[valid_idx[order[:keep]]] = True
    return mask


def _risk_weights(target_soft: pd.Series, target_hard: pd.Series, train_mask: pd.Series) -> pd.Series:
    soft = _safe_numeric(target_soft).fillna(0.0).clip(0.0, 1.0)
    hard = _safe_numeric(target_hard).fillna(0.0).gt(0.5)
    weights = (0.50 + 1.50 * soft).astype(np.float32)
    train_hard = hard[train_mask]
    if len(train_hard):
        rate = float(train_hard.mean())
        if 0.0 < rate < 1.0:
            pos_mult = min(5.0, 0.5 / max(rate, 1e-6))
            neg_mult = min(5.0, 0.5 / max(1.0 - rate, 1e-6))
            weights = weights.where(~hard, weights * pos_mult)
            weights = weights.where(hard, weights * neg_mult)
    train_mean = float(_safe_numeric(weights[train_mask]).mean()) if int(train_mask.sum()) else float("nan")
    if math.isfinite(train_mean) and train_mean > 0.0:
        weights = weights / train_mean
    return weights.clip(0.10, 5.00).astype(np.float32)


def _build_risk_head_target(
    metrics: pd.DataFrame,
    train_mask: pd.Series,
    spec: RiskHeadSpec,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    utility = _safe_numeric(metrics["u_policy_net"])
    mae_norm = _safe_numeric(metrics["mae_norm"]).fillna(0.0)
    barrier = _safe_numeric(metrics["barrier"]).fillna(0.0)
    timeout = metrics["is_timeout"].astype(float).fillna(0.0)
    bars_policy = _safe_numeric(metrics["bars_policy"]).fillna(24.0).clip(lower=0.0)
    report: dict[str, Any] = {"risk_head": spec.name, "kind": spec.kind}
    weight_multiplier: pd.Series | None = None
    if spec.kind == "bad_mae":
        target_soft = (0.70 * mae_norm.ge(1.0).astype(float) + 0.30 * (mae_norm / 4.0).clip(0.0, 1.0)).clip(
            0.0, 1.0
        )
        target_hard = mae_norm.ge(1.0).astype(float)
    elif spec.kind == "wide_barrier":
        target_soft = (
            0.70 * barrier.gt(0.025).astype(float) + 0.30 * (barrier / 0.050).clip(0.0, 1.0)
        ).clip(0.0, 1.0)
        target_hard = barrier.gt(0.025).astype(float)
    elif spec.kind == "timeout":
        target_soft = timeout.clip(0.0, 1.0)
        target_hard = timeout.gt(0.5).astype(float)
    elif spec.kind == "holding":
        train_bars = bars_policy[train_mask].dropna()
        threshold = float(train_bars.quantile(0.80)) if len(train_bars) else 16.0
        threshold = max(threshold, 1.0)
        target_soft = (bars_policy / threshold).clip(0.0, 1.0)
        target_hard = bars_policy.gt(threshold).astype(float)
        report["holding_q80_threshold"] = threshold
    elif spec.kind == "bad_mae_wide25":
        bad_mae = mae_norm.ge(1.0)
        wide25 = barrier.gt(0.025)
        joint = bad_mae & wide25
        target_soft = (
            0.70 * joint.astype(float)
            + 0.20 * (mae_norm / 4.0).clip(0.0, 1.0) * wide25.astype(float)
            + 0.10 * (barrier / 0.050).clip(0.0, 1.0) * bad_mae.astype(float)
        ).clip(0.0, 1.0)
        target_hard = joint.astype(float)
    elif spec.kind == "adverse_path_loss":
        loss = utility.lt(0.0)
        loss_severity = (-utility / 0.030).clip(0.0, 1.0)
        bad_mae = mae_norm.ge(1.0)
        near_bad_mae = mae_norm.ge(0.75)
        wide25 = barrier.gt(0.025)
        bad_wide_loss = loss & bad_mae & wide25
        adverse_loss = loss & (bad_mae | (near_bad_mae & wide25))
        target_soft = (
            0.45 * loss_severity
            + 0.25 * (mae_norm / 4.0).clip(0.0, 1.0) * loss.astype(float)
            + 0.20 * adverse_loss.astype(float)
            + 0.10 * bad_wide_loss.astype(float)
        ).clip(0.0, 1.0)
        target_hard = adverse_loss.astype(float)
    elif spec.kind == "bad_mae_loss":
        loss = utility.lt(0.0)
        loss_severity = (-utility / 0.030).clip(0.0, 1.0)
        bad_mae = mae_norm.ge(1.0)
        bad_mae_loss = loss & bad_mae
        target_soft = (
            0.55 * bad_mae_loss.astype(float)
            + 0.25 * loss_severity * bad_mae.astype(float)
            + 0.20 * (mae_norm / 4.0).clip(0.0, 1.0) * loss.astype(float)
        ).clip(0.0, 1.0)
        target_hard = bad_mae_loss.astype(float)
    elif spec.kind == "fast_bad_mae_loss":
        loss = utility.lt(0.0)
        loss_severity = (-utility / 0.030).clip(0.0, 1.0)
        bad_mae = mae_norm.ge(1.0)
        fast = bars_policy.le(4.0)
        fast_bad_mae_loss = loss & bad_mae & fast
        target_soft = (
            0.60 * fast_bad_mae_loss.astype(float)
            + 0.20 * loss_severity * bad_mae.astype(float) * fast.astype(float)
            + 0.20 * (mae_norm / 4.0).clip(0.0, 1.0) * loss.astype(float) * fast.astype(float)
        ).clip(0.0, 1.0)
        target_hard = fast_bad_mae_loss.astype(float)
    elif spec.kind == "bad_mae_recovery_failure":
        loss = utility.lt(0.0)
        recovered = utility.gt(0.0)
        loss_severity = (-utility / 0.030).clip(0.0, 1.0)
        bad_mae = mae_norm.ge(1.0)
        near_bad_mae = mae_norm.ge(0.75)
        bad_mae_failure = loss & bad_mae
        bad_mae_recovered = recovered & bad_mae
        target_soft = (
            0.70 * bad_mae_failure.astype(float)
            + 0.20 * loss_severity * bad_mae.astype(float)
            + 0.10 * (mae_norm / 4.0).clip(0.0, 1.0) * loss.astype(float)
        ).clip(0.0, 1.0)
        target_hard = bad_mae_failure.astype(float)
        weight_multiplier = pd.Series(0.25, index=metrics.index, dtype=np.float32)
        weight_multiplier.loc[near_bad_mae] = 0.75
        weight_multiplier.loc[bad_mae & ~(loss | recovered)] = 1.00
        weight_multiplier.loc[bad_mae_recovered] = 2.00
        weight_multiplier.loc[bad_mae_failure] = 2.00
        report.update(
            {
                "contrastive_bad_mae_failure_rows": int(bad_mae_failure.sum()),
                "contrastive_bad_mae_recovered_rows": int(bad_mae_recovered.sum()),
            }
        )
    else:
        raise ValueError(f"Unsupported risk head kind: {spec.kind}")
    target = pd.DataFrame({"target_soft": target_soft, "target_hard": target_hard}, index=metrics.index)
    weights = _risk_weights(target["target_soft"], target["target_hard"], train_mask)
    if weight_multiplier is not None:
        weights = (weights * weight_multiplier).astype(np.float32)
        train_mean = float(_safe_numeric(weights[train_mask]).mean()) if int(train_mask.sum()) else float("nan")
        if math.isfinite(train_mean) and train_mean > 0.0:
            weights = weights / train_mean
        weights = weights.clip(0.05, 6.00).astype(np.float32)
    train_hard_rate = _safe_mean(target.loc[train_mask, "target_hard"])
    valid_hard_rate = _safe_mean(target.loc[~train_mask, "target_hard"])
    report.update(
        {
            "train_target_hard_rate": train_hard_rate,
            "future_target_hard_rate": valid_hard_rate,
            "target_ic_utility_train": _spearman(target.loc[train_mask, "target_soft"], utility.loc[train_mask]),
        }
    )
    return target, weights, report


def _week_start(ts: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(ts, utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.to_period("W-SUN")
        .apply(lambda value: value.start_time.date().isoformat() if pd.notna(value) else "")
    )


def _path_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {
            "rows": 0,
            "mean_u": float("nan"),
            "median_u": float("nan"),
            "q10_u": float("nan"),
            "hit_u": float("nan"),
            "bad_mae_1r_rate": float("nan"),
            "p90_mae_norm": float("nan"),
            "timeout_rate": float("nan"),
            "wide_barrier_25bps_rate": float("nan"),
            "wide_barrier_35bps_rate": float("nan"),
            "mean_bars_policy": float("nan"),
            "bad_mae_negative_rate": float("nan"),
            "bad_mae_recovered_rate": float("nan"),
            "fast_bad_mae_negative_rate": float("nan"),
            "fast_bad_mae_recovered_rate": float("nan"),
            "late_bad_mae_negative_rate": float("nan"),
            "late_bad_mae_recovered_rate": float("nan"),
            "bad_mae_negative_share_of_bad_mae": float("nan"),
            "bad_mae_recovered_share_of_bad_mae": float("nan"),
            "bad_mae_negative_mean_u": float("nan"),
            "bad_mae_recovered_mean_u": float("nan"),
        }
    utility = _safe_numeric(metrics["u_policy_net"])
    mae_norm = _safe_numeric(metrics["mae_norm"])
    bars_policy = _safe_numeric(metrics["bars_policy"]).fillna(24.0)
    bad_mae = mae_norm >= 1.0
    negative_bad_mae = bad_mae & utility.lt(0.0)
    recovered_bad_mae = bad_mae & utility.gt(0.0)
    fast = bars_policy <= 4.0
    bad_mae_rate = _safe_mean(bad_mae)
    negative_bad_mae_rate = _safe_mean(negative_bad_mae)
    recovered_bad_mae_rate = _safe_mean(recovered_bad_mae)
    return {
        "rows": int(len(metrics)),
        "mean_u": _safe_mean(utility),
        "median_u": _safe_quantile(utility, 0.50),
        "q10_u": _safe_quantile(utility, 0.10),
        "hit_u": _safe_mean(utility > 0.0),
        "bad_mae_1r_rate": bad_mae_rate,
        "p90_mae_norm": _safe_quantile(mae_norm, 0.90),
        "timeout_rate": _safe_mean(metrics["is_timeout"].astype(float)),
        "wide_barrier_25bps_rate": _safe_mean(metrics["barrier"] > 0.025),
        "wide_barrier_35bps_rate": _safe_mean(metrics["barrier"] > 0.035),
        "mean_bars_policy": _safe_mean(bars_policy),
        "bad_mae_negative_rate": negative_bad_mae_rate,
        "bad_mae_recovered_rate": recovered_bad_mae_rate,
        "fast_bad_mae_negative_rate": _safe_mean(negative_bad_mae & fast),
        "fast_bad_mae_recovered_rate": _safe_mean(recovered_bad_mae & fast),
        "late_bad_mae_negative_rate": _safe_mean(negative_bad_mae & ~fast),
        "late_bad_mae_recovered_rate": _safe_mean(recovered_bad_mae & ~fast),
        "bad_mae_negative_share_of_bad_mae": (
            negative_bad_mae_rate / bad_mae_rate
            if math.isfinite(bad_mae_rate) and bad_mae_rate > 0.0
            else float("nan")
        ),
        "bad_mae_recovered_share_of_bad_mae": (
            recovered_bad_mae_rate / bad_mae_rate
            if math.isfinite(bad_mae_rate) and bad_mae_rate > 0.0
            else float("nan")
        ),
        "bad_mae_negative_mean_u": _safe_mean(utility[negative_bad_mae]),
        "bad_mae_recovered_mean_u": _safe_mean(utility[recovered_bad_mae]),
    }


def _risk_adjusted_score(
    utility_pred: pd.Series,
    risk_preds: dict[str, pd.Series],
    selection: SelectionSpec,
) -> pd.Series:
    score = _safe_numeric(utility_pred).copy()
    penalties = {
        "bad_mae_risk_v1": selection.bad_mae_lambda,
        "wide_barrier_risk_v1": selection.wide_barrier_lambda,
        "timeout_risk_v1": selection.timeout_lambda,
        "holding_risk_v1": selection.holding_lambda,
        "bad_mae_wide25_risk_v1": selection.bad_mae_wide25_lambda,
        "adverse_path_loss_risk_v1": selection.adverse_path_loss_lambda,
        "bad_mae_loss_risk_v1": selection.bad_mae_loss_lambda,
        "fast_bad_mae_loss_risk_v1": selection.fast_bad_mae_loss_lambda,
        "bad_mae_recovery_failure_risk_v1": selection.bad_mae_recovery_failure_lambda,
    }
    for head, penalty in penalties.items():
        if not penalty:
            continue
        if head not in risk_preds:
            raise ValueError(f"Selection {selection.name} requires missing risk head {head}")
        score = score - float(penalty) * _safe_numeric(risk_preds[head])
    return score


def _select_indices(
    *,
    utility_pred: pd.Series,
    risk_preds: dict[str, pd.Series],
    selection: SelectionSpec,
    top_frac: float,
) -> tuple[np.ndarray, pd.Series, dict[str, Any]]:
    score = _risk_adjusted_score(utility_pred, risk_preds, selection)
    used_heads = selection.required_heads
    base_eligible = score.notna()
    for head in used_heads:
        base_eligible = base_eligible & risk_preds[head].notna()
    eligible = base_eligible.copy()
    stage_reports: dict[str, Any] = {
        "score_eligible_rows": int(base_eligible.sum()),
        "preserve_scope_budget": bool(selection.preserve_scope_budget),
    }
    gate_specs = {
        "bad_mae_risk_v1": selection.bad_mae_keep_frac,
        "wide_barrier_risk_v1": selection.wide_barrier_keep_frac,
        "timeout_risk_v1": selection.timeout_keep_frac,
        "bad_mae_wide25_risk_v1": selection.bad_mae_wide25_keep_frac,
        "adverse_path_loss_risk_v1": selection.adverse_path_loss_keep_frac,
        "bad_mae_loss_risk_v1": selection.bad_mae_loss_keep_frac,
        "fast_bad_mae_loss_risk_v1": selection.fast_bad_mae_loss_keep_frac,
        "bad_mae_recovery_failure_risk_v1": selection.bad_mae_recovery_failure_keep_frac,
    }
    for head, keep_frac in gate_specs.items():
        if keep_frac is None:
            continue
        if head not in risk_preds:
            raise ValueError(f"Selection {selection.name} requires missing risk head {head}")
        keep_mask = _low_risk_keep_mask(risk_preds[head], keep_frac, base_eligible)
        eligible = eligible & keep_mask
        stage_reports[f"{head}_keep_frac"] = float(keep_frac)
        stage_reports[f"{head}_gate_rows"] = int(keep_mask.sum())
    eligible_count = int(eligible.sum())
    stage_reports["eligible_rows_after_stage1"] = eligible_count
    gated_score = score.where(eligible)
    if selection.preserve_scope_budget:
        budget_rows = min(
            max(1, int(math.ceil(float(top_frac) * int(base_eligible.sum())))),
            eligible_count,
        )
        selected = _rank_top_k_indices(gated_score, budget_rows)
    else:
        budget_rows = min(
            max(1, int(math.ceil(float(top_frac) * eligible_count))) if eligible_count else 0,
            eligible_count,
        )
        selected = _rank_top_fraction_indices(gated_score, top_frac)
    stage_reports["selection_budget_rows"] = int(budget_rows)
    return selected, gated_score, stage_reports


def _selected_frame(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    utility_target: pd.DataFrame,
    risk_targets: dict[str, pd.DataFrame],
    utility_pred: pd.Series,
    risk_preds: dict[str, pd.Series],
    final_score: pd.Series,
    selected_idx: np.ndarray,
    context: dict[str, Any],
) -> pd.DataFrame:
    if not len(selected_idx):
        return pd.DataFrame()
    ledger_cols = ["__ts__", "__symbol__"]
    for col in ("side", "side_name", "__side__", "timeframe", "candidate_id", "primary_source_tag"):
        if col in frame.columns:
            ledger_cols.append(col)
    selected = frame.iloc[selected_idx][ledger_cols].copy()
    if "side" not in selected.columns and "side" in metrics.columns:
        selected["side"] = metrics["side"].iloc[selected_idx].to_numpy(dtype=np.int8, copy=False)
    if "side_name" not in selected.columns and "side" in selected.columns:
        selected["side_name"] = np.where(_safe_numeric(selected["side"]) < 0.0, "short", "long")
    selected.insert(0, "candidate", context["candidate"])
    for key, value in context.items():
        if key == "candidate":
            continue
        selected[key] = value
    selected["week_start"] = _week_start(selected["__ts__"])
    selected["final_score"] = final_score.iloc[selected_idx].to_numpy(dtype=np.float32, copy=False)
    selected["utility_pred"] = utility_pred.iloc[selected_idx].to_numpy(dtype=np.float32, copy=False)
    selected["utility_target_soft"] = utility_target["target_soft"].iloc[selected_idx].to_numpy()
    for head, pred in risk_preds.items():
        selected[f"{head}_pred"] = pred.iloc[selected_idx].to_numpy(dtype=np.float32, copy=False)
        if head in risk_targets:
            selected[f"{head}_target_soft"] = risk_targets[head]["target_soft"].iloc[selected_idx].to_numpy()
            selected[f"{head}_target_hard"] = risk_targets[head]["target_hard"].iloc[selected_idx].to_numpy()
    selected["u_policy_net"] = metrics["u_policy_net"].iloc[selected_idx].to_numpy()
    selected["mae_norm"] = metrics["mae_norm"].iloc[selected_idx].to_numpy()
    selected["barrier"] = metrics["barrier"].iloc[selected_idx].to_numpy()
    selected["is_timeout"] = metrics["is_timeout"].iloc[selected_idx].to_numpy()
    selected["bars_policy"] = metrics["bars_policy"].iloc[selected_idx].to_numpy()
    return selected


def _weekly_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    group_cols = [
        "candidate",
        "period",
        "week_start",
        "label",
        "risk_heads",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "selection",
        "top_frac",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(group_cols, dropna=False, observed=True):
        context = dict(zip(group_cols, key, strict=False))
        metrics = pd.DataFrame(
            {
                "u_policy_net": group["u_policy_net"],
                "mae_norm": group["mae_norm"],
                "barrier": group["barrier"],
                "is_timeout": group["is_timeout"],
                "bars_policy": group["bars_policy"],
            }
        )
        side = _safe_numeric(group["side"]) if "side" in group.columns else pd.Series(dtype=float)
        side_name = side.map(lambda value: "short" if value < 0.0 else "long")
        side_top_share = (
            float(side_name.value_counts(normalize=True).iloc[0]) if len(side_name) else float("nan")
        )
        row = {
            **context,
            **_path_summary(metrics),
            "top_symbol_share": (
                float(group["__symbol__"].value_counts(normalize=True).iloc[0]) if len(group) else float("nan")
            ),
            "unique_symbols": int(group["__symbol__"].nunique()),
            "long_share": _safe_mean(side > 0.0) if len(side) else float("nan"),
            "short_share": _safe_mean(side < 0.0) if len(side) else float("nan"),
            "side_top_share": side_top_share,
            "mean_final_score": _safe_mean(group["final_score"]),
            "mean_utility_pred": _safe_mean(group["utility_pred"]),
        }
        for col in group.columns:
            if col.endswith("_pred") and col != "utility_pred":
                row[f"mean_{col}"] = _safe_mean(group[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate", "period", "week_start"], kind="mergesort")


def _add_utility_only_deltas(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    keys = ["period", "label", "risk_heads", "feature_set", "source_bucket", "causal_gate", "top_frac"]
    baseline_cols = [
        "selected_rows",
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "bad_mae_negative_rate",
        "bad_mae_recovered_rate",
        "fast_bad_mae_negative_rate",
        "fast_bad_mae_recovered_rate",
        "p90_mae_norm",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "mean_bars_policy",
    ]
    base = monthly[monthly["selection"].eq("utility_only")][keys + baseline_cols].rename(
        columns={col: f"utility_only_{col}" for col in baseline_cols}
    )
    out = monthly.merge(base, on=keys, how="left", validate="many_to_one")
    for col in baseline_cols:
        out[f"delta_{col}_vs_utility_only"] = _safe_numeric(out[col]) - _safe_numeric(out[f"utility_only_{col}"])
    return out


def _selected_group_stats(selected_group: pd.DataFrame) -> dict[str, Any]:
    if selected_group.empty:
        return {
            "overall_top_symbol_share": float("nan"),
            "overall_side_top_share": float("nan"),
            "utility_without_top_symbol": float("nan"),
            "positive_weeks_without_top_symbol": 0,
            "weeks_without_top_symbol": 0,
        }
    shares = selected_group["__symbol__"].value_counts(normalize=True)
    if "side" in selected_group.columns:
        side = _safe_numeric(selected_group["side"])
        side_names = side.map(lambda value: "short" if value < 0.0 else "long")
        side_shares = side_names.value_counts(normalize=True)
        overall_side_top_share = float(side_shares.iloc[0]) if len(side_shares) else float("nan")
    else:
        overall_side_top_share = float("nan")
    top_symbol = str(shares.index[0])
    without = selected_group[selected_group["__symbol__"].astype(str).ne(top_symbol)].copy()
    if without.empty:
        return {
            "overall_top_symbol_share": float(shares.iloc[0]),
            "overall_side_top_share": overall_side_top_share,
            "utility_without_top_symbol": float("nan"),
            "positive_weeks_without_top_symbol": 0,
            "weeks_without_top_symbol": 0,
        }
    weekly_without = without.groupby("week_start", dropna=False)["u_policy_net"].mean()
    return {
        "overall_top_symbol_share": float(shares.iloc[0]),
        "overall_side_top_share": overall_side_top_share,
        "utility_without_top_symbol": _safe_mean(without["u_policy_net"]),
        "positive_weeks_without_top_symbol": int((_safe_numeric(weekly_without) > 0.0).sum()),
        "weeks_without_top_symbol": int(len(weekly_without)),
    }


def _aggregate(monthly: pd.DataFrame, weekly: pd.DataFrame, selected: pd.DataFrame, *, expected_months: int) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    group_cols = [
        "label",
        "risk_heads",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "selection",
        "top_frac",
    ]
    weekly_map: dict[tuple[Any, ...], pd.DataFrame] = {}
    if not weekly.empty:
        for key, group in weekly.groupby(group_cols, dropna=False, observed=True):
            weekly_map[key] = group
    selected_map: dict[tuple[Any, ...], pd.DataFrame] = {}
    if not selected.empty:
        for key, group in selected.groupby(group_cols, dropna=False, observed=True):
            selected_map[key] = group
    rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        label, risk_heads, feature_set, source_bucket, causal_gate, selection, top_frac = key
        month_count = int(group["period"].nunique())
        positive_months = int((_safe_numeric(group["mean_u"]) > 0.0).sum())
        utility_ic_u_positive_months = int((_safe_numeric(group["utility_score_ic_u_scope"]) > 0.0).sum())
        min_selected = _safe_quantile(group["selected_rows"], 0.0)
        mean_u = _safe_mean(group["mean_u"])
        worst_month_u = _safe_quantile(group["mean_u"], 0.0)
        mean_bad = _safe_mean(group["bad_mae_1r_rate"])
        mean_bad_negative = _safe_mean(group["bad_mae_negative_rate"])
        mean_bad_recovered = _safe_mean(group["bad_mae_recovered_rate"])
        mean_fast_bad_negative = _safe_mean(group["fast_bad_mae_negative_rate"])
        mean_fast_bad_recovered = _safe_mean(group["fast_bad_mae_recovered_rate"])
        mean_timeout = _safe_mean(group["timeout_rate"])
        mean_wide = _safe_mean(group["wide_barrier_25bps_rate"])
        weekly_group = weekly_map.get(key, pd.DataFrame())
        weeks = int(len(weekly_group))
        positive_weeks = int((_safe_numeric(weekly_group.get("mean_u", pd.Series(dtype=float))) > 0.0).sum())
        q25_week_u = _safe_quantile(weekly_group.get("mean_u", pd.Series(dtype=float)), 0.25)
        worst_week_u = _safe_quantile(weekly_group.get("mean_u", pd.Series(dtype=float)), 0.0)
        max_week_top_symbol_share = _safe_quantile(weekly_group.get("top_symbol_share", pd.Series(dtype=float)), 1.0)
        max_week_side_top_share = _safe_quantile(weekly_group.get("side_top_share", pd.Series(dtype=float)), 1.0)
        selected_stats = _selected_group_stats(selected_map.get(key, pd.DataFrame()))
        monthly_ok = (
            month_count >= int(expected_months)
            and positive_months >= int(expected_months)
            and math.isfinite(mean_u)
            and mean_u > 0.0
            and math.isfinite(worst_month_u)
            and worst_month_u > 0.0
            and math.isfinite(min_selected)
            and min_selected >= 25.0
        )
        risk_ok = (
            math.isfinite(mean_bad)
            and mean_bad <= 0.45
            and math.isfinite(mean_timeout)
            and mean_timeout <= 0.15
            and math.isfinite(mean_wide)
            and mean_wide <= 0.10
        )
        concentration_ok = (
            math.isfinite(selected_stats["overall_top_symbol_share"])
            and selected_stats["overall_top_symbol_share"] < 0.35
            and math.isfinite(selected_stats["utility_without_top_symbol"])
            and selected_stats["utility_without_top_symbol"] > 0.0
        )
        weekly_ok = (
            weeks >= 10
            and positive_weeks >= math.ceil(0.75 * weeks)
            and math.isfinite(q25_week_u)
            and q25_week_u >= 0.0
        )
        risk_reduced = (
            _safe_mean(group.get("delta_bad_mae_1r_rate_vs_utility_only", pd.Series(dtype=float))) < -0.05
            or _safe_mean(group.get("delta_timeout_rate_vs_utility_only", pd.Series(dtype=float))) < -0.03
            or _safe_mean(group.get("delta_wide_barrier_25bps_rate_vs_utility_only", pd.Series(dtype=float))) < -0.03
        )
        if monthly_ok and risk_ok and concentration_ok and weekly_ok:
            decision = "candidate_for_label_ablation"
        elif monthly_ok and risk_ok and concentration_ok and not weekly_ok:
            decision = "monthly_positive_weekly_unstable"
        elif monthly_ok and (not risk_ok or not concentration_ok):
            decision = "monthly_positive_path_or_concentration_limits_fail"
        elif risk_reduced and math.isfinite(mean_u) and mean_u > 0.0:
            decision = "risk_reduced_not_enough"
        else:
            decision = "diagnostic_only"
        rows.append(
            {
                "decision": decision,
                "label": label,
                "risk_heads": risk_heads,
                "feature_set": feature_set,
                "source_bucket": source_bucket,
                "causal_gate": causal_gate,
                "selection": selection,
                "top_frac": float(top_frac),
                "months": month_count,
                "positive_months": positive_months,
                "utility_ic_u_positive_months": utility_ic_u_positive_months,
                "mean_u": mean_u,
                "worst_month_u": worst_month_u,
                "q10_u": _safe_mean(group["q10_u"]),
                "hit_u": _safe_mean(group["hit_u"]),
                "bad_mae_1r_rate": mean_bad,
                "bad_mae_negative_rate": mean_bad_negative,
                "bad_mae_recovered_rate": mean_bad_recovered,
                "fast_bad_mae_negative_rate": mean_fast_bad_negative,
                "fast_bad_mae_recovered_rate": mean_fast_bad_recovered,
                "late_bad_mae_negative_rate": _safe_mean(group["late_bad_mae_negative_rate"]),
                "late_bad_mae_recovered_rate": _safe_mean(group["late_bad_mae_recovered_rate"]),
                "bad_mae_negative_share_of_bad_mae": _safe_mean(group["bad_mae_negative_share_of_bad_mae"]),
                "bad_mae_recovered_share_of_bad_mae": _safe_mean(group["bad_mae_recovered_share_of_bad_mae"]),
                "bad_mae_negative_mean_u": _safe_mean(group["bad_mae_negative_mean_u"]),
                "bad_mae_recovered_mean_u": _safe_mean(group["bad_mae_recovered_mean_u"]),
                "p90_mae_norm": _safe_mean(group["p90_mae_norm"]),
                "timeout_rate": mean_timeout,
                "wide_barrier_25bps_rate": mean_wide,
                "wide_barrier_35bps_rate": _safe_mean(group["wide_barrier_35bps_rate"]),
                "mean_bars_policy": _safe_mean(group["mean_bars_policy"]),
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": min_selected,
                "utility_score_ic_u_scope": _safe_mean(group["utility_score_ic_u_scope"]),
                "delta_mean_u_vs_utility_only": _safe_mean(group.get("delta_mean_u_vs_utility_only")),
                "delta_bad_mae_1r_rate_vs_utility_only": _safe_mean(
                    group.get("delta_bad_mae_1r_rate_vs_utility_only")
                ),
                "delta_bad_mae_negative_rate_vs_utility_only": _safe_mean(
                    group.get("delta_bad_mae_negative_rate_vs_utility_only")
                ),
                "delta_bad_mae_recovered_rate_vs_utility_only": _safe_mean(
                    group.get("delta_bad_mae_recovered_rate_vs_utility_only")
                ),
                "delta_fast_bad_mae_negative_rate_vs_utility_only": _safe_mean(
                    group.get("delta_fast_bad_mae_negative_rate_vs_utility_only")
                ),
                "delta_fast_bad_mae_recovered_rate_vs_utility_only": _safe_mean(
                    group.get("delta_fast_bad_mae_recovered_rate_vs_utility_only")
                ),
                "delta_timeout_rate_vs_utility_only": _safe_mean(group.get("delta_timeout_rate_vs_utility_only")),
                "delta_wide_barrier_25bps_rate_vs_utility_only": _safe_mean(
                    group.get("delta_wide_barrier_25bps_rate_vs_utility_only")
                ),
                "weeks": weeks,
                "positive_weeks": positive_weeks,
                "q25_week_u": q25_week_u,
                "worst_week_u": worst_week_u,
                "max_week_top_symbol_share": max_week_top_symbol_share,
                "max_week_side_top_share": max_week_side_top_share,
                **selected_stats,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["decision", "mean_u", "q25_week_u", "bad_mae_1r_rate"],
        ascending=[True, False, False, True],
        na_position="last",
        kind="mergesort",
    )


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_report(output_dir: Path, aggregate: pd.DataFrame, weekly: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "source_utility_path_timeout_risk_report.md"
    cols = [
        "decision",
        "label",
        "risk_heads",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "selection",
        "top_frac",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_u",
        "bad_mae_1r_rate",
        "bad_mae_negative_rate",
        "bad_mae_recovered_rate",
        "fast_bad_mae_negative_rate",
        "fast_bad_mae_recovered_rate",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "mean_selected_rows",
        "min_selected_rows",
        "delta_mean_u_vs_utility_only",
        "delta_bad_mae_1r_rate_vs_utility_only",
        "delta_bad_mae_negative_rate_vs_utility_only",
        "delta_bad_mae_recovered_rate_vs_utility_only",
        "delta_timeout_rate_vs_utility_only",
        "delta_wide_barrier_25bps_rate_vs_utility_only",
        "weeks",
        "positive_weeks",
        "q25_week_u",
        "worst_week_u",
        "overall_top_symbol_share",
        "utility_without_top_symbol",
        "max_week_top_symbol_share",
    ]
    candidate = aggregate[aggregate["decision"].eq("candidate_for_label_ablation")]
    unstable = aggregate[aggregate["decision"].eq("monthly_positive_weekly_unstable")]
    path_fail = aggregate[aggregate["decision"].eq("monthly_positive_path_or_concentration_limits_fail")]
    risk_reduced = aggregate[aggregate["decision"].eq("risk_reduced_not_enough")]
    top = aggregate[aggregate["top_frac"].isin([0.01, 0.03, 0.10])].sort_values(
        ["mean_u", "q25_week_u"], ascending=[False, False]
    )
    weekly_cols = [
        "candidate",
        "period",
        "week_start",
        "rows",
        "mean_u",
        "q10_u",
        "hit_u",
        "bad_mae_1r_rate",
        "bad_mae_negative_rate",
        "bad_mae_recovered_rate",
        "fast_bad_mae_negative_rate",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "top_symbol_share",
        "unique_symbols",
    ]
    lines = [
        "# Source Utility Path-Time Risk Diagnostic",
        "",
        "Scope: utility prediction plus separate bad-MAE, wide-barrier, and timeout/holding risk heads.",
        "Training uses prior months only; selections use future-month causal predictions only.",
        "",
        f"Rows joined to label ledger: `{manifest['rows']}`",
        f"Utility source: `{manifest.get('utility_source', '')}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Labels: `{', '.join(manifest['labels'])}`",
        f"Risk heads: `{', '.join(manifest['risk_heads'])}`",
        f"Feature sets: `{', '.join(manifest['feature_sets'])}`",
        "",
        "## Promotion Gates",
        "",
        "- mean utility > 0 and every month positive",
        "- positive weeks >= 75% and weekly q25 utility >= 0",
        "- bad-MAE <= 0.45, timeout <= 0.15, wide-barrier <= 0.10",
        "- recovered-vs-negative bad-MAE is reported diagnostically only; raw bad-MAE remains the hard gate",
        "- overall top-symbol share < 0.35 and utility excluding top symbol > 0",
        "",
        "## Candidates",
        "",
        _table(candidate, cols, limit=80),
        "",
        "## Monthly Positive But Weekly Unstable",
        "",
        _table(unstable, cols, limit=80),
        "",
        "## Monthly Positive But Path Or Concentration Limits Fail",
        "",
        _table(path_fail, cols, limit=80),
        "",
        "## Risk Reduced But Not Enough",
        "",
        _table(risk_reduced, cols, limit=80),
        "",
        "## Best Rows By Mean Utility",
        "",
        _table(top, cols, limit=120),
        "",
        "## Worst Weekly Rows",
        "",
        _table(weekly.sort_values("mean_u"), weekly_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Selected rows: `{manifest['outputs']['selected_rows_parquet']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_v2_features(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        return pd.DataFrame(index=frame.index), {"enabled": False, "reason": "missing_v2_path", "path": str(path)}
    v2 = pd.read_parquet(path)
    key_cols = ["__ts__", "__symbol__"]
    if not set(key_cols).issubset(v2.columns):
        return pd.DataFrame(index=frame.index), {"enabled": False, "reason": "missing_v2_keys", "path": str(path)}
    v2["__ts__"] = pd.to_datetime(v2["__ts__"], utc=True, errors="coerce")
    wanted = [
        col
        for col in v2.columns
        if col.endswith("_archetype_score")
        or col in {"prior_symbol_event_density_score", "prior_symbol_event_density_rank"}
        or (col.startswith("tag_") and col.endswith("_archetype"))
    ]
    wanted = [col for col in wanted if col not in frame.columns]
    if not wanted:
        return pd.DataFrame(index=frame.index), {"enabled": False, "reason": "no_new_v2_columns", "path": str(path)}
    dupes = int(v2.duplicated(key_cols).sum())
    if dupes:
        v2 = v2.sort_values(key_cols, kind="mergesort").drop_duplicates(key_cols, keep="last")
    joined = frame[key_cols].merge(v2[key_cols + wanted], on=key_cols, how="left", validate="many_to_one")
    out = pd.DataFrame(index=frame.index)
    for col in wanted:
        if col.startswith("tag_"):
            out[col] = joined[col].fillna(False).astype(bool).astype(np.float32)
        else:
            out[col] = _safe_numeric(joined[col]).astype(np.float32)
    finite = out.notna().mean().to_dict()
    retained = [col for col in out.columns if float(finite.get(col, 0.0) or 0.0) >= 0.50]
    out = out.loc[:, retained].copy()
    return out, {
        "enabled": True,
        "path": str(path),
        "loaded_columns": int(len(wanted)),
        "retained_columns": int(len(retained)),
        "duplicate_keys": dupes,
        "mean_feature_finite_frac": float(np.mean([finite[col] for col in retained])) if retained else 0.0,
    }


def _score_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    month_period: pd.Series,
    month: str,
    labels: dict[str, Any],
    risk_heads: list[RiskHeadSpec],
    gates: dict[str, Any],
    feature_map: dict[str, list[str]],
    labels_requested: list[str],
    feature_sets: list[str],
    source_buckets: list[str],
    causal_gates: list[str],
    selections: list[SelectionSpec],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
    min_scope_rows: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    valid_mask = month_period.eq(month)
    train_mask = month_period < month
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior_months = sorted(month_period[train_mask].dropna().unique())
        keep = set(prior_months[-int(train_lookback_months) :])
        train_mask = train_mask & month_period.isin(keep)
    if int(valid_mask.sum()) < int(min_valid_rows):
        return [], [], [{"period": month, "skipped": True, "reason": "too_few_valid_rows"}]

    rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    risk_target_cache = {
        spec.name: _build_risk_head_target(metrics, train_mask, spec) for spec in risk_heads
    }
    gate_cache = {gate_name: _gate_mask(frame, train_mask, gates[gate_name]) for gate_name in causal_gates}
    risk_head_names = [spec.name for spec in risk_heads]
    risk_heads_key = "+".join(risk_head_names)
    for label_name in labels_requested:
        utility_target, utility_weights, _label_report = _build_target(
            frame=frame,
            metrics=metrics,
            train_mask=train_mask,
            valid_mask=valid_mask,
            spec=labels[label_name],
        )
        train_utility_mask = train_mask & utility_target["target_soft"].notna() & utility_weights.gt(0.0)
        if int(train_utility_mask.sum()) < int(min_train_rows):
            diagnostics.append(
                {
                    "period": month,
                    "label": label_name,
                    "skipped": True,
                    "reason": "too_few_utility_train_rows",
                    "train_rows": int(train_utility_mask.sum()),
                }
            )
            continue
        train_model_mask = train_utility_mask.copy()
        for head_name in risk_head_names:
            target, weights, _report = risk_target_cache[head_name]
            train_model_mask = train_model_mask & target["target_soft"].notna() & weights.gt(0.0)
        if int(train_model_mask.sum()) < int(min_train_rows):
            diagnostics.append(
                {
                    "period": month,
                    "label": label_name,
                    "risk_heads": risk_heads_key,
                    "skipped": True,
                    "reason": "too_few_joint_train_rows",
                    "train_rows": int(train_model_mask.sum()),
                }
            )
            continue
        for feature_set in feature_sets:
            features = feature_map.get(feature_set)
            if not features:
                diagnostics.append(
                    {
                        "period": month,
                        "label": label_name,
                        "risk_heads": risk_heads_key,
                        "feature_set": feature_set,
                        "skipped": True,
                        "reason": "empty_feature_set",
                    }
                )
                continue
            x_train, x_valid = _month_model_frame(
                frame,
                train_mask=train_model_mask,
                valid_mask=valid_mask,
                features=features,
            )
            utility_matrix = np.vstack(
                [
                    _fit_predict(
                        x_train=x_train,
                        y_train=utility_target.loc[train_model_mask, "target_soft"],
                        w_train=utility_weights.loc[train_model_mask],
                        x_valid=x_valid,
                        seed=seed,
                    )
                    for seed in seeds
                ]
            )
            utility_pred = pd.Series(np.nan, index=frame.index, dtype=np.float32)
            utility_pred.loc[valid_mask] = np.mean(utility_matrix, axis=0).astype(np.float32)
            risk_targets: dict[str, pd.DataFrame] = {}
            risk_preds: dict[str, pd.Series] = {}
            risk_reports: dict[str, dict[str, Any]] = {}
            for head_name in risk_head_names:
                risk_target, risk_weights, risk_report = risk_target_cache[head_name]
                risk_targets[head_name] = risk_target
                risk_reports[head_name] = risk_report
                risk_matrix = np.vstack(
                    [
                        _fit_predict(
                            x_train=x_train,
                            y_train=risk_target.loc[train_model_mask, "target_soft"],
                            w_train=risk_weights.loc[train_model_mask],
                            x_valid=x_valid,
                            seed=seed + 10007 + (997 * risk_head_names.index(head_name)),
                        )
                        for seed in seeds
                    ]
                )
                risk_pred = pd.Series(np.nan, index=frame.index, dtype=np.float32)
                risk_pred.loc[valid_mask] = np.mean(risk_matrix, axis=0).astype(np.float32)
                risk_preds[head_name] = risk_pred
            for source_bucket in source_buckets:
                bucket_mask = valid_mask & _source_bucket_mask(frame, source_bucket)
                for causal_gate in causal_gates:
                    gate_mask, gate_report = gate_cache[causal_gate]
                    scope_mask = bucket_mask & gate_mask
                    scope_rows = int(scope_mask.sum())
                    if scope_rows < int(min_scope_rows):
                        continue
                    scope_idx = np.flatnonzero(scope_mask.to_numpy())
                    scope_frame = frame.iloc[scope_idx].reset_index(drop=True)
                    scope_metrics = metrics.iloc[scope_idx].reset_index(drop=True)
                    scope_utility_target = utility_target.iloc[scope_idx].reset_index(drop=True)
                    scope_risk_targets = {
                        head: target.iloc[scope_idx].reset_index(drop=True)
                        for head, target in risk_targets.items()
                    }
                    scope_utility_pred = utility_pred.iloc[scope_idx].reset_index(drop=True)
                    scope_risk_preds = {
                        head: pred.iloc[scope_idx].reset_index(drop=True) for head, pred in risk_preds.items()
                    }
                    scope_diag = {
                        "period": month,
                        "label": label_name,
                        "risk_heads": risk_heads_key,
                        "feature_set": feature_set,
                        "source_bucket": source_bucket,
                        "causal_gate": causal_gate,
                        "train_rows": int(train_model_mask.sum()),
                        "valid_rows": int(valid_mask.sum()),
                        "scope_rows": scope_rows,
                        "gate_missing_columns": ",".join(gate_report.get("missing_gate_columns", [])),
                        "gate_thresholds_json": json.dumps(
                            _json_safe(gate_report.get("thresholds", {})), sort_keys=True
                        ),
                        "utility_score_ic_u_scope": _spearman(scope_utility_pred, scope_metrics["u_policy_net"]),
                        "utility_score_ic_label_scope": _spearman(
                            scope_utility_pred, scope_utility_target["target_soft"]
                        ),
                    }
                    for head_name, pred in scope_risk_preds.items():
                        target = scope_risk_targets[head_name]
                        scope_diag[f"{head_name}_ic_target_scope"] = _spearman(pred, target["target_soft"])
                        scope_diag[f"{head_name}_ic_hard_scope"] = _spearman(pred, target["target_hard"])
                        scope_diag[f"{head_name}_target_hard_rate_scope"] = _safe_mean(target["target_hard"])
                        scope_diag[f"{head_name}_train_hard_rate"] = risk_reports[head_name].get(
                            "train_target_hard_rate"
                        )
                    diagnostics.append({**scope_diag, "skipped": False})
                    for top_frac in top_fracs:
                        for selection in selections:
                            missing_required = [head for head in selection.required_heads if head not in scope_risk_preds]
                            if missing_required:
                                continue
                            selected_local, final_score, selection_report = _select_indices(
                                utility_pred=scope_utility_pred,
                                risk_preds=scope_risk_preds,
                                selection=selection,
                                top_frac=top_frac,
                            )
                            selected_metrics = (
                                scope_metrics.iloc[selected_local].copy()
                                if len(selected_local)
                                else scope_metrics.iloc[:0].copy()
                            )
                            row = {
                                **scope_diag,
                                "selection": selection.name,
                                "top_frac": float(top_frac),
                                "bad_mae_lambda": float(selection.bad_mae_lambda),
                                "wide_barrier_lambda": float(selection.wide_barrier_lambda),
                                "timeout_lambda": float(selection.timeout_lambda),
                                "holding_lambda": float(selection.holding_lambda),
                                "bad_mae_wide25_lambda": float(selection.bad_mae_wide25_lambda),
                                "adverse_path_loss_lambda": float(selection.adverse_path_loss_lambda),
                                "bad_mae_loss_lambda": float(selection.bad_mae_loss_lambda),
                                "fast_bad_mae_loss_lambda": float(selection.fast_bad_mae_loss_lambda),
                                "bad_mae_recovery_failure_lambda": float(
                                    selection.bad_mae_recovery_failure_lambda
                                ),
                                "bad_mae_keep_frac": selection.bad_mae_keep_frac,
                                "wide_barrier_keep_frac": selection.wide_barrier_keep_frac,
                                "timeout_keep_frac": selection.timeout_keep_frac,
                                "bad_mae_wide25_keep_frac": selection.bad_mae_wide25_keep_frac,
                                "adverse_path_loss_keep_frac": selection.adverse_path_loss_keep_frac,
                                "bad_mae_loss_keep_frac": selection.bad_mae_loss_keep_frac,
                                "fast_bad_mae_loss_keep_frac": selection.fast_bad_mae_loss_keep_frac,
                                "bad_mae_recovery_failure_keep_frac": (
                                    selection.bad_mae_recovery_failure_keep_frac
                                ),
                                "preserve_scope_budget": bool(selection.preserve_scope_budget),
                                **selection_report,
                                "eligible_rows": int(final_score.notna().sum()),
                                "selected_rows": int(len(selected_metrics)),
                                **_path_summary(selected_metrics),
                                "mean_final_score": _safe_mean(final_score.iloc[selected_local]),
                                "mean_utility_pred": _safe_mean(scope_utility_pred.iloc[selected_local]),
                            }
                            for head_name, pred in scope_risk_preds.items():
                                row[f"mean_{head_name}_pred"] = _safe_mean(pred.iloc[selected_local])
                                row[f"mean_{head_name}_target_soft"] = _safe_mean(
                                    scope_risk_targets[head_name]["target_soft"].iloc[selected_local]
                                )
                            rows.append(row)
                            context = {
                                "candidate": (
                                    f"{label_name}__{risk_heads_key}__{feature_set}__{source_bucket}__"
                                    f"{causal_gate}__{selection.name}__top{top_frac}"
                                ),
                                "period": month,
                                "label": label_name,
                                "risk_heads": risk_heads_key,
                                "feature_set": feature_set,
                                "source_bucket": source_bucket,
                                "causal_gate": causal_gate,
                                "selection": selection.name,
                                "top_frac": float(top_frac),
                                "scope_rows": scope_rows,
                            }
                            selected_frames.append(
                                _selected_frame(
                                    frame=scope_frame,
                                    metrics=scope_metrics,
                                    utility_target=scope_utility_target,
                                    risk_targets=scope_risk_targets,
                                    utility_pred=scope_utility_pred,
                                    risk_preds=scope_risk_preds,
                                    final_score=final_score,
                                    selected_idx=selected_local,
                                    context=context,
                                )
                            )
    return rows, selected_frames, diagnostics


def run_diagnostic(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    archetypes_v2_path: Path,
    max_feature_store_features: int | None,
    months: list[str],
    labels_requested: list[str],
    risk_head_names: list[str],
    feature_sets: list[str],
    source_buckets: list[str],
    causal_gate_names: list[str],
    selection_names: list[str],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
    min_scope_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {spec.name: spec for spec in _label_specs_by_name(labels_requested)}
    risk_heads = _risk_heads_by_name(risk_head_names)
    gates = {spec.name: spec for spec in _gate_specs_by_name(causal_gate_names)}
    selections = _selection_specs_by_name(selection_names)
    _assert_gate_columns_causal(list(gates.values()))

    frame, join_report = _load_joined_frame(quality_labels_path=quality_labels_path, labels_path=labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    source_features = _source_feature_columns(frame)
    v2_matrix, v2_report = _load_v2_features(frame, archetypes_v2_path)
    for col in v2_matrix.columns:
        frame[col] = v2_matrix[col].to_numpy(dtype=np.float32, copy=False)

    metrics = _path_metrics(frame)
    base_features = list(feature_matrix.columns)
    v2_features = list(v2_matrix.columns)
    feature_map = {
        "base": base_features,
        "base_plus_source": list(dict.fromkeys(base_features + source_features)),
        "source_only": source_features,
        "v2_only": v2_features,
        "base_plus_v2": list(dict.fromkeys(base_features + v2_features)),
        "base_plus_source_v2": list(dict.fromkeys(base_features + source_features + v2_features)),
    }
    month_period = frame["__ts__"].dt.to_period("M").astype(str)

    rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for month in months:
        month_rows, month_selected, month_diag = _score_month(
            frame=frame,
            metrics=metrics,
            month_period=month_period,
            month=month,
            labels=labels,
            risk_heads=risk_heads,
            gates=gates,
            feature_map=feature_map,
            labels_requested=labels_requested,
            feature_sets=feature_sets,
            source_buckets=source_buckets,
            causal_gates=causal_gate_names,
            selections=selections,
            top_fracs=top_fracs,
            seeds=seeds,
            train_lookback_months=train_lookback_months,
            min_train_rows=min_train_rows,
            min_valid_rows=min_valid_rows,
            min_scope_rows=min_scope_rows,
        )
        rows.extend(month_rows)
        selected_frames.extend(month_selected)
        diagnostics.extend(month_diag)

    monthly = _add_utility_only_deltas(pd.DataFrame(rows))
    selected = (
        pd.concat([frame for frame in selected_frames if not frame.empty], ignore_index=True)
        if selected_frames
        else pd.DataFrame()
    )
    weekly = _weekly_summary(selected)
    aggregate = _aggregate(monthly, weekly, selected, expected_months=len(months))
    diagnostics_frame = pd.DataFrame(diagnostics)

    paths = {
        "monthly": output_dir / "source_utility_path_timeout_risk_monthly.csv",
        "weekly": output_dir / "source_utility_path_timeout_risk_weekly.csv",
        "aggregate": output_dir / "source_utility_path_timeout_risk_aggregate.csv",
        "diagnostics": output_dir / "source_utility_path_timeout_risk_diagnostics.csv",
        "selected_rows_parquet": output_dir / "source_utility_path_timeout_risk_selected_rows.parquet",
        "selected_rows_csv": output_dir / "source_utility_path_timeout_risk_selected_rows.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics_frame.to_csv(paths["diagnostics"], index=False)
    selected.to_parquet(paths["selected_rows_parquet"], index=False)
    selected.to_csv(paths["selected_rows_csv"], index=False)
    manifest = {
        "scope": "source_utility_path_timeout_risk_diagnostic",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "utility_source": metrics.attrs.get("utility_source"),
        "months": list(months),
        "labels": labels_requested,
        "risk_heads": risk_head_names,
        "feature_sets": feature_sets,
        "source_buckets": source_buckets,
        "causal_gates": causal_gate_names,
        "selections": selection_names,
        "top_fracs": [float(v) for v in top_fracs],
        "seeds": [int(seed) for seed in seeds],
        "join_report": join_report,
        "feature_store": feature_report,
        "v2_feature_store": v2_report,
        "base_feature_count": int(len(base_features)),
        "source_feature_count": int(len(source_features)),
        "v2_feature_count": int(len(v2_features)),
        "outputs": {key: str(path) for key, path in paths.items()},
        "promotion_gates": {
            "mean_u_gt": 0.0,
            "worst_month_u_gt": 0.0,
            "positive_week_fraction_min": 0.75,
            "q25_week_u_min": 0.0,
            "bad_mae_rate_max": 0.45,
            "bad_mae_negative_rate": "reported_only",
            "bad_mae_recovered_rate": "reported_only",
            "fast_bad_mae_negative_rate": "reported_only",
            "timeout_rate_max": 0.15,
            "wide_barrier_25bps_rate_max": 0.10,
            "overall_top_symbol_share_max": 0.35,
            "utility_without_top_symbol_gt": 0.0,
        },
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_report(output_dir, aggregate, weekly, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--archetypes-v2-path", type=Path, default=DEFAULT_ARCHETYPES_V2)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS))
    parser.add_argument("--risk-heads", type=str, default=",".join(DEFAULT_RISK_HEADS))
    parser.add_argument("--feature-sets", type=str, default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--source-buckets", type=str, default=",".join(DEFAULT_SOURCE_BUCKETS))
    parser.add_argument("--causal-gates", type=str, default=",".join(DEFAULT_CAUSAL_GATES))
    parser.add_argument("--selections", type=str, default=",".join(DEFAULT_SELECTIONS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--min-scope-rows", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostic(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        archetypes_v2_path=args.archetypes_v2_path,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        labels_requested=_parse_csv(args.labels, DEFAULT_LABELS),
        risk_head_names=_parse_csv(args.risk_heads, DEFAULT_RISK_HEADS),
        feature_sets=_parse_csv(args.feature_sets, DEFAULT_FEATURE_SETS),
        source_buckets=_parse_csv(args.source_buckets, DEFAULT_SOURCE_BUCKETS),
        causal_gate_names=_parse_csv(args.causal_gates, DEFAULT_CAUSAL_GATES),
        selection_names=_parse_csv(args.selections, DEFAULT_SELECTIONS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        min_scope_rows=int(args.min_scope_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
