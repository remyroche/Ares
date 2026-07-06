#!/usr/bin/env python3
"""Second-stage causal risk-gate diagnostics for utility-label model scores.

This script tests whether prediction-time source/risk gates can preserve the
positive utility signal found by the utility-first label rework while reducing
bad-MAE, timeout, and wide-barrier exposure.

It is diagnostic-only. Labels still use realized outcomes as supervised targets,
but all gate columns are causal source columns and any percentile gate threshold
is calibrated from prior months only.
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
    LABEL_SPECS,
    UtilityLabelSpec,
    _build_target,
    _safe_numeric,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/utility_risk_gate")
OUTCOME_LIKE_PATTERNS = (
    "future",
    "fwd",
    "mfe",
    "mae",
    "pnl",
    "profit",
    "utility",
    "target",
    "label",
    "oracle",
    "hit",
    "timeout",
    "realized",
    "outcome",
    "barrier_result",
)


@dataclass(frozen=True)
class GateRule:
    col: str
    op: str
    q: float | None = None
    value: float | bool | None = None
    source_conditioned: bool = False
    fallback: float | None = None


@dataclass(frozen=True)
class RiskGateSpec:
    name: str
    rules: tuple[GateRule, ...]


RISK_GATES = (
    RiskGateSpec("no_gate", ()),
    RiskGateSpec("not_dirty_tag", (GateRule("tag_dirty_shock_avoid", "eq", value=False),)),
    RiskGateSpec("clean_execution_tag", (GateRule("tag_clean_execution_context", "eq", value=True),)),
    RiskGateSpec("not_misleading_location_tag", (GateRule("tag_misleading_location_risk", "eq", value=False),)),
    RiskGateSpec("low_dirty_score_q50", (GateRule("dirty_shock_avoid_score", "le", q=0.50),)),
    RiskGateSpec("high_execution_quality_q50", (GateRule("execution_quality_score", "ge", q=0.50),)),
    RiskGateSpec("high_execution_quality_q65", (GateRule("execution_quality_score", "ge", q=0.65),)),
    RiskGateSpec("low_barrier_pressure_q50", (GateRule("barrier_pressure_score", "le", q=0.50),)),
    RiskGateSpec("low_barrier_pressure_q35", (GateRule("barrier_pressure_score", "le", q=0.35),)),
    RiskGateSpec(
        "exec_q50_barrier_q50",
        (
            GateRule("execution_quality_score", "ge", q=0.50),
            GateRule("barrier_pressure_score", "le", q=0.50),
        ),
    ),
    RiskGateSpec(
        "not_dirty_exec_q50_barrier_q50",
        (
            GateRule("tag_dirty_shock_avoid", "eq", value=False),
            GateRule("execution_quality_score", "ge", q=0.50),
            GateRule("barrier_pressure_score", "le", q=0.50),
        ),
    ),
    RiskGateSpec(
        "strict_exec_q65_barrier_q35_dirty_q35",
        (
            GateRule("execution_quality_score", "ge", q=0.65),
            GateRule("barrier_pressure_score", "le", q=0.35),
            GateRule("dirty_shock_avoid_score", "le", q=0.35),
        ),
    ),
    RiskGateSpec(
        "source_exec_q50_barrier_q50",
        (
            GateRule("execution_quality_score", "ge", q=0.50, source_conditioned=True),
            GateRule("barrier_pressure_score", "le", q=0.50, source_conditioned=True),
        ),
    ),
    RiskGateSpec(
        "clean_context_low_barrier",
        (
            GateRule("tag_clean_execution_context", "eq", value=True),
            GateRule("barrier_pressure_score", "le", q=0.50),
        ),
    ),
    RiskGateSpec("high_barrier_relief_q50", (GateRule("barrier_relief_score", "ge", q=0.50),)),
    RiskGateSpec("high_recovery_excess_q50", (GateRule("excess_12h", "ge", q=0.50),)),
    RiskGateSpec("high_recovery_convexity_q50", (GateRule("convexity_t", "ge", q=0.50),)),
    RiskGateSpec("high_recovery_impulse_q50", (GateRule("impulse", "ge", q=0.50),)),
    RiskGateSpec(
        "recovery_excess_convexity_q40",
        (
            GateRule("excess_12h", "ge", q=0.40),
            GateRule("convexity_t", "ge", q=0.40),
        ),
    ),
    RiskGateSpec(
        "recovery_excess_convexity_q50",
        (
            GateRule("excess_12h", "ge", q=0.50),
            GateRule("convexity_t", "ge", q=0.50),
        ),
    ),
    RiskGateSpec(
        "recovery_excess_impulse_q40",
        (
            GateRule("excess_12h", "ge", q=0.40),
            GateRule("impulse", "ge", q=0.40),
        ),
    ),
    RiskGateSpec(
        "recovery_support_q40_barrier_q50",
        (
            GateRule("excess_12h", "ge", q=0.40),
            GateRule("convexity_t", "ge", q=0.40),
            GateRule("barrier_pressure_score", "le", q=0.50),
        ),
    ),
    RiskGateSpec(
        "clean_econ_source_tag",
        (GateRule("tag_clean_economic_capture_candidate", "eq", value=True),),
    ),
)


def _label_specs_by_name(names: list[str]) -> list[UtilityLabelSpec]:
    available = {spec.name: spec for spec in LABEL_SPECS}
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Unknown label spec(s): {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def _gate_specs_by_name(names: list[str]) -> list[RiskGateSpec]:
    available = {spec.name: spec for spec in RISK_GATES}
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Unknown risk gate(s): {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def _assert_gate_columns_causal(gates: list[RiskGateSpec]) -> None:
    offenders: list[str] = []
    for gate in gates:
        for rule in gate.rules:
            lowered = rule.col.lower()
            if any(pattern in lowered for pattern in OUTCOME_LIKE_PATTERNS):
                offenders.append(f"{gate.name}:{rule.col}")
    if offenders:
        raise ValueError(f"Outcome-like columns are not allowed in causal risk gates: {offenders}")


def _bool_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    lowered = values.astype(str).str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _source_groups(frame: pd.DataFrame) -> pd.Series:
    if "primary_source_tag" in frame.columns:
        return frame["primary_source_tag"].fillna("unknown").astype(str)
    return pd.Series("all", index=frame.index, dtype=object)


def _rule_threshold_series(
    *,
    frame: pd.DataFrame,
    train_mask: pd.Series,
    rule: GateRule,
) -> pd.Series:
    if rule.q is None:
        fallback = float(rule.fallback if rule.fallback is not None else 0.0)
        return pd.Series(fallback, index=frame.index, dtype=np.float32)
    values = _safe_numeric(frame[rule.col]).replace([np.inf, -np.inf], np.nan)
    train_values = values.loc[train_mask].dropna()
    fallback = float(train_values.quantile(float(rule.q))) if len(train_values) else float(rule.fallback or 0.5)
    out = pd.Series(fallback, index=frame.index, dtype=np.float32)
    if not rule.source_conditioned:
        return out
    groups = _source_groups(frame)
    for group_value, idx in groups.loc[train_mask].groupby(groups.loc[train_mask], sort=False).groups.items():
        if len(idx) < 200:
            continue
        threshold = float(values.loc[idx].dropna().quantile(float(rule.q)))
        if math.isfinite(threshold):
            out.loc[groups.eq(str(group_value))] = threshold
    return out


def _gate_mask(frame: pd.DataFrame, train_mask: pd.Series, gate: RiskGateSpec) -> tuple[pd.Series, dict[str, Any]]:
    mask = pd.Series(True, index=frame.index)
    thresholds: dict[str, float] = {}
    missing: list[str] = []
    for rule in gate.rules:
        if rule.col not in frame.columns:
            missing.append(rule.col)
            mask &= False
            continue
        if rule.op == "eq":
            expected = bool(rule.value)
            mask &= _bool_series(frame, rule.col).eq(expected)
            thresholds[rule.col] = float(expected)
            continue
        values = _safe_numeric(frame[rule.col])
        threshold = _rule_threshold_series(frame=frame, train_mask=train_mask, rule=rule)
        thresholds[f"{rule.col}_{rule.op}_q{rule.q}"] = float(threshold.loc[train_mask].median()) if train_mask.any() else float("nan")
        if rule.op == "ge":
            mask &= values.ge(threshold)
        elif rule.op == "le":
            mask &= values.le(threshold)
        else:
            raise ValueError(f"Unsupported gate op: {rule.op}")
    return mask.fillna(False), {"missing_gate_columns": missing, "thresholds": thresholds}


def _bucket_masks(frame: pd.DataFrame, valid_mask: pd.Series, min_bucket_rows: int) -> list[tuple[str, pd.Series]]:
    masks: list[tuple[str, pd.Series]] = [("all_rows", valid_mask.copy())]
    groups = _source_groups(frame).loc[valid_mask]
    for source_bucket, idx in groups.groupby(groups, sort=True).groups.items():
        mask = pd.Series(False, index=frame.index)
        mask.loc[idx] = True
        if int(mask.sum()) >= int(min_bucket_rows):
            masks.append((str(source_bucket), valid_mask & mask))
    return masks


def _rank_top_indices(score: pd.Series, k: int) -> np.ndarray:
    score_s = _safe_numeric(score).reset_index(drop=True)
    valid = score_s.notna().to_numpy()
    if not bool(valid.any()):
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = min(max(1, int(k)), len(valid_idx))
    order = np.argsort(-score_s.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False)


def _selection_row(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    context: dict[str, Any],
    top_frac: float,
    selection_mode: str,
    bucket_rows: int,
) -> dict[str, Any]:
    rows = int(len(frame))
    if rows == 0:
        return {
            **context,
            "top_frac": float(top_frac),
            "selection_mode": selection_mode,
            "gate_rows": 0,
            "selected_rows": 0,
            "mean_u": float("nan"),
        }
    if selection_mode == "budget_matched":
        k = max(1, int(math.ceil(float(top_frac) * int(bucket_rows))))
    else:
        k = max(1, int(math.ceil(float(top_frac) * rows)))
    selected_idx = _rank_top_indices(score, k)
    selected_metrics = metrics.iloc[selected_idx].copy() if len(selected_idx) else metrics.iloc[:0].copy()
    selected_target = target.iloc[selected_idx].copy() if len(selected_idx) else target.iloc[:0].copy()
    selected_frame = frame.iloc[selected_idx].copy() if len(selected_idx) else frame.iloc[:0].copy()
    top_symbol_share = (
        float(selected_frame["__symbol__"].value_counts(normalize=True).iloc[0])
        if "__symbol__" in selected_frame.columns and len(selected_frame)
        else float("nan")
    )
    return {
        **context,
        "top_frac": float(top_frac),
        "selection_mode": selection_mode,
        "bucket_rows": int(bucket_rows),
        "gate_rows": rows,
        "gate_coverage_vs_bucket": float(rows / bucket_rows) if bucket_rows else 0.0,
        "selected_rows": int(len(selected_metrics)),
        "selected_coverage_vs_bucket": float(len(selected_metrics) / bucket_rows) if bucket_rows else 0.0,
        "selected_coverage_vs_gate": float(len(selected_metrics) / rows) if rows else 0.0,
        "target_top_soft_mean": _safe_mean(selected_target["target_soft"]) if len(selected_target) else float("nan"),
        "target_top_hard_rate": _safe_mean(selected_target["target_hard"] > 0.5) if len(selected_target) else float("nan"),
        "mean_u": _safe_mean(selected_metrics["u_policy_net"]),
        "median_u": _safe_quantile(selected_metrics["u_policy_net"], 0.50),
        "q10_u": _safe_quantile(selected_metrics["u_policy_net"], 0.10),
        "hit_u": _safe_mean(selected_metrics["u_policy_net"] > 0.0),
        "mean_barrier": _safe_mean(selected_metrics["barrier"]),
        "p90_barrier": _safe_quantile(selected_metrics["barrier"], 0.90),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics["barrier"] > 0.025),
        "wide_barrier_35bps_rate": _safe_mean(selected_metrics["barrier"] > 0.035),
        "mean_mae_norm": _safe_mean(selected_metrics["mae_norm"]),
        "p90_mae_norm": _safe_quantile(selected_metrics["mae_norm"], 0.90),
        "bad_mae_1r_rate": _safe_mean(selected_metrics["mae_norm"] >= 1.0),
        "timeout_rate": _safe_mean(selected_metrics["is_timeout"].astype(float)),
        "score_ic_label_gate": _spearman(score, target["target_soft"]),
        "score_ic_u_gate": _spearman(score, metrics["u_policy_net"]),
        "target_ic_u_gate": _spearman(target["target_soft"], metrics["u_policy_net"]),
        "top_symbol_share": top_symbol_share,
    }


def _score_gates_for_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    label_name: str,
    feature_set: str,
    gates: list[RiskGateSpec],
    month: str,
    top_fracs: list[float],
    min_bucket_rows: int,
    min_gate_rows: int,
    train_rows: int,
    model_feature_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    gate_cache = {gate.name: _gate_mask(frame, train_mask, gate) for gate in gates}
    for source_bucket, bucket_mask in _bucket_masks(frame, valid_mask, min_bucket_rows):
        bucket_rows = int(bucket_mask.sum())
        for gate in gates:
            gate_mask, gate_report = gate_cache[gate.name]
            scope_mask = bucket_mask & gate_mask
            gate_rows = int(scope_mask.sum())
            diag_rows.append(
                {
                    "period": month,
                    "label": label_name,
                    "feature_set": feature_set,
                    "source_bucket": source_bucket,
                    "risk_gate": gate.name,
                    "bucket_rows": bucket_rows,
                    "gate_rows": gate_rows,
                    "gate_coverage_vs_bucket": float(gate_rows / bucket_rows) if bucket_rows else 0.0,
                    "missing_gate_columns": ",".join(gate_report.get("missing_gate_columns", [])),
                    "gate_thresholds_json": json.dumps(_json_safe(gate_report.get("thresholds", {})), sort_keys=True),
                }
            )
            if gate_rows < int(min_gate_rows):
                continue
            gate_frame = frame.loc[scope_mask].reset_index(drop=True)
            gate_metrics = metrics.loc[scope_mask].reset_index(drop=True)
            gate_target = target.loc[scope_mask].reset_index(drop=True)
            gate_score = score.loc[scope_mask].reset_index(drop=True)
            context = {
                "period": month,
                "label": label_name,
                "feature_set": feature_set,
                "source_bucket": source_bucket,
                "risk_gate": gate.name,
                "train_rows": int(train_rows),
                "model_feature_count": int(model_feature_count),
            }
            for top_frac in top_fracs:
                for mode in ("gate_relative", "budget_matched"):
                    rows.append(
                        _selection_row(
                            frame=gate_frame,
                            metrics=gate_metrics,
                            target=gate_target,
                            score=gate_score,
                            context=context,
                            top_frac=top_frac,
                            selection_mode=mode,
                            bucket_rows=bucket_rows,
                        )
                    )
    return rows, diag_rows


def _add_no_gate_deltas(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    keys = ["period", "label", "feature_set", "source_bucket", "top_frac", "selection_mode"]
    baseline_cols = [
        "gate_rows",
        "selected_rows",
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "score_ic_u_gate",
    ]
    base = monthly[monthly["risk_gate"].eq("no_gate")][keys + baseline_cols].rename(
        columns={col: f"no_gate_{col}" for col in baseline_cols}
    )
    out = monthly.merge(base, on=keys, how="left", validate="many_to_one")
    for col in baseline_cols:
        out[f"delta_{col}_vs_no_gate"] = _safe_numeric(out[col]) - _safe_numeric(out[f"no_gate_{col}"])
    return out


def _aggregate(monthly: pd.DataFrame, *, expected_months: int) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["label", "feature_set", "source_bucket", "risk_gate", "top_frac", "selection_mode"]
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        label, feature_set, source_bucket, risk_gate, top_frac, selection_mode = key
        mean_u = _safe_mean(group["mean_u"])
        worst_u = _safe_quantile(group["mean_u"], 0.0)
        bad_mae = _safe_mean(group["bad_mae_1r_rate"])
        timeout = _safe_mean(group["timeout_rate"])
        wide = _safe_mean(group["wide_barrier_25bps_rate"])
        min_selected = _safe_quantile(group["selected_rows"], 0.0)
        positive_months = int((_safe_numeric(group["mean_u"]) > 0.0).sum())
        ic_positive_months = int((_safe_numeric(group["score_ic_u_gate"]) > 0.0).sum())
        month_count = int(group["period"].nunique())
        risk_ok = (
            math.isfinite(bad_mae)
            and bad_mae <= 0.65
            and math.isfinite(timeout)
            and timeout <= 0.20
            and math.isfinite(wide)
            and wide <= 0.25
        )
        size_ok = math.isfinite(min_selected) and min_selected >= 25.0
        positive_ok = (
            month_count >= int(expected_months)
            and positive_months >= int(expected_months)
            and math.isfinite(mean_u)
            and mean_u > 0.0
            and math.isfinite(worst_u)
            and worst_u > 0.0
        )
        ic_ok = ic_positive_months >= int(expected_months)
        if positive_ok and ic_ok and risk_ok and size_ok:
            decision = "candidate_gate_within_economic_limits"
        elif positive_ok and ic_ok and risk_ok and not size_ok:
            decision = "narrow_gate_only"
        elif positive_ok and ic_ok and not risk_ok:
            decision = "positive_but_risk_limits_fail"
        elif risk_gate != "no_gate" and math.isfinite(_safe_mean(group.get("delta_bad_mae_1r_rate_vs_no_gate"))) and _safe_mean(group.get("delta_bad_mae_1r_rate_vs_no_gate")) < -0.05:
            decision = "risk_reducer_not_profitable"
        else:
            decision = "diagnostic_only"
        rows.append(
            {
                "decision": decision,
                "label": label,
                "feature_set": feature_set,
                "source_bucket": source_bucket,
                "risk_gate": risk_gate,
                "top_frac": float(top_frac),
                "selection_mode": selection_mode,
                "months": month_count,
                "positive_months": positive_months,
                "ic_u_positive_months": ic_positive_months,
                "mean_u": mean_u,
                "worst_month_u": worst_u,
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "bad_mae_1r_rate": bad_mae,
                "p90_mae_norm": _safe_mean(group["p90_mae_norm"]),
                "timeout_rate": timeout,
                "wide_barrier_25bps_rate": wide,
                "mean_gate_coverage_vs_bucket": _safe_mean(group["gate_coverage_vs_bucket"]),
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": min_selected,
                "score_ic_u_gate": _safe_mean(group["score_ic_u_gate"]),
                "score_ic_label_gate": _safe_mean(group["score_ic_label_gate"]),
                "delta_mean_u_vs_no_gate": _safe_mean(group.get("delta_mean_u_vs_no_gate")),
                "delta_bad_mae_1r_rate_vs_no_gate": _safe_mean(group.get("delta_bad_mae_1r_rate_vs_no_gate")),
                "delta_timeout_rate_vs_no_gate": _safe_mean(group.get("delta_timeout_rate_vs_no_gate")),
                "delta_wide_barrier_25bps_rate_vs_no_gate": _safe_mean(
                    group.get("delta_wide_barrier_25bps_rate_vs_no_gate")
                ),
                "delta_selected_rows_vs_no_gate": _safe_mean(group.get("delta_selected_rows_vs_no_gate")),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["decision", "top_frac", "mean_u", "delta_bad_mae_1r_rate_vs_no_gate"],
        ascending=[True, True, False, True],
        na_position="last",
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


def _write_report(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "source_utility_risk_gate_diagnostic.md"
    cols = [
        "decision",
        "label",
        "feature_set",
        "source_bucket",
        "risk_gate",
        "top_frac",
        "selection_mode",
        "months",
        "positive_months",
        "ic_u_positive_months",
        "mean_u",
        "worst_month_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "mean_gate_coverage_vs_bucket",
        "mean_selected_rows",
        "min_selected_rows",
        "delta_mean_u_vs_no_gate",
        "delta_bad_mae_1r_rate_vs_no_gate",
        "delta_timeout_rate_vs_no_gate",
        "delta_wide_barrier_25bps_rate_vs_no_gate",
    ]
    candidates = aggregate[aggregate["decision"].eq("candidate_gate_within_economic_limits")]
    narrow = aggregate[aggregate["decision"].eq("narrow_gate_only")]
    risk_fail = aggregate[aggregate["decision"].eq("positive_but_risk_limits_fail")]
    reducers = aggregate[aggregate["decision"].eq("risk_reducer_not_profitable")]
    top10 = aggregate[aggregate["top_frac"].eq(0.10)].copy()
    lines = [
        "# Source Utility Risk Gate Diagnostic",
        "",
        "Scope: second-stage causal gates on utility-label model scores. Gate thresholds use prior months only.",
        "",
        f"Rows joined to label ledger: `{manifest['rows']}`",
        f"Utility source: `{manifest.get('utility_source', '')}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Feature sets: `{', '.join(manifest['feature_sets'])}`",
        f"Risk gates: `{', '.join(manifest['risk_gates'])}`",
        "",
        "## Candidates Within Economic Limits",
        "",
        _table(candidates, cols, limit=80),
        "",
        "## Narrow Gate Leads",
        "",
        _table(narrow, cols, limit=80),
        "",
        "## Positive But Risk Limits Fail",
        "",
        _table(risk_fail, cols, limit=80),
        "",
        "## Risk Reducers Not Profitable",
        "",
        _table(reducers.sort_values("delta_bad_mae_1r_rate_vs_no_gate"), cols, limit=80),
        "",
        "## Top 10% Gate-Relative View",
        "",
        _table(
            top10[top10["selection_mode"].eq("gate_relative")].sort_values(
                ["mean_u", "bad_mae_1r_rate"], ascending=[False, True]
            ),
            cols,
            limit=120,
        ),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Gate diagnostics: `{manifest['outputs']['gate_diagnostics']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostic(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    top_fracs: list[float],
    seeds: list[int],
    feature_sets: list[str],
    label_names: list[str],
    risk_gate_names: list[str],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
    min_bucket_rows: int,
    min_gate_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _label_specs_by_name(label_names)
    gates = _gate_specs_by_name(risk_gate_names)
    _assert_gate_columns_causal(gates)

    frame, join_report = _load_joined_frame(quality_labels_path=quality_labels_path, labels_path=labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    base_features = list(feature_matrix.columns)
    source_features = _source_feature_columns(frame)
    feature_map = {"base": base_features, "base_plus_source": list(dict.fromkeys(base_features + source_features))}

    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    gate_diag_rows: list[dict[str, Any]] = []
    for month in months:
        valid_mask = month_period.eq(month)
        train_mask = month_period < month
        if train_lookback_months is not None and int(train_lookback_months) > 0:
            prior_months = sorted(month_period[train_mask].dropna().unique())
            keep = set(prior_months[-int(train_lookback_months) :])
            train_mask = train_mask & month_period.isin(keep)
        if int(valid_mask.sum()) < int(min_valid_rows):
            continue
        for label_spec in labels:
            target, weights, _label_report = _build_target(
                frame=frame,
                metrics=metrics,
                train_mask=train_mask,
                valid_mask=valid_mask,
                spec=label_spec,
            )
            train_target_mask = train_mask & target["target_soft"].notna() & weights.gt(0.0)
            if int(train_target_mask.sum()) < int(min_train_rows):
                continue
            for feature_set in feature_sets:
                features = feature_map.get(feature_set)
                if not features:
                    continue
                x_train, x_valid = _month_model_frame(
                    frame,
                    train_mask=train_target_mask,
                    valid_mask=valid_mask,
                    features=features,
                )
                pred_matrix = np.vstack(
                    [
                        _fit_predict(
                            x_train=x_train,
                            y_train=target.loc[train_target_mask, "target_soft"],
                            w_train=weights.loc[train_target_mask],
                            x_valid=x_valid,
                            seed=seed,
                        )
                        for seed in seeds
                    ]
                )
                score = pd.Series(np.nan, index=frame.index, dtype=np.float32)
                score.loc[valid_mask] = np.mean(pred_matrix, axis=0).astype(np.float32)
                month_rows, month_diag = _score_gates_for_month(
                    frame=frame,
                    metrics=metrics,
                    target=target,
                    score=score,
                    train_mask=train_mask,
                    valid_mask=valid_mask,
                    label_name=label_spec.name,
                    feature_set=feature_set,
                    gates=gates,
                    month=month,
                    top_fracs=top_fracs,
                    min_bucket_rows=min_bucket_rows,
                    min_gate_rows=min_gate_rows,
                    train_rows=int(train_target_mask.sum()),
                    model_feature_count=int(len(features)),
                )
                rows.extend(month_rows)
                gate_diag_rows.extend(month_diag)

    monthly = _add_no_gate_deltas(pd.DataFrame(rows))
    aggregate = _aggregate(monthly, expected_months=len(months))
    gate_diagnostics = pd.DataFrame(gate_diag_rows)
    paths = {
        "monthly": output_dir / "source_utility_risk_gate_monthly.csv",
        "aggregate": output_dir / "source_utility_risk_gate_aggregate.csv",
        "gate_diagnostics": output_dir / "source_utility_risk_gate_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    gate_diagnostics.to_csv(paths["gate_diagnostics"], index=False)
    manifest = {
        "scope": "source_utility_risk_gate_diagnostic",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "utility_source": metrics.attrs.get("utility_source"),
        "months": list(months),
        "top_fracs": [float(v) for v in top_fracs],
        "seeds": [int(seed) for seed in seeds],
        "feature_sets": list(feature_sets),
        "labels": [spec.name for spec in labels],
        "risk_gates": [gate.name for gate in gates],
        "join_report": join_report,
        "feature_store": feature_report,
        "base_feature_count": int(len(base_features)),
        "source_feature_count": int(len(source_features)),
        "min_bucket_rows": int(min_bucket_rows),
        "min_gate_rows": int(min_gate_rows),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_report(output_dir, aggregate, manifest)
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
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--feature-sets", type=str, default="base,base_plus_source")
    parser.add_argument("--labels", type=str, default=",".join(spec.name for spec in LABEL_SPECS))
    parser.add_argument("--risk-gates", type=str, default=",".join(gate.name for gate in RISK_GATES))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--min-bucket-rows", type=int, default=80)
    parser.add_argument("--min-gate-rows", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostic(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        feature_sets=_parse_csv(args.feature_sets, ("base", "base_plus_source")),
        label_names=_parse_csv(args.labels, tuple(spec.name for spec in LABEL_SPECS)),
        risk_gate_names=_parse_csv(args.risk_gates, tuple(gate.name for gate in RISK_GATES)),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        min_bucket_rows=int(args.min_bucket_rows),
        min_gate_rows=int(args.min_gate_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
