#!/usr/bin/env python3
"""Attribute near-miss failures in the source utility/path-time risk screen.

This diagnostic is read-only. It consumes the artifacts written by
``run_source_utility_path_timeout_risk_diagnostic.py`` and explains why
candidate selectors fail promotion gates by decomposing selected rows across
month/week, source tags, source archetypes, symbols, and realized risk modes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/utility_path_timeout_risk"
)
DEFAULT_ARCHETYPES_V2_PATH = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/source_archetypes_v2/"
    "candidate_source_archetypes_v2.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "utility_path_timeout_failure_attribution"
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.mean()) if len(series) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.quantile(q)) if len(series) else float("nan")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _candidate_key_parts(frame: pd.DataFrame) -> pd.Series:
    top = frame["top_frac"].map(lambda value: f"top{float(value):g}")
    return (
        frame["label"].astype(str)
        + "__"
        + frame["risk_heads"].astype(str)
        + "__"
        + frame["feature_set"].astype(str)
        + "__"
        + frame["source_bucket"].astype(str)
        + "__"
        + frame["causal_gate"].astype(str)
        + "__"
        + frame["selection"].astype(str)
        + "__"
        + top
    )


def _load_manifest(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _promotion_gates(manifest: dict[str, Any]) -> dict[str, float]:
    gates = manifest.get("promotion_gates") if isinstance(manifest, dict) else None
    if isinstance(gates, dict):
        return {
            "mean_u_gt": float(gates.get("mean_u_gt", 0.0)),
            "worst_month_u_gt": float(gates.get("worst_month_u_gt", 0.0)),
            "positive_week_fraction_min": float(gates.get("positive_week_fraction_min", 0.75)),
            "q25_week_u_min": float(gates.get("q25_week_u_min", 0.0)),
            "bad_mae_rate_max": float(gates.get("bad_mae_rate_max", 0.45)),
            "timeout_rate_max": float(gates.get("timeout_rate_max", 0.15)),
            "wide_barrier_25bps_rate_max": float(gates.get("wide_barrier_25bps_rate_max", 0.10)),
            "overall_top_symbol_share_max": float(gates.get("overall_top_symbol_share_max", 0.35)),
            "utility_without_top_symbol_gt": float(gates.get("utility_without_top_symbol_gt", 0.0)),
        }
    return {
        "mean_u_gt": 0.0,
        "worst_month_u_gt": 0.0,
        "positive_week_fraction_min": 0.75,
        "q25_week_u_min": 0.0,
        "bad_mae_rate_max": 0.45,
        "timeout_rate_max": 0.15,
        "wide_barrier_25bps_rate_max": 0.10,
        "overall_top_symbol_share_max": 0.35,
        "utility_without_top_symbol_gt": 0.0,
    }


def _add_failure_flags(aggregate: pd.DataFrame, gates: dict[str, float]) -> pd.DataFrame:
    out = aggregate.copy()
    out["candidate"] = _candidate_key_parts(out)
    out["gate_mean_u_pass"] = _safe_numeric(out["mean_u"]) > gates["mean_u_gt"]
    out["gate_worst_month_pass"] = _safe_numeric(out["worst_month_u"]) > gates["worst_month_u_gt"]
    out["gate_week_fraction_pass"] = (
        _safe_numeric(out["positive_weeks"])
        >= gates["positive_week_fraction_min"] * _safe_numeric(out["weeks"])
    )
    out["gate_q25_week_pass"] = _safe_numeric(out["q25_week_u"]) >= gates["q25_week_u_min"]
    out["gate_bad_mae_pass"] = _safe_numeric(out["bad_mae_1r_rate"]) <= gates["bad_mae_rate_max"]
    out["gate_timeout_pass"] = _safe_numeric(out["timeout_rate"]) <= gates["timeout_rate_max"]
    out["gate_wide_barrier_pass"] = (
        _safe_numeric(out["wide_barrier_25bps_rate"]) <= gates["wide_barrier_25bps_rate_max"]
    )
    out["gate_symbol_pass"] = (
        _safe_numeric(out["overall_top_symbol_share"]) < gates["overall_top_symbol_share_max"]
    )
    out["gate_utility_without_top_symbol_pass"] = (
        _safe_numeric(out["utility_without_top_symbol"]) > gates["utility_without_top_symbol_gt"]
    )
    gate_cols = [c for c in out.columns if c.startswith("gate_") and c.endswith("_pass")]
    out["failed_gate_count"] = (~out[gate_cols].fillna(False)).sum(axis=1).astype(int)
    out["failed_gates"] = out[gate_cols].apply(
        lambda row: ",".join(
            c.removeprefix("gate_").removesuffix("_pass") for c, ok in row.items() if not bool(ok)
        ),
        axis=1,
    )
    out["risk_fail_count"] = (
        (~out["gate_bad_mae_pass"]).astype(int)
        + (~out["gate_timeout_pass"]).astype(int)
        + (~out["gate_wide_barrier_pass"]).astype(int)
    )
    out["path_fail_count"] = (
        (~out["gate_mean_u_pass"]).astype(int)
        + (~out["gate_worst_month_pass"]).astype(int)
        + (~out["gate_week_fraction_pass"]).astype(int)
        + (~out["gate_q25_week_pass"]).astype(int)
        + (~out["gate_utility_without_top_symbol_pass"]).astype(int)
    )
    out["path_ok"] = out[
        [
            "gate_mean_u_pass",
            "gate_worst_month_pass",
            "gate_week_fraction_pass",
            "gate_q25_week_pass",
            "gate_symbol_pass",
            "gate_utility_without_top_symbol_pass",
        ]
    ].all(axis=1)
    out["risk_ok"] = out[["gate_bad_mae_pass", "gate_timeout_pass", "gate_wide_barrier_pass"]].all(axis=1)
    out["all_gates_pass"] = out[gate_cols].all(axis=1)
    out["failure_profile"] = out.apply(_failure_profile, axis=1)
    out["near_miss_reason"] = out.apply(_near_miss_reason, axis=1)
    return out.sort_values(
        ["all_gates_pass", "path_ok", "risk_ok", "mean_u"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )


def _failure_profile(row: pd.Series) -> str:
    if bool(row.get("all_gates_pass", False)):
        return "passes_all_gates"
    path_ok = bool(row.get("path_ok", False))
    risk_ok = bool(row.get("risk_ok", False))
    if path_ok and not risk_ok:
        return "path_ok_risk_fail"
    if risk_ok and not path_ok:
        return "risk_ok_path_fail"
    if float(row.get("mean_u", np.nan)) > 0.0 and float(row.get("worst_month_u", np.nan)) > 0.0:
        return "monthly_positive_but_week_or_risk_fail"
    if float(row.get("mean_u", np.nan)) > 0.0:
        return "mean_positive_but_unstable"
    return "diagnostic_negative_or_flat"


def _near_miss_reason(row: pd.Series) -> str:
    failed = str(row.get("failed_gates", ""))
    if not failed:
        return "none"
    parts = failed.split(",")
    if parts == ["bad_mae"]:
        return "only_bad_mae"
    if parts == ["timeout"]:
        return "only_timeout"
    if parts == ["wide_barrier"]:
        return "only_wide_barrier"
    if set(parts).issubset({"bad_mae", "timeout", "wide_barrier"}):
        return "risk_only"
    if set(parts).issubset({"week_fraction", "q25_week", "worst_month", "mean_u"}):
        return "path_only"
    if "q25_week" in parts or "week_fraction" in parts:
        return "weekly_path_plus_other"
    return "mixed"


def _risk_mode(frame: pd.DataFrame) -> pd.Series:
    bad = _safe_numeric(frame.get("bad_mae_risk_v1_target_hard", frame.get("mae_norm", 0))).ge(1.0)
    timeout = frame.get("is_timeout", False)
    if isinstance(timeout, pd.Series):
        timeout = timeout.astype(bool)
    else:
        timeout = pd.Series(bool(timeout), index=frame.index)
    wide25 = _safe_numeric(frame.get("barrier", 0.0)).gt(0.025)
    labels = []
    for b, t, w in zip(bad, timeout, wide25, strict=False):
        parts: list[str] = []
        if bool(b):
            parts.append("bad_mae")
        if bool(t):
            parts.append("timeout")
        if bool(w):
            parts.append("wide25")
        labels.append("+".join(parts) if parts else "clean")
    return pd.Series(labels, index=frame.index)


def _bad_mae_failure_bucket(frame: pd.DataFrame) -> pd.Series:
    bad = _safe_numeric(frame.get("mae_norm", np.nan)).ge(1.0)
    utility = _safe_numeric(frame.get("u_policy_net", np.nan))
    loss = utility.le(0.0)
    recovered = utility.gt(0.0)
    timeout = frame.get("is_timeout", False)
    if isinstance(timeout, pd.Series):
        timeout = timeout.astype(bool)
    else:
        timeout = pd.Series(bool(timeout), index=frame.index)
    wide25 = _safe_numeric(frame.get("barrier", np.nan)).gt(0.025)
    bars = _safe_numeric(frame.get("bars_policy", np.nan)).fillna(24.0)
    fast = bars.le(4.0)
    late = bars.ge(16.0) | timeout
    labels: list[str] = []
    for is_bad, is_loss, is_recovered, is_timeout, is_wide, is_fast, is_late in zip(
        bad, loss, recovered, timeout, wide25, fast, late, strict=False
    ):
        if not bool(is_bad):
            labels.append("not_bad_mae")
        elif bool(is_loss) and bool(is_timeout):
            labels.append("bad_mae_negative_timeout")
        elif bool(is_recovered) and bool(is_timeout):
            labels.append("bad_mae_recovered_timeout")
        elif bool(is_loss) and bool(is_wide):
            labels.append("bad_mae_negative_wide25")
        elif bool(is_recovered) and bool(is_wide):
            labels.append("bad_mae_recovered_wide25")
        elif bool(is_loss) and bool(is_fast):
            labels.append("fast_bad_mae_negative")
        elif bool(is_recovered) and bool(is_fast):
            labels.append("fast_bad_mae_recovered")
        elif bool(is_loss) and bool(is_late):
            labels.append("late_bad_mae_negative")
        elif bool(is_recovered) and bool(is_late):
            labels.append("late_bad_mae_recovered")
        elif bool(is_loss):
            labels.append("bad_mae_negative")
        else:
            labels.append("bad_mae_recovered")
    return pd.Series(labels, index=frame.index)


def _enrich_selected(selected: pd.DataFrame, archetypes_v2_path: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = selected.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["row_loss"] = _safe_numeric(out["u_policy_net"]).le(0.0)
    out["row_positive_utility"] = _safe_numeric(out["u_policy_net"]).gt(0.0)
    out["row_bad_mae_1r"] = _safe_numeric(out.get("mae_norm", np.nan)).ge(1.0)
    out["row_wide25"] = _safe_numeric(out.get("barrier", np.nan)).gt(0.025)
    out["row_wide35"] = _safe_numeric(out.get("barrier", np.nan)).gt(0.035)
    out["row_timeout"] = out.get("is_timeout", False)
    if isinstance(out["row_timeout"], pd.Series):
        out["row_timeout"] = out["row_timeout"].astype(bool)
    bars = _safe_numeric(out.get("bars_policy", np.nan)).fillna(24.0)
    out["row_fast_bad_mae"] = out["row_bad_mae_1r"] & bars.le(4.0)
    out["row_late_bad_mae"] = out["row_bad_mae_1r"] & (bars.ge(16.0) | out["row_timeout"])
    out["row_bad_mae_negative_utility"] = out["row_bad_mae_1r"] & out["row_loss"]
    out["row_bad_mae_recovered"] = out["row_bad_mae_1r"] & out["row_positive_utility"]
    out["risk_mode"] = _risk_mode(out)
    out["bad_mae_failure_bucket"] = _bad_mae_failure_bucket(out)
    report: dict[str, Any] = {
        "selected_rows": int(len(out)),
        "archetypes_v2_joined": False,
        "archetypes_v2_duplicate_keys": None,
        "missing_archetype_v2_rows": None,
    }
    if archetypes_v2_path and archetypes_v2_path.exists():
        cols = [
            "__ts__",
            "__symbol__",
            "primary_source_archetype_v2",
            "source_evidence_archetype_score",
            "path_geometry_archetype_score",
            "timeout_holding_archetype_score",
            "source_freshness_archetype_score",
            "source_independence_archetype_score",
            "symbol_behavior_archetype_score",
        ]
        v2 = pd.read_parquet(archetypes_v2_path, columns=[c for c in cols if c])
        v2["__ts__"] = pd.to_datetime(v2["__ts__"], utc=True, errors="coerce")
        dupes = int(v2.duplicated(["__ts__", "__symbol__"]).sum())
        report["archetypes_v2_duplicate_keys"] = dupes
        if dupes:
            v2 = v2.drop_duplicates(["__ts__", "__symbol__"], keep="first")
        out = out.merge(v2, on=["__ts__", "__symbol__"], how="left", validate="many_to_one")
        report["archetypes_v2_joined"] = True
        report["missing_archetype_v2_rows"] = int(out["primary_source_archetype_v2"].isna().sum())
    else:
        out["primary_source_archetype_v2"] = "missing_v2"
    out["primary_source_archetype_v2"] = out["primary_source_archetype_v2"].fillna("missing_v2")
    return out, report


def _selected_summary(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
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
            "loss_share": float("nan"),
            "bad_mae_negative_utility_rate": float("nan"),
            "bad_mae_recovered_rate": float("nan"),
            "fast_bad_mae_rate": float("nan"),
            "late_bad_mae_rate": float("nan"),
            "risk_event_share": float("nan"),
        }
    risk_event = group["row_bad_mae_1r"] | group["row_timeout"] | group["row_wide25"]
    return {
        "rows": int(len(group)),
        "mean_u": _safe_mean(group["u_policy_net"]),
        "median_u": _safe_quantile(group["u_policy_net"], 0.50),
        "q10_u": _safe_quantile(group["u_policy_net"], 0.10),
        "hit_u": _safe_mean(_safe_numeric(group["u_policy_net"]).gt(0.0).astype(float)),
        "bad_mae_1r_rate": _safe_mean(group["row_bad_mae_1r"].astype(float)),
        "p90_mae_norm": _safe_quantile(group["mae_norm"], 0.90),
        "timeout_rate": _safe_mean(group["row_timeout"].astype(float)),
        "wide_barrier_25bps_rate": _safe_mean(group["row_wide25"].astype(float)),
        "wide_barrier_35bps_rate": _safe_mean(group["row_wide35"].astype(float)),
        "mean_bars_policy": _safe_mean(group["bars_policy"]),
        "loss_share": _safe_mean(group["row_loss"].astype(float)),
        "bad_mae_negative_utility_rate": _safe_mean(group["row_bad_mae_negative_utility"].astype(float)),
        "bad_mae_recovered_rate": _safe_mean(group["row_bad_mae_recovered"].astype(float)),
        "fast_bad_mae_rate": _safe_mean(group["row_fast_bad_mae"].astype(float)),
        "late_bad_mae_rate": _safe_mean(group["row_late_bad_mae"].astype(float)),
        "risk_event_share": _safe_mean(risk_event.astype(float)),
    }


def _group_summary(selected: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if selected.empty:
        return pd.DataFrame()
    for key, group in selected.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        context = dict(zip(group_cols, key, strict=False))
        top_symbol = ""
        top_symbol_share = float("nan")
        if "__symbol__" in group.columns and len(group):
            vc = group["__symbol__"].value_counts(normalize=True)
            top_symbol = str(vc.index[0])
            top_symbol_share = float(vc.iloc[0])
        rows.append(
            {
                **context,
                **_selected_summary(group),
                "unique_symbols": int(group["__symbol__"].nunique()) if "__symbol__" in group.columns else 0,
                "top_symbol": top_symbol,
                "top_symbol_share": top_symbol_share,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["mean_u", "rows"], ascending=[True, False], kind="mergesort")


def _selector_tradeoffs(aggregate: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["label", "risk_heads", "feature_set", "source_bucket", "causal_gate", "top_frac"]
    utility = aggregate[aggregate["selection"].eq("utility_only")].copy()
    utility_cols = key_cols + [
        "candidate",
        "mean_u",
        "worst_month_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "positive_weeks",
        "weeks",
        "q25_week_u",
    ]
    utility = utility[utility_cols].rename(
        columns={
            "candidate": "utility_only_candidate",
            "mean_u": "utility_only_mean_u",
            "worst_month_u": "utility_only_worst_month_u",
            "bad_mae_1r_rate": "utility_only_bad_mae_1r_rate",
            "timeout_rate": "utility_only_timeout_rate",
            "wide_barrier_25bps_rate": "utility_only_wide_barrier_25bps_rate",
            "positive_weeks": "utility_only_positive_weeks",
            "weeks": "utility_only_weeks",
            "q25_week_u": "utility_only_q25_week_u",
        }
    )
    out = aggregate.merge(utility, on=key_cols, how="left", validate="many_to_one")
    out = out[out["selection"].ne("utility_only")].copy()
    for col in ["mean_u", "worst_month_u", "bad_mae_1r_rate", "timeout_rate", "wide_barrier_25bps_rate", "q25_week_u"]:
        out[f"delta_{col}_vs_utility_only_local"] = (
            _safe_numeric(out[col]) - _safe_numeric(out[f"utility_only_{col}"])
        )
    out["delta_positive_week_fraction_vs_utility_only_local"] = (
        _safe_numeric(out["positive_weeks"]) / _safe_numeric(out["weeks"]).replace(0, np.nan)
        - _safe_numeric(out["utility_only_positive_weeks"])
        / _safe_numeric(out["utility_only_weeks"]).replace(0, np.nan)
    )
    cols = [
        "failure_profile",
        "near_miss_reason",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "selection",
        "top_frac",
        "mean_u",
        "delta_mean_u_vs_utility_only_local",
        "bad_mae_1r_rate",
        "delta_bad_mae_1r_rate_vs_utility_only_local",
        "timeout_rate",
        "delta_timeout_rate_vs_utility_only_local",
        "wide_barrier_25bps_rate",
        "delta_wide_barrier_25bps_rate_vs_utility_only_local",
        "q25_week_u",
        "delta_q25_week_u_vs_utility_only_local",
        "delta_positive_week_fraction_vs_utility_only_local",
    ]
    return out[cols].sort_values(["mean_u"], ascending=False, kind="mergesort")


def _write_report(
    output_dir: Path,
    aggregate: pd.DataFrame,
    near_misses: pd.DataFrame,
    month_week_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    archetype_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    bad_mae_bucket_summary: pd.DataFrame,
    bad_mae_bucket_by_selector: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    selector_tradeoffs: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    report = output_dir / "source_utility_path_timeout_failure_attribution_report.md"
    lines: list[str] = []
    lines.append("# Source Utility Path-Time Failure Attribution")
    lines.append("")
    lines.append("Diagnostic-only report over the latest selected-row utility/path-time risk screen.")
    lines.append("")
    lines.append(f"Input directory: `{manifest.get('input_dir')}`")
    lines.append(f"Aggregate rows: `{len(aggregate)}`")
    lines.append(f"Selected rows: `{manifest.get('selected_rows')}`")
    lines.append(f"All-gate candidates: `{int(aggregate['all_gates_pass'].sum())}`")
    lines.append("")
    lines.append("## Failure Profile Counts")
    lines.append("")
    lines.append(_table(aggregate["failure_profile"].value_counts().rename_axis("failure_profile").reset_index(name="rows")))
    lines.append("")
    lines.append("## Gate Failure Counts")
    gate_cols = [c for c in aggregate.columns if c.startswith("gate_") and c.endswith("_pass")]
    gate_counts = pd.DataFrame(
        {
            "gate": [c.removeprefix("gate_").removesuffix("_pass") for c in gate_cols],
            "failures": [int((~aggregate[c].fillna(False)).sum()) for c in gate_cols],
        }
    ).sort_values("failures", ascending=False)
    lines.append(_table(gate_counts))
    lines.append("")
    lines.append("## Highest Utility Near-Misses")
    cols = [
        "failure_profile",
        "near_miss_reason",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "selection",
        "top_frac",
        "mean_u",
        "worst_month_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "positive_weeks",
        "weeks",
        "q25_week_u",
        "failed_gates",
    ]
    lines.append(_table(near_misses.sort_values("mean_u", ascending=False)[cols].head(20)))
    lines.append("")
    lines.append("## Worst Month/Week Near-Miss Buckets")
    if not month_week_summary.empty:
        week_cols = [
            "period",
            "week_start",
            "failure_profile",
            "selection",
            "feature_set",
            "source_bucket",
            "causal_gate",
            "rows",
            "mean_u",
            "bad_mae_1r_rate",
            "timeout_rate",
            "wide_barrier_25bps_rate",
            "top_symbol",
            "top_symbol_share",
        ]
        lines.append(_table(month_week_summary[week_cols].head(25)))
    else:
        lines.append("No month/week near-miss rows.")
    lines.append("")
    lines.append("## Worst Source Tags In Near-Misses")
    if not source_summary.empty:
        lines.append(
            _table(
                source_summary[
                    [
                        "primary_source_tag",
                        "rows",
                        "mean_u",
                        "bad_mae_1r_rate",
                        "timeout_rate",
                        "wide_barrier_25bps_rate",
                        "loss_share",
                        "risk_event_share",
                    ]
                ].head(20)
            )
        )
    else:
        lines.append("No source summary rows.")
    lines.append("")
    lines.append("## Worst V2 Archetypes In Near-Misses")
    if not archetype_summary.empty:
        lines.append(
            _table(
                archetype_summary[
                    [
                        "primary_source_archetype_v2",
                        "rows",
                        "mean_u",
                        "bad_mae_1r_rate",
                        "timeout_rate",
                        "wide_barrier_25bps_rate",
                        "loss_share",
                        "risk_event_share",
                    ]
                ].head(20)
            )
        )
    else:
        lines.append("No V2 archetype summary rows.")
    lines.append("")
    lines.append("## Risk Mode Attribution")
    if not risk_summary.empty:
        lines.append(
            _table(
                risk_summary[
                    [
                        "risk_mode",
                        "rows",
                        "mean_u",
                        "bad_mae_1r_rate",
                        "timeout_rate",
                        "wide_barrier_25bps_rate",
                        "loss_share",
                    ]
                ].head(20)
            )
        )
    else:
        lines.append("No risk-mode rows.")
    lines.append("")
    lines.append("## Bad-MAE Economic Buckets")
    if not bad_mae_bucket_summary.empty:
        lines.append(
            _table(
                bad_mae_bucket_summary[
                    [
                        "bad_mae_failure_bucket",
                        "rows",
                        "mean_u",
                        "bad_mae_1r_rate",
                        "bad_mae_negative_utility_rate",
                        "bad_mae_recovered_rate",
                        "timeout_rate",
                        "wide_barrier_25bps_rate",
                        "loss_share",
                        "top_symbol",
                        "top_symbol_share",
                    ]
                ].head(25)
            )
        )
    else:
        lines.append("No bad-MAE bucket rows.")
    lines.append("")
    lines.append("## Bad-MAE Buckets By Selector")
    if not bad_mae_bucket_by_selector.empty:
        selector_bucket_cols = [
            "selection",
            "feature_set",
            "source_bucket",
            "causal_gate",
            "bad_mae_failure_bucket",
            "rows",
            "mean_u",
            "bad_mae_negative_utility_rate",
            "bad_mae_recovered_rate",
            "timeout_rate",
            "wide_barrier_25bps_rate",
            "top_symbol_share",
        ]
        lines.append(_table(bad_mae_bucket_by_selector[selector_bucket_cols].head(30)))
    else:
        lines.append("No selector bad-MAE bucket rows.")
    lines.append("")
    lines.append("## Worst Symbol Concentrations")
    if not symbol_summary.empty:
        lines.append(
            _table(
                symbol_summary[
                    [
                        "__symbol__",
                        "rows",
                        "mean_u",
                        "bad_mae_1r_rate",
                        "timeout_rate",
                        "wide_barrier_25bps_rate",
                        "loss_share",
                    ]
                ].head(20)
            )
        )
    else:
        lines.append("No symbol rows.")
    lines.append("")
    lines.append("## Selector Tradeoff Highlights")
    if not selector_tradeoffs.empty:
        trade_cols = [
            "feature_set",
            "source_bucket",
            "causal_gate",
            "selection",
            "top_frac",
            "mean_u",
            "delta_mean_u_vs_utility_only_local",
            "delta_bad_mae_1r_rate_vs_utility_only_local",
            "delta_timeout_rate_vs_utility_only_local",
            "delta_wide_barrier_25bps_rate_vs_utility_only_local",
            "delta_q25_week_u_vs_utility_only_local",
        ]
        lines.append(_table(selector_tradeoffs[trade_cols].head(20)))
    else:
        lines.append("No selector tradeoff rows.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- High-utility rows are not promotion candidates unless they also clear monthly, weekly, and risk gates.")
    lines.append("- Risk-only fixes are not sufficient when they trade timeout/wide-barrier improvement for bad-MAE or weekly instability.")
    lines.append("- Bad-MAE buckets separate adverse-path rows that are economically negative from rows that recovered despite MAE > 1R.")
    lines.append("- Source/archetype and symbol tables identify where the near-miss risk is concentrated for the next label or selector change.")
    report.write_text("\n".join(lines) + "\n")
    return report


def _table(frame: pd.DataFrame, limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    out = frame.head(limit) if limit is not None else frame
    return out.to_markdown(index=False, floatfmt=".4f")


def run_report(input_dir: Path, output_dir: Path, archetypes_v2_path: Path | None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_in = _load_manifest(input_dir)
    gates = _promotion_gates(manifest_in)

    aggregate_path = input_dir / "source_utility_path_timeout_risk_aggregate.csv"
    monthly_path = input_dir / "source_utility_path_timeout_risk_monthly.csv"
    weekly_path = input_dir / "source_utility_path_timeout_risk_weekly.csv"
    selected_path = input_dir / "source_utility_path_timeout_risk_selected_rows.parquet"

    aggregate = _add_failure_flags(pd.read_csv(aggregate_path), gates)
    monthly = pd.read_csv(monthly_path)
    weekly = pd.read_csv(weekly_path)
    selected = pd.read_parquet(selected_path)
    selected, selected_report = _enrich_selected(selected, archetypes_v2_path)

    candidate_cols = [
        "candidate",
        "failure_profile",
        "near_miss_reason",
        "failed_gates",
        "path_ok",
        "risk_ok",
        "all_gates_pass",
    ]
    selected = selected.merge(aggregate[candidate_cols], on="candidate", how="left", validate="many_to_one")
    near_miss_candidates = aggregate[
        aggregate["failure_profile"].isin(
            [
                "path_ok_risk_fail",
                "risk_ok_path_fail",
                "monthly_positive_but_week_or_risk_fail",
                "mean_positive_but_unstable",
            ]
        )
    ].copy()
    selected_near = selected[selected["candidate"].isin(set(near_miss_candidates["candidate"]))].copy()

    source_summary = _group_summary(selected_near, ["primary_source_tag"])
    archetype_summary = _group_summary(selected_near, ["primary_source_archetype_v2"])
    risk_summary = _group_summary(selected_near, ["risk_mode"])
    bad_mae_bucket_summary = _group_summary(selected_near, ["bad_mae_failure_bucket"])
    bad_mae_bucket_by_selector = _group_summary(
        selected_near,
        [
            "selection",
            "feature_set",
            "source_bucket",
            "causal_gate",
            "bad_mae_failure_bucket",
        ],
    )
    symbol_summary = _group_summary(selected_near, ["__symbol__"])
    month_week_summary = _group_summary(
        selected_near,
        ["period", "week_start", "failure_profile", "selection", "feature_set", "source_bucket", "causal_gate"],
    )
    source_by_selector = _group_summary(
        selected_near,
        [
            "selection",
            "feature_set",
            "source_bucket",
            "causal_gate",
            "primary_source_tag",
            "primary_source_archetype_v2",
        ],
    )
    selector_tradeoffs = _selector_tradeoffs(aggregate)

    aggregate_out = output_dir / "path_timeout_failure_aggregate.csv"
    near_out = output_dir / "path_timeout_failure_near_miss_candidates.csv"
    source_out = output_dir / "path_timeout_failure_by_source.csv"
    archetype_out = output_dir / "path_timeout_failure_by_v2_archetype.csv"
    risk_out = output_dir / "path_timeout_failure_by_risk_mode.csv"
    bad_mae_bucket_out = output_dir / "path_timeout_failure_by_bad_mae_bucket.csv"
    bad_mae_bucket_selector_out = output_dir / "path_timeout_failure_bad_mae_bucket_by_selector.csv"
    symbol_out = output_dir / "path_timeout_failure_by_symbol.csv"
    week_out = output_dir / "path_timeout_failure_by_week.csv"
    source_selector_out = output_dir / "path_timeout_failure_source_by_selector.csv"
    tradeoff_out = output_dir / "path_timeout_selector_tradeoffs.csv"

    aggregate.to_csv(aggregate_out, index=False)
    near_miss_candidates.to_csv(near_out, index=False)
    source_summary.to_csv(source_out, index=False)
    archetype_summary.to_csv(archetype_out, index=False)
    risk_summary.to_csv(risk_out, index=False)
    bad_mae_bucket_summary.to_csv(bad_mae_bucket_out, index=False)
    bad_mae_bucket_by_selector.to_csv(bad_mae_bucket_selector_out, index=False)
    symbol_summary.to_csv(symbol_out, index=False)
    month_week_summary.to_csv(week_out, index=False)
    source_by_selector.to_csv(source_selector_out, index=False)
    selector_tradeoffs.to_csv(tradeoff_out, index=False)

    out_manifest = {
        "scope": "source_utility_path_timeout_failure_attribution",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "gates": gates,
        "aggregate_rows": int(len(aggregate)),
        "monthly_rows": int(len(monthly)),
        "weekly_rows": int(len(weekly)),
        "selected_rows": int(len(selected)),
        "near_miss_candidates": int(len(near_miss_candidates)),
        "selected_near_miss_rows": int(len(selected_near)),
        "all_gate_candidates": int(aggregate["all_gates_pass"].sum()),
        "failure_profile_counts": aggregate["failure_profile"].value_counts().to_dict(),
        "selected_report": selected_report,
        "outputs": {
            "aggregate": str(aggregate_out),
            "near_miss_candidates": str(near_out),
            "source": str(source_out),
            "v2_archetype": str(archetype_out),
            "risk_mode": str(risk_out),
            "bad_mae_bucket": str(bad_mae_bucket_out),
            "bad_mae_bucket_by_selector": str(bad_mae_bucket_selector_out),
            "symbol": str(symbol_out),
            "week": str(week_out),
            "source_by_selector": str(source_selector_out),
            "selector_tradeoffs": str(tradeoff_out),
            "manifest": str(output_dir / "manifest.json"),
            "markdown": str(output_dir / "source_utility_path_timeout_failure_attribution_report.md"),
        },
    }
    report_path = _write_report(
        output_dir,
        aggregate,
        near_miss_candidates,
        month_week_summary,
        source_summary,
        archetype_summary,
        risk_summary,
        bad_mae_bucket_summary,
        bad_mae_bucket_by_selector,
        symbol_summary,
        selector_tradeoffs,
        out_manifest,
    )
    out_manifest["outputs"]["markdown"] = str(report_path)
    (output_dir / "manifest.json").write_text(json.dumps(_json_safe(out_manifest), indent=2) + "\n")
    return out_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archetypes-v2-path", type=Path, default=DEFAULT_ARCHETYPES_V2_PATH)
    parser.add_argument("--no-v2", action="store_true", help="Skip the v2 archetype join.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        archetypes_v2_path=None if args.no_v2 else args.archetypes_v2_path,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
