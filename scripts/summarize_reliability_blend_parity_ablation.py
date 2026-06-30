#!/usr/bin/env python3
"""Summarize reliability-blend parity and portfolio-ablation artifacts.

The script is intentionally a reporter, not a model runner.  It consolidates
the artifacts created by the parity-fix work and records arms that failed
contract validation or were intentionally deferred.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARM_DEFINITIONS: list[dict[str, Any]] = [
    {
        "arm": "A0",
        "name": "anchor_live_pipeline",
        "description": "Anchor-only baseline through the existing simple-policy candidate path.",
        "status": "completed_reference",
        "broad_path": "data_perp/artifacts/20260620_185313_no_mkt4_evband002_policy_uncertainty_ev/simple_policy_optimiser/simple_policy_candidates_broad.parquet",
        "deployable_path": "data_perp/artifacts/20260620_185313_no_mkt4_evband002_policy_uncertainty_ev/simple_policy_optimiser/simple_policy_candidates_deployable.parquet",
        "manifest_path": "data_perp/artifacts/20260620_185313_no_mkt4_evband002_policy_uncertainty_ev/policy_optimisation.json",
        "score_contract": "anchor_meta_score",
        "rank_contract": "legacy_anchor_simple_policy_rank",
        "barrier_contract": "existing_dynamic_policy",
        "notes": "Reference anchor arm. The source policy JSON is strategy-keyed rather than a single-summary schema.",
    },
    {
        "arm": "A1",
        "name": "native_teacher_reliability_blend",
        "description": "Native teacher reliability blend replay using persisted frozen rank references.",
        "status": "completed_historical_teacher_replay",
        "broad_path": "data_perp/artifacts/reliability_blend_native_rankref_simple_policy_replay_20260624/simple_policy_optimiser/simple_policy_candidates_broad.parquet",
        "deployable_path": "data_perp/artifacts/reliability_blend_native_rankref_simple_policy_replay_20260624/simple_policy_optimiser/simple_policy_candidates_deployable.parquet",
        "manifest_path": "data_perp/artifacts/reliability_blend_native_rankref_simple_policy_replay_20260624/policy_optimisation.json",
        "score_contract": "native_teacher_blend_score",
        "rank_contract": "PolicyRankReferenceStore.lookup / lookup_auction",
        "barrier_contract": "existing_dynamic_policy",
        "notes": "Historical teacher replay. Native live scorer is implemented separately and requires full-fit component bundles for production scoring.",
    },
    {
        "arm": "A2",
        "name": "legacy_distilled_student",
        "description": "Current distilled student preserved as audit/deployment-baseline comparison.",
        "status": "completed_legacy_audit_only",
        "broad_path": "data_perp/artifacts/reliability_blend_matrix_native_candidates_20260624_jun15_22_forced_features_floor070_pnl/simple_policy_optimiser/simple_policy_candidates_broad.parquet",
        "deployable_path": "data_perp/artifacts/reliability_blend_matrix_native_candidates_20260624_jun15_22_forced_features_floor070_pnl/simple_policy_optimiser/simple_policy_candidates_deployable.parquet",
        "manifest_path": "data_perp/artifacts/reliability_blend_matrix_native_candidates_20260624_jun15_22_forced_features_floor070_pnl/live_ledger_native_materialization_manifest.json",
        "score_contract": "distilled_student_pre_parity_patch",
        "rank_contract": "legacy_policy_rank_pct",
        "barrier_contract": "existing_dynamic_policy",
        "notes": "Kept only for comparison. This artifact predates the fail-closed rank-reference and student feature-contract patches.",
    },
    {
        "arm": "A3",
        "name": "parity_fixed_distilled_student",
        "description": "Parity-fixed distilled student with head×timestamp features and semantic feature contract.",
        "status": "completed_audit_fail_closed_score_ledger",
        "broad_path": None,
        "deployable_path": None,
        "manifest_path": "data_perp/reports/reliability_blend_matrix_live_scores_20260624_jun15_22_A3_parity_fixed/live_reliability_blend_score_manifest.json",
        "score_contract": "distilled_student_parity_fixed",
        "rank_contract": "fail_closed_not_used_for_portfolio",
        "barrier_contract": "not_applicable_score_ledger_only",
        "notes": (
            "Completed after live diagnostic materialization was fixed. This arm is intentionally audit/fallback "
            "only: teacher/student top-tail disagreement is large enough that the distilled path should fail "
            "closed instead of blocking the native teacher path."
        ),
    },
    {
        "arm": "A4",
        "name": "A3_causal_rank_reference_replay",
        "description": "A3 plus causal frozen rank-reference replay.",
        "status": "completed_audit_rank_reference_replay",
        "broad_path": "data_perp/artifacts/reliability_blend_matrix_native_candidates_20260624_jun15_22_A4_parity_fixed_rankref/simple_policy_optimiser/simple_policy_candidates_broad.parquet",
        "deployable_path": "data_perp/artifacts/reliability_blend_matrix_native_candidates_20260624_jun15_22_A4_parity_fixed_rankref/simple_policy_optimiser/simple_policy_candidates_deployable.parquet",
        "manifest_path": "data_perp/artifacts/reliability_blend_matrix_native_candidates_20260624_jun15_22_A4_parity_fixed_rankref/live_ledger_native_materialization_manifest.json",
        "score_contract": "distilled_student_parity_fixed",
        "rank_contract": "PolicyRankReferenceStore.lookup / lookup_auction",
        "barrier_contract": "existing_dynamic_policy",
        "notes": "Completed as an audit arm. Portfolio replay shows the student/rank-reference path should not be promoted.",
    },
    {
        "arm": "A5",
        "name": "causal_rank_volnorm_tpsl_hierarchical_ev",
        "description": "Native blend replay with causal rank references, volatility-normalized TP/SL, and hierarchical EV.",
        "status": "completed_contract_fixed_proxy",
        "broad_path": "data_perp/artifacts/reliability_blend_volnorm_tpsl_rankref_policy_20260624/simple_policy_optimiser/simple_policy_candidates_broad.parquet",
        "deployable_path": "data_perp/artifacts/reliability_blend_volnorm_tpsl_rankref_policy_20260624/simple_policy_optimiser/simple_policy_candidates_deployable.parquet",
        "manifest_path": "data_perp/artifacts/reliability_blend_volnorm_tpsl_rankref_policy_20260624/policy_optimisation.json",
        "score_contract": "native_teacher_blend_score",
        "rank_contract": "PolicyRankReferenceStore.lookup / lookup_auction",
        "barrier_contract": "tp_mult_sl_mult_times_barrier_pct_with_caps",
        "notes": "OOF TP/SL proxy period. Uses vol-normalized barriers and hierarchical EV curves.",
    },
    {
        "arm": "A6",
        "name": "observable_regime_mixture_of_experts",
        "description": "Observable-regime mixture-of-experts after A0-A5.",
        "status": "deferred_by_plan",
        "broad_path": None,
        "deployable_path": None,
        "manifest_path": None,
        "score_contract": "deferred",
        "rank_contract": "deferred",
        "barrier_contract": "deferred",
        "notes": "Deferred until after parity/economic-contract arms are reported, per user instruction.",
    },
    {
        "arm": "B0",
        "name": "direct_native_teacher_parity",
        "description": "Deployable native q_fail/new-period component scorer with persisted component feature contract.",
        "status": "completed_all_head_native_component_replay",
        "broad_path": "data_perp/artifacts/reliability_blend_arm_B0_full_native_blend_20260625_jun15_22/simple_policy_optimiser/simple_policy_candidates_broad.parquet",
        "deployable_path": "data_perp/artifacts/reliability_blend_arm_B0_full_native_blend_20260625_jun15_22/simple_policy_optimiser/simple_policy_candidates_deployable.parquet",
        "manifest_path": "data_perp/artifacts/reliability_blend_arm_B0_full_native_blend_20260625_jun15_22/live_ledger_native_materialization_manifest.json",
        "score_contract": "native_component_models",
        "rank_contract": "PolicyRankReferenceStore.lookup / frozen fullscope CDF",
        "barrier_contract": "existing_dynamic_policy",
        "notes": (
            "All-head full-fit native component scores and component-level portfolio replay are materialized for "
            "June 15-22. This validates deployable native scoring; the portfolio replay shows ranking improves but "
            "the current fixed portfolio policy does not yet convert that ranking lift into robust PnL."
        ),
    },
]


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _infer_head(strategy_id: Any) -> str | None:
    if not isinstance(strategy_id, str):
        return None
    if strategy_id.startswith("long_bars"):
        return "long_bars"
    if strategy_id.startswith("long_dist"):
        return "long_dist"
    if strategy_id.startswith("short_asset"):
        return "short_asset"
    if strategy_id.startswith("short_boll"):
        return "short_boll"
    return None


def _ensure_head(df: pd.DataFrame) -> pd.DataFrame:
    if "head" not in df.columns:
        df = df.copy()
        df["head"] = df.get("strategy_id", pd.Series(index=df.index, dtype=object)).map(_infer_head)
    return df


def _score_col(df: pd.DataFrame) -> str | None:
    for col in (
        "reliability_blend_score",
        "calibrated_score",
        "normalized_rank_score",
        "auction_rank_score",
        "policy_rank_pct",
    ):
        if col in df.columns:
            return col
    return None


def _net_col(df: pd.DataFrame) -> str | None:
    for col in ("fixed_return_net_after_cost", "net_return", "gross_return"):
        if col in df.columns:
            return col
    return None


def _hit_series(df: pd.DataFrame) -> pd.Series:
    if "fixed_y_tp" in df.columns and df["fixed_y_tp"].notna().any():
        return df["fixed_y_tp"].fillna(0).astype(float) > 0.5
    net_col = _net_col(df)
    if net_col is not None:
        return pd.to_numeric(df[net_col], errors="coerce").fillna(-np.inf) > 0.0
    return pd.Series(False, index=df.index)


def _timestamp_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if "timestamp" not in df.columns or df.empty:
        return None, None
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return ts.min().isoformat() if ts.notna().any() else None, ts.max().isoformat() if ts.notna().any() else None


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    return float(values.mean())


def _rate(mask: pd.Series) -> float | None:
    if len(mask) == 0:
        return None
    return float(mask.fillna(False).mean())


def _sum_col(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() == 0:
        return None
    return float(values.sum())


def _exit_rate(df: pd.DataFrame, name: str) -> float | None:
    if "simple_policy_exit_reason" in df.columns:
        reason = df["simple_policy_exit_reason"].astype(str).str.lower()
        if name == "tp":
            return _rate(reason.isin(["tp", "trailing", "capital_protect"]))
        if name == "sl":
            return _rate(reason.isin(["sl", "full_sl"]))
        if name == "timeout":
            return _rate(reason.eq("timeout"))
    if f"fixed_{name}" in df.columns:
        return _rate(pd.to_numeric(df[f"fixed_{name}"], errors="coerce") > 0)
    return None


def _top_hit_metrics(df: pd.DataFrame, prefix: str = "") -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if df.empty:
        return {f"{prefix}top{int(frac * 100)}_hit_rate": None for frac in (0.10, 0.20, 0.30)}
    score = _score_col(df)
    if score is None:
        return {f"{prefix}top{int(frac * 100)}_hit_rate": None for frac in (0.10, 0.20, 0.30)}
    work = df.copy()
    work["_score"] = pd.to_numeric(work[score], errors="coerce")
    work["_hit"] = _hit_series(work).astype(float)
    work = work[np.isfinite(work["_score"])]
    for frac in (0.10, 0.20, 0.30):
        key = f"{prefix}top{int(frac * 100)}_hit_rate"
        if work.empty:
            out[key] = None
            continue
        n = max(1, int(math.ceil(len(work) * frac)))
        out[key] = float(work.nlargest(n, "_score")["_hit"].mean())
    return out


def _timestamp_balanced_top_hit(df: pd.DataFrame, frac: float) -> float | None:
    if df.empty or "timestamp" not in df.columns:
        return None
    score = _score_col(df)
    if score is None:
        return None
    work = df.copy()
    work["_score"] = pd.to_numeric(work[score], errors="coerce")
    work["_hit"] = _hit_series(work).astype(float)
    work = work[np.isfinite(work["_score"])]
    if work.empty:
        return None
    vals: list[float] = []
    for _, group in work.groupby("timestamp", sort=False):
        if group.empty:
            continue
        n = max(1, int(math.ceil(len(group) * frac)))
        vals.append(float(group.nlargest(n, "_score")["_hit"].mean()))
    if not vals:
        return None
    return float(np.mean(vals))


def _weekly_quantiles(df: pd.DataFrame, frac: float = 0.30) -> dict[str, float | None]:
    result = {f"week_hr_top{int(frac * 100)}_q{q}": None for q in (5, 10, 25, 50, 75)}
    if df.empty or "timestamp" not in df.columns:
        return result
    score = _score_col(df)
    if score is None:
        return result
    work = df.copy()
    work["_timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["_score"] = pd.to_numeric(work[score], errors="coerce")
    work["_hit"] = _hit_series(work).astype(float)
    work = work[work["_timestamp"].notna() & np.isfinite(work["_score"])]
    if work.empty:
        return result
    ts_vals: list[tuple[pd.Timestamp, float]] = []
    for ts, group in work.groupby("_timestamp", sort=False):
        n = max(1, int(math.ceil(len(group) * frac)))
        ts_vals.append((ts, float(group.nlargest(n, "_score")["_hit"].mean())))
    if not ts_vals:
        return result
    tmp = pd.DataFrame(ts_vals, columns=["timestamp", "hit"])
    tmp["week"] = tmp["timestamp"].dt.to_period("W").astype(str)
    weekly = tmp.groupby("week")["hit"].mean()
    for q in (5, 10, 25, 50, 75):
        result[f"week_hr_top{int(frac * 100)}_q{q}"] = float(weekly.quantile(q / 100.0))
    return result


def _summarize_frame(df: pd.DataFrame, frame_name: str) -> dict[str, Any]:
    df = _ensure_head(df)
    ts_min, ts_max = _timestamp_range(df)
    score = _score_col(df)
    net = _net_col(df)
    out: dict[str, Any] = {
        f"{frame_name}_rows": int(len(df)),
        f"{frame_name}_timestamp_min": ts_min,
        f"{frame_name}_timestamp_max": ts_max,
        f"{frame_name}_score_col": score,
        f"{frame_name}_hit_definition": "fixed_y_tp" if "fixed_y_tp" in df.columns and df["fixed_y_tp"].notna().any() else "net_return_gt_0",
        f"{frame_name}_head_count": int(df["head"].nunique(dropna=True)) if "head" in df.columns else None,
        f"{frame_name}_positive_hit_rate": float(_hit_series(df).mean()) if len(df) else None,
        f"{frame_name}_mean_net": _safe_mean(df[net]) if net is not None else None,
        f"{frame_name}_net_pnl": _sum_col(df, net) if net is not None else None,
        f"{frame_name}_gross_pnl": _sum_col(df, "gross_return"),
        f"{frame_name}_cost_pnl": None,
        f"{frame_name}_tp_or_trailing_rate": _exit_rate(df, "tp"),
        f"{frame_name}_sl_rate": _exit_rate(df, "sl"),
        f"{frame_name}_timeout_rate": _exit_rate(df, "timeout"),
        f"{frame_name}_same_bar_conflict_rate": _rate(pd.to_numeric(df["fixed_conflict_same_bar"], errors="coerce") > 0)
        if "fixed_conflict_same_bar" in df.columns
        else None,
        f"{frame_name}_rank_missing_rate": None,
        f"{frame_name}_score_nonfinite_rate": None,
    }
    if out[f"{frame_name}_gross_pnl"] is not None and out[f"{frame_name}_net_pnl"] is not None:
        out[f"{frame_name}_cost_pnl"] = out[f"{frame_name}_gross_pnl"] - out[f"{frame_name}_net_pnl"]
    if score is not None:
        scores = pd.to_numeric(df[score], errors="coerce")
        out[f"{frame_name}_score_nonfinite_rate"] = float((~np.isfinite(scores)).mean()) if len(scores) else None
    if "policy_rank_pct" in df.columns:
        ranks = pd.to_numeric(df["policy_rank_pct"], errors="coerce")
        out[f"{frame_name}_rank_missing_rate"] = float(ranks.isna().mean()) if len(ranks) else None
    out.update(_top_hit_metrics(df, prefix=f"{frame_name}_global_"))
    for frac in (0.10, 0.20, 0.30):
        out[f"{frame_name}_timestamp_balanced_top{int(frac * 100)}_hit_rate"] = _timestamp_balanced_top_hit(df, frac)
    out.update({f"{frame_name}_{k}": v for k, v in _weekly_quantiles(df, frac=0.30).items()})
    return out


def _per_head_rows(arm: str, name: str, path: Path, frame_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = _ensure_head(pd.read_parquet(path))
    rows: list[dict[str, Any]] = []
    for head, group in df.groupby("head", dropna=False):
        row = {"arm": arm, "name": name, "frame": frame_name, "head": head}
        row.update(_summarize_frame(group, frame_name="head"))
        rows.append(row)
    return rows


def _manifest_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "schema_version",
        "metric_type",
        "costs_included",
        "rows",
        "timestamp_min",
        "timestamp_max",
        "deployment_status",
        "rank_policy",
        "candidate_rows",
        "deployable_rows",
        "candidate_timestamp_min",
        "candidate_timestamp_max",
        "deployable_timestamp_min",
        "deployable_timestamp_max",
        "barrier_mode",
        "effective_tp_mean",
        "effective_sl_mean",
        "barrier_pct_mean",
    ):
        if key in manifest:
            out[f"manifest_{key}"] = manifest[key]
    for block_name in ("portfolio_candidate_summary", "blend_portfolio_summary", "anchor_selected_threshold_portfolio_summary"):
        block = manifest.get(block_name)
        if isinstance(block, dict):
            for key, val in block.items():
                if isinstance(val, (int, float, str, bool)) or val is None:
                    out[f"{block_name}_{key}"] = val
    rank_reference = manifest.get("rank_reference")
    if isinstance(rank_reference, dict):
        for key, val in rank_reference.items():
            out[f"rank_reference_{key}"] = val
    return out


def _teacher_student_diagnostic(root: Path) -> pd.DataFrame:
    """Record whether teacher/student overlap can be computed on a shared panel."""
    rows = [
        {
            "comparison": "native_teacher_vs_legacy_student",
            "status": "unavailable",
            "reason": "native teacher scores and legacy distilled student scores are not materialized on the same timestamp/symbol/head panel",
            "spearman": np.nan,
            "top10_jaccard": np.nan,
            "top20_jaccard": np.nan,
            "top30_jaccard": np.nan,
        },
        {
            "comparison": "native_teacher_vs_parity_fixed_student",
            "status": "available_all_head",
            "reason": "all-head native full-fit component scores now exist on the same June 15-22 panel as A3",
            "spearman": np.nan,
            "top10_jaccard": np.nan,
            "top20_jaccard": np.nan,
            "top30_jaccard": np.nan,
        },
    ]
    old_scores = root / "data_perp/reports/reliability_blend_matrix_live_scores_20260624_jun15_22_forced_features/live_reliability_blend_scores.parquet"
    if old_scores.exists():
        df = pd.read_parquet(old_scores)
        if {"anchor_score", "reliability_blend_score"}.issubset(df.columns):
            rows.append(
                {
                    "comparison": "anchor_score_vs_legacy_student_score_jun15_22",
                    "status": "diagnostic_only_not_teacher_student",
                    "reason": "legacy live score artifact contains anchor and student scores, but not native teacher scores",
                    "spearman": float(
                        pd.to_numeric(df["anchor_score"], errors="coerce").corr(
                            pd.to_numeric(df["reliability_blend_score"], errors="coerce"),
                            method="spearman",
                        )
                    ),
                    "top10_jaccard": _top_overlap(df, "anchor_score", "reliability_blend_score", 0.10),
                    "top20_jaccard": _top_overlap(df, "anchor_score", "reliability_blend_score", 0.20),
                    "top30_jaccard": _top_overlap(df, "anchor_score", "reliability_blend_score", 0.30),
                }
            )
    new_scores = root / "data_perp/reports/reliability_blend_matrix_live_scores_20260624_jun15_22_A3_parity_fixed/live_reliability_blend_scores.parquet"
    native_scores = root / "data_perp/reports/native_reliability_blend_scores_20260625_jun15_22_fullfit/native_reliability_blend_scores.parquet"
    native_comparison_name = "native_teacher_all_head_vs_parity_fixed_student_jun15_22"
    native_comparison_status = "all_head"
    native_comparison_reason = "Direct full-fit native-teacher component scores compared with A3 parity-fixed student on their shared all-head panel"
    if not native_scores.exists():
        native_scores = root / "data_perp/reports/native_reliability_blend_scores_20260624_jun15_22_fullfit_smoke_v4/native_reliability_blend_scores.parquet"
        native_comparison_name = "native_teacher_smoke_vs_parity_fixed_student_long_bars_jun15_22"
        native_comparison_status = "partial_long_bars_only"
        native_comparison_reason = "Fallback to long_bars native-teacher smoke because the all-head native score ledger is unavailable"
    if new_scores.exists():
        df = pd.read_parquet(new_scores)
        if {"anchor_score", "reliability_blend_score"}.issubset(df.columns):
            rows.append(
                {
                    "comparison": "anchor_score_vs_parity_fixed_student_score_jun15_22",
                    "status": "diagnostic_only_not_teacher_student",
                    "reason": "A3 contains anchor and parity-fixed student scores, but not native teacher scores",
                    "spearman": float(
                        pd.to_numeric(df["anchor_score"], errors="coerce").corr(
                            pd.to_numeric(df["reliability_blend_score"], errors="coerce"),
                            method="spearman",
                        )
                    ),
                    "top10_jaccard": _top_overlap(df, "anchor_score", "reliability_blend_score", 0.10),
                    "top20_jaccard": _top_overlap(df, "anchor_score", "reliability_blend_score", 0.20),
                    "top30_jaccard": _top_overlap(df, "anchor_score", "reliability_blend_score", 0.30),
                }
            )
    if native_scores.exists() and new_scores.exists():
        native = pd.read_parquet(native_scores)
        student = pd.read_parquet(new_scores)
        keys = [c for c in ("head", "timestamp", "symbol", "strategy_id") if c in native.columns and c in student.columns]
        if keys and {"reliability_blend_score"}.issubset(native.columns) and {"reliability_blend_score"}.issubset(student.columns):
            merged = native.loc[:, keys + ["reliability_blend_score"]].merge(
                student.loc[:, keys + ["reliability_blend_score"]],
                on=keys,
                how="inner",
                suffixes=("_native", "_student"),
            )
            if not merged.empty:
                rows.append(
                    {
                        "comparison": native_comparison_name,
                        "status": native_comparison_status,
                        "reason": native_comparison_reason,
                        "spearman": float(
                            pd.to_numeric(merged["reliability_blend_score_native"], errors="coerce").corr(
                                pd.to_numeric(merged["reliability_blend_score_student"], errors="coerce"),
                                method="spearman",
                            )
                        ),
                        "top10_jaccard": _top_overlap(
                            merged,
                            "reliability_blend_score_native",
                            "reliability_blend_score_student",
                            0.10,
                        ),
                        "top20_jaccard": _top_overlap(
                            merged,
                            "reliability_blend_score_native",
                            "reliability_blend_score_student",
                            0.20,
                        ),
                        "top30_jaccard": _top_overlap(
                            merged,
                            "reliability_blend_score_native",
                            "reliability_blend_score_student",
                            0.30,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _top_overlap(df: pd.DataFrame, a: str, b: str, frac: float) -> float | None:
    work = df[["timestamp", "symbol", "head", a, b]].copy()
    work[a] = pd.to_numeric(work[a], errors="coerce")
    work[b] = pd.to_numeric(work[b], errors="coerce")
    work = work[np.isfinite(work[a]) & np.isfinite(work[b])]
    if work.empty:
        return None
    vals: list[float] = []
    for _, group in work.groupby(["head", "timestamp"], sort=False):
        if len(group) < 2:
            continue
        n = max(1, int(math.ceil(len(group) * frac)))
        left = set(group.nlargest(n, a).index)
        right = set(group.nlargest(n, b).index)
        union = left | right
        vals.append(len(left & right) / len(union) if union else np.nan)
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def build_report(output_dir: Path, root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []

    for arm in ARM_DEFINITIONS:
        row = {k: v for k, v in arm.items() if not k.endswith("_path")}
        manifest_path = root / arm["manifest_path"] if arm.get("manifest_path") else None
        manifest = _read_json(manifest_path)
        row["manifest_path"] = str(manifest_path) if manifest_path else None
        row.update(_manifest_metrics(manifest))
        for frame_name in ("broad", "deployable"):
            raw_path = arm.get(f"{frame_name}_path")
            path = root / raw_path if raw_path else None
            row[f"{frame_name}_path"] = str(path) if path else None
            row[f"{frame_name}_path_exists"] = bool(path and path.exists())
            if path and path.exists():
                df = pd.read_parquet(path)
                row.update(_summarize_frame(df, frame_name))
                head_rows.extend(_per_head_rows(str(arm["arm"]), str(arm["name"]), path, frame_name))
        arm_rows.append(row)

    arms = pd.DataFrame(arm_rows)
    per_head = pd.DataFrame(head_rows)
    teacher_student = _teacher_student_diagnostic(root)

    portfolio_paths = [
        root / "data_perp/reports/native_reliability_component_arm_ablation_20260625_jun15_22/portfolio_refit_bar4_strategy_bar2_summary.csv",
        root / "data_perp/reports/reliability_blend_parity_portfolio_ablation_20260624/portfolio_policy_ablation_summary.csv",
    ]
    portfolio = pd.DataFrame()
    for portfolio_path in portfolio_paths:
        if portfolio_path.exists() and portfolio_path.stat().st_size > 0:
            portfolio = pd.read_csv(portfolio_path)
            break

    arms.to_csv(output_dir / "ablation_arm_summary.csv", index=False)
    per_head.to_csv(output_dir / "ablation_per_head_metrics.csv", index=False)
    teacher_student.to_csv(output_dir / "teacher_student_overlap_diagnostics.csv", index=False)
    if not portfolio.empty:
        portfolio.to_csv(output_dir / "portfolio_policy_ablation_summary_copy.csv", index=False)

    report = _render_markdown(arms, per_head, teacher_student, portfolio)
    (output_dir / "reliability_blend_parity_ablation_report.md").write_text(report, encoding="utf-8")


def _fmt(val: Any, digits: int = 4) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except TypeError:
        pass
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def _render_markdown(
    arms: pd.DataFrame,
    per_head: pd.DataFrame,
    teacher_student: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Reliability Blend Parity And Portfolio Ablation Report")
    lines.append("")
    lines.append("## Arm Status")
    display_cols = [
        "arm",
        "name",
        "status",
        "broad_rows",
        "deployable_rows",
        "deployable_timestamp_min",
        "deployable_timestamp_max",
        "deployable_positive_hit_rate",
        "deployable_net_pnl",
        "deployable_gross_pnl",
        "deployable_cost_pnl",
        "deployable_sl_rate",
        "deployable_timeout_rate",
        "rank_contract",
        "barrier_contract",
    ]
    lines.extend(_markdown_table(arms, display_cols))
    lines.append("")
    lines.append("## Top-K Hit Metrics")
    top_cols = [
        "arm",
        "name",
        "status",
        "broad_hit_definition",
        "broad_global_top10_hit_rate",
        "broad_global_top20_hit_rate",
        "broad_global_top30_hit_rate",
        "broad_timestamp_balanced_top10_hit_rate",
        "broad_timestamp_balanced_top20_hit_rate",
        "broad_timestamp_balanced_top30_hit_rate",
        "broad_week_hr_top30_q5",
        "broad_week_hr_top30_q10",
        "broad_week_hr_top30_q25",
        "broad_week_hr_top30_q50",
        "broad_week_hr_top30_q75",
    ]
    lines.extend(_markdown_table(arms, top_cols))
    lines.append("")
    lines.append("## Teacher Student Diagnostics")
    lines.extend(_markdown_table(teacher_student, list(teacher_student.columns)))
    lines.append("")
    lines.append("## Portfolio Policy Ablation")
    if portfolio.empty:
        lines.append("No portfolio policy ablation artifact found.")
    else:
        cols = [
            "arm",
            "sample",
            "variant",
            "candidate_rows",
            "timestamp_min",
            "timestamp_max",
            "trade_count",
            "net_pnl",
            "gross_pnl",
            "full_sl_rate",
            "timeout_rate",
            "max_drawdown",
        ]
        lines.extend(_markdown_table(portfolio, cols))
    lines.append("")
    lines.append("## Contract Findings")
    lines.append("- Native teacher scoring is now separated from the distilled student path and uses frozen rank-reference lookup in the patched replay/materialization scripts.")
    lines.append("- The patched distilled-student path now scores the June 15-22 live ledger after explicit diagnostic materialization, but remains audit/fallback only and should fail closed for active deployment.")
    lines.append("- Direct native-teacher B0 scoring now has all-head full-fit component scores, non-collapsed q_fail/period components, and June 15-22 candidate plus portfolio replay artifacts.")
    lines.append("- The fixed TP/SL proxy now records volatility-normalized effective TP/SL/barrier fields and same-bar conflict diagnostics.")
    lines.append("- Hierarchical EV curves are available in the portfolio replay path; the copied portfolio ablation table records the latest historical and June 15-22 replay outcomes.")
    lines.append("- HeadHealth active entrypoints are deprecated behind an explicit flag and are excluded from this ablation report.")
    return "\n".join(lines) + "\n"


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    present = [c for c in cols if c in df.columns]
    if not present:
        return ["No columns available."]
    lines = ["|" + "|".join(present) + "|", "|" + "|".join(["---"] * len(present)) + "|"]
    for _, row in df[present].iterrows():
        lines.append("|" + "|".join(_fmt(row[c]) for c in present) + "|")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/reliability_blend_parity_ablation_matrix_20260624"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(args.output_dir, args.root.resolve())
    print(f"Wrote reliability blend parity ablation report to {args.output_dir}")


if __name__ == "__main__":
    main()
