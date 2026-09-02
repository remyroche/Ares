#!/usr/bin/env python3
"""Audit frozen reliability challenger status across research and live gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.build_contextual_tp_sl_evidence_matrix import (
    _marginal_family_ablation_from_scorecards,
    _requested_reliability_family_verdict,
)
from scripts.run_latest_frozen_dual_scoring_gate_if_ready import DIAGNOSTIC_GROUPS


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _candidate_rows(
    decision: pd.DataFrame,
    bootstrap: pd.DataFrame,
    bundle: Dict[str, Any],
    *,
    min_bootstrap_prob: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    merged = decision.merge(bootstrap, on="rule_id", how="left", suffixes=("", "_bootstrap"))
    for _, row in merged.iterrows():
        rule_id = str(row["rule_id"])
        bundle_rule = (bundle.get("rules") or {}).get(rule_id, {})
        active_positive_week_share = float(row.get("active_positive_week_share", 0.0))
        worst_week_delta = float(row.get("worst_week_delta", 0.0))
        prob_net = float(row.get("prob_delta_net_pnl_positive", 0.0))
        prob_obj = float(row.get("prob_delta_objective_positive", 0.0))
        research_pass = (
            float(row.get("delta_net_pnl", 0.0)) > 0.0
            and float(row.get("delta_objective", 0.0)) > 0.0
            and prob_net >= min_bootstrap_prob
            and prob_obj >= min_bootstrap_prob
        )
        rows.append(
            {
                "rule_id": rule_id,
                "role": bundle_rule.get("role"),
                "delta_net_pnl": float(row.get("delta_net_pnl", 0.0)),
                "delta_objective": float(row.get("delta_objective", 0.0)),
                "active_weeks": int(row.get("active_weeks", 0)),
                "active_positive_week_share": active_positive_week_share,
                "worst_week_delta": worst_week_delta,
                "prob_delta_net_pnl_positive": prob_net,
                "prob_delta_objective_positive": prob_obj,
                "delta_net_pnl_p05": float(row.get("delta_net_pnl_p05", np.nan)),
                "delta_objective_p05": float(row.get("delta_objective_p05", np.nan)),
                "entrant_trades": float(row.get("entrant_trades", np.nan)),
                "entrant_net_pnl": float(row.get("entrant_net_pnl", np.nan)),
                "entrant_hit_rate": float(row.get("entrant_hit_rate", np.nan)),
                "entrant_full_sl_rate": float(row.get("entrant_full_sl_rate", np.nan)),
                "removed_trades": float(row.get("removed_trades", np.nan)),
                "removed_net_pnl": float(row.get("removed_net_pnl", np.nan)),
                "removed_hit_rate": float(row.get("removed_hit_rate", np.nan)),
                "removed_full_sl_rate": float(row.get("removed_full_sl_rate", np.nan)),
                "entrant_minus_removed_net_pnl": float(row.get("entrant_minus_removed_net_pnl", np.nan)),
                "entrant_minus_removed_hit_rate": float(row.get("entrant_minus_removed_hit_rate", np.nan)),
                "entrant_minus_removed_full_sl_rate": float(
                    row.get("entrant_minus_removed_full_sl_rate", np.nan)
                ),
                "research_pass": research_pass,
                "tail_clean": worst_week_delta >= 0.0 and active_positive_week_share >= 1.0,
                "promotion_note": bundle_rule.get("promotion_note"),
            }
        )
    return pd.DataFrame(rows)


def _status_from_gate(gate: Dict[str, Any]) -> tuple[bool, List[str]]:
    if not gate:
        return False, ["fresh_gate_manifest_missing"]
    if _bool(gate.get("ran_gate")):
        return True, []
    nearest = gate.get("nearest_source") or {}
    reasons = str(nearest.get("rejection_reasons") or "fresh_gate_not_run")
    return False, [part for part in reasons.split(";") if part]


def _gate_summary_rows(gate: Dict[str, Any]) -> pd.DataFrame:
    if not gate:
        return pd.DataFrame(
            [
                {
                    "ready_sources": 0,
                    "ran_gate": False,
                    "nearest_source": None,
                    "post_cutoff_rows": 0,
                    "policy_action_rows": 0,
                    "policy_outcome_rows": 0,
                    "required_head_outcome_gaps": "fresh_gate_manifest_missing",
                    "drift_finite_row_rate": np.nan,
                    "recent_hr_finite_row_rate": np.nan,
                    "ood_finite_row_rate": np.nan,
                    "uncertainty_finite_row_rate": np.nan,
                }
            ]
        )
    nearest = gate.get("nearest_source") or {}
    return pd.DataFrame(
        [
            {
                "ready_sources": int(gate.get("ready_sources") or 0),
                "ran_gate": _bool(gate.get("ran_gate")),
                "nearest_source": nearest.get("path"),
                "post_cutoff_rows": int(nearest.get("post_cutoff_rows") or 0),
                "policy_action_rows": int(nearest.get("policy_action_rows_estimate") or 0),
                "policy_outcome_rows": int(nearest.get("policy_outcome_rows_estimate") or 0),
                "required_head_outcome_gaps": nearest.get("policy_outcome_low_required_head_counts") or "{}",
                "drift_finite_row_rate": float(nearest.get("drift_finite_row_rate", np.nan)),
                "recent_hr_finite_row_rate": float(
                    nearest.get("recent_hit_rate_surprise_finite_row_rate", np.nan)
                ),
                "ood_finite_row_rate": float(nearest.get("ood_finite_row_rate", np.nan)),
                "uncertainty_finite_row_rate": float(nearest.get("uncertainty_finite_row_rate", np.nan)),
            }
        ]
    )


def _nearest_source_family_coverage_rows(gate: Dict[str, Any]) -> pd.DataFrame:
    nearest = (gate or {}).get("nearest_source") or {}
    path_raw = nearest.get("path")
    if not path_raw:
        return pd.DataFrame()
    path = Path(path_raw)
    if not path.exists():
        return pd.DataFrame()
    cutoff_raw = (gate or {}).get("cutoff")
    cutoff = pd.to_datetime(cutoff_raw, utc=True, errors="coerce") if cutoff_raw else pd.NaT
    use_cols = ["timestamp", "strategy_id"]
    for cols in DIAGNOSTIC_GROUPS.values():
        use_cols.extend(cols)
    try:
        try:
            import pyarrow.parquet as pq  # type: ignore

            available_cols = set(pq.read_schema(path).names)
        except Exception:
            available_cols = set(pd.read_parquet(path).columns)
        read_cols = [col for col in dict.fromkeys(use_cols) if col in available_cols]
        frame = pd.read_parquet(path, columns=read_cols)
    except Exception:
        return pd.DataFrame()
    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame()
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    mask = timestamp.notna()
    if pd.notna(cutoff):
        mask &= timestamp.ge(cutoff)
    post = frame.loc[mask].copy()
    if post.empty:
        return pd.DataFrame()
    if "strategy_id" in post.columns:
        head = post["strategy_id"].astype(str).str.extract(
            r"^(short_bollinger|short_boll|long_bars|long_dist|short_asset)",
            expand=False,
        ).replace({"short_boll": "short_bollinger"})
    else:
        head = pd.Series("unknown", index=post.index)
    post["_head"] = head.fillna("unknown")

    rows: List[Dict[str, Any]] = []
    for family, cols in DIAGNOSTIC_GROUPS.items():
        present_cols = [col for col in cols if col in post.columns]
        for head_name, group in post.groupby("_head", dropna=False):
            rows_total = int(len(group))
            if not present_cols or rows_total <= 0:
                finite_rows = finite_cells = 0
                finite_row_rate = finite_cell_rate = 0.0
            else:
                values = group[present_cols].apply(pd.to_numeric, errors="coerce")
                finite = values.notna()
                finite_any = finite.any(axis=1)
                finite_rows = int(finite_any.sum())
                finite_cells = int(finite.to_numpy(dtype=bool, copy=False).sum())
                finite_row_rate = float(finite_rows / max(rows_total, 1))
                finite_cell_rate = float(finite_cells / max(rows_total * len(present_cols), 1))
            rows.append(
                {
                    "source_path": str(path),
                    "family": family,
                    "head": str(head_name),
                    "post_cutoff_rows": rows_total,
                    "columns_present": len(present_cols),
                    "columns_required": len(cols),
                    "finite_rows": finite_rows,
                    "finite_row_rate": finite_row_rate,
                    "finite_cells": finite_cells,
                    "finite_cell_rate": finite_cell_rate,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["family", "finite_row_rate", "head"], ascending=[True, True, True])


def _count_map(value: Any) -> Dict[str, int]:
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if value is None:
        return {}
    try:
        payload = json.loads(str(value))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): int(v) for k, v in payload.items()}


def _fresh_evidence_gap_rows(bundle: Dict[str, Any], gate: Dict[str, Any]) -> pd.DataFrame:
    req = bundle.get("forward_validation_requirements") or {}
    nearest = (gate or {}).get("nearest_source") or {}
    requirements = [
        ("post_cutoff_rows", nearest.get("post_cutoff_rows"), req.get("minimum_post_cutoff_rows", 2000)),
        (
            "post_cutoff_timestamps",
            nearest.get("post_cutoff_timestamps"),
            req.get("minimum_post_cutoff_timestamps", 40),
        ),
        ("post_cutoff_active_heads", nearest.get("post_cutoff_active_heads"), req.get("minimum_active_heads", 3)),
        (
            "policy_action_rows",
            nearest.get("policy_action_rows_estimate"),
            req.get("minimum_policy_action_rows", 50),
        ),
        (
            "policy_action_timestamps",
            nearest.get("policy_action_timestamps_estimate"),
            req.get("minimum_policy_action_timestamps", 10),
        ),
        (
            "policy_outcome_rows",
            nearest.get("policy_outcome_rows_estimate"),
            req.get("minimum_policy_outcome_rows", 50),
        ),
        (
            "policy_outcome_timestamps",
            nearest.get("policy_outcome_timestamps_estimate"),
            req.get("minimum_policy_outcome_timestamps", 10),
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for gate_name, observed_raw, required_raw in requirements:
        observed = int(observed_raw or 0)
        required = int(required_raw or 0)
        rows.append(
            {
                "gate": gate_name,
                "head": "",
                "observed": observed,
                "required": required,
                "deficit": max(required - observed, 0),
                "pass": observed >= required,
            }
        )

    outcome_counts = _count_map(nearest.get("policy_outcome_head_counts"))
    required_heads = list(req.get("required_matured_outcome_heads") or [])
    required_per_head = int(req.get("minimum_policy_outcome_rows_per_required_head", 3))
    for head in required_heads:
        observed = int(outcome_counts.get(str(head), 0))
        rows.append(
            {
                "gate": "policy_outcome_rows_per_required_head",
                "head": str(head),
                "observed": observed,
                "required": required_per_head,
                "deficit": max(required_per_head - observed, 0),
                "pass": observed >= required_per_head,
            }
        )
    return pd.DataFrame(rows)


def _promotion_blocker_rows(fresh_gaps: pd.DataFrame) -> pd.DataFrame:
    if fresh_gaps.empty:
        return pd.DataFrame()
    blocked = fresh_gaps.loc[~fresh_gaps.get("pass", pd.Series(False, index=fresh_gaps.index)).astype(bool)].copy()
    if blocked.empty:
        return pd.DataFrame(
            [
                {
                    "blocker": "none",
                    "head": "",
                    "observed": np.nan,
                    "required": np.nan,
                    "deficit": 0,
                    "severity": "ready",
                    "next_action": "Fresh gate requirements are satisfied; run the frozen replay gate.",
                }
            ]
        )
    rows: List[Dict[str, Any]] = []
    action_map = {
        "post_cutoff_rows": "Accumulate or materialize more post-cutoff candidate rows under the frozen contract.",
        "policy_action_rows": "Wait for more accepted policy actions or broaden prospective dual-scoring coverage without changing candidates.",
        "policy_outcome_rows": "Wait for more accepted trades to mature with finite replay outcomes.",
        "policy_outcome_rows_per_required_head": "Require matured outcomes for the missing required head before promotion.",
        "post_cutoff_timestamps": "Accumulate more post-cutoff timestamps.",
        "policy_action_timestamps": "Accumulate more timestamps with accepted policy actions.",
        "policy_outcome_timestamps": "Accumulate more timestamps with matured policy outcomes.",
        "post_cutoff_active_heads": "Accumulate post-cutoff rows across more active heads.",
    }
    for _, row in blocked.iterrows():
        gate_name = str(row.get("gate") or "")
        deficit = int(row.get("deficit") or 0)
        severity = "hard_blocker" if deficit > 0 else "review"
        rows.append(
            {
                "blocker": gate_name,
                "head": "" if pd.isna(row.get("head")) else row.get("head"),
                "observed": row.get("observed"),
                "required": row.get("required"),
                "deficit": deficit,
                "severity": severity,
                "next_action": action_map.get(gate_name, "Collect more frozen post-cutoff evidence before promotion."),
            }
        )
    return pd.DataFrame(rows).sort_values(["severity", "deficit"], ascending=[True, False])


def _best_scorecard_rows(scorecard_dir: Path | None) -> pd.DataFrame:
    if scorecard_dir is None:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    specs = [
        (
            "promotion",
            scorecard_dir / "promotion_scorecard.csv",
            "scorecard_score",
            {
                "variant": "variant",
                "family": "family",
                "delta_net_pnl": "delta_vs_baseline_net_pnl",
                "delta_objective": "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
                "hit_rate": "hit_rate",
                "full_sl_rate": "full_sl_rate",
                "tail_metric": "weekly_q05_pnl",
            },
        ),
        (
            "expanding_family",
            scorecard_dir / "expanding_family_scorecard.csv",
            "scorecard_score",
            {
                "variant": "variant",
                "family": "family",
                "delta_net_pnl": "delta_net_pnl",
                "delta_objective": "delta_objective_week",
                "hit_rate": "delta_hit_rate",
                "full_sl_rate": "delta_full_sl_rate",
                "tail_metric": "delta_worst_week_net_pnl",
            },
        ),
        (
            "recent_hr_tailgrid",
            scorecard_dir / "tailgrid_recent_hr_scorecard.csv",
            "scorecard_score",
            {
                "variant": "variant",
                "family": "family",
                "delta_net_pnl": "delta_net_pnl",
                "delta_objective": "tail_objective_delta",
                "hit_rate": None,
                "full_sl_rate": "delta_full_sl_rate",
                "tail_metric": "delta_q05",
            },
        ),
        (
            "recent_hr_headscope",
            scorecard_dir / "headscope_recent_hr_scorecard.csv",
            "scorecard_score",
            {
                "variant": "variant",
                "family": "family",
                "delta_net_pnl": "delta_net_pnl",
                "delta_objective": "tail_objective_delta",
                "hit_rate": None,
                "full_sl_rate": "delta_full_sl_rate",
                "tail_metric": "delta_q05",
            },
        ),
    ]
    for source, path, sort_col, cols in specs:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or sort_col not in frame.columns:
            continue
        frame = frame.loc[~frame.get("family", pd.Series("", index=frame.index)).eq("baseline_or_other")].copy()
        if frame.empty:
            continue
        best = frame.sort_values(sort_col, ascending=False).iloc[0]
        row = {"source": source, "scorecard_score": float(best.get(sort_col, np.nan))}
        for out_col, in_col in cols.items():
            row[out_col] = best.get(in_col, np.nan) if in_col is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _ab_verdict(row: pd.Series) -> str:
    delta_net = float(row.get("delta_net_pnl", np.nan))
    delta_objective = float(row.get("delta_objective", np.nan))
    delta_full_sl = float(row.get("delta_full_sl_rate", np.nan))
    delta_q20 = float(row.get("delta_q20_pnl", np.nan))
    delta_q35 = float(row.get("delta_q35_pnl", np.nan))
    tail_metric = float(row.get("tail_metric", np.nan))
    if not np.isfinite(delta_net) or delta_net <= 0.0:
        return "reject_nonpositive_pnl"
    tail_flags = [
        np.isfinite(delta_objective) and delta_objective > 0.0,
        (not np.isfinite(delta_full_sl)) or delta_full_sl <= 0.0,
        (not np.isfinite(delta_q20)) or delta_q20 >= 0.0,
        (not np.isfinite(delta_q35)) or delta_q35 >= 0.0,
        (not np.isfinite(tail_metric)) or tail_metric >= 0.0,
    ]
    if all(tail_flags):
        return "pnl_and_tail_supportive"
    if tail_flags[0] and tail_flags[1] and sum(bool(flag) for flag in tail_flags[2:]) >= 1:
        return "pnl_positive_tail_mixed"
    return "pnl_positive_tail_weak"


def _reliability_ab_scorecard_rows(scorecard_dir: Path | None) -> pd.DataFrame:
    if scorecard_dir is None:
        return pd.DataFrame()
    specs = [
        (
            "promotion",
            scorecard_dir / "promotion_scorecard.csv",
            {
                "variant": "variant",
                "role": "role",
                "family": "family",
                "net_pnl": "net_pnl",
                "delta_net_pnl": "delta_vs_baseline_net_pnl",
                "hit_rate": "hit_rate",
                "full_sl_rate": "full_sl_rate",
                "delta_full_sl_rate": "delta_vs_baseline_full_sl_rate",
                "delta_objective": "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20",
                "tail_metric": "weekly_q05_pnl",
                "delta_q20_pnl": "weekly_q20_pnl",
                "delta_q35_pnl": None,
                "scorecard_score": "scorecard_score",
            },
        ),
        (
            "expanding_family",
            scorecard_dir / "expanding_family_scorecard.csv",
            {
                "variant": "variant",
                "role": None,
                "family": "family",
                "net_pnl": None,
                "delta_net_pnl": "delta_net_pnl",
                "hit_rate": "delta_hit_rate",
                "full_sl_rate": None,
                "delta_full_sl_rate": "delta_full_sl_rate",
                "delta_objective": "delta_objective_week",
                "tail_metric": "delta_worst_week_net_pnl",
                "delta_q20_pnl": "delta_q20_week_net_pnl",
                "delta_q35_pnl": "delta_q35_week_net_pnl",
                "scorecard_score": "scorecard_score",
            },
        ),
        (
            "tailgrid_recent_hr",
            scorecard_dir / "tailgrid_recent_hr_scorecard.csv",
            {
                "variant": "variant",
                "role": None,
                "family": "family",
                "net_pnl": "net_pnl",
                "delta_net_pnl": "delta_net_pnl",
                "hit_rate": None,
                "full_sl_rate": "full_sl_rate",
                "delta_full_sl_rate": "delta_full_sl_rate",
                "delta_objective": "tail_objective_delta",
                "tail_metric": "delta_q05",
                "delta_q20_pnl": "delta_q20",
                "delta_q35_pnl": "delta_q35",
                "scorecard_score": "scorecard_score",
            },
        ),
        (
            "headscope_recent_hr",
            scorecard_dir / "headscope_recent_hr_scorecard.csv",
            {
                "variant": "variant",
                "role": None,
                "family": "family",
                "net_pnl": "net_pnl",
                "delta_net_pnl": "delta_net_pnl",
                "hit_rate": None,
                "full_sl_rate": "full_sl_rate",
                "delta_full_sl_rate": "delta_full_sl_rate",
                "delta_objective": "tail_objective_delta",
                "tail_metric": "delta_q05",
                "delta_q20_pnl": "delta_q20",
                "delta_q35_pnl": "delta_q35",
                "scorecard_score": "scorecard_score",
            },
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for source, path, cols in specs:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        for _, item in frame.iterrows():
            row: Dict[str, Any] = {"source": source}
            row["evidence_family"] = item.get("evidence_family", source)
            for out_col, in_col in cols.items():
                row[out_col] = item.get(in_col, np.nan) if in_col is not None else np.nan
            family = str(row.get("family") or "")
            row["contains_drift"] = "drift" in family.lower()
            row["contains_recent_hit_rate_surprise"] = (
                "recent_hr" in family.lower() or "recent_hit_rate" in family.lower()
            )
            row["contains_ood"] = "ood" in family.lower()
            row["contains_uncertainty"] = "uncertainty" in family.lower()
            row["ab_verdict"] = _ab_verdict(pd.Series(row))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    for col in (
        "net_pnl",
        "delta_net_pnl",
        "hit_rate",
        "full_sl_rate",
        "delta_full_sl_rate",
        "delta_objective",
        "tail_metric",
        "delta_q20_pnl",
        "delta_q35_pnl",
        "scorecard_score",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(
        ["scorecard_score", "delta_net_pnl", "delta_objective"],
        ascending=[False, False, False],
        na_position="last",
    )


def _scaled_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    clipped = values.clip(lower=0.0)
    max_val = float(clipped.max())
    if max_val <= 0.0 or not np.isfinite(max_val):
        return pd.Series(0.0, index=series.index)
    return clipped / max_val


def _tail_balance_score(frame: pd.DataFrame) -> pd.Series:
    pieces: List[pd.Series] = []
    if "delta_full_sl_rate" in frame.columns:
        pieces.append(_scaled_positive(-pd.to_numeric(frame["delta_full_sl_rate"], errors="coerce")))
    if "tail_metric" in frame.columns:
        pieces.append(_scaled_positive(frame["tail_metric"]))
    if "delta_q20_pnl" in frame.columns:
        pieces.append(_scaled_positive(frame["delta_q20_pnl"]))
    if "delta_q35_pnl" in frame.columns:
        pieces.append(_scaled_positive(frame["delta_q35_pnl"]))
    if not pieces:
        return pd.Series(0.0, index=frame.index)
    return pd.concat(pieces, axis=1).mean(axis=1)


def _reliability_ab_selection_frontier(scorecards: pd.DataFrame) -> pd.DataFrame:
    if scorecards.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["source", "evidence_family"]
    for (source, evidence_family), group in scorecards.groupby(group_cols, dropna=False):
        scoped = group.copy()
        scoped["pnl_score"] = _scaled_positive(scoped.get("delta_net_pnl", pd.Series(0.0, index=scoped.index)))
        scoped["objective_score"] = _scaled_positive(
            scoped.get("delta_objective", pd.Series(0.0, index=scoped.index))
        )
        scoped["tail_score"] = _tail_balance_score(scoped)
        scoped["balanced_score"] = (
            0.45 * scoped["pnl_score"] + 0.25 * scoped["objective_score"] + 0.30 * scoped["tail_score"]
        )

        policies = [
            ("max_pnl", "delta_net_pnl", False, "Highest net-PnL delta inside the same evidence group."),
            ("balanced_pnl_tail", "balanced_score", False, "Weighted PnL/objective/tail balance inside the same evidence group."),
            ("tail_first", "tail_score", False, "Highest tail score among PnL-positive variants inside the same evidence group."),
        ]
        for policy_id, sort_col, ascending, rationale in policies:
            candidates = scoped
            if policy_id == "tail_first":
                candidates = scoped.loc[pd.to_numeric(scoped["delta_net_pnl"], errors="coerce").gt(0.0)]
                if candidates.empty:
                    candidates = scoped
            best = candidates.sort_values(
                [sort_col, "delta_net_pnl", "delta_objective"],
                ascending=[ascending, False, False],
                na_position="last",
            ).iloc[0]
            rows.append(
                {
                    "source": source,
                    "evidence_family": evidence_family,
                    "policy_id": policy_id,
                    "selected_variant": best.get("variant"),
                    "family": best.get("family"),
                    "delta_net_pnl": best.get("delta_net_pnl"),
                    "delta_objective": best.get("delta_objective"),
                    "delta_full_sl_rate": best.get("delta_full_sl_rate"),
                    "tail_metric": best.get("tail_metric"),
                    "delta_q20_pnl": best.get("delta_q20_pnl"),
                    "delta_q35_pnl": best.get("delta_q35_pnl"),
                    "pnl_score": best.get("pnl_score"),
                    "objective_score": best.get("objective_score"),
                    "tail_score": best.get("tail_score"),
                    "balanced_score": best.get("balanced_score"),
                    "contains_drift": best.get("contains_drift"),
                    "contains_recent_hit_rate_surprise": best.get("contains_recent_hit_rate_surprise"),
                    "contains_ood": best.get("contains_ood"),
                    "contains_uncertainty": best.get("contains_uncertainty"),
                    "ab_verdict": best.get("ab_verdict"),
                    "rationale": rationale,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["source", "evidence_family", "policy_id"],
        ascending=[True, True, True],
    )


def _feature_family_readout(scorecard_dir: Path | None) -> pd.DataFrame:
    if scorecard_dir is None:
        return pd.DataFrame()
    path = scorecard_dir / "feature_family_readout.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    group_cols = ["arm"]
    agg = frame.groupby(group_cols, dropna=False).agg(
        strategies=("strategy_id", "count"),
        best_validation_strategies=("is_best_validation_arm", "sum"),
        mean_delta_objective_vs_static=("delta_objective_vs_static", "mean"),
        mean_delta_net_pnl_vs_static=("delta_net_pnl_vs_static", "mean"),
        mean_uncertainty_features=("uncertainty_feature_count", "mean"),
        mean_drift_features=("drift_feature_count", "mean"),
        mean_ood_features=("ood_feature_count", "mean"),
        mean_recent_perf_features=("recent_perf_feature_count", "mean"),
    )
    return agg.reset_index().sort_values(
        ["best_validation_strategies", "mean_delta_objective_vs_static"],
        ascending=[False, False],
    )


def _scorecard_marginal_family_ablation(scorecard_dir: Path | None) -> pd.DataFrame:
    if scorecard_dir is None:
        return pd.DataFrame()
    return _marginal_family_ablation_from_scorecards(scorecard_dir)


def _requested_family_verdict(
    marginal_family_ablation: pd.DataFrame,
    gate_summary: pd.DataFrame,
) -> pd.DataFrame:
    if gate_summary.empty:
        readiness: Dict[str, Any] = {}
    else:
        row = gate_summary.iloc[0].to_dict()
        readiness = {
            "drift_finite_row_rate": row.get("drift_finite_row_rate"),
            "recent_hit_rate_surprise_finite_row_rate": row.get("recent_hr_finite_row_rate"),
            "ood_finite_row_rate": row.get("ood_finite_row_rate"),
            "uncertainty_finite_row_rate": row.get("uncertainty_finite_row_rate"),
        }
    attribution = marginal_family_ablation.copy()
    rename_map = {
        "marginal_delta_net_pnl": "delta_net_pnl",
        "marginal_delta_objective": "delta_objective",
        "marginal_delta_full_sl_rate": "delta_full_sl_rate",
        "marginal_delta_q20_pnl": "delta_q20_pnl",
        "marginal_delta_q35_pnl": "delta_q35_pnl",
    }
    for source, target in rename_map.items():
        if source in attribution.columns and target not in attribution.columns:
            attribution[target] = attribution[source]
    return _requested_reliability_family_verdict(attribution, readiness)


def _requested_family_decision_rows(
    requested_family_verdict: pd.DataFrame,
    marginal_family_ablation: pd.DataFrame,
) -> pd.DataFrame:
    if requested_family_verdict.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    marginal = marginal_family_ablation.copy()
    for col in (
        "marginal_delta_net_pnl",
        "marginal_delta_objective",
        "marginal_delta_full_sl_rate",
        "marginal_delta_q20_pnl",
        "marginal_delta_q35_pnl",
        "marginal_scorecard_score",
    ):
        if col in marginal.columns:
            marginal[col] = pd.to_numeric(marginal[col], errors="coerce")

    for _, verdict_row in requested_family_verdict.iterrows():
        family = str(verdict_row.get("family") or "")
        finite_rate = float(verdict_row.get("finite_row_rate", np.nan))
        verdict = str(verdict_row.get("verdict") or "")
        family_rows = (
            marginal.loc[marginal["family"].astype(str).eq(family)].copy()
            if not marginal.empty and "family" in marginal.columns
            else pd.DataFrame()
        )
        if family_rows.empty:
            best = pd.Series(dtype=object)
            best_net = best_objective = best_q20 = best_q35 = best_score = np.nan
            positive_net_share = negative_net_share = np.nan
            positive_tail_share = negative_tail_share = np.nan
            tested_variants = 0
        else:
            sort_cols = [
                col
                for col in ("marginal_scorecard_score", "marginal_delta_net_pnl", "marginal_delta_objective")
                if col in family_rows.columns
            ]
            best = family_rows.sort_values(sort_cols, ascending=False, na_position="last").iloc[0]
            best_net = float(best.get("marginal_delta_net_pnl", np.nan))
            best_objective = float(best.get("marginal_delta_objective", np.nan))
            best_q20 = float(best.get("marginal_delta_q20_pnl", np.nan))
            best_q35 = float(best.get("marginal_delta_q35_pnl", np.nan))
            best_score = float(best.get("marginal_scorecard_score", np.nan))
            net = family_rows.get("marginal_delta_net_pnl", pd.Series(dtype=float))
            tails = family_rows[[c for c in ("marginal_delta_q20_pnl", "marginal_delta_q35_pnl") if c in family_rows]]
            positive_net_share = float(net.gt(0.0).mean()) if len(net) else np.nan
            negative_net_share = float(net.lt(0.0).mean()) if len(net) else np.nan
            if tails.empty:
                positive_tail_share = negative_tail_share = np.nan
            else:
                row_tail_min = tails.min(axis=1, skipna=True)
                positive_tail_share = float(row_tail_min.ge(0.0).mean())
                negative_tail_share = float(row_tail_min.lt(0.0).mean())
            tested_variants = int(len(family_rows))

        if finite_rate < 0.25:
            decision = "fix_coverage_before_use"
            rationale = "Finite coverage is too low for reliable replay or live use."
        elif verdict == "tested_no_clear_lift":
            decision = "diagnostic_only_do_not_default"
            rationale = "The family was tested but did not add clear incremental lift."
        elif verdict == "helpful_in_tests" and np.isfinite(best_net) and best_net > 0.0:
            tail_mixed = np.isfinite(best_q20) and best_q20 < 0.0
            unstable_marginal = np.isfinite(negative_net_share) and negative_net_share > 0.25
            low_coverage = np.isfinite(finite_rate) and finite_rate < 0.75
            if tail_mixed or unstable_marginal or low_coverage:
                decision = "conditional_head_scoped_use"
                rationale = "Positive evidence exists, but coverage, tail, or marginal stability is mixed."
            else:
                decision = "default_keep_candidate"
                rationale = "Positive marginal evidence with acceptable coverage and no obvious tail warning."
        elif verdict == "present_not_yet_tested":
            decision = "test_before_default_use"
            rationale = "Available in the contract but not tested with clear attribution."
        else:
            decision = "diagnostic_only_do_not_default"
            rationale = "No promotion-quality evidence under the current scorecards."

        rows.append(
            {
                "family": family,
                "decision": decision,
                "rationale": rationale,
                "finite_row_rate": finite_rate,
                "verdict": verdict,
                "tested_variants": tested_variants,
                "best_variant": best.get("variant", None),
                "best_baseline_variant": best.get("baseline_variant", None),
                "best_marginal_delta_net_pnl": best_net,
                "best_marginal_delta_objective": best_objective,
                "best_marginal_delta_q20_pnl": best_q20,
                "best_marginal_delta_q35_pnl": best_q35,
                "best_marginal_scorecard_score": best_score,
                "positive_net_variant_share": positive_net_share,
                "negative_net_variant_share": negative_net_share,
                "positive_tail_variant_share": positive_tail_share,
                "negative_tail_variant_share": negative_tail_share,
            }
        )
    return pd.DataFrame(rows)


def _preview_status(row: pd.Series) -> tuple[str, str]:
    rule_id = str(row.get("rule_id") or "")
    if rule_id == "none":
        return "baseline", "Baseline row, not a challenger."
    delta_net = float(row.get("delta_net_pnl", np.nan))
    delta_objective = float(row.get("delta_objective", np.nan))
    delta_trades = float(row.get("delta_trades", np.nan))
    entrant = float(row.get("entrant_trades", np.nan))
    removed = float(row.get("removed_trades", np.nan))
    delta_full_sl = float(row.get("delta_full_sl_rate", np.nan))
    no_trade_change = (
        np.isfinite(delta_trades)
        and abs(delta_trades) < 1e-12
        and (not np.isfinite(entrant) or abs(entrant) < 1e-12)
        and (not np.isfinite(removed) or abs(removed) < 1e-12)
    )
    if no_trade_change and np.isfinite(delta_net) and abs(delta_net) < 1e-9:
        return "no_fresh_binding", "Post-cutoff accepted trades are unchanged versus baseline."
    if np.isfinite(delta_net) and delta_net > 0.0 and (
        not np.isfinite(delta_objective) or delta_objective >= 0.0
    ) and (not np.isfinite(delta_full_sl) or delta_full_sl <= 0.0):
        return "fresh_positive", "Post-cutoff slice improves PnL without a visible tail warning."
    if (
        (np.isfinite(delta_net) and delta_net < 0.0)
        or (np.isfinite(delta_objective) and delta_objective < 0.0)
        or (np.isfinite(delta_full_sl) and delta_full_sl > 0.0)
    ):
        return "fresh_negative_or_tail_warning", "Post-cutoff slice loses PnL, worsens objective, or raises full-SL rate."
    return "fresh_mixed_or_insufficient", "Post-cutoff effect is mixed or too sparse for a directional read."


REQUESTED_RELIABILITY_FAMILIES = (
    "drift",
    "recent_hit_rate_surprise",
    "ood",
    "uncertainty",
)


def _families_from_condition(condition: Any, *, rule_id: Any = "") -> List[str]:
    text = f"{condition or ''} {rule_id or ''}".lower()
    if "any_bad_reliability" in text or "two_of_four" in text:
        return list(REQUESTED_RELIABILITY_FAMILIES)
    families: List[str] = []
    if "recent_hr" in text or "recent_hit_rate" in text or "recent_perf" in text:
        families.append("recent_hit_rate_surprise")
    if "drift" in text:
        families.append("drift")
    if "ood" in text:
        families.append("ood")
    if "uncertainty" in text:
        families.append("uncertainty")
    return [family for family in REQUESTED_RELIABILITY_FAMILIES if family in set(families)]


def _rule_specs_from_dir(root: Path) -> Dict[str, Dict[str, Any]]:
    for name in ("fresh_binding_rules.json", "frozen_reliability_rules.json", "rules.json"):
        path = root / name
        if not path.exists():
            continue
        payload = _read_json(path)
        if isinstance(payload, dict):
            return {
                str(rule_id): dict(spec) if isinstance(spec, dict) else {}
                for rule_id, spec in payload.items()
            }
    return {}


def _postcutoff_preview_rows(preview_dirs: Sequence[Path] | None) -> pd.DataFrame:
    if not preview_dirs:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for item in preview_dirs:
        root = Path(item)
        rule_specs = _rule_specs_from_dir(root)
        candidates = [
            root / "postcutoff_preview" / "postcutoff_preview_summary.csv",
            root / "postcutoff_preview_summary.csv",
        ]
        summary_path = next((path for path in candidates if path.exists()), None)
        if summary_path is None:
            continue
        frame = pd.read_csv(summary_path)
        if frame.empty:
            continue
        manifest_path = summary_path.with_name("postcutoff_preview_manifest.json")
        manifest = _read_json(manifest_path)
        for _, row in frame.iterrows():
            status, rationale = _preview_status(row)
            rule_id = str(row.get("rule_id") or "")
            spec = rule_specs.get(rule_id, {})
            condition = spec.get("condition")
            families = _families_from_condition(condition, rule_id=rule_id)
            rows.append(
                {
                    "preview_dir": str(root),
                    "cutoff": manifest.get("cutoff"),
                    "rule_id": rule_id,
                    "condition": condition,
                    "families": ",".join(families),
                    "head_scope": ",".join(str(head) for head in spec.get("heads", []) or []),
                    "action": spec.get("action"),
                    "value": spec.get("value"),
                    "fresh_status": status,
                    "fresh_status_rationale": rationale,
                    "trades": row.get("trades"),
                    "delta_trades": row.get("delta_trades"),
                    "net_pnl": row.get("net_pnl"),
                    "delta_net_pnl": row.get("delta_net_pnl"),
                    "delta_objective": row.get("delta_objective"),
                    "hit_rate": row.get("hit_rate"),
                    "delta_hit_rate": row.get("delta_hit_rate"),
                    "full_sl_rate": row.get("full_sl_rate"),
                    "delta_full_sl_rate": row.get("delta_full_sl_rate"),
                    "entrant_trades": row.get("entrant_trades"),
                    "removed_trades": row.get("removed_trades"),
                    "entrant_minus_removed_net_pnl": row.get("entrant_minus_removed_net_pnl"),
                    "entrant_minus_removed_hit_rate": row.get("entrant_minus_removed_hit_rate"),
                    "worst_day_delta": row.get("worst_day_delta"),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    status_rank = {
        "fresh_positive": 0,
        "fresh_mixed_or_insufficient": 1,
        "no_fresh_binding": 2,
        "fresh_negative_or_tail_warning": 3,
        "baseline": 4,
    }
    out["_status_rank"] = out["fresh_status"].map(status_rank).fillna(9)
    for col in ("delta_net_pnl", "delta_objective", "delta_full_sl_rate", "delta_trades"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(
        ["_status_rank", "delta_objective", "delta_net_pnl"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns=["_status_rank"])


def _long_window_preview_status(row: pd.Series) -> str:
    delta_net = float(row.get("delta_net_pnl", np.nan))
    delta_objective = float(row.get("delta_objective", np.nan))
    delta_tail = float(row.get("delta_weighted_daily_tail", np.nan))
    worst_week = float(row.get("worst_week_delta", np.nan))
    full_sl = float(row.get("entrant_minus_removed_full_sl_rate", np.nan))
    if np.isfinite(delta_net) and delta_net <= 0.0:
        return "long_window_negative"
    tail_warning = (
        (np.isfinite(delta_objective) and delta_objective < 0.0)
        or (np.isfinite(delta_tail) and delta_tail < 0.0)
        or (np.isfinite(worst_week) and worst_week < 0.0)
        or (np.isfinite(full_sl) and full_sl > 0.0)
    )
    if np.isfinite(delta_net) and delta_net > 0.0 and tail_warning:
        return "long_window_positive_tail_mixed"
    if np.isfinite(delta_net) and delta_net > 0.0:
        return "long_window_positive_tail_clean"
    return "long_window_mixed_or_insufficient"


def _preview_decision_pack_rows(preview_dirs: Sequence[Path] | None) -> pd.DataFrame:
    if not preview_dirs:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for item in preview_dirs:
        root = Path(item)
        summary_path = root / "decision_pack" / "decision_pack_summary.csv"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        if frame.empty:
            continue
        rule_specs = _rule_specs_from_dir(root)
        for _, row in frame.iterrows():
            rule_id = str(row.get("rule_id") or "")
            if rule_id == "none":
                continue
            spec = rule_specs.get(rule_id, {})
            condition = spec.get("condition")
            families = _families_from_condition(condition, rule_id=rule_id)
            rows.append(
                {
                    "preview_dir": str(root),
                    "rule_id": rule_id,
                    "condition": condition,
                    "families": ",".join(families),
                    "head_scope": ",".join(str(head) for head in spec.get("heads", []) or []),
                    "action": spec.get("action"),
                    "value": spec.get("value"),
                    "long_window_status": _long_window_preview_status(row),
                    "days": row.get("days"),
                    "weeks": row.get("weeks"),
                    "active_days": row.get("active_days"),
                    "active_weeks": row.get("active_weeks"),
                    "delta_net_pnl": row.get("delta_net_pnl"),
                    "delta_objective": row.get("delta_objective"),
                    "delta_weighted_daily_tail": row.get("delta_weighted_daily_tail"),
                    "active_positive_week_share": row.get("active_positive_week_share"),
                    "worst_day_delta": row.get("worst_day_delta"),
                    "worst_week_delta": row.get("worst_week_delta"),
                    "entrant_trades": row.get("entrant_trades"),
                    "removed_trades": row.get("removed_trades"),
                    "entrant_minus_removed_net_pnl": row.get("entrant_minus_removed_net_pnl"),
                    "entrant_minus_removed_hit_rate": row.get("entrant_minus_removed_hit_rate"),
                    "entrant_minus_removed_full_sl_rate": row.get("entrant_minus_removed_full_sl_rate"),
                }
            )
    return pd.DataFrame(rows)


def _head_family_scope_recommendation_rows(
    family_coverage_by_head: pd.DataFrame,
    requested_family_decisions: pd.DataFrame,
) -> pd.DataFrame:
    if family_coverage_by_head.empty:
        return pd.DataFrame()
    coverage = family_coverage_by_head.copy()
    if requested_family_decisions.empty:
        decisions = pd.DataFrame({"family": sorted(coverage["family"].dropna().astype(str).unique())})
    else:
        decisions = requested_family_decisions.copy()

    merged = coverage.merge(
        decisions,
        on="family",
        how="left",
        suffixes=("", "_family"),
    )
    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        family = str(row.get("family") or "")
        head = str(row.get("head") or "")
        finite_rate = float(row.get("finite_row_rate", np.nan))
        finite_cell_rate = float(row.get("finite_cell_rate", np.nan))
        family_decision = str(row.get("decision") or "diagnostic_only_do_not_default")
        family_verdict = str(row.get("verdict") or "")
        best_variant = row.get("best_variant")
        variant_compatible = _variant_scope_matches_head(best_variant, head)

        if not np.isfinite(finite_rate) or finite_rate < 0.25:
            recommendation = "coverage_gap_do_not_use"
            rationale = "Head-level finite coverage is too low for this reliability family."
        elif family_decision == "default_keep_candidate" and finite_rate >= 0.75:
            recommendation = "default_keep_for_head"
            rationale = "Family has positive global evidence and strong head-level coverage."
        elif family_decision == "default_keep_candidate":
            recommendation = "head_scoped_candidate"
            rationale = "Family has positive global evidence, but head-level coverage is below the default threshold."
        elif family_decision == "conditional_head_scoped_use" and finite_rate >= 0.75:
            if variant_compatible:
                recommendation = "head_scoped_candidate"
                rationale = "Family is mixed globally, but this head has enough coverage for a scoped A/B."
            else:
                recommendation = "head_scoped_candidate_needs_matched_variant"
                rationale = "Coverage is sufficient, but the best family variant is scoped to a different head set."
        elif family_decision == "conditional_head_scoped_use":
            recommendation = "diagnostic_only_until_coverage_improves"
            rationale = "Family is mixed globally and this head lacks strong enough coverage."
        elif family_decision == "fix_coverage_before_use":
            recommendation = "coverage_gap_do_not_use"
            rationale = "Family-level coverage must be fixed before use."
        else:
            recommendation = "diagnostic_only_do_not_default"
            rationale = "Current scorecards do not show enough incremental lift for this family."

        rows.append(
            {
                "head": head,
                "family": family,
                "recommendation": recommendation,
                "family_decision": family_decision,
                "family_verdict": family_verdict,
                "post_cutoff_rows": row.get("post_cutoff_rows"),
                "finite_row_rate": finite_rate,
                "finite_cell_rate": finite_cell_rate,
                "columns_present": row.get("columns_present"),
                "columns_required": row.get("columns_required"),
                "best_family_variant": best_variant,
                "best_variant_scope_compatible": variant_compatible,
                "best_marginal_delta_net_pnl": row.get("best_marginal_delta_net_pnl"),
                "best_marginal_delta_objective": row.get("best_marginal_delta_objective"),
                "best_marginal_delta_q20_pnl": row.get("best_marginal_delta_q20_pnl"),
                "best_marginal_delta_q35_pnl": row.get("best_marginal_delta_q35_pnl"),
                "rationale": rationale,
            }
        )
    if not rows:
        return pd.DataFrame()
    order = {
        "default_keep_for_head": 0,
        "head_scoped_candidate": 1,
        "head_scoped_candidate_needs_matched_variant": 2,
        "diagnostic_only_until_coverage_improves": 3,
        "diagnostic_only_do_not_default": 4,
        "coverage_gap_do_not_use": 5,
    }
    out = pd.DataFrame(rows)
    out["_order"] = out["recommendation"].map(order).fillna(9)
    return out.sort_values(["_order", "head", "family"]).drop(columns=["_order"])


def _variant_scope_matches_head(variant: Any, head: str) -> bool:
    text = str(variant or "").lower()
    if not text or text in {"nan", "none"}:
        return True
    canonical_head = str(head or "").lower().replace("short_boll", "short_bollinger")
    explicit_scopes = {
        "long_bars": {"long_bars"},
        "long_dist": {"long_dist"},
        "short_asset": {"short_asset"},
        "short_bollinger": {"short_bollinger"},
        "short_boll": {"short_bollinger"},
        "long_heads": {"long_bars", "long_dist"},
        "short_heads": {"short_asset", "short_bollinger"},
    }
    for token, allowed_heads in explicit_scopes.items():
        if token in text and canonical_head not in allowed_heads:
            return False
    if "no_boll" in text and canonical_head == "short_bollinger":
        return False
    if "no_short_asset" in text and canonical_head == "short_asset":
        return False
    return True


def _candidate_tradeoff_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    frame = candidates.copy()
    max_net = float(frame["delta_net_pnl"].max())
    max_objective = float(frame["delta_objective"].max())
    min_worst = float(frame["worst_week_delta"].min())
    worst_scale = max(abs(min_worst), 1.0)
    frame["net_pnl_share_of_best"] = np.where(max_net > 0.0, frame["delta_net_pnl"] / max_net, 0.0)
    frame["objective_share_of_best"] = np.where(
        max_objective > 0.0,
        frame["delta_objective"] / max_objective,
        0.0,
    )
    frame["worst_week_safety"] = np.clip(1.0 + frame["worst_week_delta"] / worst_scale, 0.0, 1.0)
    frame["bootstrap_confidence_min"] = frame[
        ["prob_delta_net_pnl_positive", "prob_delta_objective_positive"]
    ].min(axis=1)
    frame["tail_safety_score"] = (
        0.40 * frame["active_positive_week_share"]
        + 0.35 * frame["worst_week_safety"]
        + 0.25 * frame["bootstrap_confidence_min"]
    )
    frame["balanced_pnl_tail_score"] = (
        0.50 * frame["net_pnl_share_of_best"]
        + 0.25 * frame["objective_share_of_best"]
        + 0.25 * frame["tail_safety_score"]
    )
    dominated = []
    metrics = [
        "delta_net_pnl",
        "delta_objective",
        "active_positive_week_share",
        "worst_week_delta",
        "bootstrap_confidence_min",
    ]
    values = frame[metrics].to_numpy(dtype=float)
    for i, row in enumerate(values):
        is_dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if bool(np.all(other >= row) and np.any(other > row)):
                is_dominated = True
                break
        dominated.append(is_dominated)
    frame["pareto_efficient"] = ~pd.Series(dominated, index=frame.index)
    return frame.sort_values(
        ["pareto_efficient", "balanced_pnl_tail_score", "delta_net_pnl"],
        ascending=[False, False, False],
    )


def _long_window_action_status(row: pd.Series) -> str:
    if float(row.get("delta_net_pnl", 0.0) or 0.0) <= 0.0:
        return "long_window_negative"
    if float(row.get("worst_week_delta", 0.0) or 0.0) < 0.0:
        return "long_window_positive_tail_mixed"
    return "long_window_positive_tail_clean"


def _family_action_impact_rows(
    candidates: pd.DataFrame,
    postcutoff_previews: pd.DataFrame,
    preview_decision_packs: pd.DataFrame,
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rules = bundle.get("rules") or {}

    if not candidates.empty:
        for _, row in candidates.iterrows():
            rule_id = str(row.get("rule_id") or "")
            spec = rules.get(rule_id, {})
            condition = spec.get("condition")
            families = _families_from_condition(condition, rule_id=rule_id)
            entrant_trades = float(row.get("entrant_trades", 0.0) or 0.0)
            removed_trades = float(row.get("removed_trades", 0.0) or 0.0)
            for family in families:
                rows.append(
                    {
                        "evidence_scope": "long_window",
                        "family": family,
                        "rule_id": rule_id,
                        "condition": condition,
                        "head_scope": ",".join(str(head) for head in spec.get("heads", []) or []),
                        "action": spec.get("action"),
                        "value": spec.get("value"),
                        "action_binding": bool(entrant_trades > 0.0 or removed_trades > 0.0),
                        "status": _long_window_action_status(row),
                        "trades": np.nan,
                        "delta_trades": np.nan,
                        "active_weeks": row.get("active_weeks"),
                        "active_positive_week_share": row.get("active_positive_week_share"),
                        "delta_net_pnl": row.get("delta_net_pnl"),
                        "delta_objective": row.get("delta_objective"),
                        "worst_week_delta": row.get("worst_week_delta"),
                        "delta_hit_rate": np.nan,
                        "delta_full_sl_rate": np.nan,
                        "entrant_trades": entrant_trades,
                        "removed_trades": removed_trades,
                        "entrant_minus_removed_net_pnl": row.get("entrant_minus_removed_net_pnl"),
                        "entrant_minus_removed_hit_rate": row.get("entrant_minus_removed_hit_rate"),
                        "entrant_minus_removed_full_sl_rate": row.get(
                            "entrant_minus_removed_full_sl_rate"
                        ),
                    }
                )

    if not preview_decision_packs.empty:
        for _, row in preview_decision_packs.iterrows():
            rule_id = str(row.get("rule_id") or "")
            raw_families = str(row.get("families") or "")
            families = [family for family in raw_families.split(",") if family]
            if not families:
                families = _families_from_condition(row.get("condition"), rule_id=rule_id)
            entrant_trades = float(row.get("entrant_trades", np.nan))
            removed_trades = float(row.get("removed_trades", np.nan))
            action_binding = (
                (np.isfinite(entrant_trades) and entrant_trades > 0.0)
                or (np.isfinite(removed_trades) and removed_trades > 0.0)
                or float(row.get("active_weeks", 0.0) or 0.0) > 0.0
            )
            for family in families:
                rows.append(
                    {
                        "evidence_scope": "long_window_preview_replay",
                        "family": family,
                        "rule_id": rule_id,
                        "condition": row.get("condition"),
                        "head_scope": row.get("head_scope"),
                        "action": row.get("action"),
                        "value": row.get("value"),
                        "action_binding": bool(action_binding),
                        "status": row.get("long_window_status"),
                        "trades": np.nan,
                        "delta_trades": np.nan,
                        "active_weeks": row.get("active_weeks"),
                        "active_positive_week_share": row.get("active_positive_week_share"),
                        "delta_net_pnl": row.get("delta_net_pnl"),
                        "delta_objective": row.get("delta_objective"),
                        "worst_week_delta": row.get("worst_week_delta"),
                        "delta_hit_rate": np.nan,
                        "delta_full_sl_rate": np.nan,
                        "entrant_trades": row.get("entrant_trades"),
                        "removed_trades": row.get("removed_trades"),
                        "entrant_minus_removed_net_pnl": row.get("entrant_minus_removed_net_pnl"),
                        "entrant_minus_removed_hit_rate": row.get("entrant_minus_removed_hit_rate"),
                        "entrant_minus_removed_full_sl_rate": row.get(
                            "entrant_minus_removed_full_sl_rate"
                        ),
                    }
                )

    if not postcutoff_previews.empty:
        for _, row in postcutoff_previews.iterrows():
            rule_id = str(row.get("rule_id") or "")
            if rule_id == "none":
                continue
            raw_families = str(row.get("families") or "")
            families = [family for family in raw_families.split(",") if family]
            if not families:
                families = _families_from_condition(row.get("condition"), rule_id=rule_id)
            delta_trades = float(row.get("delta_trades", np.nan))
            entrant_trades = float(row.get("entrant_trades", np.nan))
            removed_trades = float(row.get("removed_trades", np.nan))
            action_binding = (
                (np.isfinite(delta_trades) and abs(delta_trades) > 1e-12)
                or (np.isfinite(entrant_trades) and entrant_trades > 0.0)
                or (np.isfinite(removed_trades) and removed_trades > 0.0)
            )
            for family in families:
                rows.append(
                    {
                        "evidence_scope": "post_cutoff_preview",
                        "family": family,
                        "rule_id": rule_id,
                        "condition": row.get("condition"),
                        "head_scope": row.get("head_scope"),
                        "action": row.get("action"),
                        "value": row.get("value"),
                        "action_binding": bool(action_binding),
                        "status": row.get("fresh_status"),
                        "trades": row.get("trades"),
                        "delta_trades": row.get("delta_trades"),
                        "active_weeks": np.nan,
                        "active_positive_week_share": np.nan,
                        "delta_net_pnl": row.get("delta_net_pnl"),
                        "delta_objective": row.get("delta_objective"),
                        "worst_week_delta": row.get("worst_day_delta"),
                        "delta_hit_rate": row.get("delta_hit_rate"),
                        "delta_full_sl_rate": row.get("delta_full_sl_rate"),
                        "entrant_trades": row.get("entrant_trades"),
                        "removed_trades": row.get("removed_trades"),
                        "entrant_minus_removed_net_pnl": row.get("entrant_minus_removed_net_pnl"),
                        "entrant_minus_removed_hit_rate": row.get("entrant_minus_removed_hit_rate"),
                        "entrant_minus_removed_full_sl_rate": np.nan,
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sort_cols = ["evidence_scope", "family", "delta_net_pnl"]
    return out.sort_values(sort_cols, ascending=[True, True, False], na_position="last")


def _tail_aversion_sensitivity(tradeoff: pd.DataFrame) -> pd.DataFrame:
    if tradeoff.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    grid = np.round(np.linspace(0.0, 1.0, 11), 2)
    frame = tradeoff.copy()
    frame["pnl_objective_score"] = (
        0.67 * frame["net_pnl_share_of_best"] + 0.33 * frame["objective_share_of_best"]
    )
    for tail_weight in grid:
        scored = frame.copy()
        scored["selection_score"] = (
            (1.0 - float(tail_weight)) * scored["pnl_objective_score"]
            + float(tail_weight) * scored["tail_safety_score"]
        )
        best = scored.sort_values(
            ["selection_score", "pareto_efficient", "delta_net_pnl"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(
            {
                "tail_weight": float(tail_weight),
                "selected_rule_id": best["rule_id"],
                "selected_role": best.get("role"),
                "selection_score": float(best["selection_score"]),
                "delta_net_pnl": float(best["delta_net_pnl"]),
                "delta_objective": float(best["delta_objective"]),
                "tail_safety_score": float(best["tail_safety_score"]),
                "worst_week_delta": float(best["worst_week_delta"]),
                "active_positive_week_share": float(best["active_positive_week_share"]),
            }
        )
    return pd.DataFrame(rows)


def _select_by_tail_weight(tradeoff: pd.DataFrame, tail_weight: float) -> pd.Series:
    frame = tradeoff.copy()
    frame["pnl_objective_score"] = (
        0.67 * frame["net_pnl_share_of_best"] + 0.33 * frame["objective_share_of_best"]
    )
    frame["selection_score"] = (
        (1.0 - float(tail_weight)) * frame["pnl_objective_score"]
        + float(tail_weight) * frame["tail_safety_score"]
    )
    return frame.sort_values(
        ["selection_score", "pareto_efficient", "delta_net_pnl"],
        ascending=[False, False, False],
    ).iloc[0]


def _selection_policy_rows(tradeoff: pd.DataFrame) -> pd.DataFrame:
    if tradeoff.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    policies = [
        ("pnl_dominant", 0.0, "Maximize normalized PnL/objective lift."),
        ("balanced_default", 0.25, "Use PnL as primary objective while giving explicit weight to tail safety."),
        ("tail_aware", 0.50, "Give tail safety equal weight to PnL/objective lift."),
        ("tail_defensive", 0.75, "Prefer tail-safe candidates unless the PnL gap is very large."),
    ]
    for policy_id, tail_weight, rationale in policies:
        selected = _select_by_tail_weight(tradeoff, tail_weight)
        rows.append(
            {
                "policy_id": policy_id,
                "tail_weight": float(tail_weight),
                "selected_rule_id": selected["rule_id"],
                "selected_role": selected.get("role"),
                "selection_score": float(selected["selection_score"]),
                "delta_net_pnl": float(selected["delta_net_pnl"]),
                "delta_objective": float(selected["delta_objective"]),
                "tail_safety_score": float(selected["tail_safety_score"]),
                "tail_clean": bool(selected.get("tail_clean", False)),
                "rationale": rationale,
            }
        )

    tail_clean = tradeoff.loc[tradeoff.get("tail_clean", pd.Series(False, index=tradeoff.index)).astype(bool)]
    if tail_clean.empty:
        selected = tradeoff.sort_values(["tail_safety_score", "delta_net_pnl"], ascending=[False, False]).iloc[0]
        rationale = "No explicit tail-clean candidate exists; choose highest tail-safety score."
    else:
        selected = tail_clean.sort_values(["delta_net_pnl", "delta_objective"], ascending=[False, False]).iloc[0]
        rationale = "Hard constraint: require no negative active weeks and full positive active-week share."
    rows.append(
        {
            "policy_id": "hard_tail_clean",
            "tail_weight": np.nan,
            "selected_rule_id": selected["rule_id"],
            "selected_role": selected.get("role"),
            "selection_score": np.nan,
            "delta_net_pnl": float(selected["delta_net_pnl"]),
            "delta_objective": float(selected["delta_objective"]),
            "tail_safety_score": float(selected["tail_safety_score"]),
            "tail_clean": bool(selected.get("tail_clean", False)),
            "rationale": rationale,
        }
    )
    return pd.DataFrame(rows)


def _freeze_decision_matrix(
    selection_policies: pd.DataFrame,
    candidates: pd.DataFrame,
    postcutoff_previews: pd.DataFrame,
    fresh_ready: bool,
    fresh_blockers: Sequence[str],
) -> pd.DataFrame:
    if selection_policies.empty:
        return pd.DataFrame()
    candidate_lookup = (
        candidates.set_index("rule_id").to_dict(orient="index")
        if not candidates.empty and "rule_id" in candidates.columns
        else {}
    )
    preview_lookup: Dict[str, Dict[str, Any]] = {}
    if not postcutoff_previews.empty and "rule_id" in postcutoff_previews.columns:
        for rule_id, group in postcutoff_previews.groupby("rule_id", dropna=False):
            group = group.copy()
            status_priority = {
                "fresh_positive": 0,
                "fresh_mixed_or_insufficient": 1,
                "no_fresh_binding": 2,
                "fresh_negative_or_tail_warning": 3,
                "baseline": 4,
            }
            group["_priority"] = group["fresh_status"].map(status_priority).fillna(9)
            row = group.sort_values(["_priority", "delta_net_pnl"], ascending=[True, False]).iloc[0]
            preview_lookup[str(rule_id)] = row.to_dict()

    rows: List[Dict[str, Any]] = []
    blocker_text = ";".join(str(item) for item in fresh_blockers if item) or "passed"
    for _, policy in selection_policies.iterrows():
        rule_id = str(policy.get("selected_rule_id") or "")
        candidate = candidate_lookup.get(rule_id, {})
        preview = preview_lookup.get(rule_id, {})
        preview_status = str(preview.get("fresh_status") or "not_previewed")
        research_pass = bool(candidate.get("research_pass", False))
        tail_clean = bool(candidate.get("tail_clean", False))
        if not research_pass:
            recommendation = "reject_research_gate"
            rationale = "The selected rule does not pass long-window/bootstrap research gates."
        elif not fresh_ready:
            if preview_status == "fresh_negative_or_tail_warning":
                recommendation = "do_not_promote_wait_for_clean_fresh"
                rationale = "Formal fresh gate is blocked and available post-cutoff preview is negative or tail-worse."
            elif preview_status == "no_fresh_binding":
                recommendation = "keep_frozen_wait_for_binding"
                rationale = "Long-window evidence passes, but the formal fresh gate is blocked and preview accepted trades did not change."
            else:
                recommendation = "keep_frozen_wait_for_fresh_gate"
                rationale = "Long-window evidence passes, but formal fresh evidence is still below gate requirements."
        elif preview_status == "fresh_negative_or_tail_warning":
            recommendation = "reject_fresh_negative"
            rationale = "Formal gate is ready, but fresh preview shows adverse PnL or tail behavior."
        elif preview_status == "fresh_positive":
            recommendation = "promotion_candidate"
            rationale = "Research gates and fresh preview both support the rule."
        else:
            recommendation = "fresh_ready_needs_binding_review"
            rationale = "Fresh gate is ready but accepted-trade binding evidence is not clearly positive."

        rows.append(
            {
                "policy_id": policy.get("policy_id"),
                "tail_weight": policy.get("tail_weight"),
                "selected_rule_id": rule_id,
                "selected_role": policy.get("selected_role"),
                "recommendation": recommendation,
                "rationale": rationale,
                "fresh_ready": bool(fresh_ready),
                "fresh_blockers": blocker_text,
                "preview_status": preview_status,
                "preview_delta_net_pnl": preview.get("delta_net_pnl", np.nan),
                "preview_delta_objective": preview.get("delta_objective", np.nan),
                "preview_delta_hit_rate": preview.get("delta_hit_rate", np.nan),
                "preview_delta_full_sl_rate": preview.get("delta_full_sl_rate", np.nan),
                "research_pass": research_pass,
                "tail_clean": tail_clean,
                "delta_net_pnl": candidate.get("delta_net_pnl", np.nan),
                "delta_objective": candidate.get("delta_objective", np.nan),
                "active_weeks": candidate.get("active_weeks", np.nan),
                "active_positive_week_share": candidate.get("active_positive_week_share", np.nan),
                "worst_week_delta": candidate.get("worst_week_delta", np.nan),
                "tail_safety_score": policy.get("tail_safety_score", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _tail_aversion_switch_points(tradeoff: pd.DataFrame) -> pd.DataFrame:
    if tradeoff.empty or len(tradeoff) < 2:
        return pd.DataFrame()
    frame = tradeoff.copy()
    frame["pnl_objective_score"] = (
        0.67 * frame["net_pnl_share_of_best"] + 0.33 * frame["objective_share_of_best"]
    )
    rows: List[Dict[str, Any]] = []
    records = frame.to_dict(orient="records")
    for i, left in enumerate(records):
        for right in records[i + 1 :]:
            left_slope = float(left["tail_safety_score"]) - float(left["pnl_objective_score"])
            right_slope = float(right["tail_safety_score"]) - float(right["pnl_objective_score"])
            denom = left_slope - right_slope
            if abs(denom) < 1e-12:
                continue
            switch_weight = (float(right["pnl_objective_score"]) - float(left["pnl_objective_score"])) / denom
            if switch_weight < 0.0 or switch_weight > 1.0:
                continue
            left_score = float(left["pnl_objective_score"]) + switch_weight * left_slope
            rows.append(
                {
                    "rule_a": left["rule_id"],
                    "rule_b": right["rule_id"],
                    "tail_weight_switch": float(switch_weight),
                    "score_at_switch": float(left_score),
                    "rule_a_pnl_objective_score": float(left["pnl_objective_score"]),
                    "rule_a_tail_safety_score": float(left["tail_safety_score"]),
                    "rule_b_pnl_objective_score": float(right["pnl_objective_score"]),
                    "rule_b_tail_safety_score": float(right["tail_safety_score"]),
                }
            )
    return pd.DataFrame(rows).sort_values("tail_weight_switch") if rows else pd.DataFrame()


def _read_csv_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _temporal_stability_rows(decision_pack_dir: Path) -> pd.DataFrame:
    week = _read_csv_optional(decision_pack_dir / "decision_pack_week_deltas.csv")
    month = _read_csv_optional(decision_pack_dir / "decision_pack_month_deltas.csv")
    head = _read_csv_optional(decision_pack_dir / "decision_pack_head_deltas.csv")
    if week.empty and month.empty and head.empty:
        return pd.DataFrame()

    rule_ids = sorted(
        set(week.get("rule_id", pd.Series(dtype=str)).astype(str))
        | set(month.get("rule_id", pd.Series(dtype=str)).astype(str))
        | set(head.get("rule_id", pd.Series(dtype=str)).astype(str))
    )
    rows: List[Dict[str, Any]] = []
    for rule_id in rule_ids:
        row: Dict[str, Any] = {"rule_id": rule_id}
        if not week.empty:
            wf = week.loc[week["rule_id"].astype(str).eq(rule_id)].copy()
            active = wf.loc[
                wf["delta_net_pnl"].abs().gt(1e-12)
                | wf.get("delta_trades", pd.Series(0.0, index=wf.index)).abs().gt(1e-12)
            ]
            row.update(
                {
                    "weeks": int(len(wf)),
                    "active_weeks": int(len(active)),
                    "active_week_share": float(len(active) / len(wf)) if len(wf) else 0.0,
                    "positive_active_week_share": float(active["delta_net_pnl"].gt(0).mean())
                    if len(active)
                    else 0.0,
                    "worst_week_delta": float(wf["delta_net_pnl"].min()) if len(wf) else np.nan,
                    "q10_week_delta": float(wf["delta_net_pnl"].quantile(0.10)) if len(wf) else np.nan,
                    "q25_week_delta": float(wf["delta_net_pnl"].quantile(0.25)) if len(wf) else np.nan,
                    "median_week_delta": float(wf["delta_net_pnl"].median()) if len(wf) else np.nan,
                    "best_week_delta": float(wf["delta_net_pnl"].max()) if len(wf) else np.nan,
                }
            )
        if not month.empty:
            mf = month.loc[month["rule_id"].astype(str).eq(rule_id)].copy()
            active_m = mf.loc[
                mf["delta_net_pnl"].abs().gt(1e-12)
                | mf.get("delta_trades", pd.Series(0.0, index=mf.index)).abs().gt(1e-12)
            ]
            row.update(
                {
                    "months": int(len(mf)),
                    "active_months": int(len(active_m)),
                    "active_month_share": float(len(active_m) / len(mf)) if len(mf) else 0.0,
                    "positive_active_month_share": float(active_m["delta_net_pnl"].gt(0).mean())
                    if len(active_m)
                    else 0.0,
                    "worst_month_delta": float(mf["delta_net_pnl"].min()) if len(mf) else np.nan,
                    "q25_month_delta": float(mf["delta_net_pnl"].quantile(0.25)) if len(mf) else np.nan,
                    "median_month_delta": float(mf["delta_net_pnl"].median()) if len(mf) else np.nan,
                    "best_month_delta": float(mf["delta_net_pnl"].max()) if len(mf) else np.nan,
                }
            )
        if not head.empty:
            hf = head.loc[head["rule_id"].astype(str).eq(rule_id)].copy()
            row.update(
                {
                    "heads": int(len(hf)),
                    "positive_heads": int(hf["delta_net_pnl"].gt(0).sum()) if len(hf) else 0,
                    "negative_heads": int(hf["delta_net_pnl"].lt(0).sum()) if len(hf) else 0,
                    "worst_head_delta": float(hf["delta_net_pnl"].min()) if len(hf) else np.nan,
                    "best_head_delta": float(hf["delta_net_pnl"].max()) if len(hf) else np.nan,
                    "head_delta_sum": float(hf["delta_net_pnl"].sum()) if len(hf) else np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["head_delta_sum", "best_month_delta"], ascending=[False, False])


def _safe_positive_ratio(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    values = values.loc[values.notna()]
    return float(values.gt(0.0).mean()) if len(values) else np.nan


def _long_period_robustness_rows(
    tradeoff: pd.DataFrame,
    temporal_stability: pd.DataFrame,
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    if tradeoff.empty:
        return pd.DataFrame()
    frame = tradeoff.copy()
    if not temporal_stability.empty:
        frame = frame.merge(temporal_stability, on="rule_id", how="left", suffixes=("", "_temporal"))

    rules = bundle.get("rules") or {}
    max_net = float(pd.to_numeric(frame.get("delta_net_pnl", pd.Series(dtype=float)), errors="coerce").max())
    max_obj = float(pd.to_numeric(frame.get("delta_objective", pd.Series(dtype=float)), errors="coerce").max())
    if not np.isfinite(max_net) or max_net <= 0.0:
        max_net = 1.0
    if not np.isfinite(max_obj) or max_obj <= 0.0:
        max_obj = 1.0

    rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        rule_id = str(row.get("rule_id") or "")
        spec = rules.get(rule_id, {})
        families = _families_from_condition(spec.get("condition"), rule_id=rule_id)
        if not families:
            families = ["unmapped"]

        delta_net = float(row.get("delta_net_pnl", 0.0) or 0.0)
        delta_objective = float(row.get("delta_objective", 0.0) or 0.0)
        worst_week = float(row.get("worst_week_delta", np.nan))
        active_positive_week_share = float(row.get("active_positive_week_share", np.nan))
        active_positive_month_share = float(row.get("positive_active_month_share", np.nan))
        positive_heads = float(row.get("positive_heads", np.nan))
        heads = float(row.get("heads", np.nan))
        negative_heads = float(row.get("negative_heads", np.nan))
        tail_clean = bool(row.get("tail_clean", False))
        head_positive_share = positive_heads / heads if np.isfinite(positive_heads) and heads > 0 else np.nan
        week_tail_clean = (
            np.isfinite(worst_week)
            and worst_week >= 0.0
            and np.isfinite(active_positive_week_share)
            and active_positive_week_share >= 1.0
        )

        if delta_net <= 0.0 or delta_objective <= 0.0:
            verdict = "reject_long_period"
        elif week_tail_clean and (not np.isfinite(negative_heads) or negative_heads <= 0):
            verdict = "tail_clean_broadly_consistent"
        elif tail_clean or week_tail_clean:
            verdict = "tail_clean_head_or_month_mixed"
        elif (
            active_positive_week_share >= 0.8
            and (not np.isfinite(head_positive_share) or head_positive_share >= 0.5)
        ):
            verdict = "pnl_strong_tail_mixed"
        else:
            verdict = "positive_incomplete_tail_evidence"

        net_share = max(delta_net / max_net, 0.0)
        objective_share = max(delta_objective / max_obj, 0.0)
        tail_safety = float(row.get("tail_safety_score", 0.0) or 0.0)
        month_share = active_positive_month_share if np.isfinite(active_positive_month_share) else 0.0
        head_share = head_positive_share if np.isfinite(head_positive_share) else 0.0
        robustness_score = (
            0.30 * min(net_share, 1.0)
            + 0.20 * min(objective_share, 1.0)
            + 0.20 * tail_safety
            + 0.15 * month_share
            + 0.15 * head_share
        )

        rows.append(
            {
                "rule_id": rule_id,
                "role": row.get("role"),
                "families": ",".join(families),
                "delta_net_pnl": delta_net,
                "delta_objective": delta_objective,
                "active_weeks": row.get("active_weeks"),
                "active_positive_week_share": active_positive_week_share,
                "worst_week_delta": worst_week,
                "q10_week_delta": row.get("q10_week_delta"),
                "q25_week_delta": row.get("q25_week_delta"),
                "median_week_delta": row.get("median_week_delta"),
                "active_months": row.get("active_months"),
                "positive_active_month_share": active_positive_month_share,
                "worst_month_delta": row.get("worst_month_delta"),
                "median_month_delta": row.get("median_month_delta"),
                "positive_heads": positive_heads,
                "negative_heads": negative_heads,
                "head_positive_share": head_positive_share,
                "worst_head_delta": row.get("worst_head_delta"),
                "tail_safety_score": tail_safety,
                "balanced_pnl_tail_score": row.get("balanced_pnl_tail_score"),
                "robustness_score": float(robustness_score),
                "long_period_verdict": verdict,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["robustness_score", "delta_net_pnl"],
        ascending=[False, False],
        na_position="last",
    )


def _long_period_family_robustness_rows(long_period: pd.DataFrame) -> pd.DataFrame:
    if long_period.empty or "families" not in long_period.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    exploded = long_period.copy()
    exploded["family"] = exploded["families"].fillna("").astype(str).str.split(",")
    exploded = exploded.explode("family")
    exploded = exploded.loc[exploded["family"].astype(str).ne("")]
    for family, group in exploded.groupby("family", dropna=False):
        sorted_group = group.sort_values(
            ["robustness_score", "delta_net_pnl"],
            ascending=[False, False],
            na_position="last",
        )
        best = sorted_group.iloc[0]
        rows.append(
            {
                "family": str(family),
                "tested_rules": int(len(group)),
                "best_rule_id": best.get("rule_id"),
                "best_verdict": best.get("long_period_verdict"),
                "best_delta_net_pnl": best.get("delta_net_pnl"),
                "best_delta_objective": best.get("delta_objective"),
                "best_active_positive_week_share": best.get("active_positive_week_share"),
                "best_worst_week_delta": best.get("worst_week_delta"),
                "best_positive_active_month_share": best.get("positive_active_month_share"),
                "best_head_positive_share": best.get("head_positive_share"),
                "mean_delta_net_pnl": float(pd.to_numeric(group["delta_net_pnl"], errors="coerce").mean()),
                "positive_rule_share": _safe_positive_ratio(group, "delta_net_pnl"),
                "tail_clean_rule_share": float(
                    group["long_period_verdict"].astype(str).str.contains("tail_clean").mean()
                ),
                "max_robustness_score": best.get("robustness_score"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_robustness_score", "best_delta_net_pnl"],
        ascending=[False, False],
        na_position="last",
    )


def _monthly_delta_rows(decision_pack_dir: Path) -> pd.DataFrame:
    month = _read_csv_optional(decision_pack_dir / "decision_pack_month_deltas.csv")
    if month.empty:
        return pd.DataFrame()
    cols = [
        "rule_id",
        "month",
        "baseline_trades",
        "trades",
        "delta_trades",
        "baseline_net_pnl",
        "net_pnl",
        "delta_net_pnl",
        "baseline_hit_rate",
        "hit_rate",
        "delta_hit_rate",
        "baseline_full_sl_rate",
        "full_sl_rate",
        "delta_full_sl_rate",
    ]
    out = month[[c for c in cols if c in month.columns]].copy()
    return out.sort_values(["rule_id", "month"])


def _worst_week_rows(decision_pack_dir: Path, *, n: int = 5) -> pd.DataFrame:
    week = _read_csv_optional(decision_pack_dir / "decision_pack_week_deltas.csv")
    if week.empty:
        return pd.DataFrame()
    cols = [
        "rule_id",
        "week",
        "baseline_trades",
        "trades",
        "delta_trades",
        "baseline_net_pnl",
        "net_pnl",
        "delta_net_pnl",
        "baseline_hit_rate",
        "hit_rate",
        "delta_hit_rate",
        "baseline_full_sl_rate",
        "full_sl_rate",
        "delta_full_sl_rate",
    ]
    rows: List[pd.DataFrame] = []
    for rule_id, group in week.groupby("rule_id", dropna=False):
        active = group.loc[
            group["delta_net_pnl"].abs().gt(1e-12)
            | group.get("delta_trades", pd.Series(0.0, index=group.index)).abs().gt(1e-12)
        ]
        source = active if not active.empty else group
        worst = source.sort_values(["delta_net_pnl", "week"], ascending=[True, True]).head(n)
        rows.append(worst[[c for c in cols if c in worst.columns]].copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _markdown_table(frame: pd.DataFrame, cols: List[str]) -> str:
    if frame.empty:
        return "_No rows._"
    use_cols = [c for c in cols if c in frame.columns]
    return frame[use_cols].round(6).to_markdown(index=False)


def _multiwindow_selection_rows(selection_dirs: Sequence[Path] | None) -> pd.DataFrame:
    if not selection_dirs:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for directory in selection_dirs:
        manifest_path = directory / "multiwindow_selection.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        summary_path = directory / "multiwindow_candidate_summary.csv"
        summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        summary_lookup = (
            summary.set_index("rule_id").to_dict(orient="index")
            if not summary.empty and "rule_id" in summary.columns
            else {}
        )

        def add_row(kind: str, record: Mapping[str, Any] | None) -> None:
            if not record:
                return
            rule_id = str(record.get("rule_id", ""))
            base = dict(summary_lookup.get(rule_id, {}))
            base.update(record)
            rows.append(
                {
                    "selection_dir": str(directory),
                    "selection_kind": kind,
                    "rule_id": rule_id,
                    "profile_pass": kind == "recommended" or kind.startswith("profile:"),
                    "core_pnl_tail_gate_count": base.get("core_pnl_tail_gate_count"),
                    "core_strict_tail_gate_count": base.get("core_strict_tail_gate_count"),
                    "core_min_delta_objective": base.get("core_min_delta_objective"),
                    "core_min_delta_net_pnl": base.get("core_min_delta_net_pnl"),
                    "full_delta_objective": base.get("full_delta_objective"),
                    "full_delta_net_pnl": base.get("full_delta_net_pnl"),
                    "full_delta_weekly_q20": base.get("full_delta_weekly_q20"),
                    "full_delta_weighted_daily_tail": base.get("full_delta_weighted_daily_tail"),
                    "june_delta_objective": base.get("june_delta_objective"),
                    "june_delta_net_pnl": base.get("june_delta_net_pnl"),
                    "entrant_minus_removed_net_pnl": base.get("entrant_minus_removed_net_pnl"),
                    "entrant_minus_removed_hit_rate": base.get("entrant_minus_removed_hit_rate"),
                }
            )

        add_row("recommended", manifest.get("recommended"))
        add_row("best_by_sort_order", manifest.get("best_by_sort_order"))
        for profile, record in (manifest.get("profile_recommendations") or {}).items():
            add_row(f"profile:{profile}", record)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop_duplicates(
        subset=["selection_dir", "selection_kind", "rule_id"],
        keep="first",
    )
    return out.sort_values(["selection_dir", "selection_kind", "full_delta_net_pnl"], ascending=[True, True, False])


def _rule_spec_from_bundle(bundle: Mapping[str, Any], rule_id: str) -> Dict[str, Any]:
    spec = dict((bundle.get("rules") or {}).get(str(rule_id), {}) or {})
    condition = spec.get("condition")
    return {
        "role": spec.get("role"),
        "heads": ",".join(map(str, spec.get("heads") or [])),
        "condition": condition,
        "families": ",".join(_families_from_condition(condition, rule_id=rule_id)),
        "action": spec.get("action"),
        "value": spec.get("value"),
    }


def _multiwindow_rule_specs(selection_dirs: Sequence[Path] | None) -> Dict[tuple[str, str], Dict[str, Any]]:
    specs: Dict[tuple[str, str], Dict[str, Any]] = {}
    if not selection_dirs:
        return specs
    for directory in selection_dirs:
        manifest_path = directory / "multiwindow_selection.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        attribution_dir = Path(str(manifest.get("attribution_dir") or ""))
        summary_path = attribution_dir / "conditional_filter_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        if "rule_id" not in summary.columns:
            continue
        for _, row in summary.iterrows():
            rule_id = str(row.get("rule_id"))
            try:
                parsed = json.loads(str(row.get("rule_spec") or "{}"))
            except json.JSONDecodeError:
                parsed = {}
            specs[(str(directory), rule_id)] = {
                "role": None,
                "heads": ",".join(map(str, parsed.get("heads") or [])),
                "condition": parsed.get("condition"),
                "families": ",".join(_families_from_condition(parsed.get("condition"), rule_id=rule_id)),
                "action": parsed.get("action"),
                "value": parsed.get("value"),
            }
    return specs


def _candidate_registry_rows(
    candidates: pd.DataFrame,
    multiwindow_selection: pd.DataFrame,
    bundle: Mapping[str, Any],
    selection_dirs: Sequence[Path] | None,
    fresh_ready: bool,
    fresh_blockers: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    blockers = ";".join(map(str, fresh_blockers))
    for _, row in candidates.iterrows():
        rule_id = str(row.get("rule_id", ""))
        spec = _rule_spec_from_bundle(bundle, rule_id)
        if bool(row.get("research_pass")):
            if bool(row.get("tail_clean")):
                state = "frozen_tail_candidate_ready" if fresh_ready else "frozen_tail_candidate_wait_fresh"
            else:
                state = "frozen_pnl_candidate_ready" if fresh_ready else "frozen_pnl_candidate_wait_fresh"
        else:
            state = "frozen_candidate_research_failed"
        rows.append(
            {
                "rule_id": rule_id,
                "candidate_stage": "frozen_bootstrap_candidate",
                "candidate_state": state,
                "profile_pass": bool(row.get("research_pass")),
                "fresh_ready": bool(fresh_ready),
                "fresh_blockers": blockers,
                **spec,
                "delta_net_pnl": row.get("delta_net_pnl"),
                "delta_objective": row.get("delta_objective"),
                "active_weeks": row.get("active_weeks"),
                "active_positive_week_share": row.get("active_positive_week_share"),
                "worst_week_delta": row.get("worst_week_delta"),
                "tail_clean": bool(row.get("tail_clean")),
                "core_min_delta_objective": np.nan,
                "full_delta_net_pnl": np.nan,
                "full_delta_objective": np.nan,
                "full_delta_weekly_q20": np.nan,
                "full_delta_weighted_daily_tail": np.nan,
                "june_delta_net_pnl": np.nan,
                "entrant_minus_removed_net_pnl": row.get("entrant_minus_removed_net_pnl"),
                "entrant_minus_removed_hit_rate": row.get("entrant_minus_removed_hit_rate"),
            }
        )

    specs = _multiwindow_rule_specs(selection_dirs)
    if not multiwindow_selection.empty:
        passed_multiwindow_keys = {
            (str(row.get("selection_dir")), str(row.get("rule_id")))
            for _, row in multiwindow_selection.iterrows()
            if bool(row.get("profile_pass"))
        }
        for _, row in multiwindow_selection.iterrows():
            kind = str(row.get("selection_kind") or "")
            rule_id = str(row.get("rule_id") or "")
            profile_pass = bool(row.get("profile_pass"))
            key = (str(row.get("selection_dir")), rule_id)
            if not profile_pass and key in passed_multiwindow_keys:
                continue
            if kind == "recommended":
                stage = "multiwindow_recommended"
            elif kind.startswith("profile:"):
                stage = "multiwindow_profile"
            else:
                stage = "multiwindow_diagnostic"
            if profile_pass:
                state = "multiwindow_research_candidate_needs_freeze_pack"
            else:
                state = "diagnostic_only_profile_failed"
            spec = specs.get((str(row.get("selection_dir")), rule_id), {})
            rows.append(
                {
                    "rule_id": rule_id,
                    "candidate_stage": stage,
                    "candidate_state": state,
                    "profile_pass": profile_pass,
                    "fresh_ready": bool(fresh_ready),
                    "fresh_blockers": blockers,
                    **spec,
                    "delta_net_pnl": np.nan,
                    "delta_objective": np.nan,
                    "active_weeks": np.nan,
                    "active_positive_week_share": np.nan,
                    "worst_week_delta": np.nan,
                    "tail_clean": False,
                    "core_min_delta_objective": row.get("core_min_delta_objective"),
                    "full_delta_net_pnl": row.get("full_delta_net_pnl"),
                    "full_delta_objective": row.get("full_delta_objective"),
                    "full_delta_weekly_q20": row.get("full_delta_weekly_q20"),
                    "full_delta_weighted_daily_tail": row.get("full_delta_weighted_daily_tail"),
                    "june_delta_net_pnl": row.get("june_delta_net_pnl"),
                    "entrant_minus_removed_net_pnl": row.get("entrant_minus_removed_net_pnl"),
                    "entrant_minus_removed_hit_rate": row.get("entrant_minus_removed_hit_rate"),
                }
            )

    if not rows:
        return pd.DataFrame()
    order = {
        "frozen_tail_candidate_wait_fresh": 0,
        "frozen_pnl_candidate_wait_fresh": 1,
        "frozen_tail_candidate_ready": 0,
        "frozen_pnl_candidate_ready": 1,
        "multiwindow_research_candidate_needs_freeze_pack": 2,
        "diagnostic_only_profile_failed": 3,
        "frozen_candidate_research_failed": 4,
    }
    out = pd.DataFrame(rows)
    out["_order"] = out["candidate_state"].map(order).fillna(9)
    return out.sort_values(["_order", "candidate_stage", "rule_id"]).drop(columns=["_order"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--decision-pack-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scorecard-dir", type=Path, default=None)
    parser.add_argument("--postcutoff-preview-dir", type=Path, action="append", default=None)
    parser.add_argument("--multiwindow-selection-dir", type=Path, action="append", default=None)
    parser.add_argument("--min-bootstrap-prob", type=float, default=0.95)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = _read_json(args.bundle)
    decision = pd.read_csv(args.decision_pack_dir / "decision_pack_summary.csv")
    bootstrap = pd.read_csv(args.bootstrap_dir / "bootstrap_confidence_summary.csv")
    gate = _read_json(args.gate_dir / "frozen_reliability_gate_manifest.json")
    candidates = _candidate_rows(decision, bootstrap, bundle, min_bootstrap_prob=args.min_bootstrap_prob)
    fresh_ready, fresh_blockers = _status_from_gate(gate)
    gate_summary = _gate_summary_rows(gate)
    family_coverage_by_head = _nearest_source_family_coverage_rows(gate)
    fresh_gaps = _fresh_evidence_gap_rows(bundle, gate)
    promotion_blockers = _promotion_blocker_rows(fresh_gaps)
    scorecard_best = _best_scorecard_rows(args.scorecard_dir)
    reliability_ab_scorecards = _reliability_ab_scorecard_rows(args.scorecard_dir)
    reliability_ab_frontier = _reliability_ab_selection_frontier(reliability_ab_scorecards)
    family_readout = _feature_family_readout(args.scorecard_dir)
    marginal_family_ablation = _scorecard_marginal_family_ablation(args.scorecard_dir)
    requested_family_verdict = _requested_family_verdict(marginal_family_ablation, gate_summary)
    requested_family_decisions = _requested_family_decision_rows(
        requested_family_verdict,
        marginal_family_ablation,
    )
    head_family_scope_recommendations = _head_family_scope_recommendation_rows(
        family_coverage_by_head,
        requested_family_decisions,
    )
    tradeoff = _candidate_tradeoff_rows(candidates)
    sensitivity = _tail_aversion_sensitivity(tradeoff)
    selection_policies = _selection_policy_rows(tradeoff)
    switch_points = _tail_aversion_switch_points(tradeoff)
    temporal_stability = _temporal_stability_rows(args.decision_pack_dir)
    long_period_robustness = _long_period_robustness_rows(tradeoff, temporal_stability, bundle)
    long_period_family_robustness = _long_period_family_robustness_rows(long_period_robustness)
    monthly_deltas = _monthly_delta_rows(args.decision_pack_dir)
    worst_weeks = _worst_week_rows(args.decision_pack_dir)
    postcutoff_previews = _postcutoff_preview_rows(args.postcutoff_preview_dir)
    preview_decision_packs = _preview_decision_pack_rows(args.postcutoff_preview_dir)
    candidate_rule_ids = set(candidates.get("rule_id", pd.Series(dtype=str)).astype(str))
    preview_decision_packs_for_action = (
        preview_decision_packs.loc[
            ~preview_decision_packs.get("rule_id", pd.Series(dtype=str)).astype(str).isin(candidate_rule_ids)
        ].copy()
        if not preview_decision_packs.empty
        else preview_decision_packs
    )
    family_action_impact = _family_action_impact_rows(
        candidates,
        postcutoff_previews,
        preview_decision_packs_for_action,
        bundle,
    )
    multiwindow_selection = _multiwindow_selection_rows(args.multiwindow_selection_dir)
    candidate_registry = _candidate_registry_rows(
        candidates,
        multiwindow_selection,
        bundle,
        args.multiwindow_selection_dir,
        fresh_ready,
        fresh_blockers,
    )
    freeze_decision_matrix = _freeze_decision_matrix(
        selection_policies,
        candidates,
        postcutoff_previews,
        fresh_ready,
        fresh_blockers,
    )
    research_ready = bool(candidates["research_pass"].all()) if not candidates.empty else False
    production_ready = bool(research_ready and fresh_ready)
    pnl_candidate = candidates.sort_values(["delta_net_pnl", "delta_objective"], ascending=[False, False]).iloc[0]
    tail_candidate = candidates.sort_values(
        ["tail_clean", "worst_week_delta", "delta_net_pnl"],
        ascending=[False, False, False],
    ).iloc[0]
    payload = {
        "generated_by": Path(__file__).name,
        "bundle": str(args.bundle),
        "decision_pack_dir": str(args.decision_pack_dir),
        "bootstrap_dir": str(args.bootstrap_dir),
        "gate_dir": str(args.gate_dir),
        "research_ready": research_ready,
        "fresh_ready": fresh_ready,
        "production_ready": production_ready,
        "fresh_blockers": fresh_blockers,
        "gate_summary": gate_summary.iloc[0].to_dict() if not gate_summary.empty else {},
        "family_coverage_by_head": family_coverage_by_head.to_dict(orient="records"),
        "fresh_evidence_gaps": fresh_gaps.to_dict(orient="records"),
        "promotion_blockers": promotion_blockers.to_dict(orient="records"),
        "scorecard_dir": str(args.scorecard_dir) if args.scorecard_dir is not None else None,
        "scorecard_best": scorecard_best.to_dict(orient="records"),
        "reliability_ab_scorecards": reliability_ab_scorecards.to_dict(orient="records"),
        "reliability_ab_frontier": reliability_ab_frontier.to_dict(orient="records"),
        "marginal_family_ablation": marginal_family_ablation.to_dict(orient="records"),
        "requested_family_verdict": requested_family_verdict.to_dict(orient="records"),
        "requested_family_decisions": requested_family_decisions.to_dict(orient="records"),
        "head_family_scope_recommendations": head_family_scope_recommendations.to_dict(orient="records"),
        "candidate_tradeoff": tradeoff.to_dict(orient="records"),
        "tail_aversion_sensitivity": sensitivity.to_dict(orient="records"),
        "selection_policies": selection_policies.to_dict(orient="records"),
        "freeze_decision_matrix": freeze_decision_matrix.to_dict(orient="records"),
        "tail_aversion_switch_points": switch_points.to_dict(orient="records"),
        "temporal_stability": temporal_stability.to_dict(orient="records"),
        "long_period_robustness": long_period_robustness.to_dict(orient="records"),
        "long_period_family_robustness": long_period_family_robustness.to_dict(orient="records"),
        "monthly_deltas": monthly_deltas.to_dict(orient="records"),
        "worst_weeks": worst_weeks.to_dict(orient="records"),
        "postcutoff_previews": postcutoff_previews.to_dict(orient="records"),
        "preview_decision_packs": preview_decision_packs.to_dict(orient="records"),
        "family_action_impact": family_action_impact.to_dict(orient="records"),
        "multiwindow_selection": multiwindow_selection.to_dict(orient="records"),
        "candidate_registry": candidate_registry.to_dict(orient="records"),
        "pnl_candidate": pnl_candidate.to_dict(),
        "tail_candidate": tail_candidate.to_dict(),
    }
    candidates.to_csv(args.out_dir / "frozen_reliability_candidate_status.csv", index=False)
    gate_summary.to_csv(args.out_dir / "frozen_reliability_gate_summary.csv", index=False)
    if not family_coverage_by_head.empty:
        family_coverage_by_head.to_csv(
            args.out_dir / "frozen_reliability_family_coverage_by_head.csv",
            index=False,
        )
    if not head_family_scope_recommendations.empty:
        head_family_scope_recommendations.to_csv(
            args.out_dir / "frozen_reliability_head_family_scope_recommendations.csv",
            index=False,
        )
    fresh_gaps.to_csv(args.out_dir / "frozen_reliability_fresh_evidence_gaps.csv", index=False)
    promotion_blockers.to_csv(args.out_dir / "frozen_reliability_promotion_blockers.csv", index=False)
    tradeoff.to_csv(args.out_dir / "frozen_reliability_candidate_tradeoff.csv", index=False)
    sensitivity.to_csv(args.out_dir / "frozen_reliability_tail_aversion_sensitivity.csv", index=False)
    selection_policies.to_csv(args.out_dir / "frozen_reliability_selection_policies.csv", index=False)
    if not freeze_decision_matrix.empty:
        freeze_decision_matrix.to_csv(args.out_dir / "frozen_reliability_freeze_decision_matrix.csv", index=False)
    switch_points.to_csv(args.out_dir / "frozen_reliability_tail_aversion_switch_points.csv", index=False)
    temporal_stability.to_csv(args.out_dir / "frozen_reliability_temporal_stability.csv", index=False)
    long_period_robustness.to_csv(args.out_dir / "frozen_reliability_long_period_robustness.csv", index=False)
    long_period_family_robustness.to_csv(
        args.out_dir / "frozen_reliability_long_period_family_robustness.csv",
        index=False,
    )
    monthly_deltas.to_csv(args.out_dir / "frozen_reliability_monthly_deltas.csv", index=False)
    worst_weeks.to_csv(args.out_dir / "frozen_reliability_worst_weeks.csv", index=False)
    if not preview_decision_packs.empty:
        preview_decision_packs.to_csv(
            args.out_dir / "frozen_reliability_preview_decision_packs.csv",
            index=False,
        )
    if not family_action_impact.empty:
        family_action_impact.to_csv(args.out_dir / "frozen_reliability_family_action_impact.csv", index=False)
    if not multiwindow_selection.empty:
        multiwindow_selection.to_csv(args.out_dir / "frozen_reliability_multiwindow_selection.csv", index=False)
    if not candidate_registry.empty:
        candidate_registry.to_csv(args.out_dir / "frozen_reliability_candidate_registry.csv", index=False)
    if not postcutoff_previews.empty:
        postcutoff_previews.to_csv(args.out_dir / "frozen_reliability_postcutoff_previews.csv", index=False)
    if not scorecard_best.empty:
        scorecard_best.to_csv(args.out_dir / "frozen_reliability_scorecard_best.csv", index=False)
    if not reliability_ab_scorecards.empty:
        reliability_ab_scorecards.to_csv(
            args.out_dir / "frozen_reliability_ab_scorecard_comparison.csv",
            index=False,
        )
    if not reliability_ab_frontier.empty:
        reliability_ab_frontier.to_csv(
            args.out_dir / "frozen_reliability_ab_selection_frontier.csv",
            index=False,
        )
    if not family_readout.empty:
        family_readout.to_csv(args.out_dir / "frozen_reliability_feature_family_readout.csv", index=False)
    if not marginal_family_ablation.empty:
        marginal_family_ablation.to_csv(args.out_dir / "frozen_reliability_marginal_family_ablation.csv", index=False)
    if not requested_family_verdict.empty:
        requested_family_verdict.to_csv(args.out_dir / "frozen_reliability_requested_family_verdict.csv", index=False)
    if not requested_family_decisions.empty:
        requested_family_decisions.to_csv(
            args.out_dir / "frozen_reliability_requested_family_decisions.csv",
            index=False,
        )
    (args.out_dir / "frozen_reliability_status.json").write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
    status_rows = pd.DataFrame(
        [
            {"check": "research_ready", "status": research_ready, "detail": "all frozen candidates pass long-period/bootstrap gates"},
            {"check": "fresh_ready", "status": fresh_ready, "detail": ";".join(fresh_blockers) or "passed"},
            {"check": "production_ready", "status": production_ready, "detail": "requires both research_ready and fresh_ready"},
        ]
    )
    status_rows.to_csv(args.out_dir / "frozen_reliability_status_checks.csv", index=False)
    lines = [
        "# Frozen Reliability Challenger Status",
        "",
        f"Bundle: `{args.bundle}`",
        f"Research ready: `{research_ready}`",
        f"Fresh/live ready: `{fresh_ready}`",
        f"Production ready: `{production_ready}`",
        "",
        "## Status Checks",
        "",
        status_rows.to_markdown(index=False),
        "",
        "## Candidates",
        "",
        _markdown_table(
            candidates,
            [
                "rule_id",
                "role",
                "delta_net_pnl",
                "delta_objective",
                "active_weeks",
                "active_positive_week_share",
                "worst_week_delta",
                "prob_delta_net_pnl_positive",
                "prob_delta_objective_positive",
                "research_pass",
                "tail_clean",
            ],
        ),
        "",
        "## Selection",
        "",
        f"- PnL candidate: `{pnl_candidate['rule_id']}`.",
        f"- Tail candidate: `{tail_candidate['rule_id']}`.",
        "",
        "## PnL/Tail Tradeoff",
        "",
        _markdown_table(
            tradeoff,
            [
                "rule_id",
                "role",
                "delta_net_pnl",
                "delta_objective",
                "active_positive_week_share",
                "worst_week_delta",
                "net_pnl_share_of_best",
                "objective_share_of_best",
                "tail_safety_score",
                "balanced_pnl_tail_score",
                "pareto_efficient",
            ],
        ),
        "",
        "## Tail Aversion Sensitivity",
        "",
        _markdown_table(
            sensitivity,
            [
                "tail_weight",
                "selected_rule_id",
                "selected_role",
                "selection_score",
                "delta_net_pnl",
                "delta_objective",
                "tail_safety_score",
                "worst_week_delta",
                "active_positive_week_share",
            ],
        ),
        "",
        "## Selection Policies",
        "",
        _markdown_table(
            selection_policies,
            [
                "policy_id",
                "tail_weight",
                "selected_rule_id",
                "selected_role",
                "selection_score",
                "delta_net_pnl",
                "delta_objective",
                "tail_safety_score",
                "tail_clean",
                "rationale",
            ],
        ),
        "",
        "## Freeze Decision Matrix",
        "",
        _markdown_table(
            freeze_decision_matrix,
            [
                "policy_id",
                "tail_weight",
                "selected_rule_id",
                "selected_role",
                "recommendation",
                "fresh_ready",
                "preview_status",
                "preview_delta_net_pnl",
                "preview_delta_objective",
                "research_pass",
                "tail_clean",
                "delta_net_pnl",
                "delta_objective",
                "active_weeks",
                "active_positive_week_share",
                "worst_week_delta",
                "rationale",
            ],
        ),
        "",
        "## Candidate Registry",
        "",
        _markdown_table(
            candidate_registry,
            [
                "rule_id",
                "candidate_stage",
                "candidate_state",
                "profile_pass",
                "fresh_ready",
                "heads",
                "condition",
                "families",
                "action",
                "value",
                "delta_net_pnl",
                "delta_objective",
                "active_positive_week_share",
                "worst_week_delta",
                "core_min_delta_objective",
                "full_delta_net_pnl",
                "full_delta_objective",
                "full_delta_weekly_q20",
                "full_delta_weighted_daily_tail",
                "june_delta_net_pnl",
                "entrant_minus_removed_net_pnl",
                "entrant_minus_removed_hit_rate",
                "tail_clean",
                "fresh_blockers",
            ],
        ),
        "",
        "## Tail Aversion Switch Points",
        "",
        _markdown_table(
            switch_points,
            [
                "rule_a",
                "rule_b",
                "tail_weight_switch",
                "score_at_switch",
                "rule_a_pnl_objective_score",
                "rule_a_tail_safety_score",
                "rule_b_pnl_objective_score",
                "rule_b_tail_safety_score",
            ],
        ),
        "",
        "## Temporal Stability",
        "",
        _markdown_table(
            temporal_stability,
            [
                "rule_id",
                "weeks",
                "active_weeks",
                "positive_active_week_share",
                "worst_week_delta",
                "q10_week_delta",
                "q25_week_delta",
                "months",
                "active_months",
                "positive_active_month_share",
                "worst_month_delta",
                "median_month_delta",
                "heads",
                "positive_heads",
                "negative_heads",
                "worst_head_delta",
                "head_delta_sum",
            ],
        ),
        "",
        "## Long-Period Robustness",
        "",
        _markdown_table(
            long_period_robustness,
            [
                "rule_id",
                "role",
                "families",
                "delta_net_pnl",
                "delta_objective",
                "active_weeks",
                "active_positive_week_share",
                "worst_week_delta",
                "q10_week_delta",
                "q25_week_delta",
                "active_months",
                "positive_active_month_share",
                "positive_heads",
                "negative_heads",
                "head_positive_share",
                "tail_safety_score",
                "robustness_score",
                "long_period_verdict",
            ],
        ),
        "",
        "## Long-Period Family Robustness",
        "",
        _markdown_table(
            long_period_family_robustness,
            [
                "family",
                "tested_rules",
                "best_rule_id",
                "best_verdict",
                "best_delta_net_pnl",
                "best_delta_objective",
                "best_active_positive_week_share",
                "best_worst_week_delta",
                "best_positive_active_month_share",
                "best_head_positive_share",
                "mean_delta_net_pnl",
                "positive_rule_share",
                "tail_clean_rule_share",
                "max_robustness_score",
            ],
        ),
        "",
        "## Monthly Deltas",
        "",
        _markdown_table(
            monthly_deltas,
            [
                "rule_id",
                "month",
                "baseline_trades",
                "trades",
                "delta_trades",
                "delta_net_pnl",
                "delta_hit_rate",
                "delta_full_sl_rate",
            ],
        ),
        "",
        "## Worst Active Weeks",
        "",
        _markdown_table(
            worst_weeks,
            [
                "rule_id",
                "week",
                "baseline_trades",
                "trades",
                "delta_trades",
                "delta_net_pnl",
                "delta_hit_rate",
                "delta_full_sl_rate",
            ],
        ),
        "",
        "## Requested Family Action Impact",
        "",
        _markdown_table(
            family_action_impact,
            [
                "evidence_scope",
                "family",
                "rule_id",
                "condition",
                "head_scope",
                "action_binding",
                "status",
                "active_weeks",
                "active_positive_week_share",
                "trades",
                "delta_trades",
                "delta_net_pnl",
                "delta_objective",
                "worst_week_delta",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "entrant_trades",
                "removed_trades",
                "entrant_minus_removed_net_pnl",
                "entrant_minus_removed_hit_rate",
            ],
        ),
        "",
        "## Multi-Window Selection Evidence",
        "",
        _markdown_table(
            multiwindow_selection,
            [
                "selection_kind",
                "rule_id",
                "profile_pass",
                "core_pnl_tail_gate_count",
                "core_strict_tail_gate_count",
                "core_min_delta_objective",
                "core_min_delta_net_pnl",
                "full_delta_objective",
                "full_delta_net_pnl",
                "full_delta_weekly_q20",
                "full_delta_weighted_daily_tail",
                "june_delta_objective",
                "june_delta_net_pnl",
                "entrant_minus_removed_net_pnl",
                "entrant_minus_removed_hit_rate",
            ],
        ),
        "",
        "## Post-Cutoff Preview Evidence",
        "",
        _markdown_table(
            postcutoff_previews,
            [
                "preview_dir",
                "cutoff",
                "rule_id",
                "fresh_status",
                "trades",
                "delta_trades",
                "delta_net_pnl",
                "delta_objective",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "entrant_trades",
                "removed_trades",
                "entrant_minus_removed_net_pnl",
                "fresh_status_rationale",
            ],
        ),
        "",
        "## Fresh Gate Evidence",
        "",
        _markdown_table(
            gate_summary,
            [
                "ready_sources",
                "ran_gate",
                "post_cutoff_rows",
                "policy_action_rows",
                "policy_outcome_rows",
                "required_head_outcome_gaps",
                "drift_finite_row_rate",
                "recent_hr_finite_row_rate",
                "ood_finite_row_rate",
                "uncertainty_finite_row_rate",
            ],
        ),
        "",
        "## Reliability Family Coverage By Head",
        "",
        _markdown_table(
            family_coverage_by_head,
            [
                "family",
                "head",
                "post_cutoff_rows",
                "columns_present",
                "columns_required",
                "finite_rows",
                "finite_row_rate",
                "finite_cell_rate",
            ],
        ),
        "",
        "## Fresh Evidence Gaps",
        "",
        _markdown_table(
            fresh_gaps,
            ["gate", "head", "observed", "required", "deficit", "pass"],
        ),
        "",
        "## Promotion Blockers",
        "",
        _markdown_table(
            promotion_blockers,
            ["blocker", "head", "observed", "required", "deficit", "severity", "next_action"],
        ),
        "",
        "## Reliability Family Evidence",
        "",
        _markdown_table(
            scorecard_best,
            [
                "source",
                "variant",
                "family",
                "delta_net_pnl",
                "delta_objective",
                "hit_rate",
                "full_sl_rate",
                "tail_metric",
                "scorecard_score",
            ],
        ),
        "",
        "## A/B Reliability Scorecard Comparison",
        "",
        _markdown_table(
            reliability_ab_scorecards,
            [
                "source",
                "evidence_family",
                "variant",
                "family",
                "delta_net_pnl",
                "delta_objective",
                "delta_full_sl_rate",
                "tail_metric",
                "delta_q20_pnl",
                "delta_q35_pnl",
                "scorecard_score",
                "contains_drift",
                "contains_recent_hit_rate_surprise",
                "contains_ood",
                "contains_uncertainty",
                "ab_verdict",
            ],
        ),
        "",
        "## A/B Same-Scope Selection Frontier",
        "",
        _markdown_table(
            reliability_ab_frontier,
            [
                "source",
                "evidence_family",
                "policy_id",
                "selected_variant",
                "family",
                "delta_net_pnl",
                "delta_objective",
                "delta_full_sl_rate",
                "tail_metric",
                "delta_q20_pnl",
                "delta_q35_pnl",
                "pnl_score",
                "objective_score",
                "tail_score",
                "balanced_score",
                "contains_drift",
                "contains_recent_hit_rate_surprise",
                "contains_ood",
                "contains_uncertainty",
                "ab_verdict",
                "rationale",
            ],
        ),
        "",
        "## Requested Reliability Family Verdict",
        "",
        _markdown_table(
            requested_family_verdict,
            [
                "family",
                "finite_row_rate",
                "tested_in_scorecards",
                "best_long_window_delta_net_pnl",
                "best_tail_objective_delta",
                "best_q20_delta_pnl",
                "positive_head_count",
                "tested_head_count",
                "verdict",
            ],
        ),
        "",
        "## Requested Reliability Family Decisions",
        "",
        _markdown_table(
            requested_family_decisions,
            [
                "family",
                "decision",
                "finite_row_rate",
                "tested_variants",
                "best_variant",
                "best_marginal_delta_net_pnl",
                "best_marginal_delta_objective",
                "best_marginal_delta_q20_pnl",
                "best_marginal_delta_q35_pnl",
                "positive_net_variant_share",
                "negative_net_variant_share",
                "positive_tail_variant_share",
                "negative_tail_variant_share",
                "rationale",
            ],
        ),
        "",
        "## Head-Scoped Reliability Family Recommendations",
        "",
        _markdown_table(
            head_family_scope_recommendations,
            [
                "head",
                "family",
                "recommendation",
                "finite_row_rate",
                "finite_cell_rate",
                "family_decision",
                "family_verdict",
                "best_family_variant",
                "best_variant_scope_compatible",
                "best_marginal_delta_net_pnl",
                "best_marginal_delta_objective",
                "best_marginal_delta_q20_pnl",
                "best_marginal_delta_q35_pnl",
                "rationale",
            ],
        ),
        "",
        "## Marginal Reliability Family Ablation",
        "",
        _markdown_table(
            marginal_family_ablation,
            [
                "family",
                "comparison_type",
                "source_table",
                "evidence_family",
                "variant",
                "variant_family",
                "baseline_variant",
                "marginal_delta_net_pnl",
                "marginal_delta_objective",
                "marginal_delta_full_sl_rate",
                "marginal_delta_q20_pnl",
                "marginal_delta_q35_pnl",
                "marginal_scorecard_score",
            ],
        ),
        "",
        "## Feature Family Readout",
        "",
        _markdown_table(
            family_readout,
            [
                "arm",
                "strategies",
                "best_validation_strategies",
                "mean_delta_objective_vs_static",
                "mean_delta_net_pnl_vs_static",
                "mean_uncertainty_features",
                "mean_drift_features",
                "mean_ood_features",
                "mean_recent_perf_features",
            ],
        ),
        "",
        "## Fresh Blockers",
        "",
        "\n".join(f"- `{blocker}`" for blocker in fresh_blockers) if fresh_blockers else "_None._",
    ]
    (args.out_dir / "frozen_reliability_status_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(_json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
