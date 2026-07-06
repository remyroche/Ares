#!/usr/bin/env python3
"""Compare frozen reliability candidates over the long-period evidence window.

This script is artifact-only. It consumes status directories produced by
``audit_frozen_reliability_challenger_status.py`` and optionally a profile
selection directory produced by ``select_frozen_reliability_candidate_profile``.
It does not replay trades or refit models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


OBJECTIVE_NAME = "avg_week_delta_net_pnl + 0.7*q35_day_delta_net_pnl + 0.3*q20_day_delta_net_pnl"


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
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _source_name(path: Path) -> str:
    name = path.name
    for suffix in (
        "_status_20260701",
        "_challenger_status_allroots_20260701",
        "_candidate_status",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)] or name
    return name


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_status_json(status_dir: Path) -> Dict[str, Any]:
    path = status_dir / "frozen_reliability_status.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    gate_summary = data.get("gate_summary") if isinstance(data.get("gate_summary"), dict) else {}
    blockers = data.get("fresh_blockers")
    if isinstance(blockers, list):
        blocker_text = ";".join(str(item) for item in blockers)
    elif blockers is None:
        blocker_text = ""
    else:
        blocker_text = str(blockers)
    return {
        "research_ready": bool(data.get("research_ready", False)),
        "fresh_ready": bool(data.get("fresh_ready", False)),
        "production_ready": bool(data.get("production_ready", False)),
        "fresh_blockers": blocker_text,
        "post_cutoff_rows": gate_summary.get("post_cutoff_rows"),
        "policy_action_rows": gate_summary.get("policy_action_rows"),
        "policy_outcome_rows": gate_summary.get("policy_outcome_rows"),
    }


def _with_source(frame: pd.DataFrame, status_dir: Path, status: Dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out.insert(0, "source_dir", str(status_dir))
    out.insert(1, "source_name", _source_name(status_dir))
    for col, value in status.items():
        if col not in out.columns:
            out[col] = value
    return out


def _load_status_dirs(status_dirs: Sequence[Path]) -> Dict[str, pd.DataFrame]:
    candidates: List[pd.DataFrame] = []
    monthly: List[pd.DataFrame] = []
    stability: List[pd.DataFrame] = []
    worst: List[pd.DataFrame] = []
    for status_dir in status_dirs:
        status = _read_status_json(status_dir)
        candidates.append(
            _with_source(
                _read_csv(status_dir / "frozen_reliability_candidate_status.csv"),
                status_dir,
                status,
            )
        )
        monthly.append(
            _with_source(
                _read_csv(status_dir / "frozen_reliability_monthly_deltas.csv"),
                status_dir,
                status,
            )
        )
        stability.append(
            _with_source(
                _read_csv(status_dir / "frozen_reliability_temporal_stability.csv"),
                status_dir,
                status,
            )
        )
        worst.append(
            _with_source(
                _read_csv(status_dir / "frozen_reliability_worst_weeks.csv"),
                status_dir,
                status,
            )
        )
    return {
        "candidates": pd.concat([f for f in candidates if not f.empty], ignore_index=True, sort=False)
        if any(not f.empty for f in candidates)
        else pd.DataFrame(),
        "monthly": pd.concat([f for f in monthly if not f.empty], ignore_index=True, sort=False)
        if any(not f.empty for f in monthly)
        else pd.DataFrame(),
        "stability": pd.concat([f for f in stability if not f.empty], ignore_index=True, sort=False)
        if any(not f.empty for f in stability)
        else pd.DataFrame(),
        "worst": pd.concat([f for f in worst if not f.empty], ignore_index=True, sort=False)
        if any(not f.empty for f in worst)
        else pd.DataFrame(),
    }


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "rule_id" not in frame.columns:
        return frame
    keys = [col for col in ("source_name", "rule_id", "role") if col in frame.columns]
    return frame.drop_duplicates(subset=keys, keep="first") if keys else frame


def _load_decision_packs(decision_pack_dirs: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for decision_pack_dir in decision_pack_dirs:
        frame = _read_csv(decision_pack_dir / "decision_pack_summary.csv")
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "decision_pack_dir", str(decision_pack_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "rule_id" not in out.columns:
        return pd.DataFrame()
    keep_cols = [
        col
        for col in (
            "rule_id",
            "delta_avg_week_pnl",
            "delta_weighted_daily_tail",
            "delta_daily_q20",
            "delta_daily_q35",
            "delta_weekly_q20",
            "delta_weekly_q35",
        )
        if col in out.columns
    ]
    out = out[keep_cols].drop_duplicates("rule_id", keep="first")
    for col in keep_cols:
        if col != "rule_id":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_gate_rows(gate_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for gate_dir in gate_dirs:
        manifest_path = gate_dir / "frozen_reliability_gate_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        bundle_path = Path(str(manifest.get("bundle", "")))
        rule_ids: List[str] = []
        if bundle_path.exists():
            try:
                bundle = json.loads(bundle_path.read_text())
                rules = bundle.get("rules") if isinstance(bundle.get("rules"), dict) else {}
                rule_ids = [str(rule_id) for rule_id in rules]
            except Exception:
                rule_ids = []
        nearest = manifest.get("nearest_source") if isinstance(manifest.get("nearest_source"), dict) else {}
        failed_deficits = [
            row
            for row in manifest.get("readiness_deficits", [])
            if isinstance(row, dict) and not bool(row.get("pass", False))
        ]
        blockers = str(nearest.get("rejection_reasons") or "")
        if not blockers and failed_deficits:
            blockers = ";".join(str(row.get("gate", "")) for row in failed_deficits if row.get("gate"))
        current_fresh_ready = bool(manifest.get("ran_gate", False)) and int(manifest.get("ready_sources") or 0) > 0
        for rule_id in rule_ids:
            rows.append(
                {
                    "rule_id": rule_id,
                    "current_gate_dir": str(gate_dir),
                    "current_gate_cutoff": manifest.get("cutoff"),
                    "current_fresh_ready": current_fresh_ready,
                    "current_fresh_blockers": blockers,
                    "current_post_cutoff_rows": nearest.get("post_cutoff_rows"),
                    "current_policy_action_rows": nearest.get("policy_action_rows_estimate"),
                    "current_policy_outcome_rows": nearest.get("policy_outcome_rows_estimate"),
                    "current_ready_sources": manifest.get("ready_sources"),
                    "current_ran_gate": manifest.get("ran_gate"),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    for col in (
        "current_post_cutoff_rows",
        "current_policy_action_rows",
        "current_policy_outcome_rows",
        "current_ready_sources",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_gate_deficits(gate_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for gate_dir in gate_dirs:
        manifest_path = gate_dir / "frozen_reliability_gate_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue

        bundle_path = Path(str(manifest.get("bundle", "")))
        rule_ids: List[str] = []
        if bundle_path.exists():
            try:
                bundle = json.loads(bundle_path.read_text())
                rules = bundle.get("rules") if isinstance(bundle.get("rules"), dict) else {}
                rule_ids = [str(rule_id) for rule_id in rules]
            except Exception:
                rule_ids = []
        if not rule_ids:
            rule_ids = [""]

        for deficit in manifest.get("readiness_deficits", []):
            if not isinstance(deficit, dict):
                continue
            for rule_id in rule_ids:
                rows.append(
                    {
                        "rule_id": rule_id,
                        "gate_dir": str(gate_dir),
                        "gate_cutoff": manifest.get("cutoff"),
                        "gate": deficit.get("gate"),
                        "head": deficit.get("head"),
                        "observed": deficit.get("observed"),
                        "required": deficit.get("required"),
                        "deficit": deficit.get("deficit"),
                        "pass": bool(deficit.get("pass", False)),
                        "source_path": str(manifest_path),
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    for col in ("observed", "required", "deficit"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    sort_cols = [col for col in ("pass", "gate_cutoff", "rule_id", "gate", "head") if col in out.columns]
    return out.sort_values(sort_cols, ascending=[True] * len(sort_cols)) if sort_cols else out


def _load_gate_overrides(gate_dirs: Sequence[Path]) -> pd.DataFrame:
    rows = _load_gate_rows(gate_dirs)
    if rows.empty:
        return rows
    return rows.drop_duplicates("rule_id", keep="last")


def _candidate_summary(
    candidates: pd.DataFrame,
    stability: pd.DataFrame,
    decision_pack_summary: pd.DataFrame | None = None,
    gate_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = _dedupe(candidates).copy()
    if not stability.empty:
        stability_cols = [
            col
            for col in (
                "source_name",
                "rule_id",
                "weeks",
                "active_weeks",
                "q10_week_delta",
                "q25_week_delta",
                "median_week_delta",
                "active_months",
                "positive_active_month_share",
                "worst_month_delta",
                "head_positive_share",
            )
            if col in stability.columns
        ]
        if {"source_name", "rule_id"}.issubset(stability_cols):
            out = out.merge(
                stability[stability_cols].drop_duplicates(["source_name", "rule_id"]),
                on=["source_name", "rule_id"],
                how="left",
                suffixes=("", "_stability"),
            )
    if decision_pack_summary is not None and not decision_pack_summary.empty:
        out = out.merge(decision_pack_summary, on="rule_id", how="left", suffixes=("", "_decision_pack"))
    if gate_overrides is not None and not gate_overrides.empty:
        out = out.merge(gate_overrides, on="rule_id", how="left", suffixes=("", "_current_gate"))
    numeric_cols = [
        "delta_net_pnl",
        "delta_objective",
        "delta_avg_week_pnl",
        "delta_weighted_daily_tail",
        "delta_daily_q20",
        "delta_daily_q35",
        "delta_weekly_q20",
        "delta_weekly_q35",
        "active_positive_week_share",
        "worst_week_delta",
        "entrant_minus_removed_hit_rate",
        "entrant_minus_removed_full_sl_rate",
        "q10_week_delta",
        "q25_week_delta",
        "median_week_delta",
        "positive_active_month_share",
        "worst_month_delta",
        "post_cutoff_rows",
        "policy_action_rows",
        "policy_outcome_rows",
        "current_post_cutoff_rows",
        "current_policy_action_rows",
        "current_policy_outcome_rows",
        "current_ready_sources",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["delta_objective", "delta_net_pnl"], ascending=[False, False])


def _month_matrix(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    cols = [
        col
        for col in (
            "source_name",
            "rule_id",
            "month",
            "delta_net_pnl",
            "delta_trades",
            "delta_hit_rate",
            "delta_full_sl_rate",
        )
        if col in monthly.columns
    ]
    out = monthly[cols].copy()
    for col in ("delta_net_pnl", "delta_trades", "delta_hit_rate", "delta_full_sl_rate"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values([c for c in ("month", "source_name", "rule_id") if c in out.columns])


def _profile_selection(profile_selection_dir: Path | None) -> pd.DataFrame:
    if profile_selection_dir is None:
        return pd.DataFrame()
    path = profile_selection_dir / "frozen_reliability_profile_selection.csv"
    return _read_csv(path)


def _load_family_evidence(family_evidence_dirs: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for evidence_dir in family_evidence_dirs:
        long_candidates = [
            evidence_dir / "diagnostic_family_long_window_summary.csv",
            evidence_dir / "diagnostic_family_evidence" / "diagnostic_family_long_window_summary.csv",
        ]
        path = next((candidate for candidate in long_candidates if candidate.exists()), None)
        if path is None:
            continue
        frame = _read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        monthly_candidates = [
            evidence_dir / "diagnostic_family_monthly_summary.csv",
            evidence_dir / "diagnostic_family_evidence" / "diagnostic_family_monthly_summary.csv",
        ]
        monthly_path = next((candidate for candidate in monthly_candidates if candidate.exists()), None)
        monthly = _read_csv(monthly_path) if monthly_path is not None else pd.DataFrame()
        if not monthly.empty and "label" in monthly.columns:
            monthly_cols = [
                col
                for col in (
                    "label",
                    "months",
                    "min_month_delta_net_pnl",
                    "apr_jun_delta_net_pnl",
                    "june_delta_net_pnl",
                )
                if col in monthly.columns
            ]
            frame = frame.merge(
                monthly[monthly_cols].drop_duplicates("label"),
                on="label",
                how="left",
                suffixes=("", "_monthly"),
            )
        frame.insert(0, "family_evidence_dir", str(evidence_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in (
        "daily_weekly_objective",
        "avg_week_delta_net_pnl",
        "q35_day_delta_net_pnl",
        "q20_day_delta_net_pnl",
        "sum_delta_net_pnl",
        "positive_week_share",
        "mean_day_full_sl_delta",
        "june_net_delta",
        "june_full_sl_delta",
        "positive_month_count",
        "months",
        "min_month_delta_net_pnl",
        "apr_jun_delta_net_pnl",
        "june_delta_net_pnl",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    sort_cols = [col for col in ("daily_weekly_objective", "sum_delta_net_pnl") if col in out.columns]
    if sort_cols:
        return out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return out


def _candidate_month_stats(month_matrix: pd.DataFrame) -> pd.DataFrame:
    if month_matrix.empty or "rule_id" not in month_matrix.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    month_col = "month" if "month" in month_matrix.columns else None
    for rule_id, group in month_matrix.groupby("rule_id", sort=False):
        pnl = pd.to_numeric(group.get("delta_net_pnl"), errors="coerce")
        months = int(group[month_col].nunique()) if month_col is not None else int(len(group))
        apr_jun_mask = (
            group[month_col].astype(str).isin(["2026-04", "2026-05", "2026-06"])
            if month_col is not None
            else pd.Series(False, index=group.index)
        )
        june_mask = (
            group[month_col].astype(str).eq("2026-06")
            if month_col is not None
            else pd.Series(False, index=group.index)
        )
        rows.append(
            {
                "rule_id": rule_id,
                "months": months,
                "positive_month_count": int((pnl > 0.0).sum()),
                "positive_month_share": float((pnl > 0.0).mean()) if len(pnl) else np.nan,
                "min_month_delta_net_pnl": float(pnl.min()) if len(pnl) else np.nan,
                "apr_jun_delta_net_pnl": float(pd.to_numeric(group.loc[apr_jun_mask, "delta_net_pnl"], errors="coerce").sum()),
                "june_delta_net_pnl": float(pd.to_numeric(group.loc[june_mask, "delta_net_pnl"], errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows)


def _profile_membership(profile: pd.DataFrame) -> Dict[str, str]:
    if profile.empty or "selected_rule_id" not in profile.columns or "profile" not in profile.columns:
        return {}
    memberships: Dict[str, List[str]] = {}
    for _, row in profile.iterrows():
        rule_id = str(row.get("selected_rule_id", ""))
        if not rule_id:
            continue
        memberships.setdefault(rule_id, []).append(str(row.get("profile", "")))
    return {rule_id: ",".join(sorted(p for p in profiles if p)) for rule_id, profiles in memberships.items()}


def _bool_series(frame: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[col]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    normalized = values.astype(str).str.lower()
    return normalized.isin(("true", "1", "yes")).fillna(default)


def _numeric_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _decision_matrix(summary: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary.copy()
    profile_map = _profile_membership(profile)
    out["selected_profiles"] = out["rule_id"].astype(str).map(profile_map).fillna("")

    delta_net = _numeric_series(out, "delta_net_pnl")
    delta_objective = _numeric_series(out, "delta_objective")
    avg_week = _numeric_series(out, "delta_avg_week_pnl")
    avg_week = avg_week.where(avg_week.notna(), delta_objective)
    weighted_daily_tail = _numeric_series(out, "delta_weighted_daily_tail")
    daily_q20 = _numeric_series(out, "delta_daily_q20")
    daily_q35 = _numeric_series(out, "delta_daily_q35")
    q25_week = _numeric_series(out, "q25_week_delta")
    worst_week = _numeric_series(out, "worst_week_delta")
    replacement_hr = _numeric_series(out, "entrant_minus_removed_hit_rate")
    replacement_sl = _numeric_series(out, "entrant_minus_removed_full_sl_rate")
    active_week_share = _numeric_series(out, "active_positive_week_share")

    out["pnl_objective_pass"] = (delta_net > 0.0) & (delta_objective > 0.0) & (avg_week > 0.0)
    out["daily_tail_pass"] = (weighted_daily_tail >= 0.0) & (daily_q20 >= 0.0) & (daily_q35 >= 0.0)
    out["weekly_tail_pass"] = (q25_week >= 0.0) & (worst_week >= 0.0)
    out["replacement_quality_pass"] = (replacement_hr > 0.0) & (replacement_sl <= 0.0)
    out["long_period_pass"] = (
        out["pnl_objective_pass"]
        & out["daily_tail_pass"]
        & (active_week_share >= 0.75)
        & out["replacement_quality_pass"]
    )
    out["tail_robust_pass"] = out["long_period_pass"] & out["weekly_tail_pass"]
    out["research_ready_pass"] = _bool_series(out, "research_ready") | _bool_series(out, "research_pass")
    current_fresh = _bool_series(out, "current_fresh_ready")
    stale_fresh = _bool_series(out, "fresh_ready")
    out["fresh_ready_pass"] = current_fresh.where(out.get("current_fresh_ready", pd.Series(index=out.index)).notna(), stale_fresh)
    out["production_ready_pass"] = _bool_series(out, "production_ready")

    decisions: List[str] = []
    for _, row in out.iterrows():
        if bool(row.get("production_ready_pass")) and bool(row.get("tail_robust_pass")):
            decisions.append("production_ready_tail_robust")
        elif bool(row.get("production_ready_pass")) and bool(row.get("long_period_pass")):
            decisions.append("production_ready_tail_mixed")
        elif bool(row.get("fresh_ready_pass")) and bool(row.get("tail_robust_pass")):
            decisions.append("fresh_ready_tail_robust")
        elif bool(row.get("fresh_ready_pass")) and bool(row.get("long_period_pass")):
            decisions.append("fresh_ready_tail_mixed")
        elif bool(row.get("tail_robust_pass")):
            decisions.append("research_tail_robust_wait_fresh")
        elif bool(row.get("long_period_pass")):
            decisions.append("research_tail_mixed_wait_fresh")
        elif bool(row.get("pnl_objective_pass")):
            decisions.append("diagnostic_positive_incomplete_gates")
        else:
            decisions.append("reject_or_monitor")
    out["decision_state"] = decisions

    cols = [
        "rule_id",
        "source_name",
        "selected_profiles",
        "decision_state",
        "pnl_objective_pass",
        "daily_tail_pass",
        "weekly_tail_pass",
        "replacement_quality_pass",
        "long_period_pass",
        "tail_robust_pass",
        "research_ready_pass",
        "fresh_ready_pass",
        "production_ready_pass",
        "delta_net_pnl",
        "delta_objective",
        "delta_avg_week_pnl",
        "delta_weighted_daily_tail",
        "q25_week_delta",
        "worst_week_delta",
        "entrant_minus_removed_hit_rate",
        "entrant_minus_removed_full_sl_rate",
        "fresh_blockers",
        "current_fresh_blockers",
    ]
    return out[[col for col in cols if col in out.columns]]


def _comparison_scope(evidence_type: str, item_id: Any, family: Any) -> str:
    if evidence_type == "frozen_candidate":
        return "frozen_candidate"
    text = str(item_id or "").lower()
    if "longbars" in text or "long_bars" in text:
        return "long_bars"
    if "longdist" in text or "long_dist" in text:
        return "long_dist"
    if "shortasset" in text or "short_asset" in text:
        return "short_asset"
    if "shortboll" in text or "short_boll" in text:
        return "short_bollinger"
    if text == "combined" or "combined" in text:
        return "combined"
    return str(family or "unknown")


def _promotion_frontier(
    summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    family_evidence: pd.DataFrame,
    month_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    decision_lookup: Dict[str, Dict[str, Any]] = {}
    if not decision_matrix.empty and "rule_id" in decision_matrix.columns:
        decision_lookup = {
            str(row.get("rule_id", "")): row.to_dict()
            for _, row in decision_matrix.iterrows()
        }
    month_lookup: Dict[str, Dict[str, Any]] = {}
    month_stats = _candidate_month_stats(month_matrix)
    if not month_stats.empty:
        month_lookup = {
            str(row.get("rule_id", "")): row.to_dict()
            for _, row in month_stats.iterrows()
        }

    if not summary.empty:
        for _, row in summary.iterrows():
            rule_id = str(row.get("rule_id", ""))
            decision = decision_lookup.get(rule_id, {})
            month = month_lookup.get(rule_id, {})
            state = str(decision.get("decision_state", ""))
            if bool(decision.get("tail_robust_pass", False)):
                verdict = "frozen_tail_robust_wait_fresh"
            elif bool(decision.get("long_period_pass", False)):
                verdict = "frozen_pnl_positive_tail_mixed_wait_fresh"
            elif bool(decision.get("pnl_objective_pass", False)):
                verdict = "frozen_pnl_positive_incomplete_tail"
            else:
                verdict = "reject_or_monitor"
            rows.append(
                {
                    "evidence_type": "frozen_candidate",
                    "item_id": rule_id,
                    "family": row.get("source_name", ""),
                    "verdict": verdict,
                    "decision_state": state,
                    "objective_score": row.get("delta_objective"),
                    "net_pnl_delta": row.get("delta_net_pnl"),
                    "avg_week_delta": row.get("delta_avg_week_pnl"),
                    "q35_day_delta": row.get("delta_daily_q35"),
                    "q20_day_delta": row.get("delta_daily_q20"),
                    "q25_week_delta": row.get("q25_week_delta"),
                    "worst_week_delta": row.get("worst_week_delta"),
                    "full_sl_delta": row.get("entrant_minus_removed_full_sl_rate"),
                    "positive_week_share": row.get("active_positive_week_share"),
                    "months": month.get("months", row.get("active_months")),
                    "positive_month_count": month.get("positive_month_count"),
                    "positive_month_share": month.get("positive_month_share", row.get("positive_active_month_share")),
                    "min_month_delta_net_pnl": month.get("min_month_delta_net_pnl", row.get("worst_month_delta")),
                    "apr_jun_delta_net_pnl": month.get("apr_jun_delta_net_pnl"),
                    "june_delta_net_pnl": month.get("june_delta_net_pnl"),
                    "fresh_ready": decision.get("fresh_ready_pass", row.get("fresh_ready", False)),
                    "production_ready": decision.get("production_ready_pass", row.get("production_ready", False)),
                    "notes": decision.get("current_fresh_blockers", row.get("fresh_blockers", "")),
                }
            )

    if not family_evidence.empty:
        for _, row in family_evidence.iterrows():
            objective = pd.to_numeric(pd.Series([row.get("daily_weekly_objective")]), errors="coerce").iloc[0]
            pnl = pd.to_numeric(pd.Series([row.get("sum_delta_net_pnl")]), errors="coerce").iloc[0]
            q35 = pd.to_numeric(pd.Series([row.get("q35_day_delta_net_pnl")]), errors="coerce").iloc[0]
            q20 = pd.to_numeric(pd.Series([row.get("q20_day_delta_net_pnl")]), errors="coerce").iloc[0]
            full_sl = pd.to_numeric(pd.Series([row.get("mean_day_full_sl_delta")]), errors="coerce").iloc[0]
            positive_week_share = pd.to_numeric(pd.Series([row.get("positive_week_share")]), errors="coerce").iloc[0]
            pnl_pass = bool(pd.notna(objective) and pd.notna(pnl) and objective > 0.0 and pnl > 0.0)
            daily_tail_clean = bool(
                pd.notna(q35)
                and pd.notna(q20)
                and pd.notna(full_sl)
                and q35 >= 0.0
                and q20 >= 0.0
                and full_sl <= 0.0
            )
            recurrence_pass = bool(pd.notna(positive_week_share) and positive_week_share >= 0.5)
            if pnl_pass and daily_tail_clean and recurrence_pass:
                verdict = "family_pnl_tail_clean_research"
            elif pnl_pass and recurrence_pass:
                verdict = "family_pnl_positive_tail_tradeoff"
            elif pnl_pass:
                verdict = "family_pnl_positive_low_recurrence"
            else:
                verdict = "reject_or_monitor"
            rows.append(
                {
                    "evidence_type": "diagnostic_family_ablation",
                    "item_id": row.get("label", ""),
                    "family": row.get("diagnostic_family", ""),
                    "verdict": verdict,
                    "decision_state": "",
                    "objective_score": objective,
                    "net_pnl_delta": pnl,
                    "avg_week_delta": row.get("avg_week_delta_net_pnl"),
                    "q35_day_delta": q35,
                    "q20_day_delta": q20,
                    "q25_week_delta": np.nan,
                    "worst_week_delta": np.nan,
                    "full_sl_delta": full_sl,
                    "positive_week_share": positive_week_share,
                    "months": row.get("months"),
                    "positive_month_count": row.get("positive_month_count"),
                    "positive_month_share": (
                        row.get("positive_month_count") / row.get("months")
                        if pd.notna(row.get("positive_month_count")) and pd.notna(row.get("months")) and row.get("months")
                        else np.nan
                    ),
                    "min_month_delta_net_pnl": row.get("min_month_delta_net_pnl"),
                    "apr_jun_delta_net_pnl": row.get("apr_jun_delta_net_pnl"),
                    "june_delta_net_pnl": row.get("june_delta_net_pnl", row.get("june_net_delta")),
                    "fresh_ready": False,
                    "production_ready": False,
                    "notes": "diagnostic family evidence; not a frozen deployable candidate",
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["comparison_scope"] = [
        _comparison_scope(str(row.get("evidence_type", "")), row.get("item_id"), row.get("family"))
        for _, row in out.iterrows()
    ]
    for col in (
        "objective_score",
        "net_pnl_delta",
        "avg_week_delta",
        "q35_day_delta",
        "q20_day_delta",
        "q25_week_delta",
        "worst_week_delta",
        "full_sl_delta",
        "positive_week_share",
        "months",
        "positive_month_count",
        "positive_month_share",
        "min_month_delta_net_pnl",
        "apr_jun_delta_net_pnl",
        "june_delta_net_pnl",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["full_sl_improvement"] = -pd.to_numeric(out.get("full_sl_delta"), errors="coerce")
    dominance_metrics = [
        "objective_score",
        "net_pnl_delta",
        "q35_day_delta",
        "q20_day_delta",
        "q25_week_delta",
        "worst_week_delta",
        "full_sl_improvement",
        "positive_week_share",
        "positive_month_share",
        "min_month_delta_net_pnl",
        "apr_jun_delta_net_pnl",
        "june_delta_net_pnl",
    ]
    out["pareto_dominated"] = False
    out["dominated_by"] = ""
    for _, group in out.groupby(["evidence_type", "comparison_scope"], sort=False):
        idxs = list(group.index)
        for idx in idxs:
            row = out.loc[idx]
            best_dominator = ""
            best_dominator_score = -np.inf
            for other_idx in idxs:
                if other_idx == idx:
                    continue
                other = out.loc[other_idx]
                comparable = [
                    metric
                    for metric in dominance_metrics
                    if metric in out.columns and pd.notna(row.get(metric)) and pd.notna(other.get(metric))
                ]
                if len(comparable) < 3:
                    continue
                row_vals = row[comparable].astype(float)
                other_vals = other[comparable].astype(float)
                weakly_better = bool((other_vals >= row_vals - 1e-12).all())
                strictly_better = bool((other_vals > row_vals + 1e-12).any())
                if not (weakly_better and strictly_better):
                    continue
                score = float(other.get("objective_score")) if pd.notna(other.get("objective_score")) else 0.0
                if score > best_dominator_score:
                    best_dominator = str(other.get("item_id", ""))
                    best_dominator_score = score
            if best_dominator:
                out.loc[idx, "pareto_dominated"] = True
                out.loc[idx, "dominated_by"] = best_dominator
    verdict_rank = {
        "frozen_tail_robust_wait_fresh": 0,
        "frozen_pnl_positive_tail_mixed_wait_fresh": 1,
        "family_pnl_tail_clean_research": 2,
        "family_pnl_positive_tail_tradeoff": 3,
        "family_pnl_positive_low_recurrence": 4,
        "frozen_pnl_positive_incomplete_tail": 5,
        "reject_or_monitor": 6,
    }
    out["verdict_rank"] = out["verdict"].map(verdict_rank).fillna(99).astype(int)
    return out.sort_values(
        ["pareto_dominated", "verdict_rank", "objective_score", "net_pnl_delta"],
        ascending=[True, True, False, False],
    ).drop(columns=["verdict_rank"])


def _minmax_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(0.0, index=series.index, dtype=float)
    lo = float(finite.min())
    hi = float(finite.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series(np.where(values.notna(), 0.5, 0.0), index=series.index, dtype=float)
    return ((values - lo) / (hi - lo)).fillna(0.0)


def _scope_champions(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["evidence_type", "comparison_scope"]
    for (evidence_type, scope), group in frontier.groupby(group_cols, sort=False):
        pool = group.loc[~_bool_series(group, "pareto_dominated")].copy()
        if pool.empty:
            pool = group.copy()
        if pool.empty:
            continue

        scored = pool.copy()
        scored["_pnl_score"] = (
            0.6 * _minmax_score(scored.get("objective_score", pd.Series(index=scored.index)))
            + 0.4 * _minmax_score(scored.get("net_pnl_delta", pd.Series(index=scored.index)))
        )
        scored["_tail_score"] = (
            0.25 * _minmax_score(scored.get("q35_day_delta", pd.Series(index=scored.index)))
            + 0.25 * _minmax_score(scored.get("q20_day_delta", pd.Series(index=scored.index)))
            + 0.20 * _minmax_score(scored.get("q25_week_delta", pd.Series(index=scored.index)))
            + 0.15 * _minmax_score(scored.get("worst_week_delta", pd.Series(index=scored.index)))
            + 0.15 * _minmax_score(scored.get("full_sl_improvement", pd.Series(index=scored.index)))
        )
        scored["_stability_score"] = (
            0.4 * _minmax_score(scored.get("positive_week_share", pd.Series(index=scored.index)))
            + 0.4 * _minmax_score(scored.get("positive_month_share", pd.Series(index=scored.index)))
            + 0.2 * _minmax_score(scored.get("min_month_delta_net_pnl", pd.Series(index=scored.index)))
        )
        scored["_balanced_score"] = (
            0.45 * scored["_pnl_score"]
            + 0.35 * scored["_tail_score"]
            + 0.20 * scored["_stability_score"]
        )

        selectors = {
            "best_pnl": ("_pnl_score", "objective_score", "net_pnl_delta"),
            "best_tail": (
                "q20_day_delta",
                "q35_day_delta",
                "q25_week_delta",
                "worst_week_delta",
                "full_sl_improvement",
                "positive_week_share",
                "positive_month_share",
                "objective_score",
            ),
            "balanced": ("_balanced_score", "_pnl_score", "_tail_score"),
        }
        for role, sort_cols in selectors.items():
            use_cols = [col for col in sort_cols if col in scored.columns]
            champion = scored.sort_values(use_cols, ascending=[False] * len(use_cols)).iloc[0]
            rows.append(
                {
                    "evidence_type": evidence_type,
                    "comparison_scope": scope,
                    "champion_role": role,
                    "champion_item_id": champion.get("item_id"),
                    "family": champion.get("family"),
                    "verdict": champion.get("verdict"),
                    "objective_score": champion.get("objective_score"),
                    "net_pnl_delta": champion.get("net_pnl_delta"),
                    "q35_day_delta": champion.get("q35_day_delta"),
                    "q20_day_delta": champion.get("q20_day_delta"),
                    "full_sl_improvement": champion.get("full_sl_improvement"),
                    "positive_week_share": champion.get("positive_week_share"),
                    "positive_month_share": champion.get("positive_month_share"),
                    "min_month_delta_net_pnl": champion.get("min_month_delta_net_pnl"),
                    "pnl_score": champion.get("_pnl_score"),
                    "tail_score": champion.get("_tail_score"),
                    "stability_score": champion.get("_stability_score"),
                    "balanced_score": champion.get("_balanced_score"),
                    "fresh_ready": champion.get("fresh_ready"),
                    "production_ready": champion.get("production_ready"),
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, cols: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    use_cols = [col for col in cols if col in frame.columns]
    return frame[use_cols].round(6).to_markdown(index=False)


def run(
    status_dirs: Sequence[Path],
    out_dir: Path,
    profile_selection_dir: Path | None = None,
    decision_pack_dirs: Sequence[Path] = (),
    gate_dirs: Sequence[Path] = (),
    family_evidence_dirs: Sequence[Path] = (),
) -> Dict[str, Any]:
    loaded = _load_status_dirs(status_dirs)
    decision_pack_summary = _load_decision_packs(decision_pack_dirs)
    gate_snapshots = _load_gate_rows(gate_dirs)
    gate_deficits = _load_gate_deficits(gate_dirs)
    gate_overrides = _load_gate_overrides(gate_dirs)
    summary = _candidate_summary(
        loaded["candidates"],
        loaded["stability"],
        decision_pack_summary,
        gate_overrides,
    )
    month_matrix = _month_matrix(loaded["monthly"])
    profile = _profile_selection(profile_selection_dir)
    family_evidence = _load_family_evidence(family_evidence_dirs)
    decision_matrix = _decision_matrix(summary, profile)
    promotion_frontier = _promotion_frontier(summary, decision_matrix, family_evidence, month_matrix)
    scope_champions = _scope_champions(promotion_frontier)
    worst = loaded["worst"].copy()
    if not worst.empty and "delta_net_pnl" in worst.columns:
        worst["delta_net_pnl"] = pd.to_numeric(worst["delta_net_pnl"], errors="coerce")
        worst = worst.sort_values("delta_net_pnl")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "frozen_reliability_long_period_candidate_summary.csv", index=False)
    decision_matrix.to_csv(out_dir / "frozen_reliability_long_period_decision_matrix.csv", index=False)
    gate_snapshots.to_csv(out_dir / "frozen_reliability_long_period_gate_snapshots.csv", index=False)
    gate_deficits.to_csv(out_dir / "frozen_reliability_long_period_gate_deficits.csv", index=False)
    month_matrix.to_csv(out_dir / "frozen_reliability_long_period_monthly_deltas.csv", index=False)
    worst.to_csv(out_dir / "frozen_reliability_long_period_worst_weeks.csv", index=False)
    family_evidence.to_csv(out_dir / "frozen_reliability_long_period_family_evidence.csv", index=False)
    promotion_frontier.to_csv(out_dir / "frozen_reliability_long_period_promotion_frontier.csv", index=False)
    scope_champions.to_csv(out_dir / "frozen_reliability_long_period_scope_champions.csv", index=False)
    if not profile.empty:
        profile.to_csv(out_dir / "frozen_reliability_long_period_profile_winners.csv", index=False)

    payload = {
        "generated_by": Path(__file__).name,
        "status_dirs": [str(path) for path in status_dirs],
        "profile_selection_dir": str(profile_selection_dir) if profile_selection_dir else None,
        "decision_pack_dirs": [str(path) for path in decision_pack_dirs],
        "gate_dirs": [str(path) for path in gate_dirs],
        "out_dir": str(out_dir),
        "objective": OBJECTIVE_NAME,
        "candidate_count": int(len(summary)),
        "decision_matrix_rows": int(len(decision_matrix)),
        "month_rows": int(len(month_matrix)),
        "worst_week_rows": int(len(worst)),
        "decision_pack_rows": int(len(decision_pack_summary)),
        "gate_snapshot_rows": int(len(gate_snapshots)),
        "gate_deficit_rows": int(len(gate_deficits)),
        "gate_override_rows": int(len(gate_overrides)),
        "family_evidence_rows": int(len(family_evidence)),
        "promotion_frontier_rows": int(len(promotion_frontier)),
        "scope_champion_rows": int(len(scope_champions)),
    }
    (out_dir / "frozen_reliability_long_period_comparison.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Frozen Reliability Long-Period Comparison",
        "",
        "This report compares already-evaluated frozen reliability candidates across long-period candidate, monthly, and worst-week artifacts.",
        "",
        f"Objective: `{OBJECTIVE_NAME}`.",
        "",
        "Scope: artifact-only comparison. No replay, refit, or candidate-definition change is performed.",
        "",
        "## Profile Winners",
        "",
        _markdown_table(
            profile,
            [
                "profile",
                "selected_rule_id",
                "delta_net_pnl",
                "delta_objective",
                "tail_clean",
                "fresh_ready",
                "production_ready",
                "fresh_blockers",
            ],
        ),
        "",
        "## Candidate Summary",
        "",
        _markdown_table(
            summary,
            [
                "rule_id",
                "source_name",
                "role",
                "delta_net_pnl",
                "delta_objective",
                "delta_avg_week_pnl",
                "delta_weighted_daily_tail",
                "delta_daily_q20",
                "delta_daily_q35",
                "active_positive_week_share",
                "q25_week_delta",
                "worst_week_delta",
                "entrant_minus_removed_hit_rate",
                "tail_clean",
                "fresh_ready",
                "production_ready",
                "fresh_blockers",
                "current_fresh_ready",
                "current_fresh_blockers",
                "current_post_cutoff_rows",
                "current_policy_action_rows",
                "current_policy_outcome_rows",
                "post_cutoff_rows",
                "policy_action_rows",
                "policy_outcome_rows",
            ],
        ),
        "",
        "## Decision Matrix",
        "",
        _markdown_table(
            decision_matrix,
            [
                "rule_id",
                "selected_profiles",
                "decision_state",
                "pnl_objective_pass",
                "daily_tail_pass",
                "weekly_tail_pass",
                "replacement_quality_pass",
                "tail_robust_pass",
                "fresh_ready_pass",
                "production_ready_pass",
                "delta_net_pnl",
                "delta_objective",
                "worst_week_delta",
                "fresh_blockers",
                "current_fresh_blockers",
            ],
        ),
        "",
        "## Diagnostic Family Evidence",
        "",
        _markdown_table(
            family_evidence,
            [
                "label",
                "diagnostic_family",
                "daily_weekly_objective",
                "sum_delta_net_pnl",
                "positive_week_count",
                "positive_week_share",
                "q35_day_delta_net_pnl",
                "q20_day_delta_net_pnl",
                "mean_day_full_sl_delta",
                "june_net_delta",
                "june_full_sl_delta",
                "positive_month_count",
                "family_evidence_dir",
            ],
        ),
        "",
        "## PnL/Tail Promotion Frontier",
        "",
        _markdown_table(
            promotion_frontier,
            [
                "evidence_type",
                "item_id",
                "comparison_scope",
                "family",
                "verdict",
                "pareto_dominated",
                "dominated_by",
                "objective_score",
                "net_pnl_delta",
                "q35_day_delta",
                "q20_day_delta",
                "q25_week_delta",
                "worst_week_delta",
                "full_sl_delta",
                "full_sl_improvement",
                "positive_week_share",
                "months",
                "positive_month_count",
                "positive_month_share",
                "min_month_delta_net_pnl",
                "apr_jun_delta_net_pnl",
                "june_delta_net_pnl",
                "fresh_ready",
                "production_ready",
                "notes",
            ],
        ),
        "",
        "## Scope Champions",
        "",
        _markdown_table(
            scope_champions,
            [
                "evidence_type",
                "comparison_scope",
                "champion_role",
                "champion_item_id",
                "family",
                "verdict",
                "objective_score",
                "net_pnl_delta",
                "q35_day_delta",
                "q20_day_delta",
                "full_sl_improvement",
                "positive_month_share",
                "pnl_score",
                "tail_score",
                "stability_score",
                "balanced_score",
                "fresh_ready",
                "production_ready",
            ],
        ),
        "",
        "## Gate Snapshots",
        "",
        _markdown_table(
            gate_snapshots,
            [
                "rule_id",
                "current_gate_cutoff",
                "current_fresh_ready",
                "current_post_cutoff_rows",
                "current_policy_action_rows",
                "current_policy_outcome_rows",
                "current_ready_sources",
                "current_fresh_blockers",
                "current_gate_dir",
            ],
        ),
        "",
        "## Gate Deficits",
        "",
        _markdown_table(
            gate_deficits,
            [
                "rule_id",
                "gate_cutoff",
                "gate",
                "head",
                "observed",
                "required",
                "deficit",
                "pass",
                "gate_dir",
            ],
        ),
        "",
        "## Monthly Deltas",
        "",
        _markdown_table(
            month_matrix,
            [
                "month",
                "rule_id",
                "source_name",
                "delta_net_pnl",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
            ],
        ),
        "",
        "## Worst Weeks",
        "",
        _markdown_table(
            worst.head(20),
            [
                "week",
                "rule_id",
                "source_name",
                "delta_net_pnl",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
            ],
        ),
    ]
    (out_dir / "frozen_reliability_long_period_comparison.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-dir", type=Path, action="append", required=True)
    parser.add_argument("--profile-selection-dir", type=Path)
    parser.add_argument("--decision-pack-dir", type=Path, action="append", default=[])
    parser.add_argument("--gate-dir", type=Path, action="append", default=[])
    parser.add_argument("--family-evidence-dir", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        args.status_dir,
        args.out_dir,
        args.profile_selection_dir,
        args.decision_pack_dir,
        args.gate_dir,
        args.family_evidence_dir,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "out_dir": payload["out_dir"],
                    "candidate_count": payload["candidate_count"],
                    "decision_matrix_rows": payload["decision_matrix_rows"],
                    "month_rows": payload["month_rows"],
                    "worst_week_rows": payload["worst_week_rows"],
                    "decision_pack_rows": payload["decision_pack_rows"],
                    "gate_snapshot_rows": payload["gate_snapshot_rows"],
                    "gate_deficit_rows": payload["gate_deficit_rows"],
                    "gate_override_rows": payload["gate_override_rows"],
                    "family_evidence_rows": payload["family_evidence_rows"],
                    "promotion_frontier_rows": payload["promotion_frontier_rows"],
                    "scope_champion_rows": payload["scope_champion_rows"],
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
