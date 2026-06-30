#!/usr/bin/env python3
"""Create a promotion/status report for head-native C3el candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _readiness_context(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "readiness_supplied": False,
            "readiness_decision": "",
            "readiness_unlabeled_target_rows": np.nan,
            "readiness_robust_unlabeled_target_rows": np.nan,
            "readiness_postjun_preferred_firings": np.nan,
            "allow_new_exact_state_replay": False,
            "allow_label_materialization_only": False,
            "production_blocked_by_readiness": False,
            "readiness_blocker": "readiness_not_supplied",
        }
    payload = _read_json(path)
    decision = str(payload.get("decision", ""))
    target_rows = int(payload.get("unlabeled_target_rows", 0) or 0)
    robust_target_rows = int(payload.get("robust_unlabeled_target_rows", 0) or 0)
    postjun_firings = int(payload.get("postjun_preferred_firings", 0) or 0)
    blockers: list[str] = []
    if decision == "monitor_only_wait_for_forward_recurrence":
        blockers.append("forward_recurrence_missing")
    if target_rows <= 0:
        blockers.append("no_unique_forward_target_rows")
    allow_new_exact_state_replay = target_rows >= 10
    allow_label_materialization_only = 0 < target_rows < 10
    if 0 < target_rows < 10:
        blockers.append("target_rows_below_replay_threshold")
    production_blocked = bool(blockers)
    return {
        "readiness_supplied": True,
        "readiness_decision": decision,
        "readiness_unlabeled_target_rows": target_rows,
        "readiness_robust_unlabeled_target_rows": robust_target_rows,
        "readiness_postjun_preferred_firings": postjun_firings,
        "allow_new_exact_state_replay": allow_new_exact_state_replay,
        "allow_label_materialization_only": allow_label_materialization_only,
        "production_blocked_by_readiness": production_blocked,
        "readiness_blocker": ";".join(blockers) if blockers else "none",
    }


def _parse_candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    name, value = raw.split("=", 1)
    return name.strip(), Path(value.strip())


def _load_support(path: Path) -> pd.DataFrame:
    support = pd.read_csv(path)
    required = {"head", "status", "positive_e50_groups", "positive_e50_weeks"}
    missing = sorted(required.difference(support.columns))
    if missing:
        raise ValueError(f"support decision file missing required columns: {missing}")
    support["head"] = support["head"].astype(str)
    support["status"] = support["status"].astype(str)
    return support


def _overall_metrics(path: Path) -> tuple[pd.Series, pd.Series]:
    overall = pd.read_csv(path)
    required = {"arm", "net_pnl", "net_hit_rate_pct", "full_sl_rate_pct", "trade_count"}
    missing = sorted(required.difference(overall.columns))
    if missing:
        raise ValueError(f"overall file missing required columns: {missing}")
    baseline = overall.loc[overall["arm"].astype(str).eq("C0_baseline")]
    candidate = overall.loc[~overall["arm"].astype(str).eq("C0_baseline")]
    if baseline.empty:
        raise ValueError(f"{path} does not contain C0_baseline")
    if candidate.empty:
        raise ValueError(f"{path} does not contain a candidate arm")
    return baseline.iloc[0], candidate.iloc[0]


def _support_summary(active_heads: list[str], support: pd.DataFrame) -> tuple[str, int, int, int, int, str]:
    if not active_heads:
        return "no_active_heads", 0, 0, 0, 0, "no_active_heads"
    rows = support.loc[support["head"].isin(active_heads)].copy()
    if rows.empty or rows["head"].nunique() != len(set(active_heads)):
        return "missing_support", 0, 0, 0, 0, "missing_support"
    statuses = sorted(set(rows["status"].astype(str)))
    pos_groups = int(pd.to_numeric(rows["positive_e50_groups"], errors="coerce").fillna(0).min())
    pos_weeks = int(pd.to_numeric(rows["positive_e50_weeks"], errors="coerce").fillna(0).min())
    recent_groups = (
        int(pd.to_numeric(rows["recent_positive_e50_groups"], errors="coerce").fillna(0).min())
        if "recent_positive_e50_groups" in rows.columns
        else 0
    )
    recent_weeks = (
        int(pd.to_numeric(rows["recent_positive_e50_weeks"], errors="coerce").fillna(0).min())
        if "recent_positive_e50_weeks" in rows.columns
        else 0
    )
    if "support_blocker" in rows.columns:
        blockers = [
            f"{row.head}:{row.support_blocker}"
            for row in rows[["head", "support_blocker"]].itertuples(index=False)
            if str(row.support_blocker) and str(row.support_blocker) != "none"
        ]
        blocker = " | ".join(blockers) if blockers else "none"
    else:
        blocker = "support_gap_unavailable"
    return "+".join(statuses), pos_groups, pos_weeks, recent_groups, recent_weeks, blocker


def _json_compact(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _numeric_column(frame: pd.DataFrame, col: str, *, default: float = 0.0) -> pd.Series:
    """Return a numeric column, defaulting missing optional diagnostics safely."""
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _active_heads_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return heads that were trained/scored by the head-native runner."""
    active = [str(x) for x in manifest.get("active_heads", []) if str(x)]
    if active:
        return active
    if str(manifest.get("c3el_contract", "")) == "head_native":
        configs = manifest.get("active_head_configs")
        if isinstance(configs, dict) and configs:
            return sorted(str(head) for head in configs)
        return list(HEADS)
    return []


def _applied_heads_from_manifest(manifest: dict[str, Any], scored_heads: list[str]) -> list[str]:
    """Return heads whose multipliers were actually applied in replay.

    Head-native C3el separates training/scoring scope from policy scope:
    ``active_heads`` are scored, while ``selected_heads`` are allowed to alter
    the replay schedule.  Promotion/support gates must use the applied policy
    scope; otherwise a diagnostic run that scores extra heads can be rejected
    for heads that never affected the candidate replay.
    """
    selected = manifest.get("selected_heads")
    if isinstance(selected, list):
        return sorted(str(head) for head in selected if str(head))
    return list(scored_heads)


def _candidate_head_diagnostics(run_dir: Path, active_heads: list[str]) -> dict[str, Any]:
    """Load optional head-native diagnostics without making old runs invalid."""
    active = {str(head) for head in active_heads}
    diagnostics: dict[str, Any] = {
        "head_cut_counts": {},
        "head_cut_shares": {},
        "head_delta_net_pnl": {},
        "head_delta_hr_pp": {},
        "head_delta_full_sl_pp": {},
        "fallback_used_week_count": 0,
        "used_model_week_count": 0,
        "fallback_used_week_rate": 0.0,
        "kept_eval_groups": 0,
        "threshold_keep_sum": 0,
        "threshold_value_sum": 0.0,
        "positive_threshold_week_count": 0,
        "guarded_eval_groups": 0,
        "action_feature_min_guarded_eval_groups": 0,
        "threshold_trial_file_present": False,
        "threshold_trial_count": 0,
        "threshold_trial_eligible_count": 0,
        "threshold_trial_positive_count": 0,
        "threshold_trial_best_value": np.nan,
        "threshold_trial_best_by_head": {},
    }

    schedule_path = run_dir / "head_native_size_schedule.csv"
    if schedule_path.exists():
        schedule = pd.read_csv(schedule_path)
        if {"strategy_id", "multiplier"}.issubset(schedule.columns):
            schedule["head"] = schedule["strategy_id"].astype(str).str.extract(
                r"^(long_bars|long_dist|short_asset|short_boll|short_bollinger)",
                expand=False,
            )
            schedule["head"] = schedule["head"].replace({"short_bollinger": "short_boll"})
            schedule["cut"] = pd.to_numeric(schedule["multiplier"], errors="coerce").fillna(1.0).lt(1.0)
            for head, group in schedule.groupby("head", dropna=True):
                if active and str(head) not in active:
                    continue
                diagnostics["head_cut_counts"][str(head)] = int(group["cut"].sum())
                diagnostics["head_cut_shares"][str(head)] = float(group["cut"].mean()) if len(group) else 0.0

    by_head_path = run_dir / "by_head.csv"
    if by_head_path.exists():
        by_head = pd.read_csv(by_head_path)
        if {"arm", "head", "net_pnl", "net_hit_rate_pct", "full_sl_rate_pct"}.issubset(by_head.columns):
            baseline = by_head.loc[by_head["arm"].astype(str).eq("C0_baseline")].set_index("head")
            candidate = by_head.loc[~by_head["arm"].astype(str).eq("C0_baseline")].set_index("head")
            for head in sorted(set(baseline.index).intersection(set(candidate.index))):
                if active and str(head) not in active:
                    continue
                diagnostics["head_delta_net_pnl"][str(head)] = float(candidate.loc[head, "net_pnl"] - baseline.loc[head, "net_pnl"])
                diagnostics["head_delta_hr_pp"][str(head)] = float(
                    candidate.loc[head, "net_hit_rate_pct"] - baseline.loc[head, "net_hit_rate_pct"]
                )
                diagnostics["head_delta_full_sl_pp"][str(head)] = float(
                    candidate.loc[head, "full_sl_rate_pct"] - baseline.loc[head, "full_sl_rate_pct"]
                )

    folds_path = run_dir / "head_native_folds.csv"
    if folds_path.exists():
        folds = pd.read_csv(folds_path)
        if "head" in folds.columns and active:
            folds = folds.loc[folds["head"].astype(str).isin(active)].copy()
        if not folds.empty:
            used = folds.loc[folds.get("used_model", pd.Series(False, index=folds.index)).astype(bool)].copy()
            diagnostics["used_model_week_count"] = int(len(used))
            diagnostics["fallback_used_week_count"] = int(
                used.get("fallback_used", pd.Series(False, index=used.index)).astype(bool).sum()
            )
            diagnostics["fallback_used_week_rate"] = (
                float(diagnostics["fallback_used_week_count"] / len(used)) if len(used) else 0.0
            )
            diagnostics["kept_eval_groups"] = int(_numeric_column(used, "kept_eval_groups").sum())
            threshold_keep = _numeric_column(used, "threshold_keep")
            threshold_value = _numeric_column(used, "threshold_value")
            diagnostics["threshold_keep_sum"] = int(threshold_keep.sum())
            diagnostics["threshold_value_sum"] = float(threshold_value.sum())
            diagnostics["positive_threshold_week_count"] = int(threshold_value.gt(0.0).sum())
            diagnostics["guarded_eval_groups"] = int(_numeric_column(used, "guarded_eval_groups").sum())
            diagnostics["action_feature_min_guarded_eval_groups"] = int(
                _numeric_column(used, "action_feature_min_guarded_eval_groups").sum()
            )

    trials_path = run_dir / "head_native_threshold_trials.csv"
    if trials_path.exists():
        trials = pd.read_csv(trials_path)
        if "head" in trials.columns and active:
            trials = trials.loc[trials["head"].astype(str).isin(active)].copy()
        if not trials.empty:
            diagnostics["threshold_trial_file_present"] = True
            value = _numeric_column(trials, "value")
            eligible = trials.get("eligible", pd.Series(False, index=trials.index))
            if eligible.dtype != bool:
                eligible = eligible.astype(str).str.lower().isin({"true", "1", "yes"})
            trials = trials.assign(_value=value, _eligible=eligible.astype(bool))
            diagnostics["threshold_trial_count"] = int(len(trials))
            diagnostics["threshold_trial_eligible_count"] = int(trials["_eligible"].sum())
            diagnostics["threshold_trial_positive_count"] = int((trials["_eligible"] & trials["_value"].gt(0.0)).sum())
            eligible_trials = trials.loc[trials["_eligible"]].copy()
            if not eligible_trials.empty:
                best_idx = eligible_trials["_value"].idxmax()
                best = eligible_trials.loc[best_idx]
                diagnostics["threshold_trial_best_value"] = float(best["_value"])
            best_by_head: dict[str, dict[str, Any]] = {}
            for head, group in trials.groupby("head", dropna=True):
                eligible_group = group.loc[group["_eligible"]].copy()
                if eligible_group.empty:
                    best_by_head[str(head)] = {
                        "eligible_trials": 0,
                        "positive_trials": 0,
                        "best_value": None,
                    }
                    continue
                best_idx = eligible_group["_value"].idxmax()
                best = eligible_group.loc[best_idx]
                best_by_head[str(head)] = {
                    "eligible_trials": int(len(eligible_group)),
                    "positive_trials": int(eligible_group["_value"].gt(0.0).sum()),
                    "best_value": float(best["_value"]),
                    "best_threshold": float(best["threshold"]) if "threshold" in best.index else None,
                    "best_min_pred_delta": float(best["min_pred_delta"]) if "min_pred_delta" in best.index else None,
                    "best_keep": int(best["keep"]) if "keep" in best.index and pd.notna(best["keep"]) else None,
                }
            diagnostics["threshold_trial_best_by_head"] = best_by_head
    return diagnostics


def _selection_evidence_status(head_diag: dict[str, Any]) -> tuple[str, str]:
    """Summarize whether head-native gates had non-fallback holdout evidence."""
    used = int(head_diag.get("used_model_week_count", 0) or 0)
    fallback = int(head_diag.get("fallback_used_week_count", 0) or 0)
    threshold_keep = int(head_diag.get("threshold_keep_sum", 0) or 0)
    positive_weeks = int(head_diag.get("positive_threshold_week_count", 0) or 0)
    threshold_value = float(head_diag.get("threshold_value_sum", 0.0) or 0.0)
    if used <= 0:
        return "not_available", "no_head_native_fold_diagnostics"
    if threshold_keep > 0 and positive_weeks > 0 and threshold_value > 0.0:
        return "holdout_positive", "none"
    if bool(head_diag.get("threshold_trial_file_present")) and int(head_diag.get("threshold_trial_positive_count", 0) or 0) <= 0:
        return "fallback_only", "no_positive_holdout_threshold_trial"
    if fallback >= used and threshold_keep <= 0 and positive_weeks <= 0:
        return "fallback_only", "fallback_only_threshold_selection"
    return "weak_or_zero_holdout", "weak_or_zero_holdout_threshold_selection"


def _disposition(
    *,
    interventions: int,
    delta_net_pnl: float,
    delta_hr_pp: float,
    delta_full_sl_pp: float,
    support_status: str,
    selection_evidence_status: str,
    min_net_pnl_delta: float,
    min_hr_delta_pp: float,
    max_full_sl_delta_pp: float,
) -> str:
    if interventions <= 0:
        return "no_op"
    replay_pass = (
        delta_net_pnl >= float(min_net_pnl_delta)
        and delta_hr_pp >= float(min_hr_delta_pp)
        and delta_full_sl_pp <= float(max_full_sl_delta_pp)
    )
    if not replay_pass:
        return "reject_replay"
    if selection_evidence_status in {"fallback_only", "weak_or_zero_holdout"}:
        return "replay_validated_research"
    if support_status == "production_candidate":
        return "promotion_candidate"
    if support_status in {"research_candidate", "research_candidate+research_candidate"}:
        return "replay_validated_research"
    if "insufficient_support" in support_status or "missing_support" in support_status:
        return "diagnostic_only"
    return "research_challenger"


def _gate_flags(disposition: str) -> tuple[bool, bool, str]:
    """Return monitored/production gates and a concise reason."""
    if disposition == "promotion_candidate":
        return True, True, "replay_passed_and_production_support"
    if disposition == "replay_validated_research":
        return True, False, "replay_passed_but_support_is_research_level"
    if disposition == "research_challenger":
        return True, False, "replay_passed_but_support_or_status_is_mixed"
    if disposition == "no_op":
        return False, False, "candidate_made_no_interventions"
    if disposition == "reject_replay":
        return False, False, "replay_metrics_failed_gate"
    if disposition == "diagnostic_only":
        return False, False, "support_missing_or_insufficient"
    return False, False, "unknown_disposition"


def _candidate_role(
    *,
    disposition: str,
    candidate: str,
    active_head_count: int,
    min_recent_groups: int,
    support_blocker: str,
    selected_default: str | None,
) -> str:
    if disposition == "promotion_candidate":
        return "production_candidate"
    if disposition != "replay_validated_research":
        return "rejected_or_diagnostic"
    if selected_default is not None and candidate == selected_default:
        return "monitored_default"
    if min_recent_groups <= 0 or "no_recent_e50_positive_groups" in str(support_blocker):
        return "monitor_research_stale_support"
    if active_head_count > 1:
        return "monitor_research_complex_challenger"
    return "monitor_research_challenger"


def _fmt_metric(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return f"`n/a{suffix}`"
    if not np.isfinite(num):
        return f"`n/a{suffix}`"
    return f"`{num:.{digits}f}{suffix}`"


def _compact_head_delta_summary(raw: Any) -> str:
    try:
        payload = json.loads(str(raw)) if str(raw).strip() else {}
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict) or not payload:
        return ""
    parts = []
    for head, value in sorted(payload.items()):
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        sign = "+" if num >= 0.0 else ""
        parts.append(f"{head} {sign}{num:.2f}")
    return ", ".join(parts)


def _evidence_reading_lines(report: pd.DataFrame) -> list[str]:
    """Create a concise, reproducible interpretation from the report table."""
    lines = ["## Evidence Reading", ""]
    if report.empty:
        lines.append("No candidate evidence is available.")
        return lines

    for row in report.itertuples(index=False):
        role = str(row.candidate_role)
        candidate = str(row.candidate)
        selected_heads = str(getattr(row, "selected_heads", getattr(row, "active_heads", "")))
        scored_heads = str(getattr(row, "scored_heads", selected_heads))
        head_delta = _compact_head_delta_summary(getattr(row, "head_delta_net_pnl", ""))
        sentence = (
            f"- `{candidate}` ({role}): selected heads `{selected_heads}`"
            f" from scored heads `{scored_heads}`; interventions {_fmt_metric(row.interventions, digits=0)},"
            f" delta net PnL {_fmt_metric(row.delta_net_pnl)},"
            f" delta HR {_fmt_metric(row.delta_hr_pp, suffix='pp')},"
            f" delta full-SL {_fmt_metric(row.delta_full_sl_pp, suffix='pp')}."
        )
        lines.append(sentence)
        blockers: list[str] = []
        if str(row.selection_evidence_status) != "holdout_positive":
            blockers.append(f"selection evidence is `{row.selection_evidence_status}`")
        if not bool(getattr(row, "threshold_trial_file_present", False)):
            blockers.append("threshold-trial diagnostics are missing")
        elif int(getattr(row, "threshold_trial_positive_count", 0) or 0) <= 0:
            blockers.append(
                "threshold trials had no positive holdout value"
                f" (best eligible {_fmt_metric(getattr(row, 'threshold_trial_best_value', np.nan))})"
            )
        if bool(row.production_blocked_by_readiness):
            blockers.append(f"readiness blocker `{row.readiness_blocker}`")
        if str(row.support_status) != "production_candidate":
            blockers.append(f"support is `{row.support_status}`")
        if head_delta:
            blockers.append(f"head PnL deltas: {head_delta}")
        if blockers:
            lines.append(f"  Current blocker reading: {'; '.join(blockers)}.")

    max_targets = int(pd.to_numeric(report.get("readiness_unlabeled_target_rows"), errors="coerce").fillna(0).max())
    has_fallback_only = report.get("selection_evidence_status", pd.Series(dtype=object)).astype(str).eq("fallback_only").any()
    has_complex = pd.to_numeric(report.get("active_head_count"), errors="coerce").fillna(0).gt(1).any()

    lines.extend(["", "## Next Ablation Rules", ""])
    if max_targets <= 0:
        lines.append(
            "- Do not run another exact-state replay yet: forward recurrence has not produced unique target rows."
        )
    elif max_targets < 10:
        lines.append(
            "- Materialize labels only; defer exact-state replay until at least 10 unique target rows exist across multiple days."
        )
    else:
        lines.append(
            "- Exact-state replay is allowed by target count, but still compare cooldown, robust, and all-four rules before promotion."
        )
    if has_fallback_only:
        lines.append(
            "- Prioritize non-fallback per-head threshold evidence; fallback-only replay gains stay research-level."
        )
    if has_complex:
        lines.append(
            "- Keep multi-head challengers separate from the monitored default unless each added head has positive head-native support and non-negative contribution."
        )
    lines.append(
        "- Promotion remains blocked until forward recurrence, unique targets, production-level action support, and holdout-positive selection evidence all clear together."
    )
    return lines


def build_report(
    candidates: list[tuple[str, Path]],
    *,
    support_decision: Path,
    min_net_pnl_delta: float,
    min_hr_delta_pp: float,
    max_full_sl_delta_pp: float,
    readiness_manifest: Path | None = None,
) -> pd.DataFrame:
    support = _load_support(support_decision)
    readiness = _readiness_context(readiness_manifest)
    rows: list[dict[str, Any]] = []
    for name, run_dir in candidates:
        manifest_path = run_dir / "manifest.json"
        overall_path = run_dir / "overall.csv"
        manifest = _read_json(manifest_path)
        baseline, candidate = _overall_metrics(overall_path)
        scored_heads = _active_heads_from_manifest(manifest)
        active_heads = _applied_heads_from_manifest(manifest, scored_heads)
        support_status, min_pos_groups, min_pos_weeks, min_recent_groups, min_recent_weeks, support_blocker = _support_summary(
            active_heads,
            support,
        )
        head_diag = _candidate_head_diagnostics(run_dir, active_heads)
        selection_evidence_status, selection_evidence_blocker = _selection_evidence_status(head_diag)
        delta_net_pnl = float(candidate["net_pnl"] - baseline["net_pnl"])
        delta_hr_pp = float(candidate["net_hit_rate_pct"] - baseline["net_hit_rate_pct"])
        delta_full_sl_pp = float(candidate["full_sl_rate_pct"] - baseline["full_sl_rate_pct"])
        interventions = int(manifest.get("interventions", 0))
        disposition = _disposition(
            interventions=interventions,
            delta_net_pnl=delta_net_pnl,
            delta_hr_pp=delta_hr_pp,
            delta_full_sl_pp=delta_full_sl_pp,
            support_status=support_status,
            selection_evidence_status=selection_evidence_status,
            min_net_pnl_delta=min_net_pnl_delta,
            min_hr_delta_pp=min_hr_delta_pp,
            max_full_sl_delta_pp=max_full_sl_delta_pp,
        )
        allow_monitored_replay, allow_production, gate_reason = _gate_flags(disposition)
        allow_production = bool(allow_production) and not bool(readiness["production_blocked_by_readiness"])
        if bool(readiness["production_blocked_by_readiness"]) and disposition == "promotion_candidate":
            gate_reason = f"{gate_reason};{readiness['readiness_blocker']}"
        active_head_count = len(set(active_heads))
        rows.append(
            {
                "candidate": name,
                "run_dir": str(run_dir),
                "start": manifest.get("start", ""),
                "end": manifest.get("end", ""),
                "scored_heads": ",".join(scored_heads),
                "selected_heads": ",".join(active_heads),
                "active_heads": ",".join(active_heads),
                "active_head_count": active_head_count,
                "interventions": interventions,
                "support_status": support_status,
                "min_positive_e50_groups": min_pos_groups,
                "min_positive_e50_weeks": min_pos_weeks,
                "min_recent_positive_e50_groups": min_recent_groups,
                "min_recent_positive_e50_weeks": min_recent_weeks,
                "support_blocker": support_blocker,
                "baseline_trade_count": int(baseline["trade_count"]),
                "candidate_trade_count": int(candidate["trade_count"]),
                "baseline_net_pnl": float(baseline["net_pnl"]),
                "candidate_net_pnl": float(candidate["net_pnl"]),
                "delta_net_pnl": delta_net_pnl,
                "delta_hr_pp": delta_hr_pp,
                "delta_full_sl_pp": delta_full_sl_pp,
                "candidate_ev_bps": float(candidate.get("net_ev_bps_turnover", np.nan)),
                "head_cut_counts": _json_compact(head_diag["head_cut_counts"]),
                "head_cut_shares": _json_compact(head_diag["head_cut_shares"]),
                "head_delta_net_pnl": _json_compact(head_diag["head_delta_net_pnl"]),
                "head_delta_hr_pp": _json_compact(head_diag["head_delta_hr_pp"]),
                "head_delta_full_sl_pp": _json_compact(head_diag["head_delta_full_sl_pp"]),
                "fallback_used_week_count": int(head_diag["fallback_used_week_count"]),
                "used_model_week_count": int(head_diag["used_model_week_count"]),
                "fallback_used_week_rate": float(head_diag["fallback_used_week_rate"]),
                "kept_eval_groups": int(head_diag["kept_eval_groups"]),
                "threshold_keep_sum": int(head_diag["threshold_keep_sum"]),
                "threshold_value_sum": float(head_diag["threshold_value_sum"]),
                "positive_threshold_week_count": int(head_diag["positive_threshold_week_count"]),
                "threshold_trial_file_present": bool(head_diag["threshold_trial_file_present"]),
                "threshold_trial_count": int(head_diag["threshold_trial_count"]),
                "threshold_trial_eligible_count": int(head_diag["threshold_trial_eligible_count"]),
                "threshold_trial_positive_count": int(head_diag["threshold_trial_positive_count"]),
                "threshold_trial_best_value": float(head_diag["threshold_trial_best_value"]),
                "threshold_trial_best_by_head": _json_compact(head_diag["threshold_trial_best_by_head"]),
                "selection_evidence_status": selection_evidence_status,
                "selection_evidence_blocker": selection_evidence_blocker,
                "guarded_eval_groups": int(head_diag["guarded_eval_groups"]),
                "action_feature_min_guarded_eval_groups": int(head_diag["action_feature_min_guarded_eval_groups"]),
                "readiness_supplied": bool(readiness["readiness_supplied"]),
                "readiness_decision": readiness["readiness_decision"],
                "readiness_unlabeled_target_rows": readiness["readiness_unlabeled_target_rows"],
                "readiness_robust_unlabeled_target_rows": readiness["readiness_robust_unlabeled_target_rows"],
                "readiness_postjun_preferred_firings": readiness["readiness_postjun_preferred_firings"],
                "allow_label_materialization_only": bool(readiness["allow_label_materialization_only"]),
                "allow_new_exact_state_replay": bool(readiness["allow_new_exact_state_replay"]),
                "production_blocked_by_readiness": bool(readiness["production_blocked_by_readiness"]),
                "readiness_blocker": readiness["readiness_blocker"],
                "disposition": disposition,
                "allow_monitored_replay": bool(allow_monitored_replay),
                "allow_production": bool(allow_production),
                "gate_reason": gate_reason,
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    eligible_default = report.loc[
        report["disposition"].astype(str).eq("replay_validated_research")
        & report["allow_monitored_replay"].astype(bool)
        & pd.to_numeric(report["min_recent_positive_e50_groups"], errors="coerce").fillna(0).gt(0)
    ].copy()
    if eligible_default.empty:
        selected_default: str | None = None
    else:
        eligible_default["_default_rank"] = (
            pd.to_numeric(eligible_default["delta_net_pnl"], errors="coerce").fillna(0.0)
            - 500.0 * pd.to_numeric(eligible_default["active_head_count"], errors="coerce").fillna(99.0)
            + 100.0 * pd.to_numeric(eligible_default["min_recent_positive_e50_groups"], errors="coerce").fillna(0.0)
        )
        selected_default = str(
            eligible_default.sort_values(
                ["_default_rank", "active_head_count", "delta_net_pnl"],
                ascending=[False, True, False],
            )["candidate"].iloc[0]
        )
    report["candidate_role"] = [
        _candidate_role(
            disposition=str(row.disposition),
            candidate=str(row.candidate),
            active_head_count=int(row.active_head_count),
            min_recent_groups=int(row.min_recent_positive_e50_groups),
            support_blocker=str(row.support_blocker),
            selected_default=selected_default,
        )
        for row in report.itertuples(index=False)
    ]
    role_order = {
        "production_candidate": 0,
        "monitored_default": 1,
        "monitor_research_challenger": 2,
        "monitor_research_complex_challenger": 3,
        "monitor_research_stale_support": 4,
        "rejected_or_diagnostic": 5,
    }
    report["_role_order"] = report["candidate_role"].map(role_order).fillna(99).astype(int)
    return report.sort_values(
        ["_role_order", "delta_net_pnl", "delta_hr_pp"],
        ascending=[True, False, False],
    ).drop(columns=["_role_order"])


def _write_markdown(path: Path, report: pd.DataFrame, *, support_decision: Path) -> None:
    lines = [
        "# C3el promotion status",
        "",
        f"Support source: `{support_decision}`.",
        "",
        "Interpretation:",
        "",
        "- `promotion_candidate`: replay passes, all active heads have production support, and head-native selection has positive holdout threshold evidence.",
        "- `replay_validated_research`: replay passes, but action-label support or selection evidence is still research-level.",
        "- `research_challenger`: replay passes weakly or via mixed research statuses; monitor only.",
        "- `reject_replay`: replay does not pass PnL/HR/full-SL gates.",
        "- `diagnostic_only`: support is missing or insufficient.",
        "",
        "`allow_monitored_replay` means the candidate can be tracked in shadow/monitored replay. `allow_production` is stricter and requires replay pass, production-level action support, and non-fallback holdout threshold evidence.",
        "",
        "`candidate_role` picks the lowest-fragility monitored default first. Small PnL improvements from adding stale or weak-support heads remain research challengers.",
        "",
        "When a C3el readiness manifest is supplied, `allow_new_exact_state_replay` and `production_blocked_by_readiness` reconcile this replay status with forward-recurrence gates. A candidate may remain eligible for monitored research replay while still being blocked from production or new exact-state replay.",
        "",
        "## Candidates",
        "",
    ]
    if report.empty:
        lines.append("No candidates.")
    else:
        cols = [
            "candidate",
            "candidate_role",
            "scored_heads",
            "selected_heads",
            "active_heads",
            "active_head_count",
            "interventions",
            "support_status",
            "min_positive_e50_groups",
            "min_positive_e50_weeks",
            "min_recent_positive_e50_groups",
            "min_recent_positive_e50_weeks",
            "support_blocker",
            "baseline_trade_count",
            "candidate_trade_count",
            "delta_net_pnl",
            "delta_hr_pp",
            "delta_full_sl_pp",
            "candidate_ev_bps",
            "head_cut_counts",
            "head_delta_net_pnl",
            "fallback_used_week_count",
            "used_model_week_count",
            "fallback_used_week_rate",
            "threshold_keep_sum",
            "threshold_value_sum",
            "positive_threshold_week_count",
            "threshold_trial_file_present",
            "threshold_trial_eligible_count",
            "threshold_trial_positive_count",
            "threshold_trial_best_value",
            "selection_evidence_status",
            "selection_evidence_blocker",
            "readiness_decision",
            "readiness_unlabeled_target_rows",
            "allow_label_materialization_only",
            "allow_new_exact_state_replay",
            "production_blocked_by_readiness",
            "readiness_blocker",
            "disposition",
            "allow_monitored_replay",
            "allow_production",
            "gate_reason",
        ]
        lines.append(report[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            *_evidence_reading_lines(report),
            "",
            "## Decision",
            "",
            "Keep only sparse, replay-validated research candidates active for monitored replay. Do not promote a head to production unless it also clears action-label support, because sparse positive action groups are easy to overfit.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", default=[], help="NAME=run_dir or run_dir")
    parser.add_argument("--support-decision", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-net-pnl-delta", type=float, default=0.0)
    parser.add_argument("--min-hr-delta-pp", type=float, default=0.0)
    parser.add_argument("--max-full-sl-delta-pp", type=float, default=0.0)
    args = parser.parse_args()

    candidates = [_parse_candidate(raw) for raw in args.candidate]
    if not candidates:
        raise ValueError("at least one --candidate is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        candidates,
        support_decision=args.support_decision,
        readiness_manifest=args.readiness_manifest,
        min_net_pnl_delta=float(args.min_net_pnl_delta),
        min_hr_delta_pp=float(args.min_hr_delta_pp),
        max_full_sl_delta_pp=float(args.max_full_sl_delta_pp),
    )
    report.to_csv(args.out_dir / "c3el_promotion_status.csv", index=False)
    _write_markdown(args.out_dir / "summary.md", report, support_decision=args.support_decision)
    args.out_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "report_c3el_promotion_status",
                "support_decision": str(args.support_decision),
                "readiness_manifest": str(args.readiness_manifest) if args.readiness_manifest else None,
                "candidates": [{"name": name, "run_dir": str(run_dir)} for name, run_dir in candidates],
                "min_net_pnl_delta": float(args.min_net_pnl_delta),
                "min_hr_delta_pp": float(args.min_hr_delta_pp),
                "max_full_sl_delta_pp": float(args.max_full_sl_delta_pp),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
