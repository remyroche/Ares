#!/usr/bin/env python3
"""Materialize a frozen reliability challenger bundle from replay artifacts.

This is intentionally artifact-driven.  It does not replay candidates or refit
models; it turns an already selected multi-window research rule into a physical
bundle that can be used by ``run_frozen_reliability_challenger_gate_if_ready``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_FORWARD_VALIDATION_REQUIREMENTS: Dict[str, Any] = {
    "same_candidate_universe": True,
    "same_cost_model": True,
    "minimum_post_cutoff_rows": 2000,
    "minimum_post_cutoff_timestamps": 40,
    "minimum_active_heads": 3,
    "minimum_policy_action_rows": 50,
    "minimum_policy_action_timestamps": 10,
    "minimum_policy_outcome_rows": 50,
    "minimum_policy_outcome_timestamps": 10,
    "required_matured_outcome_heads": [
        "long_bars",
        "long_dist",
        "short_asset",
        "short_bollinger",
    ],
    "minimum_policy_outcome_rows_per_required_head": 3,
    "report_baseline_and_frozen_rules": True,
    "primary_metrics": [
        "net_pnl",
        "objective_avgweek_0p7dayq35_0p3dayq20",
        "weekly_q20_pnl",
        "daily_q20_pnl",
        "full_sl_rate",
        "replacement_quality",
    ],
}

REQUESTED_RELIABILITY_FAMILIES = (
    "drift",
    "recent_hit_rate_surprise",
    "ood",
    "uncertainty",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


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
    present = set(families)
    return [family for family in REQUESTED_RELIABILITY_FAMILIES if family in present]


def _parse_rule_spec(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _conditional_summary(attribution_dir: Path) -> pd.DataFrame:
    path = attribution_dir / "conditional_filter_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing conditional filter summary: {path}")
    frame = pd.read_csv(path)
    if frame.empty or "rule_id" not in frame.columns:
        raise ValueError(f"{path} must contain rule_id rows")
    return frame


def _rule_summary_row(attribution_dir: Path, rule_id: str) -> pd.Series:
    frame = _conditional_summary(attribution_dir)
    rows = frame.loc[frame["rule_id"].astype(str).eq(str(rule_id))]
    if rows.empty:
        raise ValueError(f"Rule `{rule_id}` not found in {attribution_dir / 'conditional_filter_summary.csv'}")
    return rows.iloc[0]


def _rule_spec(attribution_dir: Path, rule_id: str) -> Dict[str, Any]:
    row = _rule_summary_row(attribution_dir, rule_id)
    spec = _parse_rule_spec(row.get("rule_spec"))
    if not spec:
        raise ValueError(f"Rule `{rule_id}` has no parseable rule_spec")
    condition = spec.get("condition")
    return {
        "heads": list(spec.get("heads") or []),
        "condition": condition,
        "families": _families_from_condition(condition, rule_id=rule_id),
        "action": spec.get("action"),
        "value": float(spec.get("value", 0.0)),
    }


def _multiwindow_metrics(selection_dir: Path, rule_id: str) -> Dict[str, Any]:
    path = selection_dir / "multiwindow_candidate_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing multi-window summary: {path}")
    frame = pd.read_csv(path)
    if frame.empty or "rule_id" not in frame.columns:
        raise ValueError(f"{path} must contain rule_id rows")
    rows = frame.loc[frame["rule_id"].astype(str).eq(str(rule_id))]
    if rows.empty:
        raise ValueError(f"Rule `{rule_id}` not found in {path}")
    return rows.iloc[0].to_dict()


def _selection_evidence(selection_dir: Path, rule_id: str) -> Dict[str, Any]:
    manifest = _read_json(selection_dir / "multiwindow_selection.json")
    rows: List[Dict[str, Any]] = []

    def add(kind: str, record: Mapping[str, Any] | None) -> None:
        if not record:
            return
        if str(record.get("rule_id") or "") != str(rule_id):
            return
        rows.append({"selection_kind": kind, **dict(record)})

    add("recommended", manifest.get("recommended"))
    add("best_by_sort_order", manifest.get("best_by_sort_order"))
    for profile, record in (manifest.get("profile_recommendations") or {}).items():
        add(f"profile:{profile}", record)
    return {
        "selection_dir": str(selection_dir),
        "baseline_rule": manifest.get("baseline_rule"),
        "tolerant_profile": manifest.get("tolerant_profile"),
        "matching_selection_rows": rows,
        "profile_pass": any(
            str(row.get("selection_kind") or "").startswith("profile:")
            or str(row.get("selection_kind") or "") == "recommended"
            for row in rows
        ),
    }


def _candidate_universe(attribution_dir: Path, rule_id: str) -> Dict[str, Any]:
    summary = _conditional_summary(attribution_dir)
    row = _rule_summary_row(attribution_dir, rule_id)
    daily_path = attribution_dir / "conditional_filter_daily.csv"
    daily = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
    accepted_path = attribution_dir / "conditional_filter_accepted_decisions.parquet"
    accepted = pd.read_parquet(accepted_path) if accepted_path.exists() else pd.DataFrame()

    daily_start = daily_end = None
    if not daily.empty and "day" in daily.columns:
        days = pd.to_datetime(daily["day"], utc=True, errors="coerce").dropna()
        if not days.empty:
            daily_start = days.min()
            daily_end = days.max()

    accepted_end = None
    if not accepted.empty and "timestamp" in accepted.columns:
        ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce").dropna()
        if not ts.empty:
            accepted_end = ts.max()

    source = _read_json(attribution_dir / "conditional_filter_summary.json")
    return {
        "combo_id": row.get("combo_id"),
        "source": source.get("source") or source.get("source_dir"),
        "source_mode": source.get("source_mode"),
        "threshold_mode": row.get("threshold_mode") or source.get("threshold_mode"),
        "min_threshold_history": int(row.get("min_threshold_history") or source.get("min_threshold_history") or 0),
        "candidate_rows": int(row.get("candidate_rows") or 0),
        "candidate_start": row.get("candidate_start"),
        "candidate_end": row.get("candidate_end"),
        "daily_start": daily_start,
        "daily_end": daily_end,
        "accepted_decision_end": accepted_end,
        "costs_included": True,
        "baseline_rule_count": int(summary["rule_id"].astype(str).eq("none").sum()),
    }


def _diagnostic_selection_rows(selection_dirs: Sequence[Path], max_rows: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for selection_dir in selection_dirs:
        manifest = _read_json(selection_dir / "multiwindow_selection.json")
        attribution_dir = Path(str(manifest.get("attribution_dir") or ""))
        summary_path = selection_dir / "multiwindow_candidate_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        if summary.empty or "rule_id" not in summary.columns:
            continue
        spec_lookup: Dict[str, Dict[str, Any]] = {}
        attr_summary_path = attribution_dir / "conditional_filter_summary.csv"
        if attr_summary_path.exists():
            attr_summary = pd.read_csv(attr_summary_path)
            for _, item in attr_summary.iterrows():
                spec_lookup[str(item.get("rule_id"))] = _parse_rule_spec(item.get("rule_spec"))
        for _, item in summary.head(int(max_rows)).iterrows():
            rule_id = str(item.get("rule_id") or "")
            spec = spec_lookup.get(rule_id, {})
            condition = spec.get("condition")
            rows.append(
                {
                    "selection_dir": str(selection_dir),
                    "rule_id": rule_id,
                    "condition": condition,
                    "families": ",".join(_families_from_condition(condition, rule_id=rule_id)),
                    "core_pnl_tail_gate_count": item.get("core_pnl_tail_gate_count"),
                    "core_min_delta_objective": item.get("core_min_delta_objective"),
                    "full_delta_net_pnl": item.get("full_delta_net_pnl"),
                    "full_delta_objective": item.get("full_delta_objective"),
                    "full_delta_weekly_q20": item.get("full_delta_weekly_q20"),
                    "full_delta_weighted_daily_tail": item.get("full_delta_weighted_daily_tail"),
                }
            )
    return rows


def _write_rules_file(bundle: Mapping[str, Any], out_dir: Path) -> Path:
    rules: Dict[str, Any] = {"none": {}}
    for rule_id, spec in (bundle.get("rules") or {}).items():
        rules[str(rule_id)] = {
            "heads": list(spec.get("heads") or []),
            "condition": spec.get("condition"),
            "action": spec.get("action"),
            "value": float(spec.get("value", 0.0)),
        }
    path = out_dir / "frozen_reliability_rules.json"
    path.write_text(json.dumps(_json_safe(rules), indent=2, sort_keys=True) + "\n")
    return path


def _materialize_bundle(
    *,
    bundle_id: str,
    attribution_dir: Path,
    selection_dir: Path,
    rule_id: str,
    out_dir: Path,
    role: str,
    promotion_note: str,
    baseline_rule: str,
    diagnostic_selection_dirs: Sequence[Path],
) -> Dict[str, Any]:
    spec = _rule_spec(attribution_dir, rule_id)
    metrics = _multiwindow_metrics(selection_dir, rule_id)
    selection = _selection_evidence(selection_dir, rule_id)
    rule = {
        "role": role,
        **spec,
        "metrics": metrics,
        "selection_evidence": selection,
        "promotion_note": promotion_note,
    }
    bundle = {
        "bundle_id": bundle_id,
        "generated_by": Path(__file__).name,
        "created_from": {
            "attribution_dir": str(attribution_dir),
            "multiwindow_selection_dir": str(selection_dir),
            "diagnostic_selection_dirs": [str(path) for path in diagnostic_selection_dirs],
        },
        "baseline_rule": baseline_rule,
        "candidate_universe": _candidate_universe(attribution_dir, rule_id),
        "rules": {rule_id: rule},
        "diagnostic_family_evidence": _diagnostic_selection_rows(diagnostic_selection_dirs, max_rows=20),
        "forward_validation_requirements": DEFAULT_FORWARD_VALIDATION_REQUIREMENTS,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / f"{bundle_id}.json"
    bundle_path.write_text(json.dumps(_json_safe(bundle), indent=2, sort_keys=True) + "\n")
    rules_path = _write_rules_file(bundle, out_dir)
    pd.DataFrame(
        [
            {
                "bundle_id": bundle_id,
                "rule_id": rule_id,
                "role": role,
                "heads": ",".join(map(str, spec.get("heads") or [])),
                "condition": spec.get("condition"),
                "families": ",".join(map(str, spec.get("families") or [])),
                "action": spec.get("action"),
                "value": spec.get("value"),
                "core_pnl_tail_gate_count": metrics.get("core_pnl_tail_gate_count"),
                "core_strict_tail_gate_count": metrics.get("core_strict_tail_gate_count"),
                "core_min_delta_objective": metrics.get("core_min_delta_objective"),
                "full_delta_net_pnl": metrics.get("full_delta_net_pnl"),
                "full_delta_objective": metrics.get("full_delta_objective"),
                "full_delta_weekly_q20": metrics.get("full_delta_weekly_q20"),
                "full_delta_weighted_daily_tail": metrics.get("full_delta_weighted_daily_tail"),
                "entrant_minus_removed_net_pnl": metrics.get("entrant_minus_removed_net_pnl"),
                "entrant_minus_removed_hit_rate": metrics.get("entrant_minus_removed_hit_rate"),
                "bundle_path": str(bundle_path),
                "rules_path": str(rules_path),
            }
        ]
    ).to_csv(out_dir / "frozen_reliability_bundle_summary.csv", index=False)
    lines = [
        "# Frozen Reliability Research Bundle",
        "",
        f"Bundle id: `{bundle_id}`",
        f"Rule: `{rule_id}`",
        f"Role: `{role}`",
        f"Condition: `{spec.get('condition')}`",
        f"Families: `{','.join(map(str, spec.get('families') or []))}`",
        f"Action: `{spec.get('action')}` `{spec.get('value')}`",
        "",
        "## Multi-Window Metrics",
        "",
        pd.DataFrame([metrics]).to_markdown(index=False),
        "",
        "## Candidate Universe",
        "",
        pd.DataFrame([bundle["candidate_universe"]]).to_markdown(index=False),
        "",
        "## Diagnostic Family Evidence",
        "",
        pd.DataFrame(bundle["diagnostic_family_evidence"]).head(20).to_markdown(index=False)
        if bundle["diagnostic_family_evidence"]
        else "_No diagnostic selection dirs supplied._",
    ]
    (out_dir / "frozen_reliability_bundle_report.md").write_text("\n".join(lines) + "\n")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--multiwindow-selection-dir", type=Path, required=True)
    parser.add_argument("--rule-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--role", default="multiwindow_balanced_research_challenger")
    parser.add_argument(
        "--promotion-note",
        default=(
            "Multi-window research candidate. Requires a frozen fresh gate before production promotion."
        ),
    )
    parser.add_argument("--baseline-rule", default="none")
    parser.add_argument("--diagnostic-selection-dir", type=Path, action="append", default=None)
    args = parser.parse_args()

    bundle = _materialize_bundle(
        bundle_id=str(args.bundle_id),
        attribution_dir=args.attribution_dir,
        selection_dir=args.multiwindow_selection_dir,
        rule_id=str(args.rule_id),
        out_dir=args.out_dir,
        role=str(args.role),
        promotion_note=str(args.promotion_note),
        baseline_rule=str(args.baseline_rule),
        diagnostic_selection_dirs=list(args.diagnostic_selection_dir or []),
    )
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "bundle_id": bundle.get("bundle_id")}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
