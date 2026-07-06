#!/usr/bin/env python3
"""Select frozen reliability candidates under explicit PnL/tail profiles.

The script consumes candidate-status CSVs produced by
``audit_frozen_reliability_challenger_status.py``.  It does not replay trades;
it ranks already-evaluated candidates under transparent profile weights and
marks a Pareto frontier over PnL, objective, active-week stability, and worst
week downside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


PROFILE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "tail_first": {
        "pnl_score": 0.20,
        "objective_score": 0.20,
        "tail_clean_score": 0.25,
        "active_week_score": 0.20,
        "worst_week_score": 0.15,
    },
    "balanced": {
        "pnl_score": 0.30,
        "objective_score": 0.25,
        "tail_clean_score": 0.15,
        "active_week_score": 0.15,
        "worst_week_score": 0.15,
    },
    "pnl_first": {
        "pnl_score": 0.45,
        "objective_score": 0.30,
        "tail_clean_score": 0.05,
        "active_week_score": 0.10,
        "worst_week_score": 0.10,
    },
    "replacement_quality": {
        "pnl_score": 0.10,
        "objective_score": 0.10,
        "tail_clean_score": 0.05,
        "active_week_score": 0.10,
        "worst_week_score": 0.05,
        "replacement_hit_score": 0.60,
    },
}

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
    parent = path.parent.name
    suffixes = (
        "_status_20260701",
        "_challenger_status_allroots_20260701",
        "_candidate_status",
    )
    for suffix in suffixes:
        if parent.endswith(suffix):
            return parent[: -len(suffix)] or parent
    return parent


def _source_status(path: Path) -> Dict[str, Any]:
    status_path = path.parent / "frozen_reliability_status.json"
    if not status_path.exists():
        return {}
    try:
        payload = json.loads(status_path.read_text())
    except Exception:
        return {}
    gate_summary = payload.get("gate_summary") if isinstance(payload.get("gate_summary"), dict) else {}
    blockers = payload.get("fresh_blockers")
    if isinstance(blockers, list):
        blocker_text = ";".join(str(item) for item in blockers)
    elif blockers is None:
        blocker_text = ""
    else:
        blocker_text = str(blockers)
    return {
        "source_research_ready": bool(payload.get("research_ready", False)),
        "source_fresh_ready": bool(payload.get("fresh_ready", False)),
        "source_production_ready": bool(payload.get("production_ready", False)),
        "source_fresh_blockers": blocker_text,
        "source_post_cutoff_rows": gate_summary.get("post_cutoff_rows"),
        "source_policy_action_rows": gate_summary.get("policy_action_rows"),
        "source_policy_outcome_rows": gate_summary.get("policy_outcome_rows"),
    }


def _load_candidates(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        source_status = _source_status(path)
        for col, value in source_status.items():
            if col not in frame.columns:
                frame[col] = value
        frame.insert(0, "source_path", str(path))
        frame.insert(1, "source_name", _source_name(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "rule_id" not in out.columns:
        raise ValueError("Candidate status inputs must contain rule_id")
    dedupe_cols = ["source_name", "rule_id", "role"]
    return out.drop_duplicates(subset=[c for c in dedupe_cols if c in out.columns], keep="first")


def _positive_scaled(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    max_val = float(values.max())
    if max_val <= 0.0 or not np.isfinite(max_val):
        return pd.Series(0.0, index=series.index)
    return values / max_val


def _worst_week_score(series: pd.Series) -> pd.Series:
    worst = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if worst.empty:
        return pd.Series(dtype=float)
    min_val = float(worst.min())
    max_val = float(worst.max())
    if max_val == min_val:
        return pd.Series(1.0, index=series.index)
    return ((worst - min_val) / (max_val - min_val)).clip(0.0, 1.0)


def _score_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["delta_net_pnl"] = pd.to_numeric(out.get("delta_net_pnl"), errors="coerce").fillna(0.0)
    out["delta_objective"] = pd.to_numeric(out.get("delta_objective"), errors="coerce").fillna(0.0)
    out["active_positive_week_share"] = pd.to_numeric(
        out.get("active_positive_week_share"), errors="coerce"
    ).fillna(0.0)
    out["worst_week_delta"] = pd.to_numeric(out.get("worst_week_delta"), errors="coerce").fillna(0.0)
    out["entrant_minus_removed_hit_rate"] = pd.to_numeric(
        out.get("entrant_minus_removed_hit_rate"), errors="coerce"
    ).fillna(0.0)
    out["tail_clean"] = out.get("tail_clean", False).astype(bool)
    out["research_pass"] = out.get("research_pass", False).astype(bool)
    out["pnl_score"] = _positive_scaled(out["delta_net_pnl"])
    out["objective_score"] = _positive_scaled(out["delta_objective"])
    out["tail_clean_score"] = out["tail_clean"].astype(float)
    out["active_week_score"] = out["active_positive_week_share"].clip(0.0, 1.0)
    out["worst_week_score"] = _worst_week_score(out["worst_week_delta"])
    out["replacement_hit_score"] = _positive_scaled(out["entrant_minus_removed_hit_rate"])
    out["bootstrap_p05_net_positive"] = pd.to_numeric(
        out.get("delta_net_pnl_p05"), errors="coerce"
    ).fillna(0.0).gt(0.0)
    out["bootstrap_p05_objective_positive"] = pd.to_numeric(
        out.get("delta_objective_p05"), errors="coerce"
    ).fillna(0.0).gt(0.0)
    for profile, weights in PROFILE_WEIGHTS.items():
        score = pd.Series(0.0, index=out.index)
        for col, weight in weights.items():
            score = score + float(weight) * pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
        out[f"{profile}_score"] = score
    return out


def _dominates(left: pd.Series, right: pd.Series) -> bool:
    metrics = (
        "delta_net_pnl",
        "delta_objective",
        "active_positive_week_share",
        "worst_week_delta",
        "entrant_minus_removed_hit_rate",
    )
    left_vals = [float(left.get(metric, 0.0)) for metric in metrics]
    right_vals = [float(right.get(metric, 0.0)) for metric in metrics]
    return all(l >= r for l, r in zip(left_vals, right_vals)) and any(
        l > r for l, r in zip(left_vals, right_vals)
    )


def _pareto_frontier(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    frontier = []
    for idx, row in frame.iterrows():
        dominated = False
        for other_idx, other in frame.iterrows():
            if idx == other_idx:
                continue
            if _dominates(other, row):
                dominated = True
                break
        frontier.append(not dominated)
    return pd.Series(frontier, index=frame.index, dtype=bool)


def _profile_selection(scored: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    eligible = scored.loc[
        scored.get("research_pass", pd.Series(False, index=scored.index)).astype(bool)
        & scored.get("bootstrap_p05_net_positive", pd.Series(False, index=scored.index)).astype(bool)
        & scored.get("bootstrap_p05_objective_positive", pd.Series(False, index=scored.index)).astype(bool)
    ].copy()
    if eligible.empty:
        eligible = scored.copy()
    for profile in PROFILE_WEIGHTS:
        score_col = f"{profile}_score"
        best = eligible.sort_values(
            [score_col, "tail_clean", "delta_net_pnl", "delta_objective"],
            ascending=[False, False, False, False],
        ).iloc[0]
        rows.append(
            {
                "profile": profile,
                "selected_rule_id": best.get("rule_id"),
                "source_name": best.get("source_name"),
                "role": best.get("role"),
                "score": best.get(score_col),
                "delta_net_pnl": best.get("delta_net_pnl"),
                "delta_objective": best.get("delta_objective"),
                "active_positive_week_share": best.get("active_positive_week_share"),
                "worst_week_delta": best.get("worst_week_delta"),
                "entrant_minus_removed_hit_rate": best.get("entrant_minus_removed_hit_rate"),
                "tail_clean": best.get("tail_clean"),
                "research_pass": best.get("research_pass"),
                "fresh_ready": best.get("source_fresh_ready"),
                "production_ready": best.get("source_production_ready"),
                "fresh_blockers": best.get("source_fresh_blockers"),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, cols: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    use_cols = [col for col in cols if col in frame.columns]
    return frame[use_cols].round(6).to_markdown(index=False)


def run(candidate_status: Sequence[Path], out_dir: Path) -> Dict[str, Any]:
    candidates = _load_candidates(candidate_status)
    scored = _score_candidates(candidates)
    if not scored.empty:
        scored["pareto_frontier"] = _pareto_frontier(scored)
    selected = _profile_selection(scored)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "frozen_reliability_profile_scored_candidates.csv", index=False)
    selected.to_csv(out_dir / "frozen_reliability_profile_selection.csv", index=False)
    payload = {
        "generated_by": Path(__file__).name,
        "candidate_status": [str(path) for path in candidate_status],
        "out_dir": str(out_dir),
        "objective": OBJECTIVE_NAME,
        "deployment_scope": "research_profile_selection_only",
        "fresh_gate_note": (
            "Candidate-status CSVs do not prove fresh-gate readiness. Use the matching "
            "frozen_reliability_status_report.md or frozen_reliability_gate_report.md "
            "before treating any profile winner as deployable."
        ),
        "profiles": PROFILE_WEIGHTS,
        "selected": selected.to_dict(orient="records"),
        "candidate_count": int(len(scored)),
    }
    (out_dir / "frozen_reliability_profile_selection.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    candidate_cols = [
        "rule_id",
        "source_name",
        "role",
        "pareto_frontier",
        "tail_clean",
        "delta_net_pnl",
        "delta_objective",
        "active_positive_week_share",
        "worst_week_delta",
        "entrant_minus_removed_hit_rate",
        "source_fresh_ready",
        "source_production_ready",
        "source_fresh_blockers",
        "source_post_cutoff_rows",
        "source_policy_action_rows",
        "source_policy_outcome_rows",
        "tail_first_score",
        "balanced_score",
        "pnl_first_score",
        "replacement_quality_score",
    ]
    selection_cols = [
        "profile",
        "selected_rule_id",
        "source_name",
        "score",
        "delta_net_pnl",
        "delta_objective",
        "active_positive_week_share",
        "worst_week_delta",
        "entrant_minus_removed_hit_rate",
        "tail_clean",
        "fresh_ready",
        "production_ready",
        "fresh_blockers",
    ]
    lines = [
        "# Frozen Reliability Profile Selection",
        "",
        "This report ranks already-evaluated frozen candidates under explicit PnL/tail profiles.",
        "",
        f"Objective: `{OBJECTIVE_NAME}`.",
        "",
        (
            "Scope: research profile selection only. This report does not prove fresh-gate "
            "or production readiness; check the corresponding frozen reliability status/gate "
            "reports before deployment."
        ),
        "",
        "## Profile Winners",
        "",
        _markdown_table(selected, selection_cols),
        "",
        "## Scored Candidates",
        "",
        _markdown_table(scored.sort_values("balanced_score", ascending=False), candidate_cols),
    ]
    (out_dir / "frozen_reliability_profile_selection.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-status", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.candidate_status, args.out_dir)
    print(json.dumps(_json_safe({"out_dir": payload["out_dir"], "candidate_count": payload["candidate_count"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
