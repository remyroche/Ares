#!/usr/bin/env python3
"""Map reliability-family A/B evidence to freezeable rule candidates.

The acceptance matrix says whether already-frozen candidates pass.  This report
answers the next question: when an A/B scorecard shows OOD or uncertainty
evidence, is there an explicit conditional-filter rule that can be promoted into
a frozen candidate, or is the evidence still diagnostic-only?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


FAMILIES = ("drift", "recent_hit_rate_surprise", "ood", "uncertainty")
OBJECTIVE_COL = "objective_avgweek_0p7dayq35_0p3dayq20"


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


def _families_from_text(*parts: Any) -> List[str]:
    text = " ".join(str(part or "") for part in parts).lower()
    if "any_bad_reliability" in text or "two_signal" in text or "two_of_four" in text:
        return list(FAMILIES)
    found: List[str] = []
    if "drift" in text:
        found.append("drift")
    if "recent_hr" in text or "recent_hit_rate" in text or "recent_perf" in text:
        found.append("recent_hit_rate_surprise")
    if "ood" in text:
        found.append("ood")
    if "uncertainty" in text:
        found.append("uncertainty")
    return [family for family in FAMILIES if family in set(found)]


def _parse_rule_spec(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _load_candidate_status(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "candidate_status_source", str(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["rule_id"] = out.get("rule_id", "").astype(str)
    out["families"] = [
        ",".join(_families_from_text(rule_id, role, note))
        for rule_id, role, note in zip(
            out["rule_id"],
            out.get("role", pd.Series("", index=out.index)),
            out.get("promotion_note", pd.Series("", index=out.index)),
        )
    ]
    for col in ("delta_net_pnl", "delta_objective", "entrant_minus_removed_hit_rate"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_conditional_rules(paths: Sequence[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty or "rule_id" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            spec = _parse_rule_spec(row.get("rule_spec"))
            condition = spec.get("condition", "")
            families = _families_from_text(row.get("rule_id"), condition)
            if not families:
                continue
            rows.append(
                {
                    "conditional_source": str(path),
                    "rule_id": str(row.get("rule_id")),
                    "families": ",".join(families),
                    "family_count": int(len(families)),
                    "condition": condition,
                    "heads": ",".join(str(x) for x in (spec.get("heads") or [])),
                    "action": spec.get("action", ""),
                    "value": spec.get("value", np.nan),
                    "objective": pd.to_numeric(row.get(OBJECTIVE_COL), errors="coerce"),
                    "rule_spec": json.dumps(spec, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def _load_family_verdict(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "family_verdict_source", str(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates(subset=["family"], keep="first") if "family" in out.columns else pd.DataFrame()


def _load_marginal(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "marginal_source", str(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in (
        "marginal_delta_net_pnl",
        "marginal_delta_objective",
        "marginal_delta_q20_pnl",
        "marginal_scorecard_score",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _best_by_family(frame: pd.DataFrame, family: str, score_cols: Sequence[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    mask = frame.get("families", pd.Series("", index=frame.index)).astype(str).str.contains(family, regex=False)
    scoped = frame.loc[mask].copy()
    if scoped.empty:
        return pd.Series(dtype=object)
    sort_cols = [col for col in score_cols if col in scoped.columns]
    if not sort_cols:
        return scoped.iloc[0]
    return scoped.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").iloc[0]


def _best_focused_rule(frame: pd.DataFrame, family: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    mask = frame.get("families", pd.Series("", index=frame.index)).astype(str).str.contains(family, regex=False)
    scoped = frame.loc[mask].copy()
    if scoped.empty:
        return pd.Series(dtype=object)
    if "family_count" not in scoped.columns:
        scoped["family_count"] = scoped["families"].astype(str).str.count(",") + 1
    return scoped.sort_values(
        ["family_count", "objective"],
        ascending=[True, False],
        na_position="last",
    ).iloc[0]


def _best_marginal(frame: pd.DataFrame, family: str) -> pd.Series:
    if frame.empty or "family" not in frame.columns:
        return pd.Series(dtype=object)
    scoped = frame.loc[frame["family"].astype(str).eq(family)].copy()
    if scoped.empty:
        return pd.Series(dtype=object)
    sort_cols = [col for col in ("marginal_scorecard_score", "marginal_delta_net_pnl") if col in scoped.columns]
    return scoped.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").iloc[0]


def _decision(
    *,
    family: str,
    frozen: pd.Series,
    explicit: pd.Series,
    verdict: pd.Series,
    marginal: pd.Series,
) -> str:
    frozen_rule = str(frozen.get("rule_id", "") or "")
    if frozen_rule:
        return "already_frozen_candidate_wait_fresh"
    verdict_text = str(verdict.get("verdict", "") or "")
    marginal_net = float(marginal.get("marginal_delta_net_pnl", np.nan))
    marginal_obj = float(marginal.get("marginal_delta_objective", np.nan))
    marginal_q20 = float(marginal.get("marginal_delta_q20_pnl", np.nan))
    explicit_rule = str(explicit.get("rule_id", "") or "")
    if "tested_no_clear_lift" in verdict_text or (np.isfinite(marginal_net) and marginal_net <= 0.0):
        return "diagnostic_only_no_positive_marginal_lift"
    if explicit_rule and np.isfinite(marginal_net) and marginal_net > 0.0 and np.isfinite(marginal_obj) and marginal_obj > 0.0:
        if np.isfinite(marginal_q20) and marginal_q20 < 0.0:
            return "explicit_rule_available_tail_mixed_needs_multiwindow"
        return "explicit_rule_available_needs_multiwindow"
    if explicit_rule:
        return "explicit_rule_available_but_marginal_unproven"
    if family == "uncertainty":
        return "diagnostic_only_no_freeze_rule"
    return "scorecard_only_needs_rule_materialization"


def freezeability_rows(
    *,
    candidate_status: pd.DataFrame,
    conditional_rules: pd.DataFrame,
    family_verdict: pd.DataFrame,
    marginal: pd.DataFrame,
) -> pd.DataFrame:
    verdict_by_family = (
        family_verdict.set_index("family") if not family_verdict.empty and "family" in family_verdict.columns else pd.DataFrame()
    )
    rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        frozen = _best_by_family(candidate_status, family, ("delta_net_pnl", "delta_objective"))
        explicit = _best_focused_rule(conditional_rules, family)
        broad_explicit = _best_by_family(conditional_rules, family, ("objective",))
        verdict = verdict_by_family.loc[family] if family in verdict_by_family.index else pd.Series(dtype=object)
        marginal_best = _best_marginal(marginal, family)
        rows.append(
            {
                "family": family,
                "freezeability_decision": _decision(
                    family=family,
                    frozen=frozen,
                    explicit=explicit,
                    verdict=verdict,
                    marginal=marginal_best,
                ),
                "frozen_rule_id": frozen.get("rule_id", ""),
                "frozen_delta_net_pnl": frozen.get("delta_net_pnl", np.nan),
                "frozen_delta_objective": frozen.get("delta_objective", np.nan),
                "explicit_rule_id": explicit.get("rule_id", ""),
                "explicit_condition": explicit.get("condition", ""),
                "explicit_heads": explicit.get("heads", ""),
                "explicit_action": explicit.get("action", ""),
                "explicit_value": explicit.get("value", np.nan),
                "explicit_objective": explicit.get("objective", np.nan),
                "explicit_family_count": explicit.get("family_count", np.nan),
                "broad_explicit_rule_id": broad_explicit.get("rule_id", ""),
                "broad_explicit_condition": broad_explicit.get("condition", ""),
                "broad_explicit_heads": broad_explicit.get("heads", ""),
                "broad_explicit_objective": broad_explicit.get("objective", np.nan),
                "broad_explicit_family_count": broad_explicit.get("family_count", np.nan),
                "scorecard_verdict": verdict.get("verdict", ""),
                "finite_row_rate": verdict.get("finite_row_rate", np.nan),
                "marginal_variant": marginal_best.get("variant", ""),
                "marginal_baseline_variant": marginal_best.get("baseline_variant", ""),
                "marginal_delta_net_pnl": marginal_best.get("marginal_delta_net_pnl", np.nan),
                "marginal_delta_objective": marginal_best.get("marginal_delta_objective", np.nan),
                "marginal_delta_q20_pnl": marginal_best.get("marginal_delta_q20_pnl", np.nan),
                "marginal_scorecard_score": marginal_best.get("marginal_scorecard_score", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, cols: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    use_cols = [col for col in cols if col in frame.columns]
    return frame[use_cols].round(6).to_markdown(index=False)


def run(
    *,
    candidate_status: Sequence[Path],
    conditional_summary: Sequence[Path],
    family_verdict: Sequence[Path],
    marginal_family_ablation: Sequence[Path],
    out_dir: Path,
) -> Dict[str, Any]:
    candidates = _load_candidate_status(candidate_status)
    rules = _load_conditional_rules(conditional_summary)
    verdict = _load_family_verdict(family_verdict)
    marginal = _load_marginal(marginal_family_ablation)
    rows = freezeability_rows(
        candidate_status=candidates,
        conditional_rules=rules,
        family_verdict=verdict,
        marginal=marginal,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "reliability_family_freezeability.csv", index=False)
    rules.to_csv(out_dir / "reliability_explicit_rule_candidates.csv", index=False)
    payload = {
        "generated_by": Path(__file__).name,
        "candidate_status": [str(path) for path in candidate_status],
        "conditional_summary": [str(path) for path in conditional_summary],
        "family_verdict": [str(path) for path in family_verdict],
        "marginal_family_ablation": [str(path) for path in marginal_family_ablation],
        "family_count": int(len(rows)),
        "explicit_rule_count": int(len(rules)),
    }
    (out_dir / "reliability_family_freezeability_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Reliability Family Freezeability",
        "",
        "This report maps A/B evidence to deployable conditional-filter rules. It does not replay trades.",
        "",
        "## Family Decisions",
        "",
        _markdown_table(
            rows,
            [
                "family",
                "freezeability_decision",
                "frozen_rule_id",
                "frozen_delta_net_pnl",
                "explicit_rule_id",
                "explicit_condition",
                "explicit_heads",
                "explicit_action",
                "explicit_value",
                "explicit_objective",
                "explicit_family_count",
                "broad_explicit_rule_id",
                "broad_explicit_condition",
                "broad_explicit_objective",
                "broad_explicit_family_count",
                "scorecard_verdict",
                "finite_row_rate",
                "marginal_variant",
                "marginal_baseline_variant",
                "marginal_delta_net_pnl",
                "marginal_delta_objective",
                "marginal_delta_q20_pnl",
            ],
        ),
        "",
        "## Explicit Rule Candidates",
        "",
        _markdown_table(
            rules.sort_values("objective", ascending=False, na_position="last").head(30),
            ["rule_id", "families", "condition", "heads", "action", "value", "objective", "conditional_source"],
        ),
    ]
    (out_dir / "reliability_family_freezeability.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-status", type=Path, action="append", default=[])
    parser.add_argument("--conditional-summary", type=Path, action="append", required=True)
    parser.add_argument("--family-verdict", type=Path, action="append", default=[])
    parser.add_argument("--marginal-family-ablation", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        candidate_status=args.candidate_status,
        conditional_summary=args.conditional_summary,
        family_verdict=args.family_verdict,
        marginal_family_ablation=args.marginal_family_ablation,
        out_dir=args.out_dir,
    )
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), **payload}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
