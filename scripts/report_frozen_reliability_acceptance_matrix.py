#!/usr/bin/env python3
"""Report frozen reliability candidate acceptance status.

This is an artifact-only report.  It combines candidate-status CSVs with the
fresh-evidence gate and makes the reliability family tradeoff explicit:
drift, recent hit-rate surprise, OOD, and uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


REQUESTED_FAMILIES = ("drift", "recent_hit_rate_surprise", "ood", "uncertainty")


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


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    raw = frame[column]
    if raw.dtype == bool:
        return raw.fillna(default).astype(bool)
    return raw.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _source_name(path: Path) -> str:
    name = path.parent.name
    for suffix in ("_status_20260701", "_challenger_status_allroots_20260701"):
        if name.endswith(suffix):
            return name[: -len(suffix)] or name
    return name


def _families_from_text(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    if "any_bad_reliability" in text or "two_of_four" in text:
        return ",".join(REQUESTED_FAMILIES)
    families: List[str] = []
    if "drift" in text:
        families.append("drift")
    if "recent_hr" in text or "recent_hit_rate" in text or "recent_perf" in text:
        families.append("recent_hit_rate_surprise")
    if "ood" in text:
        families.append("ood")
    if "uncertainty" in text:
        families.append("uncertainty")
    return ",".join(family for family in REQUESTED_FAMILIES if family in set(families))


def _load_candidates(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "source_path", str(path))
        frame.insert(1, "source_name", _source_name(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "rule_id" not in out.columns:
        raise ValueError("candidate-status inputs must contain rule_id")
    out["rule_id"] = out["rule_id"].astype(str)
    if "families" not in out.columns:
        out["families"] = ""
    out["families"] = [
        str(families)
        if str(families or "").strip() and str(families).lower() != "nan"
        else _families_from_text(rule_id, role, note)
        for families, rule_id, role, note in zip(
            out["families"],
            out["rule_id"],
            out.get("role", pd.Series("", index=out.index)),
            out.get("promotion_note", pd.Series("", index=out.index)),
        )
    ]
    return out.drop_duplicates(subset=["source_name", "rule_id", "role"], keep="first")


def _fresh_gate(paths: Sequence[Path]) -> Dict[str, Any]:
    if not paths:
        return {
            "fresh_gate_known": False,
            "fresh_gate_pass": False,
            "fresh_blockers": "fresh_evidence_gaps_missing",
            "fresh_gaps": [],
        }
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "source_path", str(path))
        frames.append(frame)
    if not frames:
        return {
            "fresh_gate_known": False,
            "fresh_gate_pass": False,
            "fresh_blockers": "fresh_evidence_gaps_empty",
            "fresh_gaps": [],
        }
    gaps = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(
        subset=["gate", "head"], keep="first"
    )
    if "pass" not in gaps.columns:
        raise ValueError("fresh-evidence gap inputs must contain pass")
    pass_mask = _bool_series(gaps, "pass")
    blockers = gaps.loc[~pass_mask].copy()
    blocker_text = "none"
    if not blockers.empty:
        parts = []
        for _, row in blockers.iterrows():
            head = "" if pd.isna(row.get("head")) else str(row.get("head"))
            suffix = f":{head}" if head else ""
            parts.append(f"{row.get('gate')}{suffix}_deficit_{int(row.get('deficit') or 0)}")
        blocker_text = ";".join(parts)
    return {
        "fresh_gate_known": True,
        "fresh_gate_pass": bool(pass_mask.all()),
        "fresh_blockers": blocker_text,
        "fresh_gaps": gaps.to_dict(orient="records"),
    }


def _classify(row: pd.Series) -> str:
    if not bool(row.get("research_pass", False)):
        return "reject_research_gate"
    if not bool(row.get("bootstrap_gate_pass", False)):
        return "reject_bootstrap_gate"
    if not bool(row.get("pnl_gate_pass", False)):
        return "reject_nonpositive_pnl_objective"
    if not bool(row.get("fresh_gate_pass", False)):
        if bool(row.get("tail_clean_gate_pass", False)):
            return "tail_research_ready_wait_fresh"
        if bool(row.get("replacement_quality_gate_pass", False)):
            return "replacement_research_ready_wait_fresh"
        if bool(row.get("balanced_tail_gate_pass", False)):
            return "balanced_research_ready_wait_fresh"
        return "pnl_research_ready_wait_fresh"
    if bool(row.get("tail_clean_gate_pass", False)):
        return "production_tail_candidate"
    if bool(row.get("replacement_quality_gate_pass", False)):
        return "production_replacement_candidate"
    if bool(row.get("balanced_tail_gate_pass", False)):
        return "production_balanced_candidate"
    return "production_pnl_candidate"


def acceptance_matrix(
    candidates: pd.DataFrame,
    *,
    fresh_gate: Dict[str, Any],
    min_bootstrap_prob: float = 0.95,
    min_active_positive_week_share: float = 0.80,
    min_worst_week_delta: float = -150.0,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["research_pass"] = _bool_series(out, "research_pass")
    out["tail_clean"] = _bool_series(out, "tail_clean")
    out["delta_net_pnl"] = _num(out, "delta_net_pnl")
    out["delta_objective"] = _num(out, "delta_objective")
    out["active_positive_week_share"] = _num(out, "active_positive_week_share")
    out["worst_week_delta"] = _num(out, "worst_week_delta")
    out["prob_delta_net_pnl_positive"] = _num(out, "prob_delta_net_pnl_positive")
    out["prob_delta_objective_positive"] = _num(out, "prob_delta_objective_positive")
    out["delta_net_pnl_p05"] = _num(out, "delta_net_pnl_p05")
    out["delta_objective_p05"] = _num(out, "delta_objective_p05")
    out["entrant_minus_removed_hit_rate"] = _num(out, "entrant_minus_removed_hit_rate")
    out["entrant_minus_removed_full_sl_rate"] = _num(
        out, "entrant_minus_removed_full_sl_rate", default=np.nan
    )
    out["fresh_gate_known"] = bool(fresh_gate.get("fresh_gate_known", False))
    out["fresh_gate_pass"] = bool(fresh_gate.get("fresh_gate_pass", False))
    out["fresh_blockers"] = str(fresh_gate.get("fresh_blockers") or "")
    out["pnl_gate_pass"] = out["delta_net_pnl"].gt(0.0) & out["delta_objective"].gt(0.0)
    out["bootstrap_gate_pass"] = (
        out["prob_delta_net_pnl_positive"].ge(float(min_bootstrap_prob))
        & out["prob_delta_objective_positive"].ge(float(min_bootstrap_prob))
        & out["delta_net_pnl_p05"].gt(0.0)
        & out["delta_objective_p05"].gt(0.0)
    )
    out["tail_clean_gate_pass"] = out["tail_clean"] & out["worst_week_delta"].ge(0.0) & out[
        "active_positive_week_share"
    ].ge(1.0)
    out["balanced_tail_gate_pass"] = out["active_positive_week_share"].ge(
        float(min_active_positive_week_share)
    ) & out["worst_week_delta"].ge(float(min_worst_week_delta))
    out["replacement_quality_gate_pass"] = out["entrant_minus_removed_hit_rate"].gt(0.0) & (
        out["entrant_minus_removed_full_sl_rate"].isna()
        | out["entrant_minus_removed_full_sl_rate"].le(0.0)
    )
    for family in REQUESTED_FAMILIES:
        out[f"uses_{family}"] = out["families"].astype(str).str.contains(family, regex=False)
    out["acceptance_state"] = out.apply(_classify, axis=1)
    return out


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
    if "family" not in out.columns:
        raise ValueError("family-verdict inputs must contain family")
    out["family"] = out["family"].astype(str)
    return out.drop_duplicates(subset=["family"], keep="first")


def _load_ab_scorecards(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "ab_scorecard_source_path", str(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "variant" not in out.columns:
        raise ValueError("A/B scorecard inputs must contain variant")
    if "family" not in out.columns:
        out["family"] = ""
    out["family"] = out["family"].astype(str)
    for family in REQUESTED_FAMILIES:
        col = f"contains_{family}"
        if col not in out.columns:
            out[col] = out["family"].str.contains(family, regex=False)
    if "contains_recent_hit_rate_surprise" not in out.columns:
        out["contains_recent_hit_rate_surprise"] = out["family"].str.contains(
            "recent_hr|recent_hit_rate|recent_perf", regex=True
        )
    return out


def _load_marginal_family_ablation(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "marginal_source_path", str(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "family" not in out.columns:
        raise ValueError("marginal-family-ablation inputs must contain family")
    out["family"] = out["family"].astype(str)
    return out


def _ab_tail_gate(frame: pd.DataFrame) -> pd.Series:
    checks: List[pd.Series] = []
    if "delta_full_sl_rate" in frame.columns:
        checks.append(pd.to_numeric(frame["delta_full_sl_rate"], errors="coerce").le(0.0))
    for col in ("tail_metric", "delta_q20_pnl", "delta_q35_pnl"):
        if col in frame.columns:
            vals = pd.to_numeric(frame[col], errors="coerce")
            checks.append(vals.isna() | vals.ge(0.0))
    if not checks:
        return pd.Series(False, index=frame.index, dtype=bool)
    return pd.concat(checks, axis=1).all(axis=1)


def _ab_acceptance_state(row: pd.Series) -> str:
    if not bool(row.get("ab_pnl_gate_pass", False)):
        return "ab_reject_nonpositive_pnl_objective"
    if bool(row.get("ab_tail_gate_pass", False)):
        return "ab_pnl_tail_supportive"
    verdict = str(row.get("ab_verdict") or "")
    if "tail_mixed" in verdict:
        return "ab_pnl_tail_mixed"
    return "ab_pnl_tail_weak"


def ab_acceptance_matrix(scorecards: pd.DataFrame) -> pd.DataFrame:
    if scorecards.empty:
        return scorecards.copy()
    out = scorecards.copy()
    out["delta_net_pnl"] = pd.to_numeric(out.get("delta_net_pnl"), errors="coerce").fillna(0.0)
    out["delta_objective"] = pd.to_numeric(out.get("delta_objective"), errors="coerce").fillna(0.0)
    out["scorecard_score"] = pd.to_numeric(out.get("scorecard_score"), errors="coerce").fillna(0.0)
    out["ab_pnl_gate_pass"] = out["delta_net_pnl"].gt(0.0) & out["delta_objective"].gt(0.0)
    out["ab_tail_gate_pass"] = out["ab_pnl_gate_pass"] & _ab_tail_gate(out)
    for family in REQUESTED_FAMILIES:
        col = f"contains_{family}"
        if col not in out.columns:
            out[col] = out["family"].astype(str).str.contains(family, regex=False)
    out["ab_acceptance_state"] = out.apply(_ab_acceptance_state, axis=1)
    return out


def _summary_rows(
    matrix: pd.DataFrame,
    family_verdict: pd.DataFrame | None = None,
    ab_matrix: pd.DataFrame | None = None,
    marginal_ablation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if matrix.empty:
        matrix = pd.DataFrame(columns=["families", "delta_net_pnl", "entrant_minus_removed_hit_rate"])
    verdict = pd.DataFrame() if family_verdict is None else family_verdict.copy()
    ab = pd.DataFrame() if ab_matrix is None else ab_matrix.copy()
    marginal = pd.DataFrame() if marginal_ablation is None else marginal_ablation.copy()
    verdict_by_family = (
        verdict.set_index("family")
        if not verdict.empty and "family" in verdict.columns
        else pd.DataFrame()
    )
    rows = []
    for family in REQUESTED_FAMILIES:
        scoped = matrix.loc[matrix.get(f"uses_{family}", False).astype(bool)].copy()
        ab_scoped = ab.loc[ab.get(f"contains_{family}", False).astype(bool)].copy() if not ab.empty else pd.DataFrame()
        marginal_scoped = (
            marginal.loc[marginal.get("family", pd.Series(dtype=str)).astype(str).eq(family)].copy()
            if not marginal.empty
            else pd.DataFrame()
        )
        if not marginal_scoped.empty:
            for col in (
                "marginal_delta_net_pnl",
                "marginal_delta_objective",
                "marginal_delta_full_sl_rate",
                "marginal_delta_q20_pnl",
                "marginal_delta_q35_pnl",
                "marginal_scorecard_score",
            ):
                if col in marginal_scoped.columns:
                    marginal_scoped[col] = pd.to_numeric(marginal_scoped[col], errors="coerce")
            marginal_best = marginal_scoped.sort_values(
                ["marginal_scorecard_score", "marginal_delta_net_pnl"],
                ascending=[False, False],
                na_position="last",
            ).iloc[0]
        else:
            marginal_best = pd.Series(dtype=object)
        evidence = verdict_by_family.loc[family].to_dict() if family in verdict_by_family.index else {}
        rows.append(
            {
                "family": family,
                "candidate_count": int(len(scoped)),
                "research_pass_count": int(scoped.get("research_pass", pd.Series(dtype=bool)).astype(bool).sum()),
                "tail_clean_count": int(scoped.get("tail_clean_gate_pass", pd.Series(dtype=bool)).astype(bool).sum()),
                "replacement_quality_count": int(
                    scoped.get("replacement_quality_gate_pass", pd.Series(dtype=bool)).astype(bool).sum()
                ),
                "best_delta_net_pnl": float(scoped["delta_net_pnl"].max()) if not scoped.empty else np.nan,
                "best_entrant_minus_removed_hit_rate": float(scoped["entrant_minus_removed_hit_rate"].max())
                if not scoped.empty
                else np.nan,
                "best_acceptance_state": str(
                    scoped.sort_values(
                        ["research_pass", "delta_net_pnl", "entrant_minus_removed_hit_rate"],
                        ascending=[False, False, False],
                    ).iloc[0]["acceptance_state"]
                )
                if not scoped.empty
                else "not_tested_in_candidates",
                "scorecard_verdict": evidence.get("verdict", "not_available"),
                "finite_row_rate": evidence.get("finite_row_rate", np.nan),
                "tested_in_scorecards": evidence.get("tested_in_scorecards", np.nan),
                "scorecard_best_delta_net_pnl": evidence.get("best_long_window_delta_net_pnl", np.nan),
                "scorecard_best_q20_delta_pnl": evidence.get("best_q20_delta_pnl", np.nan),
                "ab_candidate_count": int(len(ab_scoped)),
                "ab_tail_supportive_count": int(
                    ab_scoped.get("ab_tail_gate_pass", pd.Series(dtype=bool)).astype(bool).sum()
                ),
                "ab_best_delta_net_pnl": float(ab_scoped["delta_net_pnl"].max()) if not ab_scoped.empty else np.nan,
                "ab_best_scorecard_score": float(ab_scoped["scorecard_score"].max()) if not ab_scoped.empty else np.nan,
                "ab_best_acceptance_state": str(
                    ab_scoped.sort_values(
                        ["ab_tail_gate_pass", "scorecard_score", "delta_net_pnl"],
                        ascending=[False, False, False],
                    ).iloc[0]["ab_acceptance_state"]
                )
                if not ab_scoped.empty
                else "not_available",
                "marginal_test_count": int(len(marginal_scoped)),
                "best_marginal_delta_net_pnl": marginal_best.get("marginal_delta_net_pnl", np.nan),
                "best_marginal_delta_objective": marginal_best.get("marginal_delta_objective", np.nan),
                "best_marginal_delta_q20_pnl": marginal_best.get("marginal_delta_q20_pnl", np.nan),
                "best_marginal_delta_q35_pnl": marginal_best.get("marginal_delta_q35_pnl", np.nan),
                "best_marginal_scorecard_score": marginal_best.get("marginal_scorecard_score", np.nan),
                "best_marginal_variant": marginal_best.get("variant", ""),
                "best_marginal_baseline_variant": marginal_best.get("baseline_variant", ""),
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
    fresh_evidence_gap: Sequence[Path],
    family_verdict: Sequence[Path] = (),
    ab_scorecard: Sequence[Path] = (),
    marginal_family_ablation: Sequence[Path] = (),
    out_dir: Path,
    min_bootstrap_prob: float = 0.95,
    min_active_positive_week_share: float = 0.80,
    min_worst_week_delta: float = -150.0,
) -> Dict[str, Any]:
    candidates = _load_candidates(candidate_status)
    fresh = _fresh_gate(fresh_evidence_gap)
    family_evidence = _load_family_verdict(family_verdict)
    ab_matrix = ab_acceptance_matrix(_load_ab_scorecards(ab_scorecard))
    marginal = _load_marginal_family_ablation(marginal_family_ablation)
    matrix = acceptance_matrix(
        candidates,
        fresh_gate=fresh,
        min_bootstrap_prob=min_bootstrap_prob,
        min_active_positive_week_share=min_active_positive_week_share,
        min_worst_week_delta=min_worst_week_delta,
    )
    summary = _summary_rows(matrix, family_evidence, ab_matrix, marginal)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_dir / "frozen_reliability_acceptance_matrix.csv", index=False)
    ab_matrix.to_csv(out_dir / "frozen_reliability_ab_acceptance_matrix.csv", index=False)
    marginal.to_csv(out_dir / "frozen_reliability_marginal_family_acceptance_input.csv", index=False)
    summary.to_csv(out_dir / "frozen_reliability_acceptance_family_summary.csv", index=False)
    payload = {
        "generated_by": Path(__file__).name,
        "candidate_status": [str(path) for path in candidate_status],
        "fresh_evidence_gap": [str(path) for path in fresh_evidence_gap],
        "family_verdict": [str(path) for path in family_verdict],
        "ab_scorecard": [str(path) for path in ab_scorecard],
        "marginal_family_ablation": [str(path) for path in marginal_family_ablation],
        "fresh_gate": fresh,
        "candidate_count": int(len(matrix)),
        "ab_candidate_count": int(len(ab_matrix)),
        "production_ready_count": int(matrix["acceptance_state"].astype(str).str.startswith("production_").sum())
        if not matrix.empty
        else 0,
        "parameters": {
            "min_bootstrap_prob": float(min_bootstrap_prob),
            "min_active_positive_week_share": float(min_active_positive_week_share),
            "min_worst_week_delta": float(min_worst_week_delta),
        },
    }
    (out_dir / "frozen_reliability_acceptance_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Frozen Reliability Acceptance Matrix",
        "",
        "This report classifies already-evaluated frozen reliability candidates. It does not replay or refit.",
        "",
        f"Fresh gate pass: `{fresh.get('fresh_gate_pass')}`",
        f"Fresh blockers: `{fresh.get('fresh_blockers')}`",
        "",
        "## Candidate Acceptance",
        "",
        _markdown_table(
            matrix.sort_values(["research_pass", "delta_net_pnl"], ascending=[False, False]),
            [
                "rule_id",
                "source_name",
                "families",
                "acceptance_state",
                "delta_net_pnl",
                "delta_objective",
                "active_positive_week_share",
                "worst_week_delta",
                "entrant_minus_removed_hit_rate",
                "entrant_minus_removed_full_sl_rate",
                "tail_clean_gate_pass",
                "balanced_tail_gate_pass",
                "replacement_quality_gate_pass",
                "fresh_gate_pass",
            ],
        ),
        "",
        "## Family Summary",
        "",
        _markdown_table(
            summary,
            [
                "family",
                "candidate_count",
                "research_pass_count",
                "tail_clean_count",
                "replacement_quality_count",
                "best_delta_net_pnl",
                "best_entrant_minus_removed_hit_rate",
                "best_acceptance_state",
                "scorecard_verdict",
                "finite_row_rate",
                "tested_in_scorecards",
                "scorecard_best_delta_net_pnl",
                "scorecard_best_q20_delta_pnl",
                "ab_candidate_count",
                "ab_tail_supportive_count",
                "ab_best_delta_net_pnl",
                "ab_best_scorecard_score",
                "ab_best_acceptance_state",
                "marginal_test_count",
                "best_marginal_delta_net_pnl",
                "best_marginal_delta_objective",
                "best_marginal_delta_q20_pnl",
                "best_marginal_delta_q35_pnl",
                "best_marginal_scorecard_score",
                "best_marginal_variant",
                "best_marginal_baseline_variant",
            ],
        ),
        "",
        "## A/B Scorecard Acceptance",
        "",
        _markdown_table(
            ab_matrix.sort_values(["ab_tail_gate_pass", "scorecard_score"], ascending=[False, False]),
            [
                "source",
                "evidence_family",
                "variant",
                "family",
                "ab_acceptance_state",
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
            ],
        ),
    ]
    (out_dir / "frozen_reliability_acceptance_matrix.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-status", type=Path, action="append", required=True)
    parser.add_argument("--fresh-evidence-gap", type=Path, action="append", default=[])
    parser.add_argument("--family-verdict", type=Path, action="append", default=[])
    parser.add_argument("--ab-scorecard", type=Path, action="append", default=[])
    parser.add_argument("--marginal-family-ablation", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-bootstrap-prob", type=float, default=0.95)
    parser.add_argument("--min-active-positive-week-share", type=float, default=0.80)
    parser.add_argument("--min-worst-week-delta", type=float, default=-150.0)
    args = parser.parse_args()
    payload = run(
        candidate_status=args.candidate_status,
        fresh_evidence_gap=args.fresh_evidence_gap,
        family_verdict=args.family_verdict,
        ab_scorecard=args.ab_scorecard,
        marginal_family_ablation=args.marginal_family_ablation,
        out_dir=args.out_dir,
        min_bootstrap_prob=args.min_bootstrap_prob,
        min_active_positive_week_share=args.min_active_positive_week_share,
        min_worst_week_delta=args.min_worst_week_delta,
    )
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "candidate_count": payload["candidate_count"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
