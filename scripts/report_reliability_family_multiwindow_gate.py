#!/usr/bin/env python3
"""Consolidate reliability-family multi-window validation summaries.

This report consumes existing ``multiwindow_candidate_summary.csv`` artifacts.
It is intended to answer whether a family such as OOD has enough broad
multi-window support to move from diagnostic evidence to a frozen candidate.
No replay or model fitting is performed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


FAMILIES = ("drift", "recent_hit_rate_surprise", "ood", "uncertainty")


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


def _families_from_rule(rule_id: Any) -> List[str]:
    text = str(rule_id or "").lower()
    if "two_of_four" in text or "two_signal" in text or "any_bad_reliability" in text:
        return list(FAMILIES)
    found: List[str] = []
    if "drift" in text:
        found.append("drift")
    if "recent_hr" in text or "recent_rank" in text or "recent_perf" in text:
        found.append("recent_hit_rate_surprise")
    if "ood" in text:
        found.append("ood")
    if "uncertainty" in text:
        found.append("uncertainty")
    return [family for family in FAMILIES if family in set(found)]


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _load_multiwindow(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty or "rule_id" not in frame.columns:
            continue
        frame = frame.copy()
        frame.insert(0, "multiwindow_source", str(path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["rule_id"] = out["rule_id"].astype(str)
    out["families"] = [",".join(_families_from_rule(rule_id)) for rule_id in out["rule_id"]]
    out["family_count"] = out["families"].where(out["families"].ne(""), "").str.count(",") + out[
        "families"
    ].ne("").astype(int)
    for col in (
        "core_pnl_tail_gate_count",
        "core_strict_tail_gate_count",
        "core_min_delta_objective",
        "core_min_delta_net_pnl",
        "core_min_delta_weekly_q20",
        "core_min_delta_weighted_daily_tail",
        "full_delta_objective",
        "full_delta_net_pnl",
        "full_delta_weekly_q20",
        "full_delta_weighted_daily_tail",
        "june_delta_objective",
        "june_delta_net_pnl",
        "june_delta_weekly_q20",
        "entrant_minus_removed_hit_rate",
        "entrant_minus_removed_net_pnl",
    ):
        out[col] = _num(out, col)
    return out


def _gate_state(row: pd.Series, min_core_tail_gates: int) -> str:
    if float(row.get("full_delta_net_pnl", 0.0)) <= 0.0 or float(row.get("full_delta_objective", 0.0)) <= 0.0:
        return "reject_nonpositive_full"
    if int(row.get("core_pnl_tail_gate_count", 0)) >= int(min_core_tail_gates) and float(
        row.get("core_min_delta_weekly_q20", 0.0)
    ) >= 0.0:
        if int(row.get("core_strict_tail_gate_count", 0)) >= int(min_core_tail_gates):
            return "multiwindow_strict_tail_pass"
        return "multiwindow_tail_pass"
    if int(row.get("core_pnl_tail_gate_count", 0)) >= max(1, int(min_core_tail_gates) - 1):
        return "multiwindow_mixed_positive"
    return "reject_tail_instability"


def candidate_gate(frame: pd.DataFrame, min_core_tail_gates: int = 3) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["multiwindow_gate_state"] = out.apply(_gate_state, axis=1, min_core_tail_gates=min_core_tail_gates)
    out["family_focus_score"] = 1.0 / out["family_count"].replace(0, np.nan)
    out["family_focus_score"] = out["family_focus_score"].fillna(0.0)
    out["selection_score"] = (
        out["core_pnl_tail_gate_count"].clip(lower=0.0)
        + out["core_strict_tail_gate_count"].clip(lower=0.0)
        + 0.01 * out["full_delta_objective"].clip(lower=0.0)
        + 0.001 * out["full_delta_net_pnl"].clip(lower=0.0)
        + out["family_focus_score"]
    )
    return out.sort_values(
        ["multiwindow_gate_state", "selection_score"],
        ascending=[True, False],
    )


def _family_summary(gated: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    gate_rank = {
        "multiwindow_strict_tail_pass": 4,
        "multiwindow_tail_pass": 3,
        "multiwindow_mixed_positive": 2,
        "reject_tail_instability": 1,
        "reject_nonpositive_full": 0,
    }
    for family in FAMILIES:
        scoped = gated.loc[gated["families"].astype(str).str.contains(family, regex=False)].copy()
        if scoped.empty:
            rows.append(
                {
                    "family": family,
                    "candidate_count": 0,
                    "strict_tail_pass_count": 0,
                    "tail_pass_count": 0,
                    "focused_strict_tail_pass_count": 0,
                    "focused_tail_pass_count": 0,
                    "composite_tail_pass_count": 0,
                    "mixed_positive_count": 0,
                    "best_rule_id": "",
                    "best_gate_state": "not_tested",
                    "best_full_delta_net_pnl": np.nan,
                    "best_full_delta_objective": np.nan,
                    "best_core_min_weekly_q20": np.nan,
                    "best_family_count": np.nan,
                    "recommendation": "not_tested",
                }
            )
            continue
        scoped["focused_family_rule"] = scoped["family_count"].le(2)
        scoped["gate_rank"] = scoped["multiwindow_gate_state"].map(gate_rank).fillna(0).astype(float)
        tail_mask = scoped["multiwindow_gate_state"].isin(["multiwindow_strict_tail_pass", "multiwindow_tail_pass"])
        focused_tail = scoped.loc[tail_mask & scoped["focused_family_rule"]].copy()
        if not focused_tail.empty:
            best_pool = focused_tail
        elif tail_mask.any():
            best_pool = scoped.loc[tail_mask].copy()
        elif scoped["multiwindow_gate_state"].eq("multiwindow_mixed_positive").any():
            best_pool = scoped.loc[scoped["multiwindow_gate_state"].eq("multiwindow_mixed_positive")].copy()
        else:
            best_pool = scoped
        best = best_pool.sort_values(
            ["gate_rank", "family_count", "selection_score", "full_delta_net_pnl"],
            ascending=[False, True, False, False],
        ).iloc[0]
        strict_count = int(scoped["multiwindow_gate_state"].eq("multiwindow_strict_tail_pass").sum())
        tail_count = int(
            scoped["multiwindow_gate_state"].isin(["multiwindow_strict_tail_pass", "multiwindow_tail_pass"]).sum()
        )
        focused_strict_count = int(
            (
                scoped["multiwindow_gate_state"].eq("multiwindow_strict_tail_pass")
                & scoped["focused_family_rule"]
            ).sum()
        )
        focused_tail_count = int((tail_mask & scoped["focused_family_rule"]).sum())
        composite_tail_count = int((tail_mask & ~scoped["focused_family_rule"]).sum())
        mixed_count = int(scoped["multiwindow_gate_state"].eq("multiwindow_mixed_positive").sum())
        if focused_strict_count > 0:
            rec = "freeze_candidate_available"
        elif focused_tail_count > 0:
            rec = "tail_pass_candidate_available"
        elif tail_count > 0:
            rec = "composite_only_needs_family_isolation"
        elif mixed_count > 0:
            rec = "research_only_tail_mixed"
        else:
            rec = "diagnostic_only_reject_multiwindow"
        rows.append(
            {
                "family": family,
                "candidate_count": int(len(scoped)),
                "strict_tail_pass_count": strict_count,
                "tail_pass_count": tail_count,
                "focused_strict_tail_pass_count": focused_strict_count,
                "focused_tail_pass_count": focused_tail_count,
                "composite_tail_pass_count": composite_tail_count,
                "mixed_positive_count": mixed_count,
                "best_rule_id": best.get("rule_id", ""),
                "best_gate_state": best.get("multiwindow_gate_state", ""),
                "best_full_delta_net_pnl": best.get("full_delta_net_pnl", np.nan),
                "best_full_delta_objective": best.get("full_delta_objective", np.nan),
                "best_core_min_weekly_q20": best.get("core_min_delta_weekly_q20", np.nan),
                "best_family_count": best.get("family_count", np.nan),
                "recommendation": rec,
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, cols: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    use_cols = [col for col in cols if col in frame.columns]
    return frame[use_cols].round(6).to_markdown(index=False)


def run(*, multiwindow_summary: Sequence[Path], out_dir: Path, min_core_tail_gates: int = 3) -> Dict[str, Any]:
    loaded = _load_multiwindow(multiwindow_summary)
    gated = candidate_gate(loaded, min_core_tail_gates=min_core_tail_gates)
    summary = _family_summary(gated)
    out_dir.mkdir(parents=True, exist_ok=True)
    gated.to_csv(out_dir / "reliability_family_multiwindow_candidates.csv", index=False)
    summary.to_csv(out_dir / "reliability_family_multiwindow_summary.csv", index=False)
    payload = {
        "generated_by": Path(__file__).name,
        "multiwindow_summary": [str(path) for path in multiwindow_summary],
        "min_core_tail_gates": int(min_core_tail_gates),
        "candidate_count": int(len(gated)),
        "family_count": int(len(summary)),
    }
    (out_dir / "reliability_family_multiwindow_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Reliability Family Multi-Window Gate",
        "",
        "This report consolidates existing multi-window candidate summaries. It does not replay trades.",
        "",
        "## Family Summary",
        "",
        _markdown_table(
            summary,
            [
                "family",
                "recommendation",
                "candidate_count",
                "strict_tail_pass_count",
                "tail_pass_count",
                "focused_strict_tail_pass_count",
                "focused_tail_pass_count",
                "composite_tail_pass_count",
                "mixed_positive_count",
                "best_rule_id",
                "best_gate_state",
                "best_full_delta_net_pnl",
                "best_full_delta_objective",
                "best_core_min_weekly_q20",
                "best_family_count",
            ],
        ),
        "",
        "## Candidate Gate",
        "",
        _markdown_table(
            gated.sort_values(["selection_score", "full_delta_net_pnl"], ascending=[False, False]).head(40),
            [
                "rule_id",
                "families",
                "family_count",
                "multiwindow_gate_state",
                "core_pnl_tail_gate_count",
                "core_strict_tail_gate_count",
                "core_min_delta_objective",
                "core_min_delta_net_pnl",
                "core_min_delta_weekly_q20",
                "full_delta_net_pnl",
                "full_delta_objective",
                "june_delta_net_pnl",
                "june_delta_objective",
                "entrant_minus_removed_hit_rate",
                "multiwindow_source",
            ],
        ),
    ]
    (out_dir / "reliability_family_multiwindow_gate.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiwindow-summary", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-core-tail-gates", type=int, default=3)
    args = parser.parse_args()
    payload = run(
        multiwindow_summary=args.multiwindow_summary,
        out_dir=args.out_dir,
        min_core_tail_gates=args.min_core_tail_gates,
    )
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), **payload}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
