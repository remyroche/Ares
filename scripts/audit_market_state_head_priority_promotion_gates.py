#!/usr/bin/env python3
"""Audit promotion gates for market-state head-priority modulation.

Head-priority modulation is an opportunity-allocation mechanism: it may replace
one accepted trade with another. Some shadow arms may also test a tiny
pre-filter rank-prior. This audit therefore separates a single-window replay
mechanics pass from production promotion. Production remains blocked when the
candidate changes rank contracts, uses a broad diagnostic candidate universe,
or is evaluated only on the June development window.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PRIORITY_DIR = Path(
    "data_perp/reports/market_state_head_priority_learning_topcandidate_forced_shadow_20260626_jun15_22_v2"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_head_priority_promotion_gate_audit_20260626"
)
BASELINE_ARM = "P0_static_priority"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _num(value: Any, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _row_by_arm(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    if frame.empty or "arm" not in frame.columns:
        return {}
    rows = frame.loc[frame["arm"].astype(str).eq(str(arm))]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _swap_row(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    work = frame.loc[frame.get("arm", pd.Series(dtype=str)).astype(str).eq(str(arm))]
    if "scope" in work.columns:
        work = work.loc[work["scope"].astype(str).eq("all")]
    if work.empty:
        return {}
    return work.iloc[0].to_dict()


def _diagnostic_row(frame: pd.DataFrame, arm: str | None = None) -> dict[str, Any]:
    if frame.empty:
        return {}
    if arm and "arm" in frame.columns:
        rows = frame.loc[frame["arm"].astype(str).eq(str(arm))]
        if not rows.empty:
            return rows.iloc[0].to_dict()
    return frame.iloc[0].to_dict()


def _selection_gate_passed(diagnostics: pd.DataFrame, selection: pd.DataFrame, arm: str) -> bool:
    diag = _diagnostic_row(diagnostics, arm)
    if "selection_gate_passed" in diag:
        return _bool(diag.get("selection_gate_passed"))
    if not selection.empty and "selection_gate_passed" in selection.columns:
        return bool(selection["selection_gate_passed"].map(_bool).any())
    return False


def _fail_reasons(
    *,
    selection_gate_passed: bool,
    base: dict[str, Any],
    candidate: dict[str, Any],
    swap: dict[str, Any],
    diag: dict[str, Any],
    min_jaccard: float,
    min_trade_retention: float,
    min_fold_action_positive_share: float,
    max_full_sl_delta: float,
    max_timeout_delta: float,
    gate_mode: str,
) -> list[str]:
    gate_mode = str(gate_mode or "defensive").strip().lower()
    reasons: list[str] = []
    if not selection_gate_passed:
        reasons.append("selection_gate_not_passed")
    if not base:
        reasons.append("missing_static_baseline_row")
    if not candidate:
        reasons.append("missing_candidate_row")
        return reasons

    base_trades = _num(base.get("trade_count"), 0.0)
    candidate_trades = _num(candidate.get("trade_count"), 0.0)
    retention = candidate_trades / max(base_trades, 1.0)
    net_delta = _num(candidate.get("net_pnl"), 0.0) - _num(base.get("net_pnl"), 0.0)
    full_sl_delta = _num(candidate.get("full_sl_rate")) - _num(base.get("full_sl_rate"))
    timeout_delta = _num(candidate.get("timeout_rate")) - _num(base.get("timeout_rate"))

    if not np.isfinite(net_delta) or net_delta <= 0.0:
        reasons.append("net_pnl_delta_not_positive")
    if not np.isfinite(retention) or retention < float(min_trade_retention):
        reasons.append("trade_retention_below_gate")
    if np.isfinite(full_sl_delta) and full_sl_delta > float(max_full_sl_delta):
        reasons.append("full_sl_rate_worsened")
    if np.isfinite(timeout_delta) and timeout_delta > float(max_timeout_delta):
        reasons.append("timeout_rate_worsened")

    jaccard = _num(swap.get("jaccard_vs_baseline"), np.nan)
    if not np.isfinite(jaccard):
        # The accepted-swap utility file does not carry jaccard; the caller may
        # attach it before evaluating reasons. Missing jaccard is weak evidence.
        reasons.append("missing_accepted_jaccard")
    elif jaccard < float(min_jaccard):
        reasons.append("accepted_jaccard_below_gate")

    entrants = int(_num(swap.get("entrants"), 0.0))
    removed = int(_num(swap.get("removed"), 0.0))
    if entrants + removed <= 0:
        reasons.append("no_accepted_set_movement")
    if _num(swap.get("net_replacement_pnl"), 0.0) <= 0.0:
        reasons.append("net_replacement_pnl_not_positive")
    if _num(swap.get("net_action_pnl_delta"), 0.0) <= 0.0:
        reasons.append("net_action_pnl_delta_not_positive")
    if _num(swap.get("entrant_net_pnl"), 0.0) <= _num(swap.get("removed_net_pnl"), 0.0):
        reasons.append("entrants_not_better_than_removed")

    fold_share = _num(diag.get("fold_action_positive_delta_share"), np.nan)
    fold_delta = _num(diag.get("fold_mean_action_utility_delta"), np.nan)
    if not np.isfinite(fold_share) or fold_share < float(min_fold_action_positive_share):
        reasons.append("fold_action_positive_share_below_gate")
    if not np.isfinite(fold_delta) or fold_delta <= 0.0:
        reasons.append("fold_action_utility_delta_not_positive")

    return reasons


def _production_blockers(manifest: dict[str, Any], *, window_count: int = 1) -> list[str]:
    contract = dict(manifest.get("contract") or {})
    parity = dict(manifest.get("static_baseline_candidate_parity") or {})
    universe = dict(manifest.get("candidate_universe") or {})
    blockers: list[str] = []
    if contract.get("operational_status") != "shadow_only":
        blockers.append("contract_not_shadow_only")
    if contract.get("execution_enabled") is not False:
        blockers.append("execution_not_disabled")
    if contract.get("qfail_active") is not False:
        blockers.append("qfail_active")
    if contract.get("head_health_active") is not False:
        blockers.append("head_health_active")
    if contract.get("market_state_threshold_controller_active") is not False:
        blockers.append("threshold_controller_active")
    if contract.get("changes_thresholds") is not False:
        blockers.append("changes_thresholds")
    if contract.get("changes_position_sizing") is not False:
        blockers.append("changes_position_sizing")
    if contract.get("changes_scores_or_ranks") is True:
        blockers.append("changes_scores_or_ranks_rank_prior_shadow_only")
    if parity.get("promotion_grade_scope") is not True:
        blockers.append("candidate_universe_not_promotion_grade")
    if int(window_count) < 3:
        blockers.append("fewer_than_3_replay_windows")
    start_raw = universe.get("timestamp_min")
    end_raw = universe.get("timestamp_max")
    if start_raw and end_raw:
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
        if start >= pd.Timestamp("2026-06-15T00:00:00Z") and end <= pd.Timestamp(
            "2026-06-22T23:59:59Z"
        ):
            blockers.append("june_15_22_development_window_not_promotion_oos")
    return blockers


def audit_priority_gates(
    priority_dir: Path,
    *,
    gate_mode: str = "auto",
    min_jaccard: float = 0.90,
    min_trade_retention: float = 0.90,
    min_fold_action_positive_share: float = 0.75,
    max_full_sl_delta: float = 0.0,
    max_timeout_delta: float = 0.0,
) -> dict[str, Any]:
    summary = _read_csv(priority_dir / "head_priority_learning_replay_summary.csv")
    if summary.empty:
        summary = _read_csv(priority_dir / "head_priority_replay_summary.csv")
    diagnostics = _read_csv(priority_dir / "head_priority_learning_model_diagnostics.csv")
    selection = _read_csv(priority_dir / "head_priority_config_selection.csv")
    overlap = _read_csv(priority_dir / "head_priority_learning_accepted_overlap.csv")
    if overlap.empty:
        overlap = _read_csv(priority_dir / "head_priority_accepted_overlap.csv")
    swap = _read_csv(priority_dir / "head_priority_learning_accepted_swap_utility.csv")
    if swap.empty:
        swap = _read_csv(priority_dir / "head_priority_accepted_swap_utility.csv")
    manifest = _read_json(priority_dir / "manifest.json")
    resolved_gate_mode = str(gate_mode or "auto").strip().lower()
    if resolved_gate_mode == "auto":
        resolved_gate_mode = str(
            (manifest.get("params") or {}).get("selection_gate_mode") or "defensive"
        ).strip().lower()
    if resolved_gate_mode not in {"defensive", "opportunity"}:
        raise ValueError(f"unknown priority gate mode: {resolved_gate_mode}")
    if resolved_gate_mode == "opportunity":
        max_full_sl_delta = max(float(max_full_sl_delta), 0.02)
        max_timeout_delta = max(float(max_timeout_delta), 0.01)

    base = _row_by_arm(summary, BASELINE_ARM)
    candidate_arms = [
        str(arm)
        for arm in summary.get("arm", pd.Series(dtype=str)).dropna().astype(str).unique()
        if str(arm) != BASELINE_ARM
    ]

    rows: list[dict[str, Any]] = []
    for arm in candidate_arms:
        cand = _row_by_arm(summary, arm)
        sw = _swap_row(swap, arm)
        ov = _row_by_arm(overlap, arm)
        if ov:
            sw = dict(sw)
            sw["jaccard_vs_baseline"] = ov.get("jaccard_vs_baseline")
        diag = _diagnostic_row(diagnostics, arm)
        gate = _selection_gate_passed(diagnostics, selection, arm)
        reasons = _fail_reasons(
            selection_gate_passed=gate,
            base=base,
            candidate=cand,
            swap=sw,
            diag=diag,
            min_jaccard=float(min_jaccard),
            min_trade_retention=float(min_trade_retention),
            min_fold_action_positive_share=float(min_fold_action_positive_share),
            max_full_sl_delta=float(max_full_sl_delta),
            max_timeout_delta=float(max_timeout_delta),
            gate_mode=resolved_gate_mode,
        )
        net_delta = _num(cand.get("net_pnl"), 0.0) - _num(base.get("net_pnl"), 0.0)
        full_sl_delta = _num(cand.get("full_sl_rate")) - _num(base.get("full_sl_rate"))
        timeout_delta = _num(cand.get("timeout_rate")) - _num(base.get("timeout_rate"))
        row = {
            "arm": arm,
            "selection_gate_passed": gate,
            "candidate_promotable": len(reasons) == 0,
            "fail_reasons": ";".join(reasons),
            "net_pnl_delta": net_delta,
            "full_sl_delta": full_sl_delta,
            "timeout_delta": timeout_delta,
            "trade_count_delta": int(_num(cand.get("trade_count"), 0.0) - _num(base.get("trade_count"), 0.0)),
            "accepted_jaccard": _num(ov.get("jaccard_vs_baseline"), np.nan),
            "entrants": int(_num(sw.get("entrants"), 0.0)),
            "removed": int(_num(sw.get("removed"), 0.0)),
            "entrant_net_pnl": _num(sw.get("entrant_net_pnl"), 0.0),
            "removed_net_pnl": _num(sw.get("removed_net_pnl"), 0.0),
            "net_replacement_pnl": _num(sw.get("net_replacement_pnl"), 0.0),
            "net_action_pnl_delta": _num(sw.get("net_action_pnl_delta"), 0.0),
            "defensive_success": _num(sw.get("defensive_success"), 0.0),
            "fold_action_positive_delta_share": _num(diag.get("fold_action_positive_delta_share"), np.nan),
            "fold_mean_action_utility_delta": _num(diag.get("fold_mean_action_utility_delta"), np.nan),
            "selection_objective": _num(diag.get("selection_objective"), np.nan),
        }
        rows.append(row)

    audit = pd.DataFrame(rows)
    passing = audit.loc[audit.get("candidate_promotable", pd.Series(dtype=bool)).astype(bool)] if not audit.empty else pd.DataFrame()
    production_blockers = _production_blockers(manifest, window_count=1)
    production_passing_count = 0 if production_blockers else int(len(passing))
    best_raw = (
        audit.sort_values(["candidate_promotable", "net_pnl_delta"], ascending=[False, False])
        .iloc[0]
        .to_dict()
        if not audit.empty
        else {}
    )
    return {
        "audit_version": "market_state_head_priority_promotion_gate_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "priority_dir": str(priority_dir),
        "manifest_generated_by": manifest.get("generated_by"),
        "baseline_arm": BASELINE_ARM,
        "gate_mode": resolved_gate_mode,
        "candidate_count": int(len(audit)),
        "passing_candidate_count": int(len(passing)),
        "single_window_replay_gate_passed": bool(not passing.empty),
        "production_passing_candidate_count": int(production_passing_count),
        "production_blockers": production_blockers,
        "priority_should_remain_shadow": bool(production_blockers or passing.empty),
        "best_raw_candidate": best_raw,
        "gate_config": {
            "min_jaccard": float(min_jaccard),
            "min_trade_retention": float(min_trade_retention),
            "min_fold_action_positive_share": float(min_fold_action_positive_share),
            "max_full_sl_delta": float(max_full_sl_delta),
            "max_timeout_delta": float(max_timeout_delta),
        },
        "candidates": audit.to_dict("records"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    best = report.get("best_raw_candidate") or {}
    lines = [
        "# Market-State Head-Priority Promotion Gate Audit",
        "",
        f"Priority dir: `{report['priority_dir']}`",
        f"Gate mode: `{report.get('gate_mode')}`",
        f"Priority should remain shadow: `{report['priority_should_remain_shadow']}`",
        f"Passing candidate count: `{report['passing_candidate_count']}`",
        f"Single-window replay gate passed: `{report.get('single_window_replay_gate_passed')}`",
        f"Production passing candidate count: `{report.get('production_passing_candidate_count')}`",
        "",
        "## Best Raw Candidate",
        "",
        f"- Arm: `{best.get('arm')}`",
        f"- Promotable: `{best.get('candidate_promotable')}`",
        f"- Fail reasons: `{best.get('fail_reasons')}`",
        f"- Net PnL delta: `{_num(best.get('net_pnl_delta')):.6f}`",
        f"- Full-SL delta: `{_num(best.get('full_sl_delta')):.6f}`",
        f"- Timeout delta: `{_num(best.get('timeout_delta')):.6f}`",
        f"- Accepted Jaccard: `{_num(best.get('accepted_jaccard')):.6f}`",
        f"- Entrants / removed: `{int(_num(best.get('entrants'), 0.0))}` / `{int(_num(best.get('removed'), 0.0))}`",
        f"- Net replacement PnL: `{_num(best.get('net_replacement_pnl')):.6f}`",
        f"- Net action PnL delta: `{_num(best.get('net_action_pnl_delta')):.6f}`",
        f"- Defensive success: `{_num(best.get('defensive_success')):.6f}`",
        "",
        "## Production Blockers",
        "",
    ]
    blockers = report.get("production_blockers") or []
    lines.extend([f"- `{item}`" for item in blockers] if blockers else ["- none"])
    lines.extend(
        [
            "",
            "## Candidates",
            "",
        ]
    )
    frame = pd.DataFrame(report.get("candidates") or [])
    if frame.empty:
        lines.append("_No candidate rows._")
    else:
        cols = [
            "arm",
            "candidate_promotable",
            "fail_reasons",
            "net_pnl_delta",
            "full_sl_delta",
            "timeout_delta",
            "accepted_jaccard",
            "entrants",
            "removed",
            "net_replacement_pnl",
            "net_action_pnl_delta",
            "fold_action_positive_delta_share",
        ]
        lines.append(frame[[c for c in cols if c in frame.columns]].to_markdown(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-dir", type=Path, default=DEFAULT_PRIORITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gate-mode", choices=["auto", "defensive", "opportunity"], default="auto")
    parser.add_argument("--min-jaccard", type=float, default=0.90)
    parser.add_argument("--min-trade-retention", type=float, default=0.90)
    parser.add_argument("--min-fold-action-positive-share", type=float, default=0.75)
    parser.add_argument("--max-full-sl-delta", type=float, default=0.0)
    parser.add_argument("--max-timeout-delta", type=float, default=0.0)
    args = parser.parse_args()

    report = audit_priority_gates(
        args.priority_dir,
        gate_mode=str(args.gate_mode),
        min_jaccard=float(args.min_jaccard),
        min_trade_retention=float(args.min_trade_retention),
        min_fold_action_positive_share=float(args.min_fold_action_positive_share),
        max_full_sl_delta=float(args.max_full_sl_delta),
        max_timeout_delta=float(args.max_timeout_delta),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "market_state_head_priority_promotion_gate_audit.json").write_text(
        json.dumps(_json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_head_priority_promotion_gate_audit.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    pd.DataFrame(report.get("candidates") or []).to_csv(
        args.output_dir / "market_state_head_priority_promotion_gate_candidates.csv",
        index=False,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": str(args.output_dir),
                    "priority_should_remain_shadow": report["priority_should_remain_shadow"],
                    "passing_candidate_count": report["passing_candidate_count"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
