#!/usr/bin/env python3
"""Recompute direct accepted-frontier controller selection from a walk-forward run.

This is a replay-free audit step.  It reads an existing market-state
threshold-controller walk-forward directory and reruns the conservative
selection logic while forcing suppression gates to come from baseline-accepted
candidate suppression, not broad candidate suppression or post-selection replay
action removals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

from scripts import run_market_state_threshold_controller_walkforward as walkforward  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "data_perp/reports/market_state_threshold_controller_walkforward_globalrank_no_backfill_20260627_v1"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_controller_accepted_frontier_reselection")


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


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _policy_from_source(source_dir: Path) -> dict[str, Any]:
    selected = _read_json(source_dir / "walkforward_selected_controller_candidate.json")
    policy = dict(selected.get("selection_policy") or {})
    if not policy:
        config = _read_json(source_dir / "strategy_threshold_controller_config.json")
        policy = dict((config.get("selection") or {}).get("selection_policy") or {})
    return policy


def _reselection_policy(source_dir: Path) -> dict[str, Any]:
    policy = _policy_from_source(source_dir)
    return {
        "min_positive_delta_share": float(policy.get("min_positive_delta_share", 0.5)),
        "min_median_delta_net_pnl": float(policy.get("min_median_delta_net_pnl", 0.0)),
        "min_q25_delta_net_pnl": float(policy.get("min_q25_delta_net_pnl", 0.0)),
        "min_defensive_success": float(policy.get("min_defensive_success", 0.0)),
        "min_positive_suppression_share": float(
            policy.get("min_positive_suppression_share", 0.5)
        ),
        "max_mean_state_ood_share": float(policy.get("max_mean_state_ood_share", 0.1)),
        "min_median_delta_max_drawdown": float(
            policy.get("min_median_delta_max_drawdown", 0.0)
        ),
        "min_median_delta_worst_24h": float(
            policy.get("min_median_delta_worst_24h", 0.0)
        ),
        "max_median_delta_full_sl_rate": float(
            policy.get("max_median_delta_full_sl_rate", 0.0)
        ),
        "min_median_trade_retention_share": float(
            policy.get("min_median_trade_retention_share", 0.8)
        ),
        "median_delta_tie_abs_tol": float(policy.get("median_delta_tie_abs_tol", 1.0)),
        "median_delta_tie_rel_tol": float(policy.get("median_delta_tie_rel_tol", 0.05)),
        "require_post_selection_confirmation": bool(
            policy.get("require_post_selection_confirmation", True)
        ),
        "select_no_backfill_overlay_only": True,
    }


def _arm_summary(selection: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "arm",
        "median_delta_net_pnl",
        "q25_delta_net_pnl",
        "positive_delta_share",
        "realized_defensive_success",
        "positive_suppression_fold_share",
        "candidate_realized_defensive_success",
        "action_defensive_success",
        "passed_selection_gates",
        "selection_fail_reasons",
    ]
    if selection.empty:
        return []
    available = [col for col in columns if col in selection.columns]
    return _json_safe(selection[available].to_dict(orient="records"))


def reselect_accepted_frontier(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    aggregate = _read_csv(source_dir / "walkforward_aggregate_delta.csv")
    suppression_aggregate = _read_csv(
        source_dir / "walkforward_threshold_candidate_suppression_aggregate.csv",
        required=False,
    )
    controller_diagnostics = _read_csv(
        source_dir / "walkforward_controller_state_diagnostics.csv",
        required=False,
    )
    action_utility_aggregate = _read_csv(
        source_dir / "walkforward_threshold_action_utility_aggregate.csv",
        required=False,
    )
    baseline_accepted_suppression_aggregate = _read_csv(
        source_dir / "walkforward_threshold_baseline_accepted_suppression_aggregate.csv",
        required=True,
    )
    policy = _reselection_policy(source_dir)
    selection, payload = walkforward._select_controller_candidate(
        aggregate,
        suppression_aggregate,
        controller_diagnostics,
        action_utility_aggregate,
        baseline_accepted_suppression_aggregate,
        min_positive_delta_share=policy["min_positive_delta_share"],
        min_median_delta_net_pnl=policy["min_median_delta_net_pnl"],
        min_q25_delta_net_pnl=policy["min_q25_delta_net_pnl"],
        min_defensive_success=policy["min_defensive_success"],
        min_positive_suppression_share=policy["min_positive_suppression_share"],
        max_mean_state_ood_share=policy["max_mean_state_ood_share"],
        min_median_delta_max_drawdown=policy["min_median_delta_max_drawdown"],
        min_median_delta_worst_24h=policy["min_median_delta_worst_24h"],
        max_median_delta_full_sl_rate=policy["max_median_delta_full_sl_rate"],
        min_median_trade_retention_share=policy["min_median_trade_retention_share"],
        median_delta_tie_abs_tol=policy["median_delta_tie_abs_tol"],
        median_delta_tie_rel_tol=policy["median_delta_tie_rel_tol"],
        require_post_selection_confirmation=policy["require_post_selection_confirmation"],
        select_no_backfill_overlay_only=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / "accepted_frontier_controller_candidate_selection.csv", index=False)
    summary_columns = [
        "arm",
        "median_delta_net_pnl",
        "q25_delta_net_pnl",
        "positive_delta_share",
        "suppressed_candidates",
        "realized_defensive_success",
        "positive_suppression_fold_share",
        "action_defensive_success",
        "passed_selection_gates",
        "selection_fail_reasons",
    ]
    summary = selection[[col for col in summary_columns if col in selection.columns]].copy()
    summary.to_csv(output_dir / "accepted_frontier_selection_summary.csv", index=False)
    payload = {
        **payload,
        "generated_by": "reselect_market_state_accepted_frontier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "selection_contract": (
            "direct baseline-accepted suppression only; action replay removals "
            "and broad candidate suppression cannot satisfy suppression gates"
        ),
        "arm_summary": _arm_summary(selection),
    }
    (output_dir / "accepted_frontier_selection_payload.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    report = _render_report(source_dir, selection, summary, payload)
    (output_dir / "accepted_frontier_reselection_report.md").write_text(
        report,
        encoding="utf-8",
    )
    return payload


def _render_report(
    source_dir: Path,
    selection: pd.DataFrame,
    summary: pd.DataFrame,
    payload: dict[str, Any],
) -> str:
    policy = payload.get("selection_policy") or {}
    selected = payload.get("selected_arm")
    reason = payload.get("reason")
    lines = [
        "# Accepted-Frontier Controller Reselection",
        "",
        f"Source: `{source_dir}`",
        "",
        f"Selected arm: `{selected}`",
        f"Reason: `{reason}`",
        "",
        (
            "Selection gates on direct baseline-accepted suppression, not broad "
            "candidate suppression or post-selection replay action removals."
        ),
        "",
        "## Selection Policy",
        "",
        "```json",
        json.dumps(_json_safe(policy), indent=2),
        "```",
        "",
        "## Arm Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No candidate arms._",
        "",
        "## Interpretation",
        "",
    ]
    if selected:
        lines.extend(
            [
                "- A controller arm passed the direct accepted-frontier gate.",
                "- Promotion still requires later matured shadow confirmation under the active rank contract.",
            ]
        )
    else:
        lines.extend(
            [
                "- No controller arm passes once the direct accepted-frontier gate is enforced.",
                "- Overlay replay gains are not sufficient if direct threshold suppression is absent or not recurrent.",
                "- The active execution stack should remain static T1; market-state threshold control remains shadow-only.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = reselect_accepted_frontier(args.source_dir, args.output_dir)
    print(json.dumps(_json_safe(payload), indent=2))


if __name__ == "__main__":
    main()
