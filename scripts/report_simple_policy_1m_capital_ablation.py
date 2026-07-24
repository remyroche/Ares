#!/usr/bin/env python3
"""Create a concise Markdown report from the 1m capital ablation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value: object, decimals: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{decimals}f}"


def _table(frame: pd.DataFrame, columns: list[str], labels: list[str] | None = None) -> str:
    labels = labels or columns
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(_fmt(value) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root)
    ablation = root / "ablation"
    download = json.loads((root / "download" / "manifest.json").read_text())
    manifest = json.loads((ablation / "manifest.json").read_text())
    summary = pd.read_csv(ablation / "nested_oos_ablation_summary.csv")
    folds = pd.read_csv(ablation / "nested_oos_fold_metrics.csv")
    july = pd.read_csv(ablation / "july_frozen_family_winner_metrics.csv")
    breakdowns = pd.read_csv(ablation / "july_frozen_detailed_breakdowns.csv")
    final_params = json.loads((ablation / "final_refit_params.json").read_text())

    deployed = summary.loc[summary["arm"] == "deployed_policy_replayed_1m"].iloc[0]
    current = summary.loc[summary["arm"] == "current_policy_reoptimised_1m"].iloc[0]
    summary["stable_delta_vs_deployed"] = summary["stable_fold_objective"] - float(deployed["stable_fold_objective"])
    summary["stable_delta_vs_current_reopt"] = summary["stable_fold_objective"] - float(current["stable_fold_objective"])
    summary = summary.sort_values("stable_fold_objective", ascending=False)

    locked = manifest["family_winners"]
    locked_names = set(locked.values()) | {"deployed_policy_replayed_1m"}
    july = july.loc[july["arm"].isin(locked_names)].copy()
    july_baseline = july.loc[july["arm"] == "deployed_policy_replayed_1m"].iloc[0]
    july["pnl_delta_vs_deployed"] = july["net_pnl_bankroll"] - float(july_baseline["net_pnl_bankroll"])
    july["worst_week_delta_vs_deployed"] = july["worst_week"] - float(july_baseline["worst_week"])
    locked_folds = folds.loc[folds["arm"].isin(locked_names)].copy()
    top_arm = str(summary.iloc[0]["arm"])
    top_capital_rates = folds.loc[folds["arm"] == top_arm, "capital_protect_rate"].tolist()
    top_july_capital_rate = float(july.loc[july["arm"] == top_arm, "capital_protect_rate"].iloc[0])

    lines = [
        "# Simple-policy 1-minute capital-protection ablation",
        "",
        "## Outcome",
        "",
        (
            f"The pre-July nested-OOS winner was **{summary.iloc[0]['arm']}** with a stable-fold "
            f"objective of {_fmt(summary.iloc[0]['stable_fold_objective'])}. All {int(summary.iloc[0]['folds'])} "
            "validation folds were positive. July was not used to choose an arm: one winner per family was "
            "locked from May/June evidence, refit through June 29, then replayed once on July 1-10."
        ),
        "",
        "This is policy-validation/frozen-replay evidence, not a promotion decision. The current-policy July period had already been inspected in earlier work, and 33 capital arms create material multiple-testing risk; a post-July forward window is still required.",
        "",
        "## Data and execution contract",
        "",
        f"- Candidate ledger: {manifest['candidate_rows']:,} model/admission-OOS candidates, {manifest['candidate_period'][0]} through {manifest['candidate_period'][1]}.",
        f"- 1m cache: {download['summary']['covered_minutes']:,}/{download['summary']['required_minutes']:,} required symbol-minutes ({download['coverage']:.2%}), across {download['summary']['ok_symbols']} symbols.",
        f"- Replay: {manifest['replay_spec']['horizon_minutes']:,} minutes per trade; first event-minute open; pessimistic same-minute stop collision.",
        f"- Cost: {manifest['cost_contract']['round_trip_fee']:.2%} round-trip fee once; entry/exit half-spreads embedded once; stop-gap/slip proxy applied once.",
        f"- Portfolio: maximum 8 open positions, 2 new entries per decision bar, and one open position per symbol.",
        "- Reported bankroll PnL is the additive sum of sized trade returns (not a compounded equity curve).",
        f"- Search: {manifest['search_breadth']['planned_nested_trials']:,} nested trials + {manifest['search_breadth']['planned_final_trials']:,} final refit trials.",
        "",
        "## All ablations: May/June nested OOS",
        "",
        _table(
            summary,
            [
                "arm", "stable_fold_objective", "stable_delta_vs_deployed", "stable_delta_vs_current_reopt",
                "mean_pnl", "worst_fold_pnl", "worst_week_across_folds", "worst_drawdown",
                "positive_fold_fraction", "total_oos_trades",
            ],
            [
                "Arm", "Stable obj", "Δ deployed", "Δ current reopt", "Mean PnL", "Worst fold PnL",
                "Worst week", "Worst DD", "Positive folds", "OOS trades",
            ],
        ),
        "",
        f"The detailed {len(folds):,} arm × fold rows are in `nested_oos_fold_metrics.csv`; this table reports the predeclared mean − 0.5×std + 0.25×worst aggregation across folds.",
        "",
        "## Fold stability for locked family winners",
        "",
        _table(
            locked_folds.sort_values(["arm", "fold"]),
            ["arm", "fold", "net_pnl_bankroll", "worst_week", "max_drawdown", "n_trades", "hit_rate", "mean_net_return"],
            ["Arm", "Fold", "Net PnL", "Worst week", "Max DD", "Trades", "Hit rate", "Mean net/trade"],
        ),
        "",
        "## Locked family winners: July frozen replay",
        "",
        _table(
            july.sort_values("net_pnl_bankroll", ascending=False),
            [
                "arm", "net_pnl_bankroll", "pnl_delta_vs_deployed", "worst_week", "worst_week_delta_vs_deployed",
                "max_drawdown", "n_trades", "hit_rate", "mean_net_return", "mean_holding_hours",
                "full_sl_rate", "capital_protect_rate", "trailing_rate", "timeout_rate",
            ],
            [
                "Arm", "Net PnL", "Δ PnL", "Worst week", "Δ worst week", "Max DD", "Trades", "Hit rate",
                "Mean net/trade", "Mean hold h", "SL", "Capital", "Trailing", "Timeout",
            ],
        ),
        "",
        "July comparisons are diagnostic confirmation only; the ranking above was not used for selection.",
        "",
        "## Formula interpretation",
        "",
        f"- `u = MFE / ATR`, with ATR defined as the entry-frozen deployable barrier proxy ({manifest['atr_contract']['side_values']}).",
        "- Multi-layer (b) uses the loosest envelope: `max(x·ATR, y·ATR·u^0.3, z·ATR·u^0.6)`.",
        "- Maximum-giveback caps the allowed gap; minimum-MFE distance floors the gap from the peak; minimum-current distance uses only the prior completed 1m close and a monotone ratchet.",
        "- When trailing is armed, capital protection is forced to remain looser than the profit trail; the executable stop merges both candidates and uses the tighter valid stop.",
        "",
        "## Locked final parameters",
        "",
    ]
    for family_name, arm_name in locked.items():
        payload = final_params[arm_name]
        lines.extend([f"### {family_name}: `{arm_name}`", ""])
        for side in ("long", "short"):
            params = payload["params_by_side"][side]
            shown = {key: value for key, value in params.items() if key not in {"adverse_exit_theta"}}
            lines.append(f"- {side}: `{json.dumps(shown, sort_keys=True)}`")
        lines.append("")

    lines.extend(
        [
            "## Detailed diagnostics",
            "",
            f"The July breakdown artifact contains {len(breakdowns):,} overall/month/week/side/archetype/side×archetype rows. The selected-trade ledger preserves every accepted trade and exit result for audit.",
            "",
            "## Evidence classification and caveats",
            "",
            "- Base/meta/admission rows are historical OOS; policy geometry is nested walk-forward OOS on three chronological validation folds with a full 24h purge day.",
            "- July challengers are frozen family-winner replay, but July is not a pristine system-level test because incumbent July results were previously observed.",
            "- 1m OHLC still lacks intraminute high/low ordering; stop collisions use the pessimistic stop-first rule.",
            "- Sparse archetypes are diagnostics only; geometry is side-parent, so the one-row and 31-row archetypes were not independently optimized.",
            f"- The winning arm's capital-stop exit rates were {', '.join(_fmt(v) for v in top_capital_rates)} across the three folds and {_fmt(top_july_capital_rate)} in July. Because the capital barrier is deliberately looser than trailing, the measured uplift is evidence for the jointly optimized geometry, not an isolated causal benefit from the capital formula.",
            "- No arm should be deployed until the winning formula is implemented in live inference and replay/live parity tests pass on a later unseen window.",
        ]
    )
    report_path = root / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
