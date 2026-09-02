#!/usr/bin/env python3
"""Build the immutable report for the rich-policy stop-cap comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("control", "cap5", "cap4", "cap3", "cap2")


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    def render(value: object) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.4f}"
        return str(value).replace("|", "\\|")
    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    rows.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    summary_rows: list[dict[str, object]] = []
    monthly: list[pd.DataFrame] = []
    exits: list[pd.DataFrame] = []
    entry: list[pd.DataFrame] = []
    development: list[dict[str, object]] = []
    for arm in ARMS:
        replay_root = ROOT / f"data_perp/artifacts/strict_r3_p8u_15m_entry_policycap_portfolio_20260830_{args.version}_{arm}"
        trainer_root = ROOT / f"data_perp/artifacts/strict_r3_p8u_15m_entry_policycap_retrain_20260830_{args.version}_{arm}"
        row = pd.read_parquet(replay_root / "summary.parquet").iloc[0].to_dict()
        row["arm"] = arm
        candidates = pd.read_parquet(replay_root / "candidates.parquet")
        accepted = pd.read_parquet(replay_root / "accepted.parquet")
        selected = candidates.iloc[pd.to_numeric(accepted.candidate_index, errors="raise").astype(int).to_numpy()].copy()
        net_bps = pd.to_numeric(selected.net_return, errors="raise") * 10_000.0
        row["total_policy_net_bps"] = float(net_bps.sum())
        row["policy_net_bps_per_trade"] = float(net_bps.mean())
        row["policy_net_win_rate"] = float((net_bps > 0.0).mean())
        summary_rows.append(row)
        table = pd.read_parquet(replay_root / "monthly.parquet")
        table["arm"] = arm
        monthly.append(table)
        exit_table = selected.groupby("simple_policy_exit_reason", as_index=False).agg(
            trades=("candidate_id", "size"),
            net_bps_per_trade=("net_return", lambda value: float(value.mean() * 10_000.0)),
            total_net_bps=("net_return", lambda value: float(value.sum() * 10_000.0)),
        )
        exit_table["arm"] = arm
        exits.append(exit_table)
        table = pd.read_parquet(trainer_root / "aggregate_metrics.parquet")
        table["arm"] = arm
        entry.append(table)
        if arm != "control":
            hpo_root = ROOT / f"data_perp/artifacts/strict_r3_p8u_15m_rich_policy_slcap_hpo_20260830_v2_{arm}"
            frozen = json.loads((hpo_root / "frozen_challenger.json").read_text())
            trials = pd.read_parquet(hpo_root / "trials.parquet")
            best = trials.loc[trials.trial.ge(0)].sort_values(["objective", "trial"], ascending=[False, True], kind="stable").iloc[0]
            development.append({
                "arm": arm, "hard_max_sl_pct": frozen["params"]["sl_abs_cap_pct"],
                "hpo_objective": best["objective"], "development_trades": best["metric_trades"],
                "development_net_bps_per_trade": best["metric_net_bps_per_trade"],
                "development_worst_month_bps": best["metric_worst_month_net_bps_per_trade"],
            })
    summary = pd.DataFrame(summary_rows)
    control = summary.loc[summary.arm.eq("control")].iloc[0]
    for metric in (
        "entry_selected", "portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps",
        "compounded_return", "max_drawdown", "sortino", "worst_week", "net_pnl",
    ):
        summary[f"delta_vs_control_{metric}"] = summary[metric] - control[metric]
    summary = summary.loc[:, [
        "arm", "entry_selected", "portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps",
        "policy_net_win_rate", "trades_per_day", "compounded_return", "max_drawdown", "sortino", "worst_week",
        "net_pnl", "delta_vs_control_entry_selected", "delta_vs_control_portfolio_accepted",
        "delta_vs_control_policy_net_bps_per_trade", "delta_vs_control_total_policy_net_bps",
        "delta_vs_control_compounded_return", "delta_vs_control_max_drawdown", "delta_vs_control_sortino",
        "delta_vs_control_worst_week",
    ]]
    month = pd.concat(monthly, ignore_index=True)
    entry_table = pd.concat(entry, ignore_index=True)
    exit_table = pd.concat(exits, ignore_index=True)
    dev = pd.DataFrame(development)
    out.mkdir(parents=True, exist_ok=False)
    summary.to_parquet(out / "portfolio_summary.parquet", index=False)
    month.to_parquet(out / "monthly_policy_net_bps.parquet", index=False)
    entry_table.to_parquet(out / "entry_head_summary.parquet", index=False)
    exit_table.to_parquet(out / "exit_causes.parquet", index=False)
    dev.to_parquet(out / "policy_hpo_development.parquet", index=False)
    manifest = {
        "schema": "strict_r3_p8u_15m_policycap_comparison_v1",
        "scope": "offline research only; no live/canonical change",
        "oos_window": "2026-04-01 through 2026-08-31; every held fold has exactly two preceding complete calendar months with labels resolved before the held month",
        "arms": list(ARMS),
        "control": "existing frozen rich policy, maximum SL 5%, entry head retrained under the same strict-OOS protocol",
        "capped_arms": "independently HPO-selected rich parents over Sep-Oct 2024 calibration / Nov-Dec 2024 selection, then policy labels and entry head retrained",
        "entry_head": "fixed LGBM Huber bps model; dual-MC1 >=30 bps; veto_pred_ge_0; demotion-only",
        "continuation": "kept unchanged but excluded from altered-parent comparison because its state target is parent-policy-specific",
        "portfolio": "identical existing P8U global auction contract across every arm; BCF MC1 priority; 100-bps policy cost embedded once",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join([
        "# Strict-R3 P8U rich-policy stop-cap comparison",
        "",
        "Strict OOS window: April-August 2026.  Each entry fold uses exactly the two prior fully resolved calendar months.",
        "",
        "## Portfolio-constrained results",
        "",
        _markdown(summary),
        "",
        "## Development HPO receipts",
        "",
        _markdown(dev),
        "",
        "## Entry-head OOS receipts",
        "",
        _markdown(entry_table),
        "",
        "## Monthly policy net bps per trade",
        "",
        _markdown(month.pivot(index="month", columns="arm", values="net_bps_per_trade").reset_index()),
        "",
        "## Exit-cause mix",
        "",
        _markdown(exit_table.pivot(index="simple_policy_exit_reason", columns="arm", values="trades").fillna(0).reset_index()),
        "",
        "The continuation head is intentionally frozen and not applied across the changed parent policy geometries.  Its state/utility contract would otherwise be cross-policy mismatched.",
    ])
    (out / "REPORT.md").write_text(report + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
