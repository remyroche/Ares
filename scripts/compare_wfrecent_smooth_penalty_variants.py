#!/usr/bin/env python3
"""Compare fixed wf_recent smooth rank-penalty variants from replay artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.validate_wfrecent_row_guard_walkforward import _fmt_table, _json_safe


DEFAULT_VARIANTS = {
    "q90_conservative": Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_rank_penalty_fixed_replay_20260701"),
    "q85_aggressive": Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_rank_penalty_fixed_q85_replay_v3_20260701"),
}


def _load_variant(label: str, path: Path) -> dict[str, pd.DataFrame]:
    return {
        "label": pd.DataFrame({"variant": [label], "path": [str(path)]}),
        "summary": pd.read_csv(path / "fixed_smooth_rank_penalty_summary.csv").assign(variant=label),
        "monthly": pd.read_csv(path / "fixed_smooth_rank_penalty_monthly.csv").assign(variant=label),
        "per_head": pd.read_csv(path / "fixed_smooth_rank_penalty_per_head.csv").assign(variant=label),
        "baseline_weekly": pd.read_csv(path / "fixed_smooth_rank_penalty_baseline_weekly.csv").assign(variant=label),
        "challenger_weekly": pd.read_csv(path / "fixed_smooth_rank_penalty_challenger_weekly.csv").assign(variant=label),
    }


def _global_weekly_delta(variant: str, baseline: pd.DataFrame, challenger: pd.DataFrame) -> pd.DataFrame:
    base = baseline[baseline["period_type"].eq("week")].copy()
    ch = challenger[challenger["period_type"].eq("week")].copy()
    merged = base.merge(ch, on=["period_type", "week"], suffixes=("_baseline", "_challenger"))
    out = pd.DataFrame(
        {
            "variant": variant,
            "week": merged["week"],
            "delta_net_pnl": merged["net_pnl_challenger"] - merged["net_pnl_baseline"],
            "delta_gross_pnl": merged["gross_pnl_challenger"] - merged["gross_pnl_baseline"],
            "delta_trades": merged["trades_challenger"] - merged["trades_baseline"],
            "delta_hit_rate": merged["hit_rate_challenger"] - merged["hit_rate_baseline"],
            "delta_full_sl_rate": merged["full_sl_rate_challenger"] - merged["full_sl_rate_baseline"],
            "delta_timeout_rate": merged["timeout_rate_challenger"] - merged["timeout_rate_baseline"],
            "baseline_net_pnl": merged["net_pnl_baseline"],
            "challenger_net_pnl": merged["net_pnl_challenger"],
        }
    )
    return out


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 5000, seed: int = 42) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "sum": np.nan, "q05": np.nan, "q50": np.nan, "q95": np.nan, "prob_positive_sum": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, finite.size, size=(int(n_boot), finite.size))
    samples = finite[idx].sum(axis=1)
    return {
        "mean": float(np.mean(finite)),
        "sum": float(np.sum(finite)),
        "q05": float(np.quantile(samples, 0.05)),
        "q50": float(np.quantile(samples, 0.50)),
        "q95": float(np.quantile(samples, 0.95)),
        "prob_positive_sum": float(np.mean(samples > 0.0)),
    }


def _variant_score(summary_row: pd.Series, weekly_ci: dict[str, float]) -> float:
    # A conservative scalar for comparing two fixed challengers. PnL dominates,
    # tail and recurrence enter as penalties/rewards.
    return float(
        summary_row["delta_net_pnl"]
        + 0.7 * summary_row["delta_q35_week_net_pnl"]
        + 0.3 * summary_row["delta_worst_week_net_pnl"]
        + 50000.0 * max(0.0, -float(summary_row["delta_full_sl_rate"]))
        + 1000.0 * float(weekly_ci["prob_positive_sum"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_rank_penalty_variant_comparison_20260701"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {label: _load_variant(label, path) for label, path in DEFAULT_VARIANTS.items()}
    summaries = pd.concat([cur["summary"] for cur in loaded.values()], ignore_index=True)
    monthly = pd.concat([cur["monthly"] for cur in loaded.values()], ignore_index=True)
    per_head = pd.concat([cur["per_head"] for cur in loaded.values()], ignore_index=True)
    weekly = pd.concat(
        [
            _global_weekly_delta(label, cur["baseline_weekly"], cur["challenger_weekly"])
            for label, cur in loaded.items()
        ],
        ignore_index=True,
    )
    ci_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for label, group in weekly.groupby("variant"):
        ci = _bootstrap_ci(group["delta_net_pnl"].to_numpy(dtype=float), n_boot=args.bootstrap_samples)
        ci_rows.append({"variant": label, **ci})
        srow = summaries[summaries["variant"].eq(label)].iloc[0]
        score_rows.append(
            {
                "variant": label,
                "decision_score": _variant_score(srow, ci),
                "delta_net_pnl": float(srow["delta_net_pnl"]),
                "delta_objective_week": float(srow["delta_objective_week"]),
                "delta_q35_week_net_pnl": float(srow["delta_q35_week_net_pnl"]),
                "delta_worst_week_net_pnl": float(srow["delta_worst_week_net_pnl"]),
                "delta_hit_rate": float(srow["delta_hit_rate"]),
                "delta_full_sl_rate": float(srow["delta_full_sl_rate"]),
                "delta_timeout_rate": float(srow["delta_timeout_rate"]),
                "weekly_bootstrap_q05_sum": ci["q05"],
                "weekly_bootstrap_q50_sum": ci["q50"],
                "weekly_bootstrap_q95_sum": ci["q95"],
                "weekly_bootstrap_prob_positive_sum": ci["prob_positive_sum"],
            }
        )
    ci_df = pd.DataFrame(ci_rows)
    decision = pd.DataFrame(score_rows).sort_values("decision_score", ascending=False).reset_index(drop=True)

    summaries.to_csv(args.output_dir / "variant_summary.csv", index=False)
    monthly.to_csv(args.output_dir / "variant_monthly.csv", index=False)
    per_head.to_csv(args.output_dir / "variant_per_head.csv", index=False)
    weekly.to_csv(args.output_dir / "variant_weekly_delta.csv", index=False)
    ci_df.to_csv(args.output_dir / "variant_weekly_bootstrap_ci.csv", index=False)
    decision.to_csv(args.output_dir / "variant_decision_table.csv", index=False)

    best = decision.iloc[0]
    lines = [
        "# wf_recent Smooth Penalty Variant Comparison",
        "",
        "Compares fixed q85 and q90 long_dist composite-risk smooth penalties using existing continuous replay artifacts. Costs are inherited from `portfolio_policy_replay.replay_candidates`.",
        "",
        "## Decision Table",
        "",
        _fmt_table(
            decision,
            [
                "variant",
                "decision_score",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_q35_week_net_pnl",
                "delta_worst_week_net_pnl",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "weekly_bootstrap_q05_sum",
                "weekly_bootstrap_prob_positive_sum",
            ],
        ),
        "",
        "## Monthly Deltas",
        "",
        _fmt_table(
            monthly,
            [
                "variant",
                "month",
                "delta_net_pnl",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_worst_week_net_pnl",
            ],
        ),
        "",
        "## Per-Head Deltas",
        "",
        _fmt_table(
            per_head,
            [
                "variant",
                "head",
                "delta_net_pnl",
                "delta_gross_pnl",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
            ],
        ),
        "",
        "## Readout",
        "",
        f"- Best decision score: `{best['variant']}`.",
        "- q85 is the stronger total-PnL/tail variant; q90 remains the more conservative recurrence fallback.",
        "- This is still a proxy execution backtest over an existing candidate universe, not fresh prospective OOS.",
    ]
    (args.output_dir / "variant_comparison_report.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "generated_by": "compare_wfrecent_smooth_penalty_variants",
        "variants": {label: str(path) for label, path in DEFAULT_VARIANTS.items()},
        "bootstrap_samples": int(args.bootstrap_samples),
        "best_variant": str(best["variant"]),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
