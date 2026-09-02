#!/usr/bin/env python3
"""Robustness comparison for wf_recent smooth-penalty combo expanding replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_wfrecent_row_guard_walkforward import _fmt_table, _json_safe


def _global_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[frame["period_type"].eq("week")].copy()
    if "head" in out.columns:
        out = out[out["head"].isna()].copy()
    out["week"] = out["week"].astype(str)
    return out.reset_index(drop=True)


def _weekly_delta(baseline: pd.DataFrame, challenger: pd.DataFrame, variant: str) -> pd.DataFrame:
    base = _global_weekly(baseline)
    cur = _global_weekly(challenger)
    merged = base.merge(cur, on=["period_type", "week"], suffixes=("_baseline", "_challenger"))
    return pd.DataFrame(
        {
            "variant": variant,
            "week": merged["week"],
            "baseline_net_pnl": merged["net_pnl_baseline"],
            "challenger_net_pnl": merged["net_pnl_challenger"],
            "delta_net_pnl": merged["net_pnl_challenger"] - merged["net_pnl_baseline"],
            "delta_gross_pnl": merged["gross_pnl_challenger"] - merged["gross_pnl_baseline"],
            "baseline_trades": merged["trades_baseline"],
            "challenger_trades": merged["trades_challenger"],
            "delta_trades": merged["trades_challenger"] - merged["trades_baseline"],
            "baseline_hit_rate": merged["hit_rate_baseline"],
            "challenger_hit_rate": merged["hit_rate_challenger"],
            "delta_hit_rate": merged["hit_rate_challenger"] - merged["hit_rate_baseline"],
            "baseline_full_sl_rate": merged["full_sl_rate_baseline"],
            "challenger_full_sl_rate": merged["full_sl_rate_challenger"],
            "delta_full_sl_rate": merged["full_sl_rate_challenger"] - merged["full_sl_rate_baseline"],
            "baseline_timeout_rate": merged["timeout_rate_baseline"],
            "challenger_timeout_rate": merged["timeout_rate_challenger"],
            "delta_timeout_rate": merged["timeout_rate_challenger"] - merged["timeout_rate_baseline"],
        }
    )


def _week_end(week: pd.Series) -> pd.Series:
    parts = week.astype(str).str.split("/", n=1, expand=True)
    raw = parts[1] if parts.shape[1] > 1 else parts[0]
    return pd.to_datetime(raw, utc=True, errors="coerce")


def _tail_objective(values: np.ndarray, q35_weight: float, q20_weight: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite) + q35_weight * np.quantile(finite, 0.35) + q20_weight * np.quantile(finite, 0.20))


def _bootstrap(values: pd.DataFrame, *, n_boot: int, seed: int, q35_weight: float, q20_weight: float) -> dict[str, float]:
    if values.empty:
        return {
            "boot_sum_q05": np.nan,
            "boot_sum_q50": np.nan,
            "boot_sum_q95": np.nan,
            "boot_prob_sum_positive": np.nan,
            "boot_objective_delta_q05": np.nan,
            "boot_objective_delta_q50": np.nan,
            "boot_objective_delta_q95": np.nan,
            "boot_prob_objective_positive": np.nan,
        }
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    delta = values["delta_net_pnl"].to_numpy(dtype=float)
    base = values["baseline_net_pnl"].to_numpy(dtype=float)
    challenger = values["challenger_net_pnl"].to_numpy(dtype=float)
    sample_sums = delta[idx].sum(axis=1)
    objective_deltas = np.empty(int(n_boot), dtype=float)
    for i, take in enumerate(idx):
        objective_deltas[i] = _tail_objective(challenger[take], q35_weight, q20_weight) - _tail_objective(
            base[take], q35_weight, q20_weight
        )
    return {
        "boot_sum_q05": float(np.quantile(sample_sums, 0.05)),
        "boot_sum_q50": float(np.quantile(sample_sums, 0.50)),
        "boot_sum_q95": float(np.quantile(sample_sums, 0.95)),
        "boot_prob_sum_positive": float(np.mean(sample_sums > 0.0)),
        "boot_objective_delta_q05": float(np.quantile(objective_deltas, 0.05)),
        "boot_objective_delta_q50": float(np.quantile(objective_deltas, 0.50)),
        "boot_objective_delta_q95": float(np.quantile(objective_deltas, 0.95)),
        "boot_prob_objective_positive": float(np.mean(objective_deltas > 0.0)),
    }


def _quantile_row(group: pd.DataFrame, *, n_boot: int, seed: int, q35_weight: float, q20_weight: float) -> dict[str, Any]:
    base = group["baseline_net_pnl"].to_numpy(dtype=float)
    challenger = group["challenger_net_pnl"].to_numpy(dtype=float)
    delta = group["delta_net_pnl"].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "variant": str(group["variant"].iloc[0]),
        "weeks": int(len(group)),
        "sum_delta_net_pnl": float(np.sum(delta)),
        "mean_delta_net_pnl": float(np.mean(delta)),
        "median_delta_net_pnl": float(np.median(delta)),
        "positive_delta_week_share": float(np.mean(delta > 0.0)),
        "worst_delta_week": float(np.min(delta)),
        "best_delta_week": float(np.max(delta)),
        "baseline_q05_week_net_pnl": float(np.quantile(base, 0.05)),
        "challenger_q05_week_net_pnl": float(np.quantile(challenger, 0.05)),
        "delta_q05_week_net_pnl": float(np.quantile(challenger, 0.05) - np.quantile(base, 0.05)),
        "baseline_q10_week_net_pnl": float(np.quantile(base, 0.10)),
        "challenger_q10_week_net_pnl": float(np.quantile(challenger, 0.10)),
        "delta_q10_week_net_pnl": float(np.quantile(challenger, 0.10) - np.quantile(base, 0.10)),
        "baseline_q20_week_net_pnl": float(np.quantile(base, 0.20)),
        "challenger_q20_week_net_pnl": float(np.quantile(challenger, 0.20)),
        "delta_q20_week_net_pnl": float(np.quantile(challenger, 0.20) - np.quantile(base, 0.20)),
        "baseline_q35_week_net_pnl": float(np.quantile(base, 0.35)),
        "challenger_q35_week_net_pnl": float(np.quantile(challenger, 0.35)),
        "delta_q35_week_net_pnl": float(np.quantile(challenger, 0.35) - np.quantile(base, 0.35)),
        "baseline_tail_objective": _tail_objective(base, q35_weight, q20_weight),
        "challenger_tail_objective": _tail_objective(challenger, q35_weight, q20_weight),
    }
    row["delta_tail_objective"] = float(row["challenger_tail_objective"] - row["baseline_tail_objective"])
    row.update(_bootstrap(group, n_boot=n_boot, seed=seed, q35_weight=q35_weight, q20_weight=q20_weight))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_expanding_serious_feb_jun_20260701"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_expanding_robustness_20260701"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument("--first-guard-month", default="")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    first_guard_text = str(args.first_guard_month or manifest.get("first_guard_month") or "")
    first_guard = pd.Timestamp(first_guard_text, tz="UTC") if first_guard_text else None

    baseline = pd.read_csv(args.input_dir / "combo_expanding_baseline_weekly.csv")
    challenger_weekly = pd.read_csv(args.input_dir / "combo_expanding_weekly.csv")
    summary = pd.read_csv(args.input_dir / "combo_expanding_summary.csv")
    deltas = []
    for variant, group in challenger_weekly.groupby("variant", sort=False):
        deltas.append(_weekly_delta(baseline, group, str(variant)))
    weekly_delta = pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame()
    if first_guard is not None and not weekly_delta.empty:
        weekly_delta["week_end"] = _week_end(weekly_delta["week"])
        weekly_delta = weekly_delta[weekly_delta["week_end"].ge(first_guard)].drop(columns=["week_end"]).reset_index(drop=True)
    robustness = pd.DataFrame(
        [
            _quantile_row(
                group,
                n_boot=args.bootstrap_samples,
                seed=int(args.seed),
                q35_weight=float(args.q35_weight),
                q20_weight=float(args.q20_weight),
            )
            for _, group in weekly_delta.groupby("variant", sort=False)
        ]
    )
    merged = robustness.merge(
        summary[
            [
                "variant",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_q35_week_net_pnl",
                "delta_worst_week_net_pnl",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_trade_count",
            ]
        ],
        on="variant",
        how="left",
        suffixes=("_weekly", "_summary"),
    )
    merged = merged.sort_values(
        ["boot_prob_objective_positive", "delta_tail_objective", "sum_delta_net_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    weekly_delta.to_csv(args.output_dir / "combo_expanding_weekly_delta.csv", index=False)
    robustness.to_csv(args.output_dir / "combo_expanding_weekly_robustness.csv", index=False)
    merged.to_csv(args.output_dir / "combo_expanding_robustness_decision.csv", index=False)

    manifest = {
        "generated_by": "compare_wfrecent_combo_expanding_robustness",
        "input_dir": str(args.input_dir),
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
        "first_guard_month": first_guard.isoformat() if first_guard is not None else None,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Combo Expanding Robustness",
        "",
        "Paired weekly robustness analysis for the expanding-reference combo replay. Bootstrap resamples complete weeks with replacement, preserving the baseline/challenger pairing.",
        "",
        f"Guard-period filter: `{first_guard.isoformat() if first_guard is not None else 'not applied'}`.",
        "",
        "## Decision Table",
        "",
        _fmt_table(
            merged,
            [
                "variant",
                "sum_delta_net_pnl",
                "delta_tail_objective",
                "positive_delta_week_share",
                "delta_q05_week_net_pnl",
                "delta_q10_week_net_pnl",
                "delta_q20_week_net_pnl",
                "delta_q35_week_net_pnl_weekly",
                "boot_sum_q05",
                "boot_sum_q50",
                "boot_prob_sum_positive",
                "boot_objective_delta_q05",
                "boot_objective_delta_q50",
                "boot_prob_objective_positive",
                "delta_hit_rate",
                "delta_full_sl_rate",
            ],
        ),
        "",
        "## Weekly Delta Extremes",
        "",
        _fmt_table(
            weekly_delta.sort_values(["variant", "delta_net_pnl"]).groupby("variant").head(3),
            ["variant", "week", "delta_net_pnl", "delta_hit_rate", "delta_full_sl_rate", "delta_trades"],
        ),
    ]
    (args.output_dir / "combo_expanding_robustness_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
