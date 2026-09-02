#!/usr/bin/env python3
"""Restricted temporal fallback selector for contextual TP/SL candidates.

The selector uses existing monthly temporal-holdout metrics only. It tests
whether simple prior-period gates can improve on a fixed robust benchmark
without searching the full contextual TP/SL grid.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATES = {
    "static": "long_bars:S_long_dist:S_short_asset:S_short_bollinger:S",
    "wf_recent": "long_bars:S_long_dist:R_short_asset:R_short_bollinger:J",
    "best_net": "long_bars:S_long_dist:R_short_asset:R_short_bollinger:R",
    "best_balanced": "long_bars:J_long_dist:R_short_asset:I_short_bollinger:R",
}


@dataclass(frozen=True)
class RuleConfig:
    default_label: str
    score_net_weight: float
    score_week_q20_weight: float
    score_drawdown_weight: float
    min_delta_net_pnl: float
    min_delta_week_q20: float
    min_positive_week_share: float
    min_delta_max_drawdown: float


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[cols].head(max_rows).copy() if max_rows else frame[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def _load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["combo_id"].isin(CANDIDATES.values())].copy()
    reverse = {combo: label for label, combo in CANDIDATES.items()}
    frame["candidate_label"] = frame["combo_id"].map(reverse)
    return frame


def _score_train(train_row: pd.Series, cfg: RuleConfig) -> float:
    return float(
        cfg.score_net_weight * float(train_row["delta_net_pnl"])
        + cfg.score_week_q20_weight * float(train_row["delta_week_q20_pnl"])
        + cfg.score_drawdown_weight * float(train_row["delta_max_drawdown_pnl"])
    )


def _passes(train_row: pd.Series, cfg: RuleConfig) -> bool:
    return bool(
        float(train_row["delta_net_pnl"]) >= cfg.min_delta_net_pnl
        and float(train_row["delta_week_q20_pnl"]) >= cfg.min_delta_week_q20
        and float(train_row["positive_week_delta_share"]) >= cfg.min_positive_week_share
        and float(train_row["delta_max_drawdown_pnl"]) >= cfg.min_delta_max_drawdown
    )


def _configs() -> list[RuleConfig]:
    configs: list[RuleConfig] = []
    for default_label in ("wf_recent", "static"):
        for score_week_q20_weight in (0.0, 0.25, 0.50, 1.00):
            for score_drawdown_weight in (0.0, 0.25, 0.50):
                for min_delta_net_pnl in (0.0, 10_000.0, 25_000.0):
                    for min_delta_week_q20 in (-500.0, 0.0, 250.0, 1_000.0):
                        for min_positive_week_share in (0.60, 0.75, 0.85):
                            for min_delta_max_drawdown in (-1e18, 0.0):
                                configs.append(
                                    RuleConfig(
                                        default_label=default_label,
                                        score_net_weight=1.0,
                                        score_week_q20_weight=score_week_q20_weight,
                                        score_drawdown_weight=score_drawdown_weight,
                                        min_delta_net_pnl=min_delta_net_pnl,
                                        min_delta_week_q20=min_delta_week_q20,
                                        min_positive_week_share=min_positive_week_share,
                                        min_delta_max_drawdown=min_delta_max_drawdown,
                                    )
                                )
    return configs


def _evaluate(metrics: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, group in metrics.groupby("split", sort=False):
        train = group[group["period_role"].eq("train")].copy()
        holdout = group[group["period_role"].eq("holdout")].copy()
        default_combo = CANDIDATES[cfg.default_label]
        default_holdout = holdout[holdout["combo_id"].eq(default_combo)].iloc[0]
        best_label = cfg.default_label
        best_score = -np.inf
        candidates = train[train["candidate_label"].ne("static")].copy()
        for _, train_row in candidates.iterrows():
            if not _passes(train_row, cfg):
                continue
            score = _score_train(train_row, cfg)
            if score > best_score:
                best_score = score
                best_label = str(train_row["candidate_label"])
        selected_combo = CANDIDATES[best_label]
        selected_holdout = holdout[holdout["combo_id"].eq(selected_combo)].iloc[0]
        static_holdout = holdout[holdout["candidate_label"].eq("static")].iloc[0]
        wf_holdout = holdout[holdout["candidate_label"].eq("wf_recent")].iloc[0]
        rows.append(
            {
                "split": split,
                "holdout_start": selected_holdout.get("holdout_start", ""),
                "selected_label": best_label,
                "selected_combo_id": selected_combo,
                "default_label": cfg.default_label,
                "net_pnl": float(selected_holdout["net_pnl"]),
                "objective": float(selected_holdout["objective"]),
                "weekly_q10_pnl": float(selected_holdout["weekly_q10_pnl"]),
                "weekly_min_pnl": float(selected_holdout["weekly_min_pnl"]),
                "daily_q20_pnl": float(selected_holdout["daily_q20_pnl"]),
                "max_drawdown_pnl": float(selected_holdout["max_drawdown_pnl"]),
                "hit_rate": float(selected_holdout["hit_rate"]),
                "delta_vs_static_net_pnl": float(selected_holdout["net_pnl"] - static_holdout["net_pnl"]),
                "delta_vs_wf_recent_net_pnl": float(selected_holdout["net_pnl"] - wf_holdout["net_pnl"]),
                "delta_vs_static_objective": float(selected_holdout["objective"] - static_holdout["objective"]),
                "delta_vs_wf_recent_objective": float(selected_holdout["objective"] - wf_holdout["objective"]),
            }
        )
    return pd.DataFrame(rows)


def _summary(rows: pd.DataFrame, cfg: RuleConfig, run_id: int) -> dict[str, object]:
    return {
        "run_id": run_id,
        **cfg.__dict__,
        "splits": int(len(rows)),
        "sum_net_pnl": float(rows["net_pnl"].sum()),
        "mean_objective": float(rows["objective"].mean()),
        "mean_weekly_q10_pnl": float(rows["weekly_q10_pnl"].mean()),
        "mean_max_drawdown_pnl": float(rows["max_drawdown_pnl"].mean()),
        "mean_hit_rate": float(rows["hit_rate"].mean()),
        "sum_delta_vs_static_net_pnl": float(rows["delta_vs_static_net_pnl"].sum()),
        "sum_delta_vs_wf_recent_net_pnl": float(rows["delta_vs_wf_recent_net_pnl"].sum()),
        "sum_delta_vs_static_objective": float(rows["delta_vs_static_objective"].sum()),
        "sum_delta_vs_wf_recent_objective": float(rows["delta_vs_wf_recent_objective"].sum()),
        "worst_split_delta_vs_wf_recent_net_pnl": float(rows["delta_vs_wf_recent_net_pnl"].min()),
        "positive_vs_wf_recent_splits": int((rows["delta_vs_wf_recent_net_pnl"] > 0.0).sum()),
        "selected_counts": json.dumps(rows["selected_label"].value_counts().to_dict(), sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "data_perp/reports/contextual_tp_sl_temporal_holdout_monthly_tailgate_with_perf_q35w07_q20w03_20260701/temporal_holdout_all_combo_metrics.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_candidate_fallback_20260701"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics(args.metrics)
    summaries: list[dict[str, object]] = []
    runs: list[pd.DataFrame] = []
    for idx, cfg in enumerate(_configs()):
        rows = _evaluate(metrics, cfg)
        rows["run_id"] = idx
        summaries.append(_summary(rows, cfg, idx))
        runs.append(rows)
    summary = pd.DataFrame(summaries).sort_values(
        ["sum_delta_vs_wf_recent_net_pnl", "sum_delta_vs_wf_recent_objective", "worst_split_delta_vs_wf_recent_net_pnl"],
        ascending=[False, False, False],
    )
    all_runs = pd.concat(runs, ignore_index=True)
    best_id = int(summary.iloc[0]["run_id"])
    best_rows = all_runs[all_runs["run_id"].eq(best_id)].copy()
    summary.to_csv(args.output_dir / "candidate_fallback_grid.csv", index=False)
    best_rows.to_csv(args.output_dir / "candidate_fallback_best_holdouts.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"metrics": str(args.metrics), "best_run_id": best_id, "grid_runs": int(len(summary))}, indent=2) + "\n"
    )
    best = summary.iloc[0]
    lines = [
        "# Contextual TP/SL Candidate Fallback",
        "",
        "Restricted monthly temporal-holdout ablation over `static`, `wf_recent`, `best_net`, and `best_balanced`. This does not rerun portfolio replay.",
        "",
        "## Best Configuration",
        "",
        _fmt_table(
            pd.DataFrame([best]),
            [
                "run_id",
                "default_label",
                "score_week_q20_weight",
                "score_drawdown_weight",
                "min_delta_net_pnl",
                "min_delta_week_q20",
                "min_positive_week_share",
                "min_delta_max_drawdown",
                "sum_delta_vs_static_net_pnl",
                "sum_delta_vs_wf_recent_net_pnl",
                "sum_delta_vs_wf_recent_objective",
                "worst_split_delta_vs_wf_recent_net_pnl",
                "positive_vs_wf_recent_splits",
                "selected_counts",
            ],
        ),
        "",
        "## Best Holdouts",
        "",
        _fmt_table(
            best_rows,
            [
                "split",
                "selected_label",
                "net_pnl",
                "objective",
                "weekly_q10_pnl",
                "weekly_min_pnl",
                "max_drawdown_pnl",
                "delta_vs_static_net_pnl",
                "delta_vs_wf_recent_net_pnl",
                "delta_vs_wf_recent_objective",
            ],
        ),
        "",
        "## Top Grid Runs",
        "",
        _fmt_table(
            summary,
            [
                "run_id",
                "default_label",
                "score_week_q20_weight",
                "score_drawdown_weight",
                "min_delta_net_pnl",
                "min_delta_week_q20",
                "min_positive_week_share",
                "sum_delta_vs_wf_recent_net_pnl",
                "sum_delta_vs_wf_recent_objective",
                "worst_split_delta_vs_wf_recent_net_pnl",
                "positive_vs_wf_recent_splits",
                "selected_counts",
            ],
            max_rows=20,
        ),
        "",
        "## Readout",
        "",
        "- A positive result versus static is insufficient; this branch must beat fixed `wf_recent` to matter.",
        "- If the best run cannot beat `wf_recent`, the robust action is to keep `wf_recent` as the benchmark and look for a separate full-SL/timeout guard.",
    ]
    report = args.output_dir / "candidate_fallback_report.md"
    report.write_text("\n".join(lines) + "\n")
    print(report)


if __name__ == "__main__":
    main()
