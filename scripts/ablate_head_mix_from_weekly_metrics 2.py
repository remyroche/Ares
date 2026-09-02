#!/usr/bin/env python3
"""Proxy head-mix ablation from materialized weekly per-head replay metrics.

This is intentionally lightweight: it uses additive per-head weekly metrics from
an already materialized replay and evaluates static or walk-forward head subsets.
It does not rerun the auction, so treat it as a headroom diagnostic before any
production policy change.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_OBJECTIVE_Q35_WEIGHT = 0.70
DEFAULT_OBJECTIVE_Q20_WEIGHT = 0.30


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _subsets(items: list[str], min_heads: int) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for n in range(min_heads, len(items) + 1):
        out.extend(combinations(items, n))
    return out


def _objective(values: np.ndarray, q35_weight: float, q20_weight: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("-inf")
    return float(np.mean(values) + q35_weight * np.quantile(values, 0.35) + q20_weight * np.quantile(values, 0.20))


def _summarize(values: np.ndarray, q35_weight: float, q20_weight: float) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "weeks": 0,
            "sum_net_pnl": np.nan,
            "avg_week_net_pnl": np.nan,
            "median_week_net_pnl": np.nan,
            "q05_week_net_pnl": np.nan,
            "q10_week_net_pnl": np.nan,
            "q15_week_net_pnl": np.nan,
            "q20_week_net_pnl": np.nan,
            "q25_week_net_pnl": np.nan,
            "q35_week_net_pnl": np.nan,
            "worst_week_net_pnl": np.nan,
            "positive_weeks": 0,
            "objective": np.nan,
        }
    return {
        "weeks": int(values.size),
        "sum_net_pnl": float(np.sum(values)),
        "avg_week_net_pnl": float(np.mean(values)),
        "median_week_net_pnl": float(np.median(values)),
        "q05_week_net_pnl": float(np.quantile(values, 0.05)),
        "q10_week_net_pnl": float(np.quantile(values, 0.10)),
        "q15_week_net_pnl": float(np.quantile(values, 0.15)),
        "q20_week_net_pnl": float(np.quantile(values, 0.20)),
        "q25_week_net_pnl": float(np.quantile(values, 0.25)),
        "q35_week_net_pnl": float(np.quantile(values, 0.35)),
        "worst_week_net_pnl": float(np.min(values)),
        "positive_weeks": int(np.sum(values > 0)),
        "objective": _objective(values, q35_weight, q20_weight),
    }


def _load_weekly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["head"] = frame["head"].fillna("__global__")
    frame = frame[frame["head"] != "__global__"].copy()
    frame["week_start"] = pd.to_datetime(frame["week"].str.split("/").str[0])
    for col in ("net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values(["week_start", "head"]).reset_index(drop=True)


def _variant_weekly(panel: pd.DataFrame, heads: Iterable[str]) -> pd.DataFrame:
    heads = tuple(heads)
    weeks = panel[["week", "week_start"]].drop_duplicates().sort_values("week_start")
    sub = panel[panel["head"].isin(heads)].copy()
    grouped = sub.groupby(["week", "week_start"], sort=True)
    out = grouped[["net_pnl", "gross_pnl", "trades"]].sum(min_count=1).reset_index()
    out = weeks.merge(out, on=["week", "week_start"], how="left")
    for col in ("net_pnl", "gross_pnl", "trades"):
        out[col] = out[col].fillna(0.0)
    for rate in ("hit_rate", "full_sl_rate", "timeout_rate"):
        weighted = sub.assign(_weighted=sub[rate] * sub["trades"]).groupby(["week", "week_start"], sort=True)
        denom = grouped["trades"].sum(min_count=1).replace(0, np.nan)
        rate_frame = (weighted["_weighted"].sum(min_count=1) / denom).reset_index(name=rate)
        out = out.merge(rate_frame, on=["week", "week_start"], how="left")
    out["enabled_heads"] = ",".join(heads)
    out["disabled_heads"] = ""
    return out


def _static_table(panel: pd.DataFrame, heads: list[str], min_heads: int, q35_weight: float, q20_weight: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    weekly_frames = []
    for subset in _subsets(heads, min_heads):
        weekly = _variant_weekly(panel, subset)
        disabled = sorted(set(heads) - set(subset))
        weekly["disabled_heads"] = ",".join(disabled)
        vals = weekly["net_pnl"].to_numpy()
        row = {
            "enabled_heads": ",".join(subset),
            "disabled_heads": ",".join(disabled),
            **_summarize(vals, q35_weight, q20_weight),
            "trades": int(weekly["trades"].sum()),
            "hit_rate": float((weekly["hit_rate"] * weekly["trades"]).sum() / weekly["trades"].sum()) if weekly["trades"].sum() else np.nan,
            "full_sl_rate": float((weekly["full_sl_rate"] * weekly["trades"]).sum() / weekly["trades"].sum()) if weekly["trades"].sum() else np.nan,
            "timeout_rate": float((weekly["timeout_rate"] * weekly["trades"]).sum() / weekly["trades"].sum()) if weekly["trades"].sum() else np.nan,
        }
        rows.append(row)
        weekly["policy"] = "static:" + row["enabled_heads"]
        weekly_frames.append(weekly)
    return pd.DataFrame(rows).sort_values("objective", ascending=False), pd.concat(weekly_frames, ignore_index=True)


def _walk_forward(
    panel: pd.DataFrame,
    heads: list[str],
    min_heads: int,
    min_train_weeks: int,
    q35_weight: float,
    q20_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = {}
    for subset in _subsets(heads, min_heads):
        key = ",".join(subset)
        variants[key] = _variant_weekly(panel, subset).set_index("week")
    weeks = list(panel[["week", "week_start"]].drop_duplicates().sort_values("week_start")["week"])
    rows = []
    eval_frames = []
    for pos, week in enumerate(weeks):
        if pos < min_train_weeks:
            continue
        train_weeks = weeks[:pos]
        best_key = None
        best_obj = float("-inf")
        for key, weekly in variants.items():
            vals = weekly.loc[train_weeks, "net_pnl"].to_numpy()
            obj = _objective(vals, q35_weight, q20_weight)
            if obj > best_obj:
                best_obj = obj
                best_key = key
        assert best_key is not None
        picked = variants[best_key].loc[[week]].reset_index()
        picked["policy"] = "walk_forward"
        eval_frames.append(picked)
        rows.append(
            {
                "week": week,
                "selected_heads": best_key,
                "disabled_heads": ",".join(sorted(set(heads) - set(best_key.split(",")))),
                "train_objective": best_obj,
                "net_pnl": float(picked["net_pnl"].iloc[0]),
                "trades": int(picked["trades"].iloc[0]),
                "hit_rate": float(picked["hit_rate"].iloc[0]),
                "full_sl_rate": float(picked["full_sl_rate"].iloc[0]),
                "timeout_rate": float(picked["timeout_rate"].iloc[0]),
            }
        )
    eval_weekly = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    selections = pd.DataFrame(rows)
    return selections, eval_weekly


def _with_baseline_deltas(frame: pd.DataFrame, baseline: pd.DataFrame, key: str = "week") -> pd.DataFrame:
    base = baseline[[key, "net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"]].rename(
        columns={
            "net_pnl": "baseline_net_pnl",
            "gross_pnl": "baseline_gross_pnl",
            "trades": "baseline_trades",
            "hit_rate": "baseline_hit_rate",
            "full_sl_rate": "baseline_full_sl_rate",
            "timeout_rate": "baseline_timeout_rate",
        }
    )
    out = frame.merge(base, on=key, how="left")
    for col in ("net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"):
        bcol = f"baseline_{col}"
        if col in out.columns and bcol in out.columns:
            out[f"delta_{col}_vs_baseline"] = out[col] - out[bcol]
    return out


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    view = frame[columns].head(max_rows).copy() if max_rows else frame[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heads", default="")
    parser.add_argument("--min-heads", type=int, default=1)
    parser.add_argument("--min-train-weeks", type=int, default=8)
    parser.add_argument("--q35-weight", type=float, default=DEFAULT_OBJECTIVE_Q35_WEIGHT)
    parser.add_argument("--q20-weight", type=float, default=DEFAULT_OBJECTIVE_Q20_WEIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_weekly(args.weekly_metrics)
    heads = _parse_csv(args.heads) or sorted(panel["head"].unique())
    panel = panel[panel["head"].isin(heads)].copy()

    static_summary, static_weekly = _static_table(panel, heads, args.min_heads, args.q35_weight, args.q20_weight)
    static_summary["disabled_heads"] = static_summary["disabled_heads"].fillna("")
    all_heads_weekly = _variant_weekly(panel, heads)
    all_heads_summary = _summarize(all_heads_weekly["net_pnl"].to_numpy(), args.q35_weight, args.q20_weight)
    all_heads_summary.update({"enabled_heads": ",".join(heads), "disabled_heads": ""})

    selections, wf_weekly = _walk_forward(panel, heads, args.min_heads, args.min_train_weeks, args.q35_weight, args.q20_weight)
    wf_weekly = _with_baseline_deltas(wf_weekly, all_heads_weekly) if not wf_weekly.empty else wf_weekly
    wf_summary = _summarize(wf_weekly["net_pnl"].to_numpy(), args.q35_weight, args.q20_weight) if not wf_weekly.empty else {}
    baseline_eval = all_heads_weekly[all_heads_weekly["week"].isin(set(wf_weekly["week"]))] if not wf_weekly.empty else all_heads_weekly.iloc[:0]
    baseline_eval_summary = _summarize(baseline_eval["net_pnl"].to_numpy(), args.q35_weight, args.q20_weight)

    static_summary.to_csv(args.output_dir / "head_mix_static_summary.csv", index=False)
    static_weekly.to_csv(args.output_dir / "head_mix_static_weekly.csv", index=False)
    selections.to_csv(args.output_dir / "head_mix_walk_forward_selections.csv", index=False)
    wf_weekly.to_csv(args.output_dir / "head_mix_walk_forward_weekly.csv", index=False)
    all_heads_weekly.to_csv(args.output_dir / "head_mix_all_heads_weekly.csv", index=False)

    june_weeks = [w for w in sorted(all_heads_weekly["week"].unique()) if str(w).startswith("2026-06")]
    june_static = static_weekly[static_weekly["week"].isin(june_weeks)]
    june_summary = (
        june_static.groupby("enabled_heads", as_index=False)
        .agg(net_pnl=("net_pnl", "sum"), worst_week_net_pnl=("net_pnl", "min"), trades=("trades", "sum"))
        .sort_values("net_pnl", ascending=False)
    )
    june_summary.to_csv(args.output_dir / "head_mix_june_static_summary.csv", index=False)

    lines = [
        "# Head-Mix Proxy Ablation",
        "",
        "This is a proxy using additive weekly per-head replay metrics. It does not rerun the portfolio auction or backfill released capacity.",
        "",
        f"Input weekly metrics: `{args.weekly_metrics}`",
        f"Objective: `avg_week_net_pnl + {args.q35_weight:.2f} * q35_week_net_pnl + {args.q20_weight:.2f} * q20_week_net_pnl`",
        "",
        "## Static Head Subsets: Top 10",
        "",
        _markdown_table(
            static_summary,
            [
                "enabled_heads",
                "disabled_heads",
                "objective",
                "sum_net_pnl",
                "avg_week_net_pnl",
                "q15_week_net_pnl",
                "q20_week_net_pnl",
                "q35_week_net_pnl",
                "worst_week_net_pnl",
                "positive_weeks",
                "trades",
            ],
            10,
        ),
        "",
        "## Walk-Forward Head Selection",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"policy": "walk_forward", **wf_summary},
                    {"policy": "all_heads_on_same_eval_weeks", **baseline_eval_summary},
                ]
            ),
            [
                "policy",
                "weeks",
                "sum_net_pnl",
                "avg_week_net_pnl",
                "q15_week_net_pnl",
                "q20_week_net_pnl",
                "q35_week_net_pnl",
                "worst_week_net_pnl",
                "positive_weeks",
                "objective",
            ],
        ),
        "",
        "## Walk-Forward Selections",
        "",
        _markdown_table(
            selections,
            ["week", "selected_heads", "disabled_heads", "train_objective", "net_pnl", "trades", "hit_rate", "full_sl_rate"],
        ),
        "",
        "## June Static Head-Mix Headroom",
        "",
        _markdown_table(june_summary, ["enabled_heads", "net_pnl", "worst_week_net_pnl", "trades"], 12),
        "",
        "## Readout Guidance",
        "",
        "- If a static subset improves June but loses too much full-window objective, the issue is a conditional head-allocation problem.",
        "- If walk-forward selection fails to pick the useful subset before June, the available trailing metrics are not sufficient as deployed signals.",
        "- Because this is additive, any promising subset still needs a real portfolio replay before promotion.",
    ]
    (args.output_dir / "head_mix_proxy_ablation_report.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "head_mix_proxy_ablation_report.md")


if __name__ == "__main__":
    main()
