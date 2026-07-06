#!/usr/bin/env python3
"""Week-forward validation for contextual TP/SL overlay variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        return json.loads((path / "combo_replay_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _variant_dirs(root: Path, exclude_parts: set[str]) -> List[Path]:
    out: List[Path] = []
    for path in root.glob("*/materialized/*"):
        if any(part in exclude_parts for part in path.parts):
            continue
        if (path / "combo_replay_manifest.json").exists():
            out.append(path)
    return sorted(out)


def _global_week_rows(path: Path) -> pd.DataFrame:
    frame = _read_csv(path / "combo_replay_weekly_metrics.csv")
    if frame.empty:
        return pd.DataFrame()
    out = frame.loc[frame["period_type"].astype(str).eq("week")].copy()
    if "head" in out.columns:
        out = out.loc[out["head"].isna()].copy()
    out["week"] = out["week"].astype(str)
    return out


def _selector_stats(history: pd.DataFrame) -> Dict[str, float]:
    delta = pd.to_numeric(history["delta_net_pnl"], errors="coerce").dropna()
    if delta.empty:
        return {
            "train_weeks": 0,
            "train_sum_delta_net_pnl": np.nan,
            "train_avg_delta_net_pnl": np.nan,
            "train_q15_delta_net_pnl": np.nan,
            "train_q25_delta_net_pnl": np.nan,
            "train_positive_week_share": np.nan,
        }
    return {
        "train_weeks": int(len(delta)),
        "train_sum_delta_net_pnl": float(delta.sum()),
        "train_avg_delta_net_pnl": float(delta.mean()),
        "train_q15_delta_net_pnl": float(delta.quantile(0.15)),
        "train_q25_delta_net_pnl": float(delta.quantile(0.25)),
        "train_positive_week_share": float((delta > 0.0).mean()),
    }


def _selector_score(row: pd.Series, positive_weight: float) -> float:
    avg = float(row["train_avg_delta_net_pnl"])
    q25 = float(row["train_q25_delta_net_pnl"])
    q15 = float(row["train_q15_delta_net_pnl"])
    pos = float(row["train_positive_week_share"])
    return avg + 0.35 * q25 + 0.15 * q15 + positive_weight * pos * max(abs(avg), 1.0)


def _markdown_table(frame: pd.DataFrame, columns: List[str], limit: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    cur = frame[[col for col in columns if col in frame.columns]].head(limit).copy()
    for col in cur.columns:
        if pd.api.types.is_float_dtype(cur[col]):
            cur[col] = cur[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    return cur.to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", action="append", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-train-weeks", type=int, default=8)
    parser.add_argument("--rolling-train-weeks", type=int, default=0, help="0 means expanding history.")
    parser.add_argument("--positive-week-weight", type=float, default=0.10)
    parser.add_argument(
        "--exclude-root-names",
        default="noop_baseline,summary,tail_tradeoff_summary,grid_decision_summary,interaction_summary",
    )
    args = parser.parse_args()

    roots = [Path(path) for path in args.root_dir]
    baseline_dir = Path(args.baseline_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_parts = {part.strip() for part in str(args.exclude_root_names).split(",") if part.strip()}

    baseline_week = _global_week_rows(baseline_dir)
    if baseline_week.empty:
        raise ValueError(f"Missing baseline weekly rows under {baseline_dir}")
    baseline_week = baseline_week.rename(
        columns={
            "net_pnl": "baseline_net_pnl",
            "gross_pnl": "baseline_gross_pnl",
            "trades": "baseline_trades",
            "hit_rate": "baseline_hit_rate",
            "full_sl_rate": "baseline_full_sl_rate",
            "timeout_rate": "baseline_timeout_rate",
        }
    )

    rows: List[pd.DataFrame] = []
    for root in roots:
        for path in _variant_dirs(root, exclude_parts):
            manifest = _load_manifest(path)
            week = _global_week_rows(path)
            if week.empty:
                continue
            week = week.merge(baseline_week, on="week", how="inner")
            if week.empty:
                continue
            week.insert(0, "source_root", str(root))
            week.insert(1, "label", path.name)
            week["rule_count"] = len(manifest.get("rules", []))
            for col in ["net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"]:
                week[f"delta_{col}"] = pd.to_numeric(week[col], errors="coerce") - pd.to_numeric(
                    week[f"baseline_{col}"], errors="coerce"
                )
            rows.append(week)
    all_weeks = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if all_weeks.empty:
        raise ValueError("No variant weekly rows found")

    weeks = sorted(all_weeks["week"].unique())
    selections: List[Dict[str, Any]] = []
    rankings: List[pd.DataFrame] = []
    for pos, eval_week in enumerate(weeks):
        prior_weeks = weeks[:pos]
        if len(prior_weeks) < int(args.min_train_weeks):
            continue
        if int(args.rolling_train_weeks) > 0:
            prior_weeks = prior_weeks[-int(args.rolling_train_weeks) :]
        history = all_weeks.loc[all_weeks["week"].isin(prior_weeks)].copy()
        eval_rows = all_weeks.loc[all_weeks["week"].eq(eval_week)].copy()
        train_rows: List[Dict[str, Any]] = []
        for (label, source_root), group in history.groupby(["label", "source_root"], sort=False):
            stats = _selector_stats(group)
            stats.update({"label": label, "source_root": source_root, "eval_week": eval_week})
            train_rows.append(stats)
        rank = pd.DataFrame(train_rows).dropna(subset=["train_avg_delta_net_pnl"])
        if rank.empty:
            continue
        rank["selector_score"] = rank.apply(lambda row: _selector_score(row, float(args.positive_week_weight)), axis=1)
        rank = rank.sort_values(["selector_score", "train_sum_delta_net_pnl"], ascending=False)
        rank["rank"] = np.arange(1, len(rank) + 1)
        rankings.append(rank)
        chosen = rank.iloc[0]
        chosen_eval = eval_rows.loc[
            eval_rows["label"].eq(chosen["label"]) & eval_rows["source_root"].eq(chosen["source_root"])
        ]
        if chosen_eval.empty:
            continue
        erow = chosen_eval.iloc[0]
        selections.append(
            {
                "eval_week": eval_week,
                "chosen_label": chosen["label"],
                "chosen_source_root": chosen["source_root"],
                "selector_score": float(chosen["selector_score"]),
                "train_weeks": int(chosen["train_weeks"]),
                "train_sum_delta_net_pnl": float(chosen["train_sum_delta_net_pnl"]),
                "train_avg_delta_net_pnl": float(chosen["train_avg_delta_net_pnl"]),
                "train_q15_delta_net_pnl": float(chosen["train_q15_delta_net_pnl"]),
                "train_q25_delta_net_pnl": float(chosen["train_q25_delta_net_pnl"]),
                "train_positive_week_share": float(chosen["train_positive_week_share"]),
                "eval_net_pnl": float(erow["net_pnl"]),
                "baseline_eval_net_pnl": float(erow["baseline_net_pnl"]),
                "delta_eval_net_pnl": float(erow["delta_net_pnl"]),
                "delta_eval_gross_pnl": float(erow["delta_gross_pnl"]),
                "delta_eval_hit_rate": float(erow["delta_hit_rate"]),
                "delta_eval_full_sl_rate": float(erow["delta_full_sl_rate"]),
                "rule_count": int(erow.get("rule_count", 0)),
            }
        )

    selections_df = pd.DataFrame(selections)
    rankings_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    all_weeks.to_csv(out_dir / "week_forward_all_variant_weeks.csv", index=False)
    rankings_df.to_csv(out_dir / "week_forward_training_rankings.csv", index=False)
    selections_df.to_csv(out_dir / "week_forward_selections.csv", index=False)

    static_rows: List[Dict[str, Any]] = []
    eval_weeks = selections_df["eval_week"].tolist() if not selections_df.empty else []
    for (label, source_root), group in all_weeks.loc[all_weeks["week"].isin(eval_weeks)].groupby(
        ["label", "source_root"], sort=False
    ):
        delta = pd.to_numeric(group["delta_net_pnl"], errors="coerce").dropna()
        if delta.empty:
            continue
        static_rows.append(
            {
                "label": label,
                "source_root": source_root,
                "eval_weeks": int(len(delta)),
                "sum_delta_net_pnl": float(delta.sum()),
                "mean_delta_net_pnl": float(delta.mean()),
                "q15_delta_net_pnl": float(delta.quantile(0.15)),
                "q25_delta_net_pnl": float(delta.quantile(0.25)),
                "median_delta_net_pnl": float(delta.median()),
                "positive_week_count": int((delta > 0.0).sum()),
                "worst_delta_net_pnl": float(delta.min()),
                "rule_count": int(group["rule_count"].iloc[0]) if "rule_count" in group else 0,
            }
        )
    static_df = pd.DataFrame(static_rows).sort_values(
        ["mean_delta_net_pnl", "q25_delta_net_pnl", "sum_delta_net_pnl"], ascending=False
    )
    static_df.to_csv(out_dir / "week_forward_static_candidates.csv", index=False)

    if not selections_df.empty:
        summary = {
            "eval_weeks": int(len(selections_df)),
            "sum_delta_net_pnl": float(selections_df["delta_eval_net_pnl"].sum()),
            "mean_delta_net_pnl": float(selections_df["delta_eval_net_pnl"].mean()),
            "q15_delta_net_pnl": float(selections_df["delta_eval_net_pnl"].quantile(0.15)),
            "q25_delta_net_pnl": float(selections_df["delta_eval_net_pnl"].quantile(0.25)),
            "median_delta_net_pnl": float(selections_df["delta_eval_net_pnl"].median()),
            "positive_week_count": int((selections_df["delta_eval_net_pnl"] > 0.0).sum()),
            "worst_delta_net_pnl": float(selections_df["delta_eval_net_pnl"].min()),
            "chosen_labels": selections_df["chosen_label"].value_counts().to_dict(),
        }
    else:
        summary = {"eval_weeks": 0}
    (out_dir / "week_forward_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Contextual TP/SL Overlay Week-Forward Validation",
        "",
        "Candidate selection uses only weeks before the evaluation week.",
        f"History mode: {'rolling ' + str(args.rolling_train_weeks) + ' weeks' if int(args.rolling_train_weeks) > 0 else 'expanding'}; min train weeks: {args.min_train_weeks}.",
        "",
        "## Adaptive Selector Summary",
        "",
        _markdown_table(pd.DataFrame([summary]), list(summary.keys())),
        "",
        "## Adaptive Weekly Selections",
        "",
        _markdown_table(
            selections_df,
            [
                "eval_week",
                "chosen_label",
                "train_weeks",
                "train_positive_week_share",
                "delta_eval_net_pnl",
                "delta_eval_gross_pnl",
                "rule_count",
            ],
            80,
        ),
        "",
        "## Static Candidates On Same Eval Weeks",
        "",
        _markdown_table(
            static_df,
            [
                "label",
                "eval_weeks",
                "sum_delta_net_pnl",
                "mean_delta_net_pnl",
                "q15_delta_net_pnl",
                "q25_delta_net_pnl",
                "positive_week_count",
                "worst_delta_net_pnl",
                "rule_count",
            ],
            40,
        ),
    ]
    (out_dir / "week_forward_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "generated_by": "validate_contextual_tp_sl_overlay_week_forward",
        "root_dirs": [str(root) for root in roots],
        "baseline_dir": str(baseline_dir),
        "min_train_weeks": int(args.min_train_weeks),
        "rolling_train_weeks": int(args.rolling_train_weeks),
        "positive_week_weight": float(args.positive_week_weight),
        "outputs": [
            "week_forward_report.md",
            "week_forward_summary.json",
            "week_forward_all_variant_weeks.csv",
            "week_forward_training_rankings.csv",
            "week_forward_selections.csv",
            "week_forward_static_candidates.csv",
        ],
    }
    (out_dir / "week_forward_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
