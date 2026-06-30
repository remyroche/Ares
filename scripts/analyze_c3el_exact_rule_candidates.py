#!/usr/bin/env python3
"""Evaluate predeclared C3el monitoring rule candidates on exact-state labels.

The current C3el evidence is too small for a learned gate.  This diagnostic
tests explicit rule candidates and combinations on existing exact-state labels
to decide which hypotheses are worth shadow monitoring or collecting forward
exact-state labels for.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "data_perp/reports/c3el_exact_feature_slice_audit_20260628/"
    "exact_labels_with_action_features.csv"
)
DEFAULT_OUT_DIR = Path("data_perp/reports/c3el_exact_rule_candidate_audit_20260628")


RuleFn = Callable[[pd.DataFrame], pd.Series]


def _strict(frame: pd.DataFrame) -> pd.Series:
    return frame["bucket"].astype(str).eq("p80_d320")


BASE_CONDITIONS: dict[str, RuleFn] = {
    "cooldown_count_lte_38_5": lambda f: pd.to_numeric(f["cooldown_count"], errors="coerce").le(38.5),
    "timestamp_rank_q90_lte_0_8641": lambda f: pd.to_numeric(f["timestamp_rank_q90"], errors="coerce").le(0.8641),
    "open_or_cooldown_share_lte_0_3949": lambda f: pd.to_numeric(
        f["strategy_candidate_open_or_cooldown_symbol_share"], errors="coerce"
    ).le(0.3949),
    "strategy_rank_max_lte_0_9054": lambda f: pd.to_numeric(f["strategy_rank_max"], errors="coerce").le(0.9054),
}


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[frame["timestamp"].notna()].copy()
    frame["day"] = frame["timestamp"].dt.floor("D")
    for col in [
        "delta_full_J",
        "delta_immediate_J",
        "delta_full_net_pnl",
        "delta_full_cost_pnl",
        "delta_full_turnover",
        "p_intervene",
        "pred_action_delta_J",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _summarize(mask: pd.Series, frame: pd.DataFrame, *, name: str, definition: str) -> dict[str, Any]:
    selected = frame.loc[mask.fillna(False)].copy()
    rejected = frame.loc[~mask.fillna(False)].copy()
    day = (
        selected.groupby("day", dropna=False)["delta_full_J"].sum().reset_index(name="day_delta_full_J")
        if not selected.empty
        else pd.DataFrame(columns=["day", "day_delta_full_J"])
    )
    strict = frame.loc[_strict(frame)]
    selected_strict = selected.loc[selected["bucket"].astype(str).eq("p80_d320")]
    rejected_strict = strict.loc[~strict.index.isin(selected_strict.index)]
    return {
        "rule": name,
        "definition": definition,
        "rows": int(len(selected)),
        "strict_rows": int(len(selected_strict)),
        "rejected_strict_rows": int(len(rejected_strict)),
        "day_count": int(day["day"].nunique()) if not day.empty else 0,
        "positive_day_share": float(day["day_delta_full_J"].gt(0.0).mean()) if not day.empty else np.nan,
        "worst_day_delta_full_J": float(day["day_delta_full_J"].min()) if not day.empty else np.nan,
        "sum_delta_full_J": float(selected["delta_full_J"].sum()),
        "mean_delta_full_J": float(selected["delta_full_J"].mean()) if not selected.empty else np.nan,
        "median_delta_full_J": float(selected["delta_full_J"].median()) if not selected.empty else np.nan,
        "worst_delta_full_J": float(selected["delta_full_J"].min()) if not selected.empty else np.nan,
        "positive_share": float(selected["delta_full_J"].gt(0.0).mean()) if not selected.empty else np.nan,
        "sum_immediate_J": float(selected["delta_immediate_J"].sum()) if "delta_immediate_J" in selected else np.nan,
        "sum_net_pnl_delta": float(selected["delta_full_net_pnl"].sum()) if "delta_full_net_pnl" in selected else np.nan,
        "sum_cost_delta": float(selected["delta_full_cost_pnl"].sum()) if "delta_full_cost_pnl" in selected else np.nan,
        "sum_turnover_delta": float(selected["delta_full_turnover"].sum()) if "delta_full_turnover" in selected else np.nan,
        "rejected_strict_sum_delta_full_J": float(rejected_strict["delta_full_J"].sum()),
        "rejected_strict_positive_share": (
            float(rejected_strict["delta_full_J"].gt(0.0).mean()) if not rejected_strict.empty else np.nan
        ),
        "defensive_success_vs_strict_rejects": float(
            -rejected_strict["delta_full_J"].sum()
            + min(float(selected["delta_full_J"].sum()), 0.0)
        ),
        "coverage_of_strict": float(len(selected_strict) / len(strict)) if len(strict) else np.nan,
    }


def _rule_masks(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    strict_mask = _strict(frame)
    conditions = {name: fn(frame).fillna(False) for name, fn in BASE_CONDITIONS.items()}
    rules: list[tuple[str, str, pd.Series]] = [
        ("strict_p80_d320", "bucket == p80_d320", strict_mask),
    ]
    for name, cond in conditions.items():
        rules.append((f"strict__{name}", f"p80_d320 AND {name}", strict_mask & cond))
    for a, b in itertools.combinations(conditions, 2):
        mask = strict_mask & conditions[a] & conditions[b]
        rules.append((f"strict__{a}__{b}", f"p80_d320 AND {a} AND {b}", mask))
    condition_count = sum(cond.astype(int) for cond in conditions.values())
    for threshold in (2, 3, 4):
        rules.append(
            (
                f"strict__at_least_{threshold}_conditions",
                f"p80_d320 AND at least {threshold} of 4 monitoring conditions",
                strict_mask & condition_count.ge(threshold),
            )
        )
    return rules


def evaluate(frame: pd.DataFrame, *, min_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    day_rows = []
    for name, definition, mask in _rule_masks(frame):
        summary = _summarize(mask, frame, name=name, definition=definition)
        summaries.append(summary)
        selected = frame.loc[mask.fillna(False)].copy()
        if not selected.empty:
            by_day = selected.groupby("day", dropna=False).agg(
                rows=("timestamp", "size"),
                positive_share=("delta_full_J", lambda s: float((s > 0).mean())),
                sum_delta_full_J=("delta_full_J", "sum"),
                median_delta_full_J=("delta_full_J", "median"),
                worst_delta_full_J=("delta_full_J", "min"),
            )
            by_day = by_day.reset_index()
            by_day.insert(0, "rule", name)
            day_rows.append(by_day)
    summary_df = pd.DataFrame(summaries)
    summary_df["passes_min_rows"] = summary_df["rows"].ge(int(min_rows))
    summary_df["score"] = (
        summary_df["sum_delta_full_J"]
        + 0.5 * summary_df["median_delta_full_J"].fillna(0.0)
        + 0.25 * summary_df["worst_day_delta_full_J"].fillna(0.0)
        + 500.0 * summary_df["positive_day_share"].fillna(0.0)
    )
    summary_df = summary_df.sort_values(["passes_min_rows", "score"], ascending=[False, False])
    day_df = pd.concat(day_rows, ignore_index=True, sort=False) if day_rows else pd.DataFrame()
    return summary_df, day_df


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def write_report(summary_df: pd.DataFrame, day_df: pd.DataFrame, frame: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "rule_candidate_summary.csv", index=False)
    day_df.to_csv(out_dir / "rule_candidate_by_day.csv", index=False)

    display_cols = [
        "rule",
        "rows",
        "strict_rows",
        "coverage_of_strict",
        "positive_share",
        "positive_day_share",
        "sum_delta_full_J",
        "median_delta_full_J",
        "worst_delta_full_J",
        "worst_day_delta_full_J",
        "rejected_strict_sum_delta_full_J",
        "defensive_success_vs_strict_rejects",
        "passes_min_rows",
    ]
    top = summary_df[display_cols].head(20)
    strict = summary_df.loc[summary_df["rule"].eq("strict_p80_d320")]
    best = summary_df.loc[summary_df["passes_min_rows"]].head(1)

    lines = [
        "# C3el Exact-State Rule Candidate Audit",
        "",
        "This report evaluates predeclared strict p80/d320 monitoring rules and simple combinations.",
        "",
        f"Input rows: `{len(frame)}`",
        f"Strict p80/d320 rows: `{int(frame['bucket'].astype(str).eq('p80_d320').sum())}`",
        "",
        "## Top Rule Candidates",
        "",
        top.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Baseline Strict Rule",
        "",
        strict[display_cols].to_markdown(index=False, floatfmt=".4f") if not strict.empty else "Missing strict baseline.",
        "",
        "## Current Readout",
        "",
    ]
    if best.empty:
        lines.append("No rule passed the minimum-row threshold.")
    else:
        row = best.iloc[0]
        lines.extend(
            [
                f"Best minimum-support candidate: `{row['rule']}`.",
                "",
                f"- rows: `{int(row['rows'])}`",
                f"- strict coverage: `{row['coverage_of_strict']:.2%}`",
                f"- positive share: `{row['positive_share']:.2%}`",
                f"- positive day share: `{row['positive_day_share']:.2%}`",
                f"- sum delta full-path utility: `{row['sum_delta_full_J']:.2f}`",
                f"- rejected strict sum delta: `{row['rejected_strict_sum_delta_full_J']:.2f}`",
                "",
                "This remains a monitoring hypothesis, not a deployment rule. The same exact-state labels were used to discover the candidate slices.",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Keep the base strict p80/d320 rule as the primary high-conviction C3el state.",
            "- Shadow-monitor the best minimum-support conjunctive rule, but do not hard-gate live trades yet.",
            "- New exact-state labels should be collected for all strict p80/d320 firings and tagged by these rule candidates before promotion.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    manifest = {
        "generated_by": "analyze_c3el_exact_rule_candidates",
        "rows": int(len(frame)),
        "strict_p80_d320_rows": int(frame["bucket"].astype(str).eq("p80_d320").sum()),
        "outputs": {
            "summary": str(out_dir / "summary.md"),
            "rule_summary": str(out_dir / "rule_candidate_summary.csv"),
            "by_day": str(out_dir / "rule_candidate_by_day.csv"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-rows", type=int, default=10)
    args = parser.parse_args()

    frame = _load(args.input)
    summary_df, day_df = evaluate(frame, min_rows=args.min_rows)
    manifest = write_report(summary_df, day_df, frame, args.out_dir)
    print((args.out_dir / "summary.md").read_text())
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
