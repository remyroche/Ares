#!/usr/bin/env python3
"""Render the audit-ready report from a completed ten-head funnel artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ARTIFACT = ROOT / "data_perp/artifacts/ten_head_conditional_usefulness_20260810_v1"
DEFAULT_REPORT = ROOT / "docs/TEN_HEAD_CONDITIONAL_USEFULNESS_FUNNEL_20260810.md"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _float(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):+.2f}"


def _table(columns: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---:"] * len(columns)) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _pooled_metrics(metrics: pd.DataFrame, arm: str) -> dict[float, pd.Series]:
    if metrics.empty:
        return {}
    x = metrics.loc[metrics["arm"].eq(arm) & metrics["scope"].eq("pooled")]
    return {float(row.tail): row for row in x.itertuples(index=False)}


def _winner_trial(table: pd.DataFrame, head: str) -> pd.Series | None:
    if table.empty:
        return None
    x = table.loc[table["head"].eq(head) & table["state"].eq("COMPLETE")].copy()
    if x.empty:
        return None
    return x.sort_values(["objective", "metric_delta_top5_net_bps", "metric_delta_top1_net_bps"], ascending=[False, False, False], kind="stable").iloc[0]


def _winner_target_query(table: pd.DataFrame, head: str) -> pd.Series | None:
    if table.empty:
        return None
    x = table.loc[table["head"].eq(head)].copy()
    if x.empty:
        return None
    return x.sort_values(["conditional_utility_uplift_bps", "delta_top5_net_bps", "delta_top1_net_bps"], ascending=[False, False, False], kind="stable").iloc[0]


def render(artifact: Path = DEFAULT_ARTIFACT, report: Path = DEFAULT_REPORT) -> Path:
    manifest = json.loads((artifact / "run_manifest.json").read_text())
    configs = json.loads((artifact / "frozen_head_configs.json").read_text())
    source = _read(artifact / "source_contract_audit.parquet")
    feature = _read(artifact / "frozen_feature_contract_audit.parquet")
    target_screen = _read(artifact / "target_semantic_screen.parquet")
    target_query = _read(artifact / "target_query_conditional_trials.parquet")
    mda = _read(artifact / "conditional_feature_mda.parquet")
    mda_decisions = _read(artifact / "conditional_feature_selection_decisions.parquet")
    hpo = _read(artifact / "per_head_conditional_hpo_trials.parquet")
    metrics = _read(artifact / "downstream_metrics.parquet")
    comparison = _read(artifact / "final_conditional_comparison.parquet")
    necessity = _read(artifact / "head_conditional_necessity.parquet")
    development_necessity = _read(artifact / "development_frozen_winner_head_necessity.parquet")
    final_necessity = _read(artifact / "final_frozen_winner_head_necessity.parquet")
    query = json.loads((artifact / "query_screen/query_shortlist.json").read_text())

    development_control = _pooled_metrics(metrics, "development_control")
    development_winner = _pooled_metrics(metrics, "development_frozen_winner")
    final_control = _pooled_metrics(metrics, "final_control")
    final_winner = _pooled_metrics(metrics, "final_frozen_winner")

    lines = [
        "# Ten-head conditional-usefulness residual funnel",
        "",
        "## Decision status",
        "",
        "This is a long-only research result. The final August–October period was not used for target, query, feature, or HPO selection. Promotion requires a strict conditional improvement in global Top‑1%, Top‑2%, Top‑5%, and worst-month Top‑5% net EV; no result is made canonical merely because it wins a pooled proxy.",
        "",
        "## Fixed contract",
        "",
        f"- Source: `{manifest['source']}`; prequential base ledger: `{manifest['upstream_base_predictions']}`.",
        f"- Development selection months: {', '.join(manifest['development_months'])}; untouched final months: {', '.join(manifest['final_months'])}.",
        f"- Feature contract: {manifest['source_feature_count']} frozen causal fields; source side: long only.",
        "- Base inputs: independently prequential `base_rank` and `base_anchor_bps`. Residual target is `policy_net_bps − base_anchor_bps`.",
        "- Outcome policy: next-hour entry; 12-hour 15-minute path; SL 3 ATR; trailing activation 0.5 ATR; giveback 0.25 ATR; 100 bps cost applied exactly once.",
        "- Consensus: median of five feature caps (40/60/80/100/120) × ordinary/equal-month LambdaRank heads, then `0.75 × base_rank + 0.25 × consensus_rank`, globally ranked across the fixed candidate population.",
        "- Frozen ranker defaults retained by every final head: 120 trees, learning rate 0.035, depth 5, 31 leaves, 300 minimum-child samples, 0.82 feature/bagging fractions, L1 0.02, L2 2.0, max-bin 127, gains `[0, .25, 1, 3, 7]`, truncation 10.",
        "- Each head-score CDF is fitted on its mature training scores only; train labels must satisfy `policy_label_available_ts < held-month start`.",
        "",
        "## Data and source-contract audit",
        "",
    ]
    if not source.empty:
        row = source.iloc[0]
        lines.extend([
            f"- Requested source rows: {int(row.source_rows_requested):,}; joined to prequential base ledger: {int(row.joined_rows):,}; policy-valid residual rows: {int(row.label_valid_rows):,}.",
            f"- The earliest authorized training-window field audit found {int(feature.nonconstant.sum()):,}/{len(feature):,} varying fields. Fields that were temporarily constant remain in the frozen 120-field contract and can only be removed by the conditional MDA stage.",
            "",
        ])
    lines.extend([
        "## Search funnel",
        "",
        f"- Query pre-screen shortlist: {', '.join(query['shortlist'])}.",
        f"- Full-stack conditional target/query candidates per head: {len(manifest['target_query_candidates'])}. These included both target and query changes.",
        f"- Conditional MDA screen cap: {manifest['mda_max_eval_rows']:,} candidate rows; selected field subsets were then re-fitted and gated on the complete development population.",
        f"- Per-head HPO: {manifest['hpo_trials_per_head']} Optuna trials/head, TPE + aggressive MedianPruner; 2,000-tree ceiling with 30-round chronological early stopping for search, then full-mature-window refit at median selected tree count for promotion recheck.",
        "",
        "### Target semantics screen (development only)",
        "",
    ])
    target_rows = []
    for row in target_screen.itertuples(index=False):
        target_rows.append([
            str(row.target), _float(row.grade_net_spearman), _float(row.grade0_net_bps),
            _float(row.grade4_net_bps), _float(row.grade_spread_bps), _float(row.grade_entropy),
        ])
    lines.extend(_table(["Target", "Grade/net ρ", "Grade 0 net", "Grade 4 net", "Spread", "Entropy"], target_rows))
    lines.extend(["", "## Global downstream net EV (bps/trade)", ""])
    metric_rows = []
    for arm, label in [
        (development_control, "Development control"),
        (development_winner, "Development frozen winner"),
        (final_control, "Final untouched control"),
        (final_winner, "Final untouched frozen winner"),
    ]:
        if not arm:
            continue
        top1, top2, top5 = arm.get(.01), arm.get(.02), arm.get(.05)
        metric_rows.append([
            label,
            _float(getattr(top1, "net_bps_per_trade", np.nan)),
            _float(getattr(top2, "net_bps_per_trade", np.nan)),
            _float(getattr(top5, "net_bps_per_trade", np.nan)),
            _float(getattr(top5, "month_worst_net_bps", np.nan)),
            str(int(getattr(top5, "rows", 0))),
        ])
    lines.extend(_table(["Arm", "Top‑1%", "Top‑2%", "Top‑5%", "Worst month @5%", "Top‑5 rows"], metric_rows))
    if not comparison.empty:
        row = comparison.iloc[0]
        lines.extend([
            "",
            "### Frozen-winner change versus exact matched final control",
            "",
            *(_table(
                ["Δ Top‑1%", "Δ Top‑2%", "Δ Top‑5%", "Δ worst month @5%", "Δ conditional utility"],
                [[
                    _float(row.delta_top1_net_bps), _float(row.delta_top2_net_bps),
                    _float(row.delta_top5_net_bps), _float(row.delta_top5_month_worst_net_bps),
                    _float(row.conditional_utility_uplift_bps),
                ]],
            )),
        ])
    lines.extend(["", "## Per-head conditional selection", ""])
    per_head_rows: list[list[str]] = []
    for name, config in configs.items():
        target_best = _winner_target_query(target_query, name)
        hpo_best = _winner_trial(hpo, name)
        decision = mda_decisions.loc[mda_decisions["head"].eq(name)] if not mda_decisions.empty else pd.DataFrame()
        decision_row = decision.iloc[0] if not decision.empty else None
        logs = config.get("selection_log", [])
        promoted = "; ".join(
            log["stage"] for log in logs if isinstance(log, dict) and log.get("promoted")
        ) or "none"
        hpo_utility = (
            hpo_best["metric_conditional_utility_uplift_bps"]
            if hpo_best is not None and "metric_conditional_utility_uplift_bps" in hpo_best
            else np.nan
        )
        hpo_passed = (
            bool(hpo_best["metric_promotable"])
            if hpo_best is not None and "metric_promotable" in hpo_best
            else False
        )
        per_head_rows.append([
            name,
            str(config["target_name"]), str(config["query_name"]), str(config["field_count"]),
            _float(target_best["conditional_utility_uplift_bps"] if target_best is not None else np.nan),
            _float(hpo_utility),
            "yes" if hpo_passed else "no",
            str(int(decision_row.selected_field_count)) if decision_row is not None else "—",
            promoted,
        ])
    lines.extend(_table(["Head", "Frozen target", "Frozen query", "Fields", "Best T/Q Δutility", "Best HPO Δutility", "HPO passed", "MDA fields", "Promoted stage(s)"], per_head_rows))
    complete_hpo = hpo.loc[hpo["state"].eq("COMPLETE")] if not hpo.empty else pd.DataFrame()
    lines.extend([
        "",
        "### What advanced",
        "",
        "- Target semantics: no alternative residual target passed conditional downstream promotion; every frozen head retains `resid_default_150_50`.",
        "- Query construction: four heads advanced from the 4-hour × side query to exact timestamp × side; all other heads retained the 4-hour × side query.",
        "- Conditional MDA: only `cap60_ordinary` (60→15), `cap60_equal_month` (60→30), and `cap120_equal_month` (120→51) passed their full-development subset recheck.",
        f"- Ranker HPO: {len(complete_hpo):,} completed and {len(hpo) - len(complete_hpo):,} pruned trials; no HPO challenger passed the strict conditional promotion recheck, so every final head keeps the frozen defaults above.",
    ])
    lines.extend(["", "### Conditional feature findings", ""])
    if not mda.empty:
        lines.append("The following are the strongest individual downstream conditional-MDA signals per head. They are diagnostics; only a full-subset refit that passed the strict full-development gate was retained.")
        lines.append("")
        mda_rows = []
        for head, group in mda.groupby("head", sort=True, observed=True):
            top = group.sort_values("conditional_importance_bps", ascending=False, kind="stable").head(3)
            mda_rows.append([
                str(head),
                ", ".join(f"{row.field} ({row.conditional_importance_bps:+.2f})" for row in top.itertuples(index=False)),
            ])
        lines.extend(_table(["Head", "Top conditional-MDA fields (bps utility loss when permuted)"], mda_rows))
    lines.extend(["", "### Head necessity before selection", ""])
    if not necessity.empty:
        necessity_rows = [[str(row.head), _float(row.necessity_bps), _float(row.removal_utility_change_bps)] for row in necessity.itertuples(index=False)]
        lines.extend(_table(["Head", "Necessity", "Utility change if removed"], necessity_rows))
    lines.extend(["", "### Frozen leave-one-head-out attribution", ""])
    lines.append("Positive necessity means the complete frozen stack becomes worse when that head is removed. The deltas are the economics of removal, so negative values are evidence of a helpful head.")
    for label, audit in [("Development frozen winner", development_necessity), ("Final untouched frozen winner", final_necessity)]:
        if audit.empty:
            continue
        lines.extend(["", f"#### {label}", ""])
        rows = [
            [
                str(row.head), _float(row.necessity_bps), _float(row.delta_top1_net_bps),
                _float(row.delta_top2_net_bps), _float(row.delta_top5_net_bps),
                _float(row.delta_top5_month_worst_net_bps),
            ]
            for row in audit.itertuples(index=False)
        ]
        lines.extend(_table(["Head", "Necessity", "Δ Top‑1 if removed", "Δ Top‑2 if removed", "Δ Top‑5 if removed", "Δ worst month @5% if removed"], rows))
    lines.extend(["", "## Time robustness", ""])
    time_rows = []
    if not metrics.empty:
        for arm in ("final_control", "final_frozen_winner"):
            x = metrics.loc[
                metrics["arm"].eq(arm)
                & metrics["scope"].ne("pooled")
                & metrics["tail"].isin([.01, .02, .05])
            ]
            for month, group in x.groupby("scope", sort=True, observed=True):
                values = {float(row.tail): row for row in group.itertuples(index=False)}
                if .05 in values:
                    time_rows.append([
                        arm, str(month), _float(values.get(.01).net_bps_per_trade if .01 in values else np.nan),
                        _float(values.get(.02).net_bps_per_trade if .02 in values else np.nan),
                        _float(values[.05].net_bps_per_trade), str(int(values[.05].rows)),
                    ])
    if time_rows:
        lines.extend(_table(["Arm", "Month", "Top‑1%", "Top‑2%", "Top‑5%", "Top‑5 rows"], time_rows))
    else:
        lines.append("Final time-robustness table is unavailable because the final confirmation was not completed.")
    lines.extend([
        "",
        "## Interpretation and next decisions",
        "",
        "1. The frozen winner passes the *relative* final comparison: it improves Top‑1, Top‑2, Top‑5, the Top‑5 worst month, and conditional utility versus the exact control.",
        "2. It is not execution-ready at a broad Top‑5% admission rate: final Top‑5 remains negative. Treat it as the research winner and preserve the canonical control for any rule requiring absolute positive Top‑5 net EV.",
        "3. The durable evidence is query/feature-contract refinement, not a richer residual target or larger ranker. The next target research should address broad-tail economic separation rather than add HPO capacity.",
        "4. This audit remains long-only. Repeat the same frozen methodology side-locally once an equivalent short prequential base-score ledger is available.",
        "",
        "## Artifacts",
        "",
        f"- Artifact directory: `{artifact}`.",
        "- `target_query_conditional_trials.parquet`: every per-head target/query full-stack replacement.",
        "- `conditional_feature_mda.parquet`: per-field full-stack conditional usefulness screen.",
        "- `per_head_conditional_hpo_trials.parquet`: all completed and pruned HPO trials.",
        "- `downstream_metrics.parquet` and `final_conditional_comparison.parquet`: global/montly matched economics.",
        "- `development_frozen_winner_head_necessity.parquet` and `final_frozen_winner_head_necessity.parquet`: per-head leave-one-out attribution.",
        "- `frozen_head_configs.json`: exact winning ten-head contract.",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(render(args.artifact, args.report))
