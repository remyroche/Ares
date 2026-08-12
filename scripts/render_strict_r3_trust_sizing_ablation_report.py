#!/usr/bin/env python3
"""Consolidate the three strict-R3 trust-sizing ablation funnels.

The source experiment directories are immutable.  This renderer creates one
new immutable comparison bundle plus a Markdown decision report.  It never
refits a model and never changes the frozen 2025 selections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
DISPLAY_TAILS = (0.01, 0.02, 0.05)
SOURCES = {
    "bayesian": {
        2025: ROOT / "data_perp/artifacts/strict_r3_trust_sizing_bayesian_2025_3m_20260810_v1",
        2026: ROOT / "data_perp/artifacts/strict_r3_trust_sizing_bayesian_2026_3m_20260810_v2",
    },
    "gam": {
        2025: ROOT / "data_perp/artifacts/strict_r3_trust_sizing_gam_2025_3m_20260810_v1",
        2026: ROOT / "data_perp/artifacts/strict_r3_trust_sizing_gam_2026_3m_20260810_v2",
    },
    "nonlinear": {
        2025: ROOT / "data_perp/artifacts/strict_r3_trust_sizing_nonlinear_drf_2025_3m_20260810_v2",
        2026: ROOT / "data_perp/artifacts/strict_r3_trust_sizing_nonlinear_2026_3m_20260810_v3",
    },
}
NONLINEAR_DEVELOPMENT_SHARDS = {
    "ngboost": ROOT / "data_perp/artifacts/strict_r3_trust_sizing_nonlinear_ngboost_2025_3m_20260810_v2",
    "distributional_forest": ROOT / "data_perp/artifacts/strict_r3_trust_sizing_nonlinear_drf_2025_3m_20260810_v2",
    "distributional_mlp": ROOT / "data_perp/artifacts/strict_r3_trust_sizing_nonlinear_mlp_2025_3m_20260810_v3",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(path / name)


def _top3(path: Path) -> list[str]:
    return list(json.loads((path / "selected_top3.json").read_text())["selected_top3"])


def _md(frame: pd.DataFrame, *, decimals: int = 2) -> str:
    copy = frame.copy()
    for column in copy.select_dtypes(include=[np.number]).columns:
        copy[column] = copy[column].round(decimals)
    headers = [str(column).replace("|", "\\|") for column in copy.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for values in copy.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if pd.isna(value):
                text = ""
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.{decimals}f}"
            else:
                text = str(value)
            rendered.append(text.replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join(rows)


def _weekly_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    values = weekly.loc[weekly["tail"].isin(DISPLAY_TAILS)].copy()
    result = (
        values.groupby(["pipeline", "year", "arm", "tail"], sort=False)
        ["exposure_weighted_net_bps"]
        .agg(
            week_median_bps="median",
            worst_week_bps="min",
            best_week_bps="max",
            positive_weeks=lambda series: int((series > 0).sum()),
            weeks="count",
        )
        .reset_index()
    )
    return result


def _selection_rows() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for pipeline in ("bayesian", "gam"):
        frame = _read(SOURCES[pipeline][2025], "selection_metrics.parquet")
        parts.append(frame.assign(pipeline=pipeline, development_shard=pipeline))
    for shard, path in NONLINEAR_DEVELOPMENT_SHARDS.items():
        frame = _read(path, "selection_metrics.parquet")
        parts.append(frame.assign(pipeline="nonlinear", development_shard=shard))
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["pipeline", "selection_score"], ascending=[True, False])
        .drop_duplicates(["pipeline", "arm"])
        .reset_index(drop=True)
    )


def _collect() -> dict[str, pd.DataFrame]:
    names = (
        "metrics_global.parquet",
        "metrics_monthly.parquet",
        "metrics_weekly.parquet",
        "stability.parquet",
        "portfolio_summary.parquet",
        "portfolio_monthly.parquet",
        "portfolio_weekly.parquet",
        "fold_audit.parquet",
        "cmi_edge_audit.parquet",
    )
    buckets: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    for pipeline, years in SOURCES.items():
        selected = _top3(years[2025])
        for year, path in years.items():
            global_frame = _read(path, "metrics_global.parquet")
            keep = set(selected)
            controls = [value for value in global_frame["arm"].unique() if value.endswith("_equal_control")]
            keep.update(controls)
            for name in names:
                frame = _read(path, name)
                if "arm" in frame:
                    frame = frame.loc[frame["arm"].isin(keep)].copy()
                buckets[name].append(frame.assign(pipeline=pipeline, year=year))
    return {name: pd.concat(parts, ignore_index=True) for name, parts in buckets.items()}


def _report(tables: dict[str, pd.DataFrame], selection: pd.DataFrame) -> str:
    global_metrics = tables["metrics_global.parquet"]
    monthly = tables["metrics_monthly.parquet"]
    weekly = tables["metrics_weekly.parquet"]
    stability = tables["stability.parquet"]
    portfolios = tables["portfolio_summary.parquet"]
    lines = [
        "# Strict-R3 three-month trust-sizing ablations — 2025 OOF and frozen 2026 confirmation",
        "",
        "## Decision",
        "",
        "The Local Distribution Forest Proxy (LDF) support-shrinkage arm (legacy artifact ID `N5_drf_support_l110_meanrisk`) is the best tail-EV sizing overlay: it improves 2025 development and 2026 confirmation at top 1%, 2%, and 5%. The Bayesian GAM arm `G3_bayes_gam_cmi_l110_meanrisk` is the best constrained-portfolio risk overlay: it lowers matched 2026 maximum drawdown from about 82.1% to 71.4% while slightly improving portfolio net EV/trade. Neither is production-approved because drawdown remains severe and the worst 2026 top-5 month remains negative.",
        "",
        "The Bayesian arms are small positive sizing refinements. The GAM family overfits the 2025 sizing gain and loses it in 2026. NGBoost improves only modestly. The MLP arms fail the development selection score and do not advance.",
        "",
        "## Fixed experiment contract",
        "",
        "- Long side only; frozen strict-R3 final score and pooled-global candidate ranking.",
        "- Trust models change relative position size only; they cannot rerank or admit candidates.",
        "- Three-month train blocks and three-month held blocks; labels must be resolved before the held-block boundary.",
        "- 2025 is development OOF model selection. The top three arm names are frozen before 2026 is opened.",
        "- 'Frozen 2026 confirmation' is untouched for this trust-sizing funnel only. The upstream canonical strict-R3 stack predates this funnel and has separate 2026 research history.",
        "- Canonical SimplePolicyOptimiser outcome, including 100 bps cost exactly once.",
        "- Causal side-local hierarchical 21/42/84-day EV mapping; admission requires mapped net EV >= +50 bps.",
        "- Portfolio: eight concurrent, two new entries per bar, one position per asset, 80% margin, 7x leverage.",
        "- Raw Geometry/K9 cluster memberships are excluded because bundle meanings change. Only bundle-invariant entropy, top-two margin, OOD, drift, and support summaries are pooled.",
        "- Implementation fidelity: NGBoost is the actual NGBoost categorical classifier; LDF is a declared random-forest predictive-distribution proxy with local-support/parent shrinkage, not a specialized external distributional-random-forest package; the MLP is optimized directly under a bounded Student-t negative-log-likelihood; Bayesian GAM uncertainty uses Bayesian-ridge and conditional-scale components.",
        "",
        "## 2025 development selection — every arm",
        "",
        _md(selection[["pipeline", "development_shard", "arm", "weighted_tail_score", "mean_portability_top1_2_5", "worst_month_top1_2_5", "selection_score"]]),
        "",
        "## Global tail metrics for each pipeline's control and top three",
        "",
        "Values are exposure-weighted net bps/trade. Candidate membership is the frozen global ranking; therefore improvements are sizing improvements, not selection improvements.",
        "",
    ]
    for pipeline in SOURCES:
        for year in (2025, 2026):
            frame = global_metrics.loc[
                (global_metrics["pipeline"] == pipeline)
                & (global_metrics["year"] == year)
            ]
            pivot = frame.pivot(index="arm", columns="tail", values="exposure_weighted_net_bps").reset_index()
            pivot.columns = ["arm", *[f"top_{100*float(value):g}%" for value in pivot.columns[1:]]]
            lines.extend([f"### {pipeline} — {year}", "", _md(pivot), ""])
    lines.extend(["## Monthly top-1/2/5 metrics", ""])
    for pipeline in SOURCES:
        for year in (2025, 2026):
            frame = monthly.loc[
                (monthly["pipeline"] == pipeline)
                & (monthly["year"] == year)
                & monthly["tail"].isin(DISPLAY_TAILS)
            ]
            table = frame.pivot_table(
                index=["period", "arm"], columns="tail",
                values="exposure_weighted_net_bps", aggfunc="first",
            ).reset_index()
            table.columns = ["month", "arm", "top_1%", "top_2%", "top_5%"]
            role = "development OOF" if year == 2025 else "frozen confirmation"
            lines.extend([f"### {pipeline} — {year} ({role})", "", _md(table), ""])
    lines.extend(["## Stability by month", ""])
    stable = stability.loc[stability["tail"].isin(DISPLAY_TAILS), [
        "pipeline", "year", "arm", "tail", "portability", "month_median_bps",
        "month_mad_bps", "worst_month_bps", "positive_months", "months",
    ]]
    lines.extend([_md(stable), "", "## Stability by week", ""])
    lines.extend([_md(_weekly_summary(weekly)), ""])
    lines.extend([
        "The exact per-week, per-arm, per-tail rows are retained in `top3_weekly.parquet`; the table above is their compact stability summary.",
        "",
        "## 2026 causal admission plus portfolio constraints",
        "",
    ])
    p2026 = portfolios.loc[portfolios["year"] == 2026, [
        "pipeline", "arm", "accepted_trades", "trades_per_day",
        "gross_bps_per_trade", "net_bps_per_trade", "positive_rate",
        "max_drawdown",
    ]]
    lines.extend([_md(p2026), ""])
    lines.extend([
        "Wallet endpoints are intentionally not promoted: compounding 80% margin at 7x over thousands of overlapping research trades produces numerically explosive wallet paths. Net bps/trade, trades/day, and drawdown are the interpretable portfolio diagnostics here.",
        "",
        "## Interpretation",
        "",
        "1. Local Distribution Forest Proxy (LDF; legacy artifact ID `N5_drf_support_l110_meanrisk`) transports best for global tails: it raises top-1/2/5 weighted EV in both eras and improves the 2026 top-5 worst month slightly.",
        "2. `G3_bayes_gam_cmi_l110_meanrisk` is the portfolio-risk challenger: versus equal sizing in 2026 it raises constrained net EV from about 149.1 to 150.3 bps/trade and improves maximum drawdown from about 82.1% to 71.4%, despite losing about 1.2 bps at the globally ranked top 1%.",
        "3. These gains are still small relative to the underlying alpha and do not change candidate order. They are sizing overlays, not new alpha layers.",
        "4. The 2026 top-5 tail still has one negative month and only 23/31 positive weeks. The overlays are reliable at top 1%, less so deeper in the tail.",
        "5. Portfolio drawdown remains about 71-79% for the best challengers. This fails production risk acceptance despite positive per-trade EV.",
        "6. No raw K9/archetype ID crosses bundle boundaries. A future cluster-conditioned ablation must either freeze one geometry bundle or train/evaluate only inside identical bundle hashes.",
        "",
        "## Artifact map",
        "",
        "The comparison bundle contains full global, monthly, weekly, stability, fold, CMI-edge, causal-admission portfolio, and manifest tables. The nonlinear NGBoost and MLP development shards are kept separately so negative results remain auditable.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    tables = _collect()
    selection = _selection_rows()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    output_names = {
        "metrics_global.parquet": "top3_global.parquet",
        "metrics_monthly.parquet": "top3_monthly.parquet",
        "metrics_weekly.parquet": "top3_weekly.parquet",
        "stability.parquet": "top3_stability.parquet",
        "portfolio_summary.parquet": "top3_portfolio_summary.parquet",
        "portfolio_monthly.parquet": "top3_portfolio_monthly.parquet",
        "portfolio_weekly.parquet": "top3_portfolio_weekly.parquet",
        "fold_audit.parquet": "top3_fold_audit.parquet",
        "cmi_edge_audit.parquet": "top3_cmi_edge_audit.parquet",
    }
    for source_name, output_name in output_names.items():
        tables[source_name].to_parquet(args.out_dir / output_name, index=False)
    selection.to_parquet(args.out_dir / "all_2025_selection.parquet", index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_report(tables, selection) + "\n")
    manifest = {
        "schema": "strict_r3_three_pipeline_trust_sizing_report_v1",
        "selection_era": "2025 development OOF",
        "confirmation_era": "2026 frozen; not used for selection",
        "source_directories": {
            pipeline: {str(year): str(path) for year, path in years.items()}
            for pipeline, years in SOURCES.items()
        },
        "nonlinear_development_shards": {
            name: str(path) for name, path in NONLINEAR_DEVELOPMENT_SHARDS.items()
        },
        "report": str(args.report),
        "report_sha256": _sha(args.report),
        "raw_k9_memberships_used": False,
        "ranking_changed": False,
        "trust_changes_size_only": True,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
