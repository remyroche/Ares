#!/usr/bin/env python3
"""Generate pooled, side, fold and monthly metrics for the two-year TP6/SL4/H12 run.

The report deliberately keeps the ledger window separate from the causal OOF
windows.  The ledger is the complete 24-month valid population; base OOF starts
after its first training fold and the residual OOF starts after fresh base OOF
history is available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "data_perp/artifacts/tp6_sl4_h12_two_year_ledger_20260806_v1"
BASE_DIR = ROOT / "data_perp/artifacts/tp6_sl4_h12_two_year_base_oof_20260806_v1"
META_DIR = ROOT / "data_perp/artifacts/tp6_sl4_h12_two_year_specialists_meta_20260806_v2"
OUT = ROOT / "data_perp/artifacts/tp6_sl4_h12_two_year_report_20260806_v1"
TAILS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40)


def _rank_ic(frame: pd.DataFrame, score: str, target: str = "net_bps") -> float:
    if len(frame) < 3 or frame[score].nunique(dropna=True) < 2 or frame[target].nunique(dropna=True) < 2:
        return float("nan")
    return float(frame[score].rank(method="average").corr(frame[target].rank(method="average")))


def _tail(frame: pd.DataFrame, score: str, tail: float, dataset: str, arm: str,
          period: str, scope: str = "global", key: str = "all") -> dict[str, object]:
    frame = frame[np.isfinite(frame[score]) & np.isfinite(frame["net_bps"])].copy()
    n = max(1, int(np.ceil(len(frame) * tail))) if len(frame) else 0
    top = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n) if n else frame
    return {
        "dataset": dataset,
        "arm": arm,
        "period": period,
        "scope": scope,
        "key": key,
        "tail": tail,
        "rows": int(len(top)),
        "pool_rows": int(len(frame)),
        "gross_bps": float(top["gross_bps"].mean()) if len(top) else float("nan"),
        "net_bps": float(top["net_bps"].mean()) if len(top) else float("nan"),
        "long_share": float(top["side_name"].eq("long").mean()) if len(top) else float("nan"),
        "rank_ic": _rank_ic(frame, score),
    }


def _normalize_base(base: pd.DataFrame) -> pd.DataFrame:
    return base.rename(columns={"exact_gross_bps": "gross_bps", "exact_net_bps": "net_bps"}).copy()


def _add_metrics(out: list[dict[str, object]], frame: pd.DataFrame, dataset: str,
                 arm: str, period: str, *, include_sides: bool = True) -> None:
    for tail in TAILS:
        out.append(_tail(frame, arm, tail, dataset, arm, period))
        if include_sides:
            # Side metrics are computed on the globally selected tail, not on
            # separate side-local tails, matching the deployed top-k policy.
            n = max(1, int(np.ceil(len(frame) * tail))) if len(frame) else 0
            ranked = frame.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable")
            top = ranked.head(n)
            for side, z in top.groupby("side_name", sort=True):
                full_side = frame[frame.side_name.eq(side)]
                out.append({
                    "dataset": dataset,
                    "arm": arm,
                    "period": period,
                    "scope": "global_tail_side",
                    "key": str(side),
                    "tail": tail,
                    "rows": int(len(z)),
                    "pool_rows": int(len(full_side)),
                    "gross_bps": float(z.gross_bps.mean()),
                    "net_bps": float(z.net_bps.mean()),
                    "long_share": 1.0 if side == "long" else 0.0,
                    "rank_ic": _rank_ic(full_side, arm),
                })


def _monthly_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    x = metrics[(metrics.scope == "global") & metrics.period.str.match(r"^\d{4}-\d{2}$")].copy()
    if x.empty:
        return x
    rows = []
    for (dataset, arm, tail), g in x.groupby(["dataset", "arm", "tail"], sort=True):
        vals = g.net_bps.dropna()
        rows.append({
            "dataset": dataset, "arm": arm, "tail": tail, "months": int(vals.size),
            "mean_month_net_bps": float(vals.mean()), "median_month_net_bps": float(vals.median()),
            "std_month_net_bps": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "min_month_net_bps": float(vals.min()), "max_month_net_bps": float(vals.max()),
            "positive_months": int((vals > 0).sum()),
            "positive_month_fraction": float((vals > 0).mean()) if vals.size else float("nan"),
            "worst_month": str(g.loc[g.net_bps.idxmin(), "period"]),
            "best_month": str(g.loc[g.net_bps.idxmax(), "period"]),
        })
    return pd.DataFrame(rows)


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    ledger_manifest = json.loads((LEDGER_DIR / "manifest.json").read_text())
    base_manifest = json.loads((BASE_DIR / "manifest.json").read_text())
    meta_manifest = json.loads((META_DIR / "manifest.json").read_text())
    availability = pd.read_parquet(LEDGER_DIR / "feature_availability.parquet")
    audit = pd.read_parquet(LEDGER_DIR / "monthly_population_audit.parquet")
    base = _normalize_base(pd.read_parquet(BASE_DIR / "base_oof_predictions.parquet"))
    base["__ts__"] = pd.to_datetime(base["__ts__"], utc=True)
    meta = pd.read_parquet(META_DIR / "oof_predictions.parquet")
    meta["__ts__"] = pd.to_datetime(meta["__ts__"], utc=True)
    # The base prediction is the matched comparator on exactly the residual
    # rows.  All three scores are ranked globally across both sides.
    matched = meta.merge(base[["candidate_id", "base_expected_net_bps"]], on="candidate_id", how="left", validate="one_to_one")
    if matched["base_expected_net_bps"].isna().any():
        raise AssertionError("residual rows missing freshly retrained base output")
    matched["base_only"] = matched["base_expected_net_bps"]
    matched["meta_no_specialists"] = matched["baseline_score_bps"]
    matched["meta_with_specialists"] = matched["augmented_score_bps"]
    metrics: list[dict[str, object]] = []
    # Complete base OOF and the matched residual OOF are separate datasets.
    for arm in ["base_expected_net_bps"]:
        _add_metrics(metrics, base, "base_oof", arm, "pooled", include_sides=True)
        for month, g in base.groupby(base.__ts__.dt.strftime("%Y-%m"), sort=True):
            _add_metrics(metrics, g, "base_oof", arm, month, include_sides=True)
    for arm in ["base_only", "meta_no_specialists", "meta_with_specialists"]:
        _add_metrics(metrics, matched, "matched_residual_oof", arm, "pooled", include_sides=True)
        for month, g in matched.groupby(matched.__ts__.dt.strftime("%Y-%m"), sort=True):
            _add_metrics(metrics, g, "matched_residual_oof", arm, month, include_sides=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_parquet(out / "two_year_tail_metrics.parquet", index=False, compression="zstd")
    summary = _monthly_summary(metrics_df)
    summary.to_parquet(out / "two_year_monthly_summary.parquet", index=False, compression="zstd")
    matched.to_parquet(out / "matched_residual_predictions.parquet", index=False, compression="zstd")

    provenance = json.loads((BASE_DIR / "base_fold_provenance.json").read_text())
    selected = pd.read_parquet(META_DIR / "top20_selection.parquet")
    selected_counts = selected.groupby(["fold", "side_name"]).size().to_dict() if not selected.empty else {}
    usable = availability[availability.usable_90pct_nonconstant.astype(bool)]
    report: list[str] = []
    report.append("# Two-year TP6/SL4/H12 retraining audit")
    report.append("")
    report.append("## Scope and contract")
    report.append("")
    report.append(f"- Unified valid ledger: **{ledger_manifest['window_start_utc'][:10]} through 2024-08-31** (24 calendar months).")
    report.append(f"- Ledger rows: **{ledger_manifest['rows']:,}** ({ledger_manifest['rows_by_side']['long']:,} long / {ledger_manifest['rows_by_side']['short']:,} short).")
    report.append("- Label: authoritative R3 three-state target on exact TP6/SL4 path, H12 horizon, 100 bps cost; `label_available_ts = decision_ts + 12h`.")
    report.append(f"- Numeric feature columns: {ledger_manifest['feature_count_numeric']:,}; usable at >=90% coverage and nonconstant: **{ledger_manifest['feature_count_usable_90pct_nonconstant']:,}**.")
    report.append("- Invalid/incomplete label rows are excluded from the supervised ledger; the population audit remains available separately. The materialized ledger asserts every included row is valid and cost-complete.")
    report.append("")
    report.append("## Training and OOF windows")
    report.append("")
    report.append("The 24 months are the available training/evaluation population. Strict OOF starts after causal warm-up, so the OOF metrics do not falsely claim predictions for the first training-only months.")
    report.append("")
    report.append("| Base fold | Training rows | Training window | Test window | Test rows | Features |")
    report.append("|---|---:|---|---|---:|---:|")
    for p in provenance:
        report.append(f"| {p['fold']} {p['side']} | {p['train_rows']:,} | {p['train_start'][:10]}–{p['train_end'][:10]} | {p['test_start'][:10]}–{p['test_end'][:10]} | {p['test_rows']:,} | {p['feature_count']} |")
    report.append("")
    report.append(f"- Base OOF rows: **{len(base):,}**, from {base.__ts__.min().date()} through {base.__ts__.max().date()}.")
    report.append(f"- Residual/meta OOF rows: **{len(meta):,}**, from {meta.__ts__.min().date()} through {meta.__ts__.max().date()}.")
    report.append(f"- Residual history: {meta_manifest['history_window_days']:.0f} days, strict prior label-resolved rows; all same-side base OOF rows are handed to the residual model per row.")
    report.append(f"- Residual context: {meta_manifest['causal_context_candidate_count']} causal context candidates, full five-state/transition/geometry/continuous contract; top-20 selection counts by fold/side: {selected_counts}.")
    report.append("")

    def pooled_table(dataset: str, arms: list[str]) -> None:
        report.append(f"## Pooled {dataset} tails")
        report.append("")
        report.append("| Arm | Top 1% net/gross | Top 5% net/gross | Top 10% net/gross | Top 20% net/gross | Rank IC | Rows |")
        report.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm in arms:
            q = metrics_df[(metrics_df.dataset == dataset) & (metrics_df.arm == arm) & (metrics_df.period == "pooled") & (metrics_df.scope == "global")].sort_values("tail")
            def cell(t):
                z=q[q["tail"].eq(t)].iloc[0]
                return f"{z.net_bps:+.2f}/{z.gross_bps:+.2f}"
            rank=float(q.rank_ic.iloc[0]) if len(q) else float("nan")
            rows=int(q.pool_rows.iloc[0]) if len(q) else 0
            report.append(f"| {arm} | {cell(.01)} | {cell(.05)} | {cell(.10)} | {cell(.20)} | {rank:+.3f} | {rows:,} |")
        report.append("")
    pooled_table("base_oof", ["base_expected_net_bps"])
    pooled_table("matched_residual_oof", ["base_only", "meta_no_specialists", "meta_with_specialists"])

    report.append("## Per-side contribution inside the global tail")
    report.append("")
    report.append("These are the long/short rows actually selected by the pooled global ranking. They are not independently ranked side-local tails.")
    report.append("")
    report.append("| Dataset | Arm | Tail | Long rows/net | Short rows/net |")
    report.append("|---|---|---:|---:|---:|")
    for dataset, arms in (("base_oof", ["base_expected_net_bps"]), ("matched_residual_oof", ["base_only", "meta_no_specialists", "meta_with_specialists"])):
        for arm in arms:
            q = metrics_df[(metrics_df.dataset == dataset) & (metrics_df.arm == arm) & (metrics_df.period == "pooled") & (metrics_df.scope == "global_tail_side")]
            for tail in (.01, .05, .10, .20):
                z = q[q["tail"].eq(tail)].set_index("key")
                def side_cell(side: str) -> str:
                    if side not in z.index:
                        return "0 / n/a"
                    return f"{int(z.loc[side, 'rows'])} / {z.loc[side, 'net_bps']:+.2f}"
                report.append(f"| {dataset} | {arm} | {tail:.0%} | {side_cell('long')} | {side_cell('short')} |")
    report.append("")

    report.append("## Matched residual OOF uplift")
    report.append("")
    for tail in (.01, .05, .10, .20):
        q = metrics_df[(metrics_df.dataset == "matched_residual_oof") & (metrics_df.period == "pooled") & (metrics_df.scope == "global") & metrics_df["tail"].eq(tail)].set_index("arm")
        if {"base_only", "meta_no_specialists", "meta_with_specialists"}.issubset(q.index):
            report.append(f"- Top {tail:.0%}: base-only {q.loc['base_only','net_bps']:+.2f} bps; meta without specialists {q.loc['meta_no_specialists','net_bps']:+.2f} ({q.loc['meta_no_specialists','net_bps']-q.loc['base_only','net_bps']:+.2f}); with specialists {q.loc['meta_with_specialists','net_bps']:+.2f} ({q.loc['meta_with_specialists','net_bps']-q.loc['base_only','net_bps']:+.2f}).")
    report.append("")

    report.append("## Month-to-month stability")
    report.append("")
    report.append("The summary below is over monthly global top-k rankings; it is not a per-timestamp or side-local tail. `positive_months` counts months whose selected global tail had positive mean net bps.")
    report.append("")
    report.append("| Dataset | Arm | Tail | Months | Mean | Median | Std | Worst | Best | Positive months |")
    report.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.sort_values(["dataset", "arm", "tail"]).iterrows():
        if r["tail"] not in (.01, .05, .10, .20):
            continue
        report.append(f"| {r['dataset']} | {r['arm']} | {r['tail']:.0%} | {int(r['months'])} | {r['mean_month_net_bps']:+.2f} | {r['median_month_net_bps']:+.2f} | {r['std_month_net_bps']:.2f} | {r['min_month_net_bps']:+.2f} ({r['worst_month']}) | {r['max_month_net_bps']:+.2f} ({r['best_month']}) | {int(r['positive_months'])}/{int(r['months'])} |")
    report.append("")

    report.append("## Interpretation and limitations")
    report.append("")
    report.append("- `base_only` is the fresh side-local base expected-net map on exactly the residual OOF rows.")
    report.append("- `meta_no_specialists` is the residual model using its baseline feature contract; `meta_with_specialists` adds the selected leaf/regime representations. Both are ranked globally after the side-local class-to-bps conversion.")
    report.append("- These are exact H12 net labels, not a trailing-profit replay. Gross is net plus the fixed 100 bps cost, so positive net requires clearing that cost floor.")
    report.append("- The ledger spans 24 months, but strict OOF metrics begin in March 2023 for the base and September 2023 for the residual stage. The first months are therefore training/substrate coverage, not claimed OOF performance.")
    report.append("- No HPO or post-test tuning was run in this report; selected base features are the frozen Stage-I side-local lists (87 long / 34 short), and the residual selection is nested within each outer fold.")
    report.append("")
    report.append("## Artifacts")
    report.append("")
    report.append("- `ledger.parquet`, `feature_availability.parquet`, and `monthly_population_audit.parquet` in the two-year ledger directory.")
    report.append("- `base_oof_predictions.parquet`, `base_metrics.parquet`, and `base_fold_provenance.json` in the base directory.")
    report.append("- `oof_predictions.parquet`, `top20_selection.parquet`, `selection_candidate_audit.parquet`, `target_horizon_audit.parquet`, and `meta_comparison_metrics.parquet` in the residual directory.")
    (out / "TWO_YEAR_TP6_SL4_H12_REPORT.md").write_text("\n".join(report) + "\n")
    manifest = {
        "status": "COMPLETED",
        "ledger_manifest": str(LEDGER_DIR / "manifest.json"),
        "base_manifest": str(BASE_DIR / "manifest.json"),
        "meta_manifest": str(META_DIR / "manifest.json"),
        "ledger_rows": int(len(audit)),
        "base_oof_rows": int(len(base)),
        "matched_residual_oof_rows": int(len(matched)),
        "usable_feature_count": int(len(usable)),
        "outputs": ["two_year_tail_metrics.parquet", "two_year_monthly_summary.parquet", "matched_residual_predictions.parquet", "TWO_YEAR_TP6_SL4_H12_REPORT.md"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    print(run(args.out))
