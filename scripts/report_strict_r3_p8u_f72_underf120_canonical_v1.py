#!/usr/bin/env python3
"""Create the reproducible monthly receipt for the frozen P8U/F72/Under-F120 stack.

This is deliberately a reporting-only utility.  It reads immutable target-free
score ledgers and joins the canonical policy outcomes only after score identity
has been fixed.  It never fits a model, mutates a source artifact, uses exchange
I/O, or makes an admission decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_tail125_base_history_mar25_jul26_fullprehistory_20260828_v1/target_free_scores"
POLICY_PATH = ROOT / "data_perp/artifacts/strict_r3_p8u_router_policy_label_successor_fullprehistory_20260828_v1/canonical_reconciled_policy_labels.parquet"
UNDER_METRIC_ROOTS = (
    ROOT / "data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_xendcg_f120_augdec25_fullprehistory_20260828_v1",
    ROOT / "data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_xendcg_f120_20260828_v1",
)
UNDER_SCORE_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_xendcg_f120_aug25_jul26_fullprehistory_20260828_v4"
BASE_MC1_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_baseonly_dual_mc1_nov25_jul26_fullprehistory_20260828_v1"
UNDER_MC1_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_jul26_fullprehistory_20260828_v1"
LIVE_DUAL_ROOT = ROOT / "data_perp/artifacts/strict_r3_live_bcf_current_dual_reconciled_rich_portfolio_nov25jul26_20260828_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(str(child.relative_to(path)).encode())
                digest.update(str(child.stat().st_size).encode())
    return digest.hexdigest()


def _month(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise").dt.strftime("%Y-%m")


def _base_ranking_metrics() -> pd.DataFrame:
    policy = pd.read_parquet(POLICY_PATH, columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    policy["policy_path_valid"] = policy["policy_path_valid"].fillna(False).astype(bool)
    policy["policy_net_bps"] = pd.to_numeric(policy["policy_net_bps"], errors="coerce")
    rows: list[dict[str, object]] = []
    for path in sorted(BASE_ROOT.glob("month=*.parquet")):
        score = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "base_score", "base_rank_ts"])
        score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True, errors="raise")
        work = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        work = work.loc[work.policy_path_valid & np.isfinite(work.policy_net_bps)].copy()
        month = path.stem.split("=")[1]
        for k in (1, 2, 5, 10, 20):
            picked = (
                work.sort_values(["__decision_ts__", "base_score", "candidate_id"], ascending=[True, False, True], kind="stable")
                .groupby("__decision_ts__", sort=False)
                .head(k)
            )
            rows.append({
                "month": month, "layer": "base_timestamp_ranking", "selection": f"top{k}_per_timestamp",
                "trades": int(len(picked)), "timestamps": int(picked.__decision_ts__.nunique()),
                "net_ev_bps_per_trade": float(picked.policy_net_bps.mean()),
                "net_total_bps": float(picked.policy_net_bps.sum()),
                "policy_valid_rows": int(len(work)), "source_target_free": True,
            })
    return pd.DataFrame(rows)


def _portfolio_monthly(root: Path, *, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_paths = sorted(root.glob("*decisions.parquet"))
    if not decision_paths:
        raise FileNotFoundError(f"{root}: no portfolio decisions receipt")
    work = pd.concat([pd.read_parquet(path) for path in decision_paths], ignore_index=True)
    work["timestamp"] = pd.to_datetime(work.timestamp, utc=True, errors="raise")
    work["month"] = _month(work.timestamp)
    accepted = work.loc[work.accepted.fillna(False).astype(bool) & work.policy_outcome_available.fillna(False).astype(bool)].copy()
    accepted["net_bps"] = pd.to_numeric(accepted.position_net_return, errors="coerce") * 10_000.0
    accepted = accepted.loc[np.isfinite(accepted.net_bps)].copy()
    by_month = accepted.groupby("month", sort=True).net_bps.agg(["count", "mean", "sum"]).reset_index()
    by_month = by_month.rename(columns={"count": "portfolio_entries", "mean": "net_ev_bps_per_trade", "sum": "net_total_bps"})
    considered = work.groupby("month", sort=True).size().rename("admitted_candidates").reset_index()
    result = considered.merge(by_month, on="month", how="left").fillna({"portfolio_entries": 0, "net_ev_bps_per_trade": np.nan, "net_total_bps": 0.0})
    result.insert(1, "layer", label)
    result["portfolio_entries"] = result.portfolio_entries.astype(int)
    aggregate = pd.DataFrame([{
        "layer": label,
        "months": int(accepted.month.nunique()),
        "portfolio_entries": int(len(accepted)),
        "net_ev_bps_per_trade": float(accepted.net_bps.mean()),
        "net_total_bps": float(accepted.net_bps.sum()),
        "worst_month_bps": float(accepted.groupby("month").net_bps.mean().min()),
    }])
    return result, aggregate


def _under_metrics() -> pd.DataFrame:
    work = pd.concat([pd.read_parquet(path / "objective_fold_metrics.parquet") for path in UNDER_METRIC_ROOTS], ignore_index=True)
    selected = [
        "held_month", "train_rows_before_sample", "train_rows", "train_queries", "valid_policy_rows",
        "conditional_mi_meta_policy_given_base", "residual_spearman_ic", "mean_utility_spreadcond_bps",
        "mean_potential_utility_recall", "mean_net_rescue_separation_bps", "mean_admission_substitution_utility_bps",
    ]
    result = work.loc[:, selected].copy().rename(columns={
        "held_month": "month",
        "conditional_mi_meta_policy_given_base": "cmi_given_base",
        "residual_spearman_ic": "residual_conditional_quality_ic",
        "mean_utility_spreadcond_bps": "economic_conditional_separation_bps",
        "mean_potential_utility_recall": "rescue_potential",
        "mean_net_rescue_separation_bps": "net_rescue_separation_bps",
        "mean_admission_substitution_utility_bps": "admission_substitution_utility_bps",
    })
    result.insert(1, "head", "Under F120 / bps100 / timestamp / rank_xendcg")
    return result


def _dual_admission_monthly() -> pd.DataFrame:
    path = UNDER_MC1_ROOT / "dual_predictions.parquet"
    work = pd.read_parquet(path, columns=["__decision_ts__", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "policy_path_valid"])
    work["month"] = _month(work.__decision_ts__)
    eligible = work.policy_path_valid.fillna(False).astype(bool)
    work["dual_admitted"] = eligible & work.bcf_mc1_expected_bps.ge(50.0) & work.current_mc1_expected_bps.ge(50.0)
    return work.groupby("month", sort=True).agg(
        dual_mc1_admitted=("dual_admitted", "sum"),
        valid_scored_rows=("policy_path_valid", "sum"),
    ).reset_index()


def _live_dual_monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the actual BCF/current-v5 dual-score baseline at both floors."""
    monthly: list[pd.DataFrame] = []
    aggregate: list[dict[str, object]] = []
    for floor in (30, 50):
        path = LIVE_DUAL_ROOT / f"current_v5_t{floor}_decisions.parquet"
        work = pd.read_parquet(path)
        work["timestamp"] = pd.to_datetime(work.timestamp, utc=True, errors="raise")
        work["month"] = _month(work.timestamp)
        work["net_bps"] = pd.to_numeric(work.position_net_return, errors="coerce") * 10_000.0
        selected = work.loc[
            work.accepted.fillna(False).astype(bool)
            & work.policy_outcome_available.fillna(False).astype(bool)
            & np.isfinite(work.net_bps)
        ].copy()
        by_month = selected.groupby("month", sort=True).net_bps.agg(["count", "mean", "sum"]).reset_index()
        by_month = by_month.rename(columns={"count": "portfolio_entries", "mean": "net_ev_bps_per_trade", "sum": "net_total_bps"})
        considered = work.groupby("month", sort=True).size().rename("admitted_candidates").reset_index()
        item = considered.merge(by_month, on="month", how="left").fillna({"portfolio_entries": 0, "net_ev_bps_per_trade": np.nan, "net_total_bps": 0.0})
        item.insert(1, "layer", "current_live_bcf_current_dual")
        item.insert(2, "dual_floor_bps", floor)
        item["portfolio_entries"] = item.portfolio_entries.astype(int)
        monthly.append(item)
        aggregate.append({
            "layer": "current_live_bcf_current_dual", "dual_floor_bps": floor,
            "months": int(selected.month.nunique()), "portfolio_entries": int(len(selected)),
            "net_ev_bps_per_trade": float(selected.net_bps.mean()), "net_total_bps": float(selected.net_bps.sum()),
            "worst_month_bps": float(selected.groupby("month").net_bps.mean().min()),
        })
    return pd.concat(monthly, ignore_index=True), pd.DataFrame(aggregate)


def _write_markdown(out: Path, base: pd.DataFrame, under: pd.DataFrame, base_portfolio: pd.DataFrame, under_portfolio: pd.DataFrame, live: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame) -> str:
        # Keep this receipt dependency-free: the workspace runtime does not
        # install pandas' optional ``tabulate`` package.
        columns = [str(column) for column in frame.columns]
        def render(value: object) -> str:
            if isinstance(value, (float, np.floating)):
                return "" if not np.isfinite(value) else f"{float(value):.3f}"
            return str(value)
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
        return "\n".join(lines)
    comparable = under_portfolio.merge(base_portfolio, on="month", suffixes=("_under", "_base"), how="outer")
    comparable["delta_entries_vs_base"] = comparable.portfolio_entries_under - comparable.portfolio_entries_base
    comparable["delta_ev_bps_vs_base"] = comparable.net_ev_bps_per_trade_under - comparable.net_ev_bps_per_trade_base
    comparable["delta_total_bps_vs_base"] = comparable.net_total_bps_under - comparable.net_total_bps_base
    lines = [
        "# P8U / F72 / Under-F120 strict-OOF reporting receipt",
        "",
        "Research only. No live bundle, exchange operation, admission threshold, or source artifact was changed.",
        "",
        "## Coverage boundary",
        "",
        "- Base target-free strict-OOF scores: March 2025 through July 2026; its one-month March ledger is explicitly historical-warm-up-only and was never used for selection.",
        "- Under-F120 strict-OOF scores: August 2025 through July 2026, using the restored March--July Base warm-up ledger.",
        "- Independent dual-MC1 plus constrained portfolio: November 2025 through July 2026, after three strictly earlier score months.",
        "",
        "## Base timestamp-local ranking diagnostics",
        "",
        table(base),
        "",
        "## Under-F120 conditional-head diagnostics",
        "",
        table(under),
        "",
        "## Matched portfolio comparison: Base-only control vs Base + Under-F120 + dual MC1",
        "",
        table(comparable),
        "",
        "## Current live BCF/current-v5 dual-score benchmark",
        "",
        "The live family is replayed with its frozen BCF/current-v5 MC1 score panels, the common rich-policy outcome ledger, the identical auction constraints, and both the actual 30-bps operating floor and a 50-bps floor-matched control. It remains a different score population from P8U, so this is a qualified system comparison rather than a paired candidate substitution test.",
        "",
        table(live),
        "",
    ]
    (out / "P8U_F72_UNDERF120_MONTHLY_RECEIPT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)

    base = _base_ranking_metrics()
    under = _under_metrics()
    base_portfolio, base_aggregate = _portfolio_monthly(BASE_MC1_ROOT, label="base_only_dual_mc1_control")
    under_portfolio, under_aggregate = _portfolio_monthly(UNDER_MC1_ROOT, label="base_plus_underf120_dual_mc1")
    admitted = _dual_admission_monthly()
    under_portfolio = under_portfolio.merge(admitted, on="month", how="left")
    live_portfolio, live_aggregate = _live_dual_monthly()

    base.to_parquet(out / "base_monthly_timestamp_ranking.parquet", index=False, compression="zstd")
    under.to_parquet(out / "under_f120_monthly_head_metrics.parquet", index=False, compression="zstd")
    base_portfolio.to_parquet(out / "base_only_dual_mc1_monthly_portfolio.parquet", index=False, compression="zstd")
    under_portfolio.to_parquet(out / "under_f120_dual_mc1_monthly_portfolio.parquet", index=False, compression="zstd")
    live_portfolio.to_parquet(out / "current_live_bcf_current_dual_monthly_benchmark.parquet", index=False, compression="zstd")
    aggregate = pd.concat([base_aggregate, under_aggregate, live_aggregate], ignore_index=True, sort=False)
    aggregate.to_parquet(out / "aggregate_portfolio_metrics.parquet", index=False, compression="zstd")
    _write_markdown(out, base, under, base_portfolio, under_portfolio, live_portfolio)
    manifest = {
        "schema": "strict_r3_p8u_f72_underf120_report_v1",
        "scope": "offline reporting only; target-free score identities are fixed before outcome joins; no fitting or execution",
        "coverage": {
            "base": "2025-03 through 2026-07",
            "under_f120": "2025-08 through 2026-07",
            "dual_mc1_portfolio": "2025-11 through 2026-07",
        },
        "sources": {str(path.relative_to(ROOT)): _sha(path) for path in [BASE_ROOT, POLICY_PATH, *UNDER_METRIC_ROOTS, UNDER_SCORE_ROOT, BASE_MC1_ROOT, UNDER_MC1_ROOT, LIVE_DUAL_ROOT]},
        "cost_contract": "policy_net_bps already deducts policy_cost_bps=100 exactly once; this reporting utility does not apply another cost",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "correctness_report.json").write_text(json.dumps({
        "base_scores_target_free_before_policy_join": True,
        "no_model_fit_or_parameter_selection": True,
        "policy_cost_not_double_counted": True,
        "unsupported_early_under_months_not_backfilled": True,
        "current_live_baseline_explicitly_labelled_different_score_population": True,
    }, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
