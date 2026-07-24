#!/usr/bin/env python3
"""Add mechanism controls, detailed breakdowns, and final report to v2 search."""

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

from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    FOLDS,
    _load_deployed_side_params,
    _load_or_build_path_cache,
    _write_json,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    ExperimentData,
    _evaluate,
    _indices_between,
    _markdown_table,
    _summary,
)


def _breakdowns(ledger: pd.DataFrame, arm: str) -> pd.DataFrame:
    work = ledger.copy()
    rank = pd.to_numeric(work["rank_pct"], errors="coerce").fillna(0.9).clip(0.0, 1.0)
    work["position_size"] = 0.075 + 0.075 * np.power(rank, 1.1)
    work["pnl"] = pd.to_numeric(work["net_return"], errors="coerce") * work["position_size"]
    ts = pd.to_datetime(work["timestamp"], utc=True)
    work["week"] = ts.dt.tz_localize(None).dt.to_period("W").astype(str)
    work["month"] = ts.dt.strftime("%Y-%m")
    groups = [
        ("overall", []), ("side", ["side_name"]), ("archetype", ["policy_archetype"]),
        ("side_x_archetype", ["side_name", "policy_archetype"]), ("week", ["week"]),
    ]
    rows = []
    for slice_name, columns in groups:
        iterator = [((), work)] if not columns else work.groupby(columns, dropna=False)
        for keys, group in iterator:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {"family": arm, "slice": slice_name}
            row.update({column: value for column, value in zip(columns, keys)})
            row.update(
                {
                    "n_trades": int(len(group)), "net_pnl_bankroll": float(group["pnl"].sum()),
                    "mean_net_return": float(group["net_return"].mean()),
                    "hit_rate": float((group["net_return"] > 0.0).mean()),
                    "mean_holding_hours": float((group["exit_bars"] + 1).mean() / 60.0),
                    "full_sl_rate": float((group["reason"] == 1).mean()),
                    "capital_rate": float((group["reason"] == 2).mean()),
                    "trailing_rate": float((group["reason"] == 3).mean()),
                    "timeout_rate": float((group["reason"] == 0).mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--deployed-parent-summary", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--path-cache-dir", required=True)
    parser.add_argument("--download-dir", required=True)
    args = parser.parse_args()
    out = Path(args.experiment_dir)
    rows = pd.read_parquet(args.candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["timestamp", "symbol", "side", "rank_pct"]).copy()
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    deployed, _ = _load_deployed_side_params(Path(args.deployed_parent_summary))
    atr_audit = pd.read_parquet(out / "causal_entry_atr_audit.parquet")
    atr = np.full(len(rows), np.nan)
    ok = atr_audit[atr_audit["status"] == "ok"]
    atr[ok["row"].to_numpy(dtype=int)] = ok["effective_atr_fraction"].to_numpy(dtype=float)
    spec = ConstrainedReplaySpec()
    open0, high, low, close, valid, _manifest = _load_or_build_path_cache(
        rows, store_root=Path(args.store_root), cache_dir=Path(args.path_cache_dir), spec=spec, rebuild=False,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)

    folds = pd.read_csv(out / "nested_oos_fold_metrics.csv")
    fixed_rows = []
    for fold in FOLDS:
        idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        outputs = data.simulate(idx, deployed, FAMILY_TRAILING_ONLY)
        metrics, _selected = _evaluate(data, idx, outputs, family=FAMILY_TRAILING_ONLY)
        fixed_rows.append({"stage": "baseline", "family": "fixed_deployed_trailing_no_capital", "fold": fold["fold"], **metrics})
    folds = folds[~((folds["stage"] == "baseline") & (folds["family"] == "fixed_deployed_trailing_no_capital"))]
    folds = pd.concat([folds, pd.DataFrame(fixed_rows)], ignore_index=True)
    folds.to_csv(out / "nested_oos_fold_metrics.csv", index=False)
    summary = _summary(folds)
    summary.to_csv(out / "nested_oos_summary.csv", index=False)

    july_idx = _indices_between(data, "2026-07-01", "2026-07-11")
    fixed_outputs = data.simulate(july_idx, deployed, FAMILY_TRAILING_ONLY)
    fixed_metrics, fixed_selected = _evaluate(data, july_idx, fixed_outputs, family=FAMILY_TRAILING_ONLY)
    july = pd.read_csv(out / "july_post_selection_diagnostic.csv")
    july = july[july["family"] != "fixed_deployed_trailing_no_capital"]
    july = pd.concat([july, pd.DataFrame([{"family": "fixed_deployed_trailing_no_capital", **fixed_metrics}])], ignore_index=True)
    july.to_csv(out / "july_post_selection_diagnostic.csv", index=False)

    detailed = []
    for path in sorted(out.glob("july_selected_*.parquet")):
        name = path.stem.removeprefix("july_selected_")
        detailed.append(_breakdowns(pd.read_parquet(path), name))
    fixed_ledger = rows.iloc[july_idx].reset_index(drop=True).copy()
    for key, values in fixed_outputs.items():
        fixed_ledger[key] = values
    detailed.append(_breakdowns(fixed_ledger.loc[fixed_selected].copy(), "fixed_deployed_trailing_no_capital"))
    if detailed:
        pd.concat(detailed, ignore_index=True).to_csv(out / "july_detailed_breakdowns.csv", index=False)

    worker_manifests = [json.loads(path.read_text()) for path in sorted(Path(args.download_dir).glob("worker_*.json"))]
    download_summary = {
        "workers": len(worker_manifests),
        "symbols": int(sum(m["summary"]["ok_symbols"] for m in worker_manifests)),
        "required_minutes": int(sum(m["summary"]["required_minutes"] for m in worker_manifests)),
        "covered_minutes": int(sum(m["summary"]["covered_minutes"] for m in worker_manifests)),
        "fetched_rows": int(sum(m["summary"]["fetched_rows"] for m in worker_manifests)),
        "failed_symbols": int(sum(m["summary"]["failed_symbols"] for m in worker_manifests)),
        "incomplete_symbols": int(sum(m["summary"]["incomplete_symbols"] for m in worker_manifests)),
        "warmup_minutes": 2880,
    }
    download_summary["coverage"] = download_summary["covered_minutes"] / download_summary["required_minutes"]
    _write_json(Path(args.download_dir) / "manifest.json", download_summary)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["download_summary"] = download_summary
    manifest["search"]["optimizer_trials"] = 6336
    manifest["search"]["local_perturbation_replays"] = 192
    manifest["mechanism_control"] = "fixed deployed SL/trailing/adverse geometry with capital disabled"
    manifest.setdefault("outputs", {})["july_detailed_breakdowns"] = "july_detailed_breakdowns.csv"
    _write_json(manifest_path, manifest)

    deployed_row = summary[(summary.stage == "baseline") & (summary.family == "deployed_policy")].iloc[0]
    fixed_row = summary[(summary.stage == "baseline") & (summary.family == "fixed_deployed_trailing_no_capital")].iloc[0]
    stage1 = summary[summary.stage == "stage1_frozen_trailing"].copy()
    stage1["delta_stable_vs_deployed"] = stage1["stable_fold_objective"] - deployed_row["stable_fold_objective"]
    stage1["delta_stable_vs_fixed_no_cap"] = stage1["stable_fold_objective"] - fixed_row["stable_fold_objective"]
    summary_report = pd.concat([
        summary[summary.stage == "baseline"], summary[(summary.stage == "joint") & (summary.family == "trailing_only")], stage1,
    ], ignore_index=True)
    report = [
        "# Constrained 1-minute capital/trailing search", "",
        "## Main comparison", "",
        _markdown_table(summary_report, ["stage", "family", "folds", "stable_fold_objective", "mean_pnl", "worst_fold_pnl", "worst_week", "worst_drawdown", "positive_fold_fraction", "capital_before_trailing_rate", "initial_capital_active_rate", "ordering_violation_rate", "mean_pretrail_protected_minutes", "total_trades"]),
        "", "## Stage-1 capital attribution", "",
        _markdown_table(stage1, ["family", "stable_fold_objective", "delta_stable_vs_deployed", "delta_stable_vs_fixed_no_cap", "mean_pnl", "worst_week", "worst_drawdown", "capital_before_trailing_rate", "mean_pretrail_protected_minutes"]),
        "", "## July post-selection diagnostic", "",
        _markdown_table(july.sort_values("net_pnl_bankroll", ascending=False), ["family", "net_pnl_bankroll", "worst_week", "max_drawdown", "n_trades", "hit_rate", "capital_protect_rate", "trailing_rate", "capital_before_trailing_rate", "mean_pretrail_protected_minutes", "local_median_objective", "local_worst_objective", "local_positive_fraction"]),
        "", "## Contract and evidence", "",
        f"- Warm-up/data coverage: {download_summary['covered_minutes']:,}/{download_summary['required_minutes']:,} required symbol-minutes ({download_summary['coverage']:.2%}) across {download_summary['symbols']} symbols.",
        "- ATR is causal and entry-frozen: 48 completed pre-entry hours aggregated from 1m bars; the overlapping decision hour is excluded.",
        "- All constrained families structurally enforce immediate capital protection, capital-before-trailing, and a non-loosening handover.",
        "- Search breadth: 6,336 optimizer trials across three seeds plus 192 local perturbation replays.",
        "- Stage-1 rows are policy-selection OOS with trailing frozen. Joint rows exist only for the two families advanced independently inside each fold and should not be compared as three-fold family estimates.",
        "- July is post-selection diagnostic only because it was previously inspected; it is not an untouched promotion test.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
