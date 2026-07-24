#!/usr/bin/env python3
"""Matched OOS comparison of deployed, trailing-only, and Bayesian-sized trailing-only policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_constrained import FAMILY_TRAILING_ONLY, ConstrainedReplaySpec  # noqa: E402
from extreme_price_movements.simple_policy_1m_contextual import stable_fold_objective  # noqa: E402
from scripts.run_simple_policy_1m_capital_ablation import FOLDS, _load_deployed_side_params, _load_or_build_path_cache, _write_json  # noqa: E402
from scripts.run_simple_policy_1m_constrained_search import INNER_FOLDS, ExperimentData, _evaluate, _indices_between, _optimise  # noqa: E402
from scripts.run_simple_policy_1m_contextual_ablation import (  # noqa: E402
    _bar_neutral_sizes, _bayesian_sizes, _load_atr, _load_context, _weighted_evaluate,
)


def _metric(data: ExperimentData, idx: np.ndarray, outputs: Mapping[str, np.ndarray], size: np.ndarray | None = None) -> dict[str, Any]:
    metrics, _ = _evaluate(data, idx, outputs, family=FAMILY_TRAILING_ONLY)
    if size is not None:
        metrics.update(_weighted_evaluate(data, idx, outputs, size))
    else:
        metrics.update({
            "oos_exposure_ratio": 1.0,
            "exposure_normalized_objective": metrics["objective"],
            "exposure_normalized_pnl": metrics["net_pnl_bankroll"],
        })
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rich-ledger", required=True)
    parser.add_argument("--posterior-state", required=True)
    parser.add_argument("--deployed-parent-summary", required=True)
    parser.add_argument("--path-cache-dir", required=True)
    parser.add_argument("--atr-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-trials", type=int, default=32)
    parser.add_argument("--parent-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(args.candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, _, provenance = _load_context(rows, Path(args.rich_ledger), Path(args.posterior_state))
    atr = _load_atr(rows, Path(args.atr_audit))
    deployed, _ = _load_deployed_side_params(Path(args.deployed_parent_summary))
    spec = ConstrainedReplaySpec()
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows, store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"),
        cache_dir=Path(args.path_cache_dir), spec=spec, rebuild=False,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)
    ones = np.ones(len(rows), dtype=np.float64)
    result_rows: list[dict[str, Any]] = []
    params_out: dict[str, Any] = {}
    for fold_no, fold in enumerate(FOLDS, 1):
        inner = INNER_FOLDS[fold["fold"]]
        search_idx = _indices_between(data, fold["train_start"], inner["search_end"])
        inner_idx = _indices_between(data, inner["inner_start"], inner["inner_end"])
        train_idx = _indices_between(data, fold["train_start"], fold["train_end"])
        outer_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        seeds = [args.seed + fold_no * 10_000 + i * 1_000 for i in range(args.parent_seeds)]
        search_parent, search_diag = _optimise(
            data, search_idx, family=FAMILY_TRAILING_ONLY, joint=True,
            trials_per_seed=max(args.parent_trials, 24), seeds=[s + 555 for s in seeds], sampler_kind="tpe",
        )
        parent, parent_diag = _optimise(
            data, train_idx, family=FAMILY_TRAILING_ONLY, joint=True,
            trials_per_seed=max(args.parent_trials, 24), seeds=[s + 333 for s in seeds], sampler_kind="tpe",
        )
        params_out[fold["fold"]] = {
            "search_parent": search_parent, "search_optimizer": search_diag,
            "full_train_parent": parent, "full_train_optimizer": parent_diag,
        }

        deployed_outer = data.simulate_deployed(outer_idx)
        result_rows.append({"fold": fold["fold"], "policy": "current_deployed", **_metric(data, outer_idx, deployed_outer)})
        trailing_search = data.simulate(search_idx, search_parent, FAMILY_TRAILING_ONLY)
        trailing_inner = data.simulate(inner_idx, search_parent, FAMILY_TRAILING_ONLY)
        trailing_train = data.simulate(train_idx, parent, FAMILY_TRAILING_ONLY)
        trailing_outer = data.simulate(outer_idx, parent, FAMILY_TRAILING_ONLY)
        result_rows.append({"fold": fold["fold"], "policy": "joint_trailing_only", **_metric(data, outer_idx, trailing_outer)})

        candidates = []
        for strength in (1.5, 3.0, 4.5):
            for ood_weight in (0.0, 0.5, 1.0):
                sizes, _ = _bayesian_sizes(
                    data, search_idx, inner_idx, trailing_search, context,
                    strength=strength, ood_weight=ood_weight,
                )
                metrics = _weighted_evaluate(data, inner_idx, trailing_inner, sizes)
                candidates.append((metrics["objective"], strength, ood_weight))
        _, strength, ood_weight = max(candidates)
        sizes, size_state = _bayesian_sizes(
            data, train_idx, outer_idx, trailing_train, context,
            strength=strength, ood_weight=ood_weight,
        )
        result_rows.append({
            "fold": fold["fold"], "policy": "joint_trailing_plus_bayesian_raw",
            "size_strength": strength, "ood_weight": ood_weight,
            **_metric(data, outer_idx, trailing_outer, sizes),
        })
        neutral = _bar_neutral_sizes(data, outer_idx, trailing_outer, sizes)
        result_rows.append({
            "fold": fold["fold"], "policy": "joint_trailing_plus_bayesian_bar_neutral",
            "size_strength": strength, "ood_weight": ood_weight,
            **_metric(data, outer_idx, trailing_outer, neutral),
        })
        params_out[fold["fold"]]["sizing"] = {
            "strength": strength, "ood_weight": ood_weight, "state": size_state,
            "inner_grid": [{"objective": a, "strength": b, "ood_weight": c} for a, b, c in candidates],
        }
        pd.DataFrame(result_rows).to_csv(output / "fold_metrics.partial.csv", index=False)
        _write_json(output / "params.partial.json", params_out)

    fold_metrics = pd.DataFrame(result_rows)
    deployed_fold = fold_metrics[fold_metrics.policy == "current_deployed"].set_index("fold")
    summary_rows = []
    for policy, group in fold_metrics.groupby("policy", sort=False):
        objective = group.objective.to_numpy(dtype=float)
        neutral_objective = group.exposure_normalized_objective.to_numpy(dtype=float)
        delta = np.asarray([row.objective - deployed_fold.loc[row.fold, "objective"] for row in group.itertuples()])
        neutral_delta = np.asarray([row.exposure_normalized_objective - deployed_fold.loc[row.fold, "objective"] for row in group.itertuples()])
        summary_rows.append({
            "policy": policy, "folds": len(group),
            "stable_objective": stable_fold_objective(objective),
            "stable_exposure_normalized_objective": stable_fold_objective(neutral_objective),
            "mean_objective": float(objective.mean()),
            "mean_delta_vs_deployed": float(delta.mean()),
            "mean_neutral_delta_vs_deployed": float(neutral_delta.mean()),
            "positive_delta_folds": int(np.sum(delta > 0.0)),
            "positive_neutral_delta_folds": int(np.sum(neutral_delta > 0.0)),
            "mean_pnl": float(group.net_pnl_bankroll.mean()),
            "mean_exposure_normalized_pnl": float(group.exposure_normalized_pnl.mean()),
            "worst_fold_pnl": float(group.net_pnl_bankroll.min()),
            "worst_week": float(group.worst_week.min()),
            "worst_drawdown": float(group.max_drawdown.min()),
            "total_trades": int(group.n_trades.sum()),
            "mean_return_per_trade": float(group.mean_net_return.mean()),
            "hit_rate": float(group.hit_rate.mean()),
            "mean_exposure_ratio": float(group.oos_exposure_ratio.mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("stable_exposure_normalized_objective", ascending=False)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(output / "params.json", params_out)
    _write_json(output / "manifest.json", {
        "evidence_status": "nested policy-validation OOS; July excluded",
        "comparison": "identical rows, exact 1m paths, causal entry ATR, 1% fee, spread, 8-open/2-new capacity",
        "sizing_target": "positive net-after-cost outcome refit from each trailing-only train replay",
        "bar_neutral": "base-size-weighted normalization across size-independent capacity-admitted entries at each UTC timestamp",
        "provenance": provenance, "path": path_manifest,
    })
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
