#!/usr/bin/env python3
"""Train/inner local robustness audit for the contextual sizing winner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_constrained import ConstrainedReplaySpec  # noqa: E402
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    FOLDS, _load_deployed_side_params, _load_or_build_path_cache, _write_json,
)
from scripts.run_simple_policy_1m_constrained_search import INNER_FOLDS, ExperimentData, _indices_between  # noqa: E402
from scripts.run_simple_policy_1m_contextual_ablation import (  # noqa: E402
    _bar_neutral_sizes, _bayesian_sizes, _load_atr, _load_context, _score, _weighted_evaluate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rich-ledger", required=True)
    parser.add_argument("--posterior-state", required=True)
    parser.add_argument("--deployed-parent-summary", required=True)
    parser.add_argument("--path-cache-dir", required=True)
    parser.add_argument("--atr-audit", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--perturbations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    rows = pd.read_parquet(args.candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, _, _ = _load_context(rows, Path(args.rich_ledger), Path(args.posterior_state))
    atr = _load_atr(rows, Path(args.atr_audit))
    deployed, _ = _load_deployed_side_params(Path(args.deployed_parent_summary))
    spec = ConstrainedReplaySpec()
    open0, high, low, close, valid, _ = _load_or_build_path_cache(
        rows, store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"),
        cache_dir=Path(args.path_cache_dir), spec=spec, rebuild=False,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)
    params = json.loads((result_dir / "parent_params.json").read_text(encoding="utf-8"))
    selections = json.loads((result_dir / "selections.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    rows_out = []
    ones = np.ones(len(rows), dtype=np.float64)
    for fold in FOLDS:
        inner = INNER_FOLDS[fold["fold"]]
        search_idx = _indices_between(data, fold["train_start"], inner["search_end"])
        inner_idx = _indices_between(data, inner["inner_start"], inner["inner_end"])
        parent = params[f"{fold['fold']}__parent"]["inner_selection_parent"]
        search_out = _score(data, search_idx, parent, ones)[2]
        inner_out = _score(data, inner_idx, parent, ones)[2]
        inner_parent_metric = _weighted_evaluate(data, inner_idx, inner_out, ones)
        selected_strength = float(selections[fold["fold"]]["size"]["strength"])
        selected_ood = float(selections[fold["fold"]]["size"]["ood_weight"])
        for perturbation in range(int(args.perturbations)):
            strength = float(np.clip(selected_strength * np.exp(rng.normal(0.0, 0.10)), 0.5, 6.0))
            ood_weight = float(np.clip(selected_ood + rng.normal(0.0, 0.15), 0.0, 1.0))
            sizes, _ = _bayesian_sizes(
                data, search_idx, inner_idx, search_out, context,
                strength=strength, ood_weight=ood_weight,
            )
            sizes = _bar_neutral_sizes(data, inner_idx, inner_out, sizes)
            metric = _weighted_evaluate(data, inner_idx, inner_out, sizes)
            rows_out.append({
                "fold": fold["fold"], "perturbation": perturbation, "strength": strength,
                "ood_weight": ood_weight, "objective": metric["objective"],
                "objective_delta_vs_parent": metric["objective"] - inner_parent_metric["objective"],
                "pnl": metric["net_pnl_bankroll"],
                "pnl_delta_vs_parent": metric["net_pnl_bankroll"] - inner_parent_metric["net_pnl_bankroll"],
                "exposure_ratio": metric["oos_exposure_ratio"],
            })
    frame = pd.DataFrame(rows_out)
    frame.to_csv(result_dir / "size_winner_inner_local_perturbations.csv", index=False)
    summary = {
        "scope": "inner validation only; outer OOS was not reused",
        "perturbations_per_fold": int(args.perturbations),
        "median_objective": float(frame.objective.median()),
        "worst_objective": float(frame.objective.min()),
        "positive_objective_fraction": float((frame.objective > 0.0).mean()),
        "median_objective_delta_vs_parent": float(frame.objective_delta_vs_parent.median()),
        "worst_objective_delta_vs_parent": float(frame.objective_delta_vs_parent.min()),
        "positive_delta_fraction": float((frame.objective_delta_vs_parent > 0.0).mean()),
        "median_pnl": float(frame.pnl.median()),
        "worst_pnl": float(frame.pnl.min()),
        "max_abs_exposure_error": float(np.max(np.abs(frame.exposure_ratio - 1.0))),
        "per_fold": frame.groupby("fold").agg(
            median_objective=("objective", "median"), worst_objective=("objective", "min"),
            median_objective_delta=("objective_delta_vs_parent", "median"),
            worst_objective_delta=("objective_delta_vs_parent", "min"),
            median_pnl=("pnl", "median"), worst_pnl=("pnl", "min"),
        ).reset_index().to_dict(orient="records"),
    }
    _write_json(result_dir / "size_winner_inner_local_robustness.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
