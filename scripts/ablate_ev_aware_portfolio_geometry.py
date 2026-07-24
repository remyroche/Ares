#!/usr/bin/env python3
"""Ablate EV-aware exit geometry with final portfolio competition in-loop.

The canonical side/archetype geometry remains the center point. Optuna applies
small side-level multipliers, replays the exact executable paths, and evaluates
each trial through the frozen global auction. July is replay-only by default.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablate_simple_policy_exit_geometry import _load_bundles, _prepare_rows
from scripts.materialize_canonical_exit_policy_replay import (
    _json_safe,
    _load_ev_curve,
    _materialize_exit_rows,
    _portfolio_candidates,
)
from extreme_price_movements.portfolio_policy_replay import (
    load_portfolio_policy_params,
    replay_candidates,
)


def _parse_geometry(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _scaled_policy_tables(
    parent: pd.DataFrame,
    local: pd.DataFrame,
    scales: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_out = parent.copy()
    local_out = local.copy()
    scale_keys = {
        "sl_mult": "sl_scale",
        "trailing_activation_mult": "activation_scale",
        "giveback_beta": "giveback_scale",
        "capital_protect_mfe_mult": "protect_scale",
    }
    for idx, row in parent_out.iterrows():
        side = str(row.get("side") or ("short" if str(row.get("strategy_id", "")).startswith("short") else "long"))
        side_scales = scales[side]
        for param, scale_name in scale_keys.items():
            column = f"param_{param}"
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if np.isfinite(value):
                parent_out.at[idx, column] = float(value) * float(side_scales[scale_name])
        parent_out.at[idx, "param_giveback_beta"] = float(
            np.clip(float(parent_out.at[idx, "param_giveback_beta"]), 0.05, 1.0)
        )
        parent_out.at[idx, "param_capital_protect_spread_lock_mult"] = 1.5

    for idx, row in local_out.iterrows():
        side = str(row.get("side") or ("short" if str(row.get("strategy_id", "")).startswith("short") else "long"))
        side_scales = scales[side]
        geometry = _parse_geometry(row.get("shrinkage_final_geometry"))
        for param, scale_name in scale_keys.items():
            if param in geometry and np.isfinite(float(geometry[param])):
                geometry[param] = float(geometry[param]) * float(side_scales[scale_name])
        if "giveback_beta" in geometry:
            geometry["giveback_beta"] = float(np.clip(geometry["giveback_beta"], 0.05, 1.0))
        geometry["capital_protect_spread_lock_mult"] = 1.5
        local_out.at[idx, "shrinkage_final_geometry"] = repr(geometry)
    return parent_out, local_out


def _fold_masks(timestamps: pd.Series, *, fit_end: pd.Timestamp, n_folds: int) -> list[np.ndarray]:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    eligible = ts.lt(fit_end)
    days = np.asarray(sorted(ts.loc[eligible].dt.floor("D").dropna().unique()))
    chunks = [chunk for chunk in np.array_split(days, max(2, int(n_folds))) if len(chunk)]
    return [eligible.to_numpy() & ts.dt.floor("D").isin(chunk).to_numpy() for chunk in chunks]


def _evaluate_fold(
    rows: pd.DataFrame,
    *,
    portfolio_params: Any,
    ev_curve: dict[str, Any],
) -> dict[str, float]:
    if rows.empty:
        return {"score": -1.0, "preportfolio_ev": -1.0, "portfolio_ev": -1.0, "trades": 0}
    decisions, _equity, summary = replay_candidates(
        _portfolio_candidates(rows),
        portfolio_params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)]
    portfolio_ev = float(
        pd.to_numeric(accepted.get("position_net_return"), errors="coerce").mean()
    ) if len(accepted) else -1.0
    preportfolio_ev = float(pd.to_numeric(rows["net_return"], errors="coerce").mean())
    # Equal-unit objective: executable notional EV plus the EV of the rows that
    # survive actual global portfolio competition. Total PnL remains diagnostic.
    score = 0.5 * preportfolio_ev + 0.5 * portfolio_ev
    return {
        "score": float(score),
        "preportfolio_ev": preportfolio_ev,
        "portfolio_ev": portfolio_ev,
        "trades": int(len(accepted)),
        "portfolio_pnl": float(summary.get("net_pnl", 0.0)),
        "max_drawdown": float(summary.get("max_drawdown", np.nan)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--portfolio-config", type=Path, required=True)
    parser.add_argument("--portfolio-ev-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.01)
    parser.add_argument("--fit-end", default="2026-07-01T00:00:00Z")
    parser.add_argument("--holdout-end", default="2026-07-11T00:00:00Z")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _prepare_rows(
        args.candidates,
        min_rank=0.0,
        rank_score_col="rank_pct",
        rank_scope="per_strategy",
        apply_regime_ev_calibration_artifact=False,
    )
    bundles = _load_bundles(
        rows,
        data_root=str(args.data_root),
        market_mode="perps",
        path_len=int(args.path_len),
        min_rows_per_strategy=5,
    )
    parent = pd.read_csv(args.policy_dir / "side_parent_policy_summary.csv")
    local = pd.read_csv(args.policy_dir / "side_archetype_policy_summary.csv")
    portfolio_params = load_portfolio_policy_params(args.portfolio_config)
    ev_curve = _load_ev_curve(args.portfolio_ev_reference)
    fit_end = pd.Timestamp(args.fit_end)
    holdout_end = pd.Timestamp(args.holdout_end)
    trial_records: list[dict[str, Any]] = []
    baseline_rows = _materialize_exit_rows(
        bundles,
        parent_summary=parent,
        archetype_summary=local,
        cost_pct=float(args.round_trip_cost_pct) / 2.0,
    )
    baseline_masks = _fold_masks(
        baseline_rows["timestamp"], fit_end=fit_end, n_folds=args.folds
    )
    baseline_folds = [
        _evaluate_fold(
            baseline_rows.loc[mask],
            portfolio_params=portfolio_params,
            ev_curve=ev_curve,
        )
        for mask in baseline_masks
    ]

    def objective(trial: optuna.Trial) -> float:
        scales = {
            side: {
                "sl_scale": trial.suggest_float(f"{side}_sl_scale", 0.75, 1.25, step=0.05),
                "activation_scale": trial.suggest_float(f"{side}_activation_scale", 0.75, 1.25, step=0.05),
                "giveback_scale": trial.suggest_float(f"{side}_giveback_scale", 0.75, 1.25, step=0.05),
                "protect_scale": trial.suggest_float(f"{side}_protect_scale", 0.75, 1.25, step=0.05),
            }
            for side in ("long", "short")
        }
        trial_parent, trial_local = _scaled_policy_tables(parent, local, scales)
        exit_rows = _materialize_exit_rows(
            bundles,
            parent_summary=trial_parent,
            archetype_summary=trial_local,
            cost_pct=float(args.round_trip_cost_pct) / 2.0,
        )
        masks = _fold_masks(exit_rows["timestamp"], fit_end=fit_end, n_folds=args.folds)
        folds = [_evaluate_fold(exit_rows.loc[mask], portfolio_params=portfolio_params, ev_curve=ev_curve) for mask in masks]
        scores = np.asarray([fold["score"] for fold in folds], dtype=np.float64)
        for idx, fold in enumerate(folds):
            baseline_trades = max(int(baseline_folds[idx]["trades"]), 1)
            activity_ratio = float(fold["trades"] / baseline_trades)
            fold["activity_ratio_vs_baseline"] = activity_ratio
            # Avoid an apparent EV improvement obtained by collapsing the
            # globally competed book. The penalty starts below 80% activity.
            scores[idx] -= 0.02 * max(0.0, 0.80 - activity_ratio)
        stable = float(scores.mean() - 0.5 * scores.std(ddof=0) + 0.25 * scores.min())
        record = {
            "trial": int(trial.number),
            "objective": stable,
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std(ddof=0)),
            "worst_score": float(scores.min()),
            "scales": scales,
            "folds": folds,
        }
        trial_records.append(record)
        trial.set_user_attr("record", record)
        return stable

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(args.seed)))
    study.enqueue_trial(
        {
            f"{side}_{name}": 1.0
            for side in ("long", "short")
            for name in ("sl_scale", "activation_scale", "giveback_scale", "protect_scale")
        }
    )
    study.optimize(objective, n_trials=int(args.trials), show_progress_bar=False)
    best_record = dict(study.best_trial.user_attrs["record"])
    best_parent, best_local = _scaled_policy_tables(parent, local, best_record["scales"])
    best_rows = _materialize_exit_rows(
        bundles,
        parent_summary=best_parent,
        archetype_summary=best_local,
        cost_pct=float(args.round_trip_cost_pct) / 2.0,
    )
    ts = pd.to_datetime(best_rows["timestamp"], utc=True, errors="coerce")
    fit_rows = best_rows.loc[ts.lt(fit_end)].copy()
    holdout_rows = best_rows.loc[ts.ge(fit_end) & ts.lt(holdout_end)].copy()
    fit_metrics = _evaluate_fold(fit_rows, portfolio_params=portfolio_params, ev_curve=ev_curve)
    holdout_metrics = _evaluate_fold(holdout_rows, portfolio_params=portfolio_params, ev_curve=ev_curve)
    best_rows.to_parquet(args.output_dir / "best_exit_rows.parquet", index=False, compression="zstd")
    best_parent.to_csv(args.output_dir / "best_side_parent_policy_summary.csv", index=False)
    best_local.to_csv(args.output_dir / "best_side_archetype_policy_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "trial": row["trial"],
                "objective": row["objective"],
                "mean_score": row["mean_score"],
                "std_score": row["std_score"],
                "worst_score": row["worst_score"],
                **{f"{side}_{key}": value for side, vals in row["scales"].items() for key, value in vals.items()},
            }
            for row in trial_records
        ]
    ).to_csv(args.output_dir / "trials.csv", index=False)
    manifest = {
        "schema": "ev_aware_portfolio_geometry_ablation_v1",
        "canonical_policy_unchanged": True,
        "objective": "stable(mean - 0.5*std + 0.25*worst) of 0.5*preportfolio_EV + 0.5*post_global_auction_EV",
        "fit_end": str(fit_end),
        "holdout_end": str(holdout_end),
        "trials": int(args.trials),
        "baseline_folds": baseline_folds,
        "best": best_record,
        "fit_metrics": fit_metrics,
        "holdout_metrics": holdout_metrics,
        "timeout_contract": "last executable close at horizon; never encoded as full loss",
        "capital_protection_contract": "gross locked return floor=max(policy floor, 1.5*asset expected full spread)",
        "cost_contract": "1% round trip once plus executable entry/exit half-spreads",
        "legacy_regime_ev_calibration_applied": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
