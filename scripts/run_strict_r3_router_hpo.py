#!/usr/bin/env python3
"""Pruned, strict-OOF HPO for the economic-recall router.

Each trial is evaluated month by month through the normal target-free router
producer.  The trial can be median-pruned only after a completed held month;
there is no shared held label, score panel, or post-date calibration between
trials.  Tree count is selected with the router's chronological inner reserve
and the final ranker is refit on all eligible rows of that outer fold.

This is research only.  It never writes inference bundles or touches live
state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "scripts" / "run_strict_r3_economic_recall_router.py"
SEED = 1729
DEFAULT_MONTHS = ("2025-11", "2026-03", "2026-07")
GAINS = "0,1,2,4,7,11"


def _fold_router_scores(paths: list[Path]) -> list[float]:
    """Return the declared primary Top-50 ``S_router`` per held fold.

    The active Router optimisation contract is deliberately fold-local rather
    than candidate-pooled.  It reads immutable strict-OOF timestamp receipts
    and uses p=.75/cap=225 utility with bounded sqrt timestamp weight.
    """
    frames = []
    for path in paths:
        frame = pd.read_parquet(path / "router_timestamp_metrics.parquet")
        frame = frame.loc[
            frame["score"].eq("router_primary_only_rank")
            & np.isclose(frame["route_fraction"].to_numpy(float), .50)
        ].copy()
        if frame.empty:
            raise AssertionError(f"missing primary-only Top-50 timestamp metrics in {path}")
        frames.append(frame)
    def score(frame: pd.DataFrame) -> float:
        available = frame["utility_sum"].to_numpy(float)
        positive = available[np.isfinite(available) & (available > 0)]
        if not len(positive):
            raise AssertionError("no positive utility support in strict OOF held fold")
        median_available = float(np.median(positive))
        weight = np.minimum(np.sqrt(np.maximum(available, 0.0) / median_available), 2.0)
        valid_utility = frame["timestamp_utility_recall"].notna().to_numpy()
        valid50 = frame["timestamp_recall_50bps"].notna().to_numpy()
        valid100 = frame["timestamp_recall_100bps"].notna().to_numpy()
        utility = float(np.average(frame.loc[valid_utility, "timestamp_utility_recall"], weights=weight[valid_utility]))
        recall50 = float(frame.loc[valid50, "timestamp_recall_50bps"].mean())
        recall100 = float(frame.loc[valid100, "timestamp_recall_100bps"].mean())
        return float(.70 * utility + .15 * recall50 + .15 * recall100)

    fold_scores = [score(frame) for frame in frames]
    if not fold_scores or not np.all(np.isfinite(fold_scores)):
        raise AssertionError("non-finite held-fold router score")
    return fold_scores


def _stability_score(fold_scores: list[float]) -> float:
    """Frozen selector: .65 mean + .25 Q25 + .10 worst held fold."""
    values = np.asarray(fold_scores, dtype=float)
    if len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("at least three finite held-fold scores are required")
    return float(.65 * values.mean() + .25 * np.quantile(values, .25) + .10 * values.min())


def _trial_params(trial: optuna.Trial) -> dict[str, str]:
    depth = trial.suggest_int("max_depth", 3, 5)
    # Optuna requires one invariant distribution per parameter name across
    # all trials.  Sample the declared leaf domain once, then deterministically
    # project infeasible depth/leaf combinations to the largest valid value.
    requested_leaves = trial.suggest_categorical("num_leaves", [7, 15, 31])
    max_leaves = min(31, 2 ** depth - 1)
    leaves = max(value for value in (7, 15, 31) if value <= min(requested_leaves, max_leaves))
    trial.set_user_attr("effective_num_leaves", leaves)
    return {
        "--n-estimators": "2000",
        "--learning-rate": str(trial.suggest_float("learning_rate", .02, .08, log=True)),
        "--max-depth": str(depth),
        "--num-leaves": str(leaves),
        "--min-child-fraction": str(trial.suggest_float("min_child_fraction", .004, .020, log=True)),
        "--min-child-floor": str(trial.suggest_int("min_child_floor", 200, 1000, step=100)),
        "--min-split-gain": str(trial.suggest_float("min_split_gain", 1e-4, .02, log=True)),
        "--feature-fraction": str(trial.suggest_float("feature_fraction", .70, .92)),
        "--subsample": str(trial.suggest_float("subsample", .70, .92)),
        "--l1": str(trial.suggest_float("l1", 1e-4, 5.0, log=True)),
        "--l2": str(trial.suggest_float("l2", .1, 30.0, log=True)),
        "--max-bin": str(trial.suggest_categorical("max_bin", [63, 127])),
    }


def _run_trial(args: argparse.Namespace, trial: optuna.Trial) -> float:
    params = _trial_params(trial)
    receipt_paths: list[Path] = []
    for step, month in enumerate(args.months):
        out = args.out / "trials" / f"trial={trial.number:03d}" / f"month={month}"
        command = [
            sys.executable, str(ROUTER), "--out", str(out), "--months", month,
            "--primary-target", args.primary_target, "--primary-only",
            "--train-months", str(args.train_months), "--reserve-days", str(args.reserve_days),
            "--train-cap", str(args.train_cap), "--n-jobs", str(args.n_jobs),
            "--route-fractions", "0.30,0.40,0.50",
            "--objective", args.objective, "--truncation", str(args.truncation),
            "--label-gains", args.label_gains,
            "--row-weight-scheme", args.row_weight_scheme,
            "--early-stopping-rounds", "30",
            "--inner-validation-fraction", ".20",
        ]
        for root in args.feature_root:
            command.extend(("--feature-root", str(root)))
        for option, value in (
            ("--aux-root", args.aux_root),
            ("--policy-path", args.policy_path),
            ("--bundle", args.bundle),
            ("--full-feature-contract", args.full_feature_contract),
        ):
            if value is not None:
                command.extend((option, str(value)))
        for key, value in params.items():
            command.extend((key, value))
        subprocess.run(command, cwd=ROOT, check=True)
        receipt_paths.append(out)
        fold_scores = _fold_router_scores(receipt_paths)
        # The predeclared objective is defined across all three outer folds.
        # Do not prune on a partial-fold surrogate.
        if step == len(args.months) - 1:
            trial.report(_stability_score(fold_scores), step)
        if step == len(args.months) - 1 and trial.should_prune():
            trial.set_user_attr("fold_scores", fold_scores)
            raise optuna.TrialPruned()
    fold_scores = _fold_router_scores(receipt_paths)
    trial.set_user_attr("fold_scores", fold_scores)
    return _stability_score(fold_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary-target", default="U50_p075_c225")
    parser.add_argument("--row-weight-scheme", default="sqrt_excess")
    parser.add_argument("--objective", choices=("lambdarank", "rank_xendcg"), default="rank_xendcg")
    parser.add_argument("--label-gains", default=GAINS)
    parser.add_argument("--feature-root", type=Path, action="append", default=[])
    parser.add_argument("--aux-root", type=Path)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--full-feature-contract", type=Path)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=240_000)
    parser.add_argument("--truncation", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--startup-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED, help="deterministic sampler seed; permits isolated HPO shards")
    args = parser.parse_args()
    args.months = tuple(token.strip() for token in args.months.split(",") if token.strip())
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite immutable HPO root: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.startup_trials, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: _run_trial(args, trial), n_trials=args.trials, n_jobs=1, gc_after_trial=True)
    rows = []
    for trial in study.trials:
        rows.append({
            "trial": trial.number, "state": trial.state.name, "value": trial.value,
            "params_json": json.dumps(trial.params, sort_keys=True),
            "fold_scores_json": json.dumps(trial.user_attrs.get("fold_scores", [])),
        })
    pd.DataFrame(rows).to_parquet(args.out / "trial_results.parquet", index=False, compression="zstd")
    (args.out / "manifest.json").write_text(json.dumps({
        "schema": "strict_r3_router_hpo_v1", "scope": "research-only; no live mutation",
        "primary_target": args.primary_target, "months": args.months,
        "selection": "0.65*mean(fold S_router) + 0.25*Q25(fold S_router) + 0.10*worst(fold S_router)",
        "score": "S_router=0.70*Top50 utility recall + 0.15*Top50 count recall>50 + 0.15*Top50 count recall>100; timestamp-local, utility=max(policy_net-50,0) clipped at 225 and powered by .75",
        "objective": args.objective,
        "label_gains": [float(value.strip()) for value in args.label_gains.split(",") if value.strip()],
        "row_weight_scheme": args.row_weight_scheme,
        "feature_roots": [str(root) for root in args.feature_root],
        "full_feature_contract": str(args.full_feature_contract) if args.full_feature_contract else None,
        "truncation_requested": args.truncation,
        "early_stopping": "30 rounds on latest 20% chronological training queries; final refit on all eligible training rows",
        "pruner": {"kind": "MedianPruner", "startup_trials": args.startup_trials, "warmup_steps": 1},
        "sampler_seed": args.seed,
        "best_trial": study.best_trial.number, "best_value": study.best_value, "best_params": study.best_trial.params,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
