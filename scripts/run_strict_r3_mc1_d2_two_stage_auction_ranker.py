#!/usr/bin/env python3
"""MC1 admission plus chronological LambdaRank auction ablation, offline only.

Frozen strict MC1_d2 expected net remains the exclusive +50-bps admission
authority.  The second stage sees only target-free stack fields plus the
prequential Huber-arcsine output and ranks candidates *after* that admission
gate.  HPO is confined to chronological 2025 folds; the final ranker is fitted
on resolved 2025 rows and evaluated on 2026 without retraining.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    _candidate_table,
    _metrics,
    _params as portfolio_params,
    CAUSAL_AUCTION_CURVE,
)
from scripts.run_strict_r3_mc1_admission_ablation_v2 import metric_row  # noqa: E402
from extreme_price_movements.portfolio_policy_replay import replay_candidates  # noqa: E402


SEED = 1729
FEATURES = (
    "frozen_final_score", "huber_expected_bps", "base_rank42",
    "conditional_consensus_rank", "upstream", "ordinary_shadow_consensus_rank",
    "correctness_rank",
)
EDGES = np.asarray((-np.inf, -200.0, -50.0, 50.0, 150.0, 250.0, np.inf))
FOLDS = tuple(pd.Timestamp(x, tz="UTC") for x in ("2025-04-01", "2025-07-01", "2025-10-01"))


def _matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = train.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    return (
        train.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").fillna(medians),
        test.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").fillna(medians),
    )


def hpo_params(trial: optuna.Trial) -> dict[str, float | int | list[int]]:
    gain_name = trial.suggest_categorical("label_gain", ("moderate", "tail"))
    return {
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", .02, .08),
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_samples": trial.suggest_int("min_child_samples", 150, 1200),
        "feature_fraction": trial.suggest_float("feature_fraction", .70, .95),
        "subsample": trial.suggest_float("subsample", .70, .95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", .1, 30.0, log=True),
        "lambdarank_truncation_level": trial.suggest_categorical("truncation", (3, 5, 8, 12)),
        "label_gain": [0, 1, 2, 5, 10, 20] if gain_name == "tail" else [0, 1, 2, 4, 7, 12],
        "gain_name": gain_name,
    }


def _fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, params: dict[str, float | int | list[int]], *, seed: int = SEED,
) -> np.ndarray:
    x_train, x_test = _matrix(train, test)
    labels = np.digitize(pd.to_numeric(train.policy_net_bps, errors="coerce"), EDGES[1:-1], right=True)
    order = train.assign(__label__=labels).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 2, 5],
        n_estimators=int(params["n_estimators"]), learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]), max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]), feature_fraction=float(params["feature_fraction"]),
        subsample=float(params["subsample"]), subsample_freq=1,
        reg_alpha=float(params["reg_alpha"]), reg_lambda=float(params["reg_lambda"]),
        lambdarank_truncation_level=int(params["lambdarank_truncation_level"]),
        label_gain=list(params["label_gain"]), random_state=seed, n_jobs=6, verbosity=-1,
    )
    model.fit(
        x_train.loc[order.index], order["__label__"],
        group=order.groupby("__decision_ts__", sort=False).size().to_numpy(int),
    )
    return np.asarray(model.predict(x_test), dtype=float)


def _quick_score(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    metric = metric_row(frame, "frozen_mc1_expected_bps", "ranker_score", admission_threshold_bps=50.0)
    values = np.asarray([metric["portfolio_net_ev_bps"], metric["worst_week_bps"]], dtype=float)
    if not np.isfinite(values).all():
        return -1e9, metric
    return float(metric["portfolio_net_ev_bps"] - max(0.0, -metric["worst_week_bps"])), metric


def _attach_outcomes(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        result = decisions.copy()
        result["policy_outcome_available"] = pd.Series(dtype=bool)
        return result
    lookup = candidates.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
    lookup.index.name = "candidate_index"
    return decisions.merge(lookup, on="candidate_index", how="left", validate="many_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--huber", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    control_cols = ["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps", *FEATURES[2:]]
    control = pd.read_parquet(args.control, columns=control_cols)
    control["__decision_ts__"] = pd.to_datetime(control["__decision_ts__"], utc=True)
    control = control.rename(columns={"final_score": "frozen_final_score", "mc1_expected_bps": "frozen_mc1_expected_bps"})
    huber = pd.read_parquet(args.huber, columns=["candidate_id", "mc1_expected_bps"])
    huber = huber.rename(columns={"mc1_expected_bps": "huber_expected_bps"})
    policy_cols = [
        "candidate_id", "__symbol__", "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        "policy_gross_bps", "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason",
    ]
    policy = pd.read_parquet(args.ledger, columns=policy_cols)
    data = control.merge(huber, on="candidate_id", how="inner", validate="one_to_one").merge(
        policy, on="candidate_id", how="inner", validate="one_to_one",
    )
    data["policy_label_available_ts"] = pd.to_datetime(data["policy_label_available_ts"], utc=True)
    data = data.loc[data.frozen_mc1_expected_bps.ge(50.0)].copy()
    trials: list[dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        params = hpo_params(trial)
        scores: list[float] = []
        for start in FOLDS:
            stop = start + pd.DateOffset(months=3)
            train = data.loc[
                data.__decision_ts__.lt(start)
                & data.policy_label_available_ts.lt(start)
                & data.policy_path_valid.fillna(False).astype(bool)
                & data.policy_net_bps.notna()
            ].copy()
            test = data.loc[data.__decision_ts__.between(start, stop, inclusive="left")].copy()
            if len(train) < 3_000 or test.empty:
                continue
            test["ranker_score"] = _fit_predict(train, test, params)
            score, metric = _quick_score(test)
            scores.append(score)
            trials.append({"trial": trial.number, "fold": start.isoformat(), "score": score, **metric, **params})
        if not scores:
            return -1e9
        ordered = np.sort(np.asarray(scores, dtype=float))
        median = float(np.median(ordered))
        return median - .5 * float(np.median(np.abs(ordered - median))) - max(0.0, -float(ordered.min()))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.trials)
    # Convert Optuna's winning primitive fields back to the model contract.
    winner = study.best_trial.params.copy()
    gain = winner.pop("label_gain")
    truncation = winner.pop("truncation")
    winner_params: dict[str, float | int | list[int]] = {
        **winner, "n_estimators": 500,
        "lambdarank_truncation_level": int(truncation),
        "label_gain": [0, 1, 2, 5, 10, 20] if gain == "tail" else [0, 1, 2, 4, 7, 12],
        "gain_name": gain,
    }
    train = data.loc[
        data.__decision_ts__.lt(pd.Timestamp("2026-01-01", tz="UTC"))
        & data.policy_label_available_ts.lt(pd.Timestamp("2026-01-01", tz="UTC"))
        & data.policy_path_valid.fillna(False).astype(bool)
        & data.policy_net_bps.notna()
    ].copy()
    test = data.loc[data.__decision_ts__.dt.year.eq(2026)].copy()
    test["ranker_score"] = _fit_predict(train, test, winner_params)
    candidates = _candidate_table(
        test.rename(columns={"frozen_mc1_expected_bps": "mc1_expected_bps", "ranker_score": "final_score"}),
        policy, 50.0,
    )
    decisions, equity, _ = replay_candidates(
        candidates, portfolio_params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perps", initial_wallet=1000.0,
    )
    decisions = _attach_outcomes(decisions, candidates)
    metric = _metrics(decisions, equity, "two_stage_huber_lambdarank", "2026")
    test.to_parquet(args.out_dir / "two_stage_predictions_2026.parquet", index=False, compression="zstd")
    pd.DataFrame(trials).to_parquet(args.out_dir / "hpo_trials_2025.parquet", index=False)
    (args.out_dir / "hpo_winner.json").write_text(json.dumps(winner_params, indent=2) + "\n")
    (args.out_dir / "portfolio_metrics_2026.json").write_text(json.dumps(metric, indent=2) + "\n")
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_mc1_d2_two_stage_auction_ranker_v1", "status": "complete",
        "admission": "frozen MC1_d2 expected policy net >= +50 bps",
        "ranker": "LambdaRank on Huber-arcsine output plus frozen target-free stack fields",
        "features": list(FEATURES), "target": "policy net ordinal six bins",
        "hpo": "2025 chronological Apr/Jul/Oct three-month folds only",
        "validation": "single 2026 fit-on-resolved-2025 evaluation; not untouched promotion evidence",
        "exclusions": ["live state", "exchange I/O", "admission authority changes"],
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "hpo_score": study.best_value, "metric": metric}), flush=True)


if __name__ == "__main__":
    main()
