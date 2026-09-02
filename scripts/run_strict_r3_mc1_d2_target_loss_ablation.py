#!/usr/bin/env python3
"""MC1_d2-only target/loss ablation on the original full-universe contract.

Static MC1 prediction is varied; all other semantics are frozen: six original
inputs, full-universe day-balanced history, monthly chronological fits, daily
prior-resolved residual shift, +50-bps admission, and final-score-only auction.
2025 selects HPO winners.  2026 is only reported after those choices freeze.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_mc1_d2_historical_parity import (  # noqa: E402
    CORE, SEED, day_balanced, utc,
)


ORDINAL_EDGES = np.asarray((-np.inf, -200.0, -50.0, 50.0, 150.0, 250.0, np.inf))
ORDINAL_CENTRES = np.asarray((-300.0, -125.0, 0.0, 100.0, 200.0, 350.0))
FOLDS = tuple(pd.Timestamp(value, tz="UTC") for value in ("2025-04-01", "2025-07-01", "2025-10-01"))


def _target(train: pd.DataFrame, kind: str) -> tuple[np.ndarray, dict[str, float]]:
    y = pd.to_numeric(train.policy_net_bps, errors="coerce").to_numpy(float)
    low, high = np.nanquantile(y, [.02, .98])
    clipped = np.clip(y, low, high)
    if kind == "ordinal6":
        return np.digitize(clipped, ORDINAL_EDGES[1:-1], right=True), {"low": low, "high": high}
    if kind.endswith("asin"):
        scale = max(abs(low), abs(high), 250.0)
        return np.arcsin(np.clip(clipped / scale, -.999, .999)), {"low": low, "high": high, "scale": scale}
    return clipped, {"low": low, "high": high}


def _fit_predict(
    train: pd.DataFrame, current: pd.DataFrame, kind: str, params: Mapping[str, float | int], seed: int,
) -> np.ndarray:
    medians = train.loc[:, CORE].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    x = train.loc[:, CORE].apply(pd.to_numeric, errors="coerce").fillna(medians)
    z = current.loc[:, CORE].apply(pd.to_numeric, errors="coerce").fillna(medians)
    y, transform = _target(train, kind)
    common: dict[str, float | int | str] = {
        "n_estimators": int(params["n_estimators"]), "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]), "max_depth": int(params["max_depth"]),
        "min_child_samples": int(params["min_child_samples"]),
        "feature_fraction": float(params["feature_fraction"]),
        "bagging_fraction": float(params["bagging_fraction"]), "bagging_freq": 1,
        "reg_alpha": float(params["reg_alpha"]), "reg_lambda": float(params["reg_lambda"]),
        "verbosity": -1, "random_state": seed, "n_jobs": 6,
    }
    if kind == "ordinal6":
        model = lgb.LGBMClassifier(objective="multiclass", num_class=6, **common).fit(x, y)
        return model.predict_proba(z).dot(ORDINAL_CENTRES)
    objective = "regression_l1" if kind.startswith("l1") else "huber"
    model = lgb.LGBMRegressor(objective=objective, **common).fit(x, y)
    pred = np.asarray(model.predict(z), dtype=float)
    return np.sin(pred) * float(transform["scale"]) if kind.endswith("asin") else pred


def _quick_portfolio(frame: pd.DataFrame) -> dict[str, float]:
    """Deterministic HPO proxy; full engine is used after HPO is frozen."""
    work = frame.loc[pd.to_numeric(frame.mapper_expected_bps, errors="coerce").ge(50.0)].copy()
    work = work.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    active: list[pd.Timestamp] = []
    accepted: list[bool] = []
    for decision, group in work.groupby("__decision_ts__", sort=True):
        active = [value for value in active if value > decision]
        taken = 0
        for row in group.itertuples(index=False):
            bar = pd.to_numeric(pd.Series([getattr(row, "policy_exit_bar_15m")]), errors="coerce").iloc[0]
            bars = max(48.0, float(bar) if np.isfinite(bar) else 48.0)
            yes = taken < 2 and len(active) < 8
            accepted.append(yes)
            if yes:
                active.append(decision + pd.Timedelta(minutes=15 * bars))
                taken += 1
    work["accepted"] = accepted
    realised = work.loc[
        work.accepted & work.policy_path_valid.fillna(False).astype(bool) & work.policy_net_bps.notna()
    ].copy()
    if realised.empty:
        return {"net_ev_bps": -1e6, "net_sum_bps": -1e9, "worst_week_bps": -1e6, "rows": 0.0}
    weekly = realised.groupby(realised.__decision_ts__.dt.strftime("%G-W%V")).policy_net_bps.mean()
    return {
        "net_ev_bps": float(realised.policy_net_bps.mean()), "net_sum_bps": float(realised.policy_net_bps.sum()),
        "worst_week_bps": float(weekly.min()), "rows": float(len(realised)),
    }


def _sample(train: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    return train.sample(cap, random_state=seed) if len(train) > cap else train


def _suggest(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "n_estimators": 500, "learning_rate": trial.suggest_float("learning_rate", .02, .08),
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_samples": trial.suggest_int("min_child_samples", 150, 1200),
        "feature_fraction": trial.suggest_float("feature_fraction", .70, .95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", .70, .95),
        "reg_alpha": trial.suggest_float("reg_alpha", .001, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", .1, 30.0, log=True),
    }


def _run_hpo(data: pd.DataFrame, source: pd.DataFrame, shifts: pd.Series, kind: str, trials: int, cap: int) -> tuple[dict[str, float | int], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        fold_metrics: list[dict[str, float]] = []
        for start in FOLDS:
            train = source.loc[
                source.policy_label_available_ts.lt(start) & source.policy_path_valid.fillna(False) & source.policy_net_bps.notna()
            ]
            test = data.loc[data.__decision_ts__.between(start, start + pd.offsets.MonthBegin(1), inclusive="left")].copy()
            if len(train) < 5_000 or test.empty:
                continue
            test["mapper_expected_bps"] = _fit_predict(_sample(train, cap, SEED + trial.number), test, kind, params, SEED + trial.number)
            test["mapper_expected_bps"] += test.day.map(shifts).to_numpy(float)
            fold_metrics.append(_quick_portfolio(test))
        if not fold_metrics:
            return -1e9
        values = np.asarray([item["net_ev_bps"] for item in fold_metrics], dtype=float)
        worst = min(item["worst_week_bps"] for item in fold_metrics)
        score = float(np.median(values) - .5 * np.median(np.abs(values - np.median(values))) - max(0.0, -worst))
        rows.append({"trial": trial.number, "kind": kind, "score": score, "median_fold_ev_bps": float(np.median(values)), "worst_fold_week_bps": worst, **params})
        return score
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    return {key: value for key, value in study.best_params.items()} | {"n_estimators": 500}, pd.DataFrame(rows)


def _predict_full(data: pd.DataFrame, source: pd.DataFrame, shifts: pd.Series, kind: str, params: Mapping[str, float | int]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for start in pd.date_range("2025-01-01", "2026-07-01", freq="MS", tz="UTC"):
        stop = start + pd.offsets.MonthBegin(1)
        train = source.loc[
            source.policy_label_available_ts.lt(start) & source.policy_path_valid.fillna(False) & source.policy_net_bps.notna()
        ]
        test = data.loc[data.__decision_ts__.between(start, stop, inclusive="left")].copy()
        if len(train) < 5_000 or test.empty:
            continue
        test["static_expected_bps"] = _fit_predict(_sample(train, 50_000, SEED), test, kind, params, SEED)
        test["recent_shift_bps"] = test.day.map(shifts).to_numpy(float)
        test["mc1_expected_bps"] = test.static_expected_bps + test.recent_shift_bps
        test["mapper_kind"] = kind
        test["fold_start"] = start
        pieces.append(test.loc[:, [
            "candidate_id", "__decision_ts__", "__symbol__", "policy_label_available_ts",
            "policy_path_valid", "policy_net_bps", "policy_exit_bar_15m", "final_score",
            "static_expected_bps", "recent_shift_bps", "mc1_expected_bps", "mapper_kind", "fold_start",
        ]])
        del train, test
        gc.collect()
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--parity-predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hpo-trials", type=int, default=6)
    parser.add_argument("--hpo-train-cap", type=int, default=80_000)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "policy_label_available_ts",
        "policy_path_valid", "policy_net_bps", "policy_exit_bar_15m", *CORE,
    ]
    data = pd.read_parquet(args.ledger, columns=columns)
    data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True)
    data["policy_label_available_ts"] = pd.to_datetime(data["policy_label_available_ts"], utc=True)
    if not data.side_name.astype(str).str.lower().eq("long").all():
        raise ValueError("target/loss ablation is long-only")
    data["day"] = data.__decision_ts__.dt.normalize()
    source = day_balanced(data)
    parity = pd.read_parquet(args.parity_predictions, columns=["candidate_id", "__decision_ts__", "recent_shift_bps"])
    parity["__decision_ts__"] = pd.to_datetime(parity["__decision_ts__"], utc=True)
    parity["day"] = parity.__decision_ts__.dt.normalize()
    check = parity.groupby("day", sort=True).recent_shift_bps.nunique(dropna=False)
    if int(check.max()) != 1:
        raise ValueError("original parity residual shift is not constant within a calendar day")
    shifts = parity.drop_duplicates("day").set_index("day").recent_shift_bps
    kinds = ("huber_clip", "huber_asin", "l1_clip", "l1_asin", "ordinal6")
    winner_rows: list[dict[str, object]] = []
    all_trials: list[pd.DataFrame] = []
    for kind in kinds:
        print(json.dumps({"event": "hpo_start", "kind": kind}), flush=True)
        params, trial_rows = _run_hpo(data, source, shifts, kind, args.hpo_trials, args.hpo_train_cap)
        all_trials.append(trial_rows)
        winner_rows.append({"kind": kind, **params})
        print(json.dumps({"event": "hpo_complete", "kind": kind, "params": params}), flush=True)
    winners = pd.DataFrame(winner_rows)
    pd.concat(all_trials, ignore_index=True).to_parquet(args.out_dir / "hpo_trials_2025.parquet", index=False)
    winners.to_parquet(args.out_dir / "hpo_winners_2025.parquet", index=False)
    for winner in winner_rows:
        kind = str(winner.pop("kind"))
        print(json.dumps({"event": "full_prediction_start", "kind": kind}), flush=True)
        prediction = _predict_full(data, source, shifts, kind, winner)
        prediction.to_parquet(args.out_dir / f"predictions_{kind}.parquet", index=False, compression="zstd")
        print(json.dumps({"event": "full_prediction_complete", "kind": kind, "rows": len(prediction)}), flush=True)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_mc1_d2_target_loss_ablation_v1", "status": "complete",
        "purpose": "MC1-only static target/loss ablation; no live mutation",
        "features": list(CORE), "static_training": "full-universe day-balanced; chronological monthly; deterministic 50k cap",
        "dynamic_shift": "reused original-mechanics daily 21d strictly prior-resolved residual shift",
        "label_availability_boundary": "policy_label_available_ts < fit/prediction cutoff",
        "hpo": "2025 Apr/Jul/Oct chronological folds only; full canonical replay deferred until after HPO",
        "validation": "2026 prediction outputs are opened validation, not selection input",
        "models": list(kinds), "admission": "+50 bps", "auction": "final_score only in reporting replay",
        "exclusions": ["R5", "live state", "exchange I/O", "new upstream base/consensus training"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
