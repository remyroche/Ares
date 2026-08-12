#!/usr/bin/env python3
"""Specialist target/query screen and bounded LambdaRank HPO.

This stage deliberately tunes shared specialist hyperparameters on a strictly
pre-transport development split.  The frozen seven side-specific views are
then refit with the selected target/query/parameters in the downstream stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from extreme_price_movements.query_candidate_definitions import assign_query_ids, query_definitions_by_name
from extreme_price_movements.query_funnel import load_frozen_query_shortlist
from extreme_price_movements.residual_lambdarank_hpo import (
    adjusted_hpo_score,
    era_portability_summary,
    make_pruned_study,
    materialize_lambdarank_params,
    ranker_early_stopping_callbacks,
    report_portability_progress,
    restore_broad_lambdarank_params,
    select_portability_winner,
    suggest_broad_lambdarank_params,
)
from scripts.run_frozen_multiview_specialist_input_ablation import (
    LEDGER, MAX_PROXY_ROWS, MAX_TRAIN_ROWS, SEED, STORE, _base, _sample, _store_rows, _utc,
)
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS

# Binary exact-H12 net > +50 bps is the currently requested default.  The
# command-line target/query inputs keep the broader predeclared grid available
# for a separately declared development ablation.
TARGETS = {
    "binary_h12_net50": None,
}
DEFAULT_CONTRACT = ROOT / "data_perp/artifacts/frozen_multiview_specialist_input_ablation_20260805_v1/frozen_view_contract.json"
DEFAULT_QUERY_POP = ROOT / "data_perp/artifacts/query_screen_population_20260810_v1.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/frozen_specialist_query_hpo_20260810_v1"
TRIALS = 12
HPO_PROXY_ROWS = 30_000
EARLY_STOPPING_ROUNDS = 30
SPECIALIST_QUERY_CANDIDATES = (
    "q0_exact_timestamp_side",
    "q1_cycle_1h_side",
    "q1_cycle_4h_side",
    "q1_cycle_8h_side",
    "q1_cycle_12h_side",
)


def _query_id(frame: pd.DataFrame, query_name: str) -> pd.Series:
    definition, = query_definitions_by_name([query_name])
    return assign_query_ids(frame, definition)


def _params(suggested: dict[str, object], *, train_rows: int) -> dict[str, object]:
    params = materialize_lambdarank_params(suggested, training_rows=train_rows)
    return {
        **params,
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": 1,
    }


def _rank_frame(frame: pd.DataFrame, fields: list[str], label: str, *, query_column: str,
                params: dict[str, object]) -> tuple[lgb.LGBMRanker, list[str], pd.DataFrame]:
    """Fit with an inner chronological early-stopping slice.

    The outer validation period remains untouched for target/query/HPO scoring.
    Early stopping only observes a later, label-resolved slice of the current
    training frame, and entire queries are kept on one side of that split.
    """
    x = frame[["candidate_id", "__ts__", query_column, *fields, label]].copy()
    x["_row"] = np.arange(len(x))
    x = x.sort_values([query_column, "candidate_id"], kind="stable")
    sizes = x.groupby(query_column, sort=False).size()
    x = x[x[query_column].isin(sizes.index[sizes.ge(2)])].copy()
    if x.empty:
        raise ValueError("no rankable queries")
    query_time = x.groupby(query_column, sort=False)["__ts__"].min().sort_values(kind="stable")
    holdout_count = max(1, int(np.ceil(len(query_time) * .2)))
    holdout_queries = set(query_time.index[-holdout_count:])
    fit = x[~x[query_column].isin(holdout_queries)].copy()
    early = x[x[query_column].isin(holdout_queries)].copy()
    # Keep a single fit when the proxy does not have enough independent query
    # groups for a valid inner slice.  The manifest records this fallback.
    if fit.empty or early.empty or fit[query_column].nunique() < 2:
        fit, early = x, pd.DataFrame(columns=x.columns)
    groups = fit.groupby(query_column, sort=False).size().to_numpy(np.int32)
    med = x[fields].apply(pd.to_numeric, errors="coerce").median()
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", **params)
    fit_args: dict[str, object] = {
        "group": groups,
    }
    if not early.empty:
        fit_args.update({
            "eval_set": [(early[fields].apply(pd.to_numeric, errors="coerce").fillna(med), early[label].to_numpy(float))],
            "eval_group": [early.groupby(query_column, sort=False).size().to_numpy(np.int32)],
            "callbacks": ranker_early_stopping_callbacks(rounds=EARLY_STOPPING_ROUNDS),
        })
    model.fit(fit[fields].apply(pd.to_numeric, errors="coerce").fillna(med), fit[label].to_numpy(float), **fit_args)
    x.attrs["early_stopping_used"] = bool(not early.empty)
    x.attrs["best_iteration"] = int(model.best_iteration_ or params["n_estimators"])
    return model, fields, x


def _load(contract: Path, query_population: Path, *, targets: list[str] | None) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]]]:
    base = _base()
    contract_json = json.loads(contract.read_text())
    views = {side: {name: list(fields) for name, fields in mapping.items()} for side, mapping in contract_json["views_by_side"].items()}
    query_columns = ["candidate_id"]
    for requested in targets or []:
        if requested == "binary_h12_net50":
            continue
        query_columns.append(requested if requested.startswith("grade_") else "grade_" + requested)
    q = pd.read_parquet(query_population, columns=list(dict.fromkeys(query_columns)))
    q = q.drop_duplicates("candidate_id")
    frame = base.merge(q, on="candidate_id", how="inner", validate="one_to_one")
    frame["binary_h12_net50"] = (frame.net_bps > 50.).astype(np.int32)
    return frame, views


def _development_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp("2024-05-01", tz="UTC")
    val_end = pd.Timestamp("2024-07-01", tz="UTC")
    train = frame[frame.__ts__.lt(train_end) & frame.label_available_ts.lt(train_end)].copy()
    val = frame[frame.__ts__.between(train_end, val_end, inclusive="left")].copy()
    return train, val


def _fit_and_score(train: pd.DataFrame, val: pd.DataFrame, views: dict[str, dict[str, list[str]]],
                   target_column: str, query_name: str, suggested: dict[str, object]) -> pd.DataFrame:
    chunks = []
    for side in ("long", "short"):
        tr, va = train[train.side_name.eq(side)].copy(), val[val.side_name.eq(side)].copy()
        # Shared hyperparameters are assessed on the complete specialist field
        # union; individual heads are refit with these settings downstream.
        fields = sorted({f for name in views[side] for f in views[side][name]})
        tr = _sample(tr, min(HPO_PROXY_ROWS, len(tr)))
        va = _sample(va, min(HPO_PROXY_ROWS, len(va)))
        fit_fields = _store_rows(tr, fields)
        val_fields = _store_rows(va, fields)
        tr = tr.merge(fit_fields, on="candidate_id", validate="one_to_one")
        va = va.merge(val_fields, on="candidate_id", validate="one_to_one")
        tr["query_id"] = _query_id(tr, query_name)
        params = _params(suggested, train_rows=len(tr))
        model, used, _ = _rank_frame(tr, fields, target_column, query_column="query_id", params=params)
        med = tr[used].apply(pd.to_numeric, errors="coerce").median()
        va_score = model.predict(va[used].apply(pd.to_numeric, errors="coerce").fillna(med))
        out = va[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps"]].copy()
        out["score"] = va_score
        chunks.append(out)
    return pd.concat(chunks, ignore_index=True)


def _monthly_top5_evs(pred: pd.DataFrame) -> list[float]:
    x = pred.copy()
    x["_month"] = pd.to_datetime(x["__ts__"], utc=True).dt.to_period("M").astype(str)
    return [
        float(global_tail_metrics(group)["top5_net_bps"])
        for _, group in x.groupby("_month", sort=True, observed=True)
    ]


def _objective(frame_train: pd.DataFrame, frame_val: pd.DataFrame,
               views: dict[str, dict[str, list[str]]], target_column: str,
               query_name: str, trial: optuna.Trial) -> float:
    query = _query_id(frame_train, query_name)
    suggested = suggest_broad_lambdarank_params(
        trial,
        retained_fraction=.05,
        median_candidates_per_query=float(query.groupby(query, observed=True).size().median()),
    )
    pred = _fit_and_score(frame_train, frame_val, views, target_column, query_name, suggested)
    era_evs: list[float] = []
    for value in _monthly_top5_evs(pred):
        era_evs.append(value)
        report_portability_progress(trial, era_evs)
    metrics = global_tail_metrics(pred)
    stability = monthly_stability(pred)
    summary = era_portability_summary(era_evs)
    for key, value in {**metrics, **stability, **summary}.items():
        trial.set_user_attr(key, value)
    return adjusted_hpo_score(
        era_evs=era_evs,
        max_depth=int(suggested["max_depth"]),
        num_leaves=int(suggested["num_leaves"]),
    )


def _target_query_arms(frame: pd.DataFrame, *, target_names: list[str] | None,
                       query_names: list[str] | None) -> list[tuple[str, str, str]]:
    targets = dict(TARGETS)
    if target_names:
        targets = {
            str(name).removeprefix("grade_"): (
                None if name == "binary_h12_net50"
                else str(name) if str(name).startswith("grade_") else "grade_" + str(name)
            )
            for name in target_names
        }
    if not targets:
        raise ValueError("no requested specialist targets are present in query population")
    queries = query_names or list(SPECIALIST_QUERY_CANDIDATES)
    # Validate names against the frozen grammar before any expensive fit.
    query_definitions_by_name(queries)
    return [
        (name, "binary_h12_net50" if column is None else column, query)
        for name, column in targets.items()
        for query in queries
    ]


def run(out: Path = DEFAULT_OUT, *, contract: Path = DEFAULT_CONTRACT,
        query_population: Path = DEFAULT_QUERY_POP, trials: int = TRIALS,
        targets: list[str] | None = None, queries: list[str] | None = None,
        query_shortlist: Path | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    frame, views = _load(contract, query_population, targets=targets)
    train, val = _development_splits(frame)
    if query_shortlist is not None:
        if queries is not None:
            raise ValueError("pass either explicit queries or query_shortlist, not both")
        queries = list(load_frozen_query_shortlist(query_shortlist))
    trial_rows = []
    winner_rows = []
    for name, target_column, query_name in _target_query_arms(frame, target_names=targets, query_names=queries):
        if target_column not in frame:
            raise KeyError(f"target {target_column!r} is absent from {query_population}")
        study = make_pruned_study(seed=SEED + len(trial_rows), n_startup_trials=3, n_warmup_steps=1)
        study.optimize(lambda trial: _objective(train, val, views, target_column, query_name, trial), n_trials=trials, show_progress_bar=False)
        for trial in study.trials:
            trial_rows.append({"target": name, "target_column": target_column, "query": query_name, "trial": trial.number, "state": trial.state.name, "value": trial.value, **trial.params, **{f"metric_{key}": value for key, value in trial.user_attrs.items()}})
        best = study.best_trial
        suggested = restore_broad_lambdarank_params(best.params)
        best_pred = _fit_and_score(train, val, views, target_column, query_name, suggested)
        winner_rows.append({"target": name, "target_column": target_column, "query": query_name, "trial": best.number,
                            "adjusted_hpo_score": best.value, **global_tail_metrics(best_pred), **monthly_stability(best_pred),
                            "params_json": json.dumps(best.params, sort_keys=True)})
    pd.DataFrame(trial_rows).to_parquet(out / "specialist_hpo_trials.parquet", index=False)
    winners = pd.DataFrame(winner_rows)
    winners.to_parquet(out / "specialist_target_query_winners.parquet", index=False)
    selected = select_portability_winner(
        winners.assign(
            arm=lambda x: x.target + "::" + x.query,
        ),
        tie_tolerance_bps=1.0,
    )
    pd.DataFrame([selected]).to_parquet(out / "specialist_target_query_selected.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps({"schema": "frozen_specialist_query_hpo_v2", "contract": str(contract), "targets": targets or list(TARGETS), "queries": queries or list(SPECIALIST_QUERY_CANDIDATES), "query_shortlist": str(query_shortlist) if query_shortlist else None, "trials_per_target_query": trials, "selection": "adjusted portability score during HPO; within one bps, monthly stability then top1 net", "early_stopping_rounds": EARLY_STOPPING_ROUNDS}, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--query-population", type=Path, default=DEFAULT_QUERY_POP)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--queries", nargs="*", default=None)
    parser.add_argument("--query-shortlist", type=Path, default=None)
    args = parser.parse_args()
    print(run(args.out, contract=args.contract, query_population=args.query_population,
              trials=args.trials, targets=args.targets, queries=args.queries,
              query_shortlist=args.query_shortlist))
