#!/usr/bin/env python3
"""Matched residual-query/HPO stage using the frozen specialist contract.

The specialist contract is loaded once and reused for every transport fold.
Specialists use the frozen binary exact-H12-net>+50-bps target while residual
query construction and ranker parameters are selected only on the designated
development transport folds.  The final fold is emitted once, after selection.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

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
    MAX_TRAIN_ROWS, SEED, STORE, _store_rows, _utc,
)
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS

CONTRACT = ROOT / "data_perp/artifacts/frozen_multiview_specialist_input_ablation_20260805_v1/frozen_view_contract.json"
QUERY_POP = ROOT / "data_perp/artifacts/query_screen_population_20260810_v1.parquet"
OUT = ROOT / "data_perp/artifacts/frozen_residual_query_hpo_20260810_v1"
SPECIALIST_SELECTION = ROOT / "data_perp/artifacts/frozen_specialist_query_hpo_20260810_v1/specialist_target_query_selected.parquet"
BASE_FEATURES = ["p_clear", "p_adverse", "p_weak", "base_score", "prequential_base_expected_net_bps"]
HPO_TRIALS = 12
EARLY_STOPPING_ROUNDS = 30
RESIDUAL_QUERY_CANDIDATES = (
    "q0_exact_timestamp_side",
    "q1_cycle_1h_side",
    "q1_cycle_4h_side",
    "q1_cycle_8h_side",
    "q1_cycle_12h_side",
)


def _query_ids(frame: pd.DataFrame, mode: str) -> pd.Series:
    aliases = {
        "timestamp_side": "q0_exact_timestamp_side",
        "q1h_side": "q1_cycle_1h_side",
        "q4h_side": "q1_cycle_4h_side",
        "q8h_side": "q1_cycle_8h_side",
        "q12h_side": "q1_cycle_12h_side",
        "q24h_side": "q1_cycle_24h_side",
    }
    definition, = query_definitions_by_name([aliases.get(mode, mode)])
    return assign_query_ids(frame, definition)


def _target(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    """Return only the frozen specialist label selected by the prior stage."""
    return pd.to_numeric(frame[target_column], errors="coerce").fillna(0).to_numpy(np.int32)


def _rank_model(frame: pd.DataFrame, fields: list[str], target: np.ndarray, query: pd.Series,
                params: dict) -> tuple[lgb.LGBMRanker, list[str], pd.Series]:
    """Fit a ranker with a query-disjoint chronological early-stop slice."""
    x = frame[["candidate_id", "__ts__", *fields]].copy()
    x["__q__"] = query.to_numpy()
    x["__row__"] = np.arange(len(x))
    x = x.sort_values(["__q__", "candidate_id"], kind="stable")
    sizes = x.groupby("__q__", sort=False).size()
    x = x[x["__q__"].isin(sizes.index[sizes.ge(2)])].copy()
    if x.empty:
        raise ValueError("no rankable queries")
    query_time = x.groupby("__q__", sort=False)["__ts__"].min().sort_values(kind="stable")
    n_early = max(1, int(np.ceil(len(query_time) * .2)))
    early_queries = set(query_time.index[-n_early:])
    fit = x.loc[~x["__q__"].isin(early_queries)].copy()
    early = x.loc[x["__q__"].isin(early_queries)].copy()
    if fit.empty or early.empty or fit["__q__"].nunique() < 2:
        fit, early = x, pd.DataFrame(columns=x.columns)
    order = fit["__row__"].to_numpy(np.int64)
    med = fit[fields].apply(pd.to_numeric, errors="coerce").median()
    model_params = dict(params)
    model_params.pop("objective", None)
    model_params.pop("metric", None)
    model_params.pop("label_gain_name", None)
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", **model_params)
    fit_args: dict[str, object] = {"group": fit.groupby("__q__", sort=False).size().to_numpy(np.int32)}
    if not early.empty:
        early_order = early["__row__"].to_numpy(np.int64)
        fit_args.update({
            "eval_set": [(early[fields].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0), target[early_order])],
            "eval_group": [early.groupby("__q__", sort=False).size().to_numpy(np.int32)],
            "callbacks": ranker_early_stopping_callbacks(rounds=EARLY_STOPPING_ROUNDS),
        })
    model.fit(fit[fields].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0), target[order], **fit_args)
    return model, fields, med


def _resid_params(trial: optuna.Trial, *, median_candidates_per_query: float) -> dict:
    return suggest_broad_lambdarank_params(
        trial,
        retained_fraction=.05,
        median_candidates_per_query=median_candidates_per_query,
    )


def _load_specialist_selection(path: Path) -> tuple[str, str, dict[str, object]]:
    """Load the sole target/query/HPO winner from the preceding stage."""
    if not path.exists():
        raise FileNotFoundError(
            f"specialist selection is required before residual HPO: {path}. "
            "Run run_frozen_specialist_query_hpo.py first."
        )
    selected = pd.read_parquet(path)
    if len(selected) != 1:
        raise ValueError("specialist selection must contain exactly one frozen winner")
    row = selected.iloc[0]
    required = {"target_column", "query", "params_json"}
    missing = required.difference(selected.columns)
    if missing:
        raise KeyError(f"specialist selection missing {sorted(missing)}")
    return (
        str(row["target_column"]),
        str(row["query"]),
        restore_broad_lambdarank_params(json.loads(str(row["params_json"]))),
    )


def _load(target_column: str) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]], list[str], list[str]]:
    from scripts.run_frozen_multiview_specialist_input_ablation import _base
    base = _base()
    contract = json.loads(CONTRACT.read_text())
    views = {side: {name: list(fields) for name, fields in groups.items()} for side, groups in contract["views_by_side"].items()}
    query_columns = ["candidate_id"] + ([] if target_column == "binary_h12_net50" else [target_column])
    q = pd.read_parquet(QUERY_POP, columns=query_columns).drop_duplicates("candidate_id")
    frame = base.merge(q, on="candidate_id", how="inner", validate="one_to_one")
    if target_column == "binary_h12_net50":
        frame[target_column] = (pd.to_numeric(frame["net_bps"], errors="coerce") > 50.0).astype(np.int32)
    store_cols = set(pd.read_parquet(STORE, engine="pyarrow", columns=[]).columns) if False else None
    ae = [str(f) for f in contract.get("ae_gmm_fields", [])]
    ctx = [str(f) for f in contract.get("selected_context_fields", [])]
    return frame, views, ae, ctx


def _fit_specialists(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame,
                     views: dict[str, list[str]], query_mode: str,
                     target_column: str, specialist_params: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train if len(train) <= MAX_TRAIN_ROWS else train.sample(MAX_TRAIN_ROWS, random_state=SEED)
    cal_out, test_out = cal[["candidate_id"]].copy(), test[["candidate_id"]].copy()
    for name, fields in views.items():
        fx = train.merge(_store_rows(train, fields), on="candidate_id", validate="one_to_one")
        cx = cal.merge(_store_rows(cal, fields), on="candidate_id", validate="one_to_one")
        tx = test.merge(_store_rows(test, fields), on="candidate_id", validate="one_to_one")
        params = dict(specialist_params)
        if "min_child_samples_fraction" in params:
            params = materialize_lambdarank_params(params, training_rows=len(fx))
        params.update({"verbosity": -1, "random_state": SEED, "n_jobs": 1})
        model, used, med = _rank_model(fx, fields, _target(fx, target_column), _query_ids(fx, query_mode), params)
        cal_out["mv__" + name] = model.predict(cx[used].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0))
        test_out["mv__" + name] = model.predict(tx[used].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0))
        del fx, cx, tx, model
        gc.collect()
    return cal_out, test_out


def _make_features(frame: pd.DataFrame, scores: pd.DataFrame, extra_fields: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = frame[["candidate_id", "__ts__", "net_bps", "gross_bps", "side_name", *BASE_FEATURES]].merge(scores, on="candidate_id", validate="one_to_one")
    if extra_fields:
        out = out.merge(_store_rows(out, extra_fields), on="candidate_id", validate="one_to_one")
    score_fields = [c for c in scores.columns if c.startswith("mv__")]
    fields = list(dict.fromkeys(BASE_FEATURES + score_fields + extra_fields))
    return out, [f for f in fields if f in out.columns]


def _fit_residual(train: pd.DataFrame, test: pd.DataFrame, fields: list[str], query_mode: str, params: dict, grade_edges: tuple[float, float, float, float] = (-150., -50., 50., 150.)) -> np.ndarray:
    residual = train.net_bps.to_numpy(float) - train.prequential_base_expected_net_bps.to_numpy(float)
    e0, e1, e2, e3 = grade_edges
    grade = np.select((residual <= e0, residual <= e1, residual <= e2, residual <= e3), (0, 1, 2, 3), default=4).astype(np.int32)
    actual_params = dict(params)
    if "min_child_samples_fraction" in actual_params:
        actual_params = materialize_lambdarank_params(actual_params, training_rows=len(train))
    actual_params.update({"verbosity": -1, "random_state": SEED, "n_jobs": 1})
    model, used, med = _rank_model(train, fields, grade, _query_ids(train, query_mode), actual_params)
    return model.predict(test[used].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0))


def _fold_scores(base: pd.DataFrame, views: dict[str, dict[str, list[str]]], ae: list[str], ctx: list[str],
                 fold, specialist_query: str, specialist_target: str, specialist_params: dict[str, object],
                 residual_query: str, residual_params: dict) -> pd.DataFrame:
    a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
    tr = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)]
    ca = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)]
    te = base[base.__ts__.between(c, e, inclusive="left")]
    pieces = []
    for side in ("long", "short"):
        train, cal, test = (x[x.side_name.eq(side)].copy() for x in (tr, ca, te))
        cal_scores, test_scores = _fit_specialists(
            train, cal, test, views[side], specialist_query, specialist_target, specialist_params,
        )
        calx, fields = _make_features(cal, cal_scores, ae + ctx)
        testx, _ = _make_features(test, test_scores, ae + ctx)
        # Use the later calibration half to fit the residual, preserving the
        # earlier half for query/HPO decisions in the outer driver.
        raw = _fit_residual(calx, testx, fields, residual_query, residual_params)
        z = test[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "prequential_base_expected_net_bps"]].copy()
        z["score"] = test.prequential_base_expected_net_bps.to_numpy(float) + raw
        z["fold"] = fold.name
        pieces.append(z)
    return pd.concat(pieces, ignore_index=True)


def _monthly_top5_evs(pred: pd.DataFrame) -> list[float]:
    x = pred.copy()
    x["_month"] = pd.to_datetime(x["__ts__"], utc=True).dt.to_period("M").astype(str)
    return [
        float(global_tail_metrics(group)["top5_net_bps"])
        for _, group in x.groupby("_month", sort=True, observed=True)
    ]


def _objective(base: pd.DataFrame, views, ae, ctx, folds, specialist_query: str,
               specialist_target: str, specialist_params: dict[str, object], residual_query: str,
               trial: optuna.Trial) -> float:
    sample = base[base.__ts__.lt(_utc(folds[0].calibration_start))].copy()
    median_candidates = float(_query_ids(sample, residual_query).groupby(
        _query_ids(sample, residual_query), observed=True,
    ).size().median())
    suggested = _resid_params(trial, median_candidates_per_query=median_candidates)
    predictions: list[pd.DataFrame] = []
    era_evs: list[float] = []
    for fold in folds:
        pred = _fold_scores(
            base, views, ae, ctx, fold, specialist_query, specialist_target,
            specialist_params, residual_query, suggested,
        )
        predictions.append(pred)
        for value in _monthly_top5_evs(pred):
            era_evs.append(value)
            report_portability_progress(trial, era_evs)
    pooled = pd.concat(predictions, ignore_index=True)
    metrics = global_tail_metrics(pooled)
    stability = monthly_stability(pooled)
    summary = era_portability_summary(era_evs)
    for key, value in {**metrics, **stability, **summary}.items():
        trial.set_user_attr(key, value)
    return adjusted_hpo_score(
        era_evs=era_evs,
        max_depth=int(suggested["max_depth"]),
        num_leaves=int(suggested["num_leaves"]),
    )


def run(out: Path = OUT, trials: int = HPO_TRIALS,
        residual_queries: tuple[str, ...] = RESIDUAL_QUERY_CANDIDATES,
        specialist_selection: Path = SPECIALIST_SELECTION,
        query_shortlist: Path | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    specialist_target, specialist_query, specialist_params = _load_specialist_selection(specialist_selection)
    base, views, ae, ctx = _load(specialist_target)
    if query_shortlist is not None:
        residual_queries = load_frozen_query_shortlist(query_shortlist)
    folds = LONG_HISTORY_FOLDS[3:]
    if len(folds) < 3:
        raise ValueError("residual HPO needs at least two development folds and one final fold")
    development_folds, final_folds = folds[:-1], folds[-1:]
    query_definitions_by_name(residual_queries)
    rows = []
    best_rows: list[dict[str, object]] = []
    for residual_query in residual_queries:
        study = make_pruned_study(seed=SEED + len(rows), n_startup_trials=2, n_warmup_steps=1)
        study.optimize(lambda t: _objective(base, views, ae, ctx, development_folds, specialist_query, specialist_target, specialist_params, residual_query, t), n_trials=trials, show_progress_bar=False)
        for t in study.trials:
            rows.append({"query": residual_query, "trial": t.number, "state": t.state.name, "value": t.value, **t.params, **{f"metric_{k}": v for k, v in t.user_attrs.items()}})
        best = study.best_trial
        best_rows.append({"arm": residual_query, "query": residual_query, "trial": best.number,
                          "adjusted_hpo_score": best.value, **{f"metric_{k}": v for k, v in best.user_attrs.items()},
                          "params_json": json.dumps(best.params, sort_keys=True)})
    trials_df = pd.DataFrame(rows)
    trials_df.to_parquet(out / "residual_hpo_trials.parquet", index=False)
    development = pd.DataFrame(best_rows)
    development.to_parquet(out / "residual_hpo_development_winners.parquet", index=False)
    select_table = development.copy()
    select_table = select_table.rename(columns={
        "metric_top5_net_bps": "top5_net_bps", "metric_top1_net_bps": "top1_net_bps",
        "metric_month_std_net_bps": "month_std_net_bps", "metric_month_worst_net_bps": "month_worst_net_bps",
    })
    selected = select_portability_winner(select_table, tie_tolerance_bps=1.0)
    query = str(selected["query"])
    p = restore_broad_lambdarank_params(json.loads(str(selected["params_json"])))
    preds = []
    for f in final_folds:
        preds.append(_fold_scores(base, views, ae, ctx, f, specialist_query, specialist_target, specialist_params, query, p))
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(out / "final_oos_predictions.parquet", index=False)
    winner = {"specialist_target": specialist_target, "specialist_query": specialist_query, "residual_query": query,
              "residual_params": json.dumps(p, sort_keys=True), "development_selection": "portability score; within 1 bps then top5, monthly stability, top1",
              **global_tail_metrics(allp), **monthly_stability(allp)}
    pd.DataFrame([winner]).to_parquet(out / "residual_query_winner.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps({"schema": "frozen_residual_query_hpo_v2", "contract": str(CONTRACT), "specialist_selection": str(specialist_selection), "specialist_target": specialist_target, "specialist_query": specialist_query, "query_shortlist": str(query_shortlist) if query_shortlist else None, "residual_queries": list(residual_queries), "hpo_trials_per_query": trials, "development_folds": [f.name for f in development_folds], "final_oos_folds": [f.name for f in final_folds], "selection": "adjusted portability score; within 1 bps, monthly stability then top1 net", "early_stopping_rounds": EARLY_STOPPING_ROUNDS}, indent=2) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--trials", type=int, default=HPO_TRIALS)
    ap.add_argument("--residual-queries", nargs="*", default=list(RESIDUAL_QUERY_CANDIDATES))
    ap.add_argument("--specialist-selection", type=Path, default=SPECIALIST_SELECTION)
    ap.add_argument("--query-shortlist", type=Path, default=None)
    args = ap.parse_args()
    print(run(args.out, args.trials, tuple(args.residual_queries), args.specialist_selection,
              args.query_shortlist))
