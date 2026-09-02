#!/usr/bin/env python3
"""Strict-prequential short residual LambdaRank control.

This is intentionally a *research control*, not a promotion path.  It first
creates side-local, chronological base OOF predictions, maps timestamp-local
base rank to H12 net only from prior OOF outcomes, and trains a residual
ranker on the remaining error.  No in-sample base prediction or April--June
outcome is consumed by the residual fit or its HPO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.query_candidate_definitions import assign_query_ids, query_definitions_by_name
from extreme_price_movements.residual_lambdarank_hpo import (
    adjusted_hpo_score,
    era_portability_summary,
    make_pruned_study,
    materialize_lambdarank_params,
    ranker_early_stopping_callbacks,
    restore_broad_lambdarank_params,
    stop_after_no_improvement,
    suggest_broad_lambdarank_params,
)
from scripts.run_frozen_specialist_query_hpo import (
    DEFAULT_CANDIDATES,
    DEFAULT_FEATURES,
    DEFAULT_LABELS,
    MAX_TRAIN_QUERY_ROWS,
    _ledger,
    _rank_oos_metrics,
)
from scripts.run_strict_r3_short_base_3m_oos import OOS_END, OOS_START, TRAIN_START, _matrix
from scripts.run_strict_r3_short_target_ablations_3m_oos import _valid_label


DEFAULT_WINNERS = ROOT / "data_perp/artifacts/strict_r3_short_specialist_query_hpo_20260820_v5/specialist_target_query_winners.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_short_residual_query_hpo_20260820_v1"
SEED = 1729
HPO_TRIALS = 4
NO_IMPROVEMENT_PATIENCE = 20
EARLY_STOPPING_ROUNDS = 30
RESIDUAL_QUERIES = ("q0_exact_timestamp_side", "q1_cycle_4h_side")
RESIDUAL_EDGES_BPS = (-200.0, -50.0, 50.0, 200.0)
MIN_MAP_ROWS = 500
RESIDUAL_BLEND = 0.25


def _winner(path: Path) -> dict[str, Any]:
    winners = pd.read_parquet(path)
    required = {"spec", "query", "params_json", "adjusted_hpo_score", "mean_best_iteration"}
    missing = required.difference(winners.columns)
    if missing:
        raise KeyError(f"specialist winner artifact missing {sorted(missing)}")
    # Development score only.  The later Apr--Jun tail is never used here.
    return winners.sort_values(["adjusted_hpo_score", "spec"], ascending=[False, True], kind="stable").iloc[0].to_dict()


def _base_fit(
    frame: pd.DataFrame,
    fields: list[str],
    *, spec: str, query: str, params: dict[str, Any], cutoff: pd.Timestamp,
    start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    """Fit only labels resolved before ``cutoff`` and score one held window."""
    from scripts.run_frozen_specialist_query_hpo import _fit
    from scripts.run_strict_r3_short_target_ablations_3m_oos import SPECS

    train = frame.loc[
        frame.__ts__.ge(TRAIN_START) & frame.__ts__.lt(cutoff)
        & _valid_label(frame) & frame.__label_available_at__.lt(cutoff)
    ].copy()
    held = frame.loc[frame.__ts__.ge(start) & frame.__ts__.lt(end)].copy()
    if train.empty or held.empty:
        raise ValueError(f"base prequential window {start} has no strict train/held rows")
    # `_fit` accepts the target selector name and applies the same fixed
    # target transformation that won the prior development funnel.
    _, _, score, iteration = _fit(train, held, fields, spec, query, params)
    result = held.loc[:, [
        "candidate_id", "__ts__", "__decision_ts__", "__label_available_at__", "__symbol__", "side_name",
        "label_valid", "target_invalid", "invalid_reason", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
    ]].copy()
    result["base_score"] = score
    result["base_iteration"] = int(iteration)
    result["base_rank_ts"] = result.groupby("__ts__", sort=False)["base_score"].rank(pct=True, method="first")
    return result


def _rank_bin(value: pd.Series) -> pd.Series:
    return np.minimum(19, np.floor(np.clip(pd.to_numeric(value, errors="coerce").to_numpy(float), 0.0, .999999) * 20.0)).astype(np.int16)


def _prior_map(history: pd.DataFrame, ranks: pd.Series) -> np.ndarray:
    net = pd.to_numeric(history.t4_tp6_sl4_net_bps, errors="coerce")
    valid = np.isfinite(net.to_numpy(float))
    if int(valid.sum()) < MIN_MAP_ROWS:
        return np.full(len(ranks), np.nan, dtype=np.float32)
    x = history.loc[valid, ["base_rank_ts", "t4_tp6_sl4_net_bps"]].copy()
    x["bin"] = _rank_bin(x.base_rank_ts)
    overall = float(pd.to_numeric(x.t4_tp6_sl4_net_bps, errors="coerce").mean())
    grouped = x.groupby("bin", observed=True).t4_tp6_sl4_net_bps.agg(["mean", "count"])
    # Side-local shrinkage only; it makes sparse rank cells conservative
    # without accessing same- or future-time outcomes.
    estimates = np.full(20, overall, dtype=np.float64)
    for index, row in grouped.iterrows():
        estimates[int(index)] = (float(row["count"]) * float(row["mean"]) + 200.0 * overall) / (float(row["count"]) + 200.0)
    return estimates[_rank_bin(ranks)].astype(np.float32)


def _prequential_base_anchor(oof: pd.DataFrame) -> pd.DataFrame:
    """Map OOF base ranks using only fully resolved earlier OOF outcomes."""
    source = oof.sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    source["base_anchor_bps"] = np.nan
    pending: list[pd.DataFrame] = []
    history: list[pd.DataFrame] = []
    for timestamp, group in source.groupby("__ts__", sort=True, observed=True):
        if pending:
            ready = [piece for piece in pending if piece.__label_available_at__.max() <= timestamp]
            pending = [piece for piece in pending if piece.__label_available_at__.max() > timestamp]
            if ready:
                history.extend(ready)
        previous = pd.concat(history, ignore_index=True) if history else pd.DataFrame(columns=group.columns)
        source.loc[group.index, "base_anchor_bps"] = _prior_map(previous, group.base_rank_ts)
        labelled = group.loc[_valid_label(group)].copy()
        if not labelled.empty:
            pending.append(labelled)
    source["policy_residual_bps"] = (
        pd.to_numeric(source.t4_tp6_sl4_net_bps, errors="coerce")
        - pd.to_numeric(source.base_anchor_bps, errors="coerce")
    )
    return source


def _frozen_prior_map(oof: pd.DataFrame, ranks: pd.Series) -> np.ndarray:
    """Return the pre-April rank-to-net map for held OOS rows."""
    resolved = oof.loc[_valid_label(oof)].copy()
    return _prior_map(resolved, ranks)


def _residual_grade(value: pd.Series) -> pd.Series:
    return pd.Series(np.digitize(pd.to_numeric(value, errors="coerce"), RESIDUAL_EDGES_BPS, right=True), index=value.index, dtype="Int8")


def _cap_and_query(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    definition, = query_definitions_by_name([query])
    x = frame.copy()
    x["__query__"] = assign_query_ids(x, definition)
    x["__sample__"] = x.candidate_id.astype(str).map(
        lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
    )
    x = (
        x.sort_values(["__query__", "__sample__", "candidate_id"], kind="stable")
        .groupby("__query__", sort=False, observed=True).head(MAX_TRAIN_QUERY_ROWS)
        .drop(columns="__sample__")
    )
    sizes = x.groupby("__query__", observed=True)["__grade__"].agg(["size", "nunique"])
    return x.loc[x.__query__.isin(sizes.index[sizes["size"].ge(2) & sizes["nunique"].ge(2)])].sort_values(["__query__", "candidate_id"], kind="stable")


def _inner_early(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    starts = frame.groupby("__query__", sort=False).__ts__.min().sort_values(kind="stable")
    count = max(1, int(math.ceil(len(starts) * .20)))
    names = set(starts.index[-count:])
    early = frame.loc[frame.__query__.isin(names)].copy()
    fit = frame.loc[~frame.__query__.isin(names)].copy()
    if fit.empty or early.empty or fit.__query__.nunique() < 2:
        return frame, None
    return fit, early


def _fit_residual(
    train: pd.DataFrame, predict: pd.DataFrame, fields: list[str], query: str, params: dict[str, Any], *, early: pd.DataFrame | None = None,
) -> tuple[np.ndarray, int]:
    prepared = _cap_and_query(train, query)
    if prepared.empty:
        raise ValueError("no rankable residual training queries")
    medians = prepared.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    actual = materialize_lambdarank_params(params, training_rows=len(prepared)) if "min_child_samples_fraction" in params else dict(params)
    actual.update({"verbosity": -1, "random_state": SEED, "n_jobs": 1, "deterministic": True, "force_col_wise": True})
    model = lgb.LGBMRanker(**actual)
    kwargs: dict[str, Any] = {"group": prepared.groupby("__query__", sort=False).size().to_numpy(np.int32)}
    if early is not None and not early.empty:
        early = _cap_and_query(early, query)
        if not early.empty:
            kwargs.update({
                "eval_set": [(_matrix(early, fields, medians), early.__grade__.astype(np.int32).to_numpy())],
                "eval_group": [early.groupby("__query__", sort=False).size().to_numpy(np.int32)],
                "callbacks": ranker_early_stopping_callbacks(rounds=EARLY_STOPPING_ROUNDS),
            })
    model.fit(_matrix(prepared, fields, medians), prepared.__grade__.astype(np.int32).to_numpy(), **kwargs)
    iteration = int(model.best_iteration_ or actual["n_estimators"])
    return np.asarray(model.predict(_matrix(predict, fields, medians), num_iteration=iteration), dtype=np.float32), iteration


def _combine(base_rank: pd.Series, residual_score: np.ndarray, timestamp: pd.Series) -> np.ndarray:
    residual_rank = pd.Series(residual_score, index=base_rank.index).groupby(timestamp, sort=False).rank(pct=True, method="first")
    return (pd.to_numeric(base_rank, errors="coerce") + RESIDUAL_BLEND * (residual_rank - .5)).to_numpy(np.float32)


def _tail(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    ordered = frame.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable")
    valid = _valid_label(ordered)
    values: dict[str, float] = {}
    for fraction in (.01, .02, .05):
        picked = ordered.iloc[:max(1, int(math.ceil(len(ordered) * fraction)))]
        outcome = picked.loc[_valid_label(picked)]
        values[f"top{int(fraction * 100)}_net_bps"] = float(pd.to_numeric(outcome.t4_tp6_sl4_net_bps, errors="coerce").mean())
    return values


def run(*, winners: Path, features: Path, labels: Path, candidates: Path, out: Path, hpo_trials: int) -> Path:
    if out.exists():
        raise FileExistsError(f"output must be new: {out}")
    selected = _winner(winners)
    params = restore_broad_lambdarank_params(json.loads(str(selected["params_json"])))
    params["n_estimators"] = int(round(float(selected["mean_best_iteration"])))
    frame, fields = _ledger(features, labels, candidates)
    feb = _base_fit(frame, fields, spec=str(selected["spec"]), query=str(selected["query"]), params=params,
                    cutoff=pd.Timestamp("2024-02-01", tz="UTC"), start=pd.Timestamp("2024-02-01", tz="UTC"), end=pd.Timestamp("2024-03-01", tz="UTC"))
    mar = _base_fit(frame, fields, spec=str(selected["spec"]), query=str(selected["query"]), params=params,
                    cutoff=pd.Timestamp("2024-03-01", tz="UTC"), start=pd.Timestamp("2024-03-01", tz="UTC"), end=pd.Timestamp("2024-04-01", tz="UTC"))
    oof = _prequential_base_anchor(pd.concat([feb, mar], ignore_index=True))
    oof = oof.loc[oof.base_anchor_bps.notna() & _valid_label(oof)].copy()
    oof["__grade__"] = _residual_grade(oof.policy_residual_bps)
    # Carry all causal base fields only after OOF target construction.
    oof = oof.merge(frame[["candidate_id", *fields]], on="candidate_id", validate="one_to_one")
    residual_fields = ["base_score", "base_rank_ts", "base_anchor_bps", *fields]
    feb_train = oof.loc[oof.__ts__.lt(pd.Timestamp("2024-03-01", tz="UTC"))].copy()
    mar_valid = oof.loc[oof.__ts__.ge(pd.Timestamp("2024-03-01", tz="UTC"))].copy()
    if feb_train.empty or mar_valid.empty:
        raise ValueError("prequential residual control has insufficient February/March support")
    trials: list[dict[str, Any]] = []
    winners_rows: list[dict[str, Any]] = []
    for query in RESIDUAL_QUERIES:
        probe = _cap_and_query(feb_train, query)
        median_query = float(probe.groupby("__query__", observed=True).size().median())
        study = make_pruned_study(seed=SEED + len(winners_rows), n_startup_trials=2, n_warmup_steps=1)

        def objective(trial: optuna.Trial) -> float:
            suggested = suggest_broad_lambdarank_params(trial, retained_fraction=.05, median_candidates_per_query=median_query)
            prepared = _cap_and_query(feb_train, query)
            fit, early = _inner_early(prepared)
            score, iteration = _fit_residual(fit, mar_valid, residual_fields, query, suggested, early=early)
            combined = _combine(mar_valid.base_rank_ts, score, mar_valid.__ts__)
            current = mar_valid.copy()
            current["score"] = combined
            metrics = _tail(current, "score")
            for key, value in {**metrics, "best_iteration": iteration}.items():
                trial.set_user_attr(key, value)
            return adjusted_hpo_score(era_evs=[metrics["top5_net_bps"]], max_depth=int(suggested["max_depth"]), num_leaves=int(suggested["num_leaves"]))

        study.optimize(
            objective,
            n_trials=hpo_trials,
            show_progress_bar=False,
            callbacks=[stop_after_no_improvement(patience=NO_IMPROVEMENT_PATIENCE)],
        )
        for trial in study.trials:
            trials.append({"query": query, "trial": trial.number, "state": trial.state.name, "adjusted_hpo_score": trial.value, **trial.params, **{f"metric_{key}": value for key, value in trial.user_attrs.items()}})
        best = study.best_trial
        winners_rows.append({"query": query, "trial": best.number, "adjusted_hpo_score": best.value, "params_json": json.dumps(best.params, sort_keys=True), "stop_reason": study.user_attrs.get("stop_reason", "trial_budget"), "no_improvement_patience": int(NO_IMPROVEMENT_PATIENCE), **{f"metric_{key}": value for key, value in best.user_attrs.items()}})
    selected_resid = pd.DataFrame(winners_rows).sort_values(["adjusted_hpo_score", "query"], ascending=[False, True], kind="stable").iloc[0].to_dict()
    residual_params = restore_broad_lambdarank_params(json.loads(str(selected_resid["params_json"])))
    residual_params["n_estimators"] = int(selected_resid["metric_best_iteration"])
    final_base = _base_fit(frame, fields, spec=str(selected["spec"]), query=str(selected["query"]), params=params,
                           cutoff=OOS_START, start=OOS_START, end=OOS_END)
    final_base["base_anchor_bps"] = _frozen_prior_map(oof, final_base.base_rank_ts)
    final_base = final_base.merge(frame[["candidate_id", *fields]], on="candidate_id", validate="one_to_one")
    score, iteration = _fit_residual(oof, final_base, residual_fields, str(selected_resid["query"]), residual_params)
    final_base["residual_score"] = score
    final_base["combined_score"] = _combine(final_base.base_rank_ts, score, final_base.__ts__)
    final_base["score"] = final_base.combined_score
    total, tails = _rank_oos_metrics(final_base, str(selected["spec"]), scope="2024-04_to_2024-06")
    months: list[dict[str, Any]] = []
    month_tails: list[dict[str, Any]] = []
    for month, group in final_base.groupby(final_base.__ts__.dt.strftime("%Y-%m"), sort=True):
        result, current_tails = _rank_oos_metrics(group, str(selected["spec"]), scope=str(month))
        months.append(result)
        month_tails.extend(current_tails)
    out.mkdir(parents=True)
    oof.to_parquet(out / "prequential_base_residual_training.parquet", index=False, compression="zstd")
    final_base.to_parquet(out / "final_oos_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(trials).to_parquet(out / "residual_hpo_trials.parquet", index=False, compression="zstd")
    pd.DataFrame(winners_rows).to_parquet(out / "residual_query_winners.parquet", index=False, compression="zstd")
    pd.DataFrame([{**total, "final_iteration": iteration}, *months]).to_parquet(out / "final_oos_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(tails + month_tails).to_parquet(out / "final_oos_tail_metrics.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_prequential_residual_lambdarank_v1", "side": "short",
        "base_selection": {key: selected[key] for key in ("spec", "query", "adjusted_hpo_score", "mean_best_iteration")},
        "base_oof": "Jan model -> Feb, Jan-Feb model -> Mar; target labels resolved before each fit",
        "base_anchor": "timestamp-local base rank mapped only from prior resolved base-OOF outcomes; 20 rank bins with side-local shrinkage",
        "residual_target": f"exact H12 TP6/SL4 net minus causal base anchor; ordinal bins {RESIDUAL_EDGES_BPS}",
        "residual_features": "base score/rank/anchor plus repaired 120 short causal base fields; research-only shared contract",
        "residual_queries": list(RESIDUAL_QUERIES), "query_train_cap": MAX_TRAIN_QUERY_ROWS,
        "hpo": "broad residual_lambdarank_hpo search, 4 trials/query, chronological inner 20% query early-stop, 30 rounds",
        "combination": f"timestamp-local base rank + {RESIDUAL_BLEND} * (residual rank - 0.5)",
        "final_oos": "April-June scored once after all selection; ranks all entry-executable candidates before outcome resolution",
        "no_promotion": "This is a short residual negative control. It cannot promote an architecture without an independently positive base target.",
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winners", type=Path, default=DEFAULT_WINNERS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hpo-trials", type=int, default=HPO_TRIALS)
    args = parser.parse_args()
    print(run(winners=args.winners, features=args.features, labels=args.labels, candidates=args.candidates, out=args.out, hpo_trials=args.hpo_trials))


if __name__ == "__main__":
    main()
