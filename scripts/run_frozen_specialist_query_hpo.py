#!/usr/bin/env python3
"""Sequential short-base LambdaRank target/query funnel.

The historical script name is retained because this is the specialist
target/query-HPO stage.  This implementation is explicitly side-local: it
uses only short, target-free entry-executable candidates, the repaired
120-field contract and strict chronological labels.  It is not a production
specialist contract or a residual-stack promotion.
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
from extreme_price_movements.query_funnel import load_frozen_query_shortlist
from extreme_price_movements.residual_lambdarank_hpo import (
    adjusted_hpo_score,
    era_portability_summary,
    make_pruned_study,
    materialize_lambdarank_params,
    ranker_early_stopping_callbacks,
    report_portability_progress,
    restore_broad_lambdarank_params,
    stop_after_no_improvement,
    suggest_broad_lambdarank_params,
)
from scripts.materialize_short_lambdarank_query_population import _to_grade
from scripts.run_strict_r3_short_base_3m_oos import (
    FEATURE_CONTRACT,
    OOS_END,
    OOS_START,
    TRAIN_START,
    _causal_coverage_fields,
    _load_candidates,
    _load_feature_contract,
    _load_features,
    _matrix,
)
from scripts.run_strict_r3_short_target_ablations_3m_oos import (
    SPECS,
    _load_labels,
    _metrics,
    _prediction_columns,
    _target_values,
    _valid_label,
)


DEFAULT_FEATURES = ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v5/features/canonical120_features.parquet"
DEFAULT_LABELS = ROOT / "data_perp/artifacts/strict_r3_short_target_labels_2024_20260820_v1"
DEFAULT_CANDIDATES = ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v1/short_target_free_candidate_population.parquet"
DEFAULT_QUERY_ROOT = ROOT / "data_perp/artifacts/strict_r3_short_query_screen_20260820_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_short_specialist_query_hpo_20260820_v1"
SEED = 1729
SCREEN_TOP_ARMS = 3
HPO_TRIALS = 6
NO_IMPROVEMENT_PATIENCE = 20
PROXY_ROWS = 60_000
EARLY_STOPPING_ROUNDS = 30
# LambdaRank computes a comparatively expensive gradient inside every query.
# The four-hour candidate universe is intentionally broad, so cap only the
# *training representation* of an oversized group with a deterministic hash
# of decision-time identity.  OOS prediction and economic ranking continue
# over every executable candidate.
MAX_TRAIN_QUERY_ROWS = 64


def _sample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Deterministically thin a development fold without random reshuffling."""
    if len(frame) <= maximum:
        return frame.copy()
    ordered = frame.sort_values(["__ts__", "candidate_id"], kind="stable")
    positions = np.linspace(0, len(ordered) - 1, num=maximum, dtype=np.int64)
    return ordered.iloc[np.unique(positions)].copy()


def _ledger(features_path: Path, labels_root: Path, candidates_path: Path) -> tuple[pd.DataFrame, list[str]]:
    fields = _load_feature_contract(FEATURE_CONTRACT)
    features = _load_features(features_path, fields)
    candidates = _load_candidates(candidates_path)
    frame = features.merge(
        candidates,
        on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"],
        how="left", validate="one_to_one",
    )
    if frame.entry_executable.isna().any():
        raise AssertionError("features do not have exact target-free candidate identities")
    frame = frame.loc[
        frame.entry_executable.astype(bool)
        & frame.__ts__.ge(TRAIN_START)
        & frame.__ts__.lt(OOS_END)
    ].copy()
    labels = _load_labels(labels_root)
    frame = frame.merge(
        labels, on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"],
        how="left", validate="one_to_one",
    )
    coverage_pop = frame.loc[frame.__ts__.ge(TRAIN_START) & frame.__ts__.lt(OOS_START)].copy()
    coverage_pop["entry_executable"] = True
    kept, coverage = _causal_coverage_fields(coverage_pop, fields)
    if kept != fields:
        raise ValueError("short LambdaRank funnel requires the full repaired 120-field contract")
    return frame, fields


def _grade(frame: pd.DataFrame, spec_name: str) -> pd.Series:
    spec = next(spec for spec in SPECS if spec.name == spec_name)
    return _to_grade(_target_values(frame, spec), spec_name=spec_name).astype(np.int32)


def _prepared(frame: pd.DataFrame, fields: list[str], spec_name: str, query_name: str) -> tuple[pd.DataFrame, pd.Series]:
    target = _grade(frame, spec_name)
    x = frame.loc[target.notna()].copy()
    target = target.loc[x.index]
    x["__target__"] = target.to_numpy(np.int32)
    definition, = query_definitions_by_name([query_name])
    x["__query__"] = assign_query_ids(x, definition)
    if MAX_TRAIN_QUERY_ROWS:
        # Do not use labels, outcomes, score, or row order to thin an
        # oversized query.  A stable SHA-256 key makes the sample exact across
        # reruns and cannot leak label information into the ranker geometry.
        key = x["candidate_id"].astype(str).map(
            lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
        )
        x["__query_sample_key__"] = key
        x = (
            x.sort_values(["__query__", "__query_sample_key__", "candidate_id"], kind="stable")
            .groupby("__query__", sort=False, observed=True)
            .head(MAX_TRAIN_QUERY_ROWS)
            .drop(columns="__query_sample_key__")
        )
    # LambdaRank cannot learn from singleton / equal-label groups; filtering is
    # query-local and uses no future data.
    grouped = x.groupby("__query__", observed=True)["__target__"]
    size = grouped.size()
    distinct = grouped.nunique()
    retained = size.index[size.ge(2) & distinct.ge(2)]
    x = x.loc[x.__query__.isin(retained)].copy()
    if x.empty:
        raise ValueError(f"{spec_name}/{query_name}: no rankable training groups")
    x = x.sort_values(["__query__", "candidate_id"], kind="stable")
    return x, target


def _fit(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    fields: list[str],
    spec_name: str,
    query_name: str,
    params: dict[str, Any],
    *,
    early: pd.DataFrame | None = None,
) -> tuple[lgb.LGBMRanker, pd.Series, np.ndarray, int]:
    train, _ = _prepared(train, fields, spec_name, query_name)
    medians = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    fit_params = dict(params)
    if "min_child_samples_fraction" in fit_params:
        fit_params = materialize_lambdarank_params(fit_params, training_rows=len(train))
    fit_params.update({"verbosity": -1, "random_state": SEED, "n_jobs": 1, "deterministic": True, "force_col_wise": True})
    model = lgb.LGBMRanker(**fit_params)
    kwargs: dict[str, Any] = {
        "group": train.groupby("__query__", sort=False).size().to_numpy(np.int32),
    }
    if early is not None and not early.empty:
        early, _ = _prepared(early, fields, spec_name, query_name)
        kwargs.update({
            "eval_set": [(_matrix(early, fields, medians), early.__target__.to_numpy(np.int32))],
            "eval_group": [early.groupby("__query__", sort=False).size().to_numpy(np.int32)],
            "callbacks": ranker_early_stopping_callbacks(rounds=EARLY_STOPPING_ROUNDS),
        })
    model.fit(_matrix(train, fields, medians), train.__target__.to_numpy(np.int32), **kwargs)
    best_iteration = int(model.best_iteration_ or fit_params["n_estimators"])
    prediction = model.predict(_matrix(predict, fields, medians), num_iteration=best_iteration)
    return model, medians, np.asarray(prediction, dtype=np.float32), best_iteration


def _chronological_inner_early(
    train: pd.DataFrame, fields: list[str], spec_name: str, query_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Reserve the latest 20% of complete training queries for early stopping.

    This split is deliberately query-disjoint and entirely inside the outer
    training fold.  The outer validation month therefore remains available
    only for target/query/HPO selection, never for LightGBM stopping.
    """
    prepared, _ = _prepared(train, fields, spec_name, query_name)
    query_time = prepared.groupby("__query__", sort=False)["__ts__"].min().sort_values(kind="stable")
    count = max(1, int(math.ceil(len(query_time) * 0.20)))
    early_queries = set(query_time.index[-count:])
    early = prepared.loc[prepared.__query__.isin(early_queries)].copy()
    fit = prepared.loc[~prepared.__query__.isin(early_queries)].copy()
    # A tiny early period can leave fewer than two fit queries.  In that case
    # fitting all strict-training rows without early stopping is safer than a
    # malformed split; the receipt makes the fallback explicit.
    if fit.empty or fit.__query__.nunique() < 2 or early.empty:
        return prepared, None
    return fit, early


def _fixed_params(training_rows: int, median_query_size: float) -> dict[str, Any]:
    retained = 0.05
    # Uses the common LambdaRank materializer so min-leaf support is a training
    # population fraction, not a misleading fixed row count.
    base = {
        "objective": "lambdarank", "metric": "ndcg", "learning_rate": 0.03,
        "lambdarank_norm": True, "bagging_freq": 1, "bagging_by_query": True,
        "path_smooth": 3.0, "n_estimators": 220, "max_depth": 4, "num_leaves": 15,
        "min_child_samples_fraction": 0.01, "min_sum_hessian_in_leaf": 1.0,
        "min_gain_to_split": 0.0, "feature_fraction": 0.80, "bagging_fraction": 0.80,
        "lambda_l1": 0.0, "lambda_l2": 3.0, "max_bin": 63,
        "lambdarank_truncation_level": max(3, min(16, int(math.ceil(max(1.0, median_query_size * retained))) + 3)),
        "label_gain": [0, 0.25, 1, 3, 7, 12],
    }
    return materialize_lambdarank_params(base, training_rows=training_rows)


def _development_folds(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    bounds = (
        ("2024-02", "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z"),
        ("2024-03", "2024-03-01T00:00:00Z", "2024-04-01T00:00:00Z"),
    )
    folds = []
    for name, validation_start, validation_end in bounds:
        start = pd.Timestamp(validation_start)
        end = pd.Timestamp(validation_end)
        train = frame.loc[
            frame.__ts__.ge(TRAIN_START) & frame.__ts__.lt(start)
            & _valid_label(frame) & frame.__label_available_at__.lt(start)
        ].copy()
        validation = frame.loc[
            frame.__ts__.ge(start) & frame.__ts__.lt(end) & _valid_label(frame)
        ].copy()
        if train.empty or validation.empty:
            raise ValueError(f"{name}: insufficient chronological HPO support")
        folds.append((name, train, validation))
    return folds


def _eval_prediction(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    x = frame.loc[:, ["candidate_id", "__ts__", "t4_tp6_sl4_net_bps"]].copy()
    x["score"] = score
    x = x.rename(columns={"t4_tp6_sl4_net_bps": "net_bps"})
    out: dict[str, float] = {}
    for tail in (0.01, 0.02, 0.05):
        take = max(1, int(math.ceil(len(x) * tail)))
        out[f"top{int(tail * 100)}_net_bps"] = float(x.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(take).net_bps.mean())
    return out


def _screen_arm(frame: pd.DataFrame, fields: list[str], spec_name: str, query_name: str) -> dict[str, Any]:
    metrics = []
    for month, train, validation in _development_folds(frame):
        proxy = _sample(train, PROXY_ROWS)
        definition, = query_definitions_by_name([query_name])
        q = assign_query_ids(proxy, definition)
        params = _fixed_params(len(proxy), float(q.groupby(q, observed=True).size().median()))
        fit, early = _chronological_inner_early(proxy, fields, spec_name, query_name)
        _, _, score, iteration = _fit(fit, validation, fields, spec_name, query_name, params, early=early)
        metrics.append((month, _eval_prediction(validation, score), iteration))
    evs = [values["top5_net_bps"] for _, values, _ in metrics]
    summary = era_portability_summary(evs)
    return {
        "arm": f"{spec_name}::{query_name}", "spec": spec_name, "query": query_name,
        "stage": "fixed_ranker_screen", "mean_best_iteration": float(np.mean([i for _, _, i in metrics])),
        **summary,
        **{f"{month}_{key}": value for month, values, _ in metrics for key, value in values.items()},
        "top1_net_bps": float(np.mean([values["top1_net_bps"] for _, values, _ in metrics])),
        "top2_net_bps": float(np.mean([values["top2_net_bps"] for _, values, _ in metrics])),
        "top5_net_bps": float(np.mean(evs)),
        "month_mad_net_bps": float(np.median(np.abs(np.asarray(evs) - np.median(evs)))),
        "month_worst_net_bps": float(np.min(evs)),
        "adjusted_hpo_score": adjusted_hpo_score(era_evs=evs, max_depth=4, num_leaves=15),
    }


def _hpo_arm(frame: pd.DataFrame, fields: list[str], spec_name: str, query_name: str, trials: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folds = _development_folds(frame)
    sample = _sample(folds[-1][1], PROXY_ROWS)
    definition, = query_definitions_by_name([query_name])
    median_size = float(assign_query_ids(sample, definition).groupby(assign_query_ids(sample, definition), observed=True).size().median())
    arm_seed = int(hashlib.sha256(f"{spec_name}|{query_name}".encode()).hexdigest()[:8], 16)
    study = make_pruned_study(seed=SEED + arm_seed % 10000, n_startup_trials=2, n_warmup_steps=1)

    def objective(trial: optuna.Trial) -> float:
        suggested = suggest_broad_lambdarank_params(trial, retained_fraction=.05, median_candidates_per_query=median_size)
        values: list[float] = []
        fold_metrics: dict[str, float] = {}
        iterations: list[int] = []
        for month, train, validation in folds:
            proxy = _sample(train, PROXY_ROWS)
            fit, early = _chronological_inner_early(proxy, fields, spec_name, query_name)
            _, _, score, iteration = _fit(fit, validation, fields, spec_name, query_name, suggested, early=early)
            iterations.append(iteration)
            current = _eval_prediction(validation, score)
            values.append(current["top5_net_bps"])
            fold_metrics.update({f"{month}_{key}": value for key, value in current.items()})
            report_portability_progress(trial, values)
        summary = era_portability_summary(values)
        for key, value in {**fold_metrics, **summary, "mean_best_iteration": float(np.mean(iterations))}.items():
            trial.set_user_attr(key, value)
        return adjusted_hpo_score(era_evs=values, max_depth=int(suggested["max_depth"]), num_leaves=int(suggested["num_leaves"]))

    study.optimize(
        objective,
        n_trials=trials,
        show_progress_bar=False,
        callbacks=[stop_after_no_improvement(patience=NO_IMPROVEMENT_PATIENCE)],
    )
    rows = []
    for trial in study.trials:
        rows.append({
            "spec": spec_name, "query": query_name, "trial": trial.number,
            "state": trial.state.name, "adjusted_hpo_score": trial.value,
            **trial.params, **{f"metric_{key}": value for key, value in trial.user_attrs.items()},
        })
    best = study.best_trial
    return rows, {
        "spec": spec_name, "query": query_name, "trial": best.number,
        "adjusted_hpo_score": best.value, "params_json": json.dumps(best.params, sort_keys=True),
        "mean_best_iteration": float(best.user_attrs["mean_best_iteration"]),
        "stop_reason": study.user_attrs.get("stop_reason", "trial_budget"),
        "no_improvement_patience": int(NO_IMPROVEMENT_PATIENCE),
        **{f"metric_{key}": value for key, value in best.user_attrs.items()},
    }


def _rank_oos_metrics(frame: pd.DataFrame, spec_name: str, *, scope: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = _valid_label(frame)
    resolved = frame.loc[valid].copy()
    result = {
        "spec": spec_name,
        "family": next(spec.family for spec in SPECS if spec.name == spec_name),
        "scope": scope,
        "scored_executable_rows": int(len(frame)),
        "resolved_rows": int(valid.sum()),
        "resolved_fraction": float(valid.mean()),
        "score_net_bps_spearman": float(resolved.score.corr(pd.to_numeric(resolved.t4_tp6_sl4_net_bps, errors="coerce"), method="spearman")),
    }
    tails: list[dict[str, Any]] = []
    ordered = frame.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
    for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
        chosen = ordered.iloc[: max(1, int(math.ceil(len(ordered) * fraction)))]
        outcome = chosen.loc[_valid_label(chosen)]
        tails.append({
            "spec": spec_name, "family": result["family"], "scope": scope,
            "tail_fraction": fraction, "tail_rows_requested": int(len(chosen)),
            "tail_rows_resolved": int(len(outcome)), "tail_label_coverage": float(len(outcome) / len(chosen)),
            "mean_score": float(chosen.score.mean()),
            "mean_gross_bps": float(pd.to_numeric(outcome.t4_tp6_sl4_gross_bps, errors="coerce").mean()),
            "mean_net_bps": float(pd.to_numeric(outcome.t4_tp6_sl4_net_bps, errors="coerce").mean()),
            "median_net_bps": float(pd.to_numeric(outcome.t4_tp6_sl4_net_bps, errors="coerce").median()),
        })
    return result, tails


def _final_oos(
    frame: pd.DataFrame, fields: list[str], spec_name: str, query_name: str,
    params_json: str, final_estimators: int, model_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = restore_broad_lambdarank_params(json.loads(params_json))
    params["n_estimators"] = int(final_estimators)
    train = frame.loc[
        frame.__ts__.ge(TRAIN_START) & frame.__ts__.lt(OOS_START)
        & _valid_label(frame) & frame.__label_available_at__.lt(OOS_START)
    ].copy()
    test = frame.loc[frame.__ts__.ge(OOS_START) & frame.__ts__.lt(OOS_END)].copy()
    model, _, score, iteration = _fit(train, test, fields, spec_name, query_name, params)
    prediction = test.loc[:, [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "label_valid", "target_invalid", "invalid_reason",
        "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "t2_tp6_sl4_event", "robust_clear_event_b25",
        "r3_b50_robust_clear", "r3_b75_robust_clear", "r3_b50_soft_adverse", "r3_b50_soft_weak", "r3_b50_soft_clear",
        "r3_b75_soft_adverse", "r3_b75_soft_weak", "r3_b75_soft_clear",
    ]].copy()
    prediction["score"] = score
    model.booster_.save_model(str(model_path))
    total, tails = _rank_oos_metrics(prediction, spec_name, scope="2024-04_to_2024-06")
    prediction["month"] = prediction.__ts__.dt.strftime("%Y-%m")
    month_metrics = []
    month_tails = []
    for month, group in prediction.groupby("month", sort=True):
        current, current_tails = _rank_oos_metrics(group, spec_name, scope=str(month))
        month_metrics.append(current)
        month_tails.extend(current_tails)
    total["final_iteration"] = iteration
    return prediction, {"global": total, "months": month_metrics, "tails": tails + month_tails}


def run(*, features: Path, labels: Path, candidates: Path, query_root: Path, out: Path, hpo_trials: int, screen_top: int) -> Path:
    if out.exists():
        raise FileExistsError(f"output must be new: {out}")
    frame, fields = _ledger(features, labels, candidates)
    out.mkdir(parents=True)
    screen_rows: list[dict[str, Any]] = []
    for spec in SPECS:
        shortlist = load_frozen_query_shortlist(query_root / f"grade_{spec.name}" / "query_shortlist.json")
        for query in shortlist:
            screen_rows.append(_screen_arm(frame, fields, spec.name, query))
    screen = pd.DataFrame(screen_rows).sort_values(["adjusted_hpo_score", "top1_net_bps", "arm"], ascending=[False, False, True], kind="stable")
    finalists = screen.head(screen_top).copy()
    hpo_rows: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    for row in finalists.itertuples(index=False):
        trials, winner = _hpo_arm(frame, fields, row.spec, row.query, hpo_trials)
        hpo_rows.extend(trials)
        winners.append(winner)
    final_predictions: list[pd.DataFrame] = []
    final_metrics: list[dict[str, Any]] = []
    final_tails: list[dict[str, Any]] = []
    for winner in winners:
        model_name = f"final_model_{winner['spec']}__{winner['query']}.txt"
        prediction, metrics = _final_oos(
            frame, fields, winner["spec"], winner["query"], winner["params_json"],
            int(round(float(winner["mean_best_iteration"]))), out / model_name,
        )
        prediction["arm"] = winner["spec"] + "::" + winner["query"]
        final_predictions.append(prediction)
        for current in [metrics["global"], *metrics["months"]]:
            final_metrics.append({"arm": prediction.arm.iloc[0], **current})
        for current in metrics["tails"]:
            final_tails.append({"arm": prediction.arm.iloc[0], **current})
    screen.to_parquet(out / "specialist_target_query_screen.parquet", index=False, compression="zstd")
    finalists.to_parquet(out / "specialist_target_query_finalists.parquet", index=False, compression="zstd")
    pd.DataFrame(hpo_rows).to_parquet(out / "specialist_hpo_trials.parquet", index=False, compression="zstd")
    pd.DataFrame(winners).to_parquet(out / "specialist_target_query_winners.parquet", index=False, compression="zstd")
    pd.concat(final_predictions, ignore_index=True).to_parquet(out / "final_oos_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(final_metrics).to_parquet(out / "final_oos_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(final_tails).to_parquet(out / "final_oos_tail_metrics.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_specialist_target_query_hpo_v1", "side": "short",
        "features": str(features), "labels": str(labels), "candidates": str(candidates), "query_root": str(query_root),
        "feature_contract": "short base_fields_by_side.short", "feature_count": len(fields),
        "screen_folds": ["train Jan -> validate Feb", "train Jan-Feb -> validate Mar"],
        "final_refit": "train Jan-Mar, label_available_at < Apr 1; one OOS April-June evaluation",
        "hpo": "broad residual_lambdarank_hpo space; deterministic proxy sampling; 30-round early stopping; median pruner",
        "ranker_train_query_cap": MAX_TRAIN_QUERY_ROWS,
        "ranker_query_cap_method": "smallest SHA-256(candidate_id) hashes within each decision-time query; no label/outcome use",
        "screen_top_arms": screen_top, "hpo_trials_per_arm": hpo_trials,
        "no_promotion": "final OOS results are research evidence only; this runner does not promote a short base/specialist/residual model.",
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--query-root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hpo-trials", type=int, default=HPO_TRIALS)
    parser.add_argument("--screen-top", type=int, default=SCREEN_TOP_ARMS)
    args = parser.parse_args()
    print(run(features=args.features, labels=args.labels, candidates=args.candidates, query_root=args.query_root, out=args.out, hpo_trials=args.hpo_trials, screen_top=args.screen_top))


if __name__ == "__main__":
    main()
