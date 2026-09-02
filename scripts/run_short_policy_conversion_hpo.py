#!/usr/bin/env python3
"""Chronological, query-sampled LambdaRank HPO for the short base target.

The utility consumes only pre-declared short-side development folds.  It
samples *complete timestamp queries*, never rows; keeps the held folds wholly
out of fitting; uses an earlier inner chronological validation slice solely
for early stopping; then refits the sampled outer training history before
scoring the outer OOS fold.  This avoids the common but invalid shortcut of
tuning on in-sample base predictions.
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

from extreme_price_movements.residual_lambdarank_hpo import (
    adjusted_hpo_score,
    era_portability_summary,
    ranker_early_stopping_callbacks,
)
from scripts.run_short_policy_conversion_funnel import (
    GAIN_FAMILIES,
    SIDE,
    PolicySpec,
    _coverage_fields,
    _load_candidates,
    _load_features,
    _load_policy_ledger,
    _load_supportive_ledger,
    _matrix,
    _query_order,
    _sample_weights,
    _targets,
    _valid_policy,
)


FOLDS = (
    ("mayjun", "2024-05-01", "2024-07-01"),
    ("julaug", "2024-07-01", "2024-09-01"),
    ("sep", "2024-09-01", "2024-10-01"),
)
TOP_KS = (1, 2, 4, 8, 16)
SCREEN_WEIGHTS = {1: .40, 2: .25, 4: .20, 8: .10, 16: .05}
SEED = 1729


def make_no_improvement_callback(patience: int):
    """Return a strict maximizing-study stagnation callback.

    A completed or pruned trial counts toward the patience budget unless it
    establishes a strictly larger final objective.  Failed trials are not
    comparable observations and are deliberately ignored.
    """
    if patience < 1:
        raise ValueError("patience must be positive")

    def callback(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        comparable = [
            item for item in study_.trials
            if item.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
        ]
        completed = [
            item for item in comparable
            if item.state == optuna.trial.TrialState.COMPLETE and item.value is not None
        ]
        if not completed:
            return
        # ``max`` retains the first occurrence of an equal value.  A tie is
        # intentionally *not* an improvement and does not reset patience.
        best = max(completed, key=lambda item: float(item.value))
        last_improvement_index = next(
            index
            for index, item in reversed(list(enumerate(comparable)))
            if item.number == best.number
        )
        non_improving = len(comparable) - 1 - last_improvement_index
        study_.set_user_attr("last_improving_trial", int(best.number))
        study_.set_user_attr("consecutive_non_improving_trials", int(non_improving))
        if non_improving >= int(patience):
            study_.set_user_attr("stop_reason", "no_improvement_patience")
            study_.set_user_attr("no_improvement_patience", int(patience))
            study_.stop()

    return callback


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sample_complete_queries(frame: pd.DataFrame, *, fraction: float, seed: int) -> pd.DataFrame:
    """Stable, month-stratified sampling of whole LambdaRank queries."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sample fraction must be in (0, 1]")
    work = frame.copy()
    work["__query__"] = work["__ts__"].dt.floor("1h")
    work["__month__"] = work["__ts__"].dt.strftime("%Y-%m")
    chosen: list[pd.Timestamp] = []
    for _, group in work.loc[:, ["__month__", "__query__"]].drop_duplicates().groupby("__month__", sort=True):
        keys = group["__query__"].tolist()
        take = max(1, int(math.ceil(len(keys) * fraction)))
        keys.sort(key=lambda value: hashlib.sha256(f"{seed}|{value.isoformat()}".encode()).hexdigest())
        chosen.extend(keys[:take])
    return work.loc[work["__query__"].isin(chosen)].drop(columns="__month__").copy()


def _prepare(ledger: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp, spec: PolicySpec) -> pd.DataFrame:
    result = ledger.loc[
        ledger["__ts__"].ge(start)
        & ledger["__ts__"].lt(end)
        & ledger.entry_executable.astype(bool)
    ].copy()
    result = result.loc[
        result.policy_label_available_at.lt(end) | result.__label_available_at__.lt(end)
    ].copy()
    # A target-specific validity check happens again in _targets; retaining the
    # target-free rows here is intentional for held-score population identity.
    if result.empty:
        raise ValueError("empty chronological training population")
    return result


def _hpo_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": trial.suggest_float("learning_rate", .015, .06, log=True),
        "n_estimators": 1500,
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "num_leaves": trial.suggest_int("num_leaves", 15, 61),
        "min_child_samples_fraction": trial.suggest_float("min_child_samples_fraction", .005, .03),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", .1, 30.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-4, .01, log=True),
        "colsample_bytree": trial.suggest_float("feature_fraction", .70, .90),
        "subsample": trial.suggest_float("bagging_fraction", .70, .90),
        "subsample_freq": 1,
        "bagging_by_query": True,
        "path_smooth": 3.0,
        "reg_alpha": trial.suggest_float("lambda_l1", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("lambda_l2", .1, 30.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127]),
        "lambdarank_norm": True,
        "lambdarank_truncation_level": 40,
        "label_gain": GAIN_FAMILIES["linear"],
        "random_state": SEED,
        "seed": SEED,
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _materialize_params(suggested: dict[str, Any], *, rows: int, estimators: int | None = None) -> dict[str, Any]:
    params = dict(suggested)
    fraction = float(params.pop("min_child_samples_fraction"))
    params["min_child_samples"] = max(2, int(math.ceil(rows * fraction)))
    if estimators is not None:
        params["n_estimators"] = int(estimators)
    return params


def _fit_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    fields: list[str],
    spec: PolicySpec,
    params: dict[str, Any],
    outer_start: pd.Timestamp,
    sample_fraction: float,
) -> tuple[np.ndarray, int]:
    """Inner early stopping, then sampled-outer refit and held-OOS scoring."""
    inner_start = outer_start - pd.Timedelta(days=28)
    inner_fit = train.loc[train["__ts__"].lt(inner_start)].copy()
    inner_valid = train.loc[train["__ts__"].ge(inner_start)].copy()
    # Inner fitting cannot consume labels not yet resolved at its own boundary.
    inner_fit = inner_fit.loc[
        inner_fit.policy_label_available_at.lt(inner_start) | inner_fit.__label_available_at__.lt(inner_start)
    ].copy()
    if inner_fit.empty or inner_valid.empty:
        raise ValueError("insufficient inner chronological support for early stopping")
    inner_fit = _sample_complete_queries(inner_fit, fraction=sample_fraction, seed=SEED)
    full_outer = _sample_complete_queries(train, fraction=sample_fraction, seed=SEED)

    def _ordered(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        work = frame.copy()
        work["__query__"] = work["__ts__"].dt.floor("1h")
        target = _targets(work, spec)
        return _query_order(work, target)

    fit_ordered, fit_groups, fit_y = _ordered(inner_fit)
    valid_ordered, valid_groups, valid_y = _ordered(inner_valid)
    outer_ordered, outer_groups, outer_y = _ordered(full_outer)
    medians = outer_ordered.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    if medians.isna().any():
        raise AssertionError("HPO median imputation has an empty selected feature")
    fit_weight = _sample_weights(fit_ordered, kind=spec.weight_kind, train_end=inner_start)
    estimator = lgb.LGBMRanker(**_materialize_params(params, rows=len(fit_ordered)))
    estimator.fit(
        _matrix(fit_ordered, fields, medians), fit_y, group=fit_groups,
        sample_weight=fit_weight,
        eval_set=[(_matrix(valid_ordered, fields, medians), valid_y)],
        eval_group=[valid_groups], eval_sample_weight=[_sample_weights(valid_ordered, kind=spec.weight_kind, train_end=outer_start)],
        eval_at=[4, 8],
        callbacks=ranker_early_stopping_callbacks(rounds=75),
    )
    best_iteration = int(estimator.best_iteration_ or estimator.n_estimators_ or params["n_estimators"])
    outer_weight = _sample_weights(outer_ordered, kind=spec.weight_kind, train_end=outer_start)
    refit = lgb.LGBMRanker(**_materialize_params(params, rows=len(outer_ordered), estimators=best_iteration))
    refit.fit(_matrix(outer_ordered, fields, medians), outer_y, group=outer_groups, sample_weight=outer_weight)
    return refit.predict(_matrix(test, fields, medians)).astype(np.float32), best_iteration


def _economic_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    scored = frame.loc[:, ["candidate_id", "__ts__", "p0_canonical_net_bps"]].copy()
    scored["score"] = score
    scored["p0_canonical_net_bps"] = pd.to_numeric(scored.p0_canonical_net_bps, errors="coerce")
    scored = scored.loc[_valid_policy(frame)].copy()
    ic_values: list[float] = []
    values = {key: [] for key in TOP_KS}
    uplift = {key: [] for key in TOP_KS}
    for _, group in scored.groupby("__ts__", sort=False):
        if len(group) < 2:
            continue
        ic_values.append(float(group.score.corr(group.p0_canonical_net_bps, method="spearman")))
        baseline = float(group.p0_canonical_net_bps.mean())
        ordered = group.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
        for key in TOP_KS:
            selected = ordered.head(min(key, len(ordered)))
            value = float(selected.p0_canonical_net_bps.mean())
            values[key].append(value)
            uplift[key].append(value - baseline)
    result = {
        "query_count": float(len(ic_values)),
        "policy_ic_mean": float(np.nanmean(ic_values)),
        "policy_ic_positive_fraction": float(np.mean(np.asarray(ic_values) > 0.0)),
    }
    for key in TOP_KS:
        result[f"top_{key}_net_bps"] = float(np.mean(values[key]))
        result[f"top_{key}_uplift_bps"] = float(np.mean(uplift[key]))
    result["economic_screen_bps"] = float(sum(SCREEN_WEIGHTS[key] * result[f"top_{key}_uplift_bps"] for key in TOP_KS))
    return result


def _load_ledger(*, fields: list[str], policies: Path, features: Path, candidates: Path, supportive_path: Path) -> pd.DataFrame:
    start, end = _utc("2023-10-01"), _utc("2024-10-01")
    candidate = _load_candidates(candidates, SIDE)
    candidate = candidate.loc[candidate.__ts__.ge(start) & candidate.__ts__.lt(end)].copy()
    feature = _load_features(features, fields, candidate, SIDE)
    policy = _load_policy_ledger(policies, start, end)
    support = _load_supportive_ledger(supportive_path, start, end)
    ledger = feature.merge(policy, on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    ledger = ledger.merge(support, on="candidate_id", how="left", validate="one_to_one")
    if len(ledger) != len(feature):
        raise AssertionError("HPO label joins altered target-free candidate identities")
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "selection", "policies", "features", "candidates", "supportive-path"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--target", choices=("activation_grade", "policy_bps"), default="activation_grade")
    parser.add_argument("--weight-kind", choices=("uniform", "month_query"), default="month_query")
    parser.add_argument("--folds", default="mayjun,julaug", help="comma-separated predeclared fold names")
    parser.add_argument("--sample-fraction", type=float, default=.35)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument(
        "--no-improvement-patience",
        type=int,
        default=20,
        help=(
            "Stop after this many completed or pruned trials without a strictly "
            "better adjusted portability score (default: 20)."
        ),
    )
    parser.add_argument("--study-name", default="short_policy_conversion_broad_hpo")
    parser.add_argument("--finalize-existing", action="store_true", help="Write the winner/manifest from an existing stopped study without starting another trial.")
    args = parser.parse_args()
    if args.trials <= 0 and not args.finalize_existing:
        raise ValueError("trials must be positive")
    if args.no_improvement_patience < 1:
        raise ValueError("no-improvement-patience must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=bool(args.finalize_existing))
    fields = json.loads(args.selection.read_text())["feature_sets"]["90"]
    requested = tuple(value for value in args.folds.split(",") if value)
    fold_lookup = {name: (_utc(start), _utc(end)) for name, start, end in FOLDS}
    if not requested or set(requested).difference(fold_lookup):
        raise ValueError(f"unknown predeclared folds: {requested}")
    spec = PolicySpec(
        f"hpo_{args.target}_{args.weight_kind}", "HPO candidate", args.target,
        truncation=40, gain_family="linear", query_hours=1, weight_kind=args.weight_kind,
    )
    ledger = _load_ledger(
        fields=fields, policies=args.policies.resolve(), features=args.features.resolve(),
        candidates=args.candidates.resolve(), supportive_path=getattr(args, "supportive_path").resolve(),
    )
    # Assert the F90 feature coverage against target-free entry-executable rows
    # in the broadest pre-selection population before beginning optimisation.
    coverage_train = ledger.loc[ledger.__ts__.lt(_utc("2024-05-01"))]
    kept, coverage = _coverage_fields(coverage_train, fields)
    if kept != fields:
        raise AssertionError("F90 contract failed its target-free coverage gate")

    storage = f"sqlite:///{out / 'optuna.sqlite3'}"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED), pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        suggested = _hpo_params(trial)
        fold_rows: list[dict[str, Any]] = []
        scores: list[float] = []
        for index, name in enumerate(requested, start=1):
            start, end = fold_lookup[name]
            train = _prepare(ledger, start=_utc("2023-10-01"), end=start, spec=spec)
            test = ledger.loc[
                ledger.__ts__.ge(start) & ledger.__ts__.lt(end) & ledger.entry_executable.astype(bool)
            ].copy()
            if test.empty:
                raise ValueError(f"empty held population: {name}")
            predicted, best_iteration = _fit_score(
                train, test, fields=fields, spec=spec, params=suggested,
                outer_start=start, sample_fraction=args.sample_fraction,
            )
            metrics = _economic_metrics(test, predicted)
            metrics.update({"fold": name, "best_iteration": best_iteration})
            fold_rows.append(metrics)
            scores.append(metrics["economic_screen_bps"])
            # Do not prune after the first historical era: a configuration must
            # first demonstrate that it is not a one-fold coincidence.
            if index >= 2:
                portability = era_portability_summary(scores)["portability_score_bps"]
                trial.report(float(portability), step=index)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        adjusted = adjusted_hpo_score(
            era_evs=scores, max_depth=int(suggested["max_depth"]),
            num_leaves=int(suggested["num_leaves"]), model_type="lambdarank",
        )
        trial.set_user_attr("fold_metrics", fold_rows)
        trial.set_user_attr("adjusted_hpo_score", float(adjusted))
        return float(adjusted)

    def receipt(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        rows: list[dict[str, Any]] = []
        for item in study_.trials:
            row: dict[str, Any] = {"trial": item.number, "state": item.state.name, "value": item.value, **item.params}
            row.update(item.user_attrs)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(out / "trial_receipts.parquet", index=False, compression="zstd")

    interrupted = False
    if not args.finalize_existing:
        try:
            study.optimize(
                objective,
                n_trials=args.trials,
                callbacks=[make_no_improvement_callback(args.no_improvement_patience), receipt],
                gc_after_trial=True,
            )
        except KeyboardInterrupt:
            # A deliberate user stop preserves completed trials.  We still
            # write a sealed partial-study receipt below; no incomplete trial
            # can become the winner.
            interrupted = True
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("no completed HPO trial")
    best = max(complete, key=lambda trial: float(trial.value or -np.inf))
    stop_reason = str(study.user_attrs.get("stop_reason", ""))
    study_status = (
        "manual_early_stop" if (interrupted or args.finalize_existing)
        else "stagnation_early_stop" if stop_reason == "no_improvement_patience"
        else "complete"
    )
    (out / "winner.json").write_text(json.dumps({
        "trial": best.number, "adjusted_hpo_score": best.value, "params": best.params,
        "fold_metrics": best.user_attrs.get("fold_metrics", []),
        "feature_contract": fields, "target": args.target, "weight_kind": args.weight_kind,
        "study_status": study_status,
        "stop_reason": stop_reason or None,
    }, indent=2) + "\n")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_policy_conversion_hpo_v1",
        "status": study_status, "selection_window": "pre-2024-10 only",
        "target": args.target, "weight_kind": args.weight_kind,
        "folds": list(requested), "sample_fraction": args.sample_fraction,
        "query_sampling": "stable month-stratified whole timestamp queries only",
        "inner_validation": "latest 28 calendar days before each outer OOS fold",
        "outer_refit": "sampled prequential outer training history after inner early stopping",
        "search_space": "broad LambdaRank geometry/regularisation; linear gains and K40 held fixed",
        "feature_count": len(fields), "feature_coverage": {field: float(coverage[field]) for field in fields},
        "requested_trials": args.trials,
        "completed_trials": len(complete),
        "no_improvement_patience": int(args.no_improvement_patience),
        "last_improving_trial": study.user_attrs.get("last_improving_trial"),
        "consecutive_non_improving_trials": study.user_attrs.get("consecutive_non_improving_trials"),
        "stop_reason": stop_reason or None,
        "finalize_existing": bool(args.finalize_existing),
    }, indent=2) + "\n")
    print(json.dumps({"winner_trial": best.number, "score": best.value, "params": best.params}, indent=2))


if __name__ == "__main__":
    main()
