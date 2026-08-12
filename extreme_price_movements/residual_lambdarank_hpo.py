"""Reusable, portability-scored Optuna support for residual LambdaRank.

The utility intentionally contains no dataframe or fold policy.  Callers pass
chronological validation-era EVs, so the same objective can be reused without
accidentally selecting a configuration on pooled (and potentially unstable)
performance alone.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _tail_label(tail: float) -> str:
    """Return a stable, human-readable label for a fractional tail."""
    value = float(tail) * 100.0
    if value.is_integer():
        return f"top{int(value)}"
    return f"top{value:g}".replace(".", "_")


def downstream_tail_summary(
    frame: pd.DataFrame,
    *,
    score_column: str,
    tails: Sequence[float] = (.005, .01, .02, .05, .10),
    net_column: str = "net_bps",
    gross_column: str = "gross_bps",
    timestamp_column: str = "__ts__",
    candidate_id_column: str = "candidate_id",
) -> dict[str, float]:
    """Return pooled and monthly globally-ranked economics for one score.

    This helper deliberately ranks the *complete supplied population* once per
    pooled and monthly evaluation.  It does not rank per timestamp or per
    query, because the policy selection in this research path is global after
    upstream common-bps mapping.  Callers are responsible for supplying only
    the fixed comparison population.
    """
    required = {
        score_column, net_column, gross_column, timestamp_column,
        candidate_id_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"downstream tail summary missing {sorted(missing)}")
    if not tails or any(not 0.0 < float(tail) <= 1.0 for tail in tails):
        raise ValueError("tails must be non-empty fractions in (0, 1]")

    x = frame.loc[:, [score_column, net_column, gross_column, timestamp_column, candidate_id_column]].copy()
    x[score_column] = pd.to_numeric(x[score_column], errors="coerce")
    x[net_column] = pd.to_numeric(x[net_column], errors="coerce")
    x[gross_column] = pd.to_numeric(x[gross_column], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_column, net_column, gross_column])
    if x.empty:
        return {"rows": 0.0}

    def _one(population: pd.DataFrame) -> dict[str, float]:
        ordered = population.sort_values(
            [score_column, candidate_id_column], ascending=[False, True], kind="stable"
        )
        values: dict[str, float] = {}
        for tail in tails:
            label = _tail_label(float(tail))
            n = max(1, int(math.ceil(len(ordered) * float(tail))))
            chosen = ordered.head(n)
            values[f"{label}_rows"] = float(n)
            values[f"{label}_net_bps"] = float(chosen[net_column].mean())
            values[f"{label}_gross_bps"] = float(chosen[gross_column].mean())
            values[f"{label}_net_sum_bps"] = float(chosen[net_column].sum())
        return values

    out: dict[str, float] = {"rows": float(len(x)), **_one(x)}
    months = pd.to_datetime(x[timestamp_column], utc=True, errors="raise").dt.strftime("%Y-%m")
    for tail in tails:
        label = _tail_label(float(tail))
        monthly = [
            _one(group)[f"{label}_net_bps"]
            for _, group in x.assign(__month__=months).groupby("__month__", sort=True, observed=True)
        ]
        values = np.asarray(monthly, dtype=float)
        if values.size:
            median = float(np.median(values))
            out[f"{label}_month_count"] = float(values.size)
            out[f"{label}_month_mean_net_bps"] = float(values.mean())
            out[f"{label}_month_worst_net_bps"] = float(values.min())
            out[f"{label}_month_mad_net_bps"] = float(np.median(np.abs(values - median)))
            out[f"{label}_month_std_net_bps"] = float(values.std(ddof=0))
    return out


def conditional_downstream_summary(
    frame: pd.DataFrame,
    *,
    candidate_score_column: str,
    incumbent_score_column: str,
    tails: Sequence[float] = (.01, .02, .05),
    net_column: str = "net_bps",
    gross_column: str = "gross_bps",
    timestamp_column: str = "__ts__",
    candidate_id_column: str = "candidate_id",
) -> dict[str, float]:
    """Score a proposed head change conditional on its fixed nine-head stack.

    The primary utility favours the requested global Top-1/2/5% economics,
    then gives a smaller reward to the corresponding worst month and a modest
    penalty for monthly dispersion.  Its components are emitted separately so
    that promotion never rests on a hidden scalar.  This is deliberately a
    *downstream* score: it measures the final stack score, not a head's own
    NDCG, IC, or loss.
    """
    if tuple(float(t) for t in tails) != (.01, .02, .05):
        # The utility is intentionally calibrated to the declared selection
        # contract.  Other tails can still be reported with downstream_tail_summary.
        raise ValueError("conditional downstream utility is defined for Top-1/2/5%")
    candidate = downstream_tail_summary(
        frame, score_column=candidate_score_column, tails=tails,
        net_column=net_column, gross_column=gross_column,
        timestamp_column=timestamp_column, candidate_id_column=candidate_id_column,
    )
    incumbent = downstream_tail_summary(
        frame, score_column=incumbent_score_column, tails=tails,
        net_column=net_column, gross_column=gross_column,
        timestamp_column=timestamp_column, candidate_id_column=candidate_id_column,
    )
    if candidate.get("rows", 0.0) != incumbent.get("rows", 0.0):
        raise AssertionError("conditional comparison changed the candidate population")
    weights = {"top1": .30, "top2": .30, "top5": .40}
    pooled = sum(weights[label] * candidate[f"{label}_net_bps"] for label in weights)
    pooled_incumbent = sum(weights[label] * incumbent[f"{label}_net_bps"] for label in weights)
    worst = sum(weights[label] * candidate[f"{label}_month_worst_net_bps"] for label in weights)
    worst_incumbent = sum(weights[label] * incumbent[f"{label}_month_worst_net_bps"] for label in weights)
    mad = sum(weights[label] * candidate[f"{label}_month_mad_net_bps"] for label in weights)
    mad_incumbent = sum(weights[label] * incumbent[f"{label}_month_mad_net_bps"] for label in weights)
    utility = pooled + .25 * worst - .10 * mad
    incumbent_utility = pooled_incumbent + .25 * worst_incumbent - .10 * mad_incumbent
    out: dict[str, float] = {
        "conditional_rows": candidate["rows"],
        "candidate_downstream_utility_bps": float(utility),
        "incumbent_downstream_utility_bps": float(incumbent_utility),
        "conditional_utility_uplift_bps": float(utility - incumbent_utility),
        "candidate_weighted_tail_net_bps": float(pooled),
        "incumbent_weighted_tail_net_bps": float(pooled_incumbent),
        "candidate_weighted_worst_month_net_bps": float(worst),
        "incumbent_weighted_worst_month_net_bps": float(worst_incumbent),
        "candidate_weighted_month_mad_bps": float(mad),
        "incumbent_weighted_month_mad_bps": float(mad_incumbent),
    }
    for label in weights:
        for suffix in ("net_bps", "month_worst_net_bps", "month_mad_net_bps"):
            out[f"candidate_{label}_{suffix}"] = candidate[f"{label}_{suffix}"]
            out[f"incumbent_{label}_{suffix}"] = incumbent[f"{label}_{suffix}"]
            out[f"delta_{label}_{suffix}"] = candidate[f"{label}_{suffix}"] - incumbent[f"{label}_{suffix}"]
    return out


def passes_conditional_promotion(
    summary: dict[str, float], *,
    minimum_uplift_bps: float = 0.0,
) -> bool:
    """Require every requested global tail and Top-5 worst month to improve.

    A candidate may still be reported as the exploratory utility winner when
    this returns false, but it must not silently replace the incumbent.  The
    deliberately strict gate reflects a ten-head ensemble: a change that wins
    only after harming another selected tail is not a clean conditional win.
    """
    required = (
        "delta_top1_net_bps", "delta_top2_net_bps", "delta_top5_net_bps",
        "delta_top5_month_worst_net_bps", "conditional_utility_uplift_bps",
    )
    missing = [name for name in required if name not in summary]
    if missing:
        raise KeyError(f"conditional promotion summary missing {missing}")
    return bool(all(float(summary[name]) > float(minimum_uplift_bps) for name in required))


def portability_score(era_evs: Sequence[float]) -> float:
    """Return median EV less dispersion and negative-worst-era penalties."""
    values = sorted(float(x) for x in era_evs)
    if not values:
        raise ValueError("At least one era EV is required.")
    n = len(values)
    median = values[n // 2] if n % 2 else .5 * (values[n // 2 - 1] + values[n // 2])
    deviations = sorted(abs(value - median) for value in values)
    mad = deviations[n // 2] if n % 2 else .5 * (deviations[n // 2 - 1] + deviations[n // 2])
    return median - .5 * mad - max(0.0, -min(values))


def complexity_penalty(*, max_depth: int, num_leaves: int, preferred_depth: int = 4,
                       preferred_leaves: int = 15, depth_lambda_bps: float = 1.5,
                       leaf_bps_per_doubling: float = 3.0) -> float:
    """Penalise only excess capacity, in bps/trade units."""
    depth_penalty = depth_lambda_bps * max(0, int(max_depth) - preferred_depth) ** 2
    leaf_penalty = 0.0
    if int(num_leaves) > preferred_leaves:
        leaf_penalty = leaf_bps_per_doubling * math.log2(num_leaves / preferred_leaves)
    return float(depth_penalty + leaf_penalty)


def adjusted_hpo_score(*, era_evs: Sequence[float], max_depth: int,
                       num_leaves: int, model_type: str = "lambdarank") -> float:
    """Portability score after a transparent capacity penalty."""
    if model_type.lower() != "lambdarank":
        raise ValueError("This utility is intentionally limited to LambdaRank.")
    return portability_score(era_evs) - complexity_penalty(
        max_depth=max_depth, num_leaves=num_leaves,
    )


def truncation_candidates(*, retained_fraction: float, median_candidates_per_query: float) -> list[int]:
    """Derive a small geometry-aware truncation grid from a frozen policy tail."""
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained_fraction must be in (0, 1].")
    screen_k = max(1, int(math.ceil(retained_fraction * median_candidates_per_query)))
    return sorted({
        max(3, screen_k + 2), max(4, screen_k + 3),
        max(5, int(math.ceil(1.5 * screen_k))), min(32, max(6, 2 * screen_k)),
    })


TAIL_TRUNCATION_SPACE = {1: [3, 4, 5, 6], 3: [5, 6, 8, 10], 5: [7, 8, 10, 12]}
TAIL_LABEL_GAINS = {
    "linear": [0, 1, 2, 3, 4, 5],
    "economic_step": [0, .1, 1, 3, 7, 12],
    "moderate_tail": [0, .25, 1, 3, 7, 12],
    "strong_tail": [0, .25, 1, 4, 10, 20],
    "default_exponential": [0, 1, 3, 7, 15, 31],
}


def materialize_lambdarank_params(
    suggested: dict[str, Any],
    *,
    training_rows: int,
    max_estimators: int | None = None,
) -> dict[str, Any]:
    """Convert a fold-size-independent HPO suggestion into LightGBM params.

    The public HPO space expresses ``min_data_in_leaf`` as a fraction of the
    rows available in the *current training fold*.  Keeping that fraction in
    the trial record avoids accidentally making a 0.5% support constraint mean
    something radically different for a proxy fold and its eventual refit.

    The returned dictionary is accepted by ``LGBMRanker``.  Search-only fields
    such as the human-readable gain name are intentionally excluded.
    """
    if training_rows < 2:
        raise ValueError("LambdaRank needs at least two training rows")
    params = dict(suggested)
    fraction_value = params.pop("min_child_samples_fraction", None)
    if fraction_value is None:
        fraction_value = params.pop("min_data_in_leaf", np.nan)
    fraction = float(fraction_value)
    if not .005 <= fraction <= .03:
        raise ValueError("min_child_samples_fraction must be in [.005, .03]")
    params.pop("label_gain_name", None)
    params["min_child_samples"] = max(2, int(math.ceil(training_rows * fraction)))
    if max_estimators is not None:
        if max_estimators <= 0:
            raise ValueError("max_estimators must be positive")
        params["n_estimators"] = int(max_estimators)
    return params


def ranker_early_stopping_callbacks(*, rounds: int = 30) -> list[Any]:
    """Return the one shared, quiet early-stopping contract for HPO fits."""
    if rounds <= 0:
        raise ValueError("early-stopping rounds must be positive")
    import lightgbm as lgb

    return [lgb.early_stopping(stopping_rounds=int(rounds), verbose=False)]


def era_portability_summary(era_evs: Sequence[float]) -> dict[str, float]:
    """Return transparent portability terms for a trial record."""
    values = np.asarray(list(era_evs), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("era EVs must be non-empty finite values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    worst = float(values.min())
    return {
        "era_count": int(values.size),
        "era_ev_median_bps": median,
        "era_ev_mad_bps": mad,
        "era_ev_worst_bps": worst,
        "portability_score_bps": float(median - .5 * mad - max(0.0, -worst)),
    }


def report_portability_progress(trial: Any, era_evs: Sequence[float]) -> float:
    """Report chronological progress to Optuna and prune only on seen eras."""
    summary = era_portability_summary(era_evs)
    value = float(summary["portability_score_bps"])
    trial.report(value, step=len(era_evs))
    if trial.should_prune():
        import optuna

        raise optuna.TrialPruned()
    return value


def select_portability_winner(
    table: pd.DataFrame,
    *,
    score_column: str = "adjusted_hpo_score",
    tie_tolerance_bps: float = 1.0,
) -> pd.Series:
    """Select an HPO winner by score, monthly stability, then top-1 EV.

    The primary score already contains aggregate level, dispersion, and
    worst-era penalties.  Within its declared tolerance we intentionally do
    *not* reintroduce a pooled top-5 preference: the requested tie-break is
    monthly stability first and top-1 economics second.
    """
    required = {
        score_column, "month_mad_net_bps", "month_worst_net_bps",
        "top1_net_bps", "arm",
    }
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"portability selection missing {sorted(missing)}")
    x = table.copy()
    best = float(pd.to_numeric(x[score_column], errors="coerce").max())
    x = x[pd.to_numeric(x[score_column], errors="coerce").ge(best - tie_tolerance_bps)].copy()
    return x.sort_values(
        ["month_mad_net_bps", "month_worst_net_bps", "top1_net_bps", "arm"],
        ascending=[True, False, False, True], kind="stable",
    ).iloc[0]


def make_pruned_study(*, seed: int, n_startup_trials: int = 8, n_warmup_steps: int = 1):
    """Create the requested reproducible, aggressive median-pruned study."""
    import optuna
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps,
        interval_steps=1,
    )
    return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)


def suggest_broad_lambdarank_params(trial: Any, *, retained_fraction: float,
                                    median_candidates_per_query: float) -> dict[str, Any]:
    """Suggest the requested broad but bounded LightGBM LambdaRank space.

    ``min_child_samples_fraction`` is intentionally expressed as a fraction;
    callers convert it against their *training* row count after the fold is
    materialised.  This preserves comparable support across subsamples.
    """
    def mixed_zero_log(name: str, low: float, high: float) -> float:
        return 0.0 if trial.suggest_categorical(f"{name}_zero", [True, False]) else trial.suggest_float(name, low, high, log=True)

    truncation = truncation_candidates(
        retained_fraction=retained_fraction,
        median_candidates_per_query=median_candidates_per_query,
    )
    gain_name = trial.suggest_categorical("label_gain", sorted(TAIL_LABEL_GAINS))
    return {
        "objective": "lambdarank", "metric": "ndcg", "learning_rate": .03,
        "lambdarank_norm": True, "bagging_freq": 1, "bagging_by_query": True,
        "path_smooth": 3.0, "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "num_leaves": trial.suggest_int("num_leaves", 15, 61),
        "min_child_samples_fraction": trial.suggest_float("min_data_in_leaf", .005, .03),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", .1, 30., log=True),
        "min_gain_to_split": mixed_zero_log("min_gain_to_split", 1e-4, .01),
        "feature_fraction": trial.suggest_float("feature_fraction", .7, .9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", .7, .9),
        "lambda_l1": mixed_zero_log("lambda_l1", 1e-6, 5.),
        "lambda_l2": trial.suggest_float("lambda_l2", .1, 30., log=True),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127]),
        "lambdarank_truncation_level": trial.suggest_categorical("lambdarank_truncation_level", truncation),
        "label_gain": TAIL_LABEL_GAINS[gain_name], "label_gain_name": gain_name,
    }


def restore_broad_lambdarank_params(trial_params: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one shared suggestion from Optuna's serialized trial params.

    Optuna stores the two mixed zero/log choices as separate values.  Repeating
    this reconstruction in each runner previously made winning configurations
    brittle and could leak search-only keys into LightGBM.
    """
    gain_name = str(trial_params["label_gain"])
    if gain_name not in TAIL_LABEL_GAINS:
        raise KeyError(f"unknown LambdaRank gain family {gain_name!r}")
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": .03,
        "lambdarank_norm": True,
        "bagging_freq": 1,
        "bagging_by_query": True,
        "path_smooth": 3.0,
        "n_estimators": 2000,
        "max_depth": int(trial_params["max_depth"]),
        "num_leaves": int(trial_params["num_leaves"]),
        "min_child_samples_fraction": float(trial_params["min_data_in_leaf"]),
        "min_sum_hessian_in_leaf": float(trial_params["min_sum_hessian_in_leaf"]),
        "min_gain_to_split": 0.0 if bool(trial_params["min_gain_to_split_zero"]) else float(trial_params["min_gain_to_split"]),
        "feature_fraction": float(trial_params["feature_fraction"]),
        "bagging_fraction": float(trial_params["bagging_fraction"]),
        "lambda_l1": 0.0 if bool(trial_params["lambda_l1_zero"]) else float(trial_params["lambda_l1"]),
        "lambda_l2": float(trial_params["lambda_l2"]),
        "max_bin": int(trial_params["max_bin"]),
        "lambdarank_truncation_level": int(trial_params["lambdarank_truncation_level"]),
        "label_gain": list(TAIL_LABEL_GAINS[gain_name]),
        "label_gain_name": gain_name,
    }


def suggest_base_lambdarank_params(
    trial: Any,
    *,
    retained_fraction: float,
    median_candidates_per_query: float,
    max_boost_rounds: int = 500,
) -> dict[str, Any]:
    """Return the bounded base-head LambdaRank space.

    Base heads use the same portability-oriented search geometry as residual
    heads, but their ceiling is intentionally lower.  Callers must still fit
    an internal chronological validation slice and apply early stopping; this
    function only declares the reusable, fold-size-independent space.
    """
    if max_boost_rounds <= 0 or max_boost_rounds > 500:
        raise ValueError("base-head boosting ceiling must be in 1..500")
    params = suggest_broad_lambdarank_params(
        trial,
        retained_fraction=retained_fraction,
        median_candidates_per_query=median_candidates_per_query,
    )
    params["n_estimators"] = int(max_boost_rounds)
    return params
