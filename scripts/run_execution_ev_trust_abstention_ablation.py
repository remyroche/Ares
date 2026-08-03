#!/usr/bin/env python3
"""Learn a causal trust/abstention head for the frozen execution-EV winner.

The trust target is cost-adjusted realized net utility relative to abstaining
at zero.  Weekly expanding folds enforce both a decision-time purge and strict
label-resolution cutoff.  Calendar and ex-post cost fields are never features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SCORES = (
    ROOT
    / "data_perp/artifacts/"
    "execution_ev_context_clean_exact_recent_correction_forward_july19_20260726_v2/"
    "mapped_oof_and_forward.parquet"
)
DEFAULT_FEATURES = (
    ROOT
    / "data_perp/artifacts/"
    "execution_ev_context_clean_regime_input_forward_july19_20260726_v1/"
    "joined.parquet"
)
DEFAULT_COSTS = (
    ROOT
    / "data_perp/artifacts/"
    "meaningful_mfe_catboost_v2_ablation_july20_20260726_v1/"
    "exact_policy_paired.parquet"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "execution_ev_trust_abstention_ablation_20260726_v1"
)
SCHEMA = "execution_ev_trust_abstention_ablation_v1"
SCORE = "catboost__residual__without_hpo__all_features"
FROZEN_RANK_SCORE = (
    "catboost__residual__without_hpo__all_features"
    "__recent_ev_catboost_predicted_archetype"
)
TARGET = "execution_net_ev_12h"
DECISION = "execution_decision_utc"
RESOLVED = "execution_label_end_utc"
SIDE = "side_name"
IDENTITY = ("__ts__", "__symbol__", SIDE, "candidate_id")
LABEL_VARIANTS = ("hard_positive", "logistic_50bps", "clipped_200bps")
STRATEGIES = ("no_trust", "trust_rank_input", "trust_gate", "trust_x_ev")
BASE_TRUST_FEATURES = (
    SCORE,
    FROZEN_RANK_SCORE,
    "existing_alpha_ev",
    "pred_peak_MFE_12h_ATR",
    "catboost_entropy",
    "alpha_prediction_uncertainty",
    "alpha_leaf_support",
    "base_oof_score",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "oof_clean_favorable_probability",
    "execution_expected_spread_bps",
    "execution_entry_half_spread_bps",
)
PROBABILITY_COLUMNS = tuple(f"catboost_p_{index}" for index in range(7))
SHIFT_SOURCE_COLUMNS = (
    SCORE,
    FROZEN_RANK_SCORE,
    "existing_alpha_ev",
    "pred_peak_MFE_12h_ATR",
    "catboost_entropy",
    "alpha_prediction_uncertainty",
    "alpha_leaf_support",
    "oof_clean_favorable_probability",
    "execution_expected_spread_bps",
)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": int(stat.st_size),
    }


def trust_soft_targets(net_ev: Sequence[float]) -> dict[str, np.ndarray]:
    """Return hard and soft utilities around the zero-utility abstain action."""

    values = np.asarray(net_ev, dtype=np.float64)
    logistic = 1.0 / (1.0 + np.exp(-np.clip(values / 0.005, -40.0, 40.0)))
    clipped = np.clip(0.5 + values / 0.04, 0.0, 1.0)
    return {
        "hard_positive": (values > 0.0).astype(np.float32),
        "logistic_50bps": logistic.astype(np.float32),
        "clipped_200bps": clipped.astype(np.float32),
    }


def causal_recent_unlabeled_shift(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    decision_column: str = DECISION,
    window_days: int = 7,
    min_reference_rows: int = 500,
) -> pd.DataFrame:
    """Compute past-day-only robust shifts without using any labels."""

    decision = pd.to_datetime(frame[decision_column], utc=True, errors="raise")
    day = decision.dt.floor("D")
    output = pd.DataFrame(index=frame.index)
    reference_rows = np.zeros(len(frame), dtype=np.int32)
    source = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    for column in columns:
        output[f"recent_unlabeled_shift_z__{column}"] = np.float32(0.0)
    unique_days = pd.Index(day.unique()).sort_values()
    for snapshot in unique_days:
        reference = (
            day.lt(snapshot)
            & day.ge(snapshot - pd.Timedelta(days=int(window_days)))
        )
        current = day.eq(snapshot)
        count = int(reference.sum())
        reference_rows[current.to_numpy()] = count
        if count < int(min_reference_rows):
            continue
        ref = source.loc[reference]
        cur = source.loc[current]
        median = ref.median(axis=0)
        scale = (ref.quantile(0.75) - ref.quantile(0.25)).replace(0.0, np.nan)
        fallback = ref.std(axis=0).replace(0.0, 1.0)
        scale = scale.fillna(fallback).fillna(1.0).clip(lower=1e-8)
        shifted = (cur - median) / scale
        for column in columns:
            output.loc[current, f"recent_unlabeled_shift_z__{column}"] = (
                shifted[column].clip(-10.0, 10.0).fillna(0.0).astype(np.float32)
            )
    shift_columns = [
        f"recent_unlabeled_shift_z__{column}" for column in columns
    ]
    output["recent_unlabeled_shift_mean_abs_z"] = (
        output.loc[:, shift_columns].abs().mean(axis=1).astype(np.float32)
    )
    output["recent_unlabeled_shift_reference_rows"] = reference_rows.astype(
        np.float32
    )
    output["recent_unlabeled_shift_available"] = (
        reference_rows >= int(min_reference_rows)
    ).astype(np.float32)
    return output


def prepare_inputs(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    costs: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Join the winning OOF/forward score to causal trust inputs one-to-one."""

    score_columns = [
        *IDENTITY,
        DECISION,
        RESOLVED,
        TARGET,
        "evaluation_origin",
        SCORE,
        FROZEN_RANK_SCORE,
    ]
    missing = sorted(set(score_columns) - set(scores.columns))
    if missing:
        raise ValueError("score input missing columns: " + ", ".join(missing))
    cost_proxy_columns = {
        "execution_expected_spread_bps",
        "execution_entry_half_spread_bps",
    }
    raw_columns = [
        "candidate_id",
        *[
            name
            for name in BASE_TRUST_FEATURES
            if name not in {SCORE, FROZEN_RANK_SCORE}
            and name not in cost_proxy_columns
        ],
        *PROBABILITY_COLUMNS,
    ]
    missing = sorted(set(raw_columns) - set(features.columns))
    if missing:
        raise ValueError("winning feature input missing columns: " + ", ".join(missing))
    cost_columns = [
        "candidate_id",
        "execution_expected_spread_bps",
        "execution_entry_half_spread_bps",
    ]
    missing = sorted(set(cost_columns) - set(costs.columns))
    if missing:
        raise ValueError("cost input missing columns: " + ", ".join(missing))
    if any(
        frame["candidate_id"].duplicated().any()
        for frame in (scores, features, costs)
    ):
        raise ValueError("trust inputs require unique candidate_id rows")
    work = (
        scores.loc[:, score_columns]
        .merge(
            features.loc[:, raw_columns],
            on="candidate_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            costs.loc[:, cost_columns],
            on="candidate_id",
            how="inner",
            validate="one_to_one",
        )
    )
    for column in ("__ts__", DECISION, RESOLVED):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    numeric_columns = [
        TARGET,
        *BASE_TRUST_FEATURES,
        *PROBABILITY_COLUMNS,
    ]
    numeric = work.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(np.float64)).all(axis=1)
    dropped_nonfinite = int((~finite).sum())
    work = work.loc[finite].reset_index(drop=True)
    work.loc[:, numeric_columns] = numeric.loc[finite].reset_index(drop=True)
    if (work[RESOLVED] < work[DECISION]).any():
        raise ValueError("trust label resolution precedes decision time")

    probability = work.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(np.float64)
    ordered = np.sort(probability, axis=1)
    work["expert_probability_max"] = ordered[:, -1]
    work["expert_probability_margin"] = ordered[:, -1] - ordered[:, -2]
    work["expert_probability_std"] = probability.std(axis=1)
    work["expert_disagreement"] = 1.0 - ordered[:, -1]
    work["ev_alpha_abs_disagreement"] = np.abs(
        work[SCORE].to_numpy(np.float64)
        - work["existing_alpha_ev"].to_numpy(np.float64)
    )
    work["abs_winner_ev"] = work[SCORE].abs()
    shift = causal_recent_unlabeled_shift(work, SHIFT_SOURCE_COLUMNS)
    work = pd.concat([work, shift], axis=1)
    derived = [
        "expert_probability_max",
        "expert_probability_margin",
        "expert_probability_std",
        "expert_disagreement",
        "ev_alpha_abs_disagreement",
        "abs_winner_ev",
        *shift.columns.tolist(),
    ]
    trust_features = [
        *BASE_TRUST_FEATURES,
        *PROBABILITY_COLUMNS,
        *derived,
    ]
    if not np.isfinite(
        work.loc[:, trust_features].to_numpy(np.float64)
    ).all():
        raise ValueError("derived trust features must be finite")
    work = work.sort_values(
        [DECISION, "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)
    return work, trust_features, {
        "score_rows": int(len(scores)),
        "matched_rows": int(len(work)),
        "dropped_unmatched": int(len(scores) - len(work) - dropped_nonfinite),
        "dropped_nonfinite": dropped_nonfinite,
        "explicit_leaf_drift_available": False,
        "explicit_leaf_signature_available": False,
        "leaf_support_proxy": "alpha_leaf_support",
        "uncertainty_proxy": "alpha_prediction_uncertainty",
        "causal_cost_liquidity_proxies": [
            "execution_expected_spread_bps",
            "execution_entry_half_spread_bps",
        ],
        "excluded_future_cost_fields": [
            "execution_cost_return",
            "execution_exit_half_spread_bps",
        ],
    }


def weekly_purged_folds(
    frame: pd.DataFrame,
    *,
    purge_hours: float = 12.0,
    min_train_rows: int = 20_000,
) -> list[dict[str, Any]]:
    decision = pd.to_datetime(frame[DECISION], utc=True, errors="raise")
    resolved = pd.to_datetime(frame[RESOLVED], utc=True, errors="raise")
    first_monday = decision.min().floor("D") + pd.offsets.Week(weekday=0)
    final_end = decision.max().ceil("D")
    starts = pd.date_range(first_monday, final_end, freq="7D", tz="UTC")
    purge = pd.Timedelta(hours=float(purge_hours))
    folds: list[dict[str, Any]] = []
    for start in starts:
        end = start + pd.Timedelta(days=7)
        train = np.flatnonzero(
            decision.lt(start - purge).to_numpy()
            & resolved.lt(start).to_numpy()
        )
        validation = np.flatnonzero(
            decision.ge(start).to_numpy() & decision.lt(end).to_numpy()
        )
        if len(train) < int(min_train_rows) or not len(validation):
            continue
        folds.append(
            {
                "fold": len(folds),
                "week_start": start,
                "week_end": end,
                "train_positions": train,
                "validation_positions": validation,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "max_train_decision": decision.iloc[train].max(),
                "max_train_label_resolution": resolved.iloc[train].max(),
            }
        )
    if not folds:
        raise ValueError("no weekly trust fold satisfies the strict contract")
    return folds


def _fit_trust(
    train_x: pd.DataFrame,
    train_target: np.ndarray,
    evaluation_x: pd.DataFrame,
    *,
    seed: int,
    iterations: int,
    n_jobs: int,
) -> np.ndarray:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="CrossEntropy",
        iterations=int(iterations),
        learning_rate=0.035,
        depth=5,
        l2_leaf_reg=10.0,
        random_strength=0.5,
        bagging_temperature=1.0,
        bootstrap_type="Bayesian",
        random_seed=int(seed),
        thread_count=int(n_jobs),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_x, train_target)
    validation_probability = model.predict_proba(evaluation_x)[:, 1]
    return np.asarray(validation_probability, dtype=np.float64)


def _robust_scale(reference: np.ndarray) -> float:
    q25, q75 = np.quantile(reference, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(reference))
    return max(scale, 1e-6)


def strategy_composites(
    frozen_rank_score: np.ndarray,
    trust_probability: np.ndarray,
    *,
    train_frozen_rank_score: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Layer trust on the immutable causal recent-mapped winner score."""

    scale = _robust_scale(train_frozen_rank_score)
    all_rows = np.ones(len(frozen_rank_score), dtype=bool)
    return {
        "no_trust": (frozen_rank_score.copy(), all_rows),
        "trust_rank_input": (
            frozen_rank_score + scale * (trust_probability - 0.5),
            all_rows,
        ),
        "trust_gate": (
            frozen_rank_score.copy(),
            trust_probability >= 0.5,
        ),
        "trust_x_ev": (
            frozen_rank_score * trust_probability,
            all_rows,
        ),
    }


def global_top_fraction_mask(
    score: np.ndarray,
    *,
    eligible: np.ndarray,
    population_rows: int,
    fraction: float,
) -> np.ndarray:
    quota = max(1, int(np.ceil(int(population_rows) * float(fraction))))
    positions = np.flatnonzero(eligible & np.isfinite(score))
    selected = np.zeros(len(score), dtype=bool)
    if not len(positions):
        return selected
    order = positions[np.argsort(-score[positions], kind="stable")]
    selected[order[: min(quota, len(order))]] = True
    return selected


def _head_metrics(
    net_ev: np.ndarray,
    trust_probability: np.ndarray,
    soft_target: np.ndarray,
) -> dict[str, float]:
    positive = net_ev > 0.0
    auc = (
        float(roc_auc_score(positive.astype(np.int8), trust_probability))
        if np.unique(positive).size == 2
        else np.nan
    )
    return {
        "trust_positive_utility_auc": auc,
        "trust_net_ev_spearman": float(
            spearmanr(trust_probability, net_ev).statistic
        ),
        "trust_soft_brier": float(
            np.mean((trust_probability - soft_target) ** 2)
        ),
    }


def fit_side_local_trust(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    trust_features: Sequence[str],
    train_target: np.ndarray,
    *,
    seed: int,
    iterations: int,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Fit independent long/short trust heads without a shared side feature."""

    probability = np.full(len(validation), np.nan, dtype=np.float64)
    rows: dict[str, int] = {}
    for side_index, side in enumerate(("long", "short")):
        train_mask = train[SIDE].astype(str).eq(side).to_numpy()
        validation_mask = validation[SIDE].astype(str).eq(side).to_numpy()
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(f"side-local trust fold has empty {side} rows")
        probability[validation_mask] = _fit_trust(
            train.loc[train_mask, list(trust_features)],
            train_target[train_mask],
            validation.loc[validation_mask, list(trust_features)],
            seed=int(seed + side_index),
            iterations=iterations,
            n_jobs=n_jobs,
        )
        rows[side] = int(train_mask.sum())
    if not np.isfinite(probability).all():
        raise ValueError("side-local trust heads left unscored validation rows")
    return probability, rows


def _strategy_metric_rows(
    validation: pd.DataFrame,
    *,
    variant: str,
    strategy: str,
    fold: Mapping[str, Any],
    ranking_score: np.ndarray,
    eligible: np.ndarray,
    selected: np.ndarray,
    trust_probability: np.ndarray,
    soft_target: np.ndarray,
) -> list[dict[str, Any]]:
    net_ev = validation[TARGET].to_numpy(np.float64)
    rows: list[dict[str, Any]] = []
    for scope in ("pooled", "long", "short"):
        scope_mask = (
            np.ones(len(validation), dtype=bool)
            if scope == "pooled"
            else validation[SIDE].astype(str).eq(scope).to_numpy()
        )
        local_selected = selected & scope_mask
        selected_values = net_ev[local_selected]
        rank_mask = eligible & scope_mask
        rank_target = net_ev[rank_mask]
        rank_score = ranking_score[rank_mask]
        positive = rank_target > 0.0
        auc = (
            float(roc_auc_score(positive.astype(np.int8), rank_score))
            if np.unique(positive).size == 2
            else np.nan
        )
        rows.append(
            {
                "label_variant": variant,
                "strategy": strategy,
                "fold": int(fold["fold"]),
                "week_start": fold["week_start"],
                "week_end": fold["week_end"],
                "scope": scope,
                "train_rows": int(fold["train_rows"]),
                "evaluation_rows": int(scope_mask.sum()),
                "eligible_rows": int(rank_mask.sum()),
                "coverage": float(rank_mask.sum() / max(scope_mask.sum(), 1)),
                "selected_rows": int(local_selected.sum()),
                "selected_mean_net_ev": (
                    float(selected_values.mean()) if len(selected_values) else np.nan
                ),
                "selected_sum_net_ev": float(selected_values.sum()),
                "selected_positive_rate": (
                    float((selected_values > 0.0).mean())
                    if len(selected_values)
                    else np.nan
                ),
                "rank_positive_ev_auc": auc,
                "rank_net_ev_spearman": (
                    float(spearmanr(rank_score, rank_target).statistic)
                    if len(rank_target) > 1
                    else np.nan
                ),
                **_head_metrics(
                    net_ev[scope_mask],
                    trust_probability[scope_mask],
                    soft_target[scope_mask],
                ),
            }
        )
    return rows


def run_oof(
    frame: pd.DataFrame,
    trust_features: Sequence[str],
    folds: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    n_jobs: int,
    random_state: int,
    top_k_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = trust_soft_targets(frame[TARGET].to_numpy(np.float64))
    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = frame.iloc[fold["train_positions"]].reset_index(drop=True)
        validation = frame.iloc[fold["validation_positions"]].reset_index(drop=True)
        if not (
            train[RESOLVED].lt(fold["week_start"]).all()
            and train[DECISION].lt(
                fold["week_start"] - pd.Timedelta(hours=12)
            ).all()
        ):
            raise AssertionError("weekly trust fold violates purge/resolution")
        fold_rows.append(
            {
                key: value
                for key, value in fold.items()
                if key not in {"train_positions", "validation_positions"}
            }
        )
        for variant_index, variant in enumerate(LABEL_VARIANTS):
            train_target = targets[variant][fold["train_positions"]]
            validation_target = targets[variant][fold["validation_positions"]]
            validation_trust, side_train_rows = fit_side_local_trust(
                train,
                validation,
                trust_features,
                train_target,
                seed=int(random_state + 100 * fold["fold"] + variant_index),
                iterations=iterations,
                n_jobs=n_jobs,
            )
            validation_frozen_score = validation[FROZEN_RANK_SCORE].to_numpy(
                np.float64
            )
            for strategy, (ranking_score, eligible) in strategy_composites(
                validation_frozen_score,
                validation_trust,
                train_frozen_rank_score=train[FROZEN_RANK_SCORE].to_numpy(
                    np.float64
                ),
            ).items():
                selected = global_top_fraction_mask(
                    ranking_score,
                    eligible=eligible,
                    population_rows=len(validation),
                    fraction=top_k_fraction,
                )
                metric_rows.extend(
                    _strategy_metric_rows(
                        validation,
                        variant=variant,
                        strategy=strategy,
                        fold=fold,
                        ranking_score=ranking_score,
                        eligible=eligible,
                        selected=selected,
                        trust_probability=validation_trust,
                        soft_target=validation_target,
                    )
                )
                part = validation.loc[
                    :,
                    [
                        *IDENTITY,
                        DECISION,
                        RESOLVED,
                        TARGET,
                        "evaluation_origin",
                        SCORE,
                        FROZEN_RANK_SCORE,
                    ],
                ].copy()
                part["label_variant"] = variant
                part["strategy"] = strategy
                part["fold"] = int(fold["fold"])
                part["week_start"] = fold["week_start"]
                part["trust_probability"] = validation_trust
                part["ranking_score"] = ranking_score
                part["eligible"] = eligible
                part["selected"] = selected
                part["long_trust_train_rows"] = side_train_rows["long"]
                part["short_trust_train_rows"] = side_train_rows["short"]
                prediction_parts.append(part)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_parts, ignore_index=True),
        pd.DataFrame(fold_rows),
    )


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["label_variant", "strategy"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        variant, strategy = keys
        for scope in ("pooled", "long", "short"):
            local = (
                group
                if scope == "pooled"
                else group.loc[group[SIDE].astype(str).eq(scope)]
            )
            selected = local.loc[local["selected"]]
            eligible = local.loc[local["eligible"]]
            positive = eligible[TARGET].gt(0.0)
            auc = (
                float(
                    roc_auc_score(
                        positive.astype(np.int8),
                        eligible["ranking_score"],
                    )
                )
                if positive.nunique() == 2
                else np.nan
            )
            rows.append(
                {
                    "label_variant": variant,
                    "strategy": strategy,
                    "scope": scope,
                    "evaluation_rows": int(len(local)),
                    "eligible_rows": int(len(eligible)),
                    "coverage": float(len(eligible) / max(len(local), 1)),
                    "selected_rows": int(len(selected)),
                    "selected_mean_net_ev": float(selected[TARGET].mean()),
                    "selected_sum_net_ev": float(selected[TARGET].sum()),
                    "selected_positive_rate": float(selected[TARGET].gt(0.0).mean()),
                    "rank_positive_ev_auc": auc,
                    "rank_net_ev_spearman": float(
                        spearmanr(
                            eligible["ranking_score"],
                            eligible[TARGET],
                        ).statistic
                    ),
                    "weeks": int(local["fold"].nunique()),
                    "worst_week_mean_net_ev": float(
                        selected.groupby("fold")[TARGET].mean().min()
                    ),
                }
            )
    return pd.DataFrame(rows)


def apply_pooled_global_selection(
    predictions: pd.DataFrame,
    *,
    top_k_fraction: float,
) -> pd.DataFrame:
    """Select one pooled global top decile across all weekly OOF rows."""

    output = predictions.copy()
    output["pooled_global_selected"] = False
    for _, group in output.groupby(
        ["label_variant", "strategy"],
        sort=True,
    ):
        selected = global_top_fraction_mask(
            group["ranking_score"].to_numpy(np.float64),
            eligible=group["eligible"].to_numpy(bool),
            population_rows=len(group),
            fraction=top_k_fraction,
        )
        output.loc[group.index, "pooled_global_selected"] = selected
    return output


def pooled_global_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report overall, month, latest-week, and side slices of global selection."""

    work = predictions.copy()
    decision = pd.to_datetime(work[DECISION], utc=True, errors="raise")
    work["_month"] = decision.dt.strftime("%Y-%m")
    latest_week = pd.to_datetime(work["week_start"], utc=True).max()
    work["_latest_week"] = pd.to_datetime(
        work["week_start"], utc=True
    ).eq(latest_week)
    rows: list[dict[str, Any]] = []
    for (variant, strategy), group in work.groupby(
        ["label_variant", "strategy"],
        sort=True,
    ):
        segments: list[tuple[str, str, pd.DataFrame]] = [("all", "all", group)]
        segments.extend(
            ("month", str(month), local)
            for month, local in group.groupby("_month", sort=True)
        )
        segments.append(
            (
                "latest_week",
                latest_week.isoformat(),
                group.loc[group["_latest_week"]],
            )
        )
        for segment_type, segment, segment_frame in segments:
            for scope in ("pooled", "long", "short"):
                local = (
                    segment_frame
                    if scope == "pooled"
                    else segment_frame.loc[
                        segment_frame[SIDE].astype(str).eq(scope)
                    ]
                )
                if not len(local):
                    continue
                eligible = local.loc[local["eligible"]]
                selected = local.loc[local["pooled_global_selected"]]
                positive = eligible[TARGET].gt(0.0)
                auc = (
                    float(
                        roc_auc_score(
                            positive.astype(np.int8),
                            eligible["ranking_score"],
                        )
                    )
                    if positive.nunique() == 2
                    else np.nan
                )
                rows.append(
                    {
                        "label_variant": variant,
                        "strategy": strategy,
                        "segment_type": segment_type,
                        "segment": segment,
                        "scope": scope,
                        "evaluation_rows": int(len(local)),
                        "eligible_rows": int(len(eligible)),
                        "coverage": float(len(eligible) / len(local)),
                        "selected_rows": int(len(selected)),
                        "selection_coverage": float(len(selected) / len(local)),
                        "selected_mean_net_ev": (
                            float(selected[TARGET].mean())
                            if len(selected)
                            else np.nan
                        ),
                        "selected_sum_net_ev": float(selected[TARGET].sum()),
                        "selected_positive_rate": (
                            float(selected[TARGET].gt(0.0).mean())
                            if len(selected)
                            else np.nan
                        ),
                        "rank_positive_ev_auc": auc,
                        "rank_net_ev_spearman": (
                            float(
                                spearmanr(
                                    eligible["ranking_score"],
                                    eligible[TARGET],
                                ).statistic
                            )
                            if len(eligible) > 1
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def latest_week_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    latest = pd.to_datetime(metrics["week_start"], utc=True).max()
    return metrics.loc[
        pd.to_datetime(metrics["week_start"], utc=True).eq(latest)
    ].reset_index(drop=True)


def day_block_uncertainty(
    predictions: pd.DataFrame,
    *,
    draws: int,
    random_state: int,
    selection_column: str = "pooled_global_selected",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(random_state))
    for (variant, scope), group in predictions.groupby(
        ["label_variant", SIDE],
        sort=True,
    ):
        scopes = [(scope, group)]
        if scope == sorted(predictions[SIDE].astype(str).unique())[0]:
            pooled = predictions.loc[
                predictions["label_variant"].eq(variant)
            ]
            scopes.append(("pooled", pooled))
        for scope_name, local_all in scopes:
            local_all = local_all.copy()
            local_all["_day"] = pd.to_datetime(
                local_all[DECISION], utc=True
            ).dt.floor("D")
            strategies = list(STRATEGIES)
            daily: dict[str, dict[pd.Timestamp, tuple[float, int]]] = {}
            for strategy in strategies:
                chosen = local_all.loc[
                    local_all["strategy"].eq(strategy)
                    & local_all[selection_column]
                ]
                grouped = chosen.groupby("_day")[TARGET].agg(["sum", "count"])
                daily[strategy] = {
                    day: (float(row["sum"]), int(row["count"]))
                    for day, row in grouped.iterrows()
                }
            days = np.asarray(sorted(local_all["_day"].unique()))
            for strategy in strategies:
                samples = []
                deltas = []
                for _ in range(int(draws)):
                    sampled_days = rng.choice(days, size=len(days), replace=True)
                    total = sum(daily[strategy].get(day, (0.0, 0))[0] for day in sampled_days)
                    count = sum(daily[strategy].get(day, (0.0, 0))[1] for day in sampled_days)
                    base_total = sum(
                        daily["no_trust"].get(day, (0.0, 0))[0]
                        for day in sampled_days
                    )
                    base_count = sum(
                        daily["no_trust"].get(day, (0.0, 0))[1]
                        for day in sampled_days
                    )
                    mean = total / max(count, 1)
                    base_mean = base_total / max(base_count, 1)
                    samples.append(mean)
                    deltas.append(mean - base_mean)
                values = np.asarray(samples, dtype=np.float64)
                delta_values = np.asarray(deltas, dtype=np.float64)
                observed = local_all.loc[
                    local_all["strategy"].eq(strategy)
                    & local_all[selection_column],
                    TARGET,
                ].mean()
                base_observed = local_all.loc[
                    local_all["strategy"].eq("no_trust")
                    & local_all[selection_column],
                    TARGET,
                ].mean()
                rows.append(
                    {
                        "label_variant": variant,
                        "strategy": strategy,
                        "scope": scope_name,
                        "observed_mean_net_ev": float(observed),
                        "ci025": float(np.quantile(values, 0.025)),
                        "ci50": float(np.quantile(values, 0.50)),
                        "ci975": float(np.quantile(values, 0.975)),
                        "probability_mean_le_zero": float(np.mean(values <= 0.0)),
                        "observed_delta_vs_no_trust": float(
                            observed - base_observed
                        ),
                        "delta_ci025": float(np.quantile(delta_values, 0.025)),
                        "delta_ci50": float(np.quantile(delta_values, 0.50)),
                        "delta_ci975": float(np.quantile(delta_values, 0.975)),
                        "probability_delta_le_zero": float(
                            np.mean(delta_values <= 0.0)
                        ),
                        "days": int(len(days)),
                        "draws": int(draws),
                        "bootstrap_unit": "UTC decision day",
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(
        ["label_variant", "strategy", "scope"]
    )


def conclusion(
    pooled_global: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> dict[str, Any]:
    pooled = pooled_global.loc[
        pooled_global["segment_type"].eq("all")
        & pooled_global["scope"].eq("pooled")
    ]
    challengers = pooled.loc[~pooled["strategy"].eq("no_trust")]
    best = challengers.sort_values(
        "selected_mean_net_ev",
        ascending=False,
    ).iloc[0]
    ci = uncertainty.loc[
        uncertainty["label_variant"].eq(best["label_variant"])
        & uncertainty["strategy"].eq(best["strategy"])
        & uncertainty["scope"].eq("pooled")
    ].iloc[0]
    latest_row = pooled_global.loc[
        pooled_global["segment_type"].eq("latest_week")
        & pooled_global["label_variant"].eq(best["label_variant"])
        & pooled_global["strategy"].eq(best["strategy"])
        & pooled_global["scope"].eq("pooled")
    ].iloc[0]
    robust_delta = bool(ci["delta_ci025"] > 0.0)
    positive = bool(best["selected_mean_net_ev"] > 0.0)
    latest_positive = bool(latest_row["selected_mean_net_ev"] > 0.0)
    status = (
        "trust_abstention_supported"
        if robust_delta and positive and latest_positive
        else "trust_signal_not_robust_for_abstention"
    )
    return {
        "status": status,
        "best_diagnostic_challenger": {
            "label_variant": str(best["label_variant"]),
            "strategy": str(best["strategy"]),
            "selected_mean_net_ev": float(best["selected_mean_net_ev"]),
            "coverage": float(best["coverage"]),
            "latest_week_mean_net_ev": float(
                latest_row["selected_mean_net_ev"]
            ),
            "delta_ci025": float(ci["delta_ci025"]),
            "delta_ci975": float(ci["delta_ci975"]),
        },
        "promotion_eligible": False,
        "selection_disclosure": (
            "best challenger is identified on the same OOF ablation and is "
            "diagnostic, not an untouched winner"
        ),
    }


def finalize_existing_predictions(
    output_dir: Path,
    *,
    top_k_fraction: float = 0.10,
    bootstrap_draws: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Resume reporting from completed OOF predictions without fitting models."""

    predictions_path = output_dir / "trust_ablation_oof_predictions.parquet"
    report_path = output_dir / "report.json"
    for path in (predictions_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing completed trust checkpoint: {path}")
    predictions = pd.read_parquet(predictions_path)
    predictions = apply_pooled_global_selection(
        predictions,
        top_k_fraction=top_k_fraction,
    )
    pooled_global = pooled_global_metrics(predictions)
    uncertainty = day_block_uncertainty(
        predictions,
        draws=bootstrap_draws,
        random_state=random_state,
        selection_column="pooled_global_selected",
    )
    result_conclusion = conclusion(pooled_global, uncertainty)
    predictions.to_parquet(predictions_path, index=False)
    pooled_path = output_dir / "trust_ablation_pooled_global_metrics.csv"
    uncertainty_path = output_dir / "trust_ablation_uncertainty.csv"
    pooled_global.to_csv(pooled_path, index=False)
    uncertainty.to_csv(uncertainty_path, index=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["conclusion"] = result_conclusion
    report["contract"]["selection"] = (
        "one pooled global 10% quota across all weekly OOF rows; month, "
        "latest-week, and side results slice that fixed global selection"
    )
    report["contract"]["weekly_selection_role"] = "stability diagnostic only"
    report["pooled_global_metrics"] = pooled_global.to_dict(orient="records")
    report["uncertainty"] = uncertainty.to_dict(orient="records")
    report["outputs"]["pooled_global"] = str(pooled_path.resolve())
    report_path.write_text(
        json.dumps(_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run(
    *,
    scores_path: Path,
    features_path: Path,
    costs_path: Path,
    output_dir: Path,
    purge_hours: float = 12.0,
    min_train_rows: int = 20_000,
    top_k_fraction: float = 0.10,
    iterations: int = 180,
    n_jobs: int = 1,
    bootstrap_draws: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    sources = {
        "winning_scores": scores_path,
        "winning_features_and_labels": features_path,
        "causal_entry_cost_proxies": costs_path,
    }
    missing = {name: str(path) for name, path in sources.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError("missing trust sources: " + json.dumps(missing))
    scores = pd.read_parquet(scores_path)
    features = pd.read_parquet(features_path)
    costs = pd.read_parquet(
        costs_path,
        columns=[
            "candidate_id",
            "execution_expected_spread_bps",
            "execution_entry_half_spread_bps",
        ],
    )
    frame, trust_features, join_audit = prepare_inputs(scores, features, costs)
    folds = weekly_purged_folds(
        frame,
        purge_hours=purge_hours,
        min_train_rows=min_train_rows,
    )
    metrics, predictions, split_ledger = run_oof(
        frame,
        trust_features,
        folds,
        iterations=iterations,
        n_jobs=n_jobs,
        random_state=random_state,
        top_k_fraction=top_k_fraction,
    )
    predictions = apply_pooled_global_selection(
        predictions,
        top_k_fraction=top_k_fraction,
    )
    aggregate = aggregate_metrics(predictions)
    pooled_global = pooled_global_metrics(predictions)
    latest = latest_week_metrics(metrics)
    uncertainty = day_block_uncertainty(
        predictions,
        draws=bootstrap_draws,
        random_state=random_state,
        selection_column="pooled_global_selected",
    )
    result_conclusion = conclusion(pooled_global, uncertainty)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "trust_ablation_weekly_metrics.csv",
        "aggregate": output_dir / "trust_ablation_aggregate_metrics.csv",
        "pooled_global": output_dir / "trust_ablation_pooled_global_metrics.csv",
        "latest_week": output_dir / "trust_ablation_latest_week.csv",
        "uncertainty": output_dir / "trust_ablation_uncertainty.csv",
        "splits": output_dir / "trust_ablation_splits.csv",
        "predictions": output_dir / "trust_ablation_oof_predictions.parquet",
        "report": output_dir / "report.json",
    }
    metrics.to_csv(paths["metrics"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    pooled_global.to_csv(paths["pooled_global"], index=False)
    latest.to_csv(paths["latest_week"], index=False)
    uncertainty.to_csv(paths["uncertainty"], index=False)
    split_ledger.to_csv(paths["splits"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)
    report = {
        "schema": SCHEMA,
        "status": "completed_research_oof_not_promotion_eligible",
        "created_at_utc": datetime.now(timezone.utc),
        "conclusion": result_conclusion,
        "contract": {
            "target": (
                "cost-adjusted execution_net_ev_12h utility of following the "
                "winner versus zero utility for abstaining"
            ),
            "soft_labels": {
                "hard_positive": "1(net_ev > 0)",
                "logistic_50bps": "sigmoid(net_ev / 0.005)",
                "clipped_200bps": "clip(0.5 + net_ev / 0.04, 0, 1)",
            },
            "weekly_oof": (
                "expanding weekly validation; train decision < week start - "
                "12h purge and train label resolution < week start"
            ),
            "frozen_ranking_baseline": (
                "immutable causal recent-EV corrected winner score; trust layers "
                "do not refit or remap it"
            ),
            "trust_head_routing": "independent side-local long and short heads",
            "no_calendar_regime_weights": True,
            "selection": (
                "one pooled global 10% quota across all weekly OOF rows; month, "
                "latest-week, and side results slice that fixed global selection"
            ),
            "weekly_selection_role": "stability diagnostic only",
        },
        "config": {
            "purge_hours": float(purge_hours),
            "min_train_rows": int(min_train_rows),
            "top_k_fraction": float(top_k_fraction),
            "iterations": int(iterations),
            "n_jobs": int(n_jobs),
            "bootstrap_draws": int(bootstrap_draws),
            "random_state": int(random_state),
            "trust_features": trust_features,
        },
        "data": {
            "rows": int(len(frame)),
            "decision_min": frame[DECISION].min(),
            "decision_max": frame[DECISION].max(),
            "label_resolution_max": frame[RESOLVED].max(),
            "origins": frame["evaluation_origin"].value_counts().to_dict(),
            "join_audit": join_audit,
        },
        "sources": {name: _source(path) for name, path in sources.items()},
        "folds": split_ledger.to_dict(orient="records"),
        "aggregate_metrics": aggregate.to_dict(orient="records"),
        "pooled_global_metrics": pooled_global.to_dict(orient="records"),
        "latest_week_metrics": latest.to_dict(orient="records"),
        "uncertainty": uncertainty.to_dict(orient="records"),
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
    }
    paths["report"].write_text(
        json.dumps(_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-path", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--costs-path", type=Path, default=DEFAULT_COSTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--min-train-rows", type=int, default=20_000)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    report = run(**vars(args))
    print(
        json.dumps(
            {
                "status": report["status"],
                "conclusion": report["conclusion"],
                "rows": report["data"]["rows"],
                "output": report["outputs"]["report"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
