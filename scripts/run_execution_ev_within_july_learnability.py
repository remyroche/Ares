#!/usr/bin/env python3
"""Audit whether the fixed execution-EV configuration can learn within July.

Only expanding past-to-future block predictions count as forward OOS evidence.
Matched reverse-time, shuffled cross-fit, and in-sample results are written as
diagnostics and are explicitly excluded from the conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import execution_ev_metrics

DEFAULT_INPUT = (
    ROOT
    / "data_perp/artifacts/"
    "execution_ev_context_clean_regime_input_forward_july19_20260726_v1/"
    "joined.parquet"
)
DEFAULT_CONFIG = (
    ROOT
    / "data_perp/artifacts/"
    "execution_ev_context_clean_regime_diagnosis_forward_july19_20260726_v1/"
    "regime_diagnosis_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "execution_ev_context_clean_within_july_learnability_20260726_v1"
)
SCHEMA = "execution_ev_within_july_learnability_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DECISION = "execution_decision_utc"
RESOLVED = "execution_label_end_utc"
TARGET = "execution_net_ev_12h"
BASELINE = "existing_alpha_ev"


@dataclass(frozen=True)
class TemporalSplit:
    mode: str
    fold_id: str
    train_positions: np.ndarray
    evaluation_positions: np.ndarray
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    is_valid_forward_oos: bool
    matched_size_control: bool
    training_direction: str


FitPredict = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, int],
    np.ndarray,
]


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


def _positions(mask: pd.Series | np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def _evenly_spaced(values: np.ndarray, size: int) -> np.ndarray:
    if size < 1 or size > len(values):
        raise ValueError("matched size must be within the source population")
    if size == len(values):
        return values.copy()
    local = np.floor(
        np.linspace(0, len(values), num=size, endpoint=False)
    ).astype(int)
    return values[local]


def prepare_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    july_start: str = "2026-07-01T00:00:00Z",
    july_end: str = "2026-07-20T00:00:00Z",
) -> pd.DataFrame:
    required = [
        *IDENTITY,
        DECISION,
        RESOLVED,
        TARGET,
        BASELINE,
        *feature_columns,
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("within-July input missing columns: " + ", ".join(missing))
    work = frame.loc[:, list(dict.fromkeys(required))].copy()
    for column in ("__ts__", DECISION, RESOLVED):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    start = pd.Timestamp(july_start)
    end = pd.Timestamp(july_end)
    work = work.loc[
        work[DECISION].ge(start) & work[DECISION].lt(end)
    ].reset_index(drop=True)
    if not len(work):
        raise ValueError("within-July slice is empty")
    if work["candidate_id"].duplicated().any():
        raise ValueError("within-July input contains duplicate candidate_id rows")
    numeric = work.loc[:, [TARGET, *feature_columns]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(np.float64)).all():
        raise ValueError("within-July target and features must be finite")
    work.loc[:, [TARGET, *feature_columns]] = numeric
    if (work[RESOLVED] < work[DECISION]).any():
        raise ValueError("label resolution cannot precede decision time")
    return work.sort_values(
        [DECISION, "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_temporal_splits(
    frame: pd.DataFrame,
    *,
    purge_hours: float = 12.0,
    boundaries: Sequence[str] = (
        "2026-07-08T00:00:00Z",
        "2026-07-15T00:00:00Z",
    ),
    july_start: str = "2026-07-01T00:00:00Z",
    july_end: str = "2026-07-20T00:00:00Z",
) -> list[TemporalSplit]:
    """Build primary expanding folds and exact-size directional controls."""

    decision = pd.to_datetime(frame[DECISION], utc=True, errors="raise")
    resolved = pd.to_datetime(frame[RESOLVED], utc=True, errors="raise")
    start = pd.Timestamp(july_start)
    end = pd.Timestamp(july_end)
    boundaries_utc = [pd.Timestamp(value) for value in boundaries]
    purge = pd.Timedelta(hours=float(purge_hours))
    splits: list[TemporalSplit] = []
    previous = start
    for fold_index, boundary in enumerate(boundaries_utc):
        evaluation_end = (
            boundaries_utc[fold_index + 1]
            if fold_index + 1 < len(boundaries_utc)
            else end
        )
        safe_expanding_train = _positions(
            decision.ge(start)
            & decision.lt(boundary - purge)
            & resolved.lt(boundary)
        )
        evaluation = _positions(
            decision.ge(boundary) & decision.lt(evaluation_end)
        )
        if not len(safe_expanding_train) or not len(evaluation):
            raise ValueError(f"empty forward block at {boundary.isoformat()}")
        fold_id = f"boundary_{boundary.strftime('%Y%m%d')}"
        splits.append(
            TemporalSplit(
                mode="forward_expanding",
                fold_id=fold_id,
                train_positions=safe_expanding_train,
                evaluation_positions=evaluation,
                evaluation_start=boundary,
                evaluation_end=evaluation_end,
                is_valid_forward_oos=True,
                matched_size_control=False,
                training_direction="past_to_future",
            )
        )

        safe_previous_block = _positions(
            decision.ge(previous)
            & decision.lt(boundary - purge)
            & resolved.lt(boundary)
        )
        next_block = evaluation
        matched_rows = min(len(safe_previous_block), len(next_block))
        if matched_rows:
            forward_train = _evenly_spaced(safe_previous_block, matched_rows)
            forward_eval = _evenly_spaced(next_block, matched_rows)
            splits.append(
                TemporalSplit(
                    mode="forward_block_matched",
                    fold_id=fold_id,
                    train_positions=forward_train,
                    evaluation_positions=forward_eval,
                    evaluation_start=boundary,
                    evaluation_end=evaluation_end,
                    is_valid_forward_oos=True,
                    matched_size_control=True,
                    training_direction="past_to_future",
                )
            )
            splits.append(
                TemporalSplit(
                    mode="reversed_block_matched",
                    fold_id=fold_id,
                    train_positions=forward_eval,
                    evaluation_positions=forward_train,
                    evaluation_start=previous,
                    evaluation_end=boundary,
                    is_valid_forward_oos=False,
                    matched_size_control=True,
                    training_direction="future_to_past",
                )
            )
        previous = boundary
    return splits


def assert_forward_safe(
    frame: pd.DataFrame,
    split: TemporalSplit,
    *,
    purge_hours: float,
) -> None:
    if not split.is_valid_forward_oos:
        return
    train = frame.iloc[split.train_positions]
    cutoff = split.evaluation_start - pd.Timedelta(hours=float(purge_hours))
    if not (
        train[DECISION].lt(cutoff).all()
        and train[RESOLVED].lt(split.evaluation_start).all()
    ):
        raise AssertionError("forward within-July split violates purge/resolution")


def _fixed_catboost_residual_fit_predict(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    evaluation_x: pd.DataFrame,
    seed: int,
    *,
    iterations: int = 250,
    n_jobs: int = 1,
) -> np.ndarray:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=int(iterations),
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        bagging_temperature=1.0,
        bootstrap_type="Bayesian",
        random_seed=int(seed),
        thread_count=int(n_jobs),
        verbose=False,
        allow_writing_files=False,
    )
    baseline = train_x[BASELINE].to_numpy(np.float64)
    model.fit(train_x, np.asarray(train_y, dtype=np.float64) - baseline)
    return (
        evaluation_x[BASELINE].to_numpy(np.float64)
        + np.asarray(model.predict(evaluation_x), dtype=np.float64)
    )


def _metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    top_k_fraction: float,
) -> dict[str, Any]:
    target = frame[TARGET].to_numpy(np.float64)
    report = dict(
        execution_ev_metrics(
            target,
            prediction,
            top_k_fraction=top_k_fraction,
        )
    )
    report["unconditional_mean_net_ev"] = float(target.mean())
    report["top_k_lift_vs_unconditional"] = float(
        report["top_k_mean_net_ev"] - target.mean()
    )
    return report


def _metric_rows(
    evaluation: pd.DataFrame,
    prediction: np.ndarray,
    *,
    metadata: Mapping[str, Any],
    top_k_fraction: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in ("pooled", "long", "short"):
        mask = (
            np.ones(len(evaluation), dtype=bool)
            if scope == "pooled"
            else evaluation["side_name"].astype(str).eq(scope).to_numpy()
        )
        if not mask.any():
            continue
        rows.append(
            {
                **metadata,
                "scope": scope,
                "scored_rows": int(mask.sum()),
                "coverage": float(mask.mean()) if scope == "pooled" else 1.0,
                **_metrics(
                    evaluation.loc[mask].reset_index(drop=True),
                    prediction[mask],
                    top_k_fraction=top_k_fraction,
                ),
            }
        )
    return rows


def evaluate_splits(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    splits: Sequence[TemporalSplit],
    fit_predict: FitPredict,
    *,
    purge_hours: float,
    top_k_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    for split_index, split in enumerate(splits):
        assert_forward_safe(frame, split, purge_hours=purge_hours)
        train = frame.iloc[split.train_positions]
        evaluation = frame.iloc[split.evaluation_positions]
        prediction = fit_predict(
            train.loc[:, list(feature_columns)],
            train[TARGET].to_numpy(np.float64),
            evaluation.loc[:, list(feature_columns)],
            int(random_state + split_index),
        )
        if len(prediction) != len(evaluation) or not np.isfinite(prediction).all():
            raise ValueError("fit_predict returned invalid evaluation predictions")
        metadata = {
            "mode": split.mode,
            "fold_id": split.fold_id,
            "evidence_status": (
                "valid_forward_oos"
                if split.is_valid_forward_oos
                else "diagnostic_non_oos"
            ),
            "is_valid_forward_oos": split.is_valid_forward_oos,
            "matched_size_control": split.matched_size_control,
            "training_direction": split.training_direction,
            "train_rows": int(len(train)),
            "evaluation_population_rows": int(len(evaluation)),
            "train_decision_min_utc": train[DECISION].min(),
            "train_decision_max_utc": train[DECISION].max(),
            "max_train_label_resolution_utc": train[RESOLVED].max(),
            "evaluation_start_utc": evaluation[DECISION].min(),
            "evaluation_end_utc": evaluation[DECISION].max(),
        }
        metric_rows.extend(
            _metric_rows(
                evaluation,
                prediction,
                metadata=metadata,
                top_k_fraction=top_k_fraction,
            )
        )
        part = evaluation.loc[
            :,
            [*IDENTITY, DECISION, RESOLVED, TARGET, BASELINE],
        ].copy()
        part["mode"] = split.mode
        part["fold_id"] = split.fold_id
        part["evidence_status"] = metadata["evidence_status"]
        part["is_valid_forward_oos"] = split.is_valid_forward_oos
        part["prediction"] = prediction
        prediction_parts.append(part)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_parts, ignore_index=True),
    )


def evaluate_diagnostics(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    fit_predict: FitPredict,
    *,
    top_k_fraction: float,
    random_state: int,
    crossfit_splits: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    positions = np.arange(len(frame), dtype=np.int64)
    crossfit = np.full(len(frame), np.nan, dtype=np.float64)
    fold_ids = np.full(len(frame), "", dtype=object)
    splitter = KFold(
        n_splits=int(crossfit_splits),
        shuffle=True,
        random_state=int(random_state),
    )
    for fold, (train, evaluation) in enumerate(splitter.split(positions)):
        crossfit[evaluation] = fit_predict(
            frame.iloc[train].loc[:, list(feature_columns)],
            frame.iloc[train][TARGET].to_numpy(np.float64),
            frame.iloc[evaluation].loc[:, list(feature_columns)],
            int(random_state + 100 + fold),
        )
        fold_ids[evaluation] = f"random_{fold}"
    in_sample = fit_predict(
        frame.loc[:, list(feature_columns)],
        frame[TARGET].to_numpy(np.float64),
        frame.loc[:, list(feature_columns)],
        int(random_state + 200),
    )
    for mode, prediction, local_folds in (
        ("random_crossfit", crossfit, fold_ids),
        ("in_sample", in_sample, np.full(len(frame), "all_july", dtype=object)),
    ):
        metadata = {
            "mode": mode,
            "fold_id": "aggregate",
            "evidence_status": "diagnostic_non_oos",
            "is_valid_forward_oos": False,
            "matched_size_control": False,
            "training_direction": (
                "mixed_past_future" if mode == "random_crossfit" else "same_rows"
            ),
            "train_rows": int(len(frame)),
            "evaluation_population_rows": int(len(frame)),
            "train_decision_min_utc": frame[DECISION].min(),
            "train_decision_max_utc": frame[DECISION].max(),
            "max_train_label_resolution_utc": frame[RESOLVED].max(),
            "evaluation_start_utc": frame[DECISION].min(),
            "evaluation_end_utc": frame[DECISION].max(),
        }
        metric_rows.extend(
            _metric_rows(
                frame,
                prediction,
                metadata=metadata,
                top_k_fraction=top_k_fraction,
            )
        )
        part = frame.loc[
            :,
            [*IDENTITY, DECISION, RESOLVED, TARGET, BASELINE],
        ].copy()
        part["mode"] = mode
        part["fold_id"] = local_folds
        part["evidence_status"] = "diagnostic_non_oos"
        part["is_valid_forward_oos"] = False
        part["prediction"] = prediction
        prediction_parts.append(part)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_parts, ignore_index=True),
    )


def aggregate_prediction_metrics(
    predictions: pd.DataFrame,
    *,
    top_k_fraction: float,
    population_rows_by_scope: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode, group in predictions.groupby("mode", sort=True):
        for scope in ("pooled", "long", "short"):
            local = (
                group
                if scope == "pooled"
                else group.loc[group["side_name"].astype(str).eq(scope)]
            )
            if not len(local):
                continue
            model_metrics = _metrics(
                local.reset_index(drop=True),
                local["prediction"].to_numpy(np.float64),
                top_k_fraction=top_k_fraction,
            )
            baseline_metrics = _metrics(
                local.reset_index(drop=True),
                local[BASELINE].to_numpy(np.float64),
                top_k_fraction=top_k_fraction,
            )
            population_rows = (
                int(population_rows_by_scope[scope])
                if population_rows_by_scope is not None
                else int(len(local))
            )
            rows.append(
                {
                    "mode": mode,
                    "fold_id": "aggregate",
                    "scope": scope,
                    "evidence_status": str(local["evidence_status"].iloc[0]),
                    "is_valid_forward_oos": bool(
                        local["is_valid_forward_oos"].all()
                    ),
                    "evaluation_population_rows": int(len(local)),
                    "scored_rows": int(len(local)),
                    "coverage": 1.0,
                    "coverage_of_full_july_population": float(
                        len(local) / population_rows
                    ),
                    **model_metrics,
                    "baseline_top_k_mean_net_ev": baseline_metrics[
                        "top_k_mean_net_ev"
                    ],
                    "baseline_positive_ev_auc": baseline_metrics[
                        "positive_ev_auc"
                    ],
                    "baseline_spearman": baseline_metrics["spearman"],
                    "top_k_mean_net_ev_delta_vs_baseline": float(
                        model_metrics["top_k_mean_net_ev"]
                        - baseline_metrics["top_k_mean_net_ev"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _rank_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    top_k_fraction: float,
) -> dict[str, float]:
    metric = _metrics(
        pd.DataFrame({TARGET: target}),
        prediction,
        top_k_fraction=top_k_fraction,
    )
    return {
        "top_k_mean_net_ev": float(metric["top_k_mean_net_ev"]),
        "positive_ev_auc": float(metric["positive_ev_auc"]),
        "spearman": float(metric["spearman"]),
        "top_k_positive_ev_rate": float(metric["top_k_positive_ev_rate"]),
    }


def day_block_bootstrap(
    predictions: pd.DataFrame,
    *,
    top_k_fraction: float = 0.10,
    draws: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap complete UTC decision days on fixed forward predictions."""

    rows: list[dict[str, Any]] = []
    for scope in ("pooled", "long", "short"):
        local = (
            predictions
            if scope == "pooled"
            else predictions.loc[predictions["side_name"].astype(str).eq(scope)]
        ).reset_index(drop=True)
        local["_day"] = pd.to_datetime(local[DECISION], utc=True).dt.floor("D")
        days = np.asarray(sorted(local["_day"].unique()))
        positions = {
            day: np.flatnonzero(local["_day"].to_numpy() == day) for day in days
        }
        target = local[TARGET].to_numpy(np.float64)
        model = local["prediction"].to_numpy(np.float64)
        baseline = local[BASELINE].to_numpy(np.float64)
        observed_model = _rank_metrics(
            target,
            model,
            top_k_fraction=top_k_fraction,
        )
        observed_baseline = _rank_metrics(
            target,
            baseline,
            top_k_fraction=top_k_fraction,
        )
        rng = np.random.default_rng(int(random_state + len(rows)))
        sample_metrics: dict[str, list[float]] = {
            "top_k_mean_net_ev": [],
            "positive_ev_auc": [],
            "spearman": [],
            "top_k_positive_ev_rate": [],
            "top_k_mean_net_ev_delta_vs_baseline": [],
        }
        for _ in range(int(draws)):
            sampled_days = rng.choice(days, size=len(days), replace=True)
            sampled = np.concatenate([positions[day] for day in sampled_days])
            model_metrics = _rank_metrics(
                target[sampled],
                model[sampled],
                top_k_fraction=top_k_fraction,
            )
            baseline_metrics = _rank_metrics(
                target[sampled],
                baseline[sampled],
                top_k_fraction=top_k_fraction,
            )
            for name in (
                "top_k_mean_net_ev",
                "positive_ev_auc",
                "spearman",
                "top_k_positive_ev_rate",
            ):
                sample_metrics[name].append(model_metrics[name])
            sample_metrics["top_k_mean_net_ev_delta_vs_baseline"].append(
                model_metrics["top_k_mean_net_ev"]
                - baseline_metrics["top_k_mean_net_ev"]
            )
        observed = {
            **observed_model,
            "top_k_mean_net_ev_delta_vs_baseline": (
                observed_model["top_k_mean_net_ev"]
                - observed_baseline["top_k_mean_net_ev"]
            ),
        }
        for name, samples in sample_metrics.items():
            values = np.asarray(samples, dtype=np.float64)
            finite = values[np.isfinite(values)]
            null_value = (
                0.5
                if name in {"positive_ev_auc", "top_k_positive_ev_rate"}
                else 0.0
            )
            rows.append(
                {
                    "scope": scope,
                    "metric": name,
                    "observed": observed[name],
                    "ci025": (
                        float(np.quantile(finite, 0.025)) if len(finite) else np.nan
                    ),
                    "ci50": (
                        float(np.quantile(finite, 0.50)) if len(finite) else np.nan
                    ),
                    "ci975": (
                        float(np.quantile(finite, 0.975)) if len(finite) else np.nan
                    ),
                    "null_value": float(null_value),
                    "probability_le_null": (
                        float(np.mean(finite <= null_value))
                        if len(finite)
                        else np.nan
                    ),
                    "days": int(len(days)),
                    "draws": int(draws),
                    "bootstrap_unit": "UTC decision day",
                }
            )
    return pd.DataFrame(rows)


def learnability_conclusion(
    aggregate_metrics: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> dict[str, Any]:
    primary = aggregate_metrics.loc[
        aggregate_metrics["mode"].eq("forward_expanding")
        & aggregate_metrics["scope"].eq("pooled")
    ]
    if len(primary) != 1:
        raise ValueError("missing pooled forward_expanding aggregate")
    metric = primary.iloc[0]
    top_uncertainty = uncertainty.loc[
        uncertainty["scope"].eq("pooled")
        & uncertainty["metric"].eq("top_k_mean_net_ev")
    ].iloc[0]
    delta_uncertainty = uncertainty.loc[
        uncertainty["scope"].eq("pooled")
        & uncertainty["metric"].eq(
            "top_k_mean_net_ev_delta_vs_baseline"
        )
    ].iloc[0]
    directional = bool(
        metric["positive_ev_auc"] > 0.5 and metric["spearman"] > 0.0
    )
    economic = bool(
        metric["top_k_mean_net_ev"] > 0.0
        and metric["top_k_lift_vs_unconditional"] > 0.0
    )
    robust_positive = bool(top_uncertainty["ci025"] > 0.0)
    beats_baseline_robustly = bool(delta_uncertainty["ci025"] > 0.0)
    if directional and economic and robust_positive:
        status = "forward_oos_learnability_supported"
    elif directional and economic:
        status = "forward_oos_signal_positive_but_uncertain"
    else:
        status = "forward_oos_learnability_not_supported"
    return {
        "status": status,
        "valid_evidence_mode": "forward_expanding",
        "directional_rank_signal": directional,
        "positive_top10_economics": economic,
        "day_block_ci_excludes_zero": robust_positive,
        "day_block_delta_vs_baseline_ci_excludes_zero": beats_baseline_robustly,
        "diagnostic_modes_excluded": [
            "forward_block_matched",
            "reversed_block_matched",
            "random_crossfit",
            "in_sample",
        ],
    }


def run(
    *,
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    purge_hours: float = 12.0,
    top_k_fraction: float = 0.10,
    iterations: int = 250,
    n_jobs: int = 1,
    bootstrap_draws: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    for path in (input_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing within-July source: {path}")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    feature_columns = list(config_payload["feature_columns"])
    frame = pd.read_parquet(input_path)
    archetype_columns = [
        name for name in feature_columns if name.startswith("catboost_archetype__")
    ]
    if archetype_columns:
        if "catboost_archetype" not in frame:
            raise ValueError("catboost_archetype source is missing")
        archetype = frame["catboost_archetype"].astype(str)
        for name in archetype_columns:
            frame[name] = archetype.eq(
                name.removeprefix("catboost_archetype__")
            ).astype(np.float32)
    work = prepare_frame(frame, feature_columns=feature_columns)
    splits = build_temporal_splits(work, purge_hours=purge_hours)

    def fit_predict(
        train_x: pd.DataFrame,
        train_y: np.ndarray,
        evaluation_x: pd.DataFrame,
        seed: int,
    ) -> np.ndarray:
        return _fixed_catboost_residual_fit_predict(
            train_x,
            train_y,
            evaluation_x,
            seed,
            iterations=iterations,
            n_jobs=n_jobs,
        )

    temporal_metrics, temporal_predictions = evaluate_splits(
        work,
        feature_columns,
        splits,
        fit_predict,
        purge_hours=purge_hours,
        top_k_fraction=top_k_fraction,
        random_state=random_state,
    )
    diagnostic_metrics, diagnostic_predictions = evaluate_diagnostics(
        work,
        feature_columns,
        fit_predict,
        top_k_fraction=top_k_fraction,
        random_state=random_state,
    )
    predictions = pd.concat(
        [temporal_predictions, diagnostic_predictions],
        ignore_index=True,
    )
    metrics = pd.concat(
        [temporal_metrics, diagnostic_metrics],
        ignore_index=True,
    )
    population_rows_by_scope = {
        "pooled": int(len(work)),
        "long": int(work["side_name"].astype(str).eq("long").sum()),
        "short": int(work["side_name"].astype(str).eq("short").sum()),
    }
    aggregates = aggregate_prediction_metrics(
        predictions,
        top_k_fraction=top_k_fraction,
        population_rows_by_scope=population_rows_by_scope,
    )
    primary_predictions = predictions.loc[
        predictions["mode"].eq("forward_expanding")
    ].reset_index(drop=True)
    uncertainty = day_block_bootstrap(
        primary_predictions,
        top_k_fraction=top_k_fraction,
        draws=bootstrap_draws,
        random_state=random_state,
    )
    conclusion = learnability_conclusion(aggregates, uncertainty)
    split_rows = []
    for split in splits:
        train = work.iloc[split.train_positions]
        evaluation = work.iloc[split.evaluation_positions]
        split_rows.append(
            {
                "mode": split.mode,
                "fold_id": split.fold_id,
                "evidence_status": (
                    "valid_forward_oos"
                    if split.is_valid_forward_oos
                    else "diagnostic_non_oos"
                ),
                "is_valid_forward_oos": split.is_valid_forward_oos,
                "matched_size_control": split.matched_size_control,
                "training_direction": split.training_direction,
                "train_rows": int(len(train)),
                "evaluation_rows": int(len(evaluation)),
                "train_decision_max_utc": train[DECISION].max(),
                "max_train_label_resolution_utc": train[RESOLVED].max(),
                "evaluation_start_utc": evaluation[DECISION].min(),
                "evaluation_end_utc": evaluation[DECISION].max(),
            }
        )
    split_ledger = pd.DataFrame(split_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "within_july_metrics.csv",
        "aggregate_metrics": output_dir / "within_july_aggregate_metrics.csv",
        "predictions": output_dir / "within_july_predictions.parquet",
        "uncertainty": output_dir / "within_july_day_block_uncertainty.csv",
        "splits": output_dir / "within_july_splits.csv",
        "report": output_dir / "report.json",
    }
    metrics.to_csv(paths["metrics"], index=False)
    aggregates.to_csv(paths["aggregate_metrics"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)
    uncertainty.to_csv(paths["uncertainty"], index=False)
    split_ledger.to_csv(paths["splits"], index=False)
    source_records = {
        "input": _source(input_path),
        "configuration": _source(config_path),
    }
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc),
        "conclusion": conclusion,
        "contract": {
            "valid_forward_evidence": (
                "expanding earlier-July fit to later-July blocked evaluation; "
                "decision purge and label resolution both precede evaluation"
            ),
            "diagnostic_only": (
                "matched reverse, shuffled cross-fit, and in-sample comparisons "
                "cannot support promotion or forward claims"
            ),
            "selection": "global pooled top 10%; side metrics are reporting slices",
            "model": (
                "fixed winning residual CatBoost geometry, all declared features, "
                "no HPO and no evaluation-derived calibration"
            ),
            "bootstrap": "complete UTC decision days on fixed forward predictions",
        },
        "config": {
            "purge_hours": float(purge_hours),
            "top_k_fraction": float(top_k_fraction),
            "iterations": int(iterations),
            "n_jobs": int(n_jobs),
            "bootstrap_draws": int(bootstrap_draws),
            "random_state": int(random_state),
            "feature_columns": feature_columns,
        },
        "data": {
            "rows": int(len(work)),
            "candidate_rows": int(work["candidate_id"].nunique()),
            "decision_min": work[DECISION].min(),
            "decision_max": work[DECISION].max(),
            "label_resolution_max": work[RESOLVED].max(),
            "side_rows": work["side_name"].value_counts().sort_index().to_dict(),
        },
        "sources": source_records,
        "split_ledger": split_rows,
        "aggregate_metrics": aggregates.to_dict(orient="records"),
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
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=250)
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
