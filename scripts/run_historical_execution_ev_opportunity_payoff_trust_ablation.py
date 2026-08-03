#!/usr/bin/env python3
"""Bounded canonical residual-tier opportunity/payoff/trust ablation.

The runner consumes the immutable historical v6 population and the frozen
``plus_risk_peak/direct_net`` v2 control.  Model/feature selection is confined
to strictly purged March blocks; April is scored once after all choices are
frozen.  Timing, MAE, wait/reprice, and target-price fields are forbidden model
inputs.  Selection is always one pooled global book with deterministic
candidate-ID tie breaking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_historical_execution_ev_gross_hurdle_decomposition import (
    ID,
    OOF_FOLDS,
    _arm_features,
    _features_by_rank,
    _load_frozen_population,
    _matrix,
    _purged_before,
    atomic_json,
    atomic_parquet,
    sha256,
)


SCHEMA = "historical_execution_ev_opportunity_payoff_trust_ablation_v1"
SIDES = ("long", "short")
EXIT_CLASSES = ("trailing", "timeout", "full_stop", "adverse_exit")
BASE_ARMS = (
    "matched_direct_net",
    "opportunity_0bps_signed_magnitude",
    "opportunity_25bps_signed_magnitude",
    "four_exit_signed_payoff",
)
FINAL_ARMS = (
    "frozen_control",
    *BASE_ARMS,
    "direct_primary_oof_stack",
    "causal_trust_overlay",
)
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
GEOMETRIES = ((4, 6.0), (6, 10.0))
FORBIDDEN_FEATURE_TOKENS = (
    "execution_gross",
    "execution_net",
    "execution_cost",
    "execution_exit",
    "exit_minute",
    "realized",
    "target_price",
    "wait_",
    "reprice",
    "timing",
    "time_to",
    "mfe_return",
    "mae",
    "label",
    "outcome",
    "future_",
)
REGIME_GROUPS = ("core_gross_opportunity", "past_only_transition_deltas")


@dataclass(frozen=True)
class Geometry:
    depth: int
    l2: float


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def identity_sha256(frame: pd.DataFrame) -> str:
    identity = frame.loc[:, ID].copy()
    identity["__ts__"] = pd.to_datetime(identity["__ts__"], utc=True).astype(str)
    identity = identity.astype(str).sort_values(ID, kind="stable")
    return hashlib.sha256(
        identity.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def validate_feature_columns(columns: Sequence[str]) -> list[str]:
    names = list(dict.fromkeys(map(str, columns)))
    forbidden = [
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(
            "post-entry/timing/MAE/wait/target-price features are forbidden: "
            + ", ".join(sorted(forbidden))
        )
    return names


def validate_canonical_exit_labels(frame: pd.DataFrame) -> None:
    required = {
        "execution_exit_class",
        "exit_is_trailing",
        "exit_is_timeout",
        "exit_is_full_stop",
        "exit_is_adverse_exit",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"canonical exit labels are missing {missing}")
    labels = frame["execution_exit_class"].astype(str)
    unknown = sorted(set(labels) - set(EXIT_CLASSES))
    if unknown:
        raise ValueError(f"canonical exit labels contain unknown classes: {unknown}")
    flags = frame.loc[
        :,
        [
            "exit_is_trailing",
            "exit_is_timeout",
            "exit_is_full_stop",
            "exit_is_adverse_exit",
        ],
    ].astype(bool)
    if not flags.sum(axis=1).eq(1).all():
        raise ValueError("canonical exit flags must be mutually exclusive and exhaustive")
    expected = pd.DataFrame(
        {
            "exit_is_trailing": labels.eq("trailing"),
            "exit_is_timeout": labels.eq("timeout"),
            "exit_is_full_stop": labels.eq("full_stop"),
            "exit_is_adverse_exit": labels.eq("adverse_exit"),
        },
        index=frame.index,
    )
    if not flags.equals(expected):
        raise ValueError("canonical exit flags disagree with execution_exit_class")


def strict_chronological_folds(
    frame: pd.DataFrame,
    windows: Sequence[tuple[str | pd.Timestamp, str | pd.Timestamp]],
    decision_col: str = "__ts__",
    resolution_col: str = "execution_label_end_utc",
) -> list[dict[str, Any]]:
    decision = pd.to_datetime(frame[decision_col], utc=True, errors="raise")
    resolution = pd.to_datetime(frame[resolution_col], utc=True, errors="raise")
    folds: list[dict[str, Any]] = []
    for number, (start_value, end_value) in enumerate(windows):
        start = pd.Timestamp(start_value)
        end = pd.Timestamp(end_value)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        train = np.flatnonzero(decision.lt(start).to_numpy() & resolution.lt(start).to_numpy())
        validation = np.flatnonzero(decision.ge(start).to_numpy() & decision.lt(end).to_numpy())
        if not len(train) or not len(validation):
            raise ValueError(f"strict fold {number} has empty train or validation rows")
        if not resolution.iloc[train].lt(start).all():
            raise AssertionError("strict fold admits unresolved training labels")
        folds.append(
            {
                "fold": number,
                "start": start,
                "end": end,
                "train_positions": train,
                "validation_positions": validation,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "max_train_resolution": resolution.iloc[train].max(),
            }
        )
    return folds


def stable_top_k_mask(
    scores: Sequence[float] | np.ndarray,
    candidate_ids: Sequence[str] | np.ndarray,
    k: int,
) -> np.ndarray:
    score = np.asarray(scores, dtype=float)
    candidate = np.asarray(candidate_ids, dtype=str)
    if len(score) != len(candidate) or not np.isfinite(score).all():
        raise ValueError("top-k inputs must be equal-length with finite scores")
    count = max(0, min(int(k), len(score)))
    order = np.lexsort((candidate, -score))
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:count]] = True
    return mask


def exit_mixture_from_components(
    probabilities: np.ndarray, conditional_payoffs: np.ndarray
) -> np.ndarray:
    probability = np.asarray(probabilities, dtype=float)
    payoff = np.asarray(conditional_payoffs, dtype=float)
    if probability.shape != payoff.shape or probability.ndim != 2:
        raise ValueError("exit probabilities and payoffs must be equal 2D matrices")
    if probability.shape[1] != len(EXIT_CLASSES):
        raise ValueError("exit mixture requires four canonical exit classes")
    if not np.isfinite(probability).all() or not np.isfinite(payoff).all():
        raise ValueError("exit mixture components must be finite")
    if (probability < -1e-12).any():
        raise ValueError("exit probabilities cannot be negative")
    total = probability.sum(axis=1)
    if not np.allclose(total, 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("exit probabilities must sum to one")
    return np.sum(probability * payoff, axis=1)


def planned_fit_count(n_folds: int = 2, n_sides: int = 2) -> int:
    """Maximum fit count: two-geometry HPO + 12 base + four meta fits/stage."""

    scoring_stages = int(n_sides) * (int(n_folds) + 1)
    base_fits = scoring_stages * (2 + 1 + 3 + 3 + 5)
    meta_stages = int(n_sides) * int(n_folds)
    meta_fits = meta_stages * 4
    return base_fits + meta_fits


def _regressor(geometry: Geometry, seed: int, threads: int, iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=int(iterations),
        depth=int(geometry.depth),
        learning_rate=0.05,
        l2_leaf_reg=float(geometry.l2),
        loss_function="RMSE",
        random_seed=int(seed),
        thread_count=int(threads),
        verbose=False,
        allow_writing_files=False,
    )


def _classifier(geometry: Geometry, seed: int, threads: int, iterations: int, *, multiclass: bool = False) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=int(iterations),
        depth=int(geometry.depth),
        learning_rate=0.05,
        l2_leaf_reg=float(geometry.l2),
        loss_function="MultiClass" if multiclass else "Logloss",
        random_seed=int(seed),
        thread_count=int(threads),
        verbose=False,
        allow_writing_files=False,
    )


def _top10_net(rows: pd.DataFrame, score: np.ndarray) -> float:
    count = int(np.ceil(0.10 * len(rows)))
    selected = stable_top_k_mask(score, rows["candidate_id"], count)
    return float(rows.loc[selected, "execution_net_ev_12h"].mean() * 10_000.0)


def _choose_geometry(
    train: pd.DataFrame,
    features: list[str],
    *,
    seed: int,
    threads: int,
    iterations: int,
) -> tuple[Geometry, list[dict[str, Any]], int]:
    split = pd.Timestamp(train["__ts__"].quantile(0.75))
    fit = _purged_before(train, split)
    validation = train.loc[train["__ts__"].ge(split)].copy()
    if len(fit) < 100 or len(validation) < 100:
        return Geometry(*GEOMETRIES[0]), [{"status": "default_insufficient_inner_rows"}], 0
    x_fit, x_validation = _matrix(fit, validation, features)
    board: list[dict[str, Any]] = []
    best: tuple[float, Geometry] | None = None
    fits = 0
    for number, (depth, l2) in enumerate(GEOMETRIES):
        geometry = Geometry(depth, l2)
        model = _regressor(geometry, seed + number, threads, iterations)
        model.fit(x_fit, fit["execution_net_ev_12h"])
        prediction = model.predict(x_validation)
        fits += 1
        objective = _top10_net(validation, prediction)
        board.append({"depth": depth, "l2": l2, "validation_top10_net_bps": objective})
        if best is None or objective > best[0]:
            best = (objective, geometry)
    assert best is not None
    return best[1], board, fits


def _fit_signed_opportunity(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    x_train: pd.DataFrame,
    x_evaluation: pd.DataFrame,
    *,
    margin: float,
    geometry: Geometry,
    seed: int,
    threads: int,
    iterations: int,
) -> tuple[dict[str, np.ndarray], int]:
    opportunity = (
        train["execution_gross_ev_12h"].to_numpy(float)
        - train["execution_cost_return"].to_numpy(float)
        - float(margin)
    )
    event = opportunity > 0.0
    if np.unique(event).size < 2:
        probability = np.full(len(evaluation), float(event.mean()))
        fits = 0
    else:
        classifier = _classifier(geometry, seed, threads, iterations)
        classifier.fit(x_train, event.astype(int))
        probability = classifier.predict_proba(x_evaluation)[:, 1]
        fits = 1

    def conditional(values: np.ndarray, mask: np.ndarray, local_seed: int) -> tuple[np.ndarray, int]:
        if int(mask.sum()) < 64:
            prior = float(values[mask].mean()) if mask.any() else 0.0
            return np.full(len(evaluation), prior), 0
        clipped = np.minimum(values[mask], float(np.quantile(values[mask], 0.995)))
        scale = max(float(np.median(clipped[clipped > 0])) if np.any(clipped > 0) else 0.0, 1e-4)
        model = _regressor(geometry, local_seed, threads, iterations)
        model.fit(x_train.loc[mask], np.log1p(clipped / scale))
        return np.maximum(np.expm1(model.predict(x_evaluation)) * scale, 0.0), 1

    win, win_fits = conditional(np.maximum(opportunity, 0.0), event, seed + 101)
    loss, loss_fits = conditional(np.maximum(-opportunity, 0.0), ~event, seed + 102)
    return {
        "probability": np.clip(probability, 0.0, 1.0),
        "conditional_win": win,
        "conditional_loss": loss,
        "score": probability * win - (1.0 - probability) * loss,
    }, fits + win_fits + loss_fits


def _fit_exit_payoff(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    x_train: pd.DataFrame,
    x_evaluation: pd.DataFrame,
    *,
    geometry: Geometry,
    seed: int,
    threads: int,
    iterations: int,
) -> tuple[dict[str, np.ndarray], int]:
    labels = train["execution_exit_class"].astype(str)
    unknown = sorted(set(labels) - set(EXIT_CLASSES))
    if unknown:
        raise ValueError(f"unknown canonical exit classes: {unknown}")
    probabilities = np.zeros((len(evaluation), len(EXIT_CLASSES)), dtype=float)
    fits = 0
    present = sorted(labels.unique())
    if len(present) == 1:
        probabilities[:, EXIT_CLASSES.index(present[0])] = 1.0
    else:
        model = _classifier(geometry, seed, threads, iterations, multiclass=True)
        model.fit(x_train, labels)
        predicted = model.predict_proba(x_evaluation)
        for local, label in enumerate(map(str, model.classes_)):
            probabilities[:, EXIT_CLASSES.index(label)] = predicted[:, local]
        fits += 1
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    payoffs = np.zeros_like(probabilities)
    for index, exit_class in enumerate(EXIT_CLASSES):
        mask = labels.eq(exit_class).to_numpy()
        if int(mask.sum()) < 64:
            payoffs[:, index] = (
                float(train.loc[mask, "execution_net_ev_12h"].mean()) if mask.any() else 0.0
            )
            continue
        model = _regressor(geometry, seed + 101 + index, threads, iterations)
        model.fit(x_train.loc[mask], train.loc[mask, "execution_net_ev_12h"])
        payoffs[:, index] = model.predict(x_evaluation)
        fits += 1
    return {
        "probabilities": probabilities,
        "conditional_payoffs": payoffs,
        "score": exit_mixture_from_components(probabilities, payoffs),
    }, fits


def _fit_base_heads(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    candidate_features: list[str],
    *,
    seed: int,
    threads: int,
    hpo_iterations: int,
    refit_iterations: int,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    selected = _features_by_rank(
        train,
        candidate_features,
        train["execution_net_ev_12h"].to_numpy(float),
        0.80,
    )
    selected = validate_feature_columns(selected)
    geometry, hpo, fits = _choose_geometry(
        train,
        selected,
        seed=seed,
        threads=threads,
        iterations=hpo_iterations,
    )
    x_train, x_evaluation = _matrix(train, evaluation, selected)
    direct_model = _regressor(geometry, seed + 10, threads, refit_iterations)
    direct_model.fit(x_train, train["execution_net_ev_12h"])
    direct = direct_model.predict(x_evaluation)
    fits += 1
    opportunity: dict[int, dict[str, np.ndarray]] = {}
    for offset, margin_bps in enumerate((0, 25)):
        values, local_fits = _fit_signed_opportunity(
            train,
            evaluation,
            x_train,
            x_evaluation,
            margin=margin_bps / 10_000.0,
            geometry=geometry,
            seed=seed + 1000 + offset * 100,
            threads=threads,
            iterations=refit_iterations,
        )
        opportunity[margin_bps] = values
        fits += local_fits
    exit_values, exit_fits = _fit_exit_payoff(
        train,
        evaluation,
        x_train,
        x_evaluation,
        geometry=geometry,
        seed=seed + 2000,
        threads=threads,
        iterations=refit_iterations,
    )
    fits += exit_fits
    output = evaluation.loc[
        :,
        [
            *ID,
            "execution_label_end_utc",
            "execution_gross_ev_12h",
            "execution_cost_return",
            "execution_net_ev_12h",
            "execution_exit_class",
        ],
    ].copy()
    output["score_matched_direct_net"] = direct
    for margin_bps in (0, 25):
        prefix = f"opportunity_{margin_bps}bps"
        output[f"score_{prefix}_signed_magnitude"] = opportunity[margin_bps]["score"]
        output[f"p_{prefix}"] = opportunity[margin_bps]["probability"]
        output[f"{prefix}_conditional_win"] = opportunity[margin_bps]["conditional_win"]
        output[f"{prefix}_conditional_loss"] = opportunity[margin_bps]["conditional_loss"]
    output["score_four_exit_signed_payoff"] = exit_values["score"]
    for index, exit_class in enumerate(EXIT_CLASSES):
        output[f"p_exit_{exit_class}"] = exit_values["probabilities"][:, index]
        output[f"conditional_net_{exit_class}"] = exit_values["conditional_payoffs"][:, index]
    return output, {
        "selected_features": selected,
        "geometry": {"depth": geometry.depth, "l2": geometry.l2},
        "hpo": hpo,
    }, fits


def _fit_meta_heads(
    reference: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    seed: int,
    threads: int,
    iterations: int,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    support_columns = [
        "score_matched_direct_net",
        "score_opportunity_0bps_signed_magnitude",
        "score_opportunity_25bps_signed_magnitude",
        "score_four_exit_signed_payoff",
        "p_opportunity_0bps",
        "p_opportunity_25bps",
        *[f"p_exit_{name}" for name in EXIT_CLASSES],
        *[f"conditional_net_{name}" for name in EXIT_CLASSES],
    ]
    out = evaluation.copy()
    out["score_direct_primary_oof_stack"] = out["score_matched_direct_net"]
    out["score_causal_trust_overlay"] = out["score_matched_direct_net"]
    out["predicted_residual_utility"] = 0.0
    out["predicted_absolute_mapping_error"] = 0.0
    out["predicted_trust"] = 0.0
    audit: dict[str, Any] = {}
    fits = 0
    geometry = Geometry(4, 10.0)
    for side_number, side in enumerate(SIDES):
        train = reference.loc[reference["side_name"].eq(side)].copy()
        score = out.loc[out["side_name"].eq(side)].copy()
        audit[side] = {"reference_rows": int(len(train)), "evaluation_rows": int(len(score))}
        if len(train) < 500 or not len(score):
            audit[side]["status"] = "direct_fallback_insufficient_prior_oof"
            continue
        x_train, x_score = _matrix(train, score, support_columns)
        direct = train["score_matched_direct_net"].to_numpy(float)
        residual_target = train["execution_net_ev_12h"].to_numpy(float) - direct
        stack = _regressor(geometry, seed + side_number * 100, threads, iterations)
        stack.fit(x_train, residual_target)
        predicted_residual = stack.predict(x_score)
        fits += 1
        error = _regressor(geometry, seed + side_number * 100 + 1, threads, iterations)
        error.fit(x_train, np.abs(residual_target))
        predicted_error = np.maximum(error.predict(x_score), 0.0)
        fits += 1
        trust_target = (
            1.0 / (1.0 + np.exp(-np.clip(train["execution_net_ev_12h"].to_numpy(float) / 0.005, -40.0, 40.0)))
        ) * np.exp(-np.abs(residual_target) / 0.02)
        trust = _regressor(geometry, seed + side_number * 100 + 2, threads, iterations)
        trust.fit(x_train, trust_target)
        predicted_trust = np.clip(trust.predict(x_score), 0.0, 1.0)
        fits += 1
        # Fourth bounded meta fit predicts direct-primary net from all support heads.
        primary = _regressor(geometry, seed + side_number * 100 + 3, threads, iterations)
        primary.fit(x_train, train["execution_net_ev_12h"])
        predicted_primary = primary.predict(x_score)
        fits += 1
        index = score.index
        out.loc[index, "score_direct_primary_oof_stack"] = predicted_primary
        out.loc[index, "score_causal_trust_overlay"] = (
            predicted_primary + 0.50 * predicted_residual * predicted_trust - 0.25 * predicted_error
        )
        out.loc[index, "predicted_residual_utility"] = predicted_residual
        out.loc[index, "predicted_absolute_mapping_error"] = predicted_error
        out.loc[index, "predicted_trust"] = predicted_trust
        audit[side]["status"] = "strict_prior_oof_meta"
    return out, audit, fits


def fit_hierarchical_ev_calibration(
    reference: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    score_column: str,
    min_rows: int = 500,
    side_shrinkage: float = 5_000.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = evaluation[score_column].to_numpy(float).copy()
    finite = np.isfinite(reference[score_column]) & np.isfinite(reference["execution_net_ev_12h"])
    pool = reference.loc[finite]
    audit: dict[str, Any] = {"reference_rows": int(len(pool)), "sides": {}}
    if len(pool) < int(min_rows) or pool[score_column].nunique() < 2:
        audit["status"] = "raw_fallback_insufficient_reference"
        return result, audit
    pooled = IsotonicRegression(out_of_bounds="clip").fit(
        pool[score_column], pool["execution_net_ev_12h"]
    )
    pooled_prediction = pooled.predict(evaluation[score_column])
    result = pooled_prediction.copy()
    for side in SIDES:
        reference_side = pool.loc[pool["side_name"].eq(side)]
        target = evaluation["side_name"].eq(side).to_numpy()
        side_audit = {"reference_rows": int(len(reference_side)), "evaluation_rows": int(target.sum())}
        if (
            len(reference_side) >= int(min_rows)
            and reference_side[score_column].nunique() >= 2
            and target.any()
        ):
            local = IsotonicRegression(out_of_bounds="clip").fit(
                reference_side[score_column], reference_side["execution_net_ev_12h"]
            )
            local_prediction = local.predict(evaluation.loc[target, score_column])
            weight = len(reference_side) / (len(reference_side) + float(side_shrinkage))
            result[target] = weight * local_prediction + (1.0 - weight) * pooled_prediction[target]
            side_audit.update({"status": "side_to_pooled_hierarchical", "side_weight": weight})
        else:
            side_audit["status"] = "pooled_fallback"
        audit["sides"][side] = side_audit
    audit["status"] = "hierarchical_train_only"
    return result, audit


def _probability_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    return {
        "rows": int(len(y)),
        "prevalence": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if np.unique(y).size == 2 else None,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_mean_error": float(p.mean() - y.mean()),
    }


def _tail_metrics(frame: pd.DataFrame, score_column: str, fraction: float) -> dict[str, Any]:
    count = int(np.ceil(len(frame) * float(fraction)))
    selected = stable_top_k_mask(frame[score_column], frame["candidate_id"], count)
    rows = frame.loc[selected]
    net = rows["execution_net_ev_12h"]
    side_capacity = rows.groupby("side_name").size().reindex(SIDES, fill_value=0)
    return {
        "rows": int(len(rows)),
        "gross_bps": float(rows["execution_gross_ev_12h"].mean() * 10_000.0),
        "cost_bps": float(rows["execution_cost_return"].mean() * 10_000.0),
        "net_bps": float(net.mean() * 10_000.0),
        "positive_net_precision": float(net.gt(0.0).mean()),
        "side_capacity": {side: int(side_capacity.loc[side]) for side in SIDES},
        "max_side_share": float(side_capacity.max() / max(len(rows), 1)),
    }


def _metrics(frame: pd.DataFrame, score_columns: Mapping[str, str]) -> dict[str, Any]:
    latest_start = pd.Timestamp("2025-04-24", tz="UTC")
    output: dict[str, Any] = {}
    for arm, raw_column in score_columns.items():
        variants = {"raw": raw_column, "hierarchical": f"{raw_column}__hierarchical"}
        output[arm] = {}
        for variant, column in variants.items():
            output[arm][variant] = {
                "global": {
                    f"top_{int(fraction * 100)}pct": _tail_metrics(frame, column, fraction)
                    for fraction in TOP_FRACTIONS
                },
                "latest_week_top_10pct": _tail_metrics(
                    frame.loc[frame["__ts__"].ge(latest_start)], column, 0.10
                ),
            }
    return output


def _load_regime_context(root: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    panel_path = root / "panel.parquet"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "IMMUTABLE_PREENTRY_ONLY_INPUT_PANEL":
        raise ValueError("strict regime context manifest status is invalid")
    expected_panel_hash = manifest.get("outputs_sha256", {}).get("panel.parquet")
    if expected_panel_hash != sha256(panel_path):
        raise ValueError("strict regime context panel hash is invalid")
    declared = list(map(str, manifest.get("feature_columns", [])))
    groups = manifest.get("feature_groups", {})
    selected = [
        name
        for group in REGIME_GROUPS
        for name in map(str, groups.get(group, []))
    ]
    selected = validate_feature_columns(selected)
    if not selected or not set(selected).issubset(declared):
        raise ValueError("strict regime context lacks compact declared groups")
    panel = pd.read_parquet(panel_path, columns=[*ID, *selected])
    if panel.duplicated(ID).any():
        raise ValueError("strict regime context identity is not unique")
    if len(panel) != int(manifest.get("rows", -1)):
        raise ValueError("strict regime context row count differs from its manifest")
    return panel, selected, {
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "panel": {"path": str(panel_path), "sha256": sha256(panel_path)},
    }


def _load_canonical_ledger(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = path.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "historical_score_economics_conversion_ledgers_v1":
        raise ValueError("canonical conversion-ledger manifest schema is invalid")
    resolved = path.resolve()
    records = [
        record
        for record in manifest.get("ledgers", [])
        if Path(record.get("path", "")).resolve() == resolved
    ]
    if len(records) != 1:
        raise ValueError("canonical residual ledger is not uniquely declared")
    record = records[0]
    if (
        record.get("source_family") != "canonical_residual_exact1m_current_spread_cf"
        or not record.get("promotion_eligible")
        or not record.get("exact_policy_parity")
        or record.get("path_frequency") != "exact_1m"
    ):
        raise ValueError("supplied ledger is not the canonical promotion residual tier")
    actual_hash = sha256(path)
    if record.get("sha256") != actual_hash:
        raise ValueError("canonical residual ledger hash mismatch")
    columns = [
        *ID,
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_exit_class",
        "exit_is_trailing",
        "exit_is_timeout",
        "exit_is_full_stop",
        "exit_is_adverse_exit",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if len(frame) != 140_682 or frame.duplicated(ID).any():
        raise ValueError("canonical residual ledger identity contract fails")
    validate_canonical_exit_labels(frame)
    if not np.allclose(
        frame["execution_gross_ev_12h"].to_numpy(float)
        - frame["execution_cost_return"].to_numpy(float),
        frame["execution_net_ev_12h"].to_numpy(float),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("canonical residual ledger economics do not reconcile")
    return frame, {
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "ledger": {"path": str(path), "sha256": actual_hash},
        "rows": int(len(frame)),
        "source_family": record["source_family"],
    }


def _load_control(
    root: Path,
    gate_manifest: Mapping[str, Any],
    gate_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    report_path = root / "report.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "historical_execution_ev_gross_hurdle_decomposition_manifest_v1":
        raise ValueError("frozen v2 control manifest schema is invalid")
    source_gate = manifest.get("source_gate_manifest", {})
    if source_gate.get("strict_identity_sha256") != gate_manifest.get("strict_identity_sha256"):
        raise ValueError("frozen v2 control and v6 gate identity hashes differ")
    if source_gate.get("sha256") != sha256(gate_manifest_path):
        raise ValueError("frozen v2 control is not bound to the supplied v6 manifest")
    if manifest.get("output_sha256", {}).get("report.json") != sha256(report_path):
        raise ValueError("frozen v2 report hash mismatch")
    files: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "report": {"path": str(report_path), "sha256": sha256(report_path)},
        "outputs": {},
    }
    for label, relative in (
        ("march", "plus_risk_peak/march_inner_oof_predictions.parquet"),
        ("april", "plus_risk_peak/april_outer_predictions.parquet"),
    ):
        path = root / relative
        expected = manifest.get("output_sha256", {}).get(relative)
        if expected != sha256(path):
            raise ValueError(f"frozen v2 control hash mismatch: {relative}")
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["method"].eq("direct_net")].copy()
        if frame.duplicated(ID).any():
            raise ValueError("frozen control must be unique after direct_net filtering")
        frame = frame.rename(
            columns={
                "raw_score": "score_frozen_control_raw",
                "common_unit_score": "score_frozen_control_common_unit",
            }
        )
        files[label] = frame
        audit["outputs"][label] = {"path": str(path), "sha256": expected, "rows": int(len(frame))}
    return files["march"], files["april"], audit


def _attach_control(scored: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    columns = [*ID, "score_frozen_control_raw", "score_frozen_control_common_unit"]
    merged = scored.merge(control.loc[:, columns], on=ID, how="left", validate="one_to_one")
    if merged["score_frozen_control_raw"].isna().any():
        raise ValueError("frozen control does not exactly cover scored rows")
    return merged


def _score_column_map() -> dict[str, str]:
    return {
        "frozen_control": "score_frozen_control_common_unit",
        "matched_direct_net": "score_matched_direct_net",
        "opportunity_0bps_signed_magnitude": "score_opportunity_0bps_signed_magnitude",
        "opportunity_25bps_signed_magnitude": "score_opportunity_25bps_signed_magnitude",
        "four_exit_signed_payoff": "score_four_exit_signed_payoff",
        "direct_primary_oof_stack": "score_direct_primary_oof_stack",
        "causal_trust_overlay": "score_causal_trust_overlay",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    partial = args.output_root.with_name(args.output_root.name + ".partial")
    if args.output_root.exists() or partial.exists():
        raise FileExistsError("immutable output or partial already exists")
    frame, gate_manifest = _load_frozen_population(args.gate_root)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["execution_label_end_utc"] = pd.to_datetime(frame["execution_label_end_utc"], utc=True)
    if len(frame) != 140_682:
        raise ValueError("canonical residual tier must contain exactly 140,682 rows")
    if not np.allclose(
        frame["execution_gross_ev_12h"].to_numpy(float)
        - frame["execution_cost_return"].to_numpy(float),
        frame["execution_net_ev_12h"].to_numpy(float),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("canonical gross-cost-net accounting does not reconcile")
    ledger, ledger_audit = _load_canonical_ledger(args.canonical_ledger)
    ledger = ledger.rename(
        columns={
            "execution_gross_ev_12h": "__ledger_gross",
            "execution_cost_return": "__ledger_cost",
            "execution_net_ev_12h": "__ledger_net",
        }
    )
    frame = frame.merge(ledger, on=ID, how="left", validate="one_to_one")
    if frame["execution_exit_class"].isna().any():
        raise ValueError("canonical residual ledger does not cover the v6 population")
    for source, ledger_column in (
        ("execution_gross_ev_12h", "__ledger_gross"),
        ("execution_cost_return", "__ledger_cost"),
        ("execution_net_ev_12h", "__ledger_net"),
    ):
        if not np.allclose(
            frame[source].to_numpy(float),
            frame[ledger_column].to_numpy(float),
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError(f"v6 and canonical ledger disagree on {source}")
    frame = frame.drop(columns=["__ledger_gross", "__ledger_cost", "__ledger_net"])
    validate_canonical_exit_labels(frame)
    regime, regime_features, regime_audit = _load_regime_context(args.regime_context_root)
    frame = frame.merge(regime, on=ID, how="left", validate="one_to_one", suffixes=("", "__regime"))
    for feature in regime_features:
        joined = f"{feature}__regime"
        if joined in frame:
            frame[feature] = frame.pop(joined)
    if frame[regime_features].isna().all(axis=1).any():
        raise ValueError("strict regime context does not cover every residual-tier row")
    candidate_features = validate_feature_columns(
        [*_arm_features(frame)["plus_risk_peak"], *regime_features]
    )
    march_control, april_control, control_audit = _load_control(
        args.baseline_root,
        gate_manifest,
        args.gate_root / "manifest.json",
    )
    march = frame.loc[frame["__ts__"].dt.strftime("%Y-%m").eq("2025-03")].copy()
    april = frame.loc[frame["__ts__"].dt.strftime("%Y-%m").eq("2025-04")].copy()
    folds = strict_chronological_folds(march, OOF_FOLDS)
    march_parts: list[pd.DataFrame] = []
    fold_audit: list[dict[str, Any]] = []
    actual_fits = 0
    prior_oof = pd.DataFrame()
    for fold in folds:
        fold_parts: list[pd.DataFrame] = []
        for side_number, side in enumerate(SIDES):
            side_frame = march.loc[march["side_name"].eq(side)].reset_index(drop=True)
            side_folds = strict_chronological_folds(side_frame, [(fold["start"], fold["end"])])
            local = side_folds[0]
            train = side_frame.iloc[local["train_positions"]].copy()
            evaluation = side_frame.iloc[local["validation_positions"]].copy()
            scored, fit_audit, fits = _fit_base_heads(
                train,
                evaluation,
                candidate_features,
                seed=args.seed + int(fold["fold"]) * 10_000 + side_number * 1_000,
                threads=args.threads,
                hpo_iterations=args.hpo_iterations,
                refit_iterations=args.refit_iterations,
            )
            actual_fits += fits
            fold_parts.append(scored)
            fold_audit.append(
                {
                    "fold": int(fold["fold"]),
                    "side": side,
                    "start": fold["start"],
                    "end": fold["end"],
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(evaluation)),
                    "max_train_resolution": train["execution_label_end_utc"].max(),
                    "head_fit": fit_audit,
                }
            )
        scored_fold = pd.concat(fold_parts, ignore_index=True)
        resolved_prior_oof = (
            prior_oof.loc[
                pd.to_datetime(
                    prior_oof["execution_label_end_utc"], utc=True
                ).lt(fold["start"])
            ].copy()
            if len(prior_oof)
            else prior_oof
        )
        if len(resolved_prior_oof):
            scored_fold, meta_audit, fits = _fit_meta_heads(
                resolved_prior_oof,
                scored_fold,
                seed=args.seed + int(fold["fold"]) * 50_000,
                threads=args.threads,
                iterations=args.meta_iterations,
            )
            actual_fits += fits
        else:
            scored_fold["score_direct_primary_oof_stack"] = scored_fold["score_matched_direct_net"]
            scored_fold["score_causal_trust_overlay"] = scored_fold["score_matched_direct_net"]
            scored_fold["predicted_residual_utility"] = 0.0
            scored_fold["predicted_absolute_mapping_error"] = 0.0
            scored_fold["predicted_trust"] = 0.0
            meta_audit = {"status": "direct_fallback_no_prior_oof"}
        scored_fold["oof_fold"] = int(fold["fold"])
        fold_audit[-1]["meta"] = meta_audit
        march_parts.append(scored_fold)
        prior_oof = pd.concat(march_parts, ignore_index=True)
    march_oof = _attach_control(pd.concat(march_parts, ignore_index=True), march_control)

    april_parts: list[pd.DataFrame] = []
    april_start = pd.Timestamp("2025-04-01", tz="UTC")
    for side_number, side in enumerate(SIDES):
        train = _purged_before(march.loc[march["side_name"].eq(side)], april_start)
        evaluation = april.loc[april["side_name"].eq(side)].copy()
        scored, fit_audit, fits = _fit_base_heads(
            train,
            evaluation,
            candidate_features,
            seed=args.seed + 900_000 + side_number * 1_000,
            threads=args.threads,
            hpo_iterations=args.hpo_iterations,
            refit_iterations=args.refit_iterations,
        )
        actual_fits += fits
        april_parts.append(scored)
        fold_audit.append(
            {
                "fold": "april_untouched",
                "side": side,
                "start": april_start,
                "train_rows": int(len(train)),
                "validation_rows": int(len(evaluation)),
                "max_train_resolution": train["execution_label_end_utc"].max(),
                "head_fit": fit_audit,
            }
        )
    april_scored = pd.concat(april_parts, ignore_index=True)
    resolved_march_oof = march_oof.loc[
        pd.to_datetime(march_oof["execution_label_end_utc"], utc=True).lt(april_start)
    ].copy()
    april_scored, meta_audit, fits = _fit_meta_heads(
        resolved_march_oof,
        april_scored,
        seed=args.seed + 950_000,
        threads=args.threads,
        iterations=args.meta_iterations,
    )
    actual_fits += fits
    april_scored = _attach_control(april_scored, april_control)
    april_scored["oof_fold"] = "april_untouched"

    score_columns = _score_column_map()
    calibration_audit: dict[str, Any] = {}
    for arm, column in score_columns.items():
        calibrated, audit = fit_hierarchical_ev_calibration(
            resolved_march_oof,
            april_scored,
            score_column=column,
            min_rows=args.calibration_min_rows,
            side_shrinkage=args.side_shrinkage,
        )
        april_scored[f"{column}__hierarchical"] = calibrated
        calibration_audit[arm] = audit

    opportunity_metrics = {
        "0bps": _probability_metrics(
            april_scored["execution_gross_ev_12h"].gt(april_scored["execution_cost_return"]),
            april_scored["p_opportunity_0bps"],
        ),
        "25bps": _probability_metrics(
            april_scored["execution_gross_ev_12h"].gt(april_scored["execution_cost_return"] + 0.0025),
            april_scored["p_opportunity_25bps"],
        ),
    }
    metrics = _metrics(april_scored, score_columns)
    gate_arm = "causal_trust_overlay"
    gate_metrics = metrics[gate_arm]["hierarchical"]
    global_top10 = gate_metrics["global"]["top_10pct"]
    latest_top10 = gate_metrics["latest_week_top_10pct"]
    gates = {
        "global_top10_positive": global_top10["net_bps"] > 0.0,
        "latest_week_positive": latest_top10["net_bps"] > 0.0,
        "both_sides_selected": min(global_top10["side_capacity"].values()) > 0,
        "side_share_below_95pct": global_top10["max_side_share"] < 0.95,
        "opportunity_auc_0bps_at_least_0p55": (
            opportunity_metrics["0bps"]["auc"] is not None
            and opportunity_metrics["0bps"]["auc"] >= 0.55
        ),
        "beats_frozen_control": (
            global_top10["net_bps"]
            > metrics["frozen_control"]["hierarchical"]["global"]["top_10pct"]["net_bps"]
        ),
    }
    eligible_for_replay = all(gates.values())
    maximum_fits = planned_fit_count(len(folds), len(SIDES))
    if actual_fits > maximum_fits:
        raise AssertionError(
            f"actual fit count {actual_fits} exceeds bounded plan {maximum_fits}"
        )
    partial.mkdir(parents=True)
    march_path = partial / "march_strict_oof_predictions.parquet"
    april_path = partial / "april_untouched_predictions.parquet"
    atomic_parquet(march_path, march_oof)
    atomic_parquet(april_path, april_scored)
    report = {
        "schema": SCHEMA,
        "status": "complete_no_portfolio_replay" if not eligible_for_replay else "complete_replay_eligible_not_run",
        "contract": {
            "population": "canonical 140,682-row exact-1m residual tier; March development and untouched April",
            "feature_set": "matched plus_risk_peak plus compact strict pre-entry core/transition regime context",
            "exclusions": "timing, MAE, wait/reprice, target-price, realized path and outcome fields excluded from model inputs",
            "folding": "side-local March nested OOF; train decision and label resolution strictly before validation start",
            "calibration": "April hierarchical EV map fitted only on March outer-OOF score/outcome pairs",
            "selection": "one pooled global top-k with score descending and candidate_id ascending tie break",
            "cost": "canonical gross-cost=net; cost is used in labels/evaluation exactly once",
        },
        "planned_maximum_fits": maximum_fits,
        "actual_fits": int(actual_fits),
        "arms": list(FINAL_ARMS),
        "feature_columns": candidate_features,
        "fold_audit": fold_audit,
        "april_meta_audit": meta_audit,
        "calibration_audit": calibration_audit,
        "opportunity_metrics": opportunity_metrics,
        "metrics": metrics,
        "promotion_gates": gates,
        "eligible_for_portfolio_replay": eligible_for_replay,
        "portfolio_replay": "not_run",
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = partial / "report.json"
    atomic_json(report_path, _safe(report))
    runner_path = Path(__file__).resolve()
    manifest = {
        "schema": f"{SCHEMA}_manifest",
        "status": report["status"],
        "runner": {"path": str(runner_path), "sha256": sha256(runner_path)},
        "sources": {
            "v6_manifest": {
                "path": str(args.gate_root / "manifest.json"),
                "sha256": sha256(args.gate_root / "manifest.json"),
            },
            "v2_control": control_audit,
            "regime_context": regime_audit,
            "canonical_ledger": ledger_audit,
        },
        "strict_identity_sha256": gate_manifest["strict_identity_sha256"],
        "outputs": {
            "march_strict_oof_predictions.parquet": sha256(march_path),
            "april_untouched_predictions.parquet": sha256(april_path),
            "report.json": sha256(report_path),
        },
        "planned_maximum_fits": report["planned_maximum_fits"],
        "actual_fits": report["actual_fits"],
        "eligible_for_portfolio_replay": eligible_for_replay,
        "portfolio_replay": "not_run",
    }
    atomic_json(partial / "manifest.json", _safe(manifest))
    partial.replace(args.output_root)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--regime-context-root", type=Path, required=True)
    parser.add_argument("--canonical-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--hpo-iterations", type=int, default=40)
    parser.add_argument("--refit-iterations", type=int, default=120)
    parser.add_argument("--meta-iterations", type=int, default=80)
    parser.add_argument("--calibration-min-rows", type=int, default=500)
    parser.add_argument("--side-shrinkage", type=float, default=5_000.0)
    return parser


def main() -> None:
    report = run(_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "planned_maximum_fits": report["planned_maximum_fits"],
                "actual_fits": report["actual_fits"],
                "eligible_for_portfolio_replay": report["eligible_for_portfolio_replay"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
