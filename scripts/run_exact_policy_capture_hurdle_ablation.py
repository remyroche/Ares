#!/usr/bin/env python3
"""Run a fixed exact-policy opportunity-to-capture hurdle ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import execution_ev_metrics
from extreme_price_movements.execution_ev_model_ablation import (
    fit_train_only_isotonic_ev_mapping,
)
from scripts.diagnose_within_july_opportunity_capture import economic_components
from scripts.run_execution_ev_mixed_period_remedies import (
    ARCHETYPE_COLUMN,
    BASELINE_COLUMN,
    DECISION_COLUMN,
    DEFAULT_WINDOWS,
    IDENTITY_COLUMNS,
    RESOLUTION_COLUMN,
    SIDE_COLUMN,
    TARGET_COLUMN,
    _model_features,
    _temporal_oof_blocks,
    apply_canonical_recent_mapping,
    build_forward_split,
)


SCHEMA = "exact_policy_capture_hurdle_ablation_v4"
SIDES = ("long", "short")
ARMS = (
    "direct_net",
    "direct_gross_minus_exact_cost",
    "opp_only",
    "hurdle_prob",
    "hurdle_ev",
    "capture_gross_mixture_minus_exact_cost",
    "hurdle_capture_guard",
    "clean_probability",
    "competing_clean_probability",
    "atr_soft_favorable_probability",
    "binary_decomposed_ev",
    "competing_decomposed_ev",
    "atr_soft_decomposed_ev",
    "capture_upside_minus_adverse_loss",
    "direct_binary_blend_050",
    "direct_competing_blend_050",
)
OPPORTUNITY_TEMPERATURE = 0.0025
POSITIVE_NET_CAP = 0.10


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_hurdle_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fixed row-cost-aware opportunity, capture and magnitude targets."""

    work = frame.copy()
    mfe = work["execution_mfe_return_12h"].to_numpy(dtype=float)
    gross = work["execution_gross_ev_12h"].to_numpy(dtype=float)
    cost = work["execution_cost_return"].to_numpy(dtype=float)
    net = work[TARGET_COLUMN].to_numpy(dtype=float)
    accounting_error = np.max(np.abs(gross - cost - net))
    if accounting_error > 1e-7:
        raise ValueError(f"gross-cost-net mismatch: {accounting_error}")
    opportunity_margin = mfe - cost
    work["target_opportunity_hard"] = (opportunity_margin > 0.0).astype(np.int8)
    work["target_opportunity_soft"] = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                opportunity_margin / OPPORTUNITY_TEMPERATURE,
                -40.0,
                40.0,
            )
        )
    )
    work["target_capture_positive"] = (net > 0.0).astype(np.int8)
    work["target_positive_net_log_bps"] = np.log1p(
        np.clip(net, 0.0, POSITIVE_NET_CAP) * 10_000.0
    )
    # Gross is a distinct simulator output: it already includes executable
    # spread drag, while `execution_cost_return` is the known fee component.
    # Keep the two quantities separate so an arm can subtract the row's exact
    # decision-time cost once, rather than learning the fee indirectly through
    # a net target.
    work["target_positive_gross_log_bps"] = np.log1p(
        np.clip(gross, 0.0, POSITIVE_NET_CAP) * 10_000.0
    )
    work["target_capture_ratio"] = np.clip(
        np.divide(
            np.maximum(gross, 0.0),
            np.maximum(mfe, 0.0001),
        ),
        0.0,
        1.0,
    )
    outcome = np.select(
        [
            work["timeout"].to_numpy(dtype=bool),
            work["adverse_first"].to_numpy(dtype=bool),
            work["favorable_first"].to_numpy(dtype=bool),
        ],
        [0, 1, 2],
        default=-1,
    ).astype(np.int8)
    if (outcome < 0).any():
        raise ValueError("meaningful-MFE outcome is not mutually exhaustive")
    work["target_competing_outcome"] = outcome
    work["target_clean_soft"] = pd.to_numeric(
        work["soft_label"], errors="raise"
    ).clip(0.0, 1.0)
    # Convert the ATR-normalized soft favorable label into a mutually
    # exclusive timeout/adverse/favorable target distribution.  A near-hit
    # timeout may retain substantial favorable mass; adverse-first rows keep
    # the non-favorable mass on the adverse outcome.  This preserves the
    # economic softness of the label without changing trade side.
    soft_favorable = work["target_clean_soft"].to_numpy(dtype=float)
    soft_adverse = (
        (1.0 - soft_favorable)
        * work["adverse_first"].to_numpy(dtype=float)
    )
    soft_timeout = 1.0 - soft_favorable - soft_adverse
    soft_distribution = np.column_stack(
        [soft_timeout, soft_adverse, soft_favorable]
    )
    if (
        (soft_distribution < -1e-12).any()
        or not np.allclose(soft_distribution.sum(axis=1), 1.0, atol=1e-12)
    ):
        raise ValueError("ATR-soft competing targets must be a probability simplex")
    work["target_soft_timeout"] = np.clip(soft_timeout, 0.0, 1.0)
    work["target_soft_adverse"] = np.clip(soft_adverse, 0.0, 1.0)
    work["target_soft_favorable"] = np.clip(soft_favorable, 0.0, 1.0)
    work["target_adverse_hard"] = work["adverse_first"].astype(np.int8)
    work["target_conditional_net"] = np.clip(net, -0.10, 0.10)
    return work


def compose_hurdle_scores(
    p_opportunity: np.ndarray,
    p_capture: np.ndarray,
    positive_net_log_bps: np.ndarray,
    capture_ratio: np.ndarray,
) -> dict[str, np.ndarray]:
    p_o = np.clip(np.asarray(p_opportunity, dtype=float), 0.0, 1.0)
    p_h = np.clip(np.asarray(p_capture, dtype=float), 0.0, 1.0)
    value = np.clip(
        np.expm1(np.asarray(positive_net_log_bps, dtype=float)) / 10_000.0,
        0.0,
        POSITIVE_NET_CAP,
    )
    ratio = np.clip(np.asarray(capture_ratio, dtype=float), 0.0, 1.0)
    return {
        "opp_only": p_o,
        "hurdle_prob": p_o * p_h,
        "hurdle_ev": p_o * p_h * value,
        "hurdle_capture_guard": p_o * p_h * value * ratio,
    }


def compose_gross_cost_scores(
    direct_gross: np.ndarray,
    p_opportunity: np.ndarray,
    p_capture: np.ndarray,
    positive_gross_log_bps: np.ndarray,
    exact_cost: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return gross-opportunity scores after exactly one known-cost subtraction.

    This is intentionally not a rearrangement of the existing net heads.
    `direct_gross` is independently fitted to the exact gross simulator return.
    The capture mixture estimates the gross return conditional on a net-positive
    capture, weighted by the separately trained opportunity and capture heads.
    Both use the row's frozen deterministic fee only at score composition.
    """

    gross = np.asarray(direct_gross, dtype=float)
    p_o = np.clip(np.asarray(p_opportunity, dtype=float), 0.0, 1.0)
    p_h = np.clip(np.asarray(p_capture, dtype=float), 0.0, 1.0)
    captured_gross = np.clip(
        np.expm1(np.asarray(positive_gross_log_bps, dtype=float)) / 10_000.0,
        0.0,
        POSITIVE_NET_CAP,
    )
    cost = np.asarray(exact_cost, dtype=float)
    if not np.isfinite(cost).all() or (cost < 0.0).any():
        raise ValueError("gross-cost score requires finite nonnegative exact cost")
    return {
        "direct_gross_minus_exact_cost": gross - cost,
        "capture_gross_mixture_minus_exact_cost": p_o * p_h * captured_gross - cost,
    }


def compose_decomposed_scores(
    clean_probability: np.ndarray,
    adverse_probability: np.ndarray,
    competing_probability: np.ndarray,
    timeout_net: np.ndarray,
    adverse_net: np.ndarray,
    favorable_net: np.ndarray,
    direct_net: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compose binary-hurdle and competing-risk conditional-EV scores."""

    favorable = np.clip(np.asarray(clean_probability, dtype=float), 0.0, 1.0)
    adverse = np.clip(np.asarray(adverse_probability, dtype=float), 0.0, 1.0)
    timeout = np.clip(1.0 - favorable - adverse, 0.0, 1.0)
    binary = np.column_stack([timeout, adverse, favorable])
    binary /= np.maximum(binary.sum(axis=1, keepdims=True), 1e-12)
    competing = np.clip(np.asarray(competing_probability, dtype=float), 0.0, 1.0)
    if competing.ndim != 2 or competing.shape[1] != 3:
        raise ValueError("competing probabilities must have timeout/adverse/favorable columns")
    competing /= np.maximum(competing.sum(axis=1, keepdims=True), 1e-12)
    conditional = np.column_stack(
        [
            np.asarray(timeout_net, dtype=float),
            np.asarray(adverse_net, dtype=float),
            np.asarray(favorable_net, dtype=float),
        ]
    )
    binary_ev = np.sum(binary * conditional, axis=1)
    competing_ev = np.sum(competing * conditional, axis=1)
    direct = np.asarray(direct_net, dtype=float)
    return {
        "clean_probability": binary[:, 2],
        "competing_clean_probability": competing[:, 2],
        "binary_decomposed_ev": binary_ev,
        "competing_decomposed_ev": competing_ev,
        "direct_binary_blend_050": 0.5 * direct + 0.5 * binary_ev,
        "direct_competing_blend_050": 0.5 * direct + 0.5 * competing_ev,
    }


def compose_atr_soft_scores(
    soft_competing_probability: np.ndarray,
    timeout_net: np.ndarray,
    adverse_net: np.ndarray,
    favorable_net: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compose the ATR-soft three-way probability and conditional EV arms."""

    probability = np.clip(
        np.asarray(soft_competing_probability, dtype=float), 0.0, 1.0
    )
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError(
            "ATR-soft probabilities must have timeout/adverse/favorable columns"
        )
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    conditional = np.column_stack(
        [
            np.asarray(timeout_net, dtype=float),
            np.asarray(adverse_net, dtype=float),
            np.asarray(favorable_net, dtype=float),
        ]
    )
    return {
        "atr_soft_favorable_probability": probability[:, 2],
        "atr_soft_decomposed_ev": np.sum(probability * conditional, axis=1),
    }


def compose_capture_adverse_score(
    p_opportunity: np.ndarray,
    p_capture_given_opportunity: np.ndarray,
    positive_net_log_bps: np.ndarray,
    p_adverse: np.ndarray,
    adverse_net: np.ndarray,
) -> np.ndarray:
    """Captured conditional upside minus separately estimated adverse loss."""

    p_capture = np.clip(
        np.asarray(p_opportunity, dtype=float), 0.0, 1.0
    ) * np.clip(
        np.asarray(p_capture_given_opportunity, dtype=float), 0.0, 1.0
    )
    upside = np.clip(
        np.expm1(np.asarray(positive_net_log_bps, dtype=float)) / 10_000.0,
        0.0,
        POSITIVE_NET_CAP,
    )
    adverse_probability = np.clip(np.asarray(p_adverse, dtype=float), 0.0, 1.0)
    adverse_loss = np.clip(
        -np.asarray(adverse_net, dtype=float), 0.0, POSITIVE_NET_CAP
    )
    return p_capture * upside - adverse_probability * adverse_loss


def _regressor(*, iterations: int, seed: int, n_jobs: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
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


def _classifier(*, iterations: int, seed: int, n_jobs: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="Logloss",
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


def _multiclass_classifier(*, iterations: int, seed: int, n_jobs: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="MultiClass",
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


def _fit_or_constant_classifier(
    x: pd.DataFrame,
    target: np.ndarray,
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[Any | None, float]:
    values = np.asarray(target, dtype=np.int8)
    mean = float(values.mean())
    if len(values) < 200 or np.unique(values).size < 2:
        return None, mean
    model = _classifier(iterations=iterations, seed=seed, n_jobs=n_jobs)
    model.fit(x, values)
    return model, mean


def _predict_classifier(
    model: Any | None,
    constant: float,
    x: pd.DataFrame,
) -> np.ndarray:
    if model is None:
        return np.full(len(x), constant, dtype=float)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def _fit_or_constant_regressor(
    x: pd.DataFrame,
    target: np.ndarray,
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[Any | None, float]:
    values = np.asarray(target, dtype=float)
    median = float(np.median(values))
    if len(values) < 200:
        return None, median
    model = _regressor(iterations=iterations, seed=seed, n_jobs=n_jobs)
    model.fit(x, values)
    return model, median


def _predict_regressor(
    model: Any | None,
    constant: float,
    x: pd.DataFrame,
) -> np.ndarray:
    if model is None:
        return np.full(len(x), constant, dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def _fit_raw_heads(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    fit_x = _model_features(fit, fit, feature_columns, trust_composites=False)
    score_x = _model_features(fit, score, feature_columns, trust_composites=False)

    direct_model = _regressor(iterations=iterations, seed=seed, n_jobs=n_jobs)
    direct_model.fit(
        fit_x,
        fit[TARGET_COLUMN].to_numpy(dtype=float)
        - fit[BASELINE_COLUMN].to_numpy(dtype=float),
    )
    direct = (
        score[BASELINE_COLUMN].to_numpy(dtype=float)
        + np.asarray(direct_model.predict(score_x), dtype=float)
    )

    # A separate residual model is necessary here: direct_net is trained on
    # gross-minus-cost already, so it cannot establish whether modelling gross
    # first and subtracting the exact known fee changes the ranking.
    direct_gross_model = _regressor(
        iterations=iterations, seed=seed + 1, n_jobs=n_jobs
    )
    direct_gross_model.fit(
        fit_x,
        fit["execution_gross_ev_12h"].to_numpy(dtype=float)
        - fit[BASELINE_COLUMN].to_numpy(dtype=float),
    )
    direct_gross = (
        score[BASELINE_COLUMN].to_numpy(dtype=float)
        + np.asarray(direct_gross_model.predict(score_x), dtype=float)
    )

    opportunity_model, opportunity_constant = _fit_or_constant_regressor(
        fit_x,
        fit["target_opportunity_soft"].to_numpy(dtype=float),
        iterations=iterations,
        seed=seed + 11,
        n_jobs=n_jobs,
    )
    raw_opportunity = np.clip(
        _predict_regressor(opportunity_model, opportunity_constant, score_x),
        0.0,
        1.0,
    )

    opportunity_mask = fit["target_opportunity_hard"].to_numpy(dtype=bool)
    capture_x = fit_x.loc[opportunity_mask]
    capture_target = fit.loc[
        opportunity_mask, "target_capture_positive"
    ].to_numpy(dtype=np.int8)
    capture_model, capture_constant = _fit_or_constant_classifier(
        capture_x,
        capture_target,
        iterations=iterations,
        seed=seed + 22,
        n_jobs=n_jobs,
    )
    raw_capture = _predict_classifier(capture_model, capture_constant, score_x)

    positive_mask = opportunity_mask & fit["target_capture_positive"].to_numpy(
        dtype=bool
    )
    magnitude_model, magnitude_constant = _fit_or_constant_regressor(
        fit_x.loc[positive_mask],
        fit.loc[positive_mask, "target_positive_net_log_bps"].to_numpy(
            dtype=float
        ),
        iterations=iterations,
        seed=seed + 33,
        n_jobs=n_jobs,
    )
    magnitude = _predict_regressor(
        magnitude_model, magnitude_constant, score_x
    )

    gross_magnitude_model, gross_magnitude_constant = _fit_or_constant_regressor(
        fit_x.loc[positive_mask],
        fit.loc[positive_mask, "target_positive_gross_log_bps"].to_numpy(
            dtype=float
        ),
        iterations=iterations,
        seed=seed + 34,
        n_jobs=n_jobs,
    )
    gross_magnitude = _predict_regressor(
        gross_magnitude_model, gross_magnitude_constant, score_x
    )

    ratio_model, ratio_constant = _fit_or_constant_regressor(
        capture_x,
        fit.loc[opportunity_mask, "target_capture_ratio"].to_numpy(dtype=float),
        iterations=iterations,
        seed=seed + 44,
        n_jobs=n_jobs,
    )
    ratio = np.clip(
        _predict_regressor(ratio_model, ratio_constant, score_x),
        0.0,
        1.0,
    )
    clean_model, clean_constant = _fit_or_constant_regressor(
        fit_x,
        fit["target_clean_soft"].to_numpy(dtype=float),
        iterations=iterations,
        seed=seed + 55,
        n_jobs=n_jobs,
    )
    raw_clean = np.clip(
        _predict_regressor(clean_model, clean_constant, score_x),
        0.0,
        1.0,
    )
    adverse_model, adverse_constant = _fit_or_constant_classifier(
        fit_x,
        fit["target_adverse_hard"].to_numpy(dtype=np.int8),
        iterations=iterations,
        seed=seed + 66,
        n_jobs=n_jobs,
    )
    raw_adverse = _predict_classifier(
        adverse_model, adverse_constant, score_x
    )
    competing_model = _multiclass_classifier(
        iterations=iterations,
        seed=seed + 77,
        n_jobs=n_jobs,
    )
    competing_model.fit(
        fit_x,
        fit["target_competing_outcome"].to_numpy(dtype=np.int8),
    )
    raw_competing = np.asarray(
        competing_model.predict_proba(score_x), dtype=float
    )
    raw_soft_competing = np.zeros((len(score_x), 3), dtype=float)
    for outcome, (column, offset) in enumerate(
        (
            ("target_soft_timeout", 78),
            ("target_soft_adverse", 79),
            ("target_soft_favorable", 80),
        )
    ):
        soft_model, soft_constant = _fit_or_constant_regressor(
            fit_x,
            fit[column].to_numpy(dtype=float),
            iterations=iterations,
            seed=seed + offset,
            n_jobs=n_jobs,
        )
        raw_soft_competing[:, outcome] = np.clip(
            _predict_regressor(soft_model, soft_constant, score_x),
            0.0,
            1.0,
        )
    conditional_net: dict[int, np.ndarray] = {}
    for outcome, offset in ((0, 88), (1, 99), (2, 110)):
        mask = fit["target_competing_outcome"].to_numpy(dtype=np.int8) == outcome
        outcome_model, outcome_constant = _fit_or_constant_regressor(
            fit_x.loc[mask],
            fit.loc[mask, "target_conditional_net"].to_numpy(dtype=float),
            iterations=iterations,
            seed=seed + offset,
            n_jobs=n_jobs,
        )
        conditional_net[outcome] = _predict_regressor(
            outcome_model, outcome_constant, score_x
        )
    return {
        "direct_net": direct,
        "direct_gross": direct_gross,
        "raw_opportunity": raw_opportunity,
        "raw_capture": raw_capture,
        "positive_net_log_bps": magnitude,
        "positive_gross_log_bps": gross_magnitude,
        "capture_ratio": ratio,
        "raw_clean": raw_clean,
        "raw_adverse": raw_adverse,
        "raw_competing": raw_competing,
        "raw_soft_competing": raw_soft_competing,
        "timeout_net": conditional_net[0],
        "adverse_net": conditional_net[1],
        "favorable_net": conditional_net[2],
    }


def _calibrate_probability(
    oof: np.ndarray,
    target: np.ndarray,
    evaluation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    calibrator = fit_train_only_isotonic_ev_mapping(
        oof,
        np.asarray(target, dtype=float),
        min_rows=24,
    )
    finite = np.isfinite(oof)
    calibrated_oof = np.full(len(oof), np.nan, dtype=float)
    calibrated_oof[finite] = np.clip(
        calibrator.predict(oof[finite]), 0.0, 1.0
    )
    calibrated_evaluation = np.clip(
        calibrator.predict(np.asarray(evaluation, dtype=float)), 0.0, 1.0
    )
    return calibrated_oof, calibrated_evaluation, str(calibrator.status)


def fit_hurdle_scores(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    train_scores = {arm: np.full(len(train), np.nan) for arm in ARMS}
    evaluation_scores = {arm: np.full(len(evaluation), np.nan) for arm in ARMS}
    reports: dict[str, Any] = {}
    for side_index, side in enumerate(SIDES):
        train_side = train.loc[train[SIDE_COLUMN].astype(str).eq(side)].copy()
        train_side["__global_position__"] = train_side.index
        train_side = train_side.reset_index(drop=True)
        eval_side = evaluation.loc[
            evaluation[SIDE_COLUMN].astype(str).eq(side)
        ].copy()
        eval_side["__global_position__"] = eval_side.index
        eval_side = eval_side.reset_index(drop=True)
        one_dimensional_heads = (
            "direct_net",
            "direct_gross",
            "raw_opportunity",
            "raw_capture",
            "positive_net_log_bps",
            "positive_gross_log_bps",
            "capture_ratio",
            "raw_clean",
            "raw_adverse",
            "timeout_net",
            "adverse_net",
            "favorable_net",
        )
        side_raw_oof = {
            name: np.full(len(train_side), np.nan)
            for name in one_dimensional_heads
        }
        side_raw_oof["raw_competing"] = np.full(
            (len(train_side), 3), np.nan
        )
        side_raw_oof["raw_soft_competing"] = np.full(
            (len(train_side), 3), np.nan
        )
        fold_reports = []
        for fold_number, (fit_pos, valid_pos) in enumerate(
            _temporal_oof_blocks(train_side, min_train_rows=2_000),
            start=1,
        ):
            fit = train_side.iloc[fit_pos]
            valid = train_side.iloc[valid_pos]
            fold_scores = _fit_raw_heads(
                fit,
                valid,
                feature_columns,
                iterations=iterations,
                seed=seed + 10_000 * side_index + 100 * fold_number,
                n_jobs=n_jobs,
            )
            for name, values in fold_scores.items():
                side_raw_oof[name][valid_pos] = values
            fold_reports.append(
                {
                    "fold": fold_number,
                    "fit_rows": int(len(fit)),
                    "validation_rows": int(len(valid)),
                    "max_fit_label_resolution_utc": pd.to_datetime(
                        fit[RESOLUTION_COLUMN], utc=True
                    ).max(),
                    "validation_start_utc": pd.to_datetime(
                        valid[DECISION_COLUMN], utc=True
                    ).min(),
                }
            )
        final_scores = _fit_raw_heads(
            train_side,
            eval_side,
            feature_columns,
            iterations=iterations,
            seed=seed + 10_000 * side_index + 9_000,
            n_jobs=n_jobs,
        )
        opportunity_oof, opportunity_eval, opportunity_status = (
            _calibrate_probability(
                side_raw_oof["raw_opportunity"],
                train_side["target_opportunity_hard"].to_numpy(dtype=np.int8),
                final_scores["raw_opportunity"],
            )
        )
        conditional_oof = side_raw_oof["raw_capture"].copy()
        conditional_target = train_side["target_capture_positive"].to_numpy(
            dtype=np.int8
        )
        opportunity_rows = train_side["target_opportunity_hard"].to_numpy(
            dtype=bool
        )
        capture_mapper = fit_train_only_isotonic_ev_mapping(
            conditional_oof[opportunity_rows],
            conditional_target[opportunity_rows].astype(float),
            min_rows=24,
        )
        capture_oof = np.full(len(train_side), np.nan)
        finite_capture = np.isfinite(conditional_oof)
        capture_oof[finite_capture] = np.clip(
            capture_mapper.predict(conditional_oof[finite_capture]), 0.0, 1.0
        )
        capture_eval = np.clip(
            capture_mapper.predict(final_scores["raw_capture"]), 0.0, 1.0
        )
        oof_composites = compose_hurdle_scores(
            opportunity_oof,
            capture_oof,
            side_raw_oof["positive_net_log_bps"],
            side_raw_oof["capture_ratio"],
        )
        eval_composites = compose_hurdle_scores(
            opportunity_eval,
            capture_eval,
            final_scores["positive_net_log_bps"],
            final_scores["capture_ratio"],
        )
        oof_composites["direct_net"] = side_raw_oof["direct_net"]
        eval_composites["direct_net"] = final_scores["direct_net"]
        oof_composites.update(
            compose_gross_cost_scores(
                side_raw_oof["direct_gross"],
                opportunity_oof,
                capture_oof,
                side_raw_oof["positive_gross_log_bps"],
                train_side["execution_cost_return"].to_numpy(dtype=float),
            )
        )
        eval_composites.update(
            compose_gross_cost_scores(
                final_scores["direct_gross"],
                opportunity_eval,
                capture_eval,
                final_scores["positive_gross_log_bps"],
                eval_side["execution_cost_return"].to_numpy(dtype=float),
            )
        )
        clean_oof, clean_eval, clean_status = _calibrate_probability(
            side_raw_oof["raw_clean"],
            train_side["favorable_first"].to_numpy(dtype=np.int8),
            final_scores["raw_clean"],
        )
        adverse_oof, adverse_eval, adverse_status = _calibrate_probability(
            side_raw_oof["raw_adverse"],
            train_side["adverse_first"].to_numpy(dtype=np.int8),
            final_scores["raw_adverse"],
        )
        competing_oof = np.full_like(side_raw_oof["raw_competing"], np.nan)
        competing_eval = np.zeros_like(final_scores["raw_competing"])
        competing_status: dict[str, str] = {}
        outcome_target = train_side["target_competing_outcome"].to_numpy(
            dtype=np.int8
        )
        for outcome, name in ((0, "timeout"), (1, "adverse"), (2, "favorable")):
            mapped_oof, mapped_eval, status = _calibrate_probability(
                side_raw_oof["raw_competing"][:, outcome],
                (outcome_target == outcome).astype(np.int8),
                final_scores["raw_competing"][:, outcome],
            )
            competing_oof[:, outcome] = mapped_oof
            competing_eval[:, outcome] = mapped_eval
            competing_status[name] = status
        finite_competing = np.isfinite(competing_oof).all(axis=1)
        competing_oof[finite_competing] /= np.maximum(
            competing_oof[finite_competing].sum(axis=1, keepdims=True),
            1e-12,
        )
        competing_eval /= np.maximum(
            competing_eval.sum(axis=1, keepdims=True), 1e-12
        )
        soft_competing_oof = np.full_like(
            side_raw_oof["raw_soft_competing"], np.nan
        )
        soft_competing_eval = np.zeros_like(
            final_scores["raw_soft_competing"]
        )
        soft_competing_status: dict[str, str] = {}
        for outcome, (name, target_column) in enumerate(
            (
                ("timeout", "target_soft_timeout"),
                ("adverse", "target_soft_adverse"),
                ("favorable", "target_soft_favorable"),
            )
        ):
            mapped_oof, mapped_eval, status = _calibrate_probability(
                side_raw_oof["raw_soft_competing"][:, outcome],
                train_side[target_column].to_numpy(dtype=float),
                final_scores["raw_soft_competing"][:, outcome],
            )
            soft_competing_oof[:, outcome] = mapped_oof
            soft_competing_eval[:, outcome] = mapped_eval
            soft_competing_status[name] = status
        finite_soft_competing = np.isfinite(soft_competing_oof).all(axis=1)
        soft_competing_oof[finite_soft_competing] /= np.maximum(
            soft_competing_oof[finite_soft_competing].sum(
                axis=1, keepdims=True
            ),
            1e-12,
        )
        soft_competing_eval /= np.maximum(
            soft_competing_eval.sum(axis=1, keepdims=True), 1e-12
        )
        decomposed_oof = compose_decomposed_scores(
            clean_oof,
            adverse_oof,
            competing_oof,
            side_raw_oof["timeout_net"],
            side_raw_oof["adverse_net"],
            side_raw_oof["favorable_net"],
            side_raw_oof["direct_net"],
        )
        decomposed_eval = compose_decomposed_scores(
            clean_eval,
            adverse_eval,
            competing_eval,
            final_scores["timeout_net"],
            final_scores["adverse_net"],
            final_scores["favorable_net"],
            final_scores["direct_net"],
        )
        oof_composites.update(decomposed_oof)
        eval_composites.update(decomposed_eval)
        oof_composites.update(
            compose_atr_soft_scores(
                soft_competing_oof,
                side_raw_oof["timeout_net"],
                side_raw_oof["adverse_net"],
                side_raw_oof["favorable_net"],
            )
        )
        eval_composites.update(
            compose_atr_soft_scores(
                soft_competing_eval,
                final_scores["timeout_net"],
                final_scores["adverse_net"],
                final_scores["favorable_net"],
            )
        )
        oof_composites["capture_upside_minus_adverse_loss"] = (
            compose_capture_adverse_score(
                opportunity_oof,
                capture_oof,
                side_raw_oof["positive_net_log_bps"],
                adverse_oof,
                side_raw_oof["adverse_net"],
            )
        )
        eval_composites["capture_upside_minus_adverse_loss"] = (
            compose_capture_adverse_score(
                opportunity_eval,
                capture_eval,
                final_scores["positive_net_log_bps"],
                adverse_eval,
                final_scores["adverse_net"],
            )
        )
        global_train_positions = train_side["__global_position__"].to_numpy(
            dtype=int
        )
        global_eval_positions = eval_side["__global_position__"].to_numpy(
            dtype=int
        )
        for arm in ARMS:
            raw_oof = np.asarray(oof_composites[arm], dtype=float)
            raw_eval = np.asarray(eval_composites[arm], dtype=float)
            ev_mapper = fit_train_only_isotonic_ev_mapping(
                raw_oof,
                train_side[TARGET_COLUMN].to_numpy(dtype=float),
                min_rows=24,
            )
            finite = np.isfinite(raw_oof)
            mapped_oof = np.full(len(raw_oof), np.nan)
            mapped_oof[finite] = ev_mapper.predict(raw_oof[finite])
            mapped_eval = ev_mapper.predict(raw_eval)
            train_scores[arm][global_train_positions] = mapped_oof
            evaluation_scores[arm][global_eval_positions] = mapped_eval
        reports[side] = {
            "train_rows": int(len(train_side)),
            "evaluation_rows": int(len(eval_side)),
            "oof_rows": int(np.isfinite(side_raw_oof["direct_net"]).sum()),
            "gross_cost_contract": (
                "gross heads are fitted per side on execution_gross_ev_12h; "
                "execution_cost_return is subtracted exactly once at score composition"
            ),
            "opportunity_calibrator": opportunity_status,
            "capture_calibrator": str(capture_mapper.status),
            "clean_calibrator": clean_status,
            "adverse_calibrator": adverse_status,
            "competing_calibrators": competing_status,
            "atr_soft_competing_calibrators": soft_competing_status,
            "folds": fold_reports,
        }
    for arm in ARMS:
        if not np.isfinite(evaluation_scores[arm]).all():
            raise ValueError(f"{arm} failed to score all evaluation rows")
    return train_scores, evaluation_scores, reports


def _metric_rows(
    evaluation: pd.DataFrame,
    score: np.ndarray,
    *,
    window: str,
    arm: str,
    stage: str,
) -> list[dict[str, Any]]:
    rows = []
    for scope in ("pooled_global", "side_long", "side_short"):
        mask = (
            np.ones(len(evaluation), dtype=bool)
            if scope == "pooled_global"
            else evaluation[SIDE_COLUMN]
            .astype(str)
            .eq(scope.removeprefix("side_"))
            .to_numpy()
        )
        sample = evaluation.loc[mask].reset_index(drop=True)
        prediction = np.asarray(score, dtype=float)[mask]
        metric = execution_ev_metrics(
            sample[TARGET_COLUMN].to_numpy(dtype=float),
            prediction,
            top_k_fraction=0.10,
        )
        count = max(1, int(np.ceil(0.10 * len(sample))))
        selected = sample.iloc[
            np.argsort(-prediction, kind="mergesort")[:count]
        ]
        rows.append(
            {
                "window": window,
                "arm": arm,
                "stage": stage,
                "scope": scope,
                **metric,
                **{
                    f"selected_{key}": value
                    for key, value in economic_components(selected).items()
                    if key != "rows"
                },
            }
        )
    return rows


def _replacement_rows(
    evaluation: pd.DataFrame,
    scores: Mapping[str, np.ndarray],
    *,
    window: str,
    stage: str,
) -> list[dict[str, Any]]:
    identity = list(IDENTITY_COLUMNS)
    count = max(1, int(np.ceil(0.10 * len(evaluation))))

    def selected(score: np.ndarray) -> pd.DataFrame:
        return evaluation.iloc[
            np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:count]
        ].copy()

    baseline = selected(scores["direct_net"])
    baseline_ids = pd.MultiIndex.from_frame(baseline[identity])
    baseline_components = economic_components(baseline)
    rows = []
    for arm in ARMS:
        challenger = selected(scores[arm])
        challenger_ids = pd.MultiIndex.from_frame(challenger[identity])
        components = economic_components(challenger)
        delta_mfe = (
            components["mean_path_mfe_bps"]
            - baseline_components["mean_path_mfe_bps"]
        )
        delta_gap = (
            components["mean_mfe_to_gross_gap_bps"]
            - baseline_components["mean_mfe_to_gross_gap_bps"]
        )
        delta_cost = (
            components["mean_cost_bps"] - baseline_components["mean_cost_bps"]
        )
        delta_net = components["mean_net_bps"] - baseline_components["mean_net_bps"]
        rows.append(
            {
                "window": window,
                "stage": stage,
                "arm": arm,
                "baseline": "direct_net",
                "selected_rows": int(count),
                "overlap_rows": int(challenger_ids.isin(baseline_ids).sum()),
                "delta_net_bps": delta_net,
                "delta_mfe_bps": delta_mfe,
                "delta_mfe_to_gross_gap_bps": delta_gap,
                "delta_cost_bps": delta_cost,
                "reconstructed_delta_net_bps": delta_mfe - delta_gap - delta_cost,
                "reconciliation_error_bps": (
                    delta_net - (delta_mfe - delta_gap - delta_cost)
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame = pd.read_parquet(args.input)
    grid = pd.read_parquet(args.label_grid)
    grid = grid.loc[
        grid["grid_name"].eq(args.grid_name) & grid["label_valid"],
        [
            *IDENTITY_COLUMNS,
            "soft_label",
            "favorable_first",
            "adverse_first",
            "timeout",
        ],
    ].copy()
    if grid.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("hurdle label grid contains duplicate identities")
    frame = frame.merge(
        grid,
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    frame = add_hurdle_targets(frame)
    manifest = json.loads(args.feature_manifest.read_text())
    feature_columns = list(manifest["feature_columns"])
    for column in feature_columns:
        prefix = "catboost_archetype__"
        if column.startswith(prefix) and column not in frame:
            level = column[len(prefix) :]
            frame[column] = (
                frame[ARCHETYPE_COLUMN].astype(str).eq(level).astype("float32")
            )
    required = [
        *IDENTITY_COLUMNS,
        DECISION_COLUMN,
        RESOLUTION_COLUMN,
        TARGET_COLUMN,
        BASELINE_COLUMN,
        ARCHETYPE_COLUMN,
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        *feature_columns,
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("hurdle input missing columns: " + ", ".join(missing))
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("hurdle input contains duplicate identities")
    metric_rows = []
    replacement_rows = []
    prediction_parts = []
    fold_report: dict[str, Any] = {}
    for window_index, window in enumerate(DEFAULT_WINDOWS):
        train_positions, evaluation_positions, split = build_forward_split(
            frame, window, purge_hours=args.purge_hours
        )
        train = frame.iloc[train_positions].copy().reset_index(drop=True)
        evaluation = frame.iloc[evaluation_positions].copy().reset_index(drop=True)
        train_scores, evaluation_scores, reports = fit_hurdle_scores(
            train,
            evaluation,
            feature_columns,
            iterations=args.n_estimators,
            seed=args.random_state + 100_000 * window_index,
            n_jobs=args.n_jobs,
        )
        mapped_scores = {}
        for arm in ARMS:
            metric_rows.extend(
                _metric_rows(
                    evaluation,
                    evaluation_scores[arm],
                    window=window.name,
                    arm=arm,
                    stage="pre_recent_mapping",
                )
            )
            mapped, mapping_report = apply_canonical_recent_mapping(
                train,
                evaluation,
                train_scores[arm],
                evaluation_scores[arm],
            )
            mapped_scores[arm] = mapped
            metric_rows.extend(
                _metric_rows(
                    evaluation,
                    mapped,
                    window=window.name,
                    arm=arm,
                    stage="canonical_recent_ev_mapping",
                )
            )
            part = evaluation.loc[:, list(IDENTITY_COLUMNS)].copy()
            part["window"] = window.name
            part["arm"] = arm
            part["raw_ev_score"] = evaluation_scores[arm]
            part["canonical_recent_ev_score"] = mapped
            prediction_parts.append(part)
            reports.setdefault("recent_mapping", {})[arm] = mapping_report
        replacement_rows.extend(
            _replacement_rows(
                evaluation,
                mapped_scores,
                window=window.name,
                stage="canonical_recent_ev_mapping",
            )
        )
        fold_report[window.name] = {"split": split, "models": reports}
    args.output_dir.mkdir(parents=True)
    paths = {
        "metrics": args.output_dir / "hurdle_metrics.csv",
        "replacements": args.output_dir / "hurdle_replacement_decomposition.csv",
        "predictions": args.output_dir / "hurdle_predictions.parquet",
    }
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(replacement_rows).to_csv(paths["replacements"], index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(
        paths["predictions"], index=False
    )
    output_manifest = {
        "schema": SCHEMA,
        "status": "completed_research_oos_not_promotion_evidence",
        "contract": {
            "target_source": "single canonical exact 1m deployed-policy replay",
            "opportunity_soft": "sigmoid((row MFE - row exact cost) / 25bps)",
            "capture": "P(net > 0 | MFE - exact row cost > 0)",
            "positive_magnitude": "log1p(min(positive net,10%) * 10000), conditional on opportunity and positive capture",
            "gross_cost_arms": (
                "direct gross residual prediction minus the row exact deterministic "
                "cost; and P(opportunity) x P(capture|opportunity) x "
                "E[gross|positive capture] minus that same row exact cost. "
                "Gross contains executable spread drag; cost is never learned as a target "
                "or subtracted twice."
            ),
            "capture_guard": "clip(max(gross,0)/max(MFE,1bp),0,1), conditional on opportunity",
            "clean_event": "canonical h12_u1p5atr soft favorable-first label; adverse and timeout remain separate competing risks",
            "atr_soft_competing": "three train-only calibrated regressors target a mutually exclusive timeout/adverse/favorable simplex derived from the ATR-normalized soft favorable label",
            "decomposed_ev": "P(outcome) times side-local conditional exact net for timeout/adverse/favorable; binary and calibrated competing-risk probability arms",
            "capture_adverse_ev": "P(opportunity) x P(capture|opportunity) x E[positive net|capture] minus P(adverse-first) x E[loss|adverse-first]",
            "blends": "fixed 50/50 direct-net and decomposed-EV challengers; no blend HPO",
            "models": "per-side fixed CatBoost geometry; temporal OOF head and EV calibration; no HPO",
            "ranking": "one pooled global top10 after causal recent-EV mapping; no timestamp or side quotas",
            "mfe_semantics": "whole-horizon hindsight diagnostic, not executable PnL",
        },
        "arms": list(ARMS),
        "windows": [window.__dict__ for window in DEFAULT_WINDOWS],
        "inputs": {
            "data": {"path": str(args.input), "sha256": _sha256(args.input)},
            "feature_manifest": {
                "path": str(args.feature_manifest),
                "sha256": _sha256(args.feature_manifest),
            },
            "label_grid": {
                "path": str(args.label_grid),
                "sha256": _sha256(args.label_grid),
                "grid": args.grid_name,
            },
        },
        "feature_columns": feature_columns,
        "model": {
            "iterations": args.n_estimators,
            "n_jobs": args.n_jobs,
            "random_state": args.random_state,
        },
        "folds": fold_report,
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", output_manifest)
    return output_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data_perp/artifacts/"
            "execution_ev_canonical_exact_policy_regime_input_20260727_v3/"
            "joined.parquet"
        ),
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/"
            "execution_ev_context_clean_regime_diagnosis_forward_july19_20260726_v1/"
            "regime_diagnosis_manifest.json"
        ),
    )
    parser.add_argument(
        "--label-grid",
        type=Path,
        default=Path(
            "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/"
            "meaningful_mfe_label_grid.parquet"
        ),
    )
    parser.add_argument("--grid-name", default="h12_u1p5atr")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260727)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
