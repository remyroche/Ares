#!/usr/bin/env python3
"""Strict-OOF conversion of frozen alpha/context into exact-policy EV.

This is deliberately a runner, not a promotion script.  It holds the frozen
base/context panel fixed and compares a direct exact-net residual with a
meaningful-MFE/exit-policy decomposition.  Every support label has its own
availability timestamp and all probability/EV calibration is causal within
each side's chronological OOF sequence.
"""

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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

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
    apply_canonical_recent_mapping,
)


SCHEMA = "strict_oof_ic_ev_conversion_ablation_v1"
SIDES = ("long", "short")
GRID_COLUMNS = (
    "label_resolution_utc",
    "peak_mfe_atr",
    "upper_atr",
    "favorable_first",
    "adverse_first",
    "timeout",
    "early_3bar_adverse_atr",
)
ARMS = (
    "direct_net_residual",
    "meaningful_mfe_capture_minus_adverse_diagnostic",
    "complete_exit_policy_ev",
    "direct_complete_exit_blend_050",
)
MIN_TRAIN_ROWS = 2_000
PAYOFF_CAP = 0.10


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


def prepare_conversion_frame(frame: pd.DataFrame, grid: pd.DataFrame, *, grid_name: str) -> pd.DataFrame:
    """Join immutable labels and fail closed on identity and timing provenance."""

    required_frame = [*IDENTITY_COLUMNS, DECISION_COLUMN, RESOLUTION_COLUMN, TARGET_COLUMN]
    missing_frame = sorted(set(required_frame) - set(frame.columns))
    if missing_frame:
        raise ValueError("frozen input missing columns: " + ", ".join(missing_frame))
    required_grid = [*IDENTITY_COLUMNS, "grid_name", "label_valid", *GRID_COLUMNS]
    missing_grid = sorted(set(required_grid) - set(grid.columns))
    if missing_grid:
        raise ValueError("label grid missing columns: " + ", ".join(missing_grid))
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("frozen input has duplicate immutable identities")
    selected = grid.loc[
        grid["grid_name"].eq(grid_name) & grid["label_valid"].astype(bool),
        [*IDENTITY_COLUMNS, *GRID_COLUMNS],
    ].copy()
    if selected.empty:
        raise ValueError(f"no valid label rows for grid {grid_name!r}")
    if selected.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("label grid has duplicate immutable identities")
    work = frame.merge(selected, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    if work.empty:
        raise ValueError("identity join between frozen panel and label grid is empty")
    decision = pd.to_datetime(work[DECISION_COLUMN], utc=True, errors="raise")
    execution_available = pd.to_datetime(work[RESOLUTION_COLUMN], utc=True, errors="raise")
    label_available = pd.to_datetime(work["label_resolution_utc"], utc=True, errors="raise")
    if (execution_available < decision).any() or (label_available < decision).any():
        raise ValueError("outcome availability precedes decision time")
    work["support_label_available_utc"] = pd.concat(
        [execution_available.rename("execution"), label_available.rename("mfe")], axis=1
    ).max(axis=1)
    # The support path labels must resolve no earlier than the relevant source
    # data.  This max timestamp is subsequently used for every fit/calibration
    # eligibility check, not merely the direct execution label end.
    if (pd.to_datetime(work["support_label_available_utc"], utc=True) < decision).any():
        raise ValueError("combined support-label availability precedes decision time")
    add_conversion_targets(work)
    return work.sort_values([DECISION_COLUMN, "candidate_id"], kind="stable").reset_index(drop=True)


def add_conversion_targets(frame: pd.DataFrame) -> None:
    """Add distinct incidence, path-risk, conversion and complete-state labels."""

    upper = pd.to_numeric(frame["upper_atr"], errors="raise").to_numpy(dtype=float)
    peak = pd.to_numeric(frame["peak_mfe_atr"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(upper).all() or not np.isfinite(peak).all() or (upper <= 0.0).any():
        raise ValueError("meaningful-MFE grid must have finite positive barriers")
    favorable = frame["favorable_first"].astype(bool).to_numpy()
    adverse = frame["adverse_first"].astype(bool).to_numpy()
    timeout = frame["timeout"].astype(bool).to_numpy()
    if not np.all(favorable.astype(int) + adverse.astype(int) + timeout.astype(int) == 1):
        raise ValueError("path outcome must be a mutually exclusive favorable/adverse/timeout simplex")
    net = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(net).all():
        raise ValueError("exact execution net target must be finite")
    frame["target_meaningful_mfe_incidence"] = (peak >= upper).astype(np.int8)
    frame["target_adverse_first"] = adverse.astype(np.int8)
    frame["target_favorable_first"] = favorable.astype(np.int8)
    frame["target_timeout"] = timeout.astype(np.int8)
    frame["target_capture_positive_given_favorable"] = (net > 0.0).astype(np.int8)
    frame["target_capture_positive_given_incidence"] = (net > 0.0).astype(np.int8)
    # Four mutually exclusive states preserve full exact-policy PnL support.
    state = np.select(
        [favorable & (net > 0.0), favorable & (net <= 0.0), adverse, timeout],
        [0, 1, 2, 3],
        default=-1,
    ).astype(np.int8)
    if (state < 0).any():
        raise ValueError("complete exit-policy state is not exhaustive")
    frame["target_exit_state"] = state
    frame["target_exact_net_clipped"] = np.clip(net, -PAYOFF_CAP, PAYOFF_CAP)
    frame["target_positive_net"] = np.clip(net, 0.0, PAYOFF_CAP)


def compose_conversion_scores(
    *,
    direct_net: np.ndarray,
    p_incidence: np.ndarray,
    p_capture_given_incidence: np.ndarray,
    positive_net_given_capture: np.ndarray,
    p_adverse_first: np.ndarray,
    adverse_net: np.ndarray,
    p_favorable_first: np.ndarray,
    p_capture_given_favorable: np.ndarray,
    p_timeout: np.ndarray,
    favorable_positive_net: np.ndarray,
    favorable_nonpositive_net: np.ndarray,
    timeout_net: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compose diagnostic and complete expected-EV scores without cost reuse.

    All payoff inputs are already exact *net* policy outcomes.  This function
    therefore never subtracts row costs and prevents a second deduction.
    """

    direct = np.asarray(direct_net, dtype=float)
    probability = np.column_stack(
        [
            np.asarray(p_incidence, dtype=float),
            np.asarray(p_capture_given_incidence, dtype=float),
            np.asarray(p_adverse_first, dtype=float),
            np.asarray(p_favorable_first, dtype=float),
            np.asarray(p_capture_given_favorable, dtype=float),
            np.asarray(p_timeout, dtype=float),
        ]
    )
    payoff = np.column_stack(
        [
            np.asarray(positive_net_given_capture, dtype=float),
            np.asarray(favorable_positive_net, dtype=float),
            np.asarray(favorable_nonpositive_net, dtype=float),
            np.asarray(adverse_net, dtype=float),
            np.asarray(timeout_net, dtype=float),
        ]
    )
    if len(direct) != len(probability) or len(payoff) != len(probability):
        raise ValueError("conversion-score inputs have inconsistent row counts")

    # The first chronological portion of an OOF ledger is intentionally
    # unscored because it supplies the minimum fitting history.  Preserve rows
    # whose complete model-output record is NaN.  A partially missing row is a
    # model/provenance failure and must not be silently converted into a score.
    model_output = np.column_stack([direct, probability, payoff])
    unscored = np.isnan(model_output).all(axis=1)
    scored = np.isfinite(model_output).all(axis=1)
    pathological = ~(unscored | scored)
    if pathological.any():
        examples = np.flatnonzero(pathological)[:5].tolist()
        raise ValueError(
            "conversion-score rows are partially missing or non-finite: "
            f"{examples}"
        )

    tolerance = 1e-12
    scored_probability = probability[scored]
    if (
        (scored_probability < -tolerance).any()
        or (scored_probability > 1.0 + tolerance).any()
    ):
        raise ValueError("conversion probability heads fall outside [0,1]")
    scored_probability = np.clip(scored_probability, 0.0, 1.0)

    p_i = scored_probability[:, 0]
    p_ci = scored_probability[:, 1]
    p_a = scored_probability[:, 2]
    p_f = scored_probability[:, 3]
    p_cf = scored_probability[:, 4]
    p_t = scored_probability[:, 5]
    three_way = np.column_stack([p_t, p_a, p_f])
    total_mass = three_way.sum(axis=1, keepdims=True)
    if (total_mass <= tolerance).any():
        examples = np.flatnonzero(scored)[
            np.flatnonzero(total_mass[:, 0] <= tolerance)[:5]
        ].tolist()
        raise ValueError(
            "exit-path probability heads have zero or negligible total mass: "
            f"{examples}"
        )
    three_way /= total_mass
    p_t, p_a, p_f = three_way.T
    states = np.column_stack(
        [
            p_f * p_cf,
            p_f * (1.0 - p_cf),
            p_a,
            p_t,
        ]
    )
    state_mass = states.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(states).all()
        or (states < -tolerance).any()
        or (state_mass <= tolerance).any()
    ):
        raise ValueError("complete exit-policy state probabilities are pathological")
    # Explicit normalization absorbs harmless CatBoost/isotonic floating-point
    # drift while retaining the nested favorable/capture construction.
    states = np.clip(states, 0.0, None)
    states /= states.sum(axis=1, keepdims=True)

    rows = len(direct)
    diagnostic = np.full(rows, np.nan, dtype=float)
    complete = np.full(rows, np.nan, dtype=float)
    p_f_positive = np.full(rows, np.nan, dtype=float)
    p_f_nonpositive = np.full(rows, np.nan, dtype=float)
    p_exit_adverse = np.full(rows, np.nan, dtype=float)
    p_exit_timeout = np.full(rows, np.nan, dtype=float)
    scored_payoff = payoff[scored]
    diagnostic[scored] = (
        p_i * p_ci * np.clip(scored_payoff[:, 0], 0.0, PAYOFF_CAP)
        - states[:, 2] * np.clip(-scored_payoff[:, 3], 0.0, PAYOFF_CAP)
    )
    complete[scored] = np.sum(
        states
        * scored_payoff[:, [1, 2, 3, 4]],
        axis=1,
    )
    p_f_positive[scored] = states[:, 0]
    p_f_nonpositive[scored] = states[:, 1]
    p_exit_adverse[scored] = states[:, 2]
    p_exit_timeout[scored] = states[:, 3]
    return {
        "direct_net_residual": direct,
        "meaningful_mfe_capture_minus_adverse_diagnostic": diagnostic,
        "complete_exit_policy_ev": complete,
        "direct_complete_exit_blend_050": 0.5 * direct + 0.5 * complete,
        "p_exit_favorable_positive": p_f_positive,
        "p_exit_favorable_nonpositive": p_f_nonpositive,
        "p_exit_adverse": p_exit_adverse,
        "p_exit_timeout": p_exit_timeout,
    }


def _regressor(*, iterations: int, seed: int, n_jobs: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="MAE", iterations=int(iterations), learning_rate=0.03,
        depth=6, l2_leaf_reg=6.0, random_strength=0.5, bagging_temperature=1.0,
        bootstrap_type="Bayesian", random_seed=int(seed), thread_count=int(n_jobs),
        verbose=False, allow_writing_files=False,
    )


def _classifier(*, iterations: int, seed: int, n_jobs: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="Logloss", iterations=int(iterations), learning_rate=0.03,
        depth=6, l2_leaf_reg=6.0, random_strength=0.5, bagging_temperature=1.0,
        bootstrap_type="Bayesian", random_seed=int(seed), thread_count=int(n_jobs),
        verbose=False, allow_writing_files=False,
    )


def _fit_binary(x: pd.DataFrame, y: np.ndarray, *, iterations: int, seed: int, n_jobs: int) -> tuple[Any | None, float]:
    values = np.asarray(y, dtype=np.int8)
    constant = float(values.mean()) if len(values) else 0.5
    if len(values) < 200 or np.unique(values).size < 2:
        return None, constant
    model = _classifier(iterations=iterations, seed=seed, n_jobs=n_jobs)
    model.fit(x, values)
    return model, constant


def _predict_binary(model: Any | None, constant: float, x: pd.DataFrame) -> np.ndarray:
    return np.full(len(x), constant, dtype=float) if model is None else np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def _fit_regression(x: pd.DataFrame, y: np.ndarray, *, iterations: int, seed: int, n_jobs: int) -> tuple[Any | None, float]:
    values = np.asarray(y, dtype=float)
    constant = float(np.median(values)) if len(values) else 0.0
    if len(values) < 200:
        return None, constant
    model = _regressor(iterations=iterations, seed=seed, n_jobs=n_jobs)
    model.fit(x, values)
    return model, constant


def _predict_regression(model: Any | None, constant: float, x: pd.DataFrame) -> np.ndarray:
    return np.full(len(x), constant, dtype=float) if model is None else np.asarray(model.predict(x), dtype=float)


def _strict_temporal_blocks(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, pd.Timestamp]]:
    """Chronological blocks whose fitting rows have all support labels available."""

    decision = pd.to_datetime(frame[DECISION_COLUMN], utc=True, errors="raise")
    available = pd.to_datetime(frame["support_label_available_utc"], utc=True, errors="raise")
    unique_days = pd.Index(decision.dt.floor("D").unique()).sort_values()
    blocks: list[tuple[np.ndarray, np.ndarray, pd.Timestamp]] = []
    for fraction in (0.40, 0.60, 0.80):
        position = min(max(int(np.floor(fraction * len(unique_days))), 1), len(unique_days) - 1)
        start = pd.Timestamp(unique_days[position])
        later = min(position + max(int(np.ceil(0.20 * len(unique_days))), 1), len(unique_days))
        end = pd.Timestamp(unique_days[later]) if later < len(unique_days) else decision.max() + pd.Timedelta(microseconds=1)
        fit = np.flatnonzero(((decision < start - pd.Timedelta(hours=12)) & (available < start)).to_numpy())
        valid = np.flatnonzero(((decision >= start) & (decision < end)).to_numpy())
        if len(fit) >= MIN_TRAIN_ROWS and len(valid):
            blocks.append((fit, valid, start))
    if not blocks:
        raise ValueError("no strict temporal OOF blocks meet the support-label availability contract")
    return blocks


def _fit_raw_heads(fit: pd.DataFrame, score: pd.DataFrame, feature_columns: Sequence[str], *, iterations: int, seed: int, n_jobs: int) -> dict[str, np.ndarray]:
    fit_x = _model_features(fit, fit, feature_columns, trust_composites=False)
    score_x = _model_features(fit, score, feature_columns, trust_composites=False)
    direct_model, direct_constant = _fit_regression(
        fit_x,
        fit[TARGET_COLUMN].to_numpy(dtype=float) - fit[BASELINE_COLUMN].to_numpy(dtype=float),
        iterations=iterations, seed=seed, n_jobs=n_jobs,
    )
    direct = score[BASELINE_COLUMN].to_numpy(dtype=float) + _predict_regression(direct_model, direct_constant, score_x)
    output: dict[str, np.ndarray] = {"direct_net": direct}
    binary = (
        "target_meaningful_mfe_incidence", "target_adverse_first", "target_favorable_first",
        "target_timeout",
    )
    for offset, target in enumerate(binary, start=10):
        model, constant = _fit_binary(fit_x, fit[target].to_numpy(dtype=np.int8), iterations=iterations, seed=seed + offset, n_jobs=n_jobs)
        output[f"raw_{target.removeprefix('target_')}"] = _predict_binary(model, constant, score_x)
    incidence = fit["target_meaningful_mfe_incidence"].to_numpy(dtype=bool)
    favorable = fit["target_favorable_first"].to_numpy(dtype=bool)
    capture_specs = (
        ("capture_given_incidence", incidence, "target_capture_positive_given_incidence"),
        ("capture_given_favorable", favorable, "target_capture_positive_given_favorable"),
    )
    for offset, (name, mask, target) in enumerate(capture_specs, start=30):
        model, constant = _fit_binary(fit_x.loc[mask], fit.loc[mask, target].to_numpy(dtype=np.int8), iterations=iterations, seed=seed + offset, n_jobs=n_jobs)
        output[f"raw_{name}"] = _predict_binary(model, constant, score_x)
    positive = incidence & fit["target_capture_positive_given_incidence"].to_numpy(dtype=bool)
    model, constant = _fit_regression(fit_x.loc[positive], fit.loc[positive, "target_positive_net"].to_numpy(dtype=float), iterations=iterations, seed=seed + 50, n_jobs=n_jobs)
    output["positive_net_given_capture"] = _predict_regression(model, constant, score_x)
    for offset, (name, mask) in enumerate((("favorable_positive_net", favorable & (fit[TARGET_COLUMN].to_numpy(dtype=float) > 0.0)), ("favorable_nonpositive_net", favorable & (fit[TARGET_COLUMN].to_numpy(dtype=float) <= 0.0)), ("adverse_net", fit["target_adverse_first"].to_numpy(dtype=bool)), ("timeout_net", fit["target_timeout"].to_numpy(dtype=bool))), start=60):
        model, constant = _fit_regression(fit_x.loc[mask], fit.loc[mask, "target_exact_net_clipped"].to_numpy(dtype=float), iterations=iterations, seed=seed + offset, n_jobs=n_jobs)
        output[name] = _predict_regression(model, constant, score_x)
    return output


def _causal_probability_calibration(raw_oof: np.ndarray, target: np.ndarray, fold_id: np.ndarray, availability: pd.Series, raw_eval: np.ndarray, evaluation_start: pd.Timestamp, *, calibration_condition: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Calibrate each OOF fold solely from earlier label-resolved OOF folds."""

    raw_oof = np.asarray(raw_oof, dtype=float)
    target = np.asarray(target, dtype=float)
    condition = np.ones(len(raw_oof), dtype=bool) if calibration_condition is None else np.asarray(calibration_condition, dtype=bool)
    if len(condition) != len(raw_oof):
        raise ValueError("probability calibration condition has wrong length")
    available = pd.to_datetime(availability, utc=True, errors="raise")
    mapped = np.full(len(raw_oof), np.nan, dtype=float)
    reports: list[dict[str, Any]] = []
    for fold in sorted(set(int(item) for item in fold_id if item > 0)):
        valid = fold_id == fold
        earlier = (fold_id > 0) & (fold_id < fold) & condition & (available < available.loc[valid].min()) & np.isfinite(raw_oof)
        mapper = fit_train_only_isotonic_ev_mapping(raw_oof[earlier], target[earlier], min_rows=24)
        mapped[valid] = np.clip(mapper.predict(raw_oof[valid]), 0.0, 1.0)
        reports.append({"fold": fold, "calibration_rows": int(earlier.sum()), "status": str(mapper.status)})
    eligible = (fold_id > 0) & condition & np.isfinite(raw_oof) & (available < evaluation_start)
    final_mapper = fit_train_only_isotonic_ev_mapping(raw_oof[eligible], target[eligible], min_rows=24)
    evaluation = np.clip(final_mapper.predict(np.asarray(raw_eval, dtype=float)), 0.0, 1.0)
    reports.append({"fold": "evaluation", "calibration_rows": int(eligible.sum()), "status": str(final_mapper.status)})
    return mapped, evaluation, reports


def _causal_ev_calibration(raw_oof: np.ndarray, target: np.ndarray, fold_id: np.ndarray, availability: pd.Series, raw_eval: np.ndarray, evaluation_start: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    raw_oof = np.asarray(raw_oof, dtype=float)
    target = np.asarray(target, dtype=float)
    available = pd.to_datetime(availability, utc=True, errors="raise")
    mapped = np.full(len(raw_oof), np.nan, dtype=float)
    for fold in sorted(set(int(item) for item in fold_id if item > 0)):
        valid = fold_id == fold
        earlier = (fold_id > 0) & (fold_id < fold) & (available < available.loc[valid].min()) & np.isfinite(raw_oof)
        mapper = fit_train_only_isotonic_ev_mapping(raw_oof[earlier], target[earlier], min_rows=24)
        mapped[valid] = mapper.predict(raw_oof[valid])
    eligible = (fold_id > 0) & np.isfinite(raw_oof) & (available < evaluation_start)
    mapper = fit_train_only_isotonic_ev_mapping(raw_oof[eligible], target[eligible], min_rows=24)
    return mapped, mapper.predict(np.asarray(raw_eval, dtype=float))


def _fit_side(train: pd.DataFrame, evaluation: pd.DataFrame, feature_columns: Sequence[str], *, iterations: int, seed: int, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return raw/calibrated OOF support ledger and forward score ledger for one side."""

    blocks = _strict_temporal_blocks(train)
    raw_names = (
        "direct_net", "raw_meaningful_mfe_incidence", "raw_adverse_first", "raw_favorable_first", "raw_timeout",
        "raw_capture_given_incidence", "raw_capture_given_favorable", "positive_net_given_capture",
        "favorable_positive_net", "favorable_nonpositive_net", "adverse_net", "timeout_net",
    )
    raw = {name: np.full(len(train), np.nan, dtype=float) for name in raw_names}
    fold_id = np.zeros(len(train), dtype=np.int16)
    folds: list[dict[str, Any]] = []
    for number, (fit_pos, valid_pos, start) in enumerate(blocks, start=1):
        fit, valid = train.iloc[fit_pos], train.iloc[valid_pos]
        output = _fit_raw_heads(fit, valid, feature_columns, iterations=iterations, seed=seed + number * 100, n_jobs=n_jobs)
        for name in raw_names:
            raw[name][valid_pos] = output[name]
        fold_id[valid_pos] = number
        folds.append({"fold": number, "fit_rows": int(len(fit)), "validation_rows": int(len(valid)), "max_fit_support_label_available_utc": pd.to_datetime(fit["support_label_available_utc"], utc=True).max(), "validation_start_utc": start})
    final = _fit_raw_heads(train, evaluation, feature_columns, iterations=iterations, seed=seed + 9_000, n_jobs=n_jobs)
    oof = train.loc[:, [*IDENTITY_COLUMNS, DECISION_COLUMN, "support_label_available_utc", TARGET_COLUMN, "target_meaningful_mfe_incidence", "target_adverse_first", "target_favorable_first", "target_timeout", "target_capture_positive_given_incidence", "target_capture_positive_given_favorable", "target_exit_state"]].copy()
    oof["oof_fold"] = fold_id
    scored = evaluation.loc[:, [*IDENTITY_COLUMNS, DECISION_COLUMN, "support_label_available_utc", TARGET_COLUMN, "target_meaningful_mfe_incidence", "target_adverse_first", "target_favorable_first", "target_timeout", "target_capture_positive_given_incidence", "target_capture_positive_given_favorable", "target_exit_state"]].copy()
    for name in raw_names:
        oof[f"raw_{name}"] = raw[name]
        scored[f"raw_{name}"] = final[name]
    probability_targets = {
        "meaningful_mfe_incidence": "target_meaningful_mfe_incidence",
        "adverse_first": "target_adverse_first",
        "favorable_first": "target_favorable_first",
        "timeout": "target_timeout",
        "capture_given_incidence": "target_capture_positive_given_incidence",
        "capture_given_favorable": "target_capture_positive_given_favorable",
    }
    report: dict[str, Any] = {"folds": folds, "probability_calibration": {}}
    evaluation_start = pd.to_datetime(evaluation[DECISION_COLUMN], utc=True).min()
    for name, target in probability_targets.items():
        # Conditional capture calibrators only observe the stated conditioning event.
        mask = np.ones(len(train), dtype=bool)
        if name == "capture_given_incidence":
            mask = train["target_meaningful_mfe_incidence"].to_numpy(dtype=bool)
        elif name == "capture_given_favorable":
            mask = train["target_favorable_first"].to_numpy(dtype=bool)
        calibrated, final_calibrated, calibration_report = _causal_probability_calibration(
            raw[f"raw_{name}"], train[target].to_numpy(dtype=float), fold_id, train["support_label_available_utc"], final[f"raw_{name}"], evaluation_start, calibration_condition=mask
        )
        oof[f"p_{name}"] = calibrated
        scored[f"p_{name}"] = final_calibrated
        report["probability_calibration"][name] = calibration_report
    raw_oof_scores = compose_conversion_scores(
        direct_net=raw["direct_net"], p_incidence=oof["p_meaningful_mfe_incidence"], p_capture_given_incidence=oof["p_capture_given_incidence"], positive_net_given_capture=raw["positive_net_given_capture"], p_adverse_first=oof["p_adverse_first"], adverse_net=raw["adverse_net"], p_favorable_first=oof["p_favorable_first"], p_capture_given_favorable=oof["p_capture_given_favorable"], p_timeout=oof["p_timeout"], favorable_positive_net=raw["favorable_positive_net"], favorable_nonpositive_net=raw["favorable_nonpositive_net"], timeout_net=raw["timeout_net"],
    )
    raw_final_scores = compose_conversion_scores(
        direct_net=final["direct_net"], p_incidence=scored["p_meaningful_mfe_incidence"], p_capture_given_incidence=scored["p_capture_given_incidence"], positive_net_given_capture=final["positive_net_given_capture"], p_adverse_first=scored["p_adverse_first"], adverse_net=final["adverse_net"], p_favorable_first=scored["p_favorable_first"], p_capture_given_favorable=scored["p_capture_given_favorable"], p_timeout=scored["p_timeout"], favorable_positive_net=final["favorable_positive_net"], favorable_nonpositive_net=final["favorable_nonpositive_net"], timeout_net=final["timeout_net"],
    )
    for name, values in raw_oof_scores.items():
        oof[name] = values
        scored[name] = raw_final_scores[name]
    for arm in ARMS:
        mapped_oof, mapped_final = _causal_ev_calibration(raw_oof_scores[arm], train[TARGET_COLUMN].to_numpy(dtype=float), fold_id, train["support_label_available_utc"], raw_final_scores[arm], evaluation_start)
        oof[f"side_causal_oof_ev_{arm}"] = mapped_oof
        scored[f"side_causal_oof_ev_{arm}"] = mapped_final
    return oof, scored, report


def _head_metric_rows(ledger: pd.DataFrame, *, window: str, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    heads = (
        ("meaningful_mfe_incidence", "target_meaningful_mfe_incidence", "p_meaningful_mfe_incidence", None),
        ("adverse_first", "target_adverse_first", "p_adverse_first", None),
        ("favorable_first", "target_favorable_first", "p_favorable_first", None),
        ("timeout", "target_timeout", "p_timeout", None),
        ("capture_given_incidence", "target_capture_positive_given_incidence", "p_capture_given_incidence", "target_meaningful_mfe_incidence"),
        ("capture_given_favorable", "target_capture_positive_given_favorable", "p_capture_given_favorable", "target_favorable_first"),
    )
    for side in ("pooled_global", *SIDES):
        subset = ledger if side == "pooled_global" else ledger.loc[ledger[SIDE_COLUMN].astype(str).eq(side)]
        for head, target, prediction, condition in heads:
            sample = subset if condition is None else subset.loc[subset[condition].astype(bool)]
            valid = sample[target].notna() & sample[prediction].notna()
            y = sample.loc[valid, target].to_numpy(dtype=int)
            p = sample.loc[valid, prediction].to_numpy(dtype=float)
            if len(y) == 0:
                continue
            row: dict[str, Any] = {"window": window, "stage": stage, "scope": side, "head": head, "rows": int(len(y)), "prevalence": float(y.mean()), "brier": float(brier_score_loss(y, p))}
            if np.unique(y).size == 2:
                row["roc_auc"] = float(roc_auc_score(y, p))
                row["pr_auc"] = float(average_precision_score(y, p))
            rows.append(row)
    return rows


def _arm_metric_rows(evaluation: pd.DataFrame, scores: Mapping[str, np.ndarray], *, window: str, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, prediction in scores.items():
        for scope in ("pooled_global", *SIDES):
            mask = np.ones(len(evaluation), dtype=bool) if scope == "pooled_global" else evaluation[SIDE_COLUMN].astype(str).eq(scope).to_numpy()
            sample = evaluation.loc[mask]
            score = np.asarray(prediction, dtype=float)[mask]
            metric = execution_ev_metrics(sample[TARGET_COLUMN].to_numpy(dtype=float), score, top_k_fraction=0.10)
            count = max(1, int(np.ceil(0.10 * len(sample))))
            selected = sample.iloc[np.argsort(-score, kind="mergesort")[:count]]
            rows.append({"window": window, "stage": stage, "scope": scope, "arm": arm, "eligible_rows": int(len(sample)), "coverage_rate": float(np.isfinite(score).mean()), **metric, **{f"selected_{key}": value for key, value in economic_components(selected).items() if key != "rows"}})
    return rows


def _load_features(frame: pd.DataFrame, manifest_path: Path) -> tuple[pd.DataFrame, list[str]]:
    manifest = json.loads(manifest_path.read_text())
    features = list(manifest["feature_columns"])
    for column in features:
        prefix = "catboost_archetype__"
        if column.startswith(prefix) and column not in frame:
            frame[column] = frame[ARCHETYPE_COLUMN].astype(str).eq(column[len(prefix):]).astype("float32")
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError("frozen feature manifest has missing columns: " + ", ".join(missing))
    prohibited = {TARGET_COLUMN, RESOLUTION_COLUMN, "support_label_available_utc", "label_resolution_utc"}
    leaked = sorted(prohibited.intersection(features))
    if leaked:
        raise ValueError("feature manifest includes outcome/provenance fields: " + ", ".join(leaked))
    return frame, features


def _strict_forward_split(frame: pd.DataFrame, window: Any, *, purge_hours: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    decision = pd.to_datetime(frame[DECISION_COLUMN], utc=True, errors="raise")
    available = pd.to_datetime(frame["support_label_available_utc"], utc=True, errors="raise")
    cutoff, start, end = pd.Timestamp(window.cutoff), pd.Timestamp(window.train_start), pd.Timestamp(window.evaluation_end)
    train = frame.loc[(decision >= start) & (decision < cutoff - pd.Timedelta(hours=purge_hours)) & (available < cutoff)].copy()
    evaluation = frame.loc[(decision >= cutoff) & (decision < end)].copy()
    if train.empty or evaluation.empty:
        raise ValueError(f"strict forward split {window.name!r} is empty")
    if available.loc[train.index].max() >= cutoff:
        raise RuntimeError("forward training includes unavailable support label")
    return train.reset_index(drop=True), evaluation.reset_index(drop=True), {"train_rows": int(len(train)), "evaluation_rows": int(len(evaluation)), "max_train_support_label_available_utc": available.loc[train.index].max(), "evaluation_start_utc": decision.loc[evaluation.index].min()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame = pd.read_parquet(args.input)
    grid = pd.read_parquet(args.label_grid)
    frame = prepare_conversion_frame(frame, grid, grid_name=args.grid_name)
    frame, feature_columns = _load_features(frame, args.feature_manifest)
    for column in (BASELINE_COLUMN, ARCHETYPE_COLUMN, "execution_cost_return"):
        if column not in frame:
            raise ValueError(f"frozen exact-policy panel missing {column!r}")
    oof_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    head_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for index, window in enumerate(DEFAULT_WINDOWS):
        train, evaluation, split = _strict_forward_split(frame, window, purge_hours=args.purge_hours)
        side_oof, side_score, side_report = [], [], {}
        for side_index, side in enumerate(SIDES):
            fit = train.loc[train[SIDE_COLUMN].astype(str).eq(side)].copy().reset_index(drop=True)
            score = evaluation.loc[evaluation[SIDE_COLUMN].astype(str).eq(side)].copy().reset_index(drop=True)
            if fit.empty or score.empty:
                raise ValueError(f"{window.name}: missing {side} rows")
            oof, predicted, report = _fit_side(fit, score, feature_columns, iterations=args.n_estimators, seed=args.random_state + 100_000 * index + 10_000 * side_index, n_jobs=args.n_jobs)
            oof[SIDE_COLUMN] = side
            predicted[SIDE_COLUMN] = side
            side_oof.append(oof)
            side_score.append(predicted)
            side_report[side] = report
        oof = pd.concat(side_oof, ignore_index=True)
        # Side fitting deliberately concatenates long then short.  Restore the
        # immutable frozen-panel order before causal mapping or top-k metrics:
        # score arrays must be positionally aligned with map_eval/evaluation.
        side_prediction = pd.concat(side_score, ignore_index=True)
        prediction = evaluation.loc[:, list(IDENTITY_COLUMNS)].merge(
            side_prediction,
            on=list(IDENTITY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        if len(prediction) != len(evaluation) or prediction["direct_net_residual"].isna().any():
            raise ValueError("side-local forward scores do not cover the frozen evaluation panel")
        # Reuse canonical causal mapping but make the combined source label end
        # the conservative maximum support availability timestamp.
        map_train = train.copy()
        map_train[RESOLUTION_COLUMN] = map_train["support_label_available_utc"]
        map_eval = evaluation.copy()
        map_eval[RESOLUTION_COLUMN] = map_eval["support_label_available_utc"]
        raw_scores, mapped_scores = {}, {}
        for arm in ARMS:
            raw = prediction[arm].to_numpy(dtype=float)
            # The mapper consumes strict side-local OOF score records only.
            # Align those OOF scores back to the frozen training identity order.
            aligned = map_train.loc[:, list(IDENTITY_COLUMNS)].merge(oof.loc[:, [*IDENTITY_COLUMNS, f"side_causal_oof_ev_{arm}"]], on=list(IDENTITY_COLUMNS), how="left", validate="one_to_one")[f"side_causal_oof_ev_{arm}"].to_numpy(dtype=float)
            mapped, mapping_report = apply_canonical_recent_mapping(map_train, map_eval, aligned, raw)
            raw_scores[arm] = raw
            mapped_scores[arm] = mapped
            prediction[f"canonical_recent_ev_score_{arm}"] = mapped
            side_report.setdefault("recent_mapping", {})[arm] = mapping_report
        head_rows.extend(_head_metric_rows(oof, window=window.name, stage="strict_prior_oof"))
        arm_rows.extend(_arm_metric_rows(evaluation, raw_scores, window=window.name, stage="pre_recent_mapping"))
        arm_rows.extend(_arm_metric_rows(evaluation, mapped_scores, window=window.name, stage="canonical_recent_ev_mapping"))
        oof["window"] = window.name
        prediction["window"] = window.name
        oof_parts.append(oof)
        prediction_parts.append(prediction)
        reports[window.name] = {"split": split, "models": side_report}
    args.output_dir.mkdir(parents=True)
    paths = {"support_head_oof_ledger": args.output_dir / "support_head_oof_ledger.parquet", "forward_predictions": args.output_dir / "forward_predictions.parquet", "support_head_metrics": args.output_dir / "support_head_metrics.csv", "arm_metrics": args.output_dir / "arm_metrics.csv"}
    pd.concat(oof_parts, ignore_index=True).to_parquet(paths["support_head_oof_ledger"], index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(paths["forward_predictions"], index=False)
    pd.DataFrame(head_rows).to_csv(paths["support_head_metrics"], index=False)
    pd.DataFrame(arm_rows).to_csv(paths["arm_metrics"], index=False)
    manifest = {"schema": SCHEMA, "status": "completed_research_nonpromotion_evidence", "contract": {"frozen_context": "exact-policy input and approved frozen feature manifest", "support_labels": "h12_u1p5atr: incidence is peak_MFE_ATR >= upper_ATR; path risk is favorable/adverse/timeout first-barrier outcome", "availability": "max(execution_label_end_utc, meaningful-MFE label_resolution_utc) is required for every fit and calibration row", "calibration": "per-side temporal OOF; each OOF calibration fold uses only earlier label-resolved OOF predictions", "arms": "direct exact-net residual; diagnostic incidence x capture x conditional-positive-payoff minus adverse loss; complete four-state exit-policy EV; fixed 50/50 direct blend", "ranking": "one pooled global top10 only after canonical causal recent-EV mapping; no timestamp/side/asset quota", "costs": "all decomposition payoffs are exact net policy outcomes; no second cost subtraction", "auxiliary_actions": "timing/MAE/wait layer intentionally excluded"}, "inputs": {"data": {"path": str(args.input), "sha256": _sha256(args.input)}, "label_grid": {"path": str(args.label_grid), "sha256": _sha256(args.label_grid), "grid": args.grid_name}, "feature_manifest": {"path": str(args.feature_manifest), "sha256": _sha256(args.feature_manifest)}}, "feature_columns": feature_columns, "windows": [window.__dict__ for window in DEFAULT_WINDOWS], "model": {"n_estimators": args.n_estimators, "n_jobs": args.n_jobs, "random_state": args.random_state}, "folds": reports, "outputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()}}
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/joined.parquet"))
    parser.add_argument("--feature-manifest", type=Path, default=Path("data_perp/artifacts/execution_ev_context_clean_regime_diagnosis_forward_july19_20260726_v1/regime_diagnosis_manifest.json"))
    parser.add_argument("--label-grid", type=Path, default=Path("data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/meaningful_mfe_label_grid.parquet"))
    parser.add_argument("--grid-name", default="h12_u1p5atr")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260730)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
