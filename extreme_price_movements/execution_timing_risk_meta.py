"""Leakage-safe side-local timing and loss-risk execution meta head.

This head complements, but does not replace, the execution-EV head.  It uses
the same declared pre-entry OOF/frozen feature contract and the same expanding
purged folds.  Exit-path labels are created only while fitting; scoring needs
only the serialized side-local models and their pre-entry feature columns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .execution_ev_meta import (
    ChronologicalPurgedSplit,
    FeatureProvenance,
    _oof_fold_provenance,
    _side_values,
    _utc,
    chronological_purged_splits,
    execution_ev_feature_columns,
    validate_execution_ev_feature_provenance,
    validate_execution_ev_training_contract,
)


EXECUTION_TIMING_RISK_BUNDLE_SCHEMA = "execution_timing_risk_side_local_lgbm_bundle_v1"


@dataclass(frozen=True)
class ExecutionTimingRiskTargetSpec:
    """Train-only realized execution outcomes for the 12-hour policy path."""

    net_ev_col: str = "execution_net_ev_12h"
    exit_hour_col: str = "execution_exit_hour"
    exit_reason_col: str = "execution_exit_reason"
    adverse_exit_reasons: tuple[str, ...] = ("full_stop", "adverse_exit")
    canonical_exit_reasons: tuple[str, ...] = (
        "timeout",
        "full_stop",
        "trailing",
        "adverse_exit",
    )
    horizon_hours: float = 12.0


@dataclass(frozen=True)
class TimingRiskTrainerConfig:
    """Deterministic CPU configuration for the side-local auxiliary heads."""

    n_splits: int = 3
    min_train_rows: int = 500
    purge_hours: float = 12.0
    embargo_hours: float = 12.0
    inner_n_splits: int = 2
    early_stopping_rounds: int = 100
    n_estimators: int = 1_500
    random_state: int = 42
    n_jobs: int = 1
    side_col: str = "side_name"
    catboost_archetype_col: str = "catboost_archetype"
    decision_time_col: str = "__ts__"
    label_end_time_col: str | None = None


@dataclass(frozen=True)
class _ConstantBinaryClassifier:
    """Joblib-safe fallback when an authorized side/train window has one class."""

    probability: float

    def predict_proba(self, values: pd.DataFrame) -> np.ndarray:
        p = float(np.clip(self.probability, 0.0, 1.0))
        return np.tile(np.asarray([1.0 - p, p], dtype=float), (len(values), 1))


@dataclass(frozen=True)
class _ProbabilityCalibrator:
    """Joblib-safe Platt map fitted only on held-out chronological predictions."""

    model: Any | None = None

    def predict(self, probability: Sequence[float]) -> np.ndarray:
        raw = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.model is None:
            return raw
        logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        return np.clip(self.model.predict_proba(logit)[:, 1], 0.0, 1.0)


@dataclass
class ExecutionTimingRiskModelBundle:
    """Persistable final timing and loss-risk models plus OOF audit artifacts."""

    schema: str
    config: dict[str, Any]
    target_spec: ExecutionTimingRiskTargetSpec
    provenance: dict[str, FeatureProvenance]
    feature_names: tuple[str, ...]
    models: dict[str, dict[str, Any]]
    report: dict[str, Any]
    oof_predictions: pd.DataFrame = field(repr=False)
    oof_provenance: pd.DataFrame = field(repr=False)


def _required_numeric(frame: pd.DataFrame, column: str, *, role: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Timing/risk {role} is missing required column {column!r}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def timing_risk_target_columns(spec: ExecutionTimingRiskTargetSpec) -> tuple[str, ...]:
    """Return every realized outcome column that must remain train-only."""

    return (
        spec.net_ev_col,
        spec.exit_hour_col,
        spec.exit_reason_col,
    )


def build_execution_timing_risk_targets(
    frame: pd.DataFrame,
    spec: ExecutionTimingRiskTargetSpec = ExecutionTimingRiskTargetSpec(),
) -> pd.DataFrame:
    """Derive conditional timing and adverse/loss labels without exposing features.

    A favorable row has strictly positive net EV and neither a full stop nor an
    adverse exit, and receives its realized exit hour.  Loss/adverse rows remain
    valid for the loss classifier but have no conditional timing target.  This
    avoids teaching the timing model that every loss is a slow 12-hour trade and
    then penalizing the same loss a second time through the risk probability.
    """

    if spec.horizon_hours <= 0.0:
        raise ValueError("Timing/risk horizon_hours must be positive")
    net_ev = _required_numeric(frame, spec.net_ev_col, role="net EV")
    exit_hour = _required_numeric(frame, spec.exit_hour_col, role="exit hour")
    if spec.exit_reason_col not in frame.columns:
        raise ValueError(
            f"Timing/risk exit reason is missing required column {spec.exit_reason_col!r}"
        )
    exit_reason = frame[spec.exit_reason_col].astype("string").str.strip().str.lower()
    reason_valid = exit_reason.isin(tuple(map(str.lower, spec.canonical_exit_reasons)))
    adverse = exit_reason.isin(
        tuple(str(reason).lower() for reason in spec.adverse_exit_reasons)
    ).to_numpy()
    loss = (net_ev <= 0.0) | adverse
    valid_exit = np.isfinite(exit_hour) & (exit_hour > 0.0) & (exit_hour <= float(spec.horizon_hours))
    valid = np.isfinite(net_ev) & reason_valid.to_numpy() & valid_exit
    timing = np.where(valid & ~loss, exit_hour, np.nan)
    return pd.DataFrame(
        {
            "timing_target_hours": timing,
            "loss_risk_target": np.where(valid, loss.astype(float), np.nan),
            "timing_risk_target_valid": valid,
        },
        index=frame.index,
    )


def validate_execution_timing_risk_feature_contract(
    frame: pd.DataFrame,
    provenance: Mapping[str, FeatureProvenance],
    *,
    target_spec: ExecutionTimingRiskTargetSpec = ExecutionTimingRiskTargetSpec(),
    decision_time_col: str = "__ts__",
    require_complete_execution_ev_handoff: bool = False,
) -> list[str]:
    """Validate exactly the execution-EV pre-entry OOF/frozen input contract."""

    if require_complete_execution_ev_handoff:
        names = validate_execution_ev_training_contract(
            frame,
            provenance,
            decision_time_col=decision_time_col,
        )
    else:
        names = execution_ev_feature_columns(
            frame,
            provenance,
            decision_time_col=decision_time_col,
        )
    forbidden = sorted(set(names).intersection(timing_risk_target_columns(target_spec)))
    if forbidden:
        raise ValueError(
            "Timing/risk realized train-only target fields may not be model inputs: "
            + ", ".join(forbidden)
        )
    return names


def _inner_splits(
    frame: pd.DataFrame, config: TimingRiskTrainerConfig
) -> list[ChronologicalPurgedSplit]:
    if len(frame) < max(8, config.min_train_rows):
        return []
    try:
        return chronological_purged_splits(
            frame,
            n_splits=config.inner_n_splits,
            min_train_size=max(4, min(config.min_train_rows, len(frame) // 3)),
            decision_time_col=config.decision_time_col,
            label_end_time_col=config.label_end_time_col,
            horizon_hours=config.purge_hours,
            embargo_hours=config.embargo_hours,
        )
    except ValueError:
        return []


def _regressor_params(config: TimingRiskTrainerConfig) -> dict[str, Any]:
    return {
        "objective": "regression_l1",
        "n_estimators": int(config.n_estimators),
        "learning_rate": 0.03,
        "num_leaves": 16,
        "max_depth": 5,
        "min_child_samples": 32,
        "min_split_gain": 1e-3,
        "reg_alpha": 0.1,
        "reg_lambda": 4.0,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "max_bin": 127,
        "random_state": int(config.random_state),
        "feature_fraction_seed": int(config.random_state),
        "bagging_seed": int(config.random_state),
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": int(config.n_jobs),
        "verbosity": -1,
    }


def _classifier_params(config: TimingRiskTrainerConfig) -> dict[str, Any]:
    params = _regressor_params(config)
    params.update({"objective": "binary", "metric": "binary_logloss"})
    return params


def _fit_regressor(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    early_stop: ChronologicalPurgedSplit | None,
    config: TimingRiskTrainerConfig,
    n_estimators: int | None = None,
) -> tuple[Any, int]:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("LightGBM is required for execution timing/risk training") from exc
    params = _regressor_params(config)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    model = lgb.LGBMRegressor(**params)
    if early_stop is None:
        model.fit(x, y)
    else:
        model.fit(
            x.iloc[early_stop.train_indices],
            y[early_stop.train_indices],
            eval_set=[(x.iloc[early_stop.validation_indices], y[early_stop.validation_indices])],
            callbacks=[lgb.early_stopping(int(config.early_stopping_rounds), verbose=False)],
        )
    return model, int(model.best_iteration_ or params["n_estimators"])


def _fit_classifier(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    early_stop: ChronologicalPurgedSplit | None,
    config: TimingRiskTrainerConfig,
    n_estimators: int | None = None,
) -> tuple[Any, int]:
    unique = np.unique(y.astype(int, copy=False))
    if len(unique) == 1:
        return _ConstantBinaryClassifier(float(unique[0])), 0
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("LightGBM is required for execution timing/risk training") from exc
    params = _classifier_params(config)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    model = lgb.LGBMClassifier(**params)
    if early_stop is None:
        model.fit(x, y)
    else:
        model.fit(
            x.iloc[early_stop.train_indices],
            y[early_stop.train_indices],
            eval_set=[(x.iloc[early_stop.validation_indices], y[early_stop.validation_indices])],
            callbacks=[lgb.early_stopping(int(config.early_stopping_rounds), verbose=False)],
        )
    return model, int(model.best_iteration_ or params["n_estimators"])


def _probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(model.predict_proba(x), dtype=float)
    if probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError("Timing/risk classifier did not return binary probabilities")
    return np.clip(probability[:, 1], 0.0, 1.0)


def _fit_probability_calibrator(
    probability: Sequence[float], target: Sequence[int]
) -> _ProbabilityCalibrator:
    """Fit a regularized Platt map when held-out support is adequate."""

    raw = np.asarray(probability, dtype=float)
    y = np.asarray(target, dtype=int)
    valid = np.isfinite(raw) & np.isin(y, (0, 1))
    raw, y = raw[valid], y[valid]
    counts = np.bincount(y, minlength=2)
    if len(y) < 20 or int(counts.min()) < 5 or np.unique(raw).size < 3:
        return _ProbabilityCalibrator()
    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(raw, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=0.25, solver="lbfgs", max_iter=500)
    model.fit(logit, y)
    return _ProbabilityCalibrator(model)


def timing_priority(loss_probability: Sequence[float], predicted_time_hours: Sequence[float], *, horizon_hours: float = 12.0) -> np.ndarray:
    """Return bounded priority ``(1 - p_loss) * exp(-time / horizon)``."""

    if horizon_hours <= 0.0:
        raise ValueError("Timing/risk priority horizon must be positive")
    probability = np.asarray(loss_probability, dtype=float)
    timing = np.asarray(predicted_time_hours, dtype=float)
    if probability.shape != timing.shape:
        raise ValueError("Timing/risk probability and time arrays must have identical shape")
    valid = np.isfinite(probability) & np.isfinite(timing)
    output = np.full(probability.shape, np.nan, dtype=float)
    output[valid] = (1.0 - np.clip(probability[valid], 0.0, 1.0)) * np.exp(
        -np.clip(timing[valid], 0.0, float(horizon_hours)) / float(horizon_hours)
    )
    return np.clip(output, 0.0, 1.0, out=output)


def _metric_row(part: pd.DataFrame, *, scope: str, side: str | None = None, month: str | None = None) -> dict[str, Any]:
    risk = part["loss_risk_target"].to_numpy(dtype=float)
    probability = part["oof_loss_probability"].to_numpy(dtype=float)
    timing = part["timing_target_hours"].to_numpy(dtype=float)
    predicted_time = part["oof_predicted_time_hours"].to_numpy(dtype=float)
    net_ev = part["realized_net_ev"].to_numpy(dtype=float)
    priority = part["oof_timing_priority"].to_numpy(dtype=float)
    valid_risk = np.isfinite(risk) & np.isfinite(probability)
    valid_timing = valid_risk & (risk < 0.5) & np.isfinite(timing) & np.isfinite(predicted_time)
    auc = float("nan")
    if int(valid_risk.sum()) >= 2 and np.unique(risk[valid_risk].astype(int)).size == 2:
        auc = float(roc_auc_score(risk[valid_risk].astype(int), probability[valid_risk]))
    brier = (
        float(brier_score_loss(risk[valid_risk].astype(int), probability[valid_risk]))
        if int(valid_risk.sum())
        else float("nan")
    )
    valid_priority = np.isfinite(priority) & np.isfinite(net_ev)
    top_rows = 0
    top_mean = float("nan")
    top_sum = float("nan")
    if int(valid_priority.sum()):
        ranked = np.flatnonzero(valid_priority)
        order = np.argsort(-priority[ranked], kind="stable")
        selected = ranked[order[: max(1, int(np.ceil(len(ranked) * 0.10)))]]
        top_rows = int(len(selected))
        top_mean = float(np.mean(net_ev[selected]))
        top_sum = float(np.sum(net_ev[selected]))
    return {
        "scope": scope,
        "side": side,
        "month": month,
        "rows": int(len(part)),
        "loss_rows": int(valid_risk.sum()),
        "loss_rate": float(np.mean(risk[valid_risk])) if int(valid_risk.sum()) else float("nan"),
        "loss_auc": auc,
        "loss_brier": brier,
        "timing_non_loss_rows": int(valid_timing.sum()),
        "timing_mae_non_loss": float(np.mean(np.abs(timing[valid_timing] - predicted_time[valid_timing])) if int(valid_timing.sum()) else np.nan),
        "top10_rows": top_rows,
        "top10_realized_ev_mean": top_mean,
        "top10_realized_ev_sum": top_sum,
    }


def execution_timing_risk_metrics(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    config: TimingRiskTrainerConfig,
    target_spec: ExecutionTimingRiskTargetSpec,
) -> pd.DataFrame:
    """Report OOF diagnostics overall, by side, and by UTC calendar month."""

    ts = _utc(frame[config.decision_time_col], name=config.decision_time_col)
    work = pd.DataFrame(
        {
            "side": frame[config.side_col].astype(str).str.lower().to_numpy(),
            "month": ts.dt.tz_localize(None).dt.to_period("M").astype(str).to_numpy(),
            "realized_net_ev": _required_numeric(frame, target_spec.net_ev_col, role="net EV"),
        },
        index=frame.index,
    ).join(targets.loc[:, ["timing_target_hours", "loss_risk_target"]]).join(
        predictions.loc[:, ["oof_predicted_time_hours", "oof_loss_probability", "oof_timing_priority"]]
    )
    rows = [_metric_row(work, scope="overall")]
    rows.extend(
        _metric_row(part, scope="side", side=str(side))
        for side, part in work.groupby("side", observed=True, sort=True)
    )
    rows.extend(
        _metric_row(part, scope="month", month=str(month))
        for month, part in work.groupby("month", observed=True, sort=True)
    )
    return pd.DataFrame(rows)


def _timing_risk_oof_provenance(
    frame: pd.DataFrame,
    folds: Sequence[ChronologicalPurgedSplit],
    *,
    decision_time_col: str,
) -> pd.DataFrame:
    """Reuse the execution-EV fold ledger while naming this head explicitly."""

    provenance = _oof_fold_provenance(
        frame, folds, decision_time_col=decision_time_col
    )
    return provenance.rename(
        columns={
            "execution_ev_oof_fold": "timing_risk_oof_fold",
            "execution_ev_oof_validation_start_utc": "timing_risk_oof_validation_start_utc",
            "execution_ev_oof_train_decision_cutoff_utc": "timing_risk_oof_train_decision_cutoff_utc",
        }
    )


def train_execution_timing_risk_meta(
    frame: pd.DataFrame,
    provenance: Mapping[str, FeatureProvenance],
    *,
    config: TimingRiskTrainerConfig = TimingRiskTrainerConfig(),
    target_spec: ExecutionTimingRiskTargetSpec = ExecutionTimingRiskTargetSpec(),
) -> ExecutionTimingRiskModelBundle:
    """Fit side-local timing/risk heads with outer OOF-only reported scores."""

    _side_values(frame, config.side_col)
    feature_names = validate_execution_timing_risk_feature_contract(
        frame,
        provenance,
        target_spec=target_spec,
        decision_time_col=config.decision_time_col,
        require_complete_execution_ev_handoff=True,
    )
    targets = build_execution_timing_risk_targets(frame, target_spec)
    active = targets["timing_risk_target_valid"].to_numpy(dtype=bool)
    if not active.any():
        raise ValueError("Timing/risk trainer has no valid target rows")
    folds = chronological_purged_splits(
        frame,
        n_splits=config.n_splits,
        min_train_size=config.min_train_rows,
        decision_time_col=config.decision_time_col,
        label_end_time_col=config.label_end_time_col,
        horizon_hours=config.purge_hours,
        embargo_hours=config.embargo_hours,
    )
    oof_provenance = _timing_risk_oof_provenance(
        frame, folds, decision_time_col=config.decision_time_col
    )
    sides = _side_values(frame, config.side_col)
    x = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    timing = targets["timing_target_hours"].to_numpy(dtype=float)
    risk = targets["loss_risk_target"].to_numpy(dtype=float)
    predicted_time = np.full(len(frame), np.nan, dtype=float)
    loss_probability = np.full(len(frame), np.nan, dtype=float)
    audits: list[dict[str, Any]] = []
    raw_loss_probability = np.full(len(frame), np.nan, dtype=float)
    best_iterations: dict[str, dict[str, list[int]]] = {
        "long": {"time": [], "risk": []},
        "short": {"time": [], "risk": []},
    }
    for fold in folds:
        for side in ("long", "short"):
            train = np.flatnonzero(np.isin(np.arange(len(frame)), fold.train_indices) & (sides == side) & active)
            valid = np.flatnonzero(np.isin(np.arange(len(frame)), fold.validation_indices) & (sides == side) & active)
            if len(train) < 4 or not len(valid):
                audits.append({"fold": fold.fold, "side": side, "status": "insufficient_side_rows", "train_rows": int(len(train)), "validation_rows": int(len(valid))})
                continue
            local_frame = frame.iloc[train].reset_index(drop=True)
            inner = _inner_splits(local_frame, config)
            local_x = x.iloc[train].reset_index(drop=True)
            local_time = timing[train]
            local_risk = risk[train].astype(int)
            risk_stopper = inner[-1] if inner else None
            risk_model, risk_iteration = _fit_classifier(local_x, local_risk, early_stop=risk_stopper, config=config)
            risk_calibrator = _ProbabilityCalibrator()
            if risk_stopper is not None:
                calibration_raw = _probability(
                    risk_model, local_x.iloc[risk_stopper.validation_indices]
                )
                risk_calibrator = _fit_probability_calibrator(
                    calibration_raw, local_risk[risk_stopper.validation_indices]
                )
                risk_model, _ = _fit_classifier(local_x, local_risk, early_stop=None, config=config, n_estimators=risk_iteration or 1)

            winner = np.flatnonzero(local_risk == 0)
            if len(winner) < 4:
                audits.append({"fold": fold.fold, "side": side, "status": "insufficient_non_loss_timing_rows", "train_rows": int(len(train)), "non_loss_rows": int(len(winner))})
                continue
            timing_x = local_x.iloc[winner].reset_index(drop=True)
            timing_y = local_time[winner]
            timing_frame = local_frame.iloc[winner].reset_index(drop=True)
            timing_inner = _inner_splits(timing_frame, config)
            timing_stopper = timing_inner[-1] if timing_inner else None
            time_model, time_iteration = _fit_regressor(timing_x, timing_y, early_stop=timing_stopper, config=config)
            if timing_stopper is not None:
                time_model, _ = _fit_regressor(timing_x, timing_y, early_stop=None, config=config, n_estimators=time_iteration)
            predicted_time[valid] = np.clip(time_model.predict(x.iloc[valid]), 0.0, target_spec.horizon_hours)
            fold_raw_risk = _probability(risk_model, x.iloc[valid])
            raw_loss_probability[valid] = fold_raw_risk
            loss_probability[valid] = risk_calibrator.predict(fold_raw_risk)
            audits.append({"fold": fold.fold, "side": side, "status": "ok", "train_rows": int(len(train)), "non_loss_timing_rows": int(len(winner)), "validation_rows": int(len(valid)), "time_best_iteration": int(time_iteration), "risk_best_iteration": int(risk_iteration), "risk_inner_folds": int(len(inner)), "timing_inner_folds": int(len(timing_inner)), "risk_calibrated": risk_calibrator.model is not None})
            if time_iteration > 0:
                best_iterations[side]["time"].append(int(time_iteration))
            if risk_iteration > 0:
                best_iterations[side]["risk"].append(int(risk_iteration))
    priority = timing_priority(loss_probability, predicted_time, horizon_hours=target_spec.horizon_hours)
    oof_predictions = pd.DataFrame(
        {
            "oof_predicted_time_hours": predicted_time,
            "oof_loss_probability": loss_probability,
            "oof_timing_priority": priority,
        },
        index=frame.index,
    )
    models: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        if not (
            np.isfinite(predicted_time[sides == side]).any()
            and np.isfinite(loss_probability[sides == side]).any()
        ):
            raise ValueError(
                f"Timing/risk side {side!r} has no successful outer-fold OOF evidence"
            )
        train = np.flatnonzero((sides == side) & active)
        if len(train) < 4:
            continue
        local_x = x.iloc[train]
        local_risk = risk[train].astype(int)
        winner = np.flatnonzero(local_risk == 0)
        if len(winner) < 4:
            raise ValueError(f"Timing/risk side {side!r} has insufficient non-loss timing rows")
        time_iterations = best_iterations[side]["time"]
        risk_iterations = best_iterations[side]["risk"]
        final_time_iterations = (
            int(np.median(time_iterations)) if time_iterations else int(config.n_estimators)
        )
        final_risk_iterations = (
            int(np.median(risk_iterations)) if risk_iterations else int(config.n_estimators)
        )
        time_model, time_iteration = _fit_regressor(
            local_x.iloc[winner],
            timing[train][winner],
            early_stop=None,
            config=config,
            n_estimators=final_time_iterations,
        )
        risk_model, risk_iteration = _fit_classifier(
            local_x,
            local_risk,
            early_stop=None,
            config=config,
            n_estimators=final_risk_iterations,
        )
        oof_side = (sides == side) & np.isfinite(raw_loss_probability)
        risk_calibrator = _fit_probability_calibrator(
            raw_loss_probability[oof_side], risk[oof_side].astype(int)
        )
        models[side] = {
            "features": tuple(feature_names),
            "time_model": time_model,
            "risk_model": risk_model,
            "risk_calibrator": risk_calibrator,
            "time_best_iteration": int(time_iteration),
            "risk_best_iteration": int(risk_iteration),
            "final_iteration_contract": "median_positive_outer_fold_best_iteration",
        }
    diagnostics = execution_timing_risk_metrics(
        frame, targets, oof_predictions, config=config, target_spec=target_spec
    )
    report = {
        "schema": EXECUTION_TIMING_RISK_BUNDLE_SCHEMA,
        "provenance_contract": "all model inputs declared pre_entry and oof_or_frozen; availability checked at decision timestamp; realized timing/risk fields are train-only",
        "oof_contract": "outer expanding purged folds; early stopping uses an inner chronological purged split inside each outer training window only",
        "target_contract": "timing is realized exit hour conditional on positive non-adverse execution; risk is net_ev <= 0 OR full_stop OR adverse_exit; canonical reasons and 1..12h exits are required",
        "risk_calibration_contract": "outer OOF probabilities use fold-local Platt maps fitted on inner chronological validation predictions; final side maps use raw outer OOF predictions only",
        "folds": [{"fold": split.fold, "validation_start": split.validation_start.isoformat(), "validation_end": split.validation_end.isoformat(), "purge_hours": split.purge_hours, "embargo_hours": split.embargo_hours} for split in folds],
        "diagnostics": diagnostics,
        "audits": audits,
    }
    return ExecutionTimingRiskModelBundle(
        EXECUTION_TIMING_RISK_BUNDLE_SCHEMA,
        asdict(config),
        target_spec,
        dict(provenance),
        tuple(feature_names),
        models,
        report,
        oof_predictions,
        oof_provenance,
    )


def predict_execution_timing_risk_bundle(
    bundle: ExecutionTimingRiskModelBundle,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Score the final bundle without requesting or deriving realized targets."""

    if bundle.schema != EXECUTION_TIMING_RISK_BUNDLE_SCHEMA:
        raise ValueError("not an execution timing/risk side-local LGBM bundle")
    config = TimingRiskTrainerConfig(**bundle.config)
    _side_values(frame, config.side_col)
    validate_execution_ev_feature_provenance(
        frame,
        bundle.feature_names,
        bundle.provenance,
        decision_time_col=config.decision_time_col,
    )
    forbidden = sorted(set(bundle.feature_names).intersection(timing_risk_target_columns(bundle.target_spec)))
    if forbidden:
        raise ValueError("Timing/risk bundle contains train-only target features: " + ", ".join(forbidden))
    result = pd.DataFrame(index=frame.index)
    predicted_time = np.full(len(frame), np.nan, dtype=float)
    loss_probability = np.full(len(frame), np.nan, dtype=float)
    sides = _side_values(frame, config.side_col)
    for side, state in bundle.models.items():
        position = np.flatnonzero(sides == side)
        if not len(position):
            continue
        features = list(state["features"])
        values = frame.iloc[position].loc[:, features].apply(pd.to_numeric, errors="coerce")
        predicted_time[position] = np.clip(state["time_model"].predict(values), 0.0, bundle.target_spec.horizon_hours)
        raw_probability = _probability(state["risk_model"], values)
        calibrator = state.get("risk_calibrator", _ProbabilityCalibrator())
        loss_probability[position] = calibrator.predict(raw_probability)
    result["predicted_time_hours"] = predicted_time
    result["loss_probability"] = loss_probability
    result["timing_priority"] = timing_priority(
        loss_probability, predicted_time, horizon_hours=bundle.target_spec.horizon_hours
    )
    return result


def save_execution_timing_risk_bundle(
    bundle: ExecutionTimingRiskModelBundle, path: str | Path
) -> Path:
    """Persist the joblib-friendly model bundle and its immutable OOF ledger."""

    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=3)
    return target


def load_execution_timing_risk_bundle(path: str | Path) -> ExecutionTimingRiskModelBundle:
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, ExecutionTimingRiskModelBundle) or bundle.schema != EXECUTION_TIMING_RISK_BUNDLE_SCHEMA:
        raise ValueError("not an execution timing/risk side-local LGBM bundle")
    return bundle


def write_execution_timing_risk_report(
    bundle: ExecutionTimingRiskModelBundle, output_dir: str | Path
) -> dict[str, Path]:
    """Write compact OOF diagnostics and a JSON provenance summary."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    diagnostics_path = root / "execution_timing_risk_diagnostics.csv"
    oof_path = root / "execution_timing_risk_oof_predictions.parquet"
    report_path = root / "execution_timing_risk_report.json"
    bundle.report["diagnostics"].to_csv(diagnostics_path, index=False)
    oof = bundle.oof_predictions.join(bundle.oof_provenance, how="left")
    try:
        oof.to_parquet(oof_path, index=True)
    except (ImportError, ValueError):
        oof_path = root / "execution_timing_risk_oof_predictions.pkl"
        oof.to_pickle(oof_path)
    payload = {key: value for key, value in bundle.report.items() if key != "diagnostics"}
    payload["diagnostics_path"] = diagnostics_path.name
    payload["oof_predictions_path"] = oof_path.name
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {"report": report_path, "diagnostics": diagnostics_path, "oof_predictions": oof_path}
