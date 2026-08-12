"""Causal targets and support predictions for residual-ranking ablations.

This module deliberately contains no portfolio selection.  ``support_h12`` is
an ex-post path label: it may be a target, a training weight, or an auxiliary
strict-OOF prediction, but it must never become an inference-time gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


class SupportAwareResidualError(ValueError):
    """Raised when the causal support/residual contract is violated."""


@dataclass(frozen=True)
class SupportPredictionConfig:
    """Fixed, modest support-classifier settings for a predeclared ablation."""

    refit_days: int = 14
    min_train_rows: int = 1_000
    n_estimators: int = 300
    num_leaves: int = 24
    learning_rate: float = 0.04
    min_child_samples: int = 100
    feature_fraction: float = 0.80
    bagging_fraction: float = 0.80
    reg_alpha: float = 1.0
    reg_lambda: float = 4.0

    def validate(self) -> None:
        if self.refit_days < 1 or self.min_train_rows < 32:
            raise SupportAwareResidualError("support refit interval and minimum support rows are invalid")
        if self.n_estimators < 10 or self.num_leaves < 2 or self.min_child_samples < 2:
            raise SupportAwareResidualError("support classifier tree settings are invalid")


def atr_residual_grade(value: Sequence[float] | np.ndarray) -> np.ndarray:
    """Five-grade residual target used by the existing ATR-residual arm."""

    return np.searchsorted(
        np.asarray([-1.5, -0.5, 0.75, 2.0], dtype=np.float64),
        np.asarray(value, dtype=np.float64), side="right",
    ).astype(np.int8)


def bps_residual_grade(
    value: Sequence[float] | np.ndarray, *, moderate_bps: float = 50.0,
    severe_bps: float = 150.0,
) -> np.ndarray:
    """Economic five-grade residual target, symmetric around zero."""

    if not (0.0 < float(moderate_bps) < float(severe_bps)):
        raise SupportAwareResidualError("require 0 < moderate_bps < severe_bps")
    return np.searchsorted(
        np.asarray([-severe_bps, -moderate_bps, moderate_bps, severe_bps], dtype=np.float64),
        np.asarray(value, dtype=np.float64), side="right",
    ).astype(np.int8)


def hybrid_economic_grade(
    atr_residual: Sequence[float] | np.ndarray,
    residual_bps: Sequence[float] | np.ndarray,
    *, moderate_bps: float = 50.0, severe_bps: float = 150.0,
) -> np.ndarray:
    """Keep only residual severity that is material in ATR *and* bps terms.

    Negative corrections take the less-bad grade and positive corrections take
    the less-good grade.  Thus a very large ATR surprise that is only a few
    bps cannot dominate training, nor can a large-bps change be called extreme
    when it is ordinary for the current ATR scale.
    """

    atr = np.asarray(atr_residual, dtype=np.float64)
    bps = np.asarray(residual_bps, dtype=np.float64)
    if atr.shape != bps.shape or not np.isfinite(atr).all() or not np.isfinite(bps).all():
        raise SupportAwareResidualError("ATR and bps residuals must be finite and identically shaped")
    atr_grade = atr_residual_grade(atr)
    economic_grade = bps_residual_grade(
        bps, moderate_bps=moderate_bps, severe_bps=severe_bps,
    )
    return np.where(
        bps < 0.0,
        np.maximum(atr_grade, economic_grade),
        np.where(bps > 0.0, np.minimum(atr_grade, economic_grade), 2),
    ).astype(np.int8)


def query_normalised_support_weights(
    frame: pd.DataFrame, *, support_column: str, query_column: str,
    supported_weight: float,
) -> np.ndarray:
    """Return row weights with unit total weight per timestamp-side query."""

    if float(supported_weight) < 1.0:
        raise SupportAwareResidualError("supported_weight must not downweight supported rows")
    if support_column not in frame or query_column not in frame:
        raise SupportAwareResidualError("support/query columns are absent")
    support = frame[support_column].fillna(False).astype(bool).to_numpy()
    raw = np.where(support, float(supported_weight), 1.0).astype(np.float32)
    sums = pd.Series(raw).groupby(frame[query_column].to_numpy(), sort=False).transform("sum").to_numpy(float)
    if not np.isfinite(sums).all() or (sums <= 0.0).any():
        raise SupportAwareResidualError("invalid query support-weight denominator")
    # Equal *query* total prevents wide cross-sections from mechanically
    # receiving a larger total support-loss weight.
    return (raw / sums).astype(np.float32)


def _fit_support_model(x: pd.DataFrame, y: np.ndarray, *, config: SupportPredictionConfig, seed: int):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        objective="binary", n_estimators=int(config.n_estimators),
        learning_rate=float(config.learning_rate), num_leaves=int(config.num_leaves),
        min_child_samples=int(config.min_child_samples), subsample=float(config.bagging_fraction),
        subsample_freq=1, colsample_bytree=float(config.feature_fraction),
        reg_alpha=float(config.reg_alpha), reg_lambda=float(config.reg_lambda),
        random_state=int(seed), n_jobs=4, verbosity=-1,
    )
    model.fit(x.replace([np.inf, -np.inf], np.nan), y.astype(np.int8, copy=False))
    return model


def strict_prequential_support_probabilities(
    source: pd.DataFrame,
    score: pd.DataFrame,
    *, feature_columns: Sequence[str], support_column: str = "support_h12",
    decision_column: str = "__ts__", label_available_column: str = "label_available_ts",
    config: SupportPredictionConfig = SupportPredictionConfig(), seed: int = 0,
) -> pd.DataFrame:
    """Score rows with support models trained only on prior-resolved labels.

    ``source`` may include the rows being scored.  The strict
    ``label_available < score_block_start`` predicate is enforced for every
    refit block, making in-sample support labels impossible to enter a score.
    The output preserves the input score order and records the fit cutoff.
    """

    config.validate()
    features = tuple(map(str, feature_columns))
    required = {support_column, decision_column, label_available_column, *features}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise SupportAwareResidualError(f"support source misses columns: {missing}")
    missing = sorted({decision_column, *features}.difference(score.columns))
    if missing:
        raise SupportAwareResidualError(f"support scoring frame misses columns: {missing}")
    if support_column in features:
        raise SupportAwareResidualError("the realised support label cannot be a support-model feature")
    source_work = source.copy()
    target = score.copy().reset_index(drop=True)
    source_work[decision_column] = pd.to_datetime(source_work[decision_column], utc=True, errors="coerce")
    source_work[label_available_column] = pd.to_datetime(source_work[label_available_column], utc=True, errors="coerce")
    target[decision_column] = pd.to_datetime(target[decision_column], utc=True, errors="coerce")
    if source_work[[decision_column, label_available_column]].isna().any().any() or target[decision_column].isna().any():
        raise SupportAwareResidualError("support timestamps must be finite UTC values")
    if not source_work[label_available_column].ge(source_work[decision_column]).all():
        raise SupportAwareResidualError("support labels cannot resolve before the decision")
    if source_work[support_column].isna().any():
        raise SupportAwareResidualError("support source has unresolved labels")
    target["__support_row__"] = np.arange(len(target), dtype=np.int64)
    block = target[decision_column].dt.floor(f"{int(config.refit_days)}D")
    probability = np.full(len(target), np.nan, dtype=np.float32)
    fit_cutoff = np.full(len(target), np.datetime64("NaT"), dtype="datetime64[ns]")
    for block_start, block_rows in target.groupby(block, sort=True, observed=True):
        start = pd.Timestamp(block_start)
        train = source_work.loc[source_work[label_available_column] < start]
        rows = block_rows["__support_row__"].to_numpy(dtype=np.int64)
        if len(train) < int(config.min_train_rows) or train[support_column].nunique(dropna=True) < 2:
            prior = float(train[support_column].astype(float).mean()) if len(train) else 0.0
            probability[rows] = np.float32(prior)
        else:
            model = _fit_support_model(
                train.loc[:, features], train[support_column].to_numpy(bool),
                config=config, seed=int(seed) + int(start.value % 1_000_003),
            )
            probability[rows] = model.predict_proba(block_rows.loc[:, features].replace([np.inf, -np.inf], np.nan))[:, 1].astype(np.float32)
        fit_cutoff[rows] = start.tz_convert("UTC").tz_localize(None).to_datetime64()
    result = pd.DataFrame({
        "predicted_support_probability": probability,
        "support_model_fit_cutoff_ts": pd.to_datetime(fit_cutoff, utc=True),
    })
    if not np.isfinite(result["predicted_support_probability"]).all():
        raise SupportAwareResidualError("support probabilities were not completely materialised")
    if (result["support_model_fit_cutoff_ts"] > target[decision_column]).any():
        raise SupportAwareResidualError("support model was fitted after a scored decision")
    return result


__all__ = [
    "SupportAwareResidualError", "SupportPredictionConfig", "atr_residual_grade",
    "bps_residual_grade", "hybrid_economic_grade", "query_normalised_support_weights",
    "strict_prequential_support_probabilities",
]
