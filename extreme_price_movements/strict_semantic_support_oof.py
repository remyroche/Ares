"""Strict chronological OOF heads for the semantic supportive sidecar.

The semantic sidecar contains future-resolved labels only.  This module joins
those labels to a decision-time feature frame and emits predictions from
models fit on earlier, resolved folds.  It deliberately emits no raw label
columns and never treats a censored interval as an observed event.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.strict_oof_semantic_support_audit import semantic_head_specs


SCHEMA = "strict_semantic_support_oof_v1"
FIXED_CAPACITY = {
    "n_estimators": 250,
    "learning_rate": 0.035,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 5.0,
    "random_state": 20260801,
    "n_jobs": 1,
    "verbosity": -1,
}


class StrictSemanticOOFError(ValueError):
    """The semantic OOF contract cannot be proven for the requested input."""


@dataclass(frozen=True)
class SemanticOOFResult:
    predictions: pd.DataFrame
    manifest: dict[str, Any]


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise StrictSemanticOOFError(f"missing required timestamp column: {column}")
    result = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if result.isna().any():
        raise StrictSemanticOOFError(f"invalid or missing UTC timestamps in {column}")
    return result


def _bool(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin((0.0, 1.0)).all():
        raise StrictSemanticOOFError(f"{name} must contain only 0/1 values")
    return numeric.to_numpy(dtype=bool)


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *, kind: str) -> np.ndarray:
    if len(train_y) == 0:
        raise StrictSemanticOOFError("cannot fit a semantic head with no resolved training labels")
    if kind == "binary":
        y = (np.asarray(train_y, dtype=float) > 0.5).astype(np.int8)
        if np.all(y == y[0]):
            return np.full(len(test_x), float(y[0]), dtype=float)
        model = lgb.LGBMClassifier(objective="binary", **FIXED_CAPACITY)
        model.fit(train_x, y)
        return np.clip(model.predict_proba(test_x)[:, 1], 0.0, 1.0)
    model = lgb.LGBMRegressor(objective="regression_l2", **FIXED_CAPACITY)
    model.fit(train_x, np.asarray(train_y, dtype=float))
    return model.predict(test_x)


def _head_lineage(heads: Sequence[str], *, fold: Any, model_id: str, fit_end: pd.Timestamp) -> str:
    return json.dumps(
        {
            "schema": "semantic_head_lineage_v1",
            "heads": {
                name: {
                    "model_id": f"{model_id}:{name}",
                    "fold_id": str(fold),
                    "fit_end_ts": fit_end.isoformat(),
                    "generated_ts": "decision_ts",
                }
                for name in heads
            },
        },
        sort_keys=True,
    )


def generate_strict_semantic_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    fold_column: str,
    semantic_contract_sha256: str,
) -> SemanticOOFResult:
    """Fit every declared semantic head with expanding, resolved-label OOF."""
    feature_columns = tuple(validate_feature_columns(feature_columns))
    required = {"candidate_id", fold_column, "fold_order", "__ts__", "__decision_ts__", "__label_available_at__"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise StrictSemanticOOFError(f"semantic OOF frame missing columns: {missing}")
    if frame.candidate_id.isna().any() or frame.candidate_id.astype(str).duplicated().any():
        raise StrictSemanticOOFError("candidate_id must be non-null and unique")
    work = frame.copy()
    work["__ts__"] = _utc(work, "__ts__")
    work["__decision_ts__"] = _utc(work, "__decision_ts__")
    work["__label_available_at__"] = _utc(work, "__label_available_at__")
    if (work["__ts__"] > work["__decision_ts__"]).any():
        raise StrictSemanticOOFError("feature timestamp is after decision timestamp")
    matrix = work.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if (~np.isfinite(matrix).any(axis=1)).any():
        raise StrictSemanticOOFError("a candidate has no finite causal feature input")
    specs = semantic_head_specs(work.columns)
    if not specs:
        raise StrictSemanticOOFError("semantic sidecar declares no heads")
    fold_starts = work.groupby(fold_column, observed=True)["__ts__"].min().sort_values(kind="mergesort")
    fold_position = {fold: position for position, fold in enumerate(fold_starts.index)}
    fold_ids = [str(value) for value in fold_starts.index]
    scored_rows = work["fold_order"].astype(int).ge(1)
    scored_mask = scored_rows.to_numpy(dtype=bool)
    predictions = work.loc[scored_rows, ["candidate_id", "__ts__", "__decision_ts__", "__label_available_at__", fold_column, "fold_order"]].copy()
    predictions["prediction_fold_id"] = predictions[fold_column].map(lambda value: str(value))
    predictions["prediction_generated_ts"] = predictions["__decision_ts__"]
    predictions["is_oof"] = True
    head_model_ids: dict[str, dict[Any, str]] = {}
    head_fit_ends: dict[str, dict[Any, pd.Timestamp]] = {}
    model_ids_by_fold: dict[Any, str] = {}
    fit_ends_by_fold: dict[Any, pd.Timestamp] = {}
    for fold, test_start in fold_starts.items():
        position = fold_position[fold]
        if position == 0:
            continue
        earlier_folds = set(fold_starts.index[:position])
        train_mask = work[fold_column].isin(earlier_folds) & work["__label_available_at__"].lt(test_start)
        if not bool(train_mask.any()):
            raise StrictSemanticOOFError(f"no resolved training rows for fold {fold!r}")
        fit_end = work.loc[train_mask, "__label_available_at__"].max()
        if not bool(fit_end < test_start):
            raise StrictSemanticOOFError(f"fit end is not before test start for fold {fold!r}")
        fit_ends_by_fold[fold] = fit_end
        model_ids_by_fold[fold] = f"{SCHEMA}:fold-{fold}:features-{hashlib.sha256(','.join(feature_columns).encode()).hexdigest()[:16]}"
    for spec in specs:
        target = pd.to_numeric(work[spec.target_column], errors="coerce")
        valid = _bool(work[spec.valid_column], name=spec.valid_column) & target.notna().to_numpy()
        output = np.full(len(work), np.nan, dtype=np.float64)
        head_model_ids[spec.name] = {}
        head_fit_ends[spec.name] = {}
        for fold, test_start in fold_starts.items():
            position = fold_position[fold]
            if position == 0:
                continue
            test_fold_mask = work[fold_column].eq(fold).to_numpy()
            if not test_fold_mask.any():
                continue
            earlier_folds = set(fold_starts.index[:position])
            resolved_train = work[fold_column].isin(earlier_folds).to_numpy() & work["__label_available_at__"].lt(test_start).to_numpy()
            train_mask = resolved_train & valid
            if not train_mask.any():
                continue
            output[test_fold_mask] = _fit_predict(
                matrix[train_mask],
                target.to_numpy(dtype=float)[train_mask],
                matrix[test_fold_mask],
                kind=spec.kind,
            )
            head_model_ids[spec.name][fold] = f"{model_ids_by_fold[fold]}:{spec.name}"
            head_fit_ends[spec.name][fold] = fit_ends_by_fold[fold]
        predictions[f"semantic_oof__{spec.name}"] = output[scored_mask]
    predictions["prediction_fit_end_ts"] = predictions[fold_column].map(fit_ends_by_fold)
    predictions["prediction_model_id"] = predictions[fold_column].map(model_ids_by_fold)
    predictions["semantic_target_contract_sha256"] = semantic_contract_sha256
    predictions["semantic_head_lineage"] = predictions[fold_column].map(
        lambda fold: _head_lineage(
            [spec.name for spec in specs if fold in head_model_ids.get(spec.name, {})],
            fold=fold,
            model_id=model_ids_by_fold[fold],
            fit_end=fit_ends_by_fold[fold],
        )
    )
    manifest = {
        "schema": SCHEMA,
        "status": "STRICT_OOF_RESEARCH_DIAGNOSTIC",
        "rows": int(len(predictions)),
        "feature_count": len(feature_columns),
        "folds": fold_ids,
        "scored_folds": fold_ids[1:],
        "semantic_target_contract_sha256": semantic_contract_sha256,
        "prediction_lineage_columns": [
            "is_oof", "prediction_fit_end_ts", "prediction_generated_ts",
            "prediction_model_id", "prediction_fold_id",
        ],
        "heads": [
            {"name": spec.name, "kind": spec.kind, "target_column": spec.target_column, "valid_column": spec.valid_column}
            for spec in specs
        ],
        "oof_rule": "earlier fold labels with label_available_at < test fold start; warmup fold never scored",
        "model_capacity": FIXED_CAPACITY,
    }
    return SemanticOOFResult(predictions=predictions, manifest=manifest)
