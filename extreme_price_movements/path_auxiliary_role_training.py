"""Leakage-safe role trainers for decomposed auxiliary path heads.

This module deliberately does not know how an auxiliary target is persisted.
It provides the common fitting contract for a *role* of a decomposed target,
for example ``P(reaches meaningful MFE)``, the conditional peak magnitude, or
the 80th percentile of conditional adverse excursion.  The important
distinction is that the role mask limits fitting and metrics, never the rows
which receive an outer-OOF prediction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from extreme_price_movements.path_auxiliary_lgbm import (
    MODEL_SCHEMA,
    configured_auxiliary_feature_universe,
    default_auxiliary_lgbm_n_jobs,
    expanding_purged_folds,
)

ROLE_TRAINER_SCHEMA = "path_auxiliary_role_training_v1_strict_oof"
FIXED_MAY_JULY_OOF_MONTHS: tuple[str, ...] = ("2026-05", "2026-06", "2026-07")
TaskKind = Literal["binary", "regression", "quantile"]


def select_auxiliary_role_features(
    X: pd.DataFrame,
    role_target: Sequence[Any],
    *,
    task_kind: TaskKind,
    timestamps: Sequence[Any],
    assets: Sequence[Any],
    sides: Sequence[Any],
    archetypes: Sequence[Any],
    role_name: str,
    sample_weight: Sequence[Any] | None = None,
    mandatory_features_by_side: Mapping[str, Sequence[str]] | None = None,
    random_state: int = 42,
    cfg: Mapping[str, Any] | None = None,
    purge_hours: float = 13.0,
) -> dict[str, Any]:
    """Run the full feature selector independently per side for one role.

    Binary roles use the classifier selector and hard 0/1 labels. Regression
    and quantile roles use the regression selector; a quantile role may reuse
    the selection of another role only when its target and conditioning mask
    are exactly identical (for example peak conditional mean and q80).
    """

    from extreme_price_movements import lgbm_pipeline

    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame")
    kind = _validate_task(str(task_kind), 0.80)
    rows = len(X)
    target = pd.to_numeric(pd.Series(role_target), errors="coerce").to_numpy(
        dtype=np.float32
    )
    if len(target) != rows or not np.isfinite(target).all():
        raise ValueError("role_target must be finite and aligned to X")
    if kind == "binary":
        _validate_binary_target(target, where=np.ones(rows, dtype=bool))
    side_values = np.asarray(sides).astype(str)
    timestamp_values = np.asarray(timestamps)
    asset_values = np.asarray(assets).astype(str)
    archetype_values = np.asarray(archetypes).astype(str)
    for name, values in {
        "timestamps": timestamp_values,
        "assets": asset_values,
        "sides": side_values,
        "archetypes": archetype_values,
    }.items():
        if len(values) != rows:
            raise ValueError(f"{name} must align to X")
    weights = (
        np.ones(rows, dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    if (
        weights.shape != target.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("sample_weight must be finite, positive, and aligned to X")
    if not bool(lgbm_pipeline.LGBM_PER_SIDE_FEATURE_SELECTION):
        raise RuntimeError(
            "auxiliary roles require independent long/short feature selection"
        )

    local_cfg = dict(cfg or {})
    local_mda = dict(local_cfg.get("mda_config", {}) or {})
    local_mda.update(
        {
            "archetype_conditioned_enabled": True,
            "archetype_univariate_prescreen_enabled": False,
            "archetype_relief_prescreen_enabled": False,
            "side_tail_across_archetypes_unweighted": True,
            "use_sample_weight": False,
            "correlation_pruning_before_prescreen": True,
            "correlation_pruning_threshold": 0.88,
            "correlation_threshold": 0.88,
            "correlation_pruning_floor_ratio": 0.50,
            "correlation_pruning_floor_count": 300,
        }
    )
    if kind != "binary":
        local_mda["objective"] = "auxiliary_regression"
    local_cfg.update(
        {
            "mda_config": local_mda,
            "archetype_univariate_prescreen_enabled": False,
            "archetype_relief_prescreen_enabled": False,
            "low_performance_period_weights_enabled": False,
            "lgbm_ae_gmm_features_enabled": False,
            "lgbm_joint_complete_case_filter_enabled": False,
        }
    )
    candidates, universe_report = configured_auxiliary_feature_universe(
        X.columns, cfg=local_cfg
    )
    if not candidates:
        raise RuntimeError("no configured auxiliary role features are available")
    matrix = X.loc[:, candidates]
    mandatory = {
        side: [
            feature
            for feature in map(str, (mandatory_features_by_side or {}).get(side, ()))
            if feature in matrix.columns
        ]
        for side in ("long", "short")
    }
    selected_by_side: dict[str, list[str]] = {}
    metrics_by_side: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        side_idx = np.flatnonzero(side_values == side)
        if len(side_idx) < 200:
            raise RuntimeError(
                f"auxiliary role selection has fewer than 200 {side} rows "
                f"for {role_name}"
            )
        local_target = target[side_idx]
        local_archetype = np.char.add(
            np.char.add(side_values[side_idx], "__"), archetype_values[side_idx]
        )
        binary = kind == "binary"
        params = {
            "objective": "binary" if binary else "huber",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 16,
            "min_child_samples": 300,
            "min_split_gain": 0.01,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "verbosity": -1,
        }
        label_context = {
            "feature_selection_archetype": local_archetype,
            "archetype": local_archetype,
            "side_name": side_values[side_idx],
            "side": side_values[side_idx],
            "y_ret": local_target,
            "side_mda_sample_weight": np.ones(len(side_idx), dtype=np.float32),
        }
        if binary:
            label_context["y_bin"] = local_target
        previous_short_history_fallback = (
            lgbm_pipeline.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK
        )
        previous_aux_validation_months = (
            lgbm_pipeline.LGBM_AUX_FORWARD_VALIDATION_MONTHS
        )
        previous_aux_min_valid_rows = lgbm_pipeline.LGBM_AUX_FORWARD_MIN_VALID_ROWS
        previous_purge_hours = lgbm_pipeline.LGBM_PURGE_HOURS
        auxiliary_validation_months = min(int(previous_aux_validation_months), 1)
        # Conditional regression roles can have only about three thousand
        # side-local selector rows after the chronological April holdout.  With
        # three purged folds, 250 rejects otherwise valid folds (the canonical
        # short conditional-MFE fold has 240 rows).  Two hundred is the
        # predeclared statistical floor, matches the selector's minimum
        # side-local training support, and is not tailored to the observed 240.
        auxiliary_min_valid_rows = min(int(previous_aux_min_valid_rows), 200)
        # The frozen December-April auxiliary reference window is shorter than
        # the base model's 365-day burn-in.  Use the pipeline's chronological
        # short-history fallback for this synchronous side-local selection
        # call: it retains an expanding train-before-validation split and never
        # falls back to shuffled CV.
        lgbm_pipeline.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK = True
        # Regression selection otherwise reserves six months for validation
        # from a five-month December-April reference window, leaving the first
        # fold with no causally earlier training rows.  An April validation tail
        # leaves December-March for strictly earlier fitting and matches the
        # approved largest-single-fold feature-selection convention.
        lgbm_pipeline.LGBM_AUX_FORWARD_VALIDATION_MONTHS = auxiliary_validation_months
        lgbm_pipeline.LGBM_AUX_FORWARD_MIN_VALID_ROWS = auxiliary_min_valid_rows
        lgbm_pipeline.LGBM_PURGE_HOURS = float(purge_hours)
        try:
            result = lgbm_pipeline.train_lgbm_stability_candidate(
                matrix.iloc[side_idx].reset_index(drop=True),
                local_target,
                sample_weight=weights[side_idx],
                random_state=int(random_state) + (1009 if side == "long" else 2017),
                mode="classifier" if binary else "regressor",
                timestamps=timestamp_values[side_idx],
                assets=asset_values[side_idx],
                returns=local_target,
                hard_labels=local_target if binary else None,
                hpo_objective_mode="train_base" if binary else "auxiliary_regression",
                preset_best_params=params,
                preset_source=f"{MODEL_SCHEMA}:{role_name}:{side}:selection_only",
                cfg=local_cfg,
                label_context=label_context,
            )
        finally:
            lgbm_pipeline.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK = (
                previous_short_history_fallback
            )
            lgbm_pipeline.LGBM_AUX_FORWARD_VALIDATION_MONTHS = (
                previous_aux_validation_months
            )
            lgbm_pipeline.LGBM_AUX_FORWARD_MIN_VALID_ROWS = previous_aux_min_valid_rows
            lgbm_pipeline.LGBM_PURGE_HOURS = previous_purge_hours
        if not result:
            raise RuntimeError(f"feature selection failed for {role_name}/{side}")
        side_metrics = dict(result.get("metrics") or {})
        side_metrics["auxiliary_selection_cv_contract"] = {
            "mode": "forward_burnin_with_chronological_short_history_fallback",
            "base_burn_in_days": float(lgbm_pipeline.LGBM_BASE_FORWARD_BURN_IN_DAYS),
            "short_history_fallback_fraction": float(
                lgbm_pipeline.LGBM_FORWARD_SHORT_HISTORY_FALLBACK_FRAC
            ),
            "auxiliary_validation_months": auxiliary_validation_months,
            "auxiliary_min_validation_rows": auxiliary_min_valid_rows,
            "purge_hours": float(purge_hours),
            "train_before_validation_only": True,
            "shuffled_fallback_forbidden": True,
        }
        selected = [
            str(feature)
            for feature in result.get("selected_feature_names", ())
            if str(feature) in matrix.columns
        ]
        if not selected:
            selected = list(
                map(
                    str,
                    dict(
                        side_metrics.get("per_side_feature_selection_selected_features")
                        or {}
                    ).get(side, ()),
                )
            )
        if not selected:
            raise RuntimeError(
                f"feature selection returned no features for {role_name}/{side}"
            )
        selected_by_side[side] = list(dict.fromkeys([*selected, *mandatory[side]]))
        metrics_by_side[side] = side_metrics
    return {
        "role_name": str(role_name),
        "task_kind": kind,
        "selected_features_by_side": selected_by_side,
        "selected_features": list(
            dict.fromkeys(
                feature
                for side in ("long", "short")
                for feature in selected_by_side[side]
            )
        ),
        "selection_metrics": {
            "contract": "strict_independent_side_role_selector_runs_v1",
            "by_side": metrics_by_side,
        },
        "feature_universe_report": universe_report,
        "mandatory_features_by_side": mandatory,
        "sample_weight_contract": (
            "training loss only; selector MDA and validation metrics unweighted"
        ),
        "prescreen_contract": (
            "strict side-local univariate plus Relief plus MDA; "
            "binary roles use classifier semantics"
        ),
        "correlation_pruning_threshold": 0.88,
    }


@dataclass(frozen=True)
class FixedCalendarOOFFold:
    """One full-calendar outer validation fold before role filtering."""

    base_train_idx: np.ndarray
    valid_idx: np.ndarray
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    fold_month: str


@dataclass(frozen=True)
class _PurgedRoleFold:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def _require_utc_cutoff(value: Any) -> pd.Timestamp:
    """Return an explicit, timezone-aware UTC reference cutoff."""

    if value is None:
        raise ValueError("selection_hpo_reference_end must be declared explicitly")
    cutoff = pd.Timestamp(value)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError(
            "selection_hpo_reference_end must be an explicit timezone-aware UTC timestamp"
        )
    return cutoff.tz_convert("UTC")


def _utc_series(values: Sequence[Any], *, name: str) -> pd.Series:
    result = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if len(result) != len(values):  # pragma: no cover - defensive pandas invariant.
        raise ValueError(f"{name} could not be aligned to the role target")
    return result


def _timestamp_summary(values: pd.Series) -> dict[str, Any]:
    finite = values.dropna()
    return {
        "rows": int(len(values)),
        "valid_rows": int(len(finite)),
        "min_utc": finite.min().isoformat() if not finite.empty else None,
        "max_utc": finite.max().isoformat() if not finite.empty else None,
    }


def _model_sha256(model: Any) -> str:
    booster = getattr(model, "booster_", None)
    if booster is not None and hasattr(booster, "model_to_string"):
        payload = booster.model_to_string()
    elif hasattr(model, "model_to_string"):
        payload = model.model_to_string()
    elif hasattr(model, "get_params"):
        payload = repr(sorted(model.get_params().items()))
    else:  # pragma: no cover - LightGBM always exposes a booster.
        payload = repr(model)
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def _as_mask(values: Sequence[Any], *, name: str, rows: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != rows:
        raise ValueError(f"{name} must be a one-dimensional vector aligned to X")
    if raw.dtype == bool:
        return raw.copy()
    if pd.isna(raw).any():
        raise ValueError(f"{name} must not contain null values")
    numeric = pd.to_numeric(pd.Series(raw), errors="raise").to_numpy()
    if not np.isin(numeric, (0, 1)).all():
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return numeric.astype(bool)


def _validate_task(kind: str, quantile_alpha: float) -> TaskKind:
    if kind not in {"binary", "regression", "quantile"}:
        raise ValueError("task_kind must be one of binary, regression, or quantile")
    if kind == "quantile" and not np.isclose(float(quantile_alpha), 0.80):
        raise ValueError("auxiliary quantile roles are fixed at alpha=0.8")
    return kind  # type: ignore[return-value]


def _validate_binary_target(values: np.ndarray, *, where: np.ndarray) -> None:
    checked = values[where]
    if not len(checked):
        raise ValueError("binary role has no finite training target values")
    if not np.isin(checked, (0.0, 1.0)).all():
        raise ValueError("binary role target must be exactly 0/1 on role-train rows")


def _fixed_calendar_oof_folds(
    decision: pd.Series,
    resolved: pd.Series,
    *,
    reference_end: pd.Timestamp,
    oof_months: Sequence[str],
) -> list[FixedCalendarOOFFold]:
    """Build full-row monthly OOF folds, intentionally without role filtering.

    ``expanding_monthly_oos_folds`` is not reused here because it removes rows
    without a resolved label from validation.  Role heads must instead emit a
    prediction for every candidate decision in May--July; only diagnostics are
    conditional on the role mask.
    """

    if not oof_months:
        raise ValueError("at least one fixed outer-OOF calendar month is required")
    folds: list[FixedCalendarOOFFold] = []
    seen: set[str] = set()
    for month_text in oof_months:
        month = str(month_text)
        if month in seen:
            raise ValueError(f"duplicate fixed outer-OOF month: {month}")
        seen.add(month)
        try:
            period = pd.Period(month, freq="M")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid fixed outer-OOF month: {month}") from exc
        month_start = pd.Timestamp(period.start_time, tz="UTC")
        month_stop = month_start + pd.offsets.MonthBegin(1)
        if month_start < reference_end:
            raise ValueError(
                f"fixed outer-OOF month {month} starts before reference cutoff "
                f"{reference_end.isoformat()}"
            )
        valid_mask = decision.ge(month_start) & decision.lt(month_stop)
        valid_idx = np.flatnonzero(valid_mask.to_numpy())
        if not len(valid_idx):
            raise ValueError(f"fixed outer-OOF month {month} has no decision rows")
        base_train_mask = decision.lt(month_start) & resolved.lt(month_start)
        base_train_idx = np.flatnonzero(base_train_mask.to_numpy())
        if not len(base_train_idx):
            raise ValueError(
                f"fixed outer-OOF month {month} has no causally resolved training rows"
            )
        train_decision = decision.iloc[base_train_idx].dropna()
        folds.append(
            FixedCalendarOOFFold(
                base_train_idx=base_train_idx.astype(np.int32),
                valid_idx=valid_idx.astype(np.int32),
                train_start=train_decision.min() if not train_decision.empty else None,
                train_end=train_decision.max() if not train_decision.empty else None,
                valid_start=month_start,
                valid_end=decision.iloc[valid_idx].max(),
                fold_month=month,
            )
        )
    return folds


def _role_reference_mask(
    decision: pd.Series,
    resolved: pd.Series,
    target: np.ndarray,
    role_train_mask: np.ndarray,
    cutoff: pd.Timestamp,
) -> np.ndarray:
    return (
        decision.lt(cutoff).to_numpy()
        & resolved.lt(cutoff).to_numpy()
        & np.isfinite(target)
        & role_train_mask
    )


def _build_purged_role_folds(
    timestamps: np.ndarray,
    label_resolved_at: np.ndarray,
    *,
    purge_hours: float,
) -> list[_PurgedRoleFold]:
    """Use the common chronological splitter and prove resolution safety."""

    rows = len(timestamps)
    min_train_rows = max(8, min(500, max(8, rows // 6)))
    min_valid_rows = max(4, min(100, max(4, rows // 20)))
    raw_folds = expanding_purged_folds(
        timestamps,
        n_splits=3,
        purge_hours=float(purge_hours),
        min_train_rows=min_train_rows,
        min_valid_rows=min_valid_rows,
    )
    resolved = pd.to_datetime(pd.Series(label_resolved_at), utc=True, errors="coerce")
    safe_folds: list[_PurgedRoleFold] = []
    for fold in raw_folds:
        train_idx = fold.train_idx[
            resolved.iloc[fold.train_idx].lt(fold.valid_start).to_numpy()
        ]
        if len(train_idx) < min_train_rows or len(fold.valid_idx) < min_valid_rows:
            continue
        train_resolution = resolved.iloc[train_idx]
        if train_resolution.empty or train_resolution.isna().any():
            raise AssertionError("purged HPO fold contains unresolved training labels")
        if not bool(train_resolution.max() < fold.valid_start):
            raise AssertionError(
                "purged HPO fold violates max(train label resolution) < valid start"
            )
        safe_folds.append(
            _PurgedRoleFold(
                train_idx=train_idx.astype(np.int32),
                valid_idx=fold.valid_idx.astype(np.int32),
                valid_start=fold.valid_start,
                valid_end=fold.valid_end,
            )
        )
    if not safe_folds:
        raise ValueError(
            "no role-specific chronological HPO folds survive the label-resolution purge"
        )
    return safe_folds


def _binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    probability = np.clip(np.asarray(prediction, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    observed = np.asarray(target, dtype=np.float64)
    support = len(observed)
    if not support:
        return {
            "metric_support": 0,
            "binary_logloss": float("nan"),
            "brier": float("nan"),
            "roc_auc": float("nan"),
            "ece_10bin": float("nan"),
            "calibration": {"bins": [], "available": False},
        }
    logloss = float(
        -np.mean(
            observed * np.log(probability) + (1.0 - observed) * np.log1p(-probability)
        )
    )
    brier = float(np.mean((probability - observed) ** 2))
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 0.9, 10), np.linspace(0.1, 1.0, 10)):
        if upper == 1.0:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        count = int(mask.sum())
        if not count:
            continue
        mean_prediction = float(np.mean(probability[mask]))
        observed_rate = float(np.mean(observed[mask]))
        ece += count / support * abs(mean_prediction - observed_rate)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "rows": count,
                "mean_prediction": mean_prediction,
                "observed_rate": observed_rate,
            }
        )
    if len(np.unique(observed)) < 2:
        auc = float("nan")
    else:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(observed, probability))
    return {
        "metric_support": int(support),
        "binary_logloss": logloss,
        "brier": brier,
        "roc_auc": auc,
        "ece_10bin": float(ece),
        "positive_rate": float(np.mean(observed)),
        "mean_prediction": float(np.mean(probability)),
        "calibration": {"bins": bins, "available": True},
    }


def _rank_ic(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.nanstd(target) <= 1e-12 or np.nanstd(prediction) <= 1e-12:
        return 0.0
    left = pd.Series(target).rank(method="average").to_numpy(dtype=np.float64)
    right = pd.Series(prediction).rank(method="average").to_numpy(dtype=np.float64)
    value = np.corrcoef(left, right)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _continuous_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    task_kind: TaskKind,
    quantile_alpha: float,
) -> dict[str, Any]:
    observed = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    support = len(observed)
    if not support:
        return {
            "metric_support": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "spearman_ic": float("nan"),
            "pinball_loss_alpha_0_8": float("nan"),
        }
    residual = observed - estimate
    metrics: dict[str, Any] = {
        "metric_support": int(support),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman_ic": _rank_ic(observed, estimate),
        "target_mean": float(np.mean(observed)),
        "prediction_mean": float(np.mean(estimate)),
    }
    if task_kind == "quantile":
        alpha = float(quantile_alpha)
        metrics["pinball_loss_alpha_0_8"] = float(
            np.mean(np.maximum(alpha * residual, (alpha - 1.0) * residual))
        )
        metrics["empirical_coverage_alpha_0_8"] = float(np.mean(observed <= estimate))
    return metrics


def _role_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    role_train_mask: np.ndarray,
    *,
    task_kind: TaskKind,
    quantile_alpha: float,
) -> dict[str, Any]:
    eligible = role_train_mask & np.isfinite(target) & np.isfinite(prediction)
    if task_kind == "binary":
        return _binary_metrics(target[eligible], prediction[eligible])
    return _continuous_metrics(
        target[eligible],
        prediction[eligible],
        task_kind=task_kind,
        quantile_alpha=quantile_alpha,
    )


def _hpo_score(metrics: Mapping[str, Any], *, task_kind: TaskKind) -> float:
    if not int(metrics.get("metric_support", 0)):
        return -np.inf
    if task_kind == "binary":
        auc = float(metrics.get("roc_auc", np.nan))
        auc_component = 0.0 if not np.isfinite(auc) else 0.15 * (auc - 0.5)
        return float(
            -float(metrics["binary_logloss"])
            - 0.25 * float(metrics["brier"])
            - 0.10 * float(metrics["ece_10bin"])
            + auc_component
        )
    if task_kind == "quantile":
        return -float(metrics["pinball_loss_alpha_0_8"])
    return float(
        -float(metrics["mae"])
        - 0.20 * float(metrics["rmse"])
        + 0.20 * float(metrics["spearman_ic"])
    )


def _task_objective(task_kind: TaskKind) -> str:
    if task_kind == "binary":
        return "binary"
    if task_kind == "quantile":
        return "quantile"
    return "regression"


def _bounded_n_jobs(value: int | None) -> int:
    available = default_auxiliary_lgbm_n_jobs()
    if value is None:
        return available
    requested = int(value)
    if requested < 1:
        raise ValueError("n_jobs must be positive when provided")
    return min(available, requested)


def _sanitize_params(
    params: Mapping[str, Any],
    *,
    task_kind: TaskKind,
    quantile_alpha: float,
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    result = dict(params)
    result.update(
        {
            "objective": _task_objective(task_kind),
            "random_state": int(random_state),
            "n_jobs": int(n_jobs),
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
    )
    if task_kind == "quantile":
        result["alpha"] = float(quantile_alpha)
    else:
        result.pop("alpha", None)
    return result


def _suggest_params(
    trial: Any,
    *,
    task_kind: TaskKind,
    quantile_alpha: float,
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 1200,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "num_leaves": trial.suggest_categorical("num_leaves", [8, 16, 24, 32, 48]),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 1200, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 0.05, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 40.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.0),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
    }
    return _sanitize_params(
        params,
        task_kind=task_kind,
        quantile_alpha=quantile_alpha,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _make_model(task_kind: TaskKind, params: Mapping[str, Any]) -> Any:
    import lightgbm as lgb

    if task_kind == "binary":
        return lgb.LGBMClassifier(**dict(params))
    return lgb.LGBMRegressor(**dict(params))


def _predict_role_model(
    model: Any, X: pd.DataFrame, *, task_kind: TaskKind
) -> np.ndarray:
    if task_kind != "binary":
        return np.asarray(model.predict(X), dtype=np.float32)
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float32)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.shape[1] == 2:
        return probabilities[:, 1]
    classes = np.asarray(getattr(model, "classes_", [0]))
    return np.full(len(X), 1.0 if int(classes[0]) == 1 else 0.0, dtype=np.float32)


def _fit_with_inner_validation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    weight_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    task_kind: TaskKind,
    params: Mapping[str, Any],
) -> tuple[Any, int]:
    """Fit an HPO candidate; only inner-reference validation reaches LightGBM."""

    import lightgbm as lgb

    model = _make_model(task_kind, params)
    model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )
    return model, int(model.best_iteration_ or params.get("n_estimators", 1))


def fit_auxiliary_role_model(
    X: pd.DataFrame,
    role_target: Sequence[Any],
    *,
    role_train_mask: Sequence[Any],
    task_kind: TaskKind,
    selected_features: Sequence[str],
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    selection_hpo_reference_end: Any,
    sample_weight: Sequence[Any] | None = None,
    n_trials: int = 40,
    hpo_patience: int = 12,
    random_state: int = 42,
    purge_hours: float = 13.0,
    preset_params: Mapping[str, Any] | None = None,
    hpo_rows: int = 45_000,
    oof_months: Sequence[str] = FIXED_MAY_JULY_OOF_MONTHS,
    quantile_alpha: float = 0.80,
    n_jobs: int | None = None,
    role_name: str = "auxiliary_role",
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit one role with strict reference HPO and full-row May--July OOF.

    Feature selection is deliberately outside this function.  The caller must
    supply a frozen, role-specific feature list generated solely from the April
    reference population.  ``role_train_mask`` can make a target conditional;
    it is applied to every fit and metric but never to the outer prediction
    population.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    rows = len(X)
    if not rows:
        raise ValueError("X must contain at least one row")
    kind = _validate_task(str(task_kind), float(quantile_alpha))
    target = pd.to_numeric(pd.Series(role_target), errors="coerce").to_numpy(
        dtype=np.float32
    )
    if target.ndim != 1 or len(target) != rows:
        raise ValueError("role_target must be a one-dimensional vector aligned to X")
    role_mask = _as_mask(role_train_mask, name="role_train_mask", rows=rows)
    decision = _utc_series(timestamps, name="timestamps")
    resolved = _utc_series(label_resolved_at, name="label_resolved_at")
    if len(decision) != rows or len(resolved) != rows:
        raise ValueError("timestamps and label_resolved_at must align to X")
    cutoff = _require_utc_cutoff(selection_hpo_reference_end)
    if not np.isfinite(float(purge_hours)) or float(purge_hours) <= 0.0:
        raise ValueError("purge_hours must be finite and positive")
    features = list(dict.fromkeys(map(str, selected_features)))
    if not features:
        raise ValueError("selected_features must be non-empty")
    missing_features = [feature for feature in features if feature not in X.columns]
    if missing_features:
        raise ValueError(f"selected role features missing: {missing_features[:20]}")
    matrix = X.loc[:, features].astype(np.float32, copy=False)
    weights = (
        np.ones(rows, dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    if (
        weights.ndim != 1
        or weights.shape != target.shape
        or not np.isfinite(weights).all()
    ):
        raise ValueError("sample_weight must be finite and aligned to role_target")
    if (weights <= 0.0).any():
        raise ValueError("sample_weight must be strictly positive")
    reference_mask = _role_reference_mask(decision, resolved, target, role_mask, cutoff)
    if not bool(reference_mask.any()):
        raise ValueError(
            "no role rows satisfy decision/resolution-before-cutoff training contract"
        )
    if kind == "binary":
        _validate_binary_target(target, where=reference_mask)
    role_rows = np.flatnonzero(reference_mask)
    bounded_jobs = _bounded_n_jobs(n_jobs)

    # The shared helper gives a deterministic, target-neutral beginning/middle/end
    # sample.  It is imported lazily because role-only use should not pay for the
    # broader runner's selection dependencies until HPO is requested.
    hpo_local_idx = np.arange(len(role_rows), dtype=np.int32)
    hpo_reused = preset_params is not None
    hpo_best_value: float | None = None
    hpo_trial_count = 0
    hpo_fold_provenance: list[dict[str, Any]] = []
    if hpo_reused:
        if not preset_params:
            raise ValueError("preset_params must be non-empty when supplied")
        best_params = _sanitize_params(
            preset_params,
            task_kind=kind,
            quantile_alpha=quantile_alpha,
            random_state=random_state,
            n_jobs=bounded_jobs,
        )
    else:
        if not 1 <= int(n_trials) <= 40:
            raise ValueError("n_trials must be between 1 and the production cap of 40")
        if int(hpo_patience) < 1:
            raise ValueError("hpo_patience must be positive")
        from extreme_price_movements.path_auxiliary_lgbm import (
            auxiliary_hpo_sample_indices,
        )

        hpo_local_idx = auxiliary_hpo_sample_indices(
            decision.iloc[role_rows].to_numpy(),
            max_rows=max(1, int(hpo_rows)),
            random_state=int(random_state),
        ).astype(np.int32)
        hpo_rows_global = role_rows[hpo_local_idx]
        hpo_timestamps = decision.iloc[hpo_rows_global].to_numpy()
        hpo_resolved = resolved.iloc[hpo_rows_global].to_numpy()
        inner_folds = _build_purged_role_folds(
            hpo_timestamps, hpo_resolved, purge_hours=float(purge_hours)
        )
        for fold_i, fold in enumerate(inner_folds):
            train_resolution = pd.to_datetime(
                pd.Series(hpo_resolved[fold.train_idx]), utc=True, errors="coerce"
            )
            # This is redundant by design: it is a plainly auditable assertion
            # immediately beside the HPO fold provenance.
            assert bool(train_resolution.max() < fold.valid_start), (
                "HPO role fold must resolve before its validation decision window"
            )
            hpo_fold_provenance.append(
                {
                    "fold": int(fold_i),
                    "training_rows": int(len(fold.train_idx)),
                    "validation_rows": int(len(fold.valid_idx)),
                    "valid_start": fold.valid_start.isoformat(),
                    "valid_end": fold.valid_end.isoformat(),
                    "training_label_resolved_max": train_resolution.max().isoformat(),
                    "resolution_before_valid_start_assertion": True,
                }
            )
        import optuna

        trial_iterations: dict[int, list[int]] = {}
        hpo_X = matrix.iloc[hpo_rows_global].reset_index(drop=True)
        hpo_y = target[hpo_rows_global]
        hpo_weight = weights[hpo_rows_global]

        def objective(trial: Any) -> float:
            if progress_callback is not None:
                progress_callback(
                    "hpo_trial_start",
                    {"role": role_name, "trial": int(trial.number)},
                )
            params = _suggest_params(
                trial,
                task_kind=kind,
                quantile_alpha=quantile_alpha,
                random_state=int(random_state),
                n_jobs=bounded_jobs,
            )
            scores: list[float] = []
            iterations: list[int] = []
            for fold_index, fold in enumerate(inner_folds):
                if progress_callback is not None:
                    progress_callback(
                        "hpo_fold_start",
                        {
                            "role": role_name,
                            "trial": int(trial.number),
                            "fold": int(fold_index),
                        },
                    )
                model, best_iteration = _fit_with_inner_validation(
                    hpo_X.iloc[fold.train_idx],
                    hpo_y[fold.train_idx],
                    hpo_weight[fold.train_idx],
                    hpo_X.iloc[fold.valid_idx],
                    hpo_y[fold.valid_idx],
                    task_kind=kind,
                    params=params,
                )
                prediction = _predict_role_model(
                    model, hpo_X.iloc[fold.valid_idx], task_kind=kind
                )
                metrics = _role_metrics(
                    hpo_y[fold.valid_idx],
                    prediction,
                    np.ones(len(fold.valid_idx), dtype=bool),
                    task_kind=kind,
                    quantile_alpha=quantile_alpha,
                )
                scores.append(_hpo_score(metrics, task_kind=kind))
                iterations.append(best_iteration)
                if progress_callback is not None:
                    progress_callback(
                        "hpo_fold_complete",
                        {
                            "role": role_name,
                            "trial": int(trial.number),
                            "fold": int(fold_index),
                        },
                    )
            trial_iterations[int(trial.number)] = iterations
            return float(np.mean(scores) - 0.25 * np.std(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(random_state)),
            pruner=optuna.pruners.NopPruner(),
        )
        best_seen = -np.inf
        stale_trials = 0

        def stop_after_stale_trials(study: Any, trial: Any) -> None:
            nonlocal best_seen, stale_trials
            value = trial.value
            if value is not None and np.isfinite(value) and float(value) > best_seen:
                best_seen = float(value)
                stale_trials = 0
            else:
                stale_trials += 1
            if len(study.trials) >= int(hpo_patience) and stale_trials >= int(
                hpo_patience
            ):
                study.stop()

        study.optimize(
            objective,
            n_trials=int(n_trials),
            n_jobs=1,
            show_progress_bar=False,
            callbacks=[stop_after_stale_trials],
        )
        best_params = _sanitize_params(
            study.best_params,
            task_kind=kind,
            quantile_alpha=quantile_alpha,
            random_state=random_state,
            n_jobs=bounded_jobs,
        )
        best_params["n_estimators"] = max(
            25,
            int(np.median(trial_iterations[int(study.best_trial.number)])),
        )
        best_params["subsample_freq"] = 1
        hpo_best_value = float(study.best_value)
        hpo_trial_count = int(len(study.trials))

    outer_folds = _fixed_calendar_oof_folds(
        decision,
        resolved,
        reference_end=cutoff,
        oof_months=oof_months,
    )
    oof_predictions = np.full(rows, np.nan, dtype=np.float32)
    oof_fold_ids = np.full(rows, -1, dtype=np.int16)
    oof_models: list[Any] = []
    fold_provenance: list[dict[str, Any]] = []
    for fold_i, fold in enumerate(outer_folds):
        if progress_callback is not None:
            progress_callback(
                "oof_fold_start",
                {
                    "role": role_name,
                    "fold": int(fold_i),
                    "fold_month": fold.fold_month,
                },
            )
        train_idx = fold.base_train_idx[
            role_mask[fold.base_train_idx] & np.isfinite(target[fold.base_train_idx])
        ]
        if not len(train_idx):
            raise ValueError(
                f"outer OOF fold {fold.fold_month} has no role-train rows after masking"
            )
        train_resolution = resolved.iloc[train_idx]
        if train_resolution.isna().any() or not bool(
            train_resolution.max() < fold.valid_start
        ):
            raise AssertionError(
                "outer OOF fold violates max(train label resolution) < valid start"
            )
        if kind == "binary":
            _validate_binary_target(target, where=np.isin(np.arange(rows), train_idx))
        model = _make_model(kind, best_params)
        # Deliberately no eval_set here.  Outer OOF labels must not select an
        # iteration, calibrator, threshold, or any other model state.
        model.fit(
            matrix.iloc[train_idx],
            target[train_idx],
            sample_weight=weights[train_idx],
        )
        prediction = _predict_role_model(
            model, matrix.iloc[fold.valid_idx], task_kind=kind
        )
        oof_predictions[fold.valid_idx] = prediction
        oof_fold_ids[fold.valid_idx] = int(fold_i)
        conditional_mask = role_mask[fold.valid_idx]
        metrics = _role_metrics(
            target[fold.valid_idx],
            prediction,
            conditional_mask,
            task_kind=kind,
            quantile_alpha=quantile_alpha,
        )
        fold_provenance.append(
            {
                "fold": int(fold_i),
                "fold_month": fold.fold_month,
                "train_start": fold.train_start.isoformat()
                if fold.train_start
                else None,
                "train_end": fold.train_end.isoformat() if fold.train_end else None,
                "valid_start": fold.valid_start.isoformat(),
                "valid_end": fold.valid_end.isoformat(),
                "training_rows": int(len(train_idx)),
                "validation_rows": int(len(fold.valid_idx)),
                "predicted_validation_rows": int(len(fold.valid_idx)),
                "conditional_validation_rows": int(
                    (conditional_mask & np.isfinite(target[fold.valid_idx])).sum()
                ),
                "training_label_resolved_max": train_resolution.max().isoformat(),
                "resolution_before_valid_start_assertion": True,
                "prediction_contract": (
                    "all decision rows in fixed calendar validation month predicted; "
                    "role_train_mask affects only fitting and conditional metrics"
                ),
                "outer_fit_contract": (
                    "fit uses only causally resolved role-train rows; no outer "
                    "validation labels are passed to LightGBM"
                ),
                "model_sha256": _model_sha256(model),
                "conditional_metrics": metrics,
            }
        )
        oof_models.append(model)
        if progress_callback is not None:
            progress_callback(
                "oof_fold_complete",
                {
                    "role": role_name,
                    "fold": int(fold_i),
                    "fold_month": fold.fold_month,
                },
            )

    oof_prediction_mask = np.isfinite(oof_predictions)
    oof_metrics = _role_metrics(
        target,
        oof_predictions,
        role_mask & oof_prediction_mask,
        task_kind=kind,
        quantile_alpha=quantile_alpha,
    )
    final_mask = (
        role_mask
        & np.isfinite(target)
        & decision.notna().to_numpy()
        & resolved.notna().to_numpy()
    )
    if not bool(final_mask.any()):
        raise ValueError(
            "no resolved role rows are available for the separate final refit"
        )
    if kind == "binary":
        _validate_binary_target(target, where=final_mask)
    final_model = _make_model(kind, best_params)
    final_idx = np.flatnonzero(final_mask)
    if progress_callback is not None:
        progress_callback(
            "final_model_start",
            {"role": role_name, "rows": int(len(final_idx))},
        )
    final_model.fit(
        matrix.iloc[final_idx], target[final_idx], sample_weight=weights[final_idx]
    )
    if progress_callback is not None:
        progress_callback(
            "final_model_complete",
            {"role": role_name, "rows": int(len(final_idx))},
        )
    reference_contract = {
        "selection_hpo_reference_end": cutoff.isoformat(),
        "row_rule": (
            "decision_timestamp < selection_hpo_reference_end AND "
            "label_resolved_at < selection_hpo_reference_end AND role_train_mask"
        ),
        "role_reference_rows": int(len(role_rows)),
        "decision_bounds": _timestamp_summary(decision.iloc[role_rows]),
        "label_resolved_bounds": _timestamp_summary(resolved.iloc[role_rows]),
    }
    return {
        "schema": ROLE_TRAINER_SCHEMA,
        "role_name": str(role_name),
        "task_kind": kind,
        "quantile_alpha": float(quantile_alpha) if kind == "quantile" else None,
        "selected_features": features,
        "best_params": best_params,
        "lgbm_n_jobs": int(bounded_jobs),
        "hpo": {
            "reused_preset_params": bool(hpo_reused),
            "trial_count": int(hpo_trial_count),
            "maximum_trials": 40,
            "stale_trial_patience": int(hpo_patience),
            "best_value": hpo_best_value,
            "reference_rows": int(len(role_rows)),
            "hpo_rows": int(len(hpo_local_idx)),
            "purged_fold_provenance": hpo_fold_provenance,
            "contract": (
                "role-specific HPO uses only role-train rows with decision and "
                "label resolution strictly before the explicit reference cutoff"
            ),
        },
        "reference_split_contract": reference_contract,
        "oof_predictions": oof_predictions,
        "oof_prediction_mask": oof_prediction_mask,
        "oof_fold_ids": oof_fold_ids,
        "oof_models": oof_models,
        "fold_provenance": fold_provenance,
        "oof_metrics": oof_metrics,
        "oof_contract": (
            "fixed expanding May/June/July calendar OOF; full validation decision "
            "rows receive predictions, while only role-train-mask rows contribute "
            "conditional metrics"
        ),
        "models": {"oof": oof_models, "final": final_model},
        "model": final_model,
        "final_model": final_model,
        "final_inference_model": final_model,
        "final_refit_contract": {
            "rows": int(len(final_idx)),
            "row_rule": "all resolved role-train rows; separate from and excluded from OOF metrics",
            "decision_bounds": _timestamp_summary(decision.iloc[final_idx]),
            "label_resolved_bounds": _timestamp_summary(resolved.iloc[final_idx]),
            "model_sha256": _model_sha256(final_model),
        },
        "sample_weight_contract": "sample weights apply to fitting loss only; HPO and OOF metrics are unweighted",
    }


# A short alias keeps callers from encoding the target family in their API.
train_auxiliary_role_model = fit_auxiliary_role_model
