"""Side-aware LightGBM auxiliary models for future-path timing and peak MFE."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.path_auxiliary_targets import (
    bars_before_price_stops_decreasing_regression_metrics,
    future_slope_atr_per_hour_regression_metrics,
    mae_before_meaningful_mfe_regression_metrics,
    peak_mfe_regression_metrics,
    timing_regression_metrics,
)

MODEL_SCHEMA = "path_auxiliary_lgbm_bundle_v8_expanding_oos_reusable_hpo"


def default_auxiliary_lgbm_n_jobs() -> int:
    """Bound LGBM workers by currently available RAM, with a fixed 3-worker cap."""

    cpu_count = max(1, int(os.cpu_count() or 1))
    try:
        available_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(
            os.sysconf("SC_AVPHYS_PAGES")
        )
    except (AttributeError, OSError, ValueError):
        available_bytes = 16 * 1024**3
    # Reserve four GiB for the static matrix and process overhead; budget four
    # GiB per worker.  This retains three workers on the 16-GiB production host.
    usable_gib = max(1, int(available_bytes // (1024**3)) - 4)
    return max(1, min(cpu_count, 3, usable_gib // 4 or 1))


TARGET_COLUMNS = {
    "time_to_first_meaningful_mfe": "__log1p_time_to_first_meaningful_mfe_hours_12h__",
    "peak_mfe_12h_atr": "__log1p_peak_mfe_atr_12h__",
    "mae_before_meaningful_mfe_atr": "__log1p_mae_before_meaningful_mfe_atr_12h__",
    "bars_before_price_stops_decreasing": "__log1p_bars_before_price_stops_decreasing_12h__",
    "future_slope_atr_per_hour": "__log1p_future_slope_atr_per_hour_12h__",
}

_META_HEADS: tuple[str, ...] = ("reg", "clf", "mfe", "mae", "asym")
_AE_GMM_PREFIXES: tuple[str, ...] = (
    "dae_",
    "ae_",
    "gmm_",
    "aegmm_",
    "cluster_",
)
_IDENTITY_CONTEXT_COLUMNS: tuple[str, ...] = (
    "side",
    "side_name",
    "archetype",
    "archetype_label_family",
    "policy_archetype",
    "local_side_archetype",
    "archetype_policy_key",
)
_CANDIDATE_MODEL_CONTEXT_FEATURES: tuple[str, ...] = (
    "score",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
)
_BASE_ARCHETYPE_FEATURE_PREFIX = "base_archetype_label__"


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-safe audit payload with deterministic key ordering."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_utc_reference_end(value: Any) -> pd.Timestamp:
    """Parse an explicitly timezone-aware reference cutoff as UTC."""

    if value is None:
        raise ValueError("selection_hpo_reference_end must be declared explicitly")
    cutoff = pd.Timestamp(value)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError(
            "selection_hpo_reference_end must be an explicit timezone-aware UTC timestamp"
        )
    return cutoff.tz_convert("UTC")


def _timestamp_bounds(values: Sequence[Any]) -> dict[str, Any]:
    timestamp = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    valid = timestamp.dropna()
    return {
        "rows": int(len(timestamp)),
        "valid_rows": int(len(valid)),
        "min_utc": valid.min().isoformat() if not valid.empty else None,
        "max_utc": valid.max().isoformat() if not valid.empty else None,
    }


def auxiliary_reference_split(
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    *,
    selection_hpo_reference_end: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the frozen reference and emitted-OOF populations.

    Feature selection and HPO use only rows whose decision and fully resolved
    outcome are strictly before the declared UTC cutoff. Persisted OOF
    predictions are emitted for decisions at or after it.
    """

    cutoff = _require_utc_reference_end(selection_hpo_reference_end)
    decision = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    resolved = pd.to_datetime(pd.Series(label_resolved_at), utc=True, errors="coerce")
    if len(decision) != len(resolved):
        raise ValueError("timestamps and label_resolved_at must have identical length")
    reference_mask = (decision.lt(cutoff) & resolved.lt(cutoff)).to_numpy(dtype=bool)
    oof_mask = (decision.ge(cutoff) & resolved.notna()).to_numpy(dtype=bool)
    contract: dict[str, Any] = {
        "schema": "path_auxiliary_selection_hpo_reference_split_v1",
        "selection_hpo_reference_end": cutoff.isoformat(),
        "reference_row_rule": (
            "decision_timestamp < selection_hpo_reference_end AND "
            "label_resolved_at < selection_hpo_reference_end"
        ),
        "emitted_oof_row_rule": "decision_timestamp >= selection_hpo_reference_end",
        "decision_bounds": _timestamp_bounds(decision),
        "label_resolved_bounds": _timestamp_bounds(resolved),
        "reference_decision_bounds": _timestamp_bounds(decision.loc[reference_mask]),
        "reference_label_resolved_bounds": _timestamp_bounds(
            resolved.loc[reference_mask]
        ),
        "oof_decision_bounds": _timestamp_bounds(decision.loc[oof_mask]),
        "oof_label_resolved_bounds": _timestamp_bounds(resolved.loc[oof_mask]),
        "reference_rows": int(reference_mask.sum()),
        "oof_candidate_rows": int(oof_mask.sum()),
        "boundary_rows_excluded": int(decision.eq(cutoff).sum()),
        "unresolved_rows_excluded": int(resolved.isna().sum()),
    }
    contract["contract_sha256"] = _stable_sha256(contract)
    return reference_mask, oof_mask, contract


def auxiliary_sample_weight_summary(weights: np.ndarray) -> dict[str, Any]:
    """Return a compact, JSON-safe audit of a head's bounded weights."""

    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("auxiliary sample weights must be a non-empty finite vector")
    return {
        "rows": int(len(values)),
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def _supportive_label(frame: pd.DataFrame, name: str) -> np.ndarray:
    column = f"__{name}__"
    if column not in frame.columns:
        raise ValueError(f"required supportive label is missing: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float32, copy=False
    )
    return values


def _normalize_auxiliary_weights(weights: np.ndarray) -> np.ndarray:
    """Center at one while retaining every weight inside [0.5, 2.0]."""

    values = np.clip(
        np.nan_to_num(weights, nan=0.50, posinf=2.0, neginf=0.50),
        0.50,
        2.0,
    ).astype(np.float32)
    mean = float(np.mean(values)) if len(values) else 1.0
    if not np.isfinite(mean) or mean <= 1e-12:
        return np.ones(len(values), dtype=np.float32)
    delta = values / mean - 1.0
    positive = float(np.max(delta, initial=0.0))
    negative = float(np.max(-delta, initial=0.0))
    shrink = 1.0
    if positive > 1.0:
        shrink = min(shrink, 1.0 / positive)
    if negative > 0.5:
        shrink = min(shrink, 0.5 / negative)
    return np.clip(1.0 + shrink * delta, 0.50, 2.0).astype(np.float32)


def build_auxiliary_sample_weights(
    frame: pd.DataFrame,
    target_name: str,
) -> np.ndarray:
    """Build bounded outcome-aware training weights for one auxiliary head.

    Supportive labels only shape the training loss. They never enter feature
    selection candidates or serialized inference feature contracts.
    """

    if target_name not in TARGET_COLUMNS:
        raise ValueError(f"unknown auxiliary target: {target_name}")
    rows = len(frame)
    if target_name == "peak_mfe_12h_atr":
        weights = np.full(rows, 0.75, dtype=np.float32)
        for threshold in ("1_5", "2_0", "3_0", "4_0"):
            reached = np.nan_to_num(
                _supportive_label(frame, f"mfe_ge_{threshold}atr"), nan=0.0
            )
            weights += 0.10 * np.clip(reached, 0.0, 1.0)
        persistence = 0.5 * (
            np.nan_to_num(
                _supportive_label(frame, "fraction_bars_above_50pct_peak"),
                nan=0.0,
            )
            + np.nan_to_num(
                _supportive_label(frame, "fraction_bars_above_80pct_peak"),
                nan=0.0,
            )
        )
        weights += 0.35 * np.clip(persistence, 0.0, 1.0)
    elif target_name == "time_to_first_meaningful_mfe":
        reached = np.nan_to_num(
            _supportive_label(frame, "reaches_1_5atr_within_12h"), nan=0.0
        )
        time_hours = np.expm1(
            pd.to_numeric(frame[TARGET_COLUMNS[target_name]], errors="coerce").to_numpy(
                dtype=np.float32, copy=False
            )
        )
        front_load = np.clip(1.0 - time_hours / 12.0, 0.0, 1.0)
        weights = np.where(
            reached > 0.5,
            1.0 + 0.50 * front_load,
            0.50,
        ).astype(np.float32)
    elif target_name == "mae_before_meaningful_mfe_atr":
        reached = np.nan_to_num(
            _supportive_label(frame, "reaches_1_5atr_within_12h"), nan=0.0
        )
        weights = np.where(reached > 0.5, 1.0, 0.50).astype(np.float32)
        for threshold in ("0_25", "0_50", "0_75", "1_00", "1_50"):
            adverse = np.nan_to_num(
                _supportive_label(frame, f"pre_1_5_mfe_mae_ge_{threshold}atr"),
                nan=0.0,
            )
            weights += 0.10 * (reached > 0.5) * np.clip(adverse, 0.0, 1.0)
    elif target_name == "bars_before_price_stops_decreasing":
        confirmed_bars = _supportive_label(frame, "bars_to_confirmed_adverse_trough")
        confirmed = np.isfinite(confirmed_bars).astype(np.float32)
        within_60 = np.nan_to_num(
            _supportive_label(frame, "adverse_trough_within_60m"), nan=0.0
        )
        within_120 = np.nan_to_num(
            _supportive_label(frame, "adverse_trough_within_120m"), nan=0.0
        )
        trough_first = np.nan_to_num(
            _supportive_label(frame, "trough_before_1_5atr_mfe"), nan=0.0
        )
        opportunity_first = np.nan_to_num(
            _supportive_label(frame, "reaches_1_5atr_before_trough_confirmation"),
            nan=0.0,
        )
        weights = (
            0.50
            + 0.50 * confirmed
            + 0.25 * np.clip(within_60, 0.0, 1.0)
            + 0.15 * np.clip(within_120, 0.0, 1.0)
            + 0.35 * np.clip(trough_first, 0.0, 1.0)
            - 0.25 * np.clip(opportunity_first, 0.0, 1.0)
        )
    else:
        meaningful_bars = _supportive_label(frame, "bars_to_1_5atr")
        realized = np.isfinite(meaningful_bars).astype(np.float32)
        efficiency = 0.5 * (
            np.nan_to_num(_supportive_label(frame, "path_efficiency_12h"), nan=0.0)
            + np.nan_to_num(
                _supportive_label(frame, "path_efficiency_to_first_meaningful_mfe"),
                nan=0.0,
            )
        )
        weights = 0.50 + 0.50 * realized + 0.50 * np.clip(efficiency, 0.0, 1.0)
    return _normalize_auxiliary_weights(weights)


def fit_base_archetype_label_feature_contract(
    frame: pd.DataFrame,
    *,
    source_columns: Sequence[str],
    canonical_source: str,
) -> dict[str, Any]:
    """Freeze outcome-free one-hot encodings for existing base archetype labels."""

    sources = list(dict.fromkeys(map(str, source_columns)))
    if canonical_source not in sources:
        raise ValueError(
            "canonical archetype source must be included in source_columns"
        )
    missing = [column for column in sources if column not in frame.columns]
    if missing:
        raise ValueError(f"base archetype label sources are missing: {missing}")
    features: dict[str, dict[str, str]] = {}
    canonical_features: list[str] = []
    for source in sources:
        values = frame[source].fillna("unknown").astype(str).str.strip()
        categories = sorted(
            value for value in values.unique() if value and value != "unknown"
        )
        for category in categories:
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", category).strip("_").lower()[:48]
            digest = hashlib.sha1(category.encode("utf-8")).hexdigest()[:10]
            feature = f"{_BASE_ARCHETYPE_FEATURE_PREFIX}{source}__{slug}__{digest}"
            features[feature] = {"source": source, "category": category}
            if source == canonical_source:
                canonical_features.append(feature)
    if not canonical_features:
        raise ValueError("canonical base archetype source has no usable labels")
    return {
        "schema": "base_archetype_label_onehot_v1",
        "source_columns": sources,
        "canonical_source": canonical_source,
        "features": features,
        "canonical_features": canonical_features,
        "inference_contract": (
            "pre-entry base archetype identities only; no CatBoost or realized path labels"
        ),
    }


def transform_base_archetype_label_features(
    frame: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply a frozen base-archetype one-hot contract; unseen labels map to zero."""

    if contract.get("schema") != "base_archetype_label_onehot_v1":
        raise ValueError("unsupported base archetype label feature contract")
    definitions = contract.get("features")
    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError("base archetype label feature contract has no features")
    sources = set(map(str, contract.get("source_columns", [])))
    missing = sorted(source for source in sources if source not in frame.columns)
    if missing:
        raise ValueError(
            f"base archetype label transform is missing sources: {missing}"
        )
    names = list(map(str, definitions))
    output = np.zeros((len(frame), len(names)), dtype=np.float32)
    source_values = {
        source: frame[source].fillna("unknown").astype(str).str.strip().to_numpy()
        for source in sources
    }
    for position, name in enumerate(names):
        definition = definitions[name]
        source = str(definition["source"])
        category = str(definition["category"])
        output[:, position] = (source_values[source] == category).astype(np.float32)
    return pd.DataFrame(output, index=frame.index, columns=names)


def configured_auxiliary_feature_universe(
    available_columns: Sequence[str],
    *,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Resolve the pre-selection base+meta feature universe from ``config.py``."""

    from extreme_price_movements.config import CFG
    from extreme_price_movements.training_utils import (
        get_base_feature_keys,
        get_meta_feature_keys,
    )

    config = dict(CFG)
    if cfg is not None:
        config.update(dict(cfg))
    available = [str(column) for column in available_columns]
    available_set = set(available)
    base_by_side = {
        side: list(map(str, get_base_feature_keys(side, config)))
        for side in ("long", "short")
    }
    meta_by_head = {
        head: list(map(str, get_meta_feature_keys(head, config)))
        for head in _META_HEADS
    }
    configured = list(
        dict.fromkeys(
            [
                *base_by_side["long"],
                *base_by_side["short"],
                *[feature for head in _META_HEADS for feature in meta_by_head[head]],
            ]
        )
    )
    generated_state = [
        column
        for column in available
        if column.startswith(_AE_GMM_PREFIXES)
        or column
        in {
            "reconstruction_error",
            "AE_reconstruction_error",
            "mahalanobis_distance",
            "expected_mahalanobis",
        }
    ]
    base_archetype_labels = [
        column
        for column in available
        if column.startswith(_BASE_ARCHETYPE_FEATURE_PREFIX)
    ]
    identity_context = [
        column for column in _IDENTITY_CONTEXT_COLUMNS if column in available_set
    ]
    # Side/archetype identifiers drive local selection and reporting through
    # label_context; raw string identities are not fed to LightGBM matrices.
    candidate_model_context = [
        column
        for column in _CANDIDATE_MODEL_CONTEXT_FEATURES
        if column in available_set
    ]
    requested = list(
        dict.fromkeys(
            [
                *configured,
                *generated_state,
                *candidate_model_context,
                *base_archetype_labels,
            ]
        )
    )
    selected = [
        feature
        for feature in requested
        if feature in available_set
        and not feature.startswith("__")
        and feature not in set(TARGET_COLUMNS.values())
    ]
    report = {
        "contract": "config_base_plus_meta_plus_frozen_ae_gmm_candidate_context_v2",
        "base_requested_by_side": base_by_side,
        "base_available_by_side": {
            side: [feature for feature in features if feature in available_set]
            for side, features in base_by_side.items()
        },
        "base_missing_by_side": {
            side: [feature for feature in features if feature not in available_set]
            for side, features in base_by_side.items()
        },
        "meta_requested_by_head": meta_by_head,
        "meta_available_by_head": {
            head: [feature for feature in features if feature in available_set]
            for head, features in meta_by_head.items()
        },
        "meta_missing_by_head": {
            head: [feature for feature in features if feature not in available_set]
            for head, features in meta_by_head.items()
        },
        "configured_requested_count": int(len(configured)),
        "generated_ae_gmm_available": generated_state,
        "base_archetype_label_features_available": base_archetype_labels,
        "candidate_model_context_required": list(_CANDIDATE_MODEL_CONTEXT_FEATURES),
        "candidate_model_context_available": candidate_model_context,
        "candidate_model_context_missing": [
            feature
            for feature in _CANDIDATE_MODEL_CONTEXT_FEATURES
            if feature not in available_set
        ],
        "identity_context_available": identity_context,
        "available_selected_count": int(len(selected)),
        "available_selected_features": selected,
        "configured_missing_features": [
            feature for feature in configured if feature not in available_set
        ],
        "excluded_target_columns": [
            column
            for column in available
            if column.startswith("__") or column in set(TARGET_COLUMNS.values())
        ],
    }
    return selected, report


@dataclass(frozen=True)
class ChronologicalFold:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


@dataclass(frozen=True)
class ExpandingMonthlyOOSFold:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    fold_month: str


def expanding_purged_folds(
    timestamps: Sequence[Any],
    *,
    n_splits: int = 3,
    purge_hours: float = 13.0,
    min_train_rows: int = 500,
    min_valid_rows: int = 100,
) -> list[ChronologicalFold]:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    finite = ts.notna().to_numpy()
    unique = np.sort(ts.loc[finite].unique())
    if len(unique) < n_splits + 2:
        return []
    boundaries = np.linspace(0, len(unique), n_splits + 2, dtype=int)
    purge = pd.Timedelta(hours=float(purge_hours))
    folds: list[ChronologicalFold] = []
    for fold_i in range(1, n_splits + 1):
        valid_start = pd.Timestamp(unique[boundaries[fold_i]])
        valid_stop_pos = boundaries[fold_i + 1]
        if valid_stop_pos <= boundaries[fold_i]:
            continue
        valid_end = pd.Timestamp(unique[valid_stop_pos - 1])
        train_end = valid_start - purge
        train_idx = np.flatnonzero(finite & (ts < train_end).to_numpy())
        valid_idx = np.flatnonzero(
            finite & (ts >= valid_start).to_numpy() & (ts <= valid_end).to_numpy()
        )
        if len(train_idx) < min_train_rows or len(valid_idx) < min_valid_rows:
            continue
        folds.append(
            ChronologicalFold(
                train_idx=train_idx.astype(np.int32),
                valid_idx=valid_idx.astype(np.int32),
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
    return folds


def expanding_monthly_oos_folds(
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    *,
    oos_start: Any,
) -> list[ExpandingMonthlyOOSFold]:
    """Build causal calendar-month OOS folds from a declared UTC cutoff."""

    cutoff = _require_utc_reference_end(oos_start)
    decision = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    resolved = pd.to_datetime(pd.Series(label_resolved_at), utc=True, errors="coerce")
    if len(decision) != len(resolved):
        raise ValueError("timestamps and label_resolved_at must have identical length")
    oos = decision.ge(cutoff) & resolved.notna()
    months = decision.loc[oos].dt.tz_localize(None).dt.to_period("M")
    folds: list[ExpandingMonthlyOOSFold] = []
    for month in sorted(months.dropna().unique()):
        month_start = pd.Timestamp(month.start_time, tz="UTC")
        next_month_start = month_start + pd.offsets.MonthBegin(1)
        valid_start = max(month_start, cutoff)
        valid_mask = oos & decision.ge(valid_start) & decision.lt(next_month_start)
        valid_idx = np.flatnonzero(valid_mask.to_numpy())
        if not len(valid_idx):
            continue
        train_mask = decision.lt(valid_start) & resolved.lt(valid_start)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        train_decision = decision.iloc[train_idx].dropna()
        folds.append(
            ExpandingMonthlyOOSFold(
                train_idx=train_idx.astype(np.int32),
                valid_idx=valid_idx.astype(np.int32),
                train_start=train_decision.min() if not train_decision.empty else None,
                train_end=train_decision.max() if not train_decision.empty else None,
                valid_start=valid_start,
                valid_end=decision.iloc[valid_idx].max(),
                fold_month=str(month),
            )
        )
    return folds


def _fitted_model_sha256(model: Any) -> str:
    """Fingerprint fitted LightGBM state, with a stable fallback for test doubles."""

    booster = getattr(model, "booster_", None)
    if booster is not None and hasattr(booster, "model_to_string"):
        payload = booster.model_to_string()
    elif hasattr(model, "model_to_string"):
        payload = model.model_to_string()
    elif hasattr(model, "get_params"):
        payload = json.dumps(model.get_params(), sort_keys=True, default=str)
    else:  # pragma: no cover - production LightGBM always exposes a booster.
        payload = json.dumps(
            getattr(model, "__dict__", {}), sort_keys=True, default=str
        )
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def auxiliary_hpo_objective(
    target_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, dict[str, float]]:
    if target_name == "time_to_first_meaningful_mfe":
        metrics = timing_regression_metrics(y_true, y_pred)
        accuracy = np.mean(
            [
                metrics.get(f"accuracy_meaningful_mfe_by_{h}h", 0.0)
                for h in (2, 4, 8, 12)
            ]
        )
        score = (
            -metrics.get("mae_log_time", 99.0)
            - 0.25 * metrics.get("mae_hours", 99.0) / 12.0
            + 0.50 * metrics.get("spearman_ic", 0.0)
            + 0.10 * float(accuracy)
        )
        return float(score), metrics
    if target_name == "peak_mfe_12h_atr":
        metrics = peak_mfe_regression_metrics(y_true, y_pred)
        score = (
            -metrics.get("mae_log_peak_mfe_atr", 99.0)
            - 0.50 * metrics.get("huber_loss", 99.0)
            - 0.25 * metrics.get("rmse_log_peak_mfe_atr", 99.0)
            + 0.50 * metrics.get("spearman_ic", 0.0)
        )
        return float(score), metrics
    if target_name == "mae_before_meaningful_mfe_atr":
        metrics = mae_before_meaningful_mfe_regression_metrics(y_true, y_pred)
        score = (
            -metrics.get("mae_before_meaningful_mfe_atr_log_mae", 99.0)
            - 0.20
            * metrics.get("mae_before_meaningful_mfe_atr_natural_mae", 99.0)
            / 10.0
            - 0.25 * metrics.get("mae_before_meaningful_mfe_atr_natural_huber", 99.0)
            + 0.50 * metrics.get("mae_before_meaningful_mfe_atr_spearman_ic", 0.0)
        )
        return float(score), metrics
    if target_name == "bars_before_price_stops_decreasing":
        metrics = bars_before_price_stops_decreasing_regression_metrics(y_true, y_pred)
        accuracy = np.mean(
            [
                metrics.get(
                    f"bars_before_price_stops_decreasing_accuracy_by_{bars}_bars",
                    0.0,
                )
                for bars in (1, 2, 4, 8, 12)
            ]
        )
        score = (
            -metrics.get("bars_before_price_stops_decreasing_log_mae", 99.0)
            - 0.25
            * metrics.get("bars_before_price_stops_decreasing_mae_bars", 99.0)
            / 12.0
            + 0.50 * metrics.get("bars_before_price_stops_decreasing_spearman_ic", 0.0)
            + 0.10 * float(accuracy)
        )
        return float(score), metrics
    if target_name == "future_slope_atr_per_hour":
        metrics = future_slope_atr_per_hour_regression_metrics(y_true, y_pred)
        score = (
            -metrics.get("future_slope_atr_per_hour_log_mae", 99.0)
            - 0.20 * metrics.get("future_slope_atr_per_hour_natural_mae", 99.0) / 10.0
            - 0.25 * metrics.get("future_slope_atr_per_hour_natural_huber", 99.0)
            + 0.50 * metrics.get("future_slope_atr_per_hour_spearman_ic", 0.0)
        )
        return float(score), metrics
    raise ValueError(f"unknown auxiliary target: {target_name}")


def select_features_with_current_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    timestamps: Sequence[Any],
    assets: Sequence[Any],
    sides: Sequence[Any],
    archetypes: Sequence[Any],
    mandatory_features_by_side: Mapping[str, Sequence[str]] | None = None,
    target_name: str,
    sample_weight: np.ndarray | None = None,
    random_state: int = 42,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the complete selector independently inside each directional side.

    Every staged selector decision uses the auxiliary regression objective:
    normalized MAE/RMSE plus signed Spearman IC on the requested target.
    """

    from extreme_price_movements.lgbm_pipeline import (
        LGBM_PER_SIDE_FEATURE_SELECTION,
        train_lgbm_stability_candidate,
    )

    target = np.asarray(y, dtype=np.float32)
    if target_name not in TARGET_COLUMNS:
        raise ValueError(f"unknown auxiliary target: {target_name}")
    if not bool(LGBM_PER_SIDE_FEATURE_SELECTION):
        raise RuntimeError(
            "auxiliary heads require independent long/short selection; "
            "EPM_LGBM_PER_SIDE_FEATURE_SELECTION cannot be disabled"
        )
    selector_target = target
    weights = (
        np.ones(len(target), dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    if weights.shape != target.shape or not np.isfinite(weights).all():
        raise ValueError("auxiliary sample_weight must be finite and target-aligned")
    base_params = {
        "objective": "huber",
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
    local_cfg = dict(cfg or {})
    local_mda = dict(local_cfg.get("mda_config", {}) or {})
    local_mda.update(
        {
            "archetype_conditioned_enabled": True,
            "archetype_univariate_prescreen_enabled": False,
            "archetype_relief_prescreen_enabled": False,
            "side_tail_across_archetypes_unweighted": True,
            # Supportive target-derived weights are training-loss only. MDA
            # ranking and every selector evaluation remain ordinary unweighted
            # regression comparisons.
            "use_sample_weight": False,
            "objective": "auxiliary_regression",
            "correlation_pruning_before_prescreen": True,
            "correlation_pruning_threshold": 0.88,
            "correlation_threshold": 0.88,
            "correlation_pruning_floor_ratio": 0.50,
            "correlation_pruning_floor_count": 300,
        }
    )
    local_cfg["mda_config"] = local_mda
    local_cfg["archetype_univariate_prescreen_enabled"] = False
    local_cfg["archetype_relief_prescreen_enabled"] = False
    local_cfg["low_performance_period_weights_enabled"] = False
    # The runner supplies the frozen source-feature universe directly. Do not
    # fit a target-local AE/GMM state inside either auxiliary selector: that
    # would create incomparable generated coordinates across the two heads.
    # Frozen AE/GMM outputs are still eligible when already materialized in X.
    local_cfg["lgbm_ae_gmm_features_enabled"] = False
    candidate_features, universe_report = configured_auxiliary_feature_universe(
        X.columns,
        cfg=local_cfg or None,
    )
    if not candidate_features:
        raise RuntimeError("no configured base/meta auxiliary features are available")
    X_candidate = X.loc[:, candidate_features]
    side_values = np.asarray(sides).astype(str)
    archetype_values = np.asarray(archetypes).astype(str)
    if len(side_values) != len(target) or len(archetype_values) != len(target):
        raise ValueError("side/archetype context must align with the target")
    mandatory_by_side = {
        side: [
            feature
            for feature in map(str, (mandatory_features_by_side or {}).get(side, ()))
            if feature in X_candidate.columns
        ]
        for side in ("long", "short")
    }
    timestamp_values = np.asarray(timestamps)
    asset_values = np.asarray(assets)
    selected_by_side: dict[str, list[str]] = {}
    metrics_by_side: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        side_idx = np.flatnonzero(side_values == side)
        if not len(side_idx):
            raise RuntimeError(
                f"auxiliary feature selection has no {side} rows for {target_name}"
            )
        side_archetypes = np.char.add(
            np.char.add(side_values[side_idx], "__"), archetype_values[side_idx]
        )
        result = train_lgbm_stability_candidate(
            X_candidate.iloc[side_idx].reset_index(drop=True),
            selector_target[side_idx],
            sample_weight=weights[side_idx],
            random_state=int(random_state) + (1009 if side == "long" else 2017),
            mode="regressor",
            timestamps=timestamp_values[side_idx],
            assets=asset_values[side_idx],
            returns=selector_target[side_idx],
            hpo_objective_mode="auxiliary_regression",
            preset_best_params=base_params,
            preset_source=f"{MODEL_SCHEMA}:{target_name}:{side}:selection_only",
            cfg=local_cfg,
            label_context={
                "feature_selection_archetype": side_archetypes,
                "archetype": side_archetypes,
                "side_name": side_values[side_idx],
                "side": side_values[side_idx],
                # Do not apply target-derived weights to side-wide MDA.
                "side_mda_sample_weight": np.ones(len(side_idx), dtype=np.float32),
            },
        )
        if not result:
            raise RuntimeError(f"feature selection failed for {target_name}/{side}")
        side_metrics = dict(result.get("metrics") or {})
        side_selected = list(
            map(
                str,
                dict(
                    side_metrics.get("per_side_feature_selection_selected_features")
                    or {}
                ).get(side, ()),
            )
        )
        if not side_selected:
            raise RuntimeError(
                f"feature selection returned no features for {target_name}/{side}"
            )
        selected_by_side[side] = list(
            dict.fromkeys([*side_selected, *mandatory_by_side[side]])
        )
        metrics_by_side[side] = side_metrics
    if set(selected_by_side) != {"long", "short"}:
        raise RuntimeError(
            "auxiliary feature selection did not produce independent long/short contracts"
        )
    selected = list(
        dict.fromkeys(
            [
                feature
                for side in ("long", "short")
                for feature in selected_by_side[side]
            ]
        )
    )
    if not selected:
        raise RuntimeError(f"feature selection returned no features for {target_name}")
    return {
        "selected_features": selected,
        "selected_features_by_side": selected_by_side,
        "selection_metrics": {
            "contract": "strict_independent_side_selector_runs_v1",
            "by_side": metrics_by_side,
        },
        "feature_universe_report": universe_report,
        "selection_target_orientation": TARGET_COLUMNS[target_name].strip("_"),
        "mandatory_base_archetype_features_by_side": mandatory_by_side,
        "sample_weight_summary": auxiliary_sample_weight_summary(weights),
        "sample_weight_contract": (
            "supportive_target_weights_training_loss_only; "
            "feature_selection_mda_is_unweighted"
        ),
        "prescreen_contract": (
            "strict_side_local_full_pipeline_univariate_relief_mda_v1"
        ),
        "correlation_pruning_threshold": 0.88,
    }


def auxiliary_hpo_sample_indices(
    timestamps: Sequence[Any],
    *,
    max_rows: int = 45_000,
    random_state: int = 42,
) -> np.ndarray:
    """Return a target-neutral beginning/middle/end sample for auxiliary HPO."""

    from extreme_price_movements.lgbm_pipeline import (
        _time_spread_subsample_indices,
    )

    timestamp_values = np.asarray(timestamps)
    return _time_spread_subsample_indices(
        np.arange(len(timestamp_values), dtype=np.float32),
        max_n=max(300, min(int(max_rows), len(timestamp_values))),
        random_state=int(random_state),
        classifier=False,
        timestamps=timestamp_values,
    )


def fit_hpo_oof_model(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    selected_features: Sequence[str],
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    selection_hpo_reference_end: Any,
    target_name: str,
    sample_weight: np.ndarray | None = None,
    n_trials: int = 75,
    hpo_rows: int = 45_000,
    random_state: int = 42,
    purge_hours: float = 13.0,
    preset_hpo_params: Mapping[str, Any] | None = None,
    resume_hpo: Mapping[str, Any] | None = None,
    resume_oof_folds: Mapping[int, Mapping[str, Any]] | None = None,
    resume_final_model: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Tune on the reference period, then score causal expanding monthly OOS."""

    import lightgbm as lgb

    features = [str(c) for c in selected_features]
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise ValueError(f"selected auxiliary features missing: {missing[:20]}")
    target = np.asarray(y, dtype=np.float32)
    weights = (
        np.ones(len(target), dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    if weights.shape != target.shape or not np.isfinite(weights).all():
        raise ValueError("auxiliary sample_weight must be finite and target-aligned")
    timestamp_values = np.asarray(timestamps)
    reference_mask, oof_mask, reference_contract = auxiliary_reference_split(
        timestamp_values,
        label_resolved_at,
        selection_hpo_reference_end=selection_hpo_reference_end,
    )
    if not bool(reference_mask.any()):
        raise ValueError(
            "no rows satisfy the auxiliary selection/HPO reference contract"
        )
    if not bool(oof_mask.any()):
        raise ValueError(
            "no rows at or after the reference end are available for auxiliary OOF emission"
        )
    x = X.loc[:, features].astype(np.float32, copy=False)
    reference_idx = np.flatnonzero(reference_mask)
    reference_x = x.iloc[reference_idx].reset_index(drop=True)
    reference_target = target[reference_idx]
    reference_weight = weights[reference_idx]
    reference_timestamps = timestamp_values[reference_idx]
    n_jobs = default_auxiliary_lgbm_n_jobs()
    hpo_reused = preset_hpo_params is not None or resume_hpo is not None
    hpo_best_value: float | None = None
    hpo_trial_count = 0
    hpo_idx = np.empty(0, dtype=np.int32)
    if hpo_reused:
        final_params = dict(
            (resume_hpo or {}).get("best_params", resume_hpo or preset_hpo_params or {})
        )
        if not final_params:
            raise ValueError("reused auxiliary HPO parameters must be non-empty")
        hpo_best_value = (
            float(resume_hpo["hpo_best_value"])
            if resume_hpo is not None and resume_hpo.get("hpo_best_value") is not None
            else None
        )
        hpo_trial_count = int((resume_hpo or {}).get("hpo_trial_count", 0))
        if progress_callback is not None:
            progress_callback(
                "hpo_reused", {"n_jobs": n_jobs, "hpo_trial_count": hpo_trial_count}
            )
    else:
        import optuna

        hpo_idx = auxiliary_hpo_sample_indices(
            reference_timestamps,
            max_rows=hpo_rows,
            random_state=int(random_state),
        )
        hpo_x = reference_x.iloc[hpo_idx].reset_index(drop=True)
        hpo_target = reference_target[hpo_idx]
        hpo_weight = reference_weight[hpo_idx]
        hpo_timestamps = reference_timestamps[hpo_idx]
        hpo_folds = expanding_purged_folds(
            hpo_timestamps,
            n_splits=3,
            purge_hours=purge_hours,
            min_train_rows=max(200, min(1000, len(hpo_target) // 5)),
            min_valid_rows=max(50, min(200, len(hpo_target) // 20)),
        )
        if not hpo_folds:
            raise ValueError(
                "no valid purged chronological folds on the auxiliary HPO sample"
            )
        trial_best_iterations: dict[int, list[int]] = {}

        def objective(trial: Any) -> float:
            if progress_callback is not None:
                progress_callback("hpo_trial_start", {"trial": int(trial.number)})
            params = {
                "objective": trial.suggest_categorical(
                    "objective", ["regression", "huber", "fair"]
                ),
                "n_estimators": 3000,
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.06, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "num_leaves": trial.suggest_categorical(
                    "num_leaves", [8, 16, 24, 32, 48, 64]
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 100, 1600, log=True
                ),
                "min_split_gain": trial.suggest_float(
                    "min_split_gain", 1e-4, 0.05, log=True
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 40.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.60, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.0),
                "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
                "random_state": int(random_state),
                "n_jobs": n_jobs,
                "verbosity": -1,
            }
            fold_scores: list[float] = []
            fold_iterations: list[int] = []
            for fold_i, fold in enumerate(hpo_folds):
                if progress_callback is not None:
                    progress_callback(
                        "hpo_fold_start", {"trial": int(trial.number), "fold": fold_i}
                    )
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    hpo_x.iloc[fold.train_idx],
                    hpo_target[fold.train_idx],
                    sample_weight=hpo_weight[fold.train_idx],
                    eval_set=[(hpo_x.iloc[fold.valid_idx], hpo_target[fold.valid_idx])],
                    callbacks=[lgb.early_stopping(150, verbose=False)],
                )
                pred = model.predict(hpo_x.iloc[fold.valid_idx])
                score, _ = auxiliary_hpo_objective(
                    target_name, hpo_target[fold.valid_idx], pred
                )
                fold_scores.append(score)
                fold_iterations.append(
                    int(model.best_iteration_ or params["n_estimators"])
                )
                if progress_callback is not None:
                    progress_callback(
                        "hpo_fold_complete",
                        {
                            "trial": int(trial.number),
                            "fold": fold_i,
                            "score": float(score),
                        },
                    )
                trial.report(float(np.mean(fold_scores)), step=fold_i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            trial_best_iterations[int(trial.number)] = fold_iterations
            objective_value = float(np.mean(fold_scores) - 0.25 * np.std(fold_scores))
            if progress_callback is not None:
                progress_callback(
                    "hpo_trial_complete",
                    {"trial": int(trial.number), "score": objective_value},
                )
            return objective_value

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(random_state)),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
        )
        best_completed_value = -np.inf
        non_improving_trials = 0

        def stop_after_stale_trials(study: Any, trial: Any) -> None:
            """Stop only after 30 completed/pruned trials without a new best."""

            nonlocal best_completed_value, non_improving_trials
            if (
                trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
            ):
                value = float(trial.value)
                if value > best_completed_value + 1e-12:
                    best_completed_value = value
                    non_improving_trials = 0
                else:
                    non_improving_trials += 1
            elif trial.state == optuna.trial.TrialState.PRUNED:
                non_improving_trials += 1
            if non_improving_trials >= 30:
                study.stop()

        study.optimize(
            objective,
            n_trials=int(n_trials),
            callbacks=[stop_after_stale_trials],
            show_progress_bar=False,
        )
        best_params = dict(study.best_params)
        best_params.update(
            {
                "n_estimators": 3000,
                "subsample_freq": 1,
                "random_state": int(random_state),
                "n_jobs": n_jobs,
                "verbosity": -1,
            }
        )
        best_iterations = trial_best_iterations.get(
            int(study.best_trial.number), [int(best_params["n_estimators"])]
        )
        final_params = dict(best_params)
        final_params["n_estimators"] = max(50, int(np.median(best_iterations)))
        hpo_best_value = float(study.best_value)
        hpo_trial_count = int(len(study.trials))
        if progress_callback is not None:
            progress_callback(
                "hpo_complete",
                {
                    "best_params": final_params,
                    "hpo_best_value": hpo_best_value,
                    "hpo_trial_count": hpo_trial_count,
                    "n_jobs": n_jobs,
                },
            )

    oos_folds = expanding_monthly_oos_folds(
        timestamp_values,
        label_resolved_at,
        oos_start=selection_hpo_reference_end,
    )
    if not oos_folds:
        raise ValueError(
            "no monthly OOS folds are available at or after the reference end"
        )
    oof = np.full(len(target), np.nan, dtype=np.float32)
    oof_fold_ids = np.full(len(target), -1, dtype=np.int16)
    resolved = pd.to_datetime(pd.Series(label_resolved_at), utc=True, errors="coerce")
    decision = pd.to_datetime(pd.Series(timestamp_values), utc=True, errors="coerce")
    fold_metrics: list[dict[str, Any]] = []
    cached_oof_folds = {
        int(key): value for key, value in (resume_oof_folds or {}).items()
    }
    for fold_i, fold in enumerate(oos_folds):
        train_idx = fold.train_idx[np.isfinite(target[fold.train_idx])]
        if not len(train_idx):
            raise ValueError(
                f"monthly OOS fold {fold.fold_month} has no resolved training rows"
            )
        cached = cached_oof_folds.get(fold_i)
        expected_valid_idx = np.asarray(fold.valid_idx, dtype=np.int64)
        if cached is not None:
            cached_idx = np.asarray(cached.get("valid_idx"), dtype=np.int64)
            cached_prediction = np.asarray(cached.get("prediction"), dtype=np.float32)
            cached_metric = cached.get("metric")
            if (
                not np.array_equal(cached_idx, expected_valid_idx)
                or cached_prediction.shape != (len(expected_valid_idx),)
                or not np.isfinite(cached_prediction).all()
                or not isinstance(cached_metric, Mapping)
            ):
                raise ValueError(f"invalid cached auxiliary OOF fold {fold_i}")
            oof[expected_valid_idx] = cached_prediction
            oof_fold_ids[expected_valid_idx] = int(fold_i)
            fold_metrics.append(dict(cached_metric))
            if progress_callback is not None:
                progress_callback(
                    "oof_fold_reused", {"fold": fold_i, "fold_month": fold.fold_month}
                )
            continue
        if progress_callback is not None:
            progress_callback(
                "oof_fold_start", {"fold": fold_i, "fold_month": fold.fold_month}
            )
        model = lgb.LGBMRegressor(**final_params)
        model.fit(
            x.iloc[train_idx],
            target[train_idx],
            sample_weight=weights[train_idx],
            eval_set=[(x.iloc[fold.valid_idx], target[fold.valid_idx])],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        prediction = model.predict(x.iloc[fold.valid_idx]).astype(np.float32)
        oof[fold.valid_idx] = prediction
        oof_fold_ids[fold.valid_idx] = int(fold_i)
        _, metrics = auxiliary_hpo_objective(
            target_name, target[fold.valid_idx], prediction
        )
        fold_metric = {
            "fold": fold_i,
            "fold_month": fold.fold_month,
            "train_start": fold.train_start.isoformat() if fold.train_start else None,
            "train_end": fold.train_end.isoformat() if fold.train_end else None,
            "valid_start": fold.valid_start.isoformat(),
            "valid_end": fold.valid_end.isoformat(),
            "training_rows": int(len(train_idx)),
            "validation_rows": int(len(fold.valid_idx)),
            "training_label_resolved_bounds": _timestamp_bounds(
                resolved.iloc[train_idx]
            ),
            "validation_decision_bounds": _timestamp_bounds(
                decision.iloc[fold.valid_idx]
            ),
            "oos_model_sha256": _fitted_model_sha256(model),
            "validation_weighted": False,
            **metrics,
        }
        fold_metrics.append(fold_metric)
        if progress_callback is not None:
            progress_callback(
                "oof_fold_complete",
                {
                    "fold": fold_i,
                    "fold_month": fold.fold_month,
                    "valid_idx": expected_valid_idx,
                    "prediction": prediction,
                    "metric": fold_metric,
                },
            )
    final_inference_mask = resolved.notna().to_numpy() & np.isfinite(target)
    if not bool(final_inference_mask.any()):
        raise ValueError("no resolved rows are available for the final inference model")
    expected_final_rows = int(final_inference_mask.sum())
    if resume_final_model is not None:
        final_model = resume_final_model.get("model")
        inference_fit_contract = dict(resume_final_model.get("contract") or {})
        if (
            final_model is None
            or inference_fit_contract.get("rows") != expected_final_rows
        ):
            raise ValueError("invalid cached auxiliary final model")
        if progress_callback is not None:
            progress_callback("final_model_reused", {"rows": expected_final_rows})
    else:
        if progress_callback is not None:
            progress_callback("final_model_start", {"rows": expected_final_rows})
        final_model = lgb.LGBMRegressor(**final_params)
        final_model.fit(
            x.iloc[np.flatnonzero(final_inference_mask)],
            target[final_inference_mask],
            sample_weight=weights[final_inference_mask],
        )
        inference_fit_contract = {
            "fit_row_rule": "all rows with resolved labels; excluded from OOS metrics",
            "rows": expected_final_rows,
            "decision_bounds": _timestamp_bounds(decision.loc[final_inference_mask]),
            "label_resolved_bounds": _timestamp_bounds(
                resolved.loc[final_inference_mask]
            ),
            "model_sha256": _fitted_model_sha256(final_model),
        }
        if progress_callback is not None:
            progress_callback(
                "final_model_complete",
                {"model": final_model, "contract": inference_fit_contract},
            )
    _, overall_metrics = auxiliary_hpo_objective(
        target_name, target[np.isfinite(oof)], oof[np.isfinite(oof)]
    )
    return {
        "schema": MODEL_SCHEMA,
        "target_name": target_name,
        "target_column": TARGET_COLUMNS[target_name],
        "selected_features": features,
        "best_params": final_params,
        "hpo_best_value": hpo_best_value,
        "hpo_trial_count": hpo_trial_count,
        "lgbm_n_jobs": n_jobs,
        "hpo_rows": int(len(hpo_idx)),
        "hpo_sampling_contract": (
            "target-neutral 15k beginning + 15k middle + 15k end when "
            "hpo_rows=45000; sampled only from the frozen reference population"
            if not hpo_reused
            else "reused from an exact selection/HPO fingerprint match"
        ),
        "hpo_reused": bool(hpo_reused),
        "oof_predictions": oof,
        "oof_fold_ids": oof_fold_ids,
        "oof_metrics": overall_metrics,
        "fold_metrics": fold_metrics,
        "model": final_model,
        "final_inference_model": final_model,
        "model_role": "all_resolved_final_inference_excluded_from_oos_metrics",
        "purge_hours": float(purge_hours),
        "sample_weight_summary": auxiliary_sample_weight_summary(weights),
        "reference_sample_weight_summary": auxiliary_sample_weight_summary(
            reference_weight
        ),
        "reference_split_contract": reference_contract,
        "oos_fold_contract": (
            "expanding calendar-month OOS; each fold trains only rows with "
            "decision and label resolution strictly before its validation start"
        ),
        "final_inference_fit_contract": inference_fit_contract,
        "sample_weight_contract": (
            "supportive_target_weights_training_loss_only; validation, HPO "
            "objective, early_stopping, and OOF metrics are unweighted"
        ),
    }


def fit_side_aware_auxiliary_models(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    selected_features_by_side: Mapping[str, Sequence[str]],
    timestamps: Sequence[Any],
    label_resolved_at: Sequence[Any],
    selection_hpo_reference_end: Any,
    sides: Sequence[Any],
    target_name: str,
    sample_weight: np.ndarray | None = None,
    n_trials: int = 75,
    hpo_rows: int = 45_000,
    random_state: int = 42,
    purge_hours: float = 13.0,
    preset_hpo_params_by_side: Mapping[str, Mapping[str, Any]] | None = None,
    resume_by_side: Mapping[str, Mapping[str, Any]] | None = None,
    progress_callback: Callable[[str, str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit independent long and short auxiliary heads and merge their OOF scores."""

    target = np.asarray(y, dtype=np.float32)
    weights = (
        np.ones(len(target), dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    side_values = np.asarray(sides).astype(str)
    if len(target) != len(side_values) or weights.shape != target.shape:
        raise ValueError("sides and sample_weight must align with the auxiliary target")
    if not np.isfinite(weights).all():
        raise ValueError("auxiliary sample_weight must be finite")
    reference_mask, oof_mask, reference_contract = auxiliary_reference_split(
        timestamps,
        label_resolved_at,
        selection_hpo_reference_end=selection_hpo_reference_end,
    )
    if not bool(reference_mask.any()):
        raise ValueError(
            "no rows satisfy the auxiliary selection/HPO reference contract"
        )
    if not bool(oof_mask.any()):
        raise ValueError(
            "no rows at or after the reference end are available for auxiliary OOF emission"
        )
    oof = np.full(len(target), np.nan, dtype=np.float32)
    oof_fold_ids = np.full(len(target), -1, dtype=np.int16)
    bundles: dict[str, dict[str, Any]] = {}
    for offset, side in enumerate(("long", "short"), start=1):
        idx = np.flatnonzero(side_values == side)
        if not len(idx):
            raise ValueError(f"no rows available for auxiliary side {side}")
        features = list(map(str, selected_features_by_side.get(side, [])))
        if not features:
            raise ValueError(f"no selected auxiliary features for side {side}")

        def side_progress(
            event: str, payload: Mapping[str, Any], *, _side: str = side
        ) -> None:
            if progress_callback is not None:
                progress_callback(event, _side, payload)

        resumed_side = dict((resume_by_side or {}).get(side) or {})
        bundle = fit_hpo_oof_model(
            X.iloc[idx].reset_index(drop=True),
            target[idx],
            selected_features=features,
            timestamps=np.asarray(timestamps)[idx],
            label_resolved_at=np.asarray(label_resolved_at)[idx],
            selection_hpo_reference_end=selection_hpo_reference_end,
            target_name=target_name,
            sample_weight=weights[idx],
            n_trials=n_trials,
            hpo_rows=hpo_rows,
            random_state=int(random_state) + offset * 2003,
            purge_hours=purge_hours,
            preset_hpo_params=(preset_hpo_params_by_side or {}).get(side),
            resume_hpo=resumed_side.get("hpo"),
            resume_oof_folds=resumed_side.get("oof_folds"),
            resume_final_model=resumed_side.get("final_model"),
            progress_callback=side_progress,
        )
        side_oof = np.asarray(bundle.pop("oof_predictions"), dtype=np.float32)
        side_fold_ids = np.asarray(bundle.pop("oof_fold_ids"), dtype=np.int16)
        oof[idx] = side_oof
        oof_fold_ids[idx] = side_fold_ids
        bundles[side] = bundle
    valid = np.isfinite(oof) & np.isfinite(target)
    _, overall_metrics = auxiliary_hpo_objective(target_name, target[valid], oof[valid])
    return {
        "schema": MODEL_SCHEMA,
        "target_name": target_name,
        "target_column": TARGET_COLUMNS[target_name],
        "models_by_side": bundles,
        "oof_predictions": oof,
        "oof_fold_ids": oof_fold_ids,
        "oof_metrics": overall_metrics,
        "purge_hours": float(purge_hours),
        "side_contract": "independent_long_short_models_v1",
        "sample_weight_summary": auxiliary_sample_weight_summary(weights),
        "reference_split_contract": reference_contract,
        "sample_weight_contract": (
            "supportive_target_weights_training_loss_only; validation, HPO "
            "objective, early_stopping, and OOF metrics are unweighted"
        ),
        "hpo_reused": bool(preset_hpo_params_by_side),
    }
