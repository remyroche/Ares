"""Bounded, leakage-aware CatBoost search for realised path-geometry targets.

All path values here are future realised outcomes.  They are labels and
diagnostics only, never model features.  The model matrix is supplied separately
and evaluated only on chronological out-of-sample folds.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .catboost_archetype_classifier import (
    capped_catboost_params,
    validate_preentry_features,
)

PATH_GEOMETRY_CLASSES: tuple[str, ...] = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
    "noisy_timeout_usable_mfe",
    "dead_timeout",
)
PATH_GEOMETRY_COST_RETURN = 0.01
GEOMETRY_TRAIN_MONTHS = 4
GEOMETRY_OOS_MONTHS = 4
GEOMETRY_NESTED_MONTHS = 12
GEOMETRY_EVALUATION_MODE_LEGACY = "legacy_4m4m"
GEOMETRY_EVALUATION_MODE_SHORT_HISTORY = "short_history_purged_april_v1"
DEFAULT_MAX_TRAIN_ROWS_PER_FOLD = 70_000
GEOMETRY_EARLY_STOP_VALIDATION_FRACTION = 0.20
GEOMETRY_EARLY_STOP_EMBARGO = pd.Timedelta(hours=24)
CALIBRATION_ECE_BINS = 10
_HOURS = tuple(range(1, 13))


@dataclass(frozen=True)
class PathGeometryColumns:
    """Path-summary and identity columns used by dynamic geometry labels.

    Raw MFE/MAE/variation columns are cumulative values at the end of every
    future hour.  ``risk_fraction`` may be absent when ``risk_distance`` and
    ``entry_price`` are present; :func:`ensure_risk_fraction` then derives it.
    """

    timestamp: str = "__ts__"
    label_end: str | None = "__label_end_ts__"
    symbol: str = "__symbol__"
    side: str = "side"
    close_return_r_12h: str = "path_arch_close_return_r_12h"
    time_to_stop_h: str = "path_arch_time_to_stop_h"
    time_to_trailing_h: str = "path_arch_time_to_trailing_h"
    atr_fraction: str = "path_arch_atr_fraction"
    risk_fraction: str = "path_arch_risk_fraction"
    risk_distance: str = "risk_distance"
    entry_price: str = "entry_price"
    raw_mfe_atr_prefix: str = "path_arch_raw_mfe_atr_"
    raw_mfe_r_prefix: str = "path_arch_raw_mfe_r_"
    raw_mae_r_prefix: str = "path_arch_raw_mae_r_"
    cumulative_variation_r_prefix: str = "path_arch_cumulative_variation_r_"

    def raw_column(self, kind: str, hour: int) -> str:
        prefix = getattr(self, f"{kind}_prefix")
        return f"{prefix}{int(hour)}h"

    def required(self) -> tuple[str, ...]:
        fixed = (self.close_return_r_12h, self.time_to_stop_h, self.atr_fraction)
        raw = tuple(
            self.raw_column(kind, hour)
            for kind in (
                "raw_mfe_atr",
                "raw_mfe_r",
                "raw_mae_r",
                "cumulative_variation_r",
            )
            for hour in _HOURS
        )
        return fixed + raw


@dataclass(frozen=True)
class PathGeometryConfig:
    atr_floor: float = 1.5
    net_margin_atr: float = 0.5
    early_stop_window: float = 2.0
    favorable_exemption_multiplier: float = 1.0
    fast_meaningful_time: float = 2.0
    peak_fraction: float = 0.9
    fast_peak_limit: float = 4.0
    # Separate from the broad net-margin boundary so a sparse fast state can
    # be relaxed without changing the slow/late archetype economics.
    fast_net_margin_atr: float | None = None
    reversal_peak_limit: float = 8.0
    clean_adverse_ratio: float = 0.33
    early_mfe_ceiling_r: float = 0.5
    late_mfe_floor_r: float = 1.0
    expansion_floor_r: float = 0.75
    reversal_retention_cap: float = 0.2
    reversal_mode: str = "either"
    usable_mfe_multiplier: float = 0.75
    cost_return: float = PATH_GEOMETRY_COST_RETURN

    def validate(self) -> None:
        if self.reversal_mode not in {
            "final_net_nonpositive",
            "retention_cap",
            "either",
        }:
            raise ValueError(
                "reversal_mode must be final_net_nonpositive, retention_cap, or either"
            )
        if not np.isclose(self.cost_return, PATH_GEOMETRY_COST_RETURN):
            raise ValueError("path geometry search has a fixed 1% execution cost")
        if self.peak_fraction not in {0.8, 0.9}:
            raise ValueError("peak_fraction must be one of 0.8 or 0.9")
        if (
            min(self.atr_floor, self.expansion_floor_r, self.usable_mfe_multiplier)
            <= 0.0
        ):
            raise ValueError("geometry floors and multipliers must be positive")
        if self.fast_net_margin_atr is not None and self.fast_net_margin_atr <= 0.0:
            raise ValueError("fast_net_margin_atr must be positive when supplied")

    @property
    def effective_fast_net_margin_atr(self) -> float:
        return float(
            self.net_margin_atr
            if self.fast_net_margin_atr is None
            else self.fast_net_margin_atr
        )


GEOMETRY_GRID: dict[str, tuple[Any, ...]] = {
    "atr_floor": (1.25, 1.5, 1.75),
    "net_margin_atr": (0.25, 0.5, 0.75),
    "early_stop_window": (1.0, 2.0, 3.0),
    "favorable_exemption_multiplier": (0.75, 1.0, 1.25),
    "fast_meaningful_time": (1.0, 2.0, 3.0),
    "peak_fraction": (0.8, 0.9),
    "fast_peak_limit": (3.0, 4.0, 6.0),
    "fast_net_margin_atr": (0.25, 0.5, 0.75),
    "reversal_peak_limit": (6.0, 8.0, 10.0),
    "clean_adverse_ratio": (0.25, 0.33, 0.5),
    "early_mfe_ceiling_r": (0.33, 0.5, 0.67),
    "late_mfe_floor_r": (0.75, 1.0, 1.25),
    "expansion_floor_r": (0.5, 0.75, 1.0),
    "reversal_retention_cap": (0.1, 0.2, 0.3),
    "reversal_mode": ("final_net_nonpositive", "retention_cap", "either"),
    "usable_mfe_multiplier": (0.5, 0.75, 1.0),
}


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def ensure_risk_fraction(
    frame: pd.DataFrame, columns: PathGeometryColumns = PathGeometryColumns()
) -> pd.DataFrame:
    """Return a copy with valid risk fraction, deriving it when necessary."""
    out = frame.copy()
    if columns.risk_fraction not in out:
        missing = {columns.risk_distance, columns.entry_price}.difference(out.columns)
        if missing:
            raise ValueError(
                "exact 1% cost needs risk_fraction or risk_distance/entry_price; "
                f"missing={sorted(missing)}"
            )
        risk = _numeric(out, columns.risk_distance)
        entry = _numeric(out, columns.entry_price)
        out[columns.risk_fraction] = np.divide(
            risk, entry, out=np.full(len(out), np.nan), where=entry > 0.0
        ).astype(np.float32)
    risk_fraction = _numeric(out, columns.risk_fraction)
    if not np.isfinite(risk_fraction).all() or (risk_fraction <= 0.0).any():
        raise ValueError(f"{columns.risk_fraction!r} must be finite and positive")
    return out


def validate_path_summary(
    frame: pd.DataFrame, columns: PathGeometryColumns = PathGeometryColumns()
) -> None:
    missing = sorted(set(columns.required()).difference(frame.columns))
    if missing:
        raise ValueError(
            f"path summary is missing required dynamic geometry columns: {missing}"
        )
    atr = _numeric(frame, columns.atr_fraction)
    if not np.isfinite(atr).all() or (atr <= 0.0).any():
        raise ValueError(f"{columns.atr_fraction!r} must be finite and positive")


def _raw_matrix(
    frame: pd.DataFrame, columns: PathGeometryColumns, kind: str
) -> np.ndarray:
    return np.column_stack(
        [_numeric(frame, columns.raw_column(kind, hour)) for hour in _HOURS]
    )


def _geometry_values(
    frame: pd.DataFrame, columns: PathGeometryColumns
) -> dict[str, np.ndarray]:
    validate_path_summary(frame, columns)
    values: dict[str, np.ndarray] = {
        "close_return_r_12h": _numeric(frame, columns.close_return_r_12h),
        "time_to_stop_h": _numeric(frame, columns.time_to_stop_h),
        "time_to_trailing_h": _numeric(frame, columns.time_to_trailing_h)
        if columns.time_to_trailing_h in frame
        else np.full(len(frame), np.nan),
        "atr_fraction": _numeric(frame, columns.atr_fraction),
        "risk_fraction": _numeric(frame, columns.risk_fraction),
        "mfe_atr": _raw_matrix(frame, columns, "raw_mfe_atr"),
        "mfe_r": _raw_matrix(frame, columns, "raw_mfe_r"),
        "mae_r": _raw_matrix(frame, columns, "raw_mae_r"),
        "variation_r": _raw_matrix(frame, columns, "cumulative_variation_r"),
    }
    values["cost_r"] = PATH_GEOMETRY_COST_RETURN / values["risk_fraction"]
    values["cost_atr"] = PATH_GEOMETRY_COST_RETURN / values["atr_fraction"]
    values["net_final_r"] = values["close_return_r_12h"] - values["cost_r"]
    values["net_final_atr"] = (
        values["net_final_r"] * values["risk_fraction"] / values["atr_fraction"]
    )
    values["peak_mfe_r"] = values["mfe_r"][:, -1]
    values["peak_mfe_atr"] = values["mfe_atr"][:, -1]
    values["retention_net"] = values["net_final_r"] / np.maximum(
        values["peak_mfe_r"] - values["cost_r"], 1e-6
    )
    return values


def _first_crossing(matrix: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    crossed = matrix >= thresholds[:, None]
    first = crossed.argmax(axis=1).astype(np.int16)
    first[~crossed.any(axis=1)] = -1
    return first


def _dynamic_path_metrics(
    values: Mapping[str, np.ndarray], config: PathGeometryConfig
) -> dict[str, np.ndarray]:
    meaningful_atr = np.maximum(
        config.atr_floor, values["cost_atr"] + config.net_margin_atr
    )
    meaningful_r = meaningful_atr * values["atr_fraction"] / values["risk_fraction"]
    first = _first_crossing(values["mfe_atr"], meaningful_atr)
    peak_fraction_r = config.peak_fraction * values["peak_mfe_r"]
    peak_fraction_first = _first_crossing(values["mfe_r"], peak_fraction_r)
    valid_first = first >= 0
    pre_cross_mae = np.full(len(first), np.nan)
    pre_cross_mfe = np.full(len(first), np.nan)
    row_index = np.arange(len(first))
    clipped_first = np.clip(first, 0, len(_HOURS) - 1)
    pre_cross_mae[valid_first] = values["mae_r"][
        row_index[valid_first], clipped_first[valid_first]
    ]
    pre_cross_mfe[valid_first] = values["mfe_r"][
        row_index[valid_first], clipped_first[valid_first]
    ]
    stopped = np.isfinite(values["time_to_stop_h"])
    safe_stop_time = np.where(stopped, values["time_to_stop_h"], 1.0)
    stop_index = np.clip(np.ceil(safe_stop_time).astype(np.int64) - 1, 0, 11)
    mfe_before_stop_atr = np.full(len(first), np.nan)
    mfe_before_stop_atr[stopped] = values["mfe_atr"][
        row_index[stopped], stop_index[stopped]
    ]
    efficiency = np.divide(
        values["net_final_r"],
        values["variation_r"][:, -1],
        out=np.full(len(first), np.nan),
        where=values["variation_r"][:, -1] > 0.0,
    )
    return {
        "meaningful_atr": meaningful_atr,
        "meaningful_r": meaningful_r,
        "first_meaningful_index": first,
        "first_meaningful_h": np.where(first >= 0, first + 1.0, np.nan),
        "peak_fraction_index": peak_fraction_first,
        "peak_fraction_h": np.where(
            peak_fraction_first >= 0, peak_fraction_first + 1.0, np.nan
        ),
        "pre_cross_mae_r": pre_cross_mae,
        "pre_cross_mfe_r": pre_cross_mfe,
        "mfe_before_stop_atr": mfe_before_stop_atr,
        "efficiency": efficiency,
    }


def label_path_geometry(
    frame: pd.DataFrame,
    config: PathGeometryConfig = PathGeometryConfig(),
    columns: PathGeometryColumns = PathGeometryColumns(),
    *,
    precomputed_values: Mapping[str, np.ndarray] | None = None,
    labels_only: bool = False,
) -> pd.DataFrame:
    """Materialise deterministic seven-class labels with dynamic boundaries."""
    config.validate()
    if precomputed_values is None:
        prepared = ensure_risk_fraction(frame, columns)
        values = _geometry_values(prepared, columns)
    else:
        values = precomputed_values
        required_values = {
            "close_return_r_12h",
            "time_to_stop_h",
            "time_to_trailing_h",
            "atr_fraction",
            "risk_fraction",
            "mfe_atr",
            "mfe_r",
            "mae_r",
            "variation_r",
            "cost_r",
            "cost_atr",
            "net_final_r",
            "net_final_atr",
            "peak_mfe_r",
            "peak_mfe_atr",
            "retention_net",
        }
        missing = sorted(required_values.difference(values))
        if missing:
            raise ValueError(f"precomputed geometry values are missing: {missing}")
        if any(len(np.asarray(values[name])) != len(frame) for name in required_values):
            raise ValueError("precomputed geometry values do not align with frame rows")
    dynamic = _dynamic_path_metrics(values, config)
    finite = np.isfinite(values["close_return_r_12h"]) & np.isfinite(
        values["peak_mfe_r"]
    )
    finite &= np.isfinite(values["mfe_r"][:, 3]) & np.isfinite(values["mfe_r"][:, 11])
    finite &= np.isfinite(values["mae_r"]).all(axis=1) & np.isfinite(
        values["variation_r"][:, -1]
    )
    stopped_early = np.isfinite(values["time_to_stop_h"]) & (
        values["time_to_stop_h"] <= config.early_stop_window
    )
    favorable_exempt = dynamic["mfe_before_stop_atr"] >= (
        config.favorable_exemption_multiplier * dynamic["meaningful_atr"]
    )
    retention_cap = values["retention_net"] <= config.reversal_retention_cap
    final_net_bad = values["net_final_r"] <= 0.0
    reversal_outcome = {
        "final_net_nonpositive": final_net_bad,
        "retention_cap": retention_cap,
        "either": final_net_bad | retention_cap,
    }[config.reversal_mode]
    usable_floor_r = config.usable_mfe_multiplier * dynamic["meaningful_r"]
    usable_mfe = values["peak_mfe_r"] >= usable_floor_r
    fast = (
        np.isfinite(dynamic["first_meaningful_h"])
        & (dynamic["first_meaningful_h"] <= config.fast_meaningful_time)
        & np.isfinite(dynamic["peak_fraction_h"])
        & (dynamic["peak_fraction_h"] <= config.fast_peak_limit)
        & (values["net_final_atr"] >= config.effective_fast_net_margin_atr)
    )
    clean_ratio = np.divide(
        np.abs(dynamic["pre_cross_mae_r"]),
        dynamic["meaningful_r"],
        out=np.full(len(frame), np.nan),
        where=dynamic["meaningful_r"] > 0.0,
    )
    late = (
        (values["mfe_r"][:, 3] <= config.early_mfe_ceiling_r)
        & (values["mfe_r"][:, 11] >= config.late_mfe_floor_r)
        & ((values["mfe_r"][:, 11] - values["mfe_r"][:, 3]) >= config.expansion_floor_r)
        & (values["net_final_atr"] >= config.net_margin_atr)
    )
    reversal = (
        (values["peak_mfe_atr"] >= dynamic["meaningful_atr"])
        & np.isfinite(dynamic["peak_fraction_h"])
        & (dynamic["peak_fraction_h"] <= config.reversal_peak_limit)
        & reversal_outcome
    )
    slow = usable_mfe & (values["net_final_r"] > 0.0)
    ordered_rules = (
        (
            "immediate_adverse_path",
            "immediate_adverse",
            stopped_early & ~favorable_exempt,
        ),
        ("early_mfe_full_reversal", "early_reversal", reversal),
        ("fast_realization_winner", "fast_realization", fast),
        ("late_breakout", "late_breakout", late),
        ("slow_grinder", "slow_usable_positive", slow),
        ("noisy_timeout_usable_mfe", "usable_timeout", usable_mfe),
    )
    labels = np.full(len(frame), None, dtype=object)
    rules = np.full(len(frame), "invalid", dtype=object)
    remaining = finite.copy()
    for label, rule, mask in ordered_rules:
        selected = remaining & mask
        labels[selected], rules[selected] = label, rule
        remaining &= ~selected
    labels[remaining], rules[remaining] = "dead_timeout", "dead_timeout"
    if labels_only:
        return pd.DataFrame(
            {"path_geometry_label": pd.array(labels, dtype="string")},
            index=frame.index,
        )

    predicate_margins = {
        "immediate_stop": config.early_stop_window - values["time_to_stop_h"],
        "favorable_exemption": dynamic["mfe_before_stop_atr"]
        - config.favorable_exemption_multiplier * dynamic["meaningful_atr"],
        "reversal_meaningful": values["peak_mfe_atr"] - dynamic["meaningful_atr"],
        "reversal_peak_time": config.reversal_peak_limit - dynamic["peak_fraction_h"],
        "reversal_final_net": -values["net_final_r"],
        "reversal_retention": config.reversal_retention_cap - values["retention_net"],
        "fast_meaningful_time": config.fast_meaningful_time
        - dynamic["first_meaningful_h"],
        "fast_peak_time": config.fast_peak_limit - dynamic["peak_fraction_h"],
        "fast_net_margin": values["net_final_atr"]
        - config.effective_fast_net_margin_atr,
        "clean_mae_ratio": config.clean_adverse_ratio - clean_ratio,
        "late_early_ceiling": config.early_mfe_ceiling_r - values["mfe_r"][:, 3],
        "late_late_floor": values["mfe_r"][:, 11] - config.late_mfe_floor_r,
        "late_expansion": (values["mfe_r"][:, 11] - values["mfe_r"][:, 3])
        - config.expansion_floor_r,
        "late_net_margin": values["net_final_atr"] - config.net_margin_atr,
        "slow_usable_mfe": values["peak_mfe_r"] - usable_floor_r,
        "slow_positive_net": values["net_final_r"],
        "noisy_usable_mfe": values["peak_mfe_r"] - usable_floor_r,
    }
    candidate_matches = (
        stopped_early & ~favorable_exempt,
        reversal,
        fast,
        late,
        slow,
        usable_mfe,
    )
    matching_count = np.sum(np.column_stack(candidate_matches), axis=1).astype(np.int8)
    dead_timeout_match = finite & (matching_count == 0)
    matching_count = matching_count + dead_timeout_match.astype(np.int8)
    margin_matrix = np.column_stack(list(predicate_margins.values()))
    absolute_margins = np.abs(margin_matrix)
    finite_margin = np.isfinite(absolute_margins).any(axis=1)
    minimum_boundary_distance = np.where(
        finite_margin,
        np.where(np.isfinite(absolute_margins), absolute_margins, np.inf).min(axis=1),
        np.nan,
    )
    return pd.DataFrame(
        {
            "path_geometry_label": pd.array(labels, dtype="string"),
            "path_geometry_rule": pd.array(rules, dtype="string"),
            "dynamic_meaningful_mfe_atr": dynamic["meaningful_atr"],
            "dynamic_meaningful_mfe_r": dynamic["meaningful_r"],
            "first_dynamic_meaningful_h": dynamic["first_meaningful_h"],
            "peak_fraction_time_h": dynamic["peak_fraction_h"],
            "pre_dynamic_meaningful_mae_r": dynamic["pre_cross_mae_r"],
            "pre_dynamic_meaningful_mae_ratio": clean_ratio,
            "dynamic_efficiency": dynamic["efficiency"],
            "net_retention_after_1pct": values["retention_net"],
            "number_of_matching_archetypes": matching_count,
            "precedence_override_flag": matching_count > 1,
            "minimum_archetype_boundary_distance": minimum_boundary_distance,
            "boundary_meaningful_atr": values["peak_mfe_atr"]
            - dynamic["meaningful_atr"],
            "boundary_usable_mfe_r": values["peak_mfe_r"] - usable_floor_r,
            "boundary_reversal_retention": config.reversal_retention_cap
            - values["retention_net"],
            "boundary_late_expansion_r": (
                values["mfe_r"][:, 11] - values["mfe_r"][:, 3]
            )
            - config.expansion_floor_r,
            "net_ev_after_1pct_return": values["close_return_r_12h"]
            * values["risk_fraction"]
            - PATH_GEOMETRY_COST_RETURN,
            "net_ev_after_1pct_r": values["net_final_r"],
            "stop_probability": np.isfinite(values["time_to_stop_h"]).astype(
                np.float32
            ),
            "trailing_conversion": np.isfinite(values["time_to_trailing_h"]).astype(
                np.float32
            ),
            **{f"margin_{name}": margin for name, margin in predicate_margins.items()},
        },
        index=frame.index,
    )


def boundary_diagnostics(labels: pd.DataFrame) -> pd.DataFrame:
    valid = labels.loc[labels["path_geometry_label"].notna()]
    rows: list[dict[str, Any]] = []
    for class_name in PATH_GEOMETRY_CLASSES:
        group = valid.loc[valid["path_geometry_label"].eq(class_name)]
        row: dict[str, Any] = {
            "path_geometry_label": class_name,
            "rows": int(len(group)),
        }
        for column in (
            "boundary_meaningful_atr",
            "boundary_usable_mfe_r",
            "boundary_reversal_retention",
            "boundary_late_expansion_r",
        ):
            data = pd.to_numeric(group.get(column), errors="coerce").to_numpy(
                dtype=float
            )
            row[f"median_{column}"] = float(np.nanmedian(data)) if len(data) else np.nan
            row[f"within_0p05_{column}"] = (
                float(np.nanmean(np.abs(data) <= 0.05)) if len(data) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _normalise_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    """Validate and normalise one non-negative multiclass probability matrix."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("probability matrix must be non-empty and two-dimensional")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("probability matrix must be finite and non-negative")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0.0).any():
        raise ValueError("each probability row must have positive mass")
    return values / totals


def _align_probabilities(
    probabilities: np.ndarray, classes: Sequence[str]
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(classes):
        raise ValueError("probability matrix does not match its supplied class order")
    if len(set(classes)) != len(classes):
        raise ValueError("probability class order contains duplicate labels")
    values = _normalise_probability_rows(values)
    aligned = np.full((len(values), len(PATH_GEOMETRY_CLASSES)), 1e-12, dtype=float)
    for source, name in enumerate(classes):
        if name in PATH_GEOMETRY_CLASSES:
            aligned[:, PATH_GEOMETRY_CLASSES.index(name)] = values[:, source]
    return _normalise_probability_rows(aligned)


def confidence_metrics(probabilities: np.ndarray, prefix: str) -> dict[str, float]:
    """Return confidence summaries for raw probabilities."""
    values = _normalise_probability_rows(probabilities)
    entropy = -np.sum(values * np.log(np.clip(values, 1e-12, 1.0)), axis=1)
    ordered = np.sort(values, axis=1)
    return {
        f"{prefix}_mean_max_probability": float(values.max(axis=1).mean()),
        f"{prefix}_entropy": float(entropy.mean()),
        f"{prefix}_normalized_entropy": float(
            (entropy / np.log(len(PATH_GEOMETRY_CLASSES))).mean()
        ),
        f"{prefix}_top1_top2_probability_margin": float(
            (ordered[:, -1] - ordered[:, -2]).mean()
        ),
    }


def _raw_probability_reliability_metrics(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    prefix: str,
    *,
    bins: int = CALIBRATION_ECE_BINS,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return raw multiclass Brier/ECE metrics and reliability bins."""
    if bins < 2:
        raise ValueError("probability ECE requires at least two bins")
    values = _normalise_probability_rows(probabilities)
    labels = np.asarray(
        [PATH_GEOMETRY_CLASSES.index(value) for value in y_true], dtype=np.int64
    )
    if len(labels) != len(values):
        raise ValueError("probability labels and predictions must have equal length")
    observed = np.eye(len(PATH_GEOMETRY_CLASSES), dtype=float)[labels]
    metrics: dict[str, float] = {
        f"{prefix}_oos_logloss": float(
            -np.mean(
                np.log(np.clip(values[np.arange(len(labels)), labels], 1e-12, 1.0))
            )
        ),
        f"{prefix}_multiclass_brier": float(
            np.mean(np.sum((values - observed) ** 2, axis=1))
        ),
    }
    rows: list[dict[str, Any]] = []
    eces: list[float] = []
    for class_index, class_name in enumerate(PATH_GEOMETRY_CLASSES):
        score = values[:, class_index]
        truth = observed[:, class_index]
        bin_index = np.minimum((score * bins).astype(np.int64), bins - 1)
        ece = 0.0
        for bin_id in range(bins):
            mask = bin_index == bin_id
            support = int(mask.sum())
            mean_probability = float(score[mask].mean()) if support else np.nan
            empirical_frequency = float(truth[mask].mean()) if support else np.nan
            absolute_error = (
                abs(mean_probability - empirical_frequency) if support else np.nan
            )
            if support:
                ece += (support / len(labels)) * float(absolute_error)
            rows.append(
                {
                    "probability_variant": prefix,
                    "class_name": class_name,
                    "bin_id": int(bin_id),
                    "bin_lower": float(bin_id / bins),
                    "bin_upper": float((bin_id + 1) / bins),
                    "rows": support,
                    "mean_probability": mean_probability,
                    "empirical_frequency": empirical_frequency,
                    "absolute_probability_error": absolute_error,
                }
            )
        metrics[f"{prefix}_classwise_ece_{class_name}"] = float(ece)
        eces.append(float(ece))
    metrics[f"{prefix}_macro_ece"] = float(np.mean(eces))
    return metrics, pd.DataFrame(rows)


def _f1(y: np.ndarray, predicted: np.ndarray, weighted: bool) -> float:
    scores, supports = [], []
    for label in range(len(PATH_GEOMETRY_CLASSES)):
        tp = float(np.sum((y == label) & (predicted == label)))
        fp = float(np.sum((y != label) & (predicted == label)))
        fn = float(np.sum((y == label) & (predicted != label)))
        scores.append(
            0.0 if 2.0 * tp + fp + fn == 0.0 else 2.0 * tp / (2.0 * tp + fp + fn)
        )
        supports.append(float(np.sum(y == label)))
    return (
        float(np.average(scores, weights=supports))
        if weighted
        else float(np.mean(scores))
    )


def multiclass_scores(
    y_true: Sequence[str], probabilities: np.ndarray, classes: Sequence[str]
) -> dict[str, float]:
    y = np.asarray([PATH_GEOMETRY_CLASSES.index(value) for value in y_true], dtype=int)
    p = _align_probabilities(np.asarray(probabilities, dtype=float), classes)
    predicted = p.argmax(axis=1)
    observed = np.cumsum(np.eye(len(PATH_GEOMETRY_CLASSES))[y], axis=1)
    rps = np.mean(
        np.sum((np.cumsum(p, axis=1)[:, :-1] - observed[:, :-1]) ** 2, axis=1) / 7.0
    )
    return {
        "oos_logloss": float(
            -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))
        ),
        "macro_f1": _f1(y, predicted, False),
        "weighted_f1": _f1(y, predicted, True),
        "ranked_probability_score": float(rps),
        "confusion_distance": float(np.mean(np.abs(predicted - y))),
    }


def _wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    quantiles = np.linspace(0.0, 1.0, 101)
    return float(
        np.mean(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles)))
    )


def _separation_for_groups(
    values: Mapping[str, np.ndarray], groups: Sequence[str], prefix: str
) -> dict[str, float]:
    assigned = np.asarray(groups, dtype=object)
    result: dict[str, float] = {}
    aggregate: list[float] = []
    for name, metric in values.items():
        effects, distances = [], []
        global_scale = max(float(np.nanstd(metric)), 1e-6)
        for left, right in combinations(PATH_GEOMETRY_CLASSES, 2):
            first, second = metric[assigned == left], metric[assigned == right]
            first, second = first[np.isfinite(first)], second[np.isfinite(second)]
            if len(first) < 2 or len(second) < 2:
                continue
            scale = max(
                np.sqrt((np.var(first, ddof=1) + np.var(second, ddof=1)) / 2.0),
                global_scale,
                1e-6,
            )
            effects.append(abs(float(first.mean() - second.mean())) / scale)
            distances.append(_wasserstein(first, second) / scale)
        result[f"{prefix}{name}_pairwise_effect_size"] = (
            float(np.mean(effects)) if effects else np.nan
        )
        result[f"{prefix}{name}_standardized_wasserstein"] = (
            float(np.mean(distances)) if distances else np.nan
        )
        if effects:
            aggregate.append(float(np.mean(effects)))
    result[f"{prefix}economic_separation_score"] = (
        float(np.mean(aggregate)) if aggregate else np.nan
    )
    return result


def economic_separation(
    outcomes: pd.DataFrame,
    true_classes: Sequence[str],
    predicted_classes: Sequence[str] | None = None,
    columns: PathGeometryColumns = PathGeometryColumns(),
) -> dict[str, float]:
    """Primary true-class economics, optionally accompanied by predicted groups."""
    values = _geometry_values(ensure_risk_fraction(outcomes, columns), columns)
    metrics = {
        "gross_mfe_r": values["peak_mfe_r"],
        "net_ev_after_1pct_return": values["close_return_r_12h"]
        * values["risk_fraction"]
        - PATH_GEOMETRY_COST_RETURN,
        "mae_r": values["mae_r"][:, -1],
        "stop_probability": np.isfinite(values["time_to_stop_h"]).astype(float),
        "time_h": _first_crossing(values["mfe_atr"], values["peak_mfe_atr"] * 0.9)
        + 1.0,
        "retention_after_1pct": values["retention_net"],
        "trailing_conversion": np.isfinite(values["time_to_trailing_h"]).astype(float),
    }
    result = _separation_for_groups(metrics, true_classes, "true_")
    result["economic_separation_score"] = result["true_economic_separation_score"]
    if predicted_classes is not None:
        result.update(_separation_for_groups(metrics, predicted_classes, "predicted_"))
    return result


def economic_confusion_diagnostics(
    train_outcomes: pd.DataFrame,
    train_classes: Sequence[str],
    oos_true_classes: Sequence[str],
    oos_predicted_classes: Sequence[str],
    columns: PathGeometryColumns = PathGeometryColumns(),
) -> dict[str, Any]:
    """Score OOS confusion using only train-derived class execution EV priors."""
    train_labels = np.asarray(train_classes, dtype=object)
    truth = np.asarray(oos_true_classes, dtype=object)
    predicted = np.asarray(oos_predicted_classes, dtype=object)
    if len(train_outcomes) != len(train_labels):
        raise ValueError("train outcomes and train classes must have equal length")
    if len(truth) != len(predicted) or not len(truth):
        raise ValueError(
            "economic confusion requires non-empty equal-length OOS labels"
        )
    unknown = (
        set(train_labels)
        .union(truth)
        .union(predicted)
        .difference(PATH_GEOMETRY_CLASSES)
    )
    if unknown:
        raise ValueError(
            f"economic confusion received unknown geometry classes: {sorted(unknown)}"
        )
    train_values = _geometry_values(
        ensure_risk_fraction(train_outcomes, columns), columns
    )
    train_ev = train_values["net_final_r"] * train_values["risk_fraction"]
    if not np.isfinite(train_ev).all():
        raise ValueError("train execution EV priors must be finite")
    global_prior = float(np.mean(train_ev))
    priors: list[float] = []
    prior_rows: list[dict[str, Any]] = []
    for class_name in PATH_GEOMETRY_CLASSES:
        mask = train_labels == class_name
        support = int(mask.sum())
        prior = float(train_ev[mask].mean()) if support else global_prior
        priors.append(prior)
        prior_rows.append(
            {
                "class_name": class_name,
                "train_rows": support,
                "reference_geometry_net_ev_prior": prior,
                "prior_source": "class_train_mean"
                if support
                else "global_train_mean_fallback_no_class_support",
            }
        )
    prior_values = np.asarray(priors, dtype=float)
    class_index = {name: index for index, name in enumerate(PATH_GEOMETRY_CLASSES)}
    actual_index = np.asarray([class_index[value] for value in truth], dtype=np.int64)
    predicted_index = np.asarray(
        [class_index[value] for value in predicted], dtype=np.int64
    )
    counts = np.zeros(
        (len(PATH_GEOMETRY_CLASSES), len(PATH_GEOMETRY_CLASSES)), dtype=float
    )
    np.add.at(counts, (actual_index, predicted_index), 1.0)
    row_normalized = np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(counts),
        where=counts.sum(axis=1, keepdims=True) > 0.0,
    )
    penalty = np.abs(prior_values[:, None] - prior_values[None, :])
    weighted = counts * penalty
    mean_contribution = weighted / len(truth)

    def matrix_rows(name: str, values: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "matrix_type": name,
                "true_class": true_class,
                **{
                    f"predicted_{predicted_class}": float(values[row, column])
                    for column, predicted_class in enumerate(PATH_GEOMETRY_CLASSES)
                },
            }
            for row, true_class in enumerate(PATH_GEOMETRY_CLASSES)
        ]

    return {
        "metrics": {
            "economic_confusion_total_weighted_cost": float(weighted.sum()),
            "economic_confusion_cost": float(mean_contribution.sum()),
            "economic_confusion_error_rate": float(
                np.mean(actual_index != predicted_index)
            ),
        },
        "matrix": pd.DataFrame(
            matrix_rows("count", counts)
            + matrix_rows("row_normalized", row_normalized)
            + matrix_rows("penalty", penalty)
            + matrix_rows("weighted_cost_contribution", weighted)
            + matrix_rows("mean_cost_contribution", mean_contribution)
        ),
        "class_ev_priors": pd.DataFrame(prior_rows),
        "provenance": {
            "penalty": "absolute_difference_between_train_only_class_reference_geometry_net_ev_priors",
            "class_ev_source": "train_label_net_ev_after_fixed_1pct_cost",
            "class_ev_semantics": (
                "fixed_12h_reference_geometry_proxy; replace with train_only_class_optimized_execution_ev "
                "after execution_policy_optimization"
            ),
            "outer_oos_labels_used_for_priors": False,
            "fallback": "global_train_mean_only_when_a_class_has_no_train_support",
        },
    }


@dataclass(frozen=True)
class ChronologicalFold:
    fold_id: int
    train_indices: np.ndarray
    oos_indices: np.ndarray
    train_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def four_month_walk_forward_folds(
    timestamps: Sequence[object],
    *,
    label_end: Sequence[object] | None = None,
) -> tuple[ChronologicalFold, ...]:
    """Build non-overlapping 4-month train -> 4-month OOS folds."""
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    end = (
        ts
        if label_end is None
        else pd.to_datetime(pd.Series(label_end), utc=True, errors="coerce")
    )
    if (
        ts.isna().any()
        or end.isna().any()
        or not ts.is_monotonic_increasing
        or (end < ts).any()
    ):
        raise ValueError(
            "timestamps/label_end must be sorted valid UTC with non-overlapping direction"
        )
    cursor, last, fold_id, folds = ts.iloc[0], ts.iloc[-1], 0, []
    while True:
        train_end = cursor + pd.DateOffset(months=GEOMETRY_TRAIN_MONTHS)
        oos_end = train_end + pd.DateOffset(months=GEOMETRY_OOS_MONTHS)
        if oos_end > last + pd.Timedelta(nanoseconds=1):
            break
        train = np.flatnonzero(
            (ts >= cursor).to_numpy()
            & (ts < train_end).to_numpy()
            & (end < train_end).to_numpy()
        )
        oos = np.flatnonzero((ts >= train_end).to_numpy() & (ts < oos_end).to_numpy())
        if len(train) and len(oos):
            folds.append(
                ChronologicalFold(fold_id, train, oos, train_end, train_end, oos_end)
            )
            fold_id += 1
        cursor += pd.DateOffset(months=GEOMETRY_TRAIN_MONTHS)
    if not folds:
        raise ValueError(
            "need at least 8 calendar months for a 4m train -> 4m OOS fold"
        )
    return tuple(folds)


def fixed_four_month_ablation_fold(
    timestamps: Sequence[object],
    start_date: str | pd.Timestamp,
    *,
    label_end: Sequence[object] | None = None,
) -> ChronologicalFold:
    """Build one explicit 4-month train then next 4-month OOS ablation split."""
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    end = (
        ts
        if label_end is None
        else pd.to_datetime(pd.Series(label_end), utc=True, errors="coerce")
    )
    start = pd.Timestamp(start_date)
    start = (
        start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    )
    train_end = start + pd.DateOffset(months=GEOMETRY_TRAIN_MONTHS)
    oos_end = train_end + pd.DateOffset(months=GEOMETRY_OOS_MONTHS)
    train = np.flatnonzero(
        (ts >= start).to_numpy()
        & (ts < train_end).to_numpy()
        & (end < train_end).to_numpy()
    )
    oos = np.flatnonzero((ts >= train_end).to_numpy() & (ts < oos_end).to_numpy())
    if not len(train) or not len(oos):
        raise ValueError(
            "requested ablation start does not contain a complete 4m train -> 4m OOS split"
        )
    return ChronologicalFold(0, train, oos, train_end, train_end, oos_end)


def short_history_purged_chronological_folds(
    timestamps: Sequence[object],
    *,
    label_end: Sequence[object],
    development_end: str | pd.Timestamp,
    subfold_count: int = 2,
    embargo: pd.Timedelta = pd.Timedelta(hours=24),
) -> tuple[ChronologicalFold, ...]:
    """Build bounded purged subfolds entirely inside a frozen short history.

    This is deliberately not a shorter version of the 4m/4m validation rule.
    It is an explicitly named development-only mode for the April v9 labels:
    callers must remove every row whose resolved label reaches the fixed May
    boundary before invoking it.  The resulting folds are expanding,
    chronology-preserving and purge both unresolved labels and the embargo.
    """
    if subfold_count < 1:
        raise ValueError("short-history geometry requires at least one subfold")
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    end = pd.to_datetime(pd.Series(label_end), utc=True, errors="coerce")
    cutoff = pd.Timestamp(development_end)
    cutoff = (
        cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    )
    if (
        ts.isna().any()
        or end.isna().any()
        or not ts.is_monotonic_increasing
        or (end < ts).any()
    ):
        raise ValueError(
            "short-history folds require sorted valid UTC timestamps/label ends"
        )
    if not bool((end < cutoff).all()) or not bool((ts < cutoff).all()):
        raise ValueError(
            "short-history geometry received rows outside its frozen development boundary"
        )
    if len(ts) < subfold_count + 2:
        raise ValueError(
            "too few development rows for requested short-history subfolds"
        )
    chunks = np.array_split(np.arange(len(ts), dtype=np.int64), subfold_count + 1)
    folds: list[ChronologicalFold] = []
    for fold_id, valid in enumerate(chunks[1:]):
        if not len(valid):
            continue
        validation_start = ts.iloc[int(valid[0])]
        prior = np.arange(int(valid[0]), dtype=np.int64)
        train = prior[
            (end.iloc[prior] < validation_start).to_numpy()
            & (ts.iloc[prior] < validation_start - embargo).to_numpy()
        ]
        if not len(train):
            continue
        validation_end = ts.iloc[int(valid[-1])] + pd.Timedelta(nanoseconds=1)
        folds.append(
            ChronologicalFold(
                fold_id,
                train,
                valid,
                validation_start,
                validation_start,
                validation_end,
            )
        )
    if not folds:
        raise ValueError("purge/embargo leaves no short-history geometry subfold")
    return tuple(folds)


@dataclass(frozen=True)
class GeometryPredictorContext:
    """Fold-local, purged data used only to select CatBoost tree count."""

    fold_id: int
    sampled_train_positions: np.ndarray
    early_stop_fit_indices: np.ndarray
    early_stop_validation_indices: np.ndarray
    early_stop_fit_positions: np.ndarray
    early_stop_validation_positions: np.ndarray
    validation_start: pd.Timestamp
    embargo: pd.Timedelta

    def audit(self) -> dict[str, Any]:
        return {
            "early_stopping_contract": (
                "chronological_tail_validation_inside_sampled_outer_train; "
                "purged_label_end_and_24h_embargo; outer_oos_never_used"
            ),
            "early_stop_validation_start_utc": self.validation_start.isoformat(),
            "early_stop_embargo_hours": float(self.embargo / pd.Timedelta(hours=1)),
            "early_stop_fit_rows": int(len(self.early_stop_fit_indices)),
            "early_stop_validation_rows": int(len(self.early_stop_validation_indices)),
            "early_stop_fit_source_rows": int(len(self.early_stop_fit_positions)),
            "early_stop_validation_source_rows": int(
                len(self.early_stop_validation_positions)
            ),
        }


PredictionOutput = tuple[np.ndarray, Sequence[str], Mapping[str, Any]]
Predictor = Callable[
    [
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        Mapping[str, Any],
        GeometryPredictorContext,
    ],
    PredictionOutput,
]
ProgressReporter = Callable[[str, Mapping[str, Any]], None]

_CHECKPOINT_SCHEMA = "catboost_path_archetype_geometry_search_checkpoint_v1"
_FINALIST_SIDECAR_SCHEMA = "path_archetype_geometry_checkpoint_finalist_predictions_v1"
_EXACT_GEOMETRY_EXPORT_SCHEMA = (
    "catboost_path_archetype_exact_geometry_raw_oos_export_v1"
)
_EXACT_GEOMETRY_SIDECAR_SCHEMA = "path_archetype_exact_geometry_raw7_sidecar_v1"
_EXACT_GEOMETRY_CLASS_MERGE = {
    "merged_class": "fast_realization_winner",
    "source_classes": ["fast_clean_winner", "fast_winner_early_drawdown"],
}
_EXACT_GEOMETRY_LABEL_REMAP = {
    source: _EXACT_GEOMETRY_CLASS_MERGE["merged_class"]
    for source in _EXACT_GEOMETRY_CLASS_MERGE["source_classes"]
}
EXACT_GEOMETRY_EXPORT_CLASSES: tuple[str, ...] = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
    "noisy_timeout_usable_mfe",
    "dead_timeout",
)
_EXACT_GEOMETRY_ADVERSE_CLASSES = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "dead_timeout",
)
_EXACT_GEOMETRY_FAVORABLE_CLASSES = (
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
)
_EXACT_GEOMETRY_NEUTRAL_CLASSES = ("noisy_timeout_usable_mfe",)
_FINALIST_DIAGNOSTIC_KEYS = (
    "folds",
    "probability_reliability_bins",
    "economic_confusion",
    "economic_confusion_priors",
    "side_diagnostics",
    "month_diagnostics",
)
_CHECKPOINT_TABULAR_KEYS = (
    "folds",
    "boundary",
    "side_support",
    "symbol_support",
    "side_stability",
    "symbol_stability",
    "temporal_month_stability",
    "side_diagnostics",
    "symbol_diagnostics",
    "month_diagnostics",
    "probability_reliability_bins",
    "economic_confusion",
    "economic_confusion_priors",
    "oos_predictions",
)
_FINALIST_SIDECAR_REQUIRED_COLUMNS = {
    "source_row_position",
    "__ts__",
    "__symbol__",
    "side",
    "candidate_id",
    "true_dynamic_label",
    "predicted_class",
    "probability_vector",
    "probability_entropy",
    "fold_id",
    "train_cutoff_utc",
    "available_at",
    "validation_start",
    "latest_train_decision_ts",
    "label_resolution_available_at",
    "train_decision_cutoff",
    "oos_start_utc",
    "oos_end_utc",
    "config_id",
}.union({f"probability_{name}" for name in PATH_GEOMETRY_CLASSES})
_FINALIST_SIDECAR_IDENTITY_COLUMNS = (
    "source_row_position",
    "candidate_id",
    "__ts__",
    "__symbol__",
    "side",
    "fold_id",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_json_ready(row) for row in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        scalar = value.item()
        return scalar if not isinstance(scalar, float) or np.isfinite(scalar) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def _checkpoint_fingerprint(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(contract), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one checkpoint so interrupted runs never expose partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        _json_ready(payload), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_identity_sha256(frame: pd.DataFrame) -> str:
    identity = frame.loc[:, list(_FINALIST_SIDECAR_IDENTITY_COLUMNS)].copy()
    for column in ("__ts__",):
        identity[column] = pd.to_datetime(
            identity[column], utc=True, errors="raise"
        ).astype(str)
    hashed = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _checkpoint_sidecar_directory(checkpoint_path: Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}_sidecars")


def _validate_finalist_sidecar_frame(
    frame: pd.DataFrame, metadata: Mapping[str, Any]
) -> None:
    if metadata.get("columns") != list(frame.columns):
        raise ValueError(
            "geometry finalist checkpoint sidecar column schema does not match its manifest"
        )
    missing = sorted(_FINALIST_SIDECAR_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            f"geometry finalist checkpoint sidecar has invalid schema; missing={missing}"
        )
    config_id = str(metadata.get("config_id", ""))
    if not config_id or set(frame["config_id"].astype(str)) != {config_id}:
        raise ValueError(
            "geometry finalist checkpoint sidecar has a mismatched config_id"
        )
    if frame.duplicated(list(_FINALIST_SIDECAR_IDENTITY_COLUMNS)).any():
        raise ValueError(
            "geometry finalist checkpoint sidecar has duplicate identities"
        )
    if int(metadata.get("rows", -1)) != len(frame):
        raise ValueError(
            "geometry finalist checkpoint sidecar row count does not match its manifest"
        )
    if metadata.get("identity_sha256") != _prediction_identity_sha256(frame):
        raise ValueError(
            "geometry finalist checkpoint sidecar identity hash does not match its manifest"
        )


def _atomic_write_finalist_sidecar(
    checkpoint_path: Path,
    config_id: str,
    predictions: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    diagnostics: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    directory = _checkpoint_sidecar_directory(checkpoint_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config_id}.parquet"
    temporary = directory / f".{config_id}.{os.getpid()}.tmp.parquet"
    predictions.to_parquet(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    metadata = {
        "schema": _FINALIST_SIDECAR_SCHEMA,
        "sidecar_path": str(path),
        "sidecar_sha256": _file_sha256(path),
        "rows": int(len(predictions)),
        "columns": list(predictions.columns),
        "identity_columns": list(_FINALIST_SIDECAR_IDENTITY_COLUMNS),
        "identity_sha256": _prediction_identity_sha256(predictions),
        "config_id": config_id,
        "config": _json_ready(config),
        "summary": _json_ready(summary),
        "diagnostics": _json_ready(diagnostics),
    }
    _validate_finalist_sidecar_frame(predictions, metadata)
    return metadata


def _load_finalist_sidecar(
    metadata: Mapping[str, Any], checkpoint_path: Path
) -> pd.DataFrame:
    if metadata.get("schema") != _FINALIST_SIDECAR_SCHEMA:
        raise ValueError(
            "geometry finalist checkpoint sidecar has an unsupported schema"
        )
    path = Path(str(metadata.get("sidecar_path", "")))
    expected_directory = _checkpoint_sidecar_directory(checkpoint_path).resolve()
    if path.parent.resolve() != expected_directory:
        raise ValueError(
            "geometry finalist checkpoint sidecar is not in the checkpoint sibling directory"
        )
    if not path.is_file():
        raise ValueError(f"geometry finalist checkpoint sidecar does not exist: {path}")
    if metadata.get("sidecar_sha256") != _file_sha256(path):
        raise ValueError(
            "geometry finalist checkpoint sidecar checksum does not match its manifest"
        )
    frame = pd.read_parquet(path)
    _validate_finalist_sidecar_frame(frame, metadata)
    return frame


def _frame_identity(frame: pd.DataFrame) -> str:
    """Stable content identity for direct API callers that do not supply one."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8")
    )
    for column in frame.columns:
        digest.update(
            pd.util.hash_pandas_object(frame[column], index=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        )
    return digest.hexdigest()


def _fold_contract(folds: Sequence[ChronologicalFold]) -> list[dict[str, Any]]:
    return [
        {
            "fold_id": int(fold.fold_id),
            "train_rows": int(len(fold.train_indices)),
            "oos_rows": int(len(fold.oos_indices)),
            "train_end": pd.Timestamp(fold.train_end).isoformat(),
            "oos_start": pd.Timestamp(fold.oos_start).isoformat(),
            "oos_end": pd.Timestamp(fold.oos_end).isoformat(),
            "train_indices_sha256": hashlib.sha256(
                np.asarray(fold.train_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "oos_indices_sha256": hashlib.sha256(
                np.asarray(fold.oos_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
        }
        for fold in folds
    ]


def _checkpoint_result(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in result.items()}
    return _json_ready(payload)


def _restore_checkpoint_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in _CHECKPOINT_TABULAR_KEYS:
        if key in result:
            result[key] = pd.DataFrame(result[key])
    return result


def _validate_uniform_pre_refinement_weights(params: Mapping[str, Any]) -> None:
    """Reject weighting modes outside the original hard-label geometry contract."""
    if params.get("auto_class_weights") not in (None, "", "None"):
        raise ValueError(
            "exact geometry export requires uniform weights; auto_class_weights is not allowed"
        )
    if "scale_pos_weight" in params and not np.isclose(
        float(params["scale_pos_weight"]), 1.0
    ):
        raise ValueError(
            "exact geometry export requires uniform weights; scale_pos_weight must be 1"
        )
    class_weights = params.get("class_weights")
    if class_weights is None:
        return
    values = (
        list(class_weights.values())
        if isinstance(class_weights, Mapping)
        else list(class_weights)
    )
    if not values or not np.allclose(np.asarray(values, dtype=float), float(values[0])):
        raise ValueError("exact geometry export requires uniform class_weights")


def _checkpoint_exact_geometry_contract(
    checkpoint: Mapping[str, Any],
    *,
    ordered: pd.DataFrame,
    feature_columns: Sequence[str],
    effective_model_params: Mapping[str, Any],
    columns: PathGeometryColumns,
    max_train_rows_per_fold: int,
    checkpoint_input_identity: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], tuple[ChronologicalFold, ...]]:
    if checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError("geometry-search checkpoint has an unsupported schema")
    contract = checkpoint.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("geometry-search checkpoint has no contract")
    if checkpoint.get("fingerprint") != _checkpoint_fingerprint(contract):
        raise ValueError(
            "geometry-search checkpoint fingerprint does not match its contract"
        )
    expected_identity = dict(
        checkpoint_input_identity or {"prepared_frame_sha256": _frame_identity(ordered)}
    )
    expected_folds = (
        (
            fixed_four_month_ablation_fold(
                ordered[columns.timestamp],
                contract.get("ablation_start_date"),
                label_end=ordered.get(columns.label_end),
            ),
        )
        if contract.get("ablation_start_date") is not None
        else four_month_walk_forward_folds(
            ordered[columns.timestamp], label_end=ordered.get(columns.label_end)
        )
    )
    expected = {
        "input_identity": expected_identity,
        "feature_columns": list(feature_columns),
        "effective_model_params": dict(effective_model_params),
        "columns": asdict(columns),
        "selection_folds": _fold_contract(expected_folds),
        "max_train_rows_per_fold": int(max_train_rows_per_fold),
    }
    changed = sorted(
        name
        for name, value in expected.items()
        if contract.get(name) != _json_ready(value)
    )
    if changed:
        raise ValueError(
            "geometry-search checkpoint fingerprint matches its stored contract but "
            f"not the exact export inputs; changed keys={changed}"
        )
    return contract, tuple(expected_folds)


def _merge_exact_geometry_labels(labels: pd.Series) -> pd.Series:
    merged = labels.astype("string").replace(_EXACT_GEOMETRY_LABEL_REMAP)
    unknown = sorted(set(merged.dropna()).difference(EXACT_GEOMETRY_EXPORT_CLASSES))
    if unknown:
        raise ValueError(
            f"exact geometry export encountered unknown path labels: {unknown}"
        )
    return merged


def _require_merged_seven_class_refit_support(
    frame: pd.DataFrame,
    config: PathGeometryConfig,
    folds: Sequence[ChronologicalFold],
    *,
    columns: PathGeometryColumns,
    max_train_rows_per_fold: int,
) -> None:
    labels = _merge_exact_geometry_labels(
        label_path_geometry(frame, config, columns)["path_geometry_label"]
    )
    required = set(EXACT_GEOMETRY_EXPORT_CLASSES)
    viable_folds = 0
    for fold in folds:
        train_raw = labels.iloc[fold.train_indices]
        test_raw = labels.iloc[fold.oos_indices]
        train_positions = fold.train_indices[train_raw.notna().to_numpy()]
        train_y = train_raw.dropna().astype(str)
        test_y = test_raw.dropna().astype(str)
        if len(train_y) < 2 or len(test_y) < 1 or train_y.nunique() < 2:
            continue
        sampled_positions = bounded_chronological_training_positions(
            frame,
            train_positions,
            train_y,
            max_rows=max_train_rows_per_fold,
            columns=columns,
        )
        sampled_classes = set(labels.iloc[sampled_positions].astype(str))
        missing = sorted(required.difference(sampled_classes))
        if missing:
            raise ValueError(
                "exact geometry export requires all seven merged hard path classes in every "
                f"refit fold; fold_id={fold.fold_id}, missing={missing}"
            )
        viable_folds += 1
    if not viable_folds:
        raise ValueError("exact geometry export has no viable OOS fold")


def _raw_exact_oos_prediction_frame(
    frame: pd.DataFrame,
    positions: np.ndarray,
    true_labels: pd.Series,
    probabilities: np.ndarray,
    config: PathGeometryConfig,
    fold: ChronologicalFold,
    columns: PathGeometryColumns,
) -> pd.DataFrame:
    values = _normalise_probability_rows(np.asarray(probabilities, dtype=np.float64))
    expected_shape = (len(positions), len(EXACT_GEOMETRY_EXPORT_CLASSES))
    if values.shape != expected_shape:
        raise ValueError(
            "exact geometry export has an invalid raw seven-class probability shape"
        )
    source = frame.iloc[positions]
    timestamp = pd.to_datetime(source[columns.timestamp], utc=True, errors="coerce")
    if timestamp.isna().any() or "candidate_id" not in source:
        raise ValueError(
            "exact geometry export requires UTC timestamps and candidate_id"
        )
    train_timestamp = pd.to_datetime(
        frame.iloc[fold.train_indices][columns.timestamp], utc=True, errors="raise"
    )
    train_resolution = (
        train_timestamp
        if columns.label_end is None or columns.label_end not in frame
        else pd.to_datetime(
            frame.iloc[fold.train_indices][columns.label_end], utc=True, errors="raise"
        )
    )
    latest_train_decision = train_timestamp.max()
    latest_train_resolution = train_resolution.max()
    train_cutoff = max(latest_train_decision, latest_train_resolution)
    if not train_cutoff < fold.oos_start:
        raise ValueError("exact geometry export training information reaches OOS")
    entropy = -np.sum(values * np.log(np.clip(values, 1e-12, 1.0)), axis=1)
    predicted = values.argmax(axis=1)
    class_index = {
        name: index for index, name in enumerate(EXACT_GEOMETRY_EXPORT_CLASSES)
    }
    sorted_probabilities = np.sort(values, axis=1)
    result = pd.DataFrame(
        {
            "source_row_position": positions.astype(np.int64),
            "__ts__": timestamp.to_numpy(),
            "__symbol__": source[columns.symbol].astype(str).to_numpy(),
            "side": source[columns.side].astype(str).to_numpy(),
            "candidate_id": source["candidate_id"].astype(str).to_numpy(),
            "true_merged_dynamic_label": true_labels.astype(str).to_numpy(),
            "predicted_class": [
                EXACT_GEOMETRY_EXPORT_CLASSES[index] for index in predicted
            ],
            "probability_vector": [row.tolist() for row in values],
            "probability_entropy": entropy,
            "max_probability": values.max(axis=1),
            "raw_max_probability": values.max(axis=1),
            "normalized_entropy": entropy / np.log(len(EXACT_GEOMETRY_EXPORT_CLASSES)),
            "raw_normalized_entropy": entropy
            / np.log(len(EXACT_GEOMETRY_EXPORT_CLASSES)),
            "top2_probability_margin": sorted_probabilities[:, -1]
            - sorted_probabilities[:, -2],
            "raw_top1_top2_probability_margin": sorted_probabilities[:, -1]
            - sorted_probabilities[:, -2],
            "adverse_probability_mass": values[
                :, [class_index[name] for name in _EXACT_GEOMETRY_ADVERSE_CLASSES]
            ].sum(axis=1),
            "raw_adverse_probability_mass": values[
                :, [class_index[name] for name in _EXACT_GEOMETRY_ADVERSE_CLASSES]
            ].sum(axis=1),
            "favorable_probability_mass": values[
                :, [class_index[name] for name in _EXACT_GEOMETRY_FAVORABLE_CLASSES]
            ].sum(axis=1),
            "raw_favorable_probability_mass": values[
                :, [class_index[name] for name in _EXACT_GEOMETRY_FAVORABLE_CLASSES]
            ].sum(axis=1),
            "fold_id": int(fold.fold_id),
            "train_cutoff_utc": train_cutoff,
            "available_at": timestamp.to_numpy(),
            "validation_start": pd.Timestamp(fold.oos_start),
            "latest_train_decision_ts": latest_train_decision,
            "label_resolution_available_at": latest_train_resolution,
            "train_decision_cutoff": train_cutoff,
            "oos_start_utc": pd.Timestamp(fold.oos_start),
            "oos_end_utc": pd.Timestamp(fold.oos_end),
            "config_id": geometry_config_id(config),
        }
    )
    for index, class_name in enumerate(EXACT_GEOMETRY_EXPORT_CLASSES):
        result[f"probability_{class_name}"] = values[:, index]
    return result


def _exact_geometry_export_fingerprint(
    checkpoint_fingerprint: str, config_id: str, config: Mapping[str, Any]
) -> str:
    return _checkpoint_fingerprint(
        {
            "schema": _EXACT_GEOMETRY_EXPORT_SCHEMA,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "config_id": config_id,
            "config": config,
            "class_merge": _EXACT_GEOMETRY_CLASS_MERGE,
            "class_order": EXACT_GEOMETRY_EXPORT_CLASSES,
            "sample_weight_contract": "uniform_weights_v1",
            "probability_output": "raw_catboost_predict_proba",
            "raw_scoring_contract": {
                "max_probability": "max(all_7_raw_probabilities)",
                "normalized_entropy": "-sum(p_i * log(p_i)) / log(7)",
                "top2_probability_margin": "largest_raw_probability - second_largest_raw_probability",
                "adverse_probability_mass": "sum(immediate_adverse_path, early_mfe_full_reversal, dead_timeout)",
                "favorable_probability_mass": "sum(fast_realization_winner, late_breakout, slow_grinder)",
                "neutral_classes": _EXACT_GEOMETRY_NEUTRAL_CLASSES,
            },
        }
    )


def _atomic_write_exact_geometry_sidecar(
    checkpoint_path: Path,
    config_id: str,
    predictions: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    export_fingerprint: str,
) -> dict[str, Any]:
    directory = _checkpoint_sidecar_directory(checkpoint_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config_id}.raw7.parquet"
    temporary = directory / f".{config_id}.{os.getpid()}.raw7.tmp.parquet"
    predictions.to_parquet(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    metadata = {
        "schema": _EXACT_GEOMETRY_SIDECAR_SCHEMA,
        "sidecar_path": str(path),
        "sidecar_sha256": _file_sha256(path),
        "rows": int(len(predictions)),
        "columns": list(predictions.columns),
        "identity_columns": list(_FINALIST_SIDECAR_IDENTITY_COLUMNS),
        "identity_sha256": _prediction_identity_sha256(predictions),
        "config_id": config_id,
        "config": _json_ready(config),
        "summary": _json_ready(summary),
        "export_fingerprint": export_fingerprint,
    }
    required = {
        "true_merged_dynamic_label",
        "probability_vector",
        "config_id",
        "raw_max_probability",
        "raw_normalized_entropy",
        "raw_top1_top2_probability_margin",
        "raw_adverse_probability_mass",
        "raw_favorable_probability_mass",
        *{f"probability_{name}" for name in EXACT_GEOMETRY_EXPORT_CLASSES},
    }
    if required.difference(predictions.columns):
        raise ValueError(
            "exact geometry sidecar lacks the required raw seven-class fields"
        )
    if metadata["identity_sha256"] != _prediction_identity_sha256(predictions):
        raise ValueError("exact geometry sidecar identity hash is invalid")
    return metadata


def _load_exact_geometry_sidecar(
    metadata: Mapping[str, Any], checkpoint_path: Path, export_fingerprint: str
) -> pd.DataFrame:
    if metadata.get("schema") != _EXACT_GEOMETRY_SIDECAR_SCHEMA:
        raise ValueError(
            "exact geometry checkpoint export has an unsupported sidecar schema"
        )
    if metadata.get("export_fingerprint") != export_fingerprint:
        raise ValueError("exact geometry checkpoint export fingerprint does not match")
    path = Path(str(metadata.get("sidecar_path", "")))
    if (
        path.parent.resolve()
        != _checkpoint_sidecar_directory(checkpoint_path).resolve()
    ):
        raise ValueError(
            "exact geometry sidecar is not in the checkpoint sibling directory"
        )
    if not path.is_file() or metadata.get("sidecar_sha256") != _file_sha256(path):
        raise ValueError(
            "exact geometry sidecar is missing or has a mismatched checksum"
        )
    frame = pd.read_parquet(path)
    if metadata.get("columns") != list(frame.columns):
        raise ValueError("exact geometry sidecar has a mismatched column schema")
    if metadata.get("identity_sha256") != _prediction_identity_sha256(frame):
        raise ValueError("exact geometry sidecar has a mismatched identity hash")
    return frame


def _persist_final_exact_geometry_model(
    frame: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    params: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    config_id: str,
    export_fingerprint: str,
    columns: PathGeometryColumns,
) -> dict[str, Any]:
    """Fit, persist, reload, and verify the non-OOS merged-seven-class model."""
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "CatBoost is required to persist the final exact geometry model"
        ) from exc
    _validate_uniform_pre_refinement_weights(params)
    positions = np.flatnonzero(labels.notna().to_numpy()).astype(np.int64)
    target = labels.iloc[positions].astype(str)
    if set(target) != set(EXACT_GEOMETRY_EXPORT_CLASSES):
        missing = sorted(set(EXACT_GEOMETRY_EXPORT_CLASSES).difference(target))
        raise ValueError(
            f"final exact geometry refit is missing merged hard classes: {missing}"
        )
    context = _early_stop_context(frame, positions, target, fold_id=-1, columns=columns)
    capped_params, resource_contract = capped_catboost_params(params)
    early_fit_y = target.iloc[context.early_stop_fit_indices]
    early_validation_y = target.iloc[context.early_stop_validation_indices]
    early_names = tuple(sorted(early_fit_y.unique()))
    if not set(early_validation_y).issubset(early_names):
        raise ValueError(
            "final exact geometry early-stop validation has unseen classes"
        )
    constructor_params = {
        "loss_function": "MultiClass",
        "verbose": False,
        "random_seed": 20260722,
        "allow_writing_files": False,
        **capped_params,
    }
    early_model = CatBoostClassifier(
        **dict(constructor_params, classes_count=len(early_names))
    )
    early_model.fit(
        feature_matrix.iloc[positions].iloc[context.early_stop_fit_indices],
        pd.Categorical(early_fit_y, categories=early_names).codes,
        eval_set=(
            feature_matrix.iloc[positions].iloc[context.early_stop_validation_indices],
            pd.Categorical(early_validation_y, categories=early_names).codes,
        ),
        early_stopping_rounds=int(capped_params.get("od_wait", 150)),
        use_best_model=True,
        verbose=False,
    )
    best_iteration = (
        early_model.get_best_iteration()
        if hasattr(early_model, "get_best_iteration")
        else None
    )
    tree_count = getattr(early_model, "tree_count_", None)
    iteration_ceiling = int(capped_params.get("iterations", 3000))
    effective_trees = (
        int(best_iteration) + 1
        if best_iteration is not None and int(best_iteration) >= 0
        else int(tree_count or 0)
    )
    if effective_trees <= 0:
        raise ValueError(
            "final exact geometry early-stop fit did not expose a usable tree count"
        )
    effective_trees = min(max(1, effective_trees), iteration_ceiling)
    model = CatBoostClassifier(
        **dict(
            constructor_params,
            iterations=effective_trees,
            classes_count=len(EXACT_GEOMETRY_EXPORT_CLASSES),
        ),
    )
    model.fit(
        feature_matrix.iloc[positions],
        pd.Categorical(target, categories=EXACT_GEOMETRY_EXPORT_CLASSES).codes,
        verbose=False,
    )
    directory = _checkpoint_sidecar_directory(checkpoint_path) / "final_models"
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{config_id}.cbm"
    temporary_model_path = directory / f".{config_id}.{os.getpid()}.tmp.cbm"
    model.save_model(str(temporary_model_path))
    os.replace(temporary_model_path, model_path)
    loaded = CatBoostClassifier()
    loaded.load_model(str(model_path))
    verification_probabilities = np.asarray(
        loaded.predict_proba(feature_matrix.iloc[positions[:1]]), dtype=np.float64
    )
    if verification_probabilities.shape != (1, len(EXACT_GEOMETRY_EXPORT_CLASSES)):
        raise ValueError(
            "reloaded final exact geometry model has an invalid probability shape"
        )
    if not np.isfinite(verification_probabilities).all() or not np.allclose(
        verification_probabilities.sum(axis=1), 1.0
    ):
        raise ValueError(
            "reloaded final exact geometry model has invalid raw probabilities"
        )
    scoring_contract = {
        "max_probability": "max(all_7_raw_probabilities)",
        "normalized_entropy": "-sum(p_i * log(p_i)) / log(7)",
        "top2_probability_margin": "largest_raw_probability - second_largest_raw_probability",
        "adverse_probability_mass": "sum(immediate_adverse_path, early_mfe_full_reversal, dead_timeout)",
        "favorable_probability_mass": "sum(fast_realization_winner, late_breakout, slow_grinder)",
        "class_groups": {
            "adverse": list(_EXACT_GEOMETRY_ADVERSE_CLASSES),
            "favorable": list(_EXACT_GEOMETRY_FAVORABLE_CLASSES),
            "neutral": list(_EXACT_GEOMETRY_NEUTRAL_CLASSES),
        },
    }
    model_manifest = {
        "schema": "catboost_path_archetype_exact_geometry_final_model_v1",
        "export_fingerprint": export_fingerprint,
        "config_id": config_id,
        "model_path": str(model_path),
        "model_sha256": _file_sha256(model_path),
        "feature_columns": list(feature_matrix.columns),
        "hard_label_target": "seven_class_path_geometry",
        "class_order": list(EXACT_GEOMETRY_EXPORT_CLASSES),
        "class_merge": dict(_EXACT_GEOMETRY_CLASS_MERGE),
        "sample_weight_contract": "uniform_weights_v1",
        "probability_output": "raw_catboost_predict_proba",
        "raw_scoring_contract": scoring_contract,
        "final_refit": {
            "rows": int(len(target)),
            "tree_count": effective_trees,
            "early_stop_contract": context.audit(),
            "contract": "early_stop_on_internal_purged_chronological_tail_then_fixed_tree_refit_on_all_eligible_rows",
            "oos_metrics_inclusion": "excluded_final_refit_not_used_in_causal_4m_4m_oos_metrics",
        },
        "load_predict_verification": {
            "status": "passed",
            "rows": 1,
            "probability_shape": [1, len(EXACT_GEOMETRY_EXPORT_CLASSES)],
            "probability_row_sum": float(verification_probabilities.sum()),
        },
        "catboost_resource_contract": resource_contract,
    }
    manifest_path = directory / f"{config_id}.model_manifest.json"
    _atomic_json_write(manifest_path, model_manifest)
    return {
        "model_path": str(model_path),
        "model_sha256": _file_sha256(model_path),
        "model_manifest_path": str(manifest_path),
        "model_manifest_sha256": _file_sha256(manifest_path),
        "load_predict_verification": model_manifest["load_predict_verification"],
        "final_refit": model_manifest["final_refit"],
    }


def export_checkpoint_geometry(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    model_params: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    config_id: str,
    columns: PathGeometryColumns = PathGeometryColumns(),
    predictor: Predictor | None = None,
    max_train_rows_per_fold: int = DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
    checkpoint_input_identity: Mapping[str, Any] | None = None,
    progress_reporter: ProgressReporter | None = None,
    persist_final_model: bool = True,
) -> dict[str, Any]:
    """Capture raw seven-class OOS predictions for one checkpointed geometry.

    This deliberately does not call :func:`staged_geometry_search`: it validates
    the completed sweep's exact contract, retains its geometry thresholds, then
    refits only ``config_id`` on its frozen selection folds. The two fast path
    labels are merged before sampling and fitting; only uniform-weight raw
    probabilities are fit and exported.
    """
    if max_train_rows_per_fold < 0:
        raise ValueError("max_train_rows_per_fold must be non-negative")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"geometry-search checkpoint does not exist: {checkpoint_path}"
        )
    with checkpoint_path.open() as handle:
        checkpoint = json.load(handle)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("geometry-search checkpoint must contain an object")
    effective_model_params, catboost_resource = capped_catboost_params(model_params)
    _validate_uniform_pre_refinement_weights(effective_model_params)
    ordered = ensure_risk_fraction(frame, columns)
    ordered[columns.timestamp] = pd.to_datetime(
        ordered[columns.timestamp], utc=True, errors="coerce"
    )
    if ordered[columns.timestamp].isna().any():
        raise ValueError("exact geometry export requires valid UTC timestamps")
    ordered = ordered.sort_values(columns.timestamp, kind="mergesort").reset_index(
        drop=True
    )
    prepared_feature_matrix = _feature_matrix(ordered, feature_columns)
    contract, selection_folds = _checkpoint_exact_geometry_contract(
        checkpoint,
        ordered=ordered,
        feature_columns=feature_columns,
        effective_model_params=effective_model_params,
        columns=columns,
        max_train_rows_per_fold=max_train_rows_per_fold,
        checkpoint_input_identity=checkpoint_input_identity,
    )
    completed = checkpoint.get("completed_configs")
    if not isinstance(completed, Mapping) or config_id not in completed:
        raise ValueError(f"geometry id does not exist in the checkpoint: {config_id}")
    completed_result = completed[config_id]
    if not isinstance(completed_result, Mapping) or not isinstance(
        completed_result.get("config"), Mapping
    ):
        raise ValueError(
            "geometry-search checkpoint has an invalid completed geometry result"
        )
    config = PathGeometryConfig(**dict(completed_result["config"]))
    if geometry_config_id(config) != config_id:
        raise ValueError(
            "geometry-search checkpoint geometry id does not match its config"
        )
    _require_merged_seven_class_refit_support(
        ordered,
        config,
        selection_folds,
        columns=columns,
        max_train_rows_per_fold=max_train_rows_per_fold,
    )
    stored_exports = checkpoint.get("exact_geometry_exports", {})
    if stored_exports and not isinstance(stored_exports, Mapping):
        raise ValueError(
            "geometry-search checkpoint has invalid exact_geometry_exports"
        )
    config_payload = _geometry_config_payload(config)
    export_fingerprint = _exact_geometry_export_fingerprint(
        str(checkpoint["fingerprint"]), config_id, config_payload
    )
    stored = dict(stored_exports).get(config_id)
    if stored is not None:
        if not isinstance(stored, Mapping) or stored.get("config") != _json_ready(
            config_payload
        ):
            raise ValueError("exact geometry checkpoint export has a mismatched config")
        predictions = _load_exact_geometry_sidecar(
            stored, checkpoint_path, export_fingerprint
        )
        reused_capture = True
    else:
        _report_progress(
            progress_reporter, "exact_geometry_capture_start", config_id=config_id
        )
        raw_predictor = catboost_predictor if predictor is None else predictor
        labels = _merge_exact_geometry_labels(
            label_path_geometry(ordered, config, columns)["path_geometry_label"]
        )
        prediction_frames: list[pd.DataFrame] = []
        fold_records: list[dict[str, Any]] = []
        for fold in selection_folds:
            train_raw = labels.iloc[fold.train_indices]
            test_raw = labels.iloc[fold.oos_indices]
            train_positions = fold.train_indices[train_raw.notna().to_numpy()]
            test_positions = fold.oos_indices[test_raw.notna().to_numpy()]
            train_y, test_y = (
                train_raw.dropna().astype(str),
                test_raw.dropna().astype(str),
            )
            if len(train_y) < 2 or len(test_y) < 1 or train_y.nunique() < 2:
                continue
            sampled_positions = bounded_chronological_training_positions(
                ordered,
                train_positions,
                train_y,
                max_rows=max_train_rows_per_fold,
                columns=columns,
            )
            sampled_y = labels.iloc[sampled_positions].astype(str)
            context = _early_stop_context(
                ordered,
                sampled_positions,
                sampled_y,
                fold_id=fold.fold_id,
                columns=columns,
            )
            probabilities, class_order, fit_report = raw_predictor(
                prepared_feature_matrix.iloc[sampled_positions],
                sampled_y,
                prepared_feature_matrix.iloc[test_positions],
                effective_model_params,
                context,
            )
            if tuple(class_order) != EXACT_GEOMETRY_EXPORT_CLASSES:
                raise ValueError(
                    "exact geometry export predictor must return the fixed merged seven-class order"
                )
            aligned = _normalise_probability_rows(
                np.asarray(probabilities, dtype=np.float64)
            )
            prediction_frames.append(
                _raw_exact_oos_prediction_frame(
                    ordered, test_positions, test_y, aligned, config, fold, columns
                )
            )
            fold_records.append(
                {
                    "fold_id": int(fold.fold_id),
                    "rows": int(len(test_y)),
                    "effective_train_rows": int(len(sampled_y)),
                    **dict(fit_report),
                }
            )
        if not prediction_frames:
            raise ValueError("exact geometry export has no viable OOS predictions")
        predictions = pd.concat(prediction_frames, ignore_index=True)
        summary = {
            "evaluated_folds": int(len(fold_records)),
            "evaluated_oos_rows": int(len(predictions)),
            "folds": fold_records,
        }
        metadata = _atomic_write_exact_geometry_sidecar(
            checkpoint_path,
            config_id,
            predictions,
            config=config_payload,
            summary=summary,
            export_fingerprint=export_fingerprint,
        )
        updated = dict(checkpoint)
        exports = dict(stored_exports)
        exports[config_id] = metadata
        updated["exact_geometry_exports"] = exports
        updated["last_checkpoint_reason"] = f"exact_geometry_export:{config_id}"
        _atomic_json_write(checkpoint_path, updated)
        _report_progress(
            progress_reporter, "exact_geometry_capture_complete", config_id=config_id
        )
        stored = metadata
        reused_capture = False
    final_model = (
        _persist_final_exact_geometry_model(
            ordered,
            prepared_feature_matrix,
            _merge_exact_geometry_labels(
                label_path_geometry(ordered, config, columns)["path_geometry_label"]
            ),
            effective_model_params,
            checkpoint_path=checkpoint_path,
            config_id=config_id,
            export_fingerprint=export_fingerprint,
            columns=columns,
        )
        if persist_final_model
        else {
            "status": "not_persisted_by_caller",
            "oos_metrics_inclusion": "excluded_final_refit_not_used_in_causal_4m_4m_oos_metrics",
        }
    )
    return {
        "schema": _EXACT_GEOMETRY_EXPORT_SCHEMA,
        "config_id": config_id,
        "config": config_payload,
        "summary": dict(stored["summary"]),
        "predictions": predictions,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_fingerprint": str(checkpoint["fingerprint"]),
        "checkpoint_contract": _json_ready(contract),
        "reused_checkpoint_capture": reused_capture,
        "catboost_resource_contract": catboost_resource,
        "export_fingerprint": export_fingerprint,
        "hard_label_target": "seven_class_path_geometry",
        "class_merge": dict(_EXACT_GEOMETRY_CLASS_MERGE),
        "class_order": list(EXACT_GEOMETRY_EXPORT_CLASSES),
        "sample_weight_contract": "uniform_weights_v1",
        "probability_output": "raw_catboost_predict_proba",
        "raw_scoring_contract": {
            "max_probability": "max(all_7_raw_probabilities)",
            "normalized_entropy": "-sum(p_i * log(p_i)) / log(7)",
            "top2_probability_margin": "largest_raw_probability - second_largest_raw_probability",
            "adverse_probability_mass": "sum(immediate_adverse_path, early_mfe_full_reversal, dead_timeout)",
            "favorable_probability_mass": "sum(fast_realization_winner, late_breakout, slow_grinder)",
            "class_groups": {
                "adverse": list(_EXACT_GEOMETRY_ADVERSE_CLASSES),
                "favorable": list(_EXACT_GEOMETRY_FAVORABLE_CLASSES),
                "neutral": list(_EXACT_GEOMETRY_NEUTRAL_CLASSES),
            },
        },
        "model_persistence": final_model,
    }


def _report_progress(
    reporter: ProgressReporter | None, event: str, **details: Any
) -> None:
    if reporter is not None:
        reporter(event, _json_ready(details))


def _early_stop_context(
    frame: pd.DataFrame,
    sampled_positions: np.ndarray,
    sampled_labels: pd.Series,
    *,
    fold_id: int,
    columns: PathGeometryColumns,
    validation_fraction: float = GEOMETRY_EARLY_STOP_VALIDATION_FRACTION,
    embargo: pd.Timedelta = GEOMETRY_EARLY_STOP_EMBARGO,
) -> GeometryPredictorContext:
    """Build a chronological, purged internal validation tail for one fold.

    The caller supplies only deterministic sampled outer-training rows.  The
    tail is never drawn from outer OOS; all sampled rows are restored only for
    the final fixed-tree refit after the early-stop tree count is selected.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "early-stop validation fraction must be strictly between zero and one"
        )
    if embargo < pd.Timedelta(0):
        raise ValueError("early-stop embargo must be non-negative")
    sampled_positions = np.asarray(sampled_positions, dtype=np.int64)
    if len(sampled_positions) != len(sampled_labels):
        raise ValueError(
            "sampled early-stop positions and labels must have equal length"
        )
    if len(sampled_positions) < 3:
        raise ValueError(
            "sampled training fold is too small for internal early stopping"
        )
    source = frame.iloc[sampled_positions]
    timestamps = pd.to_datetime(
        source[columns.timestamp], utc=True, errors="raise"
    ).reset_index(drop=True)
    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "sampled training rows must be chronologically ordered for internal early stopping"
        )
    if columns.label_end is None or columns.label_end not in source:
        label_end = timestamps
    else:
        label_end = pd.to_datetime(
            source[columns.label_end], utc=True, errors="raise"
        ).reset_index(drop=True)
    if (
        timestamps.isna().any()
        or label_end.isna().any()
        or (label_end < timestamps).any()
    ):
        raise ValueError(
            "internal early-stop labels must not resolve before their decision timestamps"
        )
    target = sampled_labels.astype(str).reset_index(drop=True)
    minimum_start = max(1, int(np.ceil(len(source) * (1.0 - validation_fraction))))
    timestamp_ns = timestamps.astype("int64", copy=False).to_numpy()
    label_end_ns = label_end.astype("int64", copy=False).to_numpy()
    # Tail boundaries must retain complete decision-timestamp groups.
    boundaries = np.flatnonzero(np.r_[True, timestamp_ns[1:] != timestamp_ns[:-1]])
    candidate_starts = boundaries[boundaries >= minimum_start]
    classes, class_codes = np.unique(target.to_numpy(), return_inverse=True)
    suffix_counts = np.zeros((len(source) + 1, len(classes)), dtype=np.int32)
    for index in range(len(source) - 1, -1, -1):
        suffix_counts[index] = suffix_counts[index + 1]
        suffix_counts[index, class_codes[index]] += 1
    # A row can enter the internal fit only after both its label has resolved
    # and the decision timestamp is outside the embargo.  Sweep those
    # eligibility cutoffs once instead of re-filtering up to 70k rows per
    # candidate tail boundary.
    eligibility_ns = np.maximum(label_end_ns, timestamp_ns + int(embargo.value))
    eligibility_order = np.argsort(eligibility_ns, kind="mergesort")
    fit_counts = np.zeros(len(classes), dtype=np.int64)
    eligible_cursor = 0
    failures: list[str] = []
    for start_index in candidate_starts:
        validation_start_ns = int(timestamp_ns[int(start_index)])
        while (
            eligible_cursor < len(eligibility_order)
            and eligibility_ns[eligibility_order[eligible_cursor]] < validation_start_ns
        ):
            fit_counts[class_codes[eligibility_order[eligible_cursor]]] += 1
            eligible_cursor += 1
        validation_class_mask = suffix_counts[int(start_index)] > 0
        fit_class_mask = fit_counts > 0
        if fit_class_mask.sum() >= 2 and np.all(
            ~validation_class_mask | fit_class_mask
        ):
            fit_indices = np.flatnonzero(eligibility_ns < validation_start_ns).astype(
                np.int64
            )
            validation_indices = np.arange(
                int(start_index), len(source), dtype=np.int64
            )
            return GeometryPredictorContext(
                fold_id=int(fold_id),
                sampled_train_positions=sampled_positions.copy(),
                early_stop_fit_indices=fit_indices,
                early_stop_validation_indices=validation_indices,
                early_stop_fit_positions=sampled_positions[fit_indices],
                early_stop_validation_positions=sampled_positions[validation_indices],
                validation_start=pd.Timestamp(validation_start_ns, tz="UTC"),
                embargo=embargo,
            )
        failures.append(
            f"start={pd.Timestamp(validation_start_ns, tz='UTC').isoformat()} "
            f"fit_classes={classes[fit_class_mask].tolist()} "
            f"validation_classes={classes[validation_class_mask].tolist()} "
            f"fit_rows={int(fit_counts.sum())}"
        )
    detail = (
        failures[-1]
        if failures
        else "no complete timestamp boundary after nominal tail start"
    )
    raise ValueError(
        "no valid internal chronological early-stop split after purge/embargo; "
        f"fold_id={fold_id}, sampled_rows={len(sampled_positions)}, {detail}"
    )


def catboost_predictor(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    params: Mapping[str, Any],
    context: GeometryPredictorContext,
) -> PredictionOutput:
    """Fit the merged seven-class contract with uniform weights and raw output."""
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "CatBoost is required for geometry-search model scoring"
        ) from exc
    _validate_uniform_pre_refinement_weights(params)
    if set(train_y.astype(str)) != set(EXACT_GEOMETRY_EXPORT_CLASSES):
        raise ValueError(
            "raw seven-class predictor requires every merged hard class in its refit rows"
        )
    capped_params, _ = capped_catboost_params(params)
    early_fit_y = train_y.iloc[context.early_stop_fit_indices].astype(str)
    early_validation_y = train_y.iloc[context.early_stop_validation_indices].astype(str)
    early_names = tuple(sorted(early_fit_y.unique()))
    if not set(early_validation_y).issubset(early_names):
        raise ValueError(
            "internal early-stop validation classes are absent from its fit rows"
        )
    constructor_params = {
        "loss_function": "MultiClass",
        "verbose": False,
        "random_seed": 20260722,
        "allow_writing_files": False,
        **capped_params,
    }
    early_model = CatBoostClassifier(
        **dict(constructor_params, classes_count=len(early_names))
    )
    early_model.fit(
        train_x.iloc[context.early_stop_fit_indices],
        pd.Categorical(early_fit_y, categories=early_names).codes,
        eval_set=(
            train_x.iloc[context.early_stop_validation_indices],
            pd.Categorical(early_validation_y, categories=early_names).codes,
        ),
        early_stopping_rounds=int(capped_params.get("od_wait", 150)),
        use_best_model=True,
        verbose=False,
    )
    best_iteration = (
        early_model.get_best_iteration()
        if hasattr(early_model, "get_best_iteration")
        else None
    )
    tree_count = getattr(early_model, "tree_count_", None)
    iteration_ceiling = int(capped_params.get("iterations", 3000))
    effective_trees = (
        int(best_iteration) + 1
        if best_iteration is not None and int(best_iteration) >= 0
        else int(tree_count or 0)
    )
    if effective_trees <= 0:
        raise ValueError("CatBoost early-stop fit did not expose a usable tree count")
    effective_trees = min(max(1, effective_trees), iteration_ceiling)
    refit_model = CatBoostClassifier(
        **dict(
            constructor_params,
            iterations=effective_trees,
            classes_count=len(EXACT_GEOMETRY_EXPORT_CLASSES),
        ),
    )
    refit_model.fit(
        train_x,
        pd.Categorical(
            train_y.astype(str), categories=EXACT_GEOMETRY_EXPORT_CLASSES
        ).codes,
        verbose=False,
    )
    return (
        refit_model.predict_proba(test_x),
        EXACT_GEOMETRY_EXPORT_CLASSES,
        {
            **context.audit(),
            "effective_tree_count": effective_trees,
            "refit_rows": int(len(train_y)),
            "refit_contract": "one_uniform_weight_merged_seven_class_refit_on_all_sampled_outer_train_rows",
        },
    )


def _feature_matrix(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> pd.DataFrame:
    feature_columns = validate_preentry_features(feature_columns)
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(
            "frozen model features are absent; provide --features-parquet or canonical feature-store join: "
            + ", ".join(missing[:12])
        )
    values = (
        frame.loc[:, list(feature_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return values.fillna(values.median()).fillna(0.0).astype(np.float32)


def bounded_chronological_training_positions(
    frame: pd.DataFrame,
    positions: np.ndarray,
    labels: pd.Series,
    *,
    max_rows: int,
    columns: PathGeometryColumns = PathGeometryColumns(),
) -> np.ndarray:
    """Bound one fold's training rows without losing chronological class support.

    The input positions must already be ordered by UTC timestamp.  Sampling is
    deterministic: each side x geometry-label stratum receives proportional
    support (at least one row when the cap permits), then rows are selected at
    evenly spaced chronological locations within that stratum.  OOS rows are
    intentionally not passed through this function.
    """
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) != len(labels):
        raise ValueError("training positions and labels must have equal length")
    if max_rows == 0 or len(positions) <= max_rows:
        return positions.copy()
    source = frame.iloc[positions]
    timestamps = pd.to_datetime(source[columns.timestamp], utc=True, errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("training positions must be chronologically ordered UTC rows")
    strata = pd.DataFrame(
        {
            "position": positions,
            "side": source[columns.side].astype(str).to_numpy(),
            "geometry_class": labels.astype(str).to_numpy(),
        }
    )
    grouped = list(strata.groupby(["side", "geometry_class"], sort=True, observed=True))
    available = np.asarray([len(group) for _, group in grouped], dtype=np.int64)
    if not len(available):
        return positions[:0]
    # Preserve every supported side x class stratum when the requested bound
    # allows it; otherwise deterministic largest-remainder allocation chooses
    # the most represented strata.
    quotas = np.zeros(len(available), dtype=np.int64)
    if max_rows >= len(available):
        quotas[:] = 1
    else:
        order = sorted(
            range(len(available)),
            key=lambda index: (-available[index], grouped[index][0]),
        )
        quotas[np.asarray(order[:max_rows], dtype=np.int64)] = 1
    remaining = int(max_rows - quotas.sum())
    capacity = available - quotas
    while remaining and capacity.sum() > 0:
        raw = remaining * capacity / capacity.sum()
        extra = np.minimum(capacity, np.floor(raw).astype(np.int64))
        assigned = int(extra.sum())
        quotas += extra
        remaining -= assigned
        capacity -= extra
        if not remaining:
            break
        order = sorted(
            (index for index, value in enumerate(capacity) if value > 0),
            key=lambda index: (
                -(raw[index] - np.floor(raw[index])),
                -capacity[index],
                grouped[index][0],
            ),
        )
        for index in order[:remaining]:
            quotas[index] += 1
            capacity[index] -= 1
        remaining = 0
    sampled: list[np.ndarray] = []
    for quota, (_, group) in zip(quotas, grouped):
        if not quota:
            continue
        group_positions = group["position"].to_numpy(dtype=np.int64)
        # Quotas never exceed stratum support.  The endpoints ensure every
        # retained stratum spans its available train-time range, not its head.
        selected = np.rint(np.linspace(0, len(group_positions) - 1, int(quota))).astype(
            np.int64
        )
        sampled.append(group_positions[selected])
    return np.sort(np.concatenate(sampled)).astype(np.int64, copy=False)


def _geometry_config_payload(config: PathGeometryConfig) -> dict[str, Any]:
    # Optional boundaries with their inherited/default value are omitted so
    # adding a new optional refinement knob does not invalidate semantically
    # identical in-flight checkpoints.
    return {key: value for key, value in asdict(config).items() if value is not None}


def geometry_config_id(config: PathGeometryConfig) -> str:
    """Stable identifier for one complete deterministic geometry rule set."""
    payload = json.dumps(
        _geometry_config_payload(config), sort_keys=True, separators=(",", ":")
    )
    return "geometry_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _oos_prediction_frame(
    frame: pd.DataFrame,
    positions: np.ndarray,
    true_labels: pd.Series,
    raw_probabilities: np.ndarray,
    config: PathGeometryConfig,
    fold: ChronologicalFold,
    columns: PathGeometryColumns,
) -> pd.DataFrame:
    """Build a strict OOS handoff with canonical identity and class ordering."""
    raw = np.asarray(raw_probabilities, dtype=np.float64)
    expected_shape = (len(positions), len(PATH_GEOMETRY_CLASSES))
    if raw.shape != expected_shape:
        raise ValueError("aligned OOS probability matrix has an invalid shape")
    raw = _normalise_probability_rows(raw)
    raw_entropy = -np.sum(raw * np.log(np.clip(raw, 1e-12, 1.0)), axis=1)
    predicted_index = raw.argmax(axis=1)
    source = frame.iloc[positions]
    timestamp = pd.to_datetime(source[columns.timestamp], utc=True, errors="coerce")
    if timestamp.isna().any():
        raise ValueError("captured OOS prediction keys contain invalid UTC timestamps")
    if "candidate_id" not in source:
        raise ValueError("captured geometry OOS predictions require candidate_id")
    train_timestamp = pd.to_datetime(
        frame.iloc[fold.train_indices][columns.timestamp], utc=True, errors="raise"
    )
    if columns.label_end is None or columns.label_end not in frame:
        train_resolution = train_timestamp
    else:
        train_resolution = pd.to_datetime(
            frame.iloc[fold.train_indices][columns.label_end], utc=True, errors="raise"
        )
    latest_train_decision = train_timestamp.max()
    latest_train_resolution = train_resolution.max()
    training_information_cutoff = max(latest_train_decision, latest_train_resolution)
    if not training_information_cutoff < fold.oos_start:
        raise ValueError("geometry OOS training information reaches validation")
    result = pd.DataFrame(
        {
            "source_row_position": positions.astype(np.int64),
            "__ts__": timestamp.to_numpy(),
            "__symbol__": source[columns.symbol].astype(str).to_numpy(),
            "side": source[columns.side].astype(str).to_numpy(),
            "candidate_id": source["candidate_id"].astype(str).to_numpy(),
            "true_dynamic_label": true_labels.astype(str).to_numpy(),
            "predicted_class": [
                PATH_GEOMETRY_CLASSES[index] for index in predicted_index
            ],
            "probability_vector": [row.tolist() for row in raw],
            "probability_entropy": raw_entropy,
            "fold_id": int(fold.fold_id),
            "train_cutoff_utc": training_information_cutoff,
            "available_at": timestamp.to_numpy(),
            "validation_start": pd.Timestamp(fold.oos_start),
            "latest_train_decision_ts": latest_train_decision,
            "label_resolution_available_at": latest_train_resolution,
            "train_decision_cutoff": training_information_cutoff,
            "oos_start_utc": pd.Timestamp(fold.oos_start),
            "oos_end_utc": pd.Timestamp(fold.oos_end),
            "config_id": geometry_config_id(config),
        }
    )
    for name, value in asdict(config).items():
        result[f"config_{name}"] = value
    for index, class_name in enumerate(PATH_GEOMETRY_CLASSES):
        result[f"probability_{class_name}"] = raw[:, index]
    result["max_probability"] = raw.max(axis=1)
    result["normalized_entropy"] = raw_entropy / np.log(len(PATH_GEOMETRY_CLASSES))
    ordered = np.sort(raw, axis=1)
    result["top1_top2_probability_margin"] = ordered[:, -1] - ordered[:, -2]
    return result


def _support(
    frame: pd.DataFrame, labels: pd.Series, column: str, name: str
) -> pd.DataFrame:
    if column not in frame:
        return pd.DataFrame(columns=[name, "path_geometry_label", "rows"])
    data = pd.DataFrame(
        {name: frame[column].astype(str), "path_geometry_label": labels.astype(str)}
    )
    return (
        data.groupby([name, "path_geometry_label"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )


def _group_scores(
    frame: pd.DataFrame,
    truth: Sequence[str],
    raw_probabilities: np.ndarray,
    column: str,
    group_name: str,
    fold_id: int,
) -> list[dict[str, Any]]:
    if column not in frame:
        return []
    labels = np.asarray(truth, dtype=object)
    output: list[dict[str, Any]] = []
    for value, positions in frame.groupby(
        column, sort=True, observed=True
    ).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        if not len(index):
            continue
        output.append(
            {
                group_name: str(value),
                "fold_id": fold_id,
                "rows": int(len(index)),
                **multiclass_scores(
                    labels[index].tolist(),
                    raw_probabilities[index],
                    PATH_GEOMETRY_CLASSES,
                ),
                **confidence_metrics(raw_probabilities[index], "raw"),
            }
        )
    return output


def _group_stability(records: list[dict[str, Any]], group_name: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                group_name,
                "support_rows",
                "folds",
                "mean_oos_logloss",
                "std_oos_logloss",
                "mean_macro_f1",
                "std_macro_f1",
            ]
        )
    frame = pd.DataFrame(records)
    grouped = frame.groupby(group_name, observed=True)
    return grouped.agg(
        support_rows=("rows", "sum"),
        folds=("fold_id", "nunique"),
        mean_oos_logloss=("oos_logloss", "mean"),
        std_oos_logloss=("oos_logloss", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
    ).reset_index()


def _stability_score(frame: pd.DataFrame) -> float:
    """Return a bounded stability score from metric dispersion records."""
    if frame.empty:
        return 1.0
    logloss_std = float(np.nan_to_num(frame["oos_logloss"].std(ddof=0), nan=0.0))
    f1_std = float(np.nan_to_num(frame["macro_f1"].std(ddof=0), nan=0.0))
    return float(1.0 / (1.0 + logloss_std + f1_std))


def _selection_score(summary: Mapping[str, float]) -> float:
    econ = float(np.nan_to_num(summary.get("economic_selection_score"), nan=0.0))
    return float(
        0.30 / (1.0 + summary["oos_logloss"])
        + 0.20 * summary["macro_f1"]
        + 0.10 * summary["weighted_f1"]
        + 0.10 * (1.0 - summary["ranked_probability_score"])
        + 0.10 / (1.0 + summary["confusion_distance"])
        + 0.15 * np.tanh(econ / 2.0)
        + 0.05 * summary["stability_score"]
    )


def evaluate_geometry_config(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    model_params: Mapping[str, Any],
    config: PathGeometryConfig,
    *,
    columns: PathGeometryColumns = PathGeometryColumns(),
    folds: Sequence[ChronologicalFold] | None = None,
    predictor: Predictor = catboost_predictor,
    capture_predictions: bool = False,
    prepared_feature_matrix: pd.DataFrame | None = None,
    max_train_rows_per_fold: int = DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
) -> dict[str, Any]:
    if max_train_rows_per_fold < 0:
        raise ValueError("max_train_rows_per_fold must be non-negative")
    prepared = ensure_risk_fraction(frame, columns)
    labels = label_path_geometry(prepared, config, columns)
    if prepared_feature_matrix is None:
        x = _feature_matrix(prepared, feature_columns)
    else:
        if len(prepared_feature_matrix) != len(prepared):
            raise ValueError(
                "prepared feature matrix row count does not match geometry frame"
            )
        if list(prepared_feature_matrix.columns) != list(feature_columns):
            raise ValueError(
                "prepared feature matrix does not match frozen feature columns"
            )
        x = prepared_feature_matrix
    ts = pd.to_datetime(prepared[columns.timestamp], utc=True, errors="coerce")
    end = (
        ts
        if columns.label_end is None or columns.label_end not in prepared
        else pd.to_datetime(prepared[columns.label_end], utc=True, errors="coerce")
    )
    use_folds = (
        tuple(folds)
        if folds is not None
        else four_month_walk_forward_folds(ts, label_end=end)
    )
    records: list[dict[str, Any]] = []
    side_records: list[dict[str, Any]] = []
    symbol_records: list[dict[str, Any]] = []
    month_records: list[dict[str, Any]] = []
    probability_reliability_bin_frames: list[pd.DataFrame] = []
    economic_confusion_frames: list[pd.DataFrame] = []
    economic_confusion_prior_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold in use_folds:
        train_raw = labels.iloc[fold.train_indices]["path_geometry_label"]
        test_raw = labels.iloc[fold.oos_indices]["path_geometry_label"]
        train_positions = fold.train_indices[train_raw.notna().to_numpy()]
        test_positions = fold.oos_indices[test_raw.notna().to_numpy()]
        train_y, test_y = train_raw.dropna().astype(str), test_raw.dropna().astype(str)
        if len(train_y) < 2 or len(test_y) < 1 or train_y.nunique() < 2:
            continue
        sampled_train_positions = bounded_chronological_training_positions(
            prepared,
            train_positions,
            train_y,
            max_rows=max_train_rows_per_fold,
            columns=columns,
        )
        sampled_train_y = labels.iloc[sampled_train_positions][
            "path_geometry_label"
        ].astype(str)
        if sampled_train_y.nunique() < 2:
            continue
        predictor_context = _early_stop_context(
            prepared,
            sampled_train_positions,
            sampled_train_y,
            fold_id=fold.fold_id,
            columns=columns,
        )
        probabilities, classes, fit_report_raw = predictor(
            x.iloc[sampled_train_positions],
            sampled_train_y,
            x.iloc[test_positions],
            model_params,
            predictor_context,
        )
        if not isinstance(fit_report_raw, Mapping):
            raise TypeError(
                "geometry predictor must return a mapping fit report as its third value"
            )
        fit_report = dict(fit_report_raw)
        aligned_probabilities = _align_probabilities(probabilities, classes)
        predicted = [
            PATH_GEOMETRY_CLASSES[index]
            for index in aligned_probabilities.argmax(axis=1)
        ]
        probability_metrics, probability_bins = _raw_probability_reliability_metrics(
            test_y.tolist(), aligned_probabilities, "raw"
        )
        probability_bins.insert(0, "fold_id", int(fold.fold_id))
        if capture_predictions:
            prediction_frames.append(
                _oos_prediction_frame(
                    prepared,
                    test_positions,
                    test_y,
                    aligned_probabilities,
                    config,
                    fold,
                    columns,
                )
            )
        economics = economic_separation(
            prepared.iloc[test_positions], test_y.tolist(), predicted, columns
        )
        raw_confusion = economic_confusion_diagnostics(
            prepared.iloc[train_positions],
            train_y.tolist(),
            test_y.tolist(),
            predicted,
            columns,
        )
        if capture_predictions:
            matrix = raw_confusion["matrix"].copy()
            matrix.insert(0, "prediction_variant", "raw")
            matrix.insert(0, "fold_id", int(fold.fold_id))
            economic_confusion_frames.append(matrix)
            priors = raw_confusion["class_ev_priors"].copy()
            priors.insert(0, "prediction_variant", "raw")
            priors.insert(0, "fold_id", int(fold.fold_id))
            economic_confusion_prior_frames.append(priors)
            probability_reliability_bin_frames.append(probability_bins)
        records.append(
            {
                "fold_id": fold.fold_id,
                "rows": int(len(test_y)),
                "requested_train_rows": int(max_train_rows_per_fold),
                "available_train_rows": int(len(train_y)),
                "effective_train_rows": int(len(sampled_train_y)),
                "full_validation_rows": int(len(test_y)),
                **predictor_context.audit(),
                **dict(fit_report),
                # Preserve the pre-existing raw-probability selection metrics.
                **multiclass_scores(
                    test_y.tolist(), aligned_probabilities, PATH_GEOMETRY_CLASSES
                ),
                **probability_metrics,
                **confidence_metrics(aligned_probabilities, "raw"),
                **{
                    f"raw_{key}": value
                    for key, value in raw_confusion["metrics"].items()
                },
                **economics,
            }
        )
        side_records.extend(
            _group_scores(
                prepared.iloc[test_positions],
                test_y.tolist(),
                aligned_probabilities,
                columns.side,
                "side",
                fold.fold_id,
            )
        )
        symbol_records.extend(
            _group_scores(
                prepared.iloc[test_positions],
                test_y.tolist(),
                aligned_probabilities,
                columns.symbol,
                "symbol",
                fold.fold_id,
            )
        )
        oos_frame = prepared.iloc[test_positions]
        oos_months = pd.to_datetime(oos_frame[columns.timestamp], utc=True).dt.strftime(
            "%Y-%m"
        )
        for month in sorted(oos_months.unique()):
            position = np.flatnonzero(oos_months.to_numpy() == month)
            month_truth = test_y.iloc[position].tolist()
            month_predicted = [predicted[index] for index in position]
            month_economics = economic_separation(
                oos_frame.iloc[position], month_truth, month_predicted, columns
            )
            month_records.append(
                {
                    "fold_id": fold.fold_id,
                    "oos_month": str(month),
                    "rows": int(len(position)),
                    **multiclass_scores(
                        month_truth,
                        aligned_probabilities[position],
                        PATH_GEOMETRY_CLASSES,
                    ),
                    **confidence_metrics(aligned_probabilities[position], "raw"),
                    **month_economics,
                }
            )
    if not records:
        raise ValueError(
            "no viable OOS fold has labelled rows and at least two train classes"
        )
    fold_metrics = pd.DataFrame(records)
    summary = {
        column: float(fold_metrics[column].mean())
        for column in fold_metrics
        if pd.api.types.is_numeric_dtype(fold_metrics[column])
        and column
        not in {
            "fold_id",
            "rows",
            "requested_train_rows",
            "available_train_rows",
            "effective_train_rows",
            "full_validation_rows",
            "early_stop_fit_rows",
            "early_stop_validation_rows",
            "early_stop_fit_source_rows",
            "early_stop_validation_source_rows",
        }
    }
    for metric in ("oos_logloss", "macro_f1", "economic_separation_score"):
        summary[f"{metric}_fold_std"] = float(fold_metrics[metric].std(ddof=0))
    monthly_metrics = pd.DataFrame(month_records)
    side_metrics = pd.DataFrame(side_records)
    symbol_metrics = pd.DataFrame(symbol_records)
    summary["fold_stability"] = _stability_score(fold_metrics)
    summary["temporal_month_stability"] = _stability_score(monthly_metrics)
    summary["side_stability_score"] = _stability_score(side_metrics)
    summary["symbol_stability_score"] = _stability_score(symbol_metrics)
    summary["stability_score"] = float(
        0.35 * summary["fold_stability"]
        + 0.35 * summary["temporal_month_stability"]
        + 0.15 * summary["side_stability_score"]
        + 0.15 * summary["symbol_stability_score"]
    )
    # A collapsed predicted class has no downstream separation, so it earns
    # zero rather than being silently excluded from target selection.
    summary["economic_selection_score"] = float(
        0.5 * np.nan_to_num(summary.get("true_economic_separation_score"), nan=0.0)
        + 0.5
        * np.nan_to_num(summary.get("predicted_economic_separation_score"), nan=0.0)
    )
    summary["evaluated_folds"], summary["evaluated_oos_rows"] = (
        int(len(fold_metrics)),
        int(fold_metrics["full_validation_rows"].sum()),
    )
    summary["requested_train_rows_per_fold"] = int(max_train_rows_per_fold)
    summary["effective_train_rows"] = int(fold_metrics["effective_train_rows"].sum())
    summary["full_validation_rows"] = int(fold_metrics["full_validation_rows"].sum())
    summary["evaluated_oos_calendar_months"] = int(
        monthly_metrics["oos_month"].nunique()
    )
    summary["selection_score"] = _selection_score(summary)
    result = {
        "config": _geometry_config_payload(config),
        "summary": summary,
        "folds": fold_metrics,
        "boundary": boundary_diagnostics(labels),
        "side_support": _support(
            prepared, labels["path_geometry_label"], columns.side, "side"
        ),
        "symbol_support": _support(
            prepared, labels["path_geometry_label"], columns.symbol, "symbol"
        ),
        "side_stability": _group_stability(side_records, "side"),
        "symbol_stability": _group_stability(symbol_records, "symbol"),
        "temporal_month_stability": monthly_metrics,
        "side_diagnostics": side_metrics,
        "symbol_diagnostics": symbol_metrics,
        "month_diagnostics": monthly_metrics.copy(),
        "probability_reliability_bins": (
            pd.concat(probability_reliability_bin_frames, ignore_index=True)
            if probability_reliability_bin_frames
            else pd.DataFrame()
        ),
        "economic_confusion": (
            pd.concat(economic_confusion_frames, ignore_index=True)
            if economic_confusion_frames
            else pd.DataFrame()
        ),
        "economic_confusion_priors": (
            pd.concat(economic_confusion_prior_frames, ignore_index=True)
            if economic_confusion_prior_frames
            else pd.DataFrame()
        ),
    }
    if capture_predictions:
        predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        )
        identity = ["source_row_position", "__ts__", "__symbol__", "side", "fold_id"]
        if not predictions.empty and predictions.duplicated(identity).any():
            raise ValueError(
                "captured OOS predictions contain duplicate row/fold identities"
            )
        result["oos_predictions"] = predictions
    return result


def _rank_key(result: Mapping[str, Any]) -> tuple[float, float, float, str]:
    summary = result["summary"]
    return (
        -summary["selection_score"],
        -summary["fold_stability"],
        summary["oos_logloss"],
        repr(sorted(result["config"].items())),
    )


def stable_plateau_select(
    results: Sequence[Mapping[str, Any]], *, score_tolerance: float = 0.02
) -> Mapping[str, Any]:
    if not results:
        raise ValueError("stable plateau selection needs candidates")
    best = max(float(item["summary"]["selection_score"]) for item in results)
    plateau = [
        item
        for item in results
        if float(item["summary"]["selection_score"]) >= best - score_tolerance
    ]
    return min(
        plateau,
        key=lambda item: (
            -item["summary"]["fold_stability"],
            -item["summary"]["selection_score"],
            item["summary"]["oos_logloss"],
            repr(sorted(item["config"].items())),
        ),
    )


def nested_finalist_validation(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    model_params: Mapping[str, Any],
    finalists: Sequence[Mapping[str, Any]],
    *,
    columns: PathGeometryColumns = PathGeometryColumns(),
    predictor: Predictor = catboost_predictor,
    prepared_feature_matrix: pd.DataFrame | None = None,
    max_train_rows_per_fold: int = DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
    completed_outer_folds: Mapping[str, Mapping[str, Any]] | None = None,
    completed_evaluations: Mapping[str, Mapping[str, Any]] | None = None,
    on_outer_fold_complete: Callable[[int, Mapping[str, Any]], None] | None = None,
    on_evaluation_complete: Callable[[str, Mapping[str, Any]], None] | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> list[dict[str, Any]]:
    """Select on 4m inner OOS, then score the following 4m outer OOS."""
    if not finalists:
        return []
    timestamps = pd.to_datetime(frame[columns.timestamp], utc=True, errors="coerce")
    end = (
        timestamps
        if columns.label_end is None or columns.label_end not in frame
        else pd.to_datetime(frame[columns.label_end], utc=True, errors="coerce")
    )
    start, last, output = timestamps.iloc[0], timestamps.iloc[-1], []
    outer_id = 0
    while start + pd.DateOffset(months=GEOMETRY_NESTED_MONTHS) <= last + pd.Timedelta(
        nanoseconds=1
    ):
        inner = fixed_four_month_ablation_fold(timestamps, start, label_end=end)
        outer_start = start + pd.DateOffset(
            months=GEOMETRY_TRAIN_MONTHS + GEOMETRY_OOS_MONTHS
        )
        outer_end = outer_start + pd.DateOffset(months=GEOMETRY_OOS_MONTHS)
        outer_train = np.flatnonzero(
            (timestamps >= start).to_numpy()
            & (timestamps < outer_start).to_numpy()
            & (end < outer_start).to_numpy()
        )
        outer_oos = np.flatnonzero(
            (timestamps >= outer_start).to_numpy() & (timestamps < outer_end).to_numpy()
        )
        if len(outer_train) and len(outer_oos):
            completed = (completed_outer_folds or {}).get(str(outer_id))
            if completed is not None:
                _report_progress(
                    progress_reporter,
                    "nested_fold_resume",
                    outer_fold_id=outer_id,
                    outer_oos_start=pd.Timestamp(outer_start).isoformat(),
                    outer_oos_end=pd.Timestamp(outer_end).isoformat(),
                )
                output.append(dict(completed))
                start += pd.DateOffset(months=GEOMETRY_TRAIN_MONTHS)
                outer_id += 1
                continue
            _report_progress(
                progress_reporter,
                "nested_fold_start",
                outer_fold_id=outer_id,
                inner_train_start=pd.Timestamp(start).isoformat(),
                inner_oos_start=pd.Timestamp(inner.oos_start).isoformat(),
                outer_oos_start=pd.Timestamp(outer_start).isoformat(),
                outer_oos_end=pd.Timestamp(outer_end).isoformat(),
                finalist_count=len(finalists),
            )
            inner_results: list[dict[str, Any]] = []
            for item in finalists:
                candidate = PathGeometryConfig(**item["config"])
                config_id = geometry_config_id(candidate)
                evaluation_key = f"{outer_id}:inner:{config_id}"
                restored_result = (completed_evaluations or {}).get(evaluation_key)
                if restored_result is not None:
                    _report_progress(
                        progress_reporter,
                        "nested_inner_candidate_resume",
                        outer_fold_id=outer_id,
                        config_id=config_id,
                    )
                    inner_results.append(_restore_checkpoint_result(restored_result))
                    continue
                _report_progress(
                    progress_reporter,
                    "nested_inner_candidate_start",
                    outer_fold_id=outer_id,
                    config_id=config_id,
                )
                result = evaluate_geometry_config(
                    frame,
                    feature_columns,
                    model_params,
                    candidate,
                    columns=columns,
                    folds=(inner,),
                    predictor=predictor,
                    prepared_feature_matrix=prepared_feature_matrix,
                    max_train_rows_per_fold=max_train_rows_per_fold,
                )
                if on_evaluation_complete is not None:
                    on_evaluation_complete(evaluation_key, result)
                inner_results.append(result)
                _report_progress(
                    progress_reporter,
                    "nested_inner_candidate_complete",
                    outer_fold_id=outer_id,
                    config_id=config_id,
                )
            selected = stable_plateau_select(inner_results)
            outer = ChronologicalFold(
                outer_id, outer_train, outer_oos, outer_start, outer_start, outer_end
            )
            selected_config = PathGeometryConfig(**selected["config"])
            selected_config_id = geometry_config_id(selected_config)
            outer_evaluation_key = f"{outer_id}:outer:{selected_config_id}"
            restored_outer = (completed_evaluations or {}).get(outer_evaluation_key)
            if restored_outer is not None:
                _report_progress(
                    progress_reporter,
                    "nested_outer_candidate_resume",
                    outer_fold_id=outer_id,
                    config_id=selected_config_id,
                )
                scored = _restore_checkpoint_result(restored_outer)
            else:
                _report_progress(
                    progress_reporter,
                    "nested_outer_candidate_start",
                    outer_fold_id=outer_id,
                    config_id=selected_config_id,
                )
                scored = evaluate_geometry_config(
                    frame,
                    feature_columns,
                    model_params,
                    selected_config,
                    columns=columns,
                    folds=(outer,),
                    predictor=predictor,
                    prepared_feature_matrix=prepared_feature_matrix,
                    max_train_rows_per_fold=max_train_rows_per_fold,
                )
                if on_evaluation_complete is not None:
                    on_evaluation_complete(outer_evaluation_key, scored)
                _report_progress(
                    progress_reporter,
                    "nested_outer_candidate_complete",
                    outer_fold_id=outer_id,
                    config_id=selected_config_id,
                )
            entry = {
                "outer_fold_id": outer_id,
                "inner_train_start": pd.Timestamp(start).isoformat(),
                "inner_oos_start": pd.Timestamp(inner.oos_start).isoformat(),
                "inner_oos_end": pd.Timestamp(inner.oos_end).isoformat(),
                "outer_oos_start": pd.Timestamp(outer.oos_start).isoformat(),
                "outer_oos_end": pd.Timestamp(outer.oos_end).isoformat(),
                "selected_config": selected["config"],
                "inner_summary": selected["summary"],
                "outer_summary": scored["summary"],
                "inner_fold_reports": selected["folds"].to_dict(orient="records"),
                "outer_fold_reports": scored["folds"].to_dict(orient="records"),
            }
            output.append(entry)
            if on_outer_fold_complete is not None:
                on_outer_fold_complete(outer_id, entry)
            _report_progress(
                progress_reporter,
                "nested_fold_complete",
                outer_fold_id=outer_id,
                selected_config_id=geometry_config_id(
                    PathGeometryConfig(**selected["config"])
                ),
                selection_score=selected["summary"]["selection_score"],
            )
        start += pd.DateOffset(months=GEOMETRY_TRAIN_MONTHS)
        outer_id += 1
    return output


def _result_row(
    stage: str, parameter: str, result: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "stage": stage,
        "parameter": parameter,
        **result["config"],
        **result["summary"],
    }


def reduced_joint_best_two_values(
    sweep_results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[Any, ...]]:
    """Keep exactly the two highest-ranked values from every 1D sweep."""
    selected: dict[str, tuple[Any, ...]] = {}
    for parameter, results in sweep_results.items():
        ranked = sorted(results, key=_rank_key)
        values: list[Any] = []
        for result in ranked:
            value = result["config"][parameter]
            if value not in values:
                values.append(value)
            if len(values) == 2:
                break
        if len(values) != min(2, len({item["config"][parameter] for item in results})):
            raise ValueError(
                f"1D sweep did not produce two distinct values for {parameter}"
            )
        selected[parameter] = tuple(values)
    return selected


def reduced_joint_design(
    incumbent: PathGeometryConfig,
    best_two_values: Mapping[str, Sequence[Any]],
    max_joint_trials: int,
) -> tuple[PathGeometryConfig, ...]:
    """Deterministic ring-pair design spanning parameters without factorial growth."""
    if max_joint_trials < 0:
        raise ValueError("max_joint_trials must be non-negative")
    parameters = sorted(best_two_values)
    if len(parameters) < 2 or max_joint_trials == 0:
        return ()
    pairs = [
        (parameters[index], parameters[(index + 1) % len(parameters)])
        for index in range(len(parameters))
    ]
    combinations_by_pair = ((0, 0), (0, 1), (1, 0), (1, 1))
    design: list[PathGeometryConfig] = []
    # Cycle combinations before repeating a pair so a small capped design has
    # broad parameter coverage rather than overfitting the alphabetic prefix.
    for combination in combinations_by_pair:
        for left, right in pairs:
            candidate = replace(
                incumbent,
                **{
                    left: best_two_values[left][combination[0]],
                    right: best_two_values[right][combination[1]],
                },
            )
            if candidate not in design:
                design.append(candidate)
            if len(design) >= max_joint_trials:
                return tuple(design)
    return tuple(design)


def staged_geometry_search(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    model_params: Mapping[str, Any],
    *,
    columns: PathGeometryColumns = PathGeometryColumns(),
    incumbent: PathGeometryConfig = PathGeometryConfig(),
    predictor: Predictor = catboost_predictor,
    max_joint_trials: int = 24,
    score_tolerance: float = 0.02,
    ablation_start_date: str | pd.Timestamp | None = None,
    nested_oof: bool = False,
    evaluation_mode: str = GEOMETRY_EVALUATION_MODE_LEGACY,
    short_history_development_end: str | pd.Timestamp | None = None,
    short_history_subfold_count: int = 2,
    capture_predictions: bool = False,
    run_post_search_refits: bool = True,
    max_train_rows_per_fold: int = DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
    checkpoint_path: Path | None = None,
    checkpoint_input_identity: Mapping[str, Any] | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """One-dimensional sweeps plus deterministic sampled cross-parameter trials."""
    incumbent.validate()
    if max_joint_trials < 0:
        raise ValueError("max_joint_trials must be non-negative")
    if max_train_rows_per_fold < 0:
        raise ValueError("max_train_rows_per_fold must be non-negative")
    if evaluation_mode not in {
        GEOMETRY_EVALUATION_MODE_LEGACY,
        GEOMETRY_EVALUATION_MODE_SHORT_HISTORY,
    }:
        raise ValueError("unknown geometry evaluation mode")
    if evaluation_mode == GEOMETRY_EVALUATION_MODE_SHORT_HISTORY:
        if short_history_development_end is None:
            raise ValueError(
                "short-history geometry requires a development-end boundary"
            )
        if nested_oof:
            raise ValueError("short-history geometry forbids nested OOF")
        if ablation_start_date is not None:
            raise ValueError("short-history geometry forbids 4m ablation overrides")
    effective_model_params, catboost_resource = capped_catboost_params(model_params)
    geometry_grid: Mapping[str, Sequence[Any]] = GEOMETRY_GRID
    incumbent_contract = asdict(incumbent)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with Path(checkpoint_path).open() as handle:
            checkpoint_preview = json.load(handle)
        preview_contract = dict(checkpoint_preview.get("contract") or {})
        preview_grid = preview_contract.get("geometry_grid")
        preview_incumbent = preview_contract.get("incumbent")
        current_without_fast_margin = {
            key: value
            for key, value in GEOMETRY_GRID.items()
            if key != "fast_net_margin_atr"
        }
        incumbent_without_fast_margin = {
            key: value
            for key, value in incumbent_contract.items()
            if key != "fast_net_margin_atr"
        }
        legacy_fast_margin_checkpoint = (
            isinstance(preview_grid, Mapping)
            and _json_ready(preview_grid) == _json_ready(current_without_fast_margin)
            and isinstance(preview_incumbent, Mapping)
            and dict(preview_incumbent) == _json_ready(incumbent_without_fast_margin)
            and incumbent.fast_net_margin_atr is None
        )
        if legacy_fast_margin_checkpoint:
            # Finish an in-flight pre-fast-margin sweep under its exact original
            # search contract. The new class-specific boundary is evaluated by
            # the separate winner-refinement stage.
            # Keep the source-code insertion order because the staged search
            # updates its incumbent after each one-dimensional parameter.
            # Checkpoint JSON is key-sorted on disk, so iterating preview_grid
            # would silently change the search trajectory.
            geometry_grid = current_without_fast_margin
            incumbent_contract = dict(preview_incumbent)
    _report_progress(
        progress_reporter,
        "feature_prep_start",
        input_rows=int(len(frame)),
        feature_count=len(feature_columns),
    )
    ordered = ensure_risk_fraction(frame, columns)
    ordered[columns.timestamp] = pd.to_datetime(
        ordered[columns.timestamp], utc=True, errors="coerce"
    )
    ordered = ordered.sort_values(columns.timestamp, kind="mergesort").reset_index(
        drop=True
    )
    prepared_feature_matrix = _feature_matrix(ordered, feature_columns)
    _report_progress(
        progress_reporter,
        "feature_prep_complete",
        ordered_rows=int(len(ordered)),
        feature_count=len(feature_columns),
        feature_matrix_rows=int(len(prepared_feature_matrix)),
    )
    if evaluation_mode == GEOMETRY_EVALUATION_MODE_SHORT_HISTORY:
        assert short_history_development_end is not None
        if columns.label_end is None or columns.label_end not in ordered:
            raise ValueError(
                "short-history geometry requires canonical resolved label-end timestamps"
            )
        selection_folds = short_history_purged_chronological_folds(
            ordered[columns.timestamp],
            label_end=ordered[columns.label_end],
            development_end=short_history_development_end,
            subfold_count=short_history_subfold_count,
            embargo=GEOMETRY_EARLY_STOP_EMBARGO,
        )
        all_folds: tuple[ChronologicalFold, ...] = ()
    else:
        all_folds = four_month_walk_forward_folds(
            ordered[columns.timestamp], label_end=ordered.get(columns.label_end)
        )
        selection_folds = (
            (
                fixed_four_month_ablation_fold(
                    ordered[columns.timestamp],
                    ablation_start_date,
                    label_end=ordered.get(columns.label_end),
                ),
            )
            if ablation_start_date is not None
            else all_folds
        )
    _report_progress(
        progress_reporter,
        "fold_definitions",
        selection_folds=_fold_contract(selection_folds),
        walk_forward_folds=_fold_contract(all_folds),
        train_row_cap=int(max_train_rows_per_fold),
        oos_rows="uncapped_all_labelled_rows",
    )
    checkpoint_contract = {
        "schema": _CHECKPOINT_SCHEMA,
        "input_identity": dict(
            checkpoint_input_identity
            or {"prepared_frame_sha256": _frame_identity(ordered)}
        ),
        "feature_columns": list(feature_columns),
        "effective_model_params": dict(effective_model_params),
        "columns": asdict(columns),
        "geometry_grid": geometry_grid,
        "incumbent": incumbent_contract,
        "selection_folds": _fold_contract(selection_folds),
        "walk_forward_folds": _fold_contract(all_folds),
        "max_train_rows_per_fold": int(max_train_rows_per_fold),
        "max_joint_trials": int(max_joint_trials),
        "score_tolerance": float(score_tolerance),
        "ablation_start_date": str(ablation_start_date)
        if ablation_start_date is not None
        else None,
        "evaluation_mode": evaluation_mode,
        "short_history_development_end": (
            str(short_history_development_end)
            if short_history_development_end is not None
            else None
        ),
        "short_history_subfold_count": int(short_history_subfold_count),
        "nested_oof": bool(nested_oof),
        "capture_predictions": bool(capture_predictions),
        "nested_months": GEOMETRY_NESTED_MONTHS,
        "early_stop_validation_fraction": GEOMETRY_EARLY_STOP_VALIDATION_FRACTION,
        "early_stop_embargo_hours": float(
            GEOMETRY_EARLY_STOP_EMBARGO / pd.Timedelta(hours=1)
        ),
    }
    checkpoint_fingerprint = _checkpoint_fingerprint(checkpoint_contract)
    checkpoint_state: dict[str, Any] = {
        "schema": _CHECKPOINT_SCHEMA,
        "status": "running",
        "fingerprint": checkpoint_fingerprint,
        "contract": checkpoint_contract,
        "completed_configs": {},
        "finalist_captures": {},
        "nested_outer_folds": {},
        "nested_completed_evaluations": {},
    }
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with Path(checkpoint_path).open() as handle:
            restored = json.load(handle)
        if (
            not isinstance(restored, Mapping)
            or restored.get("schema") != _CHECKPOINT_SCHEMA
        ):
            raise ValueError("geometry-search checkpoint has an unsupported schema")
        exact_checkpoint_match = restored.get(
            "fingerprint"
        ) == checkpoint_fingerprint and restored.get("contract") == _json_ready(
            checkpoint_contract
        )
        migrated_nested_disable = False
        if not exact_checkpoint_match and not run_post_search_refits:
            restored_contract = dict(restored.get("contract") or {})
            comparable_restored = dict(restored_contract)
            comparable_current = _json_ready(checkpoint_contract)
            comparable_restored["nested_oof"] = False
            comparable_current["nested_oof"] = False
            migrated_nested_disable = (
                restored_contract.get("nested_oof") is True
                and checkpoint_contract["nested_oof"] is False
                and comparable_restored == comparable_current
                and not restored.get("finalist_captures")
                and not restored.get("nested_outer_folds")
                and not restored.get("nested_completed_evaluations")
            )
        if not exact_checkpoint_match and not migrated_nested_disable:
            restored_contract = dict(restored.get("contract") or {})
            current_contract = _json_ready(checkpoint_contract)
            changed_keys = sorted(
                key
                for key in set(restored_contract).union(current_contract)
                if restored_contract.get(key) != current_contract.get(key)
            )
            raise ValueError(
                "geometry-search checkpoint fingerprint does not match the current "
                f"exact contract; changed top-level keys={changed_keys}"
            )
        for key in (
            "completed_configs",
            "finalist_captures",
            "nested_outer_folds",
            "nested_completed_evaluations",
        ):
            if not isinstance(restored.get(key, {}), Mapping):
                raise ValueError(f"geometry-search checkpoint has invalid {key}")
        checkpoint_state.update(dict(restored))
        if migrated_nested_disable:
            checkpoint_state["contract"] = checkpoint_contract
            checkpoint_state["fingerprint"] = checkpoint_fingerprint
            checkpoint_state["finalist_captures"] = {}
            checkpoint_state["nested_outer_folds"] = {}
            checkpoint_state["nested_completed_evaluations"] = {}
            _report_progress(
                progress_reporter,
                "checkpoint_contract_migrated",
                migration="disable_nested_and_all_post_search_refits",
                checkpoint_path=str(checkpoint_path),
            )
        _report_progress(
            progress_reporter,
            "checkpoint_resume",
            checkpoint_path=str(checkpoint_path),
            completed_config_count=len(checkpoint_state["completed_configs"]),
            finalist_capture_count=len(checkpoint_state["finalist_captures"]),
            nested_fold_count=len(checkpoint_state["nested_outer_folds"]),
            nested_evaluation_count=len(
                checkpoint_state["nested_completed_evaluations"]
            ),
        )

    def persist_checkpoint(reason: str) -> None:
        if checkpoint_path is None:
            return
        checkpoint_state["status"] = "running"
        checkpoint_state["last_checkpoint_reason"] = reason
        _atomic_json_write(Path(checkpoint_path), checkpoint_state)
        _report_progress(
            progress_reporter,
            "checkpoint_saved",
            checkpoint_path=str(checkpoint_path),
            reason=reason,
            completed_config_count=len(checkpoint_state["completed_configs"]),
        )

    cache: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}

    def evaluate(
        candidate: PathGeometryConfig, *, stage: str, parameter: str
    ) -> dict[str, Any]:
        candidate_payload = _geometry_config_payload(candidate)
        key = tuple(sorted(candidate_payload.items()))
        config_id = geometry_config_id(candidate)
        if key in cache:
            _report_progress(
                progress_reporter,
                f"{stage}_candidate_cached",
                parameter=parameter,
                config_id=config_id,
            )
        else:
            restored_result = checkpoint_state["completed_configs"].get(config_id)
            if restored_result is not None:
                if restored_result.get("config") != _json_ready(candidate_payload):
                    raise ValueError(
                        "geometry-search checkpoint config id collision or invalid result"
                    )
                cache[key] = _restore_checkpoint_result(restored_result)
                _report_progress(
                    progress_reporter,
                    f"{stage}_candidate_resume",
                    parameter=parameter,
                    config_id=config_id,
                )
            else:
                _report_progress(
                    progress_reporter,
                    f"{stage}_candidate_start",
                    parameter=parameter,
                    config_id=config_id,
                )
                cache[key] = evaluate_geometry_config(
                    ordered,
                    feature_columns,
                    effective_model_params,
                    candidate,
                    columns=columns,
                    folds=selection_folds,
                    predictor=predictor,
                    prepared_feature_matrix=prepared_feature_matrix,
                    max_train_rows_per_fold=max_train_rows_per_fold,
                )
                checkpoint_state["completed_configs"][config_id] = _checkpoint_result(
                    cache[key]
                )
                persist_checkpoint(f"{stage}:{parameter}:{config_id}")
                _report_progress(
                    progress_reporter,
                    f"{stage}_candidate_complete",
                    parameter=parameter,
                    config_id=config_id,
                    selection_score=cache[key]["summary"]["selection_score"],
                )
        return cache[key]

    current = incumbent
    baseline = evaluate(incumbent, stage="baseline", parameter="incumbent")
    rows = []
    all_results = [baseline]
    one_dimensional: dict[str, list[dict[str, Any]]] = {}
    for parameter, values in geometry_grid.items():
        candidates = [
            evaluate(
                replace(current, **{parameter: value}),
                stage="one_dimensional",
                parameter=parameter,
            )
            for value in values
        ]
        all_results.extend(candidates)
        chosen = stable_plateau_select(candidates, score_tolerance=score_tolerance)
        one_dimensional[parameter] = candidates
        current = PathGeometryConfig(**chosen["config"])
        rows.extend(
            _result_row("one_dimensional", parameter, item) for item in candidates
        )
    best_two_values = reduced_joint_best_two_values(one_dimensional)
    joint_trials = reduced_joint_design(current, best_two_values, max_joint_trials)
    joint_results = [
        evaluate(candidate, stage="joint", parameter="cross_parameter")
        for candidate in joint_trials
    ]
    if joint_results:
        all_results.extend(joint_results)
        current = PathGeometryConfig(
            **stable_plateau_select(joint_results, score_tolerance=score_tolerance)[
                "config"
            ]
        )
        rows.extend(
            _result_row("sampled_reduced_joint", "cross_parameter", item)
            for item in joint_results
        )
    finalists = sorted(
        {tuple(sorted(item["config"].items())): item for item in all_results}.values(),
        key=_rank_key,
    )[:5]
    finalist_prediction_results: list[dict[str, Any]] = []
    if capture_predictions and run_post_search_refits:
        if len(finalists) != 5:
            raise ValueError(
                "prediction capture requires exactly five unique finalist configs"
            )
        for rank, finalist in enumerate(finalists, start=1):
            finalist_config = PathGeometryConfig(**finalist["config"])
            config_id = geometry_config_id(finalist_config)
            restored_capture = checkpoint_state["finalist_captures"].get(config_id)
            restored_diagnostics = (
                restored_capture.get("diagnostics")
                if isinstance(restored_capture, Mapping)
                else None
            )
            if (
                restored_capture is not None
                and isinstance(restored_diagnostics, Mapping)
                and set(_FINALIST_DIAGNOSTIC_KEYS).issubset(restored_diagnostics)
            ):
                _report_progress(
                    progress_reporter,
                    "finalist_capture_resume",
                    rank=rank,
                    config_id=config_id,
                )
                captured_entry = dict(restored_capture)
                if captured_entry.get("config") != _json_ready(
                    _geometry_config_payload(finalist_config)
                ):
                    raise ValueError(
                        "geometry finalist checkpoint sidecar has a mismatched config"
                    )
                if checkpoint_path is None:
                    raise ValueError(
                        "geometry finalist checkpoint sidecar cannot resume without checkpoint_path"
                    )
                captured_entry["predictions"] = _load_finalist_sidecar(
                    captured_entry, Path(checkpoint_path)
                )
                captured_entry["diagnostics"] = {
                    key: pd.DataFrame(restored_diagnostics[key])
                    for key in _FINALIST_DIAGNOSTIC_KEYS
                }
            else:
                _report_progress(
                    progress_reporter,
                    "finalist_capture_start"
                    if restored_capture is None
                    else "finalist_capture_diagnostics_refresh",
                    rank=rank,
                    config_id=config_id,
                )
                captured = evaluate_geometry_config(
                    ordered,
                    feature_columns,
                    effective_model_params,
                    finalist_config,
                    columns=columns,
                    folds=selection_folds,
                    predictor=predictor,
                    capture_predictions=True,
                    prepared_feature_matrix=prepared_feature_matrix,
                    max_train_rows_per_fold=max_train_rows_per_fold,
                )
                captured_entry = {
                    "rank": rank,
                    "config_id": config_id,
                    "config": _geometry_config_payload(finalist_config),
                    "summary": captured["summary"],
                    "predictions": captured["oos_predictions"],
                    "diagnostics": {
                        key: captured[key].copy() for key in _FINALIST_DIAGNOSTIC_KEYS
                    },
                }
                if checkpoint_path is not None:
                    checkpoint_state["finalist_captures"][config_id] = (
                        _atomic_write_finalist_sidecar(
                            Path(checkpoint_path),
                            config_id,
                            captured_entry["predictions"],
                            config=captured_entry["config"],
                            summary=captured_entry["summary"],
                            diagnostics=captured_entry["diagnostics"],
                        )
                    )
                    persist_checkpoint(f"finalist_capture:{rank}:{config_id}")
                _report_progress(
                    progress_reporter,
                    "finalist_capture_complete",
                    rank=rank,
                    config_id=config_id,
                )
            captured_entry["rank"] = rank
            finalist_prediction_results.append(captured_entry)

    def save_nested(outer_fold_id: int, entry: Mapping[str, Any]) -> None:
        checkpoint_state["nested_outer_folds"][str(outer_fold_id)] = _json_ready(entry)
        persist_checkpoint(f"nested_outer_fold:{outer_fold_id}")

    def save_nested_evaluation(evaluation_key: str, result: Mapping[str, Any]) -> None:
        checkpoint_state["nested_completed_evaluations"][evaluation_key] = (
            _checkpoint_result(result)
        )
        persist_checkpoint(f"nested_evaluation:{evaluation_key}")

    nested = (
        nested_finalist_validation(
            ordered,
            feature_columns,
            effective_model_params,
            finalists,
            columns=columns,
            predictor=predictor,
            prepared_feature_matrix=prepared_feature_matrix,
            max_train_rows_per_fold=max_train_rows_per_fold,
            completed_outer_folds=checkpoint_state["nested_outer_folds"],
            completed_evaluations=checkpoint_state["nested_completed_evaluations"],
            on_outer_fold_complete=save_nested,
            on_evaluation_complete=save_nested_evaluation,
            progress_reporter=progress_reporter,
        )
        if nested_oof and run_post_search_refits
        else []
    )
    selected_result = evaluate(current, stage="selected", parameter="final")
    fold_reports = pd.concat(
        [
            result["folds"].assign(
                config_id=geometry_config_id(PathGeometryConfig(**result["config"])),
                **{f"config_{key}": value for key, value in result["config"].items()},
            )
            for result in cache.values()
        ],
        ignore_index=True,
    )
    selected_config_id = geometry_config_id(current)
    selected_capture = next(
        (
            item
            for item in finalist_prediction_results
            if str(item["config_id"]) == selected_config_id
        ),
        None,
    )

    def selected_diagnostic(name: str) -> pd.DataFrame:
        direct = selected_result.get(name)
        if isinstance(direct, pd.DataFrame) and not direct.empty:
            return direct
        if selected_capture is not None:
            captured = selected_capture.get("diagnostics", {}).get(name)
            if isinstance(captured, pd.DataFrame):
                return captured
        return pd.DataFrame()

    result = {
        "incumbent": asdict(incumbent),
        "selected": asdict(current),
        "fixed_feature_columns": list(feature_columns),
        "fixed_model_params": effective_model_params,
        "sweep_results": pd.DataFrame(rows),
        "fold_reports": fold_reports,
        "selected_fold_reports": selected_result["folds"].copy(),
        "finalists": finalists,
        "nested_oof": nested,
        "finalist_oos_predictions": finalist_prediction_results,
        "reduced_joint_best_two_values": {
            name: list(values) for name, values in best_two_values.items()
        },
        "boundary": boundary_diagnostics(
            label_path_geometry(ordered, current, columns)
        ),
        "temporal_month_stability": selected_result["temporal_month_stability"],
        "side_stability": selected_result["side_stability"],
        "symbol_stability": selected_result["symbol_stability"],
        "side_support": selected_result["side_support"],
        "symbol_support": selected_result["symbol_support"],
        "selected_side_diagnostics": selected_diagnostic("side_diagnostics"),
        "selected_month_diagnostics": selected_diagnostic("month_diagnostics"),
        "selected_probability_reliability_bins": selected_diagnostic(
            "probability_reliability_bins"
        ),
        "selected_economic_confusion": selected_diagnostic("economic_confusion"),
        "selected_economic_confusion_priors": selected_diagnostic(
            "economic_confusion_priors"
        ),
        "search_contract": {
            "evaluation_split": {
                "name": (
                    "purged_chronological_development_only"
                    if evaluation_mode == GEOMETRY_EVALUATION_MODE_SHORT_HISTORY
                    else "4_month_train_4_month_oos"
                ),
                "train_months": GEOMETRY_TRAIN_MONTHS,
                "oos_months": GEOMETRY_OOS_MONTHS,
                "walk_forward_cadence_months": GEOMETRY_TRAIN_MONTHS,
                "nested_minimum_months": GEOMETRY_NESTED_MONTHS,
                "nested_outer_oos_months": GEOMETRY_OOS_MONTHS,
                "oos_row_contract": "all_labelled_oos_rows",
                "default_max_train_rows_per_fold": DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
                "evaluation_mode": evaluation_mode,
                "short_history_development_end": (
                    str(short_history_development_end)
                    if short_history_development_end is not None
                    else None
                ),
                "short_history_subfold_count": int(short_history_subfold_count),
            },
            "cost_return": PATH_GEOMETRY_COST_RETURN,
            "model_hpo": False,
            "joint_strategy": "best_two_per_parameter_ring_pair",
            "max_joint_trials": max_joint_trials,
            "selection": "combined_learnability_economic_score",
            "ablation_start_date": str(ablation_start_date)
            if ablation_start_date is not None
            else None,
            "selection_fold_count": len(selection_folds),
            "capture_predictions": capture_predictions,
            "run_post_search_refits": run_post_search_refits,
            "prediction_capture_scope": (
                "top_5_finalists_only"
                if capture_predictions and run_post_search_refits
                else "disabled_for_this_run"
            ),
            "feature_matrix_materialization": "once_per_staged_search",
            "train_row_sampling": {
                "requested_train_rows_per_fold": max_train_rows_per_fold,
                "effective_train_rows": "per_candidate_fold_report",
                "validation_rows": "full_oos_no_subsampling",
                "stratification": "side_x_dynamic_geometry_class",
                "selection": "deterministic_chronological_time_spread",
                "disabled_when_requested_rows_is_zero": True,
            },
            "catboost_early_stopping": {
                "contract": (
                    "chronological internal validation tail within sampled outer train; "
                    "purge label_end overlap and apply 24h embargo; refit all sampled "
                    "outer-train rows at the selected fixed tree count; outer OOS never used"
                ),
                "validation_fraction": GEOMETRY_EARLY_STOP_VALIDATION_FRACTION,
                "embargo_hours": float(
                    GEOMETRY_EARLY_STOP_EMBARGO / pd.Timedelta(hours=1)
                ),
                "tree_count_reporting": "per_candidate_fold_report",
                "non_iteration_hpo_params": "frozen_from_classifier_contract",
            },
            "raw_probability_diagnostics": {
                "transform": "none",
                "metrics": "logloss_multiclass_brier_classwise_ece_macro_ece_reliability_bins",
            },
            "economic_confusion": {
                "penalty": (
                    "absolute_difference_between_train_only_class_reference_geometry_net_ev_priors; "
                    "downstream_execution_policy_reports_use_train_only_optimized_execution_ev"
                ),
                "class_ev_source": "train_label_net_ev_after_fixed_1pct_cost",
                "reported_prediction_variants": ["raw"],
            },
            "catboost_resource_contract": catboost_resource,
            "checkpoint": {
                "schema": _CHECKPOINT_SCHEMA,
                "path": str(checkpoint_path) if checkpoint_path is not None else None,
                "fingerprint": checkpoint_fingerprint,
                "completed_config_count": len(checkpoint_state["completed_configs"]),
            },
        },
    }
    if checkpoint_path is not None:
        checkpoint_state["status"] = "complete"
        checkpoint_state["selected_config"] = asdict(current)
        checkpoint_state["last_checkpoint_reason"] = "complete"
        _atomic_json_write(Path(checkpoint_path), checkpoint_state)
        _report_progress(
            progress_reporter,
            "checkpoint_complete",
            checkpoint_path=str(checkpoint_path),
        )
    return result
