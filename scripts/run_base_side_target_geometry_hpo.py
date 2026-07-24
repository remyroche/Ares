#!/usr/bin/env python3
"""Hierarchical, train-only base target/geometry/sample-weight HPO.

This helper is deliberately downstream of the A/B feature-selection arms.  It
loads the winning arm's cached fixed-window payload, so its training matrix,
feature contract, and frozen AE/GMM representation are already fixed.  The
only search dimensions are side-specific continuous target geometry and
target-strength training weights.

The Apr-Jun 2026 rows stored in the source payload are never used for model or
target selection.  They are scored once after the winning long/short contracts
have been frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.hierarchical_label_weights import (
    TARGET_EXPONENT_GRID,
    TargetStrengthWeightSpec,
    build_target_strength_weights,
)
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import _load_fold_payload

try:
    from numba import njit
except Exception:  # pragma: no cover - exercised only where numba is unavailable.
    njit = None

try:
    from lightgbm import LGBMRegressor, early_stopping
except Exception:  # pragma: no cover - focused tests do not require LightGBM.
    LGBMRegressor = None
    early_stopping = None


def _patch_lightgbm_sklearn_validation_compat() -> None:
    """Adapt LightGBM 4.3's sklearn wrapper to sklearn 1.8 argument names."""

    if LGBMRegressor is None:
        return
    try:
        import inspect

        import lightgbm.compat as lgb_compat
        import lightgbm.sklearn as lgb_sklearn
        from sklearn.utils.validation import check_X_y, check_array
    except Exception:
        return
    if "force_all_finite" in inspect.signature(check_X_y).parameters:
        return

    def compat_check_xy(X: Any, y: Any, accept_sparse: Any = False, **kwargs: Any) -> Any:
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return check_X_y(X, y, accept_sparse=accept_sparse, **kwargs)

    def compat_check_array(array: Any, accept_sparse: Any = False, **kwargs: Any) -> Any:
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return check_array(array, accept_sparse=accept_sparse, **kwargs)

    lgb_compat._LGBMCheckXY = compat_check_xy
    lgb_sklearn._LGBMCheckXY = compat_check_xy
    lgb_compat._LGBMCheckArray = compat_check_array
    lgb_sklearn._LGBMCheckArray = compat_check_array


_patch_lightgbm_sklearn_validation_compat()


TOP_FRACS = (0.10, 0.20, 0.30)
OBJECTIVE_PROFILES: dict[str, dict[str, Any]] = {
    "net_global_balanced": {
        "outcome_basis": "candidate_geometry_net",
        "top_weights": {0.10: 0.50, 0.20: 0.30, 0.30: 0.20},
        "std_penalty": 0.25,
        "worst_weight": 0.15,
    },
    "net_global_topheavy": {
        "outcome_basis": "candidate_geometry_net",
        "top_weights": {0.10: 0.65, 0.20: 0.25, 0.30: 0.10},
        "std_penalty": 0.25,
        "worst_weight": 0.15,
    },
    "net_global_stable": {
        "outcome_basis": "candidate_geometry_net",
        "top_weights": {0.10: 0.50, 0.20: 0.30, 0.30: 0.20},
        "std_penalty": 0.50,
        "worst_weight": 0.25,
    },
    "canonical_net_global_balanced": {
        "outcome_basis": "canonical_corrected_label_net",
        "top_weights": {0.10: 0.50, 0.20: 0.30, 0.30: 0.20},
        "std_penalty": 0.25,
        "worst_weight": 0.15,
    },
    "canonical_net_global_topheavy": {
        "outcome_basis": "canonical_corrected_label_net",
        "top_weights": {0.10: 0.65, 0.20: 0.25, 0.30: 0.10},
        "std_penalty": 0.25,
        "worst_weight": 0.15,
    },
    "canonical_net_global_stable": {
        "outcome_basis": "canonical_corrected_label_net",
        "top_weights": {0.10: 0.50, 0.20: 0.30, 0.30: 0.20},
        "std_penalty": 0.50,
        "worst_weight": 0.25,
    },
}
DEFAULT_TRAIN_START = pd.Timestamp("2025-04-01T00:00:00Z")
DEFAULT_TRAIN_END = pd.Timestamp("2026-04-01T00:00:00Z")
DEFAULT_PURGE_HOURS = 25.0
DEFAULT_INTERNAL_VALIDATION_MONTHS = ("2025-12", "2026-01", "2026-02", "2026-03")


@dataclass(frozen=True)
class SideTargetGeometry:
    """Soft target parameters around an exactly replayed supported geometry."""

    tp_r: float
    sl_r: float
    max_profit_bars: int
    net_edge: float
    temperature: float
    mae_penalty: float
    timeout_penalty: float
    slow_profit_bars: float
    first_pass_penalty: float
    # Short-only: demote weak, late continuation shorts when pre-entry market
    # state already signals exhaustion/rebound risk. Zero keeps the incumbent
    # target exactly unchanged (including the long contract).
    late_continuation_penalty: float = 0.0


@dataclass(frozen=True)
class SideWeightConfig:
    target_exponent: float
    weight_range_ratio: float


@dataclass(frozen=True)
class CandidateConfig:
    side: str
    geometry: SideTargetGeometry
    weight: SideWeightConfig


@dataclass(frozen=True)
class PathPrimitives:
    """Exact causal first-passage inputs shared by every Stage-C trial.

    The label artifact only records first passage at the supported R grids. A
    Stage-C trial therefore must select from those grids rather than inventing
    a continuous TP/SL outcome that cannot be reconstructed exactly.
    """

    barrier_pct: np.ndarray
    timeout_gross_return: np.ndarray
    round_trip_cost: np.ndarray
    incumbent_timeout: np.ndarray
    baseline_tp_r: np.ndarray
    baseline_sl_r: np.ndarray
    bars_to_mfe_grid: np.ndarray
    bars_to_mae_grid: np.ndarray


TP_R_GRID: tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)
SL_R_GRID: tuple[float, ...] = (0.50, 0.75, 1.00, 1.50)
MAX_PROFIT_BARS_GRID: tuple[int, ...] = (8, 12, 16, 24, 32)
SHORT_MAX_PROFIT_BARS_GRID: tuple[int, ...] = (4, 6, 8, 12, 16, 24, 32)

# These are all causal, portable state features. Positive values mean price is
# stabilising/recovering after a downside move or that selling pressure is
# decelerating. They are deliberately not realized outcomes and are normalized
# from each fold's training rows only below.
SHORT_LATE_CONTINUATION_FEATURES: tuple[str, ...] = (
    "downside_deceleration_8h_rz",
    "price_minus_oi_recovery_72h",
    "price_recovery_fraction_24h",
    "asset_minus_mkt_oi_recovery_fraction_24h",
    "climax_decay",
)


def _required_float_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Load an exact stored field, failing closed when the path contract is absent."""

    if column not in frame.columns:
        raise ValueError(f"Stage-C exact geometry requires cached label column {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Stage-C exact geometry requires finite cached label column {column}")
    return values.astype(np.float32, copy=False)


def _event_bar_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Load first-passage bars, normalizing an unhit NaN to the -1 sentinel."""

    if column not in frame.columns:
        raise ValueError(f"Stage-C exact geometry requires cached label column {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if np.isinf(values).any():
        raise ValueError(f"Stage-C exact geometry requires finite-or-unhit cached label column {column}")
    values = np.where(np.isnan(values), -1.0, values)
    if np.any((values != -1.0) & (values < 1.0)):
        raise ValueError(f"Stage-C exact geometry found invalid event bar in {column}")
    return values.astype(np.float32, copy=False)


def build_path_primitives(metrics: pd.DataFrame) -> PathPrimitives:
    """Load the label artifact's exact causal first-passage primitives once.

    ``__y_ret__`` is the artifact's causal timeout/terminal return.  It is
    used only when the candidate supported geometry has neither a TP nor an SL
    first passage. TP and SL outcomes are rebuilt exactly as ``+/- R *
    __barrier_pct__``. Cost remains separate and is subtracted once only in
    diagnostic reporting.
    """

    barrier = _required_float_column(metrics, "__barrier_pct__")
    if np.any(barrier <= 0.0):
        raise ValueError("Stage-C exact geometry requires positive __barrier_pct__")
    timeout_net = _required_float_column(metrics, "__y_ret__")
    cost = _required_float_column(metrics, "__first_touch_round_trip_cost__")
    if not np.allclose(cost, 0.01, rtol=0.0, atol=1e-8):
        raise ValueError(
            "Stage-C gross/net diagnostics require the causal label artifact's "
            "single 1% round-trip cost contract"
        )
    baseline_tp = _required_float_column(metrics, "__first_touch_effective_tp_abs__")
    baseline_sl = _required_float_column(metrics, "__first_touch_effective_sl_abs__")
    timeout = _required_float_column(metrics, "__is_timeout__")
    bars_to_mfe = np.column_stack(
        [_event_bar_column(metrics, f"__bars_to_mfe_{key}r__") for key in ("05", "075", "1", "125", "15")]
    ).astype(np.float32, copy=False)
    bars_to_mae = np.column_stack(
        [_event_bar_column(metrics, f"__bars_to_mae_{key}r__") for key in ("05", "075", "1", "15")]
    ).astype(np.float32, copy=False)
    return PathPrimitives(
        barrier_pct=barrier,
        timeout_gross_return=(timeout_net + cost).astype(np.float32, copy=False),
        round_trip_cost=cost,
        incumbent_timeout=timeout,
        baseline_tp_r=(baseline_tp / barrier).astype(np.float32, copy=False),
        baseline_sl_r=(baseline_sl / barrier).astype(np.float32, copy=False),
        bars_to_mfe_grid=bars_to_mfe,
        bars_to_mae_grid=bars_to_mae,
    )


def _grid_index(value: float, grid: Sequence[float]) -> int:
    for index, candidate in enumerate(grid):
        if math.isclose(float(value), float(candidate), rel_tol=0.0, abs_tol=1e-9):
            return index
    raise ValueError(f"Unsupported exact first-passage geometry value={value}; grid={tuple(grid)}")


def _geometry_outcomes_numpy(
    barrier_pct: np.ndarray,
    timeout_gross_return: np.ndarray,
    bars_to_mfe_grid: np.ndarray,
    bars_to_mae_grid: np.ndarray,
    tp_r: float,
    sl_r: float,
    max_profit_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay supported TP/SL first touch from cached causal path primitives.

    Same-bar TP/SL ambiguity is intentionally resolved to the SL. The cached
    first-passage fields do not expose a threshold-specific intrabar order, and
    a conservative tie is preferable to pretending an optimistic exact path.
    """

    tp_idx = _grid_index(tp_r, TP_R_GRID)
    sl_idx = _grid_index(sl_r, SL_R_GRID)
    tp_bar = bars_to_mfe_grid[:, tp_idx].astype(np.float64, copy=False)
    sl_bar = bars_to_mae_grid[:, sl_idx].astype(np.float64, copy=False)
    max_bars = int(max_profit_bars)
    tp_hit = np.isfinite(tp_bar) & (tp_bar > 0.0) & (tp_bar <= max_bars)
    sl_hit = np.isfinite(sl_bar) & (sl_bar > 0.0)
    tp_first = tp_hit & (~sl_hit | (tp_bar < sl_bar))
    sl_first = sl_hit & ~tp_first
    timeout = ~(tp_first | sl_first)
    gross = timeout_gross_return.astype(np.float64, copy=True)
    gross[tp_first] = float(tp_r) * barrier_pct[tp_first]
    gross[sl_first] = -float(sl_r) * barrier_pct[sl_first]
    resolved_bars = np.where(tp_first, tp_bar, np.where(sl_first, sl_bar, float(max_bars)))
    return (
        gross.astype(np.float32, copy=False),
        tp_first.astype(np.float32, copy=False),
        sl_first.astype(np.float32, copy=False),
        timeout.astype(np.float32, copy=False),
        resolved_bars.astype(np.float32, copy=False),
    )


def geometry_outcomes(
    primitives: PathPrimitives,
    geometry: SideTargetGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact gross outcome, TP, SL, timeout and resolved-bar arrays."""

    return _geometry_outcomes_numpy(
        primitives.barrier_pct,
        primitives.timeout_gross_return,
        primitives.bars_to_mfe_grid,
        primitives.bars_to_mae_grid,
        float(geometry.tp_r),
        float(geometry.sl_r),
        int(geometry.max_profit_bars),
    )


def _target_kernel_numpy(
    net_outcome: np.ndarray,
    tp_hit: np.ndarray,
    sl_hit: np.ndarray,
    timeout: np.ndarray,
    resolved_bars: np.ndarray,
    tp_r: float,
    sl_r: float,
    net_edge: float,
    temperature: float,
    mae_penalty: float,
    timeout_penalty: float,
    slow_profit_bars: float,
    first_pass_penalty: float,
    late_continuation_pressure: np.ndarray,
    late_continuation_penalty: float,
) -> np.ndarray:
    temp = max(float(temperature), 1e-4)
    edge = 1.0 / (1.0 + np.exp(-np.clip((net_outcome - net_edge) / temp, -50.0, 50.0)))
    speed = np.exp(-np.maximum(resolved_bars, 0.0) / max(float(slow_profit_bars), 1.0))
    path_multiplier = (
        0.50
        + 0.20 * np.clip(tp_hit, 0.0, 1.0)
        + 0.20 * speed
        - float(mae_penalty) * 0.25 * np.clip(sl_hit, 0.0, 1.0)
        - float(timeout_penalty) * 0.20 * np.clip(timeout, 0.0, 1.0)
        - float(first_pass_penalty) * 0.10 * np.clip(sl_hit, 0.0, 1.0)
    )
    # A late-continuation state must not erase a genuinely strong, fast
    # executable short. It discounts only weak net-edge rows in an observable
    # exhaustion/recovery state.
    weak_edge = 1.0 - edge
    continuation_discount = (
        float(late_continuation_penalty)
        * np.clip(late_continuation_pressure, 0.0, 1.0)
        * np.clip(weak_edge, 0.0, 1.0)
    )
    path_multiplier *= np.clip(1.0 - continuation_discount, 0.0, 1.0)
    return np.clip(edge * np.clip(path_multiplier, 0.0, 1.5), 0.0, 1.0).astype(np.float32)


if njit is not None:
    _target_kernel_numba = njit(cache=True, fastmath=False)(_target_kernel_numpy)
else:  # pragma: no cover
    _target_kernel_numba = _target_kernel_numpy


def continuous_target(
    primitives: PathPrimitives,
    geometry: SideTargetGeometry,
    *,
    late_continuation_pressure: np.ndarray | None = None,
) -> np.ndarray:
    """Return a soft target around exact first-touch net outcome after one stored cost."""

    gross, tp_hit, sl_hit, timeout, resolved_bars = geometry_outcomes(primitives, geometry)
    net = gross - primitives.round_trip_cost
    pressure = (
        np.zeros(len(net), dtype=np.float32)
        if late_continuation_pressure is None
        else np.asarray(late_continuation_pressure, dtype=np.float32)
    )
    if pressure.shape != net.shape:
        raise ValueError("late_continuation_pressure must align to path primitives")
    return _target_kernel_numba(
        net.astype(np.float64, copy=False),
        tp_hit.astype(np.float64, copy=False),
        sl_hit.astype(np.float64, copy=False),
        timeout.astype(np.float64, copy=False),
        resolved_bars.astype(np.float64, copy=False),
        float(geometry.tp_r),
        float(geometry.sl_r),
        float(geometry.net_edge),
        float(geometry.temperature),
        float(geometry.mae_penalty),
        float(geometry.timeout_penalty),
        float(geometry.slow_profit_bars),
        float(geometry.first_pass_penalty),
        pressure.astype(np.float64, copy=False),
        float(geometry.late_continuation_penalty),
    )


def short_late_continuation_pressure(
    features: pd.DataFrame,
    *,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build a fold-fitted, observable short exhaustion/rebound pressure.

    Robust centering/scaling is fit only on each fold's train rows. Missing
    features are ignored, but at least one declared signal must be available;
    silently emitting an all-zero short target modifier would make an ablation
    claim meaningless.
    """
    available = tuple(name for name in SHORT_LATE_CONTINUATION_FEATURES if name in features.columns)
    if not available:
        raise ValueError(
            "Short late-continuation target requires at least one observable "
            f"state feature from {SHORT_LATE_CONTINUATION_FEATURES}"
        )
    fit = features.iloc[np.asarray(fit_indices, dtype=np.int64)]
    columns: list[np.ndarray] = []
    used: list[str] = []
    for name in available:
        train = pd.to_numeric(fit[name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite = train[np.isfinite(train)]
        if finite.size < 64:
            continue
        median = float(np.median(finite))
        iqr = float(np.subtract(*np.percentile(finite, [75.0, 25.0])))
        scale = max(iqr, 1e-6)
        values = pd.to_numeric(features[name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        z = np.clip((np.where(np.isfinite(values), values, median) - median) / scale, -8.0, 8.0)
        # Sigmoid makes the composite bounded and less sensitive to one
        # exceptional feature. Positive state values are exhaustion pressure.
        columns.append((1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False))
        used.append(name)
    if not columns:
        raise ValueError("Short late-continuation target has no sufficiently supported state feature")
    return np.mean(np.column_stack(columns), axis=1, dtype=np.float32), tuple(used)


def _side_name(frame: pd.DataFrame) -> np.ndarray:
    raw = frame.get("side_name", frame.get("side", frame.get("__side__", 1.0)))
    values = pd.Series(raw, index=frame.index)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.lower()
    return np.where(text.str.contains("short", regex=False).to_numpy() | numeric.lt(0.0).fillna(False).to_numpy(), "short", "long")


def _archetype(frame: pd.DataFrame) -> pd.Series:
    for column in (
        "__archetype_label_family__",
        "archetype_label_family",
        "policy_archetype",
        "local_side_archetype",
        "source_archetype",
    ):
        if column in frame.columns:
            return frame[column].fillna("__missing__").astype(str)
    return pd.Series("__missing__", index=frame.index, dtype="string")


def build_internal_chronological_folds(
    frame: pd.DataFrame,
    *,
    purge_hours: float = DEFAULT_PURGE_HOURS,
    validation_months: Sequence[str] = DEFAULT_INTERNAL_VALIDATION_MONTHS,
    min_train_rows: int = 1000,
    min_valid_rows: int = 250,
) -> list[dict[str, np.ndarray]]:
    """Build purged folds strictly inside Apr-2025 through Mar-2026.

    All target/weight/model decisions are therefore made before the fixed
    Apr-Jun 2026 OOS window.  Train rows end at least one corrected label path
    before each validation boundary.
    """

    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Stage-C source payload has invalid timestamps")
    if ts.min() < DEFAULT_TRAIN_START or ts.max() >= DEFAULT_TRAIN_END:
        raise ValueError("Stage-C internal folds require an Apr-2025 through Mar-2026 training payload")
    folds: list[dict[str, np.ndarray]] = []
    for month in validation_months:
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        cutoff = start - pd.Timedelta(hours=float(purge_hours))
        train = np.flatnonzero(ts.lt(cutoff).to_numpy())
        valid = np.flatnonzero((ts.ge(start) & ts.lt(end)).to_numpy())
        if len(train) < int(min_train_rows) or len(valid) < int(min_valid_rows):
            continue
        folds.append({"month": str(month), "train_idx": train, "valid_idx": valid})
    if len(folds) < 2:
        raise RuntimeError("Stage-C needs at least two non-empty internal chronological validation folds")
    return folds


def _topk_net_objective(
    score: np.ndarray,
    gross_outcome: np.ndarray,
    round_trip_cost: np.ndarray,
    indices: np.ndarray,
    *,
    top_weights: Mapping[float, float],
) -> tuple[float, dict[str, float]]:
    """Net top-k objective using one global ordering within the evaluated side."""

    values = np.asarray(score, dtype=np.float64)
    gross = np.asarray(gross_outcome, dtype=np.float64)[indices]
    cost = np.asarray(round_trip_cost, dtype=np.float64)[indices]
    if values.shape[0] != gross.shape[0]:
        raise ValueError("Stage-C score/outcome alignment failure")
    net = gross - cost
    order = np.argsort(-values, kind="stable")
    rows: dict[float, float] = {}
    for frac in TOP_FRACS:
        count = max(1, int(math.ceil(len(order) * float(frac))))
        rows[float(frac)] = float(np.mean(net[order[:count]])) if count else float("-inf")
    objective = sum(float(top_weights[frac]) * rows[frac] for frac in TOP_FRACS)
    metrics = {f"net_top{int(frac * 100)}": value for frac, value in rows.items()}
    metrics["net_all"] = float(np.mean(net))
    metrics["net_top10_lift_vs_all"] = float(rows[0.10] - metrics["net_all"])
    return float(objective), metrics


def _topk_outcome_diagnostics(
    *,
    score: np.ndarray,
    frame: pd.DataFrame,
    gross_outcome: np.ndarray,
    round_trip_cost: np.ndarray,
    indices: np.ndarray,
) -> pd.DataFrame:
    """Report global gross and once-cost-subtracted net outcomes."""

    view = frame.iloc[indices].reset_index(drop=True)
    order = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(view["__ts__"], utc=True, errors="coerce"),
            "score": np.asarray(score, dtype=np.float64),
            "gross_return": np.asarray(gross_outcome, dtype=np.float64)[indices],
            "round_trip_cost": np.asarray(round_trip_cost, dtype=np.float64)[indices],
        }
    ).sort_values(["score", "__ts__"], ascending=[False, True], kind="mergesort")
    rows: list[dict[str, float]] = []
    for frac in TOP_FRACS:
        selected = order.iloc[: max(1, int(math.ceil(len(order) * float(frac))))]
        rows.append(
            {
                "top_fraction": float(frac),
                "selected_rows": int(len(selected)),
                "gross_return_per_trade": float(selected["gross_return"].mean()),
                "net_return_per_trade": float((selected["gross_return"] - selected["round_trip_cost"]).mean()),
                "sum_net_return": float((selected["gross_return"] - selected["round_trip_cost"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _proxy_predict(
    x_train: np.ndarray,
    target_train: np.ndarray,
    weights: np.ndarray,
    x_valid: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=96,
        max_depth=8,
        min_samples_leaf=45,
        max_features="sqrt",
        n_jobs=2,
        random_state=int(seed),
    )
    model.fit(x_train, target_train, sample_weight=weights)
    return model.predict(x_valid).astype(np.float32, copy=False)


def _full_lgbm_predict(
    x_train: np.ndarray,
    target_train: np.ndarray,
    weights: np.ndarray,
    x_valid: np.ndarray,
    target_valid: np.ndarray,
    *,
    params: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    if LGBMRegressor is None:
        raise RuntimeError("LightGBM is required for Stage-C finalists")
    model = LGBMRegressor(
        objective=str(params.get("objective", "regression_l2")),
        n_estimators=int(params.get("n_estimators", 500)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        num_leaves=int(params.get("num_leaves", 15)),
        max_depth=int(params.get("max_depth", 4)),
        min_child_samples=int(params.get("min_child_samples", 80)),
        subsample=float(params.get("subsample", 0.85)),
        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 3.0)),
        min_split_gain=float(params.get("min_split_gain", 1e-3)),
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    callbacks = [early_stopping(45, verbose=False)] if early_stopping is not None else None
    model.fit(
        x_train,
        target_train,
        sample_weight=weights,
        eval_set=[(x_valid, target_valid)],
        eval_metric="l2",
        callbacks=callbacks,
    )
    return model.predict(x_valid).astype(np.float32, copy=False)


def _full_lgbm_fit_predict_oos(
    x_train: np.ndarray,
    target_train: np.ndarray,
    weights: np.ndarray,
    x_oos: np.ndarray,
    *,
    params: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    """Fit once on the full fixed train set without consuming OOS labels."""

    if LGBMRegressor is None:
        raise RuntimeError("LightGBM is required for Stage-C final OOS scoring")
    model = LGBMRegressor(
        objective=str(params.get("objective", "regression_l2")),
        n_estimators=int(params.get("n_estimators", 500)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        num_leaves=int(params.get("num_leaves", 15)),
        max_depth=int(params.get("max_depth", 4)),
        min_child_samples=int(params.get("min_child_samples", 80)),
        subsample=float(params.get("subsample", 0.85)),
        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 3.0)),
        min_split_gain=float(params.get("min_split_gain", 1e-3)),
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(x_train, target_train, sample_weight=weights)
    return model.predict(x_oos).astype(np.float32, copy=False)


def _geometry_from_trial(
    trial: Any,
    *,
    side: str,
    enable_short_late_continuation: bool = True,
) -> SideTargetGeometry:
    is_short = str(side) == "short"
    return SideTargetGeometry(
        # The label artifact has exact first-passage timestamps only at these
        # R levels. Categorical search is therefore exact rather than a
        # continuous approximation dressed up as an exit replay.
        tp_r=float(trial.suggest_categorical("tp_r", list(TP_R_GRID))),
        sl_r=float(trial.suggest_categorical("sl_r", list(SL_R_GRID))),
        max_profit_bars=int(trial.suggest_categorical(
            "max_profit_bars",
            list(SHORT_MAX_PROFIT_BARS_GRID if is_short else MAX_PROFIT_BARS_GRID),
        )),
        # Short labels must clear a modestly stronger executable-net hurdle;
        # HPO may still choose a negative threshold where the 1% cost makes
        # the base problem otherwise too sparse.
        net_edge=float(trial.suggest_float("net_edge", -0.010 if is_short else -0.015, 0.020 if is_short else 0.012)),
        temperature=float(trial.suggest_float("temperature", 0.0025, 0.020, log=True)),
        mae_penalty=float(trial.suggest_float("mae_penalty", 0.35, 2.50)),
        timeout_penalty=float(trial.suggest_float("timeout_penalty", 0.0, 2.00)),
        slow_profit_bars=float(trial.suggest_float("slow_profit_bars", 3.0, 32.0)),
        first_pass_penalty=float(trial.suggest_float("first_pass_penalty", 0.0, 2.00)),
        late_continuation_penalty=float(
            trial.suggest_float("late_continuation_penalty", 0.0, 0.80)
            if is_short and enable_short_late_continuation
            else 0.0
        ),
    )


def _candidate_from_trial(
    trial: Any,
    side: str,
    *,
    geometry: SideTargetGeometry | None = None,
    enable_short_late_continuation: bool = True,
) -> CandidateConfig:
    return CandidateConfig(
        side=str(side),
        geometry=(
            geometry
            if geometry is not None
            else _geometry_from_trial(
                trial,
                side=side,
                enable_short_late_continuation=enable_short_late_continuation,
            )
        ),
        weight=SideWeightConfig(
            target_exponent=float(trial.suggest_categorical("target_exponent", list(TARGET_EXPONENT_GRID))),
            weight_range_ratio=float(trial.suggest_float("weight_range_ratio", 3.0, 12.0)),
        ),
    )


def _weights_for_fold(
    *,
    target: np.ndarray,
    frame: pd.DataFrame,
    indices: np.ndarray,
    config: SideWeightConfig,
) -> np.ndarray:
    subset = frame.iloc[indices]
    weights, _ = build_target_strength_weights(
        target[indices],
        timestamps=subset["__ts__"],
        archetypes=_archetype(subset),
        spec=TargetStrengthWeightSpec(
            exponent=float(config.target_exponent),
            weight_range_ratio=float(config.weight_range_ratio),
        ),
    )
    return weights.astype(np.float32, copy=False)


def _evaluate_candidate(
    *,
    candidate: CandidateConfig,
    x: np.ndarray,
    feature_frame: pd.DataFrame,
    frame: pd.DataFrame,
    primitives: PathPrimitives,
    folds: Sequence[Mapping[str, Any]],
    full_model: bool,
    lgbm_params: Mapping[str, Any],
    seed: int,
    objective_profile: Mapping[str, Any],
    trial: Any | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    sides = _side_name(frame)
    side_positions = np.flatnonzero(sides == candidate.side)
    if side_positions.size == 0:
        return float("-inf"), []
    geometry_gross_outcome, _, _, _, _ = geometry_outcomes(primitives, candidate.geometry)
    outcome_basis = str(objective_profile["outcome_basis"])
    if outcome_basis == "candidate_geometry_net":
        objective_gross_outcome = geometry_gross_outcome
    elif outcome_basis == "canonical_corrected_label_net":
        # timeout_gross_return is the corrected canonical label outcome plus
        # its one stored cost. Subtracting round_trip_cost in the objective
        # therefore recovers the fixed canonical net outcome exactly once.
        objective_gross_outcome = primitives.timeout_gross_return
    else:
        raise ValueError(f"Unsupported Stage-C outcome_basis={outcome_basis}")
    rows: list[dict[str, Any]] = []
    for fold_i, fold in enumerate(folds):
        train_idx = np.intersect1d(np.asarray(fold["train_idx"]), side_positions, assume_unique=False)
        valid_idx = np.intersect1d(np.asarray(fold["valid_idx"]), side_positions, assume_unique=False)
        if len(train_idx) < 1000 or len(valid_idx) < 200:
            continue
        pressure = None
        pressure_features: tuple[str, ...] = ()
        if candidate.side == "short" and candidate.geometry.late_continuation_penalty > 0.0:
            pressure, pressure_features = short_late_continuation_pressure(
                feature_frame, fit_indices=train_idx,
            )
        target = continuous_target(
            primitives,
            candidate.geometry,
            late_continuation_pressure=pressure,
        )
        weights = _weights_for_fold(target=target, frame=frame, indices=train_idx, config=candidate.weight)
        if full_model:
            prediction = _full_lgbm_predict(
                x[train_idx], target[train_idx], weights, x[valid_idx], target[valid_idx],
                params=lgbm_params, seed=seed + fold_i,
            )
        else:
            prediction = _proxy_predict(
                x[train_idx], target[train_idx], weights, x[valid_idx], seed=seed + fold_i,
            )
        objective, metric = _topk_net_objective(
            prediction,
            objective_gross_outcome,
            primitives.round_trip_cost,
            valid_idx,
            top_weights=objective_profile["top_weights"],
        )
        row = {
            "side": candidate.side,
            "month": str(fold["month"]),
            "objective": objective,
            "train_rows": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
            "full_lgbm": bool(full_model),
            "late_continuation_feature_count": int(len(pressure_features)),
            **metric,
        }
        rows.append(row)
        if trial is not None:
            trial.report(float(np.mean([item["objective"] for item in rows])), step=fold_i)
            if trial.should_prune():
                raise RuntimeError("__PRUNED__")
    if not rows:
        return float("-inf"), rows
    mean = float(np.mean([row["objective"] for row in rows]))
    std = float(np.std([row["objective"] for row in rows]))
    worst = float(np.min([row["objective"] for row in rows]))
    return float(
        mean
        - float(objective_profile["std_penalty"]) * std
        + float(objective_profile["worst_weight"]) * worst
    ), rows


FULL_PAYLOAD_KEYS = {"train", "valid", "valid_metrics", "x_train", "x_valid"}
COMPACT_MAIN_PAYLOAD_KEYS = {
    "train_side",
    "train_target",
    "valid",
    "valid_metrics",
    "x_train",
    "x_valid",
}
PATH_PRIMITIVE_COLUMNS = (
    "__barrier_pct__",
    "__y_ret__",
    "__first_touch_round_trip_cost__",
    "__first_touch_effective_tp_abs__",
    "__first_touch_effective_sl_abs__",
    "__is_timeout__",
    *(f"__bars_to_mfe_{key}r__" for key in ("05", "075", "1", "125", "15")),
    *(f"__bars_to_mae_{key}r__" for key in ("05", "075", "1", "15")),
)
TRAIN_ALIGNMENT_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side",
    "side_name",
    "__archetype_label_family__",
    "__regime_family__",
    "__archetype_policy_key__",
    "__first_touch_target_soft__",
    *PATH_PRIMITIVE_COLUMNS,
)
OOS_START = pd.Timestamp("2026-04-01T00:00:00Z")
OOS_END = pd.Timestamp("2026-07-01T00:00:00Z")


def _read_cache_manifest(
    path: Path,
    *,
    required_keys: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload_paths = dict(manifest.get("payload_paths") or {})
    required = set(required_keys or FULL_PAYLOAD_KEYS)
    missing = sorted(required.difference(payload_paths))
    if missing:
        raise RuntimeError(f"Stage-C cache {path.parent} lacks required payloads={missing}")
    payload = _load_fold_payload({**manifest, "payload_paths": payload_paths})
    for key in required:
        if key not in payload:
            raise RuntimeError(f"Stage-C could not load payload key={key} from {path.parent}")
    if "train" in payload and len(payload["train"]) != len(payload["x_train"]):
        raise RuntimeError(f"Stage-C train alignment failure in {path.parent}")
    if len(payload["valid"]) != len(payload["valid_metrics"]) or len(payload["valid"]) != len(payload["x_valid"]):
        raise RuntimeError(f"Stage-C valid alignment failure in {path.parent}")
    return manifest, payload


def _slice_rows(payload: Mapping[str, Any], prefix: str, mask: np.ndarray) -> dict[str, Any]:
    out = dict(payload)
    for key in (prefix, f"{prefix}_metrics", f"x_{prefix}"):
        if key not in payload:
            continue
        frame = payload[key]
        out[key] = frame.loc[mask].reset_index(drop=True)
    return out


def _read_narrow_train_labels(
    *,
    labels_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read only causal path/alignment fields from the immutable label store."""

    files = sorted(Path(labels_path).glob("*.parquet"))
    if not files:
        raise RuntimeError(f"Stage-C label store is empty: {labels_path}")
    frames: list[pd.DataFrame] = []
    import pyarrow.parquet as pq

    for path in files:
        available = set(pq.ParquetFile(path).schema.names)
        missing = sorted(set(PATH_PRIMITIVE_COLUMNS).difference(available))
        missing += sorted({"__ts__", "__symbol__", "__first_touch_target_soft__"}.difference(available))
        if missing:
            raise RuntimeError(f"Stage-C label file {path} lacks required columns={sorted(set(missing))}")
        columns = [column for column in TRAIN_ALIGNMENT_COLUMNS if column in available]
        frame = pd.read_parquet(path, columns=columns)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        if frame["__ts__"].isna().any():
            raise RuntimeError(f"Stage-C label file {path} contains invalid UTC timestamps")
        mask = frame["__ts__"].ge(start) & frame["__ts__"].lt(end)
        if mask.any():
            frames.append(frame.loc[mask])
    if not frames:
        raise RuntimeError("Stage-C label store has no rows in the fixed training interval")
    result = pd.concat(frames, ignore_index=True, copy=False)
    result = result.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)
    return result


def _rehydrate_compact_train_frame(
    *,
    source_arm_dir: Path,
    compact_payload: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Restore narrow path metadata and prove alignment with compact matrices."""

    arm_manifest_path = Path(source_arm_dir) / "manifest.json"
    if not arm_manifest_path.is_file():
        raise RuntimeError(f"Stage-C source arm lacks manifest: {arm_manifest_path}")
    arm_manifest = json.loads(arm_manifest_path.read_text(encoding="utf-8"))
    labels_path = Path(str(arm_manifest.get("labels_path") or ""))
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    frame = _read_narrow_train_labels(labels_path=labels_path, start=start, end=end)
    expected_rows = len(compact_payload["x_train"])
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Stage-C compact train row-count mismatch: labels={len(frame):,}, cache={expected_rows:,}"
        )
    cached_side = compact_payload["train_side"]
    if "side_name" not in cached_side.columns:
        raise RuntimeError("Stage-C compact train_side payload lacks side_name")
    observed_side = _side_name(frame).astype(str)
    expected_side = cached_side["side_name"].astype(str).str.lower().to_numpy()
    if not np.array_equal(observed_side, expected_side):
        mismatch = int(np.count_nonzero(observed_side != expected_side))
        raise RuntimeError(f"Stage-C compact train side alignment mismatch rows={mismatch:,}")
    cached_target = compact_payload["train_target"]
    if "target_soft" not in cached_target.columns:
        raise RuntimeError("Stage-C compact train_target payload lacks target_soft")
    observed_target = pd.to_numeric(frame["__first_touch_target_soft__"], errors="coerce").to_numpy(np.float64)
    expected_target = pd.to_numeric(cached_target["target_soft"], errors="coerce").to_numpy(np.float64)
    if not np.allclose(observed_target, expected_target, rtol=0.0, atol=1e-7, equal_nan=True):
        mismatch = int(np.count_nonzero(~np.isclose(observed_target, expected_target, rtol=0.0, atol=1e-7, equal_nan=True)))
        raise RuntimeError(f"Stage-C compact train target alignment mismatch rows={mismatch:,}")
    return frame


def _time_mask(frame: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Stage-C cache has invalid __ts__ values")
    return (ts.ge(start) & ts.lt(end)).to_numpy()


def _balanced_begin_middle_end(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Use equal early/middle/late train support for the inexpensive HPO stage."""

    frame = payload["train"]
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    thirds = (
        DEFAULT_TRAIN_START,
        DEFAULT_TRAIN_START + (DEFAULT_TRAIN_END - DEFAULT_TRAIN_START) / 3,
        DEFAULT_TRAIN_START + 2 * (DEFAULT_TRAIN_END - DEFAULT_TRAIN_START) / 3,
        DEFAULT_TRAIN_END - pd.Timedelta(hours=DEFAULT_PURGE_HOURS),
    )
    bins = [np.flatnonzero((ts.ge(thirds[index]) & ts.lt(thirds[index + 1])).to_numpy()) for index in range(3)]
    if any(len(values) == 0 for values in bins):
        raise RuntimeError("Stage-C selection cache lacks begin/middle/end train coverage")
    per_segment = min(len(values) for values in bins)
    positions = np.concatenate(
        [values[np.linspace(0, len(values) - 1, per_segment, dtype=np.int64)] for values in bins]
    )
    mask = np.zeros(len(frame), dtype=bool)
    mask[positions] = True
    return _slice_rows(payload, "train", mask)


def _load_source_payload(source_arm_dir: Path, *, purge_hours: float = DEFAULT_PURGE_HOURS) -> dict[str, Any]:
    """Load separate train-only HPO and full fixed-OOS payloads.

    Feature-selection caches are useful only for the inexpensive B/M/E target
    search. They must never become the April-June evaluation set. Final model
    fitting and scoring instead use a single full main-cache fold with exactly
    the fixed Apr-Jun 2026 OOS interval.
    """

    source_arm_dir = Path(source_arm_dir)
    selection_root = source_arm_dir / "_feature_selection_phase" / "_fold_cache"
    main_root = source_arm_dir / "_fold_cache"
    selection_paths = sorted(selection_root.glob("*/fold_manifest.json"))
    main_paths = sorted(main_root.glob("*/fold_manifest.json"))
    if not selection_paths or not main_paths:
        raise RuntimeError("Stage-C requires both feature-selection and main _fold_cache payloads")

    selection_manifest_path = next(
        (
            path
            for path in selection_paths
            if FULL_PAYLOAD_KEYS.issubset(
                set((json.loads(path.read_text(encoding="utf-8")).get("payload_paths") or {}).keys())
            )
        ),
        None,
    )
    if selection_manifest_path is None:
        raise RuntimeError("Stage-C found no outcome-bearing feature-selection cache")
    selection_manifest, selection = _read_cache_manifest(selection_manifest_path, required_keys=FULL_PAYLOAD_KEYS)
    train_cutoff = OOS_START - pd.Timedelta(hours=float(purge_hours))
    selection = _slice_rows(
        selection,
        "train",
        _time_mask(selection["train"], start=DEFAULT_TRAIN_START, end=train_cutoff),
    )
    if len(selection["train"]) == 0:
        raise RuntimeError("Stage-C feature-selection cache has no train-only rows after causal purge")
    selection = _balanced_begin_middle_end(selection)

    main_manifest_path: Path | None = None
    main_manifest: dict[str, Any] | None = None
    for path in main_paths:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        valid_start = pd.to_datetime(candidate.get("valid_start"), utc=True, errors="coerce")
        valid_end = pd.to_datetime(candidate.get("valid_end"), utc=True, errors="coerce")
        if valid_start == OOS_START and valid_end == OOS_END:
            main_manifest_path = path
            main_manifest = candidate
            break
    if main_manifest_path is None or main_manifest is None:
        raise RuntimeError(
            "Stage-C requires one full main _fold_cache with valid_start=2026-04-01 "
            "and valid_end=2026-07-01; monthly/growing caches are not a fixed 1y/3m ablation."
        )
    if "full" not in str(main_manifest.get("payload_train_sampling", "")).lower():
        raise RuntimeError("Stage-C fixed OOS cache must retain full main training rows")
    main_payload_keys = set((main_manifest.get("payload_paths") or {}).keys())
    if "train" in main_payload_keys:
        _, main = _read_cache_manifest(main_manifest_path, required_keys=FULL_PAYLOAD_KEYS)
        main = _slice_rows(main, "train", _time_mask(main["train"], start=DEFAULT_TRAIN_START, end=train_cutoff))
    else:
        _, main = _read_cache_manifest(main_manifest_path, required_keys=COMPACT_MAIN_PAYLOAD_KEYS)
        main["train"] = _rehydrate_compact_train_frame(
            source_arm_dir=source_arm_dir,
            compact_payload=main,
            start=DEFAULT_TRAIN_START,
            end=train_cutoff,
        )
    main = _slice_rows(main, "valid", _time_mask(main["valid"], start=OOS_START, end=OOS_END))
    if len(main["train"]) == 0 or len(main["valid"]) == 0:
        raise RuntimeError("Stage-C fixed OOS main cache has empty train or Apr-Jun evaluation rows")
    train_ts = pd.to_datetime(main["train"]["__ts__"], utc=True)
    valid_ts = pd.to_datetime(main["valid"]["__ts__"], utc=True)
    if train_ts.max() >= train_cutoff or valid_ts.min() < OOS_START or valid_ts.max() >= OOS_END:
        raise RuntimeError("Stage-C cache date-range validation failed")
    main_columns = list(main["x_train"].columns)
    if set(selection["x_train"].columns) != set(main_columns) or not main["x_train"].columns.equals(main["x_valid"].columns):
        raise RuntimeError("Stage-C selection and fixed-OOS caches disagree on the feature contract")
    # Side-local selection writes columns in side-discovery order, whereas the
    # compact final-fit cache persists the same union in canonical sorted order.
    # Reindexing is safe only after exact set equality has been established.
    selection["x_train"] = selection["x_train"].loc[:, main_columns]
    if "x_valid" in selection:
        selection["x_valid"] = selection["x_valid"].loc[:, main_columns]
    return {
        "selection": selection,
        "main": main,
        "selection_manifest_path": str(selection_manifest_path),
        "main_manifest_path": str(main_manifest_path),
        "selection_rows": int(len(selection["train"])),
        "main_train_rows": int(len(main["train"])),
        "main_oos_rows": int(len(main["valid"])),
    }


def _config_json(value: CandidateConfig) -> dict[str, Any]:
    return asdict(value)


def run_stage_c(
    *,
    source_arm_dir: Path,
    output_dir: Path,
    lgbm_params: Mapping[str, Any],
    proxy_trials: int = 48,
    finalists_per_side: int = 5,
    model_side_scope: str = "per_side",
    sides: Sequence[str] = ("long", "short"),
    enable_short_late_continuation: bool = True,
    objective_profile_name: str = "net_global_balanced",
    purge_hours: float = DEFAULT_PURGE_HOURS,
    seed: int = 42,
) -> dict[str, Path]:
    """Run hierarchical side target HPO then score the untouched source OOS rows."""

    try:
        import optuna
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Stage-C requires optuna") from exc
    if objective_profile_name not in OBJECTIVE_PROFILES:
        raise ValueError(
            f"Unsupported Stage-C objective profile={objective_profile_name}; "
            f"expected one of {tuple(OBJECTIVE_PROFILES)}"
        )
    objective_profile = OBJECTIVE_PROFILES[objective_profile_name]
    payloads = _load_source_payload(source_arm_dir, purge_hours=purge_hours)
    selection = payloads["selection"]
    train = selection["train"].reset_index(drop=True)
    x_train = selection["x_train"].astype(np.float32, copy=False).reset_index(drop=True)
    internal_folds = build_internal_chronological_folds(train, purge_hours=purge_hours)
    primitives = build_path_primitives(train)
    x = x_train.to_numpy(dtype=np.float32, copy=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_sides = tuple(dict.fromkeys(str(side).strip().lower() for side in sides))
    if not selected_sides or any(side not in {"long", "short"} for side in selected_sides):
        raise ValueError(f"Unsupported Stage-C sides={selected_sides}")
    trial_rows: list[dict[str, Any]] = []
    winners: dict[str, CandidateConfig] = {}
    for side_i, side in enumerate(selected_sides):
        candidates: list[tuple[float, CandidateConfig, list[dict[str, Any]]]] = []
        sampler = optuna.samplers.TPESampler(seed=int(seed) + side_i)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(8, int(proxy_trials) // 6), n_warmup_steps=1)
        geometry_trials = max(8, int(proxy_trials) // 2)
        refinement_trials = max(1, int(proxy_trials) - geometry_trials)
        geometry_study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def geometry_objective(trial: Any) -> float:
            candidate = CandidateConfig(
                side=side,
                geometry=_geometry_from_trial(
                    trial,
                    side=side,
                    enable_short_late_continuation=enable_short_late_continuation,
                ),
                weight=SideWeightConfig(target_exponent=1.0, weight_range_ratio=3.0),
            )
            try:
                score, rows = _evaluate_candidate(
                    candidate=candidate, x=x, feature_frame=x_train, frame=train, primitives=primitives,
                    folds=internal_folds, full_model=False, lgbm_params=lgbm_params,
                    seed=int(seed) + trial.number * 31,
                    objective_profile=objective_profile,
                    trial=trial,
                )
            except RuntimeError as exc:
                if str(exc) == "__PRUNED__":
                    raise optuna.TrialPruned() from exc
                raise
            candidates.append((score, candidate, rows))
            trial_rows.append({
                "phase": "geometry_proxy", "trial": int(trial.number), "side": side,
                "objective": score, "config_json": json.dumps(_config_json(candidate), sort_keys=True),
            })
            return score

        geometry_study.optimize(geometry_objective, n_trials=geometry_trials, show_progress_bar=False)
        if not candidates:
            raise RuntimeError(f"Stage-C produced no geometry proxy candidate for side={side}")
        best_geometry = max(candidates, key=lambda item: item[0])[1].geometry
        refine_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(seed) + 10_000 + side_i),
            pruner=pruner,
        )

        def weight_objective(trial: Any) -> float:
            candidate = _candidate_from_trial(
                trial,
                side,
                geometry=best_geometry,
                enable_short_late_continuation=enable_short_late_continuation,
            )
            try:
                score, rows = _evaluate_candidate(
                    candidate=candidate, x=x, feature_frame=x_train, frame=train, primitives=primitives,
                    folds=internal_folds, full_model=False, lgbm_params=lgbm_params,
                    seed=int(seed) + 5_000 + trial.number * 31,
                    objective_profile=objective_profile,
                    trial=trial,
                )
            except RuntimeError as exc:
                if str(exc) == "__PRUNED__":
                    raise optuna.TrialPruned() from exc
                raise
            candidates.append((score, candidate, rows))
            trial_rows.append({
                "phase": "weight_refine_proxy", "trial": int(trial.number), "side": side,
                "objective": score, "config_json": json.dumps(_config_json(candidate), sort_keys=True),
            })
            return score

        refine_study.optimize(weight_objective, n_trials=refinement_trials, show_progress_bar=False)
        finalists = sorted(candidates, key=lambda item: item[0], reverse=True)[: int(finalists_per_side)]
        full_candidates: list[tuple[float, CandidateConfig, list[dict[str, Any]]]] = []
        for rank, (_, candidate, _) in enumerate(finalists, start=1):
            score, rows = _evaluate_candidate(
                candidate=candidate, x=x, feature_frame=x_train, frame=train, primitives=primitives,
                folds=internal_folds, full_model=True, lgbm_params=lgbm_params,
                seed=int(seed) + 10_000 + rank,
                objective_profile=objective_profile,
            )
            full_candidates.append((score, candidate, rows))
            trial_rows.append({
                "phase": "full_lgbm", "trial": rank, "side": side,
                "objective": score, "config_json": json.dumps(_config_json(candidate), sort_keys=True),
            })
        if not full_candidates:
            raise RuntimeError(f"Stage-C produced no finalists for side={side}")
        winners[side] = max(full_candidates, key=lambda item: item[0])[1]

    scope = str(model_side_scope).strip().lower()
    if scope not in {"shared", "per_side"}:
        raise ValueError(f"Unsupported model_side_scope={model_side_scope}")
    if scope == "shared" and set(selected_sides) != {"long", "short"}:
        raise ValueError("Shared Stage-C fitting requires both long and short target contracts")
    # Fit final model(s) on the complete fixed one-year cache. Apr-Jun is never
    # supplied to LightGBM as an evaluation set; its outcomes are reporting
    # diagnostics only after the selected geometry/weight contracts are frozen.
    main = payloads["main"]
    main_train = main["train"].reset_index(drop=True)
    valid = main["valid"].reset_index(drop=True)
    valid_metrics = main["valid_metrics"].reset_index(drop=True)
    main_x_train = main["x_train"].astype(np.float32, copy=False).reset_index(drop=True)
    x_valid = main["x_valid"].astype(np.float32, copy=False).reset_index(drop=True)
    if not main_x_train.columns.equals(x_valid.columns) or not x_train.columns.equals(main_x_train.columns):
        raise RuntimeError("Stage-C source features differ between HPO and fixed OOS main cache")
    train_primitives = build_path_primitives(main_train)
    valid_primitives = build_path_primitives(valid)
    train_sides = _side_name(main_train)
    valid_sides = _side_name(valid)
    prediction = np.full(len(valid), np.nan, dtype=np.float32)
    target_manifest: dict[str, Any] = {}
    combined_target_train = np.zeros(len(main_train), dtype=np.float32)
    combined_weights = np.ones(len(main_train), dtype=np.float32)
    geometry_gross = np.full(len(valid), np.nan, dtype=np.float32)
    geometry_tp = np.zeros(len(valid), dtype=np.float32)
    geometry_sl = np.zeros(len(valid), dtype=np.float32)
    geometry_timeout = np.zeros(len(valid), dtype=np.float32)
    for side_i, side in enumerate(selected_sides):
        candidate = winners[side]
        tr_idx = np.flatnonzero(train_sides == side)
        va_idx = np.flatnonzero(valid_sides == side)
        if not len(tr_idx) or not len(va_idx):
            continue
        pressure = None
        pressure_features: tuple[str, ...] = ()
        if side == "short" and candidate.geometry.late_continuation_penalty > 0.0:
            pressure, pressure_features = short_late_continuation_pressure(
                main_x_train, fit_indices=tr_idx,
            )
        target_train = continuous_target(
            train_primitives,
            candidate.geometry,
            late_continuation_pressure=pressure,
        )
        weights = _weights_for_fold(target=target_train, frame=main_train, indices=tr_idx, config=candidate.weight)
        combined_target_train[tr_idx] = target_train[tr_idx]
        combined_weights[tr_idx] = weights
        gross, hit, stop, timeout, _ = geometry_outcomes(valid_primitives, candidate.geometry)
        geometry_gross[va_idx] = gross[va_idx]
        geometry_tp[va_idx] = hit[va_idx]
        geometry_sl[va_idx] = stop[va_idx]
        geometry_timeout[va_idx] = timeout[va_idx]
        if scope == "per_side":
            prediction[va_idx] = _full_lgbm_fit_predict_oos(
                main_x_train.iloc[tr_idx].to_numpy(dtype=np.float32, copy=False), target_train[tr_idx], weights,
                x_valid.iloc[va_idx].to_numpy(dtype=np.float32, copy=False),
                params=lgbm_params, seed=int(seed) + 20_000 + side_i,
            )
        target_manifest[side] = {
            **_config_json(candidate),
            "late_continuation_context_features": list(pressure_features),
            "late_continuation_context_contract": (
                "fold-train median/IQR normalized observable exhaustion/rebound pressure; "
                "applied only to weak-net-edge shorts"
                if pressure_features else "disabled"
            ),
        }
    if scope == "shared":
        prediction[:] = _full_lgbm_fit_predict_oos(
            main_x_train.to_numpy(dtype=np.float32, copy=False), combined_target_train, combined_weights,
            x_valid.to_numpy(dtype=np.float32, copy=False),
            params=lgbm_params, seed=int(seed) + 20_000,
        )
    selected_mask = np.isin(valid_sides, selected_sides)
    if not np.isfinite(prediction[selected_mask]).all() or not np.isfinite(geometry_gross[selected_mask]).all():
        raise RuntimeError("Stage-C final side scoring emitted non-finite predictions")
    ledger = valid.copy()
    for column in valid_metrics.columns:
        if column not in ledger.columns:
            ledger[column] = valid_metrics[column].to_numpy(copy=False)
    ledger["score"] = prediction
    ledger["stage_c_geometry_gross_return"] = geometry_gross
    ledger["stage_c_geometry_net_return"] = geometry_gross - valid_primitives.round_trip_cost
    ledger["stage_c_geometry_tp_hit"] = geometry_tp
    ledger["stage_c_geometry_sl_hit"] = geometry_sl
    ledger["stage_c_geometry_timeout"] = geometry_timeout
    ledger["base_target_contract"] = "stage_c_side_continuous_geometry_target_v1"
    ledger["base_sample_weight_contract"] = "target_power_timestamp_tempered_archetype_v1"
    ledger_path = output_dir / "best_oos_scored_ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    pd.DataFrame(trial_rows).to_csv(output_dir / "stage_c_trials.csv", index=False)
    diagnostics = []
    for side in selected_sides:
        positions = np.flatnonzero(valid_sides == side)
        if not len(positions):
            continue
        diagnostics.append(
            _topk_outcome_diagnostics(
                score=prediction[positions],
                frame=valid,
                gross_outcome=geometry_gross,
                round_trip_cost=valid_primitives.round_trip_cost,
                indices=positions,
            ).assign(side=side)
        )
    diagnostics_path = output_dir / "stage_c_oos_gross_net_diagnostics.csv"
    pd.concat(diagnostics, ignore_index=True).to_csv(diagnostics_path, index=False)
    manifest = {
        "schema": "base_side_target_geometry_hpo_v3",
        "source_arm_dir": str(source_arm_dir),
        "split": {
            "train_start_utc": DEFAULT_TRAIN_START.isoformat(),
            "train_end_exclusive_utc": DEFAULT_TRAIN_END.isoformat(),
            "internal_validation_months": list(DEFAULT_INTERNAL_VALIDATION_MONTHS),
            "purge_hours": float(purge_hours),
            "apr_jun_oos_used_for_selection": False,
        },
        "cache_contract": {
            "selection_manifest": payloads["selection_manifest_path"],
            "main_manifest": payloads["main_manifest_path"],
            "selection_hpo_rows": payloads["selection_rows"],
            "main_full_train_rows": payloads["main_train_rows"],
            "main_fixed_apr_jun_oos_rows": payloads["main_oos_rows"],
        },
        "path_primitive_contract": "cached_exact_first_passage_supported_r_grid_plus_terminal_timeout_return",
        "geometry_grid": {
            "tp_r": list(TP_R_GRID),
            "sl_r": list(SL_R_GRID),
            "max_profit_bars": list(MAX_PROFIT_BARS_GRID),
            "short_max_profit_bars": list(SHORT_MAX_PROFIT_BARS_GRID),
        },
        "short_target_context": {
            "enabled": bool(enable_short_late_continuation),
            "features": list(SHORT_LATE_CONTINUATION_FEATURES),
            "normalization": "fold-train median/IQR; sigmoid bounded composite",
            "target_effect": "late continuation penalty only on weak-net-edge short rows",
        },
        "objective": {
            "profile": objective_profile_name,
            "selection_basis": "global_within_side",
            "outcome": "geometry_gross_return_minus_exactly_one_stored_1pct_round_trip_cost",
            **objective_profile,
        },
        "search_stages": [
            "side_local_geometry_proxy_neutral_weights",
            "side_local_target_strength_weight_refinement",
            "side_local_full_lgbm_finalists_with_early_stopping",
        ],
        "weight_contract": {
            "formula": "target_soft ** exponent; p99 clip; bounded mean one",
            "exponent_grid": list(TARGET_EXPONENT_GRID),
            "ratio_search_continuous": [3.0, 12.0],
            "rebalance": "timestamp plus tempered archetype support",
        },
        "lgbm_params": dict(lgbm_params),
        "model_side_scope": scope,
        "optimized_sides": list(selected_sides),
        "feature_columns": list(map(str, x_train.columns)),
        "winner_by_side": target_manifest,
        "outputs": {
            "ledger": str(ledger_path),
            "trials": str(output_dir / "stage_c_trials.csv"),
            "gross_net_diagnostics": str(diagnostics_path),
        },
    }
    manifest_path = output_dir / "stage_c_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"ledger": ledger_path, "trials": output_dir / "stage_c_trials.csv", "manifest": manifest_path}


def _load_params(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = dict(payload.get("params") or payload)
    # HPO summary files also contain a numeric experiment ``objective``.  Do
    # not pass that diagnostic through as LightGBM's string objective.  Keep
    # this boundary deliberately narrow so Stage-C accepts both a clean model
    # bundle and a top-k HPO summary artifact.
    loss = str(raw.get("loss_function", raw.get("objective", "regression_l2"))).strip().lower()
    objective_map = {
        "regression": "regression_l2",
        "regression_l2": "regression_l2",
        "l2": "regression_l2",
        "huber": "huber",
        "fair": "fair",
    }
    params: dict[str, Any] = {
        "objective": objective_map.get(loss, "regression_l2"),
    }
    model_keys = (
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
    )
    for key in model_keys:
        if key in raw:
            params[key] = raw[key]
    return params


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-arm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-params-json", type=Path, required=True)
    parser.add_argument("--proxy-trials", type=int, default=48)
    parser.add_argument("--finalists-per-side", type=int, default=5)
    parser.add_argument("--model-side-scope", choices=("shared", "per_side"), default="per_side")
    parser.add_argument("--sides", default="long,short", help="Comma-separated target sides to optimize")
    parser.add_argument(
        "--short-late-continuation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the causal exhaustion/rebound penalty in short target HPO",
    )
    parser.add_argument("--objective-profile", choices=tuple(OBJECTIVE_PROFILES), default="net_global_balanced")
    parser.add_argument("--label-path-purge-hours", type=float, default=DEFAULT_PURGE_HOURS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = run_stage_c(
        source_arm_dir=args.source_arm_dir,
        output_dir=args.output_dir,
        lgbm_params=_load_params(args.fixed_params_json),
        proxy_trials=int(args.proxy_trials),
        finalists_per_side=int(args.finalists_per_side),
        model_side_scope=str(args.model_side_scope),
        sides=tuple(part.strip() for part in str(args.sides).split(",") if part.strip()),
        enable_short_late_continuation=bool(args.short_late_continuation),
        objective_profile_name=str(args.objective_profile),
        purge_hours=float(args.label_path_purge_hours),
        seed=int(args.seed),
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
