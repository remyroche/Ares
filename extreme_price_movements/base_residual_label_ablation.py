"""Leakage-safe contracts for fixed-window base/residual label ablations.

This module deliberately contains no feature loading or model fitting.  It
owns the chronology, target recipes, ranking, and economic evaluation used by
``scripts/run_base_residual_label_ablation.py`` so those rules can be tested
without launching a training job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCHEMA = "base_residual_label_ablation_v1"
ROUND_TRIP_COST = 0.01
CALIBRATION_DAYS = 21


@dataclass(frozen=True)
class FixedWindowCalendar:
    base_train_start: pd.Timestamp
    base_train_end: pd.Timestamp
    base_oos_end: pd.Timestamp
    meta_train_end: pd.Timestamp
    label_horizon_hours: int = 24
    decision_delay_hours: int = 1

    @classmethod
    def from_first_oos_month(
        cls,
        first_oos_month: str = "2026-01",
        *,
        label_horizon_hours: int = 24,
    ) -> "FixedWindowCalendar":
        first = pd.Timestamp(pd.Period(first_oos_month).start_time, tz="UTC")
        return cls(
            base_train_start=first - pd.DateOffset(months=4),
            base_train_end=first,
            base_oos_end=first + pd.DateOffset(months=6),
            meta_train_end=first + pd.DateOffset(months=3),
            label_horizon_hours=int(label_horizon_hours),
        )

    @property
    def purge(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.decision_delay_hours + self.label_horizon_hours)

    def masks(self, timestamps: Iterable[object]) -> dict[str, np.ndarray]:
        ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
        train_cutoff = self.base_train_end - self.purge
        return {
            "base_train": (
                ts.ge(self.base_train_start) & ts.lt(train_cutoff)
            ).to_numpy(),
            "base_oos": (
                ts.ge(self.base_train_end) & ts.lt(self.base_oos_end)
            ).to_numpy(),
            "meta_train": (
                ts.ge(self.base_train_end) & ts.lt(self.meta_train_end)
            ).to_numpy(),
            "meta_oos": (
                ts.ge(self.meta_train_end) & ts.lt(self.base_oos_end)
            ).to_numpy(),
        }

    def manifest(self) -> dict[str, object]:
        return {
            **{
                key: value.isoformat()
                for key, value in asdict(self).items()
                if isinstance(value, pd.Timestamp)
            },
            "label_horizon_hours": int(self.label_horizon_hours),
            "decision_delay_hours": int(self.decision_delay_hours),
            "purge_hours": float(self.purge / pd.Timedelta(hours=1)),
            "contract": (
                "four exact UTC calendar months -> one frozen base model -> "
                "six OOS months; first three OOS months meta fit, final three "
                "months untouched meta evaluation"
            ),
        }


@dataclass(frozen=True)
class LabelRecipe:
    recipe_id: str
    execution_weight: float
    mfe_weight: float
    mae_weight: float
    timing_weight: float
    early_path_weight: float
    slope_weight: float
    threshold: float = 0.50
    temperature: float = 0.20
    horizon_hours: int = 12

    def normalized(self) -> "LabelRecipe":
        weights = np.asarray(
            [
                self.execution_weight,
                self.mfe_weight,
                self.mae_weight,
                self.timing_weight,
                self.early_path_weight,
                self.slope_weight,
            ],
            dtype=np.float64,
        )
        if (weights < 0.0).any() or float(weights.sum()) <= 0.0:
            raise ValueError("label recipe weights must be non-negative and non-zero")
        weights /= weights.sum()
        return LabelRecipe(
            recipe_id=self.recipe_id,
            execution_weight=float(weights[0]),
            mfe_weight=float(weights[1]),
            mae_weight=float(weights[2]),
            timing_weight=float(weights[3]),
            early_path_weight=float(weights[4]),
            slope_weight=float(weights[5]),
            threshold=float(self.threshold),
            temperature=float(self.temperature),
            horizon_hours=int(self.horizon_hours),
        )

    def manifest(self) -> dict[str, object]:
        return asdict(self.normalized())


def _numeric(frame: pd.DataFrame, column: str, default: float) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), float(default), dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    return np.nan_to_num(
        values, nan=float(default), posinf=float(default), neginf=float(default)
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def label_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Return bounded 24h execution and exact 12h path-quality components.

    The 12h component uses the cost-aware meaningful-MFE threshold materialized
    by the path-target pipeline.  The early-path component is a two-hour proxy
    because the full-population target store does not contain exact 15-minute
    closes; it rewards early MFE and penalizes an early adverse trough.  This
    limitation is serialized by the runner and never represented as an exact
    next-2/3-bar close label.
    """

    execution = np.clip(_numeric(frame, "__first_touch_target_soft__", 0.0), 0.0, 1.0)
    peak_atr = np.maximum(_numeric(frame, "__peak_mfe_atr_12h__", 0.0), 0.0)
    mfe = _sigmoid((peak_atr - 1.5) / 0.45)

    mae_atr = np.maximum(
        _numeric(frame, "__mae_before_meaningful_mfe_atr_12h__", 10.0), 0.0
    )
    mae = _sigmoid((0.75 - mae_atr) / 0.25)

    time_hours = np.clip(
        _numeric(frame, "__time_to_first_meaningful_mfe_hours_12h__", 12.0),
        0.0,
        12.0,
    )
    time_80_bars = np.clip(_numeric(frame, "__bars_to_80pct_peak__", 12.0), 0.0, 12.0)
    timing = np.sqrt(
        np.clip(1.0 - time_hours / 12.0, 0.0, 1.0)
        * np.clip(1.0 - time_80_bars / 12.0, 0.0, 1.0)
    )

    early_mfe = np.maximum(_numeric(frame, "__mfe_before_60m_atr__", 0.0), 0.0)
    early_ratio = np.clip(_numeric(frame, "__mfe_2h_over_mfe_12h__", 0.0), 0.0, 1.0)
    adverse_early = np.clip(
        np.maximum(
            _numeric(frame, "__adverse_trough_within_60m__", 0.0),
            _numeric(frame, "__adverse_trough_within_120m__", 0.0),
        ),
        0.0,
        1.0,
    )
    early_path = np.clip(
        0.55 * _sigmoid((early_mfe - 0.25) / 0.15)
        + 0.45 * early_ratio
        - 0.50 * adverse_early,
        0.0,
        1.0,
    )

    slope = _sigmoid(_numeric(frame, "__future_slope_atr_per_hour_12h__", 0.0) / 0.20)
    reaches = np.clip(_numeric(frame, "__meaningful_mfe_reached_12h__", 0.0), 0.0, 1.0)
    # A 12-hour timeout is a non-hit by construction.  Retain continuous path
    # information below 0.5 instead of forcing every timeout to hard zero.
    execution_12h = np.where(
        reaches > 0.5,
        np.clip(0.35 + 0.65 * mfe * mae * (0.35 + 0.65 * timing), 0.0, 1.0),
        np.clip(0.20 * mfe * mae, 0.0, 0.49),
    )
    return pd.DataFrame(
        {
            "execution_24h": execution.astype(np.float32),
            "execution_12h": execution_12h.astype(np.float32),
            "mfe_12h": mfe.astype(np.float32),
            "mae_clean_12h": mae.astype(np.float32),
            "timing_12h": timing.astype(np.float32),
            "early_path_2h_proxy": early_path.astype(np.float32),
            "slope_12h": slope.astype(np.float32),
            "meaningful_mfe_reached_12h": reaches.astype(np.float32),
        },
        index=frame.index,
    )


def build_soft_label(
    components: pd.DataFrame, recipe: LabelRecipe
) -> tuple[np.ndarray, np.ndarray]:
    recipe = recipe.normalized()
    execution_column = (
        "execution_12h" if recipe.horizon_hours == 12 else "execution_24h"
    )
    if recipe.recipe_id == "baseline_24h":
        soft = np.clip(components["execution_24h"].to_numpy(np.float64), 0.0, 1.0)
        return soft.astype(np.float32), (soft >= 0.5).astype(np.float32)
    raw = (
        recipe.execution_weight * components[execution_column].to_numpy(np.float64)
        + recipe.mfe_weight * components["mfe_12h"].to_numpy(np.float64)
        + recipe.mae_weight * components["mae_clean_12h"].to_numpy(np.float64)
        + recipe.timing_weight * components["timing_12h"].to_numpy(np.float64)
        + recipe.early_path_weight
        * components["early_path_2h_proxy"].to_numpy(np.float64)
        + recipe.slope_weight * components["slope_12h"].to_numpy(np.float64)
    )
    temperature = max(float(recipe.temperature), 1e-4)
    soft = _sigmoid((raw - float(recipe.threshold)) / temperature)
    hard = (raw >= float(recipe.threshold)).astype(np.float32)
    return soft.astype(np.float32), hard


def default_label_recipes(seed: int = 20260725) -> list[LabelRecipe]:
    """Deterministic compact HPO space, including named baseline ablations."""

    recipes = [
        LabelRecipe("baseline_24h", 1, 0, 0, 0, 0, 0, 0.50, 1.0, 24),
        LabelRecipe("timeout_12h", 1, 0, 0, 0, 0, 0, 0.50, 0.20, 12),
        LabelRecipe(
            "time_aware_12h", 0.45, 0.15, 0.15, 0.12, 0.08, 0.05, 0.48, 0.18, 12
        ),
    ]
    rng = np.random.default_rng(int(seed))
    for index in range(9):
        # Execution remains the largest component; side is never a search axis.
        execution = rng.uniform(0.35, 0.70)
        rest = rng.dirichlet(np.array([1.3, 1.5, 1.2, 1.0, 0.7]))
        rest *= 1.0 - execution
        recipes.append(
            LabelRecipe(
                recipe_id=f"hpo_{index:02d}",
                execution_weight=float(execution),
                mfe_weight=float(rest[0]),
                mae_weight=float(rest[1]),
                timing_weight=float(rest[2]),
                early_path_weight=float(rest[3]),
                slope_weight=float(rest[4]),
                threshold=float(rng.choice([0.42, 0.48, 0.54, 0.60])),
                temperature=float(rng.choice([0.12, 0.18, 0.25])),
                horizon_hours=12,
            )
        )
    return recipes


def rank_mask(
    frame: pd.DataFrame,
    score: Sequence[float],
    *,
    fraction: float = 0.10,
    scope: str = "timestamp_side",
) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    if len(values) != len(frame):
        raise ValueError("score length differs from frame")
    work = pd.DataFrame(
        {
            "position": np.arange(len(frame), dtype=np.int64),
            "score": np.nan_to_num(values, nan=-np.inf),
            "symbol": frame["__symbol__"].astype(str).to_numpy(),
        }
    )
    if scope == "timestamp_side":
        work["ts"] = pd.to_datetime(frame["__ts__"], utc=True).to_numpy()
        work["side"] = frame["side_name"].astype(str).str.lower().to_numpy()
        group_columns = ["ts", "side"]
    elif scope == "global":
        work["group"] = "all"
        group_columns = ["group"]
    else:
        raise ValueError(f"unsupported rank scope: {scope}")
    work = work.sort_values(
        [*group_columns, "score", "symbol"],
        ascending=[*[True] * len(group_columns), False, True],
        kind="mergesort",
    )
    grouped = work.groupby(group_columns, sort=False, dropna=False)
    work["rank"] = grouped.cumcount() + 1
    work["rows"] = grouped["position"].transform("size")
    selected = work["rank"] <= np.maximum(
        1, np.ceil(work["rows"] * float(fraction)).astype(int)
    )
    mask = np.zeros(len(frame), dtype=bool)
    mask[work.loc[selected, "position"].to_numpy(np.int64)] = True
    return mask


def economic_metrics(
    frame: pd.DataFrame,
    score: Sequence[float],
    *,
    economic_column: str = "__first_touch_capture_net__",
    admitted: Sequence[bool] | None = None,
) -> dict[str, float | int]:
    economics = _numeric(frame, economic_column, np.nan)
    values = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(economics) & np.isfinite(values)
    if admitted is not None:
        valid &= np.asarray(admitted, dtype=bool)
    local = frame.loc[valid].reset_index(drop=True)
    local_score = values[valid]
    local_econ = economics[valid]
    result: dict[str, float | int] = {
        "rows": int(valid.sum()),
        "mean_net_return": float(np.mean(local_econ))
        if len(local_econ)
        else float("nan"),
        "rank_ic": (
            float(spearmanr(local_score, local_econ).statistic)
            if len(local_econ) > 2
            else float("nan")
        ),
    }
    for scope in ("global", "timestamp_side"):
        selected = (
            rank_mask(local, local_score, scope=scope)
            if len(local)
            else np.zeros(0, bool)
        )
        result[f"{scope}_top10_rows"] = int(selected.sum())
        result[f"{scope}_top10_mean_net_return"] = (
            float(np.mean(local_econ[selected])) if selected.any() else float("nan")
        )
    return result


def label_hpo_objective(metrics_by_month: Sequence[Mapping[str, float]]) -> float:
    """Stable training-only objective emphasizing residual top-10 economics."""

    if not metrics_by_month:
        return float("-inf")
    top = np.asarray(
        [
            float(row["timestamp_side_top10_mean_net_return"])
            for row in metrics_by_month
        ],
        dtype=np.float64,
    )
    global_top = np.asarray(
        [float(row["global_top10_mean_net_return"]) for row in metrics_by_month],
        dtype=np.float64,
    )
    rank_ic = np.asarray(
        [float(row["rank_ic"]) for row in metrics_by_month], dtype=np.float64
    )
    if not np.isfinite(np.r_[top, global_top, rank_ic]).all():
        return float("-inf")
    downside = max(0.0, -float(np.min(top)))
    return float(
        0.50 * np.median(top)
        + 0.25 * np.median(global_top)
        + 0.15 * np.min(top)
        + 0.10 * np.median(rank_ic)
        - 0.50 * downside
    )
