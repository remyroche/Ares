"""Matched, global-top-k evaluation helpers for regime stack ablations.

All arm comparisons are intentionally made on the exact same candidate IDs.
Selection occurs only after the supplied final/mapped score and is pooled over
sides and timestamps; neither per-timestamp nor per-side selection is exposed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .regime_oof_stack import (
    IDENTITY_COLUMNS,
    RegimeOOFStackError,
    qualify_category_stability,
    utc_period_key,
    validate_candidate_identity,
)


@dataclass(frozen=True)
class EvaluationColumns:
    """Column contract for every matched-arm evaluation."""

    mapped_score: str = "mapped_score"
    alpha_target: str = "__first_touch_target_soft__"
    net_ev: str = "execution_net_ev_12h"
    gross_ev: str = "execution_gross_ev_12h"
    cost: str = "execution_cost_12h"


def _rank_ic(left: pd.Series, right: pd.Series) -> float:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    xr, yr = x.loc[mask].rank(method="average"), y.loc[mask].rank(method="average")
    if xr.nunique() < 2 or yr.nunique() < 2:
        return float("nan")
    return float(xr.corr(yr))


def global_top_k_mask(
    frame: pd.DataFrame,
    *,
    score_col: str,
    top_fraction: float = 0.10,
) -> np.ndarray:
    """Stable global selection across the full supplied frame, never by subgroup."""

    if not 0.0 < top_fraction <= 1.0:
        raise RegimeOOFStackError("top_fraction must be in (0, 1]")
    checked = validate_candidate_identity(frame)
    if score_col not in checked:
        raise RegimeOOFStackError(f"mapped score column is missing: {score_col}")
    score = pd.to_numeric(checked[score_col], errors="coerce")
    if score.isna().any() or not np.isfinite(score.to_numpy(float)).all():
        raise RegimeOOFStackError("mapped score must be finite for every candidate")
    order = pd.DataFrame(
        {"position": np.arange(len(checked), dtype=np.int64), "score": score, "candidate_id": checked["candidate_id"].astype(str)}
    ).sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
    count = max(1, int(math.ceil(len(checked) * float(top_fraction))))
    mask = np.zeros(len(checked), dtype=bool)
    mask[order["position"].to_numpy()[:count]] = True
    return mask


def _require_metrics(frame: pd.DataFrame, columns: EvaluationColumns) -> pd.DataFrame:
    checked = validate_candidate_identity(frame)
    required = (columns.mapped_score, columns.alpha_target, columns.net_ev, columns.gross_ev, columns.cost)
    missing = [column for column in required if column not in checked]
    if missing:
        raise RegimeOOFStackError(f"evaluation frame missing required columns: {missing}")
    return checked


def _period_metrics(
    frame: pd.DataFrame,
    *,
    period_type: str,
    columns: EvaluationColumns,
    top_fraction: float,
) -> pd.DataFrame:
    work = frame.copy()
    work["__period__"] = utc_period_key(work["__ts__"], period_type)
    records: list[dict[str, Any]] = []
    for period, local in work.groupby("__period__", observed=True, sort=True):
        local = local.drop(columns="__period__")
        selected = local.loc[global_top_k_mask(local, score_col=columns.mapped_score, top_fraction=top_fraction)]
        records.append(
            {
                "period_type": period_type,
                "period": str(period),
                "candidate_rows": int(len(local)),
                "selected_rows": int(len(selected)),
                "alpha_rank_ic": _rank_ic(local[columns.mapped_score], local[columns.alpha_target]),
                "execution_net_rank_ic": _rank_ic(local[columns.mapped_score], local[columns.net_ev]),
                "mean_net_ev": float(pd.to_numeric(selected[columns.net_ev], errors="coerce").mean()),
                "mean_gross_ev": float(pd.to_numeric(selected[columns.gross_ev], errors="coerce").mean()),
                "mean_cost": float(pd.to_numeric(selected[columns.cost], errors="coerce").mean()),
                "hit_rate": float(pd.to_numeric(selected[columns.net_ev], errors="coerce").gt(0.0).mean()),
            }
        )
    return pd.DataFrame(records)


def _quantiles(periods: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    if periods.empty:
        return {f"{prefix}_ic_q10": float("nan"), f"{prefix}_ic_q50": float("nan"), f"{prefix}_net_ev_q10": float("nan"), f"{prefix}_net_ev_q50": float("nan")}
    ic = pd.to_numeric(periods["alpha_rank_ic"], errors="coerce").dropna()
    ev = pd.to_numeric(periods["mean_net_ev"], errors="coerce").dropna()
    return {
        f"{prefix}_ic_q10": float(ic.quantile(0.10)) if len(ic) else float("nan"),
        f"{prefix}_ic_q50": float(ic.quantile(0.50)) if len(ic) else float("nan"),
        f"{prefix}_net_ev_q10": float(ev.quantile(0.10)) if len(ev) else float("nan"),
        f"{prefix}_net_ev_q50": float(ev.quantile(0.50)) if len(ev) else float("nan"),
    }


def evaluate_arm(
    frame: pd.DataFrame,
    *,
    arm: str,
    columns: EvaluationColumns = EvaluationColumns(),
    top_fraction: float = 0.10,
    category_col: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate one arm using mapped-score global top-k policy selection."""

    work = _require_metrics(frame, columns)
    selected = work.loc[global_top_k_mask(work, score_col=columns.mapped_score, top_fraction=top_fraction)].copy()
    weekly = _period_metrics(work, period_type="week", columns=columns, top_fraction=top_fraction)
    monthly = _period_metrics(work, period_type="month", columns=columns, top_fraction=top_fraction)
    summary: dict[str, Any] = {
        "arm": arm,
        "selection_basis": "pooled_global_post_mapping_top_k",
        "top_fraction": float(top_fraction),
        "candidate_rows": int(len(work)),
        "top10_support": int(len(selected)),
        "alpha_rank_ic": _rank_ic(work[columns.mapped_score], work[columns.alpha_target]),
        "execution_net_rank_ic": _rank_ic(work[columns.mapped_score], work[columns.net_ev]),
        "top10_mean_net_ev": float(pd.to_numeric(selected[columns.net_ev], errors="coerce").mean()),
        "top10_mean_gross_ev": float(pd.to_numeric(selected[columns.gross_ev], errors="coerce").mean()),
        "top10_mean_cost": float(pd.to_numeric(selected[columns.cost], errors="coerce").mean()),
        "top10_hit_rate": float(pd.to_numeric(selected[columns.net_ev], errors="coerce").gt(0.0).mean()),
        "positive_week_fraction": float(pd.to_numeric(weekly["mean_net_ev"], errors="coerce").gt(0.0).mean()) if len(weekly) else float("nan"),
        "positive_month_fraction": float(pd.to_numeric(monthly["mean_net_ev"], errors="coerce").gt(0.0).mean()) if len(monthly) else float("nan"),
        "worst_week_net_ev": float(pd.to_numeric(weekly["mean_net_ev"], errors="coerce").min()) if len(weekly) else float("nan"),
        "worst_month_net_ev": float(pd.to_numeric(monthly["mean_net_ev"], errors="coerce").min()) if len(monthly) else float("nan"),
        **_quantiles(weekly, prefix="weekly"),
        **_quantiles(monthly, prefix="monthly"),
    }
    periods = pd.concat([weekly.assign(arm=arm), monthly.assign(arm=arm)], ignore_index=True)
    if category_col is None:
        category = pd.DataFrame()
    else:
        if category_col not in selected:
            raise RegimeOOFStackError(f"category column missing from selected rows: {category_col}")
        category = qualify_category_stability(
            selected,
            category_col=category_col,
            value_col=columns.net_ev,
        ).assign(arm=arm)
    return summary, periods, category


def evaluate_matched_arms(
    arm_frames: Mapping[str, pd.DataFrame],
    *,
    columns: EvaluationColumns = EvaluationColumns(),
    top_fraction: float = 0.10,
    category_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate matched arms only when every arm has exactly the same rows."""

    if not arm_frames:
        raise RegimeOOFStackError("at least one matched arm is required")
    normalized: dict[str, pd.DataFrame] = {name: _require_metrics(frame, columns) for name, frame in arm_frames.items()}
    reference_name, reference = next(iter(normalized.items()))
    reference_keys = reference.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    for name, frame in normalized.items():
        keys = frame.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
        if len(keys) != len(reference_keys) or not keys.equals(reference_keys):
            raise RegimeOOFStackError(f"matched arm {name!r} does not share exact candidate rows with {reference_name!r}")
    summaries: list[dict[str, Any]] = []
    period_parts: list[pd.DataFrame] = []
    category_parts: list[pd.DataFrame] = []
    for arm, frame in normalized.items():
        summary, periods, categories = evaluate_arm(frame, arm=arm, columns=columns, top_fraction=top_fraction, category_col=category_col)
        summaries.append(summary)
        period_parts.append(periods)
        if not categories.empty:
            category_parts.append(categories)
    return (
        pd.DataFrame(summaries).sort_values("arm", kind="stable").reset_index(drop=True),
        pd.concat(period_parts, ignore_index=True),
        pd.concat(category_parts, ignore_index=True) if category_parts else pd.DataFrame(),
    )
