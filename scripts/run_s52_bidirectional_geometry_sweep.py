#!/usr/bin/env python3
"""Bidirectional S52 first-touch geometry sweep.

This is a base-layer geometry diagnostic. It ranks vol-normalized TP/SL arms by
precision@top-k for clean first-touch outcomes, not by final strategy PnL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_first_touch_capture_proxy import (  # noqa: E402
    EXECUTABLE_MARGIN_COST_FLOOR,
    _build_grid_arms,
    _fetch_policy_paths,
    _first_touch_capture_outcome,
    _format_table,
    _infer_side,
    _load_labels,
    _rank_pct,
    _timestamp_top_indices,
    run_proxy,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    ROUND_TRIP_COST,
    _json_safe,
    _rank_top_indices,
    _safe_numeric,
)


DEFAULT_LABELS_DIR = Path(
    "data_perp/artifacts/"
    "20260702_184500_single_head_monthly_walkforward_bidirectional_sideaware_policy_net_economic_target_labels/"
    "labels"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_bidirectional_first_touch_geometry_sweep_v1")
DEFAULT_MONTHS = "2026-04,2026-05,2026-06"
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
EXECUTABLE_MARGIN_COST_FLOOR = 0.0100


def _parse_csv(value: str | None, default: tuple[str, ...] = ()) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...] = ()) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_int_csv(value: str | None, default: tuple[int, ...] = ()) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _side_file(labels_dir: Path, side: str) -> Path:
    files = sorted(labels_dir.glob(f"train_{side}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train_{side}_*.parquet found under {labels_dir}")
    if len(files) > 1:
        raise RuntimeError(f"Expected one {side} file under {labels_dir}, found {len(files)}")
    return files[0]


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str = "selected_rows") -> float:
    if frame.empty or value_col not in frame.columns:
        return float("nan")
    values = _safe_numeric(frame[value_col])
    weights = _safe_numeric(frame.get(weight_col, pd.Series(1.0, index=frame.index))).fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask].to_numpy(dtype=np.float64), weights=weights[mask].to_numpy(dtype=np.float64)))


def _weighted_mean_any(frame: pd.DataFrame, value_cols: Sequence[str], weight_col: str = "selected_rows") -> float:
    for value_col in value_cols:
        value = _weighted_mean(frame, value_col, weight_col=weight_col)
        if np.isfinite(float(value)):
            return value
    return float("nan")


def _gross_ev_weighted_first_touch_precision(frame: pd.DataFrame) -> float:
    """Aggregate clean first-touch precision weighted by gross EV opportunity.

    The monthly proxy computes gross hit value / gross absolute outcome value
    inside each month. For geometry ranking we want the same ratio across the
    whole slice instead of a row-weighted average of monthly ratios.
    """
    required = {"gross_hit_value_mean", "gross_stop_value_mean", "gross_timeout_value_mean", "selected_rows"}
    if frame.empty or not required.issubset(set(frame.columns)):
        first_touch_precision_col = (
            "ev_weighted_first_touch_precision"
            if "ev_weighted_first_touch_precision" in frame.columns
            else "ev_weighted_clean_precision"
        )
        return _weighted_mean(frame, first_touch_precision_col)
    rows = _safe_numeric(frame["selected_rows"]).fillna(0.0).clip(lower=0.0)
    hit_value = _safe_numeric(frame["gross_hit_value_mean"]).fillna(0.0).clip(lower=0.0) * rows
    stop_value = _safe_numeric(frame["gross_stop_value_mean"]).fillna(0.0).clip(lower=0.0) * rows
    timeout_value = _safe_numeric(frame["gross_timeout_value_mean"]).fillna(0.0).clip(lower=0.0) * rows
    denom = float((hit_value + stop_value + timeout_value).sum())
    if denom <= 1e-12:
        return float("nan")
    return float(hit_value.sum() / denom)


def _score_value(row: dict[str, Any], key: str) -> float:
    value = float(row.get(key, float("nan")))
    return value if np.isfinite(value) else 0.0


def _optional_score_value(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, float("nan")))
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _precision_geometry_score(row: dict[str, Any], *, include_side_floor: bool) -> float:
    """Score geometries by gross-EV-weighted clean first-touch precision only.

    Stop rate, timeout, net EV, and MAE remain diagnostics in the report. They
    should not make the base-layer geometry objective harder than necessary.
    """
    score = (
        0.55 * _score_value(row, "fit_gross_ev_weighted_first_touch_precision")
        + 1.00 * _score_value(row, "holdout_gross_ev_weighted_first_touch_precision")
    )
    if include_side_floor:
        score += 0.45 * _score_value(row, "min_side_holdout_ev_weighted_precision")
    return float(score)


def _fit_precision_geometry_score(row: dict[str, Any], *, include_side_floor: bool) -> float:
    """Fit-only version of the geometry precision score.

    This is used for choosing the expensive proxy shortlist. The shortlist is a
    model-selection decision, so it must not depend on the holdout month.
    """
    score = 1.00 * _score_value(row, "fit_gross_ev_weighted_first_touch_precision")
    if include_side_floor:
        side_floor = row.get("fit_min_side_gross_ev_weighted_first_touch_precision")
        score += 0.45 * (float(side_floor) if side_floor is not None and np.isfinite(float(side_floor)) else _score_value(row, "fit_gross_ev_weighted_first_touch_precision"))
    return float(score)


def _path_quality_geometry_score(row: dict[str, Any], *, include_side_floor: bool) -> float:
    """Rank side/archetype geometries by learnability with executable-path penalties.

    This is still a base-layer geometry objective, not realized strategy PnL.
    Gross-EV-weighted clean precision remains dominant, but geometries that
    produce path-dirty top-k rows should not win an archetype-specific sweep.
    """
    score = _precision_geometry_score(row, include_side_floor=include_side_floor)
    score += 0.20 * _score_value(row, "holdout_mean_capture_gross")
    score += 8.00 * _score_value(row, "holdout_mean_capture_net")
    score += 4.00 * _score_value(row, "fit_mean_capture_net")
    score += 12.00 * _score_value(row, "holdout_mean_executable_margin")
    score += 6.00 * _score_value(row, "fit_mean_executable_margin")
    score -= 4.00 * max(0.0, -_score_value(row, "holdout_mean_capture_net"))
    score -= 2.00 * max(0.0, -_score_value(row, "fit_mean_capture_net"))
    score -= 6.00 * max(0.0, -_score_value(row, "holdout_mean_executable_margin"))
    score -= 3.00 * max(0.0, -_score_value(row, "fit_mean_executable_margin"))
    holdout_margin_hit_rate = _optional_score_value(row, "holdout_positive_executable_margin_rate")
    if np.isfinite(holdout_margin_hit_rate):
        score -= 0.35 * max(0.0, 0.50 - holdout_margin_hit_rate)
    score -= 0.35 * max(0.0, _score_value(row, "holdout_bad_mae_1r_rate") - 0.50)
    score -= 0.45 * max(0.0, _score_value(row, "holdout_first_touch_bad_mae_1r_rate") - 0.25)
    score -= 0.18 * max(0.0, _score_value(row, "holdout_mean_max_adverse_before_mfe_1r") - 1.50)
    score -= 0.06 * max(0.0, _score_value(row, "holdout_mean_underwater_bars_before_mfe_1r") - 10.0)
    score -= 0.12 * max(0.0, _score_value(row, "holdout_mean_underwater_fraction_before_mfe_1r") - 0.45)
    score -= 0.25 * max(0.0, _score_value(row, "holdout_mae_1r_before_mfe_1r_rate") - 0.35)
    score += 0.10 * max(0.0, _score_value(row, "holdout_mfe_1r_before_mae_1r_rate") - 0.55)
    score -= 0.25 * max(0.0, _score_value(row, "holdout_stop_rate") - 0.35)
    score -= 0.20 * max(0.0, _score_value(row, "holdout_timeout_rate") - 0.12)
    score -= 0.02 * max(0.0, _score_value(row, "holdout_mae_to_sl_p90") - 4.0)
    rows = _score_value(row, "holdout_selected_rows")
    if rows <= 0.0:
        score -= 1.0
    elif rows < 50.0:
        score -= 0.20 * (1.0 - rows / 50.0)
    return float(score)


def _fit_path_quality_geometry_score(row: dict[str, Any], *, include_side_floor: bool) -> float:
    """Fit-only path-order score for shortlist selection.

    It mirrors the path-quality intent of the holdout-facing diagnostic score,
    but every term is computed from fit months only.
    """
    score = _fit_precision_geometry_score(row, include_side_floor=include_side_floor)
    score += 8.00 * _score_value(row, "fit_mean_capture_net")
    score += 12.00 * _score_value(row, "fit_mean_executable_margin")
    score -= 4.00 * max(0.0, -_score_value(row, "fit_mean_capture_net"))
    score -= 6.00 * max(0.0, -_score_value(row, "fit_mean_executable_margin"))
    fit_margin_hit_rate = _optional_score_value(row, "fit_positive_executable_margin_rate")
    if np.isfinite(fit_margin_hit_rate):
        score -= 0.35 * max(0.0, 0.50 - fit_margin_hit_rate)
    score -= 0.35 * max(0.0, _score_value(row, "fit_bad_mae_1r_rate") - 0.50)
    score -= 0.45 * max(0.0, _score_value(row, "fit_first_touch_bad_mae_1r_rate") - 0.25)
    score -= 0.18 * max(0.0, _score_value(row, "fit_mean_max_adverse_before_mfe_1r") - 1.50)
    score -= 0.06 * max(0.0, _score_value(row, "fit_mean_underwater_bars_before_mfe_1r") - 10.0)
    score -= 0.12 * max(0.0, _score_value(row, "fit_mean_underwater_fraction_before_mfe_1r") - 0.45)
    score -= 0.25 * max(0.0, _score_value(row, "fit_mae_1r_before_mfe_1r_rate") - 0.35)
    score += 0.10 * max(0.0, _score_value(row, "fit_mfe_1r_before_mae_1r_rate") - 0.55)
    score -= 0.25 * max(0.0, _score_value(row, "fit_stop_rate") - 0.35)
    score -= 0.20 * max(0.0, _score_value(row, "fit_timeout_rate") - 0.12)
    score -= 0.02 * max(0.0, _score_value(row, "fit_mae_to_sl_p90") - 4.0)
    rows = _score_value(row, "fit_selected_rows")
    if rows <= 0.0:
        score -= 1.0
    elif rows < 50.0:
        score -= 0.20 * (1.0 - rows / 50.0)
    return float(score)


def _selected_top_symbol_share(frame: pd.DataFrame) -> float:
    if frame.empty or "__symbol__" not in frame.columns:
        return float("nan")
    counts = frame["__symbol__"].astype(str).value_counts(dropna=False)
    if counts.empty:
        return float("nan")
    return float(counts.iloc[0] / max(len(frame), 1))


def _coarse_selection_metrics(
    *,
    valid: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: Any,
    side: str,
    month: str,
    top_frac: float,
    selection_mode: str,
    regime_family: str,
) -> dict[str, Any]:
    idx = (
        _timestamp_top_indices(valid, score, float(top_frac))
        if str(selection_mode) == "timestamp"
        else _rank_top_indices(score, float(top_frac))
    )
    selected = target.iloc[idx].reset_index(drop=True) if len(idx) else target.iloc[:0].copy()
    selected_rows = valid.iloc[idx].reset_index(drop=True) if len(idx) else valid.iloc[:0].copy()
    row_cost = _safe_numeric(
        selected.get("round_trip_cost", pd.Series(dtype=float))
    ).fillna(float(EXECUTABLE_MARGIN_COST_FLOOR))
    gross = _safe_numeric(selected.get("capture_net", pd.Series(dtype=float))).fillna(0.0) + row_cost
    executable_cost = row_cost.clip(lower=float(EXECUTABLE_MARGIN_COST_FLOOR))
    gross_minus_1pct = gross - float(EXECUTABLE_MARGIN_COST_FLOOR)
    executable_margin = gross - executable_cost
    hit = _safe_numeric(selected.get("capture_hit", pd.Series(dtype=float))).fillna(0.0).gt(0.0)
    stop = _safe_numeric(selected.get("capture_stop", pd.Series(dtype=float))).fillna(0.0).gt(0.0)
    timeout = _safe_numeric(selected.get("capture_timeout", pd.Series(dtype=float))).fillna(0.0).gt(0.0)
    valid_path = _safe_numeric(selected.get("capture_valid_path", pd.Series(dtype=float))).fillna(0.0).gt(0.0)
    same_bar = _safe_numeric(selected.get("same_bar_both_hit", pd.Series(dtype=float))).fillna(0.0).gt(0.0)
    clean_hit = hit & valid_path & ~same_bar & gross.gt(0.0)
    denom = float(gross.abs().sum())
    gross_precision = float(gross.where(clean_hit, 0.0).clip(lower=0.0).sum() / denom) if denom > 1e-12 else float("nan")
    hit_value = float(gross.where(clean_hit, 0.0).clip(lower=0.0).mean()) if len(selected) else float("nan")
    stop_value = float((-gross.where(stop, 0.0)).clip(lower=0.0).mean()) if len(selected) else float("nan")
    timeout_value = float(gross.where(timeout, 0.0).abs().mean()) if len(selected) else float("nan")
    mae_to_sl = _safe_numeric(selected.get("mae_to_sl", pd.Series(dtype=float)))
    first_touch_mae = _safe_numeric(selected.get("first_touch_mae_norm", pd.Series(dtype=float)))
    mfe_before_mae_1r = _safe_numeric(selected.get("mfe_1r_before_mae_1r", pd.Series(dtype=float)))
    mae_before_mfe_1r = _safe_numeric(selected.get("mae_1r_before_mfe_1r", pd.Series(dtype=float)))
    max_adverse_before_mfe_1r = _safe_numeric(
        selected.get("max_adverse_before_mfe_1r", pd.Series(dtype=float))
    )
    underwater_bars_before_mfe_1r = _safe_numeric(
        selected.get("underwater_bars_before_mfe_1r", pd.Series(dtype=float))
    )
    underwater_fraction_before_mfe_1r = _safe_numeric(
        selected.get("underwater_fraction_before_mfe_1r", pd.Series(dtype=float))
    )
    area_underwater_before_mfe_1r = _safe_numeric(
        selected.get("area_underwater_before_mfe_1r", pd.Series(dtype=float))
    )
    return {
        "period": str(month),
        "side": str(side),
        "arm": str(arm.name),
        "selection_mode": str(selection_mode),
        "top_frac": float(top_frac),
        "tp_r": float(arm.tp_r),
        "sl_r": float(arm.sl_r),
        "trail_r": float(getattr(arm, "trail_r", 0.50)),
        "max_bars_to_mfe": float(arm.max_bars_to_mfe),
        "max_barrier": float(arm.max_barrier),
        "regime_family": str(regime_family or "all"),
        "selected_rows": int(len(selected)),
        "target_top_hard_rate": float(hit.mean()) if len(selected) else float("nan"),
        "ev_weighted_first_touch_precision": gross_precision,
        "ev_weighted_clean_precision": gross_precision,
        "capture_hit_rate": float(hit.mean()) if len(selected) else float("nan"),
        "capture_stop_rate": float(stop.mean()) if len(selected) else float("nan"),
        "first_touch_timeout_rate": float(timeout.mean()) if len(selected) else float("nan"),
        "capture_net_mean": float(_safe_numeric(selected.get("capture_net", pd.Series(dtype=float))).mean())
        if len(selected)
        else float("nan"),
        "capture_gross_mean": float(gross.mean()) if len(selected) else float("nan"),
        "gross_minus_1pct_mean": float(gross_minus_1pct.mean()) if len(selected) else float("nan"),
        "executable_margin_mean": float(executable_margin.mean()) if len(selected) else float("nan"),
        "positive_gross_minus_1pct_rate": float(gross_minus_1pct.gt(0.0).mean()) if len(selected) else float("nan"),
        "positive_executable_margin_rate": float(executable_margin.gt(0.0).mean()) if len(selected) else float("nan"),
        "gross_hit_value_mean": hit_value,
        "gross_stop_value_mean": stop_value,
        "gross_timeout_value_mean": timeout_value,
        "bad_mae_1r_rate": float(
            first_touch_mae.ge(1.0).mean()
        )
        if len(selected)
        else float("nan"),
        "first_touch_bad_mae_1r_rate": float(first_touch_mae.ge(1.0).mean()) if len(selected) else float("nan"),
        "mfe_1r_before_mae_1r_rate": float(mfe_before_mae_1r.gt(0.5).mean()) if len(selected) else float("nan"),
        "mae_1r_before_mfe_1r_rate": float(mae_before_mfe_1r.gt(0.5).mean()) if len(selected) else float("nan"),
        "mean_max_adverse_before_mfe_1r": float(max_adverse_before_mfe_1r.mean()) if len(selected) else float("nan"),
        "p90_max_adverse_before_mfe_1r": float(max_adverse_before_mfe_1r.quantile(0.90))
        if len(max_adverse_before_mfe_1r.dropna())
        else float("nan"),
        "mean_underwater_bars_before_mfe_1r": float(underwater_bars_before_mfe_1r.mean())
        if len(selected)
        else float("nan"),
        "p90_underwater_bars_before_mfe_1r": float(underwater_bars_before_mfe_1r.quantile(0.90))
        if len(underwater_bars_before_mfe_1r.dropna())
        else float("nan"),
        "mean_underwater_fraction_before_mfe_1r": float(underwater_fraction_before_mfe_1r.mean())
        if len(selected)
        else float("nan"),
        "mean_area_underwater_before_mfe_1r": float(area_underwater_before_mfe_1r.mean())
        if len(selected)
        else float("nan"),
        "mae_to_sl_p90": float(np.nanquantile(mae_to_sl.to_numpy(dtype=np.float64), 0.90))
        if len(mae_to_sl.dropna())
        else float("nan"),
        "top_symbol_share": _selected_top_symbol_share(selected_rows),
    }


def _run_coarse_side(
    *,
    labels_path: Path,
    output_dir: Path,
    arms: list[Any],
    months: list[str],
    top_fracs: list[float],
    selection_modes: list[str],
    data_root: Path,
    market_mode: str,
    exchange: str,
    side: str,
    path_len: int,
    apply_delayed_entry: bool,
    outcome_mode: str,
    regime_families: list[str],
    round_trip_cost: float,
    target_mode: str,
    executable_cost_floor: float,
) -> pd.DataFrame:
    frame = _load_labels(labels_path)
    resolved_side = _infer_side(labels_path, side)
    _rows_exec, paths, _path_fetch_stats = _fetch_policy_paths(
        frame,
        labels_path=labels_path,
        side=resolved_side,
        data_root=data_root,
        market_mode=market_mode,
        exchange=exchange,
        path_len=path_len,
        apply_delayed_entry=apply_delayed_entry,
    )
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    requested_families = [str(v) for v in (regime_families or ["all"]) if str(v).strip()]
    if "auto" in requested_families:
        requested_families = ["all"]
    rows: list[dict[str, Any]] = []
    for i, arm in enumerate(arms, start=1):
        if i == 1 or i % 50 == 0 or i == len(arms):
            print(
                json.dumps(
                    {
                        "stage": "coarse_geometry",
                        "side": resolved_side,
                        "arm": i,
                        "total_arms": len(arms),
                    }
                ),
                flush=True,
            )
        target = _first_touch_capture_outcome(
            frame,
            paths,
            arm,
            side_name=resolved_side,
            outcome_mode=outcome_mode,
            round_trip_cost=float(round_trip_cost),
            target_mode=str(target_mode),
            executable_cost_floor=float(executable_cost_floor),
        )
        score_full = _rank_pct(target["target_soft"])
        for regime_family in requested_families:
            family_mask = pd.Series(True, index=frame.index)
            if regime_family and regime_family != "all" and "__regime_family__" in frame.columns:
                family_mask = frame["__regime_family__"].astype(str).eq(str(regime_family))
            for month in months:
                mask = month_period.eq(str(month)) & family_mask
                if int(mask.sum()) <= 0:
                    continue
                valid = frame.loc[mask].copy().reset_index(drop=True)
                valid_target = target.loc[mask].copy().reset_index(drop=True)
                valid_score = score_full.loc[mask].reset_index(drop=True)
                for selection_mode in selection_modes:
                    for top_frac in top_fracs:
                        rows.append(
                            _coarse_selection_metrics(
                                valid=valid,
                                target=valid_target,
                                score=valid_score,
                                arm=arm,
                                side=resolved_side,
                                month=str(month),
                                top_frac=float(top_frac),
                                selection_mode=str(selection_mode),
                                regime_family=str(regime_family or "all"),
                            )
                        )
    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "coarse_geometry_monthly.csv", index=False)
    return out


def _select_proxy_shortlist(
    arms: list[Any],
    coarse_side_family_summary: pd.DataFrame,
    *,
    shortlist_size: int,
) -> list[Any]:
    if shortlist_size <= 0 or coarse_side_family_summary.empty:
        return list(arms)
    ranked = coarse_side_family_summary.copy()
    ranked["fit_precision_geometry_score"] = ranked.apply(
        lambda row: _fit_precision_geometry_score(row.to_dict(), include_side_floor=False),
        axis=1,
    )
    ranked["fit_path_quality_geometry_score"] = ranked.apply(
        lambda row: _fit_path_quality_geometry_score(row.to_dict(), include_side_floor=False),
        axis=1,
    )
    if "fit_mean_capture_gross" not in ranked.columns:
        ranked["fit_mean_capture_gross"] = 0.0
    if "fit_mean_executable_margin" not in ranked.columns:
        ranked["fit_mean_executable_margin"] = 0.0
    ranked = ranked.sort_values(
        [
            "fit_path_quality_geometry_score",
            "fit_precision_geometry_score",
            "fit_gross_ev_weighted_first_touch_precision",
            "fit_mean_executable_margin",
            "fit_mean_capture_gross",
        ],
        ascending=[False, False, False, False, False],
    )
    keep: list[str] = []
    for _, row in ranked.iterrows():
        arm = str(row.get("arm", ""))
        if arm and arm not in keep:
            keep.append(arm)
        if len(keep) >= int(shortlist_size):
            break
    by_name = {str(arm.name): arm for arm in arms}
    return [by_name[name] for name in keep if name in by_name]


def _summarize_slice(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_selected_rows": 0,
            f"{prefix}_precision": float("nan"),
            f"{prefix}_stop_rate": float("nan"),
            f"{prefix}_timeout_rate": float("nan"),
        }
    first_touch_precision_col = (
        "ev_weighted_first_touch_precision"
        if "ev_weighted_first_touch_precision" in frame.columns
        else "ev_weighted_clean_precision"
    )
    gross_ev_weighted_precision = _gross_ev_weighted_first_touch_precision(frame)
    mean_capture_gross = _weighted_mean(frame, "capture_gross_mean")
    mean_capture_net = _weighted_mean(frame, "capture_net_mean")
    mean_gross_minus_1pct = _weighted_mean_any(
        frame,
        ("gross_minus_1pct_mean", "first_touch_gross_minus_1pct_mean", "first_touch_executable_margin_mean"),
    )
    mean_executable_margin = _weighted_mean_any(
        frame,
        ("executable_margin_mean", "first_touch_executable_margin_mean"),
    )
    implied_cost = (
        float(mean_capture_gross) - float(mean_capture_net)
        if np.isfinite(float(mean_capture_gross)) and np.isfinite(float(mean_capture_net))
        else float("nan")
    )
    cost_coverage = (
        float(mean_capture_gross) / float(implied_cost)
        if np.isfinite(float(mean_capture_gross)) and np.isfinite(float(implied_cost)) and abs(float(implied_cost)) > 1e-12
        else float("nan")
    )
    if not np.isfinite(float(mean_gross_minus_1pct)) and np.isfinite(float(mean_capture_gross)):
        mean_gross_minus_1pct = float(mean_capture_gross) - float(EXECUTABLE_MARGIN_COST_FLOOR)
    if not np.isfinite(float(mean_executable_margin)) and np.isfinite(float(mean_capture_gross)):
        cost_floor = max(
            float(EXECUTABLE_MARGIN_COST_FLOOR),
            float(implied_cost) if np.isfinite(float(implied_cost)) else float(EXECUTABLE_MARGIN_COST_FLOOR),
        )
        mean_executable_margin = float(mean_capture_gross) - cost_floor
    return {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_selected_rows": int(_safe_numeric(frame["selected_rows"]).sum()),
        f"{prefix}_precision": _weighted_mean(frame, "target_top_hard_rate"),
        f"{prefix}_gross_ev_weighted_first_touch_precision": gross_ev_weighted_precision,
        f"{prefix}_ev_weighted_first_touch_precision": gross_ev_weighted_precision,
        f"{prefix}_ev_weighted_precision": _gross_ev_weighted_first_touch_precision(frame)
        if "ev_weighted_clean_precision" in frame.columns
        else _weighted_mean(frame, first_touch_precision_col),
        f"{prefix}_hit_rate": _weighted_mean(frame, "capture_hit_rate"),
        f"{prefix}_stop_rate": _weighted_mean(frame, "capture_stop_rate"),
        f"{prefix}_timeout_rate": _weighted_mean(frame, "first_touch_timeout_rate"),
        f"{prefix}_mean_capture_net": mean_capture_net,
        f"{prefix}_mean_capture_gross": mean_capture_gross,
        f"{prefix}_mean_gross_minus_1pct": mean_gross_minus_1pct,
        f"{prefix}_mean_executable_margin": mean_executable_margin,
        f"{prefix}_positive_gross_minus_1pct_rate": _weighted_mean_any(
            frame,
            ("positive_gross_minus_1pct_rate", "first_touch_executable_margin_positive_rate"),
        ),
        f"{prefix}_positive_executable_margin_rate": _weighted_mean_any(
            frame,
            ("positive_executable_margin_rate", "first_touch_executable_margin_positive_rate"),
        ),
        f"{prefix}_implied_round_trip_cost": implied_cost,
        f"{prefix}_cost_coverage_ratio": cost_coverage,
        f"{prefix}_gross_hit_value_mean": _weighted_mean(frame, "gross_hit_value_mean"),
        f"{prefix}_gross_stop_value_mean": _weighted_mean(frame, "gross_stop_value_mean"),
        f"{prefix}_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate"),
        f"{prefix}_selected_path_bad_mae_1r_rate": _weighted_mean(frame, "selected_path_bad_mae_1r_rate"),
        f"{prefix}_selected_path_p90_mae_norm": _weighted_mean(frame, "selected_path_p90_mae_norm"),
        f"{prefix}_first_touch_bad_mae_1r_rate": _weighted_mean(frame, "first_touch_bad_mae_1r_rate"),
        f"{prefix}_first_touch_p90_mae_norm": _weighted_mean(frame, "first_touch_p90_mae_norm"),
        f"{prefix}_first_touch_mae_to_sl_p90": _weighted_mean(frame, "first_touch_mae_to_sl_p90"),
        f"{prefix}_mfe_1r_before_mae_1r_rate": _weighted_mean(frame, "mfe_1r_before_mae_1r_rate"),
        f"{prefix}_mae_1r_before_mfe_1r_rate": _weighted_mean(frame, "mae_1r_before_mfe_1r_rate"),
        f"{prefix}_mean_max_adverse_before_mfe_1r": _weighted_mean(
            frame, "mean_max_adverse_before_mfe_1r"
        ),
        f"{prefix}_p90_max_adverse_before_mfe_1r": _weighted_mean(frame, "p90_max_adverse_before_mfe_1r"),
        f"{prefix}_mean_underwater_bars_before_mfe_1r": _weighted_mean(
            frame, "mean_underwater_bars_before_mfe_1r"
        ),
        f"{prefix}_p90_underwater_bars_before_mfe_1r": _weighted_mean(
            frame, "p90_underwater_bars_before_mfe_1r"
        ),
        f"{prefix}_mean_underwater_fraction_before_mfe_1r": _weighted_mean(
            frame, "mean_underwater_fraction_before_mfe_1r"
        ),
        f"{prefix}_mean_area_underwater_before_mfe_1r": _weighted_mean(
            frame, "mean_area_underwater_before_mfe_1r"
        ),
        f"{prefix}_target_full_path_bad_mae_1r_rate": _weighted_mean(frame, "target_full_path_bad_mae_1r_rate"),
        f"{prefix}_target_full_path_mae_to_sl_p90": _weighted_mean(frame, "target_full_path_mae_to_sl_p90"),
        f"{prefix}_mae_to_sl_p90": _weighted_mean(frame, "mae_to_sl_p90"),
        f"{prefix}_top_symbol_share": float(_safe_numeric(frame["top_symbol_share"]).max())
        if "top_symbol_share" in frame.columns
        else float("nan"),
    }


def _build_summary(
    monthly: pd.DataFrame,
    *,
    fit_months: list[str],
    holdout_month: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["arm", "selection_mode", "top_frac"]
    if "regime_family" in monthly.columns:
        group_cols.append("regime_family")
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        key_dict = dict(zip(group_cols, key))
        row: dict[str, Any] = dict(key_dict)
        for col in ("tp_r", "sl_r", "trail_r", "max_bars_to_mfe", "max_barrier"):
            vals = _safe_numeric(group[col]).dropna() if col in group.columns else pd.Series(dtype=float)
            row[col] = float(vals.iloc[0]) if len(vals) else float("nan")
        row.update(_summarize_slice("all", group))
        row.update(_summarize_slice("fit", group[group["period"].astype(str).isin(fit_months)]))
        row.update(_summarize_slice("holdout", group[group["period"].astype(str).eq(str(holdout_month))]))
        side_precisions = []
        side_ev_precisions = []
        side_stops = []
        for side in ("long", "short"):
            side_group = group[group["side"].astype(str).eq(side)]
            holdout_side = side_group[side_group["period"].astype(str).eq(str(holdout_month))]
            row[f"{side}_all_precision"] = _weighted_mean(side_group, "target_top_hard_rate")
            row[f"{side}_all_gross_ev_weighted_first_touch_precision"] = _gross_ev_weighted_first_touch_precision(
                side_group
            )
            row[f"{side}_all_ev_weighted_first_touch_precision"] = row[
                f"{side}_all_gross_ev_weighted_first_touch_precision"
            ]
            row[f"{side}_all_ev_weighted_precision"] = row[f"{side}_all_gross_ev_weighted_first_touch_precision"]
            row[f"{side}_holdout_precision"] = _weighted_mean(holdout_side, "target_top_hard_rate")
            row[f"{side}_holdout_gross_ev_weighted_first_touch_precision"] = (
                _gross_ev_weighted_first_touch_precision(holdout_side)
            )
            row[f"{side}_holdout_ev_weighted_first_touch_precision"] = row[
                f"{side}_holdout_gross_ev_weighted_first_touch_precision"
            ]
            row[f"{side}_holdout_ev_weighted_precision"] = row[
                f"{side}_holdout_gross_ev_weighted_first_touch_precision"
            ]
            row[f"{side}_holdout_stop_rate"] = _weighted_mean(holdout_side, "capture_stop_rate")
            row[f"{side}_holdout_timeout_rate"] = _weighted_mean(holdout_side, "first_touch_timeout_rate")
            if np.isfinite(row[f"{side}_holdout_precision"]):
                side_precisions.append(float(row[f"{side}_holdout_precision"]))
            if np.isfinite(row[f"{side}_holdout_ev_weighted_first_touch_precision"]):
                side_ev_precisions.append(float(row[f"{side}_holdout_ev_weighted_first_touch_precision"]))
            if np.isfinite(row[f"{side}_holdout_stop_rate"]):
                side_stops.append(float(row[f"{side}_holdout_stop_rate"]))
        row["min_side_holdout_precision"] = min(side_precisions) if side_precisions else float("nan")
        row["min_side_holdout_ev_weighted_precision"] = min(side_ev_precisions) if side_ev_precisions else float("nan")
        row["max_side_holdout_stop_rate"] = max(side_stops) if side_stops else float("nan")
        row["precision_geometry_score"] = _precision_geometry_score(row, include_side_floor=True)
        row["path_quality_geometry_score"] = _path_quality_geometry_score(row, include_side_floor=True)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "path_quality_geometry_score",
            "precision_geometry_score",
            "holdout_ev_weighted_first_touch_precision",
            "holdout_mean_executable_margin",
            "min_side_holdout_ev_weighted_precision",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def _build_side_family_summary(
    monthly: pd.DataFrame,
    *,
    fit_months: list[str],
    holdout_month: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["side", "arm", "selection_mode", "top_frac"]
    if "regime_family" in monthly.columns:
        group_cols.append("regime_family")
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        row: dict[str, Any] = dict(zip(group_cols, key))
        for col in ("tp_r", "sl_r", "trail_r", "max_bars_to_mfe", "max_barrier"):
            vals = _safe_numeric(group[col]).dropna() if col in group.columns else pd.Series(dtype=float)
            row[col] = float(vals.iloc[0]) if len(vals) else float("nan")
        row.update(_summarize_slice("all", group))
        row.update(_summarize_slice("fit", group[group["period"].astype(str).isin(fit_months)]))
        row.update(_summarize_slice("holdout", group[group["period"].astype(str).eq(str(holdout_month))]))
        row["precision_geometry_score"] = _precision_geometry_score(row, include_side_floor=False)
        row["path_quality_geometry_score"] = _path_quality_geometry_score(row, include_side_floor=False)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "side",
            "regime_family",
            "path_quality_geometry_score",
            "precision_geometry_score",
            "holdout_gross_ev_weighted_first_touch_precision",
        ],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)


def _build_side_family_winners(side_family_summary: pd.DataFrame) -> pd.DataFrame:
    if side_family_summary.empty:
        return side_family_summary.copy()
    group_cols = ["side"]
    if "regime_family" in side_family_summary.columns:
        group_cols.append("regime_family")
    ranked = side_family_summary.sort_values(
        [
            *group_cols,
            "path_quality_geometry_score",
            "precision_geometry_score",
            "holdout_gross_ev_weighted_first_touch_precision",
            "holdout_mean_executable_margin",
            "holdout_mean_capture_gross",
            "holdout_mean_capture_net",
        ],
        ascending=[True] * len(group_cols) + [False, False, False, False, False, False],
    )
    return ranked.groupby(group_cols, observed=True, dropna=False).head(1).reset_index(drop=True)


def _write_report(output_dir: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "s52_bidirectional_geometry_sweep.md"
    cols = [
        "arm",
        "selection_mode",
        "top_frac",
        "tp_r",
        "sl_r",
        "trail_r",
        "max_bars_to_mfe",
        "max_barrier",
        "regime_family",
        "path_quality_geometry_score",
        "precision_geometry_score",
        "fit_gross_ev_weighted_first_touch_precision",
        "fit_ev_weighted_first_touch_precision",
        "fit_ev_weighted_precision",
        "holdout_gross_ev_weighted_first_touch_precision",
        "holdout_ev_weighted_first_touch_precision",
        "holdout_ev_weighted_precision",
        "min_side_holdout_ev_weighted_precision",
        "fit_precision",
        "holdout_precision",
        "min_side_holdout_precision",
        "long_holdout_gross_ev_weighted_first_touch_precision",
        "long_holdout_ev_weighted_first_touch_precision",
        "long_holdout_ev_weighted_precision",
        "short_holdout_gross_ev_weighted_first_touch_precision",
        "short_holdout_ev_weighted_first_touch_precision",
        "short_holdout_ev_weighted_precision",
        "long_holdout_precision",
        "short_holdout_precision",
        "holdout_stop_rate",
        "holdout_timeout_rate",
        "holdout_selected_path_bad_mae_1r_rate",
        "holdout_first_touch_bad_mae_1r_rate",
        "holdout_mfe_1r_before_mae_1r_rate",
        "holdout_mae_1r_before_mfe_1r_rate",
        "holdout_mean_max_adverse_before_mfe_1r",
        "holdout_p90_max_adverse_before_mfe_1r",
        "holdout_mean_underwater_bars_before_mfe_1r",
        "holdout_p90_underwater_bars_before_mfe_1r",
        "holdout_mean_underwater_fraction_before_mfe_1r",
        "holdout_target_full_path_bad_mae_1r_rate",
        "holdout_mae_to_sl_p90",
        "holdout_first_touch_mae_to_sl_p90",
        "holdout_target_full_path_mae_to_sl_p90",
        "holdout_mean_capture_gross",
        "holdout_mean_capture_net",
        "holdout_mean_gross_minus_1pct",
        "holdout_mean_executable_margin",
        "holdout_positive_executable_margin_rate",
        "fit_mean_executable_margin",
        "fit_positive_executable_margin_rate",
        "holdout_cost_coverage_ratio",
        "fit_cost_coverage_ratio",
        "holdout_selected_rows",
    ]
    lines = [
        "# S52 Bidirectional First-Touch Geometry Sweep",
        "",
        "Scope: base-layer geometry search. Ranking uses gross-EV-weighted precision@top-k for clean first-touch/trailing outcomes with an explicit executable-margin term against a 1% round-trip cost floor. Stop rate, timeout, and MAE remain path-quality constraints.",
        "",
        f"Labels dir: `{manifest['labels_dir']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Proxy arms: `{manifest['arm_count']}` (coarse full-grid arms: `{manifest.get('original_arm_count', manifest['arm_count'])}`)",
        f"Outcome mode: `{manifest['outcome_mode']}`",
        f"Target mode: `{manifest['target_mode']}`",
        f"Executable cost floor: `{manifest['executable_cost_floor']}`",
        f"Regime families: `{', '.join(manifest['regime_families'])}`",
        f"Top fractions: `{', '.join(str(v) for v in manifest['top_fracs'])}`",
        "",
        "## Top Geometries",
        "",
        _format_table(summary, cols, limit=50),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Coarse full-grid summary: `{manifest['outputs'].get('coarse_summary', '')}`",
        f"- Side x regime-family summary: `{manifest['outputs']['side_family_summary']}`",
        f"- Side x regime-family winners: `{manifest['outputs']['side_family_winners']}`",
        f"- Monthly combined: `{manifest['outputs']['monthly']}`",
        f"- Diagnostics combined: `{manifest['outputs']['diagnostics']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_sweep(
    *,
    labels_dir: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    tp_rs: list[float],
    sl_rs: list[float],
    trail_rs: list[float],
    fast_bars: list[float],
    max_barriers: list[float],
    top_fracs: list[float],
    selection_modes: list[str],
    seeds: list[int],
    max_feature_store_features: int | None,
    max_weight: float,
    min_weight: float,
    min_week_rows: int,
    data_root: Path,
    market_mode: str,
    exchange: str,
    path_len: int,
    apply_delayed_entry: bool,
    outcome_mode: str,
    regime_families: list[str],
    round_trip_cost: float,
    target_mode: str,
    executable_cost_floor: float,
    sides: list[str],
    coarse_shortlist_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_arms = _build_grid_arms(
        tp_rs=tp_rs,
        sl_rs=sl_rs,
        trail_rs=trail_rs,
        fast_bars=fast_bars,
        max_barriers=max_barriers,
        prefix="S52g",
    )
    if not all_arms:
        raise ValueError("Geometry grid produced no arms")

    monthly_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    fit_holdout_parts: list[pd.DataFrame] = []
    side_manifests: dict[str, Any] = {}
    resolved_sides = [str(side).strip().lower() for side in sides if str(side).strip()]
    if not resolved_sides:
        resolved_sides = ["long", "short"]
    invalid_sides = sorted(set(resolved_sides).difference({"long", "short"}))
    if invalid_sides:
        raise ValueError(f"Unsupported sides: {invalid_sides}")
    coarse_parts: list[pd.DataFrame] = []
    for side in resolved_sides:
        labels_path = _side_file(labels_dir, side)
        coarse_side = _run_coarse_side(
            labels_path=labels_path,
            output_dir=output_dir / side,
            arms=all_arms,
            months=months,
            top_fracs=top_fracs,
            selection_modes=selection_modes,
            data_root=data_root,
            market_mode=market_mode,
            exchange=exchange,
            side=side,
            path_len=path_len,
            apply_delayed_entry=apply_delayed_entry,
            outcome_mode=outcome_mode,
            regime_families=regime_families,
            round_trip_cost=round_trip_cost,
            target_mode=str(target_mode),
            executable_cost_floor=float(executable_cost_floor),
        )
        coarse_parts.append(coarse_side)
    coarse_monthly = pd.concat(coarse_parts, ignore_index=True) if coarse_parts else pd.DataFrame()
    coarse_summary = _build_summary(coarse_monthly, fit_months=fit_months, holdout_month=holdout_month)
    coarse_side_family_summary = _build_side_family_summary(
        coarse_monthly,
        fit_months=fit_months,
        holdout_month=holdout_month,
    )
    coarse_side_family_winners = _build_side_family_winners(coarse_side_family_summary)
    coarse_paths = {
        "coarse_monthly": output_dir / "s52_bidirectional_geometry_coarse_monthly.csv",
        "coarse_summary": output_dir / "s52_bidirectional_geometry_coarse_summary.csv",
        "coarse_side_family_summary": output_dir / "s52_bidirectional_geometry_coarse_side_family_summary.csv",
        "coarse_side_family_winners": output_dir / "s52_bidirectional_geometry_coarse_side_family_winners.csv",
    }
    coarse_monthly.to_csv(coarse_paths["coarse_monthly"], index=False)
    coarse_summary.to_csv(coarse_paths["coarse_summary"], index=False)
    coarse_side_family_summary.to_csv(coarse_paths["coarse_side_family_summary"], index=False)
    coarse_side_family_winners.to_csv(coarse_paths["coarse_side_family_winners"], index=False)
    arms = _select_proxy_shortlist(
        all_arms,
        coarse_side_family_summary,
        shortlist_size=int(coarse_shortlist_size),
    )
    if not arms:
        raise ValueError("Coarse geometry shortlist produced no arms")
    print(
        json.dumps(
            {
                "stage": "proxy_shortlist",
                "original_arms": len(all_arms),
                "proxy_arms": len(arms),
                "coarse_shortlist_size": int(coarse_shortlist_size),
            }
        ),
        flush=True,
    )
    for side in resolved_sides:
        labels_path = _side_file(labels_dir, side)
        side_output = output_dir / side
        manifest = run_proxy(
            labels_path=labels_path,
            output_dir=side_output,
            feature_dir=feature_dir,
            feature_list_csv=feature_list_csv,
            max_feature_store_features=max_feature_store_features,
            months=months,
            arm_names=[arm.name for arm in arms],
            custom_arms=arms,
            only_custom_arms=True,
            top_fracs=top_fracs,
            selection_modes=selection_modes,
            seeds=seeds,
            train_lookback_months=None,
            max_weight=max_weight,
            min_weight=min_weight,
            min_week_rows=min_week_rows,
            data_root=data_root,
            market_mode=market_mode,
            exchange=exchange,
            side=side,
            path_len=path_len,
            apply_delayed_entry=apply_delayed_entry,
            outcome_mode=outcome_mode,
            regime_families=regime_families,
            round_trip_cost=float(round_trip_cost),
            target_mode=str(target_mode),
            executable_cost_floor=float(executable_cost_floor),
        )
        side_manifests[side] = manifest
        monthly = pd.read_csv(side_output / "label_first_touch_capture_proxy_monthly.csv")
        monthly["side"] = side
        monthly_parts.append(monthly)
        diagnostics = pd.read_csv(side_output / "label_first_touch_capture_proxy_diagnostics.csv")
        diagnostics["side"] = side
        diagnostic_parts.append(diagnostics)
        fit_holdout = pd.read_csv(side_output / "label_first_touch_capture_proxy_fit_holdout.csv")
        fit_holdout["side"] = side
        fit_holdout_parts.append(fit_holdout)

    combined_monthly = pd.concat(monthly_parts, ignore_index=True)
    combined_diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
    combined_fit_holdout = pd.concat(fit_holdout_parts, ignore_index=True)
    summary = _build_summary(combined_monthly, fit_months=fit_months, holdout_month=holdout_month)
    side_family_summary = _build_side_family_summary(
        combined_monthly,
        fit_months=fit_months,
        holdout_month=holdout_month,
    )
    side_family_winners = _build_side_family_winners(side_family_summary)

    paths = {
        "monthly": output_dir / "s52_bidirectional_geometry_monthly.csv",
        "diagnostics": output_dir / "s52_bidirectional_geometry_diagnostics.csv",
        "side_fit_holdout": output_dir / "s52_bidirectional_geometry_side_fit_holdout.csv",
        "side_family_summary": output_dir / "s52_bidirectional_geometry_side_family_summary.csv",
        "side_family_winners": output_dir / "s52_bidirectional_geometry_side_family_winners.csv",
        "summary": output_dir / "s52_bidirectional_geometry_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    combined_monthly.to_csv(paths["monthly"], index=False)
    combined_diagnostics.to_csv(paths["diagnostics"], index=False)
    combined_fit_holdout.to_csv(paths["side_fit_holdout"], index=False)
    side_family_summary.to_csv(paths["side_family_summary"], index=False)
    side_family_winners.to_csv(paths["side_family_winners"], index=False)
    summary.to_csv(paths["summary"], index=False)
    manifest = {
        "scope": "s52_bidirectional_first_touch_geometry_sweep",
        "labels_dir": str(labels_dir),
        "output_dir": str(output_dir),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "months": months,
        "fit_months": fit_months,
        "holdout_month": holdout_month,
        "tp_rs": tp_rs,
        "sl_rs": sl_rs,
        "trail_rs": trail_rs,
        "fast_bars": fast_bars,
        "max_barriers": max_barriers,
        "arm_count": int(len(arms)),
        "original_arm_count": int(len(all_arms)),
        "proxy_arm_count": int(len(arms)),
        "coarse_shortlist_size": int(coarse_shortlist_size),
        "top_fracs": top_fracs,
        "selection_modes": selection_modes,
        "seeds": seeds,
        "max_feature_store_features": max_feature_store_features,
        "max_weight": float(max_weight),
        "min_weight": float(min_weight),
        "min_week_rows": int(min_week_rows),
        "data_root": str(data_root),
        "market_mode": str(market_mode),
        "exchange": str(exchange),
        "path_len": int(path_len),
        "apply_delayed_entry": bool(apply_delayed_entry),
        "outcome_mode": str(outcome_mode),
        "target_mode": str(target_mode),
        "executable_cost_floor": float(executable_cost_floor),
        "regime_families": [str(v) for v in regime_families],
        "round_trip_cost": float(round_trip_cost),
        "side_manifests": side_manifests,
        "sides": resolved_sides,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    manifest["outputs"].update({key: str(value) for key, value in coarse_paths.items()})
    report = _write_report(output_dir, summary, manifest)
    manifest["outputs"]["markdown"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--months", default=DEFAULT_MONTHS)
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--tp-rs", default="1.0,1.25,1.5,1.75")
    parser.add_argument("--sl-rs", default="0.75,1.0,1.25,1.5")
    parser.add_argument("--trail-rs", default="0.35,0.50,0.75")
    parser.add_argument("--fast-bars", default="8,12,16")
    parser.add_argument("--max-barriers", default="0.03,0.05")
    parser.add_argument("--top-fracs", default="0.10,0.20,0.30")
    parser.add_argument("--selection-modes", default="global")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-weight", type=float, default=12.0)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--no-delayed-entry", action="store_true")
    parser.add_argument("--outcome-mode", choices=("fixed_tp", "trailing_profit"), default="fixed_tp")
    parser.add_argument("--round-trip-cost", type=float, default=float(ROUND_TRIP_COST))
    parser.add_argument(
        "--target-mode",
        choices=("path_ordered", "executable_margin", "executable_margin_hybrid"),
        default="path_ordered",
        help="Soft-label target mode used by the trained proxy stage.",
    )
    parser.add_argument("--executable-cost-floor", type=float, default=float(EXECUTABLE_MARGIN_COST_FLOOR))
    parser.add_argument("--sides", default="long,short", help="Comma-separated sides: long,short.")
    parser.add_argument(
        "--coarse-shortlist-size",
        type=int,
        default=48,
        help="Run the expensive trained proxy only on the top N arms from the full-grid coarse path scan. Use 0 for all arms.",
    )
    parser.add_argument(
        "--regime-families",
        default="all",
        help="Comma-separated regime families. Use auto for observed pre-entry families.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_sweep(
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        months=_parse_csv(args.months, ()),
        fit_months=_parse_csv(args.fit_months, DEFAULT_FIT_MONTHS),
        holdout_month=str(args.holdout_month),
        tp_rs=_parse_float_csv(args.tp_rs, ()),
        sl_rs=_parse_float_csv(args.sl_rs, ()),
        trail_rs=_parse_float_csv(args.trail_rs, (0.50,)),
        fast_bars=_parse_float_csv(args.fast_bars, ()),
        max_barriers=_parse_float_csv(args.max_barriers, ()),
        top_fracs=_parse_float_csv(args.top_fracs, (0.10, 0.20, 0.30)),
        selection_modes=_parse_csv(args.selection_modes, ("global",)),
        seeds=_parse_int_csv(args.seeds, (42,)),
        max_feature_store_features=args.max_feature_store_features,
        max_weight=float(args.max_weight),
        min_weight=float(args.min_weight),
        min_week_rows=int(args.min_week_rows),
        data_root=args.data_root,
        market_mode=str(args.market_mode),
        exchange=str(args.exchange),
        path_len=int(args.path_len),
        apply_delayed_entry=not bool(args.no_delayed_entry),
        outcome_mode=str(args.outcome_mode),
        regime_families=_parse_csv(args.regime_families, ("all",)),
        round_trip_cost=float(args.round_trip_cost),
        target_mode=str(args.target_mode),
        executable_cost_floor=float(args.executable_cost_floor),
        sides=_parse_csv(args.sides, ("long", "short")),
        coarse_shortlist_size=int(args.coarse_shortlist_size),
    )
    summary_path = manifest["outputs"]["summary"]
    summary = pd.read_csv(summary_path)
    print(json.dumps(_json_safe({
        "output_dir": manifest["output_dir"],
        "summary": summary_path,
        "top": summary.head(5).to_dict("records"),
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
