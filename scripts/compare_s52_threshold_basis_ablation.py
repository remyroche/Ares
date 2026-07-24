#!/usr/bin/env python3
"""Compare live-compatible S52 threshold-basis policies.

The script keeps base/meta candidates, regime calibration, execution geometry,
fees, and portfolio replay fixed.  It changes only how raw/regime-calibrated
meta scores are converted into an admission rank/threshold before the global
portfolio auction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts.ablate_simple_policy_exit_geometry import (  # noqa: E402
    _load_bundles,
    _prepare_rows,
)
from scripts.ablate_s52_archetype_hit_surprise_thresholds import (  # noqa: E402
    PROMOTED_ALPHA,
    PROMOTED_HALF_LIFE_DAYS,
    PROMOTED_HIT_SURPRISE_MODE,
    PROMOTED_MAX_ADJUST,
    PROMOTED_MAX_CONCURRENT_POSITIONS,
    PROMOTED_MAX_NEW_ENTRIES_PER_BAR,
    PROMOTED_POLICY_NAME,
    PROMOTED_TOP_SLICE,
    TOP_THRESHOLDS,
    _apply_portfolio_hr_adjustments,
    _expected_probability,
    _portfolio_candidate_table,
    _safe_float,
    _simulate_selected_rows,
    _surprise_quality_map,
    _weighted_surprise,
)
from scripts.run_s52_side_archetype_simple_policy_optimiser import (  # noqa: E402
    _params_from_parent_summary_row,
)


DEFAULT_REPORT_DIR = ROOT / (
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260708_hr_threshold_modulation_top15_top5_"
    "protected_regime_rank_retained50"
)
DEFAULT_CANDIDATES = ROOT / (
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "s52_simple_policy_replay_candidates_meta_base_soft_label_top30_no_cap_oos15_20260707_v2/"
    "simple_policy_candidates_with_archetypes.parquet"
)
DEFAULT_PARENT_POLICY_SUMMARY = ROOT / (
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260707_v4_trials96_juneh2_holdout_archfix/"
    "side_parent_policy_summary.csv"
)


@dataclass(frozen=True)
class Arm:
    name: str
    family: str
    window_days: int | None
    description: str
    zscore_method: str = "flat"
    blend_weight: float = 0.5
    calibration_timing: str = "before"
    hr_rank50: bool = False
    posterior_power: float = 1.0
    posterior_threshold: float | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _load_parent_summary(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    if "strategy_id" not in frame.columns:
        raise ValueError(f"Parent summary missing strategy_id: {path}")
    return {str(row["strategy_id"]): row.to_dict() for _, row in frame.iterrows()}


def _policy_params() -> PortfolioPolicyParams:
    return PortfolioPolicyParams(
        max_concurrent_positions=int(PROMOTED_MAX_CONCURRENT_POSITIONS),
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=None,
        max_concurrent_per_symbol=1,
        max_new_entries_per_bar=int(PROMOTED_MAX_NEW_ENTRIES_PER_BAR),
        max_total_wallet_allocation_pct=0.75,
        global_threshold_floor=0.0,
        occupancy_threshold_alpha=0.30,
        occupancy_threshold_power=1.5,
        allocation_threshold_alpha=0.30,
        allocation_threshold_power=1.0,
        rank_size_power=1.5,
        rank_multiplier_min=0.5,
        rank_multiplier_max=1.5,
        min_position_size=1.0,
    )


def _score_col(rows: pd.DataFrame) -> str:
    for col in ("calibrated_score_regime_ev", "score_regime_calibrated", "calibrated_score", "meta_score_oof"):
        if col in rows.columns and pd.to_numeric(rows[col], errors="coerce").notna().any():
            return col
    raise ValueError("No usable score column found.")


def _raw_score_col(rows: pd.DataFrame) -> str:
    for col in ("calibrated_score", "meta_score_oof", "score_meta_base_soft_label", "base_score_oof"):
        if col in rows.columns and pd.to_numeric(rows[col], errors="coerce").notna().any():
            return col
    return _score_col(rows)


def _week_start(ts: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ts, utc=True, errors="coerce").dt.floor("D")
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def _add_time_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["eval_day"] = out["timestamp"].dt.floor("D")
    out["month"] = out["timestamp"].dt.strftime("%Y-%m")
    out["week_start"] = _week_start(out["timestamp"])
    if "policy_archetype" not in out.columns:
        out["policy_archetype"] = out.get("local_side_archetype", "missing").astype(str)
    return out


def _rank_against_ref(score: pd.Series, ref_score: pd.Series) -> pd.Series:
    ref = pd.to_numeric(ref_score, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ref_arr = np.sort(ref.to_numpy(dtype=np.float64, copy=False))
    out = np.full(len(score), np.nan, dtype=np.float64)
    if ref_arr.size == 0:
        return pd.Series(out, index=score.index)
    values = pd.to_numeric(score, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    out[finite] = np.searchsorted(ref_arr, values[finite], side="right") / float(ref_arr.size)
    return pd.Series(out, index=score.index).clip(0.0, 1.0)


def _posterior_cols(rows: pd.DataFrame) -> list[str]:
    """Return usable AE/GMM posterior columns in component order."""
    candidates: list[tuple[int, str]] = []
    for prefix in ("gmm_cluster_posterior_", "gmm_prob_"):
        for col in rows.columns:
            if not str(col).startswith(prefix):
                continue
            suffix = str(col)[len(prefix) :]
            if suffix.isdigit():
                candidates.append((int(suffix), str(col)))
        if candidates:
            break
    ordered = [col for _, col in sorted(candidates)]
    if not ordered:
        return []
    # Ignore all-zero padding components emitted by the AE/GMM transformer.
    usable = []
    for col in ordered:
        values = pd.to_numeric(rows[col], errors="coerce")
        if values.notna().any() and float(values.fillna(0.0).abs().sum()) > 1e-9:
            usable.append(col)
    return usable


def _effective_n(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    sw = float(w.sum())
    sw2 = float(np.square(w).sum())
    if sw <= 0.0 or sw2 <= 0.0:
        return 0.0
    return float((sw * sw) / sw2)


def _soft_posterior_matrix(
    rows: pd.DataFrame,
    cols: list[str],
    *,
    power: float,
    threshold: float | None,
) -> np.ndarray:
    if not cols or rows.empty:
        return np.zeros((len(rows), 0), dtype=np.float64)
    probs = rows[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    probs = np.clip(probs, 0.0, 1.0)
    if threshold is not None and np.isfinite(float(threshold)):
        probs = np.where(probs >= float(threshold), probs, 0.0)
    p = max(float(power), 1e-6)
    if abs(p - 1.0) > 1e-9:
        probs = np.power(probs, p)
    return np.where(np.isfinite(probs), probs, 0.0)


def _normalized_row_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    sums = np.sum(w, axis=1, keepdims=True)
    return np.divide(w, sums, out=np.zeros_like(w), where=sums > 1e-12)


def _weighted_rank_against_ref(
    score: pd.Series,
    ref_score: pd.Series,
    ref_weight: pd.Series | np.ndarray,
    *,
    min_effective_rows: int,
) -> pd.Series:
    values = pd.to_numeric(ref_score, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    weights = np.asarray(ref_weight, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if _effective_n(weights[valid]) < float(min_effective_rows):
        return pd.Series(np.nan, index=score.index, dtype="float64")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    total = float(cdf[-1]) if cdf.size else 0.0
    out = np.full(len(score), np.nan, dtype=np.float64)
    if total <= 0.0:
        return pd.Series(out, index=score.index)
    query = pd.to_numeric(score, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(query)
    pos = np.searchsorted(values, query[finite], side="right") - 1
    ranked = np.zeros(int(finite.sum()), dtype=np.float64)
    ok = pos >= 0
    ranked[ok] = cdf[pos[ok]] / total
    out[finite] = ranked
    return pd.Series(out, index=score.index).clip(0.0, 1.0)


def _weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    val = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(val) & np.isfinite(w) & (w > 0.0)
    if not valid.any():
        return float("nan")
    denom = float(w[valid].sum())
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(val[valid] * w[valid]) / denom)


def _weighted_historical_current_top10_ev(
    ref: pd.DataFrame,
    weights: np.ndarray,
    *,
    min_effective_rows: int,
) -> float:
    rank = pd.to_numeric(ref.get("rank_pct"), errors="coerce")
    ev = pd.to_numeric(ref.get("ret_net_notional"), errors="coerce")
    w = np.asarray(weights, dtype=np.float64)
    valid = rank.notna().to_numpy() & ev.notna().to_numpy() & np.isfinite(w) & (w > 0.0)
    chosen = valid & rank.ge(0.90).to_numpy()
    if _effective_n(w[valid]) < float(min_effective_rows) or _effective_n(w[chosen]) < max(1.0, float(min_effective_rows) / 2.0):
        return float("nan")
    return _weighted_mean(ev.loc[chosen], w[chosen])


def _weighted_threshold_for_target_ev(
    ref: pd.DataFrame,
    weights: np.ndarray,
    *,
    score_col: str,
    target_ev: float,
    min_effective_rows: int,
) -> float:
    if not np.isfinite(float(target_ev)):
        return float("nan")
    score = pd.to_numeric(ref[score_col], errors="coerce")
    ev = pd.to_numeric(ref["ret_net_notional"], errors="coerce")
    w = np.asarray(weights, dtype=np.float64)
    valid_mask = score.notna().to_numpy() & ev.notna().to_numpy() & np.isfinite(w) & (w > 0.0)
    if _effective_n(w[valid_mask]) < float(min_effective_rows):
        return float("nan")
    valid_score = score.to_numpy(dtype=np.float64, copy=False)[valid_mask]
    grid = np.unique(np.nanquantile(valid_score, np.linspace(0.70, 0.99, 60)))
    best_threshold = float("nan")
    best_gap = float("inf")
    ev_values = ev.to_numpy(dtype=np.float64, copy=False)
    score_values = score.to_numpy(dtype=np.float64, copy=False)
    for threshold in grid:
        chosen = valid_mask & (score_values >= float(threshold))
        if _effective_n(w[chosen]) < float(min_effective_rows):
            continue
        mean_ev = _weighted_mean(pd.Series(ev_values[chosen]), w[chosen])
        gap = abs(mean_ev - float(target_ev))
        if np.isfinite(mean_ev) and mean_ev >= float(target_ev) and gap < best_gap:
            best_gap = gap
            best_threshold = float(threshold)
    if not np.isfinite(best_threshold):
        best_threshold = float(np.nanquantile(valid_score, 0.99))
    return best_threshold


def _gmm_component_recent_quality(
    ref: pd.DataFrame,
    posterior_cols: list[str],
    *,
    power: float,
    threshold: float | None,
    min_effective_rows: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Posterior-weighted recent alpha / hit quality per GMM component."""
    n_comp = len(posterior_cols)
    if ref.empty or n_comp == 0:
        return np.zeros(n_comp, dtype=np.float64), {"baseline_ev": np.nan, "baseline_hit": np.nan}
    eligible = ref.loc[pd.to_numeric(ref.get("rank_pct"), errors="coerce").ge(0.90)].copy()
    if len(eligible) < max(1, int(min_effective_rows)):
        eligible = ref.copy()
    ev = pd.to_numeric(eligible.get("ret_net_notional"), errors="coerce")
    valid_ev = ev.notna()
    if int(valid_ev.sum()) < max(1, int(min_effective_rows)):
        return np.zeros(n_comp, dtype=np.float64), {"baseline_ev": np.nan, "baseline_hit": np.nan}
    baseline_ev = float(ev.loc[valid_ev].mean())
    hit = ev.gt(0.0).astype(float)
    baseline_hit = float(hit.loc[valid_ev].mean())
    weights = _soft_posterior_matrix(
        eligible,
        posterior_cols,
        power=float(power),
        threshold=threshold,
    )
    quality = np.zeros(n_comp, dtype=np.float64)
    # Scale alpha to a dimensionless score.  75 bps is deliberately conservative:
    # it allows recent EV separation to move rank/size, without overwhelming the
    # hard admission layer.
    alpha_scale = 0.0075
    for idx in range(n_comp):
        w = weights[:, idx]
        valid = valid_ev.to_numpy() & np.isfinite(w) & (w > 0.0)
        eff_n = _effective_n(w[valid])
        if eff_n < max(4.0, float(min_effective_rows) / 2.0):
            continue
        comp_ev = _weighted_mean(ev.loc[valid], w[valid])
        comp_hit = _weighted_mean(hit.loc[valid], w[valid])
        if not np.isfinite(comp_ev):
            continue
        support = eff_n / (eff_n + 80.0)
        alpha_score = (comp_ev - baseline_ev) / alpha_scale
        hit_score = 0.50 * ((comp_hit if np.isfinite(comp_hit) else baseline_hit) - baseline_hit)
        quality[idx] = float(np.clip(support * (alpha_score + hit_score), -1.0, 1.0))
    return quality, {"baseline_ev": baseline_ev, "baseline_hit": baseline_hit}


def _apply_gmm_rank_size_overlay(
    rows: pd.DataFrame,
    quality_by_component: np.ndarray,
    posterior_cols: list[str],
    *,
    power: float,
    threshold: float | None,
    strength: float,
) -> pd.DataFrame:
    if rows.empty or len(posterior_cols) == 0 or len(quality_by_component) == 0:
        return rows
    out = rows.copy()
    weights = _soft_posterior_matrix(
        out,
        posterior_cols,
        power=float(power),
        threshold=threshold,
    )
    norm = _normalized_row_weights(weights)
    quality = norm @ np.asarray(quality_by_component, dtype=np.float64)
    has_weight = np.sum(weights, axis=1) > 1e-12
    quality = np.where(has_weight & np.isfinite(quality), quality, 0.0)
    strength_f = float(np.clip(float(strength), 0.0, 2.0))
    rank_adjust = np.clip(0.06 * strength_f * quality, -0.06, 0.06)
    multiplier = np.clip(1.0 + 0.35 * strength_f * quality, 0.70, 1.30)
    if "portfolio_rank_adjustment" in out.columns:
        base_rank_adj = pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce").fillna(0.0)
    else:
        base_rank_adj = pd.Series(0.0, index=out.index, dtype="float64")
    if "portfolio_priority_multiplier" in out.columns:
        base_priority = pd.to_numeric(out["portfolio_priority_multiplier"], errors="coerce").fillna(1.0)
    else:
        base_priority = pd.Series(1.0, index=out.index, dtype="float64")
    if "portfolio_size_multiplier" in out.columns:
        base_size = pd.to_numeric(out["portfolio_size_multiplier"], errors="coerce").fillna(1.0)
    else:
        base_size = pd.Series(1.0, index=out.index, dtype="float64")
    out["portfolio_rank_adjustment"] = (base_rank_adj + rank_adjust).clip(-0.20, 0.20).astype(float)
    out["portfolio_priority_multiplier"] = (base_priority * multiplier).clip(0.50, 1.50).astype(float)
    out["portfolio_size_multiplier"] = (base_size * multiplier).clip(0.50, 1.50).astype(float)
    out["gmm_overlay_quality"] = quality.astype(float)
    out["gmm_overlay_rank_adjustment"] = rank_adjust.astype(float)
    out["gmm_overlay_multiplier"] = multiplier.astype(float)
    return out


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    out[finite] = 0.5 * (1.0 + np.fromiter((math.erf(float(v) / math.sqrt(2.0)) for v in vals[finite]), dtype=np.float64))
    return np.clip(out, 0.0, 1.0)


def _ewma_rank_against_ref(
    day_rows: pd.DataFrame,
    ref_rows: pd.DataFrame,
    *,
    score_col: str,
    day: pd.Timestamp,
    half_life_days: int,
) -> pd.Series:
    score = pd.to_numeric(day_rows[score_col], errors="coerce")
    ref_score = pd.to_numeric(ref_rows[score_col], errors="coerce")
    ref_ts = pd.to_datetime(ref_rows["timestamp"], utc=True, errors="coerce")
    valid = ref_score.notna() & ref_ts.notna()
    if int(valid.sum()) == 0:
        return pd.Series(np.nan, index=day_rows.index, dtype="float64")
    age_days = (pd.Timestamp(day) - ref_ts.loc[valid]).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    half_life = max(float(half_life_days), 1e-6)
    weights = np.power(0.5, np.maximum(age_days, 0.0) / half_life)
    values = ref_score.loc[valid].to_numpy(dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return pd.Series(np.nan, index=day_rows.index, dtype="float64")
    mean = float(np.sum(weights * values) / weight_sum)
    var = float(np.sum(weights * np.square(values - mean)) / weight_sum)
    std = math.sqrt(max(var, 1e-12))
    z = (score.to_numpy(dtype=np.float64) - mean) / std
    return pd.Series(_normal_cdf(z), index=day_rows.index).clip(0.0, 1.0)


def _rank_signal_with_growing_penalty(
    day_rows: pd.DataFrame,
    recent_ref: pd.DataFrame,
    all_prior: pd.DataFrame,
    *,
    score_col: str,
    day: pd.Timestamp,
    window_days: int,
    min_reference_rows: int,
    method: str,
) -> tuple[pd.Series, dict[str, float]]:
    recent_n = int(len(recent_ref))
    if recent_n >= int(min_reference_rows):
        ref = recent_ref
        reference_mode = "window"
    else:
        ref = all_prior
        reference_mode = "growing_penalized"
    if len(ref) < max(1, int(min_reference_rows)):
        empty = pd.Series(np.nan, index=day_rows.index, dtype="float64")
        return empty, {"recent_reference_rows": float(recent_n), "reference_rows": float(len(ref)), "support_penalty": 0.0}
    if method == "ewma":
        rank = _ewma_rank_against_ref(
            day_rows,
            ref,
            score_col=score_col,
            day=day,
            half_life_days=int(window_days),
        )
    else:
        rank = _rank_against_ref(day_rows[score_col], ref[score_col])
    support_penalty = min(1.0, recent_n / max(float(min_reference_rows), 1.0))
    if recent_n < int(min_reference_rows):
        # Sparse recent windows are allowed via growing history, but their
        # incremental timing signal is shrunk toward neutral.
        rank = 0.5 + support_penalty * (pd.to_numeric(rank, errors="coerce") - 0.5)
    return rank.clip(0.0, 1.0), {
        "recent_reference_rows": float(recent_n),
        "reference_rows": float(len(ref)),
        "support_penalty": float(support_penalty),
        "reference_mode": reference_mode,
    }


def _current_rank_for_blend(rows: pd.DataFrame, *, calibration_timing: str) -> pd.Series:
    if calibration_timing == "after" and "rank_pct_raw" in rows.columns:
        return pd.to_numeric(rows["rank_pct_raw"], errors="coerce")
    return pd.to_numeric(rows.get("rank_pct"), errors="coerce")


def _apply_regime_penalty_to_blended_rank(rows: pd.DataFrame, blended: pd.Series) -> pd.Series:
    risk_col = "regime_ev_risk_score"
    if risk_col not in rows.columns:
        return pd.to_numeric(blended, errors="coerce").clip(0.0, 1.0)
    risk = pd.to_numeric(rows[risk_col], errors="coerce").fillna(0.0)
    return (pd.to_numeric(blended, errors="coerce") - risk).clip(0.0, 1.0)


def _blend_current_and_global_rank(
    day_rows: pd.DataFrame,
    global_rank: pd.Series,
    *,
    blend_weight: float,
    calibration_timing: str,
) -> pd.Series:
    weight = float(np.clip(float(blend_weight), 0.0, 1.0))
    current = _current_rank_for_blend(day_rows, calibration_timing=calibration_timing)
    zrank = pd.to_numeric(global_rank.reindex(day_rows.index), errors="coerce")
    blended = ((1.0 - weight) * current + weight * zrank.fillna(current)).clip(0.0, 1.0)
    if calibration_timing == "after":
        blended = _apply_regime_penalty_to_blended_rank(day_rows, blended)
    return blended.clip(0.0, 1.0)


def _threshold_for_target_ev(ref: pd.DataFrame, *, score_col: str, target_ev: float, min_rows: int) -> float:
    score = pd.to_numeric(ref[score_col], errors="coerce")
    ev = pd.to_numeric(ref["ret_net_notional"], errors="coerce")
    valid = ref.loc[score.notna() & ev.notna(), [score_col, "ret_net_notional"]].copy()
    if len(valid) < int(min_rows):
        return float("nan")
    values = pd.to_numeric(valid[score_col], errors="coerce")
    grid = np.unique(np.nanquantile(values.to_numpy(dtype=float), np.linspace(0.70, 0.99, 60)))
    best_threshold = float("nan")
    best_gap = float("inf")
    for threshold in grid:
        chosen = valid.loc[pd.to_numeric(valid[score_col], errors="coerce").ge(float(threshold))]
        if len(chosen) < int(min_rows):
            continue
        mean_ev = float(pd.to_numeric(chosen["ret_net_notional"], errors="coerce").mean())
        gap = abs(mean_ev - float(target_ev))
        if mean_ev >= float(target_ev) and gap < best_gap:
            best_gap = gap
            best_threshold = float(threshold)
    if not np.isfinite(best_threshold):
        best_threshold = float(np.nanquantile(values.to_numpy(dtype=float), 0.99))
    return best_threshold


def _historical_top10_ev(ref: pd.DataFrame, *, score_col: str, min_rows: int) -> float:
    score = pd.to_numeric(ref[score_col], errors="coerce")
    ev = pd.to_numeric(ref["ret_net_notional"], errors="coerce")
    valid = ref.loc[score.notna() & ev.notna(), [score_col, "ret_net_notional"]]
    if len(valid) < int(min_rows):
        return float("nan")
    threshold = float(pd.to_numeric(valid[score_col], errors="coerce").quantile(0.90))
    chosen = valid.loc[pd.to_numeric(valid[score_col], errors="coerce").ge(threshold)]
    if len(chosen) < int(min_rows):
        return float("nan")
    return float(pd.to_numeric(chosen["ret_net_notional"], errors="coerce").mean())


def _historical_current_top10_ev(ref: pd.DataFrame, *, min_rows: int) -> float:
    rank = pd.to_numeric(ref.get("rank_pct"), errors="coerce")
    ev = pd.to_numeric(ref.get("ret_net_notional"), errors="coerce")
    valid = ref.loc[rank.notna() & ev.notna()].copy()
    if len(valid) < int(min_rows):
        return float("nan")
    chosen = valid.loc[pd.to_numeric(valid["rank_pct"], errors="coerce").ge(0.90)]
    if len(chosen) < max(1, int(min_rows) // 2):
        return float("nan")
    return float(pd.to_numeric(chosen["ret_net_notional"], errors="coerce").mean())


def _baseline_activity_count(day_rows: pd.DataFrame) -> int:
    return int(pd.to_numeric(day_rows.get("rank_pct"), errors="coerce").ge(0.90).sum())


def _top_n_by_score(rows: pd.DataFrame, score: pd.Series, n: int) -> pd.DataFrame:
    if rows.empty or int(n) <= 0:
        out = rows.iloc[0:0].copy()
        out["selection_score"] = pd.Series(dtype="float64")
        return out
    work = rows.copy()
    work["_activity_score"] = pd.to_numeric(score.reindex(work.index), errors="coerce")
    work = work.loc[work["_activity_score"].notna()].copy()
    if work.empty:
        out = rows.iloc[0:0].copy()
        out["selection_score"] = pd.Series(dtype="float64")
        return out
    work = work.sort_values(["_activity_score", "timestamp", "symbol"], ascending=[False, True, True]).head(int(n))
    work["selection_score"] = pd.to_numeric(work["_activity_score"], errors="coerce")
    return work.drop(columns=["_activity_score"])


def _rescale_selection_score_to_top_band(rows: pd.DataFrame, *, floor: float = 0.90) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    score = pd.to_numeric(out.get("selection_score"), errors="coerce")
    if score.notna().sum() == 0:
        out["selection_score"] = float(floor)
        return out
    rank = score.rank(method="first", pct=True)
    out["selection_score"] = float(floor) + (1.0 - float(floor)) * rank
    return out


def _match_baseline_activity(
    day_rows: pd.DataFrame,
    chosen: pd.DataFrame,
    *,
    score: pd.Series,
    target_count: int,
) -> pd.DataFrame:
    target = int(target_count)
    if target <= 0:
        out = day_rows.iloc[0:0].copy()
        out["selection_score"] = pd.Series(dtype="float64")
        return out
    if len(chosen) >= target:
        return _rescale_selection_score_to_top_band(_top_n_by_score(chosen, score, target))
    return _rescale_selection_score_to_top_band(_top_n_by_score(day_rows, score, target))


def _hard_archetype_reachable_series(
    day_rows: pd.DataFrame,
    recent_ref: pd.DataFrame,
    all_prior: pd.DataFrame,
    *,
    score_col: str,
    min_reference_rows: int,
    global_target: float,
    global_threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-row hard policy_archetype threshold/target/rank used by the 8d baseline."""
    threshold = pd.Series(float(global_threshold), index=day_rows.index, dtype="float64")
    target = pd.Series(float(global_target), index=day_rows.index, dtype="float64")
    rank = pd.Series(np.nan, index=day_rows.index, dtype="float64")
    arch_min_rows = max(8, int(min_reference_rows) // 4)
    for archetype, sub in day_rows.groupby("policy_archetype", dropna=False):
        archetype_key = str(archetype)
        ref_arch = recent_ref.loc[recent_ref["policy_archetype"].astype(str).eq(archetype_key)]
        prior_arch = all_prior.loc[all_prior["policy_archetype"].astype(str).eq(archetype_key)]
        local_target = _historical_current_top10_ev(prior_arch, min_rows=arch_min_rows)
        local_threshold = _threshold_for_target_ev(
            ref_arch,
            score_col=score_col,
            target_ev=local_target,
            min_rows=arch_min_rows,
        )
        if not np.isfinite(local_threshold):
            local_threshold = global_threshold
            local_target = global_target
        ref_score = ref_arch[score_col] if len(ref_arch) >= arch_min_rows else recent_ref[score_col]
        threshold.loc[sub.index] = float(local_threshold)
        target.loc[sub.index] = float(local_target)
        rank.loc[sub.index] = _rank_against_ref(sub[score_col], ref_score)
    return threshold, target, rank


def _select_arm_rows(
    simulated: pd.DataFrame,
    *,
    arm: Arm,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    score_col: str,
    min_reference_rows: int,
    live_compatible_selection: bool = False,
) -> pd.DataFrame:
    work = _add_time_columns(simulated)
    eval_mask = work["timestamp"].ge(eval_start) & work["timestamp"].lt(eval_end)
    raw_score_col = _raw_score_col(work)
    if arm.family == "current_policy":
        selected = work.loc[eval_mask & pd.to_numeric(work["rank_pct"], errors="coerce").ge(0.90)].copy()
        frames: list[pd.DataFrame] = []
        group_col = "timestamp" if live_compatible_selection else "eval_day"
        for period, period_rows in selected.groupby(group_col, sort=True):
            ref = selected.loc[work.loc[selected.index, "timestamp"].lt(period)].copy()
            if ref.empty:
                frames.append(period_rows.copy())
                continue
            surprise = _weighted_surprise(
                ref,
                holdout_start=pd.Timestamp(period),
                half_life_days=float(PROMOTED_HALF_LIFE_DAYS),
                base_threshold=TOP_THRESHOLDS[PROMOTED_TOP_SLICE],
                min_effective_n=20.0,
            )
            quality = _surprise_quality_map(
                surprise,
                alpha=float(PROMOTED_ALPHA),
                max_adjust=float(PROMOTED_MAX_ADJUST),
            )
            frames.append(
                _apply_portfolio_hr_adjustments(
                    period_rows,
                    mode=PROMOTED_HIT_SURPRISE_MODE,
                    quality_by_archetype=quality,
                    max_adjust=float(PROMOTED_MAX_ADJUST),
                )
            )
        out = pd.concat(frames, ignore_index=True) if frames else selected.iloc[0:0].copy()
        out["selection_score"] = pd.to_numeric(out["rank_pct"], errors="coerce")
    else:
        frames = []
        group_col = "timestamp" if live_compatible_selection else "eval_day"
        for period, day_rows in work.loc[eval_mask].groupby(group_col, sort=True):
            period_ts = pd.Timestamp(period)
            ref_start = period_ts - pd.Timedelta(days=int(arm.window_days or 0))
            recent_ref = work.loc[work["timestamp"].ge(ref_start) & work["timestamp"].lt(period_ts)].copy()
            all_prior = work.loc[work["timestamp"].lt(period_ts)].copy()
            requires_window_support = arm.family not in {"ev_target_archetype_reachable_blend_matched_activity"}
            if requires_window_support and len(recent_ref) < int(min_reference_rows):
                continue
            day_rows = day_rows.copy()
            baseline_count = _baseline_activity_count(day_rows)
            if arm.family == "zscore_global":
                rank = _rank_against_ref(day_rows[score_col], recent_ref[score_col])
                day_rows["selection_score"] = rank
                chosen = day_rows.loc[rank.ge(0.90)].copy()
            elif arm.family == "zscore_global_matched_activity":
                rank = _rank_against_ref(day_rows[score_col], recent_ref[score_col])
                chosen = _match_baseline_activity(
                    day_rows,
                    day_rows.loc[rank.ge(0.90)].copy(),
                    score=rank,
                    target_count=baseline_count,
                )
            elif arm.family == "zscore_archetype_weighted":
                global_rank = _rank_against_ref(day_rows[score_col], recent_ref[score_col])
                arch_rank = pd.Series(np.nan, index=day_rows.index, dtype="float64")
                for archetype, sub in day_rows.groupby("policy_archetype", dropna=False):
                    ref_arch = recent_ref.loc[recent_ref["policy_archetype"].astype(str).eq(str(archetype))]
                    if len(ref_arch) >= int(min_reference_rows):
                        arch_rank.loc[sub.index] = _rank_against_ref(sub[score_col], ref_arch[score_col])
                blended = 0.5 * global_rank + 0.5 * arch_rank.fillna(global_rank)
                day_rows["selection_score"] = blended
                chosen = day_rows.loc[blended.ge(0.90)].copy()
            elif arm.family == "ev_target_global":
                target_ev = _historical_top10_ev(all_prior, score_col=score_col, min_rows=min_reference_rows)
                threshold = _threshold_for_target_ev(
                    recent_ref,
                    score_col=score_col,
                    target_ev=target_ev,
                    min_rows=min_reference_rows,
                )
                chosen = day_rows.loc[pd.to_numeric(day_rows[score_col], errors="coerce").ge(threshold)].copy()
                chosen["selection_score"] = _rank_against_ref(chosen[score_col], recent_ref[score_col])
                chosen["dynamic_ev_target"] = target_ev
                chosen["dynamic_score_threshold"] = threshold
            elif arm.family == "ev_target_global_matched_activity":
                target_ev = _historical_top10_ev(all_prior, score_col=score_col, min_rows=min_reference_rows)
                threshold = _threshold_for_target_ev(
                    recent_ref,
                    score_col=score_col,
                    target_ev=target_ev,
                    min_rows=min_reference_rows,
                )
                rank = _rank_against_ref(day_rows[score_col], recent_ref[score_col])
                threshold_chosen = day_rows.loc[pd.to_numeric(day_rows[score_col], errors="coerce").ge(threshold)].copy()
                chosen = _match_baseline_activity(
                    day_rows,
                    threshold_chosen,
                    score=rank,
                    target_count=baseline_count,
                )
                chosen["dynamic_ev_target"] = target_ev
                chosen["dynamic_score_threshold"] = threshold
            elif arm.family == "ev_target_archetype":
                chosen_parts: list[pd.DataFrame] = []
                global_target = _historical_top10_ev(all_prior, score_col=score_col, min_rows=min_reference_rows)
                global_threshold = _threshold_for_target_ev(
                    recent_ref,
                    score_col=score_col,
                    target_ev=global_target,
                    min_rows=min_reference_rows,
                )
                for archetype, sub in day_rows.groupby("policy_archetype", dropna=False):
                    ref_arch = recent_ref.loc[recent_ref["policy_archetype"].astype(str).eq(str(archetype))]
                    prior_arch = all_prior.loc[all_prior["policy_archetype"].astype(str).eq(str(archetype))]
                    target = _historical_top10_ev(prior_arch, score_col=score_col, min_rows=min_reference_rows)
                    threshold = _threshold_for_target_ev(
                        ref_arch,
                        score_col=score_col,
                        target_ev=target,
                        min_rows=min_reference_rows,
                    )
                    if not np.isfinite(threshold):
                        threshold = global_threshold
                        target = global_target
                    part = sub.loc[pd.to_numeric(sub[score_col], errors="coerce").ge(threshold)].copy()
                    part["selection_score"] = _rank_against_ref(part[score_col], ref_arch[score_col] if len(ref_arch) else recent_ref[score_col])
                    part["dynamic_ev_target"] = target
                    part["dynamic_score_threshold"] = threshold
                    chosen_parts.append(part)
                chosen = pd.concat(chosen_parts, ignore_index=False) if chosen_parts else day_rows.iloc[0:0].copy()
            elif arm.family in {"ev_target_archetype_reachable", "ev_target_archetype_reachable_matched_activity"}:
                chosen_parts = []
                score_rank = pd.Series(np.nan, index=day_rows.index, dtype="float64")
                global_target = _historical_current_top10_ev(all_prior, min_rows=min_reference_rows)
                if not np.isfinite(global_target):
                    global_target = _historical_top10_ev(all_prior, score_col=score_col, min_rows=min_reference_rows)
                global_threshold = _threshold_for_target_ev(
                    recent_ref,
                    score_col=score_col,
                    target_ev=global_target,
                    min_rows=min_reference_rows,
                )
                arch_min_rows = max(8, int(min_reference_rows) // 4)
                for archetype, sub in day_rows.groupby("policy_archetype", dropna=False):
                    archetype_key = str(archetype)
                    ref_arch = recent_ref.loc[recent_ref["policy_archetype"].astype(str).eq(archetype_key)]
                    prior_arch = all_prior.loc[all_prior["policy_archetype"].astype(str).eq(archetype_key)]
                    target = _historical_current_top10_ev(prior_arch, min_rows=arch_min_rows)
                    threshold = _threshold_for_target_ev(
                        ref_arch,
                        score_col=score_col,
                        target_ev=target,
                        min_rows=arch_min_rows,
                    )
                    if not np.isfinite(threshold):
                        threshold = global_threshold
                        target = global_target
                    ref_score = ref_arch[score_col] if len(ref_arch) >= arch_min_rows else recent_ref[score_col]
                    local_rank = _rank_against_ref(sub[score_col], ref_score)
                    score_rank.loc[sub.index] = local_rank
                    part = sub.loc[pd.to_numeric(sub[score_col], errors="coerce").ge(threshold)].copy()
                    part["selection_score"] = local_rank.reindex(part.index)
                    part["dynamic_ev_target"] = target
                    part["dynamic_score_threshold"] = threshold
                    chosen_parts.append(part)
                chosen = pd.concat(chosen_parts, ignore_index=False) if chosen_parts else day_rows.iloc[0:0].copy()
                if arm.family == "ev_target_archetype_reachable_matched_activity":
                    chosen = _match_baseline_activity(
                        day_rows,
                        chosen,
                        score=score_rank.fillna(_rank_against_ref(day_rows[score_col], recent_ref[score_col])),
                        target_count=baseline_count,
                    )
                    if bool(arm.hr_rank50) and frames:
                        history = pd.concat(frames, ignore_index=True)
                        history_ts = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
                        history = history.loc[history_ts.lt(period_ts)].copy()
                        if not history.empty:
                            surprise = _weighted_surprise(
                                history,
                                holdout_start=period_ts,
                                half_life_days=float(PROMOTED_HALF_LIFE_DAYS),
                                base_threshold=TOP_THRESHOLDS[PROMOTED_TOP_SLICE],
                                min_effective_n=20.0,
                            )
                            quality = _surprise_quality_map(
                                surprise,
                                alpha=float(PROMOTED_ALPHA),
                                max_adjust=float(PROMOTED_MAX_ADJUST),
                            )
                            chosen = _apply_portfolio_hr_adjustments(
                                chosen,
                                mode=PROMOTED_HIT_SURPRISE_MODE,
                                quality_by_archetype=quality,
                                max_adjust=float(PROMOTED_MAX_ADJUST),
                            )
            elif arm.family == "ev_target_archetype_reachable_gmm_overlay_matched_activity":
                posterior_cols = _posterior_cols(work)
                if not posterior_cols:
                    raise ValueError(
                        f"{arm.name} requires gmm_cluster_posterior_* or gmm_prob_* columns. "
                        "Materialize frozen AE/GMM posteriors before running this arm."
                    )
                global_target = _historical_current_top10_ev(all_prior, min_rows=min_reference_rows)
                if not np.isfinite(global_target):
                    global_target = _historical_top10_ev(all_prior, score_col=score_col, min_rows=min_reference_rows)
                global_threshold = _threshold_for_target_ev(
                    recent_ref,
                    score_col=score_col,
                    target_ev=global_target,
                    min_rows=min_reference_rows,
                )
                if not np.isfinite(global_threshold):
                    global_threshold = float(pd.to_numeric(recent_ref[score_col], errors="coerce").quantile(0.90))
                hard_threshold, hard_target, hard_rank = _hard_archetype_reachable_series(
                    day_rows,
                    recent_ref,
                    all_prior,
                    score_col=score_col,
                    min_reference_rows=int(min_reference_rows),
                    global_target=float(global_target),
                    global_threshold=float(global_threshold),
                )
                enriched_day = day_rows.copy()
                score_rank = pd.to_numeric(hard_rank, errors="coerce").fillna(
                    _rank_against_ref(day_rows[score_col], recent_ref[score_col])
                ).clip(0.0, 1.0)
                enriched_day["selection_score"] = score_rank
                enriched_day["dynamic_ev_target"] = hard_target
                enriched_day["dynamic_score_threshold"] = hard_threshold
                threshold_chosen = enriched_day.loc[
                    pd.to_numeric(enriched_day[score_col], errors="coerce").to_numpy(dtype=np.float64)
                    >= hard_threshold.to_numpy(dtype=np.float64, copy=False)
                ].copy()
                chosen = _match_baseline_activity(
                    enriched_day,
                    threshold_chosen,
                    score=score_rank,
                    target_count=baseline_count,
                )
                quality_by_component, quality_meta = _gmm_component_recent_quality(
                    recent_ref,
                    posterior_cols,
                    power=float(arm.posterior_power),
                    threshold=arm.posterior_threshold,
                    min_effective_rows=max(8, int(min_reference_rows) // 4),
                )
                chosen = _apply_gmm_rank_size_overlay(
                    chosen,
                    quality_by_component,
                    posterior_cols,
                    power=float(arm.posterior_power),
                    threshold=arm.posterior_threshold,
                    strength=float(arm.blend_weight),
                )
                chosen["gmm_overlay_strength"] = float(arm.blend_weight)
                chosen["posterior_power"] = float(arm.posterior_power)
                chosen["posterior_threshold"] = (
                    np.nan if arm.posterior_threshold is None else float(arm.posterior_threshold)
                )
                chosen["gmm_recent_baseline_ev"] = float(quality_meta.get("baseline_ev", np.nan))
                chosen["gmm_recent_baseline_hit"] = float(quality_meta.get("baseline_hit", np.nan))
            elif arm.family in {
                "ev_target_gmm_posterior_reachable_matched_activity",
                "ev_target_hard_gmm_posterior_blend_matched_activity",
            }:
                posterior_cols = _posterior_cols(work)
                if not posterior_cols:
                    raise ValueError(
                        f"{arm.name} requires gmm_cluster_posterior_* or gmm_prob_* columns. "
                        "Materialize frozen AE/GMM posteriors before running this arm."
                    )
                global_target = _historical_current_top10_ev(all_prior, min_rows=min_reference_rows)
                if not np.isfinite(global_target):
                    global_target = _historical_top10_ev(all_prior, score_col=score_col, min_rows=min_reference_rows)
                global_threshold = _threshold_for_target_ev(
                    recent_ref,
                    score_col=score_col,
                    target_ev=global_target,
                    min_rows=min_reference_rows,
                )
                if not np.isfinite(global_threshold):
                    global_threshold = float(pd.to_numeric(recent_ref[score_col], errors="coerce").quantile(0.90))
                comp_min_rows = max(8, int(min_reference_rows) // 4)
                day_w_raw = _soft_posterior_matrix(
                    day_rows,
                    posterior_cols,
                    power=float(arm.posterior_power),
                    threshold=arm.posterior_threshold,
                )
                day_w_norm = _normalized_row_weights(day_w_raw)
                recent_w = _soft_posterior_matrix(
                    recent_ref,
                    posterior_cols,
                    power=float(arm.posterior_power),
                    threshold=arm.posterior_threshold,
                )
                prior_w = _soft_posterior_matrix(
                    all_prior,
                    posterior_cols,
                    power=float(arm.posterior_power),
                    threshold=arm.posterior_threshold,
                )
                comp_thresholds = np.full(day_w_norm.shape[1], global_threshold, dtype=np.float64)
                comp_targets = np.full(day_w_norm.shape[1], global_target, dtype=np.float64)
                comp_ranks = np.full((len(day_rows), day_w_norm.shape[1]), np.nan, dtype=np.float64)
                for comp_idx in range(day_w_norm.shape[1]):
                    target = _weighted_historical_current_top10_ev(
                        all_prior,
                        prior_w[:, comp_idx],
                        min_effective_rows=comp_min_rows,
                    )
                    threshold = _weighted_threshold_for_target_ev(
                        recent_ref,
                        recent_w[:, comp_idx],
                        score_col=score_col,
                        target_ev=target,
                        min_effective_rows=comp_min_rows,
                    )
                    if np.isfinite(threshold):
                        comp_thresholds[comp_idx] = float(threshold)
                    if np.isfinite(target):
                        comp_targets[comp_idx] = float(target)
                    rank_i = _weighted_rank_against_ref(
                        day_rows[score_col],
                        recent_ref[score_col],
                        recent_w[:, comp_idx],
                        min_effective_rows=comp_min_rows,
                    )
                    comp_ranks[:, comp_idx] = rank_i.to_numpy(dtype=np.float64, copy=False)
                global_rank = _rank_against_ref(day_rows[score_col], recent_ref[score_col])
                row_threshold = day_w_norm @ comp_thresholds
                row_target = day_w_norm @ comp_targets
                has_row_weight = np.sum(day_w_raw, axis=1) > 1e-12
                row_threshold = np.where(has_row_weight, row_threshold, global_threshold)
                row_target = np.where(has_row_weight, row_target, global_target)
                weighted_rank = np.nansum(day_w_norm * comp_ranks, axis=1)
                weighted_rank = np.where(has_row_weight & np.isfinite(weighted_rank), weighted_rank, np.nan)
                score_rank = pd.Series(weighted_rank, index=day_rows.index).fillna(global_rank).clip(0.0, 1.0)
                if arm.family == "ev_target_hard_gmm_posterior_blend_matched_activity":
                    hard_threshold, hard_target, hard_rank = _hard_archetype_reachable_series(
                        day_rows,
                        recent_ref,
                        all_prior,
                        score_col=score_col,
                        min_reference_rows=int(min_reference_rows),
                        global_target=float(global_target),
                        global_threshold=float(global_threshold),
                    )
                    gmm_weight = float(np.clip(float(arm.blend_weight), 0.0, 1.0))
                    row_threshold = (
                        (1.0 - gmm_weight) * hard_threshold.to_numpy(dtype=np.float64, copy=False)
                        + gmm_weight * row_threshold
                    )
                    row_target = (
                        (1.0 - gmm_weight) * hard_target.to_numpy(dtype=np.float64, copy=False)
                        + gmm_weight * row_target
                    )
                    hard_rank = pd.to_numeric(hard_rank, errors="coerce").fillna(global_rank).clip(0.0, 1.0)
                    score_rank = ((1.0 - gmm_weight) * hard_rank + gmm_weight * score_rank).clip(0.0, 1.0)
                enriched_day = day_rows.copy()
                enriched_day["selection_score"] = score_rank
                enriched_day["dynamic_ev_target"] = row_target
                enriched_day["dynamic_score_threshold"] = row_threshold
                enriched_day["gmm_blend_weight"] = (
                    1.0 if arm.family == "ev_target_gmm_posterior_reachable_matched_activity" else float(arm.blend_weight)
                )
                enriched_day["posterior_power"] = float(arm.posterior_power)
                enriched_day["posterior_threshold"] = (
                    np.nan if arm.posterior_threshold is None else float(arm.posterior_threshold)
                )
                enriched_day["posterior_effective_components"] = np.sum(day_w_raw > 0.0, axis=1).astype(float)
                threshold_chosen = enriched_day.loc[
                    pd.to_numeric(enriched_day[score_col], errors="coerce").to_numpy(dtype=np.float64)
                    >= row_threshold
                ].copy()
                chosen = _match_baseline_activity(
                    enriched_day,
                    threshold_chosen,
                    score=score_rank,
                    target_count=baseline_count,
                )
            elif arm.family == "ev_target_archetype_reachable_blend_matched_activity":
                chosen_parts: list[pd.DataFrame] = []
                target_score_col = score_col if arm.calibration_timing == "before" else raw_score_col
                global_target = _historical_current_top10_ev(all_prior, min_rows=min_reference_rows)
                if not np.isfinite(global_target):
                    global_target = _historical_top10_ev(
                        all_prior,
                        score_col=target_score_col,
                        min_rows=min_reference_rows,
                    )
                ref_for_global_threshold = recent_ref if len(recent_ref) >= int(min_reference_rows) else all_prior
                global_threshold = _threshold_for_target_ev(
                    ref_for_global_threshold,
                    score_col=target_score_col,
                    target_ev=global_target,
                    min_rows=min_reference_rows,
                )
                z_rank, ref_meta = _rank_signal_with_growing_penalty(
                    day_rows,
                    recent_ref,
                    all_prior,
                    score_col=target_score_col,
                    day=period_ts,
                    window_days=int(arm.window_days or 5),
                    min_reference_rows=int(min_reference_rows),
                    method=str(arm.zscore_method),
                )
                blended_rank = _blend_current_and_global_rank(
                    day_rows,
                    z_rank,
                    blend_weight=float(arm.blend_weight),
                    calibration_timing=str(arm.calibration_timing),
                )
                arch_min_rows = max(8, int(min_reference_rows) // 4)
                ref_for_arch_threshold = recent_ref if len(recent_ref) >= arch_min_rows else all_prior
                for archetype, sub in day_rows.groupby("policy_archetype", dropna=False):
                    archetype_key = str(archetype)
                    ref_arch = ref_for_arch_threshold.loc[
                        ref_for_arch_threshold["policy_archetype"].astype(str).eq(archetype_key)
                    ]
                    prior_arch = all_prior.loc[all_prior["policy_archetype"].astype(str).eq(archetype_key)]
                    target = _historical_current_top10_ev(prior_arch, min_rows=arch_min_rows)
                    threshold = _threshold_for_target_ev(
                        ref_arch,
                        score_col=target_score_col,
                        target_ev=target,
                        min_rows=arch_min_rows,
                    )
                    if not np.isfinite(threshold):
                        threshold = global_threshold
                        target = global_target
                    part = sub.loc[pd.to_numeric(sub[target_score_col], errors="coerce").ge(threshold)].copy()
                    part["selection_score"] = blended_rank.reindex(part.index)
                    part["dynamic_ev_target"] = target
                    part["dynamic_score_threshold"] = threshold
                    chosen_parts.append(part)
                threshold_chosen = (
                    pd.concat(chosen_parts, ignore_index=False)
                    if chosen_parts
                    else day_rows.iloc[0:0].copy()
                )
                chosen = _match_baseline_activity(
                    day_rows,
                    threshold_chosen,
                    score=blended_rank,
                    target_count=baseline_count,
                )
                chosen["zscore_method"] = str(arm.zscore_method)
                chosen["blend_weight"] = float(arm.blend_weight)
                chosen["calibration_timing"] = str(arm.calibration_timing)
                chosen["hr_rank50_enabled"] = bool(arm.hr_rank50)
                chosen["recent_reference_rows"] = float(ref_meta.get("recent_reference_rows", np.nan))
                chosen["reference_rows"] = float(ref_meta.get("reference_rows", np.nan))
                chosen["support_penalty"] = float(ref_meta.get("support_penalty", np.nan))
                chosen["reference_mode"] = str(ref_meta.get("reference_mode", ""))
                if bool(arm.hr_rank50) and frames:
                    history = pd.concat(frames, ignore_index=True)
                    history_ts = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
                    history = history.loc[history_ts.lt(period_ts)].copy()
                    if not history.empty:
                        surprise = _weighted_surprise(
                            history,
                            holdout_start=period_ts,
                            half_life_days=float(PROMOTED_HALF_LIFE_DAYS),
                            base_threshold=TOP_THRESHOLDS[PROMOTED_TOP_SLICE],
                            min_effective_n=20.0,
                        )
                        quality = _surprise_quality_map(
                            surprise,
                            alpha=float(PROMOTED_ALPHA),
                            max_adjust=float(PROMOTED_MAX_ADJUST),
                        )
                        chosen = _apply_portfolio_hr_adjustments(
                            chosen,
                            mode=PROMOTED_HIT_SURPRISE_MODE,
                            quality_by_archetype=quality,
                            max_adjust=float(PROMOTED_MAX_ADJUST),
                        )
            else:
                raise ValueError(f"Unsupported arm family: {arm.family}")
            frames.append(chosen)
        out = pd.concat(frames, ignore_index=True) if frames else work.iloc[0:0].copy()

    if out.empty:
        return out
    out["rank_pct"] = pd.to_numeric(out.get("selection_score", out.get("rank_pct")), errors="coerce").fillna(
        pd.to_numeric(out.get("rank_pct"), errors="coerce")
    )
    out["base_rank_threshold"] = 0.90
    out["applied_rank_threshold"] = 0.90
    out["mode"] = arm.name
    out["top_slice"] = "top10"
    out["half_life_days"] = np.nan if arm.window_days is None else float(arm.window_days)
    out["alpha"] = np.nan
    out["max_adjust"] = np.nan
    out["threshold_basis_family"] = arm.family
    return out


def _metrics_from_decisions(decisions: pd.DataFrame, metrics: dict[str, Any], *, arm: Arm, source_rows: int) -> dict[str, Any]:
    accepted = decisions.loc[decisions["accepted"]].copy() if "accepted" in decisions.columns else decisions.iloc[0:0].copy()
    return {
        "arm": arm.name,
        "family": arm.family,
        "window_days": np.nan if arm.window_days is None else int(arm.window_days),
        "description": arm.description,
        "zscore_method": arm.zscore_method,
        "blend_weight": float(arm.blend_weight),
        "calibration_timing": arm.calibration_timing,
        "hr_rank50": bool(arm.hr_rank50),
        "source_selected_rows": int(source_rows),
        "accepted_rows": int(len(accepted)),
        "trade_count": int(metrics.get("trade_count", 0) or 0),
        "trades_per_day": _safe_float(metrics.get("trades_per_day")),
        "net_pnl": _safe_float(metrics.get("net_pnl")),
        "gross_pnl": _safe_float(metrics.get("gross_pnl")),
        "compounded_return": _safe_float(metrics.get("compounded_return")),
        "max_drawdown": _safe_float(metrics.get("max_drawdown")),
        "worst_week": _safe_float(metrics.get("worst_week")),
        "notional_weighted_net_return": _safe_float(metrics.get("notional_weighted_net_return")),
        "mean_net_return_per_trade": _safe_float(metrics.get("mean_net_return_per_trade")),
        "full_sl_rate": _safe_float(metrics.get("full_sl_rate")),
        "timeout_rate": _safe_float(metrics.get("timeout_rate")),
        "avg_open_positions": _safe_float(metrics.get("avg_open_positions")),
        "position_utilization": _safe_float(metrics.get("position_utilization")),
    }


def _breakdown(decisions: pd.DataFrame, *, arm: Arm, keys: list[str]) -> pd.DataFrame:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"]].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["month"] = accepted["timestamp"].dt.strftime("%Y-%m")
    accepted["week_start"] = _week_start(accepted["timestamp"]).dt.strftime("%Y-%m-%d")
    if "side_name" not in accepted.columns and "side" in accepted.columns:
        accepted["side_name"] = np.where(pd.to_numeric(accepted["side"], errors="coerce").lt(0.0), "short", "long")
    if "policy_archetype" not in accepted.columns:
        accepted["policy_archetype"] = "missing"
    ret = pd.to_numeric(accepted.get("position_net_return", accepted.get("net_return")), errors="coerce")
    size = pd.to_numeric(accepted.get("position_size", 1.0), errors="coerce").fillna(1.0)
    reason = accepted.get("position_exit_reason", accepted.get("simple_policy_exit_reason", "")).astype(str)
    accepted["_ret"] = ret
    accepted["_pnl"] = ret * size
    accepted["_full_sl"] = reason.eq("full_sl").astype(float)
    accepted["_timeout"] = reason.eq("timeout").astype(float)
    rows = []
    for group_key, group in accepted.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        record = {
            "arm": arm.name,
            "family": arm.family,
            "window_days": np.nan if arm.window_days is None else int(arm.window_days),
            "rows": int(len(group)),
            "mean_net_return_per_trade": float(group["_ret"].mean()),
            "net_pnl": float(group["_pnl"].sum()),
            "full_sl_rate": float(group["_full_sl"].mean()),
            "timeout_rate": float(group["_timeout"].mean()),
        }
        for col, value in zip(keys, group_key, strict=False):
            record[col] = value
        rows.append(record)
    return pd.DataFrame(rows)


def _attach_candidate_context(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or candidates.empty:
        return decisions
    context_cols = [
        col
        for col in (
            "timestamp",
            "symbol",
            "strategy_id",
            "side_name",
            "policy_archetype",
            "local_side_archetype",
            "mode",
            "threshold_basis_family",
        )
        if col in candidates.columns
    ]
    key_cols = [col for col in ("timestamp", "symbol", "strategy_id") if col in decisions.columns and col in context_cols]
    if len(key_cols) < 3:
        return decisions
    ctx = candidates[context_cols].copy()
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True, errors="coerce")
    ctx = ctx.drop_duplicates(key_cols, keep="last")
    out = decisions.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    missing_context = [
        col
        for col in ("side_name", "policy_archetype", "local_side_archetype", "threshold_basis_family")
        if col in ctx.columns and (col not in out.columns or out[col].astype(str).eq("").all())
    ]
    if not missing_context:
        return out
    return out.merge(ctx[key_cols + missing_context], on=key_cols, how="left", validate="many_to_one")


def _arms() -> list[Arm]:
    def power_label(value: float) -> str:
        v = float(value)
        return str(int(v)) if float(v).is_integer() else str(v).replace(".", "")

    arms = [
        Arm(
            "current_policy_top10_hr_rank50_regimecal",
            "current_policy",
            None,
            f"{PROMOTED_POLICY_NAME}; {PROMOTED_HIT_SURPRISE_MODE}; per_regime_archetype_calibration_v1.",
        )
    ]
    for days in (5, 10, 20):
        arms.extend(
            [
                Arm(f"zscore_global_top10_{days}d", "zscore_global", days, "Top 10% by rolling global score percentile/z-score."),
                Arm(
                    f"zscore_global_match_current_activity_{days}d",
                    "zscore_global_matched_activity",
                    days,
                    "Rolling global score percentile/z-score, daily source count matched to current top10 policy.",
                ),
                Arm(
                    f"zscore_archetype_weighted_top10_{days}d",
                    "zscore_archetype_weighted",
                    days,
                    "Top 10% by 50/50 blend of rolling global and archetype score percentiles.",
                ),
                Arm(
                    f"ev_target_global_top10_{days}d",
                    "ev_target_global",
                    days,
                    "Dynamic global score threshold targeting historical top10 EV.",
                ),
                Arm(
                    f"ev_target_global_match_current_activity_{days}d",
                    "ev_target_global_matched_activity",
                    days,
                    "Dynamic global EV target with daily source count matched to current top10 policy.",
                ),
                Arm(
                    f"ev_target_archetype_top10_{days}d",
                    "ev_target_archetype",
                    days,
                    "Dynamic side/archetype score threshold targeting historical top10 EV.",
                ),
            ]
        )
    for days in (3, 5, 7, 8, 9, 10):
        arms.extend(
            [
                Arm(
                    f"ev_target_archetype_reachable_top10_{days}d",
                    "ev_target_archetype_reachable",
                    days,
                    f"Per-archetype threshold targeting the reachable EV of current-policy top10 rows over the prior {days} days.",
                ),
            ]
        )
        for hr_enabled in (False, True):
            arm_name = f"ev_target_archetype_reachable_match_current_activity_{days}d"
            if hr_enabled:
                arm_name = f"{arm_name}_hron"
            arms.append(
                Arm(
                    arm_name,
                    "ev_target_archetype_reachable_matched_activity",
                    days,
                    (
                        f"Reachable per-archetype EV threshold over {days} days with daily source count matched "
                        f"to current top10 policy; HR rank50 {'on' if hr_enabled else 'off'}."
                    ),
                    hr_rank50=hr_enabled,
                )
            )
    for days in (3, 5, 7):
        for method in ("flat", "ewma"):
            for timing in ("before", "after"):
                for hr_enabled in (False, True):
                    arms.append(
                        Arm(
                            (
                                "ev_target_arch_reachable_match_activity_"
                                f"blend50_global_{method}_{days}d_"
                                f"calib_{timing}_hr{'on' if hr_enabled else 'off'}"
                            ),
                            "ev_target_archetype_reachable_blend_matched_activity",
                            days,
                            (
                                "Per-archetype reachable EV threshold; daily activity matched to current top10; "
                                f"50/50 blend of current rank and global {method} rank; "
                                f"regime calibration {timing} blend; HR rank50 {'on' if hr_enabled else 'off'}."
                            ),
                            zscore_method=method,
                            blend_weight=0.5,
                            calibration_timing=timing,
                            hr_rank50=hr_enabled,
                        )
                    )
    for power in (0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 1.0, 2.0, 3.0):
        p_label = power_label(power)
        arms.append(
            Arm(
                f"ev_target_gmm_posterior_reachable_match_current_activity_8d_p{p_label}",
                "ev_target_gmm_posterior_reachable_matched_activity",
                8,
                (
                    "Reachable EV threshold over 8 days using frozen AE/GMM posterior-weighted "
                    f"component thresholds; posterior power={power:g}; daily source count matched "
                    "to current top10 policy; HR rank50 off."
                ),
                blend_weight=1.0,
                posterior_power=float(power),
                posterior_threshold=None,
            )
        )
    for gmm_weight in (0.25, 0.50, 0.75):
        for power in (0.5, 0.75, 1.0):
            p_label = power_label(power)
            g_label = f"{gmm_weight:.2f}".replace(".", "")
            arms.append(
                Arm(
                    f"ev_target_hard_gmm_posterior_blend_match_current_activity_8d_g{g_label}_p{p_label}",
                    "ev_target_hard_gmm_posterior_blend_matched_activity",
                    8,
                    (
                        "Hard policy_archetype reachable EV threshold blended with frozen AE/GMM "
                        f"posterior-weighted threshold over 8 days; GMM weight={gmm_weight:.2f}; "
                        f"posterior power={power:g}; daily source count matched to current top10; HR rank50 off."
                    ),
                    blend_weight=float(gmm_weight),
                    posterior_power=float(power),
                    posterior_threshold=None,
                )
            )
    for strength in (0.50, 1.00, 1.50):
        for power in (0.35, 0.50, 1.0):
            p_label = power_label(power)
            s_label = f"{strength:.2f}".replace(".", "")
            arms.append(
                Arm(
                    f"ev_target_hard8d_gmm_overlay_rank_size_8d_s{s_label}_p{p_label}",
                    "ev_target_archetype_reachable_gmm_overlay_matched_activity",
                    8,
                    (
                        "Hard policy_archetype 8d reachable EV admission; frozen AE/GMM posterior "
                        "recent alpha/hit overlay adjusts portfolio rank, priority, and size only; "
                        f"overlay strength={strength:.2f}; posterior power={power:g}; HR rank50 off."
                    ),
                    blend_weight=float(strength),
                    posterior_power=float(power),
                    posterior_threshold=None,
                )
            )
    for threshold in (0.50, 0.60, 0.70):
        for power in (1.0, 2.0, 3.0):
            label = f"{threshold:.2f}".replace(".", "")
            p_label = power_label(power)
            arms.append(
                Arm(
                    f"ev_target_gmm_posterior_reachable_match_current_activity_8d_t{label}_p{p_label}",
                    "ev_target_gmm_posterior_reachable_matched_activity",
                    8,
                    (
                        "Reachable EV threshold over 8 days using thresholded frozen AE/GMM "
                        f"posterior-weighted component thresholds; posterior >= {threshold:.2f}; "
                        f"power={power:g}; daily source count matched to current top10 policy; HR rank50 off."
                    ),
                    blend_weight=1.0,
                    posterior_power=float(power),
                    posterior_threshold=float(threshold),
                )
            )
    return arms


def _simulate_all_rows(
    rows: pd.DataFrame,
    *,
    parent_summary: Mapping[str, dict[str, Any]],
    data_root: str,
    market_mode: str,
    path_len: int,
    cost_pct: float,
) -> pd.DataFrame:
    bundles = _load_bundles(
        rows,
        data_root=str(data_root),
        market_mode=str(market_mode),
        path_len=int(path_len),
        min_rows_per_strategy=5,
    )
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        strategy_id = str(bundle.strategy_id)
        if strategy_id in parent_summary:
            params, size_power = _params_from_parent_summary_row(parent_summary[strategy_id])
        else:
            params = bundle.base_params
            size_power = bundle.best_size_power
        selected, _adv = _simulate_selected_rows(
            bundle.rows,
            bundle.paths,
            rank_threshold=0.0,
            params=params,
            size_power=float(size_power),
            cost_pct=float(cost_pct),
        )
        if not selected.empty:
            selected["expected_probability"] = _expected_probability(selected)
            frames.append(selected)
    if not frames:
        raise RuntimeError("No simulated rows available.")
    out = pd.concat(frames, ignore_index=True)
    out["candidate_uid"] = out.get(
        "candidate_uid",
        out["timestamp"].astype(str) + "|" + out["symbol"].astype(str) + "|" + out["strategy_id"].astype(str),
    )
    return out


def _verify_policy_contract(report_dir: Path, rows: pd.DataFrame) -> dict[str, Any]:
    manifest_path = report_dir / "manifest.json"
    promoted_path = report_dir / "policy_params" / "hit_surprise_archetype_portfolio_policy.json"
    out: dict[str, Any] = {
        "report_dir": str(report_dir),
        "manifest_exists": manifest_path.exists(),
        "promoted_policy_exists": promoted_path.exists(),
        "expected_policy_name": PROMOTED_POLICY_NAME,
        "expected_mode": PROMOTED_HIT_SURPRISE_MODE,
        "expected_regime_calibration": "per_regime_archetype_calibration_v1",
        "rank_basis_columns_present": {
            col: col in rows.columns
            for col in ("rank_pct", "rank_pct_raw", "rank_pct_regime_ev_unprotected", "rank_score_source")
        },
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        out["manifest_policy_name"] = ((manifest.get("promoted_policy") or {}).get("policy_name"))
        out["manifest_regime_calibration_policy_id"] = ((manifest.get("regime_ev_calibration") or {}).get("policy_id"))
    if promoted_path.exists():
        promoted = json.loads(promoted_path.read_text())
        out["promoted_policy_name"] = promoted.get("policy_name")
        out["promoted_hit_surprise_mode"] = ((promoted.get("selection") or {}).get("hit_surprise_mode"))
        out["promoted_rank_threshold"] = ((promoted.get("selection") or {}).get("base_rank_threshold"))
    if "rank_score_source" in rows.columns:
        out["rank_score_source_counts"] = rows["rank_score_source"].astype(str).value_counts().head(10).to_dict()
    out["policy_contract_pass"] = (
        out.get("manifest_policy_name") == PROMOTED_POLICY_NAME
        and out.get("manifest_regime_calibration_policy_id") == "per_regime_archetype_calibration_v1"
        and bool(out["rank_basis_columns_present"].get("rank_pct"))
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--parent-policy-summary", type=Path, default=DEFAULT_PARENT_POLICY_SUMMARY)
    parser.add_argument("--source-report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR / "threshold_basis_ablation_v1")
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", choices=("spot", "perps"), default="perps")
    parser.add_argument(
        "--exchange",
        default="krakenfutures",
        help="Exchange-scoped replay root to use. Defaults to Kraken Futures; do not leave implicit for live/parity replays.",
    )
    parser.add_argument("--eval-start", default="2026-05-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.01)
    parser.add_argument("--min-reference-rows", type=int, default=40)
    parser.add_argument("--max-window-days", type=int, default=20)
    parser.add_argument(
        "--score-overlay",
        type=Path,
        default=None,
        help=(
            "Optional frozen OOS score parquet keyed by timestamp/symbol/side. "
            "The selected overlay column replaces calibrated_score_regime_ev "
            "after candidate preparation so all execution/policy logic remains fixed."
        ),
    )
    parser.add_argument("--score-overlay-column", default="rank_market_state")
    parser.add_argument(
        "--arm-name",
        action="append",
        default=[],
        help="Run only the named arm. May be supplied multiple times.",
    )
    parser.add_argument(
        "--live-compatible-selection",
        action="store_true",
        help=(
            "Select per timestamp batch using only prior rows for rolling references. "
            "This avoids full-day activity matching, which is not available at inference."
        ),
    )
    parser.add_argument("--save-decisions", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EPM_EXCHANGE"] = str(args.exchange)
    os.environ.setdefault("EXCHANGE_NAME", str(args.exchange))
    os.environ.setdefault("PRIMARY_EXCHANGE", str(args.exchange))
    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    load_start = eval_start - pd.Timedelta(days=int(args.max_window_days))
    parent_summary = _load_parent_summary(args.parent_policy_summary)

    rows = _prepare_rows(
        args.candidates,
        min_rank=0.0,
        rank_scope="per_strategy",
        regime_ev_rerank_admission=True,
        regime_ev_protected_admission_floor=TOP_THRESHOLDS[PROMOTED_TOP_SLICE],
        regime_ev_retained_surplus_frac=0.5,
    )
    if args.score_overlay is not None:
        overlay_columns = [
            "__ts__",
            "__symbol__",
            "side_name",
            str(args.score_overlay_column),
        ]
        overlay = pd.read_parquet(args.score_overlay, columns=overlay_columns).rename(
            columns={"__ts__": "timestamp", "__symbol__": "symbol"}
        )
        overlay["timestamp"] = pd.to_datetime(
            overlay["timestamp"], utc=True, errors="coerce"
        )
        overlay = overlay.drop_duplicates(
            ["timestamp", "symbol", "side_name"], keep="last"
        )
        rows = rows.merge(
            overlay,
            on=["timestamp", "symbol", "side_name"],
            how="left",
            validate="many_to_one",
        )
        coverage = pd.to_numeric(
            rows[str(args.score_overlay_column)], errors="coerce"
        ).notna()
        if not bool(coverage.any()):
            raise ValueError("Score overlay has no overlap with prepared policy rows")
        rows = rows.loc[coverage].copy()
        rows["calibrated_score_regime_ev"] = pd.to_numeric(
            rows[str(args.score_overlay_column)], errors="coerce"
        )
    rows = _add_time_columns(rows)
    rows = rows.loc[rows["timestamp"].ge(load_start) & rows["timestamp"].lt(eval_end)].copy()
    if rows.empty:
        raise RuntimeError(f"No candidate rows in {load_start} -> {eval_end}")
    score_col = _score_col(rows)
    arms = _arms()
    if args.arm_name:
        requested_arms = set(args.arm_name)
        arms = [arm for arm in arms if arm.name in requested_arms]
        found_arms = {arm.name for arm in arms}
        missing_arms = sorted(requested_arms.difference(found_arms))
        if missing_arms:
            raise ValueError(f"Requested arm-name values not found: {missing_arms}")
    simulated = _simulate_all_rows(
        rows,
        parent_summary=parent_summary,
        data_root=str(args.data_root),
        market_mode=str(args.market_mode),
        path_len=int(args.path_len),
        cost_pct=float(args.round_trip_cost_pct) / 2.0,
    )
    simulated = _add_time_columns(simulated)

    metric_rows: list[dict[str, Any]] = []
    weekly_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    archetype_frames: list[pd.DataFrame] = []
    for arm in arms:
        selected = _select_arm_rows(
            simulated,
            arm=arm,
            eval_start=eval_start,
            eval_end=eval_end,
            score_col=score_col,
            min_reference_rows=int(args.min_reference_rows),
            live_compatible_selection=bool(args.live_compatible_selection),
        )
        candidates = _portfolio_candidate_table(selected)
        if candidates.empty:
            decisions = pd.DataFrame()
            metrics: dict[str, Any] = {}
        else:
            ev_curve = fit_hierarchical_ev_curves(candidates)
            decisions, _equity, metrics = replay_candidates(
                candidates,
                _policy_params(),
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=str(args.market_mode),
            )
            decisions = _attach_candidate_context(decisions, selected)
        if args.save_decisions:
            decisions.to_parquet(args.out_dir / f"{arm.name}_decisions.parquet", index=False)
        metric_rows.append(_metrics_from_decisions(decisions, metrics, arm=arm, source_rows=len(selected)))
        weekly_frames.append(_breakdown(decisions, arm=arm, keys=["week_start"]))
        monthly_frames.append(_breakdown(decisions, arm=arm, keys=["month"]))
        archetype_frames.append(_breakdown(decisions, arm=arm, keys=["side_name", "policy_archetype"]))

    summary = pd.DataFrame(metric_rows)
    weekly = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    archetype = pd.concat(archetype_frames, ignore_index=True) if archetype_frames else pd.DataFrame()
    if not weekly.empty:
        stability = []
        for arm_name, group in weekly.groupby("arm", sort=False):
            pnl = pd.to_numeric(group["net_pnl"], errors="coerce")
            ret = pd.to_numeric(group["mean_net_return_per_trade"], errors="coerce")
            stability.append(
                {
                    "arm": arm_name,
                    "stable_net_score": float(pnl.mean() - 0.5 * pnl.std(ddof=0) + 0.25 * pnl.min()),
                    "stable_return_score": float(ret.mean() - 0.5 * ret.std(ddof=0) + 0.25 * ret.min()),
                    "positive_weeks": int(pnl.gt(0.0).sum()),
                    "weeks": int(len(pnl)),
                }
            )
        summary = summary.merge(pd.DataFrame(stability), on="arm", how="left")
    if "current_policy_top10_hr_rank50_regimecal" in set(summary["arm"]):
        base = summary.loc[summary["arm"].eq("current_policy_top10_hr_rank50_regimecal")].iloc[0]
        for col in ("trade_count", "net_pnl", "mean_net_return_per_trade", "worst_week", "full_sl_rate", "timeout_rate", "stable_net_score"):
            if col in summary.columns:
                summary[f"delta_{col}_vs_current"] = pd.to_numeric(summary[col], errors="coerce") - _safe_float(base.get(col))
    hard_baseline_name = "ev_target_archetype_reachable_match_current_activity_8d"
    if hard_baseline_name in set(summary["arm"]):
        base = summary.loc[summary["arm"].eq(hard_baseline_name)].iloc[0]
        for col in ("trade_count", "net_pnl", "mean_net_return_per_trade", "worst_week", "full_sl_rate", "timeout_rate", "stable_net_score"):
            if col in summary.columns:
                summary[f"delta_{col}_vs_hard8d"] = pd.to_numeric(summary[col], errors="coerce") - _safe_float(base.get(col))
    summary = summary.sort_values(["stable_net_score", "net_pnl", "mean_net_return_per_trade"], ascending=False)

    summary.to_csv(args.out_dir / "summary_metrics.csv", index=False)
    weekly.to_csv(args.out_dir / "weekly_metrics.csv", index=False)
    monthly.to_csv(args.out_dir / "monthly_metrics.csv", index=False)
    archetype.to_csv(args.out_dir / "side_archetype_metrics.csv", index=False)
    contract = _verify_policy_contract(args.source_report_dir, rows)
    manifest = {
        "generated_by": "compare_s52_threshold_basis_ablation",
        "candidates": str(args.candidates),
        "parent_policy_summary": str(args.parent_policy_summary),
        "source_report_dir": str(args.source_report_dir),
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "load_start": str(load_start),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "exchange": str(args.exchange),
        "score_col": score_col,
        "score_overlay": str(args.score_overlay) if args.score_overlay else None,
        "score_overlay_column": str(args.score_overlay_column),
        "score_overlay_role": (
            "frozen lifecycle market-state rank replacing admission score only"
            if args.score_overlay
            else None
        ),
        "live_compatible_selection": bool(args.live_compatible_selection),
        "selection_group": "timestamp" if args.live_compatible_selection else "eval_day",
        "regime_calibration": {
            "policy_id": "per_regime_archetype_calibration_v1",
            "used_for_admission_rank": True,
            "protected_admission_floor": TOP_THRESHOLDS[PROMOTED_TOP_SLICE],
            "retained_surplus_frac": 0.5,
        },
        "policy_contract_verification": contract,
        "arms": [arm.__dict__ for arm in arms],
        "outputs": {
            "summary_metrics": str(args.out_dir / "summary_metrics.csv"),
            "weekly_metrics": str(args.out_dir / "weekly_metrics.csv"),
            "monthly_metrics": str(args.out_dir / "monthly_metrics.csv"),
            "side_archetype_metrics": str(args.out_dir / "side_archetype_metrics.csv"),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    (args.out_dir / "policy_contract_verification.json").write_text(
        json.dumps(_json_safe(contract), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"event": "threshold_basis_ablation_done", "out_dir": args.out_dir, "top": summary.head(5).to_dict(orient="records")}), sort_keys=True))


if __name__ == "__main__":
    main()
