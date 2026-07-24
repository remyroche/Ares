#!/usr/bin/env python3
"""First-touch fixed-capture label proxy before base/meta training.

This diagnostic tests whether adding path ordering to the wide-stop fixed-capture
label makes the target more learnable. It reuses the simple-policy production
15m replay path store and delayed-entry machinery, then computes fixed TP/SL
first-touch outcomes from those paths. It does not run production LightGBM,
Optuna, or policy geometry optimisation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import simple_policy_optimiser as spo  # noqa: E402
from extreme_price_movements.timestamp_contract import (
    causal_signal_times,
    timeframe_delta,
)  # noqa: E402
from scripts.run_label_feature_store_model_smoke import _fit_predict, _month_model_frame  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    ROUND_TRIP_COST,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _sigmoid,
    _spearman,
)
from scripts.materialize_candidate_source_tags import (  # noqa: E402
    ARCHETYPE_COLS as SOURCE_ARCHETYPE_COLS,
    COMPONENT_COLS as SOURCE_COMPONENT_COLS,
    DEFAULT_CONFIG as SOURCE_TAG_CONFIG,
    build_archetype_scores,
    build_component_scores,
    build_feature_registry,
    load_config,
)
from scripts.run_label_weighted_proxy_ablation import _effective_sample_size  # noqa: E402
from scripts.run_label_widestop_capture_proxy import (  # noqa: E402
    CAPTURE_ARMS,
    CaptureArm,
    DEFAULT_MONTHS,
    DEFAULT_SEEDS,
    DEFAULT_TOP_FRACS,
    _capture_outcome as _aggregate_capture_outcome,
    _effective_n,
    _fit_holdout_summary,
    _format_table,
    _parse_float_csv,
    _parse_int_csv,
    _parse_csv,
    _rank_pct,
    _safe_numeric,
    _selection_metrics,
    _weekly_rows,
    _weights_for_target,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_first_touch_capture_proxy_v1")
EXECUTABLE_MARGIN_COST_FLOOR = 0.0100


def _arm_token(value: float, *, scale: float, width: int = 0) -> str:
    token = str(int(round(float(value) * float(scale))))
    return token.zfill(width) if width else token


def _parse_capture_arm_specs(value: str | None) -> list[CaptureArm]:
    """Parse semicolon specs: name:tp_r:sl_r:max_bars_to_mfe:max_barrier[:trail_r]."""
    if value is None or not str(value).strip():
        return []
    arms: list[CaptureArm] = []
    for spec in str(value).split(";"):
        spec = spec.strip()
        if not spec:
            continue
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) not in {5, 6}:
            raise ValueError(
                "Custom arm specs must use name:tp_r:sl_r:max_bars_to_mfe:max_barrier[:trail_r]; "
                f"got {spec!r}"
            )
        name, tp_r, sl_r, max_bars_to_mfe, max_barrier, *trail = parts
        if not name:
            raise ValueError(f"Custom arm name is empty in spec {spec!r}")
        arms.append(
            CaptureArm(
                name=name,
                tp_r=float(tp_r),
                sl_r=float(sl_r),
                max_bars_to_mfe=float(max_bars_to_mfe),
                max_barrier=float(max_barrier),
                trail_r=float(trail[0]) if trail else 0.50,
            )
        )
    return arms


def _build_grid_arms(
    *,
    tp_rs: list[float],
    sl_rs: list[float],
    trail_rs: list[float] | None = None,
    fast_bars: list[float],
    max_barriers: list[float],
    prefix: str,
) -> list[CaptureArm]:
    base_values = [tp_rs, sl_rs, fast_bars, max_barriers]
    if not any(base_values):
        return []
    trail_values = list(trail_rs or [0.50])
    values = [tp_rs, sl_rs, trail_values, fast_bars, max_barriers]
    if not all(values):
        raise ValueError(
            "Arm grid requires all of --arm-grid-tp-rs, --arm-grid-sl-rs, "
            "--arm-grid-trail-rs, --arm-grid-fast-bars, and --arm-grid-max-barriers."
        )
    arms: list[CaptureArm] = []
    seen: set[str] = set()
    multi_trail = len(trail_values) > 1
    for tp_r in tp_rs:
        for sl_r in sl_rs:
            for trail_r in trail_values:
                for fast in fast_bars:
                    for max_barrier in max_barriers:
                        trail_part = f"_tr{_arm_token(trail_r, scale=100.0, width=3)}" if multi_trail else ""
                        name = (
                            f"{prefix}_tp{_arm_token(tp_r, scale=100.0, width=3)}"
                            f"_sl{_arm_token(sl_r, scale=100.0, width=3)}"
                            f"{trail_part}"
                            f"_fast{_arm_token(fast, scale=1.0)}"
                            f"_bar{_arm_token(max_barrier, scale=1000.0)}"
                        )
                        if name in seen:
                            raise ValueError(f"Duplicate generated arm name: {name}")
                        seen.add(name)
                        arms.append(
                            CaptureArm(
                                name=name,
                                tp_r=float(tp_r),
                                sl_r=float(sl_r),
                                max_bars_to_mfe=float(fast),
                                max_barrier=float(max_barrier),
                                trail_r=float(trail_r),
                            )
                        )
    return arms


def _resolve_arms(
    *,
    arm_names: list[str],
    custom_arms: list[CaptureArm],
    only_custom_arms: bool,
) -> list[CaptureArm]:
    default_arms = list(CAPTURE_ARMS)
    if only_custom_arms:
        if not custom_arms:
            raise ValueError("--only-custom-arms requires --custom-arms or an arm grid.")
        available = list(custom_arms)
    else:
        available = default_arms + list(custom_arms)

    arms_by_name: dict[str, CaptureArm] = {}
    for arm in available:
        if arm.name in arms_by_name:
            raise ValueError(f"Duplicate capture arm name: {arm.name}")
        arms_by_name[arm.name] = arm

    if arm_names:
        missing = sorted(set(arm_names) - set(arms_by_name))
        if missing:
            raise ValueError(f"Unknown capture arm(s): {missing}")
        return [arms_by_name[name] for name in arm_names]
    return available


@contextmanager
def _temporary_exchange(exchange: str | None) -> Iterator[None]:
    if not exchange:
        yield
        return
    old = os.environ.get("EPM_EXCHANGE")
    os.environ["EPM_EXCHANGE"] = str(exchange)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("EPM_EXCHANGE", None)
        else:
            os.environ["EPM_EXCHANGE"] = old


def _infer_side(labels_path: Path, explicit_side: str | None) -> str:
    if explicit_side and str(explicit_side).strip().lower() in {"long", "short"}:
        return str(explicit_side).strip().lower()
    name = str(labels_path).lower()
    if "train_short" in name or "_short_" in name:
        return "short"
    return "long"


def _policy_rows(
    frame: pd.DataFrame,
    *,
    side: str,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """Build replay rows while preserving the source feature timestamp externally.

    The returned timestamp is only the executable path anchor.  Label callers
    retain ``frame['__ts__']`` as the model row timestamp, so an hourly signal
    observed over [t, t+1h) can be labelled from the first executable bar at
    t+1h without leaking that candle's OHLC path into its outcome.
    """
    side_val = -1.0 if str(side).lower() == "short" else 1.0
    source_ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    _, decision_ts = causal_signal_times(
        pd.DataFrame({"signal_bar_ts": source_ts}),
        timeframe=timeframe,
    )
    return pd.DataFrame(
        {
            "timestamp": source_ts,
            "signal_bar_ts": source_ts,
            "signal_bar_close_ts": decision_ts,
            "decision_ts": decision_ts,
            "symbol": frame["__symbol__"].astype(str),
            "side": np.full(len(frame), side_val, dtype=np.float32),
            "rank_pct": np.ones(len(frame), dtype=np.float32),
            "calibrated_score": np.ones(len(frame), dtype=np.float32),
            "barrier_pct": pd.to_numeric(frame["__barrier_pct__"], errors="coerce")
            .fillna(0.02)
            .clip(lower=1e-4)
            .to_numpy(dtype=np.float32),
        }
    )


def _fetch_policy_paths(
    frame: pd.DataFrame,
    *,
    labels_path: Path,
    side: str,
    data_root: Path,
    market_mode: str,
    exchange: str,
    path_len: int,
    apply_delayed_entry: bool,
    entry_delay_hours: int | None = None,
    timeframe: str = "1h",
) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    mandatory_delay = timeframe_delta(timeframe)
    if entry_delay_hours is not None and pd.Timedelta(hours=int(entry_delay_hours)) != mandatory_delay:
        raise ValueError(
            "entry_delay_hours is deprecated and must equal the signal timeframe; "
            "the causal signal-close offset cannot be disabled or changed"
        )
    rows = _policy_rows(
        frame,
        side=side,
        timeframe=timeframe,
    )
    with _temporary_exchange(exchange):
        ds = spo._make_policy_replay_store(
            str(data_root),
            str(market_mode),
            replay_timeframe="15m",
        )
        paths = spo._fetch_policy_paths(
            rows,
            ds,
            path_len=int(path_len),
            signal_timeframe=timeframe,
        )
        if apply_delayed_entry:
            rows, paths = spo._apply_delayed_entry_execution_model(
                rows,
                paths,
                data_root=str(data_root),
                market_mode=str(market_mode),
                signal_timeframe=timeframe,
                path_timeframe=str(getattr(ds, "timeframe", "15m")),
            )
    finite_mask = spo._policy_path_finite_mask(paths)
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    coverage_by_month: dict[str, float] = {}
    for month, idx in pd.Series(np.arange(len(frame)), index=frame.index).groupby(
        ts.dt.to_period("M").astype(str),
        dropna=False,
    ):
        pos = idx.to_numpy(dtype=np.int64)
        coverage_by_month[str(month)] = float(finite_mask[pos].mean()) if len(pos) else 0.0
    stats = {
        "labels_path": str(labels_path),
        "data_root": str(data_root),
        "market_mode": str(market_mode),
        "exchange": str(exchange),
        "side": str(side),
        "path_len": int(path_len),
        "path_timeframe": str(getattr(ds, "timeframe", "15m")),
        "mandatory_signal_close_offset": str(mandatory_delay),
        "path_start_contract": (
            "signal_timestamp_plus_timeframe_then_optional_delayed_execution"
        ),
        "apply_delayed_entry": bool(apply_delayed_entry),
        "rows": int(len(frame)),
        "finite_path_rows": int(finite_mask.sum()),
        "finite_path_coverage": float(finite_mask.mean()) if len(finite_mask) else 0.0,
        "finite_path_coverage_by_month": coverage_by_month,
    }
    if "entry_execution_source" in rows.columns:
        stats["entry_execution_source_counts"] = (
            rows["entry_execution_source"].astype(str).value_counts(dropna=False).to_dict()
        )
    return rows, paths, stats


def _feature_rank(frame: pd.DataFrame, names: tuple[str, ...], *, high_good: bool = True) -> pd.Series:
    present = [name for name in names if name in frame.columns]
    if not present:
        return pd.Series(0.5, index=frame.index, dtype=np.float32)
    temp = pd.DataFrame(index=frame.index)
    ts = pd.to_datetime(frame["__ts__"], errors="coerce")
    for name in present:
        values = pd.to_numeric(frame[name], errors="coerce")
        ranked = values.groupby(ts).rank(method="average", pct=True)
        if not high_good:
            ranked = 1.0 - ranked
        temp[name] = ranked.fillna(0.5).clip(0.0, 1.0).astype(np.float32)
    return temp.mean(axis=1).fillna(0.5).clip(0.0, 1.0).astype(np.float32)


def _clean_source_tag_config() -> dict[str, Any]:
    config = load_config(SOURCE_TAG_CONFIG)
    config["timestamp_col"] = "__ts__"
    config["symbol_col"] = "__symbol__"
    return config


def _score_family(frame: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    present = [col for col in cols if col in frame.columns]
    if not present:
        return pd.Series(0.5, index=frame.index, dtype=np.float32)
    values = frame[present].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return values.max(axis=1).fillna(0.5).clip(0.0, 1.0).astype(np.float32)


REGIME_FAMILY_MIN_SCORE = 0.55
REGIME_FAMILY_MIN_SCORE_GAP = 0.03
LEGACY_REGIME_FAMILY_MIN_SCORE = 0.55


def _add_regime_family_columns(
    frame: pd.DataFrame,
    *,
    min_score: float = REGIME_FAMILY_MIN_SCORE,
    min_score_gap: float = REGIME_FAMILY_MIN_SCORE_GAP,
    legacy_min_score: float = LEGACY_REGIME_FAMILY_MIN_SCORE,
) -> dict[str, Any]:
    """Add observable pre-entry regime-family scores used for geometry slicing."""
    try:
        config = _clean_source_tag_config()
        registry = build_feature_registry(frame, config)
        components, component_report = build_component_scores(frame, registry, config)
        archetypes = build_archetype_scores(frame, components, registry, config)
        archetypes["not_dirty_shock_score"] = (
            1.0 - pd.to_numeric(archetypes["dirty_shock_avoid_score"], errors="coerce")
        ).clip(0.0, 1.0).astype(np.float32)
        archetypes["loud_clean_source_score"] = (
            pd.to_numeric(archetypes["loud_breakout_impulse_score"], errors="coerce")
            * archetypes["not_dirty_shock_score"]
        ).clip(0.0, 1.0).astype(np.float32)
        for col in list(SOURCE_COMPONENT_COLS) + [c for c in SOURCE_ARCHETYPE_COLS if c in archetypes.columns]:
            source = components[col] if col in components.columns else archetypes[col]
            frame[f"__regime_source_{col}__"] = pd.to_numeric(source, errors="coerce").fillna(0.5).to_numpy(
                dtype=np.float32,
                copy=False,
            )

        scores = pd.DataFrame(
            {
                "trend_following": _score_family(
                    archetypes,
                    (
                        "quiet_continuation_score",
                        "run_entry_score",
                        "late_run_continuation_score",
                        "clean_run_entry_score",
                    ),
                ),
                "mean_reversion": _score_family(archetypes, ("retest_reversal_score",)),
                "vol_compression": _score_family(
                    archetypes,
                    (
                        "compression_release_score",
                        "compression_capture_candidate_score",
                        "risk_adjusted_capture_candidate_score",
                        "clean_economic_capture_candidate_score",
                    ),
                ),
                "breakout_impulse": _score_family(
                    archetypes,
                    (
                        "loud_breakout_impulse_score",
                        "loud_clean_execution_score",
                        "loud_clean_source_score",
                    ),
                ),
                "dirty_avoid": _score_family(
                    archetypes,
                    (
                        "dirty_shock_avoid_score",
                        "misleading_location_risk_score",
                    ),
                ),
            },
            index=frame.index,
        )
        score_values = scores.to_numpy(dtype=np.float32, copy=False)
        finite_values = np.where(np.isfinite(score_values), score_values, -np.inf)
        order = np.argsort(finite_values, axis=1)
        best_pos = order[:, -1]
        max_score_arr = finite_values[np.arange(len(finite_values)), best_pos]
        if finite_values.shape[1] >= 2:
            second_score_arr = finite_values[np.arange(len(finite_values)), order[:, -2]]
        else:
            second_score_arr = np.full(len(finite_values), -np.inf, dtype=np.float32)
        family_values = np.asarray(scores.columns, dtype=object)[best_pos]
        mixed_mask = (
            ~np.isfinite(max_score_arr)
            | (max_score_arr < float(min_score))
            | ((max_score_arr - np.where(np.isfinite(second_score_arr), second_score_arr, 0.0)) < float(min_score_gap))
        )
        family_values = family_values.astype(object, copy=True)
        family_values[mixed_mask] = "mixed"
        family = pd.Series(family_values, index=frame.index, dtype=object)
        for col in scores.columns:
            frame[f"__regime_source_{col}_score__"] = scores[col].to_numpy(dtype=np.float32, copy=False)
        frame["__regime_family__"] = family.to_numpy(dtype=object, copy=False)
        return {
            "scorer": "source_tag_component_archetype",
            "families": {str(k): int(v) for k, v in family.value_counts(dropna=False).sort_index().items()},
            "score_means": {str(k): float(v) for k, v in scores.mean().to_dict().items()},
            "score_stds": {str(k): float(v) for k, v in scores.std().to_dict().items()},
            "assignment_thresholds": {
                "min_score": float(min_score),
                "min_score_gap": float(min_score_gap),
                "fallback_family": "mixed",
            },
            "source_columns_used": int(len(registry.get("source_columns") or [])),
            "registry_group_counts": {
                str(k): int(len(v)) for k, v in (registry.get("available") or {}).items()
            },
            "component_report": component_report,
        }
    except Exception as exc:
        # Fallback keeps the diagnostic runnable if a feature snapshot lacks the
        # richer source-tag contract, but the manifest records that downgrade.
        fallback_error = str(exc)

    trend = _feature_rank(
        frame,
        (
            "trend_strength_percentile",
            "regime_trend_score",
            "trend_z_t",
            "trend_t",
            "trend_stack_6_12_24",
            "trend_alignment_1_3_6",
            "ema20_slope_5h",
        ),
    )
    mean_reversion = _feature_rank(
        frame,
        (
            "mean_reversion_score",
            "mr_potential",
            "pullback_depth",
            "loc_pullback_depth_24",
            "loc_pullback_depth_48",
            "support_reclaim_score",
            "resistance_reject_score",
            "trend_overextension_z",
        ),
    )
    compression = _feature_rank(
        frame,
        (
            "atr_compression_ratio",
            "compression_score",
            "vol_compression",
            "vol_z_30_calm",
            "bollinger_band_width",
            "rolling_range_20",
            "path_efficiency_24",
        ),
    )
    impulse = _feature_rank(
        frame,
        (
            "shock_12h",
            "jump_intensity",
            "breakout_24h",
            "breakout_confirmed",
            "breakout_soft",
            "pct_breakout_t",
            "vw_breakout",
            "second_leg_accel_1h",
        ),
    )
    scores = pd.DataFrame(
        {
            "trend_following": trend,
            "mean_reversion": mean_reversion,
            "vol_compression": compression,
            "breakout_impulse": impulse,
        },
        index=frame.index,
    )
    max_score = scores.max(axis=1)
    family = scores.idxmax(axis=1).astype(str)
    family = family.where(max_score >= float(legacy_min_score), "mixed")
    frame["__regime_trend_following_score__"] = trend.to_numpy(dtype=np.float32, copy=False)
    frame["__regime_mean_reversion_score__"] = mean_reversion.to_numpy(dtype=np.float32, copy=False)
    frame["__regime_vol_compression_score__"] = compression.to_numpy(dtype=np.float32, copy=False)
    frame["__regime_breakout_impulse_score__"] = impulse.to_numpy(dtype=np.float32, copy=False)
    frame["__regime_family__"] = family.to_numpy(dtype=object, copy=False)
    return {
        "scorer": "legacy_rank_fallback",
        "fallback_error": fallback_error,
        "families": {str(k): int(v) for k, v in family.value_counts(dropna=False).sort_index().items()},
        "score_means": {str(k): float(v) for k, v in scores.mean().to_dict().items()},
        "assignment_thresholds": {
            "min_score": float(legacy_min_score),
            "fallback_family": "mixed",
        },
    }


def _same_bar_first_touch(
    *,
    open_px: np.ndarray,
    tp_px: np.ndarray,
    sl_px: np.ndarray,
    tp_hit: np.ndarray,
    sl_hit: np.ndarray,
) -> np.ndarray:
    """Return 1 for TP, -1 for SL, 0 for neither using execution tie-breaks."""
    result = np.zeros(len(open_px), dtype=np.int8)
    only_tp = tp_hit & ~sl_hit
    only_sl = sl_hit & ~tp_hit
    both = tp_hit & sl_hit
    result[only_tp] = 1
    result[only_sl] = -1
    if np.any(both):
        tp_dist = np.abs(tp_px[both] - open_px[both])
        sl_dist = np.abs(sl_px[both] - open_px[both])
        # Same-bar rule mirrors execution semantics: shortest distance wins;
        # exact ties are adverse.
        both_result = np.where(tp_dist < sl_dist, 1, -1).astype(np.int8)
        result[np.flatnonzero(both)] = both_result
    return result


def _first_threshold_bar(path_r: np.ndarray, threshold_r: float) -> np.ndarray:
    """First 1-indexed bar where a path reaches a threshold in R units."""
    hit = np.asarray(path_r, dtype=np.float64) >= float(threshold_r)
    any_hit = np.any(hit, axis=1)
    first = np.argmax(hit, axis=1).astype(np.float32) + 1.0
    first[~any_hit] = np.nan
    return first


def _path_order_columns(fav: np.ndarray, adv: np.ndarray, barrier: np.ndarray) -> dict[str, np.ndarray]:
    """Compute label-side first-passage order columns from favorable/adverse paths."""
    denom = np.maximum(np.asarray(barrier, dtype=np.float64), 1e-8)
    fav_r = np.asarray(fav, dtype=np.float64) / denom[:, None]
    adv_r = np.asarray(adv, dtype=np.float64) / denom[:, None]
    cols: dict[str, np.ndarray] = {}
    for threshold, suffix in ((0.50, "05"), (0.75, "075"), (1.00, "1"), (1.25, "125"), (1.50, "15")):
        cols[f"bars_to_mfe_{suffix}r"] = _first_threshold_bar(fav_r, threshold)
    for threshold, suffix in ((0.50, "05"), (0.75, "075"), (1.00, "1"), (1.50, "15")):
        cols[f"bars_to_mae_{suffix}r"] = _first_threshold_bar(adv_r, threshold)

    mfe_1 = cols["bars_to_mfe_1r"]
    for suffix in ("05", "075", "1"):
        mae = cols[f"bars_to_mae_{suffix}r"]
        cols[f"mfe_1r_before_mae_{suffix}r"] = (
            np.isfinite(mfe_1) & ((~np.isfinite(mae)) | (mfe_1 < mae))
        ).astype(np.int8)
        cols[f"mae_{suffix}r_before_mfe_1r"] = (
            np.isfinite(mae) & ((~np.isfinite(mfe_1)) | (mae < mfe_1))
        ).astype(np.int8)

    max_adverse = np.full(len(fav_r), np.nan, dtype=np.float32)
    underwater_bars = np.zeros(len(fav_r), dtype=np.float32)
    underwater_fraction = np.zeros(len(fav_r), dtype=np.float32)
    area_underwater = np.zeros(len(fav_r), dtype=np.float32)
    for i, bar in enumerate(mfe_1):
        if not np.isfinite(bar):
            upto = adv_r.shape[1]
        else:
            upto = max(1, min(int(bar), adv_r.shape[1]))
        prefix = adv_r[i, :upto]
        finite = prefix[np.isfinite(prefix)]
        if finite.size:
            max_adverse[i] = np.float32(np.nanmax(finite))
            positive = finite > 0.0
            underwater_bars[i] = np.float32(np.sum(positive))
            underwater_fraction[i] = np.float32(np.mean(positive))
            area_underwater[i] = np.float32(np.nansum(np.maximum(finite, 0.0)))
    cols["max_adverse_before_mfe_1r"] = max_adverse
    cols["underwater_bars_before_mfe_1r"] = underwater_bars
    cols["underwater_fraction_before_mfe_1r"] = underwater_fraction
    cols["area_underwater_before_mfe_1r"] = area_underwater
    return cols


def _path_ordered_capture_soft_target(
    *,
    capture_net: np.ndarray,
    round_trip_cost: float,
    executable_cost_floor: float = EXECUTABLE_MARGIN_COST_FLOOR,
    target_mode: str = "path_ordered",
    denom: np.ndarray,
    hit: np.ndarray,
    stop: np.ndarray,
    timeout: np.ndarray,
    valid_path: np.ndarray,
    same_bar_both: np.ndarray,
    first_touch_mae_norm: np.ndarray,
    path_order: dict[str, np.ndarray],
) -> pd.Series:
    """Build an S52-style soft target that rewards clean first passage.

    The geometry search is a base-layer learnability diagnostic, so the target
    uses gross executable capture for signal strength while explicitly capping
    high-MFE rows whose favorable move comes after unacceptable adverse path.
    """
    mode = str(target_mode or "path_ordered").strip().lower()
    margin_modes = {"executable_margin", "exec_margin", "margin"}
    margin_hybrid_modes = {"executable_margin_hybrid", "exec_margin_hybrid", "margin_hybrid"}
    cost_floor = max(float(round_trip_cost), float(executable_cost_floor))
    gross_capture = np.asarray(capture_net, dtype=np.float64) + float(round_trip_cost)
    executable_margin = gross_capture - cost_floor
    if mode in margin_modes:
        score_capture = executable_margin
    elif mode in margin_hybrid_modes:
        score_capture = 0.70 * gross_capture + 0.30 * executable_margin
    else:
        score_capture = gross_capture
    base = _sigmoid(score_capture / np.maximum(np.asarray(denom, dtype=np.float64), 1e-4))
    first_good = (
        np.asarray(hit, dtype=bool)
        & np.asarray(valid_path, dtype=bool)
        & ~np.asarray(same_bar_both, dtype=bool)
        & ((executable_margin > 0.0) if mode in margin_modes else (gross_capture > 0.0))
    )
    mfe_before = np.asarray(path_order.get("mfe_1r_before_mae_1r", np.zeros_like(gross_capture)), dtype=float) > 0.5
    mae_before = np.asarray(path_order.get("mae_1r_before_mfe_1r", np.zeros_like(gross_capture)), dtype=float) > 0.5
    adverse_before = np.asarray(path_order.get("max_adverse_before_mfe_1r", np.nan), dtype=np.float64)
    underwater_bars = np.asarray(path_order.get("underwater_bars_before_mfe_1r", np.nan), dtype=np.float64)
    underwater_fraction = np.asarray(path_order.get("underwater_fraction_before_mfe_1r", np.nan), dtype=np.float64)
    first_touch_mae = np.asarray(first_touch_mae_norm, dtype=np.float64)

    clean_ordered = (
        first_good
        & mfe_before
        & np.isfinite(adverse_before)
        & (adverse_before <= 1.50)
        & np.isfinite(underwater_bars)
        & (underwater_bars <= 10.0)
        & np.isfinite(underwater_fraction)
        & ((underwater_fraction <= 0.45) | (underwater_bars <= 2.0))
        & np.isfinite(first_touch_mae)
        & (first_touch_mae < 1.0)
    )

    soft = 0.35 * base
    soft += 0.25 * first_good.astype(np.float64)
    soft += 0.20 * mfe_before.astype(np.float64)
    soft += 0.25 * clean_ordered.astype(np.float64)
    if mode in margin_modes | margin_hybrid_modes:
        margin_strength = np.clip(executable_margin / np.maximum(cost_floor, 1e-6), -2.0, 3.0)
        soft += 0.20 * np.maximum(margin_strength, 0.0)
        soft -= (0.25 if mode in margin_modes else 0.10) * (executable_margin <= 0.0).astype(np.float64)
    soft -= 0.20 * mae_before.astype(np.float64)
    soft -= 0.10 * np.maximum(first_touch_mae - 0.75, 0.0)
    soft -= 0.03 * np.maximum(adverse_before - 1.0, 0.0)
    soft -= 0.01 * np.maximum(underwater_bars - 6.0, 0.0)
    soft -= 0.08 * np.maximum(underwater_fraction - 0.35, 0.0)

    soft = np.where(clean_ordered, np.maximum(soft, 0.65), soft)
    dirty_cap = np.ones_like(soft, dtype=np.float64)
    if mode in margin_modes:
        dirty_cap = np.where(executable_margin <= 0.0, np.minimum(dirty_cap, 0.25), dirty_cap)
    elif mode in margin_hybrid_modes:
        dirty_cap = np.where(executable_margin <= 0.0, np.minimum(dirty_cap, 0.45), dirty_cap)
    dirty_cap = np.where(np.asarray(stop, dtype=bool), np.minimum(dirty_cap, 0.05), dirty_cap)
    dirty_cap = np.where(np.asarray(timeout, dtype=bool), np.minimum(dirty_cap, 0.08), dirty_cap)
    dirty_cap = np.where(np.asarray(same_bar_both, dtype=bool), np.minimum(dirty_cap, 0.05), dirty_cap)
    dirty_cap = np.where(mae_before, np.minimum(dirty_cap, 0.10), dirty_cap)
    dirty_cap = np.where(first_touch_mae >= 1.0, np.minimum(dirty_cap, 0.12), dirty_cap)
    dirty_cap = np.where(adverse_before > 1.50, np.minimum(dirty_cap, 0.12), dirty_cap)
    dirty_cap = np.where(underwater_bars > 10.0, np.minimum(dirty_cap, 0.15), dirty_cap)
    dirty_cap = np.where(
        (underwater_fraction > 0.45) & (underwater_bars > 2.0),
        np.minimum(dirty_cap, 0.15),
        dirty_cap,
    )
    soft = np.minimum(soft, dirty_cap)
    soft = np.where(np.asarray(valid_path, dtype=bool), soft, 0.0)
    return pd.Series(soft).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def _first_touch_capture_outcome(
    frame: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    arm: CaptureArm,
    *,
    side_name: str,
    outcome_mode: str = "fixed_tp",
    round_trip_cost: float = ROUND_TRIP_COST,
    target_mode: str = "path_ordered",
    executable_cost_floor: float = EXECUTABLE_MARGIN_COST_FLOOR,
    first_outcome_bar: int = 0,
) -> pd.DataFrame:
    if str(outcome_mode).strip().lower() in {"trailing_profit", "trailing", "trail"}:
        return _trailing_profit_capture_outcome(
            frame,
            paths,
            arm,
            side_name=side_name,
            round_trip_cost=round_trip_cost,
            target_mode=target_mode,
            executable_cost_floor=executable_cost_floor,
            first_outcome_bar=first_outcome_bar,
        )
    f_opens, f_highs, f_lows, f_closes = paths
    n = len(frame)
    entry = np.asarray(f_opens[:, 0], dtype=np.float64)
    side_value = -1.0 if str(side_name).strip().lower() == "short" else 1.0
    side = np.full(n, side_value, dtype=np.float64)
    barrier = pd.to_numeric(frame["__barrier_pct__"], errors="coerce").to_numpy(dtype=np.float64)
    barrier = np.clip(np.nan_to_num(np.abs(barrier), nan=np.nan), 1e-8, None)
    valid_path = spo._policy_path_finite_mask(paths)
    valid_entry = valid_path & np.isfinite(entry) & (entry > 0.0) & np.isfinite(barrier)
    eligible = valid_entry & (barrier <= float(arm.max_barrier))

    tp_ret = float(arm.tp_r) * barrier
    sl_ret = float(arm.sl_r) * barrier
    tp_px = entry * (1.0 + side * tp_ret)
    sl_px = entry * (1.0 - side * sl_ret)

    hit = np.zeros(n, dtype=bool)
    stop = np.zeros(n, dtype=bool)
    timeout = np.zeros(n, dtype=bool)
    same_bar_both = np.zeros(n, dtype=bool)
    first_bar = np.full(n, np.nan, dtype=np.float32)
    cost = float(round_trip_cost)
    cost_floor = max(float(round_trip_cost), float(executable_cost_floor))
    capture_net = np.full(n, -cost, dtype=np.float64)

    first_outcome_bar = int(np.clip(first_outcome_bar, 0, max(f_opens.shape[1] - 1, 0)))
    fav = np.where(
        side[:, None] >= 0.0,
        (f_highs.astype(np.float64) - entry[:, None]) / np.maximum(entry[:, None], 1e-12),
        (entry[:, None] - f_lows.astype(np.float64)) / np.maximum(entry[:, None], 1e-12),
    )
    adv = np.where(
        side[:, None] >= 0.0,
        (entry[:, None] - f_lows.astype(np.float64)) / np.maximum(entry[:, None], 1e-12),
        (f_highs.astype(np.float64) - entry[:, None]) / np.maximum(entry[:, None], 1e-12),
    )
    fav = np.where(np.isfinite(fav), np.maximum(fav, 0.0), np.nan)
    adv = np.where(np.isfinite(adv), np.maximum(adv, 0.0), np.nan)
    outcome_fav = fav[:, first_outcome_bar:]
    outcome_adv = adv[:, first_outcome_bar:]
    full_path_max_fav = np.nanmax(outcome_fav, axis=1)
    full_path_max_adv = np.nanmax(outcome_adv, axis=1)
    max_fav_to_decision = np.full(n, np.nan, dtype=np.float64)
    max_adv_to_decision = np.full(n, np.nan, dtype=np.float64)

    active = eligible.copy()
    max_tp_bar = max(0, int(math.floor(float(arm.max_bars_to_mfe))))
    for j in range(first_outcome_bar, f_opens.shape[1]):
        active_idx = np.flatnonzero(active)
        if len(active_idx) == 0:
            break
        hi = f_highs[active_idx, j].astype(np.float64, copy=False)
        lo = f_lows[active_idx, j].astype(np.float64, copy=False)
        op = f_opens[active_idx, j].astype(np.float64, copy=False)
        tp = tp_px[active_idx]
        sl = sl_px[active_idx]
        long_mask = side[active_idx] >= 0.0
        tp_hit = np.where(long_mask, hi >= tp, lo <= tp)
        tp_hit &= (j + 1) <= max_tp_bar
        sl_hit = np.where(long_mask, lo <= sl, hi >= sl)
        both = tp_hit & sl_hit
        decision = _same_bar_first_touch(
            open_px=op,
            tp_px=tp,
            sl_px=sl,
            tp_hit=tp_hit,
            sl_hit=sl_hit,
        )
        if not np.any(decision):
            continue
        decided = active_idx[decision != 0]
        hit_decided = decided[decision[decision != 0] > 0]
        stop_decided = decided[decision[decision != 0] < 0]
        if len(hit_decided):
            hit[hit_decided] = True
            capture_net[hit_decided] = tp_ret[hit_decided] - cost
            first_bar[hit_decided] = float(j + 1)
        if len(stop_decided):
            stop[stop_decided] = True
            capture_net[stop_decided] = -sl_ret[stop_decided] - cost
            first_bar[stop_decided] = float(j + 1)
        max_fav_to_decision[decided] = np.nanmax(
            fav[decided, first_outcome_bar : j + 1], axis=1
        )
        max_adv_to_decision[decided] = np.nanmax(
            adv[decided, first_outcome_bar : j + 1], axis=1
        )
        if np.any(both):
            same_bar_both[active_idx[both]] = True
        active[decided] = False

    still_active = active & eligible
    if np.any(still_active):
        last_close = f_closes[np.flatnonzero(still_active), -1].astype(np.float64, copy=False)
        active_idx = np.flatnonzero(still_active)
        final_ret = side[active_idx] * (
            last_close / np.maximum(entry[active_idx], 1e-12) - 1.0
        )
        final_ret = np.where(np.isfinite(final_ret), final_ret, -cost)
        capture_net[active_idx] = final_ret - cost
        timeout[active_idx] = True
        first_bar[active_idx] = float(f_opens.shape[1])
        max_fav_to_decision[active_idx] = full_path_max_fav[active_idx]
        max_adv_to_decision[active_idx] = full_path_max_adv[active_idx]

    denom = np.maximum(tp_ret + sl_ret + cost, 1e-4)
    path_order = _path_order_columns(outcome_fav, outcome_adv, barrier)
    first_touch_mae_norm = max_adv_to_decision / np.maximum(barrier, 1e-8)
    target_soft = _path_ordered_capture_soft_target(
        capture_net=capture_net,
        round_trip_cost=cost,
        executable_cost_floor=cost_floor,
        target_mode=target_mode,
        denom=denom,
        hit=hit,
        stop=stop,
        timeout=timeout,
        valid_path=valid_path,
        same_bar_both=same_bar_both,
        first_touch_mae_norm=first_touch_mae_norm,
        path_order=path_order,
    )
    target_soft.index = frame.index
    capture_gross = capture_net + cost
    executable_cost = np.full(n, cost_floor, dtype=np.float64)
    executable_margin = capture_gross - executable_cost
    out = pd.DataFrame(
        {
            "target_soft": target_soft,
            "target_hard": hit.astype(float),
            "capture_net": capture_net,
            "capture_gross": capture_gross,
            "executable_cost": executable_cost,
            "executable_cost_floor": float(cost_floor),
            "executable_margin": executable_margin,
            "gross_minus_cost_floor": executable_margin,
            "executable_margin_positive": (executable_margin > 0.0).astype(float),
            "round_trip_cost": cost,
            "target_mode": str(target_mode),
            "capture_hit": hit.astype(float),
            "capture_stop": stop.astype(float),
            "capture_timeout": timeout.astype(float),
            "capture_eligible": eligible.astype(float),
            "capture_valid_path": valid_path.astype(float),
            "same_bar_both_hit": same_bar_both.astype(float),
            "first_touch_bar": first_bar,
            "tp_r": float(arm.tp_r),
            "sl_r": float(arm.sl_r),
            "effective_tp_abs": tp_ret,
            "effective_sl_abs": sl_ret,
            "mae_to_sl": (max_adv_to_decision / np.maximum(sl_ret, 1e-8)),
            "mfe_to_tp": (max_fav_to_decision / np.maximum(tp_ret, 1e-8)),
            "first_touch_mfe_norm": (max_fav_to_decision / np.maximum(barrier, 1e-8)),
            "first_touch_mae_norm": first_touch_mae_norm,
            "full_path_mae_to_sl": (full_path_max_adv / np.maximum(sl_ret, 1e-8)),
            "full_path_mfe_to_tp": (full_path_max_fav / np.maximum(tp_ret, 1e-8)),
            "full_path_mfe_norm": (full_path_max_fav / np.maximum(barrier, 1e-8)),
            "full_path_mae_norm": (full_path_max_adv / np.maximum(barrier, 1e-8)),
        },
        index=frame.index,
    )
    for col, values in path_order.items():
        out[col] = values
    for col in (
        "mae_to_sl",
        "mfe_to_tp",
        "first_touch_mfe_norm",
        "first_touch_mae_norm",
        "full_path_mae_to_sl",
        "full_path_mfe_to_tp",
        "full_path_mfe_norm",
        "full_path_mae_norm",
    ):
        out.loc[~np.isfinite(out[col]), col] = np.nan
    return out


def _trailing_profit_capture_outcome(
    frame: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    arm: CaptureArm,
    *,
    side_name: str,
    round_trip_cost: float = ROUND_TRIP_COST,
    target_mode: str = "path_ordered",
    executable_cost_floor: float = EXECUTABLE_MARGIN_COST_FLOOR,
    first_outcome_bar: int = 0,
) -> pd.DataFrame:
    f_opens, f_highs, f_lows, f_closes = paths
    n = len(frame)
    entry = np.asarray(f_opens[:, 0], dtype=np.float64)
    side_value = -1.0 if str(side_name).strip().lower() == "short" else 1.0
    side = np.full(n, side_value, dtype=np.float64)
    barrier = pd.to_numeric(frame["__barrier_pct__"], errors="coerce").to_numpy(dtype=np.float64)
    barrier = np.clip(np.nan_to_num(np.abs(barrier), nan=np.nan), 1e-8, None)
    valid_path = spo._policy_path_finite_mask(paths)
    valid_entry = valid_path & np.isfinite(entry) & (entry > 0.0) & np.isfinite(barrier)
    eligible = valid_entry & (barrier <= float(arm.max_barrier))

    activation_ret = float(arm.tp_r) * barrier
    sl_ret = float(arm.sl_r) * barrier
    trail_ret = max(float(getattr(arm, "trail_r", 0.50)), 1e-6) * barrier
    activation_px = entry * (1.0 + side * activation_ret)
    sl_px = entry * (1.0 - side * sl_ret)

    hit = np.zeros(n, dtype=bool)
    stop = np.zeros(n, dtype=bool)
    timeout = np.zeros(n, dtype=bool)
    activated = np.zeros(n, dtype=bool)
    same_bar_both = np.zeros(n, dtype=bool)
    first_bar = np.full(n, np.nan, dtype=np.float32)
    activation_bar = np.full(n, np.nan, dtype=np.float32)
    cost = float(round_trip_cost)
    cost_floor = max(float(round_trip_cost), float(executable_cost_floor))
    capture_net = np.full(n, -cost, dtype=np.float64)

    first_outcome_bar = int(np.clip(first_outcome_bar, 0, max(f_opens.shape[1] - 1, 0)))
    fav = np.where(
        side[:, None] >= 0.0,
        (f_highs.astype(np.float64) - entry[:, None]) / np.maximum(entry[:, None], 1e-12),
        (entry[:, None] - f_lows.astype(np.float64)) / np.maximum(entry[:, None], 1e-12),
    )
    adv = np.where(
        side[:, None] >= 0.0,
        (entry[:, None] - f_lows.astype(np.float64)) / np.maximum(entry[:, None], 1e-12),
        (f_highs.astype(np.float64) - entry[:, None]) / np.maximum(entry[:, None], 1e-12),
    )
    fav = np.where(np.isfinite(fav), np.maximum(fav, 0.0), np.nan)
    adv = np.where(np.isfinite(adv), np.maximum(adv, 0.0), np.nan)
    outcome_fav = fav[:, first_outcome_bar:]
    outcome_adv = adv[:, first_outcome_bar:]
    full_path_max_fav = np.nanmax(outcome_fav, axis=1)
    full_path_max_adv = np.nanmax(outcome_adv, axis=1)
    max_fav_to_decision = np.full(n, np.nan, dtype=np.float64)
    max_adv_to_decision = np.full(n, np.nan, dtype=np.float64)
    best_fav = np.zeros(n, dtype=np.float64)

    active = eligible.copy()
    max_activation_bar = max(0, int(math.floor(float(arm.max_bars_to_mfe))))
    for j in range(first_outcome_bar, f_opens.shape[1]):
        active_idx = np.flatnonzero(active)
        if len(active_idx) == 0:
            break
        hi = f_highs[active_idx, j].astype(np.float64, copy=False)
        lo = f_lows[active_idx, j].astype(np.float64, copy=False)
        op = f_opens[active_idx, j].astype(np.float64, copy=False)
        long_mask = side[active_idx] >= 0.0

        was_activated = activated[active_idx]
        trail_level_ret = np.maximum(best_fav[active_idx] - trail_ret[active_idx], 0.0)
        trail_px = entry[active_idx] * (1.0 + side[active_idx] * trail_level_ret)
        trail_hit = was_activated & np.where(long_mask, lo <= trail_px, hi >= trail_px)
        sl_hit = np.where(long_mask, lo <= sl_px[active_idx], hi >= sl_px[active_idx])

        decision = np.zeros(len(active_idx), dtype=np.int8)
        decision[trail_hit] = 1
        decision[sl_hit & ~trail_hit] = -1
        both_exit = sl_hit & trail_hit
        if np.any(both_exit):
            tie = _same_bar_first_touch(
                open_px=op[both_exit],
                tp_px=trail_px[both_exit],
                sl_px=sl_px[active_idx][both_exit],
                tp_hit=np.ones(int(both_exit.sum()), dtype=bool),
                sl_hit=np.ones(int(both_exit.sum()), dtype=bool),
            )
            decision[np.flatnonzero(both_exit)] = tie

        decided = active_idx[decision != 0]
        if len(decided):
            hit_decided = decided[decision[decision != 0] > 0]
            stop_decided = decided[decision[decision != 0] < 0]
            if len(hit_decided):
                hit[hit_decided] = True
                exit_ret = np.maximum(best_fav[hit_decided] - trail_ret[hit_decided], 0.0)
                capture_net[hit_decided] = exit_ret - cost
                first_bar[hit_decided] = float(j + 1)
            if len(stop_decided):
                stop[stop_decided] = True
                capture_net[stop_decided] = -sl_ret[stop_decided] - cost
                first_bar[stop_decided] = float(j + 1)
            max_fav_to_decision[decided] = np.nanmax(
                fav[decided, first_outcome_bar : j + 1], axis=1
            )
            max_adv_to_decision[decided] = np.nanmax(
                adv[decided, first_outcome_bar : j + 1], axis=1
            )
            active[decided] = False

        active_idx = np.flatnonzero(active)
        if len(active_idx) == 0:
            break
        hi = f_highs[active_idx, j].astype(np.float64, copy=False)
        lo = f_lows[active_idx, j].astype(np.float64, copy=False)
        op = f_opens[active_idx, j].astype(np.float64, copy=False)
        long_mask = side[active_idx] >= 0.0
        activation_hit = np.where(long_mask, hi >= activation_px[active_idx], lo <= activation_px[active_idx])
        activation_hit &= (j + 1) <= max_activation_bar
        sl_hit = np.where(long_mask, lo <= sl_px[active_idx], hi >= sl_px[active_idx])
        both = (~activated[active_idx]) & activation_hit & sl_hit
        if np.any(both):
            tie = _same_bar_first_touch(
                open_px=op[both],
                tp_px=activation_px[active_idx][both],
                sl_px=sl_px[active_idx][both],
                tp_hit=np.ones(int(both.sum()), dtype=bool),
                sl_hit=np.ones(int(both.sum()), dtype=bool),
            )
            stop_now = active_idx[np.flatnonzero(both)[tie < 0]]
            activate_now = active_idx[np.flatnonzero(both)[tie > 0]]
            if len(stop_now):
                stop[stop_now] = True
                capture_net[stop_now] = -sl_ret[stop_now] - cost
                first_bar[stop_now] = float(j + 1)
                max_fav_to_decision[stop_now] = np.nanmax(
                    fav[stop_now, first_outcome_bar : j + 1], axis=1
                )
                max_adv_to_decision[stop_now] = np.nanmax(
                    adv[stop_now, first_outcome_bar : j + 1], axis=1
                )
                active[stop_now] = False
            if len(activate_now):
                activated[activate_now] = True
                activation_bar[activate_now] = float(j + 1)
                best_fav[activate_now] = np.maximum(best_fav[activate_now], fav[activate_now, j])
            same_bar_both[active_idx[both]] = True

        active_idx = np.flatnonzero(active)
        if len(active_idx):
            long_mask = side[active_idx] >= 0.0
            activation_hit = np.where(
                long_mask,
                f_highs[active_idx, j].astype(np.float64, copy=False) >= activation_px[active_idx],
                f_lows[active_idx, j].astype(np.float64, copy=False) <= activation_px[active_idx],
            )
            activation_hit &= (j + 1) <= max_activation_bar
            activate = active_idx[(~activated[active_idx]) & activation_hit]
            if len(activate):
                activated[activate] = True
                activation_bar[activate] = float(j + 1)
            still = active_idx[activated[active_idx]]
            if len(still):
                best_fav[still] = np.maximum(best_fav[still], np.nan_to_num(fav[still, j], nan=0.0))

    still_active = active & eligible
    if np.any(still_active):
        active_idx = np.flatnonzero(still_active)
        last_close = f_closes[active_idx, -1].astype(np.float64, copy=False)
        final_ret = side[active_idx] * (
            last_close / np.maximum(entry[active_idx], 1e-12) - 1.0
        )
        final_ret = np.where(np.isfinite(final_ret), final_ret, -cost)
        capture_net[active_idx] = final_ret - cost
        timeout[active_idx] = True
        first_bar[active_idx] = float(f_opens.shape[1])
        max_fav_to_decision[active_idx] = full_path_max_fav[active_idx]
        max_adv_to_decision[active_idx] = full_path_max_adv[active_idx]

    denom = np.maximum(activation_ret + sl_ret + trail_ret + cost, 1e-4)
    path_order = _path_order_columns(outcome_fav, outcome_adv, barrier)
    first_touch_mae_norm = max_adv_to_decision / np.maximum(barrier, 1e-8)
    target_soft = _path_ordered_capture_soft_target(
        capture_net=capture_net,
        round_trip_cost=cost,
        executable_cost_floor=cost_floor,
        target_mode=target_mode,
        denom=denom,
        hit=hit,
        stop=stop,
        timeout=timeout,
        valid_path=valid_path,
        same_bar_both=same_bar_both,
        first_touch_mae_norm=first_touch_mae_norm,
        path_order=path_order,
    )
    target_soft.index = frame.index
    capture_gross = capture_net + cost
    executable_cost = np.full(n, cost_floor, dtype=np.float64)
    executable_margin = capture_gross - executable_cost
    out = pd.DataFrame(
        {
            "target_soft": target_soft,
            "target_hard": hit.astype(float),
            "capture_net": capture_net,
            "capture_gross": capture_gross,
            "executable_cost": executable_cost,
            "executable_cost_floor": float(cost_floor),
            "executable_margin": executable_margin,
            "gross_minus_cost_floor": executable_margin,
            "executable_margin_positive": (executable_margin > 0.0).astype(float),
            "round_trip_cost": cost,
            "target_mode": str(target_mode),
            "capture_hit": hit.astype(float),
            "capture_stop": stop.astype(float),
            "capture_timeout": timeout.astype(float),
            "capture_eligible": eligible.astype(float),
            "capture_valid_path": valid_path.astype(float),
            "same_bar_both_hit": same_bar_both.astype(float),
            "first_touch_bar": first_bar,
            "trailing_activation_bar": activation_bar,
            "trailing_activated": activated.astype(float),
            "tp_r": float(arm.tp_r),
            "sl_r": float(arm.sl_r),
            "trail_r": float(getattr(arm, "trail_r", 0.50)),
            "effective_tp_abs": activation_ret,
            "effective_sl_abs": sl_ret,
            "effective_trail_abs": trail_ret,
            "mae_to_sl": (max_adv_to_decision / np.maximum(sl_ret, 1e-8)),
            "mfe_to_tp": (max_fav_to_decision / np.maximum(activation_ret, 1e-8)),
            "first_touch_mfe_norm": (max_fav_to_decision / np.maximum(barrier, 1e-8)),
            "first_touch_mae_norm": first_touch_mae_norm,
            "full_path_mae_to_sl": (full_path_max_adv / np.maximum(sl_ret, 1e-8)),
            "full_path_mfe_to_tp": (full_path_max_fav / np.maximum(activation_ret, 1e-8)),
            "full_path_mfe_norm": (full_path_max_fav / np.maximum(barrier, 1e-8)),
            "full_path_mae_norm": (full_path_max_adv / np.maximum(barrier, 1e-8)),
        },
        index=frame.index,
    )
    for col, values in path_order.items():
        out[col] = values
    for col in (
        "mae_to_sl",
        "mfe_to_tp",
        "first_touch_mfe_norm",
        "first_touch_mae_norm",
        "full_path_mae_to_sl",
        "full_path_mfe_to_tp",
        "full_path_mfe_norm",
        "full_path_mae_norm",
        "effective_trail_abs",
    ):
        out.loc[~np.isfinite(out[col]), col] = np.nan
    return out


def _seed_average_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
) -> tuple[pd.Series, float, float]:
    preds = [
        _fit_predict(x_train=x_train, y_train=y_train, w_train=w_train, x_valid=x_valid, seed=seed)
        for seed in seeds
    ]
    matrix = np.vstack(preds)
    pred = np.mean(matrix, axis=0).astype(np.float32)
    std = np.std(matrix, axis=0).astype(np.float32) if len(preds) > 1 else np.zeros_like(pred)
    return pd.Series(pred), float(np.mean(std)), float(np.percentile(std, 90))


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    features: list[str],
    first_touch_targets: dict[str, pd.DataFrame],
    aggregate_targets: dict[str, pd.DataFrame],
    month: str,
    arms: list[CaptureArm],
    top_fracs: list[float],
    selection_modes: list[str],
    seeds: list[int],
    train_lookback_months: int | None,
    max_weight: float,
    min_weight: float,
    regime_family: str | None = None,
    target_mode: str = "path_ordered",
    executable_cost_floor: float = EXECUTABLE_MARGIN_COST_FLOOR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior_months = sorted(month_period[train_mask].dropna().unique())
        train_mask = train_mask & month_period.isin(set(prior_months[-int(train_lookback_months) :]))
    valid_mask = month_period == month
    if regime_family and regime_family != "all":
        family = frame.get("__regime_family__", pd.Series("mixed", index=frame.index)).astype(str)
        family_mask = family.eq(str(regime_family))
        train_mask = train_mask & family_mask
        valid_mask = valid_mask & family_mask
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return [], [], [{
            "period": month,
            "regime_family": str(regime_family or "all"),
            "skipped": True,
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
        }]

    x_train, x_valid = _month_model_frame(frame, train_mask=train_mask, valid_mask=valid_mask, features=features)
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for arm in arms:
        target_full = first_touch_targets[arm.name]
        aggregate_full = aggregate_targets[arm.name]
        train_target = target_full.loc[train_mask].copy()
        valid_target = target_full.loc[valid_mask].copy().reset_index(drop=True)
        valid_aggregate = aggregate_full.loc[valid_mask].copy().reset_index(drop=True)
        weights = _weights_for_target(train_target, max_weight=max_weight, min_weight=min_weight)
        pred, seed_std_mean, seed_std_p90 = _seed_average_predict(
            x_train=x_train,
            y_train=train_target["target_soft"],
            w_train=weights,
            x_valid=x_valid,
            seeds=seeds,
        )
        pred = pred.reset_index(drop=True)
        score = _rank_pct(pred)
        score_ic_capture_net = _spearman(score, valid_target["capture_net"])
        score_ic_executable_margin = _spearman(score, valid_target.get("executable_margin", valid_target["capture_net"]))
        score_ic_hard = _spearman(score, valid_target["target_hard"])
        score_ic_aggregate_capture_net = _spearman(score, valid_aggregate["capture_net"])
        score_ic_u_policy_net = _spearman(score, valid_metrics["u_policy_net"])
        diagnostics.append(
            {
                "period": str(month),
                "arm": arm.name,
                "tp_r": arm.tp_r,
                "sl_r": arm.sl_r,
                "trail_r": float(getattr(arm, "trail_r", 0.50)),
                "max_bars_to_mfe": arm.max_bars_to_mfe,
                "max_barrier": arm.max_barrier,
                "regime_family": str(regime_family or "all"),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "train_valid_path_rate": _safe_mean(train_target["capture_valid_path"]),
                "valid_valid_path_rate": _safe_mean(valid_target["capture_valid_path"]),
                "train_hard_rate": _safe_mean(train_target["target_hard"]),
                "valid_hard_rate": _safe_mean(valid_target["target_hard"]),
                "aggregate_valid_hard_rate": _safe_mean(valid_aggregate["target_hard"]),
                "valid_first_touch_net_mean": _safe_mean(valid_target["capture_net"]),
                "valid_executable_margin_mean": _safe_mean(valid_target.get("executable_margin")),
                "valid_executable_margin_positive_rate": _safe_mean(
                    pd.to_numeric(valid_target.get("executable_margin", pd.Series(dtype=float)), errors="coerce").gt(0.0)
                ),
                "aggregate_valid_net_mean": _safe_mean(valid_aggregate["capture_net"]),
                "first_touch_minus_aggregate_net_mean": _safe_mean(
                    valid_target["capture_net"] - valid_aggregate["capture_net"]
                ),
                "valid_same_bar_both_rate": _safe_mean(valid_target["same_bar_both_hit"]),
                "valid_timeout_rate": _safe_mean(valid_target["capture_timeout"]),
                "weight_mean": _safe_mean(weights),
                "weight_p90": _safe_quantile(weights, 0.90),
                "weight_effective_n": _effective_sample_size(weights),
                "weight_effective_frac": _effective_sample_size(weights) / float(len(weights)) if len(weights) else float("nan"),
                "score_ic_capture_net": score_ic_capture_net,
                "score_ic_executable_margin": score_ic_executable_margin,
                "score_ic_hard": score_ic_hard,
                "score_ic_aggregate_capture_net": score_ic_aggregate_capture_net,
                "score_ic_u_policy_net": score_ic_u_policy_net,
                "seed_std_mean": seed_std_mean,
                "seed_std_p90": seed_std_p90,
            }
        )
        for selection_mode in selection_modes:
            for top_frac in top_fracs:
                row = _selection_metrics(
                    frame=valid,
                    metrics=valid_metrics,
                    target=valid_target,
                    score=score,
                    arm=arm.name,
                    period=str(month),
                    top_frac=float(top_frac),
                    selection_mode=selection_mode,
                )
                idx = (
                    _timestamp_top_indices(valid, score, top_frac)
                    if selection_mode == "timestamp"
                    else _rank_top_indices(score, top_frac)
                )
                selected_aggregate = valid_aggregate.iloc[idx] if len(idx) else valid_aggregate.iloc[:0]
                row.update(
                    {
                        "selector": f"first_touch_capture_proxy_{selection_mode}",
                        "selection_mode": selection_mode,
                        "tp_r": arm.tp_r,
                        "sl_r": arm.sl_r,
                        "trail_r": float(getattr(arm, "trail_r", 0.50)),
                        "max_bars_to_mfe": arm.max_bars_to_mfe,
                        "max_barrier": arm.max_barrier,
                        "regime_family": str(regime_family or "all"),
                        "aggregate_capture_net_mean": _safe_mean(selected_aggregate.get("capture_net")),
                        "aggregate_capture_hit_rate": _safe_mean(selected_aggregate.get("capture_hit")),
                        "first_touch_minus_aggregate_net_mean": _safe_mean(
                            valid_target.iloc[idx]["capture_net"].reset_index(drop=True)
                            - selected_aggregate["capture_net"].reset_index(drop=True)
                        )
                        if len(idx)
                        else float("nan"),
                        "first_touch_timeout_rate": _safe_mean(valid_target.iloc[idx].get("capture_timeout"))
                        if len(idx)
                        else float("nan"),
                        "first_touch_executable_margin_mean": _safe_mean(
                            valid_target.iloc[idx].get("executable_margin")
                        )
                        if len(idx)
                        else float("nan"),
                        "first_touch_executable_margin_positive_rate": _safe_mean(
                            pd.to_numeric(
                                valid_target.iloc[idx].get("executable_margin", pd.Series(dtype=float)),
                                errors="coerce",
                            ).gt(0.0)
                        )
                        if len(idx)
                        else float("nan"),
                        "first_touch_same_bar_both_rate": _safe_mean(valid_target.iloc[idx].get("same_bar_both_hit"))
                        if len(idx)
                        else float("nan"),
                        "score_ic_capture_net": score_ic_capture_net,
                        "score_ic_executable_margin": score_ic_executable_margin,
                        "score_ic_hard": score_ic_hard,
                    }
                )
                monthly_rows.append(row)
                for week_row in _weekly_rows(
                    frame=valid,
                    metrics=valid_metrics,
                    target=valid_target,
                    score=score,
                    arm=arm.name,
                    period=str(month),
                    top_frac=float(top_frac),
                    selection_mode=selection_mode,
                ):
                    week_row.update(
                        {
                            "selector": f"first_touch_capture_proxy_{selection_mode}",
                            "selection_mode": selection_mode,
                            "tp_r": arm.tp_r,
                            "sl_r": arm.sl_r,
                            "trail_r": float(getattr(arm, "trail_r", 0.50)),
                            "max_bars_to_mfe": arm.max_bars_to_mfe,
                            "max_barrier": arm.max_barrier,
                            "regime_family": str(regime_family or "all"),
                            "score_ic_capture_net": score_ic_capture_net,
                            "score_ic_executable_margin": score_ic_executable_margin,
                            "score_ic_hard": score_ic_hard,
                        }
                    )
                    weekly_rows.append(week_row)
    return monthly_rows, weekly_rows, diagnostics


def _timestamp_top_indices(frame: pd.DataFrame, score: pd.Series, top_frac: float) -> np.ndarray:
    score_series = _safe_numeric(score).reset_index(drop=True)
    timestamps = pd.to_datetime(frame["__ts__"], errors="coerce").reset_index(drop=True)
    chosen: list[np.ndarray] = []
    for _, ids in pd.Series(np.arange(len(score_series)), index=score_series.index).groupby(timestamps, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        valid_pos = pos[np.isfinite(score_series.iloc[pos].to_numpy(dtype=np.float64))]
        if len(valid_pos) == 0:
            continue
        k = max(1, int(math.ceil(float(top_frac) * len(valid_pos))))
        values = score_series.iloc[valid_pos].to_numpy(dtype=np.float64)
        order = np.argsort(-values, kind="mergesort")[:k]
        chosen.append(valid_pos[order].astype(np.int64, copy=False))
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.concatenate(chosen).astype(np.int64, copy=False)


def _write_markdown(
    output_dir: Path,
    fit_holdout: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_first_touch_capture_proxy.md"
    cols = [
        "arm",
        "regime_family",
        "selection_mode",
        "top_frac",
        "capture_proxy_score",
        "fit_sign_pass",
        "fit_bounded_pass",
        "fit_strict_pass",
        "fit_mean_capture_net",
        "fit_worst_capture_net",
        "fit_material_positive_week_rate",
        "fit_hit_rate",
        "fit_ev_weighted_first_touch_precision",
        "fit_ev_weighted_clean_precision",
        "fit_capture_gross_mean",
        "fit_stop_rate",
        "fit_selected_path_bad_mae_1r_rate",
        "fit_first_touch_bad_mae_1r_rate",
        "fit_first_touch_p90_mae_norm",
        "fit_target_full_path_bad_mae_1r_rate",
        "fit_effective_sl_abs_p90",
        "holdout_bounded_pass",
        "holdout_strict_pass",
        "holdout_mean_capture_net",
        "holdout_material_positive_week_rate",
        "holdout_q25_week_capture_net",
        "holdout_hit_rate",
        "holdout_ev_weighted_first_touch_precision",
        "holdout_ev_weighted_clean_precision",
        "holdout_capture_gross_mean",
        "holdout_first_touch_executable_margin_mean",
        "holdout_first_touch_executable_margin_positive_rate",
        "holdout_stop_rate",
        "holdout_selected_path_bad_mae_1r_rate",
        "holdout_first_touch_bad_mae_1r_rate",
        "holdout_first_touch_p90_mae_norm",
        "holdout_target_full_path_bad_mae_1r_rate",
        "holdout_effective_sl_abs_p90",
        "holdout_timeout_rate",
    ]
    diag_cols = [
        "period",
        "arm",
        "regime_family",
        "valid_valid_path_rate",
        "valid_hard_rate",
        "aggregate_valid_hard_rate",
        "valid_first_touch_net_mean",
        "valid_executable_margin_mean",
        "valid_executable_margin_positive_rate",
        "aggregate_valid_net_mean",
        "first_touch_minus_aggregate_net_mean",
        "valid_same_bar_both_rate",
        "score_ic_capture_net",
        "score_ic_executable_margin",
        "score_ic_hard",
        "score_ic_u_policy_net",
        "weight_effective_frac",
    ]
    strict = fit_holdout[fit_holdout["holdout_strict_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    bounded = fit_holdout[fit_holdout["holdout_bounded_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    positive_dirty = fit_holdout[fit_holdout["positive_dirty_holdout"].eq(True)] if not fit_holdout.empty else fit_holdout
    best = fit_holdout.sort_values("capture_proxy_score", ascending=False) if not fit_holdout.empty else fit_holdout
    lines = [
        "# First-Touch Capture Label Proxy",
        "",
        "Scope: proxy diagnostic only. No production base/meta training, Optuna, or policy geometry optimisation is run.",
        "",
        "The target is computed from production simple-policy 15m replay paths. TP/SL order is resolved bar-by-bar; if both touch in the same bar, the execution tie-breaker uses shortest distance from bar open and assigns exact ties to SL.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Path coverage: `{manifest['path_fetch']['finite_path_coverage']:.4f}`",
        f"Months: `{','.join(manifest['months'])}`",
        f"Train lookback months: `{manifest['train_lookback_months']}`",
        f"Outcome mode: `{manifest['outcome_mode']}`",
        f"Target mode: `{manifest['target_mode']}`",
        f"Executable cost floor: `{manifest['executable_cost_floor']}`",
        f"Regime families: `{','.join(manifest['regime_families'])}`",
        f"Selection modes: `{','.join(manifest['selection_modes'])}`",
        "",
        "## Counts",
        "",
        f"- Rows: `{manifest['rows']}`",
        f"- Monthly rows: `{manifest['rows_monthly']}`",
        f"- Weekly rows: `{manifest['rows_weekly']}`",
        f"- Fit bounded pass: `{manifest['fit_bounded_pass_rows']}`",
        f"- Holdout bounded pass after fit: `{manifest['holdout_bounded_pass_rows']}`",
        f"- Fit strict pass: `{manifest['fit_strict_pass_rows']}`",
        f"- Holdout strict pass after fit: `{manifest['holdout_strict_pass_rows']}`",
        f"- Positive but bounded-failing holdout: `{manifest['positive_dirty_holdout_rows']}`",
        "",
        "## Strict Passes",
        "",
        _format_table(strict, cols),
        "",
        "## Bounded Passes",
        "",
        _format_table(bounded, cols),
        "",
        "## Positive But Bounded-Failing",
        "",
        _format_table(positive_dirty, cols),
        "",
        "## Best Rejected Rows",
        "",
        _format_table(best, cols),
        "",
        "## Diagnostics",
        "",
        _format_table(diagnostics.sort_values(["period", "arm"]), diag_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Fit/Holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_proxy(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    arm_names: list[str],
    custom_arms: list[CaptureArm],
    only_custom_arms: bool,
    top_fracs: list[float],
    selection_modes: list[str],
    seeds: list[int],
    train_lookback_months: int | None,
    max_weight: float,
    min_weight: float,
    min_week_rows: int,
    data_root: Path,
    market_mode: str,
    exchange: str,
    side: str | None,
    path_len: int,
    apply_delayed_entry: bool,
    outcome_mode: str = "fixed_tp",
    regime_families: list[str] | None = None,
    round_trip_cost: float = ROUND_TRIP_COST,
    target_mode: str = "path_ordered",
    executable_cost_floor: float = EXECUTABLE_MARGIN_COST_FLOOR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    resolved_side = _infer_side(labels_path, side)
    _rows_exec, paths, path_fetch_stats = _fetch_policy_paths(
        frame,
        labels_path=labels_path,
        side=resolved_side,
        data_root=data_root,
        market_mode=market_mode,
        exchange=exchange,
        path_len=path_len,
        apply_delayed_entry=apply_delayed_entry,
    )
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    regime_report = _add_regime_family_columns(frame)
    metrics = _path_metrics(frame)
    features = _feature_columns(frame)
    arms = _resolve_arms(
        arm_names=arm_names,
        custom_arms=custom_arms,
        only_custom_arms=only_custom_arms,
    )
    first_touch_targets = {
        arm.name: _first_touch_capture_outcome(
            frame,
            paths,
            arm,
            side_name=resolved_side,
            outcome_mode=outcome_mode,
            round_trip_cost=float(round_trip_cost),
            target_mode=str(target_mode),
            executable_cost_floor=float(executable_cost_floor),
        )
        for arm in arms
    }
    aggregate_targets = {
        arm.name: _aggregate_capture_outcome(metrics, arm) for arm in arms
    }
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    requested_families = [str(v) for v in (regime_families or ["all"]) if str(v).strip()]
    if "auto" in requested_families:
        auto_families = sorted(str(v) for v in frame["__regime_family__"].dropna().astype(str).unique())
        requested_families = auto_families or ["all"]
    for regime_family in requested_families:
        for month in months:
            rows, weeks, diagnostics = _run_month(
                frame=frame,
                metrics=metrics,
                features=features,
                first_touch_targets=first_touch_targets,
                aggregate_targets=aggregate_targets,
                month=str(month),
                arms=arms,
                top_fracs=top_fracs,
                selection_modes=selection_modes,
                seeds=seeds,
                train_lookback_months=train_lookback_months,
                max_weight=max_weight,
                min_weight=min_weight,
                regime_family=str(regime_family),
                target_mode=str(target_mode),
                executable_cost_floor=float(executable_cost_floor),
            )
            monthly_rows.extend(rows)
            weekly_rows.extend(weeks)
            diagnostic_rows.extend(diagnostics)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    fit_holdout = _fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=["2026-04", "2026-05"],
        holdout_month="2026-06",
        min_week_rows=min_week_rows,
    )
    paths_out = {
        "monthly": output_dir / "label_first_touch_capture_proxy_monthly.csv",
        "weekly": output_dir / "label_first_touch_capture_proxy_weekly.csv",
        "diagnostics": output_dir / "label_first_touch_capture_proxy_diagnostics.csv",
        "fit_holdout": output_dir / "label_first_touch_capture_proxy_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths_out["monthly"], index=False)
    weekly.to_csv(paths_out["weekly"], index=False)
    diagnostics.to_csv(paths_out["diagnostics"], index=False)
    fit_holdout.to_csv(paths_out["fit_holdout"], index=False)
    manifest = {
        "scope": "first_touch_capture_proxy_not_full_policy_training",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_store": feature_store_report,
        "regime_family_report": regime_report,
        "feature_count": int(len(features)),
        "features": features,
        "path_fetch": path_fetch_stats,
        "months": [str(v) for v in months],
        "arms": [arm.__dict__ for arm in arms],
        "custom_arms": [arm.__dict__ for arm in custom_arms],
        "only_custom_arms": bool(only_custom_arms),
        "top_fracs": [float(v) for v in top_fracs],
        "selection_modes": [str(v) for v in selection_modes],
        "seeds": [int(v) for v in seeds],
        "train_lookback_months": int(train_lookback_months) if train_lookback_months is not None else None,
        "max_weight": float(max_weight),
        "min_weight": float(min_weight),
        "min_week_rows": int(min_week_rows),
        "data_root": str(data_root),
        "market_mode": str(market_mode),
        "exchange": str(exchange),
        "side": str(resolved_side),
        "path_len": int(path_len),
        "apply_delayed_entry": bool(apply_delayed_entry),
        "outcome_mode": str(outcome_mode),
        "target_mode": str(target_mode),
        "executable_cost_floor": float(executable_cost_floor),
        "regime_families": requested_families,
        "round_trip_cost": float(round_trip_cost),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "fit_bounded_pass_rows": int(fit_holdout["fit_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_bounded_pass_rows": int(fit_holdout["holdout_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "fit_strict_pass_rows": int(fit_holdout["fit_strict_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_strict_pass_rows": int(fit_holdout["holdout_strict_pass"].sum()) if not fit_holdout.empty else 0,
        "positive_dirty_holdout_rows": int(fit_holdout["positive_dirty_holdout"].sum()) if not fit_holdout.empty else 0,
        "outputs": {key: str(value) for key, value in paths_out.items()},
    }
    markdown = _write_markdown(output_dir, fit_holdout, diagnostics, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths_out["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--arms", default=",".join(arm.name for arm in CAPTURE_ARMS))
    parser.add_argument(
        "--custom-arms",
        default="",
        help="Semicolon-separated specs: name:tp_r:sl_r:max_bars_to_mfe:max_barrier.",
    )
    parser.add_argument("--only-custom-arms", action="store_true")
    parser.add_argument("--arm-grid-prefix", default="C0g")
    parser.add_argument("--arm-grid-tp-rs", default="")
    parser.add_argument("--arm-grid-sl-rs", default="")
    parser.add_argument("--arm-grid-trail-rs", default="0.50")
    parser.add_argument("--arm-grid-fast-bars", default="")
    parser.add_argument("--arm-grid-max-barriers", default="")
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--selection-modes", default="global,timestamp")
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--max-weight", type=float, default=12.0)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--side", choices=("long", "short"), default=None)
    parser.add_argument("--path-len", type=int, default=int(spo.DEFAULT_FORWARD_BARS))
    parser.add_argument("--no-delayed-entry", action="store_true")
    parser.add_argument("--outcome-mode", choices=("fixed_tp", "trailing_profit"), default="fixed_tp")
    parser.add_argument("--round-trip-cost", type=float, default=float(ROUND_TRIP_COST))
    parser.add_argument(
        "--target-mode",
        choices=("path_ordered", "executable_margin", "executable_margin_hybrid"),
        default="path_ordered",
        help=(
            "Soft-label target construction mode. executable_margin rewards rows that clear the executable "
            "cost floor; executable_margin_hybrid keeps path-ordered positives but adds a cost-floor bias."
        ),
    )
    parser.add_argument("--executable-cost-floor", type=float, default=float(EXECUTABLE_MARGIN_COST_FLOOR))
    parser.add_argument(
        "--regime-families",
        default="all",
        help="Comma-separated regime families. Use auto for observed pre-entry families.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid_arms = _build_grid_arms(
        tp_rs=_parse_float_csv(args.arm_grid_tp_rs, ()),
        sl_rs=_parse_float_csv(args.arm_grid_sl_rs, ()),
        trail_rs=_parse_float_csv(args.arm_grid_trail_rs, (0.50,)),
        fast_bars=_parse_float_csv(args.arm_grid_fast_bars, ()),
        max_barriers=_parse_float_csv(args.arm_grid_max_barriers, ()),
        prefix=str(args.arm_grid_prefix),
    )
    custom_arms = _parse_capture_arm_specs(args.custom_arms) + grid_arms
    if bool(args.only_custom_arms):
        arm_names = [arm.name for arm in custom_arms]
    else:
        arm_names = _parse_csv(args.arms, tuple(arm.name for arm in CAPTURE_ARMS))
    manifest = run_proxy(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        arm_names=arm_names,
        custom_arms=custom_arms,
        only_custom_arms=bool(args.only_custom_arms),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        selection_modes=_parse_csv(args.selection_modes, ("global", "timestamp")),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        max_weight=float(args.max_weight),
        min_weight=float(args.min_weight),
        min_week_rows=int(args.min_week_rows),
        data_root=args.data_root,
        market_mode=str(args.market_mode),
        exchange=str(args.exchange),
        side=args.side,
        path_len=int(args.path_len),
        apply_delayed_entry=not bool(args.no_delayed_entry),
        outcome_mode=str(args.outcome_mode),
        regime_families=_parse_csv(args.regime_families, ("all",)),
        round_trip_cost=float(args.round_trip_cost),
        target_mode=str(args.target_mode),
        executable_cost_floor=float(args.executable_cost_floor),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
