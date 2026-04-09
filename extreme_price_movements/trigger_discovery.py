from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from extreme_price_movements.intraday_crypto_library import (
    INTRADAY_TRIGGER_COLUMNS,
    build_intraday_crypto_library,
    trigger_family_from_column,
)
from extreme_price_movements.utils import tprint


MetricFn = Callable[[str, np.ndarray, Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any], float, float], Dict[str, float]]


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0))))


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _json_dumps_sorted(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _event_symbol_summary(
    event_mask: np.ndarray,
    symbol_codes: np.ndarray,
    max_symbols: int = 5,
) -> Dict[str, Any]:
    codes = np.asarray(symbol_codes[event_mask], dtype=np.int32)
    if codes.size == 0:
        return {"event_symbol_count": 0, "top_symbol_share": 0.0, "top_symbol_codes_text": ""}
    uniq_codes, counts = np.unique(codes, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = ", ".join(
        f"{int(uniq_codes[idx])}:{int(counts[idx])}"
        for idx in order[: max(1, int(max_symbols))]
    )
    return {
        "event_symbol_count": int(uniq_codes.size),
        "top_symbol_share": float(np.max(counts) / max(codes.size, 1)),
        "top_symbol_codes_text": top,
    }


def _append_trigger_symbol_log(line: str) -> None:
    try:
        path = Path("reports")
        path.mkdir(parents=True, exist_ok=True)
        with (path / "mask_opt_symbol_concentration.log").open("a", encoding="utf-8") as fh:
            fh.write(line.rstrip() + "\n")
    except Exception:
        pass


@dataclass(frozen=True)
class TriggerDiscoveryConfig:
    enabled: bool = True
    max_parent_regimes: int = 20
    top_k_triggers_per_regime: int = 3
    random_seed: int = 42
    min_trigger_events: int = 150
    min_trigger_active_days_fraction: float = 0.15
    min_fold_events: int = 10
    min_trigger_support_ratio: float = 0.08
    min_trigger_distinct_symbols: int = 6
    max_trigger_top_symbol_share: float = 0.40
    trigger_timing_horizon_bars: int = 24
    trigger_edge_horizon_1h_bars: int = 4
    trigger_edge_horizon_3h_bars: int = 12
    trigger_score_threshold: float = 0.0
    enable_pullback_recovery: bool = True
    enable_breakout: bool = True
    enable_sweep_reversal: bool = True

    enable_ema_reclaim_touch: bool = True
    enable_simple_close_breakout: bool = True
    enable_expansion_bar_triggers: bool = True
    enable_impulse_bar_triggers: bool = True
    enable_relaxed_sweep_triggers: bool = True
    enable_compression_release_triggers: bool = False
    breakout_lookbacks: Tuple[int, ...] = (5, 10, 20)
    reclaim_ema_lens: Tuple[int, ...] = (10, 20, 30)
    wick_thresholds: Tuple[float, ...] = (0.4, 0.6)
    body_ratio_thresholds: Tuple[float, ...] = (0.4, 0.6, 0.7)
    close_location_thresholds: Tuple[float, ...] = (0.7, 0.8, 0.9)
    compression_ratio_thresholds: Tuple[float, ...] = (0.5, 0.6, 0.7)
    range_atr_thresholds: Tuple[float, ...] = (1.2, 1.5)
    distance_to_ema_thresholds: Tuple[float, ...] = (1.0, 1.5, 2.0)
    w_edge: float = 1.5
    w_edge_1h: float = 0.5
    w_edge_3h: float = 1.0
    w_stability: float = 0.8
    w_pred: float = 0.8
    w_timing: float = 1.3
    w_disp: float = 0.9
    w_parent: float = 0.8
    w_covloss: float = 0.7
    cheap_prescreen_keep_fraction: float = 0.5
    ridge_prescreen_keep_fraction: float = 0.25
    ridge_prescreen_max_templates_per_parent: int = 4
    ridge_prescreen_alpha: float = 1.0
    apply_non_dominance: bool = True
    keep_family_diversity: bool = True
    max_triggers_per_family_per_parent: int = 2

    @classmethod
    def from_mapping(cls, cfg: Dict[str, Any]) -> "TriggerDiscoveryConfig":
        return cls(
            enabled=bool(cfg.get("enable_trigger_discovery_stage", True)),
            max_parent_regimes=int(cfg.get("trigger_max_parent_regimes", cfg.get("max_parent_regimes", 20))),
            top_k_triggers_per_regime=int(cfg.get("top_k_triggers_per_regime", 3)),
            random_seed=int(cfg.get("random_seed", 42)),
            min_trigger_events=int(cfg.get("min_trigger_events", 150)),
            min_trigger_active_days_fraction=float(cfg.get("min_trigger_active_days_fraction", 0.15)),
            min_fold_events=int(cfg.get("min_fold_events", 10)),
            min_trigger_support_ratio=float(cfg.get("min_trigger_support_ratio", 0.08)),
            min_trigger_distinct_symbols=int(cfg.get("trigger_min_distinct_symbols", 6)),
            max_trigger_top_symbol_share=float(cfg.get("trigger_max_top_symbol_share", 0.40)),
            trigger_timing_horizon_bars=int(cfg.get("trigger_timing_horizon_bars", 24)),
            trigger_edge_horizon_1h_bars=int(cfg.get("trigger_edge_horizon_1h_bars", 4)),
            trigger_edge_horizon_3h_bars=int(cfg.get("trigger_edge_horizon_3h_bars", 12)),
            trigger_score_threshold=float(cfg.get("trigger_score_threshold", 0.0)),
            enable_pullback_recovery=bool(cfg.get("enable_pullback_recovery", True)),
            enable_breakout=bool(cfg.get("enable_breakout", True)),
            enable_sweep_reversal=bool(cfg.get("enable_sweep_reversal", True)),

            enable_ema_reclaim_touch=bool(cfg.get("enable_ema_reclaim_touch", True)),
            enable_simple_close_breakout=bool(cfg.get("enable_simple_close_breakout", True)),
            enable_expansion_bar_triggers=bool(cfg.get("enable_expansion_bar_triggers", True)),
            enable_impulse_bar_triggers=bool(cfg.get("enable_impulse_bar_triggers", True)),
            enable_relaxed_sweep_triggers=bool(cfg.get("enable_relaxed_sweep_triggers", True)),
            enable_compression_release_triggers=bool(cfg.get("enable_compression_release_triggers", cfg.get("enable_compression_release", False))),
            breakout_lookbacks=tuple(cfg.get("breakout_lookbacks", (5, 10, 20))),
            reclaim_ema_lens=tuple(cfg.get("reclaim_ema_lens", (10, 20, 30))),
            wick_thresholds=tuple(cfg.get("wick_thresholds", (0.4, 0.6))),
            body_ratio_thresholds=tuple(cfg.get("body_ratio_thresholds", (0.4, 0.6, 0.7))),
            close_location_thresholds=tuple(cfg.get("close_location_thresholds", (0.7, 0.8, 0.9))),
            compression_ratio_thresholds=tuple(cfg.get("compression_ratio_thresholds", (0.5, 0.6, 0.7))),
            range_atr_thresholds=tuple(cfg.get("range_atr_thresholds", (1.2, 1.5))),
            distance_to_ema_thresholds=tuple(cfg.get("distance_to_ema_thresholds", (1.0, 1.5, 2.0))),
            w_edge=float(cfg.get("trigger_w_edge", 1.5)),
            w_edge_1h=float(cfg.get("trigger_w_edge_1h", 0.5)),
            w_edge_3h=float(cfg.get("trigger_w_edge_3h", 1.0)),
            w_stability=float(cfg.get("trigger_w_stability", 0.8)),
            w_pred=float(cfg.get("trigger_w_pred", 0.8)),
            w_timing=float(cfg.get("trigger_w_timing", 1.3)),
            w_disp=float(cfg.get("trigger_w_disp", 0.9)),
            w_parent=float(cfg.get("trigger_w_parent", 0.8)),
            w_covloss=float(cfg.get("trigger_w_covloss", 0.7)),
            cheap_prescreen_keep_fraction=float(cfg.get("trigger_cheap_prescreen_keep_fraction", 0.5)),
            ridge_prescreen_keep_fraction=float(cfg.get("trigger_ridge_prescreen_keep_fraction", 0.25)),
            ridge_prescreen_max_templates_per_parent=int(cfg.get("trigger_ridge_prescreen_max_templates_per_parent", 4)),
            ridge_prescreen_alpha=float(cfg.get("trigger_ridge_prescreen_alpha", 1.0)),
            apply_non_dominance=bool(cfg.get("apply_non_dominance", True)),
            keep_family_diversity=bool(cfg.get("keep_family_diversity", True)),
            max_triggers_per_family_per_parent=int(cfg.get("max_triggers_per_family_per_parent", 2)),
        )


@dataclass(frozen=True)
class TriggerTemplate:
    trigger_family: str
    trigger_template_name: str
    params: Dict[str, Any]
    trigger_direction: str
    trigger_anchor_feature: str
    definition: str
    complexity_tier: int = 0

    @property
    def trigger_params_json(self) -> str:
        return _json_dumps_sorted(self.params)

    @property
    def trigger_id(self) -> str:
        raw = f"{self.trigger_template_name}|{self.trigger_params_json}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def current_trigger_feature_inventory() -> List[Dict[str, Any]]:
    return [
        {"feature_name": "open", "source_function": "build_trigger_feature_frame", "formula_or_description": "raw open", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "high", "source_function": "build_trigger_feature_frame", "formula_or_description": "raw high", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "low", "source_function": "build_trigger_feature_frame", "formula_or_description": "raw low", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "close", "source_function": "build_trigger_feature_frame", "formula_or_description": "raw close", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "volume", "source_function": "build_trigger_feature_frame", "formula_or_description": "raw volume", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "range", "source_function": "build_trigger_feature_frame", "formula_or_description": "high - low", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "true_range", "source_function": "build_trigger_feature_frame", "formula_or_description": "max(high-low, |high-prev_close|, |low-prev_close|)", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "atr_14", "source_function": "build_trigger_feature_frame", "formula_or_description": "EWMA true range alpha=1/14", "parameterization": 14, "dimensionless": False, "normalized": False},
        {"feature_name": "atr_100", "source_function": "build_trigger_feature_frame", "formula_or_description": "EWMA true range alpha=1/100", "parameterization": 100, "dimensionless": False, "normalized": False},
        {"feature_name": "range_atr", "source_function": "build_trigger_feature_frame", "formula_or_description": "range / atr_14", "parameterization": 14, "dimensionless": True, "normalized": True},
        {"feature_name": "compression_ratio", "source_function": "build_trigger_feature_frame", "formula_or_description": "atr_14 / atr_100", "parameterization": "14/100", "dimensionless": True, "normalized": True},
        {"feature_name": "rolling_range_5", "source_function": "build_trigger_feature_frame", "formula_or_description": "shifted rolling max(high,5) - shifted rolling min(low,5)", "parameterization": 5, "dimensionless": False, "normalized": False},
        {"feature_name": "rolling_range_10", "source_function": "build_trigger_feature_frame", "formula_or_description": "shifted rolling max(high,10) - shifted rolling min(low,10)", "parameterization": 10, "dimensionless": False, "normalized": False},
        {"feature_name": "rolling_range_20", "source_function": "build_trigger_feature_frame", "formula_or_description": "shifted rolling max(high,20) - shifted rolling min(low,20)", "parameterization": 20, "dimensionless": False, "normalized": False},
        {"feature_name": "body", "source_function": "build_trigger_feature_frame", "formula_or_description": "abs(close-open)", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "body_ratio", "source_function": "build_trigger_feature_frame", "formula_or_description": "abs(close-open)/range", "parameterization": None, "dimensionless": True, "normalized": True},
        {"feature_name": "upper_wick", "source_function": "build_trigger_feature_frame", "formula_or_description": "high - max(open, close)", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "lower_wick", "source_function": "build_trigger_feature_frame", "formula_or_description": "min(open, close) - low", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "upper_wick_ratio", "source_function": "build_trigger_feature_frame", "formula_or_description": "upper_wick / range", "parameterization": None, "dimensionless": True, "normalized": True},
        {"feature_name": "lower_wick_ratio", "source_function": "build_trigger_feature_frame", "formula_or_description": "lower_wick / range", "parameterization": None, "dimensionless": True, "normalized": True},
        {"feature_name": "close_location_in_bar", "source_function": "build_trigger_feature_frame", "formula_or_description": "(close - low) / range", "parameterization": None, "dimensionless": True, "normalized": True},
        {"feature_name": "open_location_in_bar", "source_function": "build_trigger_feature_frame", "formula_or_description": "(open - low) / range", "parameterization": None, "dimensionless": True, "normalized": True},
        {"feature_name": "signed_body_ratio", "source_function": "build_trigger_feature_frame", "formula_or_description": "(close-open) / range", "parameterization": None, "dimensionless": True, "normalized": True},
        {"feature_name": "ema_10", "source_function": "build_trigger_feature_frame", "formula_or_description": "EMA(close,10)", "parameterization": 10, "dimensionless": False, "normalized": False},
        {"feature_name": "ema_20", "source_function": "build_trigger_feature_frame", "formula_or_description": "EMA(close,20)", "parameterization": 20, "dimensionless": False, "normalized": False},
        {"feature_name": "ema_30", "source_function": "build_trigger_feature_frame", "formula_or_description": "EMA(close,30)", "parameterization": 30, "dimensionless": False, "normalized": False},
        {"feature_name": "ema_50", "source_function": "build_trigger_feature_frame", "formula_or_description": "EMA(close,50)", "parameterization": 50, "dimensionless": False, "normalized": False},
        {"feature_name": "ema_slope_ema20_3", "source_function": "build_trigger_feature_frame", "formula_or_description": "ema_20 - ema_20.shift(3)", "parameterization": 3, "dimensionless": False, "normalized": False},
        {"feature_name": "ema_slope_ema20_5", "source_function": "build_trigger_feature_frame", "formula_or_description": "ema_20 - ema_20.shift(5)", "parameterization": 5, "dimensionless": False, "normalized": False},
        {"feature_name": "distance_to_ema10", "source_function": "build_trigger_feature_frame", "formula_or_description": "close - ema_10", "parameterization": 10, "dimensionless": False, "normalized": False},
        {"feature_name": "distance_to_ema20", "source_function": "build_trigger_feature_frame", "formula_or_description": "close - ema_20", "parameterization": 20, "dimensionless": False, "normalized": False},
        {"feature_name": "distance_to_ema30", "source_function": "build_trigger_feature_frame", "formula_or_description": "close - ema_30", "parameterization": 30, "dimensionless": False, "normalized": False},
        {"feature_name": "distance_to_ema20_atr", "source_function": "build_trigger_feature_frame", "formula_or_description": "(close - ema_20) / atr_14", "parameterization": 20, "dimensionless": True, "normalized": True},
        {"feature_name": "distance_to_ema50_atr", "source_function": "build_trigger_feature_frame", "formula_or_description": "(close - ema_50) / atr_14", "parameterization": 50, "dimensionless": True, "normalized": True},
        {"feature_name": "trend_alignment_ema20_gt_ema50", "source_function": "build_trigger_feature_frame", "formula_or_description": "ema_20 > ema_50", "parameterization": "20/50", "dimensionless": True, "normalized": False},
        {"feature_name": "returns_1", "source_function": "build_trigger_feature_frame", "formula_or_description": "close.pct_change(1)", "parameterization": 1, "dimensionless": True, "normalized": True},
        {"feature_name": "returns_3", "source_function": "build_trigger_feature_frame", "formula_or_description": "close.pct_change(3)", "parameterization": 3, "dimensionless": True, "normalized": True},
        {"feature_name": "returns_5", "source_function": "build_trigger_feature_frame", "formula_or_description": "close.pct_change(5)", "parameterization": 5, "dimensionless": True, "normalized": True},
        {"feature_name": "returns_10", "source_function": "build_trigger_feature_frame", "formula_or_description": "close.pct_change(10)", "parameterization": 10, "dimensionless": True, "normalized": True},
        {"feature_name": "acceleration_close", "source_function": "build_trigger_feature_frame", "formula_or_description": "close - 2*close.shift(1) + close.shift(2)", "parameterization": None, "dimensionless": False, "normalized": False},
        {"feature_name": "acceleration_close_atr", "source_function": "build_trigger_feature_frame", "formula_or_description": "acceleration_close / atr_14", "parameterization": None, "dimensionless": True, "normalized": True},
        {"feature_name": "volume_ma_20", "source_function": "build_trigger_feature_frame", "formula_or_description": "rolling mean(volume,20)", "parameterization": 20, "dimensionless": False, "normalized": False},
        {"feature_name": "volume_spike", "source_function": "build_trigger_feature_frame", "formula_or_description": "volume / volume_ma_20", "parameterization": 20, "dimensionless": True, "normalized": True},
        {"feature_name": "rolling_high_5", "source_function": "build_trigger_feature_frame", "formula_or_description": "shifted rolling max(high,5)", "parameterization": 5, "dimensionless": False, "normalized": False},
        {"feature_name": "rolling_high_10", "source_function": "build_trigger_feature_frame", "formula_or_description": "shifted rolling max(high,10)", "parameterization": 10, "dimensionless": False, "normalized": False},
        {"feature_name": "rolling_high_20", "source_function": "build_trigger_feature_frame", "formula_or_description": "shifted rolling max(high,20)", "parameterization": 20, "dimensionless": False, "normalized": False},
        {"feature_name": "rolling_low_5", "source_function": "build_trigger_feature_frame", "formula_or_description": "shifted rolling min(low,5)", "parameterization": 5, "dimensionless": False, "normalized": False},
        {"feature_name": "rolling_low_10", "source_function": "build_trigger_feature_frame", "formula_or_description": "shifted rolling min(low,10)", "parameterization": 10, "dimensionless": False, "normalized": False},
        {"feature_name": "rolling_low_20", "source_function": "build_trigger_feature_frame", "formula_or_description": "shifted rolling min(low,20)", "parameterization": 20, "dimensionless": False, "normalized": False},
        {"feature_name": "close_gt_rolling_high_5", "source_function": "build_trigger_feature_frame", "formula_or_description": "close > shifted rolling high 5", "parameterization": 5, "dimensionless": True, "normalized": False},
        {"feature_name": "close_lt_rolling_low_5", "source_function": "build_trigger_feature_frame", "formula_or_description": "close < shifted rolling low 5", "parameterization": 5, "dimensionless": True, "normalized": False},
        {"feature_name": "high_gt_rolling_high_5", "source_function": "build_trigger_feature_frame", "formula_or_description": "high > shifted rolling high 5", "parameterization": 5, "dimensionless": True, "normalized": False},
        {"feature_name": "low_lt_rolling_low_5", "source_function": "build_trigger_feature_frame", "formula_or_description": "low < shifted rolling low 5", "parameterization": 5, "dimensionless": True, "normalized": False},
        {"feature_name": "bullish_bar", "source_function": "build_trigger_feature_frame", "formula_or_description": "close > open", "parameterization": None, "dimensionless": True, "normalized": False},
        {"feature_name": "bearish_bar", "source_function": "build_trigger_feature_frame", "formula_or_description": "close < open", "parameterization": None, "dimensionless": True, "normalized": False},
        {"feature_name": "prior_bullish_bar", "source_function": "build_trigger_feature_frame", "formula_or_description": "bullish_bar shifted by 1", "parameterization": None, "dimensionless": True, "normalized": False},
        {"feature_name": "prior_bearish_bar", "source_function": "build_trigger_feature_frame", "formula_or_description": "bearish_bar shifted by 1", "parameterization": None, "dimensionless": True, "normalized": False},
        {"feature_name": "inside_bar", "source_function": "build_trigger_feature_frame", "formula_or_description": "high <= prev_high and low >= prev_low", "parameterization": None, "dimensionless": True, "normalized": False},
        {"feature_name": "outside_bar", "source_function": "build_trigger_feature_frame", "formula_or_description": "high >= prev_high and low <= prev_low", "parameterization": None, "dimensionless": True, "normalized": False},
    ]


def current_trigger_template_inventory(config: Optional[TriggerDiscoveryConfig] = None) -> List[Dict[str, Any]]:
    templates = generate_trigger_templates(config or TriggerDiscoveryConfig())
    out: List[Dict[str, Any]] = []
    for template in templates:
        out.append(
            {
                "trigger_family": template.trigger_family,
                "trigger_name": template.trigger_template_name,
                "source_function": "generate_trigger_templates",
                "params": dict(template.params),
                "long_short_supported": True,
                "semantic_description": template.definition,
            }
        )
    return out


def _rolling_shifted_max(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).max().shift(1)


def _rolling_shifted_min(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).min().shift(1)


def _format_timestamp_bounds(timestamps: np.ndarray) -> str:
    ts_arr = np.asarray(timestamps)
    if ts_arr.size == 0:
        return "n/a"
    ts = pd.to_datetime(ts_arr, unit="s", utc=True, errors="coerce")
    if np.all(pd.isna(ts)):
        ts = pd.to_datetime(ts_arr, utc=True, errors="coerce")
    valid = pd.DatetimeIndex(ts).dropna()
    if valid.empty:
        return "n/a"
    return f"{valid[0].isoformat()} -> {valid[-1].isoformat()}"


def _event_run_duration_stats(
    event_mask: np.ndarray,
    asset_groups: Dict[int, np.ndarray],
    bars_per_hour: int,
) -> Dict[str, float]:
    run_lengths: List[int] = []
    mask_bool = np.asarray(event_mask, dtype=bool)
    for idxs in asset_groups.values():
        local = mask_bool[idxs]
        if local.size == 0 or not np.any(local):
            continue
        padded = np.concatenate(
            [np.asarray([False]), local.astype(bool, copy=False), np.asarray([False])]
        )
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        if starts.size and ends.size:
            run_lengths.extend((ends - starts).astype(int).tolist())
    if not run_lengths:
        return {
            "avg_event_duration_bars": 0.0,
            "median_event_duration_bars": 0.0,
            "avg_event_duration_hours": 0.0,
            "median_event_duration_hours": 0.0,
            "event_run_count": 0.0,
        }
    runs = np.asarray(run_lengths, dtype=np.float32)
    bph = max(int(bars_per_hour), 1)
    return {
        "avg_event_duration_bars": float(np.mean(runs)),
        "median_event_duration_bars": float(np.median(runs)),
        "avg_event_duration_hours": float(np.mean(runs) / bph),
        "median_event_duration_hours": float(np.median(runs) / bph),
        "event_run_count": float(runs.size),
    }


def _tprint_trigger_table_support_summary(stage: str, mode: str, df: pd.DataFrame) -> None:
    if df.empty:
        tprint(f"{stage} ({mode}) support: no trigger candidates")
        return

    def _fmt(col: str) -> str:
        vals = pd.to_numeric(df.get(col), errors="coerce").dropna()
        if vals.empty:
            return "n/a"
        precision = 1
        if col == "top_symbol_share":
            precision = 3
        elif col == "avg_event_duration_hours":
            precision = 2
        return (
            f"{vals.min():.{precision}f}/"
            f"{vals.median():.{precision}f}/"
            f"{vals.max():.{precision}f}"
        )

    tprint(
        f"{stage} ({mode}) support: "
        f"candidates={len(df)} | "
        f"events(min/med/max)={_fmt('total_events')} | "
        f"symbols(min/med/max)={_fmt('event_symbol_count')} | "
        f"top_share(min/med/max)={_fmt('top_symbol_share')} | "
        f"avg_duration_h(min/med/max)={_fmt('avg_event_duration_hours')}"
        + (
            f" | keep_pct_vs_parent(min/med/max)="
            f"{_fmt('keep_pct_vs_parent')}"
            if "keep_pct_vs_parent" in df.columns
            else ""
        )
    )


def build_trigger_feature_frame(
    shared: Dict[str, Any],
    asset_groups: Dict[int, np.ndarray],
    feature_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    n = int(np.asarray(shared["close"]).shape[0])
    bars_per_hour = int(shared.get("bph", shared.get("bars_per_hour", 4)))
    fwd_1h_bars = max(bars_per_hour, 1)
    fwd_3h_bars = max(3 * bars_per_hour, 1)
    eps = np.float32(1e-6)
    feature_frame: Dict[str, np.ndarray] = {
        "open": np.asarray(shared["open"], dtype=np.float32),
        "high": np.asarray(shared["high"], dtype=np.float32),
        "low": np.asarray(shared["low"], dtype=np.float32),
        "close": np.asarray(shared["close"], dtype=np.float32),
        "volume": np.asarray(shared.get("volume", np.ones(n)), dtype=np.float32),
        "day_ids": np.asarray(shared["day_ids"], dtype=np.int32),
        "symbol_codes": np.asarray(shared["symbol_codes"], dtype=np.int32),
        "timestamps": np.asarray(shared["timestamps"]),
        "_bars_per_hour": np.int32(bars_per_hour),
        "range": np.full(n, np.nan, dtype=np.float32),
        "true_range": np.full(n, np.nan, dtype=np.float32),
        "atr_14": np.full(n, np.nan, dtype=np.float32),
        "atr_100": np.full(n, np.nan, dtype=np.float32),
        "body": np.full(n, np.nan, dtype=np.float32),
        "body_ratio": np.full(n, np.nan, dtype=np.float32),
        "signed_body_ratio": np.full(n, np.nan, dtype=np.float32),
        "upper_wick": np.full(n, np.nan, dtype=np.float32),
        "lower_wick": np.full(n, np.nan, dtype=np.float32),
        "upper_wick_ratio": np.full(n, np.nan, dtype=np.float32),
        "lower_wick_ratio": np.full(n, np.nan, dtype=np.float32),
        "close_location_in_bar": np.full(n, np.nan, dtype=np.float32),
        "open_location_in_bar": np.full(n, np.nan, dtype=np.float32),
        "range_atr": np.full(n, np.nan, dtype=np.float32),
        "compression_ratio": np.full(n, np.nan, dtype=np.float32),
        "volume_ma_20": np.full(n, np.nan, dtype=np.float32),
        "volume_spike": np.full(n, np.nan, dtype=np.float32),
        "bullish_bar": np.zeros(n, dtype=bool),
        "bearish_bar": np.zeros(n, dtype=bool),
        "prior_bearish_bar": np.zeros(n, dtype=bool),
        "prior_bullish_bar": np.zeros(n, dtype=bool),
        "inside_bar": np.zeros(n, dtype=bool),
        "outside_bar": np.zeros(n, dtype=bool),
        "prev_bearish": np.zeros(n, dtype=bool),
        "prev_bullish": np.zeros(n, dtype=bool),
        "returns_1": np.full(n, np.nan, dtype=np.float32),
        "returns_3": np.full(n, np.nan, dtype=np.float32),
        "returns_5": np.full(n, np.nan, dtype=np.float32),
        "returns_10": np.full(n, np.nan, dtype=np.float32),
        "forward_return_1h": np.full(n, np.nan, dtype=np.float32),
        "forward_return_3h": np.full(n, np.nan, dtype=np.float32),
        "acceleration_close": np.full(n, np.nan, dtype=np.float32),
        "acceleration_close_atr": np.full(n, np.nan, dtype=np.float32),
        "trend_alignment_ema20_gt_ema50": np.zeros(n, dtype=bool),
        "_template_mask_cache": {},
    }
    for column_name in INTRADAY_TRIGGER_COLUMNS:
        feature_frame[column_name] = np.zeros(n, dtype=bool)

    for lens in (10, 20, 30, 50):
        feature_frame[f"ema_{lens}"] = np.full(n, np.nan, dtype=np.float32)
        feature_frame[f"distance_to_ema_{lens}"] = np.full(n, np.nan, dtype=np.float32)
        feature_frame[f"distance_to_ema_atr_{lens}"] = np.full(n, np.nan, dtype=np.float32)
    feature_frame["distance_to_ema20_atr"] = np.full(n, np.nan, dtype=np.float32)
    feature_frame["distance_to_ema50_atr"] = np.full(n, np.nan, dtype=np.float32)
    for lookback in (5, 10, 15, 20):
        feature_frame[f"rolling_high_{lookback}"] = np.full(n, np.nan, dtype=np.float32)
        feature_frame[f"rolling_low_{lookback}"] = np.full(n, np.nan, dtype=np.float32)
    for lookback in (5, 10, 20):
        feature_frame[f"rolling_range_{lookback}"] = np.full(n, np.nan, dtype=np.float32)
        feature_frame[f"close_gt_rolling_high_{lookback}"] = np.zeros(n, dtype=bool)
        feature_frame[f"close_lt_rolling_low_{lookback}"] = np.zeros(n, dtype=bool)
        feature_frame[f"high_gt_rolling_high_{lookback}"] = np.zeros(n, dtype=bool)
        feature_frame[f"low_lt_rolling_low_{lookback}"] = np.zeros(n, dtype=bool)
    for slope_bars in (3, 5):
        feature_frame[f"ema_slope_ema20_{slope_bars}"] = np.full(n, np.nan, dtype=np.float32)

    persisted_trigger_cols = {
        name for name in INTRADAY_TRIGGER_COLUMNS if feature_dict is not None and name in feature_dict
    }
    use_persisted_trigger_cols = False
    if persisted_trigger_cols and feature_dict is not None:
        persisted_finite = 0
        persisted_positive = 0
        for column_name in persisted_trigger_cols:
            arr = np.asarray(feature_dict[column_name], dtype=np.float32)
            finite = np.isfinite(arr)
            persisted_finite += int(np.sum(finite))
            persisted_positive += int(np.sum(arr[finite] > 0.0))
        use_persisted_trigger_cols = persisted_finite > 0 and persisted_positive > 0
        if not use_persisted_trigger_cols:
            tprint(
                "Persisted trigger library unavailable for this sample; "
                "rebuilding trigger columns from OHLCV because cached arrays are empty or all-NaN."
            )

    for idxs in asset_groups.values():
        o = pd.Series(feature_frame["open"][idxs], copy=False)
        h = pd.Series(feature_frame["high"][idxs], copy=False)
        l = pd.Series(feature_frame["low"][idxs], copy=False)
        c = pd.Series(feature_frame["close"][idxs], copy=False)
        v = pd.Series(feature_frame["volume"][idxs], copy=False)
        prev_close = c.shift(1)
        true_range = pd.concat(
            [(h - l), (h - prev_close).abs(), (l - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr14 = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
        atr100 = true_range.ewm(alpha=1.0 / 100.0, adjust=False, min_periods=100).mean()
        bar_range = (h - l).clip(lower=eps)
        signed_body = c - o
        body = signed_body.abs()
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
        volume_ma20 = v.rolling(20, min_periods=20).mean()
        prev_high = h.shift(1)
        prev_low = l.shift(1)

        feature_frame["range"][idxs] = bar_range.to_numpy(dtype=np.float32)
        feature_frame["true_range"][idxs] = true_range.to_numpy(dtype=np.float32)
        feature_frame["atr_14"][idxs] = atr14.to_numpy(dtype=np.float32)
        feature_frame["atr_100"][idxs] = atr100.to_numpy(dtype=np.float32)
        feature_frame["body"][idxs] = body.to_numpy(dtype=np.float32)
        feature_frame["body_ratio"][idxs] = (body / bar_range).to_numpy(dtype=np.float32)
        feature_frame["signed_body_ratio"][idxs] = (signed_body / bar_range).to_numpy(dtype=np.float32)
        feature_frame["upper_wick"][idxs] = upper_wick.to_numpy(dtype=np.float32)
        feature_frame["lower_wick"][idxs] = lower_wick.to_numpy(dtype=np.float32)
        feature_frame["upper_wick_ratio"][idxs] = (upper_wick / bar_range).to_numpy(dtype=np.float32)
        feature_frame["lower_wick_ratio"][idxs] = (lower_wick / bar_range).to_numpy(dtype=np.float32)
        feature_frame["close_location_in_bar"][idxs] = (
            (c - l) / bar_range
        ).to_numpy(dtype=np.float32)
        feature_frame["open_location_in_bar"][idxs] = ((o - l) / bar_range).to_numpy(dtype=np.float32)
        feature_frame["range_atr"][idxs] = ((h - l) / atr14.clip(lower=eps)).to_numpy(dtype=np.float32)
        feature_frame["compression_ratio"][idxs] = (
            atr14 / atr100.clip(lower=eps)
        ).to_numpy(dtype=np.float32)
        feature_frame["volume_ma_20"][idxs] = volume_ma20.to_numpy(dtype=np.float32)
        feature_frame["volume_spike"][idxs] = (v / volume_ma20.clip(lower=eps)).to_numpy(dtype=np.float32)
        feature_frame["bullish_bar"][idxs] = (c > o).fillna(False).to_numpy(dtype=bool)
        feature_frame["bearish_bar"][idxs] = (c < o).fillna(False).to_numpy(dtype=bool)
        feature_frame["prev_bearish"][idxs] = (c.shift(1) < o.shift(1)).fillna(False).to_numpy(dtype=bool)
        feature_frame["prev_bullish"][idxs] = (c.shift(1) > o.shift(1)).fillna(False).to_numpy(dtype=bool)
        feature_frame["prior_bearish_bar"][idxs] = feature_frame["prev_bearish"][idxs]
        feature_frame["prior_bullish_bar"][idxs] = feature_frame["prev_bullish"][idxs]
        feature_frame["inside_bar"][idxs] = ((h <= prev_high) & (l >= prev_low)).fillna(False).to_numpy(dtype=bool)
        feature_frame["outside_bar"][idxs] = ((h >= prev_high) & (l <= prev_low)).fillna(False).to_numpy(dtype=bool)
        feature_frame["returns_1"][idxs] = c.pct_change(1, fill_method=None).to_numpy(dtype=np.float32)
        feature_frame["returns_3"][idxs] = c.pct_change(3, fill_method=None).to_numpy(dtype=np.float32)
        feature_frame["returns_5"][idxs] = c.pct_change(5, fill_method=None).to_numpy(dtype=np.float32)
        feature_frame["returns_10"][idxs] = c.pct_change(10, fill_method=None).to_numpy(dtype=np.float32)
        fwd_1h = (c.shift(-fwd_1h_bars) / c) - 1.0
        fwd_3h = (c.shift(-fwd_3h_bars) / c) - 1.0
        feature_frame["forward_return_1h"][idxs] = fwd_1h.to_numpy(dtype=np.float32)
        feature_frame["forward_return_3h"][idxs] = fwd_3h.to_numpy(dtype=np.float32)
        accel = c - 2.0 * c.shift(1) + c.shift(2)
        feature_frame["acceleration_close"][idxs] = accel.to_numpy(dtype=np.float32)
        feature_frame["acceleration_close_atr"][idxs] = (accel / atr14.clip(lower=eps)).to_numpy(dtype=np.float32)

        for lens in (10, 20, 30, 50):
            ema = c.ewm(span=lens, adjust=False, min_periods=lens).mean()
            feature_frame[f"ema_{lens}"][idxs] = ema.to_numpy(dtype=np.float32)
            feature_frame[f"distance_to_ema_{lens}"][idxs] = (c - ema).to_numpy(dtype=np.float32)
            feature_frame[f"distance_to_ema_atr_{lens}"][idxs] = (
                (c - ema) / atr14.clip(lower=eps)
            ).to_numpy(dtype=np.float32)
        feature_frame["distance_to_ema20_atr"][idxs] = feature_frame["distance_to_ema_atr_20"][idxs]
        feature_frame["distance_to_ema50_atr"][idxs] = feature_frame["distance_to_ema_atr_50"][idxs]
        feature_frame["trend_alignment_ema20_gt_ema50"][idxs] = (
            feature_frame["ema_20"][idxs] > feature_frame["ema_50"][idxs]
        )
        for slope_bars in (3, 5):
            ema20 = pd.Series(feature_frame["ema_20"][idxs], copy=False)
            feature_frame[f"ema_slope_ema20_{slope_bars}"][idxs] = (
                ema20 - ema20.shift(slope_bars)
            ).to_numpy(dtype=np.float32)
        for lookback in (5, 10, 15, 20):
            rolling_high = _rolling_shifted_max(h, lookback)
            rolling_low = _rolling_shifted_min(l, lookback)
            feature_frame[f"rolling_high_{lookback}"][idxs] = rolling_high.to_numpy(dtype=np.float32)
            feature_frame[f"rolling_low_{lookback}"][idxs] = rolling_low.to_numpy(dtype=np.float32)
            if lookback in (5, 10, 20):
                feature_frame[f"rolling_range_{lookback}"][idxs] = (
                    rolling_high - rolling_low
                ).to_numpy(dtype=np.float32)
                feature_frame[f"close_gt_rolling_high_{lookback}"][idxs] = (c > rolling_high).fillna(False).to_numpy(dtype=bool)
                feature_frame[f"close_lt_rolling_low_{lookback}"][idxs] = (c < rolling_low).fillna(False).to_numpy(dtype=bool)
                feature_frame[f"high_gt_rolling_high_{lookback}"][idxs] = (h > rolling_high).fillna(False).to_numpy(dtype=bool)
                feature_frame[f"low_lt_rolling_low_{lookback}"][idxs] = (l < rolling_low).fillna(False).to_numpy(dtype=bool)

        if use_persisted_trigger_cols:
            for column_name in persisted_trigger_cols:
                feature_frame[column_name][idxs] = (
                    np.asarray(feature_dict[column_name][idxs], dtype=np.float32) > 0.0
                )
        else:
            local_df = pd.DataFrame(
                {
                    "open": o.astype("float32", copy=False),
                    "high": h.astype("float32", copy=False),
                    "low": l.astype("float32", copy=False),
                    "close": c.astype("float32", copy=False),
                    "volume": v.astype("float32", copy=False),
                    "session_id": pd.Series(feature_frame["day_ids"][idxs], index=c.index, dtype="int32"),
                },
                index=c.index,
            )
            local_lib = build_intraday_crypto_library(local_df)
            for column_name in INTRADAY_TRIGGER_COLUMNS:
                col = local_lib.get(column_name)
                if col is None:
                    continue
                feature_frame[column_name][idxs] = np.asarray(col, dtype=np.int8) > 0

    return feature_frame


def generate_trigger_templates(
    config: TriggerDiscoveryConfig,
    parent_regime_metadata: Optional[Dict[str, Any]] = None,
) -> List[TriggerTemplate]:
    del parent_regime_metadata
    templates: List[TriggerTemplate] = []
    for column_name in INTRADAY_TRIGGER_COLUMNS:
        trigger_direction = "long" if column_name.startswith("LONG_") else "short"
        templates.append(
            TriggerTemplate(
                trigger_family=trigger_family_from_column(column_name),
                trigger_template_name=column_name,
                params={"column_name": column_name},
                trigger_direction=trigger_direction,
                trigger_anchor_feature=column_name,
                definition=f"intraday_crypto_library[{column_name}]",
            )
        )
    legacy_aliases: List[TriggerTemplate] = [
        TriggerTemplate("pullback_recovery", "ema_reclaim_touch", {"ema_len": 10}, "conditional", "ema_10", "legacy alias"),
        TriggerTemplate("breakout", "simple_close_breakout", {"lookback": 5}, "conditional", "rolling_high_5", "legacy alias"),
        TriggerTemplate("expansion_impulse", "expansion_bar", {"range_atr_min": 1.5}, "conditional", "range_atr", "legacy alias"),
        TriggerTemplate("expansion_impulse", "impulse_bar", {"range_atr_min": 1.5, "body_ratio_min": 0.6}, "conditional", "range_atr", "legacy alias"),
        TriggerTemplate("sweep_reversal", "relaxed_sweep", {"lookback": 5, "wick_min": 0.4}, "conditional", "lower_wick_ratio", "legacy alias"),
        TriggerTemplate("compression_release", "compression_release", {"compression_ratio_max": 0.5, "range_atr_min": 1.5}, "conditional", "compression_ratio", "legacy alias"),
        TriggerTemplate("compression_release", "compressed_breakout_up_down", {"compression_ratio_max": 0.5, "lookback": 5}, "conditional", "compression_ratio", "legacy alias"),
    ]
    if config.enable_simple_close_breakout:
        templates.extend(legacy_aliases)
    return templates


def _apply_template(
    template: TriggerTemplate,
    feature_frame: Dict[str, np.ndarray],
    is_long: bool,
) -> np.ndarray:
    cache = feature_frame.setdefault("_template_mask_cache", {})
    cache_key = (template.trigger_id, bool(is_long))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    if "column_name" in template.params:
        column_name = str(template.params["column_name"])
        column = feature_frame.get(column_name)
        if column is None:
            raise ValueError(f"Missing trigger column in feature frame: {column_name}")
        out = np.asarray(column, dtype=bool)
        if template.trigger_direction == "long" and not is_long:
            out = np.zeros_like(out, dtype=bool)
        elif template.trigger_direction == "short" and is_long:
            out = np.zeros_like(out, dtype=bool)
        cache[cache_key] = out
        return out

    close = np.asarray(feature_frame["close"], dtype=np.float32)
    open_ = np.asarray(feature_frame["open"], dtype=np.float32)
    high = np.asarray(feature_frame["high"], dtype=np.float32)
    low = np.asarray(feature_frame["low"], dtype=np.float32)
    body_ratio = np.asarray(feature_frame["body_ratio"], dtype=np.float32)
    close_loc = np.asarray(feature_frame["close_location_in_bar"], dtype=np.float32)
    range_atr = np.asarray(feature_frame["range_atr"], dtype=np.float32)
    prev_bearish = np.asarray(feature_frame["prior_bearish_bar"], dtype=bool)
    prev_bullish = np.asarray(feature_frame["prior_bullish_bar"], dtype=bool)
    bullish_bar = np.asarray(feature_frame["bullish_bar"], dtype=bool)
    bearish_bar = np.asarray(feature_frame["bearish_bar"], dtype=bool)
    compression_ratio = np.asarray(feature_frame["compression_ratio"], dtype=np.float32)
    valid = np.isfinite(close)
    name = template.trigger_template_name
    params = template.params

    if name == "close_crosses_above_ema":
        ema = np.asarray(feature_frame[f"ema_{int(params['ema_len'])}"], dtype=np.float32)
        prev_close = np.roll(close, 1)
        prev_ema = np.roll(ema, 1)
        mask = (prev_close <= prev_ema) & (close > ema) if is_long else (prev_close >= prev_ema) & (close < ema)
        mask[0] = False
        out = valid & np.isfinite(ema) & np.isfinite(prev_ema) & mask
    elif name == "ema_reclaim_touch":
        ema = np.asarray(feature_frame[f"ema_{int(params['ema_len'])}"], dtype=np.float32)
        out = valid & np.isfinite(ema) & ((low <= ema) & (close > ema) if is_long else (high >= ema) & (close < ema))
    elif name == "reclaim_after_opposite_bar":
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        body_min = float(params["body_ratio_min"])
        mask = (close > prev_high) & prev_bearish & (body_ratio >= body_min) if is_long else (close < prev_low) & prev_bullish & (body_ratio >= body_min)
        mask[0] = False
        out = valid & mask
    elif name == "close_in_extreme_of_range":
        threshold = float(params["close_location_min"])
        out = valid & ((bullish_bar & (close_loc >= threshold)) if is_long else (bearish_bar & (close_loc <= (1.0 - threshold))))
    elif name == "simple_close_breakout":
        lookback = int(params["lookback"])
        anchor = np.asarray(feature_frame[f"rolling_high_{lookback}" if is_long else f"rolling_low_{lookback}"], dtype=np.float32)
        out = valid & np.isfinite(anchor) & ((close > anchor) if is_long else (close < anchor))
    elif name == "expansion_body_breakout":
        lookback = int(params["lookback"])
        body_min = float(params["body_ratio_min"])
        range_min = float(params["range_atr_min"])
        anchor = np.asarray(feature_frame[f"rolling_high_{lookback}" if is_long else f"rolling_low_{lookback}"], dtype=np.float32)
        out = valid & np.isfinite(anchor) & (((close > anchor) if is_long else (close < anchor))) & (body_ratio >= body_min) & (range_atr >= range_min)
    elif name == "expansion_bar":
        out = valid & ((bullish_bar if is_long else bearish_bar)) & (range_atr >= float(params["range_atr_min"]))
    elif name == "impulse_bar":
        out = valid & ((bullish_bar if is_long else bearish_bar)) & (range_atr >= float(params["range_atr_min"])) & (body_ratio >= float(params["body_ratio_min"]))
    elif name == "relaxed_sweep":
        lookback = int(params["lookback"])
        wick_min = float(params["wick_min"])
        anchor = np.asarray(feature_frame[f"rolling_low_{lookback}" if is_long else f"rolling_high_{lookback}"], dtype=np.float32)
        wick = np.asarray(feature_frame["lower_wick_ratio" if is_long else "upper_wick_ratio"], dtype=np.float32)
        out = valid & np.isfinite(anchor) & (((low < anchor) if is_long else (high > anchor))) & (wick >= wick_min)
    elif name == "compression_release":
        out = valid & ((bullish_bar if is_long else bearish_bar)) & (compression_ratio <= float(params["compression_ratio_max"])) & (range_atr >= float(params["range_atr_min"]))
    elif name == "compressed_breakout_up_down":
        lookback = int(params["lookback"])
        anchor = np.asarray(feature_frame[f"rolling_high_{lookback}" if is_long else f"rolling_low_{lookback}"], dtype=np.float32)
        out = valid & np.isfinite(anchor) & (compression_ratio <= float(params["compression_ratio_max"])) & (((close > anchor) if is_long else (close < anchor)))
    else:
        raise ValueError(f"Unsupported trigger template: {name}")
    cache[cache_key] = np.asarray(out, dtype=bool)
    return cache[cache_key]


def compute_timing_metrics(
    event_mask: np.ndarray,
    feature_frame: Dict[str, np.ndarray],
    asset_groups: Dict[int, np.ndarray],
    horizon_bars: int,
    is_long: bool,
) -> Dict[str, float]:
    high = feature_frame["high"]
    low = feature_frame["low"]
    close = feature_frame["close"]
    event_idx = np.flatnonzero(event_mask)
    if event_idx.size == 0:
        return {
            "bars_to_mfe_mean": float("nan"),
            "bars_to_mae_mean": float("nan"),
            "mfe_before_mae_fraction": float("nan"),
            "prompt_excursion_quality": float("nan"),
            "timing_precision_score": float("nan"),
        }

    bars_to_mfe: List[float] = []
    bars_to_mae: List[float] = []
    mfe_before: List[float] = []

    lookup: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for code, idxs in asset_groups.items():
        local_pos = np.full(feature_frame["close"].shape[0], -1, dtype=np.int32)
        local_pos[idxs] = np.arange(idxs.shape[0], dtype=np.int32)
        lookup[int(code)] = (idxs, local_pos)

    symbols = feature_frame["symbol_codes"]
    for idx in event_idx.tolist():
        symbol = int(symbols[idx])
        idxs, local_pos = lookup[symbol]
        pos = int(local_pos[idx])
        if pos < 0:
            continue
        end = min(idxs.shape[0], pos + horizon_bars + 1)
        future = idxs[pos:end]
        if future.shape[0] <= 1:
            continue
        entry = float(close[idx])
        future_high = high[future]
        future_low = low[future]
        if is_long:
            mfe_path = future_high - entry
            mae_path = entry - future_low
        else:
            mfe_path = entry - future_low
            mae_path = future_high - entry
        mfe_pos = int(np.nanargmax(mfe_path))
        mae_pos = int(np.nanargmax(mae_path))
        bars_to_mfe.append(float(mfe_pos))
        bars_to_mae.append(float(mae_pos))
        mfe_before.append(float(mfe_pos <= mae_pos))

    if not bars_to_mfe:
        return {
            "bars_to_mfe_mean": float("nan"),
            "bars_to_mae_mean": float("nan"),
            "mfe_before_mae_fraction": float("nan"),
            "prompt_excursion_quality": float("nan"),
            "timing_precision_score": float("nan"),
        }

    bars_to_mfe_mean = float(np.mean(bars_to_mfe))
    bars_to_mae_mean = float(np.mean(bars_to_mae))
    mfe_before_mae_fraction = float(np.mean(mfe_before))
    prompt_excursion_quality = _sigmoid((bars_to_mae_mean - bars_to_mfe_mean) / max(horizon_bars / 4.0, 1.0))
    fast_mfe_score = float(np.clip(1.0 - bars_to_mfe_mean / max(float(horizon_bars), 1.0), 0.0, 1.0))
    slow_mae_bonus = float(np.clip(bars_to_mae_mean / max(float(horizon_bars), 1.0), 0.0, 1.0))
    timing_precision_score = float(
        np.clip(
            0.40 * fast_mfe_score
            + 0.35 * mfe_before_mae_fraction
            + 0.15 * prompt_excursion_quality
            + 0.10 * slow_mae_bonus,
            0.0,
            1.0,
        )
    )
    return {
        "bars_to_mfe_mean": bars_to_mfe_mean,
        "bars_to_mae_mean": bars_to_mae_mean,
        "mfe_before_mae_fraction": mfe_before_mae_fraction,
        "prompt_excursion_quality": prompt_excursion_quality,
        "timing_precision_score": timing_precision_score,
    }


def compute_horizon_edge_metrics(
    event_mask: np.ndarray,
    feature_frame: Dict[str, np.ndarray],
    asset_groups: Dict[int, np.ndarray],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    horizon_bars: int,
    is_long: bool,
) -> Dict[str, float]:
    horizon = max(int(horizon_bars), 1)
    bars_per_hour = max(int(feature_frame.get("_bars_per_hour", 4)), 1)
    cache_key = None
    if horizon == bars_per_hour and "forward_return_1h" in feature_frame:
        cache_key = "forward_return_1h"
    elif horizon == (3 * bars_per_hour) and "forward_return_3h" in feature_frame:
        cache_key = "forward_return_3h"
    if cache_key is not None:
        forward_returns = np.asarray(feature_frame[cache_key], dtype=np.float32).copy()
        if not is_long:
            forward_returns = (-forward_returns).astype(np.float32, copy=False)
    else:
        close = np.asarray(feature_frame["close"], dtype=np.float32)
        forward_returns = np.full(close.shape[0], np.nan, dtype=np.float32)
        for idxs in asset_groups.values():
            arr = np.asarray(idxs, dtype=np.int32)
            if arr.size <= horizon:
                continue
            base = close[arr[:-horizon]]
            fwd = close[arr[horizon:]]
            denom = np.where(np.abs(base) > 1e-12, base, np.nan)
            directional = (fwd / denom) - 1.0
            if not is_long:
                directional = -directional
            forward_returns[arr[:-horizon]] = directional.astype(np.float32, copy=False)

    valid_fwd = np.isfinite(forward_returns)
    event_valid = event_mask & valid_fwd
    non_event_valid = (~event_mask) & valid_fwd
    if not np.any(event_valid) or not np.any(non_event_valid):
        return {
            "mean_forward_return": float("nan"),
            "delta": 0.0,
            "fold_mean": 0.0,
            "fold_std": 0.0,
            "shrunk_delta": 0.0,
            "positive_fold_fraction": 0.0,
        }

    mean_forward_return = float(np.nanmean(forward_returns[event_valid]))
    delta = float(np.nanmean(forward_returns[event_valid]) - np.nanmean(forward_returns[non_event_valid]))
    fold_deltas: List[float] = []
    for _, val_idx in cv_splits:
        val_idx_arr = np.asarray(val_idx, dtype=np.int32)
        val_event = event_mask[val_idx_arr] & valid_fwd[val_idx_arr]
        val_non_event = (~event_mask[val_idx_arr]) & valid_fwd[val_idx_arr]
        if np.any(val_event) and np.any(val_non_event):
            val_returns = forward_returns[val_idx_arr]
            fold_delta = float(np.nanmean(val_returns[val_event]) - np.nanmean(val_returns[val_non_event]))
        else:
            fold_delta = 0.0
        fold_deltas.append(fold_delta)
    fold_deltas_arr = np.asarray(fold_deltas, dtype=np.float32)
    fold_mean = float(np.mean(fold_deltas_arr)) if fold_deltas_arr.size else 0.0
    fold_std = float(np.std(fold_deltas_arr)) if fold_deltas_arr.size else 0.0
    positive_fold_fraction = float(np.mean(fold_deltas_arr > 0.0)) if fold_deltas_arr.size else 0.0
    total_events = int(np.sum(event_valid))
    shrunk_delta = float(delta * (total_events / (total_events + 750.0)))
    return {
        "mean_forward_return": mean_forward_return,
        "delta": delta,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "shrunk_delta": shrunk_delta,
        "positive_fold_fraction": positive_fold_fraction,
    }


def compute_trigger_score(row: pd.Series, config: TriggerDiscoveryConfig) -> Tuple[float, float]:
    edge = np.tanh(max(_safe_float(row.get("delta_r_shrunk"), 0.0), 0.0) * 250.0)
    edge_1h = np.tanh(max(_safe_float(row.get("trigger_edge_shrunk_1h"), 0.0), 0.0) * 250.0)
    edge_3h = np.tanh(max(_safe_float(row.get("trigger_edge_shrunk_3h"), 0.0), 0.0) * 250.0)
    stability = np.clip(_safe_float(row.get("S_r"), 0.0), 0.0, 1.0)
    pred = np.tanh(max(_safe_float(row.get("primary_predictability_gain"), 0.0), 0.0) * 25.0)
    timing = np.clip(_safe_float(row.get("timing_precision_score"), 0.0), 0.0, 1.0)
    dispersion = np.tanh(max(_safe_float(row.get("dispersion_to_edge_ratio"), 0.0), 0.0) / 5.0)
    parent_gain = np.tanh(_safe_float(row.get("trigger_gain_vs_parent"), 0.0) * 250.0)
    support_ratio = np.clip(_safe_float(row.get("trigger_delta_support_vs_parent"), 0.0), 0.0, 1.0)
    coverage_loss_penalty = max(0.0, 1.0 - support_ratio)
    info_gain = max(
        _safe_float(row.get("trigger_gain_vs_parent"), 0.0),
        0.0,
    ) + 0.5 * max(_safe_float(row.get("delta_r_shrunk"), 0.0), 0.0)
    information_efficiency = np.tanh(250.0 * info_gain) * np.sqrt(max(support_ratio, 0.0))
    raw = (
        config.w_edge * edge
        + config.w_edge_1h * edge_1h
        + config.w_edge_3h * edge_3h
        + config.w_stability * stability
        + config.w_pred * pred
        + config.w_timing * timing
        - config.w_disp * dispersion
        + config.w_parent * parent_gain
        + 0.9 * information_efficiency
        - config.w_covloss * coverage_loss_penalty
    )
    support_multiplier = np.clip(_safe_float(row.get("support_multiplier"), 0.0), 0.0, 1.0)
    positive_fold_multiplier = 0.75 + 0.25 * np.clip(_safe_float(row.get("positive_fold_fraction_r"), 0.0), 0.0, 1.0)
    simplicity_multiplier = 1.0
    final = float(raw * support_multiplier * positive_fold_multiplier * simplicity_multiplier)
    return float(raw), final


def _compute_trigger_prescore(context: Dict[str, Any], config: TriggerDiscoveryConfig) -> float:
    edge_1h = np.tanh(max(_safe_float(context.get("trigger_edge_shrunk_1h"), 0.0), 0.0) * 250.0)
    edge_3h = np.tanh(max(_safe_float(context.get("trigger_edge_shrunk_3h"), 0.0), 0.0) * 250.0)
    timing = np.clip(_safe_float(context.get("timing_precision_score"), 0.0), 0.0, 1.0)
    support = np.clip(_safe_float(context.get("support_multiplier"), 0.0), 0.0, 1.0)
    stability = np.clip(
        0.5 * _safe_float(context.get("trigger_edge_positive_fold_fraction_1h"), 0.0)
        + 0.5 * _safe_float(context.get("trigger_edge_positive_fold_fraction_3h"), 0.0),
        0.0,
        1.0,
    )
    return float(
        support
        * (
            config.w_edge_3h * edge_3h
            + config.w_edge_1h * edge_1h
            + config.w_timing * timing
            + 0.5 * config.w_stability * stability
        )
    )


def _log_trigger_rejection(
    *,
    mode: str,
    parent_name: str,
    trigger_name: str,
    reason: str,
    support_ratio: float,
) -> None:
    tprint(
        "Phase 2.75 reject "
        f"({mode}) parent={parent_name} trigger={trigger_name} "
        f"reason={reason} "
        f"parent_keep_pct={support_ratio * 100.0:.1f} "
    )


def _prescreen_trigger_for_regime(
    parent_regime_row: pd.Series,
    parent_mask: np.ndarray,
    trigger_template: TriggerTemplate,
    feature_frame: Dict[str, np.ndarray],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    signed_returns: np.ndarray,
    config: TriggerDiscoveryConfig,
    mode: str,
    shared: Dict[str, Any],
    asset_groups: Dict[int, np.ndarray],
    parent_timing_metrics: Optional[Dict[str, float]] = None,
    parent_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    is_long = mode == "long"
    trigger_mask = _apply_template(trigger_template, feature_frame, is_long=is_long)
    entry_mask = np.asarray(parent_mask & trigger_mask, dtype=bool)
    total_events = int(np.sum(entry_mask))
    parent_total_events = int(parent_context.get("parent_total_events", np.sum(parent_mask))) if parent_context else int(np.sum(parent_mask))
    support_ratio = float(total_events / max(parent_total_events, 1))
    day_ids = feature_frame["day_ids"]
    symbol_codes = feature_frame["symbol_codes"]
    n_days = max(int(np.unique(day_ids).size), 1)
    unique_event_days = int(np.unique(day_ids[entry_mask]).size) if total_events else 0
    active_days_fraction = float(unique_event_days / n_days)
    symbol_summary = _event_symbol_summary(entry_mask, symbol_codes)
    bars_per_hour = int(shared.get("bph", shared.get("bars_per_hour", 4)))
    duration_stats = _event_run_duration_stats(entry_mask, asset_groups, bars_per_hour)
    event_period = _format_timestamp_bounds(
        np.asarray(feature_frame["timestamps"])[entry_mask]
        if total_events
        else np.asarray(feature_frame["timestamps"])
    )
    parent_name = str(parent_regime_row.get("regime_id", parent_regime_row.get("name")))
    trigger_name = str(trigger_template.trigger_template_name)

    if total_events < config.min_trigger_events:
        _log_trigger_rejection(
            mode=mode,
            parent_name=parent_name,
            trigger_name=trigger_name,
            reason="too_few_events",
            support_ratio=support_ratio,
        )
        return None
    if active_days_fraction < config.min_trigger_active_days_fraction:
        _log_trigger_rejection(
            mode=mode,
            parent_name=parent_name,
            trigger_name=trigger_name,
            reason="active_days_too_low",
            support_ratio=support_ratio,
        )
        return None
    if support_ratio < config.min_trigger_support_ratio:
        _log_trigger_rejection(
            mode=mode,
            parent_name=parent_name,
            trigger_name=trigger_name,
            reason="support_ratio_too_low",
            support_ratio=support_ratio,
        )
        return None
    if symbol_summary["event_symbol_count"] < config.min_trigger_distinct_symbols:
        _log_trigger_rejection(
            mode=mode,
            parent_name=parent_name,
            trigger_name=trigger_name,
            reason="too_few_symbols",
            support_ratio=support_ratio,
        )
        return None
    if symbol_summary["top_symbol_share"] > config.max_trigger_top_symbol_share:
        _log_trigger_rejection(
            mode=mode,
            parent_name=parent_name,
            trigger_name=trigger_name,
            reason="top_symbol_share_too_high",
            support_ratio=support_ratio,
        )
        return None

    valid_fwd = (
        np.asarray(parent_context["valid_fwd"], dtype=bool)
        if parent_context is not None and "valid_fwd" in parent_context
        else np.isfinite(signed_returns)
    )
    fold_val_indices = (
        parent_context.get("fold_val_indices", [np.asarray(val_idx, dtype=np.int32) for _, val_idx in cv_splits])
        if parent_context is not None
        else [np.asarray(val_idx, dtype=np.int32) for _, val_idx in cv_splits]
    )
    counts_per_day = (
        pd.Series(day_ids[entry_mask], copy=False).value_counts().sort_index()
        if total_events
        else pd.Series(dtype=np.int64)
    )
    fold_counts = [int(np.sum(entry_mask[val_idx])) for val_idx in fold_val_indices]
    if not fold_counts or float(np.mean(fold_counts)) < config.min_fold_events:
        _log_trigger_rejection(
            mode=mode,
            parent_name=parent_name,
            trigger_name=trigger_name,
            reason="fold_support_too_low",
            support_ratio=support_ratio,
        )
        return None

    timing_metrics = compute_timing_metrics(
        event_mask=entry_mask,
        feature_frame=feature_frame,
        asset_groups=asset_groups,
        horizon_bars=max(config.trigger_timing_horizon_bars, 1),
        is_long=is_long,
    )
    horizon_1h_metrics = compute_horizon_edge_metrics(
        event_mask=entry_mask,
        feature_frame=feature_frame,
        asset_groups=asset_groups,
        cv_splits=cv_splits,
        horizon_bars=max(config.trigger_edge_horizon_1h_bars, 1),
        is_long=is_long,
    )
    horizon_3h_metrics = compute_horizon_edge_metrics(
        event_mask=entry_mask,
        feature_frame=feature_frame,
        asset_groups=asset_groups,
        cv_splits=cv_splits,
        horizon_bars=max(config.trigger_edge_horizon_3h_bars, 1),
        is_long=is_long,
    )
    support_multiplier = float(np.clip(np.sqrt(total_events / max(config.min_trigger_events * 4.0, 1.0)), 0.0, 1.0))
    context = {
        "trigger_template": trigger_template,
        "entry_mask": entry_mask,
        "total_events": total_events,
        "parent_total_events": parent_total_events,
        "support_ratio": support_ratio,
        "active_days_fraction": active_days_fraction,
        "symbol_summary": symbol_summary,
        "duration_stats": duration_stats,
        "event_period": event_period,
        "counts_per_day": counts_per_day,
        "fold_counts": fold_counts,
        "timing_metrics": timing_metrics,
        "horizon_1h_metrics": horizon_1h_metrics,
        "horizon_3h_metrics": horizon_3h_metrics,
        "support_multiplier": support_multiplier,
        "parent_timing_score": _safe_float((parent_timing_metrics or {}).get("timing_precision_score"), 0.0),
        "valid_fwd": valid_fwd,
        "trigger_edge_shrunk_1h": float(horizon_1h_metrics["shrunk_delta"]),
        "trigger_edge_shrunk_3h": float(horizon_3h_metrics["shrunk_delta"]),
        "trigger_edge_positive_fold_fraction_1h": float(horizon_1h_metrics["positive_fold_fraction"]),
        "trigger_edge_positive_fold_fraction_3h": float(horizon_3h_metrics["positive_fold_fraction"]),
        "timing_precision_score": float(timing_metrics["timing_precision_score"]),
    }
    context["cheap_prescore"] = _compute_trigger_prescore(context, config)
    return context


def prune_non_dominated_triggers(
    df: pd.DataFrame,
    config: TriggerDiscoveryConfig,
) -> pd.DataFrame:
    if df.empty or not config.apply_non_dominance:
        if "non_dominated_flag" not in df.columns:
            df = df.copy()
            df["non_dominated_flag"] = True
        return df

    kept_groups: List[pd.DataFrame] = []
    for _, group in df.groupby("parent_regime_id", sort=False):
        rows = []
        for idx, row in group.iterrows():
            dominated = False
            for other_idx, other in group.iterrows():
                if idx == other_idx:
                    continue
                same_or_simpler = (
                    _safe_float(other.get("total_events"), 0.0) >= _safe_float(row.get("total_events"), 0.0)
                )
                dominates = (
                    _safe_float(other.get("delta_r_shrunk"), -1e9) >= _safe_float(row.get("delta_r_shrunk"), -1e9)
                    and _safe_float(other.get("S_r"), -1e9) >= _safe_float(row.get("S_r"), -1e9)
                    and _safe_float(other.get("D_r"), 1e9) <= _safe_float(row.get("D_r"), 1e9)
                    and _safe_float(other.get("timing_precision_score"), -1e9) >= _safe_float(row.get("timing_precision_score"), -1e9)
                    and same_or_simpler
                )
                strict = (
                    _safe_float(other.get("delta_r_shrunk"), -1e9) > _safe_float(row.get("delta_r_shrunk"), -1e9)
                    or _safe_float(other.get("S_r"), -1e9) > _safe_float(row.get("S_r"), -1e9)
                    or _safe_float(other.get("D_r"), 1e9) < _safe_float(row.get("D_r"), 1e9)
                    or _safe_float(other.get("timing_precision_score"), -1e9) > _safe_float(row.get("timing_precision_score"), -1e9)
                )
                if dominates and strict:
                    dominated = True
                    break
            new_row = row.copy()
            new_row["non_dominated_flag"] = not dominated
            rows.append(new_row)
        kept_groups.append(pd.DataFrame(rows))
    return pd.concat(kept_groups, ignore_index=True)


def _prune_trigger_prescreen_overlap(
    prescreen_rows: Sequence[Dict[str, Any]],
    overlap_threshold: float,
) -> List[Dict[str, Any]]:
    if not prescreen_rows:
        return []
    ordered = sorted(
        prescreen_rows,
        key=lambda row: float(row.get("cheap_prescore", 0.0)),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    kept_masks: List[np.ndarray] = []
    for row in ordered:
        entry_mask = np.asarray(row.get("entry_mask"), dtype=bool)
        is_duplicate = False
        for prev_mask in kept_masks:
            union = int(np.sum(entry_mask | prev_mask))
            if union <= 0:
                continue
            overlap = float(np.sum(entry_mask & prev_mask)) / float(union)
            if overlap >= overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(row)
            kept_masks.append(entry_mask)
    return kept


def _lgbm_rank_trigger_prescreen_rows(
    prescreen_rows: Sequence[Dict[str, Any]],
    parent_mask: np.ndarray,
    signed_returns: np.ndarray,
    keep_fraction: float,
    max_keep: int,
) -> List[Dict[str, Any]]:
    """
    Non-linear prescreening using LGBM importance.
    """
    if not prescreen_rows:
        return []

    rows = list(prescreen_rows)
    if len(rows) <= 1:
        return rows

    parent_mask_arr = np.asarray(parent_mask, dtype=bool)
    signed_returns_arr = np.asarray(signed_returns, dtype=np.float32)
    valid_rows = parent_mask_arr & np.isfinite(signed_returns_arr)
    if int(np.sum(valid_rows)) < 20:
        return sorted(
            rows,
            key=lambda row: float(row.get("cheap_prescore", 0.0)),
            reverse=True,
        )[: max(1, min(len(rows), max_keep))]

    X = np.zeros((int(np.sum(valid_rows)), len(rows)), dtype=np.float32)
    for col_idx, row in enumerate(rows):
        entry_mask = np.asarray(row.get("entry_mask"), dtype=bool)
        X[:, col_idx] = np.asarray(entry_mask[valid_rows], dtype=np.float32)

    y = signed_returns_arr[valid_rows]
    
    # Cheap yet better non-linear importance
    model = LGBMRegressor(
        n_estimators=30,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.1,
        importance_type='gain',
        min_child_samples=max(5, X.shape[0] // 10),
        n_jobs=1,
        verbosity=-1,
        random_state=42
    )
    
    try:
        model.fit(X, y)
        importances = model.feature_importances_
    except:
        importances = np.zeros(len(rows), dtype=np.float32)

    rank_scores = importances.astype(np.float32)

    for idx, row in enumerate(rows):
        row["lgbm_prescreen_gain"] = float(rank_scores[idx])

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("lgbm_prescreen_gain", 0.0)),
            float(row.get("cheap_prescore", 0.0)),
        ),
        reverse=True,
    )

    n_to_keep = max(1, min(int(len(rows) * keep_fraction), max_keep))
    return sorted_rows[:n_to_keep]


def _ridge_rank_trigger_prescreen_rows(
    prescreen_rows: Sequence[Dict[str, Any]],
    parent_mask: np.ndarray,
    signed_returns: np.ndarray,
    keep_fraction: float,
    max_keep: int,
    alpha: float = 1.0,
) -> List[Dict[str, Any]]:
    if not prescreen_rows:
        return []

    rows = list(prescreen_rows)
    if len(rows) <= 1:
        for row in rows:
            row["ridge_prescreen_abs_coef"] = float(abs(row.get("cheap_prescore", 0.0)))
        return rows

    parent_mask_arr = np.asarray(parent_mask, dtype=bool)
    signed_returns_arr = np.asarray(signed_returns, dtype=np.float32)
    valid_rows = parent_mask_arr & np.isfinite(signed_returns_arr)
    if int(np.sum(valid_rows)) < 20:
        ranked = sorted(
            rows,
            key=lambda row: float(row.get("cheap_prescore", 0.0)),
            reverse=True,
        )
        n_to_keep = max(1, min(int(len(ranked) * keep_fraction), max_keep))
        kept = ranked[:n_to_keep]
        for row in kept:
            row["ridge_prescreen_abs_coef"] = float(abs(row.get("cheap_prescore", 0.0)))
        return kept

    X = np.zeros((int(np.sum(valid_rows)), len(rows)), dtype=np.float32)
    for col_idx, row in enumerate(rows):
        entry_mask = np.asarray(row.get("entry_mask"), dtype=bool)
        X[:, col_idx] = np.asarray(entry_mask[valid_rows], dtype=np.float32)
    y = signed_returns_arr[valid_rows]

    try:
        xtx = X.T @ X
        xtx += np.eye(xtx.shape[0], dtype=np.float32) * float(max(alpha, 1e-6))
        xty = X.T @ y
        coefs = np.linalg.solve(xtx, xty).astype(np.float32)
    except np.linalg.LinAlgError:
        coefs = np.zeros(len(rows), dtype=np.float32)

    for idx, row in enumerate(rows):
        row["ridge_prescreen_abs_coef"] = float(abs(coefs[idx]))

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("ridge_prescreen_abs_coef", 0.0)),
            float(row.get("cheap_prescore", 0.0)),
        ),
        reverse=True,
    )
    n_to_keep = max(1, min(int(len(ranked) * keep_fraction), max_keep))
    return ranked[:n_to_keep]


def evaluate_trigger_for_regime(
    parent_regime_row: pd.Series,
    parent_mask: np.ndarray,
    trigger_template: TriggerTemplate,
    feature_frame: Dict[str, np.ndarray],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    signed_returns: np.ndarray,
    config: TriggerDiscoveryConfig,
    compute_full_metrics_fn: MetricFn,
    mode: str,
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    asset_groups: Dict[int, np.ndarray],
    parent_timing_metrics: Optional[Dict[str, float]] = None,
    precomputed_context: Optional[Dict[str, Any]] = None,
    parent_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    context = precomputed_context or _prescreen_trigger_for_regime(
        parent_regime_row=parent_regime_row,
        parent_mask=parent_mask,
        trigger_template=trigger_template,
        feature_frame=feature_frame,
        cv_splits=cv_splits,
        signed_returns=signed_returns,
        config=config,
        mode=mode,
        shared=shared,
        asset_groups=asset_groups,
        parent_timing_metrics=parent_timing_metrics,
        parent_context=parent_context,
    )
    if context is None:
        return None

    is_long = mode == "long"
    entry_mask = np.asarray(context["entry_mask"], dtype=bool)
    total_events = int(context["total_events"])
    parent_total_events = int(context["parent_total_events"])
    support_ratio = float(context["support_ratio"])
    active_days_fraction = float(context["active_days_fraction"])
    symbol_summary = context["symbol_summary"]
    duration_stats = context["duration_stats"]
    event_period = str(context["event_period"])
    counts_per_day = context["counts_per_day"]
    fold_counts = list(context["fold_counts"])
    if symbol_summary["event_symbol_count"] <= 3:
        msg = (
            "SYMBOL_CONCENTRATION Trigger "
            f"parent={parent_regime_row.get('regime_id', parent_regime_row.get('name'))} "
            f"trigger={trigger_template.trigger_template_name} "
            f"symbols={symbol_summary['event_symbol_count']} "
            f"top_share={symbol_summary['top_symbol_share']:.3f} "
            f"top_codes={symbol_summary['top_symbol_codes_text']}"
        )
        tprint(msg)
        _append_trigger_symbol_log(msg)

    day_ids = feature_frame["day_ids"]
    symbol_codes = feature_frame["symbol_codes"]
    valid_fwd = np.asarray(context.get("valid_fwd", np.isfinite(signed_returns)), dtype=bool)
    non_event = (~entry_mask) & valid_fwd
    basic_edge = (
        float(np.nanmean(signed_returns[entry_mask & valid_fwd]) - np.nanmean(signed_returns[non_event]))
        if np.any(entry_mask & valid_fwd) and np.any(non_event)
        else 0.0
    )
    timing_metrics = context["timing_metrics"]
    horizon_1h_metrics = context["horizon_1h_metrics"]
    horizon_3h_metrics = context["horizon_3h_metrics"]
    metric_bundle = compute_full_metrics_fn(
        mode,
        entry_mask,
        shared,
        feature_dict,
        {"phase2_metric_max_samples_per_class": 15_000, **shared.get("runtime_cfg", {})},
        float(timing_metrics["bars_to_mfe_mean"] / max(config.trigger_timing_horizon_bars, 1)),
        float(basic_edge),
    )
    delta_r = _safe_float(metric_bundle.get("delta_r", metric_bundle.get("return_uplift", basic_edge)), basic_edge)
    delta_r_fold_mean = _safe_float(metric_bundle.get("delta_r_fold_mean", metric_bundle.get("magnitude_delta_fold_mean")), 0.0)
    delta_r_fold_std = _safe_float(metric_bundle.get("delta_r_fold_std", metric_bundle.get("magnitude_delta_fold_std")), 0.0)
    positive_fold_fraction_r = _safe_float(metric_bundle.get("positive_fold_fraction_r", metric_bundle.get("magnitude_positive_fold_fraction")), 0.0)
    s_r = float(
        0.5 * max(0.0, 1.0 - delta_r_fold_std / (abs(delta_r_fold_mean) + 1e-9))
        + 0.5 * positive_fold_fraction_r
    )

    fold_returns = []
    fold_cont_rates = []
    for _, val_idx in cv_splits:
        val_mask = entry_mask[val_idx] & valid_fwd[val_idx]
        fold_returns.append(float(np.nanmean(signed_returns[val_idx][val_mask])) if np.any(val_mask) else 0.0)
        labels = signed_returns[val_idx][val_mask]
        fold_cont_rates.append(float(np.mean(labels > 0.0)) if labels.size else 0.0)

    post_event_vol = pd.Series(np.asarray(signed_returns[entry_mask & valid_fwd], dtype=np.float32), copy=False)
    post_event_vol_dispersion = float(post_event_vol.rolling(20, min_periods=5).std().dropna().std()) if len(post_event_vol) >= 5 else 0.0
    fold_event_count_mean = float(np.mean(fold_counts))
    fold_event_count_std = float(np.std(fold_counts))
    fold_continuation_rate_std = float(np.std(fold_cont_rates))
    d_r = float(
        0.35 * (timing_metrics["bars_to_mfe_mean"] / max(config.trigger_timing_horizon_bars, 1))
        + 0.35 * post_event_vol_dispersion
        + 0.15 * fold_continuation_rate_std
        + 0.15 * (fold_event_count_std / max(fold_event_count_mean, 1.0))
    )
    delta_r_shrunk = float(delta_r * (total_events / (total_events + 750.0)))
    support_multiplier = float(context.get("support_multiplier", np.clip(np.sqrt(total_events / max(config.min_trigger_events * 4.0, 1.0)), 0.0, 1.0)))

    # score_r should reflect parent score + trigger boost, but here we recalculate it for the trigger context
    score_r = float(delta_r_shrunk * np.sqrt(max(total_events, 0)) * max(s_r, 0.0) / (1.0 + d_r))

    parent_timing_score = float(context.get("parent_timing_score", _safe_float((parent_timing_metrics or {}).get("timing_precision_score"), 0.0)))
    trigger_gain_vs_parent = float(delta_r_shrunk - _safe_float(parent_regime_row.get("delta_r_shrunk"), 0.0))
    trigger_delta_dispersion_vs_parent = float(d_r - _safe_float(parent_regime_row.get("D_r"), 0.0))
    trigger_delta_timing_vs_parent = float(timing_metrics["timing_precision_score"] - parent_timing_score)
    combined_sample_loss_vs_parent = float(max(1.0 - support_ratio, 0.05))

    row: Dict[str, Any] = {
        "parent_regime_id": str(parent_regime_row.get("regime_id", parent_regime_row.get("name"))),
        "parent_key": str(parent_regime_row.get("regime_id", parent_regime_row.get("name"))),
        "trigger_id": trigger_template.trigger_id,
        "trigger_family": trigger_template.trigger_family,
        "trigger_definition": trigger_template.definition,
        "trigger_params_json": trigger_template.trigger_params_json,
        "event_definition_full": f"{parent_regime_row.get('regime_id', parent_regime_row.get('name'))} AND {trigger_template.trigger_template_name}",
        "full_candidate_id": f"{parent_regime_row.get('regime_id', parent_regime_row.get('name'))}::{trigger_template.trigger_id}",
        "name": f"{parent_regime_row.get('regime_id', parent_regime_row.get('name'))}::{trigger_template.trigger_id}",
        "regime_id": f"{parent_regime_row.get('regime_id', parent_regime_row.get('name'))}::{trigger_template.trigger_id}",
        "regime_definition": f"{parent_regime_row.get('regime_id', parent_regime_row.get('name'))}::{trigger_template.trigger_id}",
        "trigger_template_name": trigger_template.trigger_template_name,
        "trigger_direction": trigger_template.trigger_direction,
        "trigger_anchor_feature": trigger_template.trigger_anchor_feature,
        "parent_regime_family": parent_regime_row.get("family"),
        "parent_regime_score": _safe_float(parent_regime_row.get("score_r"), 0.0),
        "parent_total_events": parent_total_events,
        "parent_delta_r_shrunk": _safe_float(parent_regime_row.get("delta_r_shrunk"), 0.0),
        "parent_S_r": _safe_float(parent_regime_row.get("S_r"), 0.0),
        "parent_D_r": _safe_float(parent_regime_row.get("D_r"), 0.0),
        "total_events": total_events,
        "N_r": float(total_events),
        "score_r": score_r,
        "active_days_fraction": active_days_fraction,
        "events_per_day_mean": float(counts_per_day.mean()) if not counts_per_day.empty else 0.0,
        "events_per_day_std": float(counts_per_day.std(ddof=0)) if not counts_per_day.empty else 0.0,
        "events_per_day_per_asset": float(total_events / max(np.unique(day_ids[entry_mask] * 10_000 + symbol_codes[entry_mask]).size, 1)),
        "event_symbol_count": int(symbol_summary["event_symbol_count"]),
        "top_symbol_share": float(symbol_summary["top_symbol_share"]),
        "top_symbol_codes_text": str(symbol_summary["top_symbol_codes_text"]),
        "event_period": event_period,
        "avg_event_duration_bars": float(duration_stats["avg_event_duration_bars"]),
        "median_event_duration_bars": float(duration_stats["median_event_duration_bars"]),
        "avg_event_duration_hours": float(duration_stats["avg_event_duration_hours"]),
        "median_event_duration_hours": float(duration_stats["median_event_duration_hours"]),
        "event_run_count": float(duration_stats["event_run_count"]),
        "fold_event_count_mean": fold_event_count_mean,
        "fold_event_count_std": fold_event_count_std,
        "support_multiplier": support_multiplier,
        "basic_directionality_edge_event_vs_non_event": basic_edge,
        "primary_predictability_gain": _safe_float(metric_bundle.get("primary_predictability_gain"), 0.0),
        "continuation_predictability_gain": _safe_float(metric_bundle.get("continuation_predictability_gain"), 0.0),
        "reversal_predictability_gain": _safe_float(metric_bundle.get("reversal_predictability_gain"), 0.0),
        "bucket_primary_delta_fold_mean": _safe_float(metric_bundle.get("bucket_primary_delta_fold_mean"), 0.0),
        "bucket_primary_delta_fold_std": _safe_float(metric_bundle.get("bucket_primary_delta_fold_std"), 0.0),
        "bucket_primary_delta_fold_count": _safe_float(metric_bundle.get("bucket_primary_delta_fold_count"), 0.0),
        "delta_r_raw": basic_edge,
        "delta_r": delta_r,
        "delta_r_fold_mean": delta_r_fold_mean,
        "delta_r_fold_std": delta_r_fold_std,
        "delta_r_shrunk": delta_r_shrunk,
        "trigger_edge_mean_1h": float(horizon_1h_metrics["mean_forward_return"]),
        "trigger_edge_delta_1h": float(horizon_1h_metrics["delta"]),
        "trigger_edge_fold_mean_1h": float(horizon_1h_metrics["fold_mean"]),
        "trigger_edge_fold_std_1h": float(horizon_1h_metrics["fold_std"]),
        "trigger_edge_shrunk_1h": float(horizon_1h_metrics["shrunk_delta"]),
        "trigger_edge_positive_fold_fraction_1h": float(horizon_1h_metrics["positive_fold_fraction"]),
        "trigger_edge_mean_3h": float(horizon_3h_metrics["mean_forward_return"]),
        "trigger_edge_delta_3h": float(horizon_3h_metrics["delta"]),
        "trigger_edge_fold_mean_3h": float(horizon_3h_metrics["fold_mean"]),
        "trigger_edge_fold_std_3h": float(horizon_3h_metrics["fold_std"]),
        "trigger_edge_shrunk_3h": float(horizon_3h_metrics["shrunk_delta"]),
        "trigger_edge_positive_fold_fraction_3h": float(horizon_3h_metrics["positive_fold_fraction"]),
        "positive_fold_fraction_r": positive_fold_fraction_r,
        "S_r": s_r,
        "D_r": d_r,
        "post_event_vol_dispersion": float(post_event_vol_dispersion),
        "dispersion_to_edge_ratio": float(d_r / max(abs(delta_r_shrunk), 1e-6)),
        "edge_to_dispersion_ratio": float(abs(delta_r_shrunk) / max(d_r, 1e-6)),
        "bars_to_mfe_mean": timing_metrics["bars_to_mfe_mean"],
        "bars_to_mae_mean": timing_metrics["bars_to_mae_mean"],
        "mfe_before_mae_fraction": timing_metrics["mfe_before_mae_fraction"],
        "prompt_excursion_quality": timing_metrics["prompt_excursion_quality"],
        "timing_precision_score": timing_metrics["timing_precision_score"],
        "trigger_gain_vs_parent": trigger_gain_vs_parent,
        "trigger_delta_dispersion_vs_parent": trigger_delta_dispersion_vs_parent,
        "trigger_delta_support_vs_parent": support_ratio,
        "trigger_delta_timing_vs_parent": trigger_delta_timing_vs_parent,
        "combined_sample_loss_vs_parent": combined_sample_loss_vs_parent,
        "keep_pct_vs_parent": float(100.0 * support_ratio),
        "rationale": f"trigger {trigger_template.trigger_template_name} support={support_ratio:.3f}; gain_vs_parent={trigger_gain_vs_parent:.4f}; timing_delta={trigger_delta_timing_vs_parent:.3f}",
        "complexity_tier": trigger_template.complexity_tier,
        "fold_metrics_json": _json_dumps_sorted(
            {
                "fold_event_counts": [int(x) for x in fold_counts],
                "fold_signed_returns": [float(x) for x in fold_returns],
                "fold_continuation_rates": [float(x) for x in fold_cont_rates],
            }
        ),
        "family": parent_regime_row.get("family"),
        "feature_base": parent_regime_row.get("feature_base", parent_regime_row.get("name")),
        "z_hours": parent_regime_row.get("z_hours"),
        "duration_hours": parent_regime_row.get("duration_hours"),
        "conditioner_mode": "none",
        "tier": 0,
        "parent_child_relation_type": "regime_trigger",
        "full_event_definition": f"{parent_regime_row.get('regime_id', parent_regime_row.get('name'))} AND {trigger_template.definition}",
        "support_ratio_vs_parent": support_ratio,
        "_event_mask": entry_mask,
    }
    raw_score, final_score = compute_trigger_score(pd.Series(row), config)
    row["trigger_score_raw"] = raw_score
    row["trigger_score_final"] = final_score
    row["shortlist_flag"] = final_score >= config.trigger_score_threshold
    row["non_dominated_flag"] = True
    return row


def discover_triggers_for_regime(
    parent_regime_row: pd.Series,
    parent_mask: np.ndarray,
    feature_frame: Dict[str, np.ndarray],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    signed_returns: np.ndarray,
    config: TriggerDiscoveryConfig,
    compute_full_metrics_fn: MetricFn,
    mode: str,
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    asset_groups: Dict[int, np.ndarray],
    templates: Optional[Sequence[TriggerTemplate]] = None,
    keep_fraction_override: Optional[float] = None,
) -> pd.DataFrame:
    parent_timing_metrics = compute_timing_metrics(
        event_mask=parent_mask,
        feature_frame=feature_frame,
        asset_groups=asset_groups,
        horizon_bars=max(config.trigger_timing_horizon_bars, 1),
        is_long=(mode == "long"),
    )
    parent_symbol_summary = _event_symbol_summary(parent_mask, feature_frame["symbol_codes"])
    parent_duration = _event_run_duration_stats(
        parent_mask,
        asset_groups,
        int(shared.get("bph", shared.get("bars_per_hour", 4))),
    )
    parent_period = _format_timestamp_bounds(
        np.asarray(feature_frame["timestamps"])[np.asarray(parent_mask, dtype=bool)]
    )
    tprint(
        "Phase 2.75 parent "
        f"({mode}) {parent_regime_row.get('regime_id', parent_regime_row.get('name'))}: "
        f"events={int(np.sum(parent_mask))} symbols={parent_symbol_summary['event_symbol_count']} "
        f"period={parent_period} avg_event_duration_h={parent_duration['avg_event_duration_hours']:.2f}"
    )
    parent_context = {
        "valid_fwd": np.isfinite(signed_returns),
        "fold_val_indices": [np.asarray(val_idx, dtype=np.int32) for _, val_idx in cv_splits],
        "parent_total_events": int(np.sum(parent_mask)),
    }
    prescreen_rows: List[Dict[str, Any]] = []
    for template in templates or generate_trigger_templates(config, parent_regime_row.to_dict()):
        precomputed = _prescreen_trigger_for_regime(
            parent_regime_row=parent_regime_row,
            parent_mask=parent_mask,
            trigger_template=template,
            feature_frame=feature_frame,
            cv_splits=cv_splits,
            signed_returns=signed_returns,
            config=config,
            mode=mode,
            shared=shared,
            asset_groups=asset_groups,
            parent_timing_metrics=parent_timing_metrics,
            parent_context=parent_context,
        )
        if precomputed is not None:
            prescreen_rows.append(precomputed)
    if not prescreen_rows:
        return pd.DataFrame()

    keep_fraction = float(
        np.clip(
            keep_fraction_override
            if keep_fraction_override is not None
            else config.ridge_prescreen_keep_fraction,
            0.0,
            1.0,
        )
    )
    prescreen_rows = _lgbm_rank_trigger_prescreen_rows(
        prescreen_rows=prescreen_rows,
        parent_mask=parent_mask,
        signed_returns=signed_returns,
        keep_fraction=keep_fraction,
        max_keep=int(config.ridge_prescreen_max_templates_per_parent),
    )
    tprint(
        f"Phase 2.75 ({mode}) lgbm prescreen "
        f"parent={parent_regime_row.get('regime_id', parent_regime_row.get('name'))} "
        f"kept={len(prescreen_rows)}"
    )
    prescreen_rows = _prune_trigger_prescreen_overlap(
        prescreen_rows,
        overlap_threshold=float(
            shared.get("runtime_cfg", {}).get("trigger_prescreen_overlap_threshold", 0.92)
        ),
    )

    rows: List[Dict[str, Any]] = []
    for precomputed in prescreen_rows:
        row = evaluate_trigger_for_regime(
            parent_regime_row=parent_regime_row,
            parent_mask=parent_mask,
            trigger_template=precomputed["trigger_template"],
            feature_frame=feature_frame,
            cv_splits=cv_splits,
            signed_returns=signed_returns,
            config=config,
            compute_full_metrics_fn=compute_full_metrics_fn,
            mode=mode,
            shared=shared,
            feature_dict=feature_dict,
            asset_groups=asset_groups,
            parent_timing_metrics=parent_timing_metrics,
            precomputed_context=precomputed,
            parent_context=parent_context,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("trigger_score_final", ascending=False)
    _tprint_trigger_table_support_summary("Phase 2.75 raw", mode, df)
    if config.keep_family_diversity:
        df = df.groupby("trigger_family", sort=False).head(config.max_triggers_per_family_per_parent).copy()
    df = prune_non_dominated_triggers(df, config)
    df = df[df["non_dominated_flag"]].copy()
    df = df.sort_values("trigger_score_final", ascending=False).head(config.top_k_triggers_per_regime).copy()
    df["shortlist_flag"] = True
    _tprint_trigger_table_support_summary("Phase 2.75 survivors", mode, df)
    return df


def run_trigger_discovery(
    phase2_survivors_df: pd.DataFrame,
    phase25_seeds_df: Optional[pd.DataFrame],
    parent_masks: Dict[str, np.ndarray],
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    signed_returns: np.ndarray,
    asset_groups: Dict[int, np.ndarray],
    config: TriggerDiscoveryConfig,
    compute_full_metrics_fn: MetricFn,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    del phase25_seeds_df
    if not config.enabled or phase2_survivors_df.empty:
        empty = pd.DataFrame()
        return empty, empty, {"entered_parent_regimes": 0, "survivor_count": 0}

    feature_frame = build_trigger_feature_frame(shared, asset_groups, feature_dict=feature_dict)
    templates = generate_trigger_templates(config)
    all_rows: List[pd.DataFrame] = []
    diagnostics: Dict[str, Any] = {
        "entered_parent_regimes": 0,
        "trigger_generation_counts": {},
        "survivor_counts": {},
        "candidate_masks": {},
    }
    parents = phase2_survivors_df.head(config.max_parent_regimes)
    keep_fraction_override: Optional[float] = None
    runtime_cfg = shared.get("runtime_cfg", {})
    if len(parents) >= int(runtime_cfg.get("trigger_large_parent_count_threshold", 6)):
        keep_fraction_override = float(runtime_cfg.get("trigger_large_parent_keep_fraction", 0.25))
    tprint(
        f"Phase 2.75 ({mode}) input: parents={len(parents)} "
        f"sample_period={_format_timestamp_bounds(np.asarray(feature_frame['timestamps']))}"
    )
    for _, parent_row in parents.iterrows():
        parent_id = str(parent_row.get("regime_id", parent_row.get("name")))
        parent_mask = parent_masks.get(parent_id)
        if parent_mask is None:
            continue
        diagnostics["entered_parent_regimes"] += 1
        diagnostics["trigger_generation_counts"][parent_id] = len(templates)
        survivors = discover_triggers_for_regime(
            parent_regime_row=parent_row,
            parent_mask=parent_mask,
            feature_frame=feature_frame,
            cv_splits=cv_splits,
            signed_returns=signed_returns,
            config=config,
            compute_full_metrics_fn=compute_full_metrics_fn,
            mode=mode,
            shared=shared,
            feature_dict=feature_dict,
            asset_groups=asset_groups,
            templates=templates,
            keep_fraction_override=keep_fraction_override,
        )
        diagnostics["survivor_counts"][parent_id] = int(survivors.shape[0])
        if not survivors.empty:
            for _, s_row in survivors.iterrows():
                diagnostics["candidate_masks"][str(s_row["name"])] = np.asarray(s_row["_event_mask"], dtype=bool)
            all_rows.append(survivors)

    if not all_rows:
        empty = pd.DataFrame()
        diagnostics["survivor_count"] = 0
        return empty, empty, diagnostics

    all_candidates = pd.concat(all_rows, ignore_index=True)
    survivors = all_candidates[all_candidates["shortlist_flag"]].copy()
    diagnostics["survivor_count"] = int(survivors.shape[0])
    return all_candidates, survivors, diagnostics
