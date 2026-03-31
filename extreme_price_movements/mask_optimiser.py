from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import traceback
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from extreme_price_movements.config import CFG, enable_perp_feature_keys, TEST_FEATURE_KEYS
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    load_features_selected,
    to_panel,
)
from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    SlicePlanner,
    SlicePlannerConfig,
)
from extreme_price_movements.utils import tprint
from extreme_price_movements.ridge_regime_event_assessment import (
    build_regime_features,
    RIDGE_FEATURE_COLS,
    fit_ridge_regime_scan_arrays,
    fit_lgbm_regime_scan_arrays,
)
from extreme_price_movements.intraday_crypto_library import (
    INTRADAY_TRIGGER_COLUMNS,
    LOCATION_FILTER_COLUMNS,
    build_intraday_crypto_library,
)
from extreme_price_movements.trigger_discovery import (
    TriggerDiscoveryConfig,
    build_trigger_feature_frame,
    run_trigger_discovery,
)

LOGGER = logging.getLogger(__name__)
_LOGGED_FAILURE_COUNTS: Dict[str, int] = {}
_TEMPORAL_FOLD_CACHE: Dict[Tuple[int, int, int, int, int, int], List[Tuple[np.ndarray, np.ndarray]]] = {}


def _log_bounded_warning(key: str, msg: str, limit: int = 3) -> None:
    c = _LOGGED_FAILURE_COUNTS.get(key, 0)
    if c < limit:
        LOGGER.warning(msg)
    _LOGGED_FAILURE_COUNTS[key] = c + 1


def _reports_dir_fallback() -> Path:
    return Path("reports")


def _append_symbol_concentration_log(line: str) -> None:
    try:
        path = _reports_dir_fallback()
        path.mkdir(parents=True, exist_ok=True)
        with (path / "mask_opt_symbol_concentration.log").open("a", encoding="utf-8") as fh:
            fh.write(line.rstrip() + "\n")
    except Exception:
        pass


def _persist_partial_table(df: Optional[pd.DataFrame], filename: str) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    try:
        path = _reports_dir_fallback()
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(path / filename, index=False)
    except Exception:
        pass


STAGE_LABELS: Dict[int, str] = {
    1: "Phase 1 (Location Filter)",
    2: "Phase 2 (Regime Filter)",
    3: "Phase 3 (Regime Attribution)",
    4: "Phase 4 (Trigger Discovery)",
    5: "Phase 5 (Conditioned Pattern Search)",
    6: "Phase 6 (Final Selection)",
}


def _stage_label(stage_num: int) -> str:
    return STAGE_LABELS.get(int(stage_num), f"Phase {int(stage_num)}")


def _stage_artifact_base_dir(cfg: Dict[str, Any]) -> Path:
    raw = str(
        cfg.get(
            "mask_opt_stage_artifact_dir",
            str(Path("reports") / "mask_optimiser_stage_runs"),
        )
    )
    return Path(raw)


def _mode_stage_dir(cfg: Dict[str, Any], mode: str) -> Path:
    return _stage_artifact_base_dir(cfg) / f"mode={mode}"


def _save_stage_artifacts(
    cfg: Dict[str, Any],
    mode: str,
    stage_num: int,
    payload: Dict[str, Any],
    tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    stage_dir = _mode_stage_dir(cfg, mode)
    stage_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = stage_dir / f"stage_{int(stage_num)}_bundle.pkl"
    with bundle_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "stage_num": int(stage_num),
        "stage_label": _stage_label(stage_num),
        "mode": str(mode),
        "bundle_path": str(bundle_path),
        "tables": [],
    }
    if tables:
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame):
                table_path = stage_dir / f"stage_{int(stage_num)}_{name}.csv"
                df.to_csv(table_path, index=False)
                manifest["tables"].append({"name": str(name), "path": str(table_path)})
    manifest_path = stage_dir / f"stage_{int(stage_num)}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_stage_artifacts(
    cfg: Dict[str, Any],
    mode: str,
    stage_num: int,
) -> Optional[Dict[str, Any]]:
    bundle_path = _mode_stage_dir(cfg, mode) / f"stage_{int(stage_num)}_bundle.pkl"
    if not bundle_path.exists():
        return None
    with bundle_path.open("rb") as fh:
        return pickle.load(fh)


def _stage_stop_result(
    *,
    stage_num: int,
    mode: str,
    candidate_table: pd.DataFrame,
    shortlist_table: Optional[pd.DataFrame] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": "ok",
        "reason": f"stopped_after_stage_{int(stage_num)}_{mode}",
        "layer0_candidate_table_": candidate_table.copy(),
        "layer0_shortlist_": shortlist_table.copy()
        if isinstance(shortlist_table, pd.DataFrame)
        else pd.DataFrame(),
        "layer0_basket_": [],
    }
    if extra:
        out.update(extra)
    return out


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


def _mask_run_duration_stats(
    mask: np.ndarray,
    asset_groups: Dict[int, np.ndarray],
    bph: int,
) -> Dict[str, float]:
    run_lengths: List[int] = []
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0:
        return {
            "avg_event_duration_bars": 0.0,
            "median_event_duration_bars": 0.0,
            "avg_event_duration_hours": 0.0,
            "median_event_duration_hours": 0.0,
            "event_run_count": 0.0,
        }
    for idxs in asset_groups.values():
        local = mask_bool[idxs]
        if local.size == 0 or not np.any(local):
            continue
        padded = np.concatenate(
            [
                np.asarray([False]),
                local.astype(bool, copy=False),
                np.asarray([False]),
            ]
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
    bars_per_hour = max(int(bph), 1)
    return {
        "avg_event_duration_bars": float(np.mean(runs)),
        "median_event_duration_bars": float(np.median(runs)),
        "avg_event_duration_hours": float(np.mean(runs) / bars_per_hour),
        "median_event_duration_hours": float(np.median(runs) / bars_per_hour),
        "event_run_count": float(runs.size),
    }


def _tprint_mask_support_summary(
    *,
    stage: str,
    mode: str,
    mask: np.ndarray,
    shared_like: Dict[str, Any],
    bph: int,
    note: str = "",
) -> None:
    mask_bool = np.asarray(mask, dtype=bool)
    total_rows = int(mask_bool.size)
    selected_rows = int(np.sum(mask_bool))
    symbol_codes = np.asarray(shared_like["symbol_codes"], dtype=np.int32)
    timestamps = np.asarray(shared_like["timestamps"])
    symbol_uniques = np.asarray(shared_like.get("symbol_uniques", np.array([], dtype=object)))
    selected_symbols = (
        int(np.unique(symbol_codes[mask_bool]).size) if selected_rows > 0 else 0
    )
    coverage = (
        float(selected_rows / max(total_rows, 1))
        if total_rows > 0
        else 0.0
    )
    period = _format_timestamp_bounds(timestamps[mask_bool] if selected_rows else timestamps)
    duration_stats = _mask_run_duration_stats(
        mask_bool,
        shared_like["asset_groups"],
        bph,
    )
    sample_note = f" | {note}" if note else ""
    parts = [
        f"{stage} ({mode}) support: rows={selected_rows}/{total_rows}",
        f"coverage={coverage:.3f}",
        f"symbols={selected_symbols}"
        f"{f'/{len(symbol_uniques)}' if symbol_uniques.size else ''}",
        f"period={period}",
    ]
    if selected_rows > 0 and selected_rows == total_rows:
        parts.append(
            "sample_span_h_per_symbol="
            f"{duration_stats['avg_event_duration_hours']:.2f}/"
            f"{duration_stats['median_event_duration_hours']:.2f} "
            "(avg/med)"
        )
    else:
        parts.append(
            f"avg_event_duration_h={duration_stats['avg_event_duration_hours']:.2f}"
        )
        parts.append(
            f"median_event_duration_h={duration_stats['median_event_duration_hours']:.2f}"
        )
        parts.append(f"runs={int(duration_stats['event_run_count'])}")
    tprint(" ".join(parts) + sample_note)


def _tprint_candidate_table_support_summary(
    stage: str,
    mode: str,
    df: pd.DataFrame,
) -> None:
    if df.empty:
        tprint(f"{stage} ({mode}) support: no candidates")
        return

    def _fmt(col: str) -> str:
        if col not in df.columns:
            return "n/a"
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            return "n/a"
        precision = 1
        if col == "top_symbol_share":
            precision = 3
        elif col in {"avg_event_duration_hours", "active_days_fraction"}:
            precision = 2
        return (
            f"{vals.min():.{precision}f}/"
            f"{vals.median():.{precision}f}/"
            f"{vals.max():.{precision}f}"
        )

    parts = [
        f"candidates={len(df)}",
        f"events(min/med/max)={_fmt('total_events')}",
    ]
    if "event_symbol_count" in df.columns:
        parts.append(f"symbols(min/med/max)={_fmt('event_symbol_count')}")
    if "top_symbol_share" in df.columns:
        parts.append(f"top_share(min/med/max)={_fmt('top_symbol_share')}")
    if "avg_event_duration_hours" in df.columns:
        parts.append(
            f"avg_duration_h(min/med/max)={_fmt('avg_event_duration_hours')}"
        )
    if "active_days_fraction" in df.columns:
        parts.append(
            f"active_days_frac(min/med/max)={_fmt('active_days_fraction')}"
        )
    if "keep_pct_vs_parent" in df.columns:
        parts.append(f"keep_pct_vs_parent(min/med/max)={_fmt('keep_pct_vs_parent')}")
    elif "support_ratio_vs_parent" in df.columns:
        keep_pct_vals = pd.to_numeric(df["support_ratio_vs_parent"], errors="coerce") * 100.0
        if keep_pct_vals.notna().any():
            parts.append(
                "keep_pct_vs_parent(min/med/max)="
                f"{keep_pct_vals.min():.1f}/{keep_pct_vals.median():.1f}/{keep_pct_vals.max():.1f}"
            )
    if "keep_pct_vs_original" in df.columns:
        parts.append(f"keep_pct_vs_original(min/med/max)={_fmt('keep_pct_vs_original')}")
    tprint(f"{stage} ({mode}) support: " + " | ".join(parts))


def _record_rejection_reason(reason_counts: Dict[str, int], reason: str) -> None:
    reason_counts[str(reason)] = int(reason_counts.get(str(reason), 0)) + 1


def _tprint_rejection_summary(stage: str, mode: str, reason_counts: Dict[str, int]) -> None:
    if not reason_counts:
        tprint(f"{stage} ({mode}) rejection reasons: none")
        return
    ordered = sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    text = ", ".join(f"{reason}:{count}" for reason, count in ordered)
    tprint(f"{stage} ({mode}) rejection reasons: {text}")


def _cap_stage_family_dominance(
    df: pd.DataFrame,
    *,
    score_col: str,
    stage: str,
    mode: str,
    max_per_family: int,
) -> pd.DataFrame:
    if df.empty or "family" not in df.columns or max_per_family <= 0:
        return df
    ranked = df.sort_values(score_col, ascending=False).copy()
    capped = ranked.groupby("family", sort=False).head(max_per_family).copy()
    removed = int(ranked.shape[0] - capped.shape[0])
    if removed > 0:
        tprint(
            f"{stage} ({mode}): capped family dominance to {max_per_family} per family, removed {removed} candidates"
        )
    return capped


def _ensure_min_family_representatives(
    df: pd.DataFrame,
    *,
    score_col: str,
    min_per_family: int,
    max_total: Optional[int] = None,
) -> pd.DataFrame:
    if (
        df.empty
        or "family" not in df.columns
        or min_per_family <= 0
    ):
        return df
    ranked = df.sort_values(score_col, ascending=False).copy()
    seeded = ranked.groupby("family", sort=False).head(min_per_family).copy()
    seeded_names = set(seeded["name"].astype(str).values) if "name" in seeded.columns else set()
    if max_total is not None and seeded.shape[0] >= max_total:
        return seeded.head(max_total).copy()
    if "name" in ranked.columns:
        rest = ranked[~ranked["name"].astype(str).isin(seeded_names)].copy()
    else:
        rest = ranked.iloc[0:0].copy()
    out = pd.concat([seeded, rest], ignore_index=True)
    if max_total is not None and out.shape[0] > max_total:
        out = out.head(max_total).copy()
    return out


def _tprint_family_feature_breakdown(stage: str, mode: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        tprint(f"{stage} ({mode}): 0 candidates")
        return
    family_counts = (
        df["family"].astype(str).value_counts().sort_index().to_dict()
        if "family" in df.columns
        else {}
    )
    feature_counts = (
        df["feature_base"].astype(str).value_counts().sort_index().to_dict()
        if "feature_base" in df.columns
        else {}
    )
    family_text = ", ".join(f"{k}:{v}" for k, v in family_counts.items()) or "n/a"
    feature_text = ", ".join(f"{k}:{v}" for k, v in feature_counts.items()) or "n/a"
    tprint(
        f"{stage} ({mode}): candidates={len(df)} | families={family_text} | feature_bases={feature_text}"
    )


def _cheap_support_stats(
    *,
    mask: np.ndarray,
    day_ids: np.ndarray,
    n_days: int,
    symbol_codes: np.ndarray,
    symbol_uniques: np.ndarray,
    timestamps: np.ndarray,
    asset_groups: Dict[int, np.ndarray],
    bph: int,
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    total_events = int(np.sum(mask_bool))
    active_days_fraction = active_days_fraction_nb(mask_bool, day_ids, n_days)
    symbol_summary = _mask_symbol_concentration_summary(
        mask_bool, symbol_codes, symbol_uniques
    )
    duration_stats = _mask_run_duration_stats(mask_bool, asset_groups, bph)
    ts_active = np.asarray(timestamps)[mask_bool]
    span_days = 0.0
    if ts_active.size > 1:
        ts_idx = pd.to_datetime(ts_active, unit="s", utc=True, errors="coerce")
        if np.all(pd.isna(ts_idx)):
            ts_idx = pd.to_datetime(ts_active, utc=True, errors="coerce")
        ts_idx = pd.DatetimeIndex(ts_idx).dropna()
        if not ts_idx.empty:
            span_days = float((ts_idx[-1] - ts_idx[0]) / pd.Timedelta(days=1))
    fold_event_counts: List[int] = []
    fold_symbol_counts: List[int] = []
    if folds is not None:
        for _, va in folds:
            fold_mask = mask_bool[va]
            fold_event_counts.append(int(np.sum(fold_mask)))
            if np.any(fold_mask):
                fold_symbol_counts.append(
                    int(np.unique(symbol_codes[va][fold_mask]).size)
                )
            else:
                fold_symbol_counts.append(0)
    return {
        "total_events": total_events,
        "active_days_fraction": float(active_days_fraction),
        "event_symbol_count": int(symbol_summary["event_symbol_count"]),
        "top_symbol_share": float(symbol_summary["top_symbol_share"]),
        "top_symbol_counts_text": str(symbol_summary["top_symbol_counts_text"]),
        "avg_event_duration_hours": float(duration_stats["avg_event_duration_hours"]),
        "median_event_duration_hours": float(duration_stats["median_event_duration_hours"]),
        "event_run_count": float(duration_stats["event_run_count"]),
        "span_days": float(span_days),
        "fold_event_counts": fold_event_counts,
        "fold_symbol_counts": fold_symbol_counts,
    }


def _passes_cheap_support_gate(
    *,
    stats: Dict[str, Any],
    cfg: Dict[str, Any],
    phase_prefix: str,
) -> Tuple[bool, str]:
    if int(stats["total_events"]) < int(cfg.get(f"{phase_prefix}_min_total_events", 0)):
        return False, "too_few_events"
    if float(stats["active_days_fraction"]) < float(
        cfg.get(f"{phase_prefix}_min_active_days_fraction", 0.0)
    ):
        return False, "active_days_too_low"
    if int(stats["event_symbol_count"]) < int(
        cfg.get(f"{phase_prefix}_min_distinct_symbols", 1)
    ):
        return False, "too_few_symbols"
    if float(stats["top_symbol_share"]) > float(
        cfg.get(f"{phase_prefix}_max_top_symbol_share", 1.0)
    ):
        return False, "top_symbol_share_too_high"
    if float(stats["span_days"]) < float(cfg.get(f"{phase_prefix}_min_span_days", 0.0)):
        return False, "span_days_too_low"
    fold_event_counts = np.asarray(stats.get("fold_event_counts", []), dtype=np.float32)
    if fold_event_counts.size > 0:
        if float(np.min(fold_event_counts)) < float(
            cfg.get(f"{phase_prefix}_min_fold_events", 0)
        ):
            return False, "min_fold_events_too_low"
        if float(np.mean(fold_event_counts)) < float(
            cfg.get(f"{phase_prefix}_min_mean_fold_events", 0)
        ):
            return False, "mean_fold_events_too_low"
    fold_symbol_counts = np.asarray(stats.get("fold_symbol_counts", []), dtype=np.float32)
    if fold_symbol_counts.size > 0 and float(np.min(fold_symbol_counts)) < float(
        cfg.get(f"{phase_prefix}_min_fold_symbols", 0)
    ):
        return False, "min_fold_symbols_too_low"
    return True, "ok"


def _prune_candidates_by_mask_overlap(
    df: pd.DataFrame,
    *,
    score_col: str,
    candidate_masks: Dict[str, np.ndarray],
    overlap_threshold: float,
) -> pd.DataFrame:
    if df.empty or not candidate_masks:
        return df
    kept_rows: List[int] = []
    kept_masks: List[np.ndarray] = []
    ordered = df.sort_values(score_col, ascending=False)
    for row_idx, row in ordered.iterrows():
        mask = candidate_masks.get(str(row["name"]))
        if mask is None:
            kept_rows.append(row_idx)
            continue
        mask_bool = np.asarray(mask, dtype=bool)
        is_duplicate = False
        for prev_mask in kept_masks:
            union = int(np.sum(mask_bool | prev_mask))
            if union <= 0:
                continue
            overlap = float(np.sum(mask_bool & prev_mask)) / float(union)
            if overlap >= overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_rows.append(row_idx)
            kept_masks.append(mask_bool)
    return df.loc[kept_rows].copy()


def _get_mode_cached_targets(
    shared: Dict[str, Any], mode: str, ret_threshold: float
) -> Dict[str, np.ndarray]:
    cache = shared.setdefault("_mode_target_cache", {})
    key = (str(mode), float(ret_threshold))
    if key not in cache:
        forward_returns = np.asarray(shared["forward_returns"], dtype=np.float32)
        signed_returns = _signed_mode_return(mode, forward_returns).astype(np.float32)
        cache[key] = {
            "signed_returns": signed_returns,
            "primary_target": _mode_primary_target(
                mode, forward_returns, ret_threshold
            ).astype(np.float32),
            "reversal_utility": (-signed_returns).astype(np.float32),
            "valid_forward": np.isfinite(forward_returns),
        }
    return cache[key]


def _prepare_candidate_design_bundle(
    *,
    mode: str,
    side_mask: np.ndarray,
    shared: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cache = _get_mode_design_cache(shared, mode, cfg)
    side_mask_bool = np.asarray(side_mask, dtype=bool)
    side_mask_valid = side_mask_bool[cache["valid_idx"]]
    idx_ne = np.where(~side_mask_valid)[0].astype(np.int32)
    idx_e = np.where(side_mask_valid)[0].astype(np.int32)
    max_samples_per_class = int(cfg.get("phase2_metric_max_samples_per_class", 25_000))
    idx_e_bal, idx_ne_bal = _balanced_sample_indices(
        idx_e, idx_ne, max_samples_per_class, seed=123
    )
    return {
        "learn_X": cache["learn_X_valid"],
        "timestamps": cache["timestamps_valid"],
        "symbol_codes": cache["symbol_codes_valid"],
        "y_primary": cache["y_primary_valid"],
        "signed_returns": cache["signed_returns_valid"],
        "reversal_utility": cache["reversal_utility_valid"],
        "valid_forward": np.ones(cache["valid_idx"].shape[0], dtype=bool),
        "valid_idx": cache["valid_idx"],
        "folds_valid": cache["folds_valid"],
        "side_mask_valid": side_mask_valid,
        "idx_e": idx_e,
        "idx_ne": idx_ne,
        "idx_e_bal": idx_e_bal,
        "idx_ne_bal": idx_ne_bal,
    }


def _get_mode_design_cache(
    shared: Dict[str, Any], mode: str, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    ret_threshold = float(cfg.get("phase1_ret_threshold", 0.0))
    cache = shared.setdefault("_mode_design_cache", {})
    key = (str(mode), float(ret_threshold))
    if key in cache:
        return cache[key]

    targets = _get_mode_cached_targets(shared, mode, ret_threshold)
    valid = np.asarray(targets["valid_forward"], dtype=bool)
    valid_idx = np.where(valid)[0].astype(np.int32)
    learn_X_valid = np.asarray(shared["learn_X"], dtype=np.float32)[valid_idx]
    timestamps_valid = np.asarray(shared["timestamps"])[valid_idx]
    symbol_codes_valid = np.asarray(shared["symbol_codes"], dtype=np.int32)[valid_idx]
    y_primary_valid = np.asarray(targets["primary_target"], dtype=np.float32)[valid_idx]
    signed_returns_valid = np.asarray(targets["signed_returns"], dtype=np.float32)[
        valid_idx
    ]
    reversal_utility_valid = np.asarray(
        targets["reversal_utility"], dtype=np.float32
    )[valid_idx]

    global_to_valid = np.full(valid.shape[0], -1, dtype=np.int32)
    global_to_valid[valid_idx] = np.arange(valid_idx.shape[0], dtype=np.int32)
    folds_valid: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr, va in shared["folds"]:
        tr_v = global_to_valid[tr]
        va_v = global_to_valid[va]
        tr_v = tr_v[tr_v >= 0].astype(np.int32)
        va_v = va_v[va_v >= 0].astype(np.int32)
        if tr_v.shape[0] == 0 or va_v.shape[0] == 0:
            continue
        folds_valid.append((tr_v, va_v))

    cache[key] = {
        "valid_idx": valid_idx,
        "learn_X_valid": learn_X_valid,
        "timestamps_valid": timestamps_valid,
        "symbol_codes_valid": symbol_codes_valid,
        "y_primary_valid": y_primary_valid,
        "signed_returns_valid": signed_returns_valid,
        "reversal_utility_valid": reversal_utility_valid,
        "folds_valid": folds_valid,
    }
    return cache[key]


def _get_valid_feature_items(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    valid_idx: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    cache_key = ("feature_dict_valid_float32_items", int(valid_idx.shape[0]))
    cache = shared.setdefault("_feature_design_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    feature_items = shared.get("feature_dict_float32_items")
    if feature_items is None:
        feature_items = [
            (fname, np.asarray(arr, dtype=np.float32))
            for fname, arr in feature_dict.items()
        ]
        shared["feature_dict_float32_items"] = feature_items

    valid_items = [(fname, arr[valid_idx]) for fname, arr in feature_items]
    cache[cache_key] = valid_items
    return valid_items


def _get_eval_design_cache(shared: Dict[str, Any]) -> Dict[str, Any]:
    cache = shared.setdefault("_eval_design_cache", None)
    if cache is not None:
        return cache

    close = np.asarray(shared["close"], dtype=np.float32)
    high = np.asarray(shared["high"], dtype=np.float32)
    low = np.asarray(shared["low"], dtype=np.float32)
    atr = np.asarray(shared["atr"], dtype=np.float32)
    mfe_atr = np.asarray(shared["mfe_atr"], dtype=np.float32)
    mae_atr = np.asarray(shared["mae_atr"], dtype=np.float32)
    eval_mask = (
        np.isfinite(close)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(atr)
        & np.isfinite(mfe_atr)
        & np.isfinite(mae_atr)
    )
    eval_idx = np.where(eval_mask)[0].astype(np.int32)
    eval_X = np.asarray(shared["learn_X"], dtype=np.float32)[eval_idx]
    eval_timestamps = np.asarray(shared["timestamps"])[eval_idx]
    eval_symbols = np.asarray(shared["symbol_codes"], dtype=np.int32)[eval_idx]

    global_to_eval = np.full(eval_mask.shape[0], -1, dtype=np.int32)
    global_to_eval[eval_idx] = np.arange(eval_idx.shape[0], dtype=np.int32)
    folds_eval: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr, va in shared["folds"]:
        tr_e = global_to_eval[tr]
        va_e = global_to_eval[va]
        tr_e = tr_e[tr_e >= 0].astype(np.int32)
        va_e = va_e[va_e >= 0].astype(np.int32)
        if tr_e.shape[0] == 0 or va_e.shape[0] == 0:
            continue
        folds_eval.append((tr_e, va_e))

    cache = {
        "eval_mask": eval_mask,
        "eval_idx": eval_idx,
        "eval_X": eval_X,
        "eval_timestamps": eval_timestamps,
        "eval_symbols": eval_symbols,
        "folds_eval": folds_eval,
    }
    shared["_eval_design_cache"] = cache
    return cache


def _adaptive_outer_fold_config(base_outer: Any, span_days: float) -> Any:
    span_hours = max(1.0, span_days * 24.0)
    train_h = max(12.0, span_hours * 0.50)
    valid_h = max(3.0, span_hours * 0.10)
    test_h = max(6.0, span_hours * 0.15)
    step_h = max(6.0, span_hours * 0.15)
    return base_outer.__class__(
        train_mode=base_outer.train_mode,
        train_span=pd.Timedelta(hours=train_h),
        valid_span=pd.Timedelta(hours=valid_h),
        test_span=pd.Timedelta(hours=test_h),
        step_span=pd.Timedelta(hours=step_h),
    )


# =============================================================================
# CONSTANTS
# =============================================================================

MODE_SHORT = "short"
MODE_LONG = "long"

ALL_MODES = [
    MODE_LONG,
    MODE_SHORT,
]


# Priority order for quote-currency deduplication.
_QUOTE_PRIORITY: list[str] = ["USDT", "USDC", "BUSD", "EUR"]


def _dedup_universe_by_base(symbols: list[str]) -> list[str]:
    """Return at most one symbol per base asset, preferring the highest-priority quote."""
    _KNOWN_QUOTES = set(_QUOTE_PRIORITY)

    def _parse(sym: str) -> tuple[str, str]:
        """Return (base, quote) parsed from any separator format."""
        clean = sym.replace("/", "").replace("_", "").upper()
        for q in sorted(_KNOWN_QUOTES, key=len, reverse=True):
            if clean.endswith(q) and len(clean) > len(q):
                return clean[: -len(q)], q
        return clean, ""  # unknown quote — treat as unique

    best: dict[str, tuple[int, str]] = {}  # base -> (priority_rank, original_sym)
    for sym in symbols:
        base, quote = _parse(sym)
        rank = (
            _QUOTE_PRIORITY.index(quote)
            if quote in _QUOTE_PRIORITY
            else len(_QUOTE_PRIORITY)
        )
        if base not in best or rank < best[base][0]:
            best[base] = (rank, sym)

    deduped = sorted(v for _, v in best.values())
    return deduped


# =============================================================================
# NUMBA KERNELS
# =============================================================================


@njit(cache=True)
def rolling_max_index_nb(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    out_val = np.full(n, np.nan, dtype=np.float32)
    out_idx = np.zeros(n, dtype=np.int32)
    if n == 0 or window <= 0:
        return out_val, out_idx

    deque_idx = np.zeros(n, dtype=np.int32)
    head = 0
    tail = 0

    for i in range(n):
        left = i - window + 1
        while head < tail and deque_idx[head] < left:
            head += 1

        v = x[i]
        if not np.isnan(v):
            while head < tail:
                j = deque_idx[tail - 1]
                vj = x[j]
                if np.isnan(vj) or vj <= v:
                    tail -= 1
                else:
                    break
            deque_idx[tail] = i
            tail += 1

        if head < tail:
            idx = deque_idx[head]
            out_idx[i] = idx
            out_val[i] = x[idx]

    return out_val, out_idx


@njit(cache=True)
def rolling_min_index_nb(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    out_val = np.full(n, np.nan, dtype=np.float32)
    out_idx = np.zeros(n, dtype=np.int32)
    if n == 0 or window <= 0:
        return out_val, out_idx

    deque_idx = np.zeros(n, dtype=np.int32)
    head = 0
    tail = 0

    for i in range(n):
        left = i - window + 1
        while head < tail and deque_idx[head] < left:
            head += 1

        v = x[i]
        if not np.isnan(v):
            while head < tail:
                j = deque_idx[tail - 1]
                vj = x[j]
                if np.isnan(vj) or vj >= v:
                    tail -= 1
                else:
                    break
            deque_idx[tail] = i
            tail += 1

        if head < tail:
            idx = deque_idx[head]
            out_idx[i] = idx
            out_val[i] = x[idx]

    return out_val, out_idx


@njit(cache=True)
def rolling_std_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(x.shape[0], np.nan, dtype=np.float32)
    n = x.shape[0]
    if n == 0 or window <= 0:
        return out

    sum_x = 0.0
    sum_sq = 0.0
    valid_count = 0

    for i in range(n):
        val = x[i]
        if not np.isnan(val):
            sum_x += val
            sum_sq += val * val
            valid_count += 1

        if i >= window:
            old_val = x[i - window]
            if not np.isnan(old_val):
                sum_x -= old_val
                sum_sq -= old_val * old_val
                valid_count -= 1

        if valid_count > 1:
            var = (sum_sq - (sum_x * sum_x) / valid_count) / (valid_count - 1)
            out[i] = np.sqrt(var) if var > 0 else 0.0
        elif valid_count == 1:
            out[i] = 0.0
    return out


@njit(cache=True)
def dilate_mask_by_groups_nb(
    mask: np.ndarray, group_indices: np.ndarray, duration_bars: int
) -> np.ndarray:
    out = mask.copy()
    if duration_bars <= 1:
        return out

    n_local = group_indices.shape[0]
    for local_i in range(n_local):
        gidx = group_indices[local_i]
        if mask[gidx]:
            end_local = min(n_local, local_i + duration_bars)
            for local_j in range(local_i + 1, end_local):
                out[group_indices[local_j]] = True
    return out


@njit(cache=True, fastmath=True)
def tbm_outcomes_atr_nb(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    tp_atr: float,
    sl_atr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    tp_first = np.zeros(n, dtype=np.int8)
    sl_first = np.zeros(n, dtype=np.int8)
    timeout = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        entry = close[i]
        atr_i = max(atr[i], 1e-9)

        tp_price = entry + tp_atr * atr_i
        sl_price = entry - sl_atr * atr_i

        for j in range(i + 1, i + horizon + 1):
            hi = high[j]
            lo = low[j]

            hit_tp = hi >= tp_price
            hit_sl = lo <= sl_price

            if hit_tp and not hit_sl:
                tp_first[i] = 1
                break

            if hit_sl and not hit_tp:
                sl_first[i] = 1
                break

            if hit_tp and hit_sl:
                median = 0.5 * (hi + lo)
                d_tp = abs(median - tp_price)
                d_sl = abs(median - sl_price)
                if d_tp < d_sl:
                    tp_first[i] = 1
                elif d_sl < d_tp:
                    sl_first[i] = 1
                else:
                    timeout[i] = 1
                break
        else:
            timeout[i] = 1

    return tp_first, sl_first, timeout


def dilate_mask_by_asset(
    mask: np.ndarray, asset_groups: Dict[int, np.ndarray], duration_bars: int
) -> np.ndarray:
    if duration_bars <= 1:
        return mask.copy()
    out = mask.copy()
    for _, idxs in asset_groups.items():
        if idxs.shape[0] == 0:
            continue
        out = dilate_mask_by_groups_nb(out, idxs.astype(np.int32), duration_bars)
    return out


@njit(cache=True)
def compute_impulse_coherence_nb(
    returns: np.ndarray,
    volatility: np.ndarray,
    high_val: np.ndarray,
    low_val: np.ndarray,
    start_px: np.ndarray,
    high_idx_local: np.ndarray,
    low_idx_local: np.ndarray,
    start_idx_local: np.ndarray,
    window: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    n = returns.shape[0]
    bars_to_peak_up = np.full(n, np.nan, dtype=np.float32)
    bars_to_peak_dn = np.full(n, np.nan, dtype=np.float32)
    speed_up = np.full(n, np.nan, dtype=np.float32)
    speed_dn = np.full(n, np.nan, dtype=np.float32)
    mono_up = np.full(n, np.nan, dtype=np.float32)
    mono_dn = np.full(n, np.nan, dtype=np.float32)
    vol_exp = np.full(n, np.nan, dtype=np.float32)

    pref_ret = np.zeros(n + 1, dtype=np.float32)
    pref_abs = np.zeros(n + 1, dtype=np.float32)
    for i in range(n):
        r = returns[i]
        if np.isnan(r):
            pref_ret[i + 1] = pref_ret[i]
            pref_abs[i + 1] = pref_abs[i]
        else:
            pref_ret[i + 1] = pref_ret[i] + r
            pref_abs[i + 1] = pref_abs[i] + abs(r)

    for i in range(window, n):
        st = start_idx_local[i]
        st_px = start_px[i]

        peak_h = high_idx_local[i]
        peak_l = low_idx_local[i]

        b_up = peak_h - st
        b_dn = peak_l - st

        bars_to_peak_up[i] = b_up
        bars_to_peak_dn[i] = b_dn

        imp_up = (high_val[i] - st_px) / st_px if st_px > 1e-9 else 0.0
        imp_dn = (st_px - low_val[i]) / st_px if st_px > 1e-9 else 0.0

        speed_up[i] = imp_up / max(1.0, b_up)
        speed_dn[i] = imp_dn / max(1.0, b_dn)

        up_left = min(max(st + 1, 0), n)
        up_right = min(max(peak_h + 1, up_left), n)
        dir_sum_up = pref_ret[up_right] - pref_ret[up_left]
        abs_sum_up = pref_abs[up_right] - pref_abs[up_left]
        mono_up[i] = dir_sum_up / abs_sum_up if abs_sum_up > 1e-9 else 0.0

        dn_left = min(max(st + 1, 0), n)
        dn_right = min(max(peak_l + 1, dn_left), n)
        dir_sum_dn = -(pref_ret[dn_right] - pref_ret[dn_left])
        abs_sum_dn = pref_abs[dn_right] - pref_abs[dn_left]
        mono_dn[i] = dir_sum_dn / abs_sum_dn if abs_sum_dn > 1e-9 else 0.0

        pre_vol = volatility[st]
        post_vol = volatility[i]
        vol_exp[i] = post_vol / pre_vol if pre_vol > 1e-9 else 1.0

    return (
        bars_to_peak_up,
        bars_to_peak_dn,
        speed_up,
        speed_dn,
        mono_up,
        mono_dn,
        vol_exp,
    )


def rolling_max_index_safe(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    out_val = np.full(n, np.nan, dtype=np.float32)
    out_idx = np.zeros(n, dtype=np.int32)
    if n == 0 or window <= 0:
        return out_val, out_idx
    for i in range(n):
        left = max(0, i - window + 1)
        sl = x[left : i + 1]
        valid_local = np.where(~np.isnan(sl))[0]
        if valid_local.shape[0] == 0:
            continue
        best_local = valid_local[int(np.argmax(sl[valid_local]))]
        best_idx = left + best_local
        out_idx[i] = best_idx
        out_val[i] = x[best_idx]
    return out_val, out_idx


def rolling_min_index_safe(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    out_val = np.full(n, np.nan, dtype=np.float32)
    out_idx = np.zeros(n, dtype=np.int32)
    if n == 0 or window <= 0:
        return out_val, out_idx
    for i in range(n):
        left = max(0, i - window + 1)
        sl = x[left : i + 1]
        valid_local = np.where(~np.isnan(sl))[0]
        if valid_local.shape[0] == 0:
            continue
        best_local = valid_local[int(np.argmin(sl[valid_local]))]
        best_idx = left + best_local
        out_idx[i] = best_idx
        out_val[i] = x[best_idx]
    return out_val, out_idx


def rolling_std_safe(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or window <= 0:
        return out
    for i in range(n):
        left = max(0, i - window + 1)
        sl = x[left : i + 1]
        sl = sl[np.isfinite(sl)]
        if sl.shape[0] > 1:
            out[i] = np.float32(np.std(sl, ddof=1))
        elif sl.shape[0] == 1:
            out[i] = 0.0
    return out


def compute_impulse_coherence_safe(
    returns: np.ndarray,
    volatility: np.ndarray,
    high_val: np.ndarray,
    low_val: np.ndarray,
    start_px: np.ndarray,
    high_idx_local: np.ndarray,
    low_idx_local: np.ndarray,
    start_idx_local: np.ndarray,
    window: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    n = returns.shape[0]
    bars_to_peak_up = np.full(n, np.nan, dtype=np.float32)
    bars_to_peak_dn = np.full(n, np.nan, dtype=np.float32)
    speed_up = np.full(n, np.nan, dtype=np.float32)
    speed_dn = np.full(n, np.nan, dtype=np.float32)
    mono_up = np.full(n, np.nan, dtype=np.float32)
    mono_dn = np.full(n, np.nan, dtype=np.float32)
    vol_exp = np.full(n, np.nan, dtype=np.float32)

    pref_ret = np.zeros(n + 1, dtype=np.float32)
    pref_abs = np.zeros(n + 1, dtype=np.float32)
    for i in range(n):
        r = returns[i]
        if np.isnan(r):
            pref_ret[i + 1] = pref_ret[i]
            pref_abs[i + 1] = pref_abs[i]
        else:
            pref_ret[i + 1] = pref_ret[i] + r
            pref_abs[i + 1] = pref_abs[i] + abs(r)

    for i in range(window, n):
        st = int(start_idx_local[i])
        st_px = float(start_px[i])
        peak_h = int(high_idx_local[i])
        peak_l = int(low_idx_local[i])
        b_up = peak_h - st
        b_dn = peak_l - st
        bars_to_peak_up[i] = b_up
        bars_to_peak_dn[i] = b_dn

        imp_up = (high_val[i] - st_px) / st_px if st_px > 1e-9 else 0.0
        imp_dn = (st_px - low_val[i]) / st_px if st_px > 1e-9 else 0.0
        speed_up[i] = imp_up / max(1.0, float(b_up))
        speed_dn[i] = imp_dn / max(1.0, float(b_dn))

        up_left = min(max(st + 1, 0), n)
        up_right = min(max(peak_h + 1, up_left), n)
        dir_sum_up = pref_ret[up_right] - pref_ret[up_left]
        abs_sum_up = pref_abs[up_right] - pref_abs[up_left]
        mono_up[i] = dir_sum_up / abs_sum_up if abs_sum_up > 1e-9 else 0.0

        dn_left = min(max(st + 1, 0), n)
        dn_right = min(max(peak_l + 1, dn_left), n)
        dir_sum_dn = -(pref_ret[dn_right] - pref_ret[dn_left])
        abs_sum_dn = pref_abs[dn_right] - pref_abs[dn_left]
        mono_dn[i] = dir_sum_dn / abs_sum_dn if abs_sum_dn > 1e-9 else 0.0

        pre_vol = volatility[st]
        post_vol = volatility[i]
        vol_exp[i] = post_vol / pre_vol if pre_vol > 1e-9 else 1.0

    return (
        bars_to_peak_up,
        bars_to_peak_dn,
        speed_up,
        speed_dn,
        mono_up,
        mono_dn,
        vol_exp,
    )


def dilate_mask_by_asset_safe(
    mask: np.ndarray, asset_groups: Dict[int, np.ndarray], duration_bars: int
) -> np.ndarray:
    if duration_bars <= 1:
        return mask.copy()
    out = mask.copy()
    for idxs in asset_groups.values():
        if idxs.shape[0] == 0:
            continue
        local_hits = np.where(mask[idxs])[0]
        for local_i in local_hits:
            end_local = min(idxs.shape[0], local_i + duration_bars)
            out[idxs[local_i + 1 : end_local]] = True
    return out


@njit(cache=True)
def active_days_fraction_nb(
    mask: np.ndarray, day_ids: np.ndarray, n_days: int
) -> float:
    if n_days <= 0:
        return 0.0
    seen = np.zeros(n_days, dtype=np.uint8)
    n = mask.shape[0]
    for i in range(n):
        if mask[i]:
            seen[day_ids[i]] = 1
    return float(np.sum(seen)) / float(n_days)


@njit(cache=True)
def daily_event_stats_nb(
    mask: np.ndarray, day_ids: np.ndarray, n_days: int
) -> Tuple[float, float]:
    counts = np.zeros(n_days, dtype=np.int32)
    n = mask.shape[0]
    for i in range(n):
        if mask[i]:
            counts[day_ids[i]] += 1

    active_days = 0
    total = 0.0
    for d in range(n_days):
        if counts[d] > 0:
            active_days += 1
        total += counts[d]

    mean = total / max(1, n_days)

    var = 0.0
    for d in range(n_days):
        diff = counts[d] - mean
        var += diff * diff
    std = np.sqrt(var / max(1, n_days))
    return float(mean), float(std)


@njit(cache=True)
def fold_base_rate_nb(
    mask: np.ndarray, target: np.ndarray, val_idx: np.ndarray
) -> float:
    total = 0
    pos = 0
    for k in range(val_idx.shape[0]):
        i = val_idx[k]
        if mask[i] and not np.isnan(target[i]):
            total += 1
            pos += target[i]
    if total == 0:
        return 0.0
    return float(pos) / float(total)


@njit(cache=True)
def simple_mask_count_nb(mask: np.ndarray) -> int:
    return int(np.sum(mask))


def active_days_fraction_safe(
    mask: np.ndarray, day_ids: np.ndarray, n_days: int
) -> float:
    if n_days <= 0:
        return 0.0
    if mask.shape[0] == 0:
        return 0.0
    active_days = np.unique(day_ids[mask])
    return float(active_days.shape[0]) / float(n_days)


def daily_event_stats_safe(
    mask: np.ndarray, day_ids: np.ndarray, n_days: int
) -> Tuple[float, float]:
    if n_days <= 0:
        return 0.0, 0.0
    counts = np.zeros(n_days, dtype=np.int32)
    if np.any(mask):
        vals, freqs = np.unique(day_ids[mask], return_counts=True)
        counts[vals.astype(np.int32)] = freqs.astype(np.int32)
    return float(np.mean(counts)), float(np.std(counts))


def fold_base_rate_safe(
    mask: np.ndarray, target: np.ndarray, val_idx: np.ndarray
) -> float:
    if val_idx.shape[0] == 0:
        return 0.0
    valid = mask[val_idx] & np.isfinite(target[val_idx])
    if not np.any(valid):
        return 0.0
    return float(np.mean(target[val_idx][valid]))


def simple_mask_count_safe(mask: np.ndarray) -> int:
    return int(np.sum(mask))


@njit(cache=True)
def safe_mean_nb(x: np.ndarray) -> float:
    if x.shape[0] == 0:
        return 0.0
    s = 0.0
    n = 0
    for i in range(x.shape[0]):
        v = x[i]
        if not np.isnan(v):
            s += v
            n += 1
    return s / max(1, n)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CandidateKey:
    family: str
    z_hours: int
    param: str
    duration_hours: int

    def as_str(self) -> str:
        # User requested to remove d=1 or other duration suffix from naming
        return f"{self.family}|z={self.z_hours}|p={self.param}"


# =============================================================================
# HELPERS
# =============================================================================


def _mode_is_up(mode: str) -> bool:
    return mode in {MODE_LONG, "price_up_tf", "price_up_mr"}


def _mode_is_tf(mode: str) -> bool:
    return mode in {MODE_LONG, "price_up_tf", "price_down_tf"}


def _get_side_mask(mode: str, m_high: np.ndarray, m_low: np.ndarray) -> np.ndarray:
    return m_high if _mode_is_up(mode) else m_low


def _phase3_parent_mode(cfg: Dict[str, Any]) -> str:
    if bool(cfg.get("enable_trigger_discovery_stage", True)):
        return str(cfg.get("phase3_parent_mode", "regime_trigger"))
    return "base_regime"


def _phase3_parent_seed_key(row: pd.Series | Dict[str, Any]) -> str:
    if isinstance(row, pd.Series):
        parent_regime_id = row.get("parent_regime_id")
        name = row.get("name")
    else:
        parent_regime_id = row.get("parent_regime_id")
        name = row.get("name")
    return str(parent_regime_id if isinstance(parent_regime_id, str) and parent_regime_id else name)


def _phase3_parent_relation_type(tier: int, has_trigger_parent: bool) -> str:
    if tier <= 0:
        return "regime_trigger" if has_trigger_parent else "base_regime"
    if tier == 1:
        return "regime_trigger_conditioner" if has_trigger_parent else "regime_conditioner"
    return "regime_trigger_conditioner_conditioner" if has_trigger_parent else "regime_conditioner_conditioner"


def _ensure_min_feature_representatives(
    df: pd.DataFrame,
    *,
    score_col: str,
    min_per_feature: int,
    max_total: Optional[int] = None,
) -> pd.DataFrame:
    if df.empty or "feature_base" not in df.columns or min_per_feature <= 0:
        return df

    ranked = df.sort_values(score_col, ascending=False).copy()
    floor_rows = ranked.groupby("feature_base", sort=False).head(min_per_feature).copy()
    if max_total is None or floor_rows.shape[0] >= max_total:
        return floor_rows.sort_values(score_col, ascending=False).copy()

    keep_names = set(floor_rows["name"].astype(str).tolist())
    ordered_rows: List[Dict[str, Any]] = floor_rows.to_dict("records")
    for row in ranked.to_dict("records"):
        if len(ordered_rows) >= max_total:
            break
        name = str(row.get("name"))
        if name in keep_names:
            continue
        ordered_rows.append(row)
        keep_names.add(name)
    return pd.DataFrame(ordered_rows).sort_values(score_col, ascending=False).copy()


def _mode_primary_target(
    mode: str, forward_returns: np.ndarray, ret_threshold: float
) -> np.ndarray:
    # 1 = desired outcome for that mode
    valid = np.isfinite(forward_returns)
    if mode in {MODE_LONG, "price_up_tf", "price_up_mr"}:
        # Long seeks positive moves
        out = (forward_returns > ret_threshold).astype(np.float32)
    elif mode in {MODE_SHORT, "price_down_tf", "price_down_mr"}:
        # Short seeks negative moves
        out = (forward_returns < -ret_threshold).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out[~valid] = np.nan
    return out


def _signed_mode_return(mode: str, forward_returns: np.ndarray) -> np.ndarray:
    # Positive = good for the mode
    valid = np.isfinite(forward_returns)
    if mode in {MODE_LONG, "price_up_tf", "price_up_mr"}:
        out = forward_returns.astype(np.float32)
    elif mode in {MODE_SHORT, "price_down_tf", "price_down_mr"}:
        out = (-forward_returns).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out[~valid] = np.nan
    return out


def _resolve_path(path: str) -> str:
    if not path:
        return path
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(pkg_root, path))


def _find_latest_feature_dir(data_root: str) -> Optional[str]:
    feat_dir = os.path.join(data_root, "features")
    if not os.path.isdir(feat_dir):
        return None
    dirs = sorted(glob.glob(os.path.join(feat_dir, "20*")))
    return dirs[-1] if dirs else None


def _rng_sample_half(items: List[Any], seed: int = 42) -> List[Any]:
    if len(items) <= 1:
        return items[:]
    rng = random.Random(seed)
    k = max(1, len(items) // 2)
    return rng.sample(items, k)


def _rng_sample_fraction(items: List[Any], frac: float, seed: int = 42) -> List[Any]:
    if len(items) <= 1:
        return items[:]
    frac = min(max(float(frac), 0.0), 1.0)
    if frac >= 0.999:
        return items[:]
    rng = random.Random(seed)
    k = max(1, int(round(len(items) * frac)))
    k = min(k, len(items))
    return rng.sample(items, k)


def _zscore_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - m) / s).astype(np.float32)


def _metric_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_abs_ratio(numerator: float, denominator: float) -> float:
    num = _metric_or_nan(numerator)
    den = _metric_or_nan(denominator)
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    return float(num / max(abs(den), 1e-6))


def _log_stage_snapshot(
    mode: str,
    stage: str,
    df: pd.DataFrame,
    sort_col: str,
    cols: List[str],
    top_n: int = 5,
) -> None:
    if df.empty:
        tprint(f"{stage} ({mode}): no candidates")
        return
    use_cols = [c for c in cols if c in df.columns]
    snap = df.sort_values(sort_col, ascending=False).head(top_n)[use_cols]
    tprint(f"{stage} ({mode}) top {min(top_n, len(snap))} by {sort_col}:")
    tprint(snap.to_string(index=False))


def _coherence_metrics_single_side(
    mask: np.ndarray,
    bars_to_peak: np.ndarray,
    speed: np.ndarray,
    mono: np.ndarray,
) -> Dict[str, float]:
    valid = mask & np.isfinite(bars_to_peak) & np.isfinite(speed) & np.isfinite(mono)
    if not np.any(valid):
        return {
            "bars_to_peak_dispersion": 1e9,
            "speed_dispersion": 1e9,
            "monotonicity_dispersion": 1e9,
            "impulse_shape_dispersion": 1e9,
        }
    bp = float(np.std(bars_to_peak[valid])) if np.sum(valid) > 1 else 0.0
    sp = float(np.std(speed[valid])) if np.sum(valid) > 1 else 0.0
    mo = float(np.std(mono[valid])) if np.sum(valid) > 1 else 0.0
    return {
        "bars_to_peak_dispersion": bp,
        "speed_dispersion": sp,
        "monotonicity_dispersion": mo,
        "impulse_shape_dispersion": bp + sp + mo,
    }


def _compute_regime_distinctness_single_side(
    side_mask: np.ndarray,
    mode: str,
    forward_returns: np.ndarray,
    mae_high: np.ndarray,
    mfe_high: np.ndarray,
    mae_low: np.ndarray,
    mfe_low: np.ndarray,
) -> float:
    if not np.any(side_mask):
        return 0.0

    valid = np.isfinite(forward_returns)
    ret_g = _signed_mode_return(mode, forward_returns[valid])
    ret_e = _signed_mode_return(mode, forward_returns[valid & side_mask])

    if ret_g.shape[0] < 10 or ret_e.shape[0] < 10:
        return 0.0

    std_g = np.std(ret_g)
    std_e = np.std(ret_e)
    std_ratio = std_e / std_g if std_g > 1e-9 else 1.0

    t_upper = np.percentile(ret_g, 95)
    t_lower = np.percentile(ret_g, 5)
    tail_g = np.mean((ret_g >= t_upper) | (ret_g <= t_lower))
    tail_e = np.mean((ret_e >= t_upper) | (ret_e <= t_lower))
    tail_ratio = tail_e / tail_g if tail_g > 1e-9 else 1.0

    if _mode_is_up(mode):
        mae_arr = mae_high
        mfe_arr = mfe_high
    else:
        mae_arr = mae_low
        mfe_arr = mfe_low

    mae_g = float(np.nanmean(mae_arr[valid])) if np.any(valid) else 1.0
    mae_e = (
        float(np.nanmean(mae_arr[valid & side_mask]))
        if np.any(valid & side_mask)
        else mae_g
    )
    mae_ratio = mae_e / mae_g if mae_g > 1e-9 else 1.0

    mfe_g = float(np.nanmean(mfe_arr[valid])) if np.any(valid) else 1.0
    mfe_e = (
        float(np.nanmean(mfe_arr[valid & side_mask]))
        if np.any(valid & side_mask)
        else mfe_g
    )
    mfe_ratio = mfe_e / mfe_g if mfe_g > 1e-9 else 1.0

    return float(
        np.mean(np.clip([std_ratio, tail_ratio, mae_ratio, mfe_ratio], 0.0, 5.0))
    )


def _build_temporal_folds(
    timestamps: np.ndarray,
    n_samples: int,
    n_splits: int = 4,
    symbols: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Delegate fold construction to periods_symbols_management as single source of truth."""
    if n_samples < 10:
        return []
    ts_arr = np.asarray(timestamps)
    sym_arr = np.asarray(symbols) if symbols is not None else None
    cache_key = (
        int(ts_arr.__array_interface__["data"][0]) if ts_arr.size else 0,
        int(ts_arr.shape[0]),
        int(n_splits),
        int(sym_arr.__array_interface__["data"][0]) if sym_arr is not None and sym_arr.size else 0,
        int(sym_arr.shape[0]) if sym_arr is not None else 0,
        int(sym_arr.dtype.num) if sym_arr is not None else -1,
    )
    cached = _TEMPORAL_FOLD_CACHE.get(cache_key)
    if cached is not None:
        return [(tr.copy(), va.copy()) for tr, va in cached]
    try:
        ts = pd.to_datetime(ts_arr, unit="s", utc=True, errors="coerce")
        if np.all(pd.isna(ts)):
            ts = pd.to_datetime(ts_arr, utc=True, errors="coerce")
        ts = pd.Series(pd.DatetimeIndex(ts)).ffill().bfill()
        span_days = max(
            1.0,
            float((ts.max() - ts.min()) / pd.Timedelta(days=1))
            if ts.notna().any()
            else 1.0,
        )
        events = pd.DataFrame(
            {
                "event_id": np.arange(n_samples, dtype=np.int64),
                "symbol": (
                    np.asarray(symbols).astype(str, copy=False)
                    if symbols is not None
                    else np.repeat("ALL", n_samples)
                ),
                "t0": ts.to_numpy(),
                "t1": (ts + pd.Timedelta(seconds=1)).to_numpy(),
            }
        )
        effective_symbols = int(pd.Series(events["symbol"]).nunique())
        if effective_symbols <= 1:
            _log_bounded_warning(
                "planner_fold_symbol_collapse",
                f"Temporal fold planner received {effective_symbols} effective symbol(s) for {n_samples} samples.",
                limit=20,
            )
        cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
        cfg = dc_replace(
            cfg,
            silent=True,
            preset=dc_replace(
                cfg.preset,
                outer=_adaptive_outer_fold_config(cfg.preset.outer, span_days),
                inner=dc_replace(cfg.preset.inner, n_splits=max(1, int(n_splits))),
                sampling=dc_replace(
                    cfg.preset.sampling,
                    mode="full",
                    event_fraction=1.0,
                    symbol_fraction=1.0,
                ),
                symbol_policy=dc_replace(
                    cfg.preset.symbol_policy,
                    mode="all_symbols",
                    subset_fraction=1.0,
                    min_symbols_per_split=1,
                ),
            ),
        )
        bundle = SlicePlanner(cfg).build(events)
        plans = bundle["consumer_plans"]["regime_search"]
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for plan in plans:
            tr = np.asarray(plan.fit_idx, dtype=np.int32)
            va = np.asarray(plan.predict_idx, dtype=np.int32)
            if tr.size > 0 and va.size > 0:
                folds.append((tr, va))
        if folds:
            _TEMPORAL_FOLD_CACHE[cache_key] = [(tr.copy(), va.copy()) for tr, va in folds]
            return folds
        raise ValueError(
            f"SlicePlanner failed to generate {n_splits} temporal folds from {n_samples} samples. "
            "Ensure timestamps are valid and sufficient data exists."
        )
    except Exception as e:
        _log_bounded_warning(
            "planner_fold_fallback",
            f"Planner fold delegation failed; falling back to PurgedKFold: {e}",
            limit=10,
        )
    try:
        cv = PurgedKFold(
            n_splits=n_splits, purge=43200, embargo=43200, times=timestamps
        )
        dummy = np.empty((n_samples, 1), dtype=np.float32)
        folds = list(cv.split(dummy))
        if folds:
            out = [(tr.astype(np.int32), va.astype(np.int32)) for tr, va in folds]
            _TEMPORAL_FOLD_CACHE[cache_key] = [(tr.copy(), va.copy()) for tr, va in out]
            return out
    except Exception:
        if n_samples < 2:
            return []
        uniq_ts = np.unique(np.asarray(timestamps))
        if uniq_ts.shape[0] < 2:
            return []
        mid_ts = uniq_ts.shape[0] // 2
        train_mask = np.isin(timestamps, uniq_ts[:mid_ts])
        valid_mask = np.isin(timestamps, uniq_ts[mid_ts:])
        return [
            (
                np.flatnonzero(train_mask).astype(np.int32),
                np.flatnonzero(valid_mask).astype(np.int32),
            )
        ]
    if n_samples < 2:
        return []
    uniq_ts = np.unique(np.asarray(timestamps))
    if uniq_ts.shape[0] < 2:
        return []
    mid_ts = uniq_ts.shape[0] // 2
    train_mask = np.isin(timestamps, uniq_ts[:mid_ts])
    valid_mask = np.isin(timestamps, uniq_ts[mid_ts:])
    out = [
        (
            np.flatnonzero(train_mask).astype(np.int32),
            np.flatnonzero(valid_mask).astype(np.int32),
        )
    ]
    _TEMPORAL_FOLD_CACHE[cache_key] = [(tr.copy(), va.copy()) for tr, va in out]
    return out


def _mask_symbol_concentration_summary(
    mask: np.ndarray,
    symbol_codes: np.ndarray,
    symbol_uniques: np.ndarray,
    max_symbols: int = 5,
) -> Dict[str, Any]:
    active_codes = np.asarray(symbol_codes[mask], dtype=np.int32)
    if active_codes.size == 0:
        return {
            "event_symbol_count": 0,
            "top_symbol_share": 0.0,
            "top_symbol_counts_text": "",
        }
    uniq_codes, counts = np.unique(active_codes, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_parts: List[str] = []
    for idx in order[: max(1, int(max_symbols))]:
        code = int(uniq_codes[idx])
        label = (
            str(symbol_uniques[code])
            if 0 <= code < len(symbol_uniques)
            else str(code)
        )
        top_parts.append(f"{label}:{int(counts[idx])}")
    return {
        "event_symbol_count": int(uniq_codes.size),
        "top_symbol_share": float(np.max(counts) / max(active_codes.size, 1)),
        "top_symbol_counts_text": ", ".join(top_parts),
    }


def _apply_regime_search_slice_plan(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    lookback_years: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    """Use periods_symbols_management regime_search plans to define the optimizer sample."""
    if data.empty:
        return data, feature_dict, forward_returns
    try:
        ts = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        symbol_counts_in = data["symbol"].astype(str).value_counts()
        top_symbols_in = ", ".join(
            f"{sym}:{int(cnt)}" for sym, cnt in symbol_counts_in.head(8).items()
        )
        tprint(
            "regime_search slice input: "
            f"rows={data.shape[0]} symbols={symbol_counts_in.shape[0]} "
            f"days={pd.Series(ts).dt.normalize().nunique()} top={top_symbols_in}"
        )
        span_days = max(
            1.0,
            float((ts.max() - ts.min()) / pd.Timedelta(days=1))
            if ts.notna().any()
            else 1.0,
        )
        events = pd.DataFrame(
            {
                "event_id": np.arange(data.shape[0], dtype=np.int64),
                "timestamp": data["timestamp"].to_numpy(),
                "symbol": data["symbol"].astype(str).values,
                "open": data["open"].to_numpy() if "open" in data.columns else np.full(data.shape[0], np.nan, dtype=np.float32),
                "close": data["close"].to_numpy(),
                "high": data["high"].to_numpy(),
                "low": data["low"].to_numpy(),
                "t0": ts.to_numpy(),
                "t1": (ts + pd.Timedelta(seconds=1)).to_numpy(),
            }
        )
        cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
        cfg = dc_replace(
            cfg,
            preset=dc_replace(
                cfg.preset,
                outer=_adaptive_outer_fold_config(cfg.preset.outer, span_days),
                sampling=dc_replace(
                    cfg.preset.sampling,
                    max_symbols=200,
                    symbol_fraction=1.0,
                ),
                symbol_policy=dc_replace(
                    cfg.preset.symbol_policy,
                    mode="subset_symbols",
                    subset_fraction=0.20,
                ),
            ),
            consumer_overrides={
                **dict(cfg.consumer_overrides),
                "full_inference_lookback_years": float(lookback_years),
                "mask_opt_max_rows": 600_000,
                "phase1_min_subsample_rows": 300_000,
                "mask_opt_deep_rows": 1_600_000,
                "mask_opt_pre_slice_max_rows": 1_000_000,
                "shortlist_max_candidates": 12,
                "stage3_max_candidates": 12,
                "final_top_k_for_diagnostics": 6,
            },
            silent=True,
            min_rows_per_fold=1,
            min_symbols_per_fold=1,
        )
        bundle = SlicePlanner(cfg).build(events)
        consumer_plans = bundle["consumer_plans"]
        plans = consumer_plans.get("regime_search", [])
        idx_parts: List[np.ndarray] = []
        for plan in plans:
            if plan.fit_idx.size > 0:
                idx_parts.append(np.asarray(plan.fit_idx, dtype=np.int64))
            if plan.predict_idx.size > 0:
                idx_parts.append(np.asarray(plan.predict_idx, dtype=np.int64))
        if not idx_parts:
            tprint("periods/symbols regime_search slice plan produced no rows; using capped sample")
            return data, feature_dict, forward_returns
        idx = np.unique(np.concatenate(idx_parts)).astype(np.int64)
        idx.sort()
        tprint(
            "Applied periods/symbols regime_search slice plan: "
            f"rows={idx.size}/{data.shape[0]} symbols={data.iloc[idx]['symbol'].nunique()}"
        )
        symbol_counts_out = data.iloc[idx]["symbol"].astype(str).value_counts()
        top_symbols_out = ", ".join(
            f"{sym}:{int(cnt)}" for sym, cnt in symbol_counts_out.head(8).items()
        )
        tprint(
            "regime_search slice output: "
            f"rows={idx.size} symbols={symbol_counts_out.shape[0]} "
            f"days={pd.Series(data.iloc[idx]['timestamp']).dt.normalize().nunique()} top={top_symbols_out}"
        )
        data_out = data.iloc[idx].reset_index(drop=True)
        feat_out = {k: np.asarray(v)[idx] for k, v in feature_dict.items()}
        fwd_out = np.asarray(forward_returns)[idx]
        return data_out, feat_out, fwd_out
    except Exception as e:
        tprint(f"periods/symbols regime_search slice plan failed; using capped sample ({e})")
        LOGGER.warning(
            "regime_search slice-plan delegation failed; using raw sample: %s", e
        )
        return data, feature_dict, forward_returns


def _impute_and_scale_train_valid(
    X_train: np.ndarray, X_valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    X_tr = X_train.astype(np.float32, copy=True)
    X_va = X_valid.astype(np.float32, copy=True)

    X_tr[~np.isfinite(X_tr)] = np.nan
    X_va[~np.isfinite(X_va)] = np.nan

    n_features = X_tr.shape[1]
    med = np.zeros(n_features, dtype=np.float32)
    mean = np.zeros(n_features, dtype=np.float32)
    std = np.ones(n_features, dtype=np.float32)

    for j in range(n_features):
        col = X_tr[:, j]
        valid = ~np.isnan(col)
        if np.any(valid):
            m = np.median(col[valid]).astype(np.float32)
            med[j] = m
            X_tr[~valid, j] = m
            X_va[np.isnan(X_va[:, j]), j] = m
        else:
            med[j] = 0.0
            X_tr[:, j] = 0.0
            X_va[:, j] = 0.0

        mean[j] = np.mean(X_tr[:, j]).astype(np.float32)
        s = np.std(X_tr[:, j]).astype(np.float32)
        std[j] = s if s > 1e-6 else 1.0

    X_tr = ((X_tr - mean) / std).astype(np.float32)
    X_va = ((X_va - mean) / std).astype(np.float32)
    return X_tr, X_va


def _lgbm_fit_predict_cheap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
) -> np.ndarray:
    """Cheap yet better non-linear prediction."""
    from lightgbm import LGBMRegressor
    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.float32)
    X_va = np.asarray(X_valid, dtype=np.float32)
    
    if X_tr.shape[0] < 20: # Fallback to mean if too few samples
        return np.full(X_va.shape[0], np.mean(y_tr), dtype=np.float32)
        
    model = LGBMRegressor(
        n_estimators=30,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.1,
        min_child_samples=max(5, X_tr.shape[0] // 10),
        n_jobs=1,
        verbosity=-1,
        random_state=42
    )
    try:
        model.fit(X_tr, y_tr)
        return model.predict(X_va).astype(np.float32)
    except:
        return np.full(X_va.shape[0], np.mean(y_tr), dtype=np.float32)


def _classifier_oof_auc(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    symbols: Optional[np.ndarray] = None,
    n_splits: int = 2,
) -> float:
    if X.shape[0] < 20 or np.unique(y[np.isfinite(y)]).shape[0] < 2:
        return float("nan")
    folds = _build_temporal_folds(
        timestamps,
        X.shape[0],
        n_splits=n_splits,
        symbols=symbols,
    )
    if not folds:
        return float("nan")

    preds = np.full(X.shape[0], np.nan, dtype=np.float32)
    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue
        if np.unique(y[tr][np.isfinite(y[tr])]).shape[0] < 2:
            continue
        X_tr, X_va = _impute_and_scale_train_valid(X[tr], X[va])
        clf = LGBMClassifier(
            n_estimators=30,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.1,
            min_child_samples=max(5, X_tr.shape[0] // 10),
            n_jobs=1,
            verbosity=-1,
            random_state=42
        )
        try:
            clf.fit(X_tr, y[tr])
            preds[va] = clf.predict_proba(X_va)[:, 1].astype(np.float32)
        except Exception as e:
            _log_bounded_warning("classifier_fit", f"Classifier fold fit failed: {e}")

    valid_mask = np.isfinite(preds) & np.isfinite(y)
    if np.sum(valid_mask) == 0:
        return float("nan")
    if (
        np.unique(y[valid_mask]).shape[0] < 2
        or np.unique(preds[valid_mask]).shape[0] < 2
    ):
        return float("nan")
    try:
        return float(roc_auc_score(y[valid_mask], preds[valid_mask]))
    except Exception as e:
        _log_bounded_warning("roc_auc", f"AUC scoring failed: {e}")
        return float("nan")


def _classifier_oof_auc_from_folds(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    if X.shape[0] < 20 or np.unique(y[np.isfinite(y)]).shape[0] < 2:
        return float("nan")
    if not folds:
        return float("nan")

    preds = np.full(X.shape[0], np.nan, dtype=np.float32)
    collapsed_folds = 0
    total_requested_folds = len(folds)
    
    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            collapsed_folds += 1
            continue
        if np.unique(y[tr][np.isfinite(y[tr])]).shape[0] < 2:
            collapsed_folds += 1
            continue
        X_tr, X_va = _impute_and_scale_train_valid(X[tr], X[va])
        clf = LGBMClassifier(
            n_estimators=30,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.1,
            min_child_samples=max(5, X_tr.shape[0] // 10),
            n_jobs=1,
            verbosity=-1,
            random_state=42
        )
        try:
            clf.fit(X_tr, y[tr])
            preds[va] = clf.predict_proba(X_va)[:, 1].astype(np.float32)
        except Exception as e:
            collapsed_folds += 1
            _log_bounded_warning("classifier_fit", f"Classifier fold fit failed: {e}")

    valid_mask = np.isfinite(preds) & np.isfinite(y)
    fold_auc = float("nan")
    
    if np.sum(valid_mask) > 0:
        if (
            np.unique(y[valid_mask]).shape[0] >= 2
            and np.unique(preds[valid_mask]).shape[0] >= 2
        ):
            try:
                fold_auc = float(roc_auc_score(y[valid_mask], preds[valid_mask]))
            except:
                pass

    if np.isfinite(fold_auc):
        collapse_penalty = 0.05 * (collapsed_folds / total_requested_folds)
        return fold_auc - collapse_penalty

    return float("nan")


def _lgbm_subset_auc_and_lift(
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
) -> Tuple[float, float, str]:
    if tr_idx.shape[0] < 40 or va_idx.shape[0] < 40:
        return float("nan"), float("nan"), "fold_subset_too_small"

    y_tr = y[tr_idx]
    y_va = y[va_idx]
    if (
        np.unique(y_tr[np.isfinite(y_tr)]).shape[0] < 2
        or np.unique(y_va[np.isfinite(y_va)]).shape[0] < 2
    ):
        return float("nan"), float("nan"), "fold_single_class"

    min_child = int(max(10, min(50, tr_idx.shape[0] // 20)))
    clf = LGBMClassifier(
        n_estimators=64,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=min_child,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    try:
        clf.fit(X[tr_idx], y_tr)
        preds = clf.predict_proba(X[va_idx])[:, 1].astype(np.float32)
    except Exception as e:
        _log_bounded_warning("lgbm_fit", f"LGBM fold fit failed: {e}")
        return float("nan"), float("nan"), "fit_failure"

    valid = np.isfinite(preds) & np.isfinite(y_va)
    if np.sum(valid) < 40:
        return float("nan"), float("nan"), "too_few_finite_predictions"
    preds_val = preds[valid]
    y_val = y_va[valid]
    if np.unique(y_val).shape[0] < 2 or np.unique(preds_val).shape[0] < 2:
        if np.unique(y_val).shape[0] < 2:
            return float("nan"), float("nan"), "post_filter_single_class"
        return float("nan"), float("nan"), "constant_predictions"

    try:
        auc = float(roc_auc_score(y_val, preds_val))
    except Exception as e:
        _log_bounded_warning("lgbm_auc", f"LGBM AUC scoring failed: {e}")
        return float("nan"), float("nan"), "auc_failure"

    top_q = float(np.quantile(preds_val, 0.80))
    top_mask = preds_val >= top_q
    if np.any(top_mask):
        top_rate = float(np.mean(y_val[top_mask]))
        base_rate = float(np.mean(y_val))
        lift = float(top_rate - base_rate)
    else:
        lift = float("nan")
    return auc, lift, ""


def _lgbm_subset_cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    symbols: np.ndarray,
    idx_subset: np.ndarray,
    n_splits: int = 4,
    max_subset: int = 25000,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "auc_mean": float("nan"),
        "lift_mean": float("nan"),
        "auc_folds": np.asarray([], dtype=np.float32),
        "lift_folds": np.asarray([], dtype=np.float32),
        "invalid_reason": "",
        "subset_count": 0,
        "valid_label_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "requested_splits": int(n_splits),
        "effective_splits": 0,
        "built_folds": 0,
        "class_valid_folds": 0,
        "scored_auc_folds": 0,
        "scored_lift_folds": 0,
        "used_holdout_fallback": False,
        "fold_invalid_reason_counts": {},
    }
    idx = _cap_index_count(idx_subset, max_subset)
    out["subset_count"] = int(idx.shape[0])
    if idx.shape[0] < 40:
        out["invalid_reason"] = "subset_too_small"
        return out
    y_idx = y[idx]
    valid_y_idx = np.isfinite(y_idx)
    out["valid_label_count"] = int(np.sum(valid_y_idx))
    if np.sum(valid_y_idx) < 40:
        out["invalid_reason"] = "too_few_valid_labels"
        return out
    y_idx_valid = y_idx[valid_y_idx]
    if np.unique(y_idx_valid).shape[0] < 2:
        out["invalid_reason"] = "subset_single_class"
        return out

    positive_count = int(np.sum(y_idx_valid > 0.5))
    negative_count = int(np.sum(y_idx_valid <= 0.5))
    out["positive_count"] = positive_count
    out["negative_count"] = negative_count
    max_feasible_splits = max(0, min(positive_count, negative_count, int(n_splits)))
    n_splits_local = max(2, max_feasible_splits) if max_feasible_splits >= 2 else 2
    out["effective_splits"] = int(n_splits_local)

    folds_local = _build_temporal_folds(
        timestamps[idx],
        idx.shape[0],
        n_splits=n_splits_local,
        symbols=symbols[idx],
    )
    if not folds_local:
        out["invalid_reason"] = "no_temporal_folds"
        return out
    out["built_folds"] = int(len(folds_local))

    auc_folds: List[float] = []
    lift_folds: List[float] = []
    fold_invalid_reason_counts: Dict[str, int] = {}
    for tr_local, va_local in folds_local:
        tr_idx = idx[tr_local]
        va_idx = idx[va_local]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        if (
            np.unique(y_tr[np.isfinite(y_tr)]).shape[0] < 2
            or np.unique(y_va[np.isfinite(y_va)]).shape[0] < 2
        ):
            continue
        out["class_valid_folds"] = int(out["class_valid_folds"]) + 1
        auc, lift, fold_reason = _lgbm_subset_auc_and_lift(X, y, tr_idx, va_idx)
        if fold_reason:
            fold_invalid_reason_counts[fold_reason] = (
                int(fold_invalid_reason_counts.get(fold_reason, 0)) + 1
            )
        if np.isfinite(auc):
            auc_folds.append(float(auc))
        if np.isfinite(lift):
            lift_folds.append(float(lift))

    if not auc_folds:
        idx_ts = np.asarray(timestamps[idx])
        uniq_ts = np.unique(idx_ts)
        if uniq_ts.shape[0] >= 4:
            out["used_holdout_fallback"] = True
            for frac in (0.60, 0.67, 0.75):
                split_pos = int(np.floor((uniq_ts.shape[0] - 1) * frac))
                split_pos = min(max(split_pos, 1), uniq_ts.shape[0] - 2)
                cutoff = uniq_ts[split_pos]
                tr_local = np.flatnonzero(idx_ts <= cutoff).astype(np.int32)
                va_local = np.flatnonzero(idx_ts > cutoff).astype(np.int32)
                if tr_local.size < 40 or va_local.size < 40:
                    continue
                tr_idx = idx[tr_local]
                va_idx = idx[va_local]
                y_tr = y[tr_idx]
                y_va = y[va_idx]
                if (
                    np.unique(y_tr[np.isfinite(y_tr)]).shape[0] < 2
                    or np.unique(y_va[np.isfinite(y_va)]).shape[0] < 2
                ):
                    continue
                auc, lift, fold_reason = _lgbm_subset_auc_and_lift(X, y, tr_idx, va_idx)
                if fold_reason:
                    fold_invalid_reason_counts[fold_reason] = (
                        int(fold_invalid_reason_counts.get(fold_reason, 0)) + 1
                    )
                if np.isfinite(auc):
                    auc_folds.append(float(auc))
                if np.isfinite(lift):
                    lift_folds.append(float(lift))
                if auc_folds:
                    break

    if auc_folds:
        out["auc_mean"] = float(np.mean(np.asarray(auc_folds, dtype=np.float32)))
        out["auc_folds"] = np.asarray(auc_folds, dtype=np.float32)
        out["scored_auc_folds"] = int(len(auc_folds))
    if lift_folds:
        out["lift_mean"] = float(np.mean(np.asarray(lift_folds, dtype=np.float32)))
        out["lift_folds"] = np.asarray(lift_folds, dtype=np.float32)
        out["scored_lift_folds"] = int(len(lift_folds))
    if not auc_folds:
        if int(out["class_valid_folds"]) <= 0:
            out["invalid_reason"] = "folds_lacked_class_diversity"
        else:
            out["invalid_reason"] = "no_valid_auc_after_fit"
    out["fold_invalid_reason_counts"] = dict(fold_invalid_reason_counts)
    return out


def _incremental_information_metrics(
    learn_X: np.ndarray,
    side_mask: np.ndarray,
    y_primary: np.ndarray,
    timestamps: np.ndarray,
    symbols: np.ndarray,
    idx_e: np.ndarray,
    idx_ne: np.ndarray,
    n_splits: int = 3,
) -> Dict[str, float]:
    metrics = {
        "incremental_information_delta_auc": float("nan"),
        "incremental_information_delta_auc_fold_mean": float("nan"),
        "incremental_information_delta_auc_fold_std": float("nan"),
        "incremental_information_positive_fold_fraction": float("nan"),
        "incremental_information_positive_fold_count": float("nan"),
        "incremental_information_fold_count": float("nan"),
    }

    idx_all = np.sort(np.concatenate([idx_e, idx_ne]).astype(np.int32))
    if idx_all.shape[0] < 100:
        return metrics

    y_all = y_primary[idx_all]
    ts_all = timestamps[idx_all]
    sym_all = symbols[idx_all]
    event_feature = side_mask[idx_all].astype(np.float32).reshape(-1, 1)
    X_base = learn_X[idx_all]
    X_aug = np.concatenate([X_base, event_feature], axis=1).astype(
        np.float32, copy=False
    )
    folds = _build_temporal_folds(
        ts_all,
        idx_all.shape[0],
        n_splits=n_splits,
        symbols=sym_all,
    )
    if not folds:
        return metrics

    auc_base = _classifier_oof_auc_from_folds(X_base, y_all, folds)
    auc_aug = _classifier_oof_auc_from_folds(X_aug, y_all, folds)
    metrics["incremental_information_delta_auc"] = float(auc_aug - auc_base)

    positive_fold_count = 0
    evaluated_fold_count = 0
    fold_deltas: List[float] = []
    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue
        y_tr = y_all[tr]
        y_va = y_all[va]
        if (
            np.unique(y_tr[np.isfinite(y_tr)]).shape[0] < 2
            or np.unique(y_va[np.isfinite(y_va)]).shape[0] < 2
        ):
            continue
        auc_base_fold = _classifier_oof_auc_from_folds(X_base, y_all, [(tr, va)])
        auc_aug_fold = _classifier_oof_auc_from_folds(X_aug, y_all, [(tr, va)])
        delta_fold = float(auc_aug_fold - auc_base_fold)
        fold_deltas.append(delta_fold)
        evaluated_fold_count += 1
        if delta_fold > 0:
            positive_fold_count += 1

    if evaluated_fold_count > 0:
        metrics["incremental_information_delta_auc_fold_mean"] = float(
            np.mean(np.asarray(fold_deltas, dtype=np.float32))
        )
        metrics["incremental_information_delta_auc_fold_std"] = float(
            np.std(np.asarray(fold_deltas, dtype=np.float32))
        )
        metrics["incremental_information_positive_fold_fraction"] = float(
            positive_fold_count / float(evaluated_fold_count)
        )
        metrics["incremental_information_positive_fold_count"] = float(
            positive_fold_count
        )
        metrics["incremental_information_fold_count"] = float(evaluated_fold_count)

    return metrics


def _primary_gain_fold_deltas(
    learn_X: np.ndarray,
    y_primary: np.ndarray,
    timestamps: np.ndarray,
    symbols: np.ndarray,
    idx_e: np.ndarray,
    idx_ne: np.ndarray,
    n_splits: int = 3,
) -> np.ndarray:
    if idx_e.shape[0] < 50 or idx_ne.shape[0] < 50:
        return np.asarray([], dtype=np.float32)

    folds_e = _build_temporal_folds(
        timestamps[idx_e],
        idx_e.shape[0],
        n_splits=n_splits,
        symbols=symbols[idx_e],
    )
    folds_ne = _build_temporal_folds(
        timestamps[idx_ne],
        idx_ne.shape[0],
        n_splits=n_splits,
        symbols=symbols[idx_ne],
    )
    if not folds_e or not folds_ne:
        return np.asarray([], dtype=np.float32)

    out: List[float] = []
    for (tr_e, va_e), (tr_ne, va_ne) in zip(folds_e, folds_ne):
        auc_e = _classifier_oof_auc_from_folds(
            learn_X[idx_e], y_primary[idx_e], [(tr_e, va_e)]
        )
        auc_ne = _classifier_oof_auc_from_folds(
            learn_X[idx_ne], y_primary[idx_ne], [(tr_ne, va_ne)]
        )
        out.append(float(auc_e - auc_ne))
    return np.asarray(out, dtype=np.float32)


def _stability_from_fold_deltas(delta_folds: np.ndarray) -> Dict[str, float]:
    delta_folds = np.asarray(delta_folds, dtype=np.float32)
    delta_folds = delta_folds[np.isfinite(delta_folds)]
    if delta_folds.size < 2: # Need at least 2 folds to compute std
        return {
            "delta_fold_mean": float("nan"),
            "delta_fold_std": float("nan"),
            "positive_fold_fraction": float("nan"),
            "stability_score": float("nan"),
            "fold_count": 0.0,
        }

    mean_delta = float(np.mean(delta_folds))
    std_delta = float(np.std(delta_folds))
    positive_fold_fraction = float(np.mean(delta_folds > 0))
    stability_score = (
        0.5 * max(0.0, 1.0 - std_delta / (abs(mean_delta) + 1e-9))
        + 0.5 * positive_fold_fraction
    )
    return {
        "delta_fold_mean": mean_delta,
        "delta_fold_std": std_delta,
        "positive_fold_fraction": positive_fold_fraction,
        "stability_score": float(stability_score),
        "fold_count": float(delta_folds.size),
    }


def _build_regime_rationale(row: pd.Series) -> str:
    reasons: List[str] = []
    delta_r = _metric_or_nan(row.get("delta_r"))
    delta_r_shrunk = _metric_or_nan(row.get("delta_r_shrunk"))
    s_r = _metric_or_nan(row.get("tbm_lgbm_stability", row.get("S_r")))
    positive_fraction = _metric_or_nan(
        row.get("tbm_lgbm_positive_fold_fraction", row.get("positive_fold_fraction_r"))
    )
    d_r = _metric_or_nan(row.get("D_r"))
    metric_name = str(row.get("selected_delta_metric", ""))
    incr_delta = _metric_or_nan(row.get("incremental_information_delta_auc"))
    incr_positive = _metric_or_nan(
        row.get("incremental_information_positive_fold_fraction")
    )
    disp_edge = _metric_or_nan(row.get("dispersion_to_edge_ratio"))
    primary_nan = _metric_or_nan(row.get("primary_predictability_gain_is_nan"))

    if np.isfinite(delta_r) and delta_r > 0:
        reasons.append(f"positive bucket OOS delta_r={delta_r:.4f}")
    if np.isfinite(delta_r_shrunk):
        reasons.append(f"shrunk delta={delta_r_shrunk:.4f}")
    if np.isfinite(s_r):
        reasons.append(f"stability={s_r:.3f}")
    if np.isfinite(positive_fraction):
        reasons.append(f"positive-fold fraction={positive_fraction:.3f}")
    if np.isfinite(incr_delta):
        reasons.append(f"delta-auc={incr_delta:.4f}")
    if np.isfinite(incr_positive):
        reasons.append(f"delta-auc positive folds={incr_positive:.3f}")
    if np.isfinite(disp_edge):
        reasons.append(f"dispersion/edge={disp_edge:.3f}")
    if np.isfinite(primary_nan) and primary_nan > 0.5:
        reasons.append("primary directional gain unavailable")
    if np.isfinite(d_r):
        reasons.append(f"dispersion={d_r:.3f}")
    return "; ".join(reasons)



def dispersion_to_edge(returns: np.ndarray) -> float:
    """
    DER = sigma / |mu|
    """
    mu = np.mean(returns)
    sigma = np.std(returns)

    if abs(mu) < 1e-12:
        return np.inf

    return float(sigma / abs(mu))


def fold_stability(delta_r_folds: np.ndarray) -> float:
    """
    S_r = |mean(delta_r)| / std(delta_r)
    """
    mean = np.mean(delta_r_folds)
    std = np.std(delta_r_folds)

    if std < 1e-12:
        return 0.0

    return float(abs(mean) / std)


def label_entropy(labels: np.ndarray) -> float:
    """
    Shannon entropy of discrete labels.
    """
    if len(labels) == 0:
        return 0.0
    values, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()

    return float(-(p * np.log(p + 1e-12)).sum())


def compute_net_regime_value(
    returns_E: np.ndarray,
    returns_ER: np.ndarray,
    delta_r_folds_E: np.ndarray,
    delta_r_folds_ER: np.ndarray,
    labels_E: np.ndarray,
    labels_ER: np.ndarray,
    auc_E: float,
    auc_ER: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the NetRegimeValue score.
    """
    # coverage
    coverage_ratio = len(returns_ER) / max(len(returns_E), 1)
    coverage_term = np.sqrt(coverage_ratio)

    # dispersion-to-edge
    der_E = dispersion_to_edge(returns_E)
    der_ER = dispersion_to_edge(returns_ER)
    der_ratio = der_E / der_ER if der_ER > 0 else 0
    der_ratio = float(np.clip(der_ratio, 0.5, 3.0))

    # fold stability
    sr_E = fold_stability(delta_r_folds_E)
    sr_ER = fold_stability(delta_r_folds_ER)
    sr_ratio = sr_ER / sr_E if sr_E > 0 else 1.0
    sr_ratio = float(np.clip(sr_ratio, 0.5, 3.0))

    # entropy reduction
    H_E = label_entropy(labels_E)
    H_ER = label_entropy(labels_ER)
    entropy_term = float(np.exp(np.clip(H_E - H_ER, -0.5, 0.5)))

    # AUC improvement
    auc_gain = float(np.clip(max(0.0, auc_ER - auc_E), 0.0, 0.1))
    auc_term = 1.0 + auc_gain

    score = float(coverage_term * der_ratio * sr_ratio * entropy_term * auc_term)

    diagnostics = {
        "coverage_ratio": coverage_ratio,
        "DER_E": der_E,
        "DER_ER": der_ER,
        "DER_ratio": der_ratio,
        "S_r_E": sr_E,
        "S_r_ER": sr_ER,
        "S_r_ratio": sr_ratio,
        "entropy_E": H_E,
        "entropy_ER": H_ER,
        "auc_E": auc_E,
        "auc_ER": auc_ER,
        "net_regime_value": score,
    }

    return score, diagnostics


def quick_tree_auc(
    features_df: Any,
    labels: np.ndarray,
    event_mask: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]]
) -> float:
    """
    Computes a quick out-of-sample AUC using Cheap Tree logic.
    """
    if isinstance(features_df, pd.DataFrame):
        features_arr = (
            features_df.select_dtypes(include=[np.number])
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32, copy=True)
        )
    else:
        features_arr = np.asarray(features_df, dtype=np.float32)
        if features_arr.ndim != 2:
            return float("nan")
        features_arr = features_arr.copy()
        features_arr[np.isinf(features_arr)] = np.nan
    if features_arr.size == 0:
        return float("nan")

    if np.sum(event_mask) < 20 or labels.size == 0 or labels.shape[0] != event_mask.shape[0]:
        return float("nan")

    y_event = labels[event_mask]
    if len(np.unique(y_event)) < 2:
        return float("nan")

    X_event_raw = features_arr[event_mask]
    # Mapping global indices to event indices
    global_to_local = np.full(len(event_mask), -1, dtype=np.int32)
    global_to_local[event_mask] = np.arange(np.sum(event_mask))

    oof_preds = np.full(len(y_event), np.nan, dtype=np.float32)

    for tr, va in folds:
        # Get event-only indices
        tr_local = global_to_local[tr]
        tr_local = tr_local[tr_local >= 0]

        va_local = global_to_local[va]
        va_local = va_local[va_local >= 0]

        if len(tr_local) < 10 or len(va_local) < 2:
            continue

        X_tr, X_va = _impute_and_scale_train_valid(X_event_raw[tr_local], X_event_raw[va_local])
        y_tr = y_event[tr_local]

        if len(np.unique(y_tr)) < 2:
            oof_preds[va_local] = np.mean(y_tr)
            continue

        oof_preds[va_local] = _lgbm_fit_predict_cheap(X_tr, y_tr.astype(np.float32, copy=False), X_va)

    valid_oof = np.isfinite(oof_preds)
    if np.sum(valid_oof) < 10 or len(np.unique(y_event[valid_oof])) < 2:
        return float("nan")

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_event[valid_oof], oof_preds[valid_oof]))
    except Exception:
        return float("nan")


def _predictability_gain_from_metrics(metrics: Dict[str, Any]) -> float:
    vals = [
        _metric_or_nan(metrics.get("continuation_predictability_gain")),
        _metric_or_nan(metrics.get("reversal_predictability_gain")),
        _metric_or_nan(metrics.get("MAE_predictability_gain")),
        _metric_or_nan(metrics.get("MFE_predictability_gain")),
    ]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(max(vals))


def _mode_primary_predictability_col(mode: str) -> str:
    return (
        "continuation_predictability_gain"
        if _mode_is_tf(mode)
        else "reversal_predictability_gain"
    )


def _mode_predictability_gain_from_metrics(mode: str, metrics: Dict[str, Any]) -> float:
    vals = [
        _metric_or_nan(metrics.get(_mode_primary_predictability_col(mode))),
        _metric_or_nan(metrics.get("MAE_predictability_gain")),
        _metric_or_nan(metrics.get("MFE_predictability_gain")),
    ]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(max(vals))


def _compute_legacy_conditional_learnability(
    mode: str,
    side_mask: np.ndarray,
    shared: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    ret_threshold = float(cfg.get("phase1_ret_threshold", 0.0))
    learn_X = shared["learn_X"]
    forward_returns = shared["forward_returns"]
    timestamps = shared["timestamps"]
    symbol_codes = np.asarray(shared["symbol_codes"], dtype=np.int32)
    symbol_codes = np.asarray(shared["symbol_codes"], dtype=np.int32)
    valid = np.isfinite(forward_returns)

    idx_g = np.where(valid)[0].astype(np.int32)
    idx_e = np.where(valid & side_mask)[0].astype(np.int32)

    out = {
        "continuation_predictability_gain": float("nan"),
        "reversal_predictability_gain": float("nan"),
        "MAE_predictability_gain": float("nan"),
        "MFE_predictability_gain": float("nan"),
        "predictability_gain": float("nan"),
    }
    if idx_e.shape[0] < 50 or idx_g.shape[0] < 100:
        return out

    max_global = int(cfg.get("phase2_metric_max_samples_per_class", 25_000))
    if max_global > 0 and idx_g.shape[0] > max_global:
        rng = np.random.RandomState(123)
        idx_g = np.sort(rng.choice(idx_g, max_global, replace=False).astype(np.int32))
    max_event = int(
        cfg.get(
            "legacy_stage2_event_max_samples",
            cfg.get("phase2_metric_max_samples_per_class", 25_000),
        )
    )
    if max_event > 0 and idx_e.shape[0] > max_event:
        rng = np.random.RandomState(456)
        idx_e = np.sort(rng.choice(idx_e, max_event, replace=False).astype(np.int32))

    n_splits = int(cfg.get("phase2_classifier_n_splits", 3))
    y_cont = _mode_primary_target(mode, forward_returns, ret_threshold)
    y_rev = np.full(y_cont.shape[0], np.nan, dtype=np.float32)
    valid_y = np.isfinite(y_cont)
    y_rev[valid_y] = 1.0 - y_cont[valid_y]

    auc_cont_g = _classifier_oof_auc(
        learn_X[idx_g],
        y_cont[idx_g],
        timestamps[idx_g],
        symbols=symbol_codes[idx_g],
        n_splits=n_splits,
    )
    auc_cont_e = _classifier_oof_auc(
        learn_X[idx_e],
        y_cont[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        n_splits=n_splits,
    )
    out["continuation_predictability_gain"] = float(auc_cont_e - auc_cont_g)

    auc_rev_g = _classifier_oof_auc(
        learn_X[idx_g],
        y_rev[idx_g],
        timestamps[idx_g],
        symbols=symbol_codes[idx_g],
        n_splits=n_splits,
    )
    auc_rev_e = _classifier_oof_auc(
        learn_X[idx_e],
        y_rev[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        n_splits=n_splits,
    )
    out["reversal_predictability_gain"] = float(auc_rev_e - auc_rev_g)

    if mode == MODE_LONG:
        mae_arr = shared["mae_high"]
        mfe_arr = shared["mfe_high"]
    else:
        mae_arr = shared["mae_low"]
        mfe_arr = shared["mfe_low"]

    r2_mae_g = _ridge_regression_oof_r2(
        learn_X[idx_g],
        mae_arr[idx_g],
        timestamps[idx_g],
        symbols=symbol_codes[idx_g],
        clip_q=0.98,
        n_splits=n_splits,
    )
    r2_mae_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        mae_arr[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        clip_q=0.98,
        n_splits=n_splits,
    )
    out["MAE_predictability_gain"] = float(r2_mae_e - r2_mae_g)

    r2_mfe_g = _ridge_regression_oof_r2(
        learn_X[idx_g],
        mfe_arr[idx_g],
        timestamps[idx_g],
        symbols=symbol_codes[idx_g],
        clip_q=0.98,
        n_splits=n_splits,
    )
    r2_mfe_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        mfe_arr[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        clip_q=0.98,
        n_splits=n_splits,
    )
    out["MFE_predictability_gain"] = float(r2_mfe_e - r2_mfe_g)

    out["predictability_gain"] = _mode_predictability_gain_from_metrics(mode, out)
    return out


def _apply_secondary_conditioner(
    mask_h: np.ndarray,
    mask_l: np.ndarray,
    conditioner: str,
    mono_up: np.ndarray,
    mono_dn: np.ndarray,
    vol_exp: np.ndarray,
    alternation_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    new_h = mask_h.copy()
    new_l = mask_l.copy()

    if conditioner == "none":
        return new_h, new_l
    if conditioner == "monotonicity_adjust":
        return new_h & (mono_up > 0.25), new_l & (mono_dn > 0.25)
    if conditioner == "volatility_adjust":
        return new_h & (vol_exp < 5.0), new_l & (vol_exp < 5.0)
    if conditioner == "alternation_adjust":
        return new_h & (alternation_array < 0.70), new_l & (alternation_array < 0.70)
    return new_h, new_l


def _ridge_regression_oof_r2(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    symbols: Optional[np.ndarray] = None,
    clip_q: float = 0.98,
    n_splits: int = 2,
) -> float:
    valid = np.isfinite(y)
    if np.sum(valid) < 20:
        return 0.0

    y = y.astype(np.float32, copy=True)
    folds = _build_temporal_folds(
        timestamps,
        X.shape[0],
        n_splits=n_splits,
        symbols=symbols,
    )
    if not folds:
        return 0.0

    preds = np.full(X.shape[0], np.nan, dtype=np.float32)

    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue

        tr_valid = tr[np.isfinite(y[tr])]
        if tr_valid.shape[0] < 10:
            continue

        y_tr = y[tr_valid]
        hi = np.quantile(y_tr, clip_q).astype(np.float32)
        lo = (
            np.quantile(y_tr, 1.0 - clip_q).astype(np.float32)
            if np.any(y_tr < 0)
            else 0.0
        )
        y_tr_clip = np.clip(y_tr, lo, hi).astype(np.float32)

        X_tr, X_va = _impute_and_scale_train_valid(X[tr_valid], X[va])
        try:
            preds_fold = _lgbm_fit_predict_cheap(X_tr, y_tr_clip, X_va)
            preds[va] = preds_fold.astype(np.float32, copy=False)
        except Exception as e:
            _log_bounded_warning("ridge_fit", f"Ridge fold fit failed: {e}")

    valid2 = np.isfinite(preds) & np.isfinite(y)
    if np.sum(valid2) < 10:
        return 0.0
    ssr = float(np.sum((y[valid2] - preds[valid2]) ** 2))
    sst = float(np.sum((y[valid2] - np.mean(y[valid2])) ** 2))
    if sst < 1e-9:
        return 0.0
    return float(1.0 - ssr / sst)


def _ridge_regression_fold_r2s(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    symbols: Optional[np.ndarray] = None,
    clip_q: float = 0.98,
    n_splits: int = 3,
) -> np.ndarray:
    valid = np.isfinite(y)
    if np.sum(valid) < 20:
        return np.asarray([], dtype=np.float32)

    y = y.astype(np.float32, copy=True)
    folds = _build_temporal_folds(
        timestamps,
        X.shape[0],
        n_splits=n_splits,
        symbols=symbols,
    )
    if not folds:
        return np.asarray([], dtype=np.float32)

    scores: List[float] = []
    for tr, va in folds:
        if tr.shape[0] == 0 or va.shape[0] == 0:
            continue

        tr_valid = tr[np.isfinite(y[tr])]
        va_valid = va[np.isfinite(y[va])]
        if tr_valid.shape[0] < 10 or va_valid.shape[0] < 10:
            continue

        y_tr = y[tr_valid]
        hi = np.quantile(y_tr, clip_q).astype(np.float32)
        lo = (
            np.quantile(y_tr, 1.0 - clip_q).astype(np.float32)
            if np.any(y_tr < 0)
            else 0.0
        )
        y_tr_clip = np.clip(y_tr, lo, hi).astype(np.float32)

        X_tr, X_va = _impute_and_scale_train_valid(X[tr_valid], X[va_valid])
        try:
            preds = _lgbm_fit_predict_cheap(X_tr, y_tr_clip, X_va)
        except Exception as e:
            _log_bounded_warning("ridge_fit", f"Ridge fold fit failed: {e}")
            continue

        y_va = y[va_valid]
        sst = float(np.sum((y_va - np.mean(y_va)) ** 2))
        if sst < 1e-9:
            continue
        ssr = float(np.sum((y_va - preds) ** 2))
        scores.append(float(1.0 - ssr / sst))

    return np.asarray(scores, dtype=np.float32)


def _single_feature_fold_r2(
    x: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    clip_q: float = 0.98,
) -> float:
    if tr_idx.shape[0] < 10 or va_idx.shape[0] < 10:
        return float("nan")

    y_tr = y[tr_idx].astype(np.float32, copy=True)
    y_va = y[va_idx].astype(np.float32, copy=False)
    hi = np.quantile(y_tr, clip_q).astype(np.float32)
    lo = (
        np.quantile(y_tr, 1.0 - clip_q).astype(np.float32)
        if np.any(y_tr < 0)
        else 0.0
    )
    y_tr_clip = np.clip(y_tr, lo, hi).astype(np.float32)

    X_tr, X_va = _impute_and_scale_train_valid(
        x[tr_idx].reshape(-1, 1), x[va_idx].reshape(-1, 1)
    )
    try:
        preds = _lgbm_fit_predict_cheap(X_tr, y_tr_clip, X_va)
    except Exception as e:
        _log_bounded_warning(
            "single_feature_ridge_fit", f"Single-feature ridge fit failed: {e}"
        )
        return float("nan")

    valid = np.isfinite(preds) & np.isfinite(y_va)
    if np.sum(valid) < 10:
        return float("nan")
    ssr = float(np.sum((y_va[valid] - preds[valid]) ** 2))
    sst = float(np.sum((y_va[valid] - np.mean(y_va[valid])) ** 2))
    if sst < 1e-9:
        return float("nan")
    return float(1.0 - ssr / sst)


def _cap_index_count(idx: np.ndarray, max_count: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int32)
    if idx.shape[0] <= max_count:
        return idx
    pos = np.linspace(0, idx.shape[0] - 1, num=max_count, dtype=np.int32)
    return idx[pos]


def _ridge_subset_fold_metrics(
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    clip_q: float = 0.98,
) -> Tuple[float, float]:
    if tr_idx.shape[0] < 20 or va_idx.shape[0] < 20:
        return float("nan"), float("nan")

    y_tr = y[tr_idx].astype(np.float32, copy=True)
    y_va = y[va_idx].astype(np.float32, copy=False)
    hi = np.quantile(y_tr, clip_q).astype(np.float32)
    lo = (
        np.quantile(y_tr, 1.0 - clip_q).astype(np.float32)
        if np.any(y_tr < 0)
        else 0.0
    )
    y_tr_clip = np.clip(y_tr, lo, hi).astype(np.float32)

    X_tr, X_va = _impute_and_scale_train_valid(X[tr_idx], X[va_idx])
    try:
        preds = _lgbm_fit_predict_cheap(X_tr, y_tr_clip, X_va)
    except Exception as e:
        _log_bounded_warning("subset_ridge_fit", f"Subset ridge fit failed: {e}")
        return float("nan"), float("nan")

    valid = np.isfinite(preds) & np.isfinite(y_va)
    if np.sum(valid) < 20:
        return float("nan"), float("nan")
    y_val = y_va[valid]
    preds_val = preds[valid]
    sst = float(np.sum((y_val - np.mean(y_val)) ** 2))
    if sst < 1e-9:
        r2 = float("nan")
    else:
        ssr = float(np.sum((y_val - preds_val) ** 2))
        r2 = float(1.0 - ssr / sst)

    top_q = float(np.quantile(preds_val, 0.80))
    bot_q = float(np.quantile(preds_val, 0.20))
    top_mask = preds_val >= top_q
    bot_mask = preds_val <= bot_q
    if np.any(top_mask) and np.any(bot_mask):
        spread = float(np.mean(y_val[top_mask]) - np.mean(y_val[bot_mask]))
    else:
        spread = float("nan")
    return r2, spread


def _extract_learnability_features(
    feature_dict: Dict[str, np.ndarray], n_samples: int
) -> np.ndarray:
    keys = list(_required_feature_keys())
    # Broaden to include all ridge feature cols
    learnability_keys = list(set(keys) | set(RIDGE_FEATURE_COLS))
    learnability_keys = [k for k in learnability_keys if k not in {"atr", "vol_regime_z"}]
    X = np.full((n_samples, len(learnability_keys)), np.nan, dtype=np.float32)
    for i, k in enumerate(learnability_keys):
        if k not in feature_dict:
            X[:, i] = 0.0
            continue
        arr = np.asarray(feature_dict[k], dtype=np.float32)
        arr = arr.copy()
        arr[np.isinf(arr)] = np.nan
        X[:, i] = arr
    return X


def _required_feature_keys() -> Tuple[str, ...]:
    return (
        "atr",
        "vol_regime_z",
        "range_1_atr",
        "close_location_in_bar",
        "rv_ratio_6_24",
        "impulse_vol_ratio",
        "vol_compression_ratio",
        "range_decay",
        "momentum_last_3bars_impulse_return",
        "reversal_bar_strength",
        "climax_volume_ratio",
        "rejection_volume_ratio",
        "vol_regime_shift",
        "bar_direction_entropy",
    )


def _flatten_wide_frame(
    df: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> np.ndarray:
    return (
        df.reindex(index=index, columns=columns)
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(-1)
    )


def _build_day_ids(timestamps: np.ndarray) -> Tuple[np.ndarray, int]:
    days = timestamps.astype("datetime64[D]")
    uniq, inv = np.unique(days, return_inverse=True)
    return inv.astype(np.int32), int(uniq.shape[0])


def _build_timestamp_ids(timestamps: np.ndarray) -> Tuple[np.ndarray, int]:
    uniq, inv = np.unique(timestamps, return_inverse=True)
    return inv.astype(np.int32), int(uniq.shape[0])


def _build_vol_regime_ids(vol_feature: np.ndarray) -> np.ndarray:
    x = np.asarray(vol_feature, dtype=np.float32)
    valid = np.isfinite(x)
    out = np.ones(x.shape[0], dtype=np.int8)
    if np.sum(valid) < 10:
        return out
    q1 = np.quantile(x[valid], 1 / 3).astype(np.float32)
    q2 = np.quantile(x[valid], 2 / 3).astype(np.float32)
    out[x <= q1] = 0
    out[(x > q1) & (x <= q2)] = 1
    out[x > q2] = 2
    return out


def _sample_half_history_mask(day_ids: np.ndarray, seed: int = 42) -> np.ndarray:
    uniq_days = np.unique(day_ids)
    selected_days = list(set(_rng_sample_half(uniq_days.tolist(), seed=seed)))
    return np.isin(day_ids, selected_days)


def _sample_history_fraction_mask(
    day_ids: np.ndarray, frac: float, seed: int = 42
) -> np.ndarray:
    uniq_days = np.unique(day_ids)
    selected_days = list(
        set(_rng_sample_fraction(uniq_days.tolist(), frac=frac, seed=seed))
    )
    return np.isin(day_ids, selected_days)


def _validate_long_panel_shape(
    timestamps: np.ndarray,
    symbols: np.ndarray,
    require_rectangular: bool = False,
) -> None:
    if timestamps.shape[0] == 0:
        return
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("Long panel must be sorted by timestamp ascending.")

    uniq_ts, first_idx, counts = np.unique(
        timestamps, return_index=True, return_counts=True
    )
    if require_rectangular and np.unique(counts).shape[0] > 1:
        raise ValueError(
            "Rectangular panel required, but per-timestamp row counts differ."
        )

    ref_order = None
    for pos, st in enumerate(first_idx):
        c = counts[pos]
        curr = symbols[st : st + c]
        if ref_order is None:
            ref_order = curr
        elif require_rectangular and (
            c != ref_order.shape[0] or np.any(curr != ref_order)
        ):
            raise ValueError(
                "Rectangular panel required, but symbol ordering differs by timestamp."
            )


def _safe_param_to_string(param: Any) -> str:
    if isinstance(param, tuple):
        return str(tuple(float(x) for x in param))
    return (
        str(float(param))
        if isinstance(param, (int, float, np.integer, np.floating))
        else str(param)
    )


def _build_candidate_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    del cfg
    family_buckets = {
        "moving_average_location": [
            "LOC_01_AboveEMA",
            "LOC_02_BelowEMA",
            "LOC_03_BetweenFastMidEMA",
            "LOC_04_BetweenMidSlowEMA",
            "LOC_05_StackedAboveAllEMAs",
            "LOC_06_StackedBelowAllEMAs",
            "LOC_07_TouchFastEMA_Long",
            "LOC_08_TouchFastEMA_Short",
            "LOC_09_TouchMidEMA_Long",
            "LOC_10_TouchMidEMA_Short",
            "LOC_11_DeepPullbackToSlowEMA_Long",
            "LOC_12_DeepPullbackToSlowEMA_Short",
            "LOC_13_EMAValueZone_Long",
            "LOC_14_EMAValueZone_Short",
        ],
        "vwap_location": [
            "LOC_20_AboveVWAP",
            "LOC_21_BelowVWAP",
            "LOC_22_AtVWAP_Long",
            "LOC_23_AtVWAP_Short",
            "LOC_24_VWAPPlus1Dev",
            "LOC_25_VWAPMinus1Dev",
            "LOC_26_VWAPPlus2Dev",
            "LOC_27_VWAPMinus2Dev",
            "LOC_28_BetweenVWAPAndPlus1Dev",
            "LOC_29_BetweenVWAPAndMinus1Dev",
            "LOC_30_ReclaimVWAPZone_Long",
            "LOC_31_LoseVWAPZone_Short",
        ],
        "range_location": [
            "LOC_40_UpperQuartileOfRange",
            "LOC_41_LowerQuartileOfRange",
            "LOC_42_MidRange",
            "LOC_43_NearRangeHigh",
            "LOC_44_NearRangeLow",
            "LOC_45_AtRangeBreakoutZone_Long",
            "LOC_46_AtRangeBreakdownZone_Short",
        ],
        "prior_bar_location": [
            "LOC_50_AbovePriorHigh",
            "LOC_51_BelowPriorLow",
            "LOC_52_InsidePriorRange",
            "LOC_53_NearPriorHigh",
            "LOC_54_NearPriorLow",
            "LOC_55_AboveLastSwingHigh",
            "LOC_56_BelowLastSwingLow",
            "LOC_57_NearLastSwingHigh",
            "LOC_58_NearLastSwingLow",
            "LOC_59_BetweenLastSwingLowHigh",
        ],
        "session_location": [
            "LOC_70_AboveSessionOpen",
            "LOC_71_BelowSessionOpen",
            "LOC_72_AtSessionOpen_Long",
            "LOC_73_AtSessionOpen_Short",
            "LOC_74_AboveInitialBalanceMid",
            "LOC_75_BelowInitialBalanceMid",
            "LOC_76_NearInitialBalanceHigh",
            "LOC_77_NearInitialBalanceLow",
            "LOC_78_AtSessionHighZone",
            "LOC_79_AtSessionLowZone",
            "LOC_80_UpperHalfOfSessionRange",
            "LOC_81_LowerHalfOfSessionRange",
        ],
        "higher_tf_location": [
            "LOC_90_AbovePrevDayHigh",
            "LOC_91_BelowPrevDayLow",
            "LOC_92_InsidePrevDayRange",
            "LOC_93_NearPrevDayHigh",
            "LOC_94_NearPrevDayLow",
            "LOC_95_AbovePrevDayMid",
            "LOC_96_BelowPrevDayMid",
            "LOC_97_NearPrevWeekHigh",
            "LOC_98_NearPrevWeekLow",
            "LOC_99_InsidePrevWeekRange",
        ],
        "band_location": [
            "LOC_110_AboveBBMid",
            "LOC_111_BelowBBMid",
            "LOC_112_AtBBUpper",
            "LOC_113_AtBBLower",
            "LOC_114_OutsideBBUpper",
            "LOC_115_OutsideBBLower",
            "LOC_116_AtKCUpper",
            "LOC_117_AtKCLower",
            "LOC_118_BetweenBBMidAndUpper",
            "LOC_119_BetweenBBMidAndLower",
        ],
        "pullback_location": [
            "LOC_130_ShallowPullback_Long",
            "LOC_131_DeepPullback_Long",
            "LOC_132_ShallowPullback_Short",
            "LOC_133_DeepPullback_Short",
            "LOC_134_Fib382Zone_Long",
            "LOC_135_Fib50Zone_Long",
            "LOC_136_Fib618Zone_Long",
            "LOC_137_Fib382Zone_Short",
            "LOC_138_Fib50Zone_Short",
            "LOC_139_Fib618Zone_Short",
        ],
        "microstructure_location": [
            "LOC_150_AtPivotResistance",
            "LOC_151_AtPivotSupport",
            "LOC_152_BetweenPivotAndR1",
            "LOC_153_BetweenPivotAndS1",
            "LOC_154_AtLiquidityPoolHigh",
            "LOC_155_AtLiquidityPoolLow",
            "LOC_156_AtUntestedBreakoutLevel",
            "LOC_157_AtUntestedBreakdownLevel",
        ],
        "distance_location": [
            "LOC_170_NotTooExtendedAboveEMA",
        ],
    }
    candidates: List[Dict[str, Any]] = []
    for family, cols in family_buckets.items():
        for col in cols:
            if col.endswith("_Long"):
                allowed_modes = ("long",)
            elif col.endswith("_Short"):
                allowed_modes = ("short",)
            else:
                allowed_modes = ("long", "short")
            candidates.append(
                {
                    "kind": "location_filter",
                    "feature_base": col,
                    "feature_name": col,
                    "column_name": col,
                    "family": family,
                    "direction": "bool",
                    "threshold": 1.0,
                    "lookback_h": 0,
                    "allowed_modes": allowed_modes,
                }
            )
    return candidates


def _build_shared_location_filter_frame(
    shared: Dict[str, Any],
    asset_groups: Dict[int, np.ndarray],
    feature_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    if feature_dict:
        persisted_cols = {
            col: np.asarray(feature_dict[col], dtype=np.float32)
            for col in LOCATION_FILTER_COLUMNS
            if col in feature_dict
        }
        persisted_finite = 0
        persisted_positive = 0
        for arr in persisted_cols.values():
            finite = np.isfinite(arr)
            persisted_finite += int(np.sum(finite))
            persisted_positive += int(np.sum(arr[finite] > 0.0))
        if (
            len(persisted_cols) == len(LOCATION_FILTER_COLUMNS)
            and persisted_finite > 0
            and persisted_positive > 0
        ):
            return {
                col: (np.asarray(arr, dtype=np.float32) > 0.0).astype(np.int8)
                for col, arr in persisted_cols.items()
            }
        if len(persisted_cols) == len(LOCATION_FILTER_COLUMNS):
            tprint(
                "Persisted location filters unavailable for this sample; "
                "rebuilding from OHLCV because cached arrays are empty or all-NaN."
            )
    location_cols: Dict[str, np.ndarray] = {}
    day_ids = np.asarray(shared["day_ids"], dtype=np.int32)
    timestamps = pd.to_datetime(np.asarray(shared["timestamps"]))
    for idxs in asset_groups.values():
        local_df = pd.DataFrame(
            {
                "open": np.asarray(shared["open"])[idxs].astype(np.float32),
                "high": np.asarray(shared["high"])[idxs].astype(np.float32),
                "low": np.asarray(shared["low"])[idxs].astype(np.float32),
                "close": np.asarray(shared["close"])[idxs].astype(np.float32),
                "volume": np.asarray(shared.get("volume", np.ones(shared["close"].shape[0])))[idxs].astype(np.float32),
                "session_id": day_ids[idxs].astype(np.int32),
                "timestamp": timestamps[idxs],
            }
        )
        local_library = build_intraday_crypto_library(local_df)
        for col in [c for c in local_library.columns if c.startswith("LOC_")]:
            arr = local_library[col].to_numpy(copy=False)
            if col not in location_cols:
                location_cols[col] = np.zeros(shared["close"].shape[0], dtype=np.int8)
            location_cols[col][idxs] = np.asarray(arr, dtype=np.int8)
    return location_cols

def _build_asset_groups_from_codes(
    symbol_codes: np.ndarray,
    n_symbols: int,
) -> Dict[int, np.ndarray]:
    asset_groups: Dict[int, np.ndarray] = {}
    for aid in range(n_symbols):
        idxs = np.where(symbol_codes == aid)[0].astype(np.int32)
        if idxs.shape[0] > 0:
            asset_groups[int(aid)] = idxs
    return asset_groups



@njit
def _rolling_robust_z_1d(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full_like(x, np.nan)
    for i in range(window - 1, n):
        w = x[i - window + 1: i + 1]
        valid = w[np.isfinite(w)]
        if len(valid) > 0:
            med = np.median(valid)
            mad = np.median(np.abs(valid - med))

            if mad < 1e-12:
                # Fallback to standard deviation if MAD is extremely small (constant area)
                if len(valid) > 1:
                    std = np.std(valid)
                    denom = std if std > 1e-12 else 1e-6
                else:
                    denom = 1e-6
            else:
                denom = 1.4826 * mad + 1e-6

            z = (x[i] - med) / denom
            # Clamp to prevent explosion
            out[i] = max(min(z, 10.0), -10.0)
    return out

def compute_robust_z_for_groups(x: np.ndarray, asset_groups: Dict[int, np.ndarray], window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for _, idxs in asset_groups.items():
        out[idxs] = _rolling_robust_z_1d(x[idxs], window)
    return out

def _compute_z_cache(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ret_1: np.ndarray,
    vol_g: np.ndarray,
    asset_groups: Dict[int, np.ndarray],
    z: int,
    bph: int,
    volume: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    n = close.shape[0]
    cache = {
        "up": np.zeros(n, dtype=np.float32),
        "dn": np.zeros(n, dtype=np.float32),
        "rng": np.zeros(n, dtype=np.float32),
        "std_up": np.zeros(n, dtype=np.float32),
        "std_dn": np.zeros(n, dtype=np.float32),
        "b_up": np.zeros(n, dtype=np.float32),
        "b_dn": np.zeros(n, dtype=np.float32),
        "s_up": np.zeros(n, dtype=np.float32),
        "s_dn": np.zeros(n, dtype=np.float32),
        "m_up": np.zeros(n, dtype=np.float32),
        "m_dn": np.zeros(n, dtype=np.float32),
        "v_exp": np.zeros(n, dtype=np.float32),

        # New features
        "hl_range": np.zeros(n, dtype=np.float32),
        "intrabar_range_atr": np.zeros(n, dtype=np.float32),
        "compression_expansion_transition": np.zeros(n, dtype=np.float32),
        "volume_robust_z": np.zeros(n, dtype=np.float32),
        "breakout_distance_up_atr": np.zeros(n, dtype=np.float32),
        "breakout_distance_down_atr": np.zeros(n, dtype=np.float32),
        "distance_from_ema_atr": np.zeros(n, dtype=np.float32),
        "distance_from_vwap_atr": np.zeros(n, dtype=np.float32),
        "atr_normalized_trailing_return": np.zeros(n, dtype=np.float32),
        "short_minus_long_momentum": np.zeros(n, dtype=np.float32),
        "slope_change": np.zeros(n, dtype=np.float32),
        "path_efficiency_ratio": np.zeros(n, dtype=np.float32),
        "trailing_high_prev": np.zeros(n, dtype=np.float32),
        "trailing_low_prev": np.zeros(n, dtype=np.float32),
        "sma_z": np.zeros(n, dtype=np.float32),
        "vwap_z": np.zeros(n, dtype=np.float32),
    }

    for _, idxs in asset_groups.items():
        ast_high = high[idxs]
        ast_low = low[idxs]
        ast_close = close[idxs]
        ast_ret = ret_1[idxs]
        ast_vol = vol_g[idxs]

        hv, hi = rolling_max_index_nb(ast_high, z)
        lv, li = rolling_min_index_nb(ast_low, z)
        st_idx = np.maximum(0, np.arange(ast_close.shape[0], dtype=np.int32) - z + 1)
        st_px = ast_close[st_idx]

        um = np.where(st_px > 1e-9, (hv - st_px) / st_px, 0.0).astype(np.float32)
        dm = np.where(st_px > 1e-9, (st_px - lv) / st_px, 0.0).astype(np.float32)
        rm = np.where(st_px > 1e-9, (hv - lv) / st_px, 0.0).astype(np.float32)

        b_u, b_d, s_u, s_d, m_u, m_d, v_e = compute_impulse_coherence_nb(
            ast_ret, ast_vol, hv, lv, st_px, hi, li, st_idx, z
        )

        cache["up"][idxs] = um
        cache["dn"][idxs] = dm
        cache["rng"][idxs] = rm
        cache["std_up"][idxs] = rolling_std_nb(um, 30 * 24 * bph).astype(np.float32)
        cache["std_dn"][idxs] = rolling_std_nb(dm, 30 * 24 * bph).astype(np.float32)
        cache["b_up"][idxs] = b_u
        cache["b_dn"][idxs] = b_d
        cache["s_up"][idxs] = s_u
        cache["s_dn"][idxs] = s_d
        cache["m_up"][idxs] = m_u
        cache["m_dn"][idxs] = m_d
        cache["v_exp"][idxs] = v_e

        # --- New Features ---
        # Window must be smaller than the available data slice for fast iteration
        window_14d = int(14 * 24 * bph)
        window_adaptive = max(20, min(window_14d, int(ast_close.shape[0] * 0.5)))

        # 1. Volatility / Range
        ast_hl_range = ast_high - ast_low
        safe_close_vol = np.maximum(ast_close * ast_vol, 1e-6)
        cache["hl_range"][idxs] = _rolling_robust_z_1d(ast_hl_range, window_adaptive)

        # Use ast_vol (which is approx ATR percent) or calculate explicitly if needed.
        # Vol_g is the 14-day ATR pct. So intrabar_range_atr can be approximated.
        ast_intrabar_range_atr = np.where(ast_vol > 1e-6, (ast_high - ast_low) / safe_close_vol, 0.0)
        cache["intrabar_range_atr"][idxs] = _rolling_robust_z_1d(ast_intrabar_range_atr, window_adaptive)

        ast_bollinger_width = rolling_std_nb(ast_close, 20) / np.maximum(ast_close, 1e-6)
        ast_range_spike = ast_intrabar_range_atr
        # For simplicity, compression_expansion_transition is just range_spike / (bollinger_width + eps)
        ast_comp_exp = ast_range_spike / np.maximum(ast_bollinger_width, 1e-6)
        cache["compression_expansion_transition"][idxs] = _rolling_robust_z_1d(ast_comp_exp, window_adaptive)

        # 2. Volume
        if volume is not None:
            ast_vol_raw = volume[idxs]
        else:
            ast_vol_raw = np.ones_like(ast_close)
        cache["volume_robust_z"][idxs] = _rolling_robust_z_1d(ast_vol_raw, window_adaptive)

        # 3. Breakout / Structure
        # distance from trailing max high
        ast_trailing_high, _ = rolling_max_index_nb(ast_high, z)
        trailing_high_prev = np.roll(ast_trailing_high, 1)
        cache["trailing_high_prev"][idxs] = trailing_high_prev.astype(np.float32)
        ast_breakout_up = (ast_close - trailing_high_prev) / safe_close_vol
        cache["breakout_distance_up_atr"][idxs] = _rolling_robust_z_1d(ast_breakout_up, window_adaptive)

        ast_trailing_low, _ = rolling_min_index_nb(ast_low, z)
        trailing_low_prev = np.roll(ast_trailing_low, 1)
        cache["trailing_low_prev"][idxs] = trailing_low_prev.astype(np.float32)
        ast_breakout_dn = (trailing_low_prev - ast_close) / safe_close_vol
        cache["breakout_distance_down_atr"][idxs] = _rolling_robust_z_1d(ast_breakout_dn, window_adaptive)

        # 3.5 Stretch Location
        # distance_from_ema_atr
        # EMA over z bars. Simple SMA proxy if EMA is too slow inside numba, or we can use convolve.
        # Let's use SMA as a robust proxy for EMA over 'z' window to keep it vectorized here.
        sma_z = np.convolve(ast_close, np.ones(z)/z, mode='valid')
        sma_z = np.concatenate([np.full(z-1, np.nan), sma_z])
        cache["sma_z"][idxs] = sma_z.astype(np.float32)
        ast_dist_ema = (ast_close - sma_z) / safe_close_vol
        cache["distance_from_ema_atr"][idxs] = _rolling_robust_z_1d(ast_dist_ema, window_14d)

        # distance_from_vwap_atr
        # VWAP over z bars = sum(close * volume) / sum(volume)
        if volume is not None:
            vol_w = volume[idxs]
        else:
            vol_w = np.ones_like(ast_close)

        sum_vol_z = np.convolve(vol_w, np.ones(z), mode='valid')
        sum_vol_z = np.concatenate([np.full(z-1, np.nan), sum_vol_z])

        sum_pv_z = np.convolve(ast_close * vol_w, np.ones(z), mode='valid')
        sum_pv_z = np.concatenate([np.full(z-1, np.nan), sum_pv_z])

        vwap_z = sum_pv_z / np.maximum(sum_vol_z, 1e-6)
        cache["vwap_z"][idxs] = vwap_z.astype(np.float32)
        ast_dist_vwap = (ast_close - vwap_z) / safe_close_vol
        cache["distance_from_vwap_atr"][idxs] = _rolling_robust_z_1d(ast_dist_vwap, window_14d)

        # 4. Momentum
        ast_trailing_ret = (ast_close - np.roll(ast_close, z)) / np.maximum(np.roll(ast_close, z), 1e-6)
        ast_atr_norm_ret = ast_trailing_ret / np.maximum(ast_vol, 1e-6)
        cache["atr_normalized_trailing_return"][idxs] = _rolling_robust_z_1d(ast_atr_norm_ret, window_14d)

        # Short minus long
        short_ret = (ast_close - np.roll(ast_close, max(1, z//3))) / np.maximum(np.roll(ast_close, max(1, z//3)), 1e-6)
        cache["short_minus_long_momentum"][idxs] = _rolling_robust_z_1d(short_ret - ast_trailing_ret, window_14d)

        # Slope change (diff of rolling return)
        ast_slope_change = ast_trailing_ret - np.roll(ast_trailing_ret, 1)
        cache["slope_change"][idxs] = _rolling_robust_z_1d(ast_slope_change, window_14d)

        # 5. Path Structure
        # path efficiency = net move / sum of abs moves
        ast_abs_moves = np.abs(ast_close - np.roll(ast_close, 1))
        # We need a rolling sum for path efficiency. Safe rolling sum:
        rolling_abs_moves = np.convolve(ast_abs_moves, np.ones(z, dtype=int), 'valid')
        rolling_abs_moves = np.concatenate([np.full(z-1, np.nan), rolling_abs_moves])

        ast_path_eff = np.where(rolling_abs_moves > 1e-6, (ast_close - np.roll(ast_close, z)) / (rolling_abs_moves + 1e-9), 0.0)
        cache["path_efficiency_ratio"][idxs] = _rolling_robust_z_1d(ast_path_eff, window_14d)


    return cache


def _balanced_sample_indices(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    max_each: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_each <= 0:
        return idx_a, idx_b

    rng = np.random.RandomState(seed)

    def _sample(idx: np.ndarray) -> np.ndarray:
        if idx.shape[0] <= max_each:
            return idx
        sampled = rng.choice(idx, max_each, replace=False)
        sampled.sort()
        return sampled.astype(np.int32)

    return _sample(idx_a), _sample(idx_b)


def _cap_rows_for_optimization(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    def _balanced_recent_indices(
        symbols: np.ndarray, max_rows_local: int
    ) -> np.ndarray:
        n_rows = symbols.shape[0]
        if max_rows_local <= 0 or n_rows <= max_rows_local:
            return np.arange(n_rows, dtype=np.int32)

        unique_symbols, inverse = np.unique(symbols.astype(str), return_inverse=True)
        n_symbols = int(unique_symbols.shape[0])
        if n_symbols <= 1:
            start_idx_local = max(0, n_rows - max_rows_local)
            return np.arange(start_idx_local, n_rows, dtype=np.int32)

        base_quota = max_rows_local // n_symbols
        if base_quota <= 0:
            base_quota = 1
        selected = np.zeros(n_rows, dtype=bool)

        for code in range(n_symbols):
            idxs = np.flatnonzero(inverse == code)
            if idxs.size == 0:
                continue
            keep = min(int(idxs.size), base_quota)
            if keep > 0:
                selected[idxs[-keep:]] = True

        selected_count = int(selected.sum())
        if selected_count < max_rows_local:
            remainder = max_rows_local - selected_count
            residual = np.flatnonzero(~selected)
            if residual.size > 0:
                take = residual[-min(remainder, int(residual.size)) :]
                selected[take] = True

        return np.flatnonzero(selected).astype(np.int32)

    total_rows = data.shape[0]
    full_panel_rows = int(cfg.get("mask_opt_full_panel_rows", total_rows))
    max_rows_pct = float(cfg.get("mask_opt_max_rows_pct", 0.0))
    min_full_rows = _min_rows_from_full_panel(
        cfg,
        full_panel_rows=full_panel_rows,
        fraction_key="mask_opt_min_cap_full_panel_fraction",
        default_fraction=0.04,
    )
    
    if max_rows_pct > 0:
        max_rows = int(total_rows * min(max_rows_pct, 1.0))
        tprint(
            f"Adaptive capping: taking {max_rows} rows "
            f"({max_rows_pct*100:.1f}% of {total_rows}) with symbol-balanced recent sampling..."
        )
    else:
        max_rows = int(cfg.get("mask_opt_max_rows", 10_000))
        tprint(
            f"Fixed capping at {max_rows} rows "
            "(symbol-balanced recent sampling)..."
        )

    if min_full_rows > max_rows:
        tprint(
            "Raising optimization cap to preserve full-panel coverage: "
            f"rows={min_full_rows} ({_pct_str(min_full_rows, full_panel_rows)} of full panel)"
        )
        max_rows = min_full_rows

    if max_rows <= 0 or total_rows <= max_rows:
        return data, feature_dict, forward_returns

    if "symbol" in data.columns:
        symbol_values = data["symbol"].to_numpy()
        indices = _balanced_recent_indices(symbol_values, max_rows)
    else:
        start_idx = max(0, total_rows - max_rows)
        indices = np.arange(start_idx, total_rows, dtype=np.int32)

    data_capped = data.iloc[indices].reset_index(drop=True)
    forward_capped = forward_returns[indices]
    feature_dict_capped = {k: v[indices] for k, v in feature_dict.items()}
    return data_capped, feature_dict_capped, forward_capped


def _materialize_layer_runtime_cfg(
    cfg: Dict[str, Any], layer_name: str
) -> Dict[str, Any]:
    runtime_cfg = dict(cfg)
    if layer_name == "layer1":
        runtime_cfg["mask_opt_max_rows_pct"] = float(
            cfg.get("layer1_mask_opt_max_rows_pct", cfg.get("mask_opt_max_rows_pct", 0.15))
        )
        runtime_cfg["mask_opt_max_rows"] = int(
            cfg.get("layer1_mask_opt_max_rows", cfg.get("mask_opt_max_rows", 25_000))
        )
        runtime_cfg["phase1_classifier_max_samples_per_class"] = int(
            cfg.get(
                "layer1_phase1_classifier_max_samples_per_class",
                cfg.get("phase1_classifier_max_samples_per_class", 15_000),
            )
        )
        runtime_cfg["phase2_metric_max_samples_per_class"] = int(
            cfg.get(
                "layer1_phase2_metric_max_samples_per_class",
                cfg.get("phase2_metric_max_samples_per_class", 25_000),
            )
        )
        runtime_cfg["phase1_classifier_n_splits"] = int(
            cfg.get("layer1_phase1_classifier_n_splits", 3)
        )
        runtime_cfg["phase2_classifier_n_splits"] = int(
            cfg.get("layer1_phase2_classifier_n_splits", 4)
        )
        runtime_cfg["phase2_metric_fold_splits"] = int(
            cfg.get("layer1_phase2_metric_fold_splits", 4)
        )
        runtime_cfg["incremental_information_n_splits"] = int(
            cfg.get("layer1_incremental_information_n_splits", 4)
        )
    else:
        runtime_cfg["mask_opt_max_rows_pct"] = float(
            cfg.get("mask_opt_max_rows_pct", 0.15)
        )
        runtime_cfg["phase1_classifier_n_splits"] = int(
            cfg.get("phase1_classifier_n_splits", 3)
        )
        runtime_cfg["phase2_classifier_n_splits"] = int(
            cfg.get("phase2_classifier_n_splits", 4)
        )
        runtime_cfg["phase2_metric_fold_splits"] = int(
            cfg.get("phase2_metric_fold_splits", 4)
        )
        runtime_cfg["incremental_information_n_splits"] = int(
            cfg.get("incremental_information_n_splits", 4)
        )
    runtime_cfg["regime_score_layer"] = layer_name
    return runtime_cfg


def _rescale_mode_gates_for_sample_size(
    cfg: Dict[str, Any], n_rows: int
) -> Dict[str, Any]:
    runtime_cfg = dict(cfg)
    if n_rows <= 0:
        return runtime_cfg

    # Per-bucket masks are sparse on capped samples; fixed 5k-event gates are too high.
    target_event_density = float(
        runtime_cfg.get("mask_opt_target_event_density", 0.012)
    )
    min_events_floor = int(runtime_cfg.get("mask_opt_min_events_floor", 150))
    scaled_min_events = max(min_events_floor, int(round(n_rows * target_event_density)))

    base_active = float(runtime_cfg.get("phase2_min_active_days_fraction", 0.80))
    active_days_floor = float(runtime_cfg.get("mask_opt_min_active_days_floor", 0.25))
    scaled_active_days = min(
        base_active, max(active_days_floor, base_active * np.sqrt(n_rows / 300_000.0))
    )

    runtime_cfg["phase1_min_total_events"] = int(scaled_min_events * 0.25)
    runtime_cfg["phase2_min_total_events"] = int(scaled_min_events * 0.50)
    runtime_cfg["phase1_min_active_days_fraction"] = float(scaled_active_days * 0.50)
    runtime_cfg["phase2_min_active_days_fraction"] = float(scaled_active_days * 0.75)
    tprint(
        "Rescaled per-bucket gates for capped sample: "
        f"rows={n_rows}, min_events={scaled_min_events}, min_active_days_fraction={scaled_active_days:.3f}"
    )
    return runtime_cfg


def _generate_event_masks_fast(
    *,
    candidate: Optional[Dict[str, Any]] = None,
    zc: Optional[Dict[str, np.ndarray]] = None,
    asset_groups: Optional[Dict[int, np.ndarray]] = None,
    family: Optional[str] = None,
    param_val: Any = None,
    up_move: Optional[np.ndarray] = None,
    dn_move: Optional[np.ndarray] = None,
    rolling_std_up: Optional[np.ndarray] = None,
    rolling_std_dn: Optional[np.ndarray] = None,
    duration_bars: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if candidate is not None:
        if zc is None:
            raise ValueError("zc is required when candidate is provided")
        if candidate.get("kind") == "location_filter":
            col = str(candidate["column_name"])
            if col not in zc:
                raise ValueError(f"Location filter {col} not found in location frame")
            filter_mask = np.asarray(zc[col], dtype=np.int8) > 0
            allowed_modes = tuple(candidate.get("allowed_modes", ("long", "short")))
            if allowed_modes == ("long",):
                return filter_mask.copy(), np.zeros(filter_mask.shape[0], dtype=bool)
            if allowed_modes == ("short",):
                return np.zeros(filter_mask.shape[0], dtype=bool), filter_mask.copy()
            return filter_mask.copy(), filter_mask.copy()
        f_base = candidate["feature_base"]
        direction = candidate["direction"]
        threshold = candidate["threshold"]

        if f_base not in zc:
            raise ValueError(f"Feature {f_base} not found in zc cache!")

        feature_vals = zc[f_base]
        mask_h = np.zeros(feature_vals.shape[0], dtype=bool)
        mask_l = np.zeros(feature_vals.shape[0], dtype=bool)
        valid_mask = np.isfinite(feature_vals)

        if candidate["family"] in (
            "volatility_expansion",
            "compression_transition",
            "volume",
        ):
            if direction == "gt":
                trigger = valid_mask & (feature_vals >= threshold)
                up_arr = zc.get("m_up", np.zeros_like(feature_vals))
                dn_arr = zc.get("m_dn", np.zeros_like(feature_vals))
                mask_h = trigger & (up_arr >= dn_arr)
                mask_l = trigger & (dn_arr >= up_arr)
        elif candidate["family"] == "structure":
            if f_base == "breakout_distance_up_atr" and direction == "gt":
                mask_h = valid_mask & (feature_vals >= threshold)
            elif f_base == "breakout_distance_down_atr" and direction == "gt":
                mask_l = valid_mask & (feature_vals >= threshold)
        elif candidate["family"] in (
            "momentum",
            "stretch_location",
            "path_structure",
        ):
            if direction == "gt":
                mask_h = valid_mask & (feature_vals >= threshold)
            elif direction == "lt":
                mask_l = valid_mask & (feature_vals <= threshold)
        return mask_h, mask_l

    if (
        family is None
        or up_move is None
        or dn_move is None
        or rolling_std_up is None
        or rolling_std_dn is None
    ):
        raise ValueError("legacy mask generation requires family and rolling tensors")

    mask_h = np.zeros(up_move.shape[0], dtype=bool)
    mask_l = np.zeros(dn_move.shape[0], dtype=bool)

    std_up_floored = np.maximum(rolling_std_up, 1e-6)
    std_dn_floored = np.maximum(rolling_std_dn, 1e-6)

    if family == "std_threshold":
        x_std = float(param_val)
        mask_h = up_move >= (x_std * std_up_floored)
        mask_l = dn_move >= (x_std * std_dn_floored)
    elif family == "abs_move_threshold":
        y_move = float(param_val) / 100.0
        mask_h = up_move >= y_move
        mask_l = dn_move >= y_move
    elif family == "std_plus_abs":
        std_val, abs_val_pct = param_val
        y_move = float(abs_val_pct) / 100.0
        mask_h = (up_move >= float(std_val) * std_up_floored) & (up_move >= y_move)
        mask_l = (dn_move >= float(std_val) * std_dn_floored) & (dn_move >= y_move)
    else:
        raise ValueError(f"Unknown family: {family}")

    if duration_bars > 1 and asset_groups is not None:
        mask_h = dilate_mask_by_asset_safe(mask_h, asset_groups, duration_bars)
        mask_l = dilate_mask_by_asset_safe(mask_l, asset_groups, duration_bars)
    return mask_h, mask_l


def _generate_event_masks(
    family: str,
    param_val: Any,
    up_move: np.ndarray,
    dn_move: np.ndarray,
    rolling_std_up: np.ndarray,
    rolling_std_dn: np.ndarray,
    asset_groups: Optional[Dict[int, np.ndarray]] = None,
    duration_bars: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for event mask generation.

    Several inference modules import ``_generate_event_masks`` directly.
    Keep this stable alias so those modules continue to import successfully
    after the fast path refactor.
    """
    return _generate_event_masks_fast(
        family=family,
        param_val=param_val,
        up_move=up_move,
        dn_move=dn_move,
        rolling_std_up=rolling_std_up,
        rolling_std_dn=rolling_std_dn,
        asset_groups=asset_groups,
        duration_bars=duration_bars,
    )


def _simple_score_for_mode(
    mode: str,
    feature_dict: Dict[str, np.ndarray],
    side_mask: np.ndarray,
) -> np.ndarray:
    n = side_mask.shape[0]
    score = np.zeros(n, dtype=np.float32)

    def get(name: str) -> np.ndarray:
        if name not in feature_dict:
            return np.zeros(n, dtype=np.float32)
        return np.nan_to_num(
            np.asarray(feature_dict[name], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    # very simple fixed interpretable score
    impulse = get("momentum_last_3bars_impulse_return")
    vol = get("climax_volume_ratio")
    rev = get("reversal_bar_strength")
    rng = get("range_1_atr")
    entropy = get("bar_direction_entropy")

    if mode == MODE_LONG:
        # Default to a balanced TF/reversal proxy for LONG
        score = 0.35 * impulse + 0.20 * vol + 0.20 * rng - 0.15 * rev - 0.10 * entropy
    elif mode == MODE_SHORT:
        # Default to a balanced TF/reversal proxy for SHORT
        score = 0.35 * (-impulse) + 0.20 * vol + 0.20 * rng - 0.15 * rev - 0.10 * entropy
    else:
        score = np.zeros(n, dtype=np.float32)

    score[~side_mask] = np.nan
    return score.astype(np.float32)


# =============================================================================
# SHARED CACHE BUILD
# =============================================================================


def _build_shared_cache(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    tprint("Building shared cache...")
    bph = int(cfg.get("bars_per_hour", 1))
    horizon = int(cfg.get("phase1_forward_horizon_bars", 12))
    
    tprint(f"  - Converting data to numpy arrays...")

    open_ = np.asarray(data["open"].values, dtype=np.float32) if "open" in data.columns else np.full(data.shape[0], np.nan, dtype=np.float32)
    high = np.asarray(data["high"].values, dtype=np.float32)
    low = np.asarray(data["low"].values, dtype=np.float32)
    close = np.asarray(data["close"].values, dtype=np.float32)
    volume = np.asarray(data["volume"].values, dtype=np.float32) if "volume" in data.columns else np.full(data.shape[0], np.nan, dtype=np.float32)
    timestamps = pd.to_datetime(data["timestamp"]).values
    symbols = np.asarray(data["symbol"].astype(str).values)
    _validate_long_panel_shape(timestamps, symbols, require_rectangular=False)

    forward_returns = np.asarray(forward_returns, dtype=np.float32)
    atr = np.asarray(
        feature_dict.get("atr", np.ones_like(close, dtype=np.float32)), dtype=np.float32
    )

    # ids
    tprint(f"  - Building IDs (symbol, day, timestamp, regime)...")
    symbol_uniques, symbol_codes = np.unique(symbols, return_inverse=True)
    symbol_codes = symbol_codes.astype(np.int32)
    day_ids, n_days = _build_day_ids(timestamps)
    timestamp_ids, n_timestamps = _build_timestamp_ids(timestamps)
    regime_source = np.asarray(
        feature_dict.get("vol_regime_z", np.zeros_like(close, dtype=np.float32)),
        dtype=np.float32,
    )
    if regime_source.shape[0] != close.shape[0]:
        regime_source = np.zeros_like(close, dtype=np.float32)
    regime_ids = _build_vol_regime_ids(regime_source)

    # per-asset groups
    tprint(f"  - Building asset groups for {len(symbol_uniques)} symbols...")
    asset_groups = _build_asset_groups_from_codes(symbol_codes, symbol_uniques.shape[0])
    tprint("  - Building shared location-filter library...")
    location_filter_frame = _build_shared_location_filter_frame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "day_ids": day_ids,
            "timestamps": timestamps,
        },
        asset_groups,
        feature_dict=feature_dict,
    )

    # returns / vol / alternation
    tprint(f"  - Computing returns, volatility, and alternation per asset...")
    ret_1 = np.zeros(close.shape[0], dtype=np.float32)
    vol_g = np.zeros(close.shape[0], dtype=np.float32)
    alternation = np.zeros(close.shape[0], dtype=np.float32)

    for aid, idxs in asset_groups.items():
        c = close[idxs]
        r = np.zeros(idxs.shape[0], dtype=np.float32)
        if idxs.shape[0] > 1:
            prev = np.where(c[:-1] > 1e-9, c[:-1], 1.0)
            r[1:] = ((c[1:] - c[:-1]) / prev).astype(np.float32)
        ret_1[idxs] = r
        vol_g[idxs] = rolling_std_nb(r, 30 * 24 * bph).astype(np.float32)

        sign = np.sign(r).astype(np.float32)
        prev_sign = np.zeros_like(sign)
        if sign.shape[0] > 1:
            prev_sign[1:] = sign[:-1]
        alt = (sign != prev_sign).astype(np.float32)
        # lightweight rolling mean
        window = 6
        s = 0.0
        for i in range(alt.shape[0]):
            s += alt[i]
            if i >= window:
                s -= alt[i - window]
            alternation[idxs[i]] = s / float(min(i + 1, window))

    # MAE / MFE
    tprint(f"  - Computing MAE/MFE statistics per asset...")
    n = close.shape[0]
    mae_high = np.zeros(n, dtype=np.float32)
    mfe_high = np.zeros(n, dtype=np.float32)
    mae_low = np.zeros(n, dtype=np.float32)
    mfe_low = np.zeros(n, dtype=np.float32)
    # NaN => no full forward horizon available for this row.
    mfe_atr = np.full(n, np.nan, dtype=np.float32)
    mae_atr = np.full(n, np.nan, dtype=np.float32)

    # compute by asset to avoid cross-asset leakage
    for aid, idxs in asset_groups.items():
        h = high[idxs]
        l = low[idxs]
        c = close[idxs]
        a = atr[idxs]
        m1 = np.zeros(idxs.shape[0], dtype=np.float32)
        m2 = np.zeros(idxs.shape[0], dtype=np.float32)
        m3 = np.zeros(idxs.shape[0], dtype=np.float32)
        m4 = np.zeros(idxs.shape[0], dtype=np.float32)
        for i in range(max(0, idxs.shape[0] - horizon)):
            h_sl = h[i + 1 : i + horizon + 1]
            l_sl = l[i + 1 : i + horizon + 1]
            if h_sl.shape[0] == 0:
                continue
            atr_i = max(a[i], 1e-9)
            c_i = c[i]
            mfe_atr[idxs[i]] = (np.max(h_sl) - c_i) / atr_i
            mae_atr[idxs[i]] = (c_i - np.min(l_sl)) / atr_i
            m1[i] = (c_i - np.min(l_sl)) / atr_i
            m2[i] = (np.max(h_sl) - c_i) / atr_i
            m3[i] = (np.max(h_sl) - c_i) / atr_i
            m4[i] = (c_i - np.min(l_sl)) / atr_i
        mae_high[idxs] = m1
        mfe_high[idxs] = m2
        mae_low[idxs] = m3
        mfe_low[idxs] = m4

    # learnability features
    learn_X = _extract_learnability_features(feature_dict, n)
    feature_dict_float32_items = [
        (name, np.asarray(arr, dtype=np.float32))
        for name, arr in feature_dict.items()
        if np.asarray(arr).shape[0] == n
    ]

    # full folds
    n_splits = int(cfg.get("phase2_classifier_n_splits", 4))
    folds = _build_temporal_folds(timestamps, n, n_splits=n_splits, symbols=symbols)

    return {
        "bph": bph,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "timestamps": timestamps,
        "symbols": symbols,
        "symbol_uniques": symbol_uniques,
        "symbol_codes": symbol_codes,
        "asset_groups": asset_groups,
        "forward_returns": forward_returns,
        "atr": atr,
        "ret_1": ret_1,
        "vol_g": vol_g,
        "alternation": alternation,
        "mae_high": mae_high,
        "mfe_high": mfe_high,
        "mae_low": mae_low,
        "mfe_low": mfe_low,
        "mfe_atr": mfe_atr,
        "mae_atr": mae_atr,
        "learn_X": learn_X,
        "feature_dict_float32_items": feature_dict_float32_items,
        "day_ids": day_ids,
        "n_days": n_days,
        "timestamp_ids": timestamp_ids,
        "n_timestamps": n_timestamps,
        "regime_ids": regime_ids,
        "folds": folds,
        "tbm_geometry_cache": {},
        "z_grid": sorted(
            set(int(z * bph) for z in cfg.get("z_hours_grid", [6, 10, 16]))
        ),
        "candidate_grid": _build_candidate_grid(cfg),
        "location_filter_frame": location_filter_frame,
        "volume": volume,
    }


# =============================================================================
# PHASE 1 + PHASE 2
# =============================================================================


def _phase1_subsample_indices(
    shared: Dict[str, Any], cfg: Dict[str, Any], seed: int = 42
) -> np.ndarray:
    symbol_codes = shared["symbol_codes"]
    n_total = symbol_codes.shape[0]

    full_panel_rows = int(cfg.get("mask_opt_full_panel_rows", n_total))
    min_phase1_rows = _min_rows_from_full_panel(
        cfg,
        full_panel_rows=full_panel_rows,
        fraction_key="phase1_min_full_panel_fraction",
        default_fraction=0.04,
    )
    configured_max_phase1_rows = int(cfg.get("phase1_max_subsample_rows", 20_000))
    max_phase1_rows = max(configured_max_phase1_rows, min_phase1_rows)

    if n_total <= max_phase1_rows:
        return np.ones(n_total, dtype=bool)

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=max_phase1_rows, replace=False)
    result = np.zeros(n_total, dtype=bool)
    result[indices] = True
    return result


def _build_phase_local_shared(
    shared: Dict[str, Any],
    subset_mask: np.ndarray,
) -> Dict[str, Any]:
    symbol_codes_local = shared["symbol_codes"][subset_mask]
    day_ids_local_raw = shared["day_ids"][subset_mask]
    _, day_ids_local = np.unique(day_ids_local_raw, return_inverse=True)
    open_values = np.asarray(
        shared.get("open", np.full(shared["close"].shape[0], np.nan, dtype=np.float32)),
        dtype=np.float32,
    )
    volume_values = np.asarray(
        shared.get("volume", np.full(shared["close"].shape[0], np.nan, dtype=np.float32)),
        dtype=np.float32,
    )
    phase_local = {
        "open": open_values[subset_mask],
        "high": shared["high"][subset_mask],
        "low": shared["low"][subset_mask],
        "close": shared["close"][subset_mask],
        "volume": volume_values[subset_mask],
        "ret_1": shared["ret_1"][subset_mask],
        "vol_g": shared["vol_g"][subset_mask],
        "timestamps": shared["timestamps"][subset_mask],
        "forward_returns": shared["forward_returns"][subset_mask],
        "mae_high": shared["mae_high"][subset_mask],
        "mfe_high": shared["mfe_high"][subset_mask],
        "mae_low": shared["mae_low"][subset_mask],
        "mfe_low": shared["mfe_low"][subset_mask],
        "day_ids": day_ids_local.astype(np.int32),
        "symbol_codes": symbol_codes_local,
        "asset_groups": _build_asset_groups_from_codes(
            symbol_codes_local, shared["symbol_uniques"].shape[0]
        ),
    }
    phase_local["n_days"] = (
        int(np.max(phase_local["day_ids"]) + 1)
        if phase_local["day_ids"].shape[0] > 0
        else 0
    )
    return phase_local


def _slice_z_cache(
    z_cache: Dict[str, np.ndarray],
    subset_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {key: np.asarray(values)[subset_mask] for key, values in z_cache.items()}


def _trim_z_cache(
    z_cache: Dict[int, Dict[str, np.ndarray]], keep_z_values: set[int]
) -> None:
    for z in list(z_cache.keys()):
        if z not in keep_z_values:
            del z_cache[z]


def _pct_str(numer: int, denom: int) -> str:
    if denom <= 0:
        return "n/a"
    return f"{100.0 * float(numer) / float(denom):.2f}%"


def _tprint_retention_step(
    label: str,
    rows: int,
    full_rows: int,
    prev_rows: Optional[int] = None,
) -> None:
    parts = [
        f"{label}: rows={rows}",
        f"pct_full={_pct_str(rows, full_rows)}",
    ]
    if prev_rows is not None:
        parts.append(f"pct_prev={_pct_str(rows, prev_rows)}")
    tprint(" | ".join(parts))


def _min_rows_from_full_panel(
    cfg: Dict[str, Any],
    full_panel_rows: int,
    fraction_key: str,
    default_fraction: float,
) -> int:
    if full_panel_rows <= 0:
        return 0
    fraction = float(cfg.get(fraction_key, default_fraction))
    fraction = min(max(fraction, 0.0), 1.0)
    return int(np.ceil(float(full_panel_rows) * fraction))


def _compute_primary_phase1_classifier_gain(
    mode: str,
    side_mask: np.ndarray,
    learn_X: np.ndarray,
    forward_returns: np.ndarray,
    timestamps: np.ndarray,
    ret_threshold: float,
    max_samples_per_class: int = 0,
    n_splits: int = 2,
) -> float:
    y_global = _mode_primary_target(mode, forward_returns, ret_threshold)
    valid = np.isfinite(forward_returns)
    idx_ne = np.where(valid & ~side_mask)[0].astype(np.int32)
    idx_e = np.where(valid & side_mask)[0].astype(np.int32)

    if idx_e.shape[0] < 50 or idx_ne.shape[0] < 50:
        return float("nan")

    idx_e, idx_ne = _balanced_sample_indices(
        idx_e, idx_ne, max_samples_per_class, seed=42
    )
    auc_ne = _classifier_oof_auc(
        learn_X[idx_ne], y_global[idx_ne], timestamps[idx_ne], n_splits=n_splits
    )
    auc_e = _classifier_oof_auc(
        learn_X[idx_e], y_global[idx_e], timestamps[idx_e], n_splits=n_splits
    )
    return float(auc_e - auc_ne)


def _compute_phase3_feature_learnability(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    bundle: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    min_pos_frac = float(cfg.get("min_feature_positive_fold_fraction", 0.60))
    top_k = int(cfg.get("feature_learnability_top_k", 10))
    y = np.asarray(bundle["signed_returns"], dtype=np.float32)
    regime = np.asarray(bundle["side_mask_valid"], dtype=bool)
    folds = bundle["folds_valid"]

    surviving_lifts: List[float] = []
    surviving_pos_frac: List[float] = []
    per_feature_top: List[Tuple[str, float]] = []

    feature_items = _get_valid_feature_items(shared, feature_dict, bundle["valid_idx"])

    for fname, x in feature_items:
        fold_lifts: List[float] = []

        for tr, va in folds:
            valid = np.isfinite(x)
            tr_reg = tr[regime[tr] & valid[tr]]
            va_reg = va[regime[va] & valid[va]]
            tr_non = tr[(~regime[tr]) & valid[tr]]
            va_non = va[(~regime[va]) & valid[va]]
            tr_full = tr[valid[tr]]
            va_full = va[valid[va]]

            if (
                tr_reg.shape[0] < 10
                or va_reg.shape[0] < 10
                or tr_non.shape[0] < 10
                or va_non.shape[0] < 10
            ):
                continue

            reg_r2 = _single_feature_fold_r2(x, y, tr_reg, va_reg)
            non_r2 = _single_feature_fold_r2(x, y, tr_non, va_non)
            full_r2 = _single_feature_fold_r2(x, y, tr_full, va_full)
            baseline_r2 = max(non_r2, full_r2)
            if not np.isfinite(reg_r2) or not np.isfinite(baseline_r2):
                continue
            fold_lifts.append(float(reg_r2 - baseline_r2))
        if not fold_lifts:
            continue
        folds_arr = np.asarray(fold_lifts, dtype=np.float32)
        mean_lift = float(np.mean(folds_arr))
        pos_frac = float(np.mean(folds_arr > 0.0))
        if mean_lift > 0.0 and pos_frac >= min_pos_frac:
            surviving_lifts.append(mean_lift)
            surviving_pos_frac.append(pos_frac)
            per_feature_top.append((fname, mean_lift))

    if surviving_lifts:
        top_vals = np.sort(np.asarray(surviving_lifts, dtype=np.float32))[-top_k:]
        gain = float(np.mean(top_vals))
        top_pairs = sorted(per_feature_top, key=lambda t: t[1], reverse=True)[:top_k]
    else:
        gain = 0.0
        top_pairs = []
    return {
        "feature_learnability_gain": np.float32(gain),
        "top_feature_lifts": ";".join([f"{k}:{v:.6f}" for k, v in top_pairs]),
        "feature_positive_fold_fraction": np.float32(
            float(np.mean(surviving_pos_frac)) if surviving_pos_frac else 0.0
        ),
    }


def _compute_conditional_predictability_metrics(
    bundle: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, np.float32]:
    y = np.asarray(bundle["signed_returns"], dtype=np.float32)
    X = np.asarray(bundle["learn_X"], dtype=np.float32)
    regime = np.asarray(bundle["side_mask_valid"], dtype=bool)
    nonregime = ~regime
    folds = bundle["folds_valid"]
    max_subset = int(cfg.get("phase2_metric_max_samples_per_class", 25_000))

    gain_folds: List[float] = []
    spread_folds: List[float] = []
    regime_r2_vals: List[float] = []
    baseline_r2_vals: List[float] = []

    for tr, va in folds:
        tr_reg = _cap_index_count(tr[regime[tr]], max_subset)
        va_reg = _cap_index_count(va[regime[va]], max_subset)
        tr_non = _cap_index_count(tr[nonregime[tr]], max_subset)
        va_non = _cap_index_count(va[nonregime[va]], max_subset)
        tr_full = _cap_index_count(tr, max_subset)
        va_full = _cap_index_count(va, max_subset)

        reg_r2, reg_spread = _ridge_subset_fold_metrics(X, y, tr_reg, va_reg)
        non_r2, _ = _ridge_subset_fold_metrics(X, y, tr_non, va_non)
        full_r2, _ = _ridge_subset_fold_metrics(X, y, tr_full, va_full)
        baseline_r2 = max(
            _metric_or_nan(non_r2),
            _metric_or_nan(full_r2),
        )
        if not np.isfinite(reg_r2) or not np.isfinite(baseline_r2):
            continue

        gain_folds.append(float(reg_r2 - baseline_r2))
        regime_r2_vals.append(float(reg_r2))
        baseline_r2_vals.append(float(baseline_r2))
        if np.isfinite(reg_spread):
            spread_folds.append(float(reg_spread))

    gain_arr = np.asarray(gain_folds, dtype=np.float32)
    spread_arr = np.asarray(spread_folds, dtype=np.float32)
    regime_arr = np.asarray(regime_r2_vals, dtype=np.float32)
    baseline_arr = np.asarray(baseline_r2_vals, dtype=np.float32)

    return {
        "conditional_predictability_gain": np.float32(
            float(np.mean(gain_arr)) if gain_arr.size > 0 else 0.0
        ),
        "conditional_predictability_positive_fold_fraction": np.float32(
            float(np.mean(gain_arr > 0.0)) if gain_arr.size > 0 else 0.0
        ),
        "conditional_predictability_regime_r2": np.float32(
            float(np.mean(regime_arr)) if regime_arr.size > 0 else 0.0
        ),
        "conditional_predictability_baseline_r2": np.float32(
            float(np.mean(baseline_arr)) if baseline_arr.size > 0 else 0.0
        ),
        "feature_conditioned_spread": np.float32(
            float(np.mean(spread_arr)) if spread_arr.size > 0 else 0.0
        ),
    }


def _get_tbm_geometry_outcomes(
    shared: Dict[str, Any],
    cfg: Dict[str, Any],
    tp_atr: float,
    sl_atr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (float(tp_atr), float(sl_atr))
    cache = shared.setdefault("tbm_geometry_cache", {})
    if key in cache:
        return cache[key]

    horizon = int(cfg.get("phase1_forward_horizon_bars", 12))
    close = np.asarray(shared["close"], dtype=np.float32)
    high = np.asarray(shared["high"], dtype=np.float32)
    low = np.asarray(shared["low"], dtype=np.float32)
    atr = np.asarray(shared["atr"], dtype=np.float32)
    n = close.shape[0]
    tp_first = np.zeros(n, dtype=np.int8)
    sl_first = np.zeros(n, dtype=np.int8)
    timeout = np.zeros(n, dtype=np.int8)

    for _, idxs in shared["asset_groups"].items():
        if idxs.shape[0] <= horizon + 1:
            continue
        tp_l, sl_l, to_l = tbm_outcomes_atr_nb(
            close[idxs],
            high[idxs],
            low[idxs],
            atr[idxs],
            horizon,
            float(tp_atr),
            float(sl_atr),
        )
        tp_first[idxs] = tp_l
        sl_first[idxs] = sl_l
        timeout[idxs] = to_l

    cache[key] = (tp_first, sl_first, timeout)
    return cache[key]


def _compute_tbm_economic_gain(
    shared: Dict[str, Any],
    side_mask: np.ndarray,
    mode: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    fee = float(cfg.get("round_trade_fee", 0.003))
    eval_cache = _get_eval_design_cache(shared)
    eval_idx = eval_cache["eval_idx"]
    folds_eval = eval_cache["folds_eval"]
    side_eval = np.asarray(side_mask, dtype=bool)[eval_idx]
    valid = side_eval
    baseline_mask = ~side_eval
    mfe_eval = np.asarray(shared["mfe_atr"], dtype=np.float32)[eval_idx]
    mae_eval = np.asarray(shared["mae_atr"], dtype=np.float32)[eval_idx]

    geometries = ((1.25, 0.50), (1.50, 0.60), (1.75, 0.70), (2.00, 0.90), (2.50, 1.10))
    per_geometry_metrics: List[Dict[str, Any]] = []
    geom_scores: List[float] = []
    cov_weights: List[float] = []
    cov_values: List[float] = []
    any_geometry_resolved_mask = np.zeros(eval_idx.shape[0], dtype=bool)
    any_geometry_labels = np.full(eval_idx.shape[0], np.nan, dtype=np.float32)
    min_regime_subset = int(cfg.get("phase4_tbm_lgbm_min_regime_subset", 40))
    min_regime_class_count = int(
        cfg.get("phase4_tbm_lgbm_min_regime_class_count", 2)
    )

    for tp_atr, sl_atr in geometries:
        tp_first, sl_first, timeout = _get_tbm_geometry_outcomes(
            shared, cfg, float(tp_atr), float(sl_atr)
        )
        tp_eval = tp_first[eval_idx]
        sl_eval = sl_first[eval_idx]
        timeout_eval = timeout[eval_idx]

        tp_rate_g = float(np.mean(tp_eval[valid])) if np.any(valid) else 0.0
        sl_rate_g = float(np.mean(sl_eval[valid])) if np.any(valid) else 0.0
        timeout_rate_g = float(np.mean(timeout_eval[valid])) if np.any(valid) else 1.0
        trade_rate_g = tp_rate_g + sl_rate_g
        resolved_mask = tp_eval.astype(bool) | sl_eval.astype(bool)
        labels_resolved = np.full(eval_idx.shape[0], np.nan, dtype=np.float32)
        labels_resolved[tp_eval.astype(bool)] = 1.0
        labels_resolved[sl_eval.astype(bool)] = 0.0
        newly_resolved = resolved_mask & (~any_geometry_resolved_mask)
        any_geometry_labels[newly_resolved] = labels_resolved[newly_resolved]
        any_geometry_resolved_mask |= resolved_mask
        regime_resolved_mask = valid & resolved_mask
        regime_subset_count = int(np.sum(regime_resolved_mask))
        regime_positive_count = int(np.sum(tp_eval[regime_resolved_mask] > 0))
        regime_negative_count = int(np.sum(sl_eval[regime_resolved_mask] > 0))
        phase4_regime_lgbm_eligible = bool(
            regime_subset_count >= min_regime_subset
            and regime_positive_count >= min_regime_class_count
            and regime_negative_count >= min_regime_class_count
        )
        ev_event_g = tp_rate_g * tp_atr - sl_rate_g * sl_atr - trade_rate_g * fee
        ev_trade_g = ev_event_g / max(trade_rate_g, 1e-9)
        win_rate_g = tp_rate_g / max(trade_rate_g, 1e-9)
        value_per_trade_g = ev_trade_g * win_rate_g

        base_tp = (
            float(np.mean(tp_eval[baseline_mask])) if np.any(baseline_mask) else 0.0
        )
        base_sl = (
            float(np.mean(sl_eval[baseline_mask])) if np.any(baseline_mask) else 0.0
        )
        base_trade = base_tp + base_sl
        base_ev_event = base_tp * tp_atr - base_sl * sl_atr - base_trade * fee
        baseline_ev_trade = base_ev_event / max(base_trade, 1e-9)
        lift_g = ev_trade_g - baseline_ev_trade

        mfe_cov_g = (
            float(np.mean(mfe_eval[valid] >= np.float32(tp_atr)))
            if np.any(valid)
            else 0.0
        )
        mae_pressure_g = (
            float(np.mean(mae_eval[valid] >= np.float32(sl_atr)))
            if np.any(valid)
            else 1.0
        )

        fold_ev: List[float] = []
        fold_lift: List[float] = []
        fold_trade: List[float] = []
        for _, va in folds_eval:
            vv = valid[va]
            if not np.any(vv):
                continue
            tp_f = float(np.mean(tp_eval[va][vv]))
            sl_f = float(np.mean(sl_eval[va][vv]))
            tr_f = tp_f + sl_f
            ev_f = (tp_f * tp_atr - sl_f * sl_atr - tr_f * fee) / max(tr_f, 1e-9)
            base_v = baseline_mask[va]
            btp_f = float(np.mean(tp_eval[va][base_v])) if np.any(base_v) else 0.0
            bsl_f = float(np.mean(sl_eval[va][base_v])) if np.any(base_v) else 0.0
            btr_f = btp_f + bsl_f
            bev_f = (btp_f * tp_atr - bsl_f * sl_atr - btr_f * fee) / max(btr_f, 1e-9)
            fold_ev.append(ev_f)
            fold_lift.append(ev_f - bev_f)
            fold_trade.append(tr_f)

        fold_ev_arr = np.asarray(fold_ev, dtype=np.float32)
        if fold_ev_arr.size > 0:
            econ_stability_g = 0.5 * max(
                0.0,
                1.0
                - float(np.std(fold_ev_arr))
                / (abs(float(np.mean(fold_ev_arr))) + 1e-9),
            )
            econ_stability_g += 0.5 * float(np.mean(fold_ev_arr > 0.0))
        else:
            econ_stability_g = 0.0
        opportunity_adjustment_g = (
            min(1.0, np.sqrt(trade_rate_g / 0.20)) if trade_rate_g > 0 else 0.0
        )
        geometry_score_g = (
            (
                (0.35 * value_per_trade_g)
                + (0.25 * lift_g)
                + (0.15 * trade_rate_g)
                + (0.15 * max(0.0, mfe_cov_g - 0.25))
                - (0.10 * mae_pressure_g)
            )
            * opportunity_adjustment_g
            * econ_stability_g
        )
        phase4_selection_score_g = (
            float(geometry_score_g) if phase4_regime_lgbm_eligible else float("-inf")
        )

        per_geometry_metrics.append(
            {
                "tp_atr": float(tp_atr),
                "sl_atr": float(sl_atr),
                "tp_first_rate_g": tp_rate_g,
                "sl_first_rate_g": sl_rate_g,
                "timeout_rate_g": timeout_rate_g,
                "trade_opportunity_rate_g": trade_rate_g,
                "ev_net_per_trade_g": ev_trade_g,
                "lift_g": lift_g,
                "mfe_coverage_g": mfe_cov_g,
                "mae_breach_pressure_g": mae_pressure_g,
                "econ_stability_g": econ_stability_g,
                "geometry_score_g": geometry_score_g,
                "phase4_selection_score_g": phase4_selection_score_g,
                "phase4_regime_lgbm_eligible": phase4_regime_lgbm_eligible,
                "regime_subset_count": regime_subset_count,
                "regime_positive_count": regime_positive_count,
                "regime_negative_count": regime_negative_count,
                "ev_net_per_trade_g_fold": fold_ev,
                "lift_g_fold": fold_lift,
                "trade_opportunity_rate_g_fold": fold_trade,
                "labels": tp_first.astype(np.float32),
                "labels_resolved": labels_resolved,
                "resolved_mask": resolved_mask.astype(np.int8),
            }
        )
        geom_scores.append(float(geometry_score_g))
        cov_weights.append(max(trade_rate_g, 1e-9))
        cov_values.append(mfe_cov_g)

    top3 = np.sort(np.asarray(geom_scores, dtype=np.float32))[-3:]
    economic_gain_r = (
        0.7 * float(np.mean(top3)) + 0.3 * float(np.min(top3)) if top3.size > 0 else 0.0
    )
    aggregate_mfe_coverage = (
        float(
            np.average(
                np.asarray(cov_values, dtype=np.float32),
                weights=np.asarray(cov_weights, dtype=np.float32),
            )
        )
        if cov_values
        else 0.0
    )
    return {
        "economic_gain_r": np.float32(economic_gain_r),
        "geometry_weighted_mfe_coverage": np.float32(aggregate_mfe_coverage),
        "aggregate_mfe_coverage": np.float32(aggregate_mfe_coverage),
        "per_geometry_metrics": per_geometry_metrics,
        "any_geometry_resolved_mask": any_geometry_resolved_mask.astype(np.int8),
        "any_geometry_labels": any_geometry_labels,
    }


def _compute_phase4_tbm_lgbm_metrics(
    shared: Dict[str, Any],
    side_mask: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    per_geometry_metrics: List[Dict[str, Any]],
) -> Dict[str, np.float32 | str]:
    out: Dict[str, np.float32 | str] = {
        "tbm_lgbm_auc_regime": np.float32(np.nan),
        "tbm_lgbm_auc_baseline": np.float32(np.nan),
        "tbm_lgbm_auc_lift_vs_baseline": np.float32(np.nan),
        "tbm_lgbm_top_bucket_lift_vs_baseline": np.float32(np.nan),
        "tbm_lgbm_positive_fold_fraction": np.float32(np.nan),
        "tbm_lgbm_stability": np.float32(np.nan),
        "tbm_lgbm_selected_geometry": "none",
        "tbm_lgbm_invalid_reason_regime": "",
        "tbm_lgbm_invalid_reason_baseline": "",
        "tbm_lgbm_invalid_reason_full": "",
    }
    if not per_geometry_metrics:
        return out

    eval_cache = _get_eval_design_cache(shared)
    X = eval_cache["eval_X"]
    timestamps = eval_cache["eval_timestamps"]
    symbol_codes = eval_cache["eval_symbols"]
    eval_idx = eval_cache["eval_idx"]
    max_subset = int(
        cfg.get(
            "phase4_tbm_lgbm_max_subset",
            cfg.get("phase2_metric_max_samples_per_class", 25_000),
        )
    )
    n_splits = min(3, int(cfg.get("phase4_tbm_lgbm_n_splits", 3)))
    eligible_geoms = [
        g for g in per_geometry_metrics if bool(g.get("phase4_regime_lgbm_eligible", True))
    ]
    if not eligible_geoms:
        out["tbm_lgbm_invalid_reason_regime"] = "no_phase4_geometry_with_sufficient_regime_support"
        return out
    sorted_geoms = sorted(
        eligible_geoms,
        key=lambda g: float(g.get("phase4_selection_score_g", g.get("geometry_score_g", float("-inf")))),
        reverse=True,
    )

    reg_metrics: Dict[str, Any] | None = None
    non_metrics: Dict[str, Any] | None = None
    full_metrics: Dict[str, Any] | None = None
    tp_atr = 1.25
    sl_atr = 0.50
    for geom in sorted_geoms:
        tp_atr = float(geom.get("tp_atr", 1.25))
        sl_atr = float(geom.get("sl_atr", 0.50))
        tp_first, sl_first, _ = _get_tbm_geometry_outcomes(shared, cfg, tp_atr, sl_atr)
        tp_eval = tp_first[eval_idx]
        sl_eval = sl_first[eval_idx]
        resolved_mask = tp_eval.astype(bool) | sl_eval.astype(bool)
        side_eval = np.asarray(side_mask, dtype=bool)[eval_idx]
        regime = side_eval & resolved_mask
        nonregime = (~side_eval) & resolved_mask
        full = resolved_mask
        y = (tp_eval.astype(np.float32) > 0.5).astype(np.float32)

        idx_reg = np.where(regime)[0].astype(np.int32)
        idx_non = np.where(nonregime)[0].astype(np.int32)
        idx_full = np.where(full)[0].astype(np.int32)
        reg_metrics = _lgbm_subset_cv_metrics(
            X,
            y,
            timestamps,
            symbol_codes,
            idx_reg,
            n_splits=n_splits,
            max_subset=max_subset,
        )
        non_metrics = _lgbm_subset_cv_metrics(
            X,
            y,
            timestamps,
            symbol_codes,
            idx_non,
            n_splits=n_splits,
            max_subset=max_subset,
        )
        full_metrics = _lgbm_subset_cv_metrics(
            X,
            y,
            timestamps,
            symbol_codes,
            idx_full,
            n_splits=n_splits,
            max_subset=max_subset,
        )
        geom_str = f"tp={tp_atr:.2f}|sl={sl_atr:.2f}"
        reg_auc_probe = _metric_or_nan(reg_metrics["auc_mean"])
        non_auc_probe = _metric_or_nan(non_metrics["auc_mean"])
        full_auc_probe = _metric_or_nan(full_metrics["auc_mean"])
        if np.isfinite(reg_auc_probe):
            if reg_auc_probe < 0.51:
                tprint(
                    f"  Geometry {geom_str} has low learnability "
                    f"(AUC={reg_metrics.get('auc_mean'):.3f}). Feature drivers may be unreliable."
                )
        else:
            tprint(
                "  Phase 4 regime LGBM invalid "
                f"geom={geom_str} subset={reg_metrics.get('subset_count')} "
                f"labels={reg_metrics.get('valid_label_count')} "
                f"pos/neg={reg_metrics.get('positive_count')}/{reg_metrics.get('negative_count')} "
                f"folds={reg_metrics.get('built_folds')} "
                f"class_valid={reg_metrics.get('class_valid_folds')} "
                f"auc_folds={reg_metrics.get('scored_auc_folds')} "
                f"holdout={reg_metrics.get('used_holdout_fallback')} "
                f"reason={reg_metrics.get('invalid_reason')} "
                f"fold_invalids={reg_metrics.get('fold_invalid_reason_counts')}"
            )
        if not np.isfinite(non_auc_probe):
            tprint(
                "  Phase 4 baseline(non-regime) LGBM invalid "
                f"geom={geom_str} subset={non_metrics.get('subset_count')} "
                f"labels={non_metrics.get('valid_label_count')} "
                f"pos/neg={non_metrics.get('positive_count')}/{non_metrics.get('negative_count')} "
                f"folds={non_metrics.get('built_folds')} "
                f"class_valid={non_metrics.get('class_valid_folds')} "
                f"auc_folds={non_metrics.get('scored_auc_folds')} "
                f"holdout={non_metrics.get('used_holdout_fallback')} "
                f"reason={non_metrics.get('invalid_reason')} "
                f"fold_invalids={non_metrics.get('fold_invalid_reason_counts')}"
            )
        if not np.isfinite(full_auc_probe):
            tprint(
                "  Phase 4 baseline(full) LGBM invalid "
                f"geom={geom_str} subset={full_metrics.get('subset_count')} "
                f"labels={full_metrics.get('valid_label_count')} "
                f"pos/neg={full_metrics.get('positive_count')}/{full_metrics.get('negative_count')} "
                f"folds={full_metrics.get('built_folds')} "
                f"class_valid={full_metrics.get('class_valid_folds')} "
                f"auc_folds={full_metrics.get('scored_auc_folds')} "
                f"holdout={full_metrics.get('used_holdout_fallback')} "
                f"reason={full_metrics.get('invalid_reason')} "
                f"fold_invalids={full_metrics.get('fold_invalid_reason_counts')}"
            )
        if np.isfinite(reg_auc_probe) and (
            np.isfinite(non_auc_probe) or np.isfinite(full_auc_probe)
        ):
            break

    out["tbm_lgbm_selected_geometry"] = f"tp={tp_atr:.2f}|sl={sl_atr:.2f}"
    if reg_metrics is not None:
        out["tbm_lgbm_invalid_reason_regime"] = str(
            reg_metrics.get("invalid_reason", "")
        )
    if non_metrics is not None:
        out["tbm_lgbm_invalid_reason_baseline"] = str(
            non_metrics.get("invalid_reason", "")
        )
    if full_metrics is not None:
        out["tbm_lgbm_invalid_reason_full"] = str(
            full_metrics.get("invalid_reason", "")
        )
    if reg_metrics is None or non_metrics is None or full_metrics is None:
        return out

    reg_auc_mean = _metric_or_nan(reg_metrics["auc_mean"])
    non_auc_mean = _metric_or_nan(non_metrics["auc_mean"])
    full_auc_mean = _metric_or_nan(full_metrics["auc_mean"])
    reg_lift_mean = _metric_or_nan(reg_metrics["lift_mean"])
    non_lift_mean = _metric_or_nan(non_metrics["lift_mean"])
    full_lift_mean = _metric_or_nan(full_metrics["lift_mean"])
    if not np.isfinite(reg_auc_mean):
        return out

    non_auc_cmp = non_auc_mean if np.isfinite(non_auc_mean) else float("-inf")
    full_auc_cmp = full_auc_mean if np.isfinite(full_auc_mean) else float("-inf")
    if non_auc_cmp >= full_auc_cmp:
        base_auc_mean = non_auc_mean
        base_auc_folds = np.asarray(non_metrics["auc_folds"], dtype=np.float32)
        base_lift_mean = non_lift_mean
        base_lift_folds = np.asarray(non_metrics["lift_folds"], dtype=np.float32)
    else:
        base_auc_mean = full_auc_mean
        base_auc_folds = np.asarray(full_metrics["auc_folds"], dtype=np.float32)
        base_lift_mean = full_lift_mean
        base_lift_folds = np.asarray(full_metrics["lift_folds"], dtype=np.float32)
    if not np.isfinite(base_auc_mean):
        return out

    reg_auc_folds = np.asarray(reg_metrics["auc_folds"], dtype=np.float32)
    reg_lift_folds = np.asarray(reg_metrics["lift_folds"], dtype=np.float32)
    k_auc = min(reg_auc_folds.size, base_auc_folds.size)
    if k_auc > 0:
        auc_arr = reg_auc_folds[:k_auc] - base_auc_folds[:k_auc]
    else:
        auc_arr = np.asarray([reg_auc_mean - base_auc_mean], dtype=np.float32)
    k_lift = min(reg_lift_folds.size, base_lift_folds.size)
    if k_lift > 0:
        lift_arr = reg_lift_folds[:k_lift] - base_lift_folds[:k_lift]
    elif np.isfinite(reg_lift_mean) and np.isfinite(base_lift_mean):
        lift_arr = np.asarray([reg_lift_mean - base_lift_mean], dtype=np.float32)
    else:
        lift_arr = np.asarray([], dtype=np.float32)

    stability = _stability_from_fold_deltas(auc_arr)
    out["tbm_lgbm_auc_regime"] = np.float32(reg_auc_mean)
    out["tbm_lgbm_auc_baseline"] = np.float32(base_auc_mean)
    out["tbm_lgbm_auc_lift_vs_baseline"] = np.float32(
        float(np.mean(auc_arr)) if auc_arr.size > 0 else 0.0
    )
    out["tbm_lgbm_top_bucket_lift_vs_baseline"] = np.float32(
        float(np.mean(lift_arr)) if lift_arr.size > 0 else 0.0
    )
    out["tbm_lgbm_positive_fold_fraction"] = np.float32(
        stability["positive_fold_fraction"]
        if np.isfinite(stability["positive_fold_fraction"])
        else 0.0
    )
    out["tbm_lgbm_stability"] = np.float32(
        stability["stability_score"]
        if np.isfinite(stability["stability_score"])
        else 0.0
    )
    return out


def _compute_mfe_coverage(
    shared: Dict[str, Any],
    side_mask: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    tp_atr = float(cfg.get("mfe_coverage_tp_atr", 1.25))
    mfe_atr = np.asarray(shared["mfe_atr"], dtype=np.float32)
    # fixed-threshold coverage uses only rows with full forward horizon (finite mfe_atr).
    valid = side_mask.astype(bool) & np.isfinite(mfe_atr)
    coverage = (
        float(np.mean(mfe_atr[valid] >= np.float32(tp_atr))) if np.any(valid) else 0.0
    )
    return {"fixed_tp_mfe_coverage": np.float32(coverage)}


def _compute_full_metrics_for_candidate(
    mode: str,
    side_mask: np.ndarray,
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    impulse_shape_dispersion: float,
    basic_directionality_edge: float,
    design_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    bundle = design_bundle
    if bundle is None:
        bundle = _prepare_candidate_design_bundle(
            mode=mode,
            side_mask=side_mask,
            shared=shared,
            cfg=cfg,
        )
    learn_X = bundle["learn_X"]
    timestamps = bundle["timestamps"]
    symbol_codes = bundle["symbol_codes"]
    y_primary = bundle["y_primary"]
    valid = bundle["valid_forward"]
    idx_ne = bundle["idx_ne"]
    idx_e = bundle["idx_e"]

    metrics: Dict[str, float] = {
        "primary_predictability_gain": float("nan"),
        "continuation_predictability_gain": float("nan"),
        "reversal_predictability_gain": float("nan"),
        "bucket_primary_delta_fold_mean": float("nan"),
        "bucket_primary_delta_fold_std": float("nan"),
        "bucket_primary_delta_fold_count": 0.0,
        "bucket_primary_delta_fold_min": float("nan"),
        "MAE_predictability_gain": float("nan"),
        "MFE_predictability_gain": float("nan"),
        "reversal_utility_gain": float("nan"),
        "mae_event_oos_r2": float("nan"),
        "mfe_event_oos_r2": float("nan"),
        "magnitude_delta_r": float("nan"),
        "magnitude_positive_fold_fraction": float("nan"),
        "magnitude_stability_score": float("nan"),
        "magnitude_fold_count": float("nan"),
        "magnitude_delta_fold_mean": float("nan"),
        "magnitude_delta_fold_std": float("nan"),
        "selected_delta_metric": "",
        "incremental_information_delta_auc": float("nan"),
        "incremental_information_delta_auc_fold_mean": float("nan"),
        "incremental_information_delta_auc_fold_std": float("nan"),
        "incremental_information_positive_fold_fraction": float("nan"),
        "incremental_information_positive_fold_count": float("nan"),
        "incremental_information_fold_count": float("nan"),
        "dispersion_to_edge_ratio": float("nan"),
        "edge_to_dispersion_ratio": float("nan"),
        "return_uplift": float(basic_directionality_edge),
        "primary_predictability_gain_is_nan": 1.0,
    }

    if idx_e.shape[0] < 50 or idx_ne.shape[0] < 50:
        if np.isfinite(impulse_shape_dispersion) and np.isfinite(
            basic_directionality_edge
        ):
            metrics["dispersion_to_edge_ratio"] = float(
                impulse_shape_dispersion / max(abs(basic_directionality_edge), 1e-6)
            )
            metrics["edge_to_dispersion_ratio"] = float(
                abs(basic_directionality_edge) / max(impulse_shape_dispersion, 1e-6)
            )
        metrics["return_uplift"] = float(basic_directionality_edge)
        return metrics

    classifier_n_splits = int(cfg.get("phase2_classifier_n_splits", 4))
    metric_fold_splits = int(cfg.get("phase2_metric_fold_splits", 4))
    incremental_info_n_splits = int(cfg.get("incremental_information_n_splits", 4))
    idx_e = bundle["idx_e_bal"]
    idx_ne = bundle["idx_ne_bal"]

    # primary classifier
    auc_ne = _classifier_oof_auc(
        learn_X[idx_ne],
        y_primary[idx_ne],
        timestamps[idx_ne],
        symbols=symbol_codes[idx_ne],
        n_splits=classifier_n_splits,
    )
    auc_e = _classifier_oof_auc(
        learn_X[idx_e],
        y_primary[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        n_splits=classifier_n_splits,
    )
    primary_gain = float(auc_e - auc_ne)
    metrics["primary_predictability_gain"] = primary_gain
    metrics["primary_predictability_gain_is_nan"] = 0.0
    primary_delta_folds = _primary_gain_fold_deltas(
        learn_X=learn_X,
        y_primary=y_primary,
        timestamps=timestamps,
        symbols=symbol_codes,
        idx_e=idx_e,
        idx_ne=idx_ne,
        n_splits=classifier_n_splits,
    )
    if primary_delta_folds.size > 0:
        metrics["bucket_primary_delta_fold_mean"] = float(np.mean(primary_delta_folds))
        metrics["bucket_primary_delta_fold_std"] = float(np.std(primary_delta_folds))
        metrics["bucket_primary_delta_fold_count"] = float(primary_delta_folds.size)
        metrics["bucket_primary_delta_fold_min"] = float(np.min(primary_delta_folds))

    # classify it into continuation/reversal labels for reporting
    if _mode_is_tf(mode):
        metrics["continuation_predictability_gain"] = primary_gain
    else:
        metrics["reversal_predictability_gain"] = primary_gain

    metrics.update(
        _incremental_information_metrics(
            learn_X=learn_X,
            side_mask=side_mask,
            y_primary=y_primary,
            timestamps=timestamps,
            symbols=symbol_codes,
            idx_e=idx_e,
            idx_ne=idx_ne,
            n_splits=incremental_info_n_splits,
        )
    )
    metrics["dispersion_to_edge_ratio"] = float(
        impulse_shape_dispersion / max(abs(basic_directionality_edge), 1e-6)
    )
    metrics["edge_to_dispersion_ratio"] = float(
        abs(basic_directionality_edge) / max(impulse_shape_dispersion, 1e-6)
    )

    # regression targets
    if mode == MODE_LONG:
        mae_arr = shared["mae_high"]
        mfe_arr = shared["mfe_high"]
        reversal_utility = bundle["reversal_utility"]
    else:
        mae_arr = shared["mae_low"]
        mfe_arr = shared["mfe_low"]
        reversal_utility = bundle["reversal_utility"]

    mae_ne = _ridge_regression_oof_r2(
        learn_X[idx_ne],
        mae_arr[idx_ne],
        timestamps[idx_ne],
        symbols=symbol_codes[idx_ne],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    mae_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        mae_arr[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    metrics["MAE_predictability_gain"] = float(mae_e - mae_ne)
    metrics["mae_event_oos_r2"] = float(mae_e)

    mfe_ne = _ridge_regression_oof_r2(
        learn_X[idx_ne],
        mfe_arr[idx_ne],
        timestamps[idx_ne],
        symbols=symbol_codes[idx_ne],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    mfe_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        mfe_arr[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    metrics["MFE_predictability_gain"] = float(mfe_e - mfe_ne)
    metrics["mfe_event_oos_r2"] = float(mfe_e)

    rev_ne = _ridge_regression_oof_r2(
        learn_X[idx_ne],
        reversal_utility[idx_ne],
        timestamps[idx_ne],
        symbols=symbol_codes[idx_ne],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    rev_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        reversal_utility[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    metrics["reversal_utility_gain"] = float(rev_e - rev_ne)

    mae_folds = _ridge_regression_fold_r2s(
        learn_X[idx_e],
        mae_arr[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        clip_q=0.98,
        n_splits=metric_fold_splits,
    )
    mfe_folds = _ridge_regression_fold_r2s(
        learn_X[idx_e],
        mfe_arr[idx_e],
        timestamps[idx_e],
        symbols=symbol_codes[idx_e],
        clip_q=0.98,
        n_splits=metric_fold_splits,
    )

    if np.isfinite(metrics["mfe_event_oos_r2"]) and (
        not np.isfinite(metrics["mae_event_oos_r2"])
        or metrics["mfe_event_oos_r2"] >= metrics["mae_event_oos_r2"]
    ):
        selected_folds = mfe_folds
        metrics["magnitude_delta_r"] = float(metrics["mfe_event_oos_r2"])
        metrics["selected_delta_metric"] = "mfe_event_oos_r2"
    else:
        selected_folds = mae_folds
        metrics["magnitude_delta_r"] = float(metrics["mae_event_oos_r2"])
        metrics["selected_delta_metric"] = "mae_event_oos_r2"

    stability = _stability_from_fold_deltas(selected_folds)
    metrics["magnitude_positive_fold_fraction"] = float(
        stability["positive_fold_fraction"]
    )
    metrics["magnitude_stability_score"] = float(stability["stability_score"])
    metrics["magnitude_fold_count"] = float(stability["fold_count"])
    metrics["magnitude_delta_fold_mean"] = float(stability["delta_fold_mean"])
    metrics["magnitude_delta_fold_std"] = float(stability["delta_fold_std"])

    return metrics


def _final_topk_diagnostics(
    mode: str,
    contenders: pd.DataFrame,
    candidate_masks: Dict[str, Dict[str, np.ndarray]],
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    forward_returns = shared["forward_returns"]
    timestamps = shared["timestamps"]
    symbol_codes = shared["symbol_codes"]
    symbol_uniques = shared["symbol_uniques"]
    regime_ids = shared["regime_ids"]

    ret_threshold = float(cfg.get("phase1_ret_threshold", 0.0))
    min_asset_events = int(cfg.get("diag_min_asset_events", 30))
    min_regime_events = int(cfg.get("diag_min_regime_events", 50))

    for _, row in contenders.iterrows():
        name = row["name"]
        masks = candidate_masks[name]
        side_mask = _get_side_mask(mode, masks["m_high"], masks["m_low"])

        # A. cross-asset generalization
        asset_scores: List[float] = []
        y = _mode_primary_target(mode, forward_returns, ret_threshold)
        valid_fwd = np.isfinite(forward_returns)
        for aid in range(symbol_uniques.shape[0]):
            idx = np.where((symbol_codes == aid) & valid_fwd)[0]
            sub_mask = side_mask[idx]
            if np.sum(sub_mask) < min_asset_events:
                continue
            score = float(np.mean(y[idx][sub_mask]) - np.mean(y[idx]))
            asset_scores.append(score)

        if asset_scores:
            asset_scores_arr = np.asarray(asset_scores, dtype=np.float32)
            median_asset_pred = float(np.median(asset_scores_arr))
            mean_asset_pred = float(np.mean(asset_scores_arr))
            p25_asset_pred = float(np.quantile(asset_scores_arr, 0.25))
            p75_asset_pred = float(np.quantile(asset_scores_arr, 0.75))
            share_assets_pos = float(np.mean(asset_scores_arr > 0))
            n_assets_eval = int(asset_scores_arr.shape[0])
        else:
            median_asset_pred = mean_asset_pred = p25_asset_pred = p75_asset_pred = 0.0
            share_assets_pos = 0.0
            n_assets_eval = 0

        # B. regime stability
        regime_preds = {}
        y_signed = _signed_mode_return(mode, forward_returns)
        valid_fwd = np.isfinite(forward_returns)
        for rid, lbl in [(0, "low"), (1, "normal"), (2, "high")]:
            m = side_mask & (regime_ids == rid) & valid_fwd
            if np.sum(m) < min_regime_events:
                regime_preds[lbl] = np.nan
            else:
                regime_preds[lbl] = float(np.mean(y_signed[m]))
        regime_vals = np.array(
            [regime_preds["low"], regime_preds["normal"], regime_preds["high"]],
            dtype=np.float32,
        )
        valid_reg = regime_vals[np.isfinite(regime_vals)]
        if valid_reg.shape[0] > 0:
            regime_std = float(np.std(valid_reg))
            regime_min = float(np.min(valid_reg))
            regime_max = float(np.max(valid_reg))
        else:
            regime_std = regime_min = regime_max = 0.0

        # C. feature predictability ceiling
        simple_score = _simple_score_for_mode(mode, feature_dict, side_mask)
        valid_idx = np.where(np.isfinite(simple_score) & np.isfinite(forward_returns))[
            0
        ]
        if valid_idx.shape[0] >= 20:
            s = simple_score[valid_idx]
            y_s = y_signed[valid_idx]
            q80 = np.nanquantile(s, 0.80)
            q20 = np.nanquantile(s, 0.20)
            top_mask = s >= q80
            bot_mask = s <= q20
            top_ret = float(np.nanmean(y_s[top_mask])) if np.any(top_mask) else 0.0
            bot_ret = float(np.nanmean(y_s[bot_mask])) if np.any(bot_mask) else 0.0
            spread = top_ret - bot_ret
        else:
            top_ret = 0.0
            bot_ret = 0.0
            spread = 0.0

        rows.append(
            {
                "mode": mode,
                "contender_name": name,
                "final_shortlist_score": float(row.get("shortlist_score", 0.0)),
                "n_assets_evaluated": n_assets_eval,
                "median_asset_predictability": median_asset_pred,
                "mean_asset_predictability": mean_asset_pred,
                "p25_asset_predictability": p25_asset_pred,
                "p75_asset_predictability": p75_asset_pred,
                "share_assets_positive_predictability": share_assets_pos,
                "predictability_low_vol": regime_preds["low"],
                "predictability_normal_vol": regime_preds["normal"],
                "predictability_high_vol": regime_preds["high"],
                "regime_predictability_std": regime_std,
                "min_regime_predictability": regime_min,
                "max_regime_predictability": regime_max,
                "simple_score_top20_mean_return": top_ret,
                "simple_score_bottom20_mean_return": bot_ret,
                "simple_score_spread": spread,
            }
        )

    return pd.DataFrame(rows)


def _run_mode_search(
    mode: str,
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    start_stage = int(cfg.get("mask_opt_start_stage", 1))
    stop_stage = int(cfg.get("mask_opt_stop_stage", 6))
    tprint("=" * 80)
    tprint(f"LAYER 0 MODE SEARCH: {mode}")
    tprint("=" * 80)

    bph = shared["bph"]
    timestamps = shared["timestamps"]
    day_ids = shared["day_ids"]
    n_days = shared["n_days"]
    folds = shared["folds"]
    forward_returns = shared["forward_returns"]
    global_signed_returns = _signed_mode_return(mode, forward_returns)

    ret_threshold = float(cfg.get("phase1_ret_threshold", 0.0))
    phase1_mask = _phase1_subsample_indices(shared, cfg, seed=42)
    phase1_shared = _build_phase_local_shared(shared, phase1_mask)
    _tprint_retention_step(
        f"Location Filter sample ({mode})",
        int(phase1_mask.sum()),
        int(cfg.get("mask_opt_full_panel_rows", shared["close"].shape[0])),
        prev_rows=int(shared["close"].shape[0]),
    )
    candidate_grid = shared["candidate_grid"]
    candidate_registry: Dict[str, Dict[str, Any]] = {}

    # cache by geometry key
    geom_cache_phase1: Dict[str, Dict[str, Any]] = {}
    geom_cache_phase2: Dict[str, Dict[str, Any]] = {}
    phase1_z_cache: Dict[int, Dict[str, np.ndarray]] = {}
    global_z_cache: Dict[int, Dict[str, np.ndarray]] = {}
    candidate_masks: Dict[str, Dict[str, np.ndarray]] = {}
    candidate_stage_metrics_cache: Dict[str, Dict[str, Any]] = {}

    phase1_rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Location Filter: broad sample + cheap metrics + primary classifier only
    # -------------------------------------------------------------------------
    tprint(
        f"Location Filter ({mode}): evaluating {len(candidate_grid)} candidates on broad sample..."
    )
    _tprint_mask_support_summary(
        stage="Location Filter sample",
        mode=mode,
        mask=np.ones(phase1_shared["close"].shape[0], dtype=bool),
        shared_like={
            **phase1_shared,
            "symbol_uniques": shared["symbol_uniques"],
        },
        bph=bph,
        note="subsampled rows for cheap screening",
    )

    phase1_accepted = 0
    phase1_rejected = 0

    global_target = _mode_primary_target(mode, forward_returns, ret_threshold)
    phase1_global_idx = np.where(phase1_mask)[0]
    phase1_global_target = global_target[phase1_mask]
    phase1_signed_returns = _signed_mode_return(mode, phase1_shared["forward_returns"])
    phase1_valid_fwd = np.isfinite(phase1_shared["forward_returns"])
    phase1_n_assets = max(1, len(phase1_shared["asset_groups"]))

    phase1_ratio = (
        float(np.sum(phase1_mask)) / float(phase1_mask.shape[0])
        if phase1_mask.shape[0] > 0
        else 1.0
    )
    phase1_min_total_events = int(cfg.get("phase1_min_total_events", 50))
    phase1_rejection_reasons: Dict[str, int] = {}

    global_to_phase1_local = np.full(forward_returns.shape[0], -1, dtype=np.int32)
    global_to_phase1_local[phase1_global_idx] = np.arange(
        phase1_global_idx.shape[0], dtype=np.int32
    )
    phase1_fold_val_locals: List[np.ndarray] = []
    for _, va in folds:
        loc = global_to_phase1_local[va]
        loc = loc[loc >= 0].astype(np.int32)
        phase1_fold_val_locals.append(loc)

    for candidate in candidate_grid:
        allowed_modes = tuple(candidate.get("allowed_modes", ("long", "short")))
        if mode not in allowed_modes:
            continue
        fam = candidate["family"]
        z_hr = candidate["lookback_h"]
        d_hr = 1
        key = CandidateKey(
            fam,
            int(z_hr),
            candidate["feature_base"] + "_" + candidate["direction"] + "_" + str(candidate["threshold"]),
            int(d_hr),
        ).as_str()
        reg_entry = candidate.copy()
        reg_entry["z_hours"] = z_hr
        reg_entry["duration_hours"] = d_hr
        candidate_registry[key] = reg_entry

    if start_stage > 1:
        tprint(f"Phase 2 ({mode}): loading input state from stage 1 artifacts...")
        loaded_stage1 = _load_stage_artifacts(cfg, mode, 1)
        if not loaded_stage1 or "df1" not in loaded_stage1:
            raise FileNotFoundError(
                f"Missing stage 1 artifacts for mode={mode} in {_mode_stage_dir(cfg, mode)}"
            )
        df1 = loaded_stage1["df1"].copy()
    else:
        for candidate in candidate_grid:
            allowed_modes = tuple(candidate.get("allowed_modes", ("long", "short")))
            if mode not in allowed_modes:
                continue
            fam = candidate["family"]
            z_hr = candidate["lookback_h"]
            d_hr = 1  # Default or extract if present
            
            z = int(z_hr * bph)
            duration_bars = int(d_hr * bph)
            
            key = CandidateKey(
                fam, int(z_hr), candidate["feature_base"] + "_" + candidate["direction"] + "_" + str(candidate["threshold"]), int(d_hr)
            ).as_str()
            
            if key not in geom_cache_phase1:
                if candidate.get("kind") == "location_filter":
                    zc_local = {
                        k: np.asarray(v)[phase1_mask]
                        for k, v in shared["location_filter_frame"].items()
                    }
                else:
                    if z not in phase1_z_cache:
                        tprint(f"Precomputing Location Filter tensors for z={z} bars...")
                        phase1_z_cache[z] = _compute_z_cache(
                            high=shared["high"],
                            low=shared["low"],
                            close=shared["close"],
                            ret_1=shared["ret_1"],
                            vol_g=shared["vol_g"],
                            asset_groups=shared["asset_groups"],
                            z=z,
                            bph=bph,
                            volume=shared.get("volume"),
                        )
                    zc_local = _slice_z_cache(phase1_z_cache[z], phase1_mask)
                m_high_local, m_low_local = _generate_event_masks_fast(
                    candidate=candidate,
                    zc=zc_local,
                )
                side_mask = _get_side_mask(mode, m_high_local, m_low_local)

                total_events = simple_mask_count_nb(side_mask)
                if total_events < phase1_min_total_events:
                    if phase1_rejected < 5:
                        tprint(f"DEBUG Location Filter ({mode}): Candidate {key} rejected. events={total_events}/{phase1_min_total_events}")
                    geom_cache_phase1[key] = {"rejected": True}
                    phase1_rejected += 1
                    _record_rejection_reason(phase1_rejection_reasons, "too_few_events")
                    continue

                active_days_frac = active_days_fraction_nb(
                    side_mask, phase1_shared["day_ids"], phase1_shared["n_days"]
                )
                if active_days_frac < float(
                    cfg.get("phase1_min_active_days_fraction", 0.80)
                ):
                    if phase1_rejected < 10:
                        tprint(f"DEBUG Location Filter ({mode}): Candidate {key} rejected. active_days_frac={active_days_frac:.3f}/{cfg.get('phase1_min_active_days_fraction', 0.80):.3f} (events={total_events})")
                    geom_cache_phase1[key] = {"rejected": True}
                    phase1_rejected += 1
                    _record_rejection_reason(phase1_rejection_reasons, "active_days_too_low")
                    continue
                support_stats = _cheap_support_stats(
                    mask=side_mask,
                    day_ids=phase1_shared["day_ids"],
                    n_days=phase1_shared["n_days"],
                    symbol_codes=phase1_shared["symbol_codes"],
                    symbol_uniques=shared["symbol_uniques"],
                    timestamps=phase1_shared["timestamps"],
                    asset_groups=phase1_shared["asset_groups"],
                    bph=bph,
                    folds=[(np.array([], dtype=np.int32), va) for va in phase1_fold_val_locals],
                )
                passed_support_gate, reject_reason = _passes_cheap_support_gate(
                    stats=support_stats,
                    cfg=cfg,
                    phase_prefix="phase1",
                )
                if not passed_support_gate:
                    geom_cache_phase1[key] = {"rejected": True}
                    phase1_rejected += 1
                    _record_rejection_reason(phase1_rejection_reasons, reject_reason)
                    continue

                if candidate.get("kind") == "location_filter":
                    coh = {
                        "bars_to_peak_dispersion": 0.0,
                        "speed_dispersion": 0.0,
                        "monotonicity_dispersion": 0.0,
                        "impulse_shape_dispersion": 0.0,
                    }
                else:
                    if _mode_is_up(mode):
                        coh = _coherence_metrics_single_side(
                            side_mask, zc_local["b_up"], zc_local["s_up"], zc_local["m_up"]
                        )
                    else:
                        coh = _coherence_metrics_single_side(
                            side_mask, zc_local["b_dn"], zc_local["s_dn"], zc_local["m_dn"]
                        )

                distinct = _compute_regime_distinctness_single_side(
                    side_mask=side_mask,
                    mode=mode,
                    forward_returns=phase1_shared["forward_returns"],
                    mae_high=phase1_shared["mae_high"],
                    mfe_high=phase1_shared["mfe_high"],
                    mae_low=phase1_shared["mae_low"],
                    mfe_low=phase1_shared["mfe_low"],
                )

                ev_day_mean, ev_day_std = daily_event_stats_nb(
                    side_mask, phase1_shared["day_ids"], phase1_shared["n_days"]
                )
                ev_day_per_asset = float(total_events) / float(
                    max(1, phase1_shared["n_days"] * phase1_n_assets)
                )
                symbol_summary = _mask_symbol_concentration_summary(
                    side_mask,
                    phase1_shared["symbol_codes"],
                    shared["symbol_uniques"],
                )
                duration_stats = _mask_run_duration_stats(
                    side_mask, phase1_shared["asset_groups"], bph
                )
                if symbol_summary["event_symbol_count"] <= 3:
                    msg = (
                        f"SYMBOL_CONCENTRATION Phase1 mode={mode} candidate={key} "
                        f"symbols={symbol_summary['event_symbol_count']} "
                        f"top_share={symbol_summary['top_symbol_share']:.3f} "
                        f"top={symbol_summary['top_symbol_counts_text']}"
                    )
                    tprint(msg)
                    _append_symbol_concentration_log(msg)

                fold_rates = []
                fold_event_counts = []
                for val_idx_local in phase1_fold_val_locals:
                    if val_idx_local.shape[0] == 0:
                        continue
                    fold_event_counts.append(float(np.sum(side_mask[val_idx_local])))
                    fold_rates.append(
                        fold_base_rate_nb(side_mask, phase1_global_target, val_idx_local)
                    )
                fold_rate_std = (
                    float(np.std(np.asarray(fold_rates, dtype=np.float32)))
                    if fold_rates
                    else 1.0
                )
                fold_event_count_std = (
                    float(np.std(np.asarray(fold_event_counts, dtype=np.float32)))
                    if fold_event_counts
                    else 1.0
                )

                non_event = (~side_mask) & phase1_valid_fwd
                if np.any(side_mask & phase1_valid_fwd) and np.any(non_event):
                    basic_edge = float(
                        np.nanmean(phase1_signed_returns[side_mask & phase1_valid_fwd])
                        - np.nanmean(phase1_signed_returns[non_event])
                    )
                else:
                    basic_edge = 0.0
                dispersion_ratio = _safe_abs_ratio(
                    float(coh["impulse_shape_dispersion"]), basic_edge
                )

                stats = {
                    "rejected": False,
                    "side_mask": side_mask.copy(),
                    "total_events": int(total_events),
                    "active_days_fraction": float(active_days_frac),
                    "events_per_day_mean": float(ev_day_mean),
                    "events_per_day_std": float(ev_day_std),
                    "events_per_day_per_asset": float(ev_day_per_asset),
                    "event_symbol_count": int(symbol_summary["event_symbol_count"]),
                    "top_symbol_share": float(symbol_summary["top_symbol_share"]),
                    "top_symbol_counts_text": str(symbol_summary["top_symbol_counts_text"]),
                    "avg_event_duration_bars": float(duration_stats["avg_event_duration_bars"]),
                    "median_event_duration_bars": float(duration_stats["median_event_duration_bars"]),
                    "avg_event_duration_hours": float(duration_stats["avg_event_duration_hours"]),
                    "median_event_duration_hours": float(duration_stats["median_event_duration_hours"]),
                    "event_run_count": float(duration_stats["event_run_count"]),
                    "bars_to_peak_dispersion": float(coh["bars_to_peak_dispersion"]),
                    "speed_dispersion": float(coh["speed_dispersion"]),
                    "monotonicity_dispersion": float(coh["monotonicity_dispersion"]),
                    "impulse_shape_dispersion": float(coh["impulse_shape_dispersion"]),
                    "regime_distinctness_score": float(distinct),
                    "fold_base_rate_stability": float(fold_rate_std),
                    "fold_continuation_rate_std": float(fold_rate_std),
                    "fold_event_count_std": float(fold_event_count_std),
                    "basic_directionality_edge_event_vs_non_event": float(basic_edge),
                    "dispersion_to_edge_ratio": float(dispersion_ratio),
                }
                geom_cache_phase1[key] = stats
                phase1_accepted += 1

            stats = geom_cache_phase1[key]
            if stats.get("rejected", False):
                continue

            phase1_rows.append(
                {
                    "name": key,
                    "family": fam,
                    "z_hours": z_hr,
                    "duration_hours": d_hr,
                    **{k: v for k, v in stats.items() if k not in {"rejected", "side_mask"}},
                }
            )

        if not phase1_rows:
            return {"status": "failed", "reason": f"no_phase1_candidates_{mode}"}

        df1 = pd.DataFrame(phase1_rows)
        phase1_sample_rows = max(int(np.sum(phase1_mask)), 1)
        df1["keep_fraction_vs_phase1_sample"] = (
            df1["total_events"].astype(np.float32) / np.float32(phase1_sample_rows)
        ).astype(np.float32)
        df1["phase1_information_gain"] = (
            np.maximum(
                df1["basic_directionality_edge_event_vs_non_event"].astype(np.float32).values,
                0.0,
            )
            + 0.75 * np.maximum(df1["regime_distinctness_score"].astype(np.float32).values, 0.0)
        ).astype(np.float32)
        df1["phase1_information_efficiency"] = (
            df1["phase1_information_gain"].astype(np.float32).values
            * np.sqrt(
                np.clip(
                    df1["keep_fraction_vs_phase1_sample"].astype(np.float32).values,
                    1e-6,
                    1.0,
                )
            )
        ).astype(np.float32)
        disp_edge_z = _zscore_np(df1["dispersion_to_edge_ratio"].values.astype(np.float32))
        disp_edge_z[np.isnan(disp_edge_z)] = 3.0

        df1["phase1_proxy_score"] = (
            0.20 * _zscore_np(df1["active_days_fraction"].values)
            + 0.15 * _zscore_np(df1["regime_distinctness_score"].values)
            + 0.15 * _zscore_np(np.log1p(df1["total_events"].values.astype(np.float32)))
            + 0.30 * _zscore_np(df1["basic_directionality_edge_event_vs_non_event"].values)
            + 0.20 * _zscore_np(df1["phase1_information_efficiency"].values.astype(np.float32))
            - 0.10
            * _zscore_np(
                np.nan_to_num(
                    df1["dispersion_to_edge_ratio"].values.astype(np.float32), nan=1e6
                )
            )
            - 0.30 * disp_edge_z
            - 0.15 * _zscore_np(df1["fold_continuation_rate_std"].values)
            - 0.10 * _zscore_np(df1["fold_event_count_std"].values)
        )
        tprint(f"{_stage_label(1)} ({mode}): {phase1_accepted} candidates accepted, {phase1_rejected} rejected")
        _tprint_rejection_summary(_stage_label(1), mode, phase1_rejection_reasons)
        _tprint_candidate_table_support_summary(f"{_stage_label(1)} accepted", mode, df1)

        _log_stage_snapshot(
            mode,
            _stage_label(1),
            df1,
            "phase1_proxy_score",
            [
                "name",
                "phase1_proxy_score",
                "basic_directionality_edge_event_vs_non_event",
                "dispersion_to_edge_ratio",
                "total_events",
                "active_days_fraction",
            ],
        )
        _persist_partial_table(df1, f"layer0_phase1_candidate_table_partial_{mode}.csv")

        tprint(f"{_stage_label(1)} diversity filter ({mode}): applying feature-based filtering...")
        df1 = df1.sort_values("phase1_proxy_score", ascending=False)

        df1["feature_base"] = df1["name"].apply(lambda x: candidate_registry[x].get("feature_base", x))
        df1["family"] = df1["name"].apply(lambda x: candidate_registry[x].get("family", "unknown"))
        _tprint_family_feature_breakdown(f"{_stage_label(1)} accepted breakdown", mode, df1)

        finalist_pool_size = int(cfg.get("top_k_for_learnability", 36))
        family_seed = df1.groupby("family", sort=False).head(
            int(cfg.get("phase1_min_representatives_per_family", 1))
        ).copy()
        _tprint_family_feature_breakdown(f"{_stage_label(1)} family seed", mode, family_seed)
        top_k_global = df1.head(finalist_pool_size).copy()
        _tprint_family_feature_breakdown(f"{_stage_label(1)} top-k global", mode, top_k_global)
        combined_pool = (
            pd.concat([family_seed, top_k_global], ignore_index=True)
            .drop_duplicates(subset=["name"])
            .sort_values("phase1_proxy_score", ascending=False)
            .copy()
        )
        _tprint_family_feature_breakdown(f"{_stage_label(1)} combined pool", mode, combined_pool)
        protected_names = set(family_seed["name"].astype(str).tolist())

        df1 = combined_pool.drop_duplicates(subset=["name"]).sort_values(
            "phase1_proxy_score", ascending=False
        ).copy()
        compressed = df1.groupby("feature_base").head(
            int(cfg.get("phase1_prefilter_max_per_feature", 1))
        ).copy()
        if protected_names:
            missing_protected = protected_names.difference(
                set(compressed["name"].astype(str).tolist())
            )
            if missing_protected:
                compressed = pd.concat(
                    [
                        compressed,
                        df1[df1["name"].astype(str).isin(missing_protected)].copy(),
                    ],
                    ignore_index=True,
                ).drop_duplicates(subset=["name"])
        df1 = compressed.sort_values("phase1_proxy_score", ascending=False).copy()
        _tprint_family_feature_breakdown(f"{_stage_label(1)} post feature compression", mode, df1)
        df1 = _ensure_min_family_representatives(
            df1,
            score_col="phase1_proxy_score",
            min_per_family=int(cfg.get("phase1_min_representatives_per_family", 1)),
        )
        _tprint_family_feature_breakdown(f"{_stage_label(1)} post family floor", mode, df1)
        df1 = _ensure_min_feature_representatives(
            df1,
            score_col="phase1_proxy_score",
            min_per_feature=int(cfg.get("phase1_min_representatives_per_feature", 2)),
        )
        _tprint_family_feature_breakdown(f"{_stage_label(1)} post feature floor", mode, df1)
        phase1_overlap_masks = {
            str(name): np.asarray(geom_cache_phase1.get(str(name), {}).get("side_mask"), dtype=bool)
            for name in df1["name"].astype(str).values
            if geom_cache_phase1.get(str(name), {}).get("side_mask") is not None
        }
        df1 = _prune_candidates_by_mask_overlap(
            df1,
            score_col="phase1_proxy_score",
            candidate_masks=phase1_overlap_masks,
            overlap_threshold=float(cfg.get("phase1_overlap_prune_threshold", 0.92)),
        )
        _tprint_family_feature_breakdown(f"{_stage_label(1)} post overlap prune", mode, df1)
        df1 = _ensure_min_family_representatives(
            df1,
            score_col="phase1_proxy_score",
            min_per_family=int(cfg.get("phase1_min_representatives_per_family", 1)),
        )
        _tprint_family_feature_breakdown(f"{_stage_label(1)} final post family restore", mode, df1)
        phase1_z_cache.clear()

        tprint(f"{_stage_label(1)} diversity filter ({mode}): {len(df1)} candidates after filtering")
        _tprint_candidate_table_support_summary(f"{_stage_label(1)} filtered", mode, df1)
        _save_stage_artifacts(
            cfg,
            mode,
            1,
            payload={"df1": df1.copy()},
            tables={"candidates": df1.copy()},
        )
        if stop_stage <= 1:
            return _stage_stop_result(stage_num=1, mode=mode, candidate_table=df1)


    # -------------------------------------------------------------------------
    # Phase 2: full symbols & history + full metrics, only top phase1 candidates
    # -------------------------------------------------------------------------
    tprint(f"Phase 2 ({mode}): full symbols/history for top {len(df1)} candidates...")
    _tprint_mask_support_summary(
        stage="Phase 2 sample",
        mode=mode,
        mask=np.ones(shared["close"].shape[0], dtype=bool),
        shared_like=shared,
        bph=bph,
        note="full history/full symbol evaluation sample",
    )

    phase2_rows: List[Dict[str, Any]] = []
    phase2_n_assets = max(1, len(shared["asset_groups"]))
    phase2_accepted = 0
    phase2_rejected = 0
    phase2_rejection_reasons: Dict[str, int] = {}
    original_sample_rows = int(shared["close"].shape[0])

    for _, row in df1.iterrows():
        name = row["name"]
        reg = candidate_registry[name]
        allowed_modes = tuple(reg.get("allowed_modes", ("long", "short")))
        if mode not in allowed_modes:
            continue
        fam = reg["family"]
        z_hr = int(reg["lookback_h"])
        d_hr = 1  # Standard for Phase 2

        z = int(z_hr * bph)
        duration_bars = int(d_hr * bph)
        key = name

        if key not in geom_cache_phase2:
            if reg.get("kind") == "location_filter":
                zc = shared["location_filter_frame"]
            else:
                if z not in global_z_cache:
                    tprint(f"Precomputing Phase 2 rolling tensors for z={z} bars...")
                    global_z_cache[z] = _compute_z_cache(
                        high=shared["high"],
                        low=shared["low"],
                        close=shared["close"],
                        ret_1=shared["ret_1"],
                        vol_g=shared["vol_g"],
                        asset_groups=shared["asset_groups"],
                        z=z,
                        bph=bph,
                        volume=shared.get("volume"),
                    )
                zc = global_z_cache[z]
            m_high, m_low = _generate_event_masks_fast(
                candidate=reg,
                zc=zc,
            )
            side_mask = _get_side_mask(mode, m_high, m_low)
            total_events = int(np.sum(side_mask))
            if total_events < int(cfg.get("phase2_min_total_events", 5000)):
                geom_cache_phase2[key] = {"rejected": True}
                phase2_rejected += 1
                _record_rejection_reason(phase2_rejection_reasons, "too_few_events")
                continue

            active_days_frac = active_days_fraction_nb(side_mask, day_ids, n_days)
            if active_days_frac < float(
                cfg.get("phase2_min_active_days_fraction", 0.80)
            ):
                geom_cache_phase2[key] = {"rejected": True}
                phase2_rejected += 1
                _record_rejection_reason(phase2_rejection_reasons, "active_days_too_low")
                continue
            support_stats = _cheap_support_stats(
                mask=side_mask,
                day_ids=day_ids,
                n_days=n_days,
                symbol_codes=shared["symbol_codes"],
                symbol_uniques=shared["symbol_uniques"],
                timestamps=timestamps,
                asset_groups=shared["asset_groups"],
                bph=bph,
                folds=folds,
            )
            passed_support_gate, reject_reason = _passes_cheap_support_gate(
                stats=support_stats,
                cfg=cfg,
                phase_prefix="phase2",
            )
            if not passed_support_gate:
                geom_cache_phase2[key] = {"rejected": True}
                phase2_rejected += 1
                _record_rejection_reason(phase2_rejection_reasons, reject_reason)
                continue

            if reg.get("kind") == "location_filter":
                coh = {
                    "bars_to_peak_dispersion": 0.0,
                    "speed_dispersion": 0.0,
                    "monotonicity_dispersion": 0.0,
                    "impulse_shape_dispersion": 0.0,
                }
            elif _mode_is_up(mode):
                coh = _coherence_metrics_single_side(
                    side_mask, zc["b_up"], zc["s_up"], zc["m_up"]
                )
            else:
                coh = _coherence_metrics_single_side(
                    side_mask, zc["b_dn"], zc["s_dn"], zc["m_dn"]
                )

            distinct = _compute_regime_distinctness_single_side(
                side_mask=side_mask,
                mode=mode,
                forward_returns=forward_returns,
                mae_high=shared["mae_high"],
                mfe_high=shared["mfe_high"],
                mae_low=shared["mae_low"],
                mfe_low=shared["mfe_low"],
            )

            ev_day_mean, ev_day_std = daily_event_stats_nb(side_mask, day_ids, n_days)
            ev_day_per_asset = float(total_events) / float(
                max(1, n_days * phase2_n_assets)
            )
            symbol_summary = _mask_symbol_concentration_summary(
                side_mask,
                shared["symbol_codes"],
                shared["symbol_uniques"],
            )
            min_symbols_phase2 = int(cfg.get("phase2_min_distinct_symbols", 8))
            max_top_share_phase2 = float(cfg.get("phase2_max_top_symbol_share", 0.35))
            if symbol_summary["event_symbol_count"] < min_symbols_phase2:
                geom_cache_phase2[key] = {"rejected": True}
                phase2_rejected += 1
                _record_rejection_reason(phase2_rejection_reasons, "too_few_symbols")
                continue
            if symbol_summary["top_symbol_share"] > max_top_share_phase2:
                geom_cache_phase2[key] = {"rejected": True}
                phase2_rejected += 1
                _record_rejection_reason(phase2_rejection_reasons, "top_symbol_share_too_high")
                continue
            duration_stats = _mask_run_duration_stats(
                side_mask, shared["asset_groups"], bph
            )
            if symbol_summary["event_symbol_count"] <= 3:
                msg = (
                    f"SYMBOL_CONCENTRATION Phase2 mode={mode} candidate={key} "
                    f"symbols={symbol_summary['event_symbol_count']} "
                    f"top_share={symbol_summary['top_symbol_share']:.3f} "
                    f"top={symbol_summary['top_symbol_counts_text']}"
                )
                tprint(msg)
                _append_symbol_concentration_log(msg)
            if reg.get("kind") == "location_filter":
                post_event_vol_dispersion = 0.0
            else:
                side_vol_exp = zc["v_exp"][side_mask & np.isfinite(zc["v_exp"])]
                post_event_vol_dispersion = (
                    float(np.std(side_vol_exp)) if side_vol_exp.shape[0] > 1 else 0.0
                )

            fold_rates = [
                fold_base_rate_nb(side_mask, global_target, va) for _, va in folds
            ]
            fold_event_counts = [float(np.sum(side_mask[va])) for _, va in folds]
            fold_rate_std = (
                float(np.std(np.asarray(fold_rates, dtype=np.float32)))
                if fold_rates
                else 1.0
            )
            fold_continuation_rate_std = fold_rate_std
            fold_event_count_std = (
                float(np.std(np.asarray(fold_event_counts, dtype=np.float32)))
                if fold_event_counts
                else 1.0
            )

            valid_fwd_p2 = np.isfinite(global_signed_returns)
            non_event = (~side_mask) & valid_fwd_p2
            if np.any(side_mask & valid_fwd_p2) and np.any(non_event):
                basic_edge = float(
                    np.nanmean(global_signed_returns[side_mask & valid_fwd_p2])
                    - np.nanmean(global_signed_returns[non_event])
                )
            else:
                basic_edge = 0.0

            bundle = _prepare_candidate_design_bundle(
                mode=mode,
                side_mask=side_mask,
                shared=shared,
                cfg=cfg,
            )
            full_metrics = _compute_full_metrics_for_candidate(
                mode,
                side_mask,
                shared,
                feature_dict,
                cfg,
                float(coh["impulse_shape_dispersion"]),
                float(basic_edge),
                design_bundle=bundle,
            )
            legacy_metrics = _compute_legacy_conditional_learnability(
                mode, side_mask, shared, cfg
            )

            geom_cache_phase2[key] = {
                "rejected": False,
                "total_events": total_events,
                "active_days_fraction": float(active_days_frac),
                "events_per_day_mean": float(ev_day_mean),
                "events_per_day_std": float(ev_day_std),
                "events_per_day_per_asset": float(ev_day_per_asset),
                "event_symbol_count": int(symbol_summary["event_symbol_count"]),
                "top_symbol_share": float(symbol_summary["top_symbol_share"]),
                "top_symbol_counts_text": str(symbol_summary["top_symbol_counts_text"]),
                "avg_event_duration_bars": float(duration_stats["avg_event_duration_bars"]),
                "median_event_duration_bars": float(duration_stats["median_event_duration_bars"]),
                "avg_event_duration_hours": float(duration_stats["avg_event_duration_hours"]),
                "median_event_duration_hours": float(duration_stats["median_event_duration_hours"]),
                "event_run_count": float(duration_stats["event_run_count"]),
                "bars_to_peak_dispersion": float(coh["bars_to_peak_dispersion"]),
                "speed_dispersion": float(coh["speed_dispersion"]),
                "monotonicity_dispersion": float(coh["monotonicity_dispersion"]),
                "impulse_shape_dispersion": float(coh["impulse_shape_dispersion"]),
                "post_event_vol_dispersion": float(post_event_vol_dispersion),
                "regime_distinctness_score": float(distinct),
                "fold_base_rate_stability": float(fold_rate_std),
                "fold_continuation_rate_std": float(fold_continuation_rate_std),
                "fold_event_count_std": float(fold_event_count_std),
                "basic_directionality_edge_event_vs_non_event": float(basic_edge),
                "original_sample_count": float(original_sample_rows),
                "keep_pct_vs_original": float(100.0 * total_events / max(original_sample_rows, 1)),
                **full_metrics,
                **legacy_metrics,
            }
            phase2_accepted += 1

        stats = geom_cache_phase2[key]
        if stats.get("rejected", False):
            continue

        phase2_rows.append(
            {
                "name": key,
                "family": fam,
                "z_hours": z_hr,
                "duration_hours": d_hr,
                "conditioner_mode": "none",
                **{k: v for k, v in stats.items() if k not in {"rejected"}},
            }
        )

    if not phase2_rows:
        return {"status": "failed", "reason": f"no_phase2_candidates_{mode}"}

    tprint(f"{_stage_label(2)} ({mode}): {phase2_accepted} candidates accepted, {phase2_rejected} rejected")
    _tprint_rejection_summary(_stage_label(2), mode, phase2_rejection_reasons)

    if not phase2_rows:
        return {"status": "failed", "reason": f"no_phase2_candidates_{mode}"}

    df2 = pd.DataFrame(phase2_rows)
    df2 = _cap_stage_family_dominance(
        df2,
        score_col="total_events",
        stage=f"{_stage_label(2)} accepted",
        mode=mode,
        max_per_family=int(cfg.get("phase2_max_candidates_per_family", 3)),
    )
    df2 = _ensure_min_family_representatives(
        df2,
        score_col="total_events",
        min_per_family=int(cfg.get("phase2_min_representatives_per_family", 2)),
    )
    _persist_partial_table(df2, f"layer0_phase2_candidate_table_partial_{mode}.csv")
    _tprint_candidate_table_support_summary(f"{_stage_label(2)} accepted", mode, df2)
    df2["D_r"] = 0.25 * (
        0.35 * _zscore_np(df2["impulse_shape_dispersion"].values)
        + 0.35 * _zscore_np(df2["post_event_vol_dispersion"].values if "post_event_vol_dispersion" in df2.columns else np.zeros(len(df2)))
        + 0.15 * _zscore_np(df2["fold_continuation_rate_std"].values)
        + 0.15 * _zscore_np(df2["fold_event_count_std"].values)
    )
    df2["N_r"] = df2["total_events"].astype(np.float32)
    df2["selected_delta_metric"] = df2["selected_delta_metric"].astype(str)
    primary_col = _mode_primary_predictability_col(mode)
    df2["bucket_primary_predictability_gain"] = df2[primary_col].astype(np.float32)
    df2["predictability_gain"] = df2[
        [primary_col, "MAE_predictability_gain", "MFE_predictability_gain"]
    ].max(axis=1).astype(np.float32)
    df2["delta_r_raw"] = df2["return_uplift"].astype(np.float32)
    df2["delta_r_fallback"] = (
        0.5 * df2["incremental_information_delta_auc"].astype(np.float32)
    ).astype(np.float32)

    df2["delta_r"] = df2["delta_r_raw"].astype(np.float32)

    df2["selected_delta_metric"] = "return_uplift"
    selected_fold_mean = df2["bucket_primary_delta_fold_mean"].astype(np.float32).values
    selected_fold_std = df2["bucket_primary_delta_fold_std"].astype(np.float32).values
    df2["delta_r_fold_mean"] = selected_fold_mean.astype(np.float32)
    df2["delta_r_fold_std"] = selected_fold_std.astype(np.float32)
    df2["positive_fold_fraction_r"] = df2[
        "incremental_information_positive_fold_fraction"
    ].astype(np.float32)
    # Improved robustness for stability weighting
    fold_stability_raw = 1.0 - (
        np.nan_to_num(df2["delta_r_fold_std"].values, nan=0.5)
        / (np.abs(np.nan_to_num(df2["delta_r_fold_mean"].values, nan=0.0)) + 1e-9)
    )
    # Support-Weighted Stability Penalty: penalize small N
    n_events = df2["N_r"].astype(np.float32).values
    # We want to penalize regimes with fewer than 750 events (Phase 2 scale)
    support_mult = np.clip(np.sqrt(n_events / 750.0), 0.25, 1.0)

    # Default to 0.4 stability (slight penalty) instead of 0.5 (neutral) if fold data is missing
    fold_stability = np.where(np.isfinite(fold_stability_raw), np.clip(fold_stability_raw, 0.0, 1.0), 0.4)
    pos_fold_frac = np.nan_to_num(df2["incremental_information_positive_fold_fraction"].values, nan=0.4)
    
    df2["S_r"] = ((0.5 * fold_stability + 0.5 * pos_fold_frac) * support_mult).clip(0.1, 1.0).astype(np.float32)
    df2["delta_r_shrunk"] = (
        df2["delta_r"].astype(np.float32).values
        * (
            df2["N_r"].astype(np.float32).values
            / (df2["N_r"].astype(np.float32).values + 500.0)
        )
    ).astype(np.float32)
    uplift_raw = df2["delta_r_shrunk"].astype(np.float32).values
    if mode == "short":
        uplift_raw = -uplift_raw
    # Ensure uplift_anchor is non-zero if delta_r_raw is promising
    df2["uplift_anchor"] = np.maximum(uplift_raw, 1e-6).astype(np.float32)
    df2["primary_multiplier"] = (
        1.0
        + np.tanh(
            10.0 * df2["bucket_primary_predictability_gain"].astype(np.float32).values
        )
    ).astype(np.float32)
    df2["worst_fold_multiplier"] = (
        1.0
        + 0.5
        * np.tanh(
            10.0
            * np.nan_to_num(
                df2["bucket_primary_delta_fold_min"].astype(np.float32).values,
                nan=0.0,
            )
        )
    ).astype(np.float32)
    df2["noise_penalty"] = (
        1.0
        + 0.25 * np.log1p(
            np.maximum(
                np.nan_to_num(
                    df2["dispersion_to_edge_ratio"].astype(np.float32).values,
                    nan=100.0,
                ),
                0.0,
            )
        )
    ).astype(np.float32)
    primary_sign = np.sign(
        np.nan_to_num(
            df2["bucket_primary_predictability_gain"].astype(np.float32).values, nan=0.0
        )
    ).astype(np.float32)
    uplift_sign = np.sign(np.nan_to_num(df2["delta_r_raw"].astype(np.float32).values, nan=0.0)).astype(
        np.float32
    )
    df2["disagreement_penalty"] = np.where(
        (primary_sign != 0.0) & (uplift_sign != 0.0) & (primary_sign != uplift_sign),
        np.float32(0.65),
        np.float32(1.0),
    ).astype(np.float32)

    df2["learnability_support"] = np.maximum(
        df2["incremental_information_delta_auc"].astype(np.float32).values, 0.0
    ).astype(np.float32)
    df2["keep_fraction_vs_original"] = (
        np.clip(df2["keep_pct_vs_original"].astype(np.float32).values / 100.0, 1e-6, 1.0)
    ).astype(np.float32)
    df2["regime_information_gain"] = (
        np.maximum(df2["delta_r_shrunk"].astype(np.float32).values, 0.0)
        + 0.5 * np.maximum(df2["learnability_support"].astype(np.float32).values, 0.0)
    ).astype(np.float32)
    df2["regime_information_efficiency"] = (
        df2["regime_information_gain"].astype(np.float32).values
        * np.sqrt(df2["keep_fraction_vs_original"].astype(np.float32).values)
    ).astype(np.float32)
    df2["effective_edge"] = (
        df2["uplift_anchor"].astype(np.float32).values
        * np.maximum(df2["S_r"].astype(np.float32).values, 0.0)
        * df2["primary_multiplier"].astype(np.float32).values
        * df2["worst_fold_multiplier"].astype(np.float32).values
        * df2["disagreement_penalty"].astype(np.float32).values
    ).astype(np.float32)
    df2["score_r"] = (
        (
            df2["effective_edge"].astype(np.float32).values
            + 5.0 * df2["regime_information_efficiency"].astype(np.float32).values
        )
        * (1.0 + 25.0 * df2["learnability_support"].astype(np.float32).values)
        / np.maximum(df2["noise_penalty"].astype(np.float32).values, 1e-6)
    ).astype(np.float32)
    df2["score_ml"] = df2["score_r"].astype(np.float32)
    
    _log_stage_snapshot(
        mode,
        _stage_label(2),
        df2,
        "score_r",
        [
            "name",
            "score_r",
            "effective_edge",
            "learnability_support",
            "noise_penalty",
            "delta_r_raw",
            "incremental_information_delta_auc",
            "dispersion_to_edge_ratio",
            "disagreement_penalty",
        ],
    )

    df2_full = df2.sort_values(
        ["score_r", "delta_r", "total_events"], ascending=[False, False, False]
    ).copy()
    aligned_shrunk_edge = df2_full["delta_r_shrunk"].astype(np.float32).values.copy()
    if mode == "short":
        aligned_shrunk_edge = -aligned_shrunk_edge
    ridge_edge_floor = float(cfg.get("phase2_min_shrunk_edge_for_ridge", 5e-5))
    ridge_pos_floor = float(cfg.get("phase2_min_positive_fold_fraction_for_ridge", 0.50))
    ridge_keep_mask = (
        np.nan_to_num(aligned_shrunk_edge, nan=-np.inf) >= ridge_edge_floor
    ) & (
        np.nan_to_num(
            df2_full["positive_fold_fraction_r"].astype(np.float32).values, nan=0.0
        ) >= ridge_pos_floor
    )
    ridge_filtered = int(np.sum(~ridge_keep_mask))
    if ridge_filtered > 0:
        tprint(
            f"{_stage_label(2)} -> {_stage_label(3)} ({mode}): directional sanity gate removed {ridge_filtered} candidates "
            f"(min_shrunk_edge={ridge_edge_floor:.6f}, min_pos_fold_frac={ridge_pos_floor:.2f})..."
        )
        gated_df2_full = df2_full.loc[ridge_keep_mask].copy()
        if gated_df2_full.empty and not df2_full.empty:
            min_per_family = int(cfg.get("phase2_min_representatives_per_family", 1))
            gated_df2_full = _ensure_min_family_representatives(
                df2_full,
                score_col="score_r",
                min_per_family=min_per_family,
            )
            tprint(
                f"{_stage_label(2)} -> {_stage_label(3)} ({mode}): sanity gate would remove all candidates; "
                f"retaining at least {min_per_family} candidate(s) per family as fallback..."
            )
        elif not gated_df2_full.empty:
            gated_df2_full = _ensure_min_family_representatives(
                gated_df2_full,
                score_col="score_r",
                min_per_family=int(cfg.get("phase2_min_representatives_per_family", 1)),
            )
        df2_full = gated_df2_full
    df2_full["feature_base"] = [
        candidate_registry[str(name)].get("feature_base", str(name))
        for name in df2_full["name"].astype(str).values
    ]
    df2_full["family"] = [
        candidate_registry[str(name)].get("family", "unknown")
        for name in df2_full["name"].astype(str).values
    ]
    df2_full = df2_full.groupby("feature_base").head(
        int(cfg.get("phase2_prefilter_max_per_feature", 1))
    ).copy()
    phase3_eval_max = int(
        cfg.get(
            "phase3_eval_max_candidates",
            cfg.get("shortlist_max_candidates", cfg.get("stage3_max_candidates", 10)),
        )
    )
    if df2_full.shape[0] > phase3_eval_max:
        tprint(
            f"{_stage_label(2)} -> {_stage_label(3)} ({mode}): narrowing from {df2_full.shape[0]} to top {phase3_eval_max} candidates..."
        )
    df2 = _ensure_min_feature_representatives(
        df2_full,
        score_col="score_r",
        min_per_feature=int(cfg.get("phase2_min_representatives_per_feature", 1)),
        max_total=phase3_eval_max,
    )
    df2 = _cap_stage_family_dominance(
        df2,
        score_col="score_r",
        stage=f"{_stage_label(2)} shortlist",
        mode=mode,
        max_per_family=int(cfg.get("phase2_max_candidates_per_family", 3)),
    )
    df2 = _ensure_min_family_representatives(
        df2,
        score_col="score_r",
        min_per_family=int(cfg.get("phase2_min_representatives_per_family", 2)),
    )
    _tprint_candidate_table_support_summary(f"{_stage_label(2)} shortlist", mode, df2)
    _save_stage_artifacts(
        cfg,
        mode,
        2,
        payload={"df2": df2.copy(), "candidate_masks": candidate_masks},
        tables={"shortlist": df2.copy()},
    )
    if stop_stage <= 2:
        return _stage_stop_result(stage_num=2, mode=mode, candidate_table=df2)
    if start_stage > 2:
        loaded_stage2 = _load_stage_artifacts(cfg, mode, 2)
        if loaded_stage2 is not None:
            tprint(f"{_stage_label(3)} ({mode}): loading input state from stage 2 artifacts")
            df2 = loaded_stage2.get("df2", df2).copy()
            candidate_masks = loaded_stage2.get("candidate_masks", candidate_masks)

    tprint(f"{_stage_label(3)} ({mode}): evaluating {len(df2)} candidates for feature learnability and economic gain...")

    def _get_candidate_masks(name: str) -> Dict[str, np.ndarray]:
        if name in candidate_masks:
            return candidate_masks[name]
        reg = candidate_registry[name]
        if reg.get("kind") == "location_filter":
            zc_local = shared["location_filter_frame"]
        else:
            z = int(int(reg["lookback_h"]) * bph)
            if z not in global_z_cache:
                tprint(f"Precomputing global rolling tensors for z={z} bars...")
                global_z_cache[z] = _compute_z_cache(
                    high=shared["high"],
                    low=shared["low"],
                    close=shared["close"],
                    ret_1=shared["ret_1"],
                    vol_g=shared["vol_g"],
                    asset_groups=shared["asset_groups"],
                    z=z,
                    bph=bph,
                    volume=shared.get("volume", None),
                )
            zc_local = global_z_cache[z]
        m_high, m_low = _generate_event_masks_fast(
            candidate=reg,
            zc=zc_local,
        )
        candidate_masks[name] = {
            "m_high": m_high,
            "m_low": m_low,
            "side_mask": _get_side_mask(mode, m_high, m_low),
        }
        return candidate_masks[name]

    phase2_overlap_masks = {
        str(name): _get_candidate_masks(str(name))["side_mask"]
        for name in df2_full["name"].astype(str).values
    }
    df2_full = _prune_candidates_by_mask_overlap(
        df2_full,
        score_col="score_r",
        candidate_masks=phase2_overlap_masks,
        overlap_threshold=float(cfg.get("phase2_overlap_prune_threshold", 0.92)),
    )

    def _compute_cached_stage_metrics(name: str, side_mask: np.ndarray) -> Dict[str, Any]:
        if name in candidate_stage_metrics_cache:
            return candidate_stage_metrics_cache[name]
        feat_metrics = _compute_phase3_feature_learnability(
            shared, feature_dict, bundle, cfg
        )
        cond_metrics = _compute_conditional_predictability_metrics(bundle, cfg)
        econ_metrics = _compute_tbm_economic_gain(shared, side_mask, mode, folds, cfg)
        mfe_metrics = _compute_mfe_coverage(shared, side_mask, cfg)
        tbm_lgbm_metrics = _compute_phase4_tbm_lgbm_metrics(
            shared,
            side_mask,
            folds,
            cfg,
            econ_metrics["per_geometry_metrics"],
        )
        candidate_stage_metrics_cache[name] = {
            "feature_learnability_gain": np.float32(
                feat_metrics["feature_learnability_gain"]
            ),
            "feature_positive_fold_fraction": np.float32(
                feat_metrics["feature_positive_fold_fraction"]
            ),
            "conditional_predictability_gain": np.float32(
                cond_metrics["conditional_predictability_gain"]
            ),
            "conditional_predictability_positive_fold_fraction": np.float32(
                cond_metrics["conditional_predictability_positive_fold_fraction"]
            ),
            "conditional_predictability_regime_r2": np.float32(
                cond_metrics["conditional_predictability_regime_r2"]
            ),
            "conditional_predictability_baseline_r2": np.float32(
                cond_metrics["conditional_predictability_baseline_r2"]
            ),
            "feature_conditioned_spread": np.float32(
                cond_metrics["feature_conditioned_spread"]
            ),
            "economic_gain_r": np.float32(econ_metrics["economic_gain_r"]),
            "geometry_weighted_mfe_coverage": np.float32(
                econ_metrics["geometry_weighted_mfe_coverage"]
            ),
            "fixed_tp_mfe_coverage": np.float32(
                mfe_metrics["fixed_tp_mfe_coverage"]
            ),
            "aggregate_mfe_coverage": np.float32(
                econ_metrics["geometry_weighted_mfe_coverage"]
            ),
            "tbm_lgbm_auc_regime": np.float32(
                tbm_lgbm_metrics["tbm_lgbm_auc_regime"]
            ),
            "tbm_lgbm_auc_baseline": np.float32(
                tbm_lgbm_metrics["tbm_lgbm_auc_baseline"]
            ),
            "tbm_lgbm_auc_lift_vs_baseline": np.float32(
                tbm_lgbm_metrics["tbm_lgbm_auc_lift_vs_baseline"]
            ),
            "tbm_lgbm_top_bucket_lift_vs_baseline": np.float32(
                tbm_lgbm_metrics["tbm_lgbm_top_bucket_lift_vs_baseline"]
            ),
            "tbm_lgbm_positive_fold_fraction": np.float32(
                tbm_lgbm_metrics["tbm_lgbm_positive_fold_fraction"]
            ),
            "tbm_lgbm_stability": np.float32(
                tbm_lgbm_metrics["tbm_lgbm_stability"]
            ),
            "tbm_lgbm_selected_geometry": str(
                tbm_lgbm_metrics["tbm_lgbm_selected_geometry"]
            ),
            "tbm_lgbm_invalid_reason_regime": str(
                tbm_lgbm_metrics.get("tbm_lgbm_invalid_reason_regime", "")
            ),
            "tbm_lgbm_invalid_reason_baseline": str(
                tbm_lgbm_metrics.get("tbm_lgbm_invalid_reason_baseline", "")
            ),
            "tbm_lgbm_invalid_reason_full": str(
                tbm_lgbm_metrics.get("tbm_lgbm_invalid_reason_full", "")
            ),
        }
        return candidate_stage_metrics_cache[name]


    # -------------------------------------------------------------------------
    # Ridge Economic Diagnostics: Ridge regime attribution
    # -------------------------------------------------------------------------
    tprint(f"Ridge Economic Diagnostics ({mode}): Ridge regime attribution for top {len(df2)} candidates...")

    full_df_dict = {
        "timestamp": shared["timestamps"],
        "high": shared["high"],
        "low": shared["low"],
        "close": shared["close"],
    }
    if "open" in shared:
        full_df_dict["open"] = shared["open"]
    if "volume" in shared:
        full_df_dict["volume"] = shared["volume"]

    full_df = pd.DataFrame(full_df_dict)
    regime_features_df = build_regime_features(full_df)
    ridge_feature_cols_avail = [c for c in RIDGE_FEATURE_COLS if c in regime_features_df.columns]
    ridge_feature_matrix = (
        regime_features_df[ridge_feature_cols_avail]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32, copy=True)
    )
    ridge_timestamps = full_df["timestamp"].to_numpy()
    fwd_2h_bars = int(2 * bph)
    if "close" in full_df:
        fwd_ret_arr = (
            (full_df["close"].shift(-fwd_2h_bars) / full_df["close"] - 1.0)
            .to_numpy(dtype=np.float32, copy=False)
        )
    else:
        fwd_ret_arr = np.zeros(len(full_df), dtype=np.float32)
    ridge_scan_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _get_lgbm_scan_result(candidate_name: str, side_mask_local: np.ndarray) -> Optional[Dict[str, Any]]:
        cached = ridge_scan_cache.get(candidate_name)
        if candidate_name in ridge_scan_cache:
            return cached
        if not ridge_feature_cols_avail:
            ridge_scan_cache[candidate_name] = None
            return None
        res_local = fit_lgbm_regime_scan_arrays(
            feature_matrix=ridge_feature_matrix,
            feature_cols=ridge_feature_cols_avail,
            event_mask=side_mask_local.astype(bool, copy=False),
            target_values=fwd_ret_arr,
            timestamps=ridge_timestamps,
            n_splits=max(2, len(folds)),
        )
        ridge_scan_cache[candidate_name] = res_local
        return res_local

    def _get_ridge_scan_result(candidate_name: str, side_mask_local: np.ndarray) -> Optional[Dict[str, Any]]:
        cached = ridge_scan_cache.get(candidate_name)
        if candidate_name in ridge_scan_cache:
            return cached
        if not ridge_feature_cols_avail:
            ridge_scan_cache[candidate_name] = None
            return None
        res_local = fit_ridge_regime_scan_arrays(
            feature_matrix=ridge_feature_matrix,
            feature_cols=ridge_feature_cols_avail,
            event_mask=side_mask_local.astype(bool, copy=False),
            target_values=fwd_ret_arr,
            timestamps=ridge_timestamps,
            n_splits=max(2, len(folds)),
        )
        ridge_scan_cache[candidate_name] = res_local
        return res_local

    # Identify which features are binary vs continuous
    feature_types = {}
    for c in RIDGE_FEATURE_COLS:
        if c in regime_features_df.columns:
            u_vals = regime_features_df[c].dropna().unique()
            if len(u_vals) <= 2 and set(u_vals).issubset({0.0, 1.0, 0, 1}):
                feature_types[c] = "binary"
            else:
                feature_types[c] = "continuous"

    regime_impact_rows: List[pd.DataFrame] = []
    dynamic_conditioners: Dict[str, List[Dict[str, Any]]] = {}
    loaded_stage3 = _load_stage_artifacts(cfg, mode, 3) if start_stage > 3 else None
    if loaded_stage3 is not None:
        tprint(f"{_stage_label(4)} ({mode}): loading input state from stage 3 artifacts")
        base_phase3_parents = loaded_stage3.get("base_phase3_parents", df2).copy()
        base_parent_masks = loaded_stage3.get("base_parent_masks", {})
        phase25_seeds_df = loaded_stage3.get("phase25_seeds_df", pd.DataFrame()).copy()
        dynamic_conditioners = loaded_stage3.get("dynamic_conditioners", {})
        candidate_masks = loaded_stage3.get("candidate_masks", candidate_masks)
    else:
        for i, row in enumerate(df2.itertuples()):
            if (i + 1) % 5 == 0 or i == len(df2) - 1:
                tprint(f"{_stage_label(3)} ({mode}): processing candidate {i+1}/{len(df2)}: {str(getattr(row, 'name'))}")
            base_name = str(getattr(row, "name"))
            side_mask = _get_candidate_masks(base_name)["side_mask"]
            res = _get_lgbm_scan_result(base_name, side_mask)

            cond_features = []
            if res is not None:
                seeds = res.get("phase3_conditioner_seeds", [])
                for seed in seeds:
                    cond_features.append({
                        "feature": seed.feature,
                        "coef": seed.coefficient,
                        "abs_signed_importance": seed.abs_signed_importance,
                        "type": seed.feature_type,
                        "thresholds": seed.thresholds
                    })

            dynamic_conditioners[base_name] = cond_features
            if res is not None and "ranked_features" in res:
                c_df = res["ranked_features"].copy()
                c_df["base_candidate"] = base_name
                regime_impact_rows.append(c_df)

        base_phase3_parents = df2.copy()
        if "keep_pct_vs_original" not in base_phase3_parents.columns and "total_events" in base_phase3_parents.columns:
            base_phase3_parents["keep_pct_vs_original"] = (
                100.0
                * pd.to_numeric(base_phase3_parents["total_events"], errors="coerce").fillna(0.0)
                / max(original_sample_rows, 1)
            ).astype(np.float32)
        base_parent_masks = {
            str(row["name"]): _get_candidate_masks(str(row["name"]))["side_mask"]
            for _, row in base_phase3_parents.iterrows()
        }
        phase25_seeds_df = (
            pd.concat(regime_impact_rows, ignore_index=True)
            if regime_impact_rows
            else pd.DataFrame()
        )
    _tprint_candidate_table_support_summary(f"{_stage_label(3)} parents", mode, base_phase3_parents)
    _save_stage_artifacts(
        cfg,
        mode,
        3,
        payload={
            "base_phase3_parents": base_phase3_parents.copy(),
            "base_parent_masks": base_parent_masks,
            "phase25_seeds_df": phase25_seeds_df.copy(),
            "dynamic_conditioners": dynamic_conditioners,
            "candidate_masks": candidate_masks,
        },
        tables={
            "parents": base_phase3_parents.copy(),
            "regime_impact": phase25_seeds_df.copy(),
        },
    )
    if stop_stage <= 3:
        return _stage_stop_result(
            stage_num=3,
            mode=mode,
            candidate_table=base_phase3_parents,
            extra={"phase25_regime_impact_": phase25_seeds_df},
        )
    trigger_cfg = TriggerDiscoveryConfig.from_mapping(cfg)
    trigger_parent_mode = _phase3_parent_mode(cfg)
    trigger_diagnostics: Dict[str, Any] = {}
    trigger_all_df = pd.DataFrame()
    loaded_stage4 = _load_stage_artifacts(cfg, mode, 4) if start_stage > 4 else None
    if loaded_stage4 is not None:
        tprint(f"{_stage_label(5)} ({mode}): loading input state from stage 4 artifacts")
        df2 = loaded_stage4.get("df2", base_phase3_parents).copy()
        trigger_all_df = loaded_stage4.get("trigger_all_df", pd.DataFrame()).copy()
        trigger_diagnostics = loaded_stage4.get("trigger_diagnostics", {})
        phase25_seeds_df = loaded_stage4.get("phase25_seeds_df", phase25_seeds_df).copy()
        dynamic_conditioners = loaded_stage4.get("dynamic_conditioners", dynamic_conditioners)
        candidate_masks = loaded_stage4.get("candidate_masks", candidate_masks)
    elif trigger_cfg.enabled and trigger_parent_mode == "regime_trigger":
        tprint(
            f"{_stage_label(4)} ({mode}): evaluating trigger templates for {len(base_phase3_parents)} parent regimes..."
        )
        trigger_all_df, trigger_survivors_df, trigger_diagnostics = run_trigger_discovery(
            phase2_survivors_df=base_phase3_parents,
            phase25_seeds_df=phase25_seeds_df,
            parent_masks=base_parent_masks,
            shared={**shared, "runtime_cfg": cfg},
            feature_dict=feature_dict,
            cv_splits=folds,
            signed_returns=global_signed_returns,
            asset_groups=shared["asset_groups"],
            config=trigger_cfg,
            compute_full_metrics_fn=_compute_full_metrics_for_candidate,
            mode=mode,
        )
        if not trigger_survivors_df.empty:
            for trigger_name, trigger_mask in trigger_diagnostics.get("candidate_masks", {}).items():
                candidate_masks[str(trigger_name)] = {
                    "m_high": trigger_mask.copy() if _mode_is_up(mode) else np.zeros_like(trigger_mask, dtype=bool),
                    "m_low": trigger_mask.copy() if not _mode_is_up(mode) else np.zeros_like(trigger_mask, dtype=bool),
                    "side_mask": trigger_mask.copy(),
                }
            df2 = trigger_survivors_df.drop(columns=["_event_mask"], errors="ignore").copy()
            if "keep_pct_vs_original" not in df2.columns and "total_events" in df2.columns:
                df2["keep_pct_vs_original"] = (
                    100.0
                    * pd.to_numeric(df2["total_events"], errors="coerce").fillna(0.0)
                    / max(original_sample_rows, 1)
                ).astype(np.float32)
            df2 = _cap_stage_family_dominance(
                df2,
                score_col="trigger_score_final",
                stage=_stage_label(4),
                mode=mode,
                max_per_family=int(cfg.get("max_candidates_per_family_per_stage", 3)),
            )
            tprint(
                f"{_stage_label(4)} ({mode}): {len(df2)} trigger survivors passed to {_stage_label(5)}."
            )
            _tprint_candidate_table_support_summary(f"{_stage_label(4)} survivors", mode, df2)
        else:
            tprint(f"{_stage_label(4)} ({mode}): no trigger survivors.")
            if not bool(cfg.get("fallback_to_base_regime_if_no_trigger_survives", False)):
                return {
                    "status": "failed",
                    "reason": f"no_trigger_survivors_{mode}",
                    "layer0_candidate_table_": base_phase3_parents,
                    "phase25_regime_impact_": phase25_seeds_df,
                }
            df2 = base_phase3_parents.copy()
    else:
        df2 = base_phase3_parents.copy()
    _save_stage_artifacts(
        cfg,
        mode,
        4,
        payload={
            "df2": df2.copy(),
            "trigger_all_df": trigger_all_df.copy() if isinstance(trigger_all_df, pd.DataFrame) else pd.DataFrame(),
            "trigger_diagnostics": trigger_diagnostics,
            "phase25_seeds_df": phase25_seeds_df.copy(),
            "dynamic_conditioners": dynamic_conditioners,
            "candidate_masks": candidate_masks,
        },
        tables={
            "trigger_candidates": trigger_all_df.copy() if isinstance(trigger_all_df, pd.DataFrame) else pd.DataFrame(),
            "trigger_survivors": df2.copy(),
        },
    )
    if stop_stage <= 4:
        return _stage_stop_result(
            stage_num=4,
            mode=mode,
            candidate_table=df2,
            extra={"phase25_regime_impact_": phase25_seeds_df},
        )


    metrics_list = []

    for i, row in enumerate(df2.itertuples()):
        if (i + 1) % 5 == 0 or i == len(df2) - 1:
            tprint(f"Ridge Economic Diagnostics ({mode}): processing candidate {i+1}/{len(df2)}: {getattr(row, 'name')}")
        base_name = str(getattr(row, "name"))
        side_mask = _get_candidate_masks(base_name)["side_mask"]
        metrics_list.append(_compute_cached_stage_metrics(base_name, side_mask))

        # Capture drivers for winners reporting (fixes "no_data" for triggers)
        res_ridge = _get_ridge_scan_result(base_name, side_mask)
        if res_ridge is not None and "ranked_features" in res_ridge:
            c_df_impact = res_ridge["ranked_features"].copy()
            c_df_impact["base_candidate"] = base_name
            regime_impact_rows.append(c_df_impact)

    metrics_df = pd.DataFrame(metrics_list, index=df2.index, columns=[
        "feature_learnability_gain", "feature_positive_fold_fraction",
        "conditional_predictability_gain", "conditional_predictability_positive_fold_fraction",
        "conditional_predictability_regime_r2", "conditional_predictability_baseline_r2",
        "feature_conditioned_spread", "economic_gain_r", "geometry_weighted_mfe_coverage",
        "fixed_tp_mfe_coverage", "aggregate_mfe_coverage",
        "tbm_lgbm_auc_regime", "tbm_lgbm_auc_baseline", "tbm_lgbm_auc_lift_vs_baseline",
        "tbm_lgbm_top_bucket_lift_vs_baseline", "tbm_lgbm_positive_fold_fraction",
        "tbm_lgbm_stability", "tbm_lgbm_selected_geometry",
        "tbm_lgbm_invalid_reason_regime", "tbm_lgbm_invalid_reason_baseline",
        "tbm_lgbm_invalid_reason_full"
    ])
    df2 = pd.concat([df2, metrics_df], axis=1)
    min_mfe_cov = float(cfg.get("mask_opt_min_mfe_coverage", 0.02))
    _log_stage_snapshot(
        mode,
        "Ridge Economic Diagnostics",
        df2,
        "conditional_predictability_gain",
        [
            "name",
            "conditional_predictability_gain",
            "conditional_predictability_positive_fold_fraction",
            "conditional_predictability_regime_r2",
            "conditional_predictability_baseline_r2",
            "feature_conditioned_spread",
            "feature_learnability_gain",
            "feature_positive_fold_fraction",
            "score_r",
            "delta_r_raw",
        ],
    )
    _log_stage_snapshot(
        mode,
        "Stage 4 Coverage",
        df2,
        "aggregate_mfe_coverage",
        [
            "name",
            "aggregate_mfe_coverage",
            "economic_gain_r",
            "score_r",
            "delta_r_raw",
        ],
    )
    # Ensure score_r is correctly calculated for all survivors (including triggers)
    df2["score_r"] = (
        df2["delta_r_shrunk"].astype(np.float32).values
        * np.sqrt(np.maximum(0, df2["total_events"].astype(np.float32).values))
        * df2["S_r"].astype(np.float32).values
        / (1.0 + df2["D_r"].astype(np.float32).values)
    ).astype(np.float32)

    df2["coverage_multiplier"] = (
        0.25
        + 0.75
        * np.clip(
            df2["aggregate_mfe_coverage"].astype(np.float32).values
            / max(min_mfe_cov, 1e-6),
            0.0,
            1.0,
        )
    ).astype(np.float32)
    df2["predictability_anchor"] = np.maximum(
        df2["conditional_predictability_gain"].astype(np.float32).values, 0.0
    ).astype(np.float32)
    df2["predictability_positive_multiplier"] = (
        0.75
        + 0.25
        * df2["conditional_predictability_positive_fold_fraction"].astype(np.float32).values
    ).astype(np.float32)
    df2["spread_multiplier"] = (
        1.0
        + np.tanh(10.0 * df2["feature_conditioned_spread"].astype(np.float32).values)
    ).astype(np.float32)
    df2["difference_prior"] = (
        0.85 + 0.15 * np.clip(df2["score_r"].astype(np.float32).values, 0.0, None)
    ).astype(np.float32)
    df2["score_ml"] = (
        df2["score_r"].astype(np.float32).values
        * (
            1.0
            + 5.0 * df2["predictability_anchor"].astype(np.float32).values
        )
        * df2["predictability_positive_multiplier"].astype(np.float32).values
        * df2["spread_multiplier"].astype(np.float32).values
        * df2["difference_prior"].astype(np.float32).values
    ).astype(np.float32)
    df2["tbm_auc_support"] = np.nan_to_num(np.maximum(
        df2["tbm_lgbm_auc_lift_vs_baseline"].astype(np.float32).values, 0.0
    ), nan=0.0).astype(np.float32)
    df2["tbm_lift_support"] = np.nan_to_num(np.maximum(
        df2["tbm_lgbm_top_bucket_lift_vs_baseline"].astype(np.float32).values,
        0.0,
    ), nan=0.0).astype(np.float32)
    df2["score_ml_trading"] = (
        df2["score_ml"].astype(np.float32).values
        * (1.0 + 2.0 * np.maximum(df2["feature_learnability_gain"].astype(np.float32).values, 0.0))
        * (1.0 + 25.0 * df2["tbm_auc_support"].astype(np.float32).values)
        * (1.0 + 10.0 * df2["tbm_lift_support"].astype(np.float32).values)
        * (
            0.50
            + 0.50
            * np.nan_to_num(np.maximum(df2["tbm_lgbm_stability"].astype(np.float32).values, 0.0), nan=0.0)
        )
        * (
            0.50
            + 0.50
            * np.nan_to_num(np.maximum(
                df2["tbm_lgbm_positive_fold_fraction"].astype(np.float32).values, 0.0
            ), nan=0.0)
        )
        * df2["coverage_multiplier"].astype(np.float32).values
    ).astype(np.float32)
    df2["shortlist_score"] = df2["score_ml_trading"].astype(np.float32)
    df2["decision"] = "ranked"
    df2["regime_id"] = df2["name"].astype(str)
    df2["regime_definition"] = df2["name"].astype(str)
    df2["rationale"] = df2.apply(_build_regime_rationale, axis=1)

    df2 = df2.sort_values(
        [
            "score_ml_trading",
            "economic_gain_r",
            "feature_learnability_gain",
            "delta_r",
            "total_events",
        ],
        ascending=[False, False, False, False, False],
    )

    if "feature_base" not in df2.columns:
        df2["feature_base"] = [
            candidate_registry[str(x)].get("feature_base", str(x))
            for x in df2["name"].astype(str).values
        ]
    else:
        df2["feature_base"] = df2["feature_base"].fillna(df2["name"].astype(str))
    if "family" not in df2.columns:
        df2["family"] = [
            candidate_registry[str(x)].get("family", "unknown")
            for x in df2["name"].astype(str).values
        ]
    else:
        df2["family"] = df2["family"].fillna("unknown")

    # Stage 2.5/3 diversity: feature-based pruning only.
    df2 = df2.groupby("feature_base").head(3).copy()
    df2 = _ensure_min_feature_representatives(
        df2,
        score_col="score_ml_trading",
        min_per_feature=int(cfg.get("phase25_min_representatives_per_feature", 1)),
    )

    shortlist_max = int(cfg.get("shortlist_max_candidates", cfg.get("stage3_max_candidates", 10)))
    df_short = df2.sort_values(
        ["score_ml_trading", "score_ml", "score_r", "total_events"],
        ascending=[False, False, False, False],
    ).head(shortlist_max).copy()

    if df_short.empty:
        return {
            "status": "failed",
            "reason": f"no_shortlist_candidates_{mode}",
            "layer0_candidate_table_": df2,
        }
    df_short = _ensure_min_feature_representatives(
        df_short,
        score_col="score_ml_trading",
        min_per_feature=int(cfg.get("phase25_min_representatives_per_feature", 1)),
    )

    df_short["tier"] = 0
    df_short["conditioner_mode"] = "none"
    if "parent_child_relation_type" not in df_short.columns:
        df_short["parent_child_relation_type"] = _phase3_parent_relation_type(
            0, bool(_phase3_parent_mode(cfg) == "regime_trigger")
        )
    if "full_event_definition" not in df_short.columns:
        df_short["full_event_definition"] = df_short["regime_definition"].astype(str)
    if "support_ratio_vs_parent" not in df_short.columns:
        df_short["support_ratio_vs_parent"] = 1.0

    for _, row in df_short.iterrows():
        _get_candidate_masks(str(row["name"]))

    keep_z_values = {
        int(float(z_hr) * bph)
        for z_hr in df_short["z_hours"].astype(float).values
    }
    _trim_z_cache(global_z_cache, keep_z_values)

    cond_rows: List[pd.Series] = []
    phase3_rejection_reasons: Dict[str, int] = {}
    loaded_stage5 = _load_stage_artifacts(cfg, mode, 5) if start_stage > 5 else None
    if loaded_stage5 is not None:
        tprint(f"{_stage_label(6)} ({mode}): loading input state from stage 5 artifacts")
        df2 = loaded_stage5.get("df2", df2).copy()
        df_short = loaded_stage5.get("df_short", df_short).copy()
        phase25_seeds_df = loaded_stage5.get("phase25_seeds_df", phase25_seeds_df).copy()
        candidate_masks = loaded_stage5.get("candidate_masks", candidate_masks)
    elif bool(cfg.get("enable_secondary_conditioners", True)):
        # Configurable limits
        min_events = int(cfg.get("phase3_min_conditioned_event_count", 2000))
        min_fraction = float(cfg.get("phase3_min_event_fraction_of_base", 0.10))
        tier2_min_fraction = float(cfg.get("phase3_tier2_min_event_fraction", 0.05))
        max_singles = int(cfg.get("phase3_max_single_candidates_per_base", 4))
        max_pairs = int(cfg.get("phase3_max_pair_candidates", 10))

        for _, row in df_short.iterrows():
            cand_name = str(row["name"])
            z = int(float(row["z_hours"]) * bph)
            if z not in global_z_cache:
                tprint(f"Precomputing Phase 3 rolling tensors for z={z} bars...")
                global_z_cache[z] = _compute_z_cache(
                    high=shared["high"],
                    low=shared["low"],
                    close=shared["close"],
                    ret_1=shared["ret_1"],
                    vol_g=shared["vol_g"],
                    asset_groups=shared["asset_groups"],
                    z=z,
                    bph=bph,
                    volume=shared.get("volume"),
                )
            zc = global_z_cache[z]
            base_masks = candidate_masks[cand_name]
            base_side_mask = base_masks.get("side_mask")
            if base_side_mask is None:
                base_side_mask = _get_side_mask(
                    mode, base_masks["m_high"], base_masks["m_low"]
                )
                base_masks["side_mask"] = base_side_mask
            base_event_count = int(np.sum(base_side_mask))
            base_stage_metrics = _compute_cached_stage_metrics(cand_name, base_side_mask)

            # ---------------------------------------------------------
            # 3A. Generate Single-Regime Candidates (Tier-1)
            # ---------------------------------------------------------
            tier1_candidates = []
            top_vars = dynamic_conditioners.get(_phase3_parent_seed_key(row), [])

            for var_info in top_vars:
                var_name = var_info["feature"]
                coef = var_info["coef"]
                v_type = var_info["type"]
                family = var_info.get("family", "unknown")

                if var_name not in regime_features_df.columns:
                    continue

                feature_vals = regime_features_df[var_name].values
                valid_mask = np.isfinite(feature_vals)
                active_valid = base_side_mask & valid_mask
                if np.sum(active_valid) < 50:
                    continue

                if v_type == "binary":
                    target_val = 1 if coef > 0 else 0
                    cond_mask = valid_mask & (feature_vals == target_val)
                    tier1_candidates.append({
                        "name": f"{cand_name}_{var_name}_is_{target_val}",
                        "desc": f"{var_name} == {target_val}",
                        "mask": cond_mask,
                        "features": [var_name],
                        "families": [family]
                    })
                else:
                    direction = "gt" if coef > 0 else "lt"
                    thresholds_dict = var_info.get("thresholds")
                    if not thresholds_dict:
                        continue

                    # Only evaluate directional tails. Skip median-adjacent cuts and
                    # neutral middle bands so Phase 3 focuses on genuinely extreme
                    # positive/negative feature states discovered on event rows.
                    quantiles_to_check = ["q60", "q70", "q80"] if coef > 0 else ["q40", "q30", "q20"]
                    for q_key in quantiles_to_check:
                        if q_key in thresholds_dict:
                            threshold = thresholds_dict[q_key]
                            if direction == "gt":
                                cond_mask = valid_mask & (feature_vals > threshold)
                                desc = f"{var_name} > {q_key}"
                            else:
                                cond_mask = valid_mask & (feature_vals < threshold)
                                desc = f"{var_name} < {q_key}"

                            tier1_candidates.append({
                                "name": f"{cand_name}_{var_name}_{desc.replace(' ', '').replace('>', 'gt').replace('<', 'lt')}",
                                "desc": desc,
                                "mask": cond_mask,
                                "features": [var_name],
                                "families": [family]
                            })

            # Base Evaluation Closure
            def eval_candidate(c_info, tier, parent_res=None):
                new_side_mask = base_side_mask & c_info["mask"]
                tot_events = int(np.sum(new_side_mask))
                parent_reference = parent_res if tier == 2 and parent_res is not None else row
                parent_total_events_local = int(parent_reference.get("total_events", base_event_count))

                req_fraction = min_fraction if tier == 1 else tier2_min_fraction
                if tot_events < min_events or (tot_events / max(parent_total_events_local, 1)) < req_fraction:
                    _record_rejection_reason(phase3_rejection_reasons, "too_few_events_or_support_ratio")
                    return None
                support_stats = _cheap_support_stats(
                    mask=new_side_mask,
                    day_ids=shared["day_ids"],
                    n_days=shared["n_days"],
                    symbol_codes=shared["symbol_codes"],
                    symbol_uniques=shared["symbol_uniques"],
                    timestamps=shared["timestamps"],
                    asset_groups=shared["asset_groups"],
                    bph=bph,
                    folds=folds,
                )
                passed_support_gate, reject_reason = _passes_cheap_support_gate(
                    stats=support_stats,
                    cfg=cfg,
                    phase_prefix="phase3",
                )
                if not passed_support_gate:
                    _record_rejection_reason(phase3_rejection_reasons, reject_reason)
                    return None
                symbol_summary_new = _mask_symbol_concentration_summary(
                    new_side_mask,
                    shared["symbol_codes"],
                    shared["symbol_uniques"],
                )
                min_symbols_phase3 = int(cfg.get("phase3_min_distinct_symbols", 6))
                max_top_share_phase3 = float(cfg.get("phase3_max_top_symbol_share", 0.40))
                if symbol_summary_new["event_symbol_count"] < min_symbols_phase3:
                    _record_rejection_reason(phase3_rejection_reasons, "too_few_symbols")
                    return None
                if symbol_summary_new["top_symbol_share"] > max_top_share_phase3:
                    _record_rejection_reason(phase3_rejection_reasons, "top_symbol_share_too_high")
                    return None
                duration_stats_new = _mask_run_duration_stats(
                    new_side_mask, shared["asset_groups"], bph
                )

                if reg.get("kind") == "location_filter":
                    coh = {
                        "bars_to_peak_dispersion": 0.0,
                        "speed_dispersion": 0.0,
                        "monotonicity_dispersion": 0.0,
                        "impulse_shape_dispersion": 0.0,
                    }
                else:
                    coh = (
                        _coherence_metrics_single_side(new_side_mask, zc["b_up"], zc["s_up"], zc["m_up"])
                        if _mode_is_up(mode)
                        else _coherence_metrics_single_side(new_side_mask, zc["b_dn"], zc["s_dn"], zc["m_dn"])
                    )

                valid_fwd_new = np.isfinite(global_signed_returns)
                non_event_new = (~new_side_mask) & valid_fwd_new
                basic_edge_new = (
                    float(np.nanmean(global_signed_returns[new_side_mask & valid_fwd_new]) - np.nanmean(global_signed_returns[non_event_new]))
                    if np.any(new_side_mask & valid_fwd_new) and np.any(non_event_new)
                    else 0.0
                )

                new_metrics = _compute_full_metrics_for_candidate(
                    mode,
                    new_side_mask,
                    shared,
                    feature_dict,
                    cfg,
                    float(coh["impulse_shape_dispersion"]),
                    float(basic_edge_new),
                )

                econ_metrics = _compute_tbm_economic_gain(shared, new_side_mask, mode, folds, cfg)
                mfe_metrics = _compute_mfe_coverage(shared, new_side_mask, cfg)
                new_econ = _metric_or_nan(econ_metrics.get("economic_gain_r"))
                new_mfe = _metric_or_nan(mfe_metrics.get("fixed_tp_mfe_coverage"))

                # In order to do base comparison, we need base_econ. If row doesn't have it, we must compute it.
                base_econ = _metric_or_nan(base_stage_metrics.get("economic_gain_r"))
                base_mfe = _metric_or_nan(
                    base_stage_metrics.get("aggregate_mfe_coverage")
                )

                improves_econ = (new_econ > base_econ * 1.05)
                improves_mfe = (new_mfe > base_mfe * 1.05)

                # Check net regime value
                best_geom = econ_metrics.get("per_geometry_metrics", [{}])[0]
                labels_ER = np.asarray(
                    best_geom.get("labels_resolved", best_geom.get("labels", np.array([]))),
                    dtype=np.float32,
                )
                resolved_ER = np.asarray(
                    best_geom.get("resolved_mask", np.zeros_like(new_side_mask, dtype=np.int8)),
                    dtype=bool,
                )

                base_best_geom = base_stage_metrics.get("per_geometry_metrics", [{}])[0]
                labels_E = np.asarray(
                    base_best_geom.get("labels_resolved", base_best_geom.get("labels", np.array([]))),
                    dtype=np.float32,
                )
                resolved_E = np.asarray(
                    base_best_geom.get("resolved_mask", np.zeros_like(base_side_mask, dtype=np.int8)),
                    dtype=bool,
                )

                event_mask_ER = new_side_mask & resolved_ER
                event_mask_E = base_side_mask & resolved_E

                auc_ER = quick_tree_auc(ridge_feature_matrix, labels_ER, event_mask_ER, folds)
                auc_E = quick_tree_auc(ridge_feature_matrix, labels_E, event_mask_E, folds)

                fwd_ret_ER = global_signed_returns[new_side_mask & valid_fwd_new]
                fwd_ret_E = global_signed_returns[base_side_mask & valid_fwd_new]

                nrv_score, nrv_diags = compute_net_regime_value(
                    returns_E=fwd_ret_E,
                    returns_ER=fwd_ret_ER,
                    delta_r_folds_E=np.array([float(np.nanmean(global_signed_returns[va][(base_side_mask & valid_fwd_new)[va]])) if np.any((base_side_mask & valid_fwd_new)[va]) else 0.0 for _, va in folds]),
                    delta_r_folds_ER=np.array([float(np.nanmean(global_signed_returns[va][(new_side_mask & valid_fwd_new)[va]])) if np.any((new_side_mask & valid_fwd_new)[va]) else 0.0 for _, va in folds]),
                    labels_E=labels_E[event_mask_E] if len(labels_E) == len(base_side_mask) else np.array([]),
                    labels_ER=labels_ER[event_mask_ER] if len(labels_ER) == len(new_side_mask) else np.array([]),
                    auc_E=auc_E,
                    auc_ER=auc_ER,
                )

                new_metrics["net_regime_value"] = nrv_score

                # Stronger acceptance rules based on prompt
                der_ratio = nrv_diags["DER_ratio"]
                sr_ratio = nrv_diags["S_r_ratio"]

                # Check for deterioration
                is_stability_worse = (sr_ratio < 0.90)
                is_dispersion_worse = (der_ratio < 0.90)

                if tier == 1:
                    if not (improves_econ or improves_mfe or nrv_score > 1.05):
                        _record_rejection_reason(phase3_rejection_reasons, "no_tier1_improvement")
                        return None
                    if is_stability_worse or is_dispersion_worse:
                        _record_rejection_reason(phase3_rejection_reasons, "tier1_stability_or_dispersion_worse")
                        return None

                if tier == 2:
                    # Compare against BEST single parent if provided
                    if parent_res is not None:
                        parent_econ = _metric_or_nan(parent_res.get("economic_gain_r"))
                        parent_mfe = _metric_or_nan(parent_res.get("aggregate_mfe_coverage"))
                        parent_nrv = _metric_or_nan(parent_res.get("net_regime_value"))

                        if not (new_econ > parent_econ * 1.05 or new_mfe > parent_mfe * 1.05 or nrv_score > parent_nrv * 1.05):
                            _record_rejection_reason(phase3_rejection_reasons, "no_tier2_improvement")
                            return None
                    else:
                        if not (new_econ > base_econ * 1.1 or new_mfe > base_mfe * 1.1 or nrv_score > 1.1):
                            _record_rejection_reason(phase3_rejection_reasons, "no_tier2_improvement")
                            return None
                    if is_stability_worse or is_dispersion_worse:
                        _record_rejection_reason(phase3_rejection_reasons, "tier2_stability_or_dispersion_worse")
                        return None

                # Build row
                new_row = row.copy()
                new_row["name"] = c_info["name"]
                new_row["conditioner_mode"] = c_info["desc"]
                new_row["tier"] = tier
                new_row["total_events"] = tot_events
                new_row["parent_total_events"] = parent_total_events_local
                new_row["child_total_events"] = tot_events
                new_row["support_ratio_vs_parent"] = float(
                    tot_events / max(parent_total_events_local, 1)
                )
                new_row["keep_pct_vs_original"] = float(
                    100.0 * tot_events / max(original_sample_rows, 1)
                )
                new_row["event_symbol_count"] = int(symbol_summary_new["event_symbol_count"])
                new_row["top_symbol_share"] = float(symbol_summary_new["top_symbol_share"])
                new_row["top_symbol_counts_text"] = str(symbol_summary_new["top_symbol_counts_text"])
                new_row["avg_event_duration_bars"] = float(duration_stats_new["avg_event_duration_bars"])
                new_row["median_event_duration_bars"] = float(duration_stats_new["median_event_duration_bars"])
                new_row["avg_event_duration_hours"] = float(duration_stats_new["avg_event_duration_hours"])
                new_row["median_event_duration_hours"] = float(duration_stats_new["median_event_duration_hours"])
                new_row["event_run_count"] = float(duration_stats_new["event_run_count"])
                new_row["parent_child_relation_type"] = _phase3_parent_relation_type(
                    tier, bool(isinstance(row.get("trigger_id"), str) and row.get("trigger_id"))
                )
                parent_full_definition = str(
                    parent_reference.get(
                        "full_event_definition",
                        parent_reference.get("regime_definition", parent_reference.get("name")),
                    )
                )
                new_row["full_event_definition"] = f"{parent_full_definition} AND {c_info['desc']}"
                new_row["impulse_shape_dispersion"] = float(coh["impulse_shape_dispersion"])
                new_row["post_event_vol_dispersion"] = float(
                    _mask_post_event_vol_dispersion(
                        new_side_mask, shared["asset_groups"], shared["symbol_codes"], shared["timestamps"], bph
                    )
                )
                f_counts = support_stats.get("fold_event_counts", [])
                new_row["fold_event_count_std"] = float(np.std(f_counts)) if f_counts else 0.0
                new_row["fold_continuation_rate_std"] = float(new_metrics.get("fold_continuation_rate_std", 0.0))

                for k, v in new_metrics.items():
                    new_row[k] = v

                new_row["delta_r_raw"] = float(basic_edge_new)
                new_row["delta_r_fallback"] = (
                    float(0.5 * new_row["incremental_information_delta_auc"])
                    if np.isfinite(new_row.get("incremental_information_delta_auc", np.nan))
                    else float("nan")
                )
                raw_val = _metric_or_nan(new_row["delta_r_raw"])
                new_row["delta_r"] = float(raw_val)
                candidate_masks[c_info["name"]] = {
                    "m_high": new_side_mask.copy()
                    if _mode_is_up(mode)
                    else np.zeros_like(new_side_mask, dtype=bool),
                    "m_low": new_side_mask.copy()
                    if not _mode_is_up(mode)
                    else np.zeros_like(new_side_mask, dtype=bool),
                    "side_mask": new_side_mask,
                }
                candidate_stage_metrics_cache.pop(c_info["name"], None)

                return new_row

            # Evaluate Tier-1
            surviving_tier1 = []
            for c_info in tier1_candidates:
                eval_res = eval_candidate(c_info, tier=1)
                if eval_res is not None:
                    surviving_tier1.append((c_info, eval_res))

            # ---------------------------------------------------------
            # 3B. Select Top Single Regimes
            # ---------------------------------------------------------
            surviving_tier1.sort(key=lambda x: x[1].get("net_regime_value", 0.0), reverse=True)
            top_tier1 = surviving_tier1[:max_singles]

            for c_info, eval_res in top_tier1:
                cond_rows.append(eval_res)

            # ---------------------------------------------------------
            # 3C. Generate Two-Regime Candidates (Tier-2)
            # ---------------------------------------------------------
            tier2_candidates = []

            for i in range(len(top_tier1)):
                for j in range(i + 1, len(top_tier1)):
                    if len(tier2_candidates) >= max_pairs:
                        break
                    c1_info, r1 = top_tier1[i]
                    c2_info, r2 = top_tier1[j]

                    # Avoid redundant pairs (same feature)
                    if set(c1_info["features"]).intersection(set(c2_info["features"])):
                        continue

                    # Prefer cross-family combinations (skip if same family)
                    if set(c1_info["families"]).intersection(set(c2_info["families"])):
                        continue

                    combined_mask = c1_info["mask"] & c2_info["mask"]
                    # Determine best parent for relative comparison
                    best_parent_res = r1 if r1.get("net_regime_value", 0) > r2.get("net_regime_value", 0) else r2

                    tier2_candidates.append({
                        "name": f"{c1_info['name']}_AND_{c2_info['name'].replace(cand_name + '_', '')}",
                        "desc": f"{c1_info['desc']} AND {c2_info['desc']}",
                        "mask": combined_mask,
                        "features": c1_info["features"] + c2_info["features"],
                        "families": c1_info["families"] + c2_info["families"],
                        "best_parent_res": best_parent_res
                    })

            for c_info in tier2_candidates:
                parent_res = c_info.pop("best_parent_res")
                eval_res = eval_candidate(c_info, tier=2, parent_res=parent_res)
                if eval_res is not None:
                    cond_rows.append(eval_res)

    if cond_rows:
        df_short = pd.concat([df_short, pd.DataFrame(cond_rows)], ignore_index=True)
    _tprint_rejection_summary(_stage_label(5), mode, phase3_rejection_reasons)
    if not df_short.empty and "keep_pct_vs_original" not in df_short.columns and "total_events" in df_short.columns:
        df_short["keep_pct_vs_original"] = (
            100.0
            * pd.to_numeric(df_short["total_events"], errors="coerce").fillna(0.0)
            / max(original_sample_rows, 1)
        ).astype(np.float32)
    _save_stage_artifacts(
        cfg,
        mode,
        5,
        payload={
            "df2": df2.copy(),
            "df_short": df_short.copy(),
            "phase25_seeds_df": phase25_seeds_df.copy(),
            "candidate_masks": candidate_masks,
        },
        tables={
            "candidates": df2.copy(),
            "conditioned_shortlist": df_short.copy(),
        },
    )
    if stop_stage <= 5:
        return _stage_stop_result(
            stage_num=5,
            mode=mode,
            candidate_table=df2,
            shortlist_table=df_short,
            extra={"phase25_regime_impact_": phase25_seeds_df},
        )

    if not df_short.empty:
        def _df_col_or_zeros(df: pd.DataFrame, col: str) -> np.ndarray:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(
                    dtype=np.float32
                )
            return np.zeros(len(df), dtype=np.float32)

        def _df_col_or_constant(
            df: pd.DataFrame, col: str, default: float
        ) -> np.ndarray:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(
                    dtype=np.float32
                )
            return np.full(len(df), np.float32(default), dtype=np.float32)

        if "N_r" not in df_short.columns:
            df_short["N_r"] = pd.to_numeric(
                df_short.get("total_events", 0.0), errors="coerce"
            ).fillna(0.0).astype(np.float32)
        if "delta_r_shrunk" not in df_short.columns:
            delta_r_raw_arr = _df_col_or_zeros(df_short, "delta_r_raw")
            n_r_arr = _df_col_or_constant(df_short, "N_r", 0.0)
            df_short["delta_r_shrunk"] = (
                delta_r_raw_arr * (n_r_arr / (n_r_arr + 500.0))
            ).astype(np.float32)
        if "S_r" not in df_short.columns:
            pos_frac = _df_col_or_constant(df_short, "positive_fold_fraction_r", 0.5)
            df_short["S_r"] = np.clip(pos_frac, 0.1, 1.0).astype(np.float32)

        df_short["D_r"] = 0.25 * (
            0.35 * _zscore_np(_df_col_or_zeros(df_short, "impulse_shape_dispersion"))
            + 0.35 * _zscore_np(
                _df_col_or_zeros(df_short, "post_event_vol_dispersion")
            )
            + 0.15 * _zscore_np(
                _df_col_or_zeros(df_short, "fold_continuation_rate_std")
            )
            + 0.15 * _zscore_np(_df_col_or_zeros(df_short, "fold_event_count_std"))
        )
        df_short["score_r"] = (
            df_short["delta_r_shrunk"].astype(np.float32).values
            * np.sqrt(np.maximum(df_short["N_r"].astype(np.float32).values, 0.0))
            * np.maximum(df_short["S_r"].astype(np.float32).values, 0.0)
            / (1.0 + df_short["D_r"].astype(np.float32).values)
        ).astype(np.float32)
        cond_feature_gain: List[np.float32] = []
        cond_feature_pos: List[np.float32] = []
        cond_pred_gain: List[np.float32] = []
        cond_pred_pos: List[np.float32] = []
        cond_pred_regime_r2: List[np.float32] = []
        cond_pred_base_r2: List[np.float32] = []
        cond_spread_vals: List[np.float32] = []
        cond_econ: List[np.float32] = []
        cond_cov: List[np.float32] = []
        cond_fixed_cov: List[np.float32] = []
        cond_tbm_auc_regime: List[np.float32] = []
        cond_tbm_auc_base: List[np.float32] = []
        cond_tbm_auc_lift: List[np.float32] = []
        cond_tbm_top_lift: List[np.float32] = []
        cond_tbm_pos: List[np.float32] = []
        cond_tbm_stability: List[np.float32] = []
        cond_tbm_geom_name: List[str] = []
        for _, row in df_short.iterrows():
            name = str(row["name"])
            masks = candidate_masks[name]
            side_mask = masks.get("side_mask")
            if side_mask is None:
                side_mask = _get_side_mask(mode, masks["m_high"], masks["m_low"])
                masks["side_mask"] = side_mask
            stage_metrics = _compute_cached_stage_metrics(name, side_mask)
            cond_feature_gain.append(
                np.float32(stage_metrics["feature_learnability_gain"])
            )
            cond_feature_pos.append(
                np.float32(stage_metrics["feature_positive_fold_fraction"])
            )
            cond_pred_gain.append(
                np.float32(stage_metrics["conditional_predictability_gain"])
            )
            cond_pred_pos.append(
                np.float32(
                    stage_metrics[
                        "conditional_predictability_positive_fold_fraction"
                    ]
                )
            )
            cond_pred_regime_r2.append(
                np.float32(stage_metrics["conditional_predictability_regime_r2"])
            )
            cond_pred_base_r2.append(
                np.float32(stage_metrics["conditional_predictability_baseline_r2"])
            )
            cond_spread_vals.append(
                np.float32(stage_metrics["feature_conditioned_spread"])
            )
            cond_econ.append(np.float32(stage_metrics["economic_gain_r"]))
            cond_cov.append(
                np.float32(stage_metrics["geometry_weighted_mfe_coverage"])
            )
            cond_fixed_cov.append(
                np.float32(stage_metrics["fixed_tp_mfe_coverage"])
            )
            cond_tbm_auc_regime.append(
                np.float32(stage_metrics["tbm_lgbm_auc_regime"])
            )
            cond_tbm_auc_base.append(
                np.float32(stage_metrics["tbm_lgbm_auc_baseline"])
            )
            cond_tbm_auc_lift.append(
                np.float32(stage_metrics["tbm_lgbm_auc_lift_vs_baseline"])
            )
            cond_tbm_top_lift.append(
                np.float32(stage_metrics["tbm_lgbm_top_bucket_lift_vs_baseline"])
            )
            cond_tbm_pos.append(
                np.float32(stage_metrics["tbm_lgbm_positive_fold_fraction"])
            )
            cond_tbm_stability.append(
                np.float32(stage_metrics["tbm_lgbm_stability"])
            )
            cond_tbm_geom_name.append(
                str(stage_metrics["tbm_lgbm_selected_geometry"])
            )
        cond_tbm_invalid_regime: List[str] = []
        cond_tbm_invalid_base: List[str] = []
        cond_tbm_invalid_full: List[str] = []
        for _, row in df_short.iterrows():
            name = str(row["name"])
            side_mask = candidate_masks[name]["side_mask"]
            stage_metrics = _compute_cached_stage_metrics(name, side_mask)
            cond_tbm_invalid_regime.append(
                str(stage_metrics.get("tbm_lgbm_invalid_reason_regime", ""))
            )
            cond_tbm_invalid_base.append(
                str(stage_metrics.get("tbm_lgbm_invalid_reason_baseline", ""))
            )
            cond_tbm_invalid_full.append(
                str(stage_metrics.get("tbm_lgbm_invalid_reason_full", ""))
            )

        df_short["feature_learnability_gain"] = np.asarray(
            cond_feature_gain, dtype=np.float32
        )
        df_short["feature_positive_fold_fraction"] = np.asarray(
            cond_feature_pos, dtype=np.float32
        )
        df_short["conditional_predictability_gain"] = np.asarray(
            cond_pred_gain, dtype=np.float32
        )
        df_short["conditional_predictability_positive_fold_fraction"] = np.asarray(
            cond_pred_pos, dtype=np.float32
        )
        df_short["conditional_predictability_regime_r2"] = np.asarray(
            cond_pred_regime_r2, dtype=np.float32
        )
        df_short["conditional_predictability_baseline_r2"] = np.asarray(
            cond_pred_base_r2, dtype=np.float32
        )
        df_short["feature_conditioned_spread"] = np.asarray(
            cond_spread_vals, dtype=np.float32
        )
        df_short["economic_gain_r"] = np.asarray(cond_econ, dtype=np.float32)
        df_short["geometry_weighted_mfe_coverage"] = np.asarray(
            cond_cov, dtype=np.float32
        )
        df_short["fixed_tp_mfe_coverage"] = np.asarray(cond_fixed_cov, dtype=np.float32)
        df_short["aggregate_mfe_coverage"] = df_short[
            "geometry_weighted_mfe_coverage"
        ].astype(np.float32)
        df_short["tbm_lgbm_auc_regime"] = np.asarray(
            cond_tbm_auc_regime, dtype=np.float32
        )
        df_short["tbm_lgbm_auc_baseline"] = np.asarray(
            cond_tbm_auc_base, dtype=np.float32
        )
        df_short["tbm_lgbm_auc_lift_vs_baseline"] = np.asarray(
            cond_tbm_auc_lift, dtype=np.float32
        )
        df_short["tbm_lgbm_top_bucket_lift_vs_baseline"] = np.asarray(
            cond_tbm_top_lift, dtype=np.float32
        )
        df_short["tbm_lgbm_positive_fold_fraction"] = np.asarray(
            cond_tbm_pos, dtype=np.float32
        )
        df_short["tbm_lgbm_stability"] = np.asarray(
            cond_tbm_stability, dtype=np.float32
        )
        df_short["tbm_lgbm_selected_geometry"] = cond_tbm_geom_name
        df_short["tbm_lgbm_invalid_reason_regime"] = cond_tbm_invalid_regime
        df_short["tbm_lgbm_invalid_reason_baseline"] = cond_tbm_invalid_base
        df_short["tbm_lgbm_invalid_reason_full"] = cond_tbm_invalid_full
        df_short["incremental_information_delta_auc"] = df_short["tbm_lgbm_auc_lift_vs_baseline"].astype(np.float32)
        df_short["incremental_information_positive_fold_fraction"] = df_short["tbm_lgbm_positive_fold_fraction"].astype(np.float32)
        coverage_ref = float(max(cfg.get("mask_opt_min_mfe_coverage", 0.02), 0.25))
        df_short["coverage_multiplier"] = (
            0.25
            + 0.75
            * np.clip(
                df_short["aggregate_mfe_coverage"].astype(np.float32).values
                / max(coverage_ref, 1e-6),
                0.0,
                1.0,
            )
        ).astype(np.float32)
        df_short["learnability_support"] = np.maximum(
            df_short["incremental_information_delta_auc"].astype(np.float32).values,
            0.0,
        ).astype(np.float32)
        df_short["noise_penalty"] = (
            1.0
            + 0.25 * np.log1p(
                np.maximum(
                    np.nan_to_num(
                        df_short["dispersion_to_edge_ratio"].astype(np.float32).values,
                        nan=100.0,
                    ),
                    0.0,
                )
            )
        ).astype(np.float32)
        df_short["effective_edge"] = (
            np.maximum(df_short["delta_r_shrunk"].astype(np.float32).values, 0.0)
            * np.maximum(df_short["S_r"].astype(np.float32).values, 0.0)
            * _df_col_or_constant(df_short, "primary_multiplier", 1.0)
            * _df_col_or_constant(df_short, "worst_fold_multiplier", 1.0)
            * _df_col_or_constant(df_short, "disagreement_penalty", 1.0)
        ).astype(np.float32)
        df_short["score_r"] = (
            df_short["effective_edge"].astype(np.float32).values
            * (1.0 + 25.0 * df_short["learnability_support"].astype(np.float32).values)
            / np.maximum(df_short["noise_penalty"].astype(np.float32).values, 1e-6)
        ).astype(np.float32)
        df_short["predictability_anchor"] = np.maximum(
            df_short["conditional_predictability_gain"].astype(np.float32).values, 0.0
        ).astype(np.float32)
        df_short["predictability_positive_multiplier"] = (
            0.75
            + 0.25
            * df_short[
                "conditional_predictability_positive_fold_fraction"
            ].astype(np.float32).values
        ).astype(np.float32)
        df_short["spread_multiplier"] = (
            1.0
            + np.tanh(10.0 * df_short["feature_conditioned_spread"].astype(np.float32).values)
        ).astype(np.float32)
        df_short["difference_prior"] = (
            0.85 + 0.15 * np.clip(df_short["score_r"].astype(np.float32).values, 0.0, None)
        ).astype(np.float32)
        df_short["score_ml"] = (
            np.nan_to_num(df_short["score_r"].astype(np.float32).values, nan=0.0)
            * (
                1.0
                + 5.0
                * np.nan_to_num(
                    df_short["predictability_anchor"].astype(np.float32).values, nan=0.0
                )
            )
            * np.nan_to_num(
                df_short["predictability_positive_multiplier"].astype(np.float32).values,
                nan=1.0,
            )
            * np.nan_to_num(df_short["spread_multiplier"].astype(np.float32).values, nan=1.0)
            * np.nan_to_num(df_short["difference_prior"].astype(np.float32).values, nan=1.0)
        ).astype(np.float32)
        df_short["tbm_auc_support"] = np.maximum(
            df_short["tbm_lgbm_auc_lift_vs_baseline"].astype(np.float32).values, 0.0
        ).astype(np.float32)
        df_short["tbm_lift_support"] = np.maximum(
            df_short["tbm_lgbm_top_bucket_lift_vs_baseline"].astype(np.float32).values,
            0.0,
        ).astype(np.float32)
        phase4_valid = (
            np.isfinite(df_short["score_ml"].astype(np.float32).values)
            & np.isfinite(df_short["feature_learnability_gain"].astype(np.float32).values)
            & np.isfinite(df_short["tbm_lgbm_auc_lift_vs_baseline"].astype(np.float32).values)
            & np.isfinite(df_short["tbm_lgbm_top_bucket_lift_vs_baseline"].astype(np.float32).values)
            & np.isfinite(df_short["tbm_lgbm_stability"].astype(np.float32).values)
            & np.isfinite(df_short["tbm_lgbm_positive_fold_fraction"].astype(np.float32).values)
            & np.isfinite(df_short["coverage_multiplier"].astype(np.float32).values)
        )
        if not np.all(phase4_valid):
            invalid_rows = df_short.loc[~phase4_valid, [
                "name",
                "tbm_lgbm_invalid_reason_regime",
                "tbm_lgbm_invalid_reason_baseline",
                "tbm_lgbm_invalid_reason_full",
            ]].copy()
            tprint(
                f"{_stage_label(6)} ({mode}): removing {invalid_rows.shape[0]} candidate(s) with invalid Ridge/LGBM diagnostics."
            )
            for _, invalid_row in invalid_rows.iterrows():
                tprint(
                    f"  {_stage_label(6)} invalid candidate "
                    f"name={invalid_row['name']} "
                    f"regime_reason={invalid_row['tbm_lgbm_invalid_reason_regime']} "
                    f"baseline_reason={invalid_row['tbm_lgbm_invalid_reason_baseline']} "
                    f"full_reason={invalid_row['tbm_lgbm_invalid_reason_full']}"
                )
            df_short = df_short.loc[phase4_valid].copy()
        if df_short.empty:
            tprint(f"{_stage_label(6)} ({mode}): no candidates with valid Ridge/LGBM diagnostics.")
            return {
                "status": "failed",
                "reason": f"phase4_invalid_ml_diagnostics_{mode}",
                "layer0_candidate_table_": df2,
                "phase25_regime_impact_": phase25_seeds_df,
            }
        df_short["score_ml_trading"] = (
            df_short["score_ml"].astype(np.float32).values
            * (
                1.0
                + 2.0
                * np.maximum(
                    df_short["feature_learnability_gain"].astype(np.float32).values, 0.0
                )
            )
            * (1.0 + 25.0 * df_short["tbm_auc_support"].astype(np.float32).values)
            * (1.0 + 10.0 * df_short["tbm_lift_support"].astype(np.float32).values)
            * df_short["tbm_lgbm_stability"].astype(np.float32).values
            * df_short["tbm_lgbm_positive_fold_fraction"].astype(np.float32).values
            * df_short["coverage_multiplier"].astype(np.float32).values
        ).astype(np.float32)
        df_short["shortlist_score"] = df_short["score_ml_trading"].astype(np.float32)
        # Apply complexity penalties
        phase4_single_regime_penalty = float(cfg.get("phase4_single_regime_penalty", 0.95))
        phase4_two_regime_penalty = float(cfg.get("phase4_two_regime_penalty", 0.85))

        penalties = []
        for _, row in df_short.iterrows():
            tier = row.get("tier", 0)
            if tier == 1:
                penalties.append(phase4_single_regime_penalty)
            elif tier == 2:
                penalties.append(phase4_two_regime_penalty)
            else:
                penalties.append(1.0)

        df_short["complexity_multiplier"] = np.array(penalties, dtype=np.float32)
        df_short["score_ml_trading"] = df_short["score_ml_trading"].astype(np.float32).values * df_short["complexity_multiplier"].astype(np.float32).values
        df_short["shortlist_score"] = df_short["score_ml_trading"].astype(np.float32)

        # Dominance Pruning
        base_rows = df_short[df_short["tier"] == 0].copy()

        keep_idx: List[int] = []
        tolerance = float(cfg.get("phase4_dominance_tolerance", 0.90))

        # We process each candidate and see if a simpler candidate strictly dominates it
        for idx, row in df_short.iterrows():
            tier = row.get("tier", 0)
            if tier == 0:
                keep_idx.append(idx)
                continue

            base_name = str(row["name"]).split("_" + row["conditioner_mode"].replace(" ", "").replace(">", "gt").replace("<", "lt"))[0]
            if "AND" in str(row["name"]):
                base_name = str(row["name"]).split("_AND_")[0].rsplit("_", 1)[0]

            dominated = False
            # Compare against all simpler candidates of the same base
            simpler_cands = df_short[(df_short["tier"] < tier)]

            for _, s_row in simpler_cands.iterrows():
                # Rough check if they share the same base name (ignoring conditioner suffixes)
                if not str(s_row["name"]).startswith(base_name):
                    continue

                # A dominates B if:
                if (
                    _metric_or_nan(s_row.get("score_ml_trading")) >= _metric_or_nan(row.get("score_ml_trading")) and
                    _metric_or_nan(s_row.get("economic_gain_r")) >= _metric_or_nan(row.get("economic_gain_r")) and
                    _metric_or_nan(s_row.get("S_r")) >= _metric_or_nan(row.get("S_r")) and
                    _metric_or_nan(s_row.get("total_events")) >= _metric_or_nan(row.get("total_events")) * tolerance
                ):
                    dominated = True
                    break

            if not dominated:
                keep_idx.append(idx)

        df_short = (
            df_short.loc[keep_idx]
            .sort_values("score_ml_trading", ascending=False)
            .copy()
        )

    _tprint_candidate_table_support_summary(f"{_stage_label(6)} shortlist", mode, df_short)

    _log_stage_snapshot(
        mode,
        _stage_label(6),
        df_short,
        "score_ml_trading",
        [
            "name",
            "score_ml_trading",
            "tbm_lgbm_auc_lift_vs_baseline",
            "tbm_lgbm_stability",
            "tbm_lgbm_positive_fold_fraction",
            "tbm_lgbm_top_bucket_lift_vs_baseline",
            "tbm_lgbm_selected_geometry",
            "aggregate_mfe_coverage",
            "score_ml",
            "delta_r_raw",
        ],
    )

    final_diag_k = int(cfg.get("final_top_k_for_diagnostics", 4))

    # Jaccard diversity selection
    df_short = df_short.sort_values("shortlist_score", ascending=False).reset_index(drop=True)
    selected_idx = []
    selected_masks = []

    for idx, row in df_short.iterrows():
        if len(selected_idx) >= final_diag_k:
            break

        m_info = candidate_masks.get(str(row["name"]), {})
        if not m_info:
            continue

        side_mask = m_info.get("side_mask")
        if side_mask is None:
            side_mask = _get_side_mask(
                mode,
                m_info.get("m_high", np.array([])),
                m_info.get("m_low", np.array([])),
            )
            m_info["side_mask"] = side_mask

        # Check Jaccard similarity against already selected
        is_diverse = True
        for sel_mask in selected_masks:
            intersection = np.sum(side_mask & sel_mask)
            union = np.sum(side_mask | sel_mask)
            jaccard = intersection / max(union, 1)
            if jaccard > 0.90:  # Loosened from 0.80 to allow basket size 4
                is_diverse = False
                break

        if is_diverse:
            selected_idx.append(idx)
            selected_masks.append(side_mask)

    if not selected_idx and not df_short.empty:
        selected_idx = [0]

    tprint(f"{_stage_label(6)} ({mode}): computing final diagnostics for {len(selected_idx)} selected candidates...")
    df_diag_input = df_short.loc[selected_idx].copy()
    _tprint_candidate_table_support_summary(f"{_stage_label(6)} selected", mode, df_diag_input)
    df_diag = _final_topk_diagnostics(
        mode, df_diag_input, candidate_masks, shared, feature_dict, cfg
    )

    # Instead of a single 'best', we provide a 'basket' of diverse top performers
    basket = df_short.loc[selected_idx].to_dict(orient="records")

    regime_impact_df = pd.concat(regime_impact_rows, ignore_index=True) if regime_impact_rows else pd.DataFrame()
    _save_stage_artifacts(
        cfg,
        mode,
        6,
        payload={
            "df2": df2.copy(),
            "df_short": df_short.copy(),
            "df_diag": df_diag.copy(),
            "basket": basket,
            "candidate_masks": candidate_masks,
            "regime_impact_df": regime_impact_df.copy(),
            "trigger_all_df": trigger_all_df.copy() if isinstance(trigger_all_df, pd.DataFrame) else pd.DataFrame(),
            "trigger_diagnostics": trigger_diagnostics,
        },
        tables={
            "candidates": df2.copy(),
            "shortlist": df_short.copy(),
            "diagnostics": df_diag.copy(),
            "regime_impact": regime_impact_df.copy(),
        },
    )

    tprint(f"Mode {mode} completed: {len(basket)} candidates selected for basket")

    return {
        "status": "ok",
        "mode": mode,
        "phase1_candidate_table_": df1,
        "layer0_candidate_table_": df2.copy(),
        "layer0_shortlist_": df_short,
        "layer0_basket_": basket,
        "layer0_candidate_masks_": candidate_masks,
        "final_topk_diagnostics_table_": df_diag,
        "regime_impact_table_": regime_impact_df,
        "trigger_candidate_table_": trigger_all_df,
        "trigger_diagnostics_": trigger_diagnostics,
    }


def _mode_worker(
    conn: Any,
    mode: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
) -> None:
    try:
        shared = _build_shared_cache(data, feature_dict, forward_returns, cfg)
        res = _run_mode_search(mode, shared, feature_dict, cfg)
        conn.send(("ok", res))
    except Exception:
        err_msg = traceback.format_exc()
        print(f"Exception in worker for mode {mode}:\n{err_msg}")
        conn.send(("error", err_msg))
    finally:
        conn.close()


def _run_mode_search_isolated(
    mode: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_mode_worker,
        args=(child_conn, mode, data, feature_dict, forward_returns, cfg),
    )
    proc.start()
    child_conn.close()
    payload: Optional[Tuple[str, Any]] = None
    timeout_seconds = float(cfg.get("mask_opt_mode_timeout_seconds", 0.0))
    if timeout_seconds > 0:
        if parent_conn.poll(timeout_seconds):
            payload = parent_conn.recv()
    else:
        payload = parent_conn.recv()
    if payload is None:
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return {"status": "failed", "reason": f"mode_timeout_{mode}"}
        return {
            "status": "failed",
            "reason": f"mode_crashed_{mode}_exit_{proc.exitcode}",
        }
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    status, body = payload
    if status == "ok":
        return body
    return {"status": "failed", "reason": f"mode_exception_{mode}", "traceback": body}


# =============================================================================
# PUBLIC ORCHESTRATOR
# =============================================================================


def optimize_layer0_masks_by_mode(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
    modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return optimize_layer_masks_by_mode(
        data, feature_dict, forward_returns, cfg, modes=modes, layer_name="layer0"
    )


def optimize_layer_masks_by_mode(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any],
    modes: Optional[List[str]] = None,
    layer_name: str = "layer0",
) -> Dict[str, Any]:
    tprint(f"="*80)
    tprint(f"LAYER {layer_name.upper()} OPTIMIZATION: {modes}")
    tprint(f"="*80)
    
    if modes is None:
        modes = ALL_MODES[:]

    runtime_cfg = _materialize_layer_runtime_cfg(cfg, layer_name)
    
    tprint("Capping rows for optimization...")
    rows_before_cap = int(data.shape[0])
    data, feature_dict, forward_returns = _cap_rows_for_optimization(
        data=data,
        feature_dict=feature_dict,
        forward_returns=np.asarray(forward_returns, dtype=np.float32),
        cfg=runtime_cfg,
        seed=42,
    )
    tprint(f"Data shape after capping: {data.shape[0]} rows")
    _tprint_retention_step(
        "Optimization cap",
        int(data.shape[0]),
        int(runtime_cfg.get("mask_opt_full_panel_rows", data.shape[0])),
        prev_rows=rows_before_cap,
    )
    
    runtime_cfg = _rescale_mode_gates_for_sample_size(runtime_cfg, int(data.shape[0]))

    mode_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    isolate_modes = (
        bool(runtime_cfg.get("mask_opt_isolate_modes", True)) and len(modes) > 1
    )
    shared: Optional[Dict[str, Any]] = None
    if not isolate_modes:
        tprint("Building shared cache (non-isolated mode)...")
        shared = _build_shared_cache(data, feature_dict, forward_returns, runtime_cfg)
        tprint("Shared cache built successfully")

    for i, mode in enumerate(modes):
        tprint(f"\n{'='*80}")
        tprint(f"Processing mode {i+1}/{len(modes)}: {mode}")
        tprint(f"{'='*80}\n")
        if isolate_modes:
            tprint(f"Running mode {mode} in isolated subprocess...")
            res = _run_mode_search_isolated(
                mode, data, feature_dict, forward_returns, runtime_cfg
            )
        else:
            assert shared is not None
            res = _run_mode_search(mode, shared, feature_dict, runtime_cfg)
        mode_results[mode] = res

        if res.get("status") == "ok":
            tprint(f"Mode {mode} completed successfully")
            basket = res.get("layer0_basket_", [])
            primary_col = _mode_primary_predictability_col(mode)
            
            for rank, member in enumerate(basket):
                shortlist_score = _metric_or_nan(
                    member.get("shortlist_score", member.get("score_ml_trading", member.get("score_r")))
                )
                summary_rows.append(
                    {
                        "mode": mode,
                        "rank": rank + 1,
                        "status": "ok",
                        "name": str(member.get("name", "")),
                        "candidate_count": len(res["layer0_candidate_table_"]),
                        "shortlist_count": len(res["layer0_shortlist_"]),
                        "best_shortlist_score": shortlist_score,
                        "score_r": float(member.get("score_r", member.get("shortlist_score", 0.0))),
                        "delta_r": _metric_or_nan(member.get("delta_r")),
                        "S_r": _metric_or_nan(member.get("S_r")),
                        "D_r": _metric_or_nan(member.get("D_r")),
                        "event_count": int(member.get("total_events", 0)),
                        "primary_gain": _metric_or_nan(member.get(primary_col)),
                        "incremental_information_delta_auc": _metric_or_nan(member.get("incremental_information_delta_auc")),
                        "incremental_information_positive_fold_fraction": _metric_or_nan(member.get("incremental_information_positive_fold_fraction")),
                        "dispersion_to_edge_ratio": _metric_or_nan(member.get("dispersion_to_edge_ratio")),
                    }
                )
        else:
            tprint(f"Mode {mode} failed: {res.get('reason', 'unknown')}")
            summary_rows.append(
                {
                    "mode": mode,
                    "status": res.get("reason", "failed"),
                    "best_name": "",
                    "candidate_count": 0,
                    "shortlist_count": 0,
                    "best_shortlist_score": 0.0,
                    "score_r": float("nan"),
                    "delta_r": float("nan"),
                    "N_r": 0.0,
                    "S_r": float("nan"),
                    "D_r": float("nan"),
                    "event_count": 0,
                    "active_days_fraction": 0.0,
                    "events_per_day_mean": float("nan"),
                    "events_per_day_std": float("nan"),
                    "events_per_day_per_asset": float("nan"),
                    "primary_gain": 0.0,
                    "primary_gain_is_nan": float("nan"),
                    "incremental_information_delta_auc": float("nan"),
                    "incremental_information_positive_fold_fraction": float("nan"),
                    "dispersion_to_edge_ratio": float("nan"),
                    "selected_delta_metric": "",
                    "decision": "failed",
                    "rationale": "",
                }
            )
    
    tprint(f"\n{'='*80}")
    tprint(f"LAYER {layer_name.upper()} OPTIMIZATION COMPLETE")
    tprint(f"{'='*80}\n")

    return {
        "status": "ok",
        "mode_results": mode_results,
        "mode_summary_table_": pd.DataFrame(summary_rows),
    }


# =============================================================================
# CLI
# =============================================================================


def run_mask_optimization_4modes(args: argparse.Namespace) -> None:
    from copy import deepcopy

    tprint("="*80)
    tprint("MASK OPTIMIZATION 4-MODES: STARTING")
    tprint("="*80)
    
    cfg = deepcopy(CFG)
    cfg["fast"] = bool(getattr(args, "fast", 0))

    if getattr(args, "subset", None) is not None:
        cfg["mask_opt_subset_fraction"] = args.subset
    if getattr(args, "lookback_h", None) is not None:
        cfg["lookback_h_override"] = args.lookback_h
        cfg["z_hours_grid"] = [args.lookback_h]
    if getattr(args, "min_events", None) is not None:
        cfg["mask_opt_min_events_floor"] = args.min_events
    if getattr(args, "support", None) is not None:
        cfg["mask_opt_min_active_days_floor"] = args.support
    if getattr(args, "shortlist", None) is not None:
        cfg["shortlist_max_candidates"] = args.shortlist
    if getattr(args, "parallel", None) is not None:
        cfg["mask_opt_parallel_workers"] = args.parallel
    if getattr(args, "z_start", None) is not None or getattr(args, "z_step", None) is not None:
        z_start = getattr(args, "z_start", 0) or 0
        z_step = getattr(args, "z_step", 1) or 1
        # Apply slicing to z_hours_grid if applicable
        current_grid = cfg.get("z_hours_grid", [4, 6, 8, 10])
        cfg["z_hours_grid"] = current_grid[z_start::z_step]

    # defaults aligned with requested optimization spec
    cfg["z_hours_grid"] = cfg.get("z_hours_grid", [4, 6, 8, 10])
    cfg["duration_grid"] = [1, 2, 3]
    cfg["x_std_grid"] = [1.4, 1.5, 1.6]
    cfg["y_move_pct_grid"] = [4.0, 5.0, 6.0, 7.0]
    cfg["std_plus_abs_std_grid"] = [1.4, 1.5, 1.6]
    cfg["std_plus_abs_abs_grid"] = [4.0, 5.0, 6.0]
    cfg["phase1_min_total_events"] = 5000
    cfg["phase2_min_total_events"] = 5000
    cfg["phase1_min_active_days_fraction"] = 0.80
    cfg["phase2_min_active_days_fraction"] = 0.80
    cfg["mask_opt_target_event_density"] = 0.012
    cfg["mask_opt_min_events_floor"] = 150
    cfg["mask_opt_min_active_days_floor"] = 0.25
    cfg["mask_opt_min_mfe_coverage"] = float(cfg.get("mask_opt_min_mfe_coverage", 0.02))
    cfg["mask_opt_min_primary_gain"] = float(cfg.get("mask_opt_min_primary_gain", 0.005))
    cfg["mask_opt_max_dispersion_to_edge_ratio"] = float(
        cfg.get("mask_opt_max_dispersion_to_edge_ratio", 20.0)
    )
    cfg["min_positive_fold_fraction"] = 0.60
    cfg["final_top_k_for_diagnostics"] = 6
    cfg["mask_opt_max_rows"] = 300_000        # Broader late-stage universe
    cfg["mask_opt_deep_rows"] = 900_000      # Broader SlicePlanner universe
    cfg["phase4_tbm_lgbm_max_subset"] = int(max(cfg.get("phase4_tbm_lgbm_max_subset", 100_000), 100_000))
    cfg["top_k_for_learnability"] = int(max(cfg.get("top_k_for_learnability", 48), 48))
    cfg["phase1_min_representatives_per_family"] = int(
        max(cfg.get("phase1_min_representatives_per_family", 2), 2)
    )
    cfg["phase2_min_representatives_per_family"] = int(
        max(cfg.get("phase2_min_representatives_per_family", 2), 2)
    )
    cfg["phase2_max_candidates_per_family"] = int(
        max(cfg.get("phase2_max_candidates_per_family", 3), 3)
    )
    cfg["mask_opt_min_slice_full_panel_fraction"] = float(
        max(cfg.get("mask_opt_min_slice_full_panel_fraction", 0.04), 0.04)
    )
    cfg["mask_opt_min_cap_full_panel_fraction"] = float(
        max(cfg.get("mask_opt_min_cap_full_panel_fraction", 0.04), 0.04)
    )
    cfg["phase1_min_full_panel_fraction"] = float(
        max(cfg.get("phase1_min_full_panel_fraction", 0.04), 0.04)
    )
    cfg["mask_opt_pre_slice_max_rows"] = int(
        cfg.get("mask_opt_pre_slice_max_rows", 1_000_000)
    )
    cfg["phase1_classifier_max_samples_per_class"] = int(
        cfg.get("phase1_classifier_max_samples_per_class", 15_000)
    )
    cfg["phase2_metric_max_samples_per_class"] = int(
        cfg.get("phase2_metric_max_samples_per_class", 25_000)
    )

    if args.data_root:
        cfg["data_root"] = _resolve_path(args.data_root)
    else:
        cfg["data_root"] = _resolve_path(cfg.get("data_root", "data"))
    
    tprint(f"Configuration loaded, data_root={cfg['data_root']}")

    if args.perps:
        cfg["use_perps"] = True
        if not cfg["data_root"].endswith("_perp"):
            cfg["data_root"] += "_perp"
        cfg = enable_perp_feature_keys(cfg)

    if args.features:
        feature_path = _resolve_path(args.features)
    else:
        feature_path = _find_latest_feature_dir(cfg["data_root"])

    if not feature_path:
        tprint(f"ERROR: no features found in {cfg['data_root']}/features")
        return
    
    tprint(f"Feature directory: {feature_path}")
    feature_run_id = os.path.basename(feature_path)
    reports_root = Path(str(CFG.get("reports_root", "reports")))
    artifact_root_parent = reports_root / "mask_optimiser_stage_runs" / feature_run_id
    resume_artifact_dir = getattr(args, "resume_artifact_dir", None)
    if resume_artifact_dir is None and int(getattr(args, "start_stage", 1) or 1) > 1 and artifact_root_parent.exists():
        candidates = [p for p in artifact_root_parent.iterdir() if p.is_dir()]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            resume_artifact_dir = str(candidates[0])
    run_id = str(getattr(args, "run_id", None) or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S"))
    artifact_base = Path(str(resume_artifact_dir or (artifact_root_parent / run_id)))
    if resume_artifact_dir is None:
        artifact_base.mkdir(parents=True, exist_ok=True)
    cfg["mask_opt_stage_artifact_dir"] = str(artifact_base)
    cfg["mask_opt_start_stage"] = int(getattr(args, "start_stage", 1) or 1)
    cfg["mask_opt_stop_stage"] = int(getattr(args, "stop_stage", 6) or 6)
    tprint(
        f"Stage artifacts: dir={cfg['mask_opt_stage_artifact_dir']} | "
        f"start_stage={cfg['mask_opt_start_stage']} | stop_stage={cfg['mask_opt_stop_stage']}"
    )

    tprint(f"Loading data: data_root={cfg['data_root']} | features={feature_path}")
    
    store = PartitionedOHLCVStore(
        root_dir=cfg["data_root"], timeframe=cfg.get("timeframe", "1h")
    )

    ohlcv_dir = os.path.join(cfg["data_root"], "ohlcv")
    all_symbols = []
    for path in glob.glob(os.path.join(ohlcv_dir, "symbol=*")):
        base = os.path.basename(path)
        if base.startswith("symbol="):
            raw = base.replace("symbol=", "")
            all_symbols.append(raw.replace("_", "/", 1))
    all_symbols.sort()
    
    tprint(f"Found {len(all_symbols)} symbols in OHLCV directory")

    # Apply deduplication (only use symbols passed on by universe)
    all_symbols = _dedup_universe_by_base(all_symbols)
    tprint(f"Symbols after deduplication: {len(all_symbols)}")

    symbols = list(all_symbols)
    if args.max_symbols is not None:
        symbols = symbols[: max(1, int(args.max_symbols))]
        tprint(f"Selected {len(symbols)} symbols via --max-symbols")
    else:
        tprint(f"Selected {len(symbols)} deduplicated symbols")

    start_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
        days=int(365.25 * args.lookback_years)
    )
    
    tprint(f"Loading OHLCV data from {start_ts}...")

    dfs_by_symbol: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = store.load(s, start_ts=start_ts)
        if not df.empty:
            dfs_by_symbol[s] = df
    
    tprint(f"Loaded data for {len(dfs_by_symbol)}/{len(symbols)} symbols")

    if not dfs_by_symbol:
        tprint("ERROR: no symbol data loaded")
        return

    tprint("Building panel from loaded data...")
    panel = to_panel(dfs_by_symbol)
    if not panel or "close" not in panel or panel["close"].empty:
        tprint("ERROR: panel empty or missing close")
        return
    if "open" not in panel:
        tprint("WARNING: 'open' not in panel, attempting to backfill from high/low/close mean")
        panel["open"] = (panel["high"] + panel["low"] + panel["close"]) / 3.0

    ts_str = os.path.basename(feature_path)
    try:
        ts = pd.Timestamp(ts_str.replace("_", " "))
    except Exception:
        ts = pd.Timestamp.now(tz="UTC")
    data_root_dir = os.path.dirname(os.path.dirname(feature_path))

    tprint(f"Loading features from {data_root_dir}...")
    feat_dict_raw = load_features_selected(
        ts=ts,
        root_dir=data_root_dir,
        feature_keys=(
            list(_required_feature_keys())
            + RIDGE_FEATURE_COLS
            + list(LOCATION_FILTER_COLUMNS)
            + list(INTRADAY_TRIGGER_COLUMNS)
        ),
        symbols=symbols,
        start_ts=start_ts,
    )
    if not feat_dict_raw:
        tprint("ERROR: empty feature dictionary")
        return
    
    tprint(f"Loaded {len(feat_dict_raw)} features")

    common_idx = panel["close"].index
    common_syms = panel["close"].columns

    fwd_hours = int(cfg.get("mask_opt_forward_hours", 12))
    fwd_ret_wide = (
        panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)
    )

    n_timestamps = len(common_idx)
    n_symbols = len(common_syms)
    data_stacked = pd.DataFrame(
        {
            "timestamp": np.repeat(common_idx.to_numpy(), n_symbols),
            "symbol": np.tile(common_syms.to_numpy(dtype=object), n_timestamps),
            "open": _flatten_wide_frame(panel["open"], common_idx, common_syms),
            "close": _flatten_wide_frame(panel["close"], common_idx, common_syms),
            "high": _flatten_wide_frame(panel["high"], common_idx, common_syms),
            "low": _flatten_wide_frame(panel["low"], common_idx, common_syms),
        }
    )

    feature_dict: Dict[str, np.ndarray] = {}
    for k, df in feat_dict_raw.items():
        if isinstance(df, pd.DataFrame):
            arr = _flatten_wide_frame(df, common_idx, common_syms)
            arr[np.isinf(arr)] = np.nan
            feature_dict[k] = arr.astype(np.float32)
    
    tprint(f"Processed {len(feature_dict)} features into stacked arrays")

    fwd_ret_stacked = _flatten_wide_frame(fwd_ret_wide, common_idx, common_syms)

    full_panel_rows = int(data_stacked.shape[0])
    cfg["mask_opt_full_panel_rows"] = full_panel_rows
    tprint(f"Total rows available before SlicePlanner: {full_panel_rows}")
    _tprint_retention_step("Full panel", full_panel_rows, full_panel_rows)
    pre_slice_max_rows = int(cfg.get("mask_opt_pre_slice_max_rows", 1_000_000))
    if data_stacked.shape[0] > pre_slice_max_rows:
        start_idx = data_stacked.shape[0] - pre_slice_max_rows
        data_stacked = data_stacked.iloc[start_idx:].reset_index(drop=True)
        fwd_ret_stacked = fwd_ret_stacked[start_idx:]
        for k in feature_dict:
            feature_dict[k] = feature_dict[k][start_idx:]
        tprint(
            f"Capped pre-SlicePlanner rows to {data_stacked.shape[0]} for mask optimization."
        )
    _tprint_retention_step(
        "Pre-SlicePlanner sample",
        int(data_stacked.shape[0]),
        full_panel_rows,
        prev_rows=full_panel_rows,
    )
    # Apply regime_search slice plan on the full dataset FIRST.
    # This gives temporally-structured rows spanning full history (~150K rows).
    # Phase 1/2 then cap this structured sample to 50K / 20K respectively.
    # Ridge/LGBM Diagnostics use the full SlicePlanner result (up to mask_opt_deep_rows=150K).
    
    tprint("Applying regime_search slice plan...")
    deep_data, deep_feature_dict, deep_fwd_ret = _apply_regime_search_slice_plan(
        data=data_stacked,
        feature_dict=feature_dict,
        forward_returns=fwd_ret_stacked,
        lookback_years=float(args.lookback_years),
    )
    min_slice_rows = _min_rows_from_full_panel(
        cfg,
        full_panel_rows=full_panel_rows,
        fraction_key="mask_opt_min_slice_full_panel_fraction",
        default_fraction=0.04,
    )
    if deep_data.shape[0] < min_slice_rows:
        tprint(
            "SlicePlanner sample below minimum full-panel coverage; "
            f"raising retained rows to {min_slice_rows} "
            f"({_pct_str(min_slice_rows, full_panel_rows)} of full panel)."
        )
        slice_cfg = dict(cfg)
        slice_cfg["mask_opt_full_panel_rows"] = full_panel_rows
        slice_cfg["mask_opt_max_rows"] = int(max(min_slice_rows, cfg.get("mask_opt_max_rows", min_slice_rows)))
        slice_cfg["mask_opt_max_rows_pct"] = 0.0
        deep_data, deep_feature_dict, deep_fwd_ret = _cap_rows_for_optimization(
            data_stacked,
            feature_dict,
            fwd_ret_stacked,
            slice_cfg,
        )
    deep_rows = int(cfg.get("mask_opt_deep_rows", 150_000))
    if deep_data.shape[0] > deep_rows:
        deep_data = deep_data.iloc[-deep_rows:].reset_index(drop=True)
        deep_fwd_ret = deep_fwd_ret[-deep_rows:]
        for k in deep_feature_dict:
            deep_feature_dict[k] = deep_feature_dict[k][-deep_rows:]
    tprint(f"SlicePlanner gave {deep_data.shape[0]} structured rows for Ridge/LGBM Diagnostics.")
    _tprint_retention_step(
        "SlicePlanner output",
        int(deep_data.shape[0]),
        full_panel_rows,
        prev_rows=int(data_stacked.shape[0]),
    )

    if args.mode == "all":
        modes = ALL_MODES[:]
    else:
        modes = [args.mode]

    tprint(f"Optimization modes: {modes}")
    tprint(f"Starting 4-mode Layer 0 optimization (deep={deep_data.shape[0]}, cap={cfg.get('mask_opt_max_rows', 50_000)}, p1_floor={cfg.get('phase1_min_subsample_rows', 20_000)})...")
    result = optimize_layer0_masks_by_mode(
        deep_data, deep_feature_dict, deep_fwd_ret, cfg, modes=modes
    )

    if result.get("status") != "ok":
        tprint("Optimization failed.")
        return

    tprint("=" * 80)
    tprint("MODE SUMMARY")
    tprint("=" * 80)
    tprint(result["mode_summary_table_"].to_string(index=False))

    # optional save
    try:
        from extreme_price_movements.offline_optimisers.params_store import (
            INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV,
            REPORTS_DIR,
            save_best_params_csv,
        )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = REPORTS_DIR / "inference_candidate_mask_mode_summary.csv"
        result["mode_summary_table_"].to_csv(summary_path, index=False)
        
        bucket_winner_rows: List[Dict[str, Any]] = []
        regime_impact_dfs = []

        for mode in modes:
            mode_res = result["mode_results"].get(mode, {})
            candidate_table = mode_res.get("layer0_candidate_table_")
            shortlist_table = mode_res.get("layer0_shortlist_")
            basket = mode_res.get("layer0_basket_", [])
            impact_df = mode_res.get("regime_impact_table_")

            if isinstance(candidate_table, pd.DataFrame):
                candidate_table.to_csv(REPORTS_DIR / f"layer0_candidate_table_{mode}.csv", index=False)
            if isinstance(shortlist_table, pd.DataFrame):
                shortlist_table.to_csv(REPORTS_DIR / f"layer0_shortlist_{mode}.csv", index=False)
            if isinstance(impact_df, pd.DataFrame) and not impact_df.empty:
                impact_df["mode"] = mode
                regime_impact_dfs.append(impact_df)

            for member in basket:
                out = dict(member)
                out["mode"] = mode
                bucket_winner_rows.append(out)

        if bucket_winner_rows:
            winners_df = pd.DataFrame(bucket_winner_rows)
            # Sort by mode then score
            if "mode" in winners_df.columns:
                winners_df = winners_df.sort_values(["mode", "score_ml_trading"], ascending=[True, False])
            
            winners_df.to_csv(INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV, index=False)
            winners_df.to_csv(REPORTS_DIR / "inference_candidate_mask_best_params_per_bucket.csv", index=False)
        
        if regime_impact_dfs:
            combined_impact = pd.concat(regime_impact_dfs, ignore_index=True)
            impact_path = REPORTS_DIR / "regime_impact_report.csv"
            combined_impact.to_csv(impact_path, index=False)
            tprint(f"Saved regime impact report to {impact_path}")

            # ---------------------------------------------------------
            # Enhanced Winners Report (Detailed Metrics + Regime)
            # ---------------------------------------------------------
            if bucket_winner_rows:
                # bucket_winner_rows is already sorted by mode and score above
                winners_detailed = []
                for _, row in winners_df.iterrows():
                    w_dict = row.to_dict()
                    c_name = str(w_dict.get("name"))
                    c_mode = str(w_dict.get("mode"))
                    
                    # Filter impact for this candidate
                    c_impact = combined_impact[
                        (combined_impact["base_candidate"] == c_name) & 
                        (combined_impact["mode"] == c_mode)
                    ].copy()
                    
                    if not c_impact.empty:
                        c_impact["abs_coef"] = c_impact["coef"].abs()
                        top_regimes = c_impact.sort_values("abs_coef", ascending=False).head(3)
                        
                        regime_list = []
                        for _, r_row in top_regimes.iterrows():
                            sign = "+" if r_row["coef"] > 0 else "-"
                            regime_list.append(f"{r_row['feature']}({sign})")
                        
                        w_dict["top_regime_drivers"] = ", ".join(regime_list)
                    else:
                        w_dict["top_regime_drivers"] = "no_data"
                        
                    winners_detailed.append(w_dict)
                
                detailed_winners_df = pd.DataFrame(winners_detailed)
                # Sort columns to put economic metrics near the front for user visibility
                important_cols = [
                    "mode", "name", "score_ml_trading", "economic_gain_r", 
                    "tbm_lgbm_auc_lift_vs_baseline", "tbm_lgbm_top_bucket_lift_vs_baseline",
                    "tbm_lgbm_stability", "tbm_lgbm_positive_fold_fraction",
                    "total_events", "score_r", "delta_r_shrunk", "S_r", "D_r",
                    "top_regime_drivers", "rationale"
                ]
                existing_important = [c for c in important_cols if c in detailed_winners_df.columns]
                other_cols = [c for c in detailed_winners_df.columns if c not in existing_important]
                detailed_winners_df = detailed_winners_df[existing_important + other_cols]
                
                detailed_winners_path = REPORTS_DIR / "stage4_winners_detailed_metrics.csv"
                detailed_winners_df.to_csv(detailed_winners_path, index=False)
                tprint(f"Saved detailed winners report to {detailed_winners_path}")

    except Exception as e:
        tprint(f"Warning: failed to save reports: {e}")
        traceback.print_exc()
    except Exception as e:
        tprint(f"Warning: failed to save best params: {e}")

    tprint("="*80)
    tprint("MASK OPTIMIZATION 4-MODES: COMPLETE")
    tprint("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize Layer 0 masks for Long/Short strategies"
    )
    parser.add_argument("--data-root", help="Override data root")
    parser.add_argument("--features", help="Path to features directory")
    parser.add_argument("--perps", action="store_true", help="Use perpetual mode data")
    parser.add_argument("--max-symbols", type=int, help="Cap symbols for speed")
    parser.add_argument(
        "--lookback-years", type=float, default=2.0, help="Years of data to load"
    )
    parser.add_argument(
        "--mode",
        choices=["all"] + ALL_MODES,
        default="all",
        help="Run all modes or one mode only",
    )
    parser.add_argument("--subset", type=float, help="Subset fraction of data")
    parser.add_argument("--symbols", type=int, help="Override symbol selection (deprecated, use --max-symbols)")
    parser.add_argument("--z-start", type=int, help="Start index for z_hours grid")
    parser.add_argument("--z-step", type=int, help="Step for z_hours grid")
    parser.add_argument("--lookback-h", type=int, help="Override lookback-h")
    parser.add_argument("--min-events", type=int, help="Override min events")
    parser.add_argument("--support", type=float, help="Override support ratio")
    parser.add_argument("--shortlist", type=int, help="Override shortlist max")
    parser.add_argument("--parallel", type=int, help="Parallel processes")
    parser.add_argument("--fast", type=int, default=0, help="Internal fast mode toggle")
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help="Start from this optimizer stage",
    )
    parser.add_argument(
        "--stop-stage",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=6,
        help="Stop after this optimizer stage",
    )
    parser.add_argument(
        "--resume-artifact-dir",
        help="Load stage artifacts from a previous optimizer run directory",
    )
    parser.add_argument(
        "--run-id",
        help="Artifact run id for this optimizer execution",
    )
    args = parser.parse_args()
    if int(args.start_stage) > int(args.stop_stage):
        parser.error("--start-stage must be <= --stop-stage")
    if args.symbols is not None and args.max_symbols is None:
        args.max_symbols = args.symbols
    run_mask_optimization_4modes(args)
