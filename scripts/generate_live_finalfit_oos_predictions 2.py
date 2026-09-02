#!/usr/bin/env python3
"""Generate policy-OOS predictions with the live final-fit inference stack.

This is intentionally separate from ``generate_policy_oos_predictions.py``:
that script exports the frozen policy-OOS model outputs used by the optimiser.
This script re-scores the same OOS candidate/reference rows with the model
bundle that live inference loads: final-fit base models plus the policy/meta
artifact bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.inference.config import load_inference_config  # noqa: E402
from extreme_price_movements.inference.feature_generator import (  # noqa: E402
    _meta_model_derived_raw_dependencies,
    _latest_matrix_low_finite_repair_incidents,
    _latest_matrix_low_finite_support,
    _selected_latest_cache_min_finite_fraction,
    get_features_for_candidates,
    raw_required_feature_keys,
)
from extreme_price_movements.inference.model_orchestrator import (  # noqa: E402
    LGBM_DIAGNOSTIC_LEDGER_KEYS,
    ModelOrchestrator,
    _effective_alpha_feature_contract,
    _effective_selected_feature_contract,
    _selected_feature_owner,
)
from extreme_price_movements.inference.parity import (  # noqa: E402
    calibrated_score_and_threshold,
    strategy_core_id,
)
from extreme_price_movements.inference.policy_rank_reference import (  # noqa: E402
    PolicyRankReferenceStore,
    strategy_rank_reference_aliases,
)
from extreme_price_movements.data_store import (  # noqa: E402
    _feature_schema_names as _feature_store_schema_names,
    read_symbol_features as _read_feature_store_symbol_features,
)
from extreme_price_movements.model_loader import load_full_state  # noqa: E402
from extreme_price_movements.lgbm_pipeline import (  # noqa: E402
    LGBM_META_LEAF_MAX_TREES,
    _leaf_metadata,
)
from extreme_price_movements.simple_position_sizer import load_calibration_curves  # noqa: E402
from extreme_price_movements.utils import tprint  # noqa: E402
from scripts.replay_live_signal_predictions import _normalise_symbol  # noqa: E402


RUNTIME_SYNTH_PREFIXES = ("ret1h_G_VOL_", "ret1h_G_TREND_")
RUNTIME_SYNTH_EXACT = {"G_VOL", "G_TREND", "barrier_pct"}
MODEL_DERIVED_CONTRACT_FEATURES = {
    "negative_log_likelihood",
    "signed_prediction_error",
    "surprise_error_z",
    "wrong_confident",
}
RELIABILITY_BLEND_META_DIAGNOSTIC_COLUMNS = tuple(
    dict.fromkeys(
        [
            "feature_drift_psi_core",
            "feature_drift_ks_core",
            "feature_drift_cov_shift",
            "regime_centroid_similarity_train",
            "rare_leaf_fraction",
            "leaf_count_p10",
            "leaf_count_min",
            "leaf_weight_p10",
            "leaf_depth_mean",
            "contrib_top1_abs_share",
            "contrib_top3_abs_share",
            "contrib_entropy",
            "contrib_balance",
            "num_material_contrib_features",
            "prob_uncertainty",
            "entropy",
            *[str(c) for c in LGBM_DIAGNOSTIC_LEDGER_KEYS],
        ]
    )
)
RELIABILITY_BLEND_CANONICAL_DIAGNOSTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "feature_drift_psi_core": (
        "feature_drift_psi_core_80",
        "feature_drift_psi_core_50",
        "row_drift_v1_psi_core",
        "row_drift_v1_psi_core_80",
        "row_drift_v1_psi_core_50",
    ),
    "feature_drift_ks_core": (
        "feature_drift_ks_bin_mean",
        "feature_drift_ks_bin_max",
        "row_drift_v1_ks_core",
        "row_drift_v1_ks_bin_mean",
        "row_drift_v1_ks_bin_max",
    ),
    "feature_drift_cov_shift": (
        "frobenius_corr_shift",
        "row_drift_v1_frobenius_corr_shift",
    ),
    "regime_centroid_similarity_train": (
        "regime_centroid_similarity_train_window_mean",
        "regime_centroid_similarity_train_pc0",
        "regime_centroid_similarity_train_window_p10",
    ),
    "rare_leaf_fraction": (
        "rare_leaf_low_support_score",
        "row_drift_v1_rare_leaf_low_support_score",
        "leaf_low_freq_fraction",
    ),
}
RELIABILITY_BLEND_CANONICAL_DIAGNOSTIC_KEYS = (
    "feature_drift_psi_core",
    "feature_drift_ks_core",
    "feature_drift_cov_shift",
    "regime_centroid_similarity_train",
    "rare_leaf_fraction",
    "leaf_count_p10",
    "leaf_count_min",
    "leaf_weight_p10",
    "leaf_depth_mean",
    "contrib_top1_abs_share",
    "contrib_top3_abs_share",
    "contrib_entropy",
    "contrib_balance",
    "num_material_contrib_features",
    "prob_uncertainty",
    "entropy",
)
LIVE_FINALFIT_PREDICTION_COLUMNS = (
    "timestamp",
    "symbol",
    "side",
    "strategy_id",
    "has_feature_row",
    "feature_cols",
    "base_pred",
    "meta_pred",
    "raw_prediction_score",
    "calibrated_score",
    "policy_rank_pct",
    "policy_rank_reference_n",
    "policy_rank_reference_source",
    "auction_rank_pct",
    "auction_rank_reference_n",
    "auction_rank_reference_source",
    "policy_ref_calibrated_score",
    "policy_ref_rank_pct",
    "base_prediction_error",
    "meta_prediction_error",
)


def _force_live_meta_diagnostic_flags(orchestrator: ModelOrchestrator) -> int:
    """Request deployable meta diagnostics without changing the prediction matrix."""
    changed = 0
    seen: set[int] = set()
    for model in getattr(orchestrator, "meta_models", {}).values():
        stack = [model]
        for attr in ("best_model", "estimator", "model", "clf", "classifier"):
            child = getattr(model, attr, None)
            if child is not None and child is not model:
                stack.append(child)
        while stack:
            candidate = stack.pop()
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            for attr in ("best_model", "estimator", "model", "clf", "classifier"):
                child = getattr(candidate, attr, None)
                if child is not None and child is not candidate:
                    stack.append(child)
            for attr, value in (
                ("meta_leaf_lite_diagnostics_enabled", True),
                ("meta_leaf_support_diagnostics_enabled", False),
                ("meta_leaf_target_diagnostics_enabled", False),
                ("meta_leaf_centroid_diagnostics_enabled", False),
                ("meta_leaf_diagnostics_enabled", False),
                ("meta_contrib_diagnostics_enabled", True),
                ("meta_score_path_diagnostics_enabled", True),
                ("meta_drift_features_enabled", True),
            ):
                if hasattr(candidate, attr) and getattr(candidate, attr, None) != value:
                    setattr(candidate, attr, value)
                    changed += 1
    return changed


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_strategy_filename(strategy_id: str) -> str:
    sid = str(strategy_id or "").strip()
    return "".join(ch if ch.isalnum() or ch in "_.=-" else "_" for ch in sid) or "unknown_strategy"


def _strategy_side(strategy_id: str) -> str:
    return "short" if str(strategy_id).startswith("short_") else "long"


def _is_runtime_synth_feature(key: str) -> bool:
    key_s = str(key or "")
    return (
        key_s in RUNTIME_SYNTH_EXACT
        or key_s in MODEL_DERIVED_CONTRACT_FEATURES
        or key_s.startswith(RUNTIME_SYNTH_PREFIXES)
        or "_G_VOL_" in key_s
        or "_G_TREND_" in key_s
    )


def _runtime_synth_base_key(key: str) -> str | None:
    key_s = str(key or "")
    for gate_name in ("G_VOL", "G_TREND"):
        marker = f"_{gate_name}_"
        if marker in key_s:
            base, state = key_s.rsplit(marker, 1)
            if base and state in {"0", "1"}:
                return base
    return None


def _normalised_feature_symbol(symbol: str) -> str:
    return _normalise_symbol(symbol)


def _feature_symbol_path(feature_root: Path, symbol: str) -> Path:
    norm = _normalised_feature_symbol(symbol)
    return feature_root / f"symbol={norm.replace('/', '_')}.parquet"


def _policy_rank_reference_path(
    data_root: Path,
    policy_run_id: str,
    strategy_id: str,
    *,
    rank_reference_dir: Path | None = None,
) -> Path:
    root = (
        Path(rank_reference_dir)
        if rank_reference_dir is not None
        else data_root / "artifacts" / policy_run_id / "simple_policy_optimiser" / "rank_reference"
    )
    for alias in strategy_rank_reference_aliases(strategy_id):
        path = root / f"{_safe_strategy_filename(alias)}.parquet"
        if path.exists():
            return path
    core = strategy_core_id(strategy_id)
    matches = sorted(root.glob(f"*{core}*.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No policy rank-reference parquet found for {strategy_id} in {root}")


def _load_rank_reference(
    path: Path,
    sample_rows: int = 0,
    *,
    sample_position: str = "tail",
    min_timestamp: str | None = None,
    max_timestamp: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])
    if min_timestamp:
        min_ts = pd.Timestamp(min_timestamp)
        if min_ts.tzinfo is None:
            min_ts = min_ts.tz_localize("UTC")
        else:
            min_ts = min_ts.tz_convert("UTC")
        frame = frame.loc[frame["timestamp"] >= min_ts].copy()
    if max_timestamp:
        max_ts = pd.Timestamp(max_timestamp)
        if max_ts.tzinfo is None:
            max_ts = max_ts.tz_localize("UTC")
        else:
            max_ts = max_ts.tz_convert("UTC")
        frame = frame.loc[frame["timestamp"] <= max_ts].copy()
    if sample_rows and sample_rows > 0 and len(frame) > sample_rows:
        # Preserve dense timestamp batches; this exercises the same rank context
        # that policy and live inference use.
        chosen_ts: list[pd.Timestamp] = []
        total = 0
        grouped = frame.groupby("timestamp", sort=True).size()
        items = list(grouped.items())
        if str(sample_position or "tail").lower() == "tail":
            items = list(reversed(items))
        for ts, count in items:
            chosen_ts.append(ts)
            total += int(count)
            if total >= int(sample_rows):
                break
        frame = frame[frame["timestamp"].isin(chosen_ts)].copy()
        frame = frame.sort_values(["timestamp", "symbol"])
        if len(frame) > sample_rows:
            frame = (
                frame.tail(sample_rows).copy()
                if str(sample_position or "tail").lower() == "tail"
                else frame.head(sample_rows).copy()
            )
    frame["symbol_norm"] = frame["symbol"].map(_normalised_feature_symbol)
    return frame.reset_index(drop=True)


def _load_sample_ledger(
    path: Path,
    *,
    strategy_id: str,
    min_timestamp: str | None = None,
    max_timestamp: str | None = None,
) -> pd.DataFrame:
    """Load externally materialized candidate rows for live-stack rescoring.

    The normal exporter samples from frozen policy rank-reference rows.  Later
    shadow/OOS ledgers are already concrete timestamp/symbol/strategy rows, so
    this loader preserves that decision universe while allowing the same live
    final-fit model stack to recompute anchor/meta scores.
    """
    frame = pd.read_parquet(path)
    frame = frame.copy()
    if "signal_bar_ts" in frame.columns and "timestamp" not in frame.columns:
        frame["timestamp"] = frame["signal_bar_ts"]
    missing = [c for c in ("timestamp", "symbol", "strategy_id") if c not in frame.columns]
    if missing:
        raise RuntimeError(f"sample ledger {path} missing required columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["symbol_norm"] = frame["symbol"].map(_normalised_feature_symbol)
    if frame.empty:
        return frame.reset_index(drop=True)
    frame = frame.loc[frame["strategy_id"].eq(str(strategy_id))].copy()
    frame = frame.dropna(subset=["timestamp", "symbol"])
    if min_timestamp:
        min_ts = pd.Timestamp(min_timestamp)
        min_ts = min_ts.tz_localize("UTC") if min_ts.tzinfo is None else min_ts.tz_convert("UTC")
        frame = frame.loc[frame["timestamp"] >= min_ts].copy()
    if max_timestamp:
        max_ts = pd.Timestamp(max_timestamp)
        max_ts = max_ts.tz_localize("UTC") if max_ts.tzinfo is None else max_ts.tz_convert("UTC")
        frame = frame.loc[frame["timestamp"] <= max_ts].copy()
    if frame.empty:
        return frame.reset_index(drop=True)
    frame = frame.sort_values(["timestamp", "symbol"], kind="mergesort")
    duplicate_keys = ["timestamp", "strategy_id", "symbol"]
    duplicate = frame.duplicated(duplicate_keys, keep=False)
    if bool(duplicate.any()):
        sample = frame.loc[duplicate, duplicate_keys].head(10).to_dict("records")
        raise RuntimeError(f"sample ledger has duplicate rows for {strategy_id}: {sample}")
    return frame.reset_index(drop=True)


def _filter_cached_matrix_rows(samples: pd.DataFrame, feature_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if samples.empty:
        return samples, {"cached_matrix_only": True, "input_rows": 0, "output_rows": 0}
    timestamps = pd.to_datetime(samples["timestamp"], utc=True, errors="coerce")
    has_cache = timestamps.map(lambda ts: _latest_matrix_path(feature_root, pd.Timestamp(ts)).exists())
    out = samples.loc[has_cache.fillna(False).to_numpy()].copy()
    audit = {
        "cached_matrix_only": True,
        "input_rows": int(len(samples)),
        "output_rows": int(len(out)),
        "dropped_rows_without_cached_matrix": int(len(samples) - len(out)),
        "input_timestamps": int(timestamps.nunique()),
        "output_timestamps": int(pd.to_datetime(out["timestamp"], utc=True, errors="coerce").nunique())
        if not out.empty
        else 0,
    }
    return out.reset_index(drop=True), audit


def _alpha_feature_contract(orchestrator: ModelOrchestrator, strategy_id: str, side: str) -> list[str]:
    info = orchestrator.alpha_by_strategy.get(strategy_id)
    if info is None:
        info = orchestrator.alpha_by_strategy.get(f"{side}_{strategy_id}")
    return [str(c) for c in _effective_alpha_feature_contract(info or {})]


def _meta_feature_contract(orchestrator: ModelOrchestrator, strategy_id: str, side: str) -> list[str]:
    candidates = [
        strategy_id,
        f"{strategy_id}_clf",
        f"{strategy_id}_tbm_clf",
        f"{strategy_core_id(strategy_id)}_clf",
        f"{strategy_core_id(strategy_id)}_tbm_clf",
        f"{side}_{strategy_id}_clf",
        f"{side}_{strategy_core_id(strategy_id)}_clf",
        f"{side}_{strategy_id}_tbm_clf",
        f"{side}_{strategy_core_id(strategy_id)}_tbm_clf",
    ]
    for key in candidates:
        if key and key in orchestrator.meta_models:
            return [str(c) for c in _effective_selected_feature_contract(orchestrator.meta_models[key])]
    return []


def _available_feature_columns(path: Path) -> set[str]:
    try:
        return {str(name) for name in _feature_store_schema_names(str(path)) if str(name) != "ts"}
    except Exception:
        try:
            return set(map(str, pd.read_parquet(path).columns))
        except Exception:
            return set()


def _read_symbol_features(
    path: Path,
    columns: Iterable[str],
    *,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    cols = [str(c) for c in dict.fromkeys(columns) if str(c)]
    if not cols:
        return pd.DataFrame()
    try:
        frame = _read_feature_store_symbol_features(
            str(path),
            columns=cols,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    except Exception:
        frame = pd.read_parquet(path)
        keep = [c for c in cols if c in frame.columns]
        frame = frame.loc[:, keep]
    if frame.empty:
        return frame
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    return frame


def _latest_matrix_path(feature_root: Path, ts: pd.Timestamp) -> Path:
    ts_utc = pd.Timestamp(ts)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    else:
        ts_utc = ts_utc.tz_convert("UTC")
    stamp = ts_utc.strftime("%Y%m%dT%H%M%SZ")
    return feature_root / "_live_latest_matrix" / f"matrix_{stamp}.parquet"


def _load_cached_latest_matrices(
    samples: pd.DataFrame,
    *,
    feature_root: Path,
    feature_keys: set[str],
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, Any], set[tuple[pd.Timestamp, str]]]:
    cells: dict[str, dict[str, pd.Series]] = {key: {} for key in feature_keys}
    covered: set[tuple[pd.Timestamp, str]] = set()
    matrix_count = 0
    missing_matrix_count = 0
    missing_cols: dict[str, list[str]] = {}
    started = time.monotonic()
    for ts, group in samples.groupby("timestamp", sort=True):
        ts_utc = pd.Timestamp(ts)
        path = _latest_matrix_path(feature_root, ts_utc)
        if not path.exists():
            missing_matrix_count += 1
            continue
        try:
            matrix = pd.read_parquet(path)
        except Exception:
            missing_matrix_count += 1
            continue
        if matrix.empty:
            missing_matrix_count += 1
            continue
        if "symbol" in matrix.columns:
            matrix_symbols = matrix["symbol"].map(_normalised_feature_symbol)
            matrix = matrix.copy()
            matrix.index = pd.Index(matrix_symbols.astype(str), name="symbol")
        else:
            matrix.index = pd.Index(
                [_normalised_feature_symbol(v) for v in matrix.index],
                name="symbol",
            )
        symbols = list(dict.fromkeys(group["symbol_norm"].astype(str).tolist()))
        available_symbols = [sym for sym in symbols if sym in matrix.index]
        if not available_symbols:
            continue
        matrix_count += 1
        available_cols = set(map(str, matrix.columns))
        missing = sorted(feature_keys.difference(available_cols))
        if missing:
            missing_cols[str(ts_utc)] = missing[:64]
        selected = matrix.reindex(index=available_symbols)
        low_finite = _latest_matrix_low_finite_support(
            selected,
            required_feature_keys=set(feature_keys),
            min_fraction=_selected_latest_cache_min_finite_fraction(),
        )
        repair_incidents = _latest_matrix_low_finite_repair_incidents(low_finite)
        repair_feature_set = {
            str(item.get("feature"))
            for item in repair_incidents
            if str(item.get("feature") or "")
        }
        needs_symbol_fallback = bool(missing or repair_feature_set)
        if repair_feature_set:
            missing_cols[str(ts_utc)] = (
                missing_cols.get(str(ts_utc), [])
                + [
                    "__repair_incident_low_finite__:"
                    + ",".join(str(item.get("feature")) for item in repair_incidents[:16])
                ]
            )[:64]
        for feature in feature_keys:
            if feature in repair_feature_set:
                continue
            if feature not in selected.columns:
                continue
            values = pd.Series(
                pd.to_numeric(selected[feature], errors="coerce").to_numpy(dtype=np.float32),
                index=pd.DatetimeIndex([ts_utc] * len(available_symbols)),
                name=feature,
            )
            for symbol, value in zip(available_symbols, values.to_numpy(), strict=False):
                series = pd.Series(
                    [value],
                    index=pd.DatetimeIndex([ts_utc]),
                    dtype=np.float32,
                )
                cells.setdefault(feature, {})[symbol] = (
                    pd.concat([cells[feature][symbol], series]).sort_index()
                    if symbol in cells.get(feature, {})
                    else series
                )
        if not needs_symbol_fallback:
            for symbol in available_symbols:
                covered.add((ts_utc, symbol))
    audit = {
        "cached_matrix_count": matrix_count,
        "cached_matrix_missing_timestamp_count": missing_matrix_count,
        "cached_matrix_covered_rows": len(covered),
        "cached_matrix_missing_columns_timestamp_count": len(missing_cols),
        "cached_matrix_missing_columns_sample": dict(list(missing_cols.items())[:12]),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return cells, audit, covered


def _make_feature_panels(
    samples: pd.DataFrame,
    *,
    feature_root: Path,
    feature_keys: set[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    started = time.monotonic()
    wanted = {str(k) for k in feature_keys if str(k)}
    if samples.empty:
        return {}, {
            "feature_root": str(feature_root),
            "requested_feature_count": len(wanted),
            "persisted_requested_feature_count": len(
                {k for k in wanted if not _is_runtime_synth_feature(k)}
            ),
            "loaded_feature_count": 0,
            "symbols_requested": 0,
            "symbols_with_feature_file": 0,
            "missing_symbol_files": [],
            "missing_symbol_file_count": 0,
            "missing_persisted_columns_symbol_count": 0,
            "missing_persisted_columns_sample": {},
            "cached_latest_matrix_audit": {
                "cached_matrix_count": 0,
                "cached_matrix_missing_timestamp_count": 0,
                "cached_matrix_covered_rows": 0,
                "cached_matrix_missing_columns_timestamp_count": 0,
                "cached_matrix_missing_columns_sample": {},
                "elapsed_seconds": 0.0,
            },
            "empty_sample_ledger": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    if "symbol_norm" not in samples.columns:
        raise RuntimeError("sample ledger rows must include symbol_norm before feature loading")
    symbols = sorted(set(samples["symbol_norm"].dropna().astype(str)))
    persisted_wanted = {k for k in wanted if not _is_runtime_synth_feature(k)}
    for key in list(wanted):
        base = _runtime_synth_base_key(key)
        if base:
            persisted_wanted.add(base)

    cells, matrix_audit, matrix_covered = _load_cached_latest_matrices(
        samples,
        feature_root=feature_root,
        feature_keys=wanted,
    )
    missing_symbol_files: list[str] = []
    missing_persisted_columns: dict[str, list[str]] = {}
    available_symbols = 0

    for i, symbol in enumerate(symbols, start=1):
        symbol_sample_ts = set(
            pd.to_datetime(
                samples.loc[samples["symbol_norm"].eq(symbol), "timestamp"],
                utc=True,
                errors="coerce",
            ).dropna()
        )
        uncovered_ts = sorted(
            ts for ts in symbol_sample_ts if (pd.Timestamp(ts), symbol) not in matrix_covered
        )
        if not uncovered_ts:
            continue
        path = _feature_symbol_path(feature_root, symbol)
        if not path.exists():
            missing_symbol_files.append(symbol)
            continue
        available_cols = _available_feature_columns(path)
        read_cols = sorted(c for c in persisted_wanted if c in available_cols)
        missing_cols = sorted(c for c in persisted_wanted if c not in available_cols)
        if missing_cols:
            missing_persisted_columns[symbol] = missing_cols[:64]
        if not read_cols:
            continue
        needed_ts = pd.DatetimeIndex(uncovered_ts)
        if needed_ts.empty:
            continue
        frame = _read_symbol_features(
            path,
            read_cols,
            start_ts=min(needed_ts),
            end_ts=max(needed_ts),
        )
        if frame.empty:
            continue
        selected = frame.reindex(needed_ts)
        available_symbols += 1
        for col in selected.columns:
            cells.setdefault(str(col), {})[symbol] = selected[col].astype(np.float32, copy=False)
        if i == 1 or i % 50 == 0 or i == len(symbols):
            tprint(
                "  loaded symbol features: "
                f"{i:,}/{len(symbols):,} symbols elapsed={time.monotonic() - started:.1f}s"
            )

    # Cheap deterministic runtime materialization for gate-expanded features.
    for key in wanted:
        base = _runtime_synth_base_key(key)
        if not base:
            continue
        base_cells = cells.get(base, {})
        if base_cells and not cells.get(key):
            cells[key] = {symbol: series.copy() for symbol, series in base_cells.items()}

    panels: dict[str, pd.DataFrame] = {}
    for feature, by_symbol in cells.items():
        if not by_symbol:
            continue
        panels[feature] = pd.DataFrame(by_symbol).sort_index().astype(np.float32, copy=False)

    audit = {
        "feature_root": str(feature_root),
        "requested_feature_count": len(wanted),
        "persisted_requested_feature_count": len(persisted_wanted),
        "loaded_feature_count": len(panels),
        "symbols_requested": len(symbols),
        "symbols_with_feature_file": available_symbols,
        "missing_symbol_files": missing_symbol_files[:128],
        "missing_symbol_file_count": len(missing_symbol_files),
        "missing_persisted_columns_symbol_count": len(missing_persisted_columns),
        "missing_persisted_columns_sample": dict(list(missing_persisted_columns.items())[:12]),
        "cached_latest_matrix_audit": matrix_audit,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return panels, audit


def _feature_vector_snapshot(
    samples: pd.DataFrame,
    feature_panels: dict[str, pd.DataFrame],
    feature_keys: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = [str(c) for c in dict.fromkeys(feature_keys)]
    for _, sample in samples.iterrows():
        ts = pd.Timestamp(sample["timestamp"])
        symbol = str(sample["symbol_norm"])
        finite = 0
        runtime_materialized = 0
        missing: list[str] = []
        for feature in ordered:
            if _is_runtime_synth_feature(feature) and feature not in feature_panels:
                runtime_materialized += 1
                continue
            panel = feature_panels.get(feature)
            value = np.nan
            if isinstance(panel, pd.DataFrame) and ts in panel.index and symbol in panel.columns:
                value = _safe_float(panel.at[ts, symbol])
            if np.isfinite(value):
                finite += 1
            else:
                missing.append(feature)
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "selected_feature_count": len(ordered),
                "finite_selected_feature_count": finite,
                "runtime_materialized_feature_count": runtime_materialized,
                "strict_feature_count": len(ordered) - runtime_materialized,
                "missing_selected_feature_count": len(missing),
                "missing_selected_features_json": json.dumps(missing, separators=(",", ":")),
                "feature_parity_ok": len(missing) == 0,
            }
        )
    return pd.DataFrame(rows)


def _align_diagnostics_frame(diag: Any, index: pd.Index) -> pd.DataFrame:
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return pd.DataFrame(index=index)
    out = diag.copy()
    if len(out) == len(index):
        out.index = index
    else:
        out = out.reindex(index)
    keep: dict[str, pd.Series] = {}
    for col in RELIABILITY_BLEND_META_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            continue
        keep[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    if not keep:
        return pd.DataFrame(index=index)
    return pd.DataFrame(keep, index=index)


def _predict_leaf_ids(model: Any, X: pd.DataFrame, max_trees: int) -> np.ndarray | None:
    try:
        kwargs = {"num_iteration": int(max_trees)} if int(max_trees) > 0 else {}
        leaves = np.asarray(model.predict(X, pred_leaf=True, **kwargs), dtype=np.int32)
    except TypeError:
        try:
            leaves = np.asarray(model.predict(X, pred_leaf=True), dtype=np.int32)
        except Exception:
            return None
    except Exception:
        return None
    if leaves.ndim == 1:
        leaves = leaves.reshape(len(X), 1)
    if leaves.ndim != 2 or leaves.shape[0] != len(X) or leaves.shape[1] == 0:
        return None
    if int(max_trees) > 0:
        leaves = leaves[:, : int(max_trees)]
    return leaves


def _direct_meta_lgbm_structural_diagnostics(orchestrator: ModelOrchestrator) -> pd.DataFrame:
    """Compute live-available structural LGBM summaries from boosters.

    This fills only decision-time structural diagnostics: leaf count/weight/depth
    from LightGBM tree dumps and contribution summaries from current-model
    `pred_contrib`. It deliberately avoids realized leaf outcome diagnostics.
    """
    X = getattr(orchestrator, "_last_meta_model_input", pd.DataFrame())
    key = str(getattr(orchestrator, "_last_meta_model_key", "") or "")
    if not isinstance(X, pd.DataFrame) or X.empty or not key:
        return pd.DataFrame(index=getattr(X, "index", None))
    meta_model = getattr(orchestrator, "meta_models", {}).get(key)
    if meta_model is None:
        return pd.DataFrame(index=X.index)
    owner = _selected_feature_owner(meta_model)
    models = [m for m in (getattr(owner, "models", []) or []) if hasattr(m, "predict")]
    if not models:
        return pd.DataFrame(index=X.index)
    X_model = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32, copy=False)
    max_trees = max(0, int(LGBM_META_LEAF_MAX_TREES or 0))
    count_cols: list[np.ndarray] = []
    weight_cols: list[np.ndarray] = []
    depth_cols: list[np.ndarray] = []
    for model in models:
        meta = _leaf_metadata(model)
        if not meta:
            continue
        leaves = _predict_leaf_ids(model, X_model, max_trees)
        if leaves is None:
            continue
        tree_n = min(leaves.shape[1], len(meta))
        if max_trees > 0:
            tree_n = min(tree_n, max_trees)
        for tree_i in range(tree_n):
            tree_meta = meta[tree_i]
            if not tree_meta:
                continue
            leaf_ids = np.fromiter(tree_meta.keys(), dtype=np.int32, count=len(tree_meta))
            vals = np.asarray([tree_meta[int(leaf_id)] for leaf_id in leaf_ids], dtype=np.float32)
            if leaf_ids.size == 0 or vals.ndim != 2 or vals.shape[1] < 3:
                continue
            order = np.argsort(leaf_ids, kind="mergesort")
            leaf_ids_sorted = leaf_ids[order]
            vals_sorted = vals[order]
            leaf_col = np.asarray(leaves[:, tree_i], dtype=np.int32)
            pos = np.searchsorted(leaf_ids_sorted, leaf_col)
            pos_clip = np.clip(pos, 0, max(leaf_ids_sorted.size - 1, 0))
            valid = (pos < leaf_ids_sorted.size) & (leaf_ids_sorted[pos_clip] == leaf_col)
            if not np.any(valid):
                continue
            c = np.full(len(X_model), np.nan, dtype=np.float32)
            w = np.full(len(X_model), np.nan, dtype=np.float32)
            d = np.full(len(X_model), np.nan, dtype=np.float32)
            matched = vals_sorted[pos[valid]]
            c[valid] = matched[:, 0]
            w[valid] = matched[:, 1]
            d[valid] = matched[:, 2]
            count_cols.append(c)
            weight_cols.append(w)
            depth_cols.append(d)

    out = pd.DataFrame(index=X.index)
    if count_cols:
        count_mat = np.vstack(count_cols).T.astype(np.float32, copy=False)
        weight_mat = np.vstack(weight_cols).T.astype(np.float32, copy=False)
        depth_mat = np.vstack(depth_cols).T.astype(np.float32, copy=False)
        global_count_p10 = float(np.nanpercentile(count_mat, 10.0)) if count_mat.size else 0.0
        out["leaf_count_p10"] = np.nanpercentile(count_mat, 10.0, axis=1).astype(np.float32)
        out["leaf_count_min"] = np.nanmin(count_mat, axis=1).astype(np.float32)
        out["leaf_weight_p10"] = np.nanpercentile(weight_mat, 10.0, axis=1).astype(np.float32)
        out["leaf_depth_mean"] = np.nanmean(depth_mat, axis=1).astype(np.float32)
        out["rare_leaf_fraction"] = np.mean(
            count_mat <= max(global_count_p10, 1.0),
            axis=1,
        ).astype(np.float32)

    contrib_parts: list[np.ndarray] = []
    for model in models:
        try:
            contrib = np.asarray(model.predict(X_model, pred_contrib=True), dtype=np.float32)
        except Exception:
            continue
        if contrib.ndim != 2 or contrib.shape[0] != len(X_model) or contrib.shape[1] < 2:
            continue
        contrib_parts.append(contrib[:, :-1])
    if contrib_parts:
        contrib = np.mean(np.stack(contrib_parts, axis=0), axis=0).astype(np.float32)
        abs_c = np.abs(contrib)
        total_abs = np.sum(abs_c, axis=1) + 1e-12
        sorted_abs = np.sort(abs_c, axis=1)[:, ::-1]
        top3 = np.sum(sorted_abs[:, : min(3, sorted_abs.shape[1])], axis=1)
        share = abs_c / total_abs[:, None]
        entropy = -np.sum(
            np.where(share > 0.0, share * np.log(share + 1e-12), 0.0),
            axis=1,
        )
        entropy = entropy / max(np.log(max(abs_c.shape[1], 2)), 1e-12)
        out["contrib_top1_abs_share"] = (sorted_abs[:, 0] / total_abs).astype(np.float32)
        out["contrib_top3_abs_share"] = (top3 / total_abs).astype(np.float32)
        out["contrib_entropy"] = entropy.astype(np.float32)
        out["contrib_balance"] = np.clip(np.sum(contrib, axis=1) / total_abs, -1.0, 1.0).astype(np.float32)
        out["num_material_contrib_features"] = np.sum(share >= 0.01, axis=1).astype(np.float32)

    if out.empty:
        return out
    return out.replace([np.inf, -np.inf], np.nan).astype("float32", copy=False)


def _fill_diagnostics(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(fallback, pd.DataFrame) or fallback.empty:
        return primary
    if not isinstance(primary, pd.DataFrame) or primary.empty:
        return fallback.copy()
    out = primary.copy()
    for col in fallback.columns:
        values = pd.to_numeric(fallback[col], errors="coerce").astype("float32")
        if col not in out.columns:
            out[col] = values.reindex(out.index)
            continue
        current = pd.to_numeric(out[col], errors="coerce").astype("float32")
        bad = current.replace([np.inf, -np.inf], np.nan).isna()
        aligned = values.reindex(out.index)
        usable = bad & aligned.replace([np.inf, -np.inf], np.nan).notna()
        if bool(usable.any()):
            current.loc[usable] = aligned.loc[usable]
            out[col] = current.astype("float32")
    return out


def _uncertainty_score_from_meta_diagnostics(
    meta_diag: pd.DataFrame,
    meta_pred: pd.Series,
) -> pd.Series:
    idx = meta_diag.index
    pred = pd.to_numeric(meta_pred.reindex(idx), errors="coerce").astype(float)
    prob_uncertainty = 1.0 - (2.0 * np.abs(pred.to_numpy(dtype=np.float64) - 0.5))
    if "prob_uncertainty" in meta_diag.columns:
        values = pd.to_numeric(meta_diag["prob_uncertainty"], errors="coerce").to_numpy(dtype=np.float64)
        prob_uncertainty = np.where(np.isfinite(values), values, prob_uncertainty)
    prob_uncertainty = np.nan_to_num(prob_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)

    leaf_uncertainty = np.zeros(len(idx), dtype=np.float64)
    if "rare_leaf_fraction" in meta_diag.columns:
        leaf_uncertainty += np.nan_to_num(
            pd.to_numeric(meta_diag["rare_leaf_fraction"], errors="coerce").to_numpy(dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    if "leaf_count_p10" in meta_diag.columns:
        support = pd.to_numeric(meta_diag["leaf_count_p10"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(support)
        scale = float(np.nanpercentile(support[finite], 75.0)) if np.any(finite) else 1.0
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        leaf_uncertainty += np.clip(1.0 - np.nan_to_num(support, nan=scale) / scale, 0.0, 1.0)

    contrib_uncertainty = np.zeros(len(idx), dtype=np.float64)
    if "contrib_entropy" in meta_diag.columns:
        contrib_uncertainty += np.nan_to_num(
            pd.to_numeric(meta_diag["contrib_entropy"], errors="coerce").to_numpy(dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    if "contrib_top1_abs_share" in meta_diag.columns:
        top1 = pd.to_numeric(meta_diag["contrib_top1_abs_share"], errors="coerce").to_numpy(dtype=np.float64)
        contrib_uncertainty += 1.0 - np.nan_to_num(top1, nan=1.0, posinf=1.0, neginf=1.0)

    regime_distance = np.zeros(len(idx), dtype=np.float64)
    if "regime_centroid_similarity_train" in meta_diag.columns:
        sim = pd.to_numeric(meta_diag["regime_centroid_similarity_train"], errors="coerce").to_numpy(dtype=np.float64)
        regime_distance = 1.0 - np.nan_to_num(sim, nan=1.0, posinf=1.0, neginf=1.0)

    score = (
        0.35 * prob_uncertainty
        + 0.35 * leaf_uncertainty
        + 0.20 * contrib_uncertainty
        + 0.10 * regime_distance
    )
    return pd.Series(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), index=idx)


def _fill_series_from_aliases(
    frame: pd.DataFrame,
    canonical: str,
    aliases: tuple[str, ...],
) -> pd.Series | None:
    idx = frame.index
    if canonical in frame.columns:
        out = pd.to_numeric(frame[canonical], errors="coerce").astype("float32")
    else:
        out = pd.Series(np.nan, index=idx, dtype="float32")
    missing = out.replace([np.inf, -np.inf], np.nan).isna()
    if not bool(missing.any()):
        return out
    for alias in aliases:
        if alias not in frame.columns:
            continue
        values = pd.to_numeric(frame[alias], errors="coerce").astype("float32")
        usable = missing & values.replace([np.inf, -np.inf], np.nan).notna()
        if bool(usable.any()):
            out.loc[usable] = values.loc[usable]
            missing = out.replace([np.inf, -np.inf], np.nan).isna()
            if not bool(missing.any()):
                break
    return out if out.replace([np.inf, -np.inf], np.nan).notna().any() else None


def _live_meta_diagnostics_contract(
    meta_diag: pd.DataFrame,
    meta_pred: pd.Series,
) -> pd.DataFrame:
    if meta_diag.empty:
        return pd.DataFrame(index=meta_pred.index)
    out = pd.DataFrame(index=meta_diag.index)
    for col in RELIABILITY_BLEND_META_DIAGNOSTIC_COLUMNS:
        if col not in meta_diag.columns:
            continue
        values = pd.to_numeric(meta_diag[col], errors="coerce").astype("float32")
        out[col] = values
    for canonical, aliases in RELIABILITY_BLEND_CANONICAL_DIAGNOSTIC_ALIASES.items():
        filled = _fill_series_from_aliases(out, canonical, aliases)
        if filled is not None:
            out[canonical] = filled.astype("float32")
    pred = pd.to_numeric(meta_pred.reindex(out.index), errors="coerce").astype(float)
    prob_uncertainty = (
        1.0 - (2.0 * np.abs(pred.to_numpy(dtype=np.float64) - 0.5))
    ).astype(np.float32)
    if "prob_uncertainty" in out.columns:
        existing = pd.to_numeric(out["prob_uncertainty"], errors="coerce").astype("float32")
        bad = existing.replace([np.inf, -np.inf], np.nan).isna()
        existing.loc[bad] = prob_uncertainty[bad.to_numpy()]
        out["prob_uncertainty"] = existing.astype("float32")
    else:
        out["prob_uncertainty"] = prob_uncertainty
    if "entropy" not in out.columns or out["entropy"].replace([np.inf, -np.inf], np.nan).isna().any():
        prob = np.clip(pred.to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
        entropy = -(prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob)).astype(np.float32)
        if "entropy" in out.columns:
            existing = pd.to_numeric(out["entropy"], errors="coerce").astype("float32")
            bad = existing.replace([np.inf, -np.inf], np.nan).isna()
            existing.loc[bad] = entropy[bad.to_numpy()]
            out["entropy"] = existing.astype("float32")
        else:
            out["entropy"] = entropy
    out["uncertainty_score"] = _uncertainty_score_from_meta_diagnostics(meta_diag, meta_pred)
    out["meta_lgbm_uncertainty_score"] = out["uncertainty_score"]
    for col in RELIABILITY_BLEND_CANONICAL_DIAGNOSTIC_KEYS:
        if col in out.columns:
            out[f"meta_lgbm_{col}"] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    for col in list(out.columns):
        if str(col).startswith("meta_lgbm_") or col in RELIABILITY_BLEND_CANONICAL_DIAGNOSTIC_KEYS:
            continue
        out[f"meta_lgbm_{col}"] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out.astype("float32", copy=False)


def _build_exact_feature_matrix(
    work: pd.DataFrame,
    feature_panels: dict[str, pd.DataFrame],
    *,
    inject_historical_context: bool,
) -> tuple[pd.DataFrame, set[str]]:
    """Vectorized exact timestamp/symbol lookup for feature-store panels.

    ``get_features_for_candidates`` is optimized for one live timestamp.  For a
    historical candidate ledger it repeats index conversion and feature loops
    thousands of times.  The sample ledgers built for this workflow come from
    the same hourly feature store, so exact timestamp lookup is both equivalent
    and much faster.
    """

    if work.empty or not feature_panels:
        return pd.DataFrame(), set()
    usable_panels = {
        str(name): panel
        for name, panel in feature_panels.items()
        if isinstance(panel, pd.DataFrame) and not panel.empty
    }
    if not usable_panels:
        return pd.DataFrame(), set()

    row_ids = work["row_id"].astype(str).to_numpy(dtype=object)
    timestamps = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    symbols = work["symbol_norm"].astype(str)
    valid_rows = timestamps.notna().to_numpy()
    unique_ts = pd.DatetimeIndex(pd.unique(timestamps.loc[valid_rows])).sort_values()
    unique_symbols = pd.Index(pd.unique(symbols.loc[valid_rows]), dtype=object)
    if unique_ts.empty or unique_symbols.empty:
        return pd.DataFrame(), set()

    ts_pos = unique_ts.get_indexer(timestamps)
    symbol_pos = unique_symbols.get_indexer(symbols)
    valid_lookup = (ts_pos >= 0) & (symbol_pos >= 0)
    columns: dict[str, np.ndarray] = {}
    has_feature = np.zeros(len(work), dtype=bool)

    for feature_name, panel in usable_panels.items():
        aligned = panel
        if not isinstance(aligned.index, pd.DatetimeIndex) or aligned.index.tz is None:
            aligned = aligned.copy()
            aligned.index = pd.to_datetime(aligned.index, utc=True, errors="coerce")
        elif str(aligned.index.tz) != str(unique_ts.tz):
            aligned = aligned.copy()
            aligned.index = aligned.index.tz_convert("UTC")
        selected = aligned.reindex(index=unique_ts, columns=unique_symbols)
        values = selected.to_numpy(dtype=np.float32, copy=False)
        out = np.full(len(work), np.nan, dtype=np.float32)
        if bool(valid_lookup.any()):
            out[valid_lookup] = values[ts_pos[valid_lookup], symbol_pos[valid_lookup]]
        columns[feature_name] = out
        has_feature |= np.isfinite(out)

    if not columns or not bool(has_feature.any()):
        return pd.DataFrame(), set()
    X = pd.DataFrame(columns, index=pd.Index(row_ids, name="row_id"))
    X = X.loc[has_feature]
    if inject_historical_context:
        kept_positions = np.flatnonzero(has_feature)
        X["__symbol__"] = symbols.iloc[kept_positions].to_numpy(dtype=object)
        X["__ts__"] = timestamps.iloc[kept_positions].to_numpy()
    return X, set(map(str, row_ids[has_feature]))


def _score_strategy(
    samples: pd.DataFrame,
    feature_panels: dict[str, pd.DataFrame],
    orchestrator: ModelOrchestrator,
    *,
    strategy_id: str,
    calibration_data: dict[str, dict[str, Any]],
    rank_store: PolicyRankReferenceStore,
    chunk_rows: int,
    inject_historical_context: bool,
) -> pd.DataFrame:
    started = time.monotonic()
    side = _strategy_side(strategy_id)
    work = samples.copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=list(LIVE_FINALFIT_PREDICTION_COLUMNS))
    work["row_id"] = [f"row_{i}" for i in range(len(work))]
    X, feature_row_ids = _build_exact_feature_matrix(
        work,
        feature_panels,
        inject_historical_context=inject_historical_context,
    )
    if X.empty:
        feature_frames: list[pd.DataFrame] = []
        feature_row_ids = set()
        for ts, group in work.groupby("timestamp", sort=True):
            group_symbols = group["symbol_norm"].astype(str)
            symbols = list(dict.fromkeys(group_symbols.tolist()))
            rows = get_features_for_candidates(feature_panels, symbols, ts=ts)
            if rows.empty:
                continue
            if not rows.index.is_unique:
                rows = rows.groupby(level=0, sort=False).last()
            present = group_symbols.isin(rows.index)
            if not bool(present.any()):
                continue
            selected_symbols = group_symbols.loc[present].to_numpy(dtype=object)
            selected_row_ids = group.loc[present, "row_id"].astype(str).to_numpy(dtype=object)
            frame = rows.reindex(selected_symbols)
            frame.index = pd.Index(selected_row_ids, name="row_id")
            if inject_historical_context:
                frame["__symbol__"] = selected_symbols
                frame["__ts__"] = pd.Timestamp(ts)
            feature_frames.append(frame)
            feature_row_ids.update(map(str, selected_row_ids))
        if feature_frames:
            X = pd.concat(feature_frames, axis=0, copy=False)
        else:
            X = pd.DataFrame()
    else:
        tprint(
            "Built exact feature matrix: "
            f"strategy={strategy_id} rows={len(X):,} features={X.shape[1]:,} "
            f"elapsed={time.monotonic() - started:.1f}s"
        )

    alpha = pd.Series(index=X.index, data=np.nan, dtype=float)
    meta = pd.Series(index=X.index, data=np.nan, dtype=float)
    meta_diag = pd.DataFrame(index=X.index)
    base_error = ""
    meta_error = ""
    include_lgbm_diagnostics = bool(
        getattr(orchestrator, "cfg", {}).get(
            "inference_lgbm_internal_diagnostics_enabled",
            True,
        )
    )
    if not X.empty:
        try:
            parts: list[pd.Series] = []
            for start in range(0, len(X), chunk_rows):
                chunk = X.iloc[start : start + chunk_rows]
                pred = orchestrator.predict_alpha(chunk, side=side, kind=strategy_id)
                parts.append(pred.reindex(chunk.index))
            alpha = pd.concat(parts).reindex(X.index) if parts else alpha
        except Exception as exc:
            base_error = str(exc)
        try:
            parts = []
            diag_parts: list[pd.DataFrame] = []
            finite_alpha_index = alpha.replace([np.inf, -np.inf], np.nan).dropna().index
            if len(finite_alpha_index) > 0:
                meta_base = X.reindex(finite_alpha_index).copy()
                meta_base[strategy_id] = alpha.reindex(meta_base.index)
                for start in range(0, len(meta_base), chunk_rows):
                    chunk = meta_base.iloc[start : start + chunk_rows]
                    pred = orchestrator.predict_meta(chunk, side=side, kind=strategy_id)
                    parts.append(pred.reindex(chunk.index))
                    if include_lgbm_diagnostics:
                        diag_chunk = _align_diagnostics_frame(
                            getattr(orchestrator, "_last_meta_diagnostics_frame", pd.DataFrame()),
                            chunk.index,
                        )
                        direct_diag = _direct_meta_lgbm_structural_diagnostics(orchestrator)
                        diag_parts.append(
                            _fill_diagnostics(
                                diag_chunk,
                                _align_diagnostics_frame(direct_diag, chunk.index),
                            )
                        )
                if parts:
                    meta = pd.concat(parts).reindex(X.index)
                if diag_parts:
                    meta_diag = pd.concat(diag_parts, axis=0).reindex(X.index)
        except Exception as exc:
            meta_error = str(exc)
    live_diag = _live_meta_diagnostics_contract(meta_diag, meta)

    rows: list[dict[str, Any]] = []
    for _, sample in work.iterrows():
        row_id = str(sample["row_id"])
        base_pred = _safe_float(alpha.get(row_id, np.nan))
        meta_pred = _safe_float(meta.get(row_id, np.nan))
        calibrated = float("nan")
        policy_rank = None
        auction_rank = None
        if np.isfinite(meta_pred):
            calibrated, _ = calibrated_score_and_threshold(
                raw_score=meta_pred,
                strategy_id=strategy_id,
                calibration_data=calibration_data,
                default_threshold=1.0,
            )
            policy_rank = rank_store.lookup(
                strategy_id=strategy_id,
                side=side,
                calibrated_score=calibrated,
            )
            auction_rank = rank_store.lookup_auction(calibrated_score=calibrated)
        rows.append(
            {
                "timestamp": sample["timestamp"],
                "symbol": sample["symbol_norm"],
                "side": side,
                "strategy_id": strategy_id,
                "has_feature_row": row_id in feature_row_ids,
                "feature_cols": int(X.shape[1]) if row_id in feature_row_ids else 0,
                "base_pred": base_pred,
                "meta_pred": meta_pred,
                "raw_prediction_score": meta_pred,
                "calibrated_score": _safe_float(calibrated),
                "policy_rank_pct": _safe_float(policy_rank.policy_rank_pct if policy_rank else np.nan),
                "policy_rank_reference_n": int(policy_rank.n_rows if policy_rank else 0),
                "policy_rank_reference_source": str(policy_rank.source if policy_rank else ""),
                "auction_rank_pct": _safe_float(auction_rank.policy_rank_pct if auction_rank else np.nan),
                "auction_rank_reference_n": int(auction_rank.n_rows if auction_rank else 0),
                "auction_rank_reference_source": str(auction_rank.source if auction_rank else ""),
                "policy_ref_calibrated_score": _safe_float(sample.get("calibrated_score")),
                "policy_ref_rank_pct": _safe_float(
                    sample.get(
                        "rank_pct",
                        sample.get("strategy_rank_pct", sample.get("normalized_rank_score")),
                    )
                ),
                "base_prediction_error": base_error,
                "meta_prediction_error": meta_error,
            }
        )
        if row_id in live_diag.index:
            rows[-1].update(
                {
                    str(col): _safe_float(live_diag.at[row_id, col])
                    for col in live_diag.columns
                }
            )
    out = pd.DataFrame(rows)
    tprint(
        "Scored strategy: "
        f"strategy={strategy_id} rows={len(out):,} finite_base={out['base_pred'].notna().sum():,} "
        f"finite_meta={out['meta_pred'].notna().sum():,} elapsed={time.monotonic() - started:.1f}s"
    )
    return out


def _load_strategy_ids(policy_root: Path) -> list[str]:
    manifest_path = policy_root / "simple_policy_optimiser" / "rank_reference" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    strategies = manifest.get("strategies")
    if isinstance(strategies, dict) and strategies:
        return [str(k) for k in strategies.keys()]
    return []


def _overlay_policy_meta_state(
    model_state: dict[str, Any],
    policy_state: dict[str, Any],
) -> dict[str, Any]:
    """Attach policy/meta artifact state to a final-fit base model state.

    The deployment bundle uses final-fit base models while retaining the parent
    meta/model-context artifacts. Keep final-fit alpha models authoritative and
    overlay only components that are absent from the final-fit run or are
    explicitly meta/policy context.
    """
    out = dict(model_state or {})
    out_bundle = dict(out.get("bundle", {}) or {})
    policy_bundle = dict((policy_state or {}).get("bundle", {}) or {})
    for key in (
        "meta_models",
        "spike_models",
        "specialist_models",
        "alpha_oof_metrics",
        "quality_gate_report",
        "ev_decomposition",
        "feature_transform_contract",
        "feature_transform_contract_hash",
        "feature_transform_manifest",
    ):
        value = policy_bundle.get(key)
        if value:
            out_bundle[key] = value
    out["bundle"] = out_bundle
    for key in (
        "bucket_params",
        "booster_bundles",
        "regime_adaptors",
        "ridge_sizer",
    ):
        value = (policy_state or {}).get(key)
        if value and not out.get(key):
            out[key] = value
    return out


def _summary(strategy: str, feature_snapshot: pd.DataFrame, preds: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    finite_base = pd.to_numeric(preds.get("base_pred"), errors="coerce")
    finite_meta = pd.to_numeric(preds.get("meta_pred"), errors="coerce")
    finite_cal = pd.to_numeric(preds.get("calibrated_score"), errors="coerce")
    finite_rank = pd.to_numeric(preds.get("policy_rank_pct"), errors="coerce")
    return {
        "strategy_id": strategy,
        "rows": int(len(preds)),
        "feature_parity_rows": int(len(feature_snapshot)),
        "feature_parity_ok_rows": int(feature_snapshot.get("feature_parity_ok", pd.Series(dtype=bool)).astype(bool).sum())
        if not feature_snapshot.empty
        else 0,
        "feature_missing_rows": int(
            pd.to_numeric(
                feature_snapshot.get("missing_selected_feature_count", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .sum()
        )
        if not feature_snapshot.empty
        else 0,
        "base_pred_finite": int(np.isfinite(finite_base).sum()),
        "meta_pred_finite": int(np.isfinite(finite_meta).sum()),
        "calibrated_score_finite": int(np.isfinite(finite_cal).sum()),
        "policy_rank_pct_finite": int(np.isfinite(finite_rank).sum()),
        "timestamp_min": str(preds["timestamp"].min()) if "timestamp" in preds and not preds.empty else "",
        "timestamp_max": str(preds["timestamp"].max()) if "timestamp" in preds and not preds.empty else "",
        "feature_load_audit": audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--policy-artifact-run-id", default="")
    parser.add_argument("--model-artifact-run-id", default="20260618_081800_current4_final_fit")
    parser.add_argument("--feature-run-id", default="20260627_120000")
    parser.add_argument("--strategy-id", action="append", default=[])
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--sample-position", choices=("head", "tail"), default="tail")
    parser.add_argument(
        "--sample-ledger",
        type=Path,
        default=None,
        help=(
            "Optional concrete timestamp/symbol/strategy ledger to rescore "
            "with the live final-fit stack instead of sampling frozen policy "
            "rank-reference rows."
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument(
        "--inject-historical-context",
        action="store_true",
        help=(
            "Add __symbol__/__ts__ to the model matrix so model-derived rolling "
            "rank diagnostics are recomputed from the exported historical block. "
            "By default the exporter mirrors live batch inference, which does not "
            "pass timestamp context into the meta batch."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rank-reference-dir", type=Path, default=None)
    parser.add_argument("--min-timestamp", default="")
    parser.add_argument("--max-timestamp", default="")
    parser.add_argument(
        "--cached-matrix-only",
        action="store_true",
        help="Score only policy-reference rows whose timestamp has a selected-feature matrix cache.",
    )
    parser.add_argument(
        "--disable-lgbm-internal-diagnostics",
        action="store_true",
        help=(
            "Skip expensive per-row LightGBM internal diagnostics during broad "
            "historical scoring. Predictions and calibrated scores are still written."
        ),
    )
    parser.add_argument(
        "--skip-feature-snapshot",
        action="store_true",
        help="Skip feature_vector_parity snapshot generation for large broad exports.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write parquet artifacts only.",
    )
    args = parser.parse_args()

    os.environ.setdefault("EPM_EXCHANGE", "kraken")
    os.environ.setdefault("EXCHANGE_NAME", "kraken")
    os.environ.setdefault("PRIMARY_EXCHANGE", "kraken")
    os.environ.setdefault("EPM_MARKET_MODE", "perps")

    data_root = args.data_root
    policy_run_id = str(args.policy_artifact_run_id or args.model_artifact_run_id)
    manifest_path = data_root / "artifacts" / policy_run_id / "final_model_fit_manifest.json"
    final_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_run_id = str(args.model_artifact_run_id or final_manifest.get("model_artifact_run_id") or policy_run_id)
    feature_run_id = str(args.feature_run_id or final_manifest.get("feature_run_id") or "")
    feature_root = data_root / "features" / feature_run_id
    policy_root = data_root / "artifacts" / policy_run_id
    out_dir = args.output_dir or (
        policy_root / "historical_inference_parity" / "live_finalfit_policy_rank_direct_20260621"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_ids = args.strategy_id or _load_strategy_ids(policy_root)
    if not strategy_ids:
        strategy_ids = [str(s) for s in final_manifest.get("strategy_ids", [])]
    if not strategy_ids:
        raise SystemExit("No strategies resolved from args, policy manifest, or final-fit manifest")

    tprint(
        "Loading live model state for final-fit OOS export: "
        f"model_run_id={model_run_id} policy_run_id={policy_run_id} feature_run_id={feature_run_id}"
    )
    state = load_full_state(model_run_id, str(data_root))
    policy_state: dict[str, Any] = {}
    if policy_run_id != model_run_id:
        tprint(f"Loading policy/meta overlay state: policy_run_id={policy_run_id}")
        policy_state = load_full_state(policy_run_id, str(data_root))
        state = _overlay_policy_meta_state(state, policy_state)
    cfg = load_inference_config(
        run_id=policy_run_id,
        data_root=str(data_root),
        model_artifact_run_id=model_run_id,
        policy_artifact_run_id=policy_run_id,
        market_mode="perps",
    )
    runtime_cfg = dict(cfg or {})
    runtime_cfg["inference_lgbm_internal_diagnostics_enabled"] = not bool(
        args.disable_lgbm_internal_diagnostics
    )
    runtime_cfg["inference_model_timing_enabled"] = False
    runtime_cfg["preserve_logged_meta_model_derived_features"] = False
    orchestrator = ModelOrchestrator(
        state,
        runtime_cfg={"model_bundle": state.get("bundle", {}), **runtime_cfg},
    )
    diagnostic_flag_changes = _force_live_meta_diagnostic_flags(orchestrator)
    tprint(
        "Resolved live model stack: "
        f"alpha_heads={len(orchestrator.alpha_by_strategy)} "
        f"meta_heads={len(orchestrator.meta_models)} "
        f"diagnostic_flag_changes={diagnostic_flag_changes}"
    )
    use_legacy_sizer_calibration = str(
        os.getenv("EPM_INFERENCE_USE_SIMPLE_POSITION_SIZER_CALIBRATION", "0") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if use_legacy_sizer_calibration:
        calibration_data = load_calibration_curves(str(data_root), policy_run_id)
    else:
        calibration_data = {}
        tprint(
            "simple_position_sizer calibration disabled; using raw model scores "
            "with simple_policy_optimiser rank references"
        )
    rank_store = PolicyRankReferenceStore(data_root=data_root, run_id=policy_run_id)

    all_summaries: dict[str, Any] = {
        "schema_version": "live_finalfit_policy_oos_predictions_v1",
        "policy_artifact_run_id": policy_run_id,
        "model_artifact_run_id": model_run_id,
        "feature_run_id": feature_run_id,
        "output_dir": str(out_dir),
        "sample_source": str(args.sample_ledger) if args.sample_ledger else "policy_rank_reference",
        "use_legacy_sizer_calibration": bool(use_legacy_sizer_calibration),
        "meta_diagnostic_flag_changes": int(diagnostic_flag_changes),
        "strategies": {},
    }
    combined_parts: list[pd.DataFrame] = []

    for strategy_id in strategy_ids:
        tprint(f"Processing strategy={strategy_id}")
        side = _strategy_side(strategy_id)
        ref_path: Path | None = None
        if args.sample_ledger is not None:
            samples = _load_sample_ledger(
                args.sample_ledger,
                strategy_id=strategy_id,
                min_timestamp=str(args.min_timestamp or "") or None,
                max_timestamp=str(args.max_timestamp or "") or None,
            )
        else:
            ref_path = _policy_rank_reference_path(
                data_root,
                policy_run_id,
                strategy_id,
                rank_reference_dir=args.rank_reference_dir,
            )
            samples = _load_rank_reference(
                ref_path,
                sample_rows=int(args.sample_rows),
                sample_position=str(args.sample_position),
                min_timestamp=str(args.min_timestamp or "") or None,
                max_timestamp=str(args.max_timestamp or "") or None,
            )
        sample_filter_audit: dict[str, Any] = {"cached_matrix_only": False}
        if bool(args.cached_matrix_only):
            samples, sample_filter_audit = _filter_cached_matrix_rows(samples, feature_root)
            if samples.empty:
                tprint(
                    "No rows remain after cached-matrix-only filter: "
                    f"strategy={strategy_id}"
                )
        alpha_features = _alpha_feature_contract(orchestrator, strategy_id, side)
        meta_features = _meta_feature_contract(orchestrator, strategy_id, side)
        selected_features = list(dict.fromkeys(alpha_features + meta_features))
        raw_selected_features = list(
            dict.fromkeys(
                list(raw_required_feature_keys(selected_features))
                + list(_meta_model_derived_raw_dependencies(meta_features))
            )
        )
        panels, audit = _make_feature_panels(
            samples,
            feature_root=feature_root,
            feature_keys=set(raw_selected_features),
        )
        if bool(args.skip_feature_snapshot):
            feature_snapshot = pd.DataFrame()
        else:
            feature_snapshot = _feature_vector_snapshot(samples, panels, raw_selected_features)
        preds = _score_strategy(
            samples,
            panels,
            orchestrator,
            strategy_id=strategy_id,
            calibration_data=calibration_data,
            rank_store=rank_store,
            chunk_rows=max(1, int(args.chunk_rows)),
            inject_historical_context=bool(args.inject_historical_context),
        )
        alias = _safe_strategy_filename(strategy_id)
        strategy_dir = out_dir / alias
        strategy_dir.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(strategy_dir / "live_finalfit_policy_oos_predictions.parquet", index=False)
        if not bool(args.no_csv):
            preds.to_csv(strategy_dir / "live_finalfit_policy_oos_predictions.csv", index=False)
        if not preds.empty:
            combined_parts.append(preds)
        if not bool(args.skip_feature_snapshot):
            feature_snapshot.to_parquet(strategy_dir / "feature_vector_parity.parquet", index=False)
            if not bool(args.no_csv):
                feature_snapshot.to_csv(strategy_dir / "feature_vector_parity.csv", index=False)
        summary = _summary(strategy_id, feature_snapshot, preds, audit)
        summary["policy_rank_reference_path"] = str(ref_path) if ref_path is not None else ""
        summary["sample_ledger_path"] = str(args.sample_ledger) if args.sample_ledger is not None else ""
        summary["alpha_selected_feature_count"] = len(alpha_features)
        summary["meta_selected_feature_count"] = len(meta_features)
        summary["selected_feature_count"] = len(selected_features)
        summary["raw_selected_feature_count"] = len(raw_selected_features)
        summary["sample_filter_audit"] = sample_filter_audit
        summary["inject_historical_context"] = bool(args.inject_historical_context)
        summary["lgbm_internal_diagnostics_enabled"] = not bool(
            args.disable_lgbm_internal_diagnostics
        )
        summary["feature_snapshot_skipped"] = bool(args.skip_feature_snapshot)
        (strategy_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        all_summaries["strategies"][strategy_id] = summary

    if combined_parts:
        combined = pd.concat(combined_parts, axis=0, ignore_index=True, copy=False)
        combined = combined.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    else:
        combined = pd.DataFrame(columns=list(LIVE_FINALFIT_PREDICTION_COLUMNS))
    combined_path = out_dir / "combined_prediction_ledger.parquet"
    combined_csv_path = out_dir / "combined_prediction_ledger.csv"
    combined.to_parquet(combined_path, index=False)
    if not bool(args.no_csv):
        combined.to_csv(combined_csv_path, index=False)
    all_summaries["combined_prediction_ledger"] = str(combined_path)
    all_summaries["combined_rows"] = int(len(combined))
    all_summaries["combined_timestamp_min"] = (
        str(combined["timestamp"].min()) if "timestamp" in combined.columns and not combined.empty else ""
    )
    all_summaries["combined_timestamp_max"] = (
        str(combined["timestamp"].max()) if "timestamp" in combined.columns and not combined.empty else ""
    )
    (out_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tprint(f"Wrote live final-fit OOS export summary to {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
