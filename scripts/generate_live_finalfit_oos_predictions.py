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
    ModelOrchestrator,
    _effective_alpha_feature_contract,
    _effective_selected_feature_contract,
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
    symbols = sorted(set(samples["symbol_norm"].dropna().astype(str)))
    wanted = {str(k) for k in feature_keys if str(k)}
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
    work["row_id"] = [f"row_{i}" for i in range(len(work))]
    feature_frames: list[pd.DataFrame] = []
    lookup: dict[str, pd.Series] = {}
    for ts, group in work.groupby("timestamp", sort=True):
        symbols = list(dict.fromkeys(group["symbol_norm"].astype(str).tolist()))
        rows = get_features_for_candidates(feature_panels, symbols, ts=ts)
        if rows.empty:
            continue
        for _, sample in group.iterrows():
            symbol = str(sample["symbol_norm"])
            row_id = str(sample["row_id"])
            if symbol not in rows.index:
                continue
            row = rows.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            frame = row.to_frame().T
            frame.index = pd.Index([row_id], name="row_id")
            if inject_historical_context:
                frame["__symbol__"] = symbol
                frame["__ts__"] = pd.Timestamp(ts)
            feature_frames.append(frame)
            lookup[row_id] = sample
    if feature_frames:
        X = pd.concat(feature_frames, axis=0, copy=False)
    else:
        X = pd.DataFrame()

    alpha = pd.Series(index=X.index, data=np.nan, dtype=float)
    meta = pd.Series(index=X.index, data=np.nan, dtype=float)
    base_error = ""
    meta_error = ""
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
            finite_alpha_index = alpha.replace([np.inf, -np.inf], np.nan).dropna().index
            if len(finite_alpha_index) > 0:
                meta_base = X.reindex(finite_alpha_index).copy()
                meta_base[strategy_id] = alpha.reindex(meta_base.index)
                for start in range(0, len(meta_base), chunk_rows):
                    chunk = meta_base.iloc[start : start + chunk_rows]
                    pred = orchestrator.predict_meta(chunk, side=side, kind=strategy_id)
                    parts.append(pred.reindex(chunk.index))
                if parts:
                    meta = pd.concat(parts).reindex(X.index)
        except Exception as exc:
            meta_error = str(exc)

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
                "has_feature_row": row_id in lookup,
                "feature_cols": int(X.shape[1]) if row_id in lookup else 0,
                "base_pred": base_pred,
                "meta_pred": meta_pred,
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
    parser.add_argument("--policy-artifact-run-id", default="20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--model-artifact-run-id", default="")
    parser.add_argument("--feature-run-id", default="20260605_070000")
    parser.add_argument("--strategy-id", action="append", default=[])
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--sample-position", choices=("head", "tail"), default="tail")
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
    args = parser.parse_args()

    os.environ.setdefault("EPM_EXCHANGE", "kraken")
    os.environ.setdefault("EXCHANGE_NAME", "kraken")
    os.environ.setdefault("PRIMARY_EXCHANGE", "kraken")
    os.environ.setdefault("EPM_MARKET_MODE", "perps")

    data_root = args.data_root
    policy_run_id = str(args.policy_artifact_run_id)
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
    runtime_cfg["inference_lgbm_internal_diagnostics_enabled"] = True
    runtime_cfg["inference_model_timing_enabled"] = False
    runtime_cfg["preserve_logged_meta_model_derived_features"] = False
    orchestrator = ModelOrchestrator(
        state,
        runtime_cfg={"model_bundle": state.get("bundle", {}), **runtime_cfg},
    )
    tprint(
        "Resolved live model stack: "
        f"alpha_heads={len(orchestrator.alpha_by_strategy)} "
        f"meta_heads={len(orchestrator.meta_models)}"
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
        "use_legacy_sizer_calibration": bool(use_legacy_sizer_calibration),
        "strategies": {},
    }

    for strategy_id in strategy_ids:
        tprint(f"Processing strategy={strategy_id}")
        side = _strategy_side(strategy_id)
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
        preds.to_csv(strategy_dir / "live_finalfit_policy_oos_predictions.csv", index=False)
        feature_snapshot.to_parquet(strategy_dir / "feature_vector_parity.parquet", index=False)
        feature_snapshot.to_csv(strategy_dir / "feature_vector_parity.csv", index=False)
        summary = _summary(strategy_id, feature_snapshot, preds, audit)
        summary["policy_rank_reference_path"] = str(ref_path)
        summary["alpha_selected_feature_count"] = len(alpha_features)
        summary["meta_selected_feature_count"] = len(meta_features)
        summary["selected_feature_count"] = len(selected_features)
        summary["raw_selected_feature_count"] = len(raw_selected_features)
        summary["sample_filter_audit"] = sample_filter_audit
        summary["inject_historical_context"] = bool(args.inject_historical_context)
        (strategy_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        all_summaries["strategies"][strategy_id] = summary

    (out_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tprint(f"Wrote live final-fit OOS export summary to {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
