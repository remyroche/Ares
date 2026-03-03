import os
import pickle
import glob
import time
import json
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from extreme_price_movements.pnl import CostModel, trade_return_net
from extreme_price_movements.pnl_asserts import assert_pos_w, assert_units
from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.data_store import (
    load_features,
    load_features_selected,
    save_features,
    save_artifact_df,
    load_artifact_df,
    to_panel,
    get_feature_bounds,
)
from extreme_price_movements.training import generate_label_datasets, generate_exhaustion_history, optimize_risk_params, train_models_from_artifacts
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.universe import get_training_universe, refresh_margin_universe_daily
from extreme_price_movements.engine import simulate_trade_hourly, generate_hourly_signals, _build_side_score_df
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.reports.report_generator import generate_training_report, generate_risk_report, generate_backtest_report
from extreme_price_movements.telemetry.tprint_hooks import emit_bucket_summary, emit_run_header
from extreme_price_movements.ridge_position_sizer import (
    RidgePositionSizer,
    run_ridge_position_sizer_step,
    load_meta_oof_predictions,
    load_trade_outcomes_from_oof,
)
from extreme_price_movements.offline_optimisers.params_store import apply_offline_optimizer_best_params
from extreme_price_movements.reports.bucket_report import (
    report_labels,
    report_base_training,
    report_meta_training,
    report_ridge_sizer,
    report_optimise,
)
from extreme_price_movements.entry_policy import compute_entry_policy_decision, flatten_bucket_policy

def _meta_feature_keys_union(cfg) -> set[str]:
    keys = set(cfg.get("meta_feature_keys", []) or [])
    keys.update(cfg.get("mr_meta_feature_keys", []) or [])
    keys.update(cfg.get("tf_meta_feature_keys", []) or [])
    return {k for k in keys if isinstance(k, str) and k}


def _expected_feature_keys_from_cfg(cfg) -> set[str]:
    keys: set[str] = set()
    for name in (
        "exh_feature_keys",
        "spike_feature_keys",
        "tf_feature_keys",
        "mr_feature_keys",
        "test_feature_keys",
    ):
        vals = cfg.get(name, [])
        if isinstance(vals, (list, tuple)):
            for v in vals:
                if isinstance(v, str) and v:
                    keys.add(v)
    keys.update(_meta_feature_keys_union(cfg))
    # Core features that downstream logic assumes are present.
    keys.update({"atr_pct", "ret1h", "ret24h"})
    return keys


def _align_features_to_panel(feats: dict, panel: dict[str, pd.DataFrame], symbols: list[str]) -> dict:
    close = panel["close"]
    out = {}
    keys = list(feats.keys())
    for k in keys:
        df = feats.pop(k)
        if not isinstance(df, pd.DataFrame):
            continue
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            if idx.tz is None:
                df.index = idx.tz_localize("UTC")
            else:
                df.index = idx.tz_convert("UTC")
        out[k] = df.reindex(index=close.index, columns=symbols).astype(np.float32, copy=False)
        del df
    import gc as _gc
    _gc.collect()
    return out


def _ensure_atr_pct_feature(
    feats: dict,
    panel: dict[str, pd.DataFrame],
    cfg: dict,
    symbols: list[str] | None = None,
) -> dict:
    """
    Ensure feats['atr_pct'] exists with coverage for requested symbols.
    Backfills missing/all-NaN symbol columns from panel OHLC using a causal ATR%.
    """
    if feats is None:
        return feats
    if not isinstance(panel, dict) or any(k not in panel for k in ("high", "low", "close")):
        return feats

    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    if not isinstance(high, pd.DataFrame) or not isinstance(low, pd.DataFrame) or not isinstance(close, pd.DataFrame):
        return feats

    target_syms = list(symbols) if symbols is not None else list(close.columns)
    if not target_syms:
        return feats

    get_fn = getattr(feats, "get", None)
    if get_fn is None:
        return feats

    atr_existing = get_fn("atr_pct")
    if not isinstance(atr_existing, pd.DataFrame):
        atr_existing = pd.DataFrame(index=close.index)
    else:
        atr_existing = atr_existing.reindex(index=close.index)

    missing_syms = []
    for s in target_syms:
        if s not in atr_existing.columns:
            missing_syms.append(s)
        else:
            col = pd.to_numeric(atr_existing[s], errors="coerce")
            if bool(col.isna().all()):
                missing_syms.append(s)

    if not missing_syms:
        atr_ready = atr_existing.reindex(columns=target_syms).astype(np.float32, copy=False)
        try:
            feats["atr_pct"] = atr_ready
        except Exception:
            if hasattr(feats, "_assembled"):
                feats._assembled["atr_pct"] = atr_ready
        return feats

    atr_n = int(cfg.get("atr_n", 14))
    atr_n = max(2, atr_n)
    alpha = 1.0 / float(atr_n)

    h = high.reindex(columns=missing_syms).astype(np.float32, copy=False)
    l = low.reindex(columns=missing_syms).astype(np.float32, copy=False)
    c = close.reindex(columns=missing_syms).astype(np.float32, copy=False)
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    atr_pct = (atr / (c.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    if missing_syms:
        # Use combine_first to avoid duplicate columns if there's any overlap
        atr_out = atr_pct.combine_first(atr_existing)
    else:
        atr_out = atr_existing

    atr_ready = atr_out.reindex(columns=target_syms).astype(np.float32, copy=False)
    try:
        feats["atr_pct"] = atr_ready
    except Exception:
        if hasattr(feats, "_assembled"):
            feats._assembled["atr_pct"] = atr_ready
    tprint(f"Backfilled atr_pct from panel OHLC for {len(missing_syms)} symbols.")
    return feats


def _feature_structural_gaps(
    feats: dict,
    expected_keys: set[str],
    ref_index: pd.DatetimeIndex,
    ref_symbols: list[str],
) -> tuple[list[str], list[str]]:
    """Return (missing_keys, partial_keys) vs reference period/symbol universe."""
    if not isinstance(feats, dict) or not feats:
        return sorted(expected_keys), []

    missing: list[str] = []
    partial: list[str] = []
    ref_syms = set(map(str, ref_symbols))
    ref_start = ref_index.min() if len(ref_index) else None
    ref_end = ref_index.max() if len(ref_index) else None

    for k in sorted(expected_keys):
        if k not in feats or not isinstance(feats.get(k), pd.DataFrame):
            missing.append(k)
            continue
        df = feats[k]
        have_syms = set(map(str, df.columns))
        if not ref_syms.issubset(have_syms):
            partial.append(k)
            continue
        if ref_start is not None and ref_end is not None:
            if len(df.index) == 0:
                partial.append(k)
                continue
            if df.index.min() > ref_start or df.index.max() < ref_end:
                partial.append(k)
                continue
    return missing, partial


def _generate_feature_health_reports(ts_sig: pd.Timestamp, data_root: str) -> dict | None:
    """
    Build feature quality reports from saved per-symbol parquet files.
    Outputs:
      - feature_health_symbol_summary.csv
      - feature_health_feature_detail.csv
    """
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(data_root, "features", run_id)
    files = sorted(glob.glob(os.path.join(in_dir, "symbol=*.parquet")))
    if not files:
        tprint(f"Feature health report: no feature files in {in_dir}")
        return None

    out_dir = os.path.join(data_root, "artifacts", run_id, "features")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "feature_health_symbol_summary.csv")
    detail_path = os.path.join(out_dir, "feature_health_feature_detail.csv")

    symbol_rows: list[dict] = []
    detail_rows: list[dict] = []

    total_files = len(files)
    progress_every = 25 if total_files >= 200 else 10

    for i, fpath in enumerate(files, start=1):
        try:
            df = pd.read_parquet(fpath)
        except Exception as exc:
            tprint(f"Feature health report: failed to read {fpath}: {exc}")
            continue

        if "__symbol__" in df.columns and not df.empty:
            symbol = str(df["__symbol__"].iloc[0])
            feat_cols = [c for c in df.columns if c != "__symbol__"]
        else:
            symbol = os.path.basename(fpath).replace("symbol=", "").replace(".parquet", "").replace("_", "/", 1)
            feat_cols = list(df.columns)

        rows = int(len(df))
        n_features = int(len(feat_cols))
        if rows == 0 or n_features == 0:
            symbol_rows.append(
                {
                    "symbol": symbol,
                    "rows": rows,
                    "n_features": n_features,
                    "nan_cells": 0,
                    "nan_pct_overall": 0.0,
                    "all_nan_features": 0,
                    "constant_features": 0,
                }
            )
            continue

        feat_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        nan_counts = feat_df.isna().sum(axis=0).astype(np.int64)
        uniq_counts = feat_df.nunique(dropna=True).astype(np.int64)
        all_nan_mask = nan_counts == rows
        const_mask = (uniq_counts <= 1) & (~all_nan_mask)

        nan_cells = int(nan_counts.sum())
        total_cells = int(rows * n_features)
        symbol_rows.append(
            {
                "symbol": symbol,
                "rows": rows,
                "n_features": n_features,
                "nan_cells": nan_cells,
                "nan_pct_overall": float(100.0 * nan_cells / max(total_cells, 1)),
                "all_nan_features": int(all_nan_mask.sum()),
                "constant_features": int(const_mask.sum()),
            }
        )

        for c in feat_cols:
            nan_c = int(nan_counts[c])
            detail_rows.append(
                {
                    "symbol": symbol,
                    "feature": c,
                    "rows": rows,
                    "nan_count": nan_c,
                    "nan_pct": float(100.0 * nan_c / max(rows, 1)),
                    "is_all_nan": bool(all_nan_mask[c]),
                    "is_constant_non_nan": bool(const_mask[c]),
                }
            )

        if i % progress_every == 0 or i == total_files:
            tprint(
                f"Feature health report progress: {i}/{total_files} files "
                f"({(i / total_files) * 100:.1f}%)"
            )

    if not symbol_rows:
        tprint("Feature health report: no rows produced.")
        return None

    summary_df = pd.DataFrame(symbol_rows).sort_values("symbol")
    detail_df = pd.DataFrame(detail_rows).sort_values(["symbol", "feature"])
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    n_all_nan_features = int(detail_df["is_all_nan"].sum()) if not detail_df.empty else 0
    n_const_features = int(detail_df["is_constant_non_nan"].sum()) if not detail_df.empty else 0
    tprint(
        f"Feature health report saved: {summary_path}, {detail_path} | "
        f"symbols={len(summary_df)} all_nan_feature_rows={n_all_nan_features} "
        f"constant_feature_rows={n_const_features}"
    )

    return {
        "summary_path": summary_path,
        "detail_path": detail_path,
        "symbols": int(len(summary_df)),
        "all_nan_feature_rows": n_all_nan_features,
        "constant_feature_rows": n_const_features,
    }


def _scan_feature_cache_light(
    ts_sig: pd.Timestamp,
    data_root: str,
    expected_keys: set[str],
    panel_close: pd.DataFrame,
) -> dict | None:
    """
    Lightweight cache scan:
    - avoids loading full feature matrices
    - checks expected keys against parquet schemas
    - checks symbol/time coverage using per-file metadata bounds
    """
    ts_str = ts_sig.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(data_root, "features", ts_str)
    files = sorted(glob.glob(os.path.join(in_dir, "symbol=*.parquet")))
    if not files:
        return None

    ref_symbols = [str(s) for s in panel_close.columns]
    required_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for s in ref_symbols:
        ser = panel_close[s]
        valid_idx = ser.index[ser.notna()]
        if len(valid_idx) == 0:
            continue
        required_bounds[s] = (pd.Timestamp(valid_idx[0]), pd.Timestamp(valid_idx[-1]))

    key_symbol_counts: dict[str, int] = {k: 0 for k in expected_keys}
    present_symbols: set[str] = set()
    uncovered_symbols: set[str] = set()

    total_files = len(files)
    progress_every = 100 if total_files >= 500 else 50
    for i, fpath in enumerate(files, start=1):
        fname = os.path.basename(fpath)
        sym = fname.replace("symbol=", "").replace(".parquet", "").replace("_", "/", 1)

        if sym in required_bounds:
            present_symbols.add(sym)
            req_first, req_last = required_bounds[sym]
            first_ts, last_ts = get_feature_bounds(fpath)
            if first_ts is None or last_ts is None or first_ts > req_first or last_ts < req_last:
                uncovered_symbols.add(sym)

        try:
            schema_names = set(pq.ParquetFile(fpath).schema.names)
        except Exception:
            schema_names = set()
        feat_cols = [c for c in schema_names if c in expected_keys]
        for c in feat_cols:
            key_symbol_counts[c] += 1

        if i % progress_every == 0 or i == total_files:
            tprint(
                f"Feature cache scan progress: {i}/{total_files} files "
                f"({(i / total_files) * 100:.1f}%)"
            )

    required_set = set(required_bounds.keys())
    missing_symbols = sorted(required_set - present_symbols)
    required_n = len(required_set)

    missing_keys: list[str] = []
    partial_keys: set[str] = set()
    for k in sorted(expected_keys):
        present_n = int(key_symbol_counts.get(k, 0))
        if present_n <= 0:
            missing_keys.append(k)
        elif present_n < required_n:
            partial_keys.add(k)

    # If some symbols have missing files or incomplete time bounds, all present keys are partial.
    if missing_symbols or uncovered_symbols:
        partial_keys.update(set(expected_keys) - set(missing_keys))

    available_key_count = sum(1 for k in expected_keys if key_symbol_counts.get(k, 0) > 0)
    return {
        "in_dir": in_dir,
        "file_count": total_files,
        "required_symbol_count": required_n,
        "available_key_count": available_key_count,
        "missing_symbols": missing_symbols,
        "uncovered_symbols": sorted(uncovered_symbols),
        "missing_keys": missing_keys,
        "partial_keys": sorted(partial_keys),
    }


def _build_tail_only_backfill_cutoffs(
    ts_sig: pd.Timestamp,
    data_root: str,
    panel_close: pd.DataFrame,
    backfill_keys: list[str],
) -> tuple[dict[str, pd.Timestamp], dict[str, int]]:
    """
    Build per-symbol cutoff timestamps for tail-only backfill writes.

    A symbol is tail-only eligible iff:
    - symbol parquet exists,
    - all backfill keys already exist in that symbol parquet schema,
    - feature coverage starts at/before required start,
    - and ends before required end (missing only at tail).

    Structural/interior gaps are excluded (no cutoff), which forces full write.
    """
    ts_str = ts_sig.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(data_root, "features", ts_str)
    backfill_set = set(backfill_keys)
    cutoffs: dict[str, pd.Timestamp] = {}

    stats = {
        "eligible_tail_only": 0,
        "missing_symbol_file": 0,
        "missing_backfill_columns": 0,
        "structural_or_interior": 0,
        "already_covered": 0,
    }

    if not backfill_set:
        return cutoffs, stats

    for sym in panel_close.columns:
        sym = str(sym)
        ser = panel_close[sym]
        valid_idx = ser.index[ser.notna()]
        if len(valid_idx) == 0:
            continue
        req_first = pd.Timestamp(valid_idx[0])
        req_last = pd.Timestamp(valid_idx[-1])

        safe_sym = sym.replace("/", "_")
        fpath = os.path.join(in_dir, f"symbol={safe_sym}.parquet")
        if not os.path.exists(fpath):
            stats["missing_symbol_file"] += 1
            continue

        try:
            schema_names = set(pq.ParquetFile(fpath).schema.names)
        except Exception:
            stats["structural_or_interior"] += 1
            continue

        if not backfill_set.issubset(schema_names):
            stats["missing_backfill_columns"] += 1
            continue

        first_ts, last_ts = get_feature_bounds(fpath)
        if first_ts is None or last_ts is None:
            stats["structural_or_interior"] += 1
            continue

        if first_ts > req_first:
            # Interior/leading gap: must rewrite full history.
            stats["structural_or_interior"] += 1
            continue

        if last_ts >= req_last:
            # Already covered for this symbol and key set.
            stats["already_covered"] += 1
            continue

        cutoffs[sym] = pd.Timestamp(last_ts)
        stats["eligible_tail_only"] += 1

    return cutoffs, stats


def _local_store_symbols(store) -> list[str]:
    """Best-effort local symbol discovery from partitioned OHLCV store."""
    import glob
    syms: list[str] = []
    ohlcv_dir = getattr(store, "ohlcv_dir", None)
    if not ohlcv_dir:
        return syms
    for path in glob.glob(os.path.join(ohlcv_dir, "symbol=*")):
        base = os.path.basename(path)
        if not base.startswith("symbol="):
            continue
        raw = base.replace("symbol=", "")
        syms.append(raw.replace("_", "/", 1))
    return sorted(set(syms))


def run_label_generation_step_v2(ts_sig, margin_symbols, cfg, store, ex, horizons=None):
    cfg = apply_offline_optimizer_best_params(dict(cfg))
    tprint("STEP: LABEL GENERATION START")
    _labels_use_store_universe = False
    # Labels should be able to run fully offline from local store contents.
    # If margin_symbols is not provided, source it directly from store instead of
    # triggering a network refresh of margin universe.
    if margin_symbols is None:
        margin_symbols = _local_store_symbols(store)
        if margin_symbols:
            tprint(f"Labels universe source: local store symbols ({len(margin_symbols)})")
            _labels_use_store_universe = True
        else:
            tprint("Labels universe source: local store symbols empty; falling back to market basket")
            margin_symbols = list(cfg.get("market_basket", []))
    if _labels_use_store_universe:
        from extreme_price_movements.optimization_utils import filter_low_variance_assets
        syms_all = sorted(set(margin_symbols).union(set(cfg.get("market_basket", []))))
        tprint(f"Labels offline universe build: using local store symbols + basket ({len(syms_all)} symbols)")
        train_syms = filter_low_variance_assets(
            store,
            syms_all,
            lookback_days=30,
            threshold_pct=cfg["variance_filter_pct"],
            ts_sig=ts_sig,
        )
        train_syms = sorted(list(set(train_syms).union(set(cfg.get("market_basket", [])))))
    else:
        train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    tprint(f"Universe: {len(train_syms)} symbols")

    # Load Data & Features
    dfs = {}

    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    with Timer("Data Load"):
        for s in train_syms:
            df = store.load(s)
            if not df.empty: dfs[s] = df[df.index <= ts_sig].tail(24*lookback_days)

    if bool(cfg.get("label_diagnostics_mode", False)) and dfs:
        _lens = np.array([len(df) for df in dfs.values()], dtype=np.int64)
        _mins = [df.index.min() for df in dfs.values() if len(df) > 0]
        _maxs = [df.index.max() for df in dfs.values() if len(df) > 0]
        _ts_min_all = min(_mins) if _mins else None
        _ts_max_all = max(_maxs) if _maxs else None
        tprint(
            "[LABEL_DIAG][SYMBOL_HISTORY_PRE_HOLDOUT] "
            f"symbols={len(dfs)} bars_min={int(_lens.min())} bars_med={int(np.median(_lens))} bars_max={int(_lens.max())} "
            f"ts_min={_ts_min_all} ts_max={_ts_max_all}"
        )

    if not dfs:
        tprint("No data available.")
        return

    panel = to_panel(dfs)

    # Data coverage diagnostics
    _close = panel["close"]
    _ts_min = _close.index.min()
    _ts_max = _close.index.max()
    _n_hours = len(_close)
    _n_days = (_ts_max - _ts_min).total_seconds() / 86400 if _ts_max > _ts_min else 0
    _n_syms = _close.shape[1]
    _non_nan_pct = float(_close.notna().sum().sum()) / max(_close.size, 1) * 100
    tprint(f"DATA COVERAGE: {_n_syms} symbols, {_n_hours} hourly bars, "
           f"{_n_days:.0f} days ({_ts_min.date()} to {_ts_max.date()}), "
           f"{_non_nan_pct:.1f}% non-NaN")
    if _n_days < 365:
        tprint(f"WARNING: Only {_n_days:.0f} days of data — recommend >= 365 days for robust training")

    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])

    # Keep feature-key selection consistent with the shared cache/feature generation logic.
    label_feature_keys = _expected_feature_keys_from_cfg(cfg)
    feats = load_features_selected(
        ts_sig,
        cfg["data_root"],
        feature_keys=sorted(label_feature_keys),
        symbols=train_syms,
    )
    if feats is None or len(feats) == 0:
        tprint("ERROR: Features not found. Run feature_generation first.")
        return
    feats = _ensure_atr_pct_feature(feats, panel, cfg, symbols=train_syms)

    # Restrict to symbols present in both panel and features
    sample_feat = next(iter(feats.values()))
    feat_syms = set(sample_feat.columns)
    panel_syms = set(panel["close"].columns)
    valid_syms = sorted(feat_syms & panel_syms & set(train_syms))
    tprint(f"Symbol intersection: {len(valid_syms)} (feats={len(feat_syms)}, panel={len(panel_syms)}, universe={len(train_syms)})")
    if not valid_syms:
        tprint("ERROR: No overlapping symbols between features and panel.")
        return
    train_syms = valid_syms

    # Restrict panel to valid symbols and hard-align all feature frames to panel.
    panel = {k: v.reindex(columns=valid_syms) for k, v in panel.items() if isinstance(v, pd.DataFrame)}
    feats = _align_features_to_panel(feats, panel, valid_syms)

    # 1. Exhaustion History
    p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], train_syms)

    # Save Exhaustion History
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    save_artifact_df(p_exh_hist, cfg["data_root"], run_id, "labels", "exhaustion_history")

    # 2. Label Datasets
    horizons = horizons or cfg.get("label_horizons_hours", [2, 4, 8])
    datasets = generate_label_datasets(panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist, horizons=horizons)

    for name, df in datasets.items():
        save_artifact_df(df, cfg["data_root"], run_id, "labels", name)

    try:
        rp = report_labels(run_id, cfg["data_root"], cfg, base_dir=cfg.get('reports_root'))
        tprint(f"Label bucket report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: label bucket report failed: {_re}")
    tprint("STEP: LABEL GENERATION COMPLETE")

def run_training_step(ts_sig, cfg, store=None, margin_symbols=None, base_only=False):
    """Train all models from label artifacts. Saves trained state to disk."""
    cfg = apply_offline_optimizer_best_params(dict(cfg))
    tprint(f"STEP: MODEL TRAINING START (base_only={base_only})")

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    datasets = {}

    # Ensure we always have a margin universe for downstream specialist training
    if margin_symbols is None:
        try:
            margin_cache = refresh_margin_universe_daily(None, quotes=cfg.get("margin_quotes", ("USDT", "USDC", "BUSD", "EUR")))
            margin_symbols = margin_cache.symbols if margin_cache else []
        except Exception as exc:
            tprint(f"WARNING: Failed to refresh margin universe ({exc}); proceeding without specialist training")
            margin_symbols = []

    # 1. Load label artifacts
    tprint("Loading label datasets from artifacts...")

    # Spike anatomy
    missing_spike = []
    for mode in ["best", "worst"]:
        name = f"spike_anatomy_{mode}"
        df_spike = load_artifact_df(cfg["data_root"], run_id, "labels", name)
        if df_spike is not None:
            datasets[name] = df_spike
        else:
            missing_spike.append(mode)

    if missing_spike:
        tprint(f"Adding Missing Spike artifacts: {missing_spike} (Generating in-memory...)")
        if store is None:
            tprint("ERROR: store is None, cannot generate missing spike artifacts.")
            # Critical failure if we can't generate
        else:
            # Need features and panel. Load them.
            # Mirror run_label_generation_step_v2 logic roughly but localized
            tprint("Loading features and panel for Spike Anatomy generation...")
            train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
            spike_feature_keys = set(cfg.get("spike_feature_keys", []))
            spike_feature_keys.update(_meta_feature_keys_union(cfg))
            spike_feature_keys.update({"atr_pct", "ret1h", "ret24h"})
            feats = load_features_selected(
                ts_sig,
                cfg["data_root"],
                feature_keys=sorted(spike_feature_keys),
                symbols=train_syms,
            )
            if feats is None or len(feats) == 0:
                tprint("ERROR: Features not found.")
            else:
                dfs = {}
                lookback_days = max(90, int(cfg["fetch_years"] * 365))
                for s in train_syms:
                    df = store.load(s)
                    if not df.empty: dfs[s] = df[df.index <= ts_sig].tail(24*lookback_days)
                
                if dfs:
                    panel = to_panel(dfs)
                    feats = _ensure_atr_pct_feature(feats, panel, cfg, symbols=train_syms)
                    mkt_df = compute_market_features(panel, cfg["market_basket"])
                    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
                    
                    # Intersect symbols
                    sample_feat = next(iter(feats.values()))
                    valid_syms = sorted(set(sample_feat.columns) & set(panel["close"].columns) & set(train_syms))
                    panel = {k: v[valid_syms] for k, v in panel.items() if isinstance(v, pd.DataFrame)}
                    
                    from extreme_price_movements.training import train_spike_anatomy_model
                    
                    for mode in missing_spike:
                        tprint(f"Generating Spike Anatomy ({mode})...")
                        df_spike = train_spike_anatomy_model(panel, feats, mkt_gates, cfg, valid_syms, ts_sig, mode=mode)
                        if df_spike is not None:
                            datasets[f"spike_anatomy_{mode}"] = df_spike
                            save_artifact_df(df_spike, cfg["data_root"], run_id, "labels", f"spike_anatomy_{mode}")
                            tprint(f"Saved generated spike artifact: spike_anatomy_{mode}")
    
    # Alpha models (long/short × mr/tf × horizons)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    horizons = cfg["label_horizons_hours"]

    found_count = 0
    for side in trade_sides:
        for k in kinds:
            for H in horizons:
                name = f"train_{side}_{k}_{H}"
                df = load_artifact_df(cfg["data_root"], run_id, "labels", name)
                if df is not None:
                    datasets[name] = df
                    found_count += 1
                    tprint(f"  Loaded {name}: {len(df)} rows")

    # Exhaustion models
    for d in ["up", "down"]:
        name = f"exh_{d}"
        df = load_artifact_df(cfg["data_root"], run_id, "labels", name)
        if df is not None:
            datasets[name] = df
            tprint(f"  Loaded {name}: {len(df)} rows")

    # Specialist models
    for name in ["trap_model", "gamma_model"]:
        df = load_artifact_df(cfg["data_root"], run_id, "labels", name)
        if df is not None:
            datasets[name] = df
            tprint(f"  Loaded {name}: {len(df)} rows")

    if not found_count:
        tprint("ERROR: No alpha label datasets found. Run 'labels' mode first.")
        return None

    tprint(f"Loaded {len(datasets)} datasets total.")

    # 2. Train models
    # Inject run_id so meta OOF files are saved to the correct artifacts directory
    cfg["run_id"] = run_id
    # Selector hysteresis can warm-start from a previous run when available.
    if not cfg.get("prev_run_id"):
        try:
            _art_dir = os.path.join(cfg["data_root"], "artifacts")
            _prev_runs = sorted(
                [
                    d for d in os.listdir(_art_dir)
                    if d != run_id and os.path.isdir(os.path.join(_art_dir, d))
                ],
                reverse=True,
            )
            if _prev_runs:
                cfg["prev_run_id"] = _prev_runs[0]
                tprint(f"Selector warm-start previous run: {cfg['prev_run_id']}")
        except Exception as _e_prev:
            tprint(f"Selector warm-start discovery skipped: {_e_prev}")
    with Timer("Model Training"):
        trained_bundle = train_models_from_artifacts(datasets, cfg, train_meta=not base_only)
        alpha_metrics = trained_bundle.get("alpha_oof_metrics", {}) if trained_bundle else {}
    
    # Specialist Models are now trained inside train_models_from_artifacts
    # using artifacts loaded above.


    # 3. Save trained state
    # Populate granular_risk with sensible defaults for all 8 bucket keys
    # so the backtest engine doesn't fall back to global cfg with warnings.
    # MR buckets: tighter SL, shorter hold. TF buckets: wider TP, longer hold.
    _mr_risk = {
        "tp_mult": cfg.get("tp_mult", 0.50),
        "sl_mult": cfg.get("sl_mult", 0.18),
        "trail_mult": cfg.get("trail_mult", 0.25),
        "k_sl": cfg.get("risk_k_sl", 2.0),
        "k_trail_start": cfg.get("risk_k_trail_start", 1.0),
        "k_trail_dist": cfg.get("risk_k_trail_dist", 1.0),
        "max_hold_hours": 12,
    }
    _tf_risk = {
        "tp_mult": cfg.get("tp_mult", 0.50) * 1.2,
        "sl_mult": cfg.get("sl_mult", 0.18),
        "trail_mult": cfg.get("trail_mult", 0.25),
        "k_sl": cfg.get("risk_k_sl", 2.0),
        "k_trail_start": cfg.get("risk_k_trail_start", 1.0),
        "k_trail_dist": cfg.get("risk_k_trail_dist", 1.0),
        "max_hold_hours": 24,
    }
    _granular = {
        "risk_mr_best": _mr_risk, "risk_mr_worst": _mr_risk,
        "risk_tf_best": _tf_risk, "risk_tf_worst": _tf_risk,
        "risk_long_mr": _mr_risk, "risk_short_mr": _mr_risk,
        "risk_long_tf": _tf_risk, "risk_short_tf": _tf_risk,
    }
    default_risk = {
        "k_sl": cfg.get("risk_k_sl", 2.0),
        "k_trail_start": cfg.get("risk_k_trail_start", 1.0),
        "k_trail_dist": cfg.get("risk_k_trail_dist", 0.5),
        "granular_risk": _granular,
    }

    state = {
        "ts_trained": ts_sig,
        "bundle": trained_bundle,
        "risk_params": default_risk
    }

    state_dir = os.path.join(cfg["data_root"], "artifacts", run_id, "models")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, "trained_state.pkl")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    tprint(f"Saved trained state to {state_path}")

    # Log summary
    bundle = trained_bundle
    if bundle:
        alpha = bundle.get("alpha_models", {})
        for side in trade_sides:
            for k in kinds:
                m = alpha.get(side, {}).get(k)
                if m:
                    tprint(f"  {side} {k}: H={m['H']}, features={len(m['feat_cols'])}")
                else:
                    tprint(f"  {side} {k}: NO MODEL")

        spike = bundle.get("spike_models", {})
        for mode in ["best", "worst"]:
            m = spike.get(mode)
            tprint(f"  spike_{mode}: {'fitted' if m else 'NO MODEL'}")
            if m and "oof_scores" in m:
                oof_df = m["oof_scores"]
                save_artifact_df(oof_df, cfg["data_root"], run_id, "labels", f"spike_oof_{mode}")
                tprint(f"  Saved OOF scores: spike_oof_{mode}")

        exh = bundle.get("exh_models", {})
        for d in ["up", "down"]:
            m = exh.get(d)
            tprint(f"  exh_{d}: {'fitted' if m and m.model else 'NO MODEL'}")

        meta = bundle.get("meta_models", {})
        for side in trade_sides:
            for k in kinds:
                key = f"{side}_{k}"
                m = meta.get(key)
                tprint(f"  meta_{key}: {'fitted' if m and m.model else 'NO MODEL'}")

    # Generate training report
    try:
        report_path = generate_training_report(
            run_id=run_id,
            cfg=cfg,
            bundle=bundle or {},
            datasets=datasets or {},
            specialist_models=bundle.get("specialist_models") if bundle else None,
            extra_info=alpha_metrics,
            base_dir=cfg.get('reports_root'),
        )
        tprint(f"Training report saved to {report_path}")
    except Exception as e:
        tprint(f"WARNING: Failed to generate training report: {e}")

    # Per-bucket/horizon detailed reports
    try:
        rp = report_base_training(run_id, bundle or {}, cfg, base_dir=cfg.get('reports_root'))
        tprint(f"Base training bucket report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: base training bucket report failed: {_re}")
    try:
        rp = report_meta_training(run_id, cfg["data_root"], bundle or {}, cfg, base_dir=cfg.get('reports_root'))
        tprint(f"Meta training bucket report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: meta training bucket report failed: {_re}")

    tprint("STEP: MODEL TRAINING COMPLETE")
    return state


def run_ridge_sizer_step(ts_sig, cfg, state_file):
    """Run ridge position sizer to learn optimal meta model combination weights.
    
    Processes each bucket (long_mr, long_tf, short_mr, short_tf) separately,
    combining per-horizon regressors (H1, H2, H4) + classifier + agreement
    features into a single Ridge combiner per bucket.
    
    Args:
        ts_sig: Timestamp for the training run
        cfg: Configuration dictionary
        state_file: Path to the trained state file
        
    Returns:
        Dict with ridge sizer weights and metrics per bucket, or None if failed
    """
    tprint("STEP: RIDGE POSITION SIZER START")
    
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    data_root = cfg.get("data_root", "data")
    
    # Use the per-bucket loader from run_ridge_sizer which handles per-horizon
    # grouping and agreement features
    from extreme_price_movements.run_ridge_sizer import (
        load_meta_oof_predictions as load_bucket_oofs,
        load_trade_outcomes,
    )
    
    # IV. Dynamically load latest tpsl_optimiser params for exit policy alignment
    import json as _json
    _tpsl_params = {}
    _bp_path = os.path.join(data_root, "artifacts", run_id, "models", "bucket_params.json")
    if os.path.exists(_bp_path):
        try:
            with open(_bp_path) as _f:
                _bp_data = _json.load(_f)
            _tpsl_params = _bp_data.get("buckets", {})
            tprint(f"  Loaded tpsl_optimiser params from {_bp_path} ({len(_tpsl_params)} buckets)")
        except Exception as _e:
            tprint(f"  WARNING: Could not load tpsl params: {_e}")
    else:
        # Try previous run's params
        _art_dir = os.path.join(data_root, "artifacts")
        if os.path.isdir(_art_dir):
            _prev_runs = sorted([d for d in os.listdir(_art_dir) if d != run_id and os.path.isdir(os.path.join(_art_dir, d))], reverse=True)
            for _prev_id in _prev_runs:
                _prev_bp = os.path.join(_art_dir, _prev_id, "models", "bucket_params.json")
                if os.path.exists(_prev_bp):
                    try:
                        with open(_prev_bp) as _f:
                            _bp_data = _json.load(_f)
                        _tpsl_params = _bp_data.get("buckets", {})
                        tprint(f"  Loaded tpsl_optimiser params from previous run {_prev_id} ({len(_tpsl_params)} buckets)")
                    except Exception:
                        pass
                    break
        if not _tpsl_params:
            tprint("  No tpsl_optimiser params found, Ridge sizer will use default exit policy")
    
    try:
        bucket_oofs = load_bucket_oofs(data_root, run_id)
    except FileNotFoundError as e:
        tprint(f"WARNING: {e}")
        tprint("Skipping ridge sizer step - no meta OOF predictions found.")
        return None
    
    cost_pct = float(cfg.get("ridge_cost_pct", cfg.get("fee_bps", 50.0) / 10000.0))
    output_dir = os.path.join(data_root, "artifacts", run_id, "ridge_sizer")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Group buckets by direction: long_* = up-trend sizer, short_* = down-trend.
    # A single combined sizer dilutes IC because long IC (~0.031) and short IC
    # (~0.005) are incompatible, and short_mr has inverted IC (−0.011).
    # -------------------------------------------------------------------------
    _meta_cols = {
        "timestamp", "symbol", "return", "is_long", "index",
        "oof_u_hat", "oof_log_mae_q70_hat", "oof_log_mfe_hat",
        "mae_ret", "mfe_ret", "u_policy_net", "exit_code"
    }
    direction_groups = {"long": {}, "short": {}}
    for bucket_name, oof_preds in bucket_oofs.items():
        direction = "long" if bucket_name.startswith("long") else "short"
        direction_groups[direction][bucket_name] = oof_preds

    all_direction_results = {}

    for direction, dir_buckets in direction_groups.items():
        if not dir_buckets:
            continue
        tprint(f"  Ridge sizer — direction: {direction.upper()} ({list(dir_buckets.keys())})")

        # Each bucket has its own event set (different lengths).
        # Train one Ridge per bucket within the direction; store all weights
        # together in a single per-direction weight file.
        dir_weights: dict = {}
        dir_params: dict = {}
        dir_metrics: dict = {}

        for bucket_name, oof_preds in dir_buckets.items():
            try:
                trade_outcomes = load_trade_outcomes(data_root, run_id, oof_preds)
            except FileNotFoundError as e:
                tprint(f"    Skipping {bucket_name}: {e}")
                continue
            if "return" not in trade_outcomes.columns:
                tprint(f"    Skipping {bucket_name}: missing 'return' column")
                continue
            pred_cols = [c for c in oof_preds.columns if c not in _meta_cols]
            if not pred_cols:
                tprint(f"    Skipping {bucket_name}: no prediction columns")
                continue
            oof_pred_df = oof_preds[pred_cols].copy()
            timestamps = trade_outcomes['timestamp'].values if 'timestamp' in trade_outcomes.columns else None
            symbols    = trade_outcomes['symbol'].values    if 'symbol'    in trade_outcomes.columns else None
            tprint(f"    {bucket_name}: {len(oof_pred_df)} rows, features={pred_cols}")

            # Policy-aligned trade mask for ridge sizer (entry policy can reduce trade count).
            _bp_key = bucket_name.upper()
            _bp_cfg = flatten_bucket_policy(_tpsl_params.get(_bp_key, {})) if isinstance(_tpsl_params, dict) else {}
            if _bp_cfg.get("entry_policy"):
                _scores = np.asarray(oof_pred_df[pred_cols[0]].values, dtype=float)
                _atr_vec = np.asarray(
                    trade_outcomes.get("mae_ret", pd.Series(np.full(len(trade_outcomes), 0.02))).values,
                    dtype=float,
                )
                _atr_vec = np.clip(np.where(np.isfinite(_atr_vec), np.abs(_atr_vec), 0.02), 1e-4, 0.5)
                _mask = np.ones(len(trade_outcomes), dtype=bool)
                for _i in range(len(_mask)):
                    _pol = compute_entry_policy_decision(
                        entry_px=1.0,
                        atr_frac=float(_atr_vec[_i]),
                        score=float(_scores[_i]) if _i < len(_scores) else 0.0,
                        bucket_cfg=_bp_cfg,
                        features={
                            "u_hat_z": float(trade_outcomes.get("oof_u_hat", pd.Series(np.zeros(len(trade_outcomes)))).iloc[_i]) if "oof_u_hat" in trade_outcomes.columns else float(np.tanh(_scores[_i] if _i < len(_scores) else 0.0)),
                            "mae_hat_z": float(trade_outcomes.get("oof_log_mae_q70_hat", pd.Series(np.zeros(len(trade_outcomes)))).iloc[_i]) if "oof_log_mae_q70_hat" in trade_outcomes.columns else float(abs(np.tanh(_scores[_i] if _i < len(_scores) else 0.0))),
                            "mfe_hat_z": float(trade_outcomes.get("oof_log_mfe_hat", pd.Series(np.zeros(len(trade_outcomes)))).iloc[_i]) if "oof_log_mfe_hat" in trade_outcomes.columns else 0.0,
                            "dur_hat_z": float(trade_outcomes.get("oof_log_dur_hat", pd.Series(np.zeros(len(trade_outcomes)))).iloc[_i]) if "oof_log_dur_hat" in trade_outcomes.columns else 0.0,
                        },
                    )
                    _mask[_i] = bool(_pol.get("place_order", True))
                trade_outcomes = trade_outcomes.loc[_mask].reset_index(drop=True)
                oof_pred_df = oof_pred_df.loc[_mask].reset_index(drop=True)
                if timestamps is not None:
                    timestamps = np.asarray(timestamps)[_mask]
                if symbols is not None:
                    symbols = np.asarray(symbols)[_mask]
                tprint(f"    {bucket_name}: policy mask kept {_mask.sum()}/{len(_mask)} rows for ridge training")

            try:
                sizer, metrics = run_ridge_position_sizer_step(
                    oof_preds=oof_pred_df,
                    trade_outcomes=trade_outcomes,
                    timestamps=timestamps,
                    cfg={'cost_pct': cost_pct},
                    save_model=False,
                    run_id=run_id,
                    symbols=symbols,
                    bucket_name=bucket_name,
                )
                bkt_weights = sizer.get_weights()
                for wname, wval in bkt_weights.items():
                    dir_weights[f"{bucket_name}_{wname}"] = wval
                dir_params[bucket_name] = sizer.best_params_
                dir_metrics[bucket_name] = metrics
                tprint(f"    {bucket_name} weights: {bkt_weights}")
            except Exception as e:
                tprint(f"    {bucket_name} failed: {e}")
                continue

        if not dir_weights:
            tprint(f"    No weights produced for direction {direction}, skipping")
            continue

        # Save per-direction weight file
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        dir_weights_path = os.path.join(output_dir, f"sizer_weights_{direction}.json")
        with open(dir_weights_path, 'w') as f:
            _json.dump({
                'direction': direction,
                'weights': dir_weights,
                'params_per_bucket': dir_params,
                'buckets': list(dir_buckets.keys()),
                'run_id': run_id,
                'timestamp': _dt.now(_tz.utc).isoformat(),
            }, f, indent=2)
        tprint(f"    Saved {direction} sizer weights to {dir_weights_path}")

        all_direction_results[direction] = {
            'weights': dir_weights,
            'params': dir_params,
            'metrics': dir_metrics,
            'buckets': list(dir_buckets.keys()),
        }

    # Flatten for backward-compatible combined manifest + state
    all_weights = {}
    all_params = {}
    all_metrics = {}
    for direction, res in all_direction_results.items():
        all_weights.update(res['weights'])
        for bkt in res['buckets']:
            all_params[bkt] = res['params'].get(bkt, {})
        all_metrics[direction] = res['metrics']

    import json
    from datetime import datetime, timezone
    weights_path = os.path.join(output_dir, "sizer_weights.json")
    with open(weights_path, 'w') as f:
        json.dump({
            'weights': all_weights,
            'params_per_bucket': all_params,
            'directions': {d: {'buckets': r['buckets'], 'params': r['params']}
                           for d, r in all_direction_results.items()},
            'run_id': run_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    tprint(f"Saved combined manifest to {weights_path}")

    # Update state file
    if os.path.exists(state_file):
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        state["ridge_sizer"] = {
            "weights": all_weights,
            "params_per_bucket": all_params,
            "metrics": all_metrics,
            "directions": {d: r['buckets'] for d, r in all_direction_results.items()},
        }
        with open(state_file, "wb") as f:
            pickle.dump(state, f)
        tprint("Updated state file with ridge sizer weights")

    _ridge_result = {
        "weights": all_weights,
        "params_per_bucket": all_params,
        "metrics": all_metrics,
        "directions": {d: r['buckets'] for d, r in all_direction_results.items()},
    }
    try:
        rp = report_ridge_sizer(run_id, _ridge_result, base_dir=cfg.get('reports_root'))
        tprint(f"Ridge sizer bucket report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: ridge sizer bucket report failed: {_re}")
    tprint(f"STEP: RIDGE POSITION SIZER COMPLETE — {len(all_direction_results)} directions, {len(all_params)} buckets")
    return {
        "weights": all_weights,
        "params_per_bucket": all_params,
        "metrics": all_metrics,
        "directions": {d: r['buckets'] for d, r in all_direction_results.items()},
    }


def run_position_sizer_step(cfg, data_bundle, artifacts, logger):
    """Train and persist the new position sizer bundle.

    Args:
        cfg: full config dict
        data_bundle: dict containing either `position_sizer_df` or per-bucket oof/outcomes
        artifacts: dict with paths (expects `output_dir`, optional `state_file`, `run_id`, `data_root`)
        logger: callable logger
    """
    log = logger or tprint
    if not bool(cfg.get("position_sizer_enabled", False)):
        log("Position sizer disabled; skipping new position sizer step.")
        return artifacts

    from extreme_price_movements.position_sizer.training_orchestrator import (
        train_position_sizer_models,
    )
    from extreme_price_movements.position_sizer.runtime import PositionSizerBundle, compute_schema_hash, make_bundle_metadata
    from extreme_price_movements.position_sizer.tp_sl_selection import (
        CompositeObjectiveConfig,
        evaluate_fold_metrics,
        aggregate_candidate_folds,
        select_robust_default,
        build_tp_sl_grid,
    )

    output_dir = artifacts.get("output_dir") or os.path.join(artifacts.get("data_root", cfg.get("data_root", "data")), "artifacts", artifacts.get("run_id", "latest"), "position_sizer")
    os.makedirs(output_dir, exist_ok=True)

    if "position_sizer_df" in data_bundle:
        ps_df = data_bundle["position_sizer_df"].copy()
    else:
        rows = []
        _exclude = {"timestamp", "symbol", "return", "is_long", "index"}
        for _bucket, pack in (data_bundle.get("buckets", {}) or {}).items():
            oof = pack.get("oof")
            out = pack.get("outcomes")
            if oof is None or out is None or len(oof) != len(out):
                continue
            pred_cols = [c for c in oof.columns if c not in _exclude]
            if not pred_cols:
                continue
            score_col = "reg_mean" if "reg_mean" in oof.columns else ("reg" if "reg" in oof.columns else pred_cols[0])
            _df = pd.DataFrame(index=np.arange(len(oof)))
            _df["score"] = np.asarray(oof[score_col].values, dtype=float)
            _df["pnl_label"] = np.asarray(out.get("return", pd.Series(np.zeros(len(out)))).values, dtype=float)
            _df["mfe"] = np.asarray(out.get("mfe_ret", pd.Series(np.zeros(len(out)))).values, dtype=float)
            _df["mae"] = np.asarray(out.get("mae_ret", pd.Series(np.zeros(len(out)))).values, dtype=float)
            _df["timestamp"] = np.asarray(out.get("timestamp", pd.Series(np.arange(len(out)))).values)
            _df["symbol"] = np.asarray(out.get("symbol", pd.Series(["UNK"] * len(out))).values)
            _df["bucket"] = _bucket
            # Add all numeric OOF prediction/regime features as model inputs.
            for _c in pred_cols:
                if _c == score_col or _c in _df.columns:
                    continue
                try:
                    _v = pd.to_numeric(oof[_c], errors="coerce").astype(float)
                    if np.isfinite(_v).any():
                        _df[_c] = _v.values
                except Exception:
                    continue
            rows.append(_df)
        if not rows:
            log("Position sizer: no usable data, skipping.")
            return artifacts
        ps_df = pd.concat(rows, axis=0, ignore_index=True)

    # Derived normalized excursion ratio from predicted heads.
    _eps = 1e-9
    if "oof_log_mfe_hat" in ps_df.columns and "oof_log_mae_q70_hat" in ps_df.columns:
        _mfe_hat = np.expm1(pd.to_numeric(ps_df["oof_log_mfe_hat"], errors="coerce").values.astype(float))
        _mae_hat = np.expm1(pd.to_numeric(ps_df["oof_log_mae_q70_hat"], errors="coerce").values.astype(float))
        ps_df["mfe_mae_ratio_hat"] = _mfe_hat / np.maximum(_mae_hat, _eps)
        ps_df["mfe_mae_ratio_hat"] = np.where(np.isfinite(ps_df["mfe_mae_ratio_hat"]), ps_df["mfe_mae_ratio_hat"], 0.0)

    _regime = [str(c) for c in cfg.get("position_sizer_regime_feature_keys", []) if isinstance(c, str)]
    _missing_regime_in_df = [c for c in _regime if c not in ps_df.columns]
    if _missing_regime_in_df and {"timestamp", "symbol"}.issubset(ps_df.columns):
        try:
            _run_id = str(artifacts.get("run_id", "")).strip()
            _ts_ref = pd.to_datetime(_run_id, format="%Y%m%d_%H%M%S", utc=True)
            _data_root = str(artifacts.get("data_root", cfg.get("data_root", "data")))
            _syms_raw = pd.Series(ps_df["symbol"]).astype(str).dropna().unique().tolist()
            _syms_query = sorted(
                set(_syms_raw)
                | {s.replace("_", "/", 1) for s in _syms_raw if "_" in s}
                | {s.replace("/", "_", 1) for s in _syms_raw if "/" in s}
            )
            _feat_reg = load_features_selected(
                _ts_ref,
                _data_root,
                feature_keys=_missing_regime_in_df,
                symbols=_syms_query,
            )
            _ts_vals = pd.to_datetime(ps_df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
            _sym_vals = pd.Series(ps_df["symbol"]).astype(str).values
            _sym_codes, _sym_uniques = pd.factorize(_sym_vals, sort=False)
            _rows_by_sym = [np.where(_sym_codes == i)[0] for i in range(len(_sym_uniques))]
            _added = []
            for _k in _missing_regime_in_df:
                _dfk = _feat_reg.get(_k)
                if _dfk is None or not isinstance(_dfk, pd.DataFrame) or _dfk.empty:
                    continue
                _feat_idx = pd.DatetimeIndex(_dfk.index)
                _tpos = _feat_idx.get_indexer(_ts_vals)
                _out = np.full(len(ps_df), np.nan, dtype=np.float64)
                _cols = set(_dfk.columns.astype(str))
                for _sym, _ridx in zip(_sym_uniques, _rows_by_sym):
                    _cands = [_sym]
                    if "_" in _sym:
                        _cands.append(_sym.replace("_", "/", 1))
                    if "/" in _sym:
                        _cands.append(_sym.replace("/", "_", 1))
                    _col = next((c for c in _cands if c in _cols), None)
                    if _col is None:
                        continue
                    _tp = _tpos[_ridx]
                    _valid = _tp >= 0
                    if not np.any(_valid):
                        continue
                    _src = _dfk[_col].to_numpy(dtype=np.float64, copy=False)
                    _tmp = np.full(len(_ridx), np.nan, dtype=np.float64)
                    _tmp[_valid] = _src[_tp[_valid]]
                    _out[_ridx] = _tmp
                if np.isfinite(_out).any():
                    ps_df[_k] = _out
                    _added.append(_k)
            if _added:
                log(
                    "Position sizer regime enrich: "
                    f"added={len(_added)}/{len(_missing_regime_in_df)} from feature store."
                )
        except Exception as _e:
            log(f"WARNING: Position sizer regime enrich failed: {_e}")

    _forbid = {"pnl_label", "mfe", "mae", "timestamp", "symbol", "bucket", "return", "is_long", "index"}
    _priority = [str(c) for c in cfg.get("position_sizer_feature_priority", ["score"]) if isinstance(c, str)]
    _num_cols = [c for c in ps_df.columns if c not in _forbid and pd.api.types.is_numeric_dtype(ps_df[c])]
    feature_cols = []
    for _c in _priority + _regime + _num_cols:
        if _c in _num_cols and _c not in feature_cols:
            feature_cols.append(_c)
    if not feature_cols:
        feature_cols = ["score"] if "score" in ps_df.columns else ["pnl_label"]
    _regime_present = [c for c in _regime if c in feature_cols]
    _regime_missing = [c for c in _regime if c not in _regime_present]
    log(
        "Position sizer regime features: "
        f"configured={len(_regime)} present={len(_regime_present)} missing={len(_regime_missing)}"
    )
    if _regime_missing:
        log(f"Position sizer regime features missing: {_regime_missing}")

    _ps_train = train_position_sizer_models(ps_df=ps_df, feature_cols=feature_cols, cfg=cfg)
    _ps_diag = dict(_ps_train.get("diagnostics", {}) or {})
    _ps_cfg = dict(_ps_train.get("training_config", {}) or {})
    if _ps_cfg:
        log(
            "PositionSizer models trained: "
            f"pwin={_ps_cfg.get('pwin_base_engine')} "
            f"quant={_ps_cfg.get('quantile_base_engine')} "
            f"reg={_ps_cfg.get('regularization_level')} "
            f"cal={_ps_cfg.get('calibrator_method')} "
            f"delta={_ps_cfg.get('quantile_delta')}"
        )
    _pdiag = dict(_ps_diag.get("pwin", {}) or {})
    if _pdiag:
        log(
            "PositionSizer pwin diag: "
            f"auc_cal={_pdiag.get('auc_cal', float('nan')):.4f} "
            f"bce_cal={_pdiag.get('bce_cal', float('nan')):.4f} "
            f"ece_cal={_pdiag.get('ece_cal', float('nan')):.4f}"
        )
    _wdiag = dict(_ps_diag.get("win_quantiles", {}) or {})
    _ldiag = dict(_ps_diag.get("loss_quantiles", {}) or {})
    if _wdiag:
        log(
            "PositionSizer win-quantile diag: "
            f"pin50={_wdiag.get('pinball_q50', float('nan')):.6f} "
            f"pinH={_wdiag.get('pinball_qh', float('nan')):.6f} "
            f"cov50={_wdiag.get('coverage_q50', float('nan')):.3f} "
            f"covH={_wdiag.get('coverage_qh', float('nan')):.3f}"
        )
    if _ldiag:
        log(
            "PositionSizer loss-quantile diag: "
            f"pin50={_ldiag.get('pinball_q50', float('nan')):.6f} "
            f"pinH={_ldiag.get('pinball_qh', float('nan')):.6f} "
            f"cov50={_ldiag.get('coverage_q50', float('nan')):.3f} "
            f"covH={_ldiag.get('coverage_qh', float('nan')):.3f}"
        )
    for _b, _bd in sorted((_ps_diag.get("per_bucket", {}) or {}).items()):
        _bp = dict((_bd or {}).get("pwin", {}) or {})
        _bw = dict((_bd or {}).get("win_quantiles", {}) or {})
        _bl = dict((_bd or {}).get("loss_quantiles", {}) or {})
        log(
            f"  [bucket={_b}] pwin_n={_bp.get('n', 0)} "
            f"pwin_mean_pred={_bp.get('mean_pred', float('nan')):.4f} "
            f"win_pinH={_bw.get('pinball_qh', float('nan')):.6f} "
            f"loss_pinH={_bl.get('pinball_qh', float('nan')):.6f}"
        )
    pwin_model = _ps_train["pwin_model"]
    win_model = _ps_train["win_model"]
    loss_model = _ps_train["loss_model"]
    exp_win_q = float(_ps_train["exp_win_quantile"])
    risk_loss_q = float(_ps_train["risk_loss_quantile"])
    costs_mode = str(_ps_train["costs_mode"])

    tp_sl_defaults = None
    if bool(cfg.get("tp_sl_search_enabled", False)) and str(cfg.get("tp_sl_search_optimizer", "legacy")) == "new":
        cand_metrics = {}
        k_tp_grid = cfg.get("tp_sl_search_k_tp_grid", [1.0])
        k_sl_grid = cfg.get("tp_sl_search_k_sl_grid", [1.0])
        cobj = CompositeObjectiveConfig(
            mar=float(cfg.get("objective_mar", 0.0)),
            eps_log=float(cfg.get("objective_eps_log", 1e-12)),
            eps_sortino=float(cfg.get("objective_eps_sortino", 1e-12)),
            mode=str(cfg.get("objective_composite_mode", "hard_gate")),
            q_top=float(cfg.get("objective_composite_q_top", 0.95)),
            selection=str(cfg.get("objective_composite_selection", "min_std")),
            min_trades_per_fold=int(cfg.get("tp_sl_search_min_trades_per_fold", 200)),
            elg_scale=float(cfg.get("objective_scaling_elg_scale", 10000.0)),
            mnpt_scale=float(cfg.get("objective_scaling_mnpt_scale", 10000.0)),
            elg_min=float(cfg.get("objective_clipping_elg_min", -1.0)),
            elg_max=float(cfg.get("objective_clipping_elg_max", 1.0)),
            sortino_min=float(cfg.get("objective_clipping_sortino_min", -10.0)),
            sortino_max=float(cfg.get("objective_clipping_sortino_max", 10.0)),
            mnpt_min=float(cfg.get("objective_clipping_mnpt_min", -1.0)),
            mnpt_max=float(cfg.get("objective_clipping_mnpt_max", 1.0)),
        )
        ts = pd.to_datetime(ps_df["timestamp"], utc=True, errors="coerce") if "timestamp" in ps_df.columns else pd.Series(np.arange(len(ps_df)))
        split_ids = pd.qcut(np.arange(len(ps_df)), q=min(3, max(1, len(ps_df)//200)), labels=False, duplicates="drop") if len(ps_df) > 0 else np.array([])
        for k_tp, k_sl in build_tp_sl_grid(k_tp_grid, k_sl_grid):
            folds = []
            for sid in np.unique(split_ids):
                m = np.asarray(split_ids == sid)
                pnl = np.asarray(ps_df.loc[m, "pnl_label"].values, dtype=float)
                pnl_adj = np.where(pnl >= 0.0, pnl * float(k_tp), pnl * float(k_sl))
                if "timestamp" in ps_df.columns:
                    grp = pd.DataFrame({"ts": ts[m], "p": pnl_adj}).groupby("ts", as_index=False)["p"].sum()
                    r_t = grp["p"].values
                else:
                    r_t = pnl_adj
                folds.append(evaluate_fold_metrics(r_t=r_t, pnl_net=pnl_adj, cfg=cobj))
            cand_metrics[(float(k_tp), float(k_sl))] = folds
        summary_df = aggregate_candidate_folds(cand_metrics)
        summary_df["selection_window"] = "in_fold_selection"
        summary_df["evaluation_window"] = "fold_holdout"
        best = select_robust_default(summary_df, cobj)
        tp_sl_defaults = {
            "k_tp": float(best["candidate"][0]),
            "k_sl": float(best["candidate"][1]),
            "objective_mean": float(best.get("Objective_mean", np.nan)),
            "objective_std": float(best.get("Objective_std", np.nan)),
            "objective_worst": float(best.get("Objective_worst", np.nan)),
        }
        summary_df.to_csv(os.path.join(output_dir, "tp_sl_selection_summary.csv"), index=False)

    _git_sha = ""
    try:
        import subprocess as _sp
        _git_sha = _sp.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        _git_sha = ""
    _q_cfg = {
        "exp_win_quantile": exp_win_q,
        "risk_loss_quantile": risk_loss_q,
        "costs_mode": costs_mode,
        "training_config": _ps_cfg,
        "training_diagnostics": _ps_diag,
    }
    _schema_hash = compute_schema_hash(feature_cols, extra=_q_cfg)
    _meta = make_bundle_metadata(git_sha=_git_sha)
    bundle = PositionSizerBundle(
        feature_cols=list(feature_cols),
        pwin_model=pwin_model,
        win_model=win_model,
        loss_model=loss_model,
        tp_sl_defaults=tp_sl_defaults,
        config={
            "backend": "new",
            "soft_label_enabled": bool(_ps_train.get("soft_label_enabled", False)),
            "calibration_scope": str(cfg.get("position_sizer_calibration_scope", "regime")),
            "exp_win_quantile": exp_win_q,
            "risk_loss_quantile": risk_loss_q,
            "costs_mode": costs_mode,
        },
        version="v1",
        bundle_version=1,
        created_at=_meta.get("created_at", ""),
        git_sha=_meta.get("git_sha", ""),
        schema_hash=_schema_hash,
    )

    bundle_path = os.path.join(output_dir, "position_sizer_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    manifest = {
        "bundle_path": bundle_path,
        "feature_cols": feature_cols,
        "regime_feature_keys_configured": _regime,
        "regime_feature_keys_used": _regime_present,
        "regime_feature_keys_missing": _regime_missing,
        "tp_sl_defaults": tp_sl_defaults,
        "version": "v1",
        "bundle_version": 1,
        "created_at": bundle.created_at,
        "git_sha": bundle.git_sha,
        "schema_hash": bundle.schema_hash,
        "exp_win_quantile": exp_win_q,
        "risk_loss_quantile": risk_loss_q,
        "costs_mode": costs_mode,
    }
    with open(os.path.join(output_dir, "position_sizer_bundle.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(output_dir, "position_sizer_training_diagnostics.json"), "w") as f:
        json.dump({"training_config": _ps_cfg, "diagnostics": _ps_diag}, f, indent=2)

    state_file = artifacts.get("state_file")
    if state_file and os.path.exists(state_file):
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        state["position_sizer"] = manifest
        with open(state_file, "wb") as f:
            pickle.dump(state, f)

    artifacts = dict(artifacts)
    artifacts["position_sizer"] = manifest
    log(f"STEP: POSITION SIZER COMPLETE — bundle={bundle_path} Wq={exp_win_q:.2f} Lq={risk_loss_q:.2f} costs_mode={costs_mode}")
    return artifacts


def run_sizer_step(ts_sig, cfg, state_file):
    backend = str(cfg.get("position_sizer_backend", "new")).lower()
    tprint(f"STEP: SIZER START (backend={backend})")
    if backend == "ridge":
        raise ValueError(
            "position_sizer_backend='ridge' is disabled. "
            "Use position_sizer_backend='new' (EV decomposition backend)."
        )
    if backend != "new":
        raise ValueError(f"Unknown position_sizer_backend={backend}")

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    data_root = cfg.get("data_root", "data")
    from extreme_price_movements.run_ridge_sizer import (
        load_meta_oof_predictions as load_bucket_oofs,
        load_trade_outcomes,
    )
    try:
        bucket_oofs = load_bucket_oofs(data_root, run_id)
    except FileNotFoundError as e:
        tprint(f"WARNING: {e}")
        tprint("Skipping new position sizer step - no meta OOF predictions found.")
        return None

    buckets = {}
    for bucket_name, oof_preds in bucket_oofs.items():
        try:
            trade_outcomes = load_trade_outcomes(data_root, run_id, oof_preds)
        except FileNotFoundError:
            continue
        buckets[bucket_name] = {"oof": oof_preds, "outcomes": trade_outcomes}

    artifacts = {
        "run_id": run_id,
        "data_root": data_root,
        "output_dir": os.path.join(data_root, "artifacts", run_id, "position_sizer"),
        "state_file": state_file,
    }
    return run_position_sizer_step(cfg=cfg, data_bundle={"buckets": buckets}, artifacts=artifacts, logger=tprint)


def compute_regime_boundaries(feats):
    """Compute stable tercile boundaries for granular regime features over the entire available dataset."""
    tprint("  Computing stable regime boundaries...")
    _regime_map = {
        "vol_12h": "rv_12h",
        "vol_48h": "rv_24h",
        "volume_12h": "vol_z_base",
        "volume_48h": "vol_z24_base",
        "trend_12h": "ret6h",
        "trend_48h": "trend_pct_base",
    }
    boundaries = {}
    for rname, src_col in _regime_map.items():
        if src_col in feats:
            df = feats[src_col]
            # Consider all non-NaN values across all symbols and time to define "Low/Mid/High"
            vals = df.values.flatten()
            valid = vals[np.isfinite(vals)]
            if len(valid) > 100:
                try:
                    terciles = np.nanpercentile(valid, [33.3, 66.7])
                    boundaries[rname] = [float(terciles[0]), float(terciles[1])]
                    tprint(f"  Stable boundaries for {rname} ({src_col}): {boundaries[rname][0]:.4f}, {boundaries[rname][1]:.4f} (n={len(valid)})")
                except Exception as e:
                    tprint(f"  WARNING: Failed to compute boundaries for {rname}: {e}")
    tprint(f"  Regime boundaries computed for {len(boundaries)} features.")
    return boundaries


def _extract_required_features(bundle, cfg):
    """Extract required features strictly from configured feature-key lists."""
    _ = bundle  # Keep signature stable; loading is intentionally config-key driven.

    # Minimal runtime essentials for candidate generation, backtest windowing, and regime boundaries.
    req_feats = {
        "atr_pct",
        "ret1h",
        "ret24h",
        "range_12h_pct",
        "volatility_zscore",
        "rv_12h",
        "rv_24h",
        "vol_z_base",
        "vol_z24_base",
        "ret6h",
        "trend_pct_base",
    }

    dev_metric = cfg.get("trade_deviation_metric", "dist_ema_fast")
    if dev_metric:
        req_feats.add(str(dev_metric))

    # Sizer-triggered OOS backtest: keep feature loading constrained to
    # position-sizer relevant context only.
    if bool(cfg.get("sizer_oos_mode", False)):
        reg_keys = cfg.get("position_sizer_regime_feature_keys", [])
        if isinstance(reg_keys, (list, tuple, set)):
            req_feats.update(str(v) for v in reg_keys if isinstance(v, str) and v)
        out = sorted({f for f in req_feats if not str(f).startswith("pred_")})
        tprint(f"Feature load whitelist (sizer_oos_mode): keys={len(out)} (position-sizer relevant)")
        return out

    # Only configured feature-key baskets are allowed.
    key_lists = [
        "exh_feature_keys",
        "spike_feature_keys",
        "tf_feature_keys",
        "mr_feature_keys",
        "meta_feature_keys",
        "mr_meta_feature_keys",
        "tf_meta_feature_keys",
        "position_sizer_regime_feature_keys",
    ]
    for kname in key_lists:
        vals = cfg.get(kname, [])
        if isinstance(vals, (list, tuple, set)):
            req_feats.update(str(v) for v in vals if isinstance(v, str) and v)

    # Include merged meta union (meta + overlays) for safety.
    req_feats.update(_meta_feature_keys_union(cfg))

    # Drop dynamic prediction columns from config baskets if present.
    req_feats = {f for f in req_feats if not str(f).startswith("pred_")}
    out = sorted(req_feats)
    tprint(f"Feature load whitelist: keys={len(out)} (config-only)")
    return out

def run_risk_optimization_step(ts_sig, margin_symbols, cfg, store, state_file):
    tprint("STEP: RISK OPTIMIZATION START")
    if not os.path.exists(state_file):
        tprint(f"State file {state_file} not found.")
        return

    # Prevent OOS leakage
    run_ts = ts_sig # Keep original for loading artifacts
    opt_ts = ts_sig # Used for data filtering
    
    oos_days = cfg.get("oos_holdout_days", 0)
    if oos_days > 0:
        opt_ts = ts_sig - pd.Timedelta(days=oos_days)
        tprint(f"Risk Optimization: Excluding last {oos_days} days (OOS). Training end: {opt_ts}")

    with open(state_file, "rb") as f:
        state = pickle.load(f)

    bundle = state.get("bundle")
    if not bundle:
        tprint("No model bundle in state.")
        return

    # Need Data for optimization (simulation)
    # Use opt_ts to filter training universe and data
    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=opt_ts)
    dfs = {}

    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    for s in train_syms:
        df = store.load(s)
        if not df.empty: dfs[s] = df[df.index <= opt_ts].tail(24*lookback_days)

    if not dfs: return

    panel = to_panel(dfs)
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    
    # Load only required features to prevent OOM
    req_feats = _extract_required_features(bundle, cfg)
    feats = load_features_selected(run_ts, cfg["data_root"], feature_keys=req_feats)
    if feats is None:
        tprint("ERROR: Features not found for risk optimization.")
        return
    feats = _ensure_atr_pct_feature(feats, panel, cfg, symbols=train_syms)

    # Compute stable regime boundaries for meta-models
    cfg["granular_regime_boundaries"] = compute_regime_boundaries(feats)

    # We also need p_exh_hist. Load from artifacts (using run_ts).
    run_id = run_ts.strftime("%Y%m%d_%H%M%S")
    from extreme_price_movements.data_store import load_artifact_df
    p_exh_hist = load_artifact_df(cfg["data_root"], run_id, "labels", "exhaustion_history")

    if p_exh_hist is None:
        tprint("Exhaustion history artifact missing. Regenerating...")
        # Generate up to opt_ts? Or regenerate full and slice?
        # Typically we want history up to opt_ts for optimization.
        p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, opt_ts, cfg["train_lookback_hours"], train_syms)

    alpha_models = bundle["alpha_models"]
    best_risk = optimize_risk_params(panel, feats, mkt_gates, cfg, train_syms, opt_ts, p_exh_hist, alpha_models)

    prev_risk = state.get("risk_params", {}) if isinstance(state.get("risk_params"), dict) else {}
    if isinstance(prev_risk, dict) and "signal_params" in prev_risk and isinstance(best_risk, dict):
        best_risk["signal_params"] = prev_risk["signal_params"]

    state["risk_params"] = best_risk
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    tprint("Risk params updated in state file.")

    # Generate risk optimization report
    try:
        run_id = run_ts.strftime("%Y%m%d_%H%M%S")
        granular = best_risk.get("granular_risk", {}) if isinstance(best_risk, dict) else {}
        report_path = generate_risk_report(
            run_id=run_id,
            cfg=cfg,
            granular_risk=granular,
            base_dir=cfg.get('reports_root'),
        )
        tprint(f"Risk optimization report saved to {report_path}")
    except Exception as e:
        tprint(f"WARNING: Failed to generate risk report: {e}")

    tprint("STEP: RISK OPTIMIZATION COMPLETE")

def run_backtest_step(ts_sig, margin_symbols, cfg, store, state_file):
    tprint("STEP: BACKTEST START")
    if not os.path.exists(state_file):
        tprint("State file not found.")
        return False

    with open(state_file, "rb") as f:
        model_state = pickle.load(f)
    
    bundle = model_state.get("bundle")

    def _ensure_meta_aliases(_bundle):
        if not isinstance(_bundle, dict):
            return
        _meta = _bundle.get("meta_models")
        if not isinstance(_meta, dict) or not _meta:
            return
        for _side in ("long", "short"):
            for _kind in ("mr", "tf"):
                _base = f"{_side}_{_kind}"
                _reg = _meta.get(f"{_base}_reg")
                _clf = _meta.get(f"{_base}_clf") or _meta.get(f"{_base}_early_inval")
                if _base not in _meta and _reg is not None:
                    _meta[_base] = _reg
                if f"{_base}_clf" not in _meta and _clf is not None:
                    _meta[f"{_base}_clf"] = _clf
        _bundle["meta_models"] = _meta

    def _log_meta_head_coverage(_bundle):
        _meta = _bundle.get("meta_models") if isinstance(_bundle, dict) else {}
        if not isinstance(_meta, dict):
            _meta = {}
        _suffixes = ("_reg", "_clf", "_utility", "_mae_q70", "_mfe")
        _req = []
        for _side in ("long", "short"):
            for _kind in ("mr", "tf"):
                _base = f"{_side}_{_kind}"
                _req.extend([f"{_base}{_s}" for _s in _suffixes])
        _present = [k for k in _req if k in _meta]
        _missing = [k for k in _req if k not in _meta]
        tprint(
            "Meta head coverage: "
            f"present={len(_present)}/{len(_req)} missing={len(_missing)}"
        )
        if _missing:
            tprint(f"Meta head coverage missing keys: {_missing[:12]}{' ...' if len(_missing) > 12 else ''}")

    _ensure_meta_aliases(bundle)
    if isinstance(bundle, dict):
        _meta_now = bundle.get("meta_models")
        _meta_count = len(_meta_now) if isinstance(_meta_now, dict) else 0
        if _meta_count == 0:
            try:
                _meta_state_file = os.path.join(os.path.dirname(state_file), "model_state_meta.pkl")
                if os.path.exists(_meta_state_file):
                    with open(_meta_state_file, "rb") as _mf:
                        _meta_state = pickle.load(_mf)
                    _meta_bundle = _meta_state.get("bundle", {}) if isinstance(_meta_state, dict) else {}
                    _meta_models = _meta_bundle.get("meta_models", {}) if isinstance(_meta_bundle, dict) else {}
                    if isinstance(_meta_models, dict) and len(_meta_models) > 0:
                        bundle["meta_models"] = _meta_models
                        _ensure_meta_aliases(bundle)
                        tprint(
                            "Backtest state patch: loaded meta_models from model_state_meta.pkl "
                            f"(count={len(_meta_models)})"
                        )
            except Exception as _e_meta_patch:
                tprint(f"WARNING: failed to patch meta_models from model_state_meta.pkl: {_e_meta_patch}")
    _log_meta_head_coverage(bundle)

    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    dfs = {}
    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    with Timer("Backtest Data Load"):
        for s in train_syms:
            df = store.load(s)
            if not df.empty:
                dfs[s] = df[df.index <= ts_sig].tail(24 * lookback_days)

    if not dfs:
        return

    panel = to_panel(dfs)
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    # Load only required features to prevent OOM
    req_feats = _extract_required_features(bundle, cfg)
    feats = load_features_selected(ts_sig, cfg["data_root"], feature_keys=req_feats, symbols=train_syms)
    if feats is None:
        tprint("ERROR: Features not found for backtest.")
        return
    feats = _ensure_atr_pct_feature(feats, panel, cfg, symbols=train_syms)

    # Compute stable regime boundaries for meta-models
    cfg["granular_regime_boundaries"] = compute_regime_boundaries(feats)

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    from extreme_price_movements.data_store import load_artifact_df
    p_exh_hist = load_artifact_df(cfg["data_root"], run_id, "labels", "exhaustion_history")
    if p_exh_hist is None:
        p_exh_hist = generate_exhaustion_history(panel, feats, mkt_gates, cfg, ts_sig, cfg["train_lookback_hours"], train_syms)

    test_days_cfg = int(cfg.get("oos_holdout_days", 730))
    test_days = max(730, test_days_cfg)
    if test_days != test_days_cfg:
        tprint(f"Backtest OOS holdout_days raised to minimum: cfg={test_days_cfg} -> used={test_days}")
    start_ts = ts_sig - pd.Timedelta(days=test_days)
    end_ts = ts_sig - pd.Timedelta(hours=24)
    
    idx_tz = feats["ret1h"].index.tz
    if idx_tz is None and start_ts.tz is not None:
        start_ts = start_ts.tz_localize(None)
        end_ts = end_ts.tz_localize(None)
    elif idx_tz is not None and start_ts.tz is None:
        start_ts = start_ts.tz_localize(idx_tz)
        end_ts = end_ts.tz_localize(idx_tz)

    # Align market-gates index timezone with feature index timezone to avoid
    # silent empty intersections in signal generation.
    mkt_idx_tz = mkt_gates.index.tz if hasattr(mkt_gates, "index") else None
    if idx_tz is None and mkt_idx_tz is not None:
        mkt_gates = mkt_gates.copy()
        mkt_gates.index = mkt_gates.index.tz_localize(None)
    elif idx_tz is not None and mkt_idx_tz is None:
        mkt_gates = mkt_gates.copy()
        mkt_gates.index = mkt_gates.index.tz_localize(idx_tz)

    valid_ts = [t for t in feats["ret1h"].index if t >= start_ts and t <= end_ts]
    tprint(f"Running backtest over {len(valid_ts)} hours...")
    if bool(cfg.get("signal_opt_debug", True)):
        _mkt_idx = mkt_gates.index if hasattr(mkt_gates, "index") else []
        _ov = int(sum(1 for t in valid_ts if t in _mkt_idx))
        tprint(
            "SignalOptIndexDiag: "
            f"valid_ts={len(valid_ts)} mkt_gates_idx={len(_mkt_idx)} overlap={_ov} "
            f"overlap_pct={(100.0 * _ov / max(1, len(valid_ts))):.1f}%"
        )
        if _ov == 0 and len(valid_ts) and len(_mkt_idx):
            tprint(
                "SignalOptIndexDiag sample: "
                f"valid_ts[0]={valid_ts[0]} mkt_idx[0]={_mkt_idx[0]}"
            )
    if len(valid_ts) < 48:
        tprint("Not enough timestamps for backtest optimization.")
        return

    o_s = panel["open"]; h_s = panel["high"]; l_s = panel["low"]; c_s = panel["close"]
    atr_s = feats["atr_pct"]

    # Ensure all runtime panels share the same timezone semantics as feature index.
    def _align_idx_tz(obj, target_tz):
        if not hasattr(obj, "index"):
            return obj
        obj_tz = obj.index.tz
        if target_tz is None and obj_tz is not None:
            out = obj.copy()
            out.index = out.index.tz_localize(None)
            return out
        if target_tz is not None and obj_tz is None:
            out = obj.copy()
            out.index = out.index.tz_localize(target_tz)
            return out
        return obj

    o_s = _align_idx_tz(o_s, idx_tz)
    h_s = _align_idx_tz(h_s, idx_tz)
    l_s = _align_idx_tz(l_s, idx_tz)
    c_s = _align_idx_tz(c_s, idx_tz)
    atr_s = _align_idx_tz(atr_s, idx_tz)
    risk_conf = model_state.get("risk_params", {}) or {}
    bundle = model_state.get("bundle")
    _ps_manifest = model_state.get("position_sizer", {}) if isinstance(model_state, dict) else {}
    if isinstance(_ps_manifest, dict) and _ps_manifest.get("bundle_path"):
        risk_conf["position_sizer_bundle_path"] = _ps_manifest.get("bundle_path")

    fee_bps = cfg.get("fee_bps", 25.0)
    cost = CostModel(fee_side=float(fee_bps) / 10000.0, slippage_side=float(cfg.get("slippage_bps", 0.0)) / 10000.0)
    assert_units(cost)
    emit_run_header(
        tprint=tprint,
        run_id=run_id,
        policy_version=str(run_id),
        cost_model={"fee_side": float(cost.fee_side), "slippage_side": float(cost.slippage_side), "round_trip": float(cost.round_trip)},
        extra={"stage": "backtest_signal_optimization", "sizer_backend_used": str(cfg.get("sizer_backend_used", cfg.get("position_sizer_backend", "new")))},
    )
    if bool(cfg.get("signal_opt_debug", True)):
        tprint(
            "SignalOptConfig: "
            f"use_limit_orders={bool(cfg.get('use_limit_orders', False))} "
            f"limit_offset_bps={float(cfg.get('limit_offset_bps', 0.0)):.1f} "
            f"exit_limit_offset_bps={float(cfg.get('exit_limit_offset_bps', 0.0)):.1f} "
            f"position_sizer_backend={cfg.get('position_sizer_backend', 'new')} "
            f"ranking_trade_percentile_threshold={float(cfg.get('ranking_trade_percentile_threshold', 0.90)):.2f}"
        )

    def rank01(x: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
        pct = (ranks - 1.0) / max(1.0, x.size - 1.0)
        return pct if higher_is_better else (1.0 - pct)

    def pnl_sortino_dd_utility(pnls, sortinos, max_dds, w_pnl=0.65, w_sortino=0.25, w_dd=0.10):
        return w_pnl * rank01(pnls, True) + w_sortino * rank01(sortinos, True) + w_dd * rank01(max_dds, True)

    def compute_metrics(trades):
        if not trades:
            return 0.0, 0.0, 0.0, 0.0, 0
        rets = np.array([x["pnl"] for x in trades], dtype=np.float64)
        pnl = float(np.sum(rets))
        neg = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(neg) + 1e-12)) if neg.size > 0 else float(np.mean(rets) / 1e-12)
        eq = np.cumsum(rets)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = float(np.min(dd)) if dd.size else 0.0
        count = len(trades)
        win_rate = float(np.mean(rets > 0)) if count > 0 else 0.0
        return pnl, sortino, max_dd, win_rate, count

    def run_slice(ts_list, signal_params):
        trades = []
        local_risk = dict(risk_conf)
        local_risk["signal_params"] = signal_params
        diag = {
            "hours_total": int(len(ts_list)),
            "hours_with_orders": 0,
            "orders_post_signal": 0,
            "orders_after_budget": 0,
            "skipped_policy_place_order": 0,
            "skipped_missing_open": 0,
            "executed_trades": 0,
            "reason_counts": {},
        }
        max_concurrent = int(cfg.get("max_concurrent_trades", 5))
        max_portfolio_weight = float(cfg.get("max_portfolio_weight", 0.25))

        # Daily risk budget: per-specialist and total daily caps
        max_daily_per_specialist = int(cfg.get("max_daily_per_specialist", 8))
        max_daily_total = int(cfg.get("max_daily_total", 25))

        # Drawdown-based regime throttle parameters
        throttle_lookback = int(cfg.get("throttle_lookback_trades", 20))
        throttle_dd_thr = float(cfg.get("throttle_dd_threshold", -0.02))  # cumPnL drawdown trigger
        throttle_factor = float(cfg.get("throttle_sizing_factor", 0.5))   # reduce sizing to 50%

        from collections import defaultdict
        daily_bucket_counts = defaultdict(lambda: defaultdict(int))  # date -> bucket -> count
        daily_total_counts = defaultdict(int)  # date -> total count
        if bool(cfg.get("signal_opt_debug", True)):
            tprint(
                "  RunSlice start: "
                f"hours={len(ts_list)} thr_long={float(signal_params.get('thr_long', 0.0)):.6f} "
                f"thr_short={float(signal_params.get('thr_short', 0.0)):.6f} "
                f"score_gate_q={float(signal_params.get('score_gate_q', 0.0)):.2f} "
                f"k_frac_long={signal_params.get('k_frac_long', None)} "
                f"k_frac_short={signal_params.get('k_frac_short', None)}"
            )

        for i, t in enumerate(ts_list):
            if i % 20 == 0:
                tprint(f"  Signal generation progress: {i}/{len(ts_list)} ({i/len(ts_list):.1%}) - {t}")
            # --- Regime throttle: check recent closed-trade drawdown ---
            size_mult = 1.0
            if len(trades) >= throttle_lookback:
                recent_pnls = np.array([tr["pnl"] for tr in trades[-throttle_lookback:]], dtype=np.float64)
                cum = np.cumsum(recent_pnls)
                peak = np.maximum.accumulate(cum)
                dd = cum - peak
                if dd[-1] < throttle_dd_thr:
                    size_mult = throttle_factor

            # Count currently open trades and their total weight at this timestamp
            open_trades = []
            for tr in trades:
                _tr_entry = pd.Timestamp(tr["entry_ts"])
                _tr_exit = pd.Timestamp(tr["exit_ts"])
                if t.tz is not None and _tr_entry.tz is None:
                    _tr_entry = _tr_entry.tz_localize(t.tz)
                    _tr_exit = _tr_exit.tz_localize(t.tz)
                elif t.tz is None and _tr_entry.tz is not None:
                    _tr_entry = _tr_entry.tz_convert(None)
                    _tr_exit = _tr_exit.tz_convert(None)
                if _tr_entry <= t < _tr_exit:
                    open_trades.append(tr)
            open_count = len(open_trades)
            open_weight = sum(abs(tr.get("weight", 0.0)) for tr in open_trades)
            remaining_slots = max(0, max_concurrent - open_count)
            remaining_weight = max(0.0, max_portfolio_weight - open_weight)
            orders = generate_hourly_signals(t, feats, mkt_gates, bundle, local_risk, cfg, p_exh_hist, [])
            if orders:
                diag["hours_with_orders"] += 1
                diag["orders_post_signal"] += int(len(orders))
            orders = orders[:remaining_slots]  # cap to available slots
            # Apply regime throttle to order weights
            if size_mult < 1.0:
                for o in orders:
                    o["weight"] = o.get("weight", 0.0) * size_mult
            # Further cap by portfolio weight
            capped_orders = []
            for o in orders:
                w = abs(o.get("weight", 0.0))
                if w <= remaining_weight:
                    capped_orders.append(o)
                    remaining_weight -= w
            orders = capped_orders

            # --- Daily concentration controls ---
            trade_date = t.date()
            budget_filtered = []
            for o in orders:
                bucket = f"{o['side'].upper()}_{o['dom'].upper()}"
                if daily_total_counts[trade_date] >= max_daily_total:
                    break
                if daily_bucket_counts[trade_date][bucket] >= max_daily_per_specialist:
                    continue
                budget_filtered.append(o)
            orders = budget_filtered
            diag["orders_after_budget"] += int(len(orders))
            for order in orders:
                sym = order["symbol"]
                side = order["side"]
                score = float(order["score"])
                dom = order["dom"]
                weight = float(order.get("weight", 0.0))

                entry_ts = t + pd.Timedelta(hours=1)
                if entry_ts not in o_s.index or sym not in o_s.columns:
                    diag["skipped_missing_open"] += 1
                    continue
                entry_px = float(o_s.loc[entry_ts, sym])

                if dom == "tf":
                    mode = "best" if side == "long" else "worst"
                else:
                    mode = "worst" if side == "long" else "best"
                risk_keys = [f"risk_{dom}_{mode}", f"risk_{side}_{dom}"]
                granular = local_risk.get("granular_risk", {})
                rp = {}
                matched_key = None
                for risk_key in risk_keys:
                    if risk_key in granular:
                        rp = granular[risk_key]
                        matched_key = risk_key
                        break
                if matched_key is None:
                    tprint(f"  WARNING: No granular risk for keys {risk_keys}, falling back to global cfg")
                rp = flatten_bucket_policy(rp)
                atr_val_for_policy = float(atr_s[sym].loc[entry_ts]) if entry_ts in atr_s[sym].index else 0.02
                pol = compute_entry_policy_decision(
                    entry_px=entry_px,
                    atr_frac=atr_val_for_policy,
                    score=score,
                    bucket_cfg=rp,
                    features={
                        "u_hat_z": order.get("u_hat_z", np.tanh(score)),
                        "mae_hat_z": order.get("mae_hat_z", abs(np.tanh(score))),
                        "mfe_hat_z": order.get("mfe_hat_z", max(np.tanh(score), 0.0)),
                        "dur_hat_z": order.get("dur_hat_z", 0.0),
                    },
                )
                if not bool(pol.get("place_order", True)):
                    diag["skipped_policy_place_order"] += 1
                    continue

                k_sl = rp.get("k_sl", cfg["risk_k_sl"])
                k_ts = rp.get("k_trail_start", cfg["risk_k_trail_start"])
                k_td = rp.get("k_trail_dist", cfg["risk_k_trail_dist"])

                # Extract optimized trailing-profit params
                tp_mult = rp.get("tp_mult", cfg.get("tp_mult"))
                sl_mult = rp.get("sl_mult", cfg.get("sl_mult"))
                trail_mult = rp.get("trail_mult", cfg.get("trail_mult", 0.25))

                sc_scale = rp.get("score_scale", 0.0)
                adj = (1.0 + sc_scale * abs(score))

                temp_cfg = cfg.copy()
                temp_cfg["risk_k_sl"] = k_sl * adj
                temp_cfg["risk_k_trail_start"] = k_ts
                temp_cfg["risk_k_trail_dist"] = k_td

                if tp_mult is not None: temp_cfg["tp_mult"] = tp_mult
                if sl_mult is not None: temp_cfg["sl_mult"] = sl_mult
                temp_cfg["trail_mult"] = trail_mult
                temp_cfg["use_limit_orders"] = bool(cfg.get("use_limit_orders", True))
                temp_cfg["limit_offset_bps"] = float(pol.get("limit_offset_bps_dynamic", temp_cfg.get("limit_offset_bps", cfg.get("limit_offset_bps", 0.0))))
                temp_cfg["trail_mult"] = float(pol.get("trail_mult_eff", temp_cfg.get("trail_mult", trail_mult)))
                temp_cfg["giveback_pct"] = float(pol.get("giveback_pct_eff", temp_cfg.get("giveback_pct", cfg.get("giveback_pct", 0.005))))
                temp_cfg["profit_lock_amount"] = float(pol.get("profit_lock_amount_eff", temp_cfg.get("profit_lock_amount", cfg.get("profit_lock_amount", 0.003))))
                temp_cfg["kill_c"] = float(pol.get("kill_c_eff", temp_cfg.get("kill_c", cfg.get("kill_c", 0.005))))
                if pol.get("sl_distance_atr_eff") is not None:
                    temp_cfg["sl_mult"] = float(max(0.05, pol["sl_distance_atr_eff"]))
                if pol.get("tp_distance_atr_eff") is not None:
                    temp_cfg["tp_mult"] = float(max(0.05, pol["tp_distance_atr_eff"]))

                # Per-bucket profit-protection params (absolute % of price)
                for pp_key in ("be_threshold_pct", "profit_lock_pct", "profit_lock_amount", "giveback_pct", "max_loss_pct"):
                    if pp_key in rp:
                        temp_cfg[pp_key] = rp[pp_key]

                # Per-bucket vol scaling params (from risk optimization)
                if "vol_lo" in rp: temp_cfg["vol_lo"] = rp["vol_lo"]
                if "vol_hi" in rp: temp_cfg["vol_hi"] = rp["vol_hi"]
                if "vol_z_max" in rp: temp_cfg["vol_z_max"] = rp["vol_z_max"]

                # Per-bucket max hold hours (from risk optimization, default 24)
                hold_hours = int(pol.get("max_hold_hours_eff", rp.get("max_hold_hours", 24)))

                # Initialize CCXT exchange for 15m precision if enabled
                exchange = None
                if cfg.get("use_15m_precision", False):
                    try:
                        import ccxt
                        exchange = ccxt.binance()
                    except Exception as e:
                        tprint(f"WARNING: Failed to initialize CCXT exchange: {e}")

                ret, exit_ts, reason, trade_extras = simulate_trade_hourly(
                    o_s[sym], h_s[sym], l_s[sym], c_s[sym], atr_s[sym],
                    entry_ts, entry_px, side, temp_cfg, max_hold_hours=hold_hours,
                    exchange=exchange, symbol=sym if "/" in sym else sym.replace("USDT", "/USDT"), cost=cost  # CCXT format
                )
                assert_pos_w(weight)
                side_sign = 1 if side == "long" else -1
                
                # Dynamic cost model: handle limit fills and conditional fee concessions
                fee_reduction_reasons = ["stop_loss", "early_invalidation", "giveback_exit"]
                
                # Use new fee structure: market vs limit order fees
                baseline_fee_side = float(cfg.get("fee_bps_market", 25.0)) / 10000.0
                
                # Entry fee: use limit fee if filled via limit order
                if trade_extras.get("filled_via_limit"):
                    entry_fee = float(cfg.get("fee_bps_limit_entry", 10.0)) / 10000.0
                else:
                    entry_fee = baseline_fee_side
                
                # Exit fee: determine if using limit order exit or market order exit
                # If exit reason is not timeout and we're using exit limits
                use_exit_limit = cfg.get("use_exit_limit_orders", False)
                if use_exit_limit and reason not in ["time_exit", "timeout"]:
                    exit_fee = float(cfg.get("fee_bps_limit_exit", 10.0)) / 10000.0
                elif reason in fee_reduction_reasons:
                    # Apply concession for specific exit reasons
                    exit_fee = max(0.0020, baseline_fee_side - 0.0015)
                else:
                    exit_fee = float(cfg.get("fee_bps_market_exit", 25.0)) / 10000.0
                
                # Aggregate into symmetric CostModel for trade_return_net (expects fee_side * 2)
                avg_fee_side = (entry_fee + exit_fee) / 2.0
                from extreme_price_movements.pnl import CostModel
                actual_cost = CostModel(fee_side=avg_fee_side, slippage_side=float(cfg.get("slippage_bps", 0.0)) / 10000.0)
                
                net_ret = trade_return_net(raw_ret_underlying=ret, side=side_sign, pos_w=weight, cost=actual_cost)
                pnl = net_ret
                
                # Store risk parameters + MAE/MFE for aggregate statistics
                bucket_label = f"{side.upper()}_{dom.upper()}"
                trade_record = {
                    "entry_ts": entry_ts, "symbol": sym, "side": side, "dom": dom,
                    "asset": sym, "t_entry": int(entry_ts.value), "t_exit": int(exit_ts.value),
                    "bucket": bucket_label,
                    "score": score, "weight": weight, "pos_w": weight, "ret": net_ret, "pnl": pnl,
                    "net_ret_equity": net_ret, "cost_rt": cost.round_trip, "raw_ret_underlying": ret,
                    "gross_ret": ret, "exit_ts": exit_ts, "reason": reason, "exit_reason": reason,
                    "sl_mult": temp_cfg.get("sl_mult", 0.5),
                    "tp_mult": temp_cfg.get("tp_mult", 1.0),
                    "trail_mult": temp_cfg.get("trail_mult", 0.25),
                    "entry_px": entry_px,
                    "atr": float(atr_s[sym].loc[entry_ts]) if entry_ts in atr_s[sym].index else 0.02,
                    "sl_pct": trade_extras.get("sl_pct", 0.0),
                    "tp_pct": trade_extras.get("tp_pct", 0.0),
                    "mae_pct": trade_extras.get("mae_pct", 0.0),
                    "mfe_pct": trade_extras.get("mfe_pct", 0.0),
                    "bars_to_mfe": trade_extras.get("bars_to_mfe", 0),
                    "exit_stage": trade_extras.get("exit_stage", 0),
                    "filled_via_limit": trade_extras.get("filled_via_limit", False),
                    "exit_filled_via_limit": trade_extras.get("exit_filled_via_limit", False),
                    "exit_limit_bonus": trade_extras.get("exit_limit_bonus", 0.0),
                    "u_hat_z": pol.get("u_hat_z", 0.0),
                    "mae_hat_z": pol.get("mae_hat_z", 0.0),
                    "mfe_hat_z": pol.get("mfe_hat_z", 0.0),
                    "dur_hat_z": pol.get("dur_hat_z", 0.0),
                    "signal_px": entry_px,
                    "entry_px_fill": pol.get("entry_px_fill", entry_px),
                    "delta_atr_star": pol.get("delta_atr_star", 0.0),
                    "delta_price_star": pol.get("delta_price_star", 0.0),
                    "p_fill_star": pol.get("p_fill_star", 1.0),
                    "eu_star": pol.get("eu_star", 0.0),
                    "place_order": True,
                    "sl_distance_atr_eff": pol.get("sl_distance_atr_eff", np.nan),
                    "tp_distance_atr_eff": pol.get("tp_distance_atr_eff", np.nan),
                    "trail_mult_eff": pol.get("trail_mult_eff", np.nan),
                    "giveback_pct_eff": pol.get("giveback_pct_eff", np.nan),
                    "profit_lock_amount_eff": pol.get("profit_lock_amount_eff", np.nan),
                    "kill_c_eff": pol.get("kill_c_eff", np.nan),
                    "max_hold_hours_eff": pol.get("max_hold_hours_eff", np.nan),
                }
                # Add regime context for diagnostic reporting
                if t in mkt_gates.index:
                    mrk = mkt_gates.loc[t]
                    trade_record["G_VOL"] = int(mrk.get("G_VOL", 0)) if "G_VOL" in mrk.index else 0
                    trade_record["G_TREND"] = int(mrk.get("G_TREND", 0)) if "G_TREND" in mrk.index else 0
                    trade_record["mkt_rv"] = float(mrk.get("mkt_rv", 0.0)) if "mkt_rv" in mrk.index else 0.0
                    trade_record["mkt_ret24h"] = float(mrk.get("mkt_ret24h", 0.0)) if "mkt_ret24h" in mrk.index else 0.0
                trades.append(trade_record)
                diag["executed_trades"] += 1
                _rs = str(reason)
                diag["reason_counts"][_rs] = int(diag["reason_counts"].get(_rs, 0) + 1)
                # Update daily concentration counters
                daily_bucket_counts[trade_date][bucket_label] += 1
                daily_total_counts[trade_date] += 1
        
        # Log aggregate TP/SL statistics (using actual barrier-scaled distances from engine)
        if trades:
            reason_counts = {}
            for tr in trades:
                rs = str(tr.get("reason", "unknown"))
                reason_counts[rs] = reason_counts.get(rs, 0) + 1
            n_tr = max(1, len(trades))
            tp_sl_binds = reason_counts.get("stop_loss", 0) + reason_counts.get("trailing_stop", 0)
            hold_by_reason = {}
            mean_ret_by_reason = {}
            
            entry_bps = float(cfg.get("limit_offset_bps", 0.0))
            exit_bps = float(cfg.get("exit_limit_offset_bps", 0.0))
            offset_bps = entry_bps / 10000.0
            
            tprint(f"\n  Gross PnL Comparison (Market vs {entry_bps:.0f}bps Entry / {exit_bps:.0f}bps Exit) by Exit Reason:")
            
            for rs in sorted(reason_counts.keys()):
                rs_rows = [tr for tr in trades if str(tr.get("reason")) == rs]
                holds = [float((pd.Timestamp(tr["exit_ts"]) - pd.Timestamp(tr["entry_ts"])).total_seconds()/3600.0) for tr in rs_rows]
                rets = [float(tr.get("ret", 0.0)) for tr in rs_rows]
                
                # Calculate gross unscaled returns isolated from capital limits and weights
                market_gross_rets = []
                limit_gross_rets = []
                for tr in rs_rows:
                    g = float(tr.get("gross_ret", 0.0)) # contains both limit entry and limit exit
                    limit_gross_rets.append(g)
                    
                    # Deduct the extra margin gained by limit orders to simulate the "Market Order" baseline
                    entry_bonus = offset_bps if tr.get("filled_via_limit") else 0.0
                    exit_bonus = float(tr.get("exit_limit_bonus", 0.0))
                    market_gross_rets.append(g - entry_bonus - exit_bonus)
                
                hold_by_reason[rs] = float(np.median(holds)) if holds else 0.0
                mean_ret_by_reason[rs] = float(np.mean(rets)) if rets else 0.0
                
                avg_mkt_g = float(np.mean(market_gross_rets)) if market_gross_rets else 0.0
                avg_lmt_g = float(np.mean(limit_gross_rets)) if limit_gross_rets else 0.0
                tprint(f"    {rs.ljust(20)}: Count={len(rs_rows):<3} | Market Gross: {avg_mkt_g*100:6.3f}% | Limit Gross: {avg_lmt_g*100:6.3f}% | Delta: +{(avg_lmt_g - avg_mkt_g)*100:5.3f}%")

            # === Detailed PnL Comparison Metrics ===
            try:
                from extreme_price_movements.limit_order_pricer import compute_pnl_comparison_metrics, compute_exit_limit_fill_impact
                
                # Convert trades list to DataFrame for metrics computation
                trades_df = pd.DataFrame(trades)
                
                # Validate required columns exist before computing metrics
                required_cols = ["entry_px"]
                if not all(col in trades_df.columns for col in required_cols):
                    tprint(f"  WARNING: Skipping PnL comparison - missing required columns: {required_cols}")
                elif trades_df.empty:
                    tprint("  WARNING: Skipping PnL comparison - no trades to analyze")
                else:
                    # Get exit price from trade records
                    if "exit_px" not in trades_df.columns and "gross_ret" in trades_df.columns:
                        # Calculate exit price from gross return
                        trades_df["is_long"] = (trades_df["side"] == "long").astype(int)
                        trades_df["exit_px"] = np.where(
                            trades_df["is_long"] == 1,
                            trades_df["entry_px"] * (1.0 + trades_df["gross_ret"]),
                            trades_df["entry_px"] * (1.0 - trades_df["gross_ret"])
                        )
                
                # Compute comprehensive PnL comparison metrics
                comparison_metrics = compute_pnl_comparison_metrics(
                    trades_df,
                    entry_offset_bps=entry_bps,
                    exit_offset_bps=exit_bps,
                    mae_hat_col="mae_hat_z",
                    mfe_hat_col="mfe_hat_z",
                    is_long_col="is_long",
                    entry_price_col="entry_px",
                    exit_price_col="exit_px",
                    fee_limit_entry=float(cfg.get("fee_bps_limit_entry", 10.0)) / 10000.0,
                    fee_limit_exit=float(cfg.get("fee_bps_limit_exit", 10.0)) / 10000.0,
                    fee_market_entry=float(cfg.get("fee_bps_market", 25.0)) / 10000.0,
                    fee_market_exit=float(cfg.get("fee_bps_market_exit", 25.0)) / 10000.0,
                )
                
                # Print detailed comparison
                tprint(f"\n  === Detailed PnL Comparison (Solution A vs B vs Market) ===")
                tprint(f"    Total Trades: {comparison_metrics['n_trades']}")
                tprint(f"    Fee Savings (Entry): {comparison_metrics['fee_savings_entry']*10000:.1f} bps")
                tprint(f"    Fee Savings (Exit): {comparison_metrics['fee_savings_exit']*10000:.1f} bps")
                
                tprint(f"\n  --- Mean PnL by Strategy ---")
                tprint(f"    Baseline (Limit Entry):    {comparison_metrics['baseline']['mean']*100:7.3f}% | Sharpe: {comparison_metrics['baseline']['sharpe']:.2f} | Win Rate: {comparison_metrics['baseline']['win_rate']*100:.1f}%")
                tprint(f"    Solution A (New TP/SL):   {comparison_metrics['solution_a']['mean']*100:7.3f}% | Sharpe: {comparison_metrics['solution_a']['sharpe']:.2f} | Win Rate: {comparison_metrics['solution_a']['win_rate']*100:.1f}%")
                tprint(f"    Solution B (Same Dist):   {comparison_metrics['solution_b']['mean']*100:7.3f}% | Sharpe: {comparison_metrics['solution_b']['sharpe']:.2f} | Win Rate: {comparison_metrics['solution_b']['win_rate']*100:.1f}%")
                tprint(f"    Market Order (No Offset): {comparison_metrics['market_order']['mean']*100:7.3f}% | Sharpe: {comparison_metrics['market_order']['sharpe']:.2f} | Win Rate: {comparison_metrics['market_order']['win_rate']*100:.1f}%")
                
                tprint(f"\n  --- PnL Differences ---")
                tprint(f"    Solution A vs Baseline: {comparison_metrics['diff_a_vs_baseline']*100:+6.3f}%")
                tprint(f"    Solution B vs Baseline: {comparison_metrics['diff_b_vs_baseline']*100:+6.3f}%")
                tprint(f"    Solution A vs Market:   {comparison_metrics['diff_a_vs_market']*100:+6.3f}%")
                tprint(f"    Solution B vs Market:   {comparison_metrics['diff_b_vs_market']*100:+6.3f}%")
                
                # Compute exit limit fill impact if exit_filled data is available
                exit_impact = compute_exit_limit_fill_impact(
                    trades_df,
                    exit_filled_col="exit_filled_via_limit",
                    exit_price_col="exit_px",
                    entry_price_col="entry_px",
                    is_long_col="is_long",
                )
                
                if "error" not in exit_impact:
                    tprint(f"\n  --- Exit Limit Order Fill Impact ---")
                    tprint(f"    Total Trades: {exit_impact['total_trades']}")
                    tprint(f"    Exit via Limit: {exit_impact['exit_limit_filled']} ({exit_impact['fill_rate']*100:.1f}%)")
                    tprint(f"    Exit via Market: {exit_impact['exit_limit_not_filled']}")
                    if "mean_pnl_filled" in exit_impact:
                        tprint(f"    Mean PnL (Filled): {exit_impact['mean_pnl_filled']*100:.3f}%")
                    if "mean_pnl_not_filled" in exit_impact:
                        tprint(f"    Mean PnL (Not Filled): {exit_impact['mean_pnl_not_filled']*100:.3f}%")
                
            except Exception as e:
                tprint(f"  WARNING: Could not compute detailed PnL comparison: {e}")

            emit_bucket_summary(
                tprint=tprint,
                run_id=run_id,
                bucket_id="ALL_RUNTIME",
                kind="runtime_exit",
                stats={
                    "n_trades": int(len(trades)),
                    "tp_sl_bind_rate": float(tp_sl_binds / n_tr),
                    "exit_reason_counts": reason_counts,
                    "median_hold_h_by_reason": hold_by_reason,
                    "mean_net_ret_equity_by_reason": mean_ret_by_reason,
                    "cost_rt": float(cost.round_trip),
                },
            )
            avg_sl_pct = np.mean([t["sl_pct"] * 100 for t in trades])
            avg_tp_pct = np.mean([t["tp_pct"] * 100 for t in trades])
            avg_trail_pct = np.mean([t["trail_mult"] * t["sl_pct"] * 100 for t in trades])  # trail ≈ trail_mult * barrier
            avg_mae = np.mean([t["mae_pct"] * 100 for t in trades])
            avg_mfe = np.mean([t["mfe_pct"] * 100 for t in trades])

            mae_20bps = np.mean([t["mae_pct"] * 100 >= 0.2 for t in trades]) * 100
            mae_30bps = np.mean([t["mae_pct"] * 100 >= 0.3 for t in trades]) * 100
            mae_40bps = np.mean([t["mae_pct"] * 100 >= 0.4 for t in trades]) * 100
            mae_50bps = np.mean([t["mae_pct"] * 100 >= 0.5 for t in trades]) * 100
            
            tprint(f"\n  TP/SL Statistics ({len(trades)} trades) [actual barrier-scaled]:")
            tprint(f"    Avg SL:    {avg_sl_pct:.2f}%")
            tprint(f"    Avg TP:    {avg_tp_pct:.2f}%")
            tprint(f"    Avg Trail: {avg_trail_pct:.2f}%")
            tprint(f"    Avg MAE:   {avg_mae:.2f}%")
            tprint(f"    Avg MFE:   {avg_mfe:.2f}%")
            tprint(f"    MAE >= 0.2%: {mae_20bps:.1f}% of trades")
            tprint(f"    MAE >= 0.3%: {mae_30bps:.1f}% of trades")
            tprint(f"    MAE >= 0.4%: {mae_40bps:.1f}% of trades")
            tprint(f"    MAE >= 0.5%: {mae_50bps:.1f}% of trades\n")
        
        if not trades and bool(cfg.get("signal_opt_debug", True)):
            tprint(
                "  SignalDiag[n=0]: "
                f"hours={diag['hours_total']} hours_with_orders={diag['hours_with_orders']} "
                f"orders_post_signal={diag['orders_post_signal']} orders_after_budget={diag['orders_after_budget']} "
                f"policy_blocked={diag['skipped_policy_place_order']} missing_open={diag['skipped_missing_open']}"
            )

        return trades, diag

    split = max(24, int(len(valid_ts) * 0.2))  # 20% for signal calibration, 80% for OOS test
    train_ts = valid_ts[:split]
    test_ts = valid_ts[split:]

    raw_train, _raw_diag = run_slice(train_ts, {
        "thr_long": -1e9, "thr_short": 1e9,
        "k_long": cfg.get("k_long", 10), "k_short": cfg.get("k_short", 10),
        "size_min": 0.03, "size_max": 0.15, "size_k": 2.0, "size_x0": 0.5, "size_zcap": 4.0,
    })
    train_abs = np.array([abs(t["score"]) for t in raw_train], dtype=np.float64)
    q50 = float(np.quantile(train_abs, 0.5)) if train_abs.size else 0.0
    q75 = float(np.quantile(train_abs, 0.75)) if train_abs.size else max(q50, 1e-6)
    q90 = float(np.quantile(train_abs, 0.9)) if train_abs.size else max(q50 + 1e-6, 1e-3)
    q95 = float(np.quantile(train_abs, 0.95)) if train_abs.size else max(q90 + 1e-6, 1.1 * q90)
    q98 = float(np.quantile(train_abs, 0.98)) if train_abs.size else max(q95 + 1e-6, 1.1 * q95)
    tprint(f"Global Score Distribution: P50={q50:.6f}, P75={q75:.6f}, P90={q90:.6f}, P95={q95:.6f}, P98={q98:.6f}")

    # Calibrate long/short meta-score comparability on train only.
    side_frames = []
    for t in train_ts:
        s_df = _build_side_score_df(t, feats, mkt_gates, bundle, cfg, p_exh_hist, [])
        if not s_df.empty:
            side_frames.append(s_df)

    if side_frames:
        side_all = pd.concat(side_frames, ignore_index=True)

        def _center_scale(arr, channel_name=""):
            """Robust scaling: median / IQR with winsorization to [q05, q95]."""
            v = np.asarray(arr, dtype=np.float64)
            if v.size == 0:
                return 0.0, 1.0
            # Winsorize to [q05, q95] to tame heavy tails (esp. LONG_MR)
            q05 = float(np.quantile(v, 0.05))
            q95 = float(np.quantile(v, 0.95))
            v_w = np.clip(v, q05, q95)
            c = float(np.median(v_w))
            # IQR-based scale (robust to outliers)
            q25 = float(np.quantile(v_w, 0.25))
            q75 = float(np.quantile(v_w, 0.75))
            s = q75 - q25  # IQR
            min_meaningful_scale = 0.001
            if s < min_meaningful_scale:
                tprint(f"  ScoreScale WARNING: {channel_name} has degenerate IQR "
                       f"({s:.2e}). Disabling normalization (center=0, scale=1).")
                return 0.0, 1.0
            n_clipped = int(np.sum(arr < q05) + np.sum(arr > q95))
            if n_clipped > 0:
                tprint(f"  ScoreScale: {channel_name} winsorized {n_clipped}/{len(arr)} outliers "
                       f"to [{q05:.4f}, {q95:.4f}]")
            return c, s

        lmr = side_all[side_all["side_key"] == "long"]["score_mr"].values
        smr = side_all[side_all["side_key"] == "short"]["score_mr"].values
        ltf = side_all[side_all["side_key"] == "long"]["score_tf"].values
        stf = side_all[side_all["side_key"] == "short"]["score_tf"].values

        lmr_c, lmr_s = _center_scale(lmr, "long_mr")
        smr_c, smr_s = _center_scale(smr, "short_mr")
        ltf_c, ltf_s = _center_scale(ltf, "long_tf")
        stf_c, stf_s = _center_scale(stf, "short_tf")

        score_scale_params = {
            "long_mr_center": lmr_c, "long_mr_scale": lmr_s,
            "short_mr_center": smr_c, "short_mr_scale": smr_s,
            "long_tf_center": ltf_c, "long_tf_scale": ltf_s,
            "short_tf_center": stf_c, "short_tf_scale": stf_s,
        }
        tprint(f"Score scale params: lmr=({lmr_c:.4f},{lmr_s:.4f}) smr=({smr_c:.4f},{smr_s:.4f}) "
               f"ltf=({ltf_c:.4f},{ltf_s:.4f}) stf=({stf_c:.4f},{stf_s:.4f})")
        # Log raw score distributions for diagnostics
        for name, arr in [("long_mr", lmr), ("short_mr", smr), ("long_tf", ltf), ("short_tf", stf)]:
            if len(arr) > 0:
                tprint(f"  {name} scores: n={len(arr)}, mean={np.mean(arr):.6f}, std={np.std(arr):.6f}, "
                       f"q10={np.quantile(arr,0.1):.6f}, q50={np.quantile(arr,0.5):.6f}, q90={np.quantile(arr,0.9):.6f}")
    else:
        score_scale_params = {}

    def _uniq_sorted(vals):
        out = []
        for v in vals:
            vv = float(max(0.0, v))
            if np.isfinite(vv):
                out.append(vv)
        return sorted(set(out))

    thr_long_grid = _uniq_sorted([0.0, q50, q75, q90])
    thr_short_grid = _uniq_sorted([0.0, q50, q75, q90])
    x0_grid = [0.5, 0.7, 0.9]
    k_grid = [2.0, 4.0]
    score_gate_q_grid = [0.0, 0.50, 0.70]
    top_frac = float(cfg.get("signal_top_frac", 0.30))

    combos = []
    for tl in thr_long_grid:
        for ts_ in thr_short_grid:
            for x0 in x0_grid:
                for k in k_grid:
                    for score_gate_q in score_gate_q_grid:
                        params = {
                            "thr_long": tl,
                            "thr_short": ts_,
                            "k_long": cfg.get("k_long", 10),
                            "k_short": cfg.get("k_short", 10),
                            "k_frac_long": top_frac,
                            "k_frac_short": top_frac,
                            "score_gate_q": score_gate_q,
                            "size_min": 0.03,
                            "size_max": 0.15,
                            "size_k": k,
                            "size_x0": x0,
                            "size_zcap": 4.0,
                            "size_q50": q50,
                            "size_q90": q90,
                            "size_q95": q95,
                            "size_q98": q98,
                            "score_scale_params": score_scale_params,
                        }
                        tr, _diag = run_slice(train_ts, params)
                        pnl, sortino, max_dd, win_rate, count = compute_metrics(tr)
                        tprint(
                            f"SignalOpt tl={tl:.6f} ts={ts_:.6f} gate_q={score_gate_q:.2f} "
                            f"k={k:.2f} x0={x0:.2f} -> pnl={pnl:.6f} sortino={sortino:.6f} "
                            f"maxdd={max_dd:.6f} wr={win_rate:.2f} n={count}"
                        )
                        combos.append((params, pnl, sortino, max_dd))

    if combos:
        pnls = np.array([c[1] for c in combos], dtype=np.float64)
        sorts = np.array([c[2] for c in combos], dtype=np.float64)
        dds = np.array([c[3] for c in combos], dtype=np.float64)
        util = pnl_sortino_dd_utility(pnls, sorts, dds)
        best_i = int(np.argmax(util))
        best_signal_params = combos[best_i][0]
    else:
        best_signal_params = {
            "thr_long": cfg.get("thr_long", q75), "thr_short": cfg.get("thr_short", q75),
            "k_long": cfg.get("k_long", 10), "k_short": cfg.get("k_short", 10),
            "k_frac_long": top_frac, "k_frac_short": top_frac,
            "score_gate_q": 0.70,
            "size_min": 0.03, "size_max": 0.15, "size_k": 2.0, "size_x0": 0.5,
            "size_zcap": 4.0, "size_q50": q50, "size_q90": q90, "size_q95": q95, "size_q98": q98,
            "score_scale_params": score_scale_params,
        }

    tprint(f"Selected signal params: {best_signal_params}")

    test_trades, test_diag = run_slice(test_ts, best_signal_params)
    pnl, sortino, max_dd, win_rate, count = compute_metrics(test_trades)
    avg_pnl = pnl / count if count > 0 else 0.0
    tprint(f"Backtest OOS Result: Trades={count}, PnL={pnl:.6f}, AvgPnL={avg_pnl:.6f}, Sortino={sortino:.6f}, MaxDD={max_dd:.6f}, WinRate={win_rate:.2f}")
    if bool(cfg.get("signal_opt_debug", True)):
        tprint(
            "OOS SignalDiag: "
            f"hours={test_diag.get('hours_total', 0)} hours_with_orders={test_diag.get('hours_with_orders', 0)} "
            f"orders_post_signal={test_diag.get('orders_post_signal', 0)} orders_after_budget={test_diag.get('orders_after_budget', 0)} "
            f"policy_blocked={test_diag.get('skipped_policy_place_order', 0)} missing_open={test_diag.get('skipped_missing_open', 0)} "
            f"executed={test_diag.get('executed_trades', 0)} reasons={test_diag.get('reason_counts', {})}"
        )

    # Breakdown
    if test_trades:
        df_t = pd.DataFrame(test_trades)

        # Duration & frequency diagnostics
        ts_min = pd.Timestamp(df_t["entry_ts"].min())
        ts_max = pd.Timestamp(df_t["entry_ts"].max())
        n_days = max(1, (ts_max - ts_min).total_seconds() / 86400)
        tprint(f"--- OOS Period: {ts_min.date()} to {ts_max.date()} ({n_days:.0f} days) ---")
        tprint(f"  Total trades: {len(df_t)}, Trades/day: {len(df_t)/n_days:.1f}")

        tprint("--- OOS Breakdown ---")
        for side in ["long", "short"]:
            df_s = df_t[df_t["side"] == side]
            if not df_s.empty:
                s_pnl = df_s["pnl"].sum()
                s_wr = (df_s["pnl"] > 0).mean()
                tprint(f"  {side.upper()}: Trades={len(df_s)} ({len(df_s)/n_days:.1f}/day), PnL={s_pnl:.4f}, WinRate={s_wr:.2f}")
        for dom in ["mr", "tf"]:
            df_d = df_t[df_t["dom"] == dom]
            if not df_d.empty:
                d_pnl = df_d["pnl"].sum()
                d_wr = (df_d["pnl"] > 0).mean()
                tprint(f"  {dom.upper()}: Trades={len(df_d)} ({len(df_d)/n_days:.1f}/day), PnL={d_pnl:.4f}, WinRate={d_wr:.2f}")

        # Compute hold duration for all trades
        df_t["_entry"] = pd.to_datetime(df_t["entry_ts"])
        df_t["_exit"] = pd.to_datetime(df_t["exit_ts"])
        df_t["_hold_h"] = (df_t["_exit"] - df_t["_entry"]).dt.total_seconds() / 3600.0

        # --- Per-bucket deep diagnostics ---
        tprint("=" * 70)
        tprint("PER-BUCKET DIAGNOSTICS")
        tprint("=" * 70)
        for side in ["long", "short"]:
            for dom in ["mr", "tf"]:
                df_sd = df_t[(df_t["side"] == side) & (df_t["dom"] == dom)]
                if df_sd.empty:
                    continue
                bucket = f"{side}_{dom}"
                n = len(df_sd)
                sd_pnl = df_sd["pnl"].sum()
                sd_wr = (df_sd["pnl"] > 0).mean()
                avg_pnl = sd_pnl / n

                # Sortino ratio for this bucket
                rets = df_sd["pnl"].values
                neg = rets[rets < 0]
                down_std = np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else 1e-9
                bucket_sortino = np.mean(rets) / down_std if down_std > 1e-9 else 0.0

                # Max drawdown for this bucket (sequential equity curve)
                eq = np.cumsum(rets)
                running_max = np.maximum.accumulate(eq)
                dd = eq - running_max
                bucket_mdd = float(dd.min()) if len(dd) > 0 else 0.0

                # Profit factor
                gross_win = float(rets[rets > 0].sum()) if (rets > 0).any() else 0.0
                gross_loss = float(abs(rets[rets < 0].sum())) if (rets < 0).any() else 1e-9
                pf = gross_win / gross_loss

                tprint(f"\n--- {bucket.upper()} (n={n}, {n/n_days:.1f}/day) ---")
                tprint(f"  PnL={sd_pnl:.4f}  AvgPnL={avg_pnl:.6f}  WR={sd_wr:.2f}  Sortino={bucket_sortino:.3f}  MaxDD={bucket_mdd:.4f}  PF={pf:.2f}")

                # Win/loss asymmetry
                wins = df_sd[df_sd["pnl"] > 0]
                losses = df_sd[df_sd["pnl"] <= 0]
                avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0.0
                avg_loss = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
                payoff = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-9 else 0.0
                avg_win_ret = float(wins["gross_ret"].mean()) if len(wins) > 0 else 0.0
                avg_loss_ret = float(losses["gross_ret"].mean()) if len(losses) > 0 else 0.0
                tprint(f"  Win/Loss: AvgWin={avg_win:.6f} ({avg_win_ret:.4f} ret)  AvgLoss={avg_loss:.6f} ({avg_loss_ret:.4f} ret)  Payoff={payoff:.2f}")

                # Exit reason breakdown per bucket
                if "reason" in df_sd.columns:
                    reasons = df_sd["reason"].value_counts()
                    parts = []
                    for r in sorted(reasons.index):
                        r_df = df_sd[df_sd["reason"] == r]
                        r_pnl = r_df["pnl"].sum()
                        r_wr = (r_df["pnl"] > 0).mean()
                        parts.append(f"{r}:{len(r_df)}({r_pnl:+.4f}, WR={r_wr:.2f})")
                    tprint(f"  Exits: {' | '.join(parts)}")

                # Hold duration stats
                hold = df_sd["_hold_h"]
                tprint(f"  Hold(h): mean={hold.mean():.1f}  med={hold.median():.1f}  min={hold.min():.0f}  max={hold.max():.0f}")
                # Hold duration for wins vs losses
                if len(wins) > 0 and len(losses) > 0:
                    tprint(f"  Hold wins={wins['_hold_h'].mean():.1f}h  Hold losses={losses['_hold_h'].mean():.1f}h")

                # Score distribution
                sc = df_sd["score"].abs()
                tprint(f"  |Score|: mean={sc.mean():.3f}  med={sc.median():.3f}  q10={sc.quantile(0.1):.3f}  q90={sc.quantile(0.9):.3f}")

                # Spearman(|score|, ret) — key monotonicity diagnostic
                from scipy.stats import spearmanr
                if len(sc) >= 5:
                    sp_corr, sp_pval = spearmanr(sc.values, df_sd["ret"].values)
                    tprint(f"  Spearman(|score|, ret): {sp_corr:+.3f} (p={sp_pval:.3f})"
                           f"{'  *** NEGATIVE = conviction paradox ***' if sp_corr < -0.05 else ''}")

                # Survival metric: % of trades that reach trailing stop activation
                if "reason" in df_sd.columns:
                    n_trail = (df_sd["reason"] == "trailing_stop").sum()
                    n_sl = (df_sd["reason"] == "stop_loss").sum()
                    survival_rate = n_trail / max(1, n_trail + n_sl)
                    tprint(f"  Survival-to-trail: {survival_rate:.1%} ({n_trail} trail / {n_sl} SL)")

                # Score vs outcome: high-conviction vs low-conviction
                sc_med = sc.median()
                hi_conv = df_sd[sc >= sc_med]
                lo_conv = df_sd[sc < sc_med]
                if len(hi_conv) > 0 and len(lo_conv) > 0:
                    tprint(f"  Hi-conv(|s|>={sc_med:.3f}): n={len(hi_conv)} PnL={hi_conv['pnl'].sum():.4f} WR={(hi_conv['pnl']>0).mean():.2f}")
                    tprint(f"  Lo-conv(|s|< {sc_med:.3f}): n={len(lo_conv)} PnL={lo_conv['pnl'].sum():.4f} WR={(lo_conv['pnl']>0).mean():.2f}")
                    # Survival comparison by conviction
                    if "reason" in df_sd.columns:
                        hi_surv = (hi_conv["reason"] == "trailing_stop").sum() / max(1, len(hi_conv))
                        lo_surv = (lo_conv["reason"] == "trailing_stop").sum() / max(1, len(lo_conv))
                        tprint(f"  Trail-survival: Hi-conv={hi_surv:.1%}  Lo-conv={lo_surv:.1%}")

                # ========================================================================
                # CONFIDENCE QUARTILE ANALYSIS (ENHANCED)
                # ========================================================================
                try:
                    sc_abs = sc.abs()
                    df_sd = df_sd.copy()
                    df_sd["confidence_bin"] = pd.qcut(sc_abs, q=4, labels=["Q1_Low", "Q2", "Q3", "Q4_High"], duplicates='drop')
                    
                    tprint("  Confidence Calibration:")
                    tprint(f"    {'Quartile':<10} {'N':>4} {'WR':>5} {'PnL':>8} {'AvgRet':>8} {'MFE%':>6} {'MAE%':>6} {'MFE/MAE':>7} {'Capture':>7} {'Trail%':>6}")
                    for bin_label in ["Q1_Low", "Q2", "Q3", "Q4_High"]:
                        bt = df_sd[df_sd["confidence_bin"] == bin_label]
                        if len(bt) == 0:
                            continue
                        b_wr = (bt["pnl"] > 0).mean()
                        b_pnl = bt["pnl"].sum()
                        b_ret = bt["ret"].mean()
                        b_mfe = bt["mfe_pct"].mean() * 100
                        b_mae = bt["mae_pct"].mean() * 100
                        b_ratio = b_mfe / max(b_mae, 0.01)
                        # Capture ratio: avg gross_ret / avg MFE for winners
                        bt_w = bt[bt["pnl"] > 0]
                        b_cap = (bt_w["gross_ret"].mean() / max(bt_w["mfe_pct"].mean(), 1e-9)) if len(bt_w) > 0 else 0.0
                        # Trail survival %
                        b_trail = (bt["reason"] == "trailing_stop").mean() * 100 if "reason" in bt.columns else 0.0
                        tprint(f"    {bin_label:<10} {len(bt):>4} {b_wr:>5.2f} {b_pnl:>+8.4f} {b_ret:>+8.4f} {b_mfe:>6.2f} {b_mae:>6.2f} {b_ratio:>7.2f} {b_cap:>7.2f} {b_trail:>5.1f}%")
                    # Exit reason distribution per quartile
                    if "reason" in df_sd.columns:
                        tprint("  Exit Reasons by Confidence:")
                        for bin_label in ["Q1_Low", "Q2", "Q3", "Q4_High"]:
                            bt = df_sd[df_sd["confidence_bin"] == bin_label]
                            if len(bt) == 0:
                                continue
                            rc = bt["reason"].value_counts()
                            parts = [f"{r}:{c}" for r, c in rc.items()]
                            tprint(f"    {bin_label}: {' '.join(parts)}")
                except Exception as e:
                    tprint(f"  Warning: Confidence calibration failed: {e}")

                # ========================================================================
                # REGIME ANALYSIS: G_VOL × G_TREND
                # ========================================================================
                if "G_VOL" in df_sd.columns and "G_TREND" in df_sd.columns:
                    try:
                        tprint("  Regime Analysis (G_VOL × G_TREND):")
                        tprint(f"    {'Regime':<20} {'N':>4} {'WR':>5} {'PnL':>8} {'MFE%':>6} {'MAE%':>6} {'MFE/MAE':>7} {'Capture':>7}")
                        for gv in [0, 1]:
                            for gt in [0, 1]:
                                regime = df_sd[(df_sd["G_VOL"] == gv) & (df_sd["G_TREND"] == gt)]
                                if len(regime) < 3:
                                    continue
                                label = f"VOL={'Hi' if gv else 'Lo'}_TREND={'Hi' if gt else 'Lo'}"
                                r_wr = (regime["pnl"] > 0).mean()
                                r_pnl = regime["pnl"].sum()
                                r_mfe = regime["mfe_pct"].mean() * 100
                                r_mae = regime["mae_pct"].mean() * 100
                                r_ratio = r_mfe / max(r_mae, 0.01)
                                r_w = regime[regime["pnl"] > 0]
                                r_cap = (r_w["gross_ret"].mean() / max(r_w["mfe_pct"].mean(), 1e-9)) if len(r_w) > 0 else 0.0
                                tprint(f"    {label:<20} {len(regime):>4} {r_wr:>5.2f} {r_pnl:>+8.4f} {r_mfe:>6.2f} {r_mae:>6.2f} {r_ratio:>7.2f} {r_cap:>7.2f}")
                    except Exception as e:
                        tprint(f"  Warning: Regime analysis failed: {e}")


                # Weight/sizing stats
                wt = df_sd["weight"]
                tprint(f"  Weight: mean={wt.mean():.4f}  med={wt.median():.4f}  min={wt.min():.4f}  max={wt.max():.4f}")

                # Temporal half-split: is performance degrading?
                mid = len(df_sd) // 2
                if mid > 5:
                    first_half = df_sd.iloc[:mid]
                    second_half = df_sd.iloc[mid:]
                    fh_pnl = first_half["pnl"].sum()
                    sh_pnl = second_half["pnl"].sum()
                    fh_wr = (first_half["pnl"] > 0).mean()
                    sh_wr = (second_half["pnl"] > 0).mean()
                    tprint(f"  1st half: n={len(first_half)} PnL={fh_pnl:.4f} WR={fh_wr:.2f}  |  2nd half: n={len(second_half)} PnL={sh_pnl:.4f} WR={sh_wr:.2f}")

                # Top losing symbols
                sym_pnl = df_sd.groupby("symbol")["pnl"].agg(["sum", "count"])
                sym_pnl = sym_pnl.sort_values("sum")
                worst_3 = sym_pnl.head(3)
                best_3 = sym_pnl.tail(3).iloc[::-1]
                w_parts = [f"{s}({row['sum']:+.4f}, n={int(row['count'])})" for s, row in worst_3.iterrows()]
                b_parts = [f"{s}({row['sum']:+.4f}, n={int(row['count'])})" for s, row in best_3.iterrows()]
                tprint(f"  Worst syms: {', '.join(w_parts)}")
                tprint(f"  Best syms:  {', '.join(b_parts)}")

        # --- MAE/MFE DIAGNOSTIC REPORT ---
        if "mae_pct" in df_t.columns and "mfe_pct" in df_t.columns:
            tprint("\n" + "=" * 70)
            tprint("MAE / MFE DIAGNOSTIC REPORT")
            tprint("=" * 70)

            # Global MAE/MFE
            tprint(f"\n--- Global MAE/MFE (n={len(df_t)}) ---")
            tprint(f"  MAE: mean={df_t['mae_pct'].mean()*100:.2f}%  med={df_t['mae_pct'].median()*100:.2f}%  q90={df_t['mae_pct'].quantile(0.9)*100:.2f}%")
            tprint(f"  MFE: mean={df_t['mfe_pct'].mean()*100:.2f}%  med={df_t['mfe_pct'].median()*100:.2f}%  q90={df_t['mfe_pct'].quantile(0.9)*100:.2f}%")
            tprint(f"  MFE/MAE ratio: {df_t['mfe_pct'].mean() / max(df_t['mae_pct'].mean(), 1e-9):.2f}")

            # Per-bucket MAE/MFE
            bucket_col = "bucket" if "bucket" in df_t.columns else None
            if bucket_col is None:
                df_t["bucket"] = df_t["side"].str.upper() + "_" + df_t["dom"].str.upper()
                bucket_col = "bucket"

            for bkt in sorted(df_t[bucket_col].unique()):
                df_b = df_t[df_t[bucket_col] == bkt]
                if len(df_b) < 3:
                    continue
                tprint(f"\n  --- {bkt} (n={len(df_b)}) ---")
                tprint(f"    MAE: mean={df_b['mae_pct'].mean()*100:.2f}%  med={df_b['mae_pct'].median()*100:.2f}%")
                tprint(f"    MFE: mean={df_b['mfe_pct'].mean()*100:.2f}%  med={df_b['mfe_pct'].median()*100:.2f}%")
                ratio = df_b['mfe_pct'].mean() / max(df_b['mae_pct'].mean(), 1e-9)
                tprint(f"    MFE/MAE ratio: {ratio:.2f}  {'GOOD (>1.5)' if ratio > 1.5 else 'WEAK (<1.5) — entries or stops need work'}")

                # MAE/MFE by exit reason
                for reason in sorted(df_b["reason"].dropna().unique()):
                    df_br = df_b[df_b["reason"] == reason]
                    if len(df_br) < 2:
                        continue
                    tprint(f"    {reason}: n={len(df_br)}  MAE={df_br['mae_pct'].mean()*100:.2f}%  MFE={df_br['mfe_pct'].mean()*100:.2f}%  bars_to_mfe={df_br['bars_to_mfe'].mean():.0f}")

                # Key diagnostic: losers that had meaningful MFE (exit/stop problem)
                losers = df_b[df_b["pnl"] <= 0]
                if len(losers) > 0:
                    losers_with_mfe = losers[losers["mfe_pct"] > 0.005]  # >0.5% MFE
                    pct_losers_had_mfe = len(losers_with_mfe) / len(losers)
                    tprint(f"    Losers with MFE>0.5%: {pct_losers_had_mfe:.0%} ({len(losers_with_mfe)}/{len(losers)})"
                           f"{'  *** EXIT PROBLEM: many losers saw profit first ***' if pct_losers_had_mfe > 0.4 else ''}")
                    if len(losers_with_mfe) > 0:
                        tprint(f"      Avg MFE of those losers: {losers_with_mfe['mfe_pct'].mean()*100:.2f}%")

                # Key diagnostic: winners — how much MFE was captured?
                winners = df_b[df_b["pnl"] > 0]
                if len(winners) > 0:
                    capture_ratio = winners["gross_ret"].mean() / max(winners["mfe_pct"].mean(), 1e-9)
                    tprint(f"    Winner capture ratio (ret/MFE): {capture_ratio:.2f}"
                           f"{'  *** LOW CAPTURE: trailing too loose ***' if capture_ratio < 0.3 else ''}")

        # --- PnL RECONCILIATION TABLE ---
        tprint("\n" + "=" * 70)
        tprint("PnL RECONCILIATION TABLE")
        tprint("=" * 70)

        # All units in portfolio-weighted PnL (pnl = net_ret * weight)
        total_gross_profit = float(df_t.loc[df_t["pnl"] > 0, "pnl"].sum())
        total_gross_loss = float(df_t.loc[df_t["pnl"] <= 0, "pnl"].sum())
        total_net_pnl = total_gross_profit + total_gross_loss
        recon_pf = total_gross_profit / abs(total_gross_loss) if abs(total_gross_loss) > 1e-9 else float("inf")

        tprint(f"\n  Total Gross Profit:  {total_gross_profit:+.6f}  (portfolio-weighted PnL)")
        tprint(f"  Total Gross Loss:   {total_gross_loss:+.6f}")
        tprint(f"  Net PnL:            {total_net_pnl:+.6f}")
        tprint(f"  Profit Factor:      {recon_pf:.3f}")

        # Fee impact
        total_fees = float((cost.round_trip * df_t["weight"].abs()).sum())
        gross_pnl_before_fees = float(df_t["gross_ret"].mul(df_t["weight"]).sum())
        tprint(f"\n  Gross PnL (pre-fee): {gross_pnl_before_fees:+.6f}")
        tprint(f"  Total Fees:          {total_fees:+.6f}")
        tprint(f"  Net PnL (post-fee):  {gross_pnl_before_fees - total_fees:+.6f}")

        # Per-bucket contribution (same units)
        tprint(f"\n  --- Per-Bucket Contribution (portfolio-weighted PnL) ---")
        tprint(f"  {'Bucket':<15} {'N':>5} {'GrossProfit':>12} {'GrossLoss':>12} {'NetPnL':>12} {'PF':>6} {'WR':>6} {'AvgWin':>10} {'AvgLoss':>10}")
        for bkt in sorted(df_t[bucket_col].unique()):
            df_b = df_t[df_t[bucket_col] == bkt]
            b_gp = float(df_b.loc[df_b["pnl"] > 0, "pnl"].sum())
            b_gl = float(df_b.loc[df_b["pnl"] <= 0, "pnl"].sum())
            b_net = b_gp + b_gl
            b_pf = b_gp / abs(b_gl) if abs(b_gl) > 1e-9 else float("inf")
            b_wr = (df_b["pnl"] > 0).mean()
            b_aw = float(df_b.loc[df_b["pnl"] > 0, "pnl"].mean()) if (df_b["pnl"] > 0).any() else 0.0
            b_al = float(df_b.loc[df_b["pnl"] <= 0, "pnl"].mean()) if (df_b["pnl"] <= 0).any() else 0.0
            tprint(f"  {bkt:<15} {len(df_b):>5} {b_gp:>+12.6f} {b_gl:>+12.6f} {b_net:>+12.6f} {b_pf:>6.2f} {b_wr:>6.2f} {b_aw:>+10.6f} {b_al:>+10.6f}")

        # Units sanity check
        avg_win_ret = float(df_t.loc[df_t["pnl"] > 0, "ret"].mean()) if (df_t["pnl"] > 0).any() else 0.0
        avg_loss_ret = float(df_t.loc[df_t["pnl"] <= 0, "ret"].mean()) if (df_t["pnl"] <= 0).any() else 0.0
        avg_weight = float(df_t["weight"].mean())
        tprint(f"\n  --- Units Check ---")
        tprint(f"  Avg Win (ret space):  {avg_win_ret:+.4f}")
        tprint(f"  Avg Loss (ret space): {avg_loss_ret:+.4f}")
        tprint(f"  Avg Weight:           {avg_weight:.4f}")
        tprint(f"  Implied AvgWin PnL:   {avg_win_ret * avg_weight:+.6f}  (should ~ match AvgWin above)")
        tprint(f"  Implied AvgLoss PnL:  {avg_loss_ret * avg_weight:+.6f}  (should ~ match AvgLoss above)")

        # --- Global exit reason breakdown ---
        if "reason" in df_t.columns:
            tprint("\n--- Exit Reasons (global) ---")
            for reason in sorted(df_t["reason"].dropna().unique()):
                df_r = df_t[df_t["reason"] == reason]
                r_wr = (df_r["pnl"] > 0).mean()
                r_hold = df_r["_hold_h"].mean()
                tprint(f"  {reason}: n={len(df_r)} ({len(df_r)/len(df_t)*100:.0f}%), PnL={df_r['pnl'].sum():.4f}, WR={r_wr:.2f}, AvgHold={r_hold:.1f}h")

        # Daily concentration check
        df_t["_date"] = df_t["_entry"].dt.date
        daily_counts = df_t.groupby("_date").size()
        tprint(f"\n--- Daily Concentration: max={daily_counts.max()}/day, mean={daily_counts.mean():.1f}/day ---")
        if daily_counts.max() > 20:
            worst_day = daily_counts.idxmax()
            df_wd = df_t[df_t["_date"] == worst_day]
            tprint(f"  Worst day {worst_day}: {len(df_wd)} trades, PnL={df_wd['pnl'].sum():.4f}")

        # Per-bucket daily concentration
        tprint("  Per-bucket daily max:")
        for bkt in sorted(df_t[bucket_col].unique()):
            df_b = df_t[df_t[bucket_col] == bkt]
            bkt_daily = df_b.groupby("_date").size()
            tprint(f"    {bkt}: max={bkt_daily.max()}/day, mean={bkt_daily.mean():.1f}/day")

        # Weekly PnL trend
        df_t["_week"] = df_t["_entry"].dt.isocalendar().week.astype(int)
        weekly = df_t.groupby("_week").agg(
            n=("pnl", "count"),
            pnl=("pnl", "sum"),
            wr=("pnl", lambda x: (x > 0).mean())
        )
        tprint("--- Weekly PnL ---")
        for wk, row in weekly.iterrows():
            bar = "+" * int(max(0, row["pnl"]) * 500) + "-" * int(max(0, -row["pnl"]) * 500)
            tprint(f"  W{wk:02d}: n={int(row['n']):3d}  PnL={row['pnl']:+.4f}  WR={row['wr']:.2f}  {bar}")

        df_t.drop(columns=["_date", "_entry", "_exit", "_hold_h", "_week", bucket_col], inplace=True, errors="ignore")
        tprint("-----------------------")

    if test_trades:
        df_res = pd.DataFrame(test_trades)
        out_path = os.path.join(cfg["data_root"], "artifacts", run_id, "backtest_results.parquet")
        df_res.to_parquet(out_path, index=False)
        tprint(f"Detailed results saved to {out_path}")

    # Generate backtest report
    if test_trades:
        try:
            report_path = generate_backtest_report(
                run_id=run_id,
                cfg=cfg,
                trades=test_trades,
                signal_params=best_signal_params,
                fee_rate=(cost.fee_side + cost.slippage_side),
                base_dir=cfg.get('reports_root'),
            )
            tprint(f"Backtest report saved to {report_path}")
        except Exception as e:
            tprint(f"WARNING: Failed to generate backtest report: {e}")

    risk_conf["signal_params"] = best_signal_params
    model_state["risk_params"] = risk_conf
    with open(state_file, "wb") as f:
        pickle.dump(model_state, f)
    tprint("Saved optimized signal params to trained state for inference use.")
    tprint("STEP: BACKTEST COMPLETE")
    return True

def run_feature_generation_step(ts_sig, margin_symbols, cfg, store, force_full_recompute: bool = False):
    tprint("STEP: FEATURE GENERATION START")
    tprint(f"Target Timestamp: {ts_sig}")

    # Check if feature files already exist for this timestamp (lightweight check only).
    ts_str = ts_sig.strftime("%Y%m%d_%H%M%S")
    feat_dir = os.path.join(cfg["data_root"], "features", ts_str)
    existing_files = sorted(glob.glob(os.path.join(feat_dir, "symbol=*.parquet")))
    expected_keys = _expected_feature_keys_from_cfg(cfg)
    backfill_keys: list[str] = []
    if existing_files and force_full_recompute:
        tprint(
            f"Features already exist: {len(existing_files)} symbol files. "
            "Force flag enabled: recomputing full feature set."
        )

    # 1. Define Universe
    # We want "all assets in our universe".
    # This implies the margin universe (Top M).
    try:
        # IMPORTANT: use pipeline timestamp for variance filtering.
        # Passing ts_sig=None makes variance filtering anchor to "now", which can
        # wrongly drop symbols for historical/backfill runs.
        train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    except Exception as exc:
        tprint(f"WARNING: get_training_universe failed ({exc}); falling back to local store symbol discovery")
        train_syms = _local_store_symbols(store)
        if cfg.get("market_basket"):
            train_syms = sorted(set(train_syms).union(set(cfg["market_basket"])))
        if not train_syms:
            tprint("CRITICAL: no symbols available from local store fallback.")
            return
    # Universe diagnostics still logged below; downstream skip reasons are explicit.

    tprint(f"Universe (Top {cfg['fetch_symbols_M']} Vol + Basket + VarianceFilter): {len(train_syms)} symbols")
    
    # 2. Load Data
    dfs = {}
    lookback_days = max(180, int(cfg["fetch_years"] * 365))
    
    # Load Market Basket First (Critical)
    for s in cfg["market_basket"]:
        if s not in train_syms:
            train_syms.append(s)
            
    loaded_syms = []
    skipped_log = []

    with Timer("Feature Gen Data Load"):
        for s in train_syms:
            df = store.load(s, end_ts=ts_sig) # Load up to ts_sig
            
            # Constraints Check
            if df.empty:
                skipped_log.append(f"{s}: Empty DataFrame")
                continue
            
            # Check length (at least 60 days for basic moving averages + volatility)
            min_rows = 24 * 60 
            if len(df) < min_rows:
                skipped_log.append(f"{s}: Insufficient data ({len(df)} rows < {min_rows})")
                continue
                
            # Check recent data freshness?
            last_ts = df.index[-1]
            if (ts_sig - last_ts).days > 7:
                 skipped_log.append(f"{s}: Stale data (Last: {last_ts}, Target: {ts_sig})")
                 continue

            dfs[s] = df.tail(24 * lookback_days)
            loaded_syms.append(s)

    tprint(f"Loaded {len(loaded_syms)} symbols. Skipped {len(skipped_log)}.")
    for msg in skipped_log:
        tprint(f"  [SKIP] {msg}")

    if not dfs:
        tprint("CRITICAL: No valid data found for feature generation.")
        return

    # 3. Compute Features (Panel)
    tprint("Constructing Panel...")
    panel = to_panel(dfs)

    # Strict cache completeness check against target period + symbol universe.
    if existing_files and not force_full_recompute:
        scan = _scan_feature_cache_light(
            ts_sig=ts_sig,
            data_root=cfg["data_root"],
            expected_keys=expected_keys,
            panel_close=panel["close"],
        )
        miss_keys = scan["missing_keys"] if scan else list(expected_keys)
        partial_keys = scan["partial_keys"] if scan else []
        backfill_keys = sorted(set(miss_keys + partial_keys))
        if backfill_keys:
            tprint(
                f"Feature cache incomplete for {ts_sig}: "
                f"missing={len(miss_keys)} partial={len(partial_keys)}. "
                "Backfilling missing/partial features only."
            )
            if scan:
                tprint(
                    f"Cache scan summary: files={scan['file_count']} "
                    f"required_symbols={scan['required_symbol_count']} "
                    f"available_expected_keys={scan['available_key_count']}/{len(expected_keys)}"
                )
                if scan["missing_symbols"]:
                    tprint(
                        "Missing symbol files: "
                        + ", ".join(scan["missing_symbols"][:20])
                        + (" ..." if len(scan["missing_symbols"]) > 20 else "")
                    )
                if scan["uncovered_symbols"]:
                    tprint(
                        "Time-coverage mismatch symbols: "
                        + ", ".join(scan["uncovered_symbols"][:20])
                        + (" ..." if len(scan["uncovered_symbols"]) > 20 else "")
                    )
            if miss_keys:
                tprint("Missing keys: " + ", ".join(miss_keys[:30]) + (" ..." if len(miss_keys) > 30 else ""))
            if partial_keys:
                tprint("Partial keys: " + ", ".join(partial_keys[:30]) + (" ..." if len(partial_keys) > 30 else ""))
        else:
            _n_syms = len(panel["close"].columns)
            _n_feats = len(expected_keys)
            tprint(
                f"Features already exist and cover full target period: "
                f"{_n_feats} features × {_n_syms} symbols. Skipping recomputation."
            )
            _generate_feature_health_reports(ts_sig, cfg["data_root"])
            tprint("STEP: FEATURE GENERATION COMPLETE (cached)")
            return

    tprint("Computing Market Features...")
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])

    # Memory guard: backfill mode can still be very heavy (full graph on full symbol set).
    # Run in symbol chunks and stream-save to cap peak RSS.
    chunk_size = int(cfg.get("feature_backfill_symbol_chunk_size", 140))
    use_chunked_backfill = (
        (bool(backfill_keys) or force_full_recompute)
        and chunk_size > 0
        and len(loaded_syms) > chunk_size
    )

    if use_chunked_backfill:
        import gc

        backfill_set = set(backfill_keys)
        tail_cutoffs, tail_stats = _build_tail_only_backfill_cutoffs(
            ts_sig=ts_sig,
            data_root=cfg["data_root"],
            panel_close=panel["close"],
            backfill_keys=backfill_keys,
        )
        tprint(
            "Tail-only backfill cutoffs: "
            f"eligible={tail_stats['eligible_tail_only']} "
            f"missing_file={tail_stats['missing_symbol_file']} "
            f"missing_cols={tail_stats['missing_backfill_columns']} "
            f"structural_or_interior={tail_stats['structural_or_interior']} "
            f"already_covered={tail_stats['already_covered']}"
        )
        all_syms = list(panel["close"].columns)
        total_chunks = (len(all_syms) + chunk_size - 1) // chunk_size
        unresolved_union: set[str] = set()
        total_saved_keys = 0
        tprint(
            f"Computing Asset Features (Hourly) in chunked backfill mode: "
            f"{len(all_syms)} symbols, chunk_size={chunk_size}, chunks={total_chunks}"
        )

        for ci, start in enumerate(range(0, len(all_syms), chunk_size), start=1):
            chunk_syms = all_syms[start:start + chunk_size]
            tprint(
                f"[Feature chunk {ci}/{total_chunks}] symbols={len(chunk_syms)} "
                f"({chunk_syms[0]} .. {chunk_syms[-1]})"
            )
            panel_chunk = {
                k: v.reindex(columns=chunk_syms).copy()
                for k, v in panel.items()
                if isinstance(v, pd.DataFrame)
            }
            feats_chunk, feat_index, feat_columns = compute_features_hourly(panel_chunk, mkt_gates.copy(), cfg)

            if backfill_keys and not force_full_recompute:
                unresolved = [k for k in backfill_keys if k not in feats_chunk]
                if unresolved:
                    unresolved_union.update(unresolved)
                feats_chunk = {k: v for k, v in feats_chunk.items() if k in backfill_set}
                
            tprint(f"[Feature chunk {ci}/{total_chunks}] saving {len(feats_chunk)} keys")
            if feats_chunk:
                if backfill_keys and not force_full_recompute:
                    chunk_cutoffs = {s: tail_cutoffs[s] for s in chunk_syms if s in tail_cutoffs}
                else:
                    chunk_cutoffs = None
                    
                save_features(
                    feats_chunk,
                    ts_sig,
                    cfg["data_root"],
                    min_timestamp_by_symbol=chunk_cutoffs if chunk_cutoffs else None,
                    feat_index=feat_index,
                    feat_columns=feat_columns,
                )
                total_saved_keys = len(feats_chunk)
            del panel_chunk, feats_chunk
            gc.collect()

        tprint(f"Computed + saved chunked backfill features: {total_saved_keys} keys")
        if unresolved_union:
            unresolved_sorted = sorted(unresolved_union)
            tprint(
                "Warning: some expected backfill keys were not produced by compute_features_hourly: "
                + ", ".join(unresolved_sorted[:30]) + (" ..." if len(unresolved_sorted) > 30 else "")
            )
    else:
        tprint("Computing Asset Features (Hourly)...")
        feats, feat_index, feat_columns = compute_features_hourly(panel, mkt_gates, cfg)
        min_ts_by_symbol = None

        if backfill_keys and not force_full_recompute:
            backfill_set = set(backfill_keys)
            unresolved = sorted([k for k in backfill_keys if k not in feats])
            feats = {k: v for k, v in feats.items() if k in backfill_set}
            min_ts_by_symbol, tail_stats = _build_tail_only_backfill_cutoffs(
                ts_sig=ts_sig,
                data_root=cfg["data_root"],
                panel_close=panel["close"],
                backfill_keys=backfill_keys,
            )
            tprint(f"Computed + saving only missing/partial features: {len(feats)} keys")
            tprint(
                "Tail-only backfill cutoffs: "
                f"eligible={tail_stats['eligible_tail_only']} "
                f"missing_file={tail_stats['missing_symbol_file']} "
                f"missing_cols={tail_stats['missing_backfill_columns']} "
                f"structural_or_interior={tail_stats['structural_or_interior']} "
                f"already_covered={tail_stats['already_covered']}"
            )
            if unresolved:
                tprint(
                    "Warning: some expected backfill keys were not produced by compute_features_hourly: "
                    + ", ".join(unresolved[:30]) + (" ..." if len(unresolved) > 30 else "")
                )
        elif force_full_recompute:
            tprint(f"Computed + saving full feature set: {len(feats)} keys")

        # 4. Save
        if feats:
            save_features(
                feats,
                ts_sig,
                cfg["data_root"],
                min_timestamp_by_symbol=min_ts_by_symbol,
                feat_index=feat_index,
                feat_columns=feat_columns,
            )
        else:
            tprint("No feature keys selected for save after missing/new filter; nothing to write.")

    tprint(f"Generated features for {len(loaded_syms)} symbols.")
    _generate_feature_health_reports(ts_sig, cfg["data_root"])
    tprint("STEP: FEATURE GENERATION COMPLETE")
