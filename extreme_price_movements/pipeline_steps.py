import gc
import glob
import json
import os
import pickle
import re
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.candidates import select_trade_candidates_vectorized
from extreme_price_movements.config import (
    CANON_HORIZONS,
    HELPER_BASE_FEATURES,
    POSITION_SIZER_V2_FEATURE_CONFIG,
)



from extreme_price_movements.data_store import (
    _atomic_write_parquet,
    _ensure_feature_frame_index,
    _write_feature_metadata,
    get_feature_bounds,
    load_artifact_df,
    load_features,
    load_features_selected,
    save_artifact_df,
    save_features,
    to_panel,
)
from extreme_price_movements.engine import (
    _build_side_score_df,
    generate_hourly_signals,
    simulate_trade_hourly,
)
from extreme_price_movements.entry_policy import (
    compute_entry_policy_decision,
    flatten_bucket_policy,
)
from extreme_price_movements.features import (
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.intraday_crypto_library import (
    INTRADAY_TRIGGER_COLUMNS,
    LOCATION_FILTER_COLUMNS,
    PERSISTED_INTRADAY_LIBRARY_COLUMNS,
)
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.model_loader import load_alpha_models
from extreme_price_movements.offline_optimisers.params_store import (
    apply_offline_optimizer_best_params,
)
from extreme_price_movements.pnl import CostModel, trade_return_net
from extreme_price_movements.pnl_asserts import assert_pos_w, assert_units
from extreme_price_movements.position_sizer.runtime import load_ev_decomposition_bundle
from extreme_price_movements.reports.bucket_report import (
    report_base_training,
    report_labels,
    report_meta_training,
    report_optimise,
    report_ridge_sizer,
)
try:
    from extreme_price_movements.reports.report_generator import (
        generate_backtest_report,
        generate_risk_report,
        generate_training_report,
    )
except ModuleNotFoundError:
    def generate_backtest_report(*args, **kwargs):
        raise ModuleNotFoundError(
            "extreme_price_movements.reports.report_generator is not available"
        )

    def generate_risk_report(*args, **kwargs):
        raise ModuleNotFoundError(
            "extreme_price_movements.reports.report_generator is not available"
        )

    def generate_training_report(*args, **kwargs):
        raise ModuleNotFoundError(
            "extreme_price_movements.reports.report_generator is not available"
        )
from extreme_price_movements.ridge_position_sizer import (
    RidgePositionSizer,
    run_ridge_position_sizer_step,
)
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.strategy_registry import (
    get_strategies,
    strategy_runtime_horizons,
)
from extreme_price_movements.telemetry.tprint_hooks import (
    emit_bucket_summary,
    emit_run_header,
)
from extreme_price_movements.feature_selection_extreme_events import mdi_feature_selection_v3
from extreme_price_movements.training import (  # generate_spike_anatomy_history,
    _cap_selected_features,
    _subsample_indices_time_balanced,
    generate_label_datasets,
    optimize_risk_params,
    train_models_from_artifacts,
)
from extreme_price_movements.training_utils import (
    get_base_feature_keys,
    get_meta_feature_keys,
    build_wide_tight_pair_features,
)
from extreme_price_movements.universe import (
    apply_hardcoded_universe_exclusions,
    get_training_universe,
    refresh_margin_universe_daily,
)
from extreme_price_movements.utils import Timer, tprint

# Priority order for quote-currency deduplication.
_QUOTE_PRIORITY: list[str] = ["USDT", "USDC", "BUSD", "EUR"]
FEATURE_HEALTH_CRITICAL_KEYS: tuple[str, ...] = (
    "loc_range_pos_24",
    "loc_vwap_dev_z_24",
    "loc_pullback_depth_24",
    "dist_ema50_atr",
    "ema50_slope",
    "prior_volatility",
    "trend_acceleration",
)
TRAINING_RESIDUALIZATION_FEATURE_KEYS: tuple[str, ...] = (
    "ema50_ema200_spread_atr",
    "atr_change_rate",
    "bars_in_high_vol_state_log_norm",
    "volatility_of_volatility_48",
    "trend_strength_percentile",
    "volatility_autocorr_48",
)


def load_features_for_stage_or_all(
    cfg: dict,
    ts_sig: pd.Timestamp,
    root_dir: str,
    feature_keys: list[str] | set[str] | tuple[str, ...] | None = None,
    symbols: list[str] | set[str] | tuple[str, ...] | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
):
    """Load features with optional active-stage restrictions."""
    stage_view = cfg.get("_active_stage_view")
    if stage_view:
        allowed_symbols = stage_view.get("symbols") or None
        if allowed_symbols is not None and symbols is not None:
            allowed_set = {str(sym) for sym in allowed_symbols}
            requested_set = {str(sym) for sym in symbols}
            symbols = sorted(allowed_set & requested_set)
            if not symbols:
                tprint(
                    f"Stage view {stage_view.get('stage_name', 'stage')}: "
                    "no overlapping symbols after restriction."
                )
                return None
        elif allowed_symbols is not None:
            symbols = allowed_symbols

        view_start = stage_view.get("allowed_start_ts")
        view_end = stage_view.get("allowed_end_ts")
        if view_start:
            view_start_ts = pd.to_datetime(view_start)
            start_ts = (
                view_start_ts if start_ts is None else max(start_ts, view_start_ts)
            )
        if view_end:
            view_end_ts = pd.to_datetime(view_end)
            end_ts = view_end_ts if end_ts is None else min(end_ts, view_end_ts)

        tprint(
            f"Loading features for stage {stage_view.get('stage_name', 'stage')}: "
            f"{len(symbols) if symbols is not None else 'ALL'} symbols, "
            f"from {start_ts} to {end_ts}"
        )

    return load_features_selected(
        ts=ts_sig,
        root_dir=root_dir,
        feature_keys=feature_keys,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=end_ts,
        allowed_periods=stage_view.get("allowed_periods") if stage_view else None,
    )


def _dedup_universe_by_base(symbols: list[str]) -> list[str]:
    """Return at most one symbol per base asset, preferring the highest-priority quote.

    Priority: USDT > USDC > BUSD > EUR. If a base asset has none of those quotes,
    it is kept as-is (preserves non-stablecoin pairs like BTC/ETH).

    Examples:
        [ETH/USDT, ETH/USDC, BTC/USDT] -> [ETH/USDT, BTC/USDT]
        [SOL/USDC, SOL/BUSD]            -> [SOL/USDC]
    """
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


def _meta_feature_keys_union(cfg) -> set[str]:
    keys = set()
    for head in ["reg", "clf", "mfe", "mae", "asym"]:
        keys.update(get_meta_feature_keys(head, cfg))
    keys.update(TRAINING_RESIDUALIZATION_FEATURE_KEYS)
    return {k for k in keys if isinstance(k, str) and k}


def _base_feature_keys_union(cfg) -> set[str]:
    keys: set[str] = set()
    for name in (
        "exh_feature_keys",
        "spike_feature_keys",
    ):
        vals = cfg.get(name, [])
        if isinstance(vals, (list, tuple)):
            for v in vals:
                if isinstance(v, str) and v:
                    keys.add(v)

    keys.update(get_base_feature_keys("long", cfg))
    keys.update(get_base_feature_keys("short", cfg))
    return keys
    return keys


def ensure_training_residualization_feature_keys(cfg: dict) -> dict:
    return dict(cfg)


def _cap_panel_rows(
    panel: dict[str, pd.DataFrame], max_rows: int = 300_000
) -> dict[str, pd.DataFrame]:
    """Deprecated compatibility shim.

    The old implementation randomly masked entries to reduce memory usage, but
    that destroys temporal integrity and can flatten downstream features. The
    proper fix is chunked panel processing, so this helper now leaves the panel
    untouched.
    """
    return panel
    return panel


def _cap_dataset_rows(
    df: pd.DataFrame | None, max_rows: int = 300_000
) -> pd.DataFrame | None:
    """DEPRECATED: Use SlicePlanner for temporal splitting instead of random sampling.

    Cap rows in a flat DataFrame via random sampling.
    This function breaks temporal integrity and should not be used for training data.
    """
    import warnings

    warnings.warn(
        "_cap_dataset_rows is deprecated and breaks temporal integrity. "
        "Use SlicePlanner for proper temporal splitting instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if df is None or len(df) <= max_rows:
        return df
    tprint(f"Capping dataset at {max_rows} rows (random sample)...")
    return df.sample(n=max_rows, random_state=42).sort_index()


def _expected_feature_keys_from_cfg(cfg) -> set[str]:
    keys: set[str] = set()
    for name in (
        "exh_feature_keys",
        "spike_feature_keys",
        "test_feature_keys",
    ):
        vals = cfg.get(name, [])
        if isinstance(vals, (list, tuple)):
            for v in vals:
                if isinstance(v, str) and v:
                    keys.add(v)

    keys.update(_base_feature_keys_union(cfg))
    keys.update(_meta_feature_keys_union(cfg))

    # 2. Position Sizer V2 features that are computable offline during the
    # feature-generation stage. Exclude keys that depend on OOF predictions,
    # later-stage training artifacts, or the dedicated position-sizer runtime
    # builder rather than compute_features_hourly().
    sizer_cfg = POSITION_SIZER_V2_FEATURE_CONFIG
    for sub_list in (
        "shared_feature_keys",
        "model1_edge_feature_keys",
        "model2_downside_feature_keys",
        "model3_uncertainty_feature_keys",
    ):
        for k in sizer_cfg.get(sub_list, []):
            if isinstance(k, str) and k:
                keys.add(k)

    # 3. Helper base features (for candidacy/breadth)
    keys.update(HELPER_BASE_FEATURES)

    # 4. Persisted intraday location library columns.
    # Do not include discrete LOC_*/LONG_*/SHORT_* trigger names here: they are
    # rule-state labels, not columns emitted by compute_features_hourly().
    keys.update(PERSISTED_INTRADAY_LIBRARY_COLUMNS)
    causal = cfg.get("causal_cols", [])
    if isinstance(causal, (list, tuple)):
        nonpersisted_intraday_rule_names = set(LOCATION_FILTER_COLUMNS).union(
            set(INTRADAY_TRIGGER_COLUMNS)
        )
        for k in causal:
            if isinstance(k, str) and k and k not in nonpersisted_intraday_rule_names:
                keys.add(k)

    # 5. Core features that downstream logic assumes are present.
    keys.update({"ret1h", "ret24h"})

    # 5b. Include all raw features referenced by dynamic strategy rule masks so
    # pre-meta risk optimization can resolve canonical masks without falling
    # back to zero-event default buckets.
    for strat in get_strategies(cfg):
        base_event_trigger = str(strat.get("base_event_trigger", "") or "")
        if not base_event_trigger:
            continue
        for feat_name in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|>=|==|<|>)", base_event_trigger
        ):
            if feat_name not in {"Composite"}:
                keys.add(feat_name)

    # 6. Technical Regime (Ridge) Features
    from extreme_price_movements.config import (
        CONTINUOUS_LOCATION_COLS,
        RIDGE_FEATURE_COLS,
    )

    keys.update(RIDGE_FEATURE_COLS)
    keys.update(CONTINUOUS_LOCATION_COLS)

    offline_unavailable_keys = {
        # OOF / ensemble state is produced later in training / model assembly.
        "oof_base_mean",
        "oof_base_std",
        "oof_base_min",
        "oof_base_max",
        "oof_base_range",
        "oof_meta_pred",
        "oof_meta_minus_base_mean",
        "oof_top2_gap",
        "oof_sign_agreement_frac",
        "oof_rank_among_candidates",
        "edge_pred",
        "downside_pred",
        "edge_minus_downside",
        "abs_edge_pred",
        "p_exh_lag1",
        # Position-sizer-specific runtime features are generated by
        # build_position_sizer_feature_frame(), not by compute_features_hourly().
        "impulse_speed",
        "impulse_acceleration",
        "wick_cluster_ratio",
        "rejection_bar_count",
        "rejection_ratio",
        "rejection_volume_ratio",
        "climax_volume_ratio",
        "reversal_volume_ratio",
        "reversal_bar_strength",
        "terminal_climax_volume",
        "terminal_vol_ratio",
        "terminal_volume_ratio",
        "post_impulse_persistence",
        "post_impulse_volume_persistence",
        "impulse_participation_volume",
        "impulse_volume_ratio",
        "impulse_volume_slope",
        "impulse_vol_ratio",
        "impulse_range_atr_ratio",
        "impulse_breakdown_score",
        "momentum_last_3bars_impulse_return",
        "drift_after_impulse",
        "range_last_3bars_impulse_range",
        "volatility_contraction_ratio",
        "volatility_asymmetry",
        "vol_compression_ratio",
        "volume_entropy",
        "volume_regime_shift",
        "volume_volatility",
        "wick_entropy",
        "wick_ratio_last_bar",
        "return_per_volume",
        "return_vol_ratio",
        "downside_semivol_12",
        "range_cv",
        "price_vs_ema_12_z",
        "price_vs_ema_24_z",
        "ema_12_minus_ema_24_z",
        "trend_slope_12_z",
        "trend_slope_24_z",
        "liquidity_shock_z",
        "rv_6",
        "rv_12",
        "rv_24",
        "range_1_atr",
        "range_3_atr",
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "slope_last_n_bars",
    }
    keys.difference_update(offline_unavailable_keys)
    return keys


def _labeling_feature_keys(cfg) -> set[str]:
    """Returns the minimal set of expected feature keys for the label generation step.

    This optimization aggressively strips all base features, since labels only need a
    few key metrics to compute barriers and filters. Final models will fetch their
    full feature sets dynamically during 'base_training' to avoid OOM.
    """
    keys = {
        "ret24h",
        "ret1h",
        "range_12h_pct",
        "volatility_zscore",
        "chop_score",
        "mkt_rv_24h",
        "mkt_rv_48h",
        "dist_ema_fast",
        "trend_pct",
    }
    dev_metric = cfg.get("trade_deviation_metric", "dist_ema_fast")
    if isinstance(dev_metric, str) and dev_metric:
        keys.add(dev_metric)
    exh_keys = cfg.get("exh_feature_keys", [])
    if isinstance(exh_keys, (list, tuple)):
        keys.update(exh_keys)

    try:
        from extreme_price_movements.lgbm_based_mask_generation import (
            extract_feature_names_from_key,
        )
        from extreme_price_movements.strategy_registry import get_strategies

        selected_strategies = cfg.get("strategies", [])
        if not isinstance(selected_strategies, (list, tuple)) or not selected_strategies:
            selected_strategies = get_strategies(cfg)
        tprint(
            f"Label feature key scope: using {len(selected_strategies)} selected strategies"
        )
        for strat in selected_strategies:
            rule_key = str(strat.get("base_event_trigger", "")).strip()
            if (
                not rule_key
                or rule_key.startswith("price_up")
                or rule_key.startswith("price_down")
            ):
                continue
            keys.update(extract_feature_names_from_key(rule_key))
    except Exception:
        pass
    return keys


def _align_features_to_panel(
    feats: dict, panel: dict[str, pd.DataFrame], symbols: list[str]
) -> dict:
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

        # Fast path: skip reindex if columns and index match exactly
        if list(df.columns) == list(symbols) and df.index.equals(close.index):
            out[k] = df.astype(np.float32, copy=False) if df.dtypes.iloc[0] != np.float32 else df
        else:
            out[k] = df.reindex(index=close.index, columns=symbols).astype(
                np.float32, copy=False
            )
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
    if not isinstance(panel, dict) or any(
        k not in panel for k in ("high", "low", "close")
    ):
        return feats

    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    if (
        not isinstance(high, pd.DataFrame)
        or not isinstance(low, pd.DataFrame)
        or not isinstance(close, pd.DataFrame)
    ):
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
        atr_ready = atr_existing.reindex(columns=target_syms).astype(
            np.float32, copy=False
        )
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
    atr_pct = (
        (atr / (c.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    )

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


def _generate_feature_health_reports(
    ts_sig: pd.Timestamp, data_root: str, allowed_symbols: list[str] | None = None
) -> dict | None:
    """
    Build feature quality reports from saved per-symbol parquet files.
    Outputs:
      - feature_health_symbol_summary.csv
      - feature_health_feature_detail.csv
    """
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(data_root, "features", run_id)
    files = sorted(glob.glob(os.path.join(in_dir, "symbol=*.parquet")))

    if allowed_symbols is not None:
        allowed_set = set(
            apply_hardcoded_universe_exclusions([str(s) for s in allowed_symbols])
        )
        filtered_files = []
        for f in files:
            sym_from_file = (
                os.path.basename(f)
                .replace("symbol=", "")
                .replace(".parquet", "")
                .replace("_", "/", 1)
            )
            sym_norm = apply_hardcoded_universe_exclusions([sym_from_file])
            if sym_norm and sym_norm[0] in allowed_set:
                filtered_files.append(f)
        files = filtered_files
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
            symbol = (
                os.path.basename(fpath)
                .replace("symbol=", "")
                .replace(".parquet", "")
                .replace("_", "/", 1)
            )
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
                    "leading_nan_cells": 0,
                    "interior_nan_cells": 0,
                    "trailing_nan_cells": 0,
                    "all_nan_features": 0,
                    "constant_features": 0,
                    "features_with_interior_nan": 0,
                }
            )
            continue

        feat_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        nan_counts = feat_df.isna().sum(axis=0).astype(np.int64)
        uniq_counts = feat_df.nunique(dropna=True).astype(np.int64)
        all_nan_mask = nan_counts == rows
        const_mask = (uniq_counts <= 1) & (~all_nan_mask)
        leading_nan_counts = pd.Series(0, index=feat_cols, dtype=np.int64)
        trailing_nan_counts = pd.Series(0, index=feat_cols, dtype=np.int64)
        interior_nan_counts = pd.Series(0, index=feat_cols, dtype=np.int64)

        arr = feat_df.to_numpy(dtype=np.float32, copy=False)
        for j, c in enumerate(feat_cols):
            col = arr[:, j]
            valid = np.isfinite(col)
            valid_n = int(valid.sum())
            nan_n = int(rows - valid_n)
            if nan_n <= 0:
                continue
            if valid_n <= 0:
                leading_nan_counts[c] = rows
                continue
            first_valid = int(np.argmax(valid))
            last_valid = int(rows - 1 - np.argmax(valid[::-1]))
            lead = first_valid
            trail = rows - 1 - last_valid
            interior = max(0, nan_n - lead - trail)
            leading_nan_counts[c] = lead
            trailing_nan_counts[c] = trail
            interior_nan_counts[c] = interior

        nan_cells = int(nan_counts.sum())
        total_cells = int(rows * n_features)
        symbol_rows.append(
            {
                "symbol": symbol,
                "rows": rows,
                "n_features": n_features,
                "nan_cells": nan_cells,
                "nan_pct_overall": float(100.0 * nan_cells / max(total_cells, 1)),
                "leading_nan_cells": int(leading_nan_counts.sum()),
                "interior_nan_cells": int(interior_nan_counts.sum()),
                "trailing_nan_cells": int(trailing_nan_counts.sum()),
                "all_nan_features": int(all_nan_mask.sum()),
                "constant_features": int(const_mask.sum()),
                "features_with_interior_nan": int((interior_nan_counts > 0).sum()),
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
                    "leading_nan_count": int(leading_nan_counts[c]),
                    "interior_nan_count": int(interior_nan_counts[c]),
                    "trailing_nan_count": int(trailing_nan_counts[c]),
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

    n_all_nan_features = (
        int(detail_df["is_all_nan"].sum()) if not detail_df.empty else 0
    )
    n_const_features = (
        int(detail_df["is_constant_non_nan"].sum()) if not detail_df.empty else 0
    )
    n_interior_nan_features = (
        int((detail_df["interior_nan_count"] > 0).sum()) if not detail_df.empty else 0
    )
    tprint(
        f"Feature health report saved: {summary_path}, {detail_path} | "
        f"symbols={len(summary_df)} all_nan_feature_rows={n_all_nan_features} "
        f"constant_feature_rows={n_const_features} "
        f"interior_nan_feature_rows={n_interior_nan_features}"
    )

    return {
        "summary_path": summary_path,
        "detail_path": detail_path,
        "symbols": int(len(summary_df)),
        "all_nan_feature_rows": n_all_nan_features,
        "constant_feature_rows": n_const_features,
        "interior_nan_feature_rows": n_interior_nan_features,
    }


def _feature_snapshot_health_issues(
    features: dict[str, pd.DataFrame | np.ndarray],
    critical_keys: tuple[str, ...] = FEATURE_HEALTH_CRITICAL_KEYS,
) -> list[str]:
    """Return human-readable health issues for key feature families.

    The miner should never repair a bad feature snapshot in-process. Instead, the
    generation step should rebuild the snapshot through the proper pipeline and
    this validator should fail fast if the resulting snapshot is still degenerate.
    """

    issues: list[str] = []
    for key in critical_keys:
        values = features.get(key)
        if values is None:
            issues.append(f"missing:{key}")
            continue

        if isinstance(values, pd.DataFrame):
            arr = values.to_numpy(dtype=np.float32, copy=False)
        else:
            arr = np.asarray(values, dtype=np.float32)

        if arr.size == 0:
            issues.append(f"empty:{key}")
            continue

        finite = np.isfinite(arr)
        finite_count = int(finite.sum())
        if finite_count == 0:
            issues.append(f"all_nan:{key}")
            continue

        finite_vals = arr[finite]
        if np.nanstd(finite_vals) <= 1e-12 or np.unique(finite_vals).size <= 1:
            issues.append(f"constant:{key}")

    return issues


def _feature_quality_issues_for_keys(
    features: dict[str, pd.DataFrame | np.ndarray],
    keys: list[str] | tuple[str, ...] | set[str] | None,
) -> list[str]:
    """Return missing/all-NaN/constant issues for an explicit feature-key set."""
    if not keys:
        return []

    issues: list[str] = []
    for key in keys:
        key = str(key)
        values = features.get(key)
        if values is None:
            issues.append(f"missing:{key}")
            continue

        if isinstance(values, pd.DataFrame):
            arr = values.to_numpy(dtype=np.float32, copy=False)
        else:
            arr = np.asarray(values, dtype=np.float32)

        if arr.size == 0:
            issues.append(f"empty:{key}")
            continue

        finite = np.isfinite(arr)
        finite_count = int(finite.sum())
        if finite_count == 0:
            issues.append(f"all_nan:{key}")
            continue

        finite_vals = arr[finite]
        if np.nanstd(finite_vals) <= 1e-12 or np.unique(finite_vals).size <= 1:
            issues.append(f"constant:{key}")

    return issues


def _missing_requested_feature_keys(
    features: dict[str, pd.DataFrame | np.ndarray],
    requested_keys: list[str] | tuple[str, ...] | set[str] | None,
) -> list[str]:
    """Return requested keys that were not produced in the current compute batch."""
    if not requested_keys:
        return []
    feature_keys = set(features.keys())
    return sorted(str(k) for k in requested_keys if str(k) not in feature_keys)


def _enforce_feature_snapshot_completeness(
    ts_sig: pd.Timestamp,
    data_root: str,
    expected_keys: set[str],
    panel_close: pd.DataFrame,
) -> dict[str, int]:
    """
    Rewrite per-symbol feature files so every required symbol has the full
    expected feature schema and required timestamp coverage.

    Missing feature values are left as NaN to preserve the distinction between
    "unavailable" and the valid numeric value 0.
    """
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(data_root, "features", run_id)
    if not os.path.exists(in_dir):
        return {
            "normalized_symbols": 0,
            "created_files": 0,
            "added_columns": 0,
            "added_rows": 0,
        }

    expected_cols = sorted(expected_keys)
    normalized_symbols = 0
    created_files = 0
    added_columns = 0
    added_rows = 0
    total_syms = len(panel_close.columns)

    tprint(
        "Normalizing feature snapshot completeness: "
        f"{total_syms} symbols x {len(expected_cols)} expected keys"
    )

    for i, sym in enumerate(panel_close.columns, start=1):
        sym = str(sym)
        valid_idx = panel_close.index[panel_close[sym].notna()]
        if len(valid_idx) == 0:
            continue
        required_index = pd.DatetimeIndex(valid_idx)
        safe_sym = sym.replace("/", "_")
        fpath = os.path.join(in_dir, f"symbol={safe_sym}.parquet")

        if os.path.exists(fpath):
            try:
                df_sym = pd.read_parquet(fpath)
            except Exception as exc:
                tprint(f"Warning: rebuilding unreadable feature file {fpath}: {exc}")
                df_sym = pd.DataFrame(index=required_index)
                created_files += 1
            else:
                df_sym, index_reason = _ensure_feature_frame_index(
                    df_sym, parquet_path=fpath
                )
                if index_reason not in {
                    None,
                    "ts_column_indexed",
                    "recovered_from_metadata",
                }:
                    tprint(
                        f"Warning: rebuilding feature file with invalid index {fpath}: {index_reason}"
                    )
                    df_sym = pd.DataFrame(index=required_index)
                    created_files += 1
        else:
            df_sym = pd.DataFrame(index=required_index)
            created_files += 1

        if "__symbol__" in df_sym.columns:
            df_sym = df_sym.drop(columns=["__symbol__"])

        before_cols = set(df_sym.columns)
        before_rows = len(df_sym.index)
        df_sym = df_sym.reindex(index=required_index, columns=expected_cols)
        df_sym = df_sym.astype(np.float32, copy=False)
        df_sym["__symbol__"] = sym
        _atomic_write_parquet(df_sym, fpath)
        _write_feature_metadata(fpath, sym, df_sym.index)

        added_columns += len(set(expected_cols) - before_cols)
        added_rows += max(0, len(required_index) - before_rows)
        normalized_symbols += 1

        if i % 50 == 0 or i == total_syms:
            tprint(
                f"Feature completeness normalization progress: {i}/{total_syms} "
                f"({(i / total_syms) * 100:.1f}%)"
            )

    tprint(
        "Feature completeness normalization complete: "
        f"symbols={normalized_symbols} created={created_files} "
        f"added_columns={added_columns} added_rows={added_rows}"
    )
    return {
        "normalized_symbols": normalized_symbols,
        "created_files": created_files,
        "added_columns": added_columns,
        "added_rows": added_rows,
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
        req_first = pd.Timestamp(valid_idx[0])
        req_last = pd.Timestamp(valid_idx[-1])
        if req_first.tzinfo is not None:
            req_first = req_first.tz_localize(None)
        if req_last.tzinfo is not None:
            req_last = req_last.tz_localize(None)
        required_bounds[s] = (req_first, req_last)

    key_symbol_counts: dict[str, int] = {k: 0 for k in expected_keys}
    present_symbols: set[str] = set()
    uncovered_symbols: set[str] = set()
    stale_symbols: set[str] = set()
    full_rewrite_symbols: set[str] = set()
    all_nan_symbol_keys: dict[str, set[str]] = {}

    total_files = len(files)
    progress_every = 100 if total_files >= 500 else 50
    for i, fpath in enumerate(files, start=1):
        fname = os.path.basename(fpath)
        sym = fname.replace("symbol=", "").replace(".parquet", "").replace("_", "/", 1)

        if sym in required_bounds:
            present_symbols.add(sym)
            req_first, req_last = required_bounds[sym]
            first_ts, last_ts = get_feature_bounds(fpath)
            if first_ts is not None and pd.Timestamp(first_ts).tzinfo is not None:
                first_ts = pd.Timestamp(first_ts).tz_localize(None)
            if last_ts is not None and pd.Timestamp(last_ts).tzinfo is not None:
                last_ts = pd.Timestamp(last_ts).tz_localize(None)
            if (
                first_ts is None
                or last_ts is None
                or first_ts > req_first
                or last_ts < req_last
            ):
                uncovered_symbols.add(sym)
                stale_symbols.add(sym)
                full_rewrite_symbols.add(sym)

        try:
            schema_names = set(pq.ParquetFile(fpath).schema.names)
        except Exception:
            schema_names = set()
            if sym in required_bounds:
                stale_symbols.add(sym)
                full_rewrite_symbols.add(sym)
        feat_cols = [c for c in schema_names if c in expected_keys]
        non_all_nan_cols = set(feat_cols)
        if sym in required_bounds and feat_cols:
            try:
                sample_df = pd.read_parquet(fpath, columns=feat_cols)
                non_all_nan_cols = {
                    c
                    for c in feat_cols
                    if c in sample_df.columns and not bool(sample_df[c].isna().all())
                }
                empty_cols = set(feat_cols) - non_all_nan_cols
                if empty_cols:
                    stale_symbols.add(sym)
                    full_rewrite_symbols.add(sym)
                    all_nan_symbol_keys[sym] = set(empty_cols)
            except Exception:
                stale_symbols.add(sym)
                full_rewrite_symbols.add(sym)
                non_all_nan_cols = set()

        for c in non_all_nan_cols:
            key_symbol_counts[c] += 1
        if sym in required_bounds and (
            not expected_keys.issubset(schema_names)
            or len(all_nan_symbol_keys.get(sym, set())) > 0
        ):
            stale_symbols.add(sym)

        if i % progress_every == 0 or i == total_files:
            tprint(
                f"Feature cache scan progress: {i}/{total_files} files "
                f"({(i / total_files) * 100:.1f}%)"
            )

    required_set = set(required_bounds.keys())
    missing_symbols = sorted(required_set - present_symbols)
    stale_symbols.update(missing_symbols)
    full_rewrite_symbols.update(missing_symbols)
    required_n = len(required_set)

    missing_keys: list[str] = []
    partial_keys: set[str] = set()
    for k in sorted(expected_keys):
        present_n = int(key_symbol_counts.get(k, 0))
        if present_n <= 0:
            missing_keys.append(k)
        elif present_n < required_n:
            partial_keys.add(k)

    # Removed: If some symbols have missing files or incomplete time bounds, all present keys are partial.
    # This was causing redundant re-computation of ~1000 features when only a few were missing.

    available_key_count = sum(
        1 for k in expected_keys if key_symbol_counts.get(k, 0) > 0
    )
    return {
        "in_dir": in_dir,
        "file_count": total_files,
        "required_symbol_count": required_n,
        "available_key_count": available_key_count,
        "missing_symbols": missing_symbols,
        "stale_symbols": sorted(stale_symbols),
        "full_rewrite_symbols": sorted(full_rewrite_symbols),
        "uncovered_symbols": sorted(uncovered_symbols),
        "missing_keys": missing_keys,
        "partial_keys": sorted(partial_keys),
        "all_nan_symbol_keys": {k: sorted(v) for k, v in all_nan_symbol_keys.items()},
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
        if req_first.tzinfo is not None:
            req_first = req_first.tz_localize(None)
        if req_last.tzinfo is not None:
            req_last = req_last.tz_localize(None)

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
        if first_ts is not None and pd.Timestamp(first_ts).tzinfo is not None:
            first_ts = pd.Timestamp(first_ts).tz_localize(None)
        if last_ts is not None and pd.Timestamp(last_ts).tzinfo is not None:
            last_ts = pd.Timestamp(last_ts).tz_localize(None)
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


def _validate_feature_snapshot_completeness(
    ts_sig: pd.Timestamp,
    data_root: str,
    expected_keys: set[str],
    panel_close: pd.DataFrame | None,
) -> None:
    """Fail fast if the persisted feature snapshot is still incomplete."""
    if panel_close is None or panel_close.empty:
        return

    scan = _scan_feature_cache_light(
        ts_sig=ts_sig,
        data_root=data_root,
        expected_keys=expected_keys,
        panel_close=panel_close,
    )
    if not scan:
        raise RuntimeError(
            "Feature snapshot validation failed: no cached feature files found."
        )

    remaining_missing_symbols = int(len(scan.get("missing_symbols", [])))
    remaining_uncovered_symbols = int(len(scan.get("uncovered_symbols", [])))
    remaining_missing_keys = int(len(scan.get("missing_keys", [])))
    remaining_partial_keys = int(len(scan.get("partial_keys", [])))
    if any(
        (
            remaining_missing_symbols,
            remaining_uncovered_symbols,
            remaining_missing_keys,
            remaining_partial_keys,
        )
    ):
        raise RuntimeError(
            "Feature snapshot validation failed after save: "
            f"missing_symbols={remaining_missing_symbols} "
            f"uncovered_symbols={remaining_uncovered_symbols} "
            f"missing_keys={remaining_missing_keys} "
            f"partial_keys={remaining_partial_keys}"
        )


def _load_close_panel_for_symbols(
    store,
    symbols: list[str],
    ts_sig: pd.Timestamp,
    lookback_days: int,
) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Load only close series needed for cheap cache-completeness checks."""
    close_map: dict[str, pd.Series] = {}
    loaded_syms: list[str] = []
    skipped_log: list[str] = []
    min_rows = 24 * 60
    for s in symbols:
        df = store.load(s, columns=["close"], end_ts=ts_sig)
        if df.empty:
            skipped_log.append(f"{s}: Empty DataFrame")
            continue
        if len(df) < min_rows:
            skipped_log.append(f"{s}: Insufficient data ({len(df)} rows < {min_rows})")
            continue
        last_ts = df.index[-1]
        if (ts_sig - last_ts).days > 180:
            skipped_log.append(f"{s}: Stale data (Last: {last_ts}, Target: {ts_sig})")
            continue
        ser = df["close"].tail(24 * lookback_days).rename(s)
        close_map[s] = ser
        loaded_syms.append(s)
    if not close_map:
        return None, loaded_syms, skipped_log
    close_panel = pd.concat(close_map.values(), axis=1).sort_index()
    return close_panel, loaded_syms, skipped_log


def _derive_symbol_backfill_keys(
    ts_sig: pd.Timestamp,
    data_root: str,
    expected_keys: set[str],
    symbols: list[str],
    full_rewrite_symbols: set[str],
) -> list[str]:
    """Return minimal key set needed for this symbol subset when full rewrites are not required."""
    if not symbols:
        return []
    if any(str(s) in full_rewrite_symbols for s in symbols):
        return sorted(expected_keys)

    ts_str = ts_sig.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(data_root, "features", ts_str)
    missing_keys: set[str] = set()

    for sym in symbols:
        safe_sym = str(sym).replace("/", "_")
        fpath = os.path.join(in_dir, f"symbol={safe_sym}.parquet")
        try:
            schema_names = set(pq.ParquetFile(fpath).schema.names)
        except Exception:
            return sorted(expected_keys)
        missing_keys.update(expected_keys - schema_names)
        present_keys = sorted(expected_keys.intersection(schema_names))
        if present_keys:
            try:
                sample_df = pd.read_parquet(fpath, columns=present_keys)
            except Exception:
                return sorted(expected_keys)
            for c in present_keys:
                if c in sample_df.columns and bool(sample_df[c].isna().all()):
                    missing_keys.add(c)

    return sorted(missing_keys)


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
    return apply_hardcoded_universe_exclusions(syms)


def run_label_generation_step_v2(ts_sig, margin_symbols, cfg, store, ex, horizons=None):
    cfg = apply_offline_optimizer_best_params(dict(cfg))
    run_id = pd.Timestamp(ts_sig).strftime("%Y%m%d_%H%M%S")
    tprint("STEP: LABEL GENERATION START")
    _labels_dir = os.path.join(cfg["data_root"], "artifacts", run_id, "labels")
    os.makedirs(_labels_dir, exist_ok=True)
    _stale_aux = [
        os.path.join(_labels_dir, "label_verification.csv"),
        os.path.join(_labels_dir, "label_verification.md"),
    ]
    _report_dir = os.path.join(cfg["reports_root"], run_id)
    _stale_reports = [
        os.path.join(_report_dir, "bucket_report_labels.md"),
        os.path.join(_report_dir, "label_verification.csv"),
        os.path.join(_report_dir, "label_verification.md"),
    ]
    _existing_train_files = {
        os.path.basename(_fp)
        for _fp in glob.glob(os.path.join(_labels_dir, "train_*.parquet"))
    }
    _removed = 0
    for _fp in _stale_aux + _stale_reports:
        if not os.path.exists(_fp):
            continue
        try:
            os.remove(_fp)
            _removed += 1
        except Exception as _e_rm:
            tprint(f"WARNING: failed to clear stale label artifact {_fp}: {_e_rm}")
    if _removed:
        tprint(f"Cleared {_removed} stale label artifacts for run_id={run_id}")
    _label_symbols_env = str(os.environ.get("EPM_LABEL_SYMBOLS", "")).strip()
    _label_symbols_override = [
        s.strip()
        for s in _label_symbols_env.split(",")
        if isinstance(s, str) and s.strip()
    ]
    _labels_use_store_universe = False
    # Labels should be able to run fully offline from local store contents.
    # If margin_symbols is not provided, source it directly from store instead of
    # triggering a network refresh of margin universe.
    if margin_symbols is None:
        margin_symbols = _local_store_symbols(store)
        if margin_symbols:
            tprint(
                f"Labels universe source: local store symbols ({len(margin_symbols)})"
            )
            _labels_use_store_universe = True
        else:
            tprint(
                "Labels universe source: local store symbols empty; falling back to market basket"
            )
            margin_symbols = list(cfg.get("market_basket", []))
    if _labels_use_store_universe:
        from extreme_price_movements.optimization_utils import (
            filter_low_variance_assets,
        )

        syms_all = apply_hardcoded_universe_exclusions(list(set(margin_symbols)))
        tprint(
            f"Labels offline universe build: using local store symbols ({len(syms_all)} symbols)"
        )
        train_syms = filter_low_variance_assets(
            store,
            syms_all,
            lookback_days=30,
            threshold_pct=cfg["variance_filter_pct"],
            ts_sig=ts_sig,
        )
        train_syms = apply_hardcoded_universe_exclusions(list(set(train_syms)))
    else:
        train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    tprint(f"Universe before dedup: {len(train_syms)} symbols")
    train_syms = apply_hardcoded_universe_exclusions(
        _dedup_universe_by_base(train_syms)
    )
    tprint(
        f"Universe after base-asset dedup (USDT>USDC>BUSD>EUR): {len(train_syms)} symbols"
    )
    if _label_symbols_override:
        _override_set = set(_label_symbols_override)
        _filtered_syms = [s for s in train_syms if s in _override_set]
        tprint(
            f"Label symbol override active: requested={len(_label_symbols_override)} matched={len(_filtered_syms)}"
        )
        train_syms = _filtered_syms
        if not train_syms:
            tprint(
                f"ERROR: EPM_LABEL_SYMBOLS matched no symbols in label universe: {_label_symbols_override}"
            )
            return

    # Load Data & Features
    dfs = {}

    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with Timer("Data Load"):
        def load_sym(s):
            df = store.load(s)
            if not df.empty:
                df = df[df.index <= ts_sig].tail(24 * lookback_days)
                # Downcast to float32 immediately to save memory
                for c in df.columns:
                    if pd.api.types.is_float_dtype(df[c]):
                        df[c] = df[c].astype(np.float32)
                return s, df
            return s, None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(load_sym, s): s for s in train_syms}
            for future in as_completed(futures):
                s, df = future.result()
                if df is not None:
                    dfs[s] = df

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
    del dfs
    gc.collect()

    tprint(
        f"Label data load complete: symbols={panel['close'].shape[1]} "
        f"horizons={horizons if horizons is not None else list(CANON_HORIZONS)} "
        f"label_persist_incremental={bool(cfg.get('label_persist_incremental', False))}"
    )
    # Removed random panel capping to preserve temporal integrity
    # SlicePlanner will be used later to determine walk-forward test set

    # Data coverage diagnostics
    _close = panel["close"]
    _ts_min = _close.index.min()
    _ts_max = _close.index.max()
    _n_hours = len(_close)
    _n_days = (_ts_max - _ts_min).total_seconds() / 86400 if _ts_max > _ts_min else 0
    _n_syms = _close.shape[1]
    _non_nan_pct = float(_close.notna().sum().sum()) / max(_close.size, 1) * 100
    tprint(
        f"DATA COVERAGE: {_n_syms} symbols, {_n_hours} hourly bars, "
        f"{_n_days:.0f} days ({_ts_min.date()} to {_ts_max.date()}), "
        f"{_non_nan_pct:.1f}% non-NaN"
    )
    if _n_days < 365:
        tprint(
            f"WARNING: Only {_n_days:.0f} days of data — recommend >= 365 days for robust training"
        )

    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(
        mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"]
    )

    # Keep feature-key selection consistent with the shared cache/feature generation logic.
    label_feature_keys = _labeling_feature_keys(cfg)
    feats = load_features_for_stage_or_all(cfg, ts_sig, cfg["data_root"],
        feature_keys=sorted(label_feature_keys),
        symbols=train_syms,
    )
    if feats is None or len(feats) == 0:
        tprint("ERROR: Features not found. Run feature_generation first.")
        return

    # Optimization: Downcast feature panels to float32 aggressively.
    # Impact: Halves the memory footprint of loaded features before generation starts.
    for k, v in feats.items():
        if isinstance(v, pd.DataFrame):
            for col in v.columns:
                if pd.api.types.is_float_dtype(v[col]) and v[col].dtype != np.float32:
                    v[col] = v[col].astype(np.float32)

    tprint(
        f"Label feature load complete: feature_families={len(feats)} "
        f"symbols={len(train_syms)}"
    )
    feats = _ensure_atr_pct_feature(feats, panel, cfg, symbols=train_syms)

    # Restrict to symbols present in both panel and features
    sample_feat = next(iter(feats.values()))
    feat_syms = set(sample_feat.columns)
    panel_syms = set(panel["close"].columns)
    valid_syms = sorted(feat_syms & panel_syms & set(train_syms))
    tprint(
        f"Symbol intersection: {len(valid_syms)} (feats={len(feat_syms)}, panel={len(panel_syms)}, universe={len(train_syms)})"
    )
    if not valid_syms:
        tprint("ERROR: No overlapping symbols between features and panel.")
        return
    train_syms = valid_syms

    # Restrict panel to valid symbols and hard-align all feature frames to panel.
    panel = {
        k: v.reindex(columns=valid_syms)
        for k, v in panel.items()
        if isinstance(v, pd.DataFrame)
    }
    feats = _align_features_to_panel(feats, panel, valid_syms)

    # 1. Label Datasets
    tprint(
        f"Generating label datasets for {len(train_syms)} symbols "
        f"across horizons={horizons if horizons is not None else list(CANON_HORIZONS)}"
    )
    datasets = generate_label_datasets(
        panel, feats, mkt_gates, cfg, train_syms, ts_sig, None, horizons=horizons
    )

    # Use SlicePlanner to determine walk-forward test set and exclude it from training data
    if bool(cfg.get("label_skip_slice_planner", False)):
        tprint(
            "Label slice-planner filtering disabled for label step; saving generated datasets directly."
        )
    else:
        from extreme_price_movements.periods_symbols_management import (
            EventSchema,
            SlicePlanner,
            SlicePlannerConfig,
        )

        # Build events from panel index rather than datasets to save memory and time
        _close = panel["close"]
        _idx = _close.index
        _syms = _close.columns

        # Free up memory early
        del panel
        del feats
        del mkt_gates
        gc.collect()

        # Create events frame efficiently by repeating the index for each symbol
        n_ts = len(_idx)
        n_syms = len(_syms)

        # Flattened grid
        t0_array = np.repeat(_idx, n_syms)
        sym_array = np.tile(_syms, n_ts)

        events = pd.DataFrame(
            {
                "event_id": np.arange(len(t0_array), dtype=np.int64),
                "symbol": sym_array,
                "t0": t0_array,
                "t1": t0_array,
            }
        )

        # SAVE EVENTS FOR DOWNSTREAM
        _events_path = os.path.join(
            cfg["data_root"], "artifacts", run_id, "baseline_events.parquet"
        )
        os.makedirs(os.path.dirname(_events_path), exist_ok=True)
        events.to_parquet(_events_path)
        tprint(f"Saved baseline events for planning to {_events_path}")

        # Use SlicePlanner to get training vs test split
        planner_cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
        bundle = SlicePlanner(planner_cfg).build(events)

        # Get all training indices (not test/walk-forward)
        train_indices = set()
        for plan in bundle["consumer_plans"].get("regime_search", []):
            if plan.tag in ["fit_inner", "fit_outer", "predict_inner"]:
                train_indices.update(plan.fit_idx)

        # Filter datasets to only include training rows
        if train_indices:
            # Sort array for fast boolean masking
            train_idx_arr = np.sort(np.fromiter(train_indices, dtype=np.int64))
            mask = np.zeros(len(events), dtype=bool)
            mask[train_idx_arr] = True

            for name in list(datasets.keys()):
                if len(datasets[name]) == len(events):
                    original_len = len(datasets[name])

                    # Faster boolean masking over copying via .iloc
                    # First subset, then clean up old df reference
                    new_df = datasets[name][mask].copy()
                    del datasets[name]
                    datasets[name] = new_df

                    tprint(
                        f"Filtered {name} to {len(datasets[name])} training rows (excluded {original_len - len(datasets[name])} test rows)"
                    )
        else:
            tprint(
                "WARNING: No training indices found from SlicePlanner, using all data"
            )

    dataset_manifest: dict[str, dict[str, object]] = {}
    for name in list(datasets.keys()):
        df = datasets[name]
        tprint(f"Saving label dataset {name}: rows={len(df)} cols={len(df.columns)}")
        save_artifact_df(df, cfg["data_root"], run_id, "labels", name)
        dataset_manifest[name] = {
            "file": f"{name}.parquet",
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
        del datasets[name]
        if bool(cfg.get("label_gc_after_each_dataset", True)):
            gc.collect()
        tprint(f"Saved label dataset {name} and released in-memory frame")

    _manifest_payload = {
        "run_id": run_id,
        "datasets": dataset_manifest,
    }
    try:
        _manifest_path = os.path.join(
            cfg["data_root"], "artifacts", run_id, "labels", "labels_manifest.json"
        )
        if (
            bool(cfg.get("label_persist_incremental", False))
            and not dataset_manifest
            and os.path.exists(_manifest_path)
        ):
            with open(_manifest_path, "r", encoding="utf-8") as _mf:
                _manifest_payload = json.load(_mf)
            tprint(f"Preserving existing incremental labels manifest at {_manifest_path}")
        else:
            with open(_manifest_path, "w", encoding="utf-8") as _mf:
                json.dump(_manifest_payload, _mf, indent=2, sort_keys=True)
            tprint(
                f"Wrote labels manifest with {len(_manifest_payload.get('datasets', {}))} entries to {_manifest_path}"
            )
    except Exception as _me:
        tprint(f"WARNING: labels_manifest write failed: {_me}")

    try:
        _active_manifest_names = set((_manifest_payload.get("datasets") or {}).keys())
        _obsolete_train_files = sorted(
            _fname
            for _fname in _existing_train_files
            if _fname.endswith(".parquet")
            and _fname[: -len(".parquet")] not in _active_manifest_names
        )
        for _fname in _obsolete_train_files:
            _fp = os.path.join(_labels_dir, _fname)
            if not os.path.exists(_fp):
                continue
            os.remove(_fp)
        if _obsolete_train_files:
            tprint(
                f"Removed {len(_obsolete_train_files)} obsolete label parquet files after successful regeneration"
            )
    except Exception as _cleanup_exc:
        tprint(f"WARNING: obsolete label artifact cleanup failed: {_cleanup_exc}")

    try:
        rp = report_labels(
            run_id, cfg["data_root"], cfg, base_dir=cfg.get("reports_root")
        )
        tprint(f"Label bucket report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: label bucket report failed: {_re}")
    tprint("STEP: LABEL GENERATION COMPLETE")


def _oof_consolidated_path(data_root: str, run_id: str, layer: str) -> str:
    oof_dir = os.path.join(data_root, "artifacts", run_id, "oof")
    os.makedirs(oof_dir, exist_ok=True)
    return os.path.join(oof_dir, f"{layer}_oof_all.parquet")


def _invalidate_downstream_oof_layers(
    data_root: str, run_id: str, changed_layer: str
) -> None:
    order = ["base", "meta", "sizer"]
    if changed_layer not in order:
        return
    for layer in order[order.index(changed_layer) + 1 :]:
        fp = _oof_consolidated_path(data_root, run_id, layer)
        if os.path.exists(fp):
            try:
                os.remove(fp)
                tprint(f"OOF invalidation: removed {fp}")
            except Exception as exc:
                tprint(f"WARNING: failed to remove downstream OOF {fp}: {exc}")


def _build_universe_index_from_datasets(
    datasets: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    parts = []
    for df in datasets.values():
        if (
            isinstance(df, pd.DataFrame)
            and "__ts__" in df.columns
            and "__symbol__" in df.columns
        ):
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            df["__ts__"], utc=True, errors="coerce"
                        ),
                        "symbol": df["__symbol__"].astype(str),
                    }
                )
            )
    if not parts:
        return pd.DataFrame(columns=["timestamp", "symbol"])
    uni = pd.concat(parts, ignore_index=True)
    uni = uni.dropna(subset=["timestamp", "symbol"]).drop_duplicates(
        ["timestamp", "symbol"]
    )
    return uni.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _consolidate_layer_oof_from_disk(
    data_root: str, run_id: str, layer: str, universe: pd.DataFrame
) -> int:
    oof_dir = os.path.join(data_root, "artifacts", run_id, "oof")
    if not os.path.isdir(oof_dir):
        return 0
    if layer == "base":
        contract_primary_only = False
        contract_path = os.path.join(
            data_root, "artifacts", run_id, "base_meta_contract.json"
        )
        files = []
        if os.path.exists(contract_path):
            try:
                with open(contract_path, "r", encoding="utf-8") as f:
                    contract = json.load(f)
                contract_primary_only = bool(contract.get("primary_only", False))
                if contract_primary_only:
                    files = sorted(
                        {
                            str(path)
                            for strat in contract.get("strategies", [])
                            for path in strat.get("oof_files", [])
                            if isinstance(path, str) and path.endswith(".parquet")
                        }
                    )
            except Exception as exc:
                tprint(
                    f"WARNING: failed to read base_meta_contract.json for OOF consolidation: {exc}"
                )
        if contract_primary_only and not files:
            return 0
        if not files:
            pats = [
                "oof_*.parquet",
            ]
            files = []
            for pat in pats:
                files.extend(glob.glob(os.path.join(oof_dir, pat)))
            files = sorted(
                {
                    fp
                    for fp in files
                    if os.path.basename(fp) != "base_oof_all.parquet"
                    and not fp.endswith("_tight.parquet")
                    and not fp.endswith("_wide.parquet")
                }
            )
    elif layer == "meta":
        pats = ["meta_oof_*.parquet", "oof_meta_*.parquet"]
        files = []
        for pat in pats:
            files.extend(glob.glob(os.path.join(oof_dir, pat)))
        files = sorted(set(files))
    else:
        pats = ["*.parquet"]
        files = []
        for pat in pats:
            files.extend(glob.glob(os.path.join(oof_dir, pat)))
        files = sorted(set(files))
    if not files:
        return 0

    merged = (
        universe.copy()
        if not universe.empty
        else pd.DataFrame(columns=["timestamp", "symbol"])
    )

    def _sigma_column_name(model_key: str, suffix: str) -> str:
        if model_key.endswith("_wide"):
            return model_key.replace("_wide", f"_{suffix}_wide")
        if model_key.endswith("_tight"):
            return model_key.replace("_tight", f"_{suffix}_tight")
        return f"{model_key}_{suffix}"

    for fp in files:
        try:
            df = pd.read_parquet(fp)
            if "timestamp" not in df.columns or "symbol" not in df.columns:
                continue
            pred_col = (
                "oof_pred"
                if "oof_pred" in df.columns
                else ("oof_prob" if "oof_prob" in df.columns else None)
            )
            if pred_col is None:
                cand = [
                    c
                    for c in df.columns
                    if c
                    not in {"timestamp", "symbol", "index", "y_bin", "y_ret", "oof_raw"}
                ]
                pred_col = cand[0] if cand else None
            if pred_col is None:
                continue
            model_key = os.path.splitext(os.path.basename(fp))[0]
            mini = (
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            df["timestamp"], utc=True, errors="coerce"
                        ),
                        "symbol": df["symbol"].astype(str),
                        model_key: pd.to_numeric(df[pred_col], errors="coerce").astype(
                            np.float32
                        ),
                    }
                )
                .dropna(subset=["timestamp", "symbol"])
                .drop_duplicates(["timestamp", "symbol"])
            )
            for sigma_col, suffix in (
                ("oof_sigma_trees", "sigma"),
                ("oof_sigma_robust", "robust_sigma"),
            ):
                if sigma_col in df.columns:
                    mini[_sigma_column_name(model_key, suffix)] = pd.to_numeric(
                        df[sigma_col], errors="coerce"
                    ).astype(np.float32).values[: len(mini)]
            merged = (
                mini
                if merged.empty
                else merged.merge(mini, on=["timestamp", "symbol"], how="outer")
            )
        except Exception:
            continue

    if merged.empty:
        return 0

    if layer != "base":
        pair_roots = sorted(
            {
                c[:-5]
                for c in merged.columns
                if c.endswith("_wide")
                and not c.endswith(("_sigma_wide", "_robust_sigma_wide"))
                and f"{c[:-5]}_tight" in merged.columns
            }
        )
        for root in pair_roots:
            wide_col = f"{root}_wide"
            tight_col = f"{root}_tight"
            sigma_wide_col = f"{root}_sigma_wide"
            sigma_tight_col = f"{root}_sigma_tight"
            robust_sigma_wide_col = f"{root}_robust_sigma_wide"
            robust_sigma_tight_col = f"{root}_robust_sigma_tight"
            pair_features = build_wide_tight_pair_features(
                merged[wide_col].values,
                merged[tight_col].values,
                base_name=root,
                sigma_wide=merged[sigma_wide_col].values if sigma_wide_col in merged.columns else None,
                sigma_tight=merged[sigma_tight_col].values if sigma_tight_col in merged.columns else None,
                robust_sigma_wide=merged[robust_sigma_wide_col].values if robust_sigma_wide_col in merged.columns else None,
                robust_sigma_tight=merged[robust_sigma_tight_col].values if robust_sigma_tight_col in merged.columns else None,
            )
            pair_block = pd.DataFrame(pair_features, index=merged.index)
            merged = pd.concat([merged, pair_block], axis=1, copy=False)

    for col in merged.columns:
        if col in {"symbol"}:
            merged[col] = merged[col].astype("category")
        elif col not in {"timestamp"}:
            merged[col] = pd.to_numeric(merged[col], errors="coerce", downcast="float")

    out = _oof_consolidated_path(data_root, run_id, layer)
    merged.to_parquet(out, index=False, compression="zstd")
    tprint(
        f"Saved consolidated {layer} OOF: {out} rows={len(merged)} cols={len(merged.columns)}"
    )
    return len(merged)


def run_training_step(
    ts_sig, cfg, store=None, margin_symbols=None, base_only=False, meta_only=False
):
    """Train all models from label artifacts. Saves trained state to disk."""
    cfg = apply_offline_optimizer_best_params(dict(cfg))
    cfg = ensure_training_residualization_feature_keys(cfg)
    planner_preset = str(cfg.get("slice_planner_preset", "fast")).lower()
    cfg["slice_planner_preset"] = "robust" if planner_preset == "robust" else "fast"
    cfg["train_full_inference_models"] = bool(
        cfg.get("train_full_inference_models", cfg["slice_planner_preset"] == "robust")
    )
    tprint(
        f"STEP: MODEL TRAINING START (base_only={base_only}, planner_preset={cfg['slice_planner_preset']}, "
        f"full_inference_retrain={cfg['train_full_inference_models']})"
    )

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    datasets = {}

    # 1. Load label artifacts
    tprint("Loading label datasets from artifacts...")

    from extreme_price_movements.data_store import load_artifact_manifest

    labels_manifest = load_artifact_manifest(cfg["data_root"], run_id, "labels") or {}
    available_label_names = set((labels_manifest.get("datasets") or {}).keys())

    # Alpha models (Dynamic Strategies from get_strategies)
    strategies = get_strategies(cfg)
    strategy_horizons = {
        s["strategy_id"]: strategy_runtime_horizons(s, cfg) for s in strategies
    }

    found_count = 0
    dataset_cache: dict[str, pd.DataFrame] = {}

    def _load_label_dataset(name: str) -> pd.DataFrame | None:
        cached = dataset_cache.get(name)
        if cached is not None:
            return cached
        if available_label_names and name not in available_label_names:
            return None
        df_local = load_artifact_df(cfg["data_root"], run_id, "labels", name)
        df_local = _filter_artifact_by_stage_view(df_local, cfg)
        if isinstance(df_local, pd.DataFrame) and not df_local.empty:
            sort_cols = [
                c
                for c in ("__ts__", "timestamp", "__symbol__", "symbol")
                if c in df_local.columns
            ]
            if sort_cols:
                df_local = df_local.sort_values(
                    sort_cols, kind="mergesort"
                ).reset_index(drop=True)
            else:
                df_local = df_local.reset_index(drop=True)
        if df_local is not None:
            dataset_cache[name] = df_local
        return df_local

    include_variant_datasets = bool(cfg.get("base_geometry_train_variants", False))
    base_geometry_archetypes = (
        [
            str(v)
            for v in cfg.get("base_geometry_archetypes", ["tight", "balanced", "wide"])
        ]
        if include_variant_datasets
        else []
    )
    for strat in strategies:
        side = strat["trade_side"]
        s_id = strat["strategy_id"]
        # Determine kind (mr/tf) for backward compatibility with naming if needed
        # but the actual name will be 'train_{s_id}_{H}'
        for H in strategy_horizons.get(s_id, []):
            name = f"train_{s_id}_{H}"
            df = _load_label_dataset(name)
            if df is not None:
                datasets[name] = df
                found_count += 1
                tprint(f"  Loaded {name}: {len(datasets[name])} rows")

            for variant in base_geometry_archetypes:
                if variant == "balanced":
                    continue
                vname = f"train_{s_id}_{H}_{variant}"
                df_v = _load_label_dataset(vname)
                if df_v is not None:
                    datasets[vname] = df_v
                    tprint(f"  Loaded {vname}: {len(datasets[vname])} rows")

    if not include_variant_datasets:
        tprint("Model-training dataset loader: primary-only mode, variant label artifacts are excluded.")

    tprint("Spike anatomy and specialist label datasets are disabled.")

    if not found_count:
        tprint("ERROR: No alpha label datasets found. Run 'labels' mode first.")
        return None

    _train_symbols_env = str(os.environ.get("EPM_TRAIN_SYMBOLS", "")).strip()
    _train_symbols_filter = [
        s.strip() for s in _train_symbols_env.split(",") if s.strip()
    ]
    if _train_symbols_filter:
        _filter_set = set(_train_symbols_filter)
        _filtered_datasets = {}
        for name, df in datasets.items():
            if "__symbol__" in df.columns:
                df_filtered = df[df["__symbol__"].isin(_filter_set)].reset_index(
                    drop=True
                )
                if len(df_filtered) > 0:
                    _filtered_datasets[name] = df_filtered
                    tprint(
                        f"  EPM_TRAIN_SYMBOLS filter: {name} {len(df)} -> {len(df_filtered)} rows"
                    )
                else:
                    tprint(
                        f"  EPM_TRAIN_SYMBOLS filter: {name} dropped (no matching symbols)"
                    )
            else:
                _filtered_datasets[name] = df
        datasets = _filtered_datasets
        tprint(
            f"EPM_TRAIN_SYMBOLS={_train_symbols_env}: {len(datasets)} datasets retained"
        )

    _oos_time_filter = cfg.get("oos_eval_time_filter")
    if _oos_time_filter is not None:
        _t_start, _t_end = _oos_time_filter
        _tf_datasets = {}
        for name, df in datasets.items():
            if "__ts__" in df.columns:
                import pandas as _pd

                _ts = _pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
                _mask = np.ones(len(df), dtype=bool)
                if _t_start is not None:
                    _mask &= _ts >= _t_start
                if _t_end is not None:
                    _mask &= _ts < _t_end
                df = df.loc[_mask].reset_index(drop=True)
            if len(df) > 0:
                _tf_datasets[name] = df
            else:
                tprint(f"  Time filter dropped: {name}")
        datasets = _tf_datasets
        tprint(
            f"oos_eval_time_filter=[{_t_start}, {_t_end}): {len(datasets)} datasets retained"
        )

    tprint(f"Loaded {len(datasets)} datasets total.")

    from extreme_price_movements.training import train_models_from_artifacts

    req_keys = list(_expected_feature_keys_from_cfg(cfg))
    if base_only:
        meta_keys = set(_meta_feature_keys_union(cfg))
        base_keys = set(_base_feature_keys_union(cfg))
        # Remove meta-only keys from required keys
        req_keys = list(set(req_keys) - (meta_keys - base_keys))
        tprint(
            f"Base-only mode: Loading {len(req_keys)} req features (excluded {len(meta_keys - base_keys)} meta-only keys)"
        )

    datasets = inject_features_into_datasets(datasets, ts_sig, cfg, req_keys)

    # 2. Train models
    # Inject run_id so meta OOF files are saved to the correct artifacts directory
    cfg["run_id"] = run_id
    # Selector hysteresis can warm-start from a previous run when available.
    if not cfg.get("prev_run_id"):
        try:
            _art_dir = os.path.join(cfg["data_root"], "artifacts")
            _prev_runs = sorted(
                [
                    d
                    for d in os.listdir(_art_dir)
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
        trained_bundle = train_models_from_artifacts(
            datasets, cfg, train_meta=not base_only, train_base=not meta_only
        )
        alpha_metrics = (
            trained_bundle.get("alpha_oof_metrics", {}) if trained_bundle else {}
        )

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
    # Build granular risk params dynamically from strategies
    strategies = get_strategies(cfg)

    _granular = {}
    for strat in strategies:
        strat_id = strat["strategy_id"]
        is_mr = strat.get("is_mr", "mr" in strat_id.lower())
        risk_config = _mr_risk if is_mr else _tf_risk
        _granular[f"risk_{strat_id}"] = risk_config

    # Add legacy fallback keys for backward compatibility
    _granular.update(
        {
            "risk_mr_best": _mr_risk,
            "risk_mr_worst": _mr_risk,
            "risk_tf_best": _tf_risk,
            "risk_tf_worst": _tf_risk,
            "risk_long_mr": _mr_risk,
            "risk_short_mr": _mr_risk,
            "risk_long_tf": _tf_risk,
            "risk_short_tf": _tf_risk,
        }
    )
    default_risk = {
        "k_sl": cfg.get("risk_k_sl", 2.0),
        "k_trail_start": cfg.get("risk_k_trail_start", 1.0),
        "k_trail_dist": cfg.get("risk_k_trail_dist", 0.5),
        "granular_risk": _granular,
    }

    state = {
        "ts_trained": ts_sig,
        "bundle": trained_bundle,
        "risk_params": default_risk,
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
        alpha_keys = []
        for side, side_models in alpha.items():
            for k, m in side_models.items():
                alpha_keys.append((side, k, m))
        for side, k, m in alpha_keys:
            if m:
                tprint(f"  {side} {k}: H={m['H']}, features={len(m['feat_cols'])}")
            else:
                tprint(f"  {side} {k}: NO MODEL")
        if not alpha_keys:
            tprint("  alpha_models: NONE")

        spike = bundle.get("spike_models", {})
        for mode in ["best", "worst"]:
            m = spike.get(mode)
            tprint(f"  spike_{mode}: {'fitted' if m else 'NO MODEL'}")
            if m and "oof_scores" in m:
                oof_df = m["oof_scores"]
                save_artifact_df(
                    oof_df, cfg["data_root"], run_id, "labels", f"spike_oof_{mode}"
                )
                tprint(f"  Saved OOF scores: spike_oof_{mode}")

        exh = bundle.get("exh_models", {})
        for d in ["up", "down"]:
            m = exh.get(d)
            tprint(f"  exh_{d}: {'fitted' if m and m.model else 'NO MODEL'}")

        meta = bundle.get("meta_models", {})
        if meta:
            for key, m in meta.items():
                tprint(f"  meta_{key}: {'fitted' if m and m.model else 'NO MODEL'}")
        else:
            tprint("  meta_models: NONE")

    # Generate training report
    try:
        report_path = generate_training_report(
            run_id=run_id,
            cfg=cfg,
            bundle=bundle or {},
            datasets=datasets or {},
            specialist_models=bundle.get("specialist_models") if bundle else None,
            extra_info=alpha_metrics,
            base_dir=cfg.get("reports_root"),
        )
        tprint(f"Training report saved to {report_path}")
    except Exception as e:
        tprint(f"WARNING: Failed to generate training report: {e}")

    # Per-bucket/horizon detailed reports
    try:
        rp = report_base_training(
            run_id, bundle or {}, cfg, base_dir=cfg.get("reports_root")
        )
        tprint(f"Base training bucket report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: base training bucket report failed: {_re}")
    try:
        rp = report_meta_training(
            run_id,
            cfg["data_root"],
            bundle or {},
            cfg,
            base_dir=cfg.get("reports_root"),
        )
        tprint(f"Meta training bucket report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: meta training bucket report failed: {_re}")

    # Consolidate OOF predictions (whole universe/period index; sparse model coverage allowed).
    try:
        _uni = _build_universe_index_from_datasets(datasets)
        _n_base = _consolidate_layer_oof_from_disk(
            cfg["data_root"], run_id, "base", _uni
        )
        _n_meta = _consolidate_layer_oof_from_disk(
            cfg["data_root"], run_id, "meta", _uni
        )
        if _n_base > 0:
            _invalidate_downstream_oof_layers(cfg["data_root"], run_id, "base")
        if _n_meta > 0:
            _invalidate_downstream_oof_layers(cfg["data_root"], run_id, "meta")
    except Exception as _e_oof:
        tprint(f"WARNING: OOF consolidation failed: {_e_oof}")

    tprint("STEP: MODEL TRAINING COMPLETE")
    return state


def run_ridge_sizer_step(ts_sig, cfg, state_file):
    """Run ridge position sizer to learn optimal meta model combination weights.

    Processes each bucket (loaded dynamically from final_rule_registry.csv via
    load_inference_candidate_mask_params_per_bucket) separately, combining
    per-horizon regressors (H1, H2, H4) + classifier + agreement features into
    a single Ridge combiner per bucket.

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
    # IV. Dynamically load latest tpsl_optimiser params for exit policy alignment
    import json as _json

    from extreme_price_movements.run_ridge_sizer import (
        load_meta_oof_predictions as load_strategy_oofs,
    )
    from extreme_price_movements.run_ridge_sizer import load_trade_outcomes

    _tpsl_params = {}
    _bp_path = os.path.join(
        data_root, "artifacts", run_id, "ridge_sizer", "strategy_params.json"
    )
    if os.path.exists(_bp_path):
        try:
            with open(_bp_path) as _f:
                _bp_data = _json.load(_f)
            _tpsl_params = _bp_data.get("buckets", {})
            tprint(
                f"  Loaded tpsl_optimiser params from {_bp_path} ({len(_tpsl_params)} buckets)"
            )
        except Exception as _e:
            tprint(f"  WARNING: Could not load tpsl params: {_e}")
    else:
        if not _tpsl_params:
            tprint(
                "  No strategy params found, Ridge sizer will use default exit policy"
            )

    try:
        _meta_all = _oof_consolidated_path(data_root, run_id, "meta")
        if os.path.exists(_meta_all):
            _mdf = pd.read_parquet(_meta_all)
            _pred_cols = [c for c in _mdf.columns if c not in {"timestamp", "symbol"}]
            if {"timestamp", "symbol"}.issubset(_mdf.columns) and _pred_cols:
                _use = _pred_cols[0]
                strategy_oofs = {
                    "long_mr": _mdf[["timestamp", "symbol", _use]].rename(
                        columns={_use: "oof_u_hat"}
                    )
                }
                tprint(
                    f"Ridge sizer using consolidated meta OOF dataframe: {_meta_all}"
                )
            else:
                strategy_oofs = load_strategy_oofs(data_root, run_id)
        else:
            strategy_oofs = load_strategy_oofs(data_root, run_id)
    except FileNotFoundError as e:
        tprint(f"WARNING: {e}")
        tprint("Skipping ridge sizer step - no meta OOF predictions found.")
        return None

    cost_pct = float(cfg.get("ridge_cost_pct", cfg.get("fee_bps", 50.0) / 10000.0))
    output_dir = os.path.join(data_root, "artifacts", run_id, "ridge_sizer")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Group buckets by direction using strategy definitions from final_rule_registry.csv
    # Strategies loaded via load_inference_candidate_mask_params_per_bucket() provide
    # strategy_id and trade_side for proper direction grouping.
    # -------------------------------------------------------------------------
    # Load strategies to get trade_side for each bucket
    from extreme_price_movements.offline_optimisers.params_store import (
        load_inference_candidate_mask_params_per_bucket,
    )

    strategies = load_inference_candidate_mask_params_per_bucket(
        top_n=2, ranking_metric="score_for_best_params"
    )
    strategy_side_map = {s["strategy_id"]: s["trade_side"] for s in strategies}

    _meta_cols = {
        "timestamp",
        "symbol",
        "return",
        "is_long",
        "index",
        "oof_u_hat",
        "oof_log_mae_q70_hat",
        "oof_log_mfe_hat",
        "mae_ret",
        "mfe_ret",
        "u_policy_net",
        "exit_code",
    }
    direction_groups = {"long": {}, "short": {}}
    for bucket_name, oof_preds in strategy_oofs.items():
        # Look up trade_side from strategy map, fallback to legacy naming convention
        direction = strategy_side_map.get(bucket_name)
        if direction is None:
            # Legacy fallback: infer from bucket name
            direction = "long" if bucket_name.startswith("long") else "short"
        direction_groups[direction][bucket_name] = oof_preds

    all_direction_results = {}

    for direction, dir_buckets in direction_groups.items():
        if not dir_buckets:
            continue
        tprint(
            f"  Ridge sizer — direction: {direction.upper()} ({list(dir_buckets.keys())})"
        )

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
            timestamps = (
                trade_outcomes["timestamp"].values
                if "timestamp" in trade_outcomes.columns
                else None
            )
            symbols = (
                trade_outcomes["symbol"].values
                if "symbol" in trade_outcomes.columns
                else None
            )
            _ev_cols = {"oof_u_hat", "oof_log_mae_q70_hat", "oof_log_mfe_hat"}
            _ev_present = [c for c in pred_cols if c in _ev_cols]
            _other_meta = [c for c in pred_cols if c not in _ev_cols]
            tprint(
                f"    {bucket_name}: {len(oof_pred_df)} rows, features={len(pred_cols)} "
                f"(ev_heads={len(_ev_present)} other_meta={len(_other_meta)})"
            )
            if _other_meta:
                tprint(f"      other meta features used by ridge: {_other_meta}")

            # Policy-aligned trade mask for ridge sizer (entry policy can reduce trade count).
            _bp_key = bucket_name.upper()
            _bp_cfg = (
                flatten_bucket_policy(_tpsl_params.get(_bp_key, {}))
                if isinstance(_tpsl_params, dict)
                else {}
            )
            if _bp_cfg.get("entry_policy"):
                _scores = np.asarray(oof_pred_df[pred_cols[0]].values, dtype=float)
                _atr_vec = np.asarray(
                    trade_outcomes.get(
                        "mae_ret", pd.Series(np.full(len(trade_outcomes), 0.02))
                    ).values,
                    dtype=float,
                )
                _atr_vec = np.clip(
                    np.where(np.isfinite(_atr_vec), np.abs(_atr_vec), 0.02), 1e-4, 0.5
                )
                _mask = np.ones(len(trade_outcomes), dtype=bool)
                for _i in range(len(_mask)):
                    _pol = compute_entry_policy_decision(
                        entry_px=1.0,
                        atr_frac=float(_atr_vec[_i]),
                        score=float(_scores[_i]) if _i < len(_scores) else 0.0,
                        bucket_cfg=_bp_cfg,
                        features={
                            "u_hat_z": float(
                                trade_outcomes.get(
                                    "oof_u_hat",
                                    pd.Series(np.zeros(len(trade_outcomes))),
                                ).iloc[_i]
                            )
                            if "oof_u_hat" in trade_outcomes.columns
                            else float(
                                np.tanh(_scores[_i] if _i < len(_scores) else 0.0)
                            ),
                            "mae_hat_z": float(
                                trade_outcomes.get(
                                    "oof_log_mae_q70_hat",
                                    pd.Series(np.zeros(len(trade_outcomes))),
                                ).iloc[_i]
                            )
                            if "oof_log_mae_q70_hat" in trade_outcomes.columns
                            else float(
                                abs(np.tanh(_scores[_i] if _i < len(_scores) else 0.0))
                            ),
                            "mfe_hat_z": float(
                                trade_outcomes.get(
                                    "oof_log_mfe_hat",
                                    pd.Series(np.zeros(len(trade_outcomes))),
                                ).iloc[_i]
                            )
                            if "oof_log_mfe_hat" in trade_outcomes.columns
                            else 0.0,
                            "dur_hat_z": float(
                                trade_outcomes.get(
                                    "oof_log_dur_hat",
                                    pd.Series(np.zeros(len(trade_outcomes))),
                                ).iloc[_i]
                            )
                            if "oof_log_dur_hat" in trade_outcomes.columns
                            else 0.0,
                        },
                    )
                    _mask[_i] = bool(_pol.get("place_order", True))
                trade_outcomes = trade_outcomes.loc[_mask].reset_index(drop=True)
                oof_pred_df = oof_pred_df.loc[_mask].reset_index(drop=True)
                if timestamps is not None:
                    timestamps = np.asarray(timestamps)[_mask]
                if symbols is not None:
                    symbols = np.asarray(symbols)[_mask]
                tprint(
                    f"    {bucket_name}: policy mask kept {_mask.sum()}/{len(_mask)} rows for ridge training"
                )

            try:
                sizer, metrics = run_ridge_position_sizer_step(
                    oof_preds=oof_pred_df,
                    trade_outcomes=trade_outcomes,
                    timestamps=timestamps,
                    cfg={"cost_pct": cost_pct},
                    save_model=False,
                    run_id=run_id,
                    symbols=symbols,
                    bucket_name=bucket_name,
                )
                bkt_weights = sizer.get_weights()
                for wname, wval in bkt_weights.items():
                    dir_weights[f"{bucket_name}_{wname}"] = wval
                dir_params[bucket_name] = sizer.best_params_
                _metrics_ext = dict(metrics or {})
                _metrics_ext["feature_columns"] = list(pred_cols)
                _metrics_ext["ev_head_feature_count"] = int(len(_ev_present))
                _metrics_ext["other_meta_feature_count"] = int(len(_other_meta))
                dir_metrics[bucket_name] = _metrics_ext
                tprint(f"    {bucket_name} weights: {bkt_weights}")
            except Exception as e:
                tprint(f"    {bucket_name} failed: {e}")
                continue

        if not dir_weights:
            tprint(f"    No weights produced for direction {direction}, skipping")
            continue

        # Save per-direction weight file
        import json as _json
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        dir_weights_path = os.path.join(output_dir, f"sizer_weights_{direction}.json")
        with open(dir_weights_path, "w") as f:
            _json.dump(
                {
                    "direction": direction,
                    "weights": dir_weights,
                    "params_per_strategy": dir_params,
                    "buckets": list(dir_buckets.keys()),
                    "run_id": run_id,
                    "timestamp": _dt.now(_tz.utc).isoformat(),
                },
                f,
                indent=2,
            )
        tprint(f"    Saved {direction} sizer weights to {dir_weights_path}")

        all_direction_results[direction] = {
            "weights": dir_weights,
            "params": dir_params,
            "metrics": dir_metrics,
            "buckets": list(dir_buckets.keys()),
        }

    # Flatten for backward-compatible combined manifest + state
    all_weights = {}
    all_params = {}
    all_metrics = {}
    for direction, res in all_direction_results.items():
        all_weights.update(res["weights"])
        for bkt in res["buckets"]:
            all_params[bkt] = res["params"].get(bkt, {})
        all_metrics[direction] = res["metrics"]

    import json
    from datetime import datetime, timezone

    weights_path = os.path.join(output_dir, "sizer_weights.json")
    with open(weights_path, "w") as f:
        json.dump(
            {
                "weights": all_weights,
                "params_per_strategy": all_params,
                "directions": {
                    d: {"buckets": r["buckets"], "params": r["params"]}
                    for d, r in all_direction_results.items()
                },
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    tprint(f"Saved combined manifest to {weights_path}")

    # Update state file
    if os.path.exists(state_file):
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        state["ridge_sizer"] = {
            "weights": all_weights,
            "params_per_strategy": all_params,
            "metrics": all_metrics,
            "directions": {d: r["buckets"] for d, r in all_direction_results.items()},
        }
        with open(state_file, "wb") as f:
            pickle.dump(state, f)
        tprint("Updated state file with ridge sizer weights")

    # Persist consolidated ridge input/OOF dataframe for downstream layer chaining.
    try:
        _frames = []
        for _bn, _df in strategy_oofs.items():
            if (
                _df is None
                or len(_df) == 0
                or "timestamp" not in _df.columns
                or "symbol" not in _df.columns
            ):
                continue
            _pc = [
                c
                for c in _df.columns
                if c not in {"timestamp", "symbol", "index", "return", "is_long"}
            ]
            if not _pc:
                continue
            _frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            _df["timestamp"], utc=True, errors="coerce"
                        ),
                        "symbol": _df["symbol"].astype(str),
                        os.path.basename(_bn): pd.to_numeric(
                            _df[_pc[0]], errors="coerce"
                        ).astype(np.float32),
                    }
                )
            )
        if _frames:
            _merged = _frames[0]
            for _f in _frames[1:]:
                _merged = _merged.merge(_f, on=["timestamp", "symbol"], how="outer")
            for _c in _merged.columns:
                if _c == "symbol":
                    _merged[_c] = _merged[_c].astype("category")
                elif _c not in {"timestamp"}:
                    _merged[_c] = pd.to_numeric(
                        _merged[_c], errors="coerce", downcast="float"
                    )
            _sp = _oof_consolidated_path(data_root, run_id, "sizer")
            _merged.to_parquet(_sp, index=False, compression="zstd")
            tprint(f"Saved consolidated sizer OOF: {_sp} rows={len(_merged)}")
    except Exception as _e_s_oof:
        tprint(f"WARNING: failed to save consolidated sizer OOF: {_e_s_oof}")

    try:
        _ridge_score_frames = []
        for _rs_dir_key, _rs_params in all_params.items():
            _rs_metrics = all_metrics.get(_rs_dir_key, {})
            _rs_oof_path = (
                Path(data_root)
                / "artifacts"
                / run_id
                / "ridge_sizer"
                / f"ridge_sizer_oof_{_rs_dir_key.lower()}.parquet"
            )
            if not _rs_oof_path.exists():
                continue
            _rs_df = pd.read_parquet(_rs_oof_path)
            if (
                "sizer_score_oof" not in _rs_df.columns
                or "timestamp" not in _rs_df.columns
                or "symbol" not in _rs_df.columns
            ):
                continue
            _rs_sub = _rs_df[["timestamp", "symbol", "sizer_score_oof"]].copy()
            _rs_sub = _rs_sub.rename(columns={"sizer_score_oof": _rs_dir_key})
            _rs_sub["timestamp"] = pd.to_datetime(
                _rs_sub["timestamp"], utc=True, errors="coerce"
            )
            _rs_sub["symbol"] = _rs_sub["symbol"].astype(str)
            _ridge_score_frames.append(_rs_sub)
        if _ridge_score_frames:
            _rs_merged = _ridge_score_frames[0]
            for _rsf in _ridge_score_frames[1:]:
                _rs_merged = _rs_merged.merge(
                    _rsf, on=["timestamp", "symbol"], how="outer"
                )
            _rs_out_path = (
                Path(data_root)
                / "artifacts"
                / run_id
                / "oof"
                / "ridge_sizer_scores_all.parquet"
            )
            _rs_merged.to_parquet(_rs_out_path, index=False, compression="zstd")
            tprint(
                f"Saved ridge sizer OOF scores: {_rs_out_path} rows={len(_rs_merged)}"
            )
    except Exception as _e_rs_scores:
        tprint(f"WARNING: failed to save ridge sizer OOF scores: {_e_rs_scores}")

    _ridge_result = {
        "weights": all_weights,
        "params_per_strategy": all_params,
        "metrics": all_metrics,
        "directions": {d: r["buckets"] for d, r in all_direction_results.items()},
    }
    try:
        rp = report_ridge_sizer(run_id, _ridge_result, base_dir=cfg.get("reports_root"))
        tprint(f"Ridge sizer strategy report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: ridge sizer strategy report failed: {_re}")
    tprint(
        f"STEP: RIDGE POSITION SIZER COMPLETE — {len(all_direction_results)} directions, {len(all_params)} strategies"
    )
    return {
        "weights": all_weights,
        "params_per_strategy": all_params,
        "metrics": all_metrics,
        "directions": {d: r["buckets"] for d, r in all_direction_results.items()},
    }


def run_sizer_step(ts_sig, cfg, state_file):
    """Run offline ridge sizing/offset optimization using meta OOF outputs."""
    planner_preset = str(cfg.get("slice_planner_preset", "fast")).lower()
    full_inference_retrain = bool(
        cfg.get("train_full_inference_models", planner_preset == "robust")
    )
    tprint(
        f"STEP: SIZER START (ridge, planner_preset={planner_preset}, full_inference_retrain={full_inference_retrain})"
    )
    return run_ridge_sizer_step(ts_sig, cfg, state_file)


def run_policy_optimiser_step(ts_sig, cfg):
    """Run sequential policy optimisation using fixed TP/SL and pre-generated offsets."""
    from extreme_price_movements.policy_optimiser import run_policy_optimisation

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    data_root = cfg.get("data_root", "data")
    cost_pct = float(cfg.get("ridge_cost_pct", cfg.get("fee_bps", 50.0) / 10000.0))
    holdout_frac = float(cfg.get("policy_optimiser_holdout_frac", 0.30))
    use_offset_optimiser = bool(cfg.get("run_limit_offset_optimiser", False))
    tprint(
        f"STEP: POLICY OPTIMISER START (holdout_frac={holdout_frac}, "
        f"use_offset_optimiser={use_offset_optimiser})"
    )
    result = run_policy_optimisation(
        data_root=data_root,
        run_id=run_id,
        holdout_frac=holdout_frac,
        cost_pct=cost_pct,
        use_offset_optimiser=use_offset_optimiser,
        stage_view=cfg.get("_active_stage_view"),
    )
    if result:
        tprint("POLICY OPTIMISER COMPLETE")
    else:
        tprint("POLICY OPTIMISER: no results produced")
    return result


def run_base_hpo_step(ts_sig, cfg):
    """Run Optuna HPO for the base ExtraTrees classifier.

    Runs one HPO job per primary base training dataset scope (per strategy
    and horizon). Each scope now follows:
    MDI feature selection -> HPO -> final training.
    """
    from extreme_price_movements.post_race_hpo import run_base_extratrees_hpo
    from sklearn.ensemble import ExtraTreesRegressor

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    data_root = cfg.get("data_root", "data")

    from extreme_price_movements.data_store import load_artifact_manifest

    labels_manifest = load_artifact_manifest(data_root, run_id, "labels") or {}
    available = set((labels_manifest.get("datasets") or {}).keys())
    if not available:
        tprint("BASE HPO: no label datasets found. Run labels first.")
        return None

    strategies = get_strategies(cfg)
    strategy_horizons = {
        s["strategy_id"]: strategy_runtime_horizons(s, cfg) for s in strategies
    }

    hpo_out_dir = os.path.join(data_root, "hpo_out")
    os.makedirs(hpo_out_dir, exist_ok=True)
    cfg["base_hpo_out_dir"] = hpo_out_dir

    results = {}
    for strat in strategies:
        s_id = strat["strategy_id"]
        strategy_datasets: dict[str, pd.DataFrame] = {}

        for H in strategy_horizons.get(s_id, []):
            name = f"train_{s_id}_{H}"
            if name not in available:
                continue
            df = load_artifact_df(data_root, run_id, "labels", name)
            df = _filter_artifact_by_stage_view(df, cfg)
            if df is None or df.empty:
                continue
            if "__y_bin__" not in df.columns:
                continue
            strategy_datasets[name] = df

        if not strategy_datasets:
            tprint(f"BASE HPO: no data rows loaded for strategy {s_id}.")
            continue

        base_req_keys = set(str(v) for v in (strat.get("feature_keys") or []) if isinstance(v, str) and v)
        base_req_keys.update(
            {
                "p_exh_lag1",
                "G_VOL",
                "G_TREND",
                "mkt_ret24h",
                "mkt_ret6h",
                "mkt_trend",
                "mkt_rv",
                "trap_score",
                "gamma_score",
            }
        )
        strategy_datasets = inject_features_into_datasets(
            strategy_datasets,
            ts_sig,
            cfg,
            sorted(base_req_keys),
        )

        allowed_keys = set(strat.get("feature_keys") or [])
        std_inputs = {
            "p_exh_lag1",
            "G_VOL",
            "G_TREND",
            "mkt_ret24h",
            "mkt_ret6h",
            "mkt_trend",
            "mkt_rv",
            "trap_score",
            "gamma_score",
        }

        strategy_scope_results: dict[str, dict] = {}
        for name, df in strategy_datasets.items():
            if df is None or df.empty or "__y_bin__" not in df.columns:
                continue
            feature_cols: list[str] = []
            for c in df.columns:
                if not isinstance(c, str):
                    continue
                if c.startswith("__") or c in {"ts", "timestamp", "symbol", "asset"}:
                    continue
                if c in allowed_keys or c in std_inputs:
                    feature_cols.append(c)
                    continue
                for g in ("G_VOL", "G_TREND"):
                    if f"_{g}_0" in c or f"_{g}_1" in c:
                        base = c.split(f"_{g}_")[0]
                        if base in allowed_keys:
                            feature_cols.append(c)
                            break

            if not feature_cols:
                continue

            X_parts = (
                df[feature_cols]
                .select_dtypes(include=[np.number])
                .astype(np.float32, copy=False)
            )
            if X_parts.empty:
                continue
            X_all = X_parts.values.astype(np.float32, copy=False)
            y_all = df["__y_bin__"].values.astype(np.float32)
            sym_col = (
                df["__symbol__"].values
                if "__symbol__" in df.columns
                else np.full(len(df), "unknown", dtype=object)
            )
            valid_y_mask = np.isfinite(y_all)
            if not valid_y_mask.all():
                X_all = X_all[valid_y_mask]
                y_all = y_all[valid_y_mask]
                sym_col = sym_col[valid_y_mask]

            if X_all.size == 0 or y_all.size == 0:
                tprint(f"BASE HPO[{name}]: empty matrix after target filtering.")
                continue

            missing_frac = float(np.mean(~np.isfinite(X_all)))
            if missing_frac > 0.0:
                col_medians = np.nanmedian(X_all, axis=0)
                col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0).astype(
                    np.float32,
                    copy=False,
                )
                nan_rows, nan_cols = np.where(~np.isfinite(X_all))
                X_all = X_all.astype(np.float32, copy=False)
                X_all[nan_rows, nan_cols] = col_medians[nan_cols]
                tprint(
                    f"BASE HPO[{name}]: imputed non-finite feature values "
                    f"(missing_frac={missing_frac:.2%})."
                )

            X_df = pd.DataFrame(X_all, columns=list(X_parts.columns))
            is_variant_scope = name.endswith("_tight") or name.endswith("_wide")
            _base_sel_cfg = dict(cfg.get("base_selector_cfg", {}) or {})
            _selector_report_dir = os.path.join(
                str(cfg.get("data_root", "data")),
                "artifacts",
                str(cfg.get("run_id", run_id)),
                "fs_reports",
            )
            _n_training_rows = int(len(X_df))
            _strict_row_cap = max(1, int((_n_training_rows - 1) // 20))
            _mdi_target = min(200, _strict_row_cap)
            _mdi_min = min(
                int(
                    cfg.get(
                        "base_variant_mdi_min_features" if is_variant_scope else "base_mdi_min_features",
                        30,
                    )
                ),
                _mdi_target,
            )
            _sel_idx = _subsample_indices_time_balanced(
                len(X_df), int(cfg.get("base_selector_max_samples", 30000)), y_all
            )
            mdi_base = ExtraTreesRegressor(
                n_estimators=int(_base_sel_cfg.get("analysis_n_estimators", 192)),
                max_depth=8,
                min_samples_leaf=64,
                max_features="sqrt",
                n_jobs=2,
                random_state=42,
            )
            sel_res = mdi_feature_selection_v3(
                X_df.iloc[_sel_idx],
                y_all[_sel_idx],
                candidate_cols=list(X_df.columns),
                base_model=mdi_base,
                sample_weight=None,
                selector_y=y_all[_sel_idx],
                selector_target=str(cfg.get("base_mdi_selector_target", "classification")),
                selector_loss=str(cfg.get("base_mdi_selector_loss", "binary_logloss")),
                selector_head_name=f"base_{name}",
                selector_report_dir=_selector_report_dir,
                selector_prev_selected=None,
                selector_family_map=dict(cfg.get("selector_feature_family_map", {}) or {}),
                selector_focus_top_frac=float(_base_sel_cfg.get("selector_focus_top_frac", 1.0)),
                selector_top_metric=_base_sel_cfg.get("selector_top_metric", "ic"),
                selector_frequency_hit_mode=str(
                    _base_sel_cfg.get("selector_frequency_hit_mode", "relative")
                ),
                selector_frequency_hit_quantile=float(
                    _base_sel_cfg.get("selector_frequency_hit_quantile", 0.80)
                ),
                selector_frequency_hit_abs=float(
                    _base_sel_cfg.get("selector_frequency_hit_abs", 1e-6)
                ),
                selector_interaction_mode=str(
                    _base_sel_cfg.get("selector_interaction_mode", "tree_path_lift")
                ),
                selector_interaction_topk_pairs=int(
                    _base_sel_cfg.get("selector_interaction_topk_pairs", 100)
                ),
                selector_interaction_max_pairs_per_feature=int(
                    _base_sel_cfg.get("selector_interaction_max_pairs_per_feature", 8)
                ),
                selector_interaction_corr_penalty=bool(
                    _base_sel_cfg.get("selector_interaction_corr_penalty", True)
                ),
                selector_family_penalty=bool(
                    _base_sel_cfg.get("selector_family_penalty", True)
                ),
                selector_emit_report=bool(_base_sel_cfg.get("selector_emit_report", True)),
                analysis_n_estimators=int(_base_sel_cfg.get("analysis_n_estimators", 192)),
                analysis_max_samples=int(_base_sel_cfg.get("analysis_max_samples", 3000)),
                min_samples_leaf_pct=float(_base_sel_cfg.get("min_samples_leaf_pct", 0.015)),
                selector_max_missing_frac=float(
                    _base_sel_cfg.get("selector_max_missing_frac", 0.15)
                ),
                selector_near_constant_dominance=float(
                    _base_sel_cfg.get("selector_near_constant_dominance", 0.999)
                ),
                selector_hysteresis_margin=float(
                    _base_sel_cfg.get("selector_hysteresis_margin", 0.05)
                ),
                selector_min_overlap=float(_base_sel_cfg.get("selector_min_overlap", 0.70)),
                composite_weights={
                    "top30": float(_base_sel_cfg.get("top30", 0.0)),
                    "global": float(_base_sel_cfg.get("global", 0.55)),
                    "stability": float(_base_sel_cfg.get("stability", 0.25)),
                    "frequency": float(_base_sel_cfg.get("frequency", 0.15)),
                    "interaction": float(_base_sel_cfg.get("interaction", 0.05)),
                },
                end_features=_mdi_target,
                cumulative_cap=0.99,
                min_share=0.0005,
                min_features=_mdi_min,
                max_features_pct=0.8,
            )
            selected_feats = _cap_selected_features(
                list(sel_res.selected_features),
                list(X_df.columns),
                target_cap=_mdi_target,
                min_features=_mdi_min,
            )
            if not selected_feats:
                tprint(f"BASE HPO[{name}]: MDI produced no usable selected features.")
                continue
            X_selected = X_df[selected_feats].to_numpy(dtype=np.float32, copy=False)
            tprint(
                f"BASE HPO[{name}]: MDI selected {len(selected_feats)}/{X_df.shape[1]} features before HPO"
            )

            result = run_base_extratrees_hpo(
                X_selected,
                y_all,
                selected_features=selected_feats,
                symbols=sym_col,
                n_trials=int(cfg.get("base_hpo_n_trials", 150)),
                n_splits=int(cfg.get("base_hpo_n_splits", 3)),
                max_rows=int(cfg.get("base_hpo_max_rows", 6500)),
                optuna_patience_trials=int(cfg.get("base_hpo_patience_trials", 30)),
                optuna_min_trials_before_stop=int(cfg.get("base_hpo_min_trials_before_stop", 50)),
                optuna_meaningful_improvement_pct=float(cfg.get("base_hpo_meaningful_improvement_pct", 0.005)),
                out_dir=hpo_out_dir,
                study_name=f"base_extratrees_hpo_{name}",
                scope_key=name,
            )
            strategy_scope_results[name] = result
            results[name] = result
            tprint(
                f"BASE HPO[{name}] COMPLETE: best_value={result.get('best_value', 'n/a')}"
            )

        if not strategy_scope_results:
            tprint(f"BASE HPO[{s_id}]: no dataset-level HPO results were produced.")
            continue

    if not results:
        tprint("BASE HPO: no strategy-level results were produced.")
        return None
    return results


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
                    tprint(
                        f"  Stable boundaries for {rname} ({src_col}): {boundaries[rname][0]:.4f}, {boundaries[rname][1]:.4f} (n={len(valid)})"
                    )
                except Exception as e:
                    tprint(f"  WARNING: Failed to compute boundaries for {rname}: {e}")
    tprint(f"  Regime boundaries computed for {len(boundaries)} features.")
    return boundaries


def _extract_required_features(bundle, cfg):
    """Extract required features from the actual runtime bundle first, config second."""

    def _add_model_feature_cols(_req, _model):
        if _model is None:
            return
        for _attr in (
            "selected_features",
            "selected_features_",
            "feature_names_",
            "feature_cols",
        ):
            _vals = getattr(_model, _attr, None)
            if isinstance(_vals, (list, tuple, set)):
                _req.update(str(v) for v in _vals if isinstance(v, str) and v)

    def _add_alpha_stack(_req, _alpha_models):
        if not isinstance(_alpha_models, dict):
            return
        for _side_bundle in _alpha_models.values():
            if not isinstance(_side_bundle, dict):
                continue
            for _kind_bundle in _side_bundle.values():
                if not isinstance(_kind_bundle, dict):
                    continue
                _req.update(
                    str(v)
                    for v in _kind_bundle.get("feat_cols", [])
                    if isinstance(v, str) and v
                )
                _by_h = _kind_bundle.get("models_by_h", {})
                if isinstance(_by_h, dict):
                    for _info in _by_h.values():
                        if isinstance(_info, dict):
                            _req.update(
                                str(v)
                                for v in _info.get("feat_cols", [])
                                if isinstance(v, str) and v
                            )

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

    for strat in get_strategies(cfg):
        base_event_trigger = str(strat.get("base_event_trigger", "") or "")
        if not base_event_trigger:
            continue
        for feat_name in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|>=|==|<|>)", base_event_trigger
        ):
            if feat_name not in {"Composite"}:
                req_feats.add(feat_name)

    # Time features are often used by dynamic rule masks but may not appear in
    # runtime model feature lists.
    req_feats.update({"sin_hod", "cos_hod"})

    if isinstance(bundle, dict):
        _add_alpha_stack(req_feats, bundle.get("alpha_models", {}))
        _meta_models = bundle.get("meta_models", {})
        if isinstance(_meta_models, dict):
            for _meta in _meta_models.values():
                _add_model_feature_cols(req_feats, _meta)
        _spike = bundle.get("spike_models", {})
        if isinstance(_spike, dict):
            for _sp in _spike.values():
                if isinstance(_sp, dict):
                    req_feats.update(
                        str(v)
                        for v in _sp.get("columns", [])
                        if isinstance(v, str) and v
                    )
                else:
                    _add_model_feature_cols(req_feats, _sp)
        _spec = bundle.get("specialist_models", {})
        if isinstance(_spec, dict):
            for _name, _mdl in _spec.items():
                if isinstance(_mdl, dict):
                    req_feats.update(
                        str(v)
                        for v in _mdl.get("columns", [])
                        if isinstance(v, str) and v
                    )
                else:
                    _add_model_feature_cols(req_feats, _mdl)

    # Sizer-triggered OOS backtest: keep feature loading constrained to
    # EV-decomposition relevant context only.
    if bool(cfg.get("sizer_oos_mode", False)):
        reg_keys = cfg.get("position_sizer_regime_feature_keys", [])
        if isinstance(reg_keys, (list, tuple, set)):
            req_feats.update(str(v) for v in reg_keys if isinstance(v, str) and v)
        out = sorted({f for f in req_feats if not str(f).startswith("pred_")})
        tprint(
            f"Feature load whitelist (sizer_oos_mode): keys={len(out)} (EV-decomposition relevant)"
        )
        return out

    # Config baskets are only used as a fallback for models that do not expose feature lists.
    if len(req_feats) <= 16:
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
    run_ts = ts_sig  # Keep original for loading artifacts
    opt_ts = ts_sig  # Used for data filtering

    oos_days = cfg.get("oos_holdout_days", 0)
    if oos_days > 0:
        opt_ts = ts_sig - pd.Timedelta(days=oos_days)
        tprint(
            f"Risk Optimization: Excluding last {oos_days} days (OOS). Training end: {opt_ts}"
        )

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
        if not df.empty:
            dfs[s] = df[df.index <= opt_ts].tail(24 * lookback_days)

    if not dfs:
        return

    panel = to_panel(dfs)
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(
        mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"]
    )

    # Load only required features to prevent OOM
    req_feats = _extract_required_features(bundle, cfg)
    feats = load_features_for_stage_or_all(cfg, run_ts, cfg["data_root"], feature_keys=req_feats)
    if feats is None:
        tprint("ERROR: Features not found for risk optimization.")
        return
    feats = _ensure_atr_pct_feature(feats, panel, cfg, symbols=train_syms)

    # Compute stable regime boundaries for meta-models
    cfg["granular_regime_boundaries"] = compute_regime_boundaries(feats)

    # We also need p_exh_hist. Load from artifacts (using run_ts).
    p_exh_hist = None

    alpha_models = bundle["alpha_models"]
    best_risk = optimize_risk_params(
        panel, feats, mkt_gates, cfg, train_syms, opt_ts, p_exh_hist, alpha_models
    )

    prev_risk = (
        state.get("risk_params", {})
        if isinstance(state.get("risk_params"), dict)
        else {}
    )
    if (
        isinstance(prev_risk, dict)
        and "signal_params" in prev_risk
        and isinstance(best_risk, dict)
    ):
        best_risk["signal_params"] = prev_risk["signal_params"]

    state["risk_params"] = best_risk
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    tprint("Risk params updated in state file.")

    # Generate risk optimization report
    try:
        run_id = run_ts.strftime("%Y%m%d_%H%M%S")
        granular = (
            best_risk.get("granular_risk", {}) if isinstance(best_risk, dict) else {}
        )
        report_path = generate_risk_report(
            run_id=run_id,
            cfg=cfg,
            granular_risk=granular,
            base_dir=cfg.get("reports_root"),
        )
        tprint(f"Risk optimization report saved to {report_path}")
    except Exception as e:
        tprint(f"WARNING: Failed to generate risk report: {e}")

    tprint("STEP: RISK OPTIMIZATION COMPLETE")


def run_backtest_step(ts_sig, margin_symbols, cfg, store, state_file):
    tprint("STEP: BACKTEST START")
    if not os.path.exists(state_file):
        tprint("State file not found.")
        return

    with open(state_file, "rb") as f:
        model_state = pickle.load(f)

    bundle = model_state.get("bundle")
    if isinstance(bundle, dict):
        try:
            native_dir = os.path.join(os.path.dirname(state_file), "native")
            if os.path.isdir(native_dir):
                native_alpha = load_alpha_models(native_dir)
                if native_alpha:
                    bundle["alpha_models"] = native_alpha
                    tprint(
                        "Backtest state patch: refreshed alpha_models from native model store"
                    )
        except Exception as _e_alpha_patch:
            tprint(
                f"WARNING: failed to refresh alpha_models from native model store: {_e_alpha_patch}"
            )

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
            tprint(
                f"Meta head coverage missing keys: {_missing[:12]}{' ...' if len(_missing) > 12 else ''}"
            )

    _ensure_meta_aliases(bundle)
    if isinstance(bundle, dict):
        _meta_now = bundle.get("meta_models")
        _meta_count = len(_meta_now) if isinstance(_meta_now, dict) else 0
        if _meta_count == 0:
            try:
                _meta_state_file = os.path.join(
                    os.path.dirname(state_file), "model_state_meta.pkl"
                )
                if os.path.exists(_meta_state_file):
                    _meta_state = joblib.load(_meta_state_file)
                    _meta_bundle = (
                        _meta_state.get("bundle", {})
                        if isinstance(_meta_state, dict)
                        else {}
                    )
                    _meta_models = (
                        _meta_bundle.get("meta_models", {})
                        if isinstance(_meta_bundle, dict)
                        else {}
                    )
                    if isinstance(_meta_models, dict) and len(_meta_models) > 0:
                        bundle["meta_models"] = _meta_models
                        _ensure_meta_aliases(bundle)
                        tprint(
                            "Backtest state patch: loaded meta_models from model_state_meta.pkl "
                            f"(count={len(_meta_models)})"
                        )
            except Exception as _e_meta_patch:
                tprint(
                    f"WARNING: failed to patch meta_models from model_state_meta.pkl: {_e_meta_patch}"
                )
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
    mkt_gates = add_regime_gates(
        mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"]
    )
    # Load only required features to prevent OOM
    req_feats = _extract_required_features(bundle, cfg)
    feat_start_ts = None
    feat_end_ts = ts_sig
    if dfs:
        try:
            feat_start_ts = min(df.index.min() for df in dfs.values() if not df.empty)
        except ValueError:
            feat_start_ts = None
    feats = load_features_for_stage_or_all(cfg, ts_sig, cfg["data_root"],
        feature_keys=req_feats,
        symbols=train_syms,
        start_ts=feat_start_ts,
        end_ts=feat_end_ts,
    )
    if feats is None:
        tprint("ERROR: Features not found for backtest.")
        return
    feats = _ensure_atr_pct_feature(feats, panel, cfg, symbols=train_syms)

    if hasattr(feats, "materialize"):
        # Materialize runtime-required matrices once, before the hourly loop.
        # This avoids repeated lazy assembly in the hot signal-generation path.
        _materialize_keys = sorted(set(req_feats) | {"atr_pct"})
        feats.materialize(_materialize_keys, progress_every=20)

    # Compute stable regime boundaries for meta-models
    cfg["granular_regime_boundaries"] = compute_regime_boundaries(feats)

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    from extreme_price_movements.data_store import load_artifact_df

    p_exh_hist = None

    test_days_cfg = int(cfg.get("oos_holdout_days", 730))
    test_days = max(730, test_days_cfg)
    if test_days != test_days_cfg:
        tprint(
            f"Backtest OOS holdout_days raised to minimum: cfg={test_days_cfg} -> used={test_days}"
        )
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

    o_s = panel["open"]
    h_s = panel["high"]
    l_s = panel["low"]
    c_s = panel["close"]
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
    _ridge_weights_path = os.path.join(
        os.path.dirname(state_file), "..", "ridge_sizer", "sizer_weights.json"
    )
    _ridge_weights_path = os.path.normpath(_ridge_weights_path)
    if os.path.exists(_ridge_weights_path):
        try:
            with open(_ridge_weights_path, "r") as _f_rw:
                risk_conf["ridge_weights_manifest"] = json.load(_f_rw)
        except Exception as _e_rw:
            tprint(f"WARNING: failed to load ridge sizer weights manifest: {_e_rw}")
    _ridge_model_path = os.path.join(
        cfg["data_root"],
        "models",
        f"ridge_position_sizer_{ts_sig.strftime('%Y%m%d_%H%M%S')}.json",
    )
    if not os.path.exists(_ridge_model_path):
        _ridge_model_path = os.path.join(
            "extreme_price_movements",
            "data",
            "models",
            f"ridge_position_sizer_{ts_sig.strftime('%Y%m%d_%H%M%S')}.json",
        )
    if os.path.exists(_ridge_model_path):
        try:
            risk_conf["ridge_sizer"] = RidgePositionSizer.load(_ridge_model_path)
        except Exception as _e_rs:
            tprint(
                f"WARNING: failed to load RidgePositionSizer for backtest runtime: {_e_rs}"
            )
    _ps_manifest = (
        model_state.get("ev_decomposition", {}) if isinstance(model_state, dict) else {}
    )
    if isinstance(_ps_manifest, dict) and _ps_manifest.get("bundle_path"):
        risk_conf["ev_decomposition_bundle_path"] = _ps_manifest.get("bundle_path")
        try:
            risk_conf["ev_decomposition_bundle"] = load_ev_decomposition_bundle(
                _ps_manifest.get("bundle_path"),
                allow_unknown_version=bool(
                    cfg.get("ev_decomposition_allow_unknown_bundle_version", False)
                ),
                verbose=True,
            )
        except Exception as _e_ps_bundle:
            tprint(f"WARNING: failed to preload EVDecompositionBundle: {_e_ps_bundle}")

    fee_bps = cfg.get("fee_bps", 25.0)
    cost = CostModel(
        fee_side=float(fee_bps) / 10000.0,
        slippage_side=float(cfg.get("slippage_bps", 0.0)) / 10000.0,
    )
    assert_units(cost)
    emit_run_header(
        tprint=tprint,
        run_id=run_id,
        policy_version=str(run_id),
        cost_model={
            "fee_side": float(cost.fee_side),
            "slippage_side": float(cost.slippage_side),
            "round_trip": float(cost.round_trip),
        },
        extra={
            "stage": "backtest_signal_optimization",
            "sizer_backend_used": str(cfg.get("sizer_backend_used", "ridge")),
        },
    )
    if bool(cfg.get("signal_opt_debug", True)):
        tprint(
            "SignalOptConfig: "
            f"limit_offset_mode={cfg.get('limit_offset_mode', 'heuristic')} "
            f"limit_offset_unit={cfg.get('limit_offset_unit', 'bps')} "
            f"limit_offset_bounds=[{cfg.get('limit_offset_min', 5.0)}, {cfg.get('limit_offset_max', 50.0)}] "
            f"use_limit_orders={bool(cfg.get('use_limit_orders', False))} "
            f"limit_offset_bps={float(cfg.get('limit_offset_bps', 0.0)):.1f} "
            f"exit_limit_offset_bps={float(cfg.get('exit_limit_offset_bps', 0.0)):.1f} "
            f"sizer_backend=ridge "
            f"ranking_trade_percentile_threshold={float(cfg.get('ranking_trade_percentile_threshold', 0.90)):.2f}"
        )

    def rank01(x: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
        pct = (ranks - 1.0) / max(1.0, x.size - 1.0)
        return pct if higher_is_better else (1.0 - pct)

    def pnl_sortino_dd_utility(
        pnls, sortinos, max_dds, w_pnl=0.65, w_sortino=0.25, w_dd=0.10
    ):
        return (
            w_pnl * rank01(pnls, True)
            + w_sortino * rank01(sortinos, True)
            + w_dd * rank01(max_dds, True)
        )

    def compute_metrics(trades):
        if not trades:
            return 0.0, 0.0, 0.0, 0.0, 0
        rets = np.array([x["pnl"] for x in trades], dtype=np.float64)
        pnl = float(np.sum(rets))
        neg = rets[rets < 0]
        sortino = (
            float(np.mean(rets) / (np.std(neg) + 1e-12))
            if neg.size > 0
            else float(np.mean(rets) / 1e-12)
        )
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
        local_risk.setdefault("_candidate_ts_cache", {})
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
        throttle_dd_thr = float(
            cfg.get("throttle_dd_threshold", -0.02)
        )  # cumPnL drawdown trigger
        throttle_factor = float(
            cfg.get("throttle_sizing_factor", 0.5)
        )  # reduce sizing to 50%

        from collections import defaultdict

        daily_bucket_counts = defaultdict(
            lambda: defaultdict(int)
        )  # date -> bucket -> count
        daily_total_counts = defaultdict(int)  # date -> total count

        def _align_ts_like(ref_ts, other_ts):
            ref = pd.Timestamp(ref_ts)
            other = pd.Timestamp(other_ts)
            if ref.tz is None and other.tz is not None:
                return other.tz_convert(None)
            if ref.tz is not None and other.tz is None:
                return other.tz_localize(ref.tz)
            if (
                ref.tz is not None
                and other.tz is not None
                and str(ref.tz) != str(other.tz)
            ):
                return other.tz_convert(ref.tz)
            return other

        if bool(cfg.get("signal_opt_debug", True)):
            tprint(
                "  RunSlice start: "
                f"hours={len(ts_list)} thr_long={float(signal_params.get('thr_long', 0.0)):.6f} "
                f"thr_short={float(signal_params.get('thr_short', 0.0)):.6f} "
                f"score_gate_q={float(signal_params.get('score_gate_q', 0.0)):.2f} "
                f"k_frac_long={signal_params.get('k_frac_long', None)} "
                f"k_frac_short={signal_params.get('k_frac_short', None)}"
            )

        _prefilter_key = (
            float(cfg.get("trade_extreme_pct", 0.07)),
            float(cfg.get("train_min_range_pct", 0.07)),
            float(cfg.get("train_min_vol_zscore", 1.6)),
        )
        _candidate_ts = local_risk["_candidate_ts_cache"].get(_prefilter_key)
        if _candidate_ts is None:
            _mask = select_trade_candidates_vectorized(
                panel,
                feats,
                pct=float(cfg.get("trade_extreme_pct", 0.07)),
                metric="ret24h",
                min_move_12h_pct=None,
                min_range_pct=float(cfg.get("train_min_range_pct", 0.07)),
                min_vol_zscore=float(cfg.get("train_min_vol_zscore", 1.6)),
                chop_thr=1.0,
            )
            if _mask is not None:
                _candidate_ts = set(_mask.index[_mask.any(axis=1)])
            else:
                _candidate_ts = set()
            local_risk["_candidate_ts_cache"][_prefilter_key] = _candidate_ts
        if _candidate_ts:
            _ts_orig = len(ts_list)
            ts_list = [t for t in ts_list if t in _candidate_ts]
            if bool(cfg.get("signal_opt_debug", True)):
                tprint(
                    f"  Candidate timestamp prefilter kept {len(ts_list)}/{_ts_orig} hours "
                    f"({(100.0 * len(ts_list) / max(1, _ts_orig)):.1f}%)"
                )
        elif bool(cfg.get("signal_opt_debug", True)):
            tprint("  Candidate timestamp prefilter found no eligible hours")

        for i, t in enumerate(ts_list):
            if i % 20 == 0:
                tprint(
                    f"  Signal generation progress: {i}/{len(ts_list)} ({i/len(ts_list):.1%}) - {t}"
                )
            # --- Regime throttle: check recent closed-trade drawdown ---
            size_mult = 1.0
            if len(trades) >= throttle_lookback:
                recent_pnls = np.array(
                    [tr["pnl"] for tr in trades[-throttle_lookback:]], dtype=np.float64
                )
                cum = np.cumsum(recent_pnls)
                peak = np.maximum.accumulate(cum)
                dd = cum - peak
                if dd[-1] < throttle_dd_thr:
                    size_mult = throttle_factor

            # Count currently open trades and their total weight at this timestamp
            open_trades = []
            for tr in trades:
                _t_cmp = pd.Timestamp(t)
                _tr_entry = _align_ts_like(_t_cmp, tr["entry_ts"])
                _tr_exit = _align_ts_like(_t_cmp, tr["exit_ts"])
                _t_cmp = _align_ts_like(_tr_entry, _t_cmp)
                _tr_exit = _align_ts_like(_tr_entry, _tr_exit)
                if _tr_entry <= _t_cmp < _tr_exit:
                    open_trades.append(tr)
            open_count = len(open_trades)
            open_weight = sum(abs(tr.get("weight", 0.0)) for tr in open_trades)
            remaining_slots = max(0, max_concurrent - open_count)
            remaining_weight = max(0.0, max_portfolio_weight - open_weight)
            orders = generate_hourly_signals(
                t, feats, mkt_gates, bundle, local_risk, cfg, p_exh_hist, []
            )
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
                    tprint(
                        f"  WARNING: No granular risk for keys {risk_keys}, falling back to global cfg"
                    )
                rp = flatten_bucket_policy(rp)
                atr_val_for_policy = (
                    float(atr_s[sym].loc[entry_ts])
                    if entry_ts in atr_s[sym].index
                    else 0.02
                )
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
                adj = 1.0 + sc_scale * abs(score)

                temp_cfg = cfg.copy()
                temp_cfg["risk_k_sl"] = k_sl * adj
                temp_cfg["risk_k_trail_start"] = k_ts
                temp_cfg["risk_k_trail_dist"] = k_td

                if tp_mult is not None:
                    temp_cfg["tp_mult"] = tp_mult
                if sl_mult is not None:
                    temp_cfg["sl_mult"] = sl_mult
                temp_cfg["trail_mult"] = trail_mult
                temp_cfg["use_limit_orders"] = bool(cfg.get("use_limit_orders", True))
                temp_cfg["limit_offset_bps"] = float(
                    pol.get(
                        "limit_offset_bps_dynamic",
                        temp_cfg.get(
                            "limit_offset_bps", cfg.get("limit_offset_bps", 0.0)
                        ),
                    )
                )
                temp_cfg["trail_mult"] = float(
                    pol.get("trail_mult_eff", temp_cfg.get("trail_mult", trail_mult))
                )
                temp_cfg["giveback_pct"] = float(
                    pol.get(
                        "giveback_pct_eff",
                        temp_cfg.get("giveback_pct", cfg.get("giveback_pct", 0.005)),
                    )
                )
                temp_cfg["profit_lock_amount"] = float(
                    pol.get(
                        "profit_lock_amount_eff",
                        temp_cfg.get(
                            "profit_lock_amount", cfg.get("profit_lock_amount", 0.003)
                        ),
                    )
                )
                temp_cfg["kill_c"] = float(
                    pol.get(
                        "kill_c_eff", temp_cfg.get("kill_c", cfg.get("kill_c", 0.005))
                    )
                )
                if pol.get("sl_distance_atr_eff") is not None:
                    temp_cfg["sl_mult"] = float(max(0.05, pol["sl_distance_atr_eff"]))
                if pol.get("tp_distance_atr_eff") is not None:
                    temp_cfg["tp_mult"] = float(max(0.05, pol["tp_distance_atr_eff"]))

                # Per-bucket profit-protection params (absolute % of price)
                for pp_key in (
                    "be_threshold_pct",
                    "profit_lock_pct",
                    "profit_lock_amount",
                    "giveback_pct",
                    "max_loss_pct",
                ):
                    if pp_key in rp:
                        temp_cfg[pp_key] = rp[pp_key]

                # Per-bucket vol scaling params (from risk optimization)
                if "vol_lo" in rp:
                    temp_cfg["vol_lo"] = rp["vol_lo"]
                if "vol_hi" in rp:
                    temp_cfg["vol_hi"] = rp["vol_hi"]
                if "vol_z_max" in rp:
                    temp_cfg["vol_z_max"] = rp["vol_z_max"]

                # Per-bucket max hold hours (from risk optimization, default 24)
                hold_hours = int(
                    pol.get("max_hold_hours_eff", rp.get("max_hold_hours", 24))
                )

                # Initialize CCXT exchange for 15m precision if enabled
                exchange = None
                if cfg.get("use_15m_precision", False):
                    try:
                        import ccxt

                        exchange = ccxt.binance()
                    except Exception as e:
                        tprint(f"WARNING: Failed to initialize CCXT exchange: {e}")

                ret, exit_ts, reason, trade_extras = simulate_trade_hourly(
                    o_s[sym],
                    h_s[sym],
                    l_s[sym],
                    c_s[sym],
                    atr_s[sym],
                    entry_ts,
                    entry_px,
                    side,
                    temp_cfg,
                    max_hold_hours=hold_hours,
                    exchange=exchange,
                    symbol=sym if "/" in sym else sym.replace("USDT", "/USDT"),
                    cost=cost,  # CCXT format
                )
                assert_pos_w(weight)
                side_sign = 1 if side == "long" else -1

                # Dynamic cost model: handle limit fills and conditional fee concessions
                fee_reduction_reasons = [
                    "stop_loss",
                    "early_invalidation",
                    "giveback_exit",
                ]

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

                actual_cost = CostModel(
                    fee_side=avg_fee_side,
                    slippage_side=float(cfg.get("slippage_bps", 0.0)) / 10000.0,
                )

                net_ret = trade_return_net(
                    raw_ret_underlying=ret,
                    side=side_sign,
                    pos_w=weight,
                    cost=actual_cost,
                )
                pnl = net_ret

                # Store risk parameters + MAE/MFE for aggregate statistics
                bucket_label = f"{side.upper()}_{dom.upper()}"
                trade_record = {
                    "entry_ts": entry_ts,
                    "symbol": sym,
                    "side": side,
                    "dom": dom,
                    "asset": sym,
                    "t_entry": int(entry_ts.value),
                    "t_exit": int(exit_ts.value),
                    "bucket": bucket_label,
                    "score": score,
                    "weight": weight,
                    "pos_w": weight,
                    "ret": net_ret,
                    "pnl": pnl,
                    "net_ret_equity": net_ret,
                    "cost_rt": cost.round_trip,
                    "raw_ret_underlying": ret,
                    "gross_ret": ret,
                    "exit_ts": exit_ts,
                    "reason": reason,
                    "exit_reason": reason,
                    "sl_mult": temp_cfg.get("sl_mult", 0.5),
                    "tp_mult": temp_cfg.get("tp_mult", 1.0),
                    "trail_mult": temp_cfg.get("trail_mult", 0.25),
                    "entry_px": entry_px,
                    "atr": float(atr_s[sym].loc[entry_ts])
                    if entry_ts in atr_s[sym].index
                    else 0.02,
                    "sl_pct": trade_extras.get("sl_pct", 0.0),
                    "tp_pct": trade_extras.get("tp_pct", 0.0),
                    "mae_pct": trade_extras.get("mae_pct", 0.0),
                    "mfe_pct": trade_extras.get("mfe_pct", 0.0),
                    "bars_to_mfe": trade_extras.get("bars_to_mfe", 0),
                    "exit_stage": trade_extras.get("exit_stage", 0),
                    "filled_via_limit": trade_extras.get("filled_via_limit", False),
                    "exit_filled_via_limit": trade_extras.get(
                        "exit_filled_via_limit", False
                    ),
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
                    trade_record["G_VOL"] = (
                        int(mrk.get("G_VOL", 0)) if "G_VOL" in mrk.index else 0
                    )
                    trade_record["G_TREND"] = (
                        int(mrk.get("G_TREND", 0)) if "G_TREND" in mrk.index else 0
                    )
                    trade_record["mkt_rv"] = (
                        float(mrk.get("mkt_rv", 0.0)) if "mkt_rv" in mrk.index else 0.0
                    )
                    trade_record["mkt_ret24h"] = (
                        float(mrk.get("mkt_ret24h", 0.0))
                        if "mkt_ret24h" in mrk.index
                        else 0.0
                    )
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
            tp_sl_binds = reason_counts.get("stop_loss", 0) + reason_counts.get(
                "trailing_stop", 0
            )
            hold_by_reason = {}
            mean_ret_by_reason = {}

            entry_bps = float(cfg.get("limit_offset_bps", 0.0))
            exit_bps = float(cfg.get("exit_limit_offset_bps", 0.0))
            offset_bps = entry_bps / 10000.0

            tprint(
                f"\n  Gross PnL Comparison (Market vs {entry_bps:.0f}bps Entry / {exit_bps:.0f}bps Exit) by Exit Reason:"
            )

            for rs in sorted(reason_counts.keys()):
                rs_rows = [tr for tr in trades if str(tr.get("reason")) == rs]
                holds = [
                    float(
                        (
                            pd.Timestamp(tr["exit_ts"]) - pd.Timestamp(tr["entry_ts"])
                        ).total_seconds()
                        / 3600.0
                    )
                    for tr in rs_rows
                ]
                rets = [float(tr.get("ret", 0.0)) for tr in rs_rows]

                # Calculate gross unscaled returns isolated from capital limits and weights
                market_gross_rets = []
                limit_gross_rets = []
                for tr in rs_rows:
                    g = float(
                        tr.get("gross_ret", 0.0)
                    )  # contains both limit entry and limit exit
                    limit_gross_rets.append(g)

                    # Deduct the extra margin gained by limit orders to simulate the "Market Order" baseline
                    entry_bonus = offset_bps if tr.get("filled_via_limit") else 0.0
                    exit_bonus = float(tr.get("exit_limit_bonus", 0.0))
                    market_gross_rets.append(g - entry_bonus - exit_bonus)

                hold_by_reason[rs] = float(np.median(holds)) if holds else 0.0
                mean_ret_by_reason[rs] = float(np.mean(rets)) if rets else 0.0

                avg_mkt_g = (
                    float(np.mean(market_gross_rets)) if market_gross_rets else 0.0
                )
                avg_lmt_g = (
                    float(np.mean(limit_gross_rets)) if limit_gross_rets else 0.0
                )
                tprint(
                    f"    {rs.ljust(20)}: Count={len(rs_rows):<3} | Market Gross: {avg_mkt_g*100:6.3f}% | Limit Gross: {avg_lmt_g*100:6.3f}% | Delta: +{(avg_lmt_g - avg_mkt_g)*100:5.3f}%"
                )

            # === Detailed PnL Comparison Metrics ===
            try:
                from extreme_price_movements.limit_order_pricer import (
                    compute_exit_limit_fill_impact,
                    compute_pnl_comparison_metrics,
                )

                # Convert trades list to DataFrame for metrics computation
                trades_df = pd.DataFrame(trades)

                # Validate required columns exist before computing metrics
                required_cols = ["entry_px"]
                if not all(col in trades_df.columns for col in required_cols):
                    tprint(
                        f"  WARNING: Skipping PnL comparison - missing required columns: {required_cols}"
                    )
                elif trades_df.empty:
                    tprint("  WARNING: Skipping PnL comparison - no trades to analyze")
                else:
                    # Get exit price from trade records
                    if (
                        "exit_px" not in trades_df.columns
                        and "gross_ret" in trades_df.columns
                    ):
                        # Calculate exit price from gross return
                        trades_df["is_long"] = (trades_df["side"] == "long").astype(int)
                        trades_df["exit_px"] = np.where(
                            trades_df["is_long"] == 1,
                            trades_df["entry_px"] * (1.0 + trades_df["gross_ret"]),
                            trades_df["entry_px"] * (1.0 - trades_df["gross_ret"]),
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
                    fee_limit_entry=float(cfg.get("fee_bps_limit_entry", 10.0))
                    / 10000.0,
                    fee_limit_exit=float(cfg.get("fee_bps_limit_exit", 10.0)) / 10000.0,
                    fee_market_entry=float(cfg.get("fee_bps_market", 25.0)) / 10000.0,
                    fee_market_exit=float(cfg.get("fee_bps_market_exit", 25.0))
                    / 10000.0,
                )

                # Print detailed comparison
                tprint(
                    f"\n  === Detailed PnL Comparison (Solution A vs B vs Market) ==="
                )
                tprint(f"    Total Trades: {comparison_metrics['n_trades']}")
                tprint(
                    f"    Fee Savings (Entry): {comparison_metrics['fee_savings_entry']*10000:.1f} bps"
                )
                tprint(
                    f"    Fee Savings (Exit): {comparison_metrics['fee_savings_exit']*10000:.1f} bps"
                )

                tprint(f"\n  --- Mean PnL by Strategy ---")
                tprint(
                    f"    Baseline (Limit Entry):    {comparison_metrics['baseline']['mean']*100:7.3f}% | Sharpe: {comparison_metrics['baseline']['sharpe']:.2f} | Win Rate: {comparison_metrics['baseline']['win_rate']*100:.1f}%"
                )
                tprint(
                    f"    Solution A (New TP/SL):   {comparison_metrics['solution_a']['mean']*100:7.3f}% | Sharpe: {comparison_metrics['solution_a']['sharpe']:.2f} | Win Rate: {comparison_metrics['solution_a']['win_rate']*100:.1f}%"
                )
                tprint(
                    f"    Solution B (Same Dist):   {comparison_metrics['solution_b']['mean']*100:7.3f}% | Sharpe: {comparison_metrics['solution_b']['sharpe']:.2f} | Win Rate: {comparison_metrics['solution_b']['win_rate']*100:.1f}%"
                )
                tprint(
                    f"    Market Order (No Offset): {comparison_metrics['market_order']['mean']*100:7.3f}% | Sharpe: {comparison_metrics['market_order']['sharpe']:.2f} | Win Rate: {comparison_metrics['market_order']['win_rate']*100:.1f}%"
                )

                tprint(f"\n  --- PnL Differences ---")
                tprint(
                    f"    Solution A vs Baseline: {comparison_metrics['diff_a_vs_baseline']*100:+6.3f}%"
                )
                tprint(
                    f"    Solution B vs Baseline: {comparison_metrics['diff_b_vs_baseline']*100:+6.3f}%"
                )
                tprint(
                    f"    Solution A vs Market:   {comparison_metrics['diff_a_vs_market']*100:+6.3f}%"
                )
                tprint(
                    f"    Solution B vs Market:   {comparison_metrics['diff_b_vs_market']*100:+6.3f}%"
                )

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
                    tprint(
                        f"    Exit via Limit: {exit_impact['exit_limit_filled']} ({exit_impact['fill_rate']*100:.1f}%)"
                    )
                    tprint(
                        f"    Exit via Market: {exit_impact['exit_limit_not_filled']}"
                    )
                    if "mean_pnl_filled" in exit_impact:
                        tprint(
                            f"    Mean PnL (Filled): {exit_impact['mean_pnl_filled']*100:.3f}%"
                        )
                    if "mean_pnl_not_filled" in exit_impact:
                        tprint(
                            f"    Mean PnL (Not Filled): {exit_impact['mean_pnl_not_filled']*100:.3f}%"
                        )

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
            avg_trail_pct = np.mean(
                [t["trail_mult"] * t["sl_pct"] * 100 for t in trades]
            )  # trail ≈ trail_mult * barrier
            avg_mae = np.mean([t["mae_pct"] * 100 for t in trades])
            avg_mfe = np.mean([t["mfe_pct"] * 100 for t in trades])

            mae_20bps = np.mean([t["mae_pct"] * 100 >= 0.2 for t in trades]) * 100
            mae_30bps = np.mean([t["mae_pct"] * 100 >= 0.3 for t in trades]) * 100
            mae_40bps = np.mean([t["mae_pct"] * 100 >= 0.4 for t in trades]) * 100
            mae_50bps = np.mean([t["mae_pct"] * 100 >= 0.5 for t in trades]) * 100

            tprint(
                f"\n  TP/SL Statistics ({len(trades)} trades) [actual barrier-scaled]:"
            )
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

    split = max(
        24, int(len(valid_ts) * 0.2)
    )  # 20% for signal calibration, 80% for OOS test
    train_ts = valid_ts[:split]
    test_ts = valid_ts[split:]

    raw_train, _raw_diag = run_slice(
        train_ts,
        {
            "thr_long": -1e9,
            "thr_short": 1e9,
            "k_long": cfg.get("k_long", 10),
            "k_short": cfg.get("k_short", 10),
            "size_min": 0.03,
            "size_max": 0.15,
            "size_k": 2.0,
            "size_x0": 0.5,
            "size_zcap": 4.0,
        },
    )
    train_abs = np.array([abs(t["score"]) for t in raw_train], dtype=np.float64)
    q50 = float(np.quantile(train_abs, 0.5)) if train_abs.size else 0.0
    q75 = float(np.quantile(train_abs, 0.75)) if train_abs.size else max(q50, 1e-6)
    q90 = (
        float(np.quantile(train_abs, 0.9)) if train_abs.size else max(q50 + 1e-6, 1e-3)
    )
    q95 = (
        float(np.quantile(train_abs, 0.95))
        if train_abs.size
        else max(q90 + 1e-6, 1.1 * q90)
    )
    q98 = (
        float(np.quantile(train_abs, 0.98))
        if train_abs.size
        else max(q95 + 1e-6, 1.1 * q95)
    )
    tprint(
        f"Global Score Distribution: P50={q50:.6f}, P75={q75:.6f}, P90={q90:.6f}, P95={q95:.6f}, P98={q98:.6f}"
    )

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
                tprint(
                    f"  ScoreScale WARNING: {channel_name} has degenerate IQR "
                    f"({s:.2e}). Disabling normalization (center=0, scale=1)."
                )
                return 0.0, 1.0
            n_clipped = int(np.sum(arr < q05) + np.sum(arr > q95))
            if n_clipped > 0:
                tprint(
                    f"  ScoreScale: {channel_name} winsorized {n_clipped}/{len(arr)} outliers "
                    f"to [{q05:.4f}, {q95:.4f}]"
                )
            return c, s

        # Get strategies for dynamic score scaling
        from extreme_price_movements.strategy_registry import get_strategies

        strategies = get_strategies(cfg)

        score_scale_params = {}
        score_diagnostics = []

        for strat in strategies:
            strat_id = strat["strategy_id"]
            side = strat["trade_side"]

            # Determine score column: score_mr or score_tf based on strategy type
            is_mr = strat.get("is_mr", "mr" in strat_id.lower())
            score_col = "score_mr" if is_mr else "score_tf"

            # Get scores for this side and strategy type
            side_scores = side_all[side_all["side_key"] == side][score_col].values

            if len(side_scores) > 0:
                c, s = _center_scale(side_scores, f"{side}_{strat_id}")
                score_scale_params[f"{strat_id}_center"] = c
                score_scale_params[f"{strat_id}_scale"] = s
                score_diagnostics.append((strat_id, side, len(side_scores), c, s))
            else:
                score_scale_params[f"{strat_id}_center"] = 0.0
                score_scale_params[f"{strat_id}_scale"] = 1.0

        # Log score scale params
        diag_strs = [
            f"{sid}({side}):n={n},c={c:.4f},s={s:.4f}"
            for sid, side, n, c, s in score_diagnostics
        ]
        tprint(f"Score scale params: {' '.join(diag_strs)}")

        # Log raw score distributions for diagnostics
        for strat in strategies:
            strat_id = strat["strategy_id"]
            side = strat["trade_side"]
            is_mr = strat.get("is_mr", "mr" in strat_id.lower())
            score_col = "score_mr" if is_mr else "score_tf"
            arr = side_all[side_all["side_key"] == side][score_col].values
            if len(arr) > 0:
                tprint(
                    f"  {strat_id} ({side}) scores: n={len(arr)}, mean={np.mean(arr):.6f}, std={np.std(arr):.6f}, "
                    f"q10={np.quantile(arr,0.1):.6f}, q50={np.quantile(arr,0.5):.6f}, q90={np.quantile(arr,0.9):.6f}"
                )
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
            "thr_long": cfg.get("thr_long", q75),
            "thr_short": cfg.get("thr_short", q75),
            "k_long": cfg.get("k_long", 10),
            "k_short": cfg.get("k_short", 10),
            "k_frac_long": top_frac,
            "k_frac_short": top_frac,
            "score_gate_q": 0.70,
            "size_min": 0.03,
            "size_max": 0.15,
            "size_k": 2.0,
            "size_x0": 0.5,
            "size_zcap": 4.0,
            "size_q50": q50,
            "size_q90": q90,
            "size_q95": q95,
            "size_q98": q98,
            "score_scale_params": score_scale_params,
        }

    tprint(f"Selected signal params: {best_signal_params}")

    test_trades, test_diag = run_slice(test_ts, best_signal_params)
    pnl, sortino, max_dd, win_rate, count = compute_metrics(test_trades)
    avg_pnl = pnl / count if count > 0 else 0.0
    tprint(
        f"Backtest OOS Result: Trades={count}, PnL={pnl:.6f}, AvgPnL={avg_pnl:.6f}, Sortino={sortino:.6f}, MaxDD={max_dd:.6f}, WinRate={win_rate:.2f}"
    )
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
        tprint(
            f"--- OOS Period: {ts_min.date()} to {ts_max.date()} ({n_days:.0f} days) ---"
        )
        tprint(f"  Total trades: {len(df_t)}, Trades/day: {len(df_t)/n_days:.1f}")

        tprint("--- OOS Breakdown ---")
        for side in ["long", "short"]:
            df_s = df_t[df_t["side"] == side]
            if not df_s.empty:
                s_pnl = df_s["pnl"].sum()
                s_wr = (df_s["pnl"] > 0).mean()
                tprint(
                    f"  {side.upper()}: Trades={len(df_s)} ({len(df_s)/n_days:.1f}/day), PnL={s_pnl:.4f}, WinRate={s_wr:.2f}"
                )
        for dom in ["mr", "tf"]:
            df_d = df_t[df_t["dom"] == dom]
            if not df_d.empty:
                d_pnl = df_d["pnl"].sum()
                d_wr = (df_d["pnl"] > 0).mean()
                tprint(
                    f"  {dom.upper()}: Trades={len(df_d)} ({len(df_d)/n_days:.1f}/day), PnL={d_pnl:.4f}, WinRate={d_wr:.2f}"
                )

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
                down_std = np.sqrt(np.mean(neg**2)) if len(neg) > 0 else 1e-9
                bucket_sortino = np.mean(rets) / down_std if down_std > 1e-9 else 0.0

                # Max drawdown for this bucket (sequential equity curve)
                eq = np.cumsum(rets)
                running_max = np.maximum.accumulate(eq)
                dd = eq - running_max
                bucket_mdd = float(dd.min()) if len(dd) > 0 else 0.0

                # Profit factor
                gross_win = float(rets[rets > 0].sum()) if (rets > 0).any() else 0.0
                gross_loss = (
                    float(abs(rets[rets < 0].sum())) if (rets < 0).any() else 1e-9
                )
                pf = gross_win / gross_loss

                tprint(f"\n--- {bucket.upper()} (n={n}, {n/n_days:.1f}/day) ---")
                tprint(
                    f"  PnL={sd_pnl:.4f}  AvgPnL={avg_pnl:.6f}  WR={sd_wr:.2f}  Sortino={bucket_sortino:.3f}  MaxDD={bucket_mdd:.4f}  PF={pf:.2f}"
                )

                # Win/loss asymmetry
                wins = df_sd[df_sd["pnl"] > 0]
                losses = df_sd[df_sd["pnl"] <= 0]
                avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0.0
                avg_loss = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
                payoff = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-9 else 0.0
                avg_win_ret = float(wins["gross_ret"].mean()) if len(wins) > 0 else 0.0
                avg_loss_ret = (
                    float(losses["gross_ret"].mean()) if len(losses) > 0 else 0.0
                )
                tprint(
                    f"  Win/Loss: AvgWin={avg_win:.6f} ({avg_win_ret:.4f} ret)  AvgLoss={avg_loss:.6f} ({avg_loss_ret:.4f} ret)  Payoff={payoff:.2f}"
                )

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
                tprint(
                    f"  Hold(h): mean={hold.mean():.1f}  med={hold.median():.1f}  min={hold.min():.0f}  max={hold.max():.0f}"
                )
                # Hold duration for wins vs losses
                if len(wins) > 0 and len(losses) > 0:
                    tprint(
                        f"  Hold wins={wins['_hold_h'].mean():.1f}h  Hold losses={losses['_hold_h'].mean():.1f}h"
                    )

                # Score distribution
                sc = df_sd["score"].abs()
                tprint(
                    f"  |Score|: mean={sc.mean():.3f}  med={sc.median():.3f}  q10={sc.quantile(0.1):.3f}  q90={sc.quantile(0.9):.3f}"
                )

                # Spearman(|score|, ret) — key monotonicity diagnostic
                from scipy.stats import spearmanr

                if len(sc) >= 5:
                    sp_corr, sp_pval = spearmanr(sc.values, df_sd["ret"].values)
                    tprint(
                        f"  Spearman(|score|, ret): {sp_corr:+.3f} (p={sp_pval:.3f})"
                        f"{'  *** NEGATIVE = conviction paradox ***' if sp_corr < -0.05 else ''}"
                    )

                # Survival metric: % of trades that reach trailing stop activation
                if "reason" in df_sd.columns:
                    n_trail = (df_sd["reason"] == "trailing_stop").sum()
                    n_sl = (df_sd["reason"] == "stop_loss").sum()
                    survival_rate = n_trail / max(1, n_trail + n_sl)
                    tprint(
                        f"  Survival-to-trail: {survival_rate:.1%} ({n_trail} trail / {n_sl} SL)"
                    )

                # Score vs outcome: high-conviction vs low-conviction
                sc_med = sc.median()
                hi_conv = df_sd[sc >= sc_med]
                lo_conv = df_sd[sc < sc_med]
                if len(hi_conv) > 0 and len(lo_conv) > 0:
                    tprint(
                        f"  Hi-conv(|s|>={sc_med:.3f}): n={len(hi_conv)} PnL={hi_conv['pnl'].sum():.4f} WR={(hi_conv['pnl']>0).mean():.2f}"
                    )
                    tprint(
                        f"  Lo-conv(|s|< {sc_med:.3f}): n={len(lo_conv)} PnL={lo_conv['pnl'].sum():.4f} WR={(lo_conv['pnl']>0).mean():.2f}"
                    )
                    # Survival comparison by conviction
                    if "reason" in df_sd.columns:
                        hi_surv = (hi_conv["reason"] == "trailing_stop").sum() / max(
                            1, len(hi_conv)
                        )
                        lo_surv = (lo_conv["reason"] == "trailing_stop").sum() / max(
                            1, len(lo_conv)
                        )
                        tprint(
                            f"  Trail-survival: Hi-conv={hi_surv:.1%}  Lo-conv={lo_surv:.1%}"
                        )

                # ========================================================================
                # CONFIDENCE QUARTILE ANALYSIS (ENHANCED)
                # ========================================================================
                try:
                    sc_abs = sc.abs()
                    df_sd = df_sd.copy()
                    df_sd["confidence_bin"] = pd.qcut(
                        sc_abs,
                        q=4,
                        labels=["Q1_Low", "Q2", "Q3", "Q4_High"],
                        duplicates="drop",
                    )

                    tprint("  Confidence Calibration:")
                    tprint(
                        f"    {'Quartile':<10} {'N':>4} {'WR':>5} {'PnL':>8} {'AvgRet':>8} {'MFE%':>6} {'MAE%':>6} {'MFE/MAE':>7} {'Capture':>7} {'Trail%':>6}"
                    )
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
                        b_cap = (
                            (
                                bt_w["gross_ret"].mean()
                                / max(bt_w["mfe_pct"].mean(), 1e-9)
                            )
                            if len(bt_w) > 0
                            else 0.0
                        )
                        # Trail survival %
                        b_trail = (
                            (bt["reason"] == "trailing_stop").mean() * 100
                            if "reason" in bt.columns
                            else 0.0
                        )
                        tprint(
                            f"    {bin_label:<10} {len(bt):>4} {b_wr:>5.2f} {b_pnl:>+8.4f} {b_ret:>+8.4f} {b_mfe:>6.2f} {b_mae:>6.2f} {b_ratio:>7.2f} {b_cap:>7.2f} {b_trail:>5.1f}%"
                        )
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
                        tprint(
                            f"    {'Regime':<20} {'N':>4} {'WR':>5} {'PnL':>8} {'MFE%':>6} {'MAE%':>6} {'MFE/MAE':>7} {'Capture':>7}"
                        )
                        for gv in [0, 1]:
                            for gt in [0, 1]:
                                regime = df_sd[
                                    (df_sd["G_VOL"] == gv) & (df_sd["G_TREND"] == gt)
                                ]
                                if len(regime) < 3:
                                    continue
                                label = f"VOL={'Hi' if gv else 'Lo'}_TREND={'Hi' if gt else 'Lo'}"
                                r_wr = (regime["pnl"] > 0).mean()
                                r_pnl = regime["pnl"].sum()
                                r_mfe = regime["mfe_pct"].mean() * 100
                                r_mae = regime["mae_pct"].mean() * 100
                                r_ratio = r_mfe / max(r_mae, 0.01)
                                r_w = regime[regime["pnl"] > 0]
                                r_cap = (
                                    (
                                        r_w["gross_ret"].mean()
                                        / max(r_w["mfe_pct"].mean(), 1e-9)
                                    )
                                    if len(r_w) > 0
                                    else 0.0
                                )
                                tprint(
                                    f"    {label:<20} {len(regime):>4} {r_wr:>5.2f} {r_pnl:>+8.4f} {r_mfe:>6.2f} {r_mae:>6.2f} {r_ratio:>7.2f} {r_cap:>7.2f}"
                                )
                    except Exception as e:
                        tprint(f"  Warning: Regime analysis failed: {e}")

                # Weight/sizing stats
                wt = df_sd["weight"]
                tprint(
                    f"  Weight: mean={wt.mean():.4f}  med={wt.median():.4f}  min={wt.min():.4f}  max={wt.max():.4f}"
                )

                # Temporal half-split: is performance degrading?
                mid = len(df_sd) // 2
                if mid > 5:
                    first_half = df_sd.iloc[:mid]
                    second_half = df_sd.iloc[mid:]
                    fh_pnl = first_half["pnl"].sum()
                    sh_pnl = second_half["pnl"].sum()
                    fh_wr = (first_half["pnl"] > 0).mean()
                    sh_wr = (second_half["pnl"] > 0).mean()
                    tprint(
                        f"  1st half: n={len(first_half)} PnL={fh_pnl:.4f} WR={fh_wr:.2f}  |  2nd half: n={len(second_half)} PnL={sh_pnl:.4f} WR={sh_wr:.2f}"
                    )

                # Top losing symbols
                sym_pnl = df_sd.groupby("symbol")["pnl"].agg(["sum", "count"])
                sym_pnl = sym_pnl.sort_values("sum")
                worst_3 = sym_pnl.head(3)
                best_3 = sym_pnl.tail(3).iloc[::-1]
                w_parts = [
                    f"{s}({row['sum']:+.4f}, n={int(row['count'])})"
                    for s, row in worst_3.iterrows()
                ]
                b_parts = [
                    f"{s}({row['sum']:+.4f}, n={int(row['count'])})"
                    for s, row in best_3.iterrows()
                ]
                tprint(f"  Worst syms: {', '.join(w_parts)}")
                tprint(f"  Best syms:  {', '.join(b_parts)}")

        # --- MAE/MFE DIAGNOSTIC REPORT ---
        if "mae_pct" in df_t.columns and "mfe_pct" in df_t.columns:
            tprint("\n" + "=" * 70)
            tprint("MAE / MFE DIAGNOSTIC REPORT")
            tprint("=" * 70)

            # Global MAE/MFE
            tprint(f"\n--- Global MAE/MFE (n={len(df_t)}) ---")
            tprint(
                f"  MAE: mean={df_t['mae_pct'].mean()*100:.2f}%  med={df_t['mae_pct'].median()*100:.2f}%  q90={df_t['mae_pct'].quantile(0.9)*100:.2f}%"
            )
            tprint(
                f"  MFE: mean={df_t['mfe_pct'].mean()*100:.2f}%  med={df_t['mfe_pct'].median()*100:.2f}%  q90={df_t['mfe_pct'].quantile(0.9)*100:.2f}%"
            )
            tprint(
                f"  MFE/MAE ratio: {df_t['mfe_pct'].mean() / max(df_t['mae_pct'].mean(), 1e-9):.2f}"
            )

            # Per-bucket MAE/MFE
            bucket_col = "bucket" if "bucket" in df_t.columns else None
            if bucket_col is None:
                df_t["bucket"] = (
                    df_t["side"].str.upper() + "_" + df_t["dom"].str.upper()
                )
                bucket_col = "bucket"

            for bkt in sorted(df_t[bucket_col].unique()):
                df_b = df_t[df_t[bucket_col] == bkt]
                if len(df_b) < 3:
                    continue
                tprint(f"\n  --- {bkt} (n={len(df_b)}) ---")
                tprint(
                    f"    MAE: mean={df_b['mae_pct'].mean()*100:.2f}%  med={df_b['mae_pct'].median()*100:.2f}%"
                )
                tprint(
                    f"    MFE: mean={df_b['mfe_pct'].mean()*100:.2f}%  med={df_b['mfe_pct'].median()*100:.2f}%"
                )
                ratio = df_b["mfe_pct"].mean() / max(df_b["mae_pct"].mean(), 1e-9)
                tprint(
                    f"    MFE/MAE ratio: {ratio:.2f}  {'GOOD (>1.5)' if ratio > 1.5 else 'WEAK (<1.5) — entries or stops need work'}"
                )

                # MAE/MFE by exit reason
                for reason in sorted(df_b["reason"].dropna().unique()):
                    df_br = df_b[df_b["reason"] == reason]
                    if len(df_br) < 2:
                        continue
                    tprint(
                        f"    {reason}: n={len(df_br)}  MAE={df_br['mae_pct'].mean()*100:.2f}%  MFE={df_br['mfe_pct'].mean()*100:.2f}%  bars_to_mfe={df_br['bars_to_mfe'].mean():.0f}"
                    )

                # Key diagnostic: losers that had meaningful MFE (exit/stop problem)
                losers = df_b[df_b["pnl"] <= 0]
                if len(losers) > 0:
                    losers_with_mfe = losers[losers["mfe_pct"] > 0.005]  # >0.5% MFE
                    pct_losers_had_mfe = len(losers_with_mfe) / len(losers)
                    tprint(
                        f"    Losers with MFE>0.5%: {pct_losers_had_mfe:.0%} ({len(losers_with_mfe)}/{len(losers)})"
                        f"{'  *** EXIT PROBLEM: many losers saw profit first ***' if pct_losers_had_mfe > 0.4 else ''}"
                    )
                    if len(losers_with_mfe) > 0:
                        tprint(
                            f"      Avg MFE of those losers: {losers_with_mfe['mfe_pct'].mean()*100:.2f}%"
                        )

                # Key diagnostic: winners — how much MFE was captured?
                winners = df_b[df_b["pnl"] > 0]
                if len(winners) > 0:
                    capture_ratio = winners["gross_ret"].mean() / max(
                        winners["mfe_pct"].mean(), 1e-9
                    )
                    tprint(
                        f"    Winner capture ratio (ret/MFE): {capture_ratio:.2f}"
                        f"{'  *** LOW CAPTURE: trailing too loose ***' if capture_ratio < 0.3 else ''}"
                    )

        # --- PnL RECONCILIATION TABLE ---
        tprint("\n" + "=" * 70)
        tprint("PnL RECONCILIATION TABLE")
        tprint("=" * 70)

        # All units in portfolio-weighted PnL (pnl = net_ret * weight)
        total_gross_profit = float(df_t.loc[df_t["pnl"] > 0, "pnl"].sum())
        total_gross_loss = float(df_t.loc[df_t["pnl"] <= 0, "pnl"].sum())
        total_net_pnl = total_gross_profit + total_gross_loss
        recon_pf = (
            total_gross_profit / abs(total_gross_loss)
            if abs(total_gross_loss) > 1e-9
            else float("inf")
        )

        tprint(
            f"\n  Total Gross Profit:  {total_gross_profit:+.6f}  (portfolio-weighted PnL)"
        )
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
        tprint(
            f"  {'Bucket':<15} {'N':>5} {'GrossProfit':>12} {'GrossLoss':>12} {'NetPnL':>12} {'PF':>6} {'WR':>6} {'AvgWin':>10} {'AvgLoss':>10}"
        )
        for bkt in sorted(df_t[bucket_col].unique()):
            df_b = df_t[df_t[bucket_col] == bkt]
            b_gp = float(df_b.loc[df_b["pnl"] > 0, "pnl"].sum())
            b_gl = float(df_b.loc[df_b["pnl"] <= 0, "pnl"].sum())
            b_net = b_gp + b_gl
            b_pf = b_gp / abs(b_gl) if abs(b_gl) > 1e-9 else float("inf")
            b_wr = (df_b["pnl"] > 0).mean()
            b_aw = (
                float(df_b.loc[df_b["pnl"] > 0, "pnl"].mean())
                if (df_b["pnl"] > 0).any()
                else 0.0
            )
            b_al = (
                float(df_b.loc[df_b["pnl"] <= 0, "pnl"].mean())
                if (df_b["pnl"] <= 0).any()
                else 0.0
            )
            tprint(
                f"  {bkt:<15} {len(df_b):>5} {b_gp:>+12.6f} {b_gl:>+12.6f} {b_net:>+12.6f} {b_pf:>6.2f} {b_wr:>6.2f} {b_aw:>+10.6f} {b_al:>+10.6f}"
            )

        # Units sanity check
        avg_win_ret = (
            float(df_t.loc[df_t["pnl"] > 0, "ret"].mean())
            if (df_t["pnl"] > 0).any()
            else 0.0
        )
        avg_loss_ret = (
            float(df_t.loc[df_t["pnl"] <= 0, "ret"].mean())
            if (df_t["pnl"] <= 0).any()
            else 0.0
        )
        avg_weight = float(df_t["weight"].mean())
        tprint(f"\n  --- Units Check ---")
        tprint(f"  Avg Win (ret space):  {avg_win_ret:+.4f}")
        tprint(f"  Avg Loss (ret space): {avg_loss_ret:+.4f}")
        tprint(f"  Avg Weight:           {avg_weight:.4f}")
        tprint(
            f"  Implied AvgWin PnL:   {avg_win_ret * avg_weight:+.6f}  (should ~ match AvgWin above)"
        )
        tprint(
            f"  Implied AvgLoss PnL:  {avg_loss_ret * avg_weight:+.6f}  (should ~ match AvgLoss above)"
        )

        # --- Global exit reason breakdown ---
        if "reason" in df_t.columns:
            tprint("\n--- Exit Reasons (global) ---")
            for reason in sorted(df_t["reason"].dropna().unique()):
                df_r = df_t[df_t["reason"] == reason]
                r_wr = (df_r["pnl"] > 0).mean()
                r_hold = df_r["_hold_h"].mean()
                tprint(
                    f"  {reason}: n={len(df_r)} ({len(df_r)/len(df_t)*100:.0f}%), PnL={df_r['pnl'].sum():.4f}, WR={r_wr:.2f}, AvgHold={r_hold:.1f}h"
                )

        # Daily concentration check
        df_t["_date"] = df_t["_entry"].dt.date
        daily_counts = df_t.groupby("_date").size()
        tprint(
            f"\n--- Daily Concentration: max={daily_counts.max()}/day, mean={daily_counts.mean():.1f}/day ---"
        )
        if daily_counts.max() > 20:
            worst_day = daily_counts.idxmax()
            df_wd = df_t[df_t["_date"] == worst_day]
            tprint(
                f"  Worst day {worst_day}: {len(df_wd)} trades, PnL={df_wd['pnl'].sum():.4f}"
            )

        # Per-bucket daily concentration
        tprint("  Per-bucket daily max:")
        for bkt in sorted(df_t[bucket_col].unique()):
            df_b = df_t[df_t[bucket_col] == bkt]
            bkt_daily = df_b.groupby("_date").size()
            tprint(
                f"    {bkt}: max={bkt_daily.max()}/day, mean={bkt_daily.mean():.1f}/day"
            )

        # Weekly PnL trend
        df_t["_week"] = df_t["_entry"].dt.isocalendar().week.astype(int)
        weekly = df_t.groupby("_week").agg(
            n=("pnl", "count"), pnl=("pnl", "sum"), wr=("pnl", lambda x: (x > 0).mean())
        )
        tprint("--- Weekly PnL ---")
        for wk, row in weekly.iterrows():
            bar = "+" * int(max(0, row["pnl"]) * 500) + "-" * int(
                max(0, -row["pnl"]) * 500
            )
            tprint(
                f"  W{wk:02d}: n={int(row['n']):3d}  PnL={row['pnl']:+.4f}  WR={row['wr']:.2f}  {bar}"
            )

        df_t.drop(
            columns=["_date", "_entry", "_exit", "_hold_h", "_week", bucket_col],
            inplace=True,
            errors="ignore",
        )
        tprint("-----------------------")

    if test_trades:
        df_res = pd.DataFrame(test_trades)
        out_path = os.path.join(
            cfg["data_root"], "artifacts", run_id, "backtest_results.csv"
        )
        df_res.to_csv(out_path, index=False)
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
                base_dir=cfg.get("reports_root"),
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


def run_feature_generation_step(
    ts_sig, margin_symbols, cfg, store, force_full_recompute: bool = False
):
    tprint("STEP: FEATURE GENERATION START")
    tprint(f"Target Timestamp: {ts_sig}")

    # Check if feature files already exist for this timestamp (lightweight check only).
    ts_str = ts_sig.strftime("%Y%m%d_%H%M%S")
    feat_dir = os.path.join(cfg["data_root"], "features", ts_str)
    existing_files = sorted(glob.glob(os.path.join(feat_dir, "symbol=*.parquet")))
    expected_keys = _expected_feature_keys_from_cfg(cfg)
    backfill_keys: list[str] = []
    precomputed_tail_cutoffs: dict[str, pd.Timestamp] = {}
    tail_cutoff_stats: dict[str, int] | None = None
    close_panel_light: pd.DataFrame | None = None
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
        tprint(
            f"WARNING: get_training_universe failed ({exc}); falling back to local store symbol discovery"
        )
        train_syms = _local_store_symbols(store)
        if not train_syms:
            tprint("CRITICAL: no symbols available from local store fallback.")
            return
    # Universe diagnostics still logged below; downstream skip reasons are explicit.

    tprint(
        f"Universe (Top {cfg['fetch_symbols_M']} Vol + VarianceFilter): {len(train_syms)} symbols"
    )

    lookback_days = max(180, int(cfg["fetch_years"] * 365))

    # 2. Early cache-completeness check using close-only loads.
    stale_symbols_for_backfill: list[str] = []
    full_rewrite_symbols_for_backfill: set[str] = set()
    if existing_files and not force_full_recompute:
        (
            close_panel_light,
            loaded_close_syms,
            skipped_close,
        ) = _load_close_panel_for_symbols(
            store=store,
            symbols=train_syms,
            ts_sig=ts_sig,
            lookback_days=lookback_days,
        )
        tprint(
            f"Close-only cache precheck loaded {len(loaded_close_syms)} symbols. "
            f"Skipped {len(skipped_close)}."
        )
        if close_panel_light is not None and not close_panel_light.empty:
            scan = _scan_feature_cache_light(
                ts_sig=ts_sig,
                data_root=cfg["data_root"],
                expected_keys=expected_keys,
                panel_close=close_panel_light,
            )
            miss_keys = scan["missing_keys"] if scan else list(expected_keys)
            partial_keys = scan["partial_keys"] if scan else []
            backfill_keys = sorted(set(miss_keys + partial_keys))
            stale_symbols_for_backfill = (
                list(scan.get("stale_symbols", [])) if scan else []
            )
            full_rewrite_symbols_for_backfill = (
                set(scan.get("full_rewrite_symbols", [])) if scan else set()
            )
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
                    tprint(
                        "Missing keys: "
                        + ", ".join(miss_keys[:30])
                        + (" ..." if len(miss_keys) > 30 else "")
                    )
                if partial_keys:
                    tprint(
                        "Partial keys: "
                        + ", ".join(partial_keys[:30])
                        + (" ..." if len(partial_keys) > 30 else "")
                    )
                (
                    precomputed_tail_cutoffs,
                    tail_cutoff_stats,
                ) = _build_tail_only_backfill_cutoffs(
                    ts_sig=ts_sig,
                    data_root=cfg["data_root"],
                    panel_close=close_panel_light,
                    backfill_keys=backfill_keys,
                )
                tprint(
                    "Tail-only backfill cutoffs (preload): "
                    f"eligible={tail_cutoff_stats['eligible_tail_only']} "
                    f"missing_file={tail_cutoff_stats['missing_symbol_file']} "
                    f"missing_cols={tail_cutoff_stats['missing_backfill_columns']} "
                    f"structural_or_interior={tail_cutoff_stats['structural_or_interior']} "
                    f"already_covered={tail_cutoff_stats['already_covered']}"
                )
            else:
                _n_syms = len(close_panel_light.columns)
                _n_feats = len(expected_keys)
                tprint(
                    f"Features already exist and cover full target period: "
                    f"{_n_feats} features × {_n_syms} symbols. Skipping recomputation."
                )
                _generate_feature_health_reports(
                    ts_sig, cfg["data_root"], allowed_symbols=train_syms
                )
                tprint("STEP: FEATURE GENERATION COMPLETE (cached)")
                return
        else:
            backfill_keys = sorted(expected_keys)

    # 3. Load Data
    dfs = {}

    syms_to_load = list(train_syms)
    if backfill_keys and not force_full_recompute and stale_symbols_for_backfill:
        # Incremental backfill mode: only load symbols that need backfilling
        required_syms = sorted(
            set(stale_symbols_for_backfill).union(set(cfg["market_basket"]))
        )
        if required_syms:
            syms_to_load = [s for s in train_syms if s in set(required_syms)]
            tprint(
                f"Backfill mode: Loading {len(syms_to_load)}/{len(train_syms)} symbols "
                "(stale symbols + market basket)."
            )

    loaded_syms = []
    skipped_log = []

    with Timer("Feature Gen Data Load"):
        tail_compute_warmup_hours = int(
            cfg.get("feature_tail_compute_warmup_hours", 24 * 120)
        )
        for s in syms_to_load:
            df = store.load(s, end_ts=ts_sig)  # Load up to ts_sig

            # Constraints Check
            if df.empty:
                skipped_log.append(f"{s}: Empty DataFrame")
                continue

            # Check length (at least 60 days for basic moving averages + volatility)
            min_rows = 24 * 60
            if len(df) < min_rows:
                skipped_log.append(
                    f"{s}: Insufficient data ({len(df)} rows < {min_rows})"
                )
                continue

            # Check recent data freshness?
            last_ts = df.index[-1]
            if (ts_sig - last_ts).days > 180:
                skipped_log.append(
                    f"{s}: Stale data (Last: {last_ts}, Target: {ts_sig})"
                )
                continue

            df = df.tail(24 * lookback_days)
            cutoff_ts = precomputed_tail_cutoffs.get(s)
            if cutoff_ts is not None:
                warmup_start = pd.Timestamp(cutoff_ts) - pd.Timedelta(
                    hours=tail_compute_warmup_hours
                )
                # Ensure tz-aware comparison if df.index is tz-aware
                if df.index.tz is not None and warmup_start.tz is None:
                    warmup_start = warmup_start.tz_localize("UTC")
                before_rows = len(df)
                df = df[df.index > warmup_start]
                tprint(
                    f"  [TAIL-COMPUTE] {s}: rows {before_rows}->{len(df)} "
                    f"(cutoff={cutoff_ts}, warmup_h={tail_compute_warmup_hours})"
                )
            loaded_syms.append(s)
            dfs[s] = df

    tprint(f"Loaded {len(loaded_syms)} symbols. Skipped {len(skipped_log)}.")
    for msg in skipped_log:
        tprint(f"  [SKIP] {msg}")

    if not dfs:
        tprint("CRITICAL: No valid data found for feature generation.")
        return

    # 3. Compute Features (Panel)
    tprint("Constructing Panel...")
    panel = to_panel(dfs)
    panel_close_ref = panel["close"].copy() if "close" in panel else None
    panel_symbols = (
        list(panel["close"].columns) if "close" in panel else list(loaded_syms)
    )

    tprint("Computing Market Features...")
    mkt_df = compute_market_features(panel, cfg["market_basket"])
    mkt_gates = add_regime_gates(
        mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"]
    )

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

        backfill_set = set(backfill_keys) if backfill_keys else set()
        if backfill_keys and not force_full_recompute:
            tail_cutoffs = precomputed_tail_cutoffs
            tail_stats = tail_cutoff_stats or {
                "eligible_tail_only": len(tail_cutoffs),
                "missing_symbol_file": 0,
                "missing_backfill_columns": 0,
                "structural_or_interior": 0,
                "already_covered": 0,
            }
            tprint(
                "Tail-only backfill cutoffs: "
                f"eligible={tail_stats['eligible_tail_only']} "
                f"missing_file={tail_stats['missing_symbol_file']} "
                f"missing_cols={tail_stats['missing_backfill_columns']} "
                f"structural_or_interior={tail_stats['structural_or_interior']} "
                f"already_covered={tail_stats['already_covered']}"
            )
        else:
            tail_cutoffs = {}
            tprint("Chunked processing: Full recompute mode (no tail cutoffs)")

        all_syms = list(panel["close"].columns)
        total_chunks = (len(all_syms) + chunk_size - 1) // chunk_size
        unresolved_union: set[str] = set()
        total_saved_keys = 0
        tprint(
            f"Computing Asset Features (Hourly) in chunked backfill mode: "
            f"{len(all_syms)} symbols, chunk_size={chunk_size}, chunks={total_chunks}"
        )

        key_batch_size = int(cfg.get("feature_backfill_key_batch_size", 192))

        for ci, start in enumerate(range(0, len(all_syms), chunk_size), start=1):
            chunk_syms = all_syms[start : start + chunk_size]
            tprint(
                f"[Feature chunk {ci}/{total_chunks}] symbols={len(chunk_syms)} "
                f"({chunk_syms[0]} .. {chunk_syms[-1]})"
            )
            panel_chunk_source = {
                k: v.reindex(columns=chunk_syms).copy()
                for k, v in panel.items()
                if isinstance(v, pd.DataFrame)
            }
            chunk_requested_keys = None
            if backfill_keys and not force_full_recompute:
                chunk_requested_keys = _derive_symbol_backfill_keys(
                    ts_sig=ts_sig,
                    data_root=cfg["data_root"],
                    expected_keys=backfill_set,
                    symbols=chunk_syms,
                    full_rewrite_symbols=full_rewrite_symbols_for_backfill,
                )
                tprint(
                    f"[Feature chunk {ci}/{total_chunks}] requested_keys={len(chunk_requested_keys)}"
                )
            if backfill_keys and not force_full_recompute and chunk_requested_keys:
                key_batches = [
                    chunk_requested_keys[i : i + key_batch_size]
                    for i in range(0, len(chunk_requested_keys), key_batch_size)
                ]
            else:
                key_batches = [chunk_requested_keys]

            if backfill_keys and not force_full_recompute:
                chunk_cutoffs = {
                    s: tail_cutoffs[s] for s in chunk_syms if s in tail_cutoffs
                }
            else:
                chunk_cutoffs = None

            for bi, key_batch in enumerate(key_batches, start=1):
                batch_label = (
                    f"[Feature chunk {ci}/{total_chunks} batch {bi}/{len(key_batches)}]"
                    if len(key_batches) > 1
                    else f"[Feature chunk {ci}/{total_chunks}]"
                )
                if key_batch is not None:
                    tprint(f"{batch_label} computing requested_keys={len(key_batch)}")
                panel_chunk = {k: v.copy() for k, v in panel_chunk_source.items()}
                compute_t0 = time.perf_counter()
                feats_chunk, feat_index, feat_columns = compute_features_hourly(
                    panel_chunk,
                    mkt_gates.copy(),
                    cfg,
                    requested_feature_keys=key_batch,
                )
                compute_dt = time.perf_counter() - compute_t0

                if backfill_keys and not force_full_recompute:
                    batch_backfill_keys = key_batch or []
                    unresolved = _missing_requested_feature_keys(
                        feats_chunk,
                        batch_backfill_keys,
                    )
                    if unresolved:
                        unresolved_union.update(unresolved)
                    feats_chunk = {
                        k: v
                        for k, v in feats_chunk.items()
                        if k in set(batch_backfill_keys)
                    }
                else:
                    batch_backfill_keys = []

                batch_health_keys = (
                    tuple(
                        k
                        for k in FEATURE_HEALTH_CRITICAL_KEYS
                        if k in set(batch_backfill_keys)
                    )
                    if batch_backfill_keys
                    else FEATURE_HEALTH_CRITICAL_KEYS
                )
                chunk_health_issues = _feature_snapshot_health_issues(
                    feats_chunk,
                    critical_keys=batch_health_keys,
                )
                if chunk_health_issues:
                    raise RuntimeError(
                        "Feature chunk health validation failed before save: "
                        + ", ".join(chunk_health_issues)
                    )
                if batch_backfill_keys and unresolved:
                    raise RuntimeError(
                        "Feature chunk missing requested keys before save: "
                        + ", ".join(unresolved[:30])
                        + (" ..." if len(unresolved) > 30 else "")
                    )
                if batch_backfill_keys:
                    diagnostic_issues = _feature_quality_issues_for_keys(
                        feats_chunk,
                        batch_backfill_keys,
                    )
                    noncritical_diagnostic_issues = [
                        issue
                        for issue in diagnostic_issues
                        if str(issue).split(":", 1)[-1]
                        not in set(FEATURE_HEALTH_CRITICAL_KEYS)
                    ]
                    if noncritical_diagnostic_issues:
                        tprint(
                            f"{batch_label} diagnostic feature-quality issues "
                            f"({len(noncritical_diagnostic_issues)}): "
                            + ", ".join(noncritical_diagnostic_issues[:20])
                            + (
                                " ..."
                                if len(noncritical_diagnostic_issues) > 20
                                else ""
                            )
                        )

                tprint(f"{batch_label} saving {len(feats_chunk)} keys")
                if feats_chunk:
                    save_t0 = time.perf_counter()
                    save_features(
                        feats_chunk,
                        ts_sig,
                        cfg["data_root"],
                        min_timestamp_by_symbol=chunk_cutoffs
                        if chunk_cutoffs
                        else None,
                        feat_index=feat_index,
                        feat_columns=feat_columns,
                        save_workers=int(cfg.get("feature_save_workers", 2)),
                    )
                    save_dt = time.perf_counter() - save_t0
                    tprint(
                        f"{batch_label} timings: compute={compute_dt:.1f}s save={save_dt:.1f}s"
                    )
                    total_saved_keys += len(feats_chunk)
                del panel_chunk, feats_chunk
                gc.collect()

            del panel_chunk_source
            gc.collect()

        tprint(f"Computed + saved chunked backfill features: {total_saved_keys} keys")
        if unresolved_union:
            unresolved_sorted = sorted(unresolved_union)
            raise RuntimeError(
                f"Incremental feature backfill could not produce {len(unresolved_sorted)} "
                "requested keys: "
                + ", ".join(unresolved_sorted[:30])
                + (" ..." if len(unresolved_sorted) > 30 else "")
            )
    else:
        tprint("Computing Asset Features (Hourly)...")
        requested_feature_keys = None
        if backfill_keys and not force_full_recompute:
            requested_feature_keys = _derive_symbol_backfill_keys(
                ts_sig=ts_sig,
                data_root=cfg["data_root"],
                expected_keys=set(backfill_keys),
                symbols=panel_symbols,
                full_rewrite_symbols=full_rewrite_symbols_for_backfill,
            )
            tprint(
                f"Incremental feature compute requested_keys={len(requested_feature_keys)}"
            )
        compute_t0 = time.perf_counter()
        feats, feat_index, feat_columns = compute_features_hourly(
            panel,
            mkt_gates,
            cfg,
            requested_feature_keys=requested_feature_keys,
        )
        compute_dt = time.perf_counter() - compute_t0
        min_ts_by_symbol = None

        full_health_issues = _feature_snapshot_health_issues(feats)
        if full_health_issues:
            raise RuntimeError(
                "Feature snapshot health validation failed before save: "
                + ", ".join(full_health_issues)
            )

        if backfill_keys and not force_full_recompute:
            backfill_set = set(backfill_keys)
            selected_backfill_keys = requested_feature_keys or []
            unresolved = sorted([k for k in selected_backfill_keys if k not in feats])
            feats = {k: v for k, v in feats.items() if k in set(selected_backfill_keys)}
            min_ts_by_symbol = precomputed_tail_cutoffs
            tail_stats = tail_cutoff_stats or {
                "eligible_tail_only": len(min_ts_by_symbol),
                "missing_symbol_file": 0,
                "missing_backfill_columns": 0,
                "structural_or_interior": 0,
                "already_covered": 0,
            }
            tprint(
                f"Computed + saving only missing/partial features: {len(feats)} keys"
            )
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
                    f"WARNING: Backfill could not produce {len(unresolved)} keys "
                    "(may require other pipelines): "
                    + ", ".join(unresolved[:30])
                    + (" ..." if len(unresolved) > 30 else "")
                )
        elif force_full_recompute:
            tprint(f"Computed + saving full feature set: {len(feats)} keys")

        # 4. Save
        if feats:
            save_t0 = time.perf_counter()
            save_features(
                feats,
                ts_sig,
                cfg["data_root"],
                min_timestamp_by_symbol=min_ts_by_symbol,
                feat_index=feat_index,
                feat_columns=feat_columns,
                save_workers=int(cfg.get("feature_save_workers", 2)),
            )
            tprint(
                f"Feature save timing: compute={compute_dt:.1f}s save={time.perf_counter() - save_t0:.1f}s"
            )
        else:
            tprint(
                "No feature keys selected for save after missing/new filter; nothing to write."
            )

    tprint(f"Generated features for {len(loaded_syms)} symbols.")
    # Incremental backfill should not synthesize placeholder columns, but it still
    # must prove the persisted snapshot is complete when it finishes.
    is_incremental_backfill = backfill_keys and not force_full_recompute
    completeness_panel = (
        close_panel_light
        if close_panel_light is not None and not close_panel_light.empty
        else panel_close_ref
    )
    snapshot_expected_keys = {
        k for k in expected_keys if not str(k).startswith("oof_")
    }
    skip_snapshot_validation = bool(cfg.get("skip_feature_snapshot_validation", False))
    skip_postsave_checks = bool(cfg.get("skip_feature_postsave_checks", False))
    if skip_snapshot_validation or skip_postsave_checks:
        tprint("Feature snapshot post-save checks skipped")
    else:
        if not is_incremental_backfill:
            _enforce_feature_snapshot_completeness(
                ts_sig=ts_sig,
                data_root=cfg["data_root"],
                expected_keys=snapshot_expected_keys,
                panel_close=completeness_panel,
            )
        else:
            tprint(
                "Incremental backfill mode: skipping placeholder completeness normalization "
                "and validating persisted completeness directly"
            )
        _validate_feature_snapshot_completeness(
            ts_sig=ts_sig,
            data_root=cfg["data_root"],
            expected_keys=snapshot_expected_keys,
            panel_close=completeness_panel,
        )
        _generate_feature_health_reports(
            ts_sig, cfg["data_root"], allowed_symbols=train_syms
        )
        health_feats = load_features_for_stage_or_all(cfg, ts_sig, cfg["data_root"],
            feature_keys=FEATURE_HEALTH_CRITICAL_KEYS,
            symbols=train_syms,
        )
        if health_feats is None:
            raise RuntimeError(
                "Feature snapshot health validation failed: critical feature files "
                "could not be loaded after generation."
            )
        health_issues = _feature_snapshot_health_issues(health_feats)
        if health_issues:
            raise RuntimeError(
                "Feature snapshot health validation failed after generation: "
                + ", ".join(health_issues)
                + ". Re-run features mode with --force-feature-recompute after fixing "
                "the underlying feature channel."
            )
    tprint("STEP: FEATURE GENERATION COMPLETE")


def inject_features_into_datasets(datasets, ts_sig, cfg, req_keys):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import time

    from extreme_price_movements.data_store import get_feature_path
    from extreme_price_movements.utils import tprint
    if not datasets or not req_keys:
        return datasets

    req_keys = sorted(
        {
            str(k)
            for k in req_keys
            if isinstance(k, str)
            and k
            and not k.startswith("__")
            and not k.startswith("pred_")
        }
    )
    if not req_keys:
        return datasets

    all_syms: set[str] = set()
    dataset_meta: dict[str, dict[str, object]] = {}
    missing_keys_per_dataset: dict[str, list[str]] = {}
    requested_keys_per_dataset: dict[str, list[str]] = {}
    datasets_by_symbol: dict[str, list[str]] = {}

    for name, df in datasets.items():
        if df is None or df.empty:
            continue
        s_col = (
            "__symbol__"
            if "__symbol__" in df.columns
            else ("symbol" if "symbol" in df.columns else None)
        )
        t_col = (
            "__ts__"
            if "__ts__" in df.columns
            else ("ts" if "ts" in df.columns else None)
        )
        if not s_col or not t_col:
            continue
        dataset_meta[name] = {"df": df, "s_col": s_col, "t_col": t_col}
        missing = [k for k in req_keys if k not in df.columns]
        requested_keys_per_dataset[name] = list(missing)
        if not missing:
            continue
        missing_keys_per_dataset[name] = list(missing)
        syms = pd.Index(df[s_col].dropna().astype(str).unique())
        for sym in syms:
            all_syms.add(str(sym))
            datasets_by_symbol.setdefault(str(sym), []).append(name)

    if not missing_keys_per_dataset:
        tprint("Feature injection: all requested features already present in datasets.")
        return datasets

    meta_keys_all = set(_meta_feature_keys_union(cfg))
    progress_every = max(
        1, int(cfg.get("feature_injection_progress_every_symbols", 25))
    )
    progress_interval_sec = float(
        cfg.get("feature_injection_progress_interval_sec", 60.0)
    )
    start_time = time.time()
    last_progress_time = start_time

    dataset_ts_index: dict[str, pd.DatetimeIndex] = {}
    dataset_symbol_positions: dict[str, dict[str, np.ndarray]] = {}
    target_buffers: dict[str, dict[str, np.ndarray]] = {}
    injection_stats: dict[str, dict[str, object]] = {}

    for name, meta in dataset_meta.items():
        df = meta["df"]
        s_col = meta["s_col"]
        t_col = meta["t_col"]
        missing = list(missing_keys_per_dataset.get(name, []))
        injection_stats[name] = {
            "requested_missing": list(requested_keys_per_dataset.get(name, [])),
            "loaded_keys": set(),
            "missing_after": set(),
            "n_rows": int(len(df)),
        }
        if not missing:
            continue
        ts_index = pd.DatetimeIndex(pd.to_datetime(df[t_col], utc=True, errors="coerce"))
        dataset_ts_index[name] = ts_index
        symbol_positions = (
            pd.Series(np.arange(len(df), dtype=np.int32))
            .groupby(df[s_col].astype(str).to_numpy(copy=False), sort=False)
            .indices
        )
        dataset_symbol_positions[name] = {
            str(sym): np.asarray(pos, dtype=np.int32)
            for sym, pos in symbol_positions.items()
        }
        buf: dict[str, np.ndarray] = {}
        for key in missing:
            buf[key] = np.full(len(df), np.nan, dtype=np.float32)
            if key in meta_keys_all:
                buf[f"__meta_raw__{key}"] = np.full(len(df), np.nan, dtype=np.float32)
        target_buffers[name] = buf

    loaded_symbols = 0
    injected_cells = 0
    total_symbols = int(len(all_syms))
    tprint(
        f"Feature injection: starting symbol pass for {total_symbols} symbols across "
        f"{len(missing_keys_per_dataset)} datasets; req_keys={len(req_keys)}"
    )

    for sym_i, sym in enumerate(sorted(all_syms), start=1):
        now = time.time()
        if (
            sym_i == 1
            or sym_i % progress_every == 0
            or (now - last_progress_time) >= progress_interval_sec
        ):
            elapsed = now - start_time
            tprint(
                f"Feature injection progress: symbols={sym_i}/{total_symbols} "
                f"loaded_symbol_files={loaded_symbols} injected_cells={injected_cells} "
                f"elapsed_sec={elapsed:.1f} current_symbol={sym}"
            )
            last_progress_time = now

        target_datasets = [
            name
            for name in datasets_by_symbol.get(sym, [])
            if name in missing_keys_per_dataset
        ]
        if not target_datasets:
            continue

        keys_for_symbol = sorted(
            {
                key
                for name in target_datasets
                for key in missing_keys_per_dataset.get(name, [])
            }
        )
        if not keys_for_symbol:
            continue

        fpath = get_feature_path(cfg["data_root"], ts_sig, sym)
        if not os.path.exists(fpath):
            continue

        try:
            schema_cols = set(pq.ParquetFile(fpath).schema.names)
        except Exception as exc:
            tprint(f"Feature injection: failed to inspect {fpath}: {exc}")
            continue

        load_cols = [k for k in keys_for_symbol if k in schema_cols]
        ts_in_schema = "ts" in schema_cols or "timestamp" in schema_cols
        if not load_cols:
            continue

        try:
            read_cols = list(load_cols)
            if "ts" in schema_cols:
                read_cols = ["ts", *read_cols]
            elif "timestamp" in schema_cols:
                read_cols = ["timestamp", *read_cols]
            feat_df = pd.read_parquet(fpath, columns=read_cols)
        except Exception as exc:
            tprint(f"Feature injection: failed to load {fpath}: {exc}")
            continue

        if feat_df.empty:
            continue

        ts_series = None
        if "ts" in feat_df.columns:
            ts_series = pd.to_datetime(feat_df["ts"], utc=True, errors="coerce")
            feat_df = feat_df.drop(columns=["ts"], errors="ignore")
        elif "timestamp" in feat_df.columns:
            ts_series = pd.to_datetime(feat_df["timestamp"], utc=True, errors="coerce")
            feat_df = feat_df.drop(columns=["timestamp"], errors="ignore")
        elif isinstance(feat_df.index, pd.DatetimeIndex):
            ts_series = pd.to_datetime(feat_df.index, utc=True, errors="coerce")
        elif ts_in_schema:
            tprint(
                f"Feature injection: timestamp column was not materialized from {fpath}; "
                "falling back to parquet index."
            )
            ts_series = pd.to_datetime(feat_df.index, utc=True, errors="coerce")
        else:
            ts_series = pd.to_datetime(feat_df.index, utc=True, errors="coerce")

        feat_df = feat_df.assign(__ts__=ts_series)
        feat_df = feat_df.dropna(subset=["__ts__"]).drop_duplicates(
            subset=["__ts__"], keep="last"
        )
        if feat_df.empty:
            continue
        feat_df = feat_df.set_index("__ts__").sort_index(kind="mergesort")

        feature_arrays = {
            col: pd.to_numeric(feat_df[col], errors="coerce")
            .astype(np.float32, copy=False)
            .to_numpy(copy=False)
            for col in load_cols
        }
        feat_index = feat_df.index
        loaded_symbols += 1

        for name in target_datasets:
            row_pos = dataset_symbol_positions.get(name, {}).get(sym)
            if row_pos is None or row_pos.size == 0:
                continue
            needed = [
                k for k in missing_keys_per_dataset.get(name, []) if k in feature_arrays
            ]
            if not needed:
                continue
            ts_vals = dataset_ts_index[name].take(row_pos)
            feat_row_pos = feat_index.get_indexer(ts_vals)
            valid = feat_row_pos >= 0
            if not np.any(valid):
                continue
            buffer_pos = row_pos[valid]
            feat_row_pos = feat_row_pos[valid]
            name_buffers = target_buffers[name]
            loaded_keys = injection_stats[name]["loaded_keys"]
            for key in needed:
                vals = feature_arrays[key][feat_row_pos]
                name_buffers[key][buffer_pos] = vals
                loaded_keys.add(key)
                if key in meta_keys_all:
                    raw_key = f"__meta_raw__{key}"
                    if raw_key in name_buffers:
                        name_buffers[raw_key][buffer_pos] = vals
                injected_cells += int(len(buffer_pos))

    for name, cols_dict in target_buffers.items():
        if not cols_dict:
            continue
        df = datasets[name]
        add_df = pd.DataFrame(cols_dict, index=df.index)
        datasets[name] = pd.concat([df, add_df], axis=1, copy=False)
        dataset_meta[name]["df"] = datasets[name]

    variant_base_groups: dict[str, dict[str, set[str]]] = {}
    for name, stat in injection_stats.items():
        req_missing = set(stat.get("requested_missing", []))
        loaded = set(stat.get("loaded_keys", set()))
        missing_after = req_missing - loaded
        stat["missing_after"] = missing_after
        tprint(
            f"Feature injection [{name}]: rows={stat.get('n_rows', 0)} "
            f"requested_missing={len(req_missing)} loaded={len(loaded)} "
            f"missing_after={len(missing_after)}"
        )
        if missing_after:
            tprint(
                f"  Feature injection [{name}] missing_after sample: "
                f"{sorted(list(missing_after))[:12]}"
            )
        base_name = name
        variant_tag = "primary"
        if name.endswith("_tight"):
            base_name = name[: -len("_tight")]
            variant_tag = "tight"
        elif name.endswith("_wide"):
            base_name = name[: -len("_wide")]
            variant_tag = "wide"
        variant_base_groups.setdefault(base_name, {})[variant_tag] = loaded

    for base_name, grp in sorted(variant_base_groups.items()):
        primary = grp.get("primary")
        if primary is None:
            continue
        for variant_tag in ("tight", "wide"):
            if variant_tag not in grp:
                continue
            variant_keys = grp.get(variant_tag, set())
            missing_vs_primary = sorted(list(primary - variant_keys))
            extra_vs_primary = sorted(list(variant_keys - primary))
            tprint(
                f"Feature injection parity [{base_name} vs {variant_tag}]: "
                f"primary={len(primary)} variant={len(variant_keys)} "
                f"missing_vs_primary={len(missing_vs_primary)} "
                f"extra_vs_primary={len(extra_vs_primary)}"
            )
            if missing_vs_primary:
                tprint(
                    f"  Parity missing [{base_name}/{variant_tag}] sample: "
                    f"{missing_vs_primary[:12]}"
                )

    elapsed_total = time.time() - start_time
    tprint(
        f"Feature injection: loaded {loaded_symbols} symbol files and injected requested "
        f"features into {len(missing_keys_per_dataset)} datasets in {elapsed_total:.1f}s."
    )
    return datasets

def _filter_artifact_by_stage_view(df, cfg):
    view = cfg.get("_active_stage_view")
    if not view or df is None or df.empty:
        return df

    orig_len = len(df)
    stage_name = view.get("stage_name", "unknown_stage")

    if "symbols" in view and view["symbols"] is not None:
        sym_col = "__symbol__" if "__symbol__" in df.columns else "symbol" if "symbol" in df.columns else None
        if sym_col:
            df = df[df[sym_col].isin(view["symbols"])]

    periods = view.get("allowed_periods") or None
    if periods:
        ts_col = "__ts__" if "__ts__" in df.columns else "timestamp" if "timestamp" in df.columns else "t0" if "t0" in df.columns else None
        if ts_col:
            df_ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            mask = np.zeros(len(df), dtype=bool)
            for period in periods:
                if isinstance(period, dict):
                    p_start = period.get("start_ts") or period.get("start")
                    p_end = period.get("end_ts") or period.get("end")
                elif isinstance(period, (list, tuple)) and len(period) >= 2:
                    p_start, p_end = period[0], period[1]
                else:
                    continue
                p_start = pd.to_datetime(p_start, utc=True, errors="coerce")
                p_end = pd.to_datetime(p_end, utc=True, errors="coerce")
                if pd.isna(p_start) or pd.isna(p_end) or p_end <= p_start:
                    continue
                mask |= (df_ts >= p_start) & (df_ts < p_end)
            df = df.loc[mask]

    if view.get("allowed_start_ts") or view.get("allowed_end_ts"):
        ts_col = "__ts__" if "__ts__" in df.columns else "timestamp" if "timestamp" in df.columns else "t0" if "t0" in df.columns else None
        if ts_col:
            df_ts = pd.to_datetime(df[ts_col], utc=True)
            if view.get("allowed_start_ts"):
                df = df[df_ts >= pd.to_datetime(view["allowed_start_ts"])]
                df_ts = pd.to_datetime(df[ts_col], utc=True)
            if view.get("allowed_end_ts"):
                df = df[df_ts <= pd.to_datetime(view["allowed_end_ts"])]

    new_len = len(df)
    if orig_len != new_len:
        tprint(f"[{stage_name}] Filtered artifact: {orig_len} -> {new_len} rows (-{orig_len - new_len}) based on active stage limits.")

    return df
    if "symbols" in view and view["symbols"] is not None:
        sym_col = "__symbol__" if "__symbol__" in df.columns else "symbol" if "symbol" in df.columns else None
        if sym_col:
            df = df[df[sym_col].isin(view["symbols"])]
    if view.get("allowed_start_ts") or view.get("allowed_end_ts"):
        ts_col = "__ts__" if "__ts__" in df.columns else "timestamp" if "timestamp" in df.columns else "t0" if "t0" in df.columns else None
        if ts_col:
            df_ts = pd.to_datetime(df[ts_col], utc=True)
            if view.get("allowed_start_ts"):
                df = df[df_ts >= pd.to_datetime(view["allowed_start_ts"])]
                df_ts = pd.to_datetime(df[ts_col], utc=True)
            if view.get("allowed_end_ts"):
                df = df[df_ts <= pd.to_datetime(view["allowed_end_ts"])]
    return df


    tprint("Resolving unique symbols and timestamps for feature injection...")
    all_syms = set()
    for name, df in datasets.items():
        s_col = (
            "symbol"
            if "symbol" in df.columns
            else ("__symbol__" if "__symbol__" in df.columns else None)
        )
        if s_col:
            all_syms.update(df[s_col].unique())

    if not all_syms:
        tprint("No symbols found in datasets for injection.")
        return datasets

    sorted_syms = sorted(all_syms)
    max_symbols = int(cfg.get("feature_injection_max_symbols", 75))
    if max_symbols > 0 and len(sorted_syms) > max_symbols:
        tprint(
            f"Capping feature injection to the first {max_symbols} symbols "
            f"out of {len(sorted_syms)}."
        )
        sorted_syms = sorted_syms[:max_symbols]

    # Precompute dataset metadata so the symbol loop can stay lazy and avoid
    # repeating expensive per-row scans across all datasets.
    dataset_meta = {}
    datasets_by_symbol = {}
    for name, df in datasets.items():
        s_col = (
            "symbol"
            if "symbol" in df.columns
            else ("__symbol__" if "__symbol__" in df.columns else None)
        )
        t_col = (
            "ts"
            if "ts" in df.columns
            else ("__ts__" if "__ts__" in df.columns else None)
        )
        dataset_meta[name] = {"df": df, "s_col": s_col, "t_col": t_col}
        if not s_col:
            continue
        syms = pd.Index(df[s_col].dropna().unique())
        for sym in syms:
            datasets_by_symbol.setdefault(sym, []).append(name)

    # Define per-dataset feature requirements.
    dataset_features = {}
    for name in datasets.keys():
        if name == "trap_model":
            dataset_features[name] = set(TRAP_FEATURE_KEYS)
        elif name == "gamma_model":
            dataset_features[name] = set(GAMMA_FEATURE_KEYS)
        else:
            dataset_features[name] = set(req_keys)

    # Identify which keys are actually missing and needed for each dataset
    missing_keys_per_dataset = {}
    all_needed_keys = set()
    for name, df in datasets.items():
        missing = [k for k in dataset_features[name] if k not in df.columns]
        if missing:
            missing_keys_per_dataset[name] = missing
            all_needed_keys.update(missing)

    if not all_needed_keys:
        tprint("No features need injection (all already present).")
        return datasets

    meta_keys_all = set(_meta_feature_keys_union(cfg))
    heartbeat_every = max(1, int(cfg.get("feature_injection_heartbeat_every", 50)))

    import time
    import warnings

    from pandas.errors import PerformanceWarning

    start_time = time.time()

    tprint(f"Precomputing row mappings for {len(sorted_syms)} symbols...")
    symbol_mappings = {}

    # Precompute dataset_ts_index for timezone logic
    dataset_ts_index: dict[str, pd.DatetimeIndex] = {}
    dataset_symbol_positions: dict[str, dict[str, np.ndarray]] = {}
    for name, meta in dataset_meta.items():
        df = meta.get("df")
        s_col = meta.get("s_col")
        t_col = meta.get("t_col")
        if df is None or not s_col or not t_col:
            continue
        symbol_values = df[s_col].to_numpy(copy=False)
        dataset_ts_index[name] = pd.DatetimeIndex(
            pd.to_datetime(df[t_col], utc=True, errors="coerce")
        )
        symbol_positions = (
            pd.Series(np.arange(len(df), dtype=np.int32))
            .groupby(symbol_values, sort=False)
            .indices
        )
        dataset_symbol_positions[name] = {
            s: np.asarray(pos, dtype=np.int32)
            for s, pos in symbol_positions.items()
            if s in datasets_by_symbol and name in datasets_by_symbol.get(s, [])
        }

    for i, s in enumerate(sorted_syms, 1):
        fpath = get_feature_path(cfg["data_root"], ts_sig, s)
        if not os.path.exists(fpath):
            continue

        try:
            df_index = pd.read_parquet(fpath, columns=[])
            schema_cols = set(pq.ParquetFile(fpath).schema.names)
        except Exception:
            try:
                pq_file = pq.ParquetFile(fpath)
                schema_cols = set(pq_file.schema.names)
                if not schema_cols:
                    continue
                df_index = pd.read_parquet(fpath, columns=[list(schema_cols)[0]])
            except Exception:
                continue

        if df_index.empty:
            continue

        keep_mask = ~df_index.index.duplicated(keep="last")
        idx_feat = df_index.index[keep_mask]

        dataset_mappings = {}
        touched_datasets = datasets_by_symbol.get(s, [])
        for name in touched_datasets:
            row_pos = dataset_symbol_positions.get(name, {}).get(s)
            if row_pos is None or row_pos.size == 0:
                continue

            row_idx = dataset_ts_index[name].take(row_pos)

            idx_feat_tmp = idx_feat
            if hasattr(row_idx, "tz") and hasattr(idx_feat_tmp, "tz"):
                if row_idx.tz is None and idx_feat_tmp.tz is not None:
                    row_idx = pd.DatetimeIndex(
                        pd.to_datetime(row_idx, utc=True)
                    ).tz_convert(idx_feat_tmp.tz)
                elif row_idx.tz is not None and idx_feat_tmp.tz is None:
                    idx_feat_tmp = idx_feat_tmp.tz_localize("UTC").tz_convert(
                        row_idx.tz
                    )
                elif row_idx.tz is not None and idx_feat_tmp.tz is not None:
                    row_idx = row_idx.tz_convert(idx_feat_tmp.tz)

            feat_row_pos = idx_feat_tmp.get_indexer(row_idx)
            valid = feat_row_pos >= 0
            if not np.any(valid):
                continue

            dataset_mappings[name] = {
                "buffer_pos": row_pos[valid],
                "feat_row_pos": feat_row_pos[valid],
            }

        if dataset_mappings:
            symbol_mappings[s] = {
                "schema_cols": schema_cols,
                "mappings": dataset_mappings,
                "fpath": fpath,
                "keep_mask": keep_mask.to_numpy(),
            }

    all_keys_list = list(all_needed_keys)
    batch_size = max(1, int(cfg.get("feature_injection_batch_size", 50)))

    tprint(
        f"Injecting {len(all_keys_list)} features in batches of {batch_size} using PyArrow..."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PerformanceWarning)

        for i in range(0, len(all_keys_list), batch_size):
            batch_keys = all_keys_list[i : i + batch_size]
            batch_keys_set = set(batch_keys)

            # 1. Allocate temporary buffer for just this batch
            target_buffers = {}
            for name, missing in missing_keys_per_dataset.items():
                batch_missing = [k for k in missing if k in batch_keys_set]
                if batch_missing:
                    target_buffers[name] = {}
                    df_len = len(datasets[name])
                    for k in batch_missing:
                        target_buffers[name][k] = np.full(
                            df_len, np.nan, dtype=np.float32
                        )
                        if k in meta_keys_all:
                            target_buffers[name][f"__meta_raw__{k}"] = np.full(
                                df_len, np.nan, dtype=np.float32
                            )

            if not target_buffers:
                continue

            # 2. Extract data for this batch from each symbol
            for s, s_data in symbol_mappings.items():
                schema_cols = s_data["schema_cols"]
                fpath = s_data["fpath"]
                mappings = s_data["mappings"]
                keep_mask = s_data["keep_mask"]

                cols_to_load = [k for k in batch_keys if k in schema_cols]
                if not cols_to_load:
                    continue

                relevant_datasets = []
                for name, mapping in mappings.items():
                    if name in target_buffers:
                        ds_missing = [
                            k for k in cols_to_load if k in target_buffers[name]
                        ]
                        if ds_missing:
                            relevant_datasets.append((name, mapping, ds_missing))

                if not relevant_datasets:
                    continue

                try:
                    table = pq.read_table(fpath, columns=cols_to_load)
                except Exception as e:
                    tprint(f"  WARNING: Error reading {fpath}: {e}")
                    continue

                # Extract and deduplicate columns using pyarrow
                col_arrays = {}
                for k in cols_to_load:
                    try:
                        arr = table.column(k).to_numpy()
                        if arr.dtype != np.float32:
                            arr = arr.astype(np.float32)
                        col_arrays[k] = arr[keep_mask]
                    except Exception as e:
                        # Fallback if column cannot be easily cast or read
                        col_arrays[k] = None

                for name, mapping, ds_missing in relevant_datasets:
                    buffer_pos = mapping["buffer_pos"]
                    feat_row_pos = mapping["feat_row_pos"]

                    for k in ds_missing:
                        arr_dedup = col_arrays.get(k)
                        if arr_dedup is not None:
                            vals = arr_dedup[feat_row_pos]
                            target_buffers[name][k][buffer_pos] = vals
                            if (
                                k in meta_keys_all
                                and f"__meta_raw__{k}" in target_buffers[name]
                            ):
                                target_buffers[name][f"__meta_raw__{k}"][
                                    buffer_pos
                                ] = vals

            # 3. Inject batch columns directly into datasets to avoid memory spike
            for name, cols_dict in target_buffers.items():
                if not cols_dict:
                    continue
                df = datasets[name]
                add_df = pd.DataFrame(cols_dict, index=df.index)
                if len(add_df.columns) == len(set(add_df.columns)):
                    overlap_cols = [c for c in add_df.columns if c in df.columns]
                    if overlap_cols:
                        df = df.drop(columns=overlap_cols, errors="ignore")
                df = pd.concat([df, add_df], axis=1)
                datasets[name] = df
                if name in dataset_meta:
                    dataset_meta[name]["df"] = df

            elapsed = time.time() - start_time
            tprint(
                f"  Completed batch {i//batch_size + 1}/{(len(all_keys_list) + batch_size - 1)//batch_size} (elapsed={elapsed:.1f}s)"
            )
            import gc

            gc.collect()

    tprint("Feature injection complete.")
    import gc

    gc.collect()

    return datasets
