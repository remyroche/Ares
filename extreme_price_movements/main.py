import os
import sys
import time
import uuid
import re
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.candidates import (
    detect_extreme_movement_candidates,
    entry_price_next_hour_open,
    select_trade_candidates_hourly,
)
from extreme_price_movements.config import CFG
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    check_data_health,
    get_feature_bounds,
    get_feature_path,
    load_artifact_df,
    load_features,
    make_spot_exchange,
    save_features,
    to_panel,
)
from extreme_price_movements.engine import generate_hourly_signals
from extreme_price_movements.entry_policy import (
    compute_entry_policy_decision,
    flatten_bucket_policy,
)
from extreme_price_movements.features import (
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.metrics import MetricsLogger
from extreme_price_movements.model_loader import (
    find_latest_run_id,
    load_bucket_params,
    load_full_state,
    load_model_bundle,
)
from extreme_price_movements.optimization_utils import filter_low_variance_assets
from extreme_price_movements.pipeline_steps import (
    run_backtest_step,
    run_label_generation_step_v2,
    run_risk_optimization_step,
)
from extreme_price_movements.risk import TrailingStop
from extreme_price_movements.state import StateManager
from extreme_price_movements.strategy_registry import (
    get_strategies,
    strategy_runtime_horizons,
)
from extreme_price_movements.time_utils import floor_to_hour, get_ts_sig, now_utc
from extreme_price_movements.training import (
    apply_interaction_toggles,
    optimize_risk_params,
    select_best_horizon,
    train_models_from_artifacts,
)
from extreme_price_movements.universe import (
    build_fetch_universe,
    get_training_universe,
    refresh_margin_universe_daily,
    select_live_candidates,
)
from extreme_price_movements.utils import Timer, tprint


def reconcile_state(ex, state):
    tprint("Reconciling state...")
    return True


def _get_tradeable_basket(state, ts_sig):
    basket = state.state.get("tradeable_basket", {})
    valid = {}
    for sym, expiry in basket.items():
        try:
            if pd.Timestamp(expiry) >= ts_sig:
                valid[sym] = expiry
        except Exception:
            continue
    state.state["tradeable_basket"] = valid
    return set(valid.keys())


def _update_tradeable_basket(state, ts_sig, symbols, ttl_hours):
    basket = state.state.get("tradeable_basket", {})
    expiry = (ts_sig + pd.Timedelta(hours=int(ttl_hours))).isoformat()
    for sym in symbols:
        basket[sym] = expiry
    state.state["tradeable_basket"] = basket
    state.save()


def _monitor_active_positions_5m(ex, state, logger):
    now_ts = now_utc()
    positions = state.get_positions()
    for sym, pos in positions.items():
        last_check = pos.get("last_5m_check_ts")
        if last_check and (now_ts - pd.Timestamp(last_check)) < pd.Timedelta(minutes=5):
            continue
        try:
            ticker = ex.fetch_ticker(sym)
            px = float(ticker.get("last") or ticker.get("close") or np.nan)
            if not np.isfinite(px) or px <= 0:
                continue
            ts_risk = TrailingStop.from_dict(pos["risk_state"])
            stopped, exit_px, reason = ts_risk.update(px, px, px)
            if stopped:
                entry_px = pos["entry_px"]
                side = pos["side"]
                ret = (
                    (exit_px / entry_px - 1.0)
                    if side == "long"
                    else (entry_px / exit_px - 1.0)
                )
                logger.log(
                    now_ts,
                    {
                        "event": "exit",
                        "symbol": sym,
                        "return": ret,
                        "reason": f"5m_{reason}",
                    },
                )
                state.clear_position(sym)
            else:
                pos["risk_state"] = ts_risk.to_dict()
                pos["last_5m_check_ts"] = now_ts.isoformat()
                state.set_position(sym, pos)
        except Exception:
            continue


def generate_features_daily(ts_sig, margin_symbols, cfg, store, ex):
    tprint("DAILY FEATURE GENERATION START")
    train_syms = get_training_universe(margin_symbols, cfg, store, ts_sig=ts_sig)
    tprint(f"Target universe size: {len(train_syms)}")

    pending_syms = []
    last_ts_by_symbol: dict[str, pd.Timestamp] = {}

    for s in train_syms:
        fpath = get_feature_path(cfg["data_root"], ts_sig, s)
        if not os.path.exists(fpath):
            pending_syms.append(s)
            continue

        _, last_ts = get_feature_bounds(fpath)
        if last_ts is None or last_ts < ts_sig:
            pending_syms.append(s)
            if last_ts is not None:
                last_ts_by_symbol[s] = last_ts

    if not pending_syms:
        tprint("All features already generated and up to date.")
        return

    tprint(f"Generating features for {len(pending_syms)} symbols (missing or stale)...")

    # We must load market basket to compute market features
    load_syms = sorted(list(set(pending_syms).union(set(cfg["market_basket"]))))

    dfs = {}

    # Use fetch_years to determine loading window, but ensure at least 90 days for feature safety
    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    with Timer("Feature Data Fetch"):
        for s in load_syms:
            df = store.load(s)
            if not df.empty:
                # Load history based on config
                dfs[s] = df[df.index <= ts_sig].tail(24 * lookback_days)

    if not dfs:
        tprint("No data available for feature generation.")
        return

    with Timer("Feature Computation"):
        panel = to_panel(dfs)
        del dfs  # free raw data
        import gc

        gc.collect()

        mkt_df = compute_market_features(panel, cfg["market_basket"])
        tprint("Market features computed (generation)")
        mkt_gates = add_regime_gates(
            mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"]
        )
        del mkt_df  # free intermediate

        feats, feat_index, feat_columns = compute_features_hourly(panel, mkt_gates, cfg)
        del panel, mkt_gates  # free large objects
        gc.collect()
        tprint("Hourly features computed (generation)")

        # Filter to save only missing symbols
        # feats is Dict[FeatName -> numpy array], feat_columns is list of symbol names
        available_cols = feat_columns
        valid_targets = [s for s in pending_syms if s in available_cols]

        if not valid_targets:
            tprint("No valid target symbols found in computed features.")
            return

        # Build column index for filtering
        col_idx = {s: i for i, s in enumerate(feat_columns)}
        target_indices = [col_idx[s] for s in valid_targets]

        feats_to_save = {}
        for k, arr in feats.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                feats_to_save[k] = arr[:, target_indices]
            elif isinstance(arr, pd.DataFrame):
                cols = [c for c in valid_targets if c in arr.columns]
                if cols:
                    feats_to_save[k] = arr[cols].values

        if feats_to_save:
            min_ts_map = {
                s: last_ts_by_symbol.get(s)
                for s in valid_targets
                if s in last_ts_by_symbol
            }
            save_features(
                feats_to_save,
                ts_sig,
                cfg["data_root"],
                min_timestamp_by_symbol=min_ts_map if min_ts_map else None,
                feat_index=feat_index,
                feat_columns=valid_targets,
            )
        else:
            tprint("No features to save.")

        del feats, feats_to_save
        gc.collect()

    tprint("DAILY FEATURE GENERATION COMPLETE")


def _sort_label_dataset_for_time_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize label dataset ordering for time-based CV consumers."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    sort_cols = [
        c for c in ("__ts__", "timestamp", "__symbol__", "symbol") if c in df.columns
    ]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _load_label_datasets(cfg, run_id, *, allow_oof_fallback: bool = True):
    """Load all label datasets from artifacts. Shared by train_daily / train_daily_base / train_daily_meta."""
    import os as _os

    datasets = {}
    tprint("Loading label datasets from artifacts...")
    _train_symbols_env = str(_os.environ.get("EPM_TRAIN_SYMBOLS", "")).strip()
    _train_symbols_filter = set(
        s.strip() for s in _train_symbols_env.split(",") if s.strip()
    )

    def _normalize_label_like_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame):
            return df
        out = df.copy()
        if "timestamp" in out.columns and "__ts__" not in out.columns:
            out["__ts__"] = out["timestamp"]
        if "symbol" in out.columns and "__symbol__" not in out.columns:
            out["__symbol__"] = out["symbol"]
        if "y_bin" in out.columns and "__y_bin__" not in out.columns:
            out["__y_bin__"] = out["y_bin"]
        if "y_ret" in out.columns and "__y_ret__" not in out.columns:
            out["__y_ret__"] = out["y_ret"]
        if "oof_prob" in out.columns and "oof_pred" not in out.columns:
            out["oof_pred"] = out["oof_prob"]
        return _sort_label_dataset_for_time_cv(out)

    def _load_label_or_oof_artifact(name: str) -> pd.DataFrame | None:
        df = load_artifact_df(cfg["data_root"], run_id, "labels", name)
        if df is not None:
            return _sort_label_dataset_for_time_cv(df)
        if not allow_oof_fallback:
            return None
        oof_candidates = []
        base_name = name.removeprefix("train_")
        oof_candidates.append(f"oof_{base_name}")
        m = re.match(r"^(.*)_(\d+)(?:_(tight|wide|balanced))?$", base_name)
        if m:
            stem, horizon, variant = m.groups()
            suffix = f"_{variant}" if variant else ""
            oof_candidates.insert(0, f"oof_{stem}_H{int(horizon)}{suffix}")
        for oof_name in oof_candidates:
            df = load_artifact_df(cfg["data_root"], run_id, "oof", oof_name)
            if df is None:
                continue
            tprint(
                f"Label fallback: using OOF artifact for {name} from {oof_name}.parquet"
            )
            return _normalize_label_like_df(df)
        return None

    # Spike
    for mode in ["best", "worst"]:
        name = f"spike_anatomy_{mode}"
        df_spike = _load_label_or_oof_artifact(name)
        if df_spike is not None:
            datasets[name] = df_spike

    found_count = 0
    strategies = get_strategies(cfg)
    include_variant_datasets = bool(cfg.get("base_geometry_train_variants", False))
    base_geometry_archetypes = (
        [str(v) for v in cfg.get("base_geometry_archetypes", ["tight", "wide"]) if str(v)]
        if include_variant_datasets
        else []
    )
    for strat in strategies:
        strategy_id = str(strat.get("strategy_id", ""))
        if not strategy_id:
            continue
        for H in strategy_runtime_horizons(strat, cfg):
            H_int = int(H)
            name = f"train_{strategy_id}_{H_int}"
            df = _load_label_or_oof_artifact(name)
            if df is not None:
                if _train_symbols_filter and "__symbol__" in df.columns:
                    df = df[df["__symbol__"].isin(_train_symbols_filter)].reset_index(
                        drop=True
                    )
                if len(df) > 0:
                    datasets[name] = df
                    found_count += 1
            for variant in base_geometry_archetypes:
                if variant == "balanced":
                    continue
                vname = f"train_{strategy_id}_{H_int}_{variant}"
                df_v = _load_label_or_oof_artifact(vname)
                if df_v is not None:
                    if _train_symbols_filter and "__symbol__" in df_v.columns:
                        df_v = df_v[
                            df_v["__symbol__"].isin(_train_symbols_filter)
                        ].reset_index(drop=True)
                    if len(df_v) > 0:
                        datasets[vname] = df_v

    if not include_variant_datasets:
        tprint("Label dataset loader: primary-only mode, variant label artifacts are excluded.")

    # Backward-compatible fallback for older artifact layouts.
    if not found_count:
        for name in [
            "train_long_mr_1",
            "train_short_mr_1",
            "train_long_tf_1",
            "train_short_tf_1",
        ]:
            df = _load_label_or_oof_artifact(name)
            if df is not None:
                datasets[name] = df
                found_count += 1

    # Specialist models
    for name in ["trap_model", "gamma_model"]:
        df = _load_label_or_oof_artifact(name)
        if df is not None:
            datasets[name] = df

    _oos_time_filter = cfg.get("oos_eval_time_filter")
    if _oos_time_filter is not None:
        _t_start, _t_end = _oos_time_filter
        _tf_datasets = {}
        for name, df in datasets.items():
            if "__ts__" in df.columns:
                import pandas as _pd

                _ts = _pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
                _mask = _pd.Series(np.ones(len(df), dtype=bool))
                if _t_start is not None:
                    _mask &= _ts >= _t_start
                if _t_end is not None:
                    _mask &= _ts < _t_end
                df = df.loc[_mask.values].reset_index(drop=True)
            if len(df) > 0:
                _tf_datasets[name] = df
        datasets = _tf_datasets
        tprint(
            f"oos_eval_time_filter=[{_t_start}, {_t_end}): {len(datasets)} datasets retained"
        )

    return datasets, found_count


def _load_base_oof_symbols(data_root: str, run_id: str) -> list[str]:
    """Return the symbol universe covered by saved base-model OOF predictions."""
    oof_dir = os.path.join(data_root, "artifacts", run_id, "oof")
    if not os.path.isdir(oof_dir):
        return []

    preferred = os.path.join(oof_dir, "base_oof_all.parquet")
    candidates = [preferred]

    import glob as _glob

    candidates.extend(sorted(_glob.glob(os.path.join(oof_dir, "oof_*.parquet"))))

    seen: set[str] = set()
    symbols: list[str] = []
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path, columns=["symbol"])
        except Exception as exc:
            tprint(f"WARNING: Could not read OOF symbols from {path}: {exc}")
            continue
        if "symbol" not in df.columns:
            continue
        for sym in df["symbol"].dropna().astype(str).values:
            if not sym or sym == "nan" or sym in seen:
                continue
            seen.add(sym)
            symbols.append(sym)
        if path == preferred and symbols:
            break

    return sorted(symbols)


def _filter_meta_training_to_base_oof_symbols(
    datasets: dict[str, pd.DataFrame],
    alpha_models: dict,
    base_variant_models: dict,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """Filter datasets and cached base OOF vectors to the supplied symbol universe."""
    if not symbols:
        return datasets

    symbol_set = {str(sym) for sym in symbols if str(sym)}
    if not symbol_set:
        return datasets

    filtered: dict[str, pd.DataFrame] = {}
    row_masks: dict[str, np.ndarray] = {}

    for name, df in datasets.items():
        if not isinstance(df, pd.DataFrame):
            filtered[name] = df
            continue
        if "__symbol__" not in df.columns:
            filtered[name] = df
            continue

        mask = df["__symbol__"].astype(str).isin(symbol_set).to_numpy()
        row_masks[name] = mask
        df_filtered = df.loc[mask].reset_index(drop=True)
        if len(df_filtered) > 0:
            filtered[name] = df_filtered
            tprint(
                f"Base OOF symbol filter: {name} {len(df)} -> {len(df_filtered)} rows"
            )
        else:
            tprint(f"Base OOF symbol filter: {name} dropped (no matching symbols)")

    for side_bundle in alpha_models.values():
        if not isinstance(side_bundle, dict):
            continue
        for strategy_id, conf in side_bundle.items():
            if not isinstance(conf, dict):
                continue
            models_by_h = conf.get("models_by_h", {})
            if not isinstance(models_by_h, dict):
                continue
            for h, h_info in models_by_h.items():
                if not isinstance(h_info, dict):
                    continue
                model = h_info.get("model")
                if model is None or getattr(model, "oof_probs", None) is None:
                    continue
                ds_key = f"train_{strategy_id}_{int(h)}"
                mask = row_masks.get(ds_key)
                if mask is None:
                    continue
                oof_probs = np.asarray(model.oof_probs, dtype=np.float32)
                if len(oof_probs) != len(mask):
                    tprint(
                        f"WARNING: {ds_key} OOF length {len(oof_probs)} != mask length {len(mask)}; "
                        "skipping base OOF symbol filter for this model"
                    )
                    continue
                model.oof_probs = oof_probs[mask]

    for variant_key, variant_info in (base_variant_models or {}).items():
        if not isinstance(variant_info, dict):
            continue
        model = variant_info.get("model")
        if model is None or getattr(model, "oof_probs", None) is None:
            continue
        if not isinstance(variant_key, tuple) or len(variant_key) != 4:
            continue
        _side, strategy_id, horizon, variant = variant_key
        ds_key = f"train_{strategy_id}_{int(horizon)}_{variant}"
        mask = row_masks.get(ds_key)
        if mask is None:
            continue
        oof_probs = np.asarray(model.oof_probs, dtype=np.float32)
        if len(oof_probs) != len(mask):
            tprint(
                f"WARNING: {ds_key} variant OOF length {len(oof_probs)} != mask length {len(mask)}; "
                "skipping variant OOF symbol filter for this model"
            )
            continue
        model.oof_probs = oof_probs[mask]
        best_name = getattr(model, "best_model_name", None)
        if hasattr(model, "detailed_metrics") and best_name in model.detailed_metrics:
            dm = model.detailed_metrics[best_name]
            for sigma_key in ("oof_sigma_trees", "oof_sigma_robust"):
                if sigma_key in dm:
                    sigma_vals = np.asarray(dm[sigma_key], dtype=np.float32)
                    if len(sigma_vals) == len(mask):
                        dm[sigma_key] = sigma_vals[mask]

    return filtered


def _default_risk(cfg):
    return {
        "k_sl": cfg.get("risk_k_sl", 2.0),
        "k_trail_start": cfg.get("risk_k_trail_start", 1.0),
        "k_trail_dist": cfg.get("risk_k_trail_dist", 0.5),
        "granular_risk": {},
    }


def train_daily(ts_sig, margin_symbols, cfg, store, ex):
    """Full training: base + meta in one shot."""
    tprint("DAILY TRAINING START")
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    cfg["run_id"] = run_id
    datasets, found_count = _load_label_datasets(
        cfg, run_id, allow_oof_fallback=False
    )

    if not found_count:
        tprint("ERROR: No label datasets found. Run 'labels' mode first.")
        return None

    from extreme_price_movements.pipeline_steps import (
        _expected_feature_keys_from_cfg,
        inject_features_into_datasets,
    )

    req_keys = _expected_feature_keys_from_cfg(cfg)
    datasets = inject_features_into_datasets(datasets, ts_sig, cfg, req_keys)

    from extreme_price_movements.training import train_models_from_artifacts

    with Timer("Model Training"):
        trained_bundle = train_models_from_artifacts(datasets, cfg, train_meta=True)
        tprint("Models trained.")

    tprint("DAILY TRAINING COMPLETE")
    return {
        "ts_trained": ts_sig,
        "bundle": trained_bundle,
        "risk_params": _default_risk(cfg),
    }


def train_daily_base(ts_sig, margin_symbols, cfg, store, ex):
    """Train only base (alpha) models. Saves intermediate state for train_daily_meta."""
    tprint("DAILY BASE TRAINING START")
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    from extreme_price_movements.pipeline_steps import (
        ensure_training_residualization_feature_keys,
    )

    cfg = ensure_training_residualization_feature_keys(cfg)
    cfg["run_id"] = run_id
    datasets, found_count = _load_label_datasets(
        cfg, run_id, allow_oof_fallback=False
    )

    if not found_count:
        tprint("ERROR: No label datasets found. Run 'labels' mode first.")
        return None

    from extreme_price_movements.pipeline_steps import (
        _base_feature_keys_union,
        _expected_feature_keys_from_cfg,
        _meta_feature_keys_union,
        inject_features_into_datasets,
    )
    from extreme_price_movements.training import train_models_from_artifacts

    # _expected_feature_keys_from_cfg and union functions are in pipeline_steps
    req_keys = _expected_feature_keys_from_cfg(cfg)
    meta_keys = set(_meta_feature_keys_union(cfg))
    base_keys = set(_base_feature_keys_union(cfg))
    req_keys = list(set(req_keys) - (meta_keys - base_keys))
    datasets = inject_features_into_datasets(datasets, ts_sig, cfg, req_keys)

    with Timer("Base Model Training"):
        trained_bundle = train_models_from_artifacts(datasets, cfg, train_meta=False)
        tprint("Base models trained.")

    # Save intermediate alpha models for meta training
    import pickle as _pkl

    intermediate_path = os.path.join(
        cfg["data_root"], "artifacts", run_id, "base_models_intermediate.pkl"
    )
    os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
    with open(intermediate_path, "wb") as f:
        _pkl.dump(trained_bundle, f)
    tprint(f"Base models intermediate saved to {intermediate_path}")

    tprint("DAILY BASE TRAINING COMPLETE")
    return {
        "ts_trained": ts_sig,
        "bundle": trained_bundle,
        "risk_params": _default_risk(cfg),
    }


def train_daily_meta(ts_sig, margin_symbols, cfg, store, ex):
    """Train only meta models, loading base models from intermediate state."""
    tprint("DAILY META TRAINING START")
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    from extreme_price_movements.pipeline_steps import (
        ensure_training_residualization_feature_keys,
    )

    cfg = ensure_training_residualization_feature_keys(cfg)
    cfg["run_id"] = run_id

    # Load intermediate alpha models
    import pickle as _pkl

    intermediate_path = os.path.join(
        cfg["data_root"], "artifacts", run_id, "base_models_intermediate.pkl"
    )
    if not os.path.exists(intermediate_path):
        tprint(
            f"ERROR: Base models intermediate not found at {intermediate_path}. Run 'train_base' first."
        )
        return None

    with open(intermediate_path, "rb") as f:
        base_bundle = _pkl.load(f)
    tprint(f"Loaded base models from {intermediate_path}")

    alpha_models = base_bundle.get("alpha_models", {})
    if not alpha_models:
        tprint("ERROR: No alpha models found in intermediate state.")
        return None

    def _strategies_from_alpha_models() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for side, side_models in alpha_models.items():
            if not isinstance(side_models, dict):
                continue
            trade_side = "short" if str(side).lower() == "short" else "long"
            for strategy_id, conf in side_models.items():
                if not isinstance(conf, dict):
                    continue
                h_val = conf.get("H")
                try:
                    source_horizon = int(h_val) if h_val is not None else None
                except Exception:
                    source_horizon = None
                row = {
                    "strategy_id": str(strategy_id),
                    "trade_side": trade_side,
                    # Only needed so get_strategies() keeps dynamic rows instead of
                    # falling back to legacy masks; meta label loading keys off strategy_id/H.
                    "base_event_trigger": str(strategy_id),
                }
                if source_horizon is not None:
                    row["source_horizon"] = int(source_horizon)
                out.append(row)
        return out

    _cfg_strategies = cfg.get("strategies")
    _cfg_strategy_ids = {
        str(s.get("strategy_id"))
        for s in (_cfg_strategies or [])
        if isinstance(s, dict) and s.get("strategy_id")
    }
    _alpha_strategy_ids = {
        str(strategy_id)
        for side_models in alpha_models.values()
        if isinstance(side_models, dict)
        for strategy_id in side_models.keys()
    }
    if not _cfg_strategy_ids or not (_cfg_strategy_ids & _alpha_strategy_ids):
        cfg["strategies"] = _strategies_from_alpha_models()
        tprint(
            "Meta training: strategy list overridden from primary base-model intermediate "
            f"({len(cfg['strategies'])} strategies)"
        )

    datasets, found_count = _load_label_datasets(
        cfg, run_id, allow_oof_fallback=True
    )

    if not found_count:
        tprint("ERROR: No label datasets found. Run 'labels' mode first.")
        return None

    current_strategy_ids = {
        str(s["strategy_id"]) for s in get_strategies(cfg) if isinstance(s, dict)
    }
    intermediate_strategy_ids = {
        str(strategy_id)
        for side_models in alpha_models.values()
        if isinstance(side_models, dict)
        for strategy_id in side_models.keys()
    }
    variant_only_intermediate = all(
        bool(((conf or {}).get("alpha_diag", {}) or {}).get("primary_disabled", False))
        or str(((conf or {}).get("alpha_diag", {}) or {}).get("base_source", ""))
        == "tight_wide_variants_only"
        for side_models in alpha_models.values()
        if isinstance(side_models, dict)
        for conf in side_models.values()
    )
    if variant_only_intermediate:
        tprint(
            "ERROR: Base models intermediate contains only non-primary tight/wide variant bundles. "
            "Meta training requires primary alpha models only."
        )
        tprint(
            f"  Intermediate strategies: {sorted(intermediate_strategy_ids)}"
        )
        tprint(f"  Current strategies: {sorted(current_strategy_ids)}")
        tprint(
            "  Fix: rerun base training in primary-only mode so base_models_intermediate.pkl is rebuilt from primary alpha models."
        )
        return None

    blocked_strategy_ids = set(
        base_bundle.get("blocked_strategy_ids", [])
        or (base_bundle.get("quality_gate_report", {}) or {}).get(
            "blocked_strategy_ids", []
        )
    )
    if blocked_strategy_ids:
        eligible_alpha_models = {}
        eligible_count = 0
        for side, side_models in alpha_models.items():
            if not isinstance(side_models, dict):
                continue
            kept = {}
            for strategy_id, conf in side_models.items():
                if strategy_id in blocked_strategy_ids:
                    continue
                if bool((conf or {}).get("downstream_blocked", False)):
                    continue
                kept[strategy_id] = conf
            if kept:
                eligible_alpha_models[side] = kept
                eligible_count += len(kept)
        if eligible_alpha_models:
            tprint(
                f"Meta training: restricted alpha model set to {eligible_count} non-degenerate strategies "
                f"(blocked={len(blocked_strategy_ids)})"
            )
            alpha_models = eligible_alpha_models
        base_variant_models = {
            k: v
            for k, v in base_bundle.get("base_variant_models", {}).items()
            if k[1] not in blocked_strategy_ids
        }
    else:
        base_variant_models = base_bundle.get("base_variant_models", {})

    base_oof_symbols = _load_base_oof_symbols(cfg["data_root"], run_id)
    _stage_view = cfg.get("_active_stage_view") or {}
    _stage_symbols = _stage_view.get("symbols") or None
    if base_oof_symbols and _stage_symbols:
        _stage_symbol_set = {str(sym) for sym in _stage_symbols if str(sym)}
        _before = len(base_oof_symbols)
        base_oof_symbols = [
            str(sym) for sym in base_oof_symbols if str(sym) in _stage_symbol_set
        ]
        tprint(
            f"Meta training: intersected base OOF symbol universe with active stage view "
            f"({_before} -> {len(base_oof_symbols)} symbols)"
        )
    _planned_max_assets = int(cfg.get("planned_max_assets", 0) or 0)
    if base_oof_symbols and _planned_max_assets > 0:
        _before = len(base_oof_symbols)
        base_oof_symbols = sorted(str(sym) for sym in base_oof_symbols)[
            :_planned_max_assets
        ]
        if len(base_oof_symbols) < _before:
            tprint(
                "Meta training: capped base OOF symbol universe "
                f"({_before} -> {len(base_oof_symbols)} symbols) based on "
                f"planned_max_assets={_planned_max_assets}"
            )
    if base_oof_symbols:
        tprint(
            f"Filtering meta training to {len(base_oof_symbols)} symbols with base-model OOF predictions"
        )
        datasets = _filter_meta_training_to_base_oof_symbols(
            datasets,
            alpha_models,
            base_bundle.get("base_variant_models", {}),
            base_oof_symbols,
        )
    else:
        tprint(
            "WARNING: No base-model OOF symbol universe found; proceeding without symbol filtering"
        )

    cfg["_feature_snapshot_ts"] = ts_sig
    tprint(
        "Meta training: deferring raw meta feature loading to per-bucket two-stage loading "
        "(MDI/HPO subset first, selected-feature full rows second)."
    )

    with Timer("Meta Model Training"):
        from extreme_price_movements.training import train_meta_models_from_artifacts

        meta_models, meta_gate_results = train_meta_models_from_artifacts(
            datasets,
            cfg,
            alpha_models,
            base_variant_models=base_variant_models,
        )
        tprint(f"Meta models trained: {len(meta_models)}")

    meta_blocked_strategy_ids = sorted(
        str(s) for s in (cfg.get("_meta_blocked_strategy_ids") or [])
    )
    if meta_blocked_strategy_ids:
        base_bundle["meta_blocked_strategy_ids"] = meta_blocked_strategy_ids
        base_bundle["blocked_strategy_ids"] = sorted(
            {
                *(base_bundle.get("blocked_strategy_ids", []) or []),
                *meta_blocked_strategy_ids,
            }
        )
        tprint(
            "Meta training: marked strategies as downstream-blocked after meta-quality gate: "
            f"{meta_blocked_strategy_ids}"
        )
    if cfg.get("_meta_strategy_gate_results") is not None:
        base_bundle["meta_strategy_gate_results"] = list(
            cfg.get("_meta_strategy_gate_results") or []
        )

    # Merge meta into base bundle
    base_bundle["meta_models"] = meta_models
    tprint("DAILY META TRAINING COMPLETE")
    return {
        "ts_trained": ts_sig,
        "bundle": base_bundle,
        "risk_params": _default_risk(cfg),
    }


def execute_hourly(ts_sig, margin_symbols, cfg, store, ex, state, logger, model_state):
    tprint(f"Entering function: execute_hourly in main.py")
    run_id = str(uuid.uuid4())
    tprint(f"HOURLY EXEC Start: {ts_sig} RunID={run_id}")
    candidates_pool = select_live_candidates(
        margin_symbols, cfg["market_basket"], pct=0.05
    )
    tprint(f"Candidates selected: {len(candidates_pool)}")

    current_positions = state.get_positions()
    active_syms = list(current_positions.keys())
    tprint(f"Active positions: {len(active_syms)}")
    # Ensure active symbols fetched
    fetch_syms = sorted(list(set(candidates_pool + active_syms)))

    dfs = {}

    # Execution typically only needs enough for feature calculation (90d is safe)
    # But to respect "light" vs full consistency, we can use the same logic if feasible.
    # However, for live execution, fetching 3 years of data every hour is wasteful and slow.
    # We will stick to a safe 90 days for execution as it doesn't affect model training depth.
    # Wait, the user said "model training, etc". Execution is "running" the model.
    # Ideally execution state is minimal. Let's keep 90 days or `fetch_years` if smaller?
    # No, features might break if window is too small. 90 days is a safe lower bound.
    # Let's keep 90 days for execution speed, as it's not training.

    since = (ts_sig - pd.Timedelta(days=90)).floor("D")
    since_ms = int(since.value // 10**6)
    with Timer("Candidate Data Fetch"):
        count_fetch = 0
        for s in fetch_syms:
            try:
                df = store.update_symbol(ex, s, since_ms)
                if not df.empty and df.index.max() >= ts_sig:
                    dfs[s] = df[df.index <= ts_sig].tail(24 * 90)
                    count_fetch += 1
            except Exception:
                pass
        tprint(f"Fetched data for {count_fetch}/{len(fetch_syms)} symbols")
    if not dfs:
        tprint("No data available for execution. Exiting.")
        return

    with Timer("Feature Gen (Candidates)"):
        panel = to_panel(dfs)
        mkt_df = compute_market_features(panel, cfg["market_basket"])
        mkt_gates = add_regime_gates(
            mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"]
        )
        feats_np, feat_index, feat_columns = compute_features_hourly(
            panel, mkt_gates, cfg
        )
        # Reconstruct DataFrames for execution path (needs .loc / column access)
        feats = {
            k: pd.DataFrame(v, index=feat_index, columns=feat_columns)
            if isinstance(v, np.ndarray) and v.ndim == 2
            else v
            for k, v in feats_np.items()
        }
        del feats_np
        tprint("Features generated")

    move_syms = detect_extreme_movement_candidates(
        panel,
        feats,
        ts_sig,
        event_window_hours=cfg.get("inference_event_window_hours", 12),
        move_threshold=cfg.get("inference_event_threshold", 0.07),
        perf_pct=cfg.get("inference_perf_pct", 0.10),
        draw_window_hours=cfg.get("inference_draw_window_hours", 8),
        sign_consistency_min=None,
    )
    if move_syms:
        _update_tradeable_basket(
            state, ts_sig, move_syms, cfg.get("inference_basket_ttl_hours", 24)
        )

    tradeable_basket = _get_tradeable_basket(state, ts_sig)
    tradeable_basket.update(active_syms)

    if not model_state or not model_state.get("bundle"):
        tprint("No trained models available. Skipping execution.")
        return

    bundle = model_state["bundle"]
    alpha_models = bundle["alpha_models"]
    meta_models = bundle["meta_models"]

    risk_conf = model_state.get("risk_params")
    granular_risk = risk_conf.get("granular_risk", {}) if risk_conf else {}

    # Merge bucket_params (optimized exit policy from tpsl_optimiser) into granular_risk
    # This ensures live trading uses the optimized TP/SL/exit parameters
    bucket_params = model_state.get("bucket_params", {})
    if bucket_params and not granular_risk:
        # Build granular_risk from bucket_params
        granular_risk = {}
        for bucket_key, bucket_cfg in bucket_params.items():
            bucket_cfg = flatten_bucket_policy(bucket_cfg)
            # bucket_key is like "LONG_MR", "SHORT_TF", etc.
            # Map to risk keys: "risk_mr_best", "risk_long_mr", etc.
            parts = bucket_key.split("_")
            if len(parts) == 2:
                side, dom = parts[0].lower(), parts[1].lower()
                # Add both naming conventions for compatibility
                granular_risk[f"risk_{dom}_best"] = bucket_cfg
                granular_risk[f"risk_{dom}_worst"] = bucket_cfg
                granular_risk[f"risk_{side}_{dom}"] = bucket_cfg
        if granular_risk:
            risk_conf = risk_conf or {}
            risk_conf["granular_risk"] = granular_risk
            tprint(f"Applied {len(bucket_params)} bucket params to risk config")

    o = panel["open"]
    h = panel["high"]
    l = panel["low"]
    c = panel["close"]
    exits_count = 0
    for sym in active_syms:
        if sym not in c.columns or ts_sig not in c.index:
            tprint(f"Warning: {sym} not in data/index for position update")
            continue
        pos = current_positions[sym]
        ts_risk = TrailingStop.from_dict(pos["risk_state"])
        curr_h = float(h.loc[ts_sig, sym])
        curr_l = float(l.loc[ts_sig, sym])
        curr_c = float(c.loc[ts_sig, sym])
        stopped, exit_px, reason = ts_risk.update(curr_h, curr_l, curr_c)
        if stopped:
            entry_px = pos["entry_px"]
            side = pos["side"]
            if reason == "ambiguous_neutral":
                ret = 0.0
            else:
                ret = (
                    (exit_px / entry_px - 1.0)
                    if side == "long"
                    else (entry_px / exit_px - 1.0)
                )
            logger.log(
                ts_sig,
                {"event": "exit", "symbol": sym, "return": ret, "reason": reason},
            )
            state.clear_position(sym)
            tprint(f"EXIT {sym} ({reason}): ret={ret:.4%}")
            exits_count += 1
        else:
            pos["risk_state"] = ts_risk.to_dict()
            state.set_position(sym, pos)

    tprint(f"Position updates complete. Exits: {exits_count}")
    target_orders = generate_hourly_signals(
        ts_sig,
        feats,
        mkt_gates,
        bundle,
        risk_conf,
        cfg,
        active_syms,
        tradeable_candidates=sorted(tradeable_basket),
    )
    tprint(f"Generated {len(target_orders) if target_orders else 0} signals")

    if target_orders:
        for order in target_orders:
            sym = order["symbol"]
            side = order["side"]
            score = order["score"]
            dom = order["dom"]
            w_alloc = order["weight"]

            if sym not in c.columns:
                continue
            atr = float(feats["atr_pct"].loc[ts_sig, sym])
            entry_px = float(c.loc[ts_sig, sym])

            # Risk Params Lookup
            # Already injected by generate_hourly_signals into order['risk_params']
            # But let's double check or use defaults
            g_risk = order.get("risk_params", {})
            g_risk = flatten_bucket_policy(g_risk)
            pol = compute_entry_policy_decision(
                entry_px=entry_px,
                atr_frac=atr,
                score=float(score),
                bucket_cfg=g_risk,
            )
            if not bool(pol.get("place_order", True)):
                continue
            entry_px = float(pol.get("entry_px_fill", entry_px))

            # Check if Triple Barrier Params are present
            tp_mult = g_risk.get("tp_mult")
            sl_mult = g_risk.get("sl_mult")

            # Apply Score Confidence scaling to SL?
            # Standard logic: k_sl adj = k_sl * (1 + score_scale * abs(score))
            # If using fixed TP/SL mults, do we scale them?
            # Probably scaled SL multiplier is good.

            k_sl = g_risk.get("k_sl", cfg["risk_k_sl"])
            sc_scale = g_risk.get("score_scale", 0.0)  # usually 0 in defaults
            adj = 1.0 + sc_scale * abs(score)

            # Config for TrailingStop
            # If tp_mult is present, we use it for activation (approx) or specialized logic?
            # TrailingStop class currently handles k_sl, k_trail_start...
            # We need to enhance TrailingStop or use a different class if we want fixed barriers.
            # But TrailingStop is serialized.
            # Let's map TP/SL mult to TrailingStop params if possible.
            # TP -> Activation? If we want fixed exit at TP, we can set activation=TP and trail_dist=tiny.
            # Then once activated, stop jumps to Price - tiny ~ Price. Next tick exit.

            if tp_mult and sl_mult:
                # Use dynamic barrier logic
                # We need ATR stats history for dynamic scaling?
                # simulate_trade_hourly computes it.
                # Here we are in live execution.
                # We need to compute the barrier level NOW.

                # To compute dynamic barrier, we need rolling Z.
                # We have `feats`. `feats["atr_pct"]` is the series.
                # We can compute it here.
                from extreme_price_movements.training import scaled_atr_pct

                atr_series = feats["atr_pct"][sym]
                # Slice history
                if len(atr_series) > 30 * 24:
                    win = atr_series.iloc[-(30 * 24) :]
                    base = win.median()
                    std = win.std()
                    z = (atr - base) / (std + 1e-12)
                    barrier_pct = scaled_atr_pct(
                        atr, z, base, z_max=3.0, lo=0.03, hi=0.06
                    )
                else:
                    barrier_pct = 0.045  # Safe mid-range fraction fallback (NOT raw price-space ATR)

                # Convert to k factors relative to CURRENT ATR?
                # barrier_pct is absolute percent.
                # TrailingStop expects k factors relative to `atr_val` passed to it.
                # k_effective = barrier_pct / atr

                k_barrier = barrier_pct / (atr + 1e-12)

                # TP distance = tp_mult * barrier_pct
                # SL distance = sl_mult * barrier_pct

                # Map to TrailingStop:
                # k_sl = sl_mult * k_barrier
                # k_trail_start = tp_mult * k_barrier
                # k_trail_dist = trail_mult * k_barrier (tight trail, from config)
                trail_mult = float(g_risk.get("trail_mult", 0.25))

                k_sl_adj = (
                    float(max(0.05, pol.get("sl_distance_atr_eff", sl_mult)))
                    * k_barrier
                    * adj
                )
                k_ts = (
                    float(max(0.05, pol.get("tp_distance_atr_eff", tp_mult)))
                    * k_barrier
                )
                k_td = float(pol.get("trail_mult_eff", trail_mult)) * k_barrier

            else:
                # Legacy Trailing Logic
                k_sl_adj = k_sl * adj
                k_ts = g_risk.get("k_trail_start", cfg["risk_k_trail_start"])
                k_td = g_risk.get("k_trail_dist", cfg["risk_k_trail_dist"])

            ts_risk = TrailingStop(
                entry_px=entry_px,
                side=side,
                atr_val=atr,
                k_sl=k_sl_adj,
                k_trail_start=k_ts,
                k_trail_dist=k_td,
            )
            pos = {
                "symbol": sym,
                "side": side,
                "entry_px": entry_px,
                "entry_ts": ts_sig.isoformat(),
                "score": float(score),
                "weight": float(w_alloc),
                "risk_state": ts_risk.to_dict(),
                "run_id": run_id,
            }
            state.set_position(sym, pos)
            tprint(
                f"ENTRY {side} {sym} @ {entry_px} (score={score:.4f}, w={w_alloc:.4f}, dom={dom})"
            )
            logger.log(
                ts_sig,
                {
                    "event": "entry",
                    "symbol": sym,
                    "side": side,
                    "score": score,
                    "weight": w_alloc,
                    "dom": dom,
                },
            )

    state.set_last_ts_sig(ts_sig)
    n_orders = len(target_orders) if target_orders else 0
    logger.log(ts_sig, {"n_orders": n_orders, "run_id": run_id})
    tprint("HOURLY EXEC COMPLETE")


def run_live_cycle(initial_model_state=None):
    # Maintain state in function scope (for live loop)
    # But usually this script restarts?
    # For robust persistent state, we need to save/load from disk (pickle).
    # But for this refactor, we just keep it in memory for the process life.

    # Initialize state
    tprint(f"Entering function: run_live_cycle in main.py")
    if initial_model_state:
        model_state = initial_model_state
    else:
        model_state = {"ts_trained": None, "bundle": None, "risk_params": None}
        # Try load from default file
        if os.path.exists("model_state.pkl"):
            try:
                with open("model_state.pkl", "rb") as f:
                    model_state = pickle.load(f)
                tprint("Loaded model state from model_state.pkl")
            except Exception as e:
                tprint(f"Failed to load model state: {e}")

        # If still no bundle, try loading from latest run_id artifacts
        if model_state.get("bundle") is None:
            cfg = CFG.copy()
            latest_run_id = find_latest_run_id(cfg["data_root"])
            if latest_run_id:
                tprint(f"Loading model bundle from latest run_id: {latest_run_id}")
                model_state = load_full_state(latest_run_id, cfg["data_root"])

    cfg = CFG.copy()
    state = StateManager()
    logger = MetricsLogger()

    # Start loop
    while True:
        try:
            ts_sig = get_ts_sig()
            last_ts = state.get_last_ts_sig()
            tprint(f"Current ts_sig: {ts_sig}")

            if last_ts and ts_sig <= last_ts:
                tprint(f"Already processed {ts_sig}. Waiting...")
                time.sleep(60)
                continue

            ex = make_spot_exchange()
            reconcile_state(ex, state)
            with Timer("Margin universe refresh"):
                mu = refresh_margin_universe_daily(
                    None, quotes=["USDT", "USDC", "BUSD", "EUR"]
                )
            store = PartitionedOHLCVStore(
                root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
            )

            last_train = model_state.get("ts_trained")
            need_train = False
            if last_train is None:
                need_train = True
            else:
                if ts_sig.floor("D") > last_train.floor("D"):
                    need_train = True

            if need_train:
                new_state = train_daily(ts_sig, mu.symbols, cfg, store, ex)
                if new_state:
                    model_state = new_state

            execute_hourly(
                ts_sig, mu.symbols, cfg, store, ex, state, logger, model_state
            )
            _monitor_active_positions_5m(ex, state, logger)

        except Exception as e:
            tprint(f"CRITICAL ERROR: {e}")
            import traceback

            traceback.print_exc()

        tprint("Sleeping 60s...")
        time.sleep(60)


if __name__ == "__main__":
    run_live_cycle()
