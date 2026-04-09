"""
Out-of-sample evaluation pipeline with random basket sampling and SlicePlanner dispatching.

Usage:
    python -m extreme_price_movements.run_pipeline oos_eval --n-assets 400

Steps:
1. Sample N assets randomly from the filtered universe (basket)
2. Run feature + label generation for all symbols
3. Dispatch basket events into 4 disjoint temporal slices via SlicePlanner:
   - 35% -> base model training
   - 35% -> meta model training
   - 15% -> simple position sizer
4. Run holdout_strategy_eval on assets NOT in the basket (cross-asset OOS)

Design:
- SlicePlanner validates events and enforces purging between adjacent windows.
- Each stage receives strictly disjoint (time, symbol) slices.
- The OOS holdout uses complement symbols never seen during training.
"""

from __future__ import annotations

import gc
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    InnerFoldConfig,
    OuterFoldConfig,
    PlannerPresetConfig,
    PurgePolicy,
    SamplingPolicy,
    SlicePlanner,
    SlicePlannerConfig,
    SymbolPolicy,
)
from extreme_price_movements.universe import (
    apply_hardcoded_universe_exclusions,
    deduplicate_symbols_by_base,
    get_training_universe,
    refresh_margin_universe_daily,
)
from extreme_price_movements.utils import tprint


def sample_universe_assets(
    n_assets: int,
    cfg: dict,
    store: Optional[PartitionedOHLCVStore] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Sample *n_assets* symbols from the training universe.

    Returns (basket_symbols, oos_symbols).
    """
    rng = np.random.RandomState(seed)

    if store is None:
        store = PartitionedOHLCVStore(
            root_dir=cfg["data_root"], timeframe=cfg.get("timeframe", "1h")
        )

    all_symbols = get_training_universe(None, cfg, store, ts_sig=None)

    if len(all_symbols) <= n_assets:
        tprint(
            f"Universe has {len(all_symbols)} symbols (<= requested {n_assets}). "
            "Using all as basket; OOS set will be empty."
        )
        return sorted(all_symbols), []

    rng.shuffle(all_symbols)
    basket = sorted(all_symbols[:n_assets])
    oos = sorted(all_symbols[n_assets:])

    tprint(
        f"Universe sampled: {len(basket)} basket + {len(oos)} OOS symbols "
        f"(seed={seed})"
    )
    return basket, oos


def _build_events_from_labels(
    cfg: dict,
    run_id: str,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build an event DataFrame from label parquet artifacts.

    Each row in a label file is one event.  We extract (event_id, symbol, t0, t1).
    """
    labels_dir = Path(cfg["data_root"]) / "artifacts" / run_id / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"No labels directory at {labels_dir}")

    frames: List[pd.DataFrame] = []
    event_offset = 0

    sym_set = set(symbols) if symbols is not None else None

    for pq in sorted(labels_dir.glob("train_*.parquet")):
        if "_tight" in pq.name or "_wide" in pq.name or "_balanced" in pq.name:
            continue
        try:
            df = pd.read_parquet(pq, columns=["__ts__", "__symbol__"])
        except Exception:
            continue
        if df.empty:
            continue
        if sym_set is not None and "__symbol__" in df.columns:
            df = df[df["__symbol__"].isin(sym_set)]
        if df.empty:
            continue

        ts_col = df["__ts__"] if "__ts__" in df.columns else df.iloc[:, 0]
        sym_col = df["__symbol__"] if "__symbol__" in df.columns else "ALL"

        chunk = pd.DataFrame(
            {
                "event_id": np.arange(event_offset, event_offset + len(df)),
                "symbol": sym_col.values if hasattr(sym_col, "values") else sym_col,
                "t0": pd.to_datetime(ts_col.values, utc=True, errors="coerce"),
                "t1": pd.to_datetime(ts_col.values, utc=True, errors="coerce")
                + pd.Timedelta(hours=6),
            }
        )
        frames.append(chunk)
        event_offset += len(df)

    if not frames:
        raise RuntimeError(f"No label events loaded from {labels_dir}")

    events = pd.concat(frames, ignore_index=True)
    events = events.dropna(subset=["t0"]).sort_values("t0").reset_index(drop=True)
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    tprint(f"Built {len(events)} events from {labels_dir}")
    return events


def dispatch_events_temporal(
    events: pd.DataFrame,
    basket: List[str],
    fractions: Tuple[float, ...] = (0.35, 0.35, 0.15, 0.15),
    purge_hours: float = 24.0,
) -> Dict[str, np.ndarray]:
    """Dispatch basket events into disjoint temporal windows via SlicePlanner.

    Uses SlicePlanner for event validation and purging.  Assigns events to
    sequential temporal windows in the order: base, meta, sizer, temporal_oos.
    A purge gap of *purge_hours* is enforced between adjacent windows.

    Returns dict mapping stage name -> array of original event positional indices.
    """
    basket_set = set(basket)
    basket_mask = events["symbol"].isin(basket_set)
    basket_events = events.loc[basket_mask].sort_values("t0").copy()

    if basket_events.empty:
        raise RuntimeError("No basket events found for dispatching")

    purge_td = pd.Timedelta(hours=purge_hours)

    schema = EventSchema()
    planner_cfg = SlicePlannerConfig.fast_defaults(schema=schema)
    try:
        planner = SlicePlanner(planner_cfg)
        validated = planner.build(basket_events)
        clean = validated["events"]
    except Exception as exc:
        tprint(f"SlicePlanner validation fallback ({exc}); using raw events")
        clean = basket_events

    t0_vals = clean["t0"].values
    t_min = pd.Timestamp(t0_vals[0])
    t_max = pd.Timestamp(t0_vals[-1])
    total_range = (t_max - t_min).total_seconds()
    if total_range <= 0:
        raise RuntimeError("Events span zero time; cannot dispatch")

    def _frac_ts(frac: float) -> pd.Timestamp:
        return t_min + pd.Timedelta(seconds=frac * total_range)

    boundaries = [
        _frac_ts(fractions[0]),
        _frac_ts(fractions[0] + fractions[1]),
        _frac_ts(fractions[0] + fractions[1] + fractions[2]),
    ]

    base_idx = clean.index[t0_vals < boundaries[0]].values
    meta_idx = clean.index[
        (t0_vals >= boundaries[0] + purge_td) & (t0_vals < boundaries[1])
    ].values
    sizer_idx = clean.index[
        (t0_vals >= boundaries[1] + purge_td) & (t0_vals < boundaries[2])
    ].values
    temporal_oos_idx = clean.index[t0_vals >= boundaries[2] + purge_td].values

    oos_mask = ~events["symbol"].isin(basket_set)
    oos_idx = events.index[oos_mask].values

    tprint(
        f"Dispatch: base={len(base_idx)} meta={len(meta_idx)} "
        f"sizer={len(sizer_idx)} temporal_oos={len(temporal_oos_idx)} "
        f"cross_asset_oos={len(oos_idx)}"
    )

    return {
        "base": base_idx,
        "meta": meta_idx,
        "sizer": sizer_idx,
        "temporal_oos": temporal_oos_idx,
        "cross_asset_oos": oos_idx,
    }


def _time_window_from_dispatch(
    events: pd.DataFrame, indices: np.ndarray
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return (t_start, t_end) for the events at *indices*."""
    if len(indices) == 0:
        return None, None
    subset = events.loc[indices]
    return pd.Timestamp(subset["t0"].min()), pd.Timestamp(subset["t0"].max())


def _apply_time_filter_to_datasets(
    datasets: Dict[str, pd.DataFrame],
    t_start: Optional[pd.Timestamp],
    t_end: Optional[pd.Timestamp],
) -> Dict[str, pd.DataFrame]:
    """Filter label datasets to [t_start, t_end)."""
    if t_start is None and t_end is None:
        return datasets
    filtered: Dict[str, pd.DataFrame] = {}
    for name, df in datasets.items():
        ts_col = "__ts__" if "__ts__" in df.columns else None
        if ts_col is None:
            filtered[name] = df
            continue
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        mask = np.ones(len(df), dtype=bool)
        if t_start is not None:
            mask &= ts >= t_start
        if t_end is not None:
            mask &= ts < t_end
        df_f = df.loc[mask].reset_index(drop=True)
        if len(df_f) > 0:
            filtered[name] = df_f
    return filtered


def _load_label_datasets_filtered(
    cfg: dict,
    run_id: str,
    symbols: Optional[List[str]] = None,
    t_start: Optional[pd.Timestamp] = None,
    t_end: Optional[pd.Timestamp] = None,
) -> Tuple[Dict[str, pd.DataFrame], int]:
    """Load label datasets with optional symbol and time filtering."""
    from extreme_price_movements.main import _load_label_datasets as _orig_load

    _orig_env = os.environ.get("EPM_TRAIN_SYMBOLS", "")
    try:
        if symbols:
            os.environ["EPM_TRAIN_SYMBOLS"] = ",".join(symbols)
        else:
            os.environ.pop("EPM_TRAIN_SYMBOLS", None)

        datasets, found_count = _orig_load(cfg, run_id)
    finally:
        if _orig_env:
            os.environ["EPM_TRAIN_SYMBOLS"] = _orig_env
        else:
            os.environ.pop("EPM_TRAIN_SYMBOLS", None)

    if t_start is not None or t_end is not None:
        datasets = _apply_time_filter_to_datasets(datasets, t_start, t_end)

    kept = sum(1 for df in datasets.values() if len(df) > 0)
    tprint(
        f"Label datasets: {found_count} loaded, {kept} retained after "
        f"symbol={len(symbols) if symbols else 'all'} time=[{t_start}, {t_end})"
    )
    return datasets, kept


def _save_base_models_intermediate(cfg: dict, run_id: str, state: Any) -> Path:
    """Persist base models to the intermediate path expected by train_daily_meta."""
    import pickle

    models_dir = Path(cfg["data_root"]) / "artifacts" / run_id / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "base_models_intermediate.pkl"
    with open(path, "wb") as f:
        pickle.dump(state, f)
    tprint(f"Saved base models intermediate to {path}")
    return path


def _run_base_training(
    cfg: dict,
    run_id: str,
    basket: List[str],
    t_start: Optional[pd.Timestamp],
    t_end: Optional[pd.Timestamp],
    store: PartitionedOHLCVStore,
) -> Optional[Any]:
    """Run base model training on basket symbols within [t_start, t_end)."""
    from extreme_price_movements.pipeline_steps import run_training_step
    from extreme_price_movements.run_pipeline import _load_mask_params_by_mode

    tprint("=" * 80)
    tprint("STAGE 1/4: BASE TRAINING")
    tprint("=" * 80)
    tprint(f"  Symbols: {len(basket)} | Time: [{t_start}, {t_end})")

    cfg_s = dict(cfg)
    _load_mask_params_by_mode(cfg_s)
    cfg_s["oos_eval_time_filter"] = (t_start, t_end)

    ts_sig = pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True)

    _orig_env = os.environ.get("EPM_TRAIN_SYMBOLS", "")
    try:
        os.environ["EPM_TRAIN_SYMBOLS"] = ",".join(basket)
        state = run_training_step(
            ts_sig,
            cfg_s,
            store=store,
            margin_symbols=None,
            base_only=True,
            meta_only=False,
        )
    finally:
        if _orig_env:
            os.environ["EPM_TRAIN_SYMBOLS"] = _orig_env
        else:
            os.environ.pop("EPM_TRAIN_SYMBOLS", None)

    if state is None:
        tprint("WARNING: Base training returned None")
        return None

    if "alpha_models" in state:
        _save_base_models_intermediate(cfg, run_id, state)

    tprint("STAGE 1/4: BASE TRAINING COMPLETE")
    gc.collect()
    return state


def _run_meta_training(
    cfg: dict,
    run_id: str,
    basket: List[str],
    t_start: Optional[pd.Timestamp],
    t_end: Optional[pd.Timestamp],
    store: PartitionedOHLCVStore,
) -> Optional[Any]:
    """Run meta model training on basket symbols within [t_start, t_end)."""
    from extreme_price_movements.main import train_daily_meta
    from extreme_price_movements.run_pipeline import _load_mask_params_by_mode

    tprint("=" * 80)
    tprint("STAGE 2/4: META TRAINING")
    tprint("=" * 80)
    tprint(f"  Symbols: {len(basket)} | Time: [{t_start}, {t_end})")

    cfg_s = dict(cfg)
    _load_mask_params_by_mode(cfg_s)
    cfg_s["oos_eval_time_filter"] = (t_start, t_end)

    ts_sig = pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True)

    _orig_env = os.environ.get("EPM_TRAIN_SYMBOLS", "")
    try:
        os.environ["EPM_TRAIN_SYMBOLS"] = ",".join(basket)
        result = train_daily_meta(ts_sig, None, cfg_s, store, None)
    finally:
        if _orig_env:
            os.environ["EPM_TRAIN_SYMBOLS"] = _orig_env
        else:
            os.environ.pop("EPM_TRAIN_SYMBOLS", None)

    if result:
        import joblib

        models_dir = Path(cfg["data_root"]) / "artifacts" / run_id / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        meta_state_path = models_dir / "model_state_meta.pkl"
        joblib.dump(result, meta_state_path)
        tprint(f"Meta model state saved to {meta_state_path}")
        del result
        gc.collect()
        tprint("STAGE 2/4: META TRAINING COMPLETE")
    else:
        tprint("WARNING: Meta training returned None")

    return result


def _run_simple_position_sizer(
    cfg: dict,
    run_id: str,
    basket: List[str],
    t_start: Optional[pd.Timestamp],
    t_end: Optional[pd.Timestamp],
) -> Optional[Dict[str, Any]]:
    """Run simple position sizer on basket symbols within [t_start, t_end)."""
    from extreme_price_movements.simple_position_sizer import (
        run_simple_position_sizer_from_artifacts,
    )

    tprint("=" * 80)
    tprint("STAGE 3/4: SIMPLE POSITION SIZER")
    tprint("=" * 80)
    tprint(f"  Symbols: {len(basket)} | Time: [{t_start}, {t_end})")

    cfg_s = dict(cfg)
    cfg_s["oos_eval_time_filter"] = (t_start, t_end)

    _orig_env = os.environ.get("EPM_TRAIN_SYMBOLS", "")
    try:
        os.environ["EPM_TRAIN_SYMBOLS"] = ",".join(basket)
        results = run_simple_position_sizer_from_artifacts(
            data_root=cfg["data_root"],
            run_id=run_id,
            top_n_strategies=4,
            time_filter=(t_start, t_end),
        )
    finally:
        if _orig_env:
            os.environ["EPM_TRAIN_SYMBOLS"] = _orig_env
        else:
            os.environ.pop("EPM_TRAIN_SYMBOLS", None)

    if results:
        tprint(f"STAGE 3/4: SIZER COMPLETE ({len(results)} strategies evaluated)")
    else:
        tprint("WARNING: Simple position sizer produced no results")

    gc.collect()
    return results


def _run_holdout_eval(
    cfg: dict,
    run_id: str,
    oos_symbols: List[str],
) -> Optional[List[Dict[str, Any]]]:
    """Run holdout strategy eval on OOS symbols (fully out-of-sample)."""
    from extreme_price_movements.holdout_strategy_eval import (
        _load_or_write_freeze_file,
        evaluate_holdout_symbol,
    )

    tprint("=" * 80)
    tprint("STAGE 4/4: HOLDOUT STRATEGY EVALUATION (OOS)")
    tprint("=" * 80)
    tprint(f"  OOS symbols: {len(oos_symbols)}")

    if not oos_symbols:
        tprint("No OOS symbols available; skipping holdout eval.")
        return None

    freeze = _load_or_write_freeze_file(cfg["data_root"], run_id)
    all_results: List[Dict[str, Any]] = []

    for symbol in oos_symbols:
        try:
            result = evaluate_holdout_symbol(cfg["data_root"], run_id, symbol, freeze)
            all_results.append(result)
            if result.get("portfolio"):
                tprint(
                    f"  {symbol}: wallet_pnl="
                    f"{result['portfolio'].get('wallet_pnl', 'N/A')}"
                )
        except Exception as e:
            tprint(f"  {symbol}: SKIPPED ({e})")

    import json

    output_path = (
        Path(cfg["data_root"])
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "oos_holdout_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    tprint(f"OOS holdout results: {len(all_results)} symbols -> {output_path}")
    tprint("STAGE 4/4: HOLDOUT EVALUATION COMPLETE")
    return all_results


def run_oos_eval_pipeline(
    cfg: dict,
    n_assets: int = 400,
    seed: int = 42,
    ts_override: Optional[str] = None,
    fractions: Tuple[float, ...] = (0.35, 0.35, 0.15, 0.15),
    purge_hours: float = 24.0,
    skip_features: bool = False,
    skip_labels: bool = False,
) -> None:
    """Main orchestrator for the OOS evaluation pipeline.

    Steps:
    1. Sample N assets from universe
    2. Generate features + labels for all symbols
    3. Dispatch basket events into temporal windows via SlicePlanner
    4. Run base training on 35% of basket events
    5. Run meta training on next 35% of basket events
    6. Run simple_position_sizer on next 15% of basket events
    7. Run holdout_strategy_eval on OOS symbols (complement of basket)
    """
    tprint("=" * 80)
    tprint("OOS EVAL PIPELINE")
    tprint(f"  n_assets={n_assets} seed={seed} fractions={fractions}")
    tprint("=" * 80)

    store = PartitionedOHLCVStore(
        root_dir=cfg["data_root"], timeframe=cfg.get("timeframe", "1h")
    )

    # --- 1. Resolve timestamp ---
    from extreme_price_movements.run_pipeline import _resolve_ts_sig

    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        ts_sig = pd.Timestamp.utcnow().floor("h")
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    tprint(f"Run ID: {run_id}")

    # --- 2. Sample universe ---
    basket, oos_symbols = sample_universe_assets(n_assets, cfg, store, seed)
    if not basket:
        tprint("ERROR: No symbols in basket. Aborting.")
        return

    basket_path = Path(cfg["data_root"]) / "artifacts" / run_id / "oos_eval_basket.json"
    basket_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    basket_path.write_text(
        json.dumps(
            {
                "basket": basket,
                "oos": oos_symbols,
                "n_assets": n_assets,
                "seed": seed,
                "run_id": run_id,
                "fractions": list(fractions),
                "purge_hours": purge_hours,
            },
            indent=2,
        )
    )

    # --- 3. Feature + label generation ---
    if not skip_features:
        from extreme_price_movements.run_pipeline import run_features

        tprint("Generating features...")
        run_features(cfg, ts_override=run_id, store=store)

    if not skip_labels:
        from extreme_price_movements.run_pipeline import run_labels

        tprint("Generating labels...")
        run_labels(cfg, ts_override=run_id, store=store)

    # --- 4. Build events and dispatch ---
    events = _build_events_from_labels(cfg, run_id, symbols=basket + oos_symbols)
    dispatch = dispatch_events_temporal(events, basket, fractions, purge_hours)

    t_start_base, t_end_base = _time_window_from_dispatch(events, dispatch["base"])
    t_start_meta, t_end_meta = _time_window_from_dispatch(events, dispatch["meta"])
    t_start_sizer, t_end_sizer = _time_window_from_dispatch(events, dispatch["sizer"])

    tprint(
        f"Temporal windows:\n"
        f"  base:  [{t_start_base}, {t_end_base})\n"
        f"  meta:  [{t_start_meta}, {t_end_meta})\n"
        f"  sizer: [{t_start_sizer}, {t_end_sizer})\n"
        f"  OOS:   complement symbols ({len(oos_symbols)})"
    )

    # --- 5. Stage 1: Base Training ---
    base_state = _run_base_training(
        cfg, run_id, basket, t_start_base, t_end_base, store
    )
    if base_state is None:
        tprint("ERROR: Base training failed. Aborting pipeline.")
        return

    # --- 6. Stage 2: Meta Training ---
    meta_result = _run_meta_training(
        cfg, run_id, basket, t_start_meta, t_end_meta, store
    )
    if meta_result is None:
        tprint("WARNING: Meta training failed. Continuing with base-only.")

    # --- 7. Stage 3: Simple Position Sizer ---
    sizer_results = _run_simple_position_sizer(
        cfg, run_id, basket, t_start_sizer, t_end_sizer
    )

    # --- 8. Stage 4: Holdout Evaluation (OOS) ---
    _run_holdout_eval(cfg, run_id, oos_symbols)

    # --- Final summary ---
    tprint("=" * 80)
    tprint("OOS EVAL PIPELINE COMPLETE")
    tprint(f"  Run ID: {run_id}")
    tprint(f"  Basket: {len(basket)} symbols")
    tprint(f"  OOS:    {len(oos_symbols)} symbols")
    tprint(
        f"  Dispatch: base={len(dispatch['base'])} meta={len(dispatch['meta'])} "
        f"sizer={len(dispatch['sizer'])} temporal_oos={len(dispatch['temporal_oos'])} "
        f"cross_asset_oos={len(dispatch['cross_asset_oos'])}"
    )
    tprint("=" * 80)
