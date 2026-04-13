import re

with open('extreme_price_movements/pipeline_steps.py', 'r') as f:
    code = f.read()

# 1. Update _align_features_to_panel to skip redundant reindexing
old_align = """def _align_features_to_panel(
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
        out[k] = df.reindex(index=close.index, columns=symbols).astype(
            np.float32, copy=False
        )
        del df
    import gc as _gc

    _gc.collect()
    return out"""

new_align = """def _align_features_to_panel(
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
    return out"""

code = code.replace(old_align, new_align)

# 2. Update data load loop to be parallel + float32 and delete dfs quickly
old_load = """    dfs = {}

    lookback_days = max(90, int(cfg["fetch_years"] * 365))

    with Timer("Data Load"):
        for s in train_syms:
            df = store.load(s)
            if not df.empty:
                dfs[s] = df[df.index <= ts_sig].tail(24 * lookback_days)"""

new_load = """    dfs = {}

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
                    dfs[s] = df"""

code = code.replace(old_load, new_load)


# 3. Del dfs and gc.collect after to_panel
old_panel = """    if not dfs:
        tprint("No data available.")
        return

    panel = to_panel(dfs)
    tprint(
        f"Label data load complete: symbols={len(dfs)} "
        f"horizons={horizons if horizons is not None else list(CANON_HORIZONS)} "
        f"label_persist_incremental={bool(cfg.get('label_persist_incremental', False))}"
    )"""

new_panel = """    if not dfs:
        tprint("No data available.")
        return

    panel = to_panel(dfs)
    del dfs
    gc.collect()

    tprint(
        f"Label data load complete: symbols={panel['close'].shape[1]} "
        f"horizons={horizons if horizons is not None else list(CANON_HORIZONS)} "
        f"label_persist_incremental={bool(cfg.get('label_persist_incremental', False))}"
    )"""

code = code.replace(old_panel, new_panel)

# 4. Optimize SlicePlanner events extraction and in-place filtering
# Find the slice planner section
slice_planner_start = code.find("        from extreme_price_movements.periods_symbols_management import (")
slice_planner_end = code.find("    dataset_manifest: dict[str, dict[str, object]] = {}")

if slice_planner_start != -1 and slice_planner_end != -1:
    old_slice_planner = code[slice_planner_start:slice_planner_end]

    new_slice_planner = """        from extreme_price_movements.periods_symbols_management import (
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

"""
    code = code.replace(old_slice_planner, new_slice_planner)

with open('extreme_price_movements/pipeline_steps.py', 'w') as f:
    f.write(code)
