import re

with open("extreme_price_movements/training.py", "r") as f:
    content = f.read()

new_func = """def _build_optimal_candidate_mask(panel, feats, cfg):
    \"\"\"Build candidate mask strictly from persisted offline-optimal threshold conditions.\"\"\"
    cfg_resolved = _resolve_training_cfg_with_offline_optimisers(cfg)

    from extreme_price_movements.strategy_registry import get_strategies
    from extreme_price_movements.lgbm_based_mask_generation import FeatureProcessor, CanonicalRuleMaskResolver

    strategies = get_strategies(cfg_resolved)

    tprint(f"Building context-based masks from LGBM strategies: {len(strategies)} strategies found.")

    close_df = panel['close']
    n_ts, n_syms = close_df.shape

    idx_flat = np.repeat(close_df.index.to_numpy(), n_syms)
    sym_flat = np.tile(close_df.columns.to_numpy(), n_ts)

    feats_1d = {}
    for k, v in feats.items():
        if hasattr(v, 'to_numpy'):
            feats_1d[k] = v.to_numpy(dtype=np.float32).ravel()
        else:
            feats_1d[k] = np.asarray(v, dtype=np.float32).ravel()

    fp = FeatureProcessor()
    X, metadata, audits = fp.prepare_features(
        feats_1d, idx_flat, sym_flat, cfg_resolved
    )
    resolver = CanonicalRuleMaskResolver(X, metadata)

    mask_by_strategy = {}
    global_mask = None

    for strat in strategies:
        strat_id = strat['strategy_id']
        key = strat['base_event_trigger']
        try:
            mask_1d = resolver.get_mask(key)
            mask_2d = mask_1d.reshape((n_ts, n_syms))
            mask_df = pd.DataFrame(mask_2d, index=close_df.index, columns=close_df.columns)
            mask_by_strategy[strat_id] = mask_df

            if global_mask is None:
                global_mask = mask_df
            else:
                global_mask = global_mask | mask_df
        except KeyError as e:
            tprint(f"Failed to generate mask for {strat_id} with key {key}: {e}")
            mask_by_strategy[strat_id] = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)

    if global_mask is None:
        global_mask = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)

    return global_mask, cfg_resolved, mask_by_strategy
"""

content = re.sub(
    r"def _build_optimal_candidate_mask\(panel, feats, cfg\):.*?return cand_mask, cfg_resolved",
    new_func,
    content,
    flags=re.DOTALL
)

with open("extreme_price_movements/training.py", "w") as f:
    f.write(content)
