import re

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    content = f.read()

new_func = """def build_strategy_masks(
    artifacts: RunArtifacts, cfg_runtime: Dict[str, Any] | None = None
) -> Dict[str, pd.DataFrame]:
    \"\"\"Build strategy masks from full strategy definitions using LGBM Contexts.\"\"\"
    from extreme_price_movements.strategy_registry import get_strategies
    from extreme_price_movements.lgbm_based_mask_generation import FeatureProcessor, CanonicalRuleMaskResolver

    cfg = cfg_runtime or {}
    strategies = get_strategies(cfg)

    close_df = artifacts.panel['close']
    n_ts, n_syms = close_df.shape

    idx_flat = np.repeat(close_df.index.to_numpy(), n_syms)
    sym_flat = np.tile(close_df.columns.to_numpy(), n_ts)

    feats_1d = {}
    for k, v in artifacts.features.items():
        if hasattr(v, 'to_numpy'):
            feats_1d[k] = v.to_numpy(dtype=np.float32).ravel()
        else:
            feats_1d[k] = np.asarray(v, dtype=np.float32).ravel()

    fp = FeatureProcessor()
    X, metadata, _ = fp.prepare_features(
        feats_1d, idx_flat, sym_flat, cfg
    )
    resolver = CanonicalRuleMaskResolver(X, metadata)

    out = {}
    for strat in strategies:
        sid = str(strat.get("strategy_id"))
        base = str(strat.get("base_event_trigger", "")).strip()
        if not base:
            continue

        try:
            mask_1d = resolver.get_mask(base)
            mask_2d = mask_1d.reshape((n_ts, n_syms))
            out[sid] = pd.DataFrame(mask_2d, index=close_df.index, columns=close_df.columns, dtype=bool)
        except Exception as e:
            out[sid] = pd.DataFrame(False, index=close_df.index, columns=close_df.columns, dtype=bool)

    return out
"""

content = re.sub(
    r"def build_strategy_masks\(\s*artifacts: RunArtifacts, cfg_runtime: Dict\[str, Any\] \| None = None\s*\) -> Dict\[str, pd\.DataFrame\]:.*?return out",
    new_func,
    content,
    flags=re.DOTALL
)

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(content)
