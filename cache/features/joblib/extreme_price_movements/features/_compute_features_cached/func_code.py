# first line: 189
@_cache.cache
def _compute_features_cached(panel_hash, mkt_gates_hash, cfg_tuple, panel, mkt_gates):
    """Cached implementation of feature computation."""
    return _compute_features_impl(panel, mkt_gates, dict(cfg_tuple))
