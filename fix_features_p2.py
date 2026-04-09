with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

import re

# Add helper methods for _roll_robust_zscore and _roll_rank_pct
helpers = """    def _roll_robust_zscore(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_robust_zscore", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = ff.apply_to_frame(
                src, ff._numba_rolling_robust_zscore_1d, int(window)
            ).astype(np.float32)
        return primitive_cache[key]

    def _roll_rank_pct(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_rank_pct", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = ff.apply_to_frame(
                src, ff._numba_rolling_rank_pct_1d, int(window)
            ).astype(np.float32)
        return primitive_cache[key]"""

# Need to check `fast_funcs.py` for correct function names for robust zscore and rank pct on 1D arrays or 2D.
