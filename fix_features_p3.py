with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

import re

# `numba_rolling_robust_zscore` takes a DataFrame directly.
# `numba_rolling_rank_pct` also takes a DataFrame/Array directly? Let's check.

helpers = """    def _roll_min(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_min", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = ff.numba_rolling_min(src, int(window)).astype(
                np.float32
            )
        return primitive_cache[key]

    def _roll_robust_zscore(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_robust_zscore", name, int(window))
        if key not in primitive_cache:
            # We must pass numpy array to numba_rolling_robust_zscore in fast_funcs ?
            # Wait, `numba_rolling_robust_zscore` takes a df but the caller code passed `.to_numpy()`.
            primitive_cache[key] = pd.DataFrame(
                ff.numba_rolling_robust_zscore(src.to_numpy() if hasattr(src, 'to_numpy') else src, int(window)),
                index=src.index if hasattr(src, 'index') else None,
                columns=src.columns if hasattr(src, 'columns') else None
            ).astype(np.float32)
        return primitive_cache[key]

    def _roll_rank_pct(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:
        key = ("roll_rank_pct", name, int(window))
        if key not in primitive_cache:
            primitive_cache[key] = pd.DataFrame(
                ff.numba_rolling_rank_pct(src.to_numpy() if hasattr(src, 'to_numpy') else src, int(window)),
                index=src.index if hasattr(src, 'index') else None,
                columns=src.columns if hasattr(src, 'columns') else None
            ).astype(np.float32)
        return primitive_cache[key]"""

content = re.sub(r'\s*def _roll_min.*?return primitive_cache\[key\]', helpers, content, flags=re.DOTALL)

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
