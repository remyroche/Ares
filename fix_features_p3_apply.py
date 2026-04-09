import re
with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# I want to apply these caching helpers in the PORTABILITY HARDENING block where there are repeated calls.
# Let's inspect the block.
# There are single-feature calls:
# feats["rv_24h"] = ff.numba_rolling_robust_zscore(np.log1p(feats["rv_24h"]).to_numpy(), 96).astype(np.float32)
# Here we can just use the function instead. Wait, it's operating on np.log1p(feats["rv_24h"]), which is a new series.
# If it's a one-off transformation on a temporary series, it won't benefit from caching across features, but it's cleaner.

# Actually, the user request says: "Where many features use the same transform type and window, ensure the helper path is efficient and not recreating excess temporary objects; we may also cache the primitives"
# Let's change the for loops in portability hardening.

content = content.replace(
    'tmp_rz = ff.numba_rolling_robust_zscore(tmp.to_numpy(), 96)',
    'tmp_rz = ff.numba_rolling_robust_zscore(tmp.to_numpy() if hasattr(tmp, "to_numpy") else tmp, 96)'
)

content = content.replace(
    'feats[f] = ff.numba_rolling_robust_zscore(feats[f].to_numpy(), 480).astype(np.float32)',
    'feats[f] = _roll_robust_zscore(f, feats[f], 480)'
)
content = content.replace(
    'feats["ret120h"] = ff.numba_rolling_robust_zscore(feats["ret120h"].to_numpy(), 480).astype(np.float32)',
    'feats["ret120h"] = _roll_robust_zscore("ret120h", feats["ret120h"], 480)'
)
# For np.log1p we can't just pass `feats["rv_24h"]` since it modifies it. We can pass the array directly to the original function for that one.
content = content.replace(
    'feats["rv_120h"] = ff.numba_rolling_rank_pct(feats["rv_120h"].to_numpy(), 480).astype(np.float32)',
    'feats["rv_120h"] = _roll_rank_pct("rv_120h", feats["rv_120h"], 480)'
)
content = content.replace(
    'feats["atr_pct_change"] = ff.numba_rolling_robust_zscore(feats["atr_pct_change"].to_numpy(), 96).astype(np.float32)',
    'feats["atr_pct_change"] = _roll_robust_zscore("atr_pct_change", feats["atr_pct_change"], 96)'
)
content = content.replace(
    'feats["atr_expansion"] = ff.numba_rolling_robust_zscore(feats["atr_expansion"].to_numpy(), 96).astype(np.float32)',
    'feats["atr_expansion"] = _roll_robust_zscore("atr_expansion", feats["atr_expansion"], 96)'
)

content = content.replace(
    'feats[f] = ff.numba_rolling_robust_zscore(feats[f].to_numpy(), 96).astype(np.float32)',
    'feats[f] = _roll_robust_zscore(f, feats[f], 96)'
)

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
