with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# Fix `ret1h_std_96` and `_roll_mean`/`_roll_std` missing usages inside `compute_regime_features`

# Inside compute_regime_features, these variables are not available
# So we need to calculate them using ff.numba_* and the arrays passed (c)

import re

# `ret1h_std_96` means standard deviation over 96 bars of ret1h.
# In `compute_regime_features` context, we only have `c`, `h`, `l`, `v`. `ret1h` is `c.diff(1).fillna(0.0)`.

# First, fix `ret1h_std_96` to be `ff.numba_rolling_std(ret1h, 96)`
content = content.replace("ret1h_std_96", "ff.numba_rolling_std(ret1h, 96)")

# Fix `_roll_std` to `ff.numba_rolling_std`
content = content.replace('_roll_std("ret1h", feats["ret1h"], 4)', 'ff.numba_rolling_std(ret1h, 4)')
content = content.replace('_roll_std("ret1h", feats["ret1h"], 12)', 'ff.numba_rolling_std(ret1h, 12)')

# Fix `_roll_mean`
content = content.replace('_roll_mean("ret1h", feats["ret1h"], 24)', 'ff.numba_rolling_mean(ret1h, 24)')
content = content.replace('_roll_mean("volume", v, 24)', 'ff.numba_rolling_mean(v, 24)')

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
