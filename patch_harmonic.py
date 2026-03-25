import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# 1. Update make_regime_weights signature to take horizon instead of window
content = re.sub(
    r"def make_regime_weights\(\n    fwd_ret: np\.ndarray,\n    symbol_id: np\.ndarray,\n    window: int = 4,",
    "def make_regime_weights(\n    fwd_ret: np.ndarray,\n    symbol_id: np.ndarray,\n    horizon: int = 10,",
    content
)

# 2. Update make_regime_weights body to calculate window = int(np.sqrt(horizon))
content = re.sub(
    r"    abs_ret = np\.abs\(returns\)\n    weights = np\.ones\(n, dtype=np\.float32\)",
    "    abs_ret = np.abs(returns)\n    weights = np.ones(n, dtype=np.float32)\n    window = int(np.sqrt(horizon))",
    content
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(content)
