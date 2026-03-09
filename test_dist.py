import numpy as np
from extreme_price_movements.mask_optimiser import _compute_regime_distinctness

mask_high = np.array([True, False, False, True, False])
mask_low = np.array([False, True, False, False, True])
fwd_ret = np.random.randn(5)
mae_high = np.random.rand(5)
mfe_high = np.random.rand(5)
mae_low = np.random.rand(5)
mfe_low = np.random.rand(5)

# Will it crash?
res = _compute_regime_distinctness(mask_high, mask_low, fwd_ret, mae_high, mfe_high, mae_low, mfe_low)
print(f"Distinctness returned successfully: {res}")
