import numpy as np
from extreme_price_movements.limit_order_pricer import estimate_entry_limit_offset

mae_hat = 0.02
mfe_hat = 0.01
u_hat = 0.5
confidence = 0.8
res = estimate_entry_limit_offset(mae_hat, mfe_hat, u_hat, confidence)
print(res)
