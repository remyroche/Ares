import numpy as np

sl_lo = 0.005
sl_hi = 0.06
z_norm = 0.0

sl_mult = sl_lo + (sl_hi - sl_lo) * z_norm
tp_vals = 0.02
sl_base_mult = 0.6

sl_vals_compounded = sl_base_mult * sl_mult * tp_vals
print(f"sl_mult: {sl_mult}, sl_vals_compounded: {sl_vals_compounded}")
