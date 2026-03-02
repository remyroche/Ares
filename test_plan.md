Wait, look at `extreme_price_movements/labeling.py` for `_numba_triple_barrier`.
`labels[i] = OUT_TP if trailing_active else OUT_SL`
Where do `OUT_TP` and `OUT_SL` come from?
They are from `extreme_price_movements/labeling.py` imports!
Let's see if they are defined in `labeling.py`.
