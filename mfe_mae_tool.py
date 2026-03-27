import re
with open("extreme_price_movements/labeling.py", "r") as f:
    c = f.read()
    if "def _numba_triple_barrier_fast(" in c:
        print(c.split("def _numba_triple_barrier_fast(")[1][:1000])
