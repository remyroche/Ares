with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# Fix undefined `_roll_std` by replacing with `ff.numba_rolling_std` if needed, but it seems there was a helper `_roll_std` defined elsewhere or not ported to this function correctly.
# Wait, `_roll_std` is defined at line 1281: `def _roll_std(name: str, src: pd.DataFrame, window: int) -> pd.DataFrame:`
# It IS defined! Why did Ruff complain?
# Let's check where it is used vs defined.
