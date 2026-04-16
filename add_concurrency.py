import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

# Implement _fast_concurrency_mask right before run_policy_optimisation
concurrency_func = """
from typing import Tuple

def _fast_concurrency_mask(
    entry_timestamps: np.ndarray,
    exit_timestamps: np.ndarray,
    symbols: np.ndarray,
    scores: np.ndarray,
    max_global_concurrent: int = 3,
) -> np.ndarray:
    \"\"\"
    Generates a boolean mask indicating which trades should be executed,
    enforcing that no two trades on the same symbol overlap, and that
    a maximum of `max_global_concurrent` trades can be open globally at any time.
    Trades are processed chronologically. In case of exact timestamp ties, higher confidence wins.
    \"\"\"
    n_trades = len(entry_timestamps)
    mask = np.zeros(n_trades, dtype=bool)

    if n_trades == 0:
        return mask

    # Sort trades chronologically, then by score (descending)
    sort_idx = np.lexsort((-scores, entry_timestamps))

    active_trades = []  # List to store tuples of (exit_time, symbol)

    for idx in sort_idx:
        entry_t = entry_timestamps[idx]
        exit_t = exit_timestamps[idx]
        sym = symbols[idx]

        # Remove trades that have already exited before or exactly at current entry time
        active_trades = [t for t in active_trades if t[0] > entry_t]

        # Check symbol constraint
        symbol_already_active = any(t[1] == sym for t in active_trades)

        # Check global constraint
        global_cap_reached = len(active_trades) >= max_global_concurrent

        if not symbol_already_active and not global_cap_reached:
            mask[idx] = True
            active_trades.append((exit_t, sym))

    return mask

"""

content = content.replace("def run_policy_optimisation(", concurrency_func + "def run_policy_optimisation(")

with open("extreme_price_movements/policy_optimiser.py", "w") as f:
    f.write(content)
