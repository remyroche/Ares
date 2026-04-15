import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

# Modify _simulate_barwise_path_policy to compute exit_lengths
new_simulate_logic = """
    # We will keep track of exactly what bar a trade exits to compute time-in-market
    exit_lengths = np.full(n_trades, max_bars, dtype=np.int32)

    for bar in range(max_bars):
        active = (~exited) & (bar < future_lengths)
"""
content = content.replace(
"""
    for bar in range(max_bars):
        active = (~exited) & (bar < future_lengths)""",
    new_simulate_logic
)

new_simulate_exit = """
        exit_now = np.isfinite(bar_exit)
        if np.any(exit_now):
            exit_idx = idx[exit_now]
            exit_rets[exit_idx] = bar_exit[exit_now]
            exited[exit_idx] = True
            exit_lengths[exit_idx] = bar + 1

    # Inject lengths into context so downstream components can use them
    context["_cached_lengths_"] = exit_lengths

    return exit_rets.astype(np.float32)
"""
content = content.replace(
"""
        exit_now = np.isfinite(bar_exit)
        if np.any(exit_now):
            exit_idx = idx[exit_now]
            exit_rets[exit_idx] = bar_exit[exit_now]
            exited[exit_idx] = True

    return exit_rets.astype(np.float32)
""",
    new_simulate_exit
)

with open("extreme_price_movements/policy_optimiser.py", "w") as f:
    f.write(content)
