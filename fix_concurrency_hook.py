import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

# Hook the fast concurrency mask into _simulate_policy
# Need to find:
#         sizes = None
#         if rets is not None:
#             sizes = _get_position_sizes(rets, params_to_use, evaluate_mask)
#         return rets, sizes

new_logic = """        sizes = None
        if rets is not None:
            # Apply concurrency logic
            if "timestamps_ms" in context and "symbol" in context:
                # We need exit timestamps for concurrency. We can approximate them using lengths.
                # If length is nan, trade wasn't taken or didn't finish.
                lengths = context.get("_cached_lengths_")
                if lengths is None:
                    # Very rough fallback if length calculation is unavailable in simulation scope
                    lengths = np.full(len(rets), 24, dtype=np.float32)

                # Approximate exit ms
                bar_ms = 3600000  # assuming 1h bars, rough heuristic
                exit_ms = context["timestamps_ms"] + (np.nan_to_num(lengths, nan=0.0) * bar_ms).astype(np.int64)

                conc_mask = _fast_concurrency_mask(
                    context["timestamps_ms"],
                    exit_ms,
                    context["symbol"],
                    context.get("confidence", np.zeros(len(rets))),
                )

                # Zero out returns for trades rejected by concurrency
                rets = np.where(conc_mask, rets, np.nan)

            sizes = _get_position_sizes(rets, params_to_use, evaluate_mask)
        return rets, sizes"""

content = content.replace(
"""        sizes = None
        if rets is not None:
            sizes = _get_position_sizes(rets, params_to_use, evaluate_mask)
        return rets, sizes""",
    new_logic
)

with open("extreme_price_movements/policy_optimiser.py", "w") as f:
    f.write(content)
