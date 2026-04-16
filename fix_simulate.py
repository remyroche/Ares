import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

# Make sure _simulate_policy returns lengths so we can accurately build the exit timestamps
# Change replay_exit_policy to return (rets, lengths)
# But wait, replay_exit_policy just returns rets right now.
