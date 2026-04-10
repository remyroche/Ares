import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    code = f.read()

# policy_optimiser reads from `trade_outcomes.parquet` and `oof_predictions.parquet`
# But it does not take `cfg`. Let's see its signature:
# `def run_policy_optimisation(data_root: str, run_id: str, holdout_frac: float = 0.10, cost_pct: float = 0.0005):`

# We need to filter the artifacts it reads, but since we pass limits via cfg and run_policy_optimisation doesn't take cfg, we might need to modify `run_policy_optimisation` or `run_policy_optimiser_step`.
