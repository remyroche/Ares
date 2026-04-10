import re

with open("extreme_price_movements/run_pipeline.py", "r") as f:
    code = f.read()

# I want to check how `run_optimise` and `run_policy_optimiser_step` handle artifacts.
# run_policy_optimiser_step is in `pipeline_steps.py`.
# run_optimise is in `optimise.py`.
