import re

with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

# Call 1: build_hourly_training_set_and_weights (stage="base")
# Wait, do we have `strat` available here?
# Let's check where build_hourly_training_set_and_weights is called.
# It is called from build_grid_aggregated_tb_cache ?
