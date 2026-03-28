import re

with open("extreme_price_movements/training.py", "r") as f:
    content = f.read()

# Fix the "for side in trade_sides" and "for k in kinds" nested loops
# This happens in train_models_from_artifacts where it iterates over base models
old_base_loop = '''    if train_base:
        for side in trade_sides:
            final_models[side] = {}
            for k in kinds:
                best_ic = -1.0
                best_m = None
                per_h_models = {}
                feature_selection_by_h = {}
                horizons = cfg["label_horizons_hours"]

                for H in horizons:
                    key = f"train_{k}_{H}"'''

new_base_loop = '''    if train_base:
        strategies = get_strategies(cfg)
        for strat in strategies:
            side = strat["trade_side"]
            k = strat["strategy_id"]
            if side not in final_models:
                final_models[side] = {}

            # Using same indentation logic to minimize diffs
            if True:
                best_ic = -1.0
                best_m = None
                per_h_models = {}
                feature_selection_by_h = {}
                horizons = cfg["label_horizons_hours"]

                for H in horizons:
                    key = f"train_{k}_{H}"'''

content = content.replace(old_base_loop, new_base_loop)

with open("extreme_price_movements/training.py", "w") as f:
    f.write(content)
