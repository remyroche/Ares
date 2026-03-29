import re

with open("extreme_price_movements/training.py", "r") as f:
    content = f.read()

# Replace hardcoded base archetypes in train_models_from_artifacts
old_archetypes_get = '''"base_geometry_archetypes", ["tight", "balanced", "wide"]'''
new_archetypes_get = '''"base_geometry_archetypes", ["tight", "wide"]'''

content = content.replace(old_archetypes_get, new_archetypes_get)

# Also remove the continue for balanced
old_variant_balanced = '''                        variant = str(variant)
                        if variant == "balanced":
                            continue
                        ds_key = f"train_{k}_{H}_{variant}"'''
new_variant_balanced = '''                        variant = str(variant)
                        ds_key = f"train_{k}_{H}_{variant}"'''

content = content.replace(old_variant_balanced, new_variant_balanced)

# 3. Fix the "for side in trade_sides" and "for k in kinds" nested loops
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
                    key = f"train_{k}_{H}"
                    if key not in datasets:
                        continue'''

new_base_loop = '''    if train_base:
        strategies = get_strategies(cfg)
        for strat in strategies:
            side = strat["trade_side"]
            k = strat["strategy_id"]
            if side not in final_models:
                final_models[side] = {}

            best_ic = -1.0
            best_m = None
            per_h_models = {}
            feature_selection_by_h = {}
            horizons = cfg["label_horizons_hours"]

            for H in horizons:
                key = f"train_{k}_{H}"
                if key not in datasets:
                    continue'''

content = content.replace(old_base_loop, new_base_loop)

# Also need to fix the indentation in that whole block that was previously indented for `side` then `k`.
# But wait, looking at `old_base_loop`, replacing it will change `for side... for k...` to `for strat... side=... k=...`
# Which removes one level of indentation. We need to be careful with indentation.
# Let's just do a regex replace or AST manipulation to be safe, or manually dedent.
