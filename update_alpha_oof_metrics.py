import re

with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

# Original dict comprehension:
#         "alpha_oof_metrics": {
#             f"{side}_{kind}": ((final_models.get(side) or {}).get(kind) or {}).get(
#                 "alpha_diag", {}
#             )
#             for side in trade_sides
#             for kind in kinds
#         },

# New one should be based on strategies = get_strategies(cfg) ? Actually `get_strategies` gives us strategy_id, trade_side.
# We can do:
#         "alpha_oof_metrics": {
#             strat["strategy_id"]: (final_models.get(strat["strategy_id"]) or {}).get(
#                 "alpha_diag", {}
#             )
#             for strat in get_strategies(cfg)
#         },
# Or maybe the key should still be f"{side}_{kind}" if it's expected by downstream but kind is strategy_id here. Wait, what does the key say? f"{side}_{kind}". If kind is already "long_tf" then side_kind becomes "long_long_tf" which is weird. No, wait, if strategy_id is "long_tf", then kind is "long_tf", side is "long".
# Ah, earlier in the code:
# side = strat["trade_side"]
# kind = strat["strategy_id"]
# so kind IS "long_tf". The original key f"{side}_{kind}" for "long_tf" (where side='long', kind='tf') was "long_tf".
# So the new key should just be `kind` (the strategy_id).

replacement = """        "alpha_oof_metrics": {
            strat["strategy_id"]: (final_models.get(strat["strategy_id"]) or {}).get(
                "alpha_diag", {}
            )
            for strat in get_strategies(cfg)
        },"""

pattern = r'"alpha_oof_metrics": \{\s*f"\{side\}_\{kind\}": \(\(final_models\.get\(side\) or \{\}\)\.get\(kind\) or \{\}\)\.get\(\s*"alpha_diag", \{\}\s*\)\s*for side in trade_sides\s*for kind in kinds\s*\},'

new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

with open('extreme_price_movements/training.py', 'w') as f:
    f.write(new_content)

print("Replacement done for alpha_oof_metrics.")
