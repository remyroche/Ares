import re

with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

# Make sure any calls use strategy_id instead of kind. In train_models_from_artifacts, it passes k which is strategy_id now.
# So we don't need to change much if the variable names just align.
old_call_1 = """                    trade_side = side
                    cand_filter, move_bucket, strategy_label = _strategy_bucket_context(
                        trade_side, k
                    )"""

new_call_1 = """                    trade_side = side
                    cand_filter, move_bucket, strategy_label = _strategy_bucket_context(
                        trade_side, k, cfg
                    )"""

old_call_2 = """        trade_side = side
        cand_filter, move_bucket, strategy_label = _strategy_bucket_context(
            trade_side, k
        )"""

new_call_2 = """        trade_side = side
        cand_filter, move_bucket, strategy_label = _strategy_bucket_context(
            trade_side, k, cfg
        )"""

content = content.replace(old_call_1, new_call_1)
content = content.replace(old_call_2, new_call_2)

with open('extreme_price_movements/training.py', 'w') as f:
    f.write(content)

print("Patched _strategy_bucket_context calls!")
