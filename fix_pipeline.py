with open("extreme_price_movements/run_pipeline.py", "r") as f:
    content = f.read()

import re
pattern = r"""    for h in horizons:
        required\.extend\(
            \[
                f"train_long_mr_\{h\}",
                f"train_long_tf_\{h\}",
                f"train_short_mr_\{h\}",
                f"train_short_tf_\{h\}",
            \]
        \)"""

replacement = r"""    from extreme_price_movements.strategy_registry import get_strategies
    strategies = get_strategies(cfg)
    for h in horizons:
        for strat in strategies:
            side = strat["trade_side"]
            k = strat["strategy_id"]
            required.append(f"train_{side}_{k}_{h}")"""

content = re.sub(pattern, replacement, content)

with open("extreme_price_movements/run_pipeline.py", "w") as f:
    f.write(content)
print("Updated _label_artifacts_ready in run_pipeline.py")
