import re

with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

# Replace compute_weights_logic signature and implementation
old_logic = """def compute_weights_logic(df, cfg, model_kind):
    tprint(f"Entering function: compute_weights_logic in training.py")
    from .model_mr import compute_mr_weights
    from .model_tf import compute_tf_weights

    if model_kind == "mr":
        return compute_mr_weights(df, cfg)
    else:
        return compute_tf_weights(df, cfg)"""

new_logic = """def compute_weights_logic(df, cfg, strategy=None):
    tprint(f"Entering function: compute_weights_logic in training.py")
    from .model_mr import compute_mr_weights
    from .model_tf import compute_tf_weights

    # Assume TF if strategy is not provided or if it's explicitly TF
    is_mr = strategy.get("is_mr", False) if strategy else False

    if is_mr:
        return compute_mr_weights(df, cfg)
    else:
        return compute_tf_weights(df, cfg)"""

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    with open('extreme_price_movements/training.py', 'w') as f:
        f.write(content)
    print("Patched compute_weights_logic!")
else:
    print("Could not find old_logic in training.py!")
