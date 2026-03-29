import re

with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

old_loop_1 = """    # 2. Train Alpha Models
    # directions (up/down) replaced by sides (long/short)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    final_models = {}
    base_variant_models = {}"""

new_loop_1 = """    # 2. Train Alpha Models
    final_models = {}
    base_variant_models = {}"""

content = content.replace(old_loop_1, new_loop_1)


old_return = """    return {
        "alpha_models": final_models,
        "alpha_oof_metrics": {
            f"{side}_{kind}": ((final_models.get(side) or {}).get(kind) or {}).get(
                "alpha_diag", {}
            )
            for side in trade_sides
            for kind in kinds
        },
        "exh_models": exh_models,"""

new_return = """    # Build alpha metrics correctly from dynamic strategies
    alpha_metrics = {}
    for side, side_models in final_models.items():
        for kind, kind_model in side_models.items():
            alpha_metrics[f"{side}_{kind}"] = kind_model.get("alpha_diag", {})

    return {
        "alpha_models": final_models,
        "alpha_oof_metrics": alpha_metrics,
        "exh_models": exh_models,"""

content = content.replace(old_return, new_return)

with open('extreme_price_movements/training.py', 'w') as f:
    f.write(content)
print("Patched loops in train_models_from_artifacts")
