import re

with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

# conf = final_models.get(side, {}).get(kind) -> conf = final_models.get(kind)
content = content.replace("conf = final_models.get(side, {}).get(kind)", "conf = final_models.get(kind)")

# ds_key = f"train_{side}_{kind}_{H_rep}" -> ds_key = f"train_{kind}_{H_rep}"
content = content.replace('ds_key = f"train_{side}_{kind}_{H_rep}"', 'ds_key = f"train_{kind}_{H_rep}"')

# model_name=f"{side}_{kind}_H{H_rep}:{cand_name}" -> model_name=f"{kind}_H{H_rep}:{cand_name}"
content = content.replace('model_name=f"{side}_{kind}_H{H_rep}:{cand_name}"', 'model_name=f"{kind}_H{H_rep}:{cand_name}"')

# ds_key = f"train_{side}_{kind}_{h}" -> ds_key = f"train_{kind}_{h}"
content = content.replace('ds_key = f"train_{side}_{kind}_{h}"', 'ds_key = f"train_{kind}_{h}"')

# k = f"train_{side}_{kind}_{h}" -> k = f"train_{kind}_{h}"
content = content.replace('k = f"train_{side}_{kind}_{h}"', 'k = f"train_{kind}_{h}"')

with open('extreme_price_movements/training.py', 'w') as f:
    f.write(content)

print("Replacement done.")
