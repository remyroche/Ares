with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

old_logic_3 = """        if len(indices) < 50:
            tprint("Not enough events, using defaults.")
            is_mr_default = "mr" in k.lower()
            default_risk = {"""

new_logic_3 = """        if len(indices) < 50:
            tprint("Not enough events, using defaults.")
            is_mr_default = strat.get("is_mr", False)
            default_risk = {"""

content = content.replace(old_logic_3, new_logic_3)

old_logic_4 = """        # Per-bucket max hold hours: MR = shorter (reversion is fast), TF = longer
        is_mr = "mr" in k.lower()
        bucket_hold = 12 if is_mr else 24"""

new_logic_4 = """        # Per-bucket max hold hours: MR = shorter (reversion is fast), TF = longer
        is_mr = strat.get("is_mr", False)
        bucket_hold = 12 if is_mr else 24"""

content = content.replace(old_logic_4, new_logic_4)

with open('extreme_price_movements/training.py', 'w') as f:
    f.write(content)
print("Patched MR checks in optimize_risk_params")
