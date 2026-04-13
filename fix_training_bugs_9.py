with open("extreme_price_movements/training.py", "r") as f:
    code = f.read()

code = code.replace(
    "from sklearn.model_selection import PurgedKFold\n                    inner = PurgedKFold(",
    "from extreme_price_movements.optimise_tpsl_ratio import PurgedKFold\n                    inner = PurgedKFold("
)

with open("extreme_price_movements/training.py", "w") as f:
    f.write(code)
