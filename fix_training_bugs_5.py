with open("extreme_price_movements/training.py", "r") as f:
    code = f.read()

code = code.replace("pass #\n        y_pred, 0.30, np.asarray(groups) if groups is not None else None\n    )", "pass")
code = code.replace("pass #\n        y_pred, 0.30, np.asarray(groups) if groups is not None else None\n", "pass")

# _PKF still undefined somehow? Let's check
import re

with open("extreme_price_movements/training.py", "w") as f:
    f.write(code)
