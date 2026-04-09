with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

content = content.replace("from numba import njit\n", "from numba import njit, prange\nfrom typing import Dict, List, Tuple\n")

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
