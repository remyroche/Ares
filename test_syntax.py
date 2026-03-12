import py_compile
import sys

try:
    py_compile.compile("extreme_price_movements/mask_optimiser.py", doraise=True)
    print("Syntax OK")
except Exception as e:
    print(e)
    sys.exit(1)
