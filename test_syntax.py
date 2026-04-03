import py_compile

try:
    py_compile.compile("extreme_price_movements/lgbm_based_mask_generation.py", doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print(f"Syntax error: {e}")
