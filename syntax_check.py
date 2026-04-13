import py_compile
try:
    py_compile.compile("extreme_price_movements/pipeline_steps.py", doraise=True)
    print("Syntax is OK")
except Exception as e:
    print(f"Syntax error: {e}")
