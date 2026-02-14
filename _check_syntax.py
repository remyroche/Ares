#!/usr/bin/env python3
import sys
for f in ["extreme_price_movements/training.py", "extreme_price_movements/run_ridge_sizer.py", "extreme_price_movements/meta_model.py"]:
    try:
        compile(open(f).read(), f, "exec")
        print(f"{f}: OK")
    except SyntaxError as e:
        print(f"{f}: SYNTAX ERROR: {e}")
        sys.exit(1)
print("All files OK")
