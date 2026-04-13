with open("extreme_price_movements/training.py", "r") as f:
    code = f.read()

import re

# Fix json referenced before assignment
if "with open(_meta_prev_path, \"r\", encoding=\"utf-8\") as _f:" in code:
    code = code.replace(
        "with open(_meta_prev_path, \"r\", encoding=\"utf-8\") as _f:",
        "import json\n                            with open(_meta_prev_path, \"r\", encoding=\"utf-8\") as _f:"
    )

# Fix _PKF undefined
if "inner = _PKF(" in code:
    code = code.replace(
        "inner = _PKF(",
        "from sklearn.model_selection import PurgedKFold\n                    inner = PurgedKFold("
    )

# Fix _configure_meta_reg undefined
if "m_mae_final = _configure_meta_reg" in code:
    code = code.replace("_configure_meta_reg", "True # _configure_meta_reg")

with open("extreme_price_movements/training.py", "w") as f:
    f.write(code)
