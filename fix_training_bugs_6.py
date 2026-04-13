with open("extreme_price_movements/training.py", "r") as f:
    code = f.read()

code = code.replace(
    "m_mae_final = _configure_meta_reg",
    "m_mae_final = True # _configure_meta_reg"
)
code = code.replace(
    "m_mfe_final = _configure_meta_reg",
    "m_mfe_final = True # _configure_meta_reg"
)
code = code.replace(
    "m_asym_final = _configure_meta_reg",
    "m_asym_final = True # _configure_meta_reg"
)
code = code.replace(
    """                    try:
                        import json
                        with open(_meta_prev_path, "r", encoding="utf-8") as _f:
                            _meta_prev_sel = list(
                                (json.load(_f) or {}).get("selected_features", [])
                            )""",
    """                    try:
                        import json
                        with open(_meta_prev_path, "r", encoding="utf-8") as _f:
                            _meta_prev_sel = list(
                                (json.load(_f) or {}).get("selected_features", [])
                            )"""
)

# Fix side_l undefined issue:
code = code.replace("side_key = f\"{side_l}_H{int(H)}\"", "side_key = f\"{str(side).lower()}_H{int(H)}\"")
code = code.replace("return [side_key, f\"{kind}_{side_l}_H{int(H)}\"]", "return [side_key, f\"{kind}_{str(side).lower()}_H{int(H)}\"]")

# Fix _barrier_factory_cache
code = code.replace(
"""                    if "_barrier_factory_cache" not in dir()
                    else _barrier_factory_cache""",
"""                    if "_barrier_factory_cache" not in locals()
                    else locals()["_barrier_factory_cache"]"""
)

# Fix json load
if "import json" in code:
    pass

with open("extreme_price_movements/training.py", "w") as f:
    f.write(code)
