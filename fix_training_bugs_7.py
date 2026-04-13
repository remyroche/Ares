with open("extreme_price_movements/training.py", "r") as f:
    code = f.read()

code = code.replace(
    "f\"[LABEL_DIAG][PRE_TO_FILTER] side={side} kind={k} H={H} n={_n_all} \"",
    "f\"[LABEL_DIAG][PRE_TO_FILTER] side={side} H={H} n={_n_all} \""
)
code = code.replace(
    "f\"[LABEL_DIAG][POST_TO_FILTER] side={side} kind={k} H={H} n={_n_kept} \"",
    "f\"[LABEL_DIAG][POST_TO_FILTER] side={side} H={H} n={_n_kept} \""
)
code = code.replace("import json\n                        import json", "import json")

with open("extreme_price_movements/training.py", "w") as f:
    f.write(code)
