with open("extreme_price_movements/training.py", "r") as f:
    code = f.read()

old_executor = """    n_workers = _choose_parallel_cells(len(tasks), cfg)
    if bool(cfg.get("label_parallel_enable", True)) and n_workers > 1:"""

new_executor = """    # Enforce maximum of 2 workers as requested for speed/memory tradeoff
    n_workers = min(2, _choose_parallel_cells(len(tasks), cfg))

    # We want to enable parallel execution if n_workers > 1,
    # regardless of legacy label_parallel_enable setting, as long as the user hasn't explicitly disabled it
    parallel_enabled = bool(cfg.get("label_parallel_enable", True)) or n_workers > 1

    if parallel_enabled and n_workers > 1:"""

code = code.replace(old_executor, new_executor)

with open("extreme_price_movements/training.py", "w") as f:
    f.write(code)
