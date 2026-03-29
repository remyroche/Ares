import re

with open("extreme_price_movements/training.py", "r") as f:
    text = f.read()

code_to_replace = """        if bool(cfg.get("base_geometry_train_variants", True)):
            tprint("Training grouped base-geometry variant models (tight/wide)...")
            _run_id = cfg.get("run_id", "default")
            oof_dir = os.path.join(cfg["data_root"], "artifacts", _run_id, "oof")
            os.makedirs(oof_dir, exist_ok=True)
            for side in trade_sides:
                for k in kinds:"""

new_code = """        if bool(cfg.get("base_geometry_train_variants", True)):
            tprint("Training grouped base-geometry variant models (tight/wide)...")
            _run_id = cfg.get("run_id", "default")
            oof_dir = os.path.join(cfg["data_root"], "artifacts", _run_id, "oof")
            os.makedirs(oof_dir, exist_ok=True)
            strats = get_strategies(cfg)
            for strategy in strats:
                for side, k in [(strategy["trade_side"], strategy["strategy_id"])]:"""

if code_to_replace in text:
    print("Found block 2!")
    text = text.replace(code_to_replace, new_code)
    with open("extreme_price_movements/training.py", "w") as f:
        f.write(text)
else:
    print("Block 2 not found!")
