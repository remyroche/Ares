import re

with open("extreme_price_movements/training.py", "r") as f:
    text = f.read()

# 1. build_grid_aggregated_tb_cache
code_to_replace_tb1 = """def build_grid_aggregated_tb_cache(panel, feats, cfg, horizons, trade_sides):"""
new_code_tb1 = """def build_grid_aggregated_tb_cache(panel, feats, cfg, horizons):"""
if code_to_replace_tb1 in text:
    print("Replacing tb1")
    text = text.replace(code_to_replace_tb1, new_code_tb1)

code_to_replace_tb2 = """    # Cache raw triple barrier results per (H, side, k_tp, sl_base_mult) to avoid recomputation.
    _raw_tb_cache = {}
    _kinds = ["mr", "tf"]
    _prod_events_rows = []"""
new_code_tb2 = """    # Cache raw triple barrier results per (H, side, k_tp, sl_base_mult) to avoid recomputation.
    _raw_tb_cache = {}

    _prod_events_rows = []"""
if code_to_replace_tb2 in text:
    print("Replacing tb2")
    text = text.replace(code_to_replace_tb2, new_code_tb2)

code_to_replace_tb3 = """    # ── ALPHA MODELS (long/short × mr/tf × horizons) ──
    # Note: Using horizons=horizons for explicit control.
    horizons = horizons or list(CANON_HORIZONS)
    for side in trade_sides:
        for k_label in _kinds:
            for H in horizons:"""
new_code_tb3 = """    # ── ALPHA MODELS ──
    # Note: Using horizons=horizons for explicit control.
    horizons = horizons or list(CANON_HORIZONS)
    from extreme_price_movements.strategy_registry import get_strategies
    strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        k_label = strat["strategy_id"]
        for H in horizons:"""
if code_to_replace_tb3 in text:
    print("Replacing tb3")
    text = text.replace(code_to_replace_tb3, new_code_tb3)

# Replace the calls
code_to_replace_tb4 = """        _tb_new, _geom_new = build_grid_aggregated_tb_cache(
            panel, feats, cfg, horizons=horizons, trade_sides=trade_sides
        )"""
new_code_tb4 = """        _tb_new, _geom_new = build_grid_aggregated_tb_cache(
            panel, feats, cfg, horizons=horizons
        )"""
if code_to_replace_tb4 in text:
    print("Replacing tb4")
    text = text.replace(code_to_replace_tb4, new_code_tb4)

code_to_replace_tb5 = """        _tb_all, _geom_all = build_grid_aggregated_tb_cache(
            panel, feats, cfg, horizons=horizons, trade_sides=trade_sides
        )"""
new_code_tb5 = """        _tb_all, _geom_all = build_grid_aggregated_tb_cache(
            panel, feats, cfg, horizons=horizons
        )"""
if code_to_replace_tb5 in text:
    print("Replacing tb5")
    text = text.replace(code_to_replace_tb5, new_code_tb5)


# 2. train_meta_models_from_artifacts
code_to_replace_meta1 = """        # Ensure side-level meta models see all same-side base outputs (TF + MR)
        for k_other in kinds:"""
new_code_meta1 = """        # Ensure side-level meta models see all same-side base outputs
        from extreme_price_movements.strategy_registry import get_strategies
        available_kinds = [s["strategy_id"] for s in get_strategies(cfg) if s["trade_side"] == side]
        for k_other in available_kinds:"""
if code_to_replace_meta1 in text:
    print("Replacing meta1")
    text = text.replace(code_to_replace_meta1, new_code_meta1)


# 3. train_models_from_artifacts (exh, alpha metrics)
code_to_replace_train1 = """def train_models_from_artifacts(datasets, cfg, train_meta=True, train_base=True):
    tprint(f"Entering function: train_models_from_artifacts in training.py")
    tprint(f"train_base={train_base}, train_meta={train_meta}")
    cfg = _resolve_training_cfg_with_offline_optimisers(cfg)
    directions = ["up", "down"]
    kinds = ["mr", "tf"]
    final_models = {}"""
new_code_train1 = """def train_models_from_artifacts(datasets, cfg, train_meta=True, train_base=True):
    tprint(f"Entering function: train_models_from_artifacts in training.py")
    tprint(f"train_base={train_base}, train_meta={train_meta}")
    cfg = _resolve_training_cfg_with_offline_optimisers(cfg)


    final_models = {}"""
if code_to_replace_train1 in text:
    print("Replacing train1")
    text = text.replace(code_to_replace_train1, new_code_train1)

code_to_replace_train2 = """    exh_models = {}
    if train_base:
        for d in directions:
            key = f"exh_{d}\""""
new_code_train2 = """    exh_models = {}
    if train_base:
        for d in ["up", "down"]:
            key = f"exh_{d}\""""
if code_to_replace_train2 in text:
    print("Replacing train2")
    text = text.replace(code_to_replace_train2, new_code_train2)

code_to_replace_train3 = """    # 2. Train Alpha Models
    # directions (up/down) replaced by sides (long/short)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]
    final_models = {}
    base_variant_models = {}"""
new_code_train3 = """    # 2. Train Alpha Models
    # directions (up/down) replaced by sides (long/short)


    final_models = {}
    base_variant_models = {}"""
if code_to_replace_train3 in text:
    print("Replacing train3")
    text = text.replace(code_to_replace_train3, new_code_train3)

code_to_replace_train4 = """        "alpha_oof_metrics": {
            f"{side}_{kind}": ((final_models.get(side) or {}).get(kind) or {}).get(
                "alpha_diag", {}
            )
            for side in trade_sides
            for kind in kinds
        },"""
new_code_train4 = """        "alpha_oof_metrics": {
            f"{s['trade_side']}_{s['strategy_id']}": ((final_models.get(s['trade_side']) or {}).get(s['strategy_id']) or {}).get(
                "alpha_diag", {}
            )
            for s in get_strategies(cfg)
        },"""
if code_to_replace_train4 in text:
    print("Replacing train4")
    text = text.replace(code_to_replace_train4, new_code_train4)

# 4. optimize_risk_params
code_to_replace_opt1 = """    # 2. Iterate over strategies (buckets)
    trade_sides = ["long", "short"]
    kinds = ["mr", "tf"]"""
new_code_opt1 = """    # 2. Iterate over strategies (buckets)
    from extreme_price_movements.strategy_registry import get_strategies
    strategies = get_strategies(cfg)"""
if code_to_replace_opt1 in text:
    print("Replacing opt1")
    text = text.replace(code_to_replace_opt1, new_code_opt1)

code_to_replace_opt2 = """    # Optimize per side/kind
    for side in trade_sides:
        for k in kinds:"""
new_code_opt2 = """    # Optimize per side/kind
    for strat in strategies:
        for side, k in [(strat["trade_side"], strat["strategy_id"])]:"""
if code_to_replace_opt2 in text:
    print("Replacing opt2")
    text = text.replace(code_to_replace_opt2, new_code_opt2)

# 5. base geometry variant training in train_models_from_artifacts
code_to_replace_var1 = """        if bool(cfg.get("base_geometry_train_variants", True)):
            tprint("Training grouped base-geometry variant models (tight/wide)...")
            _run_id = cfg.get("run_id", "default")
            oof_dir = os.path.join(cfg["data_root"], "artifacts", _run_id, "oof")
            os.makedirs(oof_dir, exist_ok=True)
            for side in trade_sides:
                for k in kinds:"""
new_code_var1 = """        if bool(cfg.get("base_geometry_train_variants", True)):
            tprint("Training grouped base-geometry variant models (tight/wide)...")
            _run_id = cfg.get("run_id", "default")
            oof_dir = os.path.join(cfg["data_root"], "artifacts", _run_id, "oof")
            os.makedirs(oof_dir, exist_ok=True)
            from extreme_price_movements.strategy_registry import get_strategies
            strats = get_strategies(cfg)
            for strat in strats:
                for side, k in [(strat["trade_side"], strat["strategy_id"])]:"""
if code_to_replace_var1 in text:
    print("Replacing var1")
    text = text.replace(code_to_replace_var1, new_code_var1)

with open("extreme_price_movements/training.py", "w") as f:
    f.write(text)
