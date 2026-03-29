with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

old_logic = """def _strategy_bucket_context(
    trade_side: str, model_kind: str, cfg: dict | None = None
) -> tuple:
    \"\"\"Return (candidate_bucket, move_bucket, strategy_label) for (trade_side, model_kind).

    Strategy definitions (cfg['strategies']) are authoritative; legacy 4-way mapping is fallback.
    \"\"\"
    side = str(trade_side).lower()
    kind = str(model_kind).lower()

    for strat in get_strategies(cfg or {}):
        s_side = str(strat.get("trade_side", "")).lower()
        mode = str(strat.get("base_event_trigger", "")).lower()
        if s_side != side:
            continue
        is_tf = mode.endswith("_tf")
        is_mr = mode.endswith("_mr")
        if (kind == "tf" and is_tf) or (kind == "mr" and is_mr):
            move_bucket = "up" if "price_up" in mode else "down"
            cand_filter = "best" if move_bucket == "up" else "worst"
            return cand_filter, move_bucket, str(strat.get("strategy_id", mode))

    # Legacy fallback
    if side == "long":
        cand_filter = "worst" if kind == "mr" else "best"
    else:
        cand_filter = "best" if kind == "mr" else "worst"
    move_bucket = "up" if cand_filter == "best" else "down"
    label_map = {
        ("long", "mr"): "buy_dips",
        ("long", "tf"): "buy_momentum",
        ("short", "mr"): "sell_rips",
        ("short", "tf"): "sell_weakness",
    }
    return cand_filter, move_bucket, label_map.get((side, kind), f"{side}_{kind}")"""

new_logic = """def _strategy_bucket_context(
    trade_side: str, strategy_id: str, cfg: dict | None = None
) -> tuple:
    \"\"\"Return (candidate_bucket, move_bucket, strategy_label) for (trade_side, strategy_id).

    Strategy definitions (cfg['strategies']) are authoritative; legacy mapping is fallback.
    \"\"\"
    side = str(trade_side).lower()
    strat_id = str(strategy_id)

    for strat in get_strategies(cfg or {}):
        s_side = str(strat.get("trade_side", "")).lower()
        s_id = str(strat.get("strategy_id", ""))
        mode = str(strat.get("base_event_trigger", "")).lower()
        if s_side == side and s_id == strat_id:
            move_bucket = "up" if "price_up" in mode else "down"
            cand_filter = "best" if move_bucket == "up" else "worst"
            return cand_filter, move_bucket, s_id

    # Legacy fallback
    # To retain backwards compat for existing keys that might just be "mr" or "tf"
    is_mr = "mr" in strat_id.lower()
    is_tf = "tf" in strat_id.lower()

    if side == "long":
        cand_filter = "worst" if is_mr else "best"
    else:
        cand_filter = "best" if is_mr else "worst"

    move_bucket = "up" if cand_filter == "best" else "down"
    return cand_filter, move_bucket, strat_id"""

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    with open('extreme_price_movements/training.py', 'w') as f:
        f.write(content)
    print("Patched _strategy_bucket_context definition!")
else:
    print("Could not find _strategy_bucket_context definition!")
