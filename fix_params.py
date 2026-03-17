import re
with open("extreme_price_movements/offline_optimisers/params_store.py", "r") as f:
    content = f.read()

new_func = """def load_inference_candidate_mask_params_per_bucket() -> list[dict[str, Any]]:
    \"\"\"Load all dynamically generated strategy parameters from the mask-optimiser.\"\"\"
    path = Path("production_lgbm_outputs") / "combined_accepted_rule_registry.csv"
    if not path.exists():
        path = REPORTS_DIR / "lgbm_accepted_rule_registry.csv"
    if not path.exists():
        return []

    import pandas as pd
    df = pd.read_csv(path)
    if df.empty:
        return []

    strategies = []
    for _, row in df.iterrows():
        key = str(row.get("canonical_key", ""))
        side = str(row.get("side", "long")).lower()
        if side == "mixed":
            side = "long" # fallback
        if not key:
            continue

        import re
        safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', key)
        # remove duplicate underscores
        safe_id = re.sub(r'_+', '_', safe_id)
        # trim trailing underscore
        safe_id = safe_id.strip('_')

        strategies.append({
            "strategy_id": safe_id,
            "trade_side": side,
            "base_event_trigger": key,
            "mask_params": {"canonical_key": key}
        })
    return strategies
"""

content = re.sub(
    r"def load_inference_candidate_mask_params_per_bucket\(\) -> list\[dict\[str, Any\]\]:.*?return strategies",
    new_func,
    content,
    flags=re.DOTALL
)

with open("extreme_price_movements/offline_optimisers/params_store.py", "w") as f:
    f.write(content)
