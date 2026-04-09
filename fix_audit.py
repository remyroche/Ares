with open("extreme_price_movements/training_utils.py", "r") as f:
    content = f.read()

# Update audit_feature_coverage to accept all_feature_columns directly
old_audit_def = "def audit_feature_coverage(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, List[str]]:"
new_audit_def = "def audit_feature_coverage(all_feature_columns: List[str], cfg: Dict[str, Any]) -> Dict[str, List[str]]:"

old_audit_loop = """    all_cols = []
    for c in df.columns:
        if c in {"id", "timestamp", "symbol"}:
            continue
        if any(c.startswith(p) for p in exclude_prefixes) and not c.startswith("p_vol_high") and not c.startswith("p_cusum_high") and not c.startswith("p_liq_low"):
            continue
        all_cols.append(c)"""

new_audit_loop = """    all_cols = []
    for c in all_feature_columns:
        if c in {"id", "timestamp", "symbol"}:
            continue
        if any(c.startswith(p) for p in exclude_prefixes) and not c.startswith("p_vol_high") and not c.startswith("p_cusum_high") and not c.startswith("p_liq_low"):
            continue
        all_cols.append(c)"""

old_audit_returns = """    return {
        "base_unused": base_unused,
        "meta_unused": meta_unused,
        "global_unused": global_unused,
        "stale_orphans": stale_orphans
    }"""

new_audit_returns = """    computed_but_unused = sorted(list(set(all_cols) - global_all))
    configured_but_missing = sorted(list(global_all - set(all_cols)))

    return {
        "computed_but_unused": computed_but_unused,
        "configured_but_missing": configured_but_missing,
        "base_unused": base_unused,
        "meta_unused": meta_unused,
        "stale_orphans": stale_orphans
    }"""

content = content.replace(old_audit_def, new_audit_def)
content = content.replace(old_audit_loop, new_audit_loop)
content = content.replace(old_audit_returns, new_audit_returns)

with open("extreme_price_movements/training_utils.py", "w") as f:
    f.write(content)
