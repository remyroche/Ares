import re

with open("extreme_price_movements/config.py", "r") as f:
    content = f.read()

# Replace meta_asym_feature_keys: [] with the correct keys
old_asym = '"meta_asym_feature_keys": [],'
new_asym = '''"meta_asym_feature_keys": [
        "vol_asym",
        "vol_asym_6",
        "tail_asymmetry_q90_q10_atr_norm",
        "dir_path_risk_skew_2h",
        "tail_against",
        "asym_ratio",
        "asym_ft",
        "vol_shock_asym_8_24"
    ],'''

content = content.replace(old_asym, new_asym)

with open("extreme_price_movements/config.py", "w") as f:
    f.write(content)

with open("extreme_price_movements/training_utils.py", "r") as f:
    content2 = f.read()

# Update get_meta_feature_keys for asym
old_asym_logic = '''    elif head == "asym":
        mfe_specific = expand_feature_group_refs(cfg.get("meta_mfe_feature_keys", []), cfg)
        mae_specific = expand_feature_group_refs(cfg.get("meta_mae_feature_keys", []), cfg)
        return dedupe_keep_order(shared + mfe_specific + mae_specific)'''

new_asym_logic = '''    elif head == "asym":
        mfe_specific = expand_feature_group_refs(cfg.get("meta_mfe_feature_keys", []), cfg)
        mae_specific = expand_feature_group_refs(cfg.get("meta_mae_feature_keys", []), cfg)
        asym_specific = expand_feature_group_refs(cfg.get("meta_asym_feature_keys", []), cfg)
        return dedupe_keep_order(shared + mfe_specific + mae_specific + asym_specific)'''

content2 = content2.replace(old_asym_logic, new_asym_logic)

with open("extreme_price_movements/training_utils.py", "w") as f:
    f.write(content2)
