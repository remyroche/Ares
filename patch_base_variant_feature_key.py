with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

old_logic = """    def _train_base_variant_dataset(
        side_name, kind_name, horizon, dataset_key, df_variant
    ):"""

new_logic = """    def _train_base_variant_dataset(
        side_name, kind_name, horizon, dataset_key, df_variant, strategy=None
    ):"""

content = content.replace(old_logic, new_logic)

old_logic_2 = """        X = df_variant.drop(columns=[c for c in drop_cols if c in df_variant.columns])
        feat_key_name = "tf_feature_keys" if kind_name == "tf" else "mr_feature_keys"
        allowed_keys = set(cfg.get(feat_key_name, []))"""

new_logic_2 = """        X = df_variant.drop(columns=[c for c in drop_cols if c in df_variant.columns])
        # Use strategy dict to extract feature keys
        allowed_keys = set(strategy.get("feature_keys", [])) if strategy else set()"""

if old_logic_2 in content:
    content = content.replace(old_logic_2, new_logic_2)
    with open('extreme_price_movements/training.py', 'w') as f:
        f.write(content)
    print("Patched base variant feature key selection!")
else:
    print("Could not find old_logic_2 for base variant feature keys!")
