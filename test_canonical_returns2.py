import extreme_price_movements.config as cfg

for name, lst in [
    ("TEST_FEATURE_KEYS", cfg.TEST_FEATURE_KEYS),
    ("MODEL_FEATURES", cfg.MODEL_FEATURES),
    ("HELPER_BASE_FEATURES", cfg.HELPER_BASE_FEATURES),
    ("base_long_feature_keys", cfg.CFG["base_long_feature_keys"]),
    ("base_short_feature_keys", cfg.CFG["base_short_feature_keys"]),
    ("meta_reg_feature_keys", cfg.CFG["meta_reg_feature_keys"]),
]:
    print(f"--- {name} ---")
    print([x for x in lst if "ret" in x])
