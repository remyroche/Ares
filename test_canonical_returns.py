import extreme_price_movements.config as cfg

# Let's inspect MODEL_FEATURES and other feature keys for return names.
print("MODEL_FEATURES returns:")
print([x for x in cfg.MODEL_FEATURES if 'ret' in x])

print("HELPER_BASE_FEATURES returns:")
print([x for x in cfg.HELPER_BASE_FEATURES if 'ret' in x])

print("base_long_feature_keys returns:")
print([x for x in cfg.base_long_feature_keys if 'ret' in x])

print("meta_reg_feature_keys returns:")
print([x for x in cfg.meta_reg_feature_keys if 'ret' in x])
