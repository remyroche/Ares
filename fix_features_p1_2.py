with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

import re

new_function = """def compute_features_hourly(panel, mkt_gates, cfg, requested_feature_keys=None):
    \"\"\"
    Compute features. Joblib caching removed — features are persisted to parquet
    by save_features, and the joblib serialization doubled peak memory.
    \"\"\"
    if requested_feature_keys is None:
        import extreme_price_movements.training_utils as tu
        import extreme_price_movements.config as cfg_mod

        all_keys = set()

        # Base features
        all_keys.update(tu.get_base_feature_keys("long", cfg))
        all_keys.update(tu.get_base_feature_keys("short", cfg))

        # Meta features
        for head in ["reg", "clf", "mfe", "mae", "asym"]:
            all_keys.update(tu.get_meta_feature_keys(head, cfg))

        # Other runtimes
        for group in ["RIDGE_FEATURE_COLS", "CONTINUOUS_LOCATION_COLS", "TEST_FEATURE_KEYS", "FEATURE_SELECTION_KEYS", "TRAINING_RESIDUALIZATION_FEATURE_KEYS", "MODEL_FEATURES"]:
            if group in cfg_mod.CFG:
                all_keys.update(tu.expand_feature_group_refs(cfg_mod.CFG[group], cfg))
            elif hasattr(cfg_mod, group):
                all_keys.update(tu.expand_feature_group_refs(getattr(cfg_mod, group), cfg))

        requested_feature_keys = list(all_keys)

    return _compute_features_impl(
        panel, mkt_gates, cfg, requested_feature_keys=requested_feature_keys
    )"""

content = re.sub(r'def compute_features_hourly\(panel, mkt_gates, cfg, requested_feature_keys=None\):.*?return _compute_features_impl\(\n\s*panel, mkt_gates, cfg, requested_feature_keys=requested_feature_keys\n\s*\)', new_function, content, flags=re.DOTALL)

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
