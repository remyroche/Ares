with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

old_logic = """                    # Filter features strictly for the Alpha Model (exclude meta-only features)
                    # We need to know which feature_key was used.
                    # k is "mr" or "tf"
                    feat_key_name = (
                        "tf_feature_keys" if k == "tf" else "mr_feature_keys"
                    )

                    allowed_keys = set(cfg.get(feat_key_name, []))"""

new_logic = """                    # Filter features strictly for the Alpha Model (exclude meta-only features)
                    # We need to know which feature_key was used.
                    # k is strategy_id
                    allowed_keys = set(strategy.get("feature_keys", []))"""

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    with open('extreme_price_movements/training.py', 'w') as f:
        f.write(content)
    print("Patched alpha feature key selection!")
else:
    print("Could not find old_logic for alpha feature keys!")
