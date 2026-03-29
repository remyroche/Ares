with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

old_logic = """                        _variant_fit = _train_base_variant_dataset(
                            side, k, H, ds_key, datasets[ds_key]
                        )"""

new_logic = """                        _variant_fit = _train_base_variant_dataset(
                            side, k, H, ds_key, datasets[ds_key], strategy=strat
                        )"""

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    with open('extreme_price_movements/training.py', 'w') as f:
        f.write(content)
    print("Patched base variant call!")
else:
    print("Could not find base variant call!")
