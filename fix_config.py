with open("extreme_price_movements/config.py", "r") as f:
    content = f.read()

idx = content.find('"meta_shared_feature_keys": [')
idx2 = content.find("]", idx)
content = content[:idx2] + '        "TRAINING_RESIDUALIZATION_FEATURE_KEYS",\n    ' + content[idx2:]

with open("extreme_price_movements/config.py", "w") as f:
    f.write(content)
