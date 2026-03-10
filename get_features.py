import re

with open('extreme_price_movements/features.py') as f:
    text = f.read()

feats = set(re.findall(r'feats\["([^"]+)"\]', text))
feats.update(re.findall(r"feats\['([^']+)'\]", text))
feats.update(re.findall(r"feats\[f\"([^\"]+)\"\]", text))

# Print all feature names
for feat in sorted(feats):
    print(feat)
