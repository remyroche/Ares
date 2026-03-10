import re

with open('extreme_price_movements/meta_model.py', 'r') as f:
    content = f.read()

# check where `feature_names` or `features` are being set up
matches = re.findall(r'def .+\(', content)
for match in matches:
    print(match)
