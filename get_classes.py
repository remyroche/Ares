import re

with open('extreme_price_movements/training.py', 'r') as f:
    text = f.read()

classes = re.findall(r'^class (Meta\w+)', text, re.MULTILINE)
print("Meta classes found in training.py:")
for c in classes:
    print(c)
