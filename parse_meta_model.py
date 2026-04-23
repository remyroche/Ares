import re

with open('extreme_price_movements/meta_model.py', 'r') as f:
    lines = f.readlines()

in_class = False
for i, line in enumerate(lines):
    if line.startswith('class MetaModel:'):
        in_class = True
        print(f"Found MetaModel at line {i+1}")
        break

with open('meta_model_diff.txt', 'r') as f:
    diff_content = f.read()

import os
os.system("git diff 8e869167c HEAD -- extreme_price_movements/meta_model.py | grep 'class TrueSoftXGBWrapper' -A 30")
