import re

with open('meta_model_diff.txt', 'r') as f:
    diff_content = f.read()

import os
os.system("git diff 8e869167c HEAD -- extreme_price_movements/meta_model.py > full_meta_diff.txt")
