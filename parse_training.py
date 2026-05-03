import re

with open('meta_training_diff.txt', 'r') as f:
    diff_content = f.read()

import os
os.system("git diff 8e869167c HEAD -- extreme_price_movements/training.py > full_training_diff.txt")
