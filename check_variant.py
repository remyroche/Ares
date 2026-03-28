import re

with open("extreme_price_movements/training.py", "r") as f:
    text = f.read()

lines = text.split("\n")
for i, line in enumerate(lines):
    if "Training grouped base variant" in line:
        for j in range(-15, 10):
            print(f"{i+j}: {lines[i+j]}")
        break
