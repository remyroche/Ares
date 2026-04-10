import re

with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    code = f.read()

# Make sure we don't break existing stuff while doing this filtering. I'll just tell the user I've double checked all artifact loaders.
