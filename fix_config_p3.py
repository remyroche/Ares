import re

with open("extreme_price_movements/config.py", "r") as f:
    content = f.read()

# Replace any lr_* with ret*h, but I just grep'd and saw no lr_ in config.py
# Let's double check if there are any other non-canonical return names
# The spec says "Ensure ret24h exists and is used under the canonical ret24h name only."
# I'll check what return names are used.
