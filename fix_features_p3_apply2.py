with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

content = content.replace("return primitive_cache[key]    def _roll_min", "return primitive_cache[key]\n\n    def _roll_min")

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
