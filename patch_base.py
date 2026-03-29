import re

with open("extreme_price_movements/training.py", "r") as f:
    content = f.read()

# Replace ['tight', 'balanced', 'wide'] with ['tight', 'wide'] in two places:
content = content.replace(
    '"base_geometry_archetypes", ["tight", "balanced", "wide"]',
    '"base_geometry_archetypes", ["tight", "wide"]'
)

# And remove:
#                     if _variant == "balanced":
#                         continue
content = re.sub(
    r'\s+if _?variant == "balanced":\n\s+continue',
    '',
    content
)

with open("extreme_price_movements/training.py", "w") as f:
    f.write(content)
