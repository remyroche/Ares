import re

with open("extreme_price_movements/config.py", "r") as f:
    content = f.read()

# Define FEATURE_SELECTION_KEYS
feature_keys = [
    "FEATURE_SELECTION_KEYS",
    "TRAINING_RESIDUALIZATION_FEATURE_KEYS",
]

# Find where CONTINUOUS_LOCATION_COLS is defined and insert FEATURE_SELECTION_KEYS before it
# wait, it is in HELPER_BASE_FEATURES list.
# Let's define FEATURE_SELECTION_KEYS as a top-level variable explicitly.

feature_sel_keys_def = """
FEATURE_SELECTION_KEYS = [
    "base_shared_feature_keys",
    "meta_shared_feature_keys",
]
"""
# Insert it after CONTINUOUS_LOCATION_COLS definition
idx = content.find("CONTINUOUS_LOCATION_COLS = [")
if idx != -1:
    idx2 = content.find("]", idx) + 2
    content = content[:idx2] + "\n" + feature_sel_keys_def + "\n" + content[idx2:]

# Remove the legacy code
legacy_code = """# Legacy cleanup: stop requesting upstream boolean LOC/trigger columns.
CFG["FEATURE_SELECTION_KEYS"] = [
    k
    for k in CFG.get("FEATURE_SELECTION_KEYS", [])
    if not (k.startswith("LOC_") or k.startswith("LONG_") or k.startswith("SHORT_"))
]"""
content = content.replace(legacy_code, "")

with open("extreme_price_movements/config.py", "w") as f:
    f.write(content)
