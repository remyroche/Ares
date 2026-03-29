with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

# Replace all calls: _meta_feature_keys_for_kind(cfg, k) -> _meta_feature_keys_for_kind(cfg, strat)
# Wait, let's look at the contexts.
import re
lines = content.split('\n')
for i, line in enumerate(lines):
    if "_meta_feature_keys_for_kind(" in line and "def _meta_feature_keys_for_kind(" not in line:
        if "cfg, k" in line:
            lines[i] = line.replace("cfg, k", "cfg, strat")

with open('extreme_price_movements/training.py', 'w') as f:
    f.write('\n'.join(lines))
print("Patched calls!")
