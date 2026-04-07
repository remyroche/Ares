import sys
from extreme_price_movements.config import CFG, REGIME_FEATURE_KEYS

meta_keys = CFG.get("meta_feature_keys", [])
mr_meta_keys = CFG.get("mr_meta_feature_keys", [])
tf_meta_keys = CFG.get("tf_meta_feature_keys", [])

all_meta_keys = set(meta_keys + mr_meta_keys + tf_meta_keys)

missing = [f for f in REGIME_FEATURE_KEYS if f not in all_meta_keys]

print("Total REGIME_FEATURE_KEYS:", len(REGIME_FEATURE_KEYS))
print("Missing in meta:", len(missing))
for m in missing:
    print("-", m)
