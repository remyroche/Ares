import re

with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

# Add imports for our new training_utils functions
content = content.replace(
    'from extreme_price_movements.training_utils import (',
    'from extreme_price_movements.training_utils import (\n    get_base_feature_keys,\n    get_meta_feature_keys,\n    validate_feature_keys_exist,\n'
)

# 1. build_hourly_training_set_and_weights
# The signature is:
# def build_hourly_training_set_and_weights(
#     feats, mkt_gates, cfg, target_col, feat_key="base_feature_keys",
#     extra_feature_keys=None, label_method="atr", ...

content = re.sub(
    r'def build_hourly_training_set_and_weights\(\n(.*?)feat_key="base_feature_keys",\n(.*?)extra_feature_keys=None,',
    r'def build_hourly_training_set_and_weights(\n\1\2',
    content, flags=re.DOTALL
)

# Inside build_hourly_training_set_and_weights, replace how feat_keys are acquired
# Instead of: feat_keys = list(cfg.get(feat_key, []) or [])
# It's now: feat_keys = get_base_feature_keys(side, cfg)
# And we no longer need `extra_feature_keys` appending since `get_base_feature_keys` handles all base keys including specific ones.
# Also it uses `side` argument.

# Let's write a python replacement instead of messy regex
