import re

with open('extreme_price_movements/config.py', 'r') as f:
    content = f.read()

# Replace MR/TF feature keys with BASE/META ones. We need to parse out the whole blocks.
# And add the new base and meta lists.

# Let's just find the entire block of feature keys. We can search for `"tf_feature_keys": [` and replace everything down to `mr_meta_feature_keys`
# Actually it's probably easier to just replace the whole feature key dict manually. Let's see if we can locate it.

import ast

def find_block(name, text):
    match = re.search(r'"' + name + r'":\s*\[(.*?)\]\s*,(\s*#|\s*"[a-zA-Z_]+":)', text, re.DOTALL)
    if match:
        return match
    match = re.search(r'"' + name + r'":\s*list\(dict\.fromkeys\(\(\[(.*?)\]\s*\+\s*neutral_feature_keys', text, re.DOTALL)
    return match

# Remove the old keys
for k in ["tf_feature_keys", "mr_feature_keys", "base_feature_keys", "meta_feature_keys", "mr_meta_feature_keys", "tf_meta_feature_keys"]:
    match = re.search(r'\s*"' + k + r'":\s*(?:\[|list\()[^]*?(?:\s*\+\s*[a-zA-Z_]+)*\s*\]\s*,', content, re.DOTALL)
    if match:
        # A bit hard to match perfectly with regex. Let's use a simpler approach.
        pass

# I'll just append the new keys at the end of the `CFG = {` block before `}`
# or right after `neutral_feature_keys`.

# Actually, replacing the existing ones is better.
