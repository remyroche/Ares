import re

with open('tests/test_lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Fix the test_dilate_mask_by_symbol_is_symbol_safe
content = re.sub(
    r"dilated = consolidator\._dilate_mask_by_symbol\(mask, df, bars=1\)",
    "dilated = consolidator._dilate_mask_by_symbol(mask, df, bars=2)",
    content
)

with open('tests/test_lgbm_based_mask_generation.py', 'w') as f:
    f.write(content)
