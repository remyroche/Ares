import re

with open('extreme_price_movements/config.py', 'r') as f:
    content = f.read()

target = "POSITION_SIZER_V2_FEATURE_CONFIG ="

new_str = '''
from .feature_views import get_feature_view

POSITION_SIZER_V2_FEATURE_CONFIG ='''

content = content.replace(target, new_str)

with open('extreme_price_movements/config.py', 'w') as f:
    f.write(content)
