import re

with open('extreme_price_movements/feature_transforms.py', 'r') as f:
    content = f.read()

# Need to update CausalFeatureTransformer.transform and transform_batch
# We import get_feature_family and FeatureFamily
import_stmt = "from .feature_family_registry import get_feature_family, FeatureFamily\n"

# Locate the right place for import (after standard imports)
content = re.sub(r'from \.utils import tprint, check_inf_nan\n',
                 r'from \.utils import tprint, check_inf_nan\n' + import_stmt,
                 content)

with open('extreme_price_movements/feature_transforms.py', 'w') as f:
    f.write(content)
