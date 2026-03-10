import re

file_path = 'extreme_price_movements/elasticnet_feature_selection_v2.py'
with open(file_path, 'r') as f:
    content = f.read()

# Need to find where features are initially passed to the selector or defined.
# It seems `select_features_via_staged_en_rfe` takes `X_cols` list of strings.
# Or we can just import get_feature_view and apply it directly inside the function where features are extracted.
