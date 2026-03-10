import re

with open('extreme_price_movements/elasticnet_feature_selection.py', 'r') as f:
    content = f.read()

import_stmt = "from .feature_views import get_feature_view\n"
if import_stmt not in content:
    content = content.replace("import numpy as np\n", "import numpy as np\n" + import_stmt)

target = '''    n_features = X_train.shape[1]

    # Pre-rank features via univariate correlation for potential fallback
    # Handle NaN in X_train for correlation by masking or just using nan_to_num (already done)'''

replacement = '''    # --- Ensure we only use X_linear view features ---
    linear_cols_set = set(get_feature_view(feature_names, "X_linear"))
    linear_indices = [i for i, col in enumerate(feature_names) if col in linear_cols_set]

    if len(linear_indices) < len(feature_names):
        tprint(f"ElasticNet: Filtering {len(feature_names)} input features to {len(linear_indices)} linear view features.")
        X_train = X_train[:, linear_indices]
        feature_names = [feature_names[i] for i in linear_indices]

    n_features = X_train.shape[1]

    # Pre-rank features via univariate correlation for potential fallback
    # Handle NaN in X_train for correlation by masking or just using nan_to_num (already done)'''

content = content.replace(target, replacement)

with open('extreme_price_movements/elasticnet_feature_selection.py', 'w') as f:
    f.write(content)
