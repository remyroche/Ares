import re

with open('extreme_price_movements/elasticnet_feature_selection_v2.py', 'r') as f:
    content = f.read()

import_stmt = "\nfrom .feature_views import get_feature_view\n"
if "from .feature_views import get_feature_view" not in content:
    content = content.replace("import numpy as np\n", "import numpy as np\n" + import_stmt)


# In `select_features_via_staged_en_rfe`:
# We want to apply the view to the `X_cols`
target = '''def select_features_via_staged_en_rfe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cols: List[str],'''

new_target = '''def select_features_via_staged_en_rfe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cols: List[str],'''

# We actually need to filter X_train to only include columns in X_linear, but
# X_train is a numpy array. We need to filter by index.

# Better:
replace_code = '''    # Restrict to linear view
    linear_cols_set = set(get_feature_view(X_cols, "X_linear"))
    linear_indices = [i for i, col in enumerate(X_cols) if col in linear_cols_set]

    # Check if we actually need to filter (to avoid unnecessary copies)
    if len(linear_indices) < len(X_cols):
        tprint(f"EN RFE: Filtering {len(X_cols)} input features to {len(linear_indices)} linear view features.")
        X_train = X_train[:, linear_indices]
        X_cols = [X_cols[i] for i in linear_indices]

    n_features = X_train.shape[1]'''

# Let's see where n_features is defined inside the function.
