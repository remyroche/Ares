import re

with open('extreme_price_movements/feature_selection_extreme_events.py', 'r') as f:
    content = f.read()

import_stmt = "\nfrom .feature_views import get_feature_view\n"
if import_stmt not in content:
    content = content.replace("import numpy as np\n", "import numpy as np\n" + import_stmt)

target = '''def mdi_feature_selection_v3(
    X_train: pd.DataFrame,
    y_train: np.ndarray,'''

# The function `mdi_feature_selection_v3` takes `X_train: pd.DataFrame`.
# We can filter `X_train` at the start of `mdi_feature_selection_v3`

# Let's locate the start of the function body
func_start = '''    do_topk_ranking: bool = False,
    topk_rank_k: int = 15,
) -> Dict:
    tprint("Starting advanced MDI feature selection (v3)...")'''

replacement = '''    do_topk_ranking: bool = False,
    topk_rank_k: int = 15,
) -> Dict:
    tprint("Starting advanced MDI feature selection (v3)...")

    # --- Ensure we only use X_tree view features ---
    feature_names = list(X_train.columns)
    tree_cols_set = set(get_feature_view(feature_names, "X_tree"))
    tree_indices = [i for i, col in enumerate(feature_names) if col in tree_cols_set]

    if len(tree_indices) < len(feature_names):
        tprint(f"MDI RFE: Filtering {len(feature_names)} input features to {len(tree_indices)} tree view features.")
        X_train = X_train.iloc[:, tree_indices]
'''

content = content.replace(func_start, replacement)


# Also do v4 if it's there
func_start2 = '''def mdi_feature_selection_v4_topk(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    timestamps_train: Optional[np.ndarray] = None,
    min_features: int = 10,
    max_features: int = 25,
    top_k: int = 15,
    budget_minutes: int = 5,
    random_state: int = 42,
) -> Dict:
    tprint("Starting MDI feature selection (v4_topk)...")'''

replacement2 = '''def mdi_feature_selection_v4_topk(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    timestamps_train: Optional[np.ndarray] = None,
    min_features: int = 10,
    max_features: int = 25,
    top_k: int = 15,
    budget_minutes: int = 5,
    random_state: int = 42,
) -> Dict:
    tprint("Starting MDI feature selection (v4_topk)...")

    # --- Ensure we only use X_tree view features ---
    feature_names = list(X_train.columns)
    tree_cols_set = set(get_feature_view(feature_names, "X_tree"))
    tree_indices = [i for i, col in enumerate(feature_names) if col in tree_cols_set]

    if len(tree_indices) < len(feature_names):
        tprint(f"MDI RFE: Filtering {len(feature_names)} input features to {len(tree_indices)} tree view features.")
        X_train = X_train.iloc[:, tree_indices]
'''

content = content.replace(func_start2, replacement2)


with open('extreme_price_movements/feature_selection_extreme_events.py', 'w') as f:
    f.write(content)
