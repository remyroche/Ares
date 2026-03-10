import re

file_path = 'extreme_price_movements/quantile_feature_selection_extreme_events.py'
with open(file_path, 'r') as f:
    content = f.read()

import_stmt = "\nfrom .feature_views import get_feature_view\n"
if import_stmt not in content:
    content = content.replace("import numpy as np\n", "import numpy as np\n" + import_stmt)

func_start = '''def mdi_feature_selection_quantile(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    timestamps_train: Optional[np.ndarray] = None,
    min_features: int = 10,
    max_features: int = 30,
    budget_minutes: int = 15,
    random_state: int = 42,
    base_alpha: float = 0.5,
    calibration_mode: bool = False,
    cv_strategy: str = "purged",
) -> Dict:
    tprint(f"Starting Quantile MDI feature selection (alpha={base_alpha})...")'''

replacement = '''def mdi_feature_selection_quantile(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    timestamps_train: Optional[np.ndarray] = None,
    min_features: int = 10,
    max_features: int = 30,
    budget_minutes: int = 15,
    random_state: int = 42,
    base_alpha: float = 0.5,
    calibration_mode: bool = False,
    cv_strategy: str = "purged",
) -> Dict:
    tprint(f"Starting Quantile MDI feature selection (alpha={base_alpha})...")

    # --- Ensure we only use X_tree view features ---
    feature_names = list(X_train.columns)
    tree_cols_set = set(get_feature_view(feature_names, "X_tree"))
    tree_indices = [i for i, col in enumerate(feature_names) if col in tree_cols_set]

    if len(tree_indices) < len(feature_names):
        tprint(f"MDI RFE: Filtering {len(feature_names)} input features to {len(tree_indices)} tree view features.")
        X_train = X_train.iloc[:, tree_indices]
'''

content = content.replace(func_start, replacement)

with open(file_path, 'w') as f:
    f.write(content)
