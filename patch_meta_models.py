import re
import os

files_to_check = [
    'extreme_price_movements/meta_model.py',
    'extreme_price_movements/meta_model_complex.py'
]

# We need to make sure LightGBM/XGBoost models in the meta models use X_tree.
# Actually, the user spec specifically mentioned the position_sizer_v2 models and feature_selection.
# Let's see if meta_model.py defines its own features.
