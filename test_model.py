import numpy as np
import pandas as pd
from extreme_price_movements.meta_model import MetaClassifierModel
X = pd.DataFrame(np.random.randn(100, 5))
X.columns = [f"f{i}" for i in range(5)]
y_ret = np.random.randn(100)
# soft labels
y_class_override = np.random.rand(100, 3)
y_class_override /= y_class_override.sum(axis=1, keepdims=True)

model = MetaClassifierModel(strategy_name="test")
model.candidate_mode = "xgb_parallel_forest"
model.xgb_parallel_forest_params = {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.1, "verbosity": 0}
model.fit(X_meta=X, y_ret=y_ret, y_class_override=y_class_override)
print("Soft labels target OK!")
