import numpy as np
import pandas as pd
from extreme_price_movements.meta_model import MetaClassifierModel
from extreme_price_movements.soft_labels import DynamicSoftLabels
X = pd.DataFrame(np.random.randn(100, 5))
X.columns = [f"f{i}" for i in range(5)]
y_ret = np.random.randn(100)
# soft labels
mfe = np.random.rand(100) * 0.1
mae = np.random.rand(100) * 0.1
t_mfe = np.random.rand(100) * 4.0
t_mae = np.random.rand(100) * 4.0
atr_1h = np.ones(100) * 0.01

y_class_override = DynamicSoftLabels(mfe, mae, t_mfe, t_mae, 4, atr_1h)

model = MetaClassifierModel(strategy_name="test")
model.candidate_mode = "xgb_parallel_forest"
model.xgb_parallel_forest_params = {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.1, "verbosity": 0}
model.fit(X_meta=X, y_ret=y_ret, y_class_override=y_class_override)
print("Soft labels target OK!")
