import numpy as np
import xgboost as xgb

y_true = np.array([0, 1, 2])
X = np.random.randn(3, 10)
# Custom objective for cross entropy soft targets
def soft_logloss_obj(preds, dtrain):
    labels = dtrain.get_label()
    # If soft labels
    if labels.ndim == 2:
        pass
    return np.zeros_like(preds), np.ones_like(preds)

dtrain = xgb.DMatrix(X, label=y_true)
params = {"objective": "multi:softprob", "num_class": 3}
bst = xgb.train(params, dtrain, num_boost_round=10)
