import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

y_true = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])
X = np.random.randn(3, 10)
dtrain = xgb.DMatrix(X, label=y_true)
params = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss"}
bst = xgb.train(params, dtrain, num_boost_round=10)
p = bst.predict(xgb.DMatrix(X))
print("Log loss:", log_loss(y_true, p))
