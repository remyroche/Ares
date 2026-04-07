import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

# 100 rows, 3 classes
y_true = np.random.rand(100, 3)
y_true /= np.sum(y_true, axis=1, keepdims=True)
X = np.random.randn(100, 10)

def get_obj(y_soft):
    def obj(preds, dtrain):
        if preds.ndim == 1:
            preds = preds.reshape(-1, 3)
        exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        grad = prob - y_soft
        hess = prob * (1.0 - prob)
        return grad, hess
    return obj

dummy_y = np.zeros(100)
dtrain = xgb.DMatrix(X, label=dummy_y)

params = {"num_class": 3, "disable_default_eval_metric": 1}
# Set learning rate and train for more rounds to check convergence
params.update({"learning_rate": 0.1})
bst = xgb.train(params, dtrain, num_boost_round=100, obj=get_obj(y_true))

preds = bst.predict(xgb.DMatrix(X), output_margin=True) # Must use output_margin=True for raw logits!
if preds.ndim == 1:
    preds = preds.reshape(-1, 3)
exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

print("Shape:", prob.shape)

# custom log loss
print("Log loss custom:", -np.mean(np.sum(y_true * np.log(prob + 1e-12), axis=1)))
