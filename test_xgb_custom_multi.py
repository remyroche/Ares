import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

y_true = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])
X = np.random.randn(3, 10)

def custom_soft_crossentropy(labels, preds):
    # preds is flattened (N * num_class)
    preds = preds.reshape(-1, 3)
    # Apply softmax
    exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    grad = prob - labels
    hess = prob * (1.0 - prob)
    return grad.ravel(), hess.flatten()

dtrain = xgb.DMatrix(X, label=y_true.flatten()) # Flatten labels to bypass DMatrix checks?
# Actually DMatrix might complain if label shape != (N,)
# Wait, labels are passed to objective via dtrain.get_label()
# Let's bypass DMatrix label check by passing dummy labels and storing real ones in a closure
def get_obj(y_soft):
    def obj(preds, dtrain):
        preds = preds.reshape(-1, 3)
        exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        grad = prob - y_soft
        hess = prob * (1.0 - prob)
        return grad.ravel(), hess.ravel()
    return obj

dummy_y = np.zeros(3)
dtrain = xgb.DMatrix(X, label=dummy_y)

params = {"num_class": 3, "disable_default_eval_metric": 1}
bst = xgb.train(params, dtrain, num_boost_round=10, obj=get_obj(y_true))

preds = bst.predict(xgb.DMatrix(X))
# preds are logits. Apply softmax
exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

print("Shape:", prob.shape)
print("Log loss:", log_loss(y_true, prob))
