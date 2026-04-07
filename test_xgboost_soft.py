import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

y_true = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])
X = np.random.randn(3, 10)
clf = XGBClassifier(objective="multi:softprob", use_label_encoder=False)
clf.fit(X, y_true)
p = clf.predict_proba(X)
print("Log loss:", log_loss(y_true, p))
