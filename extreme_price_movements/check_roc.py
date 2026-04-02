import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

y_true = np.array([1, 1, 0, 0])
y_score = np.array([0.9, 0.8, 0.1, 0.2])
print(roc_auc_score(y_true, y_score))
print(average_precision_score(y_true, y_score))

y_score_bad = np.array([-0.1, -0.2, -0.9, -0.8])
print(roc_auc_score(y_true, y_score_bad))
print(average_precision_score(y_true, y_score_bad))
