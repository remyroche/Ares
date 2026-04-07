from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
X = np.random.randn(10, 5)
y = np.random.rand(10, 3)
y /= y.sum(axis=1, keepdims=True)
clf = ExtraTreesRegressor()
clf.fit(X, y)
p = clf.predict(X)
print("Shape:", p.shape)
