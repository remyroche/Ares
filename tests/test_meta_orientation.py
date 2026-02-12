import numpy as np
import pandas as pd

from extreme_price_movements.meta_model import MetaModel


class _DummyModel:
    def __init__(self, c=1.0):
        self.c = c

    def predict(self, X):
        return np.full(X.shape[0], self.c, dtype=float)


def test_meta_predict_respects_score_sign():
    m = MetaModel()
    m.selected_features = ["a", "b"]
    m.model = {"models": [_DummyModel(2.0)]}

    X = pd.DataFrame({"a": [1, 2, 3], "b": [0, 0, 0]})

    m.score_sign = 1
    p1 = m.predict(X)
    m.score_sign = -1
    p2 = m.predict(X)

    np.testing.assert_allclose(p1, -p2, atol=1e-12)
