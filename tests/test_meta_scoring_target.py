import numpy as np

from extreme_price_movements.meta_model import MetaModel


def test_cv_train_predict_scores_on_raw_target_not_transformed_target(monkeypatch):
    m = MetaModel()

    captured = {}

    def fake_oof_score(y, pred, baseline, quantile_like=True):
        captured["y"] = np.asarray(y, dtype=float)
        captured["pred"] = np.asarray(pred, dtype=float)
        return 0.0, {"score": 0.0}, True

    class _DummyModel:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=float)

    monkeypatch.setattr(m, "_oof_score", fake_oof_score)
    monkeypatch.setattr(m, "_fit_model", lambda *a, **k: _DummyModel())

    X = np.arange(30.0, dtype=float).reshape(-1, 1)
    y_train = np.linspace(-1.0, 1.0, 30)
    y_score = y_train * 10.0

    _, _, _ = m._cv_train_predict("ridge", [0.85], {}, X, y_train, None, score_y=y_score)

    score_values = set(np.round(y_score, 6))
    train_values = set(np.round(y_train, 6))
    captured_values = np.round(captured["y"], 6)

    # OOF mask can drop rows, so we only assert captured points come from score_y scale.
    assert all(v in score_values for v in captured_values)
    assert any(v not in train_values for v in captured_values)
