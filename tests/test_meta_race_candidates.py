import numpy as np
import pandas as pd

from extreme_price_movements.meta_model import MetaModel


def test_meta_race_candidates_exclude_huber_and_tailweighted(monkeypatch):
    rng = np.random.default_rng(123)
    n = 220
    X = pd.DataFrame(rng.normal(size=(n, 24)), columns=[f"x{i}" for i in range(24)])
    y = (0.04 * X["x0"].values + 0.03 * X["x1"].values + rng.normal(scale=0.08, size=n)).astype(float)

    # Keep compute low in test.
    monkeypatch.setattr(MetaModel, "_select_tail_features", lambda self, X, y, max_features=40: list(X.columns))
    monkeypatch.setattr(MetaModel, "_optuna_hpo", lambda self, _n, _k, params, *_a, **_kw: params)

    m = MetaModel(strategy_name="smoke")
    m.fit(X, y)

    model_names = {r["model"] for r in m.report_rows}
    assert "ridge" in model_names
    assert "ridge" in model_names
    assert "huber" not in model_names
    assert "extratrees_tailweighted" not in model_names

    for row in m.report_rows:
        assert "spread10" in row
        assert "ece_top30" in row
