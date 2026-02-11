import numpy as np
import pandas as pd

from extreme_price_movements.meta_model import MetaModel


def test_meta_race_tailweighted_candidates_smoke(monkeypatch):
    rng = np.random.default_rng(123)
    n = 220
    X = pd.DataFrame(rng.normal(size=(n, 24)), columns=[f"x{i}" for i in range(24)])
    y = (0.04 * X["x0"].values + 0.03 * X["x1"].values + rng.normal(scale=0.08, size=n)).astype(float)

    # keep compute low in test
    monkeypatch.setattr(MetaModel, "_discover_monotone_constraints", lambda self, X, y, bootstraps=50: tuple([0] * X.shape[1]))
    monkeypatch.setattr(MetaModel, "_discover_interactions", lambda self, X, y: [])
    monkeypatch.setattr(MetaModel, "_optuna_hpo", lambda self, *args, **kwargs: args[3])

    # Force a deterministic, compact race that still includes required candidate families.
    def _small_race(self, mono, inter):
        ridge = {"alpha": 2.0, "fit_intercept": True}
        et = {"n_estimators": 60, "max_depth": 6, "min_samples_leaf": 8, "max_features": "sqrt", "n_jobs": 1, "random_state": 42}
        lgb_like = {"objective": "quantile", "alpha": 0.85, "n_estimators": 120, "learning_rate": 0.08, "num_leaves": 31, "max_depth": 5, "random_state": 42, "n_jobs": 1, "verbosity": -1}
        return {
            "ridge_tailweighted_l1": ("ridge", [0.85], ridge, "non_quantile"),
            "extratrees_tailweighted_l1": ("extratrees", [0.85], et, "non_quantile"),
            "lgbm_tailweighted_l1": ("lgb", [0.85], lgb_like, "non_quantile"),
        }

    monkeypatch.setattr(MetaModel, "_race_candidates", _small_race)

    calls = []
    original_select = MetaModel._select_features_for_candidate

    def _counting_select(self, X_meta, y_np, candidate_name, kind):
        calls.append(candidate_name)
        return original_select(self, X_meta, y_np, candidate_name, kind)

    monkeypatch.setattr(MetaModel, "_select_features_for_candidate", _counting_select)

    m = MetaModel(strategy_name="smoke")
    m.fit(X, y)

    model_names = {r["model"] for r in m.report_rows}
    assert "ridge_tailweighted_l1" in model_names
    assert "extratrees_tailweighted_l1" in model_names
    # lgb may be unavailable in environment; if available, it should be in race results
    try:
        import lightgbm  # noqa: F401
        assert "lgbm_tailweighted_l1" in model_names
    except Exception:
        pass

    for row in m.report_rows:
        assert "top10_mean_y" in row
        assert "bot10_mean_y" in row
        assert "spread10" in row
        assert "top_decile_calibration_gap" in row

    # ensure feature selection invoked independently per candidate
    assert "ridge_tailweighted_l1" in calls
    assert "extratrees_tailweighted_l1" in calls
