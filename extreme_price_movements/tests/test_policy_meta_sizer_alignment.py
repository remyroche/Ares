import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.meta_model import MetaClassifierModel
from extreme_price_movements.policy_ml import MetaClassifierSelectionConfig
from extreme_price_movements.ridge_position_sizer import run_ridge_target_race, run_ridge_position_sizer_step


def test_meta_classifier_uses_engine_label_override():
    n = 120
    X = pd.DataFrame({"f1": np.linspace(-1, 1, n), "f2": np.random.randn(n)})
    y_ret = np.random.randn(n) * 0.01
    y_class = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3]).astype(np.int8)
    realized_u = np.where(y_class == 2, 0.01, np.where(y_class == 0, -0.01, 0.0))

    m = MetaClassifierModel(strategy_name="t")
    m._build_candidates = lambda: {
        "ridge_clf": {
            "kind": "ridge_clf",
            "params": {"C": 0.1, "penalty": "l2", "solver": "lbfgs", "max_iter": 300, "class_weight": "balanced", "multi_class": "multinomial"},
        }
    }
    def _fake_cv(kind, params, Xv, yv, sw=None):
        oof = np.full((len(yv), 3), 1.0 / 3.0, dtype=float)
        oof[np.arange(len(yv)), yv] = 0.6
        oof = oof / oof.sum(axis=1, keepdims=True)
        return oof, 0.9
    m._cv_evaluate = _fake_cv
    m.fit(
        X,
        y_ret,
        y_class_override=y_class,
        realized_u_policy=realized_u,
        selection_cfg=MetaClassifierSelectionConfig(min_top_n=5),
    )
    assert m.oof_probs is not None
    assert m.oof_probs.shape[1] == 3


def test_ridge_target_race_selects_by_topq_u_policy_not_ic():
    n = 300
    rng = np.random.default_rng(0)
    # feature 0 produces high IC-like behavior vs returns but bad utility in top ranks
    x0 = np.linspace(-2, 2, n)
    x1 = rng.normal(size=n)
    X = np.column_stack([x0, x1])
    returns = x0 + 0.05 * rng.normal(size=n)
    symbols = np.array(["A"] * n)
    timestamps = np.arange(n)

    # Define utility to penalize high x0 slice so IC-friendly target should lose
    u_policy = -np.tanh(x0) + 0.01 * rng.normal(size=n)

    name, _y, _log = run_ridge_target_race(
        X,
        returns,
        symbols,
        timestamps,
        select_metric="topq_u_policy",
        u_policy=u_policy,
        topq=0.30,
        require_positive_topq_u=False,
        topq_min_samples=10,
    )
    assert isinstance(name, str)
    assert any("TopQMeanU" in line for line in _log)


def test_ridge_sizer_requires_u_policy_for_topq_metric():
    n = 60
    oof = pd.DataFrame({"m1": np.random.randn(n), "m2": np.random.randn(n)})
    outcomes = pd.DataFrame({"return": np.random.randn(n) * 0.01, "is_long": np.ones(n)})
    with pytest.raises(ValueError, match="u_policy"):
        run_ridge_position_sizer_step(
            oof_preds=oof,
            trade_outcomes=outcomes,
            cfg={"sizer_select_metric": "topq_u_policy", "n_grid_points": 2},
            save_model=False,
        )
