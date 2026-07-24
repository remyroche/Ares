from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _fit_lgbm_models,
    _predict_lgbm_models,
)


def _params() -> dict[str, object]:
    return {
        "loss_function": "regression",
        "n_estimators": 12,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_split_gain": 0.0,
    }


def test_per_side_models_use_independent_feature_contracts() -> None:
    rng = np.random.default_rng(7)
    rows = 1_200
    sides = np.where(np.arange(rows) % 2 == 0, "long", "short")
    frame = pd.DataFrame(
        {
            "long_signal": rng.normal(size=rows).astype(np.float32),
            "short_signal": rng.normal(size=rows).astype(np.float32),
            "noise": rng.normal(size=rows).astype(np.float32),
        }
    )
    target = pd.Series(
        np.where(
            sides == "long",
            frame["long_signal"],
            frame["short_signal"],
        ).astype(np.float32)
    )
    weight = pd.Series(np.ones(rows, dtype=np.float32))
    contracts = {
        "long": ["long_signal", "noise"],
        "short": ["short_signal", "noise"],
    }

    models, fitted_contracts = _fit_lgbm_models(
        x_train=frame,
        y_train=target,
        w_train=weight,
        train_sides=sides,
        params=_params(),
        seed=11,
        model_side_scope="per_side",
        features_by_side=contracts,
    )
    prediction = _predict_lgbm_models(
        models=models,
        x_valid=frame,
        valid_sides=sides,
        model_side_scope="per_side",
        feature_contracts=fitted_contracts,
    )

    assert set(models) == {"long", "short"}
    assert fitted_contracts == contracts
    assert np.isfinite(prediction).all()
    assert np.corrcoef(
        prediction[sides == "long"], frame.loc[sides == "long", "long_signal"]
    )[0, 1] > 0.7
    assert np.corrcoef(
        prediction[sides == "short"], frame.loc[sides == "short", "short_signal"]
    )[0, 1] > 0.7
