import os
import pickle

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.position_sizer.runtime import (
    EVDecompositionBundle,
    load_ev_decomposition_bundle,
    predict_ev_components,
    compute_ev_risk,
    gate_and_size,
)
from extreme_price_movements.position_sizer.sizer import PositionSizerConfig


class _DummyPwin:
    def predict_proba(self, X, regime_labels=None, row_ids=None):
        X = np.asarray(X, dtype=float)
        p = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1.0 - p, p])


class _DummyQuantile:
    def __init__(self, scale):
        self.scale = float(scale)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.maximum(0.0, self.scale * np.abs(X[:, 0]))


def _make_bundle(tmp_path):
    bundle = EVDecompositionBundle(
        feature_cols=["score"],
        pwin_model=_DummyPwin(),
        win_model={"q50": _DummyQuantile(0.5), "q80": _DummyQuantile(0.8)},
        loss_model={"q50": _DummyQuantile(0.3), "q90": _DummyQuantile(0.6)},
        tp_sl_defaults=None,
        config={"backend": "ev_decomposition", "costs_mode": "included_in_labels"},
    )
    bundle_path = tmp_path / "ev_decomposition_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    return bundle_path


def test_ev_decomposition_bundle_loads_and_predicts(tmp_path):
    bundle_path = _make_bundle(tmp_path)
    assert os.path.exists(bundle_path)

    bundle = load_ev_decomposition_bundle(bundle_path)
    X_shuffled = pd.DataFrame({"extra": [1.0, 2.0], "score": [0.1, -0.2]})
    preds = predict_ev_components(bundle, X_shuffled)
    ev, risk = compute_ev_risk(preds, costs=0.0, cfg=PositionSizerConfig(costs_mode="included_in_labels"))
    allow, size = gate_and_size(ev, risk, cfg=PositionSizerConfig(), alpha_score=np.array([0.1, -0.2]))
    assert len(size) == 2
    assert len(allow) == 2


def test_ev_decomposition_runtime_validates_inputs_and_cost_policy(tmp_path):
    bundle_path = _make_bundle(tmp_path)
    bundle = load_ev_decomposition_bundle(bundle_path)
    preds = predict_ev_components(bundle, pd.DataFrame({"score": [0.1, -0.2]}))

    with pytest.raises(ValueError):
        predict_ev_components(bundle, pd.DataFrame({"not_score": [1.0, 2.0]}))

    with pytest.raises(ValueError):
        compute_ev_risk(preds, costs=0.001, cfg=PositionSizerConfig(costs_mode="included_in_labels"))
