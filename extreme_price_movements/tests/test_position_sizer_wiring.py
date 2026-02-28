import os
import pickle

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.pipeline_steps import run_position_sizer_step
from extreme_price_movements.position_sizer.runtime import load_bundle, predict_all, compute_ev_risk, gate_and_size
from extreme_price_movements.position_sizer.sizer import PositionSizerConfig


def _make_bundle(tmp_path):
    n = 240
    rng = np.random.default_rng(7)
    oof = pd.DataFrame({
        "reg_mean": rng.normal(size=n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "symbol": ["BTCUSDT"] * n,
    })
    outcomes = pd.DataFrame({
        "return": rng.normal(0.001, 0.01, size=n),
        "mfe_ret": np.abs(rng.normal(0.02, 0.01, size=n)),
        "mae_ret": np.abs(rng.normal(0.01, 0.008, size=n)),
        "timestamp": oof["timestamp"],
        "symbol": oof["symbol"],
    })

    state_file = tmp_path / "state.pkl"
    with open(state_file, "wb") as f:
        pickle.dump({"bundle": {}}, f)

    cfg = {
        "position_sizer_enabled": True,
        "position_sizer_calibration_scope": "regime",
        "position_sizer_calibration_rolling_window": 100,
        "position_sizer_pwin_soft_label_enabled": True,
        "position_sizer_pwin_soft_label_tp": 0.02,
        "position_sizer_pwin_soft_label_sl": 0.01,
        "position_sizer_pwin_soft_label_alpha": 15.0,
        "position_sizer_exp_win_quantile": 0.5,
        "position_sizer_risk_loss_quantile": 0.9,
        "position_sizer_costs_mode": "included_in_labels",
        "tp_sl_search_enabled": False,
    }
    artifacts = {
        "run_id": "20240101_000000",
        "data_root": str(tmp_path),
        "output_dir": str(tmp_path / "position_sizer"),
        "state_file": str(state_file),
    }
    out = run_position_sizer_step(
        cfg=cfg,
        data_bundle={"buckets": {"long_mr": {"oof": oof, "outcomes": outcomes}}},
        artifacts=artifacts,
        logger=lambda *_a, **_k: None,
    )
    bundle_path = out["position_sizer"]["bundle_path"]
    return bundle_path, state_file


def test_run_position_sizer_step_persists_bundle_and_updates_state(tmp_path):
    bundle_path, state_file = _make_bundle(tmp_path)
    assert os.path.exists(bundle_path)

    with open(state_file, "rb") as f:
        state = pickle.load(f)
    assert "position_sizer" in state


def test_runtime_validates_schema_and_cost_policy(tmp_path):
    bundle_path, _ = _make_bundle(tmp_path)
    bundle = load_bundle(bundle_path)

    # Reorder columns -> runtime should handle by feature name ordering.
    X_shuffled = pd.DataFrame({"extra": [1.0, 2.0], "score": [0.1, -0.2]})
    preds = predict_all(bundle, X_shuffled)
    ev, risk = compute_ev_risk(preds, costs=0.0, cfg=PositionSizerConfig(costs_mode="included_in_labels"))
    allow, size = gate_and_size(ev, risk, cfg=PositionSizerConfig(), alpha_score=np.array([0.1, -0.2]))
    assert len(size) == 2
    assert len(allow) == 2

    # Missing required feature should fail loudly.
    with pytest.raises(ValueError):
        predict_all(bundle, pd.DataFrame({"not_score": [1.0, 2.0]}))

    # Prevent double-costing when labels are net.
    with pytest.raises(ValueError):
        compute_ev_risk(preds, costs=0.001, cfg=PositionSizerConfig(costs_mode="included_in_labels"))
