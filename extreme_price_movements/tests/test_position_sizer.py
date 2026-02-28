import numpy as np
import pandas as pd

from extreme_price_movements.position_sizer.dataset import build_position_sizer_dataset
from extreme_price_movements.position_sizer.models import train_pwin_classifier, predict_quantiles
from extreme_price_movements.position_sizer.sizer import (
    PositionSizerConfig,
    conviction_threshold_from_opportunities,
    percentile_rank,
    sharpen_alpha_score,
    size_position,
    size_positions_ranked,
    temperature_scale_score,
)


def test_dataset_builder_outputs_expected_labels():
    df = pd.DataFrame({"f1": [1, 2, 3], "pnl_label": [0.1, -0.2, 0.0]})
    ds = build_position_sizer_dataset(df, feature_cols=["f1"])
    assert ds.y_win.tolist() == [1, 0, 0]
    assert ds.pwin_target.tolist() == [1.0, 0.0, 0.0]
    assert np.isclose(ds.y_win_mag[0], 0.1)
    assert np.isclose(ds.y_loss_mag[1], 0.2)


def test_soft_pwin_label_from_mfe_mae():
    df = pd.DataFrame(
        {
            "f1": [1, 2],
            "pnl_label": [0.01, -0.01],
            "mfe": [0.06, 0.001],
            "mae": [0.001, 0.06],
        }
    )
    ds = build_position_sizer_dataset(
        df,
        feature_cols=["f1"],
        pwin_soft_cfg={"enabled": True, "tp": 0.02, "sl": 0.01, "alpha": 20.0},
    )
    assert ds.pwin_target[0] > 0.6
    assert ds.pwin_target[1] < 0.4
    assert np.all((ds.pwin_target >= 0.0) & (ds.pwin_target <= 1.0))


def test_rank_helpers():
    u = np.array([0.1, 0.2, 0.3, 0.4])
    r = percentile_rank(0.3, u)
    assert 0.0 <= r <= 1.0
    cutoff = conviction_threshold_from_opportunities(u, trade_percentile_threshold=0.75)
    assert np.isclose(cutoff, 0.325)


def test_size_position_uses_ranking_gate_and_conviction_weight():
    cfg = PositionSizerConfig(
        trade_percentile_threshold=0.80,
        rank_exponent=2.0,
        size_k=1.0,
        max_position_size=2.0,
        risk_epsilon=1e-6,
        ev_threshold=-1.0,
    )

    out = size_position(
        pwin_hat=0.7,
        qwin50_hat=0.03,
        qwin80_hat=0.05,
        qloss50_hat=0.02,
        qloss90_hat=0.04,
        cfg=cfg,
        costs=0.001,
        opportunity_evs=np.array([0.001, 0.002, 0.004, 0.006, 0.008]),
        alpha_score=0.4,
    )
    assert out["trade_allowed"] is True
    assert out["rank"] >= 0.80
    assert out["conviction_score"] > 0.0

    out_low = size_position(
        pwin_hat=0.55,
        qwin50_hat=0.02,
        qwin80_hat=0.03,
        qloss50_hat=0.02,
        qloss90_hat=0.03,
        cfg=cfg,
        costs=0.001,
        opportunity_evs=np.array([0.001, 0.002, 0.004, 0.006, 0.008]),
        alpha_score=0.1,
    )
    assert out_low["rank"] < 0.80
    assert out_low["trade_allowed"] is False
    assert out_low["size"] == 0.0


def test_batch_ranked_allocator_only_allocates_top_quantile():
    cfg = PositionSizerConfig(trade_percentile_threshold=0.75, rank_exponent=2.0)
    ev = np.array([0.1, 0.3, 0.2, 0.4])
    risk = np.array([0.1, 0.1, 0.1, 0.1])
    alpha = np.array([0.2, 0.2, 0.2, 0.2])
    out = size_positions_ranked(ev, risk, alpha, cfg)
    assert out["trade_allowed"].sum() == 1
    assert out["size"][3] > 0


def test_score_sharpening_transforms():
    assert np.isclose(sharpen_alpha_score(-2.0, alpha_power=2.0), -4.0)
    sc = temperature_scale_score(1.0, score_temperature=0.5)
    assert 0.0 < sc < 1.0


def test_pwin_soft_target_training_and_diagnostics():
    n = 300
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, 3))
    y_hard = (x[:, 0] + 0.3 * x[:, 1] > 0.0).astype(int)
    y_soft = np.clip(0.5 + 0.35 * np.tanh(x[:, 0]), 0.0, 1.0)
    pnl = rng.normal(0.0, 0.01, size=n)
    regs = np.where(x[:, 2] > 0.0, "high", "low")

    m_reg = train_pwin_classifier(
        x,
        y_soft,
        calibration_mode="regime",
        regime_labels=regs,
        y_hard_ref=y_hard,
        pnl_ref=pnl,
    )
    p_reg = m_reg.predict_proba(x, regime_labels=regs)[:, 1]
    assert np.all((p_reg > 0.0) & (p_reg < 1.0))
    assert m_reg.diagnostics is not None
    assert "bce" in m_reg.diagnostics
    assert "spearman_pwin_soft" in m_reg.diagnostics

    m_roll = train_pwin_classifier(x, y_soft, calibration_mode="rolling", rolling_window=100)
    p_roll = m_roll.predict_proba(x, row_ids=np.arange(n))[:, 1]
    assert np.all((p_roll > 0.0) & (p_roll < 1.0))


def test_quantile_prediction_is_monotonic():
    class Dummy:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    low, high = predict_quantiles({"q50": Dummy(0.3), "q90": Dummy(0.1)}, np.zeros((4, 2)), high_key="q90")
    assert np.all(high >= low)
