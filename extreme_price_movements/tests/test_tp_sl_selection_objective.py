import numpy as np

from extreme_price_movements.position_sizer.tp_sl_selection import (
    CompositeObjectiveConfig,
    aggregate_candidate_folds,
    build_tp_sl_grid,
    composite_objective,
    evaluate_fold_metrics,
    expected_log_growth,
    select_robust_default,
    sortino_ratio,
)


def test_metric_definitions_basic():
    r = np.array([0.01, -0.005, 0.002])
    elg = expected_log_growth(r)
    sr = sortino_ratio(r)
    assert np.isfinite(elg)
    assert np.isfinite(sr)


def test_composite_hard_gate_negative_metric():
    cfg = CompositeObjectiveConfig(mode="hard_gate")
    obj = composite_objective(elg=-0.1, sr=1.0, mnpt=0.1, cfg=cfg)
    assert obj == float("-inf")


def test_fold_min_trades_enforced():
    cfg = CompositeObjectiveConfig(min_trades_per_fold=5)
    out = evaluate_fold_metrics(r_t=np.array([0.001, 0.001]), pnl_net=np.array([0.01, 0.01]), cfg=cfg)
    assert out["Objective"] == float("-inf")


def test_aggregate_and_select_robust_default():
    cand_to_folds = {
        (1.0, 0.5): [
            {"Objective": 1.0, "ELG": 0.01, "SR": 1.2, "MNPT": 0.005, "n_trades": 300},
            {"Objective": 0.8, "ELG": 0.009, "SR": 1.0, "MNPT": 0.004, "n_trades": 280},
        ],
        (1.5, 0.75): [
            {"Objective": 1.1, "ELG": 0.011, "SR": 1.1, "MNPT": 0.0055, "n_trades": 310},
            {"Objective": 0.2, "ELG": 0.003, "SR": 0.4, "MNPT": 0.001, "n_trades": 260},
        ],
    }
    summary = aggregate_candidate_folds(cand_to_folds)
    cfg = CompositeObjectiveConfig(q_top=0.5, selection="min_std")
    best = select_robust_default(summary, cfg)
    assert best["candidate"] == (1.0, 0.5)


def test_build_tp_sl_grid():
    grid = build_tp_sl_grid([1.0, 2.0], [0.5, 1.0])
    assert grid == [(1.0, 0.5), (1.0, 1.0), (2.0, 0.5), (2.0, 1.0)]
