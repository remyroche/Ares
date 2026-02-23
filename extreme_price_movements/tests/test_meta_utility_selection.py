import numpy as np

from extreme_price_movements.policy_ml import MetaClassifierSelectionConfig, pick_meta_classifier_by_utility_top30


def test_meta_utility_selection_applies_top_n_and_lift_guards():
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_true[80:140] = 1
    y_true[140:] = 2

    # weakly informative probs
    p = np.full((n, 3), 1/3, dtype=float)
    p[140:, 2] = 0.6
    p[140:, 0] = 0.2
    p[140:, 1] = 0.2
    p[:80, 0] = 0.6
    p[:80, 2] = 0.2
    p[:80, 1] = 0.2

    realized_u = np.where(y_true == 2, 0.02, np.where(y_true == 1, -0.001, -0.02)).astype(float)

    cfg = MetaClassifierSelectionConfig(
        max_logloss=2.0,
        top_frac=0.30,
        min_top_n=50,
        min_lift_vs_baseline=0.001,
        dynamic_utility_from_realized=True,
        require_positive_oof_utility=True,
    )
    out = pick_meta_classifier_by_utility_top30(y_true=y_true, p_pred=p, realized_u_policy=realized_u, cfg=cfg)

    assert out["passed_gate"] > 0.5
    assert out["top_n_ok"] > 0.5
    assert out["lift_ok"] > 0.5
    assert out["passed_econ"] > 0.5
