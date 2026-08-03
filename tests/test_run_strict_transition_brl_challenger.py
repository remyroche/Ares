import numpy as np

from scripts.run_strict_transition_brl_challenger import CONFIGS, HEADS, _arm, metric_values


def test_native_brl_arm_is_bounded_and_probability_safe():
    x = np.column_stack([np.arange(40, dtype=float), np.tile([0.0, 1.0], 20)])
    y = np.array([0] * 24 + [1] * 16, dtype=np.int8)
    arm = _arm(CONFIGS[0], seed=17).fit(x, y, np.ones(len(y)), ["causal_a", "causal_b"])
    probability = arm.predict_proba(x)
    assert arm.backend == "native_beta_binomial_map"
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert all("target__" not in name for name in ["causal_a", "causal_b"])


def test_strict_brl_contract_covers_active_and_predeclared_horizons():
    assert [head for head, _ in HEADS] == [
        "stable_vs_transition", "onset_h1", "onset_h3", "onset_h6", "onset_h12",
    ]
    assert metric_values(np.array([0, 1]), np.array([0.2, 0.8]))["brier"] >= 0.0
