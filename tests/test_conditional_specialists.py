import numpy as np
import pandas as pd

from extreme_price_movements.conditional_specialists import (
    ConditionalSpecialistConfig,
    effective_rows,
    ordinal_residual_grade,
    portability_score,
    soft_regions,
)


def test_fold_geometry_control_uses_train_fit_and_emits_soft_memberships():
    from scripts.run_pair_condition_specialists import _add_fold_gmm_geometry

    rng = np.random.default_rng(7)
    fields = [
        "mkt_drawdown_from_7d_high_atr", "mkt_recovery_from_24h_low_atr",
        "breadth_dispersion", "downside_breadth_intensity",
        "rv_24h_peer_resid", "ob_depth_mkt_resid",
        "mkt_funding_dispersion_z_30d", "mkt_oi_flush_z_30d",
        "oi_expansion_compression_balance_24h", "mkt_ret_per_oi_change_4h",
    ]
    def make(n, offset):
        x = pd.DataFrame({f: rng.normal(offset, 1.0, n) for f in fields})
        x["candidate_id"] = [f"x{i}" for i in range(n)]
        return x
    train, cal, test = make(160, 0.0), make(32, 0.5), make(32, 1.0)
    tr, ca, te, outputs, meta = _add_fold_gmm_geometry(train, cal, test, side="long")
    assert meta["status"] == "fit"
    assert len(outputs) == 7
    probs = tr[[f"geometry_gmm_p_{i}" for i in range(4)]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(te[outputs].to_numpy(dtype=float)).all()


def test_soft_regions_are_bounded_and_ordered():
    values = pd.Series([-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    low, high, _ = soft_regions(values.to_numpy())
    assert np.all(low >= 0.0) and np.all(low <= 1.0)
    assert np.all(high >= 0.0) and np.all(high <= 1.0)
    assert low[0] > high[0]
    assert high[-1] > low[-1]


def test_effective_rows_uses_membership_mass_not_row_count():
    assert effective_rows(np.array([1.0, 0.5, 0.0])) == 1.8
    assert effective_rows(np.array([0.0, 0.0])) == 0.0


def test_ordinal_residual_grade_is_monotone_and_bounded():
    residual = pd.Series([-1000.0, -100.0, 0.0, 100.0, 1000.0])
    grade = ordinal_residual_grade(residual, [-150.0, -50.0, 50.0, 150.0])
    assert grade.tolist() == [0, 1, 2, 3, 4]
    assert grade.dtype.kind in "iu"


def test_portability_penalises_dispersion_and_negative_worst_case():
    stable = portability_score([5.0, 5.0, 5.0])
    unstable = portability_score([-5.0, 5.0, 15.0])
    assert stable == 5.0
    assert unstable < stable


def test_config_freezes_causal_specialist_contract():
    cfg = ConditionalSpecialistConfig()
    assert cfg.max_raw_spine_features >= cfg.min_raw_spine_features >= 40
    assert cfg.specialist_min_features <= cfg.specialist_max_features <= max(cfg.specialist_feature_caps)
    assert tuple(cfg.specialist_feature_caps) == (40, 60, 80, 100, 120)
    assert cfg.minimum_supported_months >= 3
    assert cfg.residual_grade_edges == (-150.0, -50.0, 50.0, 150.0)


def test_weighting_contract_has_predeclared_exponents_and_month_balance():
    cfg = ConditionalSpecialistConfig()
    assert cfg.equal_condition_month_weighting is True
    assert cfg.condition_weight_exponent == 1.5
    assert tuple(sorted(cfg.membership_exponents)) == (1.0, 1.5, 2.0)


def test_unary_control_condition_uses_single_activation():
    from scripts.run_pair_condition_specialists import _condition_weight

    activation = {"vol": {"low": np.array([0.2, 0.7], dtype=np.float32)}}
    value = _condition_weight(
        pd.DataFrame({"x": [1, 2]}),
        {"context_feature_a": "vol", "activation_a": "low", "unary": True},
        activation,
    )
    assert np.allclose(value, [0.2, 0.7])


def test_config_contains_nonadjacent_support_contract():
    cfg = ConditionalSpecialistConfig()
    assert cfg.minimum_nonadjacent_months >= cfg.minimum_supported_months
    assert cfg.minimum_month_effective_queries >= 1
