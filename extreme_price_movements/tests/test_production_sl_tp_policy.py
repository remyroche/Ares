from extreme_price_movements.production_sl_tp_policy import (
    SLTPPolicy,
    passes_tp_superior_additive,
    expand_configs_wide_sl_tp_additive_superiority,
)


def test_additive_superiority_rule():
    assert passes_tp_superior_additive(0.02, 0.01, 0.005)
    assert not passes_tp_superior_additive(0.02, 0.018, 0.005)


def test_wide_sl_tp_expansion_filters_violations():
    base = {"k_tp": 1.0}
    pol = SLTPPolicy(sl_as_tp_pct_grid=(0.2, 0.6, 1.25), superiority_add=0.0075, drop_on_violation=True)
    out = expand_configs_wide_sl_tp_additive_superiority(base, tp_eff=0.02, policy=pol)
    # pass when 0.02 >= s*0.02 + 0.0075 => s <= 0.625
    vals = sorted(x["sl_as_tp_pct"] for x in out)
    assert vals == [0.2, 0.6]
