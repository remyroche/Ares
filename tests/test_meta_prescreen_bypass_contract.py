from extreme_price_movements.lgbm_pipeline import _resolve_lgbm_pre_mda_bypass_features
from scripts.run_s52_train_meta_regime_handoff_smoke import (
    _meta_pre_mda_bypass_features,
    _meta_structural_features,
)


def test_meta_conditional_features_bypass_cheap_screens_but_not_mda():
    columns = [
        "score_base",
        "side_name_long",
        "side_name_short",
        "archetype_label_family_long_vol_compression",
        "archetype_policy_key_short_default",
        "gmm_prob_0",
        "rel_rankband_edge",
        "meta_sel_ood_abs_z_p95",
        "lr_1h",
    ]

    structural = _meta_structural_features(columns)
    conditional = _meta_pre_mda_bypass_features(columns)
    resolved = _resolve_lgbm_pre_mda_bypass_features(
        columns,
        {"pre_mda_bypass_features": conditional},
    )

    assert structural == ["side_name_long", "side_name_short"]
    assert "score_base" not in conditional
    assert "lr_1h" not in conditional
    assert set(resolved) == set(conditional)
    assert "archetype_label_family_long_vol_compression" in resolved
    assert "gmm_prob_0" in resolved
    assert "rel_rankband_edge" in resolved
