from __future__ import annotations

from scripts.run_meaningful_mfe_base_residual_catboost_ablation import (
    _configured_features_by_side,
    _expand_config_features,
)


def test_config_alias_expansion_preserves_base_meta_routing() -> None:
    base, meta = _configured_features_by_side()
    assert len(base["long"]) > 100
    assert len(base["short"]) > 100
    assert len(meta) > 100
    assert "FEATURE_SELECTION_KEYS" not in base["long"]
    assert "FEATURE_SELECTION_KEYS" not in meta
    assert "mkt_ret_eq_24h" in meta


def test_config_alias_expansion_handles_lowercase_family_aliases() -> None:
    expanded = _expand_config_features(["spread_proxy_features"])
    assert len(expanded) > 1
    assert "spread_proxy_features" not in expanded
