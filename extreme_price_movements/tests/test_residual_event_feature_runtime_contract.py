from extreme_price_movements.inference.run_inference import (
    _build_residual_event_feature_runtime_cfg,
    _expanded_live_refresh_feature_keys,
    _full_universe_ae_gmm_feature_contract,
)


def test_residual_event_runtime_uses_batch_transform_contract() -> None:
    cfg = _build_residual_event_feature_runtime_cfg(
        {"run_id": "test", "live_feature_offline_cache_enabled": True},
        coverage_symbols=["BTC/USD:USD", "ETH/USD:USD"],
        optional_feature_keys=["optional_b", "optional_a"],
        same_cycle_memory=True,
    )

    assert cfg["live_feature_cache_namespace"] == "residual_event"
    assert cfg["live_feature_coverage_symbols"] == [
        "BTC/USD:USD",
        "ETH/USD:USD",
    ]
    assert cfg["live_feature_cache_optional_feature_keys"] == [
        "optional_a",
        "optional_b",
    ]
    assert cfg["live_feature_memory_cache_enabled"] is True
    assert cfg["live_feature_prefer_offline_cache"] is True
    assert cfg["live_feature_offline_cache_enabled"] is True
    assert cfg["live_feature_offline_cache_authoritative"] is True
    assert cfg["live_feature_snapshot_cache_enabled"] is False
    assert cfg["live_feature_rolling_cache_enabled"] is False
    assert cfg["live_causal_transform_state_enabled"] is False
    assert cfg["feature_causal_transform_state_enabled"] is False
    assert cfg["live_raw_rolling_state_enabled"] is False
    assert cfg["feature_raw_rolling_state_enabled"] is False


def test_residual_refresh_keeps_meta_alias_and_observable_source() -> None:
    required = _expanded_live_refresh_feature_keys(
        ["__meta_raw__rv_ratio_6_24", "gmm_mahal_3"]
    )

    assert "__meta_raw__rv_ratio_6_24" in required
    assert "rv_ratio_6_24" in required
    assert "gmm_mahal_3" in required


class _SideExpert:
    def required_input_features(self, side: str) -> list[str]:
        assert side == "short"
        return ["score", "dae_b16_06", "gmm_dist_center_4"]


def test_full_universe_contract_includes_downstream_ae_gmm_outputs() -> None:
    required = _full_universe_ae_gmm_feature_contract(
        ["gmm_entropy", "price_feature"],
        side="short",
        side_residual_expert=_SideExpert(),
    )

    assert "gmm_entropy" in required
    assert "dae_b16_06" in required
    assert "gmm_dist_center_4" in required
