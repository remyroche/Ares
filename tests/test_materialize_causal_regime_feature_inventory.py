from pathlib import Path

from scripts.materialize_causal_regime_feature_inventory import causal_status, feature_family, missing_observable_suggestions, unavailable_liquidity_details


def test_family_assignment_covers_required_inventory_groups() -> None:
    assert feature_family("market_median_ret_24h") == "returns_trend"
    assert feature_family("ob_spread_bps_z_24h") == "liquidity_spread_depth"
    assert feature_family("market_average_pair_corr_24h") == "cross_asset_dependence_covariance"
    assert feature_family("funding_deleveraging_divergence") == "funding_oi_liquidation"
    assert feature_family("health__raw_score_mean") == "model_health"


def test_outcome_and_state_labels_are_forbidden_not_merely_unselected() -> None:
    status, reason = causal_status("target__transition_active", source="source_ledger")
    assert status == "outcome_or_label_forbidden"
    assert "target" in reason


def test_historical_store_and_health_are_distinguished() -> None:
    assert causal_status("atr_fraction_14h", source="historical_feature_store")[0] == "pit_contract_not_verified_here"
    assert causal_status("realized_volatility_24h", source="historical_feature_store")[0] == "pit_contract_not_verified_here"
    assert causal_status("health__raw_score_mean", source="historical_model_health")[0] == "causal_historical_lineage_only"


def test_missing_observable_suggestions_are_non_promotional_and_cover_families() -> None:
    suggestions = missing_observable_suggestions()
    assert {"liquidity_spread_depth", "funding_oi_liquidation", "model_health"}.issubset(set(suggestions.family))
    assert suggestions.required_contract.str.len().gt(10).all()


def test_source_missing_liquidity_fields_are_not_reported_as_unselected() -> None:
    rows = unavailable_liquidity_details({"sources": {"skipped_feature_files": {"x.parquet": ["ob_depth_usd_l20_z"]}}})
    assert rows.iloc[0].availability_status == "source_unavailable"
