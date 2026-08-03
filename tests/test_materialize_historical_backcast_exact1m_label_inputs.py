from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from scripts import materialize_historical_backcast_exact1m_label_inputs as adapter


class _Store:
    def __init__(self, *args, **kwargs):
        pass

    def load(self, symbol, columns, start_ts, end_ts):
        index = pd.date_range(
            pd.Timestamp("2023-10-03T00:00:00Z"),
            pd.Timestamp("2024-01-01T00:00:00Z"),
            freq="1h",
        )
        return pd.DataFrame(
            {"high": 101.0, "low": 99.0, "close": 100.0}, index=index
        )


def _inputs(
    tmp_path,
    *,
    outcome_column: bool = False,
    inverse_pi: bool = False,
    final_causal_inverse_pi: bool = False,
):
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    staged = pd.DataFrame(
        {
            "candidate_id": ["candidate-1"],
            "signal_timestamp": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "decision_timestamp": [pd.Timestamp("2024-01-01T01:00:00Z")],
            "path_end_exclusive": [pd.Timestamp("2024-01-01T13:00:00Z")],
            "symbol": ["BTC/USD:USD"],
            "side_name": ["long"],
            "archetype_policy_key": ["long_breakout_diagnostic_candidate"],
            "__barrier_pct__": [0.02],
            "evidence_scope": ["frozen_backcast_diagnostic"],
            "lineage": ["historical_frozen_backcast_exact1m_research_only"],
            "execution_parity_claim": [False],
            "promotion_eligible": [False],
        }
    )
    if outcome_column:
        staged["execution_net_ev_12h"] = 0.1
    if inverse_pi:
        staged["evidence_scope"] = "inverse_pi_market_grid_bootstrap_research"
        staged["lineage"] = "historical_inverse_pi_market_grid_exact1m_research_only"
        staged["candidate_population_lineage"] = (
            "jan_jul_2022_inverse_pi_market_grid_bootstrap_v1"
        )
        staged["source_product_lineage"] = (
            "kraken_inverse_pi_exact_product_binding_v1"
        )
        staged["source_product_id"] = "PI_XBTUSD"
        staged["source_contract_family"] = "PI"
        staged["bootstrap_barrier_data_acquisition_only"] = True
    if final_causal_inverse_pi:
        staged["symbol"] = "BTC/USD:BTC"
        staged["evidence_scope"] = "inverse_pi_market_grid_causal_features_research"
        staged["lineage"] = "historical_inverse_pi_market_grid_exact1m_research_only"
        staged["candidate_population_lineage"] = (
            "jan_jul_2022_inverse_pi_market_grid_causal_features_v1"
        )
        staged["source_product_lineage"] = (
            "kraken_inverse_pi_exact_product_binding_v1"
        )
        staged["product_id"] = "PI_XBTUSD"
        staged["policy_archetype_assignment_source"] = (
            "explicit_deployed_side_parent_inverse_grid"
        )
        staged["archetype_policy_key"] = "parent"
        staged["bootstrap_barrier_data_acquisition_only"] = False
    staged_path = stage_dir / "staged_candidates.parquet"
    staged.to_parquet(staged_path, index=False)
    stage_manifest = {
        "schema": "historical_backcast_exact1m_request_stage_v2",
        "outputs": {
            "staged_candidates": {"sha256": adapter._sha256(staged_path)}
        },
    }
    if inverse_pi:
        stage_manifest.update(
            {
                "evidence_scope": "inverse_pi_market_grid_bootstrap_research_not_oof",
                "lineage": "historical_inverse_pi_market_grid_exact1m_research_only",
                "candidate_population_lineage": "jan_jul_2022_inverse_pi_market_grid_bootstrap_v1",
                "product_lineage": "kraken_inverse_pi_exact_product_binding_v1",
                "bootstrap_barrier_data_acquisition_only": True,
            }
        )
    if final_causal_inverse_pi:
        stage_manifest.update(
            {
                "evidence_scope": "inverse_pi_market_grid_causal_features_research_not_oof",
                "lineage": "historical_inverse_pi_market_grid_exact1m_research_only",
                "candidate_population_lineage": "jan_jul_2022_inverse_pi_market_grid_causal_features_v1",
                "product_lineage": "kraken_inverse_pi_exact_product_binding_v1",
                "bootstrap_barrier_data_acquisition_only": False,
                "economics_contract": "inverse_quote_notional_current_spread_counterfactual_only",
                "return_unit": "quote_notional_price_return_not_inverse_collateral_roe",
                "parent_policy_binding": {
                    "assignment_source": "explicit_deployed_side_parent_inverse_grid",
                    "archetype_policy_key": "parent",
                    "side_policy_keys": {"long": "long__parent", "short": "short__parent"},
                },
            }
        )
    (stage_dir / "manifest.json").write_text(json.dumps(stage_manifest))
    product_manifest = tmp_path / "product_manifest.json"
    product_manifest.write_text(
        json.dumps(
            {
                "schema": "kraken_historical_product_map_v1",
                "stage_candidates": {"sha256": adapter._sha256(staged_path)},
            }
        )
    )
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    return stage_dir, product_manifest, policy


def _run(
    monkeypatch,
    tmp_path,
    *,
    outcome_column=False,
    inverse_pi=False,
    final_causal_inverse_pi=False,
):
    stage_dir, product_manifest, policy = _inputs(
        tmp_path,
        outcome_column=outcome_column,
        inverse_pi=inverse_pi,
        final_causal_inverse_pi=final_causal_inverse_pi,
    )
    monkeypatch.setattr(adapter, "PartitionedOHLCVStore", _Store)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_historical_backcast_exact1m_label_inputs.py",
            "--stage-dir",
            str(stage_dir),
            "--product-map-manifest",
            str(product_manifest),
            "--hourly-root",
            str(tmp_path / "hourly"),
            "--policy-json",
            str(policy),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    return adapter.main()


def test_adapter_materializes_exact_identity_and_causal_atr(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(monkeypatch, tmp_path) == 0
    targets = pd.read_parquet(tmp_path / "output" / "path_targets.parquet")
    context = pd.read_parquet(tmp_path / "output" / "context.parquet")
    assert targets.loc[0, "__path_auxiliary_atr_fraction__"] == pytest.approx(0.02)
    assert targets.loc[0, "atr_available_at"] == pd.Timestamp(
        "2024-01-01T00:00:00Z"
    )
    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert manifest["oof_status"] == "not_oof"
    assert manifest["execution_parity_claim"] is False
    assert manifest["atr_90d_uninterrupted_rows"] == 1
    assert context.loc[0, "policy_archetype"] == (
        "long__long_breakout_diagnostic_candidate"
    )


def test_adapter_rejects_outcome_columns(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="forbidden outcome columns"):
        _run(monkeypatch, tmp_path, outcome_column=True)


def test_adapter_propagates_validated_inverse_pi_stage_lineage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(monkeypatch, tmp_path, inverse_pi=True) == 0

    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert manifest["evidence_scope"] == (
        "inverse_pi_market_grid_bootstrap_research_not_oof"
    )
    assert manifest["lineage"] == "historical_inverse_pi_market_grid_exact1m_research_only"
    assert manifest["candidate_population_lineage"] == (
        "jan_jul_2022_inverse_pi_market_grid_bootstrap_v1"
    )
    assert manifest["product_lineage"] == "kraken_inverse_pi_exact_product_binding_v1"
    assert manifest["bootstrap_barrier_data_acquisition_only"] is True


def test_adapter_rejects_manifest_lineage_disagreement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir, product_manifest, policy = _inputs(tmp_path, inverse_pi=True)
    stage_manifest_path = stage_dir / "manifest.json"
    payload = json.loads(stage_manifest_path.read_text())
    payload["candidate_population_lineage"] = "wrong_population"
    stage_manifest_path.write_text(json.dumps(payload))
    monkeypatch.setattr(adapter, "PartitionedOHLCVStore", _Store)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_historical_backcast_exact1m_label_inputs.py",
            "--stage-dir", str(stage_dir),
            "--product-map-manifest", str(product_manifest),
            "--hourly-root", str(tmp_path / "hourly"),
            "--policy-json", str(policy),
            "--output-dir", str(tmp_path / "output"),
        ],
    )
    with pytest.raises(ValueError, match="candidate_population_lineage"):
        adapter.main()


def test_adapter_propagates_final_causal_inverse_economics_and_lineage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(monkeypatch, tmp_path, final_causal_inverse_pi=True) == 0

    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert manifest["evidence_scope"] == "inverse_pi_market_grid_causal_features_research_not_oof"
    assert manifest["candidate_population_lineage"] == (
        "jan_jul_2022_inverse_pi_market_grid_causal_features_v1"
    )
    assert manifest["economics"] == "inverse_quote_notional_current_spread_counterfactual"
    assert manifest["return_unit"] == "quote_notional_price_return_not_inverse_collateral_roe"
    assert manifest["parent_policy_binding"]["archetype_policy_key"] == "parent"
    assert manifest["promotion_eligible"] is False


def test_adapter_rejects_final_causal_inverse_bootstrap_marker(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir, product_manifest, policy = _inputs(tmp_path, final_causal_inverse_pi=True)
    staged_path = stage_dir / "staged_candidates.parquet"
    staged = pd.read_parquet(staged_path)
    staged["bootstrap_barrier_data_acquisition_only"] = True
    staged.to_parquet(staged_path, index=False)
    payload = json.loads((stage_dir / "manifest.json").read_text())
    payload["outputs"]["staged_candidates"]["sha256"] = adapter._sha256(staged_path)
    (stage_dir / "manifest.json").write_text(json.dumps(payload))
    product_payload = json.loads(product_manifest.read_text())
    product_payload["stage_candidates"]["sha256"] = adapter._sha256(staged_path)
    product_manifest.write_text(json.dumps(product_payload))
    monkeypatch.setattr(adapter, "PartitionedOHLCVStore", _Store)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_historical_backcast_exact1m_label_inputs.py",
            "--stage-dir", str(stage_dir),
            "--product-map-manifest", str(product_manifest),
            "--hourly-root", str(tmp_path / "hourly"),
            "--policy-json", str(policy),
            "--output-dir", str(tmp_path / "output"),
        ],
    )
    with pytest.raises(ValueError, match="must not carry a bootstrap barrier"):
        adapter.main()
