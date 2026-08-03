from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from scripts import materialize_historical_backcast_exact1m_stage as stage


def _row(ts: str, *, barrier: float = 0.02) -> dict[str, object]:
    return {
        "__ts__": pd.Timestamp(ts),
        "__symbol__": "BTC/USD:USD",
        "side_name": "long",
        "__barrier_pct__": barrier,
        "archetype_policy_key": "long_breakout_diagnostic_candidate",
        "selected_for_monitor": True,
        "evidence_scope": "frozen_backcast_diagnostic",
        "base_score": 0.7,
        "historical_rank": 1,
    }


def _run(monkeypatch: pytest.MonkeyPatch, source, output, *extra: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_historical_backcast_exact1m_stage.py",
            "--candidate-root",
            str(source),
            "--output-dir",
            str(output),
            *extra,
        ],
    )
    assert stage.main() == 0


def test_stage_freezes_source_native_identity_and_exact_window(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        [
            _row("2024-01-01T00:00:00Z"),
            _row("2024-04-01T00:00:00Z"),
        ]
    ).to_parquet(source / "candidates_202401.parquet", index=False)
    output = tmp_path / "output"

    _run(
        monkeypatch,
        source,
        output,
        "--signal-start",
        "2024-01-01T00:00:00Z",
        "--signal-end-exclusive",
        "2024-04-01T00:00:00Z",
    )

    candidates = pd.read_parquet(output / "staged_candidates.parquet")
    assert len(candidates) == 1
    assert candidates["candidate_id"].is_unique
    assert candidates.loc[0, "source_row_number"] == 0
    assert candidates.loc[0, "decision_timestamp"] == pd.Timestamp(
        "2024-01-01T01:00:00Z"
    )
    assert candidates.loc[0, "path_end_exclusive"] == pd.Timestamp(
        "2024-01-01T13:00:00Z"
    )
    path_map = pd.read_parquet(output / "candidate_path_map.parquet")
    assert path_map["candidate_id"].tolist() == candidates["candidate_id"].tolist()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == "historical_backcast_exact1m_request_stage_v2"
    assert manifest["logical_collision_policy"] == "fail_closed"
    assert manifest["sources"][0]["selected_rows_after_time_filter"] == 1


def test_stage_fails_closed_on_logical_identity_collision(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    row = _row("2024-01-01T00:00:00Z")
    pd.DataFrame([row, dict(row)]).to_parquet(
        source / "candidates_202401.parquet", index=False
    )

    with pytest.raises(ValueError, match="logical-identity collision"):
        _run(monkeypatch, source, tmp_path / "output")


def test_geometry_changes_source_native_candidate_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        [
            _row("2024-01-01T00:00:00Z", barrier=0.02),
            _row("2024-01-01T00:00:00Z", barrier=0.03),
        ]
    ).to_parquet(source / "candidates_202401.parquet", index=False)

    _run(monkeypatch, source, tmp_path / "output")
    candidates = pd.read_parquet(tmp_path / "output" / "staged_candidates.parquet")
    assert len(candidates) == 2
    assert candidates["candidate_id"].nunique() == 2


def test_stage_can_fail_closed_to_usd_settled_symbol_lineage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    inverse = _row("2022-07-14T00:00:00Z")
    inverse["__symbol__"] = "BTC/USD:BTC"
    pd.DataFrame(
        [_row("2022-07-14T00:00:00Z"), inverse]
    ).to_parquet(source / "candidates_202207.parquet", index=False)

    _run(
        monkeypatch,
        source,
        tmp_path / "output",
        "--require-symbol-suffix",
        ":USD",
    )
    candidates = pd.read_parquet(tmp_path / "output" / "staged_candidates.parquet")
    assert candidates["symbol"].tolist() == ["BTC/USD:USD"]


def _inverse_pi_row() -> dict[str, object]:
    row = _row("2022-01-15T00:00:00Z")
    row.update(
        {
            "__symbol__": "BTC/USD:BTC",
            "evidence_scope": "inverse_pi_market_grid_bootstrap_research",
            "candidate_population_lineage": "jan_jul_2022_inverse_pi_market_grid_bootstrap_v1",
            "source_product_lineage": "kraken_inverse_pi_exact_product_binding_v1",
            "source_product_id": "PI_XBTUSD",
            "source_contract_family": "PI",
            "bootstrap_barrier_data_acquisition_only": True,
        }
    )
    return row


def test_stage_preserves_source_declared_inverse_population_lineage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame([_inverse_pi_row()]).to_parquet(source / "candidates_pi.parquet", index=False)

    _run(
        monkeypatch,
        source,
        tmp_path / "output",
        "--population-lineage",
        "jan_jul_2022_inverse_pi_market_grid_bootstrap_v1",
    )

    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    candidates = pd.read_parquet(tmp_path / "output" / "staged_candidates.parquet")
    assert manifest["candidate_population_lineage"] == (
        "jan_jul_2022_inverse_pi_market_grid_bootstrap_v1"
    )
    assert manifest["lineage"] == "historical_inverse_pi_market_grid_exact1m_research_only"
    assert candidates.loc[0, "candidate_population_lineage"] == manifest[
        "candidate_population_lineage"
    ]


def test_stage_rejects_population_lineage_that_does_not_match_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame([_inverse_pi_row()]).to_parquet(source / "candidates_pi.parquet", index=False)

    with pytest.raises(ValueError, match="exactly match"):
        _run(
            monkeypatch,
            source,
            tmp_path / "output",
            "--population-lineage",
            "some_other_population",
        )


def _causal_inverse_pi_row() -> dict[str, object]:
    row = _row("2022-07-15T00:00:00Z")
    row.update(
        {
            "__symbol__": "BTC/USD:BTC",
            "evidence_scope": "inverse_pi_market_grid_causal_features_research",
            "candidate_population_lineage": "jan_jul_2022_inverse_pi_market_grid_causal_features_v1",
            "source_product_lineage": "kraken_inverse_pi_exact_product_binding_v1",
            "product_id": "PI_XBTUSD",
            "bootstrap_barrier_data_acquisition_only": False,
            "archetype_policy_key": "parent",
            "policy_archetype_assignment_source": "explicit_deployed_side_parent_inverse_grid",
            "ret_24h": 0.03,
            "transition_raw__market_median_rv_24h__delta_1h": -0.02,
        }
    )
    return row


def test_stage_preserves_final_causal_inverse_features_and_parent_binding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame([_causal_inverse_pi_row()]).to_parquet(source / "candidates_pi.parquet", index=False)

    _run(
        monkeypatch,
        source,
        tmp_path / "output",
        "--population-lineage",
        "jan_jul_2022_inverse_pi_market_grid_causal_features_v1",
    )

    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    staged = pd.read_parquet(tmp_path / "output" / "staged_candidates.parquet")
    assert manifest["evidence_scope"] == "inverse_pi_market_grid_causal_features_research_not_oof"
    assert manifest["feature_columns_preserved"] is True
    assert manifest["economics_contract"] == "inverse_quote_notional_current_spread_counterfactual_only"
    assert manifest["parent_policy_binding"]["side_policy_keys"]["long"] == "long__parent"
    assert staged.loc[0, "ret_24h"] == pytest.approx(0.03)
    assert staged.loc[0, "transition_raw__market_median_rv_24h__delta_1h"] == pytest.approx(-0.02)


def test_stage_rejects_final_causal_inverse_without_parent_binding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    row = _causal_inverse_pi_row()
    row["archetype_policy_key"] = "invented_archetype"
    pd.DataFrame([row]).to_parquet(source / "candidates_pi.parquet", index=False)

    with pytest.raises(ValueError, match="parent policy key"):
        _run(monkeypatch, source, tmp_path / "output")
