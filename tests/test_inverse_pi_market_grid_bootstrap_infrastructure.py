from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import materialize_historical_backcast_exact1m_stage as stage
from scripts import materialize_inverse_pi_market_grid_candidate_source as source
from scripts import materialize_kraken_historical_inverse_pi_product_map as product_map


class _Exchange:
    def load_markets(self):
        return {}


def _stage_from_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    candidate_root = tmp_path / "source"
    candidate_root.mkdir(parents=True)
    first_hour = source.build_population().loc[
        lambda frame: frame["__ts__"].eq(source.START)
    ]
    first_hour.to_parquet(candidate_root / "candidates_202201.parquet", index=False)
    stage_dir = tmp_path / "stage"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_historical_backcast_exact1m_stage.py",
            "--candidate-root",
            str(candidate_root),
            "--output-dir",
            str(stage_dir),
        ],
    )
    assert stage.main() == 0
    return stage_dir


def _stage_from_causal_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    candidate_root = tmp_path / "source"
    candidate_root.mkdir(parents=True)
    causal = source.build_population().loc[
        lambda frame: frame["__ts__"].eq(source.START)
    ].copy()
    causal["evidence_scope"] = stage.INVERSE_CAUSAL_SCOPE
    causal["candidate_population_lineage"] = stage.INVERSE_CAUSAL_POPULATION_LINEAGE
    causal["bootstrap_barrier_data_acquisition_only"] = False
    causal["policy_archetype_assignment_source"] = (
        stage.INVERSE_PARENT_ASSIGNMENT_SOURCE
    )
    causal["product_id"] = causal["source_product_id"]
    causal.to_parquet(candidate_root / "candidates_202201.parquet", index=False)
    stage_dir = tmp_path / "stage"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_historical_backcast_exact1m_stage.py",
            "--candidate-root",
            str(candidate_root),
            "--output-dir",
            str(stage_dir),
            "--population-lineage",
            stage.INVERSE_CAUSAL_POPULATION_LINEAGE,
        ],
    )
    assert stage.main() == 0
    return stage_dir


def test_inverse_grid_is_deterministic_paired_and_has_fixed_dates() -> None:
    first = source.build_population()
    second = source.build_population()
    pd.testing.assert_frame_equal(first, second)

    expected_hours = 212 * 24
    assert len(first) == expected_hours * len(source.INVERSE_PRODUCTS) * len(source.SIDES)
    assert first["__ts__"].min() == source.START
    assert first["__ts__"].max() == source.END_EXCLUSIVE - pd.Timedelta(hours=1)
    assert first["source_candidate_id"].is_unique
    assert first.groupby(["__ts__", "__symbol__"])["side_name"].nunique().eq(2).all()
    assert first.groupby("side_name").size().to_dict() == {
        "long": expected_hours * len(source.INVERSE_PRODUCTS),
        "short": expected_hours * len(source.INVERSE_PRODUCTS),
    }
    assert set(first["archetype_policy_key"]) == {"parent"}
    assert set(
        first.assign(
            signed_key=first["side_name"] + "__" + first["archetype_policy_key"]
        )["signed_key"]
    ) == {"long__parent", "short__parent"}
    assert first["bootstrap_barrier_data_acquisition_only"].all()
    assert first["__barrier_pct__"].eq(source.BOOTSTRAP_BARRIER_PCT).all()


def test_inverse_grid_source_writes_versioned_shards_and_manifest(tmp_path: Path) -> None:
    manifest = source.run(tmp_path)
    assert manifest["schema"] == source.SCHEMA
    assert manifest["rows"] == 212 * 24 * len(source.INVERSE_PRODUCTS) * len(source.SIDES)
    assert len(manifest["candidate_shards"]) == 7
    assert manifest["bootstrap_barrier"]["data_acquisition_only"] is True
    assert manifest["inverse_product_limitations"]["notional_return_comparable_to_usd_linear_pf"] is False
    assert all(Path(record["path"]).is_file() for record in manifest["candidate_shards"])


def test_inverse_grid_stage_preserves_separate_lineage_and_product_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = _stage_from_grid(tmp_path, monkeypatch)
    staged = pd.read_parquet(stage_dir / "staged_candidates.parquet")
    manifest = json.loads((stage_dir / "manifest.json").read_text())

    assert set(staged["source_product_id"]) == set(dict(source.INVERSE_PRODUCTS).values())
    assert set(staged["source_contract_family"]) == {"PI"}
    assert set(staged["archetype_policy_key"]) == {"parent"}
    assert set(staged["side_name"] + "__" + staged["archetype_policy_key"]) == {
        "long__parent",
        "short__parent",
    }
    assert manifest["candidate_population_lineage"] == source.POPULATION_LINEAGE
    assert manifest["product_lineage"] == source.PRODUCT_LINEAGE
    assert manifest["bootstrap_barrier_data_acquisition_only"] is True
    assert manifest["evidence_scope"] == product_map.STAGE_EVIDENCE_SCOPE


def test_inverse_product_map_binds_exact_pi_products_and_records_limitations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = _stage_from_grid(tmp_path, monkeypatch)
    monkeypatch.setattr(product_map, "make_perp_exchange", lambda: _Exchange())
    calls: list[str] = []

    def _probe(exchange, symbol, since, until, **kwargs):
        del exchange, since, until
        calls.append(f"{symbol}|{kwargs['product_id']}")
        return pd.DataFrame({"close": [1.0]})

    monkeypatch.setattr(product_map, "_fetch_kraken_futures_charts_ohlcv", _probe)
    manifest = product_map.run(stage_dir, tmp_path / "products", probe_hours=6)
    frozen = pd.read_parquet(tmp_path / "products" / "product_map.parquet")

    assert dict(zip(frozen["symbol"], frozen["product_id"], strict=True)) == dict(source.INVERSE_PRODUCTS)
    assert set(frozen["contract_family"]) == {"PI"}
    assert frozen["inverse"].all()
    assert not frozen["notional_return_comparable_to_usd_linear_pf"].any()
    assert len(calls) == 2 * len(source.INVERSE_PRODUCTS)
    assert manifest["schema"] == "kraken_historical_product_map_v1"
    assert manifest["inverse_pi_allowed"] is True
    assert manifest["fallback_mapping_allowed"] is False
    assert manifest["inverse_contract_limitations"]["must_not_be_pooled_with_usd_linear_pf_population"] is True
    assert manifest["promotion_eligible"] is False


def test_inverse_product_map_rejects_mismatched_binding_and_empty_boundary_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = _stage_from_grid(tmp_path, monkeypatch)
    staged_path = stage_dir / "staged_candidates.parquet"
    staged = pd.read_parquet(staged_path)
    staged.loc[staged["symbol"].eq("BTC/USD:BTC"), "source_product_id"] = "PI_WRONG"
    staged.to_parquet(staged_path, index=False)
    stage_manifest_path = stage_dir / "manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text())
    stage_manifest["outputs"]["staged_candidates"]["sha256"] = product_map._sha256(staged_path)
    stage_manifest_path.write_text(json.dumps(stage_manifest))
    monkeypatch.setattr(product_map, "make_perp_exchange", lambda: _Exchange())
    with pytest.raises(ValueError, match="exact frozen PI product"):
        product_map.run(stage_dir, tmp_path / "products")

    clean_stage = _stage_from_grid(tmp_path / "clean", monkeypatch)
    monkeypatch.setattr(product_map, "make_perp_exchange", lambda: _Exchange())
    monkeypatch.setattr(
        product_map,
        "_fetch_kraken_futures_charts_ohlcv",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="no trade candles"):
        product_map.run(clean_stage, tmp_path / "empty-probe")


def test_inverse_product_map_accepts_final_causal_parent_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = _stage_from_causal_grid(tmp_path, monkeypatch)
    monkeypatch.setattr(product_map, "make_perp_exchange", lambda: _Exchange())
    monkeypatch.setattr(
        product_map,
        "_fetch_kraken_futures_charts_ohlcv",
        lambda *args, **kwargs: pd.DataFrame({"close": [1.0]}),
    )

    manifest = product_map.run(stage_dir, tmp_path / "products", probe_hours=6)
    frozen = pd.read_parquet(tmp_path / "products" / "product_map.parquet")

    assert manifest["evidence_scope"] == product_map.CAUSAL_STAGE_EVIDENCE_SCOPE
    assert manifest["candidate_population_lineage"] == (
        stage.INVERSE_CAUSAL_POPULATION_LINEAGE
    )
    assert manifest["bootstrap_barrier_data_acquisition_only"] is False
    assert manifest["parent_policy_binding"]["side_policy_keys"] == {
        "long": "long__parent",
        "short": "short__parent",
    }
    assert set(frozen["mapping_source"]) == {
        "causal_source_exact_inverse_pi_binding_no_catalogue_fallback"
    }


def test_inverse_product_map_rejects_final_causal_nonparent_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = _stage_from_causal_grid(tmp_path, monkeypatch)
    staged_path = stage_dir / "staged_candidates.parquet"
    staged = pd.read_parquet(staged_path)
    staged.loc[:, "archetype_policy_key"] = "invented"
    staged.to_parquet(staged_path, index=False)
    manifest_path = stage_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["staged_candidates"]["sha256"] = product_map._sha256(
        staged_path
    )
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(product_map, "make_perp_exchange", lambda: _Exchange())

    with pytest.raises(ValueError, match="parent policy key"):
        product_map.run(stage_dir, tmp_path / "products")
