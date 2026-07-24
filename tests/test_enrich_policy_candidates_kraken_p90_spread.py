from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import enrich_policy_candidates_kraken_p90_spread as subject


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BBB_USD:USD", "AAA/USD:USD"],
            "score_base": np.array([0.123456789, 0.987654321], dtype=np.float64),
            "expected_net_ev_after_1pct": np.array([0.02, -0.01], dtype=np.float32),
            "admitted": [True, False],
            "policy_archetype": ["long_default", "long_breakout"],
        },
        index=pd.Index([11, 7], name="input_row"),
    )


def _spreads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"],
            "p90_spread_bps": [10.5, 25.25],
        }
    )


def test_enrichment_normalizes_symbols_and_preserves_all_existing_values_and_order() -> None:
    candidates = _candidates()
    mapping, lineage = subject.build_spread_mapping(_spreads())

    enriched, audit = subject.enrich_candidates(
        candidates,
        mapping,
        mapping_sha256=lineage["mapping_sha256"],
    )

    pd.testing.assert_frame_equal(enriched.loc[:, candidates.columns], candidates, check_exact=True)
    assert enriched.index.equals(candidates.index)
    assert len(enriched) == len(candidates)
    assert enriched[subject.OUTPUT_SPREAD_BPS_COLUMN].tolist() == [25.25, 10.5]
    assert enriched[subject.OUTPUT_SPREAD_RETURN_COLUMN].tolist() == pytest.approx([0.002525, 0.00105])
    assert enriched[subject.OUTPUT_POLICY_SPREAD_BPS_COLUMN].tolist() == [25.25, 10.5]
    assert enriched[subject.OUTPUT_COMPAT_P90_SPREAD_BPS_COLUMN].tolist() == [25.25, 10.5]
    assert enriched[subject.OUTPUT_MAPPING_HASH_COLUMN].nunique() == 1
    assert audit["costs_applied"] is False
    assert audit["protected_columns_sha256_before"] == audit["protected_columns_sha256_after"]


def test_duplicate_or_ambiguous_normalized_source_symbols_fail_closed() -> None:
    spreads = pd.DataFrame(
        {
            "symbol": ["AAA/USD:USD", "aaa_usd:usd"],
            "p90_spread_bps": [10.0, 20.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate/ambiguous"):
        subject.build_spread_mapping(spreads)
    with pytest.raises(ValueError, match="disagree"):
        subject.build_spread_mapping(spreads, duplicate_policy="require_equal")


def test_explicit_duplicate_policy_can_allow_identical_aliases_only() -> None:
    spreads = pd.DataFrame(
        {
            "symbol": ["AAA/USD:USD", "aaa_usd:usd"],
            "p90_spread_bps": [10.0, 10.0],
        }
    )
    mapping, lineage = subject.build_spread_mapping(spreads, duplicate_policy="require_equal")
    assert len(mapping) == 1
    assert lineage["duplicate_source_rows"] == 2


def test_missing_candidate_mapping_fails_closed_unless_explicitly_allowed() -> None:
    candidates = _candidates()
    candidates.loc[7, "symbol"] = "MISSING/USD:USD"
    mapping, _ = subject.build_spread_mapping(_spreads())
    with pytest.raises(ValueError, match="no unambiguous p90 spread mapping"):
        subject.enrich_candidates(candidates, mapping)

    enriched, audit = subject.enrich_candidates(candidates, mapping, missing_policy="allow_null")
    assert np.isnan(enriched.loc[7, subject.OUTPUT_SPREAD_BPS_COLUMN])
    assert np.isnan(enriched.loc[7, subject.OUTPUT_SPREAD_RETURN_COLUMN])
    assert audit["missing_spread_rows"] == 1


def test_existing_optimizer_spread_must_match_explicit_mapping() -> None:
    candidates = _candidates()
    candidates[subject.OUTPUT_POLICY_SPREAD_BPS_COLUMN] = [25.25, 999.0]
    mapping, _ = subject.build_spread_mapping(_spreads())
    with pytest.raises(ValueError, match="conflicts"):
        subject.enrich_candidates(candidates, mapping)

    candidates.loc[:, subject.OUTPUT_POLICY_SPREAD_BPS_COLUMN] = [25.25, 10.5]
    enriched, audit = subject.enrich_candidates(candidates, mapping)
    pd.testing.assert_series_equal(
        enriched[subject.OUTPUT_POLICY_SPREAD_BPS_COLUMN],
        candidates[subject.OUTPUT_POLICY_SPREAD_BPS_COLUMN],
        check_exact=True,
    )
    assert subject.OUTPUT_POLICY_SPREAD_BPS_COLUMN not in audit["canonical_columns_added"]


def test_cli_writes_lineage_without_changing_candidate_scores_or_ev(tmp_path: Path) -> None:
    candidates = _candidates()
    candidate_path = tmp_path / "candidates.parquet"
    spread_path = tmp_path / "eligible.csv"
    output_path = tmp_path / "candidates_p90.parquet"
    candidates.to_parquet(candidate_path, index=False)
    _spreads().to_csv(spread_path, index=False)

    assert subject.main(
        [
            "--candidates",
            str(candidate_path),
            "--eligible-spreads",
            str(spread_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    enriched = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(enriched.loc[:, candidates.columns], candidates.reset_index(drop=True), check_exact=True)
    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text())
    assert manifest["enrichment"]["cost_contract"] == "enrichment_only_no_fee_or_spread_deduction"
    assert manifest["candidate_input_sha256"]
    assert manifest["eligible_spreads_input_sha256"]
    assert manifest["output_sha256"]
    assert manifest["mapping"]["mapping_sha256"] == enriched[subject.OUTPUT_MAPPING_HASH_COLUMN].iloc[0]
