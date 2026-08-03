import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_historical_execution_ev_mapping_repair import (  # noqa: E402
    adjacent_selected_asset_turnover,
    validate_gate_manifest,
)


def test_asset_turnover_does_not_treat_timestamp_candidate_ids_as_different_assets():
    rows = pd.DataFrame(
        {
            "candidate_id": ["BTC|one", "BTC|two", "ETH|two"],
            "__symbol__": ["BTC", "BTC", "ETH"],
            "__ts__": pd.to_datetime(
                ["2025-04-01T00:00:00Z", "2025-04-01T01:00:00Z", "2025-04-01T01:00:00Z"]
            ),
        }
    )
    result = adjacent_selected_asset_turnover(rows, pd.Series([True, True, False]), "h")
    assert result == {"comparisons": 1, "selected_asset_jaccard_mean": 1.0, "selected_asset_turnover": 0.0}


def test_source_manifest_enforcement_rejects_hash_mismatch(tmp_path):
    source = tmp_path / "residual_only"
    source.mkdir()
    inner = source / "march_inner_oof_scores.parquet"
    outer = source / "april_outer_predictions.parquet"
    pd.DataFrame({"candidate_id": ["inner"]}).to_parquet(inner)
    pd.DataFrame({"candidate_id": ["outer"]}).to_parquet(outer)
    manifest = {
        "schema": "historical_execution_ev_add_drop_gate_v6",
        "status": "research_only_diagnostic",
        "outputs": {
            "residual_only": {
                "march_inner_oof_scores": {"path": "residual_only/march_inner_oof_scores.parquet", "sha256": "not-a-real-hash", "rows": 1},
                "april_outer_predictions": {"path": "residual_only/april_outer_predictions.parquet", "sha256": hashlib.sha256(outer.read_bytes()).hexdigest(), "rows": 1},
            }
        },
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_gate_manifest(tmp_path, manifest, arms=("residual_only",))
