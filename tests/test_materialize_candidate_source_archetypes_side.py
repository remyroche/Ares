from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_candidate_source_archetypes_v2 import run_materialization


def test_archetype_materialization_expands_side_blind_source_tags_from_label_contract(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
    source_tags = pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC"],
            "primary_source_tag": ["risk_adjusted_capture_candidate"],
            "base_positive_source_score": [0.9],
            "calm_positive_source_score": [0.8],
            "risk_adjusted_capture_candidate_score": [0.9],
            "compression_capture_candidate_score": [0.1],
            "dirty_shock_avoid_score": [1.0],
            "execution_quality_score": [0.8],
            "barrier_relief_score": [0.8],
        }
    )
    candidate_ids = [
        "BTC|2026-04-01T12:00:00Z|1h|long",
        "BTC|2026-04-01T12:00:00Z|1h|short",
    ]
    labels = pd.DataFrame(
        {
            "__ts__": [ts, ts],
            "__symbol__": ["BTC", "BTC"],
            "side": [1, -1],
            "side_name": ["long", "short"],
            "timeframe": ["1h", "1h"],
            "candidate_id": candidate_ids,
            "__u_policy_net__": [0.01, -0.02],
            "__is_timeout__": [0.0, 1.0],
            "__bars_policy__": [3.0, 24.0],
            "__barrier_pct__": [0.01, 0.01],
            "__mae_ret__": [0.002, 0.012],
            "__mfe_ret__": [0.010, 0.001],
            "__y_ret__": [0.011, -0.013],
            "__y_bin__": [1.0, 0.0],
        }
    )
    source_path = tmp_path / "source_tags.parquet"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "archetypes"
    labels_dir.mkdir()
    source_tags.to_parquet(source_path, index=False)
    labels.to_parquet(labels_dir / "part.parquet", index=False)

    manifest = run_materialization(
        source_tags_path=source_path,
        labels_path=labels_dir,
        output_dir=output_dir,
        timestamp_col="__ts__",
        symbol_col="__symbol__",
        min_timestamp_rows=1,
        prior_symbol_window=2,
        min_coverage=0.0,
        max_top_symbol_share=1.0,
    )

    materialized = pd.read_parquet(output_dir / "candidate_source_archetypes_v2.parquet")
    quality = pd.read_csv(output_dir / "source_archetypes_v2_quality.csv")
    scorecard = pd.read_csv(output_dir / "source_archetypes_v2_scorecard.csv")

    assert manifest["join_report"]["join_mode"] == "candidate_id"
    assert manifest["join_report"]["joined_rows"] == 2
    assert manifest["side_counts_full"] == {"long": 1, "short": 1}
    assert manifest["side_counts_joined"] == {"long": 1, "short": 1}
    assert manifest["side_contract_report"]["side_contract_expanded"] is True
    for col in ("side", "side_name", "timeframe", "candidate_id"):
        assert col in materialized.columns
    assert set(materialized["candidate_id"].astype(str)) == set(candidate_ids)
    assert "top_side_share" in quality.columns
    assert "top_side_share" in scorecard.columns
