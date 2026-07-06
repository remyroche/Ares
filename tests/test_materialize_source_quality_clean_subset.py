from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_source_quality_clean_subset import materialize_clean_subset  # noqa: E402


def _quality_frame() -> pd.DataFrame:
    ts = pd.date_range("2026-04-01", periods=4, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "BTC/USD:USD", "SOL/USD:USD"],
            "tag_quiet_continuation": [True, False, True, False],
            "tag_dirty_shock_avoid": [False, True, False, False],
            "primary_source_tag": ["quiet_continuation", "dirty_shock_avoid", "quiet_continuation", "ambiguous_none"],
            "quality_label_v0": [1, 0, -1, 0],
            "quality_label_source_rank_v1": [1, 0, -1, 0],
            "quality_label_source_wf_v1": [1, -1, -1, 0],
            "quality_label_clean_path_v2": [1, 0, -1, 0],
            "quality_label_recoverable_opportunity_v2": [1, 1, -1, 0],
            "quality_label_opportunity_capture_v3": [-1, 1, -1, 0],
            "quality_label_economic_capture_v4": [-1, 0, -1, 1],
            "sample_weight_base_v0": [1.0, 1.0, 0.0, 1.0],
        }
    )


def _label_frame(*, duplicate_candidate: bool = False) -> pd.DataFrame:
    ts = pd.date_range("2026-04-01", periods=4, freq="h", tz="UTC")
    candidate_ids = ["c0", "c1", "c2", "c3"]
    if duplicate_candidate:
        candidate_ids[3] = "c2"
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "BTC/USD:USD", "SOL/USD:USD"],
            "candidate_id": candidate_ids,
            "side": [1, 1, 1, 1],
            "side_name": ["long"] * 4,
            "timeframe": ["1h"] * 4,
            "__barrier_pct__": [0.01, 0.02, 0.01, 0.03],
            "__mfe_ret__": [0.03, 0.01, 0.00, 0.05],
            "__mae_ret__": [-0.003, -0.025, -0.015, -0.01],
            "__bars_to_mfe__": [2, 5, 24, 3],
            "__bars_policy__": [4, 24, 24, 6],
            "__y_ret__": [0.02, -0.01, -0.02, 0.03],
            "__y_bin__": [1, 0, 0, 1],
            "__is_timeout__": [0, 1, 1, 0],
            "__u_policy_net__": [0.018, -0.012, -0.018, 0.026],
            "__y_outcome__": [1, 0, 0, 1],
        }
    )


def _v2_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["c0", "c1", "c2", "c3"],
            "primary_source_archetype_v2": [
                "source_evidence_archetype",
                "path_geometry_archetype",
                "timeout_holding_archetype",
                "source_freshness_archetype",
            ],
            "source_evidence_archetype_score": [0.9, 0.2, 0.1, 0.5],
            "tag_source_evidence_archetype": [True, False, False, False],
        }
    )


def test_materialize_clean_subset_exports_joined_rows_metrics_and_v2(tmp_path: Path) -> None:
    quality_path = tmp_path / "quality.parquet"
    labels_path = tmp_path / "labels.parquet"
    v2_path = tmp_path / "v2.parquet"
    output_dir = tmp_path / "out"
    _quality_frame().to_parquet(quality_path, index=False)
    _label_frame().to_parquet(labels_path, index=False)
    _v2_frame().to_parquet(v2_path, index=False)

    manifest = materialize_clean_subset(
        quality_labels_path=quality_path,
        labels_path=labels_path,
        output_dir=output_dir,
        v2_archetypes_path=v2_path,
        min_join_match_vs_labels=1.0,
        min_rows=4,
        max_duplicate_candidate_id_rows=0,
    )

    subset = pd.read_parquet(output_dir / "source_quality_clean_joined_subset.parquet")
    source_quality = pd.read_csv(output_dir / "source_quality_clean_subset_source_quality.csv")

    assert manifest["subset_status"] == "pass"
    assert manifest["join_report"]["join_match_rate_vs_labels"] == 1.0
    assert manifest["v2_merge_report"]["match_rate"] == 1.0
    assert len(subset) == 4
    assert "candidate_id" in subset.columns
    assert "mae_norm" in subset.columns
    assert "bad_mae_negative" in subset.columns
    assert "primary_source_archetype_v2" in subset.columns
    assert subset["candidate_id"].tolist() == ["c0", "c1", "c2", "c3"]
    assert set(source_quality["scope"]).issuperset({"primary_source_tag", "multi_tag"})


def test_materialize_clean_subset_fails_duplicate_candidate_ids(tmp_path: Path) -> None:
    quality_path = tmp_path / "quality.parquet"
    labels_path = tmp_path / "labels.parquet"
    output_dir = tmp_path / "out"
    _quality_frame().to_parquet(quality_path, index=False)
    _label_frame(duplicate_candidate=True).to_parquet(labels_path, index=False)

    manifest = materialize_clean_subset(
        quality_labels_path=quality_path,
        labels_path=labels_path,
        output_dir=output_dir,
        v2_archetypes_path=None,
        min_join_match_vs_labels=1.0,
        min_rows=4,
        max_duplicate_candidate_id_rows=0,
    )

    assert manifest["subset_status"] == "fail"
    assert any("duplicate_candidate_id_rows" in item for item in manifest["failures"])
