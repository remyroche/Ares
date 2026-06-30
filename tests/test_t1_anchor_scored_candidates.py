from pathlib import Path

import pandas as pd
import pytest

from scripts.materialize_t1_anchor_scored_candidates import materialize


def _candidate_rows() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-06-23T00:00:00Z", "2026-06-23T01:00:00Z"], utc=True)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"],
            "side": ["short", "short"],
            "strategy_id": ["short_asset_demo", "short_boll_demo"],
            "head": ["short_asset", "short_boll"],
            "calibrated_score": [0.1, 0.2],
            "reliability_blend_score": [0.1, 0.2],
            "normalized_rank_score": [0.1, 0.2],
            "strategy_rank_pct": [0.1, 0.2],
            "policy_rank_pct": [0.1, 0.2],
            "rank_pct": [0.1, 0.2],
            "base_strategy_threshold": [0.7, 0.7],
            "deployment_rank_threshold": [0.7, 0.7],
            "entry_price": [100.0, 100.0],
            "exit_price": [101.0, 99.0],
            "exit_timestamp": ts + pd.Timedelta(hours=4),
            "net_return": [0.01, -0.01],
            "gross_return": [0.011, -0.009],
            "holding_bars": [4, 4],
            "simple_policy_exit_reason": ["tp", "sl"],
        }
    )


def _score_rows() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-06-23T00:00:00Z", "2026-06-23T01:00:00Z"], utc=True)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"],
            "strategy_id": ["short_asset_demo", "short_boll_demo"],
            "calibrated_score": [0.81, 0.82],
            "policy_rank_pct": [0.91, 0.92],
            "auction_rank_pct": [0.71, 0.72],
            "base_pred": [0.41, 0.42],
            "meta_pred": [0.81, 0.82],
            "raw_prediction_score": [0.81, 0.82],
        }
    )


def test_materialize_replaces_native_scores_with_anchor_scores(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    scores_path = tmp_path / "scores.parquet"
    _candidate_rows().to_parquet(candidates_path, index=False)
    _score_rows().to_parquet(scores_path, index=False)

    manifest = materialize(
        candidates_path=candidates_path,
        score_ledger_path=scores_path,
        output_dir=tmp_path / "out",
    )

    out = pd.read_parquet(manifest["outputs"]["candidates_broad"])
    assert manifest["rows"] == 2
    assert manifest["deployable_rows"] == 2
    assert out["calibrated_score"].tolist() == pytest.approx([0.81, 0.82])
    assert out["reliability_anchor_only_score"].tolist() == pytest.approx([0.81, 0.82])
    assert out["source_reliability_blend_score"].tolist() == pytest.approx([0.1, 0.2])
    assert out["policy_rank_pct"].tolist() == pytest.approx([0.91, 0.92])
    assert out["normalized_rank_score"].tolist() == pytest.approx([0.91, 0.92])
    assert set(out["score_source"]) == {"live_finalfit_anchor_meta_score"}


def test_materialize_rejects_missing_anchor_score_rows(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    scores_path = tmp_path / "scores.parquet"
    _candidate_rows().to_parquet(candidates_path, index=False)
    _score_rows().iloc[[0]].to_parquet(scores_path, index=False)

    with pytest.raises(RuntimeError, match="Missing anchor scores"):
        materialize(
            candidates_path=candidates_path,
            score_ledger_path=scores_path,
            output_dir=tmp_path / "out",
        )


def test_materialize_preserves_empty_active_head_candidate_artifacts(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        columns=[
            "timestamp",
            "symbol",
            "strategy_id",
            "head",
            "calibrated_score",
            "policy_rank_pct",
            "normalized_rank_score",
            "base_pred",
            "meta_pred",
        ]
    ).to_parquet(candidates_path, index=False)
    pd.DataFrame(
        columns=[
            "timestamp",
            "symbol",
            "strategy_id",
            "calibrated_score",
            "policy_rank_pct",
            "auction_rank_pct",
            "base_pred",
            "meta_pred",
            "raw_prediction_score",
        ]
    ).to_parquet(scores_path, index=False)

    manifest = materialize(
        candidates_path=candidates_path,
        score_ledger_path=scores_path,
        output_dir=tmp_path / "out",
    )

    assert manifest["rows"] == 0
    assert manifest["deployable_rows"] == 0
    assert manifest["empty_candidate_ledger"] is True
    broad = pd.read_parquet(manifest["outputs"]["candidates_broad"])
    deployable = pd.read_parquet(manifest["outputs"]["simple_policy_candidates"])
    assert broad.empty
    assert deployable.empty
    assert "timestamp" in broad.columns
