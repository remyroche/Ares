from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import evaluate_j4_j5_contextual_meta_fresh_oos as mod


def _write_freeze(tmp_path: Path) -> Path:
    path = tmp_path / "freeze.csv"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "rank_threshold": 0.70,
                "effective_fresh_oos_after": "2026-06-15T04:00:00+00:00",
                "selected_contextual_feature_arm": "B_current_plus_model_state",
                "selected_capacity_config": "none_retain_context_arm",
                "selected_distillation_variant": "hard_label_context_arm",
            }
        ]
    ).to_csv(path, index=False)
    return path


def _write_audit(tmp_path: Path, status: str) -> Path:
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps({"status": status, "items": []}))
    return path


def test_evaluator_refuses_when_readiness_not_ready(tmp_path: Path) -> None:
    summary, timestamps, audit = mod.evaluate(
        freeze_manifest_path=_write_freeze(tmp_path),
        readiness_audit_path=_write_audit(tmp_path, "not_ready"),
        score_dirs=[],
        score_files=[],
        label_dir=tmp_path,
        min_later_hours=24.0,
        min_timestamp_rows=3,
        require_ready=True,
    )

    assert summary.empty
    assert timestamps.empty
    assert audit["status"] == "not_ready"


def test_evaluator_computes_timestamp_local_top30_metrics(tmp_path: Path) -> None:
    freeze = _write_freeze(tmp_path)
    readiness = _write_audit(tmp_path, "ready")
    rows = []
    for ts in ["2026-06-16T05:00:00+00:00", "2026-06-16T06:00:00+00:00"]:
        for i in range(10):
            # Baseline ranks the first three negatives highest; candidate ranks
            # positives highest. With top30, k=3 for each timestamp.
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": f"S{i}",
                    "baseline_score": 1.0 - i * 0.01,
                    "candidate_score": i * 0.01,
                    "y_bin": 1 if i >= 7 else 0,
                }
            )
    score_file = tmp_path / "fresh_short_asset_scores.parquet"
    pd.DataFrame(rows).to_parquet(score_file)

    summary, timestamps, audit = mod.evaluate(
        freeze_manifest_path=freeze,
        readiness_audit_path=readiness,
        score_dirs=[],
        score_files=[score_file],
        label_dir=tmp_path,
        min_later_hours=24.0,
        min_timestamp_rows=3,
        require_ready=True,
    )

    assert audit["status"] == "passed"
    assert len(timestamps) == 2
    row = summary.iloc[0]
    assert row["timestamp_count"] == 2
    assert row["timestamp_weighted_delta_hr_top30"] == 1.0
    assert row["timestamp_weighted_delta_ndcg_top30"] > 0
    assert row["total_net_correct_trades_gained"] == 6


def test_evaluator_filters_rows_before_guard(tmp_path: Path) -> None:
    freeze = _write_freeze(tmp_path)
    readiness = _write_audit(tmp_path, "ready")
    score_file = tmp_path / "fresh_short_asset_scores.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-15T05:00:00+00:00",
                "baseline_score": 0.1,
                "candidate_score": 0.9,
                "y_bin": 1,
            }
            for _ in range(5)
        ]
    ).to_parquet(score_file)

    summary, timestamps, audit = mod.evaluate(
        freeze_manifest_path=freeze,
        readiness_audit_path=readiness,
        score_dirs=[],
        score_files=[score_file],
        label_dir=tmp_path,
        min_later_hours=24.0,
        min_timestamp_rows=3,
        require_ready=True,
    )

    assert summary.empty
    assert timestamps.empty
    assert audit["status"] == "failed"


def test_top_k_avoids_float_boundary_expansion() -> None:
    assert mod._top_k(10, 1.0 - 0.70) == 3
