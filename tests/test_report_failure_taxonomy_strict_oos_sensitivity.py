from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_failure_taxonomy_strict_oos_sensitivity import run


def _base_ledger() -> pd.DataFrame:
    timestamps = pd.date_range("2025-02-01", periods=2, freq="D", tz="UTC").tz_localize(None)
    rows: list[dict[str, object]] = []
    for offset, timestamp in enumerate(timestamps):
        for rank, symbol in enumerate(("AAA/USD:USD", "BBB/USD:USD", "CCC/USD:USD", "DDD/USD:USD")):
            rows.append(
                {
                    "__ts__": timestamp,
                    "__symbol__": symbol,
                    "side_name": "long" if rank % 2 == 0 else "short",
                    "__archetype_policy_key__": "long_state" if rank % 2 == 0 else "short_state",
                    "score": [0.95, 0.75, 0.50, 0.25][rank],
                    "__u_policy_net__": [0.02, -0.01, 0.003, -0.002][(rank + offset) % 4],
                    "__long_path_clean_exec_label__": [1, np.nan, 0, np.nan][rank],
                    "__long_path_dirty_positive_label__": [0, np.nan, 1, np.nan][rank],
                    "__path_full_bad_mae_1r__": [0, 1, 0, 1][rank],
                    "__first_touch_timeout__": [0, 0, 1, 0][rank],
                    "__first_touch_stop__": [0, 1, 0, 1][rank],
                    "oos_fold": "fold_a",
                }
            )
    return pd.DataFrame(rows)


def _meta_ledger() -> pd.DataFrame:
    source = _base_ledger().copy()
    return pd.DataFrame(
        {
            "__ts__": pd.to_datetime(source["__ts__"], utc=True),
            "__symbol__": source["__symbol__"],
            "side_name": source["side_name"],
            "archetype_policy_key": source["__archetype_policy_key__"],
            "score_base": source["score"],
            "score": source["score"] + np.where(source["side_name"].eq("long"), 0.01, -0.01),
            "ev_after_1pct": source["__u_policy_net__"],
            "clean_exec": source["__long_path_clean_exec_label__"].fillna(0.0),
            "dirty_positive": source["__long_path_dirty_positive_label__"].fillna(0.0),
            "full_path_bad_mae_1r": source["__path_full_bad_mae_1r__"],
            "timeout": source["__first_touch_timeout__"],
            "score_base_ev_mapped": source["score"] / 100.0,
            "score_base_ev_residual_expert": source["score"] / 90.0,
            "score_base_ev_residual_expert_hier_mapped": source["score"] / 80.0,
            "meta_residual_expert_delta_ev": 0.001,
            "score_base_rank": source["score"],
            "score_base_ev_rank_train_reference": source["score"],
            "score_base_residual_ev_rank_train_reference": np.where(
                source["side_name"].eq("long"),
                1.0 - source["score"],
                source["score"],
            ),
        }
    )


def _taxonomy(root: Path) -> None:
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"source": {"provenance": "frozen_backcast_diagnostic"}})
    )
    days = pd.date_range("2025-02-01", periods=2, freq="D", tz="UTC")
    calendar = pd.DataFrame(
        {
            "day": [days[0], days[0], days[1], days[1]],
            "side_name": ["long", "short", "long", "short"],
            "archetype_policy_key": ["long_state", "short_state", "long_state", "short_state"],
            "event_block": ["event_001", "normal", "normal", "normal"],
            "adverse_event": [True, False, False, False],
        }
    )
    assignments = pd.DataFrame(
        {
            "side_name": ["long"],
            "archetype_policy_key": ["long_state"],
            "event_block": ["event_001"],
            "semantic_label": ["ranking_collapse__liquidation_pressure"],
            "failure_mode_id": ["frozen__c0"],
        }
    )
    calendar.to_parquet(root / "local_adverse_calendar.parquet", index=False)
    assignments.to_parquet(root / "local_frozen_failure_mode_semantic_assignments.parquet", index=False)


def _detector(path: Path) -> None:
    pd.DataFrame(
        {
            "day": [pd.Timestamp("2025-02-01", tz="UTC")],
            "side_name": ["long"],
            "archetype_policy_key": ["long_state"],
            "failure_mode": ["negative_ev_onset"],
            "risk": [0.8],
            "alert": [True],
            "threshold": [0.75],
            "fold_index": [1],
            "train_end": [pd.Timestamp("2025-01-31", tz="UTC")],
            "eval_end": [pd.Timestamp("2025-02-28", tz="UTC")],
            "target_horizon_days": [0],
        }
    ).to_parquet(path, index=False)


def test_strict_oos_sensitivity_separates_scopes_and_marks_taxonomy_descriptive(
    tmp_path: Path,
) -> None:
    base, meta = tmp_path / "base.parquet", tmp_path / "meta.parquet"
    taxonomy, detector, output = tmp_path / "taxonomy", tmp_path / "detector.parquet", tmp_path / "report"
    _base_ledger().to_parquet(base, index=False)
    _meta_ledger().to_parquet(meta, index=False)
    _taxonomy(taxonomy)
    _detector(detector)

    manifest = run(
        base_ledger=base,
        meta_ledger=meta,
        taxonomy=taxonomy,
        detector=detector,
        output=output,
    )

    assert manifest["taxonomy_not_full_oos"] is True
    assert manifest["taxonomy_source_provenance"] == "frozen_backcast_diagnostic"
    base_coverage = pd.read_csv(output / "strict_base_oos_coverage.csv")
    meta_coverage = pd.read_csv(output / "strict_base_meta_oos_coverage.csv")
    assert base_coverage.loc[0, "strict_rows"] == 8
    assert meta_coverage.loc[0, "strict_rows"] == 8
    assert "2025-02-01" in str(base_coverage.loc[0, "start"])

    modes = pd.read_csv(output / "strict_base_oos_top10_mode.csv")
    assert "ranking_collapse__liquidation_pressure" in set(modes["frozen_failure_mode"])
    detector_alignment = pd.read_csv(output / "strict_base_oos_detector_alignment.csv")
    assert detector_alignment["detector_probability_coverage"].max() == 1.0
    assert detector_alignment["cross_source_diagnostic_only"].all()
    overlap = pd.read_csv(output / "strict_base_meta_intersection_overall.csv")
    assert overlap.loc[0, "rows"] == 8
    assert overlap.loc[0, "base_top10_rows"] == overlap.loc[0, "meta_top10_rows"]
    assert overlap.loc[0, "mean_abs_score_delta"] > 0.0
    assert manifest["base_meta_intersection"]["meta_score"] == (
        "score_base_residual_ev_rank_train_reference"
    )
    assert manifest["base_meta_intersection"]["overlap_rows"] == 8


def test_strict_oos_sensitivity_allows_absent_detector_without_promoting_it(tmp_path: Path) -> None:
    base, meta = tmp_path / "base.parquet", tmp_path / "meta.parquet"
    taxonomy, output = tmp_path / "taxonomy", tmp_path / "report"
    _base_ledger().to_parquet(base, index=False)
    _meta_ledger().to_parquet(meta, index=False)
    _taxonomy(taxonomy)

    manifest = run(
        base_ledger=base,
        meta_ledger=meta,
        taxonomy=taxonomy,
        detector=tmp_path / "missing_detector.parquet",
        output=output,
    )

    assert manifest["scopes"]["base_oos"]["detector"]["detector_available"] is False
    assert pd.read_csv(output / "strict_base_oos_detector_alignment.csv").empty


def test_reused_base_scope_recomputes_detector_alignment_for_requested_detector(
    tmp_path: Path,
) -> None:
    base, meta = tmp_path / "base.parquet", tmp_path / "meta.parquet"
    taxonomy = tmp_path / "taxonomy"
    detector_v5, detector_v7 = tmp_path / "detector_v5.parquet", tmp_path / "detector_v7.parquet"
    initial, reused = tmp_path / "initial", tmp_path / "reused"
    _base_ledger().to_parquet(base, index=False)
    _meta_ledger().to_parquet(meta, index=False)
    _taxonomy(taxonomy)
    _detector(detector_v5)
    _detector(detector_v7)

    run(
        base_ledger=base,
        meta_ledger=meta,
        taxonomy=taxonomy,
        detector=detector_v5,
        output=initial,
    )
    manifest = run(
        base_ledger=base,
        meta_ledger=meta,
        taxonomy=taxonomy,
        detector=detector_v7,
        output=reused,
        reuse_base_report=initial,
    )

    assert manifest["scopes"]["base_oos"]["detector"]["detector_path"] == str(
        detector_v7.resolve()
    )
    assert manifest["scopes"]["base_meta_oos"]["detector"]["detector_path"] == str(
        detector_v7.resolve()
    )
