from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_febapr2025_historical_six_class_catboost import _convergence, _geometry_labels, _write_parquet
from scripts.run_febapr2025_historical_competing_risk_catboost import _pit_features


def test_geometry_thresholds_are_discriminating_for_affected_sources() -> None:
    frame = pd.DataFrame({
        "class_label": ["mfe_reversal_or_timeout"] * 4,
        "path_shape_archetype": [
            "early_mfe_full_reversal", "noisy_timeout_usable_mfe",
            "early_mfe_full_reversal", "slow_grinder",
        ],
        "path_arch_peak_mfe_r": [1.4, 1.8, 3.1, 0.1],
    })
    assert _geometry_labels(frame, 1.5).tolist() == [
        "dead_timeout", "mfe_reversal_or_timeout",
        "mfe_reversal_or_timeout", "mfe_reversal_or_timeout",
    ]
    assert _geometry_labels(frame, 2.0).tolist() == [
        "dead_timeout", "dead_timeout", "mfe_reversal_or_timeout",
        "mfe_reversal_or_timeout",
    ]
    assert _geometry_labels(frame, 3.0).tolist() == [
        "dead_timeout", "dead_timeout", "mfe_reversal_or_timeout",
        "mfe_reversal_or_timeout",
    ]


def test_atomic_parquet_writer_leaves_only_complete_file(tmp_path: Path) -> None:
    target = tmp_path / "oof_2025-03.parquet"
    expected = pd.DataFrame({"candidate_id": [1, 2], "oof_month": ["2025-03", "2025-03"]})
    _write_parquet(target, expected)
    assert target.exists()
    assert not target.with_suffix(".parquet.partial").exists()
    pd.testing.assert_frame_equal(pd.read_parquet(target), expected)


def test_hpo_convergence_rejects_cap_hitting_winner() -> None:
    trials = [
        {"trial": 0, "best_iteration": 127, "validation_multiclass_logloss": 1.0},
        {"trial": 1, "best_iteration": 118, "validation_multiclass_logloss": 1.1},
    ]
    report = _convergence(trials, 128)
    assert report["accepted"]
    assert report["raw_winner"]["trial"] == 0
    assert report["eligible_trials"] == [1]
    assert _convergence(trials[:1], 128)["accepted"] is False


def test_competing_risk_excludes_stored_triple_barrier_outcomes_from_features() -> None:
    frame = pd.DataFrame({
        "candidate_id": [1], "side_name": ["long"], "__symbol__": ["BTC_USD"],
        "__ts__": [pd.Timestamp("2025-02-01", tz="UTC")], "risk_class": ["timeout"],
        "__soft_tb_upper_hit_12h__": [1], "__soft_tb_order_ambiguous__": [0],
        "observable_market_feature": [0.5],
    })
    assert _pit_features(frame) == ["observable_market_feature"]
