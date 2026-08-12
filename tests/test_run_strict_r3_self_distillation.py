from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_strict_r3_self_distillation.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_distill_runner", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fold_seed_matches_canonical_2024_anchor() -> None:
    assert MODULE._fold_index(pd.Timestamp("2024-01-01", tz="UTC")) == 0
    assert MODULE._fold_index(pd.Timestamp("2025-01-01", tz="UTC")) == 12


def test_rank_ic_is_within_decision_query() -> None:
    frame = pd.DataFrame(
        {
            "__decision_ts__": pd.to_datetime(
                ["2025-01-01"] * 3 + ["2025-01-01 01:00"] * 3,
                utc=True, format="mixed",
            ),
            "r3_class": [0, 1, 2, 0, 1, 2],
            "base_score": [0, 1, 2, 0, 1, 2],
        }
    )
    assert np.isclose(MODULE._rank_ic(frame), 1.0)


def test_base_metrics_use_global_top_fraction_not_per_timestamp() -> None:
    n = 100
    frame = pd.DataFrame(
        {
            "arm": "D0",
            "held_month": "2025-01",
            "candidate_id": [str(index) for index in range(n)],
            "__decision_ts__": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            "r3_class": ([2] * 10) + ([1] * 40) + ([0] * 50),
            "base_score": np.arange(n)[::-1],
            "p_adverse": np.full(n, 0.2),
            "p_weak": np.full(n, 0.3),
            "p_clear": np.full(n, 0.5),
        }
    )
    pooled, _ = MODULE._base_metrics(frame)
    assert np.isclose(pooled.iloc[0]["top30_clear_recall"], 1.0)


def test_positive_refinement_is_one_threshold_dimension() -> None:
    class Args:
        phase = "base_refine_positive"
        positive_top_fractions = [0.6, 0.4, 0.2]
        positive_boost = 1.5

    specs = MODULE._base_specs(Args())
    assert [spec.name for spec in specs] == [
        "D0", "D2_top60_boost1.5", "D2_top40_boost1.5", "D2_top20_boost1.5",
    ]


def test_residual_screen_cap_preserves_complete_queries() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [str(index) for index in range(120)],
            "__decision_ts__": pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC"),
            "side_name": "long",
        }
    )
    capped = MODULE._cap_complete_queries(frame, 40)
    original_query = frame["__decision_ts__"].dt.floor("4h")
    capped_query = capped["__decision_ts__"].dt.floor("4h")
    for query in capped_query.unique():
        assert (capped_query == query).sum() == (original_query == query).sum()


def test_base_override_replaces_all_upstream_outputs_and_requires_history(tmp_path: Path) -> None:
    timestamps = pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True)
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "__decision_ts__": timestamps,
            "prequential_p_adverse": [0.7, 0.6],
            "prequential_p_weak": [0.2, 0.2],
            "prequential_p_clear": [0.1, 0.2],
            "prequential_base_score": [-0.25, -0.1],
            "prequential_base_rank42": [0.1, 0.2],
        }
    )
    path = tmp_path / "base.parquet"
    pd.DataFrame(
        {
            "candidate_id": ["a", "b"], "arm": ["D2", "D2"],
            "p_adverse": [0.1, 0.2], "p_weak": [0.2, 0.2],
            "p_clear": [0.7, 0.6], "base_score": [0.65, 0.5],
            "base_rank42": [0.8, 0.9],
        }
    ).to_parquet(path, index=False)
    output, audit = MODULE._apply_base_prediction_overrides(
        frame, path=path, arm="D2",
        evaluation_start=pd.Timestamp("2025-01-01", tz="UTC"),
        evaluation_end=pd.Timestamp("2025-02-01", tz="UTC"),
    )
    assert output["prequential_base_rank42"].tolist() == [0.8, 0.9]
    assert output["prequential_p_clear"].tolist() == [0.7, 0.6]
    assert audit["anchor_and_residual_recomputed"] is True
    assert audit["base_override_required_coverage"] == 1.0


def test_declared_policy_outcomes_are_authoritative_on_overlap(tmp_path: Path) -> None:
    timestamp = pd.to_datetime(["2026-01-01"], utc=True)
    frame = pd.DataFrame(
        {
            "candidate_id": ["a"], "__decision_ts__": timestamp,
            "policy_path_valid": [False], "policy_net_bps": [0.0],
            "policy_label_available_ts": timestamp + pd.Timedelta(hours=12),
        }
    )
    path = tmp_path / "policy.parquet"
    pd.DataFrame(
        {
            "candidate_id": ["a"], "policy_path_valid": [True],
            "policy_net_bps": [125.0],
            "policy_label_available_ts": timestamp + pd.Timedelta(hours=6),
        }
    ).to_parquet(path, index=False)
    output, audit = MODULE._apply_policy_outcome_overrides(
        frame, path=path,
        evaluation_start=pd.Timestamp("2026-01-01", tz="UTC"),
        evaluation_end=pd.Timestamp("2026-02-01", tz="UTC"),
    )
    assert output.loc[0, "policy_path_valid"]
    assert output.loc[0, "policy_net_bps"] == 125.0
    assert audit["policy_override_held_coverage"] == 1.0
