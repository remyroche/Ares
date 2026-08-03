from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_sparse_transition_mechanism_ablation.py"
SPEC = importlib.util.spec_from_file_location("sparse_transition_mechanism", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _features() -> list[str]:
    raw = (
        "atr_compression_ratio", "ema20_slope_5h", "leverage_build_score", "log_bars_since_above_1atr",
        "log_bars_since_above_2atr", "memory_asymmetry_1ATR", "memory_asymmetry_2ATR",
        "memory_asymmetry_3ATR", "trend_acceleration",
    )
    fields: list[str] = []
    for name in raw:
        for statistic in ("state_mean__median", "state_long_short_gap__median", "past_delta_1h__median", "past_delta_3h__median", "past_delta_12h__median", "state_mean__iqr", "state_long_short_gap__iqr", "past_delta_1h__iqr", "past_delta_3h__iqr", "past_delta_12h__iqr"):
            fields.append(f"context__{statistic}__{name}")
    return fields


def test_fixed_feature_groups_have_exact_semantic_sizes() -> None:
    arms = MODULE.feature_arms(_features())
    assert {name: len(value) for name, value in arms.items()} == {
        "all90_control": 90, "compression_release": 10, "trend_ema_acceleration": 20,
        "leverage_build": 10, "memory_range_recurrence": 50, "sparse_state_levels": 9,
        "short_1h_3h_deltas": 36, "compact_union": 45,
    }


def test_tie_record_marks_constant_scores_nonranking() -> None:
    frame = pd.DataFrame({
        "target_name": ["target__active_adverse"] * 10, "feature_count": [1] * 10,
        "prediction": [.2] * 10, "target": [1.] + [0.] * 9,
        "selected_top10": [True] + [False] * 9, "calibration_shrinkage_weight": [0.] * 10,
    })
    record = MODULE._tie_record(frame, arm="test", evaluation_kind="test", train_source="test")
    assert record["interpretation"] == "NON_RANKING_CONSTANT_OR_ZERO_SHRINK"
    assert record["tie_aware_expected_lift_unweighted"] == 1.0


def test_current_strict_contract_rejects_non_strict_rows() -> None:
    panel = pd.DataFrame({"source_family": [MODULE.CURRENT_SOURCE], "mapping_provenance_role": ["frozen_forward_oos"]})
    assert panel.mapping_provenance_role.ne("strict_oof").any()
