from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_v5_conversion_residual_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("conversion_residual_ablation", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_competing_classes_separate_clean_timeout_and_adverse_paths() -> None:
    frame = pd.DataFrame(
        {
            MODULE.TARGET: [0.01, 0.01, -0.01, -0.01, -0.04],
            "execution_exit_reason": [
                "trailing",
                "timeout",
                "trailing",
                "timeout",
                "full_sl",
            ],
        }
    )
    assert MODULE.competing_class(frame).tolist() == [
        "trailing_positive",
        "timeout_positive",
        "trailing_nonpositive",
        "timeout_nonpositive",
        "full_stop_or_adverse",
    ]


def test_feature_groups_are_staged_and_mae_only_in_optional_arm() -> None:
    roles = {
        "baseline_model_features": list(MODULE.materialized.BASELINE_FEATURES),
        "optional_adverse_risk_ablation_only": list(
            MODULE.materialized.OPTIONAL_RISK_FEATURES
        ),
    }
    groups = MODULE.feature_groups(roles)
    assert set(groups) == {
        "scores",
        "scores_peak_slope",
        "scores_peak_slope_levels",
        "scores_peak_slope_levels_transitions",
        "scores_peak_slope_levels_regimes",
        "all_compact",
        "all_compact_optional_mae",
    }
    mae = set(MODULE.materialized.OPTIONAL_RISK_FEATURES)
    assert not mae.intersection(groups["all_compact"])
    assert mae.issubset(groups["all_compact_optional_mae"])


def test_selection_map_is_causal_and_objective_uses_mapped_scores() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(30)],
            "side_name": ["long"] * 30,
            "__symbol__": ["A"] * 30,
            "__ts__": pd.date_range("2025-03-23", periods=30, freq="h", tz="UTC"),
            MODULE.TIME: pd.date_range("2025-03-23", periods=30, freq="h", tz="UTC"),
            MODULE.END: pd.date_range("2025-03-23 12:00", periods=30, freq="h", tz="UTC"),
            MODULE.TARGET: np.linspace(-0.01, 0.02, 30),
            "execution_gross_ev_12h": np.linspace(0, 0.03, 30),
            "execution_cost_return": [0.01] * 30,
            "raw_score": np.linspace(-1, 1, 30),
            "score_available_utc": pd.date_range("2025-03-23", periods=30, freq="h", tz="UTC"),
            "challenger_score": np.linspace(-1, 1, 30),
            "selection_mapped_score": np.linspace(1, -1, 30),
            "selection_mapped_eligible": [True] * 30,
            "selection_mapping_status": ["mapped"] * 30,
            "selection_fold": np.repeat(
                [fold[0] for fold in MODULE.MARCH_FOLDS],
                10,
            ),
            "fold_train_label_end_max": pd.Timestamp("2025-03-22T00:00:00Z"),
            "fold_validation_start": pd.Timestamp("2025-03-23T00:00:00Z"),
        }
    )
    result = MODULE.selection_metrics(
        frame,
        config="x",
        group="scores",
        architecture="direct_residual",
        feature_count=4,
    )
    values = np.array([row["top10_net_bps"] for row in result["fold_metrics"]])
    expected = values.mean() - 0.5 * values.std(ddof=0) + 0.25 * values.min()
    assert np.isclose(result["stability_objective_bps"], expected)
    assert not np.isclose(
        result["march_oof_global_top10_net_bps"],
        result["march_oof_raw_diagnostic_top10_net_bps"],
    )
    expected_tie = MODULE.mapping.bound(
        frame,
        "selection_mapped_score",
        0.10,
    )["random_tie_expected_net_bps"]
    assert np.isclose(result["march_oof_global_top10_net_bps"], expected_tie)


def test_causal_selection_map_leaves_unsupported_warmup_unmapped() -> None:
    hours = pd.date_range("2025-03-23", periods=96, freq="h", tz="UTC")
    rows = []
    for position, timestamp in enumerate(hours):
        for side_name in ("long", "short"):
            for asset_index in range(20):
                rows.append(
                    {
                        "candidate_id": f"{position}-{side_name}-{asset_index}",
                        "side_name": side_name,
                        "__symbol__": f"A{asset_index}",
                        "__ts__": timestamp,
                        MODULE.TIME: timestamp,
                        MODULE.END: timestamp + pd.Timedelta(hours=12),
                        MODULE.TARGET: ((position + asset_index) % 11 - 5) / 1_000,
                        "execution_gross_ev_12h": ((position + asset_index) % 11) / 1_000,
                        "execution_cost_return": 0.005,
                        "raw_score": float(asset_index),
                        "score_available_utc": timestamp,
                        "challenger_score": float(asset_index + position / 100),
                        "selection_fold": "fold",
                        "fold_train_label_end_max": pd.Timestamp(
                            "2025-03-22T00:00:00Z"
                        ),
                        "fold_validation_start": pd.Timestamp(
                            "2025-03-23T00:00:00Z"
                        ),
                    }
                )
    mapped, audit = MODULE.causal_selection_map(pd.DataFrame(rows))
    assert audit.strict_causal_window_pass.all()
    assert not audit.iloc[0].pooled_support_pass
    assert not mapped.loc[
        mapped[MODULE.TIME].dt.floor("D").eq(pd.Timestamp("2025-03-23T00:00:00Z")),
        "selection_mapped_eligible",
    ].any()
    assert mapped.selection_mapped_eligible.any()
    assert mapped.loc[
        mapped.selection_mapped_eligible, "selection_mapped_score"
    ].notna().all()


def test_calibration_fold_is_never_part_of_selection_objective() -> None:
    assert MODULE.CALIBRATION_FOLD[0] not in MODULE.SELECTION_FOLD_NAMES
    assert len(MODULE.ALL_MARCH_FOLDS) == len(MODULE.MARCH_FOLDS) + 1
    assert MODULE.DEFAULT_OUTPUT.name.endswith("_v4")
