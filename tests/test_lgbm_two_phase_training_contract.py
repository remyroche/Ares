from __future__ import annotations

import sys
import json

import numpy as np
import pandas as pd

from extreme_price_movements.lgbm_pipeline import (
    BASE_SINGLE_CYCLE_MDA_SELECTION_CONTRACT,
    LGBM_TWO_PHASE_SELECTION_CONTRACT,
    LGBM_TWO_PHASE_FULL_FIT_ROW_CAP,
    LGBM_HPO_SAMPLE_ROWS,
    LGBM_TWO_PHASE_SELECTION_SAMPLE_ROWS,
    canonical_base_feature_selection_recipe,
    cumulative_positive_mda_keep_count,
    materialize_bme_parquet_sample,
    use_canonical_two_phase_feature_selection,
)
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _fit_predict_lgbm,
    _load_fold_payload,
    _load_fixed_params,
    _load_projected_labels,
    _reuse_complete_fold_payload,
    _smallest_subset_within_fractional_se,
    _suggest_params,
    _write_fold_payload,
    parse_args,
)


def test_canonical_two_phase_decision_is_fail_safe() -> None:
    assert LGBM_TWO_PHASE_SELECTION_CONTRACT
    assert LGBM_TWO_PHASE_SELECTION_SAMPLE_ROWS >= 300
    assert LGBM_TWO_PHASE_FULL_FIT_ROW_CAP == 0
    assert use_canonical_two_phase_feature_selection(
        has_frozen_feature_contract=False
    )
    assert not use_canonical_two_phase_feature_selection(
        has_frozen_feature_contract=True
    )
    assert not use_canonical_two_phase_feature_selection(
        has_frozen_feature_contract=False,
        diagnostic_single_phase=True,
    )


def test_promoted_base_selection_contract_is_process_not_feature_count() -> None:
    recipe = canonical_base_feature_selection_recipe()
    assert recipe["contract"] == BASE_SINGLE_CYCLE_MDA_SELECTION_CONTRACT
    assert recipe["selection_method"] == "direct_topk_economic_permutation_mda"
    assert recipe["explicit_feature_count"] is None
    assert recipe["cumulative_positive_importance_fraction"] == 0.99
    assert recipe["maximum_feature_count"] == 150

    keep_n, status, floor = cumulative_positive_mda_keep_count(
        [0.60, 0.30, 0.09, 0.01, -1.0],
    )
    assert keep_n == 3
    assert status == "auto_mda_cumulative_positive_99pct"
    assert floor > 0.0


def test_base_cli_defaults_to_wide_sample_then_narrow_full_fit(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_materialized_trailing_label_topk_lgbm_hpo.py",
            "--labels-path",
            "labels",
        ],
    )
    args = parse_args()
    assert args.single_phase_wide_feature_selection is False
    assert args.feature_selection_sample_rows == LGBM_TWO_PHASE_SELECTION_SAMPLE_ROWS
    assert args.feature_selection_sample_rows == 300_000
    assert args.hpo_max_train_rows == LGBM_HPO_SAMPLE_ROWS
    assert args.hpo_max_train_rows == 45_000
    assert args.max_train_rows == LGBM_TWO_PHASE_FULL_FIT_ROW_CAP
    assert args.feature_selection_method == "mda"
    assert args.feature_selection_top_n == 0
    assert args.fixed_selected_features_csv is None


def test_base_full_fit_label_reload_projects_frozen_contract(tmp_path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    for month in range(2):
        pd.DataFrame(
            {
                "__ts__": pd.date_range(
                    f"2025-0{month + 1}-01", periods=10, freq="h"
                ),
                "__symbol__": ["X"] * 10,
                "side": np.where(np.arange(10) % 2, -1, 1),
                "__first_touch_target_soft__": np.linspace(0, 1, 10),
                "selected_feature": np.arange(10, dtype=np.float32),
                "discarded_wide_feature": np.arange(10, dtype=np.float32) * 2,
            }
        ).to_parquet(labels / f"part_{month}.parquet", index=False)

    frame, contract = _load_projected_labels(
        labels,
        selected_features=["selected_feature"],
        ae_gmm_input_features=[],
    )

    assert len(frame) == 20
    assert "selected_feature" in frame
    assert "__first_touch_target_soft__" in frame
    assert "discarded_wide_feature" not in frame
    assert contract["mode"] == "narrow_full_population"
    assert contract["loaded_column_count"] < contract["source_column_count"]


def test_bme_materializer_keeps_all_rows_when_below_cap(tmp_path) -> None:
    source = tmp_path / "labels.parquet"
    output = tmp_path / "sample.parquet"
    rows = 90
    pd.DataFrame(
        {
            "__ts__": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "__symbol__": [f"S{i % 3}" for i in range(rows)],
            "side": np.where(np.arange(rows) % 2, -1, 1),
            "value": np.arange(rows),
        }
    ).to_parquet(source, index=False)

    contract = materialize_bme_parquet_sample(
        source,
        output,
        max_rows=300,
        seed=42,
        timestamp_column="__ts__",
        identity_columns=("__symbol__", "side"),
    )

    sample = pd.read_parquet(output)
    assert len(sample) == rows
    assert sample["value"].tolist() == list(range(rows))
    assert contract["sampling_policy"] == "all_eligible_rows_below_bme_cap_v1"


def test_compact_full_fit_payload_omits_wide_train_context(tmp_path) -> None:
    fold = {
        "fold": "2026-04",
        "x_train": pd.DataFrame({"x": [0.1, 0.2]}),
        "x_valid": pd.DataFrame({"x": [0.3]}),
        "ae_gmm_context_valid": pd.DataFrame({"gmm_prob_0": [0.75]}),
        "valid": pd.DataFrame({"__ts__": [pd.Timestamp("2026-04-01")]}),
        "valid_metrics": pd.DataFrame({"ret_net": [0.01]}),
        "train_target": pd.DataFrame(
            {"target_soft": [0.2, 0.8], "target_hard": [0.0, 1.0]}
        ),
        "train_weight": pd.DataFrame({"sample_weight": [1.25, 0.75]}),
        "compact_fixed_training_payload": True,
        "train_rows_uncapped": 5,
        "train_rows_payload": 2,
        "payload_train_sampling": "beginning_middle_end_time_spread",
        "fixed_training_target_mode": "target_soft",
        "fixed_training_weight_arm": "W0_base",
        "train_median_imputation_values": {"x": 0.2},
    }
    slim = _write_fold_payload(fold, tmp_path / "cache")
    assert "train" not in slim.get("payload_paths", {})
    assert "train_metrics" not in slim.get("payload_paths", {})
    restored = _load_fold_payload(slim)
    np.testing.assert_allclose(
        restored["train_target"]["target_soft"], [0.2, 0.8], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        restored["train_weight"]["sample_weight"], [1.25, 0.75], rtol=0, atol=1e-7
    )

    cached = _reuse_complete_fold_payload(
        cache_dir=tmp_path / "cache",
        window={
            "fold": "2026-04",
            "month": "2026-04",
            "valid_start": pd.Timestamp("2026-04-01", tz="UTC"),
            "valid_end": pd.Timestamp("2026-05-01", tz="UTC"),
            "train_rows_estimate": 5,
            "valid_rows_estimate": 1,
            "valid_rows_raw_estimate": 1,
        },
        selected_features=["x"],
        fixed_training_contract={"target_mode": "target_soft", "weight_arm": "W0_base"},
    )
    assert cached is not None
    assert cached["cache_reused"] is True
    assert cached["train_rows"] == 2
    assert cached["train_rows_uncapped"] == 5


def test_hpo_search_uses_l2_loss_and_mild_split_gain() -> None:
    params = _suggest_params(None, np.random.default_rng(7))
    assert params["loss_function"] == "regression"
    assert 0.0 <= params["min_split_gain"] <= 0.01


def test_hpo_search_respects_available_target_modes() -> None:
    params = _suggest_params(
        None,
        np.random.default_rng(7),
        target_modes=("target_soft",),
    )
    assert params["target_mode"] == "target_soft"


def test_fit_passes_loss_and_split_gain_to_lightgbm(monkeypatch) -> None:
    from scripts import run_materialized_trailing_label_topk_lgbm_hpo as runner

    captured: dict[str, object] = {}

    class DummyModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, *_args, **_kwargs):
            return self

        def predict(self, frame):
            return np.full(len(frame), 0.5, dtype=np.float32)

    monkeypatch.setattr(runner, "_LIGHTGBM_AVAILABLE", True)
    monkeypatch.setattr(runner, "LGBMRegressor", DummyModel)
    params = {
        "n_estimators": 10,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "loss_function": "huber",
        "min_split_gain": 0.002,
    }
    _fit_predict_lgbm(
        x_train=pd.DataFrame({"x": [0.0, 1.0]}, dtype=np.float32),
        y_train=pd.Series([0.1, 0.9]),
        w_train=pd.Series([1.0, 1.0]),
        x_valid=pd.DataFrame({"x": [0.5]}, dtype=np.float32),
        params=params,
        seed=42,
    )
    assert captured["objective"] == "huber"
    assert captured["min_split_gain"] == 0.002


def test_fixed_param_loader_drops_stale_hpo_metrics(tmp_path) -> None:
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "n_estimators": 147,
                "learning_rate": 0.023,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 98,
                "subsample": 0.78,
                "colsample_bytree": 0.91,
                "reg_alpha": 0.02,
                "reg_lambda": 0.34,
                "target_mode": "target_soft",
                "weight_arm": "W7_timestamp_balanced",
                "objective": 99.0,
                "rank": 1,
                "mean_top10_mean_first_touch_net": 42.0,
                "trial_number": 135,
            }
        ),
        encoding="utf-8",
    )
    params = _load_fixed_params(path)
    assert "objective" not in params
    assert "rank" not in params
    assert "mean_top10_mean_first_touch_net" not in params
    assert params["loss_function"] == "regression"
    assert params["min_split_gain"] == 0.0
    assert params["_fixed_trial_number"] == 135


def test_fractional_se_rule_selects_smallest_stable_mda_subset() -> None:
    chosen = _smallest_subset_within_fractional_se(
        [
            {"feature_count": 24, "mean_objective": 1.04, "se_objective": 0.01},
            {"feature_count": 48, "mean_objective": 1.09, "se_objective": 0.04},
            {"feature_count": 72, "mean_objective": 1.10, "se_objective": 0.08},
        ],
        se_mult=0.75,
    )
    # The best arm's floor is 1.10 - 0.75*0.08 = 1.04, so the compact arm
    # is retained instead of selecting the numerical maximum mechanically.
    assert chosen["feature_count"] == 24
    assert chosen["best_feature_count"] == 72
    assert chosen["selection_floor"] == 1.04
    assert chosen["selection_se_mult"] == 0.75


def test_cumulative_positive_mda_count_respects_feature_ceiling() -> None:
    keep_n, status, _floor = cumulative_positive_mda_keep_count(
        np.linspace(1.0, 0.01, 300),
        cumulative_fraction=0.99,
        maximum_feature_count=150,
    )
    assert keep_n == 150
    assert status.endswith("_capped_150")
    canonical_base_feature_selection_recipe,
    cumulative_positive_mda_keep_count,
