from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

import scripts.run_meta_v9_ev_mapped_side_residual_ablation as ablation
from extreme_price_movements.supervised_market_state_calibration import (
    expected_ev_rank,
    fit_hierarchical_ev_calibrator,
    predict_hierarchical_ev,
)
from scripts.verify_meta_v9_oos_fold_prediction_parity import (
    verify_meta_v9_oos_fold_prediction_parity,
)


def _calibrator(frame: pd.DataFrame, score: np.ndarray):
    return fit_hierarchical_ev_calibrator(
        frame,
        score,
        frame["ev_after_1pct"].to_numpy(dtype=np.float32),
        shrink_rows=4.0,
        min_local_rows=4,
        local_weight_cap=0.65,
        tail_weight_top10=5.0,
        tail_weight_top20=2.0,
        tail_weight_by_score_quantile=True,
        rank_blend=1.0,
    )


def _write_replayable_fold(tmp_path) -> pd.DataFrame:
    train = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-03-01", periods=120, freq="h", tz="UTC"),
            "__label_path_end_ts__": pd.date_range(
                "2026-03-02", periods=120, freq="h", tz="UTC"
            ),
            "__symbol__": [f"S{index % 5}" for index in range(120)],
            "side_name": np.where(np.arange(120) % 2, "long", "short"),
            "archetype_policy_key": np.where(np.arange(120) % 3, "a", "b"),
            "score_base": np.linspace(0.05, 0.95, 120, dtype=np.float32),
            "feature": np.linspace(-1.0, 1.0, 120, dtype=np.float32),
            "ev_after_1pct": np.sin(np.arange(120) / 9.0).astype(np.float32) / 20.0,
        }
    )
    baseline = _calibrator(train, train["score_base"].to_numpy(dtype=np.float32))
    residual_target = train["ev_after_1pct"].to_numpy(
        dtype=np.float32
    ) - predict_hierarchical_ev(
        baseline, train, train["score_base"].to_numpy(dtype=np.float32)
    )
    model = lgb.train(
        {
            "objective": "regression_l2",
            "seed": 7,
            "verbosity": -1,
            "min_data_in_leaf": 5,
        },
        lgb.Dataset(train[["score_base", "feature"]], label=residual_target),
        num_boost_round=4,
    )
    residual_train = np.zeros(len(train), dtype=np.float32)
    long_mask = train["side_name"].eq("long").to_numpy()
    residual_train[long_mask] = model.predict(
        train.loc[long_mask, ["score_base", "feature"]]
    ).astype(np.float32)
    alpha = {"long": 0.5, "short": 0.0}
    corrected_train = (
        predict_hierarchical_ev(
            baseline, train, train["score_base"].to_numpy(dtype=np.float32)
        )
        + np.asarray([alpha[side] for side in train["side_name"]], dtype=np.float32)
        * residual_train
    )
    corrected = _calibrator(train, corrected_train)
    test = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-04-01", periods=4, freq="h", tz="UTC"),
            "__label_path_end_ts__": pd.date_range(
                "2026-04-02", periods=4, freq="h", tz="UTC"
            ),
            "__symbol__": ["BTC", "ETH", "SOL", "XRP"],
            "side_name": ["long", "short", "long", "short"],
            "archetype_policy_key": ["a", "a", "b", "b"],
            "score_base": np.asarray([0.2, 0.45, 0.7, 0.9], dtype=np.float32),
            "feature": np.asarray([-0.5, -0.1, 0.3, 0.9], dtype=np.float32),
        }
    )
    raw = test["score_base"].to_numpy(dtype=np.float32)
    baseline_ev = predict_hierarchical_ev(baseline, test, raw)
    residual = np.zeros(len(test), dtype=np.float32)
    long_mask = test["side_name"].eq("long").to_numpy()
    residual[long_mask] = model.predict(
        test.loc[long_mask, ["score_base", "feature"]]
    ).astype(np.float32)
    alpha_values = np.asarray(
        [alpha[side] for side in test["side_name"]], dtype=np.float32
    )
    corrected_ev = baseline_ev + alpha_values * residual
    hierarchical_ev = predict_hierarchical_ev(corrected, test, corrected_ev)
    test["score_base_ev_mapped"] = baseline_ev
    test["score_base_ev_residual_expert"] = corrected_ev
    test["score_base_ev_residual_expert_hier_mapped"] = hierarchical_ev
    test["meta_residual_expert_delta_ev"] = alpha_values * residual
    test["score_base_ev_rank_train_reference"] = expected_ev_rank(
        baseline, baseline_ev, raw
    )
    test["score_base_residual_ev_rank_train_reference"] = expected_ev_rank(
        corrected, hierarchical_ev, corrected_ev
    )
    ablation._persist_oos_fold_bundle(
        out_dir=tmp_path,
        fold_id="2026-04-01_2026-05-01",
        oos_fit_mode="expanding_monthly",
        backbone_score="base",
        backbone_score_col="score_base",
        train=train,
        test=test,
        baseline_ev_map=baseline,
        residual_models={"long": model},
        corrected_ev_map=corrected,
        alpha_by_side=alpha,
        features_by_side={"long": ["score_base", "feature"], "short": []},
        params_by_side={"long": {}, "short": {}},
    )
    test.to_parquet(tmp_path / "oos_predictions.parquet", index=False)
    return test


def test_verifier_replays_persisted_fold_without_drift(tmp_path) -> None:
    _write_replayable_fold(tmp_path)

    report = verify_meta_v9_oos_fold_prediction_parity(report_dir=tmp_path)

    assert report["pass"] is True
    assert report["overall"]["rows"] == 4
    assert report["overall"]["missing_or_invalid_rows"] == 0
    assert {row["side_name"] for row in report["side_reports"]} == {"long", "short"}
    assert report["overall"]["drift"]["hierarchical_ev"]["max_abs_drift"] <= 1e-6


def test_verifier_reports_score_drift_by_fold_and_side(tmp_path) -> None:
    stored = _write_replayable_fold(tmp_path)
    stored.loc[0, "score_base_ev_residual_expert_hier_mapped"] += 0.01
    stored.to_parquet(tmp_path / "oos_predictions.parquet", index=False)

    report = verify_meta_v9_oos_fold_prediction_parity(report_dir=tmp_path)

    assert report["pass"] is False
    assert report["fold_reports"][0]["missing_or_invalid_rows"] == 0
    assert report["fold_reports"][0]["drift_rows"] == 1
    assert (
        report["fold_reports"][0]["drift"]["hierarchical_ev"]["max_abs_drift"] > 0.009
    )
    assert (
        next(row for row in report["side_reports"] if row["side_name"] == "long")[
            "pass"
        ]
        is False
    )


def test_verifier_fails_closed_when_exact_model_input_is_absent(tmp_path) -> None:
    stored = _write_replayable_fold(tmp_path)
    stored.drop(columns=["feature"]).to_parquet(
        tmp_path / "oos_predictions.parquet", index=False
    )

    with pytest.raises(ValueError, match="cannot reconstruct exact model inputs"):
        verify_meta_v9_oos_fold_prediction_parity(report_dir=tmp_path)
