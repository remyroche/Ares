import pandas as pd

from scripts.historical_inference_parity import (
    _feature_columns_for_state,
    _feature_vector_hash_report,
    _reference_feature_run_id,
    _sample_policy_candidate_rows,
    _summary,
)


class _DummyModel:
    def __init__(self, selected_features, input_feature_names=None):
        self.selected_features = selected_features
        if input_feature_names is not None:
            self.input_feature_names = input_feature_names


def test_historical_replay_uses_decision_feature_scope():
    state = {
        "bundle": {
            "alpha_models": {
                "long": {
                    "demo_head": {
                        "feat_cols": ["selected_alpha", "unused_union_alpha"],
                        "model": _DummyModel(["selected_alpha"]),
                    }
                }
            },
            "meta_models": {
                "long_demo_head": _DummyModel(["selected_meta"]),
            },
        }
    }

    keys = _feature_columns_for_state(state, "long_demo_head")

    assert "selected_alpha" in keys
    assert "selected_meta" in keys
    assert "unused_union_alpha" not in keys


def test_policy_candidate_sample_source_filters_strategy_and_keeps_recent_tail(tmp_path):
    run_id = "run_a"
    path = (
        tmp_path
        / "artifacts"
        / run_id
        / "simple_policy_optimiser"
        / "simple_policy_candidates.parquet"
    )
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-05-24T23:00:00Z",
                    "2026-05-25T00:00:00Z",
                    "2026-05-25T01:00:00Z",
                    "2026-05-25T02:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAA/USD:USD", "BBB/USD:USD", "CCC/USD:USD", "DDD/USD:USD"],
            "strategy_id": ["short_loc", "short_loc", "short_dist", "short_loc"],
        }
    ).to_parquet(path, index=False)

    rows = _sample_policy_candidate_rows(
        tmp_path,
        run_id,
        "short_loc",
        sample_rows=1,
        min_timestamp="2026-05-25T00:00:00Z",
    )

    assert rows["symbol"].tolist() == ["DDD/USD:USD"]


def test_feature_vector_hash_report_detects_exact_and_tolerance_mismatches():
    ts = pd.Timestamp("2026-05-25T00:00:00Z")
    samples = pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"],
            "strategy_id": ["short_loc", "short_loc"],
        }
    )
    fresh_feats = {
        "f1": pd.DataFrame(
            {"AAA/USD:USD": [1.0], "BBB/USD:USD": [2.5]},
            index=pd.DatetimeIndex([ts]),
        ),
        "f2": pd.DataFrame(
            {"AAA/USD:USD": [3.0], "BBB/USD:USD": [4.0]},
            index=pd.DatetimeIndex([ts]),
        ),
    }
    reference_rows = {
        ("AAA/USD:USD", ts): pd.Series({"f1": 1.0, "f2": 3.0}),
        ("BBB/USD:USD", ts): pd.Series({"f1": 2.0, "f2": 4.0}),
    }

    report = _feature_vector_hash_report(
        samples,
        fresh_feats,
        reference_rows,
        {"f1", "f2"},
        tolerance=1e-9,
    )

    assert bool(report.loc[0, "parity_ok"]) is True
    assert bool(report.loc[0, "exact_hash_equal"]) is True
    assert bool(report.loc[1, "parity_ok"]) is False
    assert report.loc[1, "mismatch_count_gt_tolerance"] == 1
    assert report.loc[1, "worst_feature"] == "f1"


def test_reference_feature_run_id_prefers_parity_contract_source():
    feature_cfg = {
        "runtime_cfg": {
            "training_live_parity_contract": {
                "feature_source": {"run_id": "features_20260605"},
            }
        }
    }

    assert (
        _reference_feature_run_id(feature_cfg, active_run_id="active_20260612")
        == "features_20260605"
    )
    assert (
        _reference_feature_run_id(
            feature_cfg,
            active_run_id="active_20260612",
            override_run_id="manual_source",
        )
        == "manual_source"
    )


def test_summary_excludes_both_missing_from_common_feature_rows():
    features = pd.DataFrame(
        {
            "inference_missing": [False, False],
            "training_missing": [False, False],
            "both_missing": [False, True],
            "abs_diff": [0.0, float("nan")],
        }
    )

    summary = _summary(features, pd.DataFrame())

    assert summary["feature_common_rows"] == 1
    assert summary["feature_mismatches_gt_1e_6"] == 0
