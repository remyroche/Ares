import numpy as np
import pandas as pd

from extreme_price_movements.stage_i_experiment_runner import (
    StageIExperimentJob,
    _strict_oof,
    _pooled_global_layer_metrics,
    run_stage_i_sequential_funnel,
)
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
)


def _cfg() -> dict:
    return {
        "base_shared_feature_keys": ["BASE"],
        "base_long_feature_keys": [],
        "base_short_feature_keys": [],
        "meta_shared_feature_keys": ["META"],
        "meta_product_feature_keys": [],
        "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS": [],
        "STAGE_I_REQUIRED_SAME_SIDE_BASE_OOF_HANDOFF_FEATURE_KEYS": list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES),
        "BASE": ["f1", "f2"],
        "META": ["f1", "f2"],
    }


def test_pooled_global_metrics_select_once_before_month_side_attribution() -> None:
    ledger = pd.DataFrame(
        {
            "candidate_key": ["long::a", "short::b", "long::c", "short::d"],
            "side_name": ["long", "short", "long", "short"],
            "layer": "base",
            "signal_close_ts": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z",
                    "2024-02-01T00:00:00Z", "2024-02-02T00:00:00Z",
                ]
            ),
            "score_bps": [100.0, 90.0, 80.0, 70.0],
            "net_bps": [25.0, -100.0, 500.0, 400.0],
        }
    )
    metrics = _pooled_global_layer_metrics(ledger, top_fractions=(0.25,))
    pooled = next(row for row in metrics if row["scope"] == "pooled_global")
    contributions = [row for row in metrics if row["scope"] == "selected_contribution"]
    assert pooled["selected_rows"] == 1
    assert pooled["net_bps_per_trade"] == 25.0
    assert [(row["month"], row["side"]) for row in contributions] == [("2024-01", "long")]


def test_sequential_funnel_derives_same_side_residual_and_reconstructs(tmp_path) -> None:
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"f1": [0.1, 0.2, 0.3], "f2": [0.4, 0.5, 0.6]})
    jobs = []
    for contract in STAGE_I_ACTIVE_CONTRACTS:
        kwargs = {
            "timestamps": ts,
            "label_available_timestamps": ts + pd.Timedelta(hours=13),
            "exact_net_bps": np.array([100.0, -20.0, 40.0]),
            "exact_net_units": "bps",
        }
        target = None
        if contract.layer == "base":
            target = np.array([0, 1, 2], dtype=np.int8)
            kwargs["r3_metric_target"] = np.array([-1.0, 0.0, 1.0])
        jobs.append(StageIExperimentJob(contract, frame, ["a", "b", "c"], kwargs, target))

    seen = {}

    def train(_frame, target, **kwargs):
        seen[kwargs["cfg"]["mda_config"]["stage_i_contract"]["layer"] + kwargs.get("mode", "")] = np.asarray(target)
        if kwargs["cfg"]["mda_config"]["stage_i_contract"]["layer"] == "meta":
            assert list(_frame.columns) == ["f1", "f2", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES]
            return {
                "selected_feature_names": ["f1", "f2", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES],
                "pruning_history": [
                    {"mda_economic_baseline_score_mean": 10.0, "mda_economic_baseline_score_se": 1.0, "n_features_end": 7}
                ],
            }
        return {
            "pruning_history": [
                {"mda_economic_baseline_score_mean": 10.0, "mda_economic_baseline_score_se": 1.0, "n_features_end": 2}
            ]
        }

    def strict_oof(job, _result):
        base = np.array([10.0, 20.0, 30.0]) if job.contract.layer == "base" else np.array([1.0, -2.0, 3.0])
        if job.contract.layer == "meta":
            assert list(job.frame.columns) == [
                "f1", "f2", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
            ]
            np.testing.assert_allclose(job.frame["r3_p_adverse"], [.1, .2, .3])
            np.testing.assert_allclose(job.frame["r3_p_weak"], [.4, .3, .2])
            np.testing.assert_allclose(job.frame["r3_p_clear"], [.5, .5, .5])
            np.testing.assert_allclose(job.frame["r3_opportunity_score"], [.4, .3, .2])
            np.testing.assert_allclose(
                job.frame["prequential_base_expected_net_bps"], [10., 20., 30.]
            )
        payload = {"prediction": base, "provenance": {
            "strict_oof": True, "side": job.contract.side, "units": "bps",
            "score_semantics": (
                "prequential_base_expected_net_bps" if job.contract.layer == "base"
                else "raw_predicted_residual_bps"
            ),
            "folds": [{
                "train_max_label_available_ts": "2023-12-31T23:59:59Z",
                "validation_start_ts": "2024-01-01T00:00:00Z",
            }],
        }}
        if job.contract.layer == "base":
            payload["base_oof_handoff"] = {
                "r3_p_adverse": np.array([.1, .2, .3], dtype=np.float32),
                "r3_p_weak": np.array([.4, .3, .2], dtype=np.float32),
                "r3_p_clear": np.array([.5, .5, .5], dtype=np.float32),
                "r3_opportunity_score": np.array([.4, .3, .2], dtype=np.float32),
                "prequential_base_expected_net_bps": base.astype(np.float32),
            }
        return payload

    result = run_stage_i_sequential_funnel(
        jobs, cfg=_cfg(), report_root=tmp_path, train_candidate=train, strict_oof_generator=strict_oof
    )
    for side in ("long", "short"):
        cell = result["cells"][f"meta__{side}__shared_exact_net_residual"]
        np.testing.assert_allclose(cell["reconstructed_common_bps"], [11.0, 18.0, 33.0])
        assert cell["frozen_meta_feature_contract"] == [
            "f1", "f2", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
        ]
    # Meta target is exactly net minus the frozen same-side base score.
    assert any(np.allclose(value, [90.0, -40.0, 10.0]) for value in seen.values())
    metrics = result["pooled_global_layer_metrics"]
    assert {row["layer"] for row in metrics} == {
        "base", "meta_residual_reconstructed"
    }
    pooled_top10 = [
        row for row in metrics
        if row["scope"] == "pooled_global" and row["top_fraction"] == 0.10
    ]
    assert len(pooled_top10) == 2
    assert all(row["selected_global_rows"] == 1 for row in pooled_top10)
    assert result["causal_21d_admission_metrics"]
    assert not result["causal_21d_admission_candidates"][
        "causal_21d_side_admitted_ge_50bps"
    ].any()


def test_sequential_funnel_rejects_non_strict_predecessor_oof(tmp_path) -> None:
    # One missing job is sufficient to establish the bounded-cell guard.
    with np.testing.assert_raises_regex(ValueError, "exactly the four"):
        run_stage_i_sequential_funnel(
            [], cfg=_cfg(), report_root=tmp_path,
            train_candidate=lambda *_a, **_k: {}, strict_oof_generator=lambda *_a: {},
        )


def test_strict_oof_rejects_truthy_string_provenance_flag() -> None:
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    job = StageIExperimentJob(
        contract=STAGE_I_ACTIVE_CONTRACTS[0],
        frame=pd.DataFrame({"f1": [.1, .2], "f2": [.3, .4]}),
        candidate_ids=["a", "b"],
        candidate_kwargs={
            "timestamps": ts,
            "label_available_timestamps": ts + pd.Timedelta(hours=13),
            "exact_net_bps": np.array([1., -1.]),
            "exact_net_units": "bps",
            "r3_metric_target": np.array([-1., 1.]),
        },
        target=np.array([0, 2], dtype=np.int8),
    )

    def invalid(_job, _result):
        return {
            "prediction": np.array([1., 2.]),
            "provenance": {
                "strict_oof": "true", "side": "long", "units": "bps",
                "score_semantics": "prequential_base_expected_net_bps",
                "folds": [{
                    "train_max_label_available_ts": "2023-12-31T23:00:00Z",
                    "validation_start_ts": "2024-01-01T00:00:00Z",
                }],
            },
        }

    with np.testing.assert_raises_regex(ValueError, "not strict OOF"):
        _strict_oof(invalid, job, {})


def test_strict_oof_rejects_unresolved_fold_boundary(tmp_path) -> None:
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    jobs = []
    for contract in STAGE_I_ACTIVE_CONTRACTS:
        kwargs = {
            "timestamps": ts,
            "label_available_timestamps": ts + pd.Timedelta(hours=13),
            "exact_net_bps": np.array([100.0, -20.0, 40.0]),
            "exact_net_units": "bps",
        }
        target = np.array([0, 1, 2], dtype=np.int8) if contract.layer == "base" else None
        if contract.layer == "base":
            kwargs["r3_metric_target"] = np.array([-1.0, 0.0, 1.0])
        jobs.append(StageIExperimentJob(contract, frame=pd.DataFrame({"f1": [.1, .2, .3], "f2": [.4, .5, .6]}), candidate_ids=["a", "b", "c"], candidate_kwargs=kwargs, target=target))

    def train(*_args, **_kwargs):
        return {"pruning_history": [{"mda_economic_baseline_score_mean": 1.0, "mda_economic_baseline_score_se": .1, "n_features_end": 2}]}

    def bad_oof(job, _result):
        return {"prediction": np.ones(3), "provenance": {
            "strict_oof": True, "side": job.contract.side, "units": "bps",
            "score_semantics": "prequential_base_expected_net_bps" if job.contract.layer == "base" else "raw_predicted_residual_bps",
            "folds": [{"train_max_label_available_ts": "2024-01-01T00:00:00Z", "validation_start_ts": "2024-01-01T00:00:00Z"}],
        }}

    with np.testing.assert_raises_regex(ValueError, "not prior-resolved"):
        run_stage_i_sequential_funnel(jobs, cfg=_cfg(), report_root=tmp_path, train_candidate=train, strict_oof_generator=bad_oof)


def test_strict_oof_rejects_mapped_base_score_before_residual_handoff(tmp_path) -> None:
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"f1": [.1, .2, .3], "f2": [.4, .5, .6]})
    jobs = []
    for contract in STAGE_I_ACTIVE_CONTRACTS:
        kwargs = {
            "timestamps": ts, "label_available_timestamps": ts + pd.Timedelta(hours=13),
            "exact_net_bps": np.array([1.0, 2.0, 3.0]), "exact_net_units": "bps",
        }
        if contract.layer == "base":
            kwargs["r3_metric_target"] = np.array([-1.0, 0.0, 1.0])
        jobs.append(StageIExperimentJob(contract, frame, ["a", "b", "c"], kwargs,
                                        np.array([0, 1, 2]) if contract.layer == "base" else None))

    def oof(job, _result):
        return {"prediction": np.ones(3), "provenance": {
            "strict_oof": True, "side": job.contract.side, "units": "bps",
            "score_semantics": "causal_21_day_mapped_bps",
            "folds": [{"train_max_label_available_ts": "2023-12-31T23:00:00Z", "validation_start_ts": "2024-01-01T00:00:00Z"}],
        }}

    with np.testing.assert_raises_regex(ValueError, "mapped/calibrated"):
        run_stage_i_sequential_funnel(
            jobs, cfg=_cfg(), report_root=tmp_path,
            train_candidate=lambda *_a, **_k: {"pruning_history": [{"mda_economic_baseline_score_mean": 1.0, "mda_economic_baseline_score_se": .1, "n_features_end": 2}]},
            strict_oof_generator=oof,
        )
