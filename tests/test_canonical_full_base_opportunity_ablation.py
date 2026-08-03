import numpy as np
import pandas as pd
import pytest

from scripts import run_canonical_full_base_opportunity_ablation as runner


def _small_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-02-01", periods=120, freq="h", tz="UTC")
    rows = []
    for timestamp_index, timestamp in enumerate(timestamps):
        for side in runner.SIDES:
            rows.append(
                {
                    "candidate_id": f"{timestamp_index:02d}-{side}",
                    "side_name": side,
                    "__symbol__": "BTC",
                    "__ts__": timestamp,
                    "__decision_ts__": timestamp + pd.Timedelta(hours=1),
                    "execution_label_end_utc": timestamp + pd.Timedelta(hours=13),
                    "effective_label_resolution_utc": timestamp
                    + pd.Timedelta(hours=25),
                    "execution_net_ev_12h": timestamp_index / 10_000,
                    "opportunity_gross_above_cost_0bps": timestamp_index > 10,
                    "opportunity_gross_above_cost_25bps": timestamp_index > 25,
                }
            )
    return pd.DataFrame(rows)


def test_exact_feature_arms_and_no_geometry_sensitivity():
    assert runner.arm_features("S0", "long") == ("base_oof_score",)
    score_context = runner.arm_features("S1", "short")
    assert score_context == ("base_oof_score", *runner.SCORE_CONTEXT)

    regime = runner.arm_features("S1+R", "long")
    assert regime == (
        "base_oof_score",
        *runner.SCORE_CONTEXT,
        *runner.REGIME_LEVELS,
        *runner.REGIME_TRANSITIONS,
    )
    assert runner.arm_features("S1+B", "long")[-31:] == runner.BASE_LONG
    assert runner.arm_features("S1+B", "short")[-8:] == runner.BASE_SHORT

    combined = runner.arm_features("S1+R+B", "long")
    sensitivity = runner.arm_features("S1+R+B-no-DAE-GMM", "long")
    assert set(combined).difference(sensitivity) == runner.GEOMETRY_FEATURES
    assert runner.arm_features(
        "S1+R+B", "short"
    ) == runner.arm_features("S1+R+B-no-DAE-GMM", "short")


def test_mapped_and_outcome_fields_are_rejected_as_features():
    for feature in (
        "mapped_expected_gross",
        "causal_score_percentile",
        "execution_net_ev_12h",
        "opportunity_margin_0bps",
        "execution_mfe_return_12h",
        "timing_prediction",
        "wait_action",
    ):
        with pytest.raises(ValueError, match="forbidden model feature"):
            runner.validate_feature_names([feature])


def test_blocked_folds_cover_every_timestamp_once_and_purge_exact_execution_paths():
    frame = _small_frame()
    folds = runner.make_blocked_folds(frame, n_folds=5)
    assignments = np.zeros(len(frame), dtype=np.int8)
    for fold in folds:
        training, validation = runner.fold_masks(frame, fold, purge_hours=12)
        assignments += validation.astype(np.int8)
        validation_start = frame.loc[validation, "__ts__"].min()
        validation_label_end = frame.loc[
            validation, "execution_label_end_utc"
        ].max()
        training_rows = frame.loc[training]
        assert (
            training_rows["execution_label_end_utc"].lt(validation_start)
            | training_rows["__decision_ts__"].gt(validation_label_end)
        ).all()
    assert np.all(assignments == 1)


def test_split_uses_resolution_cutoff_not_nominal_month():
    frame = pd.DataFrame(
        {
            "candidate_id": ["resolved", "late", "april"],
            "side_name": ["long", "long", "short"],
            "__symbol__": ["A", "B", "C"],
            "__ts__": pd.to_datetime(
                [
                    "2025-03-01T00:00:00Z",
                    "2025-03-31T23:00:00Z",
                    "2025-04-01T00:00:00Z",
                ],
                utc=True,
            ),
            "execution_label_end_utc": pd.to_datetime(
                [
                    "2025-03-02T00:00:00Z",
                    "2025-04-02T00:00:00Z",
                    "2025-04-02T01:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    development, april = runner.split_development_april(frame)
    assert development["candidate_id"].tolist() == ["resolved"]
    assert april["candidate_id"].tolist() == ["april"]


def test_crossfit_mapper_never_uses_held_fold(monkeypatch):
    seen = []
    original = runner.ShrunkMapper.fit

    def recording_fit(cls, score, side, net, *, shrinkage_rows):
        seen.append(set(np.asarray(score).tolist()))
        return original(score, side, net, shrinkage_rows=shrinkage_rows)

    monkeypatch.setattr(
        runner.ShrunkMapper,
        "fit",
        classmethod(recording_fit),
    )
    score = np.arange(10, dtype=float)
    side = np.array(["long", "short"] * 5)
    fold = np.array([0] * 5 + [1] * 5)
    net = score / 100.0
    mapped, final = runner.crossfit_expected_net_mapping(
        score, side, fold, net, shrinkage_rows=2
    )
    assert np.isfinite(mapped).all()
    assert seen[0] == set(score[fold != 0])
    assert seen[1] == set(score[fold != 1])
    assert seen[2] == set(score)
    assert final.side_support == {"long": 5, "short": 5}


def test_global_top_k_is_pooled_and_candidate_id_breaks_ties():
    frame = pd.DataFrame(
        {
            "candidate_id": ["z-long", "a-short", "b-long", "c-short"],
            "side_name": ["long", "short", "long", "short"],
        }
    )
    mask = runner.stable_global_top_mask(frame, [1.0, 1.0, 0.5, 0.4], 0.25)
    assert frame.loc[mask, "candidate_id"].tolist() == ["a-short"]


def test_fit_budget_is_bounded_and_declared():
    assert runner.fit_budget() == {
        "fixed_oof_model_fits": 240,
        "additional_hpo_oof_model_fits_max": 160,
        "fixed_april_final_model_fits": 48,
        "selected_hpo_april_final_model_fits_max": 16,
        "maximum_model_fits": 464,
    }


def test_existing_soft_target_contract():
    frame = pd.DataFrame(
        {
            "execution_net_ev_12h": [-0.01, 0.0, 0.01],
            "execution_soft_positive_12h": [
                1.0 / (1.0 + np.exp(1.0)),
                0.5,
                1.0 / (1.0 + np.exp(-1.0)),
            ],
        }
    )
    expected = runner.target_values(frame, "soft")
    assert np.allclose(expected, frame["execution_soft_positive_12h"])
