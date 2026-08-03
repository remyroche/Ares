import numpy as np
import pandas as pd
import joblib

from scripts.run_cross_era_direct_net_transfer_adapter_ablation import (
    CORRECTION_STATE_COLUMNS,
    add_corrected_transition_inputs,
    correction_feature_columns,
    fit_reliability,
    inner_parent_splits,
    severe_calibration_summary,
    score_parent,
    select_weight_profile,
    training_weights,
    weight_cell_diagnostics,
)


def _weights_frame() -> pd.DataFrame:
    rows = []
    for side in ("long", "short"):
        for era, month, count in (
            ("2025_feb_apr", "2025-03", 8),
            ("2025_feb_apr", "2025-04", 16),
            ("2026_may_jul19", "2026-05", 32),
        ):
            for index in range(count):
                ts = pd.Timestamp(f"{month}-01T00:00:00Z") + pd.Timedelta(hours=index)
                rows.append({"candidate_id": f"{side}-{era}-{month}-{index}", "side_name": side, "era": era, "__ts__": ts, "label_resolution_utc": ts + pd.Timedelta(hours=12)})
    return pd.DataFrame(rows)


def test_era_month_weights_equalize_only_train_cells_per_side():
    frame = _weights_frame()
    positions = np.arange(len(frame), dtype=int)
    diagnostics = weight_cell_diagnostics(frame, positions, "era_month_balanced")
    for _, local in diagnostics.groupby("side_name", observed=True):
        assert np.allclose(local["weight_mass"], local["weight_mass"].iloc[0])
    values = training_weights(frame, positions, "era_month_balanced")
    assert np.isclose(values.mean(), 1.0)
    assert np.isfinite(values).all() and (values > 0).all()


def test_inner_parent_splits_purge_unresolved_rows_before_each_validation_start():
    frame = _weights_frame().sort_values("__ts__", kind="stable").reset_index(drop=True)
    positions = np.arange(len(frame), dtype=int)
    splits = inner_parent_splits(frame, positions, blocks=3)
    assert splits
    for train, valid in splits:
        start = frame.iloc[valid]["__ts__"].min()
        assert (frame.iloc[train]["label_resolution_utc"] < start).all()
        assert (frame.iloc[train]["__ts__"] < start).all()


def test_corrected_transition_terms_use_transformed_space_not_legacy_probability_formula():
    frame = pd.DataFrame(
        {
            "regime_transition_entropy_12h": [1.5, -1.0],
            "regime_transition_entropy_48h": [.5, .75],
            "regime_stability_24h": [-.25, 1.75],
            "volatility_of_volatility_48": [2.0, -.5],
        }
    )
    result = add_corrected_transition_inputs(frame)
    np.testing.assert_allclose(result["transition_pressure_z"], [.75, -1.0])
    np.testing.assert_allclose(result["entropy_acceleration_z"], [1.0, -1.75])
    np.testing.assert_allclose(result["entropy_vov_interaction_z"], [1.0, -.375])
    assert "regime_transition_instability" not in CORRECTION_STATE_COLUMNS


def test_correction_feature_contract_rejects_prohibited_geometry_and_requires_corrected_inputs():
    columns = {
        "base_oof_score": [0.], "base_rank_pct_timestamp_side": [.1], "base_score_z_timestamp_side": [.2],
        "q25_net_bps": [1.], "q50_net_bps": [2.], "p_loss_le_100": [.1], "p_loss_le_200": [.05],
    }
    columns.update({name: [0.] for name in CORRECTION_STATE_COLUMNS})
    frame = pd.DataFrame(columns)
    selected = correction_feature_columns(frame)
    assert "candidate_group_size" not in selected
    assert "transition_pressure_z" in selected


def test_weight_selection_ranks_eligible_profiles_by_economics_not_ic():
    records = pd.DataFrame(
        [
            {"weight_profile": "uniform", "min_side_ic": .20, "min_latest_domain_ic": .10, "month_coverage_complete": True, "calibration_no_worse_than_uniform": True, "global_top10_net_ev_bps": 1.0, "worst_month_top10_net_ev_bps": -2.0, "global_top10_cvar05_bps": -10.0},
            {"weight_profile": "era_balanced", "min_side_ic": .05, "min_latest_domain_ic": .01, "month_coverage_complete": True, "calibration_no_worse_than_uniform": True, "global_top10_net_ev_bps": 3.0, "worst_month_top10_net_ev_bps": -1.0, "global_top10_cvar05_bps": -9.0},
            {"weight_profile": "era_month_balanced", "min_side_ic": -.50, "min_latest_domain_ic": -.10, "month_coverage_complete": True, "calibration_no_worse_than_uniform": True, "global_top10_net_ev_bps": 99.0, "worst_month_top10_net_ev_bps": 99.0, "global_top10_cvar05_bps": 99.0},
        ]
    )
    assert select_weight_profile(records) == "era_balanced"


def test_weight_selection_rejects_better_ev_when_severe_calibration_degrades():
    records = pd.DataFrame(
        [
            {"weight_profile": "uniform", "min_side_ic": .05, "min_latest_domain_ic": .01, "month_coverage_complete": True, "calibration_no_worse_than_uniform": True, "global_top10_net_ev_bps": 2.0, "worst_month_top10_net_ev_bps": -2.0, "global_top10_cvar05_bps": -10.0},
            {"weight_profile": "era_balanced", "min_side_ic": .10, "min_latest_domain_ic": .02, "month_coverage_complete": True, "calibration_no_worse_than_uniform": False, "global_top10_net_ev_bps": 20.0, "worst_month_top10_net_ev_bps": 10.0, "global_top10_cvar05_bps": -5.0},
        ]
    )
    assert select_weight_profile(records) == "uniform"


def test_severe_calibration_summary_is_exact_for_constant_probabilities():
    frame = pd.DataFrame(
        {
            "execution_net_ev_12h": [-.02, -.005, .01, -.03],
            "p_loss_le_100": [.5, .5, .5, .5],
            "p_loss_le_200": [.25, .25, .25, .25],
        }
    )
    metrics = severe_calibration_summary(frame)
    assert np.isclose(metrics["mean_severe_brier"], (.25 + .3125) / 2.0)
    assert metrics["mean_severe_ece10"] >= 0.0


def test_scaled_reliability_fit_routes_sample_weights_and_converges():
    rng = np.random.default_rng(7)
    matrix = pd.DataFrame(
        {
            "large_bps": rng.normal(0.0, 500.0, 500),
            "small_probability": rng.uniform(0.0, 1.0, 500),
        }
    )
    target = (matrix["large_bps"].to_numpy() > 0.0).astype(int)
    weights = np.linspace(.5, 1.5, len(matrix))
    model, iterations = fit_reliability(
        matrix,
        target,
        weights,
        seed=11,
    )
    assert iterations < 1_000
    assert model.predict_proba(matrix.iloc[:5]).shape == (5, 2)


def test_plain_parent_bundle_round_trips_without_script_local_dataclass(tmp_path):
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "side_name": ["long", "short"],
            "__symbol__": ["BTC_USDT", "BTC_USDT"],
            "__ts__": pd.to_datetime(
                ["2026-07-20T00:00:00Z", "2026-07-20T00:00:00Z"]
            ),
        }
    )
    matrix = pd.DataFrame({"x": [1.0, 2.0]})
    parent = {
        side: {
            "features": {name: ["x"] for name in ("q25", "q50", "p100", "p200")},
            "medians": {name: pd.Series({"x": 0.0}) for name in ("q25", "q50", "p100", "p200")},
            "models": {
                "q25": 1.0,
                "q50": 2.0,
                "p100": .1,
                "p200": .05,
            },
        }
        for side in ("long", "short")
    }
    path = tmp_path / "plain_parent.joblib"
    joblib.dump(parent, path)
    loaded = joblib.load(path)
    scored = score_parent(frame, matrix, loaded)
    assert scored["q25_net_bps"].tolist() == [1.0, 1.0]
    assert scored["p_loss_le_200"].tolist() == [.05, .05]
