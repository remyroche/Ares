from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.causal_market_regime_assessment import (
    RegimeAssessmentError,
    assess_regime_candidate_timeline,
    regime_feature_bundle,
    select_regime_parameter_recommendation,
)
from scripts.assess_causal_market_regime_parameters import proxy_windows


def _timeline() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    origin = pd.Timestamp("2024-01-01T00:00:00Z")
    for candidate, stable in (("primary__k3__rho0.35", True), ("primary__k6__rho0.80", False)):
        for window, offset in (("proxy_01", 0), ("proxy_02", 72)):
            for seed in (11, 17):
                for hour in range(12):
                    ts = origin + pd.Timedelta(hours=offset + hour)
                    if stable:
                        posterior = (0.85, 0.10, 0.05)
                        age = float(hour)
                        phase = (1.0, 0.0, 0.0, 0.0)
                    else:
                        posterior = tuple(float(number == (hour % 3)) for number in range(3))
                        age = 0.0
                        phase = (0.20, 0.35, 0.35, 0.10)
                    rows.append({
                        "source_utc": ts,
                        "candidate_id": candidate,
                        "assessment_fold_id": f"{window}__seed{seed}",
                        "assessment_window_id": window,
                        "assessment_seed": seed,
                        "regime_train_end_utc": ts - pd.Timedelta(hours=1),
                        "system": "primary",
                        "candidate_k": 3 if stable else 6,
                        "candidate_stickiness": 0.35 if stable else 0.80,
                        "centroid_min_separation": 1.0 if stable else 0.1,
                        "centroid_mean_separation": 2.0 if stable else 0.2,
                        "market_regime__state_p_0": posterior[0],
                        "market_regime__state_p_1": posterior[1],
                        "market_regime__state_p_2": posterior[2],
                        "market_regime__input_coverage": 1.0,
                        "market_regime__state_age_hours": age,
                        "market_regime__state_switch_probability": 0.01 if stable else 0.8,
                        "market_regime__entropy": 0.5 if stable else 0.0,
                        "market_regime__top2_margin": 0.75 if stable else 1.0,
                        "market_regime__ood_distance_percentile": 0.2,
                        "market_regime__phase_p_stable": phase[0],
                        "market_regime__phase_p_onset": phase[1],
                        "market_regime__phase_p_active": phase[2],
                        "market_regime__phase_p_settling": phase[3],
                    })
    return pd.DataFrame(rows)


def test_assessment_is_label_free_and_prefers_supported_portable_state_shape() -> None:
    result = assess_regime_candidate_timeline(_timeline(), prefix="market_regime")
    assert len(result.fold_diagnostics) == 8
    assert set(result.portability_diagnostics.candidate_id) == {
        "primary__k3__rho0.35", "primary__k6__rho0.80"
    }
    stable = result.candidate_summary.iloc[0]
    assert stable.candidate_id == "primary__k3__rho0.35"
    assert bool(stable.parameter_gate_passed)
    assert stable.posterior_seed_stability == pytest.approx(1.0)
    assert result.candidate_summary.loc[result.candidate_summary.candidate_id.eq("primary__k6__rho0.80"), "persistence_passed"].iloc[0] == False
    recommendation = select_regime_parameter_recommendation(result.candidate_summary)
    assert recommendation["recommended_candidate_id"] == "primary__k3__rho0.35"


def test_assessment_rejects_noncausal_and_outcome_contaminated_timelines() -> None:
    noncausal = _timeline()
    noncausal.loc[noncausal.index[0], "regime_train_end_utc"] = noncausal.loc[noncausal.index[0], "source_utc"]
    with pytest.raises(RegimeAssessmentError, match="strictly prior"):
        assess_regime_candidate_timeline(noncausal, prefix="market_regime")
    contaminated = _timeline().assign(exact_net_bps=1.0)
    with pytest.raises(RegimeAssessmentError, match="label-free"):
        assess_regime_candidate_timeline(contaminated, prefix="market_regime")


def test_assessment_handles_all_null_padded_states_from_mixed_k_screen() -> None:
    frame = _timeline()
    wide = frame.candidate_id.eq("primary__k6__rho0.80")
    for state in range(3, 6):
        frame[f"market_regime__state_p_{state}"] = np.nan
    frame.loc[wide, [f"market_regime__state_p_{state}" for state in range(6)]] = (
        0.45, 0.20, 0.10, 0.10, 0.10, 0.05
    )
    # K=3 rows retain all-null K=4..6 columns after concatenation.  Those
    # padding columns are not part of their local posterior simplex.
    result = assess_regime_candidate_timeline(frame, prefix="market_regime")
    assert len(result.fold_diagnostics) == 8


def test_per_view_bundle_and_proxy_windows_keep_complete_geometry_and_time_order() -> None:
    frame = _timeline()
    bundle = regime_feature_bundle(frame, prefix="market_regime")
    assert bundle[:3] == (
        "market_regime__state_p_0", "market_regime__state_p_1", "market_regime__state_p_2",
    )
    assert bundle[-4:] == (
        "market_regime__entropy", "market_regime__top2_margin",
        "market_regime__state_age_hours", "market_regime__state_switch_probability",
    )
    timestamp = pd.date_range("2023-01-01", periods=90, freq="h", tz="UTC")
    windows = proxy_windows(pd.Series(timestamp), evaluation_start="2023-01-02", evaluation_end="2023-01-04", folds=3)
    assert len(windows) == 3
    assert all(left < right for _name, left, right in windows)
    assert all(windows[index][2] == windows[index + 1][1] for index in range(2))
