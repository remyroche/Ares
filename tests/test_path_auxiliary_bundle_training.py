from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.path_auxiliary_bundle_training import (
    HEAD_ROLE_KEYS,
    MEANINGFUL_EVENT_ROLE,
    canonical_role_targets,
    compose_head_oof,
    fit_role_by_side,
    select_bundle_feature_contracts,
)


def _labels(rows: int = 8) -> pd.DataFrame:
    hit = np.resize(np.array([1.0, 0.0, 1.0, 0.0]), rows)
    timing = np.where(hit > 0.5, np.resize([2.0, 8.0], rows), 12.0)
    return pd.DataFrame(
        {
            "__path_auxiliary_target_valid__": np.ones(rows),
            "__meaningful_mfe_reached_12h__": hit,
            "__time_to_first_meaningful_mfe_hours_12h__": timing,
            "__peak_mfe_atr_12h__": np.where(hit > 0.5, 3.0, 0.0),
            "__mae_before_meaningful_mfe_atr_12h__": np.where(hit > 0.5, 0.5, 3.0),
            "__bars_to_adverse_extreme_before_mfe_12h__": np.resize(
                [1.0, 4.0, 2.0, 5.0], rows
            ),
            "__bars_to_confirmed_adverse_trough__": np.resize(
                [4.0, np.nan, 5.0, np.nan], rows
            ),
            "__future_slope_atr_per_hour_12h__": np.resize([1.0, 0.0, 0.5, 0.1], rows),
        }
    )


def _selection_contracts() -> dict[str, dict[str, object]]:
    groups = {
        MEANINGFUL_EVENT_ROLE,
        "peak_conditional_magnitude",
        "timing_hit_by_2h",
        "timing_hit_by_4h",
        "timing_hit_by_8h",
        "mae_if_hit",
        "mae_if_no_hit",
        "legacy_adverse_extreme",
        "confirmed_adverse_trough",
        "slope_diagnostic",
    }
    return {
        group: {
            "selected_features_by_side": {
                "long": ["efficiency_ratio_20"],
                "short": ["prog_eff_24"],
            }
        }
        for group in groups
    }


def test_canonical_roles_share_one_exact_meaningful_event_target() -> None:
    roles = canonical_role_targets(_labels())
    event = roles[MEANINGFUL_EVENT_ROLE]

    assert "peak_mfe_12h_atr.p_hit" not in roles
    assert "mae_before_meaningful_mfe_atr.p_hit" not in roles
    assert np.array_equal(event.target, np.resize([1.0, 0.0, 1.0, 0.0], 8))
    assert event.source_column == "__meaningful_mfe_reached_12h__"


def test_selection_runs_once_per_unique_target_and_conditioning_mask(
    monkeypatch,
) -> None:
    labels = _labels(800)
    calls: list[tuple[str, int, str]] = []

    def fake_select(
        X: pd.DataFrame,
        target: np.ndarray,
        **kwargs,
    ) -> dict[str, object]:
        calls.append((str(kwargs["role_name"]), len(target), str(kwargs["task_kind"])))
        return {
            "selected_features_by_side": {
                "long": ["efficiency_ratio_20"],
                "short": ["prog_eff_24"],
            }
        }

    monkeypatch.setattr(
        "extreme_price_movements.path_auxiliary_bundle_training."
        "select_auxiliary_role_features",
        fake_select,
    )
    timestamps = pd.date_range("2026-04-01", periods=800, freq="h", tz="UTC")
    selections = select_bundle_feature_contracts(
        pd.DataFrame(
            {
                "efficiency_ratio_20": np.ones(800),
                "prog_eff_24": np.ones(800),
            }
        ),
        labels,
        timestamps=timestamps,
        assets=np.repeat(["AAA", "BBB"], 400),
        sides=np.repeat(["long", "short"], 400),
        archetypes=np.repeat("base", 800),
    )

    assert len(calls) == 10
    assert set(selections) == set(_selection_contracts())
    rows_by_group = {group: rows for group, rows, _task in calls}
    assert rows_by_group[MEANINGFUL_EVENT_ROLE] == 800
    assert rows_by_group["peak_conditional_magnitude"] == 400


def test_role_training_is_scattered_back_to_independent_sides(monkeypatch) -> None:
    labels = _labels()
    sides = np.repeat(["long", "short"], 4)
    timestamps = pd.date_range("2026-05-01", periods=8, freq="h", tz="UTC")
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_fit(
        X: pd.DataFrame,
        target: np.ndarray,
        *,
        selected_features,
        **kwargs,
    ) -> dict[str, object]:
        side_feature = tuple(selected_features)
        calls.append((str(kwargs["role_name"]), side_feature))
        value = 0.25 if side_feature == ("efficiency_ratio_20",) else 0.75
        return {
            "oof_predictions": np.full(len(X), value, dtype=np.float32),
            "oof_fold_ids": np.zeros(len(X), dtype=np.int16),
            "oof_prediction_mask": np.ones(len(X), dtype=bool),
        }

    monkeypatch.setattr(
        "extreme_price_movements.path_auxiliary_bundle_training."
        "fit_auxiliary_role_model",
        fake_fit,
    )
    result = fit_role_by_side(
        pd.DataFrame(
            {
                "efficiency_ratio_20": np.ones(8),
                "prog_eff_24": np.ones(8),
            }
        ),
        labels,
        role_name=MEANINGFUL_EVENT_ROLE,
        selection_contracts=_selection_contracts(),
        timestamps=timestamps,
        label_resolved_at=timestamps + pd.Timedelta(hours=13),
        sides=sides,
        selection_hpo_reference_end="2026-05-01T00:00:00Z",
    )

    assert calls == [
        (MEANINGFUL_EVENT_ROLE, ("efficiency_ratio_20",)),
        (MEANINGFUL_EVENT_ROLE, ("prog_eff_24",)),
    ]
    assert np.allclose(result["oof_predictions"][:4], 0.25)
    assert np.allclose(result["oof_predictions"][4:], 0.75)


def _role_result(values: list[float]) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "oof_predictions": array,
        "oof_prediction_mask": np.isfinite(array),
    }


def test_head_composition_uses_common_oof_rows_and_monotone_timing() -> None:
    peak_roles = {
        MEANINGFUL_EVENT_ROLE: _role_result([0.5, 0.2, np.nan]),
        "peak_mfe_12h_atr.conditional_mean": _role_result([4.0, 3.0, 2.0]),
        "peak_mfe_12h_atr.conditional_q80": _role_result([7.0, 5.0, 4.0]),
    }
    peak = compose_head_oof("peak_mfe_12h_atr", peak_roles)
    assert peak["oof_prediction_available"].tolist() == [True, True, False]
    assert np.allclose(peak.loc[:1, "expected_peak_mfe_atr"], [2.0, 0.6])

    timing_roles = {
        MEANINGFUL_EVENT_ROLE: _role_result([0.6, 0.9]),
        "time_to_first_meaningful_mfe.hit_by_2h": _role_result([0.8, 0.1]),
        "time_to_first_meaningful_mfe.hit_by_4h": _role_result([0.2, 0.5]),
        "time_to_first_meaningful_mfe.hit_by_8h": _role_result([0.7, 0.4]),
    }
    timing = compose_head_oof("time_to_first_meaningful_mfe", timing_roles)
    assert np.all(timing["p_hit_by_2h"] <= timing["p_hit_by_4h"])
    assert np.all(timing["p_hit_by_4h"] <= timing["p_hit_by_8h"])
    assert np.all(timing["p_hit_by_8h"] <= timing["p_hit_by_12h"])
    assert set(HEAD_ROLE_KEYS) == {
        "peak_mfe_12h_atr",
        "time_to_first_meaningful_mfe",
        "mae_before_meaningful_mfe_atr",
        "bars_before_price_stops_decreasing",
        "future_slope_atr_per_hour",
    }
