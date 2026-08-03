from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_execution_ev_decomposition_calibration_ablation import MULTITASK_FEATURES
from scripts.run_execution_ev_direct_primary_multitask_timescale_ablation import (
    AUXILIARY_GROUPS,
    build_timescale_features,
    fit_direct_primary_multitask_oof,
    parser,
)


def _frame(rows: int = 120) -> pd.DataFrame:
    decision = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    net = np.linspace(-0.02, 0.025, rows)
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c-{item}" for item in range(rows)],
            "__symbol__": ["BTC/USD:USD"] * rows,
            "side_name": ["long", "short"] * (rows // 2),
            "execution_decision_utc": decision,
            "execution_label_end_utc": decision + pd.Timedelta(hours=12),
            "execution_net_ev_12h": net,
            "clean_favorable_first_exact_policy": net > 0.004,
            "severe_loss_floor": 0.01,
            "oof_fold": np.repeat(np.arange(3), rows // 3),
        }
    )
    for number, column in enumerate(MULTITASK_FEATURES):
        frame[column] = net + number * 0.0001
    return frame


def test_multitask_side_heads_have_exact_direct_fallback_then_prior_oof_fit() -> None:
    frame = _frame()
    score, audit = fit_direct_primary_multitask_oof(
        frame,
        active_auxiliary=AUXILIARY_GROUPS,
        direct_weight=2,
        clip_direct_z=3.0,
        residual_to_frozen_direct=False,
        min_prior_rows=10,
        max_train_rows=100,
        max_iter=3,
        random_state=7,
    )
    frozen = frame["direct_net_ev"].to_numpy(float)
    first = frame["oof_fold"].eq(0).to_numpy()
    np.testing.assert_allclose(score[first], frozen[first])
    assert all(row["status"].startswith("frozen_direct_fallback") for row in audit if row["fold"] == 0)
    assert any(row["status"].startswith("side_head_shared") for row in audit if row["fold"] == 2)
    assert all(row["direct_loss_repetitions"] == 2 for row in audit)


def test_timescale_features_are_causal_and_do_not_use_transition_labels() -> None:
    rows = 6
    decision = pd.date_range("2026-02-01", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c-{item}" for item in range(rows)],
            "__symbol__": ["BTC/USD:USD"] * rows,
            "side_name": ["long"] * rows,
            "execution_decision_utc": decision,
        }
    )
    for horizon in (1, 3, 6, 12):
        frame[f"transition_p_h{horizon}"] = np.linspace(0.1, 0.7, rows)
        frame[f"persistence_p_h{horizon}"] = np.linspace(0.9, 0.3, rows)
    features = build_timescale_features(frame)
    assert "timing_transition_risk_1h3h" in features
    assert "environment_horizon_disagreement_6h12h" in features
    assert "expected_causal_state_age_hours" in features
    assert "raw_state_transition_label" not in features
    # Altering a future persistence forecast may change later age, never the past.
    altered = frame.copy()
    altered.loc[5, "persistence_p_h1"] = 0.0
    rerun = build_timescale_features(altered)
    np.testing.assert_allclose(
        features.loc[:4, "expected_causal_state_age_hours"],
        rerun.loc[:4, "expected_causal_state_age_hours"],
    )


def test_raw_state_velocity_rejects_future_observable_source() -> None:
    frame = _frame(6).loc[:, ["candidate_id", "__symbol__", "side_name", "execution_decision_utc"]].copy()
    for horizon in (1, 3, 6, 12):
        frame[f"transition_p_h{horizon}"] = 0.2
        frame[f"persistence_p_h{horizon}"] = 0.8
    raw = frame.loc[:, ["__symbol__", "execution_decision_utc"]].drop_duplicates().copy()
    raw["raw_state_source_utc_h0"] = raw["execution_decision_utc"] + pd.Timedelta(hours=1)
    raw["mkt_state__example__h0"] = 1.0
    try:
        build_timescale_features(frame, raw)
    except ValueError as error:
        assert "point-in-time" in str(error)
    else:
        raise AssertionError("future raw-state source must fail closed")


def test_parser_accepts_named_confirmation_subset() -> None:
    args = parser().parse_args([
        "--output-dir", "unused",
        "--variants", "direct_only", "full_aux_w2",
        "--multitask-only",
    ])
    assert args.variants == ["direct_only", "full_aux_w2"]
    assert args.multitask_only
