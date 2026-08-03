from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_first_binary import (
    BinaryFailureDetectorConfig,
    add_causal_transition_deltas,
    build_hourly_binary_failure_targets,
    chronological_binary_failure_oof,
)
from scripts.run_failure_first_binary_ablation import _economics


def _health(periods: int = 60) -> pd.DataFrame:
    start = pd.date_range(
        "2025-01-01", periods=periods, freq="6h", tz="UTC"
    )
    failure = np.zeros(periods, dtype=bool)
    failure[np.arange(4, periods, 5)] = True
    return pd.DataFrame(
        {
            "decision_bin_start_utc": start,
            "bin_available_utc": start + pd.Timedelta(hours=18),
            "evaluation_origin": "historical",
            "model_failure_bin": failure,
        }
    )


def test_binary_targets_use_resolved_health_without_taxonomy() -> None:
    targets = build_hourly_binary_failure_targets(_health(12))
    assert set(targets["target__current_state"].dropna()) == {
        "stable",
        "failure",
    }
    assert targets["target__failure_onset_within_3h"].eq(1.0).any()
    assert targets["target__failure_active_or_within_3h"].eq(1.0).any()
    available = pd.to_datetime(
        targets["binary_failure_label_available_at"], utc=True
    )
    future = pd.to_datetime(
        targets["target__future_label_resolution_utc"], utc=True
    )
    assert available.loc[future.notna()].ge(future.dropna()).all()


def test_transition_deltas_are_exact_lag_and_prefix_invariant() -> None:
    timestamp = pd.date_range(
        "2025-01-01", periods=20, freq="h", tz="UTC"
    )
    frame = pd.DataFrame(
        {
            "execution_decision_utc": timestamp,
            "side_name": "global",
            "evaluation_origin": "a",
            "signal": np.arange(20, dtype=float),
        }
    )
    full, columns = add_causal_transition_deltas(
        frame, signal_columns=["signal"]
    )
    prefix, _ = add_causal_transition_deltas(
        frame.iloc[:12], signal_columns=["signal"]
    )
    pd.testing.assert_frame_equal(
        full.iloc[:12][columns].reset_index(drop=True),
        prefix[columns].reset_index(drop=True),
    )
    assert full.loc[3, "failure_transition_delta_3h__signal"] == 3.0


def test_binary_oof_purges_unresolved_labels() -> None:
    targets = build_hourly_binary_failure_targets(_health())
    panel = targets.copy()
    panel["state__market"] = np.sin(np.arange(len(panel)) / 12.0)
    first_eval = panel["execution_decision_utc"].min() + pd.Timedelta(
        hours=180
    )
    predictions, bundles = chronological_binary_failure_oof(
        panel,
        feature_columns=["state__market"],
        config=BinaryFailureDetectorConfig(
            first_eval_time=first_eval.isoformat(),
            eval_hours=72,
            min_train_rows=100,
            min_positive_rows=3,
            max_iter=8,
            depth=3,
        ),
    )
    assert len(predictions)
    assert len(bundles)
    assert predictions["p_failure_onset_within_3h"].between(0, 1).all()
    assert predictions["p_failure_active_or_within_3h"].between(0, 1).all()
    assert (
        pd.to_datetime(
            predictions["train_label_available_max"], utc=True
        )
        < pd.to_datetime(predictions["train_end_exclusive"], utc=True)
    ).all()


def test_binary_economics_accepts_explicit_forward_provenance_flag() -> None:
    timestamp = pd.Timestamp("2026-07-12", tz="UTC")
    ledger = pd.DataFrame(
        {
            "candidate_id": [f"c-{index}" for index in range(10)],
            "execution_decision_utc": timestamp,
            "evaluation_origin": "forward",
            "causal_recent_side_isotonic_ev": np.linspace(0.0, 0.1, 10),
            "execution_net_ev_12h": np.linspace(-0.02, 0.03, 10),
            "causal_recent_side_isotonic_ev__is_oof": False,
            "causal_recent_side_isotonic_ev__is_forward_oos": True,
        }
    )
    predictions = pd.DataFrame(
        {
            "execution_decision_utc": [timestamp],
            "evaluation_origin": ["forward"],
            "p_failure_onset_within_3h": [0.2],
            "p_failure_active_or_within_3h": [0.3],
        }
    )
    covered, report = _economics(
        ledger,
        predictions,
        eligibility_flag=(
            "causal_recent_side_isotonic_ev__is_forward_oos"
        ),
    )
    assert len(covered) == 10
    assert report["aggregate"]["mapped_score"]["eligible_rows"] == 10
