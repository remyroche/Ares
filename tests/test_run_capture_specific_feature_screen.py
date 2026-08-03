from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_capture_specific_feature_screen import (
    feature_family,
    select_capture_features,
)


def test_feature_family_routes_capture_inputs() -> None:
    assert feature_family("capture_candidate__atr_percentile") == "volatility"
    assert feature_family("capture_candidate__funding_rate") == "leverage"
    assert feature_family("capture_candidate__hour_sin") == "time"


def test_selector_keeps_stable_economic_feature_and_rejects_sign_flip() -> None:
    rows = 800
    decision = pd.date_range("2026-05-01", periods=rows, freq="h", tz="UTC")
    half_signal = np.linspace(-1.0, 1.0, rows // 2)
    stable = np.concatenate([half_signal, half_signal])
    unstable = np.concatenate(
        [half_signal, -half_signal]
    )
    capture = stable > 0.0
    frame = pd.DataFrame(
        {
            "execution_decision_utc": decision,
            "execution_net_ev_12h": np.where(capture, 0.02, -0.02),
            "capture_candidate__stable_ret": stable,
            "capture_candidate__unstable_ret": unstable,
        }
    )
    selected, report = select_capture_features(
        frame,
        [
            "capture_candidate__stable_ret",
            "capture_candidate__unstable_ret",
        ],
        max_features=4,
        minimum_coverage=0.99,
        maximum_per_family=4,
        correlation_cap=0.95,
    )
    assert "capture_candidate__stable_ret" in selected
    assert "capture_candidate__unstable_ret" not in selected
    assert report["status"] == "selected"
