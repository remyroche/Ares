import numpy as np
import pandas as pd

from extreme_price_movements.strict_r3_bcf_mc1_mapper import (
    BCFMC1D2Bundle,
    FEATURES,
)


class _ZeroModel:
    def predict(self, frame):
        return np.zeros(len(frame), dtype=float)


def _controller() -> BCFMC1D2Bundle:
    return BCFMC1D2Bundle(
        root=None,  # type: ignore[arg-type]
        manifest={"fit_cutoff": "2026-08-01T00:00:00Z", "side": "long", "bundle_id": "test"},
        payload={
            "structural_curve_bps": np.zeros(10, dtype=float),
            "model": _ZeroModel(),
            "features_ordered": FEATURES,
        },
    )


def _current(ts: str) -> pd.DataFrame:
    row = {"candidate_id": "x", "__decision_ts__": pd.Timestamp(ts), "side_name": "long"}
    row.update({field: 0.5 for field in FEATURES})
    return pd.DataFrame([row])


def test_bcf_mc1_fails_closed_when_recent_same_bundle_replay_is_empty():
    controller = _controller()
    history = pd.DataFrame({
        "candidate_id": ["old"],
        "__decision_ts__": [pd.Timestamp("2026-07-01T00:00:00Z")],
        "side_name": ["long"],
        "policy_label_available_ts": [pd.Timestamp("2026-07-01T12:00:00Z")],
        "policy_path_valid": [True],
        "policy_net_bps": [80.0],
        "final_score": [0.7],
    })
    result = controller.score(
        _current("2026-08-22T00:00:00Z"),
        resolved_history=history,
        decision_ts="2026-08-22T00:00:00Z",
    )
    assert not result.loc[0, "bcf_mc1_available"]
    assert np.isnan(result.loc[0, "bcf_mc1_expected_net_bps"])
    assert np.isnan(result.loc[0, "bcf_mc1_recent_global_shift_bps"])
    assert result.loc[0, "bcf_mc1_recent_support_rows"] == 0
    assert result.loc[0, "bcf_mc1_recent_support_days"] == 0
    assert result.loc[0, "bcf_mc1_shift_source"] == "unavailable_no_recent_same_bundle_replay_support"


def test_bcf_mc1_uses_strictly_prior_recent_history_when_available():
    controller = _controller()
    history = pd.DataFrame({
        "candidate_id": ["prior", "same_time"],
        "__decision_ts__": [pd.Timestamp("2026-08-21T00:00:00Z"), pd.Timestamp("2026-08-22T00:00:00Z")],
        "side_name": ["long", "long"],
        "policy_label_available_ts": [pd.Timestamp("2026-08-21T12:00:00Z"), pd.Timestamp("2026-08-22T00:00:00Z")],
        "policy_path_valid": [True, True],
        "policy_net_bps": [80.0, 500.0],
        "final_score": [0.7, 0.7],
    })
    result = controller.score(
        _current("2026-08-22T00:00:00Z"),
        resolved_history=history,
        decision_ts="2026-08-22T00:00:00Z",
    )
    assert result.loc[0, "bcf_mc1_available"]
    assert result.loc[0, "bcf_mc1_expected_net_bps"] == 80.0
    assert result.loc[0, "bcf_mc1_recent_global_shift_bps"] == 80.0
    assert result.loc[0, "bcf_mc1_recent_support_rows"] == 1
    assert result.loc[0, "bcf_mc1_shift_source"] == "robust_21d_residual_shift"
