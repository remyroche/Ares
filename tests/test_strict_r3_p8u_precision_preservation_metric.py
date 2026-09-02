from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "strict_r3_p8u_precision_preservation_metric.py"
SPEC = importlib.util.spec_from_file_location("p8u_precision_preservation_metric", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _panel(*, reversed_score: bool = False) -> pd.DataFrame:
    # Five calendar weeks exercise the lower-tail weekly aggregation rather
    # than a degenerate single-week statistic.
    timestamps = pd.date_range("2026-01-01", periods=35 * 24, freq="h", tz="UTC")
    rows: list[dict[str, object]] = []
    for stamp in timestamps:
        # One clean winner, two positive but weaker rows, and two losses per
        # timestamp.  Reversing the score preserves candidate identities but
        # destroys precision and utility capture.
        outcome = [250.0, 150.0, 75.0, -50.0, -100.0]
        score = list(range(5, 0, -1))
        if reversed_score:
            score = list(reversed(score))
        for index, (net, value) in enumerate(zip(outcome, score, strict=True)):
            rows.append({
                "candidate_id": f"{stamp.isoformat()}-{index}", "__decision_ts__": stamp,
                "side_name": "long", "score": float(value), "policy_net_bps": net,
                "policy_ordinal_valid": True,
            })
    return pd.DataFrame(rows)


def test_components_are_timestamp_local_and_rank_before_outcome_filtering() -> None:
    panel = _panel()
    # Make the first highest-ranked row unresolved in one timestamp.  The
    # selection count stays one, but its realised outcome coverage becomes 0.
    panel.loc[0, "policy_ordinal_valid"] = False
    components = MODULE.timestamp_components(panel, score_column="score")
    first = components.iloc[0]
    assert first.candidate_rows == 5
    assert first.dtp2_bps_coverage == 0.0
    assert np.isnan(first.dtp2_bps)
    # Later timestamps retain the best candidate at Top-2% (ceil(5*.02)=1).
    assert components.dtp2_bps.iloc[1:].eq(250.0).all()


def test_residual_utility_does_not_reward_utility_already_captured_at_top10() -> None:
    panel = _panel()
    components = MODULE.timestamp_components(panel, score_column="score")
    # With five rows Top-10 is one winner and Top-30 is two.  The preserved
    # residual is therefore the second row's (150-50) utility divided by the
    # utility still left after the first row's (250-50) utility is removed.
    expected = 100.0 / (325.0 - 200.0)
    assert np.allclose(components.residual_utility_recall10_to30, expected)
    # Top-20% is a separate diagnostic.  With five rows the deterministic
    # ``ceil(n * 20%)`` contract selects one row, so it captures only the
    # first row's (250-50) utility share.
    assert np.allclose(components.utility_recall20, 200.0 / 325.0)


def test_perfect_score_beats_reversed_score_under_stable_objective() -> None:
    control = MODULE.timestamp_components(_panel(), score_column="score")
    summary_control, _ = MODULE.stable_score(control, control)
    reversed_components = MODULE.timestamp_components(_panel(reversed_score=True), score_column="score")
    summary_reversed, _ = MODULE.stable_score(reversed_components, control)
    assert summary_control.score_stable > summary_reversed.score_stable
    assert summary_control.week_rows >= 4


def test_control_timestamp_identity_must_match() -> None:
    control = MODULE.timestamp_components(_panel(), score_column="score")
    with np.testing.assert_raises(AssertionError):
        MODULE.stable_score(control.iloc[:-1], control)
