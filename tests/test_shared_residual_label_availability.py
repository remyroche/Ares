from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_tp6_m6_regime_residual_features import (
    _ensure_label_available_ts,
    _prior_resolved_mask,
)
from scripts.run_tp6_shared_residual_a0_a3 import materialise_prequential
from scripts.run_tp6_shared_residual_d0_d4 import assert_outer_train_resolved
from scripts.run_tp6_shared_residual_round_e_calibration import _ensure_label_available_ts as e_label_available


def test_signal_close_h12_labels_are_available_at_plus_13h_and_ties_are_not_prior() -> None:
    ts = pd.Series(pd.to_datetime(["2024-01-01 10:00Z", "2024-01-01 11:00Z"], utc=True))
    frame = _ensure_label_available_ts(pd.DataFrame({"__ts__": ts}))
    assert (frame.label_available_ts - frame.__ts__).eq(pd.Timedelta(hours=13)).all()
    # The 11:00 signal resolves at midnight; a midnight map may not use it.
    prior = _prior_resolved_mask(frame, pd.Timestamp("2024-01-02 00:00Z"))
    np.testing.assert_array_equal(prior, np.asarray([True, False]))


def test_future_outcome_mutation_cannot_change_earlier_a_prequential_features() -> None:
    ts = pd.to_datetime(["2024-01-01 00:00Z", "2024-01-02 00:00Z", "2024-01-03 00:00Z"], utc=True)
    base = pd.DataFrame(
        {
            "__ts__": ts,
            "side_name": ["long", "long", "long"],
            "p_adverse": [.2, .2, .2],
            "p_weak": [.3, .3, .3],
            "p_clear": [.5, .5, .5],
            "net_bps": [100., 80., -50.],
            "hard_regime": [0, 0, 0],
            "p_stable": [1., 1., 1.],
            "p_transition": [0., 0., 0.],
            "p_change": [0., 0., 0.],
        }
    )
    changed = base.copy()
    changed.loc[2, "net_bps"] = -100_000.
    before = materialise_prequential(base)
    after = materialise_prequential(changed)
    columns = ["base_expected_bps", "prior_side", "prior_hard", "prior_soft"]
    np.testing.assert_allclose(before.loc[:1, columns], after.loc[:1, columns])


def test_d_outer_fit_rejects_a_label_available_at_the_test_cutoff() -> None:
    cutoff = pd.Timestamp("2024-02-01 00:00Z")
    train = pd.DataFrame({"__ts__": [cutoff - pd.Timedelta(hours=13)], "label_available_ts": [cutoff]})
    test = pd.DataFrame({"__ts__": [cutoff], "label_available_ts": [cutoff + pd.Timedelta(hours=13)]})
    with pytest.raises(ValueError, match="unresolved labels"):
        assert_outer_train_resolved(train, test)


def test_round_e_never_reconstructs_h12_availability_as_plus_12h() -> None:
    ts = pd.Timestamp("2024-02-01 00:00Z")
    out = e_label_available(pd.DataFrame({"__ts__": [ts]}))
    assert out.loc[0, "label_available_ts"] == ts + pd.Timedelta(hours=13)
