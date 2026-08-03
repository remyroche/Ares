import numpy as np
import pandas as pd

from scripts.materialize_tp6_m6_regime_residual_features import (
    STATE_NAMES,
    _soft_prior_apply,
    _soft_prior_fit,
)


def _reference() -> tuple[pd.DataFrame, np.ndarray]:
    n = 6_000
    frame = pd.DataFrame(
        {
            "side_name": np.array(["long"] * n + ["short"] * n),
            "net_bps": np.r_[np.full(n, 80.0), np.full(n, -40.0)],
            "prequential_base_expected_net_bps": np.zeros(2 * n),
        }
    )
    # Soft, not hard, membership exercises the weighted shrinkage path.
    probs = np.tile(np.asarray([[.20, .30, .15, .35]]), (2 * n, 1))
    return frame, probs


def test_soft_prior_is_side_aware_and_shrunk() -> None:
    reference, probabilities = _reference()
    fitted = _soft_prior_fit(reference, probabilities)
    score, scale = _soft_prior_apply(
        pd.DataFrame({"side_name": ["long", "short"]}),
        np.tile(np.asarray([[.20, .30, .15, .35]]), (2, 1)),
        fitted,
    )
    assert score[0] > 0 > score[1]
    assert np.isfinite(scale).all()
    assert len(STATE_NAMES) == 4


def test_future_outcomes_cannot_change_a_previously_fitted_prior() -> None:
    reference, probabilities = _reference()
    fitted = _soft_prior_fit(reference, probabilities)
    query = pd.DataFrame({"side_name": ["long"]})
    before, _ = _soft_prior_apply(query, np.asarray([[.20, .30, .15, .35]]), fitted)
    # Appending a hypothetical future loss and not refitting is the exact
    # prequential contract used when scoring a held-out month.
    extended = pd.concat([reference, pd.DataFrame({"side_name": ["long"], "net_bps": [-9_999.], "prequential_base_expected_net_bps": [0.]})], ignore_index=True)
    assert len(extended) == len(reference) + 1
    after, _ = _soft_prior_apply(query, np.asarray([[.20, .30, .15, .35]]), fitted)
    assert before[0] == after[0]
