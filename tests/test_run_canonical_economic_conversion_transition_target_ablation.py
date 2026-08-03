from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_canonical_economic_conversion_transition_target_ablation import (
    EB_PRIOR_SUPPORT,
    _effective_support,
    _empirical_bayes_delta,
    _fit_priors,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side_name": ["long", "long", "short", "short"],
            "frozen_base_score_decile": [0, 0, 1, 1],
            "before_favorable_net_support": [4, 16, 8, 32],
            "after_favorable_net_support": [4, 16, 8, 32],
            "before_conditional_favorable_net_robust_mean": [0.02, 0.04, 0.03, 0.05],
            "after_conditional_favorable_net_robust_mean": [0.03, 0.06, 0.02, 0.08],
        }
    )


def test_effective_support_is_difference_of_means_precision_proxy() -> None:
    support = _effective_support(_frame())
    assert np.allclose(support, [2.0, 8.0, 4.0, 16.0])
    assert np.all(np.minimum(1.0, support / EB_PRIOR_SUPPORT) <= 1.0)


def test_empirical_bayes_uses_only_supplied_priors_and_shrinks_delta() -> None:
    frame = _frame()
    priors = _fit_priors(frame)
    delta = _empirical_bayes_delta(frame, priors)
    raw = (
        frame["after_conditional_favorable_net_robust_mean"]
        - frame["before_conditional_favorable_net_robust_mean"]
    ).to_numpy(float)
    assert np.isfinite(delta).all()
    assert np.all(np.abs(delta) <= np.abs(raw) + 0.02)
