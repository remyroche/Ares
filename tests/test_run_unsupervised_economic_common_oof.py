from __future__ import annotations

import numpy as np

from scripts.run_unsupervised_economic_common_oof import RETAINED_REPRESENTATION_DIAGNOSTICS, _mapper, feature_lists


def test_unsupervised_arms_keep_regime_and_transition_layers_distinct() -> None:
    fields = feature_lists()
    assert "gmm_ood_score" in fields["gmm_only"]
    assert "gmm_posterior_max" not in fields["gmm_only"]
    assert "gmm_entropy" not in fields["gmm_only"]
    assert "dae_b16_00" in fields["dae_only"]
    assert "gmm_ood_score" in fields["gmm_plus_dae"] and "dae_b16_00" in fields["gmm_plus_dae"]
    assert fields["failure_first_context"][-2:] == ["p_failure_destination_3h", "p_transition_within_3h"]
    assert fields["failure_destination_only"][-1] == "p_failure_destination_3h"
    assert fields["failure_transition_only"][-1] == "p_transition_within_3h"
    assert all("timing" not in x and "mae" not in x and "wait" not in x for values in fields.values() for x in values)
    assert "dae_reconstruction_error_zscore" in RETAINED_REPRESENTATION_DIAGNOSTICS


def test_causal_mapper_falls_back_safely_for_constant_scores() -> None:
    mapper = _mapper(np.array([.1, .1, .1]), np.array([.02, .01, .00]))
    values = mapper(np.array([.2, .3]))
    assert values.shape == (2,)
    assert np.isfinite(values).all()
