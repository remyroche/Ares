import numpy as np
import pandas as pd

from extreme_price_movements.transport_supervised_archetypes import (
    SoftRule, configured_available_meta_features, effective_support, soft_membership,
)


def test_full_config_pool_excludes_label_like_fields() -> None:
    config = {"META_X": ["context", "future_net_bps", "target__bad", "missing"]}
    assert configured_available_meta_features(config, ["context", "future_net_bps", "target__bad"]) == ["context"]


def test_soft_rules_overlap_and_do_not_form_simplex() -> None:
    frame = pd.DataFrame({"x": [0., 1.], "y": [1., 0.]})
    first = soft_membership(frame, SoftRule(("x",), (1,), (0.,), (.2,)))
    second = soft_membership(frame, SoftRule(("y",), (1,), (0.,), (.2,)))
    assert np.all((first >= 0) & (first <= 1))
    assert np.all((second >= 0) & (second <= 1))
    assert not np.allclose(first + second, 1.)
    assert effective_support(first) > 0
