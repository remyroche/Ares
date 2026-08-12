import numpy as np
import pandas as pd

from extreme_price_movements.complementary_base_heads import (
    AGREEMENT_FEATURES,
    agreement_features,
    causal_rank_norm,
)
from extreme_price_movements.query_candidate_definitions import base_head_query_definitions


def test_causal_rank_norm_uses_training_distribution_only() -> None:
    first = causal_rank_norm([0.0, 1.0, 2.0], [1.5])[0]
    # Appending held-period values must not change a held row's rank.
    second = causal_rank_norm([0.0, 1.0, 2.0], [1.5, 99.0])[0]
    assert np.isclose(first, second)
    assert np.isclose(first, 2.0 / 3.0)


def test_agreement_features_cover_requested_contract() -> None:
    frame = pd.DataFrame({"h1": [.99, .10], "h2": [.95, .90], "h3": [.91, .20]})
    result = agreement_features(frame, ["h1", "h2", "h3"])
    assert set(AGREEMENT_FEATURES).issubset(result.columns)
    assert np.isclose(result.loc[0, "base_heads_frac_rank_ge_p90"], 1.0)
    assert result.loc[0, "base_heads_prediction_std"] > 0.0
    assert 0.0 <= result.loc[1, "base_heads_agreement_entropy"] <= 1.0


def test_base_head_query_grammar_is_bounded_and_side_local() -> None:
    definitions = base_head_query_definitions()
    assert [item.cycle_hours for item in definitions] == [1, 2, 4, 6, 8, 12]
    assert all(item.side_local for item in definitions)
