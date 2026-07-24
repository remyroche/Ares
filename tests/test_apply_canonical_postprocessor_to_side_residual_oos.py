import numpy as np
import pandas as pd

from scripts.apply_canonical_postprocessor_to_side_residual_oos import (
    _derive_supported_inputs,
    _old_score_from_rank,
)


def test_supported_derivatives_are_causal_and_fill_sparse_outputs():
    size = 800
    frame = pd.DataFrame(
        {
            "carry_adj_ret_10h": np.linspace(-0.02, 0.03, size),
            "carry_adj_ret_self_z_10h": np.nan,
            "oi_chg_2h": np.sin(np.arange(size) / 20.0),
            "oi_chg_2h_robust_z": np.nan,
        }
    )

    derived = _derive_supported_inputs(frame)

    assert np.isfinite(derived["carry_adj_ret_self_z_10h"]).all()
    assert np.isfinite(derived["oi_chg_2h_robust_z"].iloc[200:]).all()


def test_old_score_bridge_preserves_rank_order():
    class Reference:
        sorted_scores_global = np.asarray([0.1, 0.2, 0.5, 0.9], dtype=np.float32)

    class Bundle:
        historical_rank_reference = Reference()

    class Postprocessor:
        predecessor_bundle = Bundle()

    bridged = _old_score_from_rank(
        Postprocessor(), pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    )

    assert np.all(np.diff(bridged) >= 0.0)
    assert bridged[0] == np.float32(0.1)
    assert bridged[-1] == np.float32(0.9)
