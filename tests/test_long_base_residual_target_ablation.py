from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_long_base_residual_target_ablation import base_target, calendar_masks, global_top_mask


def test_calendar_is_requested_12_8_4_4_split() -> None:
    masks = calendar_masks([
        "2023-04-01T00:00:00Z", "2024-03-31T23:00:00Z", "2024-04-01T00:00:00Z",
        "2024-07-31T23:00:00Z", "2024-08-01T00:00:00Z", "2024-11-30T23:00:00Z",
    ])
    assert masks["base_train"].tolist() == [True, True, False, False, False, False]
    assert masks["base_oos"].tolist() == [False, False, True, True, True, True]
    assert masks["meta_train"].tolist() == [False, False, True, True, False, False]
    assert masks["meta_oos"].tolist() == [False, False, False, False, True, True]


def test_base_targets_raise_with_cost_clearing_net() -> None:
    frame = pd.DataFrame({"execution_net_ev_12h": [-0.01, 0.01], "execution_gross_ev_12h": [0.00, 0.02], "execution_cost_return": [0.01, 0.01]})
    for name in ("cost_clear_0bps", "cost_clear_25bps", "cost_clear_upside"):
        values = base_target(frame, name)
        assert values[1] > values[0]
        assert np.all((values >= 0.0) & (values <= 1.0))


def test_global_top_book_is_not_timestamp_local() -> None:
    selected = global_top_mask([1.0, 100.0, 2.0, 99.0], 0.5)
    assert selected.tolist() == [False, True, False, True]
