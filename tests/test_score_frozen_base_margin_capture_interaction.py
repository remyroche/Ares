from __future__ import annotations

import pandas as pd

from scripts.score_frozen_base_margin_capture_interaction import _top10_metrics


def test_top10_metric_is_a_single_global_book() -> None:
    frame = pd.DataFrame(
        {
            "score": [0.1, 0.9, 0.8, 0.2],
            "execution_net_ev_12h": [-0.02, 0.01, 0.02, -0.01],
            "base_margin_to_cutoff_z": [0.0, 1.0, 2.0, -1.0],
            "capture_probability": [0.2, 0.8, 0.9, 0.1],
        }
    )
    metrics = _top10_metrics(frame, "score", scope="one_book")
    # ceil(10% * 4) is one: the scorer must not form timestamp/side books.
    assert metrics["selected_rows"] == 1
    assert metrics["top10_net_bps"] == 100.0
