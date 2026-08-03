from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_identical_row_four_layer_ic_ev_diagnostic import top, unit_score


def test_global_top_uses_candidate_id_ties_not_side_or_timestamp_books() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["z", "a", "b"], "side_name": ["long", "short", "long"],
        "__symbol__": ["BTC"] * 3, "__ts__": [3, 2, 1], "score": [1., 1., 0.],
    })
    assert top(frame, "score", .34).candidate_id.tolist() == ["a", "z"]


def test_direct_bps_is_converted_only_for_calibration_not_ranking() -> None:
    frame = pd.DataFrame({"score_direct_q25_bps": [100., -50.]})
    assert np.allclose(unit_score(frame, "direct_ev_q25"), [.01, -.005])
