import numpy as np
import pandas as pd

from scripts.diagnose_base_margin_historical_transfer import (
    FEATURE,
    SCORE,
    TARGET,
    global_top_fraction,
    screen_metrics,
)


def test_global_top_fraction_has_no_timestamp_or_side_quota() -> None:
    frame = pd.DataFrame(
        {
            SCORE: [0.1, 0.9, 0.8, 0.2],
            TARGET: [0.0, 0.0, 0.0, 0.0],
            FEATURE: [0.0, 0.0, 0.0, 0.0],
            "side_name": ["long", "short", "short", "long"],
        }
    )
    selected = global_top_fraction(frame, 0.5)
    assert selected.index.tolist() == [1, 2]


def test_frozen_screen_reports_keep_minus_drop_without_retuning() -> None:
    selected = pd.DataFrame(
        {
            FEATURE: [0.8, 0.7, 0.2, 0.1],
            TARGET: [0.02, 0.01, -0.01, -0.02],
        }
    )
    result = screen_metrics(
        selected, threshold=0.5, direction=1.0, scope="test"
    )
    assert result["keep_rows"] == 2
    assert result["drop_rows"] == 2
    assert np.isclose(result["keep_net_bps"], 150.0)
    assert np.isclose(result["drop_net_bps"], -150.0)
    assert np.isclose(result["keep_minus_drop_net_bps"], 300.0)
