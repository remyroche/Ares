import numpy as np
import pandas as pd

from scripts.run_residual_mechanism_lookalike_validation import (
    _frozen_threshold_metrics,
    _parse_fold,
)


def test_parse_fold_requires_forward_chronology() -> None:
    start, end = _parse_fold("2025-01-01::2025-02-01")
    assert start < end


def test_frozen_threshold_metrics_never_use_oos_rank() -> None:
    frame = pd.DataFrame(
        {
            "risk": np.array([0.9, 0.8, 0.1, 0.0], dtype=np.float32),
            "event_start": [True, False, False, False],
        }
    )
    metrics = _frozen_threshold_metrics(frame, threshold=0.85)
    assert metrics["selected_days"] == 1
    assert metrics["selected_rate"] == 0.25
    assert metrics["event_recall"] == 1.0
