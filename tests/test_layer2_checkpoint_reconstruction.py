import pandas as pd

from src.training.steps.labeling.label_based_layer_2 import _reconstruct_ortho_geometries


def test_checkpoint_reconstruction_preserves_labels_and_weights():
    events = ["2025-07-28T15:20:00", "2025-07-28T15:32:00", "2025-07-29T10:40:00"]
    labels = {events[0]: -1.0, events[1]: 1.0, events[2]: 0.0}
    weights = {events[0]: 0.5, events[1]: 1.5, events[2]: 1.0}

    candidates = [
        {
            "family": "CAUSAL_SURPRISE",
            "events": events,
            "labels": labels,
            "weights": weights,
            "params": {"risk_budget": 0.7},
            "metrics_raw": {"lift": 0.12},
        }
    ]

    geometries = _reconstruct_ortho_geometries(candidates)

    assert len(geometries) == 1
    geom = geometries[0]
    assert isinstance(geom.labels, pd.Series)
    assert isinstance(geom.weights, pd.Series)
    assert geom.labels.index.equals(pd.DatetimeIndex(events))
    assert geom.weights.index.equals(pd.DatetimeIndex(events))
    assert geom.labels.loc[pd.Timestamp(events[0])] == -1.0
    assert geom.weights.loc[pd.Timestamp(events[1])] == 1.5
