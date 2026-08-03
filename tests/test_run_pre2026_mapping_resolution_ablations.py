import numpy as np
import pandas as pd

from scripts.run_pre2026_mapping_resolution_ablations import (
    METHODS,
    MappingMethod,
    _fit_method,
    _select,
    _strict_rank,
    _tie_metrics,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame({
        "raw_score": np.linspace(-1, 1, 40),
        "execution_net_ev_12h": np.r_[np.linspace(-.02, .01, 20), np.linspace(-.01, .03, 20)],
        "side_name": ["long"] * 20 + ["short"] * 20,
    })


def test_all_registered_maps_are_increasing_and_fit_pre2026_scores_only():
    history = _history()
    for method in METHODS:
        mapper, _ = _fit_method(history, method)
        mapped = mapper(np.array([-.8, -.2, .3, .8]), np.array(["long", "long", "short", "short"]))
        assert np.isfinite(mapped).all()
        assert mapped[0] <= mapped[1]
        assert mapped[2] <= mapped[3]


def test_strict_rank_removes_exact_plateaus_without_reversing_score_order():
    scores = np.array([.1, .3, .2, .4])
    resolved = _strict_rank(np.zeros(4), scores)
    assert len(np.unique(resolved)) == 4
    assert np.array_equal(np.argsort(scores), np.argsort(resolved))


def test_selection_remains_one_global_top10_with_raw_only_inside_map_tie():
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        "mapped_score": [1.0] * 3 + [0.0] * 7,
        "raw_score": [.2, .9, .5, .1, .2, .3, .4, .5, .6, .7],
    })
    selected = _select(frame)
    assert selected.loc[selected.selected_global_top10, "candidate_id"].tolist() == ["b"]
    ties = _tie_metrics(frame)
    assert ties["cutoff_tie_rows"] == 3
    assert not ties["resolution_gate_pass"]
