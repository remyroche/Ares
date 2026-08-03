import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nonlinear_alpha_tail", ROOT / "scripts/run_nonlinear_alpha_tail_cost_clearing_hurdle.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)


def test_timestamp_side_alpha_rank_uses_score_symbol_candidate_stable_ties():
    frame = pd.DataFrame({
        "candidate_id": ["z", "a", "b", "c"],
        "side_name": ["long", "long", "long", "short"],
        "__symbol__": ["BBB", "AAA", "AAA", "AAA"],
        "__ts__": pd.to_datetime(["2026-01-01T00:00Z"] * 4),
        MOD.ALPHA: [.5, .5, .5, .1],
    })
    result = MOD.alpha_rank_features(frame)
    long = result.loc[result.side_name.eq("long")].sort_values("alpha_rank_timestamp_side")
    assert long.candidate_id.tolist() == ["a", "b", "z"]
    assert long.alpha_rank_timestamp_side.tolist() == [1, 2, 3]
    assert long.alpha_ventile_timestamp_side.tolist() == [7, 14, 20]
    assert result.loc[result.side_name.eq("short"), "alpha_percentile_timestamp_side"].iloc[0] == .5


def test_fractional_top_splits_only_the_cutoff_tie_without_side_quota():
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__symbol__": ["A", "B", "C", "D"],
        "side_name": ["long", "long", "short", "short"],
        "mapped_ev": [4., 3., 3., 1.],
    })
    top, audit = MOD.fractional_top(frame, "mapped_ev", .375)
    assert top.candidate_id.tolist() == ["a", "b", "c"]
    assert np.allclose(top.selection_weight.to_numpy(), [1., .25, .25])
    assert audit["realized_selection_mass"] == 1.5
    assert audit["cutoff_tie_rows"] == 2


def test_causal_map_excludes_labels_resolving_at_fold_start():
    prior = pd.DataFrame({
        "execution_label_end_utc": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-02T00:00Z"]),
        "raw_score": [0., 1.], MOD.NET: [-.01, .01],
    })
    mapped, audit = MOD.causal_map(prior, np.array([.5]), pd.Timestamp("2026-01-02T00:00Z"))
    assert not audit["map_eligible"]
    assert np.isnan(mapped[0])
