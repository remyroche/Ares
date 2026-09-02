from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_singlebase_meta_mc1_diagnostic_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_singlebase_meta_mc1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_short_warmup_month_parser_requires_a_chronological_panel() -> None:
    assert [str(item)[:7] for item in MODULE._months("2026-06,2026-07")] == ["2026-06", "2026-07"]
    with np.testing.assert_raises(ValueError):
        MODULE._months("2026-07")


def test_score_bands_are_timestamp_local_and_invariant_to_other_timestamps() -> None:
    ts = pd.date_range("2026-06-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "candidate_id": [f"x{i}" for i in range(12)],
        "__decision_ts__": np.repeat(ts, 4),
        "base_rank_ts": np.tile([.9, .7, .4, .1], 3),
    })
    bands = MODULE._rank_bands(frame)
    assert bands.tolist()[:4] == bands.tolist()[4:8] == bands.tolist()[8:]
    assert bands.min() >= 0 and bands.max() <= 9


def test_fit_accepts_only_targetfree_coordinates_plus_prior_labels() -> None:
    ts = pd.date_range("2026-06-01", periods=8 * 24, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "candidate_id": [f"x{i}" for i in range(len(ts) * 4)],
        "__decision_ts__": np.repeat(ts, 4),
        "base_rank_ts": np.tile([.9, .7, .4, .1], len(ts)),
        "enhanced_base_bps": np.tile([2.0, 1.0, 0.0, -1.0], len(ts)),
        "meta_rank_ts": np.tile([.8, .6, .3, .2], len(ts)),
        "policy_net_bps": np.random.default_rng(1729).normal(20.0, 150.0, len(ts) * 4),
    })
    model, medians, curve = MODULE._fit(frame, ("base_rank_ts", "enhanced_base_bps", "meta_rank_ts"))
    assert len(medians) == 3
    assert curve.shape == (10,)
    assert np.isfinite(model.predict(frame.loc[:7, ["base_rank_ts", "enhanced_base_bps", "meta_rank_ts"]])).all()
