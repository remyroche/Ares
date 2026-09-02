from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_meta_catboost_hpo_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("strict_r3_p8u_meta_catboost_hpo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_inner_early_stop_never_splits_exact_timestamp_queries() -> None:
    timestamps = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    frame = pd.DataFrame({"__decision_ts__": np.repeat(timestamps, 3)})
    fit, valid = MODULE._inner_masks(frame)
    assert fit.any() and valid.any()
    for _timestamp, group in frame.assign(fit=fit).groupby("__decision_ts__"):
        assert group.fit.nunique() == 1


def test_catboost_hpo_fit_scores_tiny_query_safe_panel() -> None:
    timestamps = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    train = pd.DataFrame({"__decision_ts__": np.repeat(timestamps, 5)})
    rng = np.random.default_rng(1729)
    x_train = rng.normal(size=(len(train), 4)).astype(np.float32)
    labels = np.tile(np.array([0, 0, 1, 1, 1], dtype=np.int32), len(timestamps))
    x_held = rng.normal(size=(9, 4)).astype(np.float32)
    values, best_iteration = MODULE._fit_predict(
        train_x=x_train,
        labels=labels,
        train=train,
        held_x=x_held,
        params={"learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 8.0, "random_strength": 0.5, "rsm": 0.8, "subsample": 0.8},
        seed=1729,
    )
    assert values.shape == (len(x_held),)
    assert np.isfinite(values).all()
    assert best_iteration >= 0
