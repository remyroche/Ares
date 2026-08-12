"""Contract tests for the latent joint-correctness MLP experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "joint_correctness_mlp_meta",
    ROOT / "scripts/run_joint_correctness_mlp_meta.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _rows(n: int = 12) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(n)],
            "__ts__": ts,
            "side_name": "long",
            "gross_bps": np.linspace(-50, 250, n),
            "net_bps": np.linspace(-150, 150, n),
            "base_expected_bps": np.linspace(-20, 20, n),
            "label_available_ts": ts + pd.Timedelta(hours=12),
            "query_id": [f"q{i // 3}" for i in range(n)],
        }
    )


def test_target_ablation_contracts_are_ordered_and_economic() -> None:
    frame = _rows()
    for arm in MODULE.TARGET_ARMS:
        target = MODULE._target(frame, arm)
        assert target.dtype.kind in "iu"
        assert target.min() >= 0
        assert target.max() <= (1 if arm == "clear_binary" else 4)
    assert MODULE._target(frame, "clear_binary").sum() == 4


def test_joint_activation_state_features_have_fixed_head_width() -> None:
    rng = np.random.default_rng(7)
    correctness = rng.uniform(0.0, 1.0, size=(80, 10)).astype("float32")
    pairs = MODULE._pair_indices(correctness)
    joint = MODULE._joint_matrix(correctness, pairs)
    # Ten soft correctness values + ten bits + five summaries + pair products.
    assert joint.shape == (80, 25 + len(pairs))


def test_recent_features_are_prior_only() -> None:
    frame = _rows(12)
    correctness = np.ones((len(frame), 10), dtype="float32")
    recent = MODULE._recent_features(frame, correctness, windows=(7,))
    # The first row's own outcome resolves twelve hours later and cannot be in
    # its own prior window.  No synthetic same-row value is admitted.
    assert float(recent.iloc[0].max()) == 0.0
    assert set(recent.columns) == {f"head_{i:02d}_recent_correct_7d" for i in range(10)}


def test_global_tail_helper_is_not_timestamp_top_k() -> None:
    frame = _rows(12)
    frame["score"] = np.arange(len(frame), dtype=float)
    rows = pd.DataFrame(MODULE._tail_rows(frame, "score"))
    pooled = rows[(rows["period"] == "pooled") & (rows["tail"] == 0.05)].iloc[0]
    assert int(pooled.trades) == 1
    assert float(pooled.net_bps_per_trade) == 150.0
