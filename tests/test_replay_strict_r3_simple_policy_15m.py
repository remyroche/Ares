from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "replay_strict_r3_simple_policy_15m.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_policy_replay", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_timestamp_complete_flat_paths_are_suspicious() -> None:
    valid = np.ones(10, dtype=bool)
    high = np.ones((10, 48), dtype=np.float32)
    low = np.ones((10, 48), dtype=np.float32)
    assert MODULE._coarse_paths_are_suspicious(valid, high, low)


def test_moving_paths_are_not_suspicious() -> None:
    valid = np.ones(10, dtype=bool)
    high = np.tile(np.linspace(1.0, 1.1, 48), (10, 1)).astype(np.float32)
    low = high - 0.001
    assert not MODULE._coarse_paths_are_suspicious(valid, high, low)
