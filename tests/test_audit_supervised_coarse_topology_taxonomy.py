from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "supervised_topology", ROOT / "scripts/audit_supervised_coarse_topology_taxonomy.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_topology_labels_keep_rotation_and_other_abstained() -> None:
    labels = MODULE.taxonomy_label(
        pd.Series([0, 2, 3, 0]), pd.Series([4, 0, 1, 0])
    )
    assert labels.tolist() == [
        "baseline_to_nonbaseline_onset",
        "nonbaseline_to_baseline_normalization",
        "nonbaseline_to_nonbaseline_rotation",
        "other_abstain",
    ]


def test_pairwise_profile_correlations_are_era_explicit() -> None:
    output = MODULE.corrs(
        {2022: pd.Series([1.0, 0.0]), 2023: pd.Series([0.0, 1.0])}, "test", 0
    )
    assert len(output) == 1
    assert output[0]["era_a"] == 2022
    assert output[0]["era_b"] == 2023
