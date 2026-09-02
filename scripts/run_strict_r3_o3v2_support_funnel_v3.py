#!/usr/bin/env python3
"""Schema-v3 O3-v2 supportive-label and policy-state screen.

The runner keeps realised path/state semantics strictly inside resolved
training-fold sample weights.  It adds the predeclared policy-state variants
needed to distinguish coarse TBM, four-state policy exit, five-state policy
exit, and the canonical sequential state.  Held score receipts remain
target-free, and the shared runner now uses exact-timestamp causal queries and
the calibrated-residual pair target for T3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_o3v2_support_funnel as impl  # noqa: E402
import run_strict_r3_o3v2_support_funnel_v2 as v2  # noqa: E402


SCHEMA = "strict_r3_o3v2_support_funnel_v4"
SUPPORT_ARMS = (
    "S0_uniform",
    "S1_archetype_balance",
    "S2_semantic_certainty",
    "S3_pair_semantic_confidence",
    "S4_hard_base_error",
    "S5_tbm_coarse",
    "S5_exit4_policy",
    "S5_exit5_policy",
    "S5_sequential_policy",
    "SB1_error_archetype",
    "SB2_error_policy_state",
    "SB3_error_semantic",
    "SB3_error_pair_semantic",
)


def _balanced_state_weight(values: pd.Series) -> np.ndarray:
    """Mild, capped inverse-frequency support for one fixed semantic axis."""
    labels = values.astype("string").fillna("invalid")
    n = len(labels)
    counts = labels.value_counts(dropna=False)
    raw = labels.map(lambda value: np.sqrt(n / max(float(counts.loc[value]), 1.0))).to_numpy(float)
    return impl._normalise(np.clip(raw, .25, 4.0))


def _components(train: pd.DataFrame) -> dict[str, np.ndarray]:
    base = dict(v2._components(train))
    tbm = train["semantic_tbm_event"].astype("string").fillna("invalid")
    exit4 = train["semantic_axis_f_exit4"].astype("string").fillna("invalid")
    exit5 = train["semantic_axis_f_exit5"].astype("string").fillna("invalid")
    # The sequential state is an explicit causal-policy trajectory label,
    # distinct from either the coarse first-barrier outcome or terminal exit.
    sequential = tbm + "|" + exit5
    base.update({
        "tbm_coarse": _balanced_state_weight(tbm),
        "exit4": _balanced_state_weight(exit4),
        "exit5": _balanced_state_weight(exit5),
        "sequential": _balanced_state_weight(sequential),
    })
    return base


def _weights(train: pd.DataFrame, arm: str) -> np.ndarray:
    comp = _components(train)
    if arm == "S0_uniform":
        raw = comp["uniform"]
    elif arm == "S1_archetype_balance":
        raw = comp["archetype"]
    elif arm == "S2_semantic_certainty":
        raw = comp["certainty"]
    elif arm == "S3_pair_semantic_confidence":
        raw = comp["pair_semantic"]
    elif arm == "S4_hard_base_error":
        raw = comp["hard_base_error"]
    elif arm == "S5_tbm_coarse":
        raw = comp["tbm_coarse"]
    elif arm == "S5_exit4_policy":
        raw = comp["exit4"]
    elif arm == "S5_exit5_policy":
        raw = comp["exit5"]
    elif arm == "S5_sequential_policy":
        raw = comp["sequential"]
    elif arm == "SB1_error_archetype":
        raw = comp["hard_base_error"] * comp["archetype"]
    elif arm == "SB2_error_policy_state":
        raw = comp["hard_base_error"] * comp["sequential"]
    elif arm == "SB3_error_semantic":
        raw = comp["hard_base_error"] * comp["certainty"]
    elif arm == "SB3_error_pair_semantic":
        raw = comp["hard_base_error"] * comp["pair_semantic"]
    else:
        raise ValueError(f"unsupported support arm: {arm}")
    return impl._normalise(raw)


def main() -> None:
    impl.SCHEMA = SCHEMA
    impl.SUPPORT_ARMS = SUPPORT_ARMS
    impl._components = _components
    impl._weights = _weights
    impl.main()


if __name__ == "__main__":
    main()
