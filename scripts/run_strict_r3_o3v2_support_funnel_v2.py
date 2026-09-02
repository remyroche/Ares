#!/usr/bin/env python3
"""Corrected O3-v2 support/weight screen.

This is a thin schema-v2 wrapper around the original support-screen engine.
It deliberately changes only the *training-row weighting* contract and always
writes a fresh immutable output root.  It never emits a semantic/path outcome
field into held score receipts, invokes MC1, or alters live/canonical artifacts.

Pair-semantic confidence (S3) is implemented as a bounded *training-row*
weight around the closest base-ranked peer, so it remains valid for the
retained continuous T2/T6 targets without pretending it is an inference input.
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


SCHEMA = "strict_r3_o3v2_support_funnel_v2"
SUPPORT_ARMS = (
    "S0_uniform",
    "S1_archetype_balance",
    "S2_semantic_certainty",
    "S3_pair_semantic_confidence",
    "S4_hard_base_error",
    "S5_policy_state",
    "SB1_error_archetype",
    "SB2_error_policy_state",
    "SB3_error_semantic",
    "SB3_error_pair_semantic",
)


def _nearest_base_pair_error(train: pd.DataFrame) -> np.ndarray:
    """Return an ex-post training-only hard-error indicator for closest pairs.

    A row is emphasised only where the base gave two candidates materially
    similar rank, their policy results differ materially, and the base ordering
    has the opposite sign from the outcome ordering.  No such value survives
    the fit: it is a sample weight, never a score feature.
    """
    result = np.zeros(len(train), dtype=bool)
    work = train.loc[:, ["__decision_ts__", "base_rank_ts", "semantic_policy_net_bps"]].copy()
    work["__row_id__"] = np.arange(len(work), dtype=np.int64)
    work["base_rank_ts"] = pd.to_numeric(work["base_rank_ts"], errors="coerce")
    work["semantic_policy_net_bps"] = pd.to_numeric(work["semantic_policy_net_bps"], errors="coerce")
    for _stamp, group in work.groupby("__decision_ts__", sort=False):
        group = group.dropna().sort_values("base_rank_ts")
        if len(group) < 2:
            continue
        base = group["base_rank_ts"].to_numpy(float)
        policy = group["semantic_policy_net_bps"].to_numpy(float)
        left_gap = np.r_[np.inf, np.abs(np.diff(base))]
        right_gap = np.r_[np.abs(np.diff(base)), np.inf]
        choose_left = left_gap <= right_gap
        other = np.where(choose_left, np.arange(len(group)) - 1, np.arange(len(group)) + 1)
        valid = (other >= 0) & (other < len(group))
        pair_gap = np.minimum(left_gap, right_gap)
        policy_gap = np.full(len(group), np.nan, dtype=float)
        policy_gap[valid] = policy[valid] - policy[other[valid]]
        base_delta = np.full(len(group), np.nan, dtype=float)
        base_delta[valid] = base[valid] - base[other[valid]]
        hard = valid & (pair_gap <= 0.10) & (np.abs(policy_gap) >= 100.0) & (base_delta * policy_gap < 0.0)
        result[group["__row_id__"].to_numpy(dtype=np.int64)] = hard
    return result


def _nearest_pair_semantic_confidence(train: pd.DataFrame) -> np.ndarray:
    """Bounded agreement with the closest base-ranked training peer.

    This is a pair-quality weight, not a learned feature.  It captures whether
    a local base-rank comparison is semantically coherent before the model
    sees it; held rows never receive this quantity.
    """
    result = np.full(len(train), 0.75, dtype=float)
    work = train.loc[:, ["__decision_ts__", "base_rank_ts", "semantic_tbm_event", "semantic_axis_f_exit5"]].copy()
    work["__row_id__"] = np.arange(len(work), dtype=np.int64)
    work["base_rank_ts"] = pd.to_numeric(work["base_rank_ts"], errors="coerce")
    for _stamp, group in work.dropna(subset=["base_rank_ts"]).groupby("__decision_ts__", sort=False):
        group = group.sort_values("base_rank_ts")
        if len(group) < 2:
            continue
        base = group["base_rank_ts"].to_numpy(float)
        left_gap = np.r_[np.inf, np.abs(np.diff(base))]
        right_gap = np.r_[np.abs(np.diff(base)), np.inf]
        other = np.where(left_gap <= right_gap, np.arange(len(group)) - 1, np.arange(len(group)) + 1)
        valid = (other >= 0) & (other < len(group))
        event = group["semantic_tbm_event"].astype("string").fillna("invalid").to_numpy(str)
        exit5 = group["semantic_axis_f_exit5"].astype("string").fillna("invalid").to_numpy(str)
        matches = np.zeros(len(group), dtype=float)
        matches[valid] = (event[valid] == event[other[valid]]).astype(float) + (exit5[valid] == exit5[other[valid]]).astype(float)
        confidence = np.where(matches >= 2.0, 1.00, np.where(matches >= 1.0, 0.80, 0.60))
        result[group["__row_id__"].to_numpy(dtype=np.int64)] = confidence
    return result


def _components(train: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(train)
    archetype = train["semantic_archetype"].astype("string").fillna("invalid")
    archetype_counts = archetype.value_counts(dropna=False)
    archetype_weight = archetype.map(
        lambda value: np.sqrt(n / max(float(archetype_counts.loc[value]), 1.0))
    ).to_numpy(float)

    # Certainty is purposely a bounded confidence proxy, not an economic gain.
    event = train["semantic_tbm_event"].astype("string").fillna("invalid")
    sequence = train["semantic_axis_a_sequence"].astype("string").fillna("unknown")
    persistence = train["semantic_axis_c_persistence"].astype("string").fillna("unknown")
    certainty = np.where(event.eq("ambiguous"), 0.50, np.where(event.eq("vertical"), 0.75, 1.00))
    certainty *= np.where(sequence.str.contains("ambiguous|same", case=False, regex=True), 0.75, 1.00)
    certainty *= np.where(persistence.str.contains("unknown|mixed", case=False, regex=True), 0.85, 1.00)

    hard_error = 1.0 + 0.75 * _nearest_base_pair_error(train).astype(float)
    pair_semantic = _nearest_pair_semantic_confidence(train)

    # TB0 + canonical exit state form the explicit policy-state support label.
    policy_state = (
        train["semantic_tbm_event"].astype("string").fillna("invalid")
        + "|" + train["semantic_axis_f_exit5"].astype("string").fillna("invalid")
    )
    state_counts = policy_state.value_counts(dropna=False)
    policy_state_weight = policy_state.map(
        lambda value: np.sqrt(n / max(float(state_counts.loc[value]), 1.0))
    ).to_numpy(float)
    return {
        "uniform": np.ones(n, dtype=float),
        "archetype": impl._normalise(archetype_weight),
        "certainty": impl._normalise(certainty.astype(float)),
        "pair_semantic": impl._normalise(pair_semantic),
        "hard_base_error": impl._normalise(hard_error),
        "policy_state": impl._normalise(policy_state_weight),
    }


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
    elif arm == "S5_policy_state":
        raw = comp["policy_state"]
    elif arm == "SB1_error_archetype":
        raw = comp["hard_base_error"] * comp["archetype"]
    elif arm == "SB2_error_policy_state":
        raw = comp["hard_base_error"] * comp["policy_state"]
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
