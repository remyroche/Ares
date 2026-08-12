from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_tp6_sl4_canonical_residual_meta_block_ablation import (
    ARM_BLOCKS,
    BLOCKS,
    _groups,
    _map_canonical,
)
from scripts.run_tp6_sl4_canonical_residual_meta_granular_ablation import (
    DEFAULT_CONTRACT,
    DEFAULT_STRUCTURAL,
    _load_structural,
)


def test_meta_target_is_exact_net_minus_canonical_expected() -> None:
    train = pd.DataFrame({"base_plus_consensus25": [0.1, 0.2, 0.8, 0.9], "exact_net_bps": [-100.0, -50.0, 50.0, 100.0]})
    held = pd.DataFrame({"base_plus_consensus25": [0.15, 0.85], "exact_net_bps": [9999.0, -9999.0]})
    tr_map, te_map = _map_canonical(train, held)
    assert np.isfinite(tr_map).all()
    assert np.isfinite(te_map).all()
    # Held outcomes are not consulted by the monotonic map.
    held2 = held.copy()
    held2["exact_net_bps"] = [-1e9, 1e9]
    _, te_map2 = _map_canonical(train, held2)
    np.testing.assert_allclose(te_map, te_map2)


def test_query_groups_are_four_hour_and_side_specific() -> None:
    frame = pd.DataFrame({
        "__ts__": pd.to_datetime(["2025-01-01 00:00Z", "2025-01-01 01:00Z", "2025-01-01 00:00Z", "2025-01-01 01:00Z"]),
        "side_name": ["long", "long", "short", "short"],
    })
    order, groups = _groups(frame)
    assert len(groups) == 2
    assert sorted(groups.tolist()) == [2, 2]
    assert sorted(order.tolist()) == [0, 1, 2, 3]


def test_block_contract_contains_no_gam_field() -> None:
    assert all("gam" not in field.lower() for fields in BLOCKS.values() for field in fields)
    assert all("gam" not in field.lower() for fields in ARM_BLOCKS.values() for field in fields)


def test_structural_contract_is_exposure_only_and_cluster_fields_are_present() -> None:
    frame, groups, audit = _load_structural(DEFAULT_STRUCTURAL, DEFAULT_CONTRACT)
    assert frame.candidate_id.is_unique
    assert audit["clusters"] == 6
    assert len(groups["structural_archetype"]) >= 40
    assert len(groups["structural_cluster"]) >= 20
    assert all("net" not in c.lower() and "gross" not in c.lower() and "residual" not in c.lower() for c in frame.columns)
    assert frame[groups["structural_transport"]].notna().all().all()
