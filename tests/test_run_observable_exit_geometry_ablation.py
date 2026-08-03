from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_observable_exit_geometry_ablation import (
    ARM_NAMES,
    BRACKET_OVERRIDES,
    PARENT_INDEX,
    _fold_local_top_k,
    _one_se_arm,
    _variant,
)


def test_fold_local_top_k_is_global_within_each_fold_and_deterministic() -> None:
    rows = []
    for fold in (0, 1):
        for index in range(10):
            rows.append(
                {
                    "__ts__": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(hours=fold * 10 + index),
                    "__symbol__": f"S{index}",
                    "side_name": "long" if index % 2 else "short",
                    "candidate_id": f"{fold}-{index}",
                    "fold": fold,
                    "score": float(index),
                }
            )
    selected, audit = _fold_local_top_k(
        pd.DataFrame(rows),
        fold_col="fold",
        score_col="score",
        top_fraction=0.20,
    )
    assert len(selected) == 4
    assert selected.groupby("fold").size().to_dict() == {0: 2, 1: 2}
    assert set(selected["score"]) == {8.0, 9.0}
    assert [row["selected_rows"] for row in audit] == [2, 2]
    assert all(len(row["selected_identity_sha256"]) == 64 for row in audit)


def test_one_se_arm_falls_back_without_support_or_clear_gain() -> None:
    net = np.zeros((20, len(ARM_NAMES)), dtype=float)
    net[:, ARM_NAMES.index("stop_0p90")] = 0.01
    insufficient = _one_se_arm(net, np.arange(4), min_support=5)
    assert insufficient["arm"] == "parent"
    selected = _one_se_arm(net, np.arange(20), min_support=5)
    assert selected["arm"] == "stop_0p90"

    tied = np.zeros_like(net)
    fallback = _one_se_arm(tied, np.arange(20), min_support=5)
    assert fallback["arm"] == ARM_NAMES[PARENT_INDEX]


def test_variant_changes_only_requested_geometry_axis() -> None:
    source = {
        "sl_mult": 4.0,
        "trailing_activation_mult": 2.0,
        "giveback_beta": 0.5,
    }
    changed = _variant(source, "giveback_beta", 1.1)
    assert changed["giveback_beta"] == 0.55
    assert changed["sl_mult"] == source["sl_mult"]
    assert changed["trailing_activation_mult"] == source["trailing_activation_mult"]
    assert source["giveback_beta"] == 0.5
    bracket = dict(source)
    bracket.update(BRACKET_OVERRIDES["bracket_tp2p00_sl1p00"])
    assert bracket["hard_tp_abs_pct"] == 0.02
    assert bracket["sl_abs_cap_pct"] == 0.01
