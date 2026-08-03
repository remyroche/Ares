from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_marapr2025_identical_causal_score_bridge import (
    BridgeError,
    LAYERS,
    build_bridge,
    causal_pooled_maps,
)


def _row(day: str, offset: int) -> dict[str, object]:
    timestamp = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=offset)
    return {
        "candidate_id": f"{day}|{offset}",
        "side_name": "long" if offset % 2 == 0 else "short",
        "__symbol__": f"S{offset}",
        "__ts__": timestamp,
    }


def _sources() -> tuple[pd.DataFrame, ...]:
    identities = [_row("2025-03-01", i) for i in range(4)] + [
        _row("2025-03-02", i) for i in range(4)
    ]
    raw = pd.DataFrame(identities)
    raw["candidate_month"] = "2025-03"
    raw["execution_decision_utc"] = raw["__ts__"] + pd.Timedelta(hours=1)
    raw["execution_label_end_utc"] = (
        raw["execution_decision_utc"] + pd.Timedelta(hours=12)
    )
    raw["execution_gross_ev_12h"] = np.linspace(-0.01, 0.04, len(raw))
    raw["execution_cost_return"] = 0.01
    raw["execution_net_ev_12h"] = (
        raw["execution_gross_ev_12h"] - raw["execution_cost_return"]
    )
    raw["execution_mfe_return_12h"] = np.linspace(0.0, 0.05, len(raw))
    raw["execution_mae_return_12h"] = np.linspace(-0.04, -0.01, len(raw))
    raw["execution_exit_class"] = "timeout"
    raw["opportunity_gross_above_cost_0bps"] = (
        raw["execution_net_ev_12h"] > 0
    )
    raw["__first_touch_target_soft__"] = np.linspace(0.0, 1.0, len(raw))
    raw["score_base_alpha"] = np.linspace(0.1, 0.8, len(raw))
    raw["score_residual_expected_ev"] = np.linspace(-0.02, 0.03, len(raw))
    raw["direct_q25_return"] = np.linspace(-0.03, 0.02, len(raw))

    base = raw.loc[:, list(("candidate_id", "side_name", "__symbol__", "__ts__"))].copy()
    base["base_oof_score"] = raw["score_base_alpha"]
    base["__decision_ts__"] = raw["execution_decision_utc"]
    residual = base.copy()
    residual["residual_expected_ev"] = raw["score_residual_expected_ev"]
    residual["residual_is_oof"] = True
    direct = raw.loc[
        :,
        [
            "candidate_id",
            "side_name",
            "__symbol__",
            "__ts__",
            "execution_net_ev_12h",
            "execution_label_end_utc",
        ],
    ].rename(columns={"execution_label_end_utc": "label_resolution_utc"})
    direct["q25_net_bps"] = raw["direct_q25_return"] * 1e4
    direct["era"] = "old_march"
    membership = direct.loc[
        :,
        [
            "candidate_id",
            "side_name",
            "__symbol__",
            "__ts__",
            "label_resolution_utc",
            "era",
        ],
    ].copy()
    membership["direct_oof_fold"] = "old_march"
    membership["direct_fit_cutoff_utc"] = pd.Timestamp("2025-03-01", tz="UTC")
    return raw, base, residual, direct, membership


def test_causal_maps_share_prior_resolved_reference_rows() -> None:
    raw, *_ = _sources()
    raw["score_raw_base_alpha"] = raw["score_base_alpha"]
    raw["score_raw_direct_q25_ev"] = raw["direct_q25_return"]
    mapped, audit = causal_pooled_maps(
        raw,
        score_columns=("score_raw_base_alpha", "score_raw_direct_q25_ev"),
        minimum_rows=2,
    )
    assert not mapped.loc[mapped["__ts__"].dt.day.eq(1), "map_available"].any()
    assert mapped.loc[mapped["__ts__"].dt.day.eq(2), "map_available"].all()
    assert audit["reference_rows"].tolist() == [0, 4]
    assert pd.Timestamp(audit.iloc[1]["reference_label_end_max_utc"]) < pd.Timestamp(
        audit.iloc[1]["snapshot_utc"]
    )


def test_build_bridge_preserves_all_oof_lineages() -> None:
    raw, base, residual, direct, membership = _sources()
    bridge, _ = build_bridge(
        raw,
        base,
        residual,
        direct,
        membership,
        expected_raw_rows=8,
        expected_common_rows=4,
        minimum_rows=2,
    )
    assert len(bridge) == 4
    assert set(LAYERS.values()).issubset(bridge.columns)
    assert bridge["map_reference_rows"].eq(4).all()
    assert np.isfinite(bridge.loc[:, list(LAYERS.values())]).all().all()
    assert np.allclose(
        bridge["execution_gross_ev_12h"] - bridge["execution_cost_return"],
        bridge["execution_net_ev_12h"],
    )


def test_build_bridge_fails_if_direct_score_is_not_identical() -> None:
    raw, base, residual, direct, membership = _sources()
    direct["q25_net_bps"] += 1.0
    with pytest.raises(BridgeError, match="direct raw-score lineage"):
        build_bridge(
            raw,
            base,
            residual,
            direct,
            membership,
            expected_raw_rows=8,
            expected_common_rows=4,
            minimum_rows=2,
        )
