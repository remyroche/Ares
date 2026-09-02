import numpy as np
import pandas as pd

from extreme_price_movements.strict_r3_bcf_mc1_mapper import derive_bcf_mc1_features


def test_native_bcf_mc1_features_are_target_free_and_timestamp_local():
    timestamps = [pd.Timestamp("2026-08-17T00:00:00Z")] * 2
    frame = pd.DataFrame({
        "candidate_id": ["a", "b"], "__decision_ts__": timestamps,
        "final_score": [0.9, 0.8], "base_rank42": [0.8, 0.7],
        "upstream": [0.75, 0.65], "consensus_rank": [0.7, 0.6],
        # Deliberately present target-looking fields: derive must neither
        # require nor consume them.
        "policy_net_bps": [9999.0, -9999.0],
    })
    for cap in (40, 60, 80, 100, 120):
        frame[f"residual_head__cap{cap}_ordinary__rank"] = [0.9, 0.2]
        frame[f"residual_head__cap{cap}_equal_month__rank"] = [0.8, 0.3]
    first = derive_bcf_mc1_features(frame)
    changed = frame.copy()
    changed["policy_net_bps"] *= -123.0
    second = derive_bcf_mc1_features(changed)
    assert first.equals(second)
    assert np.isfinite(first.drop(columns="candidate_id").to_numpy(float)).all()
    assert first.loc[first.candidate_id.eq("a"), "correctness_rank"].item() > first.loc[
        first.candidate_id.eq("b"), "correctness_rank"
    ].item()


def test_frozen_promoted_head_contract_is_target_free_and_does_not_infer_fields():
    frame = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": [pd.Timestamp("2026-08-17T00:00:00Z")] * 2,
        "final_score": [.9, .8], "base_rank42": [.8, .7],
        "upstream": [.75, .65], "consensus_rank": [.7, .6],
        "head__one__rank": [.9, .2], "head__two__rank": [.8, .3],
        "policy_net_bps": [500.0, -500.0],
    })
    first = derive_bcf_mc1_features(
        frame, rank_fields=["head__one__rank", "head__two__rank"],
        ordinary_rank_fields=["head__one__rank"],
    )
    changed = frame.copy()
    changed["policy_net_bps"] *= -10.0
    second = derive_bcf_mc1_features(
        changed, rank_fields=["head__one__rank", "head__two__rank"],
        ordinary_rank_fields=["head__one__rank"],
    )
    assert first.equals(second)
    assert np.isfinite(first.drop(columns="candidate_id").to_numpy(float)).all()
