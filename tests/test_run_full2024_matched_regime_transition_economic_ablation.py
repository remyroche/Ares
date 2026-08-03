import numpy as np
import pandas as pd

from scripts.run_full2024_matched_regime_transition_economic_ablation import (
    ARMS,
    arm_features,
    causal_recent_ev_mapping,
    pooled_global_top_mask,
    summarize_ablation,
)


def _panel() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=320, freq="6h", tz="UTC")
    frame = pd.DataFrame({
        "candidate_id": [f"c{index}" for index in range(len(timestamps))], "__ts__": timestamps,
        "__symbol__": ["BTC/USD:USD"] * len(timestamps), "side_name": np.where(np.arange(len(timestamps)) % 2, "long", "short"),
        "score_residual_expected_ev": np.linspace(-0.02, 0.02, len(timestamps)),
        "execution_net_ev_12h": np.linspace(-0.01, 0.02, len(timestamps)),
        "execution_gross_ev_12h": np.linspace(0.0, 0.03, len(timestamps)), "execution_cost_return": [0.01] * len(timestamps),
        "__reconstructed_soft_alpha_12h__": np.linspace(0.0, 1.0, len(timestamps)), "__opportunity_occurred_12h__": np.where(np.arange(len(timestamps)) % 3, 1, 0),
        "side_is_long": np.where(np.arange(len(timestamps)) % 2, 1.0, 0.0),
    })
    for name in ("regime_state_p__0", "regime_state_p__1", "transition_state_p__stable", "transition_state_p__transition"):
        frame[name] = np.linspace(0.1, 0.9, len(frame))
    frame["transition_active_probability"] = np.linspace(0.0, 1.0, len(frame))
    return frame


def test_global_top_mask_is_not_per_timestamp_or_side() -> None:
    frame = _panel().iloc[:4].copy()
    selected = pooled_global_top_mask(frame, pd.Series([0.1, 0.9, 0.8, 0.2], index=frame.index), fraction=0.5)
    assert frame.loc[selected, "candidate_id"].tolist() == ["c1", "c2"]


def test_context_arms_exclude_actions_and_keep_cold_start_matched() -> None:
    frame = _panel()
    features = arm_features(frame.columns)
    assert set(features) == set(ARMS)
    mapped, folds = causal_recent_ev_mapping(frame)
    january = mapped.loc[mapped["__ts__"].dt.month.eq(1)]
    assert january[[f"mapped_score__{arm}" for arm in ARMS]].nunique(axis=1).eq(1).all()
    assert folds.loc[folds["fold_month"].eq("2024-01"), "mode"].eq("cold_start_raw_residual_score").all()


def test_summary_uses_same_global_selection_count_for_all_arms() -> None:
    mapped, _ = causal_recent_ev_mapping(_panel())
    aggregate, period, distribution, side, worst = summarize_ablation(mapped)
    assert aggregate["selected_rows"].nunique() == 1
    assert set(aggregate["arm"]) == set(ARMS)
    assert {"week", "month"}.issubset(distribution["frequency"])
    assert not period.empty and not side.empty and not worst.empty
