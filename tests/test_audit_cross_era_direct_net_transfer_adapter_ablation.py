import pandas as pd
import pytest

from scripts.audit_cross_era_direct_net_transfer_adapter_ablation import (
    assert_complete_coverage,
    promotion_gates,
    raw_ic_gate,
)


def _frame() -> pd.DataFrame:
    rows = []
    for side in ("long", "short"):
        for index in range(20):
            ts = pd.Timestamp("2026-07-20T00:00:00Z") + pd.Timedelta(hours=index)
            rows.append({"candidate_id": f"{side}-{index}", "side_name": side, "__symbol__": "BTC_USDT", "__ts__": ts, "execution_net_ev_12h": (index - 10) / 10_000.0, "score_parent_bps": float(index)})
    return pd.DataFrame(rows)


def test_exact_identity_coverage_fails_closed_on_missing_or_duplicate_labels():
    predictions = _frame().drop(columns="execution_net_ev_12h")
    labels = _frame().loc[:, ["candidate_id", "side_name", "__symbol__", "__ts__", "execution_net_ev_12h"]]
    assert assert_complete_coverage(predictions, labels)["identity_complete_one_to_one"] is True
    with pytest.raises(ValueError, match="coverage mismatch"):
        assert_complete_coverage(predictions, labels.iloc[:-1])
    duplicate = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        assert_complete_coverage(predictions, duplicate)


def test_negative_raw_within_side_ic_forbids_mapping_rescue():
    frame = _frame()
    frame.loc[frame["side_name"].eq("short"), "score_parent_bps"] *= -1.0
    gates = raw_ic_gate(frame, "score_parent_bps")
    short = gates.loc[(gates["side_name"] == "short") & (gates["period"] == "all")].iloc[0]
    assert not bool(short["mapping_eligible"])
    assert short["mapping_prohibition"] == "negative_or_undefined_raw_within_side_ic"


def test_promotion_gate_never_authorizes_mapping_and_rejects_negative_current_tail():
    historical = _frame()
    current = _frame()
    current["execution_net_ev_12h"] = -abs(current["execution_net_ev_12h"]) - .0001
    gates = promotion_gates(
        historical,
        current,
        "score_parent_bps",
        old_to_recent={
            "global_top10_net_ev_bps": 1.0,
            "raw_ic_gate_passed": True,
        },
    )
    assert gates["mapping_authorized"] is False
    assert gates["current_global_top10_positive"] is False
    assert gates["portfolio_replay_authorized"] is False


def test_promotion_gate_requires_positive_old_to_recent_transfer():
    frame = _frame()
    gates = promotion_gates(
        frame,
        frame,
        "score_parent_bps",
        old_to_recent={
            "global_top10_net_ev_bps": -1.0,
            "raw_ic_gate_passed": True,
        },
    )
    assert gates["old_to_recent_global_top10_positive"] is False
    assert gates["portfolio_replay_authorized"] is False
