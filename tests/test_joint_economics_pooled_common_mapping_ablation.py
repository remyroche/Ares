import numpy as np
import pandas as pd
import pytest

from scripts import run_joint_economics_pooled_common_mapping_ablation as runner


def _frame() -> pd.DataFrame:
    rows = []
    for fold_id, start in enumerate(("2025-02-15", "2025-03-01", "2025-03-16")):
        for index in range(4):
            ts = pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=index)
            for side, score in (("long", 10.0 + index), ("short", -10.0 - index)):
                rows.append({"candidate_id": f"{fold_id}-{index}-{side}", "side_name": side, "__symbol__": "BTC", "__ts__": ts, "execution_label_end_utc": ts + pd.Timedelta(hours=13), "execution_net_ev_12h": 0.01 if side == "long" else -0.01, "fold_id": fold_id, "arm": "S0", "direct_primary_score": score})
    return pd.DataFrame(rows)


def test_oof_mapping_never_uses_current_or_future_fold_rows():
    frame = _frame()
    folds = tuple(runner.Fold(index, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")) for index, (start, end) in enumerate((("2025-02-15", "2025-03-01"), ("2025-03-01", "2025-03-16"), ("2025-03-16", "2025-04-01"))))
    mapped, audit = runner.causal_oof_mappings(frame, folds, "direct_primary_score")
    assert np.array_equal(mapped["pooled_affine_ridge_net"][frame.fold_id.eq(0)], frame.loc[frame.fold_id.eq(0), "direct_primary_score"])
    assert [item["prior_rows"] for item in audit] == [0, 8, 16]
    assert all(item["status"] == "prior_resolved_oof_mapping" for item in audit[1:])
    for fold in folds:
        prior = frame.fold_id.lt(fold.fold_id) & frame.execution_label_end_utc.lt(fold.validation_start)
        assert frame.loc[prior, "fold_id"].lt(fold.fold_id).all()


def test_side_residual_is_shrunk_and_capped_from_train_only_robust_net_scale():
    target = np.array([0.01] * 2 + [-0.01] * 2)
    prediction = np.zeros(4)
    values, audit = runner.side_residual_corrections(target, prediction, ["long", "long", "short", "short"], support=5000.0, cap_multiplier=0.1)
    assert audit["long"]["shrink"] == pytest.approx(2 / 5002)
    assert abs(values["long"]) <= audit["long"]["cap"]
    assert values["long"] > 0
    assert values["short"] < 0


def test_global_rank_is_deterministic_and_ties_break_by_candidate_id():
    frame = pd.DataFrame({"candidate_id": ["z", "a", "b", "c"], "side_name": ["long", "short", "long", "short"]})
    first = runner.stable_global_top_mask(frame, [1.0, 1.0, 0.5, 0.4], 0.20)
    second = runner.stable_global_top_mask(frame, [1.0, 1.0, 0.5, 0.4], 0.20)
    assert np.array_equal(first, second)
    assert frame.loc[first, "candidate_id"].tolist() == ["a"]


def test_selection_has_no_side_quota_or_replacement():
    frame = pd.DataFrame({"candidate_id": ["l1", "l2", "s1", "s2"], "side_name": ["long", "long", "short", "short"], "execution_net_ev_12h": [0.1, 0.09, 0.01, 0.0]})
    row, sides, gate = runner._selection_rows(frame, np.array([4.0, 3.0, 2.0, 1.0]), arm="S0", score_name="direct", split="oof", scope="raw", fraction=0.20)
    assert row["selected_rows"] == 1
    assert [side["selected_rows"] for side in sides] == [1, 0]
    assert gate["selection_modified"] is False
    assert gate["side_balance_gate_pass"] is False
