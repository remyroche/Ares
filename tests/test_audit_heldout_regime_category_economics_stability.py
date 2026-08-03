import pandas as pd

from scripts.audit_heldout_regime_category_economics_stability import qualify, select_global_top10


def test_global_top10_is_single_book_and_tie_deterministic():
    frame = pd.DataFrame({"candidate_id": ["b", "a", "c"], "score": [1.0, 1.0, 0.0], "side_name": ["long", "short", "long"]})
    selected = select_global_top10(frame, "score")
    assert selected.sum() == 1
    assert selected.iloc[1]


def test_qualification_requires_both_sides_and_three_eras():
    loo = pd.DataFrame({"taxonomy": ["state"] * 2, "category": ["1"] * 2, "economics_cohort": ["x"] * 2, "heldout_era": ["2025", "2026"], "side_name": ["long", "short"], "qualified_cell_support": [True, True], "good_sign_confirmed": [True, True], "poor_sign_confirmed": [False, False]})
    result = qualify(loo).iloc[0]
    assert not result.stable_good_net_ev
    assert "fewer_than_3_eras_in_comparable_cohort" in result.reasons
