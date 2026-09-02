from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_meta_selector_declares_strict_staged_contract() -> None:
    source = (ROOT / "scripts" / "select_strict_r3_p8u_meta_fullfeatures_v1.py").read_text()
    for token in (
        "full target-free universe",
        "under_bps100__timestamp",
        "strict_prequential_base_residual_anchor",
        "group_mda",
        "group_mda_permutations_are_within_timestamp_only",
        "randomized shallow strict-OOF ranker subspaces",
        "no_mc1_admission_portfolio_live_or_exchange_mutation",
    ):
        assert token in source


def test_full_meta_selector_keeps_the_requested_feature_ladder() -> None:
    source = (ROOT / "scripts" / "select_strict_r3_p8u_meta_fullfeatures_v1.py").read_text()
    assert "(120, 90, 70, 50, 35, 25)" in source
    assert "top three MDA/ladder contracts" in source


def test_full_meta_selector_conditions_cmi_on_joint_base_explanation() -> None:
    source = (ROOT / "scripts" / "select_strict_r3_p8u_meta_fullfeatures_v1.py").read_text()
    for token in (
        "BASE_EXPLANATION_V1 = GEOMETRY",
        "MiniBatchKMeans",
        "cmi_policy_given_base_explanation_v1",
        "mi_feature_base_explanation_v1",
        "outcome-free, deterministic strata",
    ):
        assert token in source
