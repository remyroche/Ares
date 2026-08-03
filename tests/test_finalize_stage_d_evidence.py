from scripts.finalize_stage_d_evidence import (
    LINEAGE,
    NAMED_TESTS,
    TERMINAL,
    validate_canonical,
)


def test_canonical_stage_d_evidence_is_complete_and_single_decision() -> None:
    evidence = validate_canonical()
    assert evidence["terminal_decision"] == TERMINAL
    assert evidence["lineage_disposition"] == LINEAGE
    assert evidence["population_rows"] == 108139
    assert evidence["no_entry_or_portfolio_policy_change"] is True
    assert len(NAMED_TESTS) == 21
