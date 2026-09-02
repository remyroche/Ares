from __future__ import annotations

from scripts.run_strict_r3_p8u_exact_1m_simple_policy_optimiser import (
    _early_stop_reached,
    _hpo_branch_specs,
)


def test_hpo_branch_specs_cover_both_mutually_exclusive_trailing_paths() -> None:
    assert _hpo_branch_specs(120) == (("fixed_gap", 60), ("dynamic_giveback", 60))
    assert _hpo_branch_specs(5) == (("fixed_gap", 3), ("dynamic_giveback", 2))
    assert _hpo_branch_specs(1, trials_per_branch=120) == (("fixed_gap", 120), ("dynamic_giveback", 120))


def test_hpo_branch_specs_rejects_insufficient_budget() -> None:
    try:
        _hpo_branch_specs(1)
    except ValueError as exc:
        assert "at least two trials" in str(exc)
    else:
        raise AssertionError("one HPO trial cannot cover two mutually exclusive trailing branches")


def test_branch_early_stop_requires_full_patience_after_last_improvement() -> None:
    assert not _early_stop_reached(trial_number=34, best_trial_number=5, patience=30)
    assert _early_stop_reached(trial_number=35, best_trial_number=5, patience=30)
    assert not _early_stop_reached(trial_number=35, best_trial_number=5, patience=0)
