import numpy as np
import pytest

from scripts.evaluate_execution_ev_variable_admission import (
    ROOT,
    _repo_relative,
    select_variable_admission,
)


def test_variable_admission_can_return_fewer_than_global_capacity() -> None:
    score = np.array([-0.01, 0.001, 0.004, 0.003, -0.002])
    selected = select_variable_admission(
        score, capacity_fraction=0.40, predicted_net_floor=0.0025
    )
    assert selected.tolist() == [2, 3]


def test_variable_admission_can_fail_closed_to_zero() -> None:
    selected = select_variable_admission(
        np.array([-0.01, -0.02, 0.0]),
        capacity_fraction=0.50,
        predicted_net_floor=0.0,
    )
    assert selected.size == 0


def test_forced_capacity_is_one_global_stable_topk() -> None:
    selected = select_variable_admission(
        np.array([0.1, 0.4, 0.3, 0.2]),
        capacity_fraction=0.50,
        predicted_net_floor=None,
    )
    assert selected.tolist() == [1, 2]


def test_invalid_score_or_capacity_fails_closed() -> None:
    with pytest.raises(ValueError):
        select_variable_admission(
            np.array([0.1, np.nan]),
            capacity_fraction=0.10,
            predicted_net_floor=0.0,
        )
    with pytest.raises(ValueError):
        select_variable_admission(
            np.array([0.1]),
            capacity_fraction=0.0,
            predicted_net_floor=0.0,
        )


def test_repo_relative_accepts_relative_and_absolute_paths() -> None:
    relative = ROOT / "tests/test_evaluate_execution_ev_variable_admission.py"
    assert _repo_relative(relative) == (
        "tests/test_evaluate_execution_ev_variable_admission.py"
    )
    assert _repo_relative(relative.relative_to(ROOT)) == (
        "tests/test_evaluate_execution_ev_variable_admission.py"
    )
