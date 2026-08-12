import numpy as np

from scripts.run_side_local_conversion_residual_ev import _residual_grade


def test_residual_grade_is_ordinal_and_monotone():
    values = np.array([-300.0, -25.0, 0.0, 20.0, 500.0])
    grades = _residual_grade(values)
    assert grades.dtype.kind in "iu"
    assert np.all(np.diff(grades) >= 0)
    assert grades[0] < grades[-1]
