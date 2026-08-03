import ast
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_trajectory_missingness_identical_row_ablation.py"


def _namespace():
    source = SCRIPT.read_text()
    ast.parse(source)
    namespace = {"__name__": "trajectory_ablation_test", "__file__": str(SCRIPT)}
    exec(compile(source, str(SCRIPT), "exec"), namespace)
    return namespace


def test_preregistered_arm_feature_placements_are_fixed():
    ns = _namespace()
    assert ns["features"]("baseline_existing_transition_control") == [
        "raw_score", "transition_lgbm_probability", "transition_lgbm_entropy", "transition_lgbm_margin"
    ]
    assert ns["features"]("trajectory_availability_neutral_only") == [
        "raw_score", "trajectory_available", "trajectory_transition_probability", "probability_entropy", "top2_margin"
    ]
    assert ns["features"]("regime_plus_existing_transition_plus_trajectory") == [
        "raw_score", "transition_lgbm_probability", "transition_lgbm_entropy", "transition_lgbm_margin",
        "trajectory_available", "trajectory_transition_probability", "probability_entropy", "top2_margin",
        "regime_change_probability_mean", "regime_state_age_hours",
    ]


def test_script_contains_fixed_neutral_missingness_constants_and_hourly_contract():
    source = SCRIPT.read_text()
    assert "fillna(.5)" in source
    assert "fillna(np.log(2))" in source
    assert "fillna(0.)" in source
    assert "selected_global_top10" in source
