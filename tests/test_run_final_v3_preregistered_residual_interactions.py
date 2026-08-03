from scripts.run_final_v3_preregistered_residual_interactions import ARMS, CONTROL_NAMES, _features
from scripts import run_final_identical_row_regime_stack_gam_ablation as base
import pandas as pd


def test_preregistration_has_only_the_approved_side_local_arms() -> None:
    names = {arm.name for arm in ARMS}
    assert names == {"long_residual_x_regime_state_age", "long_residual_x_transition_probability", "long_residual_x_combined", "short_residual_x_transition_probability", "short_residual_x_combined"}
    assert not any(arm.active_side == "short" and arm.context_fields == ("regime_state_age_hours",) for arm in ARMS)
    assert CONTROL_NAMES == ("baseline", "gam_regime_only", "gam_transition_only", "gam_combined")


def test_interaction_fields_are_explicit_residual_products() -> None:
    arm = ARMS[-1]
    frame = pd.DataFrame({base.RESIDUAL: [1.0], "regime_state_age_hours": [2.0], "transition_lgbm_probability": [.5]})
    out, fields = _features(frame, arm)
    assert fields[0] == base.RESIDUAL
    assert out[f"{base.RESIDUAL}__x__regime_state_age_hours"].iloc[0] == 2.0
    assert out[f"{base.RESIDUAL}__x__transition_lgbm_probability"].iloc[0] == .5
