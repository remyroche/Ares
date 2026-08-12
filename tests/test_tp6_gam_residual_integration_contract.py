from pathlib import Path

import pandas as pd

from scripts.run_tp6_sl4_rolling_gam_residual_integration import GAM_INPUT_FIELDS
from scripts.run_tp6_sl4_rolling_gam_residual_integration import _hard_gate_gam_score


def test_residual_meta_receives_only_canonical_gam_disagreement():
    assert GAM_INPUT_FIELDS == ["gam_delta_bps"]


def test_invalid_transport_is_exact_control_fallback():
    gam = [10.0, 20.0, 30.0]
    control = [1.0, 2.0, 3.0]
    gate = [True, False, True]
    assert _hard_gate_gam_score(gam, control, gate).tolist() == [10.0, 2.0, 30.0]


def test_completed_month_ahead_artifact_has_no_duplicate_gam_inputs():
    path = Path(
        "data_perp/artifacts/"
        "tp6_sl4_rolling_gam_residual_integration_20260815_v4/predictions.parquet"
    )
    if not path.exists():
        return
    columns = set(pd.read_parquet(path, engine="pyarrow").columns)
    assert "gam_delta_bps" in columns
    assert not {"gam_residual_bps", "gam_matched_mass", "gam_unmatched_mass"} & columns
