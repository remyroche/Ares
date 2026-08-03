from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_febmar_overlap_crowding_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("febmar_overlap_crowding_sensitivity", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_crowding_sensitivities_are_fixed_and_outcome_free() -> None:
    configs = MODULE._sensitivity_configs()
    assert [item[0] for item in configs] == ["omit_absolute_crowding", "continuous_raw_log_crowding"]
    fields = [field for _, continuous, categorical, _ in configs for field in (*continuous, *categorical)]
    assert not any("execution" in field or "outcome" in field or "label" in field for field in fields)
    assert "candidate_group_size_bin" not in fields
    assert set(configs[1][1]) == {"candidate_group_rows", "log1p_candidate_group_rows"}


def test_structural_rollups_are_outcome_free_and_preserve_mass() -> None:
    excluded = pd.DataFrame({
        "covariate_set": ["core", "core", "core"], "excluded_by_common_support": [True, False, True],
        "role": ["target_march", "target_march", "source_february"], "side_name": ["long", "long", "short"],
        "score_ventile": ["19", "19", "18"], "candidate_group_size_bin": ["q2", "q2", "q0"],
        "__symbol__": ["BTC", "BTC", "ETH"], "rows": [5, 100, 3],
    })
    result = MODULE._structural_rollups(excluded)
    role_side = result.loc[result.rollup.eq("role_side")]
    assert role_side.excluded_rows.sum() == 8
    assert set(result.rollup) == {"role_side", "role_side_score_band", "role_side_crowding_bin", "role_side_asset", "role_side_score_crowding_asset"}
