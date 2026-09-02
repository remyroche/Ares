from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_loss_funnel_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_precision_preservation_loss_funnel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stage2_retains_two_stage1_geometries_per_family(tmp_path: Path) -> None:
    root = tmp_path / "stage1"
    root.mkdir()
    rows: list[dict[str, object]] = []
    for index, arm in enumerate(MODULE.stage1.ARMS):
        rows.append({"arm": arm.key, "family": arm.family, "score_stable": float(100 - index)})
    pd.DataFrame(rows).to_parquet(root / "target_summary.parquet", index=False)
    selected = MODULE._selected_arms(root, top_per_family=2)
    families = pd.Series([arm.family for arm in selected]).value_counts().to_dict()
    assert families == {"atr": 2, "policy_ordinal": 2, "raw_bps": 2, "sqrt_atr": 2}
    assert MODULE.CONTROL_ARM in {arm.key for arm in selected}


def test_gain_schedules_match_predeclared_contract() -> None:
    assert MODULE.GAIN_SCHEDULES == {
        "g1_moderate_convex": [0.0, 1.0, 2.0, 4.0, 7.0, 11.0],
        "g2_stronger_top_tail": [0.0, 1.0, 3.0, 6.0, 11.0, 18.0],
        "g3_clipped_economic": [0.0, 0.5, 2.0, 3.0, 6.0, 8.0],
    }


def test_loss_funnel_ranker_is_lambdarank_with_exact_gain() -> None:
    arm = MODULE.stage1.ARMS[0]
    candidate = MODULE.Candidate(arm, "g2_stronger_top_tail")
    params = MODULE._ranker(candidate, seed=1729).get_params()
    assert params["objective"] == "lambdarank"
    assert params["label_gain"] == MODULE.GAIN_SCHEDULES["g2_stronger_top_tail"]
    assert params["lambdarank_truncation_level"] == 12
