from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_objective_funnel_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_precision_preservation_objective_funnel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_finalists_must_be_three_distinct_target_geometries(tmp_path: Path) -> None:
    root = tmp_path / "gain"
    root.mkdir()
    arms = [MODULE.stage1.ARMS[0], MODULE.stage1.ARMS[2], MODULE.stage1.ARMS[5]]
    pd.DataFrame([
        {"candidate": f"{arm.key}__g1_moderate_convex", "arm": arm.key, "family": arm.family,
         "gain_name": "g1_moderate_convex", "score_stable": 1.0 - index}
        for index, arm in enumerate(arms)
    ]).to_parquet(root / "objective_finalists.parquet", index=False)
    selected = MODULE._finalist_targets(root)
    assert [(arm.key, gain) for arm, gain in selected] == [(arm.key, "g1_moderate_convex") for arm in arms]


def test_objective_rankers_hold_all_nonobjective_settings_fixed() -> None:
    arm = MODULE.stage1.ARMS[2]
    ranker = MODULE._ranker(MODULE.Candidate(arm, "g3_clipped_economic", "lambdarank"), seed=1).get_params()
    xendcg = MODULE._ranker(MODULE.Candidate(arm, "g3_clipped_economic", "rank_xendcg"), seed=1).get_params()
    assert ranker["objective"] == "lambdarank"
    assert xendcg["objective"] == "rank_xendcg"
    for name in ("n_estimators", "learning_rate", "max_depth", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        assert ranker[name] == xendcg[name]
    assert ranker["label_gain"] == MODULE.gain.GAIN_SCHEDULES["g3_clipped_economic"]
