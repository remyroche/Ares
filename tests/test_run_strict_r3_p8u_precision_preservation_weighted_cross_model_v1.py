from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_weighted_cross_model_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_weighted_cross_model", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_all_predeclared_model_families_are_present() -> None:
    assert set(MODULE.MODEL_FAMILIES) == {
        "lgbm_rank_xendcg", "lgbm_lambdarank", "catboost_queryrmse", "catboost_yetirank", "xgb_ndcg", "xgb_pairwise",
    }


def test_xgb_is_explicitly_excluded_when_it_cannot_represent_row_weights() -> None:
    assert set(MODULE.WEIGHT_COMPATIBLE_MODEL_FAMILIES) == {
        "lgbm_rank_xendcg", "lgbm_lambdarank", "catboost_queryrmse",
    }
    assert "one weight per query" in MODULE.XGB_WEIGHT_INCOMPATIBILITY
    assert "do not support object" in MODULE.CATBOOST_YETIRANK_WEIGHT_INCOMPATIBILITY


def test_cross_model_contract_freezes_weighting_and_external_metric() -> None:
    source = SCRIPT.read_text()
    assert 'scheme != "tail_linear_125"' in source
    assert "weights._query_safe_weights" in source
    assert "stable_score(panel, control)" in source
