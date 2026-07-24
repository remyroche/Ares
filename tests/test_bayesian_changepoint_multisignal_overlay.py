import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_bayesian_changepoint_multisignal_overlay.py"
SPEC = importlib.util.spec_from_file_location("bocpd_multisignal_overlay", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_shrunken_centroid_separates_simple_adverse_state() -> None:
    features = ["market_bocpd_score", "brl_bocpd"]
    positives = pd.DataFrame({"market_bocpd_score": [3.0, 3.1, 2.9, 3.2], "brl_bocpd": [2.0, 2.1, 1.9, 2.2]})
    controls = pd.DataFrame({"market_bocpd_score": [0.0, 0.1, -0.1, 0.2, 0.0], "brl_bocpd": [0.0, 0.2, -0.2, 0.1, 0.0]})
    score = pd.DataFrame({"market_bocpd_score": [3.05, 0.05], "brl_bocpd": [2.05, 0.05]})
    pos_score, control_score, oos_score = MODULE._centroid_scores(positives, controls, score, features)
    assert float(np.median(pos_score)) > float(np.median(control_score))
    assert float(oos_score[0]) > float(oos_score[1])


def test_high_risk_leaf_path_pressure_exports_observable_paths() -> None:
    import lightgbm as lgb

    x = np.vstack([
        np.tile(np.asarray([[0.0, 0.0], [0.1, 0.0], [0.2, 0.1]], dtype=np.float32), (5, 1)),
        np.tile(np.asarray([[2.0, 2.0], [2.1, 2.2], [2.2, 2.1]], dtype=np.float32), (5, 1)),
    ])
    y = np.asarray([0] * 15 + [1] * 15, dtype=np.int8)
    model = lgb.train(
        {"objective": "binary", "num_leaves": 3, "max_depth": 2, "min_data_in_leaf": 1, "verbosity": -1},
        lgb.Dataset(x, label=y, feature_name=["state_a", "state_b"]),
        num_boost_round=12,
    )
    score, rules = MODULE._high_risk_leaf_path_pressure(
        model, x, y, x, ["state_a", "state_b"], 0.85
    )
    assert score.shape == (len(x),)
    assert rules
    assert any("state_" in str(rule["path"]) for rule in rules)
    assert float(score[-1]) >= float(score[0])
