from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_winner_hpo_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_precision_preservation_winner_hpo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _trial(contract: object) -> optuna.trial.FixedTrial:
    values = {"learning_rate": .05, "max_depth": 3, "feature_fraction": .8, "bagging_fraction": .8, "lambda_l2": 8.0}
    if contract.model_family == "catboost_queryrmse":
        values["random_strength"] = .5
    else:
        values.update({"lambda_l1": .05, "min_child_weight": 5.0, "min_gain_to_split": .001, "pairs_per_sample": 2})
    return optuna.trial.FixedTrial(values)


def test_hpo_suggestions_exclude_irrelvant_catboost_l1_and_fit() -> None:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    train = pd.DataFrame({"__decision_ts__": np.repeat(timestamps, 5), "candidate_id": range(40), "side_name": "long"})
    x = np.random.default_rng(1729).normal(size=(40, 4)).astype(np.float32)
    train["x0"] = x[:, 0]
    y = np.tile(np.arange(5, dtype=np.int8), 8)
    held = train.iloc[:10].copy()
    contracts = [
        MODULE.Contract(MODULE.stage1.ARMS[2], "g3_clipped_economic", "catboost_queryrmse", "cat"),
        MODULE.Contract(MODULE.stage1.ARMS[5], "g1_moderate_convex", "xgb_ndcg", "xgb"),
    ]
    for contract in contracts:
        params = MODULE._suggest(_trial(contract), contract)
        if contract.model_family == "catboost_queryrmse":
            assert "lambda_l1" not in params
        else:
            assert "lambda_l1" in params
        output = MODULE._fit_predict(contract=contract, params=params, train=train, labels=y, held=held, fields=("x0",), seed=1729)
        assert output["base_score"].notna().all()
        assert output.columns.tolist() == [*MODULE.IDENTITY, "base_score", "base_rank_ts"]


def test_compact_hpo_screen_still_requires_cross_year_temporal_support() -> None:
    months = MODULE._parse_screen_months("2025-11,2026-03,2026-07")
    assert [f"{month:%Y-%m}" for month in months] == ["2025-11", "2026-03", "2026-07"]
    with np.testing.assert_raises(ValueError):
        MODULE._parse_screen_months("2026-01,2026-03,2026-05")


def test_weighted_contract_uses_timestamp_normalised_row_weights() -> None:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    train = pd.DataFrame({"__decision_ts__": np.repeat(timestamps, 5), "candidate_id": range(40), "side_name": "long"})
    train["x0"] = np.random.default_rng(1730).normal(size=40).astype(np.float32)
    held = train.iloc[:10].copy()
    contract = MODULE.Contract(MODULE.stage1.ARMS[2], "g3_clipped_economic", "catboost_queryrmse", "cat", "tail_linear_125")
    labels = np.tile(np.arange(5, dtype=np.int8), 8)
    output = MODULE._fit_predict(
        contract=contract, params=MODULE._suggest(_trial(contract), contract), train=train,
        labels=labels, held=held, fields=("x0",), seed=1730,
    )
    assert output["base_score"].notna().all()
