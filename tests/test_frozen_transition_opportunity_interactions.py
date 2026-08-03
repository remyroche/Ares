from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_frozen_transition_opportunity_interactions.py"
)
SPEC = importlib.util.spec_from_file_location("interaction_audit", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows(month: str, size: int = 200) -> pd.DataFrame:
    rows = []
    for index in range(size):
        side = "long" if index % 2 == 0 else "short"
        score = index / size
        rows.append(
            {
                "candidate_id": f"{month}-{index:04d}",
                "side_name": side,
                "__symbol__": f"asset-{index % 4}",
                "__ts__": pd.Timestamp(f"{month}-01", tz="UTC")
                + pd.Timedelta(hours=index),
                "execution_net_ev_12h": score / 100 - 0.005,
                "direct_net_score": score / 100,
                "active_transition_probability": score,
                **{
                    risk: (size - index) / size
                    for risk in MODULE.RISK_COLUMNS
                },
            }
        )
    return pd.DataFrame(rows)


def test_thresholds_are_frozen_from_train_by_side() -> None:
    train = _rows("2025-03")
    test = _rows("2025-04")
    _, flagged = MODULE.threshold_flags(train, test)
    for side in ("long", "short"):
        expected = train.loc[train.side_name.eq(side), "risk__adverse"].quantile(0.8)
        observed = flagged.loc[
            flagged.side_name.eq(side), "threshold__risk__adverse"
        ].iloc[0]
        assert observed == expected


def test_side_local_fit_returns_all_test_rows() -> None:
    train = _rows("2025-03")
    test = _rows("2025-04")
    prediction = MODULE.fit_side_local_ridge(
        train, test, ["direct_net_score", "active_transition_probability"]
    )
    assert len(prediction) == len(test)
    assert np.isfinite(prediction).all()


def test_global_top_uses_candidate_id_ties() -> None:
    rows = _rows("2025-04", 20)
    rows["score"] = 1.0
    selected = MODULE.stable_top(rows, "score", 0.10)
    assert selected.candidate_id.tolist() == sorted(rows.candidate_id)[:2]


def test_conditional_output_contains_every_predeclared_modifier() -> None:
    april = _rows("2025-04")
    april["high__active_transition_probability"] = (
        april.active_transition_probability >= 0.8
    ).astype(float)
    for risk in MODULE.RISK_COLUMNS:
        april[f"high__{risk}"] = (april[risk] >= april[risk].quantile(0.8)).astype(float)
    result = MODULE.conditional_interactions(april)
    assert set(result.modifier) == set(MODULE.RISK_COLUMNS)
    did = result.loc[result.metric.eq("difference_in_differences")]
    assert len(did) == len(MODULE.RISK_COLUMNS)
