import numpy as np
import pandas as pd
import pytest

from scripts.materialize_cross_era_tail_payoff_dataset import (
    derive_event_columns,
    add_candidate_relative_context,
    normalize_identity,
    validate_feature_contract,
)


def test_validate_feature_contract_requires_exact_unique_256():
    columns = [f"x{i}" for i in range(256)]
    assert validate_feature_contract({"feature_columns": columns}) == columns
    with pytest.raises(ValueError):
        validate_feature_contract({"feature_columns": columns[:-1]})
    with pytest.raises(ValueError):
        validate_feature_contract({"feature_columns": [*columns[:-1], "x0"]})


def test_competing_risk_targets_are_mutually_exclusive():
    frame = pd.DataFrame(
        {
            "event": [
                "favorable_first",
                "adverse_first_or_conflict",
                "timeout",
            ],
            "execution_net_ev_12h": [0.02, -0.03, -0.005],
        }
    )
    out = derive_event_columns(frame, "event")
    assert out[["clean_first", "adverse_first", "timeout_event"]].sum(axis=1).eq(1).all()
    assert out["positive_net"].tolist() == [1, 0, 0]
    assert out["negative_net"].tolist() == [0, 1, 1]


def test_unknown_event_is_rejected():
    frame = pd.DataFrame(
        {"event": ["future_magic"], "execution_net_ev_12h": [np.nan]}
    )
    with pytest.raises(ValueError, match="unexpected event"):
        derive_event_columns(frame, "event")


def test_identity_normalizes_only_symbol_storage_and_timestamp():
    frame = pd.DataFrame(
        {
            "__symbol__": ["BTC/USD:USD"],
            "__ts__": ["2026-01-01T00:00:00Z"],
        }
    )
    out = normalize_identity(frame)
    assert out.loc[0, "__symbol__"] == "BTC_USD:USD"
    assert str(out.loc[0, "__ts__"].tz) == "UTC"


def test_candidate_relative_context_is_timestamp_side_local():
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01"] * 3, utc=True),
            "side_name": ["long"] * 3,
            "candidate_id": ["a", "b", "c"],
            "base_oof_score": [3.0, 2.0, 1.0],
        }
    )
    out = add_candidate_relative_context(frame)
    assert out["candidate_group_size"].tolist() == [3, 3, 3]
    assert out["base_rank_timestamp_side"].tolist() == [1.0, 2.0, 3.0]
    assert out["base_rank_pct_timestamp_side"].tolist() == [0.0, 0.5, 1.0]
    assert out["base_margin_to_candidate_cutoff"].tolist() == [2.0, 1.0, 0.0]
