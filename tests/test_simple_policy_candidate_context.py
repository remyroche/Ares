import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.simple_policy_candidate_context import (
    EXECUTION_CONTEXT_COLUMNS,
    join_candidate_execution_context,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-01T00:00Z", "2026-05-01T01:00Z"]),
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "side": [1.0, -1.0],
            "payload": [11, 22],
        },
        index=[9, 4],
    )


def _source() -> pd.DataFrame:
    source = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-05-01T00:00Z", "2026-05-01T01:00Z"]),
            "timestamp": pd.to_datetime(["2026-05-01T00:00Z", "2026-05-01T01:00Z"]),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "side_name": ["long", "short"],
            "threshold_basis_reference_asof": [
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
            ],
            "threshold_basis_selected": [True, True],
            "threshold_basis_mapped_expected_ev_valid": [True, True],
            "threshold_basis_invalid_mapped_expected_ev_sentinel": [False, False],
        }
    )
    for offset, column in enumerate(EXECUTION_CONTEXT_COLUMNS):
        source[column] = np.array([0.1, 0.2]) + offset
    return source


def test_join_is_keyed_and_preserves_candidate_order_and_index() -> None:
    candidates = _candidates()
    source = _source().iloc[::-1].reset_index(drop=True)
    joined, audit = join_candidate_execution_context(candidates, source)

    assert joined.index.tolist() == [9, 4]
    assert joined["payload"].tolist() == [11, 22]
    assert joined[EXECUTION_CONTEXT_COLUMNS[0]].tolist() == pytest.approx([0.1, 0.2])
    assert joined["ev_rank_pct"].tolist() == pytest.approx([1.0, 1.0])
    assert audit.matched_rows == 2
    assert audit.exact_source_coverage
    assert not audit.source_positionally_aligned


@pytest.mark.parametrize("which", ["candidate", "source"])
def test_duplicate_trade_keys_are_rejected(which: str) -> None:
    candidates = _candidates()
    source = _source()
    if which == "candidate":
        candidates = pd.concat([candidates, candidates.iloc[[0]]])
    else:
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate trade keys"):
        join_candidate_execution_context(candidates, source)


def test_missing_and_extra_keys_are_rejected_under_exact_contract() -> None:
    with pytest.raises(ValueError, match="lack admitted context"):
        join_candidate_execution_context(_candidates(), _source().iloc[[0]])

    with pytest.raises(ValueError, match="extra trade keys"):
        join_candidate_execution_context(_candidates().iloc[[0]], _source())


def test_future_reference_and_conflicting_existing_context_are_rejected() -> None:
    source = _source()
    source.loc[0, "threshold_basis_reference_asof"] = "2026-05-01T00:01:00Z"
    with pytest.raises(ValueError, match="future corrected-EV reference"):
        join_candidate_execution_context(_candidates(), source)

    candidates = _candidates()
    candidates[EXECUTION_CONTEXT_COLUMNS[0]] = [999.0, 999.0]
    with pytest.raises(ValueError, match="conflicts with admitted execution ledger"):
        join_candidate_execution_context(candidates, _source())


def test_invalid_side_and_conflicting_static_aliases_are_rejected() -> None:
    candidates = _candidates()
    candidates.loc[9, "side"] = 0
    with pytest.raises(ValueError, match="invalid side values"):
        join_candidate_execution_context(candidates, _source())

    source = _source()
    source.loc[0, "symbol"] = "SOL/USD:USD"
    with pytest.raises(ValueError, match="conflicting symbol aliases"):
        join_candidate_execution_context(_candidates(), source)
