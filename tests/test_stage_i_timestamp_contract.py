from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.stage_i_timestamp_contract import (
    attach_stage_i_decision_timestamp,
    resolve_stage_i_timestamp_contract,
)


def _ledger() -> pd.DataFrame:
    signal = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "__ts__": signal,
            "label_available_ts": signal + pd.Timedelta(hours=13),
        }
    )


def test_resolves_decision_from_signal_identity_without_mutating_signal() -> None:
    ledger = _ledger()
    signal_before = ledger["__ts__"].copy()
    timing = resolve_stage_i_timestamp_contract(ledger)
    expected_decision = signal_before + pd.Timedelta(hours=1)
    pd.testing.assert_series_equal(timing.signal_close, signal_before)
    pd.testing.assert_series_equal(
        timing.decision, expected_decision.rename("__ts__"), check_names=False
    )
    assert timing.label_available.equals(timing.decision + pd.Timedelta(hours=12))
    pd.testing.assert_series_equal(ledger["__ts__"], signal_before)
    assert "decision_ts" not in ledger
    assert timing.audit["selector_mda_hpo_map_timestamp_semantics"] == "decision_ts"


def test_attaches_explicit_decision_but_preserves_signal_identity() -> None:
    ledger = _ledger()
    attached = attach_stage_i_decision_timestamp(ledger)
    pd.testing.assert_series_equal(attached["__ts__"], ledger["__ts__"])
    assert attached["decision_ts"].equals(ledger["__ts__"] + pd.Timedelta(hours=1))


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda frame: frame.assign(
                decision_ts=frame["__ts__"] + pd.Timedelta(minutes=30)
            ),
            "decision_ts must equal",
        ),
        (
            lambda frame: frame.assign(
                label_available_ts=frame["__ts__"] + pd.Timedelta(hours=12)
            ),
            r"decision_ts \+ 12h",
        ),
    ],
)
def test_rejects_nonproduction_decision_or_label_horizon(mutator, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_stage_i_timestamp_contract(mutator(_ledger()))


def test_equal_twelve_hour_decision_horizon_is_exactly_accepted() -> None:
    ledger = _ledger()
    ledger["decision_ts"] = ledger["__ts__"] + pd.Timedelta(hours=1)
    ledger["__decision_ts__"] = ledger["decision_ts"]
    timing = resolve_stage_i_timestamp_contract(ledger)
    assert ((timing.label_available - timing.decision) == pd.Timedelta(hours=12)).all()
