import pandas as pd
import pytest

from scripts.audit_stage_d_oi_funding_lineage import build_ledger, validate_ledger


def _ledger() -> pd.DataFrame:
    return build_ledger()


def test_observation_and_availability_are_required_for_admission() -> None:
    ledger = _ledger()
    ledger.loc[ledger.source_id.eq("oi_kraken_analytics"), "disposition"] = "ADMITTED_CAUSAL"
    with pytest.raises(ValueError, match="missing_observation_or_availability"):
        validate_ledger(ledger)


def test_no_unbounded_oi_or_funding_forward_fill() -> None:
    ledger = _ledger()
    ledger.loc[ledger.source_id.eq("oi_hourly_sidecars"), "disposition"] = "ADMITTED_CAUSAL"
    with pytest.raises(ValueError, match="unbounded_staleness"):
        validate_ledger(ledger)


def test_pf_pi_products_must_remain_separate() -> None:
    ledger = _ledger()
    ledger.loc[ledger.source_id.eq("funding_official_export_pi"), "disposition"] = "REJECTED_NO_AVAILABILITY_TIMESTAMP"
    with pytest.raises(ValueError, match="inverse_funding_product_not_rejected"):
        validate_ledger(ledger)


def test_future_funding_inputs_are_forbidden() -> None:
    ledger = _ledger()
    ledger.loc[ledger.source_id.eq("funding_live_ticker"), "future_funding_safe"] = True
    with pytest.raises(ValueError, match="future_safe"):
        validate_ledger(ledger)


def test_live_parity_is_required_for_admission() -> None:
    ledger = _ledger()
    mask = ledger.source_id.eq("oi_kraken_analytics")
    ledger.loc[mask, ["observation_timestamp", "availability_timestamp"]] = "source_proven"
    ledger.loc[mask, ["bounded_staleness", "product_separated"]] = True
    ledger.loc[mask, "live_parity"] = False
    ledger.loc[mask, "disposition"] = "ADMITTED_CAUSAL"
    with pytest.raises(ValueError, match="lacks_live_parity"):
        validate_ledger(ledger)


def test_reference_funding_archive_has_own_disposition() -> None:
    ledger = _ledger()
    row = ledger.loc[ledger.source_id.eq("funding_reference_export_copy")]
    assert len(row) == 1
    assert row.iloc[0].disposition == "REJECTED_NO_AVAILABILITY_TIMESTAMP"
    assert "byte_identical_to_raw=True" in row.iloc[0].evidence
