from scripts.audit_strict_r3_execution_adjusted_ev_calibration import audit


def _record(*, adjusted: float | None, net: float | None, verified: bool) -> dict:
    prediction = {} if adjusted is None else {
        "execution_adjusted_expected_net_bps": adjusted,
    }
    return {
        "candidate_id": "candidate",
        "symbol": "ABC/USD:USD",
        "trade_telemetry": {
            "prediction": prediction,
            "pnl": {"net_bps": net, "fees_verified": verified},
        },
    }


def test_fee_confirmed_calibration_uses_only_complete_prediction_and_outcome() -> None:
    result = audit({"records": {
        "a": _record(adjusted=10.0, net=5.0, verified=True),
        "b": _record(adjusted=80.0, net=100.0, verified=True),
        "c": _record(adjusted=120.0, net=-20.0, verified=False),
        "d": _record(adjusted=None, net=50.0, verified=True),
    }})

    assert result["status"] == "insufficient_sample_for_monotonicity_assessment"
    assert result["confirmed_observations"] == 2
    assert result["execution_adjusted_prediction_available"] == 3
    assert result["fee_confirmed_net_outcome_available"] == 3
    assert result["fee_pending_or_missing_outcome"] == 1
    assert result["prediction_with_pending_or_missing_fee_confirmed_outcome"] == 1
    assert result["missing_execution_adjusted_prediction"] == 1
    by_bucket = {row["bucket"]: row for row in result["bucket_metrics"]}
    assert by_bucket["0-25"]["mean_realised_net_bps"] == 5.0
    assert by_bucket["75-100"]["mean_realised_net_bps"] == 100.0
    assert result["populated_bucket_means_monotonic_non_decreasing"] is True
    assert result["sufficient_sample_for_monotonicity_assessment"] is False


def test_calibration_refuses_to_claim_monotonicity_without_confirmed_outcomes() -> None:
    result = audit({"records": {
        "a": _record(adjusted=80.0, net=None, verified=False),
    }})

    assert result["status"] == "insufficient_confirmed_net_outcomes"
    assert result["confirmed_observations"] == 0
    assert result["populated_bucket_means_monotonic_non_decreasing"] is None


def test_fee_confirmed_sidecar_can_recover_immutable_legacy_outcome() -> None:
    ledger = {"records": {
        "a": _record(adjusted=80.0, net=None, verified=False),
    }}
    sidecar = {
        "schema": "strict_r3_fee_confirmed_execution_sidecar_v1",
        "rows": [{"record_key": "a", "status": "confirmed", "net_bps": 90.0}],
    }
    result = audit(ledger, sidecar=sidecar)
    assert result["confirmed_observations"] == 1
    assert result["fee_confirmed_sidecar_rows_used"] == 1
    bucket = {row["bucket"]: row for row in result["bucket_metrics"]}["75-100"]
    assert bucket["mean_realised_net_bps"] == 90.0


def test_prediction_sidecar_can_recover_immutable_entry_economics() -> None:
    ledger = {"records": {
        "a": _record(adjusted=None, net=None, verified=False),
    }}
    fee_sidecar = {
        "schema": "strict_r3_fee_confirmed_execution_sidecar_v1",
        "rows": [{"record_key": "a", "status": "confirmed", "net_bps": 90.0}],
    }
    prediction_sidecar = {
        "schema": "strict_r3_execution_prediction_recovery_sidecar_v1",
        "rows": [{
            "record_key": "a", "status": "confirmed",
            "execution_adjusted_expected_net_bps": 80.0,
        }],
    }
    result = audit(
        ledger,
        sidecar=fee_sidecar,
        prediction_sidecar=prediction_sidecar,
    )
    assert result["confirmed_observations"] == 1
    assert result["execution_prediction_sidecar_rows_used"] == 1
    observation = result["observations"][0]
    assert observation["prediction_source"] == "persisted_entry_receipt"
    assert observation["outcome_source"] == "fee_confirmed_sidecar"
