#!/usr/bin/env python3
"""Audit fee-confirmed live execution-adjusted EV calibration.

The live execution boundary uses a prediction after the adverse-only price
delay and observable entry microstructure debits.  This tool deliberately
joins that immutable terminal prediction only to fee-confirmed realised net
outcomes.  Gross PnL, a policy-cost proxy, and pending fee data are never
silently treated as realised net performance.

It is strictly read-only with respect to trading and exchange state.  The
output path is immutable so calibration evidence cannot be rewritten after
newer fees arrive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_execution_adjusted_ev_calibration_v1"
BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<0", -np.inf, 0.0),
    ("0-25", 0.0, 25.0),
    ("25-50", 25.0, 50.0),
    ("50-75", 50.0, 75.0),
    ("75-100", 75.0, 100.0),
    ("100-150", 100.0, 150.0),
    (">150", 150.0, np.inf),
)
MIN_DIRECTIONAL_OBSERVATIONS = 20
MIN_DIRECTIONAL_POPULATED_BUCKETS = 3


def _finite(value: object) -> float | None:
    result = pd.to_numeric(value, errors="coerce")
    return float(result) if np.isfinite(result) else None


def _bucket(value: float) -> str:
    for name, lower, upper in BUCKETS:
        if lower <= value < upper:
            return name
    raise AssertionError(f"value does not belong to a calibration bucket: {value}")


def _records(ledger: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = ledger.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("close-notification ledger lacks records")
    return [row for row in records.values() if isinstance(row, Mapping)]


def _confirmed_sidecar_outcomes(
    sidecar: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Return only confirmed fee outcomes from an immutable sidecar.

    Close notifications are append-only.  Older entries cannot be rewritten
    when the exchange later exposes fees, so reconciliation is deliberately a
    separate immutable receipt.  Gross or estimated sidecar values are never
    accepted as a realised outcome.
    """
    if not isinstance(sidecar, Mapping):
        return {}
    if str(sidecar.get("schema") or "") != "strict_r3_fee_confirmed_execution_sidecar_v1":
        raise ValueError("unsupported fee-confirmed sidecar schema")
    rows = sidecar.get("rows")
    if not isinstance(rows, list):
        raise ValueError("fee-confirmed sidecar lacks rows")
    recovered: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "confirmed":
            continue
        record_key = str(row.get("record_key") or "")
        net_bps = _finite(row.get("net_bps"))
        if not record_key or net_bps is None:
            continue
        if record_key in recovered:
            raise ValueError(
                f"duplicate confirmed sidecar outcome for ledger key={record_key}"
            )
        recovered[record_key] = net_bps
    return recovered


def _confirmed_sidecar_predictions(
    sidecar: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Return exact persisted entry predictions from the recovery sidecar."""
    if not isinstance(sidecar, Mapping):
        return {}
    if str(sidecar.get("schema") or "") != "strict_r3_execution_prediction_recovery_sidecar_v1":
        raise ValueError("unsupported execution-prediction sidecar schema")
    rows = sidecar.get("rows")
    if not isinstance(rows, list):
        raise ValueError("execution-prediction sidecar lacks rows")
    recovered: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "confirmed":
            continue
        record_key = str(row.get("record_key") or "")
        predicted = _finite(row.get("execution_adjusted_expected_net_bps"))
        if not record_key or predicted is None:
            continue
        if record_key in recovered:
            raise ValueError(
                f"duplicate confirmed execution prediction for ledger key={record_key}"
            )
        recovered[record_key] = predicted
    return recovered


def audit(
    ledger: Mapping[str, Any], *, sidecar: Mapping[str, Any] | None = None,
    prediction_sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the requested predicted-adjusted-EV vs realised-net bucket map."""
    observed: list[dict[str, Any]] = []
    missing_prediction = 0
    prediction_available = 0
    fee_confirmed_outcome_available = 0
    fee_pending_or_missing_outcome = 0
    pending_or_missing_fee_for_prediction = 0
    recovered_outcomes = _confirmed_sidecar_outcomes(sidecar)
    recovered_predictions = _confirmed_sidecar_predictions(prediction_sidecar)
    records = ledger.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("close-notification ledger lacks records")
    for record_key, record in records.items():
        if not isinstance(record, Mapping):
            continue
        telemetry = record.get("trade_telemetry")
        if not isinstance(telemetry, Mapping):
            missing_prediction += 1
            continue
        prediction = telemetry.get("prediction")
        pnl = telemetry.get("pnl")
        direct_fees_confirmed = isinstance(pnl, Mapping) and pnl.get("fees_verified") is True
        direct_realised = _finite(pnl.get("net_bps")) if isinstance(pnl, Mapping) else None
        sidecar_realised = recovered_outcomes.get(str(record_key))
        fees_confirmed = direct_fees_confirmed and direct_realised is not None
        realised = direct_realised if fees_confirmed else sidecar_realised
        if sidecar_realised is not None:
            fees_confirmed = True
        if fees_confirmed and realised is not None:
            fee_confirmed_outcome_available += 1
        else:
            fee_pending_or_missing_outcome += 1
        direct_adjusted = (
            _finite(prediction.get("execution_adjusted_expected_net_bps"))
            if isinstance(prediction, Mapping) else None
        )
        adjusted = direct_adjusted if direct_adjusted is not None else recovered_predictions.get(str(record_key))
        if adjusted is None:
            missing_prediction += 1
            continue
        prediction_available += 1
        if not fees_confirmed or realised is None:
            pending_or_missing_fee_for_prediction += 1
            continue
        observed.append({
            "symbol": record.get("symbol"),
            "candidate_id": record.get("candidate_id"),
            "bucket": _bucket(adjusted),
            "predicted_adjusted_ev_bps": adjusted,
            "realised_net_bps": realised,
            "prediction_source": (
                "terminal_telemetry" if direct_adjusted is not None else "persisted_entry_receipt"
            ),
            "outcome_source": (
                "terminal_telemetry" if direct_fees_confirmed and direct_realised is not None
                else "fee_confirmed_sidecar"
            ),
        })

    frame = pd.DataFrame(observed)
    rows: list[dict[str, Any]] = []
    for name, _, _ in BUCKETS:
        subset = frame.loc[frame["bucket"].eq(name)] if not frame.empty else frame
        count = int(len(subset))
        rows.append({
            "bucket": name,
            "confirmed_trade_count": count,
            "mean_predicted_adjusted_ev_bps": (
                float(subset["predicted_adjusted_ev_bps"].mean()) if count else None
            ),
            "mean_realised_net_bps": (
                float(subset["realised_net_bps"].mean()) if count else None
            ),
            "mean_prediction_error_bps": (
                float((subset["realised_net_bps"] - subset["predicted_adjusted_ev_bps"]).mean())
                if count else None
            ),
            "positive_realised_fraction": (
                float((subset["realised_net_bps"] > 0.0).mean()) if count else None
            ),
        })
    populated = [row for row in rows if row["confirmed_trade_count"] > 0]
    realised_means = [float(row["mean_realised_net_bps"]) for row in populated]
    monotonic = (
        all(next_ >= current for current, next_ in zip(realised_means, realised_means[1:]))
        if len(realised_means) >= 2 else None
    )
    correlation = None
    if len(frame) >= 2 and frame["predicted_adjusted_ev_bps"].nunique() > 1:
        correlation = float(
            frame["predicted_adjusted_ev_bps"].corr(frame["realised_net_bps"])
        )
    sufficient_directional_sample = (
        len(frame) >= MIN_DIRECTIONAL_OBSERVATIONS
        and len(populated) >= MIN_DIRECTIONAL_POPULATED_BUCKETS
    )
    if not len(frame):
        status = "insufficient_confirmed_net_outcomes"
    elif not sufficient_directional_sample:
        status = "insufficient_sample_for_monotonicity_assessment"
    else:
        status = "complete"
    return {
        "schema": SCHEMA,
        "status": status,
        "contract": {
            "prediction": "execution_adjusted_expected_net_bps",
            "outcome": "fee-confirmed pnl.net_bps only",
            "buckets": [name for name, _, _ in BUCKETS],
            "gross_or_policy_proxy_substitution": "prohibited",
        },
        "ledger_records": len(_records(ledger)),
        "missing_execution_adjusted_prediction": missing_prediction,
        "execution_adjusted_prediction_available": prediction_available,
        "fee_confirmed_net_outcome_available": fee_confirmed_outcome_available,
        "fee_pending_or_missing_outcome": fee_pending_or_missing_outcome,
        "prediction_with_pending_or_missing_fee_confirmed_outcome": (
            pending_or_missing_fee_for_prediction
        ),
        "confirmed_observations": int(len(frame)),
        "monotonicity_assessment_minimums": {
            "confirmed_observations": MIN_DIRECTIONAL_OBSERVATIONS,
            "populated_buckets": MIN_DIRECTIONAL_POPULATED_BUCKETS,
        },
        "sufficient_sample_for_monotonicity_assessment": sufficient_directional_sample,
        "bucket_metrics": rows,
        "populated_bucket_means_monotonic_non_decreasing": monotonic,
        "row_level_pearson_correlation": correlation,
        "observations": observed,
        "fee_confirmed_sidecar_rows_used": len(recovered_outcomes),
        "execution_prediction_sidecar_rows_used": len(recovered_predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--fee-sidecar", type=Path, default=None)
    parser.add_argument("--prediction-sidecar", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("calibration receipt path must be immutable")
    sidecar = (
        json.loads(args.fee_sidecar.read_text(encoding="utf-8"))
        if args.fee_sidecar is not None
        else None
    )
    prediction_sidecar = (
        json.loads(args.prediction_sidecar.read_text(encoding="utf-8"))
        if args.prediction_sidecar is not None
        else None
    )
    result = audit(
        json.loads(args.ledger.read_text(encoding="utf-8")),
        sidecar=sidecar,
        prediction_sidecar=prediction_sidecar,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
