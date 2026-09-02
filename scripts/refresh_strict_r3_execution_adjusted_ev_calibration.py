#!/usr/bin/env python3
"""Write one immutable, read-only execution-adjusted EV calibration receipt.

This is intentionally an observer, not an account reconciler.  It never
calls an exchange, changes a live state file, or rebuilds an outcome.  It
recomputes the requested prediction-bucket map only from:

* the append-only terminal close-notification ledger;
* an immutable fee-confirmed outcome sidecar, when available; and
* an immutable recovered-entry-prediction sidecar, when available.

Fee-pending, gross-only, and missing-prediction rows remain explicitly
excluded.  Every run creates a new timestamped receipt so new fee evidence
cannot rewrite prior calibration history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_strict_r3_execution_adjusted_ev_calibration import audit


SCHEMA = "strict_r3_execution_adjusted_ev_calibration_refresh_v1"
DEFAULT_LEDGER = ROOT / "data_perp/live/strict_r3_trade_close_notification_ledger_v1.json"
DEFAULT_FEE_SIDECAR = ROOT / "data_perp/artifacts/strict_r3_fee_confirmed_execution_sidecar_20260901_v1/sidecar.json"
DEFAULT_PREDICTION_SIDECAR = ROOT / "data_perp/artifacts/strict_r3_execution_prediction_recovery_20260901_v1/sidecar.json"
DEFAULT_OUT_ROOT = ROOT / "data_perp/artifacts/strict_r3_execution_adjusted_ev_calibration_refreshes"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_optional(path: Path | None) -> tuple[Mapping[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    resolved = path.resolve()
    if not resolved.is_file():
        return None, {
            "path": str(resolved),
            "status": "unavailable",
            "reason": "file_missing",
        }
    return json.loads(resolved.read_text(encoding="utf-8")), {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "status": "loaded",
    }


def refresh(
    *,
    ledger_path: Path,
    fee_sidecar_path: Path | None,
    prediction_sidecar_path: Path | None,
    out_root: Path,
    as_of: pd.Timestamp,
) -> Path:
    """Read immutable sources and write a new immutable calibration receipt."""
    ledger_path = ledger_path.resolve()
    if not ledger_path.is_file():
        raise FileNotFoundError(f"terminal close ledger is missing: {ledger_path}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    fee_sidecar, fee_source = _read_optional(fee_sidecar_path)
    prediction_sidecar, prediction_source = _read_optional(prediction_sidecar_path)
    result = audit(
        ledger,
        sidecar=fee_sidecar,
        prediction_sidecar=prediction_sidecar,
    )
    stamp = pd.Timestamp(as_of)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    run = out_root.resolve() / f"refresh_{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    if run.exists():
        raise FileExistsError(f"calibration refresh receipt already exists: {run}")
    run.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": SCHEMA,
        "as_of_ts": stamp.isoformat(),
        "scope": "read-only calibration observer; no exchange I/O, state mutation, feature scoring, admission, portfolio, or order submission",
        "sources": {
            "terminal_close_ledger": {
                "path": str(ledger_path),
                "sha256": _sha256(ledger_path),
                "status": "loaded",
            },
            "fee_confirmed_sidecar": fee_source,
            "execution_prediction_sidecar": prediction_source,
        },
        "calibration": result,
    }
    receipt = run / "receipt.json"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--fee-sidecar", type=Path, default=DEFAULT_FEE_SIDECAR)
    parser.add_argument("--prediction-sidecar", type=Path, default=DEFAULT_PREDICTION_SIDECAR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--as-of-utc",
        type=str,
        default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    args = parser.parse_args()
    receipt = refresh(
        ledger_path=args.ledger,
        fee_sidecar_path=args.fee_sidecar,
        prediction_sidecar_path=args.prediction_sidecar,
        out_root=args.out_root,
        as_of=pd.Timestamp(args.as_of_utc),
    )
    print(receipt)


if __name__ == "__main__":
    main()
