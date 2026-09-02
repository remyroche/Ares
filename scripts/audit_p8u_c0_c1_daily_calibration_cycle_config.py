#!/usr/bin/env python3
"""Verify the hash-bound, no-order C0/C1 daily-calibration cycle config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_day(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if stamp != stamp.normalize():
        raise ValueError("daily state receipt is not bound to UTC midnight")
    return stamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-current-day", action="store_true")
    parser.add_argument("--out", type=Path, help="optional immutable JSON receipt")
    args = parser.parse_args()
    config_path = args.config.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "p8u-c0-c1-daily-calibration-cycle-config-v1":
        raise ValueError("unexpected calibration-cycle config schema")
    if payload.get("order_submission") is not False:
        raise ValueError("daily calibration config must be structurally no-order")
    artifacts = dict(payload.get("artifacts") or {})
    checks: dict[str, str] = {}
    for name in ("cycle_runner", "ledger_publisher", "candidate_builder", "exact_policy_materializer"):
        descriptor = dict(artifacts.get(name) or {})
        path = (ROOT / str(descriptor.get("path") or "")).resolve()
        expected = str(descriptor.get("sha256") or "")
        if not path.is_file() or not expected or _sha256(path) != expected:
            raise ValueError(f"{name} hash binding failed")
        checks[name] = expected
    for name, key in (("effective_ledger", "ledger_manifest_sha256"), ("daily_state", "state_manifest_sha256")):
        descriptor = dict(artifacts.get(name) or {})
        root = (ROOT / str(descriptor.get("root") or "")).resolve()
        manifest_name = "ledger_manifest.json" if name == "effective_ledger" else "state_manifest.json"
        manifest = root / manifest_name
        if not manifest.is_file() or _sha256(manifest) != str(descriptor.get(key) or ""):
            raise ValueError(f"{name} manifest hash binding failed")
        checks[name] = _sha256(manifest)
    for name in ("c0_package", "c1_package"):
        descriptor = dict(artifacts.get(name) or {})
        manifest = (ROOT / str(descriptor.get("path") or "") / "package_manifest.json").resolve()
        if not manifest.is_file() or _sha256(manifest) != str(descriptor.get("package_manifest_sha256") or ""):
            raise ValueError(f"{name} package manifest binding failed")
        checks[name] = _sha256(manifest)
    for name in ("frozen_source_manifest", "frozen_kraken_product_ledger", "frozen_rich_policy"):
        descriptor = dict(artifacts.get(name) or {})
        path = (ROOT / str(descriptor.get("path") or "")).resolve()
        if not path.is_file() or _sha256(path) != str(descriptor.get("sha256") or ""):
            raise ValueError(f"{name} hash binding failed")
        checks[name] = _sha256(path)
    state_root = (ROOT / str(artifacts["daily_state"]["root"])).resolve()
    latest_path = state_root / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    day = _utc_day(latest.get("decision_day"))
    receipt = state_root / str(latest.get("receipt_dir") or "") / "run_manifest.json"
    if not receipt.is_file() or _sha256(receipt) != str(latest.get("receipt_manifest_sha256") or ""):
        raise ValueError("daily state latest receipt binding failed")
    if args.require_current_day and day != pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError(f"daily state is stale: latest={day.isoformat()}")
    result = {
        "schema": "p8u-c0-c1-daily-calibration-cycle-config-audit-v1",
        "status": "PASS_NO_ORDER_DAILY_CALIBRATION_CONFIG",
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "latest_state_day": day.isoformat(), "latest_state_receipt_sha256": _sha256(receipt),
        "require_current_day": bool(args.require_current_day), "checks": checks,
        "authority": "no feature/model/map/portfolio/exchange/account/order authority",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        output = args.out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
