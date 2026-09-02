#!/usr/bin/env python3
"""Materialise a target-free full-universe canonical P8U feature reference.

This is a narrow offline parity producer, not an inference path.  It creates
one immutable 175-field reference snapshot from the canonical vector graph so
that a newly bootstrapped warm vector state can be tested before it is wired
into the transactional Router-first scorer.  It never reads labels, outcomes,
policies, portfolio state, an exchange, or an order interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_canonical_feature_adapter import (  # noqa: E402
    canonical_features_from_saved_panel,
)
from extreme_price_movements.inference.p8u_production_contract import IDENTITY_COLUMNS  # noqa: E402


FORBIDDEN = ("future_", "outcome", "policy_net", "label_available", "exact_net", "gross_net")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _utc(raw: str) -> pd.Timestamp:
    stamp = pd.Timestamp(raw)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _target_free(frame: pd.DataFrame, *, name: str) -> None:
    forbidden = [
        str(column) for column in frame.columns
        if any(token in str(column).lower() for token in FORBIDDEN)
    ]
    if forbidden:
        raise ValueError(f"{name} is not target-free: {forbidden[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--signal-ts", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable canonical vector reference exists: {args.out_dir}")
    signal = _utc(args.signal_ts)
    plan = json.loads(args.feature_plan.read_text())
    fields = tuple(map(str, plan.get("full_union") or ()))
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("canonical vector reference requires a non-empty unique feature union")
    declared_count = int((plan.get("counts") or {}).get("full_union", len(fields)))
    if declared_count != len(fields):
        raise ValueError("feature plan full-union count does not match its declared contract")
    manifest = json.loads(args.canonical_manifest.read_text())
    symbols = tuple(sorted(dict.fromkeys(map(str, manifest.get("symbols") or ()))))
    if len(symbols) != 160:
        raise ValueError("canonical vector reference requires the frozen 160-symbol universe")
    candidates = pd.read_parquet(args.candidates)
    _target_free(candidates, name="target-free candidates")
    required = {*IDENTITY_COLUMNS, "__ts__", "__symbol__"}
    if missing := sorted(required.difference(candidates.columns)):
        raise ValueError(f"candidate input lacks {missing}")
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    candidates = candidates.loc[candidates["__ts__"].eq(signal)].copy()
    if len(candidates) != 160 or candidates["__symbol__"].astype(str).nunique() != 160:
        raise ValueError("canonical vector reference requires exactly one candidate per frozen symbol")
    if set(candidates["__symbol__"].astype(str)) != set(symbols):
        raise ValueError("canonical vector reference candidate universe mismatch")
    source = joblib.load(args.source_state)
    panel = source.get("panel") if isinstance(source, Mapping) else None
    if not isinstance(panel, Mapping) or not isinstance(panel.get("close"), pd.DataFrame):
        raise ValueError("source state lacks an append-only primitive close panel")
    close = panel["close"]
    if signal not in close.index:
        raise ValueError("source state lacks reference signal timestamp")
    causal_panel = {
        name: value.loc[value.index <= signal].copy(deep=False)
        if isinstance(value, pd.DataFrame) else value
        for name, value in panel.items()
    }
    started = time.perf_counter()
    generated = canonical_features_from_saved_panel(
        causal_panel,
        universe_symbols=symbols,
        requested_features=fields,
        full_config_causal_universe=True,
    )
    output = candidates.loc[
        :, [*IDENTITY_COLUMNS, "__ts__", "__symbol__"]
    ].copy()
    candidate_symbols = candidates["__symbol__"].astype(str)
    for field in fields:
        frame = generated.get(field)
        if not isinstance(frame, pd.DataFrame) or signal not in frame.index:
            raise KeyError(f"canonical graph did not materialise {field}")
        output[field] = frame.loc[signal].reindex(candidate_symbols).to_numpy(np.float32)
    args.out_dir.mkdir(parents=True)
    feature_path = args.out_dir / "reference_features.parquet"
    output.to_parquet(feature_path, index=False, compression="zstd")
    receipt = {
        "schema": "strict_r3_p8u_canonical_vector_reference_v1",
        "status": "pass_target_free",
        "signal_ts": signal.isoformat(),
        "candidate_rows": int(len(output)),
        "feature_fields": int(len(fields)),
        # Retain both spellings used by the two existing warm-state receipts.
        # The snapshot consumer accepts either form but always verifies the
        # actual feature-file SHA before it scores anything.
        "feature_path": str(feature_path.resolve()),
        "feature_sha256": _sha256(feature_path),
        "features": str(feature_path.resolve()),
        "features_sha256": _sha256(feature_path),
        "source_state": str(args.source_state.resolve()),
        "source_state_sha256": _sha256(args.source_state),
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": _sha256(args.candidates),
        "feature_plan_sha256": _sha256(args.feature_plan),
        "canonical_manifest_sha256": _sha256(args.canonical_manifest),
        "runtime_seconds": float(time.perf_counter() - started),
        "outcome_columns_consumed": [],
        "policy_or_portfolio_called": False,
        "exchange_or_order_submission_called": False,
    }
    _atomic_json(args.out_dir / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
