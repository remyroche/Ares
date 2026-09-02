#!/usr/bin/env python3
"""Audit a target-free canonical-vector feature slice against a frozen P8U row.

This is intentionally narrower than the full 175-column warm-graph audit.  It
is used while migrating an individual inexpensive feature family into the
Router-first timestamp path: the canonical vectoriser receives the complete
raw source panel and full frozen universe, then this tool projects only the
named source timestamp into the frozen candidate identities.

No outcomes, policy labels, scores, Router gate, or order-execution state are
accepted by this tool.  It therefore gives a durable receipt for a regular
vectorised feature without creating a second feature implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_canonical_feature_adapter import (  # noqa: E402
    canonical_features_from_saved_panel,
)


FORBIDDEN_TOKENS = ("outcome", "policy_net", "label_available", "exact_net", "gross_net")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _target_free(columns: list[str], *, name: str) -> None:
    forbidden = [
        column for column in columns
        if any(token in str(column).lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"{name} has forbidden outcome-like columns: {forbidden[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--signal-ts", required=True)
    parser.add_argument("--features", required=True, help="Comma-separated canonical feature names")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable vector-stage audit already exists: {args.out_dir}")

    fields = tuple(dict.fromkeys(item.strip() for item in args.features.split(",") if item.strip()))
    if not fields:
        raise ValueError("--features must contain at least one feature")
    _target_free(list(fields), name="requested features")
    stamp = _utc(args.signal_ts)
    manifest = json.loads(args.canonical_manifest.read_text())
    raw_universe = manifest.get("symbols")
    if not isinstance(raw_universe, list) or not raw_universe:
        raise ValueError("canonical manifest lacks frozen symbol universe")
    universe = tuple(sorted(dict.fromkeys(map(str, raw_universe))))

    identity = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]
    reference = pd.read_parquet(args.reference, columns=[*identity, *fields])
    _target_free(reference.columns.tolist(), name="reference")
    reference["__ts__"] = pd.to_datetime(reference["__ts__"], utc=True, errors="raise")
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True, errors="raise")
    expected = reference.loc[reference["__ts__"].eq(stamp)].copy()
    if expected.empty or expected.loc[:, identity].duplicated().any():
        raise ValueError("reference has no unique target-free rows for the signal timestamp")
    if not set(expected["__symbol__"].astype(str)).issubset(universe):
        raise ValueError("reference identity escapes frozen universe")

    state = joblib.load(args.source_panel)
    panel = state.get("panel") if isinstance(state, dict) else None
    if not isinstance(panel, dict):
        raise ValueError("source state lacks primitive panel")
    for name, frame in panel.items():
        if isinstance(frame, pd.DataFrame):
            _target_free(frame.columns.tolist(), name=f"source panel {name}")
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError(f"source panel {name} has no timestamp index")
            if frame.index.max() < stamp:
                raise ValueError(f"source panel {name} does not reach requested timestamp")

    started = time.monotonic()
    generated = canonical_features_from_saved_panel(
        panel,
        universe_symbols=universe,
        requested_features=fields,
        full_config_causal_universe=False,
    )
    runtime = time.monotonic() - started
    actual = expected.loc[:, identity].copy()
    for field in fields:
        frame = generated.get(field)
        if not isinstance(frame, pd.DataFrame) or stamp not in frame.index:
            raise KeyError(f"canonical vectoriser did not produce {field} at {stamp.isoformat()}")
        actual[field] = frame.loc[stamp].reindex(expected["__symbol__"].astype(str)).to_numpy(np.float32)

    expected = expected.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    actual = actual.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not actual.loc[:, identity].equals(expected.loc[:, identity]):
        raise AssertionError("canonical vector projection changed target-free identities")
    rows: list[dict[str, object]] = []
    for field in fields:
        observed = pd.to_numeric(actual[field], errors="coerce").to_numpy(float)
        frozen = pd.to_numeric(expected[field], errors="coerce").to_numpy(float)
        close = np.isclose(observed, frozen, rtol=1e-6, atol=1e-6, equal_nan=True)
        finite = np.isfinite(observed) & np.isfinite(frozen)
        rows.append({
            "feature": field,
            "rows": len(observed),
            "mismatch_rows": int((~close).sum()),
            "max_abs_delta": float(np.nanmax(np.abs(observed[finite] - frozen[finite]))) if finite.any() else 0.0,
        })
    audit = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    actual.to_parquet(args.out_dir / "vector_stage_features.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "feature_parity_by_field.parquet", index=False, compression="zstd")
    receipt = {
        "schema": "strict_r3_p8u_vector_stage_reference_audit_v1",
        "status": "pass" if int(audit["mismatch_rows"].sum()) == 0 else "fail",
        "signal_ts": stamp.isoformat(),
        "frozen_universe_symbols": len(universe),
        "reference_rows": len(expected),
        "features": list(fields),
        "mismatch_cells": int(audit["mismatch_rows"].sum()),
        "max_abs_delta": float(audit["max_abs_delta"].max()),
        "runtime_seconds": runtime,
        "source_panel_sha256": _sha256(args.source_panel),
        "canonical_manifest_sha256": _sha256(args.canonical_manifest),
        "reference_sha256": _sha256(args.reference),
        "outcome_columns_consumed": [],
        "router_or_downstream_gate_applied": False,
        "portfolio_or_execution_called": False,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
