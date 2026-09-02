#!/usr/bin/env python3
"""Compare two target-free P8U staged-score receipt chains exactly.

This is the inexpensive second leg of the regular-vector-state proof.  The
``control_root`` must already have passed the independent full-vector audit;
this script verifies that the state-consuming candidate chain emits the same
feature matrices, Router gate, Base/Under scores, and MC1 outputs without
rebuilding the broad historical feature graph again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_production_contract import IDENTITY_COLUMNS  # noqa: E402
from extreme_price_movements.inference.p8u_staged_timestamp_executor import (  # noqa: E402
    DIRECT_EXPENSIVE_FEATURES,
)


SCHEMA = "strict_r3_p8u_staged_commit_parity_v1"
ATOL = 1e-5
RTOL = 1e-6


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


def _identity_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    columns = list(IDENTITY_COLUMNS)
    lhs = left.loc[:, columns].sort_values(columns, kind="stable").reset_index(drop=True)
    rhs = right.loc[:, columns].sort_values(columns, kind="stable").reset_index(drop=True)
    return lhs.equals(rhs)


def _compare_frame(
    *,
    component: str,
    control: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[list[dict[str, object]], bool]:
    if not _identity_equal(control, candidate):
        return [
            {
                "component": component,
                "field": "__identity__",
                "rows": len(control),
                "mismatch_rows": -1,
                "max_abs_delta": None,
            }
        ], False
    left = control.set_index(list(IDENTITY_COLUMNS), verify_integrity=True)
    right = candidate.set_index(list(IDENTITY_COLUMNS), verify_integrity=True).reindex(left.index)
    fields = sorted(set(left.columns).union(right.columns))
    passed = True
    rows: list[dict[str, object]] = []
    for field in fields:
        if field not in left.columns or field not in right.columns:
            rows.append(
                {
                    "component": component,
                    "field": field,
                    "rows": len(left),
                    "mismatch_rows": -1,
                    "max_abs_delta": None,
                }
            )
            passed = False
            continue
        observed = left[field]
        expected = right[field]
        if pd.api.types.is_bool_dtype(observed) or pd.api.types.is_bool_dtype(expected):
            mismatch = int(
                (observed.fillna(False).astype(bool) != expected.fillna(False).astype(bool)).sum()
            )
            maximum = 0.0
        elif pd.api.types.is_numeric_dtype(observed) and pd.api.types.is_numeric_dtype(expected):
            actual = pd.to_numeric(observed, errors="coerce").to_numpy(float)
            reference = pd.to_numeric(expected, errors="coerce").to_numpy(float)
            close = np.isclose(actual, reference, atol=ATOL, rtol=RTOL, equal_nan=True)
            mismatch = int((~close).sum())
            finite = np.isfinite(actual) & np.isfinite(reference)
            maximum = float(np.abs(actual[finite] - reference[finite]).max()) if finite.any() else 0.0
        else:
            mismatch = int((observed.astype(str) != expected.astype(str)).sum())
            maximum = 0.0
        rows.append(
            {
                "component": component,
                "field": field,
                "rows": len(left),
                "mismatch_rows": mismatch,
                "max_abs_delta": maximum,
            }
        )
        passed = passed and mismatch == 0
    return rows, passed


def _read_receipt(root: Path, stamp: str) -> Mapping[str, Any]:
    path = root / "commits" / stamp / "receipt.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "pass_target_free_router_first_score":
        raise ValueError(f"{path} is not a successful target-free staged receipt")
    if payload.get("outcome_columns_consumed") or payload.get("policy_or_portfolio_called") or payload.get("exchange_or_order_submission_called"):
        raise ValueError(f"{path} violates target-free parity provenance")
    if tuple(payload.get("direct_features_consumed", ())) != DIRECT_EXPENSIVE_FEATURES:
        raise ValueError(f"{path} does not use the frozen four-field direct contract")
    return payload


def _read_validated_control(path: Path, *, control_root: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "pass" or int(payload.get("mismatch_cells", -1)) != 0:
        raise ValueError("control correctness receipt is not a zero-mismatch full-vector audit")
    reported_root = Path(str(payload.get("staged_root", ""))).resolve()
    if reported_root != control_root.resolve():
        raise ValueError("control correctness receipt names a different staged root")
    return payload


def _stamps(root: Path, requested: Iterable[str] | None) -> tuple[str, ...]:
    if requested:
        return tuple(requested)
    commits = root / "commits"
    return tuple(sorted(path.name for path in commits.iterdir() if path.is_dir() and path.name != "bootstrap"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--control-correctness-receipt", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--timestamps", nargs="*")
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(args.out_root)
    for path in (args.control_root, args.candidate_root):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if not args.control_correctness_receipt.is_file():
        raise FileNotFoundError(args.control_correctness_receipt)
    control_audit = _read_validated_control(
        args.control_correctness_receipt, control_root=args.control_root
    )
    stamps = _stamps(args.control_root, args.timestamps)
    if not stamps:
        raise ValueError("no staged commits selected")
    if stamps != _stamps(args.candidate_root, stamps):
        raise ValueError("candidate root does not contain the exact requested commits")

    started = time.monotonic()
    rows: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    all_pass = True
    for stamp in stamps:
        control_receipt = _read_receipt(args.control_root, stamp)
        candidate_receipt = _read_receipt(args.candidate_root, stamp)
        for field in ("source_timestamp", "decision_timestamp", "candidate_rows", "router50_rows"):
            if control_receipt.get(field) != candidate_receipt.get(field):
                raise ValueError(f"receipt mismatch for {stamp}: {field}")
        stamp_pass = True
        for component, filename in (
            ("router_feature_matrix", "router_features.parquet"),
            ("router_score_and_gate", "router_scores.parquet"),
            ("routed_feature_matrix", "routed_features.parquet"),
            ("base_under_mc1", "routed_scores.parquet"),
        ):
            control = pd.read_parquet(args.control_root / "commits" / stamp / filename)
            candidate = pd.read_parquet(args.candidate_root / "commits" / stamp / filename)
            result_rows, passed = _compare_frame(
                component=component, control=control, candidate=candidate
            )
            for row in result_rows:
                row["source_timestamp"] = str(control_receipt["source_timestamp"])
            rows.extend(result_rows)
            stamp_pass = stamp_pass and passed
        all_pass = all_pass and stamp_pass
        checkpoints.append(
            {
                "source_timestamp": control_receipt["source_timestamp"],
                "decision_timestamp": control_receipt["decision_timestamp"],
                "candidate_rows": control_receipt["candidate_rows"],
                "router50_rows": control_receipt["router50_rows"],
                "status": "pass" if stamp_pass else "fail",
            }
        )

    args.out_root.mkdir(parents=True, exist_ok=False)
    audit = pd.DataFrame(rows)
    audit.to_parquet(args.out_root / "parity_by_component_field.parquet", index=False, compression="zstd")
    pd.DataFrame(checkpoints).to_parquet(args.out_root / "checkpoint_summary.parquet", index=False, compression="zstd")
    receipt = {
        "schema": SCHEMA,
        "status": "pass" if all_pass else "fail",
        "control_root": str(args.control_root.resolve()),
        "control_correctness_receipt": str(args.control_correctness_receipt.resolve()),
        "control_correctness_receipt_sha256": _sha256(args.control_correctness_receipt),
        "control_full_vector_feature_count": control_audit.get("feature_count"),
        "candidate_root": str(args.candidate_root.resolve()),
        "timestamps": checkpoints,
        "direct_feature_count": len(DIRECT_EXPENSIVE_FEATURES),
        "direct_features": list(DIRECT_EXPENSIVE_FEATURES),
        "regular_vector_feature_count": 171,
        "numeric_tolerance": {"atol": ATOL, "rtol": RTOL},
        "mismatch_cells": int(audit["mismatch_rows"].clip(lower=0).sum()) if not audit.empty else 0,
        "runtime_seconds": time.monotonic() - started,
        "outcome_columns_consumed": [],
        "policy_or_portfolio_called": False,
        "exchange_or_order_submission_called": False,
    }
    _atomic_json(args.out_root / "correctness_report.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
