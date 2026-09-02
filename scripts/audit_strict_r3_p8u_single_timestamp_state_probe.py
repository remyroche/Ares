#!/usr/bin/env python3
"""Audit whether a P8U transform state can reproduce one canonical hour.

This is a research-only falsification tool.  It runs two private cloned-state
transactions for exactly one complete frozen-universe source hour:

* canonical reference: the historical 1,536-hour tail; and
* candidate executor input: the same persisted state plus only that one hour.

It never publishes state, scores models, reads outcomes, or invokes exchange
I/O.  A passing receipt is necessary evidence for a true single-timestamp
executor, but it is not itself an inference route.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import shutil
import sys
import tempfile
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
from extreme_price_movements.inference.p8u_canonical_warm_runtime import clone_tree  # noqa: E402
from extreme_price_movements.inference.p8u_production_contract import (  # noqa: E402
    P8UPreproductionBundle,
)


FORBIDDEN_TOKENS = ("outcome", "label", "target", "future", "policy_net", "net_bps")
SCHEMA = "strict_r3_p8u_single_timestamp_state_probe_v1"


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_source(path: Path) -> tuple[dict[str, Any], tuple[str, ...], pd.DatetimeIndex]:
    loaded = joblib.load(path)
    panel = loaded.get("panel") if isinstance(loaded, Mapping) else None
    symbols = tuple(map(str, loaded.get("symbols") or ())) if isinstance(loaded, Mapping) else ()
    if not isinstance(panel, Mapping) or not isinstance(panel.get("close"), pd.DataFrame):
        raise ValueError("source state lacks a primitive close panel")
    if len(symbols) != 160 or len(set(symbols)) != len(symbols):
        raise ValueError("probe requires the frozen 160-symbol source universe")
    forbidden = [name for name in panel if any(token in str(name).lower() for token in FORBIDDEN_TOKENS)]
    if forbidden:
        raise ValueError(f"source state is not target-free: {forbidden[:4]}")
    index = pd.DatetimeIndex(pd.to_datetime(panel["close"].index, utc=True, errors="raise"))
    if len(index) < 1536:
        raise ValueError("source state lacks the required 1,536-hour reference tail")
    return dict(panel), symbols, index


def _panel_window(
    panel: Mapping[str, Any],
    *,
    timestamp: pd.Timestamp,
    start: pd.Timestamp,
    synthetic_copy_last: bool,
) -> dict[str, Any]:
    """Project a full-universe raw panel to a probe interval.

    A production probe requires every raw frame to contain the requested
    complete source hour.  ``synthetic_copy_last`` exists solely to test state
    semantics beyond an immutable historical panel's final timestamp; receipts
    carrying it are never promotion evidence.
    """
    output: dict[str, Any] = {}
    for name, value in panel.items():
        if not isinstance(value, pd.DataFrame):
            output[name] = value
            continue
        frame = value.loc[(value.index >= start) & (value.index <= timestamp)].copy(deep=False)
        if timestamp not in frame.index:
            if not synthetic_copy_last:
                raise ValueError(f"source field {name} lacks exact probe timestamp {timestamp}")
            available = value.loc[value.index < timestamp]
            if available.empty:
                raise ValueError(f"source field {name} has no predecessor for synthetic probe")
            frame.loc[timestamp] = available.iloc[-1].to_numpy(copy=False)
        output[str(name)] = frame.sort_index()
    return output


def _compare(
    reference: Mapping[str, pd.DataFrame],
    single: Mapping[str, pd.DataFrame],
    *,
    fields: tuple[str, ...],
    symbols: tuple[str, ...],
    timestamp: pd.Timestamp,
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in fields:
        left = reference[field].loc[timestamp].reindex(symbols).to_numpy(float)
        right = single[field].loc[timestamp].reindex(symbols).to_numpy(float)
        equal = np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
        delta = np.abs(left - right)
        finite_delta = delta[np.isfinite(delta)]
        rows.append({
            "feature": field,
            "reference_finite": int(np.isfinite(left).sum()),
            "single_finite": int(np.isfinite(right).sum()),
            "mismatch_cells": int((~equal).sum()),
            "max_abs_delta": float(finite_delta.max()) if finite_delta.size else 0.0,
        })
    return pd.DataFrame(rows).sort_values("feature", kind="stable").reset_index(drop=True)


def _latest_rows(
    features: Mapping[str, pd.DataFrame],
    *,
    fields: tuple[str, ...],
    symbols: tuple[str, ...],
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Materialise only the requested timestamp before releasing a graph run."""
    return pd.DataFrame(
        {
            field: features[field].loc[timestamp].reindex(symbols).to_numpy(float)
            for field in fields
        },
        index=pd.Index(symbols, name="symbol"),
    )


def _compare_latest_rows(
    reference: pd.DataFrame,
    single: pd.DataFrame,
    *,
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in reference.columns:
        left = reference[field].to_numpy(float)
        right = single[field].to_numpy(float)
        equal = np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
        delta = np.abs(left - right)
        finite_delta = delta[np.isfinite(delta)]
        rows.append(
            {
                "feature": str(field),
                "reference_finite": int(np.isfinite(left).sum()),
                "single_finite": int(np.isfinite(right).sum()),
                "mismatch_cells": int((~equal).sum()),
                "max_abs_delta": float(finite_delta.max()) if finite_delta.size else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("feature", kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--state-scope", required=True)
    parser.add_argument("--state-as-of", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--synthetic-copy-last", action="store_true")
    parser.add_argument(
        "--single-reduced-closure",
        action="store_true",
        help=(
            "Use the canonical selected-feature dependency closure for the "
            "one-row transaction; the 1,536-hour reference remains broad."
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable probe output exists: {args.out_dir}")
    state_as_of = _utc(args.state_as_of)
    timestamp = _utc(args.timestamp)
    if timestamp != state_as_of + pd.Timedelta(hours=1):
        raise ValueError("probe timestamp must be exactly one hour after state-as-of")
    if not args.state_dir.is_dir():
        raise FileNotFoundError(args.state_dir)

    bundle = P8UPreproductionBundle.load(args.bundle, root=ROOT)
    bundle.verify_artifacts()
    fields = bundle.feature_plan().full_union
    panel, symbols, index = _load_source(args.source_state)
    if not args.synthetic_copy_last and timestamp not in index:
        raise ValueError("source state lacks the requested complete probe timestamp")

    args.out_dir.mkdir(parents=True)
    work_root = Path(tempfile.mkdtemp(prefix="p8u_single_timestamp_probe_", dir="/private/tmp"))
    log_path = args.out_dir / "canonical_engine.log"
    receipt_base = {
        "schema": SCHEMA,
        "purpose": "research-only stateful one-hour parity audit; not an inference fallback",
        "bundle": str(args.bundle),
        "bundle_sha256": _sha256(args.bundle),
        "source_state": str(args.source_state),
        "source_state_sha256": _sha256(args.source_state),
        "state_dir": str(args.state_dir),
        "state_scope": str(args.state_scope),
        "state_as_of": state_as_of.isoformat(),
        "timestamp": timestamp.isoformat(),
        "synthetic_copy_last": bool(args.synthetic_copy_last),
        "single_reduced_closure": bool(args.single_reduced_closure),
        "features": len(fields),
        "outcome_columns_consumed": [],
        "exchange_io": False,
        "order_submission": False,
    }
    _atomic_json(args.out_dir / "receipt.json", {**receipt_base, "status": "running"})
    try:
        reference_state, single_state = work_root / "reference_state", work_root / "single_state"
        clone_tree(args.state_dir, reference_state)
        clone_tree(args.state_dir, single_state)
        reference_panel = _panel_window(
            panel,
            timestamp=timestamp,
            start=timestamp - pd.Timedelta(hours=1535),
            synthetic_copy_last=bool(args.synthetic_copy_last),
        )
        single_panel = _panel_window(
            panel,
            timestamp=timestamp,
            start=timestamp,
            synthetic_copy_last=bool(args.synthetic_copy_last),
        )
        kwargs = {
            "universe_symbols": symbols,
            "requested_features": fields,
            "full_config_causal_universe": True,
            "state_scope": str(args.state_scope),
            "state_components": [
                "raw", "causal_transform", "derived", "nested", "oi_iqr",
                "fixed_ffd", "spectral", "grouped", "ewma", "regime_transition",
            ],
            "state_contract_features": fields,
        }
        with log_path.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            started = time.perf_counter()
            reference = canonical_features_from_saved_panel(
                reference_panel, state_dir=reference_state, **kwargs
            )
            reference_seconds = time.perf_counter() - started
            reference_latest = _latest_rows(
                reference, fields=fields, symbols=symbols, timestamp=timestamp
            )
            # The canonical graph can hold several gigabytes of intermediates.
            # Keep only the 160 x 175 comparison matrix before the second,
            # independent transaction.  This is a memory-isolation change only.
            del reference, reference_panel
            gc.collect()
            started = time.perf_counter()
            single = canonical_features_from_saved_panel(
                single_panel,
                state_dir=single_state,
                **{
                    **kwargs,
                    "full_config_causal_universe": not bool(args.single_reduced_closure),
                    "hvn_lvn_max_workers": 1,
                },
            )
            single_seconds = time.perf_counter() - started
            single_latest = _latest_rows(
                single, fields=fields, symbols=symbols, timestamp=timestamp
            )
            del single, single_panel
            gc.collect()
        comparison = _compare_latest_rows(
            reference_latest, single_latest, atol=float(args.atol), rtol=float(args.rtol)
        )
        comparison.to_parquet(args.out_dir / "field_comparison.parquet", index=False, compression="zstd")
        mismatch_fields = comparison.loc[comparison.mismatch_cells.gt(0)]
        receipt = {
            **receipt_base,
            "status": "pass" if mismatch_fields.empty else "fail",
            "reference_seconds": float(reference_seconds),
            "single_seconds": float(single_seconds),
            "single_available_fields": int(comparison.single_finite.gt(0).sum()),
            "mismatch_fields": int(len(mismatch_fields)),
            "mismatch_cells": int(mismatch_fields.mismatch_cells.sum()),
            "max_abs_delta": float(comparison.max_abs_delta.max()),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
        }
        _atomic_json(args.out_dir / "receipt.json", receipt)
        print(json.dumps(receipt, sort_keys=True))
    except Exception as error:
        _atomic_json(
            args.out_dir / "receipt.json",
            {
                **receipt_base,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
