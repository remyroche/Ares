#!/usr/bin/env python3
"""Measure exact P8U state-graph parity for one completed timestamp.

This is a research-only diagnostic for the migration from a retained raw tail
to a true snapshot executor.  It runs the sealed canonical graph twice in
private copy-on-write transactions:

* the current retained-tail control; and
* a compact ``previous raw snapshot + current raw snapshot`` path.

It writes the complete per-field delta inventory.  A compact result is never
published to a scoring bundle by this tool; its only purpose is to identify
which upstream state operators must be made exact before a single-timestamp
executor can be promoted.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
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
from extreme_price_movements.inference.p8u_canonical_warm_runtime import (  # noqa: E402
    clone_tree,
)


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _panel_rows(panel: dict[str, object], index: pd.DatetimeIndex) -> dict[str, object]:
    return {
        name: frame.loc[index].copy()
        if isinstance(frame, pd.DataFrame)
        else frame
        for name, frame in panel.items()
    }


def _run(
    *,
    panel: dict[str, object],
    symbols: pd.Index,
    fields: tuple[str, ...],
    state: Path,
    state_scope: str,
    full_config_causal_universe: bool,
    transaction: Path,
    signal_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, float]:
    private_state = transaction / "state"
    clone_tree(state, private_state)
    started = time.perf_counter()
    # The canonical engine is intentionally noisy; all material audit evidence
    # is returned below in the field-level result rather than terminal logs.
    with (transaction / "engine.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        generated = canonical_features_from_saved_panel(
            panel,
            universe_symbols=symbols,
            requested_features=fields,
            full_config_causal_universe=full_config_causal_universe,
            state_dir=private_state,
            state_scope=state_scope,
        )
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(
        {
            field: generated[field].loc[signal_ts].reindex(symbols).to_numpy(np.float32)
            for field in fields
        },
        index=symbols,
    )
    return frame, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--predecessor-state", type=Path, required=True)
    parser.add_argument("--state-scope", required=True)
    parser.add_argument("--signal-ts", required=True)
    parser.add_argument("--tail-hours", type=int, default=1536)
    parser.add_argument(
        "--reference-features",
        type=Path,
        help="Optional immutable canonical rows for the signal timestamp; avoids a second batch control run.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.tail_hours < 2:
        raise ValueError("tail-hours must be at least two")
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    source = joblib.load(args.source_state)
    panel = source.get("panel") if isinstance(source, dict) else None
    if not isinstance(panel, dict) or not isinstance(panel.get("close"), pd.DataFrame):
        raise ValueError("source state lacks a primitive close panel")
    symbols = panel["close"].columns
    signal = _utc(args.signal_ts)
    close = panel["close"]
    if signal not in close.index:
        raise ValueError("source state lacks requested signal timestamp")
    expected = pd.date_range(
        signal - pd.Timedelta(hours=args.tail_hours - 1), signal, freq="h", tz="UTC"
    )
    if args.reference_features is None and not expected.isin(close.index).all():
        raise ValueError("source state lacks the retained-tail control inputs")
    fields = tuple(map(str, json.loads(args.feature_plan.read_text())["full_union"]))
    if len(fields) != 175:
        raise ValueError("single-timestamp probe requires the sealed 175-field plan")
    predecessor = args.predecessor_state.resolve()
    if not predecessor.is_dir():
        raise NotADirectoryError(predecessor)
    out.mkdir(parents=True)
    reference_symbols = symbols
    if args.reference_features is None:
        full, full_seconds = _run(
            panel=_panel_rows(panel, expected),
            symbols=symbols,
            fields=fields,
            state=predecessor,
            state_scope=args.state_scope,
            full_config_causal_universe=True,
            transaction=out / "full_tail_transaction",
            signal_ts=signal,
        )
    else:
        reference = pd.read_parquet(args.reference_features)
        required = {"__ts__", "__symbol__", *fields}
        missing = sorted(required.difference(reference.columns))
        if missing:
            raise ValueError(f"reference features lack required columns: {missing[:5]}")
        mask = pd.to_datetime(reference["__ts__"], utc=True, errors="raise").eq(signal)
        reference = reference.loc[mask].copy()
        if reference.empty or reference["__symbol__"].astype(str).duplicated().any():
            raise ValueError("reference features do not provide unique signal-timestamp identities")
        reference_symbols = pd.Index(reference["__symbol__"].astype(str))
        full = reference.assign(__symbol__=reference["__symbol__"].astype(str)).set_index("__symbol__").loc[:, list(fields)]
        full_seconds = float("nan")
    compact_index = pd.DatetimeIndex([signal - pd.Timedelta(hours=1), signal])
    compact, compact_seconds = _run(
        panel=_panel_rows(panel, compact_index),
        symbols=symbols,
        fields=fields,
        state=predecessor,
        state_scope=args.state_scope,
        full_config_causal_universe=True,
        transaction=out / "two_snapshot_transaction",
        signal_ts=signal,
    )
    expected_values = full.reindex(reference_symbols).to_numpy(dtype=float)
    observed_values = compact.reindex(reference_symbols).to_numpy(dtype=float)
    both = np.isfinite(expected_values) & np.isfinite(observed_values)
    difference = np.abs(expected_values - observed_values)
    records: list[dict[str, object]] = []
    for position, field in enumerate(fields):
        expected_finite = np.isfinite(expected_values[:, position])
        observed_finite = np.isfinite(observed_values[:, position])
        numeric_mismatch = (difference[:, position] > 1e-5) & both[:, position]
        missing_mismatch = expected_finite & ~observed_finite
        records.append(
            {
                "field": field,
                "expected_finite": int(expected_finite.sum()),
                "snapshot_finite": int(observed_finite.sum()),
                "numeric_mismatch_cells": int(numeric_mismatch.sum()),
                "missing_vs_control_cells": int(missing_mismatch.sum()),
                "max_abs_delta": float(np.nanmax(difference[:, position]))
                if bool(both[:, position].any())
                else np.nan,
                "status": "pass"
                if not bool(numeric_mismatch.any() or missing_mismatch.any())
                else "requires_stateful_operator",
            }
        )
    per_field = pd.DataFrame(records).sort_values(["status", "field"], kind="stable")
    per_field.to_parquet(out / "per_field_parity.parquet", index=False)
    summary = {
        "schema": "strict_r3_p8u_single_timestamp_graph_probe_v1",
        "status": "pass" if per_field.status.eq("pass").all() else "incomplete",
        "signal_ts": signal.isoformat(),
        "symbols": int(len(symbols)),
        "fields": len(fields),
        "full_tail_hours": int(args.tail_hours),
        "full_tail_seconds": full_seconds,
        "two_snapshot_seconds": compact_seconds,
        "passing_fields": int(per_field.status.eq("pass").sum()),
        "failing_fields": int(per_field.status.ne("pass").sum()),
        "numeric_mismatch_cells": int(per_field.numeric_mismatch_cells.sum()),
        "missing_vs_control_cells": int(per_field.missing_vs_control_cells.sum()),
        "outcome_columns_consumed": [],
        "reference_mode": "persisted_canonical" if args.reference_features else "fresh_batch_control",
        "promotion_allowed": False,
    }
    _atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
