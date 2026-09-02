#!/usr/bin/env python3
"""Build or append the causal wide-panel state used by strict-R3 features.

This cache contains source primitives only; it never contains predictions or
outcomes.  Resume mode reads source files only for timestamps newer than the
sealed state, then appends them after exact overlap and symbol-contract checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    _add_frozen_input_backfill,
    _add_oi_funding_panels,
    _add_orderbook_panels,
    _make_panel,
    _panel_sidecar_quarantine_receipt,
    _raw_15m_quarantine_receipt,
)

STATE_SCHEMA = "strict_r3_causal_feature_panel_state_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _materialize_slice(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    bar_phase_minutes: int = 0,
) -> tuple[dict[str, pd.DataFrame], object]:
    panel, source_map = _make_panel(
        symbols,
        start,
        end,
        allow_minute_fallback=False,
        bar_phase_minutes=bar_phase_minutes,
    )
    index = panel["close"].index
    _add_orderbook_panels(panel, symbols, index, start, end)
    _add_oi_funding_panels(panel, symbols, index, start, end)
    _add_frozen_input_backfill(panel, symbols, index, start, end)
    return panel, source_map


def _append_panel(
    prior: dict[str, pd.DataFrame],
    update: dict[str, pd.DataFrame],
    *,
    symbols: list[str],
    preserve_sealed_overlap: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    output: dict[str, pd.DataFrame] = {}
    mutations: list[dict[str, object]] = []
    for field in sorted(set(prior).union(update)):
        left = prior.get(field)
        right = update.get(field)
        if not isinstance(left, pd.DataFrame):
            output[field] = right.copy() if isinstance(right, pd.DataFrame) else right
            continue
        if not isinstance(right, pd.DataFrame) or right.empty:
            output[field] = left.copy()
            continue
        left = left.reindex(columns=symbols)
        right = right.reindex(columns=symbols)
        overlap = left.index.intersection(right.index)
        if len(overlap):
            a = left.loc[overlap].to_numpy(float)
            b = right.loc[overlap].to_numpy(float)
            equal = np.isclose(a, b, atol=0.0, rtol=0.0, equal_nan=True)
            if not bool(equal.all()):
                changed = ~equal
                finite_delta = np.abs(a - b)[changed & np.isfinite(a) & np.isfinite(b)]
                mutations.append({
                    "field": str(field),
                    "overlap_rows": int(len(overlap)),
                    "changed_cells": int(changed.sum()),
                    "max_abs_delta": (
                        float(finite_delta.max()) if finite_delta.size else None
                    ),
                })
                if not preserve_sealed_overlap:
                    raise ValueError(f"causal panel overlap changed for field={field}")
        output[field] = pd.concat([
            left, right.loc[~right.index.isin(left.index)],
        ]).sort_index()
    return output, mutations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument(
        "--bar-phase-minutes", type=int, default=0, choices=(0, 15, 30, 45),
        help="Completed-H1 phase boundary; source state is phase-specific.",
    )
    parser.add_argument("--state-in", type=Path)
    parser.add_argument(
        "--preserve-sealed-overlap",
        action="store_true",
        help=(
            "Audit, but never overwrite, revisions to the last already-sealed "
            "source timestamp. This is the production causal append contract; "
            "without this flag any revision remains a hard failure."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable feature-panel state exists: {args.out_dir}")
    candidates = pd.read_parquet(args.candidates, columns=["__symbol__"])
    symbols = sorted(candidates["__symbol__"].astype(str).unique())
    start = _utc(args.history_start)
    end = _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("feature-panel end must be after history start")
    if int(start.minute) != int(args.bar_phase_minutes):
        raise ValueError("history-start minute must equal --bar-phase-minutes")
    if int(end.minute) != int(args.bar_phase_minutes):
        raise ValueError("end-exclusive minute must equal --bar-phase-minutes")

    parent_hash = None
    overlap_mutations: list[dict[str, object]] = []
    if args.state_in is None:
        panel, source_map = _materialize_slice(
            symbols, start, end, bar_phase_minutes=int(args.bar_phase_minutes),
        )
        loaded_start = start
        bootstrap = True
    else:
        prior = joblib.load(args.state_in)
        if prior.get("schema") != STATE_SCHEMA:
            raise ValueError("unsupported feature-panel state")
        if list(prior["symbols"]) != symbols:
            raise ValueError("feature-panel symbol contract changed")
        if int(prior.get("bar_phase_minutes", 0)) != int(args.bar_phase_minutes):
            raise ValueError("feature-panel resume cannot cross phase contracts")
        prior_end = _utc(prior["end_exclusive"])
        if end < prior_end:
            raise ValueError(
                "feature-panel resume cannot move end-exclusive backward"
            )
        if end == prior_end:
            # A seed whose panel already reaches the requested signal hour is
            # a valid first live/reconciliation input.  Preserve it exactly;
            # do not re-read or rebuild identical source history.
            loaded_start = end
            source_map = prior.get("source_map")
            panel = {
                name: value.copy() if isinstance(value, pd.DataFrame) else value
                for name, value in prior["panel"].items()
            }
        else:
            # Include the last sealed timestamp as a read-only overlap
            # sentinel. It detects source rewrites while avoiding a full
            # historical reload.
            loaded_start = prior_end - pd.Timedelta(hours=1)
            update, source_map = _materialize_slice(
                symbols,
                loaded_start,
                end,
                bar_phase_minutes=int(args.bar_phase_minutes),
            )
            panel, overlap_mutations = _append_panel(
                prior["panel"],
                update,
                symbols=symbols,
                preserve_sealed_overlap=bool(args.preserve_sealed_overlap),
            )
        parent_hash = _sha(args.state_in)
        bootstrap = False

    args.out_dir.mkdir(parents=True)
    state = {
        "schema": STATE_SCHEMA,
        "symbols": symbols,
        "history_start": start,
        "end_exclusive": end,
        "bar_phase_minutes": int(args.bar_phase_minutes),
        "parent_state_sha256": parent_hash,
        "source_map": source_map,
        "raw_15m_quarantine": _raw_15m_quarantine_receipt(),
        "panel_sidecar_quarantine": _panel_sidecar_quarantine_receipt(),
        "panel": panel,
    }
    state_path = args.out_dir / "feature_panel_state.joblib"
    joblib.dump(state, state_path, compress=3)
    close = panel["close"]
    receipt = {
        "schema": STATE_SCHEMA,
        "status": "pass",
        "bootstrap": bootstrap,
        "symbols": len(symbols),
        "panel_fields": len(panel),
        "panel_hours": len(close),
        "bar_phase_minutes": int(args.bar_phase_minutes),
        "raw_15m_quarantine": _raw_15m_quarantine_receipt(),
        "panel_sidecar_quarantine": _panel_sidecar_quarantine_receipt(),
        "history_start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "source_rows_loaded_from": loaded_start.isoformat(),
        "source_rows_loaded": int(max(0, (end - loaded_start) / pd.Timedelta(hours=1))),
        "parent_state_sha256": parent_hash,
        "state_sha256": _sha(state_path),
        "sealed_overlap_policy": (
            "preserve_and_audit" if args.preserve_sealed_overlap else "exact_or_fail"
        ),
        "sealed_overlap_mutation_fields": int(len(overlap_mutations)),
        "sealed_overlap_mutations": overlap_mutations,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
