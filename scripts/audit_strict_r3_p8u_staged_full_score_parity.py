#!/usr/bin/env python3
"""Audit target-free staged P8U scores against the canonical full matrix.

The staged inference path intentionally carries only the four approved
incremental outputs directly.  This audit reconstructs the canonical
full-universe feature matrix from the same append-only source panel at each
source timestamp, scores it through the sealed stack, and compares every
Router/Router50/Base/Under/MC1 hand-off to the staged immutable receipt.

It is research-only: it accepts neither outcomes nor policy, portfolio,
exchange, or order-submission inputs.
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
from extreme_price_movements.inference.p8u_sealed_inference_stack import (  # noqa: E402
    P8USealedInferenceStack,
)
from extreme_price_movements.inference.p8u_staged_timestamp_executor import (  # noqa: E402
    DIRECT_EXPENSIVE_FEATURES,
)


SCHEMA = "strict_r3_p8u_staged_full_score_parity_v1"
FORBIDDEN_TOKENS = (
    "future_",
    "outcome",
    "policy_net",
    "label_available",
    "exact_net",
    "gross_net",
)
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


def _target_free(columns: list[str], *, name: str) -> None:
    forbidden = [
        column
        for column in columns
        if any(token in str(column).lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"{name} has forbidden outcome-like columns: {forbidden[:5]}")


def _causal_panel(panel: Mapping[str, Any], *, timestamp: pd.Timestamp) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in panel.items():
        if isinstance(value, pd.DataFrame):
            _target_free(value.columns.tolist(), name=f"source panel {name}")
            if not isinstance(value.index, pd.DatetimeIndex):
                raise ValueError(f"source panel {name} has no DatetimeIndex")
            if value.index.max() < timestamp:
                raise ValueError(f"source panel {name} does not reach {timestamp.isoformat()}")
            output[name] = value.loc[value.index <= timestamp].copy(deep=False)
        else:
            output[name] = value
    return output


def _full_matrix(
    *,
    candidates: pd.DataFrame,
    panel: Mapping[str, Any],
    symbols: tuple[str, ...],
    fields: tuple[str, ...],
    direct_reference: pd.DataFrame,
) -> pd.DataFrame:
    stamp = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise").iloc[0]
    # The selected four fields intentionally have a separate direct-state
    # implementation.  Asking the regular adapter to recreate them would be
    # a category error: its source-alignment contract deliberately leaves
    # some direct-only values unavailable.  The score reference below is the
    # production composition: direct-state outputs plus the regular canonical
    # vector path for every other sealed field.
    vector_fields = tuple(field for field in fields if field not in DIRECT_EXPENSIVE_FEATURES)
    generated = canonical_features_from_saved_panel(
        _causal_panel(panel, timestamp=stamp),
        universe_symbols=symbols,
        requested_features=vector_fields,
        full_config_causal_universe=False,
    )
    output = candidates.loc[:, list(IDENTITY_COLUMNS)].copy().reset_index(drop=True)
    direct = direct_reference.loc[:, [*IDENTITY_COLUMNS, *DIRECT_EXPENSIVE_FEATURES]].copy()
    if not _identity_equal(output, direct):
        raise ValueError("direct-state reference does not cover the complete target-free universe")
    direct = direct.set_index(list(IDENTITY_COLUMNS), verify_integrity=True).reindex(
        pd.MultiIndex.from_frame(output.loc[:, list(IDENTITY_COLUMNS)])
    )
    symbol_index = candidates["__symbol__"].astype(str)
    values: dict[str, np.ndarray] = {}
    for field in fields:
        if field in DIRECT_EXPENSIVE_FEATURES:
            values[field] = pd.to_numeric(direct[field], errors="coerce").to_numpy(np.float32)
            continue
        frame = generated.get(field)
        if not isinstance(frame, pd.DataFrame) or stamp not in frame.index:
            raise KeyError(f"canonical vectoriser did not materialise {field}")
        values[field] = frame.loc[stamp].reindex(symbol_index).to_numpy(np.float32)
    return pd.concat([output, pd.DataFrame(values)], axis=1)


def _identity_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    lhs = left.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    rhs = right.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    return lhs.equals(rhs)


def _compare(
    *,
    name: str,
    staged: pd.DataFrame,
    canonical: pd.DataFrame,
    fields: tuple[str, ...],
) -> tuple[list[dict[str, object]], bool]:
    if not _identity_equal(staged, canonical):
        return [{"component": name, "field": "__identity__", "rows": len(staged), "mismatch_rows": -1, "max_abs_delta": None}], False
    left = staged.set_index(list(IDENTITY_COLUMNS), verify_integrity=True)
    right = canonical.set_index(list(IDENTITY_COLUMNS), verify_integrity=True).reindex(left.index)
    rows: list[dict[str, object]] = []
    passed = True
    for field in fields:
        if field not in left.columns or field not in right.columns:
            rows.append({"component": name, "field": field, "rows": len(left), "mismatch_rows": -1, "max_abs_delta": None})
            passed = False
            continue
        observed = left[field]
        expected = right[field]
        if pd.api.types.is_bool_dtype(observed) or pd.api.types.is_bool_dtype(expected):
            mismatch = int((observed.fillna(False).astype(bool) != expected.fillna(False).astype(bool)).sum())
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
        rows.append({"component": name, "field": field, "rows": len(left), "mismatch_rows": mismatch, "max_abs_delta": maximum})
        passed = passed and mismatch == 0
    return rows, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--timestamps", nargs="*", help="Optional source timestamps; defaults to all staged commits.")
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"immutable parity root already exists: {args.out_root}")
    for path in (args.bundle, args.source_state, args.candidates):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.staged_root.is_dir():
        raise NotADirectoryError(args.staged_root)

    state = joblib.load(args.source_state)
    panel = state.get("panel") if isinstance(state, Mapping) else None
    symbols = tuple(map(str, state.get("symbols") or ())) if isinstance(state, Mapping) else ()
    if not isinstance(panel, Mapping) or len(symbols) != 160:
        raise ValueError("parity audit requires a frozen 160-symbol append-only source panel")
    candidates = pd.read_parquet(args.candidates)
    _target_free(candidates.columns.tolist(), name="target-free candidates")
    for column in ("__ts__", "__decision_ts__"):
        candidates[column] = pd.to_datetime(candidates[column], utc=True, errors="raise")
    required = {*IDENTITY_COLUMNS, "__symbol__", "__ts__", "__decision_ts__"}
    if missing := sorted(required.difference(candidates.columns)):
        raise ValueError(f"candidate rows miss {missing}")
    stack = P8USealedInferenceStack.load(args.bundle, root=ROOT)
    plan = stack.preproduction.feature_plan()
    full_fields = tuple(dict.fromkeys((*plan.router_features, *plan.base_features, *plan.under_features)))
    selected = (
        tuple(pd.Timestamp(value).tz_localize("UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC") for value in args.timestamps)
        if args.timestamps else tuple(sorted(pd.DatetimeIndex(candidates["__ts__"].unique())))
    )
    if not selected:
        raise ValueError("no parity timestamps selected")

    started = time.monotonic()
    report_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    all_pass = True
    args.out_root.mkdir(parents=True, exist_ok=False)
    for stamp in selected:
        commit = args.staged_root / "commits" / stamp.strftime("%Y%m%dT%H%M%SZ")
        if not commit.is_dir():
            raise FileNotFoundError(f"staged commit absent: {commit}")
        block = candidates.loc[candidates["__ts__"].eq(stamp)].copy()
        if len(block) != 160 or block["__symbol__"].nunique() != 160:
            raise ValueError(f"{stamp.isoformat()} does not have one target-free candidate per frozen symbol")
        staged_router_features = pd.read_parquet(commit / "router_features.parquet")
        staged_routed_features = pd.read_parquet(commit / "routed_features.parquet")
        staged_router = pd.read_parquet(commit / "router_scores.parquet")
        staged_scores = pd.read_parquet(commit / "routed_scores.parquet")
        canonical_matrix = _full_matrix(
            candidates=block,
            panel=panel,
            symbols=symbols,
            fields=full_fields,
            direct_reference=staged_router_features,
        )
        canonical = stack.score(canonical_matrix)
        canonical_router_features = canonical_matrix.loc[:, [*IDENTITY_COLUMNS, *plan.router_features]]
        canonical_router50 = canonical.router_population.loc[
            canonical.router_population["router50_eligible"].fillna(False).astype(bool)
        ].copy()
        routed_ids = canonical_router50.loc[:, list(IDENTITY_COLUMNS)]
        routed_feature_fields = tuple(dict.fromkeys((*plan.base_features, *plan.under_features)))
        canonical_routed_features = routed_ids.merge(
            canonical_matrix.loc[:, [*IDENTITY_COLUMNS, *routed_feature_fields]],
            on=list(IDENTITY_COLUMNS), how="left", validate="one_to_one",
        )
        pairs = (
            ("router_feature_matrix", staged_router_features, canonical_router_features, tuple(plan.router_features)),
            ("router50_gate", staged_router, canonical.router_population, ("router_score", "router50_eligible", "router_fraction", "router_timestamp_ordinal", "router_timestamp_count")),
            ("routed_feature_matrix", staged_routed_features, canonical_routed_features, routed_feature_fields),
            ("base_under_mc1", staged_scores, canonical.routed_scores, ("base_rank42", "conditional_consensus_rank", "ordinary_shadow_consensus_rank", "correctness_rank", "upstream", "final_score", "current_mc1_expected_bps", "bcf_mc1_expected_bps", "dual_mc1_admitted")),
        )
        stamp_pass = True
        for name, staged, expected, fields in pairs:
            rows, passed = _compare(name=name, staged=staged, canonical=expected, fields=fields)
            for row in rows:
                row["source_timestamp"] = stamp.isoformat()
            report_rows.extend(rows)
            stamp_pass = stamp_pass and passed
        all_pass = all_pass and stamp_pass
        checkpoint_rows.append({
            "source_timestamp": stamp.isoformat(),
            "decision_timestamp": (stamp + pd.Timedelta(hours=1)).isoformat(),
            "candidate_rows": len(block),
            "router50_rows": len(canonical_router50),
            "admitted_rows": len(canonical.admitted),
            "status": "pass" if stamp_pass else "fail",
        })
    audit = pd.DataFrame(report_rows)
    audit.to_parquet(args.out_root / "parity_by_component_field.parquet", index=False, compression="zstd")
    pd.DataFrame(checkpoint_rows).to_parquet(args.out_root / "checkpoint_summary.parquet", index=False, compression="zstd")
    receipt = {
        "schema": SCHEMA,
        "status": "pass" if all_pass else "fail",
        "bundle": str(args.bundle.resolve()),
        "bundle_sha256": _sha256(args.bundle),
        "source_state": str(args.source_state.resolve()),
        "source_state_sha256": _sha256(args.source_state),
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": _sha256(args.candidates),
        "staged_root": str(args.staged_root.resolve()),
        "timestamps": checkpoint_rows,
        "feature_count": len(full_fields),
        "direct_feature_groups": ["price_rv_7d_15d_robust_z", "liquidity_peer_residual", "orderbook_depth_24h_normalisation"],
        "direct_reference_contract": "sealed direct-state outputs; regular canonical vector path is used for all other fields",
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
