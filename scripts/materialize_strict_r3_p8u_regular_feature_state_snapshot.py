#!/usr/bin/env python3
"""Append one target-free P8U regular-vector feature-state snapshot.

This is the producer half of the P8U single-timestamp path.  It advances the
persisted canonical vector state in a private immutable transaction and emits
only the 171 fields *not* owned by the four-field direct executor.  The score
process subsequently reads the resulting parquet snapshot and never invokes a
batch feature graph itself.

The normal retained-tail mode is kept only as a reference/recovery path.  The
state-only mode accepts exactly one completed primitive hour and advances its
saved transform state; it never rebuilds a 1,536-hour raw panel.  The command
has no model scoring, policy, portfolio, exchange, or order submission
capability.
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
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_canonical_warm_runtime import (  # noqa: E402
    clone_tree,
)
from extreme_price_movements.inference.p8u_production_contract import (  # noqa: E402
    IDENTITY_COLUMNS,
)
from extreme_price_movements.inference.p8u_regular_state_forward import (  # noqa: E402
    _advance_perp_tail_supplement,
    _persisted_state_contract_features,
)
from extreme_price_movements.inference.p8u_router_first_vectorized import (  # noqa: E402
    P8URouterFirstFeaturePlan,
    P8URouterFirstVectorizedStage,
)
from extreme_price_movements.inference.p8u_sealed_inference_stack import (  # noqa: E402
    P8USealedInferenceStack,
)
from extreme_price_movements.inference.p8u_staged_timestamp_executor import (  # noqa: E402
    DIRECT_EXPENSIVE_FEATURES,
)


SCHEMA = "strict_r3_p8u_regular_feature_state_snapshot_v1"
_FORBIDDEN = ("future_", "outcome", "policy_net", "label_available", "exact_net", "gross_net")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _utc(raw: str) -> pd.Timestamp:
    value = pd.Timestamp(raw)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _target_free(frame: pd.DataFrame) -> None:
    forbidden = [
        str(column) for column in frame.columns
        if any(token in str(column).lower() for token in _FORBIDDEN)
    ]
    if forbidden:
        raise ValueError(f"regular state candidates are not target-free: {forbidden[:5]}")


def _tail(panel: Mapping[str, Any], *, signal: pd.Timestamp, hours: int) -> dict[str, Any]:
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or signal not in close.index:
        raise ValueError("regular state source panel lacks its signal close row")
    start = signal - pd.Timedelta(hours=hours - 1)
    expected = pd.date_range(start, signal, freq="h", tz="UTC")
    observed = pd.DatetimeIndex(close.index[(close.index >= start) & (close.index <= signal)])
    if not observed.equals(expected):
        raise ValueError("regular state source panel lacks the exact retained hourly tail")
    return {
        name: (
            value.loc[(value.index >= start) & (value.index <= signal)].copy(deep=False)
            if isinstance(value, pd.DataFrame)
            else value
        )
        for name, value in panel.items()
    }


def _one_row(panel: Mapping[str, Any], *, signal: pd.Timestamp) -> dict[str, Any]:
    """Return one complete primitive hour for the saved-state executor.

    The canonical feature adapter owns all rolling history through the cloned
    transform state.  Passing an accidental retained panel here would defeat
    the single-timestamp contract, so every DataFrame is constrained to the
    exact completed source hour.
    """

    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or signal not in close.index:
        raise ValueError("regular state source panel lacks its signal close row")
    output: dict[str, Any] = {}
    for name, value in panel.items():
        if not isinstance(value, pd.DataFrame):
            output[name] = value
            continue
        if signal not in value.index:
            raise ValueError(f"regular state source field lacks the signal row: {name}")
        row = value.loc[[signal]].copy(deep=False)
        if len(row) != 1:
            raise AssertionError(f"regular state source field is not one row: {name}")
        output[str(name)] = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--signal-ts", required=True)
    parser.add_argument("--vector-state-root", type=Path, required=True)
    parser.add_argument("--vector-state-scope", required=True)
    parser.add_argument("--vector-state-components", nargs="+", required=True)
    parser.add_argument(
        "--state-only-single-row",
        action="store_true",
        help=(
            "Advance the saved regular transform state from exactly one completed "
            "primitive hour. This is the required preproduction execution mode; "
            "the retained-tail mode is reference/recovery only."
        ),
    )
    parser.add_argument(
        "--state-as-of",
        help=(
            "UTC timestamp represented by --vector-state-root before the one-row "
            "advance. Required with --state-only-single-row and must precede the "
            "signal by exactly one hour."
        ),
    )
    parser.add_argument("--tail-hours", type=int, default=1536)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable regular feature-state output exists: {args.out_dir}")
    if not args.state_only_single_row and args.tail_hours < 1536:
        raise ValueError("regular feature-state producer requires the proven 1,536-hour tail")
    if args.state_only_single_row and not args.state_as_of:
        raise ValueError("--state-only-single-row requires --state-as-of")
    for path in (args.bundle, args.source_state, args.candidates):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.vector_state_root.is_dir():
        raise NotADirectoryError(args.vector_state_root)
    signal = _utc(args.signal_ts)
    state_as_of = _utc(args.state_as_of) if args.state_as_of else None
    if args.state_only_single_row and signal != state_as_of + pd.Timedelta(hours=1):
        raise ValueError("one-row regular state must advance exactly one hour from --state-as-of")
    stack = P8USealedInferenceStack.load(args.bundle, root=ROOT)
    plan = stack.preproduction.feature_plan()
    source = joblib.load(args.source_state)
    panel = source.get("panel") if isinstance(source, Mapping) else None
    symbols = tuple(map(str, source.get("symbols") or ())) if isinstance(source, Mapping) else ()
    if not isinstance(panel, Mapping) or len(symbols) != 160 or len(set(symbols)) != 160:
        raise ValueError("regular feature-state source must retain the frozen 160-symbol universe")
    candidates = pd.read_parquet(args.candidates)
    _target_free(candidates)
    required = {*IDENTITY_COLUMNS, "__symbol__", "__ts__"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"regular feature-state candidates lack {missing}")
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    candidates["__decision_ts__"] = pd.to_datetime(
        candidates["__decision_ts__"], utc=True, errors="raise"
    )
    candidates = candidates.loc[candidates["__ts__"].eq(signal)].copy()
    if len(candidates) != 160 or candidates["__symbol__"].astype(str).nunique() != 160:
        raise ValueError("regular feature-state candidates must contain one full-universe row per symbol")
    if set(candidates["__symbol__"].astype(str)) != set(symbols):
        raise ValueError("regular feature-state candidate universe differs from frozen source universe")
    if candidates.loc[:, list(IDENTITY_COLUMNS)].duplicated().any():
        raise ValueError("regular feature-state candidates have duplicate identity")
    if not (candidates["__decision_ts__"] == signal + pd.Timedelta(hours=1)).all():
        raise ValueError("regular feature-state candidates have the wrong decision clock")

    args.out_dir.mkdir(parents=True)
    state = args.out_dir / "vector_state"
    clone_mode = clone_tree(args.vector_state_root.resolve(), state)
    started = time.perf_counter()
    stage = P8URouterFirstVectorizedStage(
        universe_symbols=symbols,
        plan=P8URouterFirstFeaturePlan(
            router_features=plan.router_features,
            base_features=plan.base_features,
            under_features=plan.under_features,
        ),
        direct_fields=DIRECT_EXPENSIVE_FEATURES,
    )
    source_panel = (
        _one_row(panel, signal=signal)
        if args.state_only_single_row
        else _tail(panel, signal=signal, hours=int(args.tail_hours))
    )
    state_contract_features, state_contract_source, state_contract_sha256 = _persisted_state_contract_features(
        state,
        symbols=symbols,
        required_features=plan.full_union,
    )
    supplemental_features, perp_tail_history_hours = _advance_perp_tail_supplement(
        state_dir=state,
        source_panel=panel,
        timestamp=signal,
        symbols=symbols,
    )
    snapshot = stage.materialize_regular_feature_state_snapshot(
        candidates=candidates,
        panel=source_panel,
        state_dir=str(state),
        state_scope=str(args.vector_state_scope),
        state_components=tuple(map(str, args.vector_state_components)),
        state_contract_features=state_contract_features,
        supplemental_features=supplemental_features,
    )
    expected_fields = tuple(field for field in plan.full_union if field not in DIRECT_EXPENSIVE_FEATURES)
    if tuple(column for column in snapshot.columns if column not in {"candidate_id", "__decision_ts__", "side_name", "__symbol__", "__ts__"}) != expected_fields:
        raise AssertionError("regular feature-state output changed the sealed regular-field order")
    if len(snapshot) != 160 or snapshot.loc[:, list(IDENTITY_COLUMNS)].duplicated().any():
        raise AssertionError("regular feature-state output changed candidate identity")
    feature_path = args.out_dir / "regular_vector_features.parquet"
    snapshot.to_parquet(feature_path, index=False, compression="zstd")
    receipt = {
        "schema": SCHEMA,
        "status": "pass_target_free_regular_state_append",
        "signal_ts": signal.isoformat(),
        "candidate_rows": int(len(snapshot)),
        "regular_feature_count": int(len(expected_fields)),
        "state_contract_feature_count": int(len(state_contract_features)),
        "state_contract_source": state_contract_source,
        "state_contract_sha256": state_contract_sha256,
        "perp_tail_history_hours": perp_tail_history_hours,
        "supplemental_regular_fields": sorted(supplemental_features),
        "direct_features_excluded": list(DIRECT_EXPENSIVE_FEATURES),
        "features": str(feature_path.resolve()),
        "features_sha256": _sha256(feature_path),
        "vector_state_parent": str(args.vector_state_root.resolve()),
        "vector_state_parent_tree_sha256": _sha256_tree(args.vector_state_root),
        "vector_state": str(state.resolve()),
        "vector_state_tree_sha256": _sha256_tree(state),
        "vector_state_scope": str(args.vector_state_scope),
        "vector_state_components": list(map(str, args.vector_state_components)),
        "input_mode": (
            "single_completed_source_hour_from_saved_state"
            if args.state_only_single_row
            else "retained_tail_reference_or_recovery"
        ),
        "state_as_of": None if state_as_of is None else state_as_of.isoformat(),
        "source_rows_fed": int(len(source_panel["close"])),
        "tail_hours": None if args.state_only_single_row else int(args.tail_hours),
        "clone_mode": clone_mode,
        "source_state": str(args.source_state.resolve()),
        "source_state_sha256": _sha256(args.source_state),
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": _sha256(args.candidates),
        "bundle": str(args.bundle.resolve()),
        "bundle_sha256": _sha256(args.bundle),
        "runtime_seconds": float(time.perf_counter() - started),
        "broad_retained_tail_feature_graph_called": False,
        "single_timestamp_canonical_projection_called": bool(args.state_only_single_row),
        "outcome_columns_consumed": [],
        "policy_or_portfolio_called": False,
        "exchange_or_order_submission_called": False,
    }
    _atomic_json(args.out_dir / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
