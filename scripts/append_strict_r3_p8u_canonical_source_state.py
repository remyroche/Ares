#!/usr/bin/env python3
"""Append completed, target-free P8U source hours to an immutable state.

The P8U feature graph is defined on the complete 160-symbol contemporaneous
universe.  This producer is deliberately source-only: it never creates
features, candidates, model scores, policy values, or orders.  Its sole job is
to create an auditable successor primitive panel whose new rows are fetched
with the same canonical source builder as the historical bootstrap.

The predecessor is never modified.  A failed source refresh writes no output;
the caller must retry the missing completed hour rather than skip it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    _add_frozen_input_backfill,
    _add_oi_funding_panels,
    _add_orderbook_panels,
    _make_panel,
)


SOURCE_SCHEMA = "strict_r3_p8u_canonical_source_panel_state_v1"
APPEND_SCHEMA = "strict_r3_p8u_canonical_source_append_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _load_source(path: Path) -> dict[str, Any]:
    loaded = joblib.load(path)
    if not isinstance(loaded, Mapping) or loaded.get("schema") != SOURCE_SCHEMA:
        raise ValueError("predecessor is not a P8U canonical source state")
    panel = loaded.get("panel")
    symbols = tuple(map(str, loaded.get("symbols") or ()))
    if not isinstance(panel, Mapping) or not symbols:
        raise ValueError("P8U source state lacks its panel or universe")
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("P8U source state lacks a non-empty close panel")
    if list(map(str, close.columns)) != list(symbols):
        raise ValueError("P8U source close columns differ from its sealed universe")
    observed = pd.DatetimeIndex(close.index)
    expected = pd.date_range(observed[0], observed[-1], freq="h", tz="UTC")
    if not observed.equals(expected):
        raise ValueError("predecessor source state has a non-contiguous hourly close panel")
    for name, frame in panel.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        if list(map(str, frame.columns)) != list(symbols):
            raise ValueError(f"predecessor source field {name} has a universe mismatch")
        if not pd.DatetimeIndex(frame.index).equals(observed):
            raise ValueError(f"predecessor source field {name} has an hourly-index mismatch")
    return dict(loaded)


def _canonical_symbols(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    symbols = tuple(map(str, payload.get("symbols") or ()))
    if len(symbols) != 160 or len(set(symbols)) != len(symbols):
        raise ValueError("canonical manifest must provide the frozen 160-symbol P8U universe")
    return symbols


def _build_new_panel(
    *,
    symbols: tuple[str, ...],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], Mapping[str, Any]]:
    panel, source_map = _make_panel(
        list(symbols),
        start,
        end_exclusive,
        allow_minute_fallback=False,
        bar_phase_minutes=0,
    )
    index = panel["close"].index
    expected = pd.date_range(start, end_exclusive - pd.Timedelta(hours=1), freq="h", tz="UTC")
    if not pd.DatetimeIndex(index).equals(expected):
        raise ValueError("canonical source builder did not return every required completed hour")
    _add_orderbook_panels(panel, list(symbols), index, start, end_exclusive)
    _add_oi_funding_panels(panel, list(symbols), index, start, end_exclusive)
    _add_frozen_input_backfill(panel, list(symbols), index, start, end_exclusive)
    output: dict[str, pd.DataFrame] = {}
    for name, frame in panel.items():
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"canonical source builder returned non-frame field {name}")
        if not pd.DatetimeIndex(frame.index).equals(expected):
            raise ValueError(f"canonical source field {name} has incomplete hourly coverage")
        if list(map(str, frame.columns)) != list(symbols):
            raise ValueError(f"canonical source field {name} changed the frozen universe")
        output[str(name)] = frame.loc[:, list(symbols)].astype("float32", copy=False)
    return output, source_map


def _validate_live_append_coverage(
    panel: Mapping[str, pd.DataFrame],
    *,
    symbols: tuple[str, ...],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> dict[str, Any]:
    """Require executable primitive coverage before publishing an append.

    A source panel is an upstream input to the P8U state machine.  Publishing
    a chronological row whose OHLCV primitives are entirely absent is worse
    than leaving the hour unprocessed: downstream model imputers would make
    the resulting scores look valid even though no contemporaneous market
    observation was available.  Keep this check here, before the immutable
    source successor exists, so the stateful executor cannot accidentally
    advance across a synthetic all-missing hour.

    Mark, index, funding and order-book fields have their own causal
    availability contracts and may be row-locally unavailable.  The five
    traded-price primitives are non-negotiable for every retained symbol and
    are therefore the publication gate.
    """
    required = ("open", "high", "low", "close", "volume")
    expected = pd.date_range(
        start,
        end_exclusive - pd.Timedelta(hours=1),
        freq="h",
        tz="UTC",
    )
    missing_fields = [name for name in required if name not in panel]
    if missing_fields:
        raise ValueError(f"canonical source build lacks required fields: {missing_fields}")
    records: list[dict[str, Any]] = []
    for name in required:
        frame = panel[name]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"canonical source field {name} is not a frame")
        finite = frame.reindex(index=expected, columns=list(symbols)).notna()
        coverage = finite.mean(axis=1)
        records.append({
            "field": name,
            "minimum_hourly_symbol_coverage": float(coverage.min()),
            "minimum_hourly_symbol_count": int(finite.sum(axis=1).min()),
            "complete_hours": int(coverage.eq(1.0).sum()),
            "expected_hours": int(len(expected)),
        })
        if not bool(finite.all().all()):
            bad = finite.index[~finite.all(axis=1)]
            preview = [stamp.isoformat() for stamp in bad[:3]]
            raise ValueError(
                "refusing to publish incomplete P8U OHLCV source append "
                f"field={name} missing_hours={len(bad)} preview={preview}"
            )
    return {
        "required_fields": list(required),
        "symbols": int(len(symbols)),
        "hours": int(len(expected)),
        "field_coverage": records,
        "status": "pass_complete_ohlcv",
    }


def _append_panel(
    previous: Mapping[str, Any],
    addition: Mapping[str, pd.DataFrame],
    *,
    symbols: tuple[str, ...],
) -> dict[str, Any]:
    original = previous.get("panel")
    if not isinstance(original, Mapping):
        raise ValueError("predecessor source state lacks its panel")
    old_fields = set(map(str, original))
    new_fields = set(map(str, addition))
    if old_fields != new_fields:
        raise ValueError(
            "canonical source field contract changed; "
            f"missing={sorted(old_fields - new_fields)[:5]} extra={sorted(new_fields - old_fields)[:5]}"
        )
    output: dict[str, Any] = {}
    for name in sorted(old_fields):
        old = original[name]
        new = addition[name]
        if not isinstance(old, pd.DataFrame):
            raise ValueError(f"predecessor source field {name} is not a DataFrame")
        if old.index[-1] >= new.index[0]:
            raise ValueError(f"source append overlaps or rewrites existing field {name}")
        combined = pd.concat([old, new], axis=0, copy=False)
        expected = pd.date_range(combined.index[0], combined.index[-1], freq="h", tz="UTC")
        if not pd.DatetimeIndex(combined.index).equals(expected):
            raise ValueError(f"source append left a time gap in {name}")
        if list(map(str, combined.columns)) != list(symbols):
            raise ValueError(f"source append changed the frozen universe in {name}")
        output[name] = combined.astype("float32", copy=False)
    return output


def _atomic_joblib(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(dict(payload), temporary, compress=3)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument(
        "--canonical-manifest",
        type=Path,
        help=(
            "optional original 160-symbol manifest; when supplied its SHA-256 "
            "must equal the predecessor's sealed manifest identity"
        ),
    )
    parser.add_argument(
        "--allow-reconstructed-source-map",
        action="store_true",
        help=(
            "Allow a replacement manifest only when it explicitly declares "
            "it was derived from this exact sealed predecessor and its "
            "source_map is an exact identity match. This is an auditable "
            "recovery path for a deleted manifest file; it never permits a "
            "different universe."
        ),
    )
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--require-complete-ohlcv",
        action="store_true",
        help=(
            "Fail before publication unless every frozen symbol has finite "
            "open/high/low/close/volume in every appended hour. Required for "
            "live source successors."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    predecessor = args.source_state.resolve()
    out_dir = args.out_dir.resolve()
    if not predecessor.is_file():
        raise FileNotFoundError("source state is unavailable")
    if out_dir.exists():
        raise FileExistsError(f"immutable source append output already exists: {out_dir}")
    state = _load_source(predecessor)
    symbols = tuple(map(str, state["symbols"]))
    sealed_manifest_hash = str(state.get("canonical_manifest_sha256") or "")
    if not sealed_manifest_hash:
        raise ValueError("predecessor source state lacks its sealed manifest hash")
    manifest_path: Path | None = None
    manifest_mode = "sealed_predecessor_identity"
    manifest_input_hash: str | None = None
    if args.canonical_manifest is not None:
        manifest_path = args.canonical_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError("canonical manifest is unavailable")
        manifest_input_hash = _sha256(manifest_path)
        if manifest_input_hash == sealed_manifest_hash:
            if symbols != _canonical_symbols(manifest_path):
                raise ValueError("predecessor source universe differs from canonical manifest")
            manifest_mode = "validated_input"
        else:
            if not args.allow_reconstructed_source_map:
                raise ValueError("supplied canonical manifest does not match predecessor manifest identity")
            payload = json.loads(manifest_path.read_text())
            if not isinstance(payload, Mapping):
                raise ValueError("reconstructed manifest must be a JSON object")
            if payload.get("derivation") != "sealed_source_state_source_map_only":
                raise ValueError("reconstructed manifest lacks the exact-source-map derivation")
            if str(payload.get("source_state_sha256") or "") != _sha256(predecessor):
                raise ValueError("reconstructed manifest is not bound to this predecessor")
            source_map = payload.get("source_map")
            if not isinstance(source_map, Mapping):
                raise ValueError("reconstructed manifest has no source_map")
            reconstructed_symbols = tuple(sorted(map(str, source_map)))
            if reconstructed_symbols != tuple(sorted(symbols)):
                raise ValueError("reconstructed source_map differs from predecessor universe")
            supplied_symbols = tuple(map(str, payload.get("symbols") or ()))
            if supplied_symbols != tuple(sorted(symbols)):
                raise ValueError("reconstructed manifest symbol list differs from predecessor universe")
            manifest_mode = "reconstructed_exact_source_map"
    close = state["panel"]["close"]
    assert isinstance(close, pd.DataFrame)
    start = pd.Timestamp(close.index[-1]) + pd.Timedelta(hours=1)
    end_exclusive = _utc(args.end_exclusive)
    if end_exclusive <= start:
        raise ValueError("end-exclusive must follow the next missing completed source hour")
    preview = {
        "schema": APPEND_SCHEMA,
        "predecessor_source_panel": str(predecessor),
        "predecessor_source_panel_sha256": _sha256(predecessor),
        "append_start": start.isoformat(),
        "end_exclusive": end_exclusive.isoformat(),
        "hours": int((end_exclusive - start) / pd.Timedelta(hours=1)),
        "symbols": len(symbols),
        "canonical_manifest_sha256": sealed_manifest_hash,
        "canonical_manifest_input_sha256": manifest_input_hash,
        "canonical_manifest_mode": manifest_mode,
        "outcome_columns_consumed": [],
    }
    if args.dry_run:
        print(json.dumps({**preview, "status": "dry_run"}, sort_keys=True))
        return
    addition, source_map = _build_new_panel(symbols=symbols, start=start, end_exclusive=end_exclusive)
    coverage = (
        _validate_live_append_coverage(
            addition,
            symbols=symbols,
            start=start,
            end_exclusive=end_exclusive,
        )
        if args.require_complete_ohlcv
        else None
    )
    combined = _append_panel(state, addition, symbols=symbols)
    out_dir.mkdir(parents=True)
    successor = {
        "schema": SOURCE_SCHEMA,
        "history_start": pd.Timestamp(combined["close"].index[0]),
        "end_exclusive": end_exclusive,
        "symbols": list(symbols),
        "source_map": state.get("source_map"),
        "append_source_map": source_map,
        "canonical_manifest_sha256": sealed_manifest_hash,
        "parent_source_panel_sha256": preview["predecessor_source_panel_sha256"],
        "panel": combined,
    }
    state_path = out_dir / "source_panel_state.joblib"
    _atomic_joblib(state_path, successor)
    receipt = {
        **preview,
        "schema": "strict_r3_p8u_canonical_source_append_receipt_v1",
        "status": "pass_target_free_append_only",
        "source_panel": str(state_path),
        "source_panel_sha256": _sha256(state_path),
        "canonical_manifest_sha256": sealed_manifest_hash,
        "field_count": len(combined),
        "rows_after_append": int(len(combined["close"])),
        "latest_source_timestamp": pd.Timestamp(combined["close"].index[-1]).isoformat(),
        "outcome_columns_consumed": [],
        "feature_state_published": False,
        "live_ohlcv_coverage": coverage,
    }
    _atomic_json(out_dir / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
