#!/usr/bin/env python3
"""Materialise frozen causal router inputs for a strict-R3 forward month.

This offline helper preserves target-free identities from historical
``scores_features`` panels and a schema-v2 forward candidate grid, then
regenerates only an explicit frozen router feature contract with the canonical
causal feature engine.  It never opens labels, trains a model, or accesses an
exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.run_tp6_sl4_exact170_canonical_consensus import materialize_features  # noqa: E402
import run_strict_r3_o3v2_target_funnel as target_contract  # noqa: E402


SCHEMA = "strict_r3_router_forward_selected_causal_features_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
NATIVE_IDENTITY = ("__ts__", "__symbol__")


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_months(raw: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(
        pd.Timestamp(f"{value.strip()}-01", tz="UTC")
        for value in raw.split(",")
        if value.strip()
    )
    if not months or len(set(months)) != len(months) or tuple(sorted(months)) != months:
        raise ValueError("--months must contain unique chronological YYYY-MM values")
    expected = tuple(pd.date_range(months[0], months[-1], freq="MS", tz="UTC"))
    if months != expected:
        raise ValueError("--months must be a contiguous interval")
    return months


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _chunks(items: tuple[pd.Timestamp, ...], size: int) -> Iterable[tuple[pd.Timestamp, ...]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _decision_batches(frame: pd.DataFrame, days: int) -> Iterable[pd.DataFrame]:
    """Yield chronological target slices while retaining all symbols per time.

    The causal feature engine creates several wide upstream panels before it
    selects the requested fields.  Slicing *target timestamps* (not symbols)
    prevents one long historical replay from exhausting memory.  Every slice
    still receives its own full prior warm-up interval, so no feature ever
    receives a future value or a symbol-subset cross-sectional universe.
    """
    start = pd.Timestamp(frame["__ts__"].min())
    end = pd.Timestamp(frame["__ts__"].max()) + pd.Timedelta(hours=1)
    cursor = start
    while cursor < end:
        stop = min(cursor + pd.Timedelta(days=days), end)
        result = frame.loc[frame["__ts__"].ge(cursor) & frame["__ts__"].lt(stop)].copy()
        if not result.empty:
            yield result
        cursor = stop


def _feature_contract(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    raw = payload.get("feature_contract", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) and item for item in raw):
        raise ValueError(f"{path}: feature contract must be a non-empty list")
    fields = tuple(raw)
    if len(fields) != len(set(fields)):
        raise ValueError(f"{path}: duplicate feature")
    return fields


def _feature_contract_from_source(path: Path) -> tuple[str, ...]:
    """Read the fixed model fields from a frozen target-free base panel.

    The source's first thirteen fields are the documented identity/base-score
    handoff.  The remaining ordered fields are the immutable base feature
    contract.  This is a convenience for a forward-only materialisation, not
    feature discovery: the complete resolved list and source hash are written
    to the destination manifest before feature generation starts.
    """
    names = pq.ParquetFile(path).schema_arrow.names
    prefix = (
        "candidate_id", "__decision_ts__", "base_bps", "efficiency_bps",
        "timing_bps", "enhanced_base_bps", "base_rank_ts",
        "enhanced_base_routed", "e_minus_t", "e_minus_b0", "t_minus_b0",
        "base_component_std", "side_name",
    )
    if tuple(names[:len(prefix)]) != prefix:
        raise AssertionError(f"{path}: unexpected frozen base handoff prefix")
    fields = tuple(names[len(prefix):])
    if len(fields) != 120 or len(set(fields)) != len(fields):
        raise AssertionError(f"{path}: expected one ordered 120-field base contract")
    return fields


def _assert_target_free(path: Path, names: set[str]) -> None:
    leaked = sorted(set(target_contract.PROHIBITED_SCORE_COLUMNS).intersection(names))
    if leaked:
        raise AssertionError(f"{path}: target-free source leaks {leaked}")
    missing = sorted(set(IDENTITY).union(NATIVE_IDENTITY).difference(names))
    if missing:
        raise AssertionError(f"{path}: missing identity {missing}")


def _historical_source_path(
    source_root: Path,
    raw_month_store: Path | None,
    month: pd.Timestamp,
) -> Path:
    """Return the immutable target-free identity source for one month.

    The normal source is the enhanced-base ``scores_features`` handoff.  A
    retained raw target-free monthly store can explicitly fill an historical
    gap when that handoff was cleaned up.  It is a fallback only: a normal
    handoff always wins, so existing reconstructed months retain byte parity.
    """
    source = source_root / f"month={month:%Y-%m}" / "scores_features.parquet"
    if source.exists():
        return source
    if raw_month_store is not None:
        # Historical monthly stores have used both zero- and one-indexed part
        # names.  Select an existing explicit part only; never synthesize a
        # predecessor or infer one from a later month.
        candidates = tuple(
            raw_month_store / f"month={month:%Y-%m}" / name
            for name in ("part-001.parquet", "part-000.parquet")
        )
        found = tuple(path for path in candidates if path.exists())
        # iCloud can retain an obsolete sparse placeholder beside the current
        # partition (for example ``part-000`` beside a valid ``part-001``).
        # Do not choose by filename or mtime: select only when exactly one
        # candidate is itself a readable Parquet file.  Two readable parts
        # remain genuinely ambiguous and fail closed; zero readable parts
        # surface the original source error instead of being misreported as
        # an identity choice.
        readable: list[Path] = []
        for path in found:
            try:
                pq.ParquetFile(path).schema_arrow
            except Exception:
                continue
            readable.append(path)
        if len(readable) == 1:
            return readable[0]
        if len(readable) > 1:
            raise AssertionError(
                f"{month:%Y-%m}: ambiguous readable raw target-free parts {tuple(readable)}"
            )
        if len(found) == 1:
            return found[0]
    raise FileNotFoundError(source)


def _historical_identities(source: Path, month: pd.Timestamp) -> pd.DataFrame:
    names = set(pq.ParquetFile(source).schema_arrow.names)
    leaked = sorted(set(target_contract.PROHIBITED_SCORE_COLUMNS).intersection(names))
    if leaked:
        raise AssertionError(f"{source}: target-free source leaks {leaked}")
    if not set(IDENTITY).issubset(names):
        raise AssertionError(f"{source}: missing model identity")
    raw_columns = list(IDENTITY)
    # The raw target-free store preserves the native symbol whereas the
    # enhanced-base handoff does not.  Keep it when available; otherwise
    # derive the same deterministic symbol from the candidate identity.
    has_symbol = "__symbol__" in names
    if has_symbol:
        raw_columns.append("__symbol__")
    raw = pd.read_parquet(source, columns=raw_columns)
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    raw["__ts__"] = raw["__decision_ts__"] - pd.Timedelta(hours=1)
    if not has_symbol:
        raw["__symbol__"] = raw["candidate_id"].astype(str).str.split("|", n=1, expand=True)[0]
    return raw.loc[:, [*IDENTITY, *NATIVE_IDENTITY]]


def _forward_identities(grid: Path, month: pd.Timestamp) -> pd.DataFrame:
    manifest_path = grid / "run_manifest.json"
    generic_manifest_path = grid / "target_free_candidate_population.manifest.json"
    source = grid / "target_free_candidate_population.parquet"
    if not manifest_path.exists() and generic_manifest_path.exists():
        manifest_path = generic_manifest_path
    if not manifest_path.exists() or not source.exists():
        raise FileNotFoundError(f"{grid}: expected target-free manifest and panel")
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema")
    accepted = {
        "strict_r3_canonical_forward_v2_target_free_hourly_grid",
        "strict_r3_recall_target_free_candidate_grid_v1",
    }
    if schema not in accepted:
        raise AssertionError(f"{grid}: unexpected candidate-grid schema")
    if schema == "strict_r3_canonical_forward_v2_target_free_hourly_grid":
        if manifest.get("future_path_columns_consumed") != []:
            raise AssertionError(f"{grid}: candidate grid has a future-path input")
    elif bool(manifest.get("outcome_fields_read")) or bool(manifest.get("score_fields_read")):
        raise AssertionError(f"{grid}: generic candidate grid is not target-free")
    _assert_target_free(source, set(pq.ParquetFile(source).schema_arrow.names))
    raw = pd.read_parquet(source, columns=[*IDENTITY, *NATIVE_IDENTITY])
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True, errors="raise")
    return raw.loc[raw["__decision_ts__"].ge(month) & raw["__decision_ts__"].lt(_month_end(month))].copy()


def _validate_identities(frame: pd.DataFrame, month: pd.Timestamp, source: Path) -> pd.DataFrame:
    work = frame.loc[:, [*IDENTITY, *NATIVE_IDENTITY]].copy()
    if work.empty:
        raise AssertionError(f"{source}: no identities for {month:%Y-%m}")
    if work.duplicated(IDENTITY).any() or work.duplicated(NATIVE_IDENTITY).any():
        raise AssertionError(f"{source}: duplicate target-free identity")
    if not work["side_name"].eq("long").all():
        raise AssertionError(f"{source}: expected long-only source")
    if not work["__decision_ts__"].ge(month).all() or not work["__decision_ts__"].lt(_month_end(month)).all():
        raise AssertionError(f"{source}: identity outside declared month")
    if not work["__decision_ts__"].eq(work["__ts__"] + pd.Timedelta(hours=1)).all():
        raise AssertionError(f"{source}: noncanonical decision timestamp")
    return work.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _coverage(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in fields:
        value = pd.to_numeric(frame[field], errors="coerce")
        rows.append({"feature": field, "rows": int(len(value)), "finite_rows": int(value.notna().sum()), "finite_fraction": float(value.notna().mean()), "n_unique": int(value.nunique(dropna=True))})
    return pd.DataFrame(rows)


def _materialise_chunk(
    *, out: Path, identities_by_month: dict[pd.Timestamp, pd.DataFrame], fields: tuple[str, ...], warmup_days: int,
    decision_batch_days: int,
) -> list[dict[str, Any]]:
    months = tuple(identities_by_month)
    candidates = pd.concat(list(identities_by_month.values()), ignore_index=True)
    if candidates.duplicated(IDENTITY).any():
        raise AssertionError("target-free candidate identities duplicate across feature chunk")
    batch = out / "_feature_batches" / f"{months[0]:%Y%m}_{months[-1]:%Y%m}"
    if batch.exists():
        raise FileExistsError(f"{batch}: immutable feature batch already exists")
    generated_parts: list[pd.DataFrame] = []
    slice_audit: list[dict[str, Any]] = []
    for ordinal, target_slice in enumerate(_decision_batches(candidates, decision_batch_days)):
        target_start = pd.Timestamp(target_slice["__ts__"].min())
        target_end = pd.Timestamp(target_slice["__ts__"].max()) + pd.Timedelta(hours=1)
        context_start = target_start - pd.Timedelta(days=warmup_days)
        slice_dir = batch / f"slice={ordinal:03d}"
        generated_path = materialize_features(
            slice_dir,
            target_slice,
            {"long": list(fields), "short": []},
            context_start,
            target_end,
            full_feature_universe=False,
        )
        generated_parts.append(pd.read_parquet(generated_path, columns=[*NATIVE_IDENTITY, *fields]))
        slice_audit.append({
            "slice": int(ordinal), "target_rows": int(len(target_slice)),
            "target_start": target_start.isoformat(), "target_end_exclusive": target_end.isoformat(),
            "context_start": context_start.isoformat(),
        })
    generated = pd.concat(generated_parts, ignore_index=True)
    generated["__ts__"] = pd.to_datetime(generated["__ts__"], utc=True, errors="raise")
    if generated.duplicated(NATIVE_IDENTITY).any():
        raise AssertionError(f"{batch}: causal engine generated duplicate native identity")
    audits: list[dict[str, Any]] = []
    for month, identities in identities_by_month.items():
        target = out / f"month={month:%Y-%m}"
        if target.exists():
            raise FileExistsError(f"{target}: immutable monthly panel already exists")
        panel = identities.merge(generated, on=list(NATIVE_IDENTITY), how="left", validate="one_to_one")
        if len(panel) != len(identities) or panel.duplicated(IDENTITY).any():
            raise AssertionError(f"{month:%Y-%m}: feature generation changed identities")
        target.mkdir(parents=True)
        panel.to_parquet(target / "causal_feature_universe.parquet", index=False, compression="zstd")
        _coverage(panel, fields).to_parquet(target / "feature_coverage.parquet", index=False, compression="zstd")
        audits.append({"month": f"{month:%Y-%m}", "rows": int(len(panel)), "finite_all_selected_fraction": float(panel.loc[:, list(fields)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).mean()), "feature_batch": batch.name, "decision_batch_days": int(decision_batch_days), "feature_slices": slice_audit, "target_free": True})
    return audits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True, help="contiguous YYYY-MM months")
    parser.add_argument("--historical-source-root", type=Path, required=True)
    parser.add_argument(
        "--historical-raw-month-store", type=Path,
        help="optional immutable target-free monthly fallback used only when a historical scores_features month is absent",
    )
    parser.add_argument("--forward-grid", type=Path)
    parser.add_argument("--forward-month")
    parser.add_argument(
        "--historical-only", action="store_true",
        help="materialise every requested month from immutable historical target-free identities",
    )
    contract = parser.add_mutually_exclusive_group(required=True)
    contract.add_argument("--feature-contract", type=Path)
    contract.add_argument("--feature-contract-source", type=Path)
    parser.add_argument("--warmup-days", type=int, default=180)
    parser.add_argument("--chunk-months", type=int, default=1)
    parser.add_argument("--decision-batch-days", type=int, default=31)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out}: immutable output exists")
    # The frozen Router contract's longest explicit transform is the
    # 1,458-hour FFD window (~61 days).  Seventy-five days retains a 14-day
    # causal settling margin while allowing bounded replay slices to remain
    # below the materialiser's wide-panel memory ceiling.  Coverage is still
    # audited per output field and per candidate row before any model fit.
    if args.warmup_days < 75 or args.chunk_months < 1 or args.chunk_months > 2 or not 1 <= args.decision_batch_days <= 31:
        raise ValueError("--warmup-days must be >=75, --chunk-months in [1,2], and --decision-batch-days in [1,31]")
    months = _parse_months(args.months)
    if args.historical_only:
        if args.forward_grid is not None or args.forward_month is not None:
            raise ValueError("--historical-only is mutually exclusive with --forward-grid/--forward-month")
        forward_month = None
    else:
        if args.forward_grid is None or args.forward_month is None:
            raise ValueError("--forward-grid and --forward-month are required unless --historical-only is set")
        forward_month = pd.Timestamp(f"{args.forward_month}-01", tz="UTC")
        if forward_month != months[-1]:
            raise ValueError("--forward-month must be the final requested month")
    fields = (
        _feature_contract(args.feature_contract)
        if args.feature_contract is not None
        else _feature_contract_from_source(args.feature_contract_source)
    )
    args.out.mkdir(parents=True)
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-free causal selected-router feature materialisation; no labels, models, MC1, admission, portfolio, inference, live, or exchange mutation",
        "months": [f"{month:%Y-%m}" for month in months],
        "forward_month": f"{forward_month:%Y-%m}" if forward_month is not None else None,
        "historical_only": bool(args.historical_only),
        "historical_source_root": str(args.historical_source_root),
        "historical_raw_month_store": (
            str(args.historical_raw_month_store)
            if args.historical_raw_month_store is not None else None
        ),
        "forward_grid": str(args.forward_grid) if args.forward_grid is not None else None,
        "feature_contract": list(fields), "feature_contract_sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest(),
        "feature_contract_source": (
            str(args.feature_contract) if args.feature_contract is not None else str(args.feature_contract_source)
        ),
        "feature_contract_source_sha256": _sha_file(
            args.feature_contract if args.feature_contract is not None else args.feature_contract_source
        ),
        # A forward-only invocation has no historical source partition in its
        # requested months.  Record that fact explicitly rather than probing a
        # nonexistent future-score panel merely to manufacture a hash.
        "historical_source_sha256": (
            None
            if forward_month is not None and months[0] == forward_month
            else _sha_file(
                _historical_source_path(
                    args.historical_source_root,
                    args.historical_raw_month_store,
                    months[0],
                )
            )
        ),
        "forward_grid_sha256": (
            _sha_file(args.forward_grid / "target_free_candidate_population.parquet")
            if args.forward_grid is not None else None
        ),
        "warmup_days": int(args.warmup_days), "chunk_months": int(args.chunk_months), "decision_batch_days": int(args.decision_batch_days),
        "identity_contract": "candidate_id, UTC decision timestamp, side, native timestamp/symbol; full target-free identities retained independent of eligibility and future paths",
    })
    identities: dict[pd.Timestamp, pd.DataFrame] = {}
    for month in months:
        is_forward = forward_month is not None and month == forward_month
        source = (
            args.forward_grid
            if is_forward
            else _historical_source_path(
                args.historical_source_root, args.historical_raw_month_store, month
            )
        )
        raw = (
            _forward_identities(args.forward_grid, month)
            if is_forward
            else _historical_identities(source, month)
        )
        identities[month] = _validate_identities(raw, month, source)
    audits: list[dict[str, Any]] = []
    for block in _chunks(months, args.chunk_months):
        print(json.dumps({"event": "chunk_start", "months": [f"{month:%Y-%m}" for month in block]}), flush=True)
        audits.extend(_materialise_chunk(out=args.out, identities_by_month={month: identities[month] for month in block}, fields=fields, warmup_days=args.warmup_days, decision_batch_days=args.decision_batch_days))
    audit = pd.DataFrame(audits).sort_values("month", kind="stable")
    audit.to_parquet(args.out / "identity_and_coverage_audit.parquet", index=False, compression="zstd")
    print(json.dumps({"event": "complete", "months": len(audit), "rows": int(audit.rows.sum())}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
