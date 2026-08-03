#!/usr/bin/env python3
"""Audit and materialise the Stage-I 2024--26 causal feature surface.

This is deliberately a *source-separated* surface.  It does not pretend that
the full-universe 2024 panel and the Pack-B 2025--26 candidate ledger are one
homogeneous population.  Each output partition has an explicit source id and
schema/lineage hash.  Downstream experiments may compare the sources, but must
not concatenate them without an explicit common-feature contract.

The ``--dry-run`` default is non-mutating: it inventories every month, reports
the declared-vs-available feature contract by layer and side, and identifies
gaps.  ``--materialize`` writes only named partitions and never fills a missing
feature with zero.  Exact TP6/SL4/H12 labels are reused for the 2024 source;
the Pack-B source is intentionally reported as label-ready-but-not-yet-TP6
materialised until its minute-path relabeller is invoked in a separate bounded
job.  Both sources require the R3 robust-clear B25/T50 primitives.  This
prevents a legacy 24-hour label from being relabelled as TP6/H12.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.stage_i_feature_selection import (
    resolve_stage_i_feature_universe,
    stage_i_active_contracts,
)
PANEL_2024 = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
LABELS_2024 = ROOT / "data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1"
ROBUST_2024 = ROOT / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1"
PACKB_LABELS = ROOT / "data_perp/artifacts/20260720_s59_h5_fullthroughjul10_candleclose_trailing_cost100bps_labels/labels"
FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
LEGACY_2022_2023_CANDIDATES = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_label_inputs_20260730_v2/candidates.parquet"
HISTORICAL_2022_TP6 = ROOT / "data_perp/artifacts/historical_2022_tp6_sl4_h12_20260809_v2"
SCHEMA = "stage_i_source_separated_tp6_surface_v1"
SIDES = ("long", "short")
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__")
LABEL_COLUMNS = (
    "tp6_sl4_entry_price", "t2_tp6_sl4_event", "t2_tp6_sl4_exit_minute",
    "t4_tp6_sl4_exit_pnl_atr", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
    "t4_tp6_sl4_terminal_pnl_atr", "__label_available_at__",
)
OUTPUT_LABEL_COLUMNS = (*LABEL_COLUMNS, "label_available_ts")
R3_PRIMITIVES = (
    "label_valid", "pre_adverse_mfe_atr", "lower_touch_minute",
    "robust_clear_event_b25", "robust_clear_soft_b25_t50",
)

# These are raw decision-time inputs used by the one shared residual experiment.
# Model scores / priors are deliberately absent: they must be generated
# chronological-OOF after the base fit, not copied from an old residual run.
# These groups contain values generated only after a chronological base fit.
# They remain in the declared selector universe, but must never be read from a
# historical residual artifact or replaced with zero in this surface.  The
# Stage-I runner generates them later from same-side chronological-OOF outputs.
META_GENERATED_LATER_GROUPS = (
    "META_BASE_PERFORMANCE_FEATURE_KEYS",
    "META_MODEL_UNCERTAINTY_FEATURE_KEYS",
    "META_RECENT_EFFECTIVENESS_FEATURE_KEYS",
    "BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS",
    "BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def expand_feature_refs(refs: Iterable[str], cfg: Mapping[str, Any]) -> list[str]:
    """Recursively expand config feature-list references without cross-layer fallbacks."""
    def visit(value: str, stack: frozenset[str]) -> list[str]:
        nested = cfg.get(value)
        if isinstance(nested, (list, tuple)):
            if value in stack:
                raise ValueError(f"cyclic config feature group: {value}")
            result: list[str] = []
            for item in nested:
                result.extend(visit(str(item), stack | {value}))
            return result
        return [value]
    result: list[str] = []
    for ref in refs:
        result.extend(visit(str(ref), frozenset()))
    return _ordered_unique(result)


def declared_feature_contract(cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return the selector-exact, layer-separated Stage-I candidate pools.

    ``meta_selector`` is deliberately the exact selector universe, not a
    convenient subset of columns found in a historical panel.  Its fields are
    then classified by provenance: decision-time store fields can be joined
    now, while model/OOF fields are recorded as generated later.  This keeps
    the surface honest without shrinking the meta selector by 73 fields.
    """
    meta_contract = next(
        contract for contract in stage_i_active_contracts() if contract.layer == "meta"
    )
    meta_selector = resolve_stage_i_feature_universe(
        cfg, layer="meta", side="long", head=meta_contract.head
    )
    generated_later = set(expand_feature_refs(META_GENERATED_LATER_GROUPS, cfg))
    meta_generated_later = [field for field in meta_selector if field in generated_later]
    meta_raw_store = [field for field in meta_selector if field not in generated_later]
    contract = {
        "base_long": resolve_stage_i_feature_universe(cfg, layer="base", side="long"),
        "base_short": resolve_stage_i_feature_universe(cfg, layer="base", side="short"),
        "meta_selector": meta_selector,
        "meta_raw_store": meta_raw_store,
        "meta_generated_later": meta_generated_later,
    }
    # Retained only for existing readers; it is now selector-exact rather than
    # an accidentally truncated raw-context proxy.
    contract["meta_shared_residual"] = list(meta_selector)
    return contract


@dataclass(frozen=True)
class SourceMonth:
    source_id: str
    month: str
    rows: int
    symbols: int
    candidate_id_contract: str
    label_status: str
    features_path: str
    labels_path: str | None
    available_columns: tuple[str, ...]


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _packb_month_paths(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in sorted(root.glob("train_global_*_5_????_??.parquet")):
        year, month = path.stem.rsplit("_", 2)[-2:]
        result.setdefault(f"{year}-{month}", []).append(path)
    return result


def _nearest_existing_path(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def packb_month_preflight(
    month: str,
    *,
    contract: Mapping[str, list[str]],
    labels_root: Path = PACKB_LABELS,
    feature_store: Path = FEATURE_STORE,
    disk_path: Path = ROOT,
) -> dict[str, Any]:
    """Estimate a bounded Pack-B write without loading feature values.

    The estimate deliberately counts only decision-time raw-store fields.  OOF
    / model-derived meta fields are generated by the runner later and occupy no
    raw-surface bytes.  It is advisory only: a reference layout remains the
    default and reuses the authorised PIT store in place.
    """
    paths = _packb_month_paths(labels_root).get(month, [])
    if len(paths) != 2:
        raise FileNotFoundError(f"expected exactly long+short Pack-B shards for {month}, found {len(paths)}")
    symbols = pd.concat([pd.read_parquet(path, columns=["__symbol__"]) for path in paths], ignore_index=True)
    symbol_names = symbols["__symbol__"].astype(str).drop_duplicates().tolist()
    raw_fields = _ordered_unique([
        *contract["base_long"], *contract["base_short"], *contract["meta_raw_store"],
    ])
    schemas: list[set[str]] = []
    missing_stores: list[str] = []
    for symbol in symbol_names:
        store = feature_store / f"symbol={symbol.replace('/', '_')}.parquet"
        if store.exists():
            schemas.append(_parquet_columns(store))
        else:
            missing_stores.append(symbol)
    any_present = set().union(*schemas) if schemas else set()
    every_present = set.intersection(*schemas) if schemas else set()
    rows = int(sum(pq.ParquetFile(path).metadata.num_rows for path in paths))
    raw_present = [field for field in raw_fields if field in any_present]
    raw_present_every_symbol = [field for field in raw_fields if field in every_present]
    # This is feature values only; identity/labels/Parquet metadata add a small
    # amount, so the estimate is intentionally labelled rather than a quota.
    estimate_bytes = rows * len(raw_present) * np.dtype("float32").itemsize
    free_bytes = shutil.disk_usage(_nearest_existing_path(disk_path)).free
    return {
        "month": month,
        "rows": rows,
        "symbols": len(symbol_names),
        "raw_store_declared_fields": len(raw_fields),
        "raw_store_fields_present_any_symbol": raw_present,
        "raw_store_fields_present_every_symbol": raw_present_every_symbol,
        "raw_store_fields_missing_all_symbols": [field for field in raw_fields if field not in any_present],
        "stores_missing": missing_stores,
        "estimated_uncompressed_float32_bytes": int(estimate_bytes),
        "free_disk_bytes": int(free_bytes),
        "estimated_fraction_of_free_disk": float(estimate_bytes / free_bytes) if free_bytes else float("inf"),
        "default_layout": "reference_identity_labels",
        "materialized_matrix_requires_explicit_opt_in": True,
    }


def _month_from_part(path: Path) -> pd.Series:
    value = pd.read_parquet(path, columns=["__ts__"])
    return _month_from_timestamps(value["__ts__"])


def _month_from_timestamps(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True)
    return ts.dt.year.astype("int32") * 100 + ts.dt.month.astype("int32")


def _inventory_pre2024_sources() -> list[SourceMonth]:
    """Inventory older training candidates without pretending they are ready.

    January 2024 OOS needs training history.  These sources therefore appear
    in the surface plan, but stay source-separated and blocked from fitting
    until the exact TP6/R3 relabel contract has been verified for every row.
    """
    result: list[SourceMonth] = []
    for path in sorted((HISTORICAL_2022_TP6 / "parts").glob("*.parquet")):
        fields = _parquet_columns(path)
        rows = pq.ParquetFile(path).metadata.num_rows
        dates = pd.read_parquet(path, columns=["__ts__", "__symbol__"])
        for month, group in dates.assign(__month__=_month_from_timestamps(dates["__ts__"])).groupby("__month__", sort=False):
            token = f"{int(month) // 100:04d}-{int(month) % 100:02d}"
            result.append(SourceMonth(
                "historical_2022_exact_tp6", token, int(len(group)), int(group["__symbol__"].nunique()),
                "historical_exact_candidate_id", "exact_tp6_present_requires_r3_b25_t50_relabel",
                str(path), str(path), tuple(sorted(fields)),
            ))
    if LEGACY_2022_2023_CANDIDATES.exists():
        fields = _parquet_columns(LEGACY_2022_2023_CANDIDATES)
        values = pd.read_parquet(LEGACY_2022_2023_CANDIDATES, columns=["__ts__", "__symbol__"])
        values["__month__"] = _month_from_timestamps(values["__ts__"])
        for month, group in values.groupby("__month__", sort=True):
            token = f"{int(month) // 100:04d}-{int(month) % 100:02d}"
            result.append(SourceMonth(
                "historical_2022_2023_candidate_inputs", token, int(len(group)), int(group["__symbol__"].nunique()),
                "failure_recovery_candidate_id", "requires_exact_tp6_r3_relabel",
                str(LEGACY_2022_2023_CANDIDATES), None, tuple(sorted(fields)),
            ))
    return result


def inventory_sources(
    *, panel_2024: Path = PANEL_2024, labels_2024: Path = LABELS_2024,
    robust_2024: Path = ROBUST_2024, packb_labels: Path = PACKB_LABELS, feature_store: Path = FEATURE_STORE,
) -> list[SourceMonth]:
    """Read schemas / identity projections only; no feature-surface writes."""
    result: list[SourceMonth] = _inventory_pre2024_sources()
    try:
        _require_completed_exact_tp6_sidecar(labels_2024, require_r3=False)
        _require_completed_exact_tp6_sidecar(robust_2024, require_r3=True)
        r3_manifest_ready = True
    except (FileNotFoundError, ValueError):
        r3_manifest_ready = False
    panel_parts = sorted((panel_2024 / "parts").glob("*.parquet"))
    side_parts = {path.name: labels_2024 / "parts" / path.name for path in panel_parts}
    robust_parts = {path.name: robust_2024 / "parts" / path.name for path in panel_parts}
    # Read each large panel part once.  Re-reading every part once per month
    # turns a harmless dry run into a multi-gigabyte scan.
    by_month: dict[str, dict[str, Any]] = {}
    for part in panel_parts:
        fields = _parquet_columns(part)
        x = pd.read_parquet(part, columns=["__ts__", "__symbol__"])
        ts = pd.to_datetime(x["__ts__"], utc=True)
        x["__month__"] = ts.dt.year.astype("int32") * 100 + ts.dt.month.astype("int32")
        for month, group in x.groupby("__month__", sort=False):
            token = f"{int(month) // 100:04d}-{int(month) % 100:02d}"
            item = by_month.setdefault(token, {"rows": 0, "symbols": set(), "available": set(), "label_ready": True})
            item["rows"] += int(len(group))
            item["symbols"].update(group["__symbol__"].astype(str))
            item["available"].update(fields)
            item["label_ready"] &= side_parts[part.name].exists() and robust_parts[part.name].exists() and r3_manifest_ready
    for month, item in sorted(by_month.items()):
        if month < "2024-01":
            continue
        # The panel is an identity/label substrate.  Its original static
        # subset is not the Stage-I feature ceiling: reference the same
        # pre-existing PIT store used to create it, without copying it.
        store_columns: set[str] = set()
        for symbol in sorted(item["symbols"]):
            store = feature_store / f"symbol={str(symbol).replace('/', '_')}.parquet"
            if store.exists():
                store_columns.update(_parquet_columns(store))
        item["available"].update(store_columns)
        result.append(SourceMonth(
            "full_universe_2024", month, int(item["rows"]), len(item["symbols"]),
            "panel_candidate_id__signal_close_plus_1h", "exact_tp6_sl4_h12_r3_b25_t50_pit_reference" if item["label_ready"] else "missing_exact_or_r3_sidecar",
            str(feature_store), str(labels_2024) if item["label_ready"] else None, tuple(sorted(item["available"])),
        ))
    for month, parts in sorted(_packb_month_paths(packb_labels).items()):
        if len(parts) != 2:
            continue
        sample = pd.read_parquet(parts[0], columns=["candidate_id", "__symbol__"])
        cols = set.intersection(*(_parquet_columns(path) for path in parts))
        # The raw store is the only feature source for Pack-B.  Its schema is
        # queried here; this is not a claim that every candidate is covered.
        store_columns: set[str] = set()
        for symbol in sample["__symbol__"].astype(str).str.replace("/", "_", regex=False).drop_duplicates().head(8):
            store = feature_store / f"symbol={symbol}.parquet"
            if store.exists():
                store_columns.update(_parquet_columns(store))
        result.append(SourceMonth(
            "packb_2025_2026", month, int(sum(pq.ParquetFile(path).metadata.num_rows for path in parts)),
            int(sample["__symbol__"].nunique()), "packb_candidate_id__signal_close_plus_1h", "requires_exact_tp6_sl4_h12_relabel",
            str(feature_store), str(packb_labels), tuple(sorted(store_columns | cols)),
        ))
    return result


def _feature_availability(contract: Mapping[str, list[str]], available: Iterable[str]) -> dict[str, dict[str, list[str]]]:
    fields = set(available)
    return {
        key: {"present": [name for name in values if name in fields], "missing": [name for name in values if name not in fields]}
        for key, values in contract.items()
    }


def _pit_store_path(store: Path, symbol: str) -> Path:
    return store / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _pit_store_bounds(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Read compact cache metadata only; never infer coverage from future rows."""
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    payload = json.loads(meta_path.read_text())
    try:
        first = pd.Timestamp(payload["first_ts"])
        last = pd.Timestamp(payload["last_ts"])
        first = first.tz_localize("UTC") if first.tzinfo is None else first.tz_convert("UTC")
        last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
        return first, last
    except (KeyError, TypeError, ValueError):
        return None


def audit_2024_pit_reference_readiness(
    *,
    panel_2024: Path = PANEL_2024,
    feature_store: Path = FEATURE_STORE,
    contract: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    """Report whether the 2024 candidate population can *reference* the PIT store.

    This is a schema/bounds audit, not a claim that every numeric value is
    usable.  Actual materialisation calls the exact timestamp join below and
    fails on any missing or duplicate candidate timestamp.  That separation
    avoids a large duplicate matrix while making the causal contract explicit.
    """
    groups: dict[str, dict[str, Any]] = {}
    for part in sorted((panel_2024 / "parts").glob("*.parquet")):
        required = ["candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__"]
        _require_columns(part, required, source=f"2024 candidate panel {part.name}")
        frame = pd.read_parquet(part, columns=required)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
        for month_number, candidate in frame.assign(__month__=_month_from_timestamps(frame["__ts__"])).groupby("__month__", sort=False):
            month = f"{int(month_number) // 100:04d}-{int(month_number) % 100:02d}"
            state = groups.setdefault(month, {"rows": 0, "symbols": {}, "decision_contract": True, "duplicate_candidate_ids": 0})
            state["rows"] += int(len(candidate))
            state["decision_contract"] &= candidate["__decision_ts__"].eq(candidate["__ts__"] + pd.Timedelta(hours=1)).all()
            state["duplicate_candidate_ids"] += int(candidate.candidate_id.duplicated().sum())
            for symbol, subframe in candidate.groupby("__symbol__", sort=False):
                item = state["symbols"].setdefault(str(symbol), {"rows": 0, "min": None, "max": None})
                item["rows"] += int(len(subframe))
                lower, upper = subframe["__ts__"].min(), subframe["__ts__"].max()
                item["min"] = lower if item["min"] is None else min(item["min"], lower)
                item["max"] = upper if item["max"] is None else max(item["max"], upper)
    raw_groups = {"base_long": contract["base_long"], "base_short": contract["base_short"], "meta_raw_store": contract["meta_raw_store"]}
    results: list[dict[str, Any]] = []
    for month, state in sorted(groups.items()):
        if month < "2024-01":
            continue
        schemas: list[set[str]] = []
        missing_symbols: list[str] = []
        no_clock_symbols: list[str] = []
        bounds_fail_symbols: list[str] = []
        for symbol, candidate in state["symbols"].items():
            path = _pit_store_path(feature_store, symbol)
            if not path.exists():
                missing_symbols.append(symbol)
                continue
            schema = _parquet_columns(path)
            schemas.append(schema)
            if "ts" not in schema:
                no_clock_symbols.append(symbol)
            bounds = _pit_store_bounds(path)
            if bounds is None or bounds[0] > candidate["min"] or bounds[1] < candidate["max"]:
                bounds_fail_symbols.append(symbol)
        any_columns = set().union(*schemas) if schemas else set()
        every_columns = set.intersection(*schemas) if schemas else set()
        present = {
            key: {
                "declared": len(fields), "present_any_store": len(set(fields) & any_columns),
                "present_every_store": len(set(fields) & every_columns),
            }
            for key, fields in raw_groups.items()
        }
        ready = not missing_symbols and not no_clock_symbols and not bounds_fail_symbols and bool(state["decision_contract"])
        results.append({
            "source_id": "full_universe_2024", "month": month, "candidate_rows": int(state["rows"]),
            "candidate_symbols": len(state["symbols"]), "duplicate_candidate_ids_within_shard": int(state["duplicate_candidate_ids"]),
            "signal_close_to_decision_plus_1h": bool(state["decision_contract"]),
            "store_symbols_missing": missing_symbols, "store_symbols_without_ts": no_clock_symbols,
            "store_symbols_outside_metadata_bounds": bounds_fail_symbols, "raw_feature_schema": present,
            "status": "READY_REFERENCE_EXACT_JOIN_REQUIRED" if ready else "BLOCKED_PIT_REFERENCE_LINEAGE",
            "no_lookahead_rule": "read only store ts == candidate __ts__; never asof; decision is __ts__ +1h",
            "coverage_note": "schema/bounds ready only; exact per-row timestamp coverage and >=90% non-null/nonconstant remain materialization gates",
        })
    return results


def _require_completed_exact_tp6_sidecar(sidecar: Path, *, require_r3: bool) -> dict[str, Any]:
    """Fail closed unless an explicit, completed exact TP6 sidecar is supplied."""
    manifest_path = _exact_sidecar_manifest_path(sidecar)
    if not manifest_path.exists():
        raise FileNotFoundError(f"exact TP6 sidecar manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    status = str(manifest.get("status", "")).strip().lower()
    if manifest.get("complete") is not True and status not in {"complete", "completed"}:
        raise ValueError("exact TP6 sidecar manifest is not complete")
    blob = json.dumps(manifest, sort_keys=True).lower().replace(" ", "")
    if "tp6" not in blob or "sl4" not in blob or "100" not in blob or ("h12" not in blob and "720" not in blob):
        raise ValueError("sidecar manifest does not attest TP6/SL4/H12 and the fixed 100-bps cost")
    if require_r3 and ("robust_clear" not in blob or "25" not in blob or "50" not in blob):
        raise ValueError("sidecar manifest does not attest the R3 robust-clear B25/T50 primitive")
    return manifest


def _exact_sidecar_manifest_path(sidecar: Path) -> Path:
    """Resolve the canonical manifest, retaining legacy read-only fallback."""
    for name in ("manifest.json", "run_manifest.json"):
        path = sidecar / name
        if path.exists():
            return path
    raise FileNotFoundError(f"exact TP6 sidecar manifest is required: {sidecar / 'manifest.json'}")


def _require_columns(path: Path, required: Iterable[str], *, source: str) -> None:
    missing = sorted(set(required) - _parquet_columns(path))
    if missing:
        raise ValueError(f"{source} lacks required fields: {missing}")


def audit(*, output_dir: Path | None = None) -> dict[str, Any]:
    from extreme_price_movements.config import CFG

    contract = declared_feature_contract(CFG)
    inventory = inventory_sources()
    pit_2024_readiness = audit_2024_pit_reference_readiness(contract=contract)
    preflight_root = output_dir if output_dir is not None else ROOT
    packb_preflight = [
        packb_month_preflight(month, contract=contract, disk_path=preflight_root)
        for month in sorted(_packb_month_paths(PACKB_LABELS))
    ]
    rows = []
    for item in inventory:
        avail = _feature_availability(contract, item.available_columns)
        rows.append({
            "source_id": item.source_id, "month": item.month, "rows": item.rows, "symbols": item.symbols,
            "candidate_id_contract": item.candidate_id_contract, "label_status": item.label_status,
            "feature_source": item.features_path, "label_source": item.labels_path,
            **{f"{layer}_declared": len(contract[layer]) for layer in contract},
            **{f"{layer}_present_schema": len(avail[layer]["present"]) for layer in contract},
            **{f"{layer}_missing_schema": len(avail[layer]["missing"]) for layer in contract},
        })
    report = {
        "schema": SCHEMA,
        "mode": "dry_run_inventory",
        "contract": {"geometry": "TP6/SL4/H12", "cost_bps": 100.0, "entry": "signal close +1h, exact next-minute open", "label_available_ts": "entry +12h"},
        "layer_separation": {
            "base": "selector-exact base_shared + side-specific keys",
            "meta": "selector-exact M6 meta universe; raw store fields are joined now and OOF/model fields are generated later",
        },
        "declared_contract": contract,
        "contract_hash": _json_hash(contract),
        "source_months": rows,
        "full_universe_2024_pit_reference_readiness": pit_2024_readiness,
        "packb_write_preflight": packb_preflight,
        "limitations": [
            "The 2022 historical exact TP6 source and 2022-23 candidate-input source are training inventory only: both require exact TP6/R3 B25/T50 contract integration before fitting.",
            "A January-2024 OOS cannot use a surface beginning in January 2024 as its training source; the older sources above must be made label-ready first.",
            "2024 full-universe rows require both exact TP6/SL4/H12 and a completed R3 B25/T50 robust-clear sidecar; their source period ends 2024-11.",
            "Pack-B candidate rows begin 2025-01 and need a bounded exact-minute TP6 relabel pass before modelling.",
            "December 2024 is absent from both input sources and is explicitly not interpolated.",
            "Pack-B defaults to identity+label references to the PIT store. Full raw-matrix duplication requires explicit opt-in after checking the disk estimate.",
            "Source schemas are reported independently. No Pack-B/common30/full-universe concatenation or zero imputation is authorised.",
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output_dir / "source_month_inventory.parquet", index=False)
        pd.DataFrame([{key: value for key, value in item.items() if not isinstance(value, list)} for item in packb_preflight]).to_parquet(
            output_dir / "packb_write_preflight.parquet", index=False
        )
        (output_dir / "full_universe_2024_pit_reference_readiness.json").write_text(
            json.dumps(pit_2024_readiness, indent=2, default=str) + "\n"
        )
        (output_dir / "stage_i_surface_dry_run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _validate_exact_join(panel: pd.DataFrame, labels: pd.DataFrame, robust: pd.DataFrame | None = None) -> pd.DataFrame:
    if panel.candidate_id.duplicated().any() or labels.candidate_id.duplicated().any() or (robust is not None and robust.candidate_id.duplicated().any()):
        raise ValueError("candidate_id must be unique in every exact source part")
    result = panel.merge(labels.loc[:, [*IDENTITY, *LABEL_COLUMNS]], on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(result) != len(labels):
        raise ValueError("TP6 sidecar contains candidate ids absent from its panel part")
    if not pd.to_datetime(result["__label_available_at__"], utc=True).eq(pd.to_datetime(result["__decision_ts__"], utc=True) + pd.Timedelta(hours=12)).all():
        raise ValueError("label_available_ts is not exact entry + H12")
    result["label_available_ts"] = result["__label_available_at__"]
    if robust is None:
        raise ValueError("R3 requires a separately verified robust-clear B25/T50 sidecar")
    _require_columns_from_frame(robust, [*IDENTITY, *R3_PRIMITIVES], source="robust-clear sidecar")
    result = result.merge(robust.loc[:, [*IDENTITY, *R3_PRIMITIVES]], on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(result) != len(labels):
        raise ValueError("robust-clear sidecar does not cover every exact TP6 label")
    if result["label_valid"].isna().any():
        raise ValueError("R3 label validity must be explicit; invalid paths cannot become ordinary failures")
    valid = result["label_valid"].astype(bool)
    economics = ["t4_tp6_sl4_exit_pnl_atr", "t4_tp6_sl4_terminal_pnl_atr", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps"]
    if result.loc[~valid, economics].notna().any(axis=None):
        raise ValueError("invalid exact TP6 rows must not encode economic failures")
    if not np.allclose(
        result.loc[valid, "t4_tp6_sl4_gross_bps"].to_numpy(float) - 100.,
        result.loc[valid, "t4_tp6_sl4_net_bps"].to_numpy(float), atol=2e-3, rtol=0.,
    ):
        raise ValueError("TP6 cost must be applied exactly once on label-valid rows")
    return result


def _require_columns_from_frame(frame: pd.DataFrame, required: Iterable[str], *, source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks required fields: {missing}")


def _is_month(values: pd.Series, month: str) -> pd.Series:
    ts = pd.to_datetime(values, utc=True)
    year, number = (int(token) for token in month.split("-"))
    return ts.dt.year.eq(year) & ts.dt.month.eq(number)


def _packb_candidates_for_month(month: str, *, labels_root: Path) -> pd.DataFrame:
    """Read only Pack-B identity rows; their legacy targets are never retained."""
    paths = _packb_month_paths(labels_root).get(month, [])
    if len(paths) != 2:
        raise FileNotFoundError(f"expected exactly long+short Pack-B candidate shards for {month}, found {len(paths)}")
    fields = ("candidate_id", "__ts__", "__symbol__", "side_name")
    pieces = []
    for path in paths:
        _require_columns(path, fields, source=f"Pack-B candidates {path.name}")
        pieces.append(pd.read_parquet(path, columns=list(fields)))
    result = pd.concat(pieces, ignore_index=True)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    if not result.side_name.isin(SIDES).all() or result.candidate_id.duplicated().any():
        raise ValueError("Pack-B candidate identity must be unique and side-local")
    result["__decision_ts__"] = result["__ts__"] + pd.Timedelta(hours=1)
    return result


def _label_files_for_month(sidecar: Path, month: str) -> list[Path]:
    """Require a bounded monthly sidecar layout; never scan an all-era ledger."""
    choices = (
        sidecar / "parts" / f"month={month}", sidecar / f"month={month}",
        sidecar / "parts" / month,
    )
    for directory in choices:
        files = sorted(directory.glob("*.parquet")) if directory.exists() else []
        if files:
            return files
    files = sorted((sidecar / "parts").glob(f"*{month.replace('-', '')}*.parquet")) if (sidecar / "parts").exists() else []
    if files:
        return files
    raise FileNotFoundError(
        f"exact sidecar must expose a bounded {month} partition under parts/month={month}; refusing an all-era scan"
    )


def _packb_exact_labels_for_month(month: str, *, sidecar: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _require_completed_exact_tp6_sidecar(sidecar, require_r3=True)
    required = [*IDENTITY, *LABEL_COLUMNS, *R3_PRIMITIVES]
    pieces = []
    for path in _label_files_for_month(sidecar, month):
        _require_columns(path, required, source=f"exact Pack-B TP6 sidecar {path.name}")
        pieces.append(pd.read_parquet(path, columns=required))
    result = pd.concat(pieces, ignore_index=True)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True)
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True)
    result["__label_available_at__"] = pd.to_datetime(result["__label_available_at__"], utc=True)
    if result.candidate_id.duplicated().any():
        raise ValueError("exact Pack-B TP6 sidecar has duplicate candidate ids")
    return result, manifest


def _join_packb_exact_labels(candidates: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    # Candidate IDs alone are not sufficient: the sidecar must bind the exact
    # signal timestamp, symbol, side, and decision timestamp too.
    identity_by_id = labels.loc[:, list(IDENTITY)].rename(columns={column: f"sidecar_{column}" for column in IDENTITY if column != "candidate_id"})
    probe = candidates.merge(identity_by_id, on="candidate_id", how="left", validate="one_to_one")
    if len(probe) != len(candidates) or probe[[f"sidecar_{column}" for column in IDENTITY if column != "candidate_id"]].isna().any(axis=None):
        raise ValueError("Pack-B candidate and exact TP6 sidecar identities must match exactly")
    for column in IDENTITY:
        if column == "candidate_id":
            continue
        if not probe[column].eq(probe[f"sidecar_{column}"]).all():
            raise ValueError(f"Pack-B candidate-sidecar full identity mismatch: {column}")
    result = candidates.merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(result) != len(candidates) or len(result) != len(labels):
        raise ValueError("Pack-B candidate and exact TP6 sidecar identities must match exactly")
    if not result["__label_available_at__"].eq(result["__decision_ts__"] + pd.Timedelta(hours=12)).all():
        raise ValueError("Pack-B exact TP6 sidecar violates the H12 label availability contract")
    if result["label_valid"].isna().any():
        raise ValueError("Pack-B R3 label validity must be explicit")
    valid = result["label_valid"].astype(bool)
    economics = ["t4_tp6_sl4_exit_pnl_atr", "t4_tp6_sl4_terminal_pnl_atr", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps"]
    if result.loc[~valid, economics].notna().any(axis=None):
        raise ValueError("Pack-B invalid exact TP6 rows must not encode economic failures")
    if not np.allclose(
        result.loc[valid, "t4_tp6_sl4_gross_bps"].to_numpy(float) - 100.,
        result.loc[valid, "t4_tp6_sl4_net_bps"].to_numpy(float), atol=2e-3, rtol=0.,
    ):
        raise ValueError("Pack-B TP6 cost must be applied exactly once on label-valid rows")
    result["label_available_ts"] = result["__label_available_at__"]
    return result


def _store_features_at_signal_close(
    candidates: pd.DataFrame, *, store: Path, fields: Iterable[str], start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    """Exact, backward-free timestamp join for one symbol's existing PIT store."""
    if candidates["__symbol__"].nunique() != 1:
        raise ValueError("feature-store joins are per symbol to preserve identity")
    symbol = str(candidates["__symbol__"].iloc[0]).replace("/", "_")
    path = store / f"symbol={symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Pack-B feature store misses candidate symbol: {symbol}")
    available = _parquet_columns(path)
    present = [field for field in _ordered_unique(fields) if field in available]
    if "ts" not in available:
        raise ValueError(f"PIT feature store lacks its physical UTC timestamp field: {symbol}")
    # ``ts`` is a physical, UTC timestamp field in the authorised PIT store.
    # Pandas restores it as an index for this store layout; normalise both
    # physical-column and index forms before the exact equality join.
    source = pd.read_parquet(path, columns=[*present, "ts"], filters=[("ts", ">=", start), ("ts", "<", end)])
    if "ts" in source:
        timestamp = pd.to_datetime(source.pop("ts"), utc=True)
    elif source.index.name == "ts" and isinstance(source.index, pd.DatetimeIndex):
        timestamp = pd.to_datetime(source.index, utc=True)
        source = source.reset_index(drop=True)
    else:
        raise ValueError(f"PIT feature store timestamp did not materialize: {symbol}")
    source.insert(0, "ts", timestamp)
    if source.ts.duplicated().any():
        raise ValueError(f"PIT feature store has duplicate timestamps: {symbol}")
    merged = candidates.merge(source, left_on="__ts__", right_on="ts", how="left", validate="many_to_one")
    if len(merged) != len(candidates):
        raise AssertionError("exact timestamp feature join changed candidate cardinality")
    return merged.drop(columns="ts")


def _join_pit_store_by_symbol_exact(
    frame: pd.DataFrame,
    *,
    store: Path,
    fields: Iterable[str],
) -> pd.DataFrame:
    """Attach only same-timestamp PIT values, preserving immutable row order."""
    pieces: list[pd.DataFrame] = []
    for _, candidate in frame.groupby("__symbol__", sort=False):
        start = pd.to_datetime(candidate["__ts__"], utc=True).min()
        end = pd.to_datetime(candidate["__ts__"], utc=True).max() + pd.Timedelta(hours=1)
        pieces.append(_store_features_at_signal_close(candidate.copy(), store=store, fields=fields, start=start, end=end))
    enriched = pd.concat(pieces, ignore_index=True)
    if len(enriched) != len(frame) or enriched.candidate_id.duplicated().any():
        raise ValueError("exact PIT join changed 2024 candidate identity/cardinality")
    # Every feature-store timestamp must equal the signal-close timestamp. The
    # underlying helper deliberately uses an equality merge rather than asof.
    if not enriched.loc[:, list(IDENTITY)].sort_values(list(IDENTITY)).reset_index(drop=True).equals(
        frame.loc[:, list(IDENTITY)].sort_values(list(IDENTITY)).reset_index(drop=True)
    ):
        raise ValueError("exact PIT join changed 2024 candidate-side identity")
    return enriched


def _coverage_records(frame: pd.DataFrame, *, fields: Iterable[str], layer: str, side: str, month: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in fields:
        value = pd.to_numeric(frame[field], errors="coerce").replace([np.inf, -np.inf], np.nan) if field in frame else pd.Series(dtype=float)
        result.append({
            "source_id": "packb_2025_2026", "month": month, "layer": layer, "side": side, "feature": field,
            "rows": int(len(frame)), "present": field in frame, "nonnull": int(value.notna().sum()),
            "min": float(value.min()) if value.notna().any() else np.nan,
            "max": float(value.max()) if value.notna().any() else np.nan,
        })
    return result


def _finalize_coverage(records: list[dict[str, Any]]) -> pd.DataFrame:
    audit = pd.DataFrame(records)
    if audit.empty:
        raise ValueError("no feature coverage records")
    keys = ["source_id", "month", "layer", "side", "feature"]
    result = audit.groupby(keys, as_index=False).agg(rows=("rows", "sum"), present=("present", "any"), nonnull=("nonnull", "sum"), min=("min", "min"), max=("max", "max"))
    result["coverage"] = result["nonnull"] / result["rows"].clip(lower=1)
    result["nonconstant"] = result["min"].ne(result["max"]) & result["min"].notna() & result["max"].notna()
    result["fit_eligible"] = result["present"] & result["coverage"].ge(.90) & result["nonconstant"]
    return result


def materialize_packb_month(
    *, month: str, exact_sidecar: Path, output_dir: Path,
    labels_root: Path = PACKB_LABELS, feature_store: Path = FEATURE_STORE,
    materialize_feature_matrices: bool = False,
) -> dict[str, Any]:
    """Write one sidecar-gated Pack-B month as small per-symbol layer parts.

    The safe default writes an identity+label reference layout and reuses the
    authorised PIT store.  Full raw-feature duplication needs explicit opt-in
    because even float32 matrices can consume a material fraction of free disk.
    Neither layout writes zero-filled placeholders for missing or later-OOF
    fields.
    """
    from extreme_price_movements.config import CFG

    candidates = _packb_candidates_for_month(month, labels_root=labels_root)
    labels, label_manifest = _packb_exact_labels_for_month(month, sidecar=exact_sidecar)
    frame = _join_packb_exact_labels(candidates, labels)
    contract = declared_feature_contract(CFG)
    destination = output_dir / "source=packb_2025_2026" / f"month={month}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Pack-B surface partition: {destination}")
    destination.mkdir(parents=True)
    preflight = packb_month_preflight(
        month, contract=contract, labels_root=labels_root, feature_store=feature_store,
        disk_path=output_dir,
    )
    lineage = {
        "schema": SCHEMA, "source_id": "packb_2025_2026", "month": month,
        "geometry": "TP6/SL4/H12", "cost_bps": 100., "entry": "candidate signal close +1h; exact sidecar entry",
        "label_available_ts": "__decision_ts__ +12h", "rows": int(len(frame)), "symbols": int(frame.__symbol__.nunique()),
        "layer_contract": contract, "layer_contract_hash": _json_hash(contract),
        "meta_feature_provenance": {
            "raw_store": contract["meta_raw_store"],
            "generated_later_chronological_oof": contract["meta_generated_later"],
            "rule": "generated-later fields are absent until same-side chronological-OOF generation; never zero-filled",
        },
        "r3_primitives": {"fields": list(R3_PRIMITIVES), "validity_gate": "label_valid; invalid targets remain null and are excluded in fitting"},
        "lineage": {
            "candidate_shards": [str(path) for path in _packb_month_paths(labels_root)[month]],
            "feature_store": str(feature_store), "exact_label_sidecar": str(exact_sidecar),
            "exact_label_manifest_sha256": _sha256(_exact_sidecar_manifest_path(exact_sidecar)),
            "exact_label_manifest_schema": label_manifest.get("schema"),
        },
        "write_preflight": preflight,
        "no_zero_imputation": True, "same_side_identity_only": True, "packb_not_concatenated_with_2024_or_common30": True,
    }
    if not materialize_feature_matrices:
        reference_destination = destination / "identity_labels"
        reference_destination.mkdir()
        fields = _ordered_unique([*IDENTITY, *OUTPUT_LABEL_COLUMNS, *R3_PRIMITIVES])
        for symbol, part in frame.groupby("__symbol__", sort=True):
            part.loc[:, fields].to_parquet(
                reference_destination / f"symbol={str(symbol).replace('/', '_')}.parquet",
                index=False, compression="zstd",
            )
        lineage.update({
            "layout": "reference_identity_labels",
            "feature_matrices_materialized": False,
            "fit_eligible_features": {},
            "runner_requirement": "join only declared raw_store fields at exact signal-close timestamps; generate listed OOF fields later",
        })
        lineage["lineage_hash"] = _json_hash(lineage)
        (destination / "manifest.json").write_text(json.dumps(lineage, indent=2) + "\n")
        return lineage
    coverage_records: list[dict[str, Any]] = []
    all_fields = _ordered_unique([*contract["base_long"], *contract["base_short"], *contract["meta_raw_store"]])
    start = frame["__ts__"].min()
    end = frame["__ts__"].max() + pd.Timedelta(hours=1)
    try:
        for symbol, candidate_part in frame.groupby("__symbol__", sort=True):
            enriched = _store_features_at_signal_close(candidate_part.copy(), store=feature_store, fields=all_fields, start=start, end=end)
            safe_symbol = str(symbol).replace("/", "_")
            for side in SIDES:
                subset = enriched.loc[enriched.side_name.eq(side)].copy()
                if subset.empty:
                    continue
                fields = contract[f"base_{side}"]
                present = [field for field in fields if field in subset]
                coverage_records.extend(_coverage_records(subset, fields=fields, layer="base", side=side, month=month))
                base_destination = destination / f"base_{side}"
                base_destination.mkdir(exist_ok=True)
                subset.loc[:, _ordered_unique([*IDENTITY, *OUTPUT_LABEL_COLUMNS, *R3_PRIMITIVES, *present])].to_parquet(
                    base_destination / f"symbol={safe_symbol}.parquet", index=False, compression="zstd"
                )
            meta_fields = contract["meta_raw_store"]
            present = [field for field in meta_fields if field in enriched]
            coverage_records.extend(_coverage_records(enriched, fields=meta_fields, layer="meta", side="same_side", month=month))
            (destination / "meta_same_side").mkdir(exist_ok=True)
            enriched.loc[:, _ordered_unique([*IDENTITY, *OUTPUT_LABEL_COLUMNS, *R3_PRIMITIVES, *present])].to_parquet(
                destination / "meta_same_side" / f"symbol={safe_symbol}.parquet", index=False, compression="zstd"
            )
    except Exception:
        # A partial feature surface is unsafe to consume.  Leave a human-readable
        # marker for diagnosis rather than presenting an incomplete month as valid.
        (destination / "INCOMPLETE.txt").write_text("materialization failed; do not consume this partition\n")
        raise
    coverage = _finalize_coverage(coverage_records)
    coverage.to_parquet(destination / "feature_coverage_nonconstant.parquet", index=False)
    eligible = {
        f"{row.layer}_{row.side}": list(group.feature)
        for (row.layer, row.side), group in coverage.loc[coverage.fit_eligible].groupby(["layer", "side"], sort=True)
    }
    for required in ("base_long", "base_short", "meta_same_side"):
        if not eligible.get(required):
            (destination / "INCOMPLETE.txt").write_text("coverage/nonconstant gate failed; do not consume this partition\n")
            raise ValueError(f"{required}: no fields pass >=90% coverage and nonconstant gate; no zero-fill fallback is permitted")
    lineage.update({
        "layout": "bounded_materialized_raw_matrices",
        "feature_matrices_materialized": True,
        "fit_eligible_features": eligible,
    })
    lineage["lineage_hash"] = _json_hash(lineage)
    (destination / "manifest.json").write_text(json.dumps(lineage, indent=2) + "\n")
    return lineage


def materialize_2024_month(
    *, month: str, output_dir: Path, feature_store: Path = FEATURE_STORE,
    materialize_feature_matrices: bool = False,
) -> dict[str, Any]:
    """Materialise one 2024 source month with distinct base/meta partitions.

    This is intentionally restricted to the source that already has a proven
    exact TP6 label sidecar.  Pack-B writes are refused until the exact relabel
    job has produced a compatible label sidecar.
    """
    from extreme_price_movements.config import CFG

    contract = declared_feature_contract(CFG)
    readiness = next(
        (item for item in audit_2024_pit_reference_readiness(
            panel_2024=PANEL_2024, feature_store=feature_store, contract=contract
        ) if item["month"] == month),
        None,
    )
    if readiness is None or readiness["status"] != "READY_REFERENCE_EXACT_JOIN_REQUIRED":
        raise ValueError(f"2024 {month} lacks a proved PIT reference contract: {readiness}")
    _require_completed_exact_tp6_sidecar(LABELS_2024, require_r3=False)
    _require_completed_exact_tp6_sidecar(ROBUST_2024, require_r3=True)
    labels_by_name = {path.name: path for path in (LABELS_2024 / "parts").glob("*.parquet")}
    robust_by_name = {path.name: path for path in (ROBUST_2024 / "parts").glob("*.parquet")}
    pieces: list[pd.DataFrame] = []
    for part in sorted((PANEL_2024 / "parts").glob("*.parquet")):
        label = labels_by_name.get(part.name)
        if label is None:
            raise FileNotFoundError(f"missing exact TP6 label sidecar: {part.name}")
        robust_path = robust_by_name.get(part.name)
        if robust_path is None:
            raise FileNotFoundError(f"missing R3 robust-clear sidecar: {part.name}")
        panel_columns = _parquet_columns(part)
        needed = list(dict.fromkeys([*IDENTITY, "atr_1h", "decision_price"]))
        panel = pd.read_parquet(part, columns=[col for col in needed if col in panel_columns])
        panel["__ts__"] = pd.to_datetime(panel["__ts__"], utc=True)
        panel = panel.loc[_is_month(panel["__ts__"], month)].copy()
        if panel.empty:
            continue
        label_columns = set(_parquet_columns(label))
        labels = pd.read_parquet(label, columns=[col for col in [*IDENTITY, *LABEL_COLUMNS] if col in label_columns])
        labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
        labels = labels.loc[_is_month(labels["__ts__"], month)].copy()
        robust_columns = _parquet_columns(robust_path)
        robust = pd.read_parquet(robust_path, columns=[col for col in [*IDENTITY, *R3_PRIMITIVES] if col in robust_columns])
        robust["__ts__"] = pd.to_datetime(robust["__ts__"], utc=True)
        robust = robust.loc[_is_month(robust["__ts__"], month)].copy()
        pieces.append(_validate_exact_join(panel, labels, robust))
    if not pieces:
        raise ValueError(f"no full-universe 2024 candidates for {month}")
    frame = pd.concat(pieces, ignore_index=True)
    out = output_dir / "source=full_universe_2024" / f"month={month}"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite 2024 surface partition: {out}")
    out.mkdir(parents=True)
    identity_labels = _ordered_unique([*IDENTITY, *OUTPUT_LABEL_COLUMNS, *R3_PRIMITIVES, "atr_1h", "decision_price"])
    identity_labels = [field for field in identity_labels if field in frame]
    reference_manifest = {
        "schema": SCHEMA, "source_id": "full_universe_2024", "month": month,
        "geometry": "TP6/SL4/H12", "cost_bps": 100., "entry": "source-bound signal close +1h exact minute open",
        "label_available_ts": "__decision_ts__ +12h", "rows": len(frame),
        "layer_contract": contract, "layer_contract_hash": _json_hash(contract),
        "pit_store_reference": {
            "path": str(feature_store), "timestamp": "ts == candidate __ts__", "no_lookahead": True,
            "no_asof_join": True, "readiness": readiness,
            "raw_fields": _ordered_unique([*contract["base_long"], *contract["base_short"], *contract["meta_raw_store"]]),
            "generated_later_chronological_oof": contract["meta_generated_later"],
        },
        "r3_primitives": {"definition": "separate robust sidecar; B25/T50", "fields": list(R3_PRIMITIVES)},
        "source_hashes": {"panel_manifest": _sha256(PANEL_2024 / "manifest.json"), "label_manifest": _sha256(LABELS_2024 / "manifest.json"), "robust_manifest": _sha256(ROBUST_2024 / "manifest.json")},
        "no_zero_imputation": True, "packb_not_concatenated": True,
    }
    if not materialize_feature_matrices:
        destination = out / "identity_labels"
        destination.mkdir()
        for symbol, part in frame.groupby("__symbol__", sort=True):
            part.loc[:, identity_labels].to_parquet(
                destination / f"symbol={str(symbol).replace('/', '_')}.parquet", index=False, compression="zstd"
            )
        reference_manifest.update({
            "layout": "reference_identity_labels", "feature_matrices_materialized": False,
            "fit_eligible_features": {},
            "runner_requirement": "perform exact PIT ts==__ts__ joins for raw fields; add model/OOF fields only through same-side chronological generation",
        })
        reference_manifest["lineage_hash"] = _json_hash(reference_manifest)
        (out / "manifest.json").write_text(json.dumps(reference_manifest, indent=2, default=str) + "\n")
        return reference_manifest
    raw_fields = _ordered_unique([*contract["base_long"], *contract["base_short"], *contract["meta_raw_store"]])
    base_frame = frame.loc[:, identity_labels].copy()
    frame = _join_pit_store_by_symbol_exact(base_frame, store=feature_store, fields=raw_fields)
    audit_rows: list[dict[str, Any]] = []
    for side in SIDES:
        part = frame.loc[frame.side_name.eq(side)].copy()
        fields = contract[f"base_{side}"]
        present = [field for field in fields if field in part]
        values = part.loc[:, present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for field in fields:
            series = values[field] if field in values else pd.Series(dtype=float)
            audit_rows.append({"source_id": "full_universe_2024", "month": month, "side": side, "layer": "base", "feature": field, "present": field in part, "coverage": float(series.notna().mean()) if len(series) else 0., "nonconstant": bool(series.nunique(dropna=True) > 1)})
        keep = [*IDENTITY, "atr_1h", "decision_price", *OUTPUT_LABEL_COLUMNS, *R3_PRIMITIVES, *present]
        part.loc[:, keep].to_parquet(out / f"base_{side}.parquet", index=False, compression="zstd")
    meta_fields = contract["meta_raw_store"]
    present = [field for field in meta_fields if field in frame]
    values = frame.loc[:, present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for field in meta_fields:
        series = values[field] if field in values else pd.Series(dtype=float)
        audit_rows.append({"source_id": "full_universe_2024", "month": month, "side": "same_side", "layer": "meta", "feature": field, "present": field in frame, "coverage": float(series.notna().mean()) if len(series) else 0., "nonconstant": bool(series.nunique(dropna=True) > 1)})
    frame.loc[:, [*IDENTITY, "atr_1h", "decision_price", *OUTPUT_LABEL_COLUMNS, *R3_PRIMITIVES, *present]].to_parquet(out / "meta_same_side.parquet", index=False, compression="zstd")
    coverage = pd.DataFrame(audit_rows)
    coverage["fit_eligible"] = coverage["present"] & coverage["coverage"].ge(.90) & coverage["nonconstant"]
    coverage.to_parquet(out / "feature_coverage_nonconstant.parquet", index=False)
    eligible = {
        f"{row.layer}_{row.side}": list(group.feature)
        for (row.layer, row.side), group in coverage.loc[coverage.fit_eligible].groupby(["layer", "side"], sort=True)
    }
    for required in ("base_long", "base_short", "meta_same_side"):
        if not eligible.get(required):
            (out / "INCOMPLETE.txt").write_text("coverage/nonconstant gate failed; do not consume this partition\n")
            raise ValueError(f"{required}: no fields pass >=90% coverage and nonconstant gate; no zero-fill fallback is permitted")
    lineage = reference_manifest
    lineage.update({
        "layout": "bounded_materialized_raw_matrices",
        "feature_matrices_materialized": True,
        "fit_eligible_features": eligible,
    })
    lineage["lineage_hash"] = _json_hash(lineage)
    (out / "manifest.json").write_text(json.dumps(lineage, indent=2) + "\n")
    return lineage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="inventory only (default unless --materialize-2024-month is set)")
    parser.add_argument("--materialize-2024-month", metavar="YYYY-MM", help="write exactly one already-labelled 2024 month")
    parser.add_argument("--materialize-packb-month", metavar="YYYY-MM", help="write one Pack-B month after an exact completed TP6 sidecar exists")
    parser.add_argument("--packb-tp6-sidecar", type=Path, help="completed, month-partitioned exact TP6/SL4/H12 + R3 sidecar")
    parser.add_argument(
        "--materialize-feature-matrices", action="store_true",
        help="explicitly duplicate bounded raw float32-compatible PIT fields; default is identity+label references",
    )
    args = parser.parse_args()
    if args.materialize_2024_month and args.materialize_packb_month:
        parser.error("choose only one materialization source per invocation")
    if args.materialize_2024_month:
        result = materialize_2024_month(
            month=args.materialize_2024_month, output_dir=args.out,
            materialize_feature_matrices=args.materialize_feature_matrices,
        )
        print(json.dumps({"status": "materialized", "month": args.materialize_2024_month, "rows": result["rows"], "lineage_hash": result["lineage_hash"]}, indent=2))
        return
    if args.materialize_packb_month:
        if args.packb_tp6_sidecar is None:
            parser.error("--materialize-packb-month requires --packb-tp6-sidecar")
        result = materialize_packb_month(
            month=args.materialize_packb_month, exact_sidecar=args.packb_tp6_sidecar,
            output_dir=args.out, materialize_feature_matrices=args.materialize_feature_matrices,
        )
        print(json.dumps({"status": "materialized", "month": args.materialize_packb_month, "rows": result["rows"], "lineage_hash": result["lineage_hash"]}, indent=2))
        return
    result = audit(output_dir=args.out)
    print(json.dumps({"status": "dry_run_complete", "source_months": len(result["source_months"]), "contract_hash": result["contract_hash"]}, indent=2))


if __name__ == "__main__":
    main()
