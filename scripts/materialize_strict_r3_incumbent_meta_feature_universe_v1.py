#!/usr/bin/env python3
"""Materialise a target-free, full causal feature universe for incumbent meta research.

The retained upstream is exactly ``0.50 * efficiency_bps + 0.50 *
timing_bps``.  This producer deliberately obtains its candidate identities
from the immutable target-free incumbent score receipts, not from labels or
future-path eligibility.  Policy and semantic outcomes are therefore unable
to change the candidate population, feature values, or coverage audit.

Only the fully causal feature engine is reused.  Its complete current-config
numeric output (roughly 1,400 fields) is written by calendar month, together
with the exact point-in-time base coordinates needed by the meta layer.  The
caller may materialise several contiguous months in bounded chunks so the
wide-panel graph never needs to occupy memory for the entire research era.

Research only.  This command cannot train models, map EV, admit trades, alter
the live stack, or access an exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    FROZEN_GENERATION_DEPENDENCIES,
    materialize_features,
)
import run_strict_r3_o3v2_target_funnel as target_contract  # noqa: E402


SCHEMA = "strict_r3_incumbent_meta_feature_universe_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
SCORE_COLUMNS = {
    "base_bps",
    "efficiency_bps",
    "timing_bps",
    "enhanced_base_bps",
    "base_rank_ts",
    "enhanced_base_routed",
    "e_minus_t",
    "e_minus_b0",
    "t_minus_b0",
    "base_component_std",
}
SOURCE_IDENTITY_COLUMNS = [*IDENTITY, *sorted(SCORE_COLUMNS)]
DEFAULT_SOURCE_ROOT = ROOT / "data_perp/artifacts/strict_r3_enhanced_base_live_stack_challenger_20260823_v10/target_free_monthly"
DEFAULT_BASE_PRIMITIVE_CONTRACT = ROOT / "data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json"
CANONICAL_PANEL_MATERIALISER = ROOT / "scripts" / "run_tp6_sl4_exact170_canonical_consensus.py"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


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
        raise ValueError("--months must form one contiguous calendar interval")
    return months


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _score_source(source_root: Path, month: pd.Timestamp) -> Path:
    return source_root / f"month={month:%Y-%m}" / "scores_features.parquet"


def _source_feature_contract(source_root: Path, month: pd.Timestamp) -> list[str]:
    source = _score_source(source_root, month)
    names = pq.ParquetFile(source).schema_arrow.names
    prohibited = set(target_contract.PROHIBITED_SCORE_COLUMNS)
    leaked = sorted(prohibited.intersection(names))
    if leaked:
        raise AssertionError(f"{source}: source is not target-free: {leaked}")
    fields = [name for name in names if name not in set(IDENTITY) | SCORE_COLUMNS]
    # ``scores_features`` carries 120 frozen causal fields plus candidate
    # metadata.  We need a non-empty causal contract for upstream primitive
    # preflight only; full_feature_universe=True below expands the generated
    # output to the complete causal-config union.
    fields = [field for field in fields if field not in {"__ts__", "__symbol__"}]
    if len(fields) != 120:
        raise AssertionError(f"{source}: expected 120 retained causal fields, found {len(fields)}")
    return fields


def _candidate_identities(source_root: Path, month: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read decision-time identities and validate the frozen E/T contract.

    No policy/semantic label path is opened here.  The arithmetic identity
    check is a source-contract audit, not a target calculation.
    """
    source = _score_source(source_root, month)
    if not source.exists():
        raise FileNotFoundError(source)
    schema = set(pq.ParquetFile(source).schema_arrow.names)
    missing = sorted(set(SOURCE_IDENTITY_COLUMNS) - schema)
    if missing:
        raise AssertionError(f"{source}: missing incumbent coordinate(s) {missing}")
    raw = pd.read_parquet(source, columns=SOURCE_IDENTITY_COLUMNS)
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    if raw.duplicated(IDENTITY).any():
        raise AssertionError(f"{source}: duplicate target-free identities")
    if not raw.side_name.eq("long").all():
        raise AssertionError(f"{source}: incumbent meta materialisation is long-only")
    if not raw.__decision_ts__.ge(month).all() or not raw.__decision_ts__.lt(_month_end(month)).all():
        raise AssertionError(f"{source}: rows fall outside declared decision month")
    expected = .5 * pd.to_numeric(raw.efficiency_bps, errors="coerce") + .5 * pd.to_numeric(raw.timing_bps, errors="coerce")
    observed = pd.to_numeric(raw.enhanced_base_bps, errors="coerce")
    delta = (expected - observed).abs()
    if not np.isfinite(delta).all() or float(delta.max()) > 1e-6:
        raise AssertionError(f"{source}: enhanced_base_bps is not the frozen 50/50 E/T coordinate")
    symbol = raw.candidate_id.astype(str).str.split("|", n=1, expand=True)[0]
    identities = raw.loc[:, [*IDENTITY]].copy()
    identities["__ts__"] = identities["__decision_ts__"] - pd.Timedelta(hours=1)
    identities["__symbol__"] = symbol.astype(str)
    if identities.candidate_id.duplicated().any():
        raise AssertionError(f"{source}: duplicate candidate IDs")
    audit: dict[str, Any] = {
        "month": f"{month:%Y-%m}",
        "identity_rows": int(len(identities)),
        "long_rows": int(identities.side_name.eq("long").sum()),
        "max_et_upstream_delta": float(delta.max()),
        "source_sha256": _sha_file(source),
        "outcome_columns_read": False,
    }
    return identities, audit


def _base_score_path(base_score_root: Path, month: pd.Timestamp) -> Path:
    path = base_score_root / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _candidate_identities_from_base_scores(
    base_score_root: Path,
    month: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read forward Base identities without touching targets or E/T scores."""
    source = _base_score_path(base_score_root, month)
    schema = set(pq.ParquetFile(source).schema_arrow.names)
    leaked = sorted(set(target_contract.PROHIBITED_SCORE_COLUMNS).intersection(schema))
    if leaked:
        raise AssertionError(f"{source}: Base score source is not target-free: {leaked}")
    required = set(IDENTITY) | {"base_score", "base_rank_ts"}
    missing = sorted(required.difference(schema))
    if missing:
        raise AssertionError(f"{source}: missing F72 Base coordinate(s) {missing}")
    raw = pd.read_parquet(source, columns=[*IDENTITY, "base_score", "base_rank_ts"])
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    if raw.duplicated(IDENTITY).any():
        raise AssertionError(f"{source}: duplicate target-free Base identities")
    if not raw.side_name.eq("long").all():
        raise AssertionError(f"{source}: F72 source must be long-only")
    if not raw.__decision_ts__.ge(month).all() or not raw.__decision_ts__.lt(_month_end(month)).all():
        raise AssertionError(f"{source}: rows fall outside declared decision month")
    if raw[["base_score", "base_rank_ts"]].isna().any(axis=None):
        raise AssertionError(f"{source}: target-free Base coordinate is non-finite")
    identities = raw.loc[:, [*IDENTITY]].copy()
    identities["__ts__"] = identities["__decision_ts__"] - pd.Timedelta(hours=1)
    identities["__symbol__"] = identities.candidate_id.astype(str).str.split("|", n=1, expand=True)[0]
    return identities, {
        "month": f"{month:%Y-%m}",
        "identity_rows": int(len(identities)),
        "long_rows": int(len(identities)),
        "base_coordinate_source": "F72_target_free_base_score",
        "source_sha256": _sha_file(source),
        "outcome_columns_read": False,
    }


def _primitive_contract(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    values = payload.get("selected_features", payload.get("feature_contract"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: expected selected_features or feature_contract")
    fields = [str(value) for value in values]
    if len(fields) != len(set(fields)):
        raise AssertionError(f"{path}: duplicate primitive-preflight feature")
    return fields


def _identities(
    *, source_root: Path, base_score_root: Path | None, month: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if base_score_root is not None:
        return _candidate_identities_from_base_scores(base_score_root, month)
    return _candidate_identities(source_root, month)


def _coverage(frame: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in fields:
        value = pd.to_numeric(frame[field], errors="coerce")
        rows.append({
            "feature": str(field),
            "rows": int(len(value)),
            "finite_rows": int(value.notna().sum()),
            "finite_fraction": float(value.notna().mean()),
            "n_unique": int(value.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def _full_fields_from_source(path: Path) -> tuple[str, ...]:
    """Read only a frozen full-universe schema, never its values or labels.

    The historical all-at-once full-feature path can exceed the workstation
    memory envelope on early panels.  A prior target-free full-universe receipt
    provides the authoritative ordered field universe.  It is used solely to
    predeclare the same feature keys for bounded deterministic batches.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    names = tuple(pq.ParquetFile(path).schema_arrow.names)
    forbidden = set(IDENTITY) | {"__ts__", "__symbol__"}
    fields = tuple(field for field in names if field not in forbidden)
    if len(fields) < 1_000 or len(fields) != len(set(fields)):
        raise AssertionError(f"{path}: invalid frozen full-universe field schema ({len(fields)})")
    return fields


def _context_symbols_from_manifest(path: Path) -> tuple[str, ...]:
    """Read a frozen point-in-time raw-context universe from a causal receipt.

    The manifest records raw input symbols, not selected candidate identities.
    This distinction is required for cross-sectional values such as market
    breadth, relative volatility, and peer residuals: a Router-selected
    subset must never become the market universe used to calculate them.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    symbols = payload.get("symbols")
    if not isinstance(symbols, list) or len(symbols) < 20:
        raise AssertionError(f"{path}: expected a complete causal context symbol list")
    result = tuple(sorted({str(symbol) for symbol in symbols if str(symbol)}))
    if len(result) < 20:
        raise AssertionError(f"{path}: invalid causal context symbol list")
    return result


def _materialize_predeclared_full_universe(
    *,
    out_dir: Path,
    identities: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: tuple[str, ...],
    field_chunk_size: int,
    reference_symbols: tuple[str, ...],
    context_symbols: tuple[str, ...],
) -> Path:
    """Materialize an equivalent full universe without retaining all fields at once.

    Each batch calls the canonical causal feature engine against the exact same
    point-in-time candidate identity grid, context range, source controls, and
    requested field names.  Batch outputs are joined only by the native
    timestamp/symbol identity.  No feature values are substituted, imputed, or
    sourced from labels.  Temporary batch panels are removed only after the
    final complete parquet is written successfully.
    """
    if field_chunk_size < 1:
        raise ValueError("field_chunk_size must be positive")
    native = ["__ts__", "__symbol__"]
    chunks_root = out_dir / "_predeclared_field_chunks"
    if chunks_root.exists():
        raise FileExistsError(f"{chunks_root}: immutable field-batch directory already exists")
    chunks_root.mkdir(parents=True)
    keys_expected = identities.loc[:, native].drop_duplicates().sort_values(native, kind="stable").reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    batch_audit: list[dict[str, Any]] = []
    try:
        for begin in range(0, len(fields), field_chunk_size):
            batch = fields[begin: begin + field_chunk_size]
            ordinal = begin // field_chunk_size
            batch_root = chunks_root / f"batch={ordinal:03d}"
            # ``materialize_features`` appends frozen generation dependencies
            # internally.  The read immediately below retains only the
            # explicitly predeclared full-universe fields for this batch.
            generated_path = materialize_features(
                batch_root,
                identities,
                {"long": list(batch), "short": []},
                start,
                end,
                full_feature_universe=False,
                reference_symbols=reference_symbols,
                context_symbols=context_symbols,
            )
            required = [*native, *batch]
            generated = pd.read_parquet(generated_path, columns=required)
            if generated.duplicated(native).any():
                raise AssertionError(f"{batch_root}: duplicate causal native identity")
            got = generated.loc[:, native].sort_values(native, kind="stable").reset_index(drop=True)
            if not got.equals(keys_expected):
                raise AssertionError(f"{batch_root}: field batch changed target-free identity grid")
            generated = generated.sort_values(native, kind="stable").reset_index(drop=True)
            parts.append(generated.loc[:, list(batch)].copy())
            batch_audit.append({
                "batch": int(ordinal), "first_field": str(batch[0]), "last_field": str(batch[-1]),
                "fields": int(len(batch)), "rows": int(len(generated)),
            })
        universe = pd.concat([keys_expected, *parts], axis=1, copy=False)
        if tuple(universe.columns) != (*native, *fields):
            raise AssertionError("predeclared field batches did not reconstruct the frozen ordered universe")
        if universe.duplicated(native).any() or len(universe) != len(keys_expected):
            raise AssertionError("predeclared field universe identity failure")
        target = out_dir / "causal_feature_universe.parquet"
        universe.to_parquet(target, index=False, compression="zstd")
        (out_dir / "predeclared_field_batch_audit.json").write_text(
            json.dumps({
                "schema": "strict_r3_predeclared_full_feature_batches_v1",
                "field_chunk_size": int(field_chunk_size), "field_count": int(len(fields)),
                "rows": int(len(universe)), "batches": batch_audit,
                "identity_preserved_every_batch": True,
            }, indent=2, sort_keys=True)
        )
        return target
    finally:
        # These are internal, regenerated-only implementation intermediates.
        # Preserve them on failure for diagnosis; remove them only when the
        # final universe parquet and its batch audit both exist.
        if (out_dir / "causal_feature_universe.parquet").exists() and (out_dir / "predeclared_field_batch_audit.json").exists():
            shutil.rmtree(chunks_root)


def _chunks(months: tuple[pd.Timestamp, ...], size: int) -> Iterable[tuple[pd.Timestamp, ...]]:
    for begin in range(0, len(months), size):
        yield months[begin: begin + size]


def _materialise_chunk(
    *,
    out: Path,
    source_root: Path,
    base_score_root: Path | None,
    months: tuple[pd.Timestamp, ...],
    warmup_days: int,
    frozen_contract: list[str],
    full_field_source: Path | None = None,
    field_chunk_size: int | None = None,
    context_symbols: tuple[str, ...] = (),
    allow_partial_shared_recovery: bool = False,
    reference_symbols: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    identities_by_month: dict[pd.Timestamp, pd.DataFrame] = {}
    audits: dict[pd.Timestamp, dict[str, Any]] = {}
    for month in months:
        identities, audit = _identities(
            source_root=source_root, base_score_root=base_score_root, month=month,
        )
        identities_by_month[month] = identities
        audits[month] = audit
    identities = pd.concat(list(identities_by_month.values()), ignore_index=True)
    if identities.duplicated(IDENTITY).any():
        raise AssertionError("combined target-free identity ledger has duplicates")
    batch_name = f"shared_{months[0]:%Y%m}_{months[-1]:%Y%m}"
    batch_root = out / "_shared" / batch_name
    if batch_root.exists():
        # A cancelled feature-engine call can leave source-preflight receipts
        # but no materialised panel.  Preserve that evidence and use a new,
        # explicitly named shared batch for the resumed immutable run.  A
        # completed panel is never regenerated or silently substituted.
        generated = batch_root / "causal_feature_universe.parquet"
        if not allow_partial_shared_recovery or generated.exists():
            raise FileExistsError(f"{batch_root}: immutable shared feature batch already exists")
        attempt = 1
        while True:
            candidate_name = f"{batch_name}_recovery{attempt:02d}"
            candidate_root = out / "_shared" / candidate_name
            if not candidate_root.exists():
                batch_name, batch_root = candidate_name, candidate_root
                print(json.dumps({
                    "event": "partial_shared_batch_preserved",
                    "prior_batch": str(generated.parent),
                    "replacement_batch": str(batch_root),
                }, sort_keys=True), flush=True)
                break
            attempt += 1
    context_start = months[0] - pd.Timedelta(days=warmup_days)
    context_end = _month_end(months[-1])
    if full_field_source is None:
        generated_path = materialize_features(
            batch_root,
            identities,
            {"long": frozen_contract, "short": []},
            context_start,
            context_end,
            full_feature_universe=True,
            reference_symbols=reference_symbols,
            context_symbols=context_symbols,
        )
    else:
        if field_chunk_size is None:
            raise AssertionError("predeclared full field source requires a positive batch size")
        generated_path = _materialize_predeclared_full_universe(
            out_dir=batch_root,
            identities=identities,
            start=context_start,
            end=context_end,
            fields=_full_fields_from_source(full_field_source),
            field_chunk_size=field_chunk_size,
            reference_symbols=reference_symbols,
            context_symbols=context_symbols,
        )
    generated = pd.read_parquet(generated_path)
    native_identity = ["__ts__", "__symbol__"]
    if any(field not in generated.columns for field in native_identity):
        raise AssertionError(f"{batch_root}: causal engine dropped native identity")
    generated_features = [
        field for field in generated.columns
        if field not in set(native_identity)
        and pd.api.types.is_numeric_dtype(generated[field].dtype)
    ]
    if len(generated_features) < 1_000:
        raise AssertionError(f"{batch_root}: full causal universe produced only {len(generated_features)} numeric fields")
    all_audits: list[dict[str, Any]] = []
    for month in months:
        month_root = out / f"month={month:%Y-%m}"
        if month_root.exists():
            raise FileExistsError(f"{month_root}: immutable monthly feature panel already exists")
        restored = identities_by_month[month].merge(
            generated,
            on=native_identity,
            how="inner",
            validate="one_to_one",
        )
        if len(restored) != len(identities_by_month[month]):
            raise AssertionError(f"{month:%Y-%m}: causal feature engine altered target-free identities")
        if restored.duplicated(IDENTITY).any():
            raise AssertionError(f"{month:%Y-%m}: restored feature panel duplicates an identity")
        month_root.mkdir(parents=True)
        restored.to_parquet(month_root / "causal_feature_universe.parquet", index=False, compression="zstd")
        _coverage(restored, generated_features).to_parquet(month_root / "feature_coverage.parquet", index=False, compression="zstd")
        audit = dict(audits[month])
        audit.update({
            "context_start": context_start.isoformat(),
            "context_end_exclusive": context_end.isoformat(),
            "feature_rows": int(len(restored)),
            "numeric_feature_columns": int(len(generated_features)),
            "target_fields_in_output": False,
            "materialisation_mode": "bounded_contiguous_causal_chunk",
            "shared_batch": batch_name,
            "reference_symbols": list(reference_symbols),
            "context_symbols": list(context_symbols),
            "full_field_source": str(full_field_source) if full_field_source is not None else None,
            "field_chunk_size": int(field_chunk_size) if field_chunk_size is not None else None,
        })
        all_audits.append(audit)
        print(json.dumps({"event": "month_complete", **audit}, sort_keys=True), flush=True)
    return all_audits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True, help="contiguous YYYY-MM values")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--base-score-root",
        type=Path,
        help=(
            "F72 target-free Base score root with month=YYYY-MM.parquet files. "
            "When supplied, it is the candidate identity source and the legacy "
            "E/T source root is not opened."
        ),
    )
    parser.add_argument("--primitive-preflight-contract", type=Path, default=DEFAULT_BASE_PRIMITIVE_CONTRACT)
    parser.add_argument("--warmup-days", type=int, default=180)
    parser.add_argument("--chunk-months", type=int, default=2)
    parser.add_argument(
        "--full-field-source", type=Path,
        help=("Target-free causal_feature_universe.parquet whose schema freezes the full feature keys. "
              "Used only with --field-chunk-size to bound full-universe memory."),
    )
    parser.add_argument(
        "--field-chunk-size", type=int,
        help="Positive predeclared feature-key batch size; requires --full-field-source.",
    )
    parser.add_argument(
        "--context-symbol-manifest", type=Path,
        help=("Target-free causal feature_manifest.json supplying the complete raw market context "
              "symbol universe before candidate routing. Required with --base-score-root."),
    )
    parser.add_argument(
        "--reference-symbol", action="append", default=[],
        help=("Optional contemporaneous benchmark input, e.g. BTC/USD:USD or "
              "ETH/USD:USD. It augments raw feature generation only and never "
              "creates a candidate row."),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume an interrupted immutable run: preserve completed monthly panels, "
            "materialise only missing months, and never overwrite the original manifest"
        ),
    )
    args = parser.parse_args()
    if args.warmup_days < 90:
        raise ValueError("--warmup-days must be at least 90")
    if args.chunk_months < 1 or args.chunk_months > 3:
        raise ValueError("--chunk-months must be in [1, 3] to bound the wide feature graph")
    if (args.full_field_source is None) != (args.field_chunk_size is None):
        raise ValueError("--full-field-source and --field-chunk-size must be supplied together")
    if args.field_chunk_size is not None and args.field_chunk_size < 1:
        raise ValueError("--field-chunk-size must be positive")
    if args.full_field_source is not None:
        args.full_field_source = args.full_field_source.resolve()
        _full_fields_from_source(args.full_field_source)
    if args.context_symbol_manifest is not None:
        args.context_symbol_manifest = args.context_symbol_manifest.resolve()
        context_symbols = _context_symbols_from_manifest(args.context_symbol_manifest)
    else:
        context_symbols = ()
    if args.base_score_root is not None and not context_symbols:
        raise ValueError("--base-score-root requires --context-symbol-manifest so cross-sectional features use the complete market universe")
    months = _parse_months(args.months)
    if args.base_score_root is not None:
        probe_fields = _primitive_contract(args.primitive_preflight_contract)
    else:
        probe_fields = _source_feature_contract(args.source_root, months[0])
        for month in months[1:]:
            current = _source_feature_contract(args.source_root, month)
            if current != probe_fields:
                raise AssertionError(f"{month:%Y-%m}: source 120-field contract differs from first month")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline research-only full causal meta feature materialisation; no labels/outcomes enter candidate or feature construction; no model, MC1, admission, portfolio, inference, live, or exchange mutation",
        "incumbent_upstream": (
            "F72 target-free Base score coordinates"
            if args.base_score_root is not None else "0.50 * efficiency_bps + 0.50 * timing_bps"
        ),
        "candidate_source": str(args.base_score_root or args.source_root),
        "candidate_contract": (
            "immutable F72 target-free Base source; every Router50 identity is retained independent of policy/semantic path availability"
            if args.base_score_root is not None else "immutable target-free incumbent source; every point-in-time candidate identity is retained independent of policy/semantic path availability"
        ),
        "months": [f"{month:%Y-%m}" for month in months],
        "warmup_days": int(args.warmup_days),
        "chunk_months": int(args.chunk_months),
        "primitive_preflight_contract": probe_fields,
        "primitive_preflight_contract_sha256": hashlib.sha256("\n".join(probe_fields).encode()).hexdigest(),
        "generation_dependencies": list(FROZEN_GENERATION_DEPENDENCIES),
        "full_feature_universe": True,
        "policy_or_semantic_sources_opened": False,
        "canonical_panel_materialiser": str(CANONICAL_PANEL_MATERIALISER),
        "canonical_panel_materialiser_sha256": _sha_file(CANONICAL_PANEL_MATERIALISER),
        "source_precedence_contract": "cell_local_15m_cache_official_legacy_v2",
        "reference_symbols": sorted({str(symbol) for symbol in args.reference_symbol}),
        "context_symbol_manifest": str(args.context_symbol_manifest) if args.context_symbol_manifest is not None else None,
        "context_symbols": list(context_symbols),
        "context_symbols_sha256": hashlib.sha256("\n".join(context_symbols).encode()).hexdigest(),
        "full_field_source": str(args.full_field_source) if args.full_field_source is not None else None,
        "full_field_source_sha256": _sha_file(args.full_field_source) if args.full_field_source is not None else None,
        "field_chunk_size": int(args.field_chunk_size) if args.field_chunk_size is not None else None,
    }
    existing_months: tuple[pd.Timestamp, ...] = ()
    if args.out.exists():
        if not args.resume:
            raise FileExistsError(f"{args.out}: immutable output root already exists")
        existing_manifest_path = args.out / "run_manifest.json"
        if not existing_manifest_path.exists():
            raise FileNotFoundError(f"{args.out}: --resume requires the original immutable run_manifest.json")
        existing_manifest = json.loads(existing_manifest_path.read_text())
        immutable_keys = [
            "schema", "scope", "incumbent_upstream", "candidate_source",
            "candidate_contract", "months", "warmup_days", "chunk_months",
            "primitive_preflight_contract_sha256", "generation_dependencies",
            "full_feature_universe", "policy_or_semantic_sources_opened", "reference_symbols",
            "canonical_panel_materialiser", "canonical_panel_materialiser_sha256",
            "source_precedence_contract",
            "context_symbol_manifest", "context_symbols_sha256",
            "full_field_source", "full_field_source_sha256", "field_chunk_size",
        ]
        mismatch = [key for key in immutable_keys if existing_manifest.get(key) != manifest.get(key)]
        if mismatch:
            raise AssertionError(f"{args.out}: --resume contract differs for {mismatch}")
        existing_months = tuple(
            month for month in months
            if (args.out / f"month={month:%Y-%m}" / "causal_feature_universe.parquet").exists()
        )
        unexpected = [
            path for path in args.out.glob("month=*")
            if path.name not in {f"month={month:%Y-%m}" for month in months}
        ]
        if unexpected:
            raise AssertionError(f"{args.out}: --resume found month panels outside the declared contract: {unexpected}")
        if (args.out / "identity_and_coverage_audit.parquet").exists() and len(existing_months) != len(months):
            raise AssertionError(f"{args.out}: final audit exists despite incomplete monthly panels")
        print(json.dumps({"event": "resume", "completed_months": [f"{month:%Y-%m}" for month in existing_months]}), flush=True)
    else:
        args.out.mkdir(parents=True)
        _exclusive_json(args.out / "run_manifest.json", manifest)

    remaining = tuple(month for month in months if month not in set(existing_months))
    audit_rows: list[dict[str, Any]] = []
    for month in existing_months:
        identities, audit = _identities(
            source_root=args.source_root, base_score_root=args.base_score_root, month=month,
        )
        completed = pd.read_parquet(args.out / f"month={month:%Y-%m}" / "causal_feature_universe.parquet")
        if len(completed) != len(identities) or completed.duplicated(IDENTITY).any():
            raise AssertionError(f"{month:%Y-%m}: resumed monthly panel does not preserve target-free identities")
        numeric_fields = [field for field in completed.columns if field not in set(IDENTITY) and pd.api.types.is_numeric_dtype(completed[field].dtype)]
        audit.update({
            "context_start": (month - pd.Timedelta(days=args.warmup_days)).isoformat(),
            "context_end_exclusive": _month_end(month).isoformat(),
            "feature_rows": int(len(completed)),
            "numeric_feature_columns": int(len(numeric_fields)),
            "target_fields_in_output": False,
            "materialisation_mode": "resumed_existing_immutable_month",
            "shared_batch": None,
        })
        audit_rows.append(audit)
    for chunk in _chunks(remaining, args.chunk_months):
        print(json.dumps({"event": "chunk_start", "months": [f"{month:%Y-%m}" for month in chunk]}), flush=True)
        audit_rows.extend(_materialise_chunk(
            out=args.out,
            source_root=args.source_root,
            base_score_root=args.base_score_root,
            months=chunk,
            warmup_days=args.warmup_days,
            frozen_contract=probe_fields,
            allow_partial_shared_recovery=args.resume,
            reference_symbols=tuple(sorted({str(symbol) for symbol in args.reference_symbol})),
            full_field_source=args.full_field_source,
            field_chunk_size=args.field_chunk_size,
            context_symbols=context_symbols,
        ))
    audit = pd.DataFrame(audit_rows).sort_values("month", kind="stable")
    audit.to_parquet(args.out / "identity_and_coverage_audit.parquet", index=False, compression="zstd")
    print(json.dumps({"event": "materialisation_complete", "months": len(months), "rows": int(audit.feature_rows.sum())}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
