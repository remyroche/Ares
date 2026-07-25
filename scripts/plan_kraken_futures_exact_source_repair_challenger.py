#!/usr/bin/env python3
"""Freeze, but never execute, the bounded Kraken source-repair challenger.

This planner is the bridge between the 94-candle, carry-filtered Kraken patch
and a later *separate* challenger recomputation.  It intentionally does not
write an OHLCV or feature-store row.  In particular, it does not copy, link,
compact, or append to either canonical store.

The plan has three purposes:

* prove that the accepted patch is the revalidated 94-row ledger, rather than
  the unsafe 6,917-row v1 ledger;
* freeze the full candidate scope which must be re-materialized (both sides,
  including rows currently available, because a causal rolling feature can
  change without becoming missing); and
* state the only safe hand-off: copy only affected raw partitions to a raw
  challenger, recompute a delta-only raw-feature overlay, then compose a new
  context from that overlay plus the immutable baseline.  No hard-linked
  mutable feature files and no synthetic fills are permitted.

The actual feature computation remains deliberately separate.  It must first
prove numerical parity with the baseline on deterministic clean rows before it
may publish a repaired context.  This makes a code/configuration drift fail
closed rather than silently calling a fresh feature implementation a repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import (  # noqa: E402
    candidate_identity_sha256,
)
from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    TrainingResourceGuard,
    TrainingResourceLimits,
)

SCHEMA = "kraken_futures_exact_source_repair_challenger_plan_v1"
PATCH_SCHEMA = "kraken_futures_exact_source_repair_revalidated_patch_v1"
PATCH_STATUS = "REVALIDATED_EXACT_SOURCE_PATCH_NOT_APPLIED"
EXPECTED_LEDGER_ROWS = 94
EXPECTED_LEDGER_SHA256 = (
    "4c611a5721f4b93ee02f755f8ec2067b54ed51ee1fc388b925b2e5a0016399b4"
)
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")
LEDGER_COLUMNS = (
    "symbol",
    "product_id",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

DEFAULT_PATCH_ROOT = ROOT / (
    "data_perp/artifacts/"
    "kraken_futures_exact_source_repair_20260725_v1_revalidated_carry_filtered_v2"
)
DEFAULT_CONTEXT_ROOT = ROOT / (
    "data_perp/artifacts/packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm"
)
DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_RAW_ROOT = ROOT / "data_perp/exchanges/krakenfutures"
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/kraken_futures_exact_source_repair_challenger_plan_20260725_v1"
)


class KrakenRepairChallengerPlanError(RuntimeError):
    """Raised when an immutable repair challenger cannot be proven safe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KrakenRepairChallengerPlanError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise KrakenRepairChallengerPlanError(f"{name} must be a JSON object: {path}")
    return value


def _normalise_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(LEDGER_COLUMNS).difference(ledger.columns))
    if missing:
        raise KrakenRepairChallengerPlanError(
            "accepted ledger is missing columns: " + ", ".join(missing)
        )
    output = ledger.loc[:, list(LEDGER_COLUMNS)].copy()
    output["symbol"] = output["symbol"].astype(str)
    output["product_id"] = output["product_id"].astype(str)
    output["ts"] = pd.to_datetime(output["ts"], utc=True, errors="raise").dt.floor("h")
    if (
        output["symbol"].eq("").any()
        or output["product_id"].eq("").any()
        or output.duplicated(["symbol", "ts"]).any()
    ):
        raise KrakenRepairChallengerPlanError(
            "accepted ledger has blank identities or duplicate symbol/timestamp rows"
        )
    numeric = output.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_ohlcv = (
        np.isfinite(numeric.to_numpy(dtype=np.float64, copy=False)).all()
        and (numeric[["open", "high", "low", "close"]] > 0.0).all(axis=None)
        and (numeric["volume"] >= 0.0).all()
        and (numeric["low"] <= numeric[["open", "close"]].min(axis=1)).all()
        and (numeric["high"] >= numeric[["open", "close"]].max(axis=1)).all()
        and (numeric["low"] <= numeric["high"]).all()
    )
    if not valid_ohlcv:
        raise KrakenRepairChallengerPlanError("accepted ledger has invalid exact OHLCV")
    output.loc[:, numeric.columns] = numeric.astype(np.float32)
    return output.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)


def _validate_patch(
    patch_root: Path,
    *,
    expected_rows: int | None = EXPECTED_LEDGER_ROWS,
    expected_ledger_sha256: str | None = EXPECTED_LEDGER_SHA256,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    manifest_path = patch_root / "manifest.json"
    ledger_path = patch_root / "accepted_candle_ledger.parquet"
    manifest = _read_json(manifest_path, name="revalidated patch manifest")
    if (
        manifest.get("schema") != PATCH_SCHEMA
        or manifest.get("status") != PATCH_STATUS
        or manifest.get("baseline_raw_store_mutated") is not False
        or manifest.get("network_calls") != 0
        or manifest.get("synthetic_fill") is not False
    ):
        raise KrakenRepairChallengerPlanError(
            "only the immutable, offline-revalidated, not-applied patch is accepted"
        )
    meta = manifest.get("accepted_candle_ledger")
    if not isinstance(meta, Mapping) or not ledger_path.is_file():
        raise KrakenRepairChallengerPlanError("revalidated patch ledger is missing")
    observed_hash = _sha256_file(ledger_path)
    if observed_hash != str(meta.get("sha256") or ""):
        raise KrakenRepairChallengerPlanError("revalidated patch ledger hash mismatch")
    if expected_ledger_sha256 is not None and observed_hash != expected_ledger_sha256:
        raise KrakenRepairChallengerPlanError(
            "refusing any ledger other than the approved carry-filtered v2 ledger"
        )
    ledger = _normalise_ledger(pd.read_parquet(ledger_path))
    if int(meta.get("rows", -1)) != len(ledger):
        raise KrakenRepairChallengerPlanError(
            "revalidated patch ledger row count mismatch"
        )
    if expected_rows is not None and len(ledger) != int(expected_rows):
        raise KrakenRepairChallengerPlanError(
            "revalidated patch is outside the approved 94-row bound"
        )
    return ledger, manifest, ledger_path


def _validate_context(context_root: Path) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    context_path = context_root / "context.parquet"
    manifest_path = context_root / "manifest.json"
    manifest = _read_json(manifest_path, name="canonical context manifest")
    if (
        manifest.get("status")
        != "MATERIALIZED_CANONICAL_CONTEXT_WITH_FROZEN_SIDE_AE_GMM"
    ):
        raise KrakenRepairChallengerPlanError(
            "frozen AE/GMM downstream context is required"
        )
    output = manifest.get("output")
    if not isinstance(output, Mapping) or str(
        output.get("sha256") or ""
    ) != _sha256_file(context_path):
        raise KrakenRepairChallengerPlanError(
            "canonical context hash does not match manifest"
        )
    required = [*IDENTITY_COLUMNS, "gmm_representation_available"]
    context = pd.read_parquet(context_path, columns=required)
    context["__ts__"] = pd.to_datetime(context["__ts__"], utc=True, errors="raise")
    context["__symbol__"] = context["__symbol__"].astype(str)
    context["side_name"] = context["side_name"].astype(str).str.strip().str.lower()
    context["candidate_id"] = context["candidate_id"].astype(str)
    context["gmm_representation_available"] = pd.to_numeric(
        context["gmm_representation_available"], errors="coerce"
    ).astype(np.float32)
    identity = candidate_identity_sha256(context, columns=IDENTITY_COLUMNS)
    if (
        len(context) != int(output.get("rows", -1))
        or context["candidate_id"].duplicated().any()
        or set(context["side_name"]) != {"long", "short"}
        or identity != str(output.get("candidate_identity_sha256") or "")
        or ~context["gmm_representation_available"].isin((0.0, 1.0)).all()
    ):
        raise KrakenRepairChallengerPlanError(
            "canonical context identity or representation-availability contract changed"
        )
    return context, manifest, context_path


def _feature_contract_evidence(ae_root: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    union: set[str] = set()
    for side in ("long", "short"):
        path = ae_root / side / "loader_evidence" / "frozen_feature_contract.json"
        payload = _read_json(path, name=f"{side} frozen raw feature contract")
        features = payload.get("feature_columns")
        if not isinstance(features, list) or not features:
            raise KrakenRepairChallengerPlanError(
                f"{side} frozen raw feature contract has no feature columns"
            )
        names = tuple(map(str, features))
        if len(set(names)) != len(names):
            raise KrakenRepairChallengerPlanError(
                f"{side} frozen raw feature contract duplicates feature names"
            )
        expected = str(payload.get("feature_contract_sha256") or "")
        if len(expected) != 64:
            raise KrakenRepairChallengerPlanError(
                f"{side} frozen raw feature contract hash is invalid"
            )
        union.update(names)
        evidence[side] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "feature_contract_sha256": expected,
            "feature_count": len(names),
            "feature_columns": list(names),
        }
    return {
        "by_side": evidence,
        "union_feature_count": len(union),
        "union_feature_columns": sorted(union),
    }


def _assert_patch_is_absent_from_baseline_raw(
    ledger: pd.DataFrame, *, raw_root: Path
) -> dict[str, Any]:
    if not raw_root.is_dir():
        raise KrakenRepairChallengerPlanError(
            f"baseline raw root is missing: {raw_root}"
        )
    store = PartitionedOHLCVStore(str(raw_root), "1h")
    by_symbol: dict[str, Any] = {}
    for symbol, group in ledger.groupby("symbol", sort=True, observed=True):
        timestamps = pd.DatetimeIndex(group["ts"])
        local = store.load(
            str(symbol),
            start_ts=timestamps.min(),
            end_ts=timestamps.max(),
        )
        observed = pd.DatetimeIndex(pd.to_datetime(local.index, utc=True))
        overlap = observed.intersection(timestamps)
        if len(overlap):
            raise KrakenRepairChallengerPlanError(
                "baseline raw already has an accepted source timestamp for "
                f"{symbol}: {', '.join(map(str, overlap[:3]))}"
            )
        raw_dir = Path(store._get_symbol_dir(str(symbol)))
        if not raw_dir.is_dir():
            raise KrakenRepairChallengerPlanError(
                f"baseline raw partition is missing for {symbol!r}"
            )
        # This is a small (17-symbol) source-surface binding. Hashing it is
        # important: the later COW copy must start from exactly this baseline.
        records = []
        for path in sorted(
            candidate for candidate in raw_dir.rglob("*") if candidate.is_file()
        ):
            records.append(
                {
                    "relative_path": str(path.relative_to(raw_root)),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )
        by_symbol[str(symbol)] = {
            "accepted_rows": int(len(group)),
            "baseline_raw_dir": str(raw_dir),
            "baseline_raw_tree_sha256": _canonical_json_sha256({"files": records}),
            "baseline_raw_files": len(records),
            "baseline_raw_bytes": int(sum(record["bytes"] for record in records)),
            "baseline_accepted_timestamp_overlap": 0,
        }
    return by_symbol


def _feature_surface_snapshot(
    symbols: Sequence[str], *, feature_store: Path
) -> dict[str, Any]:
    if not feature_store.is_dir():
        raise KrakenRepairChallengerPlanError(
            f"baseline feature store is missing: {feature_store}"
        )
    records: list[dict[str, Any]] = []
    for symbol in sorted(set(map(str, symbols))):
        path = feature_store / f"symbol={symbol.replace('/', '_')}.parquet"
        if not path.is_file():
            raise KrakenRepairChallengerPlanError(
                f"baseline feature file is missing for {symbol!r}: {path}"
            )
        # Deliberately bind the canonical base files here. Delta/duckdb sidecars
        # are never hard-linked into the challenger: they may be mutable.
        records.append(
            {
                "symbol": symbol,
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return {
        "path": str(feature_store),
        "mode": "immutable_baseline_read_only_plus_separate_delta_only_overlay",
        "affected_base_files": records,
        "affected_base_file_count": len(records),
        "affected_base_bytes": int(sum(item["bytes"] for item in records)),
        "affected_base_surface_sha256": _canonical_json_sha256({"files": records}),
        "hard_links_for_mutable_feature_files": False,
    }


def _scoped_candidates(context: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    patch_symbols = set(ledger["symbol"].astype(str))
    first_patch_ts = pd.Timestamp(ledger["ts"].min())
    # A feature at t may consume an exact source candle at t.  Retain both
    # available and unavailable rows so rolling-value drift cannot escape the
    # repaired context merely because it did not manifest as a NaN.
    scope = context.loc[
        context["__symbol__"].isin(patch_symbols) & context["__ts__"].ge(first_patch_ts)
    ].copy()
    if scope.empty:
        raise KrakenRepairChallengerPlanError(
            "no downstream candidates occur at or after the accepted repair window"
        )
    scope["baseline_gmm_representation_available"] = scope[
        "gmm_representation_available"
    ].astype(np.float32)
    scope["repair_scope_reason"] = "same_symbol_after_exact_source_patch"
    return (
        scope.loc[
            :,
            [
                "__ts__",
                "__symbol__",
                "side_name",
                "candidate_id",
                "baseline_gmm_representation_available",
                "repair_scope_reason",
            ],
        ]
        .sort_values(["__ts__", "__symbol__", "side_name"], kind="mergesort")
        .reset_index(drop=True)
    )


def _scope_metrics(scope: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "rows": int(len(scope)),
        "candidate_identity_sha256": candidate_identity_sha256(
            scope, columns=IDENTITY_COLUMNS
        ),
    }
    by_side: dict[str, Any] = {}
    for side, frame in scope.groupby("side_name", sort=True, observed=True):
        months: dict[str, Any] = {}
        for month, monthly in frame.groupby(
            frame["__ts__"].dt.strftime("%Y-%m"), sort=True
        ):
            unavailable = int(
                pd.to_numeric(
                    monthly["baseline_gmm_representation_available"], errors="coerce"
                )
                .eq(0.0)
                .sum()
            )
            months[str(month)] = {
                "rows": int(len(monthly)),
                "baseline_unavailable_rows": unavailable,
                "baseline_available_rows": int(len(monthly) - unavailable),
            }
        unavailable = int(
            pd.to_numeric(
                frame["baseline_gmm_representation_available"], errors="coerce"
            )
            .eq(0.0)
            .sum()
        )
        by_side[str(side)] = {
            "rows": int(len(frame)),
            "baseline_unavailable_rows": unavailable,
            "baseline_available_rows": int(len(frame) - unavailable),
            "months": months,
        }
    metrics["by_side"] = by_side
    return metrics


def build_plan(
    *,
    patch_root: Path,
    context_root: Path,
    ae_root: Path,
    raw_root: Path,
    feature_store: Path,
    destination: Path,
    expected_rows: int | None = EXPECTED_LEDGER_ROWS,
    expected_ledger_sha256: str | None = EXPECTED_LEDGER_SHA256,
) -> dict[str, Any]:
    """Publish a no-mutation challenger plan and exact candidate scope."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite challenger plan: {destination}")
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=destination.parent,
        telemetry_path=stage / "training_resource_telemetry.jsonl",
    )
    try:
        guard.preflight("kraken_repair_challenger_plan:preflight")
        ledger, patch_manifest, ledger_path = _validate_patch(
            Path(patch_root),
            expected_rows=expected_rows,
            expected_ledger_sha256=expected_ledger_sha256,
        )
        guard.checkpoint("kraken_repair_challenger_plan:patch_validated")
        context, context_manifest, context_path = _validate_context(Path(context_root))
        scope = _scoped_candidates(context, ledger)
        raw_evidence = _assert_patch_is_absent_from_baseline_raw(
            ledger, raw_root=Path(raw_root)
        )
        guard.checkpoint("kraken_repair_challenger_plan:raw_surface_bound")
        contracts = _feature_contract_evidence(Path(ae_root))
        feature_evidence = _feature_surface_snapshot(
            ledger["symbol"].astype(str).tolist(), feature_store=Path(feature_store)
        )
        guard.checkpoint("kraken_repair_challenger_plan:feature_surface_bound")
        ledger_output = stage / "accepted_candle_ledger.parquet"
        # Keep a byte-identical copy so the published plan remains cryptographically
        # bound to the validated ledger, not merely to a dataframe round-trip.
        shutil.copy2(ledger_path, ledger_output)
        scope_output = stage / "candidate_recompute_scope.parquet"
        scope.to_parquet(
            scope_output, index=False, compression="zstd", compression_level=5
        )
        max_candidate_ts = pd.Timestamp(scope["__ts__"].max())
        first_patch_ts = pd.Timestamp(ledger["ts"].min())
        result = {
            "schema": SCHEMA,
            "status": "PLANNED_NOT_COMPUTED_NO_BASELINE_MUTATION",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_artifacts_mutated": False,
            "synthetic_fill": False,
            "repair_patch": {
                "path": str(Path(patch_root)),
                "manifest_sha256": _sha256_file(Path(patch_root) / "manifest.json"),
                "accepted_ledger_source_path": str(ledger_path),
                "accepted_ledger_source_sha256": _sha256_file(ledger_path),
                "accepted_ledger_path": str(destination / ledger_output.name),
                "accepted_ledger_sha256": _sha256_file(ledger_output),
                "accepted_rows": int(len(ledger)),
                "symbols": sorted(ledger["symbol"].astype(str).unique().tolist()),
                "first_patch_ts": first_patch_ts,
                "last_patch_ts": pd.Timestamp(ledger["ts"].max()),
                "revalidated_patch_status": patch_manifest["status"],
            },
            "baseline_context": {
                "path": str(context_path),
                "sha256": _sha256_file(context_path),
                "manifest_sha256": _sha256_file(Path(context_root) / "manifest.json"),
                "candidate_identity_sha256": context_manifest["output"][
                    "candidate_identity_sha256"
                ],
            },
            "candidate_recompute_scope": {
                "path": str(destination / scope_output.name),
                "sha256": _sha256_file(scope_output),
                "selection": "all sides/all rows for patched symbols at_or_after_first_patch_ts",
                "why_not_only_missing": (
                    "causal rolling features can change on baseline-available rows; "
                    "recompute both availability states before an exact-row comparison"
                ),
                **_scope_metrics(scope),
            },
            "raw_challenger": {
                "baseline_root": str(raw_root),
                "mode": "copy_only_affected_symbol_partitions_then_exact_ledger_merge",
                "hard_links": False,
                "exact_insert_only": list(LEDGER_COLUMNS),
                "precondition": "every accepted ledger timestamp is absent from the baseline raw partition",
                "evidence_by_symbol": raw_evidence,
                "forbidden": [
                    "mutating_baseline_raw",
                    "interpolation",
                    "forward_fill",
                    "backfill_fill",
                    "synthetic_candles",
                    "network_requests",
                ],
            },
            "feature_challenger": {
                **feature_evidence,
                "raw_feature_contracts": contracts,
                "mode": "separate_delta_only_feature_overlay_read_with_immutable_baseline_fallback",
                "hard_links_for_mutable_feature_files": False,
                "recompute_output_window": {
                    "start_ts_inclusive": first_patch_ts,
                    "end_ts_inclusive": max_candidate_ts,
                    "minimum_causal_warmup_hours": 24 * 120,
                    "why": "match the existing incremental feature pipeline warmup; do not infer a shorter window from feature names",
                },
                "mandatory_prepublication_parity": {
                    "description": "with the patch disabled, recompute deterministic clean scoped rows and require exact baseline raw-feature parity before applying the patch",
                    "scope": "both side contracts; deterministic rows currently representation-available; same symbols and timestamps",
                    "on_failure": "fail_closed_no_repaired_context",
                },
                "forbidden": [
                    "copying_or_mutating_the_24GiB_baseline_feature_store",
                    "hard_linking_duckdb_or_parquet_delta_sidecars",
                    "filling_missing_raw_or_feature_values",
                    "rewriting_unaffected_feature_rows",
                ],
            },
            "context_composition": {
                "mode": "copy_baseline_context_then_replace_only_same_candidate_id_rows_from_recomputed_overlay",
                "candidate_identity_must_match": context_manifest["output"][
                    "candidate_identity_sha256"
                ],
                "frozen_ae_gmm_state_reused": True,
                "retrain_ae_gmm": False,
                "required_comparisons": [
                    "baseline_vs_challenger_per_side_month_representation_availability",
                    "baseline_vs_challenger_exact_candidate_identity",
                    "baseline_vs_challenger_raw_feature_parity_on_clean_control_rows",
                    "baseline_vs_challenger_OOF_metrics_before_any_promotion",
                ],
            },
            "resource_guard": {
                "max_process_rss_bytes": 12 * 1024**3,
                "min_free_ram_bytes": 2 * 1024**3,
                "min_free_disk_bytes": 10 * 1024**3,
                "telemetry": str(guard.telemetry_path),
                "estimated_physical_raw_copy_bytes": int(
                    sum(item["baseline_raw_bytes"] for item in raw_evidence.values())
                ),
                "estimated_physical_feature_base_bytes_if_full_files_copied": int(
                    feature_evidence["affected_base_bytes"]
                ),
                "full_feature_file_copy_permitted": False,
            },
            "promotion": {
                "status": "NOT_A_PROMOTION_ARTIFACT",
                "condition": "only consider the challenger after independently validated exact-row OOF metric comparison",
            },
        }
        _write_json(stage / "manifest.json", result)
        guard.checkpoint("kraken_repair_challenger_plan:complete")
        os.replace(stage, destination)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, default=DEFAULT_PATCH_ROOT)
    parser.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-noncanonical-ledger-for-test",
        action="store_true",
        help="Test-only: do not enforce the production 94-row ledger digest.",
    )
    args = parser.parse_args()
    result = build_plan(
        patch_root=args.patch_root,
        context_root=args.context_root,
        ae_root=args.ae_root,
        raw_root=args.raw_root,
        feature_store=args.feature_store,
        destination=args.output_dir,
        expected_rows=None
        if args.allow_noncanonical_ledger_for_test
        else EXPECTED_LEDGER_ROWS,
        expected_ledger_sha256=(
            None if args.allow_noncanonical_ledger_for_test else EXPECTED_LEDGER_SHA256
        ),
    )
    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
