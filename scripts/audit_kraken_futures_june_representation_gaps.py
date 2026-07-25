#!/usr/bin/env python3
"""Read-only audit of June short representation gaps and prior repair evidence.

This script deliberately does not fetch data, alter raw partitions, or compute
features.  A missing raw bar in a representation feature window is only an
*association*: it does not establish that a new endpoint request can recover a
valid bar.  The audit therefore combines the current overlap counts with the
already completed exact-endpoint/carry-filter evidence before making a repair
recommendation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "kraken_futures_june_representation_gap_audit_v1"
JUNE_START = pd.Timestamp("2026-06-01T00:00:00Z")
JULY_START = pd.Timestamp("2026-07-01T00:00:00Z")
RAW_COLUMNS = ("open", "high", "low", "close", "volume")
CONTEXT_COLUMNS = (
    "candidate_id",
    "__ts__",
    "__symbol__",
    "side_name",
    "gmm_representation_available",
)
EXPECTED_COUNTS = {
    "short_june_candidates": 41454,
    "short_june_unavailable_candidates": 10192,
    "short_june_available_candidates": 31262,
    "target_timestamp_complete_ohlcv": 41454,
    "unavailable_with_prior_24h_raw_gap": 10190,
    "unavailable_with_prior_48h_raw_gap": 10192,
    "associated_distinct_prior_24h_raw_gaps": 15962,
    "associated_prior_24h_raw_gap_symbols": 125,
}

DEFAULT_CONTEXT = ROOT / (
    "data_perp/artifacts/packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm/context.parquet"
)
DEFAULT_RAW_ROOT = ROOT / "data_perp/exchanges/krakenfutures"
DEFAULT_PRIOR_SCOPE = (
    ROOT
    / "data_perp/artifacts/kraken_futures_exact_source_repair_20260725_v1/scope.json"
)
DEFAULT_PRIOR_MANIFEST = (
    ROOT
    / "data_perp/artifacts/kraken_futures_exact_source_repair_20260725_v1/manifest.json"
)
DEFAULT_REVALIDATED_MANIFEST = ROOT / (
    "data_perp/artifacts/kraken_futures_exact_source_repair_20260725_v1_revalidated_carry_filtered_v2/manifest.json"
)
DEFAULT_REVALIDATED_LEDGER = ROOT / (
    "data_perp/artifacts/kraken_futures_exact_source_repair_20260725_v1_revalidated_carry_filtered_v2/accepted_candle_ledger.parquet"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/kraken_futures_june_representation_gap_audit_20260725_v1"
)


class JuneRepresentationGapAuditError(RuntimeError):
    """Raised when the immutable audit input contract is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JuneRepresentationGapAuditError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise JuneRepresentationGapAuditError(f"{name} must be a JSON object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(dict(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalise_timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise").dt.floor("h")


def _load_context(context_path: Path) -> pd.DataFrame:
    missing = set(CONTEXT_COLUMNS).difference(pd.read_parquet(context_path).columns)
    if missing:
        raise JuneRepresentationGapAuditError(
            "context is missing required columns: " + ", ".join(sorted(missing))
        )
    context = pd.read_parquet(context_path, columns=list(CONTEXT_COLUMNS)).copy()
    context["candidate_id"] = context["candidate_id"].astype(str)
    context["__ts__"] = _normalise_timestamp(context["__ts__"])
    context["__symbol__"] = context["__symbol__"].astype(str)
    context["side_name"] = context["side_name"].astype(str).str.strip().str.lower()
    availability = pd.to_numeric(
        context["gmm_representation_available"], errors="coerce"
    )
    if context["candidate_id"].duplicated().any():
        raise JuneRepresentationGapAuditError("context candidate_id must be unique")
    if not availability.isin((0.0, 1.0)).all():
        raise JuneRepresentationGapAuditError(
            "gmm_representation_available must be exactly binary"
        )
    context["gmm_representation_available"] = availability.astype(np.int8)
    return context


def _raw_symbol_dir(raw_root: Path, symbol: str) -> Path:
    return raw_root / "ohlcv" / f"symbol={symbol.replace('/', '_')}"


def _partition_overlaps(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    parts = path.stem.split("-")
    if len(parts) < 3:
        return True
    try:
        first = pd.Timestamp(int(parts[-2]), unit="s", tz="UTC")
        last = pd.Timestamp(int(parts[-1]), unit="s", tz="UTC")
    except (TypeError, ValueError, OverflowError):
        return True
    return not (last < start or first > end)


def _load_raw_ohlcv(
    raw_root: Path,
    symbol: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read raw partitions without constructing a store or writing metadata."""
    symbol_dir = _raw_symbol_dir(raw_root, symbol)
    if not symbol_dir.is_dir():
        return pd.DataFrame(columns=RAW_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    frames: list[pd.DataFrame] = []
    for path in sorted(symbol_dir.rglob("*.parquet")):
        if not _partition_overlaps(path, start=start, end=end):
            continue
        try:
            frame = pd.read_parquet(path, columns=["ts", *RAW_COLUMNS])
        except (OSError, ValueError, KeyError) as exc:
            raise JuneRepresentationGapAuditError(
                f"cannot read raw partition for {symbol}: {path}"
            ) from exc
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    raw = pd.concat(frames, ignore_index=True)
    raw["ts"] = _normalise_timestamp(raw["ts"])
    raw = raw.sort_values("ts", kind="mergesort").drop_duplicates("ts", keep="last")
    raw = raw.set_index("ts").sort_index()
    return raw.loc[(raw.index >= start) & (raw.index <= end), list(RAW_COLUMNS)]


def _complete_ohlcv_mask(raw: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.Series:
    values = raw.reindex(timestamps)
    if set(RAW_COLUMNS).difference(values.columns):
        return pd.Series(False, index=timestamps)
    finite = np.isfinite(
        values.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    )
    return pd.Series(finite.all(axis=1), index=timestamps)


def _prior_gap_tables(
    june_short: pd.DataFrame, raw_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Return observed gap overlaps; these are not a recoverability claim."""
    start = JUNE_START - pd.Timedelta(hours=48)
    end = JULY_START - pd.Timedelta(hours=1)
    target_rows: list[pd.DataFrame] = []
    gap_to_candidates: dict[tuple[str, pd.Timestamp], set[str]] = {}
    missing_without_24h: list[dict[str, Any]] = []
    unavailable_with_24h = 0
    unavailable_with_48h = 0

    for symbol, group in june_short.groupby("__symbol__", sort=True):
        hourly_index = pd.date_range(start, end, freq="h", tz="UTC")
        raw = _load_raw_ohlcv(raw_root, symbol, start=start, end=end)
        complete = _complete_ohlcv_mask(raw, hourly_index)
        gaps = hourly_index[~complete.to_numpy()]
        target_complete = complete.reindex(pd.DatetimeIndex(group["__ts__"])).to_numpy()
        target_rows.append(
            pd.DataFrame(
                {
                    "candidate_id": group["candidate_id"].to_numpy(),
                    "__ts__": group["__ts__"].to_numpy(),
                    "__symbol__": symbol,
                    "target_timestamp_complete_ohlcv": target_complete,
                }
            )
        )
        unavailable = group.loc[group["gmm_representation_available"].eq(0)]
        for candidate_id, timestamp in zip(
            unavailable["candidate_id"], unavailable["__ts__"], strict=True
        ):
            prior_24h = gaps[
                (gaps >= timestamp - pd.Timedelta(hours=24)) & (gaps < timestamp)
            ]
            prior_48h = gaps[
                (gaps >= timestamp - pd.Timedelta(hours=48)) & (gaps < timestamp)
            ]
            if len(prior_24h):
                unavailable_with_24h += 1
                for gap_timestamp in prior_24h:
                    gap_to_candidates.setdefault((symbol, gap_timestamp), set()).add(
                        candidate_id
                    )
            else:
                missing_without_24h.append(
                    {
                        "candidate_id": candidate_id,
                        "__ts__": timestamp,
                        "__symbol__": symbol,
                        "has_prior_48h_raw_gap": bool(len(prior_48h)),
                    }
                )
            if len(prior_48h):
                unavailable_with_48h += 1

    targets = pd.concat(target_rows, ignore_index=True)
    gaps = pd.DataFrame(
        [
            {
                "__symbol__": symbol,
                "missing_ohlcv_ts": timestamp,
                "affected_unavailable_candidate_count": len(candidate_ids),
            }
            for (symbol, timestamp), candidate_ids in gap_to_candidates.items()
        ]
    )
    if gaps.empty:
        gaps = pd.DataFrame(
            columns=[
                "__symbol__",
                "missing_ohlcv_ts",
                "affected_unavailable_candidate_count",
            ]
        )
    else:
        gaps = gaps.sort_values(
            ["__symbol__", "missing_ohlcv_ts"], kind="mergesort"
        ).reset_index(drop=True)
    exceptions = pd.DataFrame(missing_without_24h)
    if exceptions.empty:
        exceptions = pd.DataFrame(
            columns=["candidate_id", "__ts__", "__symbol__", "has_prior_48h_raw_gap"]
        )
    return (
        targets,
        gaps,
        {
            "unavailable_with_prior_24h_raw_gap": unavailable_with_24h,
            "unavailable_with_prior_48h_raw_gap": unavailable_with_48h,
            "unavailable_without_prior_24h_raw_gap": len(missing_without_24h),
        },
        exceptions,
    )


def _prior_exact_source_evidence(
    *,
    prior_scope_path: Path,
    prior_manifest_path: Path,
    revalidated_manifest_path: Path,
    revalidated_ledger_path: Path,
) -> dict[str, Any]:
    scope = _read_json(prior_scope_path, name="prior exact-source scope")
    prior_manifest = _read_json(prior_manifest_path, name="prior exact-source manifest")
    revalidated = _read_json(
        revalidated_manifest_path, name="revalidated carry manifest"
    )
    scope_counts = scope.get("counts")
    prior_ledger = prior_manifest.get("accepted_candle_ledger")
    accepted = revalidated.get("accepted_candle_ledger")
    if (
        not isinstance(scope_counts, Mapping)
        or not isinstance(prior_ledger, Mapping)
        or not isinstance(accepted, Mapping)
    ):
        raise JuneRepresentationGapAuditError(
            "prior exact-source evidence is structurally incomplete"
        )
    if (
        scope.get("schema") != "kraken_futures_exact_source_repair_scope_v1"
        or int(scope_counts.get("scoped_missing_source_hours", -1)) != 6917
        or int(scope_counts.get("scoped_unavailable_candidates", -1)) != 4227
        or int(prior_ledger.get("rows", -1)) != 6917
        or int(accepted.get("rows", -1)) != 94
        or revalidated.get("status") != "REVALIDATED_EXACT_SOURCE_PATCH_NOT_APPLIED"
        or int(revalidated.get("network_calls", -1)) != 0
        or int(revalidated.get("rejected_requested_zero_volume_carry_candles", -1))
        != 6823
    ):
        raise JuneRepresentationGapAuditError(
            "prior exact-source evidence does not match the reviewed pass"
        )
    if _sha256_file(revalidated_ledger_path) != str(accepted.get("sha256") or ""):
        raise JuneRepresentationGapAuditError(
            "revalidated accepted ledger hash mismatch"
        )
    ledger = pd.read_parquet(revalidated_ledger_path, columns=["ts"])
    timestamps = _normalise_timestamp(ledger["ts"])
    june_rows = int(((timestamps >= JUNE_START) & (timestamps < JULY_START)).sum())
    if len(timestamps) != 94 or june_rows != 0:
        raise JuneRepresentationGapAuditError(
            "reviewed exact-source ledger unexpectedly contains June accepted candles"
        )
    return {
        "prior_scope": {
            "path": str(prior_scope_path),
            "sha256": _sha256_file(prior_scope_path),
            "top_n_symbols": int(scope.get("selection", {}).get("top_n", -1)),
            "scoped_missing_source_hours": int(
                scope_counts["scoped_missing_source_hours"]
            ),
            "scoped_unavailable_candidates": int(
                scope_counts["scoped_unavailable_candidates"]
            ),
        },
        "prior_endpoint_manifest": {
            "path": str(prior_manifest_path),
            "sha256": _sha256_file(prior_manifest_path),
            "endpoint_response_records": int(
                prior_manifest.get("endpoint_responses", {}).get("records", -1)
            ),
            "pre_revalidation_rows": int(prior_ledger["rows"]),
        },
        "carry_filtered_revalidation": {
            "path": str(revalidated_manifest_path),
            "sha256": _sha256_file(revalidated_manifest_path),
            "accepted_rows": int(accepted["rows"]),
            "accepted_fraction_of_prior_scope": float(
                int(accepted["rows"]) / int(prior_ledger["rows"])
            ),
            "rejected_requested_zero_volume_carry_candles": int(
                revalidated["rejected_requested_zero_volume_carry_candles"]
            ),
            "accepted_june_window_rows": june_rows,
            "accepted_timestamp_min": timestamps.min(),
            "accepted_timestamp_max": timestamps.max(),
            "ledger_path": str(revalidated_ledger_path),
            "ledger_sha256": _sha256_file(revalidated_ledger_path),
        },
    }


def build_audit(
    *,
    context_path: Path,
    raw_root: Path,
    prior_scope_path: Path,
    prior_manifest_path: Path,
    revalidated_manifest_path: Path,
    revalidated_ledger_path: Path,
    expected_counts: Mapping[str, int] | None = EXPECTED_COUNTS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Build a deterministic observation-only audit, without persistent writes."""
    context = _load_context(context_path)
    june_short = context.loc[
        context["side_name"].eq("short")
        & context["__ts__"].ge(JUNE_START)
        & context["__ts__"].lt(JULY_START)
    ].copy()
    if june_short.empty:
        raise JuneRepresentationGapAuditError("June short context is empty")
    targets, gap_table, gap_counts, exceptions = _prior_gap_tables(june_short, raw_root)
    unavailable = june_short["gmm_representation_available"].eq(0)
    observed = {
        "short_june_candidates": int(len(june_short)),
        "short_june_available_candidates": int((~unavailable).sum()),
        "short_june_unavailable_candidates": int(unavailable.sum()),
        "short_june_representation_coverage": float((~unavailable).mean()),
        "target_timestamp_complete_ohlcv": int(
            targets["target_timestamp_complete_ohlcv"].sum()
        ),
        "target_timestamp_missing_or_nonfinite_ohlcv": int(
            (~targets["target_timestamp_complete_ohlcv"]).sum()
        ),
        "associated_distinct_prior_24h_raw_gaps": int(len(gap_table)),
        "associated_prior_24h_raw_gap_symbols": int(gap_table["__symbol__"].nunique()),
        **gap_counts,
    }
    if expected_counts is not None:
        drift = {
            key: {"expected": int(value), "observed": int(observed.get(key, -1))}
            for key, value in expected_counts.items()
            if int(observed.get(key, -1)) != int(value)
        }
        if drift:
            raise JuneRepresentationGapAuditError(
                "audit drifted from its frozen baseline counts: "
                + json.dumps(drift, sort_keys=True)
            )
    prior_evidence = _prior_exact_source_evidence(
        prior_scope_path=prior_scope_path,
        prior_manifest_path=prior_manifest_path,
        revalidated_manifest_path=revalidated_manifest_path,
        revalidated_ledger_path=revalidated_ledger_path,
    )
    report = {
        "schema": SCHEMA,
        "status": "READ_ONLY_NO_BROAD_BACKFILL_RECOMMENDED",
        "network_calls": 0,
        "baseline_raw_store_mutated": False,
        "feature_or_model_recomputed": False,
        "analysis_window": {"start": JUNE_START, "end_exclusive": JULY_START},
        "context": {"path": str(context_path), "sha256": _sha256_file(context_path)},
        "raw_root": str(raw_root),
        "observation": observed,
        "prior_exact_source_evidence": prior_evidence,
        "interpretation": {
            "preceding_raw_gap_overlap_is_recoverability_evidence": False,
            "reason": "The reviewed exact endpoint pass already sampled the top-30 missing symbols, and carry filtering retained only 94/6917 rows (1.36%) with zero accepted June-window rows.",
            "recommendation": "Do not request a broad backfill. A second sample is justified only with a materially independent exact source and a predeclared bounded test that rejects carry/zero-volume rows before any recomputation.",
        },
    }
    return report, gap_table, exceptions


def write_audit_artifact(
    *,
    destination: Path,
    report: Mapping[str, Any],
    gap_table: pd.DataFrame,
    exceptions: pd.DataFrame,
) -> Path:
    """Atomically publish a new audit artifact; never overwrite prior evidence."""
    if destination.exists():
        raise JuneRepresentationGapAuditError(
            f"refusing to overwrite artifact: {destination}"
        )
    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        stage.mkdir(parents=True, exist_ok=False)
        gap_path = stage / "associated_prior_24h_raw_gaps.parquet"
        exception_path = stage / "unavailable_without_prior_24h_gap.parquet"
        gap_table.to_parquet(gap_path, index=False)
        exceptions.to_parquet(exception_path, index=False)
        manifest = dict(report)
        manifest["outputs"] = {
            "associated_prior_24h_raw_gaps": {
                "path": gap_path.name,
                "rows": int(len(gap_table)),
                "sha256": _sha256_file(gap_path),
            },
            "unavailable_without_prior_24h_gap": {
                "path": exception_path.name,
                "rows": int(len(exceptions)),
                "sha256": _sha256_file(exception_path),
            },
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--prior-scope", type=Path, default=DEFAULT_PRIOR_SCOPE)
    parser.add_argument("--prior-manifest", type=Path, default=DEFAULT_PRIOR_MANIFEST)
    parser.add_argument(
        "--revalidated-manifest", type=Path, default=DEFAULT_REVALIDATED_MANIFEST
    )
    parser.add_argument(
        "--revalidated-ledger", type=Path, default=DEFAULT_REVALIDATED_LEDGER
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-count-drift",
        action="store_true",
        help="allow a noncanonical input fixture rather than enforcing the frozen production counts",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    report, gaps, exceptions = build_audit(
        context_path=args.context,
        raw_root=args.raw_root,
        prior_scope_path=args.prior_scope,
        prior_manifest_path=args.prior_manifest,
        revalidated_manifest_path=args.revalidated_manifest,
        revalidated_ledger_path=args.revalidated_ledger,
        expected_counts=None if args.allow_count_drift else EXPECTED_COUNTS,
    )
    destination = write_audit_artifact(
        destination=args.output,
        report=report,
        gap_table=gaps,
        exceptions=exceptions,
    )
    print(
        json.dumps(
            {"status": report["status"], "artifact": str(destination)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
