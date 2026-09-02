#!/usr/bin/env python3
"""Materialise a target-free historical P8U successor feature panel.

The retained source panel is replayed once through the same canonical feature
adapter used by the one-timestamp inference-reference producer.  The output is
chunked by decision hour so downstream walk-forward training can stream it
without reconstructing feature history.  It consumes no outcome, policy,
portfolio, score, exchange, or order input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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


IDENTITY = ("candidate_id", "__decision_ts__", "side_name", "__ts__", "__symbol__")
FORBIDDEN = ("future_", "outcome", "policy_net", "label_available", "exact_net", "gross_net")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: Any) -> pd.Series | pd.Timestamp:
    if isinstance(values, pd.Series):
        return pd.to_datetime(values, utc=True, errors="raise")
    value = pd.Timestamp(values)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _forbid_columns(frame: pd.DataFrame, *, name: str) -> None:
    violations = [
        str(column) for column in frame.columns
        if any(token in str(column).lower() for token in FORBIDDEN)
    ]
    if violations:
        raise AssertionError(f"{name} is not target-free: {violations[:8]}")


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _feature_values(
    feature: pd.DataFrame, *, timestamps: pd.Series, symbols: pd.Series,
) -> np.ndarray:
    row_index = feature.index.get_indexer(pd.DatetimeIndex(timestamps))
    column_index = feature.columns.get_indexer(symbols.astype(str))
    if (row_index < 0).any() or (column_index < 0).any():
        raise AssertionError("canonical adapter omitted a requested candidate coordinate")
    return feature.to_numpy(copy=False)[row_index, column_index].astype(np.float32, copy=False)


def _reference_parity(
    *, reference: Path, final_part: Path, fields: tuple[str, ...], timestamp: pd.Timestamp,
) -> dict[str, Any]:
    expected = pd.read_parquet(reference).copy()
    actual = pd.read_parquet(final_part).copy()
    for frame, name in ((expected, "reference"), (actual, "materialised")):
        frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
        frame["__symbol__"] = frame["__symbol__"].astype(str)
    expected = expected.loc[expected["__decision_ts__"].eq(timestamp)].sort_values(
        "__symbol__", kind="stable"
    ).reset_index(drop=True)
    actual = actual.loc[actual["__decision_ts__"].eq(timestamp)].sort_values(
        "__symbol__", kind="stable"
    ).reset_index(drop=True)
    if len(expected) != len(actual) or not expected["__symbol__"].equals(actual["__symbol__"]):
        raise AssertionError("final successor feature part does not match reference universe")
    max_delta = 0.0
    compared = 0
    for field in fields:
        left = pd.to_numeric(expected[field], errors="coerce").to_numpy(float)
        right = pd.to_numeric(actual[field], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            raise AssertionError(f"reference parity differs in missingness for {field}")
        delta = np.nanmax(np.abs(left - right)) if np.isfinite(left).any() else 0.0
        if not np.isfinite(delta) or float(delta) > 1e-6:
            raise AssertionError(f"reference parity exceeds 1e-6 for {field}: {delta}")
        max_delta = max(max_delta, float(delta))
        compared += int(np.isfinite(left).sum())
    return {"reference_rows": int(len(expected)), "numeric_values_compared": compared, "max_abs_delta": max_delta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--reference-features", type=Path,
        help=(
            "One-timestamp canonical reference used for mandatory parity in a "
            "normal successor panel.  Omit only with --allow-no-reference-parity "
            "for a separately sealed historical warm-up panel whose interval "
            "cannot contain the current inference reference."
        ),
    )
    parser.add_argument(
        "--allow-no-reference-parity", action="store_true",
        help=(
            "Explicitly permit a separately auditable historical-only panel "
            "without a contemporaneous inference reference.  This never claims "
            "reference parity and is rejected unless the flag is present."
        ),
    )
    parser.add_argument("--chunk-hours", type=int, default=168)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if int(args.chunk_hours) < 1:
        raise ValueError("chunk-hours must be positive")
    if args.reference_features is None and not args.allow_no_reference_parity:
        raise ValueError(
            "--reference-features is required unless this is an explicit "
            "separate historical warm-up panel with --allow-no-reference-parity"
        )

    plan = json.loads(args.feature_plan.read_text(encoding="utf-8"))
    fields = tuple(map(str, plan.get("full_union") or ()))
    if not fields or len(fields) != len(set(fields)):
        raise AssertionError("feature plan must declare a unique nonempty union")
    candidates = pd.read_parquet(args.candidates, columns=list(IDENTITY)).copy()
    _forbid_columns(candidates, name="candidate ledger")
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    candidates["__decision_ts__"] = _utc(candidates["__decision_ts__"])
    candidates["__ts__"] = _utc(candidates["__ts__"])
    candidates["__symbol__"] = candidates["__symbol__"].astype(str)
    candidates["side_name"] = candidates["side_name"].astype(str).str.lower()
    if candidates["candidate_id"].duplicated().any() or not candidates["side_name"].eq("long").all():
        raise AssertionError("candidate identities must be unique and long-only")
    if not candidates["__decision_ts__"].eq(candidates["__ts__"] + pd.Timedelta(hours=1)).all():
        raise AssertionError("candidate decision timestamps do not bind to completed source hours")

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    symbols = tuple(sorted(map(str, source_manifest.get("symbols") or source_manifest.get("source_map", {}).keys())))
    if len(symbols) != 160 or candidates["__symbol__"].nunique() != 160:
        raise AssertionError("successor panel requires exactly the frozen 160-symbol universe")
    if set(candidates["__symbol__"]) != set(symbols):
        raise AssertionError("candidate and frozen source universes differ")

    loaded = joblib.load(args.source_panel)
    panel = loaded.get("panel") if isinstance(loaded, Mapping) else None
    close = panel.get("close") if isinstance(panel, Mapping) else None
    if not isinstance(close, pd.DataFrame):
        raise AssertionError("source state lacks an append-only close panel")
    source_index = pd.DatetimeIndex(close.index)
    candidate_source_ts = pd.DatetimeIndex(candidates["__ts__"])
    if not candidate_source_ts.isin(source_index).all():
        raise AssertionError("candidate source timestamps are absent from the source panel")

    started = time.monotonic()
    generated = canonical_features_from_saved_panel(
        panel, universe_symbols=symbols, requested_features=fields,
        full_config_causal_universe=True,
    )
    temporary = args.out_dir.with_name(f".{args.out_dir.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    parts_dir = temporary / "features"
    parts_dir.mkdir()
    timestamps = pd.DatetimeIndex(sorted(candidates["__decision_ts__"].unique()))
    parts: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    final_part: Path | None = None
    for number, start in enumerate(range(0, len(timestamps), int(args.chunk_hours))):
        chosen = timestamps[start:start + int(args.chunk_hours)]
        chunk = candidates.loc[candidates["__decision_ts__"].isin(chosen)].copy()
        chunk = chunk.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        for field in fields:
            source = generated.get(field)
            if not isinstance(source, pd.DataFrame):
                raise AssertionError(f"canonical graph did not materialise {field}")
            values = _feature_values(source, timestamps=chunk["__ts__"], symbols=chunk["__symbol__"])
            chunk[field] = values
            coverage_rows.append({
                "part": number, "field": field, "rows": int(len(values)),
                "finite_fraction": float(np.isfinite(values).mean()),
            })
        path = parts_dir / f"part_{number:04d}.parquet"
        chunk.to_parquet(path, index=False, compression="zstd")
        parts.append({
            "name": path.name, "rows": int(len(chunk)), "first_timestamp": chosen.min().isoformat(),
            "last_timestamp": chosen.max().isoformat(), "sha256": _sha256(path),
        })
        final_part = path
    if final_part is None:
        raise AssertionError("no successor feature parts were written")
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_parquet(temporary / "feature_coverage.parquet", index=False, compression="zstd")
    if args.reference_features is not None:
        reference = pd.read_parquet(args.reference_features, columns=["__decision_ts__"]).copy()
        reference_ts = _utc(reference["__decision_ts__"]).max()
        part_for_reference = next(
            parts_dir / str(part["name"])
            for part in parts if pd.Timestamp(part["first_timestamp"]) <= reference_ts <= pd.Timestamp(part["last_timestamp"])
        )
        parity = _reference_parity(
            reference=args.reference_features, final_part=part_for_reference,
            fields=fields, timestamp=reference_ts,
        )
        reference_audit: dict[str, Any] = {
            "status": "pass", "path": str(args.reference_features.resolve()),
            "sha256": _sha256(args.reference_features), "detail": parity,
        }
    else:
        reference_audit = {
            "status": "not_applicable_separate_historical_window",
            "reason": (
                "This sealed warm-up interval predates the current inference "
                "reference.  No cross-window parity claim is made."
            ),
        }
    _write_once(temporary / "run_manifest.json", {
        "schema": "strict_r3_p8u_successor_feature_panel_v1",
        "status": "complete_target_free",
        "scope": "offline canonical feature materialisation; no target/outcome/policy/portfolio/exchange/order input",
        "candidate_rows": int(len(candidates)), "timestamps": int(len(timestamps)), "symbols": len(symbols),
        "feature_fields": len(fields), "parts": parts,
        "source_panel": str(args.source_panel.resolve()), "source_panel_sha256": _sha256(args.source_panel),
        "candidates": str(args.candidates.resolve()), "candidates_sha256": _sha256(args.candidates),
        "feature_plan": str(args.feature_plan.resolve()), "feature_plan_sha256": _sha256(args.feature_plan),
        "source_manifest": str(args.source_manifest.resolve()), "source_manifest_sha256": _sha256(args.source_manifest),
        "reference_parity": reference_audit,
        "runtime_seconds": time.monotonic() - started,
    })
    os.replace(temporary, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
