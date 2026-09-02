#!/usr/bin/env python3
"""Materialise the strict-R3 upstream heads without consuming outcomes.

The lock-step history persisted only base and consensus aggregates.  This
producer restores the individual canonical ten-head and shadow-ten-head raw
scores/ranks from the exact serialized monthly upstream bundles and the
recorded target-free source panel.  It is intentionally *not* a scorer for the
live stack: it writes an immutable research sidecar only.

Historical blocks are admitted only after their re-scored base and aggregate
consensus values match the originally persisted target-free score receipt.
August extensions are labelled separately because they come from post-history
target-free feature receipts and do not have an original lock-step receipt to
compare against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    MonthlyUpstreamBundle,
    load_monthly_upstream_bundle,
    score_monthly_upstream_bundle,
)


SCHEMA = "strict_r3_ten_head_history_v1"
HISTORICAL_RUNS = (
    ROOT / "data_perp/artifacts/strict_r3_lockstep_long_2024apr_dec_strictfull_prior28_optimizedpolicy_20260812_v2",
    ROOT / "data_perp/artifacts/strict_r3_lockstep_long_2025_jul2026_strictfull_prior28_optimizedpolicy_severefix_manifest_20260812_v3",
)
HISTORICAL_SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_source_panel_targetfree_long_"
    "2023_aug7_2026_raw15m_strictfull_20260812_v1/canonical_source_panel.parquet"
)
AUGUST_FEATURES = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_features_targetfree_long_aug1_12_"
    "fulluniverse_20260813_v2/canonical120_features.parquet"
)
AUGUST_UPSTREAM = ROOT / (
    "data_perp/artifacts/strict_r3_lockstep_successor28_homogeneous28_long_aug1_7_"
    "20260813_v1/bundles/cutoff=20260801/upstream"
)

KEYS = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")
PARITY_COLUMNS = (
    "base_score",
    "base_rank42",
    "conditional_consensus_rank",
    "ordinary_shadow_consensus_rank",
)
TOLERANCE = 1e-7


@dataclass(frozen=True)
class BundleRecord:
    source_run: str
    bundle_dir: str
    cutoff: str
    end_exclusive: str
    bundle_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def _scalar(value: pd.Timestamp) -> pa.Scalar:
    return pa.scalar(value.to_pydatetime(), type=pa.timestamp("ns", tz="UTC"))


def _head_columns(bundle: MonthlyUpstreamBundle) -> tuple[str, ...]:
    canonical = tuple(f"conditional_head__{head.spec.name}" for head in bundle.conditional_heads)
    shadow = tuple(f"ordinary_shadow_head__{head.spec.name}" for head in bundle.ordinary_shadow_heads)
    if len(canonical) != 10 or len(shadow) != 10:
        raise ValueError("expected exactly ten canonical and ten shadow heads")
    if len(set(canonical)) != 10 or len(set(shadow)) != 10:
        raise ValueError("head names are not unique")
    return canonical + shadow


def _source_fields(bundle: MonthlyUpstreamBundle) -> tuple[str, ...]:
    fields = set(KEYS)
    fields.update(bundle.base_fields)
    for head in (*bundle.conditional_heads, *bundle.ordinary_shadow_heads):
        fields.update(head.spec.fields)
    return tuple(sorted(fields))


def _read_window(source: Path, fields: Iterable[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dataset = ds.dataset(source, format="parquet")
    required = tuple(fields)
    missing = sorted(set(required) - set(dataset.schema.names))
    if missing:
        raise ValueError(f"target-free feature source misses required fields: {missing[:20]}")
    expression = (
        (ds.field("__decision_ts__") >= _scalar(start))
        & (ds.field("__decision_ts__") < _scalar(end))
    )
    result = dataset.to_table(columns=list(required), filter=expression, use_threads=True).to_pandas()
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result.empty:
        raise ValueError(f"target-free feature window is empty: {start.isoformat()} to {end.isoformat()}")
    if result["candidate_id"].duplicated().any():
        raise ValueError("target-free feature source has duplicate candidate identities")
    return result


def _discover_historical_bundles(runs: Iterable[Path]) -> list[tuple[Path, Path, MonthlyUpstreamBundle]]:
    result: list[tuple[Path, Path, MonthlyUpstreamBundle]] = []
    seen: set[pd.Timestamp] = set()
    for run in runs:
        if not run.is_dir():
            raise FileNotFoundError(run)
        for bundle_dir in sorted((run / "bundles").glob("cutoff=*/upstream")):
            bundle = load_monthly_upstream_bundle(bundle_dir)
            cutoff = _utc(bundle.cutoff)
            if cutoff in seen:
                raise ValueError(f"duplicate historical upstream cutoff {cutoff.isoformat()}")
            seen.add(cutoff)
            result.append((run, bundle_dir, bundle))
    return sorted(result, key=lambda item: _utc(item[2].cutoff))


def _stored_scores(run: Path, cutoff: pd.Timestamp) -> Path:
    path = run / "bundles" / f"cutoff={cutoff:%Y%m%d}" / "scores" / "held_target_free_scores.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parity_audit(*, stored_path: Path, score: pd.DataFrame) -> dict[str, object]:
    stored = pd.read_parquet(stored_path, columns=["candidate_id", *PARITY_COLUMNS])
    observed = score.loc[:, ["candidate_id", *PARITY_COLUMNS]]
    merged = stored.merge(observed, on="candidate_id", how="outer", validate="one_to_one", indicator=True, suffixes=("__stored", "__regen"))
    matched = merged["_merge"].eq("both")
    audit: dict[str, object] = {
        "stored_rows": int(len(stored)),
        "regenerated_rows": int(len(score)),
        "matched_rows": int(matched.sum()),
        "stored_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "regenerated_only_rows": int(merged["_merge"].eq("right_only").sum()),
    }
    if not bool(matched.all()):
        raise AssertionError(f"identity parity failed: {audit}")
    for column in PARITY_COLUMNS:
        delta = np.abs(
            pd.to_numeric(merged[f"{column}__stored"], errors="coerce").to_numpy(float)
            - pd.to_numeric(merged[f"{column}__regen"], errors="coerce").to_numpy(float)
        )
        max_delta = float(np.nanmax(delta)) if len(delta) else float("nan")
        audit[f"{column}_max_abs_delta"] = max_delta
        audit[f"{column}_mismatched_rows"] = int((delta > TOLERANCE).sum())
        if not np.isfinite(max_delta) or max_delta > TOLERANCE:
            raise AssertionError(f"{column} parity failed: max_delta={max_delta}")
    return audit


def _original_target_free_ids(stored_path: Path) -> pd.Series:
    """Return the original point-in-time scored population for one block.

    The canonical source panel is intentionally a superset of some older
    block receipts after later source repairs.  The persisted target-free
    receipt—not the later superset—is the authoritative historical candidate
    universe.  This is selection lineage, never outcome information.
    """
    stored = pd.read_parquet(stored_path, columns=["candidate_id"])
    if stored["candidate_id"].duplicated().any():
        raise ValueError(f"historical target-free receipt has duplicate IDs: {stored_path}")
    return stored["candidate_id"]


def _receipt_bounds(stored_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pd.to_datetime(
        pd.read_parquet(stored_path, columns=["__decision_ts__"])["__decision_ts__"],
        utc=True,
        errors="raise",
    )
    if timestamps.empty:
        raise ValueError(f"historical target-free receipt is empty: {stored_path}")
    # Candidate decisions are hourly.  This half-open bound represents what
    # was actually scored, which can legitimately be shorter than a bundle's
    # nominal four-week end at a source-run hand-off.
    return timestamps.min(), timestamps.max() + pd.Timedelta(hours=1)


def _project(score: pd.DataFrame, *, source_kind: str, source_id: str) -> pd.DataFrame:
    head_columns = sorted(
        column
        for column in score.columns
        if column.startswith("conditional_head__") or column.startswith("ordinary_shadow_head__")
    )
    expected = 40  # 10 canonical + 10 shadow, each raw and rank.
    if len(head_columns) != expected:
        raise AssertionError(f"expected {expected} individual head columns; got {len(head_columns)}")
    keep = [
        *KEYS,
        "base_score",
        "base_rank42",
        "base_anchor_bps",
        "conditional_consensus_rank",
        "ordinary_shadow_consensus_rank",
        "upstream",
        "ordinary_shadow_upstream",
        "upstream_bundle_sha256",
        *head_columns,
    ]
    output = score.loc[:, keep].copy()
    output["source_kind"] = source_kind
    output["source_id"] = source_id
    output["head_output_schema"] = "canonical_ten_plus_shadow_ten_raw_and_rank"
    return output


def _attach_symbol(score: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Restore the non-model identity dropped by the upstream scorer."""
    if "__symbol__" in score.columns:
        return score
    symbol = source.loc[:, ["candidate_id", "__symbol__"]]
    result = score.merge(symbol, on="candidate_id", how="left", validate="one_to_one")
    if result["__symbol__"].isna().any():
        raise AssertionError("upstream score lost a source symbol identity")
    return result


def _append(writer: pq.ParquetWriter | None, frame: pd.DataFrame, destination: Path) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(destination, table.schema, compression="zstd", compression_level=7)
        writer.write_table(table)
        return writer
    if writer.schema != table.schema:
        raise ValueError("ten-head output schema changed across blocks")
    writer.write_table(table)
    return writer


def _live_feature_receipts() -> list[Path]:
    # Each is an immutable, current-hour feature receipt.  They are an
    # extension only and intentionally retain their source vintage separately.
    receipts = sorted((ROOT / "data_perp/artifacts").glob(
        "strict_r3_successor_v*_live_202608*/current_hour_inputs/canonical120_features.parquet",
    ))
    return [path for path in receipts if path.is_file()]


def _read_complete_features(path: Path, fields: Iterable[str]) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    required = set(fields)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"feature receipt {path} misses required fields: {missing[:20]}")
    frame = pd.read_parquet(path, columns=sorted(required))
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"feature receipt has duplicate identities: {path}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2024-04-01T00:00:00Z")
    parser.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--include-august-receipts", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start, end = _utc(args.start), _utc(args.end)
    if end <= start:
        raise ValueError("end must follow start")
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite immutable output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    output_path = out_dir / "ten_head_target_free_scores.parquet"
    audit_rows: list[dict[str, object]] = []
    writer: pq.ParquetWriter | None = None
    expected_contract: tuple[str, ...] | None = None
    seen_ids: set[str] = set()
    # Do not infer historical coverage from a bundle's nominal 28-day end.
    # The last persisted historical producer may deliberately stop short of
    # that boundary; the August target-free extension must begin immediately
    # after the final actually re-scored candidate, not after the nominal end.
    historical_end = start
    try:
        for run, bundle_dir, bundle in _discover_historical_bundles(HISTORICAL_RUNS):
            cutoff, block_end = _utc(bundle.cutoff), _utc(bundle.end_exclusive)
            stored_path = _stored_scores(run, cutoff)
            receipt_start, receipt_end = _receipt_bounds(stored_path)
            if receipt_end <= start or receipt_start >= end:
                continue
            if receipt_start < start or receipt_end > end:
                raise ValueError(
                    "start/end must align to complete persisted target-free receipts; "
                    f"receipt={receipt_start.isoformat()}..{receipt_end.isoformat()}"
                )
            head_contract = _head_columns(bundle)
            if expected_contract is None:
                expected_contract = head_contract
            elif head_contract != expected_contract:
                raise AssertionError("individual head identities changed across historical bundles")
            source = _read_window(HISTORICAL_SOURCE, _source_fields(bundle), cutoff, block_end)
            original_ids = _original_target_free_ids(stored_path)
            source = source.loc[source["candidate_id"].isin(set(original_ids))].copy()
            if len(source) != len(original_ids):
                missing = len(original_ids) - len(source)
                raise AssertionError(
                    f"source panel cannot recover {missing} original target-free candidate IDs "
                    f"for {cutoff.isoformat()}"
                )
            score = score_monthly_upstream_bundle(bundle, source, route_top_fraction=None)
            parity = _parity_audit(stored_path=stored_path, score=score)
            projected = _project(
                _attach_symbol(score, source),
                source_kind="historical_exact_source_panel",
                source_id=str(run),
            )
            duplicated = set(projected["candidate_id"]).intersection(seen_ids)
            if duplicated:
                raise AssertionError(f"candidate identity repeats across block outputs: {next(iter(duplicated))}")
            seen_ids.update(projected["candidate_id"].astype(str))
            writer = _append(writer, projected, output_path)
            audit_rows.append({
                "kind": "historical_exact",
                "source_run": str(run),
                "bundle_dir": str(bundle_dir),
                "cutoff": cutoff,
                "end_exclusive": block_end,
                "bundle_sha256": str(bundle.manifest["bundle_sha256"]),
                "source_rows": int(len(source)),
                "output_rows": int(len(projected)),
                "head_contract": json.dumps(head_contract),
                **parity,
            })
            historical_end = max(
                historical_end,
                pd.to_datetime(projected["__decision_ts__"], utc=True).max() + pd.Timedelta(hours=1),
            )
            print(json.dumps({"event": "historical_block_complete", "cutoff": cutoff.isoformat(), "rows": int(len(projected))}), flush=True)

        if args.include_august_receipts and end > historical_end:
            august_bundle = load_monthly_upstream_bundle(AUGUST_UPSTREAM)
            if expected_contract is not None and _head_columns(august_bundle) != expected_contract:
                raise AssertionError("August upstream head contract differs from historical contract")
            fields = _source_fields(august_bundle)
            extensions: list[tuple[str, Path]] = []
            if AUGUST_FEATURES.is_file():
                extensions.append(("august_full_universe_feature_receipt", AUGUST_FEATURES))
            extensions.extend(("live_current_hour_feature_receipt", path) for path in _live_feature_receipts())
            extension_rows: list[pd.DataFrame] = []
            for source_kind, receipt in extensions:
                frame = _read_complete_features(receipt, fields)
                frame = frame.loc[
                    frame["__decision_ts__"].ge(historical_end)
                    & frame["__decision_ts__"].lt(end)
                ].copy()
                if not frame.empty:
                    frame["__source_kind__"] = source_kind
                    frame["__source_id__"] = str(receipt)
                    extension_rows.append(frame)
            if extension_rows:
                extension = pd.concat(extension_rows, ignore_index=True)
                extension = extension.sort_values(["candidate_id", "__decision_ts__", "__source_id__"], kind="stable")
                # Preserve the immutable full-universe August receipt in
                # preference to later current-hour copies of the same identity.
                extension["__priority__"] = extension["__source_kind__"].eq("august_full_universe_feature_receipt").astype(int)
                extension = extension.sort_values(["candidate_id", "__priority__", "__source_id__"], ascending=[True, False, True], kind="stable")
                extension = extension.drop_duplicates("candidate_id", keep="first").drop(columns="__priority__")
                extension = extension.loc[~extension["candidate_id"].isin(seen_ids)].copy()
                if not extension.empty:
                    for source_id, part in extension.groupby("__source_id__", sort=True):
                        score = score_monthly_upstream_bundle(
                            august_bundle,
                            part.drop(columns=["__source_kind__", "__source_id__"]),
                            route_top_fraction=None,
                        )
                        projected = _project(
                            _attach_symbol(score, part),
                            source_kind=str(part["__source_kind__"].iloc[0]),
                            source_id=str(source_id),
                        )
                        seen_ids.update(projected["candidate_id"].astype(str))
                        writer = _append(writer, projected, output_path)
                        audit_rows.append({
                            "kind": "extension_no_historical_aggregate_receipt",
                            "source_run": str(source_id),
                            "bundle_dir": str(AUGUST_UPSTREAM),
                            "cutoff": _utc(august_bundle.cutoff),
                            "end_exclusive": _utc(august_bundle.end_exclusive),
                            "bundle_sha256": str(august_bundle.manifest["bundle_sha256"]),
                            "source_rows": int(len(part)),
                            "output_rows": int(len(projected)),
                            "head_contract": json.dumps(_head_columns(august_bundle)),
                            "stored_rows": np.nan,
                            "regenerated_rows": int(len(projected)),
                            "matched_rows": np.nan,
                            "stored_only_rows": np.nan,
                            "regenerated_only_rows": np.nan,
                        })
                        print(json.dumps({"event": "extension_complete", "source": str(source_id), "rows": int(len(projected))}), flush=True)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("no rows were materialised")
    audit = pd.DataFrame(audit_rows)
    audit_path = out_dir / "ten_head_regeneration_block_audit.parquet"
    audit.to_parquet(audit_path, index=False, compression="zstd")
    output_rows = pq.ParquetFile(output_path).metadata.num_rows
    manifest = {
        "schema": SCHEMA,
        "purpose": "causal individual ten-head output sidecar for MC1 admission research",
        "start": start.isoformat(),
        "requested_end": end.isoformat(),
        "historical_source_panel": str(HISTORICAL_SOURCE),
        "historical_source_panel_sha256": _sha256(HISTORICAL_SOURCE),
        "historical_runs": [str(path) for path in HISTORICAL_RUNS],
        "historical_parity": "mandatory exact aggregate parity within 1e-7",
        "extension_contract": "target-free feature receipts only; explicitly lacks original historical aggregate parity receipt",
        "head_contract": list(expected_contract or ()),
        "output": str(output_path),
        "output_rows": int(output_rows),
        "output_sha256": _sha256(output_path),
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": str(Path(__file__).relative_to(ROOT)),
        "producer_sha256": _sha256(Path(__file__)),
        "outcomes_consumed_during_scoring": [],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "rows": int(output_rows), "out_dir": str(out_dir)}), flush=True)


if __name__ == "__main__":
    main()
