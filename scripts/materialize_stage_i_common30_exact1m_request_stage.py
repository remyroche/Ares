#!/usr/bin/env python3
"""Freeze the Stage-I Jan--Apr 2026 common-30 exact-1m backfill request.

The source is the completed PackB TP6/SL4/R3 label sidecar.  This program
does not infer candidates from labels: it retains every source candidate in
the fixed common-30 universe, including already-valid rows, so the downloader
can independently verify all requested H12 windows.  It never reassigns a
candidate identity or a decision time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKB_ROOT = (
    ROOT / "data_perp/artifacts/stage_i_packb_tp6_sl4_h12_r3_20260803_v1"
)
DEFAULT_UNIVERSE = ROOT / "configs/historical_exact1m_common_universe_2025_v1.txt"
DEFAULT_MONTHS = ("2026-01", "2026-02", "2026-03", "2026-04")
SIDES = ("long", "short")
SCHEMA = "stage_i_common30_exact1m_request_stage_v1"
REQUIRED_COLUMNS = {
    "candidate_id",
    "__ts__",
    "__symbol__",
    "side_name",
    "__decision_ts__",
    "kraken_minute_symbol",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def load_common_universe(path: Path) -> tuple[str, ...]:
    """Load the exact, commented text universe without silently broadening it."""
    symbols = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not symbols:
        raise ValueError("common universe is empty")
    if len(symbols) != len(set(symbols)):
        raise ValueError("common universe has duplicate symbols")
    if not all(symbol.endswith(":USD") for symbol in symbols):
        raise ValueError("common universe must contain only USD-settled symbols")
    return symbols


def _parse_months(values: Iterable[str]) -> tuple[str, ...]:
    months = tuple(values) or DEFAULT_MONTHS
    if len(months) != len(set(months)):
        raise ValueError("months must be unique")
    unsupported = sorted(set(months) - set(DEFAULT_MONTHS))
    if unsupported:
        raise ValueError(f"Stage-I request only permits Jan--Apr 2026, got {unsupported}")
    return tuple(sorted(months))


def _utc(value: str, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset")
    return timestamp.tz_convert("UTC")


def _source_manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "manifest.json"
    manifest = _json(path)
    if manifest.get("schema") != "exact_tp6_sl4_h12_r3_relabel_v2":
        raise ValueError("PackB source must be exact_tp6_sl4_h12_r3_relabel_v2")
    if manifest.get("status") != "complete" or manifest.get("complete") is not True:
        raise ValueError("PackB source labels are not complete")
    if manifest.get("candidate_source_kind") != "packb":
        raise ValueError("source is not a PackB candidate population")
    return path, manifest


def materialize(
    *,
    packb_root: Path,
    universe_path: Path,
    output_dir: Path,
    months: Iterable[str] = DEFAULT_MONTHS,
    horizon_minutes: int = 720,
) -> dict[str, Any]:
    """Write an immutable stage and return its hash-bound manifest."""
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    selected_months = _parse_months(months)
    source_manifest_path, source_manifest = _source_manifest(packb_root)
    universe = load_common_universe(universe_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")

    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for month in selected_months:
        for side in SIDES:
            shard = packb_root / "parts" / f"month={month}" / f"side={side}.parquet"
            if not shard.exists():
                raise FileNotFoundError(f"missing frozen PackB shard: {shard}")
            schema = set(pq.read_schema(shard).names)
            missing = sorted(REQUIRED_COLUMNS - schema)
            if missing:
                raise ValueError(f"{shard} missing required columns: {missing}")
            columns = sorted(REQUIRED_COLUMNS | ({"label_valid", "target_invalid"} & schema))
            raw = pd.read_parquet(shard, columns=columns)
            if not raw["side_name"].astype(str).eq(side).all():
                raise ValueError(f"{shard} has a side-name mismatch")
            filtered = raw.loc[raw["__symbol__"].astype(str).isin(universe)].copy()
            if filtered.empty:
                raise ValueError(f"{shard} has no rows in the frozen common universe")
            filtered["signal_timestamp"] = pd.to_datetime(
                filtered.pop("__ts__"), utc=True, errors="raise"
            )
            filtered["decision_timestamp"] = pd.to_datetime(
                filtered.pop("__decision_ts__"), utc=True, errors="raise"
            )
            expected_decision = filtered["signal_timestamp"] + pd.Timedelta(hours=1)
            if not filtered["decision_timestamp"].eq(expected_decision).all():
                raise ValueError(f"{shard} violates the frozen signal-close +1h entry convention")
            filtered["symbol"] = filtered.pop("__symbol__").astype(str)
            expected_minute_symbol = filtered["symbol"].str.replace(
                "/", "_", regex=False
            )
            if not filtered["kraken_minute_symbol"].astype(str).eq(expected_minute_symbol).all():
                raise ValueError(f"{shard} has an inconsistent Kraken minute symbol")
            filtered["source_month"] = month
            filtered["source_side"] = side
            filtered["source_shard_path"] = str(shard.resolve())
            filtered["source_shard_sha256"] = _sha256(shard)
            filtered["path_end_exclusive"] = filtered["decision_timestamp"] + pd.Timedelta(
                minutes=horizon_minutes
            )
            frames.append(filtered)
            sources.append(
                {
                    "month": month,
                    "side": side,
                    "path": str(shard.resolve()),
                    "rows_before_common30_filter": int(len(raw)),
                    "rows_selected": int(len(filtered)),
                    "sha256": filtered["source_shard_sha256"].iat[0],
                }
            )

    staged = pd.concat(frames, ignore_index=True)
    if staged["candidate_id"].isna().any() or staged["candidate_id"].astype(str).eq("").any():
        raise ValueError("source candidate IDs must be present")
    if staged["candidate_id"].duplicated().any():
        raise ValueError("frozen PackB candidate IDs must remain globally unique")
    staged = staged.sort_values(
        ["decision_timestamp", "symbol", "side_name", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    path_map = staged[["candidate_id", "signal_timestamp", "decision_timestamp", "symbol", "path_end_exclusive"]]
    download = path_map.rename(columns={"decision_timestamp": "timestamp"})[
        ["timestamp", "symbol"]
    ].drop_duplicates().sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_path = output_dir / "staged_candidates.parquet"
    path_map_path = output_dir / "candidate_path_map.parquet"
    download_path = output_dir / "download_candidates.parquet"
    staged.to_parquet(staged_path, index=False, compression="zstd")
    path_map.to_parquet(path_map_path, index=False, compression="zstd")
    download.to_parquet(download_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "request_population_frozen_before_download",
        "promotion_eligible": False,
        "evidence_scope": "stage_i_common30_exact1m_backfill_provenance",
        "source": {
            "packb_manifest": {"path": str(source_manifest_path.resolve()), "sha256": _sha256(source_manifest_path)},
            "packb_schema": source_manifest["schema"],
            "universe": {"path": str(universe_path.resolve()), "sha256": _sha256(universe_path), "symbols": len(universe)},
        },
        "months": list(selected_months),
        "sides": list(SIDES),
        "entry_convention": "preserved source __decision_ts__; validated source __ts__ signal close +1h",
        "path_horizon_minutes": int(horizon_minutes),
        "path_interval": "[decision_timestamp, path_end_exclusive)",
        "candidate_identity_contract": "source-native candidate_id preserved byte-for-byte",
        "source_shards": sources,
        "selected_rows": int(len(staged)),
        "download_rows": int(len(download)),
        "distinct_symbols": int(staged["symbol"].nunique()),
        "signal_start": staged["signal_timestamp"].min().isoformat(),
        "signal_end": staged["signal_timestamp"].max().isoformat(),
        "decision_start": staged["decision_timestamp"].min().isoformat(),
        "path_end": staged["path_end_exclusive"].max().isoformat(),
        "outputs": {
            "staged_candidates": {"path": str(staged_path.resolve()), "rows": int(len(staged)), "sha256": _sha256(staged_path)},
            "candidate_path_map": {"path": str(path_map_path.resolve()), "rows": int(len(path_map)), "sha256": _sha256(path_map_path)},
            "download_candidates": {"path": str(download_path.resolve()), "rows": int(len(download)), "sha256": _sha256(download_path)},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packb-root", type=Path, default=DEFAULT_PACKB_ROOT)
    parser.add_argument("--universe-path", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--month", action="append", default=[])
    parser.add_argument("--horizon-minutes", type=int, default=720)
    args = parser.parse_args()
    manifest = materialize(
        packb_root=args.packb_root,
        universe_path=args.universe_path,
        output_dir=args.output_dir,
        months=args.month or DEFAULT_MONTHS,
        horizon_minutes=args.horizon_minutes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
