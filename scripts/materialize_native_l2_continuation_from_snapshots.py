#!/usr/bin/env python3
"""Materialize native-L2 continuation features from raw spread snapshots.

The hourly sidecar is intentionally not the only admissible native source.
This runner consumes the raw ``orderbook_history`` files, aggregates levels
with the observed timestamp as feature availability, and then applies the
same bounded causal continuation generator.  It never reads OHLCV proxies,
labels, scores, or portfolio fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.market_microstructure_features import (  # noqa: E402
    NATIVE_L2_RAW_COLUMNS,
    materialize_native_l2_continuation_features,
    summarize_native_l2_snapshot_rows,
)


DEFAULT_INPUT = ROOT / "data_perp/exchanges/krakenfutures/spread_snapshots"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/native_l2_continuation_sidecar_20260801_v3"
SCHEMA = "native_l2_continuation_sidecar_v2_raw_snapshot_source"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _history_files(input_dir: Path) -> list[Path]:
    root = input_dir / "orderbook_history" if input_dir.name != "orderbook_history" else input_dir
    return sorted(root.glob("date=*/snapshots.parquet"))


def _read_raw(path: Path) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema.names)
    missing = sorted(set(NATIVE_L2_RAW_COLUMNS).difference(names))
    if missing:
        raise ValueError(f"{path} is missing raw native-L2 columns: {missing}")
    frame = parquet.read(columns=list(NATIVE_L2_RAW_COLUMNS)).to_pandas()
    if "timestamp" not in frame.columns and frame.index.name == "timestamp":
        frame = frame.reset_index()
    return frame


def _write_report(output: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Native-L2 raw snapshot sidecar",
        "",
        "Status: `RESEARCH_ONLY_NATIVE_SOURCE_COHORT`",
        "",
        "The source is restricted to `kraken_futures_l2_snapshot` rows from the raw Kraken spread-snapshot history. Raw levels are aggregated by product and snapshot bucket with `observed_ts` retained as `snapshot_ts`; no future fill is used. This artifact contains no labels, model scores, or portfolio fields.",
        "",
        "## Coverage",
        "",
        f"- Input files: **{manifest['input_files']:,}**.",
        f"- Raw native level rows: **{manifest['raw_rows']:,}**.",
        f"- Aggregated native snapshots: **{manifest['summary_rows']:,}**.",
        f"- Derived feature rows: **{manifest['feature_rows']:,}**.",
        f"- Products: **{manifest['symbols']:,}**.",
        f"- Coverage: **{manifest['min_snapshot_ts']}** to **{manifest['max_snapshot_ts']}**.",
        f"- Bounded prior-snapshot fields available: **{manifest['lag_ready_rows']:,}** ({manifest['lag_ready_coverage']:.3%}).",
        "- Trade-flow fields are not present in the raw level source and are not emitted.",
        "",
        "## Contract",
        "",
        "- Source allow-list: `kraken_futures_l2_snapshot` only.",
        "- Product identity: exact `symbol` from the native source.",
        "- Availability: `observed_ts`; exchange/bucket timestamps are not used as an earlier availability time.",
        "- Prior-snapshot changes: valid only for a preceding same-product snapshot within two hours.",
        "- Promotion: false; candidate join and labels were not used.",
        "",
        "This is a denser current-period native cohort than the hourly sidecar, but it still begins after the April candidate window. It cannot establish historical OOF economics until a longer native history is available and the exact-product overlap audit passes.",
    ]
    (output / "NATIVE_L2_SNAPSHOT_SIDECAR_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    files = _history_files(args.input)
    if not files:
        raise SystemExit(f"no raw orderbook history files found under {args.input}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    file_inventory: list[dict[str, Any]] = []
    raw_rows = 0
    for path in files:
        raw = _read_raw(path)
        raw_rows += int(len(raw))
        if raw["source"].astype("string").dropna().unique().tolist() != [
            "kraken_futures_l2_snapshot"
        ]:
            raise ValueError(f"{path} contains a non-native source tag")
        summary = summarize_native_l2_snapshot_rows(raw)
        summary_frames.append(summary)
        file_inventory.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "raw_rows": int(len(raw)),
                "summary_rows": int(len(summary)),
                "min_observed_ts": pd.to_datetime(raw["observed_ts"], utc=True).min().isoformat(),
                "max_observed_ts": pd.to_datetime(raw["observed_ts"], utc=True).max().isoformat(),
            }
        )

    summary = pd.concat(summary_frames, ignore_index=True)
    summary["snapshot_ts"] = pd.to_datetime(summary["snapshot_ts"], utc=True, errors="coerce")
    summary = summary.sort_values(["symbol", "snapshot_ts"], kind="stable")
    summary = summary.drop_duplicates(["symbol", "snapshot_ts"], keep="last").reset_index(drop=True)
    features = materialize_native_l2_continuation_features(summary)
    features.to_parquet(output / "native_l2_continuation_features.parquet", index=False, compression="zstd")
    summary.to_parquet(output / "native_l2_snapshot_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(file_inventory).to_csv(output / "input_file_inventory.csv", index=False)

    lag_ready = features["l2_mid_return_prev_snapshot"].notna()
    manifest = {
        "schema": SCHEMA,
        "status": "RESEARCH_ONLY_NATIVE_SOURCE_COHORT",
        "promotion_eligible": False,
        "candidate_joined": False,
        "labels_used": False,
        "portfolio_constraints_in_scope": False,
        "source_allow_list": ["kraken_futures_l2_snapshot"],
        "input_directory": str(args.input),
        "input_files": len(files),
        "raw_rows": raw_rows,
        "summary_rows": int(len(summary)),
        "feature_rows": int(len(features)),
        "symbols": int(features["symbol"].nunique()),
        "min_snapshot_ts": pd.to_datetime(features["snapshot_ts"], utc=True).min().isoformat(),
        "max_snapshot_ts": pd.to_datetime(features["snapshot_ts"], utc=True).max().isoformat(),
        "lag_ready_rows": int(lag_ready.sum()),
        "lag_ready_coverage": float(lag_ready.mean()),
        "outputs": {
            "features": "native_l2_continuation_features.parquet",
            "summary": "native_l2_snapshot_summary.parquet",
            "input_inventory": "input_file_inventory.csv",
            "report": "NATIVE_L2_SNAPSHOT_SIDECAR_REPORT.md",
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
