#!/usr/bin/env python3
"""Materialize a source-verified native-L2 continuation sidecar.

The historical orderbook surface contains many OHLCV-derived proxy rows.  The
runner records those rows but never passes them to the native-L2 generator.
Only rows tagged ``kraken_futures_l2_snapshot`` are emitted as feature data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.market_microstructure_features import (
    NATIVE_L2_ALLOWED_SOURCES,
    NATIVE_L2_CONTINUATION_FEATURE_KEYS,
    NATIVE_L2_REQUIRED_COLUMNS,
    materialize_native_l2_continuation_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_perp" / "exchanges" / "krakenfutures" / "orderbook_hourly"
DEFAULT_OUTPUT = ROOT / "data_perp" / "artifacts" / "native_l2_continuation_sidecar_20260801_v1"
SCHEMA = "native_l2_continuation_sidecar_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _read_source_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    names = set(pq.ParquetFile(path).schema.names)
    columns = [name for name in (*NATIVE_L2_REQUIRED_COLUMNS, "l2_bid_notional_l20", "l2_ask_notional_l20") if name in names]
    if "source" not in columns:
        return pd.DataFrame(), {"file": path.name, "rows": 0, "native_rows": 0, "proxy_rows": 0, "status": "MISSING_SOURCE_COLUMN"}
    frame = pd.read_parquet(path, columns=columns)
    source = frame["source"].astype(str)
    native = frame.loc[source.isin(NATIVE_L2_ALLOWED_SOURCES)].copy()
    missing_native = sorted(set(NATIVE_L2_REQUIRED_COLUMNS).difference(native.columns)) if len(native) else []
    status = "NATIVE_ROWS" if len(native) and not missing_native else ("NATIVE_ROWS_MISSING_COLUMNS" if len(native) else "PROXY_ONLY")
    if len(native) and not missing_native:
        native["symbol"] = path.stem
    return native if not missing_native else pd.DataFrame(), {
        "file": path.name,
        "rows": int(len(frame)),
        "native_rows": int(len(native)),
        "proxy_rows": int((~source.isin(NATIVE_L2_ALLOWED_SOURCES)).sum()),
        "status": status,
        "missing_native_columns": missing_native,
        "sha256": _sha256(path) if len(native) else None,
    }


def run(*, input_dir: Path = DEFAULT_INPUT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files found under {input_dir}")
    frames: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for path in paths:
        frame, record = _read_source_file(path)
        coverage.append(record)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("no native-L2 rows were found")
    native = pd.concat(frames, ignore_index=True)
    features = materialize_native_l2_continuation_features(native, symbol_column="symbol")
    features = features.sort_values(["symbol", "snapshot_ts"], kind="stable").reset_index(drop=True)
    coverage_df = pd.DataFrame(coverage).sort_values("file", kind="stable")
    ts = pd.to_datetime(features["snapshot_ts"], utc=True)
    month = ts.dt.strftime("%Y-%m")
    month_summary = (
        pd.DataFrame({"month": month, "rows": 1, "symbols": features["symbol"]})
        .groupby("month", observed=True)
        .agg(rows=("rows", "sum"), symbols=("symbols", "nunique"))
        .reset_index()
    )
    source_hashes = [record["sha256"] for record in coverage if record.get("sha256")]
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        features.to_parquet(stage / "native_l2_continuation_features.parquet", index=False, compression="zstd")
        coverage_df.to_parquet(stage / "native_l2_source_coverage.parquet", index=False, compression="zstd")
        month_summary.to_parquet(stage / "native_l2_month_coverage.parquet", index=False, compression="zstd")
        feature_dictionary = {
            name: {
                "source": "native Kraken futures L2 snapshot",
                "available_at": "snapshot_ts",
                "point_in_time": True,
                "lookback": "current snapshot or immediately preceding same-symbol snapshot when gap <= 2h",
                "units": "bps, ratio, log-notional, seconds, or binary flag as implied by field name",
                "target_or_outcome": False,
                "production_eligible": False,
            }
            for name in NATIVE_L2_CONTINUATION_FEATURE_KEYS
        }
        (stage / "native_l2_feature_dictionary.json").write_text(json.dumps(feature_dictionary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = (
            "# Native-L2 continuation sidecar\n\n"
            "Status: research-only; no production base/meta feature list was changed.\n\n"
            "Only rows tagged `kraken_futures_l2_snapshot` were emitted. Historical "
            "`local_ohlcv_summary` rows were counted as proxies and excluded from the feature panel. "
            "No labels, scores, ranks, selection weights, or portfolio constraints are present.\n\n"
            f"Native rows: {len(features):,}; symbols: {features['symbol'].nunique():,}; "
            f"coverage: {ts.min().isoformat()} to {ts.max().isoformat()}.\n\n"
            "## Month coverage\n\n"
            + _markdown_table(month_summary)
            + "\n\n## Disposition\n\n"
            "The sidecar is eligible for a later candidate-level as-of join only where "
            "`snapshot_ts <= decision_ts` and symbol/product identity is exact. It is not yet "
            "a model-training cohort because candidate overlap and full-period support have not "
            "been established.\n"
        )
        (stage / "NATIVE_L2_CONTINUATION_REPORT.md").write_text(report, encoding="utf-8")
        manifest = {
            "schema": SCHEMA,
            "status": "RESEARCH_ONLY_NATIVE_SOURCE_COHORT",
            "promotion_eligible": False,
            "candidate_joined": False,
            "labels_used": False,
            "portfolio_constraints_in_scope": False,
            "input_directory": str(input_dir),
            "allowed_sources": sorted(NATIVE_L2_ALLOWED_SOURCES),
            "proxy_rows_excluded": int(coverage_df["proxy_rows"].fillna(0).sum()),
            "input_files": int(len(paths)),
            "native_files": int((coverage_df["native_rows"].fillna(0) > 0).sum()),
            "native_rows": int(len(features)),
            "native_symbols": int(features["symbol"].nunique()),
            "min_snapshot_ts": ts.min().isoformat(),
            "max_snapshot_ts": ts.max().isoformat(),
            "native_source_file_sha256": sorted(source_hashes),
            "feature_keys": list(NATIVE_L2_CONTINUATION_FEATURE_KEYS),
            "outputs": {
                "features": "native_l2_continuation_features.parquet",
                "source_coverage": "native_l2_source_coverage.parquet",
                "month_coverage": "native_l2_month_coverage.parquet",
                "feature_dictionary": "native_l2_feature_dictionary.json",
                "report": "NATIVE_L2_CONTINUATION_REPORT.md",
            },
        }
        manifest["outputs_sha256"] = {name: _sha256(stage / name) for name in manifest["outputs"].values()}
        runner = Path(__file__).resolve()
        manifest["runner"] = {"path": str(runner.relative_to(ROOT)), "sha256": _sha256(runner)}
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(input_dir=args.input_dir, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
