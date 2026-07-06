#!/usr/bin/env python3
"""Materialize simple-policy executable net-return labels into label artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.side_aware import add_side_contract_columns
from extreme_price_movements.training import _materialize_policy_net_replay_labels


def _coerce_utc(value: str | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def _read_manifest(labels_dir: Path) -> dict[str, Any]:
    path = labels_dir / "labels_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing labels manifest: {path}")
    with path.open("r") as f:
        return json.load(f)


def _infer_side(dataset_name: str, file_name: str) -> str:
    joined = f"{dataset_name} {file_name}".lower()
    if "train_short" in joined or "_short_" in joined:
        return "short"
    if "train_long" in joined or "_long_" in joined:
        return "long"
    raise ValueError(
        f"Could not infer side from dataset/file name: {dataset_name} / {file_name}. "
        "Pass a manifest with train_long/train_short names or extend side inference."
    )


def _materialize_file(
    *,
    source_path: Path,
    output_path: Path,
    dataset_name: str,
    side: str,
    data_root: str,
    market_mode: str,
    exchange: str,
    timeframe: str,
    chunk_size: int,
    min_coverage: float,
    overwrite_columns: bool,
    min_ts: pd.Timestamp | None,
    max_ts: pd.Timestamp | None,
) -> dict[str, Any]:
    df = pd.read_parquet(source_path)
    required = {"__ts__", "__symbol__", "__barrier_pct__"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"{source_path}: missing required columns {missing}")
    if (not overwrite_columns) and "__u_policy_net__" in df.columns:
        raise RuntimeError(
            f"{source_path}: already contains __u_policy_net__; use --overwrite-columns."
        )

    source_rows = int(len(df))
    if min_ts is not None or max_ts is not None:
        ts = pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
        keep = pd.Series(True, index=df.index)
        if min_ts is not None:
            keep &= ts >= min_ts
        if max_ts is not None:
            keep &= ts < max_ts
        df = df.loc[keep].copy()
        if df.empty:
            raise RuntimeError(
                f"{dataset_name}: no rows remain after applying time window "
                f"[{min_ts}, {max_ts})."
            )

    n = len(df)
    vals = np.full(n, np.nan, dtype=np.float32)
    chunk_stats: list[dict[str, Any]] = []
    for start in range(0, n, max(1, int(chunk_size))):
        end = min(n, start + max(1, int(chunk_size)))
        chunk = df.iloc[start:end]
        chunk_vals, stats = _materialize_policy_net_replay_labels(
            timestamps=chunk["__ts__"],
            symbols=chunk["__symbol__"],
            side=side,
            barrier_pct=chunk["__barrier_pct__"].to_numpy(dtype=np.float32),
            cfg={
                "data_root": data_root,
                "market_mode": market_mode,
                "training_exchange_contract": {
                    "exchange_id": exchange,
                    "market_mode": market_mode,
                },
                "label_policy_net_replay_min_coverage": min_coverage,
            },
            label=f"{dataset_name}:{start}:{end}",
        )
        vals[start:end] = chunk_vals[: end - start]
        chunk_stats.append(
            {
                "start": int(start),
                "end": int(end),
                "finite": int(stats.get("finite", 0)),
                "finite_frac": float(stats.get("finite_frac", float("nan"))),
                "mean": float(stats.get("mean", float("nan"))),
                "std": float(stats.get("std", float("nan"))),
            }
        )

    finite = np.isfinite(vals)
    finite_frac = float(np.mean(finite)) if n else 0.0
    if finite_frac < float(min_coverage):
        raise RuntimeError(
            f"{dataset_name}: full-file replay coverage {finite_frac:.2%} "
            f"is below required {float(min_coverage):.2%}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    out_df["__u_policy_net__"] = vals
    out_df["__r_policy_net__"] = vals
    out_df = add_side_contract_columns(
        out_df,
        side=side,
        timestamp_col="__ts__",
        asset_col="__symbol__",
        timeframe=timeframe,
        copy=False,
    )
    out_df.to_parquet(output_path, index=False)
    return {
        "dataset": dataset_name,
        "source_file": str(source_path),
        "output_file": str(output_path),
        "side": side,
        "source_rows": source_rows,
        "rows": int(n),
        "min_ts": str(pd.to_datetime(df["__ts__"], utc=True).min()) if n else None,
        "max_ts": str(pd.to_datetime(df["__ts__"], utc=True).max()) if n else None,
        "finite": int(np.sum(finite)),
        "finite_frac": finite_frac,
        "mean": float(np.nanmean(vals)) if np.any(finite) else float("nan"),
        "std": float(np.nanstd(vals)) if np.any(finite) else float("nan"),
        "p10": float(np.nanpercentile(vals, 10)) if np.any(finite) else float("nan"),
        "p90": float(np.nanpercentile(vals, 90)) if np.any(finite) else float("nan"),
        "chunk_stats": chunk_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument("--min-ts", default=None, help="Inclusive lower timestamp bound for source rows.")
    parser.add_argument("--max-ts", default=None, help="Exclusive upper timestamp bound for source rows.")
    parser.add_argument("--overwrite-columns", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    min_ts = _coerce_utc(args.min_ts)
    max_ts = _coerce_utc(args.max_ts)
    source_labels = data_root / "artifacts" / args.source_run_id / "labels"
    output_labels = data_root / "artifacts" / args.output_run_id / "labels"
    manifest = _read_manifest(source_labels)
    datasets = manifest.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError(f"No datasets found in {source_labels / 'labels_manifest.json'}")

    out_manifest = {
        "run_id": args.output_run_id,
        "source_run_id": args.source_run_id,
        "datasets": {},
        "materialized_policy_net_replay": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root),
            "market_mode": str(args.market_mode),
            "exchange": str(args.exchange),
            "timeframe": str(args.timeframe),
            "chunk_size": int(args.chunk_size),
            "min_coverage": float(args.min_coverage),
            "min_ts": str(min_ts) if min_ts is not None else None,
            "max_ts": str(max_ts) if max_ts is not None else None,
        },
    }
    summaries = []
    for dataset_name, meta in datasets.items():
        if not isinstance(meta, dict):
            continue
        file_name = str(meta.get("file") or "")
        if not file_name or not file_name.startswith("train_"):
            continue
        source_path = source_labels / file_name
        output_path = output_labels / file_name
        side = _infer_side(str(dataset_name), file_name)
        summary = _materialize_file(
            source_path=source_path,
            output_path=output_path,
            dataset_name=str(dataset_name),
            side=side,
            data_root=str(data_root),
            market_mode=str(args.market_mode),
            exchange=str(args.exchange),
            timeframe=str(args.timeframe),
            chunk_size=int(args.chunk_size),
            min_coverage=float(args.min_coverage),
            overwrite_columns=bool(args.overwrite_columns),
            min_ts=min_ts,
            max_ts=max_ts,
        )
        summaries.append(summary)
        columns = list(meta.get("columns", []))
        for col in ("__u_policy_net__", "__r_policy_net__"):
            if col not in columns:
                columns.append(col)
        out_meta = dict(meta)
        out_meta["columns"] = columns
        out_meta["rows"] = int(summary["rows"])
        out_manifest["datasets"][dataset_name] = out_meta

    if not summaries:
        raise RuntimeError(f"No train_*.parquet datasets were materialized from {source_labels}")
    output_labels.mkdir(parents=True, exist_ok=True)
    with (output_labels / "labels_manifest.json").open("w") as f:
        json.dump(out_manifest, f, indent=2, sort_keys=True)
    with (output_labels / "policy_net_replay_materialization_summary.json").open("w") as f:
        json.dump({"datasets": summaries}, f, indent=2, sort_keys=True)
    print(json.dumps({"output_labels": str(output_labels), "datasets": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
