#!/usr/bin/env python3
"""Audit in-sample LGBM archetype labels and economic viability by dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.lgbm_pipeline import (
    _archetype_viability_matrix,
    _label_context_behavior_archetype_labels,
)
from extreme_price_movements.training import _subsample_indices_time_balanced


LABEL_CONTEXT_COLUMNS = [
    "side",
    "side_name",
    "__is_timeout__",
    "__quality__",
    "__mae_ret__",
    "__mfe_ret__",
    "__bars_to_mfe__",
    "__bars_to_mae__",
    "__n_tp__",
    "__n_sl__",
    "__y_bin__",
    "__y_ret__",
    "__y_outcome__",
    "__tp__",
    "__sl__",
    "__barrier_pct__",
    "__ts__",
    "__symbol__",
    "__side__",
]


def _existing_columns(path: Path, wanted: list[str]) -> list[str]:
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    available = set(schema.names)
    return [col for col in wanted if col in available]


def _load_row_universe(
    *,
    data_root: Path,
    run_id: str,
    dataset: str,
) -> pd.DataFrame | None:
    path = data_root / "artifacts" / run_id / "row_universe" / "train_row_universe_all.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["dataset", "timestamp", "symbol"])
    df = df[df["dataset"].astype(str).eq(str(dataset))].copy()
    if df.empty:
        return None
    df = df.rename(columns={"timestamp": "__ts__", "symbol": "__symbol__"})
    df["__ts__"] = pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
    df["__symbol__"] = df["__symbol__"].astype(str)
    return df[["__ts__", "__symbol__"]]


def _label_context_from_frame(df: pd.DataFrame) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    mapping = {
        "__side__": "side",
        "side": "side",
        "side_name": "side_name",
        "__is_timeout__": "is_timeout",
        "__quality__": "quality",
        "__mae_ret__": "mae",
        "__mfe_ret__": "mfe",
        "__bars_to_mfe__": "bars_to_mfe",
        "__bars_to_mae__": "bars_to_mae",
        "__n_tp__": "n_tp",
        "__n_sl__": "n_sl",
        "__y_bin__": "y_bin",
        "__y_ret__": "y_ret",
        "__y_outcome__": "exit_code",
        "__tp__": "tp",
        "__sl__": "sl",
    }
    for src, dst in mapping.items():
        if src in df.columns:
            ctx[dst] = df[src].to_numpy()
    if "__mae_ret__" in df.columns and ("__barrier_pct__" in df.columns or "__tp__" in df.columns):
        barrier_col = "__barrier_pct__" if "__barrier_pct__" in df.columns else "__tp__"
        mae = pd.to_numeric(df["__mae_ret__"], errors="coerce").to_numpy(dtype=np.float64)
        barrier = pd.to_numeric(df[barrier_col], errors="coerce").to_numpy(dtype=np.float64)
        ctx["bad_mae_1r"] = (np.abs(mae) >= np.maximum(np.abs(barrier), 1e-9)).astype(np.float32)
    return ctx


def _audit_dataset(
    *,
    data_root: Path,
    label_run_id: str,
    row_universe_run_id: str,
    dataset: str,
    sample_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_path = data_root / "artifacts" / label_run_id / "labels" / f"{dataset}.parquet"
    if not label_path.exists():
        return [], {"dataset": dataset, "error": f"missing label artifact: {label_path}"}
    cols = _existing_columns(label_path, LABEL_CONTEXT_COLUMNS)
    df = pd.read_parquet(label_path, columns=cols)
    if "__ts__" in df.columns:
        df["__ts__"] = pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
    if "__symbol__" in df.columns:
        df["__symbol__"] = df["__symbol__"].astype(str)
    row_universe = _load_row_universe(
        data_root=data_root,
        run_id=row_universe_run_id,
        dataset=dataset,
    )
    source_rows = int(len(df))
    if row_universe is not None and {"__ts__", "__symbol__"}.issubset(df.columns):
        df = df.merge(row_universe.assign(__keep__=1), on=["__ts__", "__symbol__"], how="inner")
    filtered_rows = int(len(df))
    if filtered_rows <= 0:
        return [], {
            "dataset": dataset,
            "source_rows": source_rows,
            "filtered_rows": filtered_rows,
            "error": "empty after row-universe filter",
        }
    if "__symbol__" in df.columns and "__ts__" in df.columns:
        df = df.sort_values(["__symbol__", "__ts__"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    y = (
        pd.to_numeric(df["__y_bin__"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "__y_bin__" in df.columns
        else np.arange(len(df), dtype=np.float32)
    )
    if sample_cap > 0 and len(df) > sample_cap:
        idx = _subsample_indices_time_balanced(len(df), int(sample_cap), y)
        df = df.iloc[idx].reset_index(drop=True)
    ctx = _label_context_from_frame(df)
    timestamps = df["__ts__"].to_numpy() if "__ts__" in df.columns else None
    labels, source = _label_context_behavior_archetype_labels(
        pd.DataFrame(index=df.index),
        label_context=ctx,
        timestamps=timestamps,
    )
    if labels is None:
        return [], {
            "dataset": dataset,
            "source_rows": source_rows,
            "filtered_rows": filtered_rows,
            "sample_rows": int(len(df)),
            "error": "no archetype labels",
        }
    viability = _archetype_viability_matrix(labels, label_context=ctx, timestamps=timestamps)
    rows: list[dict[str, Any]] = []
    for row in viability.get("matrix", []) or []:
        out = dict(row)
        out["dataset"] = dataset
        out["archetype_source"] = source
        rows.append(out)
    summary = {
        "dataset": dataset,
        "source_rows": source_rows,
        "filtered_rows": filtered_rows,
        "sample_rows": int(len(df)),
        "archetype_source": source,
        "archetype_count": int(len(set(labels.astype(str).tolist()))),
        "active_candidate_count": int(viability.get("active_candidate_count", 0) or 0),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--label-run-id", required=True)
    parser.add_argument("--row-universe-run-id", default="")
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--sample-cap", type=int, default=60000)
    parser.add_argument("--datasets", default="")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    row_universe_run_id = args.row_universe_run_id or args.output_run_id
    if args.datasets.strip():
        datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    else:
        label_dir = data_root / "artifacts" / args.label_run_id / "labels"
        datasets = sorted(path.stem for path in label_dir.glob("train_global_*.parquet"))
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        rows, summary = _audit_dataset(
            data_root=data_root,
            label_run_id=args.label_run_id,
            row_universe_run_id=row_universe_run_id,
            dataset=dataset,
            sample_cap=int(args.sample_cap),
        )
        all_rows.extend(rows)
        summaries.append(summary)
    out_dir = data_root / "artifacts" / args.output_run_id / "diagnostics" / "archetype_viability"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "archetype_viability_matrix.csv"
    summary_path = out_dir / "archetype_viability_summary.json"
    pd.DataFrame(all_rows).to_csv(matrix_path, index=False)
    summary = {
        "label_run_id": args.label_run_id,
        "row_universe_run_id": row_universe_run_id,
        "output_run_id": args.output_run_id,
        "sample_cap": int(args.sample_cap),
        "datasets": summaries,
        "matrix_rows": int(len(all_rows)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {matrix_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
