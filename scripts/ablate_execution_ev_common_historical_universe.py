#!/usr/bin/env python3
"""Compare the current global top-k book with a frozen historical universe.

The diagnostic keeps the score, exact target, causal recent mapping and one
pooled global top 10% contract fixed.  It reports both:

* the common-universe slice of the original unrestricted global book; and
* a separately reranked top 10% inside the frozen common universe.

The second is an explicit universe ablation, not a production recommendation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SCORE = "canonical_recent_ev_score"
TARGET = "execution_net_ev_12h"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pooled_top_fraction(
    frame: pd.DataFrame, *, score: str = SCORE, fraction: float = 0.10
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows = max(1, int(np.ceil(float(fraction) * len(frame))))
    return frame.nlargest(rows, score).copy()


def summarize(
    selected: pd.DataFrame, *, candidate_rows: int, book: str
) -> dict[str, Any]:
    return {
        "book": book,
        "candidate_rows": int(candidate_rows),
        "selected_rows": int(len(selected)),
        "selected_fraction": len(selected) / max(candidate_rows, 1),
        "mean_net_ev_bps": (
            float(selected[TARGET].mean() * 1e4) if len(selected) else None
        ),
        "mean_gross_ev_bps": (
            float(selected["execution_gross_ev_12h"].mean() * 1e4)
            if len(selected)
            else None
        ),
        "mean_cost_bps": (
            float(selected["execution_cost_return"].mean() * 1e4)
            if len(selected)
            else None
        ),
        "positive_rate": (
            float(selected[TARGET].gt(0.0).mean()) if len(selected) else None
        ),
        "long_rows": int(selected["side_name"].eq("long").sum()),
        "short_rows": int(selected["side_name"].eq("short").sum()),
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise ValueError("refusing to overwrite common-universe ablation")
    symbols = {
        line.strip()
        for line in args.symbol_allowlist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not symbols:
        raise ValueError("symbol allowlist is empty")
    predictions = pd.read_parquet(args.predictions)
    targets = pd.read_parquet(
        args.targets,
        columns=[
            *IDENTITY,
            TARGET,
            "execution_gross_ev_12h",
            "execution_cost_return",
        ],
    )
    for frame in (predictions, targets):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["__symbol__"] = frame["__symbol__"].astype(str)
        if frame.duplicated(
            [*IDENTITY, *(["window", "arm"] if "arm" in frame.columns else [])],
            keep=False,
        ).any():
            raise ValueError("duplicate prediction or target identity")
    joined = predictions.merge(
        targets, on=list(IDENTITY), how="left", validate="many_to_one"
    )
    if joined[TARGET].isna().any() or not np.isfinite(
        joined[[SCORE, TARGET]].to_numpy(np.float64)
    ).all():
        raise ValueError("prediction-to-exact-target join is incomplete or nonfinite")

    rows: list[dict[str, Any]] = []
    assignments: list[pd.DataFrame] = []
    for (window, arm), group in joined.groupby(["window", "arm"], sort=True):
        unrestricted = pooled_top_fraction(group)
        common = group.loc[group["__symbol__"].isin(symbols)].copy()
        common_reranked = pooled_top_fraction(common)
        common_slice = unrestricted.loc[
            unrestricted["__symbol__"].isin(symbols)
        ].copy()
        for book, selected, candidates in (
            ("unrestricted_global_top10", unrestricted, group),
            ("common_universe_reranked_top10", common_reranked, common),
            ("common_universe_slice_of_unrestricted_top10", common_slice, common),
        ):
            row = {
                "window": str(window),
                "arm": str(arm),
                **summarize(selected, candidate_rows=len(candidates), book=book),
            }
            rows.append(row)
            tagged = selected.loc[:, [*IDENTITY, SCORE, TARGET]].copy()
            tagged["window"] = str(window)
            tagged["arm"] = str(arm)
            tagged["book"] = book
            assignments.append(tagged)

    metrics = pd.DataFrame(rows)
    selected_rows = pd.concat(assignments, ignore_index=True)
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.csv"
    selected_path = args.output_dir / "selected_rows.parquet"
    manifest_path = args.output_dir / "manifest.json"
    metrics.to_csv(metrics_path, index=False)
    selected_rows.to_parquet(selected_path, index=False, compression="zstd")
    payload: Mapping[str, Any] = {
        "schema": "execution_ev_common_historical_universe_ablation_v1",
        "status": "diagnostic_not_promotion_evidence",
        "contract": {
            "ranking": "one pooled global top 10% across sides and timestamps",
            "mapping": SCORE,
            "target": TARGET,
            "universe_ablation": (
                "frozen symbols with complete May-July 2025 exact-one-minute "
                "coverage; outcome-free availability filter"
            ),
        },
        "inputs": {
            "predictions": str(args.predictions),
            "predictions_sha256": _sha(args.predictions),
            "targets": str(args.targets),
            "targets_sha256": _sha(args.targets),
            "symbol_allowlist": str(args.symbol_allowlist),
            "symbol_allowlist_sha256": _sha(args.symbol_allowlist),
            "symbol_count": len(symbols),
        },
        "outputs": {
            "metrics": str(metrics_path),
            "metrics_sha256": _sha(metrics_path),
            "selected_rows": str(selected_path),
            "selected_rows_sha256": _sha(selected_path),
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "metrics": metrics_path,
        "selected": selected_path,
        "manifest": manifest_path,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--targets", type=Path, required=True)
    result.add_argument("--symbol-allowlist", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps({k: str(v) for k, v in run(args).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
