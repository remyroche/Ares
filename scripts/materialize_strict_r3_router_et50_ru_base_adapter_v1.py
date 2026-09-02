#!/usr/bin/env python3
"""Materialise ET50's native target-free Base receipt for the R/U comparator.

This adapter has one purpose: expose the frozen ET50 base score in the exact
``base_score``/``base_rank_ts`` schema consumed by the single-head R/U probe.
It preserves ET50's average-tie percentile rank rather than replacing tied
mapped scores with a new candidate-ID tie-break.  It neither fits, scores,
calibrates, admits, nor evaluates any model; no outcome is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
FORBIDDEN = {
    "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_label_available_ts", "label", "target", "outcome",
}


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(value: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in value.split(",") if item.strip())
    if not result or tuple(sorted(result)) != result:
        raise ValueError("months must be a chronological non-empty comma-separated sequence")
    return result


def _native_rank(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("__decision_ts__", sort=False)["base_score"].rank(
        pct=True, method="average",
    ).to_numpy(float)


def run(source: Path, out: Path, months: tuple[pd.Timestamp, ...]) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    sources: list[Path] = []
    audit: list[dict[str, object]] = []
    for month in months:
        path = source / "target_free_monthly" / f"month={month:%Y-%m}" / "scores_features.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        columns = set(pd.read_parquet(path, columns=None).columns)
        leaked = sorted(item for item in columns if item in FORBIDDEN or item.startswith("policy_"))
        if leaked:
            raise AssertionError(f"{path}: forbidden outcome fields in ET50 target-free source: {leaked}")
        required = {*IDENTITY, "enhanced_base_bps", "base_rank_ts"}
        if missing := required - columns:
            raise AssertionError(f"{path}: missing ET50 base fields: {sorted(missing)}")
        frame = pd.read_parquet(path, columns=[*IDENTITY, "enhanced_base_bps", "base_rank_ts"])
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame = frame.rename(columns={"enhanced_base_bps": "base_score"})
        frame["base_score"] = pd.to_numeric(frame["base_score"], errors="coerce")
        frame["base_rank_ts"] = pd.to_numeric(frame["base_rank_ts"], errors="coerce")
        if frame.duplicated(list(IDENTITY)).any() or not frame.side_name.eq("long").all():
            raise AssertionError(f"{path}: invalid ET50 identity or side")
        if not np.isfinite(frame[["base_score", "base_rank_ts"]].to_numpy(float)).all():
            raise AssertionError(f"{path}: non-finite ET50 base value")
        rebuilt = _native_rank(frame)
        if not np.allclose(rebuilt, frame.base_rank_ts.to_numpy(float), rtol=0.0, atol=1e-7):
            raise AssertionError(f"{path}: ET50 stored rank does not match average-tie target-free rank")
        destination = out / "target_free_scores" / "et50" / f"month={month:%Y-%m}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[:, [*IDENTITY, "base_score", "base_rank_ts"]].to_parquet(destination, index=False, compression="zstd")
        sources.append(path)
        audit.append({
            "month": f"{month:%Y-%m}", "rows": int(len(frame)),
            "source": str(path), "rank_contract": "average_tie_percentile",
            "target_free": True, "outcome_fields_consumed": [],
        })
    _write_once(out / "correctness_report.json", {
        "all_source_rows_target_free": True,
        "all_identity_side_long_and_unique": True,
        "all_native_average_tie_ranks_verified": True,
        "no_outcomes_consumed": True,
    })
    _write_once(out / "run_manifest.json", {
        "schema": "strict_r3_router_et50_ru_base_adapter_v1",
        "scope": "offline target-free ET50 Base schema adapter; no model, MC1, portfolio, live, or exchange mutation",
        "source": str(source),
        "months": [f"{month:%Y-%m}" for month in months],
        "rank_contract": "average_tie_percentile of ET50 enhanced_base_bps",
        "source_sha256": _sha256(sources),
        "audit": audit,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", default="2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    args = parser.parse_args()
    print(run(args.source.resolve(), args.out.resolve(), _months(args.months)))


if __name__ == "__main__":
    main()
