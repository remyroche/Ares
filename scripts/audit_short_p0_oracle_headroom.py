#!/usr/bin/env python3
"""Diagnostic-only oracle headroom of the frozen short P0 hourly winners.

The source predictions are strict-OOS score ledgers.  This script first
selects the already-scored top candidate at each decision timestamp, then
uses its subsequently resolved canonical policy outcome only to estimate how
much *hour-level admission* headroom exists.  It never trains a model, maps a
score, or produces an inference input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES: tuple[Path, ...] = (
    ROOT / "data_perp/artifacts/strict_r3_short_policy_conversion_path_screen_dev_20260820_v4/mayjun/oos_predictions_P0_policy_bps.parquet",
    ROOT / "data_perp/artifacts/strict_r3_short_policy_conversion_path_screen_dev_20260820_v4/julaug/oos_predictions_P0_policy_bps.parquet",
    ROOT / "data_perp/artifacts/strict_r3_short_policy_conversion_path_screen_dev_20260820_v4/sep/oos_predictions_P0_policy_bps.parquet",
)
THRESHOLDS_BPS = (0.0, 50.0, 100.0, 200.0, 400.0)
ORACLE_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_policy(frame: pd.DataFrame) -> pd.Series:
    return frame.policy_path_valid.fillna(False).astype(bool) & pd.to_numeric(
        frame.p0_canonical_net_bps, errors="coerce"
    ).notna()


def load_top1(sources: tuple[Path, ...]) -> pd.DataFrame:
    required = ["candidate_id", "__ts__", "__symbol__", "side_name", "score", "policy_path_valid", "p0_canonical_net_bps"]
    frames: list[pd.DataFrame] = []
    for path in sources:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=required)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["p0_canonical_net_bps"] = pd.to_numeric(frame["p0_canonical_net_bps"], errors="coerce")
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        frames.append(frame)
    population = pd.concat(frames, ignore_index=True)
    if population.candidate_id.duplicated().any():
        raise ValueError("oracle source ledgers overlap in candidate identity")
    valid = population.loc[_valid_policy(population) & population.score.notna()].copy()
    if valid.empty:
        raise ValueError("no policy-resolved frozen P0 candidates")
    winners = (
        valid.sort_values(["__ts__", "score", "candidate_id"], ascending=[True, False, True], kind="stable")
        .groupby("__ts__", sort=False, as_index=False)
        .head(1)
        .sort_values("__ts__", kind="stable")
        .reset_index(drop=True)
    )
    if winners["__ts__"].duplicated().any():
        raise AssertionError("top-one selection failed to produce one row per hour")
    return winners


def distribution_metrics(winners: pd.DataFrame, *, scope: str) -> list[dict[str, Any]]:
    outcome = winners.p0_canonical_net_bps.to_numpy(float)
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS_BPS:
        rows.append({
            "scope": scope, "metric": "fraction_above_bps", "threshold_bps": threshold,
            "hours": int(len(winners)), "value": float(np.mean(outcome > threshold)),
        })
    for quantile in (.50, .60, .70, .80, .90, .95):
        rows.append({
            "scope": scope, "metric": "outcome_quantile_bps", "quantile": quantile,
            "hours": int(len(winners)), "value": float(np.quantile(outcome, quantile)),
        })
    rows.append({"scope": scope, "metric": "mean_net_bps", "hours": int(len(winners)), "value": float(np.mean(outcome))})
    return rows


def oracle_metrics(winners: pd.DataFrame, *, scope: str) -> list[dict[str, Any]]:
    ordered = winners.sort_values(["p0_canonical_net_bps", "candidate_id"], ascending=[False, True], kind="stable")
    rows: list[dict[str, Any]] = []
    for fraction in ORACLE_FRACTIONS:
        count = max(1, int(np.ceil(len(ordered) * fraction)))
        selection = ordered.head(count).p0_canonical_net_bps.to_numpy(float)
        rows.append({
            "scope": scope, "oracle_fraction_hours": fraction, "hours_selected": count,
            "mean_policy_net_bps": float(np.mean(selection)),
            "median_policy_net_bps": float(np.median(selection)),
            "p10_policy_net_bps": float(np.quantile(selection, .10)),
        })
    return rows


def run(*, out: Path, sources: tuple[Path, ...]) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    winners = load_top1(sources)
    winners.to_parquet(out / "frozen_p0_top1_per_hour.parquet", index=False, compression="zstd")
    distribution: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for scope, part in [("pooled", winners), *[(str(month), frame) for month, frame in winners.groupby(winners.__ts__.dt.strftime("%Y-%m"), sort=True)]]:
        distribution.extend(distribution_metrics(part, scope=scope))
        oracle.extend(oracle_metrics(part, scope=scope))
    pd.DataFrame(distribution).to_parquet(out / "distribution_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(oracle).to_parquet(out / "oracle_hour_selection_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_p0_oracle_headroom_v1",
        "status": "complete_diagnostic_only",
        "side": "short",
        "sources": [{"path": str(path), "sha256": _sha256(path)} for path in sources],
        "selection": "Frozen P0 score: select the highest score per resolved decision timestamp, candidate_id ascending as a deterministic tie-break.",
        "outcome": "Canonical exact 1m parent-policy net bps; decision-time entry; H12 timeout; 100-bps cost once.",
        "oracle": "Future outcomes are used only after the frozen P0 hourly winner has been selected. These rows cannot be used for training, admission, or deployment.",
        "hours": int(len(winners)), "start": winners["__ts__"].min().isoformat(), "end": winners["__ts__"].max().isoformat(),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()
    sources = tuple(args.source) if args.source else DEFAULT_SOURCES
    print(run(out=args.out.resolve(), sources=sources))


if __name__ == "__main__":
    main()
