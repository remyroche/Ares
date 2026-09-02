#!/usr/bin/env python3
"""Strict-OOS agreement gate over frozen post-FS entry selections.

The underlying q50/depth-3 and q50/depth-2 pairwise models were each fitted
only on their preceding resolved data.  This runner intersects their already
target-free selections.  It never re-fits, promotes a new candidate, or uses
held outcomes to form the intersection; it merely asks whether agreement is a
more portable replacement authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as base


HPO_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_postfs_hpo_20260830_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_agreement_ablation_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"candidate_id", "__decision_ts__", "__symbol__"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"selection receipt lacks {sorted(missing)}")
    frame["candidate_id"] = frame.candidate_id.astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame.__decision_ts__, utc=True, errors="raise")
    if frame.candidate_id.duplicated().any():
        raise AssertionError(f"selection receipt has duplicate candidate identities: {path}")
    return frame


def _scope_replay(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for scope, frame in (
        ("selection_jun_jul", selection.loc[selection.__decision_ts__.lt(SELECTION_END)].copy()),
        ("august_holdout", selection.loc[selection.__decision_ts__.ge(SELECTION_END)].copy()),
        ("all_oos", selection),
    ):
        if frame.empty:
            continue
        metrics = base._replay(frame, labels, f"{arm}__{scope}", output)
        metrics["model_arm"], metrics["evaluation_scope"] = arm, scope
        results.append(metrics)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-root", type=Path, default=HPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root, output = args.hpo_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    paths = {
        "B0_bcf_top2": root / "B0_bcf_top2_selection_target_free.parquet",
        "E0_q50_d3": root / "H0_q50_d3_l7_baseline_selection_target_free.parquet",
        "E1_q50_d2": root / "H3_q50_d2_l3_strict_selection_target_free.parquet",
    }
    tables = {name: _read(path) for name, path in paths.items()}
    h0, h3 = tables["E0_q50_d3"], tables["E1_q50_d2"]
    agreement_ids = set(h0.candidate_id).intersection(h3.candidate_id)
    tables["E2_q50_agreement"] = h0.loc[h0.candidate_id.isin(agreement_ids)].copy()
    labels = base._labels(base.LABEL_ROOT)
    output.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for name, frame in tables.items():
        frame.to_parquet(output / f"{name}_selection_target_free.parquet", index=False, compression="zstd")
        summaries.extend(_scope_replay(frame, labels, name, output))
        for scope, scoped in (("selection_jun_jul", frame.loc[frame.__decision_ts__.lt(SELECTION_END)]), ("august_holdout", frame.loc[frame.__decision_ts__.ge(SELECTION_END)])):
            cohort_rows.append({"arm": name, "evaluation_scope": scope, "selected_target_free": len(scoped), "unique_timestamps": scoped.__decision_ts__.nunique()})
    summary = pd.DataFrame(summaries)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        baseline = group.loc[group.model_arm.eq("B0_bcf_top2")]
        if len(baseline) != 1:
            raise AssertionError(f"missing B0 control for {scope}")
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_B0_{metric}"] = group[metric] - baseline.iloc[0][metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    pd.DataFrame(cohort_rows).to_parquet(output / "cohort_counts.parquet", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "schema": "strict-r3-p8u-entry-agreement-ablation-v1", "scope": "offline strict-OOS research only; no live/canonical mutation",
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        "authority": "E2 is the intersection of two pre-existing target-free q50 replacement selections; it cannot introduce candidates absent from either model",
        "selection_period": "2026-06 through 2026-07; August is untouched holdout",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
