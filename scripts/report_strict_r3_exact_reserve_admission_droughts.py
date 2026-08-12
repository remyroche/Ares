#!/usr/bin/env python3
"""Report weekly exact-reserve EV-map coverage and admission droughts.

The report treats no-admission weeks as a diagnostic, not as a data-quality
failure by default.  It separates an unavailable exact producer map from a
fully mapped population whose causal expected EV simply fails the declared
+50-bps threshold.  Realised policy outcomes are evaluation-only context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXACT_MODE = "strict_oof_exact_producer_reserve_map_plus_causal_residual_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _diagnosis(block: pd.DataFrame) -> str:
    scored = len(block)
    mapped = block["causal_21d_side_expected_net_bps"].notna().sum()
    admitted = block["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool).sum()
    if not scored:
        return "NO_POINT_IN_TIME_CANDIDATES"
    if not mapped:
        return "EXACT_MAP_OR_LINEAGE_FAILURE"
    if not admitted:
        return "ECONOMIC_ADMISSION_DROUGHT_ALL_MAPPED_EV_BELOW_50BPS"
    return "ACTIVE_ADMISSION"


def _week_metrics(block: pd.DataFrame, period: str) -> dict[str, object]:
    observed_hours = pd.to_datetime(block["__decision_ts__"], utc=True).nunique()
    expected = pd.to_numeric(block["causal_21d_side_expected_net_bps"], errors="coerce")
    mapped = expected.notna()
    admitted = block["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)
    policy_valid = block["policy_path_valid"].fillna(False).astype(bool)
    net = pd.to_numeric(block["policy_net_bps"], errors="coerce")
    top = block.loc[mapped].assign(__expected__=expected.loc[mapped]).sort_values(
        "__expected__", ascending=False, kind="stable",
    )
    tail_count = max(1, int(np.ceil(len(top) * 0.05)))
    top = top.head(tail_count)
    top_net = pd.to_numeric(top["policy_net_bps"], errors="coerce")
    selected = admitted & policy_valid & np.isfinite(net)
    return {
        "week": period,
        "observed_hours": int(observed_hours),
        "complete_utc_week": bool(observed_hours == 7 * 24),
        "scored_rows": int(len(block)),
        "mapped_rows": int(mapped.sum()),
        "exact_map_coverage": float(mapped.mean()) if len(block) else np.nan,
        "admitted_rows": int(admitted.sum()),
        "admission_rate": float(admitted.mean()) if len(block) else np.nan,
        "max_expected_net_bps": float(expected.max()) if mapped.any() else np.nan,
        "top5_expected_net_bps": float(pd.to_numeric(top["__expected__"], errors="coerce").mean()) if len(top) else np.nan,
        "top5_realised_net_bps": float(top_net.mean()) if top_net.notna().any() else np.nan,
        "admitted_realised_net_bps": float(net.loc[selected].mean()) if selected.any() else np.nan,
        "admitted_positive_rate": float(net.loc[selected].gt(0.0).mean()) if selected.any() else np.nan,
        "drought_diagnosis": _diagnosis(block),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable drought report already exists: {args.out_dir}")
    frame = pd.read_parquet(args.predictions)
    required = {
        "candidate_id", "__decision_ts__", "producer_bundle_id",
        "ev_mapping_vintage_mode", "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps", "policy_path_valid", "policy_net_bps",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"exact-reserve drought report lacks: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("exact-reserve drought report requires unique candidate IDs")
    modes = set(frame["ev_mapping_vintage_mode"].dropna().astype(str))
    if modes != {EXACT_MODE}:
        raise ValueError(
            "drought report requires only exact-producer reserve maps; "
            f"observed {sorted(modes)}",
        )
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame = frame.assign(__week__=decision.dt.tz_localize(None).dt.to_period("W-MON").astype(str))
    weeks = [
        _week_metrics(block, str(period))
        for period, block in frame.groupby("__week__", sort=True, observed=True)
    ]
    first_hour = frame.assign(__hour__=decision.dt.floor("h")).groupby(
        "producer_bundle_id", sort=True, observed=True,
    ).apply(
        lambda block: block.loc[block["__hour__"].eq(block["__hour__"].min())],
        include_groups=False,
    ).reset_index(level=0).reset_index(drop=True)
    first_hour["mapped"] = first_hour["causal_21d_side_expected_net_bps"].notna()
    first_hour_report = first_hour.groupby("producer_bundle_id", sort=True, observed=True).agg(
        first_live_hour=("__hour__", "min"),
        first_hour_rows=("candidate_id", "size"),
        first_hour_mapped_rows=("mapped", "sum"),
    ).reset_index()
    first_hour_report["first_hour_exact_map_coverage"] = (
        first_hour_report["first_hour_mapped_rows"] / first_hour_report["first_hour_rows"]
    )
    weekly = pd.DataFrame(weeks)
    args.out_dir.mkdir(parents=True)
    weekly.to_parquet(args.out_dir / "weekly_admission_drought_audit.parquet", index=False)
    first_hour_report.to_parquet(args.out_dir / "refit_first_hour_coverage.parquet", index=False)
    manifest = {
        "schema": "strict_r3_exact_producer_reserve_admission_drought_report_v1",
        "predictions": str(args.predictions),
        "predictions_sha256": _sha(args.predictions),
        "weeks": int(len(weekly)),
        "zero_admission_complete_weeks": int(
            (weekly["admitted_rows"].eq(0) & weekly["complete_utc_week"]).sum(),
        ),
        "zero_admission_partial_weeks": int(
            (weekly["admitted_rows"].eq(0) & ~weekly["complete_utc_week"]).sum(),
        ),
        "exact_map_failure_complete_weeks": int(
            (weekly["drought_diagnosis"].eq("EXACT_MAP_OR_LINEAGE_FAILURE") & weekly["complete_utc_week"]).sum(),
        ),
        "economic_admission_drought_complete_weeks": int(
            (
                weekly["drought_diagnosis"].eq("ECONOMIC_ADMISSION_DROUGHT_ALL_MAPPED_EV_BELOW_50BPS")
                & weekly["complete_utc_week"]
            ).sum(),
        ),
        "contract": (
            "exact producer reserve map only; weekly admission is causal; policy outcomes "
            "are evaluation-only diagnostics and never determine drought classification"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
