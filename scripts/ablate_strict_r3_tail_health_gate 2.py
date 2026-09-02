#!/usr/bin/env python3
"""Causally ablate producer-local tail-health admission overlays.

This does not retrain or alter a strict-R3 producer.  It compares the existing
exact-reserve admission with predeclared overlays that react only to prior
resolved outcomes from the same producer's reserve-defined positive tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_tail_health import (  # noqa: E402
    TailHealthSpec,
    apply_exact_producer_tail_health,
)


ARM_SPECS: dict[str, TailHealthSpec | None] = {
    "B0_exact_reserve_control": None,
    "B1_tail_mean_14d_7d_3d": TailHealthSpec(
        residual_windows_days=(14, 7, 3),
        residual_shrinkage_rows=(100.0, 50.0, 25.0),
        lower_confidence_z=0.0,
    ),
    "B2_tail_mean_7d_3d": TailHealthSpec(
        residual_windows_days=(7, 3),
        residual_shrinkage_rows=(50.0, 25.0),
        lower_confidence_z=0.0,
    ),
    "B3_tail_lcb1_7d_3d": TailHealthSpec(
        residual_windows_days=(7, 3),
        residual_shrinkage_rows=(50.0, 25.0),
        lower_confidence_z=1.0,
    ),
    "B4_tail_lcb1_3d": TailHealthSpec(
        residual_windows_days=(3,),
        residual_shrinkage_rows=(25.0,),
        lower_confidence_z=1.0,
    ),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(frame: pd.DataFrame, *, arm: str, selected: pd.Series, frequency: str, period: str) -> dict[str, object]:
    valid = frame["policy_path_valid"].fillna(False).astype(bool)
    rows = frame.loc[selected & valid].copy()
    net = pd.to_numeric(rows["policy_net_bps"], errors="coerce")
    gross = pd.to_numeric(rows["policy_gross_bps"], errors="coerce")
    expected_column = (
        "causal_21d_side_expected_net_bps"
        if arm == "B0_exact_reserve_control" else "tail_health_expected_net_bps"
    )
    expected = pd.to_numeric(rows[expected_column], errors="coerce")
    return {
        "arm": arm,
        "frequency": frequency,
        "period": period,
        "scored_rows": int(len(frame)),
        "selected_rows": int(selected.sum()),
        "policy_valid_selected_rows": int(len(rows)),
        "admission_rate": float(selected.mean()) if len(frame) else np.nan,
        "net_bps_per_trade": float(net.mean()) if len(rows) else np.nan,
        "gross_bps_per_trade": float(gross.mean()) if len(rows) else np.nan,
        "expected_net_bps_per_trade": float(expected.mean()) if len(rows) else np.nan,
        "median_net_bps": float(net.median()) if len(rows) else np.nan,
        "positive_net_rate": float(net.gt(0.0).mean()) if len(rows) else np.nan,
    }


def _summarise(frame: pd.DataFrame, *, arm: str, selected: pd.Series) -> list[dict[str, object]]:
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    output = [_metrics(frame, arm=arm, selected=selected, frequency="all", period="all")]
    for frequency in ("M", "W-MON"):
        periods = decision.dt.tz_localize(None).dt.to_period(frequency).astype(str)
        for period, positions in periods.groupby(periods, sort=True).groups.items():
            index = np.asarray(list(positions), dtype=np.int64)
            output.append(_metrics(
                frame.iloc[index], arm=arm, selected=selected.iloc[index],
                frequency=frequency, period=str(period),
            ))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable tail-health ablation output exists: {args.out_dir}")
    frame = pd.read_parquet(args.predictions)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("tail-health ablation requires unique candidate IDs")
    args.out_dir.mkdir(parents=True)
    metrics: list[dict[str, object]] = []
    audits: list[pd.DataFrame] = []
    selection = frame[["candidate_id", "__decision_ts__", "producer_bundle_id"]].copy()
    for name, spec in ARM_SPECS.items():
        if spec is None:
            scored = frame
            selected = frame["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)
            selection[f"{name}__admitted"] = selected.to_numpy(bool)
            selection[f"{name}__expected_net_bps"] = pd.to_numeric(
                frame["causal_21d_side_expected_net_bps"], errors="coerce",
            )
        else:
            scored, audit = apply_exact_producer_tail_health(frame, spec=spec)
            selected = scored["tail_health_admitted_ge_50bps"].astype(bool)
            selection[f"{name}__admitted"] = selected.to_numpy(bool)
            selection[f"{name}__expected_net_bps"] = scored["tail_health_expected_net_bps"].to_numpy(float)
            selection[f"{name}__lcb_bps"] = scored["tail_health_lcb_bps"].to_numpy(float)
            selection[f"{name}__support_3d"] = scored.get(
                "tail_health_reference_rows_3d", pd.Series(np.nan, index=scored.index),
            ).to_numpy()
            audit["arm"] = name
            audits.append(audit)
        metrics.extend(_summarise(scored, arm=name, selected=selected))
    pd.DataFrame(metrics).to_parquet(args.out_dir / "tail_health_metrics.parquet", index=False)
    selection.to_parquet(args.out_dir / "tail_health_selection.parquet", index=False, compression="zstd")
    pd.concat(audits, ignore_index=True).to_parquet(
        args.out_dir / "tail_health_causal_audit.parquet", index=False, compression="zstd",
    )
    manifest = {
        "schema": "strict_r3_exact_producer_tail_health_ablation_v1",
        "predictions": str(args.predictions),
        "predictions_sha256": _sha(args.predictions),
        "arms": {
            name: "existing exact-reserve admission" if spec is None else asdict(spec)
            for name, spec in ARM_SPECS.items()
        },
        "contract": (
            "no score/model retraining; producer-local high-tail residuals; "
            "only policy outcomes resolved before each decision; reserve-defined "
            "tail eligibility; no cross-producer raw-score or outcome pooling"
        ),
        "rows": int(len(frame)),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
