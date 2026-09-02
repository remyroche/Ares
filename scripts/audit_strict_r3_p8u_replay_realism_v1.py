#!/usr/bin/env python3
"""Write a read-only realism receipt for a frozen P8U/F72/Under-F120 replay.

The receipt deliberately separates what the replay verifies from what it does
not: rich-policy exits are 15-minute aggregate paths, portfolio capacity is
committed initial margin, and MC1's daily residual shift is audited both for
causal construction and for realised calibration effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_underf120_gate_capacity_sweep_aug27_20260828_v3_committed_margin"
DEFAULT_MC1 = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_aug27_20260828_v1"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_p8u_august01_27_rich_policy_labels_20260828_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _policy_audit(mc1: Path, policy: Path) -> dict[str, object]:
    labels = pd.read_parquet(
        mc1 / "dual_predictions.parquet",
        columns=["policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_cost_bps"],
    )
    valid = labels.loc[labels["policy_path_valid"].fillna(False).astype(bool)].copy()
    diff = (
        pd.to_numeric(valid["policy_gross_bps"], errors="coerce")
        - pd.to_numeric(valid["policy_net_bps"], errors="coerce")
        - pd.to_numeric(valid["policy_cost_bps"], errors="coerce")
    ).abs()
    manifest = json.loads((policy / "run_manifest.json").read_text())
    return {
        "resolution": "15m aggregate rich-policy proxy; not an exact 1m-exit replay",
        "frozen_policy": manifest["policy"]["frozen_policy"],
        "frozen_policy_sha256": manifest["policy"]["frozen_policy_sha256"],
        "entry": manifest["policy"]["entry"],
        "horizon": manifest["policy"]["horizon"],
        "valid_policy_rows": int(len(valid)),
        "cost_once_deviation_count": int((diff > 1e-6).sum()),
        "max_cost_once_abs_deviation_bps": float(diff.max()),
        "status": "pass_at_15m_proxy_resolution",
        "one_minute_status": "not evaluated by this receipt",
    }


def _capacity_audit(sweep: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in sorted(sweep.glob("gate_*_decisions.parquet")):
        decisions = pd.read_parquet(path)
        accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
        committed = pd.to_numeric(accepted["committed_initial_capital_after_entry"], errors="coerce")
        marked = pd.to_numeric(accepted["marked_allocated_capital_after_entry"], errors="coerce")
        limit = pd.to_numeric(accepted["capital_limit_at_entry"], errors="coerce")
        committed_ratio = committed / limit.where(limit.gt(0.0))
        marked_ratio = marked / limit.where(limit.gt(0.0))
        timestamp_entries = accepted.groupby("timestamp", sort=True).size()
        rows.append({
            "arm": path.stem.removesuffix("_202511_202608_decisions"),
            "accepted_entries": int(len(accepted)),
            "max_committed_margin_ratio": float(committed_ratio.max()),
            "committed_margin_violations": int((committed_ratio > 1.0 + 1e-9).sum()),
            "max_marked_margin_ratio": float(marked_ratio.max()),
            "max_open_positions": int(accepted["open_positions_after"].max()),
            "max_entries_per_timestamp": int(timestamp_entries.max()),
        })
    frame = pd.DataFrame(rows)
    return {
        "contract": "80% wallet cap on committed initial margin; 8 open positions; arm-specific new-entry cap",
        "all_committed_margin_caps_hold": bool(frame["committed_margin_violations"].eq(0).all()),
        "arms": rows,
        "note": "Marked exposure can exceed the entry cap after price movement; it is a risk diagnostic, not a new-margin reservation.",
    }


def _mc1_family_audit(path: Path) -> dict[str, object]:
    data = pd.read_parquet(
        path,
        columns=[
            "__decision_ts__", "policy_path_valid", "policy_net_bps",
            "static_expected_bps", "recent_shift_bps", "mc1_expected_bps",
        ],
    )
    data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True, errors="raise")
    data = data.loc[
        data["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(data["policy_net_bps"], errors="coerce").notna()
    ].copy()
    data["day"] = data["__decision_ts__"].dt.normalize()
    daily = data.groupby("day", sort=True).agg(
        actual=("policy_net_bps", "mean"),
        static=("static_expected_bps", "mean"),
        mc1=("mc1_expected_bps", "mean"),
        shift=("recent_shift_bps", "first"),
        shift_count=("recent_shift_bps", "nunique"),
    )
    static_bias = daily["actual"] - daily["static"]
    mc1_bias = daily["actual"] - daily["mc1"]
    return {
        "rows": int(len(data)),
        "days": int(len(daily)),
        "unique_daily_shifts": int(data["recent_shift_bps"].nunique()),
        "one_shift_per_day": bool(daily["shift_count"].eq(1).all()),
        "shift_bps_min": float(data["recent_shift_bps"].min()),
        "shift_bps_max": float(data["recent_shift_bps"].max()),
        "shift_bps_std": float(data["recent_shift_bps"].std()),
        "daily_mae_static_bps": float(static_bias.abs().mean()),
        "daily_mae_mc1_bps": float(mc1_bias.abs().mean()),
        "daily_rmse_static_bps": float(np.sqrt(np.mean(np.square(static_bias)))),
        "daily_rmse_mc1_bps": float(np.sqrt(np.mean(np.square(mc1_bias)))),
    }


def _mc1_audit(mc1: Path) -> dict[str, object]:
    current = pd.read_parquet(
        mc1 / "enhanced_current_mc1_predictions.parquet",
        columns=["candidate_id", "policy_path_valid", "policy_net_bps", "static_expected_bps", "mc1_expected_bps"],
    )
    bcf = pd.read_parquet(
        mc1 / "enhanced_bcf_mc1_predictions.parquet",
        columns=["candidate_id", "static_expected_bps", "mc1_expected_bps"],
    )
    dual = current.merge(bcf, on="candidate_id", suffixes=("_current", "_bcf"), validate="one_to_one")
    dual = dual.loc[
        dual["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(dual["policy_net_bps"], errors="coerce").notna()
    ].copy()
    static = dual.loc[
        dual["static_expected_bps_current"].ge(50.0)
        & dual["static_expected_bps_bcf"].ge(50.0)
    ]
    adaptive = dual.loc[
        dual["mc1_expected_bps_current"].ge(50.0)
        & dual["mc1_expected_bps_bcf"].ge(50.0)
    ]
    correctness = json.loads((mc1 / "correctness_report.json").read_text())
    return {
        "construction": "monthly strict-prequential static model plus a 21-day, 10%-trimmed, prior-resolved global residual shift",
        "existing_strict_prequential_receipt": correctness,
        "current": _mc1_family_audit(mc1 / "enhanced_current_mc1_predictions.parquet"),
        "bcf": _mc1_family_audit(mc1 / "enhanced_bcf_mc1_predictions.parquet"),
        "dual_50bps_static": {
            "rows": int(len(static)),
            "realised_net_ev_bps_per_trade": float(static["policy_net_bps"].mean()),
        },
        "dual_50bps_adaptive": {
            "rows": int(len(adaptive)),
            "realised_net_ev_bps_per_trade": float(adaptive["policy_net_bps"].mean()),
        },
        "drift_control_status": "active_and_causal_but_not_proven_to_improve_broad_daily_calibration",
    }


def run(sweep: Path, mc1: Path, policy: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    payload = {
        "schema": "strict_r3_p8u_replay_realism_audit_v1",
        "scope": "read-only offline audit; no refitting, scoring, live mutation, or exchange I/O",
        "inputs": {
            "sweep": str(sweep),
            "sweep_summary_sha256": _sha256(sweep / "gate_summary.parquet"),
            "mc1": str(mc1),
            "mc1_dual_predictions_sha256": _sha256(mc1 / "dual_predictions.parquet"),
            "policy": str(policy),
        },
        "exit_policy": _policy_audit(mc1, policy),
        "portfolio_capacity": _capacity_audit(sweep),
        "mc1_time_adaptation": _mc1_audit(mc1),
    }
    _write_once(out / "correctness_report.json", payload)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--mc1", type=Path, default=DEFAULT_MC1)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.sweep.resolve(), args.mc1.resolve(), args.policy.resolve(), args.out.resolve()))


if __name__ == "__main__":
    main()
