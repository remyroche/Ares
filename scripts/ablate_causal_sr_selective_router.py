#!/usr/bin/env python3
"""Test causal routing between C0 core and C1 core+S/R MC1 outputs.

The C1 map is evaluated only when its decision-time S/R snapshot is present.
All missing-S/R rows retain C0's already-prequential core map.  This is an
offline, no-retraining, no-order challenger using the existing June--August
held predictions and the same portfolio replay as the input study.
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

from scripts import run_canonical_sr_e2_mc1_input_ablation as base


JUN_JUL = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_20260831_v5"
AUGUST = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_august_20260831_v3"
DEFAULT_OUT = ROOT / "data_perp/artifacts/causal_sr_selective_router_20260831_v1"
ADMISSION_BPS = 50.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_period(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    c0 = pd.read_parquet(root / "C0_refit_core_postfeb_target_free_admission.parquet")
    c1 = pd.read_parquet(root / "C1_refit_core_plus_causal_sr_target_free_admission.parquet")
    keep = ["candidate_id", "__decision_ts__", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "sr_snapshot_available"]
    c1 = c1.loc[:, [c for c in keep if c in c1.columns]].copy()
    c0 = c0.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", "bcf_mc1_expected_bps", "current_mc1_expected_bps"]].copy()
    joined = c0.merge(c1, on=["candidate_id", "__decision_ts__"], suffixes=("_c0", "_c1"), validate="one_to_one")
    if "sr_snapshot_available" not in joined:
        raise AssertionError(f"{root}: C1 target-free panel lacks S/R availability")
    labels = pd.read_parquet(root / "C1_refit_core_plus_causal_sr_bcf_mc1_predictions.parquet")
    labels = labels.loc[:, ["candidate_id", *base.POLICY_COLUMNS]].copy()
    if labels["candidate_id"].duplicated().any():
        raise AssertionError(f"{root}: duplicate outcome identity")
    return joined, labels


def _route(panel: pd.DataFrame, arm: str) -> pd.DataFrame:
    panel = panel.copy()
    sr = panel["sr_snapshot_available"].astype("boolean").fillna(False).astype(bool)
    c0_bcf, c0_current = panel["bcf_mc1_expected_bps_c0"], panel["current_mc1_expected_bps_c0"]
    c1_bcf, c1_current = panel["bcf_mc1_expected_bps_c1"], panel["current_mc1_expected_bps_c1"]
    c0_dual = c0_bcf.ge(ADMISSION_BPS) & c0_current.ge(ADMISSION_BPS)
    c1_dual = c1_bcf.ge(ADMISSION_BPS) & c1_current.ge(ADMISSION_BPS)
    if arm == "C0_core":
        choose_c1 = pd.Series(False, index=panel.index)
    elif arm == "C1_all":
        choose_c1 = pd.Series(True, index=panel.index)
    elif arm == "R1_C1_when_sr_available_else_C0":
        choose_c1 = sr
    elif arm == "R2_C1_additive_on_sr_available":
        # C1 can add a causal S/R-observed candidate, but cannot reject an
        # already-admitted C0 candidate.  This separates promotion value from
        # C1's ability to demote C0.
        choose_c1 = sr & c1_dual
    else:
        raise ValueError(arm)
    panel["route_used_c1"] = choose_c1
    panel["routed_bcf_mc1_expected_bps"] = np.where(choose_c1, c1_bcf, c0_bcf)
    panel["routed_current_mc1_expected_bps"] = np.where(choose_c1, c1_current, c0_current)
    if arm == "R2_C1_additive_on_sr_available":
        panel["dual_admitted"] = c0_dual | (sr & c1_dual)
        # Preserve C0 priority for its own existing admission.  Only a C1-only
        # addition takes C1's BCF map as its auction priority.
        panel["auction_priority_bps"] = np.where((~c0_dual) & (sr & c1_dual), c1_bcf, c0_bcf)
    else:
        panel["dual_admitted"] = panel["routed_bcf_mc1_expected_bps"].ge(ADMISSION_BPS) & panel["routed_current_mc1_expected_bps"].ge(ADMISSION_BPS)
        panel["auction_priority_bps"] = panel["routed_bcf_mc1_expected_bps"]
    return panel


def _replay(panel: pd.DataFrame, labels: pd.DataFrame, arm: str, out: Path) -> tuple[dict[str, object], pd.DataFrame]:
    forbidden = base.POLICY_FORBIDDEN.intersection(panel.columns)
    if forbidden:
        raise AssertionError(f"{arm}: target-free routed panel leaked policy fields: {sorted(forbidden)}")
    outcome = panel.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    candidates = base._to_candidates(outcome, admission=panel["dual_admitted"], priority=panel["auction_priority_bps"])
    decisions, equity, _ = base.replay_candidates(
        candidates, base._params(), mode="global_auction", ev_curve=base.CAUSAL_AUCTION_CURVE,
        market_mode="perps", initial_wallet=1000.0,
    )
    if not decisions.empty:
        provenance = candidates.loc[:, ["candidate_id"]].reset_index(drop=True)
        provenance.index.name = "candidate_index"
        decisions = decisions.merge(provenance, on="candidate_index", how="left", validate="many_to_one")
        decisions["policy_outcome_available"] = True
    decisions.to_parquet(out / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
    metric = base._metrics(decisions, equity, arm, "2026_jun_to_aug18")
    metric["dual_admitted_rows"] = int(panel["dual_admitted"].sum())
    metric["routed_c1_rows"] = int(panel["route_used_c1"].sum())
    metric["routed_c1_admitted_rows"] = int((panel["route_used_c1"] & panel["dual_admitted"]).sum())
    accepted = decisions.loc[decisions.get("accepted", pd.Series(index=decisions.index, dtype=bool)).fillna(False).astype(bool)].copy()
    if accepted.empty:
        monthly = pd.DataFrame(columns=["arm", "month", "trades", "net_ev_bps_per_trade", "net_sum_bps"])
    else:
        accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="raise")
        accepted["month"] = accepted["timestamp"].dt.strftime("%Y-%m")
        accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
        monthly = accepted.groupby("month", sort=True).agg(
            trades=("net_bps", "size"), net_ev_bps_per_trade=("net_bps", "mean"), net_sum_bps=("net_bps", "sum"),
        ).reset_index()
        monthly.insert(0, "arm", arm)
    return metric, monthly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True)
    jj_panel, jj_labels = _read_period(JUN_JUL)
    aug_panel, aug_labels = _read_period(AUGUST)
    panel = pd.concat([jj_panel, aug_panel], ignore_index=True)
    labels = pd.concat([jj_labels, aug_labels], ignore_index=True)
    if panel["candidate_id"].duplicated().any() or labels["candidate_id"].duplicated().any():
        raise AssertionError("period stitching changed candidate identity")
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True, errors="raise")
    arms = ("C0_core", "C1_all", "R1_C1_when_sr_available_else_C0", "R2_C1_additive_on_sr_available")
    metrics, monthly = [], []
    for arm in arms:
        routed = _route(panel, arm)
        routed.to_parquet(out / f"{arm}_target_free_admission.parquet", index=False, compression="zstd")
        metric, period = _replay(routed, labels, arm, out)
        metrics.append(metric); monthly.append(period)
    summary = pd.DataFrame(metrics)
    control = summary.loc[summary["arm"].eq("C0_core")].iloc[0]
    for field in ["accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown"]:
        summary[f"delta_vs_C0_{field}"] = pd.to_numeric(summary[field], errors="coerce") - float(control[field])
    summary.to_csv(out / "portfolio_summary.csv", index=False)
    summary.to_parquet(out / "portfolio_summary.parquet", index=False, compression="zstd")
    pd.concat(monthly, ignore_index=True).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal_sr_selective_router_v1",
        "scope": "offline no-retraining challenger; no live/canonical mutation or exchange calls",
        "policy": "same source-aligned 15m parent-policy labels and controlled long-only auction used by C0/C1",
        "admission": "dual BCF/current MC1 >= 50 bps; BCF map auction priority",
        "routing": "C1 MC1 outputs only when sr_snapshot_available was true at decision time; R2 preserves all C0 admissions",
        "inputs": {"jun_jul": {"path": str(JUN_JUL), "manifest_sha256": _sha256(JUN_JUL / "run_manifest.json")}, "august": {"path": str(AUGUST), "manifest_sha256": _sha256(AUGUST / "run_manifest.json")}},
        "status": "complete",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "out": str(out), "summary": summary.to_dict(orient="records")}, default=str))


if __name__ == "__main__":
    main()
