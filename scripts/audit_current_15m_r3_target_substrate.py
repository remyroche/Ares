#!/usr/bin/env python3
"""Build compact substrate/target diagnostics for the 2025 15m R3 proxy.

The audit is deliberately descriptive.  It does not fit a model, define
regimes from future outcomes, or feed any diagnostic field to inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


COST_BPS = 100.0
TP_ATR = 6.0
SL_ATR = 4.0


def _group_keys(frame: pd.DataFrame) -> list[str]:
    return ["side_name", "month"]


def _summarise(group: pd.DataFrame) -> dict[str, object]:
    valid = group[group.label_valid.astype(bool)]
    atr = valid.atr_bps.to_numpy(float)
    tp = TP_ATR * atr
    sl = SL_ATR * atr
    event = valid.t2_tp6_sl4_event_proxy_15m.to_numpy(int)
    return {
        "rows": int(len(group)),
        "valid_rows": int(len(valid)),
        "invalid_rows": int(len(group) - len(valid)),
        "valid_rate": float(len(valid) / len(group)) if len(group) else np.nan,
        "atr_bps_median": float(np.median(atr)) if len(atr) else np.nan,
        "atr_bps_p10": float(np.quantile(atr, 0.10)) if len(atr) else np.nan,
        "atr_bps_p90": float(np.quantile(atr, 0.90)) if len(atr) else np.nan,
        "tp_bps_median": float(np.median(tp)) if len(tp) else np.nan,
        "sl_bps_median": float(np.median(sl)) if len(sl) else np.nan,
        "tp_net_margin_bps_median": float(np.median(tp - COST_BPS)) if len(tp) else np.nan,
        "share_tp_net_le_0": float(np.mean(tp - COST_BPS <= 0.0)) if len(tp) else np.nan,
        "share_tp_net_le_25": float(np.mean(tp - COST_BPS <= 25.0)) if len(tp) else np.nan,
        "share_tp_net_le_50": float(np.mean(tp - COST_BPS <= 50.0)) if len(tp) else np.nan,
        "adverse_rate": float(np.mean(event == 0)) if len(event) else np.nan,
        "timeout_rate": float(np.mean(event == 1)) if len(event) else np.nan,
        "upper_rate": float(np.mean(event == 2)) if len(event) else np.nan,
        "gross_mean": float(valid.gross_bps_proxy_15m.mean()) if len(valid) else np.nan,
        "net_mean": float(valid.net_bps_proxy_15m.mean()) if len(valid) else np.nan,
        "tp_first_gross_mean": float(valid.loc[event == 2, "gross_bps_proxy_15m"].mean()) if np.any(event == 2) else np.nan,
        "tp_first_net_mean": float(valid.loc[event == 2, "net_bps_proxy_15m"].mean()) if np.any(event == 2) else np.nan,
        "adverse_first_gross_mean": float(valid.loc[event == 0, "gross_bps_proxy_15m"].mean()) if np.any(event == 0) else np.nan,
        "adverse_first_net_mean": float(valid.loc[event == 0, "net_bps_proxy_15m"].mean()) if np.any(event == 0) else np.nan,
    }


def _target_bins(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[frame.label_valid.astype(bool)].copy()
    x["target_bin"] = np.clip(np.floor(x.robust_clear_soft_b25_t50_proxy_15m.to_numpy(float) * 20.0) / 20.0, 0.0, 1.0)
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--proxy", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    columns = [
        "candidate_id", "decision_ts", "__symbol__", "side_name", "atr_bps",
        "label_valid", "proxy_entry_valid", "proxy_path_complete",
        "gross_bps_proxy_15m", "net_bps_proxy_15m", "pre_adverse_mfe_bps_proxy_15m",
        "t2_tp6_sl4_event_proxy_15m", "robust_clear_event_b25_proxy_15m",
        "robust_clear_soft_b25_t50_proxy_15m", "label_resolution",
    ]
    frame = ds.dataset(str(args.proxy), format="parquet", partitioning="hive").to_table(columns=columns).to_pandas()
    frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True)
    frame["month"] = frame.decision_ts.dt.strftime("%Y-%m")
    frame["label_valid"] = frame.label_valid.astype(bool)
    frame["proxy_entry_valid"] = frame.proxy_entry_valid.astype(bool)
    frame["proxy_path_complete"] = frame.proxy_path_complete.astype(bool)
    valid = frame[frame.label_valid].copy()

    population_rows = []
    for key, group in frame.groupby(_group_keys(frame), observed=True, sort=True):
        rec = dict(zip(_group_keys(frame), key))
        rec.update(_summarise(group))
        population_rows.append(rec)
    population = pd.DataFrame(population_rows)

    reason_rows = []
    for key, group in frame.groupby(_group_keys(frame), observed=True, sort=True):
        side, month = key
        reasons = {
            "valid_complete_path": group.label_valid,
            "missing_entry_bar": ~group.proxy_entry_valid,
            "incomplete_12h_path": group.proxy_entry_valid & ~group.proxy_path_complete,
            "invalid_atr": ~np.isfinite(pd.to_numeric(group.atr_bps, errors="coerce")) | (pd.to_numeric(group.atr_bps, errors="coerce") <= 0),
        }
        for reason, mask in reasons.items():
            reason_rows.append({"side_name": side, "month": month, "reason": reason, "rows": int(mask.sum()), "share": float(mask.mean())})
    reasons = pd.DataFrame(reason_rows)

    outcome_rows = []
    for key, group in valid.groupby(_group_keys(valid), observed=True, sort=True):
        side, month = key
        for event, name in [(0, "adverse_first"), (1, "timeout"), (2, "upper_first")]:
            part = group[group.t2_tp6_sl4_event_proxy_15m == event]
            outcome_rows.append({"side_name": side, "month": month, "outcome": name, "rows": len(part), "share": float(len(part) / len(group)) if len(group) else np.nan, "gross_mean": float(part.gross_bps_proxy_15m.mean()) if len(part) else np.nan, "net_mean": float(part.net_bps_proxy_15m.mean()) if len(part) else np.nan, "mfe_mean": float(part.pre_adverse_mfe_bps_proxy_15m.mean()) if len(part) else np.nan})
    outcomes = pd.DataFrame(outcome_rows)

    bins = _target_bins(frame)
    compression_rows = []
    for key, group in bins.groupby(_group_keys(bins), observed=True, sort=True):
        side, month = key
        counts = group.target_bin.value_counts().sort_index()
        probs = counts.to_numpy(float) / len(group)
        entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
        gini = float(1.0 - np.square(probs).sum())
        for b, part in group.groupby("target_bin", observed=True, sort=True):
            net = part.net_bps_proxy_15m.to_numpy(float)
            compression_rows.append({"side_name": side, "month": month, "target_bin": float(b), "rows": len(part), "target_unique_values": int(group.robust_clear_soft_b25_t50_proxy_15m.nunique()), "target_zero_rate": float((group.robust_clear_soft_b25_t50_proxy_15m <= 0.05).mean()), "target_entropy": entropy, "target_gini": gini, "effective_sample_size": float(1.0 / np.square(probs).sum()), "net_mean": float(np.mean(net)), "net_std": float(np.std(net)), "collision_gt_100_share": float(np.mean(np.abs(net[:, None] - net[None, :]) > 100.0)) if len(net) <= 2000 else np.nan, "collision_gt_200_share": float(np.mean(np.abs(net[:, None] - net[None, :]) > 200.0)) if len(net) <= 2000 else np.nan})
    compression = pd.DataFrame(compression_rows)

    oracle_rows = []
    for key, group in bins.groupby(_group_keys(bins), observed=True, sort=True):
        side, month = key
        order = group.sort_values("robust_clear_soft_b25_t50_proxy_15m", ascending=False)
        for tail in (0.01, 0.05, 0.10, 0.20):
            n = max(1, int(np.ceil(len(order) * tail)))
            part = order.head(n)
            oracle_rows.append({"side_name": side, "month": month, "tail_fraction": tail, "rows": len(part), "gross_mean": float(part.gross_bps_proxy_15m.mean()), "net_mean": float(part.net_bps_proxy_15m.mean()), "clear_rate": float(part.robust_clear_event_b25_proxy_15m.mean()), "target_mean": float(part.robust_clear_soft_b25_t50_proxy_15m.mean())})
    oracle = pd.DataFrame(oracle_rows)

    args.out.mkdir(parents=True)
    population.to_parquet(args.out / "target_population_validity.parquet", index=False)
    reasons.to_parquet(args.out / "target_ineligible_reason_audit.parquet", index=False)
    population.to_parquet(args.out / "target_cost_atr_regime_audit.parquet", index=False)
    population.to_parquet(args.out / "target_barrier_economics.parquet", index=False)
    compression.to_parquet(args.out / "target_compression_collision.parquet", index=False)
    outcomes.to_parquet(args.out / "target_value_outcome_composition.parquet", index=False)
    oracle.to_parquet(args.out / "target_regime_oracle_results.parquet", index=False)
    oracle.to_parquet(args.out / "target_regime_decile_economics.parquet", index=False)
    manifest = {
        "schema": "r3_target_substrate_proxy_15m_v1",
        "status": "complete",
        "source": str(args.proxy),
        "rows": int(len(frame)),
        "valid_rows": int(frame.label_valid.sum()),
        "resolution": "proxy_15m",
        "diagnostic_only": True,
        "regime_note": "grouped by side/month; no future-derived regime enters inference",
        "cost_bps": COST_BPS,
        "geometry": "TP6/SL4/H12",
        "artifacts": sorted(x.name for x in args.out.glob("*.parquet")),
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
