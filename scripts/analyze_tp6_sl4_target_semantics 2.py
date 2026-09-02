#!/usr/bin/env python3
"""Measure TP6/SL4 B3 target compression, collision, and regime robustness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _deciles(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = [((), frame)] if not groups else frame.groupby(groups, observed=True, sort=True)
    for key, part in grouped:
        key = key if isinstance(key, tuple) else (key,)
        rank = part.target_b3_upper.rank(method="first", pct=True)
        decile = np.minimum(np.ceil(rank * 10).astype(int), 10)
        for bucket, subset in part.groupby(decile, observed=True, sort=True):
            row = {"scope": "global" if not groups else "+".join(groups), "target_decile": int(bucket), "rows": len(subset),
                   "target_mean": float(subset.target_b3_upper.mean()), "gross_bps": float(subset.gross_bps.mean()),
                   "net_bps": float(subset.net_bps.mean()), "upper_rate": float(subset.event_upper.mean()),
                   "lower_rate": float(subset.event_lower.mean()), "timeout_rate": float(subset.event_timeout.mean())}
            row.update(dict(zip(groups, key, strict=True)))
            records.append(row)
    return pd.DataFrame(records)


def _collision(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic within-bin pair sample; avoids an O(n²) fake precision."""
    rng = np.random.default_rng(20260802)
    records: list[dict[str, object]] = []
    groups = ["side_name", "diagnostic_cost_atr_regime", "target_bin"]
    for key, part in frame.groupby(groups, observed=True, sort=True):
        values = part.net_bps.to_numpy(float)
        n = len(values)
        draws = min(20_000, max(0, n * 4))
        if draws:
            left = values[rng.integers(0, n, draws)]
            right = values[rng.integers(0, n, draws)]
            spread = np.abs(left - right)
            collision100, collision200 = float((spread > 100.).mean()), float((spread > 200.).mean())
        else:
            collision100 = collision200 = float("nan")
        records.append({"side_name": key[0], "diagnostic_cost_atr_regime": key[1], "target_bin": int(key[2]), "rows": n,
                        "net_mean_bps": float(values.mean()), "net_std_bps": float(values.std()),
                        "net_p10_bps": float(np.quantile(values, .1)), "net_p90_bps": float(np.quantile(values, .9)),
                        "pair_net_difference_gt_100_rate": collision100, "pair_net_difference_gt_200_rate": collision200,
                        "event_upper_rate": float(part.event_upper.mean()), "event_lower_rate": float(part.event_lower.mean()),
                        "timeout_rate": float(part.event_timeout.mean())})
    return pd.DataFrame(records)


def _distribution(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for groups in ([], ["side_name"], ["side_name", "diagnostic_cost_atr_regime"]):
        grouped = [((), frame)] if not groups else frame.groupby(groups, observed=True, sort=True)
        for key, part in grouped:
            key = key if isinstance(key, tuple) else (key,)
            values = part.target_b3_upper.to_numpy(float)
            counts = pd.Series(np.round(values, 6)).value_counts()
            probability = counts.to_numpy(float) / len(values)
            record = {"scope": "global" if not groups else "+".join(groups), "rows": len(values),
                      "unique_target_values_rounded_1e6": int(len(counts)), "zero_label_rate": float((values == 0).mean()),
                      "entropy_nats": float(-(probability * np.log(np.maximum(probability, 1e-15))).sum()),
                      "effective_sample_size": float(1. / np.square(probability).sum()),
                      "gini_concentration": float(1. - np.square(probability).sum()),
                      "target_p01": float(np.quantile(values, .01)), "target_p50": float(np.quantile(values, .5)), "target_p99": float(np.quantile(values, .99))}
            record.update(dict(zip(groups, key, strict=True)))
            records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    columns = ["candidate_id", "side_name", "diagnostic_cost_atr_regime", "label_valid", "target_b3_upper", "target_bin", "gross_bps", "net_bps", "event_upper", "event_lower", "event_timeout"]
    frame = pd.read_parquet(args.population, columns=columns)
    frame = frame.loc[frame.label_valid].copy()
    if frame.empty or frame.loc[:, ["target_b3_upper", "gross_bps", "net_bps"]].isna().any().any():
        raise ValueError("complete target population must have finite target and economics")
    args.out.mkdir(parents=True)
    distribution = _distribution(frame)
    collision = _collision(frame)
    deciles = pd.concat([_deciles(frame, []), _deciles(frame, ["side_name"]), _deciles(frame, ["diagnostic_cost_atr_regime"]), _deciles(frame, ["side_name", "diagnostic_cost_atr_regime"])], ignore_index=True)
    distribution.to_parquet(args.out / "target_distribution_diagnostics.parquet", index=False)
    collision.to_parquet(args.out / "target_compression_collision.parquet", index=False)
    deciles.to_parquet(args.out / "target_regime_decile_economics.parquet", index=False)
    global_deciles = deciles.loc[deciles.scope.eq("global")].sort_values("target_decile")
    monotone = bool(np.all(np.diff(global_deciles.net_bps.to_numpy(float)) >= -1e-8))
    worst = deciles.groupby("scope", observed=True).net_bps.min().min()
    diagnosis = {"schema": "tp6_sl4_target_semantics_v1", "rows": int(len(frame)), "target": "selected B3 upper membership", "global_net_deciles_monotone": monotone, "worst_decile_net_bps": float(worst), "regime": "cost_to_tp bands are diagnostic-only; no noncausal state is an inference input", "collision_method": "deterministic sampled within-bin pairs, capped at 20k pairs per side/regime/bin"}
    (args.out / "target_diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
