#!/usr/bin/env python3
"""Compare TP6/SL4 target-repair oracle ordering without fitting a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(frame: pd.DataFrame, score: str, target: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for groups in ([], ["side_name"], ["diagnostic_cost_atr_regime"]):
        grouped = [((), frame)] if not groups else frame.groupby(groups, observed=True, sort=True)
        for key, part in grouped:
            key = key if isinstance(key, tuple) else (key,)
            ordered = part.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort")
            for fraction in (.01, .05, .10, .20):
                selected = ordered.head(max(1, int(np.ceil(len(ordered) * fraction))))
                row = {"target": target, "scope": "global" if not groups else "+".join(groups), "top_fraction": fraction,
                       "rows": len(part), "selected_rows": len(selected), "gross_bps": float(selected.gross_bps.mean()), "net_bps": float(selected.net_bps.mean())}
                row.update(dict(zip(groups, key, strict=True)))
                records.append(row)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--robust-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    base_cols = ["candidate_id", "side_name", "label_valid", "diagnostic_cost_atr_regime", "gross_bps", "net_bps", "target_b3_upper"]
    base = pd.read_parquet(args.population, columns=base_cols)
    labels = pd.concat([pd.read_parquet(path, columns=["candidate_id", "label_valid", "robust_clear_soft_b0_t50", "robust_clear_soft_b25_t50", "robust_clear_soft_b50_t50"]) for path in sorted((args.robust_root / "parts").glob("*.parquet"))], ignore_index=True)
    if labels.candidate_id.duplicated().any():
        raise ValueError("robust-clear candidate identity must be unique")
    frame = base.merge(labels, on="candidate_id", how="left", validate="one_to_one", suffixes=("", "_robust"))
    if not frame.label_valid.eq(frame.label_valid_robust).all():
        raise ValueError("robust-clear validity differs from the frozen TP6/SL4 population")
    frame = frame.loc[frame.label_valid].copy()
    # R1 is intentionally retained as the direct realised-net negative control:
    # it is useful as an oracle but should not be promoted unless causal fits
    # demonstrate learnability.
    frame["r1_direct_net_margin"] = 1. / (1. + np.exp(-np.clip(frame.net_bps / 50., -35., 35.)))
    frame["r4_ordinal_net_margin"] = np.select([frame.net_bps.le(-200), frame.net_bps.le(0), frame.net_bps.le(50)], [0., 1., 2.], default=3.)
    targets = {"R0_B3_control": "target_b3_upper", "R1_direct_net_control": "r1_direct_net_margin", "R2_robust_clear_b0_t50": "robust_clear_soft_b0_t50", "R2_robust_clear_b25_t50": "robust_clear_soft_b25_t50", "R2_robust_clear_b50_t50": "robust_clear_soft_b50_t50", "R4_ordinal_net_margin": "r4_ordinal_net_margin"}
    rows: list[dict[str, object]] = []
    for name, column in targets.items():
        rows.extend(_metrics(frame, column, name))
    args.out.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(args.out / "target_regime_oracle_results.parquet", index=False)
    frame[["candidate_id", "side_name", "label_valid", "net_bps", "target_b3_upper", "robust_clear_soft_b0_t50", "robust_clear_soft_b25_t50", "robust_clear_soft_b50_t50", "r1_direct_net_margin", "r4_ordinal_net_margin"]].to_parquet(args.out / "target_repair_labels.parquet", index=False, compression="zstd")
    global_top10 = pd.DataFrame(rows).query("scope == 'global' and top_fraction == .1")[['target','net_bps']].sort_values('net_bps', ascending=False)
    manifest = {"schema": "tp6_sl4_target_repair_oracles_v1", "rows": int(len(frame)), "status": "TARGET_ORACLE_ONLY_NO_MODEL_FIT", "targets": targets, "r1_interpretation": "direct realised-net negative control, not a causal promotion candidate", "r2_interpretation": "pre-adverse robust-clear soft target; buffers are cost margin requirements", "global_top10_net_bps": global_top10.to_dict(orient="records")}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
