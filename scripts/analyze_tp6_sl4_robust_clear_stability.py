#!/usr/bin/env python3
"""Declared cost-buffer stability audit for the exact TP6/SL4 robust label."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=ROOT / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    cols = ["side_name", "label_valid", "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50", "robust_clear_soft_b0_t50", "robust_clear_soft_b25_t50", "robust_clear_soft_b50_t50"]
    x = pd.concat([pd.read_parquet(part, columns=cols) for part in sorted((a.labels / "parts").glob("*.parquet"))], ignore_index=True)
    x = x.loc[x.label_valid.eq(True)]
    rows = []
    for side, group in x.groupby("side_name", observed=True):
        events = group[["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]].to_numpy(int)
        soft = group[["robust_clear_soft_b0_t50", "robust_clear_soft_b25_t50", "robust_clear_soft_b50_t50"]]
        for left, right in ((0, 25), (25, 50), (0, 50)):
            event_left, event_right = events[:, (0, 1, 2)[(0,25,50).index(left)]], events[:, (0, 1, 2)[(0,25,50).index(right)]]
            soft_left, soft_right = soft[f"robust_clear_soft_b{left}_t50"], soft[f"robust_clear_soft_b{right}_t50"]
            rows.append({"side_name": side, "comparison": f"b{left}_vs_b{right}", "rows": len(group),
                         "event_agreement": float((event_left == event_right).mean()),
                         "event_flip_rate": float((event_left != event_right).mean()),
                         "soft_spearman": float(soft_left.corr(soft_right, method="spearman")),
                         "top_decile_jaccard": float(len(set(soft_left.nlargest(int(np.ceil(len(group)*.1))).index) & set(soft_right.nlargest(int(np.ceil(len(group)*.1))).index)) / len(set(soft_left.nlargest(int(np.ceil(len(group)*.1))).index) | set(soft_right.nlargest(int(np.ceil(len(group)*.1))).index)))})
    a.out.mkdir(parents=True)
    result = pd.DataFrame(rows)
    result.to_parquet(a.out / "target_contract_stability.parquet", index=False)
    manifest = {"schema": "tp6_sl4_robust_clear_buffer_stability_v1", "status": "COMPLETED",
                "scope": "declared cost-buffer sensitivity at frozen TP6/SL4/H12 contract; this is not a substitute for a TP/SL geometry sweep",
                "results": result.to_dict("records")}
    (a.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__": main()
