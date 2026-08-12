#!/usr/bin/env python3
"""Compare causal 21-day admission support across frozen score columns."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)


SCORES = (
    "cap120_policy_correction",
    "arm_A_cap120",
    "arm_C_family_state",
    "arm_D_mlp_state",
    "arm_E_near_tie",
    "arm_G_recent_family_reliability",
    "arm_H_support_ood_abstain",
    "arm_J_dynamic_family_mlp",
)
FLOORS = (0.0, 10.0, 25.0, 50.0)
TAILS = (0.005, 0.01, 0.05, 0.10)


def _tail_rows(test: pd.DataFrame, mapped_score: str, raw_score: str, floor: float) -> list[dict[str, object]]:
    eligible = test.loc[test["causal_21d_side_expected_net_bps"].ge(floor)].copy()
    rows: list[dict[str, object]] = []
    for frac in TAILS:
        requested = max(1, int(math.ceil(len(test) * frac)))
        chosen = eligible.sort_values(
            ["causal_21d_side_expected_net_bps", raw_score, "admission_identity"],
            ascending=[False, False, True], kind="stable",
        ).head(requested)
        rows.append({
            "score": raw_score, "floor_bps": floor, "tail": frac,
            "eligible_rows": int(len(eligible)), "selected_rows": int(len(chosen)),
            "mapped_max_test_bps": float(test["causal_21d_side_expected_net_bps"].max()),
            "mapped_selected_mean_bps": float(chosen["causal_21d_side_expected_net_bps"].mean()) if len(chosen) else np.nan,
            "realized_net_bps": float(chosen["policy_net_bps"].mean()) if len(chosen) else np.nan,
            "realized_gross_bps": float(chosen["policy_gross_bps"].mean()) if len(chosen) else np.nan,
            "positive_net_rate": float((chosen["policy_net_bps"] > 0).mean()) if len(chosen) else np.nan,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.input)
    spec = Causal21dAdmissionSpec(
        min_reference_rows=500, bins=20, trim_fraction=0.05,
        mode="pooled_parent_side_shrinkage_v1", side_shrinkage_rows=500.0,
        min_side_reference_rows=20,
    )
    test_mask = source["split"].eq("test")
    summary: list[dict[str, object]] = []
    audits: list[pd.DataFrame] = []
    score_manifest = {}
    for score in SCORES:
        if score not in source.columns:
            continue
        mapped, audit = apply_causal_21d_side_admission(
            source, score_column=score, net_column="policy_net_bps",
            decision_column="__ts__", label_available_column="policy_label_available_ts",
            identity_column="admission_identity", spec=spec,
        )
        test = mapped.loc[test_mask].copy()
        score_manifest[score] = {
            "rows": int(len(mapped)),
            "mapped_test_rows": int(test["causal_21d_side_expected_net_bps"].notna().sum()),
            "max_mapped_test_bps": float(test["causal_21d_side_expected_net_bps"].max()),
            "median_mapped_test_bps": float(test["causal_21d_side_expected_net_bps"].median()),
        }
        for floor in FLOORS:
            summary.extend(_tail_rows(test, "causal_21d_side_expected_net_bps", score, floor))
        audit = audit.assign(score=score)
        audits.append(audit)
    summary_df = pd.DataFrame(summary)
    summary_df.to_parquet(args.out / "score_sweep_metrics.parquet", index=False, compression="zstd")
    pd.concat(audits, ignore_index=True).to_parquet(args.out / "score_sweep_mapping_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "family_admission_score_sweep_v1", "status": "complete",
        "input": str(args.input), "scores": list(score_manifest),
        "spec": dict(spec.__dict__), "floors_bps": list(FLOORS), "tails": list(TAILS),
        "test_rows": int(test_mask.sum()), "selection": "global pooled test tails after causal mapped floor; no timestamp quota",
        "outcome_labels": "policy_net_bps/policy_gross_bps; exact policy labels already materialized",
        "scores_detail": score_manifest,
        "oos_outcomes_used_for_mapping_selection": False,
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
