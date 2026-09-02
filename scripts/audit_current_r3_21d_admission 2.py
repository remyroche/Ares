#!/usr/bin/env python3
"""Apply the canonical causal 21-day side-local admission map to R3 OOF."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
    pooled_global_admission_comparison,
)
from extreme_price_movements.stage_i_ranking import RANKING_POLICY, stable_stage_i_rank_frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oof", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    frame = pd.read_parquet(args.oof, columns=["candidate_id", "decision_ts", "label_available_ts", "side_name", "score", "net_bps", "gross_bps", "robust_clear_event_b25"])
    frame.decision_ts = pd.to_datetime(frame.decision_ts, utc=True)
    frame.label_available_ts = pd.to_datetime(frame.label_available_ts, utc=True)
    frame = frame.loc[frame.side_name.astype(str).str.lower().eq("long")].copy()
    if frame.empty:
        raise ValueError("long-only admission audit has no long rows")
    spec = Causal21dAdmissionSpec(window_days=21, min_reference_rows=500, bins=20, trim_fraction=0.05, net_floor_bps=50.0)
    mapped, audit = apply_causal_21d_side_admission(
        frame,
        score_column="score",
        net_column="net_bps",
        decision_column="decision_ts",
        label_available_column="label_available_ts",
        identity_column="candidate_id",
        spec=spec,
    )
    # The active workstream is explicitly long-only.  The canonical mapper
    # emits empty audit snapshots for the absent short side; do not persist
    # those rows as if they were part of this experiment.
    audit = audit.loc[audit["side_name"].astype(str).str.lower().eq("long")].copy()
    comparison = pooled_global_admission_comparison(
        mapped,
        raw_score_column="score",
        net_column="net_bps",
        gross_column="gross_bps",
        identity_column="candidate_id",
        top_fractions=(0.01, 0.05, 0.10, 0.20),
        original_population_rows=len(mapped),
    )
    # The 21-day map is a gate.  The live score remains the ranking variable
    # after admission; keep the map-ranked comparison only as a diagnostic.
    admitted = mapped[
        mapped["causal_21d_side_admitted_ge_50bps"].astype(bool)
        & mapped["causal_21d_side_expected_net_bps"].notna()
    ]
    post_gate_rows = []
    for fraction in (0.01, 0.05, 0.10, 0.20):
        requested = max(1, int(np.ceil(len(mapped) * fraction)))
        selected = stable_stage_i_rank_frame(
            admitted,
            score_column="score",
            candidate_id_column="candidate_id",
            decision_column="decision_ts",
        ).head(min(len(admitted), requested))
        post_gate_rows.append(
            {
                "comparison": "with_admission_raw_score_global",
                "top_fraction_of_original_population": fraction,
                "original_population_rows": len(mapped),
                "full_candidate_population_rows": len(mapped),
                "strict_oof_scored_population_rows": len(mapped),
                "requested_rows_from_full_candidate_denominator": requested,
                "eligible_rows": len(admitted),
                "selected_rows": len(selected),
                "mean_realised_net_bps": float(selected["net_bps"].mean()) if len(selected) else np.nan,
                "mean_realised_gross_bps": float(selected["gross_bps"].mean()) if len(selected) else np.nan,
                "selected_long_rows": int(selected.side_name.eq("long").sum()) if len(selected) else 0,
                "selected_short_rows": 0,
                "ranking_tie_policy": RANKING_POLICY,
            }
        )
    comparison = pd.concat([comparison, pd.DataFrame(post_gate_rows)], ignore_index=True)
    mapped["month"] = mapped.decision_ts.dt.strftime("%Y-%m")
    mapped["admitted"] = mapped.causal_21d_side_admitted_ge_50bps.astype(bool)
    admission_month = mapped.groupby(["month", "side_name"], observed=True).agg(
        rows=("candidate_id", "size"), admitted_rows=("admitted", "sum"),
        admission_rate=("admitted", "mean"), mapped_expected_net_mean=("causal_21d_side_expected_net_bps", "mean"),
        realised_net_mean=("net_bps", "mean"), realised_gross_mean=("gross_bps", "mean"),
    ).reset_index()
    # Global tail membership after admission, including month/side attribution.
    admitted = mapped[mapped.admitted & mapped.causal_21d_side_expected_net_bps.notna()].copy()
    tail_rows = []
    for frac in (0.01, 0.05, 0.10, 0.20):
        n = min(len(admitted), max(1, int(np.ceil(len(mapped) * frac))))
        selected = admitted.nlargest(n, "causal_21d_side_expected_net_bps")
        selected = selected.assign(tail_fraction=frac)
        tail_rows.append(selected[["candidate_id", "decision_ts", "side_name", "month", "tail_fraction", "causal_21d_side_expected_net_bps", "gross_bps", "net_bps"]])
    membership = pd.concat(tail_rows, ignore_index=True) if tail_rows else pd.DataFrame()

    # Quota-preserving diagnostic: the canonical admission rule is an
    # absolute threshold, so it can return fewer rows than a requested global
    # tail.  Report the raw population and conditional admitted tails
    # separately instead of silently treating the admitted set as every quota.
    tail_metrics = []
    month_tail_metrics = []
    tail_fractions = (0.01, 0.05, 0.10, 0.20, 0.40, 1.0)
    ranking_specs = (
        ("all_population_raw_score", mapped, "score"),
        (
            "admitted_mapped_expected_net",
            admitted,
            "causal_21d_side_expected_net_bps",
        ),
        ("admitted_raw_score", admitted, "score"),
    )
    for rank_mode, population, ranking_column in ranking_specs:
        population = population.loc[population[ranking_column].notna()].copy()
        if population.empty:
            continue
        for fraction in tail_fractions:
            n = max(1, int(np.ceil(len(population) * fraction)))
            selected = population.nlargest(n, ranking_column)
            month_stats = selected.groupby("month", observed=True).agg(
                rows=("candidate_id", "size"),
                gross_bps=("gross_bps", "mean"),
                net_bps=("net_bps", "mean"),
            ).reset_index()
            tail_metrics.append(
                {
                    "rank_mode": rank_mode,
                    "ranking_column": ranking_column,
                    "tail_fraction": fraction,
                    "population_rows": len(population),
                    "selected_rows": len(selected),
                    "mean_rank_value": float(selected[ranking_column].mean()),
                    "mean_realised_gross_bps": float(selected["gross_bps"].mean()),
                    "mean_realised_net_bps": float(selected["net_bps"].mean()),
                    "month_count": int(len(month_stats)),
                    "positive_month_count": int((month_stats["net_bps"] > 0).sum()),
                    "worst_month_net_bps": float(month_stats["net_bps"].min()),
                    "median_month_net_bps": float(month_stats["net_bps"].median()),
                }
            )
            month_stats["rank_mode"] = rank_mode
            month_stats["tail_fraction"] = fraction
            month_stats["ranking_column"] = ranking_column
            month_tail_metrics.append(month_stats)
    tail_metrics = pd.DataFrame(tail_metrics)
    month_tail_metrics = (
        pd.concat(month_tail_metrics, ignore_index=True)
        if month_tail_metrics
        else pd.DataFrame()
    )
    args.out.mkdir(parents=True)
    mapped.to_parquet(args.out / "admission_oof_predictions.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out / "admission_audit.parquet", index=False, compression="zstd")
    comparison.to_parquet(args.out / "admission_comparison.parquet", index=False, compression="zstd")
    admission_month.to_parquet(args.out / "admission_month_side_metrics.parquet", index=False, compression="zstd")
    membership.to_parquet(args.out / "global_tail_membership.parquet", index=False, compression="zstd")
    tail_metrics.to_parquet(args.out / "admission_tail_metrics.parquet", index=False, compression="zstd")
    month_tail_metrics.to_parquet(args.out / "admission_month_tail_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "r3_causal_21d_admission_audit_v1",
        "status": "complete",
        "input": str(args.oof),
        "contract": "long-only 21-calendar-day pooled-parent/side-shrunk map; 20 rank bins; 5% trim; 50 bps floor; labels strictly prior-resolved",
        "side_scope": ["long"],
        "rows": len(mapped),
        "admitted_rows": int(mapped.admitted.sum()),
        "admission_rate": float(mapped.admitted.mean()),
        "outputs": sorted(x.name for x in args.out.glob("*.parquet")),
        "tail_diagnostic": "raw global, admitted mapped-EV, and admitted raw-score quota-preserving tails",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
