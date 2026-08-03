#!/usr/bin/env python3
"""Reproducibly align the Round-1B universe to the sealed causal/oracle ladder.

This is a diagnostic-only bridge.  It never trains a model: it verifies the
candidate, timestamp and exact-H12 outcome identity of the 75,200 Round-1B
meta-OOS rows, then replays deterministic pooled-global selections from
already-materialised scores.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp" / "artifacts"
ROUND = ART / "sequential_funnel_round1_g0_tau025_20260801_v6"
LADDER = ART / "root_cause_base_residual_learning_20260731_v1"
LEDGER = ART / "root_cause_diagnostic_substrate_20260731_v4" / "diagnostic_row_ledger.parquet"
DEFAULT_OUTPUT = ART / "sequential_funnel_round1b_ceiling_audit_20260801_v1"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def _stable_hash(ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(ids.astype(str).unique())).encode()).hexdigest()


def _selected_metrics(frame: pd.DataFrame, score: str, arm: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fraction in FRACTIONS:
        selected_rows = int(np.ceil(len(frame) * fraction))
        selected = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort").head(selected_rows)
        rows.append(
            {
                "arm": arm,
                "score_column": score,
                "top_fraction": fraction,
                "population_rows": len(frame),
                "selected_rows": len(selected),
                "gross_bps_per_trade": float(selected["gross_bps"].mean()),
                "net_bps_per_trade": float(selected["net_bps"].mean()),
                "cost_bps_per_trade": float((selected["gross_bps"] - selected["net_bps"]).mean()),
            }
        )
    return rows


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output}")

    round_predictions = pd.read_parquet(ROUND / "base_meta_stack_predictions.parquet")
    round_t1 = round_predictions.loc[
        round_predictions["target_arm"].eq("T1_exact_net_huber")
        & round_predictions["model_variant"].eq("base_plus_meta")
    ].copy()
    if len(round_t1) != 75_200 or round_t1["candidate_id"].nunique() != len(round_t1):
        raise ValueError("Round-1 T1 base+meta must contain exactly one score for 75,200 candidates")
    round_t1["gross_bps"] = pd.to_numeric(round_t1["execution_gross_ev_12h"], errors="raise") * 10_000.0
    round_t1["net_bps"] = pd.to_numeric(round_t1["execution_net_ev_12h"], errors="raise") * 10_000.0

    ladder = pd.read_parquet(LADDER / "base_residual_oof_predictions.parquet")
    ladder = ladder.loc[
        ladder["split"].eq("later_oos")
        & ladder["evaluation_scope"].eq("outer_heldout")
        & ladder["model_family"].isin(["causal_capacity_oracle", "future_feature_oracle"])
    ].copy()
    expected = {"causal_capacity_oracle", "future_feature_oracle"}
    counts = ladder.groupby(["model_family", "seed"], observed=True)["candidate_id"].nunique()
    if set(counts.index.get_level_values("model_family")) != expected or not counts.eq(len(round_t1)).all():
        raise ValueError("sealed ladder does not have complete per-seed coverage of Round-1 candidate count")

    # Candidate-id is the identity key.  The raw feature time is the exact
    # cutoff time; this is intentionally one hour before the decision/entry
    # timestamp in the target contract.
    joined = ladder.merge(
        round_t1[["candidate_id", "__ts__", "__decision_ts__", "side_name", "gross_bps", "net_bps"]],
        on="candidate_id", how="outer", validate="many_to_one", indicator=True, suffixes=("_ladder", "_round"),
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("candidate universe mismatch between Round-1 and ladder")
    if not joined["side_name_ladder"].eq(joined["side_name_round"]).all():
        raise ValueError("side mismatch between Round-1 and ladder")
    if not pd.to_datetime(joined["__ts___ladder"], utc=True).eq(pd.to_datetime(joined["__ts___round"], utc=True)).all():
        raise ValueError("raw feature cutoff timestamp mismatch between Round-1 and ladder")
    decision_offset = pd.to_datetime(joined["__decision_ts__"], utc=True) - pd.to_datetime(joined["__ts___ladder"], utc=True)
    if not decision_offset.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("expected decision timestamp to be raw feature cutoff plus one hour")
    if not np.allclose(joined["gross_h12_bps"], joined["gross_bps"], rtol=0.0, atol=1e-10):
        raise ValueError("gross outcome mismatch")
    if not np.allclose(joined["net_h12_bps"], joined["net_bps"], rtol=0.0, atol=1e-10):
        raise ValueError("net outcome mismatch")

    # This aggregation is specified before looking at the economic result: a
    # simple three-seed mean, never an OOS-best-seed pick.
    output_frame = round_t1[["candidate_id", "gross_bps", "net_bps"]].copy()
    output_frame["t1_production_like_stack_bps"] = round_t1["score_bps"].to_numpy(float)
    all_seed_rows: list[dict[str, object]] = []
    for family in ("causal_capacity_oracle", "future_feature_oracle"):
        pivot = ladder.loc[ladder["model_family"].eq(family)].pivot(
            index="candidate_id", columns="seed", values="combined_economic_prediction_bps"
        )
        pivot = pivot.reindex(output_frame["candidate_id"])
        if pivot.isna().any().any():
            raise ValueError(f"{family} has incomplete score coverage after identity join")
        output_frame[f"{family}_seed_mean_bps"] = pivot.mean(axis=1).to_numpy(float)
        for seed in pivot.columns:
            output_frame[f"{family}_{int(seed)}_bps"] = pivot[seed].to_numpy(float)
            all_seed_rows.extend(_selected_metrics(output_frame, f"{family}_{int(seed)}_bps", f"{family}_seed_{int(seed)}"))
    output_frame["realised_gross_oracle_bps"] = output_frame["gross_bps"]

    summary_rows: list[dict[str, object]] = []
    summary_rows.extend(_selected_metrics(output_frame, "t1_production_like_stack_bps", "T1_production_like_stack"))
    summary_rows.extend(_selected_metrics(output_frame, "causal_capacity_oracle_seed_mean_bps", "M6_causal_capacity_seed_mean"))
    summary_rows.extend(_selected_metrics(output_frame, "future_feature_oracle_seed_mean_bps", "M7_future_feature_oracle"))
    summary_rows.extend(_selected_metrics(output_frame, "realised_gross_oracle_bps", "O1_realised_gross_oracle"))
    summary = pd.DataFrame(summary_rows)
    seed_metrics = pd.DataFrame(all_seed_rows)

    output.mkdir(parents=True)
    summary.to_csv(output / "aligned_ceiling_comparison.csv", index=False)
    summary.to_parquet(output / "aligned_ceiling_comparison.parquet", index=False, compression="zstd")
    seed_metrics.to_csv(output / "causal_and_future_seed_sensitivity.csv", index=False)
    identity = {
        "status": "VALID_EXACT_ALIGNMENT",
        "round1_candidate_count": len(round_t1),
        "candidate_id_sha256_sorted": _stable_hash(round_t1["candidate_id"]),
        "candidate_coverage": {family: {str(int(seed)): int(count) for seed, count in counts.loc[family].items()} for family in expected},
        "candidate_identity": "candidate_id exact equality, complete coverage for every M6/M7 seed",
        "feature_cutoff_identity": "Round __ts__ == sealed ladder __ts__ for every row",
        "decision_timestamp_contract": "Round decision_ts == raw feature cutoff __ts__ + 1h for every row",
        "side_identity": "side exact equality for every row",
        "outcome_identity": {
            "gross": "execution_gross_ev_12h*10000 == gross_h12_bps exactly",
            "net": "execution_net_ev_12h*10000 == net_h12_bps exactly",
        },
        "score_contracts": {
            "T1_production_like_stack": "Round-1 direct exact-net Huber base-plus-stopped-gradient-meta score",
            "M6_causal_capacity_seed_mean": "predeclared arithmetic mean of three high-capacity raw-causal LGBM base-plus-gross-residual scores; no OOS-best-seed selection",
            "M7_future_feature_oracle": "hindsight-only LGBM on sealed post-entry fields; non-promotable",
            "O1_realised_gross_oracle": "realised exact H12 gross sorted directly; non-promotable upper bound",
        },
    }
    (output / "identity_audit.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    plan = """# Future-oracle attribution: materialised field plan

All fields below are exact-H12 labels and must remain hindsight-only.  They
are already materialised in `root_cause_exact_h12_execution_target_pack_20260801_v5/supportive_labels.parquet` on the same candidate IDs.  Family-only and LOFO models should retain the sealed chronological protocol of M7, fit per side, and be reported as diagnostics only.

| Family | Primary materialised fields | Notes |
|---|---|---|
| Reachability | `__meaningful_mfe_reached_12h__`, `__mfe_ge_{0_5,1,1_5,2,3,4}atr__`, `competing_risk_event`, `clean_economic_favorable_first` | Separate event occurrence from conditional payoff magnitude. |
| MFE magnitude | `__peak_mfe_atr_12h__`, `__peak_mfe_atr_clip_{6,8}__`, `__log1p_peak_mfe_12h_atr__`, `__mfe_integral_atr_hours_12h__`, `conditional_peak_mfe_atr_given_meaningful_mfe` | Use only after a reachability-only benchmark; avoid letting peak magnitude proxy the realised P&L label without reporting it as an oracle. |
| Adverse ordering / MAE | `adverse_first`, `__mae_before_meaningful_mfe_atr_12h__`, `__pre_mfe_mae_ge_{0_25,0_5,0_75,1,1_5}atr_12h__`, `__meaningful_mfe_before_mae_{0_25,0_5,0_75,1,1_5}atr_12h__`, adverse-trough/recovery fields | Includes ordering as well as magnitude; retain same-minute conflict semantics. |
| Timing | `__time_to_first_meaningful_mfe_hours_12h__`, `__time_to_{50pct,80pct}_peak_mfe_hours_12h__`, `__bars_to_{1,1_5,2}atr__`, `first_{favorable,adverse,event}_minute`, `__future_slope_atr_per_hour_{2h,4h,8h,12h}__` | Split earliest trajectory from eventual magnitude. |
| Persistence / giveback | `__peak_mfe_fraction_above_{50pct,80pct}_12h__`, `__mfe_ratio_to_peak_at_{2h,4h,8h}_12h__`, `__mfe_persistence_path_efficiency_12h__`, `postcost_h0_retained_net`, `postcost_h0_giveback_after_clear` | Legacy M7 already contains the last two fields. |
| Future path quality | `__mfe_mae_path_efficiency_12h__`, `__mfe_integral_path_efficiency_12h__`, `__mfe_timing_path_efficiency_12h__`, `__path_efficiency_to_{1_5atr,2atr,80pct_peak,90pct_peak,first_meaningful_mfe}__` | Keep distinct from MFE magnitude and timing. |
| Future market confirmation | **Not materialised as a distinct future family.** | Do not fabricate this result from future candidate outcomes. It requires a separately sealed future cross-asset/market panel. |

The legacy M7 oracle itself is restricted to ten fields: `postcost_h0_event`, `postcost_h0_favorable_minute`, `postcost_h0_adverse_minute`, `postcost_h0_resolved_minute`, `postcost_h25_favorable_minute`, `postcost_h25_adverse_minute`, `postcost_h25_resolved_minute`, `postcost_h0_retained_net`, `postcost_h0_giveback_after_clear`, and `exit_hour`.  It therefore does **not** test MFE magnitude or future-market-confirmation attribution.  The new exact-H12 support pack is the appropriate materialised source for those family-only/LOFO experiments.
"""
    (output / "future_oracle_attribution_plan.md").write_text(plan)
    manifest = {
        "schema": "sequential_funnel_round1b_ceiling_audit_v1",
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "output": str(output),
        "inputs": {"round1": str(ROUND), "ladder": str(LADDER), "ledger": str(LEDGER)},
        "selection": "one pooled global ranking with deterministic candidate_id ascending ties; no quotas or portfolio constraints",
        "oracle_status": "M7 and O1 are non-promotable hindsight diagnostics",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    run()
