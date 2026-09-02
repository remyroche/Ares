# P8U Meta GateProxy top-three challenger rerun — 2026-08-31

## Scope

This is the first execution of the frozen operating rule established in
`P8U_META_GATEPROXY_AUDIT_20260831.md`:

```text
incumbent (external anchor, always confirmed)
+ GateProxy P0 top-3 new challengers
→ strict dual MC1 + constrained chronological portfolio
```

For speed, the two previously selected exploration controls (one uncertainty
and one descriptor-diverse candidate) were intentionally excluded.  No HPO
bank was retrained and no candidate was proxy-promoted.  This rerun validates
the new **confirmation funnel mechanics** on the same February–July 2026
historical evaluation window; it is not an independent promotion test.

## Frozen inputs

- GateProxy P0 top-three challengers, ranks 1–3:
  `lgbm_final_00_sparse_control`, `lgbm_final_02_depth3_sparse`, and
  `lgbm_final_08_depth6_guarded`.
- Fixed F72 Base coordinate and target-free Meta score ledgers, August 2025 to
  July 2026.
- Fixed canonical reconciled rich-policy label source.
- Fixed dual MC1 map, `+50 bps` gate, and constrained chronological portfolio.
- The retained Under-F120 depth-4 sparse model is read from its pre-existing
  strict-MC1 receipt, outside the GateProxy ranking domain.

## Exact parity checks

- The rerun contains exactly three candidates, with three workers and the
  frozen `+50 bps` threshold.
- All 36 target-free monthly score panels (three candidates × 12 months) are
  byte-for-dataframe equal to the corresponding panels in the previous full
  confirmation receipt.
- Every selected candidate’s strict-MC1 correctness receipt passed.
- Every constrained portfolio metric matches the corresponding full-bank
  confirmation result exactly (maximum numeric delta `0.0`).

## Strict-MC1 constrained portfolio results

| Role | Trial | Entries | Net EV / trade | Total net bps | Worst month | Worst week | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|
| Incumbent external anchor | `lgbm_hpo_05_depth4_sparse` | 5,006 | **+135.73** | **+679,478.84** | **+68.26** | **+43.83** | -32.75% |
| GateProxy rank 1 challenger | `lgbm_final_00_sparse_control` | 5,142 | +130.75 | +672,341.09 | +65.13 | +44.31 | -28.30% |
| GateProxy rank 2 challenger | `lgbm_final_02_depth3_sparse` | 5,068 | +128.38 | +650,621.35 | +63.94 | +42.86 | -32.68% |
| GateProxy rank 3 challenger | `lgbm_final_08_depth6_guarded` | 4,906 | +132.94 | +652,221.32 | +66.13 | +43.57 | -28.67% |

The top-three funnel retained the best **new** challenger (rank 3), consistent
with `Regret@3 = 0` in the retrospective GateProxy audit.  The incumbent
remains better on net EV per trade, total net contribution, and worst-month
stability; therefore no challenger advances.

## Decision

The top-three challenger funnel is mechanically validated and is the required
fast path for the next **independent** Meta HPO bank.  The incumbent remains
the external confirmed control.  This historical rerun does not change the
canonical Meta contract, any live bundle, or any exchange setting.

## Artifacts

- [Top-three strict-MC1 rerun](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_top3_challenger_confirmation_rerun_20260831_v1/)
- [Rerun summary](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_top3_challenger_confirmation_rerun_20260831_v1/candidate_mc1_summary.parquet)
- [Rerun correctness receipt](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_top3_challenger_confirmation_rerun_20260831_v1/correctness_report.json)
- [GateProxy role audit](P8U_META_GATEPROXY_AUDIT_20260831.md)
