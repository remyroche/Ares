# P8U CMI Under / Over / Magnitude Retest — Source-Repaired 2024–25

## Decision

**No CMI-selected Under, Over, or Magnitude Meta head advances to 2026.**

All three 2025 strict-OOF screens fail the economic advancement gate: their
best trial has negative mean Top-2 substitution EV relative to the frozen Base
score, and none produces a positive, portable `SStableMeta` result.  The
requested 2026 validation is therefore intentionally not run.  It is a
reserved untouched period, not a rescue set for a failed 2025 selection.

This is a research-only result.  Nothing in the live, MC1, admission,
portfolio, execution, or exchange path was changed.

## Scope and chronology

The retest uses eight source-repaired, long-only monthly feature panels:

`2024-12`, `2025-01`, `2025-02`, `2025-03`, `2025-04`, `2025-05`, `2025-06`,
and `2025-07`.

All panels are generated under the sealed
`cell_local_15m_cache_official_legacy_v2` source-precedence contract.  The
Base score panel and full feature panel are joined by exact
`candidate_id × __decision_ts__ × side_name` identity.  The target-free feature
panels contain 1,412 columns (1,407 feature-coverage rows), have no duplicate
identity, and exclude policy/path outcome fields.

The CMI screen evaluates the common full causal universe conditional on the
nine-field `BASE_EXPLANATION_V1` contract, within timestamp-local Base top-15%
candidates.  Hygiene requires presence across every source month, at least 90%
finite coverage, and nontrivial variance.  It creates a frozen 80-field contract
for each target family.  This retest intentionally does **not** use MDA.

The held OOF months are May, June, and July 2025.  Training labels are resolved
before each fold reserve; target-free held scores are persisted before outcome
metrics are joined.  The `target_free_only` marker in the screen report is
`false` because this is an evaluation run that joins held outcomes after score
persistence.  It is not a score-input leakage flag; the relevant target-free
input and post-score outcome-join checks are all true.

## CMI contracts

Each family had 160 CMI candidates after hygiene and a frozen 80-field Meta
contract.  The leading selected fields are economically plausible but did not
translate into out-of-sample Top-2 improvement.

| Family | First selected fields (abbreviated) |
|---|---|
| Magnitude | `liq_buffer_short_mark_frac`, `dir_path_risk_short_2h`, `loc_bb_channel_pos_24`, ADX directional fields, breadth and high-volatility-state fields |
| Over | `liq_buffer_short_mark_frac`, breadth/VWAP state, `ffd_rv_6h_06`, recovery/drawdown state, BTC return, range/volatility fields |
| Under | `liq_buffer_short_mark_frac`, Bollinger/pivot location, breadth and recovery state, high-volatility age, ADX, market drawdown fields |

The held score files each have 15 score/provenance columns and no policy/path
outcome field.

## Strict-OOF result

The table compares the highest-ranked trial in each CMI family with its frozen
pre-CMI family control.  `Δ` is CMI best minus control.  Positive Top-2 EV and
positive `SStableMeta` are required to advance.

| Family | Best CMI trial | CMI `SStableMeta` | Control | Δ | CMI mean Top-2 EV (bps) | Control | Δ | Worst-week `SMeta` | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Magnitude | `lambda_tail_t12_s1` | -0.0234 | +0.0303 | -0.0537 | -22.38 | -7.22 | -15.15 | -0.0900 | Reject |
| Over | `lambda_tail_t12_s1` | -0.0297 | -0.0457 | +0.0160 | -4.02 | -4.96 | +0.94 | -0.1106 | Reject |
| Under | `lambda_tail_t12_s1` | -0.0709 | -0.0171 | -0.0538 | -33.42 | -25.46 | -7.97 | -0.1242 | Reject |

Additional diagnostics reinforce the decision:

* Magnitude has positive residual IC (+0.0581) and conditional MI (+0.0434),
  but loses 22.38 bps at Top-2 and has negative stability.
* Over is closest to neutral in Top-2 EV (-4.02 bps), but remains negative,
  worsens its worst-week stability versus the control, and has negative residual
  IC (-0.0100).
* Under has the strongest conditional MI (+0.1221) and residual IC (+0.1009),
  but loses 33.42 bps at Top-2 and is the least stable configuration.

Thus, conditional information is present in a predictive sense, but it does not
produce the required economic conversion.  It must not be promoted as a Meta
input or downstream score coordinate.

## Immutable evidence

* Full source-repaired panels:
  * `data_perp/artifacts/strict_r3_p8u_full1400_meta_features_dec24_feb25_source_repaired_completecontext_batched64_20260829_v5_expanded_base_lineage`
  * `data_perp/artifacts/strict_r3_p8u_full1400_meta_features_marjul_source_repaired_completecontext_batched64_20260829_v1`
* CMI handoff:
  `data_perp/artifacts/strict_r3_p8u_uom_cmi_handoff_source_repaired_expandedbase_20260829_v1`
* Frozen contracts and objective handoff:
  `data_perp/artifacts/strict_r3_p8u_uom_objective_handoff_source_repaired_expandedbase_20260829_v1`
* Family objective receipts:
  * `data_perp/artifacts/strict_r3_p8u_uom_magnitude_full1400_objective_source_repaired_20260829_v1`
  * `data_perp/artifacts/strict_r3_p8u_uom_over_full1400_objective_source_repaired_20260829_v1`
  * `data_perp/artifacts/strict_r3_p8u_uom_under_full1400_objective_source_repaired_20260829_v1`
* Frozen pre-CMI control:
  `data_perp/artifacts/strict_r3_p8u_uom_target_grid_f72_source_repaired_expandedbase_20260829_v1/one_winner_per_family_pre_mc1.parquet`

## Follow-up

Do not run 2026 for these heads.  Any future successor should be a separately
predeclared experiment, and must first pass source-repaired multi-month strict
OOF Top-2 economics, week stability, and the frozen-control comparison.
