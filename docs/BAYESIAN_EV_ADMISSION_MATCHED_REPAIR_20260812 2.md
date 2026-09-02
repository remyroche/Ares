# Bayesian EV-admission matched repair — 2026-08-12

## Decision

**No Bayesian correction advances.**

Every arm was compared with the unmodified strict-R3 score on the identical
long-only, point-in-time candidate population.  The correction was explicitly
allowed to alter the causal EV-admission map.  Advancement requires positive
matched realised-policy-net uplift at all-admitted and Top-5%, with credible
Top-1/Top-2 support and no material selected-set instability.  None qualifies.

## Correct matched contract

The replay is
`strict_r3_lockstep_exactreserve_sourcealigned_policyexact_long_2025_janmar_20260812_v2`:

- 312,955 January--March 2025 target-free candidates;
- four strict prequential 28-day producer bundles;
- held outcomes joined only after score generation;
- no held-window percentile operation;
- one frozen October--December 2024 geometry/K9 bundle;
- canonical policy-net labels;
- canonical admission partitioned by exact score-family × conversion × upstream
  × geometry producer vintage, with prior-resolved labels only and a 50-bps
  floor.

The completed base ledger has 299,035 mapped rows; 13,920 insufficient-support
rows fail closed.

For a Bayesian model trained through 2025-02-26 and held out from 2025-02-26
through 2025-04-01, all earlier full-ledger rows retain `final_score` exactly.
Only the held model's timestamp-top-30% candidates receive the bounded score
correction.  Hence the altered map has the actual prior 21-day history it
would have live.

## Matched results

Values are corrected minus control realised net bps/trade.  `Δn` is the change
in selected rows, and Jaccard is selected-ID overlap.  Apparent extreme-tail
gains with a few changed rows are diagnostic only.

| Arm | All admitted | Top 1% | Top 2% | Top 5% | Result |
|---|---:|---:|---:|---:|---|
| B5, timestamp top-30%, α=.0125 | −0.14 | −31.37 | +13.49 | +1.55 | reject |
| B5, timestamp top-30%, α=.025 | −0.39 | −31.96 | −10.86 | −35.17 | reject |
| B5, timestamp top-30%, α=.05 | −0.34 | −38.86 | −14.62 | −52.50 | reject |
| B5, legacy global top-30%, α=.025 | −0.08 | −6.74 | −0.06 | −42.80 | reject |
| B1, timestamp top-30%, all 56 fields, α=.05 | −0.51 | −12.91 | −27.08 | −49.11 | reject |
| B1, timestamp top-30%, all 56 fields, α=.10 | −1.02 | −44.42 | −6.23 | −77.75 | reject |
| B1, timestamp top-30%, all 56 fields, α=.20 | −4.56 | −68.41 | +22.81 | −140.11 | reject |
| B1, timestamp top-30%, train-only residual-MI-12, α=.025 | −0.11 | −51.85 | −4.90 | −7.73 | reject |
| B1, timestamp top-30%, train-only residual-MI-12, α=.05 | −0.40 | −47.37 | −11.16 | −46.81 | reject |
| B1, timestamp top-30%, train-only residual-MI-12, α=.10 | −0.98 | −17.92 | −1.00 | −97.25 | reject |
| B1, MI-12 timestamp rank-delta, α=.05 | −5.70 | −16.34 | −19.18 | −60.19 | reject |
| B1, MI-12 timestamp rank-delta, α=.10 | −7.15 | +14.30 | −3.86 | −74.64 | reject |

The B1 MI-12 selection was train-only.  Its retained fields were:

1. `k9_cluster_timestamp_support_weighted`
2. `rule_support_p95`
3. `correctness_raw`
4. `k9_cluster_timestamp_support_p05`
5. `correctness_rank`
6. `conditional_consensus_rank`
7. `rule_support_median`
8. `rule_support_p50`
9. `model_ood_mahalanobis_diag`
10. `k9_ood_distance`
11. `k9_cluster_timestamp_ood_weighted`
12. `rule_support_contribution_weighted`

This supports the view that support/OOD state has residual information, but it
does not establish that a bounded correction of the live score produces a
usable economic uplift under the current admission contract.

The rank-delta arms specifically preserve the base score level more closely
than the additive formulation while reordering only its timestamp-top-30%
domain.  They still increased admissions by about 1,000 rows and lost 5.7--7.2
bps/trade all-admitted and 60--75 bps/trade at Top-5%.  The failure is thus not
explained merely by an additive score-level shift.

## Less-noisy target check

A final compact Beta--Binomial residual-event arm was fitted using only
training labels for:

```text
policy net − causal mapped EV >= +50 bps
policy net − causal mapped EV <= −100 bps
```

It selected 12 support/OOD/path fields using train-only binned MI and excluded
the mapped-EV anchor itself from inference features to avoid circularity.  It
failed its held-out standalone gate before admission replay: within the base
timestamp-top-30% domain its policy-net Spearman was only `+0.034`, compared
with `+0.170` for the base score, and its own top tails were weaker.  It is
rejected without an admission-map replay.

## Why older positive Bayesian figures do not promote

The historical B1 score-correction result that appeared positive used a
different score surface and a generic admission-map construction.  The old
and current score surfaces have already been measured as only weakly
comparable (roughly 0.42 rank correlation over the 2025 intersection and 0.34
over 2026).  Reproducing B1 on the current strict producer-vintage ledger is
the relevant reconciliation, and it is negative.

The old result is therefore research provenance, not canonical performance
evidence.

## Implemented safeguards and reusable tools

- `scripts/apply_strict_r3_current_admission.py`
  applies the canonical exact-producer-vintage causal admission map.
- `scripts/materialize_strict_r3_bayesian_full_ledger_correction.py`
  preserves the baseline score before the Bayesian holdout and applies a
  correction only where the posterior exists.
- `scripts/evaluate_strict_r3_ledger_bayesian_admission.py`
  now writes `matched_uplift.parquet` and
  `matched_selection_overlap.parquet`, including month and ISO-week rows;
  it uses the exact current producer-vintage map whenever lineage is present.
- `scripts/run_strict_r3_current_exact_b5_fold.py`
  now supports timestamp-local versus legacy global training selection,
  selectable frozen Bayesian specifications, and optional train-only
  residual-MI compact selection.
- `scripts/run_strict_r3_beta_residual_event_fold.py`
  is the causal, compact Beta--Binomial residual-event control.  Its target
  anchor is explicitly prohibited from the feature contract; regression tests
  cover that exclusion and timestamp-local top-30% selection.

## Next evidence needed before reopening Bayesian promotion

1. Extend the same repaired producer contract over later untouched blocks;
   Q1 alone is too short to tune a correction.
2. Freeze the correction form and strength on an earlier block, then test it
   only on a later block.  Do not select from the table above.
3. If Bayesian work is resumed, compare a less-noisy reliability target (for
   example a shrunk adverse/positive residual event) against the current
   ideal-lambda target, still measured only as matched admission uplift.
