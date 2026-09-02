# TP6/SL4 month-ahead GAM confirmation

## Contract tested

The structural archetypes and GAM are fit prequentially. For each held month,
the target month is scored once using only the preceding one, two, or three
available months. Archetype discovery, leaf matching, cluster selection,
CMI context selection, and GAM fitting exclude the target month. The residual
and consensus heads are then refit before that held month and ranked globally
only after within-month/side percentile normalization.

The intended frozen production contract is:

```text
gam_disagreement = gam_delta_bps
transport_valid => residual/meta receives gam_disagreement
transport_invalid => exact matched control score
```

`gam_residual_bps` is exactly `4 * gam_delta_bps` (maximum absolute deviation
5.5e-14 in the frozen replay), so it is not an additional information source.
GAM does not replace or modulate BaseEV, and mass/count fields are not part of
the proposed production field contract.

## Frozen 2025 evidence before simplification

The earlier one-month gated two-field replay (the previous full-stack confirmation)
reported Top-5 net bps/trade of +16.55 for the hard-gated arm versus -9.65 for
its matched control. The paired-month bootstrap for the hard-gated-minus-
control difference was +27.68 bps, 95% interval [+11.26, +46.82]. This is
encouraging but was obtained after repeated development work on 2025.

## One-field seed/order robustness

The corrected full-stack replay refits 10 deterministic seeds over the same
12 held 2025 months. Every arm is month-normalized before global ranking.

| Arm | Mean Top-5 | Median | MAD | Worst seed | Best seed | Positive seed deltas |
|---|---:|---:|---:|---:|---:|---:|
| Control | -14.14 | -10.70 | 2.53 | -41.18 | -4.66 | — |
| One field, gated (`gam_disagreement`) | -12.77 | -11.60 | 9.05 | -28.96 | +0.91 | 6/10 |
| One field, reversed feature order, gated | -13.36 | -11.95 | 4.81 | -24.48 | -5.68 | 4/10 |

Matched Top-5 uplift versus the seed-matched control:

* normal one-field: mean **+1.37 bps**, median **+5.24 bps**, 6/10 positive;
* reversed order: mean **+0.78 bps**, median **-1.02 bps**, 4/10 positive.

This is materially weaker and less stable than the original single-seed
two-field result. The one-field contract is therefore not promoted from this
development sample; it remains the only defensible candidate for untouched
transport testing.

## Strict month-ahead residual/meta integration

The requested integration was rerun with the unchanged residual/meta stack and
the canonical one-field GAM contract. For every held 2025 month, the GAM was
fit on the immediately preceding month only; the residual and consensus heads
were also refit before the held month. The four matched arms were:

* `control`: no GAM input and the native base anchor;
* `gam_input`: add only `gam_delta_bps` to the residual/meta heads;
* `gam_modulation`: use the gated GAM expected-bps value as the anchor (diagnostic only);
* `gam_input_modulation`: both changes.

The ranking is global after within-month percentile normalization. Net bps
already subtract the frozen 100-bps cost.

| Arm | Top 1% net | Top 2% net | Top 5% net | Top 10% net | Top-5 mean monthly net | Worst month | Positive Top-5 months | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matched control | -27.33 | +0.52 | -4.25 | -25.49 | -6.80 | -152.57 | 6/12 | 0.052 |
| `gam_delta_bps` input | **+4.10** | **+7.73** | -15.40 | -27.12 | -17.11 | -220.92 | 6/12 | 0.052 |
| GAM modulation (diagnostic) | -52.83 | -21.49 | -40.82 | -40.10 | -40.26 | -248.03 | 4/12 | 0.044 |
| Input + modulation | -66.19 | -40.72 | -31.45 | -36.14 | -33.56 | -215.56 | 4/12 | 0.042 |

The single GAM disagreement field is mildly helpful at the extreme Top-1/2
tails but worsens the primary Top-5 objective and increases worst-month loss.
Replacing the anchor with the GAM score is clearly harmful. The result does
not support production modulation or promotion of the GAM input at this stage;
the control remains the winner under the predeclared Top-5 criterion.

The hard transport gate activated in 7 of the 12 held 2025 months. In the
other five months the GAM input was exactly zero and modulation fell back to
the native base anchor, so the comparison does not hide an invalid-transport
fallback behind a separate score.

For reference, the standalone gated GAM window sensitivity was also scored
month-by-month (the next month is never included in fitting):

| GAM fit window | Target months | Gated γ=.25 Top-5 pooled net | Mean monthly Top-5 net | Worst month | Positive months |
|---:|---:|---:|---:|---:|---:|
| 1 preceding month | 19 | -13.94 | -13.94 | -186.43 | 9/19 |
| 2 preceding months | 18 | -34.24 | -34.24 | -175.92 | 8/18 |
| 3 preceding months | 17 | -18.62 | -18.62 | -146.33 | 8/17 |

The one-month window is the least damaging of the three, but none clears
costs. This supports retaining one month as the inference-matched diagnostic
window if the branch is revisited, without treating the GAM as a promoted
economic ranker.

## Invalid-month abstention diagnostic

On the first seed, simply abstaining when the target month's transport gate is
invalid retains 5,964 of 10,224 rows (58.3% exposure):

| Tail | Net bps/trade | Trades |
|---:|---:|---:|
| Top 0.5% | -19.25 | 30 |
| Top 1% | -13.20 | 60 |
| Top 2% | +3.14 | 120 |
| Top 5% | **+4.23** | 299 |
| Top 10% | -8.27 | 597 |

The validity gate itself has useful environment-selection information, but it
does not by itself create a positive Top-5 strategy.

## Placebo attribution

The 200-seed within-month permutation placebo is the residual-head-only
experiment, not the final full-stack score. Its real reference remains
Top-1 +82.89 and Top-5 +17.94 bps/trade. Empirical p-values are:

| Metric | Real | Placebo mean | p(real or higher) |
|---|---:|---:|---:|
| Top-1 | +82.89 | +16.69 | **0.0199** |
| Top-5 | +17.94 | +4.40 | **0.0746** |
| Mean monthly Top-5 | +2.78 | +3.60 | 0.5274 |

Thus attribution is suggestive at Top-5 and stronger at Top-1, but it cannot
be read as significance for the final hard-gated full-stack arm.

## Untouched chronological transport status

The scan found a suitable later candidate population with exact outcomes:
the 2026-07-20 through 2026-07-23 population contains 5,760 rows and both
sides. It does **not** contain the frozen structural/base contract required to
compute the unchanged GAM disagreement: base expected bps, structural leaf
assignments/catalogues, archetype exposures, transport validity, and GAM
disagreement are absent. Other May--July 2026 artifacts use different score,
feature, or label contracts and cannot be substituted without invalidating the
transport test.

At the time of the initial scan, the untouched OOS result was not claimable.
The exact blocker and field inventory from that scan are recorded in
`data_perp/artifacts/tp6_sl4_gam_transport_readiness_20260815_v2/transport_readiness.json`.
The audit rejects generic `execution_net_ev_12h` and adapter/base scores as
substitutes for the exact TP6/SL4 labels and frozen structural base output.

## July 2026 unchanged-contract transport result

The canonical R3 TP6/SL4 source was subsequently found to contain valid 2026
rows and exact labels. The same residual/path contract was materialized
prequentially for January--July 2026. July was scored once using only June for
archetype/cluster/GAM fitting and only pre-July rows for the residual/meta fit.
The production-style field contract was one `gam_delta_bps` field, gamma 0.25,
with no BaseEV modulation.

This is held out from the frozen GAM branch and its target-month outcomes are
not used by the run. It is not claimed as a globally untouched market period:
the repository contains other July 2026 research artifacts, so a future
post-July data release remains the final independent confirmation gate.

| Arm | Top 1% net | Top 5% net | Top 10% net | Rank IC |
|---|---:|---:|---:|---:|
| Matched control | -392.21 bps | -116.17 bps | -108.37 bps | 0.021 |
| One-field gated GAM | -295.98 bps | -112.37 bps | -54.06 bps | 0.027 |

The GAM improves the July ranking relative to control at every reported tail,
but remains economically negative after the 100-bps cost. This is a failed
economic promotion gate, not evidence to retune the contract on July.

The transport run used 852 July rows, 23,131 pre-July residual/meta training
rows, and 100% transport-valid target rows. Correctness assertions confirm
that July outcomes were not used in the GAM or meta fit, global ranking was
performed after score generation, and invalid transport rows would have used
the exact control score.

## Exact later-population label recovery

The previously identified July 20--23 candidate population has now been
relabelled independently under the frozen TP6/SL4/H12 contract.  The relabeler
uses the signal-close +1h decision timestamp, the exact next-minute entry open,
causal Wilder ATR(14) from completed hourly candles, TP=+6 ATR, SL=-4 ATR,
adverse same-minute precedence, and a single 100-bps cost subtraction.

It contains 14,400 candidate rows (7,200 per side).  13,562 rows (94.18%) have
complete exact labels.  The remaining rows are retained as explicit
`target_invalid` coverage: 646 lack a causal ATR(14) substrate at the signal
cutoff and 192 belong to TRUMP/USD:USD, whose Kraken one-minute source contains
an unreadable fragment.  No alternate exchange or 15-minute data was used as a
silent substitute.  Invalid rows carry null economic/R3 targets.

The older 5,760-row execution-policy label table was not reused: on its
5,432-row exact-label overlap, its net outcome has Spearman correlation 0.802
with the exact TP6/SL4 net outcome, but it is a trailing/full-stop policy with
different geometry and payoff units.  Correlation is not contract identity;
the frozen transport test therefore uses the exact sidecar only.

This recovers the exact label substrate, but it does **not** make the period a
valid frozen-GAM transport test yet.  The later population still lacks 17
long-side and 18 short-side fields from the 32-field F0 contract, most of the
frozen context fields, the serialized frozen F0 model, structural leaf
assignments, archetype posteriors, transport validity, and the one-field GAM
disagreement.  The exact contract-recovery audit is
`data_perp/artifacts/tp6_sl4_frozen_f0_recovery_audit_20260815_v1/frozen_f0_recovery_audit.json`.

The native Kraken 1h/OI/funding/order-book generator was attempted in bounded
symbol chunks.  It reached roughly 1.3 GB working-set memory even for one
symbol before an atomic feature chunk could be emitted.  We therefore did not
skip causal transforms, fill missing fields, or substitute a newer model.  The
final later OOS remains fail-closed until the same graph is run in a larger
memory worker (or an equivalent frozen transform-state artifact is recovered).

## Decision

The month-ahead implementation is correct and reusable. The economic uplift
seen in the original two-field hard-gated replay is not robust enough to freeze
as a production improvement after the one-field seed/order test. Keep the
canonical contract narrow (one disagreement field plus the causal validity
gate), materialize that same contract on the untouched July population, and
run the final transport OOS without HPO or post-test tuning. Do not use the
200-seed residual-head placebo to promote the full stack.

## Artifacts

* [Month-ahead GAM implementation](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_rolling_archetype_gam_oos.py)
* [Residual/meta integration](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_rolling_gam_residual_integration.py)
* [Strict one-field month-ahead integration artifacts](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_rolling_gam_residual_integration_20260815_v4/metrics_global.parquet)
* [Strict integration month stability](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_rolling_gam_residual_integration_20260815_v4/metrics_stability.parquet)
* [One-field seed/order replay](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_gam_onefield_robustness.py)
* [One-field robustness artifacts](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gam_onefield_robustness_20260815_v1/seed_order_summary.parquet)
* [200-seed placebo p-values](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gamres_placebo_distribution_20260815_v2/placebo_empirical_pvalues.parquet)
* [Transport-readiness audit](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gam_transport_readiness_20260815_v2/transport_readiness.json)
* [Untouched July 2026 report](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gam_untouched_oos_2026_20260815_v3/TP6_SL4_GAM_UNTOUCHED_OOS_2026_REPORT.md)
* [Untouched July 2026 metrics](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gam_untouched_oos_2026_20260815_v3/untouched_oos_metrics.parquet)
* [Exact later-population label sidecar](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/assembled_exact_labels/run_manifest.json)
* [Later label coverage](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/assembled_exact_labels/coverage.parquet)
* [Updated transport-readiness audit](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gam_transport_readiness_20260815_v3/transport_readiness.json)
* [Frozen F0 recovery audit](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_f0_recovery_audit_20260815_v1/frozen_f0_recovery_audit.json)
* [Later F0/context materializer](/Users/remyroche/Documents/Ares/scripts/materialize_tp6_sl4_later_f0_context.py)
