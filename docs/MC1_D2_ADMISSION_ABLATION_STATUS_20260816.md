# MC1_d2 admission ablation status — 2026-08-16

## Scope

This is an offline, long-only MC1 admission-mapper research receipt.  It keeps
the upstream strict-R3 base, consensus, execution outcome, +50-bps admission
rule, and final-score auction fixed.  It does not change any live artifact.

The frozen control is the six-input depth-two HGB mapper:

```text
final_score
base_rank42
conditional_consensus_rank
upstream
ordinary_shadow_consensus_rank
correctness_rank
```

Its static model is HGB depth 2 / 80 iterations / learning rate 0.04 / L2 20 /
minimum leaf 100 / seed 1729.  Its dynamic component is the 21-day 10%-trimmed
global residual shift.  The validation below makes label availability strict:
`policy_label_available_ts < decision boundary`.

## Strict control versus legacy-compatible boundary

35,267 source labels resolve exactly at a UTC daily boundary.  Excluding them
is the conservative rule.  The strict result remains economically strong and
is the control for the new target/loss and cadence work.

| Period | Accepted | Net bps/trade | Net sum bps | Worst month | Worst week |
|---|---:|---:|---:|---:|---:|
| 2025 strict control | 6,902 | +182.39 | +1,258,824 | +150.77 | +13.19 |
| 2026 Jan–Jul strict control | 3,741 | +160.51 | +600,480 | +41.86 | -2.25 |
| 2025 legacy-compatible `<=` boundary | 6,924 | +177.58 | +1,229,591 | +154.72 | +12.30 |
| 2026 Jan–Jul legacy-compatible `<=` boundary | 3,667 | +163.40 | +599,201 | +12.11 | -13.46 |

This demonstrates boundary sensitivity, not evidence of leakage: the strict
rule is used for all new comparisons.

Artifacts:

- `data_perp/artifacts/strict_r3_mc1_d2_historical_strictlt_2025_2026_20260816_v1`
- `data_perp/artifacts/strict_r3_mc1_d2_historical_strictlt_portfolio_20260816_v1`

## MC1-only target/loss funnel

All five challengers retain exactly the six frozen MC1 inputs, the original
full-universe day-balanced history, 2025-only chronological HPO, strict
availability, frozen daily residual adjustment, +50-bps admission, and
final-score auction.  Huber/L1 use 2nd–98th percentile clipping; `asin` also
uses a bounded arcsine transform.  Ordinal has six net-EV bins with centres
`[-300, -125, 0, 100, 200, 350]` bps.

| Arm | 2025 net bps/trade | Delta vs strict | 2026 net bps/trade | Delta vs strict | Decision |
|---|---:|---:|---:|---:|---|
| Frozen strict MC1_d2 control | +182.39 | — | +160.51 | — | retain control |
| Huber, clipped net | no +50-bps admissions | — | no +50-bps admissions | — | reject: output scale unsuitable |
| Huber, arcsine-winsorized net | +162.64 | -19.74 | +142.16 | -18.36 | reject |
| L1, clipped net | +145.95 | -36.43 | +134.63 | -25.88 | reject |
| L1, arcsine-winsorized net | +144.38 | -38.01 | +136.47 | -24.04 | reject |
| Six-bin ordinal net | +163.01 | -19.38 | +149.30 | -11.22 | reject |

The closest challenger, Huber-arcsine, had more accepted rows but lower total
net contribution in both eras.  No target/loss model advances.

Artifacts:

- `data_perp/artifacts/strict_r3_mc1_d2_target_loss_ablation_2025hpo_strictlt_20260816_v1`
- `data_perp/artifacts/strict_r3_mc1_d2_target_loss_ablation_portfolio_20260816_v4`

## Six-hour calibration-cadence ablation

The static frozen MC1_d2 prediction was reused byte-for-byte.  Only the global
residual adjustment was recalculated every six hours from labels resolved
strictly before the block.  The score-band curve remains fixed within each UTC
day and is fitted only from labels resolved before that day opened.  This
isolates a finer dynamic-calibration cadence without altering model semantics.

| Period | Control net bps/trade | Six-hour net bps/trade | Delta | Delta net sum | Other change |
|---|---:|---:|---:|---:|---|
| 2025 | +182.39 | +183.42 | +1.03 | +6,201 bps | 5 fewer trades; worst week +4.35 bps; worst month -0.78 bps |
| 2026 Jan–Jul | +160.51 | +161.82 | +1.31 | +9,430 bps | 28 more trades; same worst month/week to reported precision |

The gain is directionally positive but too small, and 2025 drawdown is mildly
worse (−88.69% vs −84.88% under the intentionally aggressive unit replay).
This is a **forward-shadow challenger**, not a canonical replacement.

Artifacts:

- `data_perp/artifacts/strict_r3_mc1_d2_6h_cadence_strictlt_20260816_v2`
- `data_perp/artifacts/strict_r3_mc1_d2_6h_cadence_portfolio_20260816_v1`

## Admission-threshold recall test

This changes no model, calibration, or auction field: only the frozen MC1
expected-policy-net admission floor is varied in the full strict replay.

| Threshold | 2025 trades | 2025 net bps/trade | 2025 net sum | 2026 trades | 2026 net bps/trade | 2026 net sum |
|---:|---:|---:|---:|---:|---:|---:|
| +30 bps | 7,621 | +161.54 | +1,231,097 | 4,198 | +141.08 | +592,237 |
| +40 bps | 7,272 | +171.20 | +1,244,939 | 4,009 | +148.67 | +596,016 |
| **+50 bps** | **6,902** | **+182.39** | **+1,258,824** | **3,741** | **+160.51** | **+600,480** |

The +50-bps threshold wins total contribution and per-trade EV in both eras.
Its worst-month result is also strongest (+150.77 bps in 2025; +41.86 in
2026), while its drawdown is least severe under the same deliberately
aggressive replay sizing.  Do not lower the threshold to restore recall.

Artifacts:

- `data_perp/artifacts/strict_r3_mc1_d2_threshold30_portfolio_20260816_v1`
- `data_perp/artifacts/strict_r3_mc1_d2_threshold40_portfolio_20260816_v1`
- `data_perp/artifacts/strict_r3_mc1_d2_threshold50_portfolio_20260816_v1`

## Static-model complexity and seed stress

This falsification holds the static six-input contract, strict prequential
monthly refit, causal daily shift, +50-bps admission, and final-score auction
fixed.  It changes only HGB depth, seed/sample, or leaf support.

| Arm | 2025 EV/trade | 2025 net sum | 2026 EV/trade | 2026 net sum | Decision |
|---|---:|---:|---:|---:|---|
| Frozen depth 2, seed 1729, leaf 100 | +182.39 | +1,258,824 | +160.51 | +600,480 | retain |
| Depth 1, seed 1729, leaf 100 | +189.54 | +1,236,530 | +173.42 | +606,287 | reject: loses 2025 contribution and worsens 2026 worst month |
| Depth 2, seed 17, leaf 100 | +174.75 | +1,254,158 | +156.87 | +587,950 | reject |
| Depth 2, seed 2718, leaf 100 | +177.16 | +1,246,675 | +158.04 | +609,710 | reject: lower EV and weaker drawdown |
| Depth 2, seed 1729, leaf 200 | +182.14 | +1,256,230 | +160.04 | +598,382 | reject: no improvement |
| Depth 3, seed 1729, leaf 100 | +177.64 | +1,259,818 | +161.14 | +609,897 | reject: lower EV and substantially weaker worst-month/drawdown behavior |

The model is not dependent on a single fragile depth: depth 1 and depth 3 both
remain economically positive.  But the frozen depth-2/seed-1729 contract is
the best balanced choice under the stated precision, total-contribution, and
downside objectives.

Artifact: `data_perp/artifacts/strict_r3_mc1_d2_complexity_stress_20260816_v2`.

## Candidate-scope and agreement-feature evidence

The frozen source already takes the top 50 ranks per timestamp (roughly the
top 30% of the 170-symbol universe) plus a random background sample for model
training.  In the strict control, every candidate admitted by MC1 at +50 bps
has `conditional_consensus_rank >= 0.7503`; a top-30% consensus scope has a
rank cutoff near 0.70.  Therefore opening that inference pool does not change
the +50-bps admitted set in the historical control.

The prior timestamp-local top-30 feature work retained three agreement fields
on top of the six-field contract in 2025:

```text
agr_rank_iqr
agr_frac_far_10sd
agr_head_mean
```

It improved that 2025 development contract's net sum by +36,681 bps.  However,
its 2024 matched validation lost 3.15 bps/trade, reduced total net by 19,633
bps, and worsened worst-week EV by 48.02 bps.  Its 2026 result improved net
by 6.41 bps/trade and total net by 7,166 bps but worsened worst-month EV by
6.96 bps.  It is informative but not portable enough to replace frozen MC1_d2.

The fine ordinal-bin variants also failed their 2025 total-net gate.  Existing
causal blend tests show no stable case to make a rolling control map an
admission authority alongside MC1; retain it as telemetry rather than blending
it into MC1's live decision.

The existing two-stage regression-plus-LambdaRank experiment is also not
promoted.  Its 2026-only forward split improves selected EV from +151.23 to
+156.85 bps/trade and net sum from +473,952 to +496,424 bps relative to the
same MC1-only top-30 control, but reduces worst-month EV from +22.71 to +18.42
bps and worst-week EV from -13.61 to -14.99 bps.  It has no independently held
post-selection era, so it remains a research observation rather than an
auction replacement.

## Alternate target/loss scores as auction-only signals

The target/loss challengers are poor replacements for MC1 admission: their
within-admission IC is below frozen MC1 in both eras (frozen: 0.179 in 2025,
0.205 in 2026).  They can nevertheless carry a different ordering signal.
This ablation therefore keeps **frozen MC1_d2 expected net >= +50 bps** as the
only admission rule and changes the timestamp-local auction order only.

| Auction order | 2025 EV/trade | 2025 net sum | 2026 EV/trade | 2026 net sum | Result |
|---|---:|---:|---:|---:|---|
| Frozen final score | +182.39 | +1,258,824 | +160.51 | +600,480 | control |
| Huber-arcsine score | +195.49 | +1,332,867 | +161.23 | +608,802 | 2026 worst month falls by 15.76 bps |
| Huber only when >=4 MC1-admitted candidates compete | +195.82 | +1,336,054 | +164.54 | +622,302 | 2026 worst month falls by 14.36 bps |
| Fractional Huber authority at >=4 candidates | no version advances | — | no version advances | — | reject |

The conditional Huber arm is economically interesting: it raises total net in
both eras while admission, target, feature inputs, and policy remain frozen.
However, it was chosen on the same historical development sequence and fails
the declared worst-month guardrail in 2026.  It is therefore a **forward-shadow
auction challenger**, not a replacement for final-score ordering.  L1 and
ordinal auction orders also fail to dominate across both eras.

Artifacts:

- `data_perp/artifacts/strict_r3_mc1_d2_auction_score_ablation_20260816_v2`
- `data_perp/artifacts/strict_r3_mc1_d2_auction_score_ablation_20260816_v3`
- `data_perp/artifacts/strict_r3_mc1_d2_auction_score_ablation_20260816_v4`

## Two-stage Huber + LambdaRank auction

This is the cleanest two-stage test in this batch:

```text
frozen MC1_d2 expected net >= +50 bps
    -> LambdaRank auction among admitted rows only
```

The ranker inputs are frozen final score, prequential Huber-arcsine expected
net, and the five remaining target-free MC1 context fields.  Its target is the
six-bin policy-net grade.  HPO uses April/July/October 2025 chronological
three-month folds; its winning configuration is depth 4, 24 leaves, 915
minimum child rows, learning rate 0.0454, tail gains `[0,1,2,5,10,20]`, and
truncation 5.

| Evaluation | Frozen final-score auction | Two-stage winner | Delta |
|---|---:|---:|---:|
| 2025 Apr–Jun OOF EV/trade | +189.70 | +195.34 | +5.64 |
| 2025 Jul–Sep OOF EV/trade | +177.59 | +184.43 | +6.84 |
| 2025 Oct–Dec OOF EV/trade | +189.00 | +203.29 | +14.28 |
| 2026 portfolio EV/trade | +160.51 | +160.73 | +0.21 |
| 2026 total net bps | +600,480 | +608,031 | +7,551 |
| 2026 worst week | -2.25 | +10.83 | +13.08 |
| 2026 max drawdown | -35.32% | -33.91% | +1.41 pp |

The 2026 worst month falls from +41.86 to +35.33 bps; that remains inside the
predeclared 10-bps guardrail.  However, the next two 2025 HPO candidates fail
to reproduce the 2026 uplift.  The winner is thus a **specific forward-shadow
challenger**, not yet a canonical auction replacement.

The selected configuration itself is seed-robust on the same 2026 replay:

| Seed | 2026 EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---:|---:|---:|---:|---:|---:|
| Frozen final-score control | +160.51 | +600,480 | +41.86 | -2.25 | -35.32% |
| 17 | +161.83 | +612,058 | +35.78 | +3.86 | -34.76% |
| 1729 | +160.73 | +608,031 | +35.33 | +10.83 | -33.91% |
| 2718 | +163.05 | +614,224 | +33.12 | +5.46 | -33.91% |

This supports a real auction-order signal rather than a one-seed result.  It
does not remove the need for an untouched forward period, because the HPO
configuration was selected on the historical development sequence.

Artifact: `data_perp/artifacts/strict_r3_mc1_d2_two_stage_auction_ranker_20260816_v2`.

### Monthly-prequential refit falsification

The selected two-stage ranker was also refit at the beginning of every 2026
month using only labels with `policy_label_available_ts < held-month start`.
This is a deployment-cadence test, not a new HPO: it preserves the exact
already-selected target, seven target-free inputs, 2025 HPO parameters, frozen
MC1 +50-bps admission authority, and portfolio contract.  The control uses
the frozen final-score auction on the *identical MC1-admitted population*.

| Seed | EV/trade | Total net bps | Worst month | Worst week | Max DD | Decision |
|---:|---:|---:|---:|---:|---:|---|
| Frozen final-score control | +160.51 | +600,480 | +41.86 | -2.25 | -35.32% | control |
| 1729 | +162.74 | +612,393 | +39.53 | -2.08 | -33.92% | encouraging, but not robust alone |
| 17 | +162.65 | +615,779 | +35.23 | -3.36 | -33.91% | encouraging, but not robust alone |
| 2718 | +159.33 | +602,576 | +38.12 | -0.79 | -36.24% | fails EV/drawdown improvement |

All folds obey the strict availability cutoff; the persisted fold ledger shows
the latest training label at or before the day before every held-month start.
However, the seed-2718 failure means monthly re-fitting is **not promoted**.
The two-stage idea remains a useful forward-shadow research arm, but MC1_d2
plus frozen final-score ordering remains canonical.

Artifacts:

- `scripts/run_strict_r3_mc1_d2_two_stage_monthly_refit.py`
- `data_perp/artifacts/strict_r3_mc1_d2_two_stage_monthly_refit_20260816_v1`
- `data_perp/artifacts/strict_r3_mc1_d2_two_stage_monthly_refit_20260816_v2`
- `data_perp/artifacts/strict_r3_mc1_d2_two_stage_monthly_refit_20260816_v4`

## Current decision

1. Retain frozen six-field MC1_d2 as the admission authority.
2. Do not promote any altered target/loss, fine ordinal, agreement-feature, or
   control-map blend.
3. Do not promote the monthly-prequential two-stage LambdaRank auction: its
   seed-2718 replay fails the joint EV/drawdown gate.
4. Keep the six-hour residual-shift arm in shadow until it accumulates an
   untouched forward period; do not retune it.
5. Any next feature experiment must use the top-30 target-free candidate
   universe, strict prequential labels, and a matched core-six control.
