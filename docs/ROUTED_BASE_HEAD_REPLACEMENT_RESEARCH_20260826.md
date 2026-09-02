# Routed base-head replacement research — 2026-08-26

## Status

`B0_policy_ordinal_G3_F72` is the frozen **research challenger** for the B0
role. It is not part of the live inference bundle. The existing E and T
heads remain unchanged.

The required downstream consensus/MC1 replay has now completed and rejected
the HPO/grid blend at both predeclared admission thresholds.  The canonical
decision record is
[STRICT_R3_THREEWAY_HEAD_SELECTION_DECISION_20260826.md](STRICT_R3_THREEWAY_HEAD_SELECTION_DECISION_20260826.md).
This document remains the detailed upstream research receipt; it must not be
read as a promotion of F72/HPO into the active stack.

The reproducible research contract is
[strict_r3_routed_b0_policy_ordinal_g3_f72_hpo_research_20260826_v1.json](../config/strict_r3_routed_b0_policy_ordinal_g3_f72_hpo_research_20260826_v1.json).

## Upstream contract

All results use the frozen strict-OOF router receipt and rank only the
timestamp-local top 50% routed long candidates. Train labels resolve before a
28-day reserve that precedes each held month. The three blocked development
folds are February, March, and April 2026; they are development evidence, not
an untouched promotion period.

No outcome, path, or label-validity field is part of the B0 feature matrix.
Invalid policy rows are excluded from fitting and evaluation rather than
encoded as losses.

## E / T result

The full-universe selection process was completed independently for E and T:
1,407 numeric causal fields → 1,199 hygiene-valid → 1,094 after the 0.995
near-duplicate veto → Screen120 → OOF MDA / subset ladder. Neither compact
challenger beat the frozen existing head on the same February–April contract.

| Head | Existing frozen control, Top-10 EV | Best compact challenger, Top-10 EV | Decision |
|---|---:|---:|---|
| E | +80.38 bps | +5.90 bps | retain existing E |
| T | +48.49 bps | −51.53 bps | retain existing T |

The full receipts are in
[ROUTED_E_T_FULL_UNIVERSE_FEATURE_SELECTION_20260826.md](ROUTED_E_T_FULL_UNIVERSE_FEATURE_SELECTION_20260826.md).

### Strengthened E/T conditional selection — rejected

A second E/T feature-selection pass used the complete B0-style procedure:
12 randomized screen models per fold, two-seed within-timestamp OOF MDA,
Top-10-boundary substitution diagnostics, semantic-family rescue, and a
120/90/70/50/35/25 subset ladder. Unlike the first pass, this selection was
measured through the actual equal timestamp-rank **incumbent** B0+E+T blend:
the candidate head replaced only E or only T while the incumbent B0 and the
other frozen target-free head remained fixed. The new B0 challenger is not an
input to this E/T-only experiment. This is the relevant non-redundancy test.

The control and challengers use the identical routed, policy-valid,
time-balanced February--April 2026 OOF population. The control is the frozen
three-way score on that exact population, not a pooled-tail metric.

| Arm | Top-1 EV | Top-5 EV | Top-10 EV | Stable Top-10 | Decision |
|---|---:|---:|---:|---:|---|
| Incumbent B0+E+T control | +89.77 | +76.84 | +57.16 | +51.23 | retain |
| Replace E, best F25 | +64.25 | +54.11 | +40.04 | +35.15 | reject |
| Replace T, best F120 | +37.88 | +36.01 | +28.62 | +23.31 | reject |

The enhanced selection method therefore did its intended job: it ruled out
feature contracts that may look useful to a direct supportive target but are
not complementary enough to improve the live-relevant three-head ranking.
Existing E and T remain frozen. No E/T HPO, downstream consensus, MC1, or
live-artifact work follows from this rejected result.

Primary receipts are
`strict_r3_routed_et_fulluniverse_screen_improved_20260826_v1_{e,t}`,
`strict_r3_routed_et_conditional_mda_20260826_v1_{e,t}`, and
`strict_r3_routed_et_conditional_subset_ladder_20260826_v1_{e,t}`. The
conditional tools are
[run_strict_r3_routed_et_conditional_mda.py](../scripts/run_strict_r3_routed_et_conditional_mda.py)
and
[evaluate_strict_r3_routed_et_conditional_subset_ladder.py](../scripts/evaluate_strict_r3_routed_et_conditional_subset_ladder.py).

## B0 challenger

### Target and model

The candidate uses canonical rich-policy net bps ordinalised as:

| Net bps | Grade |
|---|---:|
| ≤0 | 0 |
| 0–50 | 1 |
| 50–100 | 2 |
| 100–200 | 3 |
| 200–400 | 4 |
| >400 | 5 |

It uses LambdaRank, G3 clipped-economic gains `[0, 0.5, 2, 3, 6, 8]`,
decision-timestamp × long-side queries, 2,000-tree ceiling and 30-round early
stopping. The frozen HPO winner is depth 4, 18 leaves, learning rate 0.09717,
minimum leaf fraction 2.295%, feature fraction 0.8251, bagging fraction
0.7964, L1 0.00065, L2 0.2212, min gain 0.00201, truncation 10, sigmoid
1.4293. Exact fields and parameters are in the research configuration above.

### Feature selection

| Stage | Result |
|---|---:|
| Causal numeric universe | 1,407 |
| Coverage/variance hygiene | 1,199 |
| 0.995 near-duplicate veto | 1,094 |
| Full-screen contract | 120 |
| OOF MDA subset winner | 70 |
| Structure/location add-back | **72** |

The 72 fields are frozen in
[selection.json](../data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json).
They were selected using OOF gain, general and p70–100 TreeSHAP, univariate
timestamp-local economics, randomized stability, two-seed individual / Top-10
boundary / semantic-family MDA, then the 120/90/70/50/35/25 ladder.

The F72 add-back is real: blend Top-10 rises from +90.82 bps for F70 to
+91.48 bps, and its stable blend score rises from +84.50 to +85.20. Targeted
drop confirmation shows why the structural fields remain: removing
`mark_perp_dislocation` costs −6.29 bps at blend Top-10; removing the full
structure/location family costs −4.99 bps. Two apparent single-feature
removals did not survive their combined-removal confirmation, so F72 remains
intact.

### Funding result

The B0 Screen120 initially included eight funding/carry fields. Funding is
not a core explanation of the challenger: its MDA family evidence was
negative/unstable. A direct F72 funding-family drop nevertheless lowers blend
Top-10 by 0.08 bps and the stable score by 0.15, so the five retained funding
fields stay as weak complementary context, not as a promoted standalone
signal.

## Same-candidate B0 and three-head comparison

This is the clean February–April comparison on identical, routed, valid OOF
candidates. Scores are timestamp-ranks; the three-head arms are equal-rank
blends.

| Arm | Top-1 EV | Top-2 EV | Top-5 EV | Top-10 EV | Stable Top-10 | q10 week | Top-10 >50 precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current B0 | +100.96 | +80.14 | +65.03 | +58.99 | +54.10 | +26.09 | 49.10% |
| New B0 F72/HPO | +284.39 | +219.10 | +151.60 | +93.64 | +82.39 | +40.43 | 58.34% |
| Current equal B0+E+T | +153.86 | +135.75 | +100.58 | +81.25 | +75.92 | +43.98 | 51.72% |
| New equal B0+E+T | **+225.67** | **+194.46** | **+135.69** | **+92.95** | **+86.57** | **+51.00** | **56.08%** |

The new equal blend’s deltas versus the current equal blend are +71.81,
+58.71, +35.12 and +11.70 bps at Top-1/2/5/10 respectively; stable Top-10
improves +10.65 bps. This is base-layer development evidence only. It does
not establish downstream consensus, MC1 admission, portfolio or live uplift.

The equivalent fixed-*k* timestamp-local comparison (select exactly *k*
candidates at each timestamp, then equal-weight timestamps) is:

| *k* candidates / timestamp | Incumbent B0+E+T | New B0+E+T | Delta |
|---:|---:|---:|---:|
| 1 | +153.89 | +225.57 | +71.68 |
| 2 | +132.80 | +194.48 | +61.68 |
| 3 | +124.47 | +171.14 | +46.67 |
| 5 | +102.74 | +133.32 | +30.58 |
| 10 | +73.54 | +83.45 | +9.91 |

### Timestamp-local interpretation

Every Top-*x* figure in this document is calculated **inside each decision
timestamp**, then averaged across the three held months. It is not a pooled
global-tail statistic. For the new B0 alone, timestamp-local Top-1/2/5/10 EV
is +284.39 / +219.10 / +151.60 / +93.64 bps. In the equal E+T+B0 blend it is
+225.67 / +194.46 / +135.69 / +92.95 bps. The research configuration hash is
`fb36aa86a082828f7fe511ae2cffb1d1a25a9d544820852ff20d922cdd02209f`.

## Receipts and scripts

| Stage | Primary receipt / script |
|---|---|
| Candidate labels | [materialize_strict_r3_b0_replacement_targets.py](../scripts/materialize_strict_r3_b0_replacement_targets.py) |
| Target / gain / objective funnel | [run_strict_r3_b0_replacement_ranker_screen.py](../scripts/run_strict_r3_b0_replacement_ranker_screen.py) |
| Full-universe B0 screen | [run_strict_r3_b0_fulluniverse_screen.py](../scripts/run_strict_r3_b0_fulluniverse_screen.py) |
| Individual / boundary / family MDA | [run_strict_r3_b0_fulluniverse_mda.py](../scripts/run_strict_r3_b0_fulluniverse_mda.py) |
| Compact subset ladder | [evaluate_strict_r3_b0_subset_ladder.py](../scripts/evaluate_strict_r3_b0_subset_ladder.py) |
| Add-back and drop confirmation | [evaluate_strict_r3_b0_family_addback.py](../scripts/evaluate_strict_r3_b0_family_addback.py), [evaluate_strict_r3_b0_drop_confirmation.py](../scripts/evaluate_strict_r3_b0_drop_confirmation.py) |
| Compact HPO / fixed OOF rescore | [run_strict_r3_b0_compact_hpo.py](../scripts/run_strict_r3_b0_compact_hpo.py) |

## Downstream decision

The required strict-prequential consensus/MC1 replay on the matched routed
population completed after this upstream receipt.  Although the B40/E55/T05
blend was better upstream, it produced weaker constrained portfolio economics
than the matched incumbent at both the 30- and 50-bps dual-MC1 thresholds.
The current frozen B/E/T upstream heads therefore remain the retained
contract.  Exact metrics, source-population caveats and reproducibility paths
are in
[STRICT_R3_THREEWAY_HEAD_SELECTION_DECISION_20260826.md](STRICT_R3_THREEWAY_HEAD_SELECTION_DECISION_20260826.md).
