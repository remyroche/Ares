# K9 membership-weighted history portability audit

## Question

Can frozen-K9, candidate-specific recent cluster-history features improve the
causal LDF relative sizing overlay?

All tests used one frozen Geometry/K9 bundle.  Cluster history was calculated
per candidate as a soft membership-weighted aggregate of strictly
prior-resolved cluster outcomes.  The activation-sharpened version used:

```text
w(i, k) = membership(i, k)^8 / sum_j membership(i, j)^8
history(i) = sum_k w(i, k) * history(k, decision_time -)
```

No raw K9 cluster slot was supplied to a model.

## Matched 2025 development replay

The matched core-12 LDF and core-12 plus six highest-ranked K9-history fields
used identical causal candidates, admission, policy labels, model parameters,
and ranking.  Only the relative LDF multiplier differed.

| Tail | Core-12 net bps | Core-12 + K9 history-6 net bps | Difference |
|---|---:|---:|---:|
| Top 0.5% | 14.868 | 15.095 | +0.227 |
| Top 1% | 7.021 | 7.189 | +0.168 |
| Top 2% | -2.969 | -2.866 | +0.103 |
| Top 5% | -16.140 | -16.068 | +0.072 |

The sizing gain was small.  Frozen K9 support/OOD/covariance fields were
weaker than the history-6 arm.

## Untouched 2026 confirmation

The same frozen Geometry/K9 definition, power-eight history, and contracts
were replayed causally for April--July 2026 after a January--March training
window and prior-2025 resolved history context.

| Tail | Core-12 net bps | Core-12 + K9 history-6 net bps | Difference |
|---|---:|---:|---:|
| Top 0.5% | 13.646 | 13.651 | +0.005 |
| Top 1% | -15.343 | -15.342 | +0.001 |
| Top 2% | -26.546 | -26.545 | +0.001 |
| Top 5% | -41.674 | -41.675 | -0.001 |

The mean absolute difference between the history-aware and core LDF
multipliers was only 0.000158 in 2026 (95th percentile 0.001961), so the
history model had almost no practical sizing authority in that period.

## Locality-gate ablation

Three train-only gates were tested: hard top-25% membership margin plus
median recent support, hard top-5% margin plus upper-quartile support, and a
soft product of the two.  The thresholds used only the preceding three months
of decision-time feature values.  None improved the ungated history arm in
2025 and all were identical to the core control to displayed precision in
2026.

## Interpretation and decision

- The fields are present, causal, and candidate-specific; they are not a
  global average in their implementation.
- The frozen posterior remains diffuse even after sharpening, limiting local
  differentiation.
- More importantly, conditional association with policy net changes sign by
  month in a diagnostic top-score population.  The problem is lack of
  portable conditional economics, not MDA protection or feature availability.
- Do not add the K9-history, K9 structural, or locality-gate arm to the
  canonical LDF contract.

Further K9 work should first change the representation/target question (for
example a separately validated, high-confidence local downside calibration)
rather than continue tuning this current history multiplier.

## Superseded frozen-temperature representation screen

The first temperature-times-0.25 screen is retained only as a historical
diagnostic.  Its structural quantities were regenerated, but the transient
memberships used to build cluster history were still inherited from the
upstream scorer.  It therefore did **not** test one internally coherent
representation and must not support a promotion or rejection decision.

The materializer has since been repaired: it regenerates the memberships,
structural state, bundle identity, target history, and earlier context history
together before dropping the raw membership slots.  All results below use that
repaired lineage.

## Policy-residual Geometry/K9 challenger

The original frozen geometry target was `H12 TP6/SL4 net > base anchor`, while
the downstream LDF is judged on the trailing-policy net outcome.  A separate,
explicitly noncanonical challenger therefore froze a new October--December
2024 geometry definition with target:

```text
policy_net_bps - prequential_base_anchor_bps > +50 bps
```

It used 135,727 valid, strictly prequential warm-up rows; the encoder retained
the 240k cap, K9 used 100k equal-month sampled rows, and all later evaluation
rows were re-materialized under that one challenger bundle.

On matched 2025 development, its 12 structural fields modestly improved the
global tails over the repaired core: +15.093 / +7.202 / -2.818 / -16.033 bps
at top 0.5 / 1 / 2 / 5%, versus +14.868 / +7.021 / -2.969 / -16.140.
On untouched April--July 2026 it was effectively identical: +13.649 / -15.342
/ -26.546 / -41.677, versus +13.646 / -15.343 / -26.546 / -41.674.

**Decision:** do not promote the policy-residual geometry challenger.  Better
target semantics alone did not create portable sizing information.

## Repaired candidate-specific relative-sizing challenger

The repaired canonical 2025 surface was rematerialized from the exact
target-free source panel, with 737,257 identities, a single regenerated frozen
Geometry/K9 hash, and raw memberships retained only while calculating history.
The research LDF used a 22-field mixed structural contract (base/consensus,
leaf support/OOD, K9 support/OOD/drift, membership-weighted cluster history,
and committee state).  This is an authority test, not an MDA-selected or
canonical feature contract.

For each candidate, its local quality multiplier is compared to its global
quality multiplier within one of five **training-only base-score bins**:

```text
local_i = multiplier(quality_i | train rows in score bin_i)
global_i = multiplier(quality_i | all train rows)
size_i = clip(global_size_i * (1 + alpha * (local_i / global_i - 1)))
```

Thus it changes only size; candidate identities, final-score ranking, and the
causal 21-day EV admission are unchanged.  It is candidate-specific: a trade
with different K9 posterior memberships receives different local history and
may receive a different size even at the same broad score level.

The prior absolute blend was rejected because it saturated essentially every
candidate at the 1.75x cap.  The relative alpha=0.50 arm was selected on 2025:

| Tail | Global LDF | Relative K9 LDF | Difference | Worst month: global → relative |
|---|---:|---:|---:|---:|
| Top 0.5% | 14.943 | 15.357 | +0.414 | -18.612 → -17.801 |
| Top 1% | 7.146 | 7.484 | +0.338 | -6.173 → -5.453 |
| Top 2% | -2.826 | -2.422 | +0.404 | -19.900 → -19.361 |
| Top 5% | -16.048 | -15.750 | +0.298 | -27.512 → -27.242 |

It failed its untouched April--July 2026 confirmation:

| Tail | Global LDF | Relative K9 LDF | Difference |
|---|---:|---:|---:|
| Top 0.5% | 13.644 | 13.638 | -0.006 |
| Top 1% | -15.345 | -15.344 | +0.001 |
| Top 2% | -26.542 | -26.509 | +0.033 |
| Top 5% | -41.664 | -41.597 | +0.067 |

The 2026 5th--95th multiplier range remained narrow in each fold (for example
1.6988--1.7182 in June), so the layer retained too little authority to alter
portfolio economics materially.

**Decision:** reject the relative-sizing challenger.  Candidate-specific
activation weighting is correctly implemented, but the current frozen K9
states do not yet supply portable conditional economics at the admitted tail.

## Neutral all-fields-equal MDA on the repaired surface

The previous selections were not sufficient to answer whether the new fields
were being disadvantaged.  A fresh MDA was therefore run on the repaired
canonical 2025 surface with 107 eligible fields.  There was no protected
baseline: all existing and all newly-derived fields were equal candidates for
conditional permutation and retrained elimination.  It used four chronological
April--July folds, 40,000 equal-month training rows per fold, 12,000 held rows
per permutation, two repeats, then nine group and up to 24 individual
retrained removal checks.

The 12-field compact proposal was entirely made of membership-weighted cluster
history:

```text
3d:  support, directional rate, adverse-200 rate
7d:  support, positive rate, approximately-correct rate, adverse-100 rate
14d: support, mean residual, positive rate, positive-100 rate, adverse-100 rate
```

This means the MDA did *not* suppress the new fields.  They were the most
redundant-tolerant development proposal.  But the full held-population replay
and untouched validation did not support simplification or promotion:

| Period / tail | 12-field K9 compact | Full 107-field equal-status LDF | Better |
|---|---:|---:|---|
| 2025 Top 0.5% | 14.662 | 15.025 | Full |
| 2025 Top 1% | 6.925 | 7.185 | Full |
| 2025 Top 2% | -3.100 | -2.822 | Full |
| 2025 Top 5% | -16.246 | -16.030 | Full |
| 2026 Top 0.5% | 13.637 | 13.641 | Full |
| 2026 Top 1% | -15.348 | -15.346 | Full |
| 2026 Top 2% | -26.558 | -26.548 | Full |
| 2026 Top 5% | -41.696 | -41.673 | Full |

The absolute differences are tiny; this is a non-result, not evidence that
the full contract has strong useful information.  It does, however, rule out
the claim that the K9 additions failed merely because of feature tiers or an
old selection process.

**Decision:** retain the current canonical LDF contract.  Do not promote the
K9-only compact or any isolated K9 structural/history feature.  The next K9
research step must change the *state information itself* (for example a
separate, causally validated downside/dispersion target or a representation
with materially sharper and stable membership), not repeat feature selection
or multiplier tuning.

## Fully consistent membership-sharpening confirmation

The previous `power=8` experiment predated atomic membership/state
regeneration. It was rerun from scratch on the repaired materializer:

```text
w(i,k) = membership(i,k)^8 / sum_j membership(i,j)^8
history(i) = sum_k w(i,k) * history(k, t-)
```

This was a material representation change: 99.78% of candidate history values
changed; the mean absolute change in 3-day membership-weighted residual history
was 9.11 bps (95th percentile 28.06). Raw memberships, structural inputs,
target months and 2025 context months were all rebuilt from the same frozen K9
state before history was calculated.

The same neutral 107-field MDA removed the entire 24-field cluster-path group
and selected 12 causal global-recent-reliability fields instead. Matched and
untouched replays again favored the full equal-status contract:

| Period / tail | Power-8 compact | Power-8 full | Better |
|---|---:|---:|---|
| 2025 Top 0.5% | 14.841 | 14.934 | Full |
| 2025 Top 1% | 7.046 | 7.137 | Full |
| 2025 Top 2% | -3.011 | -2.861 | Full |
| 2025 Top 5% | -16.222 | -16.055 | Full |
| 2026 Top 0.5% | 13.632 | 13.643 | Full |
| 2026 Top 1% | -15.347 | -15.346 | Full |
| 2026 Top 2% | -26.556 | -26.548 | Full |
| 2026 Top 5% | -41.693 | -41.674 | Full |

**Prior representation decision:** do not promote raw-posterior, power-eight, structural,
history, locality-gate, policy-geometry, or relative-sizing variants. The
candidate-specific activation weighting and all-fields-equal feature selection
are now both verified. The remaining failure is economic/representation-level:
within the current frozen K9 state, prior cluster outcomes do not add stable
decision-time information beyond the existing causal global reliability state.

## Score-conditioned cluster-history challenger

The remaining semantic mismatch was that unconditional cluster history lets a
high-score candidate inherit cluster outcomes generated by weak-score rows. A
new causal surface therefore assigned every candidate to a fixed final-score
CDF band (`<.70`, `.70-.85`, `.85-.95`, `>=.95`) and used only strictly
prior-resolved outcomes from that same band before taking the candidate's own
soft K9-membership aggregate. The bands are predeclared constants, not held
cross-sectional ranks or outcome-fitted thresholds.

All 15 new fields were equal MDA candidates. The 122-field neutral MDA selected
15 compact fields, including six score-conditioned features: 3/7/14-day
residual histories and 7/14-day adverse-rate histories. Thus the design is
incremental in development, rather than being excluded by the selector.

It did not survive full-population comparison or untouched validation:

| Period / tail | Score-conditioned compact | Score-conditioned full | Better |
|---|---:|---:|---|
| 2025 Top 0.5% | 14.710 | 14.857 | Full |
| 2025 Top 1% | 6.960 | 7.102 | Full |
| 2025 Top 2% | -3.058 | -2.853 | Full |
| 2025 Top 5% | -16.227 | -16.046 | Full |
| 2026 Top 0.5% | 13.647 | 13.637 | Compact by 0.010 |
| 2026 Top 1% | -15.349 | -15.347 | Full |
| 2026 Top 2% | -26.564 | -26.546 | Full |
| 2026 Top 5% | -41.705 | -41.672 | Full |

The apparent 2026 Top-0.5% edge is economically immaterial and comes with
weaker Top-1 through Top-10 results. It does not clear the portability gate.

**Final K9-state decision:** reject score-conditioned cluster history as well.
The implementation, membership weighting, representation coherence, neutral
selection, and target-conditioned aggregation have now all been tested. More
work on this family should require a genuinely different state source—not a
new transformation of the current K9 memberships or their historical outcomes.

## Why the new fields did not transport

This was tested directly, rather than inferred from the MDA result.  On every
valid policy-labelled row, and separately inside the monthly top-score 20%, we
measured the within-month Spearman association of each K9/leaf/cluster feature
with the realised policy residual:

```text
policy residual = policy_net_bps - prequential base_anchor_bps
```

The development panel is January--July 2025 (613,587 valid rows) and the
untouched panel is January--July 2026 (512,814 valid rows).  This is a
diagnostic outcome analysis only; it does not feed labels into any inference
input or selection decision.

The structural fields plainly vary and some have substantial in-month tail
association.  Examples below are the median monthly correlation in the top
20% of the frozen score:

| Feature | 2025 | 2026 | Reading |
|---|---:|---:|---|
| `k9_ood_distance` | +0.171 | +0.145 | Stable, but is a broad OOD descriptor already represented by correlated structural fields. |
| `k9_cluster_weighted_distance` | +0.168 | +0.105 | Same direction, weaker later. |
| `leaf_ood_marginal` | +0.147 | +0.136 | Stable, but redundant with the other distance/OOD variables. |
| `leaf_support_effective` | -0.164 | -0.127 | Stable association but counter-intuitive and redundant with support transforms. |
| `cluster_recent_3d_mean_residual_bps` | -0.103 | +0.003 | Development effect disappears. |
| `cluster_scorecond_3d_mean_residual_bps` | -0.095 | +0.040 | Sign changes. |
| `cluster_recent_3d_adverse100_rate` | +0.118 | +0.000 | Development effect disappears. |
| `cluster_scorecond_3d_adverse200_rate` | +0.108 | +0.005 | Development effect disappears. |

Across all 71 K9/leaf/cluster candidates, median absolute monthly association
was only 0.024 in the full 2025 population and 0.028 in 2026.  In the
top-score population it was 0.047 in 2025 but only 0.028 in 2026; only 62% of
fields retained their sign.  The important distinction is that *structural
distance/support* has modest, redundant association, while the intended
cluster-history conversion signal is the part that fails to persist.

This also explains the MDA behaviour.  MDA is conditional permutation:
it asks whether a field changes the frozen LDF objective **after** base score,
base/consensus disagreement, global recent correctness, and correlated
structure are present.  A field can have standalone correlation but zero or
negative conditional value because it is redundant, changes sign, or is useful
only in one development month.  The score-conditioned fields were selected in
the development MDA precisely because they were locally incremental there;
their null 2026 replay is the portability veto, not an availability or feature-
tier failure.

**Next direction:** do not discard causal support/OOD signals.  Retain them as
eligible inputs in the canonical all-fields-equal selection process.  But the
next challenger should use an independent state system with its own stable
causal inputs and an explicit cross-era semantic test—for example the existing
market transition/funding/OI state surface—then expose only role-aligned
outputs (support, downside probability, residual expectation, and confidence)
to the LDF.  It must be compared on the same 2025 development and untouched
2026 protocol before it can alter the canonical stack.

## Independent continuous-state challenger

To make the prior result a fair test of the new market-state work, the MDA
surface was extended with 87 independently materialised, target-free,
decision-time fields: 63 causal continuous market-state fields and 24
relationship-break fields.  The contract deliberately excludes fold-local
latent/posterior coordinates, so each input has stable semantics at inference.
Every field was eligible on exactly the same basis as every pre-existing
reliability field; there is no protected or additive feature tier.

The January--July 2025 MDA considered 209 eligible fields.  Its retrained
backward-elimination contract retained 15 fields, all cluster-history fields;
none of the 87 new continuous-state fields was retained.  The result was then
tested unchanged in the supported untouched April--June 2026 population:

| Tail | Full 209-field contract | 15-field compact contract | Difference |
|---|---:|---:|---:|
| Top 0.5% | +16.17 | +16.17 | 0.00 bps/trade |
| Top 1% | +3.82 | +3.82 | 0.00 |
| Top 2% | -21.70 | -21.70 | 0.00 |
| Top 5% | -39.25 | -39.25 | 0.00 |

The equality is substantive: backward elimination excluded every continuous
field, so the two fitted contracts make the same selected-set decisions.  By
month, the shared Top-2 result is +29.33 bps in April, +11.86 in May, and
+66.04 in June; this is not evidence that continuous state adds value.

The validation intentionally stops at June.  The upstream hourly state source
currently ends on 2026-07-11, leaving later July candidates without a valid
state observation.  Those rows were not imputed or silently dropped.  The
continuous-state challenger is therefore **rejected for now**: it has no
incremental 2025 or supported-2026 benefit, and its source needs extension
before any later forward test can be meaningful.

### State-only demotion check

The continuous fields were also tested in the role for which a broad market
state is most naturally suited: a causal, high-score-only conversion-risk
demoter rather than an all-row rank feature.  For each month, a regularised
state-only ridge was fitted on the preceding three resolved months and on the
upstream top 20% or top 30% only.  Its target was winsorised policy-net minus
the train-only monotone score expectation.  Its authority was strictly one
sided:

```text
adjusted expected bps = parent expected bps
                        - alpha * max(-state residual prediction, 0)
```

The first version gave every candidate-row equal weight.  The repaired version
gave every decision timestamp equal total weight, so a large concurrent asset
cross-section could not manufacture repeated evidence about a shared market
state.  Neither version cleared the 2025 development control:

| 2025 global tail | Parent-score control | Best timestamp-equal demoter |
|---|---:|---:|
| Top 1% | +2.00 | -3.19 |
| Top 2% | -3.29 | -11.12 |
| Top 5% | -16.36 | -31.51 |

This rules out the specific explanation that ordinary MDA simply used the
wrong *form* for the fields.  On the present contract, the continuous state
does not identify a sufficiently stable adverse conversion environment even
when granted only a conservative demotion role.  It was not sent to 2026
validation, by predeclared development gating.

## Candidate-specific meta-context and binned-Bayesian follow-up

The previous continuous-state test was mostly market-wide: a value was shared
by every contemporaneous candidate.  A separate target-free exact170 panel
therefore exposed 109 decision-time, candidate-specific meta-owned fields
(asset OI/funding interactions, liquidity residuals, cross-sectional tails,
market breadth and structural-price context).  The join is exact on candidate
identity, timestamp, symbol and side; the source and availability timestamp
are each the decision timestamp.  It includes causal warm-up nulls rather than
dropping candidates.

All fields were equal candidates.  The new 2025 retrained MDA considered 280
eligible fields.  Its 84-field compact challenger contains candidate-specific
meta context only; this is a consequence of the conservative backward
elimination tolerance, not a protected feature tier.  Its separate 30-field
MDA proposal remains a more compact, cross-fold-positive candidate list.

The frozen 2025 selection was evaluated unchanged on April--June 2026 using
the exact same frozen Geometry/K9 bundle (`5ed9…ab5c`) on development and
validation.  Full and compact ranks/economics were effectively identical:

| 2026 global tail | Full 280 | Compact 84 |
|---|---:|---:|
| Top 0.5% | +16.17 | +16.17 |
| Top 1% | +3.82 | +3.82 |
| Top 2% | -21.70 | -21.70 |
| Top 5% | -39.25 | -39.25 |

This is **not** evidence that the 84 fields are worthless.  The test uses a
trust-sizing overlay, and its multipliers are almost fully saturated near the
configured 1.75x cap.  The LDF active-candidate multiplier has standard
deviation only `0.00013`, so neither MDA nor a sizing-only comparison can
reliably distinguish an input that changes relative risk modestly.

To test the requested pre-forest alternative, the historical binned empirical
Bayes `B5_stable_ranklossfp_l125_predictive` was replayed on identical
three-month prequential 2026 folds alongside the current LDF N5 arm:

1. current LDF N5 with the 84-field candidate-context compact;
2. historical B5 on its 43 available legacy fields;
3. B5 on legacy plus all 84 enhanced fields;
4. B5 on legacy plus the 30-field MDA proposal.

The B5 method uses fitting-fold quantile bins, posterior cell shrinkage,
stable train-only CMI interactions and predictive-risk sizing.  All arms keep
the frozen score ranking and causal EV admission; they can only alter size.
The full top-30%-fit multiplier comparison differs by at most 0.02 bps/trade
at Top-0.5/1/2/5 because all arms are similarly cap-saturated.  Calibrating the
multiplier against the full preceding candidate population did not cure this;
the active LDF standard deviation fell further to `0.00003`.

Posterior outputs themselves are not constant.  In the held 2026 candidates,
the LDF adverse-tail probability has Spearman association `-0.056` with
realised policy net (Bayesian variants: about `-0.026` to `-0.037`).  A
strictly causal train-CDF demoter was therefore evaluated without refitting:

```text
risk-only:       final_score - alpha * P_adverse_rank_against_train
mean-minus-risk: final_score + alpha * (E[net]_rank_train - P_adverse_rank_train)
alpha:           0.025, 0.05, 0.10
```

The development selection used April--July 2025 only.  It chose the current
LDF risk-only `alpha=0.05`; its 2025 Top-1/2/5 net was `+17.13 / +4.64 /
-16.79` bps versus the frozen-score control `+6.83 / -3.13 / -16.25`.
The matching untouched 2026 result was `-20.85 / -29.22 / -38.07` versus
control `-22.83 / -28.05 / -39.34`.  It improves Top-1 and Top-5 slightly but
damages Top-2, so it fails the portability gate.

**Decision:** do not promote the new candidate-context compact, the binned
Bayesian overlay, or the posterior demoter.  Preserve all as reproducible
research artifacts.  The next valid improvement is not another feature
transformation: it is a non-saturated, causally calibrated integration of
trust outputs (for example a separately selected EV/risk admission modifier),
selected on development before an untouched era is opened.
