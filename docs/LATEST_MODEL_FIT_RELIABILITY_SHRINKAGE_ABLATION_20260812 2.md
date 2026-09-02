# Latest-model-fit reliability and shrinkage ablation — 2026-08-12

> **Longer-period correction:** the original two-period N5 conclusion below is
> superseded by `N5_LONG_PERIOD_CELL_DAY_AUDIT_20260812.md`.  The original
> evaluator took each arm's tail as a percentage of its own admitted set, so a
> demoting challenger could be compared with fewer trades than the control.
> It also used the reserve-seeded map instead of the canonical Cell-day trim
> 15% map.  Under fixed-cardinality diagnostics and the canonical constrained
> portfolio, N5 has a modest positive aggregate effect but fails portability:
> 2025 portfolio EV rises only +4.99 bps/trade while max drawdown worsens, and
> June 2026 changes 69 trades at -26.40 bps/trade into 19 trades at -102.69.
> N5 therefore remains a research challenger and must not replace the
> canonical Cell-day admission/portfolio path.

## Decision

The current-fit reliability feature contract advances as a causal research
input. It does **not** replace frozen K9 geometry. K9 remains the stable
archetype coordinate system; the new fields answer a different question:
whether the feature paths activated by the latest conversion-model fit are
well supported and close to that fit's own pre-cutoff training distribution.

The leading use is `N5_ldf_support_l110_meanrisk` as a **25% demotion-only
common-bps overlay** on the canonical EV admission. It is the only tested arm
whose worst held-period uplift is positive at top 1%, 2%, and 5% across March
2025 and July 2026. It is not promoted to the executable canonical stack yet:
the confirmation covers only two informative held months, July admission is
thin, and top-10 uplift is negative.

Full posterior replacement, broad posterior ranking, and the empirical-Bayes
arms do not advance.

## Architecture and decay repair

Each four-week conversion bundle now fits a 64-tree binary R3-clear reliability
proxy using only labels resolved before its calibration-reserve cutoff. Raw
leaf IDs are not exposed. The bundle persists the feature path of every leaf,
leaf support, leaf contribution, and the training-distribution moments of the
contribution-weighted active-path signature. Held and prior-reference rows are
then transformed by the same bundle.

This produces ten stable semantic fields:

1. `active_rule_candidate_mahalanobis_train`
2. `active_rule_feature_entropy`
3. `active_rule_support_contribution_weighted`
4. `active_rule_ood_contribution_weighted`
5. `active_rule_timestamp_cov_break_train`
6. `active_rule_timestamp_corr_break_train`
7. `active_rule_timestamp_mahalanobis_train`
8. `active_rule_timestamp_support_weighted`
9. `active_rule_timestamp_support_p05`
10. `active_rule_timestamp_ood_weighted`

Candidate fields describe the current candidate's active paths. Timestamp
fields compare the contemporaneous cross-section's weighted active-path state
with the current model's own training baseline. Support and OOD are weighted by
absolute leaf contribution. Covariance, correlation, and Mahalanobis breaks are
computed against moments frozen with the latest model fit—not against the
October–December 2024 K9 definition.

The conversion manifest now records the fit cutoff, training rows, ordered
input fields, field hash, and semantic contract under
`latest_fit_rule_reliability`. The latest-fit fields change model vintage at
the same periodic boundary as the conversion bundle; their names and meanings
do not change. Raw leaves remain prohibited downstream.

## Causality and coverage

- Latest-fit proxy training requires `decision_ts < cutoff` and
  `r3_label_available_ts < cutoff`.
- The 42-day calibration reserve is excluded from supervised conversion fits.
- Held rows and reference rows use the same fitted bundle.
- No held-window percentiles or outcomes enter the fields.
- Frozen K9 is not refit: its SHA was
  `dbf7de6da6bad6927bcbe577d7ad2d2118ecc24a6bdfd35fb7fa190be13d7638`
  for all five 2026 blocks.
- The April–July 2026 replay contains 491,286 rows. Every new field has 100%
  coverage. Candidate fields have 247,202–263,725 distinct values; timestamp
  fields have 2,836–2,928 distinct values.
- The full latest-fit trust contract has 66 eligible fields. The matched legacy
  control removes only these ten fields and has 56.
- The core correctness suite passes: 22/22 tests.

The completed replay started before the new manifest stanza was added, so its
top-level manifest does not contain that stanza. Its runtime fields and bundle
hash lineage are valid. All newly produced bundles include the stanza.

## Models tested

All trust models use the identical chronological population, timestamp-top-30%
training domain, equal-month sampling, 60,000-row cap, policy-net target, and
train-derived transforms/interactions.

| Arm | Method | Main distinction |
|---|---|---|
| B1 | Empirical Bayes | Singleton bins, lambda cap 1.00, posterior mean |
| B3 | Empirical Bayes | Stable CMI interactions, rank/loss weights, lambda cap 1.10, mean-risk |
| B5 | Empirical Bayes | Stable CMI, rank/loss/false-positive weights, lambda cap 1.25, predictive risk |
| N4 | Local Distribution Forest Proxy (LDF) | Raw local forest mean, lambda cap 1.25 |
| N5 | LDF | Local-support shrinkage to parent, rank/loss weights, lambda cap 1.10, mean-risk |
| N6 | LDF | Parent/predictive shrinkage, rank/false-positive weights, lambda cap 1.25 |

LDF uses 64 random-forest trees, depth 8, minimum leaf size 120, 70% features,
75% bootstrap samples, and stable-CMI interaction features. Local effective
support is the median training-leaf support across trees; the local estimate is
shrunk toward the causal parent with `support / (support + 300)`.

The empirical-Bayes arms use 16-bin train-only marginals and stable-CMI pairs.
Singleton effects shrink to lambda 1 with prior strength 200; pair effects use
prior strength 500. Risk cells use prior strengths 300/600. Outputs are
Student-t predictive mean, uncertainty, adverse-tail probability, and support.

## Integration methods tested

1. Direct posterior admission: admit when posterior expected policy net is at
   least +50 bps.
2. Frozen-admission posterior reorder: keep the canonical causal admission set
   and rank it by the trust posterior.
3. Symmetric common-bps blend at 10%, 25%, and 50%.
4. Demotion-only common-bps blend at 10%, 25%, and 50%:

   `corrected_ev = canonical_ev + alpha * min(posterior_ev - canonical_ev, 0)`

Outcomes are consulted only after each target-free selection is frozen. Equal
20-bin EV-map values are broken by the target-free canonical final score, then
candidate ID.

## Matched held-period results

### N5, 25% demotion-only

All numbers are policy-net bps/trade and include the canonical 100-bps cost.

| Held period | Metric | Canonical | N5 latest-fit | Uplift |
|---|---:|---:|---:|---:|
| 2025-03 | Top 0.5% | +192.61 | +166.71 | -25.90 |
| 2025-03 | Top 1% | +346.40 | +364.24 | +17.84 |
| 2025-03 | Top 2% | +465.42 | +550.36 | +84.94 |
| 2025-03 | Top 5% | +525.64 | +548.94 | +23.30 |
| 2025-03 | Top 10% | +414.67 | +402.19 | -12.48 |
| 2025-03 | All admitted | +136.10 | +136.10 | +0.00 |
| 2026-07 | Top 0.5% | +361.50 | +1,071.70 | +710.20 |
| 2026-07 | Top 1% | +182.04 | +348.16 | +166.12 |
| 2026-07 | Top 2% | +141.95 | +284.52 | +142.57 |
| 2026-07 | Top 5% | -143.37 | -6.76 | +136.61 |
| 2026-07 | Top 10% | +16.98 | -82.68 | -99.66 |
| 2026-07 | All admitted | +11.48 | +69.61 | +58.13 |

July has only 568 canonical admitted candidates. Its top 0.5/1/2/5% controls
contain 3/6/12/29 trades, so the very largest bps changes have wide sampling
uncertainty.

### Cross-period demotion uplift by model

| Arm | Mean Top 1% | Worst Top 1% | Mean Top 2% | Worst Top 2% | Mean Top 5% | Worst Top 5% |
|---|---:|---:|---:|---:|---:|---:|
| B1 | -36.51 | -104.10 | -40.43 | -69.88 | +60.94 | -9.54 |
| B3 | -40.96 | -104.10 | -5.88 | -30.54 | +83.74 | -4.02 |
| B5 | -46.93 | -104.10 | -2.69 | -30.54 | +86.85 | +2.21 |
| N4 | -77.11 | -199.70 | +18.84 | -49.30 | -105.13 | -199.78 |
| **N5** | **+91.98** | **+17.84** | **+113.76** | **+84.94** | **+79.96** | **+23.30** |
| N6 | +140.95 | +10.63 | +43.17 | +39.63 | +44.24 | -8.54 |

N5 is the only arm with positive worst-period uplift at all three decision
tails. N6 has higher mean Top-1 uplift but fails the Top-5 worst-period gate.

### Weekly stability of N5 25% demotion

Across the ten March/July weeks containing admitted trades:

| Tail | Positive uplift weeks | Non-negative uplift weeks | Positive-N5 weeks | Median uplift | Worst uplift |
|---|---:|---:|---:|---:|---:|
| Top 1% | 5/10 | 7/10 | 8/10 | +14.14 | -75.82 |
| Top 2% | 5/10 | 7/10 | 8/10 | +46.09 | -125.99 |
| Top 5% | 8/10 | 9/10 | 9/10 | +108.42 | -12.45 |
| Top 10% | 6/10 | 8/10 | 9/10 | +26.23 | -33.41 |

The Top-5 behavior is the most stable. Weekly Top-1/2 samples are too small for
a promotion claim.

## Feature-helpfulness result

The new fields are useful but not uniformly portable:

- On March training, train-only residual MI selected only
  `active_rule_timestamp_cov_break_train` among its top 24.
- On April–June training, seven latest-fit fields occupied the first seven MI
  positions; current-fit OOD, support, entropy, and timestamp correlation break
  were dominant.
- Full 66-field N5 produced the most balanced top-1/2/5 July uplift. Compact
  12/24-field variants improved selected tails but did not dominate across all
  three tails or transport the same feature contract from March.
- On the broad timestamp-top-30% diagnostic, replacing the canonical score by
  the N5 posterior reduced March Top-0.5/1/2/5/10 results and reduced July
  Top-1/2/5. The posterior is therefore a trust correction, not an alpha
  replacement.
- Compared with the 56-field N5 control, the ten fields materially improved
  some March/July tails but not every tail. Their incremental value is real but
  regime-dependent.

June 2026 is not informative: the canonical EV map admitted only ten candidates
for the month. It is recorded as an admission-support failure, not a trust-model
failure.

## Recommendation

Keep the ten latest-fit fields available to the trust layer and preserve their
bundle-vintage lineage. Keep N5 25% demotion as a named research challenger.
Do not make it canonical until a later untouched period supplies at least 500
admitted trades and confirms positive worst-period Top-1/2/5 uplift.

Next improvements should preserve stable semantics:

1. Compute the same semantic aggregates for the actual strict-R3 base and each
   residual/consensus head, then aggregate by stable model role. Never expose
   raw leaf IDs.
2. Add model-fit age and deltas between the latest-fit reliability state and the
   preceding model vintage. The level and the rate of deterioration answer
   different questions.
3. Normalize support by training rows, tree count, and expected leaf occupancy
   so support remains comparable when sample caps or tree geometry change.
4. Maintain two causal baselines: the model's full pre-cutoff fit distribution
   and a short prior-resolved reference. Their divergence separates structural
   OOD from temporary market drift.
5. Select trust features using a portability objective across several held
   vintages, not fold-local MI alone. Penalize worst-period and feature-selection
   instability.
6. Keep demotion-only authority bounded. Do not let this layer promote a trade
   rejected by the canonical alpha/EV stack until broader evidence supports an
   admission role.

## Relevant code and artifacts

- `extreme_price_movements/strict_r3_canonical_current.py`
- `extreme_price_movements/trust_sizing_ablation.py`
- `scripts/run_strict_r3_current_exact_b5_fold.py`
- `scripts/evaluate_strict_r3_trust_posterior_admission.py`
- `tests/test_strict_r3_canonical_current.py`
- `data_perp/artifacts/strict_r3_lockstep_latestfit_reliability_long_2025_q1_20260812_v1`
- `data_perp/artifacts/strict_r3_lockstep_latestfit_reliability_long_2026_aprjul_20260812_v1`
- `data_perp/artifacts/strict_r3_latestfit_trust_N5_all_posterior_blends_tiebreak_long_2025mar_20260812_v3`
- `data_perp/artifacts/strict_r3_latestfit_trust_N5_all_posterior_blends_tiebreak_long_2026jul_20260812_v1`
