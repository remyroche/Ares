# Regime and Transition Workstream Handover

Updated: 2026-07-30

This is the authoritative handover for regime and transition research.  From
this date onward, regime/transition progress belongs here rather than in the
full-pipeline handover.

## Objective

Find low-performance periods, construct robust and economically interpretable
regime and transition representations, emit causal soft OOF outputs with
uncertainty and stable identity, integrate them into base/residual models and
a regime-aware GAM calibrator, then assess IC, EV and weekly/monthly tail
stability.

The final model boundary is strict:

- fit, feature-select, tune, calibrate and freeze on 2022–2025 only;
- assess once on untouched 2026;
- keep current regime and transition state as separate layers;
- keep ex-post transition phase and post-entry path geometry out of causal
  model inputs; use them for labels, qualification and action-layer analysis;
- select one pooled global top-k after each arm's causal EV mapping, never per
  timestamp, side, state or transition category.

The training, validation/OOF, HPO, feature-selection, mapping and assessment
sample cadence is strictly `1h`.  Multi-timeframe values, including native
`15m` where available, are causal lookbacks sampled onto that one hourly
decision row; they never create sub-hourly model or assessment examples.
Exact `1m` bars are allowed only as nested observations inside each hourly
candidate's 12-hour barrier/path/fill/policy replay.  Each replay must return
one outcome record to its originating hourly candidate and must never expand
the statistical training or assessment sample.

This is an enforced cadence contract, not only a reporting convention.  A
valid artifact must declare `model_sample_cadence=1h` and
`assessment_sample_cadence=1h`, reject non-hourly decision timestamps, and
retain a one-to-one link between each hourly candidate and its nested replay
outcome.  A replay may use every available `1m` bar for path ordering, fills,
barriers and exits, but it may not turn those minute observations into
training, OOF, calibration, mapping, HPO or assessment samples.

## Data and label status

The full-2024 exact-one-minute request is frozen at
`failure_2024_transition_exact1m_request_stage_20260730_v2`: 190,398
candidates, 182,718 distinct download requests and 141 symbols, with
decision-to-12-hour half-open paths.

On 2026-07-30, four independent read-only verification partitions covered
all 43,660,200 required minutes:

| Partition | Symbols | Required | Covered | Incomplete | Failed |
|---|---:|---:|---:|---:|---:|
| 0 | 36 | 10,697,460 | 10,697,460 | 0 | 0 |
| 1 | 40 | 13,156,080 | 13,156,080 | 0 | 0 |
| 2 | 33 | 10,814,520 | 10,814,520 | 0 | 0 |
| 3 | 32 | 8,992,140 | 8,992,140 | 0 | 0 |

The vanished downloader processes are not used as proof; these exact
verification results are.  The four manifests are sealed in
`failure_2024_exact1m_download_verify_20260730_v1`.  Fresh immutable-lineage
artifacts now contain 190,398 hourly label-input rows, 190,398 candidate-local
policy replays and 190,398 timing candidates with complete nested 720-minute
physical paths.  `failure_2024_exact1m_candidate_coverage_20260730_v2` binds
the candidate population to both the full-universe verifier and the
four-partition aggregate seal, with zero incomplete candidates.  The joined
`failure_2024_exact1m_multitask_labels_20260730_v2` artifact now contains
190,398 hourly rows, the direct 12-hour policy-EV target, soft triple-barrier
labels, physical-path targets and the expanded supportive label family.  The
exact old-versus-new parity audit is the remaining replay gate before
calendar regeneration.

That final gate now passes in
`failure_2024_exact1m_replay_v1_v2_parity_20260730_v1`: all 190,398
candidate identities and every candidate/context/target/policy/timing/path/
physical/multitask record match the prior replay; decoded 720-minute
timestamp/OHLC paths match at zero tolerance and every raw artifact hash is
equal.  The four-partition aggregate seal is also bound.  Stage A is complete
and the evidence-plane regeneration may proceed.

The cadence separation is sealed at
`regime_transition_hourly_cadence_audit_20260730_v2`.  It verifies 33,907
hourly multiview rows, 4,627 hourly 2026 regime rows, 4,627 hourly 2026
transition timestamps and 190,398 hourly 2024 candidates.  It also covers the
current final-v3 OOF/forward ledgers, mapping suite and GAM mixture, with zero
non-hourly timestamps or invalid future-label endpoints.  The one-minute path
store remains nested one-row-per-hourly-candidate label evidence.

The enforcement was rechecked after the trajectory, H2 and model-failure
extensions on 2026-07-30.  Their contracts continue to declare `1h` decision,
model and assessment rows and `1m_labels_only` replay cadence.  The focused
cadence/trajectory enforcement suite passed 5/5 tests.  Future artifacts fail
closed unless they emit the same cadence/provenance checks.

## Existing discovery methods

### Multi-timeframe regime features

`regime_multiview.py` generates causal transforms at 1h, 3h, 6h, 12h, 24h,
48h, 72h and 168h, plus 15m where native cadence supports it.
`regime_multiview_panel_2022_2026_20260730_v2` contains 33,907 hourly rows and
14,536 fields.  It covers:

- feature distributions and distribution dynamics;
- correlations, covariances, dispersion and dependence changes;
- realized volatility, range and volatility-of-volatility;
- liquidity proxies including spread, depth, volume, Amihud and
  cross-sectional liquidity;
- leverage, OI, funding, breadth, dominance and deleveraging indicators.

Known missing or incomplete families are multi-level impact/depth imbalance,
cross-venue funding/OI/liquidations, correlation-network topology,
conditional trend, pre-entry fragility, model-health drift and
options/basis/skew.  Eighty-three asset-field liquidity combinations are
unavailable and must be represented by explicit availability masks, not
imputation disguised as observation.

### Regime discovery

- Strict diagonal GMM:
  `strict_forward_regime_only_2022aug_2025_to_2026_20260730_v3`.
  It uses train-only family balance, winsorisation, robust scaling, BIC and
  persistence selection.  The split is valid, but the architecture is
  rejected: six states, near-zero entropy, median dwell two hours and
  28–35% monthly hourly switching.
- Strict sticky/full-covariance GMM challenger:
  `strict_forward_sticky_fullcov_regime_challenger_2022aug_2025_to_2026_20260730_v1`.
  It uses the identical 1-hour panel and 2022-08-30→2025 training /
  untouched-2026 boundary, but chooses a family-balanced 32-feature contract,
  full covariance, `k=3..6` and the sticky prior exclusively on a final
  blocked training segment.  It freezes model-local identity, posterior,
  entropy, margin and train-99th-percentile density OOD before 2026.  It is
  also rejected: the best blocked configuration is 3 states / sticky 50;
  2026 predictive score improves from -1.058 (diagonal) to -0.785, but
  filtered median dwell remains 2h and switching is 29.86%/hour (diagonal
  32.35%).  Neither passes the pre-registered >=6h dwell and <=10% switching
  gate.  This is a useful likelihood improvement, not a usable persistent
  regime representation.  Regime remains separate from transition throughout.
- Strict DAE→GMM challenger:
  `strict_forward_dae_gmm_regime_challenger_2022aug_2025_to_2026_20260730_v1`.
  The frozen neural denoising-AE representation, its bottleneck sweep (4/8/12;
  noise 0.05), full-covariance GMM geometry, sticky prior, density/reconstruction
  OOD thresholds and state semantics are selected on 2022–2025 only; 2026 is
  untouched.  The chosen blocked configuration is bottleneck 4, three GMM
  states and sticky 200.  It freezes both density and reconstruction OOD,
  posterior/entropy/margin, model-local identity and exact side/economic
  attribution.  It is rejected: 2026 log score -0.828 is better than diagonal
  (-1.058) but worse than sticky/full-covariance (-0.785); dwell remains 2h,
  with 32.13% hourly switching.  Its 1.75% mean OOD fraction and 0.0061 mean
  entropy do not solve state instability.  Exact attribution is diagnostic,
  not a top-k policy replay: long/short mean net EV is -26.61/-18.58 bps per
  candidate, versus -26.37/-17.18 bps for sticky/full-covariance.  Therefore
  no GMM/DAE arm is a promotable persistent regime representation.
- `unsupervised_economic_all_era_strict_20260730_v1` closes the available-arm
  economic validation gap without pretending all stored sidecars are
  comparable.  It fits the frozen representations on 2022-08→2025 hourly
  data, fits/maps the economic conversion only on 449,814 pre-2026 candidate
  rows (377,928 chronological OOF rows per arm), then assesses one identical
  114,096-row May–July 2026 candidate set.  Candidate rows are 1-hour; 1-minute
  bars remain nested only inside the 12-hour execution label.  The book is one
  pooled global top-10 after each arm's frozen historical OOF EV map, with
  week/month tables decomposing that fixed membership rather than re-ranking.
  Baseline is negative at -74.97 bps net EV; sticky/full-covariance geometry
  is -102.60 bps and DAE→GMM geometry -99.22 bps.  Neither is promotable.
  The diagonal arm is explicitly fail-closed because its sealed artifact lacks
  an immutable transform to replay the historical hourly panel; failure-first
  is fail-closed because its historical OOF overlay is only November 2025 and
  has no semantically identical joint residual-EV/economics cohort.  These are
  availability gaps, not negative model results or substitutes for either arm.
- BOCPD has a genuinely online changepoint implementation.  The strict
  24h/48h checkpointed benchmark is now sealed in
  `strict_bocpd_regime_transition_challenger_20260730_v2`, including causal
  posterior change probability, run-length mean/q05 and normalized entropy.
  Its logistic heads emit convergence warnings and have poor 2026 calibration
  (for example onset-1h all-2026 Brier 0.316 and ECE10 0.536); they are
  diagnostic-only and must never be promoted into a gate, quota or standalone
  trading score.
- KMeans and AE/GMM have limited-scope materialized runs.
- HDBSCAN and categorical HMM are implementation/dependency-only and cannot
  be counted as completed experiments.

### Transition discovery

- Strict LGBM onset heads:
  `strict_transition_v3_multihorizon_competing_risk_20260730_v2`.
  The 1h and 3h heads transfer modestly to 2026 (AUC 0.618 and 0.687);
  6h/12h do not (0.468/0.498).  The lifecycle head is unusable
  (macro-F1 0.093), and high-risk economic slices remain negative.
- Bayesian Rule List:
  the native Beta-Binomial MAP BRL exists as an interpretable OOF challenger,
  not MCMC BRL.  Historical AUC/AP/Brier is 0.600/0.571/0.265 versus
  LGBM 0.874/0.871/0.149.  The strict frozen successor is now
  `strict_transition_brl_challenger_20260730_v1`: the identical causal hourly
  panel, all 2022--2025 feature/rule-list/weight/calibration selection and a
  single 4,627-hour untouched-2026 assessment.  It evaluates stable-versus-
  transition plus 1/3/6/12-hour onset.  It is explicitly
  `native_beta_binomial_map`, rather than MCMC BRL, and seals human-readable
  rules, monthly calibration/discrimination, and pooled-global-top-10
  attribution split by long/short side.  It fails as a challenger: 2026 AUC
  is 0.519 (active), 0.513 (1h), 0.500 (3h), 0.495 (6h), and 0.491 (12h);
  all remain diagnostic-only.  Its one apparently non-negative June high-risk
  slice is too small/isolated and does not override the negative month/side
  attribution or the aggregate discrimination failure.
- Transition catalogue:
  phases include precondition, approach, acceleration, trigger,
  active-dislocation, confirmation, settled, failed transition, reversal,
  stable origin and stable destination.  Fixed reference windows are 168h,
  24h, 6h, 3h, 6h, 24h and a 72h reversal search where applicable.
- Recurring morphology:
  157 events and matched stable controls have fold-local PCA/GMM morphology,
  posterior, entropy, margin and abstention.  Component identity is not
  continuous across folds; the current support bound correctly says
  `NO_GLOBAL_MORPHOLOGY_TYPES_OR_GATES`.
- The missing semantic alignment is now tested in
  `leave_one_era_out_transition_morphology_alignment_20260730_v1`.  In every
  leave-era-out fold, both the reference prototypes and scoring GMM are fit on
  training eras only, then Hungarian matching freezes the held-era component
  ordering.  Mapping confidence/posterior correlation fail sharply outside
  2022 (for example: 2024 mean confidence 0.301 / correlation 0.377; 2026
  0.252 / 0.286); all three slots have negative minimum cross-fold prototype
  correlation and no slot meets the recurrence gate.  The negative verdict is
  therefore stronger than the previous "IDs are fold-local" caveat: no global
  morphology type is currently justified.

## Performance qualification status

Existing reports contain worst-week calendars, feature shifts, covariance
shifts and interaction diagnostics.  No BH-controlled recurring common
driver has yet survived.  Historical state categories show relative
differences, but all remain net-negative; the held-out category audit found
no stable-good, exact-comparable economic category.

The reporting code has now been extended, with focused tests passing, to
produce week/month Q10/Q50 and positive-period shares, side × state ×
transition × period attribution with explicit availability, conservative
category-stability qualification, and worst-versus-regular composition and
within-category gross/cost/net decomposition.  Category attribution is
strictly downstream of the pooled global top-10 selection.

The refreshed evidence plane is materialized in
`reconstructed_base_residual_stack_2022_2024_20260730_v4`,
`stack_performance_calendar_2022_2026_20260730_v4` and
`stack_regime_failure_analysis_2022_2026_20260730_v5`.  It contains 628,471
hourly candidates and 168 complete weeks / 39 complete months.  Meaningfully
positive alpha IC occurs in 127/168 weeks (75.60%) and 32/39 months (82.05%),
whereas meaningful positive pooled-global top-10 net EV occurs in only
1/168 weeks (0.60%) and 0/39 months.  Median week/month net EV is
-88.47/-95.93 bps.  This is now an explicit target-mismatch and
rank-to-execution-conversion investigation, not evidence that better alpha IC
improves the final book.

No feature, covariance or pair interaction is both BH-significant and
recurrent under the weekly inference contract.  `range_climax_reversal` is
the strongest single exploratory discriminator (oriented AUC 0.708), but
does not satisfy the recurrence gate.  The worst-versus-regular net gap is
-130.16 bps: only -4.33 bps is explained by category composition and
-125.84 bps by worse payoff inside the same side × observed-state ×
transition-phase cells.  This argues against a simple regime reweighting
repair and for learning trust/conversion within regimes and transitions.

The sealed v5 worst-period report additionally gives asset and policy-exit
attribution after that identical pooled-global selection.  It covers 216
weekly/monthly reporting books, with asset shares correctly denominated over
the complete selected book (not an evidence-grade-local denominator); all 216
asset-share sums reconcile to one.  Median largest-asset share is 11.2% by
week and 9.2% by month (median HHI 0.048/0.039), while the occasional 100%
book remains visible rather than averaged away.  Exit reasons are available
only for 26,853 weekly and 26,848 monthly selected exact-source rows; 36,046
and 36,010 reconstructed selected rows are explicitly `unavailable`, never
imputed.  Consequently the exit-reason worst-versus-regular decomposition is
an availability-limited diagnostic, not evidence for a particular exit mode.
For the two historical lineage/evidence grades that support the decomposition,
the worst-minus-regular net gaps are -116.35 and -156.19 bps and are almost
entirely within-asset payoff shifts (-107.67/-155.48 bps), not asset-mix
composition (-8.68/-0.71 bps).  This remains descriptive and grade-separated;
it is not a promotion result or an exit-policy inference.

`alpha_execution_ev_divergence_2022_2026_20260730_v3` diagnoses this without
pooling evidence grades.  In the A-grade cohorts, alpha-to-first-touch IC is
0.164 for March–April 2025 and 0.156 for May–July 2026, while alpha-to-net IC
falls to 0.088 and 0.029.  The first-touch target itself still has strong
net-EV rank IC (0.603 and 0.566), so the target is not intrinsically
irrelevant.  The weak link is conversion from the base score into
cost-clearing payoff under an approximately 100 bps explicit cost hurdle.
July is a separate failure mode: alpha-to-net IC is -0.118 and alpha decile
10 averages -143.9 bps versus -90.9 bps for decile 1.  The next conversion
ablation should therefore estimate a cost-clearing/payoff hurdle conditional
on alpha and regime/transition context, rather than replacing the alpha
target wholesale.

### Strict cost-clearing conversion checkpoint (sealed 30 July)

`a_grade_cost_clearing_conversion_ablation_20260730_v6` is the A-grade
conversion checkpoint.  It uses hourly candidate rows; one-minute bars remain
nested only in the pre-existing exact 12-hour execution labels.  The runner
now seals immutable, resumable 14-day chronological OOF fold checkpoints and
loads prior checkpoint payloads once per fold before vectorised arm partition,
removing the prior growing-ledger lookup.  Seven checkpoints were verified
(five scored, two warmup); the slowest scored fold was 9.84 seconds and total
checkpoint work was 36.84 seconds.  A later fold fails closed unless every
earlier checkpoint has been sealed against the same exact/context identity
intersection and its labels resolved before the new block.

The frozen 2025→2026 verdict is negative.  Both eligible arms fit on 2025
only and map with 80,682 blocked-OOF 2025 score/outcome rows; no 2026 label
is used in fitting or mapping.  On 52,295 common 2026 rows and pooled-global
monthly top-10 selection after each causal map, residual baseline is -96.16
bps average monthly net EV (-126.02 bps in July) and the alpha cost-clearing
hurdle is -106.21 bps (-114.36 bps in July).  Thus the simple hurdle does not
clear the cost barrier and is not promotable.  The regime, transition and
combined hurdle arms remain explicitly
`fail_closed_noncomparable_2025_2026_context_feature_contract`: their 2025
and 2026 sidecars have incompatible semantic columns, so they were not fit,
mapped, or forward-scored.  This is a valid baseline/conversion result, not
evidence for cross-era context transfer.

The morphology outcome binding remains non-promotional.  It covers 118 event
outcome slices but has no 2026 slice and no matched causal regime/transition
baseline.  It is reported separately by `A_2022_23`, `A_2024` and `B_2025`;
the alignment artifact correctly marks its outcome increment as
`NOT_IDENTIFIABLE_NO_MATCHED_CAUSAL_BASELINE`, rather than pooling grades or
inferring an EV benefit.

Calendar gaps remain explicit rather than silently treated as normal:
January–February and May–December 2025, January–April 2026, plus partial
August 2022 and July 2026.  The early-2022 inverse-contract population is a
separate research lineage.

Therefore the healthy-classification requirement is open.  A category may be
called useful only when it repeatedly shows the same directional performance
effect across train-era folds and untouched 2026, with independent long/short
support and exact-policy economics.

## Soft outputs and stack integration

`frozen_contextual_score_arms_2023apr_2025jun_20260730_v1` contains 521,570
exact pre-2026 blocked-OOF candidates and four fixed Ridge arms: baseline,
regime-only, transition-only and combined.  It excludes non-semantic raw
regime leaf/posterior identities.  Fit diagnostics are in-sample only and
must not select an arm.  July–December 2025 compatible candidate OOF coverage
is explicitly unavailable.

The 2026 applicator and evaluator exist, but no 2026 arm comparison has yet
been run.  The applicator requires sealed candidate scores and authoritative
regime/transition sidecars; the evaluator applies separate causal monthly EV
maps and one pooled global monthly top 10%.

That authority is now sealed in
`authoritative_soft_regime_transition_sidecars_20260730_v1`: separate
`soft_regime_hourly.parquet` and `soft_transition_hourly.parquet`, each with
33,895 unique hourly timestamps from 2022-08-30 through 2026-07-12 19:00 UTC.
They retain 17,532 blocked-OOF timestamps, 11,736 explicit unavailable warm-up
timestamps and 4,627 untouched-2026 forward timestamps.  The checksum-bound
`cadence_audit.csv` finds zero duplicate and zero non-hourly rows; it records
that native 15m values are lookbacks sampled onto the hourly row and that 1m
is nested replay/label evidence only.  `label_resolution_audit.csv` contains
58,536 source/audit records with no missing resolution time, and all scored
historical fit-label maxima are strictly before their fold train end.

The regime sidecar is limited to causal BOCPD change/run-length/uncertainty
context.  The transition sidecar adds frozen strict-LGBM probability plus
BOCPD stable/1h/3h/6h/12h probabilities, entropy/margin, availability and
train-end provenance.  No diagonal/sticky/DAE identity, posterior axis or
morphology ID is present.  OOD values are explicitly unavailable rather than
invented.  `bocpd_reliability.csv` marks every BOCPD logistic output
`CONVERGENCE_AND_CALIBRATION_LIMITED_DIAGNOSTIC_ONLY`; its probability fields
are context/provenance only, not a promotion, gate, quota or standalone score.

`final_v3_context_interaction_diagnostics_20260730_v2` is the valid
interaction diagnostic.  It uses 521,570 final-v3 conflict-checked/coalesced
pre-2026 OOF rows for fixed, side-local tree-SHAP discovery and conditional
permutation, then assesses 127,777 untouched-2026 rows once.  Residual score
conditional importance is positive in all 24 pre-2026 periods for every
regime/transition/combined conditioning case; long confirms direction in all
three forward cases, short confirms transition and combined but not regime.
Base score effects are not stable enough to pre-register.  The leading SHAP
interactions are residual × BOCPD state-age/run-length fields, not raw state
IDs.  Large pre-to-2026 covariance reversals between residual score and LGBM
probability/entropy/margin (long absolute shift up to 0.413) reinforce that a
follow-on must be a fixed residual-interaction arm, assessed separately by
period and side.  It must not tune on 2026 or promote BOCPD.  The earlier
final-v2 interaction diagnostic is retained only as invalid diagnostic evidence:
v2 had complementary empty era-specific score/label columns.

`run_final_identical_row_regime_stack_gam_ablation.py` is now prepared but
intentionally not run.  Its only valid authority is the sealed, timestamp-level
sidecar pair `soft_regime_hourly.parquet` / `soft_transition_hourly.parquet`;
these are not candidate-level files.  It checksum-binds both files and their
manifest, requires `model_sample_cadence=1h` and `assessment_sample_cadence=1h`,
and rejects duplicate or non-hourly timestamps.  The exact candidate identity
remains owned by the score ledgers and is preserved through a many-candidates to
one-hourly-context join on `__ts__=source_utc`; missing, warm-up or provenance
mismatches fail rather than filtering/substituting rows.  Historical rows must
be `blocked_oof_2022_2025` with each fit-label resolution strictly before its
fold train end; 2026 rows must be `untouched_2026_forward`.

The pinned ledgers are the 521,570-row
`frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet`
and the 127,777-row May–July 10
`mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1/allscore_waterfall.parquet`.
Their sealed lineage is checked: the former proves candidate-held blocked OOF
with labels resolved before freeze; the latter proves strict prior-resolved,
side-local OOF base/residual scores and the exact 12-hour replay economics.
The forward ledger ends 2026-07-10 and the currently expected sidecar ends
2026-07-12, so it may run only if the exact common hourly join and availability
checks pass—no extension or manufactured context is permitted.

Its ten arms are baseline; regime-only, transition-only and combined context
separately in base and residual/trust; and bounded additive spline-GAM
calibrators for the same three contexts.  It uses semantic BOCPD change,
run-length, age and persistence fields plus LGBM/BOCPD transition probabilities;
raw GMM/state identities, posterior axes, morphology and unavailable OOD fields
are excluded.  Every arm is side-local, trained/calibrated only before 2026,
uses a frozen arm-local causal EV map, and reports the fixed pooled-global
top-10 book with IC, mapped/raw EV, Q10/Q50 week/month, latest/worst, sides,
recall, calibration, ties and concentration.  One-minute bars are never model
or assessment examples: they remain nested inside the exact 12-hour labels and
replay only.

### Final identical-row ablation (sealed 30 July)

`final_identical_row_regime_stack_gam_ablation_20260730_v3` is the
authoritative strict-forward result.  V1 and v2 are diagnostic-only and must
not be used.  V1 exposed automatic isotonic direction and candidate-ID
tie-breaking.  V2 fixed those but revealed a second lineage bug: 2023–2024
stored scores under the canonical columns and label resolution under
`execution_label_available_at`, whereas 2025 stored the same OOF scores under
`base_oof_score` / `residual_expected_ev` and its label endpoint in
`execution_label_end_utc`.  The columns were complementary, but v2 did not
coalesce them, so its historical baseline map was constant and learned arms
did not receive the intended score history.

V3 conflict-checks and coalesces these era aliases.  All 521,570 historical
rows now have a resolved 13-hour label endpoint and non-null base/residual
scores.  Residual and GAM arms use five chronological OOF folds from April
2024 through June 2025; base-alpha arms retain one 2025 fold because the
first-touch alpha target exists only in 2025.  Mapping is monotone increasing,
and selection is one pooled global top 10% by mapped EV, with raw score used
only inside exact mapped-EV ties.

All arms assess the same 127,777 May–10 July 2026 hourly candidates.  The
521,570-row historical ledger contributes 418,140 context-available candidates;
103,430 explicit pre-OOF warm-up candidates are reported and excluded from
every arm rather than imputed.  The map is frozen and 304 days old at the
forward boundary; this transfer age is retained in every metric.

No arm clears the economic promotion gate.  The raw residual baseline is
-77.51 bps at pooled-global top 10%.  GAM+regime is best aggregate at
-57.92 bps (+19.59 bps) and improves both long and short economics, but July
worsens to -134.68 bps.  GAM+combined is -59.28 bps (+18.23 bps), raises
execution IC from 0.053 to 0.094, improves both sides and improves July to
-105.32 bps.  GAM+transition is the most balanced time-transfer arm:
-70.77 bps aggregate, -72.22 bps in July, -52.44 bps in the latest week, and
better weekly/monthly Q10 than baseline; its short-side EV is slightly worse.
All remain negative because selected gross opportunity remains well below the
approximately 100 bps cost burden.  Mapping tie mass is materially lower than
v2 but remains 13–49%, so resolution is still an explicit failure mode.

The useful signal is diagnostic: separate regime and transition context can
change ranking and can repair particular months, but the current
2022–2025-trained representations do not transfer consistently enough to
control trading.  No portfolio replay or policy gate is authorized from this
result.

The v3-only fixed convex GAM-expert mixture is also sealed at
`final_v3_gam_convex_mixture_ablation_20260730_v1`. It uses 15 raw-score
simplex weights over the separate regime-only, transition-only and combined
GAM arms, selected solely by pre-2026 OOF aggregate, Q10/Q50 week/month and
both-side gates before one frozen 2026 application. No mixture passes. The
best diagnostic weight (25% regime / 75% transition) is -64.86 bps OOF
aggregate and -203.70 bps weekly Q10; it is non-promotable. A learned gate
was not attempted because the fixed grid already fails every OOF gate.

### Pre-2026 mapping-resolution suite (sealed 30 July)

`pre2026_mapping_resolution_ablations_20260730_v1` is **invalid and
non-authoritative**: it was bound to the superseded v2 stack and must never be
used for comparison, selection or promotion.  It is retained only as a
lineage record.  The authoritative replacement,
`pre2026_mapping_resolution_ablations_20260730_v2`, checksum-binds the v3
manifest and evaluates the same 127,777 frozen 2026 candidates for all ten
arms.  Its seven methods were fixed before reading 2026 outcomes: monotone
isotonic control; strict rank-preserving isotonic; side-isotonic shrinkage at
5k and 25k support; 64-bin/1k and 32-bin/2.5k support-constrained maps; and
strict-rank 64-bin/1k.  All map fitting uses only pre-2026 OOF raw scores and
resolved targets; the 2026 labels are assessment-only.  Resolution gates are
diagnostic-only and never filter, admit or reorder a candidate.

The important split is clear.  Strict rank preservation removes the artificial
isotonic plateaus—mean largest tie mass falls from 24.93% to 0.017%, cutoff
tie share from 5.96% to 0.010%, and all 10 resolution gates pass—while mean
top-10 EV is effectively unchanged at -72.14 bps versus -72.15 bps for the
control.  Thus the tie repair is valid bookkeeping, not an economic edge.
Support-shrunk and binned maps alter cross-side ordering but do not improve
the suite consistently: mean top-10 EV is -73.61 to -74.31 bps, and the
weekly/monthly Q10 and latest-month averages worsen.  There are isolated
aggregate gains (for example GAM+regime reaches -56.82 bps under 64-bin/1k,
about +1.10 bps versus its control), but its July loss deepens; no method has
positive aggregate, latest, Q10 or both-side net economics.  Strict-ranking a
binned map repairs ties again but does not repair that economic failure.  No
mapping method is promoted and portfolio replay remains unauthorized.

Cadence invariant: **no violation found.**  The authoritative regime and
transition sidecars declare model and assessment cadence `1h`; v3 historical
OOF (3,141,652 arm rows), v3 forward assessment (1,277,770 arm rows), and the
894,460 selected mapping rows all have zero non-hourly decision timestamps and
zero label endpoints at or before their decision.  Mapping has no HPO or model
refit; its predeclared method grid, fit, OOF selection and forward assessment
operate on those 1h rows.  One-minute data remains nested exclusively in the
existing barrier/path/fill replay that supplies each hourly 12-hour label.

### Fixed residual-interaction follow-on (sealed 30 July)

`final_v3_preregistered_residual_interactions_20260730_v1` is the only
permitted follow-on from corrected
`final_v3_context_interaction_diagnostics_20260730_v2`.  Its pre-registration
is frozen before reading 2026 outcomes: long residual × `regime_state_age_hours`,
long residual × `transition_lgbm_probability`, long combined, short residual ×
transition probability and short combined.  Short regime-only is deliberately
excluded because it did not qualify in pre-2026 evidence.  Each arm uses a
fixed low-capacity StandardScaler + Ridge(alpha=80) learner for the stated
side; the other side remains the frozen residual baseline, preserving every
candidate in pooled global selection.  Context fields appear only as explicit
residual products—no raw state identity, GMM posterior, morphology field, or
BOCPD standalone score/gate/quota enters the run.

The result is negative.  On the identical 127,777-candidate 2026 global
top-10, baseline is -77.51 bps.  Long regime, transition and combined are
-77.39, -77.51 and -77.30 bps; short transition and combined are -99.23 and
-99.34 bps.  July is negative in every follow-on arm (-143.40, -143.30,
-143.26, -93.14 and -93.14 bps respectively).  This is not an action-layer
test and authorizes no timing, waiting, target-price or portfolio change.

Cadence was audited separately for this follow-on and **no violation was
found**.  `row_cadence_audit.csv` records zero non-hourly and zero duplicate
candidate-identity rows in the 521,570-row historical score ledger, 418,140
context-available historical fit/OOF rows, and every one of the 127,777
forward assessment rows.  Mapping uses only those pre-2026 hourly OOF rows;
1-minute source data remains nested exclusively inside existing 12-hour
label/path/replay inputs and never forms a fit, OOF, mapping or assessment
observation.

### Hourly transition-semantic support extension (sealed 30 July)

`hourly_transition_semantic_signature_ablation_20260730_v1` tests the
strongest valid way to increase transition support without inventing durable
cluster IDs.  It labels whether an event onset occurs in the next 1--3 hours
on an hourly causal panel, but retains `next_event_id` on every positive row:
three onset-window rows remain one physical event, not three independent
examples.  The split is strict: 2022--2025 fit and semantic naming, with
label availability before 2026; 2026 is assessment only.  No execution
outcome, candidate score, policy result, GMM/HDBSCAN component, raw state ID,
or held-era outcome enters the diagnostic.

The support increase is real but not transferable: training has 423 positive
hourly windows from 141 events; 2026 has 48 windows from 16 events.  Fixed
causal feature-group signatures score 2026 AUC 0.436 (breadth dislocation),
0.526 (washout/reversal), 0.436 (funding/positioning) and 0.467 (combined).
All 95% UTC-week block-bootstrap AUC intervals cross 0.5.  Leave-one-era-out
coefficient directions are also unstable: the strongest group,
washout/reversal, has minimum pairwise correlation 0.238, below the fixed
0.70 criterion.  It therefore neither supplies a reusable semantic transition
type nor rescues `recurring_transition_clusters`; no model, gate or policy
input is authorized.

Cadence remains strictly `1h`: the panel, onset labels, 2022--25 fit and 2026
assessment each use exact hourly rows; 1-minute data is not read by this
diagnostic.

### Objective audit snapshot

`regime_objective_completion_audit_20260730_v20` is sealed with 52 evidence
requirements proved and four still incomplete.  The incomplete requirements
are substantive, not missing execution: no recurring transition morphology
survives leave-era-out alignment; the final 12 December context hours cannot
yet reproduce the frozen state; no pre-2026 failure/value correction earns a
2026 application; and no stable-good regime category survives the fixed
global-top-10 economic gate. This is why the research objective remains active
even though the infrastructure and requested ablations are materialized.

## Open requirements

### July 2025 compatible score-ledger bridge and raw-context extension (sealed)

The v3 H2 bridge audit has been narrowed: a **compatible July route is
sealed**, but it is a separately scoped frozen 30-asset common-universe
bridge, not a replacement for the wider final-ledger population.  The exact
1h candidate identity is verified for all 44,640 July rows (22,320 per side):
every candidate joins one-to-one to both native first-touch supervision and
the candidate-local 1m-derived 12h execution ledger.  Both model features and
scores use the hourly clock; 1m is nested only in the existing execution
label/path replay.

`scripts/materialize_july2025_common30_final_base_residual_oof_bridge.py`
materialises the frozen side-local 31-long/8-short base and accepted residual
contracts with no feature selection or HPO.  Base and residual fit rows must
have native decision+24h label resolution strictly before
2025-07-01T00:00:00Z; exact execution end/availability timestamps remain in
the output separately.  It reads no 2026 outcome.  The planned sealed output
is `july2025_common30_final_base_residual_oof_bridge_20260730_v1`, containing
the canonical `score_base_alpha` and `score_residual_expected_ev` pair,
per-side fold provenance, candidate-identity hash, source hashes and 1h
cadence declaration.

Materialisation is safely resumable because the execution environment stops a
single long historical feature read.  The June frozen PIT snapshots match
82.789% (long) and 82.690% (short) of the July expanding-window base training
selection exactly; only 17,211/17,310 new rows are read through the strict PIT
loader.  Existing partial matrices are accepted only at the exact expected
row count and immutable selected-column sequence; score files are not reused
without their exact candidate identity.  The accepted LightGBM constructor's
frozen worker setting is retained unchanged, as are learned/HPO parameters,
features, labels, split and seed.  This is not a new model search.

The completed separate raw-score context extension is
`july2025_common30_regime_context_raw_score_extension_20260730_v1`.  It joins
the sealed authoritative hourly regime/transition sidecars one-to-one to the
July bridge and fits frozen low-capacity arm definitions side-locally on the
418,140 context-available final-v3 rows whose exact execution label endpoint
is strictly before 2025-07-01.  It tests residual-LGBM and bounded spline-GAM
in regime-only, transition-only and combined placements, plus the frozen
residual raw-score control.  It has no EV map, no HPO, no feature selection,
no July training row and no 2026 outcome.

The July diagnostic is directionally useful but not tradable: GAM+regime has
the least-negative global raw top-10 at **-88.40 bps** and a +0.0226 execution
rank IC, compared with **-101.75 bps** and -0.0391 for the raw residual
control.  Its long/short selected EV remains -90.75/-87.49 bps.  Every
residual-LGBM arm, GAM+transition and GAM+combined is worse.  Therefore this
supports only a future pre-registered low-capacity regime-context test; it
does not refresh mapping or authorize a gate, policy replay or promotion.

`july2025_common30_all_context_map_refresh_20260730_v1` then performs the
requested causal sensitivity for baseline plus every compatible final-v3
residual/GAM arm.  Each arm compares its 381,814 historical OOF raw-score map
with the same map after appending all 44,640 sealed July OOF rows, both as
ordinary and rank-preserving monotone isotonic mappings, before one fixed
assessment of the exact 127,777 hourly 2026 candidates.  The refreshed map
uses labels ending no later than 2025-08-01 12:00Z; no 2026 label or outcome
is read for fit, selection or tuning.  Every one of the seven arms has an
exactly identical selected 2026 global top-10 set under all four variants.
This is expected: monotone mapping with raw-score tie-break cannot reorder
different raw scores.  Thus GAM+regime remains -57.92 bps aggregate,
-134.68 bps in July, -85.30 bps weekly Q10 and -121.02 bps monthly Q10, with
-72.16/-51.38 bps long/short; it remains non-promotable.

The old August--November availability statement is superseded.  The strict
score-only PIT preflight
`augnov2025_pit_scoring_preflight_20260730_v2` verifies all 175,680 common-30
hourly candidates, and the subsequent frozen-through-July base/residual OOS
bridge is sealed.  Exact 1m-derived paths remain nested label/economics
evidence only; all features, scores, mapping inputs and assessments retain one
hourly row per candidate.  This is a common-30 sensitivity, not a
population-identical replacement for the wider v3 ledger.

### Aug--November materialisation challenge (sealed)

The preceding availability conclusion is now being tested at the correct
boundary.  The exact ledgers contain 44,640 hourly common-30 candidates in
August and October and 43,200 in September and November.  The sealed
full-population (not a probe) preflight covers all **175,680** rows: every
candidate has one-to-one native-base and exact-execution identity, an exact
hourly PIT key and finite frozen inputs for the 31-long/8-short base contracts
and all 69 residual fields per side in every month/side/symbol cell.  Therefore
exact execution labels can serve as the OOS candidate/economics ledger even
though no future native first-touch targets are required for scoring.

`scripts/materialize_augnov2025_frozen_july_oos_bridge.py` is an explicitly
staged, resumable test: fit base, score each side/month, fit residual, score
each side/month, then seal.  Its base and residual fits use only native labels
resolved before 2025-08-01, the frozen 31/8 and residual contracts, and no
HPO or 2026 data.  The final 175,680-row OOS score bridge is now sealed.  It
remains non-promotable because it is a common-30 OOS sensitivity rather than
an identical-population blocked-OOF extension.  Preserve score/identity
lineage; do not reclassify labels as model evidence without it.

### Aug--November bridge and fixed-context assessment (sealed 30 July)

The staged challenge succeeded.  The sealed
`augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1` has
175,680 unique common-30 candidates: 87,840 per side, exact hourly cadence,
execution endpoints strictly after decision time, and both score-fit cutoffs
fixed at 2025-08-01.  Its sealed validation/economics report is
`augnov2025_frozen_july_oos_bridge_validation_economics_20260730_v1`.
Residual raises overall execution rank IC from 0.0269 (base) to 0.0470 and
top-10 EV from -109.92 to -97.53 bps, but remains negative in every month and
has week/month Q10 of -131.78/-112.09 bps.  The score bridge is therefore
valid evidence of scope-limited OOS degradation, not promotion evidence.

Six fixed side-local context arms were then fitted only on compatible
historical plus July OOF labels resolved before 2025-08-01 and scored on all
August--November rows.  The authoritative result is
`augnov2025_common30_fixed_preaug_context_oos_extension_20260730_v2`.
The v1 artifact is invalid/non-authoritative because its period file contained
selected candidates rather than period aggregates; v2 corrects this without
changing the fixed arm/training contract.  Best aggregate economics is bounded
GAM-regime (-93.60 bps, IC 0.0557), followed closely by residual-transition
(-97.26 bps); neither clears costs or tail gates.  GAM-regime's monthly Q10 is
-110.29 bps and residual-transition's is -104.45 bps.  All selection is one
pooled global top-10 per arm; week/month tables only decompose that fixed
membership.  No EV map, action layer, portfolio replay or promotion is
authorized.

### H2 common-30 final-refit GAM sensitivity (sealed 30 July)

`final_refit_h2_common30_gam_sensitivity_20260730_v2` refits only the fixed
bounded GAM regime, transition and combined arms, side-locally, on compatible
pre-2026 labels: v3 historical OOF plus July OOF plus the sealed
August--November frozen-July OOS bridge.  Their increasing/rank-preserving
maps use only the corresponding concatenated pre-2026 OOF/OOS raw-score
ledgers; the existing 127,777-candidate 2026 hourly universe is assessment
only.  This is necessarily a common-30 H2 sensitivity, not a population-
identical v3 replacement.

The refit regime GAM is the best new arm (execution IC 0.0855, -65.91 bps
global top-10), ahead of refit combined (-66.51 bps) and transition (-78.60
bps), and improves the frozen residual baseline (-77.51 bps).  It still fails
economically: its week/month Q10 is -96.48/-116.40 bps and its latest July is
-126.35 bps.  The frozen v3 GAM controls remain stronger: regime -57.92 bps
and combined -59.28 bps in the sealed
`final_refit_h2_common30_gam_vs_v3_controls_20260730_v2` comparison.  V1 is
non-authoritative only because its reported aggregate H2 map-support count was
wrong; v2 records actual per-arm support with unchanged scores and economics.
No H2
refit surpasses the established control or clears tail/cost gates; do not
promote, replay or alter policy.

### Three-era common-30 category-stability rerun (sealed 30 July)

`h2_common30_regime_category_performance_stability_20260730_v3` supersedes
the prior availability-only conclusion for the H2 common-30 scope.  It has
three non-overlapping pre-2026 evaluation eras (July, August--September and
October--November), both-side support, 22,032 candidates from one pooled H2
global top-10 before category attribution, and a separate untouched-2026
assessment.  Regime state, transition probability and their fixed combined
layer remain distinct; no ex-post phase is a gate.  Seven observed categories
have all three eras and both sides, but **zero** has stable positive
leave-era-out transfer.  The `regime_category_performance_stability`
requirement is therefore still incomplete, and cannot authorize promotion.

The follow-on failure-risk/trust preflight,
`h2_category_failure_risk_trust_ablation_20260730_v1`, also fails closed.
Its pre-registered minimum leave-era rank-stability threshold is 0.70; regime
and combined reach -0.429 and transition -1.0. No stable relative failure
ordering exists, so no 2026 exclusion, downweighting, GAM trust correction or
portfolio test is run.

### Frozen-July residual diagnosis: why August--September loses while October--November helps

`augnov2025_frozen_july_residual_regime_diagnosis_20260730_v2` is the
checksum-sealed explanation layer.  It uses the same 175,680 common-30
candidates, frozen pre-2025-08-01 base/residual scores and exact
1m-derived execution labels as the bridge, but every model, score, context and
assessment row remains one **hour**.  The 1m path is nested only in the label.
It reads no 2026 outcome, fits no calibrator and keeps current-regime and
transition fields separate.

The residual improves aggregate rank IC from 0.0269 to 0.0470 and pooled
global-top-10 EV from -109.92 to -97.53 bps, but its economic effect is
explicitly phase dependent:

| Period | Base top-10 EV | Residual top-10 EV | Residual minus base |
|---|---:|---:|---:|
| Aug 2025 | -107.84 bps | -115.00 bps | -7.16 bps |
| Sep 2025 | -96.99 bps | -99.08 bps | -2.10 bps |
| Oct 2025 | -148.75 bps | -105.31 bps | +43.44 bps |
| Nov 2025 | -79.76 bps | -69.47 bps | +10.29 bps |

The book replacement accounting is exact: 9,333 residual entries replace
9,333 base entries, whose mean label is -132.20 bps; entrants average
-108.87 bps, a +23.33 bps replacement gap.  The mechanism reverses in August
and September (-11.15/-11.07 bps entry-minus-exit gaps), then becomes strongly
positive in October (+75.46 bps) and modestly positive in November (+27.37
bps).  Long is responsible for the pooled gain (+22.17 bps versus base), while
short worsens (-10.33 bps).  The residual changes 53.1% of the global top-10
book; it is not a tie-resolution effect.

From Aug/Sep to Oct/Nov, BOCPD state age falls 15.43 to 9.91 hours (-0.43
pooled SD), persistent-24h state incidence falls 21.8% to 8.6%, and long
`range_12h_pct` rises 0.039 to 0.058 (+0.40 SD).  Shorter run-length states
become more common while BOCPD onset probabilities rise by about 0.17--0.23
SD.  State-age versus persistent-24h correlation moves from 0.715 to 0.486,
and BOCPD run-length-q05 versus LGBM transition probability from +0.050 to
-0.170.  These remain separate **regime-state** and **transition** facts, not
interchangeable categories.

Conditional residual-target alignment creates hypotheses only.  In late
Oct/Nov, long alignment is weaker when BOCPD onset-H1 is high (0.097 versus
0.173 in the low half) and short alignment is weaker when
stable-vs-transition probability is high (0.012 versus 0.103).  In early
Aug/Sep, short residual alignment is lower at high LGBM transition probability
(0.060 versus 0.135).  Future work must preregister these side-specific fields
and cutoffs from earlier data, then assess on an independent post-cutoff era;
this bridge cannot tune or promote a trust gate.

Category stability remains data-limited. The sealed v2 availability report,
`heldout_regime_category_economics_stability_20260730_v2_availability`,
requires three compatible pre-2026 eras plus untouched 2026, both-side support
and one hourly candidate score/economics/context lineage; it creates no gate.

1. Keep strict rank preservation as the reporting-safe mapping representation,
   but do not treat it as an economic improvement.  Do not add a mapping gate
   until positive aggregate/latest/Q10/both-side economics exist independently.
2. The complementary GAM effects have now been diagnosed in
   `final_v3_context_interaction_diagnostics_20260730_v2`, and the only
   permitted fixed follow-on is sealed at
   `final_v3_preregistered_residual_interactions_20260730_v1`.  It remains
   negative.  Do not widen or promote it before a fresher compatible H2-2025
   hourly blocked-OOF score ledger reduces the current 304-day map age.
3. `h2_2025_identical_row_oof_bridge_audit_20260730_v7` supersedes v6.
   July now has a sealed 44,640-row, both-side, hourly common-30 base+residual
   blocked-OOF bridge (labels through 2025-08-01 12:00Z), reducing the map's
   last-label age by 31.54 days.  This is **not** an identical-population
   extension of the wider v3 ledger.  The sealed baseline-only sensitivity,
   `july_common30_baseline_map_refresh_20260730_v1`, and the all-context
   extension `july2025_common30_all_context_map_refresh_20260730_v1`, therefore
   have no promotion authority: monotone maps retain the same frozen 2026
   pooled books (baseline remains -77.51 bps top-10 EV).  August--November is
   now sealed as a 175,680-row frozen-July common-30 OOS bridge. December is
   now a 44,640-row exact-path common-30 bridge (22,320 per side), scored by
   an immutable pre-August base+residual pair: every fit label resolves no
   later than 2025-07-31 23:00Z.  Its 1h candidate inputs, native identities,
   frozen base/residual PIT fields and exact nested 1m [decision, decision+12h)
   policy labels are all complete. The label boundary reaches 2026-01-01
   12:00Z, but is assessment-only: no December/January execution outcome was
   used to fit, tune, map or select scores. On all 44,640 December rows under
   one pooled global raw top-10, base alpha is -117.55 bps and residual EV is
   -84.69 bps (residual long -61.37, short -116.28 bps).

   The December fixed regime/transition-context sensitivity is deliberately
   partial: the authoritative hourly sidecars end at 2025-12-31 11:00Z, so the
   final 12 timestamps (720 candidate-side rows) are excluded without fill,
   forward-fill or reroute. On the exact 43,920-row common-context subset,
   frozen residual global top-10 is -83.79 bps. Every tested context arm is
   worse (-101.05 to -120.91 bps); no context arm, map, promotion or portfolio
   replay is authorized. The twelve missing decision-time raw input rows do
   exist. However, the separate frozen-pre-December reconstruction is fail
   closed: regime fields reproduce the preceding canonical twelve hours
   exactly, while transition fields do not (LGBM probability max delta
   0.000344; BOCPD stable-vs-transition probability 0.01413 and margin
   0.02826). No final-hour context is appended. Recover the exact serialized
   fold-03 LGBM imputer/calibrator and BOCPD head states—or the original
   persisted reconstruction details—before retrying; never refit, fill,
   forward-fill or reroute. The July raw-context OOF score extension remains
   negative net of costs; never use 2026 labels for map fitting.
4. Improve transition-label support and recurrence.  BOCPD heads remain too
   sparse and poorly calibrated, and no morphology type currently passes
   leave-era-out alignment.
5. Re-run the identical-row arms only after the above changes.  Portfolio
   replay remains blocked until aggregate, latest, weekly/monthly Q10/Q50 and
   both-side economics pass together.

Trajectory-sidecar identical-row integration is fail-closed at
`trajectory_transition_identical_row_stack_coverage_20260730_v1`: historical
coverage is 9,515/9,515 hourly timestamps, but frozen 2026 is only
1,042/1,699 (61.33%). No missingness policy was pre-registered, so no
drop/fill/reroute or stack arm is allowed.

The subsequent missingness-aware treatment is pre-registered at
`trajectory_missingness_identical_row_ablation_preregistration_20260730_v1`
before arm economics: retain all rows; use availability as a feature; neutral
fill unavailable trajectory probability/entropy/margin with 0.5/log(2)/0;
retain existing transition context as fallback. Five fixed GAM arms and all
global-book/tail/both-side/availability-stratified reports are frozen for the
next execution; no cluster IDs, 2026 tuning or 1m model rows.

### Train-only recurring transition prototype and hourly trajectory context (sealed 30 July)

The stronger recurrence study is sealed at
`trainonly_recurring_transition_prototype_study_20260730_v3`.  It uses 314
one-hour causal anchors: 141 transition and 130 matched stable controls in
2022--25, then 16 transition and 27 stable controls in untouched 2026.  Its
fixed outcome-free contract summarizes 12 semantic signals over precondition
(168h), approach (24h), acceleration (6h), and trigger (3h) windows.  Each
sequence is available at or before its anchor.  Current/source state is causal
context; destination state is reported only as post-labelled topology and is
excluded from causal transition-versus-stable scoring.

The useful result is binary, not a subtype taxonomy: trajectory-only
transition-versus-stable scoring transfers to 2026 at AUC 0.745/AP 0.721; the
combined current-regime/transition-context arm is worse (0.713/0.688).  No
K=2--5 train-only prototype passes recurrence: every candidate has a singleton
component, bootstrap ARI is only 0.320--0.555, and none has the required
three-era support.  The best non-promotable geometry (K=4) assigns 131/141
training and 15/16 2026 events to one local component; leave-era matched
centroid cosine falls to -0.200.  Component IDs remain diagnostic-only and
cannot become type names, sidecar fields, gates or policy inputs.

The transferable binary result is now materialised at every hourly source
timestamp in `hourly_trajectory_transition_soft_sidecar_20260730_v1`, replacing
the previous readiness blocker.  It has 33,907 unique 1h rows: 29,060
complete-lookback pre-2026 rows are calendar-era-held OOF scores, 3,927
complete-lookback 2026 rows use the frozen 2022--25 fit, and 920 warm-up or
missing-lookback rows explicitly fail closed.  It emits only source timestamp,
source state, availability, probability, entropy, top-2 margin, held era and
fit provenance—no destination or type ID.  All 96 causal trajectory fields are
feature-identical to the anchor contract (zero mismatch; maximum float error
2.01e-7).  The 2026 complete-lookback anchor subset (38 rows / 15 transitions)
has AUC/AP 0.849/0.832, Brier 0.174 and ECE10 0.174.  This is a diagnostic
OOF/frozen context candidate for a separately specified GAM/base-residual
ablation, never direct policy authority.

`recurring_transition_clusters` therefore remains **incomplete**: binary
detection is materially stronger and stack-ready, but no stable recurring
transition *type* exists yet.

### Explicit model-failure and incremental-value targets (sealed 30 July)

`pre2026_oof_model_failure_incremental_value_20260730_v3`, with the immutable
hourly-cadence supplement `..._v4`, replaces generic regime-category labels
with three selected-book targets: residual-versus-base incremental utility,
residual top-10 net-cost failure, and top-tail false-positive severity. Each
era label uses one pooled global top-10 across timestamps and sides before
feature-coverage filtering. Fitting is side-local leave-era-out, fixed
low-capacity and pre-2026 only; neither artifact contains a 2026 score, label,
economics or policy application.

Coverage is arm-local: trajectory has 785,750 rows across nine eras, including
2023-Apr--Dec. Regime, transition and combined have 682,320 rows across eight
eras because standard regime/transition sidecars are unavailable in 2023.
Trajectory availability must not be used to claim their coverage. The
alternative March--April bridge is excluded because its base map differs.

All four context arms pass incremental-utility transfer (median held-era
Spearman 0.1065--0.1095; every held era positive), and all pass selected-net-
failure transfer (median AUC 0.5408--0.5507). Every false-positive-severity
arm fails: its worst held-era rank metric is negative. Those raw target results
are not incremental context evidence because every passing head includes CORE
scores. The authoritative joint rerun,
`pre2026_joint_score_context_incremental_gate_20260730_v2`, supersedes the
provisional v1 score-control artifact (which remains lineage only and
non-authoritative). It freezes one side-local implementation, 150k candidate-
hash cap, arm-availability mask, folds and rows for context and CORE-only
control: all **132/132** arm/target/era/side train+test cohorts are equal.
Every context gate fails. Utility median deltas are combined -0.005055, regime
-0.000275, trajectory -0.000769 and transition -0.003997. Failure medians are
combined +0.001935, regime +0.005315, trajectory +0.002259 and transition
-0.009070, but their positive-era fractions are only 0.500/0.625/0.556/0.375
and their worst deltas are -0.060785/-0.047085/-0.028010/-0.054822. None
clears the fixed median/positive-era/minimum/era-count gate. This is now a
valid negative result: generic regime, transition and trajectory context adds
no stable incremental value beyond CORE scores. Do not apply any head to
frozen 2026.

The sealed environment provenance,
`pre2026_joint_score_context_incremental_gate_environment_20260730_v1`, binds
the joint-v2 manifest and code hashes (Python 3.12.2, NumPy 1.26.4, pandas
2.3.3 and scikit-learn 1.6.1). Independent review
`pre2026_joint_score_context_incremental_gate_independent_review_20260730_v3`
recomputes the matched deltas and OOF metrics, verifies source/output hashes,
arm-local identity, 1h cadence and the no-2026 boundary; it independently
concludes no context correction is authorized.

Final `frozen_2026_failure_value_correction_preregistration_20260730_v3`
supersedes v1/v2. It checksum-binds joint v2 and its environment provenance,
records zero eligible heads and `authorized=false`, and explicitly prohibits
any 2026 candidate/economics read or correction score. No application, replay
or promotion is allowed; severity remains excluded. A future application
requires a materially different mechanism that first beats the exact jointly
reviewed score-only gate, then a new no-2026-read preregistration.

The v4 cadence supplement proves candidate timestamps are UTC hour-aligned and
labels resolve before 2026. Newer IDs encode `1h`; legacy 2023 IDs are hashed,
with cadence established by their sealed 1h source contract and timestamp
audit. 1m remains nested exact-12h label/economics evidence only.

### Nested failure-overlay and gamma-HPO result (sealed 30 July)

`pre2026_nested_residual_context_failure_overlay_20260730_v3` is the
authoritative nested residual-context failure overlay; v4 is its audit
supplement. Together they pass every cohort, 1h-cadence, label-boundary and
high-minus-low EV assertion (66 core-overlay cohorts and 66 v1-outer rows),
without any 2026 input. Exact regime-overlay gamma HPO at
`pre2026_regime_overlay_gamma_hpo_20260730_v2` has 68,234-row prediction
parity. Gamma 0.125/0.25/0.5 improves median failure AUC by
+0.002716/+0.005021/+0.006788 (each 75% positive eras) and improves Brier/AP
with positive both-side medians. That diagnostic improvement does not convert
to stable economics: median high-minus-low EV deltas are
+0.000096/+0.000813/+0.000577, with only 0.375/0.375/0.500 improving eras.
Every gamma fails the economic gate; `selected_gamma=null` and
`authorized_for_2026=false`. This is completed negative HPO evidence, not an
exception to the joint score-control result or frozen-2026 no-read fence.

### Regime-only hourly expected-downside broadcast (sealed 30 July)

`pre2026_regime_only_downside_risk_broadcast_20260730_v2` is superseded and
non-authoritative: it mislabeled a candidate-level failure probability as
expected downside. The authoritative replacement,
`pre2026_regime_only_downside_risk_broadcast_20260730_v3`, uses the corrected
v2_r1 hourly OOF heads separately for score-only and regime context:
`clip(book_opportunity) * clip(book_failure_rate_if_selected) *
clip(book_downside_severity_if_selected)`. This is an expected loss per trade
in return units. It broadcasts that exact scalar equally to every candidate in
the hour, so within-hour order is invariant (maximum numerical penalty span at
most 7.9e-18). All eight supported pre-2026 eras have complete hourly head
coverage; zero fallback remains explicit but is used zero times. Selection is
one deterministic pooled global top-10 within each held era, with week/month
tables only decomposing that fixed book.

The disclosed fixed lambda grid is 0.25/0.5/1.0 because the scalar is already
in return units. Every score-only and context arm is rejected against the
absolute residual control. The least-bad score-only lambda .25 gains +0.53 bps
aggregate but has minimum-era -2.74 bps, week-Q10 -14.33 bps and month-Q10
-13.28 bps. Context lambda .25 gains +0.68 bps aggregate with positive
long/short deltas (+1.07/+0.49 bps), but minimum-era is -2.55 bps and
week/month Q10 are -14.50/-12.29 bps. At lambda .5 and 1.0, aggregate and/or
both-side results deteriorate further. No lambda is selected and
`authorized_for_2026=false`; do not turn this broadcast into a 2026
application, replay or policy change.

### One-row-per-hour book-risk calibrator (sealed 30 July)

`pre2026_hourly_book_risk_calibrator_20260730_v2_r1` is the corrected
authoritative timestamp-level study. The earlier v2 is retained only as failed
self-audit lineage because it counted score-only and context prediction streams
together. In v2_r1, all 48 coverage checks reconcile exactly and every manifest
and output hash validates.

The learner uses one statistical row per available UTC hour, not one repeated
row per candidate. It separately estimates all-hour opportunity and conditional
selected count, mean/sum net EV, failure rate and downside severity, then
broadcasts one hourly scalar correction to candidate scores. This preserves
within-hour ordering, side semantics and one pooled global top-10 selection
across timestamps. Regime, transition and combined context begin in 2024:
their 2023 candidates remain in replay with exactly zero adjustment and are
excluded from the eight-era evidence gate. Trajectory has nine supported eras.

Regime gamma 0.10 is the only encouraging diagnostic: median era net-EV delta
is +0.000347 (+3.47 bps), 75% of eras improve, and median long/short deltas are
+2.95/+0.94 bps. It still fails because median weekly-Q10 delta is -3.57 bps.
The absolute residual control is negative in every supported era (aggregate
about -85 bps), so no relative correction can be promoted even where it
improves a negative baseline. Transition, trajectory and combined corrections
are weaker. All 12 arm/gamma candidates are ineligible; no 2026 input was read.
Training and assessment are strictly 1h, with 1m restricted to nested labels
and execution economics.

### Final supervised topology taxonomy audit (sealed 30 July — stop condition)

The final bounded alternative is
`supervised_coarse_topology_taxonomy_audit_20260730_v1`.  It does **not** fit a
classifier unless state identities are semantically admissible and each
ex-post target has independent pre-2026 support.  Candidate targets are
baseline-to-nonbaseline onset, nonbaseline-to-baseline normalization/reversal,
nonbaseline-to-different-nonbaseline rotation, and an abstain/other bucket.
Destination topology is a target/reporting field only; it never enters causal
trajectory or current/source-state features.  The audit uses no trading
outcome or 1m model row.

State 0 is numerically modal (86.2--93.8% of hours in every 2022--25 era),
but it is not a stable semantic baseline: its zero-versus-nonzero contrast has
pairwise correlations from -0.155 to 0.988, while every individual state
profile fails the minimum alignment criterion (state 0 minimum -1.000; states
1--4 minima -0.306 to -0.458).  More fundamentally, event-ledger source-state
identity has **0.0%** exact correspondence to the hourly pooled-state identity
at the same anchor, so source/destination topology cannot be safely combined
with the hourly causal state representation.  Rotation also has just one
training event in one era, versus the required 12 events across three eras;
onset (79/4 eras) and normalization (61/4) alone do pass support.

The preconditions therefore fail before supervised fitting.  No multiclass
balanced accuracy, macro AUC/AP, Brier, calibration or confusion result is
reported because creating one would require invalid labels or collapsing the
unsupported rotation category.  This seals the stop condition for subtype
work: preserve the separate hourly binary trajectory transition-versus-stable
context, but do not pursue transition subtype classifiers until state lineage
is reconciled and independent rotation support exists.

### Stability-constrained coarse subtype retry (sealed 30 July)

`constrained_coarse_transition_taxonomy_20260730_v3` directly tested the
remaining subtype question without another unconstrained clustering sweep.
Discovery uses only the 141 2022--25 transition anchors; 2026's 16 events are
untouched transfer.  The fixed representation is four semantic families across
the causal 168/24/6/3h phase windows.  It uses train-only median imputation,
5--95% winsorisation, robust scaling with a fixed sparse-feature scale floor,
and a train 97.5%-distance **unclassified** bucket.  K is restricted to 2 or
3; any core type must have at least 12 events, three eras, bootstrap ARI
mean/q05 >= 0.60/0.35 and Hungarian leave-era minimum semantic cosine >= 0.50.
Source/destination topology is reported after labels but does not enter the
causal profile.

The retry is stronger negative evidence, not a failed implementation.  Both
candidates now have adequate core support after four explicit outliers
(K=2: minimum 20 events/3 eras; K=3: 16/3).  Yet K=2 reaches only bootstrap
ARI 0.505 with q05 -0.069 and leave-era minimum cosine 0.029; K=3 is
0.434/-0.034/-0.055.  The 2025-excluded folds are the key failure: their
matched type geometry does not align with 2022--24.  No type name, cluster ID,
sidecar field or routing rule is emitted.  Binary transition-vs-stable metrics
remain separate and unchanged.  The status of `recurring_transition_clusters`
cannot honestly change: stable coarse types still do not exist.

### Sealed trajectory missingness-aware identical-row result (30 July)

The pre-registered five-arm evaluation is now sealed at
`trajectory_missingness_identical_row_ablation_20260730_v1`.  It retains all
127,777 frozen-2026 candidate rows at the **1h decision cadence** (1m remains
labels/replay-only); 85,870 rows have trajectory context and 41,907 receive
only the pre-registered neutral 0.5/log(2)/0 fill plus availability flag.  The
historical side-local GAM fit uses 381,814 rows and a pre-2026 OOF monotone
map; no 2026 fitting, tuning, row dropping, type IDs or 1m model rows occurred.

None of the four trajectory arms clears the baseline's global pooled top-10
economics.  Baseline existing-transition is -69.99 bps (rank IC 0.0664);
trajectory availability only is -79.22 bps (0.0727), existing+trajectory
-75.43 bps (0.0756), regime+trajectory -74.77 bps (0.0788), and the complete
regime+existing-transition+trajectory stack -70.75 bps (0.0832).  The latter
therefore raises execution rank IC by 0.0168 while worsening aggregate EV by
0.76 bps.  It also worsens the latest (July) month from -121.95 to -170.80
bps, despite improving June from -50.10 to -37.98 bps.  That is not a usable
economic gain.

The apparent long-side benefit does not transfer across the book: the complete
stack changes long from -80.19 to -49.61 bps but short from -62.38 to -74.94
bps.  It replaces 54.12% of the frozen global book, strongly shifts selection
toward the availability-missing rows (available selected share 75.07% to
29.72%), and leaves both availability strata net-negative.  Keep trajectory
probability as diagnostic context only.  Do not promote it, replay it, or add
type clusters; any later attempt needs a separately pre-registered mechanism
that explains this selection shift and passes both-side/latest/tail economics.

### March-20 regime boundary versus economic trust

`marapr2025_direct_residual_regime_trust_diagnostic_20260730_v1` now binds all
136,074 March/April identical direct/residual candidates to the authoritative
causal regime, transition and trajectory fields.  Strict pre-March OOF
provenance is verified on disk.  Direct-only selected rows change from
+12.85 bps before March 20 to -100.63 bps afterward and -107.96 bps in
April; selection overlap with residual is only 6.4--10.0%.

`marapr2025_direct_residual_regime_break_learnability_20260730_v1` proves that
the market boundary and economic routing are different targets.  The
trajectory-only fields identify the boundary at grouped-OOF AUC 0.700, with
all day-group folds above 0.676.  Yet best side-local direct-over-residual
trust AUC is only 0.536 long and 0.532 short, and every predicted short-trust
top decile remains negative.  Do not convert trajectory probability into a
gate.  Keep residual as incumbent; reconstruct older identical-row OOF
direct/residual scores and inference-parity causal market mechanics before
training conditional capture/adverse-loss and bounded trust heads.

The next compatible current ledger is now precisely scoped.  May--July 2026
has 125,551 common rows with a true direct-q25 score and causal monthly fold
cutoffs.  Its current residual is OOF but targets legacy 24-hour economics,
not the exact H12 policy endpoint.  Freeze q25, rebuild only side-local H12
residual OOF on all 127,777 raw-score identities, then attach causal context
at signal time.  Use the already frozen neutral trajectory fill plus
availability; do not drop missing rows.  This remains a reused-period
diagnostic and cannot replace a later untouched confirmation cohort.

### Exact-H12 completion and July interpretation

The residual mismatch has been removed in
`exact_h12_side_local_residual_oof_20260730_v2`.  The 127,777-row ledger is
strictly prior-resolved and deterministic, and its long/short pair is selected
jointly on the pooled-global book.  Long uses residual blend 0.75; short uses
0.0.  The resulting May/June/July top-10 is
-67.88/-104.26/-148.98 bps.  The bounded selection procedure is diagnostic,
not the repository's complete production feature-selection stack, and no
deployment or replay is authorized.

`exact_h12_residual_regime_transfer_diagnostic_20260730_v1` shows why the
next task is mechanism learning rather than more regime classification.
Combined frozen context recognises July at grouped-day OOF AUC 0.792.
Nevertheless, residual-over-base trust is 0.435/0.521 AUC and
direct-over-residual trust is 0.476/0.577 long/short.  July's strongest
signature is lower trajectory availability plus changed transition onset,
run length, margin and model entropy.  This is a reliable state warning, not
a reliable choice of trading score.

The policy-parity check in
`exact_h12_residual_recent_ev_mapping_20260730_v1` uses a causal 21-day
label-resolved map followed by pooled-global top-k.  Exact-residual economics
fall to -72.82/-108.87/-180.00 bps and July becomes 99.54% short.  Do not
interpret calibration as model repair.

The IC/EV puzzle is also resolved.  The rising February--April headline IC
uses the native 24-hour alpha target; exact-H12 net long IC is
0.090/0.093/0.143.  Raw-base gross is +49.38/+17.05/+41.86 bps while costs
are about 100 bps.  Alpha ordering and tradable execution ranking are
therefore distinct: April recovers some gross ordering, but no month clears
cost.  Future regime work should support separate cost-clearance,
favourable-capture and adverse-loss heads, with older/untouched confirmation,
rather than directly gating on a transition probability.

### Model-derived mechanics representation: first cross-fitted result (30 July)

`pre2026_model_derived_mechanics_representation_20260730_v1` is the first
answer to the question “can the trained models derive the market-mechanics
composites?” It learns three side-local probabilities from 15 causal
primitives common to two strict-OOF score vintages: executable gross
opportunity above cost +25 bps, net-positive conversion, and net loss of at
least 100 bps. The output is an OOF probability sidecar only; current regime
and transition remain separate layers, and no output is a live gate, policy
input, replay or 2026 application.

The exactly matched foundation is 214,160 hourly candidate rows: 103,430 from
Apr--Dec 2023 and 110,730 from Mar--Apr 2025. The other 30k 2025 mechanics
rows are intentionally excluded because they lack a compatible residual score.
The 15 fields are finite and identity-checked by candidate, timestamp, symbol
and side. No compatible candidate-level mechanics panel yet exists for 2024
or May--Jun 2025. Outer evaluation is side-local leave-calendar-month-out
**OOF**, explicitly not walk-forward OOS; all decision/fit/assessment rows
are 1h and 1m remains embedded in labels.

Within each held month/side/head, CatBoost discovers the top eight positive
mechanics fields on the other months and refits a fixed geometry. Relative to
the identical score-only head, discovered mechanics raises median AUC by
+0.00755 for opportunity and +0.00497 for severe-loss risk, but changes
conversion AUC by -0.00098. Only 59.1% of matched month-side cells improve
for opportunity/downside and 45.5% for conversion; calibration worsens on
median for every head. Repeated selections—compression/fragmentation,
correlation heterogeneity/breakdown, negative breadth, flush recovery and
deleveraging-without-follow-through—are candidates for *model-derived*
mechanics structure, not proven deployable composite features.

The predeclared diagnostic blend (residual rank + .25 opportunity rank + .25
conversion rank − .25 downside rank) is economically rejected. Against the
same residual global pooled top-10 control it loses -3.64 bps on average and
-2.13 bps at the median month, improves only 3/11 months, and has a worst
month delta of -15.78 bps. Severe-loss rate falls by 89 bps on average, but
the short contribution declines by -14.42 bps and long is nearly flat (+0.09
bps). The downside head may remove bad tails, but this fixed combination gives
up more upside; neither the blend nor individual probabilities may be
promoted.

Next: materialise compatible 2024 and May--Jun 2025 candidate mechanics,
then pre-register a *nested* combiner with purged inner weight/calibration
selection. Test downside-only, opportunity-only, conversion-only and bounded
joint overlays against the matched score-only control, requiring both-side and
weekly/monthly-tail gains before a frozen later-period confirmation. Tree-SHAP
interaction discovery and regime-conditional permutation importance remain
training-fold diagnostic tools only; they cannot manufacture a raw feature or
override OOF/economic gates.
