# Actionable Regime and Transition Roadmap

Updated: 2026-07-30

This is the sole actionable roadmap for the regime/transition workstream.

## Non-negotiable contracts

1. Train/select/tune/calibrate on 2022–2025; assess on untouched 2026.
2. Regime and transition are separate layers and require separate ablations.
3. Ex-post phase, future state, post-entry path and outcomes are labels or
   attribution fields only.
4. Soft outputs must include uncertainty and identity-continuity provenance.
5. Economic selection is one pooled global top-k after arm-local causal EV
   mapping, never per timestamp, side, regime or transition.
6. No portfolio replay before aggregate, latest, worst-period, both-side,
   calibration, tie and concentration gates pass.
7. Training, validation/OOF, HPO, feature selection, mapping and assessment
   samples are hourly (`1h`).  Multi-timeframe values, including native `15m`
   where available, are causal lookbacks sampled onto that hourly decision row
   and never create sub-hourly examples.  Minute bars (`1m`) are permitted only
   inside exact barrier/path/fill/policy replay; each replay returns one outcome
   to its originating hourly candidate and never becomes extra model or
   assessment rows.
8. Cadence is fail-closed.  Every new artifact must declare `1h` model and
   assessment cadence, report duplicate/non-hourly decision-row counts, and
   prove a one-to-one hourly-candidate-to-replay-outcome join.  `1m` may improve
   the physical accuracy of fills, exits and path-dependent labels, but may
   never be used as independent training, OOF, calibration, mapping, HPO or
   assessment observations.

## Stage A — finish exact data and labels

- [x] Freeze the full-2024 request: 190,398 candidates / 141 symbols.
- [x] Verify four partitions read-only: 43,660,200 / 43,660,200 required
  minutes; zero incomplete and zero failed symbols.
- [x] Seal the aggregate verification artifact and bind all four manifests:
  `failure_2024_exact1m_download_verify_20260730_v1`.
- [x] Regenerate causal label inputs from the frozen request/product map:
  `failure_2024_exact1m_label_inputs_20260730_v2` (190,398 hourly rows).
- [x] Replay the deployed candidate-local exit policy with explicit costs and
  current frozen-spread counterfactual provenance.
- [x] Regenerate hourly entry-timing candidates and their nested 720-minute
  physical paths with zero missing candidates.
- [x] Bind candidate coverage to both the 141-symbol full-universe verifier
  and the four-partition aggregate seal: 190,398 / 190,398 complete.
- [x] Regenerate physical path, soft triple-barrier and auxiliary labels:
  `failure_2024_exact1m_multitask_labels_20260730_v2` contains 190,398 hourly
  rows and 24 month/side support cells.
- [x] Prove exact identity preservation, decision-before-path horizon,
  conservative same-minute adverse-first handling, gross-cost-net parity and
  label-availability timestamps.  The immutable v1↔v2 audit passes all
  190,398 rows, including decoded 720-minute timestamp/OHLC paths at zero
  tolerance and all raw artifact hashes.
- [x] Seal the cadence audit proving `1h` model rows and nested `1m` replay
  paths: `regime_transition_hourly_cadence_audit_20260730_v2`.  V2 also
  verifies the final-v3 OOF/forward ledgers, mapping suite and GAM mixture.
- [x] Recheck the cadence contract after the trajectory, H2 and model-failure
  extensions.  Their contracts remain `1h` for decision/model/assessment rows
  and `1m_labels_only` for nested replay; the focused enforcement suite passes
  5/5 tests.  Apply the same fail-closed audit to every subsequent artifact.

## Stage B — regenerate the evidence plane

- [x] Extend the reporting infrastructure to emit week/month Q10/Q50 and
  positive-period shares; side × state × transition × period attribution
  with availability; and composition-versus-within-category gross/cost/net
  decomposition.  All attribution occurs after pooled global top-10
  selection.
- [x] Rebuild the 2022–2026 weekly/monthly base+residual calendar:
  628,471 hourly candidate rows, 168 complete weeks and 39 complete months,
  with early-2022 inverse-contract, later frozen-PF and strict OOF evidence
  grades kept separate.
- [x] Report percentage/count of meaningfully positive-IC and positive-EV
  weeks/months, with uncertainty and sample support.
- [x] Rebuild the worst-period calendar with feature distribution,
  covariance and interaction deviations versus regular periods.
- [x] Separate composition shifts from within-cell payoff shifts.
- [x] Emit long/short, state and transition attribution after pooled global
  top-10 selection, with unavailable context explicit.
- [x] Extend the attribution output with asset and exit-reason breakdowns in
  `stack_regime_failure_analysis_2022_2026_20260730_v5`.  Attribution follows
  the same pooled-global top-10 book, reports weekly/monthly selected support,
  gross/cost/net, net Q10/Q50, largest-asset share and HHI, and keeps exit
  reason unavailable rather than fabricating it for reconstructed rows.  Its
  asset shares are denominated against the whole pooled selected book, never
  an evidence-grade-local subset; worst-versus-regular decomposition remains
  lineage/evidence-grade separated.
- [x] Diagnose the alpha-IC/execution-EV divergence directly.  Current
  evidence has meaningful positive alpha IC in 127/168 weeks and 32/39
  months, but meaningful positive net EV in only 1/168 weeks and 0/39
  months.  Decompose target mismatch, rank-to-policy conversion, cost,
  side/book concentration and lineage effects without pooling evidence
  grades.  The sealed lineage-local audit shows the first-touch target itself
  remains strongly related to net EV (IC 0.603 in Mar–Apr 2025 and 0.566 in
  May–Jul 2026), but the base score converts weakly to that target and faces
  an approximately 100 bps cost hurdle; July additionally reverses
  alpha-to-net ranking.
- [x] Seal the A-grade cost-clearing conversion checkpoint with resumable
  14-day chronological OOF folds.  Artifact
  `a_grade_cost_clearing_conversion_ablation_20260730_v6` uses 1-hour
  candidates (1-minute data nested only in existing 12-hour labels), requires
  all prior sealed folds before a later fold, and records a 9.84-second
  slowest fold.  The frozen 2025→2026 common-contract verdict is negative:
  baseline -96.16 bps and alpha-hurdle -106.21 bps mean monthly net EV.  Both
  use 2025-only fit and 80,682 blocked-OOF calibration rows; 2026 labels are
  excluded from fit/map.  Regime/transition context arms are deliberately
  fail-closed because their sidecar semantics differ across eras.

## Stage C — strict regime benchmark

Run all methods on the same causal feature panel and identical untouched 2026
rows:

- [x] Diagonal GMM baseline — valid split, rejected for two-hour dwell and
  excessive switching.
- [x] Sticky/full-covariance GMM challenger — sealed at
  `strict_forward_sticky_fullcov_regime_challenger_2022aug_2025_to_2026_20260730_v1`.
  It is train-only selected (32 family-balanced fields; full covariance;
  `k=3..6`; blocked sticky sweep) and assesses the same 1-hour 2026 rows.
  It improves 2026 predictive score (-0.785 vs -1.058 diagonal) but is
  rejected by the persistent-state gate: 2h median dwell and 29.86% hourly
  switching versus required >=6h and <=10%.  It is diagnostic-only and must
  not be added to the stack or confused with the transition layer.
- [x] DAE→GMM with frozen representation, reconstruction error and OOD —
  sealed at `strict_forward_dae_gmm_regime_challenger_2022aug_2025_to_2026_20260730_v1`.
  It is a train-only neural denoising-AE bottleneck sweep (4/8/12), followed
  by full-covariance GMM `k=3..6` and blocked sticky selection.  It emits
  model-local identity, posterior, entropy/margin, density OOD and
  reconstruction OOD, plus a three-arm stability and side/economic attribution
  comparison.  Selected 4/3/sticky-200 yields -0.828 2026 log score, 2h
  median dwell and 32.13% hourly switching: better likelihood than diagonal,
  but worse than sticky/full-covariance and rejected by the same persistent
  state gate.  Do not promote or use for allocation.
- [x] Extend the economic validation to the available all-era strict contract:
  `unsupervised_economic_all_era_strict_20260730_v1` fits representation on
  2022-08→2025 hourly rows, conversion/map on 449,814 pre-2026 candidates and
  377,928 chronological OOF rows per arm, then assesses 114,096 identical
  2026 candidates with no 2026 label in fit/map.  It uses a single pooled
  global top-10 after arm-local frozen historical EV maps; period tables do not
  rerank.  Baseline is -74.97 bps, sticky geometry -102.60 bps and DAE geometry
  -99.22 bps.  Both additions fail the economic gate.  Diagonal and
  failure-first are explicit fail-closed availability rows, not approximated
  comparisons: respectively no serialized historical transform and no
  semantically identical pre-2026 joint score/overlay cohort.
- [ ] Sticky HMM and duration-aware semi-Markov challenger.
- [x] BOCPD changepoint-derived causal change/run-length context is sealed in
  `strict_bocpd_regime_transition_challenger_20260730_v2` and emitted through
  the authoritative sidecars below.  Its logistic heads have convergence and
  calibration limitations, so they remain diagnostic-only provenance/context,
  never a gate, quota, standalone score or promoted regime representation.
- [ ] HDBSCAN only if dependency and a materialized strict run exist.

Every arm must cover multiple horizons and quantify shifts in distributions,
correlations/covariances, volatility, liquidity proxies, causal pre-entry
geometry proxies and long-versus-short model performance.

## Stage D — strict transition benchmark

- [x] LGBM 1/3/6/12h onset and competing-risk baseline.
- [x] Strict BRL stable-versus-transition and 1/3/6/12h onset challenger:
  `strict_transition_brl_challenger_20260730_v1`.  It has frozen readable
  native Beta-Binomial MAP rules, blocked-2022--2025 HPO/calibration, one
  untouched-2026 assessment, and monthly plus long/short global-top-10
  attribution.  It is rejected: 2026 AUC 0.519/0.513/0.500/0.495/0.491 for
  active/1h/3h/6h/12h; no BRL output may be a gate or policy input.
- [x] Train-only leave-era-out semantic-prototype alignment: each held era now
  uses a GMM and semantic reference both fit on the remaining eras, with
  Hungarian component matching, mapping confidence, train posterior
  correlation, OOD and abstention.  The result is negative and sealed in
  `leave_one_era_out_transition_morphology_alignment_20260730_v1`: none of
  the three slots has stable cross-fold prototypes or enough high-confidence
  held-era recurrence.  It must not become a global type or a policy feature.
- [ ] Distinguish recurring transition types from each other and from matched
  stable controls.  The strongest support-expansion alternative is sealed but
  negative: `hourly_transition_semantic_signature_ablation_20260730_v1`
  labels 1--3h-ahead onset on hourly rows while retaining each physical
  `next_event_id` (423 positive windows / 141 distinct 2022--25 events; 48 /
  16 in 2026).  Fixed causal breadth, washout/reversal, funding and combined
  semantic signatures are fit only through 2025 and assessed in 2026.  Their
  2026 AUCs are 0.436, 0.526, 0.436 and 0.467; every UTC-week bootstrap 95%
  interval crosses 0.5, and no leave-era-out coefficient group meets the 0.70
  minimum pairwise-stability criterion (best washout/reversal minimum 0.238).
  Thus hourly support does not stabilise a reusable semantic type.  It is
  diagnostic-only and the requirement remains incomplete.
- [x] Materialise the stronger **binary transition-versus-stable** trajectory
  representation at all compatible hourly timestamps.  Authority:
  `trainonly_recurring_transition_prototype_study_20260730_v3` and
  `hourly_trajectory_transition_soft_sidecar_20260730_v1`.  Its fixed causal
  168/24/6/3h trajectory contract transfers at 2026 AUC/AP 0.745/0.721 on the
  full anchor assessment and 0.849/0.832 on the complete-lookback subset.  The
  sidecar provides 29,060 calendar-era-held OOF pre-2026 and 3,927 frozen 2026
  hourly probabilities plus availability, entropy/margin and fit provenance;
  920 rows explicitly fail closed.  It contains no unstable component or
  destination identity and is a diagnostic stack-context candidate only.
- [ ] Recurring transition **types** remain unresolved.  The v3 prototype
  study tested current/source and destination topology separately, phase/horizon
  paths jointly, train-only bootstrap recurrence, leave-era identity alignment
  and untouched-2026 transfer.  No K=2--5 cluster clears support/stability;
  K=4 has a 131/141 dominant local component and a -0.200 minimum matched
  leave-era centroid cosine.  Do not force labels or promote IDs.  Reopen only
  with more independent events or a preregistered representation that clears
  recurrence, identity alignment and transfer simultaneously.
- [x] Run a stability-constrained coarse subtype retry with explicit abstention:
  `constrained_coarse_transition_taxonomy_20260730_v3`.  It restricts
  train-only discovery to K=2/3, fixed causal phase/horizon semantic profiles,
  5--95% winsorisation, robust sparse-feature scaling, and a 97.5%-distance
  unclassified bucket; it requires support, bootstrap ARI and Hungarian
  leave-era semantic identity before naming a type.  Four outliers are
  explicitly unclassified and core support is sufficient, but K=2/K=3 fail
  recurrence (ARI mean/q05 0.505/-0.069 and 0.434/-0.034) and leave-era minima
  (0.029/-0.055).  This is stronger negative evidence: do not reopen types
  without additional independent events or a new preregistered mechanism.
- [x] Run the final supervised topology-taxonomy admissibility audit:
  `supervised_coarse_topology_taxonomy_audit_20260730_v1`.  It verifies state
  0/nonzero and source/destination state semantics before fitting labels or a
  predictor.  State 0 is modal but not semantically stable; zero/nonzero
  contrasts and every individual state profile fail leave-era consistency, and
  event source state has 0% exact identity agreement with the hourly pooled
  state at its anchor.  Rotation has only one pre-2026 event in one era
  (required: 12/3); onset and normalization support alone cannot justify a
  subtype taxonomy.  No classifier was trained, no labels were collapsed, and
  subtype work is now stopped.  Retain only the separate binary trajectory
  transition-versus-stable soft context until state lineage and support change.
- [ ] Preserve phase decomposition for labels/attribution; learn only causal
  phase-age/onset proxies.
- [ ] Freeze morphology prototypes, component order, bootstrap stability,
  alignment confidence, uncertainty and OOD/abstention on 2022–2025.

## Stage E — soft OOF outputs

- [x] Seal `authoritative_soft_regime_transition_sidecars_20260730_v1`:
  separate 2022–2025 blocked-OOF plus untouched-2026-forward *hourly* regime
  and transition sidecars (33,895 rows each).  Warm-up rows are retained as
  unavailable; only causal BOCPD change/run-length/uncertainty and frozen
  strict-LGBM plus BOCPD short-horizon probabilities are present.  Rejected
  diagonal/sticky/DAE identities and unstable morphology IDs are excluded.
- [x] Emit 1/3/6/12h BOCPD transition probabilities alongside strict-LGBM
  transition probability, entropy/margin, availability, OOD availability,
  train-end provenance and source reliability.  BOCPD calibration is poor
  (its 2026 ECE is recorded in `bocpd_reliability.csv`), so these fields are
  diagnostic-only and cannot become a gate or a standalone score.
- [x] Prove cadence and fit-label resolution: the sealed audit has zero
  duplicate/non-hourly timestamps; multi-timeframe values remain lookbacks on
  `1h` rows, and `1m` remains nested replay only.  All 17,532 historical OOF
  rows have fit labels resolved strictly before their fold end.

## Stage F — stack and calibrator ablations

Category stability availability is sealed at
`heldout_regime_category_economics_stability_20260730_v2_availability`:
no gate is permitted. The compatible exact cohort has at most two eras versus
the required three and only 50,220 context-attributed rows; materialize a
common identity/score/economics/context ledger before retrying.

- [x] Seal final-v3 interaction diagnostics in
  `final_v3_context_interaction_diagnostics_20260730_v2`: 521,570 coalesced,
  candidate-held pre-2026 OOF rows discover fixed tree-SHAP interactions and
  regime-/transition-conditional permutation importance; 127,777 2026 rows
  are assessment-only.  Residual×context importance is stable pre-2026 on both
  sides, whereas base×context effects are period-specific.  Follow-on arms may
  test residual interactions only, with fixed pre-registered features and no
  BOCPD promotion.  The earlier v2 diagnostic is invalid diagnostic-only
  because v2 had complementary empty era-specific score/label columns.

- [x] Seal the fail-closed final identical-row stack/GAM runner contract.  It
  uses only the authoritative sealed timestamp-level 2022–2025 OOF /
  2026-forward hourly sidecar pair and rejects provisional or unavailable
  context.  It
  checksum-binds separate `soft_regime_hourly.parquet` and
  `soft_transition_hourly.parquet`, requires both model and assessment cadence
  to be `1h`, and joins many candidate score-ledger rows to one `source_utc`
  context row without changing candidate identity.  Missing context, warm-up
  rows, non-hourly sidecar timestamps, bad checksums or provenance fail closed;
  no candidate subset may be substituted.  It tests baseline plus regime-only,
  transition-only and combined continuous context in base, residual/trust and
  bounded additive-GAM placements; state IDs/raw posterior axes, GMM/morphology
  fields and unavailable OOD fields are forbidden.  All arms are side-local and
  use frozen pre-2026 arm-local EV maps before one pooled global top-10
  selection.  The pinned score ledgers are 521,570 blocked-OOF pre-2026 rows
  and 127,777 exact-replay May–July 10 2026 rows; their lineage is verified
  before the join.  One-minute bars are labels/replay evidence only.
- [x] Run and seal corrected
  `final_identical_row_regime_stack_gam_ablation_20260730_v3` on one common
  127,777-row May–10 July 2026 hourly universe.  Historical context coverage
  is 418,140 / 521,570 candidates; 103,430 explicit OOF warm-up rows are
  excluded from every arm, never imputed.  V1 and v2 are invalid diagnostic
  evidence: v1 broke map direction/tie handling; v2 failed to coalesce the
  complementary 2023–2024 versus 2025 score and label-resolution column
  names.  V3 conflict-checks/coalesces those aliases, has 521,570 resolved
  historical label endpoints/base/residual scores, and uses five OOF folds
  for residual/GAM arms.
- [x] Record the negative verdict: baseline top-10 net EV is -77.51 bps;
  GAM+regime is best aggregate at -57.92 bps but worsens July to -134.68 bps;
  GAM+combined is -59.28 bps, raises execution IC to 0.094, improves both
  sides and July to -105.32 bps; GAM+transition is the most balanced through
  time at -70.77 bps aggregate, -72.22 bps July and -52.44 bps latest week,
  with improved week/month Q10.  Every arm remains negative net of the
  approximately 100 bps cost burden.  No portfolio replay is authorized.
- [x] Fixed pre-2026 convex mixture of GAM experts:
  `final_v3_gam_convex_mixture_ablation_20260730_v1` tests all 15 simplex
  weights over regime-only, transition-only and combined raw GAM scores.  No
  OOF gate passes aggregate, weekly/monthly Q10/Q50 and both-side economics;
  no gate/blender was fit.  The diagnostic best 25% regime / 75% transition
  blend remains negative at -64.86 bps historical OOF aggregate with -203.70
  bps weekly Q10.  The frozen 2026 application is non-promotable.
- [x] Supersede the stale H2 readiness result with
  `h2_2025_identical_row_oof_bridge_audit_20260730_v7`.  July through
  December are now sealed common-30 score/economics bridges.  December has
  44,640 base/residual rows and 43,920 exact common-context rows; its final
  12 context timestamps remain explicitly unavailable and never filled. The
  raw input rows are present, but their score-only frozen reconstruction fails
  canonical overlap reproduction in transition fields; `dec2025_final12h_frozen_predec_regime_transition_context_extension_20260730_v1`
  withholds them. The H2
  scope remains non-promotional because it is not
  population-identical to the wider v3 ledger.  July is sealed:
  44,640 unique common-30 hourly candidates (22,320 per side), both canonical
  base/residual score fields, strict blocked-OOF score-fit provenance and
  labels through 2025-08-01 12:00Z.  This reduces the last-label map age by
  31.54 days, but is not an identical-population extension of the wider v3
  ledger.  August--November is now also sealed as a 175,680-row frozen-July
  OOS bridge.  December is sealed as a frozen-pre-August OOS sensitivity:
  residual improves base from -117.55 to -84.69 bps but remains negative, and
  every fixed regime/transition context arm is worse on the common subset.
- [x] Run the strongest valid baseline-only causal map sensitivity:
  `july_common30_baseline_map_refresh_20260730_v1`.  Fit only ordinary and
  rank-preserving isotonic maps on frozen pre-2026 baseline OOF, with/without
  the sealed July common-30 OOF; assess the same 127,777 frozen 2026 hourly
  candidates using one pooled global top-10.  No 2026 fit, tuning or selection
  occurs.  The refreshed map changes neither the 12,778 selected candidates
  nor -77.51 bps aggregate top-10 EV; long/short remain -87.24/-61.85 bps.
  Rank preservation reduces mapped tie mass from about 13.0% to 0.015%, but
  does not change economics.  No map/context/policy promotion is authorized.
- [x] Run the separately scoped July common-30 **raw-context** extension:
  `july2025_common30_regime_context_raw_score_extension_20260730_v1`.  It is
  a fixed seven-arm, side-local diagnostic: frozen residual raw-score control,
  then residual-LGBM and bounded spline-GAM with regime-only,
  transition-only and combined context.  All 418,140 context-available
  historical rows used to fit these arms resolve strictly before 2025-07-01;
  the 44,640 July candidates are never fitted, mapped or used for HPO/feature
  selection.  On the one pooled July raw top-10, GAM+regime is best at
  -88.40 bps with +0.0226 execution rank IC, versus -101.75 bps/-0.0391 for
  the frozen residual raw-score control.  It is still negative on both sides
  (-90.75 long, -87.49 short); every LGBM arm and GAM transition/combined arm
  is worse.  This is evidence for a low-capacity regime effect in this July
  cohort, not an EV-map or policy promotion.
- [x] Run the corresponding all-compatible-arm July map sensitivity:
  `july2025_common30_all_context_map_refresh_20260730_v1`.  For baseline plus
  all three residual and all three GAM contexts, it compares each frozen v3
  OOF isotonic map with the same map after appending the 44,640 sealed July
  raw OOF rows (ordinary and rank-preserving variants), then assesses the
  unchanged 127,777-row hourly 2026 universe under one pooled global top-10.
  Every mapping fit is pre-2026 (381,814 old versus 426,454 refreshed rows);
  no 2026 outcome tunes anything.  All seven arms retain **exactly identical
  top-10 membership and economics** because a monotone map plus raw-score
  tie-break preserves rank.  GAM+regime remains -57.92 bps aggregate,
  -134.68 bps latest July, weekly Q10 -85.30 bps and monthly Q10 -121.02 bps;
  long/short are -72.16/-51.38 bps.  The July raw evidence therefore cannot
  repair the stale-map issue through monotone recalibration alone.
- [x] Seal the v3-bound pre-2026 mapping-resolution suite:
  `pre2026_mapping_resolution_ablations_20260730_v2`.  The prior similarly
  named v1 is invalid/non-authoritative because it used the superseded v2
  stack.  Seven preregistered maps use only pre-2026 OOF scores/labels; 2026 is
  assessment-only.  Rank-preserving isotonic resolves mean tie mass from
  24.93% to 0.017% and passes all 10 diagnostic resolution gates, but changes
  mean top-10 EV only from -72.15 to -72.14 bps.  Binned/support-shrunk maps
  change side ordering but worsen mean aggregate/latest/Q10 economics despite
  isolated arm gains.  No mapping passes net economics or authorizes replay.
- [x] Re-verify cadence for this suite: model fitting, OOF, mapping selection
  and 2026 assessment use exact `1h` decision rows.  V3 OOF, forward and the
  894,460 selected mapping rows have zero non-hourly timestamps and zero
  invalid label endpoints.  `1m` remains nested barrier/path/fill replay only;
  no cadence violation was found.
- [x] Diagnose the complementary aggregate-versus-July effects with
  pre-registered regime × feature and transition × feature interactions,
  subsampled SHAP interaction discovery and regime-conditional permutation
  importance.  The authoritative artifact is
  `final_v3_context_interaction_diagnostics_20260730_v2`; residual-context
  importance is stable in all 24 pre-2026 side-period checks, while
  base-context effects are period-specific.  In 2026, long validates separate
  regime/transition/combined context and short validates transition/combined
  but not regime.  Leading tree-SHAP terms pair residual score with BOCPD
  state-age/run-length fields; BOCPD remains context only, never a standalone
  score or gate.
- [x] Run the fixed low-capacity residual-context follow-on justified by that
  diagnostic: long regime/transition/combined and short
  transition/combined, explicitly excluding short regime.  Use chronological
  pre-2026 OOF maps, one pooled global top-10, identical 1h forward rows and no
  2026 tuning.  `final_v3_preregistered_residual_interactions_20260730_v1`
  uses only explicit residual × selected-context products in a fixed
  StandardScaler+Ridge(alpha=80) learner.  It fails: long regime/transition/
  combined reach -77.39/-77.51/-77.30 bps and short transition/combined
  -99.23/-99.34 bps, versus -77.51 bps baseline top-10 net EV.  No follow-on
  arm is promotable.
- [x] Re-audit follow-on cadence.  Its sealed `row_cadence_audit.csv` records
  zero non-hourly and zero duplicate-candidate rows in 521,570 historical
  score-ledger rows, 418,140 context-available OOF rows and the exact 127,777
  forward assessment rows.  `1m` remains nested exact-12h label/path/replay
  evidence only; it does not form a fit, OOF, mapping or assessment row.
- [x] Materialise and seal the July 2025 **common-30 only** compatible
  blocked-OOF bridge at
  `july2025_common30_final_base_residual_oof_bridge_20260730_v1`.  Required
  score pair: `score_base_alpha`, `score_residual_expected_ev`; both sides;
  exact hourly candidate identity; native decision+24h label resolution
  strictly before 2025-07-01 for both fits; no 2026 outcomes.  Preserve exact
  execution label end/availability provenance but keep 1m nested in the
  12-hour replay only.  Use frozen 31/8 base and accepted residual contracts,
  zero new FS/HPO, and record score/source/identity hashes.  The runner is
  intentionally resumable: reuse only exact-column/row-count June frozen PIT
  matrices (82.789% long, 82.690% short overlap) and materialise the remaining
  17,211/17,310 rows through strict PIT.  Retain the accepted constructor's
  frozen worker setting; do not inject a new threading parameter.
- [x] After July is sealed, conduct a **separate, non-promotional** bridge
  readiness review.  It preserves common-30 scope and does not substitute it
  for the wider v3 ledger.  Both the baseline-only and all-context map
  sensitivities are sealed and non-promotional.
- [x] Challenge that August--November conclusion by materialising
  `augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1` in
  five resumable stages.  The sealed full-population score-only PIT preflight,
  `augnov2025_pit_scoring_preflight_20260730_v2`, is positive: all 175,680
  exact 1h common-30 identities (44,640 rows in August and October; 43,200 in
  September and November) join one-to-one to native-base and exact-execution
  identities, and every required frozen input is exact/finite in each
  month/side/symbol cell (31-long/8-short base; 69 residual per side).  It
  reads no target values.  Fit both frozen base and residual layers
  only on native labels resolved before 2025-08-01; use future exact execution
  labels only for OOS candidate identity/economics.  Seal all 175,680 rows or
  emit a precise row-level blocker report.  No HPO, no future native labels,
  no 2026 use and no promotion.
- [x] Diagnose the frozen-July residual's August/September harm versus
  October/November help using the sealed bridge and authoritative hourly
  sidecars.  Authority:
  `augnov2025_frozen_july_residual_regime_diagnosis_20260730_v2`.  It records
  monthly, weekly and side IC/pooled-global-top-10 EV, exact base-to-residual
  replacement attribution, side-local frozen base-feature/context
  distribution/covariance shifts, and separately reported BOCPD-regime versus
  LGBM/BOCPD-transition conditional alignment.  There are 175,680 hourly rows,
  complete sidecar coverage and no 2026 outcome.  Residual loses 7.16/2.10 bps
  versus base in August/September and gains 43.44/10.29 bps in October/November;
  it remains negative net of cost.  Treat the discovered onset/state-age
  effects as preregistration hypotheses for an independent next era, not as a
  gate selected on this bridge.
- [x] Seal and validate the Aug--November score bridge:
  `augnov2025_frozen_july_oos_bridge_validation_economics_20260730_v1` proves
  175,680 unique 1h candidates, both-side coverage, exact economics endpoints
  and base/residual score fits frozen before 2025-08-01.  Base/residual rank
  IC is 0.0269/0.0470; global top-10 net EV is -109.92/-97.53 bps.  Residual
  helps ranking but does not clear the roughly 1% cost burden; no replay.
- [x] Extend the six fixed regime/transition/combined residual-LGBM and
  bounded-GAM arms using only compatible OOF labels resolved before
  2025-08-01, then score all Aug--November common-30 rows with one pooled
  global top-10 per arm.  Authority:
  `augnov2025_common30_fixed_preaug_context_oos_extension_20260730_v2`.
  The best arm, bounded GAM-regime, reaches IC 0.0557 and -93.60 bps top-10
  EV; residual-transition is -97.26 bps.  Every arm remains net-negative with
  negative Q10 tails.  V1 is explicitly invalid/non-authoritative because its
  period file contained selected candidates rather than period aggregates;
  v2 is checksum-sealed and corrects only that reporting bug.  No HPO, 2026
  data, mapping promotion or portfolio replay.
- [x] Refit the fixed bounded GAM regime/transition/combined arms on all
  compatible pre-2026 labels (v3 OOF + July OOF + Aug--November common-30
  OOS), with maps restricted to those same pre-2026 raw-score ledgers; assess
  the unchanged 127,777 2026 hourly candidates globally.  Sealed authority:
  `final_refit_h2_common30_gam_sensitivity_20260730_v2`.  Rank-preserving
  GAM-regime is best of the new refits (IC 0.0855, -65.91 bps), followed by
  combined (-66.51 bps); it improves the frozen residual baseline (-77.51
  bps) but has week/month Q10 -96.48/-116.40 bps.  It does not beat the
  frozen v3 GAM-regime/-combined controls (-57.92/-59.28 bps), as recorded in
  `final_refit_h2_common30_gam_vs_v3_controls_20260730_v2`.  V1 is
  non-authoritative only because its aggregate H2 map-support field was
  wrong; v2 keeps the same scores/results and actual per-arm support. This remains a
  common-30 H2 sensitivity only: no 2026 tuning, promotion, replay or policy
  change.
- [x] Re-run category stability on the new H2 common-30 score/context/economics
  ledgers: `h2_common30_regime_category_performance_stability_20260730_v3`.
  It establishes three independent pre-2026 windows and both-side support for
  seven observed regime/transition/combined categories, using one pooled H2
  global top-10 before attribution and untouched 2026 assessment.  Zero
  category passes positive leave-era-out transfer.  Requirement remains
  incomplete; no category gate or promotion.
- [x] Close the December common-30 hourly score/label bridge and supersede H2
  readiness with `h2_2025_identical_row_oof_bridge_audit_20260730_v7`.
  December has 44,640 exact hourly candidates, both-side native/PIT coverage,
  and complete nested 1m 12h execution labels. Its base+residual scorer is
  immutable before December (fit labels end 2025-07-31 23:00Z); December and
  January-resolving outcomes are assessment-only. This is still a common-30
  sensitivity, not an identical-population v3 replacement. On all 44,640
  rows under one pooled raw global top-10, base/residual are -117.55/-84.69
  bps. The current
  regime/transition sidecars omit the last 12 December hours (720 candidate
  rows), so the context result is a 43,920-row explicitly common subset with
  no fill/reroute: baseline global top-10 is -83.79 bps and all six context
  arms are worse (-101.05 to -120.91 bps). Do not promote, map-refresh or
  replay. The raw timestamps are present, but the frozen reconstruction fails
  the canonical overlap check; recover exact serialized fold-03 model state
  before any full-month context claim.

On identical candidate rows:

1. baseline;
2. regime-only;
3. transition-only;
4. regime + transition;
5. each of the above in base features;
6. each in residual/trust features;
7. additive GAM regime calibrator;
8. bounded regime × feature and transition × feature interactions discovered
   by subsampled SHAP interaction search and regime-conditional permutation
   importance.

Report alpha IC, execution IC, raw and mapped EV, calibration, opportunity
recall, Q10/Q50 week, Q10/Q50 month, latest period, worst period, both-side
economics, tie mass and book concentration.

Morphology status: the 157-event leave-era-out alignment has no matched causal
economic baseline and its post-event outcome rows span incompatible economic
grades.  Its grade-separated outcome tables are descriptive only; do not pool
them or call them incremental EV evidence.

## Promotion gate

The sealed completion snapshot is
`regime_objective_completion_audit_20260730_v20`: 52 requirements are proved
and four remain incomplete (`recurring_transition_clusters`,
`december_2025_final12_frozen_context_reproduction`,
`frozen_2026_failure_incremental_economics_application`, and
`regime_category_performance_stability`). None authorizes a specialist or
policy gate; all currently contain negative or fail-closed evidence.

Promotion requires stable, repeated and economically coherent categories:

- positive aggregate and latest mapped EV;
- improvement over frozen baseline;
- majority-positive months and acceptable worst week/month;
- both long and short economically viable;
- sufficient support and calibrated uncertainty;
- no dependence on ex-post phase, unstable local component IDs or reused
  evaluation labels.

Until all gates pass, regime/transition outputs remain diagnostic context and
cannot control policy or portfolio allocation.

Trajectory-sidecar identical-row integration is fail-closed at
`trajectory_transition_identical_row_stack_coverage_20260730_v1`: historical
coverage is 9,515/9,515 hourly timestamps, but frozen 2026 is only
1,042/1,699 (61.33%). No missingness policy was pre-registered, so no
drop/fill/reroute or stack arm is allowed.

`trajectory_missingness_identical_row_ablation_preregistration_20260730_v1`
now pre-registers the permitted repair: retain all rows, add availability,
neutral-fill unavailable trajectory probability/entropy/margin with
0.5/log(2)/0 and retain existing transition as fallback. The five fixed
side-local GAM arms must be executed without changing this contract or using
2026 tuning/cluster IDs/1m model rows.

Trajectory-only transition stack work is blocked fail-closed by
`trajectory_transition_sidecar_readiness_20260730_v1`: await a sealed 1h
pre-2026 OOF and untouched-2026 probability/uncertainty sidecar with fit and
label-resolution provenance. Do not invent scores, use type-cluster IDs or
use 1m model rows.

The sealed `h2_category_failure_risk_trust_ablation_20260730_v1` is fail
closed: pre-registered leave-era rank stability is below 0.70 for regime and
combined (-0.429) and transition (-1.0). No 2026 trust/exclusion/downweight
or portfolio test is authorized.

## Explicit failure/value target gate

- [x] Seal `pre2026_oof_model_failure_incremental_value_20260730_v3` and its
  cadence supplement `..._v4`: all model/OOF/mapping/candidate rows are 1h;
  1m is label/economics evidence only.
- [x] Keep coverage arm-local: trajectory includes 2023-Apr--Dec (nine eras,
  785,750 rows); regime, transition and combined begin in 2024 (eight eras,
  682,320 rows). Do not conflate these availability layers.
- [x] Retain only passing pre-2026 heads: every incremental-utility and
  selected-net-failure arm passes; reject every false-positive-severity arm.
  No 2026 economics was read.
- [x] Validate the nested failure overlay and exact gamma HPO. Overlay v3 plus
  audit supplement v4 pass all cohort/cadence/label/high-low assertions (66
  core-overlay and 66 v1-outer cohorts); gamma-v2 has exact 68,234-row parity.
  Gamma .125/.25/.5 improve median AUC by +.002716/+.005021/+.006788 but fail
  economic stability: high-low EV median deltas +.000096/+.000813/+.000577
  with only .375/.375/.500 improving eras. All gamma values are rejected,
  `selected_gamma=null`, and `authorized_for_2026=false`.
- [x] Supersede the misnamed probability-only v2 broadcast with authoritative
  `pre2026_regime_only_downside_risk_broadcast_20260730_v3`. It uses sealed
  v2_r1 hourly heads exactly: opportunity × conditional failure rate ×
  conditional downside severity, in return units, with lambda .25/.5/1.0.
  The scalar is broadcast equally within each hour; order audit passes and zero
  fallback is available but unused in all eight supported eras. Every arm fails
  the aggregate/every-era/both-side/week-Q10/Q50/month-Q10/Q50 residual-control
  gate. Best score-only .25 is +0.53 bps aggregate but -2.74 bps minimum era
  and -14.33/-13.28 bps week/month Q10. Best context .25 is +0.68 bps aggregate
  and +1.07/+0.49 bps long/short, but -2.55 bps minimum era and -14.50/-12.29
  bps week/month Q10. No lambda is selected; no 2026 application is authorized.
- [x] Run the authoritative joint increment gate:
  `pre2026_joint_score_context_incremental_gate_20260730_v2` supersedes the
  provisional v1 comparison (now lineage-only). One frozen 1h side-local
  implementation, 150k cap, availability mask, folds and rows produce 132/132
  equal arm/target/era/side train+test cohorts. All eight gates fail. Utility
  median deltas are combined -0.005055, regime -0.000275, trajectory -0.000769
  and transition -0.003997. Failure medians are +0.001935/+0.005315/+0.002259/
  -0.009070, but positive-era fractions are only .50/.625/.556/.375 and worst
  deltas -0.060785/-0.047085/-0.028010/-0.054822. Generic context has no stable
  incremental value beyond CORE scores.

  Environment provenance is sealed at
  `pre2026_joint_score_context_incremental_gate_environment_20260730_v1` and
  independent recomputation/hash/identity/cadence review at
  `pre2026_joint_score_context_incremental_gate_independent_review_20260730_v3`.
  The review confirms the exact diagnostic and that no 2026 candidate,
  economics, replay or portfolio file was opened.
- [x] Run the timestamp-level correction on one statistical row per UTC hour.
  Authority:
  `pre2026_hourly_book_risk_calibrator_20260730_v2_r1`. The corrected artifact
  replaces v2's double-counted prediction-coverage audit; all 48 score-only and
  context coverage checks now match exactly. The study retains one pooled
  global top-10, preserves within-hour ordering and side semantics, uses zero
  broadcast adjustment for unsupported 2023 regime/transition/combined rows,
  and gates those arms on eight supported eras (trajectory on nine). Regime
  gamma .10 improves median era EV by +3.47 bps with 75% positive eras and
  positive long/short medians, but worsens weekly Q10 by -3.57 bps. The raw
  residual control is negative in every era, so all 12 arm/gamma candidates
  remain ineligible. No 2026 data was opened; training and assessment stay 1h
  and 1m remains nested label/replay evidence only.

Final `frozen_2026_failure_value_correction_preregistration_20260730_v3`
supersedes v1/v2, checksum-binds joint-v2 plus its environment provenance and
explicitly seals zero eligible heads and `authorized=false`: no frozen-2026
candidate/economics file was opened and no correction score was produced. Do
not apply any generic head. Only a materially different mechanism that clears
the exact jointly reviewed OOF gate may receive a new no-2026-read
preregistration, fixing its correction/calibration/weights and pooled-global
top-10 reporting before any 2026 file is read.

## Sealed missingness-aware trajectory ablation: result and next constraint

- [x] Execute the fixed five-arm contract at
  `trajectory_missingness_identical_row_ablation_20260730_v1`.  It preserves
  all 127,777 frozen-2026 **hourly** candidates (1h training/assessment;
  1m labels/replays only), uses 381,814 historical rows, pre-2026 OOF mapping,
  global pooled top-10 selection, the exact neutral missingness constants and
  no type/cluster IDs or 2026 tuning.
- [x] Report economics, tails, both sides, availability, turnover and
  concentration.  Baseline is -69.99 bps / IC 0.0664; the full trajectory
  stack is -70.75 bps / IC 0.0832 and has a worse July (-170.80 versus
  -121.95 bps), even though June improves.  All arms are net-negative, and
  full-stack short-side EV worsens (-74.94 versus -62.38 bps).  Its 54.12%
  replacement rate and availability-selection shift require explanation, not
  an ex-post gate.

Next permitted work is diagnostic only: quantify whether neutral-filled
trajectory rows are systematically different *before* score selection and
whether their feature support/side mix differs from available rows.  If a
follow-up is warranted, pre-register one causal availability-shift mechanism
and assess it on a new frozen era.  Do not run a portfolio replay, change the
top-k policy, or promote a trajectory/regime/transition gate from these
results.

## March-20 direct-versus-residual trust diagnosis

- [x] Materialise an exact 136,074-row March/April bridge to the authoritative
  soft regime, transition and trajectory sidecars at signal `__ts__`.
  Authority:
  `marapr2025_direct_residual_regime_trust_diagnostic_20260730_v1`.
  All context is strict pre-March/held-era OOF; OOD, states, clusters,
  destinations, post-entry paths and actions are excluded.
- [x] Reconcile source-specific selection.  Direct/residual book Jaccard is
  only 6.4--10.0%.  Direct-only net changes from +12.85 bps before March 20
  to -100.63 bps afterward and -107.96 in April; residual-only is
  -21.85/-58.47/-23.67.  The failure is source-specific conditional payoff
  layered on a broad market deterioration.
- [x] Separate state-boundary learnability from economic trust learnability.
  Authority:
  `marapr2025_direct_residual_regime_break_learnability_20260730_v1`.
  Trajectory-only context recognises March 20 at grouped-OOF AUC 0.700 with
  every fold above 0.676.  Best side-local direct-over-residual trust AUC is
  only 0.536 long and 0.532 short; top predicted short trust remains
  economically negative.  No gate or replay is authorized.

Next required work is not another probability-map or regime-model sweep.
Reconstruct older identical-row OOF direct/residual scores; materialise
inference-parity market mechanics; and learn side-local direct incremental
capture, cost clearing and adverse loss before a bounded trust head.  Preserve
residual as fail-closed route and require an untouched common cohort.

Readiness refinement: May--July 2026 already supplies 125,551 exact common
rows with a true OOF-compatible direct-q25 score, but its residual was trained
on a mismatched legacy 24-hour target.  Retain q25 and rebuild only the
side-local residual on exact current-policy H12 net over the 127,777 raw-score
identities, using the sealed monthly cutoffs.  Attach regime/transition at
signal time and use the frozen neutral-fill plus availability treatment for
trajectory gaps.  No trust test on the existing mismatched pair is valid.

## Exact-H12 July diagnosis completed

- [x] Rebuild the side-local residual on exact current-policy H12 net and
  select the long/short pair jointly on pooled-global top-10.  Authority:
  `exact_h12_side_local_residual_oof_20260730_v2`.  Long selects blend 0.75;
  short selects 0.0.  May/June/July economics are
  -67.88/-104.26/-148.98 bps, so promotion is rejected.
- [x] Attach frozen causal context at signal time without state IDs,
  destinations, OOD, post-entry or action fields.  Authority:
  `exact_h12_residual_regime_transfer_diagnostic_20260730_v1`.  Combined
  context recognises July at grouped-OOF AUC 0.792, but best economic-trust
  AUCs remain 0.435 long/0.521 short for residual over base and
  0.476/0.577 for direct over residual.
- [x] Run the canonical causal 21-day recent-EV map before pooled-global
  top-k.  Authority:
  `exact_h12_residual_recent_ev_mapping_20260730_v1`.  Mapped residual is
  -72.82/-108.87/-180.00 bps and July is 99.54% short.  Mapping amplifies
  the allocation failure; it does not establish a tradable route.

The regime objective is now narrower: identify transition state robustly,
then prove a causal economic mechanism for using it.  July recognition is
adequate for diagnosis, but score trust is not learnable from the present
context/labels.  Next tests require older identical-target ledgers or an
untouched era, and should target cost clearance, favourable capture and
adverse-loss severity.  Pre-register any regime weighting or specialist,
show within-regime support on both sides, and require incremental latest-fold
and tail economics.  Do not use a July classifier itself as a gate.

## Model-derived mechanics sidecars (added 30 July)

- [x] Materialise the exact schema-intersection candidate panel in
  `pre2026_model_derived_mechanics_representation_20260730_v1`: 214,160 1h
  OOF-score rows across Apr--Dec 2023 and Mar--Apr 2025, identity checked by
  candidate/timestamp/symbol/side. Keep the 2024 and May--Jun 2025 gap
  explicit; do not proxy it with an hourly many-to-one panel.
- [x] Generate side-local leave-month-out OOF mechanics sidecars for gross
  opportunity +25 bps, net-positive conversion and >=100 bps downside risk.
  Use the fixed 15 causal common primitives; CatBoost importance discovery is
  outer-train-only, retains top eight positive fields, forces score anchors,
  and forbids targets/actions/exits/GMM fields.
- [x] Reject the fixed equal-rank combination: it loses -3.64 bps average
  pooled-global top-10 EV versus residual, improves 3/11 months and worsens
  short economics. Sidecars remain research-only; do not add them directly to
  base, residual, map, policy, portfolio or replay.
- [ ] Backfill/align candidate-level causal mechanics primitives for 2024 and
  May--Jun 2025, preserving the exact H12 score and execution-label lineage.
  Fail closed on any candidate, timestamp, symbol, side, score or economics
  mismatch.
- [ ] Pre-register a nested, purged inner combiner for the three OOF sidecars:
  compare opportunity-only, conversion-only, downside-only and bounded joint
  overlays with identical score-only control, causal map and global pooled
  top-10. Fit weights/calibration only inside outer training, not the held
  month. Require higher net EV, weekly/monthly Q10, non-negative long and
  short deltas, and no worse selected expected shortfall.
- [ ] Use Tree-SHAP interaction discovery and regime-conditional permutation
  importance only to nominate stable model interactions inside each training
  fold. Keep current-regime and transition-state probabilities as distinct
  optional families; do not collapse them or emit a manually defined composite
  merely because an interaction recurs.
