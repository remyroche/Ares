# Cross-Regime Sequential Ablation Report

Status: active, incomplete. This ledger records only evidence produced by the
ordered I -> II -> III -> IV -> V -> VI workstream. A component is not promoted
merely because its implementation or tests exist.

## Frozen decision contract

- Geometry: TP6/SL4, H12, exact 100 bps total cost applied once.
- Entry: signal close + one hour, then the exact next-minute open used by the
  canonical label materialiser.
- Base target: R3 economic simplex with robust-clear B25/T50 semantics unless a
  later predeclared target gate explicitly replaces it.
- Meta target: same-side strict-OOF exact-net residual around the causal base
  expected-net mapping.
- Meta ranking score: frozen base expected-net bps plus predicted residual. A
  residual prediction must never be ranked alone.
- Ranking: pooled global ranking after long and short scores are mapped into the
  same expected-net-bps unit. No per-timestamp or per-side top-k selection.
- Admission comparison: without admission versus side-local robust causal
  21-day EV admission, using only outcomes resolved before each decision.
- Validation: chronological, purged for H12 label availability, with detailed
  side/month/week and worst-period reporting.
- AE/GMM: a frozen historical transform may be reused under the approved
  exception, but its target-free status and exact input lineage must remain in
  the manifest.

## Stage I - Layer/side/head stability MDA

### Required matrix

1. Base long, R3 economic simplex B25.
2. Base short, R3 economic simplex B25.
3. Meta long, shared exact-net residual.
4. Meta short, shared exact-net residual.

Retired auxiliary heads are excluded. Base candidates resolve only from the
declared shared plus side-specific base keys in `config.py`; meta candidates
resolve only from declared meta keys. There is no fallback to every column in a
panel.

### Selection sequence

Coverage and non-constant gate -> univariate screen -> Relief rescue ->
training-only Spearman pruning at 0.95 -> grouped/repeated signed-economic MDA
at global top 10% -> phantom/noise calibration -> smallest feature prefix within
one standard error of the best chronological development score.

MDA groups are fit on training rows. The signed objective uses realised exact
net bps for the selected tail; negative sample weights are not passed to the
estimator. For the meta layer, every baseline and permuted MDA score reconstructs
`frozen_base_expected_net_bps + predicted_residual` while keeping the base
offset unchanged.

The 20,000-row MDA budget is a per-fitted-model limit, not an aggregate
evidence limit. Stage I now freezes three disjoint chronological era cohorts,
each with an internal train/evaluation split and 13-hour purge. Each dedicated
MDA refit uses at most 20,000 training rows; aggregate targets are 60,000 unique
training rows and 20,000 evaluation rows when the side/layer population has
enough support. Training/evaluation identity hashes and all pairwise overlaps
are persisted and asserted. Only these dedicated cohort refits contribute
permutation MDA; the broader stability grid selects parameters but cannot
double-count overlapping rows as MDA evidence. Feature audits retain cohort
median, MAD/std, positive-cohort rate, worst-era MDA and latest-era MDA/label.

### Current evidence

- Infrastructure exists in `extreme_price_movements/stage_i_feature_selection.py`
  and the existing selector in `extreme_price_movements/lgbm_pipeline.py`.
- The strict selector now rejects preset-feature bypasses, requires the exact
  signal-close plus 13-hour label-availability contract, and reconstructs every
  residual MDA score in common bps.
- Exact selector coverage is readiness-aware. A pure causal/source-history
  warm-up prefix is not treated as ordinary missingness: the first-ready
  timestamp and prefix rows are frozen, the unavailable prefix remains NaN,
  and the feature must still exceed 90% aggregate coverage both after readiness
  and from the required 2024-01-01 evaluation boundary. Sporadic post-readiness
  gaps remain subject to rejection. Monthly-by-side coverage is persisted as a
  diagnostic rather than used as a noisy per-month hard gate.
- The same first-ready contract is propagated into the frozen base/meta winner
  manifests, selected-panel checkpoints, production input digests and strict
  OOS coverage audit. A finite value before readiness fails closed; production
  gates post-readiness and evaluation-window coverage instead of reapplying a
  contradictory whole-history gate.
- On the deterministic selector cohort, `btc_ex_eth_oi_dominance_z_ratio` and
  `btc_oi_dominance_z_ratio` have 85.30% whole-history coverage only because
  they are unavailable before 2023-03-14 15:00 UTC. They are retained as
  `causal_warmup_prefix`: post-readiness coverage is 95.03% and coverage from
  2024-01-01 is 95.62%, with no imputation or backward availability.
- `extreme_price_movements/prequential_r3_value_map.py` supplies the missing
  same-side, prior-resolved R3 opportunity-score to expected-net-bps map.  It is
  explicitly distinct from the trailing 21-day admission map.
- The sequential runner now selects each top-k once over the pooled OOS
  population and only then attributes the identical selected set by side and
  month.  It does not rerank within timestamps, months, or sides.  Both base
  and reconstructed-meta ledgers feed the metrics, and the meta ledger feeds
  the side-local causal 21-day admission comparison.
- The selector-to-OOS handoff now includes the exact same-side base OOF fields
  required by every meta winner: `P(adverse)`, `P(weak)`, `P(clear)`, the raw
  `P(clear)-P(adverse)` contrast, and the prior-resolved expected-net-bps map.
  These fields are forced through selection, and the strict generator receives
  the injected selected frame rather than the pre-injection raw frame.
- `stage_i_production_oos.py` now provides the missing frozen-winner production
  boundary. It binds all four side×layer cells to exact ordered features,
  runtime parameters, selector/source manifests, code revision, approved
  reuse-backward exception and evaluation calendar. Older rows remain available
  for strict base/meta fitting and for causal 21-day support at the evaluation
  boundary, but only calendar rows can enter OOS ranking or reporting.
- Base and meta availability are explicit separate strict-OOF flags. Finite
  scores on unavailable rows, omitted candidates, altered `side::candidate_id`
  keys, changed labels or incompatible panel manifests fail closed. The exact
  selected raw values, R3 targets, normalized weights, fold controls and value
  map configuration are included in the run-input digest.
- The production artifact is atomic, checksum-verified and restart-safe. It
  preserves separate full-history, evaluation raw, base-strict and meta-strict
  ledgers; a >=90%-coverage/nonconstant audit of every selected raw/generated
  feature; and detailed base/meta raw-versus-admitted pooled-global reports with
  month/week/side attribution, IC, calibration, concentration and worst-period
  diagnostics. Base and meta admission tails retain the same requested global
  k even when fewer rows pass the 50-bps causal threshold.
- The 2024 reference surface is materialised for January-November at
  `data_perp/artifacts/stage_i_surface_2024_2026_20260803_v1` (11 months,
  920,460/920,460 rows label-valid, 460,230 per side, 68 MiB).  December 2024
  is absent from the frozen candidate sources and will
  remain an explicit missing month rather than being interpolated.
- The completed historical exact-label substrate contains 118,734 rows from
  August 2022-December 2023, of which 110,813 are valid.  The completed Pack-B
  audit contains 4,515,650 rows from January 2025-July 2026, but its 181-symbol
  population has severe minute-path gaps in January-April 2026.
- A bounded common-30 repair appended only missing immutable minute candles.
  Post-write verification is 100% for all 30 symbols over both request sets:
  1,506,180/1,506,180 required minutes for Dec/Jan and
  3,812,880/3,812,880 for February-April.  This is not a claim of full
  181-symbol Pack-B coverage.  The product-bound request is now frozen for all
  30 symbols and independently verifies 5,230,800/5,230,800 Jan-Apr required
  minutes with no fallback or fetch.  Regenerated exact labels are complete at
  172,800/172,800 valid rows.
- The 2024 surface symbol alias (`BTC_USD:USD`) is normalized only at the PIT
  adapter boundary to the store payload identity (`BTC/USD:USD`); candidate ids,
  timestamps, labels and feature values remain unchanged.  The strict store
  verifier caught this mismatch before fitting.
- The integrated Stage-I through Stage-VI contract suite passes: 180 tests on
  2026-08-03. This proves implementation and causal/identity contracts only;
  it is not economic evidence.
- No Stage-I winner has been selected.  The population-balanced selector read
  was stopped at the user's request on 2026-08-03; no model fit is running.
  The stopped implementation committed only after reading every feature, so it
  discarded completed reads.  The materialiser now writes deterministic
  32-column exact-PIT checkpoints, resumes only when candidate identities and
  schemas match, invalidates only the original block containing a newly frozen
  coverage rejection, and removes duplicate checkpoints after the immutable
  combined selector manifest is durable.  This repair is tested but the large
  selector has not been restarted.
- The user explicitly authorised restart on 2026-08-03.  The first resumed
  selector materialisation failed closed before its first feature block because
  the frozen `stage_i_production_inputs_20260803_v1` generator-registry SHA no
  longer matched the current feature code.  It wrote only the deterministic
  80,000-row ledger and selector contract; no feature-selection or model fit
  ran.  A fresh production-input contract is being rebuilt against the same
  source surfaces and current generator registry before the selector is
  restarted.  The stale hash will not be overridden or silently accepted.
- Production-scale causal mapping has been repaired before the replay.  The R3
  score-to-bps map now advances stable decision and label-availability event
  streams with running global/per-bin statistics instead of rescanning a whole
  side for every decision timestamp.  It preserves strict
  `label_available_ts < decision_ts` semantics and matches the frozen quadratic
  reference in randomized regression tests; a 100,000-row sanity case runs in
  about 0.3 seconds locally.
- The causal 21-day side admission map now uses side-local sorted availability
  indexes and half-open 21-day windows rather than two full-ledger scans per
  calendar day.  It restores canonical decision/identity order before tied-score
  isotonic fitting and is exact-output equivalent to the frozen reference in
  randomized tests.  A 250,000-row production-shaped test passes inside the
  declared performance gate.
- A fail-closed winner-bundle freezer now converts exactly the four completed
  base/meta x long/short selector cells into the production boundary.  It
  normalizes only the fixed base multiclass/3-class and meta Huber semantics,
  binds ordered features, parameters, selector/source manifests and the
  user-approved reuse-backward exception, and atomically refuses conflicting
  publication.  Its focused plus production-boundary suite passes 15 tests.
- The prior shared residual D0-D4 result is a completed shadow diagnostic with
  no promotion; it is architecture evidence, not an economic winner.

### Results

Not yet executed on the verified full 2024-2026 surface.

## Stage II - Meta-specific conversion archetypes

Pending Stage-I freeze and economic execution. The code-only funnel is now
implemented as a bounded sequential comparison: up to eight predeclared
side-local path-discovery candidates (K=3--6), strict-OOF causal soft
recognition, fold/month/side/symbol economic-separation and concentration
audits, and explicit log-loss/Brier recogniser gates. Candidates are retained,
diagnostic, or rejected deterministically before any downstream meta model is
called.

The handoff is deliberately richer than an expected-net map alone. Every
downstream meta request must include the direct same-side strict-OOF R3
`P(adverse), P(weak), P(clear)` simplex *and* its prequential expected-net-bps
map. It verifies finite simplex mass, same-side source lineage, strict OOF
row/fold membership, fit-end-before-decision, direct (unconverted) semantics,
and prior-resolved map provenance. A residual prediction is accepted only as a
raw bps residual and is reconstructed with that frozen base map.

For the frozen discovery winner, four matched, identical-row **meta-only**
controls are compared: none, soft memberships, residual prior, and both. The
base is never refit, routed, or converted. Each control reports one
pooled-global common-bps top 1/5/10/20 book with and without the side-local
causal 21-day admission map; month and side figures are contributions of that
unchanged book, not local reranks. The two admission views are emitted exactly
once per tail. Control choice is lexicographic on worst selected month, worst
selected side, mean top-tail net across the required views, then lower side
concentration—rather than aggregate unadmitted top-10 alone. A missing required
admission view yields `NO_STAGE_II_META_CONTROL_ADVANCES`, not a fallback.

The separate release adapter now prevents the development funnel from being
mistaken for an OOS result. It requires ordered non-overlapping history,
development-selection and locked-evaluation windows; publishes an atomic,
checksummed winner bundle bound to the Stage-I base winner/ledger, dataset,
labels, universe, code revision, selected candidate/control/config and ordered
features; and refuses mutable/reselected evaluation. The locked OOS ledger
requires the full candidate/symbol/signal-close/decision/side identity for both
base and meta joins, close +1h entry and +13h label timing, direct R3 simplex
and raw contrast, base/meta fold and prior-resolved cutoffs, archetype soft
memberships, reconstructed bps, and layer-specific prequential 21-day
admission lineage. Its isolated report emits base and meta pooled-global raw
and admitted top 1/5/10/20 tables, unchanged-set month/week/side interactions,
worst-period, IC/calibration, coverage/cost/residual/concentration, and
selected-identity digests. This is code-only release infrastructure; it has not
published or evaluated a 2024--2026 OOS ledger.

The locked scorer is a one-shot interface: it receives the frozen winner plus
purged history/development and evaluation identity, exposes no selection/HPO
operation, and must return exact winner/feature/model/label/base/fold hashes.
The published OOS manifest retains the scorer model hash, exact feature-contract
hash and observed ledger identity/content hashes. Base value mapping and the
base/meta 21-day admission maps each carry independent same-side, prequential,
prior-resolved lineage; admission flags must equal the declared mapped value
threshold exactly.

The release gate additionally requires the base expected-net map itself to be
same-side and prequential with a prior-resolved cutoff; independent base and
meta admission map provenance and exact `mapped_bps >= 50` admission flags;
and an observed OOS content digest plus frozen feature-contract hash. The
one-shot scoring API receives only hash-bound history/development/evaluation
partitions, proves earlier labels resolve before the next cutoff, and rejects a
scorer unless it returns matching winner/feature/model/label/base/fold hashes
with reselection and HPO explicitly forbidden.

No Stage-II economic result has been generated.

## Stage III - Shared cross-era residual expert

Pending Stage II. The production candidate is now **exclusively one shared
regime-aware residual expert**. Regime-local experts, hard regime routing and
separate per-regime score paths are superseded and are not Stage-III arms.

The frozen reconstruction is:

```text
R3 same-side strict-OOF probabilities
  -> prior-resolved side-shrunk causal expected-net map
  -> prior-resolved soft-regime residual baseline
  -> one shared candidate-residual expert
  -> strongly shrunk side x soft-regime calibration correction
  -> common expected-net bps
  -> pooled-global ranking
```

The shared expert predicts only candidate-specific conversion after removing
the causal broad-regime baseline:

```text
candidate_residual = realised_exact_net
                   - causal_base_expected_net
                   - causal_regime_prior_residual

predicted_net = causal_base_expected_net
              + causal_regime_prior_residual
              + predicted_candidate_residual
```

All priors, regime-relative transforms, calibration corrections, OOD fields
and model-validity estimates must use only outcomes resolved before the row's
decision timestamp. Missing state remains explicit; it is never silently
treated as a zero-probability regime.

### Sequential Stage-III funnel

Run this as a funnel, not a factorial search:

1. **A - residual centering:** current residual; side-centered; side x broad
   regime-centered; soft-regime-centered. Use strong hierarchical shrinkage.
2. **B - robust training:** pooled rows; square-root environment balancing;
   worst-era model/HPO selection; both. Initial robust selection uses
   `mean + 0.5 * worst - 0.25 * era_std` or a stricter lexicographic gate.
3. **C - causal conditioning:** invariant core; soft regime probabilities and
   transition state; restricted base-value x regime interactions; prequential
   regime-relative residual/z features.
4. **D - model validity:** relationship breaks; contribution/distribution
   OOD; active model-failure probability; compact combination. These remain
   soft inputs, never automatic suppression gates.
5. **E - common-bps calibration:** global; side-local; side x soft-regime
   hierarchical correction, strongly shrunk toward the parent map.
6. **F - ranking loss:** only for the final two stacks, compare pointwise
   robust residual loss with a small within-side/regime/date/base-EV/cost-ATR
   pairwise term at predeclared 50/100-bps separation.

Target challengers after the centering control are clipped Huber residual,
regime-standardized residual, ordinal residual and residual quantiles. They
advance only through the sequential target gate; exact-net regression is not
expanded into a broad sweep.

Feature groups are admitted by cross-era transport MDA, not one-fold MDA:

```text
transport_MDA = median(train-era -> test-era MDA) - 0.5 * MAD
```

An admitted group must beat q95 phantom importance, be positive in at least
70% of era/fold/seed cells, avoid severe unexplained sign reversal, and remain
non-negative in the latest historical block. Groups are labelled
`INVARIANT_CORE`, `REGIME_CONDITIONAL`, `REGIME_LOCAL_DIAGNOSTIC`, `UNSTABLE`
or `REDUNDANT`; only the first two can enter the shared expert.

### Stage-III evaluation and promotion gate

Use chronological expanding train-to-next-era evaluation plus an explicit
train-era -> test-era transport matrix. Report residual IC, Huber/MAE,
calibration slope/intercept, pooled-global top 1/5/10 net and gross, side
contribution/mix, mean/median/worst era, dispersion, positive-era count and
catastrophic-era count. Selection is based on lift over the frozen causal base
in the identical held-out rows, not the absolute residual score alone.

A shared expert can advance only when it improves at least 4/5 or 5/6 eras,
has positive paired cross-era lift, reduces era dispersion, improves the worst
era, and creates no severe new side or latest-era failure. Otherwise terminate
with `SHARED_EXPERT_REMAINS_CROSS_ERA_UNSTABLE`; do not fall through to local
experts without a separate future authorization and evidence gate.

The valid terminal decisions are:

- `SHARED_RESIDUAL_EXPERT_TRANSPORTS`
- `SHARED_EXPERT_REQUIRES_REGIME_CONDITIONING`
- `SHARED_EXPERT_REQUIRES_MODEL_VALIDITY_CONTEXT`
- `SHARED_EXPERT_REMAINS_CROSS_ERA_UNSTABLE`
- `REGIME_LOCAL_EXPERTS_NOT_JUSTIFIED`

The preregistered machine-readable contract is
`configs/stage_iii_shared_regime_residual_funnel_20260803_v1.json`. A frozen
handoff implementation in
`extreme_price_movements/shared_residual_funnel_contract.py` hashes every
predecessor artifact and feature list, freezes TP6/SL4/H12, 100-bps-once,
one-hour entry delay, strict +13-hour label availability, regime-centered
target reconstruction and pooled-global ranking, and rejects local experts or
hard routing. It is infrastructure only: the current A-E scripts still require
winner-to-winner handoff wiring and the transport-MDA/paired-ranking rounds
before Stage III is executable as a complete funnel.  The residual foundation
now materialises all four A-round target controls (un-centered, side-centered,
side x causal broad-regime centered, and soft-regime centered) without using
current/future outcomes.  Hard regime identifiers remain forbidden from the
shared model itself.  Regime-relative z-scores now default to a bounded-influence
prequential absolute-deviation scale rather than variance, while retaining a
standard-deviation negative control.  These additions are infrastructure only;
no economic arm has been evaluated.

The code-only Stage-III implementation has progressed without restarting the
cancelled experiment:

- `stage_iii_shared_expert_runner.py` executes the matched A0-A3, B0-B3,
  C0-C3, D0-D4 and E0-E2 development funnel with one shared both-side model,
  an immutable candidate/symbol/time/side identity, identical OOF rows, exact
  gross/net reconciliation, paired causal-base lift, causal 21-day admission
  replay, and an explicit train-era to later-test-era transport table.
- `stage_iii_feature_admission.py` now fail-closes on fold/seed/transport and
  false-positive-loss MDA, fold-local q95 phantoms, latest-block support,
  effect reversal, >=90% finite coverage, and exact meta/live parity. Only
  `INVARIANT_CORE` and `REGIME_CONDITIONAL` groups can enter the model.
- `stage_iii_residual_target_challengers.py` freezes ordinal, five-quantile and
  context-pair contracts. It independently rejects unresolved targets,
  truthy-string/NaN lineage flags, cross-side base maps and non-prior regime
  baselines. These challengers are not yet part of an executed target round.
- `stage_iii_pairwise_shared_expert.py` implements the missing Round-F model
  primitive: F0 pointwise Huber and bounded F1/F2 shared LambdaRank corrections
  at 50/100 bps. The rank score is mapped on prior-resolved training rows back
  into residual bps before reconstruction. The runner retains all six
  two-E-finalist x three-F combinations, identifies each predecessor explicitly,
  and transports the actual selected F model plus its selected E calibrator.
  For T3/T4, the target-preserving adapter now wraps the already-fitted
  ordinal/quantile model: F0 returns its residual prediction unchanged and
  F1/F2 add only the bounded auxiliary pairwise correction. The adapter verifies
  the base target, label digest, feature digest, candidate support and cutoff;
  it cannot substitute Huber silently. The chronological OOF runner and
  train-era to later-test-era transport both use this path, and a regression
  test forces a robust-target winner through both stages.
  No Round-F economics exist yet because the experiment remains stopped.
- `stage_iii_robust_target_models.py` turns the T3/T4 contracts into actual
  one-shared-both-side fits: T3 uses three cumulative ordinal probability heads
  with CDF repair and frozen class-mean reconstruction; T4 uses five quantile
  heads with crossing repair and median/downside/width outputs. The runner now
  compares T0 Huber, T1 clipped at 200/400 bps, T2 regime-standardized, T3
  ordinal and T4 quantile on identical OOF identities immediately after the
  winning A baseline, then freezes the target winner into B-E and the
  target-preserving Round-F/transport path.
- `stage_iii_artifacts.py` publishes one compact immutable winner OOF ledger,
  all-arm metrics/audits, transport/gate evidence, reproducibility lineage and
  checksums. It also freezes exact ordered base/meta feature lists and every
  arm's feature-list digest. `stage_iii_reporting.py` performs one pooled-global
  tail selection before month/week/side attribution, reconciles cost exactly
  once, and reports chronological signed-residual autocorrelation, hit surprise
  and trade concentration before and after the causal 21-day admission map. It
  deliberately avoids storing every arm's full row ledger, limiting future
  legacy-artifact accumulation.

Passing unit/contract tests prove these structural properties only. They are
not evidence that any Stage-III arm transports economically.

## Stage IV - Broad-to-tail base funnel

Pending Stage III. Ablate top-x retention at 20/30/40/50%, broad-base burn-in,
tail-base burn-in, meta lookback, and whether the broad score enters the tail
base, meta, both, or neither. Tail-base and meta training must use row-level
same-side OOF inputs. Report both base stages and reconstructed meta economics.

The code-only execution contract is now complete. The per-cell implementation
in `stage_iv_broad_to_tail.py` is wrapped by
`stage_iv_v_orchestration.py`, which accepts only explicitly declared immutable
cells and never synthesizes a factorial. A primary sweep must cover all four x
values, while every cell independently freezes broad, tail and meta burn-ins
plus the broad-score route. Cells execute serially and are compared on the
intersection of final strict-OOF candidate identities. Each cell ranks once
pooled-global after reconstruction into common bps. The winner manifest binds
input values, feature lists, targets, cost, population, parameters, lineage
hashes and tie-breaks. No Stage-IV cell has run on the 2024-2026 surface.

## Stage V - MDA co-firing, drift, OOD, and trust controller

Pending Stage IV. Start with a compact shadow controller: 5-8 relationship-break
features, 3-5 strict-OOF contribution-pattern features, and 2-4 prediction-state
drift fields. Evaluate them beyond existing confidence/context controls on
matched high-meta false positives. Only then consider covariance or historical
false-positive similarity extensions.

The controller answers whether the stack is operating in a historically reliable
state. It is separate from the base opportunity model and meta conversion model.
Any overlay must use strict-OOF controller predictions and preserve the identical
candidate entry population.

`stage_iv_v_orchestration.py` now freezes controller contracts independently by
side and layer, including the training-only state digest and exact ordered
feature hashes compatible with Stage III. Base context can enter only the
matching broad/tail base models; meta context can enter only the matching
residual model. The adapter changes input lists only—never an existing score,
rank or admission decision. Controller arms require explicit contracts for
every scored side and are compared on one matched OOF population with
pooled-global ranking. No controller arm has been economically run.

## Stage VI - Causal/path/multi-view archetypes

Pending Stage V. Run causal-feature and realised-path workstreams separately,
per side and using positive-label rows spread over time. Temporarily exclude the
current AE/GMM and Stage-II archetypes from challenger inputs. Path memberships
must be predicted causally and strictly OOF before downstream use. Clustering is
selected by economic separation, causal predictability, temporal stability, and
concentration control—not silhouette alone.

The bounded path grid now includes K={3,4,5,6,8}, AW0/AW1/AW2/AW4/AW5 (AW3
is handled by the required per-side fit), KMeans, regularized diagonal/full
GMM, GMM on compact train-only PCA, and a small deterministic regularized
autoencoder plus GMM. Legacy AE/GMM and Stage-II archetype outputs are rejected
from challenger views.

Strict-OOF path recognition reports log loss, Brier, RPS, calibration
intercept/slope/ECE, membership correlation, top-decile membership enrichment,
and economic confusion using prior-resolved cluster payoff maps. Fold alignment
reports geometric-versus-economic semantic switching. A matched comparator
tests control, base-only, meta-only and both with pooled-global tails. The
multi-view decision surface combines path and economic separation, causal
predictability and temporal stability with a concentration penalty, then emits
the requested retain/diagnostic/reject disposition. No Stage-VI economic arm
has run.

## Experiment table

| Stage | Trial | Contract | Status | Result | Disposition |
|---|---|---|---|---|---|
| I | Infrastructure baseline | Four side/layer cells, signed top-10 MDA | Implemented and focused tests pass | Exact production adapter, prequential R3 map, strict OOF writer, pooled-global metrics and restartable selector checkpoints are wired | Selector stopped; no winner |
| I | 2024 reference surface | Jan-Nov identity/label references with exact PIT joins | Complete | 920,460 rows; Dec absent by source contract | Retained |
| I | 2026 common30 minute repair | Immutable append-missing, frozen Kraken product id, H12 + ATR window | Complete and product-bound | 5,230,800/5,230,800 product-bound Jan-Apr minutes verified without fallback/fetch; 172,800/172,800 exact relabel rows valid | Retained |
| I | 2024-2026 OOS | Frozen Stage-I winner | Selection + production OOS boundary implemented/tested; no economic run | Exact four-cell winner bundle, direct five-field base→meta handoff, older-history strict fitting/admission support, explicit base/meta OOF ledgers, full coverage/identity/lineage audit, pooled-global base/meta raw/admitted reporting | Pending selector restart and winner freeze |
| II | Meta archetypes | Frozen Stage-I inputs | Development funnel + locked production OOS boundary implemented/tested; no economic run | Bounded side-local path modes; strict-OOF causal recogniser; immutable winner; one-shot hash-bound scorer; direct R3 simplex + prequential map; independent base/meta admission lineage; pooled-global base/meta reporting; no hard routing | Pending Stage-I winner and explicit experiment restart |
| III | Shared regime-aware expert infrastructure | Frozen Stage-II winner | A/T/B/C/D/E/F runner implemented/tested; no economic run | One shared expert, immutable OOF identity/lineage, transport-MDA admission, causal calibration/admission replay, target-preserving T0-T4 Round-F and actual-winner transport/gates, pooled-global reporting and compact publication; no local experts | Pending Stage-II winner and explicit experiment restart |
| IV | Broad-to-tail base | Frozen Stage-III stack | Per-cell and sequential orchestration implemented/tested; no economic run | Explicit serial 20/30/40/50 cells, independent burn-ins/routes, matched strict-OOF intersection, pooled-global comparison and immutable winner manifest | Pending Stage-III winner and explicit experiment restart |
| V | Compact OOD controller | Frozen Stage-IV stack | Feature/controller orchestration implemented/tested; no economic run | Positive grouped-MDA co-activation, frozen training-only side/layer feature contracts, Stage-IV input adapter, matched controller selection, no reranking | Pending Stage-IV winner and explicit experiment restart |
| VI | Archetype workstreams | Frozen Stage-V stack | Full bounded grid/diagnostics/comparison infrastructure implemented/tested; no economic run | Separate causal/path workstreams, K through 8, AW4/AW5, PCA/AE-GMM, strict-OOF calibration, semantic alignment, matched base/meta/both and decision matrix | Pending Stage-V winner and explicit experiment restart |

## Reporting checklist for every promoted stage

- training/evaluation windows for every layer;
- candidate, valid-label, and feature-coverage counts;
- selected feature names and layer/side provenance;
- target, mapping, cost, entry, and exit contracts;
- base and reconstructed-meta top 1/5/10/20 gross and net bps/trade;
- with/without side-local causal 21-day admission;
- long/short contribution after pooled global ranking;
- per-month and per-week trades and economics;
- worst month/week/era and concentration diagnostics;
- calibration, rank IC, target-specific predictive metrics;
- uncertainty and paired temporal bootstrap where support permits;
- explicit invalid, negative-control, diagnostic-only, and promoted dispositions.
