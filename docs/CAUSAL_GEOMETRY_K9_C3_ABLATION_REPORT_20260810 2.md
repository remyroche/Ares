# C3 Causal Geometry/K9 Cadence Ablation

## Decision

The proposed C3 separation is viable in this matched study:

1. market geometry/K9 can be made base-independent and fitted without outcomes;
2. base-dependent leaf/path health can be regenerated for every base refit;
3. downstream models that consume geometry/K9 can be restricted to post-bundle
   training rows without losing the observed economics.

The strict rolling three-month burn-in is the economic winner in this four-month
replay.  It reaches **+48.88 bps/trade at pooled global Top-5%**, versus
**+41.40** for the frozen strict bundle, **+42.59** quarterly, **+43.54** for
the diagnostic in-sample K9 arm, and **+3.19** for the matched upstream
Base+Consensus control.

It is not yet a production promotion.  Rolling K9 centre stability weakens to
0.704 cosine in the June refresh, and this study uses a new raw-market K9
representation rather than reproducing the unavailable historical supervised
geometry artifact.  The most defensible next production candidate is therefore
the **quarterly 3-month burn-in / 9-month post-burn-in meta window**, pending a
longer frozen confirmation; rolling is the economic challenger.

## Scope and fixed contracts

- Side: long only.
- Evaluation: April--July 2026, 235,567 matched candidates.
- Candidate rows, upstream scores, policy paths, costs and exit mechanics are
  identical across arms.
- Entry: first 15-minute open one hour after decision.
- Exit: SL = 3 ATR; trailing-profit activation = 0.5 ATR; giveback = 0.25 ATR;
  12-hour timeout; 100 bps cost deducted exactly once.
- Upstream control: the persisted strict-R3 monthly base plus ten LambdaRank
  consensus handoff.  It is deliberately fixed in this experiment.

This is therefore a **downstream state-contract ablation**, not an end-to-end
replacement/reproduction of the historical canonical artifact.

## C3 state contract

### Base-independent market geometry/K9

Each bundle fits an unsupervised K=9 MiniBatchKMeans representation on the
frozen 120-field causal long feature contract.  It uses no outcomes, labels,
base scores, leaf identities or outcome-dependent sample weights.

The downstream state receives 30 raw geometry fields:

- nine soft memberships;
- nine negative distances;
- nine squared-membership confidence values;
- entropy, top-two membership margin and nearest-centre OOD distance.

It also receives four causal state-history summaries from prior decision times:

- membership-weighted 28-day support;
- adequate-support mass;
- marginal state OOD;
- 28-day distribution-drift PSI.

### Base-dependent leaf/path health

For every held month, a separate 64-round R3 robust-clear reference model is
refit using only H12 labels available before that month.  It supplies aggregate
active-leaf support/OOD only:

- effective support and support p05/p50/p95;
- adequate-support fraction and leaf coverage;
- marginal and joint leaf surprise.

Raw leaf IDs are never passed across folds.  This model may use older labelled
base-training data because it does not consume K9.  Its **aggregated outputs**
are regenerated under the current base refit before the meta layers see them.

### Downstream restriction

The Severe-200 safety classifier and Correctness LambdaRanker receive five
upstream score fields plus the 42 state fields above: 47 fields in total.

For every causal arm, their supervised training condition is:

```text
label_available_ts < held-month start
AND decision_ts >= geometry_bundle.fit_end
```

Thus a downstream model never trains on rows older than the geometry/K9 bundle
that defines its feature semantics.  The runner asserts this condition and the
fold audit records zero pre-bundle training rows in all 16 completed folds.

The ten upstream consensus heads do not consume geometry/K9 in this experiment,
so they remain the fixed upstream handoff and are not subject to this restriction.

## Arms

| Arm | Geometry fit | Geometry rows | Downstream fit restriction |
|---|---|---:|---|
| `in_sample_k9` | trailing 9-month downstream population | 69,177 | Same population; diagnostic only |
| `c3_frozen` | Jan--Mar 2025 raw market burn-in | 100,000 | Apr 2025 onward |
| `c3_quarterly` | First three months of each quarterly 12-month training period | 100,000 | The nine subsequent months only |
| `c3_rolling` | First three months of each rolling 12-month training period | 100,000 | The nine subsequent months only |

The first raw 2025 feature months are January--March.  The archive has no 2024
raw-contract rows, so an October--December 2024 static fit was not possible.

## Pooled global policy-net results

All figures are net bps per trade after the fixed 100-bps cost.

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Policy-net rank IC |
|---|---:|---:|---:|---:|---:|---:|
| Upstream Base+Consensus control | +84.08 | +48.32 | +26.72 | +3.19 | -18.44 | 0.1515 |
| In-sample K9 diagnostic | +155.16 | +118.94 | +90.95 | +43.54 | +11.63 | 0.1557 |
| C3 frozen | +137.47 | +122.09 | +84.73 | +41.40 | +11.56 | 0.1588 |
| C3 quarterly | +123.64 | +105.89 | +84.30 | +42.59 | +12.22 | 0.1589 |
| **C3 rolling** | **+149.13** | **+130.27** | **+101.20** | **+48.88** | **+15.35** | 0.1580 |

Top-5 uplift versus the matched upstream control is +40.35 bps for in-sample,
+38.21 frozen, +39.40 quarterly and **+45.69 rolling**.  The strict causal
rolling result is also +5.35 bps above the in-sample diagnostic at Top-5%; the
economic result therefore does not appear to require rows influencing their own
K9 geometry definition.

## Top-5 monthly stability

| Arm | Apr | May | Jun | Jul | Mean | Median | Worst | MAD | Positive months |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 frozen | +93.83 | +34.30 | +35.97 | +65.84 | +57.49 | +50.91 | +34.30 | 15.77 | 4/4 |
| C3 quarterly | +102.45 | +29.24 | +40.20 | +73.80 | +61.42 | +57.00 | +29.24 | 22.28 | 4/4 |
| **C3 rolling** | **+105.39** | **+47.57** | **+36.04** | **+74.97** | **+65.99** | **+61.27** | **+36.04** | 19.47 | **4/4** |
| In-sample K9 diagnostic | +113.44 | +31.15 | +32.72 | +67.14 | +61.11 | +49.93 | +31.15 | 17.99 | 4/4 |

The upstream control's monthly Top-5 was +27.92, -17.48, +9.84 and +29.45 bps.
All C3 arms repair its weak May outcome in this replay.  Rolling has the best
mean, median and worst-month outcome among the strict causal arms.

## Bundle stability

Cluster identities are matched to the preceding bundle by a Hungarian match on
normalised centre similarity before fields such as `cluster_04` are emitted.

- Quarterly is semantically constant through April--June 2026; its July bundle
  matches the prior bundle at **0.836** mean centre cosine.
- Rolling matches adjacent bundles at **0.820**, **0.704** and **0.731** for
  May, June and July respectively.
- The June/July rolling values are enough to retain stable matched IDs, but are
  not strong enough to call the geometry fully invariant.  A live design should
  expose a transition/new-state indicator and avoid treating all cluster labels
  as equally persistent across refreshes.

## Interpretation

1. The evidence supports the user’s causal restriction.  Frozen, quarterly and
   rolling arms all exclude older incompatible downstream rows and all retain a
   strong improvement over the exact same upstream handoff.
2. The evidence does **not** support retaining in-sample K9 merely for
   performance.  It is not the economic winner at Top-1/2/5/10, and rolling is
   stronger at the operational Top-5 tail.
3. Recency helps: rolling beats frozen by +11.66 bps at Top-0.5, +8.18 at
   Top-1, +16.47 at Top-2 and +7.48 at Top-5.
4. Operational stability still favours quarterly.  Its single Q2 geometry
   bundle has one semantic meaning for three held months and its Q3 transition
   is materially more stable than the two weakest rolling transitions.
5. The state outputs are helpful only as a downstream reliability correction
   here.  The experiment does not establish that K9 belongs inside the 120-field
   base or the ten consensus LambdaRank heads.

## Required next confirmation

Freeze the quarterly contract without new tuning:

1. fit raw K9 on the first three months of each quarterly 12-month base window;
2. map its cluster IDs to the preceding persisted bundle;
3. regenerate base-dependent leaf/path aggregate state with every monthly base
   refit;
4. train geometry-consuming downstream layers only after the relevant burn-in;
5. compare frozen, quarterly and rolling over a later untouched period with the
   fully persisted end-to-end base/ten-head/Severe/Correctness bundles.

Do not promote this study as a bit-for-bit reproduction of
`TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807.md`: its historical full structural
bundle was not persisted, and the canonical raw 2024 feature interval is absent.

## Artifacts

- `data_perp/artifacts/causal_geometry_k9_c3_ablation_20260810_v1/predictions.parquet`
- `data_perp/artifacts/causal_geometry_k9_c3_ablation_20260810_v1/fold_audit.parquet`
- `data_perp/artifacts/causal_geometry_k9_c3_ablation_20260810_v1/geometry_bundle_audit.parquet`
- `data_perp/artifacts/causal_geometry_k9_c3_ablation_20260810_v1/metrics_global.parquet`
- `data_perp/artifacts/causal_geometry_k9_c3_ablation_20260810_v1/metrics_monthly.parquet`
- `data_perp/artifacts/causal_geometry_k9_c3_ablation_20260810_v1/run_manifest.json`
