# Strict-R3 Long Supportive-Label Ablation — 2026-08-23

## Decision

Advance one **direct-label research challenger**, and do not change the
canonical or live stack:

```text
one-third × predicted path efficiency value
+ one-third × predicted time-to-meaningful-MFE value
+ one-third × frozen prequential upstream value
```

The three inputs are individually strict-OOF expected-policy-bps scores, so
the blend is in a common-bps space.  It is named
`S3_direct_efficiency_time_base_equal` in the sealed output.

The challenger is not production-promoted.  Its head choice and weights were
selected using the reported 2025--2026 research windows.  It must be replayed
through the full current score/admission/portfolio stack, then frozen for a
later untouched period, before any inference integration is considered.

Do **not** advance a causal-regime × path-archetype layer.  Its best soft J2
formulation was economically useful, but not materially better than the direct
challenger.  A small direct/J2 blend is recorded below as a research result,
not a promotion.  Fold-local GMM clusters remain useful diagnostics, but both
stable-geometry and causal-joint constructions lost the decisive portability
comparison against direct labels.

## Scope and label contract

This is offline, long-only target research.  No live config, model bundle,
execution policy, exchange I/O, or canonical production document was changed.

The target sidecar is
`strict_r3_long_supportive_path_labels_2024_2026_20260823_v5`:

* 2,820,951 candidate rows from January 2024 through July 2026;
* 1,663,035 valid complete-path rows (58.95%);
* entry is the frozen decision open; path data are the next 48 completed
  15-minute bars (H12);
* ATR is Wilder-14 hourly ATR from bars completed before the decision;
* path labels become available at decision + 12 hours;
* invalid source, incomplete path, or entry-parity rows are excluded from
  supervised fitting.  They are never encoded as economic failures.

### Direct supportive targets

The Stage-1 direct controls were:

1. H12 peak MFE in ATR;
2. H12 MAE before meaningful MFE in ATR;
3. time to meaningful MFE;
4. H12 final return in ATR;
5. H12 path efficiency.

`final return` is rejected: every held score landed outside its fitted
isotonic-map domain and clipped to one constant, so it provided no ranking.

### Realised-path representation

P1 used 48 target-only fields: MFE and MAE at 15m, 30m, 1h, 2h, 4h, 8h and
12h; event timing; MFE/MAE ordering; peak MFE; efficiency; reversals; final
return; retention/giveback; meaningful-MFE and near-peak timing; pre-stop and
pre-meaningful adverse movement; interim returns; and cumulative variation.

Raw direct/P1 labels are not features.  At inference, a path arm may receive
only causal classifier probabilities, confidence/entropy, and train-derived
expected-policy summaries.

## Validation protocol

Six chronological outer folds were held out:

| Cohort | Periods |
|---|---|
| Development | 2025 Q2, 2025 Q3 |
| Holdout | 2025 Q4 |
| Portability | 2026 Q1, Q2, July |

Each fold uses at most 180,000 month-balanced training rows.  Every training
label has `label_available_ts < fold_start − 12h`; all output rows are scored
only from the frozen 120-field causal base contract and allowed prequential
base/stack values.

Identity and causality audits passed:

* six folds, with 199,862 / 149,600 / 152,682 / 152,006 / 244,626 / 72,905
  held rows;
* 34 Stage-1 arms per fold, all on exactly the same candidate IDs and decision
  timestamps;
* no raw future target appears in an input contract or prediction panel;
* all direct, path and blend panels preserve one-to-one candidate identity.

Metrics below are **global within-held-fold tails**, not a live admission rule
or portfolio replay.

## Stage 1: direct controls and fold-local path diagnostics

Values are mean net bps/trade.  `Residual IC` is Spearman against realised
policy net after subtracting the prequential upstream score.

| Arm | Dev top-1 / top-5 | 2025 Q4 top-1 / top-5 | 2026 top-1 / top-5 | 2026 top-5 worst | 2026 residual IC |
|---|---:|---:|---:|---:|---:|
| B0 upstream control | +39.99 / −17.38 | −23.59 / −54.58 | +43.86 / −10.67 | −61.15 | 0.173 |
| Direct efficiency | +131.97 / +42.35 | +100.08 / +17.38 | +183.56 / +51.02 | +5.64 | 0.163 |
| Direct time-to-meaningful | +133.90 / +35.76 | +100.82 / +14.34 | +170.35 / +57.82 | +29.20 | 0.212 |
| Fold-local GMM K8 + base | +83.87 / +19.49 | +104.23 / +14.94 | +163.69 / +62.84 | +27.84 | 0.195 |

The GMM K8 recogniser itself is learnable (macro one-vs-rest archetype AUC
0.62--0.68 across the six held folds), but its GMM is refit in each outer
fold.  Therefore `path_p_00` does not denote the same realised-path mode from
one fold to the next.  Its pooled numbers are a label-quality diagnostic only.

## Stage 2: geometry-semantic falsification

The fixed-geometry contract fit a K8 GMM on 60,000 equal-month sampled valid
rows from October--December 2024 (140,729 valid definition rows total).  The
definition rows were excluded from all supervised recognition and residual
fits, which began in January 2025.  A rolling alternative refit geometry on
months −6 to −3 and trained each downstream user only on the immediately
following three-month same-bundle population.

| Stable use | Dev top-1 / top-5 | 2025 Q4 top-1 / top-5 | 2026 top-1 / top-5 | 2026 top-5 worst |
|---|---:|---:|---:|---:|
| Frozen K8 + shared policy residual | +91.80 / +24.68 | +80.05 / −5.30 | +93.31 / +26.09 | −3.20 |
| Rolling same-bundle K8 + shared policy residual | +89.15 / +21.60 | +69.94 / −2.86 | +71.57 / +0.50 | −35.20 |

Both variants passed their chronology, identity and label-separation audits.
Their weak economics falsify the proposition that the Stage-1 cluster uplift
is portable as a stable downstream archetype representation.  Do not run
additional K-medoids/HDBSCAN/DTW clustering until a representation can beat
the direct-label challenger under the same semantic contract.

## Stage 3: direct-label bps-space integration

The two strongest direct heads were combined with the sealed upstream bps
score without refitting any outcome model.  This preserves the underlying
Stage-1 strict-OOF lineage.

| Blend | Dev top-1 / top-5 | 2025 Q4 top-1 / top-5 | 2026 top-1 / top-5 | 2026 top-5 worst | 2026 residual IC |
|---|---:|---:|---:|---:|---:|
| 75% efficiency + 25% upstream | +134.48 / +43.03 | +93.79 / +19.93 | +184.03 / +51.18 | +4.77 | 0.169 |
| 75% timing + 25% upstream | +146.32 / +44.46 | +101.24 / +14.54 | +173.33 / +58.92 | +28.98 | 0.212 |
| 50% efficiency + 50% timing | +163.03 / +57.51 | +105.62 / +20.13 | +181.36 / +65.55 | +30.10 | 0.203 |
| **⅓ efficiency + ⅓ timing + ⅓ upstream** | **+161.12 / +57.17** | **+104.95 / +19.81** | **+181.76 / +65.63** | **+30.21** | **0.204** |

The last two are effectively tied.  The three-way equal blend is the research
candidate because it has marginally better 2026 top-5 stability and retains
the frozen upstream signal.  Its 2026 top-5 improvement over B0 is +76.30
bps/trade (+65.63 versus −10.67); its 2025-Q4 top-5 improvement is +74.40
bps/trade (+19.81 versus −54.58).  These are not live-stack uplift claims.

## Interpretation

What works:

* **Path efficiency** and **time to meaningful MFE** are separately causally
  recoverable and complementary in expected-policy-bps space.
* Their simple equal-weight combination improves both residual IC and broad
  tail economics across the holdout and all three 2026 portability folds.
* Invalid paths were removed rather than treated as negative outcomes, so this
  evidence is not driven by a zero-label mass.

What does not work:

* direct final return as currently calibrated;
* stable/frozen or rolling same-bundle P1 GMM geometry as a downstream
  conversion residual;
* treating fold-local GMM component IDs as a pooled inference representation.

## Stages 3--4: causal regimes and causal × path maps

The remaining linked-specification stages were run after the direct controls.
The causal regime view contains 50 point-in-time market/context fields only:
market returns and volatility, breadth, OI/flow, shared correlation state,
liquidation/rebound state, and structural-spectrum quantities.  Candidate-local
quantile and peer fields were deliberately excluded from state definition.

For each outer fold, the state ontology was fit on pre-held decision-time
panels.  The path ontology was a train-only, target-side H12 GMM K8.  A causal
LightGBM predicted its memberships under three contracts: causal 120 fields,
causal+base, and causal+full prequential stack.  Hard J1 and soft J2 maps then
used only held causal state probabilities and held predicted path probabilities:

```text
J2 expected policy net = Σ regime_probability × path_probability
                         × shrunk train-only E[policy net | regime, path]
```

The cell map has a 500-row shrinkage prior towards its train-only regime/path
parents.  No future path coordinate, realised cluster, or outcome enters a
held inference score.

### Regime structural screen

| State arm | Bootstrap ARI mean / minimum | Silhouette | Davies--Bouldin | Tiny clusters |
|---|---:|---:|---:|---:|
| C1 PCA → Ward K4 | 0.802 / 0.586 | 0.273 | 1.431 | 0 |
| C2 PCA → GMM K4 | 0.710 / 0.417 | 0.054 | 3.281 | 0 |
| C2 PCA → GMM K6 | 0.648 / 0.466 | 0.047 | 3.064 | 0 |
| C3 PCA → HDBSCAN | 4–6 persistent states; 4.3–44.3% hard-noise | — | — | 0 |

Ward K4 is the stable causal-state choice.  It was therefore used for the one
bounded direct/J2 complementarity test.  HDBSCAN was run as the explicit C3
control after installing its optional dependency.  Its 30-state/5-neighbour
target-free density topology produced 4–6 persistent states but only modest
cluster persistence (0.052–0.090), and its portable state-only top-5 was
**−81.15 bps/trade** (worst −93.64; residual Spearman 0.051).  It is rejected,
not silently substituted.

### Causal-joint economics

All figures are the same global-within-held-fold diagnostic, mean net
bps/trade.  They are not live admission or portfolio metrics.

| Arm | 2025 Q4 top-1 / top-5 | 2026 top-1 / top-2 / top-5 | 2026 worst top-5 |
|---|---:|---:|---:|
| B0 upstream | −23.59 / −54.58 | +43.86 / −10.67 / −10.67 | −61.15 |
| Direct equal challenger | +104.95 / +19.81 | +181.76 / +135.51 / +65.63 | +30.21 |
| C1 Ward K4 soft J2 | +102.37 / +13.07 | +165.69 / +119.86 / +61.81 | +26.76 |
| C2 GMM K4 soft J2 | +103.27 / +14.83 | +155.33 / +116.72 / +59.40 | +25.68 |
| **50% direct + 50% C1 soft J2** | **+113.65 / +18.87** | **+185.89 / +133.69 / +66.26** | **+30.76** |

The combined score has a small +0.63-bps 2026 top-5 improvement and +0.55-bps
worst-top-5 improvement over the direct challenger, but loses 1.83 bps at
top-2 and slightly weakens the 2025-Q4 top-5 result.  That is not a material,
independent improvement.  It is retained as a diagnostic (`M2_direct50_joint50`)
only; do not add it to MC1, BCF, trust, base, exit, or live inference.

Thus the sequential gate stops J3--J6 and policy-conversion clustering.  The
result does not justify separate hierarchy, early-fusion, or policy-archetype
research on this substrate before a more decisive causal-joint signal exists.

What the result does **not** prove:

* live admissibility, causal EV-map performance, execution-adjusted entry EV,
  portfolio-constrained PnL, or an improvement to the current live stack;
* portability beyond July 2026; the same windows informed arm selection.

## Required next validation

1. Materialise the two direct head scores in the actual current full-stack
   producer with a frozen feature and map contract.
2. Add only the frozen equal-bps blend as a shadow candidate score; do not
   alter live admission, MC1, policy, or execution.
3. Evaluate it against the incumbent on an untouched later period with the
   same causal EV admission and portfolio state.
4. Advance only if it improves admitted/portfolio results without creating a
   side, month, or score-calibration failure.

## Artifacts and scripts

* Labels: `data_perp/artifacts/strict_r3_long_supportive_path_labels_2024_2026_20260823_v5`
* Stage 1: `data_perp/artifacts/strict_r3_long_supportive_label_funnel_stage1_20260823_v6`
* Stable geometry: `data_perp/artifacts/strict_r3_long_supportive_label_stage2_frozen_geometry_20260823_v1`
* Rolling same-bundle geometry: `data_perp/artifacts/strict_r3_long_supportive_label_stage2_rolling_geometry_20260823_v1`
* Direct integration: `data_perp/artifacts/strict_r3_long_direct_support_blends_20260823_v1`
* Causal regimes / J1-J2: `data_perp/artifacts/strict_r3_long_supportive_label_causal_joint_20260823_v2`
* HDBSCAN C3 control: `data_perp/artifacts/strict_r3_long_supportive_label_hdbscan_regime_20260823_v3`
* Direct/J2 complementarity: `data_perp/artifacts/strict_r3_long_direct_joint_blends_20260823_v3`
* Label materialiser: `scripts/materialize_strict_r3_long_supportive_path_labels.py`
* Stage 1 runner: `scripts/run_strict_r3_long_supportive_label_funnel.py`
* Stable/rolling geometry runner: `scripts/run_strict_r3_long_supportive_label_stage2.py`
* Direct-blend runner: `scripts/ablate_strict_r3_long_direct_support_blends.py`
* Causal-regime / joint runner: `scripts/run_strict_r3_long_supportive_label_causal_joint.py`
* Causal-joint auditor: `scripts/audit_strict_r3_long_supportive_label_causal_joint.py`
* HDBSCAN C3 runner: `scripts/run_strict_r3_long_supportive_label_hdbscan_regime.py`
* HDBSCAN C3 auditor: `scripts/audit_strict_r3_long_supportive_label_hdbscan_regime.py`
* Direct/J2 blend runner: `scripts/ablate_strict_r3_long_direct_joint_blends.py`
* Identity/causality auditor: `scripts/audit_strict_r3_long_supportive_label_funnel.py`
