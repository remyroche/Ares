# Enhanced Base, Consensus, and MC1 — Status Report

## Scope and decision status

This is a long-only research status report as of 2026-08-23. It separates
three things that should not be conflated:

1. the sealed **current live** BCF/current dual-MC1 stack;
2. the **enhanced-base** research challenger; and
3. the newly selected **five-head** consensus successor, which is research
   only. Its residual-label control has completed a full downstream
   MC1/portfolio replay; direct policy-conversion variants remain in progress.

All reported policy outcomes use the canonical rich-policy outcome, with path
invalid rows excluded from supervised fitting and terminal evaluation. The
enhanced-base numbers are strict-OOS, but the reported 2025--2026 periods have
been used for research selection. Nothing in this report promotes a change to
the live stack.

> **2026-08-24 contract correction.** The direct-score research artifact
> correctly selected the equal `B0 + efficiency + timing` blend. The first
> downstream challenger reconstructed that source as efficiency/timing 50/50.
> Its downstream Stage-A and direct-policy results are superseded and must not
> be used for model selection. The corrected target-free three-way source is
> `data_perp/artifacts/strict_r3_enhanced_base_threeway_targetfree_20260824_v1/`;
> its coverage is 100% for every frozen base field. The raw base diagnostics
> below come from the sealed three-way direct artifact and remain valid.

## 1. Base layer: current upstream versus enhanced base

### Architecture

The current control is the frozen prequential B0 upstream score.

The enhanced challenger is a common-bps blend of three strictly OOS base
scores:

```text
enhanced_base_bps =
    (predicted path-efficiency policy value
   + predicted time-to-meaningful-MFE policy value
   + frozen prequential B0 upstream value) / 3
```

The efficiency and timing models use the frozen 120-field causal base contract
and are trained only on labels resolved before their held fold. They output
expected policy bps; they do not receive realised paths at inference. The
blend is timestamp-local rank-routed at the top 30% before any consensus work.

The underlying target labels are H12 realised path properties, available at
decision +12 hours. Incomplete paths are target-invalid rather than economic
failures. Each direct score was trained/scored in six chronological outer
folds with month-balanced training caps.

### Strict-OOS base diagnostic

Values are global within-fold tail diagnostics in canonical policy-net bps per
trade; they are not admissions or portfolio returns.

| Held cohort | Base score | Top 1% | Top 5% | Policy-residual Spearman | Worst 2026 top-5 |
|---|---|---:|---:|---:|---:|
| 2025 Q4 | B0 upstream | -23.59 | -54.58 | 0.173* | — |
| 2025 Q4 | Enhanced equal blend | +104.95 | +19.81 | 0.204* | — |
| 2026 portability folds | B0 upstream | +43.86 | -10.67 | 0.173 | -61.15 |
| 2026 portability folds | Enhanced equal blend | +181.76 | +65.63 | 0.204 | +30.21 |

`*` The residual-IC figures are aggregate diagnostic values from the full
outer-fold report; the central comparison is the strict-OOS tail improvement.

The enhanced base improves the 2025-Q4 top-1/top-5 diagnostics by
+128.54/+74.39 bps and the 2026 portability diagnostics by +137.90/+76.30
bps. This is compelling **ranking** evidence, but it is not yet a live-stack
uplift claim.

### Full-stack matched result

The historical ten-head reconstruction remains useful context. The selected
five-head residual-label control below is the first whole-stack replay using
the new research head selector. All deltas are against the same live-like
control on exact common candidate IDs.

| Period | Stack | Accepted trades | Net EV / trade | Total net bps | Worst month | Worst week | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|
| 2025 Q4 | Current live-like control | 2,071 | +140.65 | +291,295 | +98.20 | +30.93 | -29.72% |
| 2025 Q4 | Enhanced base, matched IDs | 1,984 | +145.54 | +288,742 | +100.40 | +45.87 | -27.65% |
| 2026 Apr--Jul | Current live-like control | 2,366 | +139.50 | +330,048 | +113.66 | +88.94 | -21.61% |
| 2026 Apr--Jul | Enhanced base, matched IDs | 2,756 | +124.34 | +342,680 | +92.03 | +75.42 | -25.04% |

**Selected five-head residual-label control** (`policy_net - prequential base
anchor`, edges `[-100,-30,+30,+90]`):

| Period | Stack | Accepted trades | Net EV / trade | Total net bps | Worst month | Worst week | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|
| 2025 Q4 | Current live-like control | 2,071 | +140.65 | +291,295 | +98.20 | +30.93 | -29.72% |
| 2025 Q4 | Enhanced + selected five heads | 1,992 | +144.74 | +288,332 | +103.50 | +38.98 | -28.28% |
| 2026 Apr--Jul | Current live-like control | 2,366 | +139.50 | +330,048 | +113.66 | +88.94 | -21.61% |
| 2026 Apr--Jul | Enhanced + selected five heads | 2,798 | +116.12 | +324,910 | +91.25 | +72.50 | -30.00% |

The selected-five control preserves the 2025-Q4 precision benefit (+4.09
bps/trade), but is materially weaker in 2026 Apr--Jul (-23.37 bps/trade,
-5,138 total bps, and -8.38 pp drawdown). This is the exact conversion
bottleneck the direct policy-net labels must repair; the five-head residual
control itself does not advance.

## 2. Consensus / meta layer

### Role and construction

The consensus layer is a correction to the upstream base, not a second
candidate generator:

```text
enhanced-base top-30% route
  -> selected LambdaRank correction heads
  -> median head rank
  -> upstream = 75% base rank + 25% consensus rank
  -> bounded correctness demotion
```

Each head uses a distinct frozen subset of the 120 causal base fields plus
target-free direct-score disagreement geometry:

```text
base, efficiency, timing,
E-T, E-B0, T-B0, standard deviation(base,E,T),
base timestamp rank and enhanced-base bps.
```

Training is six preceding months, with a 28-day reserve excluded from the
head, policy-map, and correctness fits. LambdaRank queries are either exact
timestamp x long or 4-hour UTC x long, as frozen per head. Each head has the
existing 240,000-row complete-query cap; equal-month heads apply only
train-time month balancing.

The historical implementation actually used ordinal residual edges
`[-100, -30, +30, +90]` bps. An old parent JSON stated a different set of
edges; the research runner now records the executed control explicitly, and
all policy-conversion challengers are hash-bound to their declared labels.

### Selected five-head research contract

| Head | Fields | Query | Training weighting | IC | Top 1% | Top 2% | Top 5% | Positive top-5 months | LOO Δ top-5 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| cap100 ordinary | 100 | exact timestamp | ordinary | 0.029 | +333.82 | +226.62 | +104.97 | 8/10 | +7.02 |
| cap80 ordinary | 80 | exact timestamp | ordinary | 0.023 | +307.55 | +213.55 | +100.31 | 9/10 | +4.15 |
| cap120 equal-month | 51 | exact timestamp | equal month | 0.053 | +298.34 | +184.19 | +80.01 | 9/10 | +3.02 |
| cap40 equal-month | 40 | exact timestamp | equal month | 0.013 | +334.53 | +204.66 | +78.60 | 8/10 | +0.63 |
| cap60 equal-month | 30 | exact timestamp | equal month | 0.011 | +283.37 | +185.12 | +75.49 | 9/10 | +0.98 |

These are strict-OOS individual-head diagnostics on 348,330 valid
enhanced-base-routed rows from 2025-10 through 2026-07. They are **not** a
five-head portfolio result. The five-head selector replaces only research
heads; it does not modify the frozen ten-head live contract.

### Layer economics before the five-head replacement

The earlier ten-head audit illustrates why the consensus contribution is
modest relative to the base:

| Layer | Policy-net rank IC | Top 1% | Top 2% | Top 5% |
|---|---:|---:|---:|---:|
| Median consensus | 0.014 | +330.41 | +200.98 | +74.73 |
| Ordinary-head shadow consensus | 0.014 | +305.94 | +187.43 | +72.60 |
| 75/25 base + consensus upstream | 0.186 | +415.01 | +350.80 | +235.36 |
| Correctness demotion | 0.007 | +274.38 | +187.08 | +85.28 |
| Current pre-MC1 final score | 0.036 | +391.30 | +304.22 | +155.29 |

The consensus has useful but highly correlated information (mean pairwise
correlations for selected heads are about 0.76--0.82). Its job is therefore
to correct local base errors and provide agreement geometry to MC1, not to
replace the stronger base ordering. The selected-five rerun and the
policy-conversion target sweep are the next test of whether that correction
can become materially more useful.

## 3. MC1 absolute-EV admission

### Current executable architecture

The live successor uses two independently mapped score families:

```text
current final score -> current MC1 expected policy net
BCF final score     -> BCF-native MC1 expected policy net
dual admission      = both expected values >= +30 bps
auction priority    = BCF MC1 expected net bps
```

Each MC1 map is a frozen `HistGradientBoostingRegressor` (depth 2, 80
iterations, learning rate 0.04, L2 20, minimum leaf 100, seed 1729) fitted on
a deterministic 50,000-row day-balanced sample from 1,231,050 causal-history
rows. Inputs are six already-causal score and agreement coordinates:
`final_score`, `base_rank42`, `conditional_consensus_rank`, `upstream`,
`ordinary_shadow_consensus_rank`, and `correctness_rank`.

At decision time it adds a causal 21-day robust residual shift, with 10% of
days trimmed from each tail. A policy outcome may enter only once resolved.
Robust-21 remains telemetry; it has no direct admission authority.

### MC1 uplift against Robust-21 control

| 2026 constrained frozen-rank replay | Trades | Trades / day | Net EV / trade | Total net bps | Positive weeks | Worst week | Sortino | Max MTM DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust-21 control | 2,761 | 13.22 | +127.48 | +351,981 | 24/26 | -56.4 | 0.460 | -65.0% |
| MC1_d2 | 3,855 | 18.19 | +155.15 | +598,095 | 31/31 | +1.3 | 0.755 | -38.5% |
| Delta | +1,094 | +4.97 | +27.67 | +246,114 | +7 weeks | +57.7 | +0.295 | +26.5 pp |

The admission-cohort decomposition supports a genuine calibration effect:

| Cohort | Rows | Realised net bps / trade | Total net bps |
|---|---:|---:|---:|
| Both MC1 and Robust-21 | 10,242 | +179.62 | +1,839,670 |
| MC1-only additions | 8,492 | +144.87 | +1,230,269 |
| Robust-21-only omissions | 18,486 | +13.05 | +241,232 |

Within frozen-score bands, agreement-to-realised-EV Spearman was roughly
`+0.76`, positive in 9 of 10 bands. MC1 is consequently the layer with the
strongest demonstrated absolute-admission uplift; it should remain fixed while
the upstream and consensus research is evaluated.

## 4. Current challenges and gates

| Challenge | Evidence | Consequence / next gate |
|---|---|---|
| Enhanced base has not translated proportionally to constrained PnL | Large base tail uplift, but 2026 matched stack loses 15.16 bps/trade | Retrain the five selected heads and MC1 maps from enhanced outputs; compare on a common, fixed portfolio replay. |
| Five-head residual control is not portable enough | It improves 2025 Q4 but loses 23.37 bps/trade, 5,138 total bps and 8.38 pp drawdown in 2026 Apr--Jul | Advance a direct policy-conversion label only if it repairs the 2026 loss without sacrificing total/risk-adjusted contribution. |
| Historical consensus target receipt was inconsistent | Code ran -100/-30/+30/+90 residual bands; parent JSON claimed -150/-50/+50/+150 | The new runner records the actual control and uses declared conversion-label variants. Historical metrics remain code-valid but metadata must not be reused blindly. |
| H12 supportive path coverage is incomplete | 1,663,035 / 2,820,951 rows (58.95%) in the original direct-label substrate | Invalid rows are correctly excluded, but label support is reduced. The coverage repair must complete before expanding path-archetype targets. |
| MC1’s 2026 record is not untouched promotion evidence | It participated in mapper selection | Freeze the existing MC1; evaluate any new upstream/consensus stack on a later untouched period. |
| Layer interactions are compressed | 75/25 blend, correctness demotion, dual MC1 gates, and auction can attenuate a base gain | Report waterfall metrics on identical candidate IDs: base route, consensus rerank, correctness, each MC1 gate, and accepted portfolio cohort. |
| Live stack is not the enhanced challenger | Live uses frozen current-v5 + BCF dual MC1 scorer and ten-head BCF bundle | No research artifact may replace live scoring until model, map, and policy are resealed and forward-validated. |

## 5. Current policy-conversion-label ablation

## 5A. Replacement meta funnel — initial Stage A waterfall (superseded)

The broad direct-label sweep was superseded on 2026-08-24 by a sequential
funnel. The first implementation of this waterfall consumed a two-way
efficiency/timing reconstruction. Preserve its receipts for implementation
audit, but do **not** use the table below to select an architecture. The
three-way replacement holds the same reserve, MC1 class, dual `>= +30 bps`
admission, BCF-MC1 auction priority and constrained portfolio rules fixed.

| Arm | Five heads | Generic correctness | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | 2026 worst month / week | 2026 DD |
|---|---|---|---:|---:|---:|---:|
| A0 base only | no | no | 1,994 / +127.15 | 2,908 / +102.28 | +83.33 / +66.67 | -27.73% |
| A2 full control | yes | yes | 1,992 / +144.74 | 2,798 / +116.12 | +91.25 / +72.50 | -30.00% |
| **A3 selected** | **yes** | **no** | **2,009 / +145.38** | **2,781 / +117.69** | **+95.88 / +80.69** | **-30.00%** |

All EVs are policy-net bps per accepted trade. These figures are retained
solely to debug the reconstruction mismatch; they make no advancement claim.
The corrected three-way A0 control is complete, while matching A2/A3 controls
are running. Only their matched result will select the Stage-B architecture.

Stage-A causal receipts:

- A0: `data_perp/artifacts/strict_r3_enhanced_base_meta_correctness_waterfall_20260824_a0_base_only_v4/`
- A2: `data_perp/artifacts/strict_r3_enhanced_base_policy_conversion_labels_20260823_control_v1/`
- A3: `data_perp/artifacts/strict_r3_enhanced_base_meta_correctness_waterfall_20260824_a3_no_correctness_v1/`

The A0 runner also exposed a terminal reporting issue: an empty grade-count
struct could not be serialised by Parquet, and the original terminal stage
held multiple million-row panels simultaneously.  The runner now stores an
explicit `not_applicable` grade marker for A0 and performs a memory-safe
receipt resume: score panels stay immutable, each MC1 family is persisted and
released before the other is fitted, and final replay reads compact receipts.
Focused invariants pass after this repair.

## 5B. Replacement meta funnel — corrected three-way Stage A

The corrected waterfall uses the frozen target-free source with

```text
enhanced_base_bps = (B0 + efficiency + timing) / 3
```

for every candidate, before timestamp-local top-30% routing. The source has
120/120 frozen base fields at 100% coverage in every relevant month, no policy
outcome column, and a maximum blend arithmetic discrepancy of `1.3e-05` bps.
The 28-day reserve, five-head contract, MC1 class, dual `>= +30 bps` gate and
global constrained auction are otherwise identical across arms.

| Corrected arm | Heads | Generic correctness | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | 2026 worst month / week | 2026 DD |
|---|---|---|---:|---:|---:|---:|
| A0 base only | no | no | 2,013 / +125.78 | 2,910 / +103.04 | +86.18 / +61.15 | -28.28% |
| **A2 selected** | **yes** | **yes** | **1,893 / +150.18** | **2,795 / +123.57** | **+95.00 / +76.10** | **-25.75%** |
| A3 no correctness | yes | no | 2,001 / +140.91 | 2,820 / +121.23 | +92.06 / +65.37 | -25.72% |

All EVs are canonical policy-net bps per accepted trade. A2 beats A3 by
`+9.27` bps/trade and `+2,332` total bps in 2025 Q4, and by `+2.33` bps/trade
and `+3,495` total bps in 2026 Apr--Jul. It also improves the worst month and
week. A2 therefore is the sole upstream architecture allowed into Stage B.
It still does not promote over the frozen live-like baseline: the corrected
2026 stack is `+123.57` bps/trade versus baseline `+139.50`; target and
integration changes must repair that gap before any challenger can advance.

Corrected receipts:

- source: `data_perp/artifacts/strict_r3_enhanced_base_threeway_targetfree_20260824_v1/`
- A0: `data_perp/artifacts/strict_r3_enhanced_base_meta_correctness_waterfall_20260824_threeway_a0_base_only_v1/`
- A2: `data_perp/artifacts/strict_r3_enhanced_base_meta_correctness_waterfall_20260824_threeway_a2_full_correctness_v2/`
- A3: `data_perp/artifacts/strict_r3_enhanced_base_meta_correctness_waterfall_20260824_threeway_a3_no_correctness_v1/`

The selected-five research runner now supports three strictly resolved-policy
targets, none of which are inference inputs:

| ID | Target | Grade boundaries (bps) | Role |
|---|---|---|---|
| Control | policy net - prequential base anchor | -100, -30, +30, +90 | Exact historical executed residual target |
| C1 | direct canonical policy net | -100, -30, +30, +90 | Tests whether residualisation discards useful conversion signal |
| C2 | direct canonical policy net | -200, 0, +50, +150 | Severe loss / loss / marginal clear / robust clear |

Each fold will use only policy labels resolved before its 28-day reserve,
persist target-free OOS scores before joining outcomes, refit both MC1 families
strictly prequentially, and evaluate the normal dual-MC1 admission and
constrained portfolio. The direct-policy target previously beat the simpler
residual target only modestly in a B0-only meta test; it must now prove itself
under the enhanced base and selected five heads.

### 5C. Stage B — policy-conversion target decision

All Stage-B arms used the corrected three-way enhanced-base source, the
selected five heads, generic correctness, the same 28-day reserve, two
prequential MC1 maps, dual `>= +30 bps` admission, and the same global
constrained auction. Therefore the table isolates the head target/objective.

| Arm | Consensus target / objective | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | 2026 worst month / week | 2026 DD | Decision |
|---|---|---:|---:|---:|---:|---|
| **A2 control** | Historical ordinal residual around the prequential B0 anchor | **1,893 / +150.18** | **2,795 / +123.57** | **+95.00 / +76.10** | -25.75% | **Advance** |
| R1 | Direct policy net, ordinal LambdaRank | 2,054 / +136.69 | 2,828 / +123.21 | +104.38 / +84.97 | -23.94% | Reject: loses EV/trade in both periods |
| R2 | `policy_net - enhanced_base_bps`, clipped +/-500, L2 | 1,960 / +141.14 | 2,876 / +123.55 | +95.79 / +66.72 | -26.18% | Reject: no material 2026 gain; weaker 2025 |
| R3 | Same enhanced residual, clipped +/-500, ordinal LambdaRank | 2,004 / +142.59 | 2,780 / +122.02 | +90.93 / +72.47 | -26.69% | Reject: weaker EV and risk in 2026 |
| R4 | Same enhanced residual, clipped +/-500, Huber regression | 1,861 / +139.44 | 2,932 / +100.34 | +76.20 / +53.13 | -23.12% | Reject: material 2026 deterioration |

R1 and R2 obtain a little more *total* 2026 bps by admitting materially more
trades, but neither meets the predeclared portability condition: they do not
improve per-trade EV and robust monthly/week economics together. No broad
policy-conversion *target* advances. The next, more narrowly scoped test was
pairwise correction: can the existing heads identify economically meaningful
inversions among near-tied enhanced-base candidates without taking broad
independent ranking authority?

### 5D. Stage C — policy-conversion pairwise correction

P0 is the selected A2 ordinal-residual control. P1--P4 retain the exact same
target-free three-way base source, five heads, generic correctness demotion,
28-day reserve, paired prequential MC1 maps, dual `>= +30 bps` admission, and
constrained portfolio. They change only the supervised pair population for the
five LambdaRank heads. A pair is always formed within one decision timestamp
and the already routed enhanced-base universe. Its two-row target is the
ordering of **resolved canonical policy net**, available strictly before the
fold reserve; it is never an inference input.

| Arm | Head supervision | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | 2026 worst month / week | 2026 DD | Decision |
|---|---|---:|---:|---:|---:|---|
| P0 | Historical ordinal residual control | 1,893 / +150.18 | 2,795 / +123.57 | +95.00 / +76.10 | -25.75% | Control |
| P1 | All adjacent base-near-ties | 2,044 / +140.89 | 2,701 / +130.03 | +102.91 / +82.41 | -24.48% | Reject: weak Q4 EV |
| **P2** | **Base-near-ties with absolute policy-net disagreement >50 bps** | **2,001 / +154.56** | **2,633 / +133.45** | **+105.84 / +82.06** | **-25.70%** | **Advance** |
| P3 | Base-near-ties with disagreement >100 bps | 1,933 / +152.75 | 2,701 / +129.64 | +106.22 / +76.36 | -24.86% | Reject: less 2026 EV than P2 |
| P4 | Large enhanced-base inversions, policy-net disagreement >100 bps | 2,011 / +136.62 | 2,871 / +105.06 | +84.59 / +61.99 | -25.92% | Reject |

P2 is the first policy-conversion arm that passes the two-era advancement
gate. Against P0 it adds `+4.38` bps/trade and `+24,982` total bps in 2025 Q4;
in 2026 Apr--Jul it adds `+9.88` bps/trade, `+5,997` total bps, `+10.84` bps to
the worst month and `+5.97` bps to the worst week. Its only trade-off is a
`0.21 pp` worse Q4 drawdown; 2026 drawdown is marginally better. It reduces
2026 accepted trades by 162 (5.8%) while raising both per-trade and aggregate
economics, which is acceptable under the predeclared capital-efficiency gate.

P2 becomes the Stage-C research winner. It remains a **research challenger**:
no live contract, admission threshold, MC1 mapping, exit policy, or portfolio
constraint was changed. The next test is whether a bounded bps-residual
integration can add value on top of P2 without surrendering its ranking
stability.

### 5E. Stage D — bounded common-bps residual integration

P2's ordinary and conditional consensus scores were mapped to a policy
residual only from each fold's **prior resolved 28-day reserve**.  The held
month was never used to fit that map.  The resulting residual was added to the
prequential base anchor with the predeclared bounded authorities below; every
other component (P2 pairwise heads, generic correctness, paired MC1 maps,
dual `>= +30 bps` admission and auction) was unchanged.

| Arm | Integration | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | 2026 worst month / week | 2026 DD | Decision |
|---|---|---:|---:|---:|---:|---|
| **D0** | **P2 rank blend control** | **2,001 / +154.56** | **2,633 / +133.45** | **+105.84 / +82.06** | **-25.70%** | **Retain** |
| D1 | `base_anchor + 0.25 × residual_bps` | 2,161 / +143.42 | 2,840 / +126.88 | +100.55 / +73.49 | -27.44% | Reject |
| D2 | `base_anchor + 0.50 × residual_bps` | 2,153 / +144.12 | 2,832 / +127.10 | +101.28 / +78.14 | -25.77% | Reject |
| D3 | `base_anchor + clip(residual, -50, +50)` | 2,176 / +139.16 | 2,952 / +122.99 | +97.78 / +78.47 | -26.32% | Reject |
| D4 | `base_anchor + clip(0.5 × residual, -100, +100)` | 2,166 / +144.28 | 2,861 / +126.93 | +101.11 / +73.22 | -25.74% | Reject |

No bps integration advances.  Every alternative increases participation but
reduces EV per trade in both evaluation blocks.  Some raise total 2026 bps,
but none improve the joint EV / worst-period / drawdown gate.  The consensus
therefore remains a **bounded rank-space correction** rather than an absolute
EV adjustment.

### 5F. Policy-native conversion-label extension

The final direct-economic target is five ordered canonical-policy-net grades:

```text
net <= -200 | -200..0 | 0..+50 | +50..+150 | >+150 bps
```

It directly isolates the admission hurdle and robust conversion.  Initial
execution through P2 was intentionally identical to P2 because a near-tie
pairwise head has its own fixed target: resolved policy-net ordering within
the pair.  That run is retained only as an invariance receipt, not counted as
a label result.  The runner now rejects any non-control conversion-label arm
combined with pairwise mode, and the corresponding focused test passes.

The valid R5 run uses `pairwise_mode=none`, so its five heads genuinely fit the
five economic grades.  It preserves the corrected three-way base, selected
five-head inputs, correctness, strict prequential MC1 maps, dual admission
and one constrained global auction.

| Arm | 2025 Q4: trades / EV / total bps | 2026 Apr--Jul: trades / EV / total bps | 2026 worst month / week | 2026 DD | Decision |
|---|---:|---:|---:|---:|---|
| **P2 pairwise residual control** | **2,001 / +154.56 / +309,276** | **2,633 / +133.45 / +351,362** | **+105.84 / +82.06** | **-25.70%** | **Retain** |
| R5 direct economic grades | 2,126 / +133.95 / +284,785 | 2,831 / +128.51 / +363,798 | +108.45 / +84.81 | -19.95% | Reject as primary; retain as risk/participation reference |

R5 improves 2026 total bps, worst month, worst week and drawdown by admitting
more candidates, but loses `20.61` bps/trade and `24,491` total bps in Q4 and
`4.94` bps/trade in 2026.  It consequently does not displace P2 under the
predeclared two-era efficiency gate.  The result is useful diagnostically: a
direct conversion objective is a more conservative, higher-participation
policy selector, but it is not the best correction of the already strong
enhanced-base ordering.

## Receipts

- `docs/STRICT_R3_LONG_SUPPORTIVE_LABEL_ABLATION_20260823.md`
- `docs/ENHANCED_BASE_LIVE_STACK_CHALLENGER_AUDIT_20260823.md`
- `data_perp/artifacts/strict_r3_enhanced_base_live_stack_challenger_20260823_v10/`
- `data_perp/artifacts/strict_r3_enhanced_base_consensus_head_audit_20260823_v2/`
- `data_perp/artifacts/strict_r3_long_direct_support_blends_coverage_repaired_20260823_v2/`
- `data_perp/artifacts/strict_r3_enhanced_base_meta_stagec_20260824_p2_near_tie_diff50_v1/`
- `data_perp/artifacts/strict_r3_enhanced_base_meta_stagec_20260824_p1_near_tie_v2/`
- `data_perp/artifacts/strict_r3_enhanced_base_meta_stagec_20260824_p3_near_tie_diff100_v1/`
- `data_perp/artifacts/strict_r3_enhanced_base_meta_stagec_20260824_p4_base_inversion100_v1/`
- `data_perp/artifacts/strict_r3_enhanced_base_meta_staged_20260824_p2_additive025_v1/`
- `data_perp/artifacts/strict_r3_enhanced_base_meta_staged_20260824_p2_additive050_v1/`
- `data_perp/artifacts/strict_r3_enhanced_base_meta_staged_20260824_p2_clip50_v1/`
- `data_perp/artifacts/strict_r3_enhanced_base_meta_staged_20260824_p2_cliphalf100_v1/`
- `data_perp/artifacts/strict_r3_enhanced_base_meta_stageb_20260824_threeway_r5_direct_economic_v2/`
- `config/strict_r3_enhanced_base_consensus_top5_v1.json`
- `scripts/run_strict_r3_enhanced_base_live_stack_challenger.py`
- `scripts/audit_strict_r3_enhanced_base_consensus_heads.py`
