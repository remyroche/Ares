# Enhanced Base, Consensus and MC1 — Decision Report

**Scope:** long-only, offline research as of 2026-08-24.  This report
separates strict-OOS *ranking diagnostics* from the causal, dual-MC1,
portfolio-constrained replay.  The enhanced-base stack remains a challenger;
no result below changes the live stack.

## Executive read

1. The new base is materially stronger than B0 at ranking policy outcomes.
   Its OOS top-5 policy net is positive in the held 2025-Q4 and 2026
   portability blocks where B0's is negative.
2. The consensus layer remains a modest correction, not an independent alpha
   source.  The most useful use so far is a pairwise correction of
   **base-near-ties whose realised policy outcomes differ by more than 50 bps**.
3. MC1 is the strongest proven *absolute-admission* improvement.  It uses
   score/agreement geometry to recover good opportunities rejected by the
   rolling historical map and to remove weak opportunities it admits.
4. The complete enhanced stack is not yet a promotion candidate: it trails the
   current live-like control in 2026 per-trade EV and drawdown, despite higher
   total 2026 bps.  The open problem is conversion of stronger ranking into
   robustly better causal admissions and portfolio economics.

## 1. Base layer — B0 control versus the new enhanced base

### How they work

| Layer | B0 control | Enhanced-base challenger |
|---|---|---|
| Input contract | Frozen 120-field causal base contract | The same frozen 120 causal fields |
| Outputs | Prequential B0 expected-policy value | Three strictly OOS expected-policy-bps scores |
| Composition | One upstream value | `(B0 + path-efficiency + time-to-meaningful-MFE) / 3` |
| Routing | Timestamp-local top 30% | The same timestamp-local top 30% route |
| Labels | Existing strict-R3 opportunity labels | H12 path efficiency and time-to-meaningful-MFE; each becomes usable only at decision +12h |
| Training | Chronological, pre-resolved labels | Six chronological outer folds, at most 180,000 month-balanced training rows per fold |

Incomplete H12 paths are **target-invalid**, not negative examples.  The
direct-label substrate currently contains 1,663,035 valid complete paths out
of 2,820,951 candidates (58.95%).  This is a real coverage limitation, not a
zero-label class.

### Strict-OOS base-only diagnostics

These are global within-held-fold tail diagnostics in canonical policy-net
bps/trade.  They measure ranking only: they are not an admission rule,
execution simulation, or portfolio return.

| Held period | Score | Top 1% | Top 5% | Residual Spearman | Worst 2026 top-5 |
|---|---|---:|---:|---:|---:|
| 2025 Q4 | B0 | -23.59 | -54.58 | 0.173* | — |
| 2025 Q4 | Enhanced 1/3–1/3–1/3 | +104.95 | +19.81 | 0.204* | — |
| 2026 portability folds | B0 | +43.86 | -10.67 | 0.173 | -61.15 |
| 2026 portability folds | Enhanced 1/3–1/3–1/3 | +181.76 | +65.63 | 0.204 | +30.21 |

The new base's improvement versus B0 is +128.54 / +74.39 bps at 2025-Q4
top-1 / top-5, and +137.90 / +76.30 bps across the 2026 portability folds.
This is the clearest result in the current research sequence: the two
path-property targets add genuine, complementary ranking information.

## 2. Consensus / meta layer

### Current research construction

```text
enhanced-base top-30% timestamp route
  -> five LambdaRank correction heads
  -> median consensus rank
  -> 75% base rank + 25% consensus rank
  -> bounded correctness demotion
  -> two prequential MC1 maps and dual admission
```

Every head is trained on preceding resolved policy outcomes with a 28-day
reserve excluded from the head, policy-map, and correctness fits.  It uses a
complete-query cap of 240,000 rows; queries are frozen as exact timestamp ×
long or 4-hour UTC × long.  Head inputs comprise a different frozen subset of
the 120 causal base fields plus target-free disagreement geometry:

```text
B0, efficiency, timing, E-T, E-B0, T-B0,
std(B0,E,T), base timestamp rank, enhanced-base bps.
```

### Stage A — does generic correctness help?

The first waterfall kept the same three-way enhanced base, five residual
heads, fixed MC1 class, dual admission and constrained replay.  It varied only
the generic correctness authority.  This answers the specification's initial
question before the later P2 target was selected.

| Arm | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | 2026 worst month / week | 2026 DD | Result |
|---|---:|---:|---:|---:|---|
| A0 base-only | 1,994 / +127.15 | 2,908 / +102.28 | +83.33 / +66.67 | -27.73% | Insufficient downstream conversion |
| A2 five heads + generic correctness | **1,893 / +150.18** | **2,795 / +123.57** | **+95.00 / +76.10** | -25.75% | Retained control |
| A3 five heads, no correctness | 2,001 / +140.91 | 2,820 / +121.23 | +92.06 / +65.37 | -25.72% | Reject |

Contrary to the initial concern, generic correctness is modestly beneficial
within the pre-P2 control: removing it lowers EV/trade in both blocks and
weakens the 2026 worst period.  It remains T1, a **bounded demotion**, rather
than an admission-expansion mechanism.

### Stages B--D — target, near-tie objective and bps integration

All variants below refit MC1 prequentially and preserve the same dual admission
and constrained portfolio.  The matched results establish three decisions:

* Direct-policy, enhanced-base L2 residual, ordinal residual and Huber
  residual targets all fail the joint two-era precision gate.
* The target can be useful only when it is directed at errors the base cannot
  resolve: P2's base-near-tie pairs with a realised difference above 50 bps.
* The residual should remain a **rank-space correction**.  Additive bps
  integrations increase participation but dilute EV/trade and worsen
  worst-period/drawdown measures.

| Stage / representative arm | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | 2026 worst month / week | Decision |
|---|---:|---:|---:|---|
| B1 direct policy | 2,054 / +136.69 | 2,828 / +123.21 | +104.38 / +84.97 | Reject |
| B2 enhanced-base continuous residual | 1,960 / +141.14 | 2,876 / +123.55 | +95.79 / +66.72 | Reject |
| B3 enhanced-base ordinal residual | 2,004 / +142.59 | 2,780 / +122.02 | +90.93 / +72.47 | Reject |
| B4 enhanced-base Huber residual | 1,861 / +139.44 | 2,932 / +100.34 | +76.20 / +53.13 | Reject |
| **C P2 near-tie, >50 bps** | **2,001 / +154.56** | **2,633 / +133.45** | **+105.84 / +82.06** | **Advance** |
| D additive 0.25 residual bps | 2,161 / +143.42 | 2,840 / +126.88 | +100.55 / +73.49 | Reject |
| D additive 0.50 residual bps | 2,153 / +144.12 | 2,832 / +127.10 | +101.28 / +78.14 | Reject |
| D bounded residual variants | 2,166--2,176 / +139--144 | 2,861--2,952 / +123--127 | +98--104 / +73--78 | Reject |

### Selected five-head contract

| Head | Fields | Query | Train weighting | IC | Top 1% | Top 2% | Top 5% | Positive top-5 months | LOO Δ top-5 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| cap100 ordinary | 100 | timestamp | ordinary | 0.029 | +333.82 | +226.62 | +104.97 | 8/10 | +7.02 |
| cap80 ordinary | 80 | timestamp | ordinary | 0.023 | +307.55 | +213.55 | +100.31 | 9/10 | +4.15 |
| cap120 equal-month | 51 | timestamp | equal-month | 0.053 | +298.34 | +184.19 | +80.01 | 9/10 | +3.02 |
| cap40 equal-month | 40 | timestamp | equal-month | 0.013 | +334.53 | +204.66 | +78.60 | 8/10 | +0.63 |
| cap60 equal-month | 30 | timestamp | equal-month | 0.011 | +283.37 | +185.12 | +75.49 | 9/10 | +0.98 |

These are strict-OOS standalone diagnostics on 348,330 valid routed rows
from 2025-10 through 2026-07.  They cannot be read as five-head portfolio
performance.  Their pairwise score correlations are high (roughly
0.76–0.82), so the consensus has limited independent capacity.

### Label and integration ablations

The ordinary residual control uses `policy_net - prequential B0 anchor`,
ordinalised with edges `[-100,-30,+30,+90]` bps.  Direct-policy and
common-bps residual targets did not improve the joint two-era gate.  The
winner is instead a *limited pairwise policy-conversion correction*:

```text
P2: train only on base-near-tied candidates where
    |realised canonical policy net difference| > 50 bps.
```

It can correct which of two nearly equally rated opportunities should rank
ahead, but does not claim broad absolute-EV authority.

| Arm | 2025 Q4: accepted / EV | 2026 Apr–Jul: accepted / EV | 2026 worst month / week | 2026 max DD | Decision |
|---|---:|---:|---:|---:|---|
| A2 residual-label control | 1,893 / +150.18 | 2,795 / +123.57 | +95.00 / +76.10 | -25.75% | Control |
| **P2 near-tie, >50 bps** | **2,001 / +154.56** | **2,633 / +133.45** | **+105.84 / +82.06** | **-25.70%** | **Research winner** |
| R5 direct economic grades | 2,126 / +133.95 | 2,831 / +128.51 | +108.45 / +84.81 | -19.95% | Risk/participation reference |

Relative to A2, P2 adds +4.38 bps/trade and +24,982 total bps in 2025-Q4;
and +9.88 bps/trade and +5,997 total bps in 2026 Apr–Jul.  It reduces 2026
trade count by 162 (5.8%) while improving both per-trade and aggregate
economics.  Additive residual-bps integrations all increased participation
but worsened EV/trade, so the consensus should remain a rank-space correction.

### Whole-stack comparison to the current live-like control

This is the relevant comparison, using common candidate IDs, retrained
downstream layers, dual MC1 admission, and one global constrained portfolio.

**Baseline definition:** `live_baseline` is the current live-stack score
family under the same evaluation substrate: its current/BCF score receipts,
frozen MC1 maps, dual `>= +30 bps` admission, BCF-priority auction, 7x/10%
slot/80%-margin constrained portfolio, and identical policy outcomes.  It is
therefore the only valid baseline for a challenger delta; unbounded
full-universe coverage outputs are not used below.

| Baseline period | Trades | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| Current live stack, 2025 Q4 | 2,071 | +140.65 | +291,295 | +98.20 | +30.93 | -29.72% |
| Current live stack, 2026 Apr--Jul | 2,366 | +139.50 | +330,048 | +113.66 | +88.94 | -21.61% |

| Period | Current live-like control: trades / EV | P2 enhanced challenger: trades / EV | Δ trades | Δ EV/trade | Δ total bps | Δ worst month | Δ worst week | Δ max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 Q4 | 2,071 / +140.65 | 2,001 / +154.56 | -70 | +13.91 | +17,980 | +16.30 | +45.47 | -1.63 pp |
| 2026 Apr–Jul | 2,366 / +139.50 | 2,633 / +133.45 | +267 | -6.05 | +21,314 | -7.82 | -6.87 | -4.09 pp |

The challenger gains total bps in both blocks, but its 2026 precision,
worst-period results and drawdown are poorer.  It is not promotable yet.

### Every retained downstream candidate versus the current live baseline

This comparison is deliberately kept in the report because a candidate can
look attractive against its immediate predecessor while still be worse than
the current live stack.  Positive drawdown deltas would be an improvement;
negative values below mean a deeper drawdown.

| Arm | Period | Δ trades | Δ EV/trade | Δ total net bps | Δ worst month | Δ worst week | Δ max DD | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P2/T1 | 2025 Q4 | -70 | **+13.91** | **+17,980** | **+16.30** | **+45.47** | -1.63 pp | Challenger gain |
| P2/T1 | 2026 Apr--Jul | +267 | **-6.05** | +21,314 | -7.82 | -6.87 | -4.09 pp | Not portable |
| H1 semantic median | 2025 Q4 | -83 | +12.66 | +13,497 | +14.71 | +26.38 | -1.66 pp | Reject |
| H1 semantic median | 2026 Apr--Jul | +356 | -10.98 | +19,772 | -15.69 | -13.62 | -4.48 pp | Reject |
| H2 quality × independence | 2025 Q4 | -72 | +10.34 | +10,535 | +7.15 | +37.46 | -1.66 pp | Reject |
| H2 quality × independence | 2026 Apr--Jul | +427 | -12.38 | +24,990 | -11.45 | -9.70 | -3.58 pp | Reject |

## 3. MC1 admission — setup and demonstrated uplift

### What MC1 does

MC1 converts a frozen score family into an *absolute expected policy-net EV*,
which is used for admission rather than simply ranking the cross-section.

```text
current final score -> current MC1 expected policy net
BCF final score     -> BCF-native MC1 expected policy net
admit only if both maps >= +30 bps
auction priority = BCF MC1 expected policy net
```

Each score-family map is a frozen `HistGradientBoostingRegressor`:

| Parameter | Value |
|---|---:|
| Depth | 2 |
| Iterations | 80 |
| Learning rate | 0.04 |
| L2 | 20 |
| Minimum leaf | 100 |
| Seed | 1729 |
| Training sample | 50,000 deterministic day-balanced rows from 1,231,050 causal-history rows |

Its six inputs are already causal and prequential:
`final_score`, `base_rank42`, `conditional_consensus_rank`, `upstream`,
`ordinary_shadow_consensus_rank`, and `correctness_rank`.

At inference it applies a causal 21-day robust residual shift with the upper
and lower 10% of days trimmed.  Only fully resolved policy outcomes can enter
that shift.  Robust-21 is telemetry, not an admission authority.

### Frozen MC1_d2 evidence versus the Robust-21 control

| 2026 constrained frozen-rank replay | Trades | Trades/day | Net EV/trade | Total net bps | Positive weeks | Worst week | Sortino | Max MTM DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust-21 | 2,761 | 13.22 | +127.48 | +351,981 | 24/26 | -56.4 | 0.460 | -65.0% |
| **MC1_d2** | **3,855** | **18.19** | **+155.15** | **+598,095** | **31/31** | **+1.3** | **0.755** | **-38.5%** |
| **MC1 uplift** | **+1,094** | **+4.97** | **+27.67** | **+246,114** | **+7 weeks** | **+57.7** | **+0.295** | **+26.5 pp** |

The supporting cohort analysis is unusually strong:

| Admission cohort | Rows | Realised net bps/trade | Total net bps |
|---|---:|---:|---:|
| Both MC1 and Robust-21 admit | 10,242 | +179.62 | +1,839,670 |
| MC1-only additions | 8,492 | +144.87 | +1,230,269 |
| Robust-21-only omissions | 18,486 | +13.05 | +241,232 |

Within frozen-score bands, agreement-to-realised-EV Spearman is approximately
`+0.76`, positive in 9 of 10 bands.  That is evidence that MC1 is adding
cross-sectional calibration information, not merely learning a nonlinear
transformation of the final score.

The MC1 artifact is held fixed during enhanced-base research.  Its general
uplift is therefore established; a clean *incremental MC1 attribution for the
P2 enhanced stack* has not yet been isolated and should not be inferred from
the tables above.

## 4. Current challenges and decision gates

| Challenge | Evidence | Required next evidence |
|---|---|---|
| Stronger base has not become a uniformly stronger portfolio | Enhanced P2 loses 6.05 bps/trade, 7.82 bps on worst month and 4.09 pp DD in 2026 despite better raw ranking | A downstream head/admission design that improves 2026 precision and risk without sacrificing Q4 gains |
| Consensus heads are correlated | 0.76–0.82 mean pairwise correlations | Test only compact feature contracts and policy-conversion labels that create complementary correction information; do not add more similar heads |
| Label support is incomplete | Only 58.95% complete H12 path rows | Repair historical path availability; exclude invalid paths everywhere until it is complete |
| Direct economic grades trade more but dilute EV | R5 adds 198 2026 entries but trails P2 by 4.94 bps/trade | Use it only as a risk/participation control, not as the primary consensus target |
| MC1 selection evidence is no longer untouched | 2026 informed mapper selection | Keep MC1 frozen; require later untouched forward confirmation for any upstream/consensus successor |
| Heterogeneous geometry/context features may be useful but are not yet validated | The compact Stage-E variants and Stage-G second-layer variants did not pass the two-era gate | Advance only with a strict target-free feature contract, fold-trained transforms, stable semantics, and a common portfolio replay |
| Live and research stacks differ | Live remains the sealed BCF/current dual-MC1 and ten-head contract | Do not transplant the challenger into live inference before resealing and forward validation |

## 5. Stage E — compact residual-feature contract results

All Stage-E arms use the selected P2 near-tie policy-conversion target, the
same three-way enhanced base, five selected heads, 75/25 rank blend,
correctness demotion, frozen MC1 class, dual `>= +30 bps` admission, and one
global constrained portfolio.  They change **only** the residual-head input
contract.  Target-free coverage was 100% in every scored month and held
policy outcomes were joined only after score receipts were persisted.

| Contract | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | Δ total bps vs live-like control (Q4 / 2026) | 2026 risk result | Decision |
|---|---:|---:|---:|---|---|
| Live-like common-ID control | 2,071 / +140.65 | 2,366 / +139.50 | — | worst month +113.66; worst week +88.94; DD -21.61% | Control |
| F1 score/disagreement geometry | 1,984 / +130.29 | 2,856 / +104.07 | -32,808 / -32,831 | worst month -32.57 bps; week -40.37; DD -7.40 pp | Reject |
| F2 geometry + score-space support/OOD | 1,911 / +138.18 | 2,765 / +108.89 | -27,243 / -28,973 | worst month -26.67; week -20.67; DD -4.31 pp | Reject |
| F3 geometry + causal recent calibration | 1,942 / +132.63 | 2,852 / +101.55 | -33,721 / -40,433 | worst month -36.05; week -32.50; DD -7.86 pp | Reject |
| F4 geometry + support/OOD + state | 1,879 / +133.82 | 2,793 / +108.19 | -39,850 / -27,871 | worst month -28.55; week -28.64; DD -5.30 pp | Reject |
| F5 all compact blocks + 25 fixed raw context fields | 2,039 / +147.32 | 2,758 / +129.55 | +9,089 / +27,264 | 2026 per-trade EV -9.94 bps; worst month -11.65; week -6.41; DD -1.35 pp | Reject: not portable enough |

F5 is the only compact-contract arm with higher total bps in both periods,
but it buys that 2026 aggregate gain with 392 additional entries and lower
per-trade EV, a weaker worst month/week, and a larger drawdown.  It therefore
does not meet the predeclared portability gate.  The residual feature contract
does **not** advance; the current selected five-head inputs remain the
research control for the next asymmetric-tail-trust stage.

## 6. Stage F — asymmetric tail-trust ablation

This stage holds the corrected three-way enhanced base, P2 near-tie target,
five selected residual heads, 75/25 rank blend, retrained MC1 class, dual
`>= +30 bps` admission, and constrained portfolio fixed.  Each arm can only
demote an existing upstream score; it cannot create an admission.  Every trust
label uses the policy residual versus the strict-OOS enhanced base and only
prior-resolved outcomes.  The maximum demotion is 10%.

| Trust arm | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | 2026 worst month / week | 2026 DD | Decision |
|---|---:|---:|---:|---:|---|
| T0 none | 2,121 / +145.47 | 2,782 / +129.64 | +105.11 / +78.48 | -23.92% | Reject |
| **T1 generic correctness** | **2,001 / +154.56** | **2,633 / +133.45** | **+105.84 / +82.06** | **-25.70%** | **Advance** |
| T2 P(residual <= -100) | 2,119 / +147.68 | 2,759 / +130.01 | +103.95 / +75.70 | -22.94% | Reject |
| T3 residual q20 | 2,130 / +144.31 | 2,760 / +131.35 | +106.22 / +76.54 | -23.87% | Reject |
| T4 P(residual <= -100) + support gate | 2,115 / +146.10 | 2,759 / +128.75 | +103.87 / +76.18 | -22.94% | Reject |

T1 has lower participation, but it is the only arm that leads every tail
substitute in both periods on per-trade EV and also has the strongest 2026
worst-month/week result.  The asymmetric controls add trades and total bps in
some cases, but buy those additions with lower precision and weaker
portability.  Do not combine T1 with a tail substitute.  Stage G therefore
starts from the P2/T1 contract and tests only a second, trust-specialised
residual layer.

## 7. Stage G — residual-depth test

Stage G tests whether a second supervised correction can make the P2/T1
consensus materially more useful.  It uses the immutable first-layer P2/T1
target-free monthly receipts rather than re-scoring any row with an in-sample
first-layer model.

For each score family, a 20-bin isotonic policy-net anchor is fitted once per
UTC day from the preceding 90 days of *resolved* first-layer OOS outcomes.
Same-day outcomes are deferred.  The Meta-2 target is then:

```text
rich policy net bps - first-layer prequential policy-net anchor
```

Every fold excludes a 28-day reserve, starts only once six months of
first-layer OOS history exist, persists target-free scores before joining
outcomes for MC1, and preserves the fixed dual-MC1 `>= +30 bps` admission and
constrained BCF-priority auction.  July--September 2025 are immutable warm-up
receipts; supervised Meta-2 fitting begins in October 2025.

| Arm | Authority | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | 2026 worst month / week | 2026 DD | Decision |
|---|---|---:|---:|---:|---:|---|
| **M0 first-layer P2/T1** | None; exact immutable control | **2,001 / +154.56** | **2,633 / +133.45** | **+105.84 / +82.06** | **-25.70%** | **Advance / control** |
| M2 broad residual | Huber residual; `0.25 × clip[-100,+100]` bps | 1,989 / +151.17 | 2,648 / +131.24 | +105.10 / +77.50 | -28.14% | Reject |
| M3 q20 trust residual | Demotion only; `0.25 × clip[-100,0]` bps | 2,012 / +149.69 | 2,730 / +130.54 | +103.53 / +76.43 | -28.68% | Reject |
| M4 severe-tail probability | Demotion only; `P(residual <= -100)` capped at 50 bps | 1,966 / +154.40 | 2,676 / +132.41 | +109.59 / +79.80 | -28.05% | Reject |

M4 is the closest alternative: it improves 2025-Q4 worst-period outcomes but
is still 1.03 bps/trade below M0 in 2026 and increases 2026 drawdown by 2.35
percentage points.  It therefore fails the same two-era precision/portability
gate; it must not be combined with M0 or deployed.  The correct Stage-G
decision is **no second residual layer**: retain the simpler P2/T1 correction
and direct the next research stage toward genuinely heterogeneous heads and a
combiner, not residual-of-residual depth.

## 8. Stage H — heterogeneous semantic heads

Stage H tested whether the stronger upstream could be made more useful to
downstream consumers by adding semantically distinct heads, rather than more
versions of the same LambdaRank residual.  It starts from the immutable P2/T1
receipts, keeps the five selected heads and their 75/25 rank blend, and leaves
the MC1 class, dual `>= +30 bps` admission and BCF-priority auction unchanged.

For every score family/month, each head is fitted only on six preceding months
of resolved policy outcomes with the latest 28 days reserved.  Scores are
persisted target-free before the unchanged MC1 maps join policy outcomes.

| Semantic head | Target / scope | Authority |
|---|---|---|
| H1 residual value | `policy_net_bps - enhanced_base_bps`, clipped at ±500 bps | residual value |
| H2 adverse overconfidence | `P(residual <= -100 bps)` | demotion |
| H3 underconfidence | `P(residual >= +100 bps)` | promotion proxy |
| H4 path usability on conflict | residual, only where base/supportive rank range >= 0.20 | conflict correction |
| H5 state transport | `P(residual > -100 bps)` | reliability |

All heads consume target-free disagreement geometry, family prequential state,
and only their declared causal context fields.  They do not receive raw
outcomes or labels at scoring time.

Two predeclared combiners were tested:

* **H1 median:** equal-weight median semantic rank.
* **H2 independence:** prior-reference quality × independence weights,
  75% shrunk to equal weights with a 40% maximum single-head weight.

Both use 75% P2/T1 family score + 25% semantic aggregate.  The table below
compares them to the selected P2/T1 first-layer control—not merely the older
live-like stack.

| Arm | 2025 Q4: trades / EV | 2026 Apr--Jul: trades / EV | Δ total bps vs P2/T1 (Q4 / 2026) | 2026 worst month / week | 2026 DD | Decision |
|---|---:|---:|---:|---:|---:|---|
| **P2/T1 first-layer control** | **2,001 / +154.56** | **2,633 / +133.45** | — | **+105.84 / +82.06** | -25.70% | **Control** |
| H1 semantic median | 1,988 / +153.32 | 2,722 / +128.52 | -4,483 / -1,542 | +97.97 / +75.31 | -26.10% | Reject |
| H2 quality × independence | 1,999 / +150.99 | 2,793 / +127.12 | -7,445 / +3,676 | +102.22 / +79.23 | -25.19% | Reject |

The semantic heads do not leverage the base improvement successfully.  H1
loses precision in both blocks; H2 gains 2026 aggregate bps only by accepting
160 extra trades while losing 6.33 bps/trade versus P2/T1, with weaker worst
month/week.  Neither meets the portability gate.  **Do not add a semantic
second layer to the P2/T1 consensus.**

For reproducibility, Stage H uses a dependency-light, research-only copy of
the fixed controlled auction (7x, 10%-wallet slots, 80% total margin, eight
concurrent positions, two entries per timestamp).  Its unchanged baseline leg
reproduces the stored live-like control metrics exactly.  It does not alter or
import the live execution path.

## 9. Strict-OOF orthogonal meta input to MC1

This follow-up is deliberately different from Stage H.  It creates a new
five-slot meta family whose outputs are not another copy of the incumbent
residual consensus, then gives **only target-free scores** from the retained
family to experimental MC1.  Path/TBM descriptors are used after outcome
resolution for training labels and weights only; no path event, label, or
policy outcome enters a held score panel or MC1 feature.

The three strict-OOF label families use six preceding resolved months with a
28-day purge reserve.  They were screened over July 2025--July 2026 before
any MC1 run:

| Family | Pooled top-1 / top-2 / top-5 policy net | Policy rank IC | Mean correlation to base rank | Decision |
|---|---:|---:|---:|---|
| O0 direct-policy control | +260.15 / +220.33 / +163.01 bps | +0.183 | +0.636 | Control |
| **O3 calibrated residual + semantic/TBM weighting** | +227.96 / +209.35 / **+171.57** bps | +0.080 | **−0.087** | **Retain** |
| O5 ordinal base-rank error + semantic/TBM weighting | +180.03 / +151.81 / +109.79 bps | +0.004 | −0.402 | Reject: weak IC and a negative final month |

O3 is retained because it adds genuinely distinct information without a
semantic label at inference: its target is the clipped policy-net residual
around a train-only isotonic base anchor, with semantic certainty/composite
weights applied only in fitting.  MC1 receives its consensus rank, five
head ranks, head-rank dispersion, and delta from the incumbent consensus;
it does **not** receive semantic/TBM values themselves.

Each score family then receives a separate six-month chronological MC1 map
with fully-resolved labels only and a prior-resolved 21-day shift.  Admission
remains unchanged: both current and BCF experimental maps must be at least
+30 bps, and BCF mapped EV remains auction priority.  The comparison below
uses the exact intersection of the live dual-score IDs and O3 IDs, canonical
reconciled policy outcomes on both sides, and the same constrained 7x /
10%-margin-slot / 80%-margin auction.  All selected trades have valid policy
outcomes.

| Period | Current live stack: trades / EV | O3 → MC1: trades / EV | Δ trades | Δ EV/trade | Δ total bps | Δ worst month | Δ worst week | Δ max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 Q4 | 2,071 / +140.65 | **1,878 / +162.12** | −193 | **+21.46** | **+13,161** | **+28.71** | **+59.11** | **+10.03 pp** |
| 2026 Apr--Jul | 2,366 / +139.50 | **2,510 / +140.77** | +144 | **+1.27** | **+23,282** | −6.73 | +0.53 | **+1.92 pp** |

This is the first semantic/meta route that has a coherent downstream effect:
it improves total policy net in both blocks, improves precision in both, and
reduces drawdown in both.  The 2026 worst month is modestly weaker, so it is a
**research challenger only**, not a live-stack replacement.  A later frozen
period is required before promotion.

## Decision

The current live stack remains the operational baseline.  P2/T1 is a useful
first-layer challenger control, but its earlier second-layer variants were not
portable enough to justify a replacement.  The retained **O3 → MC1** route is
the new genuinely-orthogonal challenger: on exact common live identities it
improves total policy net and drawdown in both blocks, while its 2026 worst
month is 6.73 bps/trade weaker.  It must therefore remain frozen and offline
until a later untouched period confirms that trade-off.

## Key sources

* Base-label study: `docs/STRICT_R3_LONG_SUPPORTIVE_LABEL_ABLATION_20260823.md`
* Enhanced-base stack status and receipts: `docs/ENHANCED_BASE_CONSENSUS_MC1_STATUS_20260823.md`
* Head audit: `docs/ENHANCED_BASE_LIVE_STACK_CHALLENGER_AUDIT_20260823.md`
* P2 selected replay: `data_perp/artifacts/strict_r3_enhanced_base_meta_stagec_20260824_p2_near_tie_diff50_v1/`
* Research runner: `scripts/run_strict_r3_enhanced_base_live_stack_challenger.py`
* Stage-G runner and receipts: `scripts/run_strict_r3_enhanced_base_meta2_depth.py` and `data_perp/artifacts/strict_r3_enhanced_base_meta_stageg_20260824_v1/`
* Stage-H runner and receipts: `scripts/run_strict_r3_enhanced_base_semantic_heads.py`; H1's complete reproducibility receipt is `data_perp/artifacts/strict_r3_enhanced_base_metah_20260824_h1audit_v2/`, and H2 is `data_perp/artifacts/strict_r3_enhanced_base_metah_20260824_v1/h2_semantic_independence/`
* O3 semantic materialiser: `scripts/materialize_strict_r3_orthogonal_meta_semantics.py`; target-free OOF funnel: `scripts/run_strict_r3_orthogonal_meta_label_funnel.py`; outputs: `data_perp/artifacts/strict_r3_orthogonal_meta_label_funnel_20260824_v1/`
* O3-to-MC1 challenger: `scripts/run_strict_r3_orthogonal_meta_mc1.py`; exact-common baseline comparison: `data_perp/artifacts/strict_r3_orthogonal_meta_mc1_20260824_v7_exact_common_baseline/`
* The O3 challenger uses the production constrained perps replay path in `extreme_price_movements/portfolio_policy_replay.py`.  The historical `scripts/strict_r3_research_light_portfolio.py` adapter was not used for this comparison.
* Orthogonal semantic materialisation: `scripts/materialize_strict_r3_orthogonal_meta_semantics.py` and `data_perp/artifacts/strict_r3_orthogonal_meta_semantics_20260824_v2/`
* OOF label funnel: `scripts/run_strict_r3_orthogonal_meta_label_funnel.py` and `data_perp/artifacts/strict_r3_orthogonal_meta_label_funnel_20260824_v1/`
* O3-to-MC1 exact-common evaluation: `scripts/run_strict_r3_orthogonal_meta_mc1.py` and `data_perp/artifacts/strict_r3_orthogonal_meta_mc1_20260824_v7_exact_common_baseline/`
