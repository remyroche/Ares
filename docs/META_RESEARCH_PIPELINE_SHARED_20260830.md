# Frozen source specification

- Source: https://chatgpt.com/s/t_6a919d35bd5c8191918ac3d2b04d3041
- Retrieved: 2026-08-30 UTC
- Retrieval: public shared-page snapshot; body decoded from its document block.
- Scope: research-only Meta-layer experiment contract.

---

# META RESEARCH PIPELINE

## 1. TARGET × QUERY SELECTION

### Goal

For each Meta target family, first identify the best **target definition × query definition** using a common lightweight model contract.

At this stage:

- use **LightGBM rank-XENDCG only**;
- use the same broad/frozen causal feature contract;
- use the same baseline sample-weight contract;
- use the same OOF folds;
- use only a short/common HPO;
- do NOT feature-select independently for every target × query candidate;
- do NOT compare CatBoost yet.

The purpose is to isolate:

```text
TARGET SEMANTICS
×
QUERY SEMANTICS
```

before spending compute adapting features/models to them.

Primary selector:

```text
head-specific MetaDecisionStable
```

Secondary diagnostics:

```text
conditional IC to Base residual
conditional CMI beyond Base
conditional economic separation
cross-era stability
```

Main requirement:

> The signal must create useful candidate promotions/demotions when combined with Base, not merely predict its own training target well.

---

# 1A. MAGNITUDE FAMILY

Targets:

```text
M_bps:
    economic Base residual in raw bps

M_sqrtATR:
    economic Base residual / ATR^0.5

M_ATR:
    economic Base residual / ATR
```

For each normalization, test relevant clipping / bin / ordinal geometries.

Queries:

```text
Q1:
    Base-score band
    irrespective of timestamp

Q2:
    Base-score band × rolling 28-day block
```

Run all:

```text
M target geometry
×
M query
```

with common LightGBM rank-XENDCG.

Select:

```text
ONE winning Magnitude target × query
```

by MetaDecisionStable.

Only this winning Magnitude configuration proceeds to feature selection.

---

# 1B. UNDER-CONFIDENCE FAMILY

Targets:

```text
positive Base residual >= threshold_bps
AND
unexpected trailing-profit activation

positive Base residual >= threshold_ATR
AND
unexpected trailing-profit activation
```

Test several threshold geometries.

Query:

```text
exact decision timestamp
```

Select:

```text
ONE winning Under target × query
```

using common LightGBM rank-XENDCG.

Only this configuration proceeds.

---

# 1C. OVER-CONFIDENCE / DEMOTION FAMILY

Targets:

```text
negative Base residual <= -threshold_bps
AND
unexpected SL / adverse path

negative Base residual <= -threshold_ATR
AND
unexpected SL / adverse path
```

Test several threshold geometries.

Query:

```text
exact decision timestamp
```

Select:

```text
ONE winning Over target × query
```

Only this configuration proceeds.

---

# 1D. SIGNED-STATE FAMILY

Targets:

```text
strong over-confidence
mild over-confidence
accurate
mild under-confidence
strong under-confidence
```

Build independently using:

```text
S_bps
S_sqrtATR
S_ATR
```

Queries:

```text
Q1:
    Base-score band

Q2:
    exact timestamp

Q3:
    Base-score band × rolling 28-day block
```

Require explicitly signed direction.

Select:

```text
ONE winning State target × query
```

Only this configuration proceeds.

For State, subsequent feature selection should emphasize broad-market /\\
cross-sectional features:

```text
breadth
dispersion
spectral state
effective rank
correlation concentration
cross-sectional tails
market liquidity
broad OI / leverage
regime transitions
session state
benchmark / cross-asset regime
```

Do not expand State using new single-asset market-context fields.

---

# 2. HEAD-SPECIFIC FEATURE SELECTION

Run this stage exactly ONCE for each winning family configuration:

```text
Magnitude winner
Under winner
Over winner
State winner
```

Feature relevance must be recalculated independently because the targets and\\
queries differ.

Start from the full causal feature universe.

---

## 2A. COMMON PRE-SCREEN

For every feature X\\_j estimate:

```text
CMI(X_j ; winning Meta target | strict-OOF Base)

conditional residual association

conditional economic separation

cross-era stability

era-local random-subspace inclusion uplift

redundancy with other useful features
```

All CMI binning / discretization must be fold-local.

Correlation grouping:

```text
abs Spearman >= 0.97
```

Within a correlated block prefer fields with:

```text
higher stable conditional CMI
higher conditional economic separation
higher era-local inclusion uplift
better availability / missingness stability
```

Do not automatically discard a feature just because it is in the lowest\\
20% of marginal CMI.

Discard the weak-CMI tail only when it is also weak on:

```text
conditional residual association
AND
random-subspace inclusion
AND
interaction / block evidence
```

---

## 2B. PORTABILITY FILTER

Calculate feature evidence independently by month / era.

Prefer:

```text
median-era CMI > 0

useful direction in >=65–75% of supported eras

Q25-era value not materially negative

no one era accounts for >30–35% of aggregate value
```

Track:

```text
mean
median
Q25
worst era
positive-era fraction
dispersion
```

Months are primary portability units.

OOF folds / broader eras are secondary confirmation.

---

# 2C. MAGNITUDE-WINNER FEATURE SELECTION

Relevant Base region:

```text
primary:
    Base Top5–30%

secondary:
    Base Top0–5%
```

Per-feature evidence:

```text
CMI(feature ; magnitude residual | Base)

partial Spearman(feature, residual | Base)

residual separation inside matched Base bands

economic utility spread inside matched Base bands

era-local random-subspace uplift
```

The feature screen is run only for the already-selected Magnitude target\\
normalization/query.

---

# 2D. UNDER-WINNER FEATURE SELECTION

Relevant Base region:

```text
0–5%:
    moderate authority

5–20%:
    highest

20–30%:
    moderate

30–40%:
    diagnostic / low

>40%:
    negligible
```

Example localization:

```text
0–5      0.5
5–20     1
20–30    0.5
>30      near-zero
```

Per-feature evidence:

```text
CMI(feature ; Under target | Base)

positive-residual separation

+50 opportunity-density separation

+100 opportunity-density separation

trailing-activation separation

false-promotion rate

era-local inclusion uplift
```

The objective is specifically to distinguish:

```text
under-ranked winners
vs
weak candidates with similar Base scores
```

---

# 2E. OVER-WINNER FEATURE SELECTION

Relevant Base region:

```text
0–2%:
    very high authority

2–5%:
    high

5–10%:
    moderate

10–15%:
    lower

>15%:
    diagnostic only
```

Example:

```text
0–2      2.0
2–5      1.5
5–10     1.0
10–15    0.5
>15      near-zero
```

Per-feature evidence:

```text
CMI(feature ; Over target | Base)

negative residual association

severe-loss discrimination

SL / adverse-event discrimination

bad Top1 / Top2 substitution separation

era-local inclusion uplift
```

Downside portability is mandatory:

```text
median severe-loss separation
Q25 severe-loss separation
worst-era behavior
fraction eras with correct demotion direction
```

---

# 2F. STATE-WINNER FEATURE SELECTION

Relevant Base region:

```text
Base Top0–30%
```

Per-feature evidence:

```text
CMI(feature ; signed state | Base)

CMI(feature ; Under-vs-rest | Base)

CMI(feature ; Over-vs-rest | Base)

signed residual association

cross-era sign consistency

era-local inclusion uplift
```

Require directional information:

```text
Base too optimistic
vs
Base accurate
vs
Base too pessimistic
```

not merely:

```text
Base error magnitude is high
```

Use stricter portability:

```text
positive median-era value

>=70% directionally consistent eras where supported

Q25 near-zero or positive

no isolated-regime dependence
```

---

# 2G. SEED CONTRACT

For each family winner build approximately:

```text
25 fields
```

using the Pareto frontier of:

```text
conditional CMI
conditional error association
cross-era stability
low redundancy
feature-family diversity
```

Do not simply choose the top 25 pooled CMI fields.

This is the seed contract only.

---

# 2H. BLOCK EXPANSION + COMPRESSION

Add remaining candidate features by semantic/correlation block.

For each block:

```text
Seed
→ Seed + Block
```

evaluate:

```text
ΔMetaStable

Δconditional economic separation

Δfixed-MC1-probe value
```

For useful blocks compress using:

```text
group permutation importance

sub-block permutation

MDA

drop-column tests

random-subspace inclusion uplift
```

All importance/uplift statistics must also be era-local.

For correlated groups, permute the full group before interpreting individual\\
feature importance.

Then run:

```text
ADD
DROP
SWAP
```

beam search with:

```text
beam width ≈ 3
```

Do not impose the same final feature count across Meta heads.

Freeze ONE final feature contract for each family winner.

---

# 3. SAMPLE-WEIGHT OPTIMIZATION

Run independently after feature selection for:

```text
Magnitude winner
Under winner
Over winner
State winner
```

General:

```text
raw_weight =
    magnitude_component
    ×
    Base-rank localization
```

Search independently:

```text
shape / power

maximum authority
```

Then:

```text
clip effective weights

renormalize within the selected query
```

Primary relative range:

```text
0.5–4
```

---

## 3A. Magnitude weights

```text
magnitude component:
    abs(Base residual)
    or bounded winning target magnitude

power:
    0.5–2.0

Base localization:
    emphasize Top2–30%
```

---

## 3B. Under weights

```text
magnitude component:
    positive residual excess above winning Under threshold

power:
    0.5–2.0

Base localization:
    progressively increase from 0-10%, plateau 10-20%, decrease 20-30% by Base rank normalized for the full population (not just the routed population)

```

---

## 3C. Over weights

```text
magnitude component:
    abs(negative residual)
    and/or adverse policy-loss magnitude

power:
    0.5–1.0

Base localization:
    extremely strong Top0–2%
    progressively lower to Top10%
```

---

## 3D. State weights

```text
magnitude:
    abs(Base residual)

power:
    0.5–1.0

Base localization:
    Top0–20%

plus:
    fold-local balancing of
    Over / Accurate / Under
```

Freeze one sample-weight contract per family.

---

# 4. MODEL-FAMILY CHOICE

Now each target family has frozen:

```text
target
query
feature contract
sample weights
OOF folds
evaluation metric
```

Only now compare model families.

Models:

```text
LightGBM rank-XENDCG

CatBoost QueryRMSE

CatBoost YetiRank
```

Use a matched short HPO:

```text
max 150 trials per model family
aggressive pruning
early stopping 30
same OOF folds
same external MetaDecisionStable
```

Do not compare default models.

---

## Model priors

Magnitude:

```text
LGBM XENDCG:
    strong

CatBoost QueryRMSE:
    primary challenger

CatBoost YetiRank:
    secondary after ordinalization
```

Under:

```text
LGBM XENDCG:
    strong

CatBoost QueryRMSE:
    secondary continuous-margin challenger

CatBoost YetiRank:
    primary challenger
```

Over:

```text
LGBM XENDCG:
    strong

CatBoost QueryRMSE:
    secondary continuous-margin challenger

CatBoost YetiRank:
    primary challenger
```

Signed State:

```text
LGBM XENDCG:
    strong

CatBoost QueryRMSE:
    strong for continuous signed residual

CatBoost YetiRank:
    strong for ordinal signed state
```

Freeze one winning model family per Meta target family.

---

# 5. FULL MODEL-SPECIFIC HPO

Only after model-family selection run the expensive:

```text
150-trial
aggressive MedianPruner
early-stopping-30
```

HPO for the winning model.

---

## 5A. LightGBM rank-XENDCG

```text
max_depth:
    3–6

num_leaves:
    7–31
    constrained by depth

learning_rate:
    0.02–0.10 log

min_data_in_leaf:
    ~0.5–5% effective training rows

feature_fraction:
    0.55–1.00

bagging_fraction:
    0.65–1.00

bagging_freq:
    1 if bagging_fraction <1

lambda_l1:
    1e-5–10 log

lambda_l2:
    1e-3–100 log

min_gain_to_split:
    0–0.10

max_bin:
    63 / 127

iterations:
    ceiling ~2000

early_stopping:
    30

seed:
    fixed
```

---

## 5B. CatBoost QueryRMSE

```text
depth:
    4–8
    prior 4–6

learning_rate:
    0.02–0.12 log

iterations:
    ceiling ~1500

l2_leaf_reg:
    0.1–100 log

random_strength:
    0 / .25 / .5 / 1 / 2

rsm:
    .60–1.00

border_count:
    64 / 128 / 254

bootstrap_type:
    Bayesian / Bernoulli
```

Bayesian:

```text
bagging_temperature:
    0–2
```

Bernoulli:

```text
subsample:
    .65–.95
```

---

## 5C. CatBoost YetiRank

Test:

```text
Classic
NDCG
```

Classic:

```text
permutations:
    5 / 10 / 20

decay:
    .70 / .85 / .95
```

NDCG:

```text
permutations:
    5 / 10 / 20

top:
    align to useful head region
```

Approximate authority:

```text
Under:
    Top5–30%

Over:
    Top5–15%
```

Tree HPO:

```text
depth:
    4–8

learning_rate:
    .02–.12

iterations:
    ~1500 ceiling

l2_leaf_reg:
    .1–100 log

random_strength:
    0 / .25 / .5 / 1 / 2

rsm:
    .60–1.00

border_count:
    64 / 128 / 254

bootstrap_type:
    Bayesian / Bernoulli
```

with corresponding:

```text
bagging_temperature 0–2
```

or:

```text
subsample .65–.95
```

---

# 6. COMMON MODEL / HPO OBJECTIVE

At every stage use the same external decision-oriented score.

Assume research authority:

```text
66% Base rank
33% Meta rank
```

unless another authority is frozen.

For Top2 timestamp selection define:

```text
CHALLENGER_ONLY

CONTROL_ONLY
```

Then:

```text
ΔSwapUtility =
    Utility(CHALLENGER_ONLY)
    -
    Utility(CONTROL_ONLY)
```

---

## Under objective

Reward:

```text
positive ΔSwapUtility

recovered +50 utility

recovered +100 utility

Top2 EV improvement
```

Penalize:

```text
false promotions

severe-loss increase
```

---

## Over objective

Reward:

```text
economic value of avoided bad trades

severe-loss reduction

Top1 / Top2 substitution improvement
```

Penalize:

```text
good trades incorrectly demoted

participation collapse
```

---

## Magnitude objective

Reward:

```text
ΔSwapUtility in both directions

Top2 EV

economic ordering improvement conditional on Base
```

---

## State objective

Reward:

```text
ΔSwapUtility in both directions

balanced promotion + demotion improvement

signed-error discrimination

cross-era portability
```

---

# 7. STABILITY / PORTABILITY

Build:

```text
MetaDecisionStable
```

from the decision score by:

```text
month
OOF fold
broader era
```

Report:

```text
mean
median
Q25
Q10
positive-era fraction
worst-era guard
```

Prefer:

```text
positive median-era contribution

majority neutral/positive eras

Q25 not materially negative

no one era producing >30–35% of uplift
```

CMI portability remains supporting evidence only.

---

# 8. FINAL FAMILY WINNERS

At the end there are only:

```text
ONE optimized Magnitude head

ONE optimized Under head

ONE optimized Over head

ONE optimized State head
```

Each has independently passed:

```text
target × query selection
→ feature selection
→ sample-weight optimization
→ model choice
→ full model HPO
```

Only these four heads advance to the expensive native MC1 comparison.

---

# 9. FINAL MC1 / PORTFOLIO ABLATION

Run:

```text
Base only

Base + retained Under F120 control

Base + optimized Magnitude

Base + optimized Under

Base + optimized Over

Base + optimized State
```

Then:

Base + retained head + best complementary new head &#x20;

Then if Base + 2 retained

heads win: Base + 2 retained heads + best complementary new head

For every arm:

```text
retrain native Current MC1

retrain BCF MC1 only if its input contract changes

fixed dual >= +50

BCF remains auction priority

same chronological constrained portfolio
```

Final metrics:

```text
admissions
portfolio entries
EV/trade
total net EV
Top1 EV
Top2 EV
worst week
worst month
max DD
severe losses
substitution economics
```

The final promotion criterion is not CMI or Meta standalone ranking.

It is:

> Does the optimized Meta head add stable incremental economic information that MC1 can convert into better decisions?
","index","0","title",{"_109":-5,"_164":-5,"_165":-5,"_166":-5},"edited","edited_at","2026-08-28T14:37:33.778000Z","updated_at","cc","bcc","subject",{"_53":449,"_160":18,"_355":18,"_356":-5},"1",{"_53":448,"_160":18,"_355":18,"_356":-5},"2",{"_53":447,"_160":18,"_355":18,"_356":-5},"3",{"_53":446,"_160":18,"_355":18,"_356":-5},"4",{"_53":445,"_160":18,"_355":18,"_356":-5},"5",{"_53":444,"_160":18,"_355":18,"_356":-5},"6",{"_53":443,"_160":18,"_355":18,"_356":-5},"7",{"_53":442,"_160":18,"_355":18,"_356":-5},"8",{"_53":441,"_160":18,"_355":18,"_356":-5},"9",{"_53":440,"_160":18,"_355":18,"_356":-5},"10",{"_53":439,"_160":18,"_355":18,"_356":-5},"11",{"_53":438,"_160":18,"_355":18,"_356":-5},"12",{"_53":437,"_160":18,"_355":18,"_356":-5},"13",{"_53":436,"_160":18,"_355":18,"_356":-5},"14",{"_53":435,"_160":18,"_355":18,"_356":-5},"15",{"_53":434,"_160":18,"_355":18,"_356":-5},"16",{"_53":433,"_160":18,"_355":18,"_356":-5},"17",{"_53":432,"_160":18,"_355":18,"_356":-5},"18",{"_53":431,"_160":18,"_355":18,"_356":-5},"19",{"_53":430,"_160":18,"_355":18,"_356":-5},"20",{"_53":429,"_160":18,"_355":18,"_356":-5},"21",{"_53":428,"_160":18,"_355":18,"_356":-5},"22",{"_53":427,"_160":18,"_355":18,"_356":-5},"23",{"_53":426,"_160":18,"_355":18,"_356":-5},"24",{"_53":425,"_160":18,"_355":18,"_356":-5},"25",{"_53":424,"_160":18,"_355":18,"_356":-5},"26",{"_53":423,"_160":18,"_355":18,"_356":-5},"27",{"_53":422,"_160":18,"_355":18,"_356":-5},"28",{"_53":421,"_160":18,"_355":18,"_356":-5},"29",{"_53":420,"_160":18,"_355":18,"_356":-5},"30",{"_53":419,"_160":18,"_355":18,"_356":-5},"31",{"_53":418,"_160":18,"_355":18,"_356":-5},"32",{"_53":417,"_160":18,"_355":18,"_356":-5},"33",{"_53":416,"_160":18,"_355":18,"_356":-5},"34",{"_53":415,"_160":18,"_355":18,"_356":-5},"35",{"_53":414,"_160":18,"_355":18,"_356":-5},"36",{"_53":413,"_160":18,"_355":18,"_356":-5},"37",{"_53":412,"_160":18,"_355":18,"_356":-5},"38",{"_53":411,"_160":18,"_355":18,"_356":-5},"39",{"_53":410,"_160":18,"_355":18,"_356":-5},"40",{"_53":409,"_160":18,"_355":18,"_356":-5},"41",{"_53":408,"_160":18,"_355":18,"_356":-5},"42",{"_53":407,"_160":18,"_355":18,"_356":-5},"43",{"_53":406,"_160":18,"_355":18,"_356":-5},"44",{"_53":405,"_160":18,"_355":18,"_356":-5},"45",{"_53":404,"_160":18,"_355":18,"_356":-5},"46",{"_53":403,"_160":18,"_355":18,"_356":-5},"47",{"_53":402,"_160":18,"_355":18,"_356":-5},"48",{"_53":401,"_160":18,"_355":18,"_356":-5},"49",{"_53":400,"_160":18,"_355":18,"_356":-5},"50",{"_53":399,"_160":18,"_355":18,"_356":-5},"51",{"_53":398,"_160":18,"_355":18,"_356":-5},"52",{"_53":397,"_160":18,"_355":18,"_356":-5},"53",{"_53":396,"_160":18,"_355":18,"_356":-5},"54",{"_53":395,"_160":18,"_355":18,"_356":-5},"55",{"_53":394,"_160":18,"_355":18,"_356":-5},"56",{"_53":393,"_160":18,"_355":18,"_356":-5},"57",{"_53":392,"_160":18,"_355":18,"_356":-5},"58",{"_53":391,"_160":18,"_355":18,"_356":-5},"59",{"_53":390,"_160":18,"_355":18,"_356":-5},"60",{"_53":389,"_160":18,"_355":18,"_356":-5},"61",{"_53":388,"_160":18,"_355":18,"_356":-5},"62",{"_53":387,"_160":18,"_355":18,"_356":-5},"63",{"_53":386,"_160":18,"_355":18,"_356":-5},"64",{"_53":385,"_160":18,"_355":18,"_356":-5},"65",{"_53":384,"_160":18,"_355":18,"_356":-5},"66",{"_53":383,"_160":18,"_355":18,"_356":-5},"67",{"_53":382,"_160":18,"_355":18,"_356":-5},"68",{"_53":381,"_160":18,"_355":18,"_356":-5},"69",{"_53":380,"_160":18,"_355":18,"_356":-5},"70",{"_53":379,"_160":18,"_355":18,"_356":-5},"71",{"_53":378,"_160":18,"_355":18,"_356":-5},"72",{"_53":377,"_160":18,"_355":18,"_356":-5},"73",{"_53":376,"_160":18,"_355":18,"_356":-5},"74",{"_53":375,"_160":18,"_355":18,"_356":-5},"75",{"_53":374,"_160":18,"_355":18,"_356":-5},"76",{"_53":373,"_160":18,"_355":18,"_356":-5},"77",{"_53":372,"_160":18,"_355":18,"_356":-5},"78",{"_53":371,"_160":18,"_355":18,"_356":-5},"79",{"_53":370,"_160":18,"_355":18,"_356":-5},"80",{"_53":369,"_160":18,"_355":18,"_356":-5},"81",{"_53":368,"_160":18,"_355":18,"_356":-5},"82",{"_53":367,"_160":18,"_355":18,"_356":-5},"83",{"_53":366,"_160":18,"_355":18,"_356":-5},"84",{"_53":365,"_160":18,"_355":18,"_356":-5},"85",{"_53":364,"_160":18,"_355":18,"_356":-5},"86",{"_53":363,"_160":18,"_355":18,"_356":-5},"87",{"_53":362,"_160":18,"_355":18,"_356":-5},"88",{"_53":361,"_160":18,"_355":18,"_356":-5},"89",{"_53":360,"_160":18,"_355":18,"_356":-5},"90",{"_53":359,"_160":18,"_355":18,"_356":-5},"91",{"_53":358,"_160":18,"_355":18,"_356":-5},"92",{"_53":357,"_160":18,"_355":18,"_356":-5},"93",{"_53":354,"_160":18,"_355":18,"_356":-5},"ie4u4l","previewable","preview_language","gfm4xb","rnwrj8","1pwnl3","smvnhe","fmmqhl","454m84","4c8xqa","v95uhh","h6i8kl","u0jogo","lc3z4l","t6t40f","b9l8ax","qv44ie","7sakx6","r7as61","ah3gqg","kfdj06","ujds4u","pmhee8","crxaz1","bfoiwd","waun4w","d64m0p","4fqoij","bo7jpr","s0jt5m","z6zztq","mqqb3l","go67k3","xa0kuu","47n24u","7pny6w","vya2fj","119lni","ieubn9","yrj0ur","051f9q","9zbbx7","tbg1af","6jsff0","lf5bhe","wrtacn","xaqwa7","zsl6nu","17zie6","kq5od6","oobp8y","31fap3","tem0b2","bectzk","wytuan","eorm6y","ijtmvm","atr1wg","mzevud","j27b73","69mch3","wlm4qw","twnwxz","nvfr99","ruz5cz","5b77d2","y7xy9d","372tc5","uvrvdk","0frzz9","owk96g","nf075w","a8zl62","1f1tha","o7m1jn","icgcqz","0zrrld","wbngug","k29v67","dupbmb","gy70o2","ahfop2","ezxi4e","oel8yv","be5lko","3aekt0","d5jfb2","un9t3l","p40phe","mxsh30","seq614","efjfc8","wwvyfb","814nfl","1jjot0","j7a6wg",{"_456":486,"_458":487,"_460":488,"_462":-5,"_454":489,"_490":491,"_53":492,"_469":493,"_494":-5,"_495":-5,"_496":497,"_498":-5,"_499":500,"_501":-5,"_502":-5,"_503":-5,"_504":505,"_506":-5,"_507":-5,"_508":509},{"_454":455,"_456":482,"_458":483,"_460":484,"_462":482,"_463":485,"_465":466,"_467":106,"_469":470,"_471":472,"_473":474,"_475":476},{"_454":455,"_456":477,"_458":478,"_460":479,"_462":477,"_463":480,"_465":466,"_467":481,"_469":470,"_471":472,"_473":474,"_475":476},{"_454":455,"_456":457,"_458":459,"_460":461,"_462":457,"_463":464,"_465":466,"_467":468,"_469":470,"_471":472,"_473":474,"_475":476},"type","followup_a","matched_text","Outline criteria used for final model-family choice and hyperparameter optimization","start_idx",16849,"end_idx",16932,"alt","prompt_text","Outline the criteria and process for choosing the final model family and performing full hyperparameter optimization as described in the Meta research pipeline.","receiver_followup_intent","tailored_continuation","receiver_followup_position",3,"source","receiver_followups_tailored","receiver_followup_treatment","tailored","receiver_followup_capability_scope","logged_out","receiver_followup_prompt_version","v2","Describe the role of sample-weight optimization in the pipeline",16783,16846,"Describe the purpose and methodology of the sample-weight optimization stage in the Meta research pipeline, highlighting how it differs across target families.",2,"Explain why target × query selection precedes feature selection",16717,16780,"Explain why the Meta research pipeline uses target × query selection as a cheap initial architecture screen before proceeding to feature selection and further modeling steps.","fileciteturn4file0L4-L10",284,312,"file","name","Fichier markdown.md collé","file_00000000955c81f4908a8dfbf3109569","my_files","snippet","cloud_doc_url","library_file_id","libfile_39dceb428d608191ad38620e2979d193","library_artifact_type","medical_file_reference",{"_518":-5,"_519":-5,"_520":-5,"_521":-5,"_522":-5,"_523":-5,"_524":-5,"_525":-5},"drug_file_reference","page_range_start","page_range_end","input_pointer",{"_510":511,"_512":513,"_514":43,"_515":511,"_516":517},"fff_metadata","connector_id","api_tool_source","files/context_stuff","message_index",4,"message_id","abdfadff-af93-4985-aa87-f1504abb0cc5","file_index","line_range_start","line_range_end",10,"authors_display","doi","publication_year","journal_name","source_label","journal_homepage_url","source_url","source_icon_url",{"_527":487,"_528":488,"_529":530,"_107":531},"start_ix","end_ix","citation_format_type","berry_file_search",{"_454":489,"_490":491,"_53":492,"_469":493,"_57":-5,"_532":533},"extra",{"_534":511,"_535":43,"_536":537,"_538":513,"_539":-5,"_540":541,"_495":-5,"_508":509,"_496":497,"_518":-5,"_519":-5,"_520":-5,"_521":-5,"_522":-5,"_523":-5,"_524":-5,"_525":-5,"_542":-5,"_543":-5,"_544":-5,"_158":-5,"_545":-5},"retrieval_turn","retrieval_file_index","cited_message_idx",735,"cited_message_id","search_result_idx","line_range",[511,517],"set_id","spl_version","run_id","effective_time","stop","stop_tokens",[549],200002,"content_type","parts",[553],"Correct. The unit of feature selection should be the **winning target × query configuration within each target family**, not every candidate target × query combination. Your source already states that target × query is selected first, then features are optimized once for the winner. fileciteturn4file0L4-L10

So the hierarchy should be:

\\[
\\boxed{
\\text{Target × Query screen}
\\rightarrow
\\text{1 winner per target family}
\\rightarrow
\\text{Feature selection}
\\rightarrow
\\text{Sample weights}
\\rightarrow
\\text{Model choice}
\\rightarrow
\\text{Full HPO}
}
\\]

:::writing{variant=\"document\" id=\"28417\"}
# META RESEARCH PIPELINE

## 1. TARGET × QUERY SELECTION

### Goal

For each Meta target family, first identify the best **target definition × query definition** using a common lightweight model contract.

At this stage:

- use **LightGBM rank-XENDCG only**;
- use the same broad/frozen causal feature contract;
- use the same baseline sample-weight contract;
- use the same OOF folds;
- use only a short/common HPO;
- do NOT feature-select independently for every target × query candidate;
- do NOT compare CatBoost yet.

The purpose is to isolate:

```text
TARGET SEMANTICS
×
QUERY SEMANTICS
```

before spending compute adapting features/models to them.

Primary selector:

```text
head-specific MetaDecisionStable
```

Secondary diagnostics:

```text
conditional IC to Base residual
conditional CMI beyond Base
conditional economic separation
cross-era stability
```

Main requirement:

> The signal must create useful candidate promotions/demotions when combined with Base, not merely predict its own training target well.

---

# 1A. MAGNITUDE FAMILY

Targets:

```text
M_bps:
    economic Base residual in raw bps

M_sqrtATR:
    economic Base residual / ATR^0.5

M_ATR:
    economic Base residual / ATR
```

For each normalization, test relevant clipping / bin / ordinal geometries.

Queries:

```text
Q1:
    Base-score band
    irrespective of timestamp

Q2:
    Base-score band × rolling 28-day block
```

Run all:

```text
M target geometry
×
M query
```

with common LightGBM rank-XENDCG.

Select:

```text
ONE winning Magnitude target × query
```

by MetaDecisionStable.

Only this winning Magnitude configuration proceeds to feature selection.

---

# 1B. UNDER-CONFIDENCE FAMILY

Targets:

```text
positive Base residual >= threshold_bps
AND
unexpected trailing-profit activation

positive Base residual >= threshold_ATR
AND
unexpected trailing-profit activation
```

Test several threshold geometries.

Query:

```text
exact decision timestamp
```

Select:

```text
ONE winning Under target × query
```

using common LightGBM rank-XENDCG.

Only this configuration proceeds.

---

# 1C. OVER-CONFIDENCE / DEMOTION FAMILY

Targets:

```text
negative Base residual <= -threshold_bps
AND
unexpected SL / adverse path

negative Base residual <= -threshold_ATR
AND
unexpected SL / adverse path
```

Test several threshold geometries.

Query:

```text
exact decision timestamp
```

Select:

```text
ONE winning Over target × query
```

Only this configuration proceeds.

---

# 1D. SIGNED-STATE FAMILY

Targets:

```text
strong over-confidence
mild over-confidence
accurate
mild under-confidence
strong under-confidence
```

Build independently using:

```text
S_bps
S_sqrtATR
S_ATR
```

Queries:

```text
Q1:
    Base-score band

Q2:
    exact timestamp

Q3:
    Base-score band × rolling 28-day block
```

Require explicitly signed direction.

Select:

```text
ONE winning State target × query
```

Only this configuration proceeds.

For State, subsequent feature selection should emphasize broad-market /
cross-sectional features:

```text
breadth
dispersion
spectral state
effective rank
correlation concentration
cross-sectional tails
market liquidity
broad OI / leverage
regime transitions
session state
benchmark / cross-asset regime
```

Do not expand State using new single-asset market-context fields.

---

# 2. HEAD-SPECIFIC FEATURE SELECTION

Run this stage exactly ONCE for each winning family configuration:

```text
Magnitude winner
Under winner
Over winner
State winner
```

Feature relevance must be recalculated independently because the targets and
queries differ.

Start from the full causal feature universe.

---

## 2A. COMMON PRE-SCREEN

For every feature X_j estimate:

```text
CMI(X_j ; winning Meta target | strict-OOF Base)

conditional residual association

conditional economic separation

cross-era stability

era-local random-subspace inclusion uplift

redundancy with other useful features
```

All CMI binning / discretization must be fold-local.

Correlation grouping:

```text
abs Spearman >= 0.97
```

Within a correlated block prefer fields with:

```text
higher stable conditional CMI
higher conditional economic separation
higher era-local inclusion uplift
better availability / missingness stability
```

Do not automatically discard a feature just because it is in the lowest
20% of marginal CMI.

Discard the weak-CMI tail only when it is also weak on:

```text
conditional residual association
AND
random-subspace inclusion
AND
interaction / block evidence
```

---

## 2B. PORTABILITY FILTER

Calculate feature evidence independently by month / era.

Prefer:

```text
median-era CMI > 0

useful direction in >=65–75% of supported eras

Q25-era value not materially negative

no one era accounts for >30–35% of aggregate value
```

Track:

```text
mean
median
Q25
worst era
positive-era fraction
dispersion
```

Months are primary portability units.

OOF folds / broader eras are secondary confirmation.

---

# 2C. MAGNITUDE-WINNER FEATURE SELECTION

Relevant Base region:

```text
primary:
    Base Top5–30%

secondary:
    Base Top0–5%
```

Per-feature evidence:

```text
CMI(feature ; magnitude residual | Base)

partial Spearman(feature, residual | Base)

residual separation inside matched Base bands

economic utility spread inside matched Base bands

era-local random-subspace uplift
```

The feature screen is run only for the already-selected Magnitude target
normalization/query.

---

# 2D. UNDER-WINNER FEATURE SELECTION

Relevant Base region:

```text
0–5%:
    moderate authority

5–10%:
    meaningful

10–20%:
    highest

20–30%:
    highest

30–40%:
    diagnostic / low

>40%:
    negligible
```

Example localization:

```text
0–5      0.5
5–10     0.8
10–20    1.0
20–30    1.0
30–40    0.25
>40      near-zero
```

Per-feature evidence:

```text
CMI(feature ; Under target | Base)

positive-residual separation

+50 opportunity-density separation

+100 opportunity-density separation

trailing-activation separation

false-promotion rate

era-local inclusion uplift
```

The objective is specifically to distinguish:

```text
under-ranked winners
vs
weak candidates with similar Base scores
```

---

# 2E. OVER-WINNER FEATURE SELECTION

Relevant Base region:

```text
0–2%:
    very high authority

2–5%:
    high

5–10%:
    moderate

10–15%:
    lower

>15%:
    diagnostic only
```

Example:

```text
0–2      2.0
2–5      1.5
5–10     1.0
10–15    0.5
>15      near-zero
```

Per-feature evidence:

```text
CMI(feature ; Over target | Base)

negative residual association

severe-loss discrimination

SL / adverse-event discrimination

bad Top1 / Top2 substitution separation

era-local inclusion uplift
```

Downside portability is mandatory:

```text
median severe-loss separation
Q25 severe-loss separation
worst-era behavior
fraction eras with correct demotion direction
```

---

# 2F. STATE-WINNER FEATURE SELECTION

Relevant Base region:

```text
Base Top0–30%
```

Per-feature evidence:

```text
CMI(feature ; signed state | Base)

CMI(feature ; Under-vs-rest | Base)

CMI(feature ; Over-vs-rest | Base)

signed residual association

cross-era sign consistency

era-local inclusion uplift
```

Require directional information:

```text
Base too optimistic
vs
Base accurate
vs
Base too pessimistic
```

not merely:

```text
Base error magnitude is high
```

Use stricter portability:

```text
positive median-era value

>=70% directionally consistent eras where supported

Q25 near-zero or positive

no isolated-regime dependence
```

---

# 2G. SEED CONTRACT

For each family winner build approximately:

```text
25 fields
```

using the Pareto frontier of:

```text
conditional CMI
conditional error association
cross-era stability
low redundancy
feature-family diversity
```

Do not simply choose the top 25 pooled CMI fields.

This is the seed contract only.

---

# 2H. BLOCK EXPANSION + COMPRESSION

Add remaining candidate features by semantic/correlation block.

For each block:

```text
Seed
→ Seed + Block
```

evaluate:

```text
ΔMetaStable

Δconditional economic separation

Δfixed-MC1-probe value
```

For useful blocks compress using:

```text
group permutation importance

sub-block permutation

MDA

drop-column tests

random-subspace inclusion uplift
```

All importance/uplift statistics must also be era-local.

For correlated groups, permute the full group before interpreting individual
feature importance.

Then run:

```text
ADD
DROP
SWAP
```

beam search with:

```text
beam width ≈ 3
```

Do not impose the same final feature count across Meta heads.

Freeze ONE final feature contract for each family winner.

---

# 3. SAMPLE-WEIGHT OPTIMIZATION

Run independently after feature selection for:

```text
Magnitude winner
Under winner
Over winner
State winner
```

General:

```text
raw_weight =
    magnitude_component
    ×
    Base-rank localization
```

Search independently:

```text
shape / power

maximum authority
```

Then:

```text
clip effective weights

renormalize within the selected query
```

Primary relative range:

```text
0.5–4
```

Only test up to ~8 as an aggressive sensitivity if clearly justified.

---

## 3A. Magnitude weights

```text
magnitude component:
    abs(Base residual)
    or bounded winning target magnitude

power:
    0.5–1.0

Base localization:
    emphasize Top5–30%
```

---

## 3B. Under weights

```text
magnitude component:
    positive residual excess above winning Under threshold

power:
    0.5–1.0

Base localization:
    strongest approximately 10–30%
```

Do not monotonically maximize weight at Base Top1.

---

## 3C. Over weights

```text
magnitude component:
    abs(negative residual)
    and/or adverse policy-loss magnitude

power:
    0.5–1.0

Base localization:
    extremely strong Top0–2%
    progressively lower to Top15%
```

---

## 3D. State weights

```text
magnitude:
    abs(Base residual)

power:
    0.5–1.0

Base localization:
    Top0–30%

plus:
    fold-local balancing of
    Over / Accurate / Under
```

Freeze one sample-weight contract per family.

---

# 4. MODEL-FAMILY CHOICE

Now each target family has frozen:

```text
target
query
feature contract
sample weights
OOF folds
evaluation metric
```

Only now compare model families.

Models:

```text
LightGBM rank-XENDCG

CatBoost QueryRMSE

CatBoost YetiRank
```

Use a matched short HPO:

```text
~30–50 trials per model family
aggressive pruning
early stopping 30
same OOF folds
same external MetaDecisionStable
```

Do not compare default models.

---

## Model priors

Magnitude:

```text
LGBM XENDCG:
    strong

CatBoost QueryRMSE:
    primary challenger

CatBoost YetiRank:
    secondary after ordinalization
```

Under:

```text
LGBM XENDCG:
    strong

CatBoost QueryRMSE:
    secondary continuous-margin challenger

CatBoost YetiRank:
    primary challenger
```

Over:

```text
LGBM XENDCG:
    strong

CatBoost QueryRMSE:
    secondary continuous-margin challenger

CatBoost YetiRank:
    primary challenger
```

Signed State:

```text
LGBM XENDCG:
    strong

CatBoost QueryRMSE:
    strong for continuous signed residual

CatBoost YetiRank:
    strong for ordinal signed state
```

Freeze one winning model family per Meta target family.

---

# 5. FULL MODEL-SPECIFIC HPO

Only after model-family selection run the expensive:

```text
150-trial
aggressive MedianPruner
early-stopping-30
```

HPO for the winning model.

---

## 5A. LightGBM rank-XENDCG

```text
max_depth:
    3–6

num_leaves:
    7–31
    constrained by depth

learning_rate:
    0.02–0.10 log

min_data_in_leaf:
    ~0.5–5% effective training rows

feature_fraction:
    0.55–1.00

bagging_fraction:
    0.65–1.00

bagging_freq:
    1 if bagging_fraction <1

lambda_l1:
    1e-5–10 log

lambda_l2:
    1e-3–100 log

min_gain_to_split:
    0–0.10

max_bin:
    63 / 127

iterations:
    ceiling ~2000

early_stopping:
    30

seed:
    fixed
```

---

## 5B. CatBoost QueryRMSE

```text
depth:
    4–8
    prior 4–6

learning_rate:
    0.02–0.12 log

iterations:
    ceiling ~1500

l2_leaf_reg:
    0.1–100 log

random_strength:
    0 / .25 / .5 / 1 / 2

rsm:
    .60–1.00

border_count:
    64 / 128 / 254

bootstrap_type:
    Bayesian / Bernoulli
```

Bayesian:

```text
bagging_temperature:
    0–2
```

Bernoulli:

```text
subsample:
    .65–.95
```

---

## 5C. CatBoost YetiRank

Test:

```text
Classic
NDCG
```

Classic:

```text
permutations:
    5 / 10 / 20

decay:
    .70 / .85 / .95
```

NDCG:

```text
permutations:
    5 / 10 / 20

top:
    align to useful head region
```

Approximate authority:

```text
Under:
    Top5–30%

Over:
    Top5–15%
```

Tree HPO:

```text
depth:
    4–8

learning_rate:
    .02–.12

iterations:
    ~1500 ceiling

l2_leaf_reg:
    .1–100 log

random_strength:
    0 / .25 / .5 / 1 / 2

rsm:
    .60–1.00

border_count:
    64 / 128 / 254

bootstrap_type:
    Bayesian / Bernoulli
```

with corresponding:

```text
bagging_temperature 0–2
```

or:

```text
subsample .65–.95
```

---

# 6. COMMON MODEL / HPO OBJECTIVE

At every stage use the same external decision-oriented score.

Assume research authority:

```text
66% Base rank
33% Meta rank
```

unless another authority is frozen.

For Top2 timestamp selection define:

```text
CHALLENGER_ONLY

CONTROL_ONLY
```

Then:

```text
ΔSwapUtility =
    Utility(CHALLENGER_ONLY)
    -
    Utility(CONTROL_ONLY)
```

---

## Under objective

Reward:

```text
positive ΔSwapUtility

recovered +50 utility

recovered +100 utility

Top2 EV improvement
```

Penalize:

```text
false promotions

severe-loss increase
```

---

## Over objective

Reward:

```text
economic value of avoided bad trades

severe-loss reduction

Top1 / Top2 substitution improvement
```

Penalize:

```text
good trades incorrectly demoted

participation collapse
```

---

## Magnitude objective

Reward:

```text
ΔSwapUtility in both directions

Top2 EV

economic ordering improvement conditional on Base
```

---

## State objective

Reward:

```text
ΔSwapUtility in both directions

balanced promotion + demotion improvement

signed-error discrimination

cross-era portability
```

---

# 7. STABILITY / PORTABILITY

Build:

```text
MetaDecisionStable
```

from the decision score by:

```text
month
OOF fold
broader era
```

Report:

```text
mean
median
Q25
Q10
positive-era fraction
worst-era guard
```

Prefer:

```text
positive median-era contribution

majority neutral/positive eras

Q25 not materially negative

no one era producing >30–35% of uplift
```

CMI portability remains supporting evidence only.

---

# 8. FINAL FAMILY WINNERS

At the end there are only:

```text
ONE optimized Magnitude head

ONE optimized Under head

ONE optimized Over head

ONE optimized State head
```

Each has independently passed:

```text
target × query selection
→ feature selection
→ sample-weight optimization
→ model choice
→ full model HPO
```

Only these four heads advance to the expensive native MC1 comparison.

---

# 9. FINAL MC1 / PORTFOLIO ABLATION

Run:

```text
Base only

Base + retained Under F120 control

Base + optimized Magnitude

Base + optimized Under

Base + optimized Over

Base + optimized State
```

Then only if justified:

```text
Base + retained head + best complementary new head
```

For every arm:

```text
retrain native Current MC1

retrain BCF MC1 only if its input contract changes

fixed dual >= +50

BCF remains auction priority

same chronological constrained portfolio
```

Final metrics:

```text
admissions
portfolio entries
EV/trade
total net EV
Top1 EV
Top2 EV
worst week
worst month
max DD
severe losses
substitution economics
```

The final promotion criterion is not CMI or Meta standalone ranking.

It is:

> Does the optimized Meta head add stable incremental economic information that MC1 can convert into better decisions?
:::

The key distinction is therefore: **target × query selection is the cheap family-level architecture screen**. Once, for example, `M_sqrtATR × Base-band/28d` wins Magnitude, *only that configuration* gets a bespoke Magnitude feature contract, sample-weight HPO, model-family comparison, and full model HPO.

---

If you want, I can:

- Explain why target × query selection precedes feature selection
- Describe the role of sample-weight optimization in the pipeline
- Outline criteria used for final model-family choice and hyperparameter optimization
